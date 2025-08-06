use crate::calibrate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::calibrate::data::TrainingData;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::{Constraint, ModelConfig};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::{Eigh, SVD, UPLO, error::LinalgError};
use std::collections::HashMap;
use std::ops::Range;

/// Computes the Kronecker product A ⊗ B for penalty matrix construction.
/// This is used to create tensor product penalties that enforce smoothness
/// in multiple dimensions for interaction terms.
fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    let mut result = Array2::zeros((a_rows * b_rows, a_cols * b_cols));
    
    for i in 0..a_rows {
        for j in 0..a_cols {
            let a_val = a[[i, j]];
            for p in 0..b_rows {
                for q in 0..b_cols {
                    result[[i * b_rows + p, j * b_cols + q]] = a_val * b[[p, q]];
                }
            }
        }
    }
    
    result
}

/// Computes the row-wise tensor product (Khatri-Rao product) of two matrices.
/// This creates the design matrix columns for tensor product interactions.
/// Each row of the result is the outer product of the corresponding rows from A and B.
fn row_wise_tensor_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let n_samples = a.nrows();
    assert_eq!(n_samples, b.nrows(), "Matrices must have same number of rows");
    
    let a_cols = a.ncols();
    let b_cols = b.ncols();
    let mut result = Array2::zeros((n_samples, a_cols * b_cols));
    
    for row in 0..n_samples {
        let mut col_idx = 0;
        for i in 0..a_cols {
            for j in 0..b_cols {
                result[[row, col_idx]] = a[[row, i]] * b[[row, j]];
                col_idx += 1;
            }
        }
    }
    
    result
}

/// Holds the layout of the design matrix `X` and penalty matrices `S_i`.
#[derive(Clone)]
pub struct ModelLayout {
    pub intercept_col: usize,
    pub pgs_main_cols: Range<usize>,
    pub penalty_map: Vec<PenalizedBlock>,
    pub total_coeffs: usize,
    pub num_penalties: usize,
    /// Number of PGS basis functions (excluding intercept) for interaction terms
    /// This is the empirical value from the actual generated basis matrix
    pub num_pgs_interaction_bases: usize,
}

/// Information about a single penalized block of coefficients.
/// For tensor product interactions, a single block may have multiple penalties.
#[derive(Clone, Debug)]
pub struct PenalizedBlock {
    pub term_name: String,
    pub col_range: Range<usize>,
    /// Indices of penalty matrices associated with this block.
    /// For main effects: single element vector.
    /// For tensor product interactions: multiple elements (one per dimension).
    pub penalty_indices: Vec<usize>,
}

impl ModelLayout {
    /// Creates a new layout based on the model configuration and basis dimensions.
    /// Enforces strict dimensional consistency across the entire GAM system.
    pub fn new(
        config: &ModelConfig,
        pc_constrained_basis_ncols: &[usize],
        pgs_main_basis_ncols: usize,
        num_pgs_interaction_bases: usize,
    ) -> Result<Self, EstimationError> {
        let mut penalty_map = Vec::new();
        let mut current_col = 0;
        let mut penalty_idx_counter = 0;

        // Calculate total coefficients first to ensure consistency
        // Formula: total_coeffs = 1 (intercept) + p_pgs_main + p_pc_main + p_interactions
        let p_pgs_main = pgs_main_basis_ncols;
        let p_pc_main: usize = pc_constrained_basis_ncols.iter().sum();
        // For tensor product interactions: each PC gets num_pgs_bases * num_pc_bases coefficients
        let p_interactions: usize = pc_constrained_basis_ncols.iter()
            .map(|&num_pc_basis| num_pgs_interaction_bases * num_pc_basis)
            .sum();
        let calculated_total_coeffs = 1 + p_pgs_main + p_pc_main + p_interactions;

        let intercept_col = current_col;
        current_col += 1;

        // Main effect for each PC
        for (i, &num_basis) in pc_constrained_basis_ncols.iter().enumerate() {
            let range = current_col..current_col + num_basis;
            penalty_map.push(PenalizedBlock {
                term_name: format!("f({})", config.pc_names[i]),
                col_range: range.clone(),
                penalty_indices: vec![penalty_idx_counter],
            });
            current_col += num_basis;
            penalty_idx_counter += 1;
        }

        // Main effect for PGS (non-constant basis terms)
        // The PGS main effect is unpenalized intentionally.
        let pgs_main_cols = current_col..current_col + pgs_main_basis_ncols;
        current_col += pgs_main_basis_ncols; // Still advance the column counter

        // Tensor product interaction effects (conditionally)
        // Each PC gets one tensor product interaction block with dual penalties
        if num_pgs_interaction_bases > 0 {
            for (i, &num_pc_basis) in pc_constrained_basis_ncols.iter().enumerate() {
                let num_tensor_coeffs = num_pgs_interaction_bases * num_pc_basis;
                let range = current_col..current_col + num_tensor_coeffs;
                
                // Create tensor product block with two penalty indices
                // One for PGS direction smoothness, one for PC direction smoothness
                let pgs_penalty_idx = penalty_idx_counter;
                let pc_penalty_idx = penalty_idx_counter + 1;
                
                penalty_map.push(PenalizedBlock {
                    term_name: format!("f(PGS,{})", config.pc_names[i]),
                    col_range: range.clone(),
                    penalty_indices: vec![pgs_penalty_idx, pc_penalty_idx],
                });
                
                current_col += num_tensor_coeffs;
                penalty_idx_counter += 2; // Two penalties per tensor product interaction
            }
        }

        // Verify that our calculation matches the actual column count
        if current_col != calculated_total_coeffs {
            return Err(EstimationError::LayoutError(format!(
                "ModelLayout dimension calculation error: calculated total_coeffs={calculated_total_coeffs} but actual current_col={current_col}"
            )));
        }

        Ok(ModelLayout {
            intercept_col,
            pgs_main_cols,
            penalty_map,
            total_coeffs: current_col,
            num_penalties: penalty_idx_counter,
            num_pgs_interaction_bases,
        })
    }
}

/// Constructs the design matrix `X` and a list of individual penalty matrices `S_i`.
/// Returns the design matrix, penalty matrices, model layout, constraint transformations, knot vectors, and PGS basis means.
pub fn build_design_and_penalty_matrices(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<
    (
        Array2<f64>,
        Vec<Array2<f64>>,
        ModelLayout,
        HashMap<String, Constraint>,
        HashMap<String, Array1<f64>>,
    ),
    EstimationError,
> {
    let n_samples = data.y.len();

    // Initialize constraint and knot vector storage
    let mut constraints = HashMap::new();
    let mut knot_vectors = HashMap::new();

    // 1. Generate basis for PGS and apply sum-to-zero constraint
    let (pgs_basis_unc, pgs_knots) = create_bspline_basis(
        data.p.view(),
        Some(data.p.view()),
        config.pgs_range,
        config.pgs_basis_config.num_knots,
        config.pgs_basis_config.degree,
    )?;

    // Save PGS knot vector
    knot_vectors.insert("pgs".to_string(), pgs_knots);

    // Apply sum-to-zero constraint to PGS main effects (excluding intercept)
    let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);
    let (pgs_main_basis, pgs_z_transform) =
        basis::apply_sum_to_zero_constraint(pgs_main_basis_unc)?;

    // For interactions, use the constrained pgs_main_basis directly
    // Cannot reconstruct "full" basis due to dimensional reduction from constraints

    // Save the PGS constraint transformation
    constraints.insert(
        "pgs_main".to_string(),
        Constraint {
            z_transform: pgs_z_transform,
        },
    );

    // 2. Generate constrained bases and unscaled penalty matrices for PCs
    let mut pc_constrained_bases = Vec::new();
    let mut s_list = Vec::new();

    // Check if we have any PCs to process
    if config.pc_names.is_empty() {
        log::info!("No PCs provided; building PGS-only model.");
    }

    for i in 0..config.pc_names.len() {
        let pc_col = data.pcs.column(i);
        let pc_name = &config.pc_names[i];
        let (pc_basis_unc, pc_knots) = create_bspline_basis(
            pc_col.view(),
            Some(pc_col.view()),
            config.pc_ranges[i],
            config.pc_basis_configs[i].num_knots,
            config.pc_basis_configs[i].degree,
        )?;

        // Save PC knot vector
        knot_vectors.insert(pc_name.clone(), pc_knots);
        // Apply sum-to-zero constraint to PC main effects (excluding intercept)
        let pc_main_basis_unc = pc_basis_unc.slice(s![.., 1..]);
        let (constrained_basis, z_transform) =
            basis::apply_sum_to_zero_constraint(pc_main_basis_unc)?;
        pc_constrained_bases.push(constrained_basis);

        // Save the PC constraint transformation
        let pc_name = &config.pc_names[i];
        constraints.insert(
            pc_name.clone(),
            Constraint {
                z_transform: z_transform.clone(),
            },
        );

        // Transform the penalty matrix: S_constrained = Z^T * S_unconstrained * Z
        let s_unconstrained =
            create_difference_penalty_matrix(pc_main_basis_unc.ncols(), config.penalty_order)?;
        let s_constrained = z_transform.t().dot(&s_unconstrained.dot(&z_transform));

        // Embed into full-sized p × p matrix (will be determined after layout is created)
        s_list.push(s_constrained);
    }

    // 3. Create tensor product penalties for interaction effects (conditionally)
    // Each PC interaction gets two penalties: one for PGS direction, one for PC direction
    if config.num_pgs_interaction_bases > 0 {
        // Create base penalty matrices for Kronecker products
        let s_pgs_base = create_difference_penalty_matrix(config.num_pgs_interaction_bases, config.penalty_order)?;
        
        for i in 0..pc_constrained_bases.len() {
            let num_pc_bases = pc_constrained_bases[i].ncols();
            let s_pc_base = create_difference_penalty_matrix(num_pc_bases, config.penalty_order)?;
            
            // Create identity matrices for Kronecker products
            let i_pgs = Array2::eye(config.num_pgs_interaction_bases);
            let i_pc = Array2::eye(num_pc_bases);
            
            // Create the two tensor product penalties:
            // 1. S_pgs ⊗ I_pc (penalizes roughness in PGS direction)
            let s_pgs_tensor = kronecker_product(&s_pgs_base, &i_pc);
            s_list.push(s_pgs_tensor);
            
            // 2. I_pgs ⊗ S_pc (penalizes roughness in PC direction) 
            let s_pc_tensor = kronecker_product(&i_pgs, &s_pc_base);
            s_list.push(s_pc_tensor);
        }
    }

    // 4. Define the model layout based on final basis dimensions
    let pc_basis_ncols: Vec<usize> = pc_constrained_bases.iter().map(|b| b.ncols()).collect();
    
    // Fix: Use config setting instead of actual basis count for layout
    let num_pgs_interaction_bases_for_layout = if config.num_pgs_interaction_bases > 0 {
        config.num_pgs_interaction_bases
    } else {
        0
    };
    
    let layout = ModelLayout::new(
        config,
        &pc_basis_ncols,
        pgs_main_basis.ncols(),
        num_pgs_interaction_bases_for_layout,
    )?;

    if s_list.len() != layout.num_penalties {
        return Err(EstimationError::LayoutError(format!(
            "Internal logic error: Mismatch in number of penalties. Layout expects {}, but {} were generated.",
            layout.num_penalties,
            s_list.len()
        )));
    }

    // Embed all penalty matrices into full-sized p × p matrices
    // This ensures all matrices in s_list are compatible for linear algebra operations
    let p = layout.total_coeffs;
    let mut s_list_full = Vec::new();

    // Validate that s_list length matches the total number of penalty indices in penalty_map
    let expected_penalty_count: usize = layout.penalty_map.iter()
        .map(|block| block.penalty_indices.len())
        .sum();
    if s_list.len() != expected_penalty_count {
        return Err(EstimationError::LayoutError(format!(
            "Penalty matrix count mismatch: s_list has {} matrices but expected {} based on penalty_map",
            s_list.len(),
            expected_penalty_count
        )));
    }

    for (k, s_k) in s_list.iter().enumerate() {
        let mut s_k_full = Array2::zeros((p, p));

        // Find the block for this penalty index and embed the matrix
        let mut found_block = false;
        for block in &layout.penalty_map {
            if block.penalty_indices.contains(&k) {
                let col_range = block.col_range.clone();
                let block_size = col_range.len();

                // Validate dimensions
                if s_k.nrows() != block_size || s_k.ncols() != block_size {
                    return Err(EstimationError::LayoutError(format!(
                        "Penalty matrix {} (term {}) has size {}×{} but block expects {}×{}",
                        k,
                        block.term_name,
                        s_k.nrows(),
                        s_k.ncols(),
                        block_size,
                        block_size
                    )));
                }

                // Embed the block-sized penalty into the full-sized matrix
                s_k_full
                    .slice_mut(s![col_range.clone(), col_range])
                    .assign(s_k);
                found_block = true;
                break;
            }
        }

        if !found_block {
            return Err(EstimationError::LayoutError(format!(
                "No block found for penalty matrix index {k}"
            )));
        }

        s_list_full.push(s_k_full);
    }

    // 5. Assemble the full design matrix `X` using the layout as the guide
    // Following a strict canonical order to match the coefficient flattening logic in model.rs
    let mut x_matrix = Array2::zeros((n_samples, layout.total_coeffs));

    // 1. Intercept - always the first column
    x_matrix.column_mut(layout.intercept_col).fill(1.0);

    // 2. Main PC effects - iterate through PC bases in order of config.pc_names
    for (pc_idx, pc_name) in config.pc_names.iter().enumerate() {
        for block in &layout.penalty_map {
            if block.term_name == format!("f({pc_name})") {
                let col_range = block.col_range.clone();
                let pc_basis = &pc_constrained_bases[pc_idx];

                // Validate dimensions before assignment
                if pc_basis.nrows() != n_samples {
                    return Err(EstimationError::LayoutError(format!(
                        "PC basis {} has {} rows but expected {} samples",
                        pc_name,
                        pc_basis.nrows(),
                        n_samples
                    )));
                }
                if pc_basis.ncols() != col_range.len() {
                    return Err(EstimationError::LayoutError(format!(
                        "PC basis {} has {} columns but layout expects {} columns",
                        pc_name,
                        pc_basis.ncols(),
                        col_range.len()
                    )));
                }

                x_matrix.slice_mut(s![.., col_range]).assign(pc_basis);
                break;
            }
        }
    }

    // 3. Main PGS effect - directly use the layout range
    let pgs_range = layout.pgs_main_cols.clone();

    // Validate dimensions before assignment
    if pgs_main_basis.nrows() != n_samples {
        return Err(EstimationError::LayoutError(format!(
            "PGS main basis has {} rows but expected {} samples",
            pgs_main_basis.nrows(),
            n_samples
        )));
    }
    if pgs_main_basis.ncols() != pgs_range.len() {
        return Err(EstimationError::LayoutError(format!(
            "PGS main basis has {} columns but layout expects {} columns",
            pgs_main_basis.ncols(),
            pgs_range.len()
        )));
    }

    x_matrix
        .slice_mut(s![.., pgs_range])
        .assign(&pgs_main_basis);

    // 4. Tensor product interaction effects (conditionally)
    // Create one unified interaction surface per PC using proper tensor products
    if config.num_pgs_interaction_bases > 0 {
        for (pc_idx, pc_name) in config.pc_names.iter().enumerate() {
            // Find the corresponding tensor product block in the layout
            let tensor_block = layout.penalty_map.iter()
                .find(|block| block.term_name == format!("f(PGS,{})", pc_name))
                .ok_or_else(|| EstimationError::LayoutError(
                    format!("Could not find tensor product block for f(PGS,{})", pc_name)
                ))?;

            // Create the tensor product design matrix columns using row-wise tensor product
            // This replaces the flawed "dimple-maker" approach with proper 2D basis functions
            let pc_constrained_basis = &pc_constrained_bases[pc_idx];
            let tensor_interaction = row_wise_tensor_product(&pgs_main_basis_unc.to_owned(), pc_constrained_basis);

            // Validate dimensions
            let col_range = tensor_block.col_range.clone();
            if tensor_interaction.nrows() != n_samples {
                return Err(EstimationError::LayoutError(format!(
                    "Tensor interaction f(PGS,{}) has {} rows but expected {} samples",
                    pc_name,
                    tensor_interaction.nrows(),
                    n_samples
                )));
            }
            if tensor_interaction.ncols() != col_range.len() {
                return Err(EstimationError::LayoutError(format!(
                    "Tensor interaction f(PGS,{}) has {} columns but layout expects {} columns",
                    pc_name,
                    tensor_interaction.ncols(),
                    col_range.len()
                )));
            }

            // No additional constraint transformation needed for the tensor product
            let z_tensor = Array2::<f64>::eye(tensor_interaction.ncols());
            let key = format!("TENSOR_PGS_{}", pc_name);
            constraints.insert(key, Constraint { z_transform: z_tensor });

            // Assign to the design matrix
            x_matrix
                .slice_mut(s![.., col_range])
                .assign(&tensor_interaction);
        }
    }

    // Simple check for obvious over-parameterization
    let n_samples = data.y.len();
    let n_coeffs = x_matrix.ncols();
    if n_coeffs > n_samples {
        log::warn!("Model is over-parameterized: {n_coeffs} coefficients for {n_samples} samples");

        // Calculate the breakdown for the new, informative error message
        let pgs_main_coeffs = layout.pgs_main_cols.len();
        let mut pc_main_coeffs = 0;
        let mut interaction_coeffs = 0;
        for block in &layout.penalty_map {
            if block.term_name.starts_with("f(PC") {
                pc_main_coeffs += block.col_range.len();
            } else if block.term_name.starts_with("f(PGS_B") {
                interaction_coeffs += block.col_range.len();
            }
        }

        return Err(EstimationError::ModelOverparameterized {
            num_coeffs: n_coeffs,
            num_samples: n_samples,
            intercept_coeffs: 1,
            pgs_main_coeffs,
            pc_main_coeffs,
            interaction_coeffs,
        });
    }

    Ok((
        x_matrix,
        s_list_full,
        layout,
        constraints,
        knot_vectors,
    ))
}

/// Result of the stable reparameterization algorithm from Wood (2011) Appendix B
#[derive(Clone)]
pub struct ReparamResult {
    /// Transformed penalty matrix S
    pub s_transformed: Array2<f64>,
    /// Log-determinant of the penalty matrix (stable computation)
    pub log_det: f64,
    /// First derivatives of log-determinant w.r.t. log-smoothing parameters
    pub det1: Array1<f64>,
    /// Orthogonal transformation matrix Qs
    pub qs: Array2<f64>,
    /// Transformed penalty square roots rS (each is p x rank_k)
    pub rs_transformed: Vec<Array2<f64>>,
    /// Lambda-dependent penalty square root from s_transformed (rank x p matrix)
    /// This is used for applying the actual penalty in the least squares solve
    pub e_transformed: Array2<f64>,
}

use indicatif::{ProgressBar, ProgressStyle};

/// Creates a lambda-independent balanced penalty root for stable rank detection
/// This follows mgcv's approach: scale each penalty to unit Frobenius norm, sum them,
/// and take the matrix square root. This balanced penalty is used ONLY for rank detection.
pub fn create_balanced_penalty_root(
    s_list: &[Array2<f64>],
) -> Result<Array2<f64>, EstimationError> {
    if s_list.is_empty() {
        // No penalties case - return empty matrix
        return Ok(Array2::zeros((0, s_list.get(0).map_or(0, |s| s.nrows()))));
    }

    let p = s_list[0].nrows();
    let mut s_balanced = Array2::zeros((p, p));

    // Scale each penalty to have unit Frobenius norm and sum them
    for s_k in s_list {
        let frob_norm = s_k.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if frob_norm > 1e-12 {
            // Scale to unit Frobenius norm and add to balanced sum
            s_balanced.scaled_add(1.0 / frob_norm, s_k);
        }
    }

    // Take the matrix square root of the balanced penalty
    let (eigenvalues, eigenvectors) = s_balanced
        .eigh(ndarray_linalg::UPLO::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;

    let tolerance = 1e-12;
    let penalty_rank = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

    if penalty_rank == 0 {
        return Ok(Array2::zeros((0, p)));
    }

    // Construct the balanced penalty square root
    let mut eb = Array2::zeros((p, penalty_rank));
    let mut col_idx = 0;
    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let sqrt_eigenval = eigenval.sqrt();
            let eigenvec = eigenvectors.column(i);
            eb.column_mut(col_idx).assign(&(&eigenvec * sqrt_eigenval));
            col_idx += 1;
        }
    }

    // Return as rank x p matrix (matching mgcv's convention)
    Ok(eb.t().to_owned())
}

/// Computes penalty square roots from full penalty matrices using eigendecomposition
/// Returns "skinny" matrices of dimension p x rank_k where rank_k is the rank of each penalty
pub fn compute_penalty_square_roots(
    s_list: &[Array2<f64>],
) -> Result<Vec<Array2<f64>>, EstimationError> {
    let mut rs_list = Vec::with_capacity(s_list.len());

    for s in s_list {
        let p = s.nrows();

        // Use eigendecomposition for symmetric positive semi-definite matrices
        let (eigenvalues, eigenvectors) = s
            .eigh(UPLO::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Count positive eigenvalues to determine rank
        let tolerance = 1e-12;
        let rank_k: usize = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

        if rank_k == 0 {
            // Zero penalty matrix - return p x 0 matrix
            rs_list.push(Array2::zeros((p, 0)));
            continue;
        }

        // Create skinny square root matrix: rS = V * diag(sqrt(λ))
        // Only include eigenvectors corresponding to positive eigenvalues
        let mut rs = Array2::zeros((p, rank_k));
        let mut col_idx = 0;

        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            if eigenval > tolerance {
                let sqrt_eigenval = eigenval.sqrt();
                let eigenvec = eigenvectors.column(i);
                // Each column of rs is sqrt(eigenvalue) * eigenvector
                rs.column_mut(col_idx).assign(&(&eigenvec * sqrt_eigenval));
                col_idx += 1;
            }
        }

        rs_list.push(rs);
    }

    Ok(rs_list)
}

/// Helper to construct the summed, weighted penalty matrix S_lambda.
/// This version works with full-sized p × p penalty matrices from s_list.
pub fn construct_s_lambda(
    lambdas: &Array1<f64>,
    s_list: &[Array2<f64>], // Now receives a list of p × p matrices
    layout: &ModelLayout,
) -> Array2<f64> {
    let p = layout.total_coeffs;
    let mut s_lambda = Array2::zeros((p, p));

    let total = s_list.len();
    if total == 0 {
        return s_lambda;
    }

    // Create a progress bar
    let pb = ProgressBar::new(total as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "    [Construction] [{bar:20.cyan/blue}] {pos}/{len} penalties (λ_{pos.minus(1)}={msg})"
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  ")
    );

    // Simple weighted sum since all matrices are now p × p
    for (i, s_k) in s_list.iter().enumerate() {
        // Format lambda value in scientific notation
        let lambda_formatted = format!("{:.2e}", lambdas[i]);
        pb.set_message(lambda_formatted);

        // Add weighted penalty matrix
        s_lambda.scaled_add(lambdas[i], s_k);

        // Update progress
        pb.inc(1);
    }

    // Finish progress bar
    pb.finish_and_clear();

    s_lambda
}

/// Implements the exact stable reparameterization algorithm from Wood (2011) Appendix B
/// This follows the complete recursive similarity transformation procedure
/// Now accepts penalty square roots (rS) instead of full penalty matrices
/// Each rs_list[i] is a p x rank_i matrix where rank_i is the rank of penalty i
///
/// The eb parameter is the pre-computed lambda-INDEPENDENT balanced penalty root
/// for rank detection, computed once at a higher level to ensure stability.
pub fn stable_reparameterization(
    rs_list: &[Array2<f64>], // penalty square roots (each is p x rank_i)
    lambdas: &[f64],
    layout: &ModelLayout,
) -> Result<ReparamResult, EstimationError> {
    // println!("DEBUG: lambdas: {:?}", lambdas);
    // println!("DEBUG: rs_list: {:?}", rs_list);
    let p = layout.total_coeffs;
    let m = rs_list.len(); // Number of penalty square roots

    if m == 0 {
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(0),
            qs: Array2::eye(p),
            rs_transformed: vec![],
            e_transformed: Array2::zeros((0, p)), // rank x p matrix
        });
    }

    // Wood (2011) Appendix B: get_stableS algorithm
    let eps = 1e-4; // A much more robust tolerance for partitioning
    // println!("DEBUG: eps = {}", eps);
    let r_tol = f64::EPSILON.powf(0.75); // rank tolerance

    // Initialize global transformation matrix and working matrices
    let mut qf = Array2::eye(p); // Final accumulated orthogonal transform Qf

    // Clone penalty square roots - we'll transform these in-place
    let mut rs_current = rs_list.to_vec();

    // Initialize iteration variables following get_stableS
    let mut k_offset = 0_usize; // K: number of parameters already processed  
    let mut q_current = p; // Q: size of current sub-problem
    let mut gamma: Vec<usize> = (0..m).collect(); // Active penalty indices
    let mut iteration = 0; // Track iterations for the termination logic

    // Main similarity transform loop - mirrors get_stableS structure
    loop {
        // Increment iteration counter
        iteration += 1;

        println!(
            "[Reparam Iteration #{}] Starting. Active penalties: {}, Problem size: {}",
            iteration,
            gamma.len(),
            q_current
        );

        if gamma.is_empty() || q_current == 0 {
            break;
        }

        log::debug!(
            "[Reparam Iteration #{}] Starting. Active penalties: {}, Problem size: {}",
            iteration,
            gamma.len(),
            q_current
        );

        log::debug!(
            "Iteration {}: k_offset={}, q_current={}, gamma={:?}",
            iteration,
            k_offset,
            q_current,
            gamma
        );

        // Step 1: Find Frobenius norms of penalties in current sub-problem
        // For penalty square roots, we need to form the full penalty matrix S_i = rS_i^T * rS_i
        let mut frob_norms = Vec::new();
        let mut max_omega: f64 = 0.0;

        for &i in &gamma {
            // Extract active rows from penalty square root (columns stay the same)
            let rs_active_rows = rs_current[i].slice(s![k_offset..k_offset + q_current, ..]);

            // Skip if penalty has no columns (zero penalty)
            if rs_current[i].ncols() == 0 {
                frob_norms.push((i, 0.0));
                continue;
            }

            // Form the active sub-block of full penalty matrix S_i = rS_i * rS_i^T
            let s_active_block = rs_active_rows.dot(&rs_active_rows.t());

            // The Frobenius norm is the sqrt of sum of squares of matrix elements
            let frob_norm = s_active_block.iter().map(|&x| x * x).sum::<f64>().sqrt();
            // Scale by lambda to get the weighted norm (omega_i)
            let omega_i = frob_norm * lambdas[i];

            // No artificial perturbation - mgcv handles zero penalties exactly
            frob_norms.push((i, omega_i));
            max_omega = max_omega.max(omega_i);
            // println!("DEBUG: Penalty {} has omega_i = {}", i, omega_i);
        }

        if max_omega < 1e-15 {
            break; // All remaining penalties are numerically zero
        }

        // Step 2: Partition into dominant α and subdominant γ' sets
        // This is the most critical part of the algorithm
        // We must ensure this logic exactly matches mgcv's get_stableS function
        let threshold = eps * max_omega;
        // println!("DEBUG: max_omega = {}, threshold = {}", max_omega, threshold);

        // Initialize alpha and gamma_prime sets as empty
        let mut alpha = Vec::new();
        let mut gamma_prime = Vec::new();

        // For each term in gamma, decide whether it goes in alpha or gamma_prime
        // based on its weighted Frobenius norm (omega)
        for &i in &gamma {
            // Find the omega value for this index
            if let Some(&(_, omega)) = frob_norms.iter().find(|&&(idx, _)| idx == i) {
                if omega >= threshold {
                    // This penalty has significant influence - put in alpha (dominant)
                    alpha.push(i);
                } else {
                    // This penalty has minor influence - put in gamma_prime (subdominant)
                    gamma_prime.push(i);
                }
            }
        }

        // Now alpha contains indices of penalties with ω_i ≥ threshold
        // gamma_prime contains indices of penalties with ω_i < threshold

        // Alpha and gamma_prime are already index lists
        // No need for conversion - they contain the actual indices from gamma

        if alpha.is_empty() {
            log::debug!("No terms in alpha set. Terminating.");
            break;
        }

        log::debug!(
            "Partitioned: alpha set = {:?}, gamma_prime set = {:?}",
            alpha,
            gamma_prime
        );

        // println!("DEBUG: Partitioned: alpha set = {:?}, gamma_prime set = {:?}", alpha, gamma_prime);

        // Step 3: Form weighted sum of ONLY dominant penalties and eigendecompose
        // This is the key difference from a naive approach - we only use alpha penalties
        let mut sb = Array2::zeros((q_current, q_current));
        for &i in &alpha {
            let rs_active_rows = rs_current[i].slice(s![k_offset..k_offset + q_current, ..]);
            let s_active_block = rs_active_rows.dot(&rs_active_rows.t());

            // Use the penalty matrix directly without artificial perturbation
            // mgcv handles zero penalties exactly in the null-space
            sb.scaled_add(lambdas[i], &s_active_block);
        }

        // println!("DEBUG: Final sb matrix: {:?}", sb);

        // Eigendecomposition to get rank and eigenvectors
        let (eigenvalues, u): (Array1<f64>, Array2<f64>) = sb
            .eigh(UPLO::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // println!("DEBUG: Eigenvalues: {:?}", eigenvalues);
        // println!("DEBUG: Eigenvectors: {:?}", u);

        // Determine rank 'r' of the dominant sub-problem. This is done by counting
        // the number of eigenvalues that are greater than a small tolerance
        // relative to the largest eigenvalue. This correctly implements the logic
        // from mgcv's C function `get_stableS`.
        let max_eigenval = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let rank_tolerance = max_eigenval * r_tol;
        let r = eigenvalues.iter().filter(|&&ev| ev > rank_tolerance).count();

        log::debug!(
            "Rank determination: found rank {} from {} eigenvalues (largest eigenval: {}, tol: {})",
            r,
            eigenvalues.len(),
            max_eigenval,
            rank_tolerance
        );

        // Step 4: Check termination criterion
        // This is the critical fix: when r == q_current on the first iteration,
        // mgcv still performs the transformation and constructs the final outputs
        // The key is to properly update the transformation matrices first
        log::debug!(
            "Rank detection: r={}, q_current={}, iteration={}",
            r,
            q_current,
            iteration
        );

        // Step 5: Update global transformation matrix Qf
        // For ascending order, we want the last r indices (largest eigenvalues)
        // let selected_indices = &sorted_indices[q - r..];

        // Use the FULL eigenvector matrix for transformation, not truncated
        // The transformation must be done with the full q_current x q_current matrix
        // to maintain proper dimensions throughout the algorithm
        let qf_block = qf.slice(s![.., k_offset..k_offset + q_current]).to_owned();
        let qf_new = qf_block.dot(&u); // Use full eigenvector matrix u
        qf.slice_mut(s![.., k_offset..k_offset + q_current])
            .assign(&qf_new);

        // Step 6: Transform ALL active penalty roots by the full eigenvector matrix U.
        // This projects them onto the new basis defined by the eigenvectors of the dominant penalties.
        for &i in &gamma {
            if rs_current[i].ncols() == 0 { continue; }

            let c_matrix = rs_current[i].slice(s![k_offset..k_offset + q_current, ..]).to_owned();
            let b_matrix = u.t().dot(&c_matrix);
            
            // Assign the fully transformed block back into the main rs_current matrix.
            rs_current[i]
                .slice_mut(s![k_offset..k_offset + q_current, ..])
                .assign(&b_matrix);
        }

        // Step 7: Partition the newly transformed space. This is the core of the algorithm.
        // We zero out the parts of the penalty matrices that are irrelevant for subsequent,
        // smaller sub-problems.
        for &i in &gamma {
            if rs_current[i].ncols() == 0 { continue; }

            if alpha.contains(&i) {
                // If a penalty was DOMINANT, its effect is now entirely within the range space
                // (the first `r` dimensions of the new basis). Its projection onto the
                // null space (the remaining `q_current - r` dimensions) must be zeroed out.
                if r < q_current {
                    rs_current[i].slice_mut(s![k_offset + r.., ..]).fill(0.0);
                }
            } else { // This penalty was SUB-DOMINANT (in gamma_prime).
                // Its effect is now entirely within the null space. Its projection onto
                // the range space must be zeroed out.
                rs_current[i].slice_mut(s![k_offset..k_offset + r, ..]).fill(0.0);
            }
        }

        // Update for next iteration
        // Update iteration variables for next loop according to mgcv
        k_offset += r; // Increase offset by the rank we processed
        q_current -= r; // Reduce problem size by the rank we processed
        gamma = gamma_prime; // Continue with the subdominant penalties

        log::debug!(
            "Updated for next iteration: k_offset={}, q_current={}, gamma.len()={}",
            k_offset,
            q_current,
            gamma.len()
        );

        log::debug!(
            "[Reparam Iteration #{}] Finished. Determined rank: {}. Next problem size: {}",
            iteration,
            r,
            q_current
        );
    }

    log::debug!(
        "[Reparam] Loop finished after {} iterations. Proceeding to generate final outputs.",
        iteration
    );

    // AFTER LOOP: Generate final outputs from the transformed penalty roots

    // Step 1: The loop has finished - rs_current now contains the fully transformed penalty roots
    let final_rs_transformed = rs_current;

    // Step 2: Construct the final transformed total penalty matrix
    let mut s_transformed = Array2::zeros((p, p));
    for i in 0..m {
        // Form full penalty from transformed root: S_k = rS_k * rS_k^T
        let s_k_transformed = final_rs_transformed[i].dot(&final_rs_transformed[i].t());
        s_transformed.scaled_add(lambdas[i], &s_k_transformed);
    }

    // Step 3: Compute the lambda-DEPENDENT penalty square root from s_transformed
    // Use eigendecomposition: if S = V*D*V^T, then sqrt(S) = V*sqrt(D)*V^T
    let (s_eigenvalues, s_eigenvectors): (Array1<f64>, Array2<f64>) = s_transformed
        .eigh(UPLO::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;

    // Count non-zero eigenvalues to determine the rank
    let tolerance = 1e-12;
    let penalty_rank = s_eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

    // Construct the lambda-DEPENDENT penalty square root matrix
    let mut e_matrix = Array2::zeros((p, penalty_rank));
    let mut col_idx = 0;
    for (i, &eigenval) in s_eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let sqrt_eigenval = eigenval.sqrt();
            let eigenvec = s_eigenvectors.column(i);
            // Each column of the matrix is sqrt(eigenvalue) * eigenvector
            e_matrix
                .column_mut(col_idx)
                .assign(&(&eigenvec * sqrt_eigenval));
            col_idx += 1;
        }
    }

    // e_transformed: Lambda-DEPENDENT penalty root for actual penalty application
    // This represents the true penalty strength and changes with lambda values
    let e_transformed = e_matrix.t().to_owned();

    // Step 4: Calculate log-determinant from the eigenvalues we already computed
    let log_det: f64 = s_eigenvalues
        .iter()
        .filter(|&&ev| ev > tolerance)
        .map(|&ev| ev.ln())
        .sum();

    // Step 5: Calculate derivatives using the correct transformed matrices
    let mut det1 = Array1::zeros(lambdas.len());

    // Compute pseudo-inverse of transformed total penalty
    let mut s_plus = Array2::zeros((p, p));
    for (i, &eigenval) in s_eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let v_i = s_eigenvectors.column(i);
            let outer_product = v_i
                .to_owned()
                .insert_axis(Axis(1))
                .dot(&v_i.to_owned().insert_axis(Axis(0)));
            s_plus.scaled_add(1.0 / eigenval, &outer_product);
        }
    }

    // Calculate derivatives: det1[k] = λ_k * tr(S_λ^+ S_k_transformed)
    for k in 0..lambdas.len() {
        let s_k_transformed = final_rs_transformed[k].dot(&final_rs_transformed[k].t());
        let s_plus_times_s_k = s_plus.dot(&s_k_transformed);
        let trace: f64 = s_plus_times_s_k.diag().sum();
        det1[k] = lambdas[k] * trace;
    }

    Ok(ReparamResult {
        s_transformed,
        log_det,
        det1,
        qs: qf,
        rs_transformed: final_rs_transformed,
        e_transformed,
    })
}

/// Result of the stable penalized least squares solve
#[derive(Clone)]
pub struct StablePLSResult {
    /// Solution vector beta
    pub beta: Array1<f64>,
    /// Final penalized Hessian matrix
    pub penalized_hessian: Array2<f64>,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Scale parameter estimate
    pub scale: f64,
}

/// Calculate the condition number of a matrix using singular value decomposition (SVD).
///
/// The condition number is the ratio of the largest to smallest singular value.
/// A high condition number indicates the matrix is close to singular and
/// solving linear systems with it may be numerically unstable.
///
/// # Arguments
/// * `matrix` - The matrix to analyze
///
/// # Returns
/// * `Ok(condition_number)` - The condition number (max_sv / min_sv)
/// * `Ok(f64::INFINITY)` - If the matrix is effectively singular (min_sv < 1e-12)
/// * `Err` - If SVD computation fails
pub fn calculate_condition_number(matrix: &Array2<f64>) -> Result<f64, LinalgError> {
    // Use SVD for all matrices - the cost depends on number of coefficients (p), not samples (n)
    // For typical GAMs, p is much smaller than n, making SVD computationally feasible and reliable
    let (_, s, _) = matrix.svd(false, false)?;

    // Get max and min singular values
    let max_sv = s.iter().fold(0.0_f64, |max, &val| max.max(val));
    let min_sv = s.iter().fold(f64::INFINITY, |min, &val| min.min(val));

    // Check for effective singularity
    if min_sv < 1e-12 {
        return Ok(f64::INFINITY);
    }

    Ok(max_sv / min_sv)
}
