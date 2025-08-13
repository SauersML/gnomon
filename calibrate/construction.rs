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
/// NOTE: Currently unused due to penalty grouping in Option 3 implementation
#[allow(dead_code)]
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
    assert_eq!(
        n_samples,
        b.nrows(),
        "Matrices must have same number of rows"
    );

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
}

/// The semantic type of a penalized term in the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TermType {
    PcMainEffect,
    Interaction,
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
    /// The semantic type of this term.
    pub term_type: TermType,
}

impl ModelLayout {
    /// Creates a new layout based on the model configuration and basis dimensions.
    /// Enforces strict dimensional consistency across the entire GAM system.
    pub fn new(
        config: &ModelConfig,
        pc_constrained_basis_ncols: &[usize],
        pgs_main_basis_ncols: usize,
        pc_unconstrained_basis_ncols: &[usize],
        num_pgs_interaction_bases: usize,
    ) -> Result<Self, EstimationError> {
        let mut penalty_map = Vec::new();
        let mut current_col = 0;
        let mut _penalty_idx_counter = 0;

        // Calculate total coefficients first to ensure consistency
        // Formula: total_coeffs = 1 (intercept) + p_pgs_main + p_pc_main + p_interactions
        let p_pgs_main = pgs_main_basis_ncols;
        let p_pc_main: usize = pc_constrained_basis_ncols.iter().sum();
        // For tensor product interactions: each PC gets num_pgs_bases * num_pc_bases coefficients
        // Use unconstrained dimensions for interaction calculations
        let p_interactions: usize = pc_unconstrained_basis_ncols
            .iter()
            .map(|&num_pc_basis_unc| num_pgs_interaction_bases * num_pc_basis_unc)
            .sum();
        let calculated_total_coeffs = 1 + p_pgs_main + p_pc_main + p_interactions;

        let intercept_col = current_col;
        current_col += 1;

        // Main effect for each PC (each gets its own unique penalty index)
        for (i, &num_basis) in pc_constrained_basis_ncols.iter().enumerate() {
            let range = current_col..current_col + num_basis;
            penalty_map.push(PenalizedBlock {
                term_name: format!("f({})", config.pc_names[i]),
                col_range: range.clone(),
                penalty_indices: vec![_penalty_idx_counter], // Each PC main gets unique penalty index
                term_type: TermType::PcMainEffect,
            });
            current_col += num_basis;
            _penalty_idx_counter += 1; // Increment for next penalty
        }

        // Main effect for PGS (non-constant basis terms)
        // The PGS main effect is unpenalized intentionally.
        let pgs_main_cols = current_col..current_col + pgs_main_basis_ncols;
        current_col += pgs_main_basis_ncols; // Still advance the column counter

        // Tensor product interaction effects (each gets two penalty indices: PGS-dir and PC-dir)
        if num_pgs_interaction_bases > 0 {
            for (i, &num_pc_basis_unc) in pc_unconstrained_basis_ncols.iter().enumerate() {
                let num_tensor_coeffs = num_pgs_interaction_bases * num_pc_basis_unc;
                let range = current_col..current_col + num_tensor_coeffs;

                penalty_map.push(PenalizedBlock {
                    term_name: format!("f(PGS,{})", config.pc_names[i]),
                    col_range: range.clone(),
                    penalty_indices: vec![_penalty_idx_counter, _penalty_idx_counter + 1], // Two penalties: PGS-dir, PC-dir
                    term_type: TermType::Interaction,
                });

                current_col += num_tensor_coeffs;
                _penalty_idx_counter += 2; // Increment by 2 for tensor-product penalty
            }
        }

        // Total number of individual penalties: one per PC main + two per interaction (PGS-dir + PC-dir)
        // _penalty_idx_counter has already been incremented to the correct total

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
            num_penalties: _penalty_idx_counter,
        })
    }
}

/// Constructs the design matrix `X` and a list of individual penalty matrices `S_i`.
/// Returns the design matrix, penalty matrices, model layout, constraint transformations, knot vectors, and range transformations.
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
        HashMap<String, Array2<f64>>,
    ),
    EstimationError,
> {
    // Validate PC configuration against available data
    if config.pc_names.len() > data.pcs.ncols() {
        return Err(EstimationError::InvalidInput(format!(
            "Configuration requests {} Principal Components ({}), but the provided data file only contains {} PC columns.",
            config.pc_names.len(),
            config.pc_names.join(", "),
            data.pcs.ncols()
        )));
    }

    let n_samples = data.y.len();

    // Initialize constraint, knot vector, and range transform storage
    let mut constraints = HashMap::new();
    let mut knot_vectors = HashMap::new();
    let mut range_transforms = HashMap::new();

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

    // Create PGS penalty matrix and decompose into null/range spaces
    let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);
    let s_pgs_base =
        create_difference_penalty_matrix(pgs_main_basis_unc.ncols(), config.penalty_order)?;
    let (_z_null_pgs, z_range_pgs) = basis::null_range_whiten(&s_pgs_base, 1e-12)?;

    // For policy mode: PGS main effect remains unpenalized (use full unconstrained basis)
    // Apply sum-to-zero constraint to maintain identifiability
    let (pgs_main_basis, pgs_z_transform) =
        basis::apply_sum_to_zero_constraint(pgs_main_basis_unc)?;

    // Store PGS range transformation for interactions and potential future penalized PGS
    range_transforms.insert("pgs".to_string(), z_range_pgs);

    // Save the PGS constraint transformation
    constraints.insert(
        "pgs_main".to_string(),
        Constraint {
            z_transform: pgs_z_transform,
        },
    );

    // 2. Generate range-only bases for PCs (functional ANOVA decomposition)
    let mut pc_range_bases = Vec::new();
    let mut pc_unconstrained_bases_main = Vec::new();

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

        // Functional ANOVA decomposition: use range-only basis for PC main effects
        let pc_main_basis_unc = pc_basis_unc.slice(s![.., 1..]);
        pc_unconstrained_bases_main.push(pc_main_basis_unc.to_owned());

        // Create penalty matrix and decompose into null/range spaces
        let s_pc_base =
            create_difference_penalty_matrix(pc_main_basis_unc.ncols(), config.penalty_order)?;
        let (_z_null_pc, z_range_pc) = basis::null_range_whiten(&s_pc_base, 1e-12)?;

        // PC main effect uses ONLY the range (penalized) part
        let pc_range_basis = pc_main_basis_unc.dot(&z_range_pc);
        pc_range_bases.push(pc_range_basis);

        // Store PC range transformation for interactions and main effects
        range_transforms.insert(pc_name.clone(), z_range_pc.clone());

        // Store the actual range transformation for backward compatibility
        // This allows prediction fallback to work correctly when range_transforms is empty
        constraints.insert(
            pc_name.clone(),
            Constraint {
                z_transform: z_range_pc.clone(),
            },
        );
    }

    // 3. Calculate layout first to determine matrix dimensions
    let pc_range_ncols: Vec<usize> = pc_range_bases.iter().map(|b| b.ncols()).collect();
    let pgs_range_ncols = range_transforms
        .get("pgs")
        .map(|rt| rt.ncols())
        .unwrap_or(0);

    let layout = ModelLayout::new(
        config,
        &pc_range_ncols,
        pgs_main_basis.ncols(),
        &pc_range_ncols, // Use range ncols for interactions too
        pgs_range_ncols, // Use PGS range size for interactions
    )?;

    // 4. Create individual penalty matrices - one per PC main and two per interaction
    // Each penalty gets its own lambda parameter for optimal smoothing control
    let p = layout.total_coeffs;
    let mut s_list = Vec::with_capacity(layout.num_penalties);

    // Initialize all penalty matrices as zeros
    for _ in 0..layout.num_penalties {
        s_list.push(Array2::zeros((p, p)));
    }

    // Fill in identity penalties for each penalized block individually
    for block in &layout.penalty_map {
        let col_range = block.col_range.clone();
        let range_len = col_range.len();

        match block.term_type {
            TermType::PcMainEffect => {
                // Main effects have single penalty index
                let penalty_idx = block.penalty_indices[0];
                s_list[penalty_idx]
                    .slice_mut(s![col_range.clone(), col_range])
                    .assign(&Array2::eye(range_len));
            }
            TermType::Interaction => {
                // Interactions have two penalty indices (PGS-dir and PC-dir)
                // For now, use identity penalties - TODO: implement proper Kronecker-sum
                for &penalty_idx in &block.penalty_indices {
                    s_list[penalty_idx]
                        .slice_mut(s![col_range.clone(), col_range])
                        .assign(&Array2::eye(range_len));
                }
            }
        }
    }

    // Range transformations will be returned directly to the caller

    // 5. Assemble the full design matrix `X` using the layout as the guide
    // Following a strict canonical order to match the coefficient flattening logic in model.rs
    let mut x_matrix = Array2::zeros((n_samples, layout.total_coeffs));

    // 1. Intercept - always the first column
    x_matrix.column_mut(layout.intercept_col).fill(1.0);

    // 2. Main PC effects - use range-only bases (fully penalized)
    for (pc_idx, pc_name) in config.pc_names.iter().enumerate() {
        for block in &layout.penalty_map {
            if block.term_name == format!("f({pc_name})") {
                let col_range = block.col_range.clone();
                let pc_basis = &pc_range_bases[pc_idx];

                // Validate dimensions before assignment
                if pc_basis.nrows() != n_samples {
                    return Err(EstimationError::LayoutError(format!(
                        "PC range basis {} has {} rows but expected {} samples",
                        pc_name,
                        pc_basis.nrows(),
                        n_samples
                    )));
                }
                if pc_basis.ncols() != col_range.len() {
                    return Err(EstimationError::LayoutError(format!(
                        "PC range basis {} has {} columns but layout expects {} columns",
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

    // 4. Tensor product interaction effects - Range × Range only (fully penalized)
    if pgs_range_ncols > 0 && !pc_range_bases.is_empty() {
        // Get PGS range basis for interactions
        let pgs_range_transform = range_transforms.get("pgs").unwrap();
        let pgs_range_basis = pgs_main_basis_unc.dot(pgs_range_transform);

        for (pc_idx, pc_name) in config.pc_names.iter().enumerate() {
            // Find the corresponding tensor product block in the layout
            let tensor_block = layout
                .penalty_map
                .iter()
                .find(|block| block.term_name == format!("f(PGS,{})", pc_name))
                .ok_or_else(|| {
                    EstimationError::LayoutError(format!(
                        "Could not find tensor product block for f(PGS,{})",
                        pc_name
                    ))
                })?;

            // Create Range × Range tensor product (guarantees full penalization)
            let pc_range_transform = range_transforms.get(pc_name).unwrap();
            let pc_range_basis = pc_unconstrained_bases_main[pc_idx].dot(pc_range_transform);
            let tensor_interaction = row_wise_tensor_product(&pgs_range_basis, &pc_range_basis);

            // Validate dimensions
            let col_range = tensor_block.col_range.clone();
            if tensor_interaction.nrows() != n_samples {
                return Err(EstimationError::LayoutError(format!(
                    "Range×Range tensor interaction f(PGS,{}) has {} rows but expected {} samples",
                    pc_name,
                    tensor_interaction.nrows(),
                    n_samples
                )));
            }
            if tensor_interaction.ncols() != col_range.len() {
                return Err(EstimationError::LayoutError(format!(
                    "Range×Range tensor interaction f(PGS,{}) has {} columns but layout expects {} columns",
                    pc_name,
                    tensor_interaction.ncols(),
                    col_range.len()
                )));
            }

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
            match block.term_type {
                TermType::PcMainEffect => {
                    pc_main_coeffs += block.col_range.len();
                }
                TermType::Interaction => {
                    interaction_coeffs += block.col_range.len();
                }
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
        s_list,
        layout,
        constraints,
        knot_vectors,
        range_transforms,
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


/// Creates a lambda-independent balanced penalty root for stable rank detection
/// This follows mgcv's approach: scale each penalty to unit Frobenius norm, sum them,
/// and take the matrix square root. This balanced penalty is used ONLY for rank detection.
pub fn create_balanced_penalty_root(
    s_list: &[Array2<f64>],
    p: usize,
) -> Result<Array2<f64>, EstimationError> {
    if s_list.is_empty() {
        // No penalties case - return empty matrix with correct number of columns
        return Ok(Array2::zeros((0, p)));
    }

    // Validate penalty matrix dimensions
    for (idx, s) in s_list.iter().enumerate() {
        if s.nrows() != p || s.ncols() != p {
            return Err(EstimationError::LayoutError(format!(
                "Penalty matrix {idx} must be {p}×{p}, got {}×{}",
                s.nrows(),
                s.ncols()
            )));
        }
    }
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
/// Returns "skinny" matrices of dimension rank_k x p where rank_k is the rank of each penalty
/// STANDARDIZED: All penalty roots use rank x p convention with S = R^T * R
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
            // Zero penalty matrix - return 0 x p matrix (STANDARDIZED: rank x p)
            rs_list.push(Array2::zeros((0, p)));
            continue;
        }

        // STANDARDIZED: Create rank x p square root matrix where S = rs^T * rs
        // Each row is sqrt(eigenvalue) * eigenvector^T
        let mut rs = Array2::zeros((rank_k, p));
        let mut row_idx = 0;

        for (i, &eigenval) in eigenvalues.iter().enumerate() {
            if eigenval > tolerance {
                let sqrt_eigenval = eigenval.sqrt();
                let eigenvec = eigenvectors.column(i);
                // Each row of rs is sqrt(eigenvalue) * eigenvector^T
                rs.row_mut(row_idx).assign(&(&eigenvec * sqrt_eigenval));
                row_idx += 1;
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

    if s_list.is_empty() {
        return s_lambda;
    }

    // Simple weighted sum since all matrices are now p × p
    for (i, s_k) in s_list.iter().enumerate() {
        // Add weighted penalty matrix
        s_lambda.scaled_add(lambdas[i], s_k);
    }

    s_lambda
}

/// Implements the stable reparameterization algorithm from Wood (2011) Appendix B.
///
/// This function performs the recursive similarity transformation using
/// penalty square roots rather than full penalty matrices. Each entry in
/// `rs_list` is a `p × rank_k` matrix (skinny square root) for penalty `k`,
/// where `rank_k` is the numerical rank of that penalty. The vector `lambdas`
/// provides the smoothing parameters for each penalty, and `layout` defines
/// the model’s coefficient block structure and sizes.
///
/// Rank detection follows mgcv’s balancing idea but is rebuilt at each
/// iteration from the currently transformed sub-blocks: we form a balanced,
/// lambda-independent sum by scaling each active penalty sub-block to unit
/// Frobenius norm and summing them, then use its eigenvalues to determine the
/// numerical rank. This avoids needing an `eb` argument.
///
/// Note: A lambda-independent balanced penalty root (“eb”) can still be
/// computed at a higher level (see `create_balanced_penalty_root`) for
/// diagnostics or alternative workflows. This function does not take `eb` as a
/// parameter; it operates solely on `rs_list`, `lambdas`, and `layout`.
pub fn stable_reparameterization(
    rs_list: &[Array2<f64>], // penalty square roots (each is rank_i x p) STANDARDIZED
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

    // Create pristine copy of original full penalty matrices S_k = rS_k * rS_k^T
    // These will NEVER be modified and are used for building the sb matrix
    let s_original_list: Vec<Array2<f64>> = rs_list.iter().map(|rs_k| rs_k.t().dot(rs_k)).collect();

    // Create the WORKING copy that will be transformed
    let mut s_current_list = s_original_list.clone();

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
            // FIXED: Extract active columns from penalty square root (rank x p convention)
            let rs_active_cols = rs_current[i].slice(s![.., k_offset..k_offset + q_current]);

            // Skip if penalty has no columns (zero penalty)
            if rs_current[i].ncols() == 0 {
                frob_norms.push((i, 0.0));
                continue;
            }

            // FIXED: Form the active sub-block of full penalty matrix S_i = rS_i^T * rS_i
            let s_active_block = rs_active_cols.t().dot(&rs_active_cols);

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

        // Step 3a: Form SCALED sum for STABLE RANK DETECTION (following mgcv's get_stableS)
        // This creates a lambda-independent, balanced matrix for reliable rank detection
        let mut sb_for_rank = Array2::zeros((q_current, q_current));
        for &i in &alpha {
            let s_current_sub_block = s_current_list[i].slice(s![
                k_offset..k_offset + q_current,
                k_offset..k_offset + q_current
            ]);

            // Calculate Frobenius norm (sqrt of sum of squared elements)
            let frob_norm = s_current_sub_block
                .iter()
                .map(|&x| x * x)
                .sum::<f64>()
                .sqrt();

            // Scale by inverse norm to create a balanced matrix for rank detection
            if frob_norm > 1e-12 {
                // Avoid division by zero for zero-matrices
                sb_for_rank.scaled_add(1.0 / frob_norm, &s_current_sub_block);
            }
        }

        // Eigendecompose the balanced matrix to get stable eigenvalues for rank detection
        let (eigenvalues_for_rank, _) = sb_for_rank
            .eigh(UPLO::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Determine rank 'r' using these stable eigenvalues
        let max_eigenval = eigenvalues_for_rank
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let rank_tolerance = max_eigenval * r_tol;
        let r = eigenvalues_for_rank
            .iter()
            .filter(|&&ev| ev > rank_tolerance)
            .count();

        log::debug!(
            "Stable rank detection: found rank {} from {} eigenvalues (max_eig: {}, tol: {})",
            r,
            eigenvalues_for_rank.len(),
            max_eigenval,
            rank_tolerance
        );

        // Step 3b: Form WEIGHTED sum for TRANSFORMATION (lambda-weighted for eigenvectors)
        // This matrix provides the eigenvectors for the similarity transform
        let mut sb_for_transform = Array2::zeros((q_current, q_current));
        for &i in &alpha {
            // Use the CURRENTLY transformed matrix, not the original
            let s_current_sub_block = s_current_list[i].slice(s![
                k_offset..k_offset + q_current,
                k_offset..k_offset + q_current
            ]);

            // Use lambda weighting for transformation eigenvectors
            sb_for_transform.scaled_add(lambdas[i], &s_current_sub_block);
        }

        // Eigendecomposition to get eigenvectors 'u' for the similarity transform
        // We DISCARD the eigenvalues from this decomposition - only use eigenvectors
        let (_, u): (Array1<f64>, Array2<f64>) = sb_for_transform
            .eigh(UPLO::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Note: The stable rank detection debug message is already logged above

        log::debug!(
            "Rank detection: r={}, q_current={}, iteration={}",
            r,
            q_current,
            iteration
        );

        // ---
        // STEP 5A: REORDER THE EIGENVECTOR MATRIX `u` TO MATCH `mgcv`'s LOGIC
        // `Eigh` returns eigenvalues in ascending order, so the eigenvectors for the range
        // space (largest eigenvalues) are at the END of `u`. We reorder them to be first.
        // The new basis is `U_reordered = [U_range | U_null]`.
        // ---
        
        // Guard against r == 0 to avoid empty slicing
        if r == 0 {
            // No range directions identified this iteration; switch to gamma' and continue
            gamma = gamma_prime;
            continue;
        }
        
        let u_range = u.slice(s![.., q_current - r..]); // Last r columns
        let u_null = u.slice(s![.., ..q_current - r]); // First q_current - r columns
        let u_reordered = ndarray::concatenate(Axis(1), &[u_range, u_null])
            .expect("Failed to reorder eigenvectors");

        // Step 5B: Update global transformation matrix Qf using the REORDERED basis
        let qf_block = qf.slice(s![.., k_offset..k_offset + q_current]).to_owned();
        let qf_new = qf_block.dot(&u_reordered);
        qf.slice_mut(s![.., k_offset..k_offset + q_current])
            .assign(&qf_new);

        // Now, apply the similarity transform to all active S_k matrices for the next iteration.
        // This is the core of the recursive update.
        for &i in &gamma {
            // Extract the current sub-problem block
            let s_sub_block = s_current_list[i]
                .slice(s![
                    k_offset..k_offset + q_current,
                    k_offset..k_offset + q_current
                ])
                .to_owned();

            // Apply the similarity transform using the REORDERED basis: U_reordered^T * S_sub * U_reordered
            let transformed_sub_block = u_reordered.t().dot(&s_sub_block).dot(&u_reordered);

            // Place it back into the full-size matrix
            s_current_list[i]
                .slice_mut(s![
                    k_offset..k_offset + q_current,
                    k_offset..k_offset + q_current
                ])
                .assign(&transformed_sub_block);
        }

        // Step 6: Transform ALL active penalty roots by the REORDERED eigenvector matrix U.
        // This projects them onto the new basis defined by the eigenvectors of the dominant penalties.
        for &i in &gamma {
            if rs_current[i].ncols() == 0 {
                continue;
            }

            // FIXED: For rank×p penalty roots, transform as R_new = R * U (not U^T * R)
            let c_matrix = rs_current[i]
                .slice(s![.., k_offset..k_offset + q_current])
                .to_owned();
            let b_matrix = c_matrix.dot(&u_reordered); // rS_sub * U_reordered

            // Assign the fully transformed block back into the main rs_current matrix.
            rs_current[i]
                .slice_mut(s![.., k_offset..k_offset + q_current])
                .assign(&b_matrix);
        }

        // ---
        // Step 7: Partitioning logic
        // After transforming with `u_reordered`, the first `r` rows correspond to the range
        // space, and the last `q_current - r` rows correspond to the null space.
        // ---
        for &i in &gamma {
            if rs_current[i].ncols() == 0 {
                continue;
            }

            if alpha.contains(&i) {
                // DOMINANT penalty: Its effect is now entirely within the range space.
                // FIXED: For rank×p roots, zero out the null space COLUMNS (not rows)
                // The null space is now the LAST `q_current - r` columns of the sub-block.
                if r < q_current {
                    rs_current[i].slice_mut(s![.., k_offset + r..]).fill(0.0);
                }
            } else {
                // SUB-DOMINANT penalty (in gamma_prime).
                // Its effect is carried forward in the null space.
                // FIXED: For rank×p roots, zero out the range space COLUMNS (not rows)
                // The range space is now the FIRST `r` columns of the sub-block.
                rs_current[i]
                    .slice_mut(s![.., k_offset..k_offset + r])
                    .fill(0.0);
            }
        }

        // Apply the same zeroing to the full S matrices.
        // This prevents dominant penalty information from contaminating the next iteration's
        // basis calculation (the cause of the numerical instability).
        for &i in &gamma {
            if alpha.contains(&i) {
                // DOMINANT penalty: Zero out its null-space block.
                if r < q_current {
                    // Zero out the null-space rows and columns (bottom-right block)
                    s_current_list[i]
                        .slice_mut(s![k_offset + r.., k_offset + r..])
                        .fill(0.0);
                    // Zero out the off-diagonal blocks connecting range and null spaces
                    s_current_list[i]
                        .slice_mut(s![k_offset..k_offset + r, k_offset + r..])
                        .fill(0.0);
                    s_current_list[i]
                        .slice_mut(s![k_offset + r.., k_offset..k_offset + r])
                        .fill(0.0);
                }
            } else {
                // SUB-DOMINANT penalty: Zero out its range-space block.
                // Zero out the range-space rows and columns (top-left block)
                s_current_list[i]
                    .slice_mut(s![k_offset..k_offset + r, k_offset..k_offset + r])
                    .fill(0.0);
                // Zero out the off-diagonal blocks connecting range and null spaces
                s_current_list[i]
                    .slice_mut(s![k_offset..k_offset + r, k_offset + r..])
                    .fill(0.0);
                s_current_list[i]
                    .slice_mut(s![k_offset + r.., k_offset..k_offset + r])
                    .fill(0.0);
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
        // Form full penalty from transformed root: S_k = rS_k^T * rS_k
        let s_k_transformed = final_rs_transformed[i].t().dot(&final_rs_transformed[i]);
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
        let s_k_transformed = final_rs_transformed[k].t().dot(&final_rs_transformed[k]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::data::TrainingData;
    use crate::calibrate::model::{BasisConfig, LinkFunction, ModelConfig};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Helper function to create test data for construction tests
    fn create_test_data_for_construction(
        n_samples: usize,
        num_pcs: usize,
    ) -> (TrainingData, ModelConfig) {
        let mut rng = StdRng::seed_from_u64(12345);

        // Generate PGS data (standard normal)
        let p: Array1<f64> = (0..n_samples).map(|_| rng.gen_range(-2.0..2.0)).collect();

        // Generate PC data if requested
        let pcs = if num_pcs > 0 {
            let mut pc_data = Array2::zeros((n_samples, num_pcs));
            for i in 0..n_samples {
                for j in 0..num_pcs {
                    pc_data[[i, j]] = rng.gen_range(-1.5..1.5);
                }
            }
            pc_data
        } else {
            Array2::zeros((n_samples, 0))
        };

        // Generate simple response (linear combination for simplicity)
        let y: Array1<f64> = (0..n_samples)
            .map(|i| {
                let mut response = 0.5 * p[i];
                for j in 0..num_pcs {
                    response += 0.3 * pcs[[i, j]];
                }
                response + rng.gen_range(-0.1..0.1) // small noise
            })
            .collect();

        // Calculate ranges before moving data into struct
        let pgs_range = (
            p.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            p.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
        );

        let pc_ranges = (0..num_pcs)
            .map(|j| {
                let col = pcs.column(j);
                (
                    col.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                )
            })
            .collect();

        let weights = Array1::ones(n_samples);
        let data = TrainingData { y, p, pcs, weights };

        // Create config with known parameters - need to provide all required fields
        let pgs_basis_config = BasisConfig {
            num_knots: 5,
            degree: 3,
        };

        let pc_basis_configs = (0..num_pcs)
            .map(|_| BasisConfig {
                num_knots: 4,
                degree: 3,
            })
            .collect();

        let pc_names: Vec<String> = (0..num_pcs).map(|i| format!("PC{}", i + 1)).collect();

        let config = ModelConfig {
            link_function: LinkFunction::Identity,
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 100,
            pgs_basis_config,
            pc_basis_configs,
            pgs_range,
            pc_ranges,
            pc_names,
            constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
        };

        (data, config)
    }

    #[test]
    fn test_matrix_and_layout_dimensions_are_consistent() {
        // Setup with 1 PC to create main effect and interaction terms
        let (data, config) = create_test_data_for_construction(100, 1);

        let (x, s_list, layout, _, _, _range_transforms) =
            build_design_and_penalty_matrices(&data, &config).unwrap();

        // Option 3 dimensional calculation - direct computation based on basis sizes and null space

        // PGS main is still unpenalized and sum-to-zero constrained
        let pgs_main_coeffs =
            config.pgs_basis_config.num_knots + config.pgs_basis_config.degree - 1; // 6 - 1 - 1 = 4

        // PC1 main is now range-only (penalized part only)
        // Range dimension = main basis cols - null space dimension
        let pc1_main_basis_cols =
            config.pc_basis_configs[0].num_knots + config.pc_basis_configs[0].degree; // 5 - 1 = 4
        let r_pc = pc1_main_basis_cols - config.penalty_order; // 4 - 2 = 2

        // Interaction is R×R (both dimensions use range-only)
        let pgs_main_basis_cols =
            config.pgs_basis_config.num_knots + config.pgs_basis_config.degree; // 6 - 1 = 5  
        let r_pgs = pgs_main_basis_cols - config.penalty_order; // 5 - 2 = 3
        let interaction_coeffs = r_pgs * r_pc; // 3 * 2 = 6

        let expected_total_coeffs = 1 + pgs_main_coeffs + r_pc + interaction_coeffs; // 1 + 4 + 2 + 6 = 13
        let expected_num_penalties = 2; // Individual: 1 for PC1 main + 1 for PC1×PGS interaction

        assert_eq!(
            layout.total_coeffs, expected_total_coeffs,
            "Layout total coefficient count is wrong."
        );
        assert_eq!(
            x.ncols(),
            expected_total_coeffs,
            "Design matrix column count is wrong."
        );
        assert_eq!(
            layout.num_penalties, expected_num_penalties,
            "Layout penalty count is wrong."
        );
        assert_eq!(
            s_list.len(),
            expected_num_penalties,
            "Number of generated penalty matrices is wrong."
        );
    }

    #[test]
    fn test_interaction_design_matrix_is_full_rank() {
        let (data, config) = create_test_data_for_construction(100, 1);
        let (x, _, _, _, _, _) = build_design_and_penalty_matrices(&data, &config).unwrap();

        // Calculate numerical rank via SVD
        let svd = x.svd(false, false).expect("SVD failed");
        let max_s_val = svd.1.iter().fold(0.0f64, |a, &b| a.max(b));
        let rank_tol = max_s_val * 1e-12;
        let rank = svd.1.iter().filter(|&&s| s > rank_tol).count();

        // The design matrix should be full-rank by construction
        // The code cleverly builds interaction terms using main effect bases that already
        // have intercept columns removed, preventing rank deficiency from the start
        assert_eq!(
            rank,
            x.ncols(),
            "The design matrix with interactions should be full-rank by construction. Rank: {}, Columns: {}",
            rank,
            x.ncols()
        );
    }

    #[test]
    fn test_interaction_term_has_correct_penalty_structure() {
        let (data, config) = create_test_data_for_construction(100, 1);
        let (_, s_list, layout, _, _, _) =
            build_design_and_penalty_matrices(&data, &config).unwrap();

        // Option 3: Expect grouped penalties - total of 2 penalty matrices
        assert_eq!(
            s_list.len(),
            2,
            "Option 3 should have exactly 2 grouped penalty matrices"
        );

        let interaction_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::Interaction)
            .expect("Interaction block not found in layout");

        // Option 3: Each block now has only 1 penalty index (grouped)
        assert_eq!(
            interaction_block.penalty_indices.len(),
            1,
            "Interaction term should have one grouped penalty index"
        );

        let interaction_penalty_idx = interaction_block.penalty_indices[0];
        assert_eq!(
            interaction_penalty_idx, 1,
            "Interaction should use penalty index 1 (PC mains use 0)"
        );

        // Verify penalty matrix structure
        let s_interactions = &s_list[1]; // Interaction penalty matrix
        let s_pc_mains = &s_list[0]; // PC main effects penalty matrix

        // Check that interaction penalty matrix has identity on interaction blocks and zeros elsewhere
        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if interaction_block.col_range.contains(&r)
                    && interaction_block.col_range.contains(&c)
                {
                    // Within interaction block: should be identity (1 on diagonal, 0 off-diagonal)
                    if r == c {
                        assert_abs_diff_eq!(s_interactions[[r, c]], 1.0, epsilon = 1e-12);
                    } else {
                        assert_abs_diff_eq!(s_interactions[[r, c]], 0.0, epsilon = 1e-12);
                    }
                } else {
                    // Outside interaction block: should be zero
                    assert_abs_diff_eq!(s_interactions[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }

        // Check that PC main penalty matrix has identity on PC main blocks and zeros elsewhere
        let pc_main_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::PcMainEffect)
            .expect("PC main effect block not found in layout");

        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if pc_main_block.col_range.contains(&r) && pc_main_block.col_range.contains(&c) {
                    // Within PC main block: should be identity
                    if r == c {
                        assert_abs_diff_eq!(s_pc_mains[[r, c]], 1.0, epsilon = 1e-12);
                    } else {
                        assert_abs_diff_eq!(s_pc_mains[[r, c]], 0.0, epsilon = 1e-12);
                    }
                } else {
                    // Outside PC main block: should be zero
                    assert_abs_diff_eq!(s_pc_mains[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_construction_with_no_pcs() {
        let (data, config) = create_test_data_for_construction(100, 0); // 0 PCs

        let (_, _, layout, _, _, _) = build_design_and_penalty_matrices(&data, &config).unwrap();

        let pgs_main_coeffs =
            config.pgs_basis_config.num_knots + config.pgs_basis_config.degree - 1;
        let expected_total_coeffs = 1 + pgs_main_coeffs; // Intercept + PGS main effect

        assert_eq!(layout.total_coeffs, expected_total_coeffs);
        assert!(
            layout
                .penalty_map
                .iter()
                .all(|b| b.term_type != TermType::Interaction),
            "No interaction terms should exist in a PGS-only model."
        );
        assert_eq!(
            layout.num_penalties, 0,
            "PGS-only model should have no penalties."
        );
    }
}
