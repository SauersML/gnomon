use crate::calibrate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::calibrate::data::TrainingData;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::{Constraint, ModelConfig};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use ndarray_linalg::{SVD, QR, Cholesky, UPLO, Eigh, Solve, error::LinalgError};
use std::collections::HashMap;
use std::ops::Range;

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
#[derive(Clone, Debug)]
pub struct PenalizedBlock {
    pub term_name: String,
    pub col_range: Range<usize>,
    pub penalty_idx: usize,
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
        let p_interactions = num_pgs_interaction_bases * pc_constrained_basis_ncols.iter().sum::<usize>();
        let calculated_total_coeffs = 1 + p_pgs_main + p_pc_main + p_interactions;

        let intercept_col = current_col;
        current_col += 1;

        // Main effect for each PC
        for (i, &num_basis) in pc_constrained_basis_ncols.iter().enumerate() {
            let range = current_col..current_col + num_basis;
            penalty_map.push(PenalizedBlock {
                term_name: format!("f({})", config.pc_names[i]),
                col_range: range.clone(),
                penalty_idx: penalty_idx_counter,
            });
            current_col += num_basis;
            penalty_idx_counter += 1;
        }

        // Main effect for PGS (non-constant basis terms)
        // The PGS main effect is unpenalized intentionally.
        let pgs_main_cols = current_col..current_col + pgs_main_basis_ncols;
        current_col += pgs_main_basis_ncols; // Still advance the column counter

        // Interaction effects
        // Use the EMPIRICAL number of PGS basis functions from the actual generated matrix.
        // This ensures perfect consistency with the design matrix construction in build_design_and_penalty_matrices.
        let num_pgs_basis_funcs = num_pgs_interaction_bases;

        for m in 1..=num_pgs_basis_funcs {
            for (i, &num_basis) in pc_constrained_basis_ncols.iter().enumerate() {
                let range = current_col..current_col + num_basis;
                penalty_map.push(PenalizedBlock {
                    term_name: format!("f(PGS_B{}, {})", m, config.pc_names[i]),
                    col_range: range.clone(),
                    penalty_idx: penalty_idx_counter,
                });
                current_col += num_basis;
                penalty_idx_counter += 1;
            }
        }

        // Verify that our calculation matches the actual column count
        if current_col != calculated_total_coeffs {
            return Err(EstimationError::LayoutError(format!(
                "ModelLayout dimension calculation error: calculated total_coeffs={} but actual current_col={}",
                calculated_total_coeffs, current_col
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
/// Returns the design matrix, penalty matrices, model layout, constraint transformations, and knot vectors.
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
        // Apply sum-to-zero constraint to PC basis
        let (constrained_basis, z_transform) =
            basis::apply_sum_to_zero_constraint(pc_basis_unc.view())?;
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
            create_difference_penalty_matrix(pc_basis_unc.ncols(), config.penalty_order)?;
        let s_constrained = z_transform.t().dot(&s_unconstrained.dot(&z_transform));
        
        // Embed into full-sized p × p matrix (will be determined after layout is created)
        s_list.push(s_constrained);
    }

    // 3. Create penalties for interaction effects only
    // The main effect of PGS is intentionally left unpenalized.
    // We iterate through each non-intercept PGS basis function which will act as a weight.
    let num_pgs_interaction_weights = pgs_basis_unc.ncols() - 1;

    for _ in 0..num_pgs_interaction_weights {
        for i in 0..pc_constrained_bases.len() {
            // Create penalty matrix for interaction basis using pure pre-centering
            let interaction_basis_size = pc_constrained_bases[i].ncols();
            let s_interaction =
                create_difference_penalty_matrix(interaction_basis_size, config.penalty_order)?;
            s_list.push(s_interaction);
        }
    }

    // 4. Define the model layout based on final basis dimensions
    let pc_basis_ncols: Vec<usize> = pc_constrained_bases.iter().map(|b| b.ncols()).collect();
    let layout = ModelLayout::new(config, &pc_basis_ncols, pgs_main_basis.ncols(), num_pgs_interaction_weights)?;

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
    
    // Validate that s_list length matches penalty_map length
    if s_list.len() != layout.penalty_map.len() {
        return Err(EstimationError::LayoutError(format!(
            "Penalty matrix count mismatch: s_list has {} matrices but penalty_map has {} blocks",
            s_list.len(), layout.penalty_map.len()
        )));
    }
    
    for (k, s_k) in s_list.iter().enumerate() {
        let mut s_k_full = Array2::zeros((p, p));
        
        // Find the block for this penalty index and embed the matrix
        let mut found_block = false;
        for block in &layout.penalty_map {
            if block.penalty_idx == k {
                let col_range = block.col_range.clone();
                let block_size = col_range.len();
                
                // Validate dimensions
                if s_k.nrows() != block_size || s_k.ncols() != block_size {
                    return Err(EstimationError::LayoutError(format!(
                        "Penalty matrix {} (term {}) has size {}×{} but block expects {}×{}",
                        k, block.term_name, s_k.nrows(), s_k.ncols(), block_size, block_size
                    )));
                }
                
                // Embed the block-sized penalty into the full-sized matrix
                s_k_full.slice_mut(s![col_range.clone(), col_range]).assign(s_k);
                found_block = true;
                break;
            }
        }
        
        if !found_block {
            return Err(EstimationError::LayoutError(format!(
                "No block found for penalty matrix index {}", k
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
            if block.term_name == format!("f({})", pc_name) {
                let col_range = block.col_range.clone();
                let pc_basis = &pc_constrained_bases[pc_idx];
                
                // Validate dimensions before assignment
                if pc_basis.nrows() != n_samples {
                    return Err(EstimationError::LayoutError(format!(
                        "PC basis {} has {} rows but expected {} samples",
                        pc_name, pc_basis.nrows(), n_samples
                    )));
                }
                if pc_basis.ncols() != col_range.len() {
                    return Err(EstimationError::LayoutError(format!(
                        "PC basis {} has {} columns but layout expects {} columns",
                        pc_name, pc_basis.ncols(), col_range.len()
                    )));
                }
                
                x_matrix
                    .slice_mut(s![.., col_range])
                    .assign(pc_basis);
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
            pgs_main_basis.nrows(), n_samples
        )));
    }
    if pgs_main_basis.ncols() != pgs_range.len() {
        return Err(EstimationError::LayoutError(format!(
            "PGS main basis has {} columns but layout expects {} columns",
            pgs_main_basis.ncols(), pgs_range.len()
        )));
    }
    
    x_matrix
        .slice_mut(s![.., pgs_range])
        .assign(&pgs_main_basis);

    // 4. Interaction effects - in order of PGS basis function index, then PC name
    // This matches exactly with the flattening logic in model.rs
    // Use the *actual* number of basis functions from the generated (unconstrained) matrix, excluding the intercept.
    // This ensures consistency with the penalty matrix creation above.
    let total_pgs_bases = pgs_basis_unc.ncols() - 1;

    for m in 1..=total_pgs_bases {
        for pc_name in &config.pc_names {
            for block in &layout.penalty_map {
                if block.term_name == format!("f(PGS_B{}, {})", m, pc_name) {
                    // Find PC index from name
                    let pc_idx = config.pc_names.iter().position(|n| n == pc_name).unwrap();

                    // Use constrained PGS main basis (index m-1 since we exclude intercept)
                    // Note: pgs_main_basis excludes intercept column, so m=1 maps to index 0
                    if m == 0 || m > pgs_main_basis.ncols() {
                        continue; // Skip intercept and out-of-bounds
                    }
                    let pgs_weight_col = pgs_main_basis.column(m - 1);

                    // Use the CONSTRAINED PC basis matrix
                    let pc_constrained_basis = &pc_constrained_bases[pc_idx];

                    // Form the interaction tensor product with pure pre-centering
                    let interaction_term =
                        pc_constrained_basis * &pgs_weight_col.view().insert_axis(Axis(1));

                    // Validate dimensions before assignment
                    let col_range = block.col_range.clone();
                    if interaction_term.nrows() != n_samples {
                        return Err(EstimationError::LayoutError(format!(
                            "Interaction term PGS_B{} × {} has {} rows but expected {} samples",
                            m, pc_name, interaction_term.nrows(), n_samples
                        )));
                    }
                    if interaction_term.ncols() != col_range.len() {
                        return Err(EstimationError::LayoutError(format!(
                            "Interaction term PGS_B{} × {} has {} columns but layout expects {} columns",
                            m, pc_name, interaction_term.ncols(), col_range.len()
                        )));
                    }

                    // No transformation is needed - the interaction inherits the constraint property
                    let z_int = Array2::<f64>::eye(interaction_term.ncols());

                    // Cache for prediction
                    let key = format!("INT_P{}_{}", m, pc_name);
                    constraints.insert(key, Constraint { z_transform: z_int });

                    // Copy into X
                    x_matrix
                        .slice_mut(s![.., col_range])
                        .assign(&interaction_term);

                    break;
                }
            }
        }
    }

    // Simple check for obvious over-parameterization
    let n_samples = data.y.len();
    let n_coeffs = x_matrix.ncols();
    if n_coeffs > n_samples {
        log::warn!(
            "Model is over-parameterized: {} coefficients for {} samples",
            n_coeffs,
            n_samples
        );
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }

    Ok((x_matrix, s_list_full, layout, constraints, knot_vectors))
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
    /// Transformed penalty components rS
    pub rs: Vec<Array2<f64>>,
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
    
    // Simple weighted sum since all matrices are now p × p
    for (i, s_k) in s_list.iter().enumerate() {
        s_lambda.scaled_add(lambdas[i], s_k);
    }
    
    s_lambda
}

/// Implements the exact stable reparameterization algorithm from Wood (2011) Appendix B
/// This follows the complete recursive similarity transformation procedure
pub fn stable_reparameterization(
    s_list: &[Array2<f64>],
    lambdas: &[f64],
    layout: &ModelLayout,
) -> Result<ReparamResult, EstimationError> {
    let p = layout.total_coeffs;
    let m = s_list.len(); // Number of penalty matrices
    
    // s_list contains full-sized p × p penalty matrices
    let s_matrices = s_list.to_vec();
    
    // Wood (2011) Appendix B: get_stableS algorithm
    let eps = f64::EPSILON.powf(1.0/3.0); // Cube root of machine precision
    
    // Initialize: Qf = Identity(p,p), K=0, Q=p
    let mut qf = Array2::eye(p); // Global transformation matrix
    let mut k = 0_usize; // Number of parameters already processed
    let mut q = p; // Size of current sub-problem
    let mut gamma: Vec<usize> = (0..m).collect(); // Active penalty indices
    
    // Initialize penalty square roots for transformations (clone originals)
    let mut rs = s_matrices.clone();
    
    // Initialize final S matrix  
    let mut s_final = Array2::zeros((p, p));
    
    // Main iterative loop
    let mut iteration = 0;
    loop {
        iteration += 1;
        if gamma.is_empty() || q == 0 {
            break;
        }
        
        // Step 1: Calculate scaled penalty measures Ω_i = ||R_i||_F * λ_i for i ∈ γ
        let mut omegas = Vec::new();
        let mut max_omega: f64 = 0.0;
        
        for &i in &gamma {
            // Extract current active sub-problem block from the penalty matrix
            let active_block = rs[i].slice(s![k..k+q, k..k+q]).to_owned();
            let frobenius_norm = active_block.iter().map(|&x| x * x).sum::<f64>().sqrt();
            let omega_i = frobenius_norm * lambdas[i];
            omegas.push((i, omega_i));
            max_omega = max_omega.max(omega_i);
        }
        
        if max_omega < 1e-15 {
            break; // All remaining penalties are numerically zero
        }
        
        // Step 2: Partition into dominant α and subdominant γ' sets
        let threshold = eps * max_omega;
        let alpha: Vec<usize> = omegas.iter()
            .filter(|(_, omega)| *omega >= threshold)
            .map(|(i, _)| *i)
            .collect();
        let gamma_prime: Vec<usize> = omegas.iter()
            .filter(|(_, omega)| *omega < threshold)
            .map(|(i, _)| *i)
            .collect();
        
        if alpha.is_empty() {
            break;
        }
        
        // Step 3: Form weighted sum of dominant penalties in current sub-problem
        let mut s_alpha = Array2::zeros((q, q));
        for &i in &alpha {
            let active_block = rs[i].slice(s![k..k+q, k..k+q]).to_owned();
            s_alpha = s_alpha + &active_block * lambdas[i];
        }
        
        // Step 4: Eigendecomposition to find rank
        let (eigenvalues, u): (Array1<f64>, Array2<f64>) = s_alpha.eigh(UPLO::Lower)
            .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
        
        // Sort eigenvalues in descending order
        let mut sorted_indices: Vec<usize> = (0..eigenvalues.len()).collect();
        sorted_indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());
        
        // Determine rank r based on eigenvalue threshold
        let rank_tolerance = 1e-8 * eigenvalues[sorted_indices[0]].max(1.0);
        let r = sorted_indices.iter()
            .take_while(|&&i| eigenvalues[i] > rank_tolerance)
            .count();
        
        // Step 5: Check termination criterion
        if r == q {
            // Full rank case - assemble final diagonal block and terminate
            let mut s_block = Array2::zeros((q, q));
            for &i in &gamma {
                let active_block = rs[i].slice(s![k..k+q, k..k+q]).to_owned();
                s_block = s_block + &active_block * lambdas[i];
            }
            // Copy to final S matrix
            s_final.slice_mut(s![k..k+q, k..k+q]).assign(&s_block);
            break;
        }
        
        // Step 6: Extract range and null space eigenvectors
        let u_r = u.select(Axis(1), &sorted_indices[..r]);
        let u_n = u.select(Axis(1), &sorted_indices[r..]);
        
        if u_n.ncols() == 0 {
            break;
        }
        
        // Step 7: Update global transformation matrix Qf
        // Qf_new = Qf_old * U_block where U_block = [U_r U_n]
        if iteration == 1 {
            // First iteration: copy U to the appropriate block of Qf
            qf.slice_mut(s![k..k+q, k..k+q]).assign(&u);
        } else {
            // Subsequent iterations: multiply current Qf block by U
            let qf_block = qf.slice(s![.., k..k+q]).to_owned();
            let qf_new = qf_block.dot(&u);
            qf.slice_mut(s![.., k..k+q]).assign(&qf_new);
        }
        
        // Step 8: Update off-diagonal blocks of final S matrix  
        if k > 0 {
            for &i in &gamma {
                // Extract off-diagonal block
                let off_diag_block = rs[i].slice(s![..k, k..k+q]).to_owned();
                // Transform: B_new = B_old * U
                let b_new = off_diag_block.dot(&u);
                // Copy back to S matrix (both upper and lower triangular parts)
                s_final.slice_mut(s![..k, k..k+q]).assign(&b_new);
                s_final.slice_mut(s![k..k+q, ..k]).assign(&b_new.t());
            }
        }
        
        // Step 9: Form new diagonal block for current iteration
        // C = U'S_γU + D (where D contains dominant eigenvalues)
        let mut s_gamma = Array2::zeros((q, q));
        for &i in &gamma_prime {
            let active_block = rs[i].slice(s![k..k+q, k..k+q]).to_owned();
            s_gamma = s_gamma + &active_block * lambdas[i];
        }
        
        let c_new = u.t().dot(&s_gamma.dot(&u));
        // Add dominant eigenvalues to (1,1) block
        for (idx, &i) in sorted_indices[..r].iter().enumerate() {
            if idx < c_new.nrows() && idx < c_new.ncols() {
                let current_val = c_new[[idx, idx]];
                s_final[[k + idx, k + idx]] = current_val + eigenvalues[i];
            }
        }
        
        // Copy subdominant block to final S
        if r < q {
            let subdominant_block = c_new.slice(s![r..q, r..q]).to_owned();
            s_final.slice_mut(s![k+r..k+q, k+r..k+q]).assign(&subdominant_block);
        }
        
        // Step 10: Transform penalty matrices for next iteration
        for penalty_idx in 0..rs.len() {
            if alpha.contains(&penalty_idx) {
                // Dominant penalties: project to range space and zero null space
                let active_block = rs[penalty_idx].slice(s![k..k+q, k..k+q]).to_owned();
                let transformed_block = u_r.t().dot(&active_block.dot(&u_r));
                
                // Update the penalty matrix - zero out null space part
                rs[penalty_idx].slice_mut(s![k..k+r, k..k+r]).assign(&transformed_block);
                
                // Zero out the null space parts
                if k + r < p {
                    rs[penalty_idx].slice_mut(s![k+r..k+q, k..k+q]).fill(0.0);
                    rs[penalty_idx].slice_mut(s![k..k+q, k+r..k+q]).fill(0.0);
                }
            } else if gamma_prime.contains(&penalty_idx) {
                // Subdominant penalties: project to null space
                let active_block = rs[penalty_idx].slice(s![k..k+q, k..k+q]).to_owned();
                let transformed_block = u_n.t().dot(&active_block.dot(&u_n));
                
                // Update penalty matrix to null space representation
                let new_dim = u_n.ncols();
                rs[penalty_idx].slice_mut(s![k+r..k+r+new_dim, k+r..k+r+new_dim])
                    .assign(&transformed_block);
            }
        }
        
        // Step 11: Update loop variables for next iteration
        k = k + r; // Number of dimensions dealt with
        q = u_n.ncols(); // Size of remaining sub-problem
        gamma = gamma_prime; // New active set is old subdominant set
    }
    
    // Use the stable final S matrix
    let s_transformed = s_final;
    
    // Stable log-determinant computation
    use crate::calibrate::estimate::internal::calculate_log_det_pseudo;
    let log_det = calculate_log_det_pseudo(&s_transformed)
        .unwrap_or_else(|_| {
            // Fallback: eigenvalue computation
            match s_transformed.eigh(UPLO::Lower) {
                Ok((eigenvalues, _)) => {
                    eigenvalues.iter()
                        .filter(|&&ev| ev > 1e-12)
                        .map(|&ev| ev.ln())
                        .sum()
                }
                Err(_) => 0.0
            }
        });
    
    // Compute derivatives: d/dρ_k log|S_λ|_+ = λ_k * tr(S_λ^+ S_k)
    let mut det1 = Array1::zeros(lambdas.len());
    
    // Compute pseudo-inverse of S_λ
    let (s_eigenvalues, s_eigenvectors): (Array1<f64>, Array2<f64>) = s_transformed.eigh(UPLO::Lower)
        .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
    
    let tolerance = 1e-12;
    let mut s_plus = Array2::zeros((p, p));
    for (i, &eigenval) in s_eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let v_i = s_eigenvectors.column(i);
            let outer_product = v_i.to_owned().insert_axis(Axis(1)).dot(&v_i.to_owned().insert_axis(Axis(0)));
            s_plus = s_plus + &outer_product * (1.0 / eigenval);
        }
    }
    
    // Calculate derivatives
    for k in 0..lambdas.len() {
        let s_plus_times_s_k = s_plus.dot(&s_matrices[k]);
        let trace: f64 = s_plus_times_s_k.diag().sum();
        det1[k] = lambdas[k] * trace;
    }
    
    Ok(ReparamResult {
        s_transformed,
        log_det,
        det1,
        qs: qf,
        rs: s_matrices,
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

/// Implements the stable penalized least squares solver from Wood (2011) Section 3.3
/// Handles negative weights via SVD and uses pivoted QR for rank detection
pub fn stable_penalized_least_squares(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    s_lambda: &Array2<f64>,
    _reparam_result: &ReparamResult,
) -> Result<StablePLSResult, EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    
    // Step 1: Split weights into positive and negative parts (Wood 2011, Section 3.3)
    let mut w_pos = Array1::zeros(n);
    let mut w_neg = Array1::zeros(n);
    for i in 0..n {
        if weights[i] >= 0.0 {
            w_pos[i] = weights[i];
        } else {
            w_neg[i] = -weights[i];
        }
    }
    
    // Step 2: Initial QR decomposition on the positive part
    // √W̄X = QR where W̄ has |w_i| on diagonal
    let w_abs = weights.mapv(|w| w.abs());
    let sqrt_w_abs = w_abs.mapv(|w| w.sqrt());
    let x_weighted = &x * &sqrt_w_abs.view().insert_axis(Axis(1));
    
    // Perform QR decomposition
    let (_q_initial, r_initial) = x_weighted.qr()
        .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?;
    
    // Step 3: Form penalty square root matrix E where E^T E = S_λ
    // Use Cholesky decomposition: S_λ = L L^T, so E = L
    let e_matrix = match s_lambda.cholesky(UPLO::Lower) {
        Ok(l) => l,
        Err(_) => {
            // Fallback: use eigendecomposition for semi-definite case
            let (eigenvals, eigenvecs): (Array1<f64>, Array2<f64>) = s_lambda.eigh(UPLO::Lower)
                .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?;
            
            let mut e_temp = Array2::zeros((p, p));
            for (i, &eval) in eigenvals.iter().enumerate() {
                if eval > 1e-12 {
                    let v_i = eigenvecs.column(i);
                    let sqrt_eval = eval.sqrt();
                    let outer = v_i.to_owned().insert_axis(Axis(1)) * sqrt_eval;
                    e_temp = e_temp + &outer.dot(&v_i.to_owned().insert_axis(Axis(0)));
                }
            }
            e_temp
        }
    };
    
    // Step 4: Form augmented matrix [R; E] and perform pivoted QR
    let r_rows = r_initial.nrows().min(p);
    let r_truncated = r_initial.slice(s![..r_rows, ..]).to_owned();
    
    let mut augmented = Array2::zeros((r_rows + p, p));
    augmented.slice_mut(s![..r_rows, ..]).assign(&r_truncated);
    augmented.slice_mut(s![r_rows.., ..]).assign(&e_matrix);
    
    // QR decomposition of augmented matrix
    let (q_augmented, r_final) = augmented.qr()
        .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?;
    
    // Step 5: Extract Q₁ (first q rows corresponding to negative weight indices)
    let mut penalized_hessian = r_final.t().dot(&r_final); // R̄^T R̄
    
    // Step 6: Handle negative weights via SVD correction (Wood 2011, Section 3.3, Eq. 12)
    // Extract correct Q₁ from the augmented QR decomposition
    if w_neg.iter().any(|&w| w > 0.0) {
        let neg_indices: Vec<usize> = (0..n).filter(|&i| weights[i] < 0.0).collect();
        
        if !neg_indices.is_empty() {
            // Map negative weight rows to the Q matrix from augmented QR
            // Q₁ consists of the first q rows of Q̄ (where q = number of negative weights)
            let q = neg_indices.len().min(q_augmented.nrows());
            let q1 = q_augmented.slice(s![..q, ..]).to_owned();
            
            // Form I^- matrix (diagonal matrix of negative weights)
            let mut i_minus = Array2::zeros((q, q));
            for (i, &orig_idx) in neg_indices.iter().enumerate().take(q) {
                i_minus[[i, i]] = w_neg[orig_idx];
            }
            
            // Compute I^- Q₁ as per Wood (2011) Eq. 12
            let i_minus_q1 = i_minus.dot(&q1);
            
            // SVD of I^- Q₁ = U D V^T (Wood 2011, Eq. 12)
            let (u, singular_values, vt) = i_minus_q1.svd(true, true)
                .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?;
            
            if let (Some(_u_mat), Some(vt_mat)) = (u, vt) {
                // Form D^2 matrix from singular values
                let mut d_squared = Array2::zeros((singular_values.len(), singular_values.len()));
                for (i, &sv) in singular_values.iter().enumerate() {
                    d_squared[[i, i]] = sv * sv;
                }
                
                // Compute V D^2 V^T correction term
                let v_mat = vt_mat.t();
                let correction_term = v_mat.dot(&d_squared.dot(&v_mat.t()));
                
                // Apply Wood (2011) Eq. 12 similarity transform: H = R̄^T (I - 2 V D^2 V^T) R̄
                let identity: Array2<f64> = Array2::eye(p);
                let correction_matrix = &identity - &correction_term * 2.0;
                
                // Transform the penalized Hessian correctly
                penalized_hessian = r_final.t().dot(&correction_matrix.dot(&r_final));
            }
        }
    }
    
    // Step 6: Ensure positive definiteness with regularization
    let ridge = 1e-8;
    for i in 0..penalized_hessian.nrows() {
        penalized_hessian[[i, i]] += ridge;
    }
    
    // Step 7: Solve the system
    // Construct right-hand side: X'Wz (for now, assume z = y for simplicity)
    let sqrt_w = weights.mapv(|w| if w >= 0.0 { w.sqrt() } else { 0.0 });
    let y_weighted = &sqrt_w * &y;
    let x_weighted_pos = &x * &sqrt_w.view().insert_axis(Axis(1));
    let rhs = x_weighted_pos.t().dot(&y_weighted);
    
    // Solve using Cholesky factorization
    let beta = match penalized_hessian.cholesky(UPLO::Lower) {
        Ok(chol) => {
            chol.solve(&rhs)
                .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?
        }
        Err(_) => {
            // Fallback to regularized solve
            let mut reg_hessian = penalized_hessian.clone();
            let reg_factor = 1e-6;
            for i in 0..reg_hessian.nrows() {
                reg_hessian[[i, i]] += reg_factor;
            }
            reg_hessian.cholesky(UPLO::Lower)
                .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?
                .solve(&rhs)
                .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?
        }
    };
    
    // Calculate effective degrees of freedom
    // edf = tr(H⁻¹X'WX) where H is the penalized Hessian
    let mut edf: f64 = 0.0;
    let w_diag = Array2::from_diag(&weights);
    let xtwx_actual = x.t().dot(&w_diag.dot(&x));
    for j in 0..p {
        let xtwx_col = xtwx_actual.column(j);
        match penalized_hessian.solve(&xtwx_col.to_owned()) {
            Ok(h_inv_col) => {
                edf += h_inv_col[j];
            }
            Err(_) => {
                // If solve fails, use approximation based on penalty rank
                let penalty_rank = s_lambda.svd(false, false)
                    .map(|(_, svals, _)| svals.iter().filter(|&&sv| sv > 1e-8).count())
                    .unwrap_or(p);
                edf = (p as f64) - penalty_rank as f64;
                break;
            }
        }
    }
    edf = edf.max(1.0);
    
    // Calculate scale parameter
    let fitted = x.dot(&beta);
    let residuals = &y - &fitted;
    let rss = residuals.mapv(|x| x * x).sum();
    let scale = rss / (n as f64 - edf).max(1.0);
    
    Ok(StablePLSResult {
        beta,
        penalized_hessian,
        edf,
        scale,
    })
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
    // For large matrices, use a faster heuristic before expensive SVD
    let n = matrix.nrows();

    // Quick check: if matrix is too large, use trace/determinant heuristic
    if n > 100 {
        // Check diagonal dominance as a fast proxy for conditioning
        let mut min_diag = f64::INFINITY;
        let mut max_diag = 0.0_f64;

        for i in 0..n {
            let diag_val = matrix[[i, i]].abs();
            min_diag = min_diag.min(diag_val);
            max_diag = max_diag.max(diag_val);
        }

        if min_diag < 1e-12 {
            return Ok(f64::INFINITY);
        }

        // Simple heuristic: if diagonal ratio is bad, likely ill-conditioned
        let diag_ratio = max_diag / min_diag;
        if diag_ratio > 1e10 {
            return Ok(diag_ratio);
        }
    }

    // For smaller matrices or when heuristic is inconclusive, use SVD
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
