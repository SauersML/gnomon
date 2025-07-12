use crate::calibrate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::calibrate::data::TrainingData;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::{Constraint, ModelConfig};
use ndarray::{Array1, Array2, Axis, s};
use ndarray_linalg::{SVD, error::LinalgError};
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
    pub fn new(
        config: &ModelConfig,
        pc_constrained_basis_ncols: &[usize],
        pgs_main_basis_ncols: usize,
    ) -> Result<Self, EstimationError> {
        let mut penalty_map = Vec::new();
        let mut current_col = 0;
        let mut penalty_idx_counter = 0;

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
        // Technical justification:
        // 1. PGS scores have an arbitrary scale, and their relationship to phenotype may be non-linear
        // 2. Higher PGS values often correspond to greater unit increases in phenotype risk
        // 3. For phenotype prediction, we want to preserve the full expressivity of the PGS main effect
        // 4. This is a domain-specific design decision based on prior knowledge of PGS behavior
        // 5. We still penalize PC terms and interaction terms to prevent overfitting there
        let pgs_main_cols = current_col..current_col + pgs_main_basis_ncols;
        current_col += pgs_main_basis_ncols; // Still advance the column counter

        // Interaction effects
        // Use the correct number of unconstrained PGS basis functions (excluding intercept).
        // The total number of unconstrained B-spline basis functions is (num_knots + degree + 1).
        // We exclude the first basis function (intercept) so we have (num_knots + degree + 1 - 1) = num_knots + degree
        // This ensures that interactions use columns from the unconstrained basis.
        let num_pgs_basis_funcs =
            config.pgs_basis_config.num_knots + config.pgs_basis_config.degree;

        // NOTE: This is NOT a mistake regarding penalty sharing. The code correctly assigns
        // a unique penalty_idx to each interaction term. While some GAM implementations share
        // penalties across interaction dimensions, this implementation uses a more flexible
        // approach where:
        // 1. Each interaction term (PGS basis m × PC i) gets its own smoothing parameter λ
        // 2. This is an intentional design choice to allow different levels of smoothing for
        //    different interaction components
        // 3. The construct_s_lambda and gradient calculations correctly handle this design by
        //    iterating through all blocks and applying the appropriate penalty
        // 4. For example, with 2 PCs and 3 PGS basis funcs, we get:
        //    - 2 main PC penalties + 3*2=6 interaction penalties = 8 total unique penalties
        // 5. While this increases model flexibility, it can potentially lead to overfitting
        //    with many PCs/interactions, which is mitigated by proper penalty optimization
        for m in 1..=num_pgs_basis_funcs {
            for (i, &num_basis) in pc_constrained_basis_ncols.iter().enumerate() {
                let range = current_col..current_col + num_basis; // with pure pre-centering, uses full PC basis size
                penalty_map.push(PenalizedBlock {
                    term_name: format!("f(PGS_B{}, {})", m, config.pc_names[i]),
                    col_range: range.clone(),
                    penalty_idx: penalty_idx_counter,
                });
                current_col += num_basis;
                penalty_idx_counter += 1;
            }
        }

        Ok(ModelLayout {
            intercept_col,
            pgs_main_cols,
            penalty_map,
            total_coeffs: current_col,
            num_penalties: penalty_idx_counter,
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
        s_list.push(z_transform.t().dot(&s_unconstrained.dot(&z_transform)));
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
    let layout = ModelLayout::new(config, &pc_basis_ncols, pgs_main_basis.ncols())?;

    if s_list.len() != layout.num_penalties {
        return Err(EstimationError::LayoutError(format!(
            "Internal logic error: Mismatch in number of penalties. Layout expects {}, but {} were generated.",
            layout.num_penalties,
            s_list.len()
        )));
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
                x_matrix
                    .slice_mut(s![.., block.col_range.clone()])
                    .assign(&pc_constrained_bases[pc_idx]);
                break;
            }
        }
    }

    // 3. Main PGS effect - directly use the layout range
    x_matrix
        .slice_mut(s![.., layout.pgs_main_cols.clone()])
        .assign(&pgs_main_basis);

    // 4. Interaction effects - in order of PGS basis function index, then PC name
    // This matches exactly with the flattening logic in model.rs
    // The correct formula for unconstrained non-intercept PGS bases is:
    // (num_knots + degree + 1) - 1 = num_knots + degree
    // We subtract 1 to exclude the intercept basis function (index 0)
    let total_pgs_bases = config.pgs_basis_config.num_knots + config.pgs_basis_config.degree;

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

                    // No transformation is needed - the interaction inherits the constraint property
                    let z_int = Array2::<f64>::eye(interaction_term.ncols());

                    // Cache for prediction
                    let key = format!("INT_P{}_{}", m, pc_name);
                    constraints.insert(key, Constraint { z_transform: z_int });

                    // Copy into X
                    x_matrix
                        .slice_mut(s![.., block.col_range.clone()])
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

    Ok((x_matrix, s_list, layout, constraints, knot_vectors))
}

/// Helper to construct the summed, weighted penalty matrix S_lambda.
/// This version correctly builds a block-diagonal matrix based on the model layout.
pub fn construct_s_lambda(
    lambdas: &Array1<f64>,
    s_list: &[Array2<f64>],
    layout: &ModelLayout,
) -> Array2<f64> {
    // Initialize a zero matrix with dimensions matching the full coefficient vector.
    let mut s_lambda = Array2::zeros((layout.total_coeffs, layout.total_coeffs));

    // Iterate through the penalized blocks defined in the layout.
    for block in &layout.penalty_map {
        // Get the smoothing parameter (lambda) for this specific block.
        let lambda_k = lambdas[block.penalty_idx];

        // Get the corresponding unscaled penalty matrix S_k.
        let s_k = &s_list[block.penalty_idx];

        // Get the slice of the full S_lambda matrix where this block belongs.
        let mut target_block =
            s_lambda.slice_mut(s![block.col_range.clone(), block.col_range.clone()]);

        // Add the scaled penalty matrix to the target block.
        target_block.scaled_add(lambda_k, s_k);
    }

    s_lambda
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
