use crate::calibrate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::calibrate::data::TrainingData;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::faer_ndarray::{FaerEigh, FaerLinalgError, FaerSvd};
use crate::calibrate::model::ModelConfig;
use faer::Side;
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, Axis, s};
use std::collections::HashMap;
use std::ops::Range;

/// Computes weighted column means for functional ANOVA decomposition.
/// Returns the weighted means that would be subtracted by center_columns_in_place.
fn weighted_column_means(x: &Array2<f64>, w: &Array1<f64>) -> Array1<f64> {
    let denom = w.sum();
    if denom <= 0.0 {
        return Array1::zeros(x.ncols());
    }
    // Vectorized: means = (X^T w) / sum(w)
    x.t().dot(w) / denom
}

/// Centers the columns of a matrix using weighted means.
/// This enforces intercept orthogonality (sum-to-zero) for the columns it is applied to.
pub fn center_columns_in_place(x: &mut Array2<f64>, w: &Array1<f64>) {
    let means = weighted_column_means(x, w);
    // Subtract means from each column
    for j in 0..x.ncols() {
        let m = means[j];
        x.column_mut(j).mapv_inplace(|v| v - m);
    }
}

/// Computes the Kronecker product A ⊗ B for penalty matrix construction.
/// This is used to create tensor product penalties that enforce smoothness
/// in multiple dimensions for interaction terms.
/// NOTE: Currently unused due to penalty grouping in Option 3 implementation
/// Function is currently unused but kept for future implementation
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (a_rows, a_cols) = a.dim();
    let (b_rows, b_cols) = b.dim();
    if a_rows == 0 || a_cols == 0 || b_rows == 0 || b_cols == 0 {
        return Array2::zeros((a_rows * b_rows, a_cols * b_cols));
    }
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
    // Trivial early return for degenerate shapes
    if n_samples == 0 || a_cols == 0 || b_cols == 0 {
        return Array2::zeros((n_samples, a_cols * b_cols));
    }
    let mut result = Array2::zeros((n_samples, a_cols * b_cols));
    result
        .axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(r, mut row)| {
            let ar = a.row(r);
            let br = b.row(r);
            let mut k = 0;
            for i in 0..a_cols {
                let ai = ar[i];
                for j in 0..b_cols {
                    row[k] = ai * br[j];
                    k += 1;
                }
            }
        });
    result
}

/// Holds the layout of the design matrix `X` and penalty matrices `S_i`.
#[derive(Clone)]
pub struct ModelLayout {
    pub intercept_col: usize,
    pub pgs_main_cols: Range<usize>,
    /// Unpenalized PC null-space columns (aligned with config.pc_configs order)
    pub pc_null_cols: Vec<Range<usize>>,
    /// Penalty-map indices for each PC's explicit null-space block, if present (aligned with config.pc_configs)
    pub pc_null_block_idx: Vec<Option<usize>>,
    pub penalty_map: Vec<PenalizedBlock>,
    /// Direct indices to avoid string lookups in hot paths
    pub pc_main_block_idx: Vec<usize>,
    pub interaction_block_idx: Vec<usize>,
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
    /// Minimal external layout for arbitrary design matrices used by the calibrator or adapters.
    /// Sets only the fields required by PIRLS and reparameterization.
    pub fn external(total_coeffs: usize, num_penalties: usize) -> Self {
        ModelLayout {
            intercept_col: 0,
            pgs_main_cols: 0..0,
            pc_null_cols: vec![],
            pc_null_block_idx: vec![],
            penalty_map: vec![],
            pc_main_block_idx: vec![],
            interaction_block_idx: vec![],
            total_coeffs,
            num_penalties,
        }
    }
    /// Creates a new layout based on the model configuration and basis dimensions.
    /// Enforces strict dimensional consistency across the entire GAM system.
    pub fn new(
        config: &ModelConfig,
        pc_null_basis_ncols: &[usize],
        pc_range_basis_ncols: &[usize],
        pgs_main_basis_ncols: usize,
        pc_unconstrained_basis_ncols: &[usize],
        num_pgs_interaction_bases: usize,
    ) -> Result<Self, EstimationError> {
        let mut penalty_map = Vec::new();
        let mut current_col = 0;
        let mut penalty_idx_counter = 0;
        let mut pc_null_cols: Vec<Range<usize>> = Vec::with_capacity(config.pc_configs.len());
        let mut pc_main_block_idx: Vec<usize> = Vec::with_capacity(config.pc_configs.len());
        let mut pc_null_block_idx: Vec<Option<usize>> = Vec::with_capacity(config.pc_configs.len());

        // Calculate total coefficients first to ensure consistency
        // Formula: total_coeffs = 1 (intercept) + p_pgs_main + p_pc_main + p_interactions
        let p_pgs_main = pgs_main_basis_ncols;
        let p_pc_main: usize = pc_null_basis_ncols
            .iter()
            .zip(pc_range_basis_ncols.iter())
            .map(|(n_null, n_range)| n_null + n_range)
            .sum();
        // For tensor product interactions: each PC gets num_pgs_bases * num_pc_bases coefficients
        // Use unconstrained dimensions for interaction calculations
        let p_interactions: usize = pc_unconstrained_basis_ncols
            .iter()
            .map(|&num_pc_basis_unc| num_pgs_interaction_bases * num_pc_basis_unc)
            .sum();
        let calculated_total_coeffs = 1 + p_pgs_main + p_pc_main + p_interactions;

        let intercept_col = current_col;
        current_col += 1;

        // Reserve capacities to avoid reallocations
        // Estimated number of penalized blocks: one per PC main effect, plus one per interaction (if enabled)
        let estimated_penalized_blocks = config.pc_configs.len()
            + if num_pgs_interaction_bases > 0 {
                config.pc_configs.len()
            } else {
                0
            };
        penalty_map.reserve_exact(estimated_penalized_blocks);
        pc_null_cols.reserve_exact(config.pc_configs.len());

        // Main effect for each PC (null-space and range-space treated as separate penalized blocks)
        for (i, (&n_null, &n_range)) in pc_null_basis_ncols
            .iter()
            .zip(pc_range_basis_ncols.iter())
            .enumerate()
        {
            // Null-space (unpenalized)
            let null_range = current_col..current_col + n_null;
            pc_null_cols.push(null_range.clone());
            current_col += n_null;

            // Add a dedicated penalty for the null-space columns (mgcv select=TRUE style)
            // This ensures the likelihood is bounded in all directions under the Logit link.
            if n_null > 0 {
                penalty_map.push(PenalizedBlock {
                    term_name: format!("f({})_null", config.pc_configs[i].name),
                    col_range: null_range.clone(),
                    penalty_indices: vec![penalty_idx_counter],
                    term_type: TermType::PcMainEffect,
                });
                pc_null_block_idx.push(Some(penalty_map.len() - 1));
                penalty_idx_counter += 1;
            } else {
                pc_null_block_idx.push(None);
            }

            // Range-space (penalized)
            let range = current_col..current_col + n_range;
            penalty_map.push(PenalizedBlock {
                term_name: format!("f({})", config.pc_configs[i].name),
                col_range: range.clone(),
                penalty_indices: vec![penalty_idx_counter], // Each PC main gets unique penalty index
                term_type: TermType::PcMainEffect,
            });
            pc_main_block_idx.push(penalty_map.len() - 1);
            current_col += n_range;
            penalty_idx_counter += 1; // Increment for next penalty
        }

        // Main effect for PGS (non-constant basis terms)
        // The PGS main effect is unpenalized intentionally.
        let pgs_main_cols = current_col..current_col + pgs_main_basis_ncols;
        current_col += pgs_main_basis_ncols; // Still advance the column counter

        // Tensor product interaction effects (each gets one penalty since we use whitened marginals)
        let mut interaction_block_idx: Vec<usize> = Vec::with_capacity(config.pc_configs.len());
        if num_pgs_interaction_bases > 0 {
            for (i, &num_pc_basis_unc) in pc_unconstrained_basis_ncols.iter().enumerate() {
                let num_tensor_coeffs = num_pgs_interaction_bases * num_pc_basis_unc;
                let range = current_col..current_col + num_tensor_coeffs;

                penalty_map.push(PenalizedBlock {
                    term_name: format!("f(PGS,{})", config.pc_configs[i].name),
                    col_range: range.clone(),
                    penalty_indices: vec![penalty_idx_counter], // Single penalty since both directions are whitened
                    term_type: TermType::Interaction,
                });

                interaction_block_idx.push(penalty_map.len() - 1);
                current_col += num_tensor_coeffs;
                penalty_idx_counter += 1; // Increment by 1 for single interaction penalty
            }
        }

        // Total number of individual penalties: one per PC main + one per interaction (whitened)
        // _penalty_idx_counter has already been incremented to the correct total

        // Verify that our calculation matches the actual column count
        if current_col != calculated_total_coeffs {
            return Err(EstimationError::LayoutError(format!(
                "ModelLayout dimension calculation error: calculated total_coeffs={calculated_total_coeffs} but actual current_col={current_col}"
            )));
        }

        // Layout invariants: PC index-aligned vectors must match lengths
        if pc_main_block_idx.len() != pc_null_cols.len() {
            return Err(EstimationError::LayoutError(format!(
                "PC layout vectors misaligned: pc_main_block_idx.len()={} vs pc_null_cols.len()={}",
                pc_main_block_idx.len(),
                pc_null_cols.len()
            )));
        }
        if pc_null_block_idx.len() != pc_null_cols.len() {
            return Err(EstimationError::LayoutError(format!(
                "PC layout vectors misaligned: pc_null_block_idx.len()={} vs pc_null_cols.len()={}",
                pc_null_block_idx.len(),
                pc_null_cols.len()
            )));
        }

        Ok(ModelLayout {
            intercept_col,
            pgs_main_cols,
            pc_null_cols,
            pc_null_block_idx,
            penalty_map,
            pc_main_block_idx,
            interaction_block_idx,
            total_coeffs: current_col,
            num_penalties: penalty_idx_counter,
        })
    }
}

/// Constructs the design matrix `X` and a list of individual penalty matrices `S_i`.
/// Returns the design matrix, penalty matrices, model layout, sum-to-zero constraints,
/// knot vectors, range transformations, interaction centering means, and interaction
/// orthogonalization maps (Alpha) for pure-interaction construction.
pub fn build_design_and_penalty_matrices(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<
    (
        Array2<f64>,                  // design matrix
        Vec<Array2<f64>>,             // penalty matrices
        ModelLayout,                  // model layout
        HashMap<String, Array2<f64>>, // sum_to_zero_constraints
        HashMap<String, Array1<f64>>, // knot_vectors
        HashMap<String, Array2<f64>>, // range_transforms
        HashMap<String, Array2<f64>>, // pc_null_transforms
        HashMap<String, Array1<f64>>, // interaction_centering_means
        HashMap<String, Array2<f64>>, // interaction_orth_alpha (per interaction block)
    ),
    EstimationError,
> {
    // PRE-EMPTIVE CHECK: Calculate the potential number of coefficients BEFORE building any matrices
    let n_samples = data.y.len();

    // Calculate coefficients for each term based on the config
    let pgs_basis_coeffs = config.pgs_basis_config.num_knots + config.pgs_basis_config.degree + 1;
    let intercept_coeffs = 1;
    let pgs_main_coeffs = pgs_basis_coeffs - 1;

    let mut pc_main_coeffs = 0;
    let mut interaction_coeffs = 0;
    for pc_config in &config.pc_configs {
        let pc_basis_coeffs = pc_config.basis_config.num_knots + pc_config.basis_config.degree + 1;
        pc_main_coeffs += pc_basis_coeffs - 1;
        interaction_coeffs += (pgs_basis_coeffs - 1) * (pc_basis_coeffs - 1);
    }

    let num_coeffs = intercept_coeffs + pgs_main_coeffs + pc_main_coeffs + interaction_coeffs;

    if num_coeffs > n_samples {
        // FAIL FAST before any expensive calculations
        log::error!(
            "Model is severely over-parameterized: {} coefficients for {} samples",
            num_coeffs,
            n_samples
        );
        return Err(EstimationError::ModelOverparameterized {
            num_coeffs,
            num_samples: n_samples,
            intercept_coeffs,
            pgs_main_coeffs,
            pc_main_coeffs,
            interaction_coeffs,
        });
    }

    // Validate PC configuration against available data
    if config.pc_configs.len() > data.pcs.ncols() {
        let pc_names: Vec<String> = config.pc_configs.iter().map(|pc| pc.name.clone()).collect();
        return Err(EstimationError::InvalidInput(format!(
            "Configuration requests {} Principal Components ({}), but the provided data file only contains {} PC columns.",
            config.pc_configs.len(),
            pc_names.join(", "),
            data.pcs.ncols()
        )));
    }

    let n_samples = data.y.len();

    // Initialize knot vector, sum-to-zero constraint, and range transform storage
    let mut sum_to_zero_constraints = HashMap::new();
    let mut knot_vectors = HashMap::new();
    let mut range_transforms = HashMap::new();
    let mut pc_null_transforms: HashMap<String, Array2<f64>> = HashMap::new();
    let mut interaction_centering_means = HashMap::new();
    let mut interaction_orth_alpha: HashMap<String, Array2<f64>> = HashMap::new();

    // Reserve capacities to avoid rehashing/reallocation
    let n_pcs = config.pc_configs.len();
    // +1 for PGS entries where applicable
    knot_vectors.reserve(n_pcs + 1);
    range_transforms.reserve(n_pcs + 1);
    pc_null_transforms.reserve(n_pcs);
    interaction_centering_means.reserve(n_pcs);
    interaction_orth_alpha.reserve(n_pcs);

    // 1. Generate basis for PGS and apply sum-to-zero constraint
    let (pgs_basis_unc, pgs_knots) = create_bspline_basis(
        data.p.view(),
        config.pgs_range,
        config.pgs_basis_config.num_knots,
        config.pgs_basis_config.degree,
    )?;

    // Save PGS knot vector
    knot_vectors.insert("pgs".to_string(), pgs_knots);

    // Build PGS penalty on FULL basis (not arbitrarily sliced)
    // Note: previous eigen-decomposition and range extraction were unused and removed

    // For PGS main effects: use non-intercept columns and apply sum-to-zero constraint
    let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);
    let (pgs_main_basis, pgs_z_transform) =
        basis::apply_sum_to_zero_constraint(pgs_main_basis_unc, Some(data.weights.view()))?;

    // Create whitened range transform for PGS (used if switching to whitened interactions)
    let s_pgs_main =
        create_difference_penalty_matrix(pgs_main_basis_unc.ncols(), config.penalty_order)?;
    let (_, z_range_pgs) = basis::null_range_whiten(&s_pgs_main)?;

    // Store PGS range transformation for interactions and potential future penalized PGS
    range_transforms.insert("pgs".to_string(), z_range_pgs);

    // Save the PGS sum-to-zero constraint transformation
    sum_to_zero_constraints.insert("pgs_main".to_string(), pgs_z_transform);

    // 2. Generate range-only bases for PCs (functional ANOVA decomposition)
    let mut pc_range_bases: Vec<Array2<f64>> = Vec::with_capacity(n_pcs);
    let mut pc_null_bases: Vec<Option<Array2<f64>>> = Vec::with_capacity(n_pcs);
    let mut pc_unconstrained_bases_main: Vec<Array2<f64>> = Vec::with_capacity(n_pcs);

    // Check if we have any PCs to process
    if config.pc_configs.is_empty() {
        log::info!("No PCs provided; building PGS-only model.");
    }

    for i in 0..config.pc_configs.len() {
        let pc_col = data.pcs.column(i);
        let pc_config = &config.pc_configs[i];
        let pc_name = &pc_config.name;
        let (pc_basis_unc, pc_knots) = create_bspline_basis(
            pc_col.view(),
            pc_config.range,
            pc_config.basis_config.num_knots,
            pc_config.basis_config.degree,
        )?;

        // Save PC knot vector
        knot_vectors.insert(pc_name.clone(), pc_knots);

        // Note: previous eigen-decomposition and range extraction were unused and removed

        // For PC main effects: use non-intercept columns for whitened range basis
        let pc_main_basis_unc = pc_basis_unc.slice(s![.., 1..]);
        pc_unconstrained_bases_main.push(pc_main_basis_unc.to_owned());

        // Create whitened range transform for PC main effects
        let s_pc_main =
            create_difference_penalty_matrix(pc_main_basis_unc.ncols(), config.penalty_order)?;
        let (z_null_pc, z_range_pc) = basis::null_range_whiten(&s_pc_main)?;

        // PC main effect uses ONLY the range (penalized) part
        let pc_range_basis = pc_main_basis_unc.dot(&z_range_pc);
        pc_range_bases.push(pc_range_basis);

        // Build null-space (if any)
        if z_null_pc.ncols() > 0 {
            let pc_null_basis = pc_main_basis_unc.dot(&z_null_pc);
            pc_null_bases.push(Some(pc_null_basis));
            pc_null_transforms.insert(pc_name.clone(), z_null_pc);
        } else {
            pc_null_bases.push(None);
        }

        // Store PC range transformation for interactions and main effects
        range_transforms.insert(pc_name.clone(), z_range_pc);
    }

    // 3. Calculate layout first to determine matrix dimensions
    let pc_range_ncols: Vec<usize> = pc_range_bases.iter().map(|b| b.ncols()).collect();
    let pc_null_ncols: Vec<usize> = pc_null_bases
        .iter()
        .map(|opt| opt.as_ref().map_or(0, |b| b.ncols()))
        .collect();
    let pgs_range_ncols = range_transforms
        .get("pgs")
        .map(|rt| rt.ncols())
        .unwrap_or(0);

    let layout = ModelLayout::new(
        config,
        &pc_null_ncols,
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

        match block.term_type {
            TermType::PcMainEffect => {
                // Main effects have single penalty index. Normalize identity penalty
                // so each block has unit Frobenius norm regardless of size.
                let penalty_idx = block.penalty_indices[0];
                let m = col_range.len() as f64;
                let alpha = if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 };
                for j in col_range.clone() {
                    s_list[penalty_idx][[j, j]] = alpha;
                }
            }
            TermType::Interaction => {
                // Interactions now have single penalty index since both marginals are whitened
                if block.penalty_indices.len() != 1 {
                    return Err(EstimationError::LayoutError(format!(
                        "Interaction block {} should have exactly 1 penalty index, found {}",
                        block.term_name,
                        block.penalty_indices.len()
                    )));
                }

                let penalty_idx = block.penalty_indices[0]; // Single penalty for whitened interaction

                // Normalize identity penalty to unit Frobenius norm for this block
                let m = col_range.len() as f64;
                let alpha = if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 };
                for j in col_range.clone() {
                    s_list[penalty_idx][[j, j]] = alpha;
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

    // 2. Main PC effects: first unpenalized null-space (if any), then penalized range
    for (pc_idx, pc_config) in config.pc_configs.iter().enumerate() {
        let pc_name = &pc_config.name;
        // Fill null-space columns
        let null_cols = &layout.pc_null_cols[pc_idx];
        if null_cols.len() > 0 {
            if let Some(null_basis) = &pc_null_bases[pc_idx] {
                if null_basis.nrows() != n_samples {
                    return Err(EstimationError::LayoutError(format!(
                        "PC null basis {} has {} rows but expected {} samples",
                        pc_name,
                        null_basis.nrows(),
                        n_samples
                    )));
                }
                if null_basis.ncols() != null_cols.len() {
                    return Err(EstimationError::LayoutError(format!(
                        "PC null basis {} has {} columns but layout expects {} columns",
                        pc_name,
                        null_basis.ncols(),
                        null_cols.len()
                    )));
                }
                x_matrix
                    .slice_mut(s![.., null_cols.clone()])
                    .assign(null_basis);
            }
        }
        // Fill penalized range-space columns
        for block in &layout.penalty_map {
            if block.term_name == format!("f({pc_name})") {
                let col_range = block.col_range.clone();
                let pc_basis = &pc_range_bases[pc_idx];

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
        // Use WHITENED marginals for interactions to match main effect scaling
        let z_range_pgs = range_transforms.get("pgs").ok_or_else(|| {
            EstimationError::LayoutError("Missing 'pgs' in range_transforms".to_string())
        })?;
        let pgs_int_basis = pgs_main_basis_unc.dot(z_range_pgs); // Use whitened basis for scale consistency
        // Precompute sqrt(weights) once for this section
        let w_sqrt = data.weights.mapv(f64::sqrt);
        let w_col = w_sqrt.view().insert_axis(Axis(1));

        for (pc_idx, pc_config) in config.pc_configs.iter().enumerate() {
            let pc_name = &pc_config.name;
            // Directly index the corresponding tensor product block
            let tensor_block = &layout.penalty_map[layout.interaction_block_idx[pc_idx]];

            // Use WHITENED PC marginal for scale consistency with main effects
            let z_range_pc = range_transforms.get(pc_name).ok_or_else(|| {
                EstimationError::LayoutError(format!("Missing '{}' in range_transforms", pc_name))
            })?;
            let pc_int_basis = pc_unconstrained_bases_main[pc_idx].dot(z_range_pc);

            let tensor_interaction = row_wise_tensor_product(&pgs_int_basis, &pc_int_basis);

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

            // Build main-effect matrix M = [Intercept | PGS_main | PC_main_for_this_pc (null + range)]
            // Extract intercept and PGS main
            let intercept = x_matrix
                .slice(s![.., layout.intercept_col..layout.intercept_col + 1])
                .to_owned();
            let pgs_main = x_matrix
                .slice(s![.., layout.pgs_main_cols.clone()])
                .to_owned();
            // Extract this PC's main effect columns: null then range
            let pc_block = &layout.penalty_map[layout.pc_main_block_idx[pc_idx]];
            let pc_null_cols = &layout.pc_null_cols[pc_idx];
            let pc_null = if pc_null_cols.len() > 0 {
                Some(x_matrix.slice(s![.., pc_null_cols.clone()]).to_owned())
            } else {
                None
            };
            let pc_range = x_matrix
                .slice(s![.., pc_block.col_range.clone()])
                .to_owned();

            // Preallocate M = [Intercept | PGS_main | PC_null (opt) | PC_range]
            let m_cols =
                1 + pgs_main.ncols() + pc_null.as_ref().map_or(0, |z| z.ncols()) + pc_range.ncols();
            let mut m_matrix = Array2::<f64>::zeros((n_samples, m_cols));
            let mut offset = 0;
            // Intercept
            m_matrix
                .slice_mut(s![.., offset..offset + 1])
                .assign(&intercept);
            offset += 1;
            // PGS_main
            m_matrix
                .slice_mut(s![.., offset..offset + pgs_main.ncols()])
                .assign(&pgs_main);
            offset += pgs_main.ncols();
            // PC_null
            if let Some(pc_n) = pc_null.as_ref() {
                m_matrix
                    .slice_mut(s![.., offset..offset + pc_n.ncols()])
                    .assign(pc_n);
                offset += pc_n.ncols();
            }
            // PC_range
            m_matrix
                .slice_mut(s![.., offset..offset + pc_range.ncols()])
                .assign(&pc_range);

            // Weighted projection: Alpha = (M^T W M)^+ (M^T W T)
            // Reuse precomputed weight column once
            let mw = &m_matrix * &w_col;
            let tw = &tensor_interaction * &w_col;

            // Compute Gram matrix and RHS
            let gram = mw.t().dot(&mw);
            let rhs = mw.t().dot(&tw);

            // Pseudo-inverse via SVD
            let (u_opt, s, vt_opt) = gram
                .svd(true, true)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let (u, vt) = match (u_opt, vt_opt) {
                (Some(u), Some(vt)) => (u, vt),
                _ => {
                    return Err(EstimationError::LayoutError(
                        "SVD did not return U/VT for interaction orthogonalization".to_string(),
                    ));
                }
            };
            let smax = s.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            let tol = smax * 1e-12;
            // Construct Σ^+ as diagonal with thresholding
            let mut s_inv = Array2::zeros((s.len(), s.len()));
            for i in 0..s.len() {
                if s[i] > tol {
                    s_inv[[i, i]] = 1.0 / s[i];
                }
            }
            let gram_pinv = vt.t().dot(&s_inv).dot(&u.t());
            let alpha = gram_pinv.dot(&rhs); // shape: m_cols x t_cols

            // Orthogonalize tensor columns to the space spanned by M (pure interaction)
            let tensor_orth = &tensor_interaction - &m_matrix.dot(&alpha);

            // Save Alpha for use at prediction time
            let interaction_key = format!("f(PGS,{})", pc_name);
            interaction_orth_alpha.insert(interaction_key.clone(), alpha);

            // Store zero means for backward compatibility
            let zeros = Array1::zeros(tensor_orth.ncols());
            interaction_centering_means.insert(interaction_key, zeros);

            // Assign the orthogonalized tensor block to design matrix (no extra centering)
            x_matrix.slice_mut(s![.., col_range]).assign(&tensor_orth);
        }
    }

    // Verify the final number of coefficients matches our pre-check estimate
    let n_samples = data.y.len();
    let n_coeffs = x_matrix.ncols();
    if n_coeffs > n_samples {
        log::warn!(
            "Model is over-parameterized: {} coefficients for {} samples. \
             This should have been caught by the pre-emptive check.",
            n_coeffs,
            n_samples
        );
        // Keep a lightweight condition number check as a backstop
        // This should rarely be reached due to the pre-emptive check
        match calculate_condition_number(&x_matrix) {
            Ok(cond) if cond > 1e9 => {
                log::error!(
                    "Design matrix is severely ill-conditioned (condition number > 1e9). Aborting."
                );
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: cond,
                });
            }
            Ok(_) => {}
            Err(_) => {
                log::error!("SVD failed during condition number check. Aborting.");
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
        }
    }

    Ok((
        x_matrix,
        s_list,
        layout,
        sum_to_zero_constraints,
        knot_vectors,
        range_transforms,
        pc_null_transforms,
        interaction_centering_means,
        interaction_orth_alpha,
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
    /// Transformed penalty square roots rS (each is rank_k x p)
    pub rs_transformed: Vec<Array2<f64>>,
    /// Cached transposes of rS (each is p x rank_k) to avoid repeated transposes in hot paths
    pub rs_transposed: Vec<Array2<f64>>,
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
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;

    // Find the maximum eigenvalue to create a relative tolerance
    let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));

    // Define a relative tolerance. Use an absolute fallback for zero matrices.
    let tolerance = if max_eig > 0.0 {
        max_eig * 1e-12
    } else {
        1e-12
    };

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
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Count positive eigenvalues to determine rank
        // Find the maximum eigenvalue to create a relative tolerance
        let max_eig = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));

        // Define a relative tolerance. Use an absolute fallback for zero matrices.
        let tolerance = if max_eig > 0.0 {
            max_eig * 1e-12
        } else {
            1e-12
        };

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

    // CRITICAL VALIDATION: lambdas length must match number of penalty matrices
    if lambdas.len() != s_list.len() {
        panic!(
            "Lambda count mismatch: expected {} lambdas for {} penalty matrices, got {}",
            s_list.len(),
            s_list.len(),
            lambdas.len()
        );
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

    // CRITICAL VALIDATION: lambdas length must match number of penalties
    if lambdas.len() != m {
        return Err(EstimationError::ParameterConstraintViolation(format!(
            "Lambda count mismatch: expected {} lambdas for {} penalties, got {}",
            m,
            m,
            lambdas.len()
        )));
    }

    if m == 0 {
        return Ok(ReparamResult {
            s_transformed: Array2::zeros((p, p)),
            log_det: 0.0,
            det1: Array1::zeros(0),
            qs: Array2::eye(p),
            rs_transformed: vec![],
            rs_transposed: vec![],
            e_transformed: Array2::zeros((0, p)), // rank x p matrix
        });
    }

    // Wood (2011) Appendix B: get_stableS algorithm
    // Use a tighter tolerance for α/γ partitioning to better match mgcv
    let eps = 1e-7; // robust partitioning tolerance (was 1e-4)
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

        if gamma.is_empty() || q_current == 0 {
            break;
        }

        // Step 1: Find Frobenius norms of penalties in current sub-problem
        // For penalty square roots, we need to form the full penalty matrix S_i = rS_i^T * rS_i
        let mut frob_norms = Vec::new();
        let mut max_omega: f64 = 0.0;

        for &i in &gamma {
            // Extract active columns from penalty square root (rank x p convention)
            let rs_active_cols = rs_current[i].slice(s![.., k_offset..k_offset + q_current]);

            // Skip if penalty has no rows (zero rank penalty)
            if rs_current[i].nrows() == 0 || q_current == 0 {
                frob_norms.push((i, 0.0));
                continue;
            }

            // Form the active sub-block of full penalty matrix S_i = rS_i^T * rS_i
            let s_active_block = rs_active_cols.t().dot(&rs_active_cols);

            // The Frobenius norm is the sqrt of sum of squares of matrix elements
            let frob_norm = s_active_block.iter().map(|&x| x * x).sum::<f64>().sqrt();
            // Weight by lambda so the transform reflects current smoothing regime
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
            println!("No terms in alpha set. Terminating.");
            break;
        }

        println!(
            "Partitioned: alpha set = {:?}, gamma_prime set = {:?}",
            alpha, gamma_prime
        );

        // println!("DEBUG: Partitioned: alpha set = {:?}, gamma_prime set = {:?}", alpha, gamma_prime);

        // Step 3a: Form SCALED sum for STABLE RANK DETECTION (lambda-INDEPENDENT)
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
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // Determine rank 'r' using these stable eigenvalues
        let max_eigenval = eigenvalues_for_rank
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let rank_tolerance = max_eigenval * r_tol;
        let mut r = eigenvalues_for_rank
            .iter()
            .filter(|&&ev| ev > rank_tolerance)
            .count();

        println!(
            "Stable rank detection: found rank {} from {} eigenvalues (max_eig: {}, tol: {})",
            r,
            eigenvalues_for_rank.len(),
            max_eigenval,
            rank_tolerance
        );

        // Correct energy capture check using eigenvalues of sb_for_rank (basis-invariant)
        if r > 1 {
            let positive_eigenvalues: Vec<f64> = eigenvalues_for_rank
                .iter()
                .filter(|&&e| e > rank_tolerance)
                .copied()
                .collect();

            if !positive_eigenvalues.is_empty() {
                let total_energy: f64 = positive_eigenvalues.iter().sum();
                let top_r_eigenvalues: Vec<f64> = positive_eigenvalues
                    .iter()
                    .rev() // Largest first (eigenvalues are in ascending order)
                    .take(r)
                    .copied()
                    .collect();
                let captured_energy: f64 = top_r_eigenvalues.iter().sum();

                let captured_energy_ratio = if total_energy > 1e-12 {
                    captured_energy / total_energy
                } else {
                    1.0
                };

                // Guardrail: if capture falls suspiciously low, log a warning and proceed conservatively
                if captured_energy_ratio < 0.95 {
                    log::warn!(
                        "Energy capture ratio low: {:.3}. Proceeding with conservative rank {}.",
                        captured_energy_ratio,
                        r
                    );
                }
            }
        }

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
        let (eigenvalues_for_transform, u): (Array1<f64>, Array2<f64>) = sb_for_transform
            .eigh(Side::Lower)
            .map_err(EstimationError::EigendecompositionFailed)?;

        // SAFETY CHECK: Validate that the two decompositions agree on the rank
        // If the lambda-weighted matrix has significantly different rank structure,
        // the reparameterization may be unreliable
        let max_eigenval_transform = eigenvalues_for_transform
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let rank_tolerance_transform = max_eigenval_transform * r_tol;
        let r_transform = eigenvalues_for_transform
            .iter()
            .filter(|&&ev| ev > rank_tolerance_transform)
            .count();

        // Check for significant disagreement between the two rank estimates
        if (r as i32 - r_transform as i32).abs() > 1 {
            log::warn!(
                "Rank disagreement detected: balanced matrix rank={}, weighted matrix rank={}. Proceeding with caution.",
                r,
                r_transform
            );

            // Fall back to more conservative rank estimate to avoid corruption
            let r_conservative = r.min(r_transform);
            if r_conservative == 0 {
                gamma = gamma_prime;
                continue;
            }
            // Use the conservative rank for the remainder of this iteration
            r = r_conservative;
        }

        // Note: The stable rank detection debug message is already logged above

        println!(
            "Rank detection: r={}, q_current={}, iteration={}",
            r, q_current, iteration
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
            if rs_current[i].nrows() == 0 || q_current == 0 {
                continue;
            }

            // For rank×p penalty roots, transform as R_new = R * U (not U^T * R)
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
            if rs_current[i].nrows() == 0 || q_current == 0 {
                continue;
            }

            if alpha.contains(&i) {
                // DOMINANT penalty: Its effect is now entirely within the range space.
                // For rank×p roots, zero out the null space COLUMNS (not rows)
                // The null space is now the LAST `q_current - r` columns of the sub-block.
                if r < q_current {
                    // Use explicit end index to avoid zeroing beyond current subblock
                    rs_current[i]
                        .slice_mut(s![.., k_offset + r..k_offset + q_current])
                        .fill(0.0);
                }
            } else {
                // SUB-DOMINANT penalty (in gamma_prime).
                // Its effect is carried forward in the null space.
                // For rank×p roots, zero out the range space COLUMNS (not rows)
                // The range space is now the FIRST `r` columns of the sub-block.
                rs_current[i]
                    .slice_mut(s![.., k_offset..k_offset + r])
                    .fill(0.0);
            }
        }

        // Apply the same zeroing to the full S matrices.
        // This prevents dominant penalty information from contaminating the next iteration's
        // basis calculation (the cause of the numerical instability).
        // Apply the same zeroing to the full S matrices to enforce exact projector structure
        for &i in &gamma {
            if alpha.contains(&i) {
                // DOMINANT penalty: Zero out its null-space block.
                if r < q_current {
                    // Zero out the null-space rows and columns (bottom-right block)
                    s_current_list[i]
                        .slice_mut(s![
                            k_offset + r..k_offset + q_current,
                            k_offset + r..k_offset + q_current
                        ])
                        .fill(0.0);
                    // Zero out the off-diagonal blocks connecting range and null spaces
                    s_current_list[i]
                        .slice_mut(s![
                            k_offset..k_offset + r,
                            k_offset + r..k_offset + q_current
                        ])
                        .fill(0.0);
                    s_current_list[i]
                        .slice_mut(s![
                            k_offset + r..k_offset + q_current,
                            k_offset..k_offset + r
                        ])
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
                    .slice_mut(s![
                        k_offset..k_offset + r,
                        k_offset + r..k_offset + q_current
                    ])
                    .fill(0.0);
                s_current_list[i]
                    .slice_mut(s![
                        k_offset + r..k_offset + q_current,
                        k_offset..k_offset + r
                    ])
                    .fill(0.0);
            }
        }

        // Update for next iteration
        // Update iteration variables for next loop according to mgcv
        k_offset += r; // Increase offset by the rank we processed
        q_current -= r; // Reduce problem size by the rank we processed
        gamma = gamma_prime; // Continue with the subdominant penalties

    }

    println!(
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

    // Step 3: Compute eigendecomposition of S_lambda
    let (s_eigenvalues_raw, s_eigenvectors): (Array1<f64>, Array2<f64>) = s_transformed
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    // Use a small constant ridge epsilon for smoothed logdet/inverse; keep it lambda-independent
    // keep for possible future smoothing: not used when using pseudo-determinant
    // let _ridge_eps: f64 = 1e-8;

    // Determine effective rank (for e_transformed shape) using raw spectrum and relative tolerance
    let max_eigenval = s_eigenvalues_raw
        .iter()
        .fold(0.0_f64, |a, &b| a.max(b.abs()));
    let tolerance = max_eigenval * 1e-12;
    let penalty_rank = s_eigenvalues_raw
        .iter()
        .filter(|&&ev| ev > tolerance)
        .count()
        .max(0);

    // Construct the lambda-DEPENDENT penalty square root matrix from the RAW spectrum
    let mut e_matrix = Array2::zeros((p, penalty_rank));
    let mut col_idx = 0;
    for (i, &eigenval) in s_eigenvalues_raw.iter().enumerate() {
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

    // Step 4: Calculate log-pseudo-determinant from the positive eigenvalues
    let log_det: f64 = s_eigenvalues_raw
        .iter()
        .filter(|&&ev| ev > tolerance)
        .map(|&ev| ev.ln())
        .sum();

    // Step 5: Calculate derivatives using the correct transformed matrices
    let mut det1 = Array1::zeros(lambdas.len());

    // Compute pseudo-inverse S_lambda^+ using only eigenvalues above tolerance
    let mut s_plus = Array2::zeros((p, p));
    for (i, &eigenval) in s_eigenvalues_raw.iter().enumerate() {
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
        rs_transformed: final_rs_transformed.clone(),
        rs_transposed: final_rs_transformed
            .iter()
            .map(|m| m.t().to_owned())
            .collect(),
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
pub fn calculate_condition_number(matrix: &Array2<f64>) -> Result<f64, FaerLinalgError> {
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

        let pc_ranges: Vec<_> = (0..num_pcs)
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

        // Create PrincipalComponentConfig objects for each PC
        let pc_configs: Vec<_> = (0..num_pcs)
            .map(|i| {
                let pc_range = pc_ranges[i];
                let pc_basis_config = BasisConfig {
                    num_knots: 4,
                    degree: 3,
                };
                let pc_name = format!("PC{}", i + 1);
                crate::calibrate::model::PrincipalComponentConfig {
                    name: pc_name,
                    basis_config: pc_basis_config,
                    range: pc_range,
                }
            })
            .collect();

        let config = ModelConfig {
            link_function: LinkFunction::Identity,
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 100,
            pgs_basis_config,
            pc_configs,
            pgs_range,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
        };

        (data, config)
    }

    #[test]
    fn test_interaction_design_matrix_is_full_rank() {
        let (data, config) = create_test_data_for_construction(100, 1);
        let (x, ..) = build_design_and_penalty_matrices(&data, &config).unwrap();

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
        let (_, s_list, layout, _, _, _, _, _, _) =
            build_design_and_penalty_matrices(&data, &config).unwrap();

        // With null-space penalization: 2 PC main penalties (null + range) + 1 interaction penalty = 3 total
        assert_eq!(
            s_list.len(),
            3,
            "Should have exactly 3 penalty matrices: 1 PC null + 1 PC range + 1 interaction (whitened)"
        );

        let interaction_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::Interaction)
            .expect("Interaction block not found in layout");

        // Each interaction block now has 1 penalty index since marginals are whitened
        assert_eq!(
            interaction_block.penalty_indices.len(),
            1,
            "Interaction term should have one penalty index (whitened marginals)"
        );

        let interaction_penalty_idx = interaction_block.penalty_indices[0];

        // Verify penalty matrix structure - single identity (Frobenius-normalized) penalty for whitened interaction
        let s_interaction = &s_list[interaction_penalty_idx]; // Single interaction penalty
        let alpha_interaction = {
            let m = interaction_block.col_range.len() as f64;
            if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 }
        };

        // Locate explicit PC range and PC null penalty blocks
        let pc_range_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::PcMainEffect && b.term_name == "f(PC1)")
            .expect("PC range main-effect block not found in layout");
        let pc_null_block = layout
            .penalty_map
            .iter()
            .find(|b| b.term_type == TermType::PcMainEffect && b.term_name == "f(PC1)_null")
            .expect("PC null-space block not found in layout");

        let s_pc_range = &s_list[pc_range_block.penalty_indices[0]];
        let s_pc_null = &s_list[pc_null_block.penalty_indices[0]];

        // Check that the interaction penalty is identity on the interaction block
        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if interaction_block.col_range.contains(&r)
                    && interaction_block.col_range.contains(&c)
                {
                    if r == c {
                        assert_abs_diff_eq!(
                            s_interaction[[r, c]],
                            alpha_interaction,
                            epsilon = 1e-12
                        );
                    } else {
                        assert_abs_diff_eq!(s_interaction[[r, c]], 0.0, epsilon = 1e-12);
                    }
                } else {
                    // Outside interaction block should be zero
                    assert_abs_diff_eq!(s_interaction[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }

        // Check that PC main penalty matrix has identity on PC main blocks and zeros elsewhere
        // Check range block identity structure
        let alpha_pc_range = {
            let m = pc_range_block.col_range.len() as f64;
            if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 }
        };
        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if pc_range_block.col_range.contains(&r) && pc_range_block.col_range.contains(&c) {
                    if r == c {
                        assert_abs_diff_eq!(s_pc_range[[r, c]], alpha_pc_range, epsilon = 1e-12);
                    } else {
                        assert_abs_diff_eq!(s_pc_range[[r, c]], 0.0, epsilon = 1e-12);
                    }
                } else {
                    assert_abs_diff_eq!(s_pc_range[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }

        // Check null block identity structure
        let alpha_pc_null = {
            let m = pc_null_block.col_range.len() as f64;
            if m > 0.0 { 1.0 / m.sqrt() } else { 1.0 }
        };
        for r in 0..layout.total_coeffs {
            for c in 0..layout.total_coeffs {
                if pc_null_block.col_range.contains(&r) && pc_null_block.col_range.contains(&c) {
                    if r == c {
                        assert_abs_diff_eq!(s_pc_null[[r, c]], alpha_pc_null, epsilon = 1e-12);
                    } else {
                        assert_abs_diff_eq!(s_pc_null[[r, c]], 0.0, epsilon = 1e-12);
                    }
                } else {
                    assert_abs_diff_eq!(s_pc_null[[r, c]], 0.0, epsilon = 1e-12);
                }
            }
        }
    }

    #[test]
    fn test_construction_with_no_pcs() {
        let (data, config) = create_test_data_for_construction(100, 0); // 0 PCs

        let (_, _, layout, _, _, _, _, _, _) =
            build_design_and_penalty_matrices(&data, &config).unwrap();

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

    #[test]
    fn test_predict_linear_equals_x_beta() {
        // Build training matrices and transforms
        let (data, config) = create_test_data_for_construction(100, 1);
        // Capture interaction_orth_alpha from the return tuple
        let (
            x_training,
            _,
            _,
            sum_to_zero_constraints,
            knot_vectors,
            range_transforms,
            pc_null_transforms,
            interaction_centering_means,
            interaction_orth_alpha,
        ) = build_design_and_penalty_matrices(&data, &config).unwrap();

        // Prepare a config carrying all saved transforms
        let mut cfg = config.clone();
        cfg.sum_to_zero_constraints = sum_to_zero_constraints.clone();
        cfg.knot_vectors = knot_vectors.clone();
        cfg.range_transforms = range_transforms.clone();
        cfg.pc_null_transforms = pc_null_transforms.clone();
        cfg.interaction_centering_means = interaction_centering_means.clone();
        cfg.interaction_orth_alpha = interaction_orth_alpha.clone(); // Add the captured map to the configuration

        // Construct coefficients in the canonical structure
        use crate::calibrate::model::{MainEffects, MappedCoefficients, TrainedModel};

        // For this config, there is 1 PC (if you built with num_pcs=1 above).
        // Determine sizes from cfg.range_transforms and sum_to_zero_constraints.
        let pgs_dim = cfg
            .sum_to_zero_constraints
            .get("pgs_main")
            .expect("missing pgs_main Z")
            .ncols();
        let pc_name = cfg.pc_configs[0].name.clone();
        let r_pc = cfg
            .range_transforms
            .get(&pc_name)
            .expect("missing PC range transform")
            .ncols();
        let n_pc = cfg
            .pc_null_transforms
            .get(&pc_name)
            .map(|z| z.ncols())
            .unwrap_or(0);
        let r_pgs = cfg
            .range_transforms
            .get("pgs")
            .expect("missing PGS range transform")
            .ncols();

        let interaction_len = r_pgs * r_pc;

        // Populate coefficients deterministically
        let coeffs = MappedCoefficients {
            intercept: 0.123,
            main_effects: MainEffects {
                pgs: (1..=pgs_dim).map(|i| i as f64).collect(),
                pcs: {
                    let mut pcs = HashMap::new();
                    let mut v: Vec<f64> = Vec::new();
                    for i in 1..=n_pc {
                        v.push(10.0 + i as f64);
                    }
                    for i in 1..=r_pc {
                        v.push(20.0 + i as f64);
                    }
                    pcs.insert(pc_name.clone(), v);
                    pcs
                },
            },
            interaction_effects: {
                let mut m = HashMap::new();
                m.insert(
                    format!("f(PGS,{})", pc_name),
                    (1..=interaction_len).map(|i| 100.0 + i as f64).collect(),
                );
                m
            },
        };

        // Manually flatten coefficients using the same canonical order as model::internal::flatten_coefficients
        let mut flat: Vec<f64> = Vec::new();
        flat.push(coeffs.intercept);
        for pc_config in &cfg.pc_configs {
            let pc = &pc_config.name;
            flat.extend_from_slice(
                coeffs
                    .main_effects
                    .pcs
                    .get(pc)
                    .expect("missing PC main coeffs"),
            );
        }
        flat.extend_from_slice(&coeffs.main_effects.pgs);
        for pc_config in &cfg.pc_configs {
            let pc = &pc_config.name;
            let key = format!("f(PGS,{})", pc);
            if let Some(v) = coeffs.interaction_effects.get(&key) {
                flat.extend_from_slice(v);
            }
        }
        let flat = ndarray::Array1::from(flat);

        // Build a trained model with this config and coefficients
        let model = TrainedModel {
            config: cfg,
            coefficients: coeffs.clone(),
            lambdas: vec![], // not used at prediction
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
        };

        // Compute predictions via predict_linear() (which rebuilds X_new internally)
        let preds_via_predict = model
            .predict_linear(data.p.view(), data.pcs.view())
            .expect("predict_linear failed");

        // Compute predictions directly from X_training (should match)
        let preds_direct = x_training.dot(&flat);

        // Compare
        let max_diff = (&preds_via_predict - &preds_direct)
            .mapv(|x| x.abs())
            .iter()
            .fold(0.0f64, |acc, &x| acc.max(x));
        assert!(
            max_diff < 1e-10,
            "Training and prediction paths must be consistent; max |diff| = {}",
            max_diff
        );
    }
}
