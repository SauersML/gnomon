use crate::calibrate::basis::{self};
use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::hull::PeeledHull;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use thiserror::Error;

// Global toggle for the optional post-process calibrator layer.
// Enabled by default. Wire this to a CLI flag like `--no-calibrate` by
// calling `model::set_calibrator_enabled(false)` before training/prediction.
use std::sync::atomic::{AtomicBool, Ordering};
static CALIBRATOR_ENABLED: AtomicBool = AtomicBool::new(true);
pub fn set_calibrator_enabled(enabled: bool) {
    CALIBRATOR_ENABLED.store(enabled, Ordering::SeqCst);
}
pub fn calibrator_enabled() -> bool {
    CALIBRATOR_ENABLED.load(Ordering::SeqCst)
}

// --- Public Data Structures ---
// These structs define the public, human-readable format of the trained model
// when serialized to a TOML file.

/// Defines the link function, connecting the linear predictor to the mean response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkFunction {
    /// The logit link, for binary or proportional outcomes (e.g., logistic regression).
    /// Maps probabilities (0, 1) to the real line (-inf, +inf).
    Logit,
    /// The identity link, for continuous outcomes (e.g., Gaussian regression).
    Identity,
}

/// Configuration toggle for tensor-product interaction penalties.
///
/// * `Isotropic` matches the historical behavior with a single penalty parameter
///   using whitened marginal bases.
/// * `Anisotropic` builds separate penalties for each marginal direction using
///   the unwhitened bases. This is the new default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionPenaltyKind {
    Isotropic,
    Anisotropic,
}

/// Configuration for a single basis expansion (e.g., for one variable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisConfig {
    pub num_knots: usize,
    pub degree: usize,
}

/// Configuration for a single Principal Component, bundling related data together.
/// This makes invalid states unrepresentable by ensuring each PC has all required
/// configuration elements together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalComponentConfig {
    /// Name of this principal component (e.g., "PC1")
    pub name: String,
    /// Basis configuration for this principal component
    pub basis_config: BasisConfig,
    /// Value range for this principal component
    pub range: (f64, f64),
}

/// Holds the transformation matrix for a sum-to-zero constraint.
/// This is serializable so it can be saved to the TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// The Z matrix that transforms an unconstrained basis B to a constrained one B_c = B.dot(Z)
    pub z_transform: Array2<f64>,
}

/// The complete blueprint of a trained model.
/// Contains all hyperparameters and structural information needed for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub link_function: LinkFunction,
    pub penalty_order: usize,
    pub convergence_tolerance: f64,
    pub max_iterations: usize,
    pub reml_convergence_tolerance: f64,
    pub reml_max_iterations: u64,
    pub pgs_basis_config: BasisConfig,
    /// Bundled configuration for each principal component.
    /// This ensures that for each PC, we have the correct basis config, name, and range together.
    pub pc_configs: Vec<PrincipalComponentConfig>,
    // Data-dependent parameters saved from training are crucial for prediction.
    pub pgs_range: (f64, f64),
    pub interaction_penalty: InteractionPenaltyKind,

    /// Sum-to-zero constraint transformations (separate from range transforms)
    /// Maps term names (e.g., "pgs_main") to their sum-to-zero transformation matrices
    pub sum_to_zero_constraints: HashMap<String, Array2<f64>>,
    /// Knot vectors used during training, required for exact reproduction during prediction
    pub knot_vectors: HashMap<String, Array1<f64>>,
    /// Range transformation matrices for functional ANOVA decomposition
    /// Maps variable names (e.g., "pgs", "PC1") to their range-space transformation matrices
    pub range_transforms: HashMap<String, Array2<f64>>,
    /// Weighted column means used for centering interaction marginals during training
    /// Maps variable names (e.g., "pgs", "PC1") to their weighted column means for ANOVA centering
    pub interaction_centering_means: HashMap<String, Array1<f64>>,
    /// Linear maps that orthogonalize each interaction block against the main effects (pure ti())
    /// Keyed by interaction term name (e.g., "f(PGS,PC1)"). Shape: m_cols x t_cols.
    pub interaction_orth_alpha: HashMap<String, Array2<f64>>,
    /// Null-space transformation matrices for PC main effects (unpenalized part).
    /// Maps PC name to Z_null (columns span the penalty null space after dropping intercept).
    pub pc_null_transforms: HashMap<String, Array2<f64>>,
}

impl ModelConfig {
    /// Minimal configuration for external designs (calibrator adapter).
    /// Only the fields used by PIRLS/REML are populated; others are left empty.
    pub fn external(link: LinkFunction, reml_tol: f64, reml_max_iter: usize) -> Self {
        ModelConfig {
            link_function: link,
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 50,
            reml_convergence_tolerance: reml_tol,
            reml_max_iterations: reml_max_iter as u64,
            pgs_basis_config: BasisConfig {
                num_knots: 0,
                degree: 0,
            },
            pc_configs: Vec::new(),
            pgs_range: (0.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
            pc_null_transforms: HashMap::new(),
        }
    }
}

/// A structured representation of the fitted model coefficients, designed for
/// human interpretation and sharing. This structure is used in the TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedCoefficients {
    pub intercept: f64,
    pub main_effects: MainEffects,
    /// Flat map for tensor product interaction terms.
    /// - Key: Interaction term name (e.g., "f(PGS,PC1)").
    /// - Value: The vector of coefficients for the entire tensor product surface.
    pub interaction_effects: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MainEffects {
    /// Coefficients for the main effect of PGS (for basis functions m > 0).
    pub pgs: Vec<f64>,
    /// Coefficients for the main effects of each PC, keyed by PC name.
    pub pcs: HashMap<String, Vec<f64>>,
}

/// The top-level, self-contained, trained model artifact.
/// This is the structure that gets saved to and loaded from a file.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainedModel {
    pub config: ModelConfig,
    pub coefficients: MappedCoefficients,
    /// Estimated smoothing parameters from REML
    pub lambdas: Vec<f64>,
    /// Robust geometric clamping hull (optional for backwards compatibility)
    #[serde(default)]
    pub hull: Option<PeeledHull>,
    /// Optional penalized Hessian (X'WX + S) at convergence in the model's coefficient order.
    /// When present, enables standard error estimates at prediction time via var = x^T H^{-1} x.
    #[serde(default)]
    pub penalized_hessian: Option<Array2<f64>>,
    /// Optional scale parameter (Gaussian identity link). Used for mean SE scaling when applicable.
    #[serde(default)]
    pub scale: Option<f64>,
    /// Optional post-fit calibrator model
    #[serde(default)]
    pub calibrator: Option<crate::calibrate::calibrator::CalibratorModel>,
}

/// Custom error type for model loading, saving, and prediction.
#[derive(Error, Debug)]
pub enum ModelError {
    #[error("Failed to read or write model file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Failed to parse TOML model file: {0}")]
    TomlParseError(#[from] toml::de::Error),
    #[error("Failed to serialize model to TOML format: {0}")]
    TomlSerializeError(#[from] toml::ser::Error),
    #[error("Prediction data has {found} PC columns, but the model was trained on {expected}.")]
    MismatchedPcCount { found: usize, expected: usize },
    #[error("Underlying basis function generation failed during prediction: {0}")]
    BasisError(#[from] basis::BasisError),
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error(
        "Internal error: failed to stack design matrix columns or constraint matrix dimensions don't match basis dimensions during prediction."
    )]
    InternalStackingError,
    #[error(
        "Constraint transformation matrix missing for term '{0}'. This usually indicates a model format mismatch."
    )]
    ConstraintMissing(String),
    #[error("Coefficient vector for term '{0}' is missing from the model file.")]
    CoefficientMissing(String),

    #[error(
        "Calibrated prediction requested but no calibrator is present (disabled or missing from model)."
    )]
    CalibratorMissing,

    #[error("Estimation error: {0}")]
    EstimationError(#[from] crate::calibrate::estimate::EstimationError),
}

impl TrainedModel {
    /// Detailed predictions including linear predictor, mean response, signed distance
    /// to the peeled hull boundary (negative inside), and optional SEs for eta.
    pub fn predict_detailed(
        &self,
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<
        (
            Array1<f64>,         // eta (linear predictor)
            Array1<f64>,         // mean (after inverse link)
            Array1<f64>,         // signed distance to peeled hull (negative inside)
            Option<Array1<f64>>, // se_eta if available
        ),
        ModelError,
    > {
        // --- Validate inputs ---
        if pcs_new.ncols() != self.config.pc_configs.len() {
            return Err(ModelError::MismatchedPcCount {
                found: pcs_new.ncols(),
                expected: self.config.pc_configs.len(),
            });
        }

        // --- Geometry: compute signed distance and projection in one pass ---
        let raw = internal::assemble_raw_from_p_and_pcs(p_new, pcs_new);
        let (signed_dist, x_corr) = if let Some(hull) = &self.hull {
            hull.signed_distance_and_project_many(raw.view())
        } else {
            (Array1::zeros(raw.nrows()), raw.clone())
        };
        let (p_corr, pcs_corr) = internal::split_p_and_pcs_from_raw(x_corr.view());

        // --- Build design and coefficients ---
        let x_new = internal::construct_design_matrix(
            p_corr.view(),
            pcs_corr.view(),
            &self.config,
            &self.coefficients,
        )?;
        let beta = internal::flatten_coefficients(&self.coefficients, &self.config)?;
        if x_new.ncols() != beta.len() {
            return Err(ModelError::InternalStackingError);
        }

        // --- Linear predictor and mean ---
        let eta = x_new.dot(&beta);
        let mean = match self.config.link_function {
            LinkFunction::Logit => {
                let eta_clamped = eta.mapv(|e| e.clamp(-700.0, 700.0));
                let mut probs = eta_clamped.mapv(|e| 1.0 / (1.0 + f64::exp(-e)));
                probs.mapv_inplace(|p| p.clamp(1e-8, 1.0 - 1e-8));
                probs
            }
            LinkFunction::Identity => eta.clone(),
        };

        // --- Optional SE for eta using the penalized Hessian ---
        let se_eta_opt = if let Some(h) = &self.penalized_hessian {
            if h.nrows() != h.ncols() || h.ncols() != x_new.ncols() {
                None
            } else {
                use crate::calibrate::faer_ndarray::FaerCholesky;
                use faer::Side;
                let chol = match h.clone().cholesky(Side::Lower) {
                    Ok(c) => c,
                    Err(_) => return Ok((eta, mean, signed_dist, None)),
                };
                let mut vars = Array1::zeros(x_new.nrows());
                for i in 0..x_new.nrows() {
                    let x_row = x_new.row(i).to_owned();
                    // Solve H v = x^T for v (1D)
                    let v = chol.solve_vec(&x_row);
                    let var_i = x_row.dot(&v);
                    vars[i] = if self.config.link_function == LinkFunction::Identity {
                        // For Gaussian identity, scale variance if scale present
                        if let Some(scale) = self.scale {
                            var_i * scale
                        } else {
                            var_i
                        }
                    } else {
                        var_i
                    };
                }
                Some(vars.mapv(|v| v.max(0.0).sqrt()))
            }
        } else {
            None
        };

        Ok((eta, mean, signed_dist, se_eta_opt))
    }
    /// Predicts outcomes for new individuals using the trained model.
    ///
    /// This is the core inference engine. It is a fast, non-iterative process that:
    /// - Reconstructs the mathematical model (design matrix and coefficient vector)
    ///   from the stored configuration.
    /// - Computes the final prediction via matrix algebra.
    ///
    /// # Arguments
    /// * `p_new`: A 1D array view of new PGS values.
    /// * `pcs_new`: A 2D array view of new PC values, with shape `[n_samples, n_pcs]`.
    ///              The order of PC columns must match `config.pc_configs` names.
    ///
    /// # Returns
    /// A `Result` containing an `Array1<f64>` of predicted outcomes (e.g., probabilities
    /// or continuous values), or a `ModelError`.
    pub fn predict(
        &self,
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        // Restore original behavior via detailed path: PHC -> design rebuild -> inverse link
        let (_, mean, _, _) = self.predict_detailed(p_new, pcs_new)?;
        Ok(mean)
    }

    /// Predicts outcomes applying the optional post-process calibrator.
    /// Baseline predictions are computed first, then the calibrator adjusts them.
    pub fn predict_calibrated(
        &self,
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        // Stage: Compute baseline predictions
        let baseline = self.predict(p_new, pcs_new)?;
        // Stage: If no calibrator is present, error loudly (no silent fallback)
        if self.calibrator.is_none() {
            return Err(ModelError::CalibratorMissing);
        }

        // Stage: Retrieve eta, signed distance, and se(eta) via the detailed path
        let (eta, _, signed_dist, se_eta_opt) = self.predict_detailed(p_new, pcs_new)?;
        let cal = self.calibrator.as_ref().unwrap();
        let pred_in = match self.config.link_function {
            LinkFunction::Logit => eta.clone(),
            LinkFunction::Identity => baseline.clone(),
        };
        let se_in = se_eta_opt.unwrap_or_else(|| Array1::zeros(pred_in.len()));

        let preds = crate::calibrate::calibrator::predict_calibrator(
            cal,
            pred_in.view(),
            se_in.view(),
            signed_dist.view(),
        )?;
        Ok(preds)
    }

    /// Returns the linear predictor (η = Xβ) for new data without applying the inverse link.
    ///
    /// This is useful for diagnostics and tests that want to validate design-matrix
    /// consistency (e.g., checking that training-time `X.dot(beta)` equals prediction-time
    /// reconstruction). For logistic models, prefer `predict` for probabilities.
    pub fn predict_linear(
        &self,
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        if pcs_new.ncols() != self.config.pc_configs.len() {
            return Err(ModelError::MismatchedPcCount {
                found: pcs_new.ncols(),
                expected: self.config.pc_configs.len(),
            });
        }

        // Assemble raw predictors, optionally project via PHC, and split back
        let raw = internal::assemble_raw_from_p_and_pcs(p_new, pcs_new);
        let (x_corr, num_projected) = if let Some(hull) = &self.hull {
            hull.project_if_needed(raw.view())
        } else {
            (raw, 0)
        };
        if x_corr.nrows() > 0 && num_projected > 0 {
            let rate = 100.0 * (num_projected as f64) / (x_corr.nrows() as f64);
            println!(
                "[PHC] Projected {} of {} points ({:.1}%).",
                num_projected,
                x_corr.nrows(),
                rate
            );
        }
        let (p_corr, pcs_corr) = internal::split_p_and_pcs_from_raw(x_corr.view());

        let x_new = internal::construct_design_matrix(
            p_corr.view(),
            pcs_corr.view(),
            &self.config,
            &self.coefficients,
        )?;
        let flattened_coeffs = internal::flatten_coefficients(&self.coefficients, &self.config)?;

        if x_new.ncols() != flattened_coeffs.len() {
            return Err(ModelError::InternalStackingError);
        }

        Ok(x_new.dot(&flattened_coeffs))
    }

    /// Saves the trained model to a file in a human-readable TOML format.
    pub fn save(&self, path: &str) -> Result<(), ModelError> {
        let toml_string = toml::to_string_pretty(self)?;
        let mut file = BufWriter::new(fs::File::create(path)?);
        file.write_all(toml_string.as_bytes())?;
        Ok(())
    }

    /// Loads a trained model from a TOML file.
    pub fn load(path: &str) -> Result<Self, ModelError> {
        let toml_string = fs::read_to_string(path)?;
        let model = toml::from_str(&toml_string)?;
        Ok(model)
    }
}

/// Internal module for prediction-specific implementation details.
mod internal {
    use super::*;
    // no extra imports

    /// Assemble raw predictors into an n x d matrix [PGS | PC1 | PC2 | ...]
    pub(super) fn assemble_raw_from_p_and_pcs(
        p: ArrayView1<f64>,
        pcs: ArrayView2<f64>,
    ) -> Array2<f64> {
        let n = p.len();
        let d = 1 + pcs.ncols();
        let mut x = Array2::zeros((n, d));
        // first column is p
        x.column_mut(0).assign(&p);
        // remaining are pcs
        if pcs.ncols() > 0 {
            x.slice_mut(s![.., 1..]).assign(&pcs);
        }
        x
    }

    /// Split raw matrix [PGS | PCs...] into (p, pcs)
    pub(super) fn split_p_and_pcs_from_raw(x: ArrayView2<f64>) -> (Array1<f64>, Array2<f64>) {
        let n = x.nrows();
        let d = x.ncols();
        let mut p = Array1::zeros(n);
        p.assign(&x.column(0));
        let pcs = if d > 1 {
            x.slice(s![.., 1..]).to_owned()
        } else {
            Array2::zeros((n, 0))
        };
        (p, pcs)
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

    /// Constructs the design matrix `X` for new data following a strict canonical order.
    /// This order is the implicit contract that allows the flattened coefficients to work correctly.
    pub(super) fn construct_design_matrix(
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        config: &ModelConfig,
        coeffs: &MappedCoefficients,
    ) -> Result<Array2<f64>, ModelError> {
        // CRITICAL: Validate that prediction data dimensions are consistent
        if p_new.len() != pcs_new.nrows() {
            return Err(ModelError::DimensionMismatch(format!(
                "Sample count mismatch: p_new has {} samples but pcs_new has {} rows",
                p_new.len(),
                pcs_new.nrows()
            )));
        }
        // Stage: Generate the PGS basis using the saved knot vector if available
        // Only use saved knot vectors - remove fallback to ensure consistency
        let saved_knots = config
            .knot_vectors
            .get("pgs")
            .ok_or_else(|| ModelError::InternalStackingError)?;

        let (pgs_basis_unc, _) = basis::create_bspline_basis_with_knots(
            p_new,
            saved_knots.view(),
            config.pgs_basis_config.degree,
        )?;

        // Build PGS main basis to MATCH TRAINING exactly.
        // Start from the unconstrained PGS basis (drop intercept col).
        let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);

        // Require sum-to-zero constraint for PGS main effect
        let pgs_z = config
            .sum_to_zero_constraints
            .get("pgs_main")
            .ok_or_else(|| ModelError::ConstraintMissing("pgs_main".to_string()))?;

        // Note: z_range_pgs is still required for interactions later, but not needed for main effects

        // Target width must match trained coefficient count
        let pgs_coef_len = coeffs.main_effects.pgs.len();

        // CRITICAL: For PGS main effect, we use ONLY the sum-to-zero constraint transform.
        // The range transform (z_range_pgs) is used ONLY for interaction terms.

        // Check shapes: B_unc (N x m), Z (m x c)
        let m = pgs_main_basis_unc.ncols();
        let c = pgs_z.ncols();

        // Verify dimensions match before matrix multiplication
        if m != pgs_z.nrows() || c != pgs_coef_len {
            return Err(ModelError::DimensionMismatch(format!(
                "PGS basis transform dimensions mismatch: B_unc {}×{}, Z {}×{}, coefs {}",
                pgs_main_basis_unc.nrows(),
                m,
                pgs_z.nrows(),
                c,
                pgs_coef_len
            )));
        }

        // Apply ONLY the sum-to-zero constraint to PGS main effect basis
        // This matches the training behavior in construction.rs
        let pgs_main_basis = pgs_main_basis_unc.dot(pgs_z);

        // CRITICAL: The interaction basis MUST be constructed from the UNCONSTRAINED PGS basis.
        // The model's coefficient layout (defined in construction.rs -> ModelLayout::new) is derived
        // from the dimensions of the unconstrained bases. Changing this to use a constrained basis
        // without a corresponding change to the layout logic will cause a dimension mismatch and lead
        // to silently incorrect predictions. The debug_assert in TrainedModel::predict guards against this.

        // Stage: Generate bases for PCs (functional ANOVA decomposition: null + range)
        let mut pc_range_bases = Vec::new();
        let mut pc_null_bases: Vec<Option<Array2<f64>>> = Vec::new();
        let mut pc_unconstrained_bases_main = Vec::new();
        for i in 0..config.pc_configs.len() {
            let pc_col = pcs_new.column(i);
            let pc_name = &config.pc_configs[i].name;
            // Only use saved knot vectors - remove fallback to ensure consistency
            let saved_knots = config
                .knot_vectors
                .get(pc_name)
                .ok_or_else(|| ModelError::InternalStackingError)?;

            let (pc_basis_unc, _) = basis::create_bspline_basis_with_knots(
                pc_col,
                saved_knots.view(),
                config.pc_configs[i].basis_config.degree,
            )?;

            // Slice the basis to remove the intercept term, just like in the training code
            let pc_main_basis_unc = pc_basis_unc.slice(s![.., 1..]);
            pc_unconstrained_bases_main.push(pc_main_basis_unc.to_owned());

            // Apply the SAVED null and range transformations for functional ANOVA
            if let Some(range_transform) = config.range_transforms.get(pc_name) {
                // Check dimensions before matrix multiplication to prevent panic
                if pc_main_basis_unc.ncols() != range_transform.nrows() {
                    return Err(ModelError::InternalStackingError);
                }

                // Use range-only transformation for PC main effects (fully penalized)
                let pc_range_basis = pc_main_basis_unc.dot(range_transform);
                pc_range_bases.push(pc_range_basis);
                // Get the required null-space transform
                let z_null = config.pc_null_transforms.get(pc_name).ok_or_else(|| {
                    ModelError::ConstraintMissing(format!("pc_null_transform for {}", pc_name))
                })?;

                if pc_main_basis_unc.ncols() != z_null.nrows() {
                    return Err(ModelError::DimensionMismatch(format!(
                        "PC null transform dimensions mismatch for {}: basis {}×{}, transform {}×{}",
                        pc_name,
                        pc_main_basis_unc.nrows(),
                        pc_main_basis_unc.ncols(),
                        z_null.nrows(),
                        z_null.ncols()
                    )));
                }

                // Build null-space basis (may have 0 columns if PC has no null space)
                let pc_null_basis = pc_main_basis_unc.dot(z_null);
                pc_null_bases.push(if z_null.ncols() > 0 {
                    Some(pc_null_basis)
                } else {
                    None
                });
            } else {
                // No range transform found for this PC
                return Err(ModelError::ConstraintMissing(format!(
                    "range transform for {}",
                    pc_name
                )));
            }
        }

        // Stage: Assemble the design matrix following the canonical order
        let n_samples = p_new.len();
        let mut owned_cols: Vec<Array1<f64>> = Vec::new();

        // Stage: Populate the intercept column
        owned_cols.push(Array1::<f64>::ones(n_samples));

        // Stage: Add main PC effects per PC—null first (if any), then range
        for pc_idx in 0..config.pc_configs.len() {
            if let Some(ref null_basis) = pc_null_bases[pc_idx] {
                for col in null_basis.axis_iter(Axis(1)) {
                    owned_cols.push(col.to_owned());
                }
            }
            for col in pc_range_bases[pc_idx].axis_iter(Axis(1)) {
                owned_cols.push(col.to_owned());
            }
        }

        // Stage: Add the main PGS effect
        for col in pgs_main_basis.axis_iter(Axis(1)) {
            owned_cols.push(col.to_owned());
        }

        // Stage: Add tensor product interaction effects (only if PCs are present)
        if !config.pc_configs.is_empty() {
            // Reconstruct interaction marginals using the same basis choice used during training
            let pgs_int_basis = match config.interaction_penalty {
                InteractionPenaltyKind::Isotropic => {
                    let z_range_pgs_pred = config
                        .range_transforms
                        .get("pgs")
                        .ok_or_else(|| {
                            ModelError::ConstraintMissing("pgs range transform".to_string())
                        })?;

                    if pgs_main_basis_unc.ncols() != z_range_pgs_pred.nrows() {
                        return Err(ModelError::InternalStackingError);
                    }

                    pgs_main_basis_unc.dot(z_range_pgs_pred)
                }
                InteractionPenaltyKind::Anisotropic => pgs_main_basis_unc.to_owned(),
            };

            for pc_idx in 0..config.pc_configs.len() {
                let pc_name = &config.pc_configs[pc_idx].name;
                let tensor_key = format!("f(PGS,{})", pc_name);

                // Only build the interaction if the trained model contains coefficients for it
                if coeffs.interaction_effects.contains_key(&tensor_key) {
                    let pc_int_basis = match config.interaction_penalty {
                        InteractionPenaltyKind::Isotropic => {
                            let z_range_pc_pred = config
                                .range_transforms
                                .get(pc_name)
                                .ok_or_else(|| ModelError::InternalStackingError)?;

                            if pc_unconstrained_bases_main[pc_idx].ncols()
                                != z_range_pc_pred.nrows()
                            {
                                return Err(ModelError::InternalStackingError);
                            }

                            pc_unconstrained_bases_main[pc_idx].dot(z_range_pc_pred)
                        }
                        InteractionPenaltyKind::Anisotropic => {
                            pc_unconstrained_bases_main[pc_idx].to_owned()
                        }
                    };

                    let mut tensor_interaction =
                        row_wise_tensor_product(&pgs_int_basis, &pc_int_basis);

                    // Apply stored orthogonalization to remove main-effect components (pure interaction)
                    let alpha =
                        config
                            .interaction_orth_alpha
                            .get(&tensor_key)
                            .ok_or_else(|| {
                                ModelError::ConstraintMissing(format!(
                                    "interaction_orth_alpha for {}",
                                    tensor_key
                                ))
                            })?;

                    // Build M = [Intercept | PGS_main | PC_main_for_this_pc (null + range)]
                    let intercept = Array1::<f64>::ones(n_samples).insert_axis(Axis(1));
                    // PGS_main is `pgs_main_basis` from above; append PC null (if any) then PC range
                    let mut m_cols: Vec<Array1<f64>> =
                        intercept.axis_iter(Axis(1)).map(|c| c.to_owned()).collect();
                    m_cols.extend(pgs_main_basis.axis_iter(Axis(1)).map(|c| c.to_owned()));
                    if let Some(ref pc_null) = pc_null_bases[pc_idx] {
                        m_cols.extend(pc_null.axis_iter(Axis(1)).map(|c| c.to_owned()));
                    }
                    m_cols.extend(
                        pc_range_bases[pc_idx]
                            .axis_iter(Axis(1))
                            .map(|c| c.to_owned()),
                    );
                    let m_matrix = ndarray::stack(
                        Axis(1),
                        &m_cols.iter().map(|c| c.view()).collect::<Vec<_>>(),
                    )
                    .map_err(|_| ModelError::InternalStackingError)?;

                    // Dimension check: alpha: m_cols x t_cols
                    if m_matrix.ncols() != alpha.nrows()
                        || tensor_interaction.ncols() != alpha.ncols()
                    {
                        return Err(ModelError::DimensionMismatch(format!(
                            "Orth map dims mismatch for {}: M {}x{}, alpha {}x{}, T {}x{}",
                            tensor_key,
                            m_matrix.nrows(),
                            m_matrix.ncols(),
                            alpha.nrows(),
                            alpha.ncols(),
                            tensor_interaction.nrows(),
                            tensor_interaction.ncols()
                        )));
                    }
                    tensor_interaction = &tensor_interaction - &m_matrix.dot(alpha);

                    // Add all columns from this tensor product to the design matrix (no extra centering)
                    for col in tensor_interaction.axis_iter(Axis(1)) {
                        owned_cols.push(col.to_owned());
                    }
                }
            }
        }
        // No fallback - only modern approach is supported now

        // Stack all column views into the final design matrix
        let col_views: Vec<_> = owned_cols.iter().map(Array1::view).collect();
        ndarray::stack(Axis(1), &col_views).map_err(|_| ModelError::InternalStackingError)
    }

    /// Flattens the structured `MappedCoefficients` into a single `Array1` vector,
    /// following the exact same canonical order used in `construct_design_matrix`.
    /// This function is the "mirror image" of the design matrix construction.
    pub(super) fn flatten_coefficients(
        coeffs: &MappedCoefficients,
        config: &ModelConfig,
    ) -> Result<Array1<f64>, ModelError> {
        let mut flattened: Vec<f64> = Vec::new();

        // Order of concatenation MUST exactly match `construct_design_matrix`.
        // Stage: Intercept
        flattened.push(coeffs.intercept);

        // Stage: Main PC effects (ordered by pc_configs for determinism)
        for pc_config in &config.pc_configs {
            let pc_name = &pc_config.name;
            let c = coeffs
                .main_effects
                .pcs
                .get(pc_name)
                .ok_or_else(|| ModelError::CoefficientMissing(pc_name.clone()))?;
            flattened.extend_from_slice(c);
        }

        // Stage: Main PGS effects (for basis functions m > 0)
        flattened.extend_from_slice(&coeffs.main_effects.pgs);

        // Stage: Tensor product interaction effects (ordered by PC)
        for pc_config in &config.pc_configs {
            let tensor_key = format!("f(PGS,{})", pc_config.name);
            if let Some(tensor_coeffs) = coeffs.interaction_effects.get(&tensor_key) {
                flattened.extend_from_slice(tensor_coeffs);
            }
        }

        Ok(Array1::from_vec(flattened))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // TrainingData is not used in tests
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, array};

    /// Test a trivially simple case with degree 1 B-spline (piecewise linear interpolation).
    /// This test uses a simple ground truth we can calculate by hand to verify the prediction.
    #[test]
    fn test_trained_model_predict() {
        // --- Setup a simple, known model configuration ---
        let knot_vector = array![0.0, 0.0, 0.5, 1.0, 1.0];
        let degree = 1;
        // For a degree=1 spline with this knot vector, there are 3 unconstrained basis functions.
        // The main effect (excluding the intercept) has 2 basis functions.
        // The sum-to-zero constraint will transform these 2 functions into 1 constrained function.
        // However, the test below will derive this transformation programmatically.

        // Create a dummy unconstrained basis to derive the constraint matrix Z
        let (unconstrained_basis_for_constraint, _) = basis::create_bspline_basis_with_knots(
            array![0.25, 0.75].view(), // two arbitrary points
            knot_vector.view(),
            degree,
        )
        .unwrap();
        let unconstrained_main_basis = unconstrained_basis_for_constraint.slice(s![.., 1..]);
        let (_, z_transform) =
            basis::apply_sum_to_zero_constraint(unconstrained_main_basis.view(), None).unwrap();

        let model = TrainedModel {
            config: ModelConfig {
                link_function: LinkFunction::Identity,
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                pgs_basis_config: BasisConfig {
                    num_knots: 1, // Number of internal knots
                    degree,
                },
                pc_configs: vec![],
                pgs_range: (0.0, 1.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: {
                    let mut constraints = HashMap::new();
                    constraints.insert("pgs_main".to_string(), z_transform.clone());
                    constraints
                },
                knot_vectors: {
                    let mut knots = HashMap::new();
                    knots.insert("pgs".to_string(), knot_vector.clone());
                    knots
                },
                range_transforms: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
            },
            coefficients: MappedCoefficients {
                intercept: 0.5, // Added an intercept for a more complete test
                main_effects: MainEffects {
                    // There is only 1 coefficient after constraint for this simple case
                    pgs: vec![2.0],
                    pcs: HashMap::new(),
                },
                interaction_effects: HashMap::new(),
            },
            lambdas: vec![],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
        };

        // --- Define Test Points ---
        let test_points = array![0.25, 0.75];
        let empty_pcs = Array2::<f64>::zeros((2, 0));

        // --- Calculate the expected result CORRECTLY ---
        // The manual calculation was flawed. Here's the correct way to derive the ground truth:
        // Stage: Generate the raw, unconstrained basis at the test points
        let (full_basis_unc, _) =
            basis::create_bspline_basis_with_knots(test_points.view(), knot_vector.view(), degree)
                .unwrap();

        // Stage: Isolate the main effect part of the basis (all columns except the intercept)
        let pgs_main_basis_unc = full_basis_unc.slice(s![.., 1..]);

        // Stage: Apply the same sum-to-zero constraint transformation
        let pgs_main_basis_con = pgs_main_basis_unc.dot(&z_transform);

        // Stage: Get the coefficients for the constrained basis
        let coeffs = Array1::from(model.coefficients.main_effects.pgs.clone());

        // Stage: Calculate the final expected linear predictor as intercept + constrained_basis * coeffs
        let expected_values = model.coefficients.intercept + pgs_main_basis_con.dot(&coeffs);

        // Get the model's prediction using the actual `predict` method
        let predictions = model.predict(test_points.view(), empty_pcs.view()).unwrap();

        // Verify the results match our correctly calculated ground truth
        assert_eq!(predictions.len(), 2);
        assert_abs_diff_eq!(
            predictions.as_slice().unwrap(),
            expected_values.as_slice().unwrap(),
            epsilon = 1e-10
        );
    }

    /// Tests that the prediction fails appropriately with invalid input dimensions.
    #[test]
    fn test_trained_model_predict_invalid_input() {
        let model = TrainedModel {
            config: ModelConfig {
                link_function: LinkFunction::Identity,
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                pgs_basis_config: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                pc_configs: vec![PrincipalComponentConfig {
                    name: "PC1".to_string(),
                    basis_config: BasisConfig {
                        num_knots: 2,
                        degree: 1,
                    },
                    range: (-1.0, 1.0),
                }],
                pgs_range: (-2.0, 2.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: HashMap::new(),
                knot_vectors: HashMap::new(),
                range_transforms: HashMap::new(),
                pc_null_transforms: HashMap::new(),
                interaction_centering_means: HashMap::new(),
                interaction_orth_alpha: HashMap::new(),
            },
            coefficients: MappedCoefficients {
                intercept: 0.0,
                main_effects: MainEffects {
                    pgs: vec![],
                    pcs: HashMap::new(),
                },
                interaction_effects: HashMap::new(),
            },
            lambdas: vec![],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
        };

        // Test with mismatched PC dimensions (model expects 1 PC, but we provide 2)
        let pgs = Array1::linspace(0.0, 1.0, 5);
        let pcs = Array2::zeros((5, 2)); // 2 PC columns, but model expects 1

        let result = model.predict(pgs.view(), pcs.view());

        // Verify we get the expected error
        assert!(
            result.is_err(),
            "Should return an error when PC dimensions don't match"
        );
        if let Err(ModelError::MismatchedPcCount { found, expected }) = result {
            assert_eq!(found, 2);
            assert_eq!(expected, 1);
        } else {
            panic!("Expected MismatchedPcCount error");
        }
    }

    /// Tests that the coefficient flattening logic works correctly.
    /// This test verifies that the structured MappedCoefficients are correctly
    /// flattened into a single vector following the canonical order.
    #[test]
    fn test_coefficient_flattening() {
        // Create a small structured coefficient object with known values
        let coeffs = MappedCoefficients {
            intercept: 1.0,
            main_effects: MainEffects {
                pgs: vec![2.0, 3.0], // 2 PGS main effect coefficients
                pcs: {
                    let mut pc_map = HashMap::new();
                    pc_map.insert("PC1".to_string(), vec![4.0, 5.0]); // 2 PC1 coefficients
                    pc_map.insert("PC2".to_string(), vec![6.0, 7.0]); // 2 PC2 coefficients
                    pc_map
                },
            },
            interaction_effects: {
                let mut interactions = HashMap::new();

                // Unified tensor product interactions - flattened coefficient vectors
                // For tensor products, coefficients are flattened in the order: pgs_basis × pc_basis
                interactions.insert("f(PGS,PC1)".to_string(), vec![8.0, 9.0, 12.0, 13.0]);
                interactions.insert("f(PGS,PC2)".to_string(), vec![10.0, 11.0, 14.0, 15.0]);

                interactions
            },
        };

        // Create a simple model config for testing
        let config = ModelConfig {
            link_function: LinkFunction::Identity,
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 50,
            pgs_basis_config: BasisConfig {
                num_knots: 3,
                degree: 3,
            },
            pc_configs: vec![
                PrincipalComponentConfig {
                    name: "PC1".to_string(),
                    basis_config: BasisConfig {
                        num_knots: 3,
                        degree: 3,
                    },
                    range: (0.0, 1.0),
                },
                PrincipalComponentConfig {
                    name: "PC2".to_string(),
                    basis_config: BasisConfig {
                        num_knots: 3,
                        degree: 3,
                    },
                    range: (0.0, 1.0),
                },
            ],
            pgs_range: (0.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
        };

        // Use the internal flatten_coefficients function
        use super::internal::flatten_coefficients;
        let flattened = flatten_coefficients(&coeffs, &config)
            .expect("flatten_coefficients should succeed for well-formed inputs");

        // Verify the canonical order of flattening:
        // Stage: Intercept
        // Stage: PC main effects (ordered by config.pc_configs)
        // Stage: PGS main effects
        // Stage: Interaction effects (PGS basis function 1, then PC1, PC2, etc.; followed by additional PGS basis functions)

        // Define expected order based on the canonical ordering rules
        let expected = vec![
            1.0, // Intercept
            4.0, 5.0, // PC1 main effects
            6.0, 7.0, // PC2 main effects
            2.0, 3.0, // PGS main effects
            8.0, 9.0, 12.0, 13.0, // f(PGS,PC1) tensor product interaction
            10.0, 11.0, 14.0, 15.0, // f(PGS,PC2) tensor product interaction
        ];

        // Convert to vectors for easier comparison
        let flat_vec = flattened.to_vec();

        // Check total length
        assert_eq!(
            flat_vec.len(),
            expected.len(),
            "Flattened vector has incorrect length: expected {}, got {}",
            expected.len(),
            flat_vec.len()
        );

        // Check each element in the expected order
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_eq!(
                flat_vec[i], expected_val,
                "Mismatch at position {}: expected {}, got {}",
                i, expected_val, flat_vec[i]
            );
        }
    }

    /// Tests that the model can be saved to and loaded from a file,
    /// preserving all its contents exactly.
    #[test]
    fn test_save_load_functionality() {
        use crate::calibrate::data::TrainingData;

        // Define the BasisConfig to use
        let pgs_basis_config = BasisConfig {
            num_knots: 6,
            degree: 3,
        };
        let pc1_basis_config = BasisConfig {
            num_knots: 6,
            degree: 3,
        };

        // Create a ModelConfig to use for both generation and testing
        let model_config = ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 100,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 50,
            pgs_basis_config: pgs_basis_config.clone(),
            pc_configs: vec![PrincipalComponentConfig {
                name: "PC1".to_string(),
                basis_config: pc1_basis_config.clone(),
                range: (-0.5, 0.5),
            }],
            pgs_range: (-1.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            knot_vectors: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            range_transforms: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
        };

        // Create a dummy dataset for generating the correct model structure
        // Using n_samples=100 to avoid over-parameterization
        let n_samples = 100;
        let dummy_data = TrainingData {
            y: Array1::linspace(0.0, 1.0, n_samples),
            p: Array1::linspace(-1.0, 1.0, n_samples),
            pcs: Array2::from_shape_vec(
                (n_samples, 1),
                Array1::linspace(-0.5, 0.5, n_samples).to_vec(),
            )
            .unwrap(),
            weights: Array1::<f64>::ones(n_samples),
        };

        // Generate the correct constraints, structure, and range transforms using the actual model-building code
        let (
            _,
            _,
            layout,
            sum_to_zero_constraints,
            knot_vectors,
            range_transforms,
            pc_null_transforms,
            interaction_centering_means,
            interaction_orth_alpha,
        ) = crate::calibrate::construction::build_design_and_penalty_matrices(
            &dummy_data,
            &model_config,
        )
        .expect("Failed to build model matrices");

        // Now we can create a test model with the correct structure
        let original_model = TrainedModel {
            config: ModelConfig {
                link_function: LinkFunction::Logit,
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                pgs_basis_config: pgs_basis_config.clone(),
                pc_configs: vec![PrincipalComponentConfig {
                    name: "PC1".to_string(),
                    basis_config: pc1_basis_config.clone(),
                    range: (-0.5, 0.5),
                }],
                pgs_range: (-1.0, 1.0),
                interaction_penalty: InteractionPenaltyKind::Anisotropic,
                sum_to_zero_constraints: sum_to_zero_constraints.clone(), // Clone the new field
                knot_vectors, // Use the knot vectors generated by the model-building code
                range_transforms: range_transforms.clone(), // Use full Option 3 pipeline
                pc_null_transforms: pc_null_transforms.clone(),
                interaction_centering_means: interaction_centering_means.clone(),
                interaction_orth_alpha: interaction_orth_alpha.clone(),
            },
            coefficients: MappedCoefficients {
                intercept: 0.5,
                main_effects: MainEffects {
                    // The number of PGS main effect coefficients must match dimensions after constraint
                    // This size is derived from the layout object which reflects the actual model structure
                    pgs: {
                        let pgs_dim = layout.pgs_main_cols.end - layout.pgs_main_cols.start;
                        // Fill with increasing values 1.0, 2.0, 3.0, ...
                        (1..=pgs_dim).map(|i| i as f64).collect()
                    },
                    pcs: {
                        // Use null + range dimension for PC1
                        let pc1_dim_null = pc_null_transforms
                            .get("PC1")
                            .map(|z| z.ncols())
                            .unwrap_or(0);
                        let pc1_dim_range = range_transforms.get("PC1").unwrap().ncols();

                        let mut pc_map = HashMap::new();
                        pc_map.insert(
                            "PC1".to_string(),
                            (1..=(pc1_dim_null + pc1_dim_range))
                                .map(|i| i as f64)
                                .collect(),
                        );
                        pc_map
                    },
                },
                interaction_effects: {
                    // Interaction term uses Range × Range dimensions
                    let r_pgs = range_transforms.get("pgs").unwrap().ncols();
                    let r_pc1 = range_transforms.get("PC1").unwrap().ncols();

                    let mut interactions = HashMap::new();

                    // Calculate the total number of interaction coefficients for Range × Range
                    let total_interaction_coeffs = r_pgs * r_pc1;

                    // Create a single flattened vector of coefficients
                    let interaction_coeffs: Vec<f64> = (1..=total_interaction_coeffs)
                        .map(|i| i as f64 * 10.0) // Example values
                        .collect();

                    // Insert under the single, correct key
                    interactions.insert("f(PGS,PC1)".to_string(), interaction_coeffs);
                    interactions
                },
            },
            // Match layout.num_penalties dynamically (PC null + PC range + interaction)
            lambdas: vec![0.1; layout.num_penalties],
            hull: None,
            penalized_hessian: None,
            scale: None,
            calibrator: None,
        };

        // Create a temporary file for testing
        use tempfile::NamedTempFile;
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let file_path = temp_file.path().to_str().unwrap();

        // Save the model
        original_model
            .save(file_path)
            .expect("Failed to save model");

        // Load the model back
        let loaded_model = TrainedModel::load(file_path).expect("Failed to load model");

        // Compare the models

        // Stage: Check model configuration
        assert_eq!(
            loaded_model.config.link_function,
            original_model.config.link_function
        );
        assert_eq!(
            loaded_model.config.penalty_order,
            original_model.config.penalty_order
        );
        assert_eq!(
            loaded_model.config.pgs_basis_config.num_knots,
            original_model.config.pgs_basis_config.num_knots
        );
        assert_eq!(
            loaded_model.config.pgs_basis_config.degree,
            original_model.config.pgs_basis_config.degree
        );
        assert_eq!(
            loaded_model.config.pc_configs.len(),
            original_model.config.pc_configs.len()
        );
        // Check that each PC config matches
        for i in 0..loaded_model.config.pc_configs.len() {
            assert_eq!(
                loaded_model.config.pc_configs[i].name,
                original_model.config.pc_configs[i].name
            );
            assert_eq!(
                loaded_model.config.pc_configs[i].range,
                original_model.config.pc_configs[i].range
            );
            assert_eq!(
                loaded_model.config.pc_configs[i].basis_config.num_knots,
                original_model.config.pc_configs[i].basis_config.num_knots
            );
            assert_eq!(
                loaded_model.config.pc_configs[i].basis_config.degree,
                original_model.config.pc_configs[i].basis_config.degree
            );
        }
        assert_eq!(
            loaded_model.config.pgs_range,
            original_model.config.pgs_range
        );

        // Stage: Confirm sum_to_zero_constraints transformations exist for the PGS main effect only
        assert!(
            loaded_model
                .config
                .sum_to_zero_constraints
                .contains_key("pgs_main")
        );

        // Stage: Check coefficient values
        assert_eq!(
            loaded_model.coefficients.intercept,
            original_model.coefficients.intercept
        );
        assert_eq!(
            loaded_model.coefficients.main_effects.pgs,
            original_model.coefficients.main_effects.pgs
        );
        assert_eq!(
            loaded_model.coefficients.main_effects.pcs.len(),
            original_model.coefficients.main_effects.pcs.len()
        );

        if let Some(pc1_loaded) = loaded_model.coefficients.main_effects.pcs.get("PC1") {
            if let Some(pc1_orig) = original_model.coefficients.main_effects.pcs.get("PC1") {
                assert_eq!(pc1_loaded, pc1_orig);
            } else {
                panic!("Original model missing PC1 main effects");
            }
        } else {
            panic!("Loaded model missing PC1 main effects");
        }

        // Stage: Check interaction effects
        assert_eq!(
            loaded_model.coefficients.interaction_effects.len(),
            original_model.coefficients.interaction_effects.len()
        );

        // Check for interaction effect f(PGS,PC1)
        let key_interaction = "f(PGS,PC1)";
        if let (Some(interaction_loaded), Some(interaction_orig)) = (
            loaded_model
                .coefficients
                .interaction_effects
                .get(key_interaction),
            original_model
                .coefficients
                .interaction_effects
                .get(key_interaction),
        ) {
            assert_eq!(
                interaction_loaded, interaction_orig,
                "Mismatch in {}",
                key_interaction
            );
        } else {
            panic!("Missing {} interaction effect", key_interaction);
        }

        // Stage: Check lambdas
        assert_eq!(loaded_model.lambdas, original_model.lambdas);

        // Stage: Verify that we can use the loaded model for prediction
        let test_pgs = Array1::linspace(-0.5, 0.5, 3);
        let test_pcs = Array2::from_shape_fn((3, 1), |(i, _)| {
            (i as f64 - 1.0) * 0.25 // Values: -0.25, 0, 0.25
        });

        let predictions_orig = original_model
            .predict(test_pgs.view(), test_pcs.view())
            .expect("Prediction with original model failed");

        let predictions_loaded = loaded_model
            .predict(test_pgs.view(), test_pcs.view())
            .expect("Prediction with loaded model failed");

        // Verify that predictions are identical
        for i in 0..predictions_orig.len() {
            assert_eq!(
                predictions_orig[i], predictions_loaded[i],
                "Predictions differ at position {}",
                i
            );
        }
    }
}

// TODO: Move this somewhere else
/// Maps the flattened coefficient vector to a structured representation.
pub fn map_coefficients(
    beta: &Array1<f64>,
    layout: &ModelLayout,
) -> Result<MappedCoefficients, EstimationError> {
    let intercept = beta[layout.intercept_col];
    let mut pcs = HashMap::new();
    let mut pgs = vec![];
    let mut interaction_effects = HashMap::new();

    // Extract the unpenalized PGS main effect coefficients
    if !layout.pgs_main_cols.is_empty() {
        pgs = beta.slice(s![layout.pgs_main_cols.clone()]).to_vec();
    }

    // Build PC main-effect coefficients deterministically by layout indices
    // For each PCj: concatenate [null-space coeffs for PCj (if any), range-space coeffs for PCj].
    // Infer PC count from layout vectors; layout invariants guarantee alignment.
    let num_pcs = layout.pc_main_block_idx.len();
    for i in 0..num_pcs {
        let range_block_idx = layout.pc_main_block_idx[i];
        let range_block = &layout.penalty_map[range_block_idx];
        let mut full = Vec::new();
        let null_range = &layout.pc_null_cols[i];
        if null_range.len() > 0 {
            full.extend_from_slice(&beta.slice(s![null_range.clone()]).to_vec());
        }
        full.extend_from_slice(&beta.slice(s![range_block.col_range.clone()]).to_vec());

        // Derive PC name robustly from the range block term name, which is guaranteed to be f(PCname)
        let name_str = &range_block.term_name;
        let pc_name = if let Some(start) = name_str.find("f(") {
            let rest = &name_str[start + 2..];
            let end = rest.find(')').ok_or_else(|| {
                EstimationError::LayoutError(format!("Malformed PC term name: {}", name_str))
            })?;
            rest[..end].to_string()
        } else {
            return Err(EstimationError::LayoutError(format!(
                "Unexpected PC term name (missing prefix): {}",
                name_str
            )));
        };
        pcs.insert(pc_name, full);
    }

    // Build interaction effects deterministically by layout.interaction_block_idx order.
    for &blk_idx in &layout.interaction_block_idx {
        let block = &layout.penalty_map[blk_idx];
        let coeffs = beta.slice(s![block.col_range.clone()]).to_vec();
        interaction_effects.insert(block.term_name.to_string(), coeffs);
    }

    Ok(MappedCoefficients {
        intercept,
        main_effects: MainEffects { pgs, pcs },
        interaction_effects,
    })
}
