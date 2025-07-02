use crate::calibrate::basis::{self, create_bspline_basis};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};
use thiserror::Error;

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

/// Configuration for a single basis expansion (e.g., for one variable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisConfig {
    pub num_knots: usize,
    pub degree: usize,
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
    pub pc_basis_configs: Vec<BasisConfig>,
    // Data-dependent parameters saved from training are crucial for prediction.
    pub pgs_range: (f64, f64),
    pub pc_ranges: Vec<(f64, f64)>,
    /// Defines the canonical order for Principal Components. This order is strictly
    /// enforced during both model fitting and prediction to ensure correctness.
    pub pc_names: Vec<String>,
    
    /// A map from a term name (e.g., "PC1", "pgs_main") to its constraint transformation.
    /// Use an Option because not all terms might be constrained.
    #[serde(default)] // For backward compatibility with old models that don't have this field
    pub constraints: HashMap<String, Constraint>,
    /// Knot vectors used during training, saved for exact reproduction during prediction
    #[serde(default)] // For backward compatibility with old models that don't have this field
    pub knot_vectors: HashMap<String, Array1<f64>>,
}

/// A structured representation of the fitted model coefficients, designed for
/// human interpretation and sharing. This structure is used in the TOML file.
#[derive(Debug, Serialize, Deserialize)]
pub struct MappedCoefficients {
    pub intercept: f64,
    pub main_effects: MainEffects,
    /// Nested map for interaction terms.
    /// - Outer key: PGS basis function name (e.g., "pgs_B1").
    /// - Inner key: PC name (e.g., "PC1").
    /// - Value: The vector of coefficients for that interaction's spline.
    pub interaction_effects: HashMap<String, HashMap<String, Vec<f64>>>,
}

#[derive(Debug, Serialize, Deserialize)]
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
    #[error("Internal error: failed to stack design matrix columns during prediction.")]
    InternalStackingError,
    #[error("Constraint transformation matrix missing for term '{0}'. This usually indicates a model format mismatch.")]
    ConstraintMissing(String),
}

impl TrainedModel {
    /// Predicts outcomes for new individuals using the trained model.
    ///
    /// This is the core inference engine. It is a fast, non-iterative process that:
    /// 1. Reconstructs the mathematical model (design matrix and coefficient vector)
    ///    from the stored configuration.
    /// 2. Computes the final prediction via matrix algebra.
    ///
    /// # Arguments
    /// * `p_new`: A 1D array view of new PGS values.
    /// * `pcs_new`: A 2D array view of new PC values, with shape `[n_samples, n_pcs]`.
    ///              The order of PC columns must match `config.pc_names`.
    ///
    /// # Returns
    /// A `Result` containing an `Array1<f64>` of predicted outcomes (e.g., probabilities
    /// or continuous values), or a `ModelError`.
    pub fn predict(
        &self,
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        // --- 1. Validate Inputs ---
        if pcs_new.ncols() != self.config.pc_names.len() {
            return Err(ModelError::MismatchedPcCount {
                found: pcs_new.ncols(),
                expected: self.config.pc_names.len(),
            });
        }

        // --- 2. Reconstruct Mathematical Objects ---
        let x_new = internal::construct_design_matrix(p_new, pcs_new, &self.config)?;
        let flattened_coeffs = internal::flatten_coefficients(&self.coefficients, &self.config);

        // --- 3. Compute Linear Predictor ---
        let eta = x_new.dot(&flattened_coeffs);

        // --- 4. Apply Inverse Link Function ---
        let predictions = match self.config.link_function {
            LinkFunction::Logit => eta.mapv(|e| 1.0 / (1.0 + (-e).exp())), // Sigmoid
            LinkFunction::Identity => eta,
        };

        Ok(predictions)
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

    /// Constructs the design matrix `X` for new data following a strict canonical order.
    /// This order is the implicit contract that allows the flattened coefficients to work correctly.
    pub(super) fn construct_design_matrix(
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        config: &ModelConfig,
    ) -> Result<Array2<f64>, ModelError> {
        // 1. Generate basis for PGS using saved knot vector if available
        let (pgs_basis_unc, _) = if let Some(saved_knots) = config.knot_vectors.get("pgs") {
            // Use saved knot vector for exact reproduction
            basis::create_bspline_basis_with_knots(p_new, saved_knots.view(), config.pgs_basis_config.degree)?
        } else {
            // Fallback to original method for backward compatibility
            create_bspline_basis(
                p_new, 
                None,
                config.pgs_range, 
                config.pgs_basis_config.num_knots, 
                config.pgs_basis_config.degree
            )?
        };

        // Apply the SAVED PGS constraint
        let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);
        let pgs_z = &config.constraints.get("pgs_main")
            .ok_or_else(|| ModelError::ConstraintMissing("pgs_main".to_string()))? 
            .z_transform;
        let pgs_main_basis = pgs_main_basis_unc.dot(pgs_z); // Now constrained

        // helper closure to fetch a Z from model or panic nicely
        let _lookup_z = |name: &str| -> Result<&Array2<f64>, ModelError> {
            Ok(&config.constraints.get(name)
                .ok_or_else(|| ModelError::ConstraintMissing(name.to_string()))?
                .z_transform)
        };

        // 2. Generate bases for PCs using saved knot vectors if available
        let mut pc_constrained_bases = Vec::new();
        for i in 0..config.pc_names.len() {
            let pc_col = pcs_new.column(i);
            let pc_name = &config.pc_names[i];
            let (pc_basis_unc, _) = if let Some(saved_knots) = config.knot_vectors.get(pc_name) {
                // Use saved knot vector for exact reproduction
                basis::create_bspline_basis_with_knots(pc_col, saved_knots.view(), config.pc_basis_configs[i].degree)?
            } else {
                // Fallback to original method for backward compatibility
                create_bspline_basis(
                    pc_col,
                    None,
                    config.pc_ranges[i],
                    config.pc_basis_configs[i].num_knots,
                    config.pc_basis_configs[i].degree
                )?
            };

            // Apply the SAVED PC constraint
            let pc_name = &config.pc_names[i];
            let pc_z = &config.constraints.get(pc_name)
                .ok_or_else(|| ModelError::ConstraintMissing(pc_name.clone()))?
                .z_transform;
            pc_constrained_bases.push(pc_basis_unc.dot(pc_z)); // Now constrained
        }

        // 3. Assemble the design matrix following the canonical order
        let n_samples = p_new.len();
        let mut owned_cols: Vec<Array1<f64>> = Vec::new();

        // 1. Intercept
        owned_cols.push(Array1::ones(n_samples));

        // 2. Main PC effects
        for pc_basis in &pc_constrained_bases {
            for col in pc_basis.axis_iter(Axis(1)) {
                owned_cols.push(col.to_owned());
            }
        }

        // 3. Main PGS effect
        for col in pgs_main_basis.axis_iter(Axis(1)) {
            owned_cols.push(col.to_owned());
        }

        // 4. Interaction effects
        for m in 0..pgs_main_basis.ncols() {
            let pgs_weight_col = pgs_basis_unc.column(m + 1);       // unconstrained
            for (_pc_idx, pc_basis_con) in pc_constrained_bases.iter().enumerate() {
                // Pure pre-centering approach: use constrained PC basis with unconstrained PGS
                // PC basis is already constrained (sum-to-zero). No need to center PGS or apply further constraints.
                let interaction_term = pc_basis_con * &pgs_weight_col.view().insert_axis(Axis(1));

                for col in interaction_term.axis_iter(Axis(1)) {
                    owned_cols.push(col.to_owned());
                }
            }
        }
        
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
    ) -> Array1<f64> {
        let mut flattened: Vec<f64> = Vec::new();

        // Order of concatenation MUST exactly match `construct_design_matrix`.
        // 1. Intercept
        flattened.push(coeffs.intercept);

        // 2. Main PC effects (ordered by `pc_names` for determinism).
        for pc_name in &config.pc_names {
            if let Some(c) = coeffs.main_effects.pcs.get(pc_name) {
                flattened.extend_from_slice(c);
            }
        }

        // 3. Main PGS effects (for basis functions m > 0).
        flattened.extend_from_slice(&coeffs.main_effects.pgs);

        // 4. Interaction effects (ordered by PGS basis index `m`, then by `pc_names`).
        let num_pgs_main_effects = config.pgs_basis_config.num_knots + config.pgs_basis_config.degree - 1; // after constraint
        for m in 1..=num_pgs_main_effects {
            let pgs_key = format!("pgs_B{}", m);
            if let Some(pc_map) = coeffs.interaction_effects.get(&pgs_key) {
                for pc_name in &config.pc_names {
                    if let Some(c) = pc_map.get(pc_name) {
                        flattened.extend_from_slice(c);
                    }
                }
            }
        }

        Array1::from_vec(flattened)
    }
}
