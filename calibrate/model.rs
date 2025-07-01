use crate::basis::{self, create_bspline_basis};
use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, BufWriter, Write};
use thiserror::Error;

// --- Public Data Structures ---
// These structs define the public, human-readable format of the trained model.

/// Defines the link function, connecting the linear predictor to the mean response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkFunction {
    Logit,    // For binary/proportional outcomes.
    Identity, // For continuous outcomes.
}

/// Configuration for a single basis expansion (e.g., for one variable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisConfig {
    pub num_knots: usize,
    pub degree: usize,
}

/// The complete blueprint of a trained model.
/// Contains all hyperparameters and structural information needed for prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub link_function: LinkFunction,
    pub penalty_order: usize,
    pub lambda: f64,
    pub pgs_basis_config: BasisConfig,
    pub pc_basis_configs: Vec<BasisConfig>,
    // Data-dependent parameters saved from training are crucial for prediction.
    pub pgs_range: (f64, f64),
    pub pc_ranges: Vec<(f64, f64)>,
    pub pc_names: Vec<String>, // Defines the canonical order for PCs.
}

/// A structured representation of the fitted model coefficients, designed for interpretability.
#[derive(Debug, Serialize, Deserialize)]
pub struct MappedCoefficients {
    pub intercept: f64,
    pub main_effects: MainEffects,
    /// Nested map for interaction terms. Outer key is the PGS basis function name
    /// (e.g., "pgs_B1"), inner key is the PC name (e.g., "PC1").
    pub interaction_effects: HashMap<String, HashMap<String, Vec<f64>>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MainEffects {
    pub pgs: Vec<f64>,
    pub pcs: HashMap<String, Vec<f64>>,
}

/// The top-level, self-contained, trained model artifact.
/// This is the structure that gets saved to and loaded from a file.
#[derive(Debug, Serialize, Deserialize)]
pub struct TrainedModel {
    pub config: ModelConfig,
    pub coefficients: MappedCoefficients,
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
    ) -> Result<Array2<f64>, basis::BasisError> {
        // --- A. Generate individual basis expansions using saved config ---
        let (pgs_basis, _) = create_bspline_basis(
            p_new,
            None, // No quantile data at prediction time
            config.pgs_range,
            config.pgs_basis_config.num_knots,
            config.pgs_basis_config.degree,
        )?;

        let pc_bases: Vec<Array2<f64>> = pcs_new
            .axis_iter(Axis(1))
            .zip(config.pc_ranges.iter().zip(config.pc_basis_configs.iter()))
            .map(|(pc_col, (&range, pc_conf))| {
                create_bspline_basis(pc_col, None, range, pc_conf.num_knots, pc_conf.degree)
                    .map(|(basis, _)| basis)
            })
            .collect::<Result<_, _>>()?;

        // --- B. Assemble columns into a Vec in the canonical order ---
        // This vec will own the product arrays for the interaction terms.
        let mut owned_cols: Vec<Array1<f64>> = Vec::new();
        
        // 1. Intercept
        owned_cols.push(Array1::ones(p_new.len()));

        // 2. Main PC effects
        for pc_basis in &pc_bases {
            for col in pc_basis.axis_iter(Axis(1)) {
                owned_cols.push(col.to_owned());
            }
        }

        // 3. Main PGS effects (m > 0)
        let pgs_main_effects = pgs_basis.slice(s![.., 1..]);
        for col in pgs_main_effects.axis_iter(Axis(1)) {
            owned_cols.push(col.to_owned());
        }

        // 4. Interaction effects
        for pgs_basis_col in pgs_main_effects.axis_iter(Axis(1)) {
            for pc_basis in &pc_bases {
                for pc_basis_col in pc_basis.axis_iter(Axis(1)) {
                    owned_cols.push(&pgs_basis_col * &pc_basis_col);
                }
            }
        }
        
        // --- C. Stack into a single matrix ---
        let col_views: Vec<_> = owned_cols.iter().map(Array1::view).collect();
        // This stack should never fail if the logic is correct.
        Ok(ndarray::stack(Axis(1), &col_views).unwrap())
    }

    /// Flattens the structured `MappedCoefficients` into a single `Array1` vector,
    /// following the exact same canonical order used in `construct_design_matrix`.
    pub(super) fn flatten_coefficients(
        coeffs: &MappedCoefficients,
        config: &ModelConfig,
    ) -> Array1<f64> {
        let mut flattened: Vec<f64> = Vec::new();

        // 1. Intercept
        flattened.push(coeffs.intercept);

        // 2. Main PC effects (ordered by pc_names)
        for pc_name in &config.pc_names {
            if let Some(c) = coeffs.main_effects.pcs.get(pc_name) {
                flattened.extend_from_slice(c);
            }
        }

        // 3. Main PGS effects
        flattened.extend_from_slice(&coeffs.main_effects.pgs);

        // 4. Interaction effects (ordered by PGS basis index, then by pc_names)
        let num_pgs_main_effects = config.pgs_basis_config.num_knots + config.pgs_basis_config.degree;
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
