use crate::calibrate::basis::{self, create_bspline_basis};
use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
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
    /// Knot vectors used during training, required for exact reproduction during prediction
    pub knot_vectors: HashMap<String, Array1<f64>>,
    /// Number of PGS interaction bases (empirical from actual basis generation)
    /// Used to ensure consistency between construction and coefficient flattening
    pub num_pgs_interaction_bases: usize,
}

/// A structured representation of the fitted model coefficients, designed for
/// human interpretation and sharing. This structure is used in the TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MappedCoefficients {
    pub intercept: f64,
    pub main_effects: MainEffects,
    /// Nested map for interaction terms.
    /// - Outer key: PGS basis function name (e.g., "PGS_B1").
    /// - Inner key: PC name (e.g., "PC1").
    /// - Value: The vector of coefficients for that interaction's spline.
    pub interaction_effects: HashMap<String, HashMap<String, Vec<f64>>>,
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
    #[error(
        "Internal error: failed to stack design matrix columns or constraint matrix dimensions don't match basis dimensions during prediction."
    )]
    InternalStackingError,
    #[error(
        "Constraint transformation matrix missing for term '{0}'. This usually indicates a model format mismatch."
    )]
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
            LinkFunction::Logit => {
                // Clamp eta to prevent numerical overflow in exp(), just like in pirls.rs
                let eta_clamped = eta.mapv(|e| e.clamp(-700.0, 700.0));
                // Apply the inverse link function (sigmoid) to the clamped eta
                let mut probs = eta_clamped.mapv(|e| 1.0 / (1.0 + f64::exp(-e))); // Sigmoid
                // BUGFIX: Clamp probabilities away from 0 and 1 to prevent numerical instability
                // This matches the behavior in the training code (pirls.rs::update_glm_vectors)
                probs.mapv_inplace(|p| p.clamp(1e-8, 1.0 - 1e-8));
                probs
            }
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
        // Only use saved knot vectors - remove fallback to ensure consistency
        let saved_knots = config.knot_vectors.get("pgs")
            .ok_or_else(|| ModelError::InternalStackingError)?;
        
        let (pgs_basis_unc, _) = basis::create_bspline_basis_with_knots(
            p_new,
            saved_knots.view(),
            config.pgs_basis_config.degree,
        )?;

        // Apply the SAVED PGS constraint
        let pgs_main_basis_unc = pgs_basis_unc.slice(s![.., 1..]);
        let pgs_z = &config
            .constraints
            .get("pgs_main")
            .ok_or_else(|| ModelError::ConstraintMissing("pgs_main".to_string()))?
            .z_transform;

        // Check that dimensions match before matrix multiplication
        if pgs_main_basis_unc.ncols() != pgs_z.nrows() {
            return Err(ModelError::InternalStackingError);
        }

        let pgs_main_basis = pgs_main_basis_unc.dot(pgs_z); // Now constrained

        // For interactions, use the constrained pgs_main_basis directly
        // Cannot reconstruct "full" basis due to dimensional reduction from constraints

        // This closure was not used - removed
        // Previously defined a helper to fetch Z transform from model

        // 2. Generate bases for PCs using saved knot vectors if available
        let mut pc_constrained_bases = Vec::new();
        for i in 0..config.pc_names.len() {
            let pc_col = pcs_new.column(i);
            let pc_name = &config.pc_names[i];
            // Only use saved knot vectors - remove fallback to ensure consistency
            let saved_knots = config.knot_vectors.get(pc_name)
                .ok_or_else(|| ModelError::InternalStackingError)?;
            
            let (pc_basis_unc, _) = basis::create_bspline_basis_with_knots(
                pc_col,
                saved_knots.view(),
                config.pc_basis_configs[i].degree,
            )?;

            // Apply the SAVED PC constraint
            let pc_name = &config.pc_names[i];
            let pc_z = &config
                .constraints
                .get(pc_name)
                .ok_or_else(|| ModelError::ConstraintMissing(pc_name.clone()))?
                .z_transform;

            // Check that dimensions match before matrix multiplication
            if pc_basis_unc.ncols() != pc_z.nrows() {
                return Err(ModelError::InternalStackingError);
            }

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
        // Use the saved empirical count from config for consistency
        // This ensures construct_design_matrix and flatten_coefficients use the same count
        let total_pgs_bases = config.num_pgs_interaction_bases;

        // Use 1-indexed loop for interaction effects to match the canonical ordering in estimate.rs
        for m in 1..=total_pgs_bases {
            // Use unconstrained PGS basis for interaction weights
            // Note: pgs_main_basis_unc excludes intercept column, so m=1 maps to index 0
            if m == 0 || m > pgs_main_basis_unc.ncols() {
                continue; // Skip out-of-bounds
            }
            let pgs_weight_col_uncentered = pgs_main_basis_unc.column(m - 1);
            
            // Center the PGS basis column to ensure orthogonality
            let mean = pgs_weight_col_uncentered.mean().unwrap_or(0.0);
            let pgs_weight_col = &pgs_weight_col_uncentered - mean;

            // Iterate through PC names in the canonical order
            for pc_name in &config.pc_names {
                // Find the PC index from the name
                let pc_idx = config.pc_names.iter().position(|n| n == pc_name).unwrap();
                let pc_basis_con = &pc_constrained_bases[pc_idx];

                // Pure interaction approach: use constrained PC basis with centered PGS
                // Both bases are now properly centered to avoid linear dependencies
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
        // CRITICAL: Use the empirical value saved during training to ensure consistency.
        let total_pgs_bases = config.num_pgs_interaction_bases;
        for m in 1..=total_pgs_bases {
            let pgs_key = format!("PGS_B{m}");
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

#[cfg(test)]
mod tests {
    use super::*;
    // TrainingData is not used in tests
    use ndarray::{Array1, Array2, array};

    /// Test a trivially simple case with degree 1 B-spline (piecewise linear interpolation).
    /// This test uses a simple ground truth we can calculate by hand to verify the prediction.
    #[test]
    fn test_trained_model_predict() {
        // Create a model with degree 1 B-spline and a single internal knot (0.5)
        // Knot vector: [0, 0, 0.5, 1, 1]
        let knot_vector = array![0.0, 0.0, 0.5, 1.0, 1.0];
        let z_transform = Array2::<f64>::eye(2); // Identity matrix for no constraint

        // Simple model with intercept=0, main effect coeffs [2.0, 4.0]
        let model = TrainedModel {
            config: ModelConfig {
                link_function: LinkFunction::Identity,
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                pgs_basis_config: BasisConfig {
                    num_knots: 1,
                    degree: 1,
                }, // Degree 1 with 1 internal knot
                pc_basis_configs: vec![], // No PC effects in this simple test
                pgs_range: (0.0, 1.0),
                pc_ranges: vec![],
                pc_names: vec![],
                constraints: {
                    let mut constraints = HashMap::new();
                    constraints.insert(
                        "pgs_main".to_string(),
                        Constraint {
                            z_transform: z_transform.clone(),
                        },
                    );
                    constraints
                },
                knot_vectors: {
                    let mut knots = HashMap::new();
                    knots.insert("pgs".to_string(), knot_vector);
                    knots
                },
                num_pgs_interaction_bases: 0, // No interactions in this test
            },
            coefficients: MappedCoefficients {
                intercept: 0.0,
                main_effects: MainEffects {
                    pgs: vec![2.0, 4.0], // Two coefficients for degree 1 B-spline
                    pcs: HashMap::new(),
                },
                interaction_effects: HashMap::new(),
            },
            lambdas: vec![],
        };

        // Test point x = 0.25, which is halfway between knots at 0 and 0.5
        let test_point = array![0.25];
        let empty_pcs = Array2::<f64>::zeros((1, 0)); // No PCs

        // Calculate expected result by hand:
        // For x = 0.25, the basis functions will have values B_0,1 = 0.5, B_1,1 = 0.5, B_2,1 = 0.0
        // But the model ignores B_0,1 (first basis column) and only uses B_1,1 and B_2,1
        // Linear predictor: eta = (0.5 * 2.0) + (0.0 * 4.0) = 1.0
        let expected_value = 1.0;

        // Get the model prediction
        let prediction = model.predict(test_point.view(), empty_pcs.view()).unwrap();

        // Verify the result matches our ground truth
        assert_eq!(prediction.len(), 1);
        assert!(
            (prediction[0] - expected_value).abs() < 1e-10,
            "Prediction {} does not match expected value {}",
            prediction[0],
            expected_value
        );

        // Also test at x = 0.75, which is halfway between knots at 0.5 and 1.0
        // For x = 0.75, the basis functions will have values B_0,1 = 0.0, B_1,1 = 0.5, B_2,1 = 0.5
        // But the model ignores B_0,1 (first basis column) and only uses B_1,1 and B_2,1
        // Linear predictor: eta = (0.5 * 2.0) + (0.5 * 4.0) = 3.0
        let test_point_2 = array![0.75];
        let expected_value_2 = 3.0;

        let prediction_2 = model
            .predict(test_point_2.view(), empty_pcs.view())
            .unwrap();
        assert!(
            (prediction_2[0] - expected_value_2).abs() < 1e-10,
            "Prediction {} does not match expected value {}",
            prediction_2[0],
            expected_value_2
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
                pc_basis_configs: vec![BasisConfig {
                    num_knots: 2,
                    degree: 1,
                }],
                pgs_range: (-2.0, 2.0),
                pc_ranges: vec![(-1.0, 1.0)],
                pc_names: vec!["PC1".to_string()],
                constraints: HashMap::new(),
                knot_vectors: HashMap::new(),
                num_pgs_interaction_bases: 1, // 1 PGS interaction base for this test
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

                // PGS_B1 interactions
                let mut pgs_b1 = HashMap::new();
                pgs_b1.insert("PC1".to_string(), vec![8.0, 9.0]);
                pgs_b1.insert("PC2".to_string(), vec![10.0, 11.0]);

                // PGS_B2 interactions
                let mut pgs_b2 = HashMap::new();
                pgs_b2.insert("PC1".to_string(), vec![12.0, 13.0]);
                pgs_b2.insert("PC2".to_string(), vec![14.0, 15.0]);

                interactions.insert("PGS_B1".to_string(), pgs_b1);
                interactions.insert("PGS_B2".to_string(), pgs_b2);
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
            pc_basis_configs: vec![
                BasisConfig {
                    num_knots: 3,
                    degree: 3,
                },
                BasisConfig {
                    num_knots: 3,
                    degree: 3,
                },
            ],
            pgs_range: (0.0, 1.0),
            pc_ranges: vec![(0.0, 1.0), (0.0, 1.0)],
            pc_names: vec!["PC1".to_string(), "PC2".to_string()], // Order matters for flattening
            constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            num_pgs_interaction_bases: 2, // 2 PGS interaction bases for this test
        };

        // Use the internal flatten_coefficients function
        use super::internal::flatten_coefficients;
        let flattened = flatten_coefficients(&coeffs, &config);

        // Verify the canonical order of flattening:
        // 1. Intercept
        // 2. PC main effects (ordered by config.pc_names)
        // 3. PGS main effects
        // 4. Interaction effects (PGS basis function 1, then PC1, PC2, etc.; then PGS basis function 2...)

        // Define expected order based on the canonical ordering rules
        let expected = vec![
            1.0, // Intercept
            4.0, 5.0, // PC1 main effects
            6.0, 7.0, // PC2 main effects
            2.0, 3.0, // PGS main effects
            8.0, 9.0, // PGS_B1 * PC1 interaction
            10.0, 11.0, // PGS_B1 * PC2 interaction
            12.0, 13.0, // PGS_B2 * PC1 interaction
            14.0, 15.0, // PGS_B2 * PC2 interaction
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
            pc_basis_configs: vec![pc1_basis_config.clone()],
            pgs_range: (-1.0, 1.0),
            pc_ranges: vec![(-0.5, 0.5)],
            pc_names: vec!["PC1".to_string()],
            constraints: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            knot_vectors: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            num_pgs_interaction_bases: 3, // 3 PGS interaction bases for this test
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
        };

        // Generate the correct constraints and structure using the actual model-building code
        let (_, _, layout, constraints, knot_vectors) =
            crate::calibrate::construction::build_design_and_penalty_matrices(
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
                pc_basis_configs: vec![pc1_basis_config.clone()],
                pgs_range: (-1.0, 1.0),
                pc_ranges: vec![(-0.5, 0.5)],
                pc_names: vec!["PC1".to_string()],
                constraints: constraints.clone(), // Clone to avoid ownership issues
                knot_vectors, // Use the knot vectors generated by the model-building code
                num_pgs_interaction_bases: 3, // 3 PGS interaction bases for this test
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
                        // Extract PC1 dimension before using constraints
                        let pc1_dim = if let Some(pc1_constraint) = constraints.get("PC1") {
                            pc1_constraint.z_transform.ncols()
                        } else {
                            9 // Default fallback
                        };

                        let mut pc_map = HashMap::new();
                        pc_map.insert("PC1".to_string(), (1..=pc1_dim).map(|i| i as f64).collect());
                        pc_map
                    },
                },
                interaction_effects: {
                    // Extract PC1 dimension before creating interactions
                    let pc1_dim = if let Some(pc1_constraint) = constraints.get("PC1") {
                        pc1_constraint.z_transform.ncols()
                    } else {
                        9 // Default fallback
                    };

                    // For the interactions, we need to use pgs_main_basis_unc dimensions
                    // which is the number of unconstrained basis functions excluding intercept
                    // This will typically be pgs_basis_config.num_knots + pgs_basis_config.degree
                    let num_pgs_basis_funcs = 6 + 3; // From pgs_basis_config

                    let mut interactions = HashMap::new();
                    // Build interaction terms for each PGS basis function
                    for i in 1..=num_pgs_basis_funcs {
                        let mut pgs_bx = HashMap::new();
                        // Create coefficients for each PC dimension
                        pgs_bx.insert(
                            "PC1".to_string(),
                            (1..=pc1_dim).map(|j| (i * 10 + j) as f64).collect(),
                        );
                        interactions.insert(format!("PGS_B{}", i), pgs_bx);
                    }
                    interactions
                },
            },
            lambdas: vec![0.1, 0.2],
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

        // 1. Check model configuration
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
        assert_eq!(loaded_model.config.pc_names, original_model.config.pc_names);
        assert_eq!(
            loaded_model.config.pgs_range,
            original_model.config.pgs_range
        );
        assert_eq!(
            loaded_model.config.pc_ranges,
            original_model.config.pc_ranges
        );

        // 2. Check constraint transformations exist
        assert!(loaded_model.config.constraints.contains_key("pgs_main"));
        assert!(loaded_model.config.constraints.contains_key("PC1"));

        // 3. Check coefficient values
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

        // 4. Check interaction effects
        assert_eq!(
            loaded_model.coefficients.interaction_effects.len(),
            original_model.coefficients.interaction_effects.len()
        );

        // Check PGS_B1 interactions
        if let (Some(pgs_b1_loaded), Some(pgs_b1_orig)) = (
            loaded_model.coefficients.interaction_effects.get("PGS_B1"),
            original_model
                .coefficients
                .interaction_effects
                .get("PGS_B1"),
        ) {
            assert_eq!(pgs_b1_loaded.len(), pgs_b1_orig.len());
            if let (Some(pc1_loaded), Some(pc1_orig)) =
                (pgs_b1_loaded.get("PC1"), pgs_b1_orig.get("PC1"))
            {
                assert_eq!(pc1_loaded, pc1_orig);
            } else {
                panic!("Missing PC1 in PGS_B1 interaction effects");
            }
        } else {
            panic!("Missing PGS_B1 interaction effects");
        }

        // Check PGS_B2 interactions
        if let (Some(pgs_b2_loaded), Some(pgs_b2_orig)) = (
            loaded_model.coefficients.interaction_effects.get("PGS_B2"),
            original_model
                .coefficients
                .interaction_effects
                .get("PGS_B2"),
        ) {
            assert_eq!(pgs_b2_loaded.len(), pgs_b2_orig.len());
            if let (Some(pc1_loaded), Some(pc1_orig)) =
                (pgs_b2_loaded.get("PC1"), pgs_b2_orig.get("PC1"))
            {
                assert_eq!(pc1_loaded, pc1_orig);
            } else {
                panic!("Missing PC1 in PGS_B2 interaction effects");
            }
        } else {
            panic!("Missing PGS_B2 interaction effects");
        }

        // 5. Check lambdas
        assert_eq!(loaded_model.lambdas, original_model.lambdas);

        // 6. Test that we can use the loaded model for prediction
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

    for block in &layout.penalty_map {
        let coeffs = beta.slice(s![block.col_range.clone()]).to_vec();

        // This logic is now driven entirely by the term_name established in the layout
        match block.term_name.as_str() {
            name if name.starts_with("f(PC") => {
                let pc_name = name.replace("f(", "").replace(")", "");
                pcs.insert(pc_name, coeffs);
            }
            name if name.starts_with("f(PGS_B") => {
                let parts: Vec<_> = name.split([',', ')']).collect();
                if parts.len() < 2 {
                    continue;
                }
                let pgs_key = parts[0].replace("f(", "").to_string();
                let pc_name = parts[1].trim().to_string();
                interaction_effects
                    .entry(pgs_key)
                    .or_insert_with(HashMap::new)
                    .insert(pc_name, coeffs);
            }
            _ => {
                return Err(EstimationError::LayoutError(format!(
                    "Unknown term name in layout during coefficient mapping: {}",
                    block.term_name
                )));
            }
        }
    }

    Ok(MappedCoefficients {
        intercept,
        main_effects: MainEffects { pgs, pcs },
        interaction_effects,
    })
}
