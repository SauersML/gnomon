use crate::calibrate::basis::{self, create_bspline_basis};
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
    /// Knot vectors used during training, saved for exact reproduction during prediction
    #[serde(default)] // For backward compatibility with old models that don't have this field
    pub knot_vectors: HashMap<String, Array1<f64>>,
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
            LinkFunction::Logit => eta.mapv(|e| 1.0 / (1.0 + f64::exp(-e))), // Sigmoid
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
            basis::create_bspline_basis_with_knots(
                p_new,
                saved_knots.view(),
                config.pgs_basis_config.degree,
            )?
        } else {
            // Fallback to original method for backward compatibility
            create_bspline_basis(
                p_new,
                None,
                config.pgs_range,
                config.pgs_basis_config.num_knots,
                config.pgs_basis_config.degree,
            )?
        };

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

        // This closure was not used - removed
        // Previously defined a helper to fetch Z transform from model

        // 2. Generate bases for PCs using saved knot vectors if available
        let mut pc_constrained_bases = Vec::new();
        for i in 0..config.pc_names.len() {
            let pc_col = pcs_new.column(i);
            let pc_name = &config.pc_names[i];
            let (pc_basis_unc, _) = if let Some(saved_knots) = config.knot_vectors.get(pc_name) {
                // Use saved knot vector for exact reproduction
                basis::create_bspline_basis_with_knots(
                    pc_col,
                    saved_knots.view(),
                    config.pc_basis_configs[i].degree,
                )?
            } else {
                // Fallback to original method for backward compatibility
                create_bspline_basis(
                    pc_col,
                    None,
                    config.pc_ranges[i],
                    config.pc_basis_configs[i].num_knots,
                    config.pc_basis_configs[i].degree,
                )?
            };

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
        // CORRECTED: Loop over the UNCONSTRAINED non-intercept PGS basis
        for m in 0..pgs_main_basis_unc.ncols() {
            // CORRECTED: Use m directly to index the unconstrained basis
            let pgs_weight_col = pgs_main_basis_unc.column(m);
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
        // The correct formula for unconstrained non-intercept PGS basis functions is:
        // total_bases - 1 = (num_knots + degree + 1) - 1 = num_knots + degree
        // We subtract 1 to exclude the intercept basis function (index 0)
        let total_pgs_bases = config.pgs_basis_config.num_knots + config.pgs_basis_config.degree;
        for m in 1..=total_pgs_bases {
            let pgs_key = format!("PGS_B{}", m);
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
        // Create a simple model for testing
        let original_model = TrainedModel {
            config: ModelConfig {
                link_function: LinkFunction::Logit,
                penalty_order: 2,
                convergence_tolerance: 1e-6,
                max_iterations: 100,
                reml_convergence_tolerance: 1e-6,
                reml_max_iterations: 50,
                pgs_basis_config: BasisConfig {
                    num_knots: 6,
                    degree: 3,
                },
                pc_basis_configs: vec![BasisConfig {
                    num_knots: 6,
                    degree: 3,
                }],
                pgs_range: (-1.0, 1.0),
                pc_ranges: vec![(-0.5, 0.5)],
                pc_names: vec!["PC1".to_string()],
                constraints: {
                    let mut constraints = HashMap::new();
                    constraints.insert(
                        "pgs_main".to_string(),
                        Constraint {
                            z_transform: Array2::eye(2),
                        },
                    );
                    constraints.insert(
                        "PC1".to_string(),
                        Constraint {
                            z_transform: Array2::eye(2),
                        },
                    );
                    constraints
                },
                knot_vectors: {
                    let mut knots = HashMap::new();
                    knots.insert(
                        "pgs".to_string(),
                        Array1::from_vec(vec![
                            -1.0, -1.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0,
                        ]),
                    );
                    knots.insert(
                        "PC1".to_string(),
                        Array1::from_vec(vec![
                            -0.5, -0.5, -0.5, -0.25, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5,
                        ]),
                    );
                    knots
                },
            },
            coefficients: MappedCoefficients {
                intercept: 0.5,
                main_effects: MainEffects {
                    // Correctly sized vector for a basis with num_knots=6, degree=3, with sum-to-zero constraint
                    // (6+3+1-1)-1 = 8 constrained basis functions
                    pgs: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    pcs: {
                        let mut pc_map = HashMap::new();
                        // Correctly sized vector for a basis with num_knots=6, degree=3, with sum-to-zero constraint
                        // (6+3+1)-1 = 9 constrained basis functions
                        pc_map.insert(
                            "PC1".to_string(),
                            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                        );
                        pc_map
                    },
                },
                interaction_effects: {
                    let mut interactions = HashMap::new();

                    // Build correct interaction terms - one for each PGS basis function (minus intercept)
                    // For each PGS basis function (PGS_B1 through PGS_B9), create an interaction with PC1
                    for i in 1..=9 {
                        let mut pgs_bx = HashMap::new();
                        // Same size as PC1 constrained basis (9 values)
                        let pc1_coefs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
                        pgs_bx.insert("PC1".to_string(), pc1_coefs);
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
