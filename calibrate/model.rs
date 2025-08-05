use crate::calibrate::basis::{self};
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
                // Clamp probabilities away from 0 and 1 to prevent numerical instability
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

    /// Constructs the design matrix `X` for new data following a strict canonical order.
    /// This order is the implicit contract that allows the flattened coefficients to work correctly.
    pub(super) fn construct_design_matrix(
        p_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        config: &ModelConfig,
    ) -> Result<Array2<f64>, ModelError> {
        // 1. Generate basis for PGS using saved knot vector if available
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
            let saved_knots = config
                .knot_vectors
                .get(pc_name)
                .ok_or_else(|| ModelError::InternalStackingError)?;

            let (pc_basis_unc, _) = basis::create_bspline_basis_with_knots(
                pc_col,
                saved_knots.view(),
                config.pc_basis_configs[i].degree,
            )?;

            // Slice the basis to remove the intercept term, just like in the training code
            let pc_main_basis_unc = pc_basis_unc.slice(s![.., 1..]);

            // Apply the SAVED PC constraint
            let pc_name = &config.pc_names[i];
            let pc_z = &config
                .constraints
                .get(pc_name)
                .ok_or_else(|| ModelError::ConstraintMissing(pc_name.clone()))?
                .z_transform;

            // Check that dimensions match before matrix multiplication
            if pc_main_basis_unc.ncols() != pc_z.nrows() {
                return Err(ModelError::InternalStackingError);
            }

            pc_constrained_bases.push(pc_main_basis_unc.dot(pc_z)); // Now constrained
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

        // 4. Tensor product interaction effects
        // Create unified interaction surfaces using row-wise tensor products
        for (pc_idx, _pc_name) in config.pc_names.iter().enumerate() {
            let pc_basis_con = &pc_constrained_bases[pc_idx];
            
            // Create tensor product interaction using the same logic as training
            let tensor_interaction = row_wise_tensor_product(&pgs_main_basis_unc.to_owned(), pc_basis_con);
            
            // Add all columns from this tensor product to the design matrix
            for col in tensor_interaction.axis_iter(Axis(1)) {
                owned_cols.push(col.to_owned());
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

        // 4. Tensor product interaction effects (ordered by PC)
        for pc_name in &config.pc_names {
            let tensor_key = format!("f(PGS,{})", pc_name);
            if let Some(tensor_coeffs) = coeffs.interaction_effects.get(&tensor_key) {
                flattened.extend_from_slice(tensor_coeffs);
            }
        }

        Array1::from_vec(flattened)
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
            basis::apply_sum_to_zero_constraint(unconstrained_main_basis.view()).unwrap();

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
                pc_basis_configs: vec![],
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
                    knots.insert("pgs".to_string(), knot_vector.clone());
                    knots
                },
                num_pgs_interaction_bases: 0,
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
        };

        // --- Define Test Points ---
        let test_points = array![0.25, 0.75];
        let empty_pcs = Array2::<f64>::zeros((2, 0));

        // --- Calculate the expected result CORRECTLY ---
        // The manual calculation was flawed. Here's the correct way to derive the ground truth:
        // 1. Generate the raw, unconstrained basis at the test points.
        let (full_basis_unc, _) =
            basis::create_bspline_basis_with_knots(test_points.view(), knot_vector.view(), degree)
                .unwrap();

        // 2. Isolate the main effect part of the basis (all columns except the intercept).
        let pgs_main_basis_unc = full_basis_unc.slice(s![.., 1..]);

        // 3. Apply the same sum-to-zero constraint transformation.
        let pgs_main_basis_con = pgs_main_basis_unc.dot(&z_transform);

        // 4. Get the coefficients for the constrained basis.
        let coeffs = Array1::from(model.coefficients.main_effects.pgs.clone());

        // 5. Calculate the final expected linear predictor: intercept + constrained_basis * coeffs
        let expected_values = model.coefficients.intercept + pgs_main_basis_con.dot(&coeffs);

        // Get the model's prediction using the actual `predict` method
        let predictions = model.predict(test_points.view(), empty_pcs.view()).unwrap();

        // Verify the results match our correctly calculated ground truth
        assert_eq!(predictions.len(), 2);
        assert_abs_diff_eq!(predictions, expected_values, epsilon = 1e-10);
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
            num_pgs_interaction_bases: 2 // 2 PGS interaction bases for this test
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
            pc_basis_configs: vec![pc1_basis_config.clone()],
            pgs_range: (-1.0, 1.0),
            pc_ranges: vec![(-0.5, 0.5)],
            pc_names: vec!["PC1".to_string()],
            constraints: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            knot_vectors: HashMap::new(), // Will be populated by build_design_and_penalty_matrices
            num_pgs_interaction_bases: 3 // 3 PGS interaction bases for this test
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
                num_pgs_interaction_bases: 3 // 3 PGS interaction bases for this test
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
                        // Create flattened vector for interaction term
                        let interaction_coeffs: Vec<f64> = (1..=pc1_dim).map(|j| (i * 10 + j) as f64).collect();
                        interactions.insert(format!("f(PGS,PC1)_B{}", i), interaction_coeffs);
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

        // Check for interaction effect f(PGS,PC1)_B1
        let key_b1 = "f(PGS,PC1)_B1";
        if let (Some(b1_loaded), Some(b1_orig)) = (
            loaded_model.coefficients.interaction_effects.get(key_b1),
            original_model.coefficients.interaction_effects.get(key_b1),
        ) {
            assert_eq!(b1_loaded, b1_orig, "Mismatch in {}", key_b1);
        } else {
            panic!("Missing {} interaction effect", key_b1);
        }

        // Check for interaction effect f(PGS,PC1)_B2
        let key_b2 = "f(PGS,PC1)_B2";
        if let (Some(b2_loaded), Some(b2_orig)) = (
            loaded_model.coefficients.interaction_effects.get(key_b2),
            original_model.coefficients.interaction_effects.get(key_b2),
        ) {
            assert_eq!(b2_loaded, b2_orig, "Mismatch in {}", key_b2);
        } else {
            panic!("Missing {} interaction effect", key_b2);
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
            name if name.starts_with("f(PGS,") => {
                // Tensor product interaction: f(PGS,PC1) -> direct storage
                interaction_effects.insert(name.to_string(), coeffs);
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
