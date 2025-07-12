//! # Main Application Logic for PGS Calibration
//!
//! This file contains the command-line interface (CLI) and the main entry
//! points for training a calibration model and using it for inference.
//!
//! The `train` command orchestrates a full REML/LAML (Restricted Maximum
//! Likelihood / Laplace Approximate Marginal Likelihood) optimization to automatically
//! find the optimal smoothing parameters for the GAM. This is a significant change
//! from the previous version which required manual specification of a fixed lambda.
//!
//! The `infer` command remains the same, loading a trained model artifact
//! (which contains the REML-estimated parameters) and applying it to new data.

use basis::BasisConfig;
use data::{load_prediction_data, load_training_data};
use std::collections::HashMap;
use estimate::train_model;
use model::{LinkFunction, ModelConfig, TrainedModel};

use clap::{Parser, Subcommand};
use ndarray::{Array1, ArrayView1};
use std::collections::HashSet;
use std::process;

// Module declarations
mod basis;
mod data;
mod estimate;
mod model;

/// Defines the CLI structure using the `clap` crate.
#[derive(Parser)]
#[command(
    name = "gnomon-calibrate",
    about = "Train and apply GAM models for polygenic score calibration.",
    long_about = "A tool for training Generalized Additive Models (GAMs) to calibrate polygenic scores \
                  using B-spline basis functions with smoothing penalties estimated by REML."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

/// Defines the available subcommands: `train` and `infer`.
#[derive(Subcommand)]
enum Commands {
    /// Train a new GAM model from training data. Smoothing parameters are estimated automatically.
    #[command(about = "Train a GAM model via REML (outputs: model.toml)")]
    Train {
        /// Path to training TSV file with phenotype,score,PC1,PC2,... columns.
        #[arg(long)]
        training_data: String,

        /// Number of principal components to use from the data.
        #[arg(long, value_name = "N")]
        num_pcs: usize,

        /// Number of internal knots for the PGS spline basis.
        #[arg(long, default_value = "10")]
        pgs_knots: usize,

        /// Polynomial degree for the PGS spline basis.
        #[arg(long, default_value = "3")]
        pgs_degree: usize,

        /// Number of internal knots for all PC spline bases.
        #[arg(long, default_value = "5")]
        pc_knots: usize,

        /// Polynomial degree for all PC spline bases.
        #[arg(long, default_value = "2")]
        pc_degree: usize,

        /// Order of the difference penalty matrix for all smooth terms.
        #[arg(long, default_value = "2")]
        penalty_order: usize,

        /// Maximum number of P-IRLS iterations for the inner loop (per REML step).
        #[arg(long, default_value = "50")]
        max_iterations: usize,

        /// Convergence tolerance for the P-IRLS inner loop deviance change.
        #[arg(long, default_value = "1e-7")]
        convergence_tolerance: f64,
        
        /// Maximum number of iterations for the outer REML/BFGS optimization loop.
        #[arg(long, default_value = "100")]
        reml_max_iterations: u64,

        /// Convergence tolerance for the gradient norm in the outer REML/BFGS loop.
        #[arg(long, default_value = "1e-3")]
        reml_convergence_tolerance: f64,

        /// Path to output the trained model file.
        #[arg(long, default_value = "model.toml")]
        output_path: String,
    },

    /// Apply a trained model to new data for prediction.
    #[command(about = "Apply a trained model to new data (outputs: predictions.tsv)")]
    Infer {
        /// Path to test TSV file with score,PC1,PC2,... columns (no phenotype needed).
        #[arg(long)]
        test_data: String,

        /// Path to the trained model file (`model.toml`).
        #[arg(long)]
        model: String,

        /// Path to output the predictions file.
        #[arg(long, default_value = "predictions.tsv")]
        output_path: String,
    },
}

/// Main application entry point.
fn main() {
    // Initialize a simple logger. Use `RUST_LOG=info` environment variable to see logs.
    env_logger::init();
    let cli = Cli::parse();

    let result = match cli.command {
        // --- Train Command ---
        // Note the absence of `lambda`. It is no longer a user-provided argument.
        Commands::Train {
            training_data,
            num_pcs,
            pgs_knots,
            pgs_degree,
            pc_knots,
            pc_degree,
            penalty_order,
            max_iterations,
            convergence_tolerance,
            reml_max_iterations,
            reml_convergence_tolerance,
            output_path,
        } => train_command(
            &training_data,
            num_pcs,
            pgs_knots,
            pgs_degree,
            pc_knots,
            pc_degree,
            penalty_order,
            max_iterations,
            convergence_tolerance,
            reml_max_iterations,
            reml_convergence_tolerance,
            &output_path,
        ),
        Commands::Infer { test_data, model, output_path } => infer_command(&test_data, &model, &output_path),
    };

    if let Err(e) = result {
        // Use the debug formatter for the error type for more detailed output.
        eprintln!("Error: {:?}", e);
        process::exit(1);
    }
}

/// Handles the `train` command logic.
fn train_command(
    training_data_path: &str,
    num_pcs: usize,
    pgs_knots: usize,
    pgs_degree: usize,
    pc_knots: usize,
    pc_degree: usize,
    penalty_order: usize,
    max_iterations: usize,
    convergence_tolerance: f64,
    reml_max_iterations: u64,
    reml_convergence_tolerance: f64,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Loading training data from: {}", training_data_path);

    let data = load_training_data(training_data_path, num_pcs)?;
    log::info!(
        "Loaded {} samples with {} PCs.",
        data.y.len(),
        data.pcs.ncols()
    );

    // Auto-detect link function based on phenotype (e.g., binary -> logit).
    let link_function = detect_link_function(&data.y);
    log::info!("Auto-detected link function: {:?}", link_function);

    // Calculate data ranges from the training data. This is crucial for creating
    // consistent basis functions during both training and prediction.
    let pgs_range = calculate_range(data.p.view());
    let pc_ranges: Vec<(f64, f64)> = (0..data.pcs.ncols())
        .map(|i| calculate_range(data.pcs.column(i)))
        .collect();

    log::info!("PGS range: ({:.3}, {:.3})", pgs_range.0, pgs_range.1);
    for (i, range) in pc_ranges.iter().enumerate() {
        log::info!("PC{} range: ({:.3}, {:.3})", i + 1, range.0, range.1);
    }
    
    // Generate PC names for the configuration.
    let pc_names: Vec<String> = (1..=num_pcs).map(|i| format!("PC{}", i)).collect();

    // Create basis configurations from CLI arguments.
    let pgs_basis_config = BasisConfig {
        num_knots: pgs_knots,
        degree: pgs_degree,
    };
    let pc_basis_configs = vec![
        BasisConfig {
            num_knots: pc_knots,
            degree: pc_degree,
        };
        num_pcs
    ];

    // --- Assemble the final ModelConfig ---
    // This struct contains all hyperparameters needed to define the model
    // structure and control the optimization process.
    let config = ModelConfig {
        link_function,
        penalty_order,
        convergence_tolerance,
        max_iterations,
        reml_convergence_tolerance,
        reml_max_iterations,
        pgs_basis_config,
        pc_basis_configs,
        pgs_range,
        pc_ranges,
        pc_names,
        constraints: HashMap::new(), // Will be filled during training
        knot_vectors: HashMap::new(), // Will be filled during training  
        num_pgs_interaction_bases: 0, // Will be set during training
    };

    // --- Train the model using REML ---
    // This function call triggers the full REML optimization pipeline in `estimate.rs`.
    // It will automatically find the optimal smoothing parameters (lambdas).
    log::info!("Starting model training with REML/LAML optimization...");
    let trained_model = train_model(&data, &config)?;

    // Save the final, trained model to disk.
    trained_model.save(output_path)?;
    log::info!("Model training complete. Saved to: {}", output_path);

    Ok(())
}

/// Handles the `infer` command logic.
fn infer_command(test_data_path: &str, model_path: &str, output_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    log::info!("Loading model from: {}", model_path);

    // Load the trained model from the specified TOML file.
    let model = TrainedModel::load(model_path)?;
    let num_pcs = model.config.pc_names.len();
    log::info!("Model expects {} PCs.", num_pcs);

    // Load the new data for which predictions are required.
    log::info!("Loading test data from: {}", test_data_path);
    let data = load_prediction_data(test_data_path, num_pcs)?;
    log::info!("Loaded {} samples for prediction.", data.p.len());

    // Generate predictions using the loaded model.
    log::info!("Generating predictions...");
    let predictions = model.predict(data.p.view(), data.pcs.view())?;

    // Save the predictions to a TSV file.
    save_predictions(&predictions, output_path)?;
    log::info!("Predictions saved to: {}", output_path);

    Ok(())
}

/// Auto-detects the appropriate link function based on the number of unique phenotype values.
fn detect_link_function(phenotype: &ArrayView1<f64>) -> LinkFunction {
    let unique_values: HashSet<_> = phenotype.iter().map(|&x| x.to_bits()).collect();

    if unique_values.len() == 2 {
        // Binary case/control data (e.g., 0 and 1)
        LinkFunction::Logit
    } else {
        // Continuous quantitative trait
        LinkFunction::Identity
    }
}

/// Calculates the min/max range of a data vector.
fn calculate_range(data: ArrayView1<f64>) -> (f64, f64) {
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    (min_val, max_val)
}

/// Saves predictions to a simple one-column TSV file.
fn save_predictions(predictions: &Array1<f64>, output_path: &str) -> Result<(), std::io::Error> {
    use std::io::Write;
    let mut file = std::fs::File::create(output_path)?;
    writeln!(file, "prediction")?;
    for &pred in predictions.iter() {
        writeln!(file, "{:.6}", pred)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TrainedModel;
    use ndarray::{Array1, Array2};
    use std::collections::HashMap;
    use std::io::{self, Write};
    use tempfile::{NamedTempFile, TempDir};
    
    /// Integration test that simulates the entire calibration workflow with realistic data
    #[test]
    fn test_calibration_integration() -> Result<(), Box<dyn std::error::Error>> {
        // Create a temporary directory for our test files
        let temp_dir = TempDir::new()?;
        let training_file_path = temp_dir.path().join("training_data.tsv");
        let test_file_path = temp_dir.path().join("test_data.tsv");
        let model_file_path = temp_dir.path().join("model.toml");
        let predictions_file_path = temp_dir.path().join("predictions.tsv");
        
        println!("\n===== RUNNING COMPREHENSIVE INTEGRATION TEST =====\n");
        println!("1. Generating synthetic data with complex relationships...");
        
        // Generate synthetic data with known relationships
        let (training_data, test_data, true_effects) = generate_realistic_dataset()?;
        
        // Prepare data files
        println!("2. Preparing data files...");
        let (train_rows, test_rows) = prepare_data_files(
            &training_data, &test_data, 
            &training_file_path, &test_file_path
        )?;
        
        println!("   - Training data: {} samples ({} cases, {} controls)", 
                 train_rows,
                 train_rows * true_effects.case_control_ratio as usize,
                 train_rows * (1.0 - true_effects.case_control_ratio) as usize);
        println!("   - Test data: {} samples", test_rows);
        
        // Set up parameters for model training - try two different configurations
        println!("\n3. Running model training with two different configurations...");
        
        let configurations = vec![
            // Config 1: Standard settings
            (
                "Standard",
                10, // pgs_knots
                3,  // pgs_degree
                5,  // pc_knots
                2,  // pc_degree
                2,  // penalty_order
            ),
            // Config 2: More flexible spline basis
            (
                "Flexible", 
                15, // pgs_knots
                3,  // pgs_degree
                8,  // pc_knots
                3,  // pc_degree
                2,  // penalty_order
            )
        ];
        
        // Common parameters
        let num_pcs = 5; // Use 5 PCs
        let max_iterations = 50;
        let convergence_tolerance = 1e-7;
        let reml_max_iterations = 100;
        let reml_convergence_tolerance = 1e-3;
        
        let mut model_files = Vec::new();
        let mut prediction_files = Vec::new();
        
        // Run training and inference for each configuration
        for (i, (config_name, pgs_knots, pgs_degree, pc_knots, pc_degree, penalty_order)) in configurations.iter().enumerate() {
            let model_file = temp_dir.path().join(format!("model_{}.toml", i));
            let predictions_file = temp_dir.path().join(format!("predictions_{}.tsv", i));
            
            println!("\n   Configuration {}: {}", i+1, config_name);
            println!("   - PGS basis: {} knots, degree {}", pgs_knots, pgs_degree);
            println!("   - PC basis: {} knots, degree {}", pc_knots, pc_degree);
            println!("   - Penalty order: {}", penalty_order);
            
            // Train the model with the temporary output path
            let train_result = train_command(
                training_file_path.to_str().unwrap(),
                num_pcs,
                *pgs_knots,
                *pgs_degree,
                *pc_knots,
                *pc_degree,
                *penalty_order,
                max_iterations,
                convergence_tolerance,
                reml_max_iterations,
                reml_convergence_tolerance,
                model_file.to_str().unwrap(),
            );
            assert!(train_result.is_ok(), "Model training failed for config {}: {:?}", config_name, train_result);
            
            // Verify the model file was created
            assert!(model_file.exists(), "Model file wasn't created for config {}", config_name);
            
            // Run inference on test data with the temporary output path
            let infer_result = infer_command(
                test_file_path.to_str().unwrap(),
                model_file.to_str().unwrap(),
                predictions_file.to_str().unwrap(),
            );
            assert!(infer_result.is_ok(), "Model inference failed for config {}: {:?}", config_name, infer_result);
            
            // Verify predictions file was created
            assert!(predictions_file.exists(), "Predictions file wasn't created for config {}", config_name);
            
            model_files.push(model_file.clone());
            prediction_files.push(predictions_file.clone());
        }
        
        println!("\n4. Loading predictions and validating models...");
        
        // Load predictions from both models
        let predictions1 = load_predictions(prediction_files[0].to_str().unwrap())?;
        let predictions2 = load_predictions(prediction_files[1].to_str().unwrap())?;
        
        // Validate each model's predictions
        println!("\n===== VALIDATION RESULTS FOR MODEL 1 =====");
        let metrics1 = validate_predictions(&predictions1, &test_data, &true_effects);
        
        println!("\n===== VALIDATION RESULTS FOR MODEL 2 =====");
        let metrics2 = validate_predictions(&predictions2, &test_data, &true_effects);
        
        // Compare the two models
        println!("\n===== COMPARING MODELS =====");
        println!("Model 1 (Standard) vs. Model 2 (Flexible):");
        println!("   - Correlation with ground truth: {:.3} vs {:.3}", metrics1.correlation, metrics2.correlation);
        println!("   - R² coefficient: {:.3} vs {:.3}", metrics1.r_squared, metrics2.r_squared);
        println!("   - AUC: {:.3} vs {:.3}", metrics1.auc, metrics2.auc);
        println!("   - Brier score: {:.3} vs {:.3}", metrics1.brier_score, metrics2.brier_score);
        println!("   - Mean calibration error: {:.3} vs {:.3}", metrics1.calibration_error, metrics2.calibration_error);
        
        // Test for consistency between models
        let consistency_correlation = calculate_correlation(&predictions1, &predictions2);
        println!("\nModel consistency correlation: {:.3}", consistency_correlation);
        assert!(consistency_correlation > 0.9, "Models produce inconsistent results: correlation = {}", consistency_correlation);
        
        // Create a combined prediction (ensemble)
        let ensemble_predictions: Vec<f64> = predictions1.iter()
            .zip(predictions2.iter())
            .map(|(&p1, &p2)| (p1 + p2) / 2.0)
            .collect();
            
        // Validate ensemble predictions
        println!("\n===== VALIDATION RESULTS FOR ENSEMBLE MODEL =====");
        let ensemble_metrics = validate_predictions(&ensemble_predictions, &test_data, &true_effects);
        
        // Check if ensemble outperforms individual models
        println!("\n===== ENSEMBLE VS INDIVIDUAL MODELS =====");
        println!("Metrics: Ensemble vs Best Individual Model");
        println!("   - R²: {:.3} vs {:.3}", 
                 ensemble_metrics.r_squared, 
                 metrics1.r_squared.max(metrics2.r_squared));
        println!("   - AUC: {:.3} vs {:.3}", 
                 ensemble_metrics.auc, 
                 metrics1.auc.max(metrics2.auc));
        
        // Extract TOML model content and analyze coefficients
        println!("\n5. Analyzing model structures and validating coefficients...");
        
        // --- Analyze Model 1 ---
        let model1_content = std::fs::read_to_string(model_files[0].to_str().unwrap())?;
        let model1: TrainedModel = toml::from_str(&model1_content)
            .expect("Failed to parse model 1 TOML file into TrainedModel struct");
        
        // We can still use the check_model_properties logic with the parsed model
        let model1_json: serde_json::Value = toml::from_str(&model1_content)?;
        
        // --- Analyze Model 2 ---
        let model2_content = std::fs::read_to_string(model_files[1].to_str().unwrap())?;
        let model2: TrainedModel = toml::from_str(&model2_content)
            .expect("Failed to parse model 2 TOML file into TrainedModel struct");
        
        let model2_json: serde_json::Value = toml::from_str(&model2_content)?;
        
        // Check for key model properties using the existing function
        check_model_properties(&model1_json, &model2_json);
        
        // Call our coefficient validation functions for both models
        validate_coefficient_structure(&model1, &true_effects, "Standard");
        validate_coefficient_structure(&model2, &true_effects, "Flexible");
        
        // Call our smooth function shape validation for both models
        validate_smooth_function_shapes(&model1, &true_effects);
        validate_smooth_function_shapes(&model2, &true_effects);
        
        println!("\n===== INTEGRATION TEST COMPLETED SUCCESSFULLY =====");
        
        Ok(())
    }
    
    #[test]
    fn test_reml_lambda_sensitivity_to_noise() -> Result<(), Box<dyn std::error::Error>> {
        // Create synthetic datasets with controlled noise levels
        println!("\n===== TESTING REML LAMBDA SENSITIVITY TO NOISE =====");
        
        // Set reproducible random seed
        use rand::{SeedableRng, Rng};
        use rand_distr::Normal;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let normal = Normal::new(0.0, 1.0).unwrap();
        
        // 1. Generate base dataset
        let n_samples = 300;
        let n_pcs = 3;
        
        // Create PGS values
        let pgs = Array1::from_vec((0..n_samples)
            .map(|_| rng.gen_range(-2.0..2.0))
            .collect());
        
        // Create PC values
        let mut pcs = Array2::zeros((n_samples, n_pcs));
        for i in 0..n_samples {
            for j in 0..n_pcs {
                pcs[[i, j]] = normal.sample(&mut rng);
            }
        }
        
        // Generate true relationship with known effects
        let mut eta = Array1::zeros(n_samples);
        
        // Intercept
        eta.fill(-1.0);
        
        // PGS main effect
        for i in 0..n_samples {
            eta[i] += 0.8 * pgs[i];
        }
        
        // PC1: Strong non-linear effect
        for i in 0..n_samples {
            let x = pcs[[i, 0]];
            eta[i] += 0.6 * x + 0.2 * x.powi(2) - 0.1 * x.powi(3);
        }
        
        // PC2: Moderate effect
        for i in 0..n_samples {
            eta[i] += 0.3 * pcs[[i, 1]];
        }
        
        // PC3: No effect
        
        // Add base noise
        let base_noise = 0.4;
        for i in 0..n_samples {
            eta[i] += base_noise * normal.sample(&mut rng);
        }
        
        // Convert to binary outcome
        let y = eta.mapv(|x| 1.0 / (1.0 + (-x).exp()))
                   .mapv(|p| if rng.gen::<f64>() < p { 1.0 } else { 0.0 });
        
        // 2. Create noisy version of PC1 with three noise components
        let mut pcs_noisy = pcs.clone();
        
        for i in 0..n_samples {
            let x = pcs[[i, 0]];
            
            // High-frequency sinusoidal noise
            let sin_noise = 0.3 * (10.0 * x).sin();
            
            // Random spikes (10% chance)
            let spike = if rng.gen::<f64>() < 0.1 { 
                rng.gen_range(-0.9..0.9) 
            } else { 
                0.0 
            };
            
            // White noise
            let white_noise = 0.25 * normal.sample(&mut rng);
            
            pcs_noisy[[i, 0]] = x + sin_noise + spike + white_noise;
        }
        
        // 3. Create training datasets
        let data_clean = data::TrainingData {
            y: y.clone(),
            p: pgs.clone(),
            pcs: pcs,
        };
        
        let data_noisy = data::TrainingData {
            y,
            p: pgs,
            pcs: pcs_noisy,
        };
        
        // 4. Create model configuration
        let pc_names: Vec<String> = (1..=n_pcs).map(|i| format!("PC{}", i)).collect();
        
        let config = ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            convergence_tolerance: 1e-6,
            max_iterations: 50,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 50,
            pgs_basis_config: BasisConfig { num_knots: 8, degree: 3 },
            pc_basis_configs: vec![
                BasisConfig { num_knots: 8, degree: 3 },
                BasisConfig { num_knots: 6, degree: 2 },
                BasisConfig { num_knots: 6, degree: 2 },
            ],
            pgs_range: (-2.5, 2.5),
            pc_ranges: vec![(-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0)],
            pc_names,
            constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            num_pgs_interaction_bases: 0, // Will be set during training
        };
        
        // 5. Train models
        println!("Training model on clean data...");
        let model_clean = train_model(&data_clean, &config)?;
        
        println!("Training model on noisy data...");
        let model_noisy = train_model(&data_noisy, &config)?;
        
        // 6. Extract lambdas for corresponding terms
        // Helper function to get lambda for a specific PC
        // In this implementation we assume the first N lambdas correspond to the N PCs
        fn get_pc_lambda(model: &TrainedModel, pc_idx: usize) -> Option<f64> {
            model.lambdas.get(pc_idx).copied()
        }
        
        // 7. Compare lambdas for PC1 (real effect)
        let pc1_idx = 0;
        if let (Some(lambda_clean), Some(lambda_noisy)) = (
            get_pc_lambda(&model_clean, pc1_idx),
            get_pc_lambda(&model_noisy, pc1_idx)
        ) {
            println!("\nPC1 (real effect with added noise):");
            println!("Lambda for clean PC1: {:.6}", lambda_clean);
            println!("Lambda for noisy PC1: {:.6}", lambda_noisy);
            println!("Ratio (noisy/clean): {:.2}x", lambda_noisy / lambda_clean);
            
            // ASSERTION: Lambda should increase by at least 50% for noisy data
            assert!(
                lambda_noisy > lambda_clean * 1.5,
                "REML didn't sufficiently increase lambda for noisy data. Expected at least 50% increase, got {:.2}x",
                lambda_noisy / lambda_clean
            );
        }
        
        // 8. Control test: Check PC3 (null effect)
        let pc3_idx = 2;
        if let (Some(lambda_clean_pc3), Some(lambda_noisy_pc3)) = (
            get_pc_lambda(&model_clean, pc3_idx),
            get_pc_lambda(&model_noisy, pc3_idx)
        ) {
            println!("\nPC3 (null effect, no added noise):");
            println!("Lambda for clean PC3: {:.6}", lambda_clean_pc3);
            println!("Lambda for noisy PC3: {:.6}", lambda_noisy_pc3);
            println!("Ratio: {:.2}x", lambda_noisy_pc3 / lambda_clean_pc3);
            
            // For null effect, both lambdas should be high but not dramatically different
            println!("Note: Both should be high since PC3 is a null effect");
            
            // No formal assertion here, as we don't know exact lambda values,
            // but both should be relatively high compared to real effects
        }
        
        println!("✓ REML lambda sensitivity test passed");
        Ok(())
    }
    
    /// Prepare data files for training and testing
    fn prepare_data_files(
        training_data: &(Vec<Vec<f64>>, Vec<String>),
        test_data: &(Vec<Vec<f64>>, Vec<String>),
        training_path: &std::path::Path,
        test_path: &std::path::Path
    ) -> Result<(usize, usize), std::io::Error> {
        // Write training data to file
        write_training_data_to_file(training_data, training_path.to_str().unwrap())?;
        
        // Write test data to file
        write_test_data_to_file(test_data, test_path.to_str().unwrap())?;
        
        // Return the number of rows in each file
        Ok((training_data.0.len(), test_data.0.len()))
    }
    
    /// Calculate correlation between two vectors
    fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let mut sum_xy = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x_sq = 0.0;
        let mut sum_y_sq = 0.0;
        
        for i in 0..x.len() {
            sum_xy += x[i] * y[i];
            sum_x += x[i];
            sum_y += y[i];
            sum_x_sq += x[i] * x[i];
            sum_y_sq += y[i] * y[i];
        }
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();
        
        if denominator.abs() < 1e-10 {
            return 0.0; // Avoid division by zero
        }
        
        numerator / denominator
    }
    
    /// Check key properties of the trained models
    fn check_model_properties(model1: &serde_json::Value, model2: &serde_json::Value) {
        // Check link function is correctly detected
        if let Some(link1) = model1["config"]["link_function"].as_str() {
            println!("   Model 1 link function: {}", link1);
            assert_eq!(link1, "Logit", "Expected logit link function for binary outcome");
        }
        
        if let Some(link2) = model2["config"]["link_function"].as_str() {
            println!("   Model 2 link function: {}", link2);
            assert_eq!(link2, "Logit", "Expected logit link function for binary outcome");
        }
        
        // Check for presence of key coefficient groups
        if let Some(obj) = model1["coefficients"].as_object() {
            assert!(obj.contains_key("intercept"), "Model 1 missing intercept");
            assert!(obj.contains_key("main_effects"), "Model 1 missing main effects");
            assert!(obj.contains_key("interaction_effects"), "Model 1 missing interaction effects");
            
            // Check if main effects contain PGS and PCs
            if let Some(main_effects) = obj["main_effects"].as_object() {
                assert!(main_effects.contains_key("pgs"), "Model 1 missing PGS main effect");
                assert!(main_effects.contains_key("pcs"), "Model 1 missing PC main effects");
                
                println!("   Model 1 has expected coefficient structure");
            }
        }
        
        // Check if smoothing parameters (lambdas) were estimated
        if let Some(lambdas1) = model1["lambdas"].as_array() {
            println!("   Model 1 estimated {} smoothing parameters", lambdas1.len());
            assert!(!lambdas1.is_empty(), "Model 1 has no smoothing parameters");
            
            // Print first few lambdas
            println!("   First few lambdas: {:.3}, {:.3}, ...", 
                    lambdas1.get(0).and_then(|v| v.as_f64()).unwrap_or(0.0),
                    lambdas1.get(1).and_then(|v| v.as_f64()).unwrap_or(0.0));
        }
        
        if let Some(lambdas2) = model2["lambdas"].as_array() {
            println!("   Model 2 estimated {} smoothing parameters", lambdas2.len());
            assert!(!lambdas2.is_empty(), "Model 2 has no smoothing parameters");
        }
    }
    
    /// Structure to hold the true effects used to generate synthetic data
    #[derive(Debug, Clone)]
    struct TrueEffects {
        intercept: f64,
        pgs_main_effect: f64,
        pc_main_effects: HashMap<String, f64>,
        interaction_effects: HashMap<String, f64>,
        noise_level: f64,
        non_linear_effects: bool,
        case_control_ratio: f64,
        sex_effect: f64,
        age_effect: f64,
        age_pgs_interaction: f64,
        random_seed: u64,
        threshold_effects: bool,
        ancestry_specific_effects: HashMap<String, f64>,
    }
    
    /// Generate a realistic dataset with known relationships between variables
    fn generate_realistic_dataset() -> Result<((Vec<Vec<f64>>, Vec<String>), (Vec<Vec<f64>>, Vec<String>), TrueEffects), io::Error> {
        // Use deterministic random seed to ensure reproducible test results
        const RANDOM_SEED: u64 = 4242;
        
        let n_train = 1000; // Larger training set for more realistic scenario
        let n_test = 250;   // Larger test set
        let n_pcs = 5;      // PCs for realistic population structure
        
        // Use seeded RNG for reproducible tests
        use rand::{SeedableRng, seq::SliceRandom};
        use rand::distributions::{Distribution, Normal, Uniform};
        use rand_distr::{LogNormal, Beta, Gamma, Weibull};
        use rand::rngs::StdRng;
        
        let mut rng = StdRng::seed_from_u64(RANDOM_SEED);
        
        // Define true effects for our simulation with comprehensive parameters
        let true_effects = TrueEffects {
            intercept: -2.0,                             // Very low base disease risk (realistic for rare conditions)
            pgs_main_effect: 3.0,                        // Strong PGS effect
            pc_main_effects: {
                let mut effects = HashMap::new();
                effects.insert("PC1".to_string(), 0.9);  // Strong PC1 effect (major population structure)
                effects.insert("PC2".to_string(), 0.6);  // Moderate PC2 effect
                effects.insert("PC3".to_string(), 0.3);  // Smaller PC3 effect 
                effects.insert("PC4".to_string(), 0.15); // Very small PC4 effect
                effects.insert("PC5".to_string(), 0.0);  // No PC5 effect (null)
                effects
            },
            interaction_effects: {
                let mut effects = HashMap::new();
                effects.insert("PC1".to_string(), 0.8);   // Strong PC1×PGS interaction (ancestry-specific effect)
                effects.insert("PC2".to_string(), -0.4);  // Negative PC2×PGS interaction
                effects.insert("PC3".to_string(), 0.0);   // No PC3×PGS interaction
                effects.insert("PC4".to_string(), 0.2);   // Subtle PC4×PGS interaction 
                effects.insert("PC5".to_string(), 0.0);   // No PC5×PGS interaction
                effects
            },
            noise_level: 0.7,                            // Realistic noise level
            non_linear_effects: true,                    // Enable non-linear effects
            case_control_ratio: 0.25,                    // Realistic case/control ratio (25% cases)
            sex_effect: 0.5,                             // Males have higher risk
            age_effect: 0.02,                            // Risk increases with age
            age_pgs_interaction: 0.01,                   // PGS effect stronger in older individuals
            random_seed: RANDOM_SEED,                    // Store the seed for reproducibility
            threshold_effects: true,                     // Enable threshold effects
            ancestry_specific_effects: {
                let mut effects = HashMap::new();
                effects.insert("cluster_0".to_string(), 0.0);   // Reference population baseline
                effects.insert("cluster_1".to_string(), 0.3);   // Population 2 has higher baseline risk
                effects.insert("cluster_2".to_string(), -0.2);  // Population 3 has lower baseline risk
                effects
            },
        };
        
        // Simulate ancestry clusters for more realistic PCA structure
        // Define 3 ancestry clusters with different PGS distributions and disease risks
        #[derive(Debug, Clone)]
        struct AncestryCluster {
            name: String,
            pc1_mean: f64,
            pc1_std: f64,
            pc2_mean: f64, 
            pc2_std: f64,
            pc3_mean: f64,
            pc3_std: f64,
            pgs_mean: f64,
            pgs_std: f64,
            proportion: f64,
            baseline_risk_modifier: f64,
            // Parameters for generating admixed individuals
            admixture_proportion: f64, // What proportion are admixed
            admixture_target: usize,   // Which population to admix with
            admixture_degree: f64,     // How much admixture (0-1)
        }
        
        let ancestry_clusters = vec![
            // Population 1 (reference population - European ancestry)
            AncestryCluster {
                name: "European".to_string(),
                pc1_mean: -0.8, 
                pc1_std: 0.3,
                pc2_mean: 0.0,
                pc2_std: 0.4,
                pc3_mean: 0.2,
                pc3_std: 0.3,
                pgs_mean: 0.0,
                pgs_std: 0.8,
                proportion: 0.5,
                baseline_risk_modifier: 0.0, // Reference
                admixture_proportion: 0.1,   // 10% admixed
                admixture_target: 1,         // With population 2
                admixture_degree: 0.3,       // 30% admixture
            },
            // Population 2 (African ancestry)
            AncestryCluster {
                name: "African".to_string(),
                pc1_mean: 0.85,
                pc1_std: 0.35,
                pc2_mean: 0.7,
                pc2_std: 0.3,
                pc3_mean: -0.3,
                pc3_std: 0.25,
                pgs_mean: 0.2, // Higher mean PGS
                pgs_std: 0.9,
                proportion: 0.3,
                baseline_risk_modifier: 0.3, // Higher baseline risk
                admixture_proportion: 0.15,  // 15% admixed
                admixture_target: 0,         // With population 1
                admixture_degree: 0.25,      // 25% admixture
            },
            // Population 3 (East Asian ancestry)
            AncestryCluster {
                name: "EastAsian".to_string(),
                pc1_mean: 0.3,
                pc1_std: 0.25,
                pc2_mean: -0.9,
                pc2_std: 0.3,
                pc3_mean: 0.5,
                pc3_std: 0.2,
                pgs_mean: -0.3, // Lower mean PGS
                pgs_std: 0.7,
                proportion: 0.2,
                baseline_risk_modifier: -0.2, // Lower baseline risk
                admixture_proportion: 0.05,   // 5% admixed
                admixture_target: 0,          // With population 1
                admixture_degree: 0.2,        // 20% admixture
            },
        ];
        
        // Cumulative proportions for cluster selection
        let cumulative_props: Vec<f64> = ancestry_clusters.iter()
            .scan(0.0, |sum, cluster| {
                *sum += cluster.proportion;
                Some(*sum)
            })
            .collect();
        
        // Function to select ancestry cluster based on population proportions
        let select_cluster = |rng: &mut StdRng| -> usize {
            let r = Uniform::new(0.0, 1.0).sample(rng);
            for (i, &cum_prop) in cumulative_props.iter().enumerate() {
                if r <= cum_prop {
                    return i;
                }
            }
            ancestry_clusters.len() - 1 // Fallback to last cluster
        };
        
        // Function to determine if an individual is admixed
        let is_admixed = |cluster: &AncestryCluster, rng: &mut StdRng| -> bool {
            Uniform::new(0.0, 1.0).sample(rng) < cluster.admixture_proportion
        };
        
        // Generate PC values accounting for admixture
        let generate_pc_values = |cluster_idx: usize, is_admixed_individual: bool, rng: &mut StdRng| -> Vec<f64> {
            let cluster = &ancestry_clusters[cluster_idx];
            let mut pcs = Vec::with_capacity(n_pcs);
            
            if !is_admixed_individual {
                // Non-admixed individual - use cluster distributions
                pcs.push(Normal::new(cluster.pc1_mean, cluster.pc1_std).unwrap().sample(rng));
                pcs.push(Normal::new(cluster.pc2_mean, cluster.pc2_std).unwrap().sample(rng));
                pcs.push(Normal::new(cluster.pc3_mean, cluster.pc3_std).unwrap().sample(rng));
                
                // Remaining PCs are standard normal
                for _ in 3..n_pcs {
                    pcs.push(Normal::new(0.0, 1.0).unwrap().sample(rng));
                }
            } else {
                // Admixed individual - blend between two populations
                let target_idx = cluster.admixture_target;
                let target = &ancestry_clusters[target_idx];
                let admix_degree = cluster.admixture_degree;
                
                // PC1 - weighted average plus noise
                let pc1_mean = (1.0 - admix_degree) * cluster.pc1_mean + admix_degree * target.pc1_mean;
                let pc1_std = (cluster.pc1_std + target.pc1_std) / 2.0; // Average std
                pcs.push(Normal::new(pc1_mean, pc1_std).unwrap().sample(rng));
                
                // PC2 - weighted average plus noise
                let pc2_mean = (1.0 - admix_degree) * cluster.pc2_mean + admix_degree * target.pc2_mean;
                let pc2_std = (cluster.pc2_std + target.pc2_std) / 2.0; // Average std
                pcs.push(Normal::new(pc2_mean, pc2_std).unwrap().sample(rng));
                
                // PC3 - weighted average plus noise
                let pc3_mean = (1.0 - admix_degree) * cluster.pc3_mean + admix_degree * target.pc3_mean;
                let pc3_std = (cluster.pc3_std + target.pc3_std) / 2.0; // Average std
                pcs.push(Normal::new(pc3_mean, pc3_std).unwrap().sample(rng));
                
                // Remaining PCs are standard normal
                for _ in 3..n_pcs {
                    pcs.push(Normal::new(0.0, 1.0).unwrap().sample(rng));
                }
            }
            
            // Apply realistic correlation structure to PCs
            apply_pc_correlation(&mut pcs, rng);
            
            pcs
        };
        
        // Generate PGS value accounting for ancestry
        let generate_pgs = |cluster_idx: usize, is_admixed_individual: bool, rng: &mut StdRng| -> f64 {
            let cluster = &ancestry_clusters[cluster_idx];
            
            if !is_admixed_individual {
                // Non-admixed individual - use cluster-specific distribution
                match cluster_idx {
                    0 => {
                        // European ancestry - slightly right-skewed
                        let base_pgs = Normal::new(cluster.pgs_mean, cluster.pgs_std).unwrap().sample(rng);
                        let skew = Gamma::new(1.5, 0.4).unwrap().sample(rng) - 0.6;
                        base_pgs + 0.15 * skew
                    },
                    1 => {
                        // African ancestry - more variance
                        let base_pgs = Normal::new(cluster.pgs_mean, cluster.pgs_std).unwrap().sample(rng);
                        let t_component = if Uniform::new(0.0, 1.0).sample(rng) < 0.05 {
                            // Add occasional outliers for realism
                            2.0 * rng.gen::<f64>().signum()
                        } else {
                            0.0
                        };
                        base_pgs + 0.1 * t_component
                    },
                    2 => {
                        // East Asian ancestry - less variance, more uniform
                        let beta_sample = Beta::new(2.5, 2.5).unwrap().sample(rng) * 2.0 - 1.0;
                        cluster.pgs_mean + cluster.pgs_std * beta_sample
                    },
                    _ => Normal::new(cluster.pgs_mean, cluster.pgs_std).unwrap().sample(rng), // Default
                }
            } else {
                // Admixed individual - blend PGS distributions
                let target_idx = cluster.admixture_target;
                let target = &ancestry_clusters[target_idx];
                let admix_degree = cluster.admixture_degree;
                
                // PGS mean is weighted average of the two populations
                let pgs_mean = (1.0 - admix_degree) * cluster.pgs_mean + admix_degree * target.pgs_mean;
                let pgs_std = (cluster.pgs_std + target.pgs_std) / 2.0; // Average std
                
                Normal::new(pgs_mean, pgs_std).unwrap().sample(rng)
            }
        };
        
        // Generate PC correlation structure (realistic correlation between PCs)
        fn apply_pc_correlation(pcs: &mut [f64], rng: &mut StdRng) {
            if pcs.len() < 3 { return; } // Need at least 3 PCs for correlation
            
            // Create structured correlations between PC pairs
            // PC3 correlated with PC1 and PC2
            pcs[2] += 0.08 * pcs[0] - 0.12 * pcs[1];
            
            if pcs.len() > 3 {
                // PC4 has complex correlation with other PCs
                pcs[3] += 0.06 * pcs[1] - 0.05 * pcs[0] + 0.03 * pcs[2];
            }
            
            if pcs.len() > 4 {
                // PC5 has weak correlation with PC1
                pcs[4] += 0.03 * pcs[0];
            }
            
            // Add small random noise to correlation structure
            for i in 2..pcs.len() {
                let noise = Normal::new(0.0, 0.05).unwrap().sample(rng);
                pcs[i] += noise;
            }
        }
        
        // Create non-linear effect function for PGS with threshold effects
        let pgs_nonlinear_effect = |pgs: f64, threshold_effects: bool| -> f64 {
            if !threshold_effects {
                return pgs; // Linear effect if threshold effects disabled
            }
            
            // Sigmoid-like function with threshold effect around PGS=0.5
            let base_effect = pgs;
            
            // Threshold effect: more pronounced effect above threshold
            let threshold = 0.5;
            let above_threshold = (pgs - threshold).max(0.0);
            
            // Blend linear and threshold components
            0.6 * base_effect + 0.4 * above_threshold
        };
        
        // Create non-linear interaction effect
        let pc_interaction_nonlinear = |pc: f64, pgs: f64, threshold_effects: bool| -> f64 {
            if !threshold_effects {
                return pc * pgs; // Regular interaction if threshold effects disabled
            }
            
            // Enhanced interaction at extremes of PC distribution
            let pc_abs = pc.abs();
            let threshold = 0.7;
            
            if pc_abs > threshold {
                // Stronger effect at extremes
                let scale_factor = 1.0 + 0.3 * (pc_abs - threshold);
                pc * pgs * scale_factor
            } else {
                pc * pgs
            }
        };
        
        // Generate age values with realistic distribution
        // Weibull distribution gives a realistic age distribution for adult population
        let age_distribution = Weibull::new(5.0, 55.0).unwrap();
        
        // Generate training data - use case-control sampling for class balance
        let mut cases = Vec::new();
        let mut controls = Vec::new();
        let desired_cases = (n_train as f64 * true_effects.case_control_ratio) as usize;
        let desired_controls = n_train - desired_cases;
        
        // Create header: phenotype, score, sex, age, PC1, PC2, ...
        let mut training_header = vec![
            "phenotype".to_string(), 
            "score".to_string(),
            "sex".to_string(),
            "age".to_string()
        ];
        
        for i in 1..=n_pcs {
            training_header.push(format!("PC{}", i));
        }
        
        // Add ancestry column for transparency in debugging
        training_header.push("ancestry_cluster".to_string());
        training_header.push("is_admixed".to_string());
        
        // Keep generating samples until we have enough cases and controls
        while cases.len() < desired_cases || controls.len() < desired_controls {
            // Select ancestry cluster
            let cluster_idx = select_cluster(&mut rng);
            let cluster = &ancestry_clusters[cluster_idx];
            
            // Determine if individual is admixed
            let is_admixed_individual = is_admixed(cluster, &mut rng);
            
            // Generate PGS value using cluster-specific distribution
            let pgs = generate_pgs(cluster_idx, is_admixed_individual, &mut rng);
            
            // Generate PC values with admixture if applicable
            let pcs = generate_pc_values(cluster_idx, is_admixed_individual, &mut rng);
            
            // Generate age and sex (binary for simplicity)
            let age = age_distribution.sample(&mut rng).min(100.0); // Cap at 100 years
            let sex = if Uniform::new(0.0, 1.0).sample(&mut rng) < 0.5 { 0.0 } else { 1.0 }; // 0=female, 1=male
            
            // Calculate ancestry-specific baseline risk
            let ancestry_risk = *true_effects.ancestry_specific_effects
                .get(&format!("cluster_{}", cluster_idx))
                .unwrap_or(&0.0);
            
            // Calculate linear predictor (log-odds) using our true effects
            let mut eta = true_effects.intercept + ancestry_risk + 
                true_effects.pgs_main_effect * pgs_nonlinear_effect(pgs, true_effects.threshold_effects);
            
            // Add sex effect (higher risk for males)
            eta += true_effects.sex_effect * sex;
            
            // Add age effect
            let age_centered = age - 50.0; // Center age around 50
            eta += true_effects.age_effect * age_centered;
            
            // Add age × PGS interaction (stronger PGS effect in older individuals)
            eta += true_effects.age_pgs_interaction * age_centered * pgs;
            
            // Add PC main effects
            for i in 0..pcs.len() {
                let pc_name = format!("PC{}", i+1);
                if let Some(effect) = true_effects.pc_main_effects.get(&pc_name) {
                    eta += effect * pcs[i];
                }
            }
            
            // Add interaction effects (PC × PGS)
            for i in 0..pcs.len() {
                let pc_name = format!("PC{}", i+1);
                if let Some(effect) = true_effects.interaction_effects.get(&pc_name) {
                    // Use non-linear interaction if enabled
                    eta += effect * pc_interaction_nonlinear(pcs[i], pgs, true_effects.threshold_effects);
                }
            }
            
            // Add random noise with cluster-specific variance
            // More heteroscedasticity - higher variance in some populations
            let noise_factor = match cluster_idx {
                0 => 1.0,     // Reference population
                1 => 1.1,     // Higher variance in population 2
                2 => 0.9,     // Lower variance in population 3
                _ => 1.0,     // Default
            };
            
            eta += Normal::new(0.0, true_effects.noise_level * noise_factor).unwrap().sample(&mut rng);
            
            // Convert from log-odds to probability
            let prob = 1.0 / (1.0 + (-eta).exp());
            
            // Generate binary outcome (0/1) using probability
            let phenotype = if Uniform::new(0.0, 1.0).sample(&mut rng) < prob { 1.0 } else { 0.0 };
            
            // Create row: [phenotype, pgs, sex, age, pc1, pc2, ..., ancestry_cluster, is_admixed]
            let mut row = vec![phenotype, pgs, sex, age];
            row.extend(pcs);
            row.push(cluster_idx as f64); // Ancestry cluster index
            row.push(if is_admixed_individual { 1.0 } else { 0.0 }); // Admixed flag
            
            // Add to appropriate group if needed
            if phenotype > 0.5 && cases.len() < desired_cases {
                cases.push(row);
            } else if phenotype < 0.5 && controls.len() < desired_controls {
                controls.push(row);
            }
            
            // Check if we've collected enough samples
            if cases.len() >= desired_cases && controls.len() >= desired_controls {
                break;
            }
        }
        
        // Combine cases and controls
        let mut training_data = Vec::with_capacity(n_train);
        training_data.extend(cases);
        training_data.extend(controls);
        
        // Shuffle training data for good measure
        training_data.shuffle(&mut rng);
        
        // Generate test data (same process, but without phenotype)
        let mut test_data = Vec::with_capacity(n_test);
        let mut test_header = vec![
            "score".to_string(), 
            "sex".to_string(),
            "age".to_string()
        ];
        
        for i in 1..=n_pcs {
            test_header.push(format!("PC{}", i));
        }
        
        // Add ancestry columns for debugging
        test_header.push("ancestry_cluster".to_string());
        test_header.push("is_admixed".to_string());
        
        for _ in 0..n_test {
            // Select ancestry cluster
            let cluster_idx = select_cluster(&mut rng);
            let cluster = &ancestry_clusters[cluster_idx];
            
            // Determine if individual is admixed
            let is_admixed_individual = is_admixed(cluster, &mut rng);
            
            // Generate PGS value using cluster-specific distribution
            let pgs = generate_pgs(cluster_idx, is_admixed_individual, &mut rng);
            
            // Generate PC values with admixture if applicable
            let pcs = generate_pc_values(cluster_idx, is_admixed_individual, &mut rng);
            
            // Generate age and sex
            let age = age_distribution.sample(&mut rng).min(100.0); // Cap at 100 years
            let sex = if Uniform::new(0.0, 1.0).sample(&mut rng) < 0.5 { 0.0 } else { 1.0 }; // 0=female, 1=male
            
            // Create row: [pgs, sex, age, pc1, pc2, ..., ancestry_cluster, is_admixed]
            let mut row = vec![pgs, sex, age];
            row.extend(pcs);
            row.push(cluster_idx as f64); // Ancestry cluster index
            row.push(if is_admixed_individual { 1.0 } else { 0.0 }); // Admixed flag
            
            test_data.push(row);
        }
        
        Ok(((training_data, training_header), (test_data, test_header), true_effects))
    }
    
    /// Write training data to a TSV file
    fn write_training_data_to_file(training_data: &(Vec<Vec<f64>>, Vec<String>), file_path: &str) -> Result<(), io::Error> {
        let (data, header) = training_data;
        let mut file = std::fs::File::create(file_path)?;
        
        // Write header
        writeln!(file, "{}", header.join("\t"))?;
        
        // Write data rows
        for row in data {
            let line = row.iter()
                .map(|x| format!("{:.6}", x))
                .collect::<Vec<String>>()
                .join("\t");
            writeln!(file, "{}", line)?;
        }
        
        Ok(())
    }
    
    /// Write test data to a TSV file
    fn write_test_data_to_file(test_data: &(Vec<Vec<f64>>, Vec<String>), file_path: &str) -> Result<(), io::Error> {
        let (data, header) = test_data;
        let mut file = std::fs::File::create(file_path)?;
        
        // Write header
        writeln!(file, "{}", header.join("\t"))?;
        
        // Write data rows
        for row in data {
            let line = row.iter()
                .map(|x| format!("{:.6}", x))
                .collect::<Vec<String>>()
                .join("\t");
            writeln!(file, "{}", line)?;
        }
        
        Ok(())
    }
    
    /// Load predictions from the output file
    fn load_predictions(file_path: &str) -> Result<Vec<f64>, io::Error> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};
        
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);
        let mut predictions = Vec::new();
        let mut skip_header = true;
        
        for line in reader.lines() {
            let line = line?;
            if skip_header {
                skip_header = false;
                continue;
            }
            
            if let Ok(value) = line.trim().parse::<f64>() {
                predictions.push(value);
            } else {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Failed to parse prediction: {}", line),
                ));
            }
        }
        
        Ok(predictions)
    }
    
    /// Metrics returned by the validation function
    #[derive(Debug)]
    struct ValidationMetrics {
        correlation: f64,
        r_squared: f64,
        mae: f64,
        auc: f64,
        brier_score: f64,
        calibration_error: f64,
        monotonicity_score: f64,
        pc1_interaction_detected: bool,
        pc2_interaction_detected: bool,
    }
    
    /// Validates that the model's internal coefficient structure reflects the ground truth.
    ///
    /// It checks if real effects have larger coefficient magnitudes (L2 Norm) than null effects,
    /// and that null effects are correctly shrunk towards zero by the penalty.
    fn validate_coefficient_structure(model: &TrainedModel, true_effects: &TrueEffects, model_name: &str) {
        println!("\n   Validating coefficient structure for {} model...", model_name);
        
        // Calculate L2 norm (Euclidean magnitude) of coefficients for each interaction term
        // This gives us a single value representing the overall "energy" or strength of each term
        let mut pc_magnitudes = HashMap::new();
        
        for pc_idx in 0..true_effects.pc_main_effects.len() {
            let pc_name = format!("PC{}", pc_idx + 1);
            let mut squared_sum = 0.0;
            
            // Go through all PGS basis functions looking for interactions with this PC
            for (pgs_key, pc_map) in &model.coefficients.interaction_effects {
                if let Some(coeffs) = pc_map.get(&pc_name) {
                    // Add squared coefficients for this interaction
                    for &coef in coeffs.iter() {
                        squared_sum += coef * coef;
                    }
                }
            }
            
            // Calculate L2 norm (square root of sum of squares)
            let l2_norm = squared_sum.sqrt();
            pc_magnitudes.insert(pc_name, l2_norm);
        }
        
        // 1. Real vs. Null: Check if real effects (PC1, PC2, PC4) are significantly larger than null effects (PC3, PC5)
        println!("   1. Validating Real vs. Null Effects");
        
        // Calculate the average magnitude for real and null effects
        let real_pcs = ["PC1", "PC2", "PC4"];
        let null_pcs = ["PC3", "PC5"];
        
        let real_magnitude_sum: f64 = real_pcs.iter()
            .filter_map(|&pc| pc_magnitudes.get(pc))
            .sum();
        let real_avg_magnitude = real_magnitude_sum / real_pcs.len() as f64;
        
        let null_magnitude_sum: f64 = null_pcs.iter()
            .filter_map(|&pc| pc_magnitudes.get(pc))
            .sum();
        let null_avg_magnitude = null_magnitude_sum / null_pcs.len() as f64;
        
        println!("     - Real effects average magnitude: {:.6}", real_avg_magnitude);
        println!("     - Null effects average magnitude: {:.6}", null_avg_magnitude);
        println!("     - Ratio (real/null): {:.2}x", real_avg_magnitude / null_avg_magnitude);
        
        // Verify real effects are significantly larger than null effects (at least 2x)
        assert!(
            real_avg_magnitude > null_avg_magnitude * 2.0,
            "Real effects (PC1, PC2, PC4) should have significantly larger magnitudes than null effects (PC3, PC5)"
        );
        
        // 2. Effect Hierarchy: Verify PC1 > PC2 > PC4 (should match the strength of simulated effects)
        println!("   2. Validating Effect Hierarchy");
        
        let pc1_magnitude = pc_magnitudes.get("PC1").unwrap_or(&0.0);
        let pc2_magnitude = pc_magnitudes.get("PC2").unwrap_or(&0.0);
        let pc4_magnitude = pc_magnitudes.get("PC4").unwrap_or(&0.0);
        
        println!("     - PC1 magnitude: {:.6}", pc1_magnitude);
        println!("     - PC2 magnitude: {:.6}", pc2_magnitude);
        println!("     - PC4 magnitude: {:.6}", pc4_magnitude);
        
        // Verify that PC1 > PC2
        assert!(
            pc1_magnitude > pc2_magnitude,
            "PC1 should have larger coefficient magnitude than PC2"
        );
        
        // Verify that PC2 > PC4
        assert!(
            pc2_magnitude > pc4_magnitude,
            "PC2 should have larger coefficient magnitude than PC4"
        );
        
        // 3. Shrinkage: Verify null effects are very close to zero (effective regularization)
        println!("   3. Validating Shrinkage of Null Effects");
        
        let pc3_magnitude = pc_magnitudes.get("PC3").unwrap_or(&0.0);
        let pc5_magnitude = pc_magnitudes.get("PC5").unwrap_or(&0.0);
        
        println!("     - PC3 magnitude (null): {:.6}", pc3_magnitude);
        println!("     - PC5 magnitude (null): {:.6}", pc5_magnitude);
        
        // Define a threshold for "close to zero" - should be very small compared to real effects
        let threshold = real_avg_magnitude * 0.1; // 10% of average real effect
        
        // Verify null effects are below threshold
        assert!(
            *pc3_magnitude < threshold,
            "PC3 coefficient magnitude should be shrunk close to zero (< {:.6})", threshold
        );
        assert!(
            *pc5_magnitude < threshold,
            "PC5 coefficient magnitude should be shrunk close to zero (< {:.6})", threshold
        );
        
        // Output overall validation results
        println!("   ✓ Coefficient structure validation passed for {} model", model_name);
        println!("     - Real effects properly distinguished from null effects");
        println!("     - Effect hierarchy correctly preserved (PC1 > PC2 > PC4)");
        println!("     - Null effects properly regularized toward zero");
    }
    
    /// Validates that null effects are learned as flat functions and real effects are not.
    /// This ensures regularization correctly identifies and penalizes effects that should be zero.
    fn validate_smooth_function_shapes(model: &TrainedModel, true_effects: &TrueEffects) {
        println!("\n===== VALIDATING SMOOTH FUNCTION SHAPES =====");
        
        // Calculate smooth function values for each PC
        let mut pc_function_stats = HashMap::new();
        
        for pc_name in &model.config.pc_names {
            // Find PC index in model configuration
            let pc_idx = model.config.pc_names.iter().position(|n| n == pc_name).unwrap();
            let pc_range = model.config.pc_ranges[pc_idx];
            
            // Create evaluation grid spanning the PC's range
            let grid = Array1::linspace(pc_range.0, pc_range.1, 100);
            
            // Get coefficients for this PC
            if let Some(pc_coeffs) = model.coefficients.main_effects.pcs.get(pc_name) {
                // Create basis matrix for evaluation
                if let Ok((basis_unc, _)) = basis::create_bspline_basis(
                    grid.view(), None, pc_range, 
                    model.config.pc_basis_configs[pc_idx].num_knots,
                    model.config.pc_basis_configs[pc_idx].degree
                ) {
                    // Apply constraint transformation
                    if let Some(constraint) = model.config.constraints.get(pc_name) {
                        let basis_con = basis_unc.dot(&constraint.z_transform);
                        
                        // Evaluate function: f(x) = B(x) * coeffs
                        let coeffs_array = Array1::from_vec(pc_coeffs.clone());
                        let function_values = basis_con.dot(&coeffs_array);
                        
                        // Calculate statistics
                        let std_dev = function_values.std(0.0);
                        let range = function_values.fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - 
                                    function_values.fold(f64::INFINITY, |a, &b| a.min(b));
                        
                        pc_function_stats.insert(pc_name.clone(), (std_dev, range));
                        
                        println!("PC {}: std_dev={:.6}, range={:.6}", pc_name, std_dev, range);
                    }
                }
            }
        }
        
        // Classify PCs based on known true effects
        let mut real_pcs = Vec::new();
        let mut null_pcs = Vec::new();
        
        for pc_name in &model.config.pc_names {
            let effect_size = true_effects.interaction_effects
                .get(pc_name).copied().unwrap_or(0.0).abs();
                
            if effect_size > 0.1 {
                real_pcs.push(pc_name.clone());
            } else {
                null_pcs.push(pc_name.clone());
            }
        }
        
        println!("\nReal effects: {:?}", real_pcs);
        println!("Null effects: {:?}", null_pcs);
        
        // Calculate average variability for real vs null effects
        let real_avg_std = real_pcs.iter()
            .filter_map(|pc| pc_function_stats.get(pc).map(|(std, _)| *std))
            .sum::<f64>() / real_pcs.len() as f64;
            
        let null_avg_std = null_pcs.iter()
            .filter_map(|pc| pc_function_stats.get(pc).map(|(std, _)| *std))
            .sum::<f64>() / null_pcs.len() as f64;
        
        println!("\nAverage std dev for real effects: {:.6}", real_avg_std);
        println!("Average std dev for null effects: {:.6}", null_avg_std);
        println!("Ratio (real/null): {:.2}x", real_avg_std / null_avg_std);
        
        // Calculate relative threshold based on data scale
        let threshold = real_avg_std * 0.1; // 10% of average real effect
        
        // ASSERTION 1: Null effects should be nearly flat
        for pc in &null_pcs {
            if let Some((std_dev, _)) = pc_function_stats.get(pc) {
                assert!(
                    *std_dev < threshold,
                    "Null effect {} not properly flattened (std_dev={:.6} > threshold={:.6})",
                    pc, std_dev, threshold
                );
            }
        }
        
        // ASSERTION 2: Real effects should have significantly higher variability
        assert!(
            real_avg_std > 5.0 * null_avg_std,
            "Real effects not sufficiently distinguished from null effects (ratio={:.2}x)",
            real_avg_std / null_avg_std
        );
        
        // ASSERTION 3: Effect hierarchy should match known pattern
        if let (Some((std_pc1, _)), Some((std_pc2, _)), Some((std_pc4, _))) = (
            pc_function_stats.get("PC1"),
            pc_function_stats.get("PC2"),
            pc_function_stats.get("PC4")
        ) {
            println!("\nEffect hierarchy check:");
            println!("PC1 std_dev: {:.6}", std_pc1);
            println!("PC2 std_dev: {:.6}", std_pc2);
            println!("PC4 std_dev: {:.6}", std_pc4);
            
            assert!(std_pc1 > std_pc2, "Expected PC1 effect > PC2 effect");
            assert!(std_pc2 > std_pc4, "Expected PC2 effect > PC4 effect");
        }
        
        println!("✓ Smooth function shape validation passed");
    }
    
    /// Validate predictions against expected values based on our known true effects
    /// Returns metrics for model comparison
    fn validate_predictions(predictions: &[f64], test_data: &(Vec<Vec<f64>>, Vec<String>), true_effects: &TrueEffects) -> ValidationMetrics {
        let (data, _) = test_data;
        
        // We can't expect exact matches due to:  
        // 1. The spline basis approximation of the true relationship
        // 2. Smoothing from regularization
        // 3. Limited training data
        // 4. Non-linear effects in the generating model
        
        // ======================================================================
        // SECTION 1: BASIC VALIDATION CHECKS
        // ======================================================================
        
        println!("\n1. RUNNING BASIC PREDICTION VALIDATION CHECKS...");
        
        // 1.1 Predictions should be probabilities between 0 and 1
        for &pred in predictions {
            assert!(pred >= 0.0 && pred <= 1.0, "Prediction out of bounds: {}", pred);
        }
        println!("✓ All predictions are valid probabilities between 0 and 1");
        
        // 1.2 Calculate expected predictions using our true effect model
        let mut expected_predictions = Vec::with_capacity(predictions.len());
        let n_pcs = true_effects.pc_main_effects.len();
        
        // Create complex non-linear effect function for PGS with multiple components
        let pgs_nonlinear_effect = |pgs: f64| -> f64 {
            if !true_effects.non_linear_effects {
                return pgs; // Linear effect if non-linear effects disabled
            }
            
            // 1. Threshold effect: minimal impact below threshold, stronger effect above
            let threshold = 0.5;
            let threshold_component = if pgs > threshold {
                1.2 * (pgs - threshold)
            } else {
                0.4 * (pgs - threshold)
            };
            
            // 2. Sigmoid-like component with inflection point
            let sigmoid_component = 1.0 / (1.0 + (-4.0 * (pgs - 0.8)).exp());
            
            // 3. Quadratic component (diminishing returns at extremes)
            let quadratic_component = -0.3 * (pgs - 1.5) * (pgs - 1.5) + 0.4;
            
            // Combine components with weights
            0.4 * pgs + 0.3 * threshold_component + 0.2 * sigmoid_component + 0.1 * quadratic_component
        };
        
        // Create complex non-linear interaction effects
        let pc_interaction_nonlinear = |pc: f64, pgs: f64| -> f64 {
            if !true_effects.non_linear_effects {
                return pc * pgs; // Regular interaction if non-linear effects disabled
            }
            
            // 1. Basic interaction
            let basic = pc * pgs;
            
            // 2. Enhanced effect at PC extremes (U-shaped)
            let pc_extreme = pc * pc * pgs;
            
            // 3. Synergistic effect when both PC and PGS are high
            let synergy = if pc > 0.0 && pgs > 0.8 {
                0.3 * pc * (pgs - 0.8)
            } else if pc < 0.0 && pgs > 0.8 {
                -0.3 * pc * (pgs - 0.8)
            } else {
                0.0
            };
            
            // 4. Antagonistic effect when PC and PGS have opposite signs
            let antagonism = if pc * pgs < 0.0 {
                -0.2 * pc.abs() * pgs.abs()
            } else {
                0.0
            };
            
            // Combine with weights
            0.5 * basic + 0.3 * pc_extreme + 0.15 * synergy + 0.05 * antagonism
        };
        
        for row in data {
            let pgs = row[0];  // First column is PGS
            
            // Calculate linear predictor using our true model
            let mut eta = true_effects.intercept + 
                true_effects.pgs_main_effect * pgs_nonlinear_effect(pgs);
            
            // Add PC main effects
            for i in 0..n_pcs.min(row.len()-1) { // Avoid out-of-bounds
                let pc_name = format!("PC{}", i+1);
                let pc_value = row[i+1];  // PC values start from index 1
                
                if let Some(effect) = true_effects.pc_main_effects.get(&pc_name) {
                    eta += effect * pc_value;
                }
            }
            
            // Add interaction effects
            for i in 0..n_pcs.min(row.len()-1) { // Avoid out-of-bounds
                let pc_name = format!("PC{}", i+1);
                let pc_value = row[i+1];
                
                if let Some(effect) = true_effects.interaction_effects.get(&pc_name) {
                    eta += effect * pc_interaction_nonlinear(pc_value, pgs);
                }
            }
            
            // Convert to probability
            let prob = 1.0 / (1.0 + (-eta).exp());
            expected_predictions.push(prob);
        }
        
        // 1.3 Calculate correlation between predicted and expected probabilities
        let mut sum_xy = 0.0;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_x_sq = 0.0;
        let mut sum_y_sq = 0.0;
        
        for i in 0..predictions.len() {
            let x = predictions[i];
            let y = expected_predictions[i];
            
            sum_xy += x * y;
            sum_x += x;
            sum_y += y;
            sum_x_sq += x * x;
            sum_y_sq += y * y;
        }
        
        let n = predictions.len() as f64;
        let correlation = (n * sum_xy - sum_x * sum_y) / 
            ((n * sum_x_sq - sum_x * sum_x) * (n * sum_y_sq - sum_y * sum_y)).sqrt();
        
        // The correlation should be high since we're testing with sufficient data
        println!("✓ Correlation between predicted and expected values: {:.3}", correlation);
        assert!(correlation > 0.7, "Correlation between predictions and expected values too low: {}", correlation);
        
        // 1.4 Calculate R² (coefficient of determination)
        let mean_y = sum_y / n;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        
        for i in 0..predictions.len() {
            ss_tot += (expected_predictions[i] - mean_y).powi(2);
            ss_res += (expected_predictions[i] - predictions[i]).powi(2);
        }
        
        let r_squared = 1.0 - (ss_res / ss_tot);
        println!("✓ R² coefficient of determination: {:.3}", r_squared);
        assert!(r_squared > 0.5, "R² too low: {}", r_squared);
        
        // 1.5 Calculate mean absolute error
        let mae = predictions.iter()
            .zip(expected_predictions.iter())
            .map(|(&p, &e)| (p - e).abs())
            .sum::<f64>() / predictions.len() as f64;
        
        // MAE should be reasonable considering noise level
        println!("✓ Mean absolute error: {:.3}", mae);
        assert!(mae < 0.15, "Mean absolute error too high: {}", mae);
        
        // ======================================================================
        // SECTION 2: ASSESSING MODEL DISCRIMINATION (AUC/ROC)
        // ======================================================================
        println!("\n2. ASSESSING MODEL DISCRIMINATION...");
        
        // 2.1 Calculate discrimination metrics using our expected values as "truth"
        // This is a pseudo-AUC that tests if the model can discriminate between high-risk and low-risk individuals
        
        // Convert expected continuous probabilities to binary outcomes using 0.5 threshold for simplicity
        let binary_expected: Vec<bool> = expected_predictions.iter()
            .map(|&p| p > 0.5)
            .collect();
            
        // Sort predicted probabilities with their expected binary outcome
        let mut pred_with_outcome: Vec<(f64, bool)> = predictions.iter()
            .zip(binary_expected.iter())
            .map(|(&pred, &outcome)| (pred, outcome))
            .collect();
            
        pred_with_outcome.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Calculate AUC using the trapezoidal rule
        let mut true_positives = 0;
        let mut false_positives = 0;
        let total_positives = binary_expected.iter().filter(|&&b| b).count();
        let total_negatives = binary_expected.len() - total_positives;
        
        if total_positives == 0 || total_negatives == 0 {
            println!("Cannot calculate AUC: all outcomes are the same class");
        } else {
            let mut auc = 0.0;
            let mut prev_tpr = 0.0;
            let mut prev_fpr = 0.0;
            
            // Add sentinel point for (0,0)
            let mut roc_points = vec![(0.0, 0.0)];
            
            for (_, outcome) in pred_with_outcome.iter().rev() {
                if *outcome {
                    true_positives += 1;
                } else {
                    false_positives += 1;
                }
                
                let tpr = true_positives as f64 / total_positives as f64;
                let fpr = false_positives as f64 / total_negatives as f64;
                
                auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;
                
                prev_tpr = tpr;
                prev_fpr = fpr;
                
                // Save ROC points for further analysis
                roc_points.push((fpr, tpr));
            }
            
            // Add sentinel point for (1,1)
            roc_points.push((1.0, 1.0));
            
            // Calculate AUC using the trapezoidal rule on the ROC points
            auc = 0.0;
            for i in 1..roc_points.len() {
                let (x1, y1) = roc_points[i-1];
                let (x2, y2) = roc_points[i];
                auc += (x2 - x1) * (y1 + y2) / 2.0;
            }
            
            println!("✓ Area Under ROC Curve (AUC): {:.3}", auc);
            assert!(auc > 0.7, "AUC too low: {}", auc);
            
            // Calculate Brier Score (mean squared error for probabilistic predictions)
            let brier_score = expected_predictions.iter()
                .zip(predictions.iter())
                .map(|(&e, &p)| (e - p).powi(2))
                .sum::<f64>() / expected_predictions.len() as f64;
            
            println!("✓ Brier Score: {:.3} (lower is better)", brier_score);
            assert!(brier_score < 0.25, "Brier Score too high: {}", brier_score);
        }
        
        // ======================================================================
        // SECTION 3: PGS EFFECT SIZE ASSESSMENT
        // ======================================================================
        println!("\n3. VALIDATING PGS EFFECT SIZE AND DIRECTION...");
        
        // 3.1 Test that the model captures PGS main effect direction correctly
        // Sort samples by PGS value
        let mut pgs_with_predictions: Vec<(f64, f64)> = data.iter()
            .map(|row| row[0])  // Get PGS value
            .zip(predictions.iter().copied())
            .collect();
        
        pgs_with_predictions.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Split into deciles for more granular analysis
        let n_samples = pgs_with_predictions.len();
        let decile_size = n_samples / 10;
        
        // Calculate mean prediction for each decile
        let mut decile_means = Vec::with_capacity(10);
        for i in 0..10 {
            let start = i * decile_size;
            let end = if i == 9 { n_samples } else { (i + 1) * decile_size };
            
            let decile_mean = pgs_with_predictions[start..end]
                .iter()
                .map(|(_, pred)| pred)
                .sum::<f64>() / (end - start) as f64;
                
            decile_means.push(decile_mean);
        }
        
        // Test for monotonically increasing trend in deciles
        let mut monotonic_violations = 0;
        for i in 1..10 {
            if decile_means[i] < decile_means[i-1] {
                monotonic_violations += 1;
            }
        }
        
        // Allow some violations due to noise, but overall trend should be monotonic
        println!("✓ PGS decile means: [{:.3}, {:.3}, {:.3}, ..., {:.3}]", 
                 decile_means[0], decile_means[1], decile_means[2], decile_means[9]);
        println!("✓ Monotonicity violations: {} (out of 9 transitions)", monotonic_violations);
        assert!(monotonic_violations <= 2, "Too many monotonicity violations in PGS effect: {}", monotonic_violations);
        
        // Also check overall trend from lowest to highest decile
        assert!(decile_means[9] > decile_means[0], 
                "Model didn't capture positive PGS effect: lowest decile = {:.3}, highest decile = {:.3}", 
                decile_means[0], decile_means[9]);
        
        // Calculate fold increase from lowest to highest decile (effect size measure)
        let fold_increase = decile_means[9] / decile_means[0];
        println!("✓ Fold increase (highest/lowest decile): {:.2}x", fold_increase);
        
        // ======================================================================
        // SECTION 4: INTERACTION EFFECT VALIDATION
        // ======================================================================
        println!("\n4. VALIDATING INTERACTION EFFECTS...");
        
        // 4.1 Create a function to test interaction effects for any PC
        fn test_pc_interaction(
            pc_index: usize,
            pc_name: &str,
            interaction_coef: f64,
            data: &Vec<Vec<f64>>,
            predictions: &[f64],
        ) -> bool {
            // Group by PC quartiles
            let mut pc_values_with_index: Vec<(f64, usize)> = data.iter()
                .map(|row| row[pc_index])
                .enumerate()
                .map(|(i, pc)| (pc, i))
                .collect();
            
            pc_values_with_index.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            let quartile_size = pc_values_with_index.len() / 4;
            let q1_indices: Vec<usize> = pc_values_with_index[..quartile_size]
                .iter()
                .map(|(_, idx)| *idx)
                .collect();
                
            let q4_indices: Vec<usize> = pc_values_with_index[3*quartile_size..]
                .iter()
                .map(|(_, idx)| *idx)
                .collect();
            
            // For each quartile group, assess PGS effect by dividing into low/high PGS
            let mut q1_pgs_values: Vec<(f64, f64)> = q1_indices.iter()
                .map(|&idx| (data[idx][0], predictions[idx]))
                .collect();
                
            let mut q4_pgs_values: Vec<(f64, f64)> = q4_indices.iter()
                .map(|&idx| (data[idx][0], predictions[idx]))
                .collect();
                
            q1_pgs_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            q4_pgs_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            let q1_size = q1_pgs_values.len();
            let q4_size = q4_pgs_values.len();
            
            let q1_low_pgs_pred = q1_pgs_values[..q1_size/2]
                .iter()
                .map(|(_, pred)| *pred)
                .sum::<f64>() / (q1_size/2) as f64;
                
            let q1_high_pgs_pred = q1_pgs_values[q1_size/2..]
                .iter()
                .map(|(_, pred)| *pred)
                .sum::<f64>() / (q1_size - q1_size/2) as f64;
                
            let q4_low_pgs_pred = q4_pgs_values[..q4_size/2]
                .iter()
                .map(|(_, pred)| *pred)
                .sum::<f64>() / (q4_size/2) as f64;
                
            let q4_high_pgs_pred = q4_pgs_values[q4_size/2..]
                .iter()
                .map(|(_, pred)| *pred)
                .sum::<f64>() / (q4_size - q4_size/2) as f64;
                
            let q1_pgs_effect = q1_high_pgs_pred - q1_low_pgs_pred;
            let q4_pgs_effect = q4_high_pgs_pred - q4_low_pgs_pred;
            
            println!("   {} interaction analysis:", pc_name);
            println!("     - {} Q1 PGS effect: {:.3}", pc_name, q1_pgs_effect);
            println!("     - {} Q4 PGS effect: {:.3}", pc_name, q4_pgs_effect);
            println!("     - Ratio (Q4/Q1): {:.2}x", q4_pgs_effect / q1_pgs_effect);
            
            // Test if the interaction direction matches our expectation
            let expected_effect = if interaction_coef > 0.0 {
                q4_pgs_effect > q1_pgs_effect
            } else if interaction_coef < 0.0 {
                q4_pgs_effect < q1_pgs_effect
            } else {
                // No expected interaction
                true
            };
            
            expected_effect
        }
        
        // 4.2 Test all PCs with significant interaction effects
        let mut interaction_tests_passed = true;
        for (i, pc_name) in (1..=n_pcs).map(|i| format!("PC{}", i)).enumerate() {
            if let Some(&interaction_coef) = true_effects.interaction_effects.get(&pc_name) {
                if interaction_coef.abs() > 0.2 {  // Only test significant interactions
                    let test_result = test_pc_interaction(
                        i+1,  // PC index in data row (add 1 because first column is PGS)
                        &pc_name,
                        interaction_coef,
                        data,
                        predictions
                    );
                    
                    if !test_result {
                        println!("✗ Failed to detect expected {} interaction (coef: {})", pc_name, interaction_coef);
                        interaction_tests_passed = false;
                    } else {
                        println!("✓ Correctly detected {} interaction (coef: {})", pc_name, interaction_coef);
                    }
                }
            }
        }
        
        // At least one interaction test should pass
        assert!(interaction_tests_passed, "Failed to detect any expected interaction effects");
        
        // ======================================================================
        // SECTION 5: CALIBRATION ASSESSMENT
        // ======================================================================
        println!("\n5. VALIDATING MODEL CALIBRATION...");
        
        // 5.1 Hosmer-Lemeshow test (simplified version)
        // Group predictions into 10 equal-sized bins
        let mut predictions_sorted: Vec<(f64, f64)> = predictions.iter()
            .zip(expected_predictions.iter())
            .map(|(&pred, &exp)| (pred, exp))
            .collect();
            
        predictions_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        
        // Create 10 equal-sized bins
        let bin_size = predictions_sorted.len() / 10;
        
        // For each bin, calculate mean predicted and expected probabilities
        let mut calibration_errors = Vec::with_capacity(10);
        for i in 0..10 {
            let start = i * bin_size;
            let end = if i == 9 { predictions_sorted.len() } else { (i + 1) * bin_size };
            
            let bin_pred_mean = predictions_sorted[start..end]
                .iter()
                .map(|(pred, _)| *pred)
                .sum::<f64>() / (end - start) as f64;
                
            let bin_exp_mean = predictions_sorted[start..end]
                .iter()
                .map(|(_, exp)| *exp)
                .sum::<f64>() / (end - start) as f64;
                
            calibration_errors.push((bin_pred_mean - bin_exp_mean).abs());
        }
        
        // Calculate mean absolute calibration error
        let mean_calibration_error = calibration_errors.iter().sum::<f64>() / 10.0;
        println!("✓ Mean absolute calibration error: {:.3}", mean_calibration_error);
        assert!(mean_calibration_error < 0.1, "Calibration error too high: {}", mean_calibration_error);
        
        // Count bins where calibration error is acceptable (< 0.05)
        let well_calibrated_bins = calibration_errors.iter().filter(|&&e| e < 0.05).count();
        println!("✓ Well-calibrated bins (error < 0.05): {}/10", well_calibrated_bins);
        assert!(well_calibrated_bins >= 5, "Too few well-calibrated bins: {}", well_calibrated_bins);
        
        println!("\nAll validation checks passed successfully!");
        
        // Return metrics for model comparison
        ValidationMetrics {
            correlation,
            r_squared,
            mae,
            auc,
            brier_score,
            calibration_error: mean_calibration_error,
            monotonicity_score: 9.0 - monotonic_violations as f64,
            pc1_interaction_detected: test_pc_interaction(
                1, "PC1", 
                *true_effects.interaction_effects.get("PC1").unwrap_or(&0.0),
                data, predictions
            ),
            pc2_interaction_detected: test_pc_interaction(
                2, "PC2", 
                *true_effects.interaction_effects.get("PC2").unwrap_or(&0.0),
                data, predictions
            ),
        }
    }
}
