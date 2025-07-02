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
    },
}

/// Main application entry point.
fn main() {
    // Initialize a simple logger. Use `RUST_LOG=info` environment variable to see logs.
    env_logger::init();
    let cli = Cli::parse();

    let result = match cli.command {
        // --- Corrected Train Command ---
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
        ),
        Commands::Infer { test_data, model } => infer_command(&test_data, &model),
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
    };

    // --- Train the model using REML ---
    // This function call triggers the full REML optimization pipeline in `estimate.rs`.
    // It will automatically find the optimal smoothing parameters (lambdas).
    log::info!("Starting model training with REML/LAML optimization...");
    let trained_model = train_model(&data, &config)?;

    // Save the final, trained model to disk.
    let output_path = "model.toml";
    trained_model.save(output_path)?;
    log::info!("Model training complete. Saved to: {}", output_path);

    Ok(())
}

/// Handles the `infer` command logic. This function remains unchanged.
fn infer_command(test_data_path: &str, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
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
    let output_path = "predictions.tsv";
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
