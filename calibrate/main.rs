use basis::BasisConfig;
use data::{load_prediction_data, load_training_data};
use estimate::train_model;
use model::{LinkFunction, ModelConfig, TrainedModel};

use clap::{Parser, Subcommand};
use ndarray::{Array1, Array2, ArrayView1};
use std::collections::HashSet;
use std::process;

mod basis;
mod data;
mod estimate;
mod model;

#[derive(Parser)]
#[command(
    name = "gnomon-calibrate",
    about = "Train and apply GAM models for polygenic score calibration",
    long_about = "A tool for training Generalized Additive Models (GAMs) to calibrate polygenic scores \
                 using B-spline basis functions with smoothing penalties."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a new GAM model from training data
    #[command(about = "Train a GAM model (outputs: model.toml)")]
    Train {
        /// Path to training TSV file with phenotype,score,PC1,PC2,... columns
        training_data: String,

        /// Number of principal components to use from the data
        #[arg(long, value_name = "N")]
        num_pcs: usize,

        /// Number of internal knots for PGS spline basis
        #[arg(long, default_value = "10")]
        pgs_knots: usize,

        /// Polynomial degree for PGS spline basis
        #[arg(long, default_value = "3")]
        pgs_degree: usize,

        /// Number of internal knots for PC spline bases
        #[arg(long, default_value = "5")]
        pc_knots: usize,

        /// Polynomial degree for PC spline bases  
        #[arg(long, default_value = "2")]
        pc_degree: usize,

        /// Order of the difference penalty matrix
        #[arg(long, default_value = "2")]
        penalty_order: usize,

        /// Maximum number of IRLS iterations
        #[arg(long, default_value = "50")]
        max_iter: usize,

        /// Convergence tolerance for IRLS
        #[arg(long, default_value = "1e-6")]
        tolerance: f64,
    },

    /// Apply a trained model to new data for prediction
    #[command(about = "Apply trained model to new data (outputs: predictions.tsv)")]
    Infer {
        /// Path to test TSV file with score,PC1,PC2,... columns (no phenotype needed)
        test_data: String,

        /// Path to trained model file (.toml)
        #[arg(long)]
        model: String,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Train {
            training_data,
            num_pcs,
            pgs_knots,
            pgs_degree,
            pc_knots,
            pc_degree,
            penalty_order,
            max_iter,
            tolerance,
        } => train_command(
            &training_data,
            num_pcs,
            pgs_knots,
            pgs_degree,
            pc_knots,
            pc_degree,
            penalty_order,
            max_iter,
            tolerance,
        ),
        Commands::Infer { test_data, model } => infer_command(&test_data, &model),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn train_command(
    training_data_path: &str,
    num_pcs: usize,
    pgs_knots: usize,
    pgs_degree: usize,
    pc_knots: usize,
    pc_degree: usize,
    penalty_order: usize,
    max_iter: usize,
    tolerance: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading training data from: {}", training_data_path);

    // Load training data
    let data = load_training_data(training_data_path, num_pcs)?;
    println!(
        "Loaded {} samples with {} PCs",
        data.y.len(),
        data.pcs.ncols()
    );

    // Auto-detect link function based on phenotype
    let link_function = detect_link_function(&data.y);
    println!("Auto-detected link function: {:?}", link_function);

    // Calculate data ranges for basis construction
    let pgs_range = calculate_range(data.p.view());
    let pc_ranges: Vec<(f64, f64)> = (0..data.pcs.ncols())
        .map(|i| calculate_range(data.pcs.column(i)))
        .collect();

    println!("PGS range: ({:.3}, {:.3})", pgs_range.0, pgs_range.1);
    println!(
        "PC ranges: {}",
        pc_ranges
            .iter()
            .enumerate()
            .map(|(i, &(min, max))| format!("PC{}: ({:.3}, {:.3})", i + 1, min, max))
            .collect::<Vec<_>>()
            .join(", ")
    );

    // Generate PC names
    let pc_names: Vec<String> = (1..=num_pcs).map(|i| format!("PC{}", i)).collect();

    // Create basis configurations
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

    // Use default lambda value tuned for genomics applications
    let lambda = 0.001;
    println!("Using lambda: {:.6e}", lambda);

    // Create final model configuration
    let config = ModelConfig {
        link_function,
        penalty_order,
        lambda,
        pgs_basis_config,
        pc_basis_configs,
        pgs_range,
        pc_ranges,
        pc_names,
    };

    // Train the final model
    println!("Training final model...");
    let trained_model = train_model(&data, &config)?;

    // Save model to hardcoded output path
    let output_path = "model.toml";
    trained_model.save(output_path)?;
    println!("Model saved to: {}", output_path);

    Ok(())
}

fn infer_command(test_data_path: &str, model_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model from: {}", model_path);

    // Load trained model
    let model = TrainedModel::load(model_path)?;
    let num_pcs = model.config.pc_names.len();

    println!("Model expects {} PCs", num_pcs);

    // Load test data
    println!("Loading test data from: {}", test_data_path);
    let data = load_prediction_data(test_data_path, num_pcs)?;
    println!("Loaded {} samples for prediction", data.p.len());

    // Make predictions
    println!("Generating predictions...");
    let predictions = model.predict(data.p.view(), data.pcs.view())?;

    // Save predictions to hardcoded output path
    let output_path = "predictions.tsv";
    save_predictions(&predictions, output_path)?;
    println!("Predictions saved to: {}", output_path);

    Ok(())
}

/// Auto-detect link function based on number of unique phenotype values
fn detect_link_function(phenotype: &Array1<f64>) -> LinkFunction {
    let unique_values: HashSet<_> = phenotype.iter().map(|&x| x as i64).collect();
    
    if unique_values.len() == 2 {
        // Binary case/control data
        LinkFunction::Logit
    } else {
        // Continuous quantitative trait
        LinkFunction::Identity
    }
}

/// Calculate the min/max range of a data vector
fn calculate_range(data: ArrayView1<f64>) -> (f64, f64) {
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    (min_val, max_val)
}



/// Save predictions to a TSV file
fn save_predictions(predictions: &Array1<f64>, output_path: &str) -> Result<(), std::io::Error> {
    use std::io::Write;
    
    let mut file = std::fs::File::create(output_path)?;
    writeln!(file, "prediction")?;
    
    for &pred in predictions.iter() {
        writeln!(file, "{:.6}", pred)?;
    }
    
    Ok(())
}
