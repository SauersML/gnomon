use clap::{Parser, Subcommand};
use std::process;
use std::path::PathBuf;

// Include the calibrate module files
#[path = "../calibrate/basis.rs"]
mod basis;
#[path = "../calibrate/data.rs"] 
mod data;
#[path = "../calibrate/estimate.rs"]
mod estimate;
#[path = "../calibrate/model.rs"]
mod model;

// Import the calibrate functionality
use basis::BasisConfig;
use data::{load_prediction_data, load_training_data};
use estimate::train_model;
use model::{LinkFunction, ModelConfig, TrainedModel};

use clap::Args;
use ndarray::{Array1, ArrayView1};
use std::collections::HashSet;

#[derive(Args)]
pub struct TrainArgs {
        /// Path to training TSV file with phenotype,score,PC1,PC2,... columns
        pub training_data: String,

        /// Number of principal components to use from the data
        #[arg(long, value_name = "N")]
        pub num_pcs: usize,

        /// Number of internal knots for PGS spline basis
        #[arg(long, default_value = "10")]
        pub pgs_knots: usize,

        /// Polynomial degree for PGS spline basis
        #[arg(long, default_value = "3")]
        pub pgs_degree: usize,

        /// Number of internal knots for PC spline bases
        #[arg(long, default_value = "5")]
        pub pc_knots: usize,

        /// Polynomial degree for PC spline bases  
        #[arg(long, default_value = "2")]
        pub pc_degree: usize,

        /// Order of the difference penalty matrix
        #[arg(long, default_value = "2")]
        pub penalty_order: usize,

        /// Maximum number of IRLS iterations
        #[arg(long, default_value = "50")]
        pub max_iter: usize,

        /// Convergence tolerance for IRLS
        #[arg(long, default_value = "1e-6")]
        pub tolerance: f64,
    }

#[derive(Args)]
pub struct InferArgs {
        /// Path to test TSV file with score,PC1,PC2,... columns (no phenotype needed)
        pub test_data: String,

        /// Path to trained model file (.toml)
        #[arg(long)]
        pub model: String,
    }

pub fn train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading training data from: {}", args.training_data);

        // Load training data
        let data = load_training_data(&args.training_data, args.num_pcs)?;
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
        let pc_names: Vec<String> = (1..=args.num_pcs).map(|i| format!("PC{}", i)).collect();

        // Create basis configurations
        let pgs_basis_config = BasisConfig {
            num_knots: args.pgs_knots,
            degree: args.pgs_degree,
        };

        let pc_basis_configs = vec![
            BasisConfig {
                num_knots: args.pc_knots,
                degree: args.pc_degree,
            };
            args.num_pcs
        ];

        // Use default lambda value tuned for genomics applications
        let lambda = 0.001;
        println!("Using lambda: {:.6e}", lambda);

        // Create final model configuration
        let config = ModelConfig {
            link_function,
            penalty_order: args.penalty_order,
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

pub fn infer(args: InferArgs) -> Result<(), Box<dyn std::error::Error>> {
        println!("Loading model from: {}", args.model);

        // Load trained model
        let model = TrainedModel::load(&args.model)?;
        let num_pcs = model.config.pc_names.len();

        println!("Model expects {} PCs", num_pcs);

        // Load test data
        println!("Loading test data from: {}", args.test_data);
        let data = load_prediction_data(&args.test_data, num_pcs)?;
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

fn calculate_range(data: ArrayView1<f64>) -> (f64, f64) {
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min_val, max_val)
    }

fn save_predictions(predictions: &Array1<f64>, output_path: &str) -> Result<(), std::io::Error> {
    use std::io::Write;
    
    let mut file = std::fs::File::create(output_path)?;
    writeln!(file, "prediction")?;
    
    for &pred in predictions.iter() {
        writeln!(file, "{:.6}", pred)?;
    }
    
    Ok(())
}

// Import score functionality  
#[path = "../score/main.rs"]
mod score_main;

#[derive(Parser)]
#[command(
    name = "gnomon",
    about = "High-performance polygenic score calculation and calibration toolkit",
    long_about = "A comprehensive toolkit for calculating polygenic scores from genotype data \
                 and training GAM models for score calibration."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate polygenic scores from genotype data
    #[command(about = "Calculate raw polygenic scores")]
    Score {
        /// Path to PLINK .bed file or directory containing .bed files
        input_path: PathBuf,
        
        /// Path to score file or directory containing score files
        #[arg(long)]
        score: PathBuf,
        
        /// Path to file containing list of individual IDs to include (optional)
        #[arg(long)]
        keep: Option<PathBuf>,
    },

    /// Train a GAM calibration model from training data
    #[command(about = "Train GAM calibration model (outputs: model.toml)")]
    Train(TrainArgs),

    /// Apply trained calibration model to new data
    #[command(about = "Apply calibration model to new data (outputs: predictions.tsv)")]
    Infer(InferArgs),
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Score { input_path, score, keep } => {
            run_score(input_path, score, keep)
        }
        Commands::Train(args) => train(args),
        Commands::Infer(args) => infer(args),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

fn run_score(input_path: PathBuf, score: PathBuf, keep: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    // Call the score calculation logic directly
    score_main::run_gnomon_with_args(input_path, score, keep)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}
