#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

#[cfg(feature = "calibrate")]
use clap::ValueEnum;
use clap::{Args, Parser};
#[cfg(any(feature = "map", feature = "calibrate"))]
use clap::{CommandFactory, Subcommand};
#[cfg(feature = "calibrate")]
use gam::probability::normal_cdf_approx;
#[cfg(feature = "calibrate")]
use gnomon::calibrate::data::{load_prediction_data, load_training_data};
#[cfg(feature = "calibrate")]
use gnomon::calibrate::estimate::{train_model, train_survival_model};
#[cfg(feature = "calibrate")]
use gnomon::calibrate::model::BasisConfig;
#[cfg(feature = "calibrate")]
use gnomon::calibrate::model::SurvivalModelConfig;
#[cfg(feature = "calibrate")]
use gnomon::calibrate::model::SurvivalPrediction;
#[cfg(feature = "calibrate")]
use gnomon::calibrate::model::SurvivalRiskType;
#[cfg(feature = "calibrate")]
use gnomon::calibrate::model::SurvivalTimeVaryingConfig;
#[cfg(feature = "calibrate")]
use gnomon::calibrate::model::{LinkFunction, ModelConfig, ModelFamily, TrainedModel};
#[cfg(feature = "calibrate")]
use gnomon::calibrate::survival::SurvivalSpec;
#[cfg(feature = "calibrate")]
use gnomon::calibrate::survival_data::{
    SurvivalPredictionData, has_survival_columns, load_survival_prediction_data,
    load_survival_training_data,
};
#[cfg(feature = "map")]
use gnomon::map::main as map_cli;
#[cfg(feature = "map")]
use gnomon::map::{DEFAULT_LD_WINDOW, LdWindow};
#[cfg(feature = "terms")]
use gnomon::terms::infer_sex_to_tsv;
#[cfg(feature = "calibrate")]
use ndarray::{Array1, ArrayView1};
#[cfg(feature = "calibrate")]
use std::collections::HashSet;
#[cfg(any(feature = "score", feature = "map", feature = "terms"))]
use std::path::PathBuf;
use std::process;

#[cfg(feature = "score")]
#[path = "../score/main.rs"]
mod score_main;

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
#[path = "all.rs"]
mod all_cmd;

#[cfg(feature = "score")]
#[derive(Args)]
struct ScoreArgs {
    /// Path to a single score file or a directory containing multiple score files.
    #[arg(value_name = "SCORE_PATH")]
    score: PathBuf,

    /// Path to a file containing a list of individual IDs (IIDs) to include.
    #[arg(long)]
    keep: Option<PathBuf>,

    /// Path to genotype data (PLINK .bed/.bim/.fam prefix, VCF, BCF, or DTC text file)
    #[arg(value_name = "GENOTYPE_PATH")]
    input_path: PathBuf,

    /// Reference genome FASTA (optional; auto-downloaded if not provided for DTC files)
    #[arg(long)]
    reference: Option<PathBuf>,

    /// Force genome build (37 or 38); auto-detected if not provided
    #[arg(long)]
    build: Option<String>,

    /// Reference panel VCF for strand harmonization
    #[arg(long)]
    panel: Option<PathBuf>,
}

#[cfg(feature = "map")]
#[derive(Args)]
struct FitArgs {
    /// Path to PLINK .bed file or directory containing .bed files
    #[arg(value_name = "GENOTYPE_PATH")]
    genotype_path: PathBuf,

    /// Optional variant list limiting SNVs used for PCA fitting
    #[arg(long, value_name = "PATH")]
    list: Option<PathBuf>,

    /// Number of principal components to retain when fitting the HWE PCA model
    #[arg(long, value_name = "N")]
    components: usize,

    /// Enable LD normalization when fitting the PCA model
    #[arg(long)]
    ld: bool,

    /// LD window expressed as the number of sites (must be odd)
    #[arg(
        long = "sites_window",
        value_name = "SITES",
        requires = "ld",
        conflicts_with = "bp_window"
    )]
    sites_window: Option<usize>,

    /// LD window expressed as the total span in base pairs
    #[arg(
        long = "bp_window",
        value_name = "BP",
        requires = "ld",
        conflicts_with = "sites_window"
    )]
    bp_window: Option<u64>,
}

#[cfg(feature = "map")]
#[derive(Args)]
struct ProjectArgs {
    /// Path to PLINK .bed file or directory containing .bed files
    #[arg(value_name = "GENOTYPE_PATH")]
    genotype_path: PathBuf,

    /// Use a built-in pre-trained model (downloads from GitHub if needed)
    #[arg(long, value_name = "MODEL_NAME")]
    model: Option<String>,

    /// Write a JSON manifest describing the exact projection outputs that were created
    #[arg(long, value_name = "PATH")]
    output_manifest: Option<PathBuf>,
}

#[cfg(feature = "terms")]
#[derive(Args)]
struct TermsArgs {
    /// Path to genotype dataset (PLINK .bed/.bim/.fam prefix or VCF/BCF file)
    #[arg(value_name = "GENOTYPE_PATH")]
    genotype_path: PathBuf,

    /// Run sex inference on the provided genotype dataset
    #[arg(long)]
    sex: bool,
}

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
#[derive(Args)]
struct AllArgs {
    /// Path to a single score file or a directory containing multiple score files.
    #[arg(value_name = "SCORE_PATH")]
    score: PathBuf,

    /// Path to genotype data (VCF/BCF strongly preferred; PLINK also accepted).
    #[arg(value_name = "GENOTYPE_PATH")]
    input_path: PathBuf,

    /// Built-in HWE-PCA model name used for projection (e.g. hwe_1kg_hgdp_gsa_v3).
    #[arg(long, value_name = "MODEL_NAME")]
    model: String,

    /// Path to a file containing a list of individual IDs (IIDs) to include.
    #[arg(long)]
    keep: Option<PathBuf>,

    /// Reference genome FASTA (optional; auto-downloaded if not provided for DTC files).
    #[arg(long)]
    reference: Option<PathBuf>,

    /// Force genome build (37 or 38); auto-detected if not provided.
    #[arg(long)]
    build: Option<String>,

    /// Reference panel VCF for strand harmonization.
    #[arg(long)]
    panel: Option<PathBuf>,

    /// Write a JSON manifest describing the exact projection outputs that were created.
    #[arg(long, value_name = "PATH")]
    output_manifest: Option<PathBuf>,
}

#[cfg(feature = "calibrate")]
#[derive(Clone, ValueEnum)]
enum ModelFamilyCli {
    Gam,
    Survival,
}

#[cfg(feature = "calibrate")]
#[derive(Args)]
struct TrainArgs {
    #[arg(long, value_enum, default_value_t = ModelFamilyCli::Gam)]
    model_family: ModelFamilyCli,

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

    /// Maximum number of P-IRLS iterations for the inner loop (per REML step)
    #[arg(long, default_value = "50")]
    max_iterations: usize,

    /// Convergence tolerance for the P-IRLS inner loop deviance change
    #[arg(long, default_value = "1e-7")]
    convergence_tolerance: f64,

    /// Maximum number of iterations for the outer REML/BFGS optimization loop
    #[arg(long, default_value = "100")]
    reml_max_iterations: u64,

    /// Convergence tolerance for the gradient norm in the outer REML/BFGS loop
    #[arg(long, default_value = "1e-3")]
    reml_convergence_tolerance: f64,

    /// Guard delta for the survival age transform
    #[arg(long, default_value = "0.1")]
    survival_guard_delta: f64,

    /// Number of internal knots for the survival baseline spline
    #[arg(long, default_value = "6")]
    survival_baseline_knots: usize,

    /// Degree for the survival baseline spline
    #[arg(long, default_value = "3")]
    survival_baseline_degree: usize,

    /// Grid size for the survival monotonicity penalty
    #[arg(long, default_value = "64")]
    survival_monotonic_grid: usize,

    /// Derivative guard threshold used inside the survival likelihood
    #[arg(long, default_value = "1e-8")]
    survival_derivative_guard: f64,

    /// Use expected information instead of observed Hessian when fitting survival models
    #[arg(long)]
    survival_expected_information: bool,

    /// Enable the optional time-varying PGS × age interaction
    #[arg(long)]
    survival_enable_time_varying: bool,

    /// Number of internal knots for the time-varying PGS spline
    #[arg(long, default_value = "5")]
    survival_time_varying_pgs_knots: usize,

    /// Degree for the time-varying PGS spline
    #[arg(long, default_value = "3")]
    survival_time_varying_pgs_degree: usize,

    /// Difference-penalty order for the time-varying PGS spline
    #[arg(long, default_value = "2")]
    survival_time_varying_pgs_penalty_order: usize,
}

#[cfg(feature = "calibrate")]
#[derive(Args)]
struct InferArgs {
    /// Path to test TSV file with score,PC1,PC2,... columns (no phenotype needed)
    test_data: String,

    /// Path to trained model file (.toml)
    #[arg(long)]
    model: String,
}

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
#[derive(Parser)]
#[command(
    name = "gnomon",
    about = "High-performance polygenic score calculation and calibration toolkit",
    long_about = "A comprehensive toolkit for calculating polygenic scores from genotype data \
                 and training GAM models for score calibration."
)]
struct FullCli {
    #[command(subcommand)]
    command: Option<FullCommands>,
}

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
#[derive(Subcommand)]
enum FullCommands {
    /// Calculate raw polygenic scores
    Score(ScoreArgs),
    /// Fit an HWE PCA model
    Fit(FitArgs),
    /// Project samples using an existing HWE PCA model
    Project(ProjectArgs),
    /// Infer sample metadata terms
    Terms(TermsArgs),
    /// Train GAM calibration model
    Train(TrainArgs),
    /// Apply calibration model to new data
    Infer(InferArgs),
    /// Run score + project + terms against a single VCF/BCF without rescanning it three times.
    All(AllArgs),
    /// Display version and build information
    Version,
}

#[cfg(feature = "score")]
#[derive(Parser)]
#[command(
    name = "gnomon-score",
    about = "Calculate raw polygenic scores from genotype data"
)]
struct ScoreCli {
    #[command(flatten)]
    args: ScoreArgs,
}

#[cfg(feature = "map")]
#[derive(Parser)]
#[command(name = "gnomon-map", about = "Fit or project HWE PCA models")]
struct MapCli {
    #[command(subcommand)]
    command: Option<MapCommands>,
}

#[cfg(feature = "map")]
#[derive(Subcommand)]
enum MapCommands {
    Fit(FitArgs),
    Project(ProjectArgs),
}

#[cfg(feature = "terms")]
#[derive(Parser)]
#[command(
    name = "gnomon-terms",
    about = "Infer sample metadata terms from genotype data"
)]
struct TermsCli {
    #[command(flatten)]
    args: TermsArgs,
}

#[cfg(feature = "calibrate")]
#[derive(Parser)]
#[command(name = "gnomon-calibrate", about = "Train or apply calibration models")]
struct CalibrateCli {
    #[command(subcommand)]
    command: Option<CalibrateCommands>,
}

#[cfg(feature = "calibrate")]
#[derive(Subcommand)]
enum CalibrateCommands {
    Train(TrainArgs),
    Infer(InferArgs),
}

fn main() {
    let result = dispatch_current_binary();
    if let Err(err) = result {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn dispatch_current_binary() -> Result<(), Box<dyn std::error::Error>> {
    match env!("CARGO_BIN_NAME") {
        #[cfg(all(
            feature = "map",
            feature = "score",
            feature = "calibrate",
            feature = "terms"
        ))]
        "gnomon" => run_full_entrypoint(),
        #[cfg(feature = "score")]
        "gnomon-score" => run_score_entrypoint(),
        #[cfg(feature = "map")]
        "gnomon-map" => run_map_entrypoint(),
        #[cfg(feature = "terms")]
        "gnomon-terms" => run_terms_entrypoint(),
        #[cfg(feature = "calibrate")]
        "gnomon-calibrate" => run_calibrate_entrypoint(),
        other => Err(format!("unsupported binary '{other}' for this feature set").into()),
    }
}

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
fn run_full_entrypoint() -> Result<(), Box<dyn std::error::Error>> {
    let cli = FullCli::parse();
    match cli.command {
        Some(FullCommands::Score(args)) => run_score(args),
        Some(FullCommands::Fit(args)) => run_map_fit(args),
        Some(FullCommands::Project(args)) => run_map_project(args),
        Some(FullCommands::Terms(args)) => run_terms(args),
        Some(FullCommands::Train(args)) => train(args),
        Some(FullCommands::Infer(args)) => infer(args),
        Some(FullCommands::All(args)) => run_all(args),
        Some(FullCommands::Version) => {
            print_version_info();
            Ok(())
        }
        None => {
            FullCli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

#[cfg(feature = "score")]
fn run_score_entrypoint() -> Result<(), Box<dyn std::error::Error>> {
    let cli = ScoreCli::parse();
    run_score(cli.args)
}

#[cfg(feature = "map")]
fn run_map_entrypoint() -> Result<(), Box<dyn std::error::Error>> {
    let cli = MapCli::parse();
    match cli.command {
        Some(MapCommands::Fit(args)) => run_map_fit(args),
        Some(MapCommands::Project(args)) => run_map_project(args),
        None => {
            MapCli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

#[cfg(feature = "terms")]
fn run_terms_entrypoint() -> Result<(), Box<dyn std::error::Error>> {
    let cli = TermsCli::parse();
    run_terms(cli.args)
}

#[cfg(feature = "calibrate")]
fn run_calibrate_entrypoint() -> Result<(), Box<dyn std::error::Error>> {
    let cli = CalibrateCli::parse();
    match cli.command {
        Some(CalibrateCommands::Train(args)) => train(args),
        Some(CalibrateCommands::Infer(args)) => infer(args),
        None => {
            CalibrateCli::command().print_help()?;
            println!();
            Ok(())
        }
    }
}

#[cfg(feature = "score")]
fn run_score(args: ScoreArgs) -> Result<(), Box<dyn std::error::Error>> {
    score_main::run_gnomon_with_args(
        args.input_path,
        args.score,
        args.keep,
        args.reference,
        args.build,
        args.panel,
    )
    .map_err(|err| err as Box<dyn std::error::Error>)
}

#[cfg(feature = "map")]
fn run_map_fit(args: FitArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ld_window = if args.ld {
        if let Some(bp) = args.bp_window {
            Some(LdWindow::BasePairs(bp))
        } else {
            let window = args.sites_window.unwrap_or(DEFAULT_LD_WINDOW);
            if window == 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--sites_window must be at least 1",
                )));
            }
            if window % 2 == 0 {
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--sites_window must be an odd number",
                )));
            }
            Some(LdWindow::Sites(window))
        }
    } else {
        None
    };

    map_cli::run(map_cli::MapCommand::Fit {
        genotype_path: args.genotype_path,
        variant_list: args.list,
        components: args.components,
        ld: ld_window,
    })
    .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
}

#[cfg(feature = "map")]
fn run_map_project(args: ProjectArgs) -> Result<(), Box<dyn std::error::Error>> {
    map_cli::run(map_cli::MapCommand::Project {
        genotype_path: args.genotype_path,
        model: args.model,
        output_manifest: args.output_manifest,
    })
    .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
}

#[cfg(feature = "terms")]
fn run_terms(args: TermsArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.sex {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "No term inference selected. Use --sex to run sex inference.",
        )));
    }

    let output_path = infer_sex_to_tsv(&args.genotype_path, None)
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    println!("Sex inference results written to {}", output_path.display());
    Ok(())
}

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
fn run_all(args: AllArgs) -> Result<(), Box<dyn std::error::Error>> {
    all_cmd::run(all_cmd::AllOptions {
        score: args.score,
        input_path: args.input_path,
        model: args.model,
        keep: args.keep,
        reference: args.reference,
        build: args.build,
        panel: args.panel,
        output_manifest: args.output_manifest,
    })
}

#[cfg(feature = "calibrate")]
fn train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.model_family {
        ModelFamilyCli::Gam => {
            if has_survival_columns(&args.training_data)? {
                println!(
                    "[AUTO] Detected survival columns in {}; training survival model.",
                    args.training_data
                );
                return train_survival_from_args(&args);
            }

            println!("Loading training data from: {}", args.training_data);
            let data = load_training_data(&args.training_data, args.num_pcs)?;
            println!(
                "Loaded {} samples with {} PCs",
                data.y.len(),
                data.pcs.ncols()
            );

            let link_function = detect_link_function(&data.y);
            println!("Auto-detected link function: {link_function:?}");

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

            let pgs_basis_config = BasisConfig {
                num_knots: args.pgs_knots,
                degree: args.pgs_degree,
            };

            let pc_configs = (0..args.num_pcs)
                .map(|i| gnomon::calibrate::model::PrincipalComponentConfig {
                    name: format!("PC{}", i + 1),
                    basis_config: BasisConfig {
                        num_knots: args.pc_knots,
                        degree: args.pc_degree,
                    },
                    range: pc_ranges[i],
                })
                .collect();

            println!("Training model with REML estimation of smoothing parameters");
            let config = ModelConfig {
                model_family: ModelFamily::Gam(link_function),
                penalty_order: args.penalty_order,
                convergence_tolerance: args.convergence_tolerance,
                max_iterations: args.max_iterations,
                reml_convergence_tolerance: args.reml_convergence_tolerance,
                reml_max_iterations: args.reml_max_iterations,
                pgs_basis_config,
                pc_configs,
                pgs_range,
                ..Default::default()
            };

            println!("Training final model...");
            let trained_model = train_model(&data, &config)?;
            trained_model.save("model.toml")?;
            println!("Model saved to: model.toml");
        }
        ModelFamilyCli::Survival => {
            return train_survival_from_args(&args);
        }
    }

    Ok(())
}

#[cfg(feature = "calibrate")]
fn train_survival_from_args(args: &TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "Loading survival training data from: {}",
        args.training_data
    );
    let bundle =
        load_survival_training_data(&args.training_data, args.num_pcs, args.survival_guard_delta)?;
    println!(
        "Loaded {} samples with {} PCs",
        bundle.data.age_entry.len(),
        bundle.data.pcs.ncols()
    );

    let mut spec = SurvivalSpec::default();
    spec.derivative_guard = args.survival_derivative_guard;
    spec.use_expected_information = args.survival_expected_information;

    let time_varying = if args.survival_enable_time_varying {
        Some(SurvivalTimeVaryingConfig {
            label: Some("pgs_by_age".to_string()),
            pgs_basis: BasisConfig {
                num_knots: args.survival_time_varying_pgs_knots,
                degree: args.survival_time_varying_pgs_degree,
            },
            pgs_penalty_order: args.survival_time_varying_pgs_penalty_order,
            lambda_age: 0.0,
            lambda_pgs: 0.0,
            lambda_null: 0.0,
        })
    } else {
        None
    };

    if let Some(settings) = &time_varying {
        println!(
            "Enabling time-varying PGS × age interaction (knots={}, degree={}, penalty order={})",
            settings.pgs_basis.num_knots, settings.pgs_basis.degree, settings.pgs_penalty_order
        );
    }

    let survival_config = SurvivalModelConfig {
        baseline_basis: BasisConfig {
            num_knots: args.survival_baseline_knots,
            degree: args.survival_baseline_degree,
        },
        guard_delta: args.survival_guard_delta,
        monotonic_grid_size: args.survival_monotonic_grid,
        time_varying,
        model_competing_risk: false,
    };

    let config = ModelConfig {
        model_family: ModelFamily::Survival(spec),
        penalty_order: args.penalty_order,
        convergence_tolerance: args.convergence_tolerance,
        max_iterations: args.max_iterations,
        reml_convergence_tolerance: args.reml_convergence_tolerance,
        reml_max_iterations: args.reml_max_iterations,
        survival: Some(survival_config),
        ..Default::default()
    };

    println!("Training survival model...");
    let trained_model = train_survival_model(&bundle, &config)?;
    trained_model.save("model.toml")?;
    println!("Model saved to: model.toml");
    Ok(())
}

#[cfg(feature = "calibrate")]
fn infer(args: InferArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model from: {}", args.model);

    let model = TrainedModel::load(&args.model)?;
    let num_pcs = model.config.pc_configs.len();
    println!("Model expects {num_pcs} PCs");

    match &model.config.model_family {
        ModelFamily::Gam(link_function) => {
            println!("Loading test data from: {}", args.test_data);
            let data = load_prediction_data(&args.test_data, num_pcs)?;
            println!("Loaded {} samples for prediction", data.p.len());

            println!("Generating predictions with diagnostics...");
            let (eta, mean, signed_dist, se_eta_opt) =
                model.predict_detailed(data.p.view(), data.sex.view(), data.pcs.view())?;

            let output_path = "predictions.tsv";
            save_predictions_detailed(
                &data.sample_ids,
                &signed_dist,
                &eta,
                &mean,
                se_eta_opt.as_ref(),
                *link_function,
                output_path,
            )?;
            println!("Predictions saved to: {output_path}");
        }
        ModelFamily::Survival(_) => {
            println!("Loading survival prediction data from: {}", args.test_data);
            let data = load_survival_prediction_data(&args.test_data, num_pcs)?;
            println!(
                "Loaded {} samples for survival prediction",
                data.age_entry.len()
            );

            println!("Generating survival predictions with diagnostics...");
            let prediction = model.predict_survival(
                data.age_entry.view(),
                data.age_exit.view(),
                data.pgs.view(),
                data.sex.view(),
                data.pcs.view(),
                SurvivalRiskType::Net,
                Some(&model.survival_companions),
            )?;

            let output_path = "predictions.tsv";
            save_survival_predictions(output_path, &data, &prediction)?;
            println!("Predictions saved to: {output_path}");
        }
    }

    Ok(())
}

#[cfg(feature = "calibrate")]
fn detect_link_function(phenotype: &Array1<f64>) -> LinkFunction {
    let unique_values: HashSet<_> = phenotype.iter().map(|&value| value as i64).collect();
    if unique_values.len() == 2 {
        LinkFunction::Logit
    } else {
        LinkFunction::Identity
    }
}

#[cfg(feature = "calibrate")]
fn calculate_range(data: ArrayView1<f64>) -> (f64, f64) {
    let min_val = data
        .iter()
        .fold(f64::INFINITY, |acc, &value| acc.min(value));
    let max_val = data
        .iter()
        .fold(f64::NEG_INFINITY, |acc, &value| acc.max(value));
    (min_val, max_val)
}

#[cfg(feature = "calibrate")]
fn write_tsv_row(
    file: &mut std::fs::File,
    base_fields: &[&dyn std::fmt::Display],
    uncalibrated: Option<&dyn std::fmt::Display>,
    tail_fields: &[&dyn std::fmt::Display],
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut first = true;
    for field in base_fields {
        if !first {
            write!(file, "\t")?;
        }
        write!(file, "{field}")?;
        first = false;
    }
    if let Some(uncalibrated_value) = uncalibrated {
        write!(file, "\t{uncalibrated_value}")?;
    }
    for field in tail_fields {
        write!(file, "\t{field}")?;
    }
    writeln!(file)?;
    Ok(())
}

#[cfg(feature = "calibrate")]
fn se_and_ci(
    se_eta_opt: Option<&Array1<f64>>,
    index: usize,
    eta_value: f64,
    link: LinkFunction,
) -> (String, String, String) {
    let Some(se_eta) = se_eta_opt else {
        return ("NA".to_string(), "NA".to_string(), "NA".to_string());
    };

    let se = se_eta[index];
    let lo_eta = eta_value - 1.959964 * se;
    let hi_eta = eta_value + 1.959964 * se;
    match link {
        LinkFunction::Identity => (se.to_string(), lo_eta.to_string(), hi_eta.to_string()),
        LinkFunction::Logit => {
            let lo_p = 1.0 / (1.0 + (-lo_eta).exp());
            let hi_p = 1.0 / (1.0 + (-hi_eta).exp());
            (
                se.to_string(),
                lo_p.clamp(0.0, 1.0).to_string(),
                hi_p.clamp(0.0, 1.0).to_string(),
            )
        }
        LinkFunction::Probit => (
            se.to_string(),
            normal_cdf_approx(lo_eta).clamp(0.0, 1.0).to_string(),
            normal_cdf_approx(hi_eta).clamp(0.0, 1.0).to_string(),
        ),
        LinkFunction::CLogLog => {
            let lo_p = 1.0 - (-lo_eta.exp()).exp();
            let hi_p = 1.0 - (-hi_eta.exp()).exp();
            (
                se.to_string(),
                lo_p.clamp(0.0, 1.0).to_string(),
                hi_p.clamp(0.0, 1.0).to_string(),
            )
        }
    }
}

#[cfg(feature = "calibrate")]
fn save_predictions_detailed(
    sample_ids: &[String],
    signed_distance: &Array1<f64>,
    eta: &Array1<f64>,
    mean: &Array1<f64>,
    se_eta_opt: Option<&Array1<f64>>,
    link: LinkFunction,
    output_path: &str,
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;
    let is_binary = !matches!(link, LinkFunction::Identity);

    if is_binary {
        writeln!(
            file,
            "sample_id\thull_signed_distance\tlog_odds\tstandard_error_log_odds\tprediction\tprobability_lower_95\tprobability_upper_95"
        )?;
    } else {
        writeln!(
            file,
            "sample_id\thull_signed_distance\tprediction\tstandard_error_mean\tmean_lower_95\tmean_upper_95"
        )?;
    }

    for index in 0..eta.len() {
        let prediction = mean[index];
        let (se_str, lo_str, hi_str) = se_and_ci(
            se_eta_opt,
            index,
            if is_binary { eta[index] } else { prediction },
            link,
        );

        if is_binary {
            write_tsv_row(
                &mut file,
                &[
                    &sample_ids[index] as &dyn std::fmt::Display,
                    &signed_distance[index],
                    &eta[index],
                    &se_str,
                ],
                None,
                &[&prediction, &lo_str, &hi_str],
            )?;
        } else {
            write_tsv_row(
                &mut file,
                &[
                    &sample_ids[index] as &dyn std::fmt::Display,
                    &signed_distance[index],
                ],
                None,
                &[&prediction, &se_str, &lo_str, &hi_str],
            )?;
        }
    }

    Ok(())
}

#[cfg(feature = "calibrate")]
fn save_survival_predictions(
    output_path: &str,
    data: &SurvivalPredictionData,
    prediction: &SurvivalPrediction,
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;
    writeln!(
        file,
        "sample_id\tage_entry\tage_exit\tcumulative_hazard_entry\tcumulative_hazard_exit\tcumulative_incidence_entry\tcumulative_incidence_exit\tconditional_risk\tlogit_risk\tlogit_risk_standard_error"
    )?;

    for index in 0..prediction.conditional_risk.len() {
        let sample_id = (index + 1).to_string();
        let se = prediction
            .logit_risk_se
            .as_ref()
            .map(|values| values[index].to_string())
            .unwrap_or_else(|| "NA".to_string());

        write_tsv_row(
            &mut file,
            &[
                &sample_id as &dyn std::fmt::Display,
                &data.age_entry[index],
                &data.age_exit[index],
                &prediction.cumulative_hazard_entry[index],
                &prediction.cumulative_hazard_exit[index],
                &prediction.cumulative_incidence_entry[index],
                &prediction.cumulative_incidence_exit[index],
                &prediction.conditional_risk[index],
                &prediction.logit_risk[index],
                &se,
            ],
            None,
            &[],
        )?;
    }

    Ok(())
}

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
fn format_duration_ago(seconds: u64) -> String {
    const UNITS: &[(u64, &str)] = &[
        (365 * 24 * 3600, "years"),
        (30 * 24 * 3600, "months"),
        (7 * 24 * 3600, "weeks"),
        (24 * 3600, "days"),
        (3600, "hours"),
        (60, "minutes"),
    ];

    for &(threshold, label) in UNITS {
        if seconds >= threshold {
            return format!("{:.1} {label} ago", seconds as f64 / threshold as f64);
        }
    }
    format!("{seconds} seconds ago")
}

#[cfg(all(
    feature = "map",
    feature = "score",
    feature = "calibrate",
    feature = "terms"
))]
fn print_version_info() {
    let version = env!("CARGO_PKG_VERSION");
    let release_tag = option_env!("GNOMON_RELEASE_TAG");
    let build_timestamp: u64 = env!("GNOMON_BUILD_TIMESTAMP").parse().unwrap_or(0);

    println!("gnomon {version}");

    match release_tag {
        Some(tag) => println!("Release: {tag}"),
        None => println!("Release: development build"),
    }

    if build_timestamp > 0 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs())
            .unwrap_or(0);

        if now > build_timestamp {
            println!("Built: {}", format_duration_ago(now - build_timestamp));
        } else {
            println!("Built: just now");
        }
    }
}
