#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unused_imports)]
#![deny(clippy::no_effect_underscore_binding)]

use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use gam::probability::normal_cdf_approx;
use ndarray::{Array1, ArrayView1};
use std::collections::HashSet;
use std::path::PathBuf;
use std::process;

use gnomon::calibrate::data::{load_prediction_data, load_training_data};
use gnomon::calibrate::estimate::{train_model, train_survival_model};
use gnomon::calibrate::model::BasisConfig;
use gnomon::calibrate::model::SurvivalModelConfig;
use gnomon::calibrate::model::SurvivalPrediction;
use gnomon::calibrate::model::SurvivalRiskType;
use gnomon::calibrate::model::SurvivalTimeVaryingConfig;
use gnomon::calibrate::model::{LinkFunction, ModelConfig, ModelFamily, TrainedModel};
use gnomon::calibrate::survival::SurvivalSpec;
use gnomon::calibrate::survival_data::{
    SurvivalPredictionData, has_survival_columns, load_survival_prediction_data,
    load_survival_training_data,
};
use gnomon::map::main as map_cli;
use gnomon::map::{DEFAULT_LD_WINDOW, LdWindow};
use gnomon::terms::infer_sex_to_tsv;

#[derive(Clone, ValueEnum)]
pub enum ModelFamilyCli {
    Gam,
    Survival,
}

#[derive(Args)]
pub struct TrainArgs {
    #[arg(long, value_enum, default_value_t = ModelFamilyCli::Gam)]
    pub model_family: ModelFamilyCli,

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

    /// Maximum number of P-IRLS iterations for the inner loop (per REML step)
    #[arg(long, default_value = "50")]
    pub max_iterations: usize,

    /// Convergence tolerance for the P-IRLS inner loop deviance change
    #[arg(long, default_value = "1e-7")]
    pub convergence_tolerance: f64,

    /// Maximum number of iterations for the outer REML/BFGS optimization loop
    #[arg(long, default_value = "100")]
    pub reml_max_iterations: u64,

    /// Convergence tolerance for the gradient norm in the outer REML/BFGS loop
    #[arg(long, default_value = "1e-3")]
    pub reml_convergence_tolerance: f64,

    /// Disable the optional post-process calibration layer (enabled by default)
    #[arg(long)]
    pub no_calibration: bool,

    /// Guard delta for the survival age transform
    #[arg(long, default_value = "0.1")]
    pub survival_guard_delta: f64,

    /// Number of internal knots for the survival baseline spline
    #[arg(long, default_value = "6")]
    pub survival_baseline_knots: usize,

    /// Degree for the survival baseline spline
    #[arg(long, default_value = "3")]
    pub survival_baseline_degree: usize,

    /// Grid size for the survival monotonicity penalty
    #[arg(long, default_value = "64")]
    pub survival_monotonic_grid: usize,

    /// Derivative guard threshold used inside the survival likelihood
    #[arg(long, default_value = "1e-8")]
    pub survival_derivative_guard: f64,

    /// Use expected information instead of observed Hessian when fitting survival models
    #[arg(long)]
    pub survival_expected_information: bool,

    /// Enable the optional time-varying PGS × age interaction
    #[arg(long)]
    pub survival_enable_time_varying: bool,

    /// Number of internal knots for the time-varying PGS spline
    #[arg(long, default_value = "5")]
    pub survival_time_varying_pgs_knots: usize,

    /// Degree for the time-varying PGS spline
    #[arg(long, default_value = "3")]
    pub survival_time_varying_pgs_degree: usize,

    /// Difference-penalty order for the time-varying PGS spline
    #[arg(long, default_value = "2")]
    pub survival_time_varying_pgs_penalty_order: usize,
}

#[derive(Args)]
pub struct InferArgs {
    /// Path to test TSV file with score,PC1,PC2,... columns (no phenotype needed)
    pub test_data: String,

    /// Path to trained model file (.toml)
    #[arg(long)]
    pub model: String,

    /// Disable applying the post-process calibration layer (enabled by default)
    #[arg(long)]
    pub no_calibration: bool,
}

#[derive(Args)]
pub struct TermsArgs {
    /// Path to genotype dataset (PLINK .bed/.bim/.fam prefix or VCF/BCF file)
    #[arg(value_name = "GENOTYPE_PATH")]
    pub genotype_path: PathBuf,

    /// Run sex inference on the provided genotype dataset
    #[arg(long)]
    pub sex: bool,
}

pub fn train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    if args.no_calibration {
        println!("Post-process calibration disabled via --no-calibration flag.");
    } else {
        println!("Post-process calibration enabled (default).");
    }

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
                calibrator_enabled: !args.no_calibration,
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
        calibrator_enabled: !args.no_calibration,
        survival: Some(survival_config),
        ..Default::default()
    };

    println!("Training survival model...");
    let trained_model = train_survival_model(&bundle, &config)?;
    trained_model.save("model.toml")?;
    println!("Model saved to: model.toml");
    Ok(())
}

pub fn infer(args: InferArgs) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading model from: {}", args.model);

    // Load trained model
    let model = TrainedModel::load(&args.model)?;
    let num_pcs = model.config.pc_configs.len();
    println!("Model expects {num_pcs} PCs");

    match &model.config.model_family {
        ModelFamily::Gam(link_function) => {
            // Load test data
            println!("Loading test data from: {}", args.test_data);
            let data = load_prediction_data(&args.test_data, num_pcs)?;
            println!("Loaded {} samples for prediction", data.p.len());

            // Make detailed predictions
            println!("Generating predictions with diagnostics...");
            let (eta, mean, signed_dist, se_eta_opt) =
                model.predict_detailed(data.p.view(), data.sex.view(), data.pcs.view())?;

            // Check if calibrator is available
            let calibrated_mean_opt = if args.no_calibration {
                println!("Skipping calibration via --no-calibration flag.");
                None
            } else if model.calibrator.is_some() {
                println!("Calibrator detected. Generating calibrated predictions.");
                // Get calibrated predictions but don't error if calibrator is missing
                match model.predict_calibrated(data.p.view(), data.sex.view(), data.pcs.view()) {
                    Ok(calibrated) => Some(calibrated),
                    Err(_) => None,
                }
            } else {
                None
            };

            // Save predictions with required columns to hardcoded output path
            let output_path = "predictions.tsv";
            save_predictions_detailed(
                &data.sample_ids,
                &signed_dist,
                &eta,
                &mean,
                se_eta_opt.as_ref(),
                *link_function,
                calibrated_mean_opt.as_ref(),
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

            let calibrated_risk = if args.no_calibration {
                println!("Skipping calibration via --no-calibration flag.");
                None
            } else if model
                .survival
                .as_ref()
                .and_then(|artifacts| artifacts.calibrator.as_ref())
                .is_some()
            {
                println!("Calibrator detected. Generating calibrated survival predictions.");
                match model.predict_survival_calibrated(
                    data.age_entry.view(),
                    data.age_exit.view(),
                    data.pgs.view(),
                    data.sex.view(),
                    data.pcs.view(),
                    Some(&model.survival_companions),
                ) {
                    Ok(calibrated) => Some(calibrated),
                    Err(_) => None,
                }
            } else {
                None
            };

            let output_path = "predictions.tsv";
            save_survival_predictions(output_path, &data, &prediction, calibrated_risk.as_ref())?;
            println!("Predictions saved to: {output_path}");
        }
    }

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

/// Write a TSV row, optionally inserting an extra column before the prediction column.
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
    if let Some(uncal) = uncalibrated {
        write!(file, "\t{uncal}")?;
    }
    for field in tail_fields {
        write!(file, "\t{field}")?;
    }
    writeln!(file)?;
    Ok(())
}

fn se_and_ci(
    se_eta_opt: Option<&Array1<f64>>,
    i: usize,
    eta_i: f64,
    link: LinkFunction,
) -> (String, String, String) {
    let Some(se_eta) = se_eta_opt else {
        return ("NA".to_string(), "NA".to_string(), "NA".to_string());
    };
    let se = se_eta[i];
    let lo_eta = eta_i - 1.959964 * se;
    let hi_eta = eta_i + 1.959964 * se;
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

fn save_predictions_detailed(
    sample_ids: &[String],
    signed_distance: &Array1<f64>,
    eta: &Array1<f64>,
    mean: &Array1<f64>,
    se_eta_opt: Option<&Array1<f64>>,
    link: LinkFunction,
    calibrated_mean_opt: Option<&Array1<f64>>,
    output_path: &str,
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;
    let has_cal = calibrated_mean_opt.is_some();
    let is_binary = !matches!(link, LinkFunction::Identity);

    // Write header
    if is_binary {
        let base = "sample_id\thull_signed_distance\tlog_odds\tstandard_error_log_odds";
        let tail = "prediction\tprobability_lower_95\tprobability_upper_95";
        if has_cal {
            writeln!(file, "{base}\tuncalibrated_prediction\t{tail}")?;
        } else {
            writeln!(file, "{base}\t{tail}")?;
        }
    } else {
        let tail = "prediction\tstandard_error_mean\tmean_lower_95\tmean_upper_95";
        if has_cal {
            writeln!(file, "sample_id\thull_signed_distance\tuncalibrated_prediction\t{tail}")?;
        } else {
            writeln!(file, "sample_id\thull_signed_distance\t{tail}")?;
        }
    }

    // Write rows
    for i in 0..eta.len() {
        let uncalibrated = mean[i];
        let prediction = calibrated_mean_opt.map_or(uncalibrated, |cal| cal[i]);
        let (se_str, lo_str, hi_str) = se_and_ci(
            se_eta_opt,
            i,
            if is_binary { eta[i] } else { prediction },
            link,
        );

        if is_binary {
            write_tsv_row(
                &mut file,
                &[
                    &sample_ids[i] as &dyn std::fmt::Display,
                    &signed_distance[i],
                    &eta[i],
                    &se_str,
                ],
                has_cal.then_some(&uncalibrated as &dyn std::fmt::Display),
                &[&prediction, &lo_str, &hi_str],
            )?;
        } else {
            write_tsv_row(
                &mut file,
                &[&sample_ids[i] as &dyn std::fmt::Display, &signed_distance[i]],
                has_cal.then_some(&uncalibrated as &dyn std::fmt::Display),
                &[&prediction, &se_str, &lo_str, &hi_str],
            )?;
        }
    }
    Ok(())
}

fn save_survival_predictions(
    output_path: &str,
    data: &SurvivalPredictionData,
    prediction: &SurvivalPrediction,
    calibrated_risk: Option<&Array1<f64>>,
) -> Result<(), std::io::Error> {
    use std::io::Write;

    let mut file = std::fs::File::create(output_path)?;
    let base_header = "sample_id\tage_entry\tage_exit\tcumulative_hazard_entry\tcumulative_hazard_exit\tcumulative_incidence_entry\tcumulative_incidence_exit\tconditional_risk\tlogit_risk\tlogit_risk_standard_error";
    if calibrated_risk.is_some() {
        writeln!(file, "{base_header}\tcalibrated_risk")?;
    } else {
        writeln!(file, "{base_header}")?;
    }

    for idx in 0..prediction.conditional_risk.len() {
        let sample_id = (idx + 1).to_string();
        let se = prediction
            .logit_risk_se
            .as_ref()
            .map(|a| a[idx].to_string())
            .unwrap_or_else(|| "NA".to_string());

        write_tsv_row(
            &mut file,
            &[
                &sample_id as &dyn std::fmt::Display,
                &data.age_entry[idx],
                &data.age_exit[idx],
                &prediction.cumulative_hazard_entry[idx],
                &prediction.cumulative_hazard_exit[idx],
                &prediction.cumulative_incidence_entry[idx],
                &prediction.cumulative_incidence_exit[idx],
                &prediction.conditional_risk[idx],
                &prediction.logit_risk[idx],
                &se,
            ],
            None,
            calibrated_risk
                .map(|cal| -> Vec<&dyn std::fmt::Display> { vec![&cal[idx]] })
                .unwrap_or_default()
                .as_slice(),
        )?;
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
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Calculate polygenic scores from genotype data
    #[command(about = "Calculate raw polygenic scores")]
    Score {
        /// Path to score file or directory containing score files
        #[arg(value_name = "SCORE_PATH")]
        score: PathBuf,

        /// Path to genotype data (PLINK .bed/.bim/.fam prefix, VCF, BCF, or DTC text file)
        #[arg(value_name = "GENOTYPE_PATH")]
        input_path: PathBuf,

        /// Path to file containing list of individual IDs to include (optional)
        #[arg(long)]
        keep: Option<PathBuf>,

        /// Reference genome FASTA (optional; auto-downloaded if not provided)
        #[arg(long)]
        reference: Option<PathBuf>,

        /// Force genome build (37 or 38); auto-detected if not provided
        #[arg(long)]
        build: Option<String>,

        /// Reference panel VCF for strand harmonization (flips alleles to match panel)
        #[arg(long)]
        panel: Option<PathBuf>,
    },

    /// Fit an HWE PCA model from the provided genotype dataset
    #[command(about = "Fit an HWE PCA model")]
    Fit {
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
    },

    /// Project samples using an existing HWE PCA model
    #[command(about = "Project samples using an existing HWE PCA model")]
    Project {
        /// Path to PLINK .bed file or directory containing .bed files
        #[arg(value_name = "GENOTYPE_PATH")]
        genotype_path: PathBuf,

        /// Use a built-in pre-trained model (downloads from GitHub if needed)
        /// Available models: hwe_1kg_hgdp_gsa_v2, hwe_1kg_hgdp_gsa_v3, hwe_1kg_hgdp_gda_v1, hwe_1kg_hgdp_intersection
        #[arg(long, value_name = "MODEL_NAME")]
        model: Option<String>,
    },

    /// Infer sample-level terms from genotype data
    #[command(about = "Infer sample metadata terms (outputs: sex.tsv)")]
    Terms(TermsArgs),

    /// Train a GAM calibration model from training data
    #[command(about = "Train GAM calibration model (outputs: model.toml)")]
    Train(TrainArgs),

    /// Apply trained calibration model to new data
    #[command(about = "Apply calibration model to new data (outputs: predictions.tsv)")]
    Infer(InferArgs),

    /// Display version and build information
    #[command(about = "Display version and build information")]
    Version,
}

fn main() {
    let cli = Cli::parse();
    let Cli { command } = cli;

    let result = match command {
        Some(Commands::Score {
            score,
            input_path,
            keep,
            reference,
            build,
            panel,
        }) => run_score(input_path, score, keep, reference, build, panel),
        Some(Commands::Fit {
            genotype_path,
            list,
            components,
            ld,
            sites_window,
            bp_window,
        }) => run_map_fit(genotype_path, list, components, ld, sites_window, bp_window),
        Some(Commands::Project {
            genotype_path,
            model,
        }) => run_map_project(genotype_path, model),
        Some(Commands::Terms(args)) => run_terms(args),
        Some(Commands::Train(args)) => train(args),
        Some(Commands::Infer(args)) => infer(args),
        Some(Commands::Version) => {
            print_version_info();
            Ok(())
        }
        None => {
            Cli::command().print_help().expect("print help");
            println!();
            Ok(())
        }
    };

    if let Err(e) = result {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}

/// Format seconds into a human-readable duration like "2.4 hours ago"
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

fn print_version_info() {
    let version = env!("CARGO_PKG_VERSION");
    let release_tag = option_env!("GNOMON_RELEASE_TAG");
    let build_timestamp: u64 = env!("GNOMON_BUILD_TIMESTAMP").parse().unwrap_or(0);

    println!("gnomon {}", version);

    match release_tag {
        Some(tag) => println!("Release: {}", tag),
        None => println!("Release: development build"),
    }

    if build_timestamp > 0 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        if now > build_timestamp {
            let age = now - build_timestamp;
            println!("Built: {}", format_duration_ago(age));
        } else {
            println!("Built: just now");
        }
    }
}

fn run_map_fit(
    genotype_path: PathBuf,
    list: Option<PathBuf>,
    components: usize,
    ld: bool,
    sites_window: Option<usize>,
    bp_window: Option<u64>,
) -> Result<(), Box<dyn std::error::Error>> {
    let ld_window = if ld {
        if let Some(bp) = bp_window {
            Some(LdWindow::BasePairs(bp))
        } else {
            let window = sites_window.unwrap_or(DEFAULT_LD_WINDOW);
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
        genotype_path,
        variant_list: list,
        components,
        ld: ld_window,
    })
    .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
}

fn run_map_project(
    genotype_path: PathBuf,
    model: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    map_cli::run(map_cli::MapCommand::Project {
        genotype_path,
        model,
    })
    .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
}

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

fn run_score(
    input_path: PathBuf,
    score: PathBuf,
    keep: Option<PathBuf>,
    reference: Option<PathBuf>,
    build: Option<String>,
    panel: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Call the score calculation logic directly
    score_main::run_gnomon_with_args(input_path, score, keep, reference, build, panel)
        .map_err(|e| e as Box<dyn std::error::Error>)
}
