//! Thin training adapter over `gam::fit_model`.
//!
//! `train_model` runs the Bernoulli marginal-slope workflow (probit base
//! link, score warp on, link wiggle off pending a CLI flag). `train_survival_model`
//! runs the survival marginal-slope workflow with the time-block builders
//! provided by `crate::calibrate::survival`. Both wrap the resulting
//! `FitResult` in a `FittedModelPayload` so `TrainedModel::saved` can be
//! serialized directly.

use crate::calibrate::construction::{build_logslope_termspec, build_marginal_termspec};
use crate::calibrate::data::TrainingData;
use crate::calibrate::model::{
    LATENT_SCORE_HEADER, MODEL_BUNDLE_VERSION, ModelConfig, ModelFamily, TrainedModel,
    predict_eta_mean, predictor_headers,
};
use crate::calibrate::survival::{build_time_block_input, build_time_wiggle_block_input};
use crate::calibrate::survival_data::SurvivalTrainingBundle;

use gam::families::bernoulli_marginal_slope::LatentMeasureKind;
use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, DeviationRuntime, LatentZPolicy,
};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::family_meta::inverse_link_to_binomial_family;
use gam::families::gamlss::{
    BlockwiseTermFitResult, GaussianLocationScaleFitResult, GaussianLocationScaleTermSpec,
};
use gam::families::lognormal_kernel::FrailtySpec;
use gam::families::survival_marginal_slope::SurvivalMarginalSlopeTermSpec;
use gam::families::transformation_normal::TransformationNormalConfig;
use gam::inference::model::{
    ColumnKindTag, DataSchema, FittedFamily, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
    SavedAnchoredDeviationRuntime, SavedLatentZNormalization, SchemaColumn,
};
use gam::resource::ResourcePolicy;
use gam::terms::smooth::{
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec, freeze_term_collection_from_design,
};
use gam::types::{InverseLink, LikelihoodFamily, LinkFunction, WigglePenaltyConfig};
use gam::{
    BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, GaussianLocationScaleFitRequest,
    LinkWiggleConfig, SurvivalMarginalSlopeFitRequest, TransformationNormalFitRequest, fit_model,
};

use ndarray::{Array2, s};

/// Errors surfaced by the training adapter.
#[derive(Debug)]
pub enum EstimationError {
    Gam(String),
    Domain(String),
}

impl std::fmt::Display for EstimationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gam(s) => write!(f, "gam error: {s}"),
            Self::Domain(s) => write!(f, "domain error: {s}"),
        }
    }
}

impl std::error::Error for EstimationError {}

impl From<String> for EstimationError {
    fn from(s: String) -> Self {
        Self::Gam(s)
    }
}

/// Predictor-only layout shared with prediction: score | sex | PC1..PCk.
struct DataColumns {
    pgs_col: usize,
    sex_col: usize,
    pc_cols: Vec<usize>,
    matrix: Array2<f64>,
}

fn build_training_matrix(data: &TrainingData) -> Result<DataColumns, EstimationError> {
    let n = data.y.len();
    let n_pcs = data.pcs.ncols();
    if n == 0
        || data.p.len() != n
        || data.sex.len() != n
        || data.pcs.nrows() != n
        || data.weights.len() != n
    {
        return Err(EstimationError::Domain(
            "training arrays must have matching, nonzero row counts".into(),
        ));
    }
    if data
        .y
        .iter()
        .chain(data.p.iter())
        .chain(data.sex.iter())
        .chain(data.pcs.iter())
        .chain(data.weights.iter())
        .any(|value| !value.is_finite())
    {
        return Err(EstimationError::Domain(
            "training arrays must contain finite values".into(),
        ));
    }
    if data.weights.iter().any(|&weight| weight < 0.0)
        || !data.weights.iter().any(|&weight| weight > 0.0)
    {
        return Err(EstimationError::Domain(
            "training weights must be nonnegative with positive total weight".into(),
        ));
    }
    let ncols = 2 + n_pcs;
    let mut matrix = Array2::<f64>::zeros((n, ncols));
    matrix.slice_mut(s![.., 0]).assign(&data.p);
    matrix.slice_mut(s![.., 1]).assign(&data.sex);
    if n_pcs > 0 {
        matrix.slice_mut(s![.., 2..2 + n_pcs]).assign(&data.pcs);
    }
    Ok(DataColumns {
        pgs_col: 0,
        sex_col: 1,
        pc_cols: (2..2 + n_pcs).collect(),
        matrix,
    })
}

fn record_training_metadata(
    payload: &mut FittedModelPayload,
    matrix: ndarray::ArrayView2<'_, f64>,
) {
    let headers = predictor_headers(matrix.ncols() - 2);
    let ranges = matrix
        .columns()
        .into_iter()
        .map(|column| {
            column
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(low, high), &value| {
                    (low.min(value), high.max(value))
                })
        })
        .collect();
    payload.data_schema = Some(DataSchema {
        columns: headers
            .iter()
            .map(|name| SchemaColumn {
                name: name.clone(),
                kind: ColumnKindTag::Continuous,
                levels: Vec::new(),
            })
            .collect(),
    });
    payload.set_training_feature_metadata(headers, ranges);
}

fn blockwise_options(config: &ModelConfig) -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: config.max_iterations,
        inner_tol: config.convergence_tolerance,
        outer_max_iter: config.reml_max_iterations,
        outer_tol: config.reml_convergence_tolerance,
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    }
}

fn spatial_options(config: &ModelConfig) -> SpatialLengthScaleOptimizationOptions {
    SpatialLengthScaleOptimizationOptions {
        max_outer_iter: config.reml_max_iterations,
        rel_tol: config.reml_convergence_tolerance,
        ..SpatialLengthScaleOptimizationOptions::default()
    }
}

fn validate_smooth_configs(config: &ModelConfig) -> Result<(), EstimationError> {
    if config.pgs_basis_config.num_centers < 4
        || config.pc_configs.iter().any(|pc| pc.basis_config.num_centers < 4)
    {
        return Err(EstimationError::Domain("Duchon smooths require at least 4 centers".into()));
    }
    Ok(())
}

fn default_link_wiggle_config() -> LinkWiggleConfig {
    let cfg = WigglePenaltyConfig::cubic_triple_operator_default();
    LinkWiggleConfig {
        degree: cfg.degree,
        num_internal_knots: cfg.num_internal_knots,
        penalty_orders: cfg.penalty_orders,
        double_penalty: cfg.double_penalty,
    }
}

fn saved_deviation(runtime: &DeviationRuntime) -> SavedAnchoredDeviationRuntime {
    SavedAnchoredDeviationRuntime {
        kernel: gam::families::cubic_cell_kernel::ANCHORED_DEVIATION_KERNEL.to_string(),
        breakpoints: runtime.breakpoints().to_vec(),
        basis_dim: runtime.basis_dim(),
        span_c0: runtime
            .span_c0()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c1: runtime
            .span_c1()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c2: runtime
            .span_c2()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
        span_c3: runtime
            .span_c3()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    }
}

/// Run a CTN prefit on the PGS column, conditional on sex + PCs, and return
/// the per-row latent normal scores (the calibrated η of the single fitted
/// block). This is the recommended way to derive a continuous z for the
/// marginal-slope calibration when the response itself is binary or
/// otherwise discrete.
fn ctn_prefit_latent_z(
    data: ndarray::ArrayView2<'_, f64>,
    pgs: &ndarray::Array1<f64>,
    weights: &ndarray::Array1<f64>,
    sex_col: usize,
    pc_cols: &[usize],
    pc_bases: &[crate::calibrate::model::SmoothConfig],
    config: &ModelConfig,
) -> Result<(ndarray::Array1<f64>, FittedModelPayload), EstimationError> {
    use gam::terms::smooth::{LinearCoefficientGeometry, LinearTermSpec};

    let mut smooth_terms = Vec::with_capacity(pc_cols.len());
    for (idx, (&col, basis)) in pc_cols.iter().zip(pc_bases.iter()).enumerate() {
        let name = format!("pc{}", idx + 1);
        smooth_terms.push(crate::calibrate::construction::duchon_smooth(
            &name,
            col,
            basis.num_centers,
        ));
    }
    let covariate_spec = TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: "sex".to_string(),
            feature_col: sex_col,
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms,
    };

    let n = pgs.len();
    let request = TransformationNormalFitRequest {
        data,
        response: pgs.clone(),
        weights: weights.clone(),
        offset: ndarray::Array1::<f64>::zeros(n),
        covariate_spec,
        config: TransformationNormalConfig::default(),
        options: blockwise_options(config),
        kappa_options: spatial_options(config),
        warm_start: None,
    };

    let result =
        fit_model(FitRequest::TransformationNormal(request)).map_err(EstimationError::Gam)?;
    let fit = match result {
        FitResult::TransformationNormal(fit) => fit,
        _ => {
            return Err(EstimationError::Gam(
                "fit_model returned the wrong FitResult variant for TransformationNormal"
                    .to_string(),
            ));
        }
    };
    let frozen =
        freeze_term_collection_from_design(&fit.covariate_spec_resolved, &fit.covariate_design)
            .map_err(|error| EstimationError::Gam(error.to_string()))?;
    let mut saved = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "score ~ sex + PCs".to_string(),
        ModelKind::TransformationNormal,
        FittedFamily::TransformationNormal {
            likelihood: LikelihoodFamily::GaussianIdentity,
        },
        "transformation-normal".to_string(),
    );
    saved.unified = Some(fit.fit.clone());
    saved.fit_result = Some(fit.fit);
    saved.resolved_termspec = Some(frozen);
    saved.transformation_response_knots = Some(fit.family.response_knots().to_vec());
    saved.transformation_response_transform = Some(
        fit.family
            .response_transform()
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect(),
    );
    saved.transformation_response_degree = Some(fit.family.response_degree());
    saved.transformation_response_median = Some(fit.family.response_median());
    saved.transformation_score_calibration = Some(fit.score_calibration);
    record_training_metadata(&mut saved, data);
    // The CTN block's eta is an internal design channel, not the PIT score.
    // Use the persisted prediction path here and for every future sample.
    let z = predict_eta_mean(
        &saved,
        pgs.view(),
        data.column(sex_col),
        data.slice(s![.., 2..]),
    )
    .map_err(|error| EstimationError::Gam(error.to_string()))?
    .eta;
    Ok((z, saved))
}

/// Build a `FittedModelPayload` for a Gaussian location-scale (GAMLSS) fit
/// whose `μ` and `log σ` channels share the marginal Duchon-smooth layout and
/// whose link wiggle (if present) carries the cubic triple-penalty block.
fn gaussian_location_scale_payload_from_fit(
    result: GaussianLocationScaleFitResult,
) -> Result<FittedModelPayload, EstimationError> {
    let GaussianLocationScaleFitResult {
        fit:
            BlockwiseTermFitResult {
                fit,
                meanspec_resolved,
                noisespec_resolved,
                mean_design,
                noise_design,
            },
        wiggle_knots,
        wiggle_degree,
        beta_link_wiggle,
    } = result;
    let frozen_mean = freeze_term_collection_from_design(&meanspec_resolved, &mean_design)
        .map_err(|e| EstimationError::Gam(e.to_string()))?;
    let frozen_noise = freeze_term_collection_from_design(&noisespec_resolved, &noise_design)
        .map_err(|e| EstimationError::Gam(e.to_string()))?;
    let likelihood = LikelihoodFamily::GaussianIdentity;
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "calibrate::gaussian-location-scale".to_string(),
        ModelKind::LocationScale,
        FittedFamily::LocationScale {
            likelihood,
            base_link: Some(InverseLink::Standard(LinkFunction::Identity)),
        },
        likelihood.name().to_string(),
    );
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit);
    payload.resolved_termspec = Some(frozen_mean);
    payload.resolved_termspec_noise = Some(frozen_noise);
    payload.formula_noise = Some("calibrate::log_sigma".to_string());
    if let Some(knots) = wiggle_knots {
        payload.linkwiggle_knots = Some(knots.to_vec());
    }
    if let Some(degree) = wiggle_degree {
        payload.linkwiggle_degree = Some(degree);
    }
    if let Some(beta) = beta_link_wiggle {
        payload.beta_link_wiggle = Some(beta);
    }
    Ok(payload)
}

/// Convert a Bernoulli marginal-slope FitResult into a serializable payload.
fn bernoulli_payload_from_fit(
    fit: gam::families::bernoulli_marginal_slope::BernoulliMarginalSlopeFitResult,
    base_link: InverseLink,
    frailty: FrailtySpec,
) -> Result<FittedModelPayload, EstimationError> {
    let frozen_marginal = gam::terms::smooth::freeze_term_collection_from_design(
        &fit.marginalspec_resolved,
        &fit.marginal_design,
    )
    .map_err(|e| EstimationError::Gam(e.to_string()))?;
    let frozen_logslope = gam::terms::smooth::freeze_term_collection_from_design(
        &fit.logslopespec_resolved,
        &fit.logslope_design,
    )
    .map_err(|e| EstimationError::Gam(e.to_string()))?;

    let likelihood = inverse_link_to_binomial_family(&base_link);
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "calibrate::bernoulli-marginal-slope".to_string(),
        ModelKind::MarginalSlope,
        FittedFamily::MarginalSlope {
            likelihood,
            base_link: Some(base_link.clone()),
            frailty,
        },
        likelihood.name().to_string(),
    );
    payload.unified = Some(fit.fit.clone());
    payload.fit_result = Some(fit.fit);
    payload.formula_logslope = Some("calibrate::logslope".to_string());
    payload.z_column = Some(LATENT_SCORE_HEADER.to_string());
    payload.latent_z_normalization = Some(SavedLatentZNormalization {
        mean: fit.z_normalization.mean,
        sd: fit.z_normalization.sd,
    });
    payload.latent_measure = Some(fit.latent_measure);
    payload.marginal_baseline = Some(fit.baseline_marginal);
    payload.logslope_baseline = Some(fit.baseline_logslope);
    payload.score_warp_runtime = fit.score_warp_runtime.as_ref().map(saved_deviation);
    payload.link_deviation_runtime = fit.link_dev_runtime.as_ref().map(saved_deviation);
    payload.resolved_termspec = Some(frozen_marginal);
    payload.resolved_termspec_logslope = Some(frozen_logslope);
    Ok(payload)
}

/// Convert a Survival marginal-slope FitResult into a serializable payload.
fn survival_payload_from_fit(
    fit: gam::families::survival_marginal_slope::SurvivalMarginalSlopeFitResult,
    frailty: FrailtySpec,
) -> Result<FittedModelPayload, EstimationError> {
    let frozen_marginal = gam::terms::smooth::freeze_term_collection_from_design(
        &fit.marginalspec_resolved,
        &fit.marginal_design,
    )
    .map_err(|e| EstimationError::Gam(e.to_string()))?;
    let frozen_logslope = gam::terms::smooth::freeze_term_collection_from_design(
        &fit.logslopespec_resolved,
        &fit.logslope_design,
    )
    .map_err(|e| EstimationError::Gam(e.to_string()))?;

    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "calibrate::survival-marginal-slope".to_string(),
        ModelKind::Survival,
        FittedFamily::Survival {
            likelihood: LikelihoodFamily::RoystonParmar,
            survival_likelihood: Some("marginal-slope".to_string()),
            survival_distribution: Some("probit".to_string()),
            frailty,
        },
        LikelihoodFamily::RoystonParmar.name().to_string(),
    );
    payload.unified = Some(fit.fit.clone());
    payload.fit_result = Some(fit.fit);
    payload.formula_logslope = Some("calibrate::logslope".to_string());
    payload.z_column = Some(LATENT_SCORE_HEADER.to_string());
    payload.latent_z_normalization = Some(SavedLatentZNormalization {
        mean: fit.z_normalization.mean,
        sd: fit.z_normalization.sd,
    });
    payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
    payload.logslope_baseline = Some(fit.baseline_slope);
    payload.score_warp_runtime = fit.score_warp_runtime.as_ref().map(saved_deviation);
    payload.link_deviation_runtime = fit.link_dev_runtime.as_ref().map(saved_deviation);
    payload.resolved_termspec = Some(frozen_marginal);
    payload.resolved_termspec_logslope = Some(frozen_logslope);
    Ok(payload)
}

pub fn train_model(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    validate_smooth_configs(config)?;
    let link = match &config.model_family {
        ModelFamily::Gam(link) => *link,
        ModelFamily::Survival => {
            return Err(EstimationError::Domain(
                "train_model expects a GAM family; use train_survival_model".to_string(),
            ));
        }
    };
    let cols = build_training_matrix(data)?;
    if config.pc_configs.len() != data.pcs.ncols() {
        return Err(EstimationError::Domain(
            "PC configuration count must match the training matrix".into(),
        ));
    }
    let pc_bases: Vec<_> = config.pc_configs.iter().map(|pc| pc.basis_config).collect();

    if matches!(link, LinkFunction::Identity) {
        return train_gaussian_location_scale(data, config, &cols, &pc_bases);
    }
    if !matches!(link, LinkFunction::Probit | LinkFunction::Logit) {
        return Err(EstimationError::Domain(format!(
            "{link:?} link not yet wired in calibrate; supported: Identity (Gaussian location-scale GAMLSS fit), Probit/Logit (Bernoulli marginal-slope)"
        )));
    }

    let marginalspec = build_marginal_termspec(
        cols.pgs_col,
        cols.sex_col,
        &cols.pc_cols,
        &config.pgs_basis_config,
        &pc_bases,
    );
    let logslopespec = build_logslope_termspec(cols.pgs_col);

    let n = data.y.len();
    let weights = data.weights.clone();
    let y = data.y.clone();
    // For binary phenotypes the response itself cannot drive the CTN warp
    // (CTN warps a continuous response to N(0,1)). Instead, derive a
    // covariate-adjusted latent normal score from the PGS via a CTN prefit
    // conditional on sex + PCs, then feed that z into the marginal-slope
    // calibration. This is the canonical "score warp" pre-step.
    let (z, latent_score_model) = ctn_prefit_latent_z(
        cols.matrix.view(),
        &data.p,
        &weights,
        cols.sex_col,
        &cols.pc_cols,
        &pc_bases,
        config,
    )?;

    let base_link = InverseLink::Standard(LinkFunction::Probit);
    let frailty = FrailtySpec::None;
    let link_dev = Some(DeviationBlockConfig::triple_penalty_default());
    let score_warp = Some(DeviationBlockConfig::triple_penalty_default());

    let request = BernoulliMarginalSlopeFitRequest {
        data: cols.matrix.view(),
        spec: BernoulliMarginalSlopeTermSpec {
            y,
            weights,
            z,
            base_link: base_link.clone(),
            marginalspec,
            logslopespec,
            marginal_offset: ndarray::Array1::<f64>::zeros(n),
            logslope_offset: ndarray::Array1::<f64>::zeros(n),
            frailty: frailty.clone(),
            score_warp,
            link_dev,
            latent_z_policy: LatentZPolicy::default(),
        },
        options: blockwise_options(config),
        kappa_options: spatial_options(config),
        policy: ResourcePolicy::default_library(),
    };

    let result =
        fit_model(FitRequest::BernoulliMarginalSlope(request)).map_err(EstimationError::Gam)?;
    let fit = match result {
        FitResult::BernoulliMarginalSlope(fit) => fit,
        _ => {
            return Err(EstimationError::Gam(
                "fit_model returned the wrong FitResult variant for BernoulliMarginalSlope"
                    .to_string(),
            ));
        }
    };

    let mut saved = bernoulli_payload_from_fit(fit, base_link, frailty)?;
    record_training_metadata(&mut saved, cols.matrix.view());
    Ok(TrainedModel {
        format_version: MODEL_BUNDLE_VERSION,
        config: config.clone(),
        saved,
        latent_score_model: Some(latent_score_model),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    struct EngineTestLogger;

    impl log::Log for EngineTestLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.target().starts_with("gam::") && metadata.level() <= log::Level::Warn
        }

        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                eprintln!("{} {}: {}", record.level(), record.target(), record.args());
            }
        }

        fn flush(&self) {}
    }

    fn init_engine_test_logging() {
        static LOGGER: EngineTestLogger = EngineTestLogger;
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            log::set_logger(&LOGGER).expect("initialize engine test logger");
            log::set_max_level(log::LevelFilter::Warn);
        });
    }

    #[test]
    fn binary_latent_score_prefit_replays_saved_pit_for_new_rows() {
        let n = 32;
        let p = Array1::from_iter((0..n).map(|index| (index as f64 - 16.0) / 8.0));
        let sex = Array1::from_iter((0..n).map(|index| (index % 2) as f64));
        let data = TrainingData {
            y: Array1::zeros(n),
            p,
            sex,
            pcs: Array2::zeros((n, 0)),
            weights: Array1::ones(n),
        };
        let columns = build_training_matrix(&data).expect("predictor matrix");
        let config = ModelConfig::default();
        let (z, saved) = ctn_prefit_latent_z(
            columns.matrix.view(),
            &data.p,
            &data.weights,
            columns.sex_col,
            &[],
            &[],
            &config,
        )
        .expect("fit CTN preprocessor");
        gam::inference::model::FittedModel::from_payload(saved.clone())
            .validate_for_persistence()
            .expect("complete CTN payload");
        assert!(z.iter().all(|value| value.is_finite()));
        assert!(
            z[0] < z[30],
            "PIT score must depend on the observed PGS within a sex stratum"
        );
        let encoded = serde_json::to_vec(&saved).expect("serialize CTN");
        let restored: FittedModelPayload =
            serde_json::from_slice(&encoded).expect("deserialize CTN");
        let predicted = predict_eta_mean(
            &restored,
            data.p.slice(s![10..11]),
            data.sex.slice(s![10..11]),
            data.pcs.slice(s![10..11, ..]),
        )
        .expect("replay CTN for a prediction row");
        assert!((predicted.eta[0] - z[10]).abs() < 1e-10);
    }

    #[test]
    fn gaussian_public_train_save_load_predict_preserves_schema_and_single_rows() {
        init_engine_test_logging();
        let n = 48;
        let p = Array1::from_iter((0..n).map(|index| (index as f64 - 24.0) / 12.0));
        let sex = Array1::from_iter((0..n).map(|index| (index % 2) as f64));
        let y = Array1::from_iter((0..n).map(|index| {
            2.0 + 0.7 * p[index] + 0.3 * sex[index] + 0.2 * (index as f64 * 1.7).sin()
        }));
        let data = TrainingData {
            y,
            p,
            sex,
            pcs: Array2::zeros((n, 0)),
            weights: Array1::ones(n),
        };
        let config = ModelConfig {
            model_family: ModelFamily::Gam(LinkFunction::Identity),
            pgs_basis_config: crate::calibrate::model::SmoothConfig { num_centers: 4 },
            ..Default::default()
        };
        let model = train_model(&data, &config).expect("train Gaussian model");
        assert_eq!(
            model.saved.training_headers.as_ref().expect("headers"),
            &["score", "sex"]
        );
        let predicted = model
            .predict(data.p.view(), data.sex.view(), data.pcs.view())
            .expect("predict training rows");
        assert!(predicted.iter().all(|value| value.is_finite()));
        let directory = tempfile::tempdir().expect("model directory");
        let path = directory.path().join("model.json");
        model
            .save(path.to_str().expect("path"))
            .expect("save model");
        assert_eq!(
            std::fs::read_dir(directory.path())
                .expect("artifacts")
                .count(),
            1
        );
        let loaded = TrainedModel::load(path.to_str().expect("path")).expect("load model");
        let restored = loaded
            .predict(data.p.view(), data.sex.view(), data.pcs.view())
            .expect("predict loaded model");
        for (before, after) in predicted.iter().zip(restored.iter()) {
            assert!((before - after).abs() < 1e-10);
        }
        let one = loaded
            .predict(
                data.p.slice(s![17..18]),
                data.sex.slice(s![17..18]),
                data.pcs.slice(s![17..18, ..]),
            )
            .expect("predict one row");
        assert!((one[0] - restored[17]).abs() < 1e-10);
        let mut invalid = loaded;
        invalid.saved.fit_result.as_mut().expect("fit coefficients").blocks[0].beta[0] = f64::NAN;
        assert!(invalid.save(path.to_str().expect("path")).is_err());
        // A rejected save must leave the previous complete artifact usable.
        TrainedModel::load(path.to_str().expect("path")).expect("preserved valid model");
    }

    #[test]
    fn public_train_rejects_mismatched_arrays_before_basis_construction() {
        let data = TrainingData {
            y: Array1::zeros(2),
            p: Array1::zeros(1),
            sex: Array1::zeros(2),
            pcs: Array2::zeros((2, 0)),
            weights: Array1::ones(2),
        };
        assert!(matches!(
            train_model(&data, &ModelConfig::default()),
            Err(EstimationError::Domain(_))
        ));
    }

    #[test]
    fn survival_public_train_save_load_predict_preserves_time_and_latent_score() {
        init_engine_test_logging();
        use crate::calibrate::model::{BasisConfig, SurvivalModelConfig, SurvivalRiskType};
        use crate::calibrate::survival::SurvivalTrainingData;

        let n = 32;
        let data = SurvivalTrainingData {
            age_entry: Array1::from_iter((0..n).map(|index| 20.0 + (index % 5) as f64)),
            age_exit: Array1::from_iter((0..n).map(|index| 40.0 + (index % 13) as f64)),
            event_target: Array1::from_iter((0..n).map(|index| u8::from(index % 3 != 0))),
            event_competing: Array1::zeros(n),
            sample_weight: Array1::ones(n),
            pgs: Array1::from_iter((0..n).map(|index| ((index * 7) % n) as f64 / 8.0 - 2.0)),
            sex: Array1::from_iter((0..n).map(|index| (index % 2) as f64)),
            pcs: Array2::zeros((n, 0)),
            extra_static_covariates: Array2::zeros((n, 0)),
            extra_static_names: Vec::new(),
        };
        let bundle = SurvivalTrainingBundle { data };
        let config = ModelConfig {
            model_family: ModelFamily::Survival,
            pgs_basis_config: crate::calibrate::model::SmoothConfig { num_centers: 4 },
            max_iterations: 40,
            reml_max_iterations: 4,
            convergence_tolerance: 1e-5,
            reml_convergence_tolerance: 1e-2,
            survival: Some(SurvivalModelConfig {
                baseline_basis: BasisConfig { num_knots: 4, degree: 3 },
                time_wiggle: None,
            }),
            ..Default::default()
        };
        let mut invalid_bundle = SurvivalTrainingBundle { data: bundle.data.clone() };
        invalid_bundle.data.sex = Array1::zeros(n - 1);
        assert!(matches!(
            train_survival_model(&invalid_bundle, &config),
            Err(EstimationError::Domain(_))
        ));
        let model = train_survival_model(&bundle, &config).expect("train survival model");
        let data = &bundle.data;
        // Reconstruct the time channel exclusively from inference metadata and
        // compare it with the actual in-memory training channel before serde
        // discards the solver's block states.
        use gam::families::survival_construction::{
            SurvivalBaselineConfig, SurvivalBaselineTarget,
            build_survival_marginal_slope_baseline_offsets,
            evaluate_survival_time_basis_row, resolved_survival_time_basis_config_from_build,
        };
        let saved = &model.saved;
        let time_basis = resolved_survival_time_basis_config_from_build(
            saved.survival_time_basis.as_deref().expect("time basis"),
            saved.survival_time_degree,
            saved.survival_time_knots.as_ref(),
            saved.survival_time_keep_cols.as_ref(),
            saved.survival_time_smooth_lambda,
        ).expect("resolve saved time basis");
        let anchor = evaluate_survival_time_basis_row(saved.survival_time_anchor.expect("anchor"), &time_basis).expect("anchor basis");
        let baseline = SurvivalBaselineConfig {
            target: SurvivalBaselineTarget::Weibull,
            scale: saved.survival_baseline_scale,
            shape: saved.survival_baseline_shape,
            rate: None,
            makeham: None,
        };
        let (_, offsets, _) = build_survival_marginal_slope_baseline_offsets(
            &data.age_entry, &data.age_exit, &baseline,
        ).expect("saved baseline offsets");
        let fit = saved.fit_result.as_ref().expect("survival fit");
        for index in 0..n {
            let row = evaluate_survival_time_basis_row(data.age_exit[index], &time_basis).expect("exit basis");
            let reconstructed = (row - &anchor).dot(&fit.blocks[0].beta) + offsets[index];
            let trained = fit.block_states[0].eta[index];
            assert!((reconstructed - trained).abs() < 1e-9 * (1.0 + trained.abs()));
        }
        let before = model.predict_survival(
            data.age_entry.view(), data.age_exit.view(), data.pgs.view(), data.sex.view(),
            data.pcs.view(), SurvivalRiskType::Net,
        ).expect("predict fitted survival model");
        assert!(before.conditional_risk.iter().all(|risk| risk.is_finite() && *risk >= 0.0 && *risk <= 1.0));
        let directory = tempfile::tempdir().expect("model directory");
        let path = directory.path().join("survival.json");
        model.save(path.to_str().expect("path")).expect("save survival model");
        let loaded = TrainedModel::load(path.to_str().expect("path")).expect("load survival model");
        let after = loaded.predict_survival(
            data.age_entry.view(), data.age_exit.view(), data.pgs.view(), data.sex.view(),
            data.pcs.view(), SurvivalRiskType::Net,
        ).expect("predict restored survival model");
        for (expected, actual) in before.cumulative_hazard_exit.iter().zip(after.cumulative_hazard_exit.iter()) {
            assert!((expected - actual).abs() < 1e-10 * (1.0 + expected.abs()));
        }
        // A single-row batch must retain the fitted time anchor and PIT scale.
        let one = loaded.predict_survival(
            data.age_entry.slice(s![17..18]), data.age_exit.slice(s![17..18]),
            data.pgs.slice(s![17..18]), data.sex.slice(s![17..18]), data.pcs.slice(s![17..18, ..]),
            SurvivalRiskType::Net,
        ).expect("predict one survival row");
        assert!((one.cumulative_hazard_entry[0] - after.cumulative_hazard_entry[17]).abs() < 1e-10);
        assert!((one.cumulative_hazard_exit[0] - after.cumulative_hazard_exit[17]).abs() < 1e-10);
        assert!((one.conditional_risk[0] - after.conditional_risk[17]).abs() < 1e-10);
    }
}

/// Identity-link branch: a GAMLSS Gaussian location-scale fit. The mean
/// channel `μ(x)` and log-scale channel `log σ(x)` both use the marginal
/// Duchon-smooth layout (PGS smooth + sex linear + per-PC smooths), with a
/// shared cubic triple-penalty link wiggle so the conditional Gaussian can
/// flex away from a strict additive form.
fn train_gaussian_location_scale(
    data: &TrainingData,
    config: &ModelConfig,
    cols: &DataColumns,
    pc_bases: &[crate::calibrate::model::SmoothConfig],
) -> Result<TrainedModel, EstimationError> {
    let meanspec = build_marginal_termspec(
        cols.pgs_col,
        cols.sex_col,
        &cols.pc_cols,
        &config.pgs_basis_config,
        pc_bases,
    );
    let log_sigmaspec = build_marginal_termspec(
        cols.pgs_col,
        cols.sex_col,
        &cols.pc_cols,
        &config.pgs_basis_config,
        pc_bases,
    );
    let n = data.y.len();
    let request = GaussianLocationScaleFitRequest {
        data: cols.matrix.view(),
        spec: GaussianLocationScaleTermSpec {
            y: data.y.clone(),
            weights: data.weights.clone(),
            meanspec,
            log_sigmaspec,
            mean_offset: ndarray::Array1::<f64>::zeros(n),
            log_sigma_offset: ndarray::Array1::<f64>::zeros(n),
        },
        wiggle: Some(default_link_wiggle_config()),
        options: blockwise_options(config),
        kappa_options: spatial_options(config),
    };
    let result =
        fit_model(FitRequest::GaussianLocationScale(request)).map_err(EstimationError::Gam)?;
    let fit = match result {
        FitResult::GaussianLocationScale(fit) => fit,
        _ => {
            return Err(EstimationError::Gam(
                "fit_model returned the wrong FitResult variant for GaussianLocationScale"
                    .to_string(),
            ));
        }
    };
    let mut saved = gaussian_location_scale_payload_from_fit(fit)?;
    record_training_metadata(&mut saved, cols.matrix.view());
    Ok(TrainedModel {
        format_version: MODEL_BUNDLE_VERSION,
        config: config.clone(),
        saved,
        latent_score_model: None,
    })
}

pub fn train_survival_model(
    bundle: &SurvivalTrainingBundle,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    validate_smooth_configs(config)?;
    match &config.model_family {
        ModelFamily::Survival => {},
        ModelFamily::Gam(_) => {
            return Err(EstimationError::Domain(
                "train_survival_model expects a Survival family".to_string(),
            ));
        }
    };
    let survival_cfg = config.survival.as_ref().ok_or_else(|| {
        EstimationError::Domain("ModelConfig.survival missing for survival training".to_string())
    })?;
    crate::calibrate::survival::validate_survival_inputs(
        bundle.data.age_entry.view(),
        bundle.data.age_exit.view(),
        bundle.data.event_target.view(),
        bundle.data.event_competing.view(),
        bundle.data.sample_weight.view(),
        bundle.data.pgs.view(),
        bundle.data.sex.view(),
        bundle.data.pcs.view(),
        bundle.data.extra_static_covariates.view(),
    )
    .map_err(|error| EstimationError::Domain(error.to_string()))?;
    if config.pc_configs.len() != bundle.data.pcs.ncols() {
        return Err(EstimationError::Domain(
            "PC configuration count must match the training matrix".into(),
        ));
    }
    if !bundle.data.sample_weight.iter().any(|weight| *weight > 0.0) {
        return Err(EstimationError::Domain(
            "survival training requires a positive sample weight".into(),
        ));
    }
    if bundle.data.extra_static_covariates.ncols() != 0 || !bundle.data.extra_static_names.is_empty() {
        return Err(EstimationError::Domain(
            "survival calibration accepts score, sex, and configured PCs; extra static covariates have no prediction schema".into(),
        ));
    }

    let n = bundle.data.age_entry.len();
    let n_pcs = bundle.data.pcs.ncols();
    // Outcomes and weights are separate fit arguments, never predictor columns.
    let ncols = 2 + n_pcs;
    let mut matrix = Array2::<f64>::zeros((n, ncols));
    matrix.slice_mut(s![.., 0]).assign(&bundle.data.pgs);
    matrix.slice_mut(s![.., 1]).assign(&bundle.data.sex);
    if n_pcs > 0 {
        matrix
            .slice_mut(s![.., 2..2 + n_pcs])
            .assign(&bundle.data.pcs);
    }
    let pgs_col = 0usize;
    let sex_col = 1usize;
    let pc_cols: Vec<usize> = (2..2 + n_pcs).collect();

    let pc_bases: Vec<_> = config.pc_configs.iter().map(|pc| pc.basis_config).collect();
    let marginalspec = build_marginal_termspec(
        pgs_col,
        sex_col,
        &pc_cols,
        &config.pgs_basis_config,
        &pc_bases,
    );
    let logslopespec = build_logslope_termspec(pgs_col);

    let (mut time_block, time_metadata) =
        build_time_block_input(bundle, &survival_cfg.baseline_basis).map_err(EstimationError::Gam)?;
    let base_time_cols = time_block.design_exit.ncols();
    let timewiggle_block = build_time_wiggle_block_input(&mut time_block, survival_cfg.time_wiggle.as_ref())
        .map_err(EstimationError::Gam)?;
    let timewiggle_metadata = timewiggle_block
        .as_ref()
        .map(|wiggle| (wiggle.knots.to_vec(), wiggle.degree, wiggle.ncols));

    let event_target_f64 = bundle.data.event_target.mapv(|v| v as f64);
    let weights = bundle.data.sample_weight.clone();
    // Survival event indicator is binary; derive a continuous latent normal
    // score from the PGS via a CTN prefit conditional on sex + PCs.
    let (z, latent_score_model) = ctn_prefit_latent_z(
        matrix.view(),
        &bundle.data.pgs,
        &weights,
        sex_col,
        &pc_cols,
        &pc_bases,
        config,
    )?;

    let base_link = InverseLink::Standard(LinkFunction::Probit);
    let frailty = FrailtySpec::None;

    let spec = SurvivalMarginalSlopeTermSpec {
        age_entry: bundle.data.age_entry.clone(),
        age_exit: bundle.data.age_exit.clone(),
        event_target: event_target_f64,
        weights,
        z,
        base_link: base_link.clone(),
        marginalspec,
        marginal_offset: ndarray::Array1::<f64>::zeros(n),
        frailty: frailty.clone(),
        derivative_guard: gam::families::survival_marginal_slope::DEFAULT_SURVIVAL_MARGINAL_SLOPE_DERIVATIVE_GUARD,
        time_block,
        timewiggle_block,
        logslopespec,
        logslope_offset: ndarray::Array1::<f64>::zeros(n),
        score_warp: Some(DeviationBlockConfig::triple_penalty_default()),
        link_dev: Some(DeviationBlockConfig::triple_penalty_default()),
        latent_z_policy: LatentZPolicy::default(),
    };

    let request = SurvivalMarginalSlopeFitRequest {
        data: matrix.view(),
        spec,
        options: blockwise_options(config),
        kappa_options: spatial_options(config),
    };

    let result =
        fit_model(FitRequest::SurvivalMarginalSlope(request)).map_err(EstimationError::Gam)?;
    let fit = match result {
        FitResult::SurvivalMarginalSlope(fit) => fit,
        _ => {
            return Err(EstimationError::Gam(
                "fit_model returned the wrong FitResult variant for SurvivalMarginalSlope"
                    .to_string(),
            ));
        }
    };

    let mut saved = survival_payload_from_fit(fit, frailty)?;
    saved.survival_entry = Some("age_entry".into());
    saved.survival_exit = Some("age_exit".into());
    saved.survival_event = Some("event_target".into());
    saved.survivalspec = Some("net".into());
    saved.survival_baseline_target = Some("weibull".into());
    saved.survival_baseline_scale = Some(time_metadata.baseline_scale);
    saved.survival_baseline_shape = Some(1.0);
    saved.survival_likelihood = Some("marginal-slope".into());
    saved.survival_time_basis = Some(time_metadata.basis);
    saved.survival_time_degree = time_metadata.degree;
    saved.survival_time_knots = time_metadata.knots;
    saved.survival_time_keep_cols = time_metadata.keep_cols;
    saved.survival_time_smooth_lambda = time_metadata.smooth_lambda;
    saved.survival_time_anchor = Some(time_metadata.anchor);
    if let Some((knots, degree, ncols)) = timewiggle_metadata {
        let fit = saved
            .fit_result
            .as_ref()
            .ok_or_else(|| EstimationError::Gam("survival fit is missing coefficients".into()))?;
        let time = fit
            .blocks
            .first()
            .ok_or_else(|| EstimationError::Gam("survival fit is missing time block".into()))?;
        if time.beta.len() != base_time_cols + ncols {
            return Err(EstimationError::Gam(
                "survival time-wiggle coefficient width mismatch".into(),
            ));
        }
        saved.beta_baseline_timewiggle = Some(time.beta.slice(s![base_time_cols..]).to_vec());
        saved.baseline_timewiggle_knots = Some(knots);
        saved.baseline_timewiggle_degree = Some(degree);
        let settings = survival_cfg.time_wiggle.as_ref().ok_or_else(|| {
            EstimationError::Gam("time-wiggle fit is missing its configuration".into())
        })?;
        saved.baseline_timewiggle_penalty_orders = Some(vec![settings.penalty_order]);
        saved.baseline_timewiggle_double_penalty = Some(settings.double_penalty);
    }
    record_training_metadata(&mut saved, matrix.view());
    Ok(TrainedModel {
        format_version: MODEL_BUNDLE_VERSION,
        config: config.clone(),
        saved,
        latent_score_model: Some(latent_score_model),
    })
}
