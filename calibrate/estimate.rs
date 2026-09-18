//! Thin training adapter over gam.
//!
//! `train_model` fits the Bernoulli marginal-slope model (probit base link,
//! score warp and link deviation on) for a binary phenotype through gam's
//! formula route, and the Gaussian location-scale model with a link wiggle for a
//! continuous one through a direct request, because the formula route refuses a
//! link wiggle on a non-binomial family. `train_survival_model` fits the
//! survival marginal-slope model through the formula route. In both
//! marginal-slope fits the score enters as given: the marginal index is anchored
//! on the law declared by `ModelConfig::latent_law`, by default the weighted
//! empirical law of the training rows' scores, and no transform of the score is
//! fitted. gam assembles every saved payload, so prediction replays what the fit
//! consumed.

use crate::calibrate::construction::{
    AGE_ENTRY_COLUMN, AGE_EXIT_COLUMN, EVENT_COLUMN, PHENOTYPE_COLUMN, SCORE_COLUMN,
    WEIGHT_COLUMN, context_formula, duchon_smooth, marginal_termspec, slope_formula,
};
use crate::calibrate::data::TrainingData;
use crate::calibrate::model::{
    LatentLaw, MODEL_BUNDLE_VERSION, ModelConfig, ModelFamily, SmoothConfig, TrainedModel,
    predictor_headers,
};
use crate::calibrate::runtime::on_gam_pool;
use crate::calibrate::survival_data::SurvivalTrainingBundle;

use gam::FailureCategory;
use gam::data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::gamlss::GaussianLocationScaleTermSpec;
use gam::inference::model::FittedModelPayload;
use gam::inference::model_payload_builders::{
    LocationScaleInputs, LocationScaleResponse, LocationScaleWiggle, SavedModelSourceMetadata,
    assemble_location_scale_payload, fit_formula_to_payload,
};
use gam::model_types::{BlockRole, EstimationError as GamEstimationError};
use gam::solver::fit_orchestration::{
    FitConfig, FitFailure, FitRequest, FitResult, GaussianLocationScaleFitRequest, LinkWiggleConfig,
    WorkflowError, fit_model,
};
use gam::terms::smooth::{SpatialLengthScaleOptimizationOptions, freeze_term_collection_from_design};
use gam::types::{LinkFunction, WigglePenaltyConfig};

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

/// Errors surfaced by the training adapter.
///
/// A failed gam fit keeps gam's typed error whole, under the variant of its
/// fixed category (gam#2937), so a caller branches on what stopped the fit
/// without reading the text.
#[derive(Debug)]
pub enum EstimationError {
    /// The outer search declined a certified optimum that an evaluated state
    /// beats and certified nothing in its place (gam#2953). The error names the
    /// checkpoint to resume from.
    DominatedCertifiedPlateau(WorkflowError),
    /// An outer smoothing search or an inner solve ended without its
    /// convergence certificate.
    Convergence(WorkflowError),
    /// Outer startup validation refused every candidate seed.
    StartupSeeds(WorkflowError),
    /// gam's own consistency contract failed: an engine defect.
    Invariant(WorkflowError),
    /// gam refused the configuration, the data or the problem's size.
    Input(WorkflowError),
    /// A factorization, eigendecomposition, root solve or row quantity failed.
    Numerical(WorkflowError),
    /// A quadrature did not reach its tolerance.
    Integration(WorkflowError),
    /// The failure reached gam's boundary as prose, so gam knows no category.
    Unclassified(WorkflowError),
    /// A gam call outside a fit whose interface reports only text.
    Gam(String),
    Domain(String),
}

impl EstimationError {
    /// The typed gam fit failure, when a fit is what failed.
    pub fn fit_failure(&self) -> Option<&WorkflowError> {
        match self {
            Self::DominatedCertifiedPlateau(error)
            | Self::Convergence(error)
            | Self::StartupSeeds(error)
            | Self::Invariant(error)
            | Self::Input(error)
            | Self::Numerical(error)
            | Self::Integration(error)
            | Self::Unclassified(error) => Some(error),
            Self::Gam(_) | Self::Domain(_) => None,
        }
    }
}

/// Whether gam's error ends in a dominated certified plateau, seen through the
/// layers that only carry it.
fn ends_in_dominated_plateau(error: &WorkflowError) -> bool {
    match error {
        WorkflowError::Fit(failure) => matches!(
            failure.estimation_error(),
            Some(GamEstimationError::DominatedCertifiedPlateau { .. })
        ),
        WorkflowError::SpatialUnderresolved {
            refit_failure: Some(refit_failure),
            ..
        } => ends_in_dominated_plateau(refit_failure),
        _ => false,
    }
}

impl From<WorkflowError> for EstimationError {
    fn from(error: WorkflowError) -> Self {
        if ends_in_dominated_plateau(&error) {
            return Self::DominatedCertifiedPlateau(error);
        }
        match error.failure_category() {
            FailureCategory::Convergence => Self::Convergence(error),
            FailureCategory::StartupSeeds => Self::StartupSeeds(error),
            FailureCategory::Invariant => Self::Invariant(error),
            FailureCategory::Input => Self::Input(error),
            FailureCategory::Numerical => Self::Numerical(error),
            FailureCategory::Integration => Self::Integration(error),
            FailureCategory::Unclassified => Self::Unclassified(error),
        }
    }
}

impl From<GamEstimationError> for EstimationError {
    fn from(error: GamEstimationError) -> Self {
        Self::from(WorkflowError::from(FitFailure::from(error)))
    }
}

impl std::fmt::Display for EstimationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DominatedCertifiedPlateau(error)
            | Self::Convergence(error)
            | Self::StartupSeeds(error)
            | Self::Invariant(error)
            | Self::Input(error)
            | Self::Numerical(error)
            | Self::Integration(error)
            | Self::Unclassified(error) => write!(
                f,
                "gam error [{}, {}]: {error}",
                error.failure_category(),
                error.variant_name()
            ),
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

fn validate_training_data(data: &TrainingData) -> Result<(), EstimationError> {
    let n = data.y.len();
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
    Ok(())
}

fn validate_smooth_configs(config: &ModelConfig) -> Result<(), EstimationError> {
    if config.pgs_basis_config.num_centers < 4
        || config.pc_configs.iter().any(|pc| pc.basis_config.num_centers < 4)
    {
        return Err(EstimationError::Domain("Duchon smooths require at least 4 centers".into()));
    }
    Ok(())
}

/// Predictor columns in `predictor_headers` order: score | sex | PC1..PCk.
fn predictor_columns<'a>(
    score: ArrayView1<'a, f64>,
    sex: ArrayView1<'a, f64>,
    pcs: ArrayView2<'a, f64>,
) -> Vec<(String, ArrayView1<'a, f64>)> {
    let values = [score, sex]
        .into_iter()
        .chain((0..pcs.ncols()).map(move |index| pcs.index_axis_move(Axis(1), index)));
    predictor_headers(pcs.ncols()).into_iter().zip(values).collect()
}

/// A training table of named continuous columns, in the given order.
fn encoded_dataset(columns: &[(String, ArrayView1<'_, f64>)]) -> EncodedDataset {
    let n = columns.first().map_or(0, |(_, column)| column.len());
    let mut values = Array2::<f64>::zeros((n, columns.len()));
    for (index, (_, column)) in columns.iter().enumerate() {
        values.column_mut(index).assign(column);
    }
    let headers: Vec<String> = columns.iter().map(|(name, _)| name.clone()).collect();
    EncodedDataset {
        schema: DataSchema {
            columns: headers
                .iter()
                .map(|name| SchemaColumn {
                    name: name.clone(),
                    kind: ColumnKindTag::Continuous,
                    levels: Vec::new(),
                })
                .collect(),
        },
        column_kinds: vec![ColumnKindTag::Continuous; headers.len()],
        headers,
        values,
    }
}

/// gam records every table column as a training feature. The saved schema
/// keeps only the leading predictor columns, so prediction needs no outcome,
/// weight or time columns beyond the ones the survival predictor names.
fn record_training_metadata(
    payload: &mut FittedModelPayload,
    dataset: &EncodedDataset,
    num_predictors: usize,
) {
    let ranges = dataset
        .values
        .slice(s![.., ..num_predictors])
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
        columns: dataset.schema.columns[..num_predictors].to_vec(),
    });
    payload.set_training_feature_metadata(dataset.headers[..num_predictors].to_vec(), ranges);
}

/// gam's spatial length-scale search, with its own iteration cap and relative
/// tolerance unless the configuration overrides them.
fn spatial_options(config: &ModelConfig) -> SpatialLengthScaleOptimizationOptions {
    let mut options = SpatialLengthScaleOptimizationOptions::default();
    if let Some(max_iterations) = config.reml_max_iterations {
        options.max_outer_iter = max_iterations;
    }
    if let Some(tolerance) = config.reml_convergence_tolerance {
        options.rel_tol = tolerance;
    }
    options
}

/// gam's blockwise fit options with the coefficient covariance on, and gam's own
/// outer iteration cap and tolerance unless the configuration overrides them.
fn blockwise_options(config: &ModelConfig) -> BlockwiseFitOptions {
    let mut options = BlockwiseFitOptions {
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    };
    if let Some(max_iterations) = config.reml_max_iterations {
        options.outer_max_iter = max_iterations;
    }
    if let Some(tolerance) = config.reml_convergence_tolerance {
        options.outer_tol = tolerance;
    }
    options
}

fn base_fit_config(config: &ModelConfig) -> FitConfig {
    FitConfig {
        weight_column: Some(WEIGHT_COLUMN.to_string()),
        spatial_optimization: spatial_options(config),
        ..FitConfig::default()
    }
}

fn fit_payload(
    formula: String,
    dataset: &EncodedDataset,
    fit_config: FitConfig,
    num_predictors: usize,
) -> Result<FittedModelPayload, EstimationError> {
    let fit_config = fit_config.resolve().map_err(EstimationError::Domain)?;
    let mut payload =
        fit_formula_to_payload(formula, dataset, &fit_config).map_err(EstimationError::from)?;
    record_training_metadata(&mut payload, dataset, num_predictors);
    Ok(payload)
}

pub fn train_model(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    on_gam_pool(|| train_model_on_pool(data, config)).map_err(EstimationError::Gam)?
}

fn train_model_on_pool(
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
    validate_training_data(data)?;
    if config.pc_configs.len() != data.pcs.ncols() {
        return Err(EstimationError::Domain(
            "PC configuration count must match the training matrix".into(),
        ));
    }
    let pc_bases: Vec<_> = config.pc_configs.iter().map(|pc| pc.basis_config).collect();
    if matches!(link, LinkFunction::Identity) {
        return train_gaussian_location_scale(data, config, &pc_bases);
    }
    let context = context_formula(&pc_bases);
    let mut fit_config = base_fit_config(config);
    let formula = match link {
        LinkFunction::Probit | LinkFunction::Logit => {
            fit_config.family = Some("bernoulli-marginal-slope".to_string());
            fit_config.z_column = Some(SCORE_COLUMN.to_string());
            fit_config.latent_measure = Some(config.latent_law.latent_measure().to_string());
            fit_config.slope_formula = Some(format!("{} + linkwiggle()", slope_formula(&pc_bases)));
            format!("{PHENOTYPE_COLUMN} ~ {context} + link(type=probit) + linkwiggle()")
        }
        other => {
            return Err(EstimationError::Domain(format!(
                "{other:?} link not yet wired in calibrate; supported: Identity (Gaussian location-scale GAMLSS fit), Probit/Logit (Bernoulli marginal-slope)"
            )));
        }
    };

    let mut columns = predictor_columns(data.p.view(), data.sex.view(), data.pcs.view());
    let num_predictors = columns.len();
    columns.push((PHENOTYPE_COLUMN.to_string(), data.y.view()));
    columns.push((WEIGHT_COLUMN.to_string(), data.weights.view()));
    let dataset = encoded_dataset(&columns);
    let saved = fit_payload(formula, &dataset, fit_config, num_predictors)?;
    Ok(TrainedModel {
        format_version: MODEL_BUNDLE_VERSION,
        config: config.clone(),
        saved,
    })
}

/// Identity link: the Gaussian location-scale fit. The mean and the log scale
/// share the score / sex / PC terms, and gam's cubic triple-penalty link wiggle
/// lets the conditional mean flex away from a strict additive form. gam's
/// formula route refuses a link wiggle on a non-binomial family, so the request
/// is built directly; the saved payload is still gam's own location-scale
/// assembly, and it records the response scale the fit standardized by.
fn train_gaussian_location_scale(
    data: &TrainingData,
    config: &ModelConfig,
    pc_bases: &[SmoothConfig],
) -> Result<TrainedModel, EstimationError> {
    let columns = predictor_columns(data.p.view(), data.sex.view(), data.pcs.view());
    let num_predictors = columns.len();
    let dataset = encoded_dataset(&columns);
    let terms = marginal_termspec(&config.pgs_basis_config, pc_bases);
    let n = data.y.len();
    let request = GaussianLocationScaleFitRequest {
        data: dataset.values.view(),
        spec: GaussianLocationScaleTermSpec {
            y: data.y.clone(),
            weights: data.weights.clone(),
            meanspec: terms.clone(),
            log_sigmaspec: terms,
            mean_offset: Array1::zeros(n),
            log_sigma_offset: Array1::zeros(n),
        },
        wiggle: Some(cubic_link_wiggle()),
        options: blockwise_options(config),
        kappa_options: spatial_options(config),
    };
    let result = match fit_model(FitRequest::GaussianLocationScale(request))
        .map_err(EstimationError::from)?
    {
        FitResult::GaussianLocationScale(result) => result,
        _ => {
            return Err(EstimationError::Gam(
                "a Gaussian location-scale request returned a different fit".into(),
            ));
        }
    };
    let wiggle = match (result.wiggle_knots, result.wiggle_degree, result.beta_link_wiggle) {
        (Some(knots), Some(degree), Some(beta_link_wiggle)) => LocationScaleWiggle {
            knots: knots.to_vec(),
            degree,
            beta_link_wiggle,
        },
        _ => {
            return Err(EstimationError::Gam(
                "the Gaussian location-scale fit returned no link wiggle".into(),
            ));
        }
    };
    let block = result.fit;
    let resolved_termspec =
        freeze_term_collection_from_design(&block.meanspec_resolved, &block.mean_design)?;
    let resolved_termspec_noise =
        freeze_term_collection_from_design(&block.noisespec_resolved, &block.noise_design)?;
    let beta_noise = block
        .fit
        .block_by_role(BlockRole::Scale)
        .map(|scale| scale.beta.to_vec());
    let rhs = format!(
        "{} + {}",
        duchon_smooth(SCORE_COLUMN, config.pgs_basis_config.num_centers),
        context_formula(pc_bases)
    );
    let mut saved = assemble_location_scale_payload(
        LocationScaleInputs {
            formula: format!("{PHENOTYPE_COLUMN} ~ {rhs} + linkwiggle()"),
            data_schema: dataset.schema.clone(),
            noise_formula: rhs,
            resolved_termspec,
            resolved_termspec_noise,
            fit_result: block.fit,
            beta_noise,
            wiggle: Some(wiggle),
        },
        LocationScaleResponse::Gaussian {
            response_scale: result.response_scale,
            base_link: None,
        },
        SavedModelSourceMetadata {
            training_headers: dataset.headers.clone(),
            training_feature_ranges: Some(dataset.feature_ranges()),
            offset_column: None,
            noise_offset_column: None,
        },
    )
    .map_err(EstimationError::Gam)?;
    record_training_metadata(&mut saved, &dataset, num_predictors);
    Ok(TrainedModel {
        format_version: MODEL_BUNDLE_VERSION,
        config: config.clone(),
        saved,
    })
}

/// gam's cubic triple-penalty link wiggle.
fn cubic_link_wiggle() -> LinkWiggleConfig {
    let penalty = WigglePenaltyConfig::cubic_triple_operator_default();
    LinkWiggleConfig {
        degree: penalty.degree,
        num_internal_knots: penalty.num_internal_knots,
        penalty_orders: penalty.penalty_orders,
        double_penalty: penalty.double_penalty,
    }
}

pub fn train_survival_model(
    bundle: &SurvivalTrainingBundle,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    on_gam_pool(|| train_survival_model_on_pool(bundle, config)).map_err(EstimationError::Gam)?
}

fn train_survival_model_on_pool(
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
    let data = &bundle.data;
    crate::calibrate::survival::validate_survival_inputs(
        data.age_entry.view(),
        data.age_exit.view(),
        data.event_target.view(),
        data.event_competing.view(),
        data.sample_weight.view(),
        data.pgs.view(),
        data.sex.view(),
        data.pcs.view(),
        data.extra_static_covariates.view(),
    )
    .map_err(|error| EstimationError::Domain(error.to_string()))?;
    if config.pc_configs.len() != data.pcs.ncols() {
        return Err(EstimationError::Domain(
            "PC configuration count must match the training matrix".into(),
        ));
    }
    if !data.sample_weight.iter().any(|weight| *weight > 0.0) {
        return Err(EstimationError::Domain(
            "survival training requires a positive sample weight".into(),
        ));
    }
    if data.extra_static_covariates.ncols() != 0 || !data.extra_static_names.is_empty() {
        return Err(EstimationError::Domain(
            "survival calibration accepts score, sex, and configured PCs; extra static covariates have no prediction schema".into(),
        ));
    }
    if survival_cfg.time_wiggle.is_some() && config.latent_law == LatentLaw::Empirical {
        return Err(EstimationError::Domain(
            "a baseline time wiggle cannot be combined with the empirical latent law: gam anchors a declared law only on a rigid time baseline; drop the time wiggle or declare a standard-normal score".into(),
        ));
    }

    let n = data.age_entry.len();
    // Start on the observed time scale, away from the derivative barrier: a
    // unit-shape Weibull baseline at the mean exit age. The time anchor is
    // gam's to choose: marginal-slope centres the time basis at the median exit,
    // because an earliest-entry anchor on delayed-entry ages inflates the
    // unpenalized time column until every smoothing seed is refused (gam #751).
    // The fitted anchor is saved with the model and replayed at prediction.
    let baseline_scale: f64 = data.age_exit.iter().map(|time| time / n as f64).sum();
    if !baseline_scale.is_finite() || baseline_scale <= 0.0 {
        return Err(EstimationError::Domain(
            "survival exit ages must have a positive finite mean".into(),
        ));
    }
    let pc_bases: Vec<_> = config.pc_configs.iter().map(|pc| pc.basis_config).collect();
    let time_wiggle = survival_cfg
        .time_wiggle
        .as_ref()
        .map_or_else(String::new, |settings| {
            format!(
                " + timewiggle(internal_knots={}, degree={}, penalty_order={}, double_penalty={})",
                settings.basis.num_knots,
                settings.basis.degree,
                settings.penalty_order,
                settings.double_penalty
            )
        });
    let formula = format!(
        "Surv({AGE_ENTRY_COLUMN}, {AGE_EXIT_COLUMN}, {EVENT_COLUMN}) ~ {}{time_wiggle}",
        context_formula(&pc_bases)
    );
    let fit_config = FitConfig {
        survival_likelihood: Some("marginal-slope".to_string()),
        slope_formula: Some(slope_formula(&pc_bases)),
        z_column: Some(SCORE_COLUMN.to_string()),
        latent_measure: Some(config.latent_law.latent_measure().to_string()),
        time_basis: "ispline".to_string(),
        time_degree: survival_cfg.baseline_basis.degree,
        time_num_internal_knots: survival_cfg.baseline_basis.num_knots,
        baseline_target: "weibull".to_string(),
        baseline_scale: Some(baseline_scale),
        baseline_shape: Some(1.0),
        ..base_fit_config(config)
    };

    // Outcomes, times and weights are separate formula roles, never predictors.
    let event_target = data.event_target.mapv(f64::from);
    let mut columns = predictor_columns(data.pgs.view(), data.sex.view(), data.pcs.view());
    let num_predictors = columns.len();
    columns.push((AGE_ENTRY_COLUMN.to_string(), data.age_entry.view()));
    columns.push((AGE_EXIT_COLUMN.to_string(), data.age_exit.view()));
    columns.push((EVENT_COLUMN.to_string(), event_target.view()));
    columns.push((WEIGHT_COLUMN.to_string(), data.sample_weight.view()));
    let dataset = encoded_dataset(&columns);
    let saved = fit_payload(formula, &dataset, fit_config, num_predictors)?;
    Ok(TrainedModel {
        format_version: MODEL_BUNDLE_VERSION,
        config: config.clone(),
        saved,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::model::SmoothConfig;
    use gam::families::bms::LatentMeasureKind;
    use ndarray::Array1;

    struct EngineTestLogger;

    impl log::Log for EngineTestLogger {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.target().starts_with("gam") && metadata.level() <= log::Level::Warn
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

    /// gam#2945's refusal, as a dominated certified plateau whose terminal certificate cannot certify a stationary
    /// optimum because the criterion declares no outer curvature.
    fn refused_for_missing_outer_curvature(error: &EstimationError) -> bool {
        let EstimationError::DominatedCertifiedPlateau(WorkflowError::Fit(failure)) = error else {
            return false;
        };
        let Some(GamEstimationError::DominatedCertifiedPlateau { terminal_refusal, .. }) =
            failure.estimation_error()
        else {
            return false;
        };
        let refusal = terminal_refusal.to_string();
        refusal.contains("did not certify a stationary optimum")
            && refusal.contains("curvature_source=unavailable")
    }

    /// The saved latent law must be the law of the training scores as given.
    /// gam stores the equal-mass compression of the weighted scores, standardized
    /// to weighted mean 0 and sd 1; with unit weights and no more rows than its
    /// 65-node grid, every row is its own node, so the nodes are exactly the
    /// sorted standardized training scores, each with weight 1/n.
    fn assert_empirical_training_law(saved: &FittedModelPayload, scores: ArrayView1<'_, f64>) {
        let grid = match saved.latent_measure.as_ref().expect("saved latent measure") {
            LatentMeasureKind::GlobalEmpirical { grid } => grid,
            other => panic!("expected the declared empirical law, got {other:?}"),
        };
        let nodes: Vec<f64> = grid.nodes.iter().copied().collect();
        let weights: Vec<f64> = grid.weights.iter().copied().collect();
        let n = scores.len();
        assert!(n <= 65, "the exact-node contract needs at most 65 rows");
        let mean = scores.sum() / n as f64;
        let sd = (scores.iter().map(|score| (score - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let mut standardized: Vec<f64> = scores.iter().map(|score| (score - mean) / sd).collect();
        standardized.sort_by(f64::total_cmp);
        assert_eq!(nodes.len(), n);
        assert_eq!(weights.len(), n);
        for (node, expected) in nodes.iter().zip(standardized.iter()) {
            assert!((node - expected).abs() < 1e-12 * (1.0 + expected.abs()), "{node} vs {expected}");
        }
        for weight in &weights {
            assert!((weight - 1.0 / n as f64).abs() < 1e-15);
        }
    }

    #[test]
    fn binary_train_anchors_on_the_training_score_law_and_replays_single_rows() {
        init_engine_test_logging();
        let n = 64;
        // A right-skewed score, on which the standard-normal closed form is biased.
        let p = Array1::from_iter(
            (0..n).map(|index| ((index as f64 + 0.5) / n as f64 * 2.5).exp() - 4.0),
        );
        let sex = Array1::from_iter((0..n).map(|index| (index % 2) as f64));
        let signal = Array1::from_iter(
            (0..n).map(|index| 0.6 * p[index] + 1.3 * (index as f64 * 2.3).sin()),
        );
        let mut sorted = signal.to_vec();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[n / 2];
        let y = signal.mapv(|value| f64::from(u8::from(value > median)));
        let data = TrainingData {
            y,
            p,
            sex,
            pcs: Array2::zeros((n, 0)),
            weights: Array1::ones(n),
        };
        let config = ModelConfig {
            model_family: ModelFamily::Gam(LinkFunction::Logit),
            pgs_basis_config: SmoothConfig { num_centers: 4 },
            ..Default::default()
        };
        let model = train_model(&data, &config).expect("train Bernoulli marginal-slope model");
        assert_eq!(
            model.saved.training_headers.as_ref().expect("headers"),
            &["score", "sex"]
        );
        assert_empirical_training_law(&model.saved, data.p.view());
        let predicted = model
            .predict(data.p.view(), data.sex.view(), data.pcs.view())
            .expect("predict training rows");
        assert!(predicted.iter().all(|risk| risk.is_finite() && *risk > 0.0 && *risk < 1.0));
        let directory = tempfile::tempdir().expect("model directory");
        let path = directory.path().join("model.json");
        model.save(path.to_str().expect("path")).expect("save model");
        let loaded = TrainedModel::load(path.to_str().expect("path")).expect("load model");
        let restored = loaded
            .predict(data.p.view(), data.sex.view(), data.pcs.view())
            .expect("predict loaded model");
        for (before, after) in predicted.iter().zip(restored.iter()) {
            assert!((before - after).abs() < 1e-10);
        }
        let one = loaded
            .predict(
                data.p.slice(s![41..42]),
                data.sex.slice(s![41..42]),
                data.pcs.slice(s![41..42, ..]),
            )
            .expect("predict one row");
        assert!((one[0] - restored[41]).abs() < 1e-10);
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

    fn survival_bundle(n: usize) -> SurvivalTrainingBundle {
        use crate::calibrate::survival::SurvivalTrainingData;
        SurvivalTrainingBundle {
            data: SurvivalTrainingData {
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
            },
        }
    }

    fn survival_config(
        time_wiggle: Option<crate::calibrate::model::SurvivalTimeWiggleConfig>,
    ) -> ModelConfig {
        use crate::calibrate::model::{BasisConfig, SurvivalModelConfig};
        ModelConfig {
            model_family: ModelFamily::Survival,
            pgs_basis_config: SmoothConfig { num_centers: 4 },
            survival: Some(SurvivalModelConfig {
                baseline_basis: BasisConfig { num_knots: 4, degree: 3 },
                time_wiggle,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn survival_time_wiggle_with_the_empirical_law_is_refused_before_fitting() {
        use crate::calibrate::model::{BasisConfig, SurvivalTimeWiggleConfig};
        let wiggle = SurvivalTimeWiggleConfig {
            basis: BasisConfig { num_knots: 4, degree: 3 },
            penalty_order: 2,
            double_penalty: true,
        };
        let error = train_survival_model(&survival_bundle(16), &survival_config(Some(wiggle)))
            .err()
            .expect("time wiggle with the empirical law is refused");
        assert!(
            matches!(&error, EstimationError::Domain(message) if message.contains("time wiggle")),
            "{error}"
        );
    }

    #[test]
    fn every_gam_failure_category_and_a_dominated_plateau_keep_their_own_variant() {
        let raised = |category| {
            EstimationError::from(WorkflowError::from(FitFailure::Raised {
                category,
                reason: "refused".to_string(),
            }))
        };
        let cases: [(FailureCategory, fn(&EstimationError) -> bool); 7] = [
            (FailureCategory::Convergence, |e| matches!(e, EstimationError::Convergence(_))),
            (FailureCategory::StartupSeeds, |e| matches!(e, EstimationError::StartupSeeds(_))),
            (FailureCategory::Invariant, |e| matches!(e, EstimationError::Invariant(_))),
            (FailureCategory::Input, |e| matches!(e, EstimationError::Input(_))),
            (FailureCategory::Numerical, |e| matches!(e, EstimationError::Numerical(_))),
            (FailureCategory::Integration, |e| matches!(e, EstimationError::Integration(_))),
            (FailureCategory::Unclassified, |e| matches!(e, EstimationError::Unclassified(_))),
        ];
        for (category, is_its_variant) in cases {
            let error = raised(category);
            assert!(is_its_variant(&error), "{category} mapped to {error:?}");
            assert_eq!(
                error.fit_failure().map(WorkflowError::failure_category),
                Some(category)
            );
            assert!(error.to_string().starts_with(&format!("gam error [{category}, ")), "{error}");
        }

        // A plateau refusal is a convergence failure to gam; it keeps its own variant through the
        // context an orchestration layer puts in front of it.
        let plateau = GamEstimationError::DominatedCertifiedPlateau {
            context: "marginal-slope".to_string(),
            kind: gam_problem::DominanceRefusalKind::IncumbentUnescapableSaddle,
            plateau_rho: vec![0.5],
            plateau_value: 73.02427,
            incumbent_rho: vec![-1.25],
            incumbent_value: 72.50521,
            incumbent_projected_grad_norm: Some(2.632e-1),
            gap: 9.541e-3,
            band: 1.088e-6,
            continuation: "declined another certified optimum".to_string(),
            terminal_refusal: Box::new(GamEstimationError::RemlOptimizationFailed(
                "not stationary".to_string(),
            )),
        };
        let error = EstimationError::from(WorkflowError::from(
            FitFailure::from(plateau).context("marginal-slope fit failed"),
        ));
        assert!(matches!(error, EstimationError::DominatedCertifiedPlateau(_)), "{error:?}");
        assert!(
            error
                .to_string()
                .starts_with("gam error [convergence, EstimationError::DominatedCertifiedPlateau]: "),
            "{error}"
        );
    }

    #[test]
    fn survival_public_train_save_load_predict_preserves_time_and_latent_score() {
        init_engine_test_logging();
        use crate::calibrate::model::SurvivalRiskType;

        let n = 32;
        let bundle = survival_bundle(n);
        let config = survival_config(None);
        let mut invalid_bundle = SurvivalTrainingBundle { data: bundle.data.clone() };
        invalid_bundle.data.sex = Array1::zeros(n - 1);
        assert!(matches!(
            train_survival_model(&invalid_bundle, &config),
            Err(EstimationError::Domain(_))
        ));
        // gam refuses this fit by name: its outer search cannot certify a stationary optimum because the survival
        // marginal-slope criterion declares no exact outer ψψ/ρψ curvature yet (gam#2945). Since gam#2954 stage 1 that
        // refusal arrives as the terminal certificate of a dominated certified plateau: gam declines the certified
        // optimum a railed checkpoint beats, and the checkpoint cannot certify without that curvature. Any other error
        // fails the test; once gam certifies the fit, every check below runs.
        let model = match train_survival_model(&bundle, &config) {
            Ok(model) => model,
            Err(error) if refused_for_missing_outer_curvature(&error) => {
                eprintln!("survival fit refused by name (gam#2945): {error}");
                return;
            }
            Err(error) => panic!("train survival model: {error}"),
        };
        let data = &bundle.data;
        assert_eq!(model.saved.survival_entry.as_deref(), Some(AGE_ENTRY_COLUMN));
        assert_eq!(model.saved.survival_exit.as_deref(), Some(AGE_EXIT_COLUMN));
        assert_empirical_training_law(&model.saved, data.pgs.view());
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
        // A single-row batch must retain the fitted time anchor and latent law.
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
