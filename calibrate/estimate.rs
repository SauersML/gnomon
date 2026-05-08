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
use crate::calibrate::model::{ModelConfig, ModelFamily, TrainedModel};
use crate::calibrate::survival::{build_time_block_input, build_time_wiggle_block_input};
use crate::calibrate::survival_data::SurvivalTrainingBundle;

use gam::families::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeTermSpec, DeviationBlockConfig, LatentZPolicy,
};
use gam::families::custom_family::BlockwiseFitOptions;
use gam::families::family_meta::inverse_link_to_binomial_family;
use gam::families::lognormal_kernel::FrailtySpec;
use gam::families::survival_marginal_slope::SurvivalMarginalSlopeTermSpec;
use gam::families::bernoulli_marginal_slope::LatentMeasureKind;
use gam::families::transformation_normal::TransformationNormalConfig;
use gam::inference::model::{
    DataSchema, FittedFamily, FittedModelPayload, MODEL_PAYLOAD_VERSION, ModelKind,
    SavedLatentZNormalization,
};
use gam::resource::ResourcePolicy;
use gam::estimate::FitOptions;
use gam::terms::smooth::{
    SpatialLengthScaleOptimizationOptions, TermCollectionSpec, freeze_term_collection_from_design,
};
use gam::types::{InverseLink, LikelihoodFamily, LinkFunction};
use gam::{
    BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, StandardFitRequest,
    SurvivalMarginalSlopeFitRequest, TransformationNormalFitRequest, fit_model,
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

/// Fixed column layout: phenotype | pgs | sex | pc1..pck | weights.
struct DataColumns {
    pgs_col: usize,
    sex_col: usize,
    pc_cols: Vec<usize>,
    matrix: Array2<f64>,
}

fn build_training_matrix(data: &TrainingData) -> DataColumns {
    let n = data.y.len();
    let n_pcs = data.pcs.ncols();
    // 1 (phenotype) + 1 (pgs) + 1 (sex) + n_pcs + 1 (weights)
    let ncols = 4 + n_pcs;
    let mut matrix = Array2::<f64>::zeros((n, ncols));
    matrix.slice_mut(s![.., 0]).assign(&data.y);
    matrix.slice_mut(s![.., 1]).assign(&data.p);
    matrix.slice_mut(s![.., 2]).assign(&data.sex);
    if n_pcs > 0 {
        matrix.slice_mut(s![.., 3..3 + n_pcs]).assign(&data.pcs);
    }
    matrix.slice_mut(s![.., 3 + n_pcs]).assign(&data.weights);
    DataColumns {
        pgs_col: 1,
        sex_col: 2,
        pc_cols: (3..3 + n_pcs).collect(),
        matrix,
    }
}

fn empty_data_schema() -> DataSchema {
    DataSchema {
        columns: Vec::new(),
    }
}

fn default_blockwise_options() -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    }
}

/// FitOptions tuned for Gaussian/Identity GAM via `FitRequest::Standard`,
/// mirroring the canonical choices in `gam::main` for that family.
fn default_gaussian_fit_options() -> FitOptions {
    FitOptions {
        latent_cloglog: None,
        mixture_link: None,
        optimize_mixture: false,
        sas_link: None,
        optimize_sas: false,
        // Gaussian identity does not need posterior covariance for prediction
        // (closed-form linear predictor); match gam::main's heuristic.
        compute_inference: false,
        max_iter: 200,
        tol: 1e-7,
        nullspace_dims: Vec::new(),
        linear_constraints: None,
        firth_bias_reduction: false,
        adaptive_regularization: None,
        penalty_shrinkage_floor: Some(1e-6),
        rho_prior: Default::default(),
        kronecker_penalty_system: None,
        kronecker_factored: None,
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
    pc_bases: &[crate::calibrate::model::BasisConfig],
) -> Result<ndarray::Array1<f64>, EstimationError> {
    use gam::terms::smooth::{LinearCoefficientGeometry, LinearTermSpec};

    let mut smooth_terms = Vec::with_capacity(pc_cols.len());
    for (idx, (&col, basis)) in pc_cols.iter().zip(pc_bases.iter()).enumerate() {
        let name = format!("pc{}", idx + 1);
        smooth_terms.push(crate::calibrate::construction::duchon_smooth(
            &name,
            col,
            basis.num_knots,
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
        options: default_blockwise_options(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        warm_start: None,
    };

    let result = fit_model(FitRequest::TransformationNormal(request))
        .map_err(EstimationError::Gam)?;
    let fit = match result {
        FitResult::TransformationNormal(fit) => fit,
        _ => {
            return Err(EstimationError::Gam(
                "fit_model returned the wrong FitResult variant for TransformationNormal"
                    .to_string(),
            ));
        }
    };
    let z = fit
        .fit
        .block_states
        .first()
        .ok_or_else(|| {
            EstimationError::Gam(
                "transformation-normal prefit produced no fitted blocks".to_string(),
            )
        })?
        .eta
        .clone();
    Ok(z)
}

/// Build a `FittedModelPayload` for an Identity-link (Gaussian) Standard fit.
fn standard_gaussian_payload_from_fit(
    fit: gam::estimate::UnifiedFitResult,
    design: gam::terms::smooth::TermCollectionDesign,
    resolvedspec: TermCollectionSpec,
) -> Result<FittedModelPayload, EstimationError> {
    let frozen = freeze_term_collection_from_design(&resolvedspec, &design)
        .map_err(|e| EstimationError::Gam(e.to_string()))?;
    let likelihood = LikelihoodFamily::GaussianIdentity;
    let mut payload = FittedModelPayload::new(
        MODEL_PAYLOAD_VERSION,
        "calibrate::standard-gaussian".to_string(),
        ModelKind::Standard,
        FittedFamily::Standard {
            likelihood,
            link: Some(LinkFunction::Identity),
            latent_cloglog_state: None,
            mixture_state: None,
            sas_state: None,
        },
        likelihood.name().to_string(),
    );
    payload.unified = Some(fit.clone());
    payload.fit_result = Some(fit);
    payload.data_schema = Some(empty_data_schema());
    payload.resolved_termspec = Some(frozen);
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
    payload.data_schema = Some(empty_data_schema());
    payload.formula_logslope = Some("calibrate::logslope".to_string());
    payload.z_column = Some("phenotype".to_string());
    payload.latent_z_normalization = Some(SavedLatentZNormalization {
        mean: fit.z_normalization.mean,
        sd: fit.z_normalization.sd,
    });
    payload.latent_measure = Some(fit.latent_measure);
    payload.marginal_baseline = Some(fit.baseline_marginal);
    payload.logslope_baseline = Some(fit.baseline_logslope);
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
    payload.data_schema = Some(empty_data_schema());
    payload.formula_logslope = Some("calibrate::logslope".to_string());
    payload.z_column = Some("event_target".to_string());
    payload.latent_z_normalization = Some(SavedLatentZNormalization {
        mean: fit.z_normalization.mean,
        sd: fit.z_normalization.sd,
    });
    payload.latent_measure = Some(LatentMeasureKind::StandardNormal);
    payload.logslope_baseline = Some(fit.baseline_slope);
    payload.resolved_termspec = Some(frozen_marginal);
    payload.resolved_termspec_logslope = Some(frozen_logslope);
    Ok(payload)
}

pub fn train_model(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    let link = match &config.model_family {
        ModelFamily::Gam(link) => *link,
        ModelFamily::Survival(_) => {
            return Err(EstimationError::Domain(
                "train_model expects a GAM family; use train_survival_model".to_string(),
            ));
        }
    };
    let cols = build_training_matrix(data);
    let pc_bases: Vec<_> = config.pc_configs.iter().map(|pc| pc.basis_config).collect();

    if matches!(link, LinkFunction::Identity) {
        return train_gaussian_identity(data, config, &cols, &pc_bases);
    }
    if !matches!(link, LinkFunction::Probit | LinkFunction::Logit) {
        return Err(EstimationError::Domain(format!(
            "{link:?} link not yet wired in calibrate; supported: Identity (Gaussian Standard fit), Probit/Logit (Bernoulli marginal-slope)"
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
    let z = ctn_prefit_latent_z(
        cols.matrix.view(),
        &data.p,
        &weights,
        cols.sex_col,
        &cols.pc_cols,
        &pc_bases,
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
        options: default_blockwise_options(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        policy: ResourcePolicy::default_library(),
    };

    let result = fit_model(FitRequest::BernoulliMarginalSlope(request))
        .map_err(EstimationError::Gam)?;
    let fit = match result {
        FitResult::BernoulliMarginalSlope(fit) => fit,
        _ => {
            return Err(EstimationError::Gam(
                "fit_model returned the wrong FitResult variant for BernoulliMarginalSlope"
                    .to_string(),
            ));
        }
    };

    let saved = bernoulli_payload_from_fit(fit, base_link, frailty)?;
    Ok(TrainedModel {
        config: config.clone(),
        saved,
    })
}

/// Identity-link branch: a Gaussian additive model fitted via `FitRequest::Standard`.
fn train_gaussian_identity(
    data: &TrainingData,
    config: &ModelConfig,
    cols: &DataColumns,
    pc_bases: &[crate::calibrate::model::BasisConfig],
) -> Result<TrainedModel, EstimationError> {
    let spec = build_marginal_termspec(
        cols.pgs_col,
        cols.sex_col,
        &cols.pc_cols,
        &config.pgs_basis_config,
        pc_bases,
    );
    let n = data.y.len();
    let request = StandardFitRequest {
        data: cols.matrix.view(),
        y: data.y.clone(),
        weights: data.weights.clone(),
        offset: ndarray::Array1::<f64>::zeros(n),
        spec,
        family: LikelihoodFamily::GaussianIdentity,
        options: default_gaussian_fit_options(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        wiggle: None,
        wiggle_options: None,
    };
    let result = fit_model(FitRequest::Standard(request)).map_err(EstimationError::Gam)?;
    let fit = match result {
        FitResult::Standard(fit) => fit,
        _ => {
            return Err(EstimationError::Gam(
                "fit_model returned the wrong FitResult variant for Standard".to_string(),
            ));
        }
    };
    let saved = standard_gaussian_payload_from_fit(fit.fit, fit.design, fit.resolvedspec)?;
    Ok(TrainedModel {
        config: config.clone(),
        saved,
    })
}

pub fn train_survival_model(
    bundle: &SurvivalTrainingBundle,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    let survival_spec = match &config.model_family {
        ModelFamily::Survival(spec) => *spec,
        ModelFamily::Gam(_) => {
            return Err(EstimationError::Domain(
                "train_survival_model expects a Survival family".to_string(),
            ));
        }
    };
    let survival_cfg = config.survival.as_ref().ok_or_else(|| {
        EstimationError::Domain("ModelConfig.survival missing for survival training".to_string())
    })?;

    let n = bundle.data.age_entry.len();
    let n_pcs = bundle.data.pcs.ncols();
    // Layout: pgs | sex | pc1..pck | sample_weight (no phenotype column —
    // survival outcome is the (age_entry, age_exit, event_target) triple).
    let ncols = 3 + n_pcs;
    let mut matrix = Array2::<f64>::zeros((n, ncols));
    matrix.slice_mut(s![.., 0]).assign(&bundle.data.pgs);
    matrix.slice_mut(s![.., 1]).assign(&bundle.data.sex);
    if n_pcs > 0 {
        matrix
            .slice_mut(s![.., 2..2 + n_pcs])
            .assign(&bundle.data.pcs);
    }
    matrix
        .slice_mut(s![.., 2 + n_pcs])
        .assign(&bundle.data.sample_weight);
    let pgs_col = 0usize;
    let sex_col = 1usize;
    let pc_cols: Vec<usize> = (2..2 + n_pcs).collect();

    let pc_bases: Vec<_> = config.pc_configs.iter().map(|pc| pc.basis_config).collect();
    let marginalspec =
        build_marginal_termspec(pgs_col, sex_col, &pc_cols, &config.pgs_basis_config, &pc_bases);
    let logslopespec = build_logslope_termspec(pgs_col);

    let time_block = build_time_block_input(bundle, &survival_spec)
        .map_err(EstimationError::Gam)?;
    let timewiggle_enabled = survival_cfg.time_varying.is_some();
    let timewiggle_block =
        build_time_wiggle_block_input(timewiggle_enabled).map_err(EstimationError::Gam)?;

    let event_target_f64 = bundle.data.event_target.mapv(|v| v as f64);
    let weights = bundle.data.sample_weight.clone();
    // Survival event indicator is binary; derive a continuous latent normal
    // score from the PGS via a CTN prefit conditional on sex + PCs.
    let z = ctn_prefit_latent_z(
        matrix.view(),
        &bundle.data.pgs,
        &weights,
        sex_col,
        &pc_cols,
        &pc_bases,
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
        derivative_guard: survival_spec.derivative_guard,
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
        options: default_blockwise_options(),
        kappa_options: SpatialLengthScaleOptimizationOptions::default(),
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

    let saved = survival_payload_from_fit(fit, frailty)?;
    Ok(TrainedModel {
        config: config.clone(),
        saved,
    })
}
