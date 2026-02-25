pub use gam::estimate::{
    EstimationError, ExternalOptimOptions, ExternalOptimResult, FitOptions, FitResult,
    evaluate_external_cost_and_ridge, evaluate_external_gradients, fit_gam,
    optimize_external_design,
};

pub use gam::estimate::internal;
use gam::basis::{BasisOptions, Dense, KnotSource, create_basis};
use gam::faer_ndarray::FaerCholesky;
use gam::hmc;
use gam::hull::build_peeled_hull;
use gam::pirls::{self, PirlsStatus, WorkingModelPirlsOptions};
use gam::types::Coefficients;
use faer::Side;
use ndarray::{Array1, Array2, s};
use std::collections::HashMap;

fn map_survival_error(err: crate::calibrate::survival::SurvivalError) -> EstimationError {
    EstimationError::InvalidSpecification(err.to_string())
}

fn build_combined_penalty_matrix(s_list: &[Array2<f64>], lambdas: &Array1<f64>, p: usize) -> Array2<f64> {
    let mut penalty = Array2::<f64>::zeros((p, p));
    for (idx, s) in s_list.iter().enumerate() {
        let lambda = lambdas.get(idx).copied().unwrap_or(0.0);
        if lambda == 0.0 {
            continue;
        }
        penalty = penalty + s.mapv(|v| v * lambda);
    }
    penalty
}

fn run_gam_nuts_if_enabled(
    enabled: bool,
    family: gam::types::LikelihoodFamily,
    link: gam::types::LinkFunction,
    firth_bias_reduction: bool,
    x: ndarray::ArrayView2<'_, f64>,
    y: ndarray::ArrayView1<'_, f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    penalty_matrix: ndarray::ArrayView2<'_, f64>,
    mode: ndarray::ArrayView1<'_, f64>,
    hessian: ndarray::ArrayView2<'_, f64>,
) -> Result<Option<Array2<f64>>, EstimationError> {
    if !enabled {
        return Ok(None);
    }
    if matches!(link, gam::types::LinkFunction::Probit) {
        return Ok(None);
    }
    let config = hmc::NutsConfig::for_dimension(mode.len());
    let inputs = hmc::FamilyNutsInputs::Glm(hmc::GlmFlatInputs {
        x,
        y,
        weights,
        penalty_matrix,
        mode,
        hessian,
        firth_bias_reduction,
    });
    let nuts = hmc::run_nuts_sampling_flattened_family(
        family,
        inputs,
        &config,
    )
    .map_err(|err| EstimationError::InvalidSpecification(format!("NUTS sampling failed: {err}")))?;
    Ok(Some(nuts.samples))
}

fn to_engine_survival_penalties(
    penalties: &crate::calibrate::survival::PenaltyBlocks,
) -> gam::survival::PenaltyBlocks {
    let blocks = penalties
        .blocks
        .iter()
        .map(|block| gam::survival::PenaltyBlock {
            matrix: block.matrix.clone(),
            lambda: block.lambda,
            range: block.range.clone(),
        })
        .collect::<Vec<_>>();
    gam::survival::PenaltyBlocks::new(blocks)
}

fn to_engine_survival_monotonicity(
    monotonicity: &crate::calibrate::survival::MonotonicityPenalty,
) -> gam::survival::MonotonicityPenalty {
    let lambda = if monotonicity.derivative_design.nrows() == 0 {
        0.0
    } else {
        1.0
    };
    gam::survival::MonotonicityPenalty {
        lambda,
        tolerance: 0.0,
    }
}

fn to_engine_survival_spec(
    data: &crate::calibrate::survival::SurvivalTrainingData,
) -> gam::survival::SurvivalSpec {
    if data.event_competing.iter().any(|&v| v > 0) {
        gam::survival::SurvivalSpec::Crude
    } else {
        gam::survival::SurvivalSpec::Net
    }
}

fn build_expected_hessian_factor(
    hessian: &Array2<f64>,
) -> Option<crate::calibrate::survival::HessianFactor> {
    let chol = hessian.clone().cholesky(Side::Lower).ok()?;
    Some(crate::calibrate::survival::HessianFactor::Expected {
        factor: crate::calibrate::survival::CholeskyFactor {
            lower: chol.lower_triangular(),
        },
    })
}

fn cross_covariance_primary_companion(
    primary_samples: &Array2<f64>,
    companion_samples: &Array2<f64>,
) -> Option<Array2<f64>> {
    let n = primary_samples.nrows().min(companion_samples.nrows());
    if n < 2 {
        return None;
    }
    let p_primary = primary_samples.ncols();
    let p_companion = companion_samples.ncols();

    let primary = primary_samples.slice(s![0..n, ..]).to_owned();
    let companion = companion_samples.slice(s![0..n, ..]).to_owned();
    let mean_primary = primary.mean_axis(ndarray::Axis(0))?;
    let mean_companion = companion.mean_axis(ndarray::Axis(0))?;

    let mut cov = Array2::<f64>::zeros((p_primary, p_companion));
    for i in 0..n {
        let row_p = &primary.row(i).to_owned() - &mean_primary;
        let row_c = &companion.row(i).to_owned() - &mean_companion;
        for r in 0..p_primary {
            let pr = row_p[r];
            for c in 0..p_companion {
                cov[[r, c]] += pr * row_c[c];
            }
        }
    }
    cov.mapv_inplace(|v| v / ((n - 1) as f64));
    Some(cov)
}

fn fit_survival_logit_calibrator(
    bundle: &crate::calibrate::survival_data::SurvivalTrainingBundle,
    config: &crate::calibrate::model::ModelConfig,
    artifacts: &crate::calibrate::survival::SurvivalModelArtifacts,
) -> Result<Option<crate::calibrate::calibrator::CalibratorModel>, EstimationError> {
    use crate::calibrate::calibrator::{
        CalibratorSpec, active_penalty_nullspace_dims, build_calibrator_design, fit_calibrator,
    };
    use crate::calibrate::model::ModelFamily;

    if !config.calibrator_enabled {
        return Ok(None);
    }
    if !matches!(config.model_family, ModelFamily::Survival(_)) {
        return Ok(None);
    }

    let temp_model = crate::calibrate::model::TrainedModel {
        config: config.clone(),
        coefficients: crate::calibrate::model::MappedCoefficients::default(),
        lambdas: Vec::new(),
        hull: None,
        penalized_hessian: None,
        scale: None,
        calibrator: None,
        joint_link: None,
        survival: Some(artifacts.clone()),
        survival_companions: HashMap::new(),
        mcmc_samples: None,
        smoothing_correction: None,
    };
    let pred = temp_model
        .predict_survival(
            bundle.data.age_entry.view(),
            bundle.data.age_exit.view(),
            bundle.data.pgs.view(),
            bundle.data.sex.view(),
            bundle.data.pcs.view(),
            crate::calibrate::model::SurvivalRiskType::Net,
            None,
        )
        .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;

    let n = pred.logit_risk.len();
    let se = pred
        .logit_risk_se
        .clone()
        .unwrap_or_else(|| Array1::<f64>::zeros(n));
    let dist = Array1::<f64>::zeros(n);
    let fisher_weights = pred
        .conditional_risk
        .mapv(|p| (p * (1.0 - p)).max(1e-8));
    let alo_features = crate::calibrate::alo::CalibratorFeatures {
        pred: pred.logit_risk,
        se,
        dist,
        pred_identity: pred.conditional_risk,
        fisher_weights,
    };

    let cal_spec = CalibratorSpec {
        link: gam::types::LinkFunction::Logit,
        pred_basis: crate::calibrate::model::BasisConfig {
            degree: 3,
            num_knots: 5,
        },
        se_basis: crate::calibrate::model::BasisConfig {
            degree: 3,
            num_knots: 5,
        },
        dist_basis: crate::calibrate::model::BasisConfig {
            degree: 3,
            num_knots: 5,
        },
        penalty_order_pred: 2,
        penalty_order_se: 2,
        penalty_order_dist: 2,
        distance_enabled: true,
        distance_hinge: true,
        prior_weights: None,
        firth: CalibratorSpec::firth_default_for_link(gam::types::LinkFunction::Logit),
    };
    let (x_cal, penalties, schema, cal_offset) = build_calibrator_design(&alo_features, &cal_spec)?;
    let penalty_nullspace_dims = active_penalty_nullspace_dims(&schema, &penalties);
    let y = bundle.data.event_target.mapv(f64::from);
    let (cal_beta, cal_lambdas, _cal_scale, _edf, _optim) = fit_calibrator(
        y.view(),
        bundle.data.sample_weight.view(),
        x_cal.view(),
        cal_offset.view(),
        &penalties,
        &penalty_nullspace_dims,
        gam::types::LinkFunction::Logit,
        &cal_spec,
    )?;
    Ok(Some(crate::calibrate::calibrator::CalibratorModel {
        spec: cal_spec,
        knots_pred: schema.knots_pred,
        knots_se: schema.knots_se,
        knots_dist: schema.knots_dist,
        pred_constraint_transform: schema.pred_constraint_transform,
        stz_se: schema.stz_se,
        stz_dist: schema.stz_dist,
        penalty_nullspace_dims: schema.penalty_nullspace_dims,
        standardize_pred: schema.standardize_pred,
        standardize_se: schema.standardize_se,
        standardize_dist: schema.standardize_dist,
        interaction_center_pred: Some(schema.interaction_center_pred),
        se_log_space: schema.se_log_space,
        se_wiggle_only_drop: schema.se_wiggle_only_drop,
        dist_wiggle_only_drop: schema.dist_wiggle_only_drop,
        lambda_pred: cal_lambdas[0],
        lambda_pred_param: cal_lambdas[1],
        lambda_se: cal_lambdas[2],
        lambda_dist: cal_lambdas[3],
        coefficients: cal_beta,
        column_spans: schema.column_spans,
        pred_param_range: schema.pred_param_range,
        scale: None,
        assumes_frequency_weights: true,
    }))
}

pub fn train_model(
    data: &crate::calibrate::data::TrainingData,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
    use crate::calibrate::calibrator::{
        CalibratorModel, CalibratorSpec, active_penalty_nullspace_dims, build_calibrator_design,
        fit_calibrator,
    };
    use crate::calibrate::model::ModelFamily;

    let (
        x,
        s_list,
        layout,
        sum_to_zero_constraints,
        knot_vectors,
        range_transforms,
        pc_null_transforms,
        interaction_centering_means,
        interaction_orth_alpha,
        _penalty_structs,
    ) = crate::calibrate::construction::build_design_and_penalty_matrices(data, config)?;

    let link = match config.model_family {
        ModelFamily::Gam(link) => link,
        ModelFamily::Survival(_) => {
            return Err(EstimationError::InvalidInput(
                "train_model expects GAM family; use train_survival_model for survival".to_string(),
            ));
        }
    };
    let family = match link {
        gam::types::LinkFunction::Identity => {
            gam::types::LikelihoodFamily::GaussianIdentity
        }
        gam::types::LinkFunction::Logit => {
            gam::types::LikelihoodFamily::BinomialLogit
        }
        gam::types::LinkFunction::Probit => {
            gam::types::LikelihoodFamily::BinomialProbit
        }
    };

    let opts = FitOptions {
        max_iter: config.reml_max_iterations as usize,
        tol: config.reml_convergence_tolerance,
        nullspace_dims: vec![0; s_list.len()],
    };
    let offset = Array1::<f64>::zeros(data.y.len());
    let fit = fit_gam(
        x.view(),
        data.y.view(),
        data.weights.view(),
        offset.view(),
        &s_list,
        family,
        &opts,
    )?;

    let pirls_cfg = crate::calibrate::model::to_engine_model_config(config)?;

    // Build raw geometry matrix used by the predictor path: [PGS | PCs...]
    let n = data.p.len();
    let raw_dim = 1 + data.pcs.ncols();
    let mut raw_train = Array2::<f64>::zeros((n, raw_dim));
    raw_train.slice_mut(s![.., 0]).assign(&data.p);
    if data.pcs.ncols() > 0 {
        raw_train.slice_mut(s![.., 1..]).assign(&data.pcs);
    }
    let hull = build_peeled_hull(&raw_train, 3).ok();

    let calibrator = if config.calibrator_enabled {
        let alo_features = crate::calibrate::alo::compute_alo_features(
            &fit,
            data.y.view(),
            raw_train.view(),
            hull.as_ref(),
            link,
        )?;
        let cal_spec = CalibratorSpec {
            link,
            pred_basis: crate::calibrate::model::BasisConfig {
                degree: 3,
                num_knots: 5,
            },
            se_basis: crate::calibrate::model::BasisConfig {
                degree: 3,
                num_knots: 5,
            },
            dist_basis: crate::calibrate::model::BasisConfig {
                degree: 3,
                num_knots: 5,
            },
            penalty_order_pred: 2,
            penalty_order_se: 2,
            penalty_order_dist: 2,
            distance_enabled: true,
            distance_hinge: true,
            prior_weights: None,
            firth: CalibratorSpec::firth_default_for_link(link),
        };
        let (x_cal, penalties, schema, cal_offset) = build_calibrator_design(&alo_features, &cal_spec)?;
        let penalty_nullspace_dims = active_penalty_nullspace_dims(&schema, &penalties);
        let (cal_beta, cal_lambdas, cal_scale, _edf, _optim) = fit_calibrator(
            data.y.view(),
            data.weights.view(),
            x_cal.view(),
            cal_offset.view(),
            &penalties,
            &penalty_nullspace_dims,
            link,
            &cal_spec,
        )?;
        Some(CalibratorModel {
            spec: cal_spec,
            knots_pred: schema.knots_pred,
            knots_se: schema.knots_se,
            knots_dist: schema.knots_dist,
            pred_constraint_transform: schema.pred_constraint_transform,
            stz_se: schema.stz_se,
            stz_dist: schema.stz_dist,
            penalty_nullspace_dims: schema.penalty_nullspace_dims,
            standardize_pred: schema.standardize_pred,
            standardize_se: schema.standardize_se,
            standardize_dist: schema.standardize_dist,
            interaction_center_pred: Some(schema.interaction_center_pred),
            se_log_space: schema.se_log_space,
            se_wiggle_only_drop: schema.se_wiggle_only_drop,
            dist_wiggle_only_drop: schema.dist_wiggle_only_drop,
            lambda_pred: cal_lambdas[0],
            lambda_pred_param: cal_lambdas[1],
            lambda_se: cal_lambdas[2],
            lambda_dist: cal_lambdas[3],
            coefficients: cal_beta,
            column_spans: schema.column_spans,
            pred_param_range: schema.pred_param_range.clone(),
            scale: if matches!(link, gam::types::LinkFunction::Identity) {
                Some(cal_scale)
            } else {
                None
            },
            assumes_frequency_weights: true,
        })
    } else {
        None
    };

    let mut trained_config = config.clone();
    trained_config.sum_to_zero_constraints = sum_to_zero_constraints;
    trained_config.knot_vectors = knot_vectors;
    trained_config.range_transforms = range_transforms;
    trained_config.pc_null_transforms = pc_null_transforms;
    trained_config.interaction_centering_means = interaction_centering_means;
    trained_config.interaction_orth_alpha = interaction_orth_alpha;

    let coefficients = crate::calibrate::model::map_coefficients(&fit.beta, &layout)?;
    let penalty_matrix = build_combined_penalty_matrix(&s_list, &fit.lambdas, x.ncols());
    let mcmc_samples = run_gam_nuts_if_enabled(
        config.mcmc_enabled,
        family,
        link,
        pirls_cfg.firth_bias_reduction,
        x.view(),
        data.y.view(),
        data.weights.view(),
        penalty_matrix.view(),
        fit.beta.view(),
        fit.penalized_hessian.view(),
    )?;

    Ok(crate::calibrate::model::TrainedModel {
        config: trained_config,
        coefficients,
        lambdas: fit.lambdas.to_vec(),
        hull,
        penalized_hessian: Some(fit.penalized_hessian.clone()),
        scale: Some(fit.scale),
        calibrator,
        joint_link: None,
        survival: None,
        survival_companions: HashMap::new(),
        mcmc_samples,
        smoothing_correction: fit.smoothing_correction,
    })
}

pub fn train_joint_model(
    data: &crate::calibrate::data::TrainingData,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
    use crate::calibrate::model::ModelFamily;
    use gam::joint::{JointLinkGeometry, JointModelConfig};

    let (
        x,
        s_list,
        layout,
        sum_to_zero_constraints,
        knot_vectors,
        range_transforms,
        pc_null_transforms,
        interaction_centering_means,
        interaction_orth_alpha,
        _penalty_structs,
    ) = crate::calibrate::construction::build_design_and_penalty_matrices(data, config)?;

    let link = match config.model_family {
        ModelFamily::Gam(link) => link,
        ModelFamily::Survival(_) => {
            return Err(EstimationError::InvalidInput(
                "train_joint_model expects GAM family".to_string(),
            ));
        }
    };

    let geometry = JointLinkGeometry {
        n_link_knots: config.pgs_basis_config.num_knots.max(3),
        degree: 3,
    };
    let mut joint_cfg = JointModelConfig::default();
    joint_cfg.max_backfit_iter = config.max_iterations;
    joint_cfg.backfit_tol = config.convergence_tolerance;
    joint_cfg.max_reml_iter = config.reml_max_iterations as usize;
    joint_cfg.reml_tol = config.reml_convergence_tolerance;
    joint_cfg.firth_bias_reduction = config.firth_bias_reduction;

    let result = gam::joint::fit_joint_model_engine(
        data.y.view(),
        data.weights.view(),
        x.view(),
        s_list.clone(),
        link,
        geometry,
        joint_cfg,
    )?;

    let mut trained_config = config.clone();
    trained_config.sum_to_zero_constraints = sum_to_zero_constraints;
    trained_config.knot_vectors = knot_vectors;
    trained_config.range_transforms = range_transforms;
    trained_config.pc_null_transforms = pc_null_transforms;
    trained_config.interaction_centering_means = interaction_centering_means;
    trained_config.interaction_orth_alpha = interaction_orth_alpha;

    let coefficients = crate::calibrate::model::map_coefficients(&result.beta_base, &layout)?;
    let n = data.p.len();
    let raw_dim = 1 + data.pcs.ncols();
    let mut raw_train = Array2::<f64>::zeros((n, raw_dim));
    raw_train.slice_mut(s![.., 0]).assign(&data.p);
    if data.pcs.ncols() > 0 {
        raw_train.slice_mut(s![.., 1..]).assign(&data.pcs);
    }
    let hull = build_peeled_hull(&raw_train, 3).ok();

    let joint_link = crate::calibrate::model::JointLinkModel {
        knot_range: result.knot_range,
        knot_vector: result.knot_vector,
        link_transform: result.link_transform,
        beta_link: result.beta_link,
        degree: result.degree,
    };

    Ok(crate::calibrate::model::TrainedModel {
        config: trained_config,
        coefficients,
        lambdas: result.lambdas,
        hull,
        penalized_hessian: None,
        scale: Some(1.0),
        calibrator: None,
        joint_link: Some(joint_link),
        survival: None,
        survival_companions: HashMap::new(),
        mcmc_samples: None,
        smoothing_correction: None,
    })
}

pub fn train_survival_model(
    bundle: &crate::calibrate::survival_data::SurvivalTrainingBundle,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
    use crate::calibrate::model::ModelFamily;
    use crate::calibrate::survival::CompanionModelHandle;

    let survival_cfg = config.survival.as_ref().ok_or_else(|| {
        EstimationError::InvalidSpecification(
            "missing survival config for survival training".to_string(),
        )
    })?;
    let survival_spec = match config.model_family {
        ModelFamily::Survival(spec) => spec,
        _ => {
            return Err(EstimationError::InvalidInput(
                "train_survival_model expects Survival model family".to_string(),
            ));
        }
    };

    let mut primary_fit = fit_single_survival_model(bundle, config, survival_cfg, survival_spec)?;
    let mut survival_companions = HashMap::new();
    let n = bundle.data.pgs.len();
    let raw_dim = 1 + bundle.data.pcs.ncols();
    let mut raw_train = Array2::<f64>::zeros((n, raw_dim));
    raw_train.slice_mut(s![.., 0]).assign(&bundle.data.pgs);
    if bundle.data.pcs.ncols() > 0 {
        raw_train.slice_mut(s![.., 1..]).assign(&bundle.data.pcs);
    }
    let hull = build_peeled_hull(&raw_train, 3).ok();

    if survival_cfg.model_competing_risk {
        let mut mortality_bundle = crate::calibrate::survival_data::SurvivalTrainingBundle {
            data: bundle.data.clone(),
            age_transform: bundle.age_transform,
        };
        mortality_bundle.data.event_target = bundle.data.event_competing.clone();
        mortality_bundle.data.event_competing = Array1::<u8>::zeros(bundle.data.event_competing.len());

        let mut mortality_fit =
            fit_single_survival_model(&mortality_bundle, config, survival_cfg, survival_spec)?;
        let cross_cov = match (
            primary_fit.artifacts.mcmc_samples.as_ref(),
            mortality_fit.artifacts.mcmc_samples.as_ref(),
        ) {
            (Some(primary_samples), Some(mortality_samples)) => {
                cross_covariance_primary_companion(primary_samples, mortality_samples)
                    .or_else(|| {
                        Some(Array2::<f64>::zeros((
                            primary_fit.artifacts.coefficients.len(),
                            mortality_fit.artifacts.coefficients.len(),
                        )))
                    })
            }
            _ => Some(Array2::<f64>::zeros((
                primary_fit.artifacts.coefficients.len(),
                mortality_fit.artifacts.coefficients.len(),
            ))),
        };
        mortality_fit.artifacts.cross_covariance_to_primary = cross_cov;

        primary_fit.artifacts.companion_models.push(CompanionModelHandle {
            reference: "__internal_mortality".to_string(),
            cif_horizons: Vec::new(),
        });
        survival_companions.insert("__internal_mortality".to_string(), mortality_fit.artifacts);
    }

    let primary_samples = primary_fit.artifacts.mcmc_samples.clone();
    let primary_hessian = primary_fit.hessian.clone();

    Ok(crate::calibrate::model::TrainedModel {
        config: config.clone(),
        coefficients: crate::calibrate::model::MappedCoefficients::default(),
        lambdas: primary_fit.lambdas,
        hull,
        penalized_hessian: Some(primary_hessian),
        scale: None,
        calibrator: None,
        joint_link: None,
        survival: Some(primary_fit.artifacts),
        survival_companions,
        mcmc_samples: primary_samples,
        smoothing_correction: None,
    })
}

struct SurvivalFitResult {
    artifacts: crate::calibrate::survival::SurvivalModelArtifacts,
    lambdas: Vec<f64>,
    hessian: Array2<f64>,
}

fn fit_single_survival_model(
    bundle: &crate::calibrate::survival_data::SurvivalTrainingBundle,
    config: &crate::calibrate::model::ModelConfig,
    survival_cfg: &crate::calibrate::model::SurvivalModelConfig,
    survival_spec: crate::calibrate::survival::SurvivalSpec,
) -> Result<SurvivalFitResult, EstimationError> {
    use crate::calibrate::survival::{
        BasisDescriptor, CovariateLayout, SurvivalLayoutBundle, SurvivalModelArtifacts,
        TensorProductConfig, WorkingModelSurvival, build_survival_layout,
    };

    let log_entry = bundle
        .age_transform
        .transform_array(&bundle.data.age_entry)
        .map_err(map_survival_error)?;
    let mut min_log = f64::INFINITY;
    let mut max_log = f64::NEG_INFINITY;
    for &v in log_entry.iter() {
        min_log = min_log.min(v);
        max_log = max_log.max(v);
    }
    if !min_log.is_finite() || !max_log.is_finite() {
        return Err(EstimationError::InvalidSpecification(
            "non-finite transformed age values".to_string(),
        ));
    }
    if (max_log - min_log).abs() < 1e-9 {
        max_log = min_log + 1e-6;
    }
    let (_, age_knots) = create_basis::<Dense>(
        log_entry.view(),
        KnotSource::Generate {
            data_range: (min_log, max_log),
            num_internal_knots: survival_cfg.baseline_basis.num_knots,
        },
        survival_cfg.baseline_basis.degree,
        BasisOptions::value(),
    )?;
    let age_basis = BasisDescriptor {
        knot_vector: age_knots,
        degree: survival_cfg.baseline_basis.degree,
    };

    let time_varying_config = if let Some(tv) = survival_cfg.time_varying.as_ref() {
        let mut min_pgs = f64::INFINITY;
        let mut max_pgs = f64::NEG_INFINITY;
        for &value in bundle.data.pgs.iter() {
            min_pgs = min_pgs.min(value);
            max_pgs = max_pgs.max(value);
        }
        if !min_pgs.is_finite() || !max_pgs.is_finite() || (max_pgs - min_pgs).abs() < 1e-12 {
            None
        } else {
            let (_, pgs_knots) = create_basis::<Dense>(
                bundle.data.pgs.view(),
                KnotSource::Generate {
                    data_range: (min_pgs, max_pgs),
                    num_internal_knots: tv.pgs_basis.num_knots,
                },
                tv.pgs_basis.degree,
                BasisOptions::value(),
            )
            .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;
            Some(TensorProductConfig {
                label: tv.label.clone(),
                pgs_basis: BasisDescriptor {
                    knot_vector: pgs_knots,
                    degree: tv.pgs_basis.degree,
                },
                pgs_penalty_order: tv.pgs_penalty_order,
                lambda_age: tv.lambda_age,
                lambda_pgs: tv.lambda_pgs,
                lambda_null: tv.lambda_null,
            })
        }
    } else {
        None
    };

    let SurvivalLayoutBundle {
        mut layout,
        monotonicity,
        mut penalty_descriptors,
        interaction_metadata,
        time_varying_basis,
    } = build_survival_layout(
        &bundle.data,
        &age_basis,
        survival_cfg.guard_delta,
        config.penalty_order,
        survival_cfg.monotonic_grid_size,
        time_varying_config.as_ref(),
    )
    .map_err(map_survival_error)?;

    let pirls_options = WorkingModelPirlsOptions {
        max_iterations: config.max_iterations,
        convergence_tolerance: config.convergence_tolerance,
        max_step_halving: 20,
        min_step_size: 1e-6,
        firth_bias_reduction: false,
    };

    if !layout.penalties.blocks.is_empty() {
        let heuristic_lambdas = layout
            .penalties
            .blocks
            .iter()
            .map(|block| block.lambda.max(1e-12))
            .collect::<Vec<_>>();
        let objective = |rho: &Array1<f64>| -> Result<f64, EstimationError> {
            let mut eval_layout = layout.clone();
            for (block, &rho_i) in eval_layout.penalties.blocks.iter_mut().zip(rho.iter()) {
                block.lambda = rho_i.exp();
            }
            let mut model = WorkingModelSurvival::new(
                eval_layout,
                &bundle.data,
                monotonicity.clone(),
                survival_spec,
            )
            .map_err(map_survival_error)?;
            let p = model.layout.combined_exit.ncols();
            let result = pirls::run_working_model_pirls(
                &mut model,
                Coefficients::zeros(p),
                &pirls_options,
                |_| {},
            )?;
            Ok(result.state.deviance)
        };
        let seed_strategy = if layout.penalties.blocks.len() >= 10 {
            gam::seeding::SeedStrategy::Light
        } else {
            gam::seeding::SeedStrategy::Exhaustive
        };
        let smooth_opts = gam::families::royston_parmar::SurvivalLambdaOptimizerOptions {
            max_iter: config.reml_max_iterations as usize,
            tol: config.reml_convergence_tolerance,
            finite_diff_step: 1e-3,
            seed_config: gam::seeding::SeedConfig {
                strategy: seed_strategy,
                bounds: (-12.0, 12.0),
            },
        };
        let smooth_sol = gam::families::royston_parmar::optimize_survival_lambdas_with_multistart(
            layout.penalties.blocks.len(),
            Some(heuristic_lambdas.as_slice()),
            objective,
            &smooth_opts,
        )?;
        for (block, &rho_i) in layout
            .penalties
            .blocks
            .iter_mut()
            .zip(smooth_sol.rho.iter())
        {
            block.lambda = rho_i.exp();
        }
        for (descriptor, &rho_i) in penalty_descriptors
            .iter_mut()
            .zip(smooth_sol.rho.iter())
        {
            descriptor.lambda = rho_i.exp();
        }
    }

    let mut model =
        WorkingModelSurvival::new(layout.clone(), &bundle.data, monotonicity.clone(), survival_spec)
            .map_err(map_survival_error)?;
    let p = layout.combined_exit.ncols();
    let outcome = pirls::run_working_model_pirls(
        &mut model,
        Coefficients::zeros(p),
        &pirls_options,
        |_| {},
    )?;

    if matches!(outcome.status, PirlsStatus::Unstable) {
        return Err(EstimationError::PirlsDidNotConverge {
            max_iterations: config.max_iterations,
            last_change: outcome.last_gradient_norm,
        });
    }

    let coefficient_vector: Array1<f64> = outcome.beta.clone().into();
    let static_ranges = (0..layout.static_covariates.ncols())
        .map(|col| {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &v in layout.static_covariates.column(col).iter() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            crate::calibrate::survival::ValueRange {
                min: min_val,
                max: max_val,
            }
        })
        .collect();

    let mcmc_samples = if config.mcmc_enabled {
        let flat = hmc::SurvivalFlatInputs {
            age_entry: bundle.data.age_entry.view(),
            age_exit: bundle.data.age_exit.view(),
            event_target: bundle.data.event_target.view(),
            event_competing: bundle.data.event_competing.view(),
            weights: bundle.data.sample_weight.view(),
            x_entry: layout.combined_entry.view(),
            x_exit: layout.combined_exit.view(),
            x_derivative: layout.combined_derivative_exit.view(),
        };
        let penalties = to_engine_survival_penalties(&layout.penalties);
        let mono = to_engine_survival_monotonicity(&monotonicity);
        let spec = to_engine_survival_spec(&bundle.data);
        let nuts_cfg = hmc::NutsConfig::for_dimension(coefficient_vector.len());
        let survival_inputs = hmc::SurvivalNutsInputs {
            flat,
            penalties,
            monotonicity: mono,
            spec,
            mode: coefficient_vector.view(),
            hessian: outcome.state.hessian.view(),
        };
        let nuts = hmc::run_nuts_sampling_flattened_family(
            gam::types::LikelihoodFamily::RoystonParmar,
            hmc::FamilyNutsInputs::Survival(survival_inputs),
            &nuts_cfg,
        )
        .map_err(|err| {
            EstimationError::InvalidSpecification(format!("survival NUTS sampling failed: {err}"))
        })?;
        Some(nuts.samples)
    } else {
        None
    };

    let mut artifacts = SurvivalModelArtifacts {
        coefficients: coefficient_vector,
        age_basis,
        time_varying_basis,
        static_covariate_layout: CovariateLayout {
            column_names: layout.static_covariate_names.clone(),
            ranges: static_ranges,
        },
        penalties: penalty_descriptors,
        age_transform: layout.age_transform.clone(),
        reference_constraint: layout.reference_constraint.clone(),
        monotonicity,
        interaction_metadata,
        companion_models: Vec::new(),
        hessian_factor: build_expected_hessian_factor(&outcome.state.hessian),
        calibrator: None,
        mcmc_samples,
        cross_covariance_to_primary: None,
    };
    artifacts.calibrator = fit_survival_logit_calibrator(bundle, config, &artifacts)?;

    let lambdas = layout
        .penalties
        .blocks
        .iter()
        .map(|b| b.lambda)
        .collect::<Vec<_>>();

    Ok(SurvivalFitResult {
        artifacts,
        lambdas,
        hessian: outcome.state.hessian,
    })
}
