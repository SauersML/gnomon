pub use gam::estimate::{
    EstimationError, ExternalOptimOptions, ExternalOptimResult, FitOptions, FitResult,
    SurvivalTrainFlatInputs, evaluate_external_cost_and_ridge, evaluate_external_gradients,
    fit_gam, optimize_external_design,
};

pub use gam::estimate::internal;
use ndarray::Array1;

pub fn train_model(
    data: &crate::calibrate::data::TrainingData,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
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

    let family = match config.model_family {
        crate::calibrate::model::ModelFamily::Gam(gam::types::LinkFunction::Identity) => {
            gam::types::LikelihoodFamily::GaussianIdentity
        }
        crate::calibrate::model::ModelFamily::Gam(gam::types::LinkFunction::Logit) => {
            gam::types::LikelihoodFamily::BinomialLogit
        }
        crate::calibrate::model::ModelFamily::Survival(_) => {
            return Err(EstimationError::InvalidInput(
                "train_model expects GAM family; use train_survival_model for survival".to_string(),
            ));
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

    let mut trained_config = config.clone();
    trained_config.sum_to_zero_constraints = sum_to_zero_constraints;
    trained_config.knot_vectors = knot_vectors;
    trained_config.range_transforms = range_transforms;
    trained_config.pc_null_transforms = pc_null_transforms;
    trained_config.interaction_centering_means = interaction_centering_means;
    trained_config.interaction_orth_alpha = interaction_orth_alpha;

    let coefficients = crate::calibrate::model::map_coefficients(&fit.beta, &layout)?;

    Ok(crate::calibrate::model::TrainedModel {
        config: trained_config,
        coefficients,
        lambdas: fit.lambdas.to_vec(),
        hull: None,
        penalized_hessian: None,
        scale: Some(fit.scale),
        calibrator: None,
        joint_link: None,
        survival: None,
        survival_companions: std::collections::HashMap::new(),
        mcmc_samples: None,
        smoothing_correction: None,
    })
}

pub fn train_survival_model(
    bundle: &crate::calibrate::survival_data::SurvivalTrainingBundle,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
    let engine_config = crate::calibrate::model::to_engine_model_config(config)?;
    let engine_model = gam::estimate::train_survival_model(
        SurvivalTrainFlatInputs {
            age_entry: bundle.data.age_entry.view(),
            age_exit: bundle.data.age_exit.view(),
            event_target: bundle.data.event_target.view(),
            event_competing: bundle.data.event_competing.view(),
            sample_weight: bundle.data.sample_weight.view(),
            static_covariates: bundle.data.static_covariates.view(),
            static_covariate_names: &bundle.data.static_covariate_names,
            time_varying_covariate: bundle.data.time_varying_covariate.as_ref().map(|v| v.view()),
            age_transform: bundle.age_transform,
        },
        &engine_config,
    )?;
    crate::calibrate::model::from_engine_trained_model(&engine_model)
}
