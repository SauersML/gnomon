pub use gam::estimate::{
    EstimationError, ExternalOptimOptions, ExternalOptimResult, FitOptions, FitResult,
    SurvivalTrainFlatInputs, evaluate_external_cost_and_ridge, evaluate_external_gradients,
    fit_gam, optimize_external_design,
};

pub use gam::estimate::internal;

pub fn train_model(
    data: &crate::calibrate::data::TrainingData,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
    let engine_config = crate::calibrate::model::to_engine_model_config(config)?;
    let engine_model = gam::estimate::train_model(
        gam::estimate::TrainModelInputs {
            response: data.y.view(),
            primary_feature: data.p.view(),
            binary_feature: data.sex.view(),
            auxiliary_features: data.pcs.view(),
            sample_weight: data.weights.view(),
        },
        &engine_config,
    )?;
    crate::calibrate::model::from_engine_trained_model(&engine_model)
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
