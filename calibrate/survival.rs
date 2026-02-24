pub use gam::survival::{
    AgeTransform, BasisDescriptor, CholeskyFactor, CompanionModelHandle, CovariateLayout,
    CovariateViews, HessianFactor, MonotonicityPenalty, ReferenceConstraint, SurvivalError,
    SurvivalLayoutInputs, SurvivalModelArtifacts, SurvivalPredictionInputs, SurvivalSpec,
    ValueRange, conditional_absolute_risk, cumulative_incidence,
    delta_method_standard_errors, survival_calibrator_features, validate_survival_inputs,
};

pub type SurvivalTrainingData = SurvivalLayoutInputs;
