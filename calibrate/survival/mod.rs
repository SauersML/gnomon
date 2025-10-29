//! Survival modelling architecture built around the Royston–Parmar parameterisation.
//!
//! This module follows the implementation plan captured in `plan/survival.md`.
//! It introduces a dedicated model family for survival analysis, data ingestion
//! pipelines, layout builders, PIRLS integration, artifact persistence, and
//! prediction/calibration helpers.  All pieces are designed to re-use the
//! existing basis/penalty infrastructure exposed by the rest of the calibrate
//! crate while providing survival-specific likelihood code.

mod age;
mod artifact;
mod data;
mod layout;
mod likelihood;
mod model_family;
mod penalties;
mod pirls;
mod prediction;
mod working;

pub use age::{AgeTransform, GuardedLogAge};
pub use artifact::{HessianFactor, SurvivalModelArtifacts};
pub use data::{
    CovariateViews, SurvivalPredictionInputs, SurvivalTrainingData, load_survival_training_data,
};
pub use layout::{BasisDescriptor, ReferenceConstraint, SurvivalLayout, SurvivalLayoutBuilder};
pub use likelihood::{FitError, fit_survival_model};
pub use model_family::{ModelFamily, SurvivalSpec};
pub use penalties::{PenaltyBlocks, PenaltyDescriptor};
pub use pirls::{PirlsOptions, PirlsResult, PirlsStatus, run_pirls};
pub use prediction::{
    Covariates, conditional_absolute_risk, cumulative_hazard, cumulative_incidence,
};
pub use working::{WorkingModel, WorkingModelGam, WorkingModelSurvival, WorkingState};
