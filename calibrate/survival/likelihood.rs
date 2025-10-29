use super::artifact::{CovariateLayout, HessianFactor, SurvivalModelArtifacts};
use super::layout::{LayoutError, build_survival_layout};
use super::model_family::SurvivalSpec;
use super::penalties::PenaltyBlocks;
use super::pirls::{PirlsOptions, PirlsResult, run_pirls};
use super::working::WorkingModelSurvival;
use super::{data::SurvivalTrainingData, pirls::PirlsError};
use ndarray::Array1;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum FitError {
    #[error("layout error: {0}")]
    Layout(#[from] LayoutError),
    #[error("pirls error: {0}")]
    Pirls(#[from] PirlsError),
}

pub fn fit_survival_model(
    data: &SurvivalTrainingData,
    spec: &SurvivalSpec,
    options: PirlsOptions,
) -> Result<(SurvivalModelArtifacts, PirlsResult), FitError> {
    let layout = build_survival_layout(data, spec)?;
    let mut working = WorkingModelSurvival::new(
        layout.clone(),
        data.event_target.clone(),
        data.sample_weight.clone(),
    );
    let total_cols = layout.baseline_exit.ncols() + layout.static_covariates.ncols();
    let beta0 = Array1::zeros(total_cols);
    let penalties = vec![layout.penalties.clone()];
    let penalties_scaled: Vec<PenaltyBlocks> = penalties
        .into_iter()
        .map(|mut p| {
            p.lambda = 1.0;
            p
        })
        .collect();
    let pirls_result = run_pirls(&mut working, &penalties_scaled, beta0, options)?;

    let artifacts = SurvivalModelArtifacts {
        coefficients: pirls_result.beta.clone(),
        age_basis: layout.age_descriptor.clone(),
        time_varying_basis: layout.time_varying_descriptor.clone(),
        static_covariate_layout: CovariateLayout {
            column_names: layout.static_covariate_names.clone(),
        },
        penalties: layout.penalties.descriptor.clone(),
        age_transform: data.age_transform,
        reference_constraint: layout.reference_constraint.clone(),
        hessian_factor: Some(HessianFactor::Observed {
            matrix: pirls_result.penalized_hessian.clone(),
        }),
    };

    Ok((artifacts, pirls_result))
}
