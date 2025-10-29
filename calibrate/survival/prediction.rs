use super::age::AgeTransform;
use super::artifact::{CovariateLayout, SurvivalModelArtifacts};
use crate::calibrate::basis;
use ndarray::{Array1, s};

#[derive(Debug, Clone)]
pub struct Covariates {
    pub pgs: f64,
    pub sex: f64,
    pub pcs: Vec<f64>,
}

fn baseline_vector(age: f64, artifacts: &SurvivalModelArtifacts) -> Array1<f64> {
    let log_age = artifacts.age_transform.transform_value(age);
    let data = Array1::from_vec(vec![log_age]);
    let (basis_row, _) = basis::create_bspline_basis_with_knots(
        data.view(),
        artifacts.age_basis.knots.view(),
        artifacts.age_basis.degree,
    )
    .expect("basis evaluation should succeed");
    let constrained = (*basis_row).dot(&artifacts.reference_constraint.transform);
    constrained.row(0).to_owned()
}

fn static_vector(covariates: &Covariates, layout: &CovariateLayout) -> Array1<f64> {
    let mut values = Vec::with_capacity(layout.column_names.len());
    for name in &layout.column_names {
        match name.as_str() {
            "intercept" => values.push(1.0),
            "pgs" => values.push(covariates.pgs),
            "sex" => values.push(covariates.sex),
            other if other.starts_with("pc") => {
                let idx: usize = other[2..].parse().unwrap_or(0);
                let value = covariates
                    .pcs
                    .get(idx.saturating_sub(1))
                    .copied()
                    .unwrap_or(0.0);
                values.push(value);
            }
            _ => values.push(0.0),
        }
    }
    Array1::from(values)
}

fn linear_predictor(age: f64, covariates: &Covariates, artifacts: &SurvivalModelArtifacts) -> f64 {
    let baseline = baseline_vector(age, artifacts);
    let static_vec = static_vector(covariates, &artifacts.static_covariate_layout);
    let baseline_len = baseline.len();
    let static_len = static_vec.len();
    let baseline_beta = artifacts.coefficients.slice(s![0..baseline_len]).to_owned();
    let static_beta = artifacts
        .coefficients
        .slice(s![baseline_len..baseline_len + static_len])
        .to_owned();
    baseline.dot(&baseline_beta) + static_vec.dot(&static_beta)
}

pub fn cumulative_hazard(
    age: f64,
    covariates: &Covariates,
    artifacts: &SurvivalModelArtifacts,
) -> f64 {
    linear_predictor(age, covariates, artifacts).exp()
}

pub fn cumulative_incidence(
    age: f64,
    covariates: &Covariates,
    artifacts: &SurvivalModelArtifacts,
) -> f64 {
    let h = cumulative_hazard(age, covariates, artifacts);
    1.0 - (-h).exp()
}

pub fn conditional_absolute_risk(
    t0: f64,
    t1: f64,
    covariates: &Covariates,
    artifacts: &SurvivalModelArtifacts,
    cif_competing_t0: f64,
) -> f64 {
    let cif_t0 = cumulative_incidence(t0, covariates, artifacts);
    let cif_t1 = cumulative_incidence(t1, covariates, artifacts);
    let delta = (cif_t1 - cif_t0).max(0.0);
    let denom = (1.0 - cif_t0 - cif_competing_t0).max(1e-12);
    delta / denom
}
