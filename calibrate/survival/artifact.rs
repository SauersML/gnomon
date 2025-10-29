use super::age::AgeTransform;
use super::layout::{BasisDescriptor, ReferenceConstraint};
use super::penalties::PenaltyDescriptor;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateLayout {
    pub column_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HessianFactor {
    Observed { matrix: Array2<f64> },
    Expected { matrix: Array2<f64> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: PenaltyDescriptor,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<HessianFactor>,
}
