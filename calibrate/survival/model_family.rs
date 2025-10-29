use serde::{Deserialize, Serialize};

use crate::calibrate::model::LinkFunction;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalSpec {
    pub monotonicity_grid_size: usize,
    pub basis_knots: usize,
    pub basis_degree: usize,
    pub penalty_order: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Gam(LinkFunction),
    Survival(SurvivalSpec),
}
