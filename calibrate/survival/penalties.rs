use crate::calibrate::basis::{self, BasisError};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PenaltyDescriptor {
    Baseline,
    TimeVarying,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlocks {
    pub descriptor: PenaltyDescriptor,
    pub matrix: Array2<f64>,
    pub lambda: f64,
}

#[derive(Debug, Error)]
pub enum PenaltyError {
    #[error("basis error: {0}")]
    Basis(#[from] BasisError),
}

impl PenaltyBlocks {
    pub fn difference_penalty(
        num_cols: usize,
        order: usize,
        descriptor: PenaltyDescriptor,
    ) -> Result<Self, PenaltyError> {
        let matrix = basis::create_difference_penalty_matrix(num_cols, order)?;
        Ok(PenaltyBlocks {
            descriptor,
            matrix,
            lambda: 1.0,
        })
    }

    pub fn apply_to(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.matrix.dot(beta)
    }

    pub fn hessian(&self) -> Array2<f64> {
        self.matrix.clone()
    }
}
