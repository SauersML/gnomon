use crate::calibrate::basis::{self, BasisError};
use ndarray::{Array1, Array2, s};
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
    pub offset: usize,
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
        offset: usize,
    ) -> Result<Self, PenaltyError> {
        let matrix = basis::create_difference_penalty_matrix(num_cols, order)?;
        Ok(PenaltyBlocks {
            descriptor,
            matrix,
            lambda: 1.0,
            offset,
        })
    }

    pub fn apply_to(&self, beta: &Array1<f64>) -> Array1<f64> {
        let start = self.offset;
        let end = start + self.matrix.ncols();
        self.matrix.dot(&beta.slice(s![start..end]).to_owned())
    }

    pub fn hessian(&self) -> Array2<f64> {
        self.matrix.clone()
    }
}
