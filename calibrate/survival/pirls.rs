use super::penalties::PenaltyBlocks;
use super::working::{WorkingModel, WorkingState};
use crate::calibrate::faer_ndarray::{FaerArrayView, FaerColView, array1_to_col_mat_mut};
use faer::Side;
use faer::linalg::solvers::{Ldlt as FaerLdlt, Solve as FaerSolve};
use ndarray::{Array1, Array2};
use thiserror::Error;

#[derive(Debug, Clone)]
pub struct PirlsOptions {
    pub tolerance: f64,
    pub max_iterations: usize,
}

#[derive(Debug, Clone)]
pub enum PirlsStatus {
    Converged,
    MaxIterations,
}

#[derive(Debug, Clone)]
pub struct PirlsResult {
    pub beta: Array1<f64>,
    pub deviance: f64,
    pub gradient: Array1<f64>,
    pub status: PirlsStatus,
    pub iterations: usize,
    pub penalized_hessian: Array2<f64>,
}

#[derive(Debug, Error)]
pub enum PirlsError {
    #[error("ldlt factorization failed")]
    Factorization,
}

fn accumulate_penalties(
    beta: &Array1<f64>,
    gradient: &mut Array1<f64>,
    hessian: &mut Array2<f64>,
    penalties: &[PenaltyBlocks],
) {
    for penalty in penalties {
        gradient += &(penalty.lambda * penalty.matrix.dot(beta));
        hessian += &(penalty.lambda * penalty.matrix.clone());
    }
}

pub fn run_pirls<M: WorkingModel>(
    model: &mut M,
    penalties: &[PenaltyBlocks],
    mut beta: Array1<f64>,
    options: PirlsOptions,
) -> Result<PirlsResult, PirlsError> {
    for iteration in 0..options.max_iterations {
        let WorkingState {
            eta: _eta,
            mut gradient,
            mut hessian,
            deviance,
        } = model.update(&beta);

        accumulate_penalties(&beta, &mut gradient, &mut hessian, penalties);

        let grad_norm = gradient.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if grad_norm < options.tolerance {
            return Ok(PirlsResult {
                beta,
                deviance,
                gradient,
                status: PirlsStatus::Converged,
                iterations: iteration,
                penalized_hessian: hessian,
            });
        }

        let mut rhs = gradient.map(|v| -v);
        let mut rhs_mat = array1_to_col_mat_mut(&mut rhs);
        let h_view = FaerArrayView::new(&hessian);
        let factor =
            FaerLdlt::new(h_view.as_ref(), Side::Lower).map_err(|_| PirlsError::Factorization)?;
        let rhs_view = FaerColView::new(&rhs);
        factor
            .solve(rhs_view.as_ref(), rhs_mat.as_mut())
            .map_err(|_| PirlsError::Factorization)?;
        beta += &rhs;

        if rhs.iter().map(|v| v.abs()).fold(0.0, f64::max) < options.tolerance {
            return Ok(PirlsResult {
                beta,
                deviance,
                gradient,
                status: PirlsStatus::Converged,
                iterations: iteration + 1,
                penalized_hessian: hessian,
            });
        }
    }

    let WorkingState {
        eta: _eta,
        mut gradient,
        mut hessian,
        deviance,
    } = model.update(&beta);
    accumulate_penalties(&beta, &mut gradient, &mut hessian, penalties);

    Ok(PirlsResult {
        beta,
        deviance,
        gradient,
        status: PirlsStatus::MaxIterations,
        iterations: options.max_iterations,
        penalized_hessian: hessian,
    })
}
