use crate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::data::TrainingData;
use crate::model::{LinkFunction, ModelConfig, TrainedModel};

use ndarray::{s, Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::Solve;
use thiserror::Error;

/// A comprehensive error type for the model estimation process.
#[derive(Error, Debug, PartialEq)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] basis::BasisError),

    #[error("The penalized matrix system is singular and could not be solved. This can happen with a very large lambda. Error: {0}")]
    MatrixInversionFailed(ndarray_linalg::error::LinalgError),

    #[error("Iterative solver did not converge within {max_iterations} iterations. Last deviance change was {last_change:.6e}.")]
    DidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },
}

/// The main entry point for model training.
///
/// This function orchestrates the entire fitting process using a penalized
/// Iteratively Reweighted Least Squares (IRLS) algorithm to find the optimal
/// coefficients for the specified GAM.
pub fn train_model(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    log::info!(
        "Starting model training with lambda = {:.4e}, penalty order = {}, {} total samples.",
        config.lambda,
        config.penalty_order,
        data.y.len()
    );

    // 1. Build the large, one-time matrices for the model. This is the most complex
    //    non-iterative part, translating the model formula into matrices.
    let (x_matrix, penalty_matrix) = internal::build_design_and_penalty_matrices(data, config)?;
    log::info!(
        "Constructed design matrix with {} samples and {} total coefficients.",
        x_matrix.nrows(),
        x_matrix.ncols()
    );

    // 2. Initialize the IRLS algorithm.
    let mut beta = Array1::zeros(x_matrix.ncols());
    let mut last_deviance = f64::INFINITY;

    // 3. Run the IRLS loop.
    for iter in 1..=config.max_iterations {
        // Calculate the linear predictor `eta` from the current coefficients.
        let eta = x_matrix.dot(&beta);

        // Update GLM-specific vectors based on the current `eta`.
        let (mu, weights, z) = internal::update_glm_vectors(data.y.view(), &eta, config.link_function);

        // Solve the core penalized weighted least squares system to get new coefficients.
        // This is the numerical workhorse of each iteration.
        beta = internal::solve_penalized_wls(
            x_matrix.view(),
            z.view(),
            weights.view(),
            penalty_matrix.view(),
            config.lambda,
        )?;

        // Check for convergence by monitoring the change in deviance.
        let deviance = internal::calculate_deviance(data.y.view(), &mu, config.link_function);
        let deviance_change = (last_deviance - deviance).abs();

        log::debug!(
            "Iter {: >3}: Deviance = {:.6}, Change = {:.6e}",
            iter,
            deviance,
            deviance_change
        );

        if deviance_change < config.convergence_tolerance {
            log::info!(
                "IRLS converged after {} iterations. Final Deviance: {:.6}",
                iter,
                deviance
            );
            // On success, package the results into a TrainedModel struct.
            return Ok(TrainedModel {
                config: config.clone(),
                coefficients: beta,
            });
        }
        last_deviance = deviance;
    }

    // If the loop finishes without converging, return an error.
    Err(EstimationError::DidNotConverge {
        max_iterations: config.max_iterations,
        last_change: (last_deviance - internal::calculate_deviance(data.y.view(), &internal::update_glm_vectors(data.y.view(), &x_matrix.dot(&beta), config.link_function).0, config.link_function)).abs(),
    })
}

/// Internal module for implementation details.
mod internal {
    use super::*;

    /// Constructs the full design matrix `X` and the block-diagonal penalty matrix `S`.
    /// This is the direct implementation of the model formula from the user's paper.
    pub(super) fn build_design_and_penalty_matrices(
        data: &TrainingData,
        config: &ModelConfig,
    ) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
        // --- 1. Generate all individual basis expansions ---
        let (pgs_basis, _) = create_bspline_basis(
            data.p.view(),
            Some(data.p.view()),
            config.pgs_range,
            config.pgs_basis_config.num_knots,
            config.pgs_basis_config.degree,
        )?;

        let pc_bases: Vec<Array2<f64>> = data
            .pcs
            .axis_iter(Axis(1))
            .zip(config.pc_ranges.iter().zip(config.pc_basis_configs.iter()))
            .map(|(pc_col, (&range, pc_conf))| {
                create_bspline_basis(
                    pc_col.view(),
                    Some(pc_col.view()),
                    range,
                    pc_conf.num_knots,
                    pc_conf.degree,
                )
                .map(|(basis, _)| basis)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // --- 2. Assemble the full design matrix X ---
        let mut design_cols_owned = Vec::new();
        design_cols_owned.push(Array1::ones(data.y.len()));

        // Term 1: Ancestry-specific baseline (main effects of PCs, f_0l terms)
        for pc_basis in &pc_bases {
            for col in pc_basis.axis_iter(Axis(1)) {
                design_cols_owned.push(col.to_owned());
            }
        }

        // Term 2: Main effects of the raw PGS (gamma_m0 * B_m(P) for m > 0)
        for col in pgs_basis.slice(s![.., 1..]).axis_iter(Axis(1)) {
            design_cols_owned.push(col.to_owned());
        }

        // Term 3: Non-linear interactions (f_ml(PC_jl) * B_m(P_j))
        for pgs_basis_col in pgs_basis.slice(s![.., 1..]).axis_iter(Axis(1)) {
            for pc_basis in &pc_bases {
                for pc_basis_col in pc_basis.axis_iter(Axis(1)) {
                    design_cols_owned.push(&pgs_basis_col * &pc_basis_col);
                }
            }
        }

        let design_views: Vec<_> = design_cols_owned.iter().map(|c| c.view()).collect();
        let x_matrix = ndarray::stack(Axis(1), &design_views)
            .expect("Stacking design matrix columns failed.");

        // --- 3. Assemble the block-diagonal penalty matrix S ---
        let mut penalty_matrix = Array2::zeros((x_matrix.ncols(), x_matrix.ncols()));
        let mut current_pos = 1; // Start after the global intercept (which is unpenalized)

        // Penalty for main effects of PCs (f_0l terms)
        for pc_conf in &config.pc_basis_configs {
            let num_basis = pc_conf.num_knots + pc_conf.degree + 1;
            let p_mat = create_difference_penalty_matrix(num_basis, config.penalty_order)?;
            penalty_matrix.slice_mut(s![current_pos.., current_pos..]).assign(&p_mat);
            current_pos += num_basis;
        }

        // Main PGS effects (gamma_m0 terms) are NOT penalized. Skip their columns.
        current_pos += pgs_basis.ncols() - 1;

        // Penalty for interaction terms (f_ml terms)
        // For each PGS basis function (m>0), we have a set of splines on the PCs.
        for _m_idx in 1..pgs_basis.ncols() {
            for pc_conf in &config.pc_basis_configs {
                 let num_basis = pc_conf.num_knots + pc_conf.degree + 1;
                 let p_mat = create_difference_penalty_matrix(num_basis, config.penalty_order)?;
                 penalty_matrix.slice_mut(s![current_pos.., current_pos..]).assign(&p_mat);
                 current_pos += num_basis;
            }
        }

        Ok((x_matrix, penalty_matrix))
    }

    /// Solves the core penalized weighted least squares system for one IRLS step.
    /// This finds `beta` that minimizes `(z - X*beta)' * W * (z - X*beta) + lambda * beta' * S * beta`.
    /// The solution is `beta = (X'WX + lambda*S)^-1 * X'Wz`.
    pub(super) fn solve_penalized_wls(
        x: ArrayView2<f64>,
        z: ArrayView1<f64>,
        w: ArrayView1<f64>, // Diagonal of the weight matrix
        s: ArrayView2<f64>,
        lambda: f64,
    ) -> Result<Array1<f64>, EstimationError> {
        let x_t = x.t();
        // This is X'W, calculated efficiently using broadcasting.
        let x_t_w = &x_t * w;
        
        // Form the left-hand side: LHS = (X'W)X + lambda*S
        let lhs = x_t_w.dot(&x) + s * lambda;

        // Form the right-hand-side: RHS = (X'W)z
        let rhs = x_t_w.dot(&z);

        // Solve the system `LHS * beta = RHS` for `beta`.
        lhs.solve_into(rhs)
            .map_err(EstimationError::MatrixInversionFailed)
    }

    /// Calculates the mean vector `mu`, weight vector `w`, and working response `z`
    /// for a given link function and linear predictor `eta`.
    pub(super) fn update_glm_vectors(
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        link: LinkFunction,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        const MIN_WEIGHT: f64 = 1e-6;

        match link {
            LinkFunction::Logit => {
                let mu: Array1<f64> = eta.mapv(|e| 1.0 / (1.0 + (-e).exp())); // Sigmoid function
                // Clamp weights to prevent division by zero for mu values of 0 or 1.
                let weights: Array1<f64> = (&mu * (1.0 - &mu)).mapv(|v| v.max(MIN_WEIGHT));
                let z: Array1<f64> = eta + (y - &mu) / &weights;
                (mu, weights, z)
            }
            LinkFunction::Identity => {
                let mu = eta.clone();
                let weights = Array1::ones(y.len());
                let z = y.to_owned();
                (mu, weights, z)
            }
        }
    }

    /// Calculates the deviance of the model given the observed and fitted values.
    pub(super) fn calculate_deviance(
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        link: LinkFunction,
    ) -> f64 {
        const EPS: f64 = 1e-9;
        match link {
            LinkFunction::Logit => {
                // Deviance for Bernoulli/Binomial distribution.
                let term1 = y * (y / mu.mapv(|v| v.max(EPS))).mapv(f64::ln);
                let term2 = (1.0 - y)
                    * ((1.0 - y) / (1.0 - mu.mapv(|v| v.max(EPS))))
                        .mapv(f64::ln);
                // The terms can be NaN if y=0 and term1 is calculated, so we use `nansum`.
                2.0 * (term1.sum_skipnan() + term2.sum_skipnan())
            }
            LinkFunction::Identity => (y - mu).mapv(|v| v.powi(2)).sum(),
        }
    }
}
