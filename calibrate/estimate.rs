use crate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::data::TrainingData;
use crate::model::{LinkFunction, MappedCoefficients, MainEffects, ModelConfig, TrainedModel};

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Solve;
use std::collections::HashMap;
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
    // Track the last deviance change to report it correctly on failure.
    let mut last_deviance_change = f64::INFINITY;

    // 3. Run the IRLS loop.
    for iter in 1..=config.max_iterations {
        // Calculate the linear predictor `eta` from the current coefficients.
        let eta = x_matrix.dot(&beta);

        // Update GLM-specific vectors based on the current `eta`.
        let (mu, weights, z) =
            internal::update_glm_vectors(data.y.view(), &eta, config.link_function);

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
        // Store the change from this iteration.
        last_deviance_change = deviance_change;

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
            // On success, "un-flatten" the beta vector into the structured MappedCoefficients.
            let mapped_coefficients = internal::map_coefficients(&beta, config);
            return Ok(TrainedModel {
                config: config.clone(),
                coefficients: mapped_coefficients,
            });
        }
        last_deviance = deviance;
    }

    // If the loop finishes without converging, return an error.
    Err(EstimationError::DidNotConverge {
        max_iterations: config.max_iterations,
        last_change: last_deviance_change,
    })
}

/// Internal module for implementation details.
mod internal {
    use super::*;

    /// Constructs the full design matrix `X` and the block-diagonal penalty matrix `S`.
    /// This is a performance-optimized implementation.
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

        // --- 2. Assemble the full design matrix X (optimized) ---
        // Pre-calculate the total number of columns to allocate the matrix only once.
        let num_samples = data.y.len();
        let num_pc_main_coeffs: usize = pc_bases.iter().map(|b| b.ncols()).sum();
        let num_pgs_main_coeffs = pgs_basis.ncols() - 1;
        let num_interaction_coeffs = num_pgs_main_coeffs * num_pc_main_coeffs;
        let total_coeffs = 1 + num_pc_main_coeffs + num_pgs_main_coeffs + num_interaction_coeffs;

        let mut x_matrix = Array2::zeros((num_samples, total_coeffs));
        let mut current_col = 0;

        // Fill columns without re-allocating.
        // Intercept
        x_matrix.column_mut(current_col).fill(1.0);
        current_col += 1;

        // Term 1: Main effects of PCs
        for pc_basis in &pc_bases {
            let num_cols = pc_basis.ncols();
            x_matrix
                .slice_mut(s![.., current_col..current_col + num_cols])
                .assign(pc_basis);
            current_col += num_cols;
        }

        // Term 2: Main effects of PGS (m > 0)
        let pgs_main_basis = pgs_basis.slice(s![.., 1..]);
        x_matrix
            .slice_mut(s![.., current_col..current_col + num_pgs_main_coeffs])
            .assign(&pgs_main_basis);
        current_col += num_pgs_main_coeffs;

        // Term 3: Non-linear interactions
        for pgs_basis_col in pgs_main_basis.axis_iter(Axis(1)) {
            for pc_basis in &pc_bases {
                let num_cols = pc_basis.ncols();
                for (j, pc_basis_col) in pc_basis.axis_iter(Axis(1)).enumerate() {
                    let interaction_term = &pgs_basis_col * &pc_basis_col;
                    x_matrix.column_mut(current_col + j).assign(&interaction_term);
                }
                current_col += num_cols;
            }
        }

        // --- 3. Assemble the block-diagonal penalty matrix S ---
        let mut penalty_matrix = Array2::zeros((total_coeffs, total_coeffs));
        let mut current_pos = 1; // Skip unpenalized global intercept

        // Penalty for main effects of PCs
        for (pc_conf, pc_basis) in config.pc_basis_configs.iter().zip(pc_bases.iter()) {
            let num_basis = pc_basis.ncols();
            let p_mat = create_difference_penalty_matrix(num_basis, config.penalty_order)?;
            penalty_matrix
                .slice_mut(s![current_pos..current_pos + num_basis, current_pos..current_pos + num_basis])
                .assign(&p_mat);
            current_pos += num_basis;
        }

        // Main PGS effects are unpenalized.
        current_pos += num_pgs_main_coeffs;

        // Penalty for interaction terms
        for _m_idx in 1..pgs_basis.ncols() {
            for (pc_conf, pc_basis) in config.pc_basis_configs.iter().zip(pc_bases.iter()) {
                 let num_basis = pc_basis.ncols();
                 let p_mat = create_difference_penalty_matrix(num_basis, config.penalty_order)?;
                 penalty_matrix
                    .slice_mut(s![current_pos..current_pos + num_basis, current_pos..current_pos + num_basis])
                    .assign(&p_mat);
                 current_pos += num_basis;
            }
        }

        Ok((x_matrix, penalty_matrix))
    }

    /// Solves the core penalized weighted least squares system for one IRLS step.
    pub(super) fn solve_penalized_wls(
        x: ArrayView2<f64>,
        z: ArrayView1<f64>,
        w: ArrayView1<f64>, // Diagonal of the weight matrix
        s: ArrayView2<f64>,
        lambda: f64,
    ) -> Result<Array1<f64>, EstimationError> {
        let x_t = x.t();
        let x_t_w = &x_t * w;
        let lhs = x_t_w.dot(&x) + &(s * lambda);
        let rhs = x_t_w.dot(&z);
        lhs.solve_into(rhs)
            .map_err(EstimationError::MatrixInversionFailed)
    }

    /// Calculates the mean vector `mu`, weight vector `w`, and working response `z`.
    pub(super) fn update_glm_vectors(
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        link: LinkFunction,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        const MIN_WEIGHT: f64 = 1e-6;

        match link {
            LinkFunction::Logit => {
                let mu: Array1<f64> = eta.mapv(|e| 1.0 / (1.0 + (-e).exp()));
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
    /// This implementation is numerically stable and memory-efficient.
    pub(super) fn calculate_deviance(
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        link: LinkFunction,
    ) -> f64 {
        const EPS: f64 = 1e-9;
        match link {
            LinkFunction::Logit => {
                // Sum the deviance residuals directly without allocating a new array.
                let total_residual = ndarray::Zip::from(y)
                    .and_from(mu)
                    .fold(0.0, |acc, &yi, &mui| {
                        let mui_c = mui.clamp(EPS, 1.0 - EPS);

                        let term1 = if yi > EPS { yi * (yi / mui_c).ln() } else { 0.0 };
                        let term2 = if yi < 1.0 - EPS { (1.0 - yi) * ((1.0 - yi) / (1.0 - mui_c)).ln() } else { 0.0 };
                        
                        acc + term1 + term2
                    });
                2.0 * total_residual
            }
            LinkFunction::Identity => (y - mu).mapv(|v| v.powi(2)).sum(),
        }
    }

    /// Converts the flat coefficient vector `beta` into a structured `MappedCoefficients`.
    /// This is the inverse of the logic in `model::internal::flatten_coefficients`.
    pub(super) fn map_coefficients(
        beta: &Array1<f64>,
        config: &ModelConfig,
    ) -> MappedCoefficients {
        let mut current_pos = 0;

        let intercept = beta[current_pos];
        current_pos += 1;

        let mut pc_coeffs_map = HashMap::new();
        for (pc_name, pc_conf) in config.pc_names.iter().zip(&config.pc_basis_configs) {
            let num_basis = pc_conf.num_knots + pc_conf.degree + 1;
            let coeffs = beta.slice(s![current_pos..current_pos + num_basis]).to_vec();
            pc_coeffs_map.insert(pc_name.clone(), coeffs);
            current_pos += num_basis;
        }

        let num_pgs_main_coeffs = config.pgs_basis_config.num_knots + config.pgs_basis_config.degree;
        let pgs_coeffs = beta.slice(s![current_pos..current_pos + num_pgs_main_coeffs]).to_vec();
        current_pos += num_pgs_main_coeffs;

        let mut interaction_effects = HashMap::new();
        for m in 1..=num_pgs_main_coeffs {
            let pgs_key = format!("pgs_B{}", m);
            let mut pc_interaction_map = HashMap::new();
            for (pc_name, pc_conf) in config.pc_names.iter().zip(&config.pc_basis_configs) {
                let num_basis = pc_conf.num_knots + pc_conf.degree + 1;
                let coeffs = beta.slice(s![current_pos..current_pos + num_basis]).to_vec();
                pc_interaction_map.insert(pc_name.clone(), coeffs);
                current_pos += num_basis;
            }
            interaction_effects.insert(pgs_key, pc_interaction_map);
        }

        MappedCoefficients {
            intercept,
            main_effects: MainEffects {
                pgs: pgs_coeffs,
                pcs: pc_coeffs_map,
            },
            interaction_effects,
        }
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_calculate_deviance_logit_stable() {
        // Test with edge cases y=0 and y=1
        let y = array![0.0, 1.0, 0.5];
        let mu = array![0.1, 0.9, 0.5]; // mu shouldn't be 0 or 1
        let deviance = internal::calculate_deviance(y.view(), &mu, LinkFunction::Logit);

        // For y=0, mu=0.1: 2 * (0 + 1*ln(1/0.9)) = 2 * ln(1.111...) = 0.21072
        let d1 = 2.0 * (1.0 / 0.9).ln();
        // For y=1, mu=0.9: 2 * (1*ln(1/0.9) + 0) = 2 * ln(1.111...) = 0.21072
        let d2 = 2.0 * (1.0 / 0.9).ln();
        // For y=0.5, mu=0.5: 2 * (0.5*ln(1) + 0.5*ln(1)) = 0
        let d3 = 0.0;

        let expected_deviance = d1 + d2 + d3;
        assert!((deviance - expected_deviance).abs() < 1e-5);

        // Test with y=0 and mu close to 0
        let y_zero = array![0.0];
        let mu_small = array![1e-12];
        let deviance_zero = internal::calculate_deviance(y_zero.view(), &mu_small, LinkFunction::Logit);
        // Should be 2 * ln(1/(1-1e-12)) which is very close to 0.
        assert!(deviance_zero.abs() < 1e-9);
    }

    #[test]
    fn test_calculate_deviance_identity() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.1, 2.2, 2.9];
        let deviance = internal::calculate_deviance(y.view(), &mu, LinkFunction::Identity);
        // Sum of Squared Errors: (-0.1)^2 + (-0.2)^2 + (0.1)^2 = 0.01 + 0.04 + 0.01 = 0.06
        assert!((deviance - 0.06).abs() < 1e-9);
    }
}
