use crate::calibrate::construction::{ModelLayout, calculate_condition_number, construct_s_lambda};
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::{LinkFunction, ModelConfig};
use log;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Solve;

/// Holds the result of a converged P-IRLS inner loop for a fixed rho.
///
/// # Fields
///
/// * `beta`: The estimated coefficient vector.
/// * `penalized_hessian`: The penalized Hessian matrix at convergence (X'WX + S_λ).
/// * `deviance`: The final deviance value. Note that this means different things depending on the link function:
///    - For `LinkFunction::Identity` (Gaussian): This is the Residual Sum of Squares (RSS).
///    - For `LinkFunction::Logit` (Binomial): This is -2 * log-likelihood, the binomial deviance.
/// * `final_weights`: The final IRLS weights at convergence.
#[derive(Clone)]
pub struct PirlsResult {
    pub beta: Array1<f64>,
    pub penalized_hessian: Array2<f64>,
    pub deviance: f64,
    #[allow(dead_code)]
    pub final_weights: Array1<f64>,
}

/// The P-IRLS inner loop for a fixed set of smoothing parameters.
pub fn fit_model_for_fixed_rho(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    s_list: &[Array2<f64>],
    layout: &ModelLayout,
    config: &ModelConfig,
) -> Result<PirlsResult, EstimationError> {
    let mut lambdas = rho_vec.mapv(f64::exp);

    // Apply lambda floor to prevent numerical issues
    const LAMBDA_FLOOR: f64 = 1e-8;
    lambdas.mapv_inplace(|l| l.max(LAMBDA_FLOOR));

    let s_lambda = construct_s_lambda(&lambdas, s_list, layout);

    let mut beta = Array1::zeros(x.ncols());
    let mut last_deviance = f64::INFINITY;
    let mut last_change = f64::INFINITY;

    for iter in 1..=config.max_iterations {
        let eta = x.dot(&beta);
        let (mu, weights, z) = update_glm_vectors(y, &eta, config.link_function);

        // Check for NaN/inf values that could cause issues
        if !eta.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite eta values at iteration {}: {:?}", iter, eta);
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }
        if !mu.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite mu values at iteration {}: {:?}", iter, mu);
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }
        if !weights.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite weights at iteration {}: {:?}", iter, weights);
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }
        if !z.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite z values at iteration {}: {:?}", iter, z);
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }

        // For weighted regression with X'WX, we need to scale the rows of X by sqrt(weights)
        // to compute X'WX = (sqrt(W)X)'(sqrt(W)X)
        let sqrt_weights = weights.mapv(|w| w.sqrt());
        let x_weighted = &x * &sqrt_weights.view().insert_axis(Axis(1));
        let xtwx = x_weighted.t().dot(&x_weighted);
        let mut penalized_hessian = xtwx + &s_lambda;

        // Add numerical ridge for stability
        penalized_hessian.diag_mut().mapv_inplace(|d| d + 1e-9);

        // Check condition number before attempting to solve
        const MAX_CONDITION_NUMBER: f64 = 1e12;
        match calculate_condition_number(&penalized_hessian) {
            Ok(condition_number) => {
                if condition_number > MAX_CONDITION_NUMBER {
                    log::error!(
                        "Penalized Hessian is ill-conditioned at iteration {}: condition number = {:.2e}",
                        iter,
                        condition_number
                    );
                    return Err(EstimationError::ModelIsIllConditioned { condition_number });
                }
                // Log warning if condition number is high but not critical
                if condition_number > 1e8 {
                    log::warn!(
                        "Penalized Hessian has high condition number at iteration {}: {:.2e}",
                        iter,
                        condition_number
                    );
                }
            }
            Err(e) => {
                log::warn!(
                    "Failed to compute condition number at iteration {}: {:?}. Proceeding anyway.",
                    iter,
                    e
                );
            }
        }

        // Right-hand side of the equation: X'Wz = (sqrt(W)X)'(sqrt(W)z)
        let z_weighted = &sqrt_weights * &z;
        let rhs = x_weighted.t().dot(&z_weighted);
        beta = penalized_hessian
            .solve_into(rhs)
            .map_err(EstimationError::LinearSystemSolveFailed)?;

        if !beta.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite beta values at iteration {}: {:?}", iter, beta);
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }

        let deviance = calculate_deviance(y, &mu, config.link_function);
        if !deviance.is_finite() {
            log::error!("Non-finite deviance at iteration {}: {}", iter, deviance);
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }

        let deviance_change = if last_deviance.is_infinite() {
            // First iteration with infinite last_deviance
            if deviance.is_finite() {
                1e10 // Large but finite value to continue iteration
            } else {
                f64::INFINITY
            }
        } else {
            (last_deviance - deviance).abs()
        };

        if !deviance_change.is_finite() {
            log::error!(
                "Non-finite deviance_change at iteration {}: {} (last: {}, current: {})",
                iter,
                deviance_change,
                last_deviance,
                deviance
            );
            last_change = if deviance_change.is_nan() {
                f64::NAN
            } else {
                f64::INFINITY
            };
            break;
        }

        // Check for convergence using both absolute and relative criteria
        // The relative criterion is especially important for binary data with
        // perfect or near-perfect separation, where convergence slows dramatically
        let relative_change = if last_deviance.abs() > 1e-10 {
            deviance_change / last_deviance.abs()
        } else {
            // If last_deviance is very close to zero, just use the absolute change
            1.0
        };

        // Consider converged if either absolute or relative criterion is met
        let absolute_converged = deviance_change < config.convergence_tolerance;
        let relative_converged = relative_change < 1e-3; // 0.1% relative change is tight enough

        if absolute_converged || relative_converged {
            let final_eta = x.dot(&beta);
            let (final_mu, final_weights, _) =
                update_glm_vectors(y, &final_eta, config.link_function);
            // Same X'WX calculation method for the final step
            let final_sqrt_weights = final_weights.mapv(|w| w.sqrt());
            let final_x_weighted = &x * &final_sqrt_weights.view().insert_axis(Axis(1));
            let final_xtwx = final_x_weighted.t().dot(&final_x_weighted);
            let final_penalized_hessian = final_xtwx + &s_lambda;

            return Ok(PirlsResult {
                beta,
                penalized_hessian: final_penalized_hessian,
                deviance: calculate_deviance(y, &final_mu, config.link_function),
                final_weights,
            });
        }
        last_deviance = deviance;
        last_change = deviance_change;
    }

    Err(EstimationError::PirlsDidNotConverge {
        max_iterations: config.max_iterations,
        last_change,
    })
}

// Pseudo-inverse functionality is handled directly in compute_gradient

pub fn update_glm_vectors(
    y: ArrayView1<f64>,
    eta: &Array1<f64>,
    link: LinkFunction,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    const MIN_WEIGHT: f64 = 1e-6;
    const PROB_EPS: f64 = 1e-8; // Epsilon for clamping probabilities

    match link {
        LinkFunction::Logit => {
            // Clamp eta to prevent overflow in exp
            let eta_clamped = eta.mapv(|e| e.clamp(-700.0, 700.0));
            // Create mu and then clamp to prevent values exactly at 0 or 1
            let mut mu = eta_clamped.mapv(|e| 1.0 / (1.0 + (-e).exp()));
            // Clamp mu to prevent it from reaching exactly 0 or 1, which is crucial for
            // numerical stability of weights and deviance calculations
            mu.mapv_inplace(|v| v.clamp(PROB_EPS, 1.0 - PROB_EPS));
            let weights = (&mu * (1.0 - &mu)).mapv(|v| v.max(MIN_WEIGHT));

            // Prevent extreme values in working response z
            let residual = &y.view() - &mu;
            let z_adj = &residual / &weights;
            // Use a more reasonable clamping range (1e4 instead of 1e6) to prevent numerical instability
            // while still allowing for reasonable convergence steps
            let z_clamped = z_adj.mapv(|v| v.clamp(-1e4, 1e4));
            let z = &eta_clamped + &z_clamped;

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

pub fn calculate_deviance(y: ArrayView1<f64>, mu: &Array1<f64>, link: LinkFunction) -> f64 {
    const EPS: f64 = 1e-8; // Increased from 1e-9 for better numerical stability
    match link {
        LinkFunction::Logit => {
            let total_residual = ndarray::Zip::from(y).and(mu).fold(0.0, |acc, &yi, &mui| {
                let mui_c = mui.clamp(EPS, 1.0 - EPS);
                let term1 = if yi > EPS {
                    yi * (yi / mui_c).ln()
                } else {
                    0.0
                };
                let term2 = if yi < 1.0 - EPS {
                    (1.0 - yi) * ((1.0 - yi) / (1.0 - mui_c)).ln()
                } else {
                    0.0
                };
                acc + term1 + term2
            });
            2.0 * total_residual
        }
        LinkFunction::Identity => (&y.view() - mu).mapv(|v| v.powi(2)).sum(),
    }
}
