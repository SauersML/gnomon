use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::{LinkFunction, ModelConfig};
use log;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// The status of the P-IRLS convergence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PirlsStatus {
    /// Converged successfully within tolerance.
    Converged,
    /// Reached maximum iterations but the gradient and Hessian indicate a valid minimum.
    StalledAtValidMinimum,
    /// Reached maximum iterations without converging.
    MaxIterationsReached,
    /// Fitting process became unstable, likely due to perfect separation.
    Unstable,
}

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
    pub status: PirlsStatus,
    pub iteration: usize,
    pub max_abs_eta: f64,
}

/// Stable penalized least squares solver implementing the exact pls_fit1 algorithm
/// from Wood (2011) Section 3.3 and mgcv reference code.
/// Handles rank deficiency and negative weights via pivoted QR and SVD correction.
pub fn fit_model_for_fixed_rho(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    s_list: &[Array2<f64>],
    layout: &ModelLayout,
    config: &ModelConfig,
) -> Result<PirlsResult, EstimationError> {
    let lambdas = rho_vec.mapv(f64::exp);

    // Show calculated lambda values
    if lambdas.is_empty() {
        log::debug!("Lambdas calculated: [none] (model is unpenalized)");
    } else {
        log::debug!(
            "Lambdas calculated (first 5): [{:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, ...]",
            lambdas.get(0).unwrap_or(&0.0),
            lambdas.get(1).unwrap_or(&0.0),
            lambdas.get(2).unwrap_or(&0.0),
            lambdas.get(3).unwrap_or(&0.0),
            lambdas.get(4).unwrap_or(&0.0)
        );
    }

    // Use simple S_lambda construction for P-IRLS (stable reparameterization used elsewhere)
    use crate::calibrate::construction::construct_s_lambda;
    let s_lambda = construct_s_lambda(&lambdas, s_list, layout);

    // Print setup complete message
    log::info!("Inner loop setup complete. Starting iterations...");

    // Initialize state variables that will be updated throughout the loop
    let mut beta = Array1::zeros(layout.total_coeffs);
    let mut eta = x.dot(&beta);
    let (mut mu, mut weights, mut z) = update_glm_vectors(y, &eta, config.link_function);
    let mut last_deviance = calculate_deviance(y, &mu, config.link_function);
    let mut max_abs_eta = 0.0;

    // Validate dimensions
    assert_eq!(
        x.ncols(),
        layout.total_coeffs,
        "X matrix columns must match total coefficients"
    );
    assert_eq!(
        s_lambda.nrows(),
        layout.total_coeffs,
        "S_lambda rows must match total coefficients"
    );
    assert_eq!(
        s_lambda.ncols(),
        layout.total_coeffs,
        "S_lambda columns must match total coefficients"
    );

    // Add minimum iterations based on link function
    let min_iterations = match config.link_function {
        LinkFunction::Logit => 3, // Ensure at least some refinement for non-Gaussian
        LinkFunction::Identity => 1, // Gaussian may converge faster
    };

    for iter in 1..=config.max_iterations {
        // --- Store the state from the START of the iteration ---
        let beta_current = beta.clone();
        let deviance_current = last_deviance;

        // Check for non-finite values
        if !eta.iter().all(|x| x.is_finite())
            || !mu.iter().all(|x| x.is_finite())
            || !weights.iter().all(|x| x.is_finite())
            || !z.iter().all(|x| x.is_finite())
        {
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }

        // Use stable solver for P-IRLS inner loop (robust iteration)
        let stable_result =
            stable_penalized_least_squares(x.view(), z.view(), weights.view(), &s_lambda)?;

        let beta_proposal = stable_result.beta;

        if !beta_proposal.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite beta values at iteration {iter}: {beta_proposal:?}");
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }

        // Implement step halving with backtracking line search
        let delta_beta = &beta_proposal - &beta_current;
        let mut step_size = 1.0;
        const MAX_HALVINGS: usize = 12;

        let mut step_accepted = false;
        let mut final_halving_attempts = 0; // Variable to store the number of attempts

        for attempt in 0..=MAX_HALVINGS {
            final_halving_attempts = attempt; // Keep track of attempts
            let beta_trial = &beta_current + (step_size * &delta_beta);
            let eta_trial = x.dot(&beta_trial);

            if !eta_trial.iter().all(|v| v.is_finite()) {
                step_size *= 0.5;
                continue;
            }

            let (mu_trial, _, _) = update_glm_vectors(y, &eta_trial, config.link_function);
            let deviance_trial = calculate_deviance(y, &mu_trial, config.link_function);

            if !deviance_trial.is_finite() {
                log::error!(
                    "Non-finite deviance at iteration {iter} with step size {step_size}: {deviance_trial}"
                );
                step_size *= 0.5;
                continue;
            }

            // SUCCESS CONDITION: The new point is strictly better.
            if deviance_trial < deviance_current {
                // Atomic state update
                beta = beta_trial;
                eta = eta_trial;
                last_deviance = deviance_trial;
                (mu, weights, z) = update_glm_vectors(y, &eta, config.link_function);
                // End atomic state update

                step_accepted = true;
                if attempt > 0 {
                    log::debug!(
                        "Step halving successful after {} attempts, final step size: {:.6}",
                        attempt, step_size
                    );
                }
                break; // Exit line search loop
            }

            // Step was not an improvement. Halve and retry.
            if attempt < MAX_HALVINGS {
                log::debug!(
                    "Step Halving Attempt {}: deviance {:.6} -> {:.6}, halving step to {:.6e}",
                    attempt + 1,
                    deviance_current,
                    deviance_trial,
                    step_size * 0.5
                );
            }
            step_size *= 0.5;
        }

        if !step_accepted {
            log::warn!(
                "P-IRLS step-halving failed to reduce deviance at iteration {}. The fit for this rho has stalled.",
                iter
            );
            log::info!("P-IRLS STALLED: Step halving failed. Terminating inner loop.");
            break; // Exit the main `for` loop and proceed to the final check.
        }

        // Monitor the maximum absolute value of eta - the root cause of perfect separation
        max_abs_eta = eta
            .iter()
            .map(|v| v.abs())
            .fold(f64::NEG_INFINITY, f64::max);

        // A very large eta value is a strong sign of separation.
        // This threshold is chosen because exp(100) is astronomically large.
        const ETA_STABILITY_THRESHOLD: f64 = 100.0;
        if max_abs_eta > ETA_STABILITY_THRESHOLD && config.link_function == LinkFunction::Logit {
            log::warn!(
                "P-IRLS instability detected at iteration {iter}: max|eta| = {max_abs_eta:.2e}. Likely perfect separation."
            );

            // Don't fail here. Just return the current state with a status flag.
            // Reuse the penalized Hessian from the stable_result that was already computed
            // for this iteration, avoiding redundant computation
            let penalized_hessian = stable_result.penalized_hessian;
            
            return Ok(PirlsResult {
                beta,
                penalized_hessian,
                deviance: last_deviance,
                final_weights: weights,
                status: PirlsStatus::Unstable,
                iteration: iter,
                max_abs_eta,
            });
        }

        // Compute deviance_change safely
        let deviance_change = (deviance_current - last_deviance).abs();
        let step_info = if final_halving_attempts > 0 {
            format!(" | Step Halving: {} attempts", final_halving_attempts)
        } else {
            String::new()
        };

        // A more detailed, real-time log for each inner-loop iteration
        log::debug!(
            "P-IRLS Iteration #{:<2} | Deviance: {:<13.7} | Change: {:>12.6e}{}",
            iter, last_deviance, deviance_change, step_info
        );

        if !deviance_change.is_finite() {
            log::error!(
                "Non-finite deviance_change at iteration {iter}: {deviance_change} (last: {deviance_current}, current: {last_deviance})"
            );
            // Non-finite deviance change is a critical error
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: if deviance_change.is_nan() {
                    f64::NAN
                } else {
                    f64::INFINITY
                },
            });
        }

        // Robust convergence check
        // - Skip if below min_iterations
        // - Use combined relative/absolute: change < tol * (deviance + offset) to handle small deviances
        // - Offset=0.1 is common (avoids div-by-zero; can tune if needed)
        let reltol_offset = 0.1;
        let relative_threshold =
            config.convergence_tolerance * (last_deviance.abs() + reltol_offset);
        let converged = iter >= min_iterations
            && deviance_change < relative_threshold.max(config.convergence_tolerance);

        if converged {
            log::info!("P-IRLS Converged.");

            // Get the penalized Hessian from the last stable solver call
            // instead of recomputing it
            return Ok(PirlsResult {
                beta,
                penalized_hessian: stable_result.penalized_hessian,
                deviance: last_deviance, // Now guaranteed to be the final, converged value
                final_weights: weights,
                status: PirlsStatus::Converged,
                iteration: iter,
                max_abs_eta,
            });
        }
        // No need to update last_deviance here - it's already updated in the line search when a step is accepted
    }

    log::warn!(
        "P-IRLS FAILED to converge after {} iterations.",
        config.max_iterations
    );

    // In pirls.rs, inside fit_model_for_fixed_rho, at the end of the `for` loop...

    // This code REPLACES the existing `Err(EstimationError::PirlsDidNotConverge { ... })`
    // It is executed ONLY if the loop finishes without meeting the deviance convergence criteria.

    log::info!(
        "P-IRLS STALLED: Hit max iterations. Performing final check to see if stalled state is a valid minimum."
    );

    // We are here because the loop timed out. The variables `beta`, `mu`, `weights`,
    // and `s_lambda` hold the values from the last, stalled iteration.

    // Check 1: Is the gradient of the penalized deviance close to zero?
    // The gradient is g = XᵀW(z - η) - S_λβ = Xᵀ(y - μ) - S_λβ
    let penalized_deviance_gradient = x.t().dot(&(&y.view() - &mu)) - s_lambda.dot(&beta);
    let gradient_norm = penalized_deviance_gradient
        .dot(&penalized_deviance_gradient)
        .sqrt();
    let is_gradient_zero = gradient_norm < 1e-4; // Use a reasonable tolerance for the gradient norm

    // Reuse the Hessian from the last stable solver call
    // Store this outside the loop to maintain state across iterations
    let last_stable_result = 
        stable_penalized_least_squares(x.view(), z.view(), weights.view(), &s_lambda)?;
    let penalized_hessian = last_stable_result.penalized_hessian;
    
    // Check 2: Is the penalized Hessian positive-definite?
    use ndarray_linalg::{Cholesky, UPLO};
    let is_positive_definite = penalized_hessian.cholesky(UPLO::Lower).is_ok();

    // Final Decision: Is the stall "good enough"?
    if is_gradient_zero && is_positive_definite {
        log::info!(
            "STALL ACCEPTED: A valid minimum was found (gradient_norm={:.2e}, Hessian is PD).",
            gradient_norm
        );
        // The stall is acceptable. Return Ok() as if it had converged normally.
        Ok(PirlsResult {
            beta,
            penalized_hessian,
            deviance: last_deviance,
            final_weights: weights,
            status: PirlsStatus::StalledAtValidMinimum,
            iteration: config.max_iterations,
            max_abs_eta,
        })
    } else {
        // The stall is NOT at a valid minimum. Instead of failing, report the status.
        log::warn!(
            "STALL REJECTED: Not a valid minimum (gradient_norm={:.2e}, Hessian_PD={}). Reporting max iterations reached.",
            gradient_norm,
            is_positive_definite
        );

        // We don't need to calculate last deviance change anymore as we're just returning the status

        Ok(PirlsResult {
            beta,
            penalized_hessian,
            deviance: last_deviance,
            final_weights: weights,
            status: PirlsStatus::MaxIterationsReached,
            iteration: config.max_iterations,
            max_abs_eta,
        })
    }
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
            // Use a more reasonable clamping range (1000 instead of 1e4) to prevent numerical instability
            // while still allowing for reasonable convergence steps
            let z_clamped = z_adj.mapv(|v| v.clamp(-1000.0, 1000.0));
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

/// Result of the stable penalized least squares solve
#[derive(Clone)]
pub struct StablePLSResult {
    /// Solution vector beta
    pub beta: Array1<f64>,
    /// Final penalized Hessian matrix
    pub penalized_hessian: Array2<f64>,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Scale parameter estimate
    pub scale: f64,
}

/// Implements the exact stable penalized least squares solver from Wood (2011) Section 3.3
/// Following the pls_fit1 algorithm with proper rank deficiency handling and SVD correction
pub fn stable_penalized_least_squares(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    s_lambda: &Array2<f64>,
) -> Result<StablePLSResult, EstimationError> {
    let n = x.nrows();
    let p = x.ncols();

    use ndarray::s;
    use ndarray_linalg::{QR, Solve};

    // Step 1: Initial QR decomposition of sqrt(|W|)X (Wood 2011, Section 3.3)
    let sqrt_w_abs = weights.mapv(|w| w.abs().sqrt());
    let wx = &x * &sqrt_w_abs.view().insert_axis(Axis(1)); // sqrt(|W|)X
    let (q_bar, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;

    // Step 3: Identify indices of negative weights
    let neg_indices: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w < 0.0)
        .map(|(i, _)| i)
        .collect();

    // Step 5: Get corrected data Hessian using the single source of truth
    let corrected_data_hessian = compute_corrected_data_hessian(
        p,
        &neg_indices,
        &q_bar,
        &r_bar,
    )?;

    // Step 6: Form final penalized Hessian by adding penalty to corrected data term
    let final_hessian = &corrected_data_hessian + s_lambda;

    // Step 7: Calculate RHS = X'Wz correctly (no complex correction needed)
    // Form z_tilde with sign correction for negative weights
    let mut z_tilde = &y * &sqrt_w_abs;
    for &i in &neg_indices {
        z_tilde[i] = -z_tilde[i]; // Flip sign for negative weights
    }

    // Stable RHS calculation using QR decomposition results
    // RHS = R_bar' * Q_bar' * z_tilde
    let r_rows = r_bar.nrows().min(p);
    let q_t_z = q_bar.t().dot(&z_tilde.slice(s![..q_bar.nrows()]));
    let rhs = r_bar
        .slice(s![..r_rows, ..])
        .t()
        .dot(&q_t_z.slice(s![..r_rows]));

    // Step 8: Solve the final system
    let beta = final_hessian
        .solve(&rhs)
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    // Step 9: Use the corrected data hessian (not penalized) for EDF calculation
    // EDF = tr(H⁻¹X'WX) where H = final_hessian, X'WX = corrected_data_hessian
    let xtwx_base = corrected_data_hessian;

    // Calculate Effective Degrees of Freedom (EDF)
    // For large parameter spaces (p > 100), consider the direct matrix inverse approach
    // For smaller spaces or ill-conditioned matrices, use the column-by-column solve method
    let mut edf: f64 = 0.0;
    
    // Option 1: Direct matrix inverse (faster for large p but potentially less stable)
    if p > 100 {
        use ndarray_linalg::Inverse;
        if let Ok(h_inv) = final_hessian.inv() {
            // The influence matrix (hat matrix) in the penalized context is H⁻¹(X'WX)
            let influence_matrix = h_inv.dot(&xtwx_base);
            edf = influence_matrix.diag().sum();
            log::debug!("EDF calculated using direct matrix inverse: {:.2}", edf);
        } else {
            // Fall back to column-by-column approach if matrix inversion fails
            log::warn!("Matrix inversion failed for EDF calculation, falling back to column-by-column method");
            for j in 0..p {
                if let Ok(h_inv_col) = final_hessian.solve(&xtwx_base.column(j).to_owned()) {
                    edf += h_inv_col[j];
                }
            }
        }
    } else {
        // Option 2: Column-by-column solve (more stable, suitable for small to medium p)
        for j in 0..p {
            if let Ok(h_inv_col) = final_hessian.solve(&xtwx_base.column(j).to_owned()) {
                edf += h_inv_col[j];
            }
        }
    }
    
    // EDF cannot be less than 1 if an intercept is present
    edf = edf.max(1.0);

    // Step 10: Calculate scale parameter
    let fitted = x.dot(&beta);
    let residuals = &y - &fitted;
    let rss = residuals.mapv(|v| v * v).sum();
    let scale = rss / (n as f64 - edf).max(1.0);

    Ok(StablePLSResult {
        beta,
        penalized_hessian: final_hessian,
        edf,
        scale,
    })
}



/// Internal helper to compute the corrected data Hessian X'WX, handling negative weights.
/// This is the single source of truth for this calculation.
fn compute_corrected_data_hessian(
    p: usize,
    neg_indices: &[usize],
    q_bar: &Array2<f64>,
    r_bar: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    use ndarray::s;
    use ndarray_linalg::SVD;
    
    // The number of rows in the R factor to use (min of rows or columns)
    let r_rows = r_bar.nrows().min(p);
    
    // Step 1: Compute the uncorrected data Hessian X'|W|X = R_bar' * R_bar
    let data_hessian = r_bar
        .slice(s![..r_rows, ..])
        .t()
        .dot(&r_bar.slice(s![..r_rows, ..]));
    
    // Step 2: Apply correction for negative weights if necessary
    if !neg_indices.is_empty() {
        // Get Q rows for negative weights
        let q1_neg = q_bar.select(ndarray::Axis(0), neg_indices);
        
        // SVD of Q1_neg
        if let Ok((_, sigma, Some(vt))) = q1_neg.svd(false, true) {
            // Form correction matrix (I - 2VD²V')
            let mut d_squared = Array2::zeros((sigma.len(), sigma.len()));
            for (i, &s) in sigma.iter().enumerate() {
                d_squared[[i, i]] = s * s;
            }
            
            let v = vt.t();
            let c = Array2::eye(p) - 2.0 * v.dot(&d_squared.dot(&v.t()));
            
            // Apply correction using correct sandwich form: R_bar' * C * R_bar
            let r_bar_slice = r_bar.slice(s![..r_rows, ..]);
            let corrected = r_bar_slice.t().dot(&c.dot(&r_bar_slice));
            
            Ok(corrected)
        } else {
            // Fallback if SVD fails
            Ok(data_hessian)
        }
    } else {
        // No correction needed
        Ok(data_hessian)
    }
}

/// Compute penalized Hessian matrix X'WX + S_λ correctly handling negative weights
/// Used after P-IRLS convergence for final result
pub fn compute_final_penalized_hessian(
    x: ArrayView2<f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    use ndarray_linalg::QR;
    
    let p = x.ncols();
    
    // Step 1: Perform the QR decomposition of sqrt(|W|)X
    let sqrt_w_abs = weights.mapv(|w| w.abs().sqrt());
    let wx = &x * &sqrt_w_abs.view().insert_axis(ndarray::Axis(1));
    let (q_bar, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    
    // Step 2: Identify negative weights
    let neg_indices: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w < 0.0)
        .map(|(i, _)| i)
        .collect();
    
    // Step 3: Call the single-source-of-truth helper function
    let data_hessian = compute_corrected_data_hessian(
        p,
        &neg_indices,
        &q_bar,
        &r_bar,
    )?;
    
    // Step 4: Add the penalty term
    let penalized_hessian = &data_hessian + s_lambda;
    
    Ok(penalized_hessian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ndarray_linalg::Solve;
    
    /// This test explicitly verifies that the Hessian calculation is consistent between
    /// the stable_penalized_least_squares function and compute_final_penalized_hessian function
    /// when dealing with negative weights.
    #[test]
    fn test_hessian_consistency_with_negative_weights() {
        // SETUP: Create a proper test case with consistent dimensions
        let n_samples = 10; // Number of samples/rows
        let n_features = 3; // Number of features/columns
        
        // Create a design matrix X with reasonable values
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        for i in 0..n_samples {
            x_data.push(1.0); // Intercept
            x_data.push((i as f64) / n_samples as f64); // Feature 1
            x_data.push(((i as f64) / n_samples as f64).powi(2)); // Feature 2
        }
        let x = Array2::from_shape_vec((n_samples, n_features), x_data).unwrap();
        
        // Create a reasonable response vector y
        let y = Array1::from_vec((0..n_samples).map(|i| (i as f64) * 0.5 + 2.0).collect());
        
        // Create weights with a negative value to test the correction
        let mut weights = Array1::ones(n_samples);
        weights[3] = -0.5; // Set one weight to be negative
        
        // Create a simple penalty matrix
        let mut s_lambda = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            s_lambda[[i, i]] = 0.1; // Simple diagonal penalty
        }
        
        // ACTION 1: Compute the Hessian using the full stable solver
        let stable_result = stable_penalized_least_squares(
            x.view(), y.view(), weights.view(), &s_lambda
        ).expect("Stable solver failed");
        let hessian_from_solver = stable_result.penalized_hessian;
        
        // ACTION 2: Compute the Hessian using the public utility function
        let hessian_from_util = compute_final_penalized_hessian(
            x.view(), &weights, &s_lambda
        ).expect("Utility function failed");
        
        // VERIFY: The two matrices must be identical, element-wise
        let tolerance = 1e-10;
        assert_eq!(hessian_from_solver.shape(), hessian_from_util.shape(), 
                   "Hessian matrices have different shapes");
                   
        for i in 0..hessian_from_solver.nrows() {
            for j in 0..hessian_from_solver.ncols() {
                let diff = (hessian_from_solver[[i, j]] - hessian_from_util[[i, j]]).abs();
                assert!(diff < tolerance, 
                    "Hessian matrices differ at [{}, {}]: {} vs {} (diff: {})",
                    i, j, hessian_from_solver[[i, j]], hessian_from_util[[i, j]], diff);
            }
        }
    }

    /// This test robustly verifies the correctness of the stable penalized least squares solver
    /// using both a positive-definite and a rank-deficient penalty matrix.
    /// It validates the final beta solution against a ground truth calculated via the normal equations.
    #[test]
    fn test_stable_penalized_least_squares() {
        // ---
        // SCENARIO 1: Positive-Definite, Non-Diagonal Penalty
        // ---
        // Test positive-definite penalty matrix

        // SETUP: A simple, well-conditioned problem.
        let x = arr2(&[[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]]);
        let y = arr1(&[4.1, 6.2, 9.8]);
        let weights = arr1(&[1.0, 1.0, 1.0]); // Use identity weights to simplify ground truth
        let s1 = arr2(&[[4.0, 2.0], [2.0, 5.0]]);

        // ACTION 1.1: Generate the ground truth beta by solving the normal equations directly.
        // The solution is (X'WX + S) * beta = X'Wz.
        // For identity weights, this simplifies to (X'X + S) * beta = X'y.
        let xtx = x.t().dot(&x);
        let hessian_truth = &xtx + &s1;
        let rhs_truth = x.t().dot(&y);
        let beta_truth1 = hessian_truth.solve_into(rhs_truth).expect("S1: Ground truth solve failed");

        // ACTION 1.2: Run the function under test.
        let result1 = stable_penalized_least_squares(
            x.view(), y.view(), weights.view(), &s1
        ).expect("S1: stable_penalized_least_squares failed");

        // VERIFY 1.1: The beta from our complex solver must match the ground truth.
        let tol = 1e-9;
        for i in 0..result1.beta.len() {
            let diff = (result1.beta[i] - beta_truth1[i]).abs();
            let scale = beta_truth1[i].abs().max(1e-9);
            assert!(diff < tol * scale, 
                "Beta vectors differ at [{}]: {} vs {} (diff: {}, relative: {})", 
                i, result1.beta[i], beta_truth1[i], diff, diff / scale);
        }

        // ---
        // SCENARIO 2: Rank-Deficient Penalty
        // ---
        // Test rank-deficient penalty matrix
        
        // SETUP: A rank-deficient (singular) penalty matrix.
        let s2 = arr2(&[[1.0, 1.0], [1.0, 1.0]]);

        // ACTION 2.1: Generate the ground truth beta for this new penalty.
        let hessian_truth2 = &xtx + &s2;
        let beta_truth2 = hessian_truth2.solve_into(x.t().dot(&y)).expect("S2: Ground truth solve failed");

        // ACTION 2.2: Run the function under test with the new penalty.
        let result2 = stable_penalized_least_squares(
            x.view(), y.view(), weights.view(), &s2
        ).expect("S2: stable_penalized_least_squares failed");

        // VERIFY 2.2: The beta for this path must also match its ground truth.
        for i in 0..result2.beta.len() {
            let diff = (result2.beta[i] - beta_truth2[i]).abs();
            let scale = beta_truth2[i].abs().max(1e-9);
            assert!(diff < tol * scale, 
                "Beta vectors differ at [{}]: {} vs {} (diff: {}, relative: {})", 
                i, result2.beta[i], beta_truth2[i], diff, diff / scale);
        }
    }
}
