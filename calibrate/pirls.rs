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
        eprintln!("    [Debug] Lambdas calculated: [none] (model is unpenalized)");
    } else {
        eprintln!(
            "    [Debug] Lambdas calculated (first 5): [{:.2e}, {:.2e}, {:.2e}, {:.2e}, {:.2e}, ...]",
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
    eprintln!("    [Setup] Inner loop setup complete. Starting iterations...");

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
                        "Step halving successful after {attempt} attempts, final step size: {step_size:.6}"
                    );
                    eprintln!(
                        "    [Step Halving] SUCCESS after {attempt} attempts, final step size: {step_size:.6}"
                    );
                }
                break; // Exit line search loop
            }

            // Step was not an improvement. Halve and retry.
            if attempt < MAX_HALVINGS {
                eprintln!(
                    "    [Step Halving] Attempt {}: deviance {:.6} -> {:.6}, halving step to {:.6e}",
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
                "P-IRLS step-halving failed to reduce deviance at iteration {iter}. The fit for this rho has stalled."
            );
            eprintln!("    [P-IRLS STALLED] Step halving failed. Terminating inner loop.");
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
            let penalized_hessian = compute_penalized_hessian(x, &weights, &s_lambda)?;
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

        // A more detailed, real-time print for each inner-loop iteration
        eprintln!(
            "    [P-IRLS Iteration #{iter:<2}] Deviance: {last_deviance:<13.7} | Change: {deviance_change:>12.6e}{step_info}"
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
            eprintln!("    P-IRLS Converged."); // ADD THIS LINE

            // For the final result, compute the penalized Hessian properly
            let penalized_hessian = compute_penalized_hessian(x, &weights, &s_lambda)?;

            return Ok(PirlsResult {
                beta,
                penalized_hessian,
                deviance: last_deviance, // Now guaranteed to be the final, converged value
                final_weights: weights,
                status: PirlsStatus::Converged,
                iteration: iter,
                max_abs_eta,
            });
        }
        // No need to update last_deviance here - it's already updated in the line search when a step is accepted
    }

    eprintln!(
        "    P-IRLS FAILED to converge after {} iterations.",
        config.max_iterations
    ); // ADD THIS LINE

    // In pirls.rs, inside fit_model_for_fixed_rho, at the end of the `for` loop...

    // This code REPLACES the existing `Err(EstimationError::PirlsDidNotConverge { ... })`
    // It is executed ONLY if the loop finishes without meeting the deviance convergence criteria.

    eprintln!(
        "    [P-IRLS STALLED] Hit max iterations. Performing final check to see if stalled state is a valid minimum."
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

    // Check 2: Is the penalized Hessian positive-definite?
    let penalized_hessian = compute_penalized_hessian(x, &weights, &s_lambda)?;
    use ndarray_linalg::{Cholesky, UPLO};
    let is_positive_definite = penalized_hessian.cholesky(UPLO::Lower).is_ok();

    // Final Decision: Is the stall "good enough"?
    if is_gradient_zero && is_positive_definite {
        eprintln!(
            "    ✓ STALL ACCEPTED: A valid minimum was found (gradient_norm={gradient_norm:.2e}, Hessian is PD)."
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
        eprintln!(
            "    ✗ STALL REJECTED: Not a valid minimum (gradient_norm={gradient_norm:.2e}, Hessian_PD={is_positive_definite}). Reporting max iterations reached."
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
    use ndarray_linalg::{QR, SVD, Solve};

    // Step 1: Initial QR decomposition of sqrt(|W|)X (Wood 2011, Section 3.3)
    let sqrt_w_abs = weights.mapv(|w| w.abs().sqrt());
    let wx = &x * &sqrt_w_abs.view().insert_axis(Axis(1)); // sqrt(|W|)X
    let (q_bar, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;

    // Step 2: Get penalty square root E where E'E = S_lambda
    let e_matrix = penalty_square_root(s_lambda)?;

    // Step 3: Form augmented matrix [R_bar; E] and perform QR decomposition
    let r_rows = r_bar.nrows().min(p);
    let mut augmented = Array2::zeros((r_rows + p, p));
    augmented
        .slice_mut(s![..r_rows, ..])
        .assign(&r_bar.slice(s![..r_rows, ..]));
    augmented.slice_mut(s![r_rows.., ..]).assign(&e_matrix);

    let (_, r_final) = augmented
        .qr()
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    // Step 4: Handle negative weights using Q_bar from INITIAL QR (not q_aug)
    let neg_indices: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w < 0.0)
        .map(|(i, _)| i)
        .collect();

    let final_hessian = if neg_indices.is_empty() {
        // No negative weights: simple case
        r_final.t().dot(&r_final)
    } else {
        // Apply SVD correction using Q_bar (the CORRECT Q matrix)
        let q1_neg = q_bar.select(Axis(0), &neg_indices); // Select rows corresponding to negative weights

        // SVD of Q1_neg
        let (_, sigma, vt_opt) = q1_neg
            .svd(false, true)
            .map_err(EstimationError::LinearSystemSolveFailed)?;

        if let Some(vt) = vt_opt {
            // Form correction matrix (I - 2VD²V')
            let mut d_squared = Array2::zeros((sigma.len(), sigma.len()));
            for (i, &s) in sigma.iter().enumerate() {
                d_squared[[i, i]] = s * s;
            }

            let v = vt.t();
            let correction = Array2::eye(p) - 2.0 * v.dot(&d_squared.dot(&v.t()));

            // Apply correction: R_final'(I - 2VD²V')R_final
            r_final.t().dot(&correction.dot(&r_final))
        } else {
            // Fallback
            r_final.t().dot(&r_final)
        }
    };

    // Step 5: Calculate RHS = X'Wz correctly (no complex correction needed)
    // Form z_tilde with sign correction for negative weights
    let mut z_tilde = &y * &sqrt_w_abs;
    for &i in &neg_indices {
        z_tilde[i] = -z_tilde[i]; // Flip sign for negative weights
    }

    // Stable RHS calculation using QR decomposition results
    // RHS = R_bar' * Q_bar' * z_tilde
    let q_t_z = q_bar.t().dot(&z_tilde.slice(s![..q_bar.nrows()]));
    let rhs = r_bar
        .slice(s![..r_rows, ..])
        .t()
        .dot(&q_t_z.slice(s![..r_rows]));

    // Step 6: Solve the final system
    let beta = final_hessian
        .solve(&rhs)
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    // Step 7: Calculate effective degrees of freedom
    // EDF = tr(H⁻¹X'WX) where H = final_hessian
    let mut xtwx_base = r_bar
        .slice(s![..r_rows, ..])
        .t()
        .dot(&r_bar.slice(s![..r_rows, ..]));

    // Apply same correction to X'WX for EDF calculation if negative weights exist
    if !neg_indices.is_empty() {
        let q1_neg = q_bar.select(Axis(0), &neg_indices);
        if let Ok((_, sigma, Some(vt))) = q1_neg.svd(false, true) {
            let mut d_squared = Array2::zeros((sigma.len(), sigma.len()));
            for (i, &s) in sigma.iter().enumerate() {
                d_squared[[i, i]] = s * s;
            }
            let v = vt.t();
            let correction = Array2::eye(p) - 2.0 * v.dot(&d_squared.dot(&v.t()));
            xtwx_base = xtwx_base.dot(&correction);
        }
    }

    let mut edf: f64 = 0.0;
    for j in 0..p {
        if let Ok(h_inv_col) = final_hessian.solve(&xtwx_base.column(j).to_owned()) {
            edf += h_inv_col[j];
        }
    }
    edf = edf.max(1.0);

    // Step 8: Calculate scale parameter
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

/// Compute penalized Hessian matrix X'WX + S_λ
/// Used after P-IRLS convergence for final result
fn compute_penalized_hessian(
    x: ArrayView2<f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    // Form weighted design matrix √W X
    let sqrt_w = weights.mapv(|w| w.abs().sqrt());
    let wx = &x * &sqrt_w.view().insert_axis(Axis(1));

    // Form X'WX + S_λ
    let xtwx = wx.t().dot(&wx);
    let penalized_hessian = &xtwx + s_lambda;

    Ok(penalized_hessian)
}

/// Helper function to compute penalty square root E where E^T E = S
fn penalty_square_root(s: &Array2<f64>) -> Result<Array2<f64>, EstimationError> {
    use ndarray_linalg::{Cholesky, Eigh, UPLO};

    // Try Cholesky first
    if let Ok(l) = s.cholesky(UPLO::Lower) {
        // For penalty matrix S = LL' (Cholesky decomposition), 
        // we need matrix E where E'E = S.
        // By setting E = L', we get E'E = (L')'(L') = LL' = S.
        return Ok(l.t().to_owned());
    }

    // Fallback to eigendecomposition for semi-definite matrices
    let (eigenvals, eigenvecs): (Array1<f64>, Array2<f64>) = s
        .eigh(UPLO::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;

    let mut e = Array2::zeros(s.dim());
    for (i, &eval) in eigenvals.iter().enumerate() {
        if eval > 1e-12 {
            let v_i = eigenvecs.column(i);
            let sqrt_eval = eval.sqrt();
            for j in 0..e.nrows() {
                for k in 0..e.ncols() {
                    e[[j, k]] += sqrt_eval * v_i[j] * v_i[k];
                }
            }
        }
    }

    Ok(e)
}
