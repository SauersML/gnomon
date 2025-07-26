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

/// P-IRLS solver that uses pre-computed reparameterization results
/// This function is called once for each set of smoothing parameters
pub fn fit_model_for_fixed_rho(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    eb: &Array2<f64>,              // Pre-computed balanced penalty square root
    rs_transformed: &[Array2<f64>], // Pre-computed transformed penalty square roots
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
    
    // Print setup complete message
    log::info!("Inner loop setup complete. Starting iterations...");

    // Build the penalty matrix S_lambda from transformed penalty square roots (once per rho)
    let mut s_lambda = Array2::zeros((layout.total_coeffs, layout.total_coeffs));
    for (k, &lambda) in lambdas.iter().enumerate() {
        if k < rs_transformed.len() {
            let s_k = rs_transformed[k].dot(&rs_transformed[k].t());
            s_lambda.scaled_add(lambda, &s_k);
        }
    }

    // Initialize state variables that will be updated throughout the loop
    let mut beta = Array1::zeros(layout.total_coeffs);
    let mut eta = x.dot(&beta);
    let (mut mu, mut weights, mut z) = update_glm_vectors(y, &eta, config.link_function);
    let mut last_deviance = calculate_deviance(y, &mu, config.link_function);
    let mut max_abs_eta = 0.0;
    let mut last_iter = 0;
    let mut broke_early = false;

    // Validate dimensions
    assert_eq!(
        x.ncols(),
        layout.total_coeffs,
        "X matrix columns must match total coefficients"
    );

    // Add minimum iterations based on link function
    let min_iterations = match config.link_function {
        LinkFunction::Logit => 3, // Ensure at least some refinement for non-Gaussian
        LinkFunction::Identity => 1, // Gaussian may converge faster
    };

    for iter in 1..=config.max_iterations {
        last_iter = iter; // Update on every iteration
        // --- Store the state from the START of the iteration ---
        let beta_current = beta.clone();
        let deviance_current = last_deviance;

        // Calculate the penalty for the current beta
        let penalty_current = beta_current.dot(&s_lambda.dot(&beta_current));
        // This is the true objective function value at the start of the iteration
        let penalized_deviance_current = deviance_current + penalty_current;

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
        let stable_result = stable_penalized_least_squares(
            x.view(),
            z.view(),
            weights.view(),
            eb,
            rs_transformed,
            lambdas.as_slice().unwrap(),
        )?;

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

            // Calculate the penalty for the trial beta
            let penalty_trial = beta_trial.dot(&s_lambda.dot(&beta_trial));
            // This is the true objective function value for the trial step
            let penalized_deviance_trial = deviance_trial + penalty_trial;

            // SUCCESS CONDITION: The new point is strictly better in terms of penalized deviance.
            if penalized_deviance_trial < penalized_deviance_current {
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
                        attempt,
                        step_size
                    );
                }
                break; // Exit line search loop
            }

            // Step was not an improvement. Halve and retry.
            if attempt < MAX_HALVINGS {
                log::debug!(
                    "Step Halving Attempt {}: penalized deviance {:.6} -> {:.6}, halving step to {:.6e}",
                    attempt + 1,
                    penalized_deviance_current,
                    penalized_deviance_trial,
                    step_size * 0.5
                );
            }
            step_size *= 0.5;
        }

        if !step_accepted {
            log::warn!(
                "P-IRLS step-halving failed to reduce penalized deviance at iteration {}. The fit for this rho has stalled.",
                iter
            );
            log::info!("P-IRLS STALLED: Step halving failed. Terminating inner loop.");
            last_iter = iter;
            broke_early = true;
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
            // Recompute the penalized Hessian with the final weights for consistency.
            let penalized_hessian = compute_final_penalized_hessian(x.view(), &weights, &s_lambda)?;

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

        // Calculate the current penalized deviance
        let penalty_new = beta.dot(&s_lambda.dot(&beta));
        let penalized_deviance_new = last_deviance + penalty_new;

        // Compute penalized deviance change safely
        let deviance_change = (penalized_deviance_current - penalized_deviance_new).abs();
        let step_info = if final_halving_attempts > 0 {
            format!(" | Step Halving: {} attempts", final_halving_attempts)
        } else {
            String::new()
        };

        // A more detailed, real-time log for each inner-loop iteration
        log::debug!(
            "P-IRLS Iteration #{:<2} | Penalized Deviance: {:<13.7} | Change: {:>12.6e}{}",
            iter,
            penalized_deviance_new,
            deviance_change,
            step_info
        );

        if !deviance_change.is_finite() {
            log::error!(
                "Non-finite penalized deviance change at iteration {iter}: {deviance_change} (last: {penalized_deviance_current}, current: {penalized_deviance_new})"
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
            config.convergence_tolerance * (penalized_deviance_new.abs() + reltol_offset);
        let converged = iter >= min_iterations
            && deviance_change < relative_threshold.max(config.convergence_tolerance);

        if converged {
            log::info!("P-IRLS Converged.");

            // Recompute the penalized Hessian with the final weights for consistency
            let penalized_hessian = compute_final_penalized_hessian(x.view(), &weights, &s_lambda)?;

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

    log::warn!("P-IRLS FAILED to converge after {} iterations.", last_iter);

    // This code is executed ONLY if the loop finishes without meeting the deviance convergence criteria.

    if broke_early {
        log::info!(
            "P-IRLS STALLED: Step halving failed at iteration {}. Performing final check to see if stalled state is a valid minimum.",
            last_iter
        );
    } else {
        log::info!(
            "P-IRLS STALLED: Hit max iterations ({}). Performing final check to see if stalled state is a valid minimum.",
            config.max_iterations
        );
    }

    // We are here because the loop timed out. The variables `beta`, `mu`, `weights`,
    // and `s_lambda` hold the values from the last, stalled iteration.

    // Check 1: Is the gradient of the penalized deviance close to zero?
    // The gradient is g = XᵀW(z - η) - S_λβ = Xᵀ(y - μ) - S_λβ
    let penalized_deviance_gradient = x.t().dot(&(&y.view() - &mu)) - s_lambda.dot(&beta);
    let gradient_norm = penalized_deviance_gradient
        .dot(&penalized_deviance_gradient)
        .sqrt();
    let is_gradient_zero = gradient_norm < 1e-4; // Use a reasonable tolerance for the gradient norm

    // Compute the penalized Hessian for the final check
    // We need to compute the Hessian using the final weights to ensure consistency
    // with the returned beta and weights. This is more efficient than running the full
    // stable_penalized_least_squares solver again.
    log::debug!("Computing final Hessian for stalled state using current weights.");
    let penalized_hessian = compute_final_penalized_hessian(x.view(), &weights, &s_lambda)?;

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
            iteration: last_iter,
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
            iteration: last_iter,
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
                // More numerically stable formulation: use difference of logs instead of log of ratio
                let term1 = if yi > EPS {
                    yi * (yi.ln() - mui_c.ln())
                } else {
                    0.0
                };
                // More numerically stable formulation: use difference of logs instead of log of ratio
                let term2 = if yi < 1.0 - EPS {
                    (1.0 - yi) * ((1.0 - yi).ln() - (1.0 - mui_c).ln())
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

/// New rank-truncating stable penalized least squares solver following mgcv's pls_fit1 algorithm
/// This solver performs rank detection and truncation at every call to ensure numerical stability
pub fn stable_penalized_least_squares(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    eb: &Array2<f64>,          // balanced penalty square root from reparameterization
    rs_transformed: &[Array2<f64>], // transformed penalty square roots
    lambdas: &[f64],           // current lambda values
) -> Result<StablePLSResult, EstimationError> {
    let p = x.ncols();

    use ndarray::s;
    use ndarray_linalg::QR;

    // Step A: Form the weighted data matrix sqrt(|W|)X and compute its QR decomposition
    let sqrt_w_abs = weights.mapv(|w| w.abs().sqrt());
    let wx = &x * &sqrt_w_abs.view().insert_axis(Axis(1)); // sqrt(|W|)X
    let (q_bar, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;

    // Identify negative weights for SVD correction
    let neg_indices: Vec<usize> = weights
        .iter()
        .enumerate()
        .filter(|(_, w)| **w < 0.0)
        .map(|(i, _)| i)
        .collect();
    
    // Step B: Construct the rank-detection matrix by stacking scaled R_bar and scaled Eb
    let r_rows = r_bar.nrows().min(p);
    let r_bar_active = r_bar.slice(s![..r_rows, ..]);
    
    // Calculate scaling factors for balancing data and penalty contributions
    let r_norm = frobenius_norm(&r_bar_active);
    let e_norm = frobenius_norm(&eb.view());
    
    // Prevent division by zero
    let r_scale = if r_norm > 1e-15 { 1.0 / r_norm } else { 1.0 };
    let e_scale = if e_norm > 1e-15 { 1.0 / e_norm } else { 1.0 };
    
    // Stack [R_bar_scaled; Eb_scaled] for rank detection
    let combined_rows = r_rows + eb.nrows();
    let mut rank_matrix = Array2::zeros((combined_rows, p));
    
    // Fill R_bar part (scaled)
    for i in 0..r_rows {
        for j in 0..p {
            rank_matrix[[i, j]] = r_bar_active[[i, j]] * r_scale;
        }
    }
    
    // Fill Eb part (scaled)
    for i in 0..eb.nrows() {
        for j in 0..p {
            rank_matrix[[r_rows + i, j]] = eb[[i, j]] * e_scale;
        }
    }

    // Step C: Perform pivoted QR decomposition on the rank-detection matrix
    let (_, r_combined, pivot) = pivoted_qr(&rank_matrix)?;

    // Step D: Determine numerical rank from diagonal of R and create dropped_cols list
    let r_diag = r_combined.diag();
    let max_diag = r_diag.iter().fold(0.0f64, |acc, &x| acc.max(x.abs()));
    // Use a stricter tolerance for better numerical stability in rank-deficient cases
    let threshold = 1e-10 * max_diag.max(1.0);
    let mut rank = 0;
    
    for &diag_val in r_diag.iter() {
        if diag_val.abs() > threshold {
            rank += 1;
        } else {
            break;
        }
    }
    
    // Create list of dropped columns (unpivoted indices)
    let mut computed_dropped_cols = Vec::new();
    for i in rank..p {
        if i < pivot.len() {
            computed_dropped_cols.push(pivot[i]);
        }
    }

    // Step E: Solve the system using the unified augmented QR solver
    let (beta_full, penalized_hessian_full) = solve_with_augmented_qr(
        x,
        y,
        weights,
        &pivot,
        rank,
        rs_transformed,
        lambdas,
        &q_bar,
        &r_bar,
        &neg_indices,
    )?;

    // Calculate EDF and scale using the full-dimensional results
    let edf = calculate_edf(&penalized_hessian_full, x, weights)?;
    let scale = calculate_scale(&beta_full, x, y, weights, edf);

    Ok(StablePLSResult {
        beta: beta_full,
        penalized_hessian: penalized_hessian_full,
        edf,
        scale,
    })
}

/// Helper function to compute Frobenius norm of a matrix
fn frobenius_norm(matrix: &ArrayView2<f64>) -> f64 {
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Perform pivoted QR decomposition using column-norm based pivoting
fn pivoted_qr(matrix: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>, Vec<usize>), EstimationError> {
    use ndarray_linalg::QR;
    
    let m = matrix.nrows();
    let n = matrix.ncols();
    
    // Initialize pivot vector and working matrix
    let mut pivot: Vec<usize> = (0..n).collect();
    let mut work_matrix = matrix.clone();
    
    // Compute initial column norms
    let mut col_norms: Vec<f64> = (0..n)
        .map(|j| work_matrix.column(j).dot(&work_matrix.column(j)).sqrt())
        .collect();
    
    // Apply column pivoting based on norms
    for k in 0..n.min(m) {
        // Find column with maximum norm from k onwards
        let (max_idx, _) = col_norms.iter()
            .enumerate()
            .skip(k)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        
        // Swap columns k and max_idx if needed
        if max_idx != k {
            // Swap in pivot vector
            pivot.swap(k, max_idx);
            
            // Swap columns in matrix
            let col_k = work_matrix.column(k).to_owned();
            let col_max = work_matrix.column(max_idx).to_owned();
            work_matrix.column_mut(k).assign(&col_max);
            work_matrix.column_mut(max_idx).assign(&col_k);
            
            // Swap column norms
            col_norms.swap(k, max_idx);
        }
        
        // Update column norms for remaining columns
        if k < m - 1 {
            let _factor = if col_norms[k] > 0.0 {
                work_matrix[[k, k]] / col_norms[k]
            } else {
                0.0
            };
            
            for j in (k + 1)..n {
                if col_norms[j] > 0.0 {
                    let temp = work_matrix[[k, j]] / col_norms[j];
                    col_norms[j] *= (1.0 - temp * temp).max(0.0).sqrt();
                }
            }
        }
    }
    
    // Perform QR decomposition on pivoted matrix
    let (q, r) = work_matrix.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    
    Ok((q, r, pivot))
}

/// Helper to build the combined square root of the penalty matrix
fn build_sqrt_s_lambda(
    rs_transformed: &[Array2<f64>],
    lambdas: &[f64],
    p: usize,
) -> Result<Array2<f64>, EstimationError> {
    let scaled_roots: Vec<_> = rs_transformed
        .iter()
        .zip(lambdas)
        .map(|(rs_k, &lambda)| rs_k * lambda.sqrt())
        .collect();

    if scaled_roots.is_empty() {
        return Ok(Array2::zeros((p, 0)));
    }

    ndarray::concatenate(
        Axis(1),
        &scaled_roots.iter().map(|m| m.view()).collect::<Vec<_>>(),
    ).map_err(|_| EstimationError::LayoutError("Failed to concatenate penalty roots".into()))
}

/// Unified solver using augmented QR decomposition
/// 
/// This implements the numerically stable approach from mgcv's C function `pls_fit1`,
/// which avoids forming the normal equations directly. Instead, it:
/// 1. Works with QR factors directly
/// 2. Truncates both R from QR(sqrt(W)X) and the penalty square root E
/// 3. Forms an augmented system by stacking these truncated matrices
/// 4. Performs another QR decomposition and solves via back-substitution
/// 5. Properly handles negative weights by correcting the RHS vector
///
/// This approach works for both full-rank and rank-deficient cases.
#[allow(clippy::too_many_arguments)]
fn solve_with_augmented_qr(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    pivot: &[usize],
    rank: usize,
    rs_transformed: &[Array2<f64>],
    lambdas: &[f64],
    q_bar: &Array2<f64>,
    r_bar: &Array2<f64>,
    neg_indices: &[usize],
) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
    use ndarray::s;
    use ndarray_linalg::{QR, SVD, Solve};

    let p = x.ncols();
    let r_rows = r_bar.nrows().min(p);

    // 1. Create truncated R factor (R1_trunc)
    //    Select the first `rank` columns from R1 according to the pivot
    let mut r1_trunc = Array2::zeros((r_rows, rank));
    for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
        r1_trunc.column_mut(j_trunc).assign(&r_bar.column(j_orig));
    }

    // 2. Create the truncated penalty square root (E_trunc)
    //    This involves building the full sqrt(S_lambda) and then selecting columns.
    let sqrt_s_lambda = build_sqrt_s_lambda(rs_transformed, lambdas, p)?;
    let mut e_trunc_t = Array2::zeros((sqrt_s_lambda.ncols(), rank));
    for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
        e_trunc_t.column_mut(j_trunc).assign(&sqrt_s_lambda.row(j_orig));
    }

    // 3. Form the augmented matrix [R1_trunc; E_trunc]
    let nr = r_rows + e_trunc_t.nrows();
    let mut r_augmented = Array2::zeros((nr, rank));
    r_augmented.slice_mut(s![..r_rows, ..]).assign(&r1_trunc);
    r_augmented.slice_mut(s![r_rows.., ..]).assign(&e_trunc_t.t()); // Transpose E_trunc_t to stack

    // 4. Final QR decomposition on the augmented matrix
    let (q_aug, r_aug) = r_augmented.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    let r_aug_final = r_aug.slice(s![..rank, ..]); // Ensure it's rank x rank

    // 5. Form the Right-Hand-Side (RHS)
    let sqrt_w_z = &weights.mapv(|w| w.abs().sqrt()) * &y;
    let rhs_transformed = q_bar.t().dot(&sqrt_w_z);
    
    // Truncate and pad with zeros
    let mut rhs_aug = Array1::zeros(nr);
    let n_rhs = rhs_transformed.len().min(r_rows);
    rhs_aug.slice_mut(s![..n_rhs]).assign(&rhs_transformed.slice(s![..n_rhs]));
    
    // Apply Q' from the augmented QR
    let mut final_rhs = q_aug.t().dot(&rhs_aug);

    // 6. Correct RHS for negative weights (mgcv's approach)
    if !neg_indices.is_empty() {
        // 6a. Get the rows of Q1 (`q_bar`) for negative weights, truncated to the correct rank
        let q1_neg = q_bar.select(Axis(0), neg_indices);
        let mut q1_neg_trunc = Array2::zeros((neg_indices.len(), rank));
        for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
            q1_neg_trunc.column_mut(j_trunc).assign(&q1_neg.column(j_orig));
        }

        // 6b. SVD of the truncated Q1_neg
        let (_, sigma, vt) = q1_neg_trunc.svd(true, true)
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        let vt = vt.unwrap();

        // 6c. Form the diagonal correction matrix (I - 2D²)^-1
        let mut d_inv = Array1::zeros(rank);
        for i in 0..rank {
            let val = if i < sigma.len() { 1.0 - 2.0 * sigma[i].powi(2) } else { 1.0 };
            // Invert, handling near-zero values
            d_inv[i] = if val.abs() > 1e-12 { 1.0 / val } else { 0.0 };
        }

        // 6d. Apply the transformation T = V * D_inv * V' to final_rhs
        let temp_vec = vt.dot(&final_rhs.slice(s![..rank]));
        let temp_vec_corrected = &d_inv * &temp_vec;
        final_rhs.slice_mut(s![..rank]).assign(&vt.t().dot(&temp_vec_corrected));
    }

    // 7. Solve for truncated beta using back-substitution
    let final_rhs_slice = final_rhs.slice(s![..rank]);
    let beta_trunc = r_aug_final.solve(&final_rhs_slice)
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    // 8. Re-inflate beta to full size
    let mut beta_full = Array1::zeros(p);
    for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
        beta_full[j_orig] = beta_trunc[j_trunc];
    }
    
    // 9. Reconstruct the full-sized penalized Hessian
    let h_trunc = r_aug_final.t().dot(&r_aug_final);
    let mut h_full = Array2::zeros((p, p));
    for r_trunc in 0..rank {
        for c_trunc in 0..rank {
            let r_orig = pivot[r_trunc];
            let c_orig = pivot[c_trunc];
            h_full[[r_orig, c_orig]] = h_trunc[[r_trunc, c_trunc]];
            if r_orig != c_orig {
                h_full[[c_orig, r_orig]] = h_trunc[[r_trunc, c_trunc]]; // Symmetrize
            }
        }
    }
    
    // Ensure perfect symmetry even with floating point noise
    let h_full_sym = (&h_full + &h_full.t()) * 0.5;

    Ok((beta_full, h_full_sym))
}

/// Calculate effective degrees of freedom
fn calculate_edf(
    penalized_hessian: &Array2<f64>,
    x: ArrayView2<f64>,
    weights: ArrayView1<f64>,
) -> Result<f64, EstimationError> {
    use ndarray_linalg::Solve;
    
    let p = x.ncols();
    let sqrt_w_abs = weights.mapv(|w| w.abs().sqrt());
    let wx = &x * &sqrt_w_abs.view().insert_axis(Axis(1));
    let xtwx = wx.t().dot(&wx);
    
    let mut edf = 0.0;
    for j in 0..p {
        if let Ok(h_inv_col) = penalized_hessian.solve(&xtwx.column(j).to_owned()) {
            edf += h_inv_col[j];
        }
    }
    
    Ok(edf.max(1.0))
}

/// Calculate scale parameter
fn calculate_scale(
    beta: &Array1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    edf: f64,
) -> f64 {
    let fitted = x.dot(beta);
    let residuals = &y - &fitted;
    let weighted_rss: f64 = weights
        .iter()
        .zip(residuals.iter())
        .map(|(&w, &r)| w * r * r)
        .sum();
    let n = x.nrows() as f64;
    weighted_rss / (n - edf).max(1.0)
}

/// Compute penalized Hessian matrix X'WX + S_λ correctly handling negative weights
/// Used after P-IRLS convergence for final result
pub fn compute_final_penalized_hessian(
    x: ArrayView2<f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>, // This is S_lambda = Σλ_k * S_k
) -> Result<Array2<f64>, EstimationError> {
    use ndarray::s;
    use ndarray_linalg::{QR, UPLO, Eigh};

    let p = x.ncols();

    // Step 1: Perform the QR decomposition of sqrt(|W|)X to get R_bar
    let sqrt_w_abs = weights.mapv(|w| w.abs().sqrt());
    let wx = &x * &sqrt_w_abs.view().insert_axis(ndarray::Axis(1));
    let (_, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    let r_rows = r_bar.nrows().min(p);
    let r1_full = r_bar.slice(s![..r_rows, ..]);

    // Step 2: Get the square root of the penalty matrix, E
    // We need to use eigendecomposition as S_lambda is not necessarily from a single root
    let (eigenvalues, eigenvectors) = s_lambda.eigh(UPLO::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;
    
    let tolerance = 1e-12;
    let rank_s = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();
    
    let mut e = Array2::zeros((p, rank_s));
    let mut col_idx = 0;
    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let scaled_eigvec = eigenvectors.column(i).mapv(|v| v * eigenval.sqrt());
            e.column_mut(col_idx).assign(&scaled_eigvec);
            col_idx += 1;
        }
    }

    // Step 3: Form the augmented matrix [R1; E_t]
    // Note: Here we use the full, un-truncated matrices because we are just computing
    // the Hessian for a given model, not performing rank detection.
    let e_t = e.t();
    let nr = r_rows + e_t.nrows();
    let mut augmented_matrix = Array2::zeros((nr, p));
    augmented_matrix.slice_mut(s![..r_rows, ..]).assign(&r1_full);
    augmented_matrix.slice_mut(s![r_rows.., ..]).assign(&e_t);

    // Step 4: Perform QR decomposition on the augmented matrix
    let (_, r_aug) = augmented_matrix.qr().map_err(EstimationError::LinearSystemSolveFailed)?;

    // Step 5: The penalized Hessian is R_aug' * R_aug
    let h_final = r_aug.t().dot(&r_aug);

    Ok(h_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use crate::calibrate::construction::{compute_penalty_square_roots, stable_reparameterization, ModelLayout};

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
        let penalty_val = 0.1;
        for i in 0..n_features {
            s_lambda[[i, i]] = penalty_val; // Simple diagonal penalty
        }

        // Create dummy eb and rs_transformed for the test
        let s_list = vec![s_lambda.clone()];
        let rs_list = compute_penalty_square_roots(&s_list).unwrap();
        let layout = ModelLayout {
            intercept_col: 0,
            pgs_main_cols: 0..0,
            penalty_map: vec![],
            total_coeffs: n_features,
            num_penalties: 1,
            num_pgs_interaction_bases: 0,
        };
        let lambdas = vec![1.0];
        let reparam_result = stable_reparameterization(&rs_list, &lambdas, &layout).unwrap();
        
        // ACTION 1: Compute the Hessian using the full stable solver
        let stable_result =
            stable_penalized_least_squares(x.view(), y.view(), weights.view(), &reparam_result.eb, &reparam_result.rs_transformed, &[1.0])
                .expect("Stable solver failed");
        let hessian_from_solver = stable_result.penalized_hessian;

        // ACTION 2: Compute the Hessian using the public utility function
        let hessian_from_util = compute_final_penalized_hessian(x.view(), &weights, &s_lambda)
            .expect("Utility function failed");

        // VERIFY: The two matrices must be identical, element-wise
        let tolerance = 1e-10;
        assert_eq!(
            hessian_from_solver.shape(),
            hessian_from_util.shape(),
            "Hessian matrices have different shapes"
        );

        for i in 0..hessian_from_solver.nrows() {
            for j in 0..hessian_from_solver.ncols() {
                let diff = (hessian_from_solver[[i, j]] - hessian_from_util[[i, j]]).abs();
                assert!(
                    diff < tolerance,
                    "Hessian matrices differ at [{}, {}]: {} vs {} (diff: {})",
                    i,
                    j,
                    hessian_from_solver[[i, j]],
                    hessian_from_util[[i, j]],
                    diff
                );
            }
        }
    }

    /// This test verifies the correctness of the stable penalized least squares solver
    /// by checking that it produces optimal solutions for well-conditioned problems.
    #[test]
    fn test_stable_penalized_least_squares() {
        // SETUP: A simple, well-conditioned problem.
        let x = arr2(&[[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]]);
        let y = arr1(&[4.1, 6.2, 9.8]);
        let weights = arr1(&[1.0, 1.0, 1.0]); // Use identity weights to simplify test
        let s1 = arr2(&[[4.0, 2.0], [2.0, 5.0]]);

        // Compute penalty square roots and perform reparameterization
        let s_list = vec![s1.clone()];
        let rs_list = compute_penalty_square_roots(&s_list)
            .expect("Failed to compute penalty square roots");
        
        // Create a minimal layout for the test
        let layout = ModelLayout {
            intercept_col: 0,
            pgs_main_cols: 0..0,
            penalty_map: vec![],
            total_coeffs: 2,
            num_penalties: 1,
            num_pgs_interaction_bases: 0,
        };
        
        let lambdas = vec![1.0];
        let reparam_result = stable_reparameterization(&rs_list, &lambdas, &layout)
            .expect("Reparameterization failed");
        
        // Run our solver
        let result = stable_penalized_least_squares(
            x.view(), 
            y.view(), 
            weights.view(), 
            &reparam_result.eb, 
            &reparam_result.rs_transformed, 
            &lambdas
        ).expect("stable_penalized_least_squares failed");

        // Verify the solution by checking if it minimizes the objective function
        let fitted_values = x.dot(&result.beta);
        let residuals = &y - &fitted_values;
        let rss = residuals.dot(&residuals);
        let penalty = result.beta.dot(&s1.dot(&result.beta));
        let objective = rss + penalty;
        
        // The objective should be finite and reasonable
        assert!(
            objective.is_finite() && objective > 0.0,
            "Objective function value is invalid: {}",
            objective
        );
        
        // Also verify that small perturbations to beta increase the objective
        let delta = 1e-4;
        let mut perturbed_better = false;
        
        // Try a few perturbations
        for i in 0..result.beta.len() {
            let mut beta_plus = result.beta.clone();
            beta_plus[i] += delta;
            
            let fitted_plus = x.dot(&beta_plus);
            let residuals_plus = &y - &fitted_plus;
            let rss_plus = residuals_plus.dot(&residuals_plus);
            let penalty_plus = beta_plus.dot(&s1.dot(&beta_plus));
            let objective_plus = rss_plus + penalty_plus;
            
            // If any perturbation decreases the objective, our solution wasn't optimal
            perturbed_better = perturbed_better || (objective_plus < objective);
        }
        
        assert!(
            !perturbed_better,
            "Found a better solution through perturbation"
        );
    }
    
}