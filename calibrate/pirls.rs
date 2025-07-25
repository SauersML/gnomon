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
    let mut dropped_cols = Vec::new();
    for i in rank..p {
        if i < pivot.len() {
            dropped_cols.push(pivot[i]);
        }
    }

    // Step E: Solve the truncated system if rank < p
    let (beta_full, penalized_hessian_full) = if rank < p {
        solve_truncated_system(
            x,
            y,
            weights,
            &pivot,
            rank,
            &dropped_cols,
            rs_transformed,
            lambdas,
            &q_bar,
            &r_bar,
            &neg_indices,
        )?
    } else {
        // Full rank case - use traditional approach
        solve_full_rank_system(
            x,
            y,
            weights,
            rs_transformed,
            lambdas,
            &q_bar,
            &r_bar,
            &neg_indices,
        )?
    };

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

/// Solve the truncated system for rank-deficient case
#[allow(clippy::too_many_arguments)]
fn solve_truncated_system(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    pivot: &[usize],
    rank: usize,
    _dropped_cols: &[usize],
    rs_transformed: &[Array2<f64>],
    lambdas: &[f64],
    q_bar: &Array2<f64>,
    r_bar: &Array2<f64>,
    neg_indices: &[usize],
) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
    use ndarray_linalg::Solve;
    
    let p = x.ncols();
    
    // Step 1: Build the full p x p penalty matrix S_lambda
    let mut s_lambda = Array2::zeros((p, p));
    for (k, &lambda) in lambdas.iter().enumerate() {
        if k < rs_transformed.len() {
            let s_k = rs_transformed[k].dot(&rs_transformed[k].t());
            s_lambda.scaled_add(lambda, &s_k);
        }
    }
    
    // Step 2: Extract rank x rank sub-matrix of S_lambda using kept indices
    let mut s_lambda_trunc = Array2::zeros((rank, rank));
    for (i_trunc, &i_orig) in pivot.iter().take(rank).enumerate() {
        for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
            s_lambda_trunc[[i_trunc, j_trunc]] = s_lambda[[i_orig, j_orig]];
        }
    }
    
    // Step 3: Form truncated penalized Hessian using the corrected data Hessian
    let data_hessian_full = compute_corrected_data_hessian(p, neg_indices, q_bar, r_bar)?;
    
    let mut data_hessian_trunc = Array2::zeros((rank, rank));
    for (i_trunc, &i_orig) in pivot.iter().take(rank).enumerate() {
        for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
            data_hessian_trunc[[i_trunc, j_trunc]] = data_hessian_full[[i_orig, j_orig]];
        }
    }
    
    let h_trunc = data_hessian_trunc + s_lambda_trunc;
    
    // Step 4: Form truncated RHS (X'Wz)
    let w_z = &weights * &y;
    let rhs_full = x.t().dot(&w_z);
    
    let mut rhs_trunc = Array1::zeros(rank);
    for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
        rhs_trunc[j_trunc] = rhs_full[j_orig];
    }
    
    // Step 5: Solve truncated system
    let beta_trunc = h_trunc.solve(&rhs_trunc).map_err(EstimationError::LinearSystemSolveFailed)?;
    
    // Step 6: Re-inflate to full dimensions
    let mut beta_full = Array1::zeros(p);
    for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
        beta_full[j_orig] = beta_trunc[j_trunc];
    }
    
    // Step 7: The full penalized Hessian is the corrected data Hessian plus full penalty
    let h_full = data_hessian_full + s_lambda;
    
    Ok((beta_full, h_full))
}

/// Solve the full-rank system using traditional approach
fn solve_full_rank_system(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    rs_transformed: &[Array2<f64>],
    lambdas: &[f64],
    q_bar: &Array2<f64>,
    r_bar: &Array2<f64>,
    neg_indices: &[usize],
) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
    use ndarray_linalg::Solve;
    
    let p = x.ncols();
    
    // Build full penalty matrix
    let mut s_lambda = Array2::zeros((p, p));
    for (k, &lambda) in lambdas.iter().enumerate() {
        if k < rs_transformed.len() {
            let s_k = rs_transformed[k].dot(&rs_transformed[k].t());
            s_lambda.scaled_add(lambda, &s_k);
        }
    }
    
    // Form penalized Hessian with SVD correction for negative weights
    let data_hessian = compute_corrected_data_hessian(p, neg_indices, q_bar, r_bar)?;
    let h_full = data_hessian + s_lambda;
    
    // Form RHS (X'Wz)
    let w_z = &weights * &y;
    let rhs = x.t().dot(&w_z);
    
    // Solve system
    let beta = h_full.solve(&rhs).map_err(EstimationError::LinearSystemSolveFailed)?;
    
    Ok((beta, h_full))
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

// SVD correction functions removed - to be implemented later

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
            let v = vt.t(); // v is n x n (e.g., 3x3)
            let k = sigma.len(); // Number of singular values (e.g., 1)

            // Robustly compute V*D by scaling the first k columns of V
            // by the singular values in sigma.
            let v_k = v.slice(s![.., ..k]);
            // Use broadcasting to scale the columns of v_k by sigma
            let vd = &v_k * &sigma;

            // The correction term is dimensionally correct
            let correction_term = 2.0 * vd.dot(&vd.t());

            let c = Array2::eye(v.nrows()) - &correction_term;

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
    let data_hessian = compute_corrected_data_hessian(p, &neg_indices, &q_bar, &r_bar)?;

    // Step 4: Add the penalty term
    let penalized_hessian = &data_hessian + s_lambda;

    Ok(penalized_hessian)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};
    use ndarray_linalg::Solve;
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
        let beta_truth1 = hessian_truth
            .solve_into(rhs_truth)
            .expect("S1: Ground truth solve failed");

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
        
        // ACTION 1.2: Run the function under test.
        let result1 = stable_penalized_least_squares(
            x.view(), 
            y.view(), 
            weights.view(), 
            &reparam_result.eb, 
            &reparam_result.rs_transformed, 
            &lambdas
        ).expect("S1: stable_penalized_least_squares failed");

        // VERIFY 1.1: The beta from our complex solver must match the ground truth.
        let tol = 1e-9;
        for i in 0..result1.beta.len() {
            let diff = (result1.beta[i] - beta_truth1[i]).abs();
            let scale = beta_truth1[i].abs().max(1e-9);
            assert!(
                diff < tol * scale,
                "Beta vectors differ at [{}]: {} vs {} (diff: {}, relative: {})",
                i,
                result1.beta[i],
                beta_truth1[i],
                diff,
                diff / scale
            );
        }

        // ---
        // SCENARIO 2: Rank-Deficient Penalty
        // ---
        // Test rank-deficient penalty matrix

        // SETUP: A rank-deficient (singular) penalty matrix.
        let s2 = arr2(&[[1.0, 1.0], [1.0, 1.0]]);

        // ACTION 2.1: Generate the ground truth beta for this new penalty.
        let hessian_truth2 = &xtx + &s2;
        let beta_truth2 = hessian_truth2
            .solve_into(x.t().dot(&y))
            .expect("S2: Ground truth solve failed");

        // Compute penalty square roots and perform reparameterization for s2
        let s_list2 = vec![s2.clone()];
        let rs_list2 = compute_penalty_square_roots(&s_list2)
            .expect("Failed to compute penalty square roots for s2");
        
        let reparam_result2 = stable_reparameterization(&rs_list2, &lambdas, &layout)
            .expect("Reparameterization failed for s2");

        // ACTION 2.2: Run the function under test with the new penalty.
        let result2 = stable_penalized_least_squares(
            x.view(), 
            y.view(), 
            weights.view(), 
            &reparam_result2.eb, 
            &reparam_result2.rs_transformed, 
            &lambdas
        ).expect("S2: stable_penalized_least_squares failed");

        // VERIFY 2.2: For rank-deficient penalty, verify the solution satisfies the normal equations
        // rather than expecting exact match with simple solver
        
        // Check that beta satisfies the normal equations: (X'X + S) * beta = X'y
        let residual = &hessian_truth2.dot(&result2.beta) - &x.t().dot(&y);
        let residual_norm = residual.dot(&residual).sqrt();
        let rhs_norm = x.t().dot(&y).dot(&x.t().dot(&y)).sqrt();
        let relative_residual = residual_norm / rhs_norm.max(1e-10);
        
        // For rank-deficient systems, we allow a larger tolerance
        let rank_deficient_tol = 1e-6;
        assert!(
            relative_residual < rank_deficient_tol,
            "Scenario 2: Solution does not satisfy normal equations. Relative residual: {:.2e}",
            relative_residual
        );
        
        // Also verify that the solution minimizes the objective function
        let objective_solver = y.dot(&y) - 2.0 * y.dot(&x.dot(&result2.beta)) 
            + result2.beta.dot(&hessian_truth2.dot(&result2.beta));
        let objective_truth = y.dot(&y) - 2.0 * y.dot(&x.dot(&beta_truth2)) 
            + beta_truth2.dot(&hessian_truth2.dot(&beta_truth2));
        
        // The stable solver should produce a solution at least as good as the simple solver
        assert!(
            objective_solver <= objective_truth + 1e-6,
            "Scenario 2: Stable solver objective ({}) is worse than simple solver ({})",
            objective_solver,
            objective_truth
        );
    }
}
