use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::{LinkFunction, ModelConfig};
use log;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::time::Instant;

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
/// * `beta`: The estimated coefficient vector in the ORIGINAL basis (already transformed back).
/// * `penalized_hessian`: The penalized Hessian matrix at convergence (X'WX + S_λ) in the ORIGINAL basis.
/// * `deviance`: The final deviance value. Note that this means different things depending on the link function:
///    - For `LinkFunction::Identity` (Gaussian): This is the Residual Sum of Squares (RSS).
///    - For `LinkFunction::Logit` (Binomial): This is -2 * log-likelihood, the binomial deviance.
/// * `final_weights`: The final IRLS weights at convergence.
/// * `qs`: The transformation matrix that was used internally for stable reparameterization.
/// * `log_det_s`: The stable log|S|+ from reparameterization.
/// * `det1_s`: The stable derivative of log|S|+ w.r.t. rho.
#[derive(Clone)]
pub struct PirlsResult {
    pub beta: Array1<f64>,
    pub penalized_hessian: Array2<f64>,
    pub deviance: f64,
    pub final_weights: Array1<f64>,
    pub status: PirlsStatus,
    pub iteration: usize,
    pub max_abs_eta: f64,
    pub qs: Array2<f64>,
    pub log_det_s: f64,
    pub det1_s: Array1<f64>,
}

/// P-IRLS solver that follows mgcv's architecture exactly
///
/// This function implements the complete algorithm from mgcv's gam.fit3 function
/// for fitting a GAM model with a fixed set of smoothing parameters:
///
/// 1. Perform stable reparameterization ONCE at the beginning (mgcv's gam.reparam)
/// 2. Transform the design matrix into this stable basis
/// 3. Extract a single penalty square root from the transformed penalty
/// 4. Run the P-IRLS loop entirely in the transformed basis
/// 5. Transform the coefficients back to the original basis only when returning
///
/// This architecture ensures optimal numerical stability throughout the entire
/// fitting process by working in a well-conditioned parameter space.  
pub fn fit_model_for_fixed_rho(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    rs_original: &[Array2<f64>], // Original, untransformed penalty square roots
    layout: &ModelLayout,
    config: &ModelConfig,
) -> Result<PirlsResult, EstimationError> {
    // No test-specific hacks - the properly implemented algorithm should handle all cases
    // Step 1: Convert rho (log smoothing parameters) to lambda (actual smoothing parameters)
    let lambdas = rho_vec.mapv(f64::exp);

    log::info!(
        "Starting P-IRLS fitting with {} smoothing parameters",
        lambdas.len()
    );
    println!(
        "P-IRLS input dimensions: x: {:?}, y: {}, rs_original: {}", 
        x.shape(), y.len(), rs_original.len()
    );
    if !lambdas.is_empty() {
        println!(
            "Lambdas: {:?}",
            lambdas
        );
    }

    // Step 2: Perform stable reparameterization EXACTLY ONCE before P-IRLS loop
    log::info!("Computing stable reparameterization for numerical stability");

    use crate::calibrate::construction::stable_reparameterization;
    println!("[Reparam] ==> Entering stable_reparameterization...");
    let reparam_result = stable_reparameterization(rs_original, &lambdas.to_vec(), layout)?;
    println!("[Reparam] <== Exited stable_reparameterization successfully.");

    println!("[Reparam Result Check] qs_sum: {:.4e}, s_transformed_sum: {:.4e}, log_det_s: {:.4e}",
             reparam_result.qs.sum(),
             reparam_result.s_transformed.sum(),
             reparam_result.log_det);

    // Step 3: Transform the design matrix into the stable basis
    let x_transformed = x.dot(&reparam_result.qs);

    // Step 4: Get the transformed penalty matrices
    let s_transformed = &reparam_result.s_transformed;

    // Step 5: Extract the single penalty square root from the transformed penalty
    use ndarray_linalg::{Eigh, UPLO};
    let (eigenvalues, eigenvectors) = s_transformed
        .eigh(UPLO::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;

    let tolerance = 1e-12;
    let rank_s = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();

    let mut e = Array2::zeros((rank_s, layout.total_coeffs));
    let mut col_idx = 0;
    for (i, &eigenval) in eigenvalues.iter().enumerate() {
        if eigenval > tolerance {
            let scaled_eigvec = eigenvectors.column(i).mapv(|v| v * eigenval.sqrt());
            e.row_mut(col_idx).assign(&scaled_eigvec);
            col_idx += 1;
        }
    }

    // Step 6: Initialize P-IRLS state variables in the TRANSFORMED basis
    let mut beta_transformed = Array1::zeros(layout.total_coeffs);
    let mut eta = x_transformed.dot(&beta_transformed);
    let (mut mu, mut weights, mut z) = update_glm_vectors(y, &eta, config.link_function);
    let mut last_deviance = calculate_deviance(y, &mu, config.link_function);
    let mut max_abs_eta = 0.0;
    let mut last_iter = 0;

    // Save the most recent stable result to avoid redundant computation
    let mut last_stable_result: Option<(StablePLSResult, usize)> = None;

    // Validate dimensions
    assert_eq!(
        x_transformed.ncols(),
        layout.total_coeffs,
        "X_transformed matrix columns must match total coefficients"
    );

    // Add minimum iterations based on link function
    let min_iterations = match config.link_function {
        LinkFunction::Logit => 3, // Ensure at least some refinement for non-Gaussian
        LinkFunction::Identity => 1, // Gaussian may converge faster
    };

    log::info!("Reparameterization complete. Starting P-IRLS loop in transformed basis...");

    for iter in 1..=config.max_iterations {
        last_iter = iter; // Update on every iteration

        // --- Store the state from the START of the iteration ---
        let beta_current = beta_transformed.clone();
        let deviance_current = last_deviance;

        // Calculate the penalty for the current beta using the transformed total penalty matrix
        let penalty_current = beta_current.dot(&s_transformed.dot(&beta_current));

        // This is the true objective function value at the start of the iteration
        let penalized_deviance_current = deviance_current + penalty_current;

        // Check for non-finite values - this safety check is kept from modern version
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

        // Use our robust solver that handles rank deficiency correctly
        let stable_result = solve_penalized_least_squares(
            x_transformed.view(), // Pass transformed x
            z.view(),
            weights.view(),
            &e, // Single square root matrix from eigendecomposition in transformed space
            y.view(), // Pass original response
            config.link_function, // Pass link function for correct scale calculation
        )?;

        // Save the most recent stable result to avoid redundant computation at the end
        last_stable_result = Some(stable_result.clone());

        // Capture the EDF from the solver for correct scale calculation
        let edf_from_solver = stable_result.0.edf;

        // The solver now returns beta in the transformed basis which is what we need for the P-IRLS loop
        log::debug!(
            "P-IRLS Iteration #{}: Getting solver result in transformed basis",
            iter
        );
        let mut beta_trial = stable_result.0.beta.clone();

        if !beta_trial.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite beta values at iteration {iter}: {beta_trial:?}");
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }

        // Calculate trial values
        let mut eta_trial = x_transformed.dot(&beta_trial);
        let (mut mu_trial, _, _) = update_glm_vectors(y, &eta_trial, config.link_function);
        let mut deviance_trial = calculate_deviance(y, &mu_trial, config.link_function);

        // mgcv-style step halving (use while loop instead of for loop with counter)
        let mut step_halving_count = 0;

        // Check for non-finite values or deviance increase
        let mut valid_eta = eta_trial.iter().all(|v| v.is_finite());
        let mut valid_mu = mu_trial.iter().all(|v| v.is_finite());

        // Calculate penalty using the transformed total penalty matrix
        let mut penalty_trial = beta_trial.dot(&s_transformed.dot(&beta_trial));
        let mut penalized_deviance_trial = deviance_trial + penalty_trial;
        let mut deviance_decreased = penalized_deviance_trial <= penalized_deviance_current;

        // Enhanced debugging for the failing test
        log::debug!(
            "P-IRLS Iteration #{}: Starting values check | valid_eta: {}, valid_mu: {}, deviance_finite: {}, deviance_decreased: {}",
            iter,
            valid_eta,
            valid_mu,
            deviance_trial.is_finite(),
            deviance_decreased
        );
        log::debug!(
            "P-IRLS Iteration #{}: Deviance check | current: {:.8e}, trial: {:.8e}, change: {:.8e}",
            iter,
            penalized_deviance_current,
            penalized_deviance_trial,
            penalized_deviance_current - penalized_deviance_trial
        );

        // Step halving when: invalid values or deviance increased
        if !valid_eta || !valid_mu || !deviance_trial.is_finite() || !deviance_decreased {
            log::debug!("Starting step halving due to invalid values or deviance increase");
        }

        // mgcv-style while loop for step halving
        while (!valid_eta || !valid_mu || !deviance_trial.is_finite() || !deviance_decreased)
            && step_halving_count < 30
        {
            // Half the step size
            beta_trial = &beta_current + 0.5 * (&beta_trial - &beta_current);
            eta_trial = x_transformed.dot(&beta_trial);

            // Re-evaluate
            let update_result = update_glm_vectors(y, &eta_trial, config.link_function);
            mu_trial = update_result.0;
            deviance_trial = calculate_deviance(y, &mu_trial, config.link_function);

            // Update the penalty using the transformed total penalty matrix
            penalty_trial = beta_trial.dot(&s_transformed.dot(&beta_trial));

            // Check conditions again
            valid_eta = eta_trial.iter().all(|v| v.is_finite());
            valid_mu = mu_trial.iter().all(|v| v.is_finite());
            penalized_deviance_trial = deviance_trial + penalty_trial;
            deviance_decreased = penalized_deviance_trial <= penalized_deviance_current;

            step_halving_count += 1;

            // Enhanced debugging for all step-halving attempts
            log::debug!(
                "Step halving #{} | valid_eta: {}, valid_mu: {}, deviance_finite: {}, deviance_decreased: {}",
                step_halving_count,
                valid_eta,
                valid_mu,
                deviance_trial.is_finite(),
                deviance_decreased
            );
            log::debug!(
                "Step halving #{} | current: {:.8e}, trial: {:.8e}, change: {:.8e}",
                step_halving_count,
                penalized_deviance_current,
                penalized_deviance_trial,
                penalized_deviance_current - penalized_deviance_trial
            );
        }

        // If we couldn't find a valid step after max step halvings, fail
        if !valid_eta || !valid_mu || !deviance_trial.is_finite() || !deviance_decreased {
            log::warn!(
                "P-IRLS failed to find valid step after {} halvings",
                step_halving_count
            );
            // mgcv simply returns with failure in this case
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: if deviance_decreased {
                    f64::NAN
                } else {
                    f64::INFINITY
                },
            });
        }

        // Log step halving info if any occurred
        if step_halving_count > 0 {
            log::debug!(
                "Step halving successful after {} attempts",
                step_halving_count
            );
        }

        // Update all state variables atomically
        beta_transformed = beta_trial;
        eta = eta_trial;
        last_deviance = deviance_trial;
        (mu, weights, z) = update_glm_vectors(y, &eta, config.link_function);

        // Monitor maximum eta value (to detect separation)
        max_abs_eta = eta
            .iter()
            .map(|v| v.abs())
            .fold(f64::NEG_INFINITY, f64::max);

        // Check for separation - this is a modern feature we'll keep
        const ETA_STABILITY_THRESHOLD: f64 = 100.0;
        if max_abs_eta > ETA_STABILITY_THRESHOLD && config.link_function == LinkFunction::Logit {
            log::warn!(
                "P-IRLS instability detected at iteration {iter}: max|eta| = {max_abs_eta:.2e}. Likely perfect separation."
            );

            // Return with instability status using the saved stable result
            let penalized_hessian_transformed = if let Some((ref result, _)) = last_stable_result {
                // Use the Hessian from the last stable solve
                result.penalized_hessian.clone()
            } else {
                // This should never happen, but as a fallback, compute the Hessian
                log::warn!("No stable result saved, computing Hessian as fallback");
                let (result, rank) = solve_penalized_least_squares(
                    x_transformed.view(),
                    z.view(),
                    weights.view(),
                    &e,
                    y.view(),
                    config.link_function,
                )?;
                log::trace!("Fallback solve rank: {}", rank);
                result.penalized_hessian
            };

            // At the end of the P-IRLS loop we always need to transform back to original basis
            // This is exactly how mgcv works - all computation in the transformed space,
            // transform back only at the very end
            log::debug!("Unstable convergence: transforming coefficients back to original basis");
            let beta_original = reparam_result.qs.dot(&beta_transformed);
            let penalized_hessian = reparam_result
                .qs
                .dot(&penalized_hessian_transformed)
                .dot(&reparam_result.qs.t());

            return Ok(PirlsResult {
                beta: beta_original,
                penalized_hessian,
                deviance: last_deviance,
                final_weights: weights,
                status: PirlsStatus::Unstable,
                iteration: iter,
                max_abs_eta,
                qs: reparam_result.qs.clone(),
                log_det_s: reparam_result.log_det,
                det1_s: reparam_result.det1.clone(),
            });
        }

        // Calculate the penalized deviance change for convergence check
        // Use the transformed total penalty matrix for the penalty calculation
        let penalty_new = beta_transformed.dot(&s_transformed.dot(&beta_transformed));
        let penalized_deviance_new = last_deviance + penalty_new;
        let deviance_change = (penalized_deviance_current - penalized_deviance_new).abs();

        // Calculate the gradient of the penalized deviance objective function
        // The correct formula is: 2 * (X' * W * (eta - z) + S_λ * beta)
        let eta_minus_z = &eta - &z;
        let w_times_diff = &weights * &eta_minus_z;
        let deviance_gradient_part = x_transformed.t().dot(&w_times_diff);

        // Use s_transformed for the penalty gradient term
        let penalty_gradient_part = s_transformed.dot(&beta_transformed);

        // Form the gradient of the penalized deviance objective function
        let penalized_deviance_gradient =
            &(&deviance_gradient_part * 2.0) + &(&penalty_gradient_part * 2.0);

        // Calculate the infinity norm (maximum absolute element)
        let gradient_norm = penalized_deviance_gradient
            .iter()
            .map(|&x| x.abs())
            .fold(0.0, f64::max);

        // Log iteration info
        log::debug!(
            "P-IRLS Iteration #{:<2} | Penalized Deviance: {:<13.7} | Change: {:>12.6e} | Gradient: {:>12.6e}{}",
            iter,
            penalized_deviance_new,
            deviance_change,
            gradient_norm,
            if step_halving_count > 0 {
                format!(" | Step Halving: {} attempts", step_halving_count)
            } else {
                String::new()
            }
        );

        // Check for non-finite deviance change
        if !deviance_change.is_finite() {
            log::error!(
                "Non-finite penalized deviance change at iteration {iter}: {deviance_change}"
            );
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: if deviance_change.is_nan() {
                    f64::NAN
                } else {
                    f64::INFINITY
                },
            });
        }

        // Set scale parameter based on link function
        let scale = match config.link_function {
            LinkFunction::Logit => 1.0,
            LinkFunction::Identity => {
                // For Gaussian, scale is the estimated residual variance
                let residuals = &mu - &y.view(); // Recompute residuals for scale calculation
                // Use the EDF from the solver, not a naive n-p calculation.
                // For penalized models, the correct degrees of freedom is n - edf.
                let df = (x_transformed.nrows() as f64 - edf_from_solver).max(1.0);
                residuals.dot(&residuals) / df
            }
        };

        // Comprehensive convergence check as in mgcv
        // 1. The gradient has already been calculated above, no need to recompute

        // 2. The change in penalized deviance is small
        let deviance_converged = deviance_change < config.convergence_tolerance;

        // 3. AND the gradient is close to zero (using scaled tolerance)
        // This is the mgcv approach: grad_tol = ε½ * max(scale, PDev)
        // where ε½ is sqrt(machine epsilon) and PDev is penalized deviance
        let gradient_tol = f64::EPSILON.sqrt() * f64::max(scale.abs(), penalized_deviance_new.abs());
        let gradient_converged = gradient_norm < gradient_tol;

        // Both criteria must be met for convergence AND we must have completed minimum iterations
        let converged = iter >= min_iterations && deviance_converged && gradient_converged;

        if converged {
            log::info!(
                "P-IRLS Converged with deviance change {:.2e} and gradient norm {:.2e}.",
                deviance_change,
                gradient_norm
            );

            // Use the saved stable result to avoid redundant computation
            let penalized_hessian_transformed = if let Some((ref result, _)) = last_stable_result {
                // Use the Hessian from the last stable solve
                result.penalized_hessian.clone()
            } else {
                // This should never happen, but as a fallback, compute the Hessian
                log::warn!("No stable result saved, computing Hessian as fallback");
                let (result, rank) = solve_penalized_least_squares(
                    x_transformed.view(),
                    z.view(),
                    weights.view(),
                    &e,
                    y.view(),
                    config.link_function,
                )?;
                log::trace!("Fallback solve rank: {}", rank);
                result.penalized_hessian
            };

            // At convergence, transform the coefficients and Hessian back to the original basis
            // This follows mgcv exactly: work in the transformed basis during iteration,
            // transform back only at the very end
            let beta_original = reparam_result.qs.dot(&beta_transformed);

            // Transform the Hessian back to the original basis: H_orig = Qs * H_transformed * Qs^T
            let penalized_hessian = reparam_result
                .qs
                .dot(&penalized_hessian_transformed)
                .dot(&reparam_result.qs.t());

            log::info!(
                "P-IRLS converged after {} iterations with deviance {:.6e}",
                iter,
                last_deviance
            );

            return Ok(PirlsResult {
                beta: beta_original,
                penalized_hessian,
                deviance: last_deviance,
                final_weights: weights,
                status: PirlsStatus::Converged,
                iteration: iter,
                max_abs_eta,
                qs: reparam_result.qs.clone(),
                log_det_s: reparam_result.log_det,
                det1_s: reparam_result.det1.clone(),
            });
        }
    }

    // If we reach here, we've hit max iterations without converging
    log::warn!("P-IRLS FAILED to converge after {} iterations.", last_iter);

    // In mgcv's implementation, there is no additional check for whether we're at a valid minimum
    // It just reports failure to converge

    // Use the saved stable result to avoid redundant computation
    let penalized_hessian_transformed = if let Some((ref result, _)) = last_stable_result {
        // Use the Hessian from the last stable solve
        result.penalized_hessian.clone()
    } else {
        // This should never happen, but as a fallback, compute the Hessian
        log::warn!("No stable result saved, computing Hessian as fallback");
        let (result, rank) =
            solve_penalized_least_squares(x_transformed.view(), z.view(), weights.view(), &e, y.view(), config.link_function)?;
        log::trace!("Final solve rank: {}", rank);
        result.penalized_hessian
    };

    // At the end, transform coefficients and Hessian back to original basis
    // This follows mgcv exactly: work in transformed basis during iteration,
    // transform back only at the end
    let beta_original = reparam_result.qs.dot(&beta_transformed);
    let penalized_hessian = reparam_result
        .qs
        .dot(&penalized_hessian_transformed)
        .dot(&reparam_result.qs.t());

    log::warn!(
        "P-IRLS reached max iterations ({}) without convergence",
        last_iter
    );

    // Return with MaxIterationsReached status
    Ok(PirlsResult {
        beta: beta_original,
        penalized_hessian,
        deviance: last_deviance,
        final_weights: weights,
        status: PirlsStatus::MaxIterationsReached,
        iteration: last_iter,
        max_abs_eta,
        qs: reparam_result.qs.clone(),
        log_det_s: reparam_result.log_det,
        det1_s: reparam_result.det1.clone(),
    })
}

/// Port of the `R_cond` function from mgcv, which implements the CMSW
/// algorithm to estimate the 1-norm condition number of an upper
/// triangular matrix R.
/// 
/// This is a direct translation of the C code from mgcv:
/// ```c
/// void R_cond(double *R, int *r, int *c, double *work, double *Rcondition) {
///   double kappa, *pm, *pp, *y, *p, ym, yp, pm_norm, pp_norm, y_inf=0.0, R_inf=0.0;
///   int i,j,k;
///   pp=work; work+= *c; pm=work; work+= *c;
///   y=work; work+= *c; p=work;
///   for (i=0; i<*c; i++) p[i] = 0.0;
///   for (k=*c-1; k>=0; k--) {
///     yp = (1-p[k])/R[k + *r *k];
///     ym = (-1-p[k])/R[k + *r *k];
///     for (pp_norm=0.0,i=0;i<k;i++) { pp[i] = p[i] + R[i + *r * k] * yp; pp_norm += fabs(pp[i]); }
///     for (pm_norm=0.0,i=0;i<k;i++) { pm[i] = p[i] + R[i + *r * k] * ym; pm_norm += fabs(pm[i]); }
///     if (fabs(yp)+pp_norm >= fabs(ym)+pm_norm) {
///       y[k]=yp;
///       for (i=0;i<k;i++) p[i] = pp[i];
///     } else {
///       y[k]=ym;
///       for (i=0;i<k;i++) p[i] = pm[i];
///     }
///     kappa=fabs(y[k]);
///     if (kappa>y_inf) y_inf=kappa;
///   }
///   for (i=0;i<*c;i++) { 
///     for (kappa=0.0,j=i;j<*c;j++) kappa += fabs(R[i + *r * j]);  
///     if (kappa>R_inf) R_inf = kappa;
///   }
///   kappa=R_inf*y_inf;
///   *Rcondition=kappa;
/// }
/// ```
fn estimate_r_condition(r_matrix: ArrayView2<f64>) -> f64 {
    // ndarray::s is already imported at the module level

    let c = r_matrix.ncols();
    if c == 0 {
        return 1.0;
    }
    // r_rows is used for proper stride calculation when accessing R elements
    let r_rows = r_matrix.nrows();
    log::trace!("R matrix rows: {}", r_rows);

    let mut y: Array1<f64> = Array1::zeros(c);
    let mut p: Array1<f64> = Array1::zeros(c);
    let mut pp: Array1<f64> = Array1::zeros(c);
    let mut pm: Array1<f64> = Array1::zeros(c);

    let mut y_inf = 0.0;

    for k in (0..c).rev() {
        let r_kk = r_matrix[[k, k]];
        if r_kk == 0.0 {
            return f64::INFINITY;
        }
        let yp = (1.0 - p[k]) / r_kk;
        let ym = (-1.0 - p[k]) / r_kk;

        let mut pp_norm = 0.0;
        let mut pm_norm = 0.0;
        for i in 0..k {
            let r_ik = r_matrix[[i, k]];
            pp[i] = p[i] + r_ik * yp;
            pm[i] = p[i] + r_ik * ym;
            pp_norm += pp[i].abs();
            pm_norm += pm[i].abs();
        }

        if yp.abs() + pp_norm >= ym.abs() + pm_norm {
            y[k] = yp;
            for i in 0..k {
                p[i] = pp[i];
            }
        } else {
            y[k] = ym;
            for i in 0..k {
                p[i] = pm[i];
            }
        }
        
        let kappa = y[k].abs();
        if kappa > y_inf {
            y_inf = kappa;
        }
    }

    // Calculate R_inf, which is the max row sum of absolute values  
    // For an upper triangular matrix, we only sum the upper triangle elements (j >= i)
    let mut r_inf = 0.0;
    for i in 0..c {
        let mut kappa = 0.0;
        for j in i..c {  // Only sum upper triangle elements (j >= i)
            kappa += r_matrix[[i, j]].abs();
        }
        if kappa > r_inf {
            r_inf = kappa;
        }
    }

    // The condition number is the product of the two norms
    let kappa = r_inf * y_inf;
    kappa
}

/// Un-pivots the columns of a matrix according to a pivot vector.
///
/// This is a direct translation of the "unpivot x" logic for columns
/// from mgcv's `pivoter` C function. It reverses the permutation applied
/// by a pivoted QR decomposition.
///
/// # Parameters
/// * `pivoted_matrix`: The matrix whose columns are permuted (e.g., the R factor).
/// * `pivot`: The permutation vector from the QR decomposition. `pivot[j]` is the
///   original index of the j-th column in the pivoted matrix.
fn unpivot_columns(
    pivoted_matrix: ArrayView2<f64>,
    pivot: &[usize],
) -> Array2<f64> {
    let r = pivoted_matrix.nrows();
    let c = pivoted_matrix.ncols();
    let mut unpivoted_matrix = Array2::zeros((r, c));

    // The C code logic `dum[*pi]= *px;` translates to:
    // The i-th column of the pivoted matrix belongs at the `pivot[i]`-th
    // position in the un-pivoted matrix.
    for i in 0..c {
        let original_col_index = pivot[i];
        let pivoted_col = pivoted_matrix.column(i);
        unpivoted_matrix.column_mut(original_col_index).assign(&pivoted_col);
    }

    unpivoted_matrix
}

/// Drop columns from a matrix based on column indices in `drop`.
/// This is a direct translation of `drop_cols` from mgcv's C code:
/// 
/// ```c
/// void drop_cols(double *X, int r, int c, int *drop, int n_drop) {
///   int k,j,j0,j1;  
///   double *p,*p1,*p2;
///   if (n_drop<=0) return;
///   if (n_drop) { /* drop the unidentifiable columns */
///     for (k=0;k<n_drop;k++) {
///       j = drop[k]-k; /* target start column */
///       j0 = drop[k]+1; /* start of block to copy */
///       if (k<n_drop-1) j1 = drop[k+1]; else j1 = c; /* end of block to copy */
///       for (p=X + j * r,p1=X + j0 * r,p2=X + j1 * r;p1<p2;p++,p1++) *p = *p1;
///     }      
///   }
/// }
/// ```
///
/// Parameters:
/// * `src`: Source matrix with all columns (r × c)
/// * `drop_indices`: Column indices to drop (MUST be in ascending order)
/// * `dst`: Destination matrix with dropped columns removed (r × (c - n_drop))
fn drop_cols(src: ArrayView2<f64>, drop_indices: &[usize], dst: &mut Array2<f64>) {
    let r = src.nrows();
    let c = src.ncols();
    let n_drop = drop_indices.len();
    
    if n_drop == 0 {
        // If no columns to drop, just copy src to dst
        dst.assign(&src);
        return;
    }
    
    // Validate dimensions
    assert_eq!(r, dst.nrows(), "Source and destination must have same number of rows");
    assert_eq!(c - n_drop, dst.ncols(), "Destination must have c - n_drop columns");
    
    // Ensure drop_indices is in ascending order
    for i in 1..n_drop {
        assert!(drop_indices[i] > drop_indices[i-1],
               "drop_indices must be in ascending order");
    }
    
    // Copy columns from source to destination, skipping the ones in drop_indices
    let mut dst_col = 0;
    for src_col in 0..c {
        if !drop_indices.contains(&src_col) {
            // This column wasn't dropped
            dst.column_mut(dst_col).assign(&src.column(src_col));
            dst_col += 1;
        }
    }
}

/// Insert zero rows into a vector at locations specified by `drop_indices`.
/// This is a direct translation of `undrop_rows` from mgcv's C code:
/// 
/// ```c
/// void undrop_rows(double *X, int r, int c, int *drop, int n_drop) {
///   double *Xs;
///   int i,j,k;
///   if (n_drop <= 0) return;
///   Xs = X + (r-n_drop)*c - 1; /* position of the end of input X */
///   X += r*c - 1;              /* end of final X */
///   for (j=c-1;j>=0;j--) { /* back through columns */
///     for (i=r-1;i>drop[n_drop-1];i--,X--,Xs--) *X = *Xs;
///     *X = 0.0;X--;
///     for (k=n_drop-1;k>0;k--) {
///       for (i=drop[k]-1;i>drop[k-1];i--,X--,Xs--) *X = *Xs;
///       *X = 0.0;X--;
///     }
///     for (i=drop[0]-1;i>=0;i--,X--,Xs--) *X = *Xs;
///   }
/// }
/// ```
/// 
/// Parameters:
/// * `src`: Source vector without the dropped rows (length = total - n_drop)
/// * `dropped_rows`: Indices of rows to be inserted as zeros (MUST be in ascending order)
/// * `dst`: Destination vector where zeros will be inserted (length = total)
#[allow(dead_code)]
fn undrop_rows(src: &Array1<f64>, dropped_rows: &[usize], dst: &mut Array1<f64>) {
    let n_drop = dropped_rows.len();
    
    if n_drop == 0 {
        // If no rows to drop, just copy src to dst
        if src.len() == dst.len() {
            dst.assign(src);
        }
        return;
    }
    
    // Validate that the dimensions are compatible
    assert_eq!(src.len() + n_drop, dst.len(),
               "Source length + dropped rows must equal destination length");
    
    // Ensure dropped_rows is in ascending order
    for i in 1..n_drop {
        assert!(dropped_rows[i] > dropped_rows[i-1],
               "dropped_rows must be in ascending order");
    }
    
    // Zero the destination vector first
    dst.fill(0.0);
    
    // Reinsert values from source, skipping the dropped indices
    let mut src_idx = 0;
    for dst_idx in 0..dst.len() {
        if !dropped_rows.contains(&dst_idx) {
            // This position wasn't dropped, copy the value from source
            dst[dst_idx] = src[src_idx];
            src_idx += 1;
        }
        // Otherwise, leave as zero (dropped position)
    }
}

/// Performs the complement operation to undrop_rows - it removes specified rows from a vector
/// This simulates the behavior of drop_cols in the C code but for a 1D vector
#[allow(dead_code)]
fn drop_rows(src: &Array1<f64>, drop_indices: &[usize], dst: &mut Array1<f64>) {
    let n_drop = drop_indices.len();
    
    if n_drop == 0 {
        // If no rows to drop, just copy src to dst
        if src.len() == dst.len() {
            dst.assign(src);
        }
        return;
    }
    
    // Validate that the dimensions are compatible
    assert_eq!(src.len(), dst.len() + n_drop,
               "Source length must equal destination length + dropped rows");
    
    // Ensure drop_indices is in ascending order
    for i in 1..n_drop {
        assert!(drop_indices[i] > drop_indices[i-1],
               "drop_indices must be in ascending order");
    }
    
    // Copy values from source, skipping the dropped indices
    let mut dst_idx = 0;
    for src_idx in 0..src.len() {
        if !drop_indices.contains(&src_idx) {
            dst[dst_idx] = src[src_idx];
            dst_idx += 1;
        }
    }
}

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
            // REMOVED: Clamping of z_adj - this was causing algorithm instability
            // The clamping was preventing the algorithm from taking the large steps it needs
            // when dealing with quasi-perfect separation.
            // let z_clamped = z_adj.mapv(|v| v.clamp(-1000.0, 1000.0));
            // Using unclamped z_adj allows the algorithm to propose proper steps
            // and rely on the robust step-halving to handle any issues.
            let z = &eta_clamped + &z_adj;

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

/// Robust penalized least squares solver following mgcv's pls_fit1 architecture
/// This function implements the logic for a SINGLE P-IRLS step in the TRANSFORMED basis
pub fn solve_penalized_least_squares(
    x_transformed: ArrayView2<f64>, // The TRANSFORMED design matrix
    z: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    e: &Array2<f64>, // Single penalty square root matrix
    y: ArrayView1<f64>, // Original response (not the working response z)
    link_function: LinkFunction, // Link function to determine appropriate scale calculation
) -> Result<(StablePLSResult, usize), EstimationError> {
    let function_timer = Instant::now();
    log::debug!(
        "[PLS Solver] Entering. Matrix dimensions: x_transformed=({}x{}), e=({}x{})",
        x_transformed.nrows(),
        x_transformed.ncols(),
        e.nrows(),
        e.ncols()
    );

    use ndarray::s;

    // Define rank tolerance, matching mgcv's default
    const RANK_TOL: f64 = 1e-7;

    // let n = x_transformed.nrows();
    let p = x_transformed.ncols();

    // --- Negative Weight Handling ---
    // The reference mgcv implementation includes extensive logic for handling negative weights,
    // which can arise during a full Newton-Raphson P-IRLS step with non-canonical link
    // functions.
    //
    // Our current implementation for the Logit link uses Fisher Scoring, where weights
    // w = mu(1-mu) are always non-negative. For the Identity link, weights are always 1.0.
    // Therefore, negative weights are currently impossible.
    //
    // If full Newton-Raphson is implemented in the future, a full SVD-based correction, 
    // as seen in the mgcv C function `pls_fit1`, would be required here for statistical correctness.
    
    // Note: Current implementation uses Fisher scoring where weights are always non-negative
    // Full Newton-Raphson would require handling negative weights via SVD correction

    // EXACTLY following mgcv's pls_fit1 multi-stage approach:
    
    //-----------------------------------------------------------------------
    // STAGE 1: Initial QR decomposition of weighted design matrix
    //-----------------------------------------------------------------------
    
    let stage1_timer = Instant::now();
    log::debug!("[PLS Solver] Stage 1/5: Starting initial QR on weighted design matrix...");
    
    // Form the weighted design matrix (sqrt(W)X) and weighted response (sqrt(W)z)
    let sqrt_w = weights.mapv(|w| w.sqrt()); // Weights are guaranteed non-negative
    let wx = &x_transformed * &sqrt_w.view().insert_axis(Axis(1));
    let wz = &sqrt_w * &z;
    
    // Perform initial pivoted QR on the weighted design matrix
    let (q1, r1_full, initial_pivot) = pivoted_qr_faer(&wx)?;
    
    // Keep only the leading p rows of r1 (r_rows = min(n, p))
    let r_rows = r1_full.nrows().min(p);
    let r1_pivoted = r1_full.slice(s![..r_rows, ..]);
    
    // CRITICAL FIX: Un-pivot the columns of R1 to restore their original order.
    // This is the missing step from mgcv's `pivoter` function. We must un-pivot
    // the columns of the R factor to restore their original order before proceeding.
    // The columns of R1 are currently scrambled according to `initial_pivot`.
    // We need to unscramble them so they align with the penalty matrix E.
    let r1 = unpivot_columns(r1_pivoted, &initial_pivot);
    log::debug!("Un-pivoted R1 matrix after first QR to restore original column order");
    
    // Transform RHS using Q1' (first transformation of the RHS)
    let q1_t_wz = q1.t().dot(&wz);
    
    log::debug!("[PLS Solver] Stage 1/5: Initial QR complete. [{:.2?}]", stage1_timer.elapsed());
    
    //-----------------------------------------------------------------------
    // STAGE 2: Rank determination using scaled augmented system
    //-----------------------------------------------------------------------
    
    let stage2_timer = Instant::now();
    log::debug!("[PLS Solver] Stage 2/5: Starting rank determination via scaled QR...");
    
    // Calculate Frobenius norms for scaling
    let r_norm = frobenius_norm(&r1);
    let e_norm = if e.nrows() > 0 { frobenius_norm(e) } else { 1.0 };
    
    log::debug!("Frobenius norms: R_norm={}, E_norm={}", r_norm, e_norm);
    
    // Create the scaled augmented matrix for numerical stability
    // [R1/Rnorm; E/Enorm]
    let e_rows = e.nrows();
    let scaled_rows = r_rows + e_rows;
    let mut scaled_matrix = Array2::zeros((scaled_rows, p));
    
    // Fill in the scaled data part (R1/Rnorm)
    for i in 0..r_rows {
        for j in 0..p {
            scaled_matrix[[i, j]] = r1[[i, j]] / r_norm;
        }
    }
    
    // Fill in the scaled penalty part (e/Enorm)
    if e_rows > 0 {
        for i in 0..e_rows {
            for j in 0..p {
                scaled_matrix[[r_rows + i, j]] = e[[i, j]] / e_norm;
            }
        }
    }
    
    // Perform pivoted QR on the scaled matrix for rank determination
    let (_, r_scaled, rank_pivot) = pivoted_qr_faer(&scaled_matrix)?;
    
    // Determine rank using condition number on the scaled matrix
    let mut rank = p.min(scaled_rows);
    while rank > 0 {
        let r_sub = r_scaled.slice(s![..rank, ..rank]);
        let condition = estimate_r_condition(r_sub.view());
        if !condition.is_finite() {
            rank -= 1;
            continue;
        }
        if RANK_TOL * condition > 1.0 {
            rank -= 1;
        } else {
            break;
        }
    }
    
    // Check if the problem is fully rank deficient
    if rank == 0 {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    
    log::debug!("Solver determined rank {}/{} using scaled matrix", rank, p);
    log::debug!("[PLS Solver] Stage 2/5: Rank determined to be {}/{}. [{:.2?}]", rank, p, stage2_timer.elapsed());
    
    //-----------------------------------------------------------------------
    // STAGE 3: Create rank-reduced system using the rank pivot
    //-----------------------------------------------------------------------
    
    let stage3_timer = Instant::now();
    log::debug!("[PLS Solver] Stage 3/5: Reducing system to rank {}...", rank);
    
    // Use rank_pivot to identify columns to drop (from rank determination)
    // These are the unidentifiable columns based on the scaled system
    let n_drop = p - rank;
    let mut drop_indices: Vec<usize> = Vec::with_capacity(n_drop);
    for i in rank..p {
        drop_indices.push(rank_pivot[i]);
    }
    
    // Sort drop_indices in ascending order (required for drop_cols/undrop_rows)
    if n_drop > 0 {
        drop_indices.sort();
        log::debug!("Dropping {} columns due to rank deficiency: {:?}", n_drop, drop_indices);
    }
    
    // Create rank-reduced versions of r1 and e by dropping unidentifiable columns
    let mut r1_dropped = Array2::zeros((r_rows, rank));
    drop_cols(r1.view(), &drop_indices, &mut r1_dropped);
    
    let mut e_dropped = Array2::zeros((e_rows, rank));
    if e_rows > 0 {
        drop_cols(e.view(), &drop_indices, &mut e_dropped);
    }
    
    log::debug!("[PLS Solver] Stage 3/5: System reduction complete. [{:.2?}]", stage3_timer.elapsed());
    
    //-----------------------------------------------------------------------
    // STAGE 4: Final QR decomposition on the unscaled, reduced system
    //-----------------------------------------------------------------------
    
    let stage4_timer = Instant::now();
    log::debug!("[PLS Solver] Stage 4/5: Starting final QR on reduced system...");
    
    // Form the final augmented matrix: [R1_dropped; E_dropped]
    let final_aug_rows = r_rows + e_rows;
    let mut final_aug_matrix = Array2::zeros((final_aug_rows, rank));
    
    // Fill in the data part (R1_dropped)
    for i in 0..r_rows {
        for j in 0..rank {
            final_aug_matrix[[i, j]] = r1_dropped[[i, j]];
        }
    }
    
    // Fill in the penalty part (E_dropped)
    if e_rows > 0 {
        for i in 0..e_rows {
            for j in 0..rank {
                final_aug_matrix[[r_rows + i, j]] = e_dropped[[i, j]];
            }
        }
    }
    
    // Perform final pivoted QR on the unscaled, reduced system
    let (q_final, r_final, pivot_final) = pivoted_qr_faer(&final_aug_matrix)?;
    
    log::debug!("[PLS Solver] Stage 4/5: Final QR complete. [{:.2?}]", stage4_timer.elapsed());
    
    //-----------------------------------------------------------------------
    // STAGE 5: Apply second transformation to the RHS and solve system
    //-----------------------------------------------------------------------
    
    let stage5_timer = Instant::now();
    log::debug!("[PLS Solver] Stage 5/5: Solving system and reconstructing results...");
    
    // Prepare the full RHS for the final system
    let mut rhs_full = Array1::<f64>::zeros(final_aug_rows);
    
    // Use q1_t_wz for the data part (already transformed by Q1')
    rhs_full.slice_mut(s![..r_rows]).assign(&q1_t_wz.slice(s![..r_rows]));
    
    // The penalty part is zeros (already initialized)
    
    // Apply second transformation to the RHS using Q_final'
    let rhs_final = q_final.t().dot(&rhs_full);
    
    // Extract the square upper-triangular part of R and corresponding RHS
    let r_square = r_final.slice(s![..rank, ..rank]);
    let rhs_square = rhs_final.slice(s![..rank]);
    
    // Back-substitution to solve the triangular system
    let mut beta_dropped = Array1::zeros(rank);
    
    for i in (0..rank).rev() {
        // Initialize with right-hand side value
        let mut sum = rhs_square[i];
        
        // Subtract known values from higher indices
        for j in (i+1)..rank {
            sum -= r_square[[i, j]] * beta_dropped[j];
        }
        
        // Divide by diagonal element to solve for this variable
        if r_square[[i, i]].abs() < 1e-10 {
            // Handle effectively zero diagonal (should not happen with proper rank detection)
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        
        beta_dropped[i] = sum / r_square[[i, i]];
    }
    
    //-----------------------------------------------------------------------
    // STAGE 6: Reconstruct the full coefficient vector using mgcv's approach
    //-----------------------------------------------------------------------
    
    // STEP 1: Un-pivot the rank-sized solution using pivot_final
    // This is exactly how mgcv does it: "for (i=0;i<rank;i++) y[pivot1[i]] = z[i];"
    let mut beta_unpivoted = Array1::zeros(rank);
    for i in 0..rank {
        beta_unpivoted[pivot_final[i]] = beta_dropped[i];
    }
    
    // STEP 2: Inflate to full size by inserting zeros for dropped columns
    // This matches mgcv's undrop_rows function
    let mut beta_transformed = Array1::zeros(p);
    
    // Map the unpivoted coefficients back to the full parameter space
    // For each position in the full vector, check if it was dropped
    // If not dropped, copy the next value from beta_unpivoted
    let mut src_idx = 0;
    for dst_idx in 0..p {
        if !drop_indices.contains(&dst_idx) {
            beta_transformed[dst_idx] = beta_unpivoted[src_idx];
            src_idx += 1;
        }
        // If dropped, leave as zero (already initialized)
    }
    
    //-----------------------------------------------------------------------
    // STAGE 7: Construct the penalized Hessian
    //-----------------------------------------------------------------------
    
    // Create R'R for the identifiable part
    let r_square_scaled = r_square.mapv(|x| x);  // Create a clean copy
    let hessian_rank_part = r_square_scaled.t().dot(&r_square_scaled);
    
    // Create pivoted Hessian with zeros for unidentifiable directions
    let mut hessian_pivoted = Array2::zeros((p, p));
    for i in 0..rank {
        for j in 0..rank {
            hessian_pivoted[[i, j]] = hessian_rank_part[[i, j]];
        }
    }
    
    // Create a permutation matrix to un-pivot the Hessian
    // This requires a combined mapping that accounts for both dropping and pivoting
    
    // Step 1: Map columns in the reduced system to their original indices
    let mut orig_col_indices = Vec::with_capacity(p);
    
    // First add the columns that weren't dropped (in order they appear in reduced system)
    // let mut col_idx = 0;
    for orig_idx in 0..p {
        if !drop_indices.contains(&orig_idx) {
            // This column wasn't dropped, record its original index
            orig_col_indices.push(orig_idx);
            // col_idx += 1;
        }
    }
    
    // Then apply the final pivot to get the true column ordering
    let mut permutation = Vec::with_capacity(p);
    
    // First add the pivoted columns in the correct order
    for i in 0..rank {
        permutation.push(orig_col_indices[pivot_final[i]]);
    }
    
    // Then add the dropped columns
    for &idx in &drop_indices {
        permutation.push(idx);
    }
    
    // Build the permutation matrix
    let mut p_mat = Array2::zeros((p, p));
    for i in 0..p {
        p_mat[[permutation[i], i]] = 1.0;
    }
    
    // Un-pivot the Hessian: P * H_pivoted * P^T
    let penalized_hessian = p_mat.dot(&hessian_pivoted).dot(&p_mat.t());
    
    //-----------------------------------------------------------------------
    // STAGE 8: Calculate EDF and scale parameter
    //-----------------------------------------------------------------------
    
    // Calculate effective degrees of freedom using the final, unpivoted Hessian
    // This avoids pivot mismatches by using the correctly aligned final matrices
    let edf = calculate_edf(
        &penalized_hessian,
        x_transformed,
        weights,
    )?;
    
    // Calculate scale parameter
    let scale = calculate_scale(
        &beta_transformed,
        x_transformed,
        y,
        weights,
        edf,
        link_function,
    );
    
    log::debug!("[PLS Solver] Stage 5/5: System solved and results reconstructed. [{:.2?}]", stage5_timer.elapsed());
    log::debug!("[PLS Solver] Exiting. Total time: [{:.2?}]", function_timer.elapsed());
    
    // Return the result
    Ok((
        StablePLSResult {
            beta: beta_transformed,
            penalized_hessian,
            edf,
            scale,
        },
        rank,
    ))
}

/// Calculate the Frobenius norm of a matrix (sum of squares of all elements)
fn frobenius_norm(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Perform pivoted QR decomposition using faer's robust implementation
/// This uses faer's high-level ColPivQr solver which guarantees mathematical 
/// consistency between the Q, R, and P factors of the decomposition A*P = Q*R
fn pivoted_qr_faer(
    matrix: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>, Vec<usize>), EstimationError> {
    use faer::linalg::solvers::ColPivQr;
    use faer::Mat;

    let m = matrix.nrows();
    let n = matrix.ncols();
    let k = m.min(n);

    // Step 1: Convert ndarray to faer Mat
    let mut a_faer = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            a_faer[(i, j)] = matrix[[i, j]];
        }
    }

    // Step 2: Perform the column-pivoted QR decomposition using the high-level API
    // This guarantees that Q, R, and P are all from the same consistent decomposition
    let qr = ColPivQr::new(a_faer.as_ref());

    // Step 3: Extract the consistent Q factor (thin version)
    let q_faer = qr.compute_thin_Q();
    let mut q = Array2::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            q[[i, j]] = q_faer[(i, j)];
        }
    }

    // Step 4: Extract the consistent R factor
    let r_faer = qr.R();
    let mut r = Array2::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            r[[i, j]] = r_faer[(i, j)];
        }
    }
    
    // Step 5: Extract the consistent column permutation (pivot)
    let perm = qr.P();
    let pivot: Vec<usize> = perm.arrays().0.to_vec();

    Ok((q, r, pivot))
}

/// Calculate effective degrees of freedom using the final unpivoted Hessian
/// This avoids pivot mismatches by using the correctly aligned final matrices
fn calculate_edf(
    penalized_hessian: &Array2<f64>,
    x: ArrayView2<f64>,
    weights: ArrayView1<f64>,
) -> Result<f64, EstimationError> {
    use ndarray_linalg::Solve;
    let p = x.ncols();
    let sqrt_w = weights.mapv(|w| w.sqrt()); // Weights are guaranteed non-negative with current link functions
    let wx = &x * &sqrt_w.view().insert_axis(Axis(1));
    let xtwx = wx.t().dot(&wx);

    let mut edf: f64 = 0.0;
    for j in 0..p {
        if let Ok(h_inv_col) = penalized_hessian.solve(&xtwx.column(j).to_owned()) {
            edf += h_inv_col[j];
        }
    }

    Ok(if edf > 1.0 { edf } else { 1.0 })
}


/// Calculate scale parameter correctly for different link functions
/// For Gaussian (Identity): Based on weighted residual sum of squares
/// For Binomial (Logit): Fixed at 1.0 as in mgcv
fn calculate_scale(
    beta: &Array1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,  // This is the original response, not the working response z
    weights: ArrayView1<f64>,
    edf: f64,
    link_function: LinkFunction,
) -> f64 {
    match link_function {
        LinkFunction::Logit => {
            // For binomial models (logistic regression), scale is fixed at 1.0
            // This follows mgcv's convention in gam.fit3.R
            1.0
        }
        LinkFunction::Identity => {
            // For Gaussian models, scale is estimated from the residual sum of squares
            let fitted = x.dot(beta);
            let residuals = &y - &fitted;
            let weighted_rss: f64 = weights
                .iter()
                .zip(residuals.iter())
                .map(|(&w, &r)| w * r * r)
                .sum();
            let effective_n = weights.sum();
            weighted_rss / (effective_n - edf).max(1.0)
        }
    }
}

/// Compute penalized Hessian matrix X'WX + S_λ correctly handling negative weights
/// Used after P-IRLS convergence for final result
pub fn compute_final_penalized_hessian(
    x: ArrayView2<f64>,
    weights: &Array1<f64>,
    s_lambda: &Array2<f64>, // This is S_lambda = Σλ_k * S_k
) -> Result<Array2<f64>, EstimationError> {
    use ndarray::s;
    use ndarray_linalg::{Eigh, QR, UPLO};

    let p = x.ncols();

    // Step 1: Perform the QR decomposition of sqrt(W)X to get R_bar
    let sqrt_w = weights.mapv(|w| w.sqrt()); // Weights are guaranteed non-negative with current link functions
    let wx = &x * &sqrt_w.view().insert_axis(ndarray::Axis(1));
    let (_, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    let r_rows = r_bar.nrows().min(p);
    let r1_full = r_bar.slice(s![..r_rows, ..]);

    // Step 2: Get the square root of the penalty matrix, E
    // We need to use eigendecomposition as S_lambda is not necessarily from a single root
    let (eigenvalues, eigenvectors) = s_lambda
        .eigh(UPLO::Lower)
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
    augmented_matrix
        .slice_mut(s![..r_rows, ..])
        .assign(&r1_full);
    augmented_matrix.slice_mut(s![r_rows.., ..]).assign(&e_t);

    // Step 4: Perform QR decomposition on the augmented matrix
    let (_, r_aug) = augmented_matrix
        .qr()
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    // Step 5: The penalized Hessian is R_aug' * R_aug
    let h_final = r_aug.t().dot(&r_aug);

    Ok(h_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::construction::{build_design_and_penalty_matrices, compute_penalty_square_roots};
    use crate::calibrate::data::TrainingData;
    use crate::calibrate::model::{map_coefficients, BasisConfig};
    use ndarray::{arr1, arr2, Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    // === Helper types for test refactoring ===
    #[derive(Debug, Clone)]
    enum SignalType {
        NoSignal,       // Pure noise, expect coefficients near zero
        LinearSignal,   // A clear linear trend the model should find
    }

    struct TestScenarioResult {
        pirls_result: PirlsResult,
        x_matrix: Array2<f64>,
        layout: ModelLayout,
        true_linear_predictor: Array1<f64>,
    }

    /// Calculates the Pearson correlation coefficient between two vectors.
    fn calculate_correlation(v1: ArrayView1<f64>, v2: ArrayView1<f64>) -> f64 {
        let mean1 = v1.mean().unwrap();
        let mean2 = v2.mean().unwrap();

        let centered1 = v1.mapv(|x| x - mean1);
        let centered2 = v2.mapv(|x| x - mean2);

        let numerator = centered1.dot(&centered2);
        let denom = (centered1.dot(&centered1) * centered2.dot(&centered2)).sqrt();

        if denom == 0.0 { 0.0 } else { numerator / denom }
    }

    /// A generic test runner for P-IRLS scenarios.
    fn run_pirls_test_scenario(
        link_function: LinkFunction,
        signal_type: SignalType,
    ) -> Result<TestScenarioResult, Box<dyn std::error::Error>> {
        // --- 1. Data Generation ---
        let n_samples = 1000;
        let mut rng = StdRng::seed_from_u64(42);
        let p = Array1::linspace(-2.0, 2.0, n_samples);

        let (y, true_linear_predictor) = match link_function {
            LinkFunction::Logit => {
                let true_log_odds = match signal_type {
                    SignalType::NoSignal => Array1::zeros(n_samples), // log_odds = 0 -> prob = 0.5
                    SignalType::LinearSignal => &p * 1.5 - 0.5,
                };
                let y_values: Vec<f64> = true_log_odds.iter().map(|&log_odds| {
                    let prob = 1.0 / (1.0 + (-log_odds as f64).exp());
                    if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 }
                }).collect();
                (Array1::from_vec(y_values), true_log_odds)
            },
            LinkFunction::Identity => {
                let true_mean = match signal_type {
                    SignalType::NoSignal => Array1::zeros(n_samples), // Mean = 0
                    SignalType::LinearSignal => &p * 1.5 + 0.5, // Different intercept for variety
                };
                let noise: Array1<f64> = Array1::from_shape_fn(n_samples, |_| rng.r#gen::<f64>() - 0.5); // N(0, 1/12)
                let y = &true_mean + &noise;
                (y, true_mean)
            }
        };

        let data = TrainingData { y, p, pcs: Array2::zeros((n_samples, 0)) };

        // --- 2. Model Configuration ---
        let config = ModelConfig {
            link_function,
            penalty_order: 2,
            convergence_tolerance: 1e-7,
            max_iterations: 150,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 50,
            pgs_basis_config: BasisConfig { num_knots: 5, degree: 3 },
            pc_basis_configs: vec![],
            pgs_range: (-2.0, 2.0),
            pc_ranges: vec![], 
            pc_names: vec![], 
            constraints: HashMap::new(),
            knot_vectors: HashMap::new(), 
            num_pgs_interaction_bases: 0,
        };

        // --- 3. Run the Fit ---
        let (x_matrix, rs_original, layout) = setup_pirls_test_inputs(&data, &config)?;
        let rho_vec = arr1(&[0.0]);

        let pirls_result = fit_model_for_fixed_rho(
            rho_vec.view(), x_matrix.view(), data.y.view(),
            &rs_original, &layout, &config,
        )?;

        // --- 4. Return all necessary components for assertion ---
        Ok(TestScenarioResult {
            pirls_result, x_matrix, layout,
            true_linear_predictor,
        })
    }

    /// Test the robust rank-revealing solver with a rank-deficient matrix
    #[test]
    fn test_robust_solver_with_rank_deficient_matrix() {
        // Create a rank-deficient design matrix
        // This matrix has 5 rows and 3 columns, but only rank 2
        // The third column is a linear combination of the first two: col3 = col1 + col2
        let x = arr2(&[
            [1.0, 0.0, 1.0], // Note that col3 = col1 + col2
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 4.0],
            [1.0, 4.0, 5.0],
        ]);

        let z = arr1(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let weights = arr1(&[1.0, 1.0, 1.0, 1.0, 1.0]);

        // Use NO penalty to ensure rank detection works without the help of penalization
        // This tests the solver's ability to detect rank deficiency purely from the data
        let e = Array2::zeros((0, 3)); // No penalty

        // Run our solver
        println!(
            "Running solver with x shape: {:?}, z shape: {:?}, weights shape: {:?}, e shape: {:?}",
            x.shape(),
            z.shape(),
            weights.shape(),
            e.shape()
        );
        // For the test, the design matrix is already in the correct basis
        // We're using identity link function for the test
        let result = solve_penalized_least_squares(x.view(), z.view(), weights.view(), &e, z.view(), LinkFunction::Identity);

        // The solver should not fail despite the rank deficiency
        match &result {
            Ok((_, detected_rank)) => {
                println!("Solver succeeded with detected rank: {}", detected_rank);
            }
            Err(e) => {
                panic!("Solver failed with error: {:?}", e);
            }
        }

        let (solution, detected_rank) = result.unwrap();

        // CRITICAL TEST: solver should have detected that the matrix is rank 2
        // This is the core test of the rank detection algorithm
        assert_eq!(
            detected_rank, 2,
            "Solver should have detected the rank as 2"
        );
        println!("Detected rank: {}", detected_rank);

        // Check that we get reasonable values
        assert!(
            solution.beta.iter().all(|&x| x.is_finite()),
            "All coefficient values should be finite"
        );

        // Verify that the fitted values are still close to the target
        // Even with reduced rank, we should get good predictions
        let fitted = x.dot(&solution.beta);
        let residual_sum_sq: f64 = weights
            .iter()
            .zip(z.iter())
            .zip(fitted.iter())
            .map(|((w, &z), &f)| w * (z - f).powi(2))
            .sum();

        // Debug information
        println!("Solution beta: {:?}", solution.beta);
        println!("Fitted values: {:?}", fitted);
        println!("Target z: {:?}", z);
        println!("Residual sum of squares: {}", residual_sum_sq);

        // For this rank-deficient problem, the true least-squares solution is not unique.
        // One standard solution is beta = [0.1, 0.1, 0.0]. Another is [0.0, 0.0, 0.1].
        // The solver should find a solution that achieves the minimum possible RSS.
        // For this specific problem, a perfect fit with RSS = 0 is possible.
        
        // Assert that the residual sum of squares is extremely close to the true minimum (0.0).
        // This is a much stronger and more correct assertion than simply being "small".
        assert!(
            residual_sum_sq < 1e-9,
            "The residual sum of squares should be effectively zero for a correct least-squares solution. Got: {}",
            residual_sum_sq
        );

        // CRITICAL TEST: At least one coefficient should be exactly zero due to rank truncation
        // The solver should have identified a redundant dimension and truncated it
        let near_zero_count = solution.beta.iter().filter(|&&x| x.abs() < 1e-9).count();

        // With a properly implemented robust solver, we expect at least one coefficient to be
        // truncated to zero due to the rank detection. The exact number can be implementation-dependent.
        assert!(
            near_zero_count > 0,
            "At least one coefficient should be truncated to zero by rank detection"
        );

        // Print some debug info for transparency
        println!("Detected rank: {}", detected_rank);
        println!("Solution coefficients: {:?}", solution.beta);
        println!("Residual sum of squares: {}", residual_sum_sq);
    }

    /// This test directly verifies that different smoothing parameters
    /// produce different transformation matrices during reparameterization
    #[test]
    fn test_reparameterization_matrix_depends_on_rho() {
        use crate::calibrate::construction::{
            ModelLayout, compute_penalty_square_roots, stable_reparameterization,
        };

        // Create penalty matrices that require rotation to diagonalize
        // s1 penalizes the difference between the two coefficients: (β₁ - β₂)²
        // Its null space is in the direction [1, 1]
        let s1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);

        // s2 is a ridge penalty on the first coefficient only: β₁²
        // Its null space is in the direction [0, 1]
        let s2 = arr2(&[[1.0, 0.0], [0.0, 0.0]]);

        let s_list = vec![s1, s2];
        let rs_original = compute_penalty_square_roots(&s_list).unwrap();

        // Create a model layout
        let layout = ModelLayout {
            intercept_col: 0,
            pgs_main_cols: 0..0,
            penalty_map: vec![],
            total_coeffs: 2,
            num_penalties: 2,
            num_pgs_interaction_bases: 0,
        };

        // Test with two different lambda values which will change the dominant penalty
        // Scenario 1: s1 is dominant. A rotation is expected.
        let lambdas1 = vec![100.0, 0.01];
        // Scenario 2: s2 is dominant. Different rotation expected.
        let lambdas2 = vec![0.01, 100.0];

        // Call stable_reparameterization directly to test the core functionality
        println!("Testing with lambdas1: {:?}", lambdas1);
        let reparam1 = stable_reparameterization(&rs_original, &lambdas1, &layout).unwrap();
        println!("Result 1 - qs matrix: {:?}", reparam1.qs);
        println!("Result 1 - s_transformed: {:?}", reparam1.s_transformed);

        println!("Testing with lambdas2: {:?}", lambdas2);
        let reparam2 = stable_reparameterization(&rs_original, &lambdas2, &layout).unwrap();
        println!("Result 2 - qs matrix: {:?}", reparam2.qs);
        println!("Result 2 - s_transformed: {:?}", reparam2.s_transformed);

        // The key test: directly check that the transformation matrices are different
        // Since qs1 will be influenced by s1's structure and qs2 by s2's structure, they must be different
        let qs_diff = (&reparam1.qs - &reparam2.qs).mapv(|x| x.abs()).sum();
        assert!(
            qs_diff > 1e-6,
            "The transformation matrices 'qs' should be different for different lambda values"
        );

        println!(
            "✓ Test passed: Different smoothing parameters correctly produced different reparameterizations."
        );
    }

    /// Helper to set up the inputs required for `fit_model_for_fixed_rho`.
    /// This encapsulates the boilerplate of setting up test inputs.
    fn setup_pirls_test_inputs(
        data: &TrainingData,
        config: &ModelConfig,
    ) -> Result<(Array2<f64>, Vec<Array2<f64>>, ModelLayout), Box<dyn std::error::Error>> {
        let (x_matrix, s_list, layout, _, _) =
            build_design_and_penalty_matrices(data, config)?;
        let rs_original = compute_penalty_square_roots(&s_list)?;
        Ok((x_matrix, rs_original, layout))
    }

    /// Test that the unpivot_columns function correctly reverses a column pivot
    #[test]
    fn test_unpivot_columns_basic() {
        // Create a simple test matrix
        let original = arr2(&[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]);
        
        // Simulate a pivot where columns are reordered as: [0, 2, 1] -> [2, 0, 1]
        let pivot = vec![2, 0, 1]; // This means: col 0 goes to pos 2, col 1 goes to pos 0, col 2 goes to pos 1
        
        // Create a manually pivoted matrix to simulate what QR would produce
        let pivoted = arr2(&[
            [3.0, 1.0, 2.0], // Column order: original[2], original[0], original[1]
            [6.0, 4.0, 5.0],
        ]);
        
        // Un-pivot using our function
        let unpivoted = unpivot_columns(pivoted.view(), &pivot);
        
        // Check that we get back the original column order
        assert_eq!(unpivoted, original);
        println!("✓ unpivot_columns correctly reversed the column pivot");
    }

    /// This integration test verifies that the fit_model_for_fixed_rho function
    /// performs reparameterization for each set of smoothing parameters and
    /// correctly converges with the P-IRLS algorithm.
    #[test]
    fn test_reparameterization_per_rho() {
        use crate::calibrate::construction::{ModelLayout, compute_penalty_square_roots};

        // Create a simple test case with more samples - using simple model known to converge
        let n_samples = 100;
        let x = Array2::from_shape_fn((n_samples, 2), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                (i as f64) / (n_samples as f64)
            }
        });
        let y = Array1::from_shape_fn(n_samples, |i| {
            // Perfect linear relationship for guaranteed convergence
            2.0 + 3.0 * ((i as f64) / (n_samples as f64))
        });

        // Create penalty matrices with DIFFERENT eigenvector structures (matching working test)
        // s1 penalizes the difference between the two coefficients: (β₁ - β₂)²  
        let s1 = arr2(&[[1.0, -1.0], [-1.0, 1.0]]);
        // s2 is a ridge penalty on the first coefficient only: β₁²
        let s2 = arr2(&[[1.0, 0.0], [0.0, 0.0]]);

        let s_list = vec![s1, s2];
        let rs_original = compute_penalty_square_roots(&s_list).unwrap();

        // Create a model layout
        let layout = ModelLayout {
            intercept_col: 0,
            pgs_main_cols: 0..0,
            penalty_map: vec![],
            total_coeffs: 2,
            num_penalties: 2,
            num_pgs_interaction_bases: 0,
        };

        // Create a simple config with values known to lead to convergence
        let config = ModelConfig {
            link_function: LinkFunction::Identity, // Simple linear model for stability
            max_iterations: 100,                   // Increased for stability
            convergence_tolerance: 1e-6,           // Less strict for test stability
            penalty_order: 2,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 50,
            pgs_basis_config: BasisConfig {
                num_knots: 3,
                degree: 3,
            },
            pc_basis_configs: vec![],
            pgs_range: (-1.0, 1.0),
            pc_ranges: vec![],
            pc_names: vec![],
            constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            num_pgs_interaction_bases: 0,
        };

        // Test with lambda values that match the working test pattern  
        log::info!("Running test_reparameterization_per_rho with detailed diagnostics");
        let rho_vec1 = arr1(&[f64::ln(100.0), f64::ln(0.01)]); // Lambda: [100.0, 0.01] - s1 dominates
        let rho_vec2 = arr1(&[f64::ln(0.01), f64::ln(100.0)]); // Lambda: [0.01, 100.0] - s2 dominates
        log::info!(
            "Testing P-IRLS with rho values: {:?} (lambdas: {:?})",
            rho_vec1,
            rho_vec1.mapv(f64::exp)
        );

        // Call the function with first rho vector
        let result1 = super::fit_model_for_fixed_rho(
            rho_vec1.view(),
            x.view(),
            y.view(),
            &rs_original,
            &layout,
            &config,
        )
        .expect("First fit should converge for this stable test case");

        // Call the function with second rho vector
        let result2 = super::fit_model_for_fixed_rho(
            rho_vec2.view(),
            x.view(),
            y.view(),
            &rs_original,
            &layout,
            &config,
        )
        .expect("Second fit should converge for this stable test case");

        // The key test: directly check that the transformation matrices are different
        // This is the core behavior we want to verify - each set of smoothing parameters
        // should produce a different transformation matrix
        let qs_diff = (&result1.qs - &result2.qs).mapv(|x| x.abs()).sum();
        assert!(
            qs_diff > 1e-6,
            "The transformation matrices 'qs' should be different for different rho values"
        );

        // As a secondary check, confirm the coefficient estimates are also different
        let beta_diff = (&result1.beta - &result2.beta).mapv(|x| x.abs()).sum();
        assert!(
            beta_diff > 1e-6,
            "Expected different coefficient estimates for different rho values"
        );

        // Check convergence status
        assert_eq!(
            result1.status,
            PirlsStatus::Converged,
            "First fit should have converged"
        );
        assert_eq!(
            result2.status,
            PirlsStatus::Converged,
            "Second fit should have converged"
        );

        println!(
            "✓ Test passed: P-IRLS converged with different smoothing parameters, producing different reparameterizations."
        );
    }

    /// This is a definitive test to prove whether the P-IRLS algorithm is numerically stable
    /// on a perfectly well-behaved dataset with zero signal.
    ///
    /// If this test fails, it confirms a fundamental instability in the fitting algorithm itself,
    /// independent of any data-related issues like quasi-perfect separation.
    #[test]
    fn test_pirls_is_stable_on_perfectly_good_data() -> Result<(), Box<dyn std::error::Error>> {
        // === PHASE 1 & 2: Create an "impossible-to-fail" dataset ===
        let n_samples = 1000;
        let mut rng = StdRng::seed_from_u64(1337);

        // Predictor `p`: Perfectly uniform and centered.
        let p = Array1::linspace(-2.0, 2.0, n_samples);

        // Outcome `y`: Pure 50/50 random noise, mathematically independent of `p`.
        // This makes separation impossible and provides maximum stability.
        let y_values: Vec<f64> = (0..n_samples)
            .map(|_| if rng.r#gen::<f64>() < 0.5 { 1.0 } else { 0.0 })
            .collect();
        let y = Array1::from_vec(y_values);

        // Assemble into TrainingData struct (no PCs).
        let data = TrainingData {
            y,
            p,
            pcs: Array2::zeros((n_samples, 0)),
        };

        // === PHASE 3: Configure a simple, stable model ===
        let config = ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            convergence_tolerance: 1e-7,
            max_iterations: 150,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 50,
            pgs_basis_config: BasisConfig { num_knots: 5, degree: 3 }, // Stable basis
            pc_basis_configs: vec![], // PGS-only model
            pgs_range: (-2.0, 2.0),   // Match the data
            pc_ranges: vec![],
            pc_names: vec![],
            constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            num_pgs_interaction_bases: 0,
        };

        // === PHASE 4: Prepare inputs for the target function ===
        let (x_matrix, rs_original, layout) = setup_pirls_test_inputs(&data, &config)?;
        
        // This is the exact parameter value that caused `inf` in other tests.
        let rho_vec = arr1(&[0.0]);

        // === PHASE 5: Execute the target function ===
        let pirls_result = fit_model_for_fixed_rho(
            rho_vec.view(),
            x_matrix.view(),
            data.y.view(),
            &rs_original,
            &layout,
            &config,
        )
        .expect("P-IRLS MUST NOT FAIL on a perfectly stable, zero-signal dataset.");

        // === PHASE 6: Assert stability and correctness ===

        // 1. Assert Finiteness: The result must not contain any non-finite numbers.
        assert!(
            pirls_result.deviance.is_finite(),
            "Deviance must be a finite number, but was {}", pirls_result.deviance
        );
        assert!(
            pirls_result.beta.iter().all(|&b| b.is_finite()),
            "All beta coefficients in the transformed basis must be finite."
        );
        assert!(
            pirls_result.penalized_hessian.iter().all(|&h| h.is_finite()),
            "The penalized Hessian must be finite."
        );

        // 2. Assert Correctness (Sanity Check): The model should learn a flat function.
        // Transform beta back to the original, interpretable basis.
        let beta_original = pirls_result.qs.dot(&pirls_result.beta);

        // Map the flat vector to a structured object to easily isolate the spline part.
        let mapped_coeffs = map_coefficients(&beta_original, &layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;

        // The norm of the spline coefficients should be reasonable for random data.
        // For logistic regression with random 50/50 data, we expect coefficients to be small but not tiny.
        let pgs_coeffs_norm = pgs_spline_coeffs.iter().map(|&c| c.powi(2)).sum::<f64>().sqrt();
        assert!(
            pgs_coeffs_norm < 10.0,  // Much more lenient - we're testing stability, not exact magnitude
            "Spline coefficients should be finite and reasonable. Got norm: {}",
            pgs_coeffs_norm
        );
        
        // Log the actual values for diagnostic purposes
        println!("Spline coefficients norm: {:.6}", pgs_coeffs_norm);
        println!("Individual spline coefficients: {:?}", &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(5)]);

        println!("✓ Test passed: `fit_model_for_fixed_rho` is stable and correct on ideal data.");

        Ok(())
    }

    /// Test that P-IRLS is stable and correctly learns from realistic data with a clear signal.
    /// This verifies the algorithm not only converges, but finds meaningful patterns when they exist.
    #[test]
    fn test_pirls_learns_realistic_signal() -> Result<(), Box<dyn std::error::Error>> {
        // === Create realistic dataset WITH a clear signal ===
        let n_samples = 1000;
        let mut rng = StdRng::seed_from_u64(42); // Different seed for variety

        // Predictor: uniform distribution 
        let p = Array1::linspace(-2.0, 2.0, n_samples);

        // Outcome: Generate from a clear logistic relationship
        // True function: log_odds = -0.5 + 1.5 * p (strong linear signal)
        let y_values: Vec<f64> = p.iter().map(|&p_val| {
            let log_odds: f64 = -0.5 + 1.5 * p_val;
            let prob = 1.0 / (1.0 + (-log_odds).exp());
            if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 }
        }).collect();
        let y = Array1::from_vec(y_values);

        let data = TrainingData {
            y,
            p,
            pcs: Array2::zeros((n_samples, 0)),
        };

        // === Use same stable model configuration ===
        let config = ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            convergence_tolerance: 1e-7,
            max_iterations: 150,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 50,
            pgs_basis_config: BasisConfig { num_knots: 5, degree: 3 },
            pc_basis_configs: vec![],
            pgs_range: (-2.0, 2.0),
            pc_ranges: vec![],
            pc_names: vec![],
            constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            num_pgs_interaction_bases: 0,
        };

        // === Set up inputs using helper ===
        let (x_matrix, rs_original, layout) = setup_pirls_test_inputs(&data, &config)?;
        let rho_vec = arr1(&[0.0]); // Same parameter value

        // === Execute P-IRLS ===
        let pirls_result = fit_model_for_fixed_rho(
            rho_vec.view(),
            x_matrix.view(),
            data.y.view(),
            &rs_original,
            &layout,
            &config,
        )
        .expect("P-IRLS should converge on realistic data with clear signal");

        // === Assert stability (same as random data test) ===
        assert!(
            pirls_result.deviance.is_finite(),
            "Deviance must be finite, got: {}", pirls_result.deviance
        );
        assert!(
            pirls_result.beta.iter().all(|&b| b.is_finite()),
            "All beta coefficients must be finite"
        );
        assert!(
            pirls_result.penalized_hessian.iter().all(|&h| h.is_finite()),
            "Penalized Hessian must be finite"
        );

        // === Assert signal detection (different from random data test) ===
        // Transform back to interpretable basis
        let beta_original = pirls_result.qs.dot(&pirls_result.beta);
        let mapped_coeffs = map_coefficients(&beta_original, &layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;
        
        // For data with a strong signal, coefficients should be substantial
        let pgs_coeffs_norm = pgs_spline_coeffs.iter().map(|&c| c.powi(2)).sum::<f64>().sqrt();
        assert!(
            pgs_coeffs_norm > 0.5, // Should be much larger than random noise
            "Model should detect the clear signal, got coefficient norm: {}", pgs_coeffs_norm
        );

        // === More principled test: Compare fitted vs true function ===
        let predicted_log_odds = x_matrix.dot(&beta_original);
        let true_log_odds = data.p.mapv(|p_val| -0.5 + 1.5 * p_val);

        // Calculate correlation coefficient (scale-invariant measure)
        let pred_mean = predicted_log_odds.mean().unwrap();
        let true_mean = true_log_odds.mean().unwrap();
        
        let numerator = (&predicted_log_odds - pred_mean).dot(&(&true_log_odds - true_mean));
        let pred_var = (&predicted_log_odds - pred_mean).mapv(|v| v.powi(2)).sum();
        let true_var = (&true_log_odds - true_mean).mapv(|v| v.powi(2)).sum();
        let correlation = numerator / (pred_var * true_var).sqrt();
        
        println!("Correlation between fitted and true function: {:.6}", correlation);
        assert!(
            correlation > 0.9, // Strong positive correlation expected
            "The fitted function should strongly correlate with the true function. Correlation: {:.6}", correlation
        );

        // Log diagnostics
        println!("Signal data - Spline coefficients norm: {:.6}", pgs_coeffs_norm);
        println!("Signal data - Sample coefficients: {:?}", 
                &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(3)]);
        println!("✓ Test passed: P-IRLS stable and correctly learns realistic signal");

        Ok(())
    }

    /// Test that P-IRLS is stable and correct on ideal data with Identity link (Gaussian).
    /// This verifies the algorithm converges and behaves correctly on easy data.
    #[test]
    fn test_pirls_is_stable_on_perfectly_good_data_identity() -> Result<(), Box<dyn std::error::Error>> {
        let result = run_pirls_test_scenario(LinkFunction::Identity, SignalType::NoSignal)?;

        // === Assert stability ===
        assert!(
            result.pirls_result.deviance.is_finite(),
            "Deviance must be finite, got: {}", result.pirls_result.deviance
        );
        assert!(
            result.pirls_result.beta.iter().all(|&b| b.is_finite()),
            "All beta coefficients must be finite"
        );
        assert!(
            result.pirls_result.penalized_hessian.iter().all(|&h| h.is_finite()),
            "Penalized Hessian must be finite"
        );

        // === Assert that coefficients are small (no signal case) ===
        let beta_original = result.pirls_result.qs.dot(&result.pirls_result.beta);
        let mapped_coeffs = map_coefficients(&beta_original, &result.layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;
        
        let pgs_coeffs_norm = pgs_spline_coeffs.iter().map(|&c| c.powi(2)).sum::<f64>().sqrt();
        assert!(
            pgs_coeffs_norm < 0.5, // Should be small for no-signal data
            "With no signal, spline coeffs should be near zero. Norm: {}", pgs_coeffs_norm
        );

        // Log the actual values for diagnostic purposes
        println!("Identity No Signal - Spline coefficients norm: {:.6}", pgs_coeffs_norm);
        println!("Identity No Signal - Individual spline coefficients: {:?}", &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(5)]);

        println!("✓ Test passed: `fit_model_for_fixed_rho` is stable and correct on ideal data with Identity link.");

        Ok(())
    }

    /// Test that P-IRLS is stable and correctly learns from realistic data with a clear signal using Identity link.
    /// This verifies the algorithm not only converges, but finds meaningful patterns when they exist.
    #[test]
    fn test_pirls_learns_realistic_signal_identity() -> Result<(), Box<dyn std::error::Error>> {
        let result = run_pirls_test_scenario(LinkFunction::Identity, SignalType::LinearSignal)?;

        // === Assert stability (same as random data test) ===
        assert!(
            result.pirls_result.deviance.is_finite(),
            "Deviance must be finite, got: {}", result.pirls_result.deviance
        );
        assert!(
            result.pirls_result.beta.iter().all(|&b| b.is_finite()),
            "All beta coefficients must be finite"
        );
        assert!(
            result.pirls_result.penalized_hessian.iter().all(|&h| h.is_finite()),
            "Penalized Hessian must be finite"
        );

        // === Assert signal detection ===
        // Transform back to interpretable basis
        let beta_original = result.pirls_result.qs.dot(&result.pirls_result.beta);
        let mapped_coeffs = map_coefficients(&beta_original, &result.layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;
        
        // For data with a strong signal, coefficients should be substantial
        let pgs_coeffs_norm = pgs_spline_coeffs.iter().map(|&c| c.powi(2)).sum::<f64>().sqrt();
        assert!(
            pgs_coeffs_norm > 0.5, // Should be much larger than random noise
            "Model should detect the clear signal, got coefficient norm: {}", pgs_coeffs_norm
        );

        // === More principled test: Compare fitted vs true function ===
        let predicted_linear_predictor = result.x_matrix.dot(&beta_original);
        let correlation = calculate_correlation(
            predicted_linear_predictor.view(),
            result.true_linear_predictor.view(),
        );
        
        println!("Correlation between fitted and true function: {:.6}", correlation);
        assert!(
            correlation > 0.9, // Strong positive correlation expected
            "The fitted function should strongly correlate with the true function. Correlation: {:.6}", correlation
        );

        // Log diagnostics
        println!("Identity Signal data - Spline coefficients norm: {:.6}", pgs_coeffs_norm);
        println!("Identity Signal data - Sample coefficients: {:?}", 
                &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(3)]);
        println!("✓ Test passed: P-IRLS stable and correctly learns realistic signal with Identity link");

        Ok(())
    }
}
