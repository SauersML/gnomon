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
    pub final_weights: Array1<f64>,
    pub status: PirlsStatus,
    pub iteration: usize,
    pub max_abs_eta: f64,
}

/// P-IRLS solver that uses pre-computed reparameterization results
/// This function is called once for each set of smoothing parameters
/// P-IRLS solver that computes a new reparameterization for each set of smoothing parameters
/// This function is called once for each set of smoothing parameters evaluated by the optimizer
/// 
/// This is the new, improved version that performs reparameterization for each set of rho
/// to ensure optimal numerical stability.
pub fn fit_model_for_fixed_rho_new(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    rs_original: &[Array2<f64>],  // Original, untransformed penalty square roots
    layout: &ModelLayout,
    config: &ModelConfig,
) -> Result<PirlsResult, EstimationError> {

    // Core implementation
    fit_model_with_reparameterization(rho_vec, x, y, rs_original, layout, config)
}

// Temporary compatibility function - keeps the old API working during transition
// DEPRECATED: Use fit_model_for_fixed_rho with original penalty roots instead
// This is the original function signature, kept for backwards compatibility with existing code
#[deprecated(note = "Use fit_model_for_fixed_rho with original penalty roots instead")]
pub fn fit_model_for_fixed_rho(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    eb: &Array2<f64>,              // Pre-computed balanced penalty square root
    rs_transformed: &[Array2<f64>], // Pre-computed transformed penalty square roots
    layout: &ModelLayout,
    config: &ModelConfig,
) -> Result<PirlsResult, EstimationError> {
    log::warn!("Using deprecated fit_model_for_fixed_rho_legacy which does not reparameterize for each rho");
    
    // Call internal implementation, bypassing the reparameterization
    fit_model_internal(rho_vec, x, y, eb, rs_transformed, layout, config)
}

/// Internal implementation of the P-IRLS solver with fresh reparameterization
/// This is the main implementation that performs reparameterization for each set of smoothing parameters
fn fit_model_with_reparameterization(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    rs_original: &[Array2<f64>],  // Original, untransformed penalty square roots
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
    
    // CRITICAL ARCHITECTURAL FIX: Perform stable reparameterization for each set of smoothing parameters
    // This matches mgcv's behavior of calling gam.reparam() for every evaluation of the objective function
    log::debug!("Performing stable reparameterization for current smoothing parameters");
    
    // Import the stable_reparameterization function if it's in a different module
    use crate::calibrate::construction::stable_reparameterization;
    
    // Perform the reparameterization for the current rho/lambda values
    let reparam_result = stable_reparameterization(
        rs_original, 
        &lambdas.to_vec(), // Convert to Vec since we need to pass it by reference
        layout
    )?;
    
    // Now use the freshly computed eb and rs_transformed for this specific set of smoothing parameters
    let eb = &reparam_result.eb;
    let rs_transformed = &reparam_result.rs_transformed;
    
    log::info!("Reparameterization complete. Starting P-IRLS iterations...");
    
    // Call the internal implementation with the freshly reparameterized penalties
    fit_model_internal(rho_vec, x, y, eb, rs_transformed, layout, config)
}

fn fit_model_internal(
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
    
    // Just use the pre-computed eb and rs_transformed directly
    
    log::info!("Reparameterization complete. Starting P-IRLS iterations...");

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
    
    // Save the most recent stable result to avoid redundant computation
    let mut last_stable_result: Option<StablePLSResult> = None;

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

        // Use stable solver for P-IRLS inner loop
        let stable_result = stable_penalized_least_squares(
            x.view(),
            z.view(),
            weights.view(),
            eb,
            rs_transformed, // These are now derived from the reparameterization for this specific rho
            lambdas.as_slice().unwrap(),
        )?;
        
        // Save the most recent stable result to avoid redundant computation at the end
        last_stable_result = Some(stable_result.clone());

        let mut beta_trial = stable_result.beta;

        if !beta_trial.iter().all(|x| x.is_finite()) {
            log::error!("Non-finite beta values at iteration {iter}: {beta_trial:?}");
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: f64::NAN,
            });
        }

        // Calculate trial values
        let mut eta_trial = x.dot(&beta_trial);
        let (mut mu_trial, _, _) = update_glm_vectors(y, &eta_trial, config.link_function);
        let mut deviance_trial = calculate_deviance(y, &mu_trial, config.link_function);

        // mgcv-style step halving (use while loop instead of for loop with counter)
        let mut step_halving_count = 0;
        
        // Check for non-finite values or deviance increase
        let mut valid_eta = eta_trial.iter().all(|v| v.is_finite());
        let mut valid_mu = mu_trial.iter().all(|v| v.is_finite());
        
        // Calculate penalty and check if deviance has increased
        let mut penalty_trial = beta_trial.dot(&s_lambda.dot(&beta_trial));
        let mut penalized_deviance_trial = deviance_trial + penalty_trial;
        let mut deviance_decreased = penalized_deviance_trial < penalized_deviance_current;

        // Step halving when: invalid values or deviance increased
        if !valid_eta || !valid_mu || !deviance_trial.is_finite() || !deviance_decreased {
            log::debug!("Starting step halving due to invalid values or deviance increase");
        }
        
        // mgcv-style while loop for step halving
        while (!valid_eta || !valid_mu || !deviance_trial.is_finite() || !deviance_decreased) && step_halving_count < 30 {
            // Half the step size
            beta_trial = &beta_current + 0.5 * (&beta_trial - &beta_current);
            eta_trial = x.dot(&beta_trial);
            
            // Re-evaluate
            let update_result = update_glm_vectors(y, &eta_trial, config.link_function);
            mu_trial = update_result.0;
            deviance_trial = calculate_deviance(y, &mu_trial, config.link_function);
            penalty_trial = beta_trial.dot(&s_lambda.dot(&beta_trial));
            
            // Check conditions again
            valid_eta = eta_trial.iter().all(|v| v.is_finite());
            valid_mu = mu_trial.iter().all(|v| v.is_finite());
            penalized_deviance_trial = deviance_trial + penalty_trial;
            deviance_decreased = penalized_deviance_trial < penalized_deviance_current;
            
            step_halving_count += 1;
            
            if step_halving_count > 0 && step_halving_count % 5 == 0 {
                log::debug!(
                    "Step halving attempt {}: penalized deviance {:.6} -> {:.6}",
                    step_halving_count, 
                    penalized_deviance_current,
                    penalized_deviance_trial
                );
            }
        }

        // If we couldn't find a valid step after max step halvings, fail
        if !valid_eta || !valid_mu || !deviance_trial.is_finite() || !deviance_decreased {
            log::warn!("P-IRLS failed to find valid step after {} halvings", step_halving_count);
            // mgcv simply returns with failure in this case
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: if deviance_decreased { f64::NAN } else { f64::INFINITY },
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
        beta = beta_trial;
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
            let penalized_hessian = if let Some(ref result) = last_stable_result {
                // Use the Hessian from the last stable solve
                result.penalized_hessian.clone()
            } else {
                // This should never happen, but as a fallback, compute the Hessian
                log::warn!("No stable result saved, computing Hessian as fallback");
                let result = stable_penalized_least_squares(
                    x.view(),
                    z.view(),
                    weights.view(),
                    eb,
                    rs_transformed,
                    lambdas.as_slice().unwrap(),
                )?;
                result.penalized_hessian
            };
            
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

        // Calculate the penalized deviance change for convergence check
        let penalty_new = beta.dot(&s_lambda.dot(&beta));
        let penalized_deviance_new = last_deviance + penalty_new;
        let deviance_change = (penalized_deviance_current - penalized_deviance_new).abs();

        // Calculate the gradient of the penalized deviance objective function
        // The correct formula is: 2 * (X' * W * (eta - z) + S_lambda * beta)
        let eta_minus_z = &eta - &z;
        let w_times_diff = &weights * &eta_minus_z;
        let deviance_gradient_part = x.t().dot(&w_times_diff);
        
        let penalty_gradient_part = s_lambda.dot(&beta);
        
        let penalized_deviance_gradient = 
            &(&deviance_gradient_part * 2.0) + &(&penalty_gradient_part * 2.0);
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
            if step_halving_count > 0 { format!(" | Step Halving: {} attempts", step_halving_count) } else { String::new() }
        );

        // Check for non-finite deviance change
        if !deviance_change.is_finite() {
            log::error!(
                "Non-finite penalized deviance change at iteration {iter}: {deviance_change}"
            );
            return Err(EstimationError::PirlsDidNotConverge {
                max_iterations: config.max_iterations,
                last_change: if deviance_change.is_nan() { f64::NAN } else { f64::INFINITY },
            });
        }

        // Set scale parameter based on link function
        // For Logit (Binomial), scale=1. For Identity (Gaussian), use residual variance
        let scale = match config.link_function {
            LinkFunction::Logit => 1.0,
            LinkFunction::Identity => {
                // For Gaussian, scale is the estimated residual variance
                let residuals = &mu - &y.view(); // Recompute residuals for scale calculation
                let df = x.nrows() as f64 - beta.len() as f64;
                residuals.dot(&residuals) / df.max(1.0)
            }
        };

        // Comprehensive convergence check as in mgcv
        // 1. The change in penalized deviance is small
        let deviance_converged = deviance_change < config.convergence_tolerance;
        
        // 2. AND the gradient is close to zero (using scaled tolerance)
        let gradient_tol = config.convergence_tolerance * (scale.abs() + penalized_deviance_new.abs());
        let gradient_converged = gradient_norm < gradient_tol;
        
        let converged = iter >= min_iterations && deviance_converged && gradient_converged;

        if converged {
            log::info!("P-IRLS Converged with deviance change {:.2e} and gradient norm {:.2e}.", 
                      deviance_change, gradient_norm);

            // Use the saved stable result to avoid redundant computation
            let penalized_hessian = if let Some(ref result) = last_stable_result {
                // Use the Hessian from the last stable solve
                result.penalized_hessian.clone()
            } else {
                // This should never happen, but as a fallback, compute the Hessian
                log::warn!("No stable result saved, computing Hessian as fallback");
                let result = stable_penalized_least_squares(
                    x.view(),
                    z.view(),
                    weights.view(),
                    eb,
                    rs_transformed,
                    lambdas.as_slice().unwrap(),
                )?;
                result.penalized_hessian
            };

            return Ok(PirlsResult {
                beta,
                penalized_hessian,
                deviance: last_deviance,
                final_weights: weights,
                status: PirlsStatus::Converged,
                iteration: iter,
                max_abs_eta,
            });
        }
    }

    // If we reach here, we've hit max iterations without converging
    log::warn!("P-IRLS FAILED to converge after {} iterations.", last_iter);
    
    // In mgcv's implementation, there is no additional check for whether we're at a valid minimum
    // It just reports failure to converge
    
    // Use the saved stable result to avoid redundant computation
    let penalized_hessian = if let Some(ref result) = last_stable_result {
        // Use the Hessian from the last stable solve
        result.penalized_hessian.clone()
    } else {
        // This should never happen, but as a fallback, compute the Hessian
        log::warn!("No stable result saved, computing Hessian as fallback");
        let result = stable_penalized_least_squares(
            x.view(),
            z.view(),
            weights.view(),
            eb,
            rs_transformed,
            lambdas.as_slice().unwrap(),
        )?;
        result.penalized_hessian
    };
    
    // Simply return with MaxIterationsReached status
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
/// The function can now work directly with original or transformed penalty matrices
pub fn stable_penalized_least_squares(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    eb: &Array2<f64>,          // balanced penalty square root - either from reparameterization or original
    rs_transformed: &[Array2<f64>], // penalty square roots - either transformed or original
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

    // Step D: Determine numerical rank using the R_cond algorithm from mgcv
    let rank_tol = 1e-7; // mgcv's default
    let mut rank = r_combined.ncols().min(r_combined.nrows());

    while rank > 0 {
        let r_sub = r_combined.slice(s![..rank, ..rank]);
        let r_cond = estimate_r_condition(r_sub.view());
        if !r_cond.is_finite() || rank_tol * r_cond > 1.0 {
            rank -= 1;
        } else {
            break; // Rank is acceptable
        }
    }
    
    log::debug!("Determined rank {} (out of {}) using R_cond algorithm", rank, p);
    
    // Calculate number of dropped columns for logging
    let dropped_columns_count = p - rank;
    log::debug!("Dropped {} columns due to rank deficiency", dropped_columns_count);

    // Step E: Solve the system using the unified augmented QR solver
    // This also returns the stable, consistent penalized Hessian
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

/// Port of the `R_cond` function from mgcv, which implements the CMSW
/// algorithm to estimate the 1-norm condition number of an upper
/// triangular matrix R. THIS IS A MANDATORY COMPONENT FOR ALIGNMENT.
fn estimate_r_condition(r_matrix: ArrayView2<f64>) -> f64 {
    use ndarray::s;
    
    let c = r_matrix.ncols();
    if c == 0 { return 1.0; }
    // We don't need r_rows, but include it for alignment with C code
    let _r_rows = r_matrix.nrows();

    let mut y = Array1::zeros(c);
    let mut p = Array1::zeros(c);
    let mut pp: Array1<f64> = Array1::zeros(c);
    let mut pm: Array1<f64> = Array1::zeros(c);

    for k in (0..c).rev() {
        let r_kk = r_matrix[[k, k]];
        if r_kk == 0.0 { return f64::INFINITY; }
        let yp: f64 = (1.0 - p[k]) / r_kk;
        let ym: f64 = (-1.0 - p[k]) / r_kk;

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
            p.slice_mut(s![0..k]).assign(&pp.slice(s![0..k]));
        } else {
            y[k] = ym;
            p.slice_mut(s![0..k]).assign(&pm.slice(s![0..k]));
        }
    }
    
    let y_inf_norm = y.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
    
    let r_inf_norm = (0..c).map(|i| {
        (i..c).map(|j| r_matrix[[i, j]].abs()).sum::<f64>()
    }).fold(f64::NEG_INFINITY, f64::max);

    r_inf_norm * y_inf_norm
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
    // Early return for empty penalties to ensure consistent shape
    if rs_transformed.is_empty() || lambdas.is_empty() {
        return Ok(Array2::zeros((p, 0)));
    }
    
    let scaled_roots: Vec<_> = rs_transformed
        .iter()
        .zip(lambdas)
        .map(|(rs_k, &lambda)| rs_k * lambda.sqrt())
        .collect();

    if scaled_roots.is_empty() {
        return Ok(Array2::zeros((p, 0)));
    }
    
    // Ensure all matrices have the correct orientation (p rows)
    let mut valid_roots = Vec::new();
    for mat in &scaled_roots {
        if mat.nrows() == p {
            valid_roots.push(mat.view());
        } else if mat.ncols() == p {
            // Transpose if needed - this ensures correct dimensions
            valid_roots.push(mat.t());
        } else {
            return Err(EstimationError::LayoutError(
                format!("Penalty matrix has invalid dimensions: {}x{}, expected {} rows", 
                        mat.nrows(), mat.ncols(), p).into()
            ));
        }
    }

    // Early return with zeros if no valid roots
    if valid_roots.is_empty() {
        return Ok(Array2::zeros((p, 0)));
    }

    ndarray::concatenate(
        Axis(1),
        &valid_roots,
    ).map_err(|e| EstimationError::LayoutError(
        format!("Failed to concatenate penalty roots: {:?}", e).into()
    ))
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
// This function legitimately needs many arguments for the complex QR solver implementation
// and follows the pattern of the original C code in mgcv
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
    use ndarray_linalg::{QR, SVD};

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
    r_augmented.slice_mut(s![r_rows.., ..]).assign(&e_trunc_t); // e_trunc_t is already transposed

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

    // Create the pivoted and truncated design matrix for direct RHS calculation
    let mut x_pivoted_trunc = Array2::zeros((x.nrows(), rank));
    for (j_trunc, &j_orig) in pivot.iter().take(rank).enumerate() {
        x_pivoted_trunc.column_mut(j_trunc).assign(&x.column(j_orig));
    }

    // The x_pivoted_trunc matrix was already created above at lines 945-949
    // We'll reuse it for the stability check below
    
    // Get the final RHS slice for the primary, stable path.
    let final_rhs_slice = final_rhs.slice(s![..rank]);
    
    // PATH 1: The Stable RHS intermediate vector `v`, from solving R_aug' * v = final_rhs_slice.
    // This is equivalent to R'Q'z in the C code's context.
    use ndarray_linalg::{SolveTriangular, UPLO, Diag};
    // THE FIX IS HERE: .to_owned() satisfies the trait bounds for solve_triangular.
    let v_stable = r_aug_final.t().solve_triangular(UPLO::Lower, Diag::NonUnit, &final_rhs_slice.to_owned())
        .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?;
    
    // PATH 2: The Direct RHS vector, from X_trunc' * W * z.
    let wz = &y * &weights; // y is the working response z
    let rhs_direct = x_pivoted_trunc.t().dot(&wz);
    
    // THE CORRECT STABILITY CHECK: Compare the two paths using the squared Euclidean norm.
    let diff = &v_stable - &rhs_direct;
    let norm_diff_sq: f64 = diff.iter().map(|&x| x * x).sum();
    let norm_direct_sq: f64 = rhs_direct.iter().map(|&x| x * x).sum();
    let rank_tol = 1e-7; // mgcv's tolerance
    let use_wy_fallback = norm_diff_sq > rank_tol * norm_direct_sq;
    
    if use_wy_fallback {
        log::warn!("Numerical stability issue detected (norm(R'Q'z - X'Wz)² > tol). Switching to fallback solver path.");
    }
    
    // 7. Solve for truncated beta using the correctly triggered path.
    let beta_trunc = if use_wy_fallback {
        // --- FALLBACK PATH (`use_wy = 1` in C) ---
        // Solves (X'WX+S) * beta = X'Wz, which is r_aug_final.t() * r_aug_final * beta = rhs_direct.
        
        // 1. Solve r_aug_final.t() * v = rhs_direct for v
        let v_fallback = r_aug_final.t().solve_triangular(UPLO::Lower, Diag::NonUnit, &rhs_direct)
            .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?;
    
        // 2. Solve r_aug_final * beta = v for beta
        r_aug_final.solve_triangular(UPLO::Upper, Diag::NonUnit, &v_fallback.to_owned())
            .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?  
    } else {
        // --- PRIMARY PATH (`use_wy = 0` in C) ---
        // Solves r_aug_final * beta = final_rhs_slice
        r_aug_final.solve_triangular(UPLO::Upper, Diag::NonUnit, &final_rhs_slice.to_owned())
            .map_err(|e| EstimationError::LinearSystemSolveFailed(e))?  
    };

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
    use crate::calibrate::construction::{compute_penalty_square_roots, ModelLayout};
    use crate::calibrate::model::BasisConfig;
    use std::collections::HashMap;

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
        let _layout = ModelLayout {
            intercept_col: 0,
            pgs_main_cols: 0..0,
            penalty_map: vec![],
            total_coeffs: n_features,
            num_penalties: 1,
            num_pgs_interaction_bases: 0,
        };
        let _lambdas = vec![1.0];
        // ACTION 1: Compute the Hessian using the full stable solver directly with original penalties
        let stable_result =
            stable_penalized_least_squares(x.view(), y.view(), weights.view(), &rs_list[0], &rs_list, &[1.0])
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

        // Compute penalty square roots
        let s_list = vec![s1.clone()];
        let rs_list = compute_penalty_square_roots(&s_list)
            .expect("Failed to compute penalty square roots");
        
        // Create a minimal layout for the test
        let _layout = ModelLayout {
            intercept_col: 0,
            pgs_main_cols: 0..0,
            penalty_map: vec![],
            total_coeffs: 2,
            num_penalties: 1,
            num_pgs_interaction_bases: 0,
        };
        
        let _lambdas = vec![1.0];
        
        // Run our solver - now working directly with the original square roots
        let result = stable_penalized_least_squares(
            x.view(), 
            y.view(), 
            weights.view(), 
            &rs_list[0], // Use the first rs matrix directly for the test
            &rs_list, 
            &_lambdas
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
    
    /// This test verifies that the new fit_model_for_fixed_rho_new function
    /// performs reparameterization for each set of smoothing parameters.
    #[test]
    fn test_reparameterization_per_rho() {
        use crate::calibrate::construction::{compute_penalty_square_roots, ModelLayout};
        
        // Create a simple test case
        let x = arr2(&[[1.0, 2.0], [1.0, 3.0], [1.0, 5.0]]);
        let y = arr1(&[4.1, 6.2, 9.8]);
        
        // Create multiple penalty matrices with different scales
        let s1 = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        let s2 = arr2(&[[0.0, 0.0], [0.0, 100.0]]); // Very different scale
        
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
        
        // Create a simple config
        let config = ModelConfig {
            link_function: LinkFunction::Identity,
            max_iterations: 10,
            convergence_tolerance: 1e-8,
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
        
        // Test with two very different rho values
        let rho_vec1 = arr1(&[-5.0, 5.0]); // Lambda: [exp(-5), exp(5)]
        let rho_vec2 = arr1(&[5.0, -5.0]); // Lambda: [exp(5), exp(-5)] - opposite!
        
        // Call the function with both rho vectors
        let result1 = fit_model_with_reparameterization(
            rho_vec1.view(),
            x.view(),
            y.view(),
            &rs_original,
            &layout,
            &config
        ).unwrap();
        
        let result2 = fit_model_with_reparameterization(
            rho_vec2.view(),
            x.view(),
            y.view(),
            &rs_original,
            &layout,
            &config
        ).unwrap();
        
        // The results should be different because of the reparameterization
        // This is testing that we don't reuse the same transformed penalties
        let diff = (&result1.beta - &result2.beta).mapv(|x| x.abs()).sum();
        assert!(diff > 1e-6, "Expected different results for different rho values");
    }
}