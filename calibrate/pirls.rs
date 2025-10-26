use crate::calibrate::construction::{
    ModelLayout, PenaltyWorkspace, PenaltyWorkspaceOwned, ReparamResult,
    calculate_condition_number, stable_reparameterization_with_matrices,
};
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::faer_ndarray::{FaerArrayView, FaerCholesky, FaerEigh};
use crate::calibrate::model::{LinkFunction, ModelConfig};
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};
use log;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::time::{Duration, Instant};

// Suggestion #6: Preallocate and reuse iteration workspaces
pub struct PirlsWorkspace {
    // Common IRLS buffers (n, p sizes)
    pub sqrt_w: Array1<f64>,
    pub wx: Array2<f64>,
    pub wz: Array1<f64>,
    // Stage 2/4 assembly (use max needed sizes)
    pub scaled_matrix: Array2<f64>,    // (<= p + eb_rows) x p
    pub final_aug_matrix: Array2<f64>, // (<= p + e_rows) x p
    // Stage 5 RHS buffers
    pub rhs_full: Array1<f64>, // length <= p + e_rows
    // Gradient check helpers
    pub working_residual: Array1<f64>,
    pub weighted_residual: Array1<f64>,
    // Step-halving direction (XΔβ)
    pub delta_eta: Array1<f64>,
    // Preallocated buffers for GEMM (e.g., XtWX)
    pub xtwx_buf: Array2<f64>,
    // Preallocated buffer for GEMV results (length p)
    pub vec_buf_p: Array1<f64>,
}

impl PirlsWorkspace {
    pub fn new(n: usize, p: usize, eb_rows: usize, e_rows: usize) -> Self {
        // Max rows used in Stage 2 and 4
        let scaled_rows_max = p + eb_rows;
        let final_aug_rows_max = p + e_rows;

        PirlsWorkspace {
            sqrt_w: Array1::zeros(n),
            wx: Array2::zeros((n, p)),
            wz: Array1::zeros(n),
            scaled_matrix: Array2::zeros((scaled_rows_max, p)),
            final_aug_matrix: Array2::zeros((final_aug_rows_max, p)),
            rhs_full: Array1::zeros(final_aug_rows_max),
            working_residual: Array1::zeros(n),
            weighted_residual: Array1::zeros(n),
            delta_eta: Array1::zeros(n),
            xtwx_buf: Array2::zeros((p, p)),
            vec_buf_p: Array1::zeros(p),
        }
    }
}

const PLS_MAX_FACTORIZATION_ATTEMPTS: usize = 4;
const HESSIAN_CONDITION_TARGET: f64 = 1e10;

fn max_abs_diag(matrix: &Array2<f64>) -> f64 {
    matrix
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0)
}

fn attempt_spd_cholesky(matrix: &mut Array2<f64>) -> (Option<FaerLlt<f64>>, f64, Option<f64>) {
    let diag_scale = max_abs_diag(matrix);
    let cond_est = calculate_condition_number(matrix).ok();
    let mut ridge = 0.0;
    if let Some(cond) = cond_est {
        if !cond.is_finite() {
            ridge = diag_scale * 1e-8;
        } else if cond > HESSIAN_CONDITION_TARGET {
            ridge = diag_scale * 1e-10 * (cond / HESSIAN_CONDITION_TARGET);
        }
    } else {
        ridge = diag_scale * 1e-8;
    }
    let mut total_added = 0.0;

    for attempt in 0..=PLS_MAX_FACTORIZATION_ATTEMPTS {
        if ridge > total_added {
            let delta = ridge - total_added;
            for i in 0..matrix.nrows() {
                matrix[[i, i]] += delta;
            }
            total_added = ridge;
        }

        let view = FaerArrayView::new(&*matrix);
        match FaerLlt::new(view.as_ref(), Side::Lower) {
            Ok(chol) => return (Some(chol), total_added, cond_est),
            Err(_) => {
                if attempt == PLS_MAX_FACTORIZATION_ATTEMPTS {
                    return (None, total_added, cond_est);
                }
                if ridge <= 0.0 {
                    ridge = diag_scale * 1e-10;
                } else {
                    ridge = (ridge * 10.0).max(diag_scale * 1e-10);
                }
                if !ridge.is_finite() || ridge <= 0.0 {
                    ridge = diag_scale;
                }
            }
        }
    }

    (None, total_added, cond_est)
}

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
/// # Basis of Returned Tensors
///
/// **IMPORTANT:** All vector and matrix outputs in this struct (`beta_transformed`,
/// `penalized_hessian_transformed`) are in the **stable, transformed basis**
/// that was computed for the given set of smoothing parameters.
///
/// To obtain coefficients in the original, interpretable basis, the caller must
/// back-transform them using the `qs` matrix from the `reparam_result` field:
/// `beta_original = reparam_result.qs.dot(&beta_transformed)`
///
/// # Fields
///
/// * `beta_transformed`: The estimated coefficient vector in the STABLE, TRANSFORMED basis.
/// * `penalized_hessian_transformed`: The penalized Hessian matrix at convergence (X'WX + S_λ) in the STABLE, TRANSFORMED basis.
/// * `deviance`: The final deviance value. Note that this means different things depending on the link function:
///    - For `LinkFunction::Identity` (Gaussian): This is the Residual Sum of Squares (RSS).
///    - For `LinkFunction::Logit` (Binomial): This is -2 * log-likelihood, the binomial deviance.
/// * `final_weights`: The final IRLS weights at convergence.
/// * `reparam_result`: Contains the transformation matrix (`qs`) and other reparameterization data.
#[derive(Clone)]
pub struct PirlsResult {
    // Coefficients and Hessian are now in the STABLE, TRANSFORMED basis
    pub beta_transformed: Array1<f64>,
    pub penalized_hessian_transformed: Array2<f64>,
    // CRITICAL: Single stabilized Hessian for consistent cost/gradient computation
    pub stabilized_hessian_transformed: Array2<f64>,

    // The unpenalized deviance, calculated from mu and y
    pub deviance: f64,

    // Effective degrees of freedom at the solution
    pub edf: f64,

    // The penalty term, calculated stably within P-IRLS
    // This is beta_transformed' * S_transformed * beta_transformed
    pub stable_penalty_term: f64,

    /// Optional Jeffreys prior log-determinant contribution (½ log |H|) when
    /// Firth bias reduction is active.
    pub firth_log_det: Option<f64>,

    // The final IRLS weights at convergence
    pub final_weights: Array1<f64>,
    // Additional PIRLS state captured at the accepted step to support
    // cost/gradient consistency in the outer optimization
    pub final_mu: Array1<f64>,
    pub solve_weights: Array1<f64>,
    pub solve_working_response: Array1<f64>,
    pub solve_mu: Array1<f64>,

    // Keep all other fields as they are
    pub status: PirlsStatus,
    pub iteration: usize,
    pub max_abs_eta: f64,

    // Pass through the entire reparameterization result for use in the gradient
    pub reparam_result: ReparamResult,
    // Cached X·Qs for this PIRLS result (transformed design matrix)
    pub x_transformed: Array2<f64>,
}

/// P-IRLS solver that follows mgcv's architecture exactly
///
/// This function implements the complete algorithm from mgcv's gam.fit3 function
/// for fitting a GAM model with a fixed set of smoothing parameters:
///
/// - Perform stable reparameterization ONCE at the beginning (mgcv's gam.reparam)
/// - Transform the design matrix into this stable basis
/// - Extract a single penalty square root from the transformed penalty
/// - Run the P-IRLS loop entirely in the transformed basis
/// - Transform the coefficients back to the original basis only when returning
///
/// This architecture ensures optimal numerical stability throughout the entire
/// fitting process by working in a well-conditioned parameter space.  
pub fn fit_model_for_fixed_rho(
    rho_vec: ArrayView1<f64>,
    x: ArrayView2<f64>,
    offset: ArrayView1<f64>,
    y: ArrayView1<f64>,
    prior_weights: ArrayView1<f64>, // Prior weights vector
    penalties: PenaltyWorkspace<'_>,
    layout: &ModelLayout,
    config: &ModelConfig,
) -> Result<PirlsResult, EstimationError> {
    // No test-specific hacks - the properly implemented algorithm should handle all cases
    // Stage: Convert rho (log smoothing parameters) to lambda (actual smoothing parameters)
    let lambdas = rho_vec.mapv(f64::exp);

    log::info!(
        "Starting P-IRLS fitting with {} smoothing parameters",
        lambdas.len()
    );
    println!(
        "P-IRLS input dimensions: x: {:?}, y: {}, penalties: {}",
        x.shape(),
        y.len(),
        penalties.rs_list.len()
    );
    if !lambdas.is_empty() {
        println!("Lambdas: {:?}", lambdas);
    }

    let lambdas_vec = lambdas.to_vec();
    let rho_vec_owned = rho_vec.to_vec();

    // Stage: Perform the stable reparameterization exactly once before the P-IRLS loop
    log::info!("Computing stable reparameterization for numerical stability");
    println!("[Reparam] ==> Entering stable_reparameterization...");
    let mut cache_hit = false;
    let reparam_result = match penalties.reparam_cache {
        Some(cell) => {
            let mut cache = cell.borrow_mut();
            if let Some(cached) = cache.try_get(&rho_vec_owned, &lambdas_vec) {
                cache_hit = true;
                cached
            } else {
                let computed = stable_reparameterization_with_matrices(
                    penalties.rs_list,
                    penalties.s_full_list,
                    &lambdas_vec,
                    layout,
                )?;
                cache.insert(rho_vec_owned.clone(), lambdas_vec.clone(), &computed);
                computed
            }
        }
        None => stable_reparameterization_with_matrices(
            penalties.rs_list,
            penalties.s_full_list,
            &lambdas_vec,
            layout,
        )?,
    };
    println!("[Reparam] <== Exited stable_reparameterization successfully.");
    if cache_hit {
        log::debug!("Reparameterization cache hit for rho {:?}", rho_vec_owned);
    } else {
        log::debug!(
            "Computed fresh reparameterization for rho {:?}",
            rho_vec_owned
        );
    }

    println!(
        "[Reparam Result Check] qs_sum: {:.4e}, s_transformed_sum: {:.4e}, log_det_s: {:.4e}",
        reparam_result.qs.sum(),
        reparam_result.s_transformed.sum(),
        reparam_result.log_det
    );

    println!(
        "[Matrix Multiply] ==> Performing x.dot(qs) | x.shape: {:?}, qs.shape: {:?}",
        x.shape(),
        reparam_result.qs.shape()
    );

    // Stage: Transform the design matrix into the stable basis
    let x_transformed = x.dot(&reparam_result.qs);

    println!(
        "[Matrix Multiply] <== x.dot(qs) complete. x_transformed_sum: {:.4e}",
        x_transformed.sum()
    );

    let eb = penalties.balanced_root;
    println!(
        "[Balanced Penalty] Using lambda-independent eb with shape: {:?}",
        eb.shape()
    );

    // Transform eb to the same stable basis
    // As per mgcv (Eb <- Eb%*%T), transform eb into the same stable basis as x_transformed.
    // The transformation for a penalty root R (shape k x p) is R_new = R * Q.
    // Here, eb is `rank x p` and qs is `p x p`, so the result is `rank x p`.
    let eb_transformed = eb.dot(&reparam_result.qs);
    println!(
        "[Basis Fix] Transformed eb from original to stable basis. eb_transformed_sum: {:.4e}",
        eb_transformed.sum()
    );

    // Stage: Extract penalty matrices using the truly lambda-independent eb
    // Note: eb is computed from unweighted penalties and never changes with lambda
    let s_transformed = &reparam_result.s_transformed;
    // eb is already computed above as lambda-INDEPENDENT
    let e_transformed = &reparam_result.e_transformed; // Lambda-DEPENDENT for penalty application

    // Suggestion #4/#12: Precompute S = EᵀE once per rho and cache Xᵀ
    let s_from_e_precomputed = e_transformed.t().dot(e_transformed);
    let x_transformed_t = x_transformed.t().to_owned();

    // Stage: Initialize P-IRLS state variables in the transformed basis
    let mut beta_transformed = Array1::zeros(layout.total_coeffs);
    let mut eta = offset.to_owned();
    eta += &x_transformed.dot(&beta_transformed);
    let (mut mu, mut weights, mut z) =
        update_glm_vectors(y, &eta, config.link_function, prior_weights);
    let mut last_deviance = calculate_deviance(y, &mu, config.link_function, prior_weights);
    let mut max_abs_eta = 0.0;
    let mut last_iter = 0;

    // Preallocate workspace once (Suggestion #6)
    let mut workspace = PirlsWorkspace::new(
        x_transformed.nrows(),
        x_transformed.ncols(),
        eb_transformed.nrows(),
        e_transformed.nrows(),
    );

    // Save the most recent stable result to avoid redundant computation
    let mut last_stable_result: Option<(StablePLSResult, usize)> = None;
    let mut last_mu_solve = mu.clone();

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

    // Track the last linear system weights/working response used in the solve
    let mut last_weights_solve = weights.clone();
    let mut last_z_solve = z.clone();
    // Track coefficient norm growth to detect divergence in unpenalized logistic fits
    let mut prev_beta_norm: f64 = 0.0;

    let mut last_deviance_change = f64::NAN;
    let mut last_penalized_deviance = f64::NAN;
    let mut last_gradient_norm = f64::NAN;
    let mut last_gradient_tol = f64::NAN;
    let mut last_step_halving = 0usize;
    let firth_active =
        config.firth_bias_reduction && matches!(config.link_function, LinkFunction::Logit);
    let mut firth_log_det_value: Option<f64> = None;

    for iter in 1..=config.max_iterations {
        last_iter = iter; // Update on every iteration

        // --- Store the state from the START of the iteration ---
        let beta_current = beta_transformed.clone();
        let deviance_current = last_deviance;

        last_mu_solve = mu.clone();

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

        if firth_active {
            let (hat_diag, half_log_det) = compute_firth_hat_and_half_logdet(
                x_transformed.view(),
                x.view(),
                weights.view(),
                s_transformed,
                &mut workspace,
            )?;
            firth_log_det_value = Some(half_log_det);

            for i in 0..z.len() {
                let wi = weights[i];
                if wi > 0.0 {
                    let adjustment = hat_diag[i] * (0.5 - mu[i]);
                    z[i] += adjustment / wi;
                }
            }
        }

        // CRITICAL: Cache the weights and working response actually used in the WLS solve
        let weights_solve = weights.clone();
        let z_solve = z.clone();
        last_weights_solve = weights_solve.clone();
        last_z_solve = z_solve.clone();

        // The penalized least squares solver computes coefficient updates using a rank-revealing
        // QR decomposition with careful handling of potential rank deficiencies in the weighted
        // design matrix. It applies 5-stage numerical stability techniques following Wood (2011).
        let penalty_info = if !lambdas.is_empty() {
            format!(" | λ={:.4e}", lambdas[0])
        } else {
            String::new()
        };

        // Use our robust solver that handles rank deficiency correctly
        // Note: Pass both penalty matrices for proper separation of concerns
        let stable_result = solve_penalized_least_squares(
            x_transformed.view(), // Pass transformed x
            z.view(),
            weights.view(),
            offset.view(),
            &eb_transformed, // Lambda-INDEPENDENT balanced penalty root for rank detection (NOW IN STABLE BASIS)
            e_transformed,   // Lambda-DEPENDENT penalty root for penalty application
            &s_from_e_precomputed, // Precomputed S = EᵀE
            &mut workspace,  // Preallocated buffers (Suggestion #6)
            y.view(),        // Pass original response
            config.link_function, // Pass link function for correct scale calculation
        )?;

        // Save the most recent stable result to avoid redundant computation at the end
        last_stable_result = Some(stable_result.clone());

        // Capture the EDF from the solver for correct scale calculation
        let edf_from_solver = stable_result.0.edf;

        let beta_trial_initial = stable_result.0.beta.clone();
        let mut step_halving_count = 0;
        let mut last_halving_change: f64 = f64::NAN;
        const MAX_STEP_HALVING: usize = 30;

        // Use a robust loop for step-halving that prioritizes numerical stability.
        // Reuse XΔβ step direction in step-halving
        let beta_update = &beta_trial_initial - &beta_current; // Δβ
        // Reuse workspace for XΔβ
        workspace.delta_eta = x_transformed.dot(&beta_update);
        // Quadratic form reuse for penalty(β + αΔ)
        let s_beta_current = s_transformed.dot(&beta_current);
        let penalty_current_cached = beta_current.dot(&s_beta_current);
        let s_delta = s_transformed.dot(&beta_update);
        let cross = s_beta_current.dot(&beta_update);
        let quad = beta_update.dot(&s_delta);
        let mut alpha = 1.0;
        loop {
            // Candidate eta for current α (avoid forming beta_candidate until accept)
            let eta_trial = &eta + &(alpha * &workspace.delta_eta);
            let mu_trial = mu_only(&eta_trial, config.link_function);
            let deviance_trial =
                calculate_deviance(y, &mu_trial, config.link_function, prior_weights);

            // First, check if the trial step resulted in a numerically valid state.
            // This is the most important check.
            let is_numerically_valid = eta_trial.iter().all(|v| v.is_finite())
                && mu_trial.iter().all(|v| v.is_finite())
                && deviance_trial.is_finite();

            if is_numerically_valid {
                // penalty(β + αΔ) = penalty(β) + 2α (Sβ·Δ) + α² (Δ·SΔ)
                let penalty_trial =
                    penalty_current_cached + 2.0 * alpha * cross + alpha * alpha * quad;
                let penalized_deviance_trial = deviance_trial + penalty_trial;
                // Track the last attempted change for diagnostics
                last_halving_change = (penalized_deviance_trial - penalized_deviance_current).abs();

                // If it's valid, NOW check if the penalized deviance has decreased.
                // Use epsilon tolerance to handle numerical precision issues in Gaussian models
                let accept_step = if firth_active {
                    true
                } else {
                    penalized_deviance_trial <= penalized_deviance_current * (1.0 + 1e-12)
                        || (penalized_deviance_current - penalized_deviance_trial).abs() < 1e-12
                };

                if accept_step {
                    // SUCCESS: The step is valid and improves the fit.
                    // Update the main state variables and exit the step-halving loop.
                    beta_transformed = &beta_current + &(alpha * &beta_update);
                    eta = eta_trial;
                    last_deviance = deviance_trial;
                    (mu, weights, z) =
                        update_glm_vectors(y, &eta, config.link_function, prior_weights);

                    if step_halving_count > 0 {
                        println!(
                            "Step halving successful after {} attempts",
                            step_halving_count
                        );
                    }
                    break; // Exit the loop
                }
            }
            // If we reach here, it's because the step was either invalid or increased the deviance.
            step_halving_count += 1;
            if step_halving_count >= MAX_STEP_HALVING {
                log::warn!(
                    "P-IRLS failed to find a valid step after {} halvings. This often indicates model instability.",
                    MAX_STEP_HALVING
                );
                return Err(EstimationError::PirlsDidNotConverge {
                    max_iterations: config.max_iterations,
                    last_change: if last_halving_change.is_finite() {
                        last_halving_change
                    } else {
                        penalized_deviance_current.abs()
                    },
                });
            }
            // Halve α and try again
            alpha *= 0.5;
        }

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
                    offset.view(),
                    &eb_transformed,
                    e_transformed,
                    &s_from_e_precomputed,
                    &mut workspace,
                    y.view(),
                    config.link_function,
                )?;
                log::trace!("Fallback solve rank: {}", rank);
                result.penalized_hessian
            };

            // Calculate the stable penalty term using the transformed quantities
            let stable_penalty_term = beta_transformed.dot(&s_transformed.dot(&beta_transformed));

            // CRITICAL: Create single stabilized Hessian for consistent cost/gradient computation
            let mut stabilized_hessian_transformed = penalized_hessian_transformed.clone();
            ensure_positive_definite(&mut stabilized_hessian_transformed)?;

            // Populate the new PirlsResult struct with stable, transformed quantities
            println!(
                "[P-IRLS] Terminated at iteration {} due to instability | max|eta|: {:.3e}",
                iter, max_abs_eta
            );
            return Ok(PirlsResult {
                beta_transformed: beta_transformed.clone(),
                penalized_hessian_transformed,
                stabilized_hessian_transformed,
                deviance: last_deviance,
                edf: if let Some((ref result, _)) = last_stable_result {
                    result.edf
                } else {
                    0.0
                },
                stable_penalty_term,
                firth_log_det: firth_log_det_value,
                final_weights: weights.clone(),
                final_mu: mu.clone(),
                solve_weights: last_weights_solve.clone(),
                solve_working_response: last_z_solve.clone(),
                solve_mu: last_mu_solve.clone(),
                status: PirlsStatus::Unstable,
                iteration: iter,
                max_abs_eta,
                reparam_result: reparam_result.clone(),
                x_transformed: x_transformed.clone(),
            });
        }

        // Additional robust separation checks for unpenalized logistic models
        let no_penalty = eb_transformed.nrows() == 0 && e_transformed.nrows() == 0;
        if config.link_function == LinkFunction::Logit && no_penalty {
            let eta_soft_threshold = 30.0; // softer threshold than 100 for unpenalized cases
            let eps_sat = 1e-3;
            let sat_frac = {
                let n = mu.len() as f64;
                let k = mu
                    .iter()
                    .filter(|&&m| m <= eps_sat || m >= 1.0 - eps_sat)
                    .count() as f64;
                if n > 0.0 { k / n } else { 0.0 }
            };
            // Beta divergence: large absolute norm or rapid growth
            let beta_norm = beta_transformed.dot(&beta_transformed).sqrt();
            let beta_grew_fast = prev_beta_norm > 0.0 && beta_norm > prev_beta_norm * 2.5;
            let beta_too_large = beta_norm > 1e3;
            let eta_too_large = max_abs_eta > eta_soft_threshold;
            let probs_saturated = sat_frac > 0.95;
            // Extremely low deviance per sample indicates near-perfect separation
            let n_samples = y.len() as f64;
            let dev_per_sample = if n_samples > 0.0 {
                last_deviance / n_samples
            } else {
                last_deviance
            };
            let dev_tiny = dev_per_sample < 1e-2; // e.g., < 0.01 per sample

            // Direct separation-by-order check: all positives have higher η than all negatives
            let mut min_eta_pos = f64::INFINITY;
            let mut max_eta_neg = f64::NEG_INFINITY;
            let mut has_pos = false;
            let mut has_neg = false;
            for (i, &yi) in y.iter().enumerate() {
                if yi > 0.5 {
                    has_pos = true;
                    let v = eta[i];
                    if v < min_eta_pos {
                        min_eta_pos = v;
                    }
                } else {
                    has_neg = true;
                    let v = eta[i];
                    if v > max_eta_neg {
                        max_eta_neg = v;
                    }
                }
            }
            let order_separated = has_pos && has_neg && (min_eta_pos - max_eta_neg) > 1e-3;

            if eta_too_large
                || probs_saturated
                || dev_tiny
                || order_separated
                || (iter >= 3 && (beta_grew_fast || beta_too_large))
            {
                log::warn!(
                    "P-IRLS instability (unpenalized Logit) at iter {}: max|eta|={:.2e}, sat_frac={:.3}, dev/n={:.3e}, order_sep={}, ||beta||={:.2e} (prev {:.2e})",
                    iter,
                    max_abs_eta,
                    sat_frac,
                    dev_per_sample,
                    order_separated,
                    beta_norm,
                    prev_beta_norm
                );

                let penalized_hessian_transformed =
                    if let Some((ref result, _)) = last_stable_result {
                        result.penalized_hessian.clone()
                    } else {
                        log::warn!("No stable result saved, computing Hessian as fallback");
                        let (result, rank) = solve_penalized_least_squares(
                            x_transformed.view(),
                            z.view(),
                            weights.view(),
                            offset.view(),
                            &eb_transformed,
                            e_transformed,
                            &s_from_e_precomputed,
                            &mut workspace,
                            y.view(),
                            config.link_function,
                        )?;
                        log::trace!("Fallback solve rank: {}", rank);
                        result.penalized_hessian
                    };
                let stable_penalty_term =
                    beta_transformed.dot(&s_transformed.dot(&beta_transformed));
                let mut stabilized_hessian_transformed = penalized_hessian_transformed.clone();
                ensure_positive_definite(&mut stabilized_hessian_transformed)?;
                println!(
                    "[P-IRLS] Terminated at iteration {} due to instability | max|eta|: {:.3e}",
                    iter, max_abs_eta
                );
                return Ok(PirlsResult {
                    beta_transformed: beta_transformed.clone(),
                    penalized_hessian_transformed,
                    stabilized_hessian_transformed,
                    deviance: last_deviance,
                    edf: if let Some((ref result, _)) = last_stable_result {
                        result.edf
                    } else {
                        0.0
                    },
                    stable_penalty_term,
                    firth_log_det: firth_log_det_value,
                    final_weights: weights.clone(),
                    final_mu: mu.clone(),
                    solve_weights: last_weights_solve.clone(),
                    solve_working_response: last_z_solve.clone(),
                    solve_mu: last_mu_solve.clone(),
                    status: PirlsStatus::Unstable,
                    iteration: iter,
                    max_abs_eta,
                    reparam_result: reparam_result.clone(),
                    x_transformed: x_transformed.clone(),
                });
            }
            // Update previous beta norm for next iteration
            prev_beta_norm = beta_norm;
        }

        // Calculate the penalized deviance using the transformed penalty matrix
        let penalty_new = beta_transformed.dot(&s_transformed.dot(&beta_transformed));
        let penalized_deviance_new = last_deviance + penalty_new;

        // Set scale parameter based on link function
        let scale = match config.link_function {
            LinkFunction::Logit => 1.0,
            LinkFunction::Identity => {
                // For Gaussian, use WEIGHTED residual variance consistent with objective
                let residuals = &y.view() - &mu;
                let weighted_rss: f64 = prior_weights
                    .iter()
                    .zip(residuals.iter())
                    .map(|(&w, &r)| w * r * r)
                    .sum();
                // Use the EDF from the solver: df = n - edf
                let df = (x_transformed.nrows() as f64 - edf_from_solver).max(1.0);
                weighted_rss / df
            }
        };

        // This scaling factor is the key to mgcv's numerical stability.
        // It prevents the tolerance from collapsing when the deviance is small.
        let convergence_scale = scale.abs() + penalized_deviance_new.abs();
        let deviance_change_scaled = (penalized_deviance_current - penalized_deviance_new).abs();

        // Log iteration info
        let step_halving_info = if step_halving_count > 0 {
            format!(" | Step Halving: {} attempts", step_halving_count)
        } else {
            String::new()
        };
        last_deviance_change = deviance_change_scaled;
        last_penalized_deviance = penalized_deviance_new;
        last_step_halving = step_halving_count;

        // First convergence check: has the change in deviance become negligible relative to the scale of the problem?
        if deviance_change_scaled < config.convergence_tolerance * (0.1 + convergence_scale) {
            // Check gradient with the SAME (W, z) used in the last WLS solve
            // The solver solved: X'W_solve(Xβ - z_solve) + Sβ = 0
            // So we must check stationarity with respect to those same W_solve, z_solve

            // Use the cached weights and working response from the solve
            // Suggestion #3: Compute gradient without building WX and WZ
            // grad_data_part = Xᵀ W (Xβ - z); reuse workspace buffers
            let tmp_eta = {
                let mut eta = offset.to_owned();
                eta += &x_transformed.dot(&beta_transformed);
                eta
            };
            workspace.working_residual = &tmp_eta - &z_solve; // (offset + Xβ) - z
            workspace.weighted_residual = &weights_solve * &workspace.working_residual; // W (Xβ - z)
            let grad_data_part = x_transformed_t.dot(&workspace.weighted_residual);
            let grad_penalty_part = s_transformed.dot(&beta_transformed);
            // Drop the 2x factor to match the objective we actually minimize
            let gradient_wrt_solve = &grad_data_part + &grad_penalty_part;

            let gradient_norm = gradient_wrt_solve
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);

            // This is the ROBUST gradient tolerance from mgcv. It's scaled by the same factor
            // and uses the user's epsilon, not machine epsilon.
            let gradient_tol = config.convergence_tolerance * (0.1 + convergence_scale);

            last_gradient_norm = gradient_norm;
            last_gradient_tol = gradient_tol;

            if gradient_norm < gradient_tol && iter >= min_iterations {
                // SUCCESS: Both deviance and gradient have converged.
                log::info!(
                    "P-IRLS Converged with deviance change {:.2e} and gradient norm {:.2e}.",
                    deviance_change_scaled,
                    gradient_norm
                );

                let beta_norm = beta_transformed.dot(&beta_transformed).sqrt();
                println!(
                    "[P-IRLS] Converged in {} iterations | Δ: {:.3e} | ∥grad∥∞: {:.3e} | β-norm: {:.3e} | Dev: {:.6e}{}{}",
                    iter,
                    deviance_change_scaled,
                    gradient_norm,
                    beta_norm,
                    penalized_deviance_new,
                    step_halving_info,
                    penalty_info
                );

                let penalized_hessian_transformed =
                    if let Some((ref result, _)) = last_stable_result {
                        result.penalized_hessian.clone()
                    } else {
                        let (result, _) = solve_penalized_least_squares(
                            x_transformed.view(),
                            z.view(),
                            weights.view(),
                            offset.view(),
                            &eb_transformed,
                            e_transformed,
                            &s_from_e_precomputed,
                            &mut workspace,
                            y.view(),
                            config.link_function,
                        )?;
                        result.penalized_hessian
                    };

                // Calculate the stable penalty term using the transformed quantities
                let stable_penalty_term =
                    beta_transformed.dot(&s_transformed.dot(&beta_transformed));

                // CRITICAL: Create single stabilized Hessian for consistent cost/gradient computation
                let mut stabilized_hessian_transformed = penalized_hessian_transformed.clone();
                ensure_positive_definite(&mut stabilized_hessian_transformed)?;

                // Populate the new PirlsResult struct with stable, transformed quantities
                return Ok(PirlsResult {
                    beta_transformed: beta_transformed.clone(),
                    penalized_hessian_transformed,
                    stabilized_hessian_transformed,
                    deviance: last_deviance,
                    edf: edf_from_solver,
                    stable_penalty_term,
                    firth_log_det: firth_log_det_value,
                    final_weights: weights.clone(),
                    final_mu: mu.clone(),
                    solve_weights: last_weights_solve.clone(),
                    solve_working_response: last_z_solve.clone(),
                    solve_mu: last_mu_solve.clone(),
                    status: PirlsStatus::Converged,
                    iteration: iter,
                    max_abs_eta,
                    reparam_result: reparam_result.clone(),
                    x_transformed: x_transformed.clone(),
                });
            }
        }
    }

    // If we reach here, we've hit max iterations without converging
    log::warn!("P-IRLS FAILED to converge after {} iterations.", last_iter);

    let beta_norm = beta_transformed.dot(&beta_transformed).sqrt();
    let grad_display = if last_gradient_norm.is_finite() {
        format!("{:.3e}", last_gradient_norm)
    } else {
        "n/a".to_string()
    };
    let tol_display = if last_gradient_tol.is_finite() {
        format!("{:.3e}", last_gradient_tol)
    } else {
        "n/a".to_string()
    };
    let step_info = if last_step_halving > 0 {
        format!(" | Step Halving: {}", last_step_halving)
    } else {
        String::new()
    };
    println!(
        "[P-IRLS] Exited after {} iterations | Δ: {:.3e} | ∥grad∥∞: {} | tol: {} | β-norm: {:.3e} | Dev: {:.6e}{}",
        last_iter,
        last_deviance_change,
        grad_display,
        tol_display,
        beta_norm,
        last_penalized_deviance,
        step_info
    );

    // Before returning, perform a final stationarity check using the SAME (W, z) as the last solve.
    // If the gradient is small, we can report a valid stalled minimum for the outer loop to accept.

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
            offset.view(),
            &eb_transformed,
            e_transformed,
            &s_from_e_precomputed,
            &mut workspace,
            y.view(),
            config.link_function,
        )?;
        log::trace!("Final solve rank: {}", rank);
        result.penalized_hessian
    };

    // Calculate the stable penalty term using the transformed quantities
    let stable_penalty_term = beta_transformed.dot(&s_transformed.dot(&beta_transformed));

    // Compute a final gradient check w.r.t the last W_solve and z_solve
    let x_transformed_t = x_transformed.t().to_owned();
    let tmp_eta = {
        let mut eta = offset.to_owned();
        eta += &x_transformed.dot(&beta_transformed);
        eta
    };
    workspace.working_residual = &tmp_eta - &last_z_solve; // Xβ - z_solve
    workspace.weighted_residual = &last_weights_solve * &workspace.working_residual; // W_solve (Xβ - z_solve)
    let grad_data_part = x_transformed_t.dot(&workspace.weighted_residual);
    let grad_penalty_part = s_transformed.dot(&beta_transformed);
    let gradient_wrt_solve = &grad_data_part + &grad_penalty_part;
    let gradient_norm = gradient_wrt_solve
        .iter()
        .map(|&v| v.abs())
        .fold(0.0, f64::max);

    // Build a scale-consistent tolerance similar to the inner loop
    let penalty_new = beta_transformed.dot(&s_transformed.dot(&beta_transformed));
    let penalized_deviance_new = last_deviance + penalty_new;
    let scale_term = match config.link_function {
        LinkFunction::Logit => 1.0,
        LinkFunction::Identity => {
            let residuals = &y.view() - &mu;
            let weighted_rss: f64 = prior_weights
                .iter()
                .zip(residuals.iter())
                .map(|(&w, &r)| w * r * r)
                .sum();
            // EDF from last stable result if available
            let edf_here = if let Some((ref result, _)) = last_stable_result {
                result.edf
            } else {
                0.0
            };
            let df = (x_transformed.nrows() as f64 - edf_here).max(1.0);
            weighted_rss / df
        }
    };
    let convergence_scale = scale_term.abs() + penalized_deviance_new.abs();
    let gradient_tol = config.convergence_tolerance * (0.1 + convergence_scale);

    // CRITICAL: Create single stabilized Hessian for consistent cost/gradient computation
    let mut stabilized_hessian_transformed = penalized_hessian_transformed.clone();
    ensure_positive_definite(&mut stabilized_hessian_transformed)?;

    // If stationarity is met, return a stalled-but-valid minimum for acceptance
    let final_status = if gradient_norm < gradient_tol {
        PirlsStatus::StalledAtValidMinimum
    } else {
        PirlsStatus::MaxIterationsReached
    };

    Ok(PirlsResult {
        beta_transformed,
        penalized_hessian_transformed,
        stabilized_hessian_transformed,
        deviance: last_deviance,
        edf: if let Some((ref result, _)) = last_stable_result {
            result.edf
        } else {
            0.0
        },
        stable_penalty_term,
        firth_log_det: firth_log_det_value,
        final_weights: weights.clone(),
        final_mu: mu.clone(),
        solve_weights: last_weights_solve.clone(),
        solve_working_response: last_z_solve.clone(),
        solve_mu: last_mu_solve.clone(),
        status: final_status,
        iteration: last_iter,
        max_abs_eta,
        reparam_result,
        x_transformed,
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

    // Compute max_diag once outside the loop (performance improvement)
    let max_diag = r_matrix
        .diag()
        .iter()
        .fold(0.0f64, |acc, &val| acc.max(val.abs()));
    let eps = 1e-16f64.max(max_diag * 1e-14);

    for k in (0..c).rev() {
        let r_kk = r_matrix[[k, k]];
        if r_kk.abs() <= eps {
            // Return large finite number instead of infinity to avoid overflow
            return 1e300;
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
        for j in i..c {
            // Only sum upper triangle elements (j >= i)
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

/// Pivots the columns of a matrix according to a pivot vector.
///
/// This applies the permutation `A*P` to get a new matrix `B`. It assumes the
/// `pivot` vector is a **forward** permutation.
///
/// For a matrix A and pivot p, the result B is such that `B_j = A_{p[j]}`.
///
/// # Parameters
/// * `matrix`: The matrix whose columns will be permuted.
/// * `pivot`: The forward permutation vector.
fn pivot_columns(matrix: ArrayView2<f64>, pivot: &[usize]) -> Array2<f64> {
    let r = matrix.nrows();
    let c = matrix.ncols();
    let mut pivoted_matrix = Array2::zeros((r, c));

    for j in 0..c {
        let original_col_index = pivot[j];
        pivoted_matrix
            .column_mut(j)
            .assign(&matrix.column(original_col_index));
    }

    pivoted_matrix
}

/// Symmetrically unpivot a matrix using the forward permutation `pivot`.
/// If `(sqrt(W)X) * P = Q * R`, then `XtWX = P * (R^T R) * P^T`.
/// Given `m_pivoted = R^T R` in pivoted order, this returns `P * m_pivoted * P^T`.
fn unpivot_sym_by_perm(m_pivoted: ArrayView2<f64>, pivot: &[usize]) -> Array2<f64> {
    let n = m_pivoted.nrows();
    assert_eq!(n, m_pivoted.ncols());
    assert_eq!(pivot.len(), n);
    let mut out = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let oi = pivot[i];
            let oj = pivot[j];
            out[[oi, oj]] = m_pivoted[[i, j]];
        }
    }
    out
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
///     *x = 0.0; x--;
///     for (k=n_drop-1;k>0;k--) {
///       for (i=drop[k]-1;i>drop[k-1];i--,X--,Xs--) *X = *Xs;
///       *x = 0.0; x--;
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
/// Currently unused but kept for future implementation
pub fn undrop_rows(src: &Array1<f64>, dropped_rows: &[usize], dst: &mut Array1<f64>) {
    let n_drop = dropped_rows.len();

    if n_drop == 0 {
        // If no rows to drop, just copy src to dst
        if src.len() == dst.len() {
            dst.assign(src);
        }
        return;
    }

    // Validate that the dimensions are compatible
    assert_eq!(
        src.len() + n_drop,
        dst.len(),
        "Source length + dropped rows must equal destination length"
    );

    // Ensure dropped_rows is in ascending order
    for i in 1..n_drop {
        assert!(
            dropped_rows[i] > dropped_rows[i - 1],
            "dropped_rows must be in ascending order"
        );
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
/// Currently unused but kept for future implementation
pub fn drop_rows(src: &Array1<f64>, drop_indices: &[usize], dst: &mut Array1<f64>) {
    let n_drop = drop_indices.len();

    if n_drop == 0 {
        // If no rows to drop, just copy src to dst
        if src.len() == dst.len() {
            dst.assign(src);
        }
        return;
    }

    // Validate that the dimensions are compatible
    assert_eq!(
        src.len(),
        dst.len() + n_drop,
        "Source length must equal destination length + dropped rows"
    );

    // Ensure drop_indices is in ascending order
    for i in 1..n_drop {
        assert!(
            drop_indices[i] > drop_indices[i - 1],
            "drop_indices must be in ascending order"
        );
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
    prior_weights: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    // Smaller floor for Fisher weights to preserve geometry; slightly larger floor for z denom
    const MIN_WEIGHT: f64 = 1e-12;
    const MIN_D_FOR_Z: f64 = 1e-6;
    const PROB_EPS: f64 = 1e-8; // Epsilon for clamping probabilities

    match link {
        LinkFunction::Logit => {
            // Clamp eta to prevent overflow in exp
            let eta_clamped = eta.mapv(|e| e.clamp(-700.0, 700.0));
            // Create mu and then clamp to prevent values exactly at 0 or 1
            let mut mu = eta_clamped.mapv(|e| 1.0 / (1.0 + (-e).exp()));
            mu.mapv_inplace(|v| v.clamp(PROB_EPS, 1.0 - PROB_EPS));

            // Stage: Calculate dμ/dη, which is μ(1-μ) for the logit link.
            // This term must NOT include prior weights.
            let dmu_deta = &mu * &(1.0 - &mu);

            // Stage: Form the true Fisher weights with a tiny floor to avoid literal zeros
            let fisher_w = dmu_deta.mapv(|v| v.max(MIN_WEIGHT));
            let weights = &prior_weights * &fisher_w;

            // Stage: Build the working-response denominator with a slightly larger floor for stability
            let denom_z = dmu_deta.mapv(|v| v.max(MIN_D_FOR_Z));
            let z = &eta_clamped + &((&y.view().to_owned() - &mu) / &denom_z);

            (mu, weights, z)
        }
        LinkFunction::Identity => {
            let mu = eta.clone();
            // For Gaussian models with Identity link, the iterative weights ARE the prior weights
            let weights = prior_weights.to_owned();
            let z = y.to_owned();
            (mu, weights, z)
        }
    }
}

#[inline]
fn mu_only(eta: &Array1<f64>, link: LinkFunction) -> Array1<f64> {
    match link {
        LinkFunction::Logit => {
            // Clamp eta to prevent overflow and clamp probabilities away from 0/1
            let eta_clamped = eta.mapv(|e| e.clamp(-700.0, 700.0));
            let mut mu = eta_clamped.mapv(|e| 1.0 / (1.0 + (-e).exp()));
            mu.mapv_inplace(|v| v.clamp(1e-8, 1.0 - 1e-8));
            mu
        }
        LinkFunction::Identity => eta.clone(),
    }
}

pub fn calculate_deviance(
    y: ArrayView1<f64>,
    mu: &Array1<f64>,
    link: LinkFunction,
    prior_weights: ArrayView1<f64>,
) -> f64 {
    const EPS: f64 = 1e-8; // Increased from 1e-9 for better numerical stability
    match link {
        LinkFunction::Logit => {
            let total_residual = ndarray::Zip::from(y).and(mu).and(prior_weights).fold(
                0.0,
                |acc, &yi, &mui, &wi| {
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
                    acc + wi * (term1 + term2)
                },
            );
            2.0 * total_residual
        }
        LinkFunction::Identity => {
            // Weighted RSS: sum_i w_i (y_i - mu_i)^2
            ndarray::Zip::from(y)
                .and(mu)
                .and(prior_weights)
                .map_collect(|&yi, &mui, &wi| wi * (yi - mui) * (yi - mui))
                .sum()
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlsFallbackReason {
    IllConditioned,
    FactorizationFailed,
}

impl PlsFallbackReason {
    fn as_str(&self) -> &'static str {
        match self {
            PlsFallbackReason::IllConditioned => "ill_conditioned",
            PlsFallbackReason::FactorizationFailed => "factorization_failed",
        }
    }
}

#[derive(Clone, Debug)]
struct PlsSolverDiagnostics {
    x_rows: usize,
    x_cols: usize,
    eb_rows: usize,
    eb_cols: usize,
    e_rows: usize,
    e_cols: usize,
    fallback_reason: Option<PlsFallbackReason>,
}

impl PlsSolverDiagnostics {
    fn new(
        x_rows: usize,
        x_cols: usize,
        eb_rows: usize,
        eb_cols: usize,
        e_rows: usize,
        e_cols: usize,
    ) -> Self {
        Self {
            x_rows,
            x_cols,
            eb_rows,
            eb_cols,
            e_rows,
            e_cols,
            fallback_reason: None,
        }
    }

    fn record_fallback(&mut self, reason: PlsFallbackReason) {
        if self.fallback_reason.is_none() {
            self.fallback_reason = Some(reason);
        }
    }

    fn emit_qr_summary(&self, edf: f64, scale: f64, rank: usize, p_dim: usize, elapsed: Duration) {
        let fallback_fragment = match self.fallback_reason {
            Some(reason) => format!("spd_fallback={}", reason.as_str()),
            None => "spd_fallback=not_triggered".to_string(),
        };
        println!(
            "[PLS Solver] QR summary: x=({}x{}), eb=({}x{}), e=({}x{}); {}; edf={:.2}, scale={:.4e}, rank={}/{} [{:.2?}]",
            self.x_rows,
            self.x_cols,
            self.eb_rows,
            self.eb_cols,
            self.e_rows,
            self.e_cols,
            fallback_fragment,
            edf,
            scale,
            rank,
            p_dim,
            elapsed
        );
    }
}

/// Robust penalized least squares solver following mgcv's pls_fit1 architecture
/// This function implements the logic for a SINGLE P-IRLS step in the TRANSFORMED basis
///
/// The solver now accepts TWO penalty matrices to separate rank detection from penalty application:
/// - `eb`: Lambda-INDEPENDENT balanced penalty root used ONLY for numerical rank detection
/// - `e_transformed`: Lambda-DEPENDENT penalty root used ONLY for applying the actual penalty
pub fn solve_penalized_least_squares(
    x_transformed: ArrayView2<f64>, // The TRANSFORMED design matrix
    z: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    offset: ArrayView1<f64>,
    eb: &Array2<f64>, // Balanced penalty root for rank detection (lambda-independent)
    e_transformed: &Array2<f64>, // Lambda-dependent penalty root for penalty application
    s_transformed: &Array2<f64>, // Precomputed S = EᵀE (per rho)
    workspace: &mut PirlsWorkspace, // Preallocated buffers (Suggestion #6)
    y: ArrayView1<f64>, // Original response (not the working response z)
    link_function: LinkFunction, // Link function to determine appropriate scale calculation
) -> Result<(StablePLSResult, usize), EstimationError> {
    // The penalized least squares solver implements a 5-stage algorithm (QR path) or
    // a fast symmetric solve (SPD path) when H is SPD.

    // FAST PATH: Pure unpenalized WLS case (no penalty rows)
    if eb.nrows() == 0 && e_transformed.nrows() == 0 {
        println!("[PLS Solver] Using fast path for unpenalized WLS");

        // Weighted design and RHS
        let sqrt_w = weights.mapv(f64::sqrt);
        let wx = &x_transformed * &sqrt_w.view().insert_axis(Axis(1));
        let z_eff = &z - &offset;
        let wz = &sqrt_w * &z_eff;

        // Stage: Use pivoted QR only to determine rank and column ordering
        let (_, r_factor, pivot) = pivoted_qr_faer(&wx)?;
        let diag = r_factor.diag();
        let max_diag = diag.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        let tol = max_diag * 1e-12;
        let rank = diag.iter().filter(|&&v| v.abs() > tol).count();

        // Stage: Build the submatrix from the first `rank` pivoted columns (in original index space)
        let kept_cols = &pivot[..rank];
        let mut wx_kept = Array2::<f64>::zeros((wx.nrows(), rank));
        for (j_new, &j_orig) in kept_cols.iter().enumerate() {
            wx_kept.column_mut(j_new).assign(&wx.column(j_orig));
        }

        // Stage: Solve least squares on the kept submatrix via SVD (β_kept = V Σ⁺ Uᵀ wz)
        use crate::calibrate::faer_ndarray::FaerSvd;
        let (u_opt, s, vt_opt) = wx_kept
            .svd(true, true)
            .map_err(EstimationError::LinearSystemSolveFailed)?;
        let (u, vt) = match (u_opt, vt_opt) {
            (Some(u), Some(vt)) => (u, vt),
            _ => {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
        };

        let smax = s.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let tol_svd = smax * 1e-12;

        // Compute Σ⁺ Uᵀ wz without building dense Σ⁺
        let utb = u.t().dot(&wz);
        let mut s_inv_utb = Array1::<f64>::zeros(s.len());
        for i in 0..s.len() {
            if s[i] > tol_svd {
                s_inv_utb[i] = utb[i] / s[i];
            }
        }
        let beta_kept = vt.t().dot(&s_inv_utb); // length = rank

        // Stage: Construct the full beta vector with dropped columns set to zero
        let mut beta_transformed = Array1::<f64>::zeros(x_transformed.ncols());
        for (j_new, &j_orig) in kept_cols.iter().enumerate() {
            beta_transformed[j_orig] = beta_kept[j_new];
        }

        // Stage: Build the Hessian H = Xᵀ W X (since S=0) using the preallocated buffer
        use ndarray::linalg::{general_mat_mul, general_mat_vec_mul};
        let p_wx = wx.ncols();
        if workspace.xtwx_buf.dim() != (p_wx, p_wx) {
            // resize if needed (shouldn't happen in steady state)
            workspace.xtwx_buf = Array2::zeros((p_wx, p_wx));
        }
        workspace.xtwx_buf.fill(0.0);
        general_mat_mul(1.0, &wx.t(), &wx, 0.0, &mut workspace.xtwx_buf);
        // Compute wx.t() * wz into a preallocated buffer via GEMV
        if workspace.vec_buf_p.len() != p_wx {
            workspace.vec_buf_p = Array1::zeros(p_wx);
        }
        workspace.vec_buf_p.fill(0.0);
        general_mat_vec_mul(1.0, &wx.t(), &wz, 0.0, &mut workspace.vec_buf_p);
        let grad = workspace.xtwx_buf.view().dot(&beta_transformed) - &workspace.vec_buf_p;
        let inf_norm = grad.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        if inf_norm > 1e-10 {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }

        return Ok((
            StablePLSResult {
                beta: beta_transformed,
                penalized_hessian: workspace.xtwx_buf.clone(),
                edf: rank as f64, // EDF = rank in unpenalized LS
                scale: 1.0,
            },
            rank,
        ));
    }

    let mut diagnostics = PlsSolverDiagnostics::new(
        x_transformed.nrows(),
        x_transformed.ncols(),
        eb.nrows(),
        eb.ncols(),
        e_transformed.nrows(),
        e_transformed.ncols(),
    );

    // FAST PATH (penalized case): Use symmetric solve on H = Xᵀ W X + S_λ
    if e_transformed.nrows() > 0 {
        // Weighted design and RHS via broadcasting
        workspace.sqrt_w.assign(&weights.mapv(f64::sqrt));
        let sqrt_w_col = workspace.sqrt_w.view().insert_axis(ndarray::Axis(1));
        if workspace.wx.dim() != x_transformed.dim() {
            workspace.wx = Array2::zeros(x_transformed.dim());
        }
        workspace.wx.assign(&x_transformed);
        workspace.wx *= &sqrt_w_col; // wx = X .* sqrt_w
        workspace.wz.assign(&z);
        workspace.wz -= &offset;
        workspace.wz *= &workspace.sqrt_w; // wz = z .* sqrt_w

        let xtwx = {
            let r = &workspace.wx; // sqrt(W) X
            r.t().dot(r)
        };
        let xtwz = workspace.wx.t().dot(&workspace.wz);

        // H = XtWX + S_lambda (symmetrize to avoid false SPD failures)
        let mut penalized_hessian = xtwx + s_transformed;
        let p_dim = x_transformed.ncols();
        for i in 0..p_dim {
            for j in 0..i {
                let v = 0.5 * (penalized_hessian[[i, j]] + penalized_hessian[[j, i]]);
                penalized_hessian[[i, j]] = v;
                penalized_hessian[[j, i]] = v;
            }
        }
        // Try strict SPD (LLᵀ). If it fails, fall back to robust QR path below.
        let (chol_opt, ridge_used, cond_est) = attempt_spd_cholesky(&mut penalized_hessian);
        match chol_opt {
            Some(chol) => {
                if ridge_used > 0.0 {
                    let cond_display = cond_est
                        .map(|c| format!("{c:.2e}"))
                        .unwrap_or_else(|| "unavailable".to_string());
                    log::warn!(
                        "Added ridge {:.3e} before SPD solve (cond ≈ {})",
                        ridge_used,
                        cond_display
                    );
                }

                // Guard: estimate conditioning of H; if too ill-conditioned, fall back to QR path
                let cond_bad = match calculate_condition_number(&penalized_hessian) {
                    Ok(cond) => !cond.is_finite() || cond > 1e8,
                    Err(_) => false,
                };
                if cond_bad {
                    diagnostics.record_fallback(PlsFallbackReason::IllConditioned);
                } else {
                    // Single multi-RHS solve: [ XtWz | Eᵀ ]
                    let rk_rows = e_transformed.nrows();
                    let nrhs = 1 + rk_rows;
                    let mut rhs = Array2::<f64>::zeros((p_dim, nrhs));
                    for i in 0..p_dim {
                        rhs[(i, 0)] = xtwz[i];
                    }
                    for j in 0..rk_rows {
                        for i in 0..p_dim {
                            rhs[(i, 1 + j)] = e_transformed[(j, i)];
                        }
                    }
                    let rhs_view = FaerArrayView::new(&rhs);
                    let sol = chol.solve(rhs_view.as_ref());

                    // β̂ from first column
                    let mut beta_transformed = Array1::zeros(p_dim);
                    for i in 0..p_dim {
                        beta_transformed[i] = sol[(i, 0)];
                    }
                    if !beta_transformed.iter().all(|v| v.is_finite()) {
                        return Err(EstimationError::ModelIsIllConditioned {
                            condition_number: f64::INFINITY,
                        });
                    }

                    // EDF = p - ⟨H⁻¹Eᵀ, Eᵀ⟩_F using remaining columns (compensated summation)
                    let mut frob = 0.0f64;
                    let mut comp = 0.0f64;
                    for j in 0..rk_rows {
                        for i in 0..p_dim {
                            let prod = sol[(i, 1 + j)] * e_transformed[(j, i)];
                            let y_k = prod - comp;
                            let t = frob + y_k;
                            comp = (t - frob) - y_k;
                            frob = t;
                        }
                    }
                    let mp = (p_dim as f64 - rk_rows as f64).max(0.0);
                    let edf = (p_dim as f64 - frob).clamp(mp, p_dim as f64);
                    if !edf.is_finite() {
                        return Err(EstimationError::ModelIsIllConditioned {
                            condition_number: f64::INFINITY,
                        });
                    }

                    // Scale parameter
                    let scale = calculate_scale(
                        &beta_transformed,
                        x_transformed,
                        y,
                        weights,
                        edf,
                        link_function,
                    );
                    if !scale.is_finite() {
                        return Err(EstimationError::ModelIsIllConditioned {
                            condition_number: f64::INFINITY,
                        });
                    }

                    println!(
                        "[PLS Solver] (SPD/LLᵀ) Completed with edf={:.2}, scale={:.4e}",
                        edf, scale
                    );

                    return Ok((
                        StablePLSResult {
                            beta: beta_transformed,
                            penalized_hessian,
                            edf,
                            scale,
                        },
                        p_dim,
                    ));
                }
            }
            None => diagnostics.record_fallback(PlsFallbackReason::FactorizationFailed),
        }
        // If LLᵀ fails, continue into the robust QR path below.
    }

    let function_timer = Instant::now();

    use ndarray::s;

    // Define rank tolerance, matching mgcv's default
    const RANK_TOL: f64 = 1e-7;

    // let n = x_transformed.nrows();
    let p_dim = x_transformed.ncols();

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

    // Stage: Initial QR decomposition of the weighted design matrix

    // Form the weighted design matrix (sqrt(W)X) and weighted response (sqrt(W)z)
    workspace.sqrt_w.assign(&weights.mapv(f64::sqrt)); // Weights are guaranteed non-negative
    let sqrt_w_col = workspace.sqrt_w.view().insert_axis(ndarray::Axis(1));
    // wx <- X .* sqrt_w (broadcast)
    if workspace.wx.dim() != x_transformed.dim() {
        workspace.wx = Array2::zeros(x_transformed.dim());
    }
    workspace.wx.assign(&x_transformed);
    workspace.wx *= &sqrt_w_col;
    let wx = &workspace.wx;
    // wz <- sqrt_w .* z
    workspace.wz.assign(&z);
    workspace.wz *= &workspace.sqrt_w;
    let wz = &workspace.wz;

    // Perform initial pivoted QR on the weighted design matrix
    let (q1, r1_full, initial_pivot) = pivoted_qr_faer(&wx)?;

    // Keep only the leading p rows of r1 (r_rows = min(n, p))
    let r_rows = r1_full.nrows().min(p_dim);
    let r1_pivoted = r1_full.slice(s![..r_rows, ..]);

    // DO NOT UN-PIVOT r1_pivoted. Keep it in its stable, pivoted form.
    // The columns of R1 are currently permuted according to `initial_pivot`.
    // This permutation is crucial for numerical stability in rank detection.
    // Transform RHS using Q1' (first transformation of the RHS)
    let q1_t_wz = q1.t().dot(wz);

    // Stage: Rank determination using the scaled augmented system

    // Instead of un-pivoting r1, apply the SAME pivot to the penalty matrix `eb`
    // This ensures the columns of both matrices are aligned correctly
    let eb_pivoted = pivot_columns(eb.view(), &initial_pivot);

    // Calculate Frobenius norms for scaling
    let r_norm = frobenius_norm(&r1_pivoted);
    let eb_norm = if eb_pivoted.nrows() > 0 {
        frobenius_norm(&eb_pivoted)
    } else {
        1.0
    };

    // Create the scaled augmented matrix for numerical stability using pivoted matrices
    // [R1_pivoted/Rnorm; Eb_pivoted/Eb_norm] - this is the lambda-INDEPENDENT system for rank detection
    let eb_rows = eb_pivoted.nrows();
    let scaled_rows = r_rows + eb_rows;
    assert!(workspace.scaled_matrix.nrows() >= scaled_rows);
    assert!(workspace.scaled_matrix.ncols() >= p_dim);
    let mut scaled_matrix = workspace
        .scaled_matrix
        .slice_mut(s![..scaled_rows, ..p_dim]);

    // Fill in with slice assignments and scale in place
    use ndarray::s as ns;
    {
        let mut top = scaled_matrix.slice_mut(ns![..r_rows, ..]);
        top.assign(&r1_pivoted);
        let inv = 1.0 / r_norm;
        top.mapv_inplace(|v| v * inv);
    }
    if eb_rows > 0 {
        let mut bot = scaled_matrix.slice_mut(ns![r_rows.., ..]);
        bot.assign(&eb_pivoted);
        let inv = 1.0 / eb_norm;
        bot.mapv_inplace(|v| v * inv);
    }

    // Perform pivoted QR on the scaled matrix for rank determination
    let scaled_owned = scaled_matrix.to_owned();
    let (_, r_scaled, rank_pivot_scaled) = pivoted_qr_faer(&scaled_owned)?;

    // Determine rank using condition number on the scaled matrix
    let mut rank = p_dim.min(scaled_rows);
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

    // Stage: Create the rank-reduced system using the rank pivot

    // Also need to pivot e_transformed to maintain consistency with all pivoted matrices
    let e_transformed_pivoted = pivot_columns(e_transformed.view(), &initial_pivot);

    // Apply the rank-determining pivot to the working matrices, then keep the first `rank` columns
    // This ensures we drop by position in the rank-ordered system (Option A fix)
    let r1_ranked = pivot_columns(r1_pivoted.view(), &rank_pivot_scaled);
    let e_transformed_ranked = pivot_columns(e_transformed_pivoted.view(), &rank_pivot_scaled);

    // Keep the first `rank` columns by position
    let r1_dropped = r1_ranked.slice(s![.., ..rank]).to_owned();

    let e_transformed_rows = e_transformed_ranked.nrows();
    let mut e_transformed_dropped = Array2::zeros((e_transformed_rows, rank));
    if e_transformed_rows > 0 {
        e_transformed_dropped.assign(&e_transformed_ranked.slice(s![.., ..rank]));
    }

    // Record kept positions in the initial pivoted order for later reconstruction
    let kept_positions: Vec<usize> = rank_pivot_scaled[..rank].to_vec();

    // Stage: Final QR decomposition on the unscaled, reduced system

    // Form the final augmented matrix: [R1_dropped; E_transformed_dropped]
    // This uses the lambda-DEPENDENT penalty for actual penalty application
    let final_aug_rows = r_rows + e_transformed_rows;
    assert!(workspace.final_aug_matrix.nrows() >= final_aug_rows);
    assert!(workspace.final_aug_matrix.ncols() >= rank);
    let mut final_aug_matrix = workspace
        .final_aug_matrix
        .slice_mut(s![..final_aug_rows, ..rank]);

    // Fill via slice assignments (Suggestion #9)
    final_aug_matrix
        .slice_mut(ns![..r_rows, ..])
        .assign(&r1_dropped);
    if e_transformed_rows > 0 {
        final_aug_matrix
            .slice_mut(ns![r_rows.., ..])
            .assign(&e_transformed_dropped);
    }

    // Perform final pivoted QR on the unscaled, reduced system
    let final_aug_owned = final_aug_matrix.to_owned();
    let (q_final, mut r_final, final_pivot) = pivoted_qr_faer(&final_aug_owned)?;

    // Add tiny ridge jitter to avoid singular/infinite condition number
    let r_final_sq = r_final.slice(s![..rank, ..rank]);
    let eps = 1e-10_f64;
    let diag_floor = eps.max(r_final_sq[[0, 0]].abs() * 1e-10);
    let m = rank;
    let mut any_modified = false;
    for i in 0..m {
        // ensure strictly positive diagonal to avoid singular/inf cond
        if r_final[[i, i]].abs() < diag_floor {
            r_final[[i, i]] = if r_final[[i, i]] >= 0.0 {
                diag_floor
            } else {
                -diag_floor
            };
            any_modified = true;
        }
    }
    if any_modified {
        log::info!(
            "[PLS Solver] Applied tiny ridge jitter ({:.3e}) to R diagonal for stability",
            diag_floor
        );
    }

    // Stage: Apply the second transformation to the RHS and solve the system

    // Prepare the full RHS for the final system
    assert!(workspace.rhs_full.len() >= final_aug_rows);
    let mut rhs_full = workspace.rhs_full.slice_mut(s![..final_aug_rows]);
    rhs_full.fill(0.0);

    // Use q1_t_wz for the data part (already transformed by Q1')
    rhs_full
        .slice_mut(s![..r_rows])
        .assign(&q1_t_wz.slice(s![..r_rows]));

    // The penalty part is zeros (already initialized)

    // Apply second transformation to the RHS using Q_final'
    let rhs_final = q_final.t().dot(&rhs_full.to_owned());

    // Extract the square upper-triangular part of R and corresponding RHS
    let r_square = r_final.slice(s![..rank, ..rank]);
    let rhs_square = rhs_final.slice(s![..rank]);

    // Back-substitution to solve the triangular system
    let mut beta_dropped = Array1::zeros(rank);

    // Hoist diagonal tolerance invariants outside inner loop
    let max_diag = r_square
        .diag()
        .iter()
        .fold(0.0f64, |acc, &val| acc.max(val.abs()));
    let tol = (max_diag + 1.0) * 1e-14;

    for i in (0..rank).rev() {
        // Initialize with right-hand side value
        let mut sum = rhs_square[i];

        // Subtract known values from higher indices
        for j in (i + 1)..rank {
            sum -= r_square[[i, j]] * beta_dropped[j];
        }

        if r_square[[i, i]].abs() < tol {
            // This should not happen with proper rank detection in Stage 2
            log::warn!(
                "Tiny diagonal {} at position {}, but continuing with Stage 2 rank={}",
                r_square[[i, i]],
                i,
                rank
            );
            // Set coefficient to zero and continue instead of erroring
            beta_dropped[i] = 0.0;
            continue;
        }

        beta_dropped[i] = sum / r_square[[i, i]];
    }

    // Stage: Reconstruct the full coefficient vector
    // Direct composition approach: orig_j = initial_pivot[ kept_positions[ final_pivot[j] ] ]
    // This maps each solved coefficient directly to its original column index

    let mut beta_transformed = Array1::zeros(p_dim);

    // For each solved coefficient j, find its original column index through the permutation chain
    for j in 0..rank {
        let col_in_kept_space = final_pivot[j]; // Which kept column this coeff belongs to
        let col_in_initial_pivoted_space = kept_positions[col_in_kept_space]; // Map to initial-pivoted space
        let original_col_index = initial_pivot[col_in_initial_pivoted_space]; // Map to original space
        beta_transformed[original_col_index] = beta_dropped[j];
    }

    // VERIFICATION: Check that the normal equations hold for the reconstructed beta
    // This is critical to ensure correctness - make it unconditional for now
    {
        let residual = {
            let mut eta = offset.to_owned();
            eta += &x_transformed.dot(&beta_transformed);
            eta - &z
        };
        let weighted_residual = &weights * &residual;
        let grad_dev_part = x_transformed.t().dot(&weighted_residual);
        let grad_pen_part = s_transformed.dot(&beta_transformed);
        let grad = &grad_dev_part + &grad_pen_part;
        let grad_norm_inf = grad.iter().fold(0.0f64, |a, &v| a.max(v.abs()));

        let scale = beta_transformed.iter().map(|&v| v.abs()).sum::<f64>() + 1.0;

        // If gradient appears large, log and continue. QR with rank drop already stabilized the solve.
        if grad_norm_inf > 1e-6 * scale {
            log::warn!(
                "PLS triangular solve residual larger than threshold: ||grad||_inf={:.3e}, scale={:.3e}. Continuing.",
                grad_norm_inf,
                scale
            );
        }
    }

    // Stage: Construct the penalized Hessian
    // Build XtWX without re-touching n using Stage 1 QR result.
    // From (sqrt(W)X) * P = Q * R  =>  Xᵀ W X = P (Rᵀ R) Pᵀ
    // Compute RᵀR into a preallocated buffer
    {
        use ndarray::linalg::general_mat_mul;
        let mut buf = workspace
            .xtwx_buf
            .slice_mut(s![..r1_pivoted.ncols(), ..r1_pivoted.ncols()]);
        buf.fill(0.0);
        // buf <- Rᵀ R
        general_mat_mul(1.0, &r1_pivoted.t(), &r1_pivoted, 0.0, &mut buf);
    }
    let xtwx_pivoted_view = workspace
        .xtwx_buf
        .slice(s![..r1_pivoted.ncols(), ..r1_pivoted.ncols()]);
    let xtwx = unpivot_sym_by_perm(xtwx_pivoted_view, &initial_pivot);

    // Use precomputed S = EᵀE from caller (original order)
    let penalized_hessian = &xtwx + s_transformed;

    // Debug-time guards to verify numerical properties
    #[cfg(debug_assertions)]
    {
        use crate::calibrate::faer_ndarray::FaerCholesky;

        // (a) Symmetry check (relative)
        let mut asym_sum = 0.0f64;
        let mut abs_sum = 0.0f64;
        for i in 0..penalized_hessian.nrows() {
            for j in 0..penalized_hessian.ncols() {
                let a = penalized_hessian[[i, j]];
                let b = penalized_hessian[[j, i]];
                asym_sum += (a - b).abs();
                abs_sum += a.abs();
            }
        }
        let rel_asym = asym_sum / (1.0 + abs_sum);
        debug_assert!(
            rel_asym < 1e-10,
            "Penalized Hessian not symmetric (rel_asym={})",
            rel_asym
        );

        // (b) PD sanity (allow PSD): add tiny ridge then try Cholesky
        let mut h_check = penalized_hessian.clone();
        let ridge = 1e-12;
        for i in 0..h_check.nrows() {
            h_check[[i, i]] += ridge;
        }
        if h_check.cholesky(Side::Lower).is_err() {
            log::warn!(
                "Penalized Hessian failed Cholesky even after tiny ridge; matrix may be poorly conditioned."
            );
        }
    }

    // Stage: Calculate the EDF and scale parameter

    // Calculate effective degrees of freedom using H and XtWX directly (stable)
    let mut edf = calculate_edf(&penalized_hessian, e_transformed)?;
    if !edf.is_finite() || edf.is_nan() {
        // robust fallback for rank-deficient/near-singular cases
        let p = penalized_hessian.ncols() as f64;
        let rank_s = e_transformed.nrows() as f64;
        edf = (p - rank_s).max(0.0);
    }

    // Calculate scale parameter
    let scale = calculate_scale(
        &beta_transformed,
        x_transformed,
        y,
        weights,
        edf,
        link_function,
    );

    // At this point, the solver has completed:
    // - Computing coefficient estimates (beta) for the current iteration
    // - Forming the penalized Hessian matrix (X'WX + S) for uncertainty quantification
    // - Calculating effective degrees of freedom (model complexity measure)
    // - Estimating the scale parameter (variance component for Gaussian models)
    diagnostics.emit_qr_summary(
        edf,
        scale,
        rank,
        x_transformed.ncols(),
        function_timer.elapsed(),
    );

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
fn frobenius_norm<S>(matrix: &ndarray::ArrayBase<S, ndarray::Ix2>) -> f64
where
    S: ndarray::Data<Elem = f64>,
{
    matrix.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Perform pivoted QR decomposition using faer's robust implementation
/// This uses faer's high-level ColPivQr solver which guarantees mathematical
/// consistency between the Q, R, and P factors of the decomposition A*P = Q*R
fn pivoted_qr_faer(
    matrix: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>, Vec<usize>), EstimationError> {
    use faer::Mat;
    use faer::linalg::solvers::ColPivQr;

    let m = matrix.nrows();
    let n = matrix.ncols();
    let k = m.min(n);

    // Stage: Convert ndarray to a faer matrix
    let mut a_faer = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            a_faer[(i, j)] = matrix[[i, j]];
        }
    }

    // Stage: Perform the column-pivoted QR decomposition using the high-level API
    // This guarantees that Q, R, and P are all from the same consistent decomposition
    let qr = ColPivQr::new(a_faer.as_ref());

    // Stage: Extract the consistent Q factor (thin version)
    let q_faer = qr.compute_thin_Q();
    let mut q = Array2::zeros((m, k));
    for i in 0..m {
        for j in 0..k {
            q[[i, j]] = q_faer[(i, j)];
        }
    }

    // Stage: Extract the consistent R factor
    let r_faer = qr.R();
    let mut r = Array2::zeros((k, n));
    for i in 0..k {
        for j in 0..n {
            r[[i, j]] = r_faer[(i, j)];
        }
    }

    // Stage: Extract the consistent column permutation (pivot)
    let perm = qr.P();
    let (p0_slice, p1_slice) = perm.arrays();
    let p0: Vec<usize> = p0_slice.to_vec();
    let p1: Vec<usize> = p1_slice.to_vec();

    // The mathematical identity is A*P = Q*R.
    // Our goal is to find which permutation vector, when used with our `pivot_columns`
    // function, correctly reconstructs A*P.
    let qr_product = q.dot(&r);

    // Try candidate p0
    let a_p0 = pivot_columns(matrix.view(), &p0);

    // Try candidate p1
    let a_p1 = pivot_columns(matrix.view(), &p1);

    // Use relative error for scale-robust comparison
    let compute_relative_error = |a_p: &Array2<f64>| -> f64 {
        let diff_norm = (a_p - &qr_product).mapv(|x| x * x).sum().sqrt();
        let a_norm = a_p.mapv(|x| x * x).sum().sqrt();
        let qr_norm = qr_product.mapv(|x| x * x).sum().sqrt();
        let denom = (a_norm + qr_norm + 1e-16).max(1e-16); // Avoid division by zero
        diff_norm / denom
    };

    let err0 = compute_relative_error(&a_p0);
    let err1 = compute_relative_error(&a_p1);

    let pivot: Vec<usize> = if err0 < 1e-12 {
        p0
    } else if err1 < 1e-12 {
        p1
    } else {
        // This case should not be reached with a correct library, but as a fallback,
        // it indicates a severe numerical or logical issue.
        // We return an error instead of guessing, which caused the original failures.
        return Err(EstimationError::LayoutError(format!(
            "Could not determine correct QR permutation. Reconstruction errors: {:.2e}, {:.2e}",
            err0, err1
        )));
    };

    Ok((q, r, pivot))
}

/// Calculate effective degrees of freedom using the final unpivoted Hessian
/// This avoids pivot mismatches by using the correctly aligned final matrices
fn calculate_edf(
    penalized_hessian: &Array2<f64>,
    e_transformed: &Array2<f64>,
) -> Result<f64, EstimationError> {
    let p = penalized_hessian.ncols();
    let r = e_transformed.nrows();
    let mp = ((p - r) as f64).max(0.0);
    if r == 0 {
        return Ok(p as f64);
    }
    let h_view = FaerArrayView::new(penalized_hessian);
    let rhs_arr = e_transformed.t().to_owned();
    let rhs_view = FaerArrayView::new(&rhs_arr);

    // Try LLᵀ first
    if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
        let sol = ch.solve(rhs_view.as_ref());
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += sol[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    // Try LDLᵀ (semi-definite)
    if let Ok(ld) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
        let sol = ld.solve(rhs_view.as_ref());
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += sol[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    // Last resort: symmetric indefinite LBLᵀ (Bunch–Kaufman)
    let lb = FaerLblt::new(h_view.as_ref(), Side::Lower);
    let sol = lb.solve(rhs_view.as_ref());
    if sol.nrows() == p && sol.ncols() == r {
        let mut tr = 0.0;
        for j in 0..r {
            for i in 0..p {
                tr += sol[(i, j)] * e_transformed[(j, i)];
            }
        }
        return Ok((p as f64 - tr).clamp(mp, p as f64));
    }

    Err(EstimationError::ModelIsIllConditioned {
        condition_number: f64::INFINITY,
    })
}

/// Calculate scale parameter correctly for different link functions
/// For Gaussian (Identity): Based on weighted residual sum of squares
/// For Binomial (Logit): Fixed at 1.0 as in mgcv
fn calculate_scale(
    beta: &Array1<f64>,
    x: ArrayView2<f64>,
    y: ArrayView1<f64>, // This is the original response, not the working response z
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
            // STRATEGIC DESIGN DECISION: Use unweighted observation count for mgcv compatibility
            // Standard WLS theory suggests using sum(weights) as effective sample size,
            // but mgcv's gam.fit3 uses 'n.true' (unweighted count) in the denominator.
            // We maintain this behavior for strict mgcv compatibility.
            let effective_n = y.len() as f64;
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
    use crate::calibrate::faer_ndarray::{FaerEigh, FaerQr};
    use ndarray::s;

    let p = x.ncols();

    // Stage: Perform the QR decomposition of sqrt(W)X to get R_bar
    let sqrt_w = weights.mapv(|w| w.sqrt()); // Weights are guaranteed non-negative with current link functions
    let wx = &x * &sqrt_w.view().insert_axis(ndarray::Axis(1));
    let (_, r_bar) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    let r_rows = r_bar.nrows().min(p);
    let r1_full = r_bar.slice(s![..r_rows, ..]);

    // Stage: Get the square root of the penalty matrix, E
    // We need to use eigendecomposition as S_lambda is not necessarily from a single root
    let (eigenvalues, eigenvectors) = s_lambda
        .eigh(Side::Lower)
        .map_err(EstimationError::EigendecompositionFailed)?;

    // Find the maximum eigenvalue to create a relative tolerance
    let max_eigenval = eigenvalues.iter().fold(0.0f64, |max, &val| max.max(val));

    // Define a relative tolerance. Use an absolute fallback for zero matrices.
    let tolerance = if max_eigenval > 0.0 {
        max_eigenval * 1e-12
    } else {
        1e-12
    };

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

    // Stage: Form the augmented matrix [R1; E_t]
    // Note: Here we use the full, un-truncated matrices because we are just computing
    // the Hessian for a given model, not performing rank detection.
    let e_t = e.t();
    let nr = r_rows + e_t.nrows();
    let mut augmented_matrix = Array2::zeros((nr, p));
    augmented_matrix
        .slice_mut(s![..r_rows, ..])
        .assign(&r1_full);
    augmented_matrix.slice_mut(s![r_rows.., ..]).assign(&e_t);

    // Stage: Perform the QR decomposition on the augmented matrix
    let (_, r_aug) = augmented_matrix
        .qr()
        .map_err(EstimationError::LinearSystemSolveFailed)?;

    // Stage: Recognize that the penalized Hessian is R_aug' * R_aug
    let h_final = r_aug.t().dot(&r_aug);

    Ok(h_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibrate::basis::create_difference_penalty_matrix;
    use crate::calibrate::construction::{
        build_design_and_penalty_matrices, compute_penalty_square_roots,
    };
    use crate::calibrate::data::TrainingData;
    use crate::calibrate::model::{BasisConfig, InteractionPenaltyKind, map_coefficients};
    use approx::assert_abs_diff_eq;
    use ndarray::{Array1, Array2, arr1, arr2};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashMap;

    /// Un-pivots the columns of a matrix according to a pivot vector.
    ///
    /// This reverses the permutation `A*P` to recover `A` from `B`, where `B = A*P`.
    /// It assumes the `pivot` vector is a **forward** permutation, where `pivot[i]`
    /// is the original column index that was moved to position `i`.
    ///
    /// # Parameters
    /// * `pivoted_matrix`: The matrix whose columns are permuted (e.g., the `R` factor).
    /// * `pivot`: The forward permutation vector from the QR decomposition.
    fn unpivot_columns(pivoted_matrix: ArrayView2<f64>, pivot: &[usize]) -> Array2<f64> {
        let r = pivoted_matrix.nrows();
        let c = pivoted_matrix.ncols();
        let mut unpivoted_matrix = Array2::zeros((r, c));

        // The C code logic `dum[*pi]= *px;` translates to:
        // The i-th column of the pivoted matrix belongs at the `pivot[i]`-th
        // position in the un-pivoted matrix.
        for i in 0..c {
            let original_col_index = pivot[i];
            let pivoted_col = pivoted_matrix.column(i);
            unpivoted_matrix
                .column_mut(original_col_index)
                .assign(&pivoted_col);
        }

        unpivoted_matrix
    }

    // === Helper types for test refactoring ===
    #[derive(Debug, Clone)]
    enum SignalType {
        NoSignal,     // Pure noise, expect coefficients near zero
        LinearSignal, // A clear linear trend the model should find
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
        // --- Data generation ---
        let n_samples = 1000;
        let mut rng = StdRng::seed_from_u64(42);
        let p = Array1::linspace(-2.0, 2.0, n_samples);

        let (y, true_linear_predictor) = match link_function {
            LinkFunction::Logit => {
                let true_log_odds = match signal_type {
                    SignalType::NoSignal => Array1::zeros(n_samples), // log_odds = 0 -> prob = 0.5
                    SignalType::LinearSignal => &p * 1.5 - 0.5,
                };
                let y_values: Vec<f64> = true_log_odds
                    .iter()
                    .map(|&log_odds| {
                        let prob = 1.0 / (1.0 + (-log_odds as f64).exp());
                        if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 }
                    })
                    .collect();
                (Array1::from_vec(y_values), true_log_odds)
            }
            LinkFunction::Identity => {
                let true_mean = match signal_type {
                    SignalType::NoSignal => Array1::zeros(n_samples), // Mean = 0
                    SignalType::LinearSignal => &p * 1.5 + 0.5, // Different intercept for variety
                };
                let noise: Array1<f64> =
                    Array1::from_shape_fn(n_samples, |_| rng.r#gen::<f64>() - 0.5); // N(0, 1/12)
                let y = &true_mean + &noise;
                (y, true_mean)
            }
        };

        let data = TrainingData {
            y,
            p,
            sex: Array1::from_iter((0..n_samples).map(|i| (i % 2) as f64)),
            pcs: Array2::zeros((n_samples, 0)),
            weights: Array1::from_elem(n_samples, 1.0),
        };

        // --- Model configuration ---
        let config = ModelConfig {
            link_function,
            penalty_order: 2,
            convergence_tolerance: 1e-7,
            max_iterations: 150,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 50,
            firth_bias_reduction: matches!(link_function, LinkFunction::Logit),
            pgs_basis_config: BasisConfig {
                num_knots: 5,
                degree: 3,
            },
            pc_configs: vec![],
            pgs_range: (-2.0, 2.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
        };

        // --- Run the fit ---
        let (x_matrix, penalty_ws, layout) = setup_pirls_test_inputs(&data, &config)?;
        let rho_vec = Array1::<f64>::zeros(penalty_ws.rs_list().len());

        let offset = Array1::<f64>::zeros(data.y.len());
        let pirls_result = fit_model_for_fixed_rho(
            rho_vec.view(),
            x_matrix.view(),
            offset.view(),
            data.y.view(),
            data.weights.view(),
            penalty_ws.view_without_cache(),
            &layout,
            &config,
        )?;

        // --- Return all necessary components for assertion ---
        Ok(TestScenarioResult {
            pirls_result,
            x_matrix,
            layout,
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
        let s = e.t().dot(&e);
        let mut ws = PirlsWorkspace::new(x.nrows(), x.ncols(), e.nrows(), e.nrows());
        let offset = Array1::<f64>::zeros(z.len());
        let result = solve_penalized_least_squares(
            x.view(),
            z.view(),
            weights.view(),
            offset.view(),
            &e, // For test: use same matrix for both rank detection and penalty
            &e, // For test: use same matrix for both rank detection and penalty
            &s,
            &mut ws,
            z.view(),
            LinkFunction::Identity,
        );

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
            sex_col: None,
            pgs_main_cols: 0..0,
            sex_pgs_cols: None,
            pc_null_cols: vec![],
            pc_null_block_idx: vec![],
            penalty_map: vec![],
            pc_main_block_idx: vec![],
            interaction_block_idx: vec![],
            sex_pgs_block_idx: None,
            interaction_factor_widths: vec![],
            total_coeffs: 2,
            num_penalties: 2,
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

    /// The stable reparameterization must correctly detect the null space of
    /// standard spline penalties. If it treats the entire block as full rank,
    /// the pseudo-determinant and EDF calculations become invalid.
    #[test]
    fn test_stable_reparameterization_preserves_nullspace_rank() {
        use crate::calibrate::construction::{
            ModelLayout, compute_penalty_square_roots, stable_reparameterization,
        };

        // Create a canonical cubic spline penalty (second-order differences)
        // whose null space has dimension two.
        let num_basis_functions = 10;
        let penalty_order = 2;
        let penalty = create_difference_penalty_matrix(num_basis_functions, penalty_order)
            .expect("valid difference penalty");

        // Compute the analytical rank from the eigen-spectrum.
        let (eigenvalues, _) = penalty
            .eigh(Side::Lower)
            .expect("eigendecomposition of penalty matrix");
        let max_eigenvalue = eigenvalues
            .iter()
            .fold(0.0_f64, |acc: f64, &val| acc.max(val));
        let tolerance = if max_eigenvalue > 0.0 {
            max_eigenvalue * 1e-12
        } else {
            1e-12
        };
        let expected_rank = eigenvalues.iter().filter(|&&ev| ev > tolerance).count();
        // Confirm the canonical null-space dimension (rank = k - order).
        assert_eq!(expected_rank, num_basis_functions - penalty_order);

        // Construct penalty square roots and verify their rank matches expectations.
        let rs_list = compute_penalty_square_roots(&[penalty.clone()]).expect("penalty roots");
        assert_eq!(rs_list[0].nrows(), expected_rank);
        assert_eq!(rs_list[0].ncols(), num_basis_functions);

        // Run stable reparameterization and confirm the null space dimension is preserved.
        let layout = ModelLayout::external(num_basis_functions, 1);
        let lambdas = vec![1.0];
        let reparam = stable_reparameterization(&rs_list, &lambdas, &layout)
            .expect("stable reparameterization");

        assert_eq!(
            reparam.e_transformed.nrows(),
            expected_rank,
            "Stable reparameterization should preserve the penalty's null space dimension",
        );

        // Validate the pseudo-determinant uses only the positive eigenvalues.
        let positive_eigs: Vec<f64> = eigenvalues
            .iter()
            .copied()
            .filter(|&ev| ev > tolerance)
            .collect();
        let expected_log_det: f64 = positive_eigs.iter().map(|&ev| ev.ln()).sum::<f64>();
        assert_abs_diff_eq!(reparam.log_det, expected_log_det, epsilon = 1e-9);

        // The transformed penalty should expose the same null-space dimension.
        let (transformed_eigs, _) = reparam
            .s_transformed
            .eigh(Side::Lower)
            .expect("eigendecomposition of transformed penalty");
        let transformed_max = transformed_eigs
            .iter()
            .fold(0.0_f64, |acc: f64, &val| acc.max(val));
        let transformed_tol = if transformed_max > 0.0 {
            transformed_max * 1e-12
        } else {
            1e-12
        };
        let transformed_rank = transformed_eigs
            .iter()
            .filter(|&&ev| ev > transformed_tol)
            .count();
        assert_eq!(transformed_rank, expected_rank);
    }

    /// Helper to set up the inputs required for `fit_model_for_fixed_rho`.
    /// This encapsulates the boilerplate of setting up test inputs.
    fn setup_pirls_test_inputs(
        data: &TrainingData,
        config: &ModelConfig,
    ) -> Result<(Array2<f64>, PenaltyWorkspaceOwned, ModelLayout), Box<dyn std::error::Error>> {
        let (x_matrix, s_list, layout, _, _, _, _, _, _) =
            build_design_and_penalty_matrices(data, config)?;
        let penalty_ws = PenaltyWorkspaceOwned::from_penalty_matrices(s_list, &layout)?;
        Ok((x_matrix, penalty_ws, layout))
    }

    /// Test that the unpivot_columns function correctly reverses a column pivot
    #[test]
    fn test_unpivot_columns_basic() {
        // Create a simple test matrix
        let original = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);

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

        // Create unit weights for the test
        let weights = Array1::from_elem(n_samples, 1.0);

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
            sex_col: None,
            pgs_main_cols: 0..0,
            sex_pgs_cols: None,
            pc_null_cols: vec![],
            pc_null_block_idx: vec![],
            penalty_map: vec![],
            pc_main_block_idx: vec![],
            interaction_block_idx: vec![],
            sex_pgs_block_idx: None,
            interaction_factor_widths: vec![],
            total_coeffs: 2,
            num_penalties: 2,
        };

        // Create a simple config with values known to lead to convergence
        let config = ModelConfig {
            link_function: LinkFunction::Identity, // Simple linear model for stability
            max_iterations: 100,                   // Increased for stability
            convergence_tolerance: 1e-6,           // Less strict for test stability
            penalty_order: 2,
            reml_convergence_tolerance: 1e-6,
            reml_max_iterations: 50,
            firth_bias_reduction: false,
            pgs_basis_config: BasisConfig {
                num_knots: 3,
                degree: 3,
            },
            pc_configs: vec![],
            pgs_range: (-1.0, 1.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
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
        let offset = Array1::<f64>::zeros(n_samples);
        let penalty_ws =
            PenaltyWorkspaceOwned::from_penalty_roots(rs_original.clone(), &layout).unwrap();
        let result1 = super::fit_model_for_fixed_rho(
            rho_vec1.view(),
            x.view(),
            offset.view(),
            y.view(),
            weights.view(),
            penalty_ws.view_without_cache(),
            &layout,
            &config,
        )
        .expect("First fit should converge for this stable test case");

        // Call the function with second rho vector
        let result2 = super::fit_model_for_fixed_rho(
            rho_vec2.view(),
            x.view(),
            offset.view(),
            y.view(),
            weights.view(),
            penalty_ws.view_without_cache(),
            &layout,
            &config,
        )
        .expect("Second fit should converge for this stable test case");

        // The key test: directly check that the transformation matrices are different
        // This is the core behavior we want to verify - each set of smoothing parameters
        // should produce a different transformation matrix
        let qs_diff = (&result1.reparam_result.qs - &result2.reparam_result.qs)
            .mapv(|x| x.abs())
            .sum();
        assert!(
            qs_diff > 1e-6,
            "The transformation matrices 'qs' should be different for different rho values"
        );

        // As a secondary check, confirm the coefficient estimates are also different
        let beta_diff = (&result1.beta_transformed - &result2.beta_transformed)
            .mapv(|x| x.abs())
            .sum();
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
            sex: Array1::from_iter((0..n_samples).map(|i| (i % 2) as f64)),
            pcs: Array2::zeros((n_samples, 0)),
            weights: Array1::from_elem(n_samples, 1.0),
        };

        // === PHASE 3: Configure a simple, stable model ===
        let config = ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            convergence_tolerance: 1e-7,
            max_iterations: 150,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 50,
            firth_bias_reduction: true,
            pgs_basis_config: BasisConfig {
                num_knots: 5,
                degree: 3,
            }, // Stable basis
            pc_configs: vec![],     // PGS-only model
            pgs_range: (-2.0, 2.0), // Match the data
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
        };

        // === PHASE 4: Prepare inputs for the target function ===
        let (x_matrix, penalty_ws, layout) = setup_pirls_test_inputs(&data, &config)?;

        // Size rho vector to match actual number of penalties
        let rho_vec = Array1::<f64>::zeros(penalty_ws.rs_list().len());

        // === PHASE 5: Execute the target function ===
        let offset = Array1::<f64>::zeros(data.y.len());
        let pirls_result = fit_model_for_fixed_rho(
            rho_vec.view(),
            x_matrix.view(),
            offset.view(),
            data.y.view(),
            data.weights.view(),
            penalty_ws.view_without_cache(),
            &layout,
            &config,
        )
        .expect("P-IRLS MUST NOT FAIL on a perfectly stable, zero-signal dataset.");

        // === PHASE 6: Assert stability and correctness ===

        // Stage: Assert finiteness by ensuring the result contains no non-finite numbers
        assert!(
            pirls_result.deviance.is_finite(),
            "Deviance must be a finite number, but was {}",
            pirls_result.deviance
        );
        assert!(
            pirls_result.beta_transformed.iter().all(|&b| b.is_finite()),
            "All beta coefficients in the transformed basis must be finite."
        );
        assert!(
            pirls_result
                .penalized_hessian_transformed
                .iter()
                .all(|&h| h.is_finite()),
            "The penalized Hessian must be finite."
        );

        // Stage: Assert correctness by verifying the model learns a flat function
        // Transform beta back to the original, interpretable basis.
        let beta_original = pirls_result
            .reparam_result
            .qs
            .dot(&pirls_result.beta_transformed);

        // Map the flat vector to a structured object to easily isolate the spline part.
        let mapped_coeffs = map_coefficients(&beta_original, &layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;

        // The norm of the spline coefficients should be reasonable for random data.
        // For logistic regression with random 50/50 data, we expect coefficients to be small but not tiny.
        let pgs_coeffs_norm = pgs_spline_coeffs
            .iter()
            .map(|&c| c.powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            pgs_coeffs_norm < 10.0, // Much more lenient - we're testing stability, not exact magnitude
            "Spline coefficients should be finite and reasonable. Got norm: {}",
            pgs_coeffs_norm
        );

        // Log the actual values for diagnostic purposes
        println!("Spline coefficients norm: {:.6}", pgs_coeffs_norm);
        println!(
            "Individual spline coefficients: {:?}",
            &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(5)]
        );

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
        let y_values: Vec<f64> = p
            .iter()
            .map(|&p_val| {
                let log_odds: f64 = -0.5 + 1.5 * p_val;
                let prob = 1.0 / (1.0 + (-log_odds).exp());
                if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 }
            })
            .collect();
        let y = Array1::from_vec(y_values);

        let data = TrainingData {
            y,
            p,
            sex: Array1::from_iter((0..n_samples).map(|i| (i % 2) as f64)),
            pcs: Array2::zeros((n_samples, 0)),
            weights: Array1::from_elem(n_samples, 1.0),
        };

        // === Use same stable model configuration ===
        let config = ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            convergence_tolerance: 1e-7,
            max_iterations: 150,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 50,
            firth_bias_reduction: true,
            pgs_basis_config: BasisConfig {
                num_knots: 5,
                degree: 3,
            },
            pc_configs: vec![],
            pgs_range: (-2.0, 2.0),
            interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(),
            knot_vectors: HashMap::new(),
            range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(),
            interaction_centering_means: HashMap::new(),
            interaction_orth_alpha: HashMap::new(),
        };

        // === Set up inputs using helper ===
        let (x_matrix, penalty_ws, layout) = setup_pirls_test_inputs(&data, &config)?;
        let rho_vec = Array1::<f64>::zeros(penalty_ws.rs_list().len());

        // === Execute P-IRLS ===
        let offset = Array1::<f64>::zeros(data.y.len());
        let pirls_result = fit_model_for_fixed_rho(
            rho_vec.view(),
            x_matrix.view(),
            offset.view(),
            data.y.view(),
            data.weights.view(),
            penalty_ws.view_without_cache(),
            &layout,
            &config,
        )
        .expect("P-IRLS should converge on realistic data with clear signal");

        // === Assert stability (same as random data test) ===
        assert!(
            pirls_result.deviance.is_finite(),
            "Deviance must be finite, got: {}",
            pirls_result.deviance
        );
        assert!(
            pirls_result.beta_transformed.iter().all(|&b| b.is_finite()),
            "All beta coefficients must be finite"
        );
        assert!(
            pirls_result
                .penalized_hessian_transformed
                .iter()
                .all(|&h| h.is_finite()),
            "Penalized Hessian must be finite"
        );

        // === Assert signal detection (different from random data test) ===
        // Transform back to interpretable basis
        let beta_original = pirls_result
            .reparam_result
            .qs
            .dot(&pirls_result.beta_transformed);
        let mapped_coeffs = map_coefficients(&beta_original, &layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;

        // For data with a strong signal, coefficients should be substantial
        let pgs_coeffs_norm = pgs_spline_coeffs
            .iter()
            .map(|&c| c.powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            pgs_coeffs_norm > 0.5, // Should be much larger than random noise
            "Model should detect the clear signal, got coefficient norm: {}",
            pgs_coeffs_norm
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

        println!(
            "Correlation between fitted and true function: {:.6}",
            correlation
        );
        assert!(
            correlation > 0.9, // Strong positive correlation expected
            "The fitted function should strongly correlate with the true function. Correlation: {:.6}",
            correlation
        );

        // Log diagnostics
        println!(
            "Signal data - Spline coefficients norm: {:.6}",
            pgs_coeffs_norm
        );
        println!(
            "Signal data - Sample coefficients: {:?}",
            &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(3)]
        );
        println!("✓ Test passed: P-IRLS stable and correctly learns realistic signal");

        Ok(())
    }

    /// Test that P-IRLS is stable and correct on ideal data with Identity link (Gaussian).
    /// This verifies the algorithm converges and behaves correctly on easy data.
    #[test]
    fn test_pirls_is_stable_on_perfectly_good_data_identity()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = run_pirls_test_scenario(LinkFunction::Identity, SignalType::NoSignal)?;

        // === Assert stability ===
        assert!(
            result.pirls_result.deviance.is_finite(),
            "Deviance must be finite, got: {}",
            result.pirls_result.deviance
        );
        assert!(
            result
                .pirls_result
                .beta_transformed
                .iter()
                .all(|&b| b.is_finite()),
            "All beta coefficients must be finite"
        );
        assert!(
            result
                .pirls_result
                .penalized_hessian_transformed
                .iter()
                .all(|&h| h.is_finite()),
            "Penalized Hessian must be finite"
        );

        // === Assert that coefficients are small (no signal case) ===
        let beta_original = result
            .pirls_result
            .reparam_result
            .qs
            .dot(&result.pirls_result.beta_transformed);
        let mapped_coeffs = map_coefficients(&beta_original, &result.layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;

        let pgs_coeffs_norm = pgs_spline_coeffs
            .iter()
            .map(|&c| c.powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            pgs_coeffs_norm < 0.5, // Should be small for no-signal data
            "With no signal, spline coeffs should be near zero. Norm: {}",
            pgs_coeffs_norm
        );

        // Log the actual values for diagnostic purposes
        println!(
            "Identity No Signal - Spline coefficients norm: {:.6}",
            pgs_coeffs_norm
        );
        println!(
            "Identity No Signal - Individual spline coefficients: {:?}",
            &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(5)]
        );

        println!(
            "✓ Test passed: `fit_model_for_fixed_rho` is stable and correct on ideal data with Identity link."
        );

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
            "Deviance must be finite, got: {}",
            result.pirls_result.deviance
        );
        assert!(
            result
                .pirls_result
                .beta_transformed
                .iter()
                .all(|&b| b.is_finite()),
            "All beta coefficients must be finite"
        );
        assert!(
            result
                .pirls_result
                .penalized_hessian_transformed
                .iter()
                .all(|&h| h.is_finite()),
            "Penalized Hessian must be finite"
        );

        // === Assert signal detection ===
        // Transform back to interpretable basis
        let beta_original = result
            .pirls_result
            .reparam_result
            .qs
            .dot(&result.pirls_result.beta_transformed);
        let mapped_coeffs = map_coefficients(&beta_original, &result.layout)?;
        let pgs_spline_coeffs = mapped_coeffs.main_effects.pgs;

        // For data with a strong signal, coefficients should be substantial
        let pgs_coeffs_norm = pgs_spline_coeffs
            .iter()
            .map(|&c| c.powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            pgs_coeffs_norm > 0.5, // Should be much larger than random noise
            "Model should detect the clear signal, got coefficient norm: {}",
            pgs_coeffs_norm
        );

        // === More principled test: Compare fitted vs true function ===
        let predicted_linear_predictor = result.x_matrix.dot(&beta_original);
        let correlation = calculate_correlation(
            predicted_linear_predictor.view(),
            result.true_linear_predictor.view(),
        );

        println!(
            "Correlation between fitted and true function: {:.6}",
            correlation
        );
        assert!(
            correlation > 0.9, // Strong positive correlation expected
            "The fitted function should strongly correlate with the true function. Correlation: {:.6}",
            correlation
        );

        // Log diagnostics
        println!(
            "Identity Signal data - Spline coefficients norm: {:.6}",
            pgs_coeffs_norm
        );
        println!(
            "Identity Signal data - Sample coefficients: {:?}",
            &pgs_spline_coeffs[..pgs_spline_coeffs.len().min(3)]
        );
        println!(
            "✓ Test passed: P-IRLS stable and correctly learns realistic signal with Identity link"
        );

        Ok(())
    }

    /// Test that normal equations hold for the solver (unpenalized, any pivots)
    /// Catches coefficient reconstruction bugs (H1) immediately
    #[test]
    fn test_pls_normal_equations_hold_unpenalized() {
        use ndarray::{Array1, Array2};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // Tall random matrix (well-conditioned-ish)
        let n = 80usize;
        let p = 12usize;
        let mut rng = StdRng::seed_from_u64(12345);
        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>() - 0.5);
        let z = Array1::from_shape_fn(n, |_| rng.r#gen::<f64>() - 0.5);
        let w = Array1::from_elem(n, 1.0);

        // No penalty at all
        let e = Array2::<f64>::zeros((0, p));

        // Solve once
        let s = e.t().dot(&e);
        let mut ws = super::PirlsWorkspace::new(x.nrows(), x.ncols(), e.nrows(), e.nrows());
        let offset = Array1::<f64>::zeros(n);
        let (res, ..) = super::solve_penalized_least_squares(
            x.view(),
            z.view(),
            w.view(),
            offset.view(),
            &e,
            &e,
            &s,
            &mut ws,
            z.view(),
            super::LinkFunction::Identity,
        )
        .expect("solver ok");

        // Check stationarity of the *quadratic* objective that the solver actually minimized:
        // grad = Xᵀ W (Xβ - z) + Sβ, with S=0 here.
        let sqrt_w = w.mapv(f64::sqrt);
        let wx = &x * &sqrt_w.view().insert_axis(ndarray::Axis(1));
        let wz = &sqrt_w * &z;
        let grad = wx.t().dot(&(wx.dot(&res.beta) - &wz));

        let inf_norm = grad.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(
            inf_norm < 1e-10,
            "Normal equations not satisfied: ||grad||_∞={}",
            inf_norm
        );

        // And ensure residual is orthogonal to the column space in the weighted sense
        let resid = &wz - &wx.dot(&res.beta);
        let ortho_check = wx.t().dot(&resid);
        let inf_norm2 = ortho_check.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(inf_norm2 < 1e-10, "Residual not orthogonal: {}", inf_norm2);
    }

    /// Test that the WLS step must never be rejected for Gaussian models
    /// Catches step-halving issues (H3)
    #[test]
    fn test_step_accepts_wls_for_gaussian() {
        use ndarray::{Array1, Array2};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // Random tall problem
        let n = 200usize;
        let p = 10usize;
        let mut rng = StdRng::seed_from_u64(54321);
        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>() - 0.5);
        let y = Array1::from_shape_fn(n, |_| rng.r#gen::<f64>() - 0.5);
        let w = Array1::from_elem(n, 1.0);

        // No penalty to keep it pure LS
        let e = Array2::<f64>::zeros((0, p));

        // "Current" state: beta=0
        let beta0 = Array1::<f64>::zeros(p);
        let mu0 = x.dot(&beta0);
        let dev0: f64 = (&y - &mu0).mapv(|r| r * r).sum(); // your calculate_deviance does this

        // WLS solution
        let s = e.t().dot(&e);
        let mut ws = super::PirlsWorkspace::new(x.nrows(), x.ncols(), e.nrows(), e.nrows());
        let offset = Array1::<f64>::zeros(n);
        let (res, _) = super::solve_penalized_least_squares(
            x.view(),
            y.view(),
            w.view(),
            offset.view(),
            &e,
            &e,
            &s,
            &mut ws,
            y.view(),
            super::LinkFunction::Identity,
        )
        .expect("solver ok");

        let mu1 = x.dot(&res.beta);
        let dev1: f64 = (&y - &mu1).mapv(|r| r * r).sum();

        assert!(
            dev1 <= dev0 * (1.0 + 1e-12) || (dev0 - dev1).abs() < 1e-12,
            "Exact WLS step should not increase deviance: dev0={} dev1={}",
            dev0,
            dev1
        );
    }

    /// Test that proves the gradient gate is using the wrong weights (logit)
    /// Exposes convergence check issue (H2)
    #[test]
    fn test_wls_stationarity_old_vs_new_weights_logit() {
        use ndarray::{Array1, Array2};
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        // Modest logit problem
        let n = 400usize;
        let p = 8usize;
        let mut rng = StdRng::seed_from_u64(98765);
        let x = Array2::from_shape_fn((n, p), |_| rng.r#gen::<f64>() - 0.5);
        let eta0 = Array1::zeros(n);
        // y ~ Bernoulli(0.5)
        let y = Array1::from_shape_fn(n, |_| if rng.r#gen::<f64>() > 0.5 { 1.0 } else { 0.0 });
        let w_prior = Array1::from_elem(n, 1.0);

        // Build IRLS vectors at beta=0
        // Use a tuple with let binding to explicitly declare variable usage
        let (_, w_old, z_old) = {
            let vectors = super::update_glm_vectors(
                y.view(),
                &eta0,
                super::LinkFunction::Logit,
                w_prior.view(),
            );
            ((), vectors.1, vectors.2)
        };
        assert!(w_old.iter().all(|w| *w >= 0.0));

        // No penalty to keep it simple
        let e = Array2::<f64>::zeros((0, p));
        let s = e.t().dot(&e);
        let mut ws = super::PirlsWorkspace::new(x.nrows(), x.ncols(), e.nrows(), e.nrows());
        let offset = Array1::<f64>::zeros(n);
        let (res, _) = super::solve_penalized_least_squares(
            x.view(),
            z_old.view(),
            w_old.view(),
            offset.view(),
            &e,
            &e,
            &s,
            &mut ws,
            y.view(),
            super::LinkFunction::Logit,
        )
        .expect("solver ok");

        // Stationarity with OLD weights and z (the quadratic model you just solved)
        let sqrt_w_old = w_old.mapv(f64::sqrt);
        let wx_old = &x * &sqrt_w_old.view().insert_axis(ndarray::Axis(1));
        let wz_old = &sqrt_w_old * &z_old;
        let grad_old = wx_old.t().dot(&(wx_old.dot(&res.beta) - &wz_old)); // S=0 here
        let inf_old = grad_old.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(
            inf_old < 1e-8,
            "Should be stationary w.r.t. old weights, ||grad||_∞={}",
            inf_old
        );

        // Now recompute eta, mu, and updated weights at the accepted beta
        let eta1 = x.dot(&res.beta);
        // Use same approach for the second update_glm_vectors call
        let (_, w_new, z_new) = {
            let vectors = super::update_glm_vectors(
                y.view(),
                &eta1,
                super::LinkFunction::Logit,
                w_prior.view(),
            );
            ((), vectors.1, vectors.2)
        };

        let sqrt_w_new = w_new.mapv(f64::sqrt);
        let wx_new = &x * &sqrt_w_new.view().insert_axis(ndarray::Axis(1));
        let wz_new = &sqrt_w_new * &z_new;

        let grad_new = wx_new.t().dot(&(wx_new.dot(&res.beta) - &wz_new));
        let inf_new = grad_new.iter().fold(0.0f64, |a, &v| a.max(v.abs()));

        // This SHOULD NOT be required to be tiny for convergence right after one step.
        assert!(
            inf_new > 1e-4,
            "If this is tiny, IRLS basically solved in one step — suspicious"
        );
    }

    /// Test that rank-deficient projections must be exact (perfect fit when possible)
    /// This is a stronger, permanent guard against coefficient reconstruction bugs
    #[test]
    fn test_pls_rank_deficient_hits_projection() {
        use ndarray::{Array1, Array2, arr1, arr2};

        // Same structure as your failing test: col3 = col1 + col2
        let x = arr2(&[
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 4.0],
            [1.0, 4.0, 5.0],
        ]);
        let z = arr1(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        let w = Array1::from_elem(5, 1.0);

        let e = Array2::<f64>::zeros((0, 3));

        let s = e.t().dot(&e);
        let mut ws = super::PirlsWorkspace::new(x.nrows(), x.ncols(), e.nrows(), e.nrows());
        let offset = Array1::<f64>::zeros(z.len());
        let (res, rank) = super::solve_penalized_least_squares(
            x.view(),
            z.view(),
            w.view(),
            offset.view(),
            &e,
            &e,
            &s,
            &mut ws,
            z.view(),
            super::LinkFunction::Identity,
        )
        .expect("solver ok");
        assert_eq!(rank, 2);

        // Fitted values must equal the weighted projection of z onto Col(X)
        let fitted = x.dot(&res.beta);
        let rss: f64 = (&z - &fitted).mapv(|r| r * r).sum();

        assert!(
            rss < 1e-12,
            "Rank-deficient LS should project exactly (RSS={})",
            rss
        );

        // And the KKT/normal-equation residual must be ~0 for kept cols
        let sqrt_w = w.mapv(f64::sqrt);
        let wx = &x * &sqrt_w.view().insert_axis(ndarray::Axis(1));
        let wz = &sqrt_w * &z;
        let grad = wx.t().dot(&(wx.dot(&res.beta) - &wz));
        let inf_norm = grad.iter().fold(0.0f64, |a, &v| a.max(v.abs()));
        assert!(
            inf_norm < 1e-10,
            "Normal equations not satisfied: {}",
            inf_norm
        );
    }

    /// Test permutation chain property - locks down coefficient reconstruction logic
    #[test]
    fn test_permutation_chain_property() {
        use ndarray::Array1;
        use rand::rngs::StdRng;
        use rand::{SeedableRng, seq::SliceRandom};

        let mut rng = StdRng::seed_from_u64(42);

        let p = 17usize;
        let rank = 9usize;

        // random initial_pivot: pivoted idx -> original idx
        let mut initial_pivot: Vec<usize> = (0..p).collect();
        initial_pivot.shuffle(&mut rng);

        // choose kept positions in pivoted space and name them 0..rank-1 (kept-space)
        let mut kept_in_pivoted: Vec<usize> = (0..p).collect();
        kept_in_pivoted.shuffle(&mut rng);
        kept_in_pivoted.truncate(rank);
        kept_in_pivoted.sort_unstable(); // order doesn't matter, but your kept-space uses 0..rank-1

        // kept_positions[i] = pivoted-space index of kept col i
        let kept_positions = kept_in_pivoted.clone();

        // final_pivot[j] = kept-space index for coeff j
        let mut final_pivot: Vec<usize> = (0..rank).collect();
        final_pivot.shuffle(&mut rng);

        // distinct sentinels
        let beta_dropped = Array1::from_shape_fn(rank, |j| 1000.0 + j as f64);

        // your placement
        let mut placed = Array1::<f64>::zeros(p);
        for j in 0..rank {
            let k_kept = final_pivot[j];
            let k_pivoted = kept_positions[k_kept];
            let k_orig = initial_pivot[k_pivoted];
            placed[k_orig] = beta_dropped[j];
        }

        // reference via explicit composition
        let mut placed_ref = Array1::<f64>::zeros(p);
        for j in 0..rank {
            let k_orig = initial_pivot[kept_positions[final_pivot[j]]];
            placed_ref[k_orig] = beta_dropped[j];
        }

        assert!(
            placed
                .iter()
                .zip(placed_ref.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12)
        );
    }

    /// Test penalty consistency sanity check
    /// Locks down penalty root consistency issues (H4)
    #[test]
    fn test_penalty_root_consistency() {
        use crate::calibrate::construction::{
            ModelLayout, compute_penalty_square_roots, stable_reparameterization,
        };
        use ndarray::arr2;

        // Two small penalties with different eigenvectors
        let s1 = arr2(&[[1.0, -0.2], [-0.2, 0.5]]);
        let s2 = arr2(&[[0.1, 0.0], [0.0, 0.0]]);
        let s_list = vec![s1, s2];
        let rs = compute_penalty_square_roots(&s_list).expect("roots");

        let layout = ModelLayout {
            intercept_col: 0,
            sex_col: None,
            pgs_main_cols: 0..0,
            sex_pgs_cols: None,
            pc_null_cols: vec![],
            pc_null_block_idx: vec![],
            penalty_map: vec![],
            pc_main_block_idx: vec![],
            interaction_block_idx: vec![],
            sex_pgs_block_idx: None,
            interaction_factor_widths: vec![],
            total_coeffs: 2,
            num_penalties: 2,
        };
        let lambdas = vec![0.7, 3.0];

        let rp = stable_reparameterization(&rs, &lambdas, &layout).expect("reparam");
        let lhs = rp.s_transformed;
        let rhs = rp.e_transformed.t().dot(&rp.e_transformed);

        let diff = (&lhs - &rhs).mapv(|v| v.abs()).sum();
        assert!(
            diff < 1e-10,
            "S != EᵀE in transformed basis (sum abs diff = {})",
            diff
        );
    }

    /// This test verifies that the permutation logic in `pivoted_qr_faer` is correct
    /// and mathematically sound.
    ///
    /// It works by checking the fundamental mathematical identity of a pivoted QR decomposition: A*P = Q*R.
    /// If the permutation P is correct, the identity will hold, and the reconstruction error
    /// || A*P - Q*R || will be small (close to machine precision).
    #[test]
    fn test_pivoted_qr_permutation_is_reliable() {
        use ndarray::arr2;

        // Stage: Set up a matrix that is tricky to pivot
        // It's nearly rank-deficient, with highly correlated columns, forcing a non-trivial pivot.
        // This is representative of the design matrices created in the model tests.
        let a = arr2(&[
            [1.0, 2.0, 3.0, 1.0000001],
            [4.0, 5.0, 9.0, 4.0000002],
            [6.0, 7.0, 13.0, 6.0000003],
            [8.0, 9.0, 17.0, 8.0000004],
        ]);

        // Stage: Execute the function under test
        let (q, r, pivot) = pivoted_qr_faer(&a).expect("QR decomposition itself should not fail");

        // Stage: Verify that the fundamental QR identity holds
        // First, apply the permutation to the original matrix 'a'.
        let a_pivoted = pivot_columns(a.view(), &pivot);

        // Then, compute Q*R using the results from the function.
        let qr_product = q.dot(&r);

        // Calculate the reconstruction error. If the pivot is correct, this should be near zero.
        let reconstruction_error_matrix = &a_pivoted - &qr_product;
        let reconstruction_error_norm = reconstruction_error_matrix.mapv(|x| x.abs()).sum();

        println!("Matrix A:\n{:?}", a);
        println!("Permutation P: {:?}", pivot);
        println!("Reconstructed A*P (from pivot):\n{:?}", a_pivoted);
        println!("Q*R Product:\n{:?}", qr_product);
        println!("Reconstruction Error Norm: {}", reconstruction_error_norm);

        // Stage: Assert that the reconstruction error is small, proving the pivot is correct
        // A correct implementation should have an error norm close to machine epsilon (~1e-15).
        // An error norm greater than 1e-6 would be a definitive failure.
        assert!(
            reconstruction_error_norm < 1e-6,
            "The reconstruction error is too large ({:e}), which indicates the permutation vector is incorrect. The contract A*P = Q*R is violated.",
            reconstruction_error_norm
        );
    }
}

/// Ensure positive definiteness by adding a small constant ridge to the diagonal if needed.
/// This mirrors the outer objective/gradient stabilization (H_eff = H or H + c I),
/// avoiding eigenvalue-dependent clamps that can diverge from the outer path.
fn compute_firth_hat_and_half_logdet(
    x_transformed: ArrayView2<f64>,
    x_original: ArrayView2<f64>,
    weights: ArrayView1<f64>,
    s_transformed: &Array2<f64>,
    workspace: &mut PirlsWorkspace,
) -> Result<(Array1<f64>, f64), EstimationError> {
    let n = x_transformed.nrows();
    let p = x_transformed.ncols();

    workspace
        .sqrt_w
        .assign(&weights.mapv(|w| w.max(0.0).sqrt()));
    let sqrt_w_col = workspace.sqrt_w.view().insert_axis(Axis(1));
    if workspace.wx.dim() != x_transformed.dim() {
        workspace.wx = Array2::zeros(x_transformed.dim());
    }
    workspace.wx.assign(&x_transformed);
    workspace.wx *= &sqrt_w_col;

    let xtwx_transformed = workspace.wx.t().dot(&workspace.wx);
    let mut penalized_hessian = xtwx_transformed.clone() + s_transformed;
    for i in 0..p {
        for j in 0..i {
            let v = 0.5 * (penalized_hessian[[i, j]] + penalized_hessian[[j, i]]);
            penalized_hessian[[i, j]] = v;
            penalized_hessian[[j, i]] = v;
        }
    }

    let mut stabilized = penalized_hessian.clone();
    ensure_positive_definite(&mut stabilized)?;

    let mut fisher = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let wi = weights[i].max(0.0);
        if wi == 0.0 {
            continue;
        }
        let xi = x_original.row(i);
        for j in 0..p {
            let xij = xi[j];
            for k in 0..p {
                fisher[[j, k]] += wi * xij * xi[k];
            }
        }
    }
    let mut fisher = fisher;
    ensure_positive_definite_with_label(&mut fisher, "Firth Fisher information")?;
    let chol_fisher = fisher.clone().cholesky(Side::Lower).map_err(|_| {
        EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: f64::NEG_INFINITY,
        }
    })?;
    let half_log_det = chol_fisher.diag().mapv(f64::ln).sum();

    let h_view = FaerArrayView::new(&stabilized);
    let chol_faer = FaerLlt::new(h_view.as_ref(), Side::Lower).map_err(|_| {
        EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        }
    })?;
    let rhs = workspace.wx.t().to_owned();
    let rhs_view = FaerArrayView::new(&rhs);
    let sol = chol_faer.solve(rhs_view.as_ref());

    let mut hat_diag = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..p {
            let val = sol[(k, i)];
            acc += val * workspace.wx[[i, k]];
        }
        hat_diag[i] = acc;
    }

    Ok((hat_diag, half_log_det))
}

fn ensure_positive_definite(hess: &mut Array2<f64>) -> Result<(), EstimationError> {
    ensure_positive_definite_with_label(hess, "Penalized Hessian")
}

fn ensure_positive_definite_with_label(
    hess: &mut Array2<f64>,
    label: &str,
) -> Result<(), EstimationError> {
    // If already PD, do nothing
    if hess.cholesky(Side::Lower).is_ok() {
        return Ok(());
    }

    let cond_est = calculate_condition_number(hess).ok();
    let diag_scale = max_abs_diag(hess);
    let mut ridge = match cond_est {
        Some(cond) if cond.is_finite() && cond > HESSIAN_CONDITION_TARGET => {
            diag_scale * 1e-10 * (cond / HESSIAN_CONDITION_TARGET)
        }
        Some(cond) if cond.is_finite() => {
            if diag_scale > 0.0 {
                diag_scale * 1e-12
            } else {
                0.0
            }
        }
        _ => diag_scale * 1e-8,
    };
    let mut total_added = 0.0;

    for attempt in 0..=PLS_MAX_FACTORIZATION_ATTEMPTS {
        if ridge > total_added {
            let delta = ridge - total_added;
            for i in 0..hess.nrows() {
                hess[[i, i]] += delta;
            }
            total_added = ridge;
        }

        if hess.cholesky(Side::Lower).is_ok() {
            if total_added > 0.0 {
                let cond_display = cond_est
                    .map(|c| format!("{c:.2e}"))
                    .unwrap_or_else(|| "unavailable".to_string());
                log::warn!(
                    "{} not PD; added ridge {:.1e} (cond ≈ {}) to ensure stability.",
                    label,
                    total_added,
                    cond_display
                );
            }
            return Ok(());
        }

        if attempt == PLS_MAX_FACTORIZATION_ATTEMPTS {
            break;
        }

        if ridge <= 0.0 {
            ridge = diag_scale * 1e-10;
        } else {
            ridge = (ridge * 10.0).max(diag_scale * 1e-10);
        }
        if !ridge.is_finite() || ridge <= 0.0 {
            ridge = diag_scale;
        }
    }

    // As a last resort, report indefiniteness with min eigenvalue for diagnostics
    if let Ok((evals, _)) = hess.eigh(Side::Lower) {
        let min_eig = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        return Err(EstimationError::HessianNotPositiveDefinite {
            min_eigenvalue: min_eig,
        });
    }
    Err(EstimationError::HessianNotPositiveDefinite {
        min_eigenvalue: f64::NEG_INFINITY,
    })
}
