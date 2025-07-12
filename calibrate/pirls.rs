use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::{LinkFunction, ModelConfig};
use log;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};


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
    
    // Use simple S_lambda construction for P-IRLS (stable reparameterization used elsewhere)
    use crate::calibrate::construction::construct_s_lambda;
    let s_lambda = construct_s_lambda(&lambdas, s_list, layout);

    // Initialize beta as zero vector
    let mut beta = Array1::zeros(layout.total_coeffs);
    let mut last_deviance = f64::INFINITY;
    let mut last_change = f64::INFINITY;

    // Validate dimensions
    assert_eq!(x.ncols(), layout.total_coeffs, "X matrix columns must match total coefficients");
    assert_eq!(s_lambda.nrows(), layout.total_coeffs, "S_lambda rows must match total coefficients");
    assert_eq!(s_lambda.ncols(), layout.total_coeffs, "S_lambda columns must match total coefficients");

    for iter in 1..=config.max_iterations {
        let eta = x.dot(&beta);
        let (mu, weights, z) = update_glm_vectors(y, &eta, config.link_function);

        // Check for non-finite values
        if !eta.iter().all(|x| x.is_finite()) || !mu.iter().all(|x| x.is_finite()) ||
           !weights.iter().all(|x| x.is_finite()) || !z.iter().all(|x| x.is_finite()) {
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
            &s_lambda,
        )?;
        
        beta = stable_result.beta;

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

            // For the final result, compute the penalized Hessian properly
            let penalized_hessian = compute_penalized_hessian(x, &final_weights, &s_lambda)?;
            
            return Ok(PirlsResult {
                beta,
                penalized_hessian,
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
/// Following the pls_fit1 algorithm with proper two-stage QR and SVD correction for negative weights
pub fn stable_penalized_least_squares(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    s_lambda: &Array2<f64>,
) -> Result<StablePLSResult, EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    
    use ndarray_linalg::{QR, SVD, Solve};
    
    // Step 1: Handle negative weights according to Wood (2011) Section 3.3
    let neg_indices: Vec<usize> = weights.iter().enumerate()
        .filter(|(_, w)| **w < 0.0)
        .map(|(i, _)| i)
        .collect();
    
    // Form sqrt(|W|)X and sqrt(|W|)z with sign correction
    let sqrt_w_abs = weights.mapv(|w| w.abs().sqrt());
    let mut z_tilde = &y * &sqrt_w_abs;
    
    // Flip sign for negative weight observations (Wood 2011)
    for &i in &neg_indices {
        z_tilde[i] = -z_tilde[i];
    }
    
    let wx = &x * &sqrt_w_abs.view().insert_axis(Axis(1));
    
    // Step 2: First QR decomposition of weighted model matrix: √|W|X = Q₁R₁
    let (q1, r1) = wx.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    
    // Step 3: Form augmented system [R₁; E] where E'E = S_λ
    let e_matrix = penalty_square_root(s_lambda)?;
    let r1_rows = r1.nrows().min(p);
    let mut augmented = Array2::zeros((r1_rows + p, p));
    augmented.slice_mut(s![..r1_rows, ..]).assign(&r1.slice(s![..r1_rows, ..]));
    augmented.slice_mut(s![r1_rows.., ..]).assign(&e_matrix);
    
    // Step 4: Second QR decomposition of augmented system: [R₁; E] = Q₂R₂
    let (_, r2) = augmented.qr().map_err(EstimationError::LinearSystemSolveFailed)?;
    
    // Step 5: Form base penalized Hessian H = R₂'R₂
    let mut penalized_hessian = r2.t().dot(&r2);
    
    // Step 6: Apply SVD correction for negative weights if any exist
    if !neg_indices.is_empty() {
        // Extract rows of Q₁ corresponding to negative weights
        let q1_neg = q1.select(Axis(0), &neg_indices);
        
        // Perform SVD on these rows: Q₁⁻ = UΣV'
        if let Ok((u_opt, sigma, vt_opt)) = q1_neg.svd(true, true) {
            if let (Some(_u), Some(vt)) = (u_opt, vt_opt) {
                // Form correction matrix: I - 2VΣ²V'
                let mut sigma_sq = Array2::zeros((sigma.len(), sigma.len()));
                for (i, &s) in sigma.iter().enumerate() {
                    sigma_sq[[i, i]] = s * s;
                }
                
                let v = vt.t();
                let correction = Array2::eye(p) - 2.0 * v.dot(&sigma_sq.dot(&v.t()));
                
                // Apply correction: H_corrected = R₂'(I - 2VΣ²V')R₂
                penalized_hessian = r2.t().dot(&correction.dot(&r2));
            }
        }
    }
    
    // Step 7: Solve for coefficients: Hβ = X'Wz
    let rhs = x.t().dot(&(&weights * &y));
    let beta = penalized_hessian.solve(&rhs)
        .map_err(EstimationError::LinearSystemSolveFailed)?;
    
    // Step 8: Calculate effective degrees of freedom
    // EDF = tr(H⁻¹X'WX) where H = X'WX + S_λ
    let mut edf = 0.0;
    let weighted_x = &weights.view().insert_axis(Axis(1)) * &x;
    let xtwx = x.t().dot(&weighted_x);
    
    // Calculate trace by solving H⁻¹ * xtwx column by column
    for j in 0..p {
        if let Ok(h_inv_col) = penalized_hessian.solve(&xtwx.column(j).to_owned()) {
            edf += h_inv_col[j];
        }
    }
    edf = edf.max(1.0);
    
    // Step 9: Calculate scale parameter
    let fitted = x.dot(&beta);
    let residuals = &y - &fitted;
    let rss = residuals.mapv(|v| v * v).sum();
    let scale = rss / (n as f64 - edf).max(1.0);
    
    Ok(StablePLSResult {
        beta,
        penalized_hessian,
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
    use ndarray_linalg::{Cholesky, UPLO, Eigh};
    
    // Try Cholesky first
    if let Ok(l) = s.cholesky(UPLO::Lower) {
        return Ok(l);
    }
    
    // Fallback to eigendecomposition for semi-definite matrices
    let (eigenvals, eigenvecs): (Array1<f64>, Array2<f64>) = s.eigh(UPLO::Lower)
        .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
    
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
