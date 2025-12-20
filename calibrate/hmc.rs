//! NUTS Sampler using mini-mcmc
//!
//! This module provides NUTS (No-U-Turn Sampler) for honest uncertainty
//! quantification after PIRLS convergence.
//!
//! # Design
//!
//! Since mini-mcmc's NUTS uses an identity mass matrix, we whiten the
//! parameter space using the Cholesky decomposition of the inverse Hessian:
//!
//! - Transform: β = μ + L @ z  (where L L^T = H^{-1})
//! - The whitened space has unit covariance, so NUTS mixes efficiently
//! - Samples are un-transformed back to the original space
//!
//! # Analytical Gradients
//!
//! We override `unnorm_logp_and_grad` to compute gradients analytically using
//! ndarray, avoiding burn's autodiff overhead. The gradient computation mirrors
//! the true log-posterior gradient (not the PIRLS working gradient).
//!
//! # Memory Efficiency
//!
//! Large data (design matrix, response, etc.) is wrapped in `Arc` to allow
//! sharing across chains without duplication when mini-mcmc clones the target.

use crate::calibrate::faer_ndarray::FaerCholesky;
use faer::Side;
use mini_mcmc::generic_hmc::HamiltonianTarget;
use mini_mcmc::generic_nuts::GenericNUTS;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::sync::Arc;


/// Solve L^T * X = I where L is lower triangular.
///
/// Returns X = L^{-T} (the inverse transpose of L).
/// Uses back-substitution since L^T is upper triangular.
///
/// This is the correct way to compute the whitening transform matrix:
/// Given H = L L^T (Cholesky), we need W where W W^T = H^{-1}
/// Since H^{-1} = L^{-T} L^{-1}, we have W = L^{-T}
fn solve_upper_triangular_transpose(l: &Array2<f64>, dim: usize) -> Array2<f64> {
    // L^T is upper triangular, so we solve L^T * X = I via back-substitution
    let mut result = Array2::<f64>::zeros((dim, dim));

    // For each column of the identity (each column of result)
    for col in 0..dim {
        // Solve L^T * x = e_col (unit vector)
        // Back-substitution: start from last row, work up
        for i in (0..dim).rev() {
            let mut sum = if i == col { 1.0 } else { 0.0 }; // e_col[i]

            // Subtract contributions from already-solved entries
            for j in (i + 1)..dim {
                sum -= l[[j, i]] * result[[j, col]]; // L^T[i,j] = L[j,i]
            }

            // Divide by diagonal (L^T[i,i] = L[i,i])
            let diag = l[[i, i]];
            if diag.abs() < 1e-15 {
                result[[i, col]] = 0.0; // Regularize near-zero diagonal
            } else {
                result[[i, col]] = sum / diag;
            }
        }
    }

    result
}

/// Shared data for NUTS posterior (wrapped in Arc to prevent cloning).
///
/// This struct holds read-only data that is shared across all chains.
/// Using Arc prevents memory explosion when mini-mcmc clones the target.
#[derive(Clone)]
struct SharedData {
    /// Design matrix X [n_samples, dim]
    x: Arc<Array2<f64>>,
    /// Response vector y [n_samples]
    y: Arc<Array1<f64>>,
    /// Prior weights [n_samples]
    weights: Arc<Array1<f64>>,
    /// Combined penalty matrix S [dim, dim]
    penalty: Arc<Array2<f64>>,
    /// MAP estimate (mode) μ [dim]
    mode: Arc<Array1<f64>>,
    /// Number of samples
    n_samples: usize,
    /// Number of coefficients
    dim: usize,
}

/// Whitened log-posterior target with analytical gradients.
///
/// Uses Arc for shared data to prevent memory explosion when cloned for chains.
/// Uses faer for numerically stable Cholesky decomposition.
#[derive(Clone)]
pub struct NutsPosterior {
    /// Shared read-only data (Arc prevents duplication)
    data: SharedData,
    /// Transform: L where L L^T = H^{-1} (computed from Hessian)
    /// This is the inverse-transpose of the Cholesky of H.
    chol: Array2<f64>,
    /// L^T for gradient chain rule: ∇_z = L^T @ ∇_β
    chol_t: Array2<f64>,
    /// Link function type
    is_logit: bool,
}

impl NutsPosterior {
    /// Creates a new posterior target from ndarray data.
    ///
    /// # Arguments
    /// * `x` - Design matrix [n_samples, dim]
    /// * `y` - Response vector [n_samples]
    /// * `weights` - Prior weights [n_samples]
    /// * `penalty_matrix` - Combined penalty S [dim, dim]
    /// * `mode` - MAP estimate μ [dim]
    /// * `hessian` - Hessian H [dim, dim] (NOT the inverse!)
    /// * `is_logit` - True for logistic regression, false for Gaussian
    ///
    /// # Numerical Stability
    /// Accepts the Hessian directly and computes L = (chol(H))^{-T} via
    /// triangular solves, which is more stable than explicitly inverting H.
    pub fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        penalty_matrix: ArrayView2<f64>,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        is_logit: bool,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let dim = x.ncols();

        // Use faer for numerically stable Cholesky decomposition of H
        // H = L_H L_H^T where L_H is lower triangular
        let hessian_owned = hessian.to_owned();
        let chol_factor = hessian_owned
            .cholesky(Side::Lower)
            .map_err(|e| format!("Hessian Cholesky decomposition failed: {:?}", e))?;

        // We need L where L L^T = H^{-1}
        // Since H = L_H L_H^T, we have H^{-1} = L_H^{-T} L_H^{-1}
        // So L = L_H^{-T} (the inverse transpose of the Cholesky factor)
        //
        // To get L_H^{-T}, we solve L_H^T * X = I using back-substitution
        // Since L_H is lower triangular, L_H^T is upper triangular
        let l_h = chol_factor.lower_triangular();
        let chol = solve_upper_triangular_transpose(&l_h, dim);
        let chol_t = chol.t().to_owned();

        let data = SharedData {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            penalty: Arc::new(penalty_matrix.to_owned()),
            mode: Arc::new(mode.to_owned()),
            n_samples,
            dim,
        };

        Ok(Self {
            data,
            chol,
            chol_t,
            is_logit,
        })
    }

    /// Compute log-posterior and gradient analytically using ndarray.
    ///
    /// Returns (log_posterior, gradient_z) where gradient_z is the gradient
    /// with respect to the whitened parameters z.
    fn compute_logp_and_grad_nd(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
        // === Step 1: Transform z (whitened) -> β (original) ===
        // β = μ + L @ z
        let beta = self.data.mode.as_ref() + &self.chol.dot(z);

        // === Step 2: Compute η = X @ β ===
        let eta = self.data.x.dot(&beta);

        // === Step 3: Compute log-likelihood and gradient ===
        let (ll, grad_ll_beta) = if self.is_logit {
            self.logit_logp_and_grad(&eta)
        } else {
            self.gaussian_logp_and_grad(&eta)
        };

        // === Step 4: Compute penalty and its gradient ===
        // penalty = 0.5 * β^T @ S @ β
        let s_beta = self.data.penalty.dot(&beta);
        let penalty = 0.5 * beta.dot(&s_beta);

        // ∇_β penalty = S @ β
        let grad_penalty_beta = s_beta;

        // === Step 5: Combined gradient in β space ===
        // ∇_β log p = ∇_β ll - ∇_β penalty
        let grad_beta = &grad_ll_beta - &grad_penalty_beta;

        // === Step 6: Chain rule to get gradient in z space ===
        // ∇_z = L^T @ ∇_β
        let grad_z = self.chol_t.dot(&grad_beta);

        let logp = ll - penalty;
        (logp, grad_z)
    }

    /// Logistic regression log-likelihood and gradient.
    fn logit_logp_and_grad(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        let n = self.data.n_samples;
        let mut ll = 0.0;
        let mut residual = Array1::<f64>::zeros(n);

        for i in 0..n {
            let eta_i = eta[i].clamp(-700.0, 700.0);
            let mu_i = 1.0 / (1.0 + (-eta_i).exp());
            let mu_clamped = mu_i.clamp(1e-10, 1.0 - 1e-10);

            // Log-likelihood: y*log(μ) + (1-y)*log(1-μ)
            let y_i = self.data.y[i];
            let w_i = self.data.weights[i];
            ll += w_i * (y_i * mu_clamped.ln() + (1.0 - y_i) * (1.0 - mu_clamped).ln());

            // Residual for gradient: y - μ (canonical link, score function)
            residual[i] = w_i * (y_i - mu_clamped);
        }

        // Gradient of log-likelihood: X^T @ (w * (y - μ))
        let grad_ll = self.data.x.t().dot(&residual);

        (ll, grad_ll)
    }

    /// Gaussian log-likelihood and gradient.
    fn gaussian_logp_and_grad(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        let n = self.data.n_samples;
        let mut ll = 0.0;
        let mut weighted_residual = Array1::<f64>::zeros(n);

        for i in 0..n {
            let residual = self.data.y[i] - eta[i];
            let w_i = self.data.weights[i];
            ll -= 0.5 * w_i * residual * residual;
            weighted_residual[i] = w_i * residual;
        }

        // Gradient of log-likelihood: X^T @ (w * (y - η))
        let grad_ll = self.data.x.t().dot(&weighted_residual);

        (ll, grad_ll)
    }

    /// Get the Cholesky factor L for un-whitening samples
    pub fn chol(&self) -> &Array2<f64> {
        &self.chol
    }

    /// Get the mode
    pub fn mode(&self) -> &Array1<f64> {
        &self.data.mode
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.data.dim
    }
}

/// Implement HamiltonianTarget for NUTS with analytical gradients.
impl HamiltonianTarget<Array1<f64>> for NutsPosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let (logp, grad_z) = self.compute_logp_and_grad_nd(position);
        grad.assign(&grad_z);
        logp
    }
}

/// Configuration for NUTS sampling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NutsConfig {
    /// Number of samples to collect (after warmup)
    pub n_samples: usize,
    /// Number of warmup samples to discard
    pub n_warmup: usize,
    /// Number of parallel chains
    pub n_chains: usize,
    /// Target acceptance probability (0.6-0.9 recommended)
    pub target_accept: f64,
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            n_warmup: 500,
            n_chains: 4,
            target_accept: 0.8,
        }
    }
}

/// Result of NUTS sampling.
#[derive(Clone, Debug)]
pub struct NutsResult {
    /// Coefficient samples in ORIGINAL space: shape (n_total_samples, n_coeffs)
    pub samples: Array2<f64>,
    /// Posterior mean
    pub posterior_mean: Array1<f64>,
    /// Posterior standard deviation
    pub posterior_std: Array1<f64>,
    /// R-hat convergence diagnostic
    pub rhat: f64,
    /// Effective sample size
    pub ess: f64,
    /// Whether sampling converged (R-hat < 1.1)
    pub converged: bool,
}

impl NutsResult {
    /// Computes the posterior mean of a function applied to coefficients.
    pub fn posterior_mean_of<F>(&self, f: F) -> f64
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let n = self.samples.nrows();
        let mut sum = 0.0;
        for i in 0..n {
            sum += f(self.samples.row(i));
        }
        sum / n as f64
    }

    /// Computes percentiles of a function applied to coefficients.
    pub fn posterior_interval_of<F>(&self, f: F, lower_pct: f64, upper_pct: f64) -> (f64, f64)
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let n = self.samples.nrows();
        let mut values: Vec<f64> = (0..n).map(|i| f(self.samples.row(i))).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = ((lower_pct / 100.0) * n as f64).floor() as usize;
        let upper_idx = ((upper_pct / 100.0) * n as f64).ceil() as usize;

        (
            values[lower_idx.min(n - 1)],
            values[upper_idx.min(n - 1)],
        )
    }
}

/// Runs NUTS sampling using mini-mcmc with whitened parameter space.
///
/// # Arguments
/// * `x` - Design matrix [n_samples, dim]
/// * `y` - Response vector [n_samples]
/// * `weights` - Prior weights [n_samples]
/// * `penalty_matrix` - Combined penalty S [dim, dim]
/// * `mode` - MAP estimate μ [dim]
/// * `hessian` - Penalized Hessian H [dim, dim] (NOT the inverse!)
/// * `is_logit` - True for logistic regression, false for Gaussian
/// * `config` - NUTS configuration
pub fn run_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    is_logit: bool,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    let dim = mode.len();

    // Create posterior target with analytical gradients
    let target = NutsPosterior::new(x, y, weights, penalty_matrix, mode, hessian, is_logit)?;

    // Get Cholesky factor for un-whitening samples later
    let chol = target.chol().clone();
    let mode_arr = target.mode().clone();

    // Initialize chains at z=0 with small jitter
    let mut rng = rand::thread_rng();
    let initial_positions: Vec<Array1<f64>> = (0..config.n_chains)
        .map(|_| {
            Array1::from_shape_fn(dim, |_| {
                let u1: f64 = rand::Rng::r#gen(&mut rng);
                let u2: f64 = rand::Rng::r#gen(&mut rng);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * 0.1
            })
        })
        .collect();

    // Create GenericNUTS sampler - it auto-tunes step size!
    let mut sampler = GenericNUTS::new(target, initial_positions, config.target_accept);

    // Run sampling with progress bar
    let (samples_array, run_stats) = sampler
        .run_progress(config.n_samples, config.n_warmup)
        .map_err(|e| format!("NUTS sampling failed: {}", e))?;

    log::info!("NUTS sampling complete: {}", run_stats);

    // Convert samples from whitened space back to original space
    // samples_array has shape [n_chains, n_samples, dim]
    let shape = samples_array.shape();
    let n_chains = shape[0];
    let n_samples_out = shape[1];
    let total_samples = n_chains * n_samples_out;

    let mut samples = Array2::<f64>::zeros((total_samples, dim));
    let mut z_buffer = Array1::<f64>::zeros(dim); // Reuse buffer to avoid per-sample allocations
    for chain in 0..n_chains {
        for sample_i in 0..n_samples_out {
            // Get z (whitened coordinates) directly from Array3
            let z_view = samples_array.slice(ndarray::s![chain, sample_i, ..]);
            z_buffer.assign(&z_view);

            // Transform to β: β = μ + L @ z
            let beta = &mode_arr + &chol.dot(&z_buffer);

            let sample_idx = chain * n_samples_out + sample_i;
            samples.row_mut(sample_idx).assign(&beta);
        }
    }

    // Compute statistics
    let posterior_mean = samples.mean_axis(Axis(0)).unwrap();
    let posterior_std = samples.std_axis(Axis(0), 0.0);
    let rhat = f64::from(run_stats.rhat.mean);
    let ess = f64::from(run_stats.ess.mean);
    let converged = rhat < 1.1;

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    })
}

// ============================================================================
// Survival Model HMC Support
// ============================================================================

#[cfg(feature = "survival-data")]
mod survival_hmc {
    use super::*;
    use crate::calibrate::survival::{
        MonotonicityPenalty, SurvivalLayout, SurvivalSpec, SurvivalTrainingData,
        WorkingModelSurvival,
    };

    /// Shared data for survival NUTS posterior (wrapped in Arc to prevent cloning).
    #[derive(Clone)]
    struct SharedSurvivalData {
        /// Survival layout with design matrices and penalties
        layout: Arc<SurvivalLayout>,
        /// Sample weights
        sample_weight: Arc<Array1<f64>>,
        /// Event indicators (1 = event, 0 = censored)
        event_target: Arc<Array1<u8>>,
        /// Entry ages
        age_entry: Arc<Array1<f64>>,
        /// Exit ages
        age_exit: Arc<Array1<f64>>,
        /// Monotonicity constraint
        monotonicity: Arc<MonotonicityPenalty>,
        /// Survival spec
        spec: SurvivalSpec,
        /// MAP estimate (mode) μ [dim]
        mode: Arc<Array1<f64>>,
    }

    /// Whitened log-posterior target for survival models with analytical gradients.
    #[derive(Clone)]
    pub struct SurvivalPosterior {
        /// Shared read-only data (Arc prevents duplication)
        data: SharedSurvivalData,
        /// Transform: L where L L^T = H^{-1}
        chol: Array2<f64>,
        /// L^T for gradient chain rule: ∇_z = L^T @ ∇_β
        chol_t: Array2<f64>,
    }

    impl SurvivalPosterior {
        /// Creates a new survival posterior target.
        pub fn new(
            layout: SurvivalLayout,
            training_data: &SurvivalTrainingData,
            monotonicity: MonotonicityPenalty,
            spec: SurvivalSpec,
            mode: ArrayView1<f64>,
            hessian: ArrayView2<f64>,
        ) -> Result<Self, String> {
            let dim = mode.len();

            // Compute whitening transform via Cholesky of Hessian
            let hessian_owned = hessian.to_owned();
            let chol_factor = hessian_owned
                .cholesky(Side::Lower)
                .map_err(|e| format!("Hessian Cholesky decomposition failed: {:?}", e))?;
            let l_h = chol_factor.lower_triangular();
            let chol = solve_upper_triangular_transpose(&l_h, dim);
            let chol_t = chol.t().to_owned();

            let data = SharedSurvivalData {
                layout: Arc::new(layout),
                sample_weight: Arc::new(training_data.sample_weight.clone()),
                event_target: Arc::new(training_data.event_target.clone()),
                age_entry: Arc::new(training_data.age_entry.clone()),
                age_exit: Arc::new(training_data.age_exit.clone()),
                monotonicity: Arc::new(monotonicity),
                spec,
                mode: Arc::new(mode.to_owned()),
            };

            Ok(Self {
                data,
                chol,
                chol_t,
            })
        }

        /// Compute log-posterior and gradient analytically.
        fn compute_logp_and_grad(&self, z: &Array1<f64>) -> Result<(f64, Array1<f64>), String> {
            // Transform z (whitened) -> β (original): β = μ + L @ z
            let beta = self.data.mode.as_ref() + &self.chol.dot(z);

            // Create a temporary working model to compute likelihood
            // We need owned copies for the working model
            let layout_clone = (*self.data.layout).clone();
            let mut model = WorkingModelSurvival {
                layout: layout_clone,
                sample_weight: (*self.data.sample_weight).clone(),
                event_target: (*self.data.event_target).clone(),
                age_entry: (*self.data.age_entry).clone(),
                age_exit: (*self.data.age_exit).clone(),
                monotonicity: (*self.data.monotonicity).clone(),
                spec: self.data.spec,
            };

            // Compute state (deviance and gradient in beta space)
            let state = model
                .update_state(&beta)
                .map_err(|e| format!("Survival state update failed: {:?}", e))?;

            // Convert deviance to log-posterior: logp = -0.5 * deviance
            // (deviance = -2 * log_likelihood + penalty, so -0.5 * deviance = log_likelihood - 0.5*penalty)
            let logp = -0.5 * state.deviance;

            // The gradient from update_state is ∂deviance/∂β = -2 * ∂log_posterior/∂β
            // So ∂log_posterior/∂β = -0.5 * gradient
            let grad_beta = state.gradient.mapv(|g| -0.5 * g);

            // Chain rule to get gradient in z space: ∇_z = L^T @ ∇_β
            let grad_z = self.chol_t.dot(&grad_beta);

            Ok((logp, grad_z))
        }

        /// Get the Cholesky factor L for un-whitening samples
        pub fn chol(&self) -> &Array2<f64> {
            &self.chol
        }

        /// Get the mode
        pub fn mode(&self) -> &Array1<f64> {
            &self.data.mode
        }
    }

    impl HamiltonianTarget<Array1<f64>> for SurvivalPosterior {
        fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
            match self.compute_logp_and_grad(position) {
                Ok((logp, grad_z)) => {
                    grad.assign(&grad_z);
                    logp
                }
                Err(e) => {
                    // On error (e.g., monotonicity violation), return -infinity log-prob
                    // This causes NUTS to reject the proposal
                    log::warn!("Survival posterior evaluation failed: {}", e);
                    grad.fill(0.0);
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Runs NUTS sampling for survival models with whitened parameter space.
    pub fn run_survival_nuts_sampling(
        layout: SurvivalLayout,
        training_data: &SurvivalTrainingData,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        config: &NutsConfig,
    ) -> Result<NutsResult, String> {
        let dim = mode.len();

        // Create posterior target
        let target = SurvivalPosterior::new(
            layout,
            training_data,
            monotonicity,
            spec,
            mode,
            hessian,
        )?;

        // Get Cholesky factor for un-whitening samples later
        let chol = target.chol().clone();
        let mode_arr = target.mode().clone();

        // Initialize chains at z=0 with small jitter
        let mut rng = rand::thread_rng();
        let initial_positions: Vec<Array1<f64>> = (0..config.n_chains)
            .map(|_| {
                Array1::from_shape_fn(dim, |_| {
                    let u1: f64 = rand::Rng::r#gen(&mut rng);
                    let u2: f64 = rand::Rng::r#gen(&mut rng);
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * 0.1
                })
            })
            .collect();

        // Create GenericNUTS sampler
        let mut sampler = GenericNUTS::new(target, initial_positions, config.target_accept);

        // Run sampling with progress bar
        let (samples_array, run_stats) = sampler
            .run_progress(config.n_samples, config.n_warmup)
            .map_err(|e| format!("NUTS sampling failed: {}", e))?;

        log::info!("Survival NUTS sampling complete: {}", run_stats);

        // Convert samples from whitened space back to original space
        let shape = samples_array.shape();
        let n_chains = shape[0];
        let n_samples_out = shape[1];
        let total_samples = n_chains * n_samples_out;

        let mut samples = Array2::<f64>::zeros((total_samples, dim));
        let mut z_buffer = Array1::<f64>::zeros(dim);
        for chain in 0..n_chains {
            for sample_i in 0..n_samples_out {
                let z_view = samples_array.slice(ndarray::s![chain, sample_i, ..]);
                z_buffer.assign(&z_view);

                // Transform to β: β = μ + L @ z
                let beta = &mode_arr + &chol.dot(&z_buffer);

                let sample_idx = chain * n_samples_out + sample_i;
                samples.row_mut(sample_idx).assign(&beta);
            }
        }

        // Compute statistics
        let posterior_mean = samples.mean_axis(Axis(0)).unwrap();
        let posterior_std = samples.std_axis(Axis(0), 0.0);
        let rhat = f64::from(run_stats.rhat.mean);
        let ess = f64::from(run_stats.ess.mean);
        let converged = rhat < 1.1;

        Ok(NutsResult {
            samples,
            posterior_mean,
            posterior_std,
            rhat,
            ess,
            converged,
        })
    }
}

#[cfg(feature = "survival-data")]
pub use survival_hmc::run_survival_nuts_sampling;

// Legacy type alias for backwards compatibility
pub type WhitenedBurnPosterior = NutsPosterior;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_creation() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![1.0];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        // Pass Hessian (not inverse) - identity matrix
        let hessian = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true,
        )
        .unwrap();

        assert_eq!(target.dim(), 1);
    }

    #[test]
    fn test_analytical_gradient_gaussian() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![0.5];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        let hessian = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            false, // Gaussian
        )
        .unwrap();

        let z = ndarray::array![0.0];
        let (logp, grad) = target.compute_logp_and_grad_nd(&z);

        // At z=0, β=0, eta=0, residual=0.5
        // log L = -0.5 * 0.5^2 = -0.125
        assert!(
            (logp - (-0.125)).abs() < 0.01,
            "Expected logp~-0.125, got {}",
            logp
        );

        // Gradient: X^T @ (w * (y - η)) = 1 * 1 * 0.5 = 0.5
        // Chain rule: L^T @ grad_β = 1 * 0.5 = 0.5
        assert!(
            (grad[0] - 0.5).abs() < 0.01,
            "Expected grad~0.5, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_analytical_gradient_logit() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![1.0];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        let hessian = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true, // Logit
        )
        .unwrap();

        let z = ndarray::array![0.0];
        let (logp, grad) = target.compute_logp_and_grad_nd(&z);

        // At z=0, β=0, eta=0, μ=0.5
        // log L = y*log(0.5) + (1-y)*log(0.5) = 1*log(0.5) = -0.693
        assert!(
            (logp - (-0.693)).abs() < 0.01,
            "Expected logp~-0.693, got {}",
            logp
        );

        // Gradient: X^T @ (w * (y - μ)) = 1 * 1 * (1 - 0.5) = 0.5
        assert!(
            (grad[0] - 0.5).abs() < 0.01,
            "Expected grad~0.5, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_gradient_vs_finite_difference() {
        let x = ndarray::array![[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]];
        let y = ndarray::array![1.0, 0.0, 1.0];
        let weights = ndarray::array![1.0, 1.0, 1.0];
        let penalty = ndarray::array![[0.1, 0.0], [0.0, 0.1]];
        let mode = ndarray::array![0.0, 0.0];
        let hessian = ndarray::array![[1.0, 0.0], [0.0, 1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true, // Logit
        )
        .unwrap();

        let z = ndarray::array![0.5, -0.3];
        let (_, grad) = target.compute_logp_and_grad_nd(&z);

        // Finite difference check
        let eps = 1e-5;
        for i in 0..2 {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[i] += eps;
            z_minus[i] -= eps;

            let (logp_plus, _) = target.compute_logp_and_grad_nd(&z_plus);
            let (logp_minus, _) = target.compute_logp_and_grad_nd(&z_minus);

            let fd_grad = (logp_plus - logp_minus) / (2.0 * eps);
            let rel_error = (grad[i] - fd_grad).abs() / (grad[i].abs().max(1e-8));

            assert!(
                rel_error < 1e-4,
                "Gradient mismatch at index {}: analytical={}, fd={}, rel_error={}",
                i,
                grad[i],
                fd_grad,
                rel_error
            );
        }
    }

    #[test]
    fn test_arc_prevents_cloning_data() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = ndarray::array![1.0, 0.0];
        let weights = ndarray::array![1.0, 1.0];
        let penalty = ndarray::array![[0.1, 0.0], [0.0, 0.1]];
        let mode = ndarray::array![0.0, 0.0];
        let hessian = ndarray::array![[1.0, 0.0], [0.0, 1.0]];

        let target1 = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true,
        )
        .unwrap();

        // Clone should share data via Arc, not duplicate
        let target2 = target1.clone();

        // Verify Arc::ptr_eq - same underlying allocation
        assert!(Arc::ptr_eq(&target1.data.x, &target2.data.x));
        assert!(Arc::ptr_eq(&target1.data.y, &target2.data.y));
    }

    #[test]
    fn test_whitening_transform_is_correct() {
        // Create a non-trivial positive definite Hessian
        let hessian = ndarray::array![[4.0, 1.0], [1.0, 3.0]];
        let dim = 2;

        // Compute the whitening transform via our function
        let hessian_owned = hessian.clone();
        let chol_factor = hessian_owned
            .cholesky(faer::Side::Lower)
            .expect("Cholesky should succeed");
        let l_h = chol_factor.lower_triangular();
        let chol = solve_upper_triangular_transpose(&l_h, dim);

        // Verify: L L^T should equal H^{-1}
        let llt = chol.dot(&chol.t());

        // Compute H^{-1} explicitly for comparison
        let det = 4.0 * 3.0 - 1.0 * 1.0; // = 11
        let h_inv = ndarray::array![
            [3.0 / det, -1.0 / det],
            [-1.0 / det, 4.0 / det]
        ];

        // Check each element
        for i in 0..dim {
            for j in 0..dim {
                let diff = (llt[[i, j]] - h_inv[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Whitening transform error at [{},{}]: got {}, expected {}",
                    i, j, llt[[i, j]], h_inv[[i, j]]
                );
            }
        }
    }
}
