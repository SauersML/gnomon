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

use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::TensorData;
use mini_mcmc::distributions::GradientTarget;
use mini_mcmc::nuts::NUTS;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// Backend type for NUTS - uses f64 for numerical precision
pub type NutsBackend = Autodiff<NdArray<f64>>;

/// Whitened log-posterior target with analytical gradients.
///
/// Stores data as ndarray (f64) and computes gradients analytically,
/// overriding mini-mcmc's default autodiff behavior.
#[derive(Clone)]
pub struct NutsPosterior {
    /// Design matrix X [n_samples, dim]
    x: Array2<f64>,
    /// Response vector y [n_samples]
    y: Array1<f64>,
    /// Prior weights [n_samples]
    weights: Array1<f64>,
    /// Combined penalty matrix S [dim, dim]
    penalty: Array2<f64>,
    /// MAP estimate (mode) μ [dim]
    mode: Array1<f64>,
    /// Precomputed transform: L where L L^T = H^{-1}
    /// Used for z -> β: β = μ + L @ z
    chol: Array2<f64>,
    /// Precomputed L^T for gradient chain rule: ∇_z = L^T @ ∇_β
    chol_t: Array2<f64>,
    /// Link function type
    is_logit: bool,
    /// Number of samples
    n_samples: usize,
    /// Number of coefficients
    dim: usize,
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
    /// * `inv_hessian` - Inverse Hessian H^{-1} [dim, dim]
    /// * `is_logit` - True for logistic regression, false for Gaussian
    pub fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        penalty_matrix: ArrayView2<f64>,
        mode: ArrayView1<f64>,
        inv_hessian: ArrayView2<f64>,
        is_logit: bool,
    ) -> Self {
        let n_samples = x.nrows();
        let dim = x.ncols();

        // Compute Cholesky factor L where L L^T = H^{-1}
        let chol = cholesky_lower(inv_hessian);
        let chol_t = chol.t().to_owned();

        Self {
            x: x.to_owned(),
            y: y.to_owned(),
            weights: weights.to_owned(),
            penalty: penalty_matrix.to_owned(),
            mode: mode.to_owned(),
            chol,
            chol_t,
            is_logit,
            n_samples,
            dim,
        }
    }

    /// Compute log-posterior and gradient analytically using ndarray.
    ///
    /// Returns (log_posterior, gradient_z) where gradient_z is the gradient
    /// with respect to the whitened parameters z.
    fn compute_logp_and_grad_nd(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
        // === Step 1: Transform z (whitened) -> β (original) ===
        // β = μ + L @ z
        let beta = &self.mode + &self.chol.dot(z);

        // === Step 2: Compute η = X @ β ===
        let eta = self.x.dot(&beta);

        // === Step 3: Compute log-likelihood and gradient ===
        let (ll, grad_ll_beta) = if self.is_logit {
            self.logit_logp_and_grad(&eta)
        } else {
            self.gaussian_logp_and_grad(&eta)
        };

        // === Step 4: Compute penalty and its gradient ===
        // penalty = 0.5 * β^T @ S @ β
        let s_beta = self.penalty.dot(&beta);
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
        let n = self.n_samples;
        let mut ll = 0.0;
        let mut residual = Array1::<f64>::zeros(n);

        for i in 0..n {
            let eta_i = eta[i].clamp(-700.0, 700.0);
            let mu_i = 1.0 / (1.0 + (-eta_i).exp());
            let mu_clamped = mu_i.clamp(1e-10, 1.0 - 1e-10);

            // Log-likelihood: y*log(μ) + (1-y)*log(1-μ)
            let y_i = self.y[i];
            let w_i = self.weights[i];
            ll += w_i * (y_i * mu_clamped.ln() + (1.0 - y_i) * (1.0 - mu_clamped).ln());

            // Residual for gradient: y - μ (canonical link, score function)
            residual[i] = w_i * (y_i - mu_clamped);
        }

        // Gradient of log-likelihood: X^T @ (w * (y - μ))
        let grad_ll = self.x.t().dot(&residual);

        (ll, grad_ll)
    }

    /// Gaussian log-likelihood and gradient.
    fn gaussian_logp_and_grad(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        let n = self.n_samples;
        let mut ll = 0.0;
        let mut weighted_residual = Array1::<f64>::zeros(n);

        for i in 0..n {
            let residual = self.y[i] - eta[i];
            let w_i = self.weights[i];
            ll -= 0.5 * w_i * residual * residual;
            weighted_residual[i] = w_i * residual;
        }

        // Gradient of log-likelihood: X^T @ (w * (y - η))
        let grad_ll = self.x.t().dot(&weighted_residual);

        (ll, grad_ll)
    }
}

/// Implement GradientTarget for NUTS with analytical gradients.
impl GradientTarget<f64, NutsBackend> for NutsPosterior {
    fn unnorm_logp(&self, z: Tensor<NutsBackend, 1>) -> Tensor<NutsBackend, 1> {
        // Convert tensor to ndarray
        let z_data: Vec<f64> = z.clone().into_data().to_vec().unwrap();
        let z_arr = Array1::from_vec(z_data);

        // Compute log-posterior (discard gradient)
        let (logp, _) = self.compute_logp_and_grad_nd(&z_arr);

        // Wrap scalar in tensor
        let device = z.device();
        Tensor::<NutsBackend, 1>::from_data(TensorData::new(vec![logp], [1]), &device)
    }

    fn unnorm_logp_and_grad(
        &self,
        z: Tensor<NutsBackend, 1>,
    ) -> (Tensor<NutsBackend, 1>, Tensor<NutsBackend, 1>) {
        let device = z.device();

        // Convert tensor to ndarray
        let z_data: Vec<f64> = z.into_data().to_vec().unwrap();
        let z_arr = Array1::from_vec(z_data);

        // Compute log-posterior AND gradient analytically
        let (logp, grad_z) = self.compute_logp_and_grad_nd(&z_arr);

        // Convert back to tensors
        let logp_tensor =
            Tensor::<NutsBackend, 1>::from_data(TensorData::new(vec![logp], [1]), &device);

        let grad_data: Vec<f64> = grad_z.to_vec();
        let grad_tensor = Tensor::<NutsBackend, 1>::from_data(
            TensorData::new(grad_data, [self.dim]),
            &device,
        );

        (logp_tensor, grad_tensor)
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
pub fn run_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    inv_hessian: ArrayView2<f64>,
    is_logit: bool,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    let dim = mode.len();

    // Create posterior target with analytical gradients
    let target = NutsPosterior::new(x, y, weights, penalty_matrix, mode, inv_hessian, is_logit);

    // Get Cholesky factor for un-whitening samples later
    let chol = cholesky_lower(inv_hessian);
    let mode_arr: Array1<f64> = mode.to_owned();

    // Initialize chains at z=0 with small jitter
    let mut rng = rand::thread_rng();
    let initial_positions: Vec<Vec<f64>> = (0..config.n_chains)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let u1: f64 = rand::Rng::r#gen(&mut rng);
                    let u2: f64 = rand::Rng::r#gen(&mut rng);
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * 0.1
                })
                .collect()
        })
        .collect();

    // Create NUTS sampler - it auto-tunes step size!
    let mut sampler = NUTS::<f64, NutsBackend, NutsPosterior>::new(
        target,
        initial_positions,
        config.target_accept, // f64 throughout
    );

    // Run sampling with progress bar
    let (samples_tensor, run_stats) = sampler
        .run_progress(config.n_samples, config.n_warmup)
        .map_err(|e| format!("NUTS sampling failed: {}", e))?;

    log::info!("NUTS sampling complete: {}", run_stats);

    // Convert samples from whitened space back to original space
    let shape = samples_tensor.dims();
    let n_chains = shape[0];
    let n_samples_out = shape[1];
    let total_samples = n_chains * n_samples_out;

    let data: Vec<f64> = samples_tensor.into_data().to_vec().unwrap();

    let mut samples = Array2::<f64>::zeros((total_samples, dim));
    for chain in 0..n_chains {
        for sample in 0..n_samples_out {
            // Get z (whitened coordinates)
            let z: Array1<f64> = (0..dim)
                .map(|d| data[chain * n_samples_out * dim + sample * dim + d])
                .collect();

            // Transform to β: β = μ + L @ z
            let beta = &mode_arr + &chol.dot(&z);

            let sample_idx = chain * n_samples_out + sample;
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

/// Cholesky decomposition (lower triangular).
fn cholesky_lower(a: ArrayView2<f64>) -> Array2<f64> {
    let n = a.nrows();
    let mut l = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                l[[i, j]] = (sum.max(1e-10)).sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]].max(1e-10);
            }
        }
    }

    l
}

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
        let inv_h = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            inv_h.view(),
            true,
        );

        assert_eq!(target.dim, 1);
        assert_eq!(target.n_samples, 1);
    }

    #[test]
    fn test_analytical_gradient_gaussian() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![0.5];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        let inv_h = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            inv_h.view(),
            false, // Gaussian
        );

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
        let inv_h = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            inv_h.view(),
            true, // Logit
        );

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
        let inv_h = ndarray::array![[1.0, 0.0], [0.0, 1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            inv_h.view(),
            true, // Logit
        );

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
}
