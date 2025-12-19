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
//! # Autodiff
//!
//! The log-posterior is implemented using ONLY burn tensor operations
//! so that mini-mcmc's autodiff can compute gradients via .backward().

use burn::backend::{Autodiff, NdArray};
use burn::prelude::*;
use burn::tensor::TensorData;
use mini_mcmc::distributions::GradientTarget;
use mini_mcmc::nuts::NUTS;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// Backend type for NUTS
pub type NutsBackend = Autodiff<NdArray>;

/// Whitened log-posterior target using burn tensors for autodiff.
///
/// Stores data as flattened f32 vectors and shapes, then constructs
/// burn tensors inside unnorm_logp to preserve the autodiff graph.
#[derive(Clone)]
pub struct WhitenedBurnPosterior {
    /// Design matrix X flattened (n_samples × n_coeffs)
    x_data: Vec<f32>,
    /// Response vector y
    y_data: Vec<f32>,
    /// Prior weights
    weights_data: Vec<f32>,
    /// Combined penalty matrix S flattened
    penalty_data: Vec<f32>,
    /// MAP estimate (mode)
    mode_data: Vec<f32>,
    /// Cholesky factor L flattened
    chol_data: Vec<f32>,
    /// Link function type
    is_logit: bool,
    /// Number of samples
    n_samples: usize,
    /// Number of coefficients
    dim: usize,
}

impl WhitenedBurnPosterior {
    /// Creates a new whitened burn posterior from ndarray data.
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
        
        // Store as flattened f32 vectors
        let x_data: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y_data: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let weights_data: Vec<f32> = weights.iter().map(|&v| v as f32).collect();
        let penalty_data: Vec<f32> = penalty_matrix.iter().map(|&v| v as f32).collect();
        let mode_data: Vec<f32> = mode.iter().map(|&v| v as f32).collect();
        
        // Compute Cholesky factor of inverse Hessian
        let chol = cholesky_lower(inv_hessian);
        let chol_data: Vec<f32> = chol.iter().map(|&v| v as f32).collect();
        
        Self {
            x_data,
            y_data,
            weights_data,
            penalty_data,
            mode_data,
            chol_data,
            is_logit,
            n_samples,
            dim,
        }
    }
}

/// Implement GradientTarget for NUTS.
impl GradientTarget<f32, NutsBackend> for WhitenedBurnPosterior {
    fn unnorm_logp(&self, z: Tensor<NutsBackend, 1>) -> Tensor<NutsBackend, 1> {
        let device = z.device();
        let dim = self.dim;
        let n = self.n_samples;
        
        // === Step 1: Transform z (whitened) -> β (original) ===
        // β = μ + L @ z
        
        // Create mode tensor [dim]
        let mode = Tensor::<NutsBackend, 1>::from_data(
            TensorData::new(self.mode_data.clone(), [dim]),
            &device,
        );
        
        // Create chol tensor [dim, dim]
        let chol = Tensor::<NutsBackend, 2>::from_data(
            TensorData::new(self.chol_data.clone(), [dim, dim]),
            &device,
        );
        
        // L @ z: [dim, dim] @ [dim, 1] -> [dim, 1]
        let z_col: Tensor<NutsBackend, 2> = z.clone().unsqueeze_dim(1);
        let lz: Tensor<NutsBackend, 2> = chol.matmul(z_col);
        let lz_flat: Tensor<NutsBackend, 1> = lz.squeeze(1);
        let beta: Tensor<NutsBackend, 1> = mode.add(lz_flat);
        
        // === Step 2: Compute log-likelihood ===
        
        // Create X tensor [n_samples, dim]
        let x = Tensor::<NutsBackend, 2>::from_data(
            TensorData::new(self.x_data.clone(), [n, dim]),
            &device,
        );
        
        // Create y tensor [n_samples]
        let y = Tensor::<NutsBackend, 1>::from_data(
            TensorData::new(self.y_data.clone(), [n]),
            &device,
        );
        
        // Create weights tensor [n_samples]
        let w = Tensor::<NutsBackend, 1>::from_data(
            TensorData::new(self.weights_data.clone(), [n]),
            &device,
        );
        
        // eta = X @ β: [n_samples, dim] @ [dim, 1] -> [n_samples, 1]
        let beta_col: Tensor<NutsBackend, 2> = beta.clone().unsqueeze_dim(1);
        let eta_col: Tensor<NutsBackend, 2> = x.matmul(beta_col.clone());
        let eta: Tensor<NutsBackend, 1> = eta_col.squeeze(1);
        
        let ll: Tensor<NutsBackend, 1> = if self.is_logit {
            // Logistic log-likelihood
            // p = sigmoid(eta) = 1 / (1 + exp(-eta))
            let eta_clamped = eta.clamp(-20.0, 20.0);
            let neg_eta = eta_clamped.neg();
            let exp_neg_eta = neg_eta.exp();
            let ones = Tensor::<NutsBackend, 1>::ones([n], &device);
            let prob = ones.clone().div(ones.clone().add(exp_neg_eta));
            
            // Clamp probabilities
            let prob_clamped = prob.clamp(1e-7, 1.0 - 1e-7);
            
            // log(p) and log(1-p)
            let log_p = prob_clamped.clone().log();
            let log_1mp = ones.clone().sub(prob_clamped).log();
            
            // y * log(p) + (1-y) * log(1-p)
            let one_minus_y = ones.sub(y.clone());
            let ll_terms = y.mul(log_p).add(one_minus_y.mul(log_1mp));
            
            // Weighted sum -> scalar, then reshape to 1D to preserve autodiff
            ll_terms.mul(w).sum().reshape([1])
        } else {
            // Gaussian log-likelihood: -0.5 * sum(w * (y - eta)^2)
            let residual = y.sub(eta);
            let residual_sq = residual.clone().mul(residual);
            let weighted_sq = residual_sq.mul(w);
            // Sum -> scalar, then reshape to 1D to preserve autodiff
            weighted_sq.sum().mul_scalar(-0.5_f32).reshape([1])
        };
        
        // === Step 3: Compute penalty ===
        // penalty = 0.5 * β^T @ S @ β
        
        let s = Tensor::<NutsBackend, 2>::from_data(
            TensorData::new(self.penalty_data.clone(), [dim, dim]),
            &device,
        );
        
        // S @ β: [dim, dim] @ [dim, 1] -> [dim, 1]
        let s_beta: Tensor<NutsBackend, 2> = s.matmul(beta_col.clone());
        
        // β^T @ (S @ β): [1, dim] @ [dim, 1] -> [1, 1]
        let beta_row: Tensor<NutsBackend, 2> = beta.unsqueeze_dim(0);
        let penalty_mat: Tensor<NutsBackend, 2> = beta_row.matmul(s_beta);
        let penalty: Tensor<NutsBackend, 1> = penalty_mat.flatten(0, 1).mul_scalar(0.5_f32);
        
        // === Step 4: Return log P = log L - penalty ===
        ll.sub(penalty)
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
    pub target_accept: f32,
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
    
    // Create whitened burn posterior
    let target = WhitenedBurnPosterior::new(
        x, y, weights, penalty_matrix, mode, inv_hessian, is_logit,
    );
    
    // Get Cholesky factor for un-whitening later
    let chol = cholesky_lower(inv_hessian);
    let mode_arr: Array1<f64> = mode.to_owned();
    
    // Initialize chains at z=0 with small jitter
    let mut rng = rand::thread_rng();
    let initial_positions: Vec<Vec<f32>> = (0..config.n_chains)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let u1: f64 = rand::Rng::r#gen(&mut rng);
                    let u2: f64 = rand::Rng::r#gen(&mut rng);
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    (z * 0.1) as f32
                })
                .collect()
        })
        .collect();
    
    // Create NUTS sampler - it auto-tunes step size!
    let mut sampler = NUTS::<f32, NutsBackend, WhitenedBurnPosterior>::new(
        target,
        initial_positions,
        config.target_accept,
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
    
    let data: Vec<f32> = samples_tensor.into_data().to_vec().unwrap();
    
    let mut samples = Array2::<f64>::zeros((total_samples, dim));
    for chain in 0..n_chains {
        for sample in 0..n_samples_out {
            // Get z (whitened coordinates)
            let z: Array1<f64> = (0..dim)
                .map(|d| data[chain * n_samples_out * dim + sample * dim + d] as f64)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whitened_posterior_creation() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![1.0];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        let inv_h = ndarray::array![[1.0]];
        
        let target = WhitenedBurnPosterior::new(
            x.view(), y.view(), weights.view(),
            penalty.view(), mode.view(), inv_h.view(),
            true,
        );
        
        assert_eq!(target.dim, 1);
        assert_eq!(target.n_samples, 1);
    }

    #[test]
    fn test_gradient_target_computes() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![0.5];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        let inv_h = ndarray::array![[1.0]];
        
        let target = WhitenedBurnPosterior::new(
            x.view(), y.view(), weights.view(),
            penalty.view(), mode.view(), inv_h.view(),
            false,  // Gaussian
        );
        
        // Verify data was stored correctly
        assert_eq!(target.x_data.len(), 1);
        assert_eq!(target.y_data.len(), 1);
        assert_eq!(target.chol_data.len(), 1);
        
        let device = Default::default();
        let z = Tensor::<NutsBackend, 1>::from_data(
            TensorData::new(vec![0.0_f32], [1]),
            &device,
        );
        let logp = target.unnorm_logp(z);
        
        // At z=0, β=0, eta=0, residual=0.5
        // log L = -0.5 * 0.5^2 = -0.125
        let logp_val: Vec<f32> = logp.into_data().to_vec().unwrap();
        assert!((logp_val[0] - (-0.125)).abs() < 0.01, 
            "Expected logp~-0.125, got {}", logp_val[0]);
    }
}
