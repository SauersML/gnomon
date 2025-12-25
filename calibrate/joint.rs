//! Joint Single-Index Model with Backfitting
//!
//! This module implements a unified model where the base linear predictor and
//! the flexible link correction are fitted jointly in one REML optimization.
//!
//! Architecture:
//!   η = g(Xβ) where g(u) = u + wiggle(u)
//!
//! Where:
//! - Xβ: High-dimensional predictors (genetics, clinical) with ridge penalty
//! - g(·): Flexible 1D link correction with scale anchor (g(u) = u + B(u)θ)
//!
//! The algorithm:
//! - Outer: BFGS over ρ = [log(λ_base), log(λ_link)]
//! - Inner: Alternating (g|β, β|g with g'(u)*X design)
//! - LAML cost computed via logdet of joint Gauss-Newton Hessian

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::cell::RefCell;
use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::LinkFunction;
use crate::calibrate::basis::{
    create_bspline_basis, create_bspline_basis_with_knots, create_difference_penalty_matrix,
};
use crate::calibrate::construction::{
    compute_penalty_square_roots, precompute_reparam_invariant, stable_reparameterization,
    stable_reparameterization_with_invariant, ReparamInvariant, ReparamResult,
};

/// State for the joint single-index model optimization.
pub struct JointModelState<'a> {
    /// Response variable
    y: ArrayView1<'a, f64>,
    /// Prior weights
    weights: ArrayView1<'a, f64>,
    /// Base design matrix (X for high-dim predictors)
    x_base: ArrayView2<'a, f64>,
    /// Current base coefficients β
    beta_base: Array1<f64>,
    /// Current link wiggle coefficients θ (identity is implicit offset)
    beta_link: Array1<f64>,
    /// Penalty matrices for base block (one per λ)
    s_base: Vec<Array2<f64>>,
    /// Transformed penalty for link block (Z'SZ)
    s_link_constrained: Array2<f64>,
    /// Constraint transform Z (basis → constrained basis)
    link_transform: Array2<f64>,
    /// Current log-smoothing parameters (one per base penalty + one for link)
    rho: Array1<f64>,
    /// Link function (Logit or Identity)
    link: LinkFunction,
    /// Layout for base model
    layout_base: ModelLayout,
    /// Number of internal knots for link spline
    n_link_knots: usize,
    /// B-spline degree (fixed at 3 = cubic)
    degree: usize,
    /// Fixed knot range from training data (min, max)
    knot_range: Option<(f64, f64)>,
    /// Knot vector for B-splines (fixed after first build)
    knot_vector: Option<Array1<f64>>,
    /// Number of constrained basis functions
    n_constrained_basis: usize,
}

/// Configuration for joint model fitting
#[derive(Clone)]
pub struct JointModelConfig {
    /// Maximum backfitting iterations
    pub max_backfit_iter: usize,
    /// Convergence tolerance for backfitting
    pub backfit_tol: f64,
    /// Maximum REML iterations per backfit cycle
    pub max_reml_iter: usize,
    /// REML convergence tolerance
    pub reml_tol: f64,
    /// Number of internal knots for link spline
    pub n_link_knots: usize,
}

impl Default for JointModelConfig {
    fn default() -> Self {
        Self {
            max_backfit_iter: 20,
            backfit_tol: 1e-4,
            max_reml_iter: 50,
            reml_tol: 1e-6,
            n_link_knots: 10,
        }
    }
}

/// Result of joint model fitting - stores everything needed for prediction
pub struct JointModelResult {
    /// Fitted base coefficients β
    pub beta_base: Array1<f64>,
    /// Fitted link wiggle coefficients θ
    pub beta_link: Array1<f64>,
    /// Fitted smoothing parameters (one per penalty)
    pub lambdas: Vec<f64>,
    /// Final deviance
    pub deviance: f64,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Number of backfitting iterations
    pub backfit_iterations: usize,
    /// Converged flag
    pub converged: bool,
    /// Stored knot range for prediction (min, max)
    pub knot_range: (f64, f64),
    /// Stored knot vector for prediction
    pub knot_vector: Array1<f64>,
    /// Constraint transform for prediction
    pub link_transform: Array2<f64>,
    /// B-spline degree
    pub degree: usize,
}

impl<'a> JointModelState<'a> {
    /// Create new joint model state
    pub fn new(
        y: ArrayView1<'a, f64>,
        weights: ArrayView1<'a, f64>,
        x_base: ArrayView2<'a, f64>,
        s_base: Vec<Array2<f64>>,
        layout_base: ModelLayout,
        link: LinkFunction,
        config: &JointModelConfig,
    ) -> Self {
        let n_base = x_base.ncols();
        let degree = 3; // Cubic B-splines
        
        // Number of B-spline basis functions = n_internal_knots + degree + 1
        // After orthogonality constraint (remove 2: intercept + linear): -2
        // So: n_constrained = n_internal_knots + degree + 1 - 2 = k + degree - 1
        let n_raw_basis = config.n_link_knots + degree + 1;
        let n_constrained = n_raw_basis.saturating_sub(2);
        
        // Initialize β to zero, link coefficients to zero (identity is implicit offset)
        let beta_base = Array1::zeros(n_base);
        let beta_link = Array1::zeros(n_constrained);
        
        // Initialize rho (log-lambdas) - one per base penalty + one for link
        let n_penalties = s_base.len() + 1;
        let rho = Array1::zeros(n_penalties);
        
        // Initialize empty transform and penalty (will be built on first basis construction)
        let link_transform = Array2::eye(n_constrained);
        let s_link_constrained = Array2::zeros((n_constrained, n_constrained));
        
        Self {
            y,
            weights,
            x_base,
            beta_base,
            beta_link,
            s_base,
            s_link_constrained,
            link_transform,
            rho,
            link,
            layout_base,
            n_link_knots: config.n_link_knots,
            degree,
            knot_range: None,
            knot_vector: None,
            n_constrained_basis: n_constrained,
        }
    }
    
    /// Set rho (log-lambdas) for REML optimization
    pub fn set_rho(&mut self, rho: Array1<f64>) {
        self.rho = rho;
    }
    
    /// Compute the current linear predictor Xβ
    pub fn base_linear_predictor(&self) -> Array1<f64> {
        self.x_base.dot(&self.beta_base)
    }
    
    /// Get number of observations
    pub fn n_obs(&self) -> usize {
        self.y.len()
    }
    
    /// Get total weight sum
    pub fn total_weight(&self) -> f64 {
        self.weights.sum()
    }
    
    /// Get link function
    pub fn link(&self) -> LinkFunction {
        self.link.clone()
    }
    
    /// Get number of base penalties
    pub fn n_base_penalties(&self) -> usize {
        self.s_base.len()
    }
    
    /// Get number of link penalties  
    pub fn n_link_penalties(&self) -> usize {
        1 // Single penalty for link wiggle
    }
    
    /// Get base model layout
    pub fn layout(&self) -> &ModelLayout {
        &self.layout_base
    }
    
    /// Build link spline basis at current Xβ values
    /// Returns ONLY the constrained wiggle basis (identity u is treated as offset)
    /// Also updates internal state with transform and projected penalty
    pub fn build_link_basis(&mut self, eta_base: &Array1<f64>) -> Array2<f64> {
        use crate::calibrate::basis::apply_weighted_orthogonality_constraint;
        
        let n = eta_base.len();
        let k = self.n_link_knots;
        let degree = self.degree;
        
        // Use fixed knot range if available, otherwise compute it (lock only when non-degenerate).
        let (min_u, max_u, lock_range) = match self.knot_range {
            Some(range) => (range.0, range.1, false),
            None => {
                let min_val = eta_base.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = eta_base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range_width = max_val - min_val;
                let lock = range_width > 1e-6;
                (min_val, max_val, lock)
            }
        };
        
        // Standardize: z = (u - min) / (max - min) to [0, 1]
        let range_width = (max_u - min_u).max(1e-6);
        let z: Array1<f64> = eta_base
            .mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
        
        // Build B-spline basis on z ∈ [0, 1]
        let data_range = (0.0, 1.0);
        match create_bspline_basis(z.view(), data_range, k, degree) {
            Ok((bspline_basis, knots)) => {
                // Store knot vector if not already stored
                if self.knot_vector.is_none() && lock_range {
                    self.knot_vector = Some(knots);
                    self.knot_range = Some((min_u, max_u));
                }
                
                let n_raw = bspline_basis.ncols();
                
                // Build constraint matrix: [ones, z] to remove intercept and linear from wiggle
                let mut constraint = Array2::<f64>::zeros((n, 2));
                for i in 0..n {
                    constraint[[i, 0]] = 1.0;     // intercept
                    constraint[[i, 1]] = z[i];    // linear term
                }
                
                // Apply orthogonality constraint: wiggle ⟂ {1, z}
                match apply_weighted_orthogonality_constraint(
                    bspline_basis.view(),
                    constraint.view(),
                    Some(self.weights),
                ) {
                    Ok((constrained_basis, transform)) => {
                        let n_constrained = constrained_basis.ncols();
                        
                        // Build raw difference penalty for original basis
                        let raw_penalty = match create_difference_penalty_matrix(n_raw, 2) {
                            Ok(p) => p,
                            Err(_) => Array2::zeros((n_raw, n_raw)),
                        };
                        
                        // Project penalty into constrained space: S_c = Z' S Z
                        let projected_penalty = transform.t().dot(&raw_penalty).dot(&transform);
                        
                        // Store transform and projected penalty
                        self.link_transform = transform;
                        self.s_link_constrained = projected_penalty;
                        self.n_constrained_basis = n_constrained;
                        
                        // Resize beta_link if needed
                        if self.beta_link.len() != n_constrained {
                            let mut new_beta = Array1::zeros(n_constrained);
                            let copy_len = self.beta_link.len().min(n_constrained);
                            for i in 0..copy_len {
                                new_beta[i] = self.beta_link[i];
                            }
                            self.beta_link = new_beta;
                        }
                        
                        constrained_basis
                    }
                    Err(_) => {
                        // Fallback: use raw basis without constraint
                        eprintln!("[JOINT] Orthogonality constraint failed");
                        let raw_penalty = match create_difference_penalty_matrix(n_raw, 2) {
                            Ok(p) => p,
                            Err(_) => Array2::zeros((n_raw, n_raw)),
                        };
                        self.link_transform = Array2::eye(n_raw);
                        self.s_link_constrained = raw_penalty;
                        self.n_constrained_basis = n_raw;
                        if self.beta_link.len() != n_raw {
                            let mut new_beta = Array1::zeros(n_raw);
                            let copy_len = self.beta_link.len().min(n_raw);
                            for i in 0..copy_len {
                                new_beta[i] = self.beta_link[i];
                            }
                            self.beta_link = new_beta;
                        }
                        (*bspline_basis).clone()
                    }
                }
            }
            Err(_) => {
                // Fallback: return empty basis
                eprintln!("[JOINT] B-spline basis construction failed");
                self.beta_link = Array1::zeros(0);
                self.link_transform = Array2::zeros((0, 0));
                self.s_link_constrained = Array2::zeros((0, 0));
                self.n_constrained_basis = 0;
                Array2::zeros((n, 0))
            }
        }
    }
    
    /// Build constrained link basis using stored knots without mutating state.
    pub fn build_link_basis_from_state(&self, eta_base: &Array1<f64>) -> Array2<f64> {
        let n = eta_base.len();
        let Some(knot_vector) = self.knot_vector.as_ref() else {
            return Array2::zeros((n, 0));
        };
        let (min_u, max_u) = self.knot_range.unwrap_or((0.0, 1.0));
        let range_width = (max_u - min_u).max(1e-6);
        let z: Array1<f64> = eta_base
            .mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
        
        let b_raw = match create_bspline_basis_with_knots(z.view(), knot_vector.view(), self.degree) {
            Ok((basis, _)) => (*basis).clone(),
            Err(_) => Array2::zeros((n, 0)),
        };
        
        if self.link_transform.ncols() > 0 && self.link_transform.nrows() == b_raw.ncols() {
            b_raw.dot(&self.link_transform)
        } else {
            b_raw
        }
    }
    
    /// Compute standardized z values and range width for current knot range.
    fn standardized_z(&self, eta_base: &Array1<f64>) -> (Array1<f64>, f64) {
        let (min_u, max_u) = self.knot_range.unwrap_or((0.0, 1.0));
        let range_width = (max_u - min_u).max(1e-6);
        let z: Array1<f64> = eta_base
            .mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
        (z, range_width)
    }
    
    /// Return the stored projected penalty for link block (Z'SZ)
    pub fn build_link_penalty(&self) -> Array2<f64> {
        self.s_link_constrained.clone()
    }
    
    /// Solve penalized weighted least squares: (X'WX + λS)β = X'Wz
    /// Returns new coefficients
    fn solve_weighted_ls(
        x: &Array2<f64>,
        z: &Array1<f64>,
        w: &Array1<f64>,
        penalty: &Array2<f64>,
        lambda: f64,
    ) -> Array1<f64> {
        use crate::calibrate::faer_ndarray::FaerCholesky;
        use crate::calibrate::faer_ndarray::{fast_ata, fast_atv};
        use faer::Side;
        
        let n = x.nrows();
        let p = x.ncols();
        
        if p == 0 {
            return Array1::zeros(0);
        }
        
        // Compute X'WX via weighted design
        let mut x_weighted = x.clone();
        let mut z_weighted = z.clone();
        for i in 0..n {
            let wi = w[i].max(0.0).sqrt();
            for j in 0..p {
                x_weighted[[i, j]] *= wi;
            }
            z_weighted[i] *= wi;
        }
        let mut xwx = fast_ata(&x_weighted);
        
        // Add penalty: X'WX + λS (only if penalty matches dimensions)
        if penalty.nrows() == p && penalty.ncols() == p {
            xwx = xwx + penalty * lambda;
        }
        
        // Add small ridge for numerical stability
        for i in 0..p {
            xwx[[i, i]] += 1e-8;
        }
        
        // Compute X'Wz via weighted design
        let xwz = fast_atv(&x_weighted, &z_weighted);
        
        // Solve (X'WX + λS)β = X'Wz using Cholesky
        match xwx.cholesky(Side::Lower) {
            Ok(chol) => chol.solve_vec(&xwz),
            Err(_) => {
                eprintln!("[JOINT] Warning: Cholesky factorization failed");
                Array1::zeros(p)
            }
        }
    }
    
    /// Perform one IRLS step for the link block
    /// Uses identity (u) as OFFSET, solves only for wiggle coefficients
    /// η = u + B_wiggle(z) · θ
    pub fn irls_link_step(
        &mut self,
        b_wiggle: &Array2<f64>,  // Constrained wiggle basis (NOT including identity)
        u: &Array1<f64>,          // Current linear predictor u = Xβ (used as offset)
        lambda_link: f64,
    ) -> f64 {
        let n = self.n_obs();
        
        // Current η = u + B_wiggle · θ
        let eta = if b_wiggle.ncols() > 0 && self.beta_link.len() > 0 {
            let wiggle = b_wiggle.dot(&self.beta_link);
            u + &wiggle
        } else {
            u.clone()
        };
        
        // Allocate working vectors
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z_glm = Array1::<f64>::zeros(n);
        
        // Compute working response and weights
        crate::calibrate::pirls::update_glm_vectors(
            self.y,
            &eta,
            self.link.clone(),
            self.weights,
            &mut mu,
            &mut weights,
            &mut z_glm,
        );
        
        // Adjust working response: solve for wiggle coefficient θ where
        // η = u + B_wiggle · θ
        // So target for θ is: z_adjusted = z_glm - u
        let z_adjusted: Array1<f64> = &z_glm - u;
        
        // Solve: (B'WB + λS)θ = B'W(z - u)
        if b_wiggle.ncols() > 0 {
            let penalty = self.build_link_penalty();
            let new_theta = Self::solve_weighted_ls(b_wiggle, &z_adjusted, &weights, &penalty, lambda_link);
            
            // Update wiggle coefficients
            if new_theta.len() == self.beta_link.len() {
                self.beta_link = new_theta;
            }
        }
        
        // Return deviance
        self.compute_deviance(&mu)
    }
    
    /// Perform one IRLS step for the base β block
    /// Uses Gauss-Newton with proper offset for nonlinear link:
    /// η = g(u) = u + wiggle(u), ∂η/∂β = g'(u) · x
    /// Working response for β: z_β = z_glm - η + g'(u)·u
    pub fn irls_base_step(
        &mut self,
        b_wiggle: &Array2<f64>,  // Constrained wiggle basis
        g_prime: &Array1<f64>,   // Derivative of link: g'(u) = 1 + B'(u)·θ
        lambda_base: &Array1<f64>,
        damping: f64,
    ) -> f64 {
        let n = self.n_obs();
        let p = self.x_base.ncols();
        
        // Current u = Xβ
        let u = self.base_linear_predictor();
        
        // Current η = u + B_wiggle · θ
        let wiggle: Array1<f64> = if b_wiggle.ncols() > 0 && self.beta_link.len() > 0 {
            b_wiggle.dot(&self.beta_link)
        } else {
            Array1::zeros(n)
        };
        let eta: Array1<f64> = &u + &wiggle;
        
        // Compute working response and weights for current η
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z_glm = Array1::<f64>::zeros(n);
        
        crate::calibrate::pirls::update_glm_vectors(
            self.y,
            &eta,
            self.link.clone(),
            self.weights,
            &mut mu,
            &mut weights,
            &mut z_glm,
        );
        
        // Correct working response for β update (Gauss-Newton offset):
        // z_β = z_glm - η + g'(u)·u
        let mut z_beta = Array1::<f64>::zeros(n);
        for i in 0..n {
            z_beta[i] = z_glm[i] - eta[i] + g_prime[i] * u[i];
        }
        
        // Weighted least squares with scaled weights to avoid explicit X_eff
        let mut w_eff = Array1::<f64>::zeros(n);
        let mut z_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            let g = g_prime[i].max(1e-8);
            w_eff[i] = weights[i] * g * g;
            z_eff[i] = z_beta[i] / g;
        }
        
        // Build penalty for base block: S_base = Σ λ_k S_k
        let mut penalty = Array2::<f64>::zeros((p, p));
        for (idx, s_k) in self.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if s_k.nrows() == p && s_k.ncols() == p && lambda_k > 0.0 {
                penalty.scaled_add(lambda_k, s_k);
            }
        }
        
        // Solve PWLS: (X'W_eff X + S)β = X'W_eff z_eff
        let new_beta = Self::solve_weighted_ls(&self.x_base.to_owned(), &z_eff, &w_eff, &penalty, 1.0);
        
        // Apply damped update
        for j in 0..p {
            if j < new_beta.len() {
                let delta = new_beta[j] - self.beta_base[j];
                self.beta_base[j] += damping * delta;
            }
        }
        
        // Return deviance
        self.compute_deviance(&mu)
    }
    
    /// Compute current linear predictor: η = u + B_wiggle · θ
    pub fn compute_eta_full(&self, u: &Array1<f64>, b_wiggle: &Array2<f64>) -> Array1<f64> {
        if b_wiggle.ncols() > 0 && self.beta_link.len() > 0 {
            let wiggle = b_wiggle.dot(&self.beta_link);
            u + &wiggle
        } else {
            u.clone()
        }
    }
    
    /// Compute deviance for logit link
    fn compute_deviance(&self, mu: &Array1<f64>) -> f64 {
        let mut dev = 0.0;
        for i in 0..self.n_obs() {
            let y_i = self.y[i];
            let mu_i = mu[i].clamp(1e-10, 1.0 - 1e-10);
            let w_i = self.weights[i];
            
            // Binomial deviance: 2 * [y*log(y/mu) + (1-y)*log((1-y)/(1-mu))]
            let d = if y_i > 0.5 {
                2.0 * w_i * y_i.ln() - 2.0 * w_i * mu_i.ln()
            } else {
                2.0 * w_i * (1.0 - y_i).ln() - 2.0 * w_i * (1.0 - mu_i).ln()
            };
            dev += d;
        }
        dev
    }
}

/// Fit joint single-index model via Gauss-Newton PIRLS with outer REML optimization
/// 
/// Architecture:
/// - Outer loop: BFGS over smoothing params ρ (same as existing GAM fitting)
/// - Inner loop: Gauss-Newton PIRLS for coefficients (β, θ)
/// 
/// The model is nonlinear because g depends on u, and u depends on β:
///   η_i = g(u_i), where u_i = x_i'β + f(covariates)
///   g(u) = B(u)θ for spline basis B evaluated at u
/// 
/// The inner solve uses a Jacobian J instead of fixed design X:
///   J_i = [g'(u_i) * x_i | B(u_i)]
/// 
/// Identifiability: scale anchor enforces g(0)≈0, g'(0)≈1
pub fn fit_joint_model<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    layout_base: ModelLayout,
    link: LinkFunction,
    config: &JointModelConfig,
) -> Result<JointModelResult, EstimationError> {
    let mut state = JointModelState::new(
        y, weights, x_base, s_base, layout_base, link, config
    );
    
    let mut prev_deviance = f64::INFINITY;
    let mut converged = false;
    let mut iter = 0;
    let mut total_design_cols = 0;
    
    // Get lambdas from rho (log-lambdas)
    let n_base = state.s_base.len();
    let mut lambda_base = Array1::<f64>::zeros(n_base);
    for i in 0..n_base {
        lambda_base[i] = state.rho.get(i).map(|r| r.exp()).unwrap_or(1.0);
    }
    let lambda_link = state.rho.get(n_base).map(|r| r.exp()).unwrap_or(1.0);
    
    // Damping schedule: start conservative, increase as we converge
    let initial_damping = 0.5;
    let final_damping = 1.0;
    
    for i in 0..config.max_backfit_iter {
        iter = i + 1;
        
        // Adaptive damping: increase towards 1.0 as we iterate
        let progress = (i as f64) / (config.max_backfit_iter as f64);
        let damping = initial_damping + progress * (final_damping - initial_damping);
        
        // Step A: Given β, build wiggle basis and update link coefficients
        let u = state.base_linear_predictor();
        let b_wiggle = state.build_link_basis(&u);  // Updates internal state (transform, penalty)
        total_design_cols = b_wiggle.ncols();
        
        // Update link coefficients (θ) via IRLS with u as OFFSET
        let deviance_after_g = state.irls_link_step(&b_wiggle, &u, lambda_link);
        
        // Step B: Given g, update β using g'(u)*X design
        // Get knot range for derivative computation
        let knot_range = state.knot_range.unwrap_or_else(|| {
            let min_val = u.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = u.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min_val, max_val)
        });
        
        // Compute g'(u) for chain rule
        let g_prime = compute_link_derivative_from_state(&state, &u, &b_wiggle);
        
        // Update β with damping (Gauss-Newton with offset)
        let deviance = state.irls_base_step(&b_wiggle, &g_prime, &lambda_base, damping);
        
        // Check for convergence
        let delta = (prev_deviance - deviance).abs() / (deviance.abs() + 1.0);
        
        eprintln!("[JOINT] Iter {}: dev_g={:.4}, dev_β={:.4}, δ={:.6}, damp={:.2}, g'∈[{:.2},{:.2}]", 
                  iter, deviance_after_g, deviance, delta, damping,
                  g_prime.iter().cloned().fold(f64::INFINITY, f64::min),
                  g_prime.iter().cloned().fold(f64::NEG_INFINITY, f64::max));
        
        if delta < config.backfit_tol {
            converged = true;
            eprintln!("[JOINT] Converged after {} iterations", iter);
            break;
        }
        prev_deviance = deviance;
    }
    
    if !converged {
        eprintln!("[JOINT] Did not converge after {} iterations", config.max_backfit_iter);
    }
    
    // Get stored values for result
    let knot_range = state.knot_range.unwrap_or((0.0, 1.0));
    let knot_vector = state.knot_vector.clone().unwrap_or_else(|| Array1::zeros(0));
    
    // Estimate EDF from design dimensions (placeholder - proper EDF requires trace(hat matrix))
    let edf = total_design_cols as f64 + state.x_base.ncols() as f64;
    
    Ok(JointModelResult {
        beta_base: state.beta_base.clone(),
        beta_link: state.beta_link.clone(),
        lambdas: state.rho.mapv(f64::exp).to_vec(),
        deviance: prev_deviance,
        edf,
        backfit_iterations: iter,
        converged,
        knot_range,
        knot_vector,
        link_transform: state.link_transform.clone(),
        degree: state.degree,
    })
}

/// State for REML optimization of the joint model
/// Wraps JointModelState and provides cost_and_grad for BFGS
pub struct JointRemlState<'a> {
    /// The underlying model state (uses RefCell for interior mutability during optimization)
    state: RefCell<JointModelState<'a>>,
    /// Configuration
    config: JointModelConfig,
    /// Cached warm-start coefficients
    cached_beta_base: RefCell<Array1<f64>>,
    cached_beta_link: RefCell<Array1<f64>>,
    /// Cached LAML value for gradient computation
    cached_laml: RefCell<Option<f64>>,
    cached_rho: RefCell<Array1<f64>>,
    base_reparam_invariant: Option<ReparamInvariant>,
    base_rs_list: Vec<Array2<f64>>,
}

struct JointRemlSnapshot {
    beta_base: Array1<f64>,
    beta_link: Array1<f64>,
    rho: Array1<f64>,
    knot_range: Option<(f64, f64)>,
    knot_vector: Option<Array1<f64>>,
    link_transform: Array2<f64>,
    s_link_constrained: Array2<f64>,
    n_constrained_basis: usize,
    cached_beta_base: Array1<f64>,
    cached_beta_link: Array1<f64>,
    cached_rho: Array1<f64>,
    cached_laml: Option<f64>,
}

impl JointRemlSnapshot {
    fn new(reml: &JointRemlState<'_>) -> Self {
        let state = reml.state.borrow();
        Self {
            beta_base: state.beta_base.clone(),
            beta_link: state.beta_link.clone(),
            rho: state.rho.clone(),
            knot_range: state.knot_range,
            knot_vector: state.knot_vector.clone(),
            link_transform: state.link_transform.clone(),
            s_link_constrained: state.s_link_constrained.clone(),
            n_constrained_basis: state.n_constrained_basis,
            cached_beta_base: reml.cached_beta_base.borrow().clone(),
            cached_beta_link: reml.cached_beta_link.borrow().clone(),
            cached_rho: reml.cached_rho.borrow().clone(),
            cached_laml: *reml.cached_laml.borrow(),
        }
    }
    
    fn restore(&self, reml: &JointRemlState<'_>) {
        let mut state = reml.state.borrow_mut();
        state.beta_base = self.beta_base.clone();
        state.beta_link = self.beta_link.clone();
        state.rho = self.rho.clone();
        state.knot_range = self.knot_range;
        state.knot_vector = self.knot_vector.clone();
        state.link_transform = self.link_transform.clone();
        state.s_link_constrained = self.s_link_constrained.clone();
        state.n_constrained_basis = self.n_constrained_basis;
        *reml.cached_beta_base.borrow_mut() = self.cached_beta_base.clone();
        *reml.cached_beta_link.borrow_mut() = self.cached_beta_link.clone();
        *reml.cached_rho.borrow_mut() = self.cached_rho.clone();
        *reml.cached_laml.borrow_mut() = self.cached_laml;
    }
}

impl<'a> JointRemlState<'a> {
    /// Create new REML state
    pub fn new(
        y: ArrayView1<'a, f64>,
        weights: ArrayView1<'a, f64>,
        x_base: ArrayView2<'a, f64>,
        s_base: Vec<Array2<f64>>,
        layout_base: ModelLayout,
        link: LinkFunction,
        config: &JointModelConfig,
    ) -> Self {
        let state = JointModelState::new(y, weights, x_base, s_base, layout_base, link, config);
        let cached_beta_base = state.beta_base.clone();
        let cached_beta_link = state.beta_link.clone();
        let base_rs_list = compute_penalty_square_roots(&state.s_base)
            .unwrap_or_else(|_| Vec::new());
        let base_reparam_invariant =
            precompute_reparam_invariant(&base_rs_list, &state.layout_base).ok();
        Self {
            state: RefCell::new(state),
            config: config.clone(),
            cached_beta_base: RefCell::new(cached_beta_base),
            cached_beta_link: RefCell::new(cached_beta_link),
            cached_laml: RefCell::new(None),
            cached_rho: RefCell::new(Array1::zeros(state.s_base.len() + 1)),
            base_reparam_invariant,
            base_rs_list,
        }
    }
    
    /// Compute LAML cost for a given ρ
    /// LAML = deviance + log|H_pen| - log|S_λ| (+ prior on ρ)
    pub fn compute_cost(&self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let mut state = self.state.borrow_mut();
        let n_base = state.s_base.len();
        
        // Set ρ and warm-start from cached coefficients
        state.set_rho(rho.clone());
        state.beta_base = self.cached_beta_base.borrow().clone();
        state.beta_link = self.cached_beta_link.borrow().clone();
        
        // Run inner alternating to convergence
        let mut lambda_base = Array1::<f64>::zeros(n_base);
        for i in 0..n_base {
            lambda_base[i] = rho.get(i).map(|r| r.exp()).unwrap_or(1.0);
        }
        let lambda_link = rho.get(n_base).map(|r| r.exp()).unwrap_or(1.0);
        
        let mut prev_deviance = f64::INFINITY;
        for i in 0..self.config.max_backfit_iter {
            let progress = (i as f64) / (self.config.max_backfit_iter as f64);
            let damping = 0.5 + progress * 0.5;
            
            let u = state.base_linear_predictor();
            let b_wiggle = state.build_link_basis(&u);
            state.irls_link_step(&b_wiggle, &u, lambda_link);
            
            let g_prime = compute_link_derivative_from_state(&state, &u, &b_wiggle);
            let deviance = state.irls_base_step(&b_wiggle, &g_prime, &lambda_base, damping);
            
            let delta = (prev_deviance - deviance).abs() / (deviance.abs() + 1.0);
            if delta < self.config.backfit_tol {
                break;
            }
            prev_deviance = deviance;
        }
        
        // Cache converged coefficients for warm-start
        *self.cached_beta_base.borrow_mut() = state.beta_base.clone();
        *self.cached_beta_link.borrow_mut() = state.beta_link.clone();
        
        // Compute LAML = deviance + log|H_pen| - log|S_λ|
        let laml = self.compute_laml_at_convergence(&state, &lambda_base, lambda_link);
        
        // Cache for gradient
        *self.cached_laml.borrow_mut() = Some(laml);
        *self.cached_rho.borrow_mut() = rho.clone();
        
        Ok(laml)
    }
    
    /// Compute LAML at the converged solution
    fn compute_laml_at_convergence(&self, state: &JointModelState, lambda_base: &Array1<f64>, lambda_link: f64) -> f64 {
        let n = state.n_obs();
        let u = state.base_linear_predictor();
        let b_wiggle = state.build_link_basis_from_state(&u);
        
        // Compute eta = u + B_wiggle * theta
        let eta = state.compute_eta_full(&u, &b_wiggle);
        
        // Compute deviance
        let mut mu = Array1::<f64>::zeros(n);
        for i in 0..n {
            mu[i] = 1.0 / (1.0 + (-eta[i]).exp());
        }
        let deviance = state.compute_deviance(&mu);
        
        // Compute weights at convergence
        let mut weights = Array1::<f64>::zeros(n);
        for i in 0..n {
            let p = mu[i].max(1e-10).min(1.0 - 1e-10);
            weights[i] = state.weights[i] * p * (1.0 - p);
        }
        
        // Build joint Jacobian blocks and penalized Hessian via Schur complement
        let p_base = state.x_base.ncols();
        let p_link = b_wiggle.ncols();
        
        let g_prime = compute_link_derivative_from_state(state, &u, &b_wiggle);
        
        // A = X' diag(W * g'^2) X + S_base
        let mut w_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            w_eff[i] = weights[i] * g_prime[i] * g_prime[i];
        }
        let mut x_weighted = state.x_base.to_owned();
        for i in 0..n {
            let scale = w_eff[i].sqrt();
            for j in 0..p_base {
                x_weighted[[i, j]] *= scale;
            }
        }
        
        let mut a_mat = crate::calibrate::faer_ndarray::fast_ata(&x_weighted);
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                a_mat.scaled_add(lambda_k, s_k);
            }
        }
        for i in 0..p_base {
            a_mat[[i, i]] += 1e-8;
        }
        
        // C = X' diag(W * g') B
        let mut wb = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i] * g_prime[i];
            for j in 0..p_link {
                wb[[i, j]] *= scale;
            }
        }
        let c_mat = crate::calibrate::faer_ndarray::fast_atb(&state.x_base, &wb);
        
        // D = B' W B + S_link
        let mut b_weighted = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i].sqrt();
            for j in 0..p_link {
                b_weighted[[i, j]] *= scale;
            }
        }
        let mut d_mat = crate::calibrate::faer_ndarray::fast_ata(&b_weighted);
        let link_penalty = state.build_link_penalty();
        if link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            d_mat.scaled_add(lambda_link, &link_penalty);
        }
        for i in 0..p_link {
            d_mat[[i, i]] += 1e-8;
        }
        
        // log|H| via Schur complement: log|A| + log|D - C^T A^{-1} C|
        use crate::calibrate::faer_ndarray::FaerCholesky;
        use faer::Side;
        let log_det_a = match a_mat.cholesky(Side::Lower) {
            Ok(chol) => {
                let log_det = 2.0 * chol.diag().mapv(|d| d.max(1e-10).ln()).sum();
                let mut a_inv_c = c_mat.clone();
                for col in 0..a_inv_c.ncols() {
                    let solved = chol.solve_vec(&a_inv_c.column(col).to_owned());
                    for row in 0..a_inv_c.nrows() {
                        a_inv_c[[row, col]] = solved[row];
                    }
                }
                let schur = &d_mat - &a_inv_c.t().dot(&c_mat);
                let log_det_schur = match schur.cholesky(Side::Lower) {
                    Ok(schol) => 2.0 * schol.diag().mapv(|d| d.max(1e-10).ln()).sum(),
                    Err(_) => 0.0,
                };
                log_det + log_det_schur
            }
            Err(_) => {
                eprintln!("[LAML] Cholesky failed, using trace approximation");
                let trace_a: f64 = (0..p_base).map(|i| a_mat[[i, i]].max(1e-10).ln()).sum();
                let trace_d: f64 = (0..p_link).map(|i| d_mat[[i, i]].max(1e-10).ln()).sum();
                trace_a + trace_d
            }
        };
        
        // log|Sλ|_+ from stable reparameterization (base) and eigenvalues (link)
        let base_reparam = if let Some(invariant) = self.base_reparam_invariant.as_ref() {
            stable_reparameterization_with_invariant(
                &self.base_rs_list,
                &lambda_base.to_vec(),
                &state.layout_base,
                invariant,
            )
        } else {
            stable_reparameterization(&self.base_rs_list, &lambda_base.to_vec(), &state.layout_base)
        }
        .unwrap_or_else(|_| ReparamResult {
            s_transformed: Array2::zeros((p_base, p_base)),
            log_det: 0.0,
            det1: Array1::zeros(lambda_base.len()),
            qs: Array2::eye(p_base),
            rs_transformed: vec![],
            rs_transposed: vec![],
            e_transformed: Array2::zeros((0, p_base)),
        });
        let base_log_det_s = base_reparam.log_det;
        let base_rank = base_reparam.e_transformed.nrows() as f64;
        
        let (link_log_det_s, link_rank) = if p_link > 0 {
            use crate::calibrate::faer_ndarray::FaerEigh;
            let (eigs, _) = link_penalty
                .clone()
                .eigh(Side::Lower)
                .unwrap_or_else(|_| (Array1::zeros(p_link), Array2::eye(p_link)));
            let max_eig = eigs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let tol = if max_eig > 0.0 { max_eig * 1e-12 } else { 1e-12 };
            let positive: Vec<f64> = eigs.iter().cloned().filter(|&ev| ev > tol).collect();
            let rank = positive.len() as f64;
            let log_det = positive.iter().map(|&ev| ev.ln()).sum::<f64>()
                + rank * lambda_link.max(1e-10).ln();
            (log_det, rank)
        } else {
            (0.0, 0.0)
        };
        
        let log_det_s = base_log_det_s + link_log_det_s;
        
        // Null space dimension
        let mp = (p_base as f64 - base_rank) + (p_link as f64 - link_rank);
        
        // LAML = -2*log_lik + log|H_pen| - log|Sλ| + mp*log(2π)
        // Or equivalently: deviance + log|H_pen| - log|Sλ| + mp*log(2π)
        let laml = deviance + 0.5 * log_det_a - 0.5 * log_det_s 
                 + (mp / 2.0) * (2.0 * std::f64::consts::PI).ln();
        
        laml
    }
    
    /// Compute numerical gradient of LAML w.r.t. ρ using central differences
    pub fn compute_gradient(&self, rho: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        let h = 1e-4; // Step size for numerical differentiation
        let n_rho = rho.len();
        let mut grad = Array1::<f64>::zeros(n_rho);
        let snapshot = JointRemlSnapshot::new(self);
        
        for k in 0..n_rho {
            snapshot.restore(self);
            // Forward step
            let mut rho_plus = rho.clone();
            rho_plus[k] += h;
            let cost_plus = self.compute_cost(&rho_plus)?;
            
            snapshot.restore(self);
            // Backward step  
            let mut rho_minus = rho.clone();
            rho_minus[k] -= h;
            let cost_minus = self.compute_cost(&rho_minus)?;
            
            // Central difference
            grad[k] = (cost_plus - cost_minus) / (2.0 * h);
        }
        
        snapshot.restore(self);
        
        Ok(grad)
    }
    
    /// Combined cost and gradient for BFGS
    pub fn cost_and_grad(&self, rho: &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError> {
        let cost = self.compute_cost(rho)?;
        let grad = self.compute_gradient(rho)?;
        Ok((cost, grad))
    }
    
    /// Extract final result after optimization
    pub fn into_result(self) -> JointModelResult {
        let state = self.state.into_inner();
        let rho = state.rho.clone();
        let knot_range = state.knot_range.unwrap_or((0.0, 1.0));
        let knot_vector = state.knot_vector.clone().unwrap_or_else(|| Array1::zeros(0));
        JointModelResult {
            beta_base: state.beta_base,
            beta_link: state.beta_link,
            lambdas: rho.mapv(f64::exp).to_vec(),
            deviance: self.cached_laml.borrow().unwrap_or(f64::INFINITY),
            edf: state.n_constrained_basis as f64 + state.x_base.ncols() as f64,
            backfit_iterations: self.config.max_backfit_iter,
            converged: true,
            knot_range,
            knot_vector,
            link_transform: state.link_transform,
            degree: state.degree,
        }
    }
}

/// Fit joint model with proper REML-based lambda selection via BFGS
/// 
/// Uses Laplace approximate marginal likelihood (LAML) with numerical gradient.
pub fn fit_joint_model_with_reml<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    layout_base: ModelLayout,
    link: LinkFunction,
    config: &JointModelConfig,
) -> Result<JointModelResult, EstimationError> {
    
    // Create REML state
    let reml_state = JointRemlState::new(
        y, weights, x_base, s_base, layout_base, link, config
    );
    
    // Initial rho = zeros (λ = 1)
    let n_base = reml_state.state.borrow().s_base.len();
    let initial_rho = Array1::from_elem(n_base + 1, 0.0);
    
    // Run BFGS optimization
    let mut best_rho = initial_rho.clone();
    let mut best_cost = reml_state.compute_cost(&initial_rho)?;
    
    if initial_rho.len() >= 2 {
        eprintln!("[REML] Initial: ρ[0]={:.2}, ρ[1]={:.2}, LAML={:.4}", 
                 initial_rho[0], initial_rho[1], best_cost);
    } else {
        eprintln!("[REML] Initial: LAML={:.4}", best_cost);
    }
    
    // Simple gradient descent with line search (BFGS lite)
    for outer_iter in 0..config.max_reml_iter {
        let grad = reml_state.compute_gradient(&best_rho)?;
        let grad_norm = grad.mapv(|g| g * g).sum().sqrt();
        
        if grad_norm < config.reml_tol {
            eprintln!("[REML] Converged at iter {}: grad_norm={:.2e}", outer_iter, grad_norm);
            break;
        }
        
        // Line search along -gradient
        let mut step_size = 1.0;
        let direction = grad.mapv(|g| -g);
        
        for _ in 0..10 {
            let candidate_rho = &best_rho + &(&direction * step_size);
            // Bound ρ to [-8, 8] (λ ∈ [e^-8, e^8])
            let bounded_rho = candidate_rho.mapv(|r| r.max(-8.0).min(8.0));
            
            match reml_state.compute_cost(&bounded_rho) {
                Ok(cost) if cost < best_cost => {
                    best_cost = cost;
                    best_rho = bounded_rho;
                    if best_rho.len() >= 2 {
                        eprintln!("[REML] Iter {}: ρ[0]={:.2}, ρ[1]={:.2}, LAML={:.4}, step={:.2e}", 
                                 outer_iter, best_rho[0], best_rho[1], best_cost, step_size);
                    } else {
                        eprintln!("[REML] Iter {}: LAML={:.4}, step={:.2e}", 
                                 outer_iter, best_cost, step_size);
                    }
                    break;
                }
                _ => {
                    step_size *= 0.5;
                }
            }
        }
    }
    
    // Final fit at optimal rho
    let _ = reml_state.compute_cost(&best_rho)?;
    Ok(reml_state.into_result())
}


/// Prediction result from joint model
pub struct JointModelPrediction {
    /// Calibrated linear predictor η_cal
    pub eta: Array1<f64>,
    /// Probabilities (posterior predictive mean if SE available)
    pub probabilities: Array1<f64>,
    /// Effective SE after derivative propagation (|g'(η)| × SE_base)
    pub effective_se: Option<Array1<f64>>,
}

/// Compute derivative of link function g'(u) = 1 + B'(z)·θ · dz/du
/// using finite differences on the spline basis (O(n * p_link)).
fn compute_link_derivative_from_state(
    state: &JointModelState,
    u: &Array1<f64>,
    b_wiggle: &Array2<f64>,
) -> Array1<f64> {
    let n = u.len();
    let mut deriv = Array1::<f64>::ones(n);
    
    if b_wiggle.ncols() == 0 || state.beta_link.is_empty() {
        return deriv;
    }
    let Some(knot_vector) = state.knot_vector.as_ref() else {
        return deriv;
    };
    
    let (z, range_width) = state.standardized_z(u);
    let h = 1e-4;
    let z_plus: Array1<f64> = z.mapv(|v| (v + h).clamp(0.0, 1.0));
    let z_minus: Array1<f64> = z.mapv(|v| (v - h).clamp(0.0, 1.0));
    
    let b_plus_raw = match create_bspline_basis_with_knots(z_plus.view(), knot_vector.view(), state.degree) {
        Ok((basis, _)) => (*basis).clone(),
        Err(_) => return deriv,
    };
    let b_minus_raw = match create_bspline_basis_with_knots(z_minus.view(), knot_vector.view(), state.degree) {
        Ok((basis, _)) => (*basis).clone(),
        Err(_) => return deriv,
    };
    
    let b_plus = if state.link_transform.ncols() > 0 && state.link_transform.nrows() == b_plus_raw.ncols() {
        b_plus_raw.dot(&state.link_transform)
    } else {
        b_plus_raw
    };
    let b_minus = if state.link_transform.ncols() > 0 && state.link_transform.nrows() == b_minus_raw.ncols() {
        b_minus_raw.dot(&state.link_transform)
    } else {
        b_minus_raw
    };
    
    for i in 0..n {
        let dz = (z_plus[i] - z_minus[i]).max(1e-8);
        if b_plus.ncols() != state.beta_link.len() || b_minus.ncols() != state.beta_link.len() {
            deriv[i] = 1.0;
            continue;
        }
        let mut d_wiggle_dz = 0.0;
        for j in 0..state.beta_link.len() {
            d_wiggle_dz += (b_plus[[i, j]] - b_minus[[i, j]]) * state.beta_link[j];
        }
        d_wiggle_dz /= dz;
        deriv[i] = 1.0 + d_wiggle_dz / range_width;
    }
    
    deriv.mapv_inplace(|d| d.max(0.1).min(10.0));
    deriv
}

fn compute_link_derivative_from_result(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    b_wiggle: &Array2<f64>,
) -> Array1<f64> {
    let n = eta_base.len();
    let mut deriv = Array1::<f64>::ones(n);
    if b_wiggle.ncols() == 0 || result.beta_link.is_empty() {
        return deriv;
    }
    
    let (min_u, max_u) = result.knot_range;
    let range_width = (max_u - min_u).max(1e-6);
    let z: Array1<f64> = eta_base
        .mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
    let h = 1e-4;
    let z_plus: Array1<f64> = z.mapv(|v| (v + h).clamp(0.0, 1.0));
    let z_minus: Array1<f64> = z.mapv(|v| (v - h).clamp(0.0, 1.0));
    
    let b_plus_raw = match create_bspline_basis_with_knots(z_plus.view(), result.knot_vector.view(), result.degree) {
        Ok((basis, _)) => (*basis).clone(),
        Err(_) => return deriv,
    };
    let b_minus_raw = match create_bspline_basis_with_knots(z_minus.view(), result.knot_vector.view(), result.degree) {
        Ok((basis, _)) => (*basis).clone(),
        Err(_) => return deriv,
    };
    
    let b_plus = if result.link_transform.ncols() > 0 && result.link_transform.nrows() == b_plus_raw.ncols() {
        b_plus_raw.dot(&result.link_transform)
    } else {
        b_plus_raw
    };
    let b_minus = if result.link_transform.ncols() > 0 && result.link_transform.nrows() == b_minus_raw.ncols() {
        b_minus_raw.dot(&result.link_transform)
    } else {
        b_minus_raw
    };
    
    for i in 0..n {
        let dz = (z_plus[i] - z_minus[i]).max(1e-8);
        if b_plus.ncols() != result.beta_link.len() || b_minus.ncols() != result.beta_link.len() {
            deriv[i] = 1.0;
            continue;
        }
        let mut d_wiggle_dz = 0.0;
        for j in 0..result.beta_link.len() {
            d_wiggle_dz += (b_plus[[i, j]] - b_minus[[i, j]]) * result.beta_link[j];
        }
        d_wiggle_dz /= dz;
        deriv[i] = 1.0 + d_wiggle_dz / range_width;
    }
    
    deriv.mapv_inplace(|d| d.max(0.1).min(10.0));
    deriv
}


/// Predict probabilities from a fitted joint model
/// 
/// Uses stored knot_range and B-spline basis for consistent prediction.
/// For SE propagation, uses derivative-propagated uncertainty.
pub fn predict_joint(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    se_base: Option<&Array1<f64>>,
) -> JointModelPrediction {
    let n = eta_base.len();
    
    // Use stored knot range from training for consistent standardization
    let (min_u, max_u) = result.knot_range;
    let range_width = (max_u - min_u).max(1e-6);
    
    // Standardize: z = (u - min) / range
    let z: Array1<f64> = eta_base.mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
    
    // Build B-spline basis at prediction points using stored parameters
    let b_wiggle = match create_bspline_basis_with_knots(z.view(), result.knot_vector.view(), result.degree) {
        Ok((basis, _)) => {
            if result.link_transform.ncols() > 0 && result.link_transform.nrows() == basis.ncols() {
                basis.dot(&result.link_transform)
            } else {
                (*basis).clone()
            }
        }
        Err(_) => Array2::zeros((n, result.beta_link.len())),
    };
    
    // Compute η_cal = u + B_wiggle · θ
    let eta_cal: Array1<f64> = if b_wiggle.ncols() > 0 && result.beta_link.len() > 0 {
        let wiggle = b_wiggle.dot(&result.beta_link);
        eta_base + &wiggle
    } else {
        eta_base.clone()
    };
    
    // Compute effective SE if base SE provided
    let (probabilities, effective_se) = if let Some(se) = se_base {
        // Compute link derivative for uncertainty propagation
        let deriv = compute_link_derivative_from_result(result, eta_base, &b_wiggle);
        
        // Effective SE = |g'(η)| × SE_base  
        let eff_se: Array1<f64> = deriv.mapv(f64::abs) * se;
        
        // Compute posterior predictive mean via GHQ
        let probs: Array1<f64> = (0..n).map(|i| {
            crate::calibrate::quadrature::logit_posterior_mean(eta_cal[i], eff_se[i])
        }).collect();
        
        (probs, Some(eff_se))
    } else {
        // No SE: use plug-in sigmoid
        let probs: Array1<f64> = eta_cal.mapv(|e| 1.0 / (1.0 + (-e).exp()));
        (probs, None)
    };
    
    JointModelPrediction {
        eta: eta_cal,
        probabilities,
        effective_se,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_joint_model_state_creation() {
        let n = 100;
        let p = 10;
        let y = Array1::zeros(n);
        let weights = Array1::ones(n);
        let x = Array2::zeros((n, p));
        let s = vec![Array2::eye(p)];
        let layout = ModelLayout::external(p, 1);
        let config = JointModelConfig::default();
        
        let state = JointModelState::new(
            y.view(),
            weights.view(), 
            x.view(),
            s,
            layout,
            LinkFunction::Logit,
            &config,
        );
        
        assert_eq!(state.beta_base.len(), p);
        assert_eq!(state.beta_link.len(), config.n_link_knots + 2);
    }
    
    #[test]
    fn test_predict_joint_basic() {
        // Create a simple result with identity link (no wiggle)
        let n_knots = 5;
        let degree = 3;
        let (_basis, knot_vector) = create_bspline_basis(
            Array1::from_vec(vec![0.0]).view(),
            (0.0, 1.0),
            n_knots,
            degree,
        )
        .expect("basis");
        let num_basis = knot_vector.len().saturating_sub(degree + 1);
        let beta_link = Array1::zeros(num_basis);
        
        let result = JointModelResult {
            beta_base: Array1::zeros(10),
            beta_link,
            lambdas: vec![1.0],
            deviance: 0.0,
            edf: 5.0,
            backfit_iterations: 1,
            converged: true,
            knot_range: (0.0, 1.0),
            knot_vector,
            link_transform: Array2::eye(num_basis),
            degree,
        };
        
        // Test with base eta values
        let eta_base = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        
        // Predict without SE (should give sigmoid of eta)
        let pred = predict_joint(&result, &eta_base, None);
        
        assert_eq!(pred.eta.len(), 5);
        assert_eq!(pred.probabilities.len(), 5);
        assert!(pred.effective_se.is_none());
        
        // Check probabilities are in [0, 1]
        for p in pred.probabilities.iter() {
            assert!(*p >= 0.0 && *p <= 1.0);
        }
        
        // With identity link, prob at eta=0 should be ~0.5
        assert!((pred.probabilities[2] - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_predict_joint_with_se() {
        let n_knots = 5;
        let degree = 3;
        let (_basis, knot_vector) = create_bspline_basis(
            Array1::from_vec(vec![0.0]).view(),
            (0.0, 1.0),
            n_knots,
            degree,
        )
        .expect("basis");
        let num_basis = knot_vector.len().saturating_sub(degree + 1);
        let beta_link = Array1::zeros(num_basis);
        
        let result = JointModelResult {
            beta_base: Array1::zeros(10),
            beta_link,
            lambdas: vec![1.0],
            deviance: 0.0,
            edf: 5.0,
            backfit_iterations: 1,
            converged: true,
            knot_range: (0.0, 1.0),
            knot_vector,
            link_transform: Array2::eye(num_basis),
            degree,
        };
        
        let eta_base = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let se_base = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        
        let pred = predict_joint(&result, &eta_base, Some(&se_base));
        
        assert!(pred.effective_se.is_some());
        let eff_se = pred.effective_se.unwrap();
        assert_eq!(eff_se.len(), 3);
        
        // With identity link, effective SE should equal base SE (derivative = 1)
        for i in 0..3 {
            assert!((eff_se[i] - se_base[i]).abs() < 0.1);
        }
    }
}
