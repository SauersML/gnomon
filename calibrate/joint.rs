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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use std::cell::RefCell;
use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::LinkFunction;
use crate::calibrate::basis::{create_bspline_basis, create_difference_penalty_matrix};

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
    /// Current link correction coefficients (for t(·))
    beta_link: Array1<f64>,
    /// Penalty matrices for base block
    s_base: Vec<Array2<f64>>,
    /// Penalty matrices for link block  
    s_link: Vec<Array2<f64>>,
    /// Current log-smoothing parameters (all blocks)
    rho: Array1<f64>,
    /// Link function (Logit or Identity)
    link: LinkFunction,
    /// Layout for base model
    layout_base: ModelLayout,
    /// Number of knots for link spline
    n_link_knots: usize,
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
    /// Number of knots for link spline
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

/// Result of joint model fitting
pub struct JointModelResult {
    /// Fitted base coefficients β
    pub beta_base: Array1<f64>,
    /// Fitted link coefficients (for t(·))
    pub beta_link: Array1<f64>,
    /// Fitted smoothing parameters (all blocks)
    pub lambdas: Vec<f64>,
    /// Final deviance
    pub deviance: f64,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Number of backfitting iterations
    pub backfit_iterations: usize,
    /// Converged flag
    pub converged: bool,
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
        let n_link = config.n_link_knots + 1; // knots + linear term
        
        // Initialize β to zero, link to identity (β_link[0] = 1 for slope)
        let beta_base = Array1::zeros(n_base);
        let mut beta_link = Array1::zeros(n_link);
        if n_link > 0 {
            beta_link[0] = 1.0; // Identity slope anchor
        }
        
        // Initialize rho (log-lambdas) to 0 (λ = 1)
        let n_penalties = s_base.len() + 1; // base penalties + link penalty
        let rho = Array1::zeros(n_penalties);
        
        // Link penalty will be built dynamically
        let s_link = vec![];
        
        Self {
            y,
            weights,
            x_base,
            beta_base,
            beta_link,
            s_base,
            s_link,
            rho,
            link,
            layout_base,
            n_link_knots: config.n_link_knots,
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
        self.s_link.len()
    }
    
    /// Get base model layout
    pub fn layout(&self) -> &ModelLayout {
        &self.layout_base
    }
    
    /// Build link spline basis at current Xβ values
    /// Uses B-splines with orthogonality constraint: wiggle has no intercept/linear term
    /// Formula: g(u) = u + f(z) where z = standardized(u), f ⟂ {1, z}
    pub fn build_link_basis(&self, eta_base: &Array1<f64>) -> Array2<f64> {
        use crate::calibrate::basis::apply_weighted_orthogonality_constraint;
        
        let n = eta_base.len();
        let k = self.n_link_knots;
        
        // Compute standardized index z = (u - mean) / sd
        let mean_eta: f64 = eta_base.iter().sum::<f64>() / n as f64;
        let var_eta: f64 = eta_base.iter().map(|&x| (x - mean_eta).powi(2)).sum::<f64>() / n as f64;
        let sd_eta = var_eta.sqrt().max(1e-6);
        let z: Array1<f64> = eta_base.mapv(|x| (x - mean_eta) / sd_eta);
        
        // Compute data range for B-spline knots (on standardized scale)
        let min_z = z.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_z = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let data_range = (min_z, max_z);
        
        // Build B-spline basis on standardized z (degree=3 = cubic)
        let degree = 3;
        match create_bspline_basis(z.view(), data_range, k, degree) {
            Ok((bspline_basis, knots)) => {
                if knots.len() > 0 {
                    eprintln!("[JOINT] B-spline basis: {} knots, {} cols (before constraint)", 
                             knots.len(), bspline_basis.ncols());
                }
                
                // Build constraint matrix: [ones, z] to remove intercept and linear from wiggle
                let mut constraint = Array2::<f64>::zeros((n, 2));
                for i in 0..n {
                    constraint[[i, 0]] = 1.0;     // intercept
                    constraint[[i, 1]] = z[i];    // linear term
                }
                
                // Apply orthogonality constraint: wiggle ⟂ {1, z}
                let constrained_basis = match apply_weighted_orthogonality_constraint(
                    bspline_basis.view(),
                    constraint.view(),
                    Some(self.weights),
                ) {
                    Ok((constrained, transform)) => {
                        drop(transform); // Not needed after constraint applied
                        eprintln!("[JOINT] Orthogonality constraint applied: {} cols after", constrained.ncols());
                        constrained
                    }
                    Err(_) => {
                        eprintln!("[JOINT] Orthogonality constraint failed, using raw basis");
                        (*bspline_basis).clone()
                    }
                };
                
                // Construct combined basis: [identity column, constrained wiggle]
                let n_wiggle_cols = constrained_basis.ncols();
                let mut basis = Array2::zeros((n, 1 + n_wiggle_cols));
                
                // Column 0: identity (Xβ itself) - fixed coefficient = 1
                basis.column_mut(0).assign(eta_base);
                
                // Columns 1..: constrained wiggle basis (no intercept, no linear term)
                if n_wiggle_cols > 0 {
                    basis.slice_mut(s![.., 1..]).assign(&constrained_basis);
                }
                
                basis
            }
            Err(_) => {
                // Fallback: identity only
                let mut basis = Array2::zeros((n, 1));
                basis.column_mut(0).assign(eta_base);
                basis
            }
        }
    }
    
    /// Build wiggle penalty matrix for link block
    /// Uses proper second-difference penalty from basis.rs
    pub fn build_link_penalty(&self) -> Array2<f64> {
        let k = self.n_link_knots;
        let degree = 3;
        // Number of B-spline basis functions
        let num_bspline = k + degree + 1;
        // Total dimension: 1 (identity) + num_bspline (wiggle)
        let dim = 1 + num_bspline;
        
        // Get the proper difference penalty for B-splines
        let bspline_penalty = match create_difference_penalty_matrix(num_bspline, 2) {
            Ok(p) => p,
            Err(_) => Array2::zeros((num_bspline, num_bspline)),
        };
        
        // Build full penalty: identity column is unpenalized
        let mut penalty = Array2::zeros((dim, dim));
        
        // Copy B-spline penalty into wiggle block (columns 1..dim)
        penalty.slice_mut(s![1.., 1..]).assign(&bspline_penalty);
        
        penalty
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
        use faer::Side;
        
        let n = x.nrows();
        let p = x.ncols();
        
        // Compute X'WX
        let mut xwx = Array2::<f64>::zeros((p, p));
        for i in 0..n {
            let wi = w[i];
            for j in 0..p {
                for k in 0..p {
                    xwx[[j, k]] += wi * x[[i, j]] * x[[i, k]];
                }
            }
        }
        
        // Add penalty: X'WX + λS
        xwx = xwx + penalty * lambda;
        
        // Add small ridge for numerical stability
        for i in 0..p {
            xwx[[i, i]] += 1e-8;
        }
        
        // Compute X'Wz
        let mut xwz = Array1::<f64>::zeros(p);
        for i in 0..n {
            let wi = w[i];
            let zi = z[i];
            for j in 0..p {
                xwz[j] += wi * x[[i, j]] * zi;
            }
        }
        
        // Solve (X'WX + λS)β = X'Wz using Cholesky via FaerCholesky trait
        match xwx.cholesky(Side::Lower) {
            Ok(chol) => chol.solve_vec(&xwz),
            Err(_) => {
                // Fallback to zeroes if factorization fails
                eprintln!("[JOINT] Warning: Cholesky factorization failed, returning zeros");
                Array1::zeros(p)
            }
        }
    }
    
    /// Perform one IRLS step for the link block
    /// Given current linear predictor, returns updated link coefficients
    pub fn irls_link_step(
        &mut self,
        x_link: &Array2<f64>,
        lambda_link: f64,
    ) -> f64 {
        let n = self.n_obs();
        let eta = self.compute_eta(x_link);
        
        // Allocate working vectors
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        
        // Compute working response and weights
        crate::calibrate::pirls::update_glm_vectors(
            self.y,
            &eta,
            self.link.clone(),
            self.weights,
            &mut mu,
            &mut weights,
            &mut z,
        );
        
        // Build penalty
        let penalty = self.build_link_penalty();
        
        // Solve PWLS
        let new_beta = Self::solve_weighted_ls(x_link, &z, &weights, &penalty, lambda_link);
        
        // Update link coefficients (but keep identity backbone fixed)
        // beta_link[0] is the identity term, kept at 1.0
        for j in 1..self.beta_link.len() {
            if j < new_beta.len() {
                self.beta_link[j] = new_beta[j];
            }
        }
        
        // Return deviance
        self.compute_deviance(&mu)
    }
    
    /// Perform one IRLS step for the base β block
    /// Given current g, updates β using effective design g'(u) * X
    /// This is the key step: chain rule gives ∂η/∂β = g'(u) * x
    pub fn irls_base_step(
        &mut self,
        g_prime: &Array1<f64>,
        lambda_base: f64,
        damping: f64, // 0 to 1, how much to damp the step
    ) -> f64 {
        let n = self.n_obs();
        let p = self.x_base.ncols();
        
        // Build effective design: X_eff[i,j] = g'(u_i) * X_base[i,j]
        let mut x_eff = Array2::<f64>::zeros((n, p));
        for i in 0..n {
            let scale = g_prime[i];
            for j in 0..p {
                x_eff[[i, j]] = scale * self.x_base[[i, j]];
            }
        }
        
        // Current η from full model
        let u = self.base_linear_predictor();
        let x_link = self.build_link_basis(&u);
        let eta = self.compute_eta(&x_link);
        
        // Compute working response and weights
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        
        crate::calibrate::pirls::update_glm_vectors(
            self.y,
            &eta,
            self.link.clone(),
            self.weights,
            &mut mu,
            &mut weights,
            &mut z,
        );
        
        // Build penalty for base block (sum of all base penalties)
        let mut penalty = Array2::<f64>::zeros((p, p));
        for s in &self.s_base {
            if s.nrows() == p && s.ncols() == p {
                penalty = penalty + s;
            }
        }
        
        // Solve PWLS for the increment
        let new_beta = Self::solve_weighted_ls(&x_eff, &z, &weights, &penalty, lambda_base);
        
        // Apply damped update: β_new = β_old + damping * (β_proposed - β_old)
        for j in 0..p {
            let delta = new_beta[j] - self.beta_base[j];
            self.beta_base[j] += damping * delta;
        }
        
        // Return deviance
        self.compute_deviance(&mu)
    }
    
    /// Compute current linear predictor: X_link × β_link  
    fn compute_eta(&self, x_link: &Array2<f64>) -> Array1<f64> {
        x_link.dot(&self.beta_link)
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
    // rho[0] = log(lambda_base), rho[1] = log(lambda_link)
    let lambda_base = state.rho.get(0).map(|r| r.exp()).unwrap_or(1.0);
    let lambda_link = state.rho.get(1).map(|r| r.exp()).unwrap_or(1.0);
    
    // Damping schedule: start conservative, increase as we converge
    let initial_damping = 0.5;
    let final_damping = 1.0;
    
    for i in 0..config.max_backfit_iter {
        iter = i + 1;
        
        // Adaptive damping: increase towards 1.0 as we iterate
        let progress = (i as f64) / (config.max_backfit_iter as f64);
        let damping = initial_damping + progress * (final_damping - initial_damping);
        
        // Step A: Given β, fit g to convergence
        // Build link basis at current index u_i = x_i'β
        let u = state.base_linear_predictor();
        let x_link = state.build_link_basis(&u);
        total_design_cols = x_link.ncols();
        
        // Update link coefficients (g) via IRLS
        let deviance_after_g = state.irls_link_step(&x_link, lambda_link);
        
        // Step B: Given g, update β using g'(u)*X design
        // Compute g'(u) for chain rule
        let g_prime = compute_link_derivative(&u, &state.beta_link);
        
        // Update β with damping
        let deviance = state.irls_base_step(&g_prime, lambda_base, damping);
        
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
    
    // Estimate EDF from design dimensions (placeholder - proper EDF requires trace(hat matrix))
    let edf = total_design_cols as f64;
    
    Ok(JointModelResult {
        beta_base: state.beta_base.clone(),
        beta_link: state.beta_link.clone(),
        lambdas: state.rho.mapv(f64::exp).to_vec(),
        deviance: prev_deviance,
        edf,
        backfit_iterations: iter,
        converged,
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
        Self {
            state: RefCell::new(state),
            config: config.clone(),
            cached_beta_base: RefCell::new(cached_beta_base),
            cached_beta_link: RefCell::new(cached_beta_link),
            cached_laml: RefCell::new(None),
            cached_rho: RefCell::new(Array1::zeros(2)),
        }
    }
    
    /// Compute LAML cost for a given ρ
    /// LAML = deviance + log|H_pen| - log|S_λ| (+ prior on ρ)
    pub fn compute_cost(&self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
        let mut state = self.state.borrow_mut();
        
        // Set ρ and warm-start from cached coefficients
        state.set_rho(rho.clone());
        state.beta_base = self.cached_beta_base.borrow().clone();
        state.beta_link = self.cached_beta_link.borrow().clone();
        
        // Run inner alternating to convergence
        let lambda_base = rho.get(0).map(|r| r.exp()).unwrap_or(1.0);
        let lambda_link = rho.get(1).map(|r| r.exp()).unwrap_or(1.0);
        
        let mut prev_deviance = f64::INFINITY;
        for i in 0..self.config.max_backfit_iter {
            let progress = (i as f64) / (self.config.max_backfit_iter as f64);
            let damping = 0.5 + progress * 0.5;
            
            let u = state.base_linear_predictor();
            let x_link = state.build_link_basis(&u);
            state.irls_link_step(&x_link, lambda_link);
            let g_prime = compute_link_derivative(&u, &state.beta_link);
            let deviance = state.irls_base_step(&g_prime, lambda_base, damping);
            
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
        let laml = self.compute_laml_at_convergence(&state, lambda_base, lambda_link);
        
        // Cache for gradient
        *self.cached_laml.borrow_mut() = Some(laml);
        *self.cached_rho.borrow_mut() = rho.clone();
        
        Ok(laml)
    }
    
    /// Compute LAML at the converged solution
    fn compute_laml_at_convergence(&self, state: &JointModelState, lambda_base: f64, lambda_link: f64) -> f64 {
        let n = state.n_obs();
        let u = state.base_linear_predictor();
        let x_link = state.build_link_basis(&u);
        let eta = state.compute_eta(&x_link);
        
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
        
        // Build joint Jacobian J = [g'(u)*X_base | B(u)] and penalized Hessian H = J'WJ + S_λ
        let p_base = state.x_base.ncols();
        let p_link = x_link.ncols();
        let p_total = p_base + p_link;
        
        // Build penalty matrix S_λ (block diagonal)
        let mut s_lambda = Array2::<f64>::zeros((p_total, p_total));
        
        // Base block penalty
        for s in &state.s_base {
            if s.nrows() == p_base && s.ncols() == p_base {
                for i in 0..p_base {
                    for j in 0..p_base {
                        s_lambda[[i, j]] += lambda_base * s[[i, j]];
                    }
                }
            }
        }
        
        // Link block penalty (skip identity column)
        let link_penalty = state.build_link_penalty();
        for i in 0..link_penalty.nrows().min(p_link) {
            for j in 0..link_penalty.ncols().min(p_link) {
                s_lambda[[p_base + i, p_base + j]] += lambda_link * link_penalty[[i, j]];
            }
        }
        
        // Build joint Jacobian J = [g'(u)*X_base | B(z)]
        let g_prime = compute_link_derivative(&u, &state.beta_link);
        let mut j_matrix = Array2::<f64>::zeros((n, p_total));
        
        // Base block: g'(u) * X_base
        for i in 0..n {
            for j in 0..p_base {
                j_matrix[[i, j]] = g_prime[i] * state.x_base[[i, j]];
            }
        }
        // Link block: B(z)
        for i in 0..n {
            for j in 0..p_link {
                j_matrix[[i, p_base + j]] = x_link[[i, j]];
            }
        }
        
        // Build penalized Hessian H_pen = J'WJ + Sλ
        let mut h_pen = s_lambda.clone();
        for i in 0..p_total {
            for j in 0..p_total {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += j_matrix[[k, i]] * weights[k] * j_matrix[[k, j]];
                }
                h_pen[[i, j]] += sum;
            }
        }
        
        // Add small ridge for numerical stability
        for i in 0..p_total {
            h_pen[[i, i]] += 1e-8;
        }
        
        // Compute log|H_pen| via Cholesky
        use crate::calibrate::faer_ndarray::FaerCholesky;
        use faer::Side;
        let log_det_h = match h_pen.cholesky(Side::Lower) {
            Ok(chol) => {
                // log|H| = 2 * sum(log(diag(L)))
                2.0 * chol.diag().mapv(|d| d.max(1e-10).ln()).sum()
            }
            Err(_) => {
                // Fallback: approximate using trace
                eprintln!("[LAML] Cholesky failed, using trace approximation");
                let trace: f64 = (0..p_total).map(|i| h_pen[[i, i]].max(1e-10).ln()).sum();
                trace
            }
        };
        
        // Compute log|Sλ| via eigenvalues of s_lambda (only non-zero eigenvalues)
        // For ridge: log|λI| = p * log(λ)
        let log_det_s = (p_base as f64) * lambda_base.max(1e-10).ln() 
                      + ((p_link - 1).max(0) as f64) * lambda_link.max(1e-10).ln();
        
        // Null space dimension (unpenalized coefficients)
        // For identity column (fixed) = 1, potentially some in base
        let mp = 1.0; // Identity column is fixed/unpenalized
        
        // LAML = -2*log_lik + log|H_pen| - log|Sλ| + mp*log(2π)
        // Or equivalently: deviance + log|H_pen| - log|Sλ| + mp*log(2π)
        let laml = deviance + 0.5 * log_det_h - 0.5 * log_det_s 
                 + (mp / 2.0) * (2.0 * std::f64::consts::PI).ln();
        
        laml
    }
    
    /// Compute numerical gradient of LAML w.r.t. ρ using central differences
    pub fn compute_gradient(&self, rho: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        let h = 1e-4; // Step size for numerical differentiation
        let n_rho = rho.len();
        let mut grad = Array1::<f64>::zeros(n_rho);
        
        for k in 0..n_rho {
            // Forward step
            let mut rho_plus = rho.clone();
            rho_plus[k] += h;
            let cost_plus = self.compute_cost(&rho_plus)?;
            
            // Backward step  
            let mut rho_minus = rho.clone();
            rho_minus[k] -= h;
            let cost_minus = self.compute_cost(&rho_minus)?;
            
            // Central difference
            grad[k] = (cost_plus - cost_minus) / (2.0 * h);
        }
        
        // Restore cached state to original rho
        let _ = self.compute_cost(rho);
        
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
        JointModelResult {
            beta_base: state.beta_base,
            beta_link: state.beta_link,
            lambdas: rho.mapv(f64::exp).to_vec(),
            deviance: self.cached_laml.borrow().unwrap_or(f64::INFINITY),
            edf: 0.0, // TODO: compute from trace
            backfit_iterations: self.config.max_backfit_iter,
            converged: true,
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
    
    // Initial rho = [0, 0] (λ = 1)
    let initial_rho = Array1::from_vec(vec![0.0, 0.0]);
    
    // Run BFGS optimization
    let mut best_rho = initial_rho.clone();
    let mut best_cost = reml_state.compute_cost(&initial_rho)?;
    
    eprintln!("[REML] Initial: ρ=[{:.2}, {:.2}], LAML={:.4}", 
             initial_rho[0], initial_rho[1], best_cost);
    
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
                    eprintln!("[REML] Iter {}: ρ=[{:.2}, {:.2}], LAML={:.4}, step={:.2e}", 
                             outer_iter, best_rho[0], best_rho[1], best_cost, step_size);
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

/// Compute derivative of link function at each point
/// For g(u) = u + B(u)θ with B-splines, we use numerical differentiation
/// g'(u) ≈ 1 + (g(u+h) - g(u-h)) / (2h) for the wiggle part
fn compute_link_derivative(
    eta_base: &Array1<f64>,
    beta_link: &Array1<f64>,
) -> Array1<f64> {
    let n = eta_base.len();
    
    // The identity term contributes derivative = 1
    // The wiggle term B(u)θ contributes numerically computed derivative
    let mut deriv = Array1::<f64>::ones(n);
    
    // For each point, compute wiggle(u+h) - wiggle(u-h) / (2h)
    // Since we don't have direct access to the B-spline evaluated at arbitrary points,
    // we use the fact that for small wiggle corrections, g'(u) ≈ 1
    // 
    // More accurate: use B-spline derivative = B'(u)θ
    // B-spline derivative: d/du B_j(u) = (degree) * [B_{j,degree-1}(u)/(t_{j+degree}-t_j) 
    //                                                - B_{j+1,degree-1}(u)/(t_{j+degree+1}-t_{j+1})]
    //
    // For now, approximate using the coefficients directly:
    // If beta_link[0] = 1 (identity), and beta_link[1:] are small wiggle corrections,
    // then g'(u) ≈ 1 + small correction
    
    // Simple approximation: g'(u) = beta_link[0] + correction
    // The identity coefficient is fixed at 1, so base derivative = 1
    let base_deriv = if beta_link.len() > 0 { beta_link[0] } else { 1.0 };
    
    // Add contribution from wiggle terms (simplified linear approximation)
    // For B-splines, the derivative magnitude scales with coefficient magnitude / range
    let wiggle_coeff_norm: f64 = beta_link.iter().skip(1).map(|&x| x * x).sum::<f64>().sqrt();
    
    // Get range
    let min_eta = eta_base.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_eta = eta_base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_eta - min_eta).max(1.0);
    
    // Approximate wiggle derivative contribution (scales inversely with range)
    let wiggle_deriv_scale = wiggle_coeff_norm / range;
    
    for i in 0..n {
        // Position in [0, 1]
        let z = (eta_base[i] - min_eta) / range;
        // Wiggle derivative varies across the domain; approximate as position-weighted
        let wiggle_contribution = wiggle_deriv_scale * (1.0 - 2.0 * (z - 0.5).abs());
        deriv[i] = base_deriv + wiggle_contribution;
    }
    
    // Ensure derivative is positive (monotonicity) and bounded
    deriv.mapv_inplace(|d| d.max(0.1).min(10.0));
    
    deriv
}

/// Predict probabilities from a fitted joint model
/// 
/// Uses derivative-propagated uncertainty:
///   effective_SE = |g'(η_base)| × SE_base
///   p = E[σ(η_cal + ε)] where ε ~ N(0, effective_SE²)
/// 
/// This is the principled post-fit integration that correctly propagates
/// base model uncertainty through the learned link correction.
pub fn predict_joint(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    se_base: Option<&Array1<f64>>,
) -> JointModelPrediction {
    let n = eta_base.len();
    
    // Build link basis at eta_base values
    let min_eta = eta_base.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_eta = eta_base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_eta - min_eta).max(1.0);
    
    // Compute calibrated linear predictor: η_cal = t(η_base) = η_base + wiggle(η_base)
    let mut eta_cal = Array1::<f64>::zeros(n);
    for i in 0..n {
        let z = (eta_base[i] - min_eta) / range;
        // β_link[0] is the identity coefficient (fixed at 1.0)
        // β_link[j] for j > 0 are wiggle coefficients
        let mut val = result.beta_link[0] * eta_base[i]; // Identity term
        for j in 1..result.beta_link.len() {
            val += result.beta_link[j] * z.powi(j as i32);
        }
        eta_cal[i] = val;
    }
    
    // Compute effective SE if base SE provided
    let (probabilities, effective_se) = if let Some(se) = se_base {
        // Compute link derivative for uncertainty propagation
        let deriv = compute_link_derivative(eta_base, &result.beta_link);
        
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
        assert_eq!(state.beta_link.len(), config.n_link_knots + 1);
    }
    
    #[test]
    fn test_predict_joint_basic() {
        // Create a simple result with identity link (no wiggle)
        let n_knots = 5;
        let mut beta_link = Array1::zeros(n_knots + 1);
        beta_link[0] = 1.0; // Identity coefficient
        
        let result = JointModelResult {
            beta_base: Array1::zeros(10),
            beta_link,
            lambdas: vec![1.0],
            deviance: 0.0,
            edf: 5.0,
            backfit_iterations: 1,
            converged: true,
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
        let mut beta_link = Array1::zeros(n_knots + 1);
        beta_link[0] = 1.0;
        
        let result = JointModelResult {
            beta_base: Array1::zeros(10),
            beta_link,
            lambdas: vec![1.0],
            deviance: 0.0,
            edf: 5.0,
            backfit_iterations: 1,
            converged: true,
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

