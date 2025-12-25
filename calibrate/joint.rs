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
    /// Link function used during training
    pub link: LinkFunction,
    /// Base trained model artifacts needed for prediction
    pub base_model: Option<crate::calibrate::model::TrainedModel>,
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
        
        // Update knot range each build to track current u spread (with padding).
        let min_val = eta_base.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = eta_base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range_width = max_val - min_val;
        let (min_u, max_u) = if range_width > 1e-6 {
            (min_val, max_val)
        } else {
            let center = 0.5 * (min_val + max_val);
            let pad = 1.0_f64.max(center.abs() * 1e-3);
            (center - pad, center + pad)
        };
        if self.knot_range != Some((min_u, max_u)) {
            self.knot_range = Some((min_u, max_u));
            self.knot_vector = None;
        }
        
        // Standardize: z = (u - min) / (max - min) to [0, 1]
        let range_width = (max_u - min_u).max(1e-6);
        let z: Array1<f64> = eta_base
            .mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
        
        // Build B-spline basis on z ∈ [0, 1]
        let data_range = (0.0, 1.0);
        let basis_result = if let Some(knots) = self.knot_vector.as_ref() {
            create_bspline_basis_with_knots(z.view(), knots.view(), degree)
                .map(|(basis, _)| (basis, knots.clone()))
        } else {
            create_bspline_basis(z.view(), data_range, k, degree)
        };
        match basis_result {
            Ok((bspline_basis, knots)) => {
                let bspline_basis = bspline_basis.as_ref();
                // Store knot vector if not already stored
                if self.knot_vector.is_none() {
                    self.knot_vector = Some(knots);
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
                        
                        // Reset beta_link to avoid mismatched coordinates after transform update
                        self.beta_link = Array1::zeros(n_constrained);
                        
                        constrained_basis
                    }
                    Err(_) => {
                        // Fallback: construct a nullspace transform via eigendecomposition.
                        eprintln!("[JOINT] Orthogonality constraint failed");
                        let mut weighted_constraints = constraint.clone();
                        for i in 0..n {
                            let w = self.weights[i];
                            weighted_constraints[[i, 0]] *= w;
                            weighted_constraints[[i, 1]] *= w;
                        }
                        let constraint_cross = bspline_basis.t().dot(&weighted_constraints); // k×2
                        let cross_prod = constraint_cross.dot(&constraint_cross.t()); // k×k
                        
                        use crate::calibrate::faer_ndarray::FaerEigh;
                        use faer::Side;
                        let (eigs, evecs): (Array1<f64>, Array2<f64>) = cross_prod
                            .eigh(Side::Lower)
                            .unwrap_or_else(|_| (Array1::zeros(n_raw), Array2::eye(n_raw)));
                        let max_eig = eigs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                        let tol = if max_eig > 0.0 { max_eig * 1e-12 } else { 1e-12 };
                        let null_indices: Vec<usize> = eigs
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &ev)| if ev <= tol { Some(i) } else { None })
                            .collect();
                        
                        let transform = if null_indices.is_empty() {
                            Array2::eye(n_raw)
                        } else {
                            let mut z = Array2::<f64>::zeros((n_raw, null_indices.len()));
                            for (col, &idx) in null_indices.iter().enumerate() {
                                let vec = evecs.column(idx);
                                z.column_mut(col).assign(&vec);
                            }
                            z
                        };
                        
                        let n_constrained = transform.ncols();
                        let constrained_basis = bspline_basis.dot(&transform);
                        
                        let raw_penalty = match create_difference_penalty_matrix(n_raw, 2) {
                            Ok(p) => p,
                            Err(_) => Array2::zeros((n_raw, n_raw)),
                        };
                        let projected_penalty = transform.t().dot(&raw_penalty).dot(&transform);
                        
                        self.link_transform = transform;
                        self.s_link_constrained = projected_penalty;
                        self.n_constrained_basis = n_constrained;
                        // Reset beta_link to avoid mismatched coordinates after transform update
                        self.beta_link = Array1::zeros(n_constrained);
                        constrained_basis
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
            Ok((basis, _)) => basis.as_ref().clone(),
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
        let eta = if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
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
        
        // Recompute deviance using updated coefficients
        let eta_updated = self.compute_eta_full(u, b_wiggle);
        let mut mu_updated = Array1::<f64>::zeros(n);
        let mut weights_updated = Array1::<f64>::zeros(n);
        let mut z_updated = Array1::<f64>::zeros(n);
        crate::calibrate::pirls::update_glm_vectors(
            self.y,
            &eta_updated,
            self.link.clone(),
            self.weights,
            &mut mu_updated,
            &mut weights_updated,
            &mut z_updated,
        );
        drop(weights_updated);
        drop(z_updated);
        self.compute_deviance(&mu_updated)
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
        let wiggle: Array1<f64> = if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
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
        
        // Recompute deviance using updated coefficients
        let u_updated = self.base_linear_predictor();
        let wiggle_updated: Array1<f64> = if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
            b_wiggle.dot(&self.beta_link)
        } else {
            Array1::zeros(n)
        };
        let eta_updated: Array1<f64> = &u_updated + &wiggle_updated;
        let mut mu_updated = Array1::<f64>::zeros(n);
        let mut weights_updated = Array1::<f64>::zeros(n);
        let mut z_updated = Array1::<f64>::zeros(n);
        crate::calibrate::pirls::update_glm_vectors(
            self.y,
            &eta_updated,
            self.link.clone(),
            self.weights,
            &mut mu_updated,
            &mut weights_updated,
            &mut z_updated,
        );
        drop(weights_updated);
        drop(z_updated);
        self.compute_deviance(&mu_updated)
    }
    
    /// Compute current linear predictor: η = u + B_wiggle · θ
    pub fn compute_eta_full(&self, u: &Array1<f64>, b_wiggle: &Array2<f64>) -> Array1<f64> {
        if b_wiggle.ncols() > 0 && self.beta_link.len() == b_wiggle.ncols() {
            let wiggle = b_wiggle.dot(&self.beta_link);
            u + &wiggle
        } else {
            u.clone()
        }
    }
    
    /// Compute deviance based on link function
    fn compute_deviance(&self, mu: &Array1<f64>) -> f64 {
        crate::calibrate::pirls::calculate_deviance(self.y, mu, self.link.clone(), self.weights)
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
        
        // Update link coefficients (θ) via IRLS with u as OFFSET
        let deviance_after_g = state.irls_link_step(&b_wiggle, &u, lambda_link);
        
        // Step B: Given g, update β using g'(u)*X design
        // Get knot range for derivative computation
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
            prev_deviance = deviance;
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
    
    let u_final = state.base_linear_predictor();
    let b_final = state.build_link_basis(&u_final);
    let eta_final = state.compute_eta_full(&u_final, &b_final);
    let mut mu_final = Array1::<f64>::zeros(state.n_obs());
    let mut weights_final = Array1::<f64>::zeros(state.n_obs());
    let mut z_final = Array1::<f64>::zeros(state.n_obs());
    crate::calibrate::pirls::update_glm_vectors(
        state.y,
        &eta_final,
        state.link.clone(),
        state.weights,
        &mut mu_final,
        &mut weights_final,
        &mut z_final,
    );
    drop(z_final);
    let g_prime_final = compute_link_derivative_from_state(&state, &u_final, &b_final);
    let edf = JointRemlState::compute_joint_edf(
        &state,
        &b_final,
        &g_prime_final,
        &weights_final,
        &lambda_base,
        lambda_link,
    )
    .unwrap_or(f64::NAN);
    
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
        link: state.link.clone(),
        base_model: None,
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
    cached_edf: RefCell<Option<f64>>,
    last_backfit_iterations: RefCell<usize>,
    last_converged: RefCell<bool>,
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
    cached_edf: Option<f64>,
    last_backfit_iterations: usize,
    last_converged: bool,
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
            cached_edf: *reml.cached_edf.borrow(),
            last_backfit_iterations: *reml.last_backfit_iterations.borrow(),
            last_converged: *reml.last_converged.borrow(),
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
        *reml.cached_edf.borrow_mut() = self.cached_edf;
        *reml.last_backfit_iterations.borrow_mut() = self.last_backfit_iterations;
        *reml.last_converged.borrow_mut() = self.last_converged;
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
        let mut state = JointModelState::new(y, weights, x_base, s_base, layout_base, link, config);
        let u0 = state.base_linear_predictor();
        let _ = state.build_link_basis(&u0);
        let n_base = state.s_base.len();
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
            cached_rho: RefCell::new(Array1::zeros(n_base + 1)),
            cached_edf: RefCell::new(None),
            last_backfit_iterations: RefCell::new(0),
            last_converged: RefCell::new(false),
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
        let mut iter_count = 0;
        let mut converged = false;
        for i in 0..self.config.max_backfit_iter {
            let progress = (i as f64) / (self.config.max_backfit_iter as f64);
            let damping = 0.5 + progress * 0.5;
            
            let u = state.base_linear_predictor();
            let b_wiggle = state.build_link_basis(&u);
            state.irls_link_step(&b_wiggle, &u, lambda_link);
            
            let g_prime = compute_link_derivative_from_state(&state, &u, &b_wiggle);
            let deviance = state.irls_base_step(&b_wiggle, &g_prime, &lambda_base, damping);
            
            let delta = (prev_deviance - deviance).abs() / (deviance.abs() + 1.0);
            iter_count = i + 1;
            if delta < self.config.backfit_tol {
                converged = true;
                break;
            }
            prev_deviance = deviance;
        }
        
        // Cache converged coefficients for warm-start
        *self.cached_beta_base.borrow_mut() = state.beta_base.clone();
        *self.cached_beta_link.borrow_mut() = state.beta_link.clone();
        
        // Compute LAML = deviance + log|H_pen| - log|S_λ|
        let (laml, edf) = self.compute_laml_at_convergence(&state, &lambda_base, lambda_link);
        
        // Cache for gradient
        *self.cached_laml.borrow_mut() = Some(laml);
        *self.cached_rho.borrow_mut() = rho.clone();
        *self.cached_edf.borrow_mut() = edf;
        *self.last_backfit_iterations.borrow_mut() = iter_count;
        *self.last_converged.borrow_mut() = converged;
        
        Ok(-laml)
    }
    
    /// Compute LAML at the converged solution
    fn compute_laml_at_convergence(&self, state: &JointModelState, lambda_base: &Array1<f64>, lambda_link: f64) -> (f64, Option<f64>) {
        let n = state.n_obs();
        let u = state.base_linear_predictor();
        let b_wiggle = state.build_link_basis_from_state(&u);
        
        // Compute eta = u + B_wiggle * theta
        let eta = state.compute_eta_full(&u, &b_wiggle);
        
        // Compute mu/weights at convergence
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        crate::calibrate::pirls::update_glm_vectors(
            state.y,
            &eta,
            state.link.clone(),
            state.weights,
            &mut mu,
            &mut weights,
            &mut z,
        );
        drop(z);
        let deviance = state.compute_deviance(&mu);
        
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
                    Err(_) => {
                        use crate::calibrate::faer_ndarray::FaerEigh;
                        let (eigs, _) = schur
                            .clone()
                            .eigh(Side::Lower)
                            .unwrap_or_else(|_| (Array1::zeros(p_link), Array2::eye(p_link)));
                        let max_eig = eigs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                        let tol = if max_eig > 0.0 { max_eig * 1e-12 } else { 1e-12 };
                        eigs.iter().map(|&ev| ev.max(tol).ln()).sum()
                    }
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
        
        // Penalized log-likelihood term: -0.5*deviance - 0.5*beta'Sλ beta
        let mut penalty_term = 0.0;
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                let sb = s_k.dot(&state.beta_base);
                penalty_term += lambda_k * state.beta_base.dot(&sb);
            }
        }
        if p_link > 0 && link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
            if state.beta_link.len() == p_link {
                let sb = link_penalty.dot(&state.beta_link);
                penalty_term += lambda_link * state.beta_link.dot(&sb);
            }
        }
        
        let laml = match state.link {
            LinkFunction::Logit => {
                let penalised_ll = -0.5 * deviance - 0.5 * penalty_term;
                let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_a
                    + (mp / 2.0) * (2.0 * std::f64::consts::PI).ln();
                laml
            }
            LinkFunction::Identity => {
                let dp = (deviance + penalty_term).max(1e-12);
                let denom = (n as f64 - mp).max(1.0);
                let phi = dp / denom;
                let reml_cost = dp / (2.0 * phi)
                    + 0.5 * (log_det_a - log_det_s)
                    + ((n as f64 - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();
                -reml_cost
            }
        };
        
        let edf = Self::compute_joint_edf(
            state,
            &b_wiggle,
            &g_prime,
            &weights,
            lambda_base,
            lambda_link,
        );
        
        (laml, edf)
    }

    fn compute_joint_edf(
        state: &JointModelState,
        b_wiggle: &Array2<f64>,
        g_prime: &Array1<f64>,
        weights: &Array1<f64>,
        lambda_base: &Array1<f64>,
        lambda_link: f64,
) -> Option<f64> {
    use crate::calibrate::faer_ndarray::FaerCholesky;
    use faer::Side;
    
    let n = state.n_obs();
    let p_base = state.x_base.ncols();
    let p_link = b_wiggle.ncols();
    let p_total = p_base + p_link;
    if p_total == 0 {
        return Some(0.0);
    }
    
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
    
    let mut wb = b_wiggle.clone();
    for i in 0..n {
        let scale = weights[i] * g_prime[i];
        for j in 0..p_link {
            wb[[i, j]] *= scale;
        }
    }
    let c_mat = crate::calibrate::faer_ndarray::fast_atb(&state.x_base, &wb);
    
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
    
    for i in 0..p_base {
        a_mat[[i, i]] += 1e-8;
    }
    for i in 0..p_link {
        d_mat[[i, i]] += 1e-8;
    }
    
    let mut h_mat = Array2::<f64>::zeros((p_total, p_total));
    for i in 0..p_base {
        for j in 0..p_base {
            h_mat[[i, j]] = a_mat[[i, j]];
        }
    }
    for i in 0..p_base {
        for j in 0..p_link {
            h_mat[[i, p_base + j]] = c_mat[[i, j]];
            h_mat[[p_base + j, i]] = c_mat[[i, j]];
        }
    }
    for i in 0..p_link {
        for j in 0..p_link {
            h_mat[[p_base + i, p_base + j]] = d_mat[[i, j]];
        }
    }
    
    let mut s_lambda = Array2::<f64>::zeros((p_total, p_total));
    for (idx, s_k) in state.s_base.iter().enumerate() {
        let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
        if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
            for i in 0..p_base {
                for j in 0..p_base {
                    s_lambda[[i, j]] += lambda_k * s_k[[i, j]];
                }
            }
        }
    }
    if link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
        for i in 0..p_link {
            for j in 0..p_link {
                s_lambda[[p_base + i, p_base + j]] += lambda_link * link_penalty[[i, j]];
            }
        }
    }
    
    let chol = h_mat.cholesky(Side::Lower).ok()?;
    let mut trace = 0.0;
    for j in 0..p_total {
        let col = s_lambda.column(j).to_owned();
        let solved = chol.solve_vec(&col);
        trace += solved[j];
    }
    
    Some(p_total as f64 - trace)
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
    pub fn cost_and_grad(&self, rho: &Array1<f64>) -> (f64, Array1<f64>) {
        let cost = match self.compute_cost(rho) {
            Ok(val) if val.is_finite() => val,
            Ok(_) => {
                eprintln!("[JOINT][REML] Non-finite cost; returning large penalty.");
                return (f64::INFINITY, Array1::zeros(rho.len()));
            }
            Err(err) => {
                eprintln!("[JOINT][REML] Cost evaluation failed: {err}");
                return (f64::INFINITY, Array1::zeros(rho.len()));
            }
        };
        let grad = match self.compute_gradient(rho) {
            Ok(grad) => grad,
            Err(err) => {
                eprintln!("[JOINT][REML] Gradient evaluation failed: {err}");
                Array1::zeros(rho.len())
            }
        };
        (cost, grad)
    }
    
    /// Extract final result after optimization
    pub fn into_result(self) -> JointModelResult {
        let cached_edf = *self.cached_edf.borrow();
        let cached_iters = *self.last_backfit_iterations.borrow();
        let cached_converged = *self.last_converged.borrow();
        let state = self.state.into_inner();
        let rho = state.rho.clone();
        let knot_range = state.knot_range.unwrap_or((0.0, 1.0));
        let knot_vector = state.knot_vector.clone().unwrap_or_else(|| Array1::zeros(0));
        let b_wiggle = state.build_link_basis_from_state(&state.base_linear_predictor());
        let eta = state.compute_eta_full(&state.base_linear_predictor(), &b_wiggle);
        let mut mu = Array1::<f64>::zeros(state.n_obs());
        let mut weights = Array1::<f64>::zeros(state.n_obs());
        let mut z = Array1::<f64>::zeros(state.n_obs());
        crate::calibrate::pirls::update_glm_vectors(
            state.y,
            &eta,
            state.link.clone(),
            state.weights,
            &mut mu,
            &mut weights,
            &mut z,
        );
        drop(weights);
        drop(z);
        let deviance = state.compute_deviance(&mu);
        JointModelResult {
            beta_base: state.beta_base,
            beta_link: state.beta_link,
            lambdas: rho.mapv(f64::exp).to_vec(),
            deviance,
            edf: cached_edf.unwrap_or(f64::NAN),
            backfit_iterations: cached_iters,
            converged: cached_converged,
            knot_range,
            knot_vector,
            link_transform: state.link_transform,
            degree: state.degree,
            link: state.link.clone(),
            base_model: None,
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
    use wolfe_bfgs::Bfgs;
    let solver = Bfgs::new(initial_rho.clone(), |rho| reml_state.cost_and_grad(rho))
        .with_tolerance(config.reml_tol)
        .with_max_iterations(config.max_reml_iter)
        .with_fp_tolerances(1e2, 1e2)
        .with_no_improve_stop(1e-8, 5)
        .with_rng_seed(0xC0FFEE_u64);
    
    let solution = match solver.run() {
        Ok(solution) => solution,
        Err(wolfe_bfgs::BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
        Err(wolfe_bfgs::BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
        Err(e) => {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "BFGS failed for joint model: {e:?}"
            )));
        }
    };
    
    let best_rho = solution.final_point.clone();
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
    let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
    let n_constrained = state.link_transform.ncols();
    if n_raw == 0 || n_constrained == 0 {
        return deriv;
    }
    if state.beta_link.len() != n_constrained {
        return deriv;
    }
    
    let mut raw_plus = vec![0.0; n_raw];
    let mut raw_minus = vec![0.0; n_raw];
    let mut plus = vec![0.0; n_constrained];
    let mut minus = vec![0.0; n_constrained];
    let mut scratch = crate::calibrate::basis::SplineScratch::new(state.degree);
    
    for i in 0..n {
        let dz = (z_plus[i] - z_minus[i]).max(1e-8);
        raw_plus.fill(0.0);
        raw_minus.fill(0.0);
        if crate::calibrate::basis::evaluate_bspline_basis_scalar(
            z_plus[i],
            knot_vector.view(),
            state.degree,
            &mut raw_plus,
            &mut scratch,
        )
        .is_err()
        {
            continue;
        }
        if crate::calibrate::basis::evaluate_bspline_basis_scalar(
            z_minus[i],
            knot_vector.view(),
            state.degree,
            &mut raw_minus,
            &mut scratch,
        )
        .is_err()
        {
            continue;
        }
        
        plus.fill(0.0);
        minus.fill(0.0);
        if state.link_transform.nrows() == n_raw && state.link_transform.ncols() == n_constrained {
            for r in 0..n_raw {
                let vrp = raw_plus[r];
                let vrm = raw_minus[r];
                for c in 0..n_constrained {
                    plus[c] += vrp * state.link_transform[[r, c]];
                    minus[c] += vrm * state.link_transform[[r, c]];
                }
            }
        } else if n_constrained == n_raw {
            plus.copy_from_slice(&raw_plus);
            minus.copy_from_slice(&raw_minus);
        } else {
            continue;
        }
        
        let mut d_wiggle_dz = 0.0;
        for j in 0..n_constrained {
            d_wiggle_dz += (plus[j] - minus[j]) * state.beta_link[j];
        }
        d_wiggle_dz /= dz;
        deriv[i] = 1.0 + d_wiggle_dz / range_width;
    }
    
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
    let n_raw = result.knot_vector.len().saturating_sub(result.degree + 1);
    let n_constrained = result.link_transform.ncols();
    if n_raw == 0 || n_constrained == 0 {
        return deriv;
    }
    if result.beta_link.len() != n_constrained {
        return deriv;
    }
    
    let mut raw_plus = vec![0.0; n_raw];
    let mut raw_minus = vec![0.0; n_raw];
    let mut plus = vec![0.0; n_constrained];
    let mut minus = vec![0.0; n_constrained];
    let mut scratch = crate::calibrate::basis::SplineScratch::new(result.degree);
    
    for i in 0..n {
        let dz = (z_plus[i] - z_minus[i]).max(1e-8);
        raw_plus.fill(0.0);
        raw_minus.fill(0.0);
        if crate::calibrate::basis::evaluate_bspline_basis_scalar(
            z_plus[i],
            result.knot_vector.view(),
            result.degree,
            &mut raw_plus,
            &mut scratch,
        )
        .is_err()
        {
            continue;
        }
        if crate::calibrate::basis::evaluate_bspline_basis_scalar(
            z_minus[i],
            result.knot_vector.view(),
            result.degree,
            &mut raw_minus,
            &mut scratch,
        )
        .is_err()
        {
            continue;
        }
        
        plus.fill(0.0);
        minus.fill(0.0);
        if result.link_transform.nrows() == n_raw && result.link_transform.ncols() == n_constrained {
            for r in 0..n_raw {
                let vrp = raw_plus[r];
                let vrm = raw_minus[r];
                for c in 0..n_constrained {
                    plus[c] += vrp * result.link_transform[[r, c]];
                    minus[c] += vrm * result.link_transform[[r, c]];
                }
            }
        } else if n_constrained == n_raw {
            plus.copy_from_slice(&raw_plus);
            minus.copy_from_slice(&raw_minus);
        } else {
            continue;
        }
        
        let mut d_wiggle_dz = 0.0;
        for j in 0..n_constrained {
            d_wiggle_dz += (plus[j] - minus[j]) * result.beta_link[j];
        }
        d_wiggle_dz /= dz;
        deriv[i] = 1.0 + d_wiggle_dz / range_width;
    }
    
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
            let raw = basis.as_ref();
            if result.link_transform.ncols() > 0 && result.link_transform.nrows() == raw.ncols() {
                raw.dot(&result.link_transform)
            } else {
                raw.clone()
            }
        }
        Err(_) => Array2::zeros((n, result.beta_link.len())),
    };
    
    // Compute η_cal = u + B_wiggle · θ
    let eta_cal: Array1<f64> = if b_wiggle.ncols() > 0 && result.beta_link.len() == b_wiggle.ncols() {
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
        
        let probs = match result.link {
            LinkFunction::Logit => (0..n)
                .map(|i| crate::calibrate::quadrature::logit_posterior_mean(eta_cal[i], eff_se[i]))
                .collect::<Array1<f64>>(),
            LinkFunction::Identity => eta_cal.clone(),
        };
        
        (probs, Some(eff_se))
    } else {
        let probs = match result.link {
            LinkFunction::Logit => eta_cal.mapv(|e| 1.0 / (1.0 + (-e).exp())),
            LinkFunction::Identity => eta_cal.clone(),
        };
        (probs, None)
    };
    
    JointModelPrediction {
        eta: eta_cal,
        probabilities,
        effective_se,
    }
}

/// Predict using the stored base_model plus the joint link calibration.
pub fn predict_joint_from_base_model(
    result: &JointModelResult,
    p_new: ArrayView1<f64>,
    sex_new: ArrayView1<f64>,
    pcs_new: ArrayView2<f64>,
) -> Result<JointModelPrediction, crate::calibrate::model::ModelError> {
    let base = result
        .base_model
        .as_ref()
        .ok_or(crate::calibrate::model::ModelError::CalibratorMissing)?;
    let pred = base.predict_detailed(p_new, sex_new, pcs_new)?;
    let eta_base = pred.0;
    let mean = pred.1;
    let se_eta_opt = pred.3;
    if base.joint_link.is_some() {
        return Ok(JointModelPrediction {
            eta: eta_base,
            probabilities: mean,
            effective_se: se_eta_opt,
        });
    }
    Ok(predict_joint(result, &eta_base, se_eta_opt.as_ref()))
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
        // Create a simple result with logit link (no wiggle)
        let n_knots = 5;
        let degree = 3;
        let (basis, knot_vector) = create_bspline_basis(
            Array1::from_vec(vec![0.0]).view(),
            (0.0, 1.0),
            n_knots,
            degree,
        )
        .expect("basis");
        drop(basis);
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
            link: LinkFunction::Logit,
            base_model: None,
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
        
        // With logit link, prob at eta=0 should be ~0.5
        assert!((pred.probabilities[2] - 0.5).abs() < 0.01);
    }
    
    #[test]
    fn test_predict_joint_with_se() {
        let n_knots = 5;
        let degree = 3;
        let (basis, knot_vector) = create_bspline_basis(
            Array1::from_vec(vec![0.0]).view(),
            (0.0, 1.0),
            n_knots,
            degree,
        )
        .expect("basis");
        drop(basis);
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
            link: LinkFunction::Logit,
            base_model: None,
        };
        
        let eta_base = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let se_base = Array1::from_vec(vec![0.5, 0.5, 0.5]);
        
        let pred = predict_joint(&result, &eta_base, Some(&se_base));
        
        assert!(pred.effective_se.is_some());
        let eff_se = pred.effective_se.unwrap();
        assert_eq!(eff_se.len(), 3);
        
        // With zero wiggle, g'(u)=1 so effective SE equals base SE.
        for i in 0..3 {
            assert!((eff_se[i] - se_base[i]).abs() < 0.1);
        }
    }
}
