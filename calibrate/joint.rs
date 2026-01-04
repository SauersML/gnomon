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
use ndarray::s;
use std::cell::RefCell;
use crate::calibrate::construction::ModelLayout;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::model::LinkFunction;
use crate::calibrate::basis::{
    baseline_lambda_seed, create_basis, create_difference_penalty_matrix, BasisOptions, Dense,
    KnotSource,
};
use crate::calibrate::quadrature::QuadratureContext;
use crate::calibrate::construction::{
    compute_penalty_square_roots, precompute_reparam_invariant, stable_reparameterization,
    stable_reparameterization_with_invariant, ReparamInvariant, ReparamResult,
};
use crate::calibrate::seeding::{generate_rho_candidates, SeedConfig, SeedStrategy};
use crate::calibrate::visualizer;
use wolfe_bfgs::BfgsSolution;


// NOTE on z standardization: We use hard clamp(0,1) everywhere.
// At clamped boundaries (z=0 or z=1), the wiggle contribution is constant (not extrapolated),
// and g'(u) = 1 (the derivative code explicitly returns 1.0 at boundaries).
// This is consistent between training and HMC.

/// Ensure a matrix is positive definite by adding a minimal conditional ridge.
/// Only adds regularization if Cholesky fails, avoiding bias on well-conditioned matrices.
fn ensure_positive_definite_joint(mat: &mut Array2<f64>) {
    use crate::calibrate::faer_ndarray::FaerCholesky;
    use faer::Side;
    
    if mat.cholesky(Side::Lower).is_ok() {
        return; // Already positive definite, no regularization needed
    }
    
    // Matrix needs regularization - use diagonal-scaled nugget
    let diag_scale = mat.diag().iter().map(|&d| d.abs()).fold(0.0_f64, f64::max).max(1.0);
    let nugget = 1e-8 * diag_scale;
    for i in 0..mat.nrows() {
        mat[[i, i]] += nugget;
    }
}


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
    /// Transformed penalty for link block (Z'SZ) - None until build_link_basis is called
    s_link_constrained: Option<Array2<f64>>,
    /// Constraint transform Z (basis → constrained basis) - None until build_link_basis is called
    link_transform: Option<Array2<f64>>,
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
    /// Optional per-observation SE for integrated (GHQ) likelihood.
    /// When present, uses update_glm_vectors_integrated for uncertainty-aware fitting.
    covariate_se: Option<Array1<f64>>,
    quad_ctx: QuadratureContext,
    /// Enable Firth bias reduction for separation protection
    firth_bias_reduction: bool,
    /// Last full linear predictor (u + wiggle) for weight-aligned constraints.
    last_eta: Option<Array1<f64>>,
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
    /// Enable Firth bias reduction (protects against separation in logistic regression)
    pub firth_bias_reduction: bool,
}

impl Default for JointModelConfig {
    fn default() -> Self {
        Self {
            max_backfit_iter: 20,
            backfit_tol: 1e-4,
            max_reml_iter: 50,
            reml_tol: 1e-6,
            n_link_knots: 10,
            firth_bias_reduction: false, // Off by default, enable for rare-event data
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
    /// Constrained link penalty matrix (Z'SZ) used in REML fit
    pub s_link_constrained: Array2<f64>,
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
        quad_ctx: QuadratureContext,
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
        
        // link_transform and s_link_constrained are None until build_link_basis is called
        Self {
            y,
            weights,
            x_base,
            beta_base,
            beta_link,
            s_base,
            s_link_constrained: None,
            link_transform: None,
            rho,
            link,
            layout_base,
            n_link_knots: config.n_link_knots,
            degree,
            knot_range: None,
            knot_vector: None,
            n_constrained_basis: n_constrained,
            covariate_se: None,
            quad_ctx,
            firth_bias_reduction: config.firth_bias_reduction,
            last_eta: None,
        }
    }
    
    /// Set per-observation SE for integrated (GHQ) likelihood.
    /// When set, the joint model uses uncertainty-aware IRLS updates.
    pub fn with_covariate_se(mut self, se: Array1<f64>) -> Self {
        self.covariate_se = Some(se);
        self
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
    /// 
    /// The orthogonality constraint is applied using IRLS working weights
    /// from the current linear predictor to align with the optimization metric.
    /// 
    /// Returns Err if z has degenerate variance (cannot build identifiable constraint).
    pub fn build_link_basis(&mut self, eta_base: &Array1<f64>) -> Result<Array2<f64>, String> {
        use crate::calibrate::basis::apply_weighted_orthogonality_constraint;
        
        let eta_for_weights = match &self.last_eta {
            Some(last) if last.len() == eta_base.len() => last,
            _ => eta_base,
        };

        // Compute IRLS working weights to align constraint metric with the IRLS quadratic approximation.
        let mut mu = Array1::<f64>::zeros(eta_base.len());
        let mut irls_weights = Array1::<f64>::zeros(eta_base.len());
        let mut z = Array1::<f64>::zeros(eta_base.len());
        match (&self.link, &self.covariate_se) {
            (LinkFunction::Logit, Some(se)) => {
                crate::calibrate::pirls::update_glm_vectors_integrated(
                    &self.quad_ctx,
                    self.y,
                    eta_for_weights,
                    se.view(),
                    self.weights,
                    &mut mu,
                    &mut irls_weights,
                    &mut z,
                );
            }
            (LinkFunction::Logit, None) => {
                crate::calibrate::pirls::update_glm_vectors(
                    self.y,
                    eta_for_weights,
                    LinkFunction::Logit,
                    self.weights,
                    &mut mu,
                    &mut irls_weights,
                    &mut z,
                );
            }
            (LinkFunction::Identity, _) => {
                irls_weights.assign(&self.weights);
            }
        }
        
        let n = eta_base.len();
        let k = self.n_link_knots;
        let degree = self.degree;
        
        // Freeze knot range after first initialization to keep the objective stable.
        let (min_u, max_u) = if let Some(range) = self.knot_range {
            range
        } else {
            let min_val = eta_base.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = eta_base.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range_width = max_val - min_val;
            let range = if range_width > 1e-6 {
                (min_val, max_val)
            } else {
                let center = 0.5 * (min_val + max_val);
                let pad = 1.0_f64.max(center.abs() * 1e-3);
                (center - pad, center + pad)
            };
            self.knot_range = Some(range);
            range
        };
        
        // Standardize: z = (u - min) / (max - min) to [0, 1]
        let range_width = (max_u - min_u).max(1e-6);
        let z: Array1<f64> = eta_base
            .mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
        
        // Build B-spline basis on z ∈ [0, 1]
        let data_range = (0.0, 1.0);
        let basis_result = if let Some(knots) = self.knot_vector.as_ref() {
            create_basis::<Dense>(
                z.view(),
                KnotSource::Provided(knots.view()),
                degree,
                BasisOptions::value(),
            )
            .map(|(basis, _)| (basis, knots.clone()))
        } else {
            create_basis::<Dense>(
                z.view(),
                KnotSource::Generate {
                    data_range,
                    num_internal_knots: k,
                },
                degree,
                BasisOptions::value(),
            )
        };
        match basis_result {
            Ok((bspline_basis, knots)) => {
                let bspline_basis = bspline_basis.as_ref();
                // Store knot vector if not already stored
                if self.knot_vector.is_none() {
                    self.knot_vector = Some(knots);
                }
                
                let n_raw = bspline_basis.ncols();

                // Check if z has sufficient variance for a well-conditioned constraint
                // If z is nearly constant, the constraint matrix [1,z] is rank-deficient
                let z_mean: f64 = z.iter().sum::<f64>() / n as f64;
                let z_var: f64 = z.iter().map(|&v| (v - z_mean).powi(2)).sum::<f64>() / n as f64;
                let z_has_spread = z_var > 1e-6;
                
                // Error on degenerate z - cannot build identifiable constraint
                if !z_has_spread {
                    return Err(format!(
                        "Base predictor z has near-zero variance (var={z_var:.2e}). \
                        Cannot build identifiable link constraint."
                    ));
                }
                
                // Always rebuild constraint from current z and IRLS weights
                // (do NOT cache transform - it depends on data that changes each iteration)
                
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
                    Some(irls_weights.view()),  // Use IRLS weights for constraint metric
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
                        self.link_transform = Some(transform);
                        self.s_link_constrained = Some(projected_penalty);
                        self.n_constrained_basis = n_constrained;
                        if self.beta_link.len() != n_constrained {
                            self.beta_link = Array1::zeros(n_constrained);
                        }
                        
                        Ok(constrained_basis)
                    }
                    Err(_) => {
                        // Fallback: construct a nullspace transform via eigendecomposition.
                        // Use IRLS weights (not data weights) for consistency with primary path
                        eprintln!("[JOINT] Orthogonality constraint failed, using eigendecomposition fallback");
                        let mut weighted_constraints = constraint.clone();
                        for i in 0..n {
                            let w = irls_weights[i];  // Use IRLS weights, not self.weights
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
                            // WARN: No null space found means constraint failed - model may confound β and θ
                            eprintln!("[JOINT WARNING] Orthogonality constraint found no null space - \
                                      falling back to unconstrained link. Model identifiability may be compromised.");
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
                        
                        self.link_transform = Some(transform);
                        self.s_link_constrained = Some(projected_penalty);
                        self.n_constrained_basis = n_constrained;
                        if self.beta_link.len() != n_constrained {
                            self.beta_link = Array1::zeros(n_constrained);
                        }
                        Ok(constrained_basis)
                    }
                }
            }
            Err(e) => {
                // B-spline basis construction failed - return error
                Err(format!("B-spline basis construction failed: {}", e))
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
        
        let b_raw = match create_basis::<Dense>(
            z.view(),
            KnotSource::Provided(knot_vector.view()),
            self.degree,
            BasisOptions::value(),
        ) {
            Ok((basis, _)) => basis.as_ref().clone(),
            Err(_) => Array2::zeros((n, 0)),
        };
        
        if let Some(ref transform) = self.link_transform {
            if transform.ncols() > 0 && transform.nrows() == b_raw.ncols() {
                return b_raw.dot(transform);
            }
        }
        Array2::zeros((n, self.beta_link.len()))
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
        self.s_link_constrained.clone().unwrap_or_else(|| Array2::zeros((0, 0)))
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
        
        // Conditional regularization for numerical stability
        ensure_positive_definite_joint(&mut xwx);
        
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
        self.last_eta = Some(eta.clone());
        
        // Allocate working vectors
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut z_glm = Array1::<f64>::zeros(n);
        
        // Compute working response and weights
        // Note: Firth bias reduction in joint model would require using GamWorkingModel pattern
        // from pirls.rs which maintains Hessian state. For now, we use standard GLM vectors
        // but the flag triggers Firth in the outer LAML cost (via config passthrough).
        if let (LinkFunction::Logit, Some(se)) = (&self.link, &self.covariate_se) {
                crate::calibrate::pirls::update_glm_vectors_integrated(
                    &self.quad_ctx,
                self.y,
                &eta,
                se.view(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        } else {
            crate::calibrate::pirls::update_glm_vectors(
                self.y,
                &eta,
                self.link.clone(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        }
        
        // When Firth is enabled for logit, compute Firth correction using hat diagonal
        // Firth modifies the working response: z_firth = z + h_ii * (0.5 - μ) / w_i
        let z_firth = if self.firth_bias_reduction && matches!(self.link, LinkFunction::Logit) && b_wiggle.ncols() > 0 {
            // Compute hat diagonal from the unpenalized Fisher information (B'WB).
            // For efficiency, use: h_ii = w_i * ||solve(H, b_i)||^2 where b_i is ith row of B√w
            let p = b_wiggle.ncols();
            
            // Build weighted design: B_w = sqrt(W) * B
            let mut b_weighted = b_wiggle.clone();
            for i in 0..n {
                let sqrt_w = weights[i].max(0.0).sqrt();
                for j in 0..p {
                    b_weighted[[i, j]] *= sqrt_w;
                }
            }
            
            // Build Fisher information: H = B'WB (no smoothing penalty for Firth adjustment)
            let btb = b_weighted.t().dot(&b_weighted);
            let mut h_fisher = btb;
            // Conditional regularization for stability without changing the objective form
            ensure_positive_definite_joint(&mut h_fisher);
            
            // Cholesky decomposition
            use crate::calibrate::faer_ndarray::FaerCholesky;
            let chol = match h_fisher.cholesky(faer::Side::Lower) {
                Ok(c) => c,
                Err(_) => {
                    // Fall back to standard IRLS if Firth fails
                    return self.compute_deviance(&self.compute_eta_full(u, b_wiggle));
                }
            };
            
            // Compute hat diagonal and Firth-adjusted z
            let mut z_adj = z_glm.clone();
            for i in 0..n {
                let mi = mu[i].clamp(1e-8, 1.0 - 1e-8);
                
                // h_ii = ||L^{-1} (b_i * sqrt(w_i))||^2
                let b_row: Array1<f64> = (0..p).map(|j| b_weighted[[i, j]]).collect();
                
                // Solve L * v = b_row using Cholesky factor
                let solved = chol.solve_mat(&b_row.insert_axis(ndarray::Axis(1))).column(0).to_owned();
                let h_ii: f64 = solved.iter().map(|x| x * x).sum();
                
                let wi = weights[i];
                if wi > 0.0 {
                    // Firth correction to working response
                    // This biases coefficients toward zero when separation threatens
                    let firth_adj = h_ii * (0.5 - mi) / wi;
                    z_adj[i] += firth_adj;
                }
            }
            z_adj
        } else {
            z_glm
        };
        
        // Adjust working response: solve for wiggle coefficient θ where
        // η = u + B_wiggle · θ
        // So target for θ is: z_adjusted = z_firth - u
        let z_adjusted: Array1<f64> = &z_firth - u;
        
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
        
        // Use integrated likelihood for final deviance if SE available
        if let (LinkFunction::Logit, Some(se)) = (&self.link, &self.covariate_se) {
                crate::calibrate::pirls::update_glm_vectors_integrated(
                    &self.quad_ctx,
                self.y,
                &eta_updated,
                se.view(),
                self.weights,
                &mut mu_updated,
                &mut weights_updated,
                &mut z_updated,
            );
        } else {
            crate::calibrate::pirls::update_glm_vectors(
                self.y,
                &eta_updated,
                self.link.clone(),
                self.weights,
                &mut mu_updated,
                &mut weights_updated,
                &mut z_updated,
            );
        }
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
        
        // Use integrated likelihood if SE available (Logit only)
        if let (LinkFunction::Logit, Some(se)) = (&self.link, &self.covariate_se) {
                crate::calibrate::pirls::update_glm_vectors_integrated(
                    &self.quad_ctx,
                self.y,
                &eta,
                se.view(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        } else {
            crate::calibrate::pirls::update_glm_vectors(
                self.y,
                &eta,
                self.link.clone(),
                self.weights,
                &mut mu,
                &mut weights,
                &mut z_glm,
            );
        }
        
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
            let g = g_prime[i];
            let g_safe = if g.abs() < 1e-8 {
                if g >= 0.0 { 1e-8 } else { -1e-8 }
            } else {
                g
            };
            w_eff[i] = weights[i] * g_safe * g_safe;
            z_eff[i] = z_beta[i] / g_safe;
        }
        
        // Build penalty for base block: S_base = Σ λ_k S_k
        let mut penalty = Array2::<f64>::zeros((p, p));
        for (idx, s_k) in self.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if s_k.nrows() == p && s_k.ncols() == p && lambda_k > 0.0 {
                penalty.scaled_add(lambda_k, s_k);
            }
        }
        
        // When Firth is enabled for logit, compute Firth correction for base block
        // This uses the effective design X_eff = sqrt(w_eff) * X and the base penalty
        if self.firth_bias_reduction && matches!(self.link, LinkFunction::Logit) {
            // Build weighted design: X_w = sqrt(w_eff) * X
            let mut x_weighted = self.x_base.to_owned();
            for i in 0..n {
                let sqrt_w = w_eff[i].max(0.0).sqrt();
                for j in 0..p {
                    x_weighted[[i, j]] *= sqrt_w;
                }
            }
            
            // Build Fisher information: H = X'W_effX (no smoothing penalty for Firth adjustment)
            let xtx = x_weighted.t().dot(&x_weighted);
            let mut h_fisher = xtx;
            ensure_positive_definite_joint(&mut h_fisher);
            
            // Cholesky decomposition
            use crate::calibrate::faer_ndarray::FaerCholesky;
            if let Ok(chol) = h_fisher.cholesky(faer::Side::Lower) {
                // Compute hat diagonal and apply Firth adjustment to z_eff
                for i in 0..n {
                    let mi = mu[i].clamp(1e-8, 1.0 - 1e-8);
                    
                    // h_ii = ||L^{-1}(x_i * sqrt(w_eff_i))||²
                    let x_row: Array1<f64> = (0..p).map(|j| x_weighted[[i, j]]).collect();
                    let solved = chol.solve_mat(&x_row.insert_axis(ndarray::Axis(1))).column(0).to_owned();
                    let h_ii: f64 = solved.iter().map(|x| x * x).sum();
                    
                    let wi = w_eff[i];
                    if wi > 0.0 {
                        // Firth correction to working response
                        let firth_adj = h_ii * (0.5 - mi) / wi;
                        z_eff[i] += firth_adj;
                    }
                }
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
        self.last_eta = Some(eta_updated.clone());
        let mut mu_updated = Array1::<f64>::zeros(n);
        let mut weights_updated = Array1::<f64>::zeros(n);
        let mut z_updated = Array1::<f64>::zeros(n);
        
        // Use integrated likelihood for final deviance if SE available
        if let (LinkFunction::Logit, Some(se)) = (&self.link, &self.covariate_se) {
            crate::calibrate::pirls::update_glm_vectors_integrated(
                &self.quad_ctx,
                self.y,
                &eta_updated,
                se.view(),
                self.weights,
                &mut mu_updated,
                &mut weights_updated,
                &mut z_updated,
            );
        } else {
            crate::calibrate::pirls::update_glm_vectors(
                self.y,
                &eta_updated,
                self.link.clone(),
                self.weights,
                &mut mu_updated,
                &mut weights_updated,
                &mut z_updated,
            );
        }
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
    let quad_ctx = QuadratureContext::new();
    let mut state = JointModelState::new(
        y, weights, x_base, s_base, layout_base, link, config, quad_ctx
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
        let b_wiggle = state.build_link_basis(&u)
            .map_err(|e| EstimationError::InvalidSpecification(e))?;
        
        // Update link coefficients (θ) via IRLS with u as OFFSET
        let deviance_after_g = state.irls_link_step(&b_wiggle, &u, lambda_link);
        
        // Step B: Given g, update β using g'(u)*X design
        // Get knot range for derivative computation
        // Compute g'(u) for chain rule
        let g_prime = compute_link_derivative_from_state(&state, &u, &b_wiggle);
        
        // Update β with damping (Gauss-Newton with offset)
        state.irls_base_step(&b_wiggle, &g_prime, &lambda_base, damping);
        
        // Rebuild basis with updated β before computing deviance for convergence check
        // This ensures we check convergence on the actual model state, not a stale version
        let u_new = state.base_linear_predictor();
        let b_new = state.build_link_basis(&u_new)
            .map_err(|e| EstimationError::InvalidSpecification(e))?;
        let deviance = state.compute_deviance(&state.compute_eta_full(&u_new, &b_new));
        
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
    
    // Rebuild basis with final β and refit θ to ensure consistency
    let u_final = state.base_linear_predictor();
    let b_final = state.build_link_basis(&u_final)
        .map_err(|e| EstimationError::InvalidSpecification(e))?;
    
    // θ is now potentially in wrong coordinates - refit it one more time
    state.irls_link_step(&b_final, &u_final, lambda_link);
    
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
        link_transform: state.link_transform.clone().unwrap_or_else(|| Array2::eye(state.n_constrained_basis)),
        degree: state.degree,
        link: state.link.clone(),
        s_link_constrained: state.s_link_constrained.clone().unwrap_or_else(|| Array2::zeros((state.n_constrained_basis, state.n_constrained_basis))),
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
    eval_count: RefCell<usize>,
}

struct JointRemlSnapshot {
    beta_base: Array1<f64>,
    beta_link: Array1<f64>,
    rho: Array1<f64>,
    knot_range: Option<(f64, f64)>,
    knot_vector: Option<Array1<f64>>,
    link_transform: Option<Array2<f64>>,
    s_link_constrained: Option<Array2<f64>>,
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
        covariate_se: Option<Array1<f64>>,
        quad_ctx: QuadratureContext,
    ) -> Self {
        let mut state =
            JointModelState::new(y, weights, x_base, s_base, layout_base, link, config, quad_ctx);
        // Set covariate_se for uncertainty-aware IRLS
        if let Some(se) = covariate_se {
            state = state.with_covariate_se(se);
        }
        let u0 = state.base_linear_predictor();
        if let Err(e) = state.build_link_basis(&u0) {
            eprintln!("[JOINT] Warning during initialization: {e}");
        }
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
            eval_count: RefCell::new(0),
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
            let b_wiggle = state.build_link_basis(&u)
                .map_err(|e| EstimationError::InvalidSpecification(e))?;
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
    
    /// Compute LAML at the converged solution.
    /// Note: for nonlinear g(u), this uses a Gauss-Newton Hessian approximation.
    fn compute_laml_at_convergence(&self, state: &JointModelState, lambda_base: &Array1<f64>, lambda_link: f64) -> (f64, Option<f64>) {
        let n = state.n_obs();
        let u = state.base_linear_predictor();
        let b_wiggle = state.build_link_basis_from_state(&u);
        
        // Compute eta = u + B_wiggle * theta
        let eta = state.compute_eta_full(&u, &b_wiggle);
        
        // Compute mu/weights/residuals at convergence
        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut residual = Array1::<f64>::zeros(n);
        match (&state.link, &state.covariate_se) {
            (LinkFunction::Logit, Some(se)) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-700.0, 700.0);
                    let se_i = se[i].max(0.0);
                    let (mu_i, dmu_deta) =
                        crate::calibrate::quadrature::logit_posterior_mean_with_deriv(
                            &state.quad_ctx,
                            e,
                            se_i,
                        );
                    let mu_c = mu_i.clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_c;
                    let var = (mu_c * (1.0 - mu_c)).max(PROB_EPS);
                    let dmu_sq = dmu_deta * dmu_deta;
                    let w = (dmu_sq / var).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * w;
                    let denom = dmu_deta.abs().max(MIN_DMU);
                    residual[i] = weights[i] * (mu_c - state.y[i]) / denom;
                }
            }
            (LinkFunction::Logit, None) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-700.0, 700.0);
                    let mu_i = (1.0 / (1.0 + (-e).exp())).clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_i;
                    let dmu = (mu_i * (1.0 - mu_i)).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * dmu;
                    let denom = dmu.max(MIN_DMU);
                    residual[i] = weights[i] * (mu_i - state.y[i]) / denom;
                }
            }
            (LinkFunction::Identity, _) => {
                for i in 0..n {
                    mu[i] = eta[i];
                    weights[i] = state.weights[i];
                    residual[i] = weights[i] * (mu[i] - state.y[i]);
                }
            }
        }
        let deviance = state.compute_deviance(&mu);
        
        // Build joint Jacobian blocks and penalized Hessian via Schur complement
        let p_base = state.x_base.ncols();
        let p_link = b_wiggle.ncols();
        
        let (g_prime, g_second, b_prime_u) = compute_link_derivative_terms_from_state(state, &u);
        
        // A = X' diag(W * g'^2 + r * g'') X + S_base
        let mut w_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            w_eff[i] = weights[i] * g_prime[i] * g_prime[i];
        }
        let mut x_weighted = state.x_base.to_owned();
        for i in 0..n {
            let scale = w_eff[i].max(0.0).sqrt();
            for j in 0..p_base {
                x_weighted[[i, j]] *= scale;
            }
        }

        let mut a_mat = crate::calibrate::faer_ndarray::fast_ata(&x_weighted);
        let mut x_scaled = state.x_base.to_owned();
        for i in 0..n {
            let scale = residual[i] * g_second[i];
            for j in 0..p_base {
                x_scaled[[i, j]] *= scale;
            }
        }
        let a_resid = crate::calibrate::faer_ndarray::fast_atb(&state.x_base, &x_scaled);
        a_mat += &a_resid;
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                a_mat.scaled_add(lambda_k, s_k);
            }
        }
        ensure_positive_definite_joint(&mut a_mat);
        
        // C = X' diag(W * g') B + X' diag(r) B'
        let mut wb = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i] * g_prime[i];
            for j in 0..p_link {
                wb[[i, j]] *= scale;
            }
        }
        let mut wb_resid = b_prime_u.clone();
        for i in 0..n {
            let scale = residual[i];
            for j in 0..p_link {
                wb_resid[[i, j]] *= scale;
            }
        }
        wb += &wb_resid;
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
        ensure_positive_definite_joint(&mut d_mat);
        
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
                        // Schur complement is not positive definite - model is ill-conditioned
                        // Return infinity cost to steer optimizer away
                        use crate::calibrate::faer_ndarray::FaerEigh;
                        let (eigs, _) = match schur.clone().eigh(Side::Lower) {
                            Ok(result) => result,
                            Err(_) => return (f64::INFINITY, None),
                        };
                        let min_eig = eigs.iter().cloned().fold(f64::INFINITY, f64::min);
                        if min_eig <= 0.0 {
                            eprintln!("[LAML] Schur complement has non-positive eigenvalue: {min_eig}");
                            return (f64::INFINITY, None);
                        }
                        eigs.iter().map(|&ev| ev.ln()).sum()
                    }
                };
                log_det + log_det_schur
            }
            Err(_) => {
                // Joint Hessian is not positive definite - cannot compute valid LAML
                // Return infinity cost to steer optimizer away
                eprintln!("[LAML] Joint Hessian Cholesky failed - returning infinity cost");
                return (f64::INFINITY, None);
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
            u_truncated: Array2::zeros((p_base, p_base)),  // All modes truncated in fallback
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
    
    ensure_positive_definite_joint(&mut a_mat);
    ensure_positive_definite_joint(&mut d_mat);
    
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
    fn compute_gradient_fd(&self, rho: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
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
            if !cost_plus.is_finite() {
                snapshot.restore(self);
                return Err(EstimationError::RemlOptimizationFailed(
                    "Non-finite LAML in +h finite-difference step.".to_string(),
                ));
            }
            
            snapshot.restore(self);
            // Backward step  
            let mut rho_minus = rho.clone();
            rho_minus[k] -= h;
            let cost_minus = self.compute_cost(&rho_minus)?;
            if !cost_minus.is_finite() {
                snapshot.restore(self);
                return Err(EstimationError::RemlOptimizationFailed(
                    "Non-finite LAML in -h finite-difference step.".to_string(),
                ));
            }
            
            // Central difference
            grad[k] = (cost_plus - cost_minus) / (2.0 * h);
        }
        
        snapshot.restore(self);

        if grad.iter().any(|v| !v.is_finite()) {
            return Err(EstimationError::RemlOptimizationFailed(
                "Non-finite gradient from finite-difference LAML.".to_string(),
            ));
        }
        
        Ok(grad)
    }

    /// Compute analytic gradient of LAML w.r.t. ρ using a Gauss-Newton Hessian
    /// and explicit differentiation of weights and constrained basis.
    fn compute_gradient_analytic(
        &self,
        rho: &Array1<f64>,
    ) -> Result<(Array1<f64>, bool), EstimationError> {
        let mut state = self.state.borrow_mut();
        let n_base = state.s_base.len();

        if rho.len() != n_base + 1 {
            return Err(EstimationError::LayoutError(
                "rho length does not match joint penalty count".to_string(),
            ));
        }

        if matches!(state.link, LinkFunction::Logit) && state.covariate_se.is_some() {
            return Err(EstimationError::RemlOptimizationFailed(
                "analytic joint gradient not implemented for integrated logit weights".to_string(),
            ));
        }
        let firth_active = state.firth_bias_reduction && matches!(state.link, LinkFunction::Logit);

        // Set ρ and warm-start from cached coefficients
        state.set_rho(rho.clone());
        state.beta_base = self.cached_beta_base.borrow().clone();
        state.beta_link = self.cached_beta_link.borrow().clone();

        let mut lambda_base = Array1::<f64>::zeros(n_base);
        for i in 0..n_base {
            lambda_base[i] = rho.get(i).map(|r| r.exp()).unwrap_or(1.0);
        }
        let lambda_link = rho.get(n_base).map(|r| r.exp()).unwrap_or(1.0);

        // Run inner alternating to convergence (same as compute_cost).
        let mut prev_deviance = f64::INFINITY;
        let mut converged = false;
        for i in 0..self.config.max_backfit_iter {
            let progress = (i as f64) / (self.config.max_backfit_iter as f64);
            let damping = 0.5 + progress * 0.5;

            let u = state.base_linear_predictor();
            let b_wiggle = state.build_link_basis(&u)
                .map_err(|e| EstimationError::InvalidSpecification(e))?;
            state.irls_link_step(&b_wiggle, &u, lambda_link);

            let g_prime = compute_link_derivative_from_state(&state, &u, &b_wiggle);
            let deviance = state.irls_base_step(&b_wiggle, &g_prime, &lambda_base, damping);

            let delta = (prev_deviance - deviance).abs() / (deviance.abs() + 1.0);
            if delta < self.config.backfit_tol {
                converged = true;
                break;
            }
            prev_deviance = deviance;
        }

        // Cache converged coefficients for warm-start
        *self.cached_beta_base.borrow_mut() = state.beta_base.clone();
        *self.cached_beta_link.borrow_mut() = state.beta_link.clone();
        *self.last_converged.borrow_mut() = converged;

        let n = state.n_obs();
        let u = state.base_linear_predictor();
        let b_wiggle = state.build_link_basis_from_state(&u);
        let eta = state.compute_eta_full(&u, &b_wiggle);

        let mut mu = Array1::<f64>::zeros(n);
        let mut weights = Array1::<f64>::zeros(n);
        let mut residual = Array1::<f64>::zeros(n);
        match (&state.link, &state.covariate_se) {
            (LinkFunction::Logit, None) => {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                const MIN_DMU: f64 = 1e-6;
                for i in 0..n {
                    let e = eta[i].clamp(-700.0, 700.0);
                    let mu_i = (1.0 / (1.0 + (-e).exp())).clamp(PROB_EPS, 1.0 - PROB_EPS);
                    mu[i] = mu_i;
                    let w = (mu_i * (1.0 - mu_i)).max(MIN_WEIGHT);
                    weights[i] = state.weights[i] * w;
                    let denom = w.max(MIN_DMU);
                    residual[i] = weights[i] * (mu_i - state.y[i]) / denom;
                }
            }
            (LinkFunction::Identity, _) => {
                for i in 0..n {
                    mu[i] = eta[i];
                    weights[i] = state.weights[i];
                    residual[i] = weights[i] * (mu[i] - state.y[i]);
                }
            }
            _ => {
                return Err(EstimationError::RemlOptimizationFailed(
                    "analytic joint gradient unsupported for this link".to_string(),
                ));
            }
        }

        let p_base = state.x_base.ncols();
        let p_link = b_wiggle.ncols();
        let p_total = p_base + p_link;

        let (g_prime, g_second, b_prime_u) = compute_link_derivative_terms_from_state(&state, &u);

        // Build Gauss-Newton blocks A, C, D (same as cost path).
        let mut w_eff = Array1::<f64>::zeros(n);
        for i in 0..n {
            w_eff[i] = weights[i] * g_prime[i] * g_prime[i];
        }
        let mut x_weighted = state.x_base.to_owned();
        for i in 0..n {
            let scale = w_eff[i].max(0.0).sqrt();
            for j in 0..p_base {
                x_weighted[[i, j]] *= scale;
            }
        }
        let mut a_mat = crate::calibrate::faer_ndarray::fast_ata(&x_weighted);
        let mut x_scaled = state.x_base.to_owned();
        for i in 0..n {
            let scale = residual[i] * g_second[i];
            for j in 0..p_base {
                x_scaled[[i, j]] *= scale;
            }
        }
        let a_resid = crate::calibrate::faer_ndarray::fast_atb(&state.x_base, &x_scaled);
        a_mat += &a_resid;
        for (idx, s_k) in state.s_base.iter().enumerate() {
            let lambda_k = lambda_base.get(idx).cloned().unwrap_or(0.0);
            if lambda_k > 0.0 && s_k.nrows() == p_base && s_k.ncols() == p_base {
                a_mat.scaled_add(lambda_k, s_k);
            }
        }
        ensure_positive_definite_joint(&mut a_mat);

        let mut wb = b_wiggle.clone();
        for i in 0..n {
            let scale = weights[i] * g_prime[i];
            for j in 0..p_link {
                wb[[i, j]] *= scale;
            }
        }
        let mut wb_resid = b_prime_u.clone();
        for i in 0..n {
            let scale = residual[i];
            for j in 0..p_link {
                wb_resid[[i, j]] *= scale;
            }
        }
        wb += &wb_resid;
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
        ensure_positive_definite_joint(&mut d_mat);

        // Assemble full Hessian H from blocks.
        let mut h_mat = Array2::<f64>::zeros((p_total, p_total));
        h_mat.slice_mut(s![..p_base, ..p_base]).assign(&a_mat);
        h_mat.slice_mut(s![..p_base, p_base..]).assign(&c_mat);
        h_mat.slice_mut(s![p_base.., ..p_base]).assign(&c_mat.t());
        h_mat.slice_mut(s![p_base.., p_base..]).assign(&d_mat);

        use crate::calibrate::faer_ndarray::FaerCholesky;
        use faer::Side;
        let h_chol = h_mat.cholesky(Side::Lower).map_err(|_| {
            EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            }
        })?;

        // Build Jacobian J = [diag(g') X | B_wiggle]
        let mut j_mat = Array2::<f64>::zeros((n, p_total));
        for i in 0..n {
            let gp = g_prime[i];
            for j in 0..p_base {
                j_mat[[i, j]] = gp * state.x_base[[i, j]];
            }
            for j in 0..p_link {
                j_mat[[i, p_base + j]] = b_wiggle[[i, j]];
            }
        }

        // Precompute K = H^{-1} J^T and J H^{-1} J^T diagonal for trace terms.
        let j_t = j_mat.t().to_owned();
        let k_mat = h_chol.solve_mat(&j_t);
        let mut k_w = k_mat.clone();
        for i in 0..n {
            let w = weights[i];
            for j in 0..p_total {
                k_w[[j, i]] *= w;
            }
        }
        let mut diag_proj = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut acc = 0.0;
            for j in 0..p_total {
                acc += j_mat[[i, j]] * k_mat[[j, i]];
            }
            diag_proj[i] = acc;
        }

        // Precompute H^{-1}_{theta,theta} for penalty sensitivity trace.
        let mut h_inv_theta = Array2::<f64>::zeros((p_link, p_link));
        if p_link > 0 {
            for col in 0..p_link {
                let mut e = Array1::<f64>::zeros(p_total);
                e[p_base + col] = 1.0;
                let solved = h_chol.solve_vec(&e);
                for row in 0..p_link {
                    h_inv_theta[[row, col]] = solved[p_base + row];
                }
            }
        }

        // Firth adjoint matrix Q = H^{-1} V H^{-1}.
        //
        // Proof-style derivation (fixed-point, logit Firth):
        //   Let I = J^T W J be the Fisher information and define the Firth term:
        //     Phi = 0.5 * log|I|.
        //   The Firth-adjusted objective adds Phi to the log-likelihood, so the
        //   REML/LAML gradient gains an extra contribution from d/d rho_k Phi.
        //
        //   Using Jacobi's identity:
        //     d Phi = 0.5 * tr(I^{-1} dI).
        //   The implicit dependence of I on the parameters introduces the adjoint
        //   (a "sandwich") term in the trace of dH/d rho_k. We encode this by
        //   augmenting H^{-1} with Q so that:
        //     tr(H^{-1} dotH_std)  ->  tr((H^{-1} + Q) dotH_std),
        //   where:
        //     Q = H^{-1} V H^{-1}.
        //
        //   For logit Jeffreys prior, V is the sensitivity of the Firth penalty
        //   to hat values and reduces to:
        //     V = J^T diag(0.5 - mu) J.
        //
        //   This choice yields:
        //     tr(Q * dotH_std) = tr(H^{-1} V H^{-1} dotH_std),
        //   which is the exact adjoint correction for the Firth term, matching
        //   the fixed-point objective defined by the Gauss-Newton Hessian H.
        let mut q_firth = Array2::<f64>::zeros((p_total, p_total));
        if firth_active {
            let mut j_weighted = j_mat.clone();
            for i in 0..n {
                let nu = 0.5 - mu[i];
                for j in 0..p_total {
                    j_weighted[[i, j]] *= nu;
                }
            }
            // V = J^T diag(0.5 - mu) J (implemented as J^T * (diag(nu) J)).
            let v_mat = crate::calibrate::faer_ndarray::fast_atb(&j_mat, &j_weighted);
            let y = h_chol.solve_mat(&v_mat);
            let y_t = y.t().to_owned();
            let q_t = h_chol.solve_mat(&y_t);
            q_firth = q_t.t().to_owned();
        }

        // Prepare basis derivatives for constraint sensitivity.
        let Some(knot_vector) = state.knot_vector.as_ref() else {
            return Err(EstimationError::RemlOptimizationFailed(
                "missing knot vector for joint analytic gradient".to_string(),
            ));
        };
        let Some(link_transform) = state.link_transform.as_ref() else {
            return Err(EstimationError::RemlOptimizationFailed(
                "missing link transform for joint analytic gradient".to_string(),
            ));
        };
        if link_transform.ncols() != p_link {
            return Err(EstimationError::RemlOptimizationFailed(
                "link transform dimension mismatch".to_string(),
            ));
        }

        let (z, range_width) = state.standardized_z(&u);
        let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
        if n_raw == 0 {
            return Err(EstimationError::RemlOptimizationFailed(
                "insufficient basis size for analytic gradient".to_string(),
            ));
        }

        use crate::calibrate::basis::{create_basis, BasisOptions, KnotSource, Dense};
        let (b_raw_arc, _) = create_basis::<Dense>(
            z.view(),
            KnotSource::Provided(knot_vector.view()),
            state.degree,
            BasisOptions::value(),
        )
        .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;
        let (b_prime_arc, _) = create_basis::<Dense>(
            z.view(),
            KnotSource::Provided(knot_vector.view()),
            state.degree,
            BasisOptions::first_derivative(),
        )
        .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;

        let b_raw = b_raw_arc.as_ref();
        let b_prime = b_prime_arc.as_ref();

        // Build constraint matrix C (n x 2) and weighted constraints.
        let mut constraint = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            constraint[[i, 0]] = 1.0;
            constraint[[i, 1]] = z[i];
        }

        let mut weighted_constraints = constraint.clone();
        for i in 0..n {
            let w = weights[i];
            weighted_constraints[[i, 0]] *= w;
            weighted_constraints[[i, 1]] *= w;
        }
        let constraint_cross =
            crate::calibrate::faer_ndarray::fast_atb(b_raw, &weighted_constraints); // k x 2
        let m = constraint_cross.t().to_owned(); // 2 x k

        let mmt = m.dot(&m.t()); // 2 x 2
        let det = mmt[(0, 0)] * mmt[(1, 1)] - mmt[(0, 1)] * mmt[(1, 0)];
        if det.abs() < 1e-14 {
            return Err(EstimationError::RemlOptimizationFailed(
                "constraint matrix nearly singular in analytic gradient".to_string(),
            ));
        }
        let inv_det = 1.0 / det;
        let mut mmt_inv = Array2::<f64>::zeros((2, 2));
        mmt_inv[(0, 0)] = mmt[(1, 1)] * inv_det;
        mmt_inv[(1, 1)] = mmt[(0, 0)] * inv_det;
        mmt_inv[(0, 1)] = -mmt[(0, 1)] * inv_det;
        mmt_inv[(1, 0)] = -mmt[(1, 0)] * inv_det;
        let m_pinv = m.t().dot(&mmt_inv); // k x 2

        // Raw penalty for link block and its projection (constant for this rho).
        let s_raw = if p_link > 0 {
            create_difference_penalty_matrix(n_raw, 2)
                .unwrap_or_else(|_| Array2::zeros((n_raw, n_raw)))
        } else {
            Array2::zeros((0, 0))
        };
        let v_pen = if p_link > 0 && s_raw.nrows() == n_raw {
            crate::calibrate::faer_ndarray::fast_ab(&s_raw, link_transform)
        } else {
            Array2::zeros((n_raw, p_link))
        };

        let mut grad = Array1::<f64>::zeros(rho.len());
        let mut clamp_z_count = 0usize;
        let mut clamp_mu_count = 0usize;
        for i in 0..n {
            if z[i] <= 1e-8 || z[i] >= 1.0 - 1e-8 {
                clamp_z_count += 1;
            }
            if matches!(state.link, LinkFunction::Logit)
                && (mu[i] <= 1e-8 || mu[i] >= 1.0 - 1e-8)
            {
                clamp_mu_count += 1;
            }
        }
        let clamp_z_frac = clamp_z_count as f64 / n.max(1) as f64;
        let clamp_mu_frac = clamp_mu_count as f64 / n.max(1) as f64;
        let det_abs = det.abs();
        let audit_needed = !converged
            || clamp_z_frac > 0.05
            || clamp_mu_frac > 0.05
            || det_abs < 1e-10;
        // Penalty det derivative for base penalties.
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
            u_truncated: Array2::zeros((p_base, p_base)),
        });

        let link_det1 = if p_link > 0 && link_penalty.nrows() == p_link {
            use crate::calibrate::faer_ndarray::FaerEigh;
            let (eigs, _) = link_penalty
                .clone()
                .eigh(Side::Lower)
                .unwrap_or_else(|_| (Array1::zeros(p_link), Array2::eye(p_link)));
            let max_eig = eigs.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
            let tol = if max_eig > 0.0 { max_eig * 1e-12 } else { 1e-12 };
            let rank = eigs.iter().filter(|&&ev| ev > tol).count() as f64;
            rank
        } else {
            0.0
        };

        let mut dot_u = Array1::<f64>::zeros(n);
        let mut dot_z = Array1::<f64>::zeros(n);
        let mut dot_eta = Array1::<f64>::zeros(n);
        let mut w_prime = Array1::<f64>::zeros(n);
        let mut w_dot = Array1::<f64>::zeros(n);
        let mut b_dot = Array2::<f64>::zeros((n, n_raw));
        let mut c_dot = Array2::<f64>::zeros((n, 2));
        let mut weighted_c_dot = Array2::<f64>::zeros((n, 2));
        let mut weighted_b = Array2::<f64>::zeros((n, n_raw));
        let mut weighted_b_dot = Array2::<f64>::zeros((n, n_raw));
        let mut dot_j_theta = Array2::<f64>::zeros((n, p_link));
        let mut dot_j_beta = Array2::<f64>::zeros((n, p_base));
        let mut dot_j = Array2::<f64>::zeros((n, p_total));

        for k in 0..rho.len() {
            let is_link = k == n_base;
            let lambda_k = if is_link { lambda_link } else { lambda_base[k] };

            let mut rhs = Array1::<f64>::zeros(p_total);
            if is_link {
                if p_link > 0 && link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
                    let sb = link_penalty.dot(&state.beta_link);
                    for i in 0..p_link {
                        rhs[p_base + i] = -lambda_k * sb[i];
                    }
                }
            } else if let Some(s_k) = state.s_base.get(k) {
                if s_k.nrows() == p_base && s_k.ncols() == p_base {
                    let sb = s_k.dot(&state.beta_base);
                    for i in 0..p_base {
                        rhs[i] = -lambda_k * sb[i];
                    }
                }
            }

            let delta = h_chol.solve_vec(&rhs);
            let delta_beta = delta.slice(s![..p_base]).to_owned();
            let delta_theta = delta.slice(s![p_base..]).to_owned();

            dot_u.assign(&state.x_base.dot(&delta_beta));

            // dot_z with clamp mask
            let inv_rw = 1.0 / range_width;
            for i in 0..n {
                let zi = z[i];
                if zi > 0.0 && zi < 1.0 {
                    dot_z[i] = dot_u[i] * inv_rw;
                } else {
                    dot_z[i] = 0.0;
                }
            }

            dot_eta.assign(&j_mat.dot(&delta));

            // w' for logit (clamped)
            if matches!(state.link, LinkFunction::Logit) {
                const PROB_EPS: f64 = 1e-8;
                const MIN_WEIGHT: f64 = 1e-12;
                for i in 0..n {
                    let mu_i = mu[i];
                    let w_base = mu_i * (1.0 - mu_i);
                    if mu_i <= PROB_EPS || mu_i >= 1.0 - PROB_EPS || w_base < MIN_WEIGHT {
                        w_prime[i] = 0.0;
                    } else {
                        w_prime[i] = state.weights[i] * w_base * (1.0 - 2.0 * mu_i);
                    }
                }
            } else {
                w_prime.fill(0.0);
            }

            for i in 0..n {
                w_dot[i] = w_prime[i] * dot_eta[i];
            }

            // B_dot = B' * diag(dot_z)
            b_dot.assign(b_prime);
            for i in 0..n {
                let scale = dot_z[i];
                if scale != 1.0 {
                    for j in 0..n_raw {
                        b_dot[[i, j]] *= scale;
                    }
                }
            }

            // M_dot = C_dot^T W B + C^T W_dot B + C^T W B_dot
            c_dot.fill(0.0);
            for i in 0..n {
                c_dot[[i, 1]] = dot_z[i];
            }
            weighted_c_dot.assign(&c_dot);
            for i in 0..n {
                let w = weights[i];
                weighted_c_dot[[i, 0]] *= w;
                weighted_c_dot[[i, 1]] *= w;
            }
            weighted_b.assign(b_raw);
            for i in 0..n {
                let w = w_dot[i];
                for j in 0..n_raw {
                    weighted_b[[i, j]] *= w;
                }
            }
            weighted_b_dot.assign(&b_dot);
            for i in 0..n {
                let w = weights[i];
                for j in 0..n_raw {
                    weighted_b_dot[[i, j]] *= w;
                }
            }
            let mut m_dot = Array2::<f64>::zeros((2, n_raw));
            {
                use crate::calibrate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                use faer::linalg::matmul::matmul;
                use faer::{Accum, Par, get_global_parallelism};

                let par = if n < 128 || n_raw < 128 {
                    Par::Seq
                } else {
                    get_global_parallelism()
                };

                let mut m_dot_view = array2_to_mat_mut(&mut m_dot);
                
                // Bind FaerArrayView temporaries to local variables to extend their lifetime
                let wc_wrapper = FaerArrayView::new(&weighted_c_dot);
                let wc_view = wc_wrapper.as_ref();
                let b_wrapper = FaerArrayView::new(b_raw);
                let b_view = b_wrapper.as_ref();
                let c_wrapper = FaerArrayView::new(&constraint);
                let c_view = c_wrapper.as_ref();
                let wb_wrapper = FaerArrayView::new(&weighted_b);
                let wb_view = wb_wrapper.as_ref();
                let wbdot_wrapper = FaerArrayView::new(&weighted_b_dot);
                let wbdot_view = wbdot_wrapper.as_ref();

                matmul(
                    m_dot_view.as_mut(),
                    Accum::Replace,
                    wc_view.transpose(),
                    b_view,
                    1.0,
                    par,
                );
                matmul(
                    m_dot_view.as_mut(),
                    Accum::Add,
                    c_view.transpose(),
                    wb_view,
                    1.0,
                    par,
                );
                matmul(
                    m_dot_view.as_mut(),
                    Accum::Add,
                    c_view.transpose(),
                    wbdot_view,
                    1.0,
                    par,
                );
            }
            let z_dot = -m_pinv.dot(&crate::calibrate::faer_ndarray::fast_ab(
                &m_dot,
                link_transform,
            ));

            dot_j_theta.assign(
                &crate::calibrate::faer_ndarray::fast_ab(&b_dot, link_transform),
            );
            dot_j_theta += &crate::calibrate::faer_ndarray::fast_ab(b_raw, &z_dot);

            // dot_g_prime
            let mut dot_g_prime = Array1::<f64>::zeros(n);
            for i in 0..n {
                dot_g_prime[i] = g_second[i] * dot_u[i];
            }
            let b_prime_delta = b_prime_u.dot(&delta_theta);
            dot_g_prime += &b_prime_delta;

            let z_dot_theta = z_dot.dot(&state.beta_link);
            if z_dot_theta.len() == n_raw {
                let mut z_term = b_prime.dot(&z_dot_theta);
                for i in 0..n {
                    z_term[i] *= inv_rw;
                }
                dot_g_prime += &z_term;
            }

            // dot_J_beta = diag(dot_g_prime) X
            dot_j_beta.fill(0.0);
            for i in 0..n {
                let scale = dot_g_prime[i];
                for j in 0..p_base {
                    dot_j_beta[[i, j]] = scale * state.x_base[[i, j]];
                }
            }

            // dot_J = [dot_J_beta | dot_J_theta]
            dot_j.fill(0.0);
            dot_j.slice_mut(s![.., ..p_base]).assign(&dot_j_beta);
            dot_j.slice_mut(s![.., p_base..]).assign(&dot_j_theta);
            // Trace for likelihood curvature: 2 tr(K_w * dot_J) + tr(diag(J H^{-1} J^T) * W_dot)
            let mut trace = 0.0;
            let mut trace_k = 0.0;
            for i in 0..n {
                let mut acc = 0.0;
                for j in 0..p_total {
                    acc += k_w[[j, i]] * dot_j[[i, j]];
                }
                trace_k += acc;
                trace += diag_proj[i] * w_dot[i];
            }
            trace += 2.0 * trace_k;

            // Trace for lambda_k * S_k
            let mut s_k_full = Array2::<f64>::zeros((p_total, p_total));
            if is_link {
                s_k_full
                    .slice_mut(s![p_base.., p_base..])
                    .assign(&link_penalty);
            } else if let Some(s_k) = state.s_base.get(k) {
                if s_k.nrows() == p_base && s_k.ncols() == p_base {
                    s_k_full.slice_mut(s![..p_base, ..p_base]).assign(s_k);
                }
            }
            if p_total > 0 {
                let solved = h_chol.solve_mat(&s_k_full);
                let mut trace_lambda = 0.0;
                for i in 0..p_total {
                    trace_lambda += solved[[i, i]];
                }
                trace += lambda_k * trace_lambda;
            }

            // Penalty manifold sensitivity for the link block: dot(S_link) = Z_dot^T S_raw Z + Z^T S_raw Z_dot.
            if p_link > 0 && v_pen.nrows() == n_raw {
                let left = crate::calibrate::faer_ndarray::fast_atb(&z_dot, &v_pen);
                let right = crate::calibrate::faer_ndarray::fast_atb(&v_pen, &z_dot);
                let dot_s_link = left + right;
                let mut trace_penalty = 0.0;
                for i in 0..p_link {
                    for j in 0..p_link {
                        trace_penalty += h_inv_theta[[i, j]] * dot_s_link[[j, i]];
                    }
                }
                trace += lambda_link * trace_penalty;
            }

            // Firth adjoint correction: add tr(Q * dotH_std) for logit Firth.
            //
            // dotH_std is the directional derivative of the standard Hessian:
            //   dotH_std = dotJ^T W J + J^T W dotJ + J^T W_dot J + dotS_lambda
            // where dotS_lambda includes the penalty manifold sensitivity of the link block.
            if firth_active {
                // dotH_std = dotJ^T W J + J^T W dotJ + J^T W_dot J, with penalty sensitivity added to theta block.
                let mut dot_h_std = Array2::<f64>::zeros((p_total, p_total));
                {
                    use crate::calibrate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                    use faer::linalg::matmul::matmul;
                    use faer::{Accum, Par, get_global_parallelism};

                    let par = if n < 128 || p_total < 128 {
                        Par::Seq
                    } else {
                        get_global_parallelism()
                    };
                    let mut dot_h_view = array2_to_mat_mut(&mut dot_h_std);

                    let mut wj = j_mat.clone();
                    for i in 0..n {
                        let w = weights[i];
                        for j in 0..p_total {
                            wj[[i, j]] *= w;
                        }
                    }
                    let mut wdotj = j_mat.clone();
                    for i in 0..n {
                        let w = w_dot[i];
                        for j in 0..p_total {
                            wdotj[[i, j]] *= w;
                        }
                    }
                    let mut wdotj2 = dot_j.clone();
                    for i in 0..n {
                        let w = weights[i];
                        for j in 0..p_total {
                            wdotj2[[i, j]] *= w;
                        }
                    }

                    // Bind FaerArrayView temporaries to local variables to extend their lifetime
                    let dj_wrapper = FaerArrayView::new(&dot_j);
                    let dj_view = dj_wrapper.as_ref();
                    let wj_wrapper = FaerArrayView::new(&wj);
                    let wj_view = wj_wrapper.as_ref();
                    let j_wrapper = FaerArrayView::new(&j_mat);
                    let j_view = j_wrapper.as_ref();
                    let wdotj_wrapper = FaerArrayView::new(&wdotj);
                    let wdotj_view = wdotj_wrapper.as_ref();
                    let wdotj2_wrapper = FaerArrayView::new(&wdotj2);
                    let wdotj2_view = wdotj2_wrapper.as_ref();

                    matmul(
                        dot_h_view.as_mut(),
                        Accum::Replace,
                        dj_view.transpose(),
                        wj_view,
                        1.0,
                        par,
                    );
                    matmul(
                        dot_h_view.as_mut(),
                        Accum::Add,
                        j_view.transpose(),
                        wdotj2_view,
                        1.0,
                        par,
                    );
                    matmul(
                        dot_h_view.as_mut(),
                        Accum::Add,
                        j_view.transpose(),
                        wdotj_view,
                        1.0,
                        par,
                    );
                }

                // Add penalty manifold sensitivity into dotH_std theta block.
                if p_link > 0 && v_pen.nrows() == n_raw {
                    let left = crate::calibrate::faer_ndarray::fast_atb(&z_dot, &v_pen);
                    let right = crate::calibrate::faer_ndarray::fast_atb(&v_pen, &z_dot);
                    let dot_s_link = left + right;
                    let mut link_block = dot_h_std.slice_mut(s![p_base.., p_base..]);
                    link_block += &dot_s_link.mapv(|v| v * lambda_link);
                }

                // Accumulate tr(Q * dotH_std) = sum_ij Q_ij * dotH_std_ji.
                let mut trace_q = 0.0;
                for i in 0..p_total {
                    for j in 0..p_total {
                        trace_q += q_firth[[i, j]] * dot_h_std[[j, i]];
                    }
                }
                trace += trace_q;
            }


            let penalty_term = if is_link {
                if p_link > 0 && link_penalty.nrows() == p_link && link_penalty.ncols() == p_link {
                    let sb = link_penalty.dot(&state.beta_link);
                    0.5 * lambda_k * state.beta_link.dot(&sb)
                } else {
                    0.0
                }
            } else if let Some(s_k) = state.s_base.get(k) {
                if s_k.nrows() == p_base && s_k.ncols() == p_base {
                    let sb = s_k.dot(&state.beta_base);
                    0.5 * lambda_k * state.beta_base.dot(&sb)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let det_term = if is_link {
                0.5 * link_det1
            } else if k < base_reparam.det1.len() {
                0.5 * base_reparam.det1[k]
            } else {
                0.0
            };

            let grad_laml = penalty_term - det_term + 0.5 * trace;
            grad[k] = -grad_laml;
        }

        Ok((grad, audit_needed))
    }

    /// Compute gradient of LAML w.r.t. ρ using analytic path, with FD fallback.
    pub fn compute_gradient(&self, rho: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
        match self.compute_gradient_analytic(rho) {
            Ok((grad, audit_needed)) => {
                const GRAD_FD_REL_TOL: f64 = 1e-2;
                if audit_needed {
                    match self.compute_gradient_fd(rho) {
                        Ok(fd_grad) => {
                            let mut diff_norm = 0.0;
                            let mut fd_norm = 0.0;
                            for i in 0..fd_grad.len() {
                                let d = grad[i] - fd_grad[i];
                                diff_norm += d * d;
                                fd_norm += fd_grad[i] * fd_grad[i];
                            }
                            let rel = diff_norm.sqrt() / (fd_norm.sqrt() + 1.0);
                            if rel > GRAD_FD_REL_TOL {
                                return Err(EstimationError::RemlOptimizationFailed(format!(
                                    "Analytic/FD gradient mismatch (rel {:.3e}) in joint REML.",
                                    rel
                                )));
                            }
                        }
                        Err(err) => {
                            eprintln!("[JOINT][REML] FD audit failed: {err}");
                        }
                    }
                }
                Ok(grad)
            }
            Err(err) => {
                eprintln!("[JOINT][REML] Analytic gradient unavailable: {err}. Falling back to FD.");
                self.compute_gradient_fd(rho)
            }
        }
    }

    #[cfg(test)]
    pub fn compute_gradient_analytic_for_test(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        self.compute_gradient_analytic(rho).map(|(grad, _)| grad)
    }

    #[cfg(test)]
    pub fn compute_gradient_fd_for_test(
        &self,
        rho: &Array1<f64>,
    ) -> Result<Array1<f64>, EstimationError> {
        self.compute_gradient_fd(rho)
    }

    /// Combined cost and gradient for BFGS
    pub fn cost_and_grad(&self, rho: &Array1<f64>) -> (f64, Array1<f64>) {
        let eval_num = {
            let mut count = self.eval_count.borrow_mut();
            *count += 1;
            *count
        };
        let cost = match self.compute_cost(rho) {
            Ok(val) if val.is_finite() => val,
            Ok(_) => {
                eprintln!("[JOINT][REML] Non-finite cost; returning large penalty.");
                return (f64::INFINITY, Array1::from_elem(rho.len(), f64::NAN));
            }
            Err(err) => {
                eprintln!("[JOINT][REML] Cost evaluation failed: {err}");
                return (f64::INFINITY, Array1::from_elem(rho.len(), f64::NAN));
            }
        };
        let grad = match self.compute_gradient(rho) {
            Ok(grad) => grad,
            Err(err) => {
                eprintln!("[JOINT][REML] Gradient evaluation failed: {err}");
                Array1::from_elem(rho.len(), f64::NAN)
            }
        };
        let grad_norm = grad.dot(&grad).sqrt();
        visualizer::update(cost, grad_norm, "optimizing", eval_num as f64, "eval");
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
            link_transform: state.link_transform.unwrap_or_else(|| Array2::eye(state.n_constrained_basis)),
            degree: state.degree,
            link: state.link.clone(),
            s_link_constrained: state.s_link_constrained.unwrap_or_else(|| Array2::zeros((state.n_constrained_basis, state.n_constrained_basis))),
            base_model: None,
        }
    }
}

/// Fit joint model with proper REML-based lambda selection via BFGS
/// 
/// Uses Laplace approximate marginal likelihood (LAML) with numerical gradient.
/// For nonlinear g(u), the Hessian is Gauss-Newton (approximate).
pub fn fit_joint_model_with_reml<'a>(
    y: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    x_base: ArrayView2<'a, f64>,
    s_base: Vec<Array2<f64>>,
    layout_base: ModelLayout,
    link: LinkFunction,
    config: &JointModelConfig,
    covariate_se: Option<Array1<f64>>,
) -> Result<JointModelResult, EstimationError> {
    visualizer::set_stage("joint", "initializing");
    let quad_ctx = QuadratureContext::new();
    
    // Create REML state
    let reml_state = JointRemlState::new(
        y, weights, x_base, s_base, layout_base, link, config, covariate_se, quad_ctx
    );
    
    let n_base = reml_state.state.borrow().s_base.len();
    let heuristic_lambda = {
        let state = reml_state.state.borrow();
        state
            .knot_vector
            .as_ref()
            .map(|knots| baseline_lambda_seed(knots, state.degree, 2))
    };
    let heuristic_lambdas = heuristic_lambda.map(|lambda| vec![lambda]);
    let seed_config = SeedConfig {
        strategy: SeedStrategy::Exhaustive,
        bounds: (-12.0, 12.0),
    };
    let seed_candidates = generate_rho_candidates(
        n_base + 1,
        heuristic_lambdas.as_deref(),
        &seed_config,
    );
    visualizer::set_stage("joint", "seed scan");
    visualizer::set_progress("Seed scan", 0, Some(seed_candidates.len()));

    // Bounded ρ optimization using tanh transformation (consistent with non-joint path)
    // This prevents exp(ρ) overflow/underflow and keeps Hessians well-conditioned
    const RHO_BOUND: f64 = 30.0;
    
    fn rho_to_z(rho: &Array1<f64>) -> Array1<f64> {
        rho.mapv(|r| {
            let ratio = (r / RHO_BOUND).clamp(-0.9999, 0.9999);
            RHO_BOUND * ratio.atanh()
        })
    }
    
    fn z_to_rho(z: &Array1<f64>) -> Array1<f64> {
        z.mapv(|v| RHO_BOUND * (v / RHO_BOUND).tanh())
    }
    
    fn drho_dz(rho: &Array1<f64>) -> Array1<f64> {
        rho.mapv(|r| (1.0 - (r / RHO_BOUND).powi(2)).max(0.0))
    }
    
    use wolfe_bfgs::Bfgs;
    const SYM_VS_ASYM_MARGIN: f64 = 1.001;

    let snapshot = JointRemlSnapshot::new(&reml_state);
    let mut best_symmetric_seed: Option<(Array1<f64>, f64, usize)> = None;
    let mut best_asymmetric_seed: Option<(Array1<f64>, f64, usize)> = None;
    let total_candidates = seed_candidates.len();
    let mut finite_count = 0usize;
    let mut inf_count = 0usize;
    let mut fail_count = 0usize;

    for (i, seed) in seed_candidates.iter().enumerate() {
        visualizer::set_stage("joint", &format!("seed scan {}/{}", i + 1, seed_candidates.len()));
        visualizer::set_progress("Seed scan", i + 1, Some(seed_candidates.len()));
        let cost = match reml_state.compute_cost(seed) {
            Ok(c) if c.is_finite() => {
                finite_count += 1;
                c
            }
            Ok(_) => {
                inf_count += 1;
                continue;
            }
            Err(_) => {
                fail_count += 1;
                continue;
            }
        };

        let is_symmetric = if seed.len() < 2 {
            true
        } else {
            let first_val = seed[0];
            seed.iter().all(|&val| (val - first_val).abs() < 1e-9)
        };

        if is_symmetric {
            if cost < best_symmetric_seed.as_ref().map_or(f64::INFINITY, |s| s.1) {
                best_symmetric_seed = Some((seed.clone(), cost, i));
            }
        } else if cost < best_asymmetric_seed.as_ref().map_or(f64::INFINITY, |s| s.1) {
            best_asymmetric_seed = Some((seed.clone(), cost, i));
        }
    }
    snapshot.restore(&reml_state);

    eprintln!(
        "[JOINT][Seed scan] candidates={} finite={} +inf={} failed={}",
        total_candidates, finite_count, inf_count, fail_count
    );

    let pick_asym = match (best_asymmetric_seed.as_ref(), best_symmetric_seed.as_ref()) {
        (Some((_, asym_cost, _)), Some((_, sym_cost, _))) => {
            *asym_cost <= *sym_cost * SYM_VS_ASYM_MARGIN + 1e-6
        }
        (Some(_), None) => true,
        (None, Some(_)) => false,
        (None, None) => false,
    };

    let mut candidate_plans: Vec<(String, Array1<f64>, Option<usize>, Option<f64>)> = Vec::new();
    if pick_asym {
        if let Some((rho, cost, idx)) = best_asymmetric_seed {
            candidate_plans.push(("best-asymmetric".to_string(), rho, Some(idx), Some(cost)));
        }
        if let Some((rho, cost, idx)) = best_symmetric_seed {
            candidate_plans.push(("best-symmetric".to_string(), rho, Some(idx), Some(cost)));
        }
    } else {
        if let Some((rho, cost, idx)) = best_symmetric_seed {
            candidate_plans.push(("best-symmetric".to_string(), rho, Some(idx), Some(cost)));
        }
        if let Some((rho, cost, idx)) = best_asymmetric_seed {
            candidate_plans.push(("best-asymmetric".to_string(), rho, Some(idx), Some(cost)));
        }
    }

    if candidate_plans.is_empty() {
        candidate_plans.push((
            "fallback-symmetric".to_string(),
            Array1::zeros(n_base + 1),
            None,
            None,
        ));
    }

    let mut successful_runs: Vec<(String, BfgsSolution, f64)> = Vec::new();
    let mut last_error: Option<EstimationError> = None;
    let total_candidates = candidate_plans.len();
    let mut candidate_idx = 0usize;

    for (label, rho, seed_index, seed_cost) in candidate_plans.into_iter() {
        candidate_idx += 1;
        visualizer::set_stage("joint", &format!("candidate {label}"));
        visualizer::set_progress("Candidates", candidate_idx, Some(total_candidates));
        if let Some(idx) = seed_index {
            eprintln!("[JOINT][Candidate {label}] Seed index: {idx}");
        }
        if let Some(cost) = seed_cost {
            eprintln!("[JOINT][Candidate {label}] Seed cost: {cost:.6}");
        }

        snapshot.restore(&reml_state);
        let initial_z = rho_to_z(&rho);
        let solver = Bfgs::new(initial_z, |z| {
            let rho = z_to_rho(z);
            let (cost, grad_rho) = reml_state.cost_and_grad(&rho);
            let jac = drho_dz(&rho);
            let grad_z = &grad_rho * &jac;
            (cost, grad_z)
        })
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
                last_error = Some(EstimationError::RemlOptimizationFailed(format!(
                    "BFGS failed for joint model: {e:?}"
                )));
                continue;
            }
        };
        let final_value = solution.final_value;
        successful_runs.push((label, solution, final_value));
    }

    let (_, best_solution, _) = match successful_runs
        .into_iter()
        .min_by(|a, b| match a.2.partial_cmp(&b.2) {
            Some(order) => order,
            None => std::cmp::Ordering::Equal,
        }) {
        Some(best) => best,
        None => {
            return Err(last_error.unwrap_or_else(|| {
                EstimationError::RemlOptimizationFailed(
                    "All joint REML candidate runs failed.".to_string(),
                )
            }))
        }
    };

    let best_rho = z_to_rho(&best_solution.final_point);
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
    use crate::calibrate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::calibrate::basis::internal::BsplineScratch;
    
    let n = u.len();
    let mut deriv = Array1::<f64>::ones(n);
    
    if b_wiggle.ncols() == 0 || state.beta_link.is_empty() {
        return deriv;
    }
    let Some(knot_vector) = state.knot_vector.as_ref() else {
        return deriv;
    };
    
    let (z, range_width) = state.standardized_z(u);
    let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
    
    // Get link_transform, return early if not set
    let Some(ref link_transform) = state.link_transform else {
        return deriv;
    };
    let n_constrained = link_transform.ncols();
    if n_raw == 0 || n_constrained == 0 || state.beta_link.len() != n_constrained {
        return deriv;
    }
    
    // Pre-allocate all buffers outside loop (zero-allocation, same as HMC)
    let mut deriv_raw = vec![0.0; n_raw];
    let num_basis_lower = knot_vector.len().saturating_sub(state.degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = BsplineScratch::new(state.degree.saturating_sub(1));
    
    for i in 0..n {
        let z_i = z[i];
        
        // At boundaries (clamped), g'(u) = 1 (wiggle is constant w.r.t. u)
        if z_i <= 1e-8 || z_i >= 1.0 - 1e-8 {
            deriv[i] = 1.0;
            continue;
        }
        
        deriv_raw.fill(0.0);
        if evaluate_bspline_derivative_scalar_into(
            z_i, knot_vector.view(), state.degree,
            &mut deriv_raw, &mut lower_basis, &mut lower_scratch
        ).is_err() {
            continue;
        }
        
        // d(wiggle)/dz = B'(z) @ Z @ θ
        let d_wiggle_dz: f64 = if link_transform.nrows() == n_raw {
            (0..n_constrained).map(|c| {
                let b_prime_c: f64 = (0..n_raw).map(|r|
                    deriv_raw[r] * link_transform[[r, c]]
                ).sum();
                b_prime_c * state.beta_link[c]
            }).sum()
        } else {
            0.0
        };
        
        deriv[i] = 1.0 + d_wiggle_dz / range_width;
    }
    
    deriv
}

fn compute_link_derivative_terms_from_state(
    state: &JointModelState,
    u: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>, Array2<f64>) {
    use crate::calibrate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::calibrate::basis::evaluate_bspline_second_derivative_scalar_into;
    use crate::calibrate::basis::internal::BsplineScratch;

    let n = u.len();
    let p_link = state.beta_link.len();
    let mut g_prime = Array1::<f64>::ones(n);
    let mut g_second = Array1::<f64>::zeros(n);
    let mut b_prime_u = Array2::<f64>::zeros((n, p_link));

    if p_link == 0 {
        return (g_prime, g_second, b_prime_u);
    }
    let Some(knot_vector) = state.knot_vector.as_ref() else {
        return (g_prime, g_second, b_prime_u);
    };
    let Some(link_transform) = state.link_transform.as_ref() else {
        return (g_prime, g_second, b_prime_u);
    };

    let n_raw = knot_vector.len().saturating_sub(state.degree + 1);
    if n_raw == 0 || link_transform.nrows() != n_raw || link_transform.ncols() != p_link {
        return (g_prime, g_second, b_prime_u);
    }

    let (z, range_width) = state.standardized_z(u);
    let inv_rw = 1.0 / range_width;
    let inv_rw2 = inv_rw * inv_rw;

    let mut deriv_raw = vec![0.0; n_raw];
    let num_basis_lower = knot_vector.len().saturating_sub(state.degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = BsplineScratch::new(state.degree.saturating_sub(1));

    let mut second_raw = vec![0.0; n_raw];
    let num_basis_lower_second = knot_vector.len().saturating_sub(state.degree - 1);
    let mut deriv_lower = vec![0.0; num_basis_lower_second.saturating_sub(1)];
    let mut lower_basis_second = vec![0.0; num_basis_lower_second];
    let mut lower_scratch_second = BsplineScratch::new(state.degree.saturating_sub(2));

    for i in 0..n {
        let z_i = z[i];
        if z_i <= 1e-8 || z_i >= 1.0 - 1e-8 {
            continue;
        }

        deriv_raw.fill(0.0);
        if evaluate_bspline_derivative_scalar_into(
            z_i,
            knot_vector.view(),
            state.degree,
            &mut deriv_raw,
            &mut lower_basis,
            &mut lower_scratch,
        ).is_err() {
            continue;
        }

        let mut d_wiggle_dz = 0.0;
        for c in 0..p_link {
            let mut b_prime_c = 0.0;
            for r in 0..n_raw {
                b_prime_c += deriv_raw[r] * link_transform[[r, c]];
            }
            b_prime_u[[i, c]] = b_prime_c * inv_rw;
            d_wiggle_dz += b_prime_c * state.beta_link[c];
        }
        g_prime[i] = 1.0 + d_wiggle_dz * inv_rw;

        second_raw.fill(0.0);
        if evaluate_bspline_second_derivative_scalar_into(
            z_i,
            knot_vector.view(),
            state.degree,
            &mut second_raw,
            &mut deriv_lower,
            &mut lower_basis_second,
            &mut lower_scratch_second,
        ).is_err() {
            continue;
        }
        let mut d2_wiggle_dz2 = 0.0;
        for c in 0..p_link {
            let mut b_second_c = 0.0;
            for r in 0..n_raw {
                b_second_c += second_raw[r] * link_transform[[r, c]];
            }
            d2_wiggle_dz2 += b_second_c * state.beta_link[c];
        }
        g_second[i] = d2_wiggle_dz2 * inv_rw2;
    }

    (g_prime, g_second, b_prime_u)
}

/// Public version for use in HMC Hessian computation
pub fn compute_link_derivative_from_result_public(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    b_wiggle: &Array2<f64>,
) -> Array1<f64> {
    compute_link_derivative_from_result(result, eta_base, b_wiggle)
}

fn compute_link_derivative_from_result(
    result: &JointModelResult,
    eta_base: &Array1<f64>,
    b_wiggle: &Array2<f64>,
) -> Array1<f64> {
    use crate::calibrate::basis::evaluate_bspline_derivative_scalar_into;
    use crate::calibrate::basis::internal::BsplineScratch;
    
    let n = eta_base.len();
    let mut deriv = Array1::<f64>::ones(n);
    if b_wiggle.ncols() == 0 || result.beta_link.is_empty() {
        return deriv;
    }
    
    let (min_u, max_u) = result.knot_range;
    let range_width = (max_u - min_u).max(1e-6);
    let z: Array1<f64> = eta_base.mapv(|u| ((u - min_u) / range_width).clamp(0.0, 1.0));
    let n_raw = result.knot_vector.len().saturating_sub(result.degree + 1);
    let n_constrained = result.link_transform.ncols();
    if n_raw == 0 || n_constrained == 0 || result.beta_link.len() != n_constrained {
        return deriv;
    }
    
    // Pre-allocate all buffers outside loop (zero-allocation, same as HMC)
    let mut deriv_raw = vec![0.0; n_raw];
    let num_basis_lower = result.knot_vector.len().saturating_sub(result.degree);
    let mut lower_basis = vec![0.0; num_basis_lower];
    let mut lower_scratch = BsplineScratch::new(result.degree.saturating_sub(1));
    
    for i in 0..n {
        let z_i = z[i];
        
        // At boundaries (clamped), g'(u) = 1 (wiggle is constant w.r.t. u)
        if z_i <= 1e-8 || z_i >= 1.0 - 1e-8 {
            deriv[i] = 1.0;
            continue;
        }
        
        deriv_raw.fill(0.0);
        if evaluate_bspline_derivative_scalar_into(
            z_i, result.knot_vector.view(), result.degree,
            &mut deriv_raw, &mut lower_basis, &mut lower_scratch
        ).is_err() {
            continue;
        }
        
        // d(wiggle)/dz = B'(z) @ Z @ θ
        let d_wiggle_dz: f64 = if result.link_transform.nrows() == n_raw {
            (0..n_constrained).map(|c| {
                let b_prime_c: f64 = (0..n_raw).map(|r|
                    deriv_raw[r] * result.link_transform[[r, c]]
                ).sum();
                b_prime_c * result.beta_link[c]
            }).sum()
        } else {
            0.0
        };
        
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
    let b_wiggle = match create_basis::<Dense>(
        z.view(),
        KnotSource::Provided(result.knot_vector.view()),
        result.degree,
        BasisOptions::value(),
    ) {
        Ok((basis, _)) => {
            let raw = basis.as_ref();
            if result.link_transform.ncols() > 0 && result.link_transform.nrows() == raw.ncols() {
                raw.dot(&result.link_transform)
            } else {
                Array2::zeros((n, result.beta_link.len()))
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
    let quad_ctx = QuadratureContext::new();
    let (probabilities, effective_se) = if let Some(se) = se_base {
        // Compute link derivative for uncertainty propagation
        let deriv = compute_link_derivative_from_result(result, eta_base, &b_wiggle);
        
        // Effective SE = |g'(η)| × SE_base
        let eff_se: Array1<f64> = deriv.mapv(f64::abs) * se;
        
        let probs = match result.link {
            LinkFunction::Logit => (0..n)
                .map(|i| crate::calibrate::quadrature::logit_posterior_mean(&quad_ctx, eta_cal[i], eff_se[i]))
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
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    
    #[test]
    fn test_joint_model_state_creation() {
        let n = 100;
        let p = 10;
        let y = Array1::zeros(n);
        let weights = Array1::ones(n);
        let x = Array2::zeros((n, p));
        let s = vec![Array2::eye(p)];
        let layout = ModelLayout::external(p, 1);
        let mut config = JointModelConfig::default();
        config.n_link_knots = 4;
        let quad_ctx = QuadratureContext::new();
        
        let state = JointModelState::new(
            y.view(),
            weights.view(), 
            x.view(),
            s,
            layout,
            LinkFunction::Logit,
            &config,
            quad_ctx,
        );
        
        assert_eq!(state.beta_base.len(), p);
        assert_eq!(state.beta_link.len(), config.n_link_knots + 2);
    }
    
    #[test]
    fn test_predict_joint_basic() {
        // Create a simple result with logit link (no wiggle)
        let n_knots = 5;
        let degree = 3;
        let (basis_arc, knot_vector) = create_basis::<Dense>(
            Array1::from_vec(vec![0.0]).view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: n_knots - degree - 1,
            },
            degree,
            BasisOptions::value(),
        )
        .expect("basis");
        let basis = (*basis_arc).clone();
        let num_basis = knot_vector.len().saturating_sub(degree + 1);
        assert_eq!(basis.ncols(), num_basis);
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
            s_link_constrained: Array2::eye(num_basis),
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
        let (basis_arc, knot_vector) = create_basis::<Dense>(
            Array1::from_vec(vec![0.0]).view(),
            KnotSource::Generate {
                data_range: (0.0, 1.0),
                num_internal_knots: n_knots - degree - 1,
            },
            degree,
            BasisOptions::value(),
        )
        .expect("basis");
        let basis = (*basis_arc).clone();
        let num_basis = knot_vector.len().saturating_sub(degree + 1);
        assert_eq!(basis.ncols(), num_basis);
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
            s_link_constrained: Array2::eye(num_basis),
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

    #[test]
    fn test_joint_analytic_gradient_matches_fd() {
        let n = 600;
        let p = 6;
        let weights = Array1::ones(n);
        let s = vec![Array2::eye(p)];
        let layout = ModelLayout::external(p, 1);
        let config = JointModelConfig::default();

        let mut grad_analytic = None;
        let mut grad_fd = None;
        let mut last_err: Option<String> = None;
        for attempt in 0..3 {
            let mut rng = StdRng::seed_from_u64(123 + attempt);
            let mut x = Array2::<f64>::zeros((n, p));
            for i in 0..n {
                for j in 0..p {
                    x[[i, j]] = rng.gen_range(-1.0..1.0);
                }
            }

            let mut beta_true = Array1::<f64>::zeros(p);
            for j in 0..p {
                beta_true[j] = rng.gen_range(-0.5..0.5);
            }
            let eta = x.dot(&beta_true);
            let mut y = Array1::<f64>::zeros(n);
            for i in 0..n {
                let mu = 1.0 / (1.0 + (-eta[i]).exp());
                y[i] = if rng.r#gen::<f64>() < mu { 1.0 } else { 0.0 };
            }

            let reml_state = JointRemlState::new(
                y.view(),
                weights.view(),
                x.view(),
                s.clone(),
                layout.clone(),
                LinkFunction::Logit,
                &config,
                None,
                QuadratureContext::new(),
            );
            {
                let mut state = reml_state.state.borrow_mut();
                state.beta_base = beta_true.clone();
                *reml_state.cached_beta_base.borrow_mut() = beta_true.clone();
                let u = state.base_linear_predictor();
                if state.build_link_basis(&u).is_err() {
                    continue;
                }
            }

            let rho = Array1::from_vec(vec![0.0, 2.0]);
            match reml_state.compute_gradient_analytic_for_test(&rho) {
                Ok(ga) => {
                    let gf = reml_state
                        .compute_gradient_fd_for_test(&rho)
                        .expect("fd gradient");
                    grad_analytic = Some(ga);
                    grad_fd = Some(gf);
                    break;
                }
                Err(err) => {
                    last_err = Some(err.to_string());
                    continue;
                }
            }
        }

        let grad_analytic = match grad_analytic {
            Some(g) => g,
            None => panic!("analytic gradient: {}", last_err.unwrap_or_else(|| "unknown".to_string())),
        };
        let grad_fd = grad_fd.expect("fd gradient");

        let mut diff_norm = 0.0;
        let mut fd_norm = 0.0;
        for i in 0..grad_fd.len() {
            let d = grad_analytic[i] - grad_fd[i];
            diff_norm += d * d;
            fd_norm += grad_fd[i] * grad_fd[i];
        }
        let rel = diff_norm.sqrt() / (fd_norm.sqrt().max(1.0));
        assert!(
            rel < 1e-2,
            "analytic/FD gradient mismatch: rel={rel:.3e}"
        );
    }
}
