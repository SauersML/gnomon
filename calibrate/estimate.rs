//! # Model Estimation via Penalized Likelihood and REML
//!
//! This module orchestrates the core model fitting procedure. It transitions from a
//! simple hyperparameter-driven model to a statistically robust one where smoothing
//! parameters are estimated automatically from the data. This is achieved through a
//! nested optimization scheme:
//!
//! 1.  **Outer Loop (BFGS):** Optimizes the log-smoothing parameters (`rho`) by
//!     maximizing the Laplace Approximate Marginal Likelihood (LAML), which serves as the
//!     Restricted Maximum Likelihood (REML) criterion. This approach is detailed in
//!     Wood (2011).
//!
//! 2.  **Inner Loop (P-IRLS):** For each set of trial smoothing parameters from the
//!     outer loop, this routine finds the corresponding model coefficients (`beta`) by
//!     running a Penalized Iteratively Reweighted Least Squares (P-IRLS) algorithm
//!     to convergence.
//!
//! This structure ensures that the model complexity for each smooth term is learned
//! from the data, fixing the key discrepancies identified in the initial implementation.

// External Crate for Optimization
use argmin::core::{CostFunction, Error, Executor, Gradient, State};
use argmin::solver::bfgs::BFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;

// Crate-level imports
use crate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::data::TrainingData;
use crate::model::{LinkFunction, MainEffects, MappedCoefficients, ModelConfig, TrainedModel};

// Ndarray and Linalg
use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::{Cholesky, Eigh, Solve, UPLO};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ops::Range;
use thiserror::Error;

/// A comprehensive error type for the model estimation process.
#[derive(Error, Debug)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] basis::BasisError),

    #[error("A linear system solve failed. The penalized Hessian may be singular. Error: {0}")]
    LinearSystemSolveFailed(ndarray_linalg::error::LinalgError),

    #[error("Eigendecomposition failed: {0}")]
    EigendecompositionFailed(ndarray_linalg::error::LinalgError),

    #[error("The P-IRLS inner loop did not converge within {max_iterations} iterations. Last deviance change was {last_change:.6e}.")]
    PirlsDidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },

    #[error("REML/BFGS optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

    #[error("An internal error occurred during model layout or coefficient mapping: {0}")]
    LayoutError(String),
}

/// The main entry point for model training. Orchestrates the REML/BFGS optimization.
pub fn train_model(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    log::info!(
        "Starting model training with REML. {} total samples.",
        data.y.len()
    );

    // 1. Build the one-time matrices and define the model structure.
    let (x_matrix, penalty_matrices, layout) =
        internal::build_design_and_penalty_matrices(data, config)?;
    log::info!(
        "Constructed design matrix with {} samples and {} total coefficients.",
        x_matrix.nrows(),
        x_matrix.ncols()
    );
    log_layout_info(&layout);

    // 2. Set up the REML optimization problem.
    let cost_function = internal::RemlState::new(
        data.y.view(),
        x_matrix.view(),
        penalty_matrices,
        &layout,
        config,
    );

    // 3. Configure the BFGS optimizer and the line search strategy.
    let linesearch = MoreThuenteLineSearch::new();
    let solver = BFGS::new(linesearch);

    // 4. Define the initial guess for log-smoothing parameters (rho).
    let initial_rho = Array1::zeros(layout.num_penalties);

    // 5. Run the optimizer.
    let res = Executor::new(cost_function, solver)
        .configure(|state| {
            state
                .param(initial_rho)
                .max_iters(config.reml_max_iterations)
                .gradient_tol(config.reml_convergence_tolerance)
        })
        .run()
        .map_err(|e| EstimationError::RemlOptimizationFailed(e.to_string()))?;

    log::info!(
        "REML optimization finished: {}",
        res.state().termination_reason.unwrap()
    );

    // 6. Extract the final results.
    let final_rho = res.state.best_param.as_ref().unwrap().clone();
    let final_lambda = final_rho.mapv(f64::exp);
    log::info!(
        "Final estimated smoothing parameters (lambda): {:?}",
        &final_lambda.to_vec()
    );

    // Fit the model one last time with the optimal lambdas to get final coefficients.
    let final_fit = internal::fit_model_for_fixed_rho(
        final_rho.view(),
        x_matrix.view(),
        data.y.view(),
        &res.state.cost_function.unwrap().s_list,
        &layout,
        config,
    )?;

    let mapped_coefficients =
        internal::map_coefficients(&final_fit.beta, &layout)?;

    Ok(TrainedModel {
        config: config.clone(),
        coefficients: mapped_coefficients,
        estimated_lambdas: Some(final_lambda.to_vec()),
    })
}

/// Helper to log the final model structure.
fn log_layout_info(layout: &internal::ModelLayout) {
    log::info!("Model structure has {} total coefficients.", layout.total_coeffs);
    log::info!("  - Intercept: 1 coefficient.");
    log::info!(
        "  - PGS Main Effect: {} coefficients.",
        layout.pgs_main_cols.len()
    );
    let mut pc_main_count = 0;
    let mut interaction_count = 0;
    for block in &layout.penalty_map {
        if block.term_name.starts_with("f(PC") {
            pc_main_count += 1;
        } else if block.term_name.starts_with("f(PGS_B") {
            interaction_count += 1;
        }
    }
    log::info!("  - PC Main Effects: {} terms.", pc_main_count);
    log::info!("  - Interaction Effects: {} terms.", interaction_count);
    log::info!("Total penalized terms: {}", layout.num_penalties);
}

/// Internal module for estimation logic.
mod internal {
    use super::*;
    use ndarray_linalg::error::LinalgError;

    /// Holds the result of a converged P-IRLS inner loop for a fixed rho.
    #[derive(Clone)]
    pub(super) struct PirlsResult {
        pub(super) beta: Array1<f64>,
        pub(super) penalized_hessian: Array2<f64>,
        pub(super) deviance: f64,
        pub(super) final_weights: Array1<f64>,
    }

    /// Holds the state for the outer REML optimization. Implements `CostFunction`
    /// and `Gradient` for the `argmin` library.
    pub(super) struct RemlState<'a> {
        y: ArrayView1<'a, f64>,
        x: ArrayView2<'a, f64>,
        pub(super) s_list: Vec<Array2<f64>>,
        layout: &'a ModelLayout,
        config: &'a ModelConfig,
        cache: RefCell<HashMap<Vec<u64>, PirlsResult>>,
    }

    impl<'a> RemlState<'a> {
        pub(super) fn new(
            y: ArrayView1<'a, f64>,
            x: ArrayView2<'a, f64>,
            s_list: Vec<Array2<f64>>,
            layout: &'a ModelLayout,
            config: &'a ModelConfig,
        ) -> Self {
            Self {
                y,
                x,
                s_list,
                layout,
                config,
                cache: RefCell::new(HashMap::new()),
            }
        }

        /// Runs the inner P-IRLS loop, caching the result.
        fn execute_pirls_if_needed(&self, rho: &Array1<f64>) -> Result<PirlsResult, Error> {
            let key: Vec<u64> = rho.iter().map(|&v| v.to_bits()).collect();
            if let Some(cached_result) = self.cache.borrow().get(&key) {
                return Ok(cached_result.clone());
            }

            let pirls_result = fit_model_for_fixed_rho(
                rho.view(),
                self.x,
                self.y,
                &self.s_list,
                self.layout,
                self.config,
            )
            .map_err(Error::from)?;

            self.cache.borrow_mut().insert(key, pirls_result.clone());
            Ok(pirls_result)
        }
    }

    /// Implementation of the LAML score (cost function) for the BFGS optimizer.
    impl CostFunction for RemlState<'_> {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            let pirls_result = self.execute_pirls_if_needed(p)?;
            let lambdas = p.mapv(f64::exp);

            let s_lambda = construct_s_lambda(&lambdas, &self.s_list, self.layout);

            let penalised_ll = -0.5 * pirls_result.deviance
                - 0.5 * pirls_result.beta.dot(&s_lambda.dot(&pirls_result.beta));

            let log_det_s = calculate_log_det_pseudo(&s_lambda).map_err(Error::from)?;

            let log_det_h = pirls_result
                .penalized_hessian
                .cholesky(UPLO::Lower)
                .map(|l| 2.0 * l.diag().mapv(f64::ln).sum())
                .unwrap_or(f64::NEG_INFINITY);

            let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_h;

            Ok(-laml)
        }
    }

    /// Implementation of the LAML gradient for the BFGS optimizer.
    impl Gradient for RemlState<'_> {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            let pirls_result = self.execute_pirls_if_needed(p)?;
            
            let h_penalized = &pirls_result.penalized_hessian;
            let beta = &pirls_result.beta;
            let lambdas = p.mapv(f64::exp);
            
            let s_lambda = construct_s_lambda(&lambdas, &self.s_list, self.layout);
            let s_lambda_plus = pseudo_inverse(&s_lambda).map_err(Error::from)?;
            
            let mut grad = Array1::zeros(p.len());

            for k in 0..p.len() {
                let d_s_lambda_d_rho_k = lambdas[k] * &self.s_list[k];

                // Term 1: from the penalty on beta
                let term1 = beta.dot(&d_s_lambda_d_rho_k.dot(beta));

                // Term 2: from log |S_lambda|+
                let term2 = (s_lambda_plus.dot(&d_s_lambda_d_rho_k)).diag().sum();
                
                // Term 3: from log |H|. This is the most complex.
                let d_beta_d_rho_k = -h_penalized
                    .solve_into(d_s_lambda_d_rho_k.dot(beta))
                    .map_err(Error::from)?;
                
                let d_eta_d_rho_k = self.x.dot(&d_beta_d_rho_k);
                
                let mu = (self.x.dot(beta)).mapv(|eta| 1.0 / (1.0 + (-eta).exp()));
                let d_w_d_eta = &mu * (1.0 - &mu) * (1.0 - 2.0 * &mu);
                let d_w_d_rho_k = &d_w_d_eta * &d_eta_d_rho_k;
                let d_xtwx_d_rho_k = self.x.t().dot(&(self.x.view() * d_w_d_rho_k.view().insert_axis(Axis(1))));
                
                let d_h_d_rho_k = d_xtwx_d_rho_k + &d_s_lambda_d_rho_k;
                
                let term3 = h_penalized.solve_into(d_h_d_rho_k).map_err(Error::from)?.diag().sum();

                grad[k] = 0.5 * (term1 - term3 + term2);
            }

            Ok(-grad)
        }
    }

    /// Holds the layout of the design matrix `X` and penalty matrices `S_i`.
    #[derive(Clone)]
    pub(super) struct ModelLayout {
        pub intercept_col: usize,
        pub pgs_main_cols: Range<usize>,
        pub penalty_map: Vec<PenalizedBlock>,
        pub total_coeffs: usize,
        pub num_penalties: usize,
    }
    
    /// Information about a single penalized block of coefficients.
    #[derive(Clone)]
    pub(super) struct PenalizedBlock {
        pub term_name: String,
        pub col_range: Range<usize>,
        pub penalty_idx: usize,
    }

    impl ModelLayout {
        /// Creates a new layout based on the model configuration and basis dimensions.
        pub(super) fn new(
            config: &ModelConfig,
            pc_constrained_basis_ncols: &[usize],
            pgs_main_basis_ncols: usize,
        ) -> Result<Self, EstimationError> {
            let mut penalty_map = Vec::new();
            let mut current_col = 0;
            let mut penalty_idx_counter = 0;

            let intercept_col = current_col;
            current_col += 1;

            // Main effect for each PC
            for (i, &num_basis) in pc_constrained_basis_ncols.iter().enumerate() {
                let range = current_col..current_col + num_basis;
                penalty_map.push(PenalizedBlock {
                    term_name: format!("f({})", config.pc_names[i]),
                    col_range: range.clone(),
                    penalty_idx: penalty_idx_counter,
                });
                current_col += num_basis;
                penalty_idx_counter += 1;
            }

            // Main effect for PGS (non-constant basis terms)
            let pgs_main_cols = current_col..current_col + pgs_main_basis_ncols;
            if pgs_main_basis_ncols > 0 {
                penalty_map.push(PenalizedBlock {
                    term_name: "f(PGS)".to_string(),
                    col_range: pgs_main_cols.clone(),
                    penalty_idx: penalty_idx_counter,
                });
                penalty_idx_counter += 1;
            }
            current_col += pgs_main_basis_ncols;
            
            // Interaction effects
            for m in 0..pgs_main_basis_ncols {
                for (i, &num_basis) in pc_constrained_basis_ncols.iter().enumerate() {
                    let range = current_col..current_col + num_basis;
                    penalty_map.push(PenalizedBlock {
                        term_name: format!("f(PGS_B{}, {})", m + 1, config.pc_names[i]),
                        col_range: range.clone(),
                        penalty_idx: penalty_idx_counter,
                    });
                    current_col += num_basis;
                    penalty_idx_counter += 1;
                }
            }

            Ok(ModelLayout {
                intercept_col,
                pgs_main_cols,
                penalty_map,
                total_coeffs: current_col,
                num_penalties: penalty_idx_counter,
            })
        }
    }

    /// Constructs the design matrix `X` and a list of individual penalty matrices `S_i`.
    pub(super) fn build_design_and_penalty_matrices(
        data: &TrainingData,
        config: &ModelConfig,
    ) -> Result<(Array2<f64>, Vec<Array2<f64>>, ModelLayout), EstimationError> {
        // --- 1. Generate basis for PGS ---
        let (pgs_basis, _) = create_bspline_basis(
            data.p.view(), Some(data.p.view()), config.pgs_range,
            config.pgs_basis_config.num_knots, config.pgs_basis_config.degree,
        )?;
        let pgs_main_basis = pgs_basis.slice(s![.., 1..]);

        // --- 2. Generate constrained bases and penalties for PCs ---
        let mut pc_constrained_bases = Vec::new();
        let mut s_list = Vec::new();
        for i in 0..config.pc_names.len() {
            let pc_col = data.pcs.column(i);
            let (pc_basis, _) = create_bspline_basis(
                pc_col.view(), Some(pc_col.view()), config.pc_ranges[i],
                config.pc_basis_configs[i].num_knots, config.pc_basis_configs[i].degree,
            )?;
            let (constrained_basis, z_transform) = basis::apply_sum_to_zero_constraint(pc_basis.view())?;
            let s_unconstrained = create_difference_penalty_matrix(pc_basis.ncols(), config.penalty_order)?;
            s_list.push(z_transform.t().dot(&s_unconstrained.dot(&z_transform)));
            pc_constrained_bases.push(constrained_basis);
        }

        // --- 3. Create penalties for PGS main effect and interactions ---
        if pgs_main_basis.ncols() > 0 {
            s_list.push(create_difference_penalty_matrix(pgs_main_basis.ncols(), config.penalty_order)?);
        }
        for _ in 0..pgs_main_basis.ncols() {
            for i in 0..pc_constrained_bases.len() {
                s_list.push(s_list[i].clone()); // Re-use the PC main effect penalty
            }
        }

        // --- 4. Define the layout and assemble the design matrix X ---
        let pc_basis_ncols: Vec<usize> = pc_constrained_bases.iter().map(|b| b.ncols()).collect();
        let layout = ModelLayout::new(config, &pc_basis_ncols, pgs_main_basis.ncols())?;
        
        if s_list.len() != layout.num_penalties {
             return Err(EstimationError::LayoutError(format!("Internal logic error: Mismatch in number of penalties. Layout expects {}, but {} were generated.", layout.num_penalties, s_list.len())));
        }

        let mut x_matrix = Array2::zeros((data.y.len(), layout.total_coeffs));
        x_matrix.column_mut(layout.intercept_col).fill(1.0);
        x_matrix.slice_mut(s![.., layout.pgs_main_cols.clone()]).assign(&pgs_main_basis);
        
        for block in &layout.penalty_map {
            if block.term_name.starts_with("f(PC") {
                let pc_idx: usize = block.term_name.chars().filter(|c| c.is_digit(10)).collect::<String>().parse::<usize>().unwrap_or(0) - 1;
                x_matrix.slice_mut(s![.., block.col_range.clone()]).assign(&pc_constrained_bases[pc_idx]);
            } else if block.term_name.starts_with("f(PGS_B") {
                let parts: Vec<_> = block.term_name.split(|c| c==',' || c==')').collect();
                let m_idx: usize = parts[0].chars().filter(|c| c.is_digit(10)).collect::<String>().parse().unwrap_or(0) - 1;
                let pc_idx: usize = parts[1].chars().filter(|c| c.is_digit(10)).collect::<String>().parse().unwrap_or(0) - 1;
                let pgs_col = pgs_main_basis.column(m_idx);
                let pc_basis = &pc_constrained_bases[pc_idx];
                let interaction_data = &pgs_col.to_shape((data.y.len(), 1)).unwrap() * pc_basis;
                x_matrix.slice_mut(s![.., block.col_range.clone()]).assign(&interaction_data);
            }
        }

        Ok((x_matrix, s_list, layout))
    }

    /// The P-IRLS inner loop for a fixed set of smoothing parameters.
    pub(super) fn fit_model_for_fixed_rho(
        rho_vec: ArrayView1<f64>,
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        s_list: &[Array2<f64>],
        layout: &ModelLayout,
        config: &ModelConfig,
    ) -> Result<PirlsResult, EstimationError> {
        let lambdas = rho_vec.mapv(f64::exp);
        let s_lambda = construct_s_lambda(&lambdas, s_list, layout);

        let mut beta = Array1::zeros(x.ncols());
        let mut last_deviance = f64::INFINITY;
        let mut last_change = f64::INFINITY;

        for _iter in 1..=config.max_iterations {
            let eta = x.dot(&beta);
            let (mu, weights, z) = update_glm_vectors(y, &eta, config.link_function);

            let x_t_w = &x.t() * &weights;
            let penalized_hessian = x_t_w.dot(&x) + &s_lambda;
            let rhs = x_t_w.dot(&z);
            beta = penalized_hessian
                .solve_into(rhs)
                .map_err(EstimationError::LinearSystemSolveFailed)?;

            let deviance = calculate_deviance(y, &mu, config.link_function);
            let deviance_change = (last_deviance - deviance).abs();

            if deviance_change < config.convergence_tolerance {
                let final_eta = x.dot(&beta);
                let (final_mu, final_weights, _) = update_glm_vectors(y, &final_eta, config.link_function);
                let final_xtwx = x.t().dot(&(x.view() * final_weights.view().insert_axis(Axis(1))));
                let final_penalized_hessian = final_xtwx + &s_lambda;

                return Ok(PirlsResult {
                    beta,
                    penalized_hessian: final_penalized_hessian,
                    deviance: calculate_deviance(y, &final_mu, config.link_function),
                    linear_predictor: final_eta,
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
    
    /// Helper to construct the summed, weighted penalty matrix S_lambda.
    fn construct_s_lambda(lambdas: &Array1<f64>, s_list: &[Array2<f64>], layout: &ModelLayout) -> Array2<f64> {
        let mut s_lambda = Array2::zeros((layout.total_coeffs, layout.total_coeffs));
        for block in &layout.penalty_map {
            let s_i = &s_list[block.penalty_idx];
            let lambda_i = lambdas[block.penalty_idx];
            let mut target_block = s_lambda.slice_mut(s![block.col_range.clone(), block.col_range.clone()]);
            target_block.scaled_add(lambda_i, s_i);
        }
        s_lambda
    }

    /// Helper to calculate log |S|+ robustly using eigendecomposition.
    fn calculate_log_det_pseudo(s: &Array2<f64>) -> Result<f64, LinalgError> {
        let eigenvalues = s.eigh(UPLO::Lower)?.0;
        Ok(eigenvalues
            .iter()
            .filter(|&&eig| eig.abs() > 1e-12)
            .map(|&eig| eig.ln())
            .sum())
    }
    
    /// Helper to calculate the pseudo-inverse robustly.
    fn pseudo_inverse(s: &Array2<f64>) -> Result<Array2<f64>, LinalgError> {
        let (eigvals, eigvecs) = s.eigh(UPLO::Lower)?;
        let mut d_plus = Array1::zeros(eigvals.len());
        for (i, &eig) in eigvals.iter().enumerate() {
            if eig.abs() > 1e-9 {
                d_plus[i] = 1.0 / eig;
            }
        }
        Ok(eigvecs.dot(&Array2::from_diag(&d_plus)).dot(&eigvecs.t()))
    }

    pub(super) fn update_glm_vectors(
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        link: LinkFunction,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        const MIN_WEIGHT: f64 = 1e-6;

        match link {
            LinkFunction::Logit => {
                let mu = eta.mapv(|e| 1.0 / (1.0 + (-e).exp()));
                let weights = (&mu * (1.0 - &mu)).mapv(|v| v.max(MIN_WEIGHT));
                let z = eta + (y - &mu) / &weights;
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

    pub(super) fn calculate_deviance(
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        link: LinkFunction,
    ) -> f64 {
        const EPS: f64 = 1e-9;
        match link {
            LinkFunction::Logit => {
                let total_residual = ndarray::Zip::from(y)
                    .and_from(mu)
                    .fold(0.0, |acc, &yi, &mui| {
                        let mui_c = mui.clamp(EPS, 1.0 - EPS);
                        let term1 = if yi > EPS { yi * (yi / mui_c).ln() } else { 0.0 };
                        let term2 = if yi < 1.0 - EPS {
                            (1.0 - yi) * ((1.0 - yi) / (1.0 - mui_c)).ln()
                        } else {
                            0.0
                        };
                        acc + term1 + term2
                    });
                2.0 * total_residual
            }
            LinkFunction::Identity => (y - mu).mapv(|v| v.powi(2)).sum(),
        }
    }

    /// Maps the flattened coefficient vector to a structured representation.
    pub(super) fn map_coefficients(
        beta: &Array1<f64>,
        layout: &ModelLayout,
    ) -> Result<MappedCoefficients, EstimationError> {
        let intercept = beta[layout.intercept_col];
        let mut pgs = vec![];
        let mut pcs = HashMap::new();
        let mut interaction_effects = HashMap::new();

        // Use the layout as the source of truth
        for block in &layout.penalty_map {
            let coeffs = beta.slice(s![block.col_range.clone()]).to_vec();
            match block.term_name.as_str() {
                "f(PGS)" => {
                    pgs = coeffs;
                }
                _ if block.term_name.starts_with("f(PC") => {
                    let pc_name = block.term_name.replace("f(", "").replace(")", "");
                    pcs.insert(pc_name, coeffs);
                }
                _ if block.term_name.starts_with("f(PGS_B") => {
                    let parts: Vec<_> = block.term_name.split(|c| c == ',' || c == ')').collect();
                    let pgs_key = parts[0].replace("f(", "").to_string();
                    let pc_name = parts[1].trim().to_string();
                    interaction_effects
                        .entry(pgs_key)
                        .or_insert_with(HashMap::new)
                        .insert(pc_name, coeffs);
                }
                _ => return Err(EstimationError::LayoutError(format!("Unknown term name in layout during coefficient mapping: {}", block.term_name))),
            }
        }
        
        // Add the unpenalized part of the PGS main effect
        if layout.pgs_main_cols.len() > pgs.len() {
            // This assumes the penalized PGS part comes first in the layout map
            let pgs_full_coeffs = beta.slice(s![layout.pgs_main_cols.clone()]).to_vec();
            // This is complex. A simpler layout would be better.
            // For now, we assume the layout map covers all penalized parts.
        } else {
            // The layout map should cover all terms, so pgs is already complete.
        }


        Ok(MappedCoefficients {
            intercept,
            main_effects: MainEffects { pgs, pcs },
            interaction_effects,
        })
    }
}


// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BasisConfig;
    use ndarray::{array, Array};

    fn create_test_config() -> ModelConfig {
        ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            convergence_tolerance: 1e-7,
            max_iterations: 15,
            reml_convergence_tolerance: 1e-3,
            reml_max_iterations: 15,
            pgs_basis_config: BasisConfig { num_knots: 5, degree: 3 },
            pc_basis_configs: vec![BasisConfig { num_knots: 4, degree: 3 }],
            pgs_range: (-3.0, 3.0),
            pc_ranges: vec![(-3.0, 3.0)],
            pc_names: vec!["PC1".to_string()],
        }
    }

    #[test]
    fn smoke_test_full_training_pipeline() {
        let n_samples = 200;
        let mut y = Array::from_elem(n_samples, 0.0);
        y.slice_mut(s![n_samples/2..]).fill(1.0); 
        let p = Array::linspace(-2.0, 2.0, n_samples);
        let pcs = Array::linspace(-2.5, 2.5, n_samples).into_shape((n_samples, 1)).unwrap();
        let data = TrainingData { y, p, pcs };
        
        let config = create_test_config();
        
        let result = train_model(&data, &config);
        
        assert!(result.is_ok(), "Model training failed: {:?}", result.err());
        let model = result.unwrap();
        
        assert!(model.estimated_lambdas.is_some());
        let lambdas = model.estimated_lambdas.unwrap();
        assert!(!lambdas.is_empty());
        assert!(lambdas.iter().all(|&l| l > 0.0));
    }
}
