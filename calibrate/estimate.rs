use crate::basis::{self, create_bspline_basis, create_difference_penalty_matrix};
use crate::data::TrainingData;
use crate::model::{LinkFunction, MainEffects, MappedCoefficients, ModelConfig, TrainedModel};

use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::Solve;
use std::collections::HashMap;
use thiserror::Error;

/// A comprehensive error type for the model estimation process.
#[derive(Error, Debug, PartialEq)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] basis::BasisError),

    #[error("The penalized matrix system is singular and could not be solved. This can happen with a very large lambda. Error: {0}")]
    MatrixInversionFailed(ndarray_linalg::error::LinalgError),

    #[error("Iterative solver did not converge within {max_iterations} iterations. Last deviance change was {last_change:.6e}.")]
    DidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },
}

/// The main entry point for model training.
///
/// This function orchestrates the entire fitting process using a penalized
/// Iteratively Reweighted Least Squares (IRLS) algorithm to find the optimal
/// coefficients for the specified GAM.
pub fn train_model(
    data: &TrainingData,
    config: &ModelConfig,
) -> Result<TrainedModel, EstimationError> {
    println!(
        "Starting model training with lambda = {:.4e}, penalty order = {}, {} total samples.",
        config.lambda,
        config.penalty_order,
        data.y.len()
    );

    // 1. Build the large, one-time matrices for the model. This is the most complex
    //    non-iterative part, translating the model formula into matrices.
    let (x_matrix, penalty_matrix, layout) =
        internal::build_design_and_penalty_matrices(data, config)?;
    println!(
        "Constructed design matrix with {} samples and {} total coefficients.",
        x_matrix.nrows(),
        x_matrix.ncols()
    );

    // 2. Initialize the IRLS algorithm.
    let mut beta = Array1::zeros(x_matrix.ncols());
    let mut last_deviance = f64::INFINITY;
    let mut last_deviance_change = f64::INFINITY;

    // 3. Run the IRLS loop.
    for iter in 1..=config.max_iterations {
        let eta = x_matrix.dot(&beta);
        let (mu, weights, z) =
            internal::update_glm_vectors(data.y.view(), &eta, config.link_function);

        beta = internal::solve_penalized_wls(
            x_matrix.view(),
            z.view(),
            weights.view(),
            penalty_matrix.view(),
            config.lambda,
        )?;

        let deviance = internal::calculate_deviance(data.y.view(), &mu, config.link_function);
        let deviance_change = (last_deviance - deviance).abs();
        last_deviance_change = deviance_change;

        println!(
            "Iter {: >3}: Deviance = {:.6}, Change = {:.6e}",
            iter,
            deviance,
            deviance_change
        );

        if deviance_change < config.convergence_tolerance {
            println!(
                "IRLS converged after {} iterations. Final Deviance: {:.6}",
                iter,
                deviance
            );
            // On success, "un-flatten" the beta vector into the structured MappedCoefficients.
            let mapped_coefficients = internal::map_coefficients(&beta, config, &layout);
            return Ok(TrainedModel {
                config: config.clone(),
                coefficients: mapped_coefficients,
            });
        }
        last_deviance = deviance;
    }

    // If the loop finishes without converging, return an error.
    Err(EstimationError::DidNotConverge {
        max_iterations: config.max_iterations,
        last_change: last_deviance_change,
    })
}

/// Internal module for implementation details.
mod internal {
    use super::*;
    use std::ops::Range;

    /// A single source of truth for the column layout of the design matrix `X`.
    ///
    /// This struct encapsulates the complex indexing logic, ensuring that matrix
    /// construction, penalty application, and coefficient mapping are always consistent.
    pub(super) struct ModelLayout {
        /// The column index for the intercept.
        pub intercept_col: usize,
        /// Column ranges for the main effect of each PC. Indexed by PC order.
        pub pc_main_cols: Vec<Range<usize>>,
        /// Column range for the main effect of the PGS (for basis functions m > 0).
        pub pgs_main_cols: Range<usize>,
        /// Column ranges for interaction terms.
        /// Outer vec: indexed by PGS basis function (m > 0).
        /// Inner vec: indexed by PC order.
        pub interaction_cols: Vec<Vec<Range<usize>>>,
        /// The total number of columns in the design matrix / coefficients in beta.
        pub total_coeffs: usize,
    }

    impl ModelLayout {
        /// Creates a new layout based on the model configuration and the dimensions
        /// of the generated basis expansions.
        pub(super) fn new(
            pc_basis_ncols: &[usize],
            pgs_basis_ncols: usize,
        ) -> Self {
            let mut pc_main_cols = Vec::with_capacity(pc_basis_ncols.len());
            let mut interaction_cols = Vec::new();
            let mut current_col = 0;

            let intercept_col = current_col;
            current_col += 1;

            for &num_basis in pc_basis_ncols {
                pc_main_cols.push(current_col..current_col + num_basis);
                current_col += num_basis;
            }

            let num_pgs_main_coeffs = pgs_basis_ncols.saturating_sub(1);
            let pgs_main_cols = current_col..current_col + num_pgs_main_coeffs;
            current_col += num_pgs_main_coeffs;

            if num_pgs_main_coeffs > 0 {
                interaction_cols.reserve(num_pgs_main_coeffs);
                for _m_idx in 0..num_pgs_main_coeffs {
                    let mut pc_interaction_group = Vec::with_capacity(pc_basis_ncols.len());
                    for &num_basis in pc_basis_ncols {
                        pc_interaction_group.push(current_col..current_col + num_basis);
                        current_col += num_basis;
                    }
                    interaction_cols.push(pc_interaction_group);
                }
            }

            ModelLayout {
                intercept_col,
                pc_main_cols,
                pgs_main_cols,
                interaction_cols,
                total_coeffs: current_col,
            }
        }
    }

    /// Constructs the full design matrix `X` and the block-diagonal penalty matrix `S`.
    /// This is orchestrated by a `ModelLayout` to ensure correctness.
    pub(super) fn build_design_and_penalty_matrices(
        data: &TrainingData,
        config: &ModelConfig,
    ) -> Result<(Array2<f64>, Array2<f64>, ModelLayout), BasisError> {
        // --- 1. Generate all individual basis expansions ---
        let (pgs_basis, _) = create_bspline_basis(
            data.p.view(),
            Some(data.p.view()),
            config.pgs_range,
            config.pgs_basis_config.num_knots,
            config.pgs_basis_config.degree,
        )?;

        let pc_bases: Vec<Array2<f64>> = data
            .pcs
            .axis_iter(Axis(1))
            .zip(config.pc_ranges.iter().zip(config.pc_basis_configs.iter()))
            .map(|(pc_col, (&range, pc_conf))| {
                create_bspline_basis(
                    pc_col.view(),
                    Some(pc_col.view()),
                    range,
                    pc_conf.num_knots,
                    pc_conf.degree,
                )
                .map(|(basis, _)| basis)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // --- 2. Create the layout as the single source of truth for column indices ---
        let pc_basis_ncols: Vec<usize> = pc_bases.iter().map(|b| b.ncols()).collect();
        let layout = ModelLayout::new(&pc_basis_ncols, pgs_basis.ncols());

        // --- 3. Assemble the full design matrix X, guided by the layout ---
        let mut x_matrix = Array2::zeros((data.y.len(), layout.total_coeffs));

        x_matrix.column_mut(layout.intercept_col).fill(1.0);

        for (i, pc_basis) in pc_bases.iter().enumerate() {
            x_matrix
                .slice_mut(s![.., layout.pc_main_cols[i].clone()])
                .assign(pc_basis);
        }

        let pgs_main_basis = pgs_basis.slice(s![.., 1..]);
        x_matrix
            .slice_mut(s![.., layout.pgs_main_cols.clone()])
            .assign(&pgs_main_basis);
        
        for (m, pgs_basis_col) in pgs_main_basis.axis_iter(Axis(1)).enumerate() {
            for (i, pc_basis) in pc_bases.iter().enumerate() {
                let interaction_block = &pgs_basis_col.to_shape((data.y.len(), 1)).unwrap() * pc_basis;
                x_matrix
                    .slice_mut(s![.., layout.interaction_cols[m][i].clone()])
                    .assign(&interaction_block);
            }
        }

        // --- 4. Assemble the block-diagonal penalty matrix S, also guided by the layout ---
        let mut penalty_matrix = Array2::zeros((layout.total_coeffs, layout.total_coeffs));

        for (i, pc_conf) in config.pc_basis_configs.iter().enumerate() {
            let p_mat = create_difference_penalty_matrix(pc_basis_ncols[i], config.penalty_order)?;
            penalty_matrix
                .slice_mut(s![layout.pc_main_cols[i].clone(), layout.pc_main_cols[i].clone()])
                .assign(&p_mat);
        }

        for (m, _pgs_basis_col) in pgs_main_basis.axis_iter(Axis(1)).enumerate() {
            for (i, pc_conf) in config.pc_basis_configs.iter().enumerate() {
                let p_mat = create_difference_penalty_matrix(pc_basis_ncols[i], config.penalty_order)?;
                penalty_matrix
                    .slice_mut(s![layout.interaction_cols[m][i].clone(), layout.interaction_cols[m][i].clone()])
                    .assign(&p_mat);
            }
        }

        Ok((x_matrix, penalty_matrix, layout))
    }

    /// Solves the core penalized weighted least squares system for one IRLS step.
    pub(super) fn solve_penalized_wls(
        x: ArrayView2<f64>,
        z: ArrayView1<f64>,
        w: ArrayView1<f64>,
        s: ArrayView2<f64>,
        lambda: f64,
    ) -> Result<Array1<f64>, EstimationError> {
        let x_t = x.t();
        let x_t_w = &x_t * w;
        let lhs = x_t_w.dot(&x) + &(s * lambda);
        let rhs = x_t_w.dot(&z);
        lhs.solve_into(rhs)
            .map_err(EstimationError::MatrixInversionFailed)
    }

    /// Calculates `mu`, `w`, and `z` for a given link function and linear predictor `eta`.
    pub(super) fn update_glm_vectors(
        y: ArrayView1<f64>,
        eta: &Array1<f64>,
        link: LinkFunction,
    ) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        const MIN_WEIGHT: f64 = 1e-6;

        match link {
            LinkFunction::Logit => {
                let mu = eta.mapv(|e| 1.0 / (1.0 + (-e).exp())); // Sigmoid
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

    /// Calculates the deviance of the model in a numerically stable way.
    pub(super) fn calculate_deviance(
        y: ArrayView1<f64>,
        mu: &Array1<f64>,
        link: LinkFunction,
    ) -> f64 {
        const EPS: f64 = 1e-9;
        match link {
            LinkFunction::Logit => {
                let total_residual =
                    ndarray::Zip::from(y)
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

    /// Converts the flat coefficient vector `beta` into a structured `MappedCoefficients`.
    pub(super) fn map_coefficients(
        beta: &Array1<f64>,
        config: &ModelConfig,
        layout: &ModelLayout,
    ) -> MappedCoefficients {
        let intercept = beta[layout.intercept_col];

        let mut pcs = HashMap::new();
        for (i, pc_name) in config.pc_names.iter().enumerate() {
            pcs.insert(
                pc_name.clone(),
                beta.slice(s![layout.pc_main_cols[i].clone()]).to_vec(),
            );
        }

        let pgs = beta.slice(s![layout.pgs_main_cols.clone()]).to_vec();

        let mut interaction_effects = HashMap::new();
        for m in 0..layout.interaction_cols.len() {
            let pgs_key = format!("pgs_B{}", m + 1); // B1, B2, ...
            let mut pc_map = HashMap::new();
            for (i, pc_name) in config.pc_names.iter().enumerate() {
                pc_map.insert(
                    pc_name.clone(),
                    beta.slice(s![layout.interaction_cols[m][i].clone()]).to_vec(),
                );
            }
            interaction_effects.insert(pgs_key, pc_map);
        }

        MappedCoefficients {
            intercept,
            main_effects: MainEffects { pgs, pcs },
            interaction_effects,
        }
    }
}

// --- Unit Tests ---
#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::BasisConfig;
    use ndarray::array;

    #[test]
    fn test_calculate_deviance_logit_stable() {
        let y = array![0.0, 1.0, 0.5];
        let mu = array![0.1, 0.9, 0.5];
        let deviance = internal::calculate_deviance(y.view(), &mu, LinkFunction::Logit);

        let d1 = 2.0 * (1.0f64.ln() - 0.0f64.ln() + (1.0 / 0.9).ln());
        let d2 = 2.0 * ((1.0 / 0.9).ln() + 1.0f64.ln() - 0.0f64.ln());
        let expected_deviance = 2.0 * ((1.0/0.9).ln() + (1.0/0.9).ln());
        assert!((deviance - expected_deviance).abs() < 1e-5);

        let y_zero = array![0.0];
        let mu_small = array![1e-12];
        let deviance_zero = internal::calculate_deviance(y_zero.view(), &mu_small, LinkFunction::Logit);
        assert!(deviance_zero.abs() < 1e-9);
    }

    #[test]
    fn test_calculate_deviance_identity() {
        let y = array![1.0, 2.0, 3.0];
        let mu = array![1.1, 2.2, 2.9];
        let deviance = internal::calculate_deviance(y.view(), &mu, LinkFunction::Identity);
        assert!((deviance - 0.06).abs() < 1e-9);
    }

    // Helper to create a simple ModelConfig for testing matrix/coefficient logic.
    fn create_test_config() -> ModelConfig {
        ModelConfig {
            link_function: LinkFunction::Logit,
            penalty_order: 2,
            lambda: 1.0,
            convergence_tolerance: 1e-6,
            max_iterations: 50,
            pgs_basis_config: BasisConfig { num_knots: 2, degree: 2 }, // 2+2+1=5 basis functions
            pc_basis_configs: vec![BasisConfig { num_knots: 1, degree: 2 }], // 1+2+1=4 basis functions
            pgs_range: (0.0, 1.0),
            pc_ranges: vec![(0.0, 1.0)],
            pc_names: vec!["PC1".to_string()],
        }
    }

    #[test]
    fn test_model_layout() {
        let pc_ncols = &[4]; // From config: 1 knot + 2 degree + 1 = 4
        let pgs_ncols = 5; // From config: 2 knots + 2 degree + 1 = 5
        let layout = internal::ModelLayout::new(pc_ncols, pgs_ncols);
        
        // 1 (intercept) + 4 (PC main) + 4 (PGS main) + 4*4 (interactions) = 25
        assert_eq!(layout.total_coeffs, 1 + 4 + 4 + 16);

        assert_eq!(layout.intercept_col, 0);
        assert_eq!(layout.pc_main_cols, vec![1..5]);
        assert_eq!(layout.pgs_main_cols, 5..9);
        assert_eq!(layout.interaction_cols.len(), 4); // 4 PGS main effect terms
        assert_eq!(layout.interaction_cols[0], vec![9..13]);
        assert_eq!(layout.interaction_cols[3], vec![21..25]);
    }
    
    #[test]
    fn test_map_coefficients() {
        let config = create_test_config();
        let pc_ncols = &[4];
        let pgs_ncols = 5;
        let layout = internal::ModelLayout::new(pc_ncols, pgs_ncols);
        
        let beta = Array1::range(0.0, layout.total_coeffs as f64, 1.0); // 0.0, 1.0, ..., 24.0

        let mapped = internal::map_coefficients(&beta, &config, &layout);

        assert_eq!(mapped.intercept, 0.0);
        assert_eq!(mapped.main_effects.pcs["PC1"], vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(mapped.main_effects.pgs, vec![5.0, 6.0, 7.0, 8.0]);
        assert_eq!(mapped.interaction_effects.len(), 4);
        assert_eq!(mapped.interaction_effects["pgs_B1"]["PC1"], vec![9.0, 10.0, 11.0, 12.0]);
        assert_eq!(mapped.interaction_effects["pgs_B4"]["PC1"], vec![21.0, 22.0, 23.0, 24.0]);
    }
}
