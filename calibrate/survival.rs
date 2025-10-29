#![allow(dead_code)]

//! Survival Royston–Parmar model infrastructure.
//!
//! This module wires together data ingestion, layout caching, likelihood
//! evaluation, scoring, and artifact persistence for the Royston–Parmar
//! survival family described in `plan/survival.md`. The implementation
//! focuses on providing numerically stable building blocks that integrate
//! with the existing calibration pipeline while remaining standalone so it
//! can be evolved independently of the GAM paths.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::sync::Arc;

/// Representation of the guarded age transform that maps raw age inputs to
/// the log-age scale while ensuring numerical stability close to the minimum
/// observed age.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    /// Minimum observed age (entry) in the training cohort.
    pub minimum_age: f64,
    /// Positive guard parameter ensuring strictly positive arguments for the
    /// logarithm.
    pub delta: f64,
}

impl AgeTransform {
    /// Construct a new age transform.
    pub fn new(minimum_age: f64, delta: f64) -> Self {
        debug_assert!(delta.is_finite() && delta > 0.0);
        Self { minimum_age, delta }
    }

    /// Apply the guarded transform `u = log(age - a_min + δ)`.
    pub fn transform(&self, age: f64) -> f64 {
        let shifted = age - self.minimum_age + self.delta;
        debug_assert!(shifted > 0.0, "Age transform requires positive argument");
        shifted.ln()
    }

    /// Compute the derivative `∂u/∂age` used when translating derivatives back
    /// to the original age scale.
    pub fn derivative(&self, age: f64) -> f64 {
        1.0 / (age - self.minimum_age + self.delta)
    }
}

/// Stores a linear reference constraint transform that removes the null space
/// of the baseline spline so predictions remain anchored to a reproducible
/// reference point.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceConstraint {
    pub z: Array2<f64>,
    pub reference_age: f64,
}

impl ReferenceConstraint {
    pub fn new(z: Array2<f64>, reference_age: f64) -> Self {
        Self { z, reference_age }
    }
}

/// Penalty metadata describing the penalty matrix applied to a contiguous
/// coefficient range.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyBlock {
    pub columns: Range<usize>,
    pub matrix: Array2<f64>,
    pub lambda: f64,
}

/// Collection of penalty blocks used during estimation. The PIRLS solver will
/// combine these with the observed information.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }

    pub fn total_dimension(&self) -> usize {
        self.blocks
            .iter()
            .map(|block| block.columns.end)
            .max()
            .unwrap_or(0)
    }
}

/// Guarded cache for the spline basis evaluations and covariate designs. The
/// matrices are stored in their constrained form so PIRLS can operate directly
/// without additional transforms.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalLayout {
    pub baseline_entry: Array2<f64>,
    pub baseline_exit: Array2<f64>,
    pub baseline_derivative_exit: Array2<f64>,
    pub time_varying_entry: Option<Array2<f64>>,
    pub time_varying_exit: Option<Array2<f64>>,
    pub time_varying_derivative_exit: Option<Array2<f64>>,
    pub static_covariates: Array2<f64>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub penalties: PenaltyBlocks,
}

impl SurvivalLayout {
    pub fn num_observations(&self) -> usize {
        self.baseline_exit.nrows()
    }

    pub fn coefficient_dimension(&self) -> usize {
        let mut dim = self.baseline_exit.ncols();
        if let Some(tv) = &self.time_varying_exit {
            dim += tv.ncols();
        }
        dim += self.static_covariates.ncols();
        dim
    }

    fn assemble_designs(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let n = self.num_observations();
        let p = self.coefficient_dimension();

        let mut entry = Array2::zeros((n, p));
        let mut exit = Array2::zeros((n, p));
        let mut derivative_exit = Array2::zeros((n, p));

        let baseline_cols = self.baseline_exit.ncols();
        entry
            .slice_mut(s![.., 0..baseline_cols])
            .assign(&self.baseline_entry);
        exit.slice_mut(s![.., 0..baseline_cols])
            .assign(&self.baseline_exit);
        derivative_exit
            .slice_mut(s![.., 0..baseline_cols])
            .assign(&self.baseline_derivative_exit);

        let mut offset = baseline_cols;
        if let Some(tv_entry) = &self.time_varying_entry {
            let tv_exit = self.time_varying_exit.as_ref().expect("exit tv missing");
            let tv_deriv = self
                .time_varying_derivative_exit
                .as_ref()
                .expect("derivative tv missing");
            let cols = tv_entry.ncols();
            entry
                .slice_mut(s![.., offset..offset + cols])
                .assign(tv_entry);
            exit.slice_mut(s![.., offset..offset + cols])
                .assign(tv_exit);
            derivative_exit
                .slice_mut(s![.., offset..offset + cols])
                .assign(tv_deriv);
            offset += cols;
        }

        let static_cols = self.static_covariates.ncols();
        entry
            .slice_mut(s![.., offset..offset + static_cols])
            .assign(&self.static_covariates);
        exit.slice_mut(s![.., offset..offset + static_cols])
            .assign(&self.static_covariates);
        // Static covariates have zero derivative contribution.

        (entry, exit, derivative_exit)
    }

    fn combine_exit_row(&self, covariates: &SurvivalCovariateAtAge) -> Array1<f64> {
        let p = self.coefficient_dimension();
        let mut row = Array1::zeros(p);
        let mut offset = 0;

        let base_cols = covariates.baseline.len();
        row.slice_mut(s![offset..offset + base_cols])
            .assign(&covariates.baseline);
        offset += base_cols;

        if let Some(tv) = &covariates.time_varying {
            row.slice_mut(s![offset..offset + tv.len()]).assign(tv);
            offset += tv.len();
        }

        let static_cols = covariates.static_covariates.len();
        row.slice_mut(s![offset..offset + static_cols])
            .assign(&covariates.static_covariates);

        row
    }

    fn combine_derivative_row(&self, covariates: &SurvivalCovariateAtAge) -> Array1<f64> {
        let p = self.coefficient_dimension();
        let mut row = Array1::zeros(p);
        let mut offset = 0;

        let base_cols = covariates.baseline_derivative.len();
        row.slice_mut(s![offset..offset + base_cols])
            .assign(&covariates.baseline_derivative);
        offset += base_cols;

        if let Some(tv) = &covariates.time_varying_derivative {
            row.slice_mut(s![offset..offset + tv.len()]).assign(tv);
        }

        row
    }
}

/// Frequency-weighted training inputs for the survival model.
#[derive(Debug, Clone, PartialEq)]
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
}

/// Prediction inputs mirror the training schema but are provided as views so
/// scoring can avoid unnecessary allocations.
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: CovariateViews<'a>,
}

/// Borrowed covariate views used when constructing prediction layouts.
pub struct CovariateViews<'a> {
    pub static_covariates: ArrayView2<'a, f64>,
    pub baseline: ArrayView2<'a, f64>,
    pub baseline_derivative: ArrayView2<'a, f64>,
    pub time_varying: Option<ArrayView2<'a, f64>>,
    pub time_varying_derivative: Option<ArrayView2<'a, f64>>,
}

/// The unified working model interface used by PIRLS.
pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
}

/// Outputs produced by a single PIRLS iteration.
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
}

/// Concrete Royston–Parmar working model implementation.
pub struct WorkingModelSurvival {
    layout: SurvivalLayout,
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    sample_weight: Array1<f64>,
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    barrier_strength: f64,
}

impl WorkingModelSurvival {
    pub fn new(
        layout: SurvivalLayout,
        event_target: Array1<u8>,
        event_competing: Array1<u8>,
        sample_weight: Array1<f64>,
    ) -> Self {
        let (design_entry, design_exit, design_derivative_exit) = layout.assemble_designs();
        Self {
            layout,
            event_target,
            event_competing,
            sample_weight,
            design_entry,
            design_exit,
            design_derivative_exit,
            barrier_strength: 1.0,
        }
    }

    fn softplus(x: f64) -> f64 {
        if x > 30.0 {
            x
        } else if x < -30.0 {
            (-x).exp()
        } else {
            (1.0 + x.exp()).ln()
        }
    }

    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            let z = (-x).exp();
            1.0 / (1.0 + z)
        } else {
            let z = x.exp();
            z / (1.0 + z)
        }
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
        let n = self.layout.num_observations();
        let p = self.layout.coefficient_dimension();

        let eta_exit = self.design_exit.dot(beta);
        let eta_entry = self.design_entry.dot(beta);
        let deta_exit = self.design_derivative_exit.dot(beta);

        let h_exit = eta_exit.mapv(f64::exp);
        let h_entry = eta_entry.mapv(f64::exp);
        let delta_h = &h_exit - &h_entry;

        let mut gradient = Array1::zeros(p);
        let mut hessian = Array2::zeros((p, p));
        let mut loglik = 0.0;

        for i in 0..n {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let x_exit = self.design_exit.row(i);
            let x_entry = self.design_entry.row(i);
            let x_deriv = self.design_derivative_exit.row(i);

            let h_exit_i = h_exit[i];
            let h_entry_i = h_entry[i];
            let delta = delta_h[i].max(1e-12);
            loglik -= weight * delta;

            // Gradient contribution from -ΔH
            let mut grad_contrib = x_exit.to_owned();
            grad_contrib *= h_exit_i;
            let mut entry_contrib = x_entry.to_owned();
            entry_contrib *= h_entry_i;
            gradient.scaled_add(-weight, &(grad_contrib - entry_contrib));

            // Hessian contribution from -ΔH
            rank_one_update(&mut hessian, x_exit, -weight * h_exit_i);
            rank_one_update(&mut hessian, x_entry, weight * h_entry_i);

            if self.event_target[i] == 1 {
                let deta = deta_exit[i];
                let guard = deta.max(1e-8);
                loglik += weight * (eta_exit[i] + guard.ln());
                let mut event_grad = x_exit.to_owned();
                event_grad += &(&x_deriv / guard);
                gradient.scaled_add(weight, &event_grad);
            }

            let barrier = Self::softplus(-deta_exit[i]);
            if barrier.is_finite() && barrier > 0.0 {
                loglik -= self.barrier_strength * weight * barrier;
                let s = Self::sigmoid(-deta_exit[i]);
                let barrier_grad = x_deriv.to_owned() * (self.barrier_strength * weight * s);
                gradient += &barrier_grad;
                let curvature = -self.barrier_strength * weight * s * (1.0 - s);
                rank_one_update(&mut hessian, x_deriv, curvature);
            }
        }

        let deviance = -2.0 * loglik;

        WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        }
    }
}

/// Rank-one update helper used when accumulating Hessian contributions.
fn rank_one_update(matrix: &mut Array2<f64>, vec: ArrayView1<'_, f64>, weight: f64) {
    if weight == 0.0 {
        return;
    }
    let len = vec.len();
    for i in 0..len {
        for j in 0..len {
            matrix[[i, j]] += weight * vec[i] * vec[j];
        }
    }
}

/// Cached covariate row evaluated at a specific age.
#[derive(Clone)]
pub struct SurvivalCovariateAtAge {
    pub baseline: Array1<f64>,
    pub baseline_derivative: Array1<f64>,
    pub time_varying: Option<Array1<f64>>,
    pub time_varying_derivative: Option<Array1<f64>>,
    pub static_covariates: Array1<f64>,
}

impl SurvivalCovariateAtAge {
    pub fn with_static(static_covariates: Array1<f64>) -> Self {
        Self {
            baseline: Array1::zeros(0),
            baseline_derivative: Array1::zeros(0),
            time_varying: None,
            time_varying_derivative: None,
            static_covariates,
        }
    }
}

/// Provider of covariate rows at arbitrary ages.
#[derive(Clone)]
pub struct Covariates {
    generator: Arc<dyn Fn(f64) -> SurvivalCovariateAtAge + Send + Sync>,
}

impl Covariates {
    pub fn new<F>(generator: F) -> Self
    where
        F: Fn(f64) -> SurvivalCovariateAtAge + Send + Sync + 'static,
    {
        Self {
            generator: Arc::new(generator),
        }
    }

    pub fn at_age(&self, age: f64) -> SurvivalCovariateAtAge {
        (self.generator)(age)
    }
}

/// Observed-information factorization metadata persisted to the artifact for
/// downstream delta-method standard errors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Inertia {
    pub negative: usize,
    pub zero: usize,
    pub positive: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermutationMatrix {
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LdltFactor {
    pub factor: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CholeskyFactor {
    pub factor: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HessianFactor {
    Observed {
        ldlt_factor: LdltFactor,
        permutation: PermutationMatrix,
        inertia: Inertia,
    },
    Expected {
        cholesky_factor: CholeskyFactor,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasisDescriptor {
    pub knots: Array1<f64>,
    pub degree: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CovariateLayout {
    pub static_columns: Range<usize>,
    pub baseline_columns: Range<usize>,
    pub time_varying_columns: Option<Range<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyDescriptor {
    pub lambdas: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub layout: SurvivalLayout,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: PenaltyDescriptor,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<HessianFactor>,
}

impl SurvivalModelArtifacts {
    fn eta_for(&self, covariates: &SurvivalCovariateAtAge) -> f64 {
        let row = self.layout.combine_exit_row(covariates);
        row.dot(&self.coefficients)
    }

    fn derivative_for(&self, covariates: &SurvivalCovariateAtAge) -> f64 {
        let row = self.layout.combine_derivative_row(covariates);
        row.dot(&self.coefficients)
    }

    pub fn cumulative_hazard(&self, age: f64, covariates: &Covariates) -> f64 {
        let cov = covariates.at_age(age);
        self.eta_for(&cov).exp()
    }

    pub fn cumulative_incidence(&self, age: f64, covariates: &Covariates) -> f64 {
        let hazard = self.cumulative_hazard(age, covariates);
        1.0 - (-hazard).exp()
    }

    pub fn conditional_absolute_risk(
        &self,
        t0: f64,
        t1: f64,
        covariates: &Covariates,
        cif_competing_t0: f64,
    ) -> f64 {
        let cif_t1 = self.cumulative_incidence(t1, covariates);
        let cif_t0 = self.cumulative_incidence(t0, covariates);
        let delta = (cif_t1 - cif_t0).max(0.0);
        let denom = (1.0 - cif_t0 - cif_competing_t0).max(1e-12);
        delta / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn simple_layout() -> SurvivalLayout {
        SurvivalLayout {
            baseline_entry: array![[1.0, 0.1], [1.0, 0.2]],
            baseline_exit: array![[1.1, 0.3], [1.0, 0.4]],
            baseline_derivative_exit: array![[0.05, 0.02], [0.06, 0.03]],
            time_varying_entry: None,
            time_varying_exit: None,
            time_varying_derivative_exit: None,
            static_covariates: array![[0.2], [0.4]],
            age_transform: AgeTransform::new(45.0, 0.1),
            reference_constraint: ReferenceConstraint::new(Array2::eye(2), 50.0),
            penalties: PenaltyBlocks::new(vec![PenaltyBlock {
                columns: 0..2,
                matrix: Array2::eye(2),
                lambda: 1.0,
            }]),
        }
    }

    #[test]
    fn working_model_basic_shapes() {
        let layout = simple_layout();
        let event_target = array![1u8, 0u8];
        let event_competing = array![0u8, 1u8];
        let weights = array![1.0, 1.0];
        let mut model =
            WorkingModelSurvival::new(layout.clone(), event_target, event_competing, weights);
        let beta = array![0.1, -0.2, 0.3];
        let state = model.update(&beta);
        assert_eq!(state.eta.len(), layout.num_observations());
        assert_eq!(state.gradient.len(), layout.coefficient_dimension());
        assert_eq!(state.hessian.nrows(), layout.coefficient_dimension());
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn hazard_monotonic_in_age() {
        let layout = simple_layout();
        let coefficients = array![0.1, -0.1, 0.05];
        let artifact = SurvivalModelArtifacts {
            coefficients: coefficients.clone(),
            layout: layout.clone(),
            age_basis: BasisDescriptor {
                knots: array![0.0, 1.0, 2.0],
                degree: 3,
            },
            time_varying_basis: None,
            static_covariate_layout: CovariateLayout {
                static_columns: 2..3,
                baseline_columns: 0..2,
                time_varying_columns: None,
            },
            penalties: PenaltyDescriptor { lambdas: vec![1.0] },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
        };

        let covariates = Covariates::new(|age: f64| {
            let scale = 1.0 + 0.01 * (age - 50.0);
            SurvivalCovariateAtAge {
                baseline: array![scale, 0.2 * scale],
                baseline_derivative: array![0.01 * scale, 0.002 * scale],
                time_varying: None,
                time_varying_derivative: None,
                static_covariates: array![0.25],
            }
        });

        let hazard_55 = artifact.cumulative_hazard(55.0, &covariates);
        let hazard_60 = artifact.cumulative_hazard(60.0, &covariates);
        assert!(hazard_60 >= hazard_55);
    }

    #[test]
    fn conditional_risk_monotone() {
        let layout = simple_layout();
        let coefficients = array![0.1, -0.1, 0.05];
        let artifact = SurvivalModelArtifacts {
            coefficients: coefficients.clone(),
            layout,
            age_basis: BasisDescriptor {
                knots: array![0.0, 1.0, 2.0],
                degree: 3,
            },
            time_varying_basis: None,
            static_covariate_layout: CovariateLayout {
                static_columns: 2..3,
                baseline_columns: 0..2,
                time_varying_columns: None,
            },
            penalties: PenaltyDescriptor { lambdas: vec![1.0] },
            age_transform: AgeTransform::new(45.0, 0.1),
            reference_constraint: ReferenceConstraint::new(Array2::eye(2), 50.0),
            hessian_factor: None,
        };

        let covariates = Covariates::new(|age: f64| {
            let scale = 1.0 + 0.02 * (age - 50.0);
            SurvivalCovariateAtAge {
                baseline: array![scale, 0.3 * scale],
                baseline_derivative: array![0.02 * scale, 0.006 * scale],
                time_varying: None,
                time_varying_derivative: None,
                static_covariates: array![0.4],
            }
        });

        let risk_short = artifact.conditional_absolute_risk(50.0, 55.0, &covariates, 0.0);
        let risk_long = artifact.conditional_absolute_risk(50.0, 65.0, &covariates, 0.0);
        assert!(risk_long >= risk_short);
    }
}
