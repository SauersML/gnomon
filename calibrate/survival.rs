//! Survival model infrastructure implementing the Royston–Parmar style
//! parameterisation described in `plan/survival.md`.
//!
//! The implementation intentionally mirrors the structure of the GAM
//! pipeline so that the existing PIRLS, basis, and calibration utilities can
//! operate on survival models without bespoke branches.  The focus here is on
//! the data structures and deterministic transformations required for
//! training and scoring a cause-specific cumulative hazard.

use crate::calibrate::basis::BasisDescriptor;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::pirls::{WorkingModel, WorkingState};
use ndarray::{s, Array1, Array2, ArrayBase, ArrayView1, ArrayView2, Axis, Data, Ix1, Ix2};
use serde::{Deserialize, Serialize};

/// Guard applied to log-age transform.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AgeTransform {
    pub a_min: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn new<T>(age_entry: &ArrayBase<T, Ix1>, age_exit: &ArrayBase<T, Ix1>) -> Self
    where
        T: Data<Elem = f64>,
    {
        let a_min_entry = age_entry
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let a_min_exit = age_exit
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let a_min = a_min_entry.min(a_min_exit);
        let delta = 0.1;
        AgeTransform { a_min, delta }
    }

    pub fn transform(&self, age: f64) -> f64 {
        (age - self.a_min + self.delta).ln()
    }

    pub fn transform_vec<T>(&self, age: &ArrayBase<T, Ix1>) -> Array1<f64>
    where
        T: Data<Elem = f64>,
    {
        age.iter()
            .map(|&a| self.transform(a))
            .collect::<Array1<f64>>()
    }

    pub fn derivative(&self, age: f64) -> f64 {
        1.0 / (age - self.a_min + self.delta)
    }
}

/// Linear reference constraint removing the null direction of the baseline
/// hazard spline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceConstraint {
    pub z: Array2<f64>,
}

impl ReferenceConstraint {
    pub fn apply<T>(&self, mat: ArrayBase<T, Ix2>) -> Array2<f64>
    where
        T: Data<Elem = f64>,
    {
        mat.dot(&self.z)
    }
}

/// Layout caches used by the survival working model.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub penalties: Vec<Array2<f64>>,
    pub monotonicity_grid_derivative: Option<Array2<f64>>,
    pub monotonicity_grid_weights: Option<Array1<f64>>,
}

impl SurvivalLayout {
    pub fn total_columns(&self) -> usize {
        let mut cols = self.baseline_exit.ncols() + self.static_covariates.ncols();
        if let Some(tv_exit) = &self.time_varying_exit {
            cols += tv_exit.ncols();
        }
        cols
    }

    pub fn assemble_exit_design(&self) -> Array2<f64> {
        let mut columns: Vec<Array2<f64>> = vec![self.baseline_exit.clone()];
        if let Some(tv_exit) = &self.time_varying_exit {
            columns.push(tv_exit.clone());
        }
        columns.push(self.static_covariates.clone());
        hstack(columns)
    }

    pub fn assemble_entry_design(&self) -> Array2<f64> {
        let mut columns: Vec<Array2<f64>> = vec![self.baseline_entry.clone()];
        if let Some(tv_entry) = &self.time_varying_entry {
            columns.push(tv_entry.clone());
        }
        columns.push(self.static_covariates.clone());
        hstack(columns)
    }

    pub fn assemble_exit_derivative(&self) -> Array2<f64> {
        let mut columns: Vec<Array2<f64>> = vec![self.baseline_derivative_exit.clone()];
        if let Some(tv_deriv) = &self.time_varying_derivative_exit {
            columns.push(tv_deriv.clone());
        }
        columns.push(Array2::zeros((self.static_covariates.nrows(), self.static_covariates.ncols())));
        hstack(columns)
    }
}

fn hstack(blocks: Vec<Array2<f64>>) -> Array2<f64> {
    let rows = blocks.first().map(|b| b.nrows()).unwrap_or(0);
    let cols = blocks.iter().map(|b| b.ncols()).sum();
    let mut out = Array2::zeros((rows, cols));
    let mut col = 0;
    for block in blocks {
        let width = block.ncols();
        out
            .slice_mut(s![.., col..col + width])
            .assign(&block);
        col += width;
    }
    out
}

/// Core training bundle for survival data.
#[derive(Debug, Clone)]
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub covariates: Array2<f64>,
}

impl SurvivalTrainingData {
    pub fn validate(&self) -> Result<(), EstimationError> {
        if self.age_entry.len() != self.age_exit.len() {
            return Err(EstimationError::InvalidInput(
                "age_entry and age_exit length mismatch".to_string(),
            ));
        }
        if self.event_target.len() != self.age_entry.len()
            || self.event_competing.len() != self.age_entry.len()
        {
            return Err(EstimationError::InvalidInput(
                "event indicators length mismatch".to_string(),
            ));
        }
        for i in 0..self.age_entry.len() {
            if !(self.age_entry[i] < self.age_exit[i]) {
                return Err(EstimationError::InvalidInput(format!(
                    "subject {} has age_entry >= age_exit",
                    i
                )));
            }
            if self.event_target[i] + self.event_competing[i] > 1 {
                return Err(EstimationError::InvalidInput(format!(
                    "subject {} has both events recorded",
                    i
                )));
            }
        }
        Ok(())
    }
}

/// Lightweight view bundle used for scoring.
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
}

/// Working model for the Royston–Parmar cumulative hazard parameterisation.
pub struct WorkingModelSurvival {
    pub layout: SurvivalLayout,
    pub training: SurvivalTrainingData,
    pub epsilon: f64,
    pub barrier_weight: f64,
}

impl WorkingModelSurvival {
    pub fn new(layout: SurvivalLayout, training: SurvivalTrainingData) -> Result<Self, EstimationError> {
        training.validate()?;
        Ok(WorkingModelSurvival {
            layout,
            training,
            epsilon: 1e-10,
            barrier_weight: 1e-4,
        })
    }

    fn assemble_designs(&self) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let x_exit = self.layout.assemble_exit_design();
        let x_entry = self.layout.assemble_entry_design();
        let d_exit = self.layout.assemble_exit_derivative();
        (x_exit, x_entry, d_exit)
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
        let (x_exit, x_entry, d_exit) = self.assemble_designs();
        let eta_exit = x_exit.dot(beta);
        let eta_entry = x_entry.dot(beta);
        let h_exit = eta_exit.mapv(f64::exp);
        let h_entry = eta_entry.mapv(f64::exp);
        let delta_h = &h_exit - &h_entry;
        let d_eta_exit = d_exit.dot(beta);

        let mut loglik = 0.0;
        let mut gradient = Array1::<f64>::zeros(beta.len());
        let mut hessian = Array2::<f64>::zeros((beta.len(), beta.len()));

        for i in 0..self.training.age_exit.len() {
            let d = f64::from(self.training.event_target[i]);
            let w = self.training.sample_weight[i];
            let guard = d_eta_exit[i].max(self.epsilon);
            let x_exit_vec = x_exit.slice(s![i, ..]).to_owned();
            let x_entry_vec = x_entry.slice(s![i, ..]).to_owned();
            let d_exit_vec = d_exit.slice(s![i, ..]).to_owned();
            let h_exit_i = h_exit[i];
            let h_entry_i = h_entry[i];
            let delta_h_i = delta_h[i];

            loglik += w * (d * (eta_exit[i] + guard.ln()) - delta_h_i);

            let mut x_tilde = x_exit_vec.clone();
            x_tilde += &(&d_exit_vec / guard);
            gradient += &(x_tilde.clone() * (w * d));
            gradient -= &(x_exit_vec.clone() * (w * h_exit_i));
            gradient += &(x_entry_vec.clone() * (w * h_entry_i));

            let derivative_basis = &d_exit_vec / guard;
            rank1_update(&mut hessian, &derivative_basis, w * d);
            rank1_update(&mut hessian, &x_exit_vec, w * h_exit_i);
            rank1_update(&mut hessian, &x_entry_vec, w * h_entry_i);
        }

        let mut penalty_value = 0.0;
        if let Some(grid) = &self.layout.monotonicity_grid_derivative {
            let weights = self.layout.monotonicity_grid_weights.as_ref();
            for (i, row) in grid.axis_iter(Axis(0)).enumerate() {
                let row_vec = row.to_owned();
                let weight = weights.map(|w| w[i]).unwrap_or(1.0);
                let derivative = row_vec.dot(beta);
                let soft_arg = -derivative;
                let softplus = (1.0 + soft_arg.exp()).ln();
                let sigma = 1.0 / (1.0 + derivative.exp());
                let barrier_weight = self.barrier_weight * weight;
                penalty_value += barrier_weight * softplus;
                let gradient_scale = -barrier_weight * sigma;
                gradient += &(row_vec.clone() * gradient_scale);
                let curvature = barrier_weight * sigma * (1.0 - sigma);
                rank1_update(&mut hessian, &row_vec, curvature);
            }
        }

        let deviance = -2.0 * loglik + 2.0 * penalty_value;

        WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        }
    }
}

fn rank1_update(target: &mut Array2<f64>, vector: &Array1<f64>, weight: f64) {
    if weight == 0.0 {
        return;
    }
    let len = vector.len();
    for i in 0..len {
        let vi = vector[i] * weight;
        for j in 0..len {
            target[[i, j]] += vi * vector[j];
        }
    }
}

/// Survival-specific scoring artifact capturing fitted coefficients and
/// metadata required for deterministic predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HessianFactor {
    Observed {
        ldlt_factor: Array2<f64>,
        permutation: Vec<usize>,
        inertia: (usize, usize, usize),
    },
    Expected {
        cholesky_factor: Array2<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: PenaltyDescriptor,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<HessianFactor>,
}

impl SurvivalModelArtifacts {
    pub fn cumulative_hazard(&self, age: f64, design: ArrayView1<'_, f64>) -> f64 {
        debug_assert!(age.is_finite(), "age must be finite for hazard evaluation");
        let eta = design.dot(&self.coefficients);
        eta.exp()
    }

    pub fn cumulative_incidence(&self, age: f64, design: ArrayView1<'_, f64>) -> f64 {
        let h = self.cumulative_hazard(age, design);
        1.0 - (-h).exp()
    }

    pub fn conditional_absolute_risk(
        &self,
        t0: f64,
        t1: f64,
        design_t0: ArrayView1<'_, f64>,
        design_t1: ArrayView1<'_, f64>,
        cif_competing_t0: f64,
    ) -> f64 {
        let cif_t0 = self.cumulative_incidence(t0, design_t0);
        let cif_t1 = self.cumulative_incidence(t1, design_t1);
        let delta = (cif_t1 - cif_t0).max(0.0);
        let denom = (1.0 - cif_t0 - cif_competing_t0).max(1e-12);
        (delta / denom).min(1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateLayout {
    pub column_names: Vec<String>,
    pub column_ranges: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyDescriptor {
    pub blocks: Vec<Array2<f64>>,
}

