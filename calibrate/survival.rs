use crate::calibrate::basis::BasisError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use thiserror::Error;

/// Minimum derivative guard applied when evaluating the log hazard.
const DEFAULT_DERIVATIVE_GUARD: f64 = 1e-8;
/// Default weight applied to the softplus barrier on negative derivatives.
const DEFAULT_BARRIER_WEIGHT: f64 = 1e-4;
/// Barrier scaling used inside the softplus argument.
const DEFAULT_BARRIER_SCALE: f64 = 1.0;

/// Small epsilon applied to denominators when computing conditional risks.
const DEFAULT_RISK_EPSILON: f64 = 1e-12;

/// Errors surfaced while validating survival-specific data structures.
#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("age_entry must be strictly less than age_exit for every subject")]
    InvalidAgeOrder,
    #[error("event_target and event_competing indicators must be mutually exclusive")]
    ConflictingEvents,
    #[error("event indicators must be 0 or 1")]
    InvalidEventFlag,
    #[error("sample weights must be finite and non-negative")]
    InvalidSampleWeight,
    #[error("age values must be finite")]
    NonFiniteAge,
    #[error("covariate matrices must have the same number of rows as the age vectors")]
    CovariateDimensionMismatch,
    #[error("design matrix columns do not match the coefficient vector length")]
    DesignDimensionMismatch,
    #[error("derivative design matrix must match the baseline design dimensions")]
    DerivativeDimensionMismatch,
    #[error("penalty matrix must be square")]
    NonSquarePenalty,
    #[error("penalty matrices must match the coefficient dimension")]
    PenaltyDimensionMismatch,
    #[error("basis evaluation failed: {0}")]
    Basis(#[from] BasisError),
}

/// Frequency-weighted survival training bundle.
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
    /// Validate the consistency of the training data and return the number of observations.
    pub fn validate(&self) -> Result<usize, SurvivalError> {
        let n = self.age_entry.len();
        if self.age_exit.len() != n
            || self.event_target.len() != n
            || self.event_competing.len() != n
            || self.sample_weight.len() != n
            || self.covariates.nrows() != n
        {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }

        for i in 0..n {
            let entry = self.age_entry[i];
            let exit = self.age_exit[i];
            if !entry.is_finite() || !exit.is_finite() {
                return Err(SurvivalError::NonFiniteAge);
            }
            if !(entry < exit) {
                return Err(SurvivalError::InvalidAgeOrder);
            }

            let target = self.event_target[i];
            let competing = self.event_competing[i];
            if target > 1 || competing > 1 {
                return Err(SurvivalError::InvalidEventFlag);
            }
            if target == 1 && competing == 1 {
                return Err(SurvivalError::ConflictingEvents);
            }

            let weight = self.sample_weight[i];
            if !weight.is_finite() || weight < 0.0 {
                return Err(SurvivalError::InvalidSampleWeight);
            }
        }

        Ok(n)
    }
}

/// Prediction-time input bundle for survival models.
#[derive(Debug, Clone)]
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
}

/// Guarded log-age transform reused between training and scoring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub a_min: f64,
    pub delta: f64,
}

impl AgeTransform {
    /// Construct a transform from the supplied training ages using the provided guard.
    pub fn from_training(ages: &Array1<f64>, delta: f64) -> Result<Self, SurvivalError> {
        if ages.is_empty() {
            return Err(SurvivalError::NonFiniteAge);
        }
        let mut min_age = f64::INFINITY;
        for &value in ages.iter() {
            if !value.is_finite() {
                return Err(SurvivalError::NonFiniteAge);
            }
            if value < min_age {
                min_age = value;
            }
        }
        Ok(Self {
            a_min: min_age,
            delta,
        })
    }

    /// Apply the guarded log-age transform to a single age value.
    pub fn apply(&self, age: f64) -> f64 {
        (age - self.a_min + self.delta).ln()
    }

    /// Derivative of the transform with respect to age.
    pub fn derivative(&self, age: f64) -> f64 {
        1.0 / (age - self.a_min + self.delta)
    }

    /// Apply the transform to an array of ages.
    pub fn apply_array(&self, ages: &Array1<f64>) -> Array1<f64> {
        ages.map(|&age| self.apply(age))
    }
}

/// Linear constraint applied to the baseline spline to remove the null direction.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceConstraint {
    pub transform: Array2<f64>,
}

impl ReferenceConstraint {
    /// Apply the stored transform to a design matrix.
    pub fn apply(&self, design: &Array2<f64>) -> Array2<f64> {
        design.dot(&self.transform)
    }
}

/// Penalty metadata stored for each smooth block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyMatrixBlock {
    pub matrix: Array2<f64>,
    pub lambda: f64,
}

impl PenaltyMatrixBlock {
    pub fn validate(&self, expected_dim: usize) -> Result<(), SurvivalError> {
        let (rows, cols) = self.matrix.dim();
        if rows != cols {
            return Err(SurvivalError::NonSquarePenalty);
        }
        if rows != expected_dim {
            return Err(SurvivalError::PenaltyDimensionMismatch);
        }
        Ok(())
    }
}

/// Collection of penalty blocks active for the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyMatrixBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyMatrixBlock>) -> Self {
        Self { blocks }
    }

    pub fn validate(&self, expected_dim: usize) -> Result<(), SurvivalError> {
        for block in &self.blocks {
            block.validate(expected_dim)?;
        }
        Ok(())
    }

    fn gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut grad = Array1::zeros(beta.len());
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            grad += &(block.lambda * block.matrix.dot(beta));
        }
        grad
    }

    fn hessian(&self, expected_dim: usize) -> Array2<f64> {
        let mut hessian = Array2::zeros((expected_dim, expected_dim));
        if expected_dim == 0 {
            return hessian;
        }
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            debug_assert_eq!(
                block.matrix.nrows(),
                expected_dim,
                "penalty dimension mismatch"
            );
            hessian += &(block.lambda * &block.matrix);
        }
        hessian
    }

    fn deviance(&self, beta: &Array1<f64>) -> f64 {
        let mut value = 0.0;
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let quad = beta.dot(&block.matrix.dot(beta));
            value += block.lambda * quad;
        }
        value
    }
}

/// Column partition describing where each design block lives within the coefficient vector.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnPartition {
    pub baseline: Range<usize>,
    pub time_varying: Option<Range<usize>>,
    pub static_covariates: Option<Range<usize>>,
}

impl ColumnPartition {
    fn validate(&self, total_dim: usize) -> Result<(), SurvivalError> {
        if self.baseline.start != 0 {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        if self.baseline.end > total_dim {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        let mut last = self.baseline.end;
        if let Some(tv) = &self.time_varying {
            if tv.start != last || tv.end > total_dim {
                return Err(SurvivalError::DesignDimensionMismatch);
            }
            last = tv.end;
        }
        if let Some(static_block) = &self.static_covariates {
            if static_block.start != last || static_block.end > total_dim {
                return Err(SurvivalError::DesignDimensionMismatch);
            }
            last = static_block.end;
        }
        if last != total_dim {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        Ok(())
    }
}

/// Cached design matrices and transforms required to evaluate the model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalLayout {
    pub baseline_entry: Array2<f64>,
    pub baseline_exit: Array2<f64>,
    pub baseline_derivative_exit: Array2<f64>,
    pub time_varying_entry: Option<Array2<f64>>,
    pub time_varying_exit: Option<Array2<f64>>,
    pub time_varying_derivative_exit: Option<Array2<f64>>,
    pub static_covariates: Option<Array2<f64>>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub penalties: PenaltyBlocks,
    pub column_partition: ColumnPartition,
}

impl SurvivalLayout {
    pub fn validate(&self) -> Result<(), SurvivalError> {
        let n = self.baseline_entry.nrows();
        if self.baseline_exit.nrows() != n
            || self.baseline_derivative_exit.nrows() != n
            || self.baseline_exit.ncols() != self.baseline_entry.ncols()
            || self.baseline_derivative_exit.ncols() != self.baseline_entry.ncols()
        {
            return Err(SurvivalError::DerivativeDimensionMismatch);
        }

        if let Some(tv_entry) = &self.time_varying_entry {
            if tv_entry.nrows() != n {
                return Err(SurvivalError::DerivativeDimensionMismatch);
            }
        }
        if let Some(tv_exit) = &self.time_varying_exit {
            if tv_exit.nrows() != n {
                return Err(SurvivalError::DerivativeDimensionMismatch);
            }
        }
        if let Some(tv_deriv) = &self.time_varying_derivative_exit {
            if tv_deriv.nrows() != n {
                return Err(SurvivalError::DerivativeDimensionMismatch);
            }
        }
        if let Some(static_cov) = &self.static_covariates {
            if static_cov.nrows() != n {
                return Err(SurvivalError::CovariateDimensionMismatch);
            }
        }

        let total_cols = self.total_columns();
        self.column_partition.validate(total_cols)?;
        self.penalties.validate(total_cols)?;
        Ok(())
    }

    pub fn total_columns(&self) -> usize {
        self.baseline_entry.ncols()
            + self
                .time_varying_entry
                .as_ref()
                .map(|m| m.ncols())
                .unwrap_or(0)
            + self
                .static_covariates
                .as_ref()
                .map(|m| m.ncols())
                .unwrap_or(0)
    }

    fn apply_block(matrix: &Array2<f64>, beta: &Array1<f64>, range: &Range<usize>) -> Array1<f64> {
        let beta_block = beta.slice(s![range.start..range.end]).to_owned();
        matrix.dot(&beta_block)
    }

    fn entry_predictor(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut eta =
            Self::apply_block(&self.baseline_entry, beta, &self.column_partition.baseline);
        if let Some(tv_entry) = &self.time_varying_entry {
            if let Some(tv_range) = &self.column_partition.time_varying {
                eta += &Self::apply_block(tv_entry, beta, tv_range);
            }
        }
        if let Some(static_cov) = &self.static_covariates {
            if let Some(static_range) = &self.column_partition.static_covariates {
                eta += &Self::apply_block(static_cov, beta, static_range);
            }
        }
        eta
    }

    fn exit_predictor(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut eta = Self::apply_block(&self.baseline_exit, beta, &self.column_partition.baseline);
        if let Some(tv_exit) = &self.time_varying_exit {
            if let Some(tv_range) = &self.column_partition.time_varying {
                eta += &Self::apply_block(tv_exit, beta, tv_range);
            }
        }
        if let Some(static_cov) = &self.static_covariates {
            if let Some(static_range) = &self.column_partition.static_covariates {
                eta += &Self::apply_block(static_cov, beta, static_range);
            }
        }
        eta
    }

    fn exit_derivative(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut deriv = Self::apply_block(
            &self.baseline_derivative_exit,
            beta,
            &self.column_partition.baseline,
        );
        if let Some(tv_deriv) = &self.time_varying_derivative_exit {
            if let Some(tv_range) = &self.column_partition.time_varying {
                deriv += &Self::apply_block(tv_deriv, beta, tv_range);
            }
        }
        deriv
    }
}

/// Specification for the survival working model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalSpec {
    pub derivative_guard: f64,
    pub barrier_weight: f64,
    pub barrier_scale: f64,
    pub use_expected_information: bool,
}

impl Default for SurvivalSpec {
    fn default() -> Self {
        Self {
            derivative_guard: DEFAULT_DERIVATIVE_GUARD,
            barrier_weight: DEFAULT_BARRIER_WEIGHT,
            barrier_scale: DEFAULT_BARRIER_SCALE,
            use_expected_information: false,
        }
    }
}

/// Unified working state consumed by PIRLS.
#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
}

/// Unified working model trait shared across GAM and survival families.
pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError>;
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

fn outer_product(row: &Array1<f64>) -> Array2<f64> {
    let len = row.len();
    let mut result = Array2::zeros((len, len));
    for i in 0..len {
        for j in 0..len {
            result[[i, j]] = row[i] * row[j];
        }
    }
    result
}

/// Working model for binomial GAMs producing a diagonal Hessian.
pub struct LogisticWorkingModel<'a> {
    design: ArrayView2<'a, f64>,
    response: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    penalties: PenaltyBlocks,
}

impl<'a> LogisticWorkingModel<'a> {
    pub fn new(
        design: ArrayView2<'a, f64>,
        response: ArrayView1<'a, f64>,
        weights: ArrayView1<'a, f64>,
        penalties: PenaltyBlocks,
    ) -> Self {
        Self {
            design,
            response,
            weights,
            penalties,
        }
    }
}

impl<'a> WorkingModel for LogisticWorkingModel<'a> {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        if beta.len() != self.design.ncols() {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        let eta = self.design.dot(beta);
        let mut mu = Array1::zeros(eta.len());
        for (m, &e) in mu.iter_mut().zip(eta.iter()) {
            *m = sigmoid(e);
        }
        let residual = &self.response - &mu;
        let weighted_residual = &self.weights * &residual;
        let mut gradient = self.design.t().dot(&weighted_residual);
        gradient -= &self.penalties.gradient(beta);

        let mut hessian = Array2::zeros((beta.len(), beta.len()));
        for (i, weight) in self.weights.iter().enumerate() {
            let m = mu[i];
            let w = weight * m * (1.0 - m);
            if w == 0.0 {
                continue;
            }
            let row = self.design.row(i).to_owned();
            hessian -= &(outer_product(&row) * w);
        }
        let mut deviance = 0.0;
        let eps = 1e-12;
        for i in 0..eta.len() {
            let y = self.response[i];
            let w = self.weights[i];
            let m = mu[i];
            deviance -= 2.0 * w * (y * (m.max(eps)).ln() + (1.0 - y) * (1.0 - m).max(eps).ln());
        }
        hessian -= &self.penalties.hessian(beta.len());
        deviance += self.penalties.deviance(beta);
        Ok(WorkingState {
            eta,
            gradient,
            hessian,
            deviance,
        })
    }
}

/// Working model for Gaussian GAMs with identity link.
pub struct GaussianWorkingModel<'a> {
    design: ArrayView2<'a, f64>,
    response: ArrayView1<'a, f64>,
    weights: ArrayView1<'a, f64>,
    penalties: PenaltyBlocks,
}

impl<'a> GaussianWorkingModel<'a> {
    pub fn new(
        design: ArrayView2<'a, f64>,
        response: ArrayView1<'a, f64>,
        weights: ArrayView1<'a, f64>,
        penalties: PenaltyBlocks,
    ) -> Self {
        Self {
            design,
            response,
            weights,
            penalties,
        }
    }
}

impl<'a> WorkingModel for GaussianWorkingModel<'a> {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        if beta.len() != self.design.ncols() {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        let eta = self.design.dot(beta);
        let residual = &self.response - &eta;
        let mut gradient = self.design.t().dot(&(&self.weights * &residual));
        gradient -= &self.penalties.gradient(beta);
        let mut hessian = Array2::zeros((beta.len(), beta.len()));
        for i in 0..self.design.nrows() {
            let w = self.weights[i];
            let row = self.design.row(i);
            hessian -= &(outer_product(&row.to_owned()) * w);
        }
        hessian -= &self.penalties.hessian(beta.len());
        let deviance = self.weights.dot(&(&residual * &residual)) + self.penalties.deviance(beta);
        Ok(WorkingState {
            eta,
            gradient,
            hessian,
            deviance,
        })
    }
}

/// Observed-information survival working model using Royston–Parmar parameterisation.
pub struct SurvivalWorkingModel<'a> {
    layout: &'a SurvivalLayout,
    data: &'a SurvivalTrainingData,
    spec: SurvivalSpec,
}

impl<'a> SurvivalWorkingModel<'a> {
    pub fn new(
        layout: &'a SurvivalLayout,
        data: &'a SurvivalTrainingData,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        layout.validate()?;
        data.validate()?;
        Ok(Self { layout, data, spec })
    }

    fn linear_predictors(&self, beta: &Array1<f64>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let eta_entry = self.layout.entry_predictor(beta);
        let eta_exit = self.layout.exit_predictor(beta);
        let derivative_exit = self.layout.exit_derivative(beta);
        (eta_entry, eta_exit, derivative_exit)
    }
}

impl<'a> WorkingModel for SurvivalWorkingModel<'a> {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        if beta.len() != self.layout.total_columns() {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        let (eta_entry, eta_exit, derivative_exit) = self.linear_predictors(beta);
        let h_entry = eta_entry.mapv(|v| v.exp());
        let h_exit = eta_exit.mapv(|v| v.exp());
        let delta_h = &h_exit - &h_entry;
        let mut gradient = Array1::zeros(beta.len());
        let mut hessian = Array2::zeros((beta.len(), beta.len()));
        let mut deviance = 0.0;

        for i in 0..eta_exit.len() {
            let weight = self.data.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let event_target = self.data.event_target[i] as f64;
            let entry_eta = eta_entry[i];
            let exit_eta = eta_exit[i];
            let exit_derivative = derivative_exit[i];
            let hazard_entry = h_entry[i];
            let hazard_exit = h_exit[i];
            let delta = delta_h[i].max(0.0);
            let log_survival = -delta;

            // Guarded derivative for the likelihood term.
            let guarded_derivative = if exit_derivative > self.spec.derivative_guard {
                exit_derivative
            } else {
                self.spec.derivative_guard
            };
            let log_hazard = exit_eta + guarded_derivative.ln();
            let loglik = weight * (event_target * log_hazard + log_survival);
            deviance -= 2.0 * loglik;

            let baseline_range = &self.layout.column_partition.baseline;
            let baseline_entry_row = self.layout.baseline_entry.row(i).to_owned();
            let baseline_exit_row = self.layout.baseline_exit.row(i).to_owned();
            let baseline_deriv_row = self.layout.baseline_derivative_exit.row(i).to_owned();

            let mut row_vectors: Vec<(Array1<f64>, std::ops::Range<usize>)> =
                vec![(baseline_exit_row.clone(), baseline_range.clone())];

            if let Some(tv_range) = &self.layout.column_partition.time_varying {
                if let Some(tv_exit) = &self.layout.time_varying_exit {
                    row_vectors.push((tv_exit.row(i).to_owned(), tv_range.clone()));
                }
            }
            if let Some(static_range) = &self.layout.column_partition.static_covariates {
                if let Some(static_cov) = &self.layout.static_covariates {
                    row_vectors.push((static_cov.row(i).to_owned(), static_range.clone()));
                }
            }

            // Gradient contributions from the exit hazard and survival term.
            for (row, range) in &row_vectors {
                let mut gradient_slice = gradient.slice_mut(s![range.start..range.end]);
                let exit_component = row * (weight * (event_target - hazard_exit));
                gradient_slice += &exit_component;
            }
            // Entry term (only affects survival component)
            let mut gradient_entry_slice =
                gradient.slice_mut(s![baseline_range.start..baseline_range.end]);
            gradient_entry_slice -= &(baseline_entry_row.clone() * (weight * hazard_entry));
            if let Some(tv_range) = &self.layout.column_partition.time_varying {
                if let Some(tv_entry) = &self.layout.time_varying_entry {
                    let mut slice = gradient.slice_mut(s![tv_range.start..tv_range.end]);
                    slice -= &(tv_entry.row(i).to_owned() * (weight * hazard_entry));
                }
            }

            // Derivative term contributes only when the guard is inactive.
            if exit_derivative > self.spec.derivative_guard {
                let mut deriv_slice =
                    gradient.slice_mut(s![baseline_range.start..baseline_range.end]);
                deriv_slice +=
                    &(baseline_deriv_row.clone() * (weight * event_target / exit_derivative));
                if let Some(tv_range) = &self.layout.column_partition.time_varying {
                    if let Some(tv_deriv) = &self.layout.time_varying_derivative_exit {
                        let mut slice = gradient.slice_mut(s![tv_range.start..tv_range.end]);
                        slice += &(tv_deriv.row(i).to_owned()
                            * (weight * event_target / exit_derivative));
                    }
                }
            }

            // Hessian contributions: exit block (observed information)
            let mut assemble_outer = |row: &Array1<f64>, range: &Range<usize>, scale: f64| {
                if scale == 0.0 {
                    return;
                }
                let design_outer = outer_product(row);
                let mut block =
                    hessian.slice_mut(s![range.start..range.end, range.start..range.end]);
                block -= &(design_outer * scale);
            };

            // Exit negative Hessian contribution
            for (row, range) in &row_vectors {
                assemble_outer(row, range, weight * hazard_exit);
            }
            // Entry positive Hessian contribution (subtracted because of -ΔH)
            assemble_outer(&baseline_entry_row, baseline_range, -weight * hazard_entry);
            if let Some(tv_range) = &self.layout.column_partition.time_varying {
                if let Some(tv_entry) = &self.layout.time_varying_entry {
                    assemble_outer(
                        &tv_entry.row(i).to_owned(),
                        tv_range,
                        -weight * hazard_entry,
                    );
                }
            }

            // Derivative Hessian contribution when active
            if exit_derivative > self.spec.derivative_guard {
                let scale = weight * event_target / (exit_derivative * exit_derivative);
                assemble_outer(&baseline_deriv_row, baseline_range, -scale);
                if let Some(tv_range) = &self.layout.column_partition.time_varying {
                    if let Some(tv_deriv) = &self.layout.time_varying_derivative_exit {
                        assemble_outer(&tv_deriv.row(i).to_owned(), tv_range, -scale);
                    }
                }
            }

            // Softplus barrier on negative derivatives.
            if self.spec.barrier_weight > 0.0 {
                let scaled = -exit_derivative / self.spec.barrier_scale;
                let softplus = (1.0 + scaled.exp()).ln();
                deviance += 2.0 * self.spec.barrier_weight * weight * softplus;
                let sigmoid = 1.0 / (1.0 + (-scaled).exp());
                let barrier_grad_coeff =
                    self.spec.barrier_weight * weight * sigmoid / self.spec.barrier_scale;
                let barrier_hess_coeff =
                    self.spec.barrier_weight * weight * sigmoid * (1.0 - sigmoid)
                        / (self.spec.barrier_scale * self.spec.barrier_scale);
                let mut barrier_slice =
                    gradient.slice_mut(s![baseline_range.start..baseline_range.end]);
                barrier_slice -= &(baseline_deriv_row.clone() * barrier_grad_coeff);
                if let Some(tv_range) = &self.layout.column_partition.time_varying {
                    if let Some(tv_deriv) = &self.layout.time_varying_derivative_exit {
                        let mut slice = gradient.slice_mut(s![tv_range.start..tv_range.end]);
                        slice -= &(tv_deriv.row(i).to_owned() * barrier_grad_coeff);
                    }
                }
                assemble_outer(&baseline_deriv_row, baseline_range, barrier_hess_coeff);
                if let Some(tv_range) = &self.layout.column_partition.time_varying {
                    if let Some(tv_deriv) = &self.layout.time_varying_derivative_exit {
                        assemble_outer(&tv_deriv.row(i).to_owned(), tv_range, barrier_hess_coeff);
                    }
                }
            }
        }

        gradient -= &self.layout.penalties.gradient(beta);
        hessian -= &self.layout.penalties.hessian(beta.len());
        deviance += self.layout.penalties.deviance(beta);
        Ok(WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        })
    }
}

/// Stored factorisation of the penalised Hessian.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Persisted survival model artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: Array2<f64>,
    pub time_varying_basis: Option<Array2<f64>>,
    pub static_covariate_layout: Array2<f64>,
    pub penalties: PenaltyBlocks,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<HessianFactor>,
}

impl SurvivalModelArtifacts {
    pub fn cumulative_hazard(&self, age: f64, covariates: &Array1<f64>) -> f64 {
        let log_age = self.age_transform.apply(age);
        let mut design_row = self.age_basis.row(0).to_owned();
        for value in design_row.iter_mut() {
            *value *= log_age;
        }
        let mut eta = design_row.dot(&self.coefficients.slice(s![..design_row.len()]).to_owned());
        if let Some(tv) = &self.time_varying_basis {
            let mut tv_row = tv.row(0).to_owned();
            for (val, cov) in tv_row.iter_mut().zip(covariates.iter()) {
                *val *= *cov;
            }
            let range = design_row.len()..design_row.len() + tv_row.len();
            eta += tv_row.dot(&self.coefficients.slice(s![range]).to_owned());
        }
        let static_range_start = design_row.len()
            + self
                .time_varying_basis
                .as_ref()
                .map(|m| m.ncols())
                .unwrap_or(0);
        let static_slice = self.coefficients.slice(s![static_range_start..]);
        eta += covariates.dot(&static_slice.to_owned());
        eta.exp()
    }

    pub fn cumulative_incidence(&self, age: f64, covariates: &Array1<f64>) -> f64 {
        let h = self.cumulative_hazard(age, covariates);
        1.0 - (-h).exp()
    }

    pub fn conditional_absolute_risk(
        &self,
        t0: f64,
        t1: f64,
        covariates: &Array1<f64>,
        cif_competing_t0: f64,
    ) -> f64 {
        let cif_t0 = self.cumulative_incidence(t0, covariates);
        let cif_t1 = self.cumulative_incidence(t1, covariates);
        let delta = (cif_t1 - cif_t0).max(0.0);
        let denom = (1.0 - cif_t0 - cif_competing_t0).max(DEFAULT_RISK_EPSILON);
        delta / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Array1, Array2, array};

    fn simple_layout(n: usize) -> (SurvivalLayout, SurvivalTrainingData) {
        let age_entry = Array::linspace(50.0, 50.0 + n as f64 - 1.0, n);
        let age_exit = &age_entry + 1.0;
        let event_target = Array1::ones(n);
        let event_competing = Array1::zeros(n);
        let sample_weight = Array1::ones(n);
        let covariates = Array2::zeros((n, 0));
        let data = SurvivalTrainingData {
            age_entry: age_entry.clone(),
            age_exit: age_exit.clone(),
            event_target,
            event_competing,
            sample_weight,
            covariates,
        };
        let age_transform = AgeTransform::from_training(&age_entry, 0.1).unwrap();
        let baseline_entry = Array2::ones((n, 2));
        let mut baseline_exit = Array2::ones((n, 2));
        for (i, mut row) in baseline_exit.rows_mut().into_iter().enumerate() {
            row[1] = (age_exit[i] - age_entry[i]) as f64;
        }
        let baseline_derivative_exit = Array2::from_elem((n, 2), 0.5);
        let penalties = PenaltyBlocks::new(vec![PenaltyMatrixBlock {
            matrix: Array2::from_diag(&Array1::ones(2)),
            lambda: 0.1,
        }]);
        let layout = SurvivalLayout {
            baseline_entry,
            baseline_exit,
            baseline_derivative_exit,
            time_varying_entry: None,
            time_varying_exit: None,
            time_varying_derivative_exit: None,
            static_covariates: None,
            age_transform,
            reference_constraint: ReferenceConstraint {
                transform: Array2::from_diag(&Array1::ones(2)),
            },
            penalties,
            column_partition: ColumnPartition {
                baseline: 0..2,
                time_varying: None,
                static_covariates: None,
            },
        };
        (layout, data)
    }

    #[test]
    fn penalty_contributes_to_working_state() {
        let (layout, data) = simple_layout(4);
        let mut penalised_model =
            SurvivalWorkingModel::new(&layout, &data, SurvivalSpec::default()).unwrap();
        let beta = array![0.12, -0.08];
        let penalised = penalised_model.update(&beta).unwrap();

        let mut unpenalised_layout = layout.clone();
        for block in &mut unpenalised_layout.penalties.blocks {
            block.lambda = 0.0;
        }
        let mut unpenalised_model =
            SurvivalWorkingModel::new(&unpenalised_layout, &data, SurvivalSpec::default()).unwrap();
        let unpenalised = unpenalised_model.update(&beta).unwrap();

        let penalty_deviance = layout.penalties.deviance(&beta);
        assert_abs_diff_eq!(
            penalised.deviance,
            unpenalised.deviance + penalty_deviance,
            epsilon = 1e-10
        );

        let penalty_gradient = layout.penalties.gradient(&beta);
        let expected_gradient = &unpenalised.gradient - &penalty_gradient;
        for (observed, expected) in penalised.gradient.iter().zip(expected_gradient.iter()) {
            assert_abs_diff_eq!(*observed, *expected, epsilon = 1e-10);
        }

        let penalty_hessian = layout.penalties.hessian(beta.len());
        let expected_hessian = &unpenalised.hessian - &penalty_hessian;
        for (observed_row, expected_row) in penalised
            .hessian
            .rows()
            .into_iter()
            .zip(expected_hessian.rows())
        {
            for (observed, expected) in observed_row.iter().zip(expected_row.iter()) {
                assert_abs_diff_eq!(*observed, *expected, epsilon = 1e-10);
            }
        }

        let epsilon = 1e-6;
        let mut numeric_penalised = Array1::zeros(beta.len());
        let mut numeric_unpenalised = Array1::zeros(beta.len());
        for j in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[j] += epsilon;
            let mut beta_minus = beta.clone();
            beta_minus[j] -= epsilon;

            let mut model_plus =
                SurvivalWorkingModel::new(&layout, &data, SurvivalSpec::default()).unwrap();
            let state_plus = model_plus.update(&beta_plus).unwrap();
            let mut model_minus =
                SurvivalWorkingModel::new(&layout, &data, SurvivalSpec::default()).unwrap();
            let state_minus = model_minus.update(&beta_minus).unwrap();
            numeric_penalised[j] = (state_plus.deviance - state_minus.deviance) / (2.0 * epsilon);

            let mut model_plus_unpen =
                SurvivalWorkingModel::new(&unpenalised_layout, &data, SurvivalSpec::default())
                    .unwrap();
            let state_plus_unpen = model_plus_unpen.update(&beta_plus).unwrap();
            let mut model_minus_unpen =
                SurvivalWorkingModel::new(&unpenalised_layout, &data, SurvivalSpec::default())
                    .unwrap();
            let state_minus_unpen = model_minus_unpen.update(&beta_minus).unwrap();
            numeric_unpenalised[j] =
                (state_plus_unpen.deviance - state_minus_unpen.deviance) / (2.0 * epsilon);
        }
        let penalty_gradient_dev = layout.penalties.gradient(&beta) * 2.0;
        for j in 0..beta.len() {
            assert_abs_diff_eq!(
                numeric_penalised[j],
                numeric_unpenalised[j] + penalty_gradient_dev[j],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn delta_h_matches_difference() {
        let (layout, data) = simple_layout(3);
        let beta = array![0.05, 0.02];
        let model = SurvivalWorkingModel::new(&layout, &data, SurvivalSpec::default()).unwrap();
        let (eta_entry, eta_exit, _) = model.linear_predictors(&beta);
        let delta_from_state =
            (&eta_exit.mapv(|v| v.exp()) - &eta_entry.mapv(|v| v.exp())).to_vec();
        for i in 0..eta_exit.len() {
            let entry = eta_entry[i].exp();
            let exit = eta_exit[i].exp();
            assert_abs_diff_eq!(delta_from_state[i], exit - entry, epsilon = 1e-10);
        }
    }

    #[test]
    fn risk_monotonicity() {
        let (layout, data) = simple_layout(1);
        let _ = data;
        let artifacts = SurvivalModelArtifacts {
            coefficients: array![0.05, 0.02],
            age_basis: Array2::from_elem((1, 2), 1.0),
            time_varying_basis: None,
            static_covariate_layout: Array2::zeros((1, 0)),
            penalties: layout.penalties.clone(),
            age_transform: layout.age_transform.clone(),
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
        };
        let covariates = Array1::zeros(0);
        let risk_short = artifacts.conditional_absolute_risk(50.0, 51.0, &covariates, 0.0);
        let risk_long = artifacts.conditional_absolute_risk(50.0, 55.0, &covariates, 0.0);
        assert!(risk_long >= risk_short);
        assert!(risk_short >= 0.0);
    }
}
