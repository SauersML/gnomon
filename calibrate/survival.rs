use crate::calibrate::basis::{
    BasisError, create_bspline_basis_with_knots, create_difference_penalty_matrix,
};
use crate::calibrate::faer_ndarray::FaerSvd;
use ndarray::prelude::*;
use ndarray::{Array1, Array2, concatenate};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use thiserror::Error;

const DEFAULT_DERIVATIVE_GUARD: f64 = 1e-8;
const DEFAULT_BARRIER_WEIGHT: f64 = 1e-4;
const DEFAULT_BARRIER_SCALE: f64 = 1.0;
const DEFAULT_RISK_EPSILON: f64 = 1e-12;

/// Errors surfaced while validating survival data structures or evaluating the model.
#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("age vectors must have at least one element")]
    EmptyAgeVector,
    #[error("age values must be finite")]
    NonFiniteAge,
    #[error("age {age} is outside the guarded log-age domain (a_min={minimum}, delta={delta})")]
    GuardDomainViolation { age: f64, minimum: f64, delta: f64 },
    #[error("age_entry must be strictly less than age_exit for every subject")]
    InvalidAgeOrder,
    #[error("event indicators must be 0 or 1")]
    InvalidEventFlag,
    #[error("event_target and event_competing indicators must be mutually exclusive")]
    ConflictingEvents,
    #[error("sample weights must be finite and non-negative")]
    InvalidSampleWeight,
    #[error("covariate arrays must have consistent dimensions")]
    CovariateDimensionMismatch,
    #[error("covariate values must be finite")]
    NonFiniteCovariate,
    #[error("design matrix columns do not match coefficient length")]
    DesignDimensionMismatch,
    #[error("basis evaluation failed: {0}")]
    Basis(#[from] BasisError),
}

/// Working model abstraction shared between GAM and survival implementations.
pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError>;
}

/// Aggregated state returned by [`WorkingModel::update`].
#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
}

/// Guarded log-age transformation used across training and scoring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AgeTransform {
    pub minimum_age: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn from_training(age_entry: &Array1<f64>, delta: f64) -> Result<Self, SurvivalError> {
        if age_entry.is_empty() {
            return Err(SurvivalError::EmptyAgeVector);
        }
        let mut min_age = f64::INFINITY;
        for &value in age_entry.iter() {
            if !value.is_finite() {
                return Err(SurvivalError::NonFiniteAge);
            }
            if value < min_age {
                min_age = value;
            }
        }
        Ok(Self {
            minimum_age: min_age,
            delta,
        })
    }

    fn guard_shift(&self, age: f64) -> Result<f64, SurvivalError> {
        if !age.is_finite() {
            return Err(SurvivalError::NonFiniteAge);
        }
        let shifted = age - self.minimum_age + self.delta;
        if !shifted.is_finite() || shifted <= 0.0 {
            return Err(SurvivalError::GuardDomainViolation {
                age,
                minimum: self.minimum_age,
                delta: self.delta,
            });
        }
        Ok(shifted)
    }

    #[inline]
    pub fn transform(&self, age: f64) -> Result<f64, SurvivalError> {
        let shifted = self.guard_shift(age)?;
        Ok(shifted.ln())
    }

    #[inline]
    pub fn derivative_factor(&self, age: f64) -> Result<f64, SurvivalError> {
        let shifted = self.guard_shift(age)?;
        Ok(1.0 / shifted)
    }

    pub fn transform_array(&self, ages: &Array1<f64>) -> Result<Array1<f64>, SurvivalError> {
        let mut result = Array1::<f64>::zeros(ages.len());
        for (idx, &age) in ages.iter().enumerate() {
            result[idx] = self.transform(age)?;
        }
        Ok(result)
    }
}

/// Linear transform that removes the baseline spline's null direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceConstraint {
    pub transform: Array2<f64>,
    pub reference_log_age: f64,
}

impl ReferenceConstraint {
    pub fn apply(&self, basis: &Array2<f64>) -> Array2<f64> {
        basis.dot(&self.transform)
    }
}

/// Describes a spline basis that can be reconstructed during scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisDescriptor {
    pub knot_vector: Array1<f64>,
    pub degree: usize,
}

/// Stored smoothing metadata for reproduction at prediction time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyDescriptor {
    pub order: usize,
    pub lambda: f64,
}

/// Column descriptions for static covariates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateLayout {
    pub column_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlock {
    pub matrix: Array2<f64>,
    pub lambda: f64,
    pub range: Range<usize>,
}

#[derive(Debug, Clone)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }

    pub fn gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        let mut grad = Array1::zeros(beta.len());
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }

            let view = beta.slice(s![block.range.clone()]);
            let contrib = block.matrix.dot(&view.to_owned());
            let mut grad_slice = grad.slice_mut(s![block.range.clone()]);
            grad_slice += &(block.lambda * contrib);
        }
        grad
    }

    pub fn hessian(&self, dim: usize) -> Array2<f64> {
        let mut hessian = Array2::zeros((dim, dim));
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let rows = block.range.clone();
            for (local_i, row_idx) in rows.clone().enumerate() {
                for (local_j, col_idx) in rows.clone().enumerate() {
                    hessian[[row_idx, col_idx]] += block.lambda * block.matrix[[local_i, local_j]];
                }
            }
        }
        hessian
    }

    pub fn deviance(&self, beta: &Array1<f64>) -> f64 {
        let mut value = 0.0;
        for block in &self.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            let view = beta.slice(s![block.range.clone()]);
            let quad = view.dot(&block.matrix.dot(&view.to_owned()));
            value += block.lambda * quad;
        }
        value
    }
}

/// Training-time cached design matrices.
#[derive(Debug, Clone)]
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
    pub combined_entry: Array2<f64>,
    pub combined_exit: Array2<f64>,
    pub combined_derivative_exit: Array2<f64>,
}

/// Frequency-weighted survival training data bundle.
#[derive(Debug, Clone)]
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

impl SurvivalTrainingData {
    pub fn validate(&self) -> Result<(), SurvivalError> {
        let n = self.age_entry.len();
        if n == 0 {
            return Err(SurvivalError::EmptyAgeVector);
        }
        let dimension_mismatch = self.age_exit.len() != n
            || self.event_target.len() != n
            || self.event_competing.len() != n
            || self.sample_weight.len() != n
            || self.pgs.len() != n
            || self.sex.len() != n
            || self.pcs.nrows() != n;
        if dimension_mismatch {
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

            let pgs = self.pgs[i];
            let sex = self.sex[i];
            if !pgs.is_finite() || !sex.is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
            for j in 0..self.pcs.ncols() {
                if !self.pcs[[i, j]].is_finite() {
                    return Err(SurvivalError::NonFiniteCovariate);
                }
            }
        }

        Ok(())
    }
}

/// Guard that constrains the baseline spline at the chosen reference point.
fn make_reference_constraint(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    reference_u: f64,
) -> Result<ReferenceConstraint, SurvivalError> {
    let data = array![reference_u];
    let (basis_arc, _) = create_bspline_basis_with_knots(data.view(), knot_vector, degree)?;
    let basis = (*basis_arc).clone();
    let row = basis.row(0).to_owned();
    let transform = nullspace_transform(&row)?;
    Ok(ReferenceConstraint {
        transform,
        reference_log_age: reference_u,
    })
}

/// Build a nullspace transform for a single-row constraint.
fn nullspace_transform(constraint_row: &Array1<f64>) -> Result<Array2<f64>, SurvivalError> {
    let k = constraint_row.len();
    let mut row_mat = Array2::<f64>::zeros((k, 1));
    row_mat.column_mut(0).assign(constraint_row);
    let (u_opt, ..) = row_mat
        .svd(true, false)
        .map_err(|err| SurvivalError::Basis(BasisError::from(err)))?;
    let u = u_opt.ok_or_else(|| SurvivalError::Basis(BasisError::ConstraintNullspaceNotFound))?;
    Ok(u.slice(s![.., 1..]).to_owned())
}

/// Evaluate a basis and its derivative with respect to the guarded log-age.
fn evaluate_basis_and_derivative(
    log_ages: ArrayView1<f64>,
    descriptor: &BasisDescriptor,
) -> Result<(Array2<f64>, Array2<f64>), SurvivalError> {
    let (basis_arc, _) = create_bspline_basis_with_knots(
        log_ages,
        descriptor.knot_vector.view(),
        descriptor.degree,
    )?;
    let basis = (*basis_arc).clone();

    let eps = 1e-6;
    let mut perturbed_plus = log_ages.to_owned();
    let mut perturbed_minus = log_ages.to_owned();
    perturbed_plus.mapv_inplace(|v| v + eps);
    perturbed_minus.mapv_inplace(|v| v - eps);
    let (basis_plus_arc, _) = create_bspline_basis_with_knots(
        perturbed_plus.view(),
        descriptor.knot_vector.view(),
        descriptor.degree,
    )?;
    let (basis_minus_arc, _) = create_bspline_basis_with_knots(
        perturbed_minus.view(),
        descriptor.knot_vector.view(),
        descriptor.degree,
    )?;
    let basis_plus = (*basis_plus_arc).clone();
    let basis_minus = (*basis_minus_arc).clone();
    let mut derivative = basis_plus;
    derivative -= &basis_minus;
    derivative.mapv_inplace(|v| v / (2.0 * eps));

    Ok((basis, derivative))
}

/// Construct the cached survival layout for PIRLS updates.
#[allow(clippy::too_many_arguments)]
pub fn build_survival_layout(
    data: &SurvivalTrainingData,
    age_basis: &BasisDescriptor,
    delta: f64,
    baseline_penalty_order: usize,
    baseline_lambda: f64,
    monotonic_grid_size: usize,
) -> Result<(SurvivalLayout, MonotonicityPenalty), SurvivalError> {
    data.validate()?;
    let n = data.age_entry.len();
    let age_transform = AgeTransform::from_training(&data.age_entry, delta)?;
    let log_entry = age_transform.transform_array(&data.age_entry)?;
    let log_exit = age_transform.transform_array(&data.age_exit)?;

    let reference_u = log_exit.mean().unwrap_or(0.0);
    let reference_constraint =
        make_reference_constraint(age_basis.knot_vector.view(), age_basis.degree, reference_u)?;

    let (baseline_entry_raw, _) = evaluate_basis_and_derivative(log_entry.view(), age_basis)?;
    let (baseline_exit_raw, baseline_exit_deriv_u) =
        evaluate_basis_and_derivative(log_exit.view(), age_basis)?;

    let constrained_entry = reference_constraint.apply(&baseline_entry_raw);
    let constrained_exit = reference_constraint.apply(&baseline_exit_raw);
    let constrained_derivative_exit_u = reference_constraint.apply(&baseline_exit_deriv_u);

    let mut baseline_derivative_exit = constrained_derivative_exit_u;
    for (mut row, age) in baseline_derivative_exit
        .rows_mut()
        .into_iter()
        .zip(data.age_exit.iter().copied())
    {
        let factor = age_transform.derivative_factor(age)?;
        row.mapv_inplace(|v| v * factor);
    }

    let static_covariates = assemble_static_covariates(data);

    let combined_entry = concatenate_design(&constrained_entry, None, &static_covariates);
    let combined_exit = concatenate_design(&constrained_exit, None, &static_covariates);
    let zero_static = Array2::<f64>::zeros((n, static_covariates.ncols()));
    let combined_derivative_exit =
        concatenate_design(&baseline_derivative_exit, None, &zero_static);

    let penalty_matrix =
        create_difference_penalty_matrix(constrained_exit.ncols(), baseline_penalty_order)?;
    let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
        matrix: penalty_matrix,
        lambda: baseline_lambda,
        range: 0..constrained_exit.ncols(),
    }]);

    let layout = SurvivalLayout {
        baseline_entry: constrained_entry,
        baseline_exit: constrained_exit,
        baseline_derivative_exit,
        time_varying_entry: None,
        time_varying_exit: None,
        time_varying_derivative_exit: None,
        static_covariates,
        age_transform,
        reference_constraint,
        penalties,
        combined_entry,
        combined_exit,
        combined_derivative_exit,
    };

    let monotonicity = build_monotonicity_penalty(
        &layout,
        age_basis,
        &data.age_exit,
        monotonic_grid_size,
        baseline_lambda * 1e-4,
    )?;

    Ok((layout, monotonicity))
}

fn assemble_static_covariates(data: &SurvivalTrainingData) -> Array2<f64> {
    let n = data.age_entry.len();
    let num_pcs = data.pcs.ncols();
    let mut matrix = Array2::<f64>::zeros((n, 2 + num_pcs));
    for i in 0..n {
        matrix[[i, 0]] = data.pgs[i];
        matrix[[i, 1]] = data.sex[i];
        for j in 0..num_pcs {
            matrix[[i, 2 + j]] = data.pcs[[i, j]];
        }
    }
    matrix
}

fn concatenate_design(
    baseline: &Array2<f64>,
    time_varying: Option<&Array2<f64>>,
    static_covariates: &Array2<f64>,
) -> Array2<f64> {
    let mut parts: Vec<ArrayView2<f64>> = Vec::new();
    parts.push(baseline.view());
    if let Some(tv) = time_varying {
        parts.push(tv.view());
    }
    parts.push(static_covariates.view());
    concatenate(Axis(1), &parts).expect("design concatenation")
}

/// Soft barrier discouraging negative exit derivatives.
#[derive(Debug, Clone)]
pub struct MonotonicityPenalty {
    pub lambda: f64,
    pub derivative_design: Array2<f64>,
}

/// Configuration controlling guard behaviour and optional softplus barrier.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
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

fn build_monotonicity_penalty(
    layout: &SurvivalLayout,
    age_basis: &BasisDescriptor,
    ages_exit: &Array1<f64>,
    grid_size: usize,
    lambda: f64,
) -> Result<MonotonicityPenalty, SurvivalError> {
    if grid_size == 0 {
        return Ok(MonotonicityPenalty {
            lambda,
            derivative_design: Array2::<f64>::zeros((0, layout.combined_exit.ncols())),
        });
    }

    let mut min_age = f64::INFINITY;
    let mut max_age = f64::NEG_INFINITY;
    for &age in ages_exit.iter() {
        if age < min_age {
            min_age = age;
        }
        if age > max_age {
            max_age = age;
        }
    }
    if !min_age.is_finite() || !max_age.is_finite() || min_age >= max_age {
        return Ok(MonotonicityPenalty {
            lambda,
            derivative_design: Array2::<f64>::zeros((0, layout.combined_exit.ncols())),
        });
    }

    let mut grid = Array1::<f64>::zeros(grid_size);
    if grid_size == 1 {
        grid[0] = min_age;
    } else {
        let span = max_age - min_age;
        for (idx, value) in grid.iter_mut().enumerate() {
            let frac = idx as f64 / (grid_size as f64 - 1.0);
            *value = min_age + frac * span;
        }
    }

    let mut log_grid = Array1::<f64>::zeros(grid_size);
    for (idx, &age) in grid.iter().enumerate() {
        log_grid[idx] = layout.age_transform.transform(age)?;
    }
    let (_, derivative_u) = evaluate_basis_and_derivative(log_grid.view(), age_basis)?;
    let constrained_derivative_u = layout.reference_constraint.apply(&derivative_u);
    let mut derivative_age = constrained_derivative_u;
    for (mut row, &age) in derivative_age.rows_mut().into_iter().zip(grid.iter()) {
        let factor = layout.age_transform.derivative_factor(age)?;
        row *= factor;
    }

    let mut combined = Array2::<f64>::zeros((grid_size, layout.combined_exit.ncols()));
    let baseline_cols = layout.baseline_exit.ncols();
    combined
        .slice_mut(s![.., ..baseline_cols])
        .assign(&derivative_age);
    Ok(MonotonicityPenalty {
        lambda,
        derivative_design: combined,
    })
}

/// Royston–Parmar working model implementation.
pub struct WorkingModelSurvival {
    pub layout: SurvivalLayout,
    pub sample_weight: Array1<f64>,
    pub event_target: Array1<u8>,
    pub monotonicity: MonotonicityPenalty,
    pub spec: SurvivalSpec,
}

impl WorkingModelSurvival {
    pub fn new(
        layout: SurvivalLayout,
        data: &SurvivalTrainingData,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
    ) -> Result<Self, SurvivalError> {
        data.validate()?;
        Ok(Self {
            layout,
            sample_weight: data.sample_weight.clone(),
            event_target: data.event_target.clone(),
            monotonicity,
            spec,
        })
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        let expected_dim = self.layout.combined_exit.ncols();
        if beta.len() != expected_dim {
            return Err(SurvivalError::DesignDimensionMismatch);
        }

        let eta_exit = self.layout.combined_exit.dot(beta);
        let eta_entry = self.layout.combined_entry.dot(beta);
        let derivative_exit = self.layout.combined_derivative_exit.dot(beta);

        let h_exit = eta_exit.mapv(f64::exp);
        let h_entry = eta_entry.mapv(f64::exp);
        let mut delta_h = &h_exit - &h_entry;
        delta_h.mapv_inplace(|v| v.max(0.0));

        let n = eta_exit.len();
        let p = beta.len();
        let mut gradient = Array1::<f64>::zeros(p);
        let mut hessian = Array2::<f64>::zeros((p, p));
        let mut deviance = 0.0;

        let guard_threshold = self.spec.derivative_guard.max(f64::EPSILON);
        for i in 0..n {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let d = f64::from(self.event_target[i]);
            let eta_e = eta_exit[i];
            let h_e = h_exit[i];
            let h_s = h_entry[i];
            let delta = delta_h[i];
            let d_eta_exit = derivative_exit[i];
            let guarded_derivative = if d_eta_exit > guard_threshold {
                d_eta_exit
            } else {
                guard_threshold
            };
            let log_guard = guarded_derivative.ln();
            let ell = weight * (d * (eta_e + log_guard) - delta);
            deviance -= 2.0 * ell;

            let x_exit = self.layout.combined_exit.row(i);
            let x_entry = self.layout.combined_entry.row(i);
            let d_exit = self.layout.combined_derivative_exit.row(i);
            let scale = 1.0 / guarded_derivative;

            for j in 0..p {
                gradient[j] += weight
                    * (d * (x_exit[j] + d_exit[j] * scale) - h_e * x_exit[j] + h_s * x_entry[j]);
            }

            for j in 0..p {
                for k in j..p {
                    let mut value = weight * h_e * x_exit[j] * x_exit[k]
                        + weight * h_s * x_entry[j] * x_entry[k];
                    if d > 0.0 {
                        let x_tilde_j = x_exit[j] + d_exit[j] * scale;
                        let x_tilde_k = x_exit[k] + d_exit[k] * scale;
                        value += weight * d * x_tilde_j * x_tilde_k;
                    }
                    hessian[[j, k]] += value;
                    if j != k {
                        hessian[[k, j]] += value;
                    }
                }
            }

            if self.spec.barrier_weight > 0.0 {
                let scaled = -d_eta_exit / self.spec.barrier_scale;
                let softplus = (1.0 + scaled.exp()).ln();
                deviance += 2.0 * self.spec.barrier_weight * weight * softplus;
                let sigmoid = 1.0 / (1.0 + (-scaled).exp());
                let barrier_grad_coeff =
                    self.spec.barrier_weight * weight * sigmoid / self.spec.barrier_scale;
                let barrier_hess_coeff =
                    self.spec.barrier_weight * weight * sigmoid * (1.0 - sigmoid)
                        / (self.spec.barrier_scale * self.spec.barrier_scale);
                for j in 0..p {
                    gradient[j] -= barrier_grad_coeff * d_exit[j];
                    for k in j..p {
                        let value = barrier_hess_coeff * d_exit[j] * d_exit[k];
                        hessian[[j, k]] += value;
                        if j != k {
                            hessian[[k, j]] += value;
                        }
                    }
                }
            }
        }

        gradient -= &self.layout.penalties.gradient(beta);
        hessian -= &self.layout.penalties.hessian(beta.len());
        deviance += self.layout.penalties.deviance(beta);

        if self.monotonicity.lambda > 0.0 && self.monotonicity.derivative_design.nrows() > 0 {
            apply_monotonicity_penalty(
                &self.monotonicity,
                beta,
                &mut gradient,
                &mut hessian,
                &mut deviance,
            );
        }

        Ok(WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        })
    }
}

fn apply_monotonicity_penalty(
    penalty: &MonotonicityPenalty,
    beta: &Array1<f64>,
    gradient: &mut Array1<f64>,
    hessian: &mut Array2<f64>,
    deviance: &mut f64,
) {
    let lambda = penalty.lambda;
    if lambda == 0.0 {
        return;
    }
    let design = &penalty.derivative_design;
    let values = design.dot(beta);
    let mut penalty_sum = 0.0;
    for (row, &value) in design.rows().into_iter().zip(values.iter()) {
        let softplus = (-value).exp().ln_1p();
        penalty_sum += softplus;
        let sigma = 1.0 / (1.0 + (value).exp());
        let grad_scale = -lambda * sigma;
        for (idx, &entry) in row.iter().enumerate() {
            gradient[idx] += grad_scale * entry;
        }
        let h_scale = lambda * sigma * (1.0 - sigma);
        for (i, &entry_i) in row.iter().enumerate() {
            for (j, &entry_j) in row.iter().enumerate() {
                hessian[[i, j]] += h_scale * entry_i * entry_j;
            }
        }
    }
    *deviance += 2.0 * lambda * penalty_sum;
}

/// Stored factorization metadata for downstream diagnostics.
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

/// Prediction inputs referencing existing arrays.
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
}

/// Evaluate the cumulative hazard at a given age.
pub fn cumulative_hazard(
    age: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let log_age = artifacts.age_transform.transform(age)?;
    let (basis_arc, _) = create_bspline_basis_with_knots(
        array![log_age].view(),
        artifacts.age_basis.knot_vector.view(),
        artifacts.age_basis.degree,
    )?;
    let basis = (*basis_arc).clone();
    let constrained = artifacts.reference_constraint.apply(&basis);

    let mut design = constrained.row(0).to_owned();
    if let Some(time_basis) = &artifacts.time_varying_basis {
        let (tv_arc, _) = create_bspline_basis_with_knots(
            array![log_age].view(),
            time_basis.knot_vector.view(),
            time_basis.degree,
        )?;
        let tv = artifacts.reference_constraint.apply(&(*tv_arc).clone());
        design = concatenate(Axis(0), &[design.view(), tv.row(0)]).expect("time concat");
    }
    design = concatenate(Axis(0), &[design.view(), covariates.view()]).expect("cov concat");
    let eta = design.dot(&artifacts.coefficients);
    Ok(eta.exp())
}

pub fn cumulative_incidence(
    age: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let h = cumulative_hazard(age, covariates, artifacts)?;
    Ok(1.0 - (-h).exp())
}

pub fn conditional_absolute_risk(
    t0: f64,
    t1: f64,
    covariates: &Array1<f64>,
    cif_competing_t0: f64,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let cif0 = cumulative_incidence(t0, covariates, artifacts)?;
    let cif1 = cumulative_incidence(t1, covariates, artifacts)?;
    let delta = (cif1 - cif0).max(0.0);
    let denom = (1.0 - cif0 - cif_competing_t0).max(DEFAULT_RISK_EPSILON);
    Ok(delta / denom)
}

/// Calibrator feature extraction for survival predictions.
pub fn survival_calibrator_features(
    predictions: &Array1<f64>,
    standard_errors: &Array1<f64>,
    leverage: Option<&Array1<f64>>,
) -> Array2<f64> {
    let n = predictions.len();
    let leverage_len_ok = leverage.map_or(true, |l| l.len() == n);
    assert!(
        leverage_len_ok,
        "leverage vector must match prediction length"
    );
    let mut features = Array2::<f64>::zeros((n, if leverage.is_some() { 3 } else { 2 }));
    match leverage {
        Some(lev) => {
            for i in 0..n {
                features[[i, 0]] = predictions[i].logit();
                features[[i, 1]] = standard_errors[i];
                features[[i, 2]] = lev[i];
            }
        }
        None => {
            for i in 0..n {
                features[[i, 0]] = predictions[i].logit();
                features[[i, 1]] = standard_errors[i];
            }
        }
    }
    features
}

trait LogitExt {
    fn logit(self) -> f64;
}

impl LogitExt for f64 {
    fn logit(self) -> f64 {
        let clamped = self.max(1e-12).min(1.0 - 1e-12);
        (clamped / (1.0 - clamped)).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    fn toy_training_data() -> SurvivalTrainingData {
        SurvivalTrainingData {
            age_entry: array![50.0, 55.0, 60.0],
            age_exit: array![55.0, 60.0, 65.0],
            event_target: array![1, 0, 1],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            pgs: array![0.1, -0.2, 0.3],
            sex: array![0.0, 1.0, 0.0],
            pcs: array![[0.01, -0.02], [0.02, 0.03], [-0.04, 0.05]],
        }
    }

    fn evaluate_state(
        layout: &SurvivalLayout,
        penalty: &MonotonicityPenalty,
        data: &SurvivalTrainingData,
        beta: &Array1<f64>,
    ) -> WorkingState {
        let mut model = WorkingModelSurvival::new(
            layout.clone(),
            data,
            penalty.clone(),
            SurvivalSpec::default(),
        )
        .unwrap();
        model.update(beta).unwrap()
    }

    #[test]
    fn logit_extension_behaves() {
        assert!(0.5f64.logit().abs() < 1e-12);
        assert!(f64::is_finite(0.01f64.logit()));
    }

    #[test]
    fn monotonic_penalty_positive() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 1.0, 10).unwrap();
        let mut model =
            WorkingModelSurvival::new(layout, &data, penalty, SurvivalSpec::default()).unwrap();
        let beta = Array1::<f64>::zeros(model.layout.combined_exit.ncols());
        let state = model.update(&beta).unwrap();
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn conditional_risk_monotone() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 10).unwrap();
        let model = WorkingModelSurvival::new(
            layout.clone(),
            &data,
            penalty.clone(),
            SurvivalSpec::default(),
        )
        .unwrap();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(model.layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: CovariateLayout {
                column_names: vec![],
            },
            penalties: PenaltyDescriptor {
                order: 2,
                lambda: 0.5,
            },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
        };
        let covs = Array1::<f64>::zeros(model.layout.static_covariates.ncols());
        let cif0 = cumulative_incidence(55.0, &covs, &artifacts).unwrap();
        let cif1 = cumulative_incidence(60.0, &covs, &artifacts).unwrap();
        assert!(cif1 >= cif0 - 1e-9);
        let risk = conditional_absolute_risk(55.0, 60.0, &covs, 0.0, &artifacts).unwrap();
        assert!(risk >= -1e-9);
    }

    #[test]
    fn working_state_shapes() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 8).unwrap();
        let mut model =
            WorkingModelSurvival::new(layout, &data, penalty, SurvivalSpec::default()).unwrap();
        let beta = Array1::<f64>::zeros(model.layout.combined_exit.ncols());
        let state = model.update(&beta).unwrap();
        assert_eq!(state.gradient.len(), beta.len());
        assert_eq!(state.hessian.nrows(), beta.len());
        assert_eq!(state.hessian.ncols(), beta.len());
    }

    #[test]
    fn penalty_contributes_to_working_state() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.6, 0).unwrap();
        let mut penalised_layout = layout.clone();
        let mut unpenalised_layout = layout.clone();
        for block in &mut unpenalised_layout.penalties.blocks {
            block.lambda = 0.0;
        }

        let zero_monotonicity = MonotonicityPenalty {
            lambda: 0.0,
            derivative_design: monotonicity.derivative_design.clone(),
        };

        let mut beta = Array1::<f64>::zeros(penalised_layout.combined_exit.ncols());
        for (idx, value) in beta.iter_mut().enumerate() {
            *value = 0.03 * (idx as f64 + 1.0);
        }

        let penalised = evaluate_state(&penalised_layout, &zero_monotonicity, &data, &beta);
        let unpenalised = evaluate_state(&unpenalised_layout, &zero_monotonicity, &data, &beta);

        let penalty_deviance = penalised_layout.penalties.deviance(&beta);
        assert_abs_diff_eq!(
            penalised.deviance,
            unpenalised.deviance + penalty_deviance,
            epsilon = 1e-10
        );

        let penalty_gradient = penalised_layout.penalties.gradient(&beta);
        let expected_gradient = &unpenalised.gradient - &penalty_gradient;
        for (observed, expected) in penalised.gradient.iter().zip(expected_gradient.iter()) {
            assert_abs_diff_eq!(*observed, *expected, epsilon = 1e-10);
        }

        let penalty_hessian = penalised_layout.penalties.hessian(beta.len());
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
    }

    #[test]
    fn smoothing_penalty_matches_finite_difference() {
        let mut data = toy_training_data();
        data.sample_weight.fill(0.0);
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.75, 0).unwrap();
        let p = layout.combined_exit.ncols();
        let baseline_cols = layout.baseline_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..baseline_cols {
            let centered = idx as f64 - (baseline_cols as f64 / 2.0);
            beta[idx] = 0.05 * centered * centered;
        }
        for idx in baseline_cols..p {
            beta[idx] = -0.02 * (idx as f64 + 1.0);
        }

        let base_state = evaluate_state(&layout, &penalty, &data, &beta);
        let eps = 1e-6;

        for j in 0..p {
            let mut beta_plus = beta.clone();
            beta_plus[j] += eps;
            let plus_state = evaluate_state(&layout, &penalty, &data, &beta_plus);

            let mut beta_minus = beta.clone();
            beta_minus[j] -= eps;
            let minus_state = evaluate_state(&layout, &penalty, &data, &beta_minus);

            let numeric_grad = (plus_state.deviance - minus_state.deviance) / (2.0 * eps);
            assert!(
                (numeric_grad - base_state.gradient[j]).abs() < 1e-4,
                "gradient mismatch at index {}",
                j
            );

            let numeric_hessian_col = (&plus_state.gradient - &minus_state.gradient) / (2.0 * eps);
            for k in 0..p {
                let diff = numeric_hessian_col[k] - base_state.hessian[[k, j]];
                assert!(diff.abs() < 1e-3, "hessian mismatch at ({}, {})", k, j);
            }
        }
    }

    #[test]
    fn cumulative_hazard_respects_guard() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 6).unwrap();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: CovariateLayout {
                column_names: vec![],
            },
            penalties: PenaltyDescriptor {
                order: 2,
                lambda: 0.5,
            },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
        };
        let covs = Array1::<f64>::zeros(layout.static_covariates.ncols());

        let guard_floor = artifacts.age_transform.minimum_age - artifacts.age_transform.delta - 0.5;
        let err = cumulative_hazard(guard_floor, &covs, &artifacts).unwrap_err();
        assert!(matches!(err, SurvivalError::GuardDomainViolation { .. }));

        let ok_age = artifacts.age_transform.minimum_age;
        assert!(cumulative_hazard(ok_age, &covs, &artifacts).is_ok());

        // Ensure the monotonicity penalty builder is still exercised.
        assert_eq!(
            penalty.derivative_design.ncols(),
            layout.combined_exit.ncols()
        );
    }
}
