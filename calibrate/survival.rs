use crate::calibrate::basis::{
    BasisError, create_bspline_basis_with_knots, create_difference_penalty_matrix,
};
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::faer_ndarray::{FaerCholesky, FaerSvd, ldlt_rook};
use crate::calibrate::pirls::{WorkingModel, WorkingState};
use faer::Side;
use log::warn;
use ndarray::prelude::*;
use ndarray::{ArrayBase, ArrayView1, ArrayView2, Data, Ix1, Zip, arr1, concatenate};
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
    #[error("age transform guard delta must be positive")]
    NonPositiveGuard,
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
    #[error("linear predictor became non-finite during evaluation")]
    NonFiniteLinearPredictor,
    #[error("design matrix columns do not match coefficient length")]
    DesignDimensionMismatch,
    #[error("basis evaluation failed: {0}")]
    Basis(#[from] BasisError),
}

/// Guarded log-age transformation used across training and scoring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub minimum_age: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn from_training(age_entry: &Array1<f64>, delta: f64) -> Result<Self, SurvivalError> {
        if delta <= 0.0 {
            return Err(SurvivalError::NonPositiveGuard);
        }
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BasisDescriptor {
    pub knot_vector: Array1<f64>,
    pub degree: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnRange {
    pub start: usize,
    pub end: usize,
}

impl ColumnRange {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ValueRange {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CenteringTransform {
    pub offsets: Array1<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InteractionDescriptor {
    #[serde(default)]
    pub label: Option<String>,
    pub column_range: ColumnRange,
    #[serde(default)]
    pub value_ranges: Vec<ValueRange>,
    #[serde(default)]
    pub centering: Option<CenteringTransform>,
}

/// Stored smoothing metadata for reproduction at prediction time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyDescriptor {
    pub order: usize,
    pub lambda: f64,
    pub matrix: Array2<f64>,
    pub column_range: ColumnRange,
}

/// Column descriptions for static covariates.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CovariateLayout {
    pub column_names: Vec<String>,
    #[serde(default)]
    pub ranges: Vec<ValueRange>,
}

/// Borrowed per-subject covariate values used during scoring.
#[derive(Debug, Clone)]
pub struct CovariateValues<'a> {
    pub static_covariates: ArrayView1<'a, f64>,
    pub time_varying: Option<ArrayView1<'a, f64>>,
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
            grad_slice += &(2.0 * block.lambda * contrib);
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
                    hessian[[row_idx, col_idx]] +=
                        2.0 * block.lambda * block.matrix[[local_i, local_j]];
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
        validate_survival_inputs(
            self.age_entry.view(),
            self.age_exit.view(),
            self.event_target.view(),
            self.event_competing.view(),
            self.sample_weight.view(),
            self.pgs.view(),
            self.sex.view(),
            self.pcs.view(),
        )
    }
}

/// Validate the shared survival training and prediction arrays.
pub fn validate_survival_inputs(
    age_entry: ArrayView1<'_, f64>,
    age_exit: ArrayView1<'_, f64>,
    event_target: ArrayView1<'_, u8>,
    event_competing: ArrayView1<'_, u8>,
    sample_weight: ArrayView1<'_, f64>,
    pgs: ArrayView1<'_, f64>,
    sex: ArrayView1<'_, f64>,
    pcs: ArrayView2<'_, f64>,
) -> Result<(), SurvivalError> {
    let n = age_entry.len();
    if n == 0 {
        return Err(SurvivalError::EmptyAgeVector);
    }

    let dimension_mismatch = age_exit.len() != n
        || event_target.len() != n
        || event_competing.len() != n
        || sample_weight.len() != n
        || pgs.len() != n
        || sex.len() != n
        || pcs.nrows() != n;
    if dimension_mismatch {
        return Err(SurvivalError::CovariateDimensionMismatch);
    }

    for i in 0..n {
        let entry = age_entry[i];
        let exit = age_exit[i];
        if !entry.is_finite() || !exit.is_finite() {
            return Err(SurvivalError::NonFiniteAge);
        }
        if !(entry < exit) {
            return Err(SurvivalError::InvalidAgeOrder);
        }

        let target = event_target[i];
        let competing = event_competing[i];
        if target > 1 || competing > 1 {
            return Err(SurvivalError::InvalidEventFlag);
        }
        if target == 1 && competing == 1 {
            return Err(SurvivalError::ConflictingEvents);
        }

        let weight = sample_weight[i];
        if !weight.is_finite() || weight < 0.0 {
            return Err(SurvivalError::InvalidSampleWeight);
        }

        let pgs_value = pgs[i];
        let sex_value = sex[i];
        if !pgs_value.is_finite() || !sex_value.is_finite() {
            return Err(SurvivalError::NonFiniteCovariate);
        }

        for value in pcs.row(i) {
            if !value.is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn validate_survival_inputs(
    age_entry: ArrayView1<'_, f64>,
    age_exit: ArrayView1<'_, f64>,
    event_target: ArrayView1<'_, u8>,
    event_competing: ArrayView1<'_, u8>,
    sample_weight: ArrayView1<'_, f64>,
    pgs: ArrayView1<'_, f64>,
    sex: ArrayView1<'_, f64>,
    pcs: ArrayView2<'_, f64>,
) -> Result<(), SurvivalError> {
    let n = age_entry.len();
    if n == 0 {
        return Err(SurvivalError::EmptyAgeVector);
    }
    if age_exit.len() != n
        || event_target.len() != n
        || event_competing.len() != n
        || sample_weight.len() != n
        || pgs.len() != n
        || sex.len() != n
        || pcs.nrows() != n
    {
        return Err(SurvivalError::CovariateDimensionMismatch);
    }

    for idx in 0..n {
        let entry = age_entry[idx];
        let exit = age_exit[idx];
        if !entry.is_finite() || !exit.is_finite() {
            return Err(SurvivalError::NonFiniteAge);
        }
        if !(entry < exit) {
            return Err(SurvivalError::InvalidAgeOrder);
        }

        let target = event_target[idx];
        let competing = event_competing[idx];
        if target > 1 || competing > 1 {
            return Err(SurvivalError::InvalidEventFlag);
        }
        if target == 1 && competing == 1 {
            return Err(SurvivalError::ConflictingEvents);
        }

        let weight = sample_weight[idx];
        if !weight.is_finite() || weight < 0.0 {
            return Err(SurvivalError::InvalidSampleWeight);
        }

        if !pgs[idx].is_finite() || !sex[idx].is_finite() {
            return Err(SurvivalError::NonFiniteCovariate);
        }

        for col in 0..pcs.ncols() {
            let value = pcs[(idx, col)];
            if !value.is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
    }

    Ok(())
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

fn stable_softplus(x: f64) -> f64 {
    if x.is_infinite() {
        if x.is_sign_positive() { x } else { 0.0 }
    } else if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp().ln_1p()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn stable_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let exp_x = x.exp();
        exp_x / (1.0 + exp_x)
    }
}

fn accumulate_weighted_vector<S>(target: &mut Array1<f64>, scale: f64, values: &ArrayBase<S, Ix1>)
where
    S: Data<Elem = f64>,
{
    if scale == 0.0 {
        return;
    }
    Zip::from(target)
        .and(values)
        .for_each(|t, &v| *t += scale * v);
}

fn accumulate_symmetric_outer<S>(target: &mut Array2<f64>, scale: f64, values: &ArrayBase<S, Ix1>)
where
    S: Data<Elem = f64>,
{
    if scale == 0.0 {
        return;
    }
    let len = values.len();
    for j in 0..len {
        let vj = values[j];
        for k in j..len {
            let contribution = scale * vj * values[k];
            target[[j, k]] += contribution;
            if j != k {
                target[[k, j]] += contribution;
            }
        }
    }
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
        &data.age_entry,
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
    pub quadrature_design: Array2<f64>,
    pub grid_ages: Array1<f64>,
    pub quadrature_left: Array1<f64>,
    pub quadrature_right: Array1<f64>,
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
    ages_entry: &Array1<f64>,
    ages_exit: &Array1<f64>,
    grid_size: usize,
    lambda: f64,
) -> Result<MonotonicityPenalty, SurvivalError> {
    if grid_size == 0 {
        let cols = layout.combined_exit.ncols();
        return Ok(MonotonicityPenalty {
            lambda,
            derivative_design: Array2::<f64>::zeros((0, cols)),
            quadrature_design: Array2::<f64>::zeros((0, cols)),
            grid_ages: Array1::<f64>::zeros(0),
            quadrature_left: Array1::<f64>::zeros(0),
            quadrature_right: Array1::<f64>::zeros(0),
        });
    }

    let mut min_age = f64::INFINITY;
    let mut max_age = f64::NEG_INFINITY;
    for &age in ages_entry.iter().chain(ages_exit.iter()) {
        if age < min_age {
            min_age = age;
        }
        if age > max_age {
            max_age = age;
        }
    }
    if !min_age.is_finite() || !max_age.is_finite() || min_age >= max_age {
        let cols = layout.combined_exit.ncols();
        return Ok(MonotonicityPenalty {
            lambda,
            derivative_design: Array2::<f64>::zeros((0, cols)),
            quadrature_design: Array2::<f64>::zeros((0, cols)),
            grid_ages: Array1::<f64>::zeros(0),
            quadrature_left: Array1::<f64>::zeros(0),
            quadrature_right: Array1::<f64>::zeros(0),
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
    let (basis_grid, derivative_u) = evaluate_basis_and_derivative(log_grid.view(), age_basis)?;
    let constrained_basis_grid = layout.reference_constraint.apply(&basis_grid);
    let constrained_derivative_u = layout.reference_constraint.apply(&derivative_u);
    let mut derivative_age = constrained_derivative_u;
    for (mut row, &age) in derivative_age.rows_mut().into_iter().zip(grid.iter()) {
        let factor = layout.age_transform.derivative_factor(age)?;
        row *= factor;
    }

    let cols = layout.combined_exit.ncols();
    let mut combined = Array2::<f64>::zeros((grid_size, cols));
    let mut quadrature_design = Array2::<f64>::zeros((grid_size, cols));
    let baseline_cols = layout.baseline_exit.ncols();
    combined
        .slice_mut(s![.., ..baseline_cols])
        .assign(&derivative_age);
    quadrature_design
        .slice_mut(s![.., ..baseline_cols])
        .assign(&constrained_basis_grid);

    let mut quadrature_left = Array1::<f64>::zeros(grid_size);
    let mut quadrature_right = Array1::<f64>::zeros(grid_size);
    for idx in 0..grid_size {
        let left_bound = if idx == 0 {
            min_age
        } else {
            0.5 * (grid[idx - 1] + grid[idx])
        };
        let right_bound = if idx == grid_size - 1 {
            max_age
        } else {
            0.5 * (grid[idx] + grid[idx + 1])
        };
        quadrature_left[idx] = left_bound;
        quadrature_right[idx] = right_bound;
    }

    Ok(MonotonicityPenalty {
        lambda,
        derivative_design: combined,
        quadrature_design,
        grid_ages: grid,
        quadrature_left,
        quadrature_right,
    })
}

/// Royston–Parmar working model implementation.
pub struct WorkingModelSurvival {
    pub layout: SurvivalLayout,
    pub sample_weight: Array1<f64>,
    pub event_target: Array1<u8>,
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
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
            age_entry: data.age_entry.clone(),
            age_exit: data.age_exit.clone(),
            monotonicity,
            spec,
        })
    }

    fn build_expected_information_hessian(
        &self,
        beta: &Array1<f64>,
        barrier_hessian: &Array2<f64>,
        penalty_hessian: &Array2<f64>,
        monotonicity_hessian: Option<&Array2<f64>>,
    ) -> Result<Option<Array2<f64>>, SurvivalError> {
        let p = beta.len();
        let mut expected = Array2::<f64>::zeros((p, p));
        let baseline_cols = self.layout.baseline_exit.ncols();
        let time_cols = self
            .layout
            .time_varying_exit
            .as_ref()
            .map(|arr| arr.ncols())
            .unwrap_or(0);
        let static_cols = self.layout.static_covariates.ncols();
        let static_offset = baseline_cols + time_cols;
        let guard_threshold = self.spec.derivative_guard.max(f64::EPSILON);
        let mut design_buffer = Array1::<f64>::zeros(p);
        let quadrature_design = &self.monotonicity.quadrature_design;
        let quadrature_left = &self.monotonicity.quadrature_left;
        let quadrature_right = &self.monotonicity.quadrature_right;
        let has_quadrature = quadrature_design.nrows() > 0;

        let eta_exit_vec = self.layout.combined_exit.dot(beta);
        let eta_entry_vec = self.layout.combined_entry.dot(beta);

        for i in 0..self.age_entry.len() {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let entry_age = self.age_entry[i];
            let exit_age = self.age_exit[i];
            if !(exit_age > entry_age) {
                continue;
            }

            let eta_exit = eta_exit_vec[i];
            if !eta_exit.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let hazard_exit = eta_exit.exp();
            if !hazard_exit.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let eta_entry = eta_entry_vec[i];
            if !eta_entry.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let hazard_entry = eta_entry.exp();
            if !hazard_entry.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }

            if has_quadrature {
                for j in 0..quadrature_design.nrows() {
                    if quadrature_left[j] >= exit_age {
                        break;
                    }
                    if quadrature_right[j] <= entry_age {
                        continue;
                    }
                    let left = quadrature_left[j].max(entry_age);
                    let right = quadrature_right[j].min(exit_age);
                    if right <= left {
                        continue;
                    }
                    design_buffer.assign(&quadrature_design.row(j));
                    if static_cols > 0 {
                        design_buffer
                            .slice_mut(s![static_offset..static_offset + static_cols])
                            .assign(&self.layout.static_covariates.row(i));
                    }
                    let eta = design_buffer.dot(beta);
                    if !eta.is_finite() {
                        return Err(SurvivalError::NonFiniteLinearPredictor);
                    }
                    let hazard = eta.exp();
                    if !hazard.is_finite() {
                        return Err(SurvivalError::NonFiniteLinearPredictor);
                    }
                    let scale = weight * (right - left) * hazard;
                    accumulate_symmetric_outer(&mut expected, scale, &design_buffer);
                }
            } else {
                let width = exit_age - entry_age;
                if width <= 0.0 {
                    continue;
                }
                design_buffer.assign(&self.layout.combined_entry.row(i));
                design_buffer += &self.layout.combined_exit.row(i);
                design_buffer.mapv_inplace(|value| 0.5 * value);
                let hazard_mean = 0.5 * (hazard_exit + hazard_entry);
                if !hazard_mean.is_finite() {
                    return Err(SurvivalError::NonFiniteLinearPredictor);
                }
                let scale = weight * width * hazard_mean;
                accumulate_symmetric_outer(&mut expected, scale, &design_buffer);
            }

            let exit_design = self.layout.combined_exit.row(i);
            let derivative_exit = self.layout.combined_derivative_exit.row(i).dot(beta);
            if !derivative_exit.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let guarded = if derivative_exit <= guard_threshold {
                guard_threshold
            } else {
                derivative_exit
            };
            let scale = 1.0 / guarded;
            let mut x_tilde = exit_design.to_owned();
            Zip::from(&mut x_tilde)
                .and(&self.layout.combined_derivative_exit.row(i))
                .for_each(|value, &deriv| *value += deriv * scale);
            let mut delta_hazard = hazard_exit - hazard_entry;
            if delta_hazard < 0.0 {
                delta_hazard = 0.0;
            }
            let event_scale = weight * delta_hazard;
            accumulate_symmetric_outer(&mut expected, event_scale, &x_tilde);
        }

        expected.mapv_inplace(|value| value * -2.0);
        expected += barrier_hessian;
        expected += penalty_hessian;
        if let Some(extra) = monotonicity_hessian {
            expected += extra;
        }

        let mut neg_expected = expected.clone();
        neg_expected.mapv_inplace(|value| -value);
        let mut total_ridge = 0.0;
        let diag_scale = neg_expected
            .diag()
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max)
            .max(1.0);
        let mut attempt = 0usize;
        let max_attempts = 12usize;
        loop {
            match neg_expected.cholesky(Side::Lower) {
                Ok(_) => {
                    if total_ridge > 0.0 {
                        for idx in 0..p {
                            expected[(idx, idx)] -= total_ridge;
                        }
                    }
                    return Ok(Some(expected));
                }
                Err(_) => {
                    if attempt >= max_attempts {
                        return Ok(None);
                    }
                    let delta = if total_ridge == 0.0 {
                        diag_scale * 1e-10
                    } else {
                        total_ridge * 10.0
                    };
                    total_ridge += delta;
                    for idx in 0..p {
                        neg_expected[(idx, idx)] += delta;
                    }
                }
            }
            attempt += 1;
        }
    }
}

impl WorkingModelSurvival {
    fn update_internal(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        let expected_dim = self.layout.combined_exit.ncols();
        if beta.len() != expected_dim {
            return Err(SurvivalError::DesignDimensionMismatch);
        }

        let eta_exit = self.layout.combined_exit.dot(beta);
        let eta_entry = self.layout.combined_entry.dot(beta);
        let derivative_exit = self.layout.combined_derivative_exit.dot(beta);

        let n = eta_exit.len();
        let p = beta.len();
        let mut gradient = Array1::<f64>::zeros(p);
        let mut hessian = Array2::<f64>::zeros((p, p));
        let mut log_likelihood = 0.0;
        let mut barrier_deviance = 0.0;
        let mut barrier_gradient = Array1::<f64>::zeros(p);
        let mut barrier_hessian = Array2::<f64>::zeros((p, p));
        let mut guard_activation_count = 0usize;
        let mut negative_derivative_count = 0usize;
        let mut guard_examples: Vec<(usize, f64)> = Vec::new();
        let guard_threshold = self.spec.derivative_guard.max(f64::EPSILON);
        let h_exit = eta_exit.mapv(f64::exp);
        let h_entry = eta_entry.mapv(f64::exp);

        for i in 0..n {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let d = f64::from(self.event_target[i]);
            let eta_e = eta_exit[i];
            let h_e = h_exit[i];
            let h_s = h_entry[i];
            if !eta_e.is_finite() || !h_e.is_finite() || !h_s.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let d_eta_exit = derivative_exit[i];
            if !d_eta_exit.is_finite() {
                return Err(SurvivalError::NonFiniteLinearPredictor);
            }
            let guard_applied = d_eta_exit <= guard_threshold;
            let guarded_derivative = if guard_applied {
                guard_threshold
            } else {
                d_eta_exit
            };
            let log_guard = guarded_derivative.ln();
            let delta = h_e - h_s;
            log_likelihood += weight * (d * (eta_e + log_guard) - delta);

            if guard_applied {
                guard_activation_count += 1;
                if d_eta_exit < 0.0 {
                    negative_derivative_count += 1;
                    if guard_examples.len() < 5 {
                        guard_examples.push((i, d_eta_exit));
                    }
                }
            }

            let x_exit = self.layout.combined_exit.row(i);
            let x_entry = self.layout.combined_entry.row(i);
            let d_exit = self.layout.combined_derivative_exit.row(i);
            accumulate_weighted_vector(&mut gradient, -weight * h_e, &x_exit);
            accumulate_weighted_vector(&mut gradient, weight * h_s, &x_entry);

            let scale = 1.0 / guarded_derivative;
            let mut x_tilde = x_exit.to_owned();
            Zip::from(&mut x_tilde)
                .and(&d_exit)
                .for_each(|value, &deriv| *value += deriv * scale);

            if d > 0.0 {
                accumulate_weighted_vector(&mut gradient, weight * d, &x_tilde);
            }

            accumulate_symmetric_outer(&mut hessian, weight * h_e, &x_exit);
            accumulate_symmetric_outer(&mut hessian, weight * h_s, &x_entry);

            let event_scale = weight * d;
            if event_scale != 0.0 {
                accumulate_symmetric_outer(&mut hessian, event_scale, &x_tilde);
            }

            if self.spec.barrier_weight > 0.0 {
                let scaled = -d_eta_exit / self.spec.barrier_scale;
                let softplus = stable_softplus(scaled);
                barrier_deviance += 2.0 * self.spec.barrier_weight * weight * softplus;
                let sigmoid = stable_sigmoid(scaled);
                let barrier_grad_coeff =
                    2.0 * self.spec.barrier_weight * weight * sigmoid / self.spec.barrier_scale;
                accumulate_weighted_vector(&mut barrier_gradient, -barrier_grad_coeff, &d_exit);
                let barrier_hess_coeff =
                    2.0 * self.spec.barrier_weight * weight * sigmoid * (1.0 - sigmoid)
                        / (self.spec.barrier_scale * self.spec.barrier_scale);
                accumulate_symmetric_outer(&mut barrier_hessian, barrier_hess_coeff, &d_exit);
            }
        }

        gradient.mapv_inplace(|value| value * -2.0);
        hessian.mapv_inplace(|value| value * -2.0);
        gradient += &barrier_gradient;
        hessian += &barrier_hessian;
        let mut deviance = -2.0 * log_likelihood + barrier_deviance;

        let penalty_gradient = self.layout.penalties.gradient(beta);
        gradient += &penalty_gradient;
        let penalty_hessian = self.layout.penalties.hessian(beta.len());
        hessian += &penalty_hessian;
        deviance += self.layout.penalties.deviance(beta);

        let monotonicity_hessian =
            if self.monotonicity.lambda > 0.0 && self.monotonicity.derivative_design.nrows() > 0 {
                apply_monotonicity_penalty(
                    &self.monotonicity,
                    beta,
                    &mut gradient,
                    &mut hessian,
                    &mut deviance,
                )
            } else {
                None
            };

        let observed_information = hessian.clone();
        let observed_factorization = if p > 0 {
            match ldlt_rook(&observed_information) {
                Ok((lower, diag, subdiag, perm_fwd, perm_inv, inertia)) => {
                    Some(HessianFactor::Observed {
                        factor: LdltFactor {
                            lower,
                            diag,
                            subdiag,
                        },
                        permutation: PermutationDescriptor {
                            forward: perm_fwd,
                            inverse: perm_inv,
                        },
                        inertia,
                    })
                }
                Err(err) => {
                    warn!("LDLᵀ factorization of observed information failed: {err:?}");
                    None
                }
            }
        } else {
            None
        };
        let mut hessian_factor = observed_factorization.clone();

        if self.spec.use_expected_information {
            if let Some(expected_hessian) = self.build_expected_information_hessian(
                beta,
                &barrier_hessian,
                &penalty_hessian,
                monotonicity_hessian.as_ref(),
            )? {
                let chol_result = expected_hessian.cholesky(Side::Lower);
                if let Err(err) = &chol_result {
                    warn!("Cholesky factorization of expected information failed: {err:?}");
                }
                hessian = expected_hessian;
                if let Ok(cholesky) = chol_result {
                    hessian_factor = Some(HessianFactor::Expected {
                        factor: CholeskyFactor {
                            lower: cholesky.lower_triangular(),
                        },
                    });
                }
            }
        }

        if guard_activation_count > 0 {
            let guard_fraction = guard_activation_count as f64 / n as f64;
            warn!(
                "Derivative guard activated for {guard_activation_count} of {n} subjects ({:.2}% of sample). Negative derivatives observed for {} subjects. Example raw dη_exit values: {:?}.",
                guard_fraction * 100.0,
                negative_derivative_count,
                guard_examples
            );
        }

        Ok(WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
            hessian_factor,
        })
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, EstimationError> {
        self.update_internal(beta)
            .map_err(|err| EstimationError::InvalidSpecification(err.to_string()))
    }
}

fn apply_monotonicity_penalty(
    penalty: &MonotonicityPenalty,
    beta: &Array1<f64>,
    gradient: &mut Array1<f64>,
    hessian: &mut Array2<f64>,
    deviance: &mut f64,
) -> Option<Array2<f64>> {
    let lambda = penalty.lambda;
    if lambda == 0.0 {
        return None;
    }
    let design = &penalty.derivative_design;
    let values = design.dot(beta);
    let mut penalty_sum = 0.0;
    let mut violation_count = 0usize;
    let mut violation_examples: Vec<f64> = Vec::new();
    let mut hessian_update = Array2::<f64>::zeros((design.ncols(), design.ncols()));
    for (row, &value) in design.rows().into_iter().zip(values.iter()) {
        let softplus = stable_softplus(-value);
        penalty_sum += softplus;
        let sigma = stable_sigmoid(-value);
        let grad_scale = -2.0 * lambda * sigma;
        accumulate_weighted_vector(gradient, grad_scale, &row);
        let h_scale = 2.0 * lambda * sigma * (1.0 - sigma);
        accumulate_symmetric_outer(hessian, h_scale, &row);
        accumulate_symmetric_outer(&mut hessian_update, h_scale, &row);
        if value < 0.0 {
            violation_count += 1;
            if violation_examples.len() < 5 {
                violation_examples.push(value);
            }
        }
    }
    *deviance += 2.0 * lambda * penalty_sum;

    if design.nrows() > 0 && violation_count > 0 {
        let fraction = violation_count as f64 / design.nrows() as f64;
        warn!(
            "Monotonicity grid penalty activated for {violation_count} of {} evaluation ages ({:.2}% of grid). Example derivative values: {:?}.",
            design.nrows(),
            fraction * 100.0,
            violation_examples
        );
        if fraction > 0.05 {
            warn!(
                "More than 5% of grid ages triggered the monotonicity penalty; consider increasing baseline knots or smoothing."
            );
        }
    }
    Some(hessian_update)
}

/// Serialized representation of an LDLᵀ factor with Bunch–Kaufman pivoting.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LdltFactor {
    pub lower: Array2<f64>,
    pub diag: Array1<f64>,
    pub subdiag: Array1<f64>,
}

/// Serialized permutation metadata captured during factorization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermutationDescriptor {
    pub forward: Vec<usize>,
    pub inverse: Vec<usize>,
}

/// Stored factorization metadata for downstream diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HessianFactor {
    Observed {
        factor: LdltFactor,
        permutation: PermutationDescriptor,
        inertia: (usize, usize, usize),
    },
    Expected {
        factor: CholeskyFactor,
    },
}

/// Serialized Cholesky factor for SPD approximations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CholeskyFactor {
    pub lower: Array2<f64>,
}

impl LdltFactor {
    pub fn reconstruct(&self, permutation: &PermutationDescriptor) -> Array2<f64> {
        let n = self.lower.nrows();
        let mut d = Array2::<f64>::zeros((n, n));
        let mut idx = 0usize;
        while idx < n {
            if idx + 1 < n && self.subdiag[idx].abs() > 0.0 {
                d[(idx, idx)] = self.diag[idx];
                d[(idx + 1, idx + 1)] = self.diag[idx + 1];
                d[(idx, idx + 1)] = self.subdiag[idx];
                d[(idx + 1, idx)] = self.subdiag[idx];
                idx += 2;
            } else {
                d[(idx, idx)] = self.diag[idx];
                idx += 1;
            }
        }

        let ld = self.lower.dot(&d);
        let permuted = ld.dot(&self.lower.t());
        let mut restored = Array2::<f64>::zeros((n, n));
        for (i, &row) in permutation.forward.iter().enumerate() {
            for (j, &col) in permutation.forward.iter().enumerate() {
                restored[(row, col)] = permuted[(i, j)];
            }
        }
        restored
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CompanionModelHandle {
    pub reference: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: Vec<PenaltyDescriptor>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    #[serde(default)]
    pub interaction_metadata: Vec<InteractionDescriptor>,
    #[serde(default)]
    pub companion_models: Vec<CompanionModelHandle>,
    pub hessian_factor: Option<HessianFactor>,
}

impl SurvivalModelArtifacts {
    fn expected_static_dim(&self) -> usize {
        self.static_covariate_layout.column_names.len()
    }

    fn validate_covariates(&self, covariates: &CovariateValues<'_>) -> Result<(), SurvivalError> {
        if covariates.static_covariates.len() != self.expected_static_dim() {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
        Ok(())
    }

    fn evaluate_baseline_row(&self, log_age: f64) -> Result<Array1<f64>, SurvivalError> {
        let age_array = arr1(&[log_age]);
        let (basis_arc, _) = create_bspline_basis_with_knots(
            age_array.view(),
            self.age_basis.knot_vector.view(),
            self.age_basis.degree,
        )?;
        let basis = (*basis_arc).clone();
        let constrained = self.reference_constraint.apply(&basis);
        Ok(constrained.row(0).to_owned())
    }

    fn evaluate_time_varying_row(
        &self,
        log_age: f64,
    ) -> Result<Option<Array1<f64>>, SurvivalError> {
        if let Some(time_basis) = &self.time_varying_basis {
            let age_array = arr1(&[log_age]);
            let (tv_arc, _) = create_bspline_basis_with_knots(
                age_array.view(),
                time_basis.knot_vector.view(),
                time_basis.degree,
            )?;
            let tv = self.reference_constraint.apply(&(*tv_arc).clone());
            Ok(Some(tv.row(0).to_owned()))
        } else {
            Ok(None)
        }
    }

    fn assemble_design(
        &self,
        baseline: Array1<f64>,
        time_varying: Option<Array1<f64>>,
        covariates: &CovariateValues<'_>,
    ) -> Result<Array1<f64>, SurvivalError> {
        let mut design = Array1::<f64>::zeros(self.coefficients.len());
        let mut offset = 0usize;
        let total_len = design.len();
        let static_len = covariates.static_covariates.len();

        let baseline_len = baseline.len();
        if baseline_len > total_len {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        design
            .slice_mut(s![offset..offset + baseline_len])
            .assign(&baseline);
        offset += baseline_len;

        if let Some(time_row) = time_varying {
            let len = time_row.len();
            if offset + len > total_len {
                return Err(SurvivalError::DesignDimensionMismatch);
            }
            design.slice_mut(s![offset..offset + len]).assign(&time_row);
            offset += len;
        }

        if offset + static_len > total_len {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        let available_for_subject_time = total_len - offset - static_len;
        if let Some(subject_time) = &covariates.time_varying {
            let len = subject_time.len();
            if len > available_for_subject_time {
                return Err(SurvivalError::CovariateDimensionMismatch);
            }
            design
                .slice_mut(s![offset..offset + len])
                .assign(subject_time);
            offset += len;
        }

        design
            .slice_mut(s![offset..offset + static_len])
            .assign(&covariates.static_covariates);
        offset += static_len;

        if offset != total_len {
            return Err(SurvivalError::DesignDimensionMismatch);
        }
        Ok(design)
    }

    pub fn cumulative_hazard_with_covariates(
        &self,
        age: f64,
        covariates: &CovariateValues<'_>,
    ) -> Result<f64, SurvivalError> {
        self.validate_covariates(covariates)?;
        let log_age = self.age_transform.transform(age)?;
        let baseline = self.evaluate_baseline_row(log_age)?;
        let time_varying = self.evaluate_time_varying_row(log_age)?;
        let design = self.assemble_design(baseline, time_varying, covariates)?;
        let eta = design.dot(&self.coefficients);
        Ok(eta.exp())
    }

    pub fn cumulative_incidence_with_covariates(
        &self,
        age: f64,
        covariates: &CovariateValues<'_>,
    ) -> Result<f64, SurvivalError> {
        let hazard = self.cumulative_hazard_with_covariates(age, covariates)?;
        Ok(1.0 - (-hazard).exp())
    }

    pub fn conditional_absolute_risk_with_covariates(
        &self,
        t0: f64,
        t1: f64,
        covariates: &CovariateValues<'_>,
        cif_competing_t0: f64,
    ) -> Result<f64, SurvivalError> {
        let cif0 = self.cumulative_incidence_with_covariates(t0, covariates)?;
        let cif1 = self.cumulative_incidence_with_covariates(t1, covariates)?;
        let delta = (cif1 - cif0).max(0.0);
        let denom = (1.0 - cif0 - cif_competing_t0).max(DEFAULT_RISK_EPSILON);
        Ok(delta / denom)
    }
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
    let covariate_values = CovariateValues {
        static_covariates: covariates.view(),
        time_varying: None,
    };
    artifacts.cumulative_hazard_with_covariates(age, &covariate_values)
}

pub fn cumulative_incidence(
    age: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let covariate_values = CovariateValues {
        static_covariates: covariates.view(),
        time_varying: None,
    };
    artifacts.cumulative_incidence_with_covariates(age, &covariate_values)
}

pub fn conditional_absolute_risk(
    t0: f64,
    t1: f64,
    covariates: &Array1<f64>,
    cif_competing_t0: f64,
    artifacts: &SurvivalModelArtifacts,
) -> Result<f64, SurvivalError> {
    let covariate_values = CovariateValues {
        static_covariates: covariates.view(),
        time_varying: None,
    };
    artifacts.conditional_absolute_risk_with_covariates(t0, t1, &covariate_values, cif_competing_t0)
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
    use crate::calibrate::faer_ndarray::FaerEigh;
    use approx::assert_abs_diff_eq;
    use faer::Side;
    use ndarray::array;
    use serde_json;

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

    fn repeat_rows(matrix: &Array2<f64>, pattern: &[usize]) -> Array2<f64> {
        let cols = matrix.ncols();
        let mut result = Array2::<f64>::zeros((pattern.len(), cols));
        for (row_idx, &source_idx) in pattern.iter().enumerate() {
            assert!(source_idx < matrix.nrows());
            result.row_mut(row_idx).assign(&matrix.row(source_idx));
        }
        result
    }

    fn repeat_optional(matrix: &Option<Array2<f64>>, pattern: &[usize]) -> Option<Array2<f64>> {
        matrix.as_ref().map(|array| repeat_rows(array, pattern))
    }

    fn compute_value_ranges(matrix: &Array2<f64>) -> Vec<ValueRange> {
        (0..matrix.ncols())
            .map(|col_idx| {
                if matrix.nrows() == 0 {
                    return ValueRange { min: 0.0, max: 0.0 };
                }
                let mut min_val = f64::INFINITY;
                let mut max_val = f64::NEG_INFINITY;
                for &value in matrix.column(col_idx).iter() {
                    if value < min_val {
                        min_val = value;
                    }
                    if value > max_val {
                        max_val = value;
                    }
                }
                ValueRange {
                    min: min_val,
                    max: max_val,
                }
            })
            .collect()
    }

    fn make_covariate_layout(layout: &SurvivalLayout) -> CovariateLayout {
        let column_names: Vec<String> = (0..layout.static_covariates.ncols())
            .map(|idx| format!("cov{idx}"))
            .collect();
        let ranges = compute_value_ranges(&layout.static_covariates);
        CovariateLayout {
            column_names,
            ranges,
        }
    }

    fn baseline_penalty_descriptor(
        layout: &SurvivalLayout,
        order: usize,
        lambda: f64,
    ) -> PenaltyDescriptor {
        let baseline_cols = layout.baseline_exit.ncols();
        let matrix =
            create_difference_penalty_matrix(baseline_cols, order).expect("baseline penalty");
        PenaltyDescriptor {
            order,
            lambda,
            matrix,
            column_range: ColumnRange::new(0, baseline_cols),
        }
    }

    fn assert_array1_close(left: &Array1<f64>, right: &Array1<f64>, tol: f64) {
        assert_eq!(left.len(), right.len());
        for (l, r) in left.iter().zip(right.iter()) {
            assert!((l - r).abs() <= tol, "array1 mismatch: {l} vs {r}");
        }
    }

    fn assert_array2_close(left: &Array2<f64>, right: &Array2<f64>, tol: f64) {
        assert_eq!(left.dim(), right.dim());
        for (l, r) in left.iter().zip(right.iter()) {
            assert!((l - r).abs() <= tol, "array2 mismatch: {l} vs {r}");
        }
    }

    fn assert_artifacts_close(left: &SurvivalModelArtifacts, right: &SurvivalModelArtifacts) {
        assert_array1_close(&left.coefficients, &right.coefficients, 1e-12);
        assert_array1_close(
            &left.age_basis.knot_vector,
            &right.age_basis.knot_vector,
            1e-12,
        );
        assert_eq!(left.age_basis.degree, right.age_basis.degree);
        assert_eq!(left.time_varying_basis, right.time_varying_basis);
        assert_eq!(
            left.static_covariate_layout.column_names,
            right.static_covariate_layout.column_names
        );
        for (l_range, r_range) in left
            .static_covariate_layout
            .ranges
            .iter()
            .zip(&right.static_covariate_layout.ranges)
        {
            assert!((l_range.min - r_range.min).abs() <= 1e-12);
            assert!((l_range.max - r_range.max).abs() <= 1e-12);
        }
        assert_eq!(left.penalties.len(), right.penalties.len());
        for (l_penalty, r_penalty) in left.penalties.iter().zip(&right.penalties) {
            assert_eq!(l_penalty.order, r_penalty.order);
            assert!((l_penalty.lambda - r_penalty.lambda).abs() <= 1e-12);
            assert_array2_close(&l_penalty.matrix, &r_penalty.matrix, 1e-12);
            assert_eq!(l_penalty.column_range, r_penalty.column_range);
        }
        assert_array2_close(
            &left.reference_constraint.transform,
            &right.reference_constraint.transform,
            1e-12,
        );
        assert!(
            (left.reference_constraint.reference_log_age
                - right.reference_constraint.reference_log_age)
                .abs()
                <= 1e-12
        );
        assert!((left.age_transform.minimum_age - right.age_transform.minimum_age).abs() <= 1e-12);
        assert!((left.age_transform.delta - right.age_transform.delta).abs() <= 1e-12);
        assert_eq!(
            left.interaction_metadata.len(),
            right.interaction_metadata.len()
        );
        for (l_meta, r_meta) in left
            .interaction_metadata
            .iter()
            .zip(&right.interaction_metadata)
        {
            assert_eq!(l_meta.label, r_meta.label);
            assert_eq!(l_meta.column_range, r_meta.column_range);
            assert_eq!(l_meta.value_ranges.len(), r_meta.value_ranges.len());
            for (l_range, r_range) in l_meta.value_ranges.iter().zip(&r_meta.value_ranges) {
                assert!((l_range.min - r_range.min).abs() <= 1e-12);
                assert!((l_range.max - r_range.max).abs() <= 1e-12);
            }
            match (&l_meta.centering, &r_meta.centering) {
                (Some(l), Some(r)) => {
                    assert_array1_close(&l.offsets, &r.offsets, 1e-12);
                }
                (None, None) => {}
                _ => panic!("centering mismatch"),
            }
        }
        assert_eq!(left.companion_models, right.companion_models);
        match (&left.hessian_factor, &right.hessian_factor) {
            (
                Some(HessianFactor::Observed {
                    factor: l_factor,
                    permutation: l_perm,
                    inertia: l_inertia,
                }),
                Some(HessianFactor::Observed {
                    factor: r_factor,
                    permutation: r_perm,
                    inertia: r_inertia,
                }),
            ) => {
                assert_array2_close(&l_factor.lower, &r_factor.lower, 1e-12);
                assert_array1_close(&l_factor.diag, &r_factor.diag, 1e-12);
                assert_array1_close(&l_factor.subdiag, &r_factor.subdiag, 1e-12);
                assert_eq!(l_perm, r_perm);
                assert_eq!(l_inertia, r_inertia);
            }
            (
                Some(HessianFactor::Expected { factor: l_chol }),
                Some(HessianFactor::Expected { factor: r_chol }),
            ) => {
                assert_array2_close(&l_chol.lower, &r_chol.lower, 1e-12);
            }
            (None, None) => {}
            _ => panic!("hessian factor mismatch"),
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
    fn age_transform_rejects_non_positive_guard() {
        let ages = array![50.0, 55.0];
        let err = AgeTransform::from_training(&ages, 0.0).unwrap_err();
        assert!(matches!(err, SurvivalError::NonPositiveGuard));
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
        assert!(state.hessian_factor.is_some());
    }

    #[test]
    fn observed_information_factor_round_trips() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 8).unwrap();
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, penalty, SurvivalSpec::default())
                .unwrap();
        let p = layout.combined_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.01 * (idx as f64 + 1.0);
        }
        let state = model.update(&beta).unwrap();
        let observed = state.hessian.clone();
        let factor = state
            .hessian_factor
            .as_ref()
            .expect("observed factor should be present");
        match factor {
            HessianFactor::Observed {
                factor: ldlt,
                permutation,
                inertia,
            } => {
                assert_eq!(inertia.0 + inertia.1 + inertia.2, p);
                let reconstructed = ldlt.reconstruct(permutation);
                assert_array2_close(&observed, &reconstructed, 1e-10);
            }
            other => panic!("expected observed factor, found {other:?}"),
        }
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
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
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
    fn likelihood_matches_manual_computation() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.0, 0).unwrap();
        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.derivative_guard = 1e-12;
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let mut beta = Array1::<f64>::zeros(layout.combined_exit.ncols());
        for (idx, value) in beta.iter_mut().enumerate() {
            *value = 0.01 * (idx as f64 + 1.0);
        }

        let state = model.update(&beta).unwrap();
        let eta_exit = layout.combined_exit.dot(&beta);
        let eta_entry = layout.combined_entry.dot(&beta);
        let derivative_exit = layout.combined_derivative_exit.dot(&beta);
        let guard = spec.derivative_guard.max(f64::EPSILON);

        let mut manual = 0.0;
        for i in 0..data.age_entry.len() {
            let d = f64::from(data.event_target[i]);
            let weight = data.sample_weight[i];
            let guarded = derivative_exit[i].max(guard);
            let h_exit = eta_exit[i].exp();
            let h_entry = eta_entry[i].exp();
            manual += weight * (d * (eta_exit[i] + guarded.ln()) - (h_exit - h_entry));
        }

        assert_abs_diff_eq!(state.deviance, -2.0 * manual, epsilon = 1e-10);
    }

    #[test]
    fn left_truncation_matches_scoring_difference() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.45, 0.7, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.0, 4).unwrap();
        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.derivative_guard = 1e-12;
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        let baseline_cols = layout.baseline_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..baseline_cols {
            beta[idx] = 0.05 * (idx as f64 + 1.0);
        }

        let state = model.update(&beta).unwrap();
        assert!(state.deviance.is_finite());

        let artifacts = SurvivalModelArtifacts {
            coefficients: beta.clone(),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.0)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
            hessian_factor: None,
        };

        let eta_exit = layout.combined_exit.dot(&beta);
        let eta_entry = layout.combined_entry.dot(&beta);

        for i in 0..data.age_entry.len() {
            let covariates = layout.static_covariates.row(i).to_owned();
            let hazard_exit = cumulative_hazard(data.age_exit[i], &covariates, &artifacts).unwrap();
            let hazard_entry =
                cumulative_hazard(data.age_entry[i], &covariates, &artifacts).unwrap();
            let delta_scoring = hazard_exit - hazard_entry;
            let delta_training = eta_exit[i].exp() - eta_entry[i].exp();
            assert_abs_diff_eq!(delta_scoring, delta_training, epsilon = 1e-10);
        }
    }

    #[test]
    fn gradient_and_hessian_match_numeric() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.0, 0).unwrap();
        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.derivative_guard = 1e-12;
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        let baseline_cols = layout.baseline_exit.ncols();
        for idx in 0..baseline_cols {
            beta[idx] = 0.05 * (idx as f64 + 1.0);
        }
        for idx in baseline_cols..p {
            beta[idx] = 0.01 * (idx as f64 + 1.0);
        }

        let guard = spec.derivative_guard.max(f64::EPSILON);
        let derivative_exit = layout.combined_derivative_exit.dot(&beta);
        if derivative_exit.iter().any(|value| *value <= guard * 10.0) {
            for idx in 0..baseline_cols {
                beta[idx] += 0.05;
            }
        }

        let base_state = model.update(&beta).unwrap();
        let eta_exit = layout.combined_exit.dot(&beta);
        let eta_entry = layout.combined_entry.dot(&beta);
        let derivative_vector = layout.combined_derivative_exit.dot(&beta);
        let h_exit = eta_exit.mapv(f64::exp);
        let h_entry = eta_entry.mapv(f64::exp);

        let mut manual_gradient = Array1::<f64>::zeros(p);
        let mut manual_hessian = Array2::<f64>::zeros((p, p));

        for i in 0..data.age_entry.len() {
            let weight = data.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let d = f64::from(data.event_target[i]);
            let x_exit_row = layout.combined_exit.row(i);
            let x_entry_row = layout.combined_entry.row(i);
            let d_exit_row = layout.combined_derivative_exit.row(i);
            let guard_threshold = spec.derivative_guard.max(f64::EPSILON);
            let raw_derivative = derivative_vector[i];
            let guard_applied = raw_derivative <= guard_threshold;
            let guarded = if guard_applied {
                guard_threshold
            } else {
                raw_derivative
            };
            let scale = 1.0 / guarded;
            let mut x_tilde = x_exit_row.to_owned();
            Zip::from(&mut x_tilde)
                .and(&d_exit_row)
                .for_each(|value, &deriv| *value += deriv * scale);

            accumulate_weighted_vector(&mut manual_gradient, 2.0 * weight * h_exit[i], &x_exit_row);
            accumulate_weighted_vector(
                &mut manual_gradient,
                -2.0 * weight * h_entry[i],
                &x_entry_row,
            );
            if d > 0.0 {
                accumulate_weighted_vector(&mut manual_gradient, -2.0 * weight * d, &x_tilde);
            }

            accumulate_symmetric_outer(&mut manual_hessian, -2.0 * weight * h_exit[i], &x_exit_row);
            accumulate_symmetric_outer(
                &mut manual_hessian,
                -2.0 * weight * h_entry[i],
                &x_entry_row,
            );
            let event_scale = weight * d;
            if event_scale != 0.0 {
                accumulate_symmetric_outer(&mut manual_hessian, -2.0 * event_scale, &x_tilde);
            }
        }

        for (observed, expected) in manual_gradient.iter().zip(base_state.gradient.iter()) {
            assert_abs_diff_eq!(*observed, *expected, epsilon = 1e-8);
        }

        for (manual_row, observed_row) in manual_hessian
            .rows()
            .into_iter()
            .zip(base_state.hessian.rows())
        {
            for (manual_val, observed_val) in manual_row.iter().zip(observed_row.iter()) {
                assert_abs_diff_eq!(*manual_val, *observed_val, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn deviance_decreases_with_expected_newton_step() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 0.75, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.2, 6).unwrap();
        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.use_expected_information = true;
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.02 * (idx as f64 + 1.0);
        }

        let state_initial = model.update(&beta).unwrap();
        let mut step = 1e-3;
        let mut beta_next = beta.clone();
        let mut state_next = state_initial.clone();
        loop {
            beta_next = &beta - &(state_initial.gradient.mapv(|g| step * g));
            match model.update(&beta_next) {
                Ok(next) => {
                    state_next = next;
                    if state_next.deviance < state_initial.deviance {
                        break;
                    }
                }
                Err(SurvivalError::NonFiniteLinearPredictor) => {
                    step *= 0.5;
                    assert!(
                        step > 1e-8,
                        "unable to reduce deviance via gradient descent"
                    );
                    continue;
                }
                Err(other) => panic!("unexpected update failure: {other:?}"),
            }
            step *= 0.5;
            assert!(
                step > 1e-8,
                "unable to reduce deviance via gradient descent"
            );
        }
        assert!(state_next.deviance < state_initial.deviance);
    }

    #[test]
    fn expected_information_adjusts_hessian() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.45, 0.75, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.0, 6).unwrap();
        let mut spec_observed = SurvivalSpec::default();
        spec_observed.barrier_weight = 0.0;
        spec_observed.use_expected_information = false;
        let mut spec_expected = spec_observed;
        spec_expected.use_expected_information = true;

        let mut observed_model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec_observed)
                .unwrap();
        let mut expected_model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec_expected)
                .unwrap();

        let p = layout.combined_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.015 * (idx as f64 + 1.0);
        }

        let observed_state = observed_model.update(&beta).unwrap();
        let expected_state = expected_model.update(&beta).unwrap();

        for (obs, exp) in observed_state
            .gradient
            .iter()
            .zip(expected_state.gradient.iter())
        {
            assert_abs_diff_eq!(*obs, *exp, epsilon = 1e-10);
        }

        let diff = &expected_state.hessian - &observed_state.hessian;
        let diff_norm: f64 = diff.iter().map(|v| v.abs()).sum();
        assert!(diff_norm > 1e-8);

        let mut neg_expected = expected_state.hessian.clone();
        neg_expected.mapv_inplace(|value| -value);
        let (eigenvalues, _) = neg_expected
            .eigh(Side::Lower)
            .expect("eigendecomposition should succeed for SPD approximation");
        for value in eigenvalues.iter() {
            assert!(
                *value >= -1e-9,
                "expected-information Hessian not SPD: eigenvalue {}",
                value
            );
        }

        match observed_state.hessian_factor {
            Some(HessianFactor::Observed { .. }) => {}
            other => panic!("observed state stored unexpected factor {other:?}"),
        }
        match expected_state.hessian_factor {
            Some(HessianFactor::Expected { .. }) => {}
            other => panic!("expected state stored unexpected factor {other:?}"),
        }
    }

    #[test]
    fn expected_information_matrix_is_spd() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.4, 0.7, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.2, 6).unwrap();
        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.use_expected_information = true;
        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();

        let p = layout.combined_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.01 * (idx as f64 + 1.0);
        }

        let state = model.update(&beta).unwrap();
        let mut neg_expected = state.hessian.clone();
        neg_expected.mapv_inplace(|value| -value);
        let factor = neg_expected
            .cholesky(Side::Lower)
            .expect("expected-information matrix should be SPD");
        for value in factor.diag().iter() {
            assert!(*value > 0.0);
        }
    }

    #[test]
    fn expected_information_step_progresses() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 0.8, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.15, 8).unwrap();
        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.use_expected_information = true;

        let mut base_model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec).unwrap();
        let p = layout.combined_exit.ncols();
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.02 * (idx as f64 + 0.5);
        }

        let base_state = base_model.update(&beta).unwrap();
        let mut neg_expected = base_state.hessian.clone();
        neg_expected.mapv_inplace(|value| -value);
        let factor = neg_expected
            .cholesky(Side::Lower)
            .expect("expected-information matrix should be SPD");
        let step = factor.solve_vec(&base_state.gradient);

        let mut scale = 1.0;
        let mut improved = None;
        while scale > 1e-6 {
            let scaled_step = step.mapv(|v| v * scale);
            let beta_candidate = &beta - &scaled_step;
            let mut trial_model =
                WorkingModelSurvival::new(layout.clone(), &data, monotonicity.clone(), spec)
                    .unwrap();
            match trial_model.update(&beta_candidate) {
                Ok(candidate_state) => {
                    if candidate_state.deviance < base_state.deviance {
                        improved = Some(candidate_state.deviance);
                        break;
                    }
                }
                Err(SurvivalError::NonFiniteLinearPredictor) => {}
                Err(other) => panic!("unexpected update failure: {other:?}"),
            }
            scale *= 0.5;
        }

        assert!(
            improved.is_some(),
            "expected-information step failed to reduce deviance"
        );
    }

    #[test]
    fn frequency_weights_match_replication() {
        let weighted_data = SurvivalTrainingData {
            age_entry: array![50.0, 55.0],
            age_exit: array![55.0, 60.0],
            event_target: array![1, 0],
            event_competing: array![0, 0],
            sample_weight: array![1.0, 2.0],
            pgs: array![0.1, -0.3],
            sex: array![0.0, 1.0],
            pcs: array![[0.01, -0.02], [0.02, 0.03]],
        };

        let expanded_data = SurvivalTrainingData {
            age_entry: array![50.0, 55.0, 55.0],
            age_exit: array![55.0, 60.0, 60.0],
            event_target: array![1, 0, 0],
            event_competing: array![0, 0, 0],
            sample_weight: array![1.0, 1.0, 1.0],
            pgs: array![0.1, -0.3, -0.3],
            sex: array![0.0, 1.0, 1.0],
            pcs: array![[0.01, -0.02], [0.02, 0.03], [0.02, 0.03]],
        };

        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 0.75, 1.0, 1.0, 1.0],
            degree: 2,
        };

        let (layout_weighted, monotonic_weighted) =
            build_survival_layout(&weighted_data, &basis, 0.1, 2, 0.0, 0).unwrap();
        let replicate_pattern = [0usize, 1, 1];
        let layout_expanded = SurvivalLayout {
            baseline_entry: repeat_rows(&layout_weighted.baseline_entry, &replicate_pattern),
            baseline_exit: repeat_rows(&layout_weighted.baseline_exit, &replicate_pattern),
            baseline_derivative_exit: repeat_rows(
                &layout_weighted.baseline_derivative_exit,
                &replicate_pattern,
            ),
            time_varying_entry: repeat_optional(
                &layout_weighted.time_varying_entry,
                &replicate_pattern,
            ),
            time_varying_exit: repeat_optional(
                &layout_weighted.time_varying_exit,
                &replicate_pattern,
            ),
            time_varying_derivative_exit: repeat_optional(
                &layout_weighted.time_varying_derivative_exit,
                &replicate_pattern,
            ),
            static_covariates: repeat_rows(&layout_weighted.static_covariates, &replicate_pattern),
            age_transform: layout_weighted.age_transform,
            reference_constraint: layout_weighted.reference_constraint.clone(),
            penalties: layout_weighted.penalties.clone(),
            combined_entry: repeat_rows(&layout_weighted.combined_entry, &replicate_pattern),
            combined_exit: repeat_rows(&layout_weighted.combined_exit, &replicate_pattern),
            combined_derivative_exit: repeat_rows(
                &layout_weighted.combined_derivative_exit,
                &replicate_pattern,
            ),
        };
        let monotonic_expanded = monotonic_weighted.clone();

        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.derivative_guard = 1e-12;

        let mut weighted_model = WorkingModelSurvival::new(
            layout_weighted.clone(),
            &weighted_data,
            monotonic_weighted.clone(),
            spec,
        )
        .unwrap();
        let mut expanded_model = WorkingModelSurvival::new(
            layout_expanded.clone(),
            &expanded_data,
            monotonic_expanded,
            spec,
        )
        .unwrap();

        let p = layout_weighted.combined_exit.ncols();
        assert_eq!(p, layout_expanded.combined_exit.ncols());
        let mut beta = Array1::<f64>::zeros(p);
        for idx in 0..p {
            beta[idx] = 0.03 * (idx as f64 + 1.0);
        }

        let state_weighted = weighted_model.update(&beta).unwrap();
        let state_expanded = expanded_model.update(&beta).unwrap();

        assert_abs_diff_eq!(
            state_weighted.deviance,
            state_expanded.deviance,
            epsilon = 1e-4
        );
        for (g_weighted, g_expanded) in state_weighted
            .gradient
            .iter()
            .zip(state_expanded.gradient.iter())
        {
            assert_abs_diff_eq!(*g_weighted, *g_expanded, epsilon = 1e-4);
        }
        for (h_weighted, h_expanded) in state_weighted
            .hessian
            .iter()
            .zip(state_expanded.hessian.iter())
        {
            assert_abs_diff_eq!(*h_weighted, *h_expanded, epsilon = 1e-4);
        }
    }

    #[test]
    fn penalty_contributes_to_working_state() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, monotonicity) = build_survival_layout(&data, &basis, 0.1, 2, 0.6, 0).unwrap();
        let penalised_layout = layout.clone();
        let mut unpenalised_layout = layout.clone();
        for block in &mut unpenalised_layout.penalties.blocks {
            block.lambda = 0.0;
        }

        let zero_monotonicity = MonotonicityPenalty {
            lambda: 0.0,
            derivative_design: monotonicity.derivative_design.clone(),
            quadrature_design: monotonicity.quadrature_design.clone(),
            grid_ages: monotonicity.grid_ages.clone(),
            quadrature_left: monotonicity.quadrature_left.clone(),
            quadrature_right: monotonicity.quadrature_right.clone(),
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
        let expected_gradient = &unpenalised.gradient + &penalty_gradient;
        for (observed, expected) in penalised.gradient.iter().zip(expected_gradient.iter()) {
            assert_abs_diff_eq!(*observed, *expected, epsilon = 1e-10);
        }

        let penalty_hessian = penalised_layout.penalties.hessian(beta.len());
        let expected_hessian = &unpenalised.hessian + &penalty_hessian;
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
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
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

    #[test]
    fn cumulative_hazard_rejects_covariate_mismatch() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, _) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 4).unwrap();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
            hessian_factor: None,
        };
        let mismatched_covs = Array1::<f64>::zeros(layout.static_covariates.ncols() + 1);
        let err = cumulative_hazard(60.0, &mismatched_covs, &artifacts).unwrap_err();
        assert!(matches!(err, SurvivalError::CovariateDimensionMismatch));
    }

    #[test]
    fn cumulative_hazard_rejects_time_varying_mismatch() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, _) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 4).unwrap();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
            hessian_factor: None,
        };

        let static_covs = Array1::<f64>::zeros(layout.static_covariates.ncols());
        let extra_time = array![1.0];
        let cov_values = CovariateValues {
            static_covariates: static_covs.view(),
            time_varying: Some(extra_time.view()),
        };

        let err = artifacts
            .cumulative_hazard_with_covariates(60.0, &cov_values)
            .unwrap_err();
        assert!(matches!(err, SurvivalError::CovariateDimensionMismatch));
    }

    #[test]
    fn survival_artifacts_round_trip_serialization() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, _) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 4).unwrap();
        let penalty = baseline_penalty_descriptor(&layout, 2, 0.5);
        let interaction = InteractionDescriptor {
            label: Some("pgs_by_age".to_string()),
            column_range: ColumnRange::new(1, 3),
            value_ranges: vec![ValueRange {
                min: -0.5,
                max: 0.5,
            }],
            centering: Some(CenteringTransform {
                offsets: array![0.1, -0.1],
            }),
        };
        let companion = CompanionModelHandle {
            reference: "competing-risk-model".to_string(),
        };
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![penalty],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            interaction_metadata: vec![interaction],
            companion_models: vec![companion],
            hessian_factor: Some(HessianFactor::Expected {
                factor: CholeskyFactor {
                    lower: Array2::<f64>::eye(layout.combined_exit.ncols()),
                },
            }),
        };

        let serialized = serde_json::to_string(&artifacts).unwrap();
        let round_trip: SurvivalModelArtifacts = serde_json::from_str(&serialized).unwrap();
        assert_artifacts_close(&artifacts, &round_trip);
    }

    #[test]
    fn survival_artifacts_round_trip_serialization_observed_factor() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, _) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 4).unwrap();
        let p = layout.combined_exit.ncols();
        let ldlt = LdltFactor {
            lower: Array2::<f64>::eye(p),
            diag: Array1::<f64>::from_elem(p, 1.0),
            subdiag: Array1::<f64>::zeros(p),
        };
        let identity_perm: Vec<usize> = (0..p).collect();
        let permutation = PermutationDescriptor {
            forward: identity_perm.clone(),
            inverse: identity_perm,
        };
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(p),
            age_basis: basis.clone(),
            time_varying_basis: None,
            static_covariate_layout: make_covariate_layout(&layout),
            penalties: vec![baseline_penalty_descriptor(&layout, 2, 0.5)],
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            interaction_metadata: Vec::new(),
            companion_models: Vec::new(),
            hessian_factor: Some(HessianFactor::Observed {
                factor: ldlt,
                permutation,
                inertia: (p, 0, 0),
            }),
        };

        let serialized = serde_json::to_string(&artifacts).unwrap();
        let round_trip: SurvivalModelArtifacts = serde_json::from_str(&serialized).unwrap();
        assert_artifacts_close(&artifacts, &round_trip);
    }
}
