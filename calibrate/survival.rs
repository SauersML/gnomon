use crate::calibrate::basis::{
    BasisError, create_bspline_basis_with_knots, create_difference_penalty_matrix,
};
use crate::calibrate::faer_ndarray::FaerSvd;
use log::warn;
use ndarray::prelude::*;
use ndarray::{ArrayBase, Data, Ix1, Zip, concatenate};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use thiserror::Error;

use crate::calibrate::pirls::{WorkingModel, WorkingState};

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
    #[error("time-varying interaction metadata missing from artifacts")]
    MissingTimeVaryingMetadata,
    #[error("time-varying centering vector length mismatch")]
    TimeVaryingDimensionMismatch,
}

/// Guarded log-age transformation used across training and scoring.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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
    pub baseline_order: usize,
    pub baseline_lambda: f64,
    pub interaction: Option<InteractionPenaltyDescriptor>,
}

/// Stored interaction penalty metadata for anisotropic tensor products.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionPenaltyDescriptor {
    pub lambda_age: f64,
    pub lambda_pgs: f64,
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
    pub time_varying_centering: Option<Array1<f64>>,
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

pub(crate) fn validate_survival_inputs(
    age_entry: ArrayView1<f64>,
    age_exit: ArrayView1<f64>,
    event_target: ArrayView1<u8>,
    event_competing: ArrayView1<u8>,
    sample_weight: ArrayView1<f64>,
    pgs: ArrayView1<f64>,
    sex: ArrayView1<f64>,
    pcs: ArrayView2<f64>,
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
        for j in 0..pcs.ncols() {
            if !pcs[[i, j]].is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
    }

    Ok(())
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
    time_varying: Option<TimeVaryingConfig>,
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

    let mut baseline_derivative_exit = constrained_derivative_exit_u.clone();
    for (mut row, age) in baseline_derivative_exit
        .rows_mut()
        .into_iter()
        .zip(data.age_exit.iter().copied())
    {
        let factor = age_transform.derivative_factor(age)?;
        row.mapv_inplace(|v| v * factor);
    }

    let mut time_varying_entry = None;
    let mut time_varying_exit = None;
    let mut time_varying_derivative_exit = None;
    let mut time_varying_centering = None;

    if time_varying.is_some() {
        let pgs_column = data.pgs.clone().insert_axis(Axis(1));
        let mut entry_interaction =
            row_wise_tensor_product(&pgs_column, &constrained_entry);
        let mut exit_interaction =
            row_wise_tensor_product(&pgs_column, &constrained_exit);
        let derivative_u_interaction =
            row_wise_tensor_product(&pgs_column, &constrained_derivative_exit_u);
        let mut derivative_interaction = derivative_u_interaction;
        for (mut row, age) in derivative_interaction
            .rows_mut()
            .into_iter()
            .zip(data.age_exit.iter().copied())
        {
            let factor = age_transform.derivative_factor(age)?;
            row.mapv_inplace(|v| v * factor);
        }

        let centering = weighted_column_means(&exit_interaction, &data.sample_weight);
        for (mut col, &mean) in entry_interaction.columns_mut().into_iter().zip(centering.iter()) {
            col.mapv_inplace(|v| v - mean);
        }
        for (mut col, &mean) in exit_interaction.columns_mut().into_iter().zip(centering.iter()) {
            col.mapv_inplace(|v| v - mean);
        }

        time_varying_entry = Some(entry_interaction);
        time_varying_exit = Some(exit_interaction);
        time_varying_derivative_exit = Some(derivative_interaction);
        time_varying_centering = Some(centering);
    }

    let static_covariates = assemble_static_covariates(data);

    let combined_entry = concatenate_design(
        &constrained_entry,
        time_varying_entry.as_ref(),
        &static_covariates,
    );
    let combined_exit = concatenate_design(
        &constrained_exit,
        time_varying_exit.as_ref(),
        &static_covariates,
    );
    let zero_static = Array2::<f64>::zeros((n, static_covariates.ncols()));
    let combined_derivative_exit = concatenate_design(
        &baseline_derivative_exit,
        time_varying_derivative_exit.as_ref(),
        &zero_static,
    );

    let baseline_cols = constrained_exit.ncols();
    let mut penalty_blocks = Vec::new();
    let baseline_penalty_matrix =
        create_difference_penalty_matrix(baseline_cols, baseline_penalty_order)?;
    penalty_blocks.push(PenaltyBlock {
        matrix: baseline_penalty_matrix,
        lambda: baseline_lambda,
        range: 0..baseline_cols,
    });

    if let (Some(tv_exit), Some(config)) = (&time_varying_exit, time_varying) {
        let interaction_cols = tv_exit.ncols();
        if interaction_cols > 0 {
            let interaction_range = baseline_cols..baseline_cols + interaction_cols;
            let age_penalty_matrix =
                create_difference_penalty_matrix(interaction_cols, baseline_penalty_order)?;
            penalty_blocks.push(PenaltyBlock {
                matrix: age_penalty_matrix,
                lambda: config.lambda_age,
                range: interaction_range.clone(),
            });
            let ridge = Array2::<f64>::eye(interaction_cols);
            penalty_blocks.push(PenaltyBlock {
                matrix: ridge,
                lambda: config.lambda_pgs,
                range: interaction_range,
            });
        }
    }

    let penalties = PenaltyBlocks::new(penalty_blocks);

    let layout = SurvivalLayout {
        baseline_entry: constrained_entry,
        baseline_exit: constrained_exit,
        baseline_derivative_exit,
        time_varying_entry,
        time_varying_exit,
        time_varying_derivative_exit,
        time_varying_centering,
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

fn row_wise_tensor_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    assert_eq!(a.nrows(), b.nrows(), "tensor product requires equal rows");
    let n = a.nrows();
    let a_cols = a.ncols();
    let b_cols = b.ncols();
    if n == 0 || a_cols == 0 || b_cols == 0 {
        return Array2::<f64>::zeros((n, a_cols * b_cols));
    }
    let mut result = Array2::<f64>::zeros((n, a_cols * b_cols));
    for i in 0..n {
        let mut idx = 0usize;
        let row_a = a.row(i);
        let row_b = b.row(i);
        for &va in row_a.iter() {
            for &vb in row_b.iter() {
                result[[i, idx]] = va * vb;
                idx += 1;
            }
        }
    }
    result
}

fn weighted_column_means(design: &Array2<f64>, weights: &Array1<f64>) -> Array1<f64> {
    assert_eq!(design.nrows(), weights.len(), "weights must match rows");
    let denom = weights.sum();
    if denom <= 0.0 {
        return Array1::<f64>::zeros(design.ncols());
    }
    let mut means = Array1::<f64>::zeros(design.ncols());
    for (row, &w) in design.rows().into_iter().zip(weights.iter()) {
        if w == 0.0 {
            continue;
        }
        Zip::from(&mut means).and(&row).for_each(|m, &value| {
            *m += w * value;
        });
    }
    means.mapv(|value| value / denom)
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
    type Error = SurvivalError;

    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, Self::Error> {
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

            let event_scale = if self.spec.use_expected_information {
                weight * h_e
            } else {
                weight * d
            };
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

        if self.monotonicity.lambda > 0.0 && self.monotonicity.derivative_design.nrows() > 0 {
            apply_monotonicity_penalty(
                &self.monotonicity,
                beta,
                &mut gradient,
                &mut hessian,
                &mut deviance,
            );
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
    let mut violation_count = 0usize;
    let mut violation_examples: Vec<f64> = Vec::new();
    for (row, &value) in design.rows().into_iter().zip(values.iter()) {
        let softplus = stable_softplus(-value);
        penalty_sum += softplus;
        let sigma = stable_sigmoid(-value);
        let grad_scale = -2.0 * lambda * sigma;
        accumulate_weighted_vector(gradient, grad_scale, &row);
        let h_scale = 2.0 * lambda * sigma * (1.0 - sigma);
        accumulate_symmetric_outer(hessian, h_scale, &row);
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

impl HessianFactor {
    /// Return the inertia (negative, zero, positive eigenvalue counts) implied by the
    /// stored factorization. This is primarily useful for diagnostics that need to
    /// understand how many directions of negative curvature were encountered while
    /// solving the penalized system.
    pub fn inertia(&self) -> (usize, usize, usize) {
        match self {
            HessianFactor::Observed { inertia, .. } => *inertia,
            HessianFactor::Expected { cholesky_factor } => {
                let dim = cholesky_factor.nrows();
                (0, 0, dim)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub time_varying_centering: Option<Array1<f64>>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: PenaltyDescriptor,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<HessianFactor>,
    pub pgs_column_index: Option<usize>,
}

/// Optional configuration for tensor-product PGS×age interactions.
#[derive(Debug, Clone, Copy)]
pub struct TimeVaryingConfig {
    pub lambda_age: f64,
    pub lambda_pgs: f64,
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
    let expected_covs = artifacts.static_covariate_layout.column_names.len();
    if covariates.len() != expected_covs {
        return Err(SurvivalError::CovariateDimensionMismatch);
    }
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
        let pgs_index = artifacts
            .pgs_column_index
            .ok_or(SurvivalError::MissingTimeVaryingMetadata)?;
        if pgs_index >= covariates.len() {
            return Err(SurvivalError::CovariateDimensionMismatch);
        }
        let pgs_value = covariates[pgs_index];
        let (tv_arc, _) = create_bspline_basis_with_knots(
            array![log_age].view(),
            time_basis.knot_vector.view(),
            time_basis.degree,
        )?;
        let tv = artifacts.reference_constraint.apply(&(*tv_arc).clone());
        let mut interaction_row = tv.row(0).to_owned();
        interaction_row.mapv_inplace(|v| v * pgs_value);
        if let Some(centering) = &artifacts.time_varying_centering {
            if centering.len() != interaction_row.len() {
                return Err(SurvivalError::TimeVaryingDimensionMismatch);
            }
            interaction_row -= centering;
        }
        design = concatenate(Axis(0), &[design.view(), interaction_row.view()])
            .expect("time concat");
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
        let (layout, penalty) =
            build_survival_layout(&data, &basis, 0.1, 2, 1.0, 10, None).unwrap();
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
        let (layout, penalty) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.5, 10, None).unwrap();
        let model = WorkingModelSurvival::new(
            layout.clone(),
            &data,
            penalty.clone(),
            SurvivalSpec::default(),
        )
        .unwrap();
        let static_names: Vec<String> = (0..layout.static_covariates.ncols())
            .map(|idx| format!("cov{idx}"))
            .collect();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(model.layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            time_varying_centering: None,
            static_covariate_layout: CovariateLayout {
                column_names: static_names,
            },
            penalties: PenaltyDescriptor {
                baseline_order: 2,
                baseline_lambda: 0.5,
                interaction: None,
            },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
            pgs_column_index: None,
        };
        let covs = Array1::<f64>::zeros(model.layout.static_covariates.ncols());
        let cif0 = cumulative_incidence(55.0, &covs, &artifacts).unwrap();
        let cif1 = cumulative_incidence(60.0, &covs, &artifacts).unwrap();
        assert!(cif1 >= cif0 - 1e-9);
        let risk = conditional_absolute_risk(55.0, 60.0, &covs, 0.0, &artifacts).unwrap();
        assert!(risk >= -1e-9);
    }

    #[test]
    fn pgs_age_interaction_layout_builds_tensor_product() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let config = TimeVaryingConfig {
            lambda_age: 0.7,
            lambda_pgs: 0.2,
        };
        let (layout, _) =
            build_survival_layout(&data, &basis, 0.1, 2, 1.0, 6, Some(config)).unwrap();

        let entry = layout.time_varying_entry.as_ref().expect("entry design");
        let exit = layout.time_varying_exit.as_ref().expect("exit design");
        let derivative = layout
            .time_varying_derivative_exit
            .as_ref()
            .expect("derivative design");
        assert_eq!(entry.ncols(), layout.baseline_entry.ncols());
        assert_eq!(exit.ncols(), layout.baseline_exit.ncols());
        assert_eq!(derivative.ncols(), layout.baseline_derivative_exit.ncols());

        let centering = layout
            .time_varying_centering
            .as_ref()
            .expect("centering vector");
        assert_eq!(centering.len(), entry.ncols());
        let mut restored_exit = exit.clone();
        for (mut column, &mean) in restored_exit.columns_mut().into_iter().zip(centering.iter()) {
            column.mapv_inplace(|v| v + mean);
        }
        let weighted_means = weighted_column_means(&restored_exit, &data.sample_weight);
        for (mean, stored) in weighted_means.iter().zip(centering.iter()) {
            assert_abs_diff_eq!(*mean, *stored, epsilon = 1e-10);
        }
        let centered_means = weighted_column_means(exit, &data.sample_weight);
        for mean in centered_means.iter() {
            assert!(mean.abs() < 1e-12, "interaction column not centered");
        }

        for (idx, (&pgs_value, base_row)) in data
            .pgs
            .iter()
            .zip(layout.baseline_derivative_exit.rows())
            .enumerate()
        {
            let interaction_row = derivative.row(idx);
            for (d_int, d_base) in interaction_row.iter().zip(base_row.iter()) {
                assert_abs_diff_eq!(*d_int, pgs_value * d_base, epsilon = 1e-12);
            }
        }

        assert_eq!(layout.penalties.blocks.len(), 3);
        let baseline_cols = layout.baseline_exit.ncols();
        let interaction_cols = exit.ncols();
        let range = baseline_cols..baseline_cols + interaction_cols;
        assert_eq!(layout.penalties.blocks[1].range, range.clone());
        assert_eq!(layout.penalties.blocks[2].range, range);
        assert_abs_diff_eq!(layout.penalties.blocks[1].lambda, config.lambda_age, epsilon = 1e-12);
        assert_abs_diff_eq!(layout.penalties.blocks[2].lambda, config.lambda_pgs, epsilon = 1e-12);
    }

    #[test]
    fn working_state_shapes() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.5, 8, None).unwrap();
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
        let (layout, monotonicity) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.0, 0, None).unwrap();
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
        let (layout, monotonicity) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.0, 4, None).unwrap();
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

        let static_names: Vec<String> = (0..layout.static_covariates.ncols())
            .map(|idx| format!("cov{idx}"))
            .collect();
        let artifacts = SurvivalModelArtifacts {
            coefficients: beta.clone(),
            age_basis: basis.clone(),
            time_varying_basis: None,
            time_varying_centering: None,
            static_covariate_layout: CovariateLayout {
                column_names: static_names,
            },
            penalties: PenaltyDescriptor {
                baseline_order: 2,
                baseline_lambda: 0.0,
                interaction: None,
            },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
            pgs_column_index: None,
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
        let (layout, monotonicity) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.0, 0, None).unwrap();
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
            let event_scale = if spec.use_expected_information {
                weight * h_exit[i]
            } else {
                weight * d
            };
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
        let (layout, monotonicity) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.2, 6, None).unwrap();
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
        let (layout, monotonicity) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.0, 0, None).unwrap();
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

        let guard = spec_observed.derivative_guard.max(f64::EPSILON);
        let eta_exit = layout.combined_exit.dot(&beta);
        let derivative_exit = layout.combined_derivative_exit.dot(&beta);
        let mut expected_diff = Array2::<f64>::zeros((p, p));
        for i in 0..data.age_entry.len() {
            let weight = data.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let h_exit = eta_exit[i].exp();
            let d = f64::from(data.event_target[i]);
            let raw_derivative = derivative_exit[i];
            let guard_applied = raw_derivative <= guard;
            let guarded = if guard_applied { guard } else { raw_derivative };
            let scale = 1.0 / guarded;
            let mut x_tilde = layout.combined_exit.row(i).to_owned();
            Zip::from(&mut x_tilde)
                .and(&layout.combined_derivative_exit.row(i))
                .for_each(|value, &deriv| *value += deriv * scale);
            let diff_scale = -2.0 * weight * (h_exit - d);
            if diff_scale == 0.0 {
                continue;
            }
            for j in 0..p {
                for k in 0..p {
                    expected_diff[[j, k]] += diff_scale * x_tilde[j] * x_tilde[k];
                }
            }
        }

        let diff = &expected_state.hessian - &observed_state.hessian;
        for (observed, expected) in diff.iter().zip(expected_diff.iter()) {
            let tolerance = 1e-12 * expected.abs().max(1.0);
            assert!(
                (*observed - *expected).abs() < tolerance,
                "expected-information difference mismatch: observed={}, expected={}, tolerance={}",
                observed,
                expected,
                tolerance
            );
        }
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

        let weighted_transform =
            AgeTransform::from_training(&weighted_data.age_entry, 0.1).unwrap();
        let (layout_weighted, monotonic_weighted) =
            build_survival_layout(&weighted_data, &basis, 0.1, 2, 0.0, 0, None).unwrap();
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
            time_varying_centering: layout_weighted.time_varying_centering.clone(),
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
        let (layout, monotonicity) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.6, 0, None).unwrap();
        let penalised_layout = layout.clone();
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
        let (layout, penalty) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.75, 0, None).unwrap();
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
        let (layout, penalty) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.5, 6, None).unwrap();
        let static_names: Vec<String> = (0..layout.static_covariates.ncols())
            .map(|idx| format!("cov{idx}"))
            .collect();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            time_varying_centering: None,
            static_covariate_layout: CovariateLayout {
                column_names: static_names,
            },
            penalties: PenaltyDescriptor {
                baseline_order: 2,
                baseline_lambda: 0.5,
                interaction: None,
            },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
            pgs_column_index: None,
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
    fn cumulative_hazard_supports_pgs_age_interaction() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let config = TimeVaryingConfig {
            lambda_age: 0.4,
            lambda_pgs: 0.1,
        };
        let (layout, _) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.5, 4, Some(config)).unwrap();
        let static_names = vec![
            "pgs".to_string(),
            "sex".to_string(),
            "pc0".to_string(),
            "pc1".to_string(),
        ];
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: Some(basis.clone()),
            time_varying_centering: layout.time_varying_centering.clone(),
            static_covariate_layout: CovariateLayout {
                column_names: static_names,
            },
            penalties: PenaltyDescriptor {
                baseline_order: 2,
                baseline_lambda: 0.5,
                interaction: Some(InteractionPenaltyDescriptor {
                    lambda_age: config.lambda_age,
                    lambda_pgs: config.lambda_pgs,
                }),
            },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
            pgs_column_index: Some(0),
        };

        let covs = layout.static_covariates.row(0).to_owned();
        let hazard = cumulative_hazard(data.age_exit[0], &covs, &artifacts).unwrap();
        assert!(hazard.is_finite());
    }

    #[test]
    fn cumulative_hazard_rejects_covariate_mismatch() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, _) =
            build_survival_layout(&data, &basis, 0.1, 2, 0.5, 4, None).unwrap();
        let static_names: Vec<String> = (0..layout.static_covariates.ncols())
            .map(|idx| format!("cov{idx}"))
            .collect();
        let artifacts = SurvivalModelArtifacts {
            coefficients: Array1::<f64>::zeros(layout.combined_exit.ncols()),
            age_basis: basis.clone(),
            time_varying_basis: None,
            time_varying_centering: None,
            static_covariate_layout: CovariateLayout {
                column_names: static_names.clone(),
            },
            penalties: PenaltyDescriptor {
                baseline_order: 2,
                baseline_lambda: 0.5,
                interaction: None,
            },
            age_transform: layout.age_transform,
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
            pgs_column_index: None,
        };
        let mismatched_covs = Array1::<f64>::zeros(static_names.len() + 1);
        let err = cumulative_hazard(60.0, &mismatched_covs, &artifacts).unwrap_err();
        assert!(matches!(err, SurvivalError::CovariateDimensionMismatch));
    }

    #[test]
    fn hessian_factor_reports_inertia() {
        let observed = HessianFactor::Observed {
            ldlt_factor: Array2::zeros((2, 2)),
            permutation: vec![0, 1],
            inertia: (1, 0, 1),
        };
        assert_eq!(observed.inertia(), (1, 0, 1));

        let expected = HessianFactor::Expected {
            cholesky_factor: Array2::eye(3),
        };
        assert_eq!(expected.inertia(), (0, 0, 3));
    }
}
