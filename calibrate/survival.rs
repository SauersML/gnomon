use crate::calibrate::basis::{create_bspline_basis_with_knots, create_difference_penalty_matrix};
use crate::calibrate::faer_ndarray::FaerSvd;
use ndarray::prelude::*;
use ndarray::{Array1, Array2, concatenate};
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Working model abstraction shared between GAM and survival implementations.
pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
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
    pub fn new(age_entry: &Array1<f64>, delta: f64) -> Self {
        let mut min_age = f64::INFINITY;
        for &value in age_entry.iter() {
            if value < min_age {
                min_age = value;
            }
        }
        Self {
            minimum_age: min_age,
            delta,
        }
    }

    #[inline]
    pub fn transform(&self, age: f64) -> f64 {
        (age - self.minimum_age + self.delta).ln()
    }

    #[inline]
    pub fn derivative_factor(&self, age: f64) -> f64 {
        1.0 / (age - self.minimum_age + self.delta)
    }

    pub fn transform_array(&self, ages: &Array1<f64>) -> Array1<f64> {
        ages.mapv(|age| self.transform(age))
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

    pub fn add_contributions(
        &self,
        beta: &Array1<f64>,
        gradient: &mut Array1<f64>,
        hessian: &mut Array2<f64>,
        deviance: &mut f64,
    ) {
        for block in &self.blocks {
            let view = beta.slice(s![block.range.clone()]);
            let s_beta = block.matrix.dot(&view);
            let penalty_value = view.dot(&s_beta);
            let mut grad_slice = gradient.slice_mut(s![block.range.clone()]);
            grad_slice.scaled_add(-2.0 * block.lambda, &s_beta);

            let mut block_hessian = block.matrix.clone();
            block_hessian *= -2.0 * block.lambda;
            let rows = block.range.clone();
            for (local_i, row_idx) in rows.clone().enumerate() {
                for (local_j, col_idx) in rows.clone().enumerate() {
                    hessian[[row_idx, col_idx]] += block_hessian[[local_i, local_j]];
                }
            }

            *deviance -= block.lambda * penalty_value;
        }
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
    pub fn validate(&self) {
        assert_eq!(self.age_entry.len(), self.age_exit.len());
        let n = self.age_entry.len();
        assert_eq!(self.event_target.len(), n);
        assert_eq!(self.event_competing.len(), n);
        assert_eq!(self.sample_weight.len(), n);
        assert_eq!(self.pgs.len(), n);
        assert_eq!(self.sex.len(), n);
        assert_eq!(self.pcs.nrows(), n);
        for i in 0..n {
            assert!(
                self.age_entry[i] < self.age_exit[i],
                "age_entry must be less than age_exit"
            );
            let target = self.event_target[i];
            let competing = self.event_competing[i];
            assert!((target == 0 || target == 1) && (competing == 0 || competing == 1));
            assert!(
                target == 0 || competing == 0,
                "events must be mutually exclusive"
            );
        }
    }
}

/// Guard that constrains the baseline spline at the chosen reference point.
fn make_reference_constraint(
    knot_vector: ArrayView1<f64>,
    degree: usize,
    reference_u: f64,
) -> ReferenceConstraint {
    let data = array![reference_u];
    let (basis_arc, _) = create_bspline_basis_with_knots(data.view(), knot_vector, degree)
        .expect("reference basis evaluation");
    let basis = (*basis_arc).clone();
    let row = basis.row(0).to_owned();
    let transform = nullspace_transform(&row);
    ReferenceConstraint {
        transform,
        reference_log_age: reference_u,
    }
}

/// Build a nullspace transform for a single-row constraint.
fn nullspace_transform(constraint_row: &Array1<f64>) -> Array2<f64> {
    let k = constraint_row.len();
    let mut row_mat = Array2::<f64>::zeros((k, 1));
    row_mat.column_mut(0).assign(constraint_row);
    let (u_opt, ..) = row_mat
        .svd(true, false)
        .expect("SVD should succeed for constraint row");
    let u = u_opt.expect("U matrix available");
    u.slice(s![.., 1..]).to_owned()
}

/// Evaluate a basis and its derivative with respect to the guarded log-age.
fn evaluate_basis_and_derivative(
    log_ages: ArrayView1<f64>,
    descriptor: &BasisDescriptor,
) -> (Array2<f64>, Array2<f64>) {
    let (basis_arc, _) =
        create_bspline_basis_with_knots(log_ages, descriptor.knot_vector.view(), descriptor.degree)
            .expect("basis evaluation");
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
    )
    .expect("basis+ evaluation");
    let (basis_minus_arc, _) = create_bspline_basis_with_knots(
        perturbed_minus.view(),
        descriptor.knot_vector.view(),
        descriptor.degree,
    )
    .expect("basis- evaluation");
    let basis_plus = (*basis_plus_arc).clone();
    let basis_minus = (*basis_minus_arc).clone();
    let mut derivative = basis_plus;
    derivative -= &basis_minus;
    derivative.mapv_inplace(|v| v / (2.0 * eps));

    (basis, derivative)
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
) -> (SurvivalLayout, MonotonicityPenalty) {
    data.validate();
    let n = data.age_entry.len();
    let age_transform = AgeTransform::new(&data.age_entry, delta);
    let log_entry = age_transform.transform_array(&data.age_entry);
    let log_exit = age_transform.transform_array(&data.age_exit);

    let reference_u = log_exit.mean().unwrap_or(0.0);
    let reference_constraint =
        make_reference_constraint(age_basis.knot_vector.view(), age_basis.degree, reference_u);

    let (baseline_entry_raw, _) = evaluate_basis_and_derivative(log_entry.view(), age_basis);
    let (baseline_exit_raw, baseline_exit_deriv_u) =
        evaluate_basis_and_derivative(log_exit.view(), age_basis);

    let constrained_entry = reference_constraint.apply(&baseline_entry_raw);
    let constrained_exit = reference_constraint.apply(&baseline_exit_raw);
    let constrained_derivative_exit_u = reference_constraint.apply(&baseline_exit_deriv_u);

    let mut baseline_derivative_exit = constrained_derivative_exit_u;
    for (mut row, age) in baseline_derivative_exit
        .rows_mut()
        .into_iter()
        .zip(data.age_exit.iter().copied())
    {
        let factor = age_transform.derivative_factor(age);
        row.mapv_inplace(|v| v * factor);
    }

    let static_covariates = assemble_static_covariates(data);

    let combined_entry = concatenate_design(&constrained_entry, None, &static_covariates);
    let combined_exit = concatenate_design(&constrained_exit, None, &static_covariates);
    let zero_static = Array2::<f64>::zeros((n, static_covariates.ncols()));
    let combined_derivative_exit =
        concatenate_design(&baseline_derivative_exit, None, &zero_static);

    let penalty_matrix =
        create_difference_penalty_matrix(constrained_exit.ncols(), baseline_penalty_order)
            .expect("difference penalty");
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
    );

    (layout, monotonicity)
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

fn build_monotonicity_penalty(
    layout: &SurvivalLayout,
    age_basis: &BasisDescriptor,
    ages_exit: &Array1<f64>,
    grid_size: usize,
    lambda: f64,
) -> MonotonicityPenalty {
    if grid_size == 0 {
        return MonotonicityPenalty {
            lambda,
            derivative_design: Array2::<f64>::zeros((0, layout.combined_exit.ncols())),
        };
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
        return MonotonicityPenalty {
            lambda,
            derivative_design: Array2::<f64>::zeros((0, layout.combined_exit.ncols())),
        };
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

    let log_grid = grid.mapv(|age| layout.age_transform.transform(age));
    let (_, derivative_u) = evaluate_basis_and_derivative(log_grid.view(), age_basis);
    let constrained_derivative_u = layout.reference_constraint.apply(&derivative_u);
    let mut derivative_age = constrained_derivative_u;
    for (mut row, &age) in derivative_age.rows_mut().into_iter().zip(grid.iter()) {
        let factor = layout.age_transform.derivative_factor(age);
        row *= factor;
    }

    let mut combined = Array2::<f64>::zeros((grid_size, layout.combined_exit.ncols()));
    let baseline_cols = layout.baseline_exit.ncols();
    combined
        .slice_mut(s![.., ..baseline_cols])
        .assign(&derivative_age);
    MonotonicityPenalty {
        lambda,
        derivative_design: combined,
    }
}

/// Royston–Parmar working model implementation.
pub struct WorkingModelSurvival {
    pub layout: SurvivalLayout,
    pub sample_weight: Array1<f64>,
    pub event_target: Array1<u8>,
    pub monotonicity: MonotonicityPenalty,
}

impl WorkingModelSurvival {
    pub fn new(
        layout: SurvivalLayout,
        data: &SurvivalTrainingData,
        monotonicity: MonotonicityPenalty,
    ) -> Self {
        Self {
            layout,
            sample_weight: data.sample_weight.clone(),
            event_target: data.event_target.clone(),
            monotonicity,
        }
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
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

        let epsilon = 1e-10;
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
            let guarded_derivative = if d_eta_exit > epsilon {
                d_eta_exit
            } else {
                epsilon
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
        }

        self.layout
            .penalties
            .add_contributions(beta, &mut gradient, &mut hessian, &mut deviance);

        if self.monotonicity.lambda > 0.0 && self.monotonicity.derivative_design.nrows() > 0 {
            apply_monotonicity_penalty(
                &self.monotonicity,
                beta,
                &mut gradient,
                &mut hessian,
                &mut deviance,
            );
        }

        WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        }
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
) -> f64 {
    let log_age = artifacts.age_transform.transform(age);
    let (basis_arc, _) = create_bspline_basis_with_knots(
        array![log_age].view(),
        artifacts.age_basis.knot_vector.view(),
        artifacts.age_basis.degree,
    )
    .expect("prediction basis");
    let basis = (*basis_arc).clone();
    let constrained = artifacts.reference_constraint.apply(&basis);

    let mut design = constrained.row(0).to_owned();
    if let Some(time_basis) = &artifacts.time_varying_basis {
        let (tv_arc, _) = create_bspline_basis_with_knots(
            array![log_age].view(),
            time_basis.knot_vector.view(),
            time_basis.degree,
        )
        .expect("time basis");
        let tv = artifacts.reference_constraint.apply(&(*tv_arc).clone());
        design = concatenate(Axis(0), &[design.view(), tv.row(0)]).expect("time concat");
    }
    design = concatenate(Axis(0), &[design.view(), covariates.view()]).expect("cov concat");
    let eta = design.dot(&artifacts.coefficients);
    eta.exp()
}

pub fn cumulative_incidence(
    age: f64,
    covariates: &Array1<f64>,
    artifacts: &SurvivalModelArtifacts,
) -> f64 {
    let h = cumulative_hazard(age, covariates, artifacts);
    1.0 - (-h).exp()
}

pub fn conditional_absolute_risk(
    t0: f64,
    t1: f64,
    covariates: &Array1<f64>,
    cif_competing_t0: f64,
    artifacts: &SurvivalModelArtifacts,
) -> f64 {
    let cif0 = cumulative_incidence(t0, covariates, artifacts);
    let cif1 = cumulative_incidence(t1, covariates, artifacts);
    let delta = (cif1 - cif0).max(0.0);
    let denom = (1.0 - cif0 - cif_competing_t0).max(1e-12);
    delta / denom
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
        let mut model = WorkingModelSurvival::new(layout.clone(), data, penalty.clone());
        model.update(beta)
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
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 1.0, 10);
        let mut model = WorkingModelSurvival::new(layout, &data, penalty);
        let beta = Array1::<f64>::zeros(model.layout.combined_exit.ncols());
        let state = model.update(&beta);
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn conditional_risk_monotone() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 10);
        let model = WorkingModelSurvival::new(layout.clone(), &data, penalty.clone());
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
        let cif0 = cumulative_incidence(55.0, &covs, &artifacts);
        let cif1 = cumulative_incidence(60.0, &covs, &artifacts);
        assert!(cif1 >= cif0 - 1e-9);
        let risk = conditional_absolute_risk(55.0, 60.0, &covs, 0.0, &artifacts);
        assert!(risk >= -1e-9);
    }

    #[test]
    fn working_state_shapes() {
        let data = toy_training_data();
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.33, 0.66, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.5, 8);
        let mut model = WorkingModelSurvival::new(layout, &data, penalty);
        let beta = Array1::<f64>::zeros(model.layout.combined_exit.ncols());
        let state = model.update(&beta);
        assert_eq!(state.gradient.len(), beta.len());
        assert_eq!(state.hessian.nrows(), beta.len());
        assert_eq!(state.hessian.ncols(), beta.len());
    }

    #[test]
    fn smoothing_penalty_matches_finite_difference() {
        let mut data = toy_training_data();
        data.sample_weight.fill(0.0);
        let basis = BasisDescriptor {
            knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            degree: 2,
        };
        let (layout, penalty) = build_survival_layout(&data, &basis, 0.1, 2, 0.75, 0);
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
}
