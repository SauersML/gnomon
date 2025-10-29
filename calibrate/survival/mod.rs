use ndarray::{Array1, Array2, ArrayView1, s};
use serde::{Deserialize, Serialize};
use thiserror::Error;

const DEFAULT_DERIVATIVE_GUARD: f64 = 1e-8;
const DEFAULT_SOFTPLUS_SCALE: f64 = 1.0;
const DEFAULT_SOFTPLUS_TEMPERATURE: f64 = 0.1;
const CONDITIONAL_DENOMINATOR_EPS: f64 = 1e-12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeTransform {
    pub a_min: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn fit(data: ArrayView1<'_, f64>, delta: f64) -> Result<Self, SurvivalModelError> {
        if data.is_empty() {
            return Err(SurvivalModelError::EmptyColumn("age".to_string()));
        }
        if !delta.is_finite() || delta <= 0.0 {
            return Err(SurvivalModelError::InvalidDelta(delta));
        }
        let mut min_val = f64::INFINITY;
        for &value in data.iter() {
            if !value.is_finite() {
                return Err(SurvivalModelError::NonFiniteValue {
                    column: "age".to_string(),
                    value,
                });
            }
            if value < min_val {
                min_val = value;
            }
        }
        Ok(Self {
            a_min: min_val,
            delta,
        })
    }

    #[inline]
    pub fn transform(&self, age: f64) -> f64 {
        (age - self.a_min + self.delta).ln()
    }

    #[inline]
    pub fn derivative_factor(&self, age: f64) -> f64 {
        1.0 / (age - self.a_min + self.delta)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceConstraint {
    pivot_index: usize,
    transform: Array2<f64>,
}

impl ReferenceConstraint {
    pub fn new(basis_at_reference: ArrayView1<'_, f64>) -> Result<Self, SurvivalModelError> {
        let m = basis_at_reference.len();
        if m < 2 {
            return Err(SurvivalModelError::InsufficientBasisColumns(m));
        }
        let mut pivot_index = 0usize;
        let mut pivot_value = 0.0f64;
        for (idx, &value) in basis_at_reference.iter().enumerate() {
            if value.abs() > pivot_value {
                pivot_value = value.abs();
                pivot_index = idx;
            }
        }
        if pivot_value == 0.0 {
            return Err(SurvivalModelError::DegenerateReferenceConstraint);
        }
        let reduced_cols = m - 1;
        let mut transform = Array2::<f64>::zeros((m, reduced_cols));
        let mut col = 0usize;
        for j in 0..m {
            if j == pivot_index {
                continue;
            }
            transform[[j, col]] = 1.0;
            let ratio = -basis_at_reference[j] / basis_at_reference[pivot_index];
            transform[[pivot_index, col]] = ratio;
            col += 1;
        }
        Ok(Self {
            pivot_index,
            transform,
        })
    }

    #[inline]
    pub fn apply(&self, row: ArrayView1<'_, f64>) -> Array1<f64> {
        row.dot(&self.transform)
    }

    #[inline]
    pub fn transform_matrix(&self) -> &Array2<f64> {
        &self.transform
    }

    #[inline]
    pub fn pivot_index(&self) -> usize {
        self.pivot_index
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyBlock {
    pub matrix: Array2<f64>,
    pub lambda: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisDescriptor {
    pub knots: Array1<f64>,
    pub degree: usize,
}

impl BasisDescriptor {
    pub fn new(knots: Array1<f64>, degree: usize) -> Result<Self, SurvivalModelError> {
        if degree == 0 {
            return Err(SurvivalModelError::InvalidBasisDegree);
        }
        if knots.len() < degree + 2 {
            return Err(SurvivalModelError::InsufficientKnots {
                degree,
                provided: knots.len(),
            });
        }
        for window in knots.windows(2) {
            if window[0] > window[1] {
                return Err(SurvivalModelError::InvalidKnotVector);
            }
        }
        Ok(Self { knots, degree })
    }

    #[inline]
    pub fn num_basis(&self) -> usize {
        self.knots.len() - self.degree - 1
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub static_covariates: Array2<f64>,
}

impl SurvivalTrainingData {
    pub fn validate(&self) -> Result<(), SurvivalModelError> {
        let n = self.age_entry.len();
        if self.age_exit.len() != n
            || self.event_target.len() != n
            || self.event_competing.len() != n
            || self.sample_weight.len() != n
            || self.static_covariates.nrows() != n
        {
            return Err(SurvivalModelError::DimensionMismatch);
        }
        for i in 0..n {
            let entry = self.age_entry[i];
            let exit = self.age_exit[i];
            if !entry.is_finite() || !exit.is_finite() {
                return Err(SurvivalModelError::NonFiniteValue {
                    column: "age".to_string(),
                    value: if entry.is_finite() { exit } else { entry },
                });
            }
            if !(exit > entry) {
                return Err(SurvivalModelError::InvalidAgeInterval { entry, exit });
            }
            let target = self.event_target[i];
            let competing = self.event_competing[i];
            if target > 1 || competing > 1 {
                return Err(SurvivalModelError::InvalidEventFlag { target, competing });
            }
            if target == 1 && competing == 1 {
                return Err(SurvivalModelError::MutuallyExclusiveEvents);
            }
            let weight = self.sample_weight[i];
            if weight <= 0.0 || !weight.is_finite() {
                return Err(SurvivalModelError::InvalidWeight(weight));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SurvivalLayout {
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    derivative_exit: Array2<f64>,
    age_transform: AgeTransform,
    reference_constraint: ReferenceConstraint,
    penalties: PenaltyBlocks,
    derivative_guard: f64,
    softplus_scale: f64,
    softplus_temperature: f64,
}

impl SurvivalLayout {
    pub fn from_training_data(
        data: &SurvivalTrainingData,
        basis: &BasisDescriptor,
        penalty: PenaltyBlocks,
        delta: f64,
    ) -> Result<Self, SurvivalModelError> {
        data.validate()?;
        let age_transform = AgeTransform::fit(data.age_entry.view(), delta)?;
        let u_entry = data
            .age_entry
            .iter()
            .map(|&age| age_transform.transform(age))
            .collect::<Array1<f64>>();
        let u_exit = data
            .age_exit
            .iter()
            .map(|&age| age_transform.transform(age))
            .collect::<Array1<f64>>();

        let reference_age =
            data.age_exit.iter().copied().sum::<f64>() / (data.age_exit.len() as f64);
        let reference_u = age_transform.transform(reference_age);

        let (baseline_entry, baseline_exit, baseline_derivative, reference_constraint) =
            build_baseline_matrices(u_entry.view(), u_exit.view(), basis, reference_u)?;

        let n = data.age_entry.len();
        let baseline_cols = baseline_entry.ncols();
        let static_cols = data.static_covariates.ncols();
        let total_cols = baseline_cols + static_cols;

        let mut design_entry = Array2::<f64>::zeros((n, total_cols));
        let mut design_exit = Array2::<f64>::zeros((n, total_cols));
        let mut derivative_exit = Array2::<f64>::zeros((n, total_cols));

        design_entry
            .slice_mut(s![.., 0..baseline_cols])
            .assign(&baseline_entry);
        design_exit
            .slice_mut(s![.., 0..baseline_cols])
            .assign(&baseline_exit);
        derivative_exit
            .slice_mut(s![.., 0..baseline_cols])
            .assign(&baseline_derivative);

        if static_cols > 0 {
            design_entry
                .slice_mut(s![.., baseline_cols..])
                .assign(&data.static_covariates);
            design_exit
                .slice_mut(s![.., baseline_cols..])
                .assign(&data.static_covariates);
        }

        // Apply chain rule to derivative rows.
        for (idx, &age) in data.age_exit.iter().enumerate() {
            let factor = age_transform.derivative_factor(age);
            derivative_exit
                .row_mut(idx)
                .mapv_inplace(|value| value * factor);
        }

        Ok(Self {
            design_entry,
            design_exit,
            derivative_exit,
            age_transform,
            reference_constraint,
            penalties: penalty,
            derivative_guard: DEFAULT_DERIVATIVE_GUARD,
            softplus_scale: DEFAULT_SOFTPLUS_SCALE,
            softplus_temperature: DEFAULT_SOFTPLUS_TEMPERATURE,
        })
    }

    #[inline]
    pub fn age_transform(&self) -> &AgeTransform {
        &self.age_transform
    }

    #[inline]
    pub fn reference_constraint(&self) -> &ReferenceConstraint {
        &self.reference_constraint
    }

    #[inline]
    pub fn penalties(&self) -> &PenaltyBlocks {
        &self.penalties
    }

    #[inline]
    pub fn num_coefficients(&self) -> usize {
        self.design_exit.ncols()
    }

    #[inline]
    pub fn design_entry(&self) -> &Array2<f64> {
        &self.design_entry
    }

    #[inline]
    pub fn design_exit(&self) -> &Array2<f64> {
        &self.design_exit
    }

    #[inline]
    pub fn derivative_exit(&self) -> &Array2<f64> {
        &self.derivative_exit
    }

    #[inline]
    pub fn derivative_guard(&self) -> f64 {
        self.derivative_guard
    }

    #[inline]
    pub fn softplus_scale(&self) -> f64 {
        self.softplus_scale
    }

    #[inline]
    pub fn softplus_temperature(&self) -> f64 {
        self.softplus_temperature
    }
}

pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
}

#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
    pub negative_derivative_fraction: f64,
}

pub struct WorkingModelSurvival<'a> {
    layout: &'a SurvivalLayout,
    data: &'a SurvivalTrainingData,
    negative_derivative_fraction: f64,
}

impl<'a> WorkingModelSurvival<'a> {
    pub fn new(layout: &'a SurvivalLayout, data: &'a SurvivalTrainingData) -> Self {
        Self {
            layout,
            data,
            negative_derivative_fraction: 0.0,
        }
    }
}

impl<'a> WorkingModel for WorkingModelSurvival<'a> {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
        let design_entry = self.layout.design_entry();
        let design_exit = self.layout.design_exit();
        let derivative_exit = self.layout.derivative_exit();
        let n = design_exit.nrows();
        let p = design_exit.ncols();

        debug_assert_eq!(beta.len(), p);

        let eta_exit = design_exit.dot(beta);
        let eta_entry = design_entry.dot(beta);
        let d_eta_exit = derivative_exit.dot(beta);

        let mut gradient = Array1::<f64>::zeros(p);
        let mut hessian = Array2::<f64>::zeros((p, p));
        let mut deviance = 0.0;
        let mut negative_count = 0usize;

        let guard = self.layout.derivative_guard();
        let softplus_scale = self.layout.softplus_scale();
        let softplus_temp = self.layout.softplus_temperature();

        for i in 0..n {
            let weight = self.data.sample_weight[i];
            let event_target = self.data.event_target[i] as f64;
            let event_competing = self.data.event_competing[i] as f64;
            debug_assert!(event_target == 0.0 || event_competing == 0.0);

            let eta_exit_i = eta_exit[i];
            let eta_entry_i = eta_entry[i];
            let h_exit = eta_exit_i.exp();
            let h_entry = eta_entry_i.exp();
            let delta_h = h_exit - h_entry;

            let d_eta = d_eta_exit[i];
            if d_eta < 0.0 {
                negative_count += 1;
            }
            let guard_val = if d_eta >= guard { d_eta } else { guard };
            let log_guard = guard_val.ln();

            let barrier_z = -d_eta / softplus_temp;
            let barrier = softplus_scale * softplus(barrier_z);
            let sigmoid_barrier = sigmoid(barrier_z);
            let barrier_grad_coeff = -(softplus_scale / softplus_temp) * sigmoid_barrier;
            let barrier_hess_coeff = softplus_scale / (softplus_temp * softplus_temp)
                * sigmoid_barrier
                * (1.0 - sigmoid_barrier);

            let mut subject_gradient = Array1::<f64>::zeros(p);
            let mut subject_hessian = Array2::<f64>::zeros((p, p));

            let x_exit = design_exit.row(i);
            let x_entry = design_entry.row(i);
            let x_deriv = derivative_exit.row(i);

            for j in 0..p {
                let exit_val = x_exit[j];
                let entry_val = x_entry[j];
                let deriv_val = x_deriv[j];
                subject_gradient[j] += h_exit * exit_val;
                subject_gradient[j] -= h_entry * entry_val;
                subject_gradient[j] -= event_target * exit_val;
                if d_eta >= guard && event_target > 0.0 {
                    subject_gradient[j] -= (event_target / d_eta) * deriv_val;
                }
                subject_gradient[j] += barrier_grad_coeff * deriv_val;
            }

            add_outer_scaled(&mut subject_hessian, x_exit, h_exit);
            add_outer_scaled(&mut subject_hessian, x_entry, -h_entry);
            if d_eta >= guard && event_target > 0.0 {
                let coeff = event_target / (d_eta * d_eta);
                add_outer_scaled(&mut subject_hessian, x_deriv, coeff);
            }
            add_outer_scaled(&mut subject_hessian, x_deriv, barrier_hess_coeff);

            for j in 0..p {
                gradient[j] += weight * subject_gradient[j];
            }
            for row in 0..p {
                for col in 0..p {
                    hessian[[row, col]] += weight * subject_hessian[[row, col]];
                }
            }

            let mut neg_log_lik = delta_h;
            if event_target > 0.0 {
                neg_log_lik -= event_target * (eta_exit_i + log_guard);
            }
            neg_log_lik += barrier;
            deviance += 2.0 * weight * neg_log_lik;
        }

        self.negative_derivative_fraction = if n > 0 {
            negative_count as f64 / n as f64
        } else {
            0.0
        };

        WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
            negative_derivative_fraction: self.negative_derivative_fraction,
        }
    }
}

fn add_outer_scaled(matrix: &mut Array2<f64>, row: ArrayView1<'_, f64>, scale: f64) {
    let p = row.len();
    for i in 0..p {
        let row_i = row[i];
        for j in 0..p {
            matrix[[i, j]] += scale * row_i * row[j];
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CovariateLayout {
    pub names: Vec<String>,
}

impl CovariateLayout {
    pub fn new(names: Vec<String>) -> Self {
        Self { names }
    }

    #[inline]
    pub fn num_static(&self) -> usize {
        self.names.len()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct LdltFactor {
    pub matrix: Array2<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PermutationMatrix {
    pub indices: Vec<usize>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Inertia {
    pub negative: usize,
    pub zero: usize,
    pub positive: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CholeskyFactor {
    pub matrix: Array2<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
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

#[derive(Clone, Serialize, Deserialize)]
pub struct PenaltyDescriptor {
    pub penalties: PenaltyBlocks,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: PenaltyDescriptor,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<HessianFactor>,
}

impl SurvivalModelArtifacts {
    fn baseline_columns(&self) -> usize {
        self.reference_constraint.transform_matrix().ncols()
    }

    pub fn cumulative_hazard(
        &self,
        age: f64,
        covariates: &CovariateValues,
    ) -> Result<f64, SurvivalModelError> {
        let u = self.age_transform.transform(age);
        let basis = evaluate_basis(u, &self.age_basis);
        let constrained = self.reference_constraint.apply(basis.view());
        let baseline_cols = self.baseline_columns();
        if covariates.static_covariates.len() != self.static_covariate_layout.num_static() {
            return Err(SurvivalModelError::InvalidCovariateDimension {
                expected: self.static_covariate_layout.num_static(),
                found: covariates.static_covariates.len(),
            });
        }
        let mut eta = 0.0;
        for (value, coeff) in constrained.iter().zip(self.coefficients.iter()) {
            eta += value * coeff;
        }
        for (value, coeff) in covariates
            .static_covariates
            .iter()
            .zip(self.coefficients.iter().skip(baseline_cols))
        {
            eta += value * coeff;
        }
        Ok(eta.exp())
    }

    pub fn cumulative_incidence(
        &self,
        age: f64,
        covariates: &CovariateValues,
    ) -> Result<f64, SurvivalModelError> {
        let hazard = self.cumulative_hazard(age, covariates)?;
        Ok(1.0 - (-hazard).exp())
    }

    pub fn conditional_absolute_risk(
        &self,
        t0: f64,
        t1: f64,
        covariates: &CovariateValues,
        cif_competing_t0: f64,
    ) -> Result<f64, SurvivalModelError> {
        if !(t1 > t0) {
            return Err(SurvivalModelError::InvalidAgeInterval {
                entry: t0,
                exit: t1,
            });
        }
        if cif_competing_t0 < 0.0 || cif_competing_t0 > 1.0 {
            return Err(SurvivalModelError::InvalidCompetingCif(cif_competing_t0));
        }
        let h0 = self.cumulative_hazard(t0, covariates)?;
        let h1 = self.cumulative_hazard(t1, covariates)?;
        let cif0 = 1.0 - (-h0).exp();
        let cif1 = 1.0 - (-h1).exp();
        let delta = (cif1 - cif0).max(0.0);
        let denom = (1.0 - cif0 - cif_competing_t0).max(CONDITIONAL_DENOMINATOR_EPS);
        Ok((delta / denom).min(1.0).max(0.0))
    }
}

#[derive(Debug, Clone)]
pub struct CovariateValues {
    pub static_covariates: Vec<f64>,
}

impl CovariateValues {
    pub fn new(static_covariates: Vec<f64>) -> Self {
        Self { static_covariates }
    }
}

fn build_baseline_matrices(
    entry: ArrayView1<'_, f64>,
    exit: ArrayView1<'_, f64>,
    basis: &BasisDescriptor,
    reference_u: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, ReferenceConstraint), SurvivalModelError> {
    let n = entry.len();
    let num_basis = basis.num_basis();
    let mut baseline_entry = Array2::<f64>::zeros((n, num_basis));
    let mut baseline_exit = Array2::<f64>::zeros((n, num_basis));
    let mut baseline_derivative = Array2::<f64>::zeros((n, num_basis));

    for (idx, &u) in entry.iter().enumerate() {
        let basis_row = evaluate_basis(u, basis);
        baseline_entry.row_mut(idx).assign(&basis_row.view());
    }

    for (idx, &u) in exit.iter().enumerate() {
        let (basis_row, derivative_row) = evaluate_basis_and_derivative(u, basis);
        baseline_exit.row_mut(idx).assign(&basis_row.view());
        baseline_derivative
            .row_mut(idx)
            .assign(&derivative_row.view());
    }

    let reference_basis = evaluate_basis(reference_u, basis);
    let reference_constraint = ReferenceConstraint::new(reference_basis.view())?;
    let transform = reference_constraint.transform_matrix();

    let entry_constrained = baseline_entry.dot(transform);
    let exit_constrained = baseline_exit.dot(transform);
    let derivative_constrained = baseline_derivative.dot(transform);

    Ok((
        entry_constrained,
        exit_constrained,
        derivative_constrained,
        reference_constraint,
    ))
}

fn evaluate_basis(x: f64, descriptor: &BasisDescriptor) -> Array1<f64> {
    let values = cox_de_boor_basis(x, descriptor.degree, descriptor.knots.as_slice().unwrap());
    Array1::from(values)
}

fn evaluate_basis_and_derivative(
    x: f64,
    descriptor: &BasisDescriptor,
) -> (Array1<f64>, Array1<f64>) {
    let basis = cox_de_boor_basis(x, descriptor.degree, descriptor.knots.as_slice().unwrap());
    let derivative = bspline_derivative(x, descriptor.degree, descriptor.knots.as_slice().unwrap());
    (Array1::from(basis), Array1::from(derivative))
}

fn cox_de_boor_basis(x: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let num_basis = knots.len() - degree - 1;
    let mut basis = vec![0.0; num_basis];
    let span = find_span(x, degree, knots);
    let mut left = vec![0.0; degree + 1];
    let mut right = vec![0.0; degree + 1];
    let mut values = vec![0.0; degree + 1];
    values[0] = 1.0;
    for j in 1..=degree {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let den = right[r + 1] + left[j - r];
            let temp = if den.abs() > f64::EPSILON {
                values[r] / den
            } else {
                0.0
            };
            values[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        values[j] = saved;
    }
    let start = span.saturating_sub(degree);
    for j in 0..=degree {
        let idx = start + j;
        if idx < num_basis {
            basis[idx] = values[j];
        }
    }
    basis
}

fn bspline_derivative(x: f64, degree: usize, knots: &[f64]) -> Vec<f64> {
    let num_basis = knots.len() - degree - 1;
    if degree == 0 {
        return vec![0.0; num_basis];
    }
    let lower = cox_de_boor_basis(x, degree - 1, knots);
    let mut derivative = vec![0.0; num_basis];
    for i in 0..num_basis {
        let left_denom = knots[i + degree] - knots[i];
        if left_denom.abs() > f64::EPSILON {
            derivative[i] += degree as f64 / left_denom * lower[i];
        }
        let right_denom = knots[i + degree + 1] - knots[i + 1];
        if right_denom.abs() > f64::EPSILON && i + 1 < lower.len() {
            derivative[i] -= degree as f64 / right_denom * lower[i + 1];
        }
    }
    derivative
}

fn find_span(x: f64, degree: usize, knots: &[f64]) -> usize {
    let n = knots.len() - degree - 1;
    if x <= knots[degree] {
        return degree;
    }
    if x >= knots[n] {
        return n - 1;
    }
    let mut low = degree;
    let mut high = n;
    let mut mid = (low + high) / 2;
    while x < knots[mid] || x >= knots[mid + 1] {
        if x < knots[mid] {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    mid
}

#[derive(Debug, Error)]
pub enum SurvivalModelError {
    #[error("column {column} contains non-finite value {value}")]
    NonFiniteValue { column: String, value: f64 },
    #[error("age column is empty")]
    EmptyColumn(String),
    #[error("age interval invalid: entry={entry}, exit={exit}")]
    InvalidAgeInterval { entry: f64, exit: f64 },
    #[error("invalid event flags target={target}, competing={competing}")]
    InvalidEventFlag { target: u8, competing: u8 },
    #[error("target and competing events must be mutually exclusive")]
    MutuallyExclusiveEvents,
    #[error("invalid weight {0}")]
    InvalidWeight(f64),
    #[error("dimension mismatch across inputs")]
    DimensionMismatch,
    #[error("insufficient basis columns {0}")]
    InsufficientBasisColumns(usize),
    #[error("reference constraint is degenerate")]
    DegenerateReferenceConstraint,
    #[error("invalid delta {0}")]
    InvalidDelta(f64),
    #[error("invalid basis degree")]
    InvalidBasisDegree,
    #[error("knot vector must be non-decreasing")]
    InvalidKnotVector,
    #[error("insufficient knots for degree {degree} (provided {provided})")]
    InsufficientKnots { degree: usize, provided: usize },
    #[error("invalid covariate dimension: expected {expected}, found {found}")]
    InvalidCovariateDimension { expected: usize, found: usize },
    #[error("invalid competing CIF {0}")]
    InvalidCompetingCif(f64),
}

fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else if x < -30.0 {
        (1.0 + x.exp()).ln()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}
