use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Describes all supported model families for the calibration module.
///
/// The GAM pathway continues to operate in terms of link functions while the
/// survival pathway bundles all of the metadata required for the Royston–Parmar
/// cumulative hazard parameterisation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelFamily {
    Gam(crate::calibrate::model::LinkFunction),
    Survival(SurvivalSpec),
}

/// Configuration required to build the survival working model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalSpec {
    pub baseline_knots: Array1<f64>,
    pub baseline_degree: usize,
    pub time_varying_knots: Option<Array1<f64>>,
    pub time_varying_degree: Option<usize>,
    pub penalty_order: usize,
}

/// Describes the guarded logarithmic age transform reused at both training and
/// prediction time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub age_min: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn new(age_min: f64, delta: f64) -> Self {
        assert!(delta.is_finite() && delta > 0.0);
        Self { age_min, delta }
    }

    #[inline]
    pub fn transform(&self, age: f64) -> f64 {
        (age - self.age_min + self.delta).ln()
    }

    #[inline]
    pub fn derivative(&self, age: f64) -> f64 {
        1.0 / (age - self.age_min + self.delta)
    }
}

/// Stores an explicit linear transformation that removes the null direction of
/// the baseline spline (reference constraint).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceConstraint {
    pub z: Array2<f64>,
}

impl ReferenceConstraint {
    pub fn apply(&self, basis: &Array2<f64>) -> Array2<f64> {
        basis.dot(&self.z)
    }
}

/// Penalty block metadata for smoothing matrices.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyBlock {
    pub lambda: f64,
    pub matrix: Array2<f64>,
}

impl PenaltyBlock {
    pub fn gradient(&self, beta: &Array1<f64>) -> Array1<f64> {
        self.matrix.dot(beta) * self.lambda
    }

    pub fn hessian(&self) -> Array2<f64> {
        &self.matrix * self.lambda
    }
}

/// A collection of penalty blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn add(&mut self, block: PenaltyBlock) {
        self.blocks.push(block);
    }

    pub fn apply_to(&self, beta: &Array1<f64>) -> (Array1<f64>, Array2<f64>) {
        let mut grad = Array1::zeros(beta.len());
        let mut hess = Array2::zeros((beta.len(), beta.len()));
        for block in &self.blocks {
            grad += &block.gradient(beta);
            hess += &block.hessian();
        }
        (grad, hess)
    }
}

/// Aggregated cached designs required for the survival working model.
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
    pub fn num_coefficients(&self) -> usize {
        let mut cols = self.baseline_entry.ncols();
        if let Some(tv) = &self.time_varying_entry {
            cols += tv.ncols();
        }
        cols += self.static_covariates.ncols();
        cols
    }

    fn concat_exit(&self) -> Array2<f64> {
        let n = self.baseline_exit.nrows();
        let mut blocks: Vec<Array2<f64>> = Vec::new();
        blocks.push(self.baseline_exit.clone());
        if let Some(tv) = &self.time_varying_exit {
            blocks.push(tv.clone());
        }
        blocks.push(self.static_covariates.clone());
        let cols = blocks.iter().map(|m| m.ncols()).sum();
        let mut exit = Array2::zeros((n, cols));
        let mut start = 0;
        for block in blocks {
            let end = start + block.ncols();
            exit.slice_mut(s![.., start..end]).assign(&block);
            start = end;
        }
        exit
    }

    fn concat_entry(&self) -> Array2<f64> {
        let n = self.baseline_entry.nrows();
        let mut blocks: Vec<Array2<f64>> = Vec::new();
        blocks.push(self.baseline_entry.clone());
        if let Some(tv) = &self.time_varying_entry {
            blocks.push(tv.clone());
        }
        blocks.push(self.static_covariates.clone());
        let cols = blocks.iter().map(|m| m.ncols()).sum();
        let mut entry = Array2::zeros((n, cols));
        let mut start = 0;
        for block in blocks {
            let end = start + block.ncols();
            entry.slice_mut(s![.., start..end]).assign(&block);
            start = end;
        }
        entry
    }

    fn concat_derivative_exit(&self) -> Array2<f64> {
        let n = self.baseline_derivative_exit.nrows();
        let mut blocks: Vec<Array2<f64>> = Vec::new();
        blocks.push(self.baseline_derivative_exit.clone());
        if let Some(tv) = &self.time_varying_derivative_exit {
            blocks.push(tv.clone());
        }
        blocks.push(Array2::zeros((n, self.static_covariates.ncols())));
        let cols = blocks.iter().map(|m| m.ncols()).sum();
        let mut deriv = Array2::zeros((n, cols));
        let mut start = 0;
        for block in blocks {
            let end = start + block.ncols();
            deriv.slice_mut(s![.., start..end]).assign(&block);
            start = end;
        }
        deriv
    }
}

/// Survival training bundle.
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

/// Prediction-time inputs.
#[derive(Debug, Clone, Copy)]
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
}

/// Numerical guard used when evaluating the log derivative.
const DERIVATIVE_GUARD: f64 = 1e-12;

/// Soft barrier strength.
const BARRIER_SCALE: f64 = 5.0;

fn softplus(x: f64) -> f64 {
    if x.is_infinite() {
        if x.is_sign_positive() { x } else { 0.0 }
    } else if x > 50.0 {
        x
    } else if x < -50.0 {
        x.exp()
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

/// Working model state returned during PIRLS iterations.
#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
}

/// Trait shared by all working models participating in P-IRLS.
pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError>;
}

/// Errors emitted during survival model construction or evaluation.
#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("Design column mismatch: expected {expected}, found {found}")]
    ColumnMismatch { expected: usize, found: usize },
    #[error("Non-finite likelihood contribution encountered")]
    NonFiniteLikelihood,
    #[error("Singular matrix encountered in Hessian solve")]
    SingularMatrix,
}

/// Survival working model.
pub struct WorkingModelSurvival {
    layout: SurvivalLayout,
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    sample_weight: Array1<f64>,
}

impl WorkingModelSurvival {
    pub fn new(
        layout: SurvivalLayout,
        event_target: Array1<u8>,
        event_competing: Array1<u8>,
        sample_weight: Array1<f64>,
    ) -> Result<Self, SurvivalError> {
        let n = layout.baseline_entry.nrows();
        if layout.baseline_exit.nrows() != n
            || layout.static_covariates.nrows() != n
            || layout.baseline_derivative_exit.nrows() != n
            || event_target.len() != n
            || event_competing.len() != n
            || sample_weight.len() != n
        {
            return Err(SurvivalError::ColumnMismatch {
                expected: n,
                found: event_target.len(),
            });
        }
        Ok(Self {
            layout,
            event_target,
            event_competing,
            sample_weight,
        })
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> Result<WorkingState, SurvivalError> {
        let entry = self.layout.concat_entry();
        let exit = self.layout.concat_exit();
        let deriv_exit = self.layout.concat_derivative_exit();
        if beta.len() != exit.ncols() {
            return Err(SurvivalError::ColumnMismatch {
                expected: exit.ncols(),
                found: beta.len(),
            });
        }
        let eta_entry = entry.dot(beta);
        let eta_exit = exit.dot(beta);
        let deriv_exit_vec = deriv_exit.dot(beta);
        let mut gradient = Array1::zeros(beta.len());
        let mut hessian = Array2::zeros((beta.len(), beta.len()));
        let mut deviance = 0.0;
        let mut eta = eta_exit.clone();
        let barrier_scale = BARRIER_SCALE;

        for i in 0..exit.nrows() {
            let w = self.sample_weight[i];
            let eta_e = eta_exit[i];
            let eta_s = eta_entry[i];
            let h_exit = eta_e.exp();
            let h_entry = eta_s.exp();
            let delta_h = h_exit - h_entry;
            let x_exit = exit.slice(s![i, ..]).to_owned();
            let x_entry = entry.slice(s![i, ..]).to_owned();
            let j_exit = deriv_exit.slice(s![i, ..]).to_owned();
            let d_eta = deriv_exit_vec[i];
            let safe_d_eta = d_eta.max(DERIVATIVE_GUARD);
            let event_target = self.event_target[i] as f64;
            let log_term = eta_e + safe_d_eta.ln();
            let ll = event_target * log_term - delta_h;
            let weight = w;

            if !ll.is_finite() {
                return Err(SurvivalError::NonFiniteLikelihood);
            }
            deviance -= 2.0 * weight * ll;

            let grad_common = -weight * (h_exit * &x_exit - h_entry * &x_entry);
            gradient += &grad_common;
            let mut hess_common = Array2::zeros((beta.len(), beta.len()));
            let outer_exit = outer(&x_exit, &x_exit);
            let outer_entry = outer(&x_entry, &x_entry);
            hess_common -= &(weight * h_exit * &outer_exit);
            hess_common += &(weight * h_entry * &outer_entry);

            if event_target > 0.0 {
                gradient += &(weight * event_target * &x_exit);
                let frac = weight * event_target / safe_d_eta;
                gradient += &(frac * &j_exit);
                let outer_j = outer(&j_exit, &j_exit);
                hess_common -= &(weight * event_target / (safe_d_eta * safe_d_eta) * &outer_j);
            }

            hessian += &hess_common;

            // Soft barrier on derivative to maintain monotonic cumulative hazard
            let barrier_arg = -d_eta / barrier_scale;
            let barrier = barrier_scale * softplus(barrier_arg);
            let barrier_grad = -sigmoid(barrier_arg);
            let barrier_hess = barrier_grad * (1.0 - barrier_grad) / barrier_scale;
            deviance += 2.0 * weight * barrier;
            gradient += &(2.0 * weight * barrier_grad * &j_exit);
            let outer_j = outer(&j_exit, &j_exit);
            hessian += &(2.0 * weight * barrier_hess * &outer_j);
        }

        let (pen_grad, pen_hess) = self.layout.penalties.apply_to(beta);
        gradient += &pen_grad;
        hessian += &pen_hess;

        Ok(WorkingState {
            eta,
            gradient,
            hessian,
            deviance,
        })
    }
}

fn outer(row: &Array1<f64>, other: &Array1<f64>) -> Array2<f64> {
    let mut mat = Array2::zeros((row.len(), other.len()));
    for i in 0..row.len() {
        for j in 0..other.len() {
            mat[[i, j]] = row[i] * other[j];
        }
    }
    mat
}

/// Stored Hessian factorisation metadata.
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

/// Persisted survival model components.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub fn cumulative_hazard(&self, age: f64, covariates: &ArrayView1<'_, f64>) -> f64 {
        let u = self.age_transform.transform(age);
        let basis_exit = self.reference_constraint.apply(&self.age_basis);
        let baseline_values = evaluate_basis_at_age(&basis_exit, u);
        let mut coeff_start = 0;
        let mut eta = baseline_values.dot(&self.coefficients.slice(s![..basis_exit.ncols()]));
        coeff_start += basis_exit.ncols();

        if let Some(tv_basis) = &self.time_varying_basis {
            let tv_values = evaluate_basis_at_age(tv_basis, u);
            eta += tv_values.dot(
                &self
                    .coefficients
                    .slice(s![coeff_start..coeff_start + tv_values.len()]),
            );
            coeff_start += tv_values.len();
        }

        debug_assert_eq!(self.static_covariate_layout.ncols(), covariates.len());
        debug_assert!(coeff_start + covariates.len() <= self.coefficients.len());
        let static_coeffs = self
            .coefficients
            .slice(s![coeff_start..coeff_start + covariates.len()]);
        let static_offset = covariates.dot(&static_coeffs);
        (eta + static_offset + u).exp()
    }

    pub fn cumulative_incidence(&self, age: f64, covariates: &ArrayView1<'_, f64>) -> f64 {
        let h = self.cumulative_hazard(age, covariates);
        1.0 - (-h).exp()
    }

    pub fn conditional_absolute_risk(
        &self,
        t0: f64,
        t1: f64,
        covariates: &ArrayView1<'_, f64>,
        cif_competing_t0: f64,
    ) -> f64 {
        let cif0 = self.cumulative_incidence(t0, covariates);
        let cif1 = self.cumulative_incidence(t1, covariates);
        let delta = (cif1 - cif0).max(0.0);
        let denom = (1.0 - cif0 - cif_competing_t0).max(1e-12);
        (delta / denom).min(1.0)
    }
}

fn evaluate_basis_at_age(basis: &Array2<f64>, u: f64) -> Array1<f64> {
    let mut values = Array1::zeros(basis.ncols());
    for col in 0..basis.ncols() {
        let mut power = 1.0;
        let mut acc = 0.0;
        for row in 0..basis.nrows() {
            acc += basis[[row, col]] * power;
            power *= u;
        }
        values[col] = acc;
    }
    values
}

/// Delta-method standard error helper using the stored Hessian factor.
pub fn delta_method_variance(
    factor: &HessianFactor,
    gradient: &Array1<f64>,
) -> Result<f64, SurvivalError> {
    match factor {
        HessianFactor::Observed {
            ldlt_factor,
            permutation,
            ..
        } => {
            let mut permuted_grad = Array1::zeros(gradient.len());
            for (idx, &p) in permutation.iter().enumerate() {
                permuted_grad[idx] = gradient[p];
            }
            let sol = gaussian_solve(ldlt_factor, &permuted_grad)?;
            let mut unpermuted = Array1::zeros(gradient.len());
            for (idx, &p) in permutation.iter().enumerate() {
                unpermuted[p] = sol[idx];
            }
            Ok(gradient.dot(&unpermuted))
        }
        HessianFactor::Expected { cholesky_factor } => {
            let sol = gaussian_solve(cholesky_factor, gradient)?;
            Ok(gradient.dot(&sol))
        }
    }
}

fn gaussian_solve(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, SurvivalError> {
    let n = matrix.nrows();
    if matrix.ncols() != n || rhs.len() != n {
        return Err(SurvivalError::SingularMatrix);
    }
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n]] = rhs[i];
    }
    for col in 0..n {
        let mut pivot = col;
        let mut max_val = aug[[pivot, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                pivot = row;
            }
        }
        if max_val <= 1e-12 {
            return Err(SurvivalError::SingularMatrix);
        }
        if pivot != col {
            for j in col..=n {
                aug.swap((col, j), (pivot, j));
            }
        }
        let pivot_val = aug[[col, col]];
        for row in (col + 1)..n {
            let factor = aug[[row, col]] / pivot_val;
            for j in col..=n {
                aug[[row, j]] -= factor * aug[[col, j]];
            }
        }
    }
    let mut solution = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * solution[j];
        }
        let denom = aug[[i, i]];
        if denom.abs() <= 1e-12 {
            return Err(SurvivalError::SingularMatrix);
        }
        solution[i] = sum / denom;
    }
    Ok(solution)
}

/// Calibration helper on the logit scale.
pub fn calibrate_logit(prob: ArrayView1<'_, f64>) -> Array1<f64> {
    prob.map(|&p| (p / (1.0 - p.max(1e-12))).ln())
}

/// Validates monotonicity diagnostics on the derivative grid.
pub fn monotonicity_violation_fraction(derivatives: ArrayView1<'_, f64>) -> f64 {
    let violations = derivatives.iter().filter(|&&d| d < 0.0).count();
    violations as f64 / derivatives.len() as f64
}

/// Computes weighted log-likelihood for diagnostics.
pub fn weighted_log_likelihood(
    weights: &Array1<f64>,
    eta_entry: &Array1<f64>,
    eta_exit: &Array1<f64>,
    deriv_exit: &Array1<f64>,
    event_target: &Array1<u8>,
) -> f64 {
    let mut acc = 0.0;
    for i in 0..weights.len() {
        let w = weights[i];
        let delta_h = eta_exit[i].exp() - eta_entry[i].exp();
        let log_term =
            (event_target[i] as f64) * (eta_exit[i] + deriv_exit[i].max(DERIVATIVE_GUARD).ln());
        acc += w * (log_term - delta_h);
    }
    acc
}
