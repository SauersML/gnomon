use crate::calibrate::basis::BasisError;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// Working model trait consumed by the unified PIRLS implementation.
///
/// The trait exposes a single update method that accepts the current
/// coefficient vector and returns the linear predictor, gradient,
/// Hessian, and deviance of the working objective. Implementations
/// are responsible for incorporating any smoothness penalties or
/// barrier terms directly into the returned gradient and Hessian so
/// the PIRLS solver can operate without branching on model families.
pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
}

/// Container returned by [`WorkingModel::update`].
#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
}

/// Frequency-weighted survival training data following the
/// Royston–Parmar cause-specific cumulative hazard parameterisation.
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
    pub fn validate(&self) -> Result<(), SurvivalDataError> {
        let n = self.age_entry.len();
        if self.age_exit.len() != n
            || self.event_target.len() != n
            || self.event_competing.len() != n
            || self.sample_weight.len() != n
        {
            return Err(SurvivalDataError::LengthMismatch);
        }
        if self.covariates.nrows() != n {
            return Err(SurvivalDataError::LengthMismatch);
        }
        for i in 0..n {
            let start = self.age_entry[i];
            let stop = self.age_exit[i];
            if !start.is_finite() || !stop.is_finite() {
                return Err(SurvivalDataError::NonFiniteAge(i));
            }
            if start >= stop {
                return Err(SurvivalDataError::InvalidAgeInterval {
                    index: i,
                    start,
                    stop,
                });
            }
            let target = self.event_target[i];
            let competing = self.event_competing[i];
            if target > 1 || competing > 1 {
                return Err(SurvivalDataError::InvalidEventFlag(i));
            }
            if target == 1 && competing == 1 {
                return Err(SurvivalDataError::MutuallyExclusiveEvents(i));
            }
            let weight = self.sample_weight[i];
            if !(weight.is_finite() && weight >= 0.0) {
                return Err(SurvivalDataError::InvalidWeight { index: i, weight });
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SurvivalDataError {
    #[error("training arrays must have matching lengths")]
    LengthMismatch,
    #[error("subject {0} has mutually exclusive event flags set")]
    MutuallyExclusiveEvents(usize),
    #[error("subject {0} has non-finite age")]
    NonFiniteAge(usize),
    #[error("subject {index} has invalid age interval [{start}, {stop})")]
    InvalidAgeInterval { index: usize, start: f64, stop: f64 },
    #[error("subject {0} has invalid event flag (must be 0 or 1)")]
    InvalidEventFlag(usize),
    #[error("subject {index} has invalid sample weight {weight}")]
    InvalidWeight { index: usize, weight: f64 },
}

/// View-based accessors used when scoring new subjects.
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: ArrayView2<'a, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgeTransform {
    pub a_min: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn new(age_entry: &Array1<f64>, guard_delta: f64) -> Result<Self, SurvivalDataError> {
        if guard_delta <= 0.0 || !guard_delta.is_finite() {
            return Err(SurvivalDataError::InvalidWeight {
                index: usize::MAX,
                weight: guard_delta,
            });
        }
        let a_min = age_entry
            .iter()
            .copied()
            .fold(f64::INFINITY, |acc, v| acc.min(v));
        if !a_min.is_finite() {
            return Err(SurvivalDataError::NonFiniteAge(0));
        }
        Ok(Self {
            a_min,
            delta: guard_delta,
        })
    }

    pub fn forward(&self, ages: ArrayView1<'_, f64>) -> Array1<f64> {
        ages.map(|age| (age - self.a_min + self.delta).ln())
    }

    pub fn derivative(&self, ages: ArrayView1<'_, f64>) -> Array1<f64> {
        ages.map(|age| 1.0 / (age - self.a_min + self.delta))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceConstraint {
    pub z_transform: Array2<f64>,
}

impl ReferenceConstraint {
    pub fn apply(&self, basis_values: &Array1<f64>) -> Array1<f64> {
        basis_values.dot(&self.z_transform)
    }

    pub fn apply_matrix(&self, basis_matrix: &Array2<f64>) -> Array2<f64> {
        basis_matrix.dot(&self.z_transform)
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
pub struct SurvivalCoefficientLayout {
    pub baseline: Range<usize>,
    pub time_varying: Option<Range<usize>>,
    pub static_covariates: Range<usize>,
    pub total: usize,
}

impl SurvivalCoefficientLayout {
    pub fn new(
        baseline: Range<usize>,
        time_varying: Option<Range<usize>>,
        static_covariates: Range<usize>,
    ) -> Self {
        let total = static_covariates.end;
        Self {
            baseline,
            time_varying,
            static_covariates,
            total,
        }
    }

    pub fn baseline_len(&self) -> usize {
        self.baseline.end - self.baseline.start
    }

    pub fn time_varying_len(&self) -> usize {
        self.time_varying
            .as_ref()
            .map(|r| r.end - r.start)
            .unwrap_or(0)
    }

    pub fn static_len(&self) -> usize {
        self.static_covariates.end - self.static_covariates.start
    }
}

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
    pub coefficients: SurvivalCoefficientLayout,
}

impl SurvivalLayout {
    pub fn aggregated_entry(&self) -> Array2<f64> {
        let n = self.baseline_entry.nrows();
        let mut design = Array2::zeros((n, self.coefficients.total));
        design
            .slice_mut(s![.., self.coefficients.baseline.clone()])
            .assign(&self.baseline_entry);
        if let Some(ref tv) = self.time_varying_entry {
            if let Some(range) = &self.coefficients.time_varying {
                design.slice_mut(s![.., range.clone()]).assign(tv);
            }
        }
        design
            .slice_mut(s![.., self.coefficients.static_covariates.clone()])
            .assign(&self.static_covariates);
        design
    }

    pub fn aggregated_exit(&self) -> Array2<f64> {
        let n = self.baseline_exit.nrows();
        let mut design = Array2::zeros((n, self.coefficients.total));
        design
            .slice_mut(s![.., self.coefficients.baseline.clone()])
            .assign(&self.baseline_exit);
        if let Some(ref tv) = self.time_varying_exit {
            if let Some(range) = &self.coefficients.time_varying {
                design.slice_mut(s![.., range.clone()]).assign(tv);
            }
        }
        design
            .slice_mut(s![.., self.coefficients.static_covariates.clone()])
            .assign(&self.static_covariates);
        design
    }

    pub fn aggregated_derivative_exit(&self) -> Array2<f64> {
        let n = self.baseline_derivative_exit.nrows();
        let mut design = Array2::zeros((n, self.coefficients.total));
        design
            .slice_mut(s![.., self.coefficients.baseline.clone()])
            .assign(&self.baseline_derivative_exit);
        if let Some(ref tv) = self.time_varying_derivative_exit {
            if let Some(range) = &self.coefficients.time_varying {
                design.slice_mut(s![.., range.clone()]).assign(tv);
            }
        }
        design
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BarrierConfig {
    pub weight: f64,
    pub scale: f64,
}

impl Default for BarrierConfig {
    fn default() -> Self {
        Self {
            weight: 1e-4,
            scale: 1.0,
        }
    }
}

/// Working model implementing the RP likelihood.
pub struct WorkingModelSurvival {
    pub data: SurvivalTrainingData,
    pub layout: SurvivalLayout,
    pub guard_epsilon: f64,
    pub barrier: BarrierConfig,
}

impl WorkingModelSurvival {
    pub fn new(
        data: SurvivalTrainingData,
        layout: SurvivalLayout,
        guard_epsilon: f64,
        barrier: BarrierConfig,
    ) -> Result<Self, SurvivalDataError> {
        data.validate()?;
        Ok(Self {
            data,
            layout,
            guard_epsilon,
            barrier,
        })
    }

    fn ensure_dimensions(&self, beta: &Array1<f64>) {
        assert_eq!(beta.len(), self.layout.coefficients.total);
    }

    fn softplus(z: f64) -> f64 {
        if z > 0.0 {
            z + (1.0 + (-z).exp()).ln()
        } else {
            (1.0 + z.exp()).ln()
        }
    }

    fn sigmoid(z: f64) -> f64 {
        if z >= 0.0 {
            let ez = (-z).exp();
            1.0 / (1.0 + ez)
        } else {
            let ez = z.exp();
            ez / (1.0 + ez)
        }
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
        self.ensure_dimensions(beta);

        let n = self.data.age_entry.len();
        let design_entry = self.layout.aggregated_entry();
        let design_exit = self.layout.aggregated_exit();
        let design_derivative_exit = self.layout.aggregated_derivative_exit();

        let eta_entry = design_entry.dot(beta);
        let eta_exit = design_exit.dot(beta);
        let deta_exit = design_derivative_exit.dot(beta);

        let h_entry = eta_entry.mapv(f64::exp);
        let h_exit = eta_exit.mapv(f64::exp);
        let delta_h = &h_exit - &h_entry;

        let mut loglik = 0.0;
        let mut barrier_value = 0.0;

        let mut grad_loglik = Array1::zeros(beta.len());
        let mut grad_barrier = Array1::zeros(beta.len());
        let mut hess_loglik = Array2::zeros((beta.len(), beta.len()));
        let mut hess_barrier = Array2::zeros((beta.len(), beta.len()));

        for i in 0..n {
            let weight = self.data.sample_weight[i];
            if weight == 0.0 {
                continue;
            }

            let target = f64::from(self.data.event_target[i]);
            let entry_row = design_entry.row(i);
            let exit_row = design_exit.row(i);
            let deriv_row = design_derivative_exit.row(i);

            let eta_exit_i = eta_exit[i];
            let h_entry_i = h_entry[i];
            let h_exit_i = h_exit[i];
            let delta_h_i = delta_h[i];

            let derivative = deta_exit[i];
            let guarded_derivative = derivative.max(self.guard_epsilon);

            let hazard_log_term = eta_exit_i + guarded_derivative.ln();
            loglik += weight * (target * hazard_log_term - delta_h_i);

            let guard_active = derivative > self.guard_epsilon;
            let mut event_grad = exit_row.to_owned();
            if guard_active {
                event_grad += &(&deriv_row / guarded_derivative);
            }
            let survival_grad = &exit_row * h_exit_i - &entry_row * h_entry_i;

            grad_loglik +=
                &(event_grad.mapv(|v| v * target * weight) - survival_grad.mapv(|v| v * weight));

            let exit_outer = outer(exit_row, exit_row).mapv(|v| v * h_exit_i * weight);
            let entry_outer = outer(entry_row, entry_row).mapv(|v| v * h_entry_i * weight);
            hess_loglik -= &exit_outer;
            hess_loglik += &entry_outer;

            if guard_active {
                let deriv_outer = outer(deriv_row, deriv_row)
                    .mapv(|v| v * weight * target / (guarded_derivative * guarded_derivative));
                hess_loglik -= &deriv_outer;
            }

            let z = -derivative / self.barrier.scale;
            let sigma = Self::sigmoid(z);
            let sp = Self::softplus(z);
            let barrier_grad_coeff = -sigma / self.barrier.scale;
            let barrier_hess_coeff =
                sigma * (1.0 - sigma) / (self.barrier.scale * self.barrier.scale);

            barrier_value += weight * self.barrier.weight * sp;
            let barrier_grad_row =
                deriv_row.mapv(|v| v * barrier_grad_coeff * weight * self.barrier.weight);
            grad_barrier += &barrier_grad_row;
            let barrier_outer = outer(deriv_row, deriv_row)
                .mapv(|v| v * barrier_hess_coeff * weight * self.barrier.weight);
            hess_barrier += &barrier_outer;
        }

        let mut gradient = grad_barrier * 2.0 - &(grad_loglik * 2.0);
        let mut hessian = hess_barrier * 2.0 - &(hess_loglik * 2.0);
        let mut deviance = -2.0 * (loglik - barrier_value);

        for block in &self.layout.penalties.blocks {
            if block.lambda == 0.0 {
                continue;
            }
            assert_eq!(block.matrix.nrows(), beta.len());
            assert_eq!(block.matrix.ncols(), beta.len());

            let s_beta = block.matrix.dot(beta);
            let scale = 2.0 * block.lambda;
            gradient += &s_beta.mapv(|v| v * scale);
            hessian += &block.matrix.mapv(|v| v * scale);
            deviance += block.lambda * beta.dot(&s_beta);
        }

        WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        }
    }
}

fn outer(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> Array2<f64> {
    let mut result = Array2::zeros((a.len(), b.len()));
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            result[[i, j]] = ai * bj;
        }
    }
    result
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisDescriptor {
    pub knots: Vec<f64>,
    pub degree: usize,
    pub num_basis: usize,
}

impl BasisDescriptor {
    pub fn validate(&self) -> Result<(), BasisError> {
        if self.degree == 0 {
            return Err(BasisError::InvalidDegree(0));
        }
        if self.knots.windows(2).any(|w| w[0] > w[1]) {
            return Err(BasisError::InvalidKnotVector(
                "knot vector must be non-decreasing".to_string(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyDescriptor {
    pub order: usize,
    pub blocks: Vec<Array2<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovariateLayout {
    pub column_names: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LdltFactor {
    pub matrix: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermutationMatrix {
    pub permutation: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Inertia {
    pub negative: usize,
    pub zero: usize,
    pub positive: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CholeskyFactor {
    pub matrix: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub coefficient_layout: SurvivalCoefficientLayout,
}

#[derive(Debug, Clone)]
pub struct Covariates {
    pub baseline: Array1<f64>,
    pub time_varying: Option<Array1<f64>>,
    pub static_covariates: Array1<f64>,
}

impl SurvivalModelArtifacts {
    pub fn cumulative_hazard(&self, covariates: &Covariates) -> f64 {
        let mut eta = 0.0;
        if self.coefficient_layout.baseline_len() > 0 {
            let baseline_constrained = self.reference_constraint.apply(&covariates.baseline);
            let beta = self
                .coefficients
                .slice(s![self.coefficient_layout.baseline.clone()])
                .to_owned();
            eta += baseline_constrained.dot(&beta);
        }
        if let Some(range) = &self.coefficient_layout.time_varying {
            if let Some(ref tv) = covariates.time_varying {
                let beta = self.coefficients.slice(s![range.clone()]).to_owned();
                eta += tv.dot(&beta);
            }
        }
        let static_slice = self
            .coefficients
            .slice(s![self.coefficient_layout.static_covariates.clone()])
            .to_owned();
        eta += covariates.static_covariates.dot(&static_slice);
        eta.exp()
    }

    pub fn cumulative_incidence(&self, covariates: &Covariates) -> f64 {
        let h = self.cumulative_hazard(covariates);
        1.0 - (-h).exp()
    }

    pub fn conditional_absolute_risk(
        &self,
        covariates_t0: &Covariates,
        covariates_t1: &Covariates,
        cif_competing_t0: f64,
    ) -> f64 {
        let cif_t0 = self.cumulative_incidence(covariates_t0);
        let cif_t1 = self.cumulative_incidence(covariates_t1);
        let delta_f = cif_t1 - cif_t0;
        let denom = (1.0 - cif_t0 - cif_competing_t0).max(1e-12);
        (delta_f / denom).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2, array};

    fn build_layout(n: usize) -> SurvivalLayout {
        let baseline_entry = Array2::ones((n, 1));
        let baseline_exit = Array2::from_shape_fn((n, 1), |(i, _)| 1.0 + i as f64 * 0.1);
        let baseline_derivative_exit = Array2::from_elem((n, 1), 1.0);
        let static_covariates = Array2::from_elem((n, 1), 1.0);

        let age_transform = AgeTransform {
            a_min: 40.0,
            delta: 0.1,
        };
        let reference_constraint = ReferenceConstraint {
            z_transform: Array2::from_elem((1, 1), 1.0),
        };

        let mut penalty_matrix = Array2::zeros((2, 2));
        penalty_matrix[[0, 0]] = 1.0;
        let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
            matrix: penalty_matrix,
            lambda: 1.0,
        }]);

        let coeffs = SurvivalCoefficientLayout::new(0..1, None, 1..2);

        SurvivalLayout {
            baseline_entry,
            baseline_exit,
            baseline_derivative_exit,
            time_varying_entry: None,
            time_varying_exit: None,
            time_varying_derivative_exit: None,
            static_covariates,
            age_transform,
            reference_constraint,
            penalties,
            coefficients: coeffs,
        }
    }

    fn build_training_data() -> SurvivalTrainingData {
        SurvivalTrainingData {
            age_entry: array![50.0, 51.0],
            age_exit: array![55.0, 56.0],
            event_target: array![1, 0],
            event_competing: array![0, 1],
            sample_weight: array![1.0, 2.0],
            covariates: Array2::zeros((2, 1)),
        }
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let layout = build_layout(2);
        let data = build_training_data();
        let mut working = WorkingModelSurvival::new(data, layout, 1e-6, BarrierConfig::default())
            .expect("valid working model");

        let beta = array![0.2, -0.1];
        let state = working.update(&beta);

        let eps = 1e-6;
        let mut numeric_grad = Array1::zeros(beta.len());
        for j in 0..beta.len() {
            let mut beta_plus = beta.clone();
            beta_plus[j] += eps;
            let dev_plus = working.update(&beta_plus).deviance;

            let mut beta_minus = beta.clone();
            beta_minus[j] -= eps;
            let dev_minus = working.update(&beta_minus).deviance;

            numeric_grad[j] = (dev_plus - dev_minus) / (2.0 * eps);
        }

        assert_relative_eq!(
            numeric_grad[0],
            state.gradient[0],
            max_relative = 1e-4,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            numeric_grad[1],
            state.gradient[1],
            max_relative = 1e-4,
            epsilon = 1e-6
        );
    }

    #[test]
    fn hessian_matches_finite_difference() {
        let layout = build_layout(2);
        let data = build_training_data();
        let mut working = WorkingModelSurvival::new(data, layout, 1e-6, BarrierConfig::default())
            .expect("valid working model");

        let beta = array![0.3, 0.2];
        let state = working.update(&beta);

        let eps = 1e-5;
        let mut numeric_hess = Array2::zeros((beta.len(), beta.len()));
        for i in 0..beta.len() {
            for j in 0..beta.len() {
                let mut beta_pp = beta.clone();
                beta_pp[i] += eps;
                beta_pp[j] += eps;
                let dev_pp = working.update(&beta_pp).deviance;

                let mut beta_pm = beta.clone();
                beta_pm[i] += eps;
                beta_pm[j] -= eps;
                let dev_pm = working.update(&beta_pm).deviance;

                let mut beta_mp = beta.clone();
                beta_mp[i] -= eps;
                beta_mp[j] += eps;
                let dev_mp = working.update(&beta_mp).deviance;

                let mut beta_mm = beta.clone();
                beta_mm[i] -= eps;
                beta_mm[j] -= eps;
                let dev_mm = working.update(&beta_mm).deviance;

                numeric_hess[[i, j]] = (dev_pp - dev_pm - dev_mp + dev_mm) / (4.0 * eps * eps);
            }
        }

        let diff = &numeric_hess - &state.hessian;
        for value in diff.iter() {
            assert!(value.abs() < 5e-3, "hessian mismatch: {}", value);
        }
    }

    #[test]
    fn deviance_is_finite() {
        let layout = build_layout(2);
        let data = build_training_data();
        let mut working = WorkingModelSurvival::new(data, layout, 1e-6, BarrierConfig::default())
            .expect("valid working model");
        let beta = array![0.1, -0.2];
        let state = working.update(&beta);

        assert!(state.deviance.is_finite());
    }

    #[test]
    fn prediction_monotonicity() {
        let layout = build_layout(2);
        let data = build_training_data();
        let _ = WorkingModelSurvival::new(data, layout.clone(), 1e-6, BarrierConfig::default())
            .expect("valid working model");
        let beta = array![0.2, 0.3];

        let artifacts = SurvivalModelArtifacts {
            coefficients: beta.clone(),
            age_basis: BasisDescriptor {
                knots: vec![0.0, 1.0],
                degree: 1,
                num_basis: 1,
            },
            time_varying_basis: None,
            static_covariate_layout: CovariateLayout {
                column_names: vec!["static".to_string()],
            },
            penalties: PenaltyDescriptor {
                order: 2,
                blocks: vec![Array2::eye(1)],
            },
            age_transform: layout.age_transform.clone(),
            reference_constraint: layout.reference_constraint.clone(),
            hessian_factor: None,
            coefficient_layout: layout.coefficients.clone(),
        };

        let covariates_t0 = Covariates {
            baseline: array![0.5],
            time_varying: None,
            static_covariates: array![1.0],
        };
        let covariates_t1 = Covariates {
            baseline: array![0.6],
            time_varying: None,
            static_covariates: array![1.0],
        };

        let cif_t0 = artifacts.cumulative_incidence(&covariates_t0);
        let cif_t1 = artifacts.cumulative_incidence(&covariates_t1);
        assert!(cif_t1 >= cif_t0);

        let risk = artifacts.conditional_absolute_risk(&covariates_t0, &covariates_t1, 0.1);
        assert!(risk >= 0.0);
    }
}
