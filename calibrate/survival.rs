use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// WorkingState returned by [`WorkingModel`] implementations.
#[derive(Debug, Clone)]
pub struct WorkingState {
    pub eta: Array1<f64>,
    pub gradient: Array1<f64>,
    pub hessian: Array2<f64>,
    pub deviance: f64,
}

/// Common interface shared by Generalised Additive Models and survival models
/// so the PIRLS implementation can operate without branching on the active
/// likelihood family.
pub trait WorkingModel {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
}

/// Guard that keeps the transformed age domain well-behaved by keeping all
/// input ages strictly above the minimum observed age.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct AgeTransform {
    pub a_min: f64,
    pub delta: f64,
}

impl AgeTransform {
    pub fn new(min_age: f64, delta: f64) -> Self {
        assert!(delta > 0.0, "delta must be positive");
        Self {
            a_min: min_age,
            delta,
        }
    }

    pub fn transform(&self, age: f64) -> f64 {
        let shifted = age - self.a_min + self.delta;
        assert!(shifted > 0.0, "age must be at least a_min - delta");
        shifted.ln()
    }

    pub fn derivative_scale(&self, age: f64) -> f64 {
        let shifted = age - self.a_min + self.delta;
        assert!(shifted > 0.0, "age must be at least a_min - delta");
        1.0 / shifted
    }
}

/// Encodes an explicit reference constraint so the baseline spline has a
/// well-defined level. The Z matrix removes the null direction from the basis
/// and is cached so scoring can faithfully reconstruct the constrained design.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ReferenceConstraint {
    pub z: Array2<f64>,
}

impl ReferenceConstraint {
    pub fn apply(&self, basis: &Array2<f64>) -> Array2<f64> {
        basis.dot(&self.z)
    }
}

/// Frequency weights (no robust-IPW support). The schema mirrors the training
/// input layout from the survival plan.
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

#[derive(Debug, Clone, Copy)]
pub struct CovariateViews<'a> {
    pub pgs: ArrayView1<'a, f64>,
    pub sex: ArrayView1<'a, f64>,
    pub pcs: ArrayView2<'a, f64>,
}

#[derive(Debug, Clone, Copy)]
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: CovariateViews<'a>,
}

/// Penalty blocks indexed by coefficient ranges.  The same representation is
/// shared between training and scoring artefacts so the penalised observed
/// information can be reconstructed exactly.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyBlock {
    pub range: Range<usize>,
    pub matrix: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyBlocks {
    pub blocks: Vec<PenaltyBlock>,
}

impl PenaltyBlocks {
    pub fn new(blocks: Vec<PenaltyBlock>) -> Self {
        Self { blocks }
    }

    pub fn apply(
        &self,
        beta: &Array1<f64>,
        gradient: &mut Array1<f64>,
        hessian: &mut Array2<f64>,
    ) -> f64 {
        let mut penalty = 0.0;
        for block in &self.blocks {
            let slice = beta.slice(s![block.range.clone()]);
            let contrib = block.matrix.dot(&slice);
            penalty += slice.dot(&contrib);

            let update = &contrib * 2.0;
            let mut grad_slice = gradient.slice_mut(s![block.range.clone()]);
            grad_slice += &update;

            let mut hess_view = hessian.slice_mut(s![block.range.clone(), block.range.clone()]);
            hess_view += &(block.matrix.clone() * 2.0);
        }
        penalty
    }
}

/// Column metadata cached on the layout so coefficient slices can be retrieved
/// without repeated range arithmetic during PIRLS iterations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnRanges {
    pub baseline: Range<usize>,
    pub time_varying: Option<Range<usize>>,
    pub static_covariates: Range<usize>,
}

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
    pub column_ranges: ColumnRanges,
    pub design_entry: Array2<f64>,
    pub design_exit: Array2<f64>,
    pub design_derivative_exit: Array2<f64>,
}

impl SurvivalLayout {
    pub fn new(
        baseline_entry: Array2<f64>,
        baseline_exit: Array2<f64>,
        baseline_derivative_exit: Array2<f64>,
        time_varying_entry: Option<Array2<f64>>,
        time_varying_exit: Option<Array2<f64>>,
        time_varying_derivative_exit: Option<Array2<f64>>,
        static_covariates: Array2<f64>,
        age_transform: AgeTransform,
        reference_constraint: ReferenceConstraint,
        penalties: PenaltyBlocks,
    ) -> Self {
        let n_obs = baseline_entry.nrows();
        assert_eq!(baseline_exit.nrows(), n_obs);
        assert_eq!(baseline_derivative_exit.nrows(), n_obs);
        if let Some(ref entry) = time_varying_entry {
            assert_eq!(entry.nrows(), n_obs);
        }
        if let Some(ref exit) = time_varying_exit {
            assert_eq!(exit.nrows(), n_obs);
        }
        if let Some(ref deriv) = time_varying_derivative_exit {
            assert_eq!(deriv.nrows(), n_obs);
        }
        assert_eq!(static_covariates.nrows(), n_obs);

        let mut offset = 0usize;
        let baseline_range = offset..offset + baseline_entry.ncols();
        offset += baseline_entry.ncols();

        let time_varying_range = time_varying_entry.as_ref().map(|entry| {
            let range = offset..offset + entry.ncols();
            offset += entry.ncols();
            range
        });

        let static_range = offset..offset + static_covariates.ncols();

        let design_entry = Self::assemble_design(
            n_obs,
            baseline_entry.view(),
            time_varying_entry.as_ref().map(|m| m.view()),
            static_covariates.view(),
        );
        let design_exit = Self::assemble_design(
            n_obs,
            baseline_exit.view(),
            time_varying_exit.as_ref().map(|m| m.view()),
            static_covariates.view(),
        );
        let design_derivative_exit = Self::assemble_design(
            n_obs,
            baseline_derivative_exit.view(),
            time_varying_derivative_exit.as_ref().map(|m| m.view()),
            Array2::zeros((n_obs, static_covariates.ncols())).view(),
        );

        Self {
            baseline_entry,
            baseline_exit,
            baseline_derivative_exit,
            time_varying_entry,
            time_varying_exit,
            time_varying_derivative_exit,
            static_covariates,
            age_transform,
            reference_constraint,
            penalties,
            column_ranges: ColumnRanges {
                baseline: baseline_range,
                time_varying: time_varying_range,
                static_covariates: static_range,
            },
            design_entry,
            design_exit,
            design_derivative_exit,
        }
    }

    fn assemble_design(
        n_obs: usize,
        primary: ArrayView2<f64>,
        time_varying: Option<ArrayView2<f64>>,
        static_covariates: ArrayView2<f64>,
    ) -> Array2<f64> {
        let mut cols = primary.ncols() + static_covariates.ncols();
        if let Some(tv) = time_varying {
            cols += tv.ncols();
        }
        let mut design = Array2::zeros((n_obs, cols));
        let mut offset = 0usize;
        design
            .slice_mut(s![.., offset..offset + primary.ncols()])
            .assign(&primary);
        offset += primary.ncols();
        if let Some(tv) = time_varying {
            design
                .slice_mut(s![.., offset..offset + tv.ncols()])
                .assign(&tv);
            offset += tv.ncols();
        }
        if static_covariates.ncols() > 0 {
            design
                .slice_mut(s![.., offset..offset + static_covariates.ncols()])
                .assign(&static_covariates);
        }
        design
    }

    pub fn n_parameters(&self) -> usize {
        self.design_entry.ncols()
    }

    pub fn n_observations(&self) -> usize {
        self.design_entry.nrows()
    }

    pub fn beta_slices<'a>(
        &self,
        beta: &'a Array1<f64>,
    ) -> (
        ArrayView1<'a, f64>,
        Option<ArrayView1<'a, f64>>,
        ArrayView1<'a, f64>,
    ) {
        let baseline = beta.slice(s![self.column_ranges.baseline.clone()]);
        let time_varying = self
            .column_ranges
            .time_varying
            .as_ref()
            .map(|range| beta.slice(s![range.clone()]));
        let static_covariates = beta.slice(s![self.column_ranges.static_covariates.clone()]);
        (baseline, time_varying, static_covariates)
    }

    pub fn design_entry(&self) -> ArrayView2<f64> {
        self.design_entry.view()
    }

    pub fn design_exit(&self) -> ArrayView2<f64> {
        self.design_exit.view()
    }

    pub fn design_derivative_exit(&self) -> ArrayView2<f64> {
        self.design_derivative_exit.view()
    }
}

fn softplus(x: f64) -> f64 {
    if x > 40.0 {
        x
    } else if x < -40.0 {
        (1.0 + x.exp()).ln()
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

fn barrier_value(strength: f64, derivative: f64) -> f64 {
    strength * softplus(-derivative)
}

fn barrier_gradient(strength: f64, derivative: f64) -> f64 {
    -strength * sigmoid(-derivative)
}

fn barrier_hessian(strength: f64, derivative: f64) -> f64 {
    let sig = sigmoid(-derivative);
    strength * sig * (1.0 - sig)
}

/// Working model implementing the Royston–Parmar likelihood with dense Hessian
/// support.  The implementation favours clarity over premature optimisation so
/// the tests can exercise each building block individually.
pub struct WorkingModelSurvival {
    layout: SurvivalLayout,
    sample_weight: Array1<f64>,
    event_target: Array1<u8>,
    event_competing: Array1<u8>,
    barrier_strength: f64,
}

impl WorkingModelSurvival {
    pub fn new(
        layout: SurvivalLayout,
        sample_weight: Array1<f64>,
        event_target: Array1<u8>,
        event_competing: Array1<u8>,
        barrier_strength: f64,
    ) -> Self {
        let n = layout.n_observations();
        assert_eq!(n, sample_weight.len());
        assert_eq!(n, event_target.len());
        assert_eq!(n, event_competing.len());
        for (&target, &competing) in event_target.iter().zip(event_competing.iter()) {
            assert!(
                target <= 1 && competing <= 1,
                "events must be encoded as 0/1 flags"
            );
            assert!(
                !(target == 1 && competing == 1),
                "target and competing events cannot both be active",
            );
        }
        Self {
            layout,
            sample_weight,
            event_target,
            event_competing,
            barrier_strength,
        }
    }
}

impl WorkingModel for WorkingModelSurvival {
    fn update(&mut self, beta: &Array1<f64>) -> WorkingState {
        assert_eq!(beta.len(), self.layout.n_parameters());
        let n = self.layout.n_observations();
        let design_entry = self.layout.design_entry();
        let design_exit = self.layout.design_exit();
        let design_derivative = self.layout.design_derivative_exit();

        let eta_entry = design_entry.dot(beta);
        let eta_exit = design_exit.dot(beta);
        let eta_derivative_exit = design_derivative.dot(beta);

        let mut gradient = Array1::zeros(beta.len());
        let mut hessian = Array2::zeros((beta.len(), beta.len()));

        let mut log_likelihood = 0.0;
        let mut barrier_total = 0.0;
        let mut barrier_grad = Array1::zeros(beta.len());
        let mut barrier_hess = Array2::zeros((beta.len(), beta.len()));

        for i in 0..n {
            let weight = self.sample_weight[i];
            if weight == 0.0 {
                continue;
            }
            let eta_e = eta_entry[i];
            let eta_x = eta_exit[i];
            let d_eta = eta_derivative_exit[i];
            let mut h_entry = eta_e.exp();
            if !h_entry.is_finite() {
                h_entry = f64::MAX;
            }
            let mut h_exit = eta_x.exp();
            if !h_exit.is_finite() {
                h_exit = f64::MAX;
            }
            let delta_h = h_exit - h_entry;
            let derivative = d_eta;
            let sp = softplus(derivative);
            let sig = sigmoid(derivative);
            let r = if sp == 0.0 { 0.0 } else { sig / sp };
            let dr_dx = if sp == 0.0 {
                0.0
            } else {
                let term = (1.0 - sig) * sp - sig;
                sig * term / (sp * sp)
            };

            let x_entry = design_entry.row(i);
            let x_exit = design_exit.row(i);
            let x_deriv = design_derivative.row(i);

            let x_exit_vec = x_exit.to_owned();
            let x_entry_vec = x_entry.to_owned();
            let x_deriv_vec = x_deriv.to_owned();

            let delta_grad = &x_exit_vec * h_exit - &x_entry_vec * h_entry;
            let delta_hess =
                h_exit * x_exit_vec.outer(&x_exit_vec) + h_entry * x_entry_vec.outer(&x_entry_vec);

            let mut grad_contrib = (-&delta_grad).to_owned();
            let mut hess_contrib = (-delta_hess).to_owned();
            let mut ll = -delta_h;

            let event_target_bool = self.event_target[i] == 1;
            let event_competing_bool = self.event_competing[i] == 1;

            if event_target_bool {
                grad_contrib += &x_exit_vec;
                grad_contrib += &(r * &x_deriv_vec);
                hess_contrib += &(dr_dx * x_deriv_vec.outer(&x_deriv_vec));
                ll += eta_x + sp.ln();
            }
            if !event_target_bool && !event_competing_bool {
                // pure censoring already handled by baseline delta term
            }
            log_likelihood += weight * ll;
            gradient += &(weight * &grad_contrib);
            hessian += &(weight * &hess_contrib);

            // Barrier contributions (treated like smoothing penalties)
            let b_val = barrier_value(self.barrier_strength, derivative);
            let b_grad = barrier_gradient(self.barrier_strength, derivative);
            let b_hess = barrier_hessian(self.barrier_strength, derivative);
            barrier_total += b_val * weight;
            let grad_adjust = &x_deriv_vec * b_grad * weight;
            barrier_grad += &grad_adjust;
            let hess_adjust = x_deriv_vec.outer(&x_deriv_vec) * (b_hess * weight);
            barrier_hess += &hess_adjust;
        }

        gradient += &barrier_grad;
        hessian += &barrier_hess;
        let penalty = self
            .layout
            .penalties
            .apply(beta, &mut gradient, &mut hessian);
        let deviance = -2.0 * (log_likelihood - barrier_total - 0.5 * penalty);

        WorkingState {
            eta: eta_exit,
            gradient,
            hessian,
            deviance,
        }
    }
}

/// Captures the penalised observed information factorisation so delta-method
/// standard errors can be reproduced verbatim at scoring time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LdltFactor {
    pub matrix: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CholeskyFactor {
    pub matrix: Array2<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PermutationMatrix {
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Inertia {
    pub negative: usize,
    pub zero: usize,
    pub positive: usize,
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
    pub degree: usize,
    pub knots: Array1<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CovariateLayout {
    pub column_ranges: ColumnRanges,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PenaltyDescriptor {
    pub blocks: Vec<PenaltyBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

#[derive(Debug, Clone)]
pub struct Covariates {
    pub design_row: Array1<f64>,
    pub derivative_row: Array1<f64>,
}

impl SurvivalModelArtifacts {
    pub fn cumulative_hazard(&self, age: f64, covariates: &Covariates) -> f64 {
        let u = self.age_transform.transform(age);
        let mut eta = covariates.design_row.dot(&self.coefficients);
        eta += u;
        eta.exp()
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
        let cif0 = self.cumulative_incidence(t0, covariates);
        let cif1 = self.cumulative_incidence(t1, covariates);
        let delta = (cif1 - cif0).max(0.0);
        let denom = (1.0 - cif0 - cif_competing_t0).max(1e-12);
        delta / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    fn toy_layout() -> SurvivalLayout {
        let baseline_entry = array![[1.0, 0.5], [1.0, 1.0]];
        let baseline_exit = array![[1.0, 1.0], [1.0, 1.5]];
        let baseline_deriv = array![[0.1, 0.2], [0.1, 0.3]];
        let static_covariates = array![[0.5], [0.2]];
        let penalties = PenaltyBlocks::new(vec![PenaltyBlock {
            range: 0..2,
            matrix: Array2::eye(2),
        }]);
        SurvivalLayout::new(
            baseline_entry,
            baseline_exit,
            baseline_deriv,
            None,
            None,
            None,
            static_covariates,
            AgeTransform::new(40.0, 0.1),
            ReferenceConstraint { z: Array2::eye(2) },
            penalties,
        )
    }

    #[test]
    fn age_transform_monotonic() {
        let transform = AgeTransform::new(40.0, 0.1);
        let age = 50.0;
        let u = transform.transform(age);
        let du = transform.derivative_scale(age);
        assert!(u.is_finite());
        assert!(du > 0.0);
    }

    #[test]
    fn working_model_shapes() {
        let layout = toy_layout();
        let weights = array![1.0, 0.5];
        let events_target = array![1, 0];
        let events_competing = array![0, 0];
        let mut model =
            WorkingModelSurvival::new(layout, weights, events_target, events_competing, 0.1);
        let beta = Array1::ones(3);
        let state = model.update(&beta);
        assert_eq!(state.eta.len(), 2);
        assert_eq!(state.gradient.len(), 3);
        assert_eq!(state.hessian.nrows(), 3);
        assert!(state.deviance.is_finite());
    }

    #[test]
    fn conditional_risk_monotonic() {
        let artifacts = SurvivalModelArtifacts {
            coefficients: array![0.1, 0.2, 0.3],
            age_basis: BasisDescriptor {
                degree: 3,
                knots: array![0.0, 1.0, 2.0],
            },
            time_varying_basis: None,
            static_covariate_layout: CovariateLayout {
                column_ranges: ColumnRanges {
                    baseline: 0..2,
                    time_varying: None,
                    static_covariates: 2..3,
                },
            },
            penalties: PenaltyDescriptor { blocks: Vec::new() },
            age_transform: AgeTransform::new(40.0, 0.1),
            reference_constraint: ReferenceConstraint { z: Array2::eye(2) },
            hessian_factor: None,
        };
        let covariates = Covariates {
            design_row: array![1.0, 0.5, 0.3],
            derivative_row: array![0.1, 0.1, 0.0],
        };
        let risk0 = artifacts.conditional_absolute_risk(50.0, 55.0, &covariates, 0.0);
        let risk1 = artifacts.conditional_absolute_risk(50.0, 60.0, &covariates, 0.0);
        assert!(risk1 >= risk0);
    }

    #[test]
    fn schema_roundtrip() {
        let training = SurvivalTrainingData {
            age_entry: array![45.0, 50.0],
            age_exit: array![50.0, 55.0],
            event_target: array![1, 0],
            event_competing: array![0, 1],
            sample_weight: array![1.0, 2.0],
            pgs: array![0.1, -0.1],
            sex: array![1.0, 0.0],
            pcs: array![[0.01, 0.02], [0.03, 0.04]],
        };
        assert_eq!(training.age_entry.len(), 2);

        let covariates = CovariateViews {
            pgs: training.pgs.view(),
            sex: training.sex.view(),
            pcs: training.pcs.view(),
        };

        let inputs = SurvivalPredictionInputs {
            age_entry: training.age_entry.view(),
            age_exit: training.age_exit.view(),
            event_target: training.event_target.view(),
            event_competing: training.event_competing.view(),
            sample_weight: training.sample_weight.view(),
            covariates,
        };
        assert_eq!(inputs.covariates.pcs.ncols(), 2);
    }
}
