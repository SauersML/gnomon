//! Survival domain glue.
//!
//! This module is intentionally thin: it owns the gnomon-side spec for a
//! survival fit, plus a small set of validation primitives that
//! `survival_data.rs` depends on, and builders that produce the gam-side
//! `TimeBlockInput` / `TimeWiggleBlockInput` payloads.
//!
//! Heavy lifting (PIRLS, REML, monotonicity, joint link, baseline
//! construction, prediction) lives in gam.

use gam::families::survival_construction::{
    SurvivalTimeBasisConfig, build_survival_time_basis,
};
use gam::families::survival_location_scale::{TimeBlockInput, TimeWiggleBlockInput};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Public, serialisable spec for the survival family.
///
/// Used inside `ModelFamily::Survival(SurvivalSpec)`. Numerical defaults are
/// deliberately small — the gam workflow's marginal-slope solver tunes
/// nuisance hyper-parameters internally.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct SurvivalSpec {
    pub derivative_guard: f64,
    pub use_expected_information: bool,
}

impl Default for SurvivalSpec {
    fn default() -> Self {
        Self {
            derivative_guard: 1e-8,
            use_expected_information: true,
        }
    }
}

/// Errors surfaced while validating survival inputs in gnomon (i.e. before
/// we hand data to gam). gam emits its own error type for fitting itself.
#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("age vectors must have at least one element")]
    EmptyAgeVector,
    #[error("age values must be finite")]
    NonFiniteAge,
    #[error("age transform guard delta must be positive")]
    NonPositiveGuard,
    #[error("age_entry must be strictly less than age_exit for every subject")]
    InvalidAgeOrder,
    #[error("event indicators must be 0 or 1")]
    InvalidEventFlag,
    #[error("event_target and event_competing indicators must be mutually exclusive")]
    ConflictingEvents,
    #[error("sample weights must be finite and non-negative")]
    InvalidSampleWeight,
    #[error("covariate arrays must have inconsistent dimensions")]
    CovariateDimensionMismatch,
    #[error("covariate values must be finite")]
    NonFiniteCovariate,
}

/// Guarded log-age transformation used across training and scoring.
///
/// Kept here (rather than pushed into gam) because gnomon's data loader
/// caches it on `SurvivalTrainingBundle`.
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

    #[inline]
    pub fn transform(&self, age: f64) -> Result<f64, SurvivalError> {
        if !age.is_finite() {
            return Err(SurvivalError::NonFiniteAge);
        }
        let shifted = age - self.minimum_age + self.delta;
        if !shifted.is_finite() || shifted <= 0.0 {
            return Err(SurvivalError::NonFiniteAge);
        }
        Ok(shifted.ln())
    }
}

/// Frequency-weighted survival training data bundle.
///
/// Owned arrays produced by `survival_data::load_survival_training_data`.
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
    pub extra_static_covariates: Array2<f64>,
    pub extra_static_names: Vec<String>,
}

#[derive(Clone)]
pub struct CovariateViews<'a> {
    pub pgs: ArrayView1<'a, f64>,
    pub sex: ArrayView1<'a, f64>,
    pub pcs: ArrayView2<'a, f64>,
    pub static_covariates: ArrayView2<'a, f64>,
}

/// Prediction-time inputs as borrowed views.
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: CovariateViews<'a>,
}

/// Re-export the bundle defined in `survival_data` so callers can grab it
/// from the `survival` module too — matches the spec.
pub use crate::calibrate::survival_data::SurvivalTrainingBundle;

#[allow(clippy::too_many_arguments)]
pub fn validate_survival_inputs(
    age_entry: ArrayView1<f64>,
    age_exit: ArrayView1<f64>,
    event_target: ArrayView1<u8>,
    event_competing: ArrayView1<u8>,
    sample_weight: ArrayView1<f64>,
    pgs: ArrayView1<f64>,
    sex: ArrayView1<f64>,
    pcs: ArrayView2<f64>,
    extra_static: ArrayView2<f64>,
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
        || pcs.nrows() != n
        || extra_static.nrows() != n;
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
        if event_target[i] > 1 || event_competing[i] > 1 {
            return Err(SurvivalError::InvalidEventFlag);
        }
        if event_target[i] == 1 && event_competing[i] == 1 {
            return Err(SurvivalError::ConflictingEvents);
        }
        let w = sample_weight[i];
        if !w.is_finite() || w < 0.0 {
            return Err(SurvivalError::InvalidSampleWeight);
        }
        if !pgs[i].is_finite() || !sex[i].is_finite() {
            return Err(SurvivalError::NonFiniteCovariate);
        }
        for j in 0..pcs.ncols() {
            if !pcs[[i, j]].is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
        for j in 0..extra_static.ncols() {
            if !extra_static[[i, j]].is_finite() {
                return Err(SurvivalError::NonFiniteCovariate);
            }
        }
    }

    Ok(())
}

/// Hard-coded defaults for the survival time-basis. The marginal-slope
/// fitter requires structural monotonicity, so we use an i-spline.
const SURVIVAL_TIME_BASIS_DEGREE: usize = 3;
const SURVIVAL_TIME_NUM_INTERNAL_KNOTS: usize = 8;
const SURVIVAL_TIME_SMOOTH_LAMBDA: f64 = 1e-2;

/// Build a `TimeBlockInput` from the survival training bundle by delegating
/// to gam's canonical i-spline survival time-basis builder. The marginal-
/// slope fitter requires `structural_monotonicity = true`, which is satisfied
/// by the i-spline basis.
pub fn build_time_block_input(
    bundle: &SurvivalTrainingBundle,
    spec: &SurvivalSpec,
) -> Result<TimeBlockInput, String> {
    let n = bundle.data.age_entry.len();
    let cfg = SurvivalTimeBasisConfig::ISpline {
        degree: SURVIVAL_TIME_BASIS_DEGREE,
        knots: Array1::zeros(0),
        keep_cols: Vec::new(),
        smooth_lambda: SURVIVAL_TIME_SMOOTH_LAMBDA,
    };
    let build = build_survival_time_basis(
        &bundle.data.age_entry,
        &bundle.data.age_exit,
        cfg,
        Some((SURVIVAL_TIME_NUM_INTERNAL_KNOTS, SURVIVAL_TIME_SMOOTH_LAMBDA)),
    )?;

    let p_time = build.x_exit_time.ncols();
    let zero_vec = Array1::<f64>::zeros(n);
    // gam's marginal-slope event term needs the derivative-offset to stay
    // strictly above the derivative guard; the i-spline derivative design
    // already enforces non-negative slopes, so a constant guard offset is
    // enough to keep the closed-form q-geometry positive.
    let derivative_offset_exit = Array1::<f64>::from_elem(n, spec.derivative_guard);
    Ok(TimeBlockInput {
        design_entry: build.x_entry_time,
        design_exit: build.x_exit_time,
        design_derivative_exit: build.x_derivative_time,
        offset_entry: zero_vec.clone(),
        offset_exit: zero_vec,
        derivative_offset_exit,
        structural_monotonicity: true,
        penalties: build.penalties,
        nullspace_dims: build.nullspace_dims,
        initial_log_lambdas: None,
        initial_beta: Some(Array1::<f64>::zeros(p_time)),
    })
}

/// Hard-coded defaults for the survival time-wiggle block. Mirrors the
/// degree / knot count chosen by gam's CLI when `--timewiggle` is on.
const SURVIVAL_TIMEWIGGLE_DEGREE: usize = 3;
const SURVIVAL_TIMEWIGGLE_NCOLS: usize = 8;

/// Build the optional time-varying wiggle block.
///
/// Returns `None` when no time-varying configuration was requested.
/// Otherwise returns a `TimeWiggleBlockInput` with hard-coded defaults
/// (8 internal knots, degree 3) on `[0, 1]` — gam re-evaluates the knot
/// support against the working time grid internally.
pub fn build_time_wiggle_block_input(
    enable: bool,
) -> Result<Option<TimeWiggleBlockInput>, String> {
    if !enable {
        return Ok(None);
    }
    let knots = Array1::linspace(0.0, 1.0, SURVIVAL_TIMEWIGGLE_NCOLS);
    Ok(Some(TimeWiggleBlockInput {
        knots,
        degree: SURVIVAL_TIMEWIGGLE_DEGREE,
        ncols: SURVIVAL_TIMEWIGGLE_NCOLS,
    }))
}
