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
    SurvivalBaselineConfig, SurvivalBaselineTarget, SurvivalTimeBasisConfig,
    append_zero_tail_columns, build_survival_marginal_slope_baseline_offsets,
    build_survival_time_basis, build_survival_timewiggle_from_baseline,
    center_survival_time_designs_at_anchor, evaluate_survival_time_basis_row,
    resolved_survival_time_basis_config_from_build,
};
use gam::families::survival_location_scale::{TimeBlockInput, TimeWiggleBlockInput};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use thiserror::Error;

/// Errors surfaced while validating survival inputs in gnomon (i.e. before
/// we hand data to gam). gam emits its own error type for fitting itself.
#[derive(Debug, Error)]
pub enum SurvivalError {
    #[error("age vectors must have at least one element")]
    EmptyAgeVector,
    #[error("age values must be finite")]
    NonFiniteAge,
    #[error("age_entry must be strictly less than age_exit for every subject")]
    InvalidAgeOrder,
    #[error("event indicators must be 0 or 1")]
    InvalidEventFlag,
    #[error("event_target and event_competing indicators must be mutually exclusive")]
    ConflictingEvents,
    #[error("sample weights must be finite and non-negative")]
    InvalidSampleWeight,
    #[error("covariate arrays have inconsistent dimensions")]
    CovariateDimensionMismatch,
    #[error("covariate values must be finite")]
    NonFiniteCovariate,
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

// The marginal-slope fitter requires structural monotonicity, so the time
// basis is an I-spline with smoothing optimized from this starting value.
const SURVIVAL_TIME_SMOOTH_LAMBDA: f64 = 1e-2;

pub struct SurvivalTimeMetadata {
    pub basis: String,
    pub degree: Option<usize>,
    pub knots: Option<Vec<f64>>,
    pub keep_cols: Option<Vec<usize>>,
    pub smooth_lambda: Option<f64>,
    pub anchor: f64,
    pub baseline_scale: f64,
}

/// Build a `TimeBlockInput` from the survival training bundle by delegating
/// to gam's canonical i-spline survival time-basis builder. The marginal-
/// slope fitter requires `structural_monotonicity = true`, which is satisfied
/// by the i-spline basis.
pub fn build_time_block_input(
    bundle: &SurvivalTrainingBundle,
    basis: &crate::calibrate::model::BasisConfig,
) -> Result<(TimeBlockInput, SurvivalTimeMetadata), String> {
    let n = bundle.data.age_entry.len();
    let cfg = SurvivalTimeBasisConfig::ISpline {
        degree: basis.degree,
        knots: Array1::zeros(0),
        keep_cols: Vec::new(),
        smooth_lambda: SURVIVAL_TIME_SMOOTH_LAMBDA,
    };
    let mut build = build_survival_time_basis(
        &bundle.data.age_entry,
        &bundle.data.age_exit,
        cfg,
        Some((
            basis.num_knots,
            SURVIVAL_TIME_SMOOTH_LAMBDA,
        )),
    )?;

    let anchor = bundle
        .data
        .age_entry
        .iter()
        .copied()
        .reduce(f64::min)
        .ok_or("survival training requires at least one row")?;
    let resolved = resolved_survival_time_basis_config_from_build(
        &build.basisname,
        build.degree,
        build.knots.as_ref(),
        build.keep_cols.as_ref(),
        build.smooth_lambda,
    )?;
    let anchor_row = evaluate_survival_time_basis_row(anchor, &resolved)?;
    center_survival_time_designs_at_anchor(
        &mut build.x_entry_time,
        &mut build.x_exit_time,
        &anchor_row,
    )?;
    let baseline_scale: f64 = bundle
        .data
        .age_exit
        .iter()
        .map(|time| time / n as f64)
        .sum();
    if !baseline_scale.is_finite() || baseline_scale <= 0.0 {
        return Err("survival exit ages must have a positive finite mean".into());
    }
    let metadata = SurvivalTimeMetadata {
        basis: build.basisname,
        degree: build.degree,
        knots: build.knots,
        keep_cols: build.keep_cols,
        smooth_lambda: build.smooth_lambda,
        anchor,
        baseline_scale,
    };

    let p_time = build.x_exit_time.ncols();
    // Start on the observed time scale, away from the derivative log-barrier.
    // The same parametric offsets are reconstructed by the prediction engine.
    let baseline = SurvivalBaselineConfig {
        target: SurvivalBaselineTarget::Weibull,
        scale: Some(baseline_scale),
        shape: Some(1.0),
        rate: None,
        makeham: None,
    };
    let (offset_entry, offset_exit, derivative_offset_exit) =
        build_survival_marginal_slope_baseline_offsets(
            &bundle.data.age_entry,
            &bundle.data.age_exit,
            &baseline,
        )?;
    Ok((
        TimeBlockInput {
            design_entry: build.x_entry_time,
            design_exit: build.x_exit_time,
            design_derivative_exit: build.x_derivative_time,
            offset_entry,
            offset_exit,
            derivative_offset_exit,
            structural_monotonicity: true,
            penalties: build.penalties,
            nullspace_dims: build.nullspace_dims,
            initial_log_lambdas: None,
            initial_beta: Some(Array1::<f64>::zeros(p_time)),
        },
        metadata,
    ))
}

/// Build the optional time-varying wiggle block.
///
/// Returns `None` when no time-varying configuration was requested.
/// Otherwise derives the basis from the training offsets and extends every
/// time design and penalty to the same coefficient layout.
pub fn build_time_wiggle_block_input(
    time_block: &mut TimeBlockInput,
    settings: Option<&crate::calibrate::model::SurvivalTimeWiggleConfig>,
) -> Result<Option<TimeWiggleBlockInput>, String> {
    let Some(settings) = settings else { return Ok(None); };
    let config = gam::inference::formula_dsl::LinkWiggleFormulaSpec {
        degree: settings.basis.degree,
        num_internal_knots: settings.basis.num_knots,
        penalty_orders: vec![settings.penalty_order],
        double_penalty: settings.double_penalty,
    };
    let wiggle = build_survival_timewiggle_from_baseline(
        &time_block.offset_entry,
        &time_block.offset_exit,
        &time_block.derivative_offset_exit,
        &config,
    )?;
    let base_cols = time_block.design_exit.ncols();
    let total_cols = base_cols + wiggle.ncols;
    append_zero_tail_columns(
        &mut time_block.design_entry,
        &mut time_block.design_exit,
        &mut time_block.design_derivative_exit,
        wiggle.ncols,
    );
    for penalty in &mut time_block.penalties {
        let mut expanded = Array2::zeros((total_cols, total_cols));
        expanded
            .slice_mut(ndarray::s![..base_cols, ..base_cols])
            .assign(penalty);
        *penalty = expanded;
    }
    for penalty in wiggle.penalties {
        let mut expanded = Array2::zeros((total_cols, total_cols));
        expanded
            .slice_mut(ndarray::s![base_cols.., base_cols..])
            .assign(&penalty);
        time_block.penalties.push(expanded);
    }
    time_block.nullspace_dims.extend(wiggle.nullspace_dims);
    if let Some(initial) = time_block.initial_beta.as_mut() {
        let mut expanded = Array1::zeros(total_cols);
        expanded.slice_mut(ndarray::s![..base_cols]).assign(initial);
        *initial = expanded;
    }
    Ok(Some(TimeWiggleBlockInput {
        knots: wiggle.knots,
        degree: wiggle.degree,
        ncols: wiggle.ncols,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_wiggle_extends_designs_penalties_and_initial_coefficients_together() {
        let n = 16;
        let age_entry = Array1::from_iter((0..n).map(|index| 40.0 + index as f64));
        let age_exit = age_entry.mapv(|age| age + 5.0);
        let data = SurvivalTrainingData {
            age_entry,
            age_exit,
            event_target: Array1::zeros(n),
            event_competing: Array1::zeros(n),
            sample_weight: Array1::ones(n),
            pgs: Array1::zeros(n),
            sex: Array1::zeros(n),
            pcs: Array2::zeros((n, 0)),
            extra_static_covariates: Array2::zeros((n, 0)),
            extra_static_names: Vec::new(),
        };
        let bundle = SurvivalTrainingBundle { data };
        let basis = crate::calibrate::model::BasisConfig { num_knots: 4, degree: 3 };
        let (mut time, _) =
            build_time_block_input(&bundle, &basis).expect("time basis");
        let denser_basis = crate::calibrate::model::BasisConfig { num_knots: 6, degree: 3 };
        let (denser_time, _) = build_time_block_input(&bundle, &denser_basis).expect("denser time basis");
        assert!(denser_time.design_exit.ncols() > time.design_exit.ncols());
        let base_width = time.design_exit.ncols();
        let settings = crate::calibrate::model::SurvivalTimeWiggleConfig {
            basis, penalty_order: 2, double_penalty: true,
        };
        let wiggle = build_time_wiggle_block_input(&mut time, Some(&settings))
            .expect("time wiggle")
            .expect("enabled");
        let width = base_width + wiggle.ncols;
        assert_eq!(time.design_entry.ncols(), width);
        assert_eq!(time.design_exit.ncols(), width);
        assert_eq!(time.design_derivative_exit.ncols(), width);
        assert_eq!(
            time.initial_beta
                .as_ref()
                .expect("initial coefficients")
                .len(),
            width
        );
        assert_eq!(time.penalties.len(), time.nullspace_dims.len());
        assert!(
            time.penalties
                .iter()
                .all(|penalty| penalty.dim() == (width, width))
        );
    }
}
