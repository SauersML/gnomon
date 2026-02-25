use crate::calibrate::estimate::EstimationError;
use gam::estimate::FitResult;
use gam::hull::PeeledHull;
use gam::pirls;
use gam::types::LinkFunction;
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Adapter-layer features used by gnomon's calibrator workflow.
#[derive(Debug, Clone)]
pub struct CalibratorFeatures {
    pub pred: Array1<f64>,
    pub se: Array1<f64>,
    pub dist: Array1<f64>,
    pub pred_identity: Array1<f64>,
    pub fisher_weights: Array1<f64>,
}

fn with_hull_distance(
    alo: gam::alo::AloDiagnostics,
    raw_train: ArrayView2<f64>,
    hull_opt: Option<&PeeledHull>,
) -> Result<CalibratorFeatures, EstimationError> {
    let dist = if let Some(hull) = hull_opt {
        hull.signed_distance_many(raw_train)
    } else {
        Array1::zeros(raw_train.nrows())
    };

    if dist.iter().any(|&x| x.is_nan()) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }

    Ok(CalibratorFeatures {
        pred: alo.eta_tilde,
        se: alo.se,
        dist,
        pred_identity: alo.pred_identity,
        fisher_weights: alo.fisher_weights,
    })
}

pub fn compute_alo_features(
    fit: &FitResult,
    y: ArrayView1<f64>,
    raw_train: ArrayView2<f64>,
    hull_opt: Option<&PeeledHull>,
    link: LinkFunction,
) -> Result<CalibratorFeatures, EstimationError> {
    let alo = gam::alo::compute_alo_diagnostics(fit, y, link)?;
    with_hull_distance(alo, raw_train, hull_opt)
}

pub fn compute_alo_features_from_fit(
    fit: &FitResult,
    y: ArrayView1<f64>,
    raw_train: ArrayView2<f64>,
    hull_opt: Option<&PeeledHull>,
    link: LinkFunction,
) -> Result<CalibratorFeatures, EstimationError> {
    compute_alo_features(fit, y, raw_train, hull_opt, link)
}

pub fn compute_alo_features_from_pirls(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
    raw_train: ArrayView2<f64>,
    hull_opt: Option<&PeeledHull>,
    link: LinkFunction,
) -> Result<CalibratorFeatures, EstimationError> {
    let alo = gam::alo::compute_alo_diagnostics_from_pirls(base, y, link)?;
    with_hull_distance(alo, raw_train, hull_opt)
}
