pub use gam::types::LinkFunction;
use gam::families::survival::{
    SurvivalPredictEstimand, SurvivalPredictRequest, SurvivalPredictionCovarianceMode,
    predict_survival,
};
use gam::inference::model::{FittedModel, FittedModelPayload};
use gam::predict::FittedModelPredictExt;

use crate::calibrate::runtime::on_gam_pool;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothConfig {
    pub num_centers: usize,
}

/// The ancestry dependence of the context and of the score's slope: one joint
/// Duchon smooth over `PC1..PCk` in each, never a smooth per component.
///
/// The kernel is gam's scale-free structural default (no length scale, affine
/// null space, spectral power `s = (k − 1)/2`, the kernel `r³` in every
/// dimension), which meets both of gam's conditions on a pure Duchon kernel,
/// `2s < k` and `2(p + s) > k + 2` (`p = 2` for the affine null space), at
/// every `k`. `power` overrides `s` and is held to the same conditions.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PcSmoothConfig {
    /// `k`, the number of leading principal components; 0 means none.
    pub num_pcs: usize,
    /// Centers of the joint smooth in the context (marginal) formula.
    pub context_centers: usize,
    /// Centers of the joint smooth in the slope formula.
    pub slope_centers: usize,
    /// An explicit Duchon spectral power, or `None` for gam's default.
    pub power: Option<f64>,
}

impl PcSmoothConfig {
    /// The joint smooth over `k` components with center counts derived from
    /// `k`: `⌈3k/2⌉` in the context and `⌈5k/4⌉` in the slope, the 24 and 20
    /// that the AoU study uses at `k = 16`, scaled linearly in `k`, and never
    /// fewer than `k + 2`, one more than the `k + 1` columns of the affine null
    /// space, which gam requires the centers to exceed.
    pub fn for_pcs(num_pcs: usize) -> Self {
        let floor = num_pcs + 2;
        Self {
            num_pcs,
            context_centers: (3 * num_pcs).div_ceil(2).max(floor),
            slope_centers: (5 * num_pcs).div_ceil(4).max(floor),
            power: None,
        }
    }

    /// Refuses, before any fit, a center count gam would refuse for `k`
    /// components and an explicit power gam would refuse or override: the pure
    /// Duchon kernel needs `2s < k` (conditional positive definiteness on the
    /// null-space complement) and `2(p + s) > k + 2` with `p = 2` (existence at
    /// the second-derivative operator of gam's default penalty; below it gam
    /// raises the null space instead). gam's default `s = (k - 1)/2` meets both.
    /// The center counts are explicit because gam's default count grows like
    /// `n^0.4` in high dimension (about 1,970 at `n = 50,000`, `k = 16`;
    /// gam#2993); `for_pcs` is an interim rule until the AoU study's center
    /// sweep at `k = 16` replaces it, and until gam#2993 fixes the default.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_pcs == 0 {
            return Ok(());
        }
        let null_space = self.num_pcs + 1;
        for (formula, centers) in [
            ("context", self.context_centers),
            ("slope", self.slope_centers),
        ] {
            if centers <= null_space {
                return Err(format!(
                    "the joint PC smooth of the {formula} has {centers} centers for {} PCs; it needs \
                     more than the {null_space} columns of its affine null space",
                    self.num_pcs
                ));
            }
        }
        let Some(power) = self.power else {
            return Ok(());
        };
        if !(power.is_finite() && power >= 0.0) {
            return Err(format!(
                "the joint PC smooth's power must be a finite non-negative number, not {power}"
            ));
        }
        let dimension = self.num_pcs as f64;
        if 2.0 * power >= dimension {
            return Err(format!(
                "the joint PC smooth's power {power} is not below half its dimension {}; the pure \
                 Duchon kernel needs 2s < k. Omit the power to use gam's default s = (k - 1)/2",
                self.num_pcs
            ));
        }
        if 2.0 * (2.0 + power) <= dimension + 2.0 {
            return Err(format!(
                "the joint PC smooth's power {power} leaves 2(p + s) = {} at or below its dimension \
                 plus the penalty's derivative order, {} (p = 2 for the affine null space); gam \
                 would raise the null space. Omit the power to use gam's default s = (k - 1)/2",
                2.0 * (2.0 + power),
                self.num_pcs + 2
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// gam's `timewiggle()` on the survival baseline; every `None` keeps gam's default.
pub struct SurvivalTimeWiggleConfig {
    pub num_knots: Option<usize>,
    pub degree: Option<usize>,
    pub penalty_order: Option<usize>,
    pub double_penalty: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurvivalModelConfig {
    /// Internal knots and degree of the baseline's I-spline time basis; `None`
    /// keeps gam's default.
    pub baseline_knots: Option<usize>,
    pub baseline_degree: Option<usize>,
    pub time_wiggle: Option<SurvivalTimeWiggleConfig>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurvivalRiskType {
    Net,
    Crude,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Gam(LinkFunction),
    Survival,
}

/// The law of the score that a marginal-slope fit anchors its marginal index
/// on. The score is never transformed to reach either law.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LatentLaw {
    /// The weighted empirical law of the training rows' scores: the same rows,
    /// weights and eligibility the outcome model is fitted on. It is pooled
    /// over the context; its adequacy within a context stratum is a
    /// diagnostic, not an assumption.
    #[default]
    Empirical,
    /// An explicit declaration that the score is standard normal given the
    /// context, for a score already on that scale (for example a reference
    /// transform's output).
    StandardNormal,
}

impl LatentLaw {
    /// gam's `latent_measure` name for this law.
    pub fn latent_measure(self) -> &'static str {
        match self {
            Self::Empirical => "global-empirical",
            Self::StandardNormal => "standard-normal",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub model_family: ModelFamily,
    pub pgs_basis_config: SmoothConfig,
    pub pcs: PcSmoothConfig,
    pub pgs_range: (f64, f64),
    pub latent_law: LatentLaw,
    #[serde(default)]
    pub survival: Option<SurvivalModelConfig>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_family: ModelFamily::Gam(LinkFunction::Probit),
            pgs_basis_config: SmoothConfig { num_centers: 8 },
            pcs: PcSmoothConfig::for_pcs(0),
            pgs_range: (0.0, 0.0),
            latent_law: LatentLaw::default(),
            survival: None,
        }
    }
}

// `FittedModelPayload` does not implement Debug, so neither can TrainedModel.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainedModel {
    pub(super) format_version: u32,
    pub config: ModelConfig,
    pub saved: FittedModelPayload,
}

/// Version 3 smooths the PCs jointly. Version 2 fitted one smooth per PC; such a
/// bundle is refused by name at load, never read.
pub(super) const MODEL_BUNDLE_VERSION: u32 = 3;
const PER_PC_BUNDLE_VERSION: u32 = 2;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictDetailed {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
    pub signed_dist: Option<Array1<f64>>,
    pub se_eta: Option<Array1<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalPrediction {
    pub cumulative_hazard_entry: Array1<f64>,
    pub cumulative_hazard_exit: Array1<f64>,
    /// `1 − exp(−H)` of the target event's cause-specific hazard at entry and
    /// exit: net risk, the risk were the competing event censoring independent
    /// of it, not the cumulative incidence the competing event lowers (#2384).
    pub net_risk_entry: Array1<f64>,
    pub net_risk_exit: Array1<f64>,
    pub conditional_risk: Array1<f64>,
    pub logit_risk: Array1<f64>,
    pub logit_risk_se: Option<Array1<f64>>,
    pub logit_risk_design: Option<Array2<f64>>,
}

#[derive(Debug)]
pub enum ModelError {
    Io(std::io::Error),
    Serde(String),
    Predict(String),
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "io error: {e}"),
            ModelError::Serde(s) => write!(f, "serde error: {s}"),
            ModelError::Predict(s) => write!(f, "predict error: {s}"),
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

// Default header names used when assembling a predict-time data matrix from
// gnomon's (pgs, sex, pcs) inputs. These match the column names produced by
// `calibrate::data::load_training_data`, which is what the trainer hands to
// gam, so saved `training_headers` are normally a permutation of these.
const PGS_HEADER: &str = "score";
const SEX_HEADER: &str = "sex";
pub(super) fn predictor_headers(num_pcs: usize) -> Vec<String> {
    let mut headers = vec![PGS_HEADER.to_string(), SEX_HEADER.to_string()];
    headers.extend((1..=num_pcs).map(|index| format!("PC{index}")));
    headers
}

// Build a (data, col_map) pair matching the SavedModel's `training_headers`.
//
// The model was trained with a known feature ordering recorded in
// `training_headers`. `build_predict_input_for_model` needs both a 2D
// data matrix and a HashMap<header, col_idx> describing it. We assemble a
// row-major matrix in the same order as `training_headers` (when present),
// pulling from `p`, `sex`, and `pcs` based on header name. Columns the model
// references but we do not own (none in v1) would error out.
fn build_predict_data(
    saved: &FittedModelPayload,
    p: ArrayView1<f64>,
    sex: ArrayView1<f64>,
    pcs: ArrayView2<f64>,
) -> Result<(Array2<f64>, HashMap<String, usize>), ModelError> {
    let n = p.len();
    if sex.len() != n || pcs.nrows() != n {
        return Err(ModelError::Predict(format!(
            "predict input length mismatch: p={}, sex={}, pcs_rows={}",
            n,
            sex.len(),
            pcs.nrows()
        )));
    }

    if p.iter()
        .chain(sex.iter())
        .chain(pcs.iter())
        .any(|value| !value.is_finite())
    {
        return Err(ModelError::Predict(
            "prediction covariates must be finite".into(),
        ));
    }
    let headers = saved.training_headers.as_ref().ok_or_else(|| {
        ModelError::Predict(
            "saved model is missing training feature metadata; refit the model".into(),
        )
    })?;

    let mut data = Array2::<f64>::zeros((n, headers.len()));
    let mut col_map = HashMap::with_capacity(headers.len());
    for (col_idx, name) in headers.iter().enumerate() {
        if col_map.insert(name.clone(), col_idx).is_some() {
            return Err(ModelError::Predict(format!(
                "duplicate training header '{name}'"
            )));
        }
        if name == PGS_HEADER {
            data.column_mut(col_idx).assign(&p);
        } else if name == SEX_HEADER {
            data.column_mut(col_idx).assign(&sex);
        } else if let Some(stripped) = name.strip_prefix("PC") {
            let pc_idx: usize = stripped
                .parse()
                .map_err(|_| ModelError::Predict(format!("unrecognized PC header '{name}'")))?;
            if pc_idx == 0 || pc_idx > pcs.ncols() {
                return Err(ModelError::Predict(format!(
                    "training header '{name}' references PC{pc_idx} but predict data has {} PCs",
                    pcs.ncols()
                )));
            }
            data.column_mut(col_idx).assign(&pcs.column(pc_idx - 1));
        } else {
            return Err(ModelError::Predict(format!(
                "unrecognized training header '{name}'; expected 'score', 'sex', or 'PCk'"
            )));
        }
    }
    Ok((data, col_map))
}

fn predict_from_data(
    payload: &FittedModelPayload,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
) -> Result<gam::predict::PredictResult, ModelError> {
    on_gam_pool(|| predict_from_data_on_pool(payload, data, col_map)).map_err(ModelError::Predict)?
}

fn predict_from_data_on_pool(
    payload: &FittedModelPayload,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
) -> Result<gam::predict::PredictResult, ModelError> {
    with_predictor(payload, data, col_map, |predictor, pred_input| {
        predictor
            .predict_plugin_response(pred_input)
            .map_err(|e| ModelError::Predict(format!("predict_plugin_response failed: {e}")))
    })
}

fn noise_scale_from_data(
    payload: &FittedModelPayload,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
) -> Result<Option<Array1<f64>>, ModelError> {
    on_gam_pool(|| {
        with_predictor(payload, data, col_map, |predictor, pred_input| {
            predictor
                .predict_noise_scale(pred_input)
                .map_err(|e| ModelError::Predict(format!("predict_noise_scale failed: {e}")))
        })
    })
    .map_err(ModelError::Predict)?
}

/// Runs `work` on the saved model's predictor and its prediction input for `data`.
fn with_predictor<R>(
    payload: &FittedModelPayload,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
    work: impl FnOnce(&dyn gam::predict::PredictableModel, &gam::predict::PredictInput) -> Result<R, ModelError>,
) -> Result<R, ModelError> {
    let model = FittedModel::from_payload(payload.clone());
    let n = data.nrows();
    let offset = Array1::<f64>::zeros(n);
    let offset_noise = Array1::<f64>::zeros(n);
    let pred_input = gam::families::inference::predict_input::build_predict_input_for_model(
        &model,
        data.view(),
        col_map,
        model.payload().training_headers.as_ref(),
        &offset,
        &offset_noise,
        false,
    )
    .map_err(ModelError::Predict)?;
    let predictor = model
        .predictor()
        .ok_or_else(|| ModelError::Predict("saved model could not construct a predictor".into()))?;
    work(predictor.as_ref(), &pred_input)
}

impl TrainedModel {
    fn validate(&self) -> Result<(), ModelError> {
        if self.format_version != MODEL_BUNDLE_VERSION {
            return Err(ModelError::Serde(format!(
                "unsupported calibration model version {}; expected {MODEL_BUNDLE_VERSION}",
                self.format_version
            )));
        }
        let saved = FittedModel::from_payload(self.saved.clone());
        saved
            .validate_for_persistence()
            .map_err(|error| ModelError::Serde(error.to_string()))?;
        saved
            .validate_numeric_finiteness()
            .map_err(|error| ModelError::Serde(error.to_string()))?;
        Ok(())
    }

    fn predict_result(
        &self,
        p: ArrayView1<f64>,
        sex: ArrayView1<f64>,
        pcs: ArrayView2<f64>,
    ) -> Result<gam::predict::PredictResult, ModelError> {
        let (data, col_map) = build_predict_data(&self.saved, p, sex, pcs)?;
        predict_from_data(&self.saved, &data, &col_map)
    }

    /// Save the complete inference contract atomically in a single JSON bundle.
    pub fn save(&self, path: &str) -> Result<(), ModelError> {
        use std::io::Write;
        self.validate()?;
        let encoded =
            serde_json::to_vec(self).map_err(|error| ModelError::Serde(error.to_string()))?;
        let pending = format!("{path}.{:016x}.pending", rand::random::<u64>());
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&pending)?;
        let result = (|| {
            file.write_all(&encoded)?;
            file.sync_all()?;
            drop(file);
            std::fs::rename(&pending, Path::new(path))?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&pending);
        }
        result
    }

    pub fn load(path: &str) -> Result<Self, ModelError> {
        let reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let bundle: serde_json::Value = serde_json::from_reader(reader)
            .map_err(|error| ModelError::Serde(error.to_string()))?;
        // The version is read before the rest, so a bundle of another version is
        // refused by name rather than by whichever of its fields fails to parse.
        let version = bundle.get("format_version").and_then(serde_json::Value::as_u64);
        if version == Some(u64::from(PER_PC_BUNDLE_VERSION)) {
            return Err(ModelError::Serde(format!(
                "{path} is a version {PER_PC_BUNDLE_VERSION} calibration bundle, which fits one smooth per \
                 principal component; gnomon no longer fits or predicts that model. Retrain it: \
                 version {MODEL_BUNDLE_VERSION} smooths the PCs jointly."
            )));
        }
        if version != Some(u64::from(MODEL_BUNDLE_VERSION)) {
            return Err(ModelError::Serde(format!(
                "{path} is calibration bundle version {}; this gnomon reads version \
                 {MODEL_BUNDLE_VERSION}",
                version.map_or_else(|| "(none)".to_string(), |version| version.to_string())
            )));
        }
        let model: Self =
            serde_json::from_value(bundle).map_err(|error| ModelError::Serde(error.to_string()))?;
        model.validate()?;
        Ok(model)
    }

    pub fn predict_detailed(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<PredictDetailed, ModelError> {
        let res = self.predict_result(p_new, sex_new, pcs_new)?;
        // SE on eta requires posterior covariance; v1 returns None and leaves
        // the uncertainty pipeline (predict_full_uncertainty / posterior_mean)
        // to a future iteration that wires fit_result_from_saved_model_for_prediction.
        Ok(PredictDetailed {
            eta: res.eta,
            mean: res.mean,
            signed_dist: None,
            se_eta: None,
        })
    }

    /// The Gaussian location-scale model's predicted standard deviation σ(x) of
    /// each row, in the response's units; `None` for a model with no per-row
    /// distribution scale.
    pub fn predict_standard_deviation(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Option<Array1<f64>>, ModelError> {
        let (data, col_map) = build_predict_data(&self.saved, p_new, sex_new, pcs_new)?;
        noise_scale_from_data(&self.saved, &data, &col_map)
    }

    pub fn predict_mean(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        Ok(self.predict_result(p_new, sex_new, pcs_new)?.mean)
    }

    pub fn predict_linear(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        Ok(self.predict_result(p_new, sex_new, pcs_new)?.eta)
    }

    pub fn predict(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        self.predict_mean(p_new, sex_new, pcs_new)
    }

    pub fn predict_survival(
        &self,
        age_entry: ArrayView1<f64>,
        age_exit: ArrayView1<f64>,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
        risk_type: SurvivalRiskType,
    ) -> Result<SurvivalPrediction, ModelError> {
        let n = p_new.len();
        if age_entry.len() != n || age_exit.len() != n || sex_new.len() != n || pcs_new.nrows() != n
        {
            return Err(ModelError::Predict(format!(
                "predict_survival input length mismatch: n={n}",
            )));
        }
        if age_entry
            .iter()
            .zip(age_exit.iter())
            .any(|(&entry, &exit)| !entry.is_finite() || !exit.is_finite() || entry > exit)
        {
            return Err(ModelError::Predict(
                "survival intervals require finite age_entry <= age_exit".into(),
            ));
        }

        // Build a (covariate) data matrix in training-header order, then
        // append entry/exit time columns under whatever names the SavedModel
        // recorded as `survival_entry` / `survival_exit`.
        let (cov_data, mut col_map) = build_predict_data(&self.saved, p_new, sex_new, pcs_new)?;
        let entry_name = self.saved.survival_entry.clone().ok_or_else(|| {
            ModelError::Predict("survival model is missing entry column metadata".into())
        })?;
        let exit_name = self.saved.survival_exit.clone().ok_or_else(|| {
            ModelError::Predict("survival model is missing exit column metadata".into())
        })?;
        let n_cov = cov_data.ncols();
        let mut data = Array2::<f64>::zeros((n, n_cov + 2));
        if n_cov > 0 {
            data.slice_mut(ndarray::s![.., 0..n_cov]).assign(&cov_data);
        }
        data.column_mut(n_cov).assign(&age_entry);
        data.column_mut(n_cov + 1).assign(&age_exit);
        col_map.insert(entry_name, n_cov);
        col_map.insert(exit_name, n_cov + 1);

        // Crude (cause-specific competing-risk) survival prediction would
        // need a companion-mortality model that gam's one-hazard predictor
        // does not surface end-to-end (its CLI rejects `spec=crude` for the
        // same reason). Refuse early with a clear error rather than
        // panicking partway through prediction.
        if matches!(risk_type, SurvivalRiskType::Crude) {
            return Err(ModelError::Predict(
                "crude (competing-risk) survival prediction is not supported by gam's \
                 one-hazard predictor; refit / export a net survival model and combine \
                 cause-specific hazards externally"
                    .to_string(),
            ));
        }

        let model = FittedModel::from_payload(self.saved.clone());
        let primary_offset = Array1::<f64>::zeros(n);
        let noise_offset = Array1::<f64>::zeros(n);

        // Per-row exit cumulative hazard (one column at age_exit). The plug-in
        // estimand keeps entry and exit hazards on one coefficient vector, so
        // their difference is a conditional risk.
        let exit_req = SurvivalPredictRequest {
            model: &model,
            data: data.view(),
            col_map: &col_map,
            training_headers: model.payload().training_headers.as_ref(),
            primary_offset: &primary_offset,
            noise_offset: &noise_offset,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        };
        let exit_result =
            on_gam_pool(|| predict_survival(exit_req, SurvivalPredictionCovarianceMode::Conditional))
                .map_err(ModelError::Predict)?
                .map_err(|error| ModelError::Predict(error.to_string()))?;
        let cumulative_hazard_exit = exit_result.cumulative_hazard.column(0).to_owned();

        // Per-row entry cumulative hazard. gam's predict_survival evaluates
        // every row at `age_exit` (per_row_eval) when `time_grid` is None;
        // to extract the hazard at each row's entry time we swap age_exit
        // for age_entry in the predict matrix and call again. Calling twice
        // is the idiomatic API surface here — gam doesn't expose a per-row
        // grid.
        let mut entry_data = data.clone();
        for i in 0..n {
            entry_data[[i, n_cov + 1]] = age_entry[i];
        }
        let entry_req = SurvivalPredictRequest {
            model: &model,
            data: entry_data.view(),
            col_map: &col_map,
            training_headers: model.payload().training_headers.as_ref(),
            primary_offset: &primary_offset,
            noise_offset: &noise_offset,
            time_grid: None,
            with_uncertainty: false,
            estimand: SurvivalPredictEstimand::Plugin,
        };
        let entry_result =
            on_gam_pool(|| predict_survival(entry_req, SurvivalPredictionCovarianceMode::Conditional))
                .map_err(ModelError::Predict)?
                .map_err(|error| ModelError::Predict(error.to_string()))?;
        let cumulative_hazard_entry = entry_result.cumulative_hazard.column(0).to_owned();

        survival_risks_from_hazards(cumulative_hazard_entry, cumulative_hazard_exit)
    }
}

fn survival_risks_from_hazards(
    cumulative_hazard_entry: Array1<f64>,
    cumulative_hazard_exit: Array1<f64>,
) -> Result<SurvivalPrediction, ModelError> {
    if cumulative_hazard_entry.len() != cumulative_hazard_exit.len() {
        return Err(ModelError::Predict("survival hazard lengths differ".into()));
    }
    let mut conditional_risk = Array1::zeros(cumulative_hazard_entry.len());
    let mut logit_risk = Array1::zeros(cumulative_hazard_entry.len());
    for (index, (&entry, &exit)) in cumulative_hazard_entry
        .iter()
        .zip(cumulative_hazard_exit.iter())
        .enumerate()
    {
        if !entry.is_finite() || !exit.is_finite() || entry < 0.0 || exit < entry {
            return Err(ModelError::Predict(format!(
                "invalid cumulative hazards at row {}: entry={entry}, exit={exit}",
                index + 1
            )));
        }
        let delta = exit - entry;
        // S(exit)/S(entry) = exp(-(H(exit)-H(entry))). Subtracting two
        // rounded CIFs loses small risks and yields 0/0 for large hazards.
        conditional_risk[index] = -(-delta).exp_m1();
        logit_risk[index] = delta + conditional_risk[index].ln();
    }
    let net_risk_entry = cumulative_hazard_entry.mapv(|hazard| -(-hazard).exp_m1());
    let net_risk_exit = cumulative_hazard_exit.mapv(|hazard| -(-hazard).exp_m1());
    Ok(SurvivalPrediction {
        cumulative_hazard_entry,
        cumulative_hazard_exit,
        net_risk_entry,
        net_risk_exit,
        conditional_risk,
        logit_risk,
        logit_risk_se: None,
        logit_risk_design: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn survival_risks_remain_accurate_for_tiny_intervals_and_large_hazards() {
        let result =
            survival_risks_from_hazards(array![0.0, 1000.0, 0.0], array![1e-18, 1001.0, 1000.0])
                .expect("risks");
        assert_eq!(result.conditional_risk[0], 1e-18);
        assert!((result.conditional_risk[1] - 0.6321205588285577).abs() < 1e-15);
        assert!((result.logit_risk[1] - 0.541324854612918).abs() < 1e-14);
        assert_eq!(result.logit_risk[2], 1000.0);
    }

    #[test]
    fn net_risk_is_one_minus_the_survival_of_the_cause_specific_hazard() {
        let result = survival_risks_from_hazards(array![0.0, 0.5, 2.0], array![1e-18, 1.5, 700.0])
            .expect("risks");
        for (hazard, risk) in [(0.0f64, result.net_risk_entry[0]), (0.5, result.net_risk_entry[1])] {
            assert_eq!(risk, -(-hazard).exp_m1());
        }
        assert_eq!(result.net_risk_exit[0], 1e-18);
        assert_eq!(result.net_risk_exit[1], -(-1.5f64).exp_m1());
        assert_eq!(result.net_risk_exit[2], 1.0);
    }

    #[test]
    fn survival_risks_reject_invalid_hazards() {
        for (entry, exit) in [
            (2.0, 1.0),
            (-1.0, 1.0),
            (f64::NAN, 1.0),
            (0.0, f64::INFINITY),
        ] {
            assert!(survival_risks_from_hazards(array![entry], array![exit]).is_err());
        }
    }

    #[test]
    fn a_per_pc_bundle_is_refused_by_name() {
        let directory = tempfile::tempdir().expect("model directory");
        let path = directory.path().join("model.json");
        std::fs::write(
            &path,
            r#"{"format_version": 2, "config": {"pc_configs": [{"name": "PC1"}], "reml_max_iterations": 10}, "saved": {}}"#,
        )
        .expect("write a version 2 bundle");
        let error = match TrainedModel::load(path.to_str().expect("path")) {
            Ok(_) => panic!("a version 2 bundle loaded"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("version 2 calibration bundle"), "{error}");
        assert!(error.contains("one smooth per principal component"), "{error}");
    }
}
