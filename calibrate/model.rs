pub use gam::types::LinkFunction;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasisConfig {
    pub num_knots: usize,
    pub degree: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SmoothConfig {
    pub num_centers: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalComponentConfig {
    pub name: String,
    pub basis_config: SmoothConfig,
    pub range: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurvivalTimeWiggleConfig {
    pub basis: BasisConfig,
    pub penalty_order: usize,
    pub double_penalty: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurvivalModelConfig {
    pub baseline_basis: BasisConfig,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub model_family: ModelFamily,
    pub pgs_basis_config: SmoothConfig,
    pub pc_configs: Vec<PrincipalComponentConfig>,
    pub pgs_range: (f64, f64),
    pub max_iterations: usize,
    pub convergence_tolerance: f64,
    pub reml_max_iterations: usize,
    pub reml_convergence_tolerance: f64,
    #[serde(default)]
    pub survival: Option<SurvivalModelConfig>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_family: ModelFamily::Gam(LinkFunction::Logit),
            pgs_basis_config: SmoothConfig { num_centers: 8 },
            pc_configs: Vec::new(),
            pgs_range: (0.0, 0.0),
            max_iterations: 200,
            convergence_tolerance: 1e-7,
            reml_max_iterations: 50,
            reml_convergence_tolerance: 1e-3,
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
    pub saved: gam::inference::model::FittedModelPayload,
    pub latent_score_model: Option<gam::inference::model::FittedModelPayload>,
}

pub(super) const MODEL_BUNDLE_VERSION: u32 = 1;
pub(super) const LATENT_SCORE_HEADER: &str = "latent_score";

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
    pub cumulative_incidence_entry: Array1<f64>,
    pub cumulative_incidence_exit: Array1<f64>,
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
    saved: &gam::inference::model::FittedModelPayload,
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

pub(super) fn predict_eta_mean(
    payload: &gam::inference::model::FittedModelPayload,
    p: ArrayView1<f64>,
    sex: ArrayView1<f64>,
    pcs: ArrayView2<f64>,
) -> Result<gam::predict::PredictResult, ModelError> {
    let (data, col_map) = build_predict_data(payload, p, sex, pcs)?;
    predict_from_data(payload, &data, &col_map)
}

fn predict_from_data(
    payload: &gam::inference::model::FittedModelPayload,
    data: &Array2<f64>,
    col_map: &HashMap<String, usize>,
) -> Result<gam::predict::PredictResult, ModelError> {
    let model = gam::inference::model::FittedModel::from_payload(payload.clone());
    let n = data.nrows();
    let offset = Array1::<f64>::zeros(n);
    let offset_noise = Array1::<f64>::zeros(n);
    let pred_input = gam::inference::predict_input::build_predict_input_for_model(
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
    predictor
        .predict_plugin_response(&pred_input)
        .map_err(|e| ModelError::Predict(format!("predict_plugin_response failed: {e}")))
}

impl TrainedModel {
    fn validate(&self) -> Result<(), ModelError> {
        if self.format_version != MODEL_BUNDLE_VERSION {
            return Err(ModelError::Serde(format!(
                "unsupported calibration model version {}; expected {MODEL_BUNDLE_VERSION}",
                self.format_version
            )));
        }
        let saved = gam::inference::model::FittedModel::from_payload(self.saved.clone());
        saved
            .validate_for_persistence()
            .map_err(ModelError::Serde)?;
        saved
            .validate_numeric_finiteness()
            .map_err(ModelError::Serde)?;
        match (&self.saved.z_column, &self.latent_score_model) {
            (None, None) => {}
            (Some(name), Some(latent)) if name == LATENT_SCORE_HEADER => {
                let latent = gam::inference::model::FittedModel::from_payload(latent.clone());
                latent
                    .validate_for_persistence()
                    .map_err(ModelError::Serde)?;
                latent
                    .validate_numeric_finiteness()
                    .map_err(ModelError::Serde)?;
            }
            _ => {
                return Err(ModelError::Serde(
                    "calibration model has inconsistent latent-score metadata".into(),
                ));
            }
        }
        Ok(())
    }

    fn prediction_data(
        &self,
        p: ArrayView1<f64>,
        sex: ArrayView1<f64>,
        pcs: ArrayView2<f64>,
    ) -> Result<(Array2<f64>, HashMap<String, usize>), ModelError> {
        let (mut data, mut col_map) = build_predict_data(&self.saved, p, sex, pcs)?;
        if let Some(latent) = &self.latent_score_model {
            let z = predict_eta_mean(latent, p, sex, pcs)?.eta;
            let index = data.ncols();
            data.push_column(z.view())
                .map_err(|error| ModelError::Predict(error.to_string()))?;
            col_map.insert(LATENT_SCORE_HEADER.to_string(), index);
        }
        Ok((data, col_map))
    }

    fn predict_result(
        &self,
        p: ArrayView1<f64>,
        sex: ArrayView1<f64>,
        pcs: ArrayView2<f64>,
    ) -> Result<gam::predict::PredictResult, ModelError> {
        let (data, col_map) = self.prediction_data(p, sex, pcs)?;
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
        let model: Self = serde_json::from_reader(reader)
            .map_err(|error| ModelError::Serde(error.to_string()))?;
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
        let (cov_data, mut col_map) = self.prediction_data(p_new, sex_new, pcs_new)?;
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

        let model = gam::inference::model::FittedModel::from_payload(self.saved.clone());
        let primary_offset = Array1::<f64>::zeros(n);
        let noise_offset = Array1::<f64>::zeros(n);

        // Per-row exit cumulative hazard (one column at age_exit).
        let exit_req = gam::families::survival_predict::SurvivalPredictRequest {
            model: &model,
            data: data.view(),
            col_map: &col_map,
            training_headers: model.payload().training_headers.as_ref(),
            primary_offset: &primary_offset,
            noise_offset: &noise_offset,
            time_grid: None,
        };
        let exit_result = gam::families::survival_predict::predict_survival(exit_req)
            .map_err(ModelError::Predict)?;
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
        let entry_req = gam::families::survival_predict::SurvivalPredictRequest {
            model: &model,
            data: entry_data.view(),
            col_map: &col_map,
            training_headers: model.payload().training_headers.as_ref(),
            primary_offset: &primary_offset,
            noise_offset: &noise_offset,
            time_grid: None,
        };
        let entry_result = gam::families::survival_predict::predict_survival(entry_req)
            .map_err(ModelError::Predict)?;
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
    let cumulative_incidence_entry = cumulative_hazard_entry.mapv(|hazard| -(-hazard).exp_m1());
    let cumulative_incidence_exit = cumulative_hazard_exit.mapv(|hazard| -(-hazard).exp_m1());
    Ok(SurvivalPrediction {
        cumulative_hazard_entry,
        cumulative_hazard_exit,
        cumulative_incidence_entry,
        cumulative_incidence_exit,
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
}
