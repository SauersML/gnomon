use crate::calibrate::survival::SurvivalSpec;
pub use gam::types::LinkFunction;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BasisConfig {
    pub num_knots: usize,
    pub degree: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrincipalComponentConfig {
    pub name: String,
    pub basis_config: BasisConfig,
    pub range: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalTimeVaryingConfig {
    #[serde(default)]
    pub label: Option<String>,
    pub pgs_basis: BasisConfig,
    pub pgs_penalty_order: usize,
    pub lambda_age: f64,
    pub lambda_pgs: f64,
    pub lambda_null: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivalModelConfig {
    pub baseline_basis: BasisConfig,
    pub guard_delta: f64,
    pub monotonic_grid_size: usize,
    #[serde(default)]
    pub time_varying: Option<SurvivalTimeVaryingConfig>,
    #[serde(default)]
    pub model_competing_risk: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurvivalRiskType {
    Net,
    Crude,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFamily {
    Gam(LinkFunction),
    Survival(SurvivalSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_family: ModelFamily,
    pub pgs_basis_config: BasisConfig,
    pub pc_configs: Vec<PrincipalComponentConfig>,
    pub pgs_range: (f64, f64),
    pub penalty_order: usize,
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
            pgs_basis_config: BasisConfig {
                num_knots: 8,
                degree: 3,
            },
            pc_configs: Vec::new(),
            pgs_range: (0.0, 0.0),
            penalty_order: 2,
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
pub struct TrainedModel {
    pub config: ModelConfig,
    pub saved: gam::inference::model::FittedModelPayload,
}

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
fn pc_header(idx: usize) -> String {
    format!("PC{}", idx + 1)
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

    // Resolve the desired column order. Prefer training_headers verbatim so
    // the predict design lines up with the fit basis. Fall back to a
    // canonical [score, sex, PC1..PCn] layout when the saved model predates
    // the training_headers field.
    let headers: Vec<String> = match saved.training_headers.as_ref() {
        Some(h) => h.clone(),
        None => {
            let mut out = vec![PGS_HEADER.to_string(), SEX_HEADER.to_string()];
            for j in 0..pcs.ncols() {
                out.push(pc_header(j));
            }
            out
        }
    };

    let mut data = Array2::<f64>::zeros((n, headers.len()));
    let mut col_map = HashMap::with_capacity(headers.len());
    for (col_idx, name) in headers.iter().enumerate() {
        col_map.insert(name.clone(), col_idx);
        if name == PGS_HEADER {
            data.column_mut(col_idx).assign(&p);
        } else if name == SEX_HEADER {
            data.column_mut(col_idx).assign(&sex);
        } else if let Some(stripped) = name.strip_prefix("PC") {
            let pc_idx: usize = stripped.parse().map_err(|_| {
                ModelError::Predict(format!("unrecognized PC header '{name}'"))
            })?;
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

fn predict_eta_mean(
    payload: &gam::inference::model::FittedModelPayload,
    p: ArrayView1<f64>,
    sex: ArrayView1<f64>,
    pcs: ArrayView2<f64>,
) -> Result<gam::predict::PredictResult, ModelError> {
    let (data, col_map) = build_predict_data(payload, p, sex, pcs)?;
    let model = gam::inference::model::FittedModel::from_payload(payload.clone());
    let n = data.nrows();
    let offset = Array1::<f64>::zeros(n);
    let offset_noise = Array1::<f64>::zeros(n);
    let pred_input = gam::inference::predict_input::build_predict_input_for_model(
        &model,
        data.view(),
        &col_map,
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

fn config_sidecar_path(path: &str) -> std::path::PathBuf {
    let p = Path::new(path);
    let mut name = p
        .file_name()
        .map(|s| s.to_os_string())
        .unwrap_or_default();
    name.push(".config.toml");
    p.with_file_name(name)
}

impl TrainedModel {
    // We persist the TrainedModel as a two-file bundle: the saved gam model
    // (JSON via `FittedModel::save_to_path`) at `path`, and the gnomon
    // `ModelConfig` as TOML at `{path}.config.toml`. Two files keeps each
    // half independently serialisable with its native format and avoids
    // wedging the gam JSON schema with an extra wrapper field.
    pub fn save(&self, path: &str) -> Result<(), ModelError> {
        let model = gam::inference::model::FittedModel::from_payload(self.saved.clone());
        model
            .save_to_path(Path::new(path))
            .map_err(ModelError::Serde)?;
        let cfg_path = config_sidecar_path(path);
        let cfg_str = toml::to_string_pretty(&self.config)
            .map_err(|e| ModelError::Serde(format!("toml serialize config: {e}")))?;
        std::fs::write(&cfg_path, cfg_str)?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, ModelError> {
        let model = gam::inference::model::FittedModel::load_from_path(Path::new(path))
            .map_err(ModelError::Serde)?;
        let saved = model.payload().clone();
        let cfg_path = config_sidecar_path(path);
        let cfg_str = std::fs::read_to_string(&cfg_path)?;
        let config: ModelConfig = toml::from_str(&cfg_str)
            .map_err(|e| ModelError::Serde(format!("toml deserialize config: {e}")))?;
        Ok(TrainedModel { config, saved })
    }

    pub fn predict_detailed(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<PredictDetailed, ModelError> {
        let res = predict_eta_mean(&self.saved, p_new, sex_new, pcs_new)?;
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
        Ok(predict_eta_mean(&self.saved, p_new, sex_new, pcs_new)?.mean)
    }

    pub fn predict_linear(
        &self,
        p_new: ArrayView1<f64>,
        sex_new: ArrayView1<f64>,
        pcs_new: ArrayView2<f64>,
    ) -> Result<Array1<f64>, ModelError> {
        Ok(predict_eta_mean(&self.saved, p_new, sex_new, pcs_new)?.eta)
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
        if age_entry.len() != n
            || age_exit.len() != n
            || sex_new.len() != n
            || pcs_new.nrows() != n
        {
            return Err(ModelError::Predict(format!(
                "predict_survival input length mismatch: n={n}",
            )));
        }

        // Build a (covariate) data matrix in training-header order, then
        // append entry/exit time columns under whatever names the SavedModel
        // recorded as `survival_entry` / `survival_exit`.
        let (cov_data, mut col_map) = build_predict_data(&self.saved, p_new, sex_new, pcs_new)?;
        let entry_name = self
            .saved
            .survival_entry
            .clone()
            .unwrap_or_else(|| "age_entry".to_string());
        let exit_name = self
            .saved
            .survival_exit
            .clone()
            .unwrap_or_else(|| "age_exit".to_string());
        let n_cov = cov_data.ncols();
        let mut data = Array2::<f64>::zeros((n, n_cov + 2));
        if n_cov > 0 {
            data.slice_mut(ndarray::s![.., 0..n_cov]).assign(&cov_data);
        }
        data.column_mut(n_cov).assign(&age_entry);
        data.column_mut(n_cov + 1).assign(&age_exit);
        col_map.insert(entry_name, n_cov);
        col_map.insert(exit_name, n_cov + 1);

        let model = gam::inference::model::FittedModel::from_payload(self.saved.clone());
        let primary_offset = Array1::<f64>::zeros(n);
        let noise_offset = Array1::<f64>::zeros(n);
        let req = gam::families::survival_predict::SurvivalPredictRequest {
            model: &model,
            data: data.view(),
            col_map: &col_map,
            training_headers: model.payload().training_headers.as_ref(),
            primary_offset: &primary_offset,
            noise_offset: &noise_offset,
            time_grid: None,
        };
        let result = gam::families::survival_predict::predict_survival(req)
            .map_err(ModelError::Predict)?;

        // gam returns hazard/survival/cum_hazard on a per-row evaluation
        // (one column when time_grid is None). Read column 0 for both entry
        // and exit. NOTE: with time_grid=None gam evaluates only at age_exit
        // per row. For now we approximate cumulative_hazard_entry = 0 (study
        // entry is the natural baseline). A future iteration should pass an
        // explicit time grid covering both entry and exit times if a
        // non-zero entry hazard is required.
        let cumulative_hazard_exit = result.cumulative_hazard.column(0).to_owned();
        let cumulative_hazard_entry = Array1::<f64>::zeros(n);
        let cumulative_incidence_entry =
            cumulative_hazard_entry.mapv(|c| 1.0 - (-c).exp());
        let cumulative_incidence_exit = cumulative_hazard_exit.mapv(|c| 1.0 - (-c).exp());

        // Conditional risk = (CIF_exit - CIF_entry) / (1 - CIF_entry).
        let mut conditional_risk = Array1::<f64>::zeros(n);
        for i in 0..n {
            let denom = 1.0 - cumulative_incidence_entry[i];
            let num = cumulative_incidence_exit[i] - cumulative_incidence_entry[i];
            conditional_risk[i] = if denom > 0.0 { (num / denom).clamp(0.0, 1.0) } else { 0.0 };
        }
        let logit_risk = conditional_risk.mapv(|p| {
            let p = p.clamp(1e-12, 1.0 - 1e-12);
            (p / (1.0 - p)).ln()
        });

        // Crude (cause-specific competing-risk) needs the companion mortality
        // model surfaced via SavedModel — gam's SurvivalPredictResult does
        // not expose that yet. v1: honor Net only.
        match risk_type {
            SurvivalRiskType::Net => {}
            SurvivalRiskType::Crude => {
                todo!(
                    "Crude (competing-risk) survival prediction needs companion-model plumbing \
                     that gam::families::survival_predict does not surface; revisit when gam \
                     SurvivalPredictResult exposes the competing-risk hazards"
                );
            }
        }

        Ok(SurvivalPrediction {
            cumulative_hazard_entry,
            cumulative_hazard_exit,
            cumulative_incidence_entry,
            cumulative_incidence_exit,
            conditional_risk,
            logit_risk,
            // SE + design row require posterior covariance + per-row design
            // assembly that the high-level predict_survival doesn't return.
            // Defer to a follow-up that wires uncertainty through.
            logit_risk_se: None,
            logit_risk_design: None,
        })
    }
}
