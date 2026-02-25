use crate::calibrate::data::TrainingData;
use crate::calibrate::estimate::EstimationError;
use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, BasisOptions,
    CenterStrategy, Dense, KnotSource, ThinPlateBasisSpec, build_bspline_basis_1d, build_thin_plate_basis,
    create_basis,
};
use gam::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, BlockwiseFitResult, CustomFamily, FamilyEvaluation,
    KnownLinkWiggle, ParameterBlockSpec, ParameterBlockState, fit_custom_family,
};
use gam::generative::{CustomFamilyGenerative, GenerativeSpec, NoiseModel};
use gam::matrix::DesignMatrix;
use gam::probability::{normal_cdf_approx, normal_pdf};
use gam::types::LinkFunction;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

// Model parameterization note:
// We fit smooth blocks directly for (T, log_sigma, m[, wiggle]) and form
// z = (m - T) / sigma with sigma = exp(log_sigma).
//
// This keeps smoothing penalties aligned with scientifically meaningful fields
// (threshold and log-scale), instead of smoothing transformed channels like
// alpha = -T/sigma and beta = 1/sigma independently.
//
// Independent smoothing in (alpha, beta) induces an implicit penalty with
// beta^2 weighting and mixed Hessian/gradient couplings:
//   P ~ ∫ beta^2 [lambda_alpha |H_T + T B - S|^2 + lambda_beta |B|^2] dx
// where B = (∇log_sigma)(∇log_sigma)^T - H_log_sigma and
// S = (∇T)(∇log_sigma)^T + (∇log_sigma)(∇T)^T.
// That coupling can distort roughness control when sigma varies spatially.
// Using (T, log_sigma) blocks avoids that specific reparameterization artifact.

const BLOCK_T: usize = 0;
const BLOCK_LOG_SIGMA: usize = 1;
const BLOCK_M: usize = 2;
const BLOCK_WIGGLE: usize = 3;

#[derive(Clone)]
pub struct LiabilityWiggleConfig {
    pub degree: usize,
    pub num_internal_knots: usize,
    pub penalty_order: usize,
    pub double_penalty: bool,
}

#[derive(Clone)]
pub struct LiabilityFitConfig {
    pub tps_center_strategy: CenterStrategy,
    pub tps_double_penalty: bool,
    pub pgs_degree: usize,
    pub pgs_num_internal_knots: usize,
    pub pgs_penalty_order: usize,
    pub pgs_double_penalty: bool,
    pub sigma_min: f64,
    pub sigma_max: f64,
    pub wiggle: Option<LiabilityWiggleConfig>,
    pub fit_options: BlockwiseFitOptions,
}

impl Default for LiabilityFitConfig {
    fn default() -> Self {
        Self {
            tps_center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
            tps_double_penalty: true,
            pgs_degree: 3,
            pgs_num_internal_knots: 8,
            pgs_penalty_order: 2,
            pgs_double_penalty: true,
            sigma_min: 1e-3,
            sigma_max: 1e3,
            wiggle: Some(LiabilityWiggleConfig {
                degree: 3,
                num_internal_knots: 6,
                penalty_order: 2,
                double_penalty: true,
            }),
            fit_options: BlockwiseFitOptions::default(),
        }
    }
}

#[derive(Clone)]
pub struct LiabilityModel {
    pub fit: BlockwiseFitResult,
    pub tps_centers_t: Array2<f64>,
    pub tps_centers_log_sigma: Array2<f64>,
    pub m_knots: Array1<f64>,
    pub m_degree: usize,
    pub wiggle_knots: Option<Array1<f64>>,
    pub wiggle_degree: Option<usize>,
    pub sigma_min: f64,
    pub sigma_max: f64,
}

pub struct LiabilityPrediction {
    pub probability: Array1<f64>,
    pub z: Array1<f64>,
    pub threshold: Array1<f64>,
    pub sigma: Array1<f64>,
    pub m: Array1<f64>,
    pub wiggle: Option<Array1<f64>>,
}

#[derive(Clone)]
struct LiabilityFamily {
    y: Array1<f64>,
    weights: Array1<f64>,
    sigma_min: f64,
    sigma_max: f64,
    wiggle_knots: Option<Array1<f64>>,
    wiggle_degree: Option<usize>,
}

fn prepend_column(col: ArrayView1<'_, f64>, rest: &Array2<f64>) -> Array2<f64> {
    let n = col.len();
    debug_assert_eq!(rest.nrows(), n);
    let mut out = Array2::<f64>::zeros((n, rest.ncols() + 1));
    out.column_mut(0).assign(&col);
    out.slice_mut(s![.., 1..]).assign(rest);
    out
}

fn expand_penalties_with_prefix_zeros(
    penalties: &[Array2<f64>],
    total_p: usize,
    prefix_cols: usize,
) -> Vec<Array2<f64>> {
    let mut out = Vec::with_capacity(penalties.len());
    for s_block in penalties {
        let p_block = s_block.nrows();
        let mut s_full = Array2::<f64>::zeros((total_p, total_p));
        s_full
            .slice_mut(s![prefix_cols..prefix_cols + p_block, prefix_cols..prefix_cols + p_block])
            .assign(s_block);
        out.push(s_full);
    }
    out
}

fn build_wiggle_design_from_z(
    z: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, EstimationError> {
    let (basis, _used_knots) = create_basis::<Dense>(
        z,
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
    let design_full = (*basis).clone();
    if design_full.ncols() < 2 {
        return Err(EstimationError::InvalidInput(
            "wiggle basis must have at least two columns".to_string(),
        ));
    }
    Ok(design_full.slice(s![.., 1..]).to_owned())
}

fn default_initial_z_from_training(data: &TrainingData) -> Array1<f64> {
    let n = data.p.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let mean = data.p.iter().copied().sum::<f64>() / n as f64;
    let var = data
        .p
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n.max(1) as f64;
    let sd = var.max(1e-12).sqrt();
    data.p.mapv(|v| (v - mean) / sd)
}

fn build_liability_specs(
    data: &TrainingData,
    cfg: &LiabilityFitConfig,
) -> Result<(Vec<ParameterBlockSpec>, Array2<f64>, Array2<f64>, Array1<f64>, Option<Array1<f64>>), EstimationError> {
    let n = data.y.len();

    let tps_spec = ThinPlateBasisSpec {
        center_strategy: cfg.tps_center_strategy.clone(),
        double_penalty: cfg.tps_double_penalty,
    };

    let tps_t = build_thin_plate_basis(data.pcs.view(), &tps_spec)
        .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
    let tps_sigma = build_thin_plate_basis(data.pcs.view(), &tps_spec)
        .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

    let centers_t = match &tps_t.metadata {
        BasisMetadata::ThinPlate { centers } => centers.clone(),
        _ => {
            return Err(EstimationError::InvalidInput(
                "expected thin-plate metadata for threshold block".to_string(),
            ));
        }
    };
    let centers_sigma = match &tps_sigma.metadata {
        BasisMetadata::ThinPlate { centers } => centers.clone(),
        _ => {
            return Err(EstimationError::InvalidInput(
                "expected thin-plate metadata for sigma block".to_string(),
            ));
        }
    };

    let x_t = prepend_column(data.sex.view(), &tps_t.design);
    let x_sigma = prepend_column(data.sex.view(), &tps_sigma.design);

    let penalties_t = expand_penalties_with_prefix_zeros(&tps_t.penalties, x_t.ncols(), 1);
    let penalties_sigma =
        expand_penalties_with_prefix_zeros(&tps_sigma.penalties, x_sigma.ncols(), 1);

    let p_min = data.p.iter().copied().fold(f64::INFINITY, f64::min);
    let mut p_max = data.p.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !p_min.is_finite() || !p_max.is_finite() {
        return Err(EstimationError::InvalidInput(
            "non-finite PGS values encountered while building liability model".to_string(),
        ));
    }
    if (p_max - p_min).abs() < 1e-12 {
        p_max = p_min + 1e-6;
    }
    let m_spec = BSplineBasisSpec {
        degree: cfg.pgs_degree,
        penalty_order: cfg.pgs_penalty_order,
        knot_spec: BSplineKnotSpec::Generate {
            data_range: (p_min, p_max),
            num_internal_knots: cfg.pgs_num_internal_knots,
        },
        double_penalty: cfg.pgs_double_penalty,
        identifiability: BSplineIdentifiability::default(),
    };
    let m_basis = build_bspline_basis_1d(data.p.view(), &m_spec)
        .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
    let m_knots = match &m_basis.metadata {
        BasisMetadata::BSpline1D { knots } => knots.clone(),
        _ => {
            return Err(EstimationError::InvalidInput(
                "expected B-spline metadata for m block".to_string(),
            ));
        }
    };

    let mut specs = vec![
        ParameterBlockSpec {
            name: "threshold".to_string(),
            design: DesignMatrix::Dense(x_t),
            offset: Array1::zeros(n),
            penalties: penalties_t,
            initial_log_lambdas: Array1::zeros(tps_t.penalties.len()),
            initial_beta: None,
        },
        ParameterBlockSpec {
            name: "log_sigma".to_string(),
            design: DesignMatrix::Dense(x_sigma),
            offset: Array1::zeros(n),
            penalties: penalties_sigma,
            initial_log_lambdas: Array1::zeros(tps_sigma.penalties.len()),
            initial_beta: None,
        },
        ParameterBlockSpec {
            name: "score_transform".to_string(),
            design: DesignMatrix::Dense(m_basis.design.clone()),
            offset: Array1::zeros(n),
            penalties: m_basis.penalties.clone(),
            initial_log_lambdas: Array1::zeros(m_basis.penalties.len()),
            initial_beta: None,
        },
    ];

    let wiggle_knots = if let Some(wcfg) = &cfg.wiggle {
        let z_ref = default_initial_z_from_training(data);
        let z_min = z_ref.iter().copied().fold(f64::INFINITY, f64::min);
        let mut z_max = z_ref.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !z_min.is_finite() || !z_max.is_finite() {
            return Err(EstimationError::InvalidInput(
                "non-finite initial Z while constructing wiggle block".to_string(),
            ));
        }
        if (z_max - z_min).abs() < 1e-12 {
            z_max = z_min + 1e-6;
        }
        let (basis_seed, wiggle_knots) = create_basis::<Dense>(
            z_ref.view(),
            KnotSource::Generate {
                data_range: (z_min, z_max),
                num_internal_knots: wcfg.num_internal_knots,
            },
            wcfg.degree,
            BasisOptions::value(),
        )
        .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;

        let seed = (*basis_seed).clone();
        if seed.ncols() < 2 {
            return Err(EstimationError::InvalidInput(
                "wiggle basis has fewer than two columns".to_string(),
            ));
        }
        let x_wiggle = seed.slice(s![.., 1..]).to_owned();
        let mut wiggle_penalties = vec![gam::basis::create_difference_penalty_matrix(
            x_wiggle.ncols(),
            wcfg.penalty_order,
            None,
        )
        .map_err(|e| EstimationError::InvalidInput(e.to_string()))?];
        if wcfg.double_penalty {
            wiggle_penalties.push(Array2::<f64>::eye(x_wiggle.ncols()));
        }
        specs.push(ParameterBlockSpec {
            name: "wiggle".to_string(),
            design: DesignMatrix::Dense(x_wiggle),
            offset: Array1::zeros(n),
            penalties: wiggle_penalties.clone(),
            initial_log_lambdas: Array1::zeros(wiggle_penalties.len()),
            initial_beta: Some(Array1::zeros(seed.ncols() - 1)),
        });

        Some(wiggle_knots)
    } else {
        None
    };

    Ok((specs, centers_t, centers_sigma, m_knots, wiggle_knots))
}

impl CustomFamily for LiabilityFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n_blocks = block_states.len();
        if n_blocks != 3 && n_blocks != 4 {
            return Err(format!(
                "liability family expects 3 or 4 blocks, got {n_blocks}"
            ));
        }
        let n = self.y.len();
        if block_states.iter().any(|b| b.eta.len() != n || b.beta.iter().any(|v| !v.is_finite())) {
            return Err("non-finite or dimension-mismatched block state".to_string());
        }

        let eta_t = &block_states[BLOCK_T].eta;
        let eta_log_sigma = &block_states[BLOCK_LOG_SIGMA].eta;
        let eta_m = &block_states[BLOCK_M].eta;
        let eta_w = if n_blocks == 4 {
            Some(&block_states[BLOCK_WIGGLE].eta)
        } else {
            None
        };

        let mut sigma = Array1::<f64>::zeros(n);
        let mut dsigma_deta_log_sigma = Array1::<f64>::zeros(n);
        let mut z = Array1::<f64>::zeros(n);
        let mut q = Array1::<f64>::zeros(n);
        let mut mu = Array1::<f64>::zeros(n);
        let mut dmu_dq = Array1::<f64>::zeros(n);

        let mut log_lik = 0.0_f64;
        for i in 0..n {
            // sigma = exp(eta_log_sigma), hard-clamped to protect the ratio map
            // z = (m - T) / sigma from blow-ups near sigma -> 0.
            let raw_sigma = eta_log_sigma[i].exp();
            sigma[i] = raw_sigma.clamp(self.sigma_min, self.sigma_max);
            // d sigma / d eta_log_sigma = sigma in the unclamped interior; 0 once clamped.
            // This yields piecewise-smooth optimization around clamp boundaries.
            dsigma_deta_log_sigma[i] = if raw_sigma >= self.sigma_min && raw_sigma <= self.sigma_max
            {
                raw_sigma
            } else {
                0.0
            };
            // Probit latent margin:
            //   q = z + wiggle(z), with z = (m - T) / sigma.
            z[i] = (eta_m[i] - eta_t[i]) / sigma[i];
            q[i] = z[i] + eta_w.map_or(0.0, |ew| ew[i]);
            let p = normal_cdf_approx(q[i]).clamp(1e-10, 1.0 - 1e-10);
            mu[i] = p;
            dmu_dq[i] = normal_pdf(q[i]).max(1e-8);
            log_lik += self.weights[i]
                * (self.y[i] * p.ln() + (1.0_f64 - self.y[i]) * (1.0_f64 - p).ln());
        }

        let mut working_sets = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            let mut wz = Array1::<f64>::zeros(n);
            let mut ww = Array1::<f64>::zeros(n);
            let mut grad = Array1::<f64>::zeros(n);

            for i in 0..n {
                let chain = match b {
                    // q = (m - T)/sigma + w
                    // dq/deta_T = -1/sigma
                    BLOCK_T => -1.0 / sigma[i],
                    BLOCK_LOG_SIGMA => {
                        let sigma_i = sigma[i].max(1e-12);
                        // z = (m - T)/sigma, sigma = exp(eta_log_sigma)
                        // dz/deta_log_sigma = -(m - T)/sigma^2 * d sigma/deta
                        //                  = -z * (d sigma/deta)/sigma
                        // and dq/deta_log_sigma = dz/deta_log_sigma.
                        -z[i] * dsigma_deta_log_sigma[i] / sigma_i
                    }
                    // dq/deta_m = +1/sigma
                    BLOCK_M => 1.0 / sigma[i],
                    // Optional additive nonlinearity in q.
                    BLOCK_WIGGLE => 1.0,
                    _ => return Err("invalid block index".to_string()),
                };
                let var = (mu[i] * (1.0 - mu[i])).max(1e-10);
                let dmu_deta = dmu_dq[i] * chain;
                let dmu_deta_abs = dmu_deta.abs().max(1e-8);
                let denom = if dmu_deta >= 0.0 {
                    dmu_deta_abs
                } else {
                    -dmu_deta_abs
                };

                // IRLS approximation for each block:
                //   w_i = (dmu/deta)^2 / Var(y_i), z_i = eta_i + (y_i - mu_i)/(dmu/deta).
                ww[i] = self.weights[i] * (dmu_deta * dmu_deta / var).max(1e-12);
                wz[i] = block_states[b].eta[i] + (self.y[i] - mu[i]) / denom;
                // Score in eta coordinates for blockwise Newton updates.
                grad[i] = self.weights[i] * (self.y[i] - mu[i]) * dmu_deta / var;
            }

            working_sets.push(BlockWorkingSet {
                working_response: wz,
                working_weights: ww,
                gradient_eta: Some(grad),
            });
        }

        Ok(FamilyEvaluation {
            log_likelihood: log_lik,
            block_working_sets: working_sets,
        })
    }

    fn known_link_wiggle(&self) -> Option<KnownLinkWiggle> {
        if self.wiggle_knots.is_some() {
            Some(KnownLinkWiggle {
                base_link: LinkFunction::Probit,
                wiggle_block: Some(BLOCK_WIGGLE),
            })
        } else {
            None
        }
    }

    fn block_geometry(
        &self,
        block_index: usize,
        block_states: &[ParameterBlockState],
        spec: &ParameterBlockSpec,
    ) -> Result<(DesignMatrix, Array1<f64>), String> {
        if block_index != BLOCK_WIGGLE {
            return Ok((spec.design.clone(), spec.offset.clone()));
        }
        let knots = self
            .wiggle_knots
            .as_ref()
            .ok_or_else(|| "wiggle geometry requested without wiggle knots".to_string())?;
        let degree = self
            .wiggle_degree
            .ok_or_else(|| "wiggle geometry requested without wiggle degree".to_string())?;

        if block_states.len() < 3 {
            return Err("wiggle geometry needs threshold/log_sigma/m blocks".to_string());
        }

        let eta_t = &block_states[BLOCK_T].eta;
        let eta_log_sigma = &block_states[BLOCK_LOG_SIGMA].eta;
        let eta_m = &block_states[BLOCK_M].eta;
        let n = eta_t.len();

        let mut z = Array1::<f64>::zeros(n);
        for i in 0..n {
            let sigma = eta_log_sigma[i].exp().clamp(self.sigma_min, self.sigma_max);
            z[i] = (eta_m[i] - eta_t[i]) / sigma;
        }

        // The wiggle basis is re-evaluated on current z each iteration.
        // This keeps the optional link-flexibility term aligned to the current
        // latent axis rather than a fixed preprocessing transform.
        let x = build_wiggle_design_from_z(z.view(), knots, degree)
            .map_err(|e| format!("failed to build dynamic wiggle design: {e}"))?;
        if x.ncols() != spec.design.ncols() {
            return Err(format!(
                "dynamic wiggle design column mismatch: got {}, expected {}",
                x.ncols(),
                spec.design.ncols()
            ));
        }

        Ok((DesignMatrix::Dense(x), Array1::zeros(n)))
    }

    fn post_update_beta(
        &self,
        block_index: usize,
        beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        if block_index != BLOCK_M {
            return Ok(beta);
        }
        let mut projected = beta;
        // Isotonic projection in coefficient space to preserve a monotone
        // score-transform channel (nondecreasing spline coefficients).
        for j in 1..projected.len() {
            if projected[j] < projected[j - 1] {
                projected[j] = projected[j - 1];
            }
        }
        Ok(projected)
    }
}

impl CustomFamilyGenerative for LiabilityFamily {
    fn generative_spec(&self, block_states: &[ParameterBlockState]) -> Result<GenerativeSpec, String> {
        if block_states.len() != 3 && block_states.len() != 4 {
            return Err(format!(
                "liability generative path expects 3 or 4 blocks, got {}",
                block_states.len()
            ));
        }
        let n = self.y.len();
        let eta_t = &block_states[BLOCK_T].eta;
        let eta_log_sigma = &block_states[BLOCK_LOG_SIGMA].eta;
        let eta_m = &block_states[BLOCK_M].eta;
        let eta_w = if block_states.len() == 4 {
            Some(&block_states[BLOCK_WIGGLE].eta)
        } else {
            None
        };
        if eta_t.len() != n || eta_log_sigma.len() != n || eta_m.len() != n {
            return Err("liability generative dimensions mismatch".to_string());
        }
        if let Some(w) = eta_w {
            if w.len() != n {
                return Err("liability wiggle generative dimensions mismatch".to_string());
            }
        }

        let mut mean = Array1::<f64>::zeros(n);
        for i in 0..n {
            let sigma = eta_log_sigma[i].exp().clamp(self.sigma_min, self.sigma_max);
            let z = (eta_m[i] - eta_t[i]) / sigma;
            let q = z + eta_w.map_or(0.0, |w| w[i]);
            mean[i] = normal_cdf_approx(q).clamp(1e-12, 1.0 - 1e-12);
        }
        Ok(GenerativeSpec {
            mean,
            noise: NoiseModel::Bernoulli,
        })
    }
}

pub fn fit_liability_model(
    data: &TrainingData,
    cfg: &LiabilityFitConfig,
) -> Result<LiabilityModel, EstimationError> {
    let (specs, centers_t, centers_sigma, m_knots, wiggle_knots) =
        build_liability_specs(data, cfg)?;

    let family = LiabilityFamily {
        y: data.y.clone(),
        weights: data.weights.clone(),
        sigma_min: cfg.sigma_min,
        sigma_max: cfg.sigma_max,
        wiggle_knots: wiggle_knots.clone(),
        wiggle_degree: cfg.wiggle.as_ref().map(|w| w.degree),
    };

    let fit = fit_custom_family(&family, &specs, &cfg.fit_options)
        .map_err(EstimationError::InvalidInput)?;

    Ok(LiabilityModel {
        fit,
        tps_centers_t: centers_t,
        tps_centers_log_sigma: centers_sigma,
        m_knots,
        m_degree: cfg.pgs_degree,
        wiggle_knots,
        wiggle_degree: cfg.wiggle.as_ref().map(|w| w.degree),
        sigma_min: cfg.sigma_min,
        sigma_max: cfg.sigma_max,
    })
}

fn build_tps_design_from_centers(
    pcs: ArrayView2<'_, f64>,
    centers: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let tps = build_thin_plate_basis(
        pcs,
        &ThinPlateBasisSpec {
            center_strategy: CenterStrategy::UserProvided(centers.clone()),
            double_penalty: false,
        },
    )
    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
    Ok(tps.design)
}

fn build_m_design_from_knots(
    pgs: ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, EstimationError> {
    let (basis, _) = create_basis::<Dense>(
        pgs,
        KnotSource::Provided(knots.view()),
        degree,
        BasisOptions::value(),
    )
    .map_err(|e| EstimationError::InvalidInput(e.to_string()))?;
    Ok((*basis).clone())
}

pub fn predict_liability_model(
    model: &LiabilityModel,
    p: ArrayView1<'_, f64>,
    sex: ArrayView1<'_, f64>,
    pcs: ArrayView2<'_, f64>,
) -> Result<LiabilityPrediction, EstimationError> {
    let n = p.len();
    if sex.len() != n || pcs.nrows() != n {
        return Err(EstimationError::InvalidInput(
            "predict_liability_model input dimensions mismatch".to_string(),
        ));
    }
    if model.fit.block_states.len() < 3 {
        return Err(EstimationError::InvalidInput(
            "liability model is missing required blocks".to_string(),
        ));
    }

    let x_tps_t = build_tps_design_from_centers(pcs, &model.tps_centers_t)?;
    let x_t = prepend_column(sex, &x_tps_t);
    let x_tps_sigma = build_tps_design_from_centers(pcs, &model.tps_centers_log_sigma)?;
    let x_sigma = prepend_column(sex, &x_tps_sigma);
    let x_m = build_m_design_from_knots(p, &model.m_knots, model.m_degree)?;

    let eta_t = x_t.dot(&model.fit.block_states[BLOCK_T].beta);
    let eta_log_sigma = x_sigma.dot(&model.fit.block_states[BLOCK_LOG_SIGMA].beta);
    let eta_m = x_m.dot(&model.fit.block_states[BLOCK_M].beta);

    // Same forward map as training:
    // sigma = exp(log_sigma), z = (m - T)/sigma, optional q = z + wiggle(z).
    let mut sigma = eta_log_sigma.mapv(f64::exp);
    sigma.mapv_inplace(|v| v.clamp(model.sigma_min, model.sigma_max));
    let z = (&eta_m - &eta_t) / &sigma;

    let wiggle = if model.fit.block_states.len() > 3 {
        let knots = model
            .wiggle_knots
            .as_ref()
            .ok_or_else(|| EstimationError::InvalidInput("missing wiggle knots".to_string()))?;
        let degree = model
            .wiggle_degree
            .ok_or_else(|| EstimationError::InvalidInput("missing wiggle degree".to_string()))?;
        let x_w = build_wiggle_design_from_z(z.view(), knots, degree)?;
        Some(x_w.dot(&model.fit.block_states[BLOCK_WIGGLE].beta))
    } else {
        None
    };

    let probability = match &wiggle {
        Some(w) => (&z + w).mapv(normal_cdf_approx),
        None => z.mapv(normal_cdf_approx),
    }
    .mapv(|v| v.clamp(1e-10, 1.0 - 1e-10));

    Ok(LiabilityPrediction {
        probability,
        z,
        threshold: eta_t,
        sigma,
        m: eta_m,
        wiggle,
    })
}

pub fn fit_liability_from_training_data(
    data: &TrainingData,
) -> Result<LiabilityModel, EstimationError> {
    fit_liability_model(data, &LiabilityFitConfig::default())
}

pub fn liability_block_names(model: &LiabilityModel) -> Vec<&'static str> {
    if model.fit.block_states.len() > 3 {
        vec!["T", "log_sigma", "m", "wiggle"]
    } else {
        vec!["T", "log_sigma", "m"]
    }
}
