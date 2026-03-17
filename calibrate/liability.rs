use crate::calibrate::data::TrainingData;
use crate::calibrate::estimate::EstimationError;
use crate::calibrate::survival_data::SurvivalTrainingBundle;
use gam::basis::{
    BSplineBasisSpec, BSplineIdentifiability, BSplineKnotSpec, BasisMetadata, CenterStrategy,
    DuchonBasisSpec, DuchonNullspaceOrder, build_bspline_basis_1d,
    evaluate_bspline_derivative_scalar,
};
use gam::bernoulli_marginal_slope::{
    BernoulliMarginalSlopeFitResult, BernoulliMarginalSlopeTermSpec, DeviationBlockConfig,
};
use gam::custom_family::BlockwiseFitOptions;
use gam::smooth::{
    ShapeConstraint, SmoothBasisSpec, SmoothTermSpec, SpatialLengthScaleOptimizationOptions,
    TermCollectionSpec,
};
use gam::survival_location_scale::TimeBlockInput;
use gam::survival_marginal_slope::{SurvivalMarginalSlopeFitResult, SurvivalMarginalSlopeTermSpec};
use gam::{
    BernoulliMarginalSlopeFitRequest, FitRequest, FitResult, SurvivalMarginalSlopeFitRequest,
    fit_model,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Configuration for the marginal-slope model.
#[derive(Clone, Debug)]
pub struct MarginalSlopeConfig {
    /// Center strategy for Duchon smooth over PC space.
    pub center_strategy: CenterStrategy,
    /// Duchon spectral power s (controls smoothness order).
    pub duchon_power: usize,
    /// Duchon nullspace order (Zero or Linear).
    pub duchon_nullspace: DuchonNullspaceOrder,
    /// Optional hybrid Duchon length scale (None = pure scale-free Duchon).
    pub duchon_length_scale: Option<f64>,
    /// Optional score warp (learnable monotone transformation of z).
    pub score_warp: Option<DeviationBlockConfig>,
    /// Optional link deviation (bends the probit link).
    pub link_dev: Option<DeviationBlockConfig>,
    /// Gauss-Hermite quadrature points for the marginal integral.
    pub quadrature_points: usize,
    /// Solver options.
    pub fit_options: BlockwiseFitOptions,
    /// Spatial length-scale optimization options.
    pub kappa_options: SpatialLengthScaleOptimizationOptions,
}

impl Default for MarginalSlopeConfig {
    fn default() -> Self {
        Self {
            center_strategy: CenterStrategy::FarthestPoint { num_centers: 24 },
            duchon_power: 2,
            duchon_nullspace: DuchonNullspaceOrder::Linear,
            duchon_length_scale: None,
            score_warp: Some(DeviationBlockConfig::default()),
            link_dev: Some(DeviationBlockConfig::default()),
            quadrature_points: 11,
            fit_options: BlockwiseFitOptions::default(),
            kappa_options: SpatialLengthScaleOptimizationOptions::default(),
        }
    }
}

/// Fitted marginal-slope model artifact.
#[derive(Clone)]
pub struct MarginalSlopeModel {
    pub result: BernoulliMarginalSlopeFitResult,
    pub num_pcs: usize,
}

/// Assemble the n × (1 + k) covariate matrix [sex | PC1 .. PCk].
fn build_covariate_matrix(sex: ArrayView1<f64>, pcs: ArrayView2<f64>) -> Array2<f64> {
    let n = sex.len();
    let k = pcs.ncols();
    let mut data = Array2::<f64>::zeros((n, 1 + k));
    data.column_mut(0).assign(&sex);
    if k > 0 {
        data.slice_mut(ndarray::s![.., 1..]).assign(&pcs);
    }
    data
}

/// Build a TermCollectionSpec with a Duchon smooth over all PC columns
/// plus a linear term for sex.
fn build_surface_spec(
    num_pcs: usize,
    center_strategy: &CenterStrategy,
    duchon_power: usize,
    duchon_nullspace: &DuchonNullspaceOrder,
    duchon_length_scale: Option<f64>,
) -> TermCollectionSpec {
    // Column 0 = sex (linear), columns 1..=num_pcs = PCs (Duchon smooth)
    let sex_linear = gam::smooth::LinearTermSpec {
        name: "sex".to_string(),
        feature_col: 0,
        double_penalty: true,
        coefficient_geometry: gam::smooth::LinearCoefficientGeometry::Unconstrained,
        coefficient_min: None,
        coefficient_max: None,
    };

    let pc_cols: Vec<usize> = (1..=num_pcs).collect();
    let duchon_smooth = SmoothTermSpec {
        name: "ancestry".to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: pc_cols,
            spec: DuchonBasisSpec {
                center_strategy: center_strategy.clone(),
                length_scale: duchon_length_scale,
                power: duchon_power,
                nullspace_order: duchon_nullspace.clone(),
                identifiability: Default::default(),
                aniso_log_scales: None,
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
    };

    TermCollectionSpec {
        linear_terms: vec![sex_linear],
        random_effect_terms: vec![],
        smooth_terms: vec![duchon_smooth],
    }
}

/// Fit a marginal-slope model for binary outcomes.
///
/// The model decomposes P(Y=1 | z, PC) into:
/// - A marginal surface (local prevalence) smooth over ancestry
/// - A log-slope surface (PGS accuracy) smooth over ancestry
/// - Optional score warp h(z) (monotone nonlinear dose-response)
/// - Optional link deviation w(η) (non-probit noise)
///
/// The intercept is chosen so that averaging over scores at each ancestry
/// recovers the local prevalence exactly (decoupling property).
pub fn fit_marginal_slope(
    data: &TrainingData,
    cfg: &MarginalSlopeConfig,
) -> Result<MarginalSlopeModel, EstimationError> {
    let num_pcs = data.pcs.ncols();

    // Standardize z (PGS) to approximately N(0,1)
    let z = standardize_score(&data.p);

    // Build covariate matrix [sex | PCs]
    let covariate_data = build_covariate_matrix(data.sex.view(), data.pcs.view());

    // Build identical surface specs for marginal and log-slope
    let marginalspec = build_surface_spec(
        num_pcs,
        &cfg.center_strategy,
        cfg.duchon_power,
        &cfg.duchon_nullspace,
        cfg.duchon_length_scale,
    );
    let logslopespec = build_surface_spec(
        num_pcs,
        &cfg.center_strategy,
        cfg.duchon_power,
        &cfg.duchon_nullspace,
        cfg.duchon_length_scale,
    );

    let spec = BernoulliMarginalSlopeTermSpec {
        y: data.y.clone(),
        weights: data.weights.clone(),
        z,
        marginalspec,
        logslopespec,
        score_warp: cfg.score_warp.clone(),
        link_dev: cfg.link_dev.clone(),
        quadrature_points: cfg.quadrature_points,
    };

    let request = BernoulliMarginalSlopeFitRequest {
        data: covariate_data.view(),
        spec,
        options: cfg.fit_options.clone(),
        kappa_options: cfg.kappa_options.clone(),
    };

    let fit_result = fit_model(FitRequest::BernoulliMarginalSlope(request))
        .map_err(|e| EstimationError::InvalidSpecification(e))?;

    let result = match fit_result {
        FitResult::BernoulliMarginalSlope(r) => r,
        _ => {
            return Err(EstimationError::InvalidSpecification(
                "unexpected fit result variant".to_string(),
            ));
        }
    };

    Ok(MarginalSlopeModel { result, num_pcs })
}

/// Convenience wrapper using default config.
pub fn fit_marginal_slope_default(
    data: &TrainingData,
) -> Result<MarginalSlopeModel, EstimationError> {
    fit_marginal_slope(data, &MarginalSlopeConfig::default())
}

// ── Survival marginal-slope model ────────────────────────────────────

/// Gompertz-Makeham baseline hazard parameters.
#[derive(Clone, Debug)]
pub struct GompertzMakehamBaseline {
    /// Gompertz hazard rate (> 0).
    pub rate: f64,
    /// Gompertz shape parameter (exponential aging rate).
    pub shape: f64,
    /// Makeham additive constant hazard (>= 0, 0 = pure Gompertz).
    pub makeham: f64,
}

impl Default for GompertzMakehamBaseline {
    fn default() -> Self {
        Self {
            rate: 0.001,
            shape: 0.08,
            makeham: 0.0,
        }
    }
}

/// Configuration for the survival marginal-slope model.
#[derive(Clone, Debug)]
pub struct SurvivalMarginalSlopeConfig {
    /// Baseline hazard parametric anchor.
    pub baseline: GompertzMakehamBaseline,
    /// Number of internal knots for the B-spline time basis.
    pub time_num_knots: usize,
    /// Degree of the B-spline time basis.
    pub time_degree: usize,
    /// Derivative guard (prevents division by zero in hazard derivatives).
    pub derivative_guard: f64,
    /// Duchon smooth config (shared with binary model).
    pub surface: MarginalSlopeConfig,
}

impl Default for SurvivalMarginalSlopeConfig {
    fn default() -> Self {
        Self {
            baseline: GompertzMakehamBaseline::default(),
            time_num_knots: 6,
            time_degree: 3,
            derivative_guard: 1e-8,
            surface: MarginalSlopeConfig {
                // Survival models don't use score_warp/link_dev (binary-only features)
                score_warp: None,
                link_dev: None,
                ..MarginalSlopeConfig::default()
            },
        }
    }
}

/// Fitted survival marginal-slope model artifact.
#[derive(Clone)]
pub struct SurvivalMarginalSlopeModel {
    pub result: SurvivalMarginalSlopeFitResult,
    pub num_pcs: usize,
    pub baseline: GompertzMakehamBaseline,
    pub time_knots: Array1<f64>,
    pub time_degree: usize,
}

const SURVIVAL_TIME_FLOOR: f64 = 1e-9;

/// Evaluate Gompertz-Makeham cumulative hazard and its time derivative.
///
/// Returns (log(H₀(t)), d log(H₀(t))/dt) where H₀(t) = makeham·t + (rate/shape)(e^{shape·t} - 1).
fn evaluate_gompertz_makeham(t: f64, baseline: &GompertzMakehamBaseline) -> (f64, f64) {
    let age = t.max(SURVIVAL_TIME_FLOOR);

    // Gompertz component: H_g(t) = (rate/shape)(e^{shape·t} - 1), h_g(t) = rate·e^{shape·t}
    let (h_gompertz, inst_gompertz) = if baseline.shape.abs() < 1e-10 {
        (baseline.rate * age, baseline.rate)
    } else {
        let shape_age = baseline.shape * age;
        let cumhaz = (baseline.rate / baseline.shape) * shape_age.exp_m1();
        let insthaz = baseline.rate * shape_age.exp();
        (cumhaz, insthaz)
    };

    // Add Makeham: H₀(t) = c·t + H_g(t), h₀(t) = c + h_g(t)
    let h_total = baseline.makeham * age + h_gompertz;
    let inst_total = baseline.makeham + inst_gompertz;

    if h_total <= 0.0 || !h_total.is_finite() {
        // Degenerate — return zero offset
        (0.0, 0.0)
    } else {
        // η(t) = log(H₀(t)), dη/dt = h₀(t)/H₀(t)
        (h_total.ln(), inst_total / h_total)
    }
}

/// Build the time block for survival models: B-spline basis over log(age) with
/// Gompertz-Makeham parametric offsets.
fn build_time_block(
    age_entry: &Array1<f64>,
    age_exit: &Array1<f64>,
    baseline: &GompertzMakehamBaseline,
    num_knots: usize,
    degree: usize,
) -> Result<(TimeBlockInput, Array1<f64>), EstimationError> {
    let n = age_entry.len();
    let log_entry: Array1<f64> = age_entry.mapv(|t| t.max(SURVIVAL_TIME_FLOOR).ln());
    let log_exit: Array1<f64> = age_exit.mapv(|t| t.max(SURVIVAL_TIME_FLOOR).ln());

    // Compute GM baseline offsets
    let mut offset_entry = Array1::<f64>::zeros(n);
    let mut offset_exit = Array1::<f64>::zeros(n);
    let mut derivative_offset_exit = Array1::<f64>::zeros(n);
    for i in 0..n {
        let (eta_e, _) = evaluate_gompertz_makeham(age_entry[i], baseline);
        let (eta_x, deriv_x) = evaluate_gompertz_makeham(age_exit[i], baseline);
        offset_entry[i] = eta_e;
        offset_exit[i] = eta_x;
        derivative_offset_exit[i] = deriv_x;
    }

    // Use exit times for knot placement (entry may be degenerate)
    let knot_input = if (log_entry.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        - log_entry.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
    .abs()
        < 1e-8
    {
        log_exit.clone()
    } else {
        let mut combined = Array1::<f64>::zeros(2 * n);
        for i in 0..n {
            combined[i] = log_entry[i];
            combined[n + i] = log_exit[i];
        }
        combined
    };

    // Build B-spline basis
    let bspec = BSplineBasisSpec {
        degree,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Automatic {
            num_internal_knots: Some(num_knots),
            placement: gam::basis::BSplineKnotPlacement::Quantile,
        },
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
    };

    let built = build_bspline_basis_1d(knot_input.view(), &bspec)
        .map_err(|e| EstimationError::InvalidInput(format!("time basis knot inference: {e}")))?;
    let knots = match &built.metadata {
        BasisMetadata::BSpline1D { knots, .. } => knots.clone(),
        _ => {
            return Err(EstimationError::InvalidInput(
                "expected BSpline1D metadata for time basis".to_string(),
            ));
        }
    };

    // Rebuild at actual entry/exit times using inferred knots
    let provided_spec = BSplineBasisSpec {
        degree,
        penalty_order: 2,
        knotspec: BSplineKnotSpec::Provided(knots.clone()),
        double_penalty: false,
        identifiability: BSplineIdentifiability::None,
    };
    let entry_basis = build_bspline_basis_1d(log_entry.view(), &provided_spec)
        .map_err(|e| EstimationError::InvalidInput(format!("time entry basis: {e}")))?;
    let exit_basis = build_bspline_basis_1d(log_exit.view(), &provided_spec)
        .map_err(|e| EstimationError::InvalidInput(format!("time exit basis: {e}")))?;

    // Derivative of B-spline at exit: d(basis)/d(log_t) * (1/t) via chain rule
    let p_time = exit_basis.design.ncols();
    let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
    let mut deriv_buf = vec![0.0_f64; p_time];
    for i in 0..n {
        deriv_buf.fill(0.0);
        evaluate_bspline_derivative_scalar(log_exit[i], knots.view(), degree, &mut deriv_buf)
            .map_err(|e| EstimationError::InvalidInput(format!("time derivative: {e}")))?;
        let chain = 1.0 / age_exit[i].max(SURVIVAL_TIME_FLOOR);
        for j in 0..p_time {
            design_derivative_exit[[i, j]] = deriv_buf[j] * chain;
        }
    }

    let time_block = TimeBlockInput {
        design_entry: entry_basis.design,
        design_exit: exit_basis.design,
        design_derivative_exit,
        offset_entry,
        offset_exit,
        derivative_offset_exit,
        penalties: entry_basis.penalties,
        nullspace_dims: entry_basis.nullspace_dims,
        initial_log_lambdas: None,
        initial_beta: None,
    };

    Ok((time_block, knots))
}

/// Fit a survival marginal-slope model.
///
/// The model uses:
/// - Gompertz-Makeham parametric baseline (as offset)
/// - B-spline deviation from baseline (learnable)
/// - Log-slope surface over ancestry (Duchon smooth)
pub fn fit_survival_marginal_slope(
    bundle: &SurvivalTrainingBundle,
    cfg: &SurvivalMarginalSlopeConfig,
) -> Result<SurvivalMarginalSlopeModel, EstimationError> {
    let data = &bundle.data;
    let num_pcs = data.pcs.ncols();

    // Standardize PGS
    let z = standardize_score(&data.pgs);

    // Build time block with GM baseline offsets
    let (time_block, time_knots) = build_time_block(
        &data.age_entry,
        &data.age_exit,
        &cfg.baseline,
        cfg.time_num_knots,
        cfg.time_degree,
    )?;

    // Build covariate matrix [sex | PCs]
    let covariate_data = build_covariate_matrix(data.sex.view(), data.pcs.view());

    // Log-slope surface spec
    let logslopespec = build_surface_spec(
        num_pcs,
        &cfg.surface.center_strategy,
        cfg.surface.duchon_power,
        &cfg.surface.duchon_nullspace,
        cfg.surface.duchon_length_scale,
    );

    let spec = SurvivalMarginalSlopeTermSpec {
        age_entry: data.age_entry.clone(),
        age_exit: data.age_exit.clone(),
        event_target: data.event_target.mapv(f64::from),
        weights: data.sample_weight.clone(),
        z,
        derivative_guard: cfg.derivative_guard,
        time_block,
        logslopespec,
    };

    let request = SurvivalMarginalSlopeFitRequest {
        data: covariate_data.view(),
        spec,
        options: cfg.surface.fit_options.clone(),
        kappa_options: cfg.surface.kappa_options.clone(),
    };

    let fit_result = fit_model(FitRequest::SurvivalMarginalSlope(request))
        .map_err(|e| EstimationError::InvalidSpecification(e))?;

    let result = match fit_result {
        FitResult::SurvivalMarginalSlope(r) => r,
        _ => {
            return Err(EstimationError::InvalidSpecification(
                "unexpected fit result variant".to_string(),
            ));
        }
    };

    Ok(SurvivalMarginalSlopeModel {
        result,
        num_pcs,
        baseline: cfg.baseline.clone(),
        time_knots,
        time_degree: cfg.time_degree,
    })
}

/// Convenience wrapper using default survival config.
pub fn fit_survival_marginal_slope_default(
    bundle: &SurvivalTrainingBundle,
) -> Result<SurvivalMarginalSlopeModel, EstimationError> {
    fit_survival_marginal_slope(bundle, &SurvivalMarginalSlopeConfig::default())
}

// ── Shared utilities ────────────────────────────────────────────────

/// Standardize a score vector to approximately N(0,1).
fn standardize_score(p: &Array1<f64>) -> Array1<f64> {
    let n = p.len();
    if n == 0 {
        return Array1::zeros(0);
    }
    let mean = p.iter().copied().sum::<f64>() / n as f64;
    let var = p
        .iter()
        .map(|&v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / n.max(1) as f64;
    let sd = var.max(1e-12).sqrt();
    p.mapv(|v| (v - mean) / sd)
}
