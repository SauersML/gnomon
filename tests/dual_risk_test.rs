use gnomon::calibrate::model::{
    BasisConfig, InteractionPenaltyKind, ModelConfig, ModelFamily,
    SurvivalRiskType, TrainedModel, MappedCoefficients,
};
use gnomon::calibrate::survival::{
    CompanionModelHandle, SurvivalModelArtifacts, AgeTransform, BasisDescriptor, CovariateLayout,
    ReferenceConstraint, MonotonicityPenalty, ValueRange,
};
use ndarray::{Array1, Array2, array};
use std::collections::HashMap;

// --- Helper: Construct a manual model with constant hazard ---
fn make_constant_hazard_artifacts(hazard_rate: f64) -> SurvivalModelArtifacts {
    // We want constant hazard lambda.
    // Royston-Parmar H(t) = exp(eta(t)). h(t) = H'(t).
    // If we want h(t) = lambda, then H(t) = lambda * t (assuming starts at 0).
    // eta(t) = ln(lambda * t) = ln(lambda) + ln(t).
    // Our AgeTransform with delta=1.0 gives u = ln(age + 1).
    // If we approximate t ~ t+1 for large t, this works, but for t near 0 it's slightly offset.
    // Actually, if we want exact constant hazard in the model structure:
    // H(t) = exp(eta(u)). u = ln(t+1).
    // We need exp(eta) = lambda * (t+1) approx lambda * t.
    // Let's target H(t) = lambda * (t+1). Then h(t) = lambda.
    // Then eta(u) = ln(lambda) + ln(t+1) = ln(lambda) + u.
    // This is a linear function of u with slope 1 and intercept ln(lambda).

    let ln_lambda = hazard_rate.ln();
    // At u=0 (age=0), eta = ln_lambda.
    // At u=10 (age=exp(10)-1), eta = ln_lambda + 10.
    // Spline coeffs for linear function f(u) = a + b*u on [0, 10]:
    // c1 = f(0) = ln_lambda
    // c2 = f(10) = ln_lambda + 10.0

    let c1 = ln_lambda;
    let c2 = ln_lambda + 10.0;

    // 2 columns for degree 1 + 2 static covs
    let coefficients = array![c1, c2, 0.0, 0.0];

    // 2. Basis: Degree 1 B-spline covering log-age 0-10.
    let age_basis = BasisDescriptor {
        knot_vector: array![0.0, 0.0, 10.0, 10.0],
        degree: 1,
    };

    // 3. Covariate Layout: Must include PGS and Sex as expected by predict_survival
    let static_covariates = CovariateLayout {
        column_names: vec!["pgs".to_string(), "sex".to_string()],
        ranges: vec![
            ValueRange { min: -100.0, max: 100.0 },
            ValueRange { min: 0.0, max: 1.0 }
        ],
    };

    // 4. Age Transform: Log transform (standard)
    let age_transform = AgeTransform {
        minimum_age: 0.0,
        delta: 1.0, // ln(age - 0 + 1)
    };

    // 5. Reference Constraint: Identity
    // Unconstrained basis has 2 columns. Reference constraint usually removes 1.
    // But for this test manual construction, we can skip the constraint logic if we provide Identity.
    // The design matrix builder applies this.
    // Basis (N x 2) -> Constraint (2 x 2) -> (N x 2).
    let reference_constraint = ReferenceConstraint {
        transform: Array2::eye(2),
        reference_log_age: 0.0,
    };

    // 6. Monotonicity: Empty
    let monotonicity = MonotonicityPenalty {
        derivative_design: Array2::zeros((0, 0)),
        quadrature_design: Array2::zeros((0, 0)),
        grid_ages: Array1::zeros(0),
        quadrature_left: Array1::zeros(0),
        quadrature_right: Array1::zeros(0),
    };

    SurvivalModelArtifacts {
        coefficients,
        age_basis,
        time_varying_basis: None,
        static_covariate_layout: static_covariates,
        penalties: vec![],
        age_transform,
        reference_constraint,
        monotonicity,
        interaction_metadata: vec![],
        companion_models: vec![],
        hessian_factor: None,
        calibrator: None,
    }
}

// --- Test 1: Analytic Verification (Constant Hazards) ---
// If h_D = 0.1 and h_M = 0.1, Crude Risk at t=10 should be:
// 0.1 / (0.1 + 0.1) * (1 - exp(-(0.1 + 0.1)*10))
// = 0.5 * (1 - exp(-2.0))
// = 0.5 * (1 - 0.135335) = 0.43233
#[test]
fn test_crude_risk_analytic_match() {
    let lambda_d = 0.1;
    let lambda_m = 0.1;

    let mut disease_model = make_constant_hazard_artifacts(lambda_d);
    // Register the companion handle
    disease_model.companion_models.push(CompanionModelHandle {
        reference: "__internal_mortality".to_string(),
        cif_horizons: vec![],
    });

    let mortality_model = make_constant_hazard_artifacts(lambda_m);

    let mut companions = HashMap::new();
    companions.insert("__internal_mortality".to_string(), mortality_model);

    // Build TrainedModel wrapper
    let model = TrainedModel {
        config: ModelConfig {
            model_family: ModelFamily::Survival(gnomon::calibrate::survival::SurvivalSpec::default()),
            penalty_order: 2, convergence_tolerance: 1e-6, max_iterations: 1, reml_convergence_tolerance: 1e-6, reml_max_iterations: 1, firth_bias_reduction: false, reml_parallel_threshold: 4,
            pgs_basis_config: BasisConfig { num_knots: 0, degree: 0 },
            pc_configs: vec![], pgs_range: (0.0, 1.0), interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(), knot_vectors: HashMap::new(), range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(), interaction_centering_means: HashMap::new(), interaction_orth_alpha: HashMap::new(),
            survival: None
        },
        coefficients: MappedCoefficients::default(),
        lambdas: vec![],
        hull: None,
        penalized_hessian: None,
        scale: None,
        calibrator: None,
        survival: Some(disease_model),
        survival_companions: companions,
    };

    // Predict
    let age_entry = array![0.0];
    let age_exit = array![10.0];
    let p_new = array![0.0]; // ignored
    let sex_new = array![0.0]; // ignored
    let pcs_new = Array2::zeros((1, 0)); // ignored

    let pred = model.predict_survival(
        age_entry.view(), age_exit.view(), p_new.view(), sex_new.view(), pcs_new.view(),
        SurvivalRiskType::Crude,
        None
    ).expect("Prediction success");

    let risk = pred.conditional_risk[0];
    let expected = (lambda_d / (lambda_d + lambda_m)) * (1.0 - (- (lambda_d + lambda_m) * 10.0).exp());

    println!("Computed Risk: {:.6}, Expected: {:.6}", risk, expected);
    assert!((risk - expected).abs() < 1e-3, "Integration deviation too large");
}

// --- Test 2: High Mortality Suppression ---
// h_D = 0.1, h_M = 100.0 (Instant Death). Crude Risk should be ~0.
#[test]
fn test_crude_risk_high_mortality_suppression() {
    let lambda_d = 0.1;
    let lambda_m = 100.0;

    let mut disease_model = make_constant_hazard_artifacts(lambda_d);
    disease_model.companion_models.push(CompanionModelHandle { reference: "__internal_mortality".to_string(), cif_horizons: vec![] });
    let mortality_model = make_constant_hazard_artifacts(lambda_m);
    let mut companions = HashMap::new();
    companions.insert("__internal_mortality".to_string(), mortality_model);

    let model = TrainedModel {
        config: ModelConfig {
            model_family: ModelFamily::Survival(gnomon::calibrate::survival::SurvivalSpec::default()),
            penalty_order: 2, convergence_tolerance: 1e-6, max_iterations: 1, reml_convergence_tolerance: 1e-6, reml_max_iterations: 1, firth_bias_reduction: false, reml_parallel_threshold: 4,
            pgs_basis_config: BasisConfig { num_knots: 0, degree: 0 },
            pc_configs: vec![], pgs_range: (0.0, 1.0), interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(), knot_vectors: HashMap::new(), range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(), interaction_centering_means: HashMap::new(), interaction_orth_alpha: HashMap::new(),
            survival: None
        },
        coefficients: MappedCoefficients::default(), lambdas: vec![], hull: None, penalized_hessian: None, scale: None, calibrator: None,
        survival: Some(disease_model),
        survival_companions: companions,
    };

    let pred = model.predict_survival(
        array![0.0].view(), array![10.0].view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Crude,
        None
    ).unwrap();

    let risk = pred.conditional_risk[0];
    println!("High Mortality Risk: {:.6}", risk);
    assert!(risk < 1e-3, "Risk should be negligible due to immediate death");

    // Compare with Net Risk
    let net_pred = model.predict_survival(
        array![0.0].view(), array![10.0].view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Net,
        None
    ).unwrap();
    let net_risk = net_pred.conditional_risk[0];
    println!("Net Risk (Immortal): {:.6}", net_risk);
    assert!(net_risk > 0.5, "Net risk should be high");
}

// --- Test 3: Missing Companion Error ---
#[test]
fn test_missing_companion_error() {
    let disease_model = make_constant_hazard_artifacts(0.1);
    // Do NOT register companion in map
    let companions = HashMap::new();

    let model = TrainedModel {
        config: ModelConfig {
            model_family: ModelFamily::Survival(gnomon::calibrate::survival::SurvivalSpec::default()),
            penalty_order: 2, convergence_tolerance: 1e-6, max_iterations: 1, reml_convergence_tolerance: 1e-6, reml_max_iterations: 1, firth_bias_reduction: false, reml_parallel_threshold: 4,
            pgs_basis_config: BasisConfig { num_knots: 0, degree: 0 },
            pc_configs: vec![], pgs_range: (0.0, 1.0), interaction_penalty: InteractionPenaltyKind::Anisotropic,
            sum_to_zero_constraints: HashMap::new(), knot_vectors: HashMap::new(), range_transforms: HashMap::new(),
            pc_null_transforms: HashMap::new(), interaction_centering_means: HashMap::new(), interaction_orth_alpha: HashMap::new(),
            survival: None
        },
        coefficients: MappedCoefficients::default(), lambdas: vec![], hull: None, penalized_hessian: None, scale: None, calibrator: None,
        survival: Some(disease_model),
        survival_companions: companions,
    };

    let res = model.predict_survival(
        array![0.0].view(), array![10.0].view(), array![0.0].view(), array![0.0].view(), Array2::zeros((1, 0)).view(),
        SurvivalRiskType::Crude,
        None
    );

    assert!(res.is_err());
}
