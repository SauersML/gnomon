use gnomon::calibrate::faer_ndarray::ldlt_rook;
use gnomon::calibrate::calibrator::{CalibratorModel, CalibratorSpec};
use gnomon::calibrate::model::{calibrator_enabled, set_calibrator_enabled, BasisConfig, LinkFunction};
use gnomon::calibrate::survival::{
    ColumnRange, CovariateLayout, PenaltyDescriptor, SurvivalModelArtifacts, SurvivalSpec,
    SurvivalTrainingData, WorkingModelSurvival, build_survival_layout, conditional_absolute_risk,
    cumulative_incidence, design_row_at_age, DEFAULT_RISK_EPSILON,
};
use ndarray::{Array1, Array2, Axis, array};

const GUARD_DELTA: f64 = 0.1;
const BASELINE_PENALTY_ORDER: usize = 2;
const BASELINE_LAMBDA: f64 = 0.5;
const MONOTONIC_GRID: usize = 6;

struct TrustedReference {
    artifacts: SurvivalModelArtifacts,
    data: SurvivalTrainingData,
    layout: gnomon::calibrate::survival::SurvivalLayout,
}

impl TrustedReference {
    fn new() -> Self {
        let data = trusted_reference_data();
        let basis = reference_basis();
        let bundle = build_survival_layout(
            &data,
            &basis,
            GUARD_DELTA,
            BASELINE_PENALTY_ORDER,
            BASELINE_LAMBDA,
            MONOTONIC_GRID,
            None,
        )
        .expect("layout");
        let layout = bundle.layout;
        let monotonicity = bundle.monotonicity;

        let mut spec = SurvivalSpec::default();
        spec.barrier_weight = 0.0;
        spec.use_expected_information = true;

        let mut model =
            WorkingModelSurvival::new(layout.clone(), &data, monotonicity, spec).expect("model");
        let beta = newton_solve(&mut model);
        let artifacts = build_artifacts(&layout, &basis, &beta);

        Self {
            artifacts,
            data,
            layout,
        }
    }
}

fn trusted_reference_data() -> SurvivalTrainingData {
    let age_entry = array![50.0, 50.0, 50.0, 50.0, 51.0, 52.0, 53.0, 54.0,];
    let age_exit = array![58.0, 61.0, 64.0, 67.0, 59.0, 62.0, 65.0, 68.0,];
    let event_target = array![1, 0, 1, 0, 0, 1, 0, 1];
    let event_competing = Array1::zeros(age_entry.len());
    let sample_weight = array![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0];
    let n = age_entry.len();
    let pgs = Array1::zeros(n);
    let sex = Array1::zeros(n);
    let pcs = Array2::zeros((n, 0));

    SurvivalTrainingData {
        age_entry,
        age_exit,
        event_target,
        event_competing,
        sample_weight,
        pgs,
        sex,
        pcs,
        extra_static_covariates: Array2::zeros((n, 0)),
        extra_static_names: Vec::new(),
    }
}

fn reference_basis() -> gnomon::calibrate::survival::BasisDescriptor {
    let min_age = (GUARD_DELTA).ln();
    let max_age = (68.0 - 50.0 + GUARD_DELTA).ln();
    let span = max_age - min_age;
    let knot1 = min_age + span / 3.0;
    let knot2 = min_age + 2.0 * span / 3.0;

    gnomon::calibrate::survival::BasisDescriptor {
        knot_vector: array![
            min_age, min_age, min_age, knot1, knot2, max_age, max_age, max_age,
        ],
        degree: 2,
    }
}

fn newton_solve(model: &mut WorkingModelSurvival) -> Array1<f64> {
    let mut beta = Array1::<f64>::zeros(model.layout.combined_exit.ncols());
    for _ in 0..64 {
        let state = model.update_state(&beta).expect("update");
        let grad_norm = l2_norm(state.gradient.view());
        if grad_norm < 1e-10 {
            break;
        }
        let delta = solve_ldlt_system(&state.hessian, &state.gradient);
        let step_norm = l2_norm(delta.view());
        beta -= &delta;
        if step_norm < 1e-10 {
            break;
        }
    }
    beta
}

fn l2_norm(view: ndarray::ArrayView1<'_, f64>) -> f64 {
    view.iter().map(|v| v * v).sum::<f64>().sqrt()
}

fn solve_ldlt_system(matrix: &Array2<f64>, rhs: &Array1<f64>) -> Array1<f64> {
    let (lower, diag, subdiag, perm_fwd, perm_inv, _) = ldlt_rook(matrix).expect("ldlt");
    let n = rhs.len();
    let mut permuted = Array1::<f64>::zeros(n);
    for i in 0..n {
        permuted[i] = rhs[perm_inv[i]];
    }
    let mut y = permuted;
    for i in 0..n {
        let mut sum = y[i];
        for j in 0..i {
            sum -= lower[[i, j]] * y[j];
        }
        y[i] = sum;
    }
    let mut z = Array1::<f64>::zeros(n);
    let mut idx = 0usize;
    while idx < n {
        if idx + 1 < n && subdiag[idx].abs() > 1e-12 {
            let a = diag[idx];
            let b = subdiag[idx];
            let c = diag[idx + 1];
            let det = a * c - b * b;
            if det.abs() <= 1e-18 {
                panic!("singular block in ldlt solve");
            }
            let y0 = y[idx];
            let y1 = y[idx + 1];
            z[idx] = (c * y0 - b * y1) / det;
            z[idx + 1] = (-b * y0 + a * y1) / det;
            idx += 2;
        } else {
            let d = diag[idx];
            if d.abs() <= 1e-18 {
                panic!("singular pivot in ldlt solve");
            }
            z[idx] = y[idx] / d;
            idx += 1;
        }
    }
    let mut x_perm = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = z[i];
        for j in i + 1..n {
            sum -= lower[[j, i]] * x_perm[j];
        }
        x_perm[i] = sum;
    }
    let mut solution = Array1::<f64>::zeros(n);
    for i in 0..n {
        solution[perm_fwd[i]] = x_perm[i];
    }
    solution
}

fn build_artifacts(
    layout: &gnomon::calibrate::survival::SurvivalLayout,
    basis: &gnomon::calibrate::survival::BasisDescriptor,
    beta: &Array1<f64>,
) -> SurvivalModelArtifacts {
    let static_covariate_layout = covariate_layout(layout);
    let penalty_descriptor = PenaltyDescriptor {
        order: BASELINE_PENALTY_ORDER,
        lambda: BASELINE_LAMBDA,
        matrix: layout.penalties.blocks[0].matrix.clone(),
        column_range: ColumnRange::new(0, layout.baseline_exit.ncols()),
    };

    SurvivalModelArtifacts {
        coefficients: beta.clone(),
        age_basis: basis.clone(),
        time_varying_basis: None,
        static_covariate_layout,
        penalties: vec![penalty_descriptor],
        age_transform: layout.age_transform,
        reference_constraint: layout.reference_constraint.clone(),
        interaction_metadata: Vec::new(),
        companion_models: Vec::new(),
        hessian_factor: None,
        calibrator: None,
    }
}

fn covariate_layout(layout: &gnomon::calibrate::survival::SurvivalLayout) -> CovariateLayout {
    // Match production logic: combine static + extra covariates
    let mut ranges = value_ranges(&layout.static_covariates);
    ranges.extend(value_ranges(&layout.extra_static_covariates));
    CovariateLayout {
        column_names: layout.static_covariate_names.clone(),
        ranges,
    }
}

/// Helper to combine static and extra covariates for a single row
fn combined_covariates_row(
    layout: &gnomon::calibrate::survival::SurvivalLayout,
    row_idx: usize,
) -> Array1<f64> {
    let static_row = layout.static_covariates.row(row_idx);
    let extra_row = layout.extra_static_covariates.row(row_idx);
    ndarray::concatenate(Axis(0), &[static_row, extra_row]).expect("concatenate covariates")
}

fn value_ranges(matrix: &Array2<f64>) -> Vec<gnomon::calibrate::survival::ValueRange> {
    (0..matrix.ncols())
        .map(|col_idx| {
            if matrix.nrows() == 0 {
                return gnomon::calibrate::survival::ValueRange { min: 0.0, max: 0.0 };
            }
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &value in matrix.column(col_idx).iter() {
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }
            gnomon::calibrate::survival::ValueRange {
                min: min_val,
                max: max_val,
            }
        })
        .collect()
}

fn weighted_brier(weights: &Array1<f64>, outcomes: &Array1<f64>, predictions: &Array1<f64>) -> f64 {
    let total_weight: f64 = weights.sum();
    let mut score = 0.0;
    for i in 0..weights.len() {
        let diff = outcomes[i] - predictions[i];
        score += weights[i] * diff * diff;
    }
    score / total_weight
}

fn replicated_brier(
    weights: &Array1<f64>,
    outcomes: &Array1<f64>,
    predictions: &Array1<f64>,
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for i in 0..weights.len() {
        let w = weights[i] as usize;
        for _ in 0..w {
            let diff = outcomes[i] - predictions[i];
            total += diff * diff;
            count += 1;
        }
    }
    total / (count as f64)
}

#[test]
fn cumulative_incidence_matches_reference_library() {
    let trusted = TrustedReference::new();
    let ages = [58.0, 61.0, 64.0, 67.0];
    let expected_cif = [
        0.6321210934815431,
        0.6321207264171644,
        0.6321204109495415,
        0.6321201334574286,
    ];
    // Test first 4 rows with expected CIF values
    for (idx, (age, expected)) in ages.iter().zip(expected_cif.iter()).enumerate() {
        let cov = combined_covariates_row(&trusted.layout, idx);
        let cif = cumulative_incidence(*age, &cov, &trusted.artifacts).unwrap();
        assert!((cif - *expected).abs() <= 1e-9);
    }

    // Test Brier score across all data
    let mut preds = Array1::<f64>::zeros(trusted.data.age_exit.len());
    for (idx, exit_age) in trusted.data.age_exit.iter().enumerate() {
        let cov = combined_covariates_row(&trusted.layout, idx);
        preds[idx] = cumulative_incidence(*exit_age, &cov, &trusted.artifacts).unwrap();
    }
    let outcomes = trusted.data.event_target.map(|v| f64::from(*v));
    let brier = weighted_brier(&trusted.data.sample_weight, &outcomes, &preds);
    assert!((brier - 0.286330117856061).abs() <= 1e-12);
}

#[test]
fn conditional_risk_monotonic_with_calibration_toggle() {
    struct CalibratorFlagGuard(bool);
    impl CalibratorFlagGuard {
        fn new(previous: bool) -> Self {
            Self(previous)
        }

        fn previous(&self) -> bool {
            self.0
        }
    }
    impl Drop for CalibratorFlagGuard {
        fn drop(&mut self) {
            set_calibrator_enabled(self.0);
        }
    }

    fn identity_calibrator() -> CalibratorModel {
        CalibratorModel {
            spec: CalibratorSpec {
                link: LinkFunction::Logit,
                pred_basis: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                se_basis: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                dist_basis: BasisConfig {
                    num_knots: 2,
                    degree: 1,
                },
                penalty_order_pred: 2,
                penalty_order_se: 2,
                penalty_order_dist: 2,
                distance_hinge: false,
                prior_weights: None,
                firth: None,
            },
            knots_pred: Array1::zeros(0),
            knots_se: Array1::zeros(0),
            knots_dist: Array1::zeros(0),
            pred_constraint_transform: Array2::zeros((0, 0)),
            stz_se: Array2::zeros((0, 0)),
            stz_dist: Array2::zeros((0, 0)),
            penalty_nullspace_dims: (0, 0, 0, 0),
            standardize_pred: (0.0, 1.0),
            standardize_se: (0.0, 1.0),
            standardize_dist: (0.0, 1.0),
            interaction_center_pred: None,
            se_wiggle_only_drop: false,
            dist_wiggle_only_drop: false,
            lambda_pred: 1.0,
            lambda_pred_param: 1.0,
            lambda_se: 1.0,
            lambda_dist: 1.0,
            coefficients: Array1::zeros(0),
            column_spans: (0..0, 0..0, 0..0),
            pred_param_range: 0..0,
            scale: None,
            assumes_frequency_weights: true,
        }
    }

    fn risk_with_gradient(
        t0: f64,
        t1: f64,
        covs: &Array1<f64>,
        artifacts: &SurvivalModelArtifacts,
    ) -> (f64, Array1<f64>) {
        let risk =
            conditional_absolute_risk(t0, t1, covs, Some(0.0), None, artifacts).expect("risk");
        let coeffs = &artifacts.coefficients;
        let design_entry = design_row_at_age(t0, covs.view(), artifacts).expect("design entry");
        let design_exit = design_row_at_age(t1, covs.view(), artifacts).expect("design exit");
        assert_eq!(design_entry.len(), coeffs.len());
        assert_eq!(design_exit.len(), coeffs.len());

        let eta_entry = design_entry.dot(coeffs);
        let eta_exit = design_exit.dot(coeffs);
        let h_entry = eta_entry.exp();
        let h_exit = eta_exit.exp();
        let exp_neg_entry = (-h_entry).exp();
        let exp_neg_exit = (-h_exit).exp();
        let f_entry = 1.0 - exp_neg_entry;
        let f_exit = 1.0 - exp_neg_exit;
        let delta_raw = f_exit - f_entry;
        let denom_raw = 1.0 - f_entry;
        let delta = delta_raw.max(0.0);
        let denom = denom_raw.max(DEFAULT_RISK_EPSILON);
        let d_f_entry = h_entry * exp_neg_entry;
        let d_f_exit = h_exit * exp_neg_exit;
        let dr_deta_exit = if delta_raw > 0.0 { d_f_exit / denom } else { 0.0 };
        let numerator = if delta_raw > 0.0 { delta } else { 0.0 };
        let dnum = if delta_raw > 0.0 { -d_f_entry } else { 0.0 };
        let dden = -d_f_entry;
        let dr_deta_entry = if denom_raw > DEFAULT_RISK_EPSILON {
            (dnum * denom_raw - numerator * dden) / (denom_raw * denom_raw)
        } else {
            0.0
        };

        let risk_clamped = risk.max(1e-12).min(1.0 - 1e-12);
        let logistic_scale = 1.0 / (risk_clamped * (1.0 - risk_clamped));
        let grad_exit = design_exit.mapv(|v| v * dr_deta_exit * logistic_scale);
        let grad_entry = design_entry.mapv(|v| v * dr_deta_entry * logistic_scale);
        let gradient = grad_exit + grad_entry;

        (risk, gradient)
    }

    let trusted = TrustedReference::new();
    let covs = combined_covariates_row(&trusted.layout, 0);
    let mut artifacts = trusted.artifacts.clone();
    artifacts.calibrator = Some(identity_calibrator());

    let t0 = 55.0;
    let horizons = [60.0, 62.0, 64.0, 66.0];

    let mut risks = Vec::new();
    let mut design = Array2::<f64>::zeros((horizons.len(), artifacts.coefficients.len()));
    for (row_idx, &t1) in horizons.iter().enumerate() {
        let (risk, gradient) = risk_with_gradient(t0, t1, &covs, &artifacts);
        risks.push(risk);
        design
            .row_mut(row_idx)
            .assign(&gradient.view());
    }

    let risks_array = Array1::from_vec(risks.clone());
    assert!(risks.windows(2).all(|w| w[1] + 1e-12 >= w[0]));

    let original_flag = calibrator_enabled();
    let guard = CalibratorFlagGuard::new(original_flag);
    let _ = guard.previous();

    set_calibrator_enabled(false);
    let uncalibrated = artifacts
        .apply_logit_risk_calibrator(&risks_array, &design, None)
        .expect("uncalibrated application");

    set_calibrator_enabled(true);
    let calibrated = artifacts
        .apply_logit_risk_calibrator(&risks_array, &design, None)
        .expect("calibrated application");

    let uncalibrated_vec = uncalibrated.to_vec();
    let calibrated_vec = calibrated.to_vec();
    assert!(uncalibrated_vec
        .windows(2)
        .all(|w| w[1] + 1e-12 >= w[0]));
    assert!(calibrated_vec
        .windows(2)
        .all(|w| w[1] + 1e-12 >= w[0]));
}

#[test]
fn weighted_brier_matches_frequency_replication() {
    let trusted = TrustedReference::new();
    let mut preds = Array1::<f64>::zeros(trusted.data.age_exit.len());
    for (idx, exit_age) in trusted.data.age_exit.iter().enumerate() {
        let cov = combined_covariates_row(&trusted.layout, idx);
        preds[idx] = cumulative_incidence(*exit_age, &cov, &trusted.artifacts).unwrap();
    }
    let outcomes = trusted.data.event_target.map(|v| f64::from(*v));
    let weighted = weighted_brier(&trusted.data.sample_weight, &outcomes, &preds);
    let replicated = replicated_brier(&trusted.data.sample_weight, &outcomes, &preds);
    assert!((weighted - replicated).abs() <= 1e-12);
}
