use gnomon::calibrate::calibrator::{
    CalibratorFeatures, CalibratorModel, CalibratorSpec, build_calibrator_design, fit_calibrator,
    predict_calibrator,
};
use gnomon::calibrate::model::{
    BasisConfig, LinkFunction, calibrator_enabled, reset_calibrator_flag, set_calibrator_enabled,
};
use gnomon::calibrate::survival::{
    CholeskyFactor, HessianFactor, delta_method_standard_errors, survival_calibrator_features,
};
use ndarray::{Array1, Array2, array};

#[test]
fn calibration_toggle_round_trip() {
    reset_calibrator_flag();
    assert!(calibrator_enabled());

    set_calibrator_enabled(false);
    assert!(!calibrator_enabled());

    reset_calibrator_flag();
    assert!(calibrator_enabled());
}

fn make_survival_factor() -> HessianFactor {
    let lower = array![[2.0, 0.0], [0.5, (2.75_f64).sqrt()]];
    HessianFactor::Expected {
        factor: CholeskyFactor { lower },
    }
}

#[test]
fn logit_risk_calibration_improves_log_loss() {
    let predictions = array![0.12, 0.22, 0.55, 0.72, 0.81, 0.35];
    let outcomes = array![0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let weights = Array1::ones(predictions.len());

    let design = Array2::from_shape_vec(
        (predictions.len(), 2),
        vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.2, 0.4, 0.6, 0.8, 0.3, 0.3, 0.7],
    )
    .unwrap();

    let factor = make_survival_factor();
    let se = delta_method_standard_errors(&factor, &design).unwrap();
    assert_eq!(se.len(), predictions.len());

    let features_matrix =
        survival_calibrator_features(&predictions, &design, Some(&factor), None).unwrap();

    let features = CalibratorFeatures {
        pred: features_matrix.column(0).to_owned(),
        se: features_matrix.column(1).to_owned(),
        dist: Array1::zeros(predictions.len()),
        pred_identity: features_matrix.column(0).to_owned(),
        fisher_weights: weights.clone(),
    };

    let spec = CalibratorSpec {
        link: LinkFunction::Logit,
        pred_basis: BasisConfig {
            degree: 1,
            num_knots: 4,
        },
        se_basis: BasisConfig {
            degree: 1,
            num_knots: 4,
        },
        dist_basis: BasisConfig {
            degree: 1,
            num_knots: 4,
        },
        penalty_order_pred: 2,
        penalty_order_se: 2,
        penalty_order_dist: 2,
        distance_hinge: false,
        prior_weights: Some(weights.clone()),
        firth: CalibratorSpec::firth_default_for_link(LinkFunction::Logit),
    };

    let (design_matrix, penalties, schema, offset) =
        build_calibrator_design(&features, &spec).expect("design build");
    assert!(design_matrix.ncols() > 0);

    let null_dims = penalties
        .iter()
        .zip(
            [
                schema.penalty_nullspace_dims.0,
                schema.penalty_nullspace_dims.1,
                schema.penalty_nullspace_dims.2,
                schema.penalty_nullspace_dims.3,
            ]
            .into_iter(),
        )
        .filter_map(|(penalty, dim)| {
            let max_abs = penalty
                .iter()
                .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
            (max_abs > 1e-12).then_some(dim)
        })
        .collect::<Vec<_>>();
    let (beta, lambdas, _, _, _) = fit_calibrator(
        outcomes.view(),
        weights.view(),
        design_matrix.view(),
        offset.view(),
        &penalties,
        &null_dims,
        LinkFunction::Logit,
    )
    .expect("fit calibrator");

    let mut saved_spec = spec.clone();
    saved_spec.prior_weights = None;

    let model = CalibratorModel {
        spec: saved_spec,
        knots_pred: schema.knots_pred,
        knots_se: schema.knots_se,
        knots_dist: schema.knots_dist,
        pred_constraint_transform: schema.pred_constraint_transform,
        stz_se: schema.stz_se,
        stz_dist: schema.stz_dist,
        penalty_nullspace_dims: schema.penalty_nullspace_dims,
        standardize_pred: schema.standardize_pred,
        standardize_se: schema.standardize_se,
        standardize_dist: schema.standardize_dist,
        interaction_center_pred: Some(schema.interaction_center_pred),
        se_wiggle_only_drop: schema.se_wiggle_only_drop,
        dist_wiggle_only_drop: schema.dist_wiggle_only_drop,
        lambda_pred: lambdas[0],
        lambda_pred_param: lambdas[1],
        lambda_se: lambdas[2],
        lambda_dist: lambdas[3],
        coefficients: beta,
        column_spans: schema.column_spans,
        pred_param_range: schema.pred_param_range,
        scale: None,
        assumes_frequency_weights: true,
    };

    let base_loss = outcomes
        .iter()
        .zip(predictions.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(1e-6, 1.0 - 1e-6);
            -((y * clipped.ln()) + ((1.0 - y) * (1.0 - clipped).ln()))
        })
        .sum::<f64>()
        / (predictions.len() as f64);

    let calibrated = predict_calibrator(
        &model,
        features.pred.view(),
        features.se.view(),
        features.dist.view(),
    )
    .expect("predict calibrated");

    let calibrated_loss = outcomes
        .iter()
        .zip(calibrated.iter())
        .map(|(&y, &p)| {
            let clipped = p.clamp(1e-6, 1.0 - 1e-6);
            -((y * clipped.ln()) + ((1.0 - y) * (1.0 - clipped).ln()))
        })
        .sum::<f64>()
        / (predictions.len() as f64);

    assert!(calibrated_loss < base_loss);
}
