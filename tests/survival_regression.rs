use gnomon::calibrate::survival::{
    SurvivalModelArtifacts, SurvivalTrainingData, conditional_absolute_risk, cumulative_incidence,
};
use ndarray::{Array1, Array2, Axis};
use serde::Deserialize;

struct TrustedReference {
    artifacts: SurvivalModelArtifacts,
    data: SurvivalTrainingData,
    static_covariates: Array2<f64>,
    extra_static_covariates: Array2<f64>,
    lifelines_cif: Vec<f64>,
    lifelines_weighted_brier: f64,
}

#[derive(Deserialize)]
struct TrustedReferenceFixture {
    age_entry: Vec<f64>,
    age_exit: Vec<f64>,
    event_target: Vec<u8>,
    event_competing: Vec<u8>,
    sample_weight: Vec<f64>,
    pgs: Vec<f64>,
    sex: Vec<f64>,
    pcs: Vec<Vec<f64>>,
    extra_static_covariates: Vec<Vec<f64>>,
    extra_static_names: Vec<String>,
    static_covariates: Vec<Vec<f64>>,
    artifacts: SurvivalModelArtifacts,
    lifelines_cif: Vec<f64>,
    lifelines_weighted_brier: f64,
}

impl TrustedReference {
    fn load() -> Self {
        let raw: TrustedReferenceFixture =
            serde_json::from_str(include_str!("trusted_reference_artifacts.json"))
                .expect("trusted reference fixture");

        let static_covariates = rows_to_array(&raw.static_covariates);
        let extra_static_covariates = rows_to_array(&raw.extra_static_covariates);
        let pcs = rows_to_array(&raw.pcs);

        let data = SurvivalTrainingData {
            age_entry: Array1::from(raw.age_entry),
            age_exit: Array1::from(raw.age_exit),
            event_target: Array1::from(raw.event_target),
            event_competing: Array1::from(raw.event_competing),
            sample_weight: Array1::from(raw.sample_weight),
            pgs: Array1::from(raw.pgs),
            sex: Array1::from(raw.sex),
            pcs,
            extra_static_covariates: extra_static_covariates.clone(),
            extra_static_names: raw.extra_static_names,
        };

        Self {
            artifacts: raw.artifacts,
            data,
            static_covariates,
            extra_static_covariates,
            lifelines_cif: raw.lifelines_cif,
            lifelines_weighted_brier: raw.lifelines_weighted_brier,
        }
    }

    fn covariates_row(&self, row_idx: usize) -> Array1<f64> {
        let static_row = self.static_covariates.row(row_idx);
        let extra_row = self.extra_static_covariates.row(row_idx);
        if extra_row.len() == 0 {
            return static_row.to_owned();
        }
        ndarray::concatenate(Axis(0), &[static_row, extra_row]).expect("concatenate covariates")
    }
}

fn rows_to_array(rows: &[Vec<f64>]) -> Array2<f64> {
    let rows_len = rows.len();
    let cols = rows.first().map(|row| row.len()).unwrap_or(0);
    let mut data = Vec::with_capacity(rows_len * cols);
    for row in rows {
        assert_eq!(row.len(), cols, "inconsistent row width in fixture");
        data.extend_from_slice(row);
    }
    if rows_len == 0 || cols == 0 {
        return Array2::zeros((rows_len, cols));
    }
    Array2::from_shape_vec((rows_len, cols), data).expect("reshape rows into matrix")
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
    let trusted = TrustedReference::load();

    // Compute CIF values using our implementation
    let mut computed_cif = Vec::new();
    for (idx, exit_age) in trusted.data.age_exit.iter().enumerate().take(4) {
        let cov = trusted.covariates_row(idx);
        let cif = cumulative_incidence(*exit_age, &cov, &trusted.artifacts).unwrap();
        computed_cif.push(cif);
    }

    // Compare computed values against lifelines reference
    let expected = &trusted.lifelines_cif[..4];
    assert_eq!(computed_cif.len(), expected.len());
    for (computed, expected) in computed_cif.iter().zip(expected.iter()) {
        assert!(
            (computed - expected).abs() <= 1e-10,
            "CIF mismatch: computed={}, expected={}",
            computed,
            expected
        );
    }
}

#[test]
fn brier_score_matches_reference_library() {
    let trusted = TrustedReference::load();

    // Compute predictions using our model
    let mut preds = Array1::<f64>::zeros(trusted.data.age_exit.len());
    for (idx, exit_age) in trusted.data.age_exit.iter().enumerate() {
        let cov = trusted.covariates_row(idx);
        preds[idx] = cumulative_incidence(*exit_age, &cov, &trusted.artifacts).unwrap();
    }

    // Compute weighted Brier score
    let outcomes = trusted.data.event_target.map(|v| f64::from(*v));
    let computed_brier = weighted_brier(&trusted.data.sample_weight, &outcomes, &preds);

    // Compare against lifelines reference
    assert!(
        (computed_brier - trusted.lifelines_weighted_brier).abs() <= 1e-10,
        "Brier score mismatch: computed={}, expected={}",
        computed_brier,
        trusted.lifelines_weighted_brier
    );
}

#[test]
fn conditional_risk_monotonic_with_calibration_toggle() {
    let trusted = TrustedReference::load();
    let covs = trusted.covariates_row(0);
    let t0 = 55.0;
    let horizons = [60.0, 62.0, 64.0, 66.0];
    let mut base = Vec::new();
    let mut calibrated = Vec::new();
    for &t1 in &horizons {
        let raw =
            conditional_absolute_risk(t0, t1, &covs, Some(0.0), None, &trusted.artifacts).unwrap();
        base.push(raw);
        let cal =
            conditional_absolute_risk(t0, t1, &covs, Some(0.12), None, &trusted.artifacts).unwrap();
        calibrated.push(cal);
    }
    assert!(base.windows(2).all(|w| w[1] + 1e-12 >= w[0]));
    assert!(calibrated.windows(2).all(|w| w[1] + 1e-12 >= w[0]));
}

#[test]
fn weighted_brier_matches_frequency_replication() {
    let trusted = TrustedReference::load();
    let mut preds = Array1::<f64>::zeros(trusted.data.age_exit.len());
    for (idx, exit_age) in trusted.data.age_exit.iter().enumerate() {
        let cov = trusted.covariates_row(idx);
        preds[idx] = cumulative_incidence(*exit_age, &cov, &trusted.artifacts).unwrap();
    }
    let outcomes = trusted.data.event_target.map(|v| f64::from(*v));
    let weighted = weighted_brier(&trusted.data.sample_weight, &outcomes, &preds);
    let replicated = replicated_brier(&trusted.data.sample_weight, &outcomes, &preds);
    assert!((weighted - replicated).abs() <= 1e-12);
}
