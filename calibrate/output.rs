//! The prediction tables `gnomon-calibrate infer` writes, one row per sample.

use std::fmt::Display;
use std::io::{self, Write};

use gam::probability::normal_cdf;
use ndarray::Array1;

use crate::calibrate::model::{LinkFunction, SurvivalPrediction};
use crate::calibrate::survival_data::SurvivalPredictionData;

/// Columns of the binary table. calibrate's binary model is gam's Bernoulli
/// marginal-slope fit, whose mean is `Φ(η)` whichever binary link the
/// configuration names, so `η` is written as the probit index it is and its
/// interval is mapped to probabilities through `Φ`.
pub const BINARY_PREDICTION_HEADER: &str = "sample_id\thull_signed_distance\tprobit_index\tstandard_error_probit_index\tprediction\tprobability_lower_95\tprobability_upper_95";

/// Columns of the continuous (Gaussian location-scale) table.
pub const CONTINUOUS_PREDICTION_HEADER: &str =
    "sample_id\thull_signed_distance\tprediction\tstandard_error_mean\tmean_lower_95\tmean_upper_95";

/// Columns of the survival table.
pub const SURVIVAL_PREDICTION_HEADER: &str = "sample_id\tage_entry\tage_exit\tcumulative_hazard_entry\tcumulative_hazard_exit\tcumulative_incidence_entry\tcumulative_incidence_exit\tconditional_risk\tlogit_risk\tlogit_risk_standard_error";

const Z_975: f64 = 1.959964;

fn write_row(out: &mut impl Write, fields: &[&dyn Display]) -> io::Result<()> {
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            write!(out, "\t")?;
        }
        write!(out, "{field}")?;
    }
    writeln!(out)
}

/// Writes the binary or continuous prediction table: one row per sample, in
/// input order.
pub fn write_predictions(
    out: &mut impl Write,
    sample_ids: &[String],
    signed_distance: &Array1<f64>,
    eta: &Array1<f64>,
    mean: &Array1<f64>,
    se_eta: Option<&Array1<f64>>,
    link: LinkFunction,
) -> io::Result<()> {
    let binary = !matches!(link, LinkFunction::Identity);
    let header = if binary {
        BINARY_PREDICTION_HEADER
    } else {
        CONTINUOUS_PREDICTION_HEADER
    };
    writeln!(out, "{header}")?;
    for index in 0..eta.len() {
        let na = || "NA".to_string();
        let (se, lower, upper) = match se_eta {
            None => (na(), na(), na()),
            Some(se_eta) => {
                let se = se_eta[index];
                let center = if binary { eta[index] } else { mean[index] };
                let (low, high) = (center - Z_975 * se, center + Z_975 * se);
                if binary {
                    (
                        se.to_string(),
                        normal_cdf(low).clamp(0.0, 1.0).to_string(),
                        normal_cdf(high).clamp(0.0, 1.0).to_string(),
                    )
                } else {
                    (se.to_string(), low.to_string(), high.to_string())
                }
            }
        };
        let sample_id = &sample_ids[index] as &dyn Display;
        if binary {
            write_row(
                out,
                &[
                    sample_id,
                    &signed_distance[index],
                    &eta[index],
                    &se,
                    &mean[index],
                    &lower,
                    &upper,
                ],
            )?;
        } else {
            write_row(
                out,
                &[sample_id, &signed_distance[index], &mean[index], &se, &lower, &upper],
            )?;
        }
    }
    Ok(())
}

/// Writes the survival prediction table: one row per sample, under its
/// identifier, in input order.
pub fn write_survival_predictions(
    out: &mut impl Write,
    data: &SurvivalPredictionData,
    prediction: &SurvivalPrediction,
) -> io::Result<()> {
    writeln!(out, "{SURVIVAL_PREDICTION_HEADER}")?;
    for index in 0..prediction.conditional_risk.len() {
        let se = prediction
            .logit_risk_se
            .as_ref()
            .map_or_else(|| "NA".to_string(), |values| values[index].to_string());
        write_row(
            out,
            &[
                &data.sample_ids[index] as &dyn Display,
                &data.age_entry[index],
                &data.age_exit[index],
                &prediction.cumulative_hazard_entry[index],
                &prediction.cumulative_hazard_exit[index],
                &prediction.cumulative_incidence_entry[index],
                &prediction.cumulative_incidence_exit[index],
                &prediction.conditional_risk[index],
                &prediction.logit_risk[index],
                &se,
            ],
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn a_binary_table_names_the_probit_index_and_maps_its_interval_through_phi() {
        let eta = array![0.5, -1.25];
        let mean = eta.mapv(normal_cdf);
        let se = array![0.1, 0.2];
        let mut table = Vec::new();
        write_predictions(
            &mut table,
            &["a".to_string(), "b".to_string()],
            &array![0.0, 0.0],
            &eta,
            &mean,
            Some(&se),
            LinkFunction::Logit,
        )
        .expect("write the table");
        let table = String::from_utf8(table).expect("utf-8");
        assert_eq!(table.lines().count(), 3);
        let mut lines = table.lines();
        assert_eq!(lines.next(), Some(BINARY_PREDICTION_HEADER));
        assert!(!BINARY_PREDICTION_HEADER.contains("log_odds"));
        for (index, line) in lines.enumerate() {
            let fields: Vec<&str> = line.split('\t').collect();
            let field = |column: usize| fields[column].parse::<f64>().expect("a number");
            assert_eq!(field(2), eta[index]);
            assert_eq!(field(4), mean[index]);
            assert_eq!(field(5), normal_cdf(eta[index] - Z_975 * se[index]));
            assert_eq!(field(6), normal_cdf(eta[index] + Z_975 * se[index]));
        }
    }
}
