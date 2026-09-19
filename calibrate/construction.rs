//! Formula text and term specifications over the gnomon score / sex / PC
//! domain.
//!
//! gam owns basis construction, identifiability and persistence; this module
//! only names the columns and sizes the smooths. In a marginal-slope fit the
//! score is the latent coordinate (`z_column`), so it never appears among the
//! context terms of the marginal or slope formulas. The Gaussian location-scale
//! fit is requested directly, because gam's formula route refuses its link
//! wiggle, but its terms are gam's own term builder applied to the model's
//! formula text, so the fitted terms are exactly the saved formula's.

use std::collections::HashMap;

use crate::calibrate::model::{PcSmoothConfig, SmoothConfig};
use gam::data::EncodedDataset;
use gam::solver::fit_orchestration::WorkflowError;
use gam::terms::inference::formula_dsl::parse_formula;
use gam::terms::smooth::TermCollectionSpec;
use gam::terms::term_builder::build_termspec;

pub(crate) const SCORE_COLUMN: &str = "score";
pub(crate) const SEX_COLUMN: &str = "sex";
pub(crate) const PHENOTYPE_COLUMN: &str = "phenotype";
pub(crate) const WEIGHT_COLUMN: &str = "weight";
pub(crate) const AGE_ENTRY_COLUMN: &str = "age_entry";
pub(crate) const AGE_EXIT_COLUMN: &str = "age_exit";
pub(crate) const EVENT_COLUMN: &str = "event_target";

/// A Duchon smooth over `columns` with an explicit center count and gam's
/// default kernel, unless `power` names one.
fn duchon_smooth(columns: &[String], centers: usize, power: Option<f64>) -> String {
    let power = power.map_or_else(String::new, |power| format!(", power={power}"));
    format!("s({}, type=duchon, centers={centers}{power})", columns.join(", "))
}

/// The Gaussian model's smooth of the score. Its center count is explicit
/// because the direct location-scale request takes term specifications, which
/// carry one; it is the user's `--pgs-centers`.
pub(crate) fn score_smooth(score_basis: &SmoothConfig) -> String {
    duchon_smooth(&[SCORE_COLUMN.to_string()], score_basis.num_centers, None)
}

fn pc_columns(num_pcs: usize) -> Vec<String> {
    (1..=num_pcs).map(|index| format!("PC{index}")).collect()
}

/// The joint Duchon smooth over `PC1..PCk` with `centers` centers; `None`
/// without PCs.
fn joint_pc_smooth(pcs: &PcSmoothConfig, centers: usize) -> Option<String> {
    (pcs.num_pcs > 0).then(|| duchon_smooth(&pc_columns(pcs.num_pcs), centers, pcs.power))
}

/// The context of a row: a penalized linear sex term and the joint smooth of
/// the principal components.
pub(crate) fn context_formula(pcs: &PcSmoothConfig) -> String {
    std::iter::once(SEX_COLUMN.to_string())
        .chain(joint_pc_smooth(pcs, pcs.context_centers))
        .collect::<Vec<_>>()
        .join(" + ")
}

/// The slope of the score along ancestry: an intercept and the joint smooth of
/// the principal components, so a model without PCs has a constant slope.
pub(crate) fn slope_formula(pcs: &PcSmoothConfig) -> String {
    std::iter::once("1".to_string())
        .chain(joint_pc_smooth(pcs, pcs.slope_centers))
        .collect::<Vec<_>>()
        .join(" + ")
}

/// The right-hand side of the Gaussian location-scale model, shared by its mean
/// and log-scale channels: the score's smooth and the context.
pub(crate) fn gaussian_rhs(score_basis: &SmoothConfig, pcs: &PcSmoothConfig) -> String {
    format!("{} + {}", score_smooth(score_basis), context_formula(pcs))
}

/// The Gaussian location-scale channels' terms over `dataset`, built by gam's own
/// term builder from `gaussian_rhs`, exactly as gam's formula route builds them
/// (no scale dimensions, no overrides: gam's defaults). One route, so the terms
/// fitted cannot drift from the saved formula (#2393).
pub(crate) fn gaussian_termspec(
    score_basis: &SmoothConfig,
    pcs: &PcSmoothConfig,
    dataset: &EncodedDataset,
) -> Result<TermCollectionSpec, WorkflowError> {
    let parsed = parse_formula(&format!("{PHENOTYPE_COLUMN} ~ {}", gaussian_rhs(score_basis, pcs)))?;
    let columns: HashMap<String, usize> = dataset
        .headers
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), index))
        .collect();
    Ok(build_termspec(&parsed.terms, dataset, &columns, &mut Vec::new())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam::data::{ColumnKindTag, DataSchema, SchemaColumn};
    use gam::terms::basis::DuchonOperatorPenaltySpec;
    use gam::terms::smooth::{SmoothBasisSpec, build_term_collection_design};
    use ndarray::Array2;

    /// A predictor table in `predictor_headers` order (score | sex | PC1..PCk).
    fn predictor_table(rows: usize, num_pcs: usize) -> EncodedDataset {
        let mut headers = vec![SCORE_COLUMN.to_string(), SEX_COLUMN.to_string()];
        headers.extend(pc_columns(num_pcs));
        let values = Array2::from_shape_fn((rows, headers.len()), |(row, column)| {
            if column == 1 {
                (row % 2) as f64
            } else {
                ((row * (column + 3) + 7 * column) as f64 * 0.618_033_988_749_895).fract() * 2.0 - 1.0
            }
        });
        EncodedDataset {
            schema: DataSchema {
                columns: headers
                    .iter()
                    .map(|name| SchemaColumn {
                        name: name.clone(),
                        kind: ColumnKindTag::Continuous,
                        levels: Vec::new(),
                    })
                    .collect(),
            },
            column_kinds: vec![ColumnKindTag::Continuous; headers.len()],
            headers,
            values,
        }
    }

    fn duchon_terms(terms: &TermCollectionSpec) -> Vec<(Vec<usize>, Option<f64>, f64, String)> {
        terms
            .smooth_terms
            .iter()
            .filter_map(|term| match &term.basis {
                SmoothBasisSpec::Duchon { feature_cols, spec, .. } => Some((
                    feature_cols.clone(),
                    spec.length_scale,
                    spec.power,
                    format!("{:?}", spec.operator_penalties),
                )),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn formulas_carry_one_joint_pc_smooth_and_keep_the_score_out_of_the_context() {
        let pcs = PcSmoothConfig::for_pcs(3);
        assert_eq!(
            context_formula(&pcs),
            "sex + s(PC1, PC2, PC3, type=duchon, centers=5)"
        );
        assert_eq!(slope_formula(&pcs), "1 + s(PC1, PC2, PC3, type=duchon, centers=5)");
        let sixteen = PcSmoothConfig::for_pcs(16);
        assert_eq!((sixteen.context_centers, sixteen.slope_centers), (24, 20));
        assert_eq!(context_formula(&PcSmoothConfig::for_pcs(0)), "sex");
        assert_eq!(slope_formula(&PcSmoothConfig::for_pcs(0)), "1");
        let powered = PcSmoothConfig { power: Some(1.0), ..pcs };
        powered.validate().expect("power 1 is admissible at 3 PCs");
        assert_eq!(
            slope_formula(&powered),
            "1 + s(PC1, PC2, PC3, type=duchon, centers=5, power=1)"
        );
        assert_eq!(score_smooth(&SmoothConfig { num_centers: 10 }), "s(score, type=duchon, centers=10)");
        assert!(!context_formula(&pcs).contains(SCORE_COLUMN));
    }

    /// The Gaussian terms are gam's formula route's: the score's and the joint PC
    /// smooth on gam's default kernel, with none of the collocated mass or tension
    /// penalties the formula route leaves off (#2393), and the linear sex term.
    #[test]
    fn gaussian_terms_are_the_formula_routes_terms() {
        let table = predictor_table(96, 2);
        let terms =
            gaussian_termspec(&SmoothConfig { num_centers: 5 }, &PcSmoothConfig::for_pcs(2), &table)
                .expect("gam builds the terms");
        let off = format!("{:?}", DuchonOperatorPenaltySpec::all_disabled());
        assert_eq!(
            duchon_terms(&terms),
            vec![(vec![0], None, 0.0, off.clone()), (vec![2, 3], None, 0.5, off)]
        );
        assert_eq!(terms.smooth_terms.len(), 2);
        assert_eq!(terms.linear_terms.len(), 1);
        assert_eq!(terms.linear_terms[0].feature_col, 1);
    }

    /// gam builds the joint smooth at its default kernel and the derived center
    /// counts for every PC count up to 16, in the context and in the slope.
    #[test]
    fn gam_builds_the_default_joint_pc_smooth_at_every_pc_count_to_sixteen() {
        for num_pcs in 1..=16 {
            let pcs = PcSmoothConfig::for_pcs(num_pcs);
            pcs.validate().expect("the derived configuration is valid");
            let table = predictor_table(96, num_pcs);
            for centers in [pcs.context_centers, pcs.slope_centers] {
                let terms = gaussian_termspec(
                    &SmoothConfig { num_centers: 5 },
                    &PcSmoothConfig { context_centers: centers, ..pcs },
                    &table,
                )
                .unwrap_or_else(|error| panic!("{num_pcs} PCs with {centers} centers: {error}"));
                build_term_collection_design(table.values.view(), &terms).unwrap_or_else(|error| {
                    panic!("{num_pcs} PCs with {centers} centers: {error}")
                });
            }
        }
    }

    #[test]
    fn a_joint_pc_smooth_gam_would_refuse_is_refused_by_name_first() {
        let pcs = PcSmoothConfig::for_pcs(16);
        let too_few = PcSmoothConfig { slope_centers: 17, ..pcs };
        let error = too_few.validate().expect_err("17 centers for 16 PCs");
        assert!(error.contains("slope has 17 centers for 16 PCs"), "{error}");
        let too_rough = PcSmoothConfig { power: Some(1.0), ..pcs };
        let error = too_rough.validate().expect_err("power 1 in 16 dimensions");
        assert!(error.contains("power 1 leaves 2(p + s) = 6"), "{error}");
        // gam's default at 16 PCs is s = 7.5: 2(p + s) = 19 clears the D2 margin 18.
        assert!(PcSmoothConfig { power: Some(7.5), ..pcs }.validate().is_ok());
        let at_the_margin = PcSmoothConfig { power: Some(7.0), ..pcs }.validate().expect_err("s = 7");
        assert!(at_the_margin.contains("2(p + s) = 18 at or below"), "{at_the_margin}");
        let not_cpd = PcSmoothConfig { power: Some(8.0), ..pcs }.validate().expect_err("s = 8");
        assert!(not_cpd.contains("not below half its dimension 16"), "{not_cpd}");
    }
}
