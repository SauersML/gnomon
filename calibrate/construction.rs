//! Formula text and term specifications over the gnomon score / sex / PC
//! domain.
//!
//! gam owns basis construction, identifiability and persistence; this module
//! only names the columns and sizes the smooths. In a marginal-slope fit the
//! score is the latent coordinate (`z_column`), so it never appears among the
//! context terms of the marginal or slope formulas. The Gaussian location-scale
//! fit is requested directly, because gam's formula route refuses its link
//! wiggle, so its terms are built here as specifications that match what the
//! formula route builds from the same text: gam's scale-free default Duchon
//! kernel (no length scale, no power) and the center geometry gam chooses for an
//! explicit count.

use crate::calibrate::model::{PcSmoothConfig, SmoothConfig};
use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonOperatorPenaltySpec, OneDimensionalBoundary,
    SpatialIdentifiability, default_spatial_center_strategy, duchon_cubic_default,
};
use gam::terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TermCollectionSpec,
};

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

/// The Gaussian location-scale channels' terms over the predictor table in
/// `predictor_headers` order (score | sex | PC1..PCk): the score's Duchon
/// smooth, the penalized linear sex term and the joint smooth of the PCs (the
/// context formula's, with its center count), each as the formula route builds
/// it from `score_smooth` and `context_formula`.
pub(crate) fn marginal_termspec(
    score_basis: &SmoothConfig,
    pcs: &PcSmoothConfig,
) -> TermCollectionSpec {
    let score = duchon_term(SCORE_COLUMN, vec![0], score_basis.num_centers, None);
    let joint = (pcs.num_pcs > 0).then(|| {
        duchon_term(
            &pc_columns(pcs.num_pcs).join("_"),
            (2..2 + pcs.num_pcs).collect(),
            pcs.context_centers,
            pcs.power,
        )
    });
    TermCollectionSpec {
        // A bare formula term: the null-recovery ridge is gam's default for a
        // parametric term (SPEC rules 12, 14).
        linear_terms: vec![LinearTermSpec {
            name: SEX_COLUMN.to_string(),
            feature_col: 1,
            feature_cols: vec![1],
            categorical_levels: Vec::new(),
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms: std::iter::once(score).chain(joint).collect(),
    }
}

/// The center geometry gam's formula route gives a Duchon smooth with an
/// explicit count (gam-terms `duchon_center_strategy`): an even grid on the
/// interval in one dimension, otherwise gam's spatial default for the dimension.
fn duchon_center_strategy(num_centers: usize, dimension: usize) -> CenterStrategy {
    if dimension == 1 {
        CenterStrategy::UniformGrid {
            points_per_dim: num_centers,
        }
    } else {
        default_spatial_center_strategy(num_centers, dimension)
    }
}

/// A Duchon term on gam's default kernel for its dimension (`duchon_cubic_default`:
/// affine null space, spectral power `(d - 1)/2`), unless `power` names one.
fn duchon_term(
    name: &str,
    feature_cols: Vec<usize>,
    num_centers: usize,
    power: Option<f64>,
) -> SmoothTermSpec {
    let dimension = feature_cols.len();
    let (nullspace_order, default_power) = duchon_cubic_default(dimension);
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols,
            spec: DuchonBasisSpec {
                center_strategy: duchon_center_strategy(num_centers, dimension),
                periodic: None,
                length_scale: None,
                power: power.unwrap_or(default_power),
                nullspace_order,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
                boundary: OneDimensionalBoundary::Open,
                radial_reparam: None,
            },
            input_scale: None,
        },
        shape: ShapeConstraint::None,
        joint_null_rotation: None,
        frozen_parametric_residualization: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam::terms::smooth::build_term_collection_design;
    use ndarray::Array2;

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

    #[test]
    fn gaussian_terms_read_score_sex_and_the_joint_pc_smooth_from_the_predictor_table() {
        let terms = marginal_termspec(&SmoothConfig { num_centers: 5 }, &PcSmoothConfig::for_pcs(2));
        let smooths: Vec<(Vec<usize>, Option<f64>, f64)> = terms
            .smooth_terms
            .iter()
            .filter_map(|term| match &term.basis {
                SmoothBasisSpec::Duchon { feature_cols, spec, .. } => {
                    Some((feature_cols.clone(), spec.length_scale, spec.power))
                }
                _ => None,
            })
            .collect();
        assert_eq!(smooths, vec![(vec![0], None, 0.0), (vec![2, 3], None, 0.5)]);
        assert_eq!(terms.smooth_terms.len(), 2);
        assert_eq!(terms.linear_terms.len(), 1);
        assert_eq!(terms.linear_terms[0].name, SEX_COLUMN);
        assert_eq!(terms.linear_terms[0].feature_col, 1);
    }

    /// gam builds the joint smooth at its default kernel and the derived center
    /// counts for every PC count up to 16.
    #[test]
    fn gam_builds_the_default_joint_pc_smooth_at_every_pc_count_to_sixteen() {
        let rows = 96;
        for num_pcs in 1..=16 {
            let pcs = PcSmoothConfig::for_pcs(num_pcs);
            pcs.validate().expect("the derived configuration is valid");
            let columns = 2 + num_pcs;
            let data = Array2::from_shape_fn((rows, columns), |(row, column)| {
                let phase = (row * (column + 3) + 7 * column) as f64;
                (phase * 0.618_033_988_749_895).fract() * 2.0 - 1.0
            });
            for centers in [pcs.context_centers, pcs.slope_centers] {
                let terms = marginal_termspec(
                    &SmoothConfig { num_centers: 5 },
                    &PcSmoothConfig { context_centers: centers, ..pcs },
                );
                build_term_collection_design(data.view(), &terms).unwrap_or_else(|error| {
                    panic!("{num_pcs} PCs with {centers} centers: {error}")
                });
            }
        }
    }

    /// The Gaussian model's term specifications are the ones gam's formula route
    /// builds from the same text: same center geometry, kernel power, null space
    /// and length scale, for the score and for the joint smooth of 16 PCs.
    #[test]
    fn gaussian_terms_match_what_the_formula_route_builds_from_the_same_text() {
        use gam::data::{ColumnKindTag, DataSchema, EncodedDataset, SchemaColumn};
        use gam::terms::inference::formula_dsl::parse_formula;
        use gam::terms::term_builder::build_termspec;
        use std::collections::HashMap;

        let pcs = PcSmoothConfig::for_pcs(16);
        let score_basis = SmoothConfig { num_centers: 10 };
        let mut headers = vec![SCORE_COLUMN.to_string(), SEX_COLUMN.to_string()];
        headers.extend(pc_columns(16));
        let rows = 96;
        let values = Array2::from_shape_fn((rows, headers.len()), |(row, column)| {
            if column == 1 {
                (row % 2) as f64
            } else {
                ((row * (column + 3) + 7 * column) as f64 * 0.618_033_988_749_895).fract()
            }
        });
        let dataset = EncodedDataset {
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
            headers: headers.clone(),
            values,
        };
        let columns: HashMap<String, usize> =
            headers.iter().enumerate().map(|(index, name)| (name.clone(), index)).collect();
        let formula = format!("y ~ {} + {}", score_smooth(&score_basis), context_formula(&pcs));
        let parsed = parse_formula(&formula).expect("parse the formula");
        let from_formula = build_termspec(&parsed.terms, &dataset, &columns, &mut Vec::new())
            .expect("the formula route's terms");
        let kernel = |terms: &TermCollectionSpec| -> Vec<String> {
            terms
                .smooth_terms
                .iter()
                .filter_map(|term| match &term.basis {
                    SmoothBasisSpec::Duchon { feature_cols, spec, .. } => Some(format!(
                        "{feature_cols:?} {:?} power={} {:?} length_scale={:?}",
                        spec.center_strategy, spec.power, spec.nullspace_order, spec.length_scale
                    )),
                    _ => None,
                })
                .collect()
        };
        assert_eq!(kernel(&marginal_termspec(&score_basis, &pcs)), kernel(&from_formula));
        assert_eq!(kernel(&from_formula).len(), 2);
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
