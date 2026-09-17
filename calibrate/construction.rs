//! Formula text and term specifications over the gnomon score / sex / PC
//! domain.
//!
//! gam owns basis construction, identifiability and persistence; this module
//! only names the columns and sizes the smooths. In a marginal-slope fit the
//! score is the latent coordinate (`z_column`), so it never appears among the
//! context terms of the marginal or slope formulas. The Gaussian location-scale
//! fit is requested directly, because gam's formula route refuses its link
//! wiggle, so its terms are built here as specifications with the same Duchon
//! kernel as the formula smooths.

use crate::calibrate::model::SmoothConfig;
use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    OneDimensionalBoundary, SpatialIdentifiability,
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

/// Every calibration Duchon smooth uses the hybrid Duchon-Matérn kernel at a
/// fixed unit length scale and unit power, the kernel gnomon has always
/// requested.
const DUCHON_LENGTH_SCALE: f64 = 1.0;
const DUCHON_POWER: f64 = 1.0;

/// A one-dimensional Duchon smooth over `column` with `num_centers` centers.
pub(crate) fn duchon_smooth(column: &str, num_centers: usize) -> String {
    format!(
        "s({column}, type=duchon, centers={num_centers}, power={DUCHON_POWER}, length_scale={DUCHON_LENGTH_SCALE})"
    )
}

fn pc_smooths(pc_bases: &[SmoothConfig]) -> impl Iterator<Item = String> + '_ {
    pc_bases
        .iter()
        .enumerate()
        .map(|(index, basis)| duchon_smooth(&format!("PC{}", index + 1), basis.num_centers))
}

/// The context of a row: a penalized linear sex term and one Duchon smooth
/// per principal component. Tensor PC interactions are intentionally omitted.
pub(crate) fn context_formula(pc_bases: &[SmoothConfig]) -> String {
    std::iter::once(SEX_COLUMN.to_string())
        .chain(pc_smooths(pc_bases))
        .collect::<Vec<_>>()
        .join(" + ")
}

/// The slope of the score along ancestry: an intercept and one Duchon smooth
/// per principal component, so a model without PCs has a constant slope.
pub(crate) fn slope_formula(pc_bases: &[SmoothConfig]) -> String {
    std::iter::once("1".to_string())
        .chain(pc_smooths(pc_bases))
        .collect::<Vec<_>>()
        .join(" + ")
}

/// The Gaussian location-scale channels' terms over the predictor table in
/// `predictor_headers` order (score | sex | PC1..PCk): a Duchon smooth of the
/// score, the penalized linear sex term and one Duchon smooth per PC.
pub(crate) fn marginal_termspec(
    score_basis: &SmoothConfig,
    pc_bases: &[SmoothConfig],
) -> TermCollectionSpec {
    let smooth_terms = std::iter::once(duchon_term(SCORE_COLUMN, 0, score_basis.num_centers))
        .chain(
            pc_bases
                .iter()
                .enumerate()
                .map(|(index, basis)| {
                    duchon_term(&format!("PC{}", index + 1), 2 + index, basis.num_centers)
                }),
        )
        .collect();
    TermCollectionSpec {
        linear_terms: vec![LinearTermSpec {
            name: SEX_COLUMN.to_string(),
            feature_col: 1,
            feature_cols: Vec::new(),
            categorical_levels: Vec::new(),
            double_penalty: true,
            coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
            coefficient_min: None,
            coefficient_max: None,
            frozen_function_mass: None,
        }],
        random_effect_terms: Vec::new(),
        smooth_terms,
    }
}

fn duchon_term(name: &str, feature_col: usize, num_centers: usize) -> SmoothTermSpec {
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: vec![feature_col],
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint { num_centers },
                periodic: None,
                length_scale: Some(DUCHON_LENGTH_SCALE),
                power: DUCHON_POWER,
                nullspace_order: DuchonNullspaceOrder::Linear,
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

    #[test]
    fn formulas_name_every_pc_and_keep_the_score_out_of_the_context() {
        let bases = [SmoothConfig { num_centers: 6 }, SmoothConfig { num_centers: 4 }];
        assert_eq!(
            context_formula(&bases),
            "sex + s(PC1, type=duchon, centers=6, power=1, length_scale=1) + s(PC2, type=duchon, centers=4, power=1, length_scale=1)"
        );
        assert_eq!(
            slope_formula(&bases),
            "1 + s(PC1, type=duchon, centers=6, power=1, length_scale=1) + s(PC2, type=duchon, centers=4, power=1, length_scale=1)"
        );
        assert_eq!(context_formula(&[]), "sex");
        assert_eq!(slope_formula(&[]), "1");
        assert!(!context_formula(&bases).contains(SCORE_COLUMN));
    }

    #[test]
    fn gaussian_terms_read_score_sex_and_each_pc_from_the_predictor_table() {
        let terms = marginal_termspec(
            &SmoothConfig { num_centers: 5 },
            &[SmoothConfig { num_centers: 4 }],
        );
        let smooths: Vec<(&str, Vec<usize>, Option<f64>, f64)> = terms
            .smooth_terms
            .iter()
            .filter_map(|term| match &term.basis {
                SmoothBasisSpec::Duchon { feature_cols, spec, .. } => Some((
                    term.name.as_str(),
                    feature_cols.clone(),
                    spec.length_scale,
                    spec.power,
                )),
                _ => None,
            })
            .collect();
        assert_eq!(
            smooths,
            vec![
                (SCORE_COLUMN, vec![0], Some(1.0), 1.0),
                ("PC1", vec![2], Some(1.0), 1.0),
            ]
        );
        assert_eq!(terms.smooth_terms.len(), 2);
        assert_eq!(terms.linear_terms.len(), 1);
        assert_eq!(terms.linear_terms[0].name, SEX_COLUMN);
        assert_eq!(terms.linear_terms[0].feature_col, 1);
    }
}
