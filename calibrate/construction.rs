//! Helpers that assemble `gam::terms::smooth::TermCollectionSpec` instances
//! for the gnomon PGS / PC / sex domain.
//!
//! These builders are intentionally thin: gam owns the basis machinery; this
//! module just chooses sensible defaults (Duchon kernels for smooths, a
//! penalized linear term for sex) and wires column indices through.

use crate::calibrate::model::BasisConfig;
use gam::terms::basis::{
    CenterStrategy, DuchonBasisSpec, DuchonNullspaceOrder, DuchonOperatorPenaltySpec,
    SpatialIdentifiability,
};
use gam::terms::smooth::{
    LinearCoefficientGeometry, LinearTermSpec, ShapeConstraint, SmoothBasisSpec, SmoothTermSpec,
    TermCollectionSpec,
};

/// Build a single 1D Duchon smooth term over `feature_col` with `num_centers`
/// farthest-point centers and a linear nullspace.
pub fn duchon_smooth(name: &str, feature_col: usize, num_centers: usize) -> SmoothTermSpec {
    let centers = num_centers.max(4);
    SmoothTermSpec {
        name: name.to_string(),
        basis: SmoothBasisSpec::Duchon {
            feature_cols: vec![feature_col],
            spec: DuchonBasisSpec {
                center_strategy: CenterStrategy::FarthestPoint {
                    num_centers: centers,
                },
                length_scale: Some(1.0),
                power: 1,
                nullspace_order: DuchonNullspaceOrder::Linear,
                identifiability: SpatialIdentifiability::default(),
                aniso_log_scales: None,
                operator_penalties: DuchonOperatorPenaltySpec::default(),
            },
            input_scales: None,
        },
        shape: ShapeConstraint::None,
    }
}

/// Build the marginal `TermCollectionSpec` for the gnomon PGS / PC / sex
/// domain.
///
/// Layout: one Duchon smooth on PGS, one Duchon smooth per principal
/// component, plus a penalized linear term for sex. Tensor PGS x PC
/// interactions are intentionally omitted in v1 — the marginal-slope warp +
/// link wiggle in the gam workflow already capture PGS-by-covariate
/// departures from linearity.
pub fn build_marginal_termspec(
    pgs_col: usize,
    sex_col: usize,
    pc_cols: &[usize],
    pgs_basis: &BasisConfig,
    pc_bases: &[BasisConfig],
) -> TermCollectionSpec {
    assert_eq!(
        pc_cols.len(),
        pc_bases.len(),
        "pc_cols and pc_bases must have matching length",
    );

    let mut smooth_terms = Vec::with_capacity(1 + pc_cols.len());
    smooth_terms.push(duchon_smooth("pgs", pgs_col, pgs_basis.num_knots));
    for (idx, (&col, basis)) in pc_cols.iter().zip(pc_bases.iter()).enumerate() {
        let name = format!("pc{}", idx + 1);
        smooth_terms.push(duchon_smooth(&name, col, basis.num_knots));
    }

    let linear_terms = vec![LinearTermSpec {
        name: "sex".to_string(),
        feature_col: sex_col,
        double_penalty: true,
        coefficient_geometry: LinearCoefficientGeometry::Unconstrained,
        coefficient_min: None,
        coefficient_max: None,
    }];

    TermCollectionSpec {
        linear_terms,
        random_effect_terms: Vec::new(),
        smooth_terms,
    }
}

/// Build a `TermCollectionSpec` for the log-slope channel: a single Duchon
/// smooth over PGS, no linear terms.
pub fn build_logslope_termspec(pgs_col: usize) -> TermCollectionSpec {
    TermCollectionSpec {
        linear_terms: Vec::new(),
        random_effect_terms: Vec::new(),
        smooth_terms: vec![duchon_smooth("pgs_logslope", pgs_col, 8)],
    }
}
