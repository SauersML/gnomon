use super::age::GuardedLogAge;
use super::data::SurvivalTrainingData;
use super::model_family::SurvivalSpec;
use super::penalties::{PenaltyBlocks, PenaltyDescriptor};
use crate::calibrate::basis::{self, BasisError};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasisDescriptor {
    pub degree: usize,
    pub knots: Array1<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceConstraint {
    pub transform: Array2<f64>,
    pub pivot: usize,
    pub reference_age: f64,
}

#[derive(Debug, Clone)]
pub struct SurvivalLayout {
    pub baseline_entry: Array2<f64>,
    pub baseline_exit: Array2<f64>,
    pub baseline_derivative_exit: Array2<f64>,
    pub time_varying_entry: Option<Array2<f64>>,
    pub time_varying_exit: Option<Array2<f64>>,
    pub time_varying_derivative_exit: Option<Array2<f64>>,
    pub static_covariates: Array2<f64>,
    pub static_covariate_names: Vec<String>,
    pub age_descriptor: BasisDescriptor,
    pub time_varying_descriptor: Option<BasisDescriptor>,
    pub reference_constraint: ReferenceConstraint,
    pub penalties: PenaltyBlocks,
}

#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("basis error: {0}")]
    Basis(#[from] BasisError),
    #[error("reference constraint encountered a zero pivot")]
    ReferenceConstraintSingular,
}

pub fn build_survival_layout(
    data: &SurvivalTrainingData,
    spec: &SurvivalSpec,
) -> Result<SurvivalLayout, LayoutError> {
    let guarded_age = GuardedLogAge::new(data.age_transform, &data.age_entry, &data.age_exit);
    let (static_covariates, static_names) = assemble_static_covariates(data);
    let builder = SurvivalLayoutBuilder::new(
        guarded_age,
        static_covariates,
        static_names,
        spec.basis_knots,
        spec.basis_degree,
        spec.penalty_order,
    );
    builder.build()
}

fn assemble_static_covariates(data: &SurvivalTrainingData) -> (Array2<f64>, Vec<String>) {
    let n = data.age_entry.len();
    let num_pcs = data.pcs.ncols();
    let mut matrix = Array2::zeros((n, 3 + num_pcs));
    matrix.column_mut(0).fill(1.0);
    matrix.column_mut(1).assign(&data.pgs);
    matrix.column_mut(2).assign(&data.sex);
    for pc_idx in 0..num_pcs {
        let target_col = 3 + pc_idx;
        matrix
            .column_mut(target_col)
            .assign(&data.pcs.column(pc_idx).to_owned());
    }
    let mut names = vec![
        "intercept".to_string(),
        "pgs".to_string(),
        "sex".to_string(),
    ];
    for pc_idx in 0..num_pcs {
        names.push(format!("pc{}", pc_idx + 1));
    }
    (matrix, names)
}

pub struct SurvivalLayoutBuilder {
    guarded_age: GuardedLogAge,
    basis_degree: usize,
    num_internal_knots: usize,
    penalty_order: usize,
    static_covariates: Array2<f64>,
    static_names: Vec<String>,
}

impl SurvivalLayoutBuilder {
    pub fn new(
        guarded_age: GuardedLogAge,
        static_covariates: Array2<f64>,
        static_names: Vec<String>,
        num_internal_knots: usize,
        basis_degree: usize,
        penalty_order: usize,
    ) -> Self {
        Self {
            guarded_age,
            static_covariates,
            static_names,
            basis_degree,
            num_internal_knots,
            penalty_order,
        }
    }

    pub fn build(&self) -> Result<SurvivalLayout, LayoutError> {
        let descriptor = self.build_basis_descriptor()?;
        let (baseline_entry, baseline_exit) = self.build_baseline_matrices(&descriptor)?;
        let derivative_exit = self.build_derivative_matrix(&descriptor, &baseline_exit)?;
        let reference_constraint = self.build_reference_constraint(&descriptor)?;

        let baseline_entry = baseline_entry.dot(&reference_constraint.transform);
        let baseline_exit = baseline_exit.dot(&reference_constraint.transform);
        let baseline_derivative_exit = derivative_exit.dot(&reference_constraint.transform);

        let static_covariates = self.static_covariates.clone();

        let penalties = PenaltyBlocks::difference_penalty(
            baseline_exit.ncols(),
            self.penalty_order,
            PenaltyDescriptor::Baseline,
            0,
        )?;

        Ok(SurvivalLayout {
            baseline_entry,
            baseline_exit,
            baseline_derivative_exit,
            time_varying_entry: None,
            time_varying_exit: None,
            time_varying_derivative_exit: None,
            static_covariates,
            static_covariate_names: self.static_names.clone(),
            age_descriptor: descriptor,
            time_varying_descriptor: None,
            reference_constraint,
            penalties,
        })
    }

    fn build_basis_descriptor(&self) -> Result<BasisDescriptor, BasisError> {
        let min_log_age = self
            .guarded_age
            .entry_log_age
            .iter()
            .chain(self.guarded_age.exit_log_age.iter())
            .fold(f64::INFINITY, |acc, &v| acc.min(v));
        let max_log_age = self
            .guarded_age
            .entry_log_age
            .iter()
            .chain(self.guarded_age.exit_log_age.iter())
            .fold(f64::NEG_INFINITY, |acc, &v| acc.max(v));
        let (_, knots) = basis::create_bspline_basis(
            self.guarded_age.exit_log_age.view(),
            (min_log_age, max_log_age),
            self.num_internal_knots,
            self.basis_degree,
        )?;
        Ok(BasisDescriptor {
            degree: self.basis_degree,
            knots,
        })
    }

    fn build_baseline_matrices(
        &self,
        descriptor: &BasisDescriptor,
    ) -> Result<(Array2<f64>, Array2<f64>), LayoutError> {
        let (exit, _) = basis::create_bspline_basis_with_knots(
            self.guarded_age.exit_log_age.view(),
            descriptor.knots.view(),
            descriptor.degree,
        )?;
        let (entry, _) = basis::create_bspline_basis_with_knots(
            self.guarded_age.entry_log_age.view(),
            descriptor.knots.view(),
            descriptor.degree,
        )?;
        Ok(((*entry).clone(), (*exit).clone()))
    }

    fn build_derivative_matrix(
        &self,
        descriptor: &BasisDescriptor,
        baseline_exit: &Array2<f64>,
    ) -> Result<Array2<f64>, LayoutError> {
        if descriptor.degree == 0 {
            return Ok(Array2::zeros(baseline_exit.raw_dim()));
        }

        let (basis_minus_one, _) = basis::create_bspline_basis_with_knots(
            self.guarded_age.exit_log_age.view(),
            descriptor.knots.view(),
            descriptor.degree - 1,
        )?;
        let basis_minus_one = (*basis_minus_one).clone();
        let num_rows = basis_minus_one.nrows();
        let num_cols = baseline_exit.ncols();
        let mut derivative = Array2::zeros((num_rows, num_cols));
        for i in 0..num_cols {
            let denom_left = descriptor.knots[i + descriptor.degree] - descriptor.knots[i];
            let denom_right = descriptor.knots[i + descriptor.degree + 1] - descriptor.knots[i + 1];
            for row in 0..num_rows {
                let left = if denom_left.abs() > 0.0 {
                    descriptor.degree as f64 / denom_left * basis_minus_one[[row, i]]
                } else {
                    0.0
                };
                let right = if denom_right.abs() > 0.0 {
                    descriptor.degree as f64 / denom_right * basis_minus_one[[row, i + 1]]
                } else {
                    0.0
                };
                derivative[[row, i]] = (left - right) * self.guarded_age.exit_derivative_scale[row];
            }
        }
        Ok(derivative)
    }

    fn build_reference_constraint(
        &self,
        descriptor: &BasisDescriptor,
    ) -> Result<ReferenceConstraint, LayoutError> {
        let reference_age = self.guarded_age.exit_log_age.mean().unwrap_or(0.0);
        let basis_at_reference = {
            let (basis, _) = basis::create_bspline_basis_with_knots(
                Array1::from_vec(vec![reference_age]).view(),
                descriptor.knots.view(),
                descriptor.degree,
            )?;
            (*basis).row(0).to_owned()
        };
        let pivot = basis_at_reference
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
            .map(|(idx, _)| idx)
            .ok_or(LayoutError::ReferenceConstraintSingular)?;
        let pivot_value = basis_at_reference[pivot];
        if pivot_value.abs() < 1e-12 {
            return Err(LayoutError::ReferenceConstraintSingular);
        }
        let num_cols = basis_at_reference.len();
        let mut constrained = Array2::zeros((num_cols, num_cols - 1));
        let mut dest_col = 0;
        for src_col in 0..num_cols {
            if src_col == pivot {
                continue;
            }
            for row in 0..num_cols {
                if row == src_col {
                    constrained[[row, dest_col]] = 1.0;
                }
            }
            constrained[[pivot, dest_col]] = -basis_at_reference[src_col] / pivot_value;
            dest_col += 1;
        }
        Ok(ReferenceConstraint {
            transform: constrained,
            pivot,
            reference_age,
        })
    }
}
