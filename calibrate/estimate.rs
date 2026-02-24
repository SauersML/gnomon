pub use gam::estimate::{
    EstimationError, ExternalOptimOptions, ExternalOptimResult, FitOptions, FitResult,
    evaluate_external_cost_and_ridge, evaluate_external_gradients, fit_gam,
    optimize_external_design,
};

pub use gam::estimate::internal;
use gam::basis::{BasisOptions, Dense, KnotSource, create_basis};
use gam::pirls::{self, PirlsStatus, WorkingModelPirlsOptions};
use gam::types::Coefficients;
use ndarray::Array1;
use std::collections::HashMap;
use wolfe_bfgs::Bfgs;

fn map_survival_error(err: crate::calibrate::survival::SurvivalError) -> EstimationError {
    EstimationError::InvalidSpecification(err.to_string())
}

fn finite_diff_gradient<F>(
    z: &Array1<f64>,
    step: f64,
    objective: &F,
) -> Result<Array1<f64>, EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<f64, EstimationError>,
{
    let mut grad = Array1::<f64>::zeros(z.len());
    for i in 0..z.len() {
        let mut zp = z.clone();
        zp[i] += step;
        let fp = objective(&zp)?;
        let mut zm = z.clone();
        zm[i] -= step;
        let fm = objective(&zm)?;
        grad[i] = (fp - fm) / (2.0 * step);
    }
    Ok(grad)
}

pub fn train_model(
    data: &crate::calibrate::data::TrainingData,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
    let (
        x,
        s_list,
        layout,
        sum_to_zero_constraints,
        knot_vectors,
        range_transforms,
        pc_null_transforms,
        interaction_centering_means,
        interaction_orth_alpha,
        _penalty_structs,
    ) = crate::calibrate::construction::build_design_and_penalty_matrices(data, config)?;

    let family = match config.model_family {
        crate::calibrate::model::ModelFamily::Gam(gam::types::LinkFunction::Identity) => {
            gam::types::LikelihoodFamily::GaussianIdentity
        }
        crate::calibrate::model::ModelFamily::Gam(gam::types::LinkFunction::Logit) => {
            gam::types::LikelihoodFamily::BinomialLogit
        }
        crate::calibrate::model::ModelFamily::Gam(gam::types::LinkFunction::Probit) => {
            gam::types::LikelihoodFamily::BinomialProbit
        }
        crate::calibrate::model::ModelFamily::Survival(_) => {
            return Err(EstimationError::InvalidInput(
                "train_model expects GAM family; use train_survival_model for survival".to_string(),
            ));
        }
    };

    let opts = FitOptions {
        max_iter: config.reml_max_iterations as usize,
        tol: config.reml_convergence_tolerance,
        nullspace_dims: vec![0; s_list.len()],
    };
    let offset = Array1::<f64>::zeros(data.y.len());
    let fit = fit_gam(
        x.view(),
        data.y.view(),
        data.weights.view(),
        offset.view(),
        &s_list,
        family,
        &opts,
    )?;

    let mut trained_config = config.clone();
    trained_config.sum_to_zero_constraints = sum_to_zero_constraints;
    trained_config.knot_vectors = knot_vectors;
    trained_config.range_transforms = range_transforms;
    trained_config.pc_null_transforms = pc_null_transforms;
    trained_config.interaction_centering_means = interaction_centering_means;
    trained_config.interaction_orth_alpha = interaction_orth_alpha;

    let coefficients = crate::calibrate::model::map_coefficients(&fit.beta, &layout)?;

    Ok(crate::calibrate::model::TrainedModel {
        config: trained_config,
        coefficients,
        lambdas: fit.lambdas.to_vec(),
        hull: None,
        penalized_hessian: None,
        scale: Some(fit.scale),
        calibrator: None,
        joint_link: None,
        survival: None,
        survival_companions: HashMap::new(),
        mcmc_samples: None,
        smoothing_correction: None,
    })
}

pub fn train_survival_model(
    bundle: &crate::calibrate::survival_data::SurvivalTrainingBundle,
    config: &crate::calibrate::model::ModelConfig,
) -> Result<crate::calibrate::model::TrainedModel, EstimationError> {
    use crate::calibrate::model::ModelFamily;
    use crate::calibrate::survival::{
        BasisDescriptor, CovariateLayout, SurvivalLayoutBundle, SurvivalModelArtifacts,
        TensorProductConfig, WorkingModelSurvival, build_survival_layout,
    };

    let survival_cfg = config.survival.as_ref().ok_or_else(|| {
        EstimationError::InvalidSpecification(
            "missing survival config for survival training".to_string(),
        )
    })?;
    let survival_spec = match config.model_family {
        ModelFamily::Survival(spec) => spec,
        _ => {
            return Err(EstimationError::InvalidInput(
                "train_survival_model expects Survival model family".to_string(),
            ));
        }
    };

    let log_entry = bundle
        .age_transform
        .transform_array(&bundle.data.age_entry)
        .map_err(map_survival_error)?;
    let mut min_log = f64::INFINITY;
    let mut max_log = f64::NEG_INFINITY;
    for &v in log_entry.iter() {
        min_log = min_log.min(v);
        max_log = max_log.max(v);
    }
    if !min_log.is_finite() || !max_log.is_finite() {
        return Err(EstimationError::InvalidSpecification(
            "non-finite transformed age values".to_string(),
        ));
    }
    if (max_log - min_log).abs() < 1e-9 {
        max_log = min_log + 1e-6;
    }
    let (_, age_knots) = create_basis::<Dense>(
        log_entry.view(),
        KnotSource::Generate {
            data_range: (min_log, max_log),
            num_internal_knots: survival_cfg.baseline_basis.num_knots,
        },
        survival_cfg.baseline_basis.degree,
        BasisOptions::value(),
    )?;
    let age_basis = BasisDescriptor {
        knot_vector: age_knots,
        degree: survival_cfg.baseline_basis.degree,
    };

    let time_varying_config = if let Some(tv) = survival_cfg.time_varying.as_ref() {
        let mut min_pgs = f64::INFINITY;
        let mut max_pgs = f64::NEG_INFINITY;
        for &value in bundle.data.pgs.iter() {
            min_pgs = min_pgs.min(value);
            max_pgs = max_pgs.max(value);
        }
        if !min_pgs.is_finite() || !max_pgs.is_finite() || (max_pgs - min_pgs).abs() < 1e-12 {
            None
        } else {
            let (_, pgs_knots) = create_basis::<Dense>(
                bundle.data.pgs.view(),
                KnotSource::Generate {
                    data_range: (min_pgs, max_pgs),
                    num_internal_knots: tv.pgs_basis.num_knots,
                },
                tv.pgs_basis.degree,
                BasisOptions::value(),
            )
            .map_err(|e| EstimationError::InvalidSpecification(e.to_string()))?;
            Some(TensorProductConfig {
                label: tv.label.clone(),
                pgs_basis: BasisDescriptor {
                    knot_vector: pgs_knots,
                    degree: tv.pgs_basis.degree,
                },
                pgs_penalty_order: tv.pgs_penalty_order,
                lambda_age: tv.lambda_age,
                lambda_pgs: tv.lambda_pgs,
                lambda_null: tv.lambda_null,
            })
        }
    } else {
        None
    };

    let SurvivalLayoutBundle {
        mut layout,
        monotonicity,
        mut penalty_descriptors,
        interaction_metadata,
        time_varying_basis,
    } = build_survival_layout(
        &bundle.data,
        &age_basis,
        survival_cfg.guard_delta,
        config.penalty_order,
        survival_cfg.monotonic_grid_size,
        time_varying_config.as_ref(),
    )
    .map_err(map_survival_error)?;

    let pirls_options = WorkingModelPirlsOptions {
        max_iterations: config.max_iterations,
        convergence_tolerance: config.convergence_tolerance,
        max_step_halving: 20,
        min_step_size: 1e-6,
        firth_bias_reduction: false,
    };

    if !layout.penalties.blocks.is_empty() {
        let mut initial_z = Array1::<f64>::zeros(layout.penalties.blocks.len());
        for (i, block) in layout.penalties.blocks.iter().enumerate() {
            initial_z[i] = block.lambda.max(1e-12).ln();
        }

        let objective = |z: &Array1<f64>| -> Result<f64, EstimationError> {
            let mut eval_layout = layout.clone();
            for (block, &zi) in eval_layout.penalties.blocks.iter_mut().zip(z.iter()) {
                block.lambda = zi.exp();
            }
            let mut model = WorkingModelSurvival::new(
                eval_layout,
                &bundle.data,
                monotonicity.clone(),
                survival_spec,
            )
            .map_err(map_survival_error)?;
            let p = model.layout.combined_exit.ncols();
            let result = pirls::run_working_model_pirls(
                &mut model,
                Coefficients::zeros(p),
                &pirls_options,
                |_| {},
            )?;
            Ok(result.state.deviance)
        };

        let mut optimizer = Bfgs::new(initial_z.clone(), |z| {
            let cost = objective(z).unwrap_or(f64::INFINITY);
            let grad = finite_diff_gradient(z, 1e-3, &objective)
                .unwrap_or_else(|_| Array1::<f64>::zeros(z.len()));
            (cost, grad)
        })
        .with_tolerance(config.reml_convergence_tolerance)
        .with_max_iterations(config.reml_max_iterations as usize)
        .with_fp_tolerances(1e2, 1e2)
        .with_no_improve_stop(1e-8, 5)
        .with_rng_seed(0xDEC0DED_u64);

        let solution = match optimizer.run() {
            Ok(solution) => solution,
            Err(wolfe_bfgs::BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
            Err(wolfe_bfgs::BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
            Err(err) => {
                return Err(EstimationError::RemlOptimizationFailed(format!(
                    "survival smoothing optimization failed: {err:?}"
                )));
            }
        };

        for (block, &zi) in layout
            .penalties
            .blocks
            .iter_mut()
            .zip(solution.final_point.iter())
        {
            block.lambda = zi.exp();
        }
        for (descriptor, &zi) in penalty_descriptors
            .iter_mut()
            .zip(solution.final_point.iter())
        {
            descriptor.lambda = zi.exp();
        }
    }

    let mut model =
        WorkingModelSurvival::new(layout.clone(), &bundle.data, monotonicity.clone(), survival_spec)
            .map_err(map_survival_error)?;
    let p = layout.combined_exit.ncols();
    let outcome = pirls::run_working_model_pirls(
        &mut model,
        Coefficients::zeros(p),
        &pirls_options,
        |_| {},
    )?;

    if matches!(outcome.status, PirlsStatus::Unstable) {
        return Err(EstimationError::PirlsDidNotConverge {
            max_iterations: config.max_iterations,
            last_change: outcome.last_gradient_norm,
        });
    }

    let coefficient_vector: Array1<f64> = outcome.beta.clone().into();

    let static_ranges = (0..layout.static_covariates.ncols())
        .map(|col| {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for &v in layout.static_covariates.column(col).iter() {
                min_val = min_val.min(v);
                max_val = max_val.max(v);
            }
            crate::calibrate::survival::ValueRange {
                min: min_val,
                max: max_val,
            }
        })
        .collect();

    let artifacts = SurvivalModelArtifacts {
        coefficients: coefficient_vector,
        age_basis,
        time_varying_basis,
        static_covariate_layout: CovariateLayout {
            column_names: layout.static_covariate_names.clone(),
            ranges: static_ranges,
        },
        penalties: penalty_descriptors,
        age_transform: layout.age_transform.clone(),
        reference_constraint: layout.reference_constraint.clone(),
        monotonicity,
        interaction_metadata,
        companion_models: Vec::new(),
        hessian_factor: None,
        calibrator: None,
        mcmc_samples: None,
        cross_covariance_to_primary: None,
    };

    let lambdas = layout
        .penalties
        .blocks
        .iter()
        .map(|b| b.lambda)
        .collect::<Vec<_>>();

    Ok(crate::calibrate::model::TrainedModel {
        config: config.clone(),
        coefficients: crate::calibrate::model::MappedCoefficients::default(),
        lambdas,
        hull: None,
        penalized_hessian: Some(outcome.state.hessian),
        scale: None,
        calibrator: None,
        joint_link: None,
        survival: Some(artifacts),
        survival_companions: HashMap::new(),
        mcmc_samples: None,
        smoothing_correction: None,
    })
}
