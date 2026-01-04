//! Gradient isolation tests to identify what triggers analytic vs FD mismatch.
//!
//! Each test toggles ONE factor while keeping others constant.
//! This identifies the root cause of gradient failures in complex models.
//!
//! Diagnostic strategy:
//!
//! 1. Factor Isolation: Toggle single factors (Firth, multi-penalty, anisotropic λ)
//! 2. Identity Link: Control for logit-specific issues
//! 3. Frozen Beta FD: Hold β and compare gradients
//! 4. Rank-deficient penalties: difference and near-null cases
//! 5. Projected tensor penalty: constraint projection

use ndarray::{array, Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use faer::Side;
use faer::linalg::solvers::Solve;
use gnomon::calibrate::calibrator::FirthSpec;
use gnomon::calibrate::basis::{
    apply_sum_to_zero_constraint, create_basis, create_difference_penalty_matrix, BasisOptions,
    Dense, KnotSource,
};
use gnomon::calibrate::construction::{
    build_design_and_penalty_matrices, compute_penalty_square_roots, stable_reparameterization,
    ModelLayout,
};
use gnomon::calibrate::data::TrainingData;
use gnomon::calibrate::estimate::{
    evaluate_external_cost_and_ridge, evaluate_external_gradients, optimize_external_design,
    ExternalOptimOptions,
};
use gnomon::calibrate::faer_ndarray::{FaerCholesky, FaerEigh};
use gnomon::calibrate::model::{
    default_reml_parallel_threshold, BasisConfig, InteractionPenaltyKind, LinkFunction,
    ModelConfig, ModelFamily, PrincipalComponentConfig,
};
use gnomon::calibrate::pirls::{calculate_deviance, fit_model_for_fixed_rho, update_glm_vectors};
use gnomon::calibrate::types::LogSmoothingParamsView;
use std::collections::HashMap;

/// Helper to create a simple diagonal penalty matrix.
fn diagonal_penalty(p: usize, start: usize, end: usize) -> Array2<f64> {
    let mut s = Array2::<f64>::zeros((p, p));
    for j in start..end.min(p) {
        s[[j, j]] = 1.0;
    }
    s
}

/// Generate synthetic logit data.
fn generate_logit_data(n: usize, p: usize, seed: u64) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0; // intercept
        for j in 1..p {
            x[[i, j]] = rng.gen_range(-1.0..1.0);
        }
    }
    
    let true_beta: Array1<f64> = (0..p)
        .map(|j| if j == 0 { 0.0 } else { 0.2 / (j as f64) })
        .collect();
    
    let eta = x.dot(&true_beta);
    let y: Array1<f64> = eta
        .iter()
        .map(|&e| {
            let prob = 1.0 / (1.0 + (-e).exp());
            if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 }
        })
        .collect();
    
    let weights = Array1::<f64>::ones(n);
    (y, x, weights)
}

fn generate_logit_data_no_intercept(
    n: usize,
    p: usize,
    seed: u64,
) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            x[[i, j]] = rng.gen_range(-1.0..1.0);
        }
    }
    let true_beta: Array1<f64> = (0..p).map(|j| 0.2 / (1.0 + j as f64)).collect();
    let eta = x.dot(&true_beta);
    let y: Array1<f64> = eta
        .iter()
        .map(|&e| {
            let prob = 1.0 / (1.0 + (-e).exp());
            if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 }
        })
        .collect();
    let weights = Array1::<f64>::ones(n);
    (y, x, weights)
}

/// Generate Gaussian data for identity link tests.
fn generate_gaussian_data(n: usize, p: usize, seed: u64) -> (Array1<f64>, Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    
    let mut x = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        x[[i, 0]] = 1.0;
        for j in 1..p {
            x[[i, j]] = rng.gen_range(-1.0..1.0);
        }
    }
    
    let true_beta: Array1<f64> = (0..p)
        .map(|j| if j == 0 { 1.0 } else { 0.5 / (j as f64) })
        .collect();
    
    let eta = x.dot(&true_beta);
    let y: Array1<f64> = eta
        .iter()
        .map(|&e| e + rng.gen_range(-0.5..0.5))
        .collect();
    
    let weights = Array1::<f64>::ones(n);
    (y, x, weights)
}

fn create_logistic_training_data(n_samples: usize, num_pcs: usize, seed: u64) -> TrainingData {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut p = Array1::zeros(n_samples);
    for val in p.iter_mut() {
        *val = rng.gen_range(-2.5..2.5);
    }

    let mut pcs = Array2::zeros((n_samples, num_pcs));
    for i in 0..n_samples {
        for j in 0..num_pcs {
            pcs[[i, j]] = rng.gen_range(-2.0..2.0);
        }
    }

    let mut eta = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let mut val = 0.6 * p[i] + rng.gen_range(-0.3..0.3);
        for j in 0..num_pcs {
            let weight = 0.2 + 0.1 * (j as f64);
            val += weight * pcs[[i, j]];
        }
        eta[i] = val;
    }

    let mut y = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let prob = 1.0 / (1.0 + (-eta[i]).exp());
        y[i] = if rng.gen_range(0.0..1.0) < prob { 1.0 } else { 0.0 };
    }

    let sex = Array1::from_iter((0..n_samples).map(|_| {
        if rng.gen_range(0.0..1.0) < 0.5 {
            1.0
        } else {
            0.0
        }
    }));
    let weights = Array1::ones(n_samples);

    TrainingData {
        y,
        p,
        sex,
        pcs,
        weights,
    }
}

fn range_from_column(col: &Array1<f64>) -> (f64, f64) {
    (
        col.iter().fold(f64::INFINITY, |acc, &v| acc.min(v)),
        col.iter().fold(f64::NEG_INFINITY, |acc, &v| acc.max(v)),
    )
}

fn logistic_model_config(
    include_pc_mains: bool,
    include_interactions: bool,
    data: &TrainingData,
) -> ModelConfig {
    let (pgs_knots, pc_knots) = if include_interactions {
        (1, 1)
    } else if include_pc_mains {
        (1, 0)
    } else {
        (1, 0)
    };

    let pgs_basis_config = BasisConfig {
        num_knots: pgs_knots,
        degree: 3,
    };
    let pc_basis_template = BasisConfig {
        num_knots: pc_knots.max(1),
        degree: 3,
    };

    let pc_configs: Vec<PrincipalComponentConfig> = if include_pc_mains {
        (0..data.pcs.ncols())
            .map(|idx| {
                let col = data.pcs.column(idx).to_owned();
                PrincipalComponentConfig {
                    name: format!("PC{}", idx + 1),
                    basis_config: pc_basis_template.clone(),
                    range: range_from_column(&col),
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    ModelConfig {
        model_family: ModelFamily::Gam(LinkFunction::Logit),
        penalty_order: 2,
        convergence_tolerance: 1e-6,
        max_iterations: 15,
        reml_convergence_tolerance: 1e-6,
        reml_max_iterations: 0,
        firth_bias_reduction: true,
        reml_parallel_threshold: default_reml_parallel_threshold(),
        pgs_basis_config,
        pc_configs,
        pgs_range: range_from_column(&data.p),
        interaction_penalty: if include_pc_mains && include_interactions {
            InteractionPenaltyKind::Anisotropic
        } else {
            InteractionPenaltyKind::Isotropic
        },
        sum_to_zero_constraints: HashMap::new(),
        knot_vectors: HashMap::new(),
        range_transforms: HashMap::new(),
        pc_null_transforms: HashMap::new(),
        interaction_centering_means: HashMap::new(),
        interaction_orth_alpha: HashMap::new(),
        mcmc_enabled: false,
        calibrator_enabled: false,
        survival: None,
    }
}

fn kron_with_identity(s: &Array2<f64>, n: usize) -> Array2<f64> {
    let p1 = s.nrows();
    let p = p1 * n;
    let mut out = Array2::<f64>::zeros((p, p));
    for i1 in 0..p1 {
        for j1 in 0..p1 {
            let v = s[[i1, j1]];
            if v == 0.0 {
                continue;
            }
            for i2 in 0..n {
                let row = i1 * n + i2;
                let col = j1 * n + i2;
                out[[row, col]] = v;
            }
        }
    }
    out
}

fn identity_kron_with(s: &Array2<f64>, n: usize) -> Array2<f64> {
    let p2 = s.nrows();
    let p = p2 * n;
    let mut out = Array2::<f64>::zeros((p, p));
    for block in 0..n {
        let row_off = block * p2;
        let col_off = block * p2;
        for i in 0..p2 {
            for j in 0..p2 {
                out[[row_off + i, col_off + j]] = s[[i, j]];
            }
        }
    }
    out
}

fn kron_with_identity_right(z: &Array2<f64>, n: usize) -> Array2<f64> {
    let r = z.nrows();
    let c = z.ncols();
    let rows = r * n;
    let cols = c * n;
    let mut out = Array2::<f64>::zeros((rows, cols));
    for i in 0..r {
        for j in 0..c {
            let v = z[[i, j]];
            if v == 0.0 {
                continue;
            }
            for k in 0..n {
                out[[i * n + k, j * n + k]] = v;
            }
        }
    }
    out
}

fn design_with_nullspace_suppression(n: usize, p: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut z = Array2::<f64>::zeros((n, p));
    for i in 0..n {
        for j in 0..p {
            z[[i, j]] = rng.gen_range(-1.0..1.0);
        }
    }
    let mut b = Array2::<f64>::zeros((p, 2));
    for j in 0..p {
        b[[j, 0]] = 1.0;
        b[[j, 1]] = j as f64;
    }
    let bt_b = b.t().dot(&b);
    let det = bt_b[[0, 0]] * bt_b[[1, 1]] - bt_b[[0, 1]] * bt_b[[1, 0]];
    let inv = array![
        [bt_b[[1, 1]] / det, -bt_b[[0, 1]] / det],
        [-bt_b[[1, 0]] / det, bt_b[[0, 0]] / det],
    ];
    let proj = b.dot(&inv).dot(&b.t());
    let mut p_mat = Array2::<f64>::eye(p);
    p_mat -= &proj;
    z.dot(&p_mat)
}

/// Run gradient check and return (cosine, rel_l2, max_analytic, max_fd).
fn check_gradient(
    y: &Array1<f64>,
    x: &Array2<f64>,
    weights: &Array1<f64>,
    s_list: &[Array2<f64>],
    rho: &Array1<f64>,
    firth: bool,
    nullspace_dims: Vec<usize>,
    link: LinkFunction,
) -> Result<(f64, f64, f64, f64), String> {
    let n = x.nrows();
    let offset = Array1::<f64>::zeros(n);
    
    let opts = ExternalOptimOptions {
        link,
        firth: if firth { Some(FirthSpec { enabled: true }) } else { None },
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    match evaluate_external_gradients(
        y.view(),
        weights.view(),
        x.view(),
        offset.view(),
        s_list,
        &opts,
        rho,
    ) {
        Ok((analytic, fd)) => {
            let dot: f64 = analytic.dot(&fd);
            let n_a: f64 = analytic.dot(&analytic).sqrt();
            let n_f: f64 = fd.dot(&fd).sqrt();
            let cosine = if n_a * n_f > 1e-12 { dot / (n_a * n_f) } else { 1.0 };
            
            let diff = &analytic - &fd;
            let rel_l2 = diff.dot(&diff).sqrt() / n_f.max(n_a).max(1e-12);
            
            let max_a = analytic.iter().copied().fold(0.0f64, |acc, v| acc.max(v.abs()));
            let max_f = fd.iter().copied().fold(0.0f64, |acc, v| acc.max(v.abs()));
            
            Ok((cosine, rel_l2, max_a, max_f))
        }
        Err(e) => Err(format!("{:?}", e)),
    }
}

fn print_result(label: &str, cos: f64, rel: f64, max_a: f64, max_f: f64) {
    let status = if cos > 0.99 && rel < 0.1 { "✓ PASS" } else { "✗ FAIL" };
    println!("  {}: cos={:.4}, rel_l2={:.3e}, |a|={:.3e}, |fd|={:.3e} {}", 
             label, cos, rel, max_a, max_f, status);
}

fn gradient_metrics(a: &Array1<f64>, b: &Array1<f64>) -> (f64, f64, f64, f64) {
    let dot: f64 = a.dot(b);
    let n_a: f64 = a.dot(a).sqrt();
    let n_b: f64 = b.dot(b).sqrt();
    let cosine = if n_a * n_b > 1e-12 { dot / (n_a * n_b) } else { 1.0 };
    
    let diff = a - b;
    let rel_l2 = diff.dot(&diff).sqrt() / n_b.max(n_a).max(1e-12);
    
    let max_a = a.iter().copied().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let max_b = b.iter().copied().fold(0.0f64, |acc, v| acc.max(v.abs()));
    
    (cosine, rel_l2, max_a, max_b)
}

fn laml_cost_logit_with_beta(
    y: &Array1<f64>,
    x: &Array2<f64>,
    weights: &Array1<f64>,
    s_list: &[Array2<f64>],
    rho: &Array1<f64>,
    beta: &Array1<f64>,
) -> Result<f64, String> {
    let p = x.ncols();
    let layout = ModelLayout::external(p, s_list.len());
    let rs_list = compute_penalty_square_roots(s_list).map_err(|e| format!("{:?}", e))?;
    let lambdas: Vec<f64> = rho.iter().map(|v| v.exp()).collect();
    let reparam =
        stable_reparameterization(&rs_list, &lambdas, &layout).map_err(|e| format!("{:?}", e))?;
    let beta_t = reparam.qs.t().dot(beta);
    let x_t = x.dot(&reparam.qs);
    let n = x_t.nrows();
    let eta = x_t.dot(&beta_t);
    let mut mu = Array1::<f64>::zeros(n);
    let mut w_work = Array1::<f64>::zeros(n);
    let mut z = Array1::<f64>::zeros(n);
    update_glm_vectors(
        y.view(),
        &eta,
        LinkFunction::Logit,
        weights.view(),
        &mut mu,
        &mut w_work,
        &mut z,
    );
    let deviance = calculate_deviance(y.view(), &mu, LinkFunction::Logit, weights.view());
    let mut xtwx = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        let wi = w_work[i];
        if wi == 0.0 {
            continue;
        }
        for a in 0..p {
            let xa = x_t[[i, a]];
            if xa == 0.0 {
                continue;
            }
            for b in 0..p {
                xtwx[[a, b]] += wi * xa * x_t[[i, b]];
            }
        }
    }
    let h = &xtwx + &reparam.s_transformed;
    let chol = h
        .clone()
        .cholesky(Side::Lower)
        .map_err(|_| "Hessian not PD".to_string())?;
    let log_det_h = 2.0 * chol.diag().mapv(f64::ln).sum();
    let log_det_s = reparam.log_det;
    let s_beta = reparam.s_transformed.dot(&beta_t);
    let penalty = beta_t.dot(&s_beta);
    let penalised_ll = -0.5 * deviance - 0.5 * penalty;
    let penalty_rank = reparam.e_transformed.nrows();
    let mp = layout.total_coeffs.saturating_sub(penalty_rank) as f64;
    let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_h
        + 0.5 * mp * (2.0 * std::f64::consts::PI).ln();
    Ok(-laml)
}

fn frozen_fd_gradient_logit(
    y: &Array1<f64>,
    x: &Array2<f64>,
    weights: &Array1<f64>,
    s_list: &[Array2<f64>],
    rho: &Array1<f64>,
    beta: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    let mut g = Array1::zeros(rho.len());
    for k in 0..rho.len() {
        let h = (1e-4 * (1.0 + rho[k].abs())).max(1e-5);
        let mut rp = rho.clone();
        rp[k] += 0.5 * h;
        let mut rm = rho.clone();
        rm[k] -= 0.5 * h;
        let fp = laml_cost_logit_with_beta(y, x, weights, s_list, &rp, beta)?;
        let fm = laml_cost_logit_with_beta(y, x, weights, s_list, &rm, beta)?;
        g[k] = (fp - fm) / h;
    }
    Ok(g)
}

// ============================================================================
// Section 1: factor isolation tests
// ============================================================================

/// Test 1: Single penalty, no Firth — baseline.
#[test]
fn isolation_baseline_single_penalty_no_firth() {
    println!("\n=== ISOLATION: Single penalty, No Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 8, 42);
    let p = x.ncols();
    let s = diagonal_penalty(p, 1, p);
    let rho = array![0.0];
    
    match check_gradient(&y, &x, &w, &[s], &rho, false, vec![1], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => print_result("Baseline", cos, rel, max_a, max_f),
        Err(e) => println!("  ERROR: {}", e),
    }
}

/// Test 2: Single penalty, WITH Firth.
#[test]
fn isolation_single_penalty_with_firth() {
    println!("\n=== ISOLATION: Single penalty, WITH Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 8, 42);
    let p = x.ncols();
    let s = diagonal_penalty(p, 1, p);
    let rho = array![0.0];
    
    match check_gradient(&y, &x, &w, &[s], &rho, true, vec![1], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => print_result("Firth alone", cos, rel, max_a, max_f),
        Err(e) => println!("  ERROR: {}", e),
    }
}

/// Test 3: Multi-penalty (isotropic λ), no Firth.
#[test]
fn isolation_multi_penalty_isotropic_no_firth() {
    println!("\n=== ISOLATION: Multi-penalty (isotropic), No Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let rho = array![0.0, 0.0, 0.0];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, false, vec![1, 0, 0], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => print_result("Multi-iso", cos, rel, max_a, max_f),
        Err(e) => println!("  ERROR: {}", e),
    }
}

/// Test 4: Multi-penalty (ANISOTROPIC λ), no Firth.
#[test]
fn isolation_multi_penalty_anisotropic_no_firth() {
    println!("\n=== ISOLATION: Multi-penalty (ANISOTROPIC), No Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let rho = array![0.0, 4.6, -4.6]; // λ = [1, 100, 0.01]
    
    match check_gradient(&y, &x, &w, &s_list, &rho, false, vec![1, 0, 0], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => print_result("Multi-aniso", cos, rel, max_a, max_f),
        Err(e) => println!("  ERROR: {}", e),
    }
}

/// Test 5: Multi-penalty + Firth (isotropic).
#[test]
fn isolation_multi_penalty_isotropic_with_firth() {
    println!("\n=== ISOLATION: Multi-penalty (isotropic) + Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let rho = array![0.0, 0.0, 0.0];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1, 0, 0], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => print_result("Multi-iso+Firth", cos, rel, max_a, max_f),
        Err(e) => println!("  ERROR: {}", e),
    }
}

/// Test 6: Multi-penalty + Firth (ANISOTROPIC) — full complexity.
#[test]
fn isolation_multi_penalty_anisotropic_with_firth() {
    println!("\n=== ISOLATION: Multi-penalty (ANISOTROPIC) + Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let rho = array![0.0, 4.6, -4.6];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1, 0, 0], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => print_result("Full complexity", cos, rel, max_a, max_f),
        Err(e) => println!("  ERROR: {}", e),
    }
}

/// Test 7: Disjoint null spaces.
#[test]
fn isolation_disjoint_null_spaces() {
    println!("\n=== ISOLATION: Disjoint null spaces ===");
    
    let (y, x, w) = generate_logit_data(80, 6, 42);
    
    let mut s1 = Array2::<f64>::zeros((6, 6));
    for j in 1..4 { s1[[j, j]] = 1.0; }
    
    let mut s2 = Array2::<f64>::zeros((6, 6));
    for j in 4..6 { s2[[j, j]] = 1.0; }
    
    let rho = array![2.0, -2.0];
    
    match check_gradient(&y, &x, &w, &[s1, s2], &rho, true, vec![1, 0], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => {
            print_result("Disjoint null", cos, rel, max_a, max_f);
        }
        Err(e) => println!("  ERROR: {}", e),
    }
}

// ============================================================================
// Section 2: identity link control
// ============================================================================

/// If Identity works but Logit fails → issue is Firth/W-derivative specific.
/// If Identity also fails → issue is in linear algebra (penalties, traces).
#[test]
fn isolation_identity_link_control() {
    println!("\n=== IDENTITY LINK CONTROL ===");
    
    let (y, x, w) = generate_gaussian_data(100, 12, 42);
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let rho = array![0.0, 4.6, -4.6]; // Same anisotropic λ
    
    match check_gradient(&y, &x, &w, &s_list, &rho, false, vec![1, 0, 0], LinkFunction::Identity) {
        Ok((cos, rel, max_a, max_f)) => {
            print_result("Identity link", cos, rel, max_a, max_f);
        }
        Err(e) => println!("  ERROR: {}", e),
    }
}

// ============================================================================
// Section 3: rank-deficient penalties
// ============================================================================

#[test]
fn isolation_difference_penalty_logit_no_firth() {
    println!("\n=== ISOLATION: Difference penalty, Logit, No Firth ===");
    
    let (y, x, w) = generate_logit_data(120, 10, 42);
    let p = x.ncols();
    let s = create_difference_penalty_matrix(p, 2).expect("difference penalty");
    let rho = array![0.0];
    
    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        Array1::<f64>::zeros(x.nrows()).view(),
        &[s],
        &ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: None,
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![2],
        },
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    
    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    print_result("Difference penalty", cos_fd, rel_fd, max_a, max_fd);
}

#[test]
fn isolation_dirty_nullspace_logit_no_firth() {
    println!("\n=== ISOLATION: Near-null penalty, Logit, No Firth ===");
    
    let (y, x, w) = generate_logit_data(120, 10, 43);
    let p = x.ncols();
    let mut s = diagonal_penalty(p, 1, p - 2);
    let eps = 1e-12;
    for j in (p - 2)..p {
        s[[j, j]] += eps;
    }
    let rho = array![0.0];
    
    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        Array1::<f64>::zeros(x.nrows()).view(),
        &[s],
        &ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: None,
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![2],
        },
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    
    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    print_result("Near-null penalty", cos_fd, rel_fd, max_a, max_fd);
}

#[test]
fn isolation_concurrent_nullspace_instability() {
    println!("\n=== ISOLATION: Concurrent nullspace, Logit, No Firth ===");
    
    let n = 160;
    let p = 12;
    let x = design_with_nullspace_suppression(n, p, 44);
    let mut rng = StdRng::seed_from_u64(44);
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        y[i] = if rng.r#gen::<f64>() < 0.5 { 0.0 } else { 1.0 };
    }
    let w = Array1::<f64>::ones(n);
    
    let mut s = create_difference_penalty_matrix(p, 2).expect("difference penalty");
    let eps = 1e-12;
    let mut b1 = Array1::<f64>::zeros(p);
    let mut b2 = Array1::<f64>::zeros(p);
    for j in 0..p {
        b1[j] = 1.0;
        b2[j] = j as f64;
    }
    let b1_outer = b1.view().insert_axis(ndarray::Axis(1)).dot(&b1.view().insert_axis(ndarray::Axis(0)));
    let b2_outer = b2.view().insert_axis(ndarray::Axis(1)).dot(&b2.view().insert_axis(ndarray::Axis(0)));
    s.scaled_add(eps, &b1_outer);
    s.scaled_add(eps, &b2_outer);
    let rho = array![5.0];
    
    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        Array1::<f64>::zeros(n).view(),
        &[s],
        &ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: None,
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![2],
        },
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    
    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    print_result("Concurrent nullspace", cos_fd, rel_fd, max_a, max_fd);
}

#[test]
fn isolation_projected_tensor_penalty_logit_no_firth() {
    println!("\n=== ISOLATION: Projected tensor penalty, Logit, No Firth ===");
    let n = 140;
    let n_obs = 200;
    let degree = 3;
    let num_internal_knots = 20;
    let data: Array1<f64> = (0..n_obs)
        .map(|i| i as f64 / (n_obs - 1) as f64)
        .collect();
    let (basis_arc, _) = create_basis::<Dense>(
        data.view(),
        KnotSource::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots,
        },
        degree,
        BasisOptions::value(),
    )
    .expect("basis");
    let basis = basis_arc.as_ref();
    let (_, z1) = apply_sum_to_zero_constraint(basis.view(), None).expect("constraint");
    let k1 = z1.nrows();
    let n2 = 5;
    let p_raw = k1 * n2;
    let s1 = create_difference_penalty_matrix(k1, 2).expect("difference penalty");
    let s_raw = kron_with_identity(&s1, n2);
    let z = kron_with_identity_right(&z1, n2);
    let s = z.t().dot(&s_raw).dot(&z);
    
    let (y, x, w) = generate_logit_data_no_intercept(n, p_raw - n2, 55);
    let rho = array![0.0];
    
    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        Array1::<f64>::zeros(n).view(),
        &[s],
        &ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: None,
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![n2],
        },
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    
    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    print_result("Projected tensor penalty", cos_fd, rel_fd, max_a, max_fd);
}

#[test]
fn isolation_projected_tensor_penalty_with_firth() {
    println!("\n=== ISOLATION: Projected tensor penalty, Logit, Firth ===");
    let n = 140;
    let n_obs = 200;
    let degree = 3;
    let num_internal_knots = 20;
    let data: Array1<f64> = (0..n_obs)
        .map(|i| i as f64 / (n_obs - 1) as f64)
        .collect();
    let (basis_arc, _) = create_basis::<Dense>(
        data.view(),
        KnotSource::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots,
        },
        degree,
        BasisOptions::value(),
    )
    .expect("basis");
    let basis = basis_arc.as_ref();
    let (_, z1) = apply_sum_to_zero_constraint(basis.view(), None).expect("constraint");
    let k1 = z1.nrows();
    let n2 = 5;
    let p_raw = k1 * n2;
    let s1_base = create_difference_penalty_matrix(k1, 2).expect("difference penalty");
    let s2_base = create_difference_penalty_matrix(n2, 2).expect("difference penalty");
    let s1_raw = kron_with_identity(&s1_base, n2);
    let s2_raw = identity_kron_with(&s2_base, k1);
    let z = kron_with_identity_right(&z1, n2);
    let s1 = z.t().dot(&s1_raw).dot(&z);
    let s2 = z.t().dot(&s2_raw).dot(&z);

    let (y, x, w) = generate_logit_data_no_intercept(n, p_raw - n2, 56);
    let rho = array![0.0, 0.0];

    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        Array1::<f64>::zeros(n).view(),
        &[s1, s2],
        &ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![n2, 2 * (k1 - 1)],
        },
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };

    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    print_result("Projected tensor + Firth", cos_fd, rel_fd, max_a, max_fd);
}

#[test]
fn isolation_multiple_overlapping_dense_penalties_with_firth() {
    println!("\n=== ISOLATION: Multiple overlapping dense penalties + Firth ===");
    let n = 120;
    let n_obs = 180;
    let degree = 3;
    let num_internal_knots = 12;
    let data: Array1<f64> = (0..n_obs)
        .map(|i| i as f64 / (n_obs - 1) as f64)
        .collect();
    let (basis_arc, _) = create_basis::<Dense>(
        data.view(),
        KnotSource::Generate {
            data_range: (0.0, 1.0),
            num_internal_knots,
        },
        degree,
        BasisOptions::value(),
    )
    .expect("basis");
    let basis = basis_arc.as_ref();
    let (_, z) = apply_sum_to_zero_constraint(basis.view(), None).expect("constraint");
    let k_eff = z.nrows();
    let s_raw_1 = create_difference_penalty_matrix(k_eff, 1).expect("diff 1");
    let s_raw_2 = create_difference_penalty_matrix(k_eff, 2).expect("diff 2");
    let s1 = z.t().dot(&s_raw_1).dot(&z);
    let s2 = z.t().dot(&s_raw_2).dot(&z);

    let p_eff = s1.nrows();
    let (y, x, w) = generate_logit_data_no_intercept(n, p_eff, 99);
    let rho = array![0.9, 0.4];

    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        Array1::<f64>::zeros(n).view(),
        &[s1, s2],
        &ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![1, 2],
        },
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };

    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    print_result("Multi dense + Firth", cos_fd, rel_fd, max_a, max_fd);
}

#[test]
fn isolation_reparam_pgs_pc_mains_firth() {
    println!("\n=== ISOLATION: Reparam PGS+PC mains, Logit, Firth ===");
    let train = create_logistic_training_data(100, 3, 31);
    let mut config = logistic_model_config(true, false, &train);
    config.firth_bias_reduction = true;
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    let rho: Array1<f64> = Array1::from_elem(layout.num_penalties, 12.0);
    let nullspace_dims = vec![0; s_list.len()];
    let offset = Array1::<f64>::zeros(train.y.len());
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    let (analytic, fd) = match evaluate_external_gradients(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    print_result("Reparam PGS+PC mains", cos_fd, rel_fd, max_a, max_fd);
    assert!(cos_fd > 0.999, "cosine too low: {cos_fd}");
    assert!(rel_fd < 5e-2, "rel_l2 too high: {rel_fd}");
}

#[test]
fn isolation_reparam_pgs_pc_mains_firth_frozen_beta_fd() {
    println!("\n=== ISOLATION: Reparam PGS+PC mains, Firth, Frozen beta FD ===");
    let train = create_logistic_training_data(100, 3, 31);
    let mut config = logistic_model_config(true, false, &train);
    config.firth_bias_reduction = true;
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    let rho: Array1<f64> = Array1::from_elem(layout.num_penalties, 12.0);
    let offset = Array1::<f64>::zeros(train.y.len());
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: vec![0; s_list.len()],
    };

    let (analytic, fd) = evaluate_external_gradients(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();

    let rs_list = compute_penalty_square_roots(&s_list).unwrap();
    let cfg = ModelConfig::external(LinkFunction::Logit, 1e-10, 200, true);
    let (pirls, _) = fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        x.view(),
        offset.view(),
        train.y.view(),
        train.weights.view(),
        &rs_list,
        None,
        None,
        &layout,
        &cfg,
        None,
        None,
    )
    .unwrap();
    let beta_orig = pirls.reparam_result.qs.dot(pirls.beta_transformed.as_ref());

    let frozen_fd = frozen_fd_gradient_logit(
        &train.y,
        &x,
        &train.weights,
        &s_list,
        &rho,
        &beta_orig,
    )
    .unwrap();

    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    let (cos_frozen, rel_frozen, max_af, max_frozen) = gradient_metrics(&analytic, &frozen_fd);
    print_result("Analytic vs FD", cos_fd, rel_fd, max_a, max_fd);
    print_result(
        "Analytic vs Frozen FD",
        cos_frozen,
        rel_frozen,
        max_af,
        max_frozen,
    );
    assert!(
        cos_frozen > cos_fd + 0.02 || rel_frozen < 0.5 * rel_fd,
        "frozen beta did not improve FD agreement: cos {cos_fd:.4}->{cos_frozen:.4}, rel {rel_fd:.3e}->{rel_frozen:.3e}"
    );
}

// ============================================================================
// Section 4: frozen beta FD
// ============================================================================

#[test]
fn isolation_frozen_beta_fd() {
    println!("\n=== ISOLATION: Frozen beta FD ===");
    
    let (y, x, w) = generate_logit_data(120, 10, 42);
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 6),
        diagonal_penalty(p, 5, 10),
    ];
    let rho = array![0.0, 3.2];
    let n = x.nrows();
    let offset = Array1::<f64>::zeros(n);
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: None,
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: vec![1, 0],
    };
    
    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    
    let layout = ModelLayout::external(p, s_list.len());
    let cfg = ModelConfig::external(LinkFunction::Logit, 1e-10, 200, false);
    let rs_list = match compute_penalty_square_roots(&s_list) {
        Ok(list) => list,
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    let (pirls, _) = match fit_model_for_fixed_rho(
        LogSmoothingParamsView::new(rho.view()),
        x.view(),
        offset.view(),
        y.view(),
        w.view(),
        &rs_list,
        None,
        None,
        &layout,
        &cfg,
        None,
        None,
    ) {
        Ok(result) => result,
        Err(e) => {
            println!("  ERROR: {:?}", e);
            return;
        }
    };
    let beta_orig = pirls.reparam_result.qs.dot(pirls.beta_transformed.as_ref());
    
    let frozen_fd = match frozen_fd_gradient_logit(&y, &x, &w, &s_list, &rho, &beta_orig) {
        Ok(g) => g,
        Err(e) => {
            println!("  ERROR: {}", e);
            return;
        }
    };
    
    let (cos_fd, rel_fd, max_a, max_fd) = gradient_metrics(&analytic, &fd);
    let (cos_frozen, rel_frozen, max_af, max_frozen) = gradient_metrics(&analytic, &frozen_fd);
    print_result("Analytic vs FD", cos_fd, rel_fd, max_a, max_fd);
    print_result("Analytic vs Frozen FD", cos_frozen, rel_frozen, max_af, max_frozen);
}

// ============================================================================
// Section 5: FD reliability diagnostic
// ============================================================================

#[test]
fn isolation_diagnostic_fd_noise_floor_at_high_smoothing() {
    let train = create_logistic_training_data(100, 3, 31);
    let mut config = logistic_model_config(true, false, &train);
    config.firth_bias_reduction = true;
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    
    let offset = Array1::<f64>::zeros(train.y.len());
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: vec![0; s_list.len()],
    };
    
    let rho_mod = Array1::from_elem(layout.num_penalties, 2.0);
    let (a_mod, fd_mod) = evaluate_external_gradients(
        train.y.view(), train.weights.view(), x.view(), offset.view(),
        &s_list, &opts, &rho_mod,
    ).unwrap();
    let (cos_mod, ..) = gradient_metrics(&a_mod, &fd_mod);
    
    let rho_ext = Array1::from_elem(layout.num_penalties, 12.0);
    let (a_ext, fd_ext) = evaluate_external_gradients(
        train.y.view(), train.weights.view(), x.view(), offset.view(),
        &s_list, &opts, &rho_ext,
    ).unwrap();
    let (cos_ext, _, max_a, _) = gradient_metrics(&a_ext, &fd_ext);
    
    assert!(cos_mod > 0.999, "rho=2 failed: cos={cos_mod}");
    assert!(cos_ext > 0.999, "rho=12 failed: cos={cos_ext}, |grad|={max_a:.2e}");
}

// ============================================================================
// Section 6: targeted noise floor diagnostics
// ============================================================================

#[test]
fn isolation_stationarity_limit_at_optimum() {
    println!("\n=== ISOLATION: Stationarity limit at optimized rho ===");
    let (y, x, w) = generate_logit_data_no_intercept(180, 6, 7);
    let p = x.ncols();
    let s_list = vec![diagonal_penalty(p, 0, 4)];
    let offset = Array1::<f64>::zeros(y.len());
    let nullspace_dims = vec![2];
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: None,
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: nullspace_dims.clone(),
    };

    let rho_baseline = Array1::<f64>::zeros(s_list.len());
    let (a_baseline, _) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho_baseline,
    )
    .unwrap();
    let n_baseline = a_baseline.dot(&a_baseline).sqrt();

    let opt = optimize_external_design(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
    )
    .unwrap();
    let rho_opt = opt.lambdas.mapv(|v| v.ln());
    let (a_opt, fd_opt) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho_opt,
    )
    .unwrap();
    let n_opt = a_opt.dot(&a_opt).sqrt();
    let (cos_opt, rel_opt, max_a, max_fd) = gradient_metrics(&a_opt, &fd_opt);
    print_result("Optimized rho", cos_opt, rel_opt, max_a, max_fd);

    assert!(
        n_opt < 1e-3 || n_opt < 1e-2 * n_baseline,
        "optimized gradient too large: n_opt={n_opt:.3e} n_baseline={n_baseline:.3e}"
    );
}

#[test]
fn isolation_high_penalty_nullspace_gradient_decay() {
    println!("\n=== ISOLATION: High-penalty nullspace gradient decay ===");
    let (y, x, w) = generate_logit_data_no_intercept(200, 4, 11);
    let p = x.ncols();
    let s_list = vec![diagonal_penalty(p, 0, 2)];
    let offset = Array1::<f64>::zeros(y.len());
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: vec![2],
    };

    let rho_mid = array![2.0];
    let (a_mid, fd_mid) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho_mid,
    )
    .unwrap();
    let (cos_mid, rel_mid, max_a_mid, max_fd_mid) = gradient_metrics(&a_mid, &fd_mid);
    print_result("Nullspace rho=2", cos_mid, rel_mid, max_a_mid, max_fd_mid);
    assert!(cos_mid > 0.99, "rho=2 cosine too low: {cos_mid}");

    let rho_high = array![15.0];
    let (a_high, fd_high) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho_high,
    )
    .unwrap();
    let (cos_high, rel_high, max_a_high, max_fd_high) = gradient_metrics(&a_high, &fd_high);
    print_result("Nullspace rho=15", cos_high, rel_high, max_a_high, max_fd_high);

    let n_mid = a_mid.dot(&a_mid).sqrt();
    let n_high = a_high.dot(&a_high).sqrt();
    assert!(
        n_high < 0.2 * n_mid,
        "high-penalty gradient did not decay: n_high={n_high:.3e} n_mid={n_mid:.3e}"
    );
}

// ============================================================================
// Section 7: diagnostic strategies from critic
// ============================================================================

#[test]
fn diagnostic_super_convergence_tight_tol_improves_fd() {
    println!("\n=== DIAGNOSTIC: Super-convergence tight tol improves FD ===");
    let train = create_logistic_training_data(100, 3, 31);
    let mut config = logistic_model_config(true, false, &train);
    config.firth_bias_reduction = true;
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    let rho = Array1::from_elem(layout.num_penalties, 12.0);
    let offset = Array1::<f64>::zeros(train.y.len());

    let opts_base = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: vec![0; s_list.len()],
    };
    let (a_base, fd_base) = evaluate_external_gradients(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts_base,
        &rho,
    )
    .unwrap();
    let (cos_base, rel_base, max_a_base, max_fd_base) =
        gradient_metrics(&a_base, &fd_base);
    print_result("Base tol", cos_base, rel_base, max_a_base, max_fd_base);

    let grad_norm = a_base.dot(&a_base).sqrt();
    let tight_tol = (grad_norm * 1e-3).max(1e-14);
    let opts_tight = ExternalOptimOptions {
        tol: tight_tol,
        max_iter: 500,
        ..opts_base.clone()
    };
    let (a_tight, fd_tight) = evaluate_external_gradients(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts_tight,
        &rho,
    )
    .unwrap();
    let (cos_tight, rel_tight, max_a_tight, max_fd_tight) =
        gradient_metrics(&a_tight, &fd_tight);
    print_result("Tight tol", cos_tight, rel_tight, max_a_tight, max_fd_tight);

    assert!(
        cos_tight > cos_base + 0.02 || rel_tight < 0.5 * rel_base,
        "tight tol did not improve FD agreement: cos {cos_base:.4}->{cos_tight:.4}, rel {rel_base:.3e}->{rel_tight:.3e}"
    );
}

#[test]
fn diagnostic_fd_ridge_jitter_at_high_smoothing() {
    println!("\n=== DIAGNOSTIC: Ridge jitter across FD probes ===");
    let train = create_logistic_training_data(100, 3, 31);
    let mut config = logistic_model_config(true, false, &train);
    config.firth_bias_reduction = true;
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    let rho = Array1::from_elem(layout.num_penalties, 12.0);
    let offset = Array1::<f64>::zeros(train.y.len());
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: vec![0; s_list.len()],
    };

    let mut rho_p = rho.clone();
    let mut rho_m = rho.clone();
    let h = 1e-4_f64 * (1.0_f64 + f64::abs(rho[0]));
    rho_p[0] += 0.5 * h;
    rho_m[0] -= 0.5 * h;

    let (_, ridge_0) = evaluate_external_cost_and_ridge(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .unwrap();
    let (_, ridge_p) = evaluate_external_cost_and_ridge(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho_p,
    )
    .unwrap();
    let (_, ridge_m) = evaluate_external_cost_and_ridge(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho_m,
    )
    .unwrap();

    println!(
        "  ridge: center={:.3e} plus={:.3e} minus={:.3e}",
        ridge_0, ridge_p, ridge_m
    );
    let jitter = (ridge_p - ridge_0).abs().max((ridge_m - ridge_0).abs());
    assert!(
        jitter == 0.0,
        "ridge changed across FD probes (jitter={jitter:.3e}) - expected fixed"
    );
}

// ============================================================================
// Section 4: Missing Term Hypothesis (Firth-LAML)
// ============================================================================

/// Replicated logic to compute H_phi components.
/// Allows injecting S to verify if S-dependence explains the gradient gap.
fn compute_firth_h_phi(
    x: &Array2<f64>,
    weights: &Array1<f64>,
    mu: &Array1<f64>,
    s_lambda: Option<&Array2<f64>>, // If Some, use Penalized Hat Matrix
) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();

    // 1. Compute Hat Matrix "A" (or components B)
    // If s_lambda is None, use Fisher: A = W^1/2 X (X' W X)^-1 X' W^1/2
    // If s_lambda is Some, use Penalized: A = W^1/2 X (X' W X + S)^-1 X' W^1/2
    
    let mut xw = x.clone();
    let mut sqrt_w = Array1::zeros(n);
    for i in 0..n {
        let sw = weights[i].max(0.0).sqrt();
        sqrt_w[i] = sw;
        for j in 0..p {
            xw[[i, j]] *= sw;
        }
    }

    let mut info = xw.t().dot(&xw);
    if let Some(s) = s_lambda {
        info = &info + s;
    }
    
    // In production, we add a small ridge for stability
    let scale = info.diag().iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let ridge = scale * 1e-8;
    for i in 0..p {
        info[[i, i]] += ridge;
    }

    let info_view = gnomon::calibrate::faer_ndarray::FaerArrayView::new(&info);
    // Note: FaerLlt might need to be imported or fully qualified
    use gnomon::calibrate::faer_ndarray::{FaerLlt, array2_to_mat_mut}; 
    
    let chol = FaerLlt::new(info_view.as_ref(), Side::Lower).unwrap();
    
    // H_hat = XW * Info^-1 * XW^T
    // Let B = XW * L^-T. Then H_hat = B B^T.
    // L L^T = Info. 
    // Solve L Y = XW^T  => Y = L^-1 XW^T.  B = Y^T.
    
    let mut rhs = xw.t().to_owned();
    let mut rhs_view = array2_to_mat_mut(&mut rhs);
    chol.solve_in_place(rhs_view.as_mut()); 
    // Now rhs contains Info^-1 XW^T ?? No.
    // solve_in_place solves A x = b. Here A is LLT (Info).
    // So rhs column j solves: Info * x_j = (XW^T)_j
    // So rhs = Info^-1 * XW^T
    // So H_hat = XW * rhs
    
    let h_hat = xw.dot(&rhs);
    
    // 2. Compute components for H_phi
    // H_phi = 0.5 * (T1 - T2)
    // T1 = X^T diag(h dot v) X
    // T2 = X^T diag(u) (H_hat dot H_hat) diag(u) X
    // u = 1 - 2mu
    // v = u^2 - 2w (where w = mu(1-mu))
    
    let mut u = Array1::zeros(n);
    let mut v = Array1::zeros(n);
    let mut h_diag = Array1::zeros(n);
    
    for i in 0..n {
        let mu_i = mu[i];
        let w_b = mu_i * (1.0 - mu_i);
        u[i] = 1.0 - 2.0 * mu_i;
        v[i] = u[i] * u[i] - 2.0 * w_b; // = (1-2mu)^2 - 2mu(1-mu)
        h_diag[i] = h_hat[[i, i]];
    }
    
    let mut t1 = Array2::zeros((p, p));
    // T1_jk = sum_i x_ij * (h_i * v_i) * x_ik
    for i in 0..n {
        let factor = h_diag[i] * v[i];
        for j in 0..p {
            for k in 0..p {
                t1[[j, k]] += x[[i, j]] * factor * x[[i, k]];
            }
        }
    }
    
    // T2
    // Middle matrix M_mid = diag(u) (H_hat ∘ H_hat) diag(u)
    // M_mid_ij = u_i * (h_hat_ij^2) * u_j
    
    // T2 = X^T M_mid X
    let mut t2 = Array2::<f64>::zeros((p, p));
    for i in 0..n {
        for j in 0..n {
            let m_val = u[i] * (h_hat[[i, j]] * h_hat[[i, j]]) * u[j];
            if m_val.abs() < 1e-12 { continue; }
            for a in 0..p {
                for b in 0..p {
                    t2[[a, b]] += x[[i, a]] * m_val * x[[j, b]];
                }
            }
        }
    }
    
    let mut h_phi = t1 - &t2;
    h_phi.mapv_inplace(|val| 0.5 * val);
    
    h_phi
}

#[test]
fn isolation_missing_term_hypothesis() {
    println!("\n=== ISOLATION: Missing Term Hypothesis (Firth-LAML) ===");
    
    // 1. Setup: High Smoothing + Firth
    // Use the "Test 6" anisotropic scenario which is complex
    let (y, x, w) = generate_logit_data(80, 8, 42); // Smaller N for speed
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 8),
    ];
    // High smoothing to exacerbate the term
    let rho = array![5.0, 5.0]; 
    
    // 2. Frozen Beta at Optimum
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-8,
        max_iter: 100,
        nullspace_dims: vec![1, 0],
    };
    
    // Get analytic gradient and beta (implicitly via wrapper or we run pirls locally)
    let (analytic_gradient, _) = evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        Array1::zeros(y.len()).view(),
        &s_list,
        &opts,
        &rho,
    ).expect("Gradient evaluation failed");
    
    // Get optimal beta using public fit_model_for_fixed_rho
    let rs_list = compute_penalty_square_roots(&s_list).unwrap();
    let balanced = create_difference_penalty_matrix(p, 1).unwrap();
    let layout = ModelLayout::external(p, s_list.len());
    let config = ModelConfig::external(
        opts.link,
        opts.tol,
        opts.max_iter,
        true,
    );

    use gnomon::calibrate::types::LogSmoothingParamsView;
    let rho_view = LogSmoothingParamsView::new(rho.view());

    let (fit, _) = fit_model_for_fixed_rho(
        rho_view,
        x.view(),
        Array1::zeros(y.len()).view(),
        y.view(),
        w.view(),
        &rs_list,
        Some(&balanced),
        None,
        &layout,
        &config,
        None,
        None
    ).expect("Fit failed");
    
    let beta = fit.beta_transformed.clone();
    let mu = fit.solve_mu; 
    
    // 3. Compute H_phi(rho)
    // Reconstruct S
    let mut s_rho = Array2::<f64>::zeros((p, p));
    for (k, s) in s_list.iter().enumerate() {
        let lambda = rho[k].exp();
        s_rho = s_rho + s.mapv(|v| v * lambda);
    }
    
    // Compute H_phi using user's definition (WITH S)
    let h_phi = compute_firth_h_phi(&x, &w, &mu, Some(&s_rho));
    
    // 4. Perturb and compute derivative
    let epsilon = 1e-4;
    let k_target = 0; // Check for first rho
    
    let mut rho_plus = rho.clone();
    rho_plus[k_target] += epsilon;
    let mut s_rho_plus = Array2::<f64>::zeros((p, p));
    for (k, s) in s_list.iter().enumerate() {
        let lambda = rho_plus[k].exp();
        s_rho_plus = s_rho_plus + s.mapv(|v| v * lambda);
    }
    
    let h_phi_plus = compute_firth_h_phi(&x, &w, &mu, Some(&s_rho_plus));
    
    let d_h_phi = (&h_phi_plus - &h_phi) / epsilon;
    
    // 5. Compute H_total and Missing Term E
    // H_total = X'WX + S - H_phi
    let mut xtwx = Array2::<f64>::zeros((p, p));
    for i in 0..y.len() {
        let sw = w[i] * mu[i] * (1.0 - mu[i]); // approx weights for logit
        for r in 0..p {
            for c in 0..p {
                xtwx[[r, c]] += x[[i, r]] * sw * x[[i, c]];
            }
        }
    }
    
    let h_total = &xtwx + &s_rho - &h_phi;
    
    let scale_h = h_total.diag().iter().fold(0.0f64, |acc, v| acc.max(v.abs()));
    let mut h_total_ridge = h_total.clone();
    for i in 0..p { h_total_ridge[[i, i]] += scale_h * 1e-8; }
    
    let h_view = gnomon::calibrate::faer_ndarray::FaerArrayView::new(&h_total_ridge);
    use gnomon::calibrate::faer_ndarray::{FaerLlt, array2_to_mat_mut};
    let chol_h = FaerLlt::new(h_view.as_ref(), Side::Lower).unwrap();
    let mut id = Array2::<f64>::eye(p);
    let mut id_view = array2_to_mat_mut(&mut id);
    chol_h.solve_in_place(id_view.as_mut());
    let h_inv = id;

    let missing_term_matrix = h_inv.dot(&d_h_phi);
    let trace_val = missing_term_matrix.diag().sum();
    
    // E = -0.5 * trace(...)
    let e = -0.5 * trace_val;
    
    println!("  Analytic Gradient[{}]: {:.5e}", k_target, analytic_gradient[k_target]);
    println!("  Missing Term E: {:.5e}", e);
    
    // 6. Frozen FD Gradient (Standard from Code)
    let fd_frozen = frozen_fd_gradient_logit(&y, &x, &w, &s_list, &rho, &beta).unwrap();
    println!("  Frozen FD (Code Cost): {:.5e}", fd_frozen[k_target]);
    
    let corrected = analytic_gradient[k_target] + e;
    let diff = (corrected - fd_frozen[k_target]).abs();
    println!("  Corrected Analytic: {:.5e} (Diff from FD: {:.2e})", corrected, diff);
    
    // Assert match. 
    if diff > 1e-3 {
         panic!("Hypothesis verification failed: Corrected ({}) != Frozen FD ({})", corrected, fd_frozen[k_target]);
    } else {
        println!("✓ Hypothesis Verified: Missing term explains the discrepancy.");
    }
}
// ============================================================================
// Section 5: Truncation Correction + Ridge Hypothesis Test
// ============================================================================

/// Test the hypothesis that spectral truncation corrections should be SKIPPED
/// when ridge regularization is active.
///
/// Hypothesis:
/// - Cost with ridge uses full determinant: log|S_λ + ridge·I| (null space included)
/// - Cost without ridge uses pseudo-determinant: log|S|_+ (null space excluded)
/// - Gradient currently ALWAYS applies truncation correction (subtracts null-space contribution)
/// - This is inconsistent: when ridge is active, the null-space is in the cost, so the gradient
///   should NOT subtract the truncation correction.
///
/// Test strategy:
/// 1. Create a rank-deficient penalty (non-zero truncation correction)
/// 2. Use high smoothing to ensure ridge is triggered
/// 3. Compute analytic gradient (with truncation correction applied by current code)
/// 4. Compute FD gradient directly from cost
/// 5. Determine if mismatch is present when ridge is active
#[test]
fn isolation_truncation_correction_ridge_hypothesis() {
    println!("\n=== HYPOTHESIS TEST: Truncation Correction + Ridge Interaction ===");
    
    // 1. Create a problem with rank-deficient penalty (difference matrix has null space)
    let n = 100;
    let p = 10;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    // Difference penalty with order 2 has 2-dimensional null space
    let s = create_difference_penalty_matrix(p, 2).expect("difference penalty");
    let s_list = vec![s];
    let nullspace_dims = vec![2]; // Order-2 difference has 2D null space
    
    // 2. Use high smoothing (large λ) to trigger ridge adjustment in PIRLS
    let rho = array![8.0]; // λ = exp(8) ≈ 2981
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: None, // No Firth - isolate the truncation effect
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: nullspace_dims.clone(),
    };
    
    // 3. Get analytic gradient (with truncation correction applied)
    let offset = Array1::<f64>::zeros(n);
    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  Gradient evaluation failed: {:?}", e);
            panic!("Test setup failed");
        }
    };
    
    // 4. Get the ridge value used (to verify ridge was active)
    let (_, ridge_used) = evaluate_external_cost_and_ridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .expect("Cost evaluation failed");
    
    println!("  Ridge used: {:.3e}", ridge_used);
    println!("  Analytic gradient: {:.6e}", analytic[0]);
    println!("  FD gradient:       {:.6e}", fd[0]);
    
    // 5. Compute error metrics
    let error_with_correction = (analytic[0] - fd[0]).abs();
    let scale = fd[0].abs().max(analytic[0].abs()).max(1e-8);
    let rel_error = error_with_correction / scale;
    
    println!("  Absolute error: {:.3e}", error_with_correction);
    println!("  Relative error: {:.3e}", rel_error);
    
    // 6. Interpretation
    if ridge_used > 0.0 {
        println!("\n  [RIDGE ACTIVE] Testing if truncation correction causes mismatch...");
        
        if rel_error > 0.05 {
            println!("  ✓ Large relative error ({:.1}%) when ridge is active.", rel_error * 100.0);
            println!("    SUPPORTS hypothesis: truncation correction should be");
            println!("    skipped when ridge is active.");
        } else {
            println!("  Relative error is small ({:.1}%).", rel_error * 100.0);
        }
    } else {
        println!("\n  [NO RIDGE] Ridge was not triggered.");
    }
    
    assert!(
        analytic[0].is_finite() && fd[0].is_finite(),
        "Gradient values must be finite"
    );
}

/// Baseline test: Verify that WITHOUT ridge, gradients should match well.
#[test]
fn isolation_truncation_correction_no_ridge_baseline() {
    println!("\n=== BASELINE: Truncation Correction Without Ridge ===");
    
    let n = 150;
    let p = 8;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s = create_difference_penalty_matrix(p, 1).expect("difference penalty");
    let s_list = vec![s];
    let nullspace_dims = vec![1];
    
    let rho = array![0.0]; // λ = 1
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: None,
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let offset = Array1::<f64>::zeros(n);
    let (analytic, fd) = match evaluate_external_gradients(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    ) {
        Ok((a, f)) => (a, f),
        Err(e) => {
            println!("  Gradient evaluation failed: {:?}", e);
            return;
        }
    };
    
    let (_, ridge_used) = evaluate_external_cost_and_ridge(
        y.view(),
        w.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    )
    .expect("Cost evaluation failed");
    
    let rel_error = (analytic[0] - fd[0]).abs() / fd[0].abs().max(1e-8);
    
    println!("  Ridge used: {:.3e}", ridge_used);
    println!("  Analytic gradient: {:.6e}", analytic[0]);
    println!("  FD gradient:       {:.6e}", fd[0]);
    println!("  Relative error:    {:.3e}", rel_error);
    
    if ridge_used == 0.0 {
        println!("\n  [NO RIDGE] Baseline without ridge regularization.");
        if rel_error < 0.05 {
            println!("  ✓ Good agreement - truncation correction works correctly without ridge.");
        }
    }
    
    assert!(analytic[0].is_finite() && fd[0].is_finite());
}

// ============================================================================
// Section 6: Systematic Failure Isolation Tests
// ============================================================================

/// Hypothesis 2: Firth alone causes gradient mismatch.
/// Testing with single penalty + Firth to isolate the Firth effect.
#[test]
fn hypothesis_firth_alone_single_penalty() {
    println!("\n=== Hypothesis 2: Firth Alone (Single Penalty) ===");
    
    let n = 100;
    let p = 10;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s = create_difference_penalty_matrix(p, 2).expect("difference penalty");
    let s_list = vec![s];
    let nullspace_dims = vec![2];
    
    let rho = array![2.0]; // Moderate smoothing
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }), // <-- Firth ON
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let offset = Array1::<f64>::zeros(n);
    let (analytic, fd) = evaluate_external_gradients(
        y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
    ).expect("Gradient evaluation failed");
    
    let (cos, rel, max_a, max_f) = gradient_metrics(&analytic, &fd);
    print_result("Firth alone", cos, rel, max_a, max_f);
    
    let status = if cos > 0.99 && rel < 0.05 { "NEGATIVE" } else { "POSITIVE" };
    println!("  Hypothesis 2: {} (cos={:.4}, rel={:.3e})", status, cos, rel);
}

/// Hypothesis 3: Multi-penalty (no Firth) causes gradient mismatch.
#[test]
fn hypothesis_multi_penalty_no_firth() {
    println!("\n=== Hypothesis 3: Multi-Penalty (No Firth) ===");
    
    let n = 100;
    let p = 12;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let nullspace_dims = vec![1, 0, 0];
    
    // Anisotropic lambdas
    let rho = array![0.0, 4.6, -4.6]; // λ = [1, 100, 0.01]
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: None,
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let offset = Array1::<f64>::zeros(n);
    let (analytic, fd) = evaluate_external_gradients(
        y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
    ).expect("Gradient evaluation failed");
    
    let (cos, rel, max_a, max_f) = gradient_metrics(&analytic, &fd);
    print_result("Multi-penalty", cos, rel, max_a, max_f);
    
    let status = if cos > 0.99 && rel < 0.05 { "NEGATIVE" } else { "POSITIVE" };
    println!("  Hypothesis 3: {} (cos={:.4}, rel={:.3e})", status, cos, rel);
}

/// Hypothesis 4: Firth + high smoothing causes mismatch.
#[test]
fn hypothesis_firth_high_smoothing() {
    println!("\n=== Hypothesis 4: Firth + High Smoothing ===");
    
    let n = 100;
    let p = 10;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s = create_difference_penalty_matrix(p, 2).expect("difference penalty");
    let s_list = vec![s];
    let nullspace_dims = vec![2];
    
    // Very high smoothing
    let rho = array![10.0]; // λ = exp(10) ≈ 22026
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let offset = Array1::<f64>::zeros(n);
    let (analytic, fd) = evaluate_external_gradients(
        y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
    ).expect("Gradient evaluation failed");
    
    let (cos, rel, max_a, max_f) = gradient_metrics(&analytic, &fd);
    print_result("Firth+HighSmooth", cos, rel, max_a, max_f);
    
    let status = if cos > 0.99 && rel < 0.05 { "NEGATIVE" } else { "POSITIVE" };
    println!("  Hypothesis 4: {} (cos={:.4}, rel={:.3e})", status, cos, rel);
}

/// Hypothesis 5: Firth + anisotropic multi-penalty causes mismatch.
/// This matches the original failing test configuration.
#[test]
fn hypothesis_firth_anisotropic_multi_penalty() {
    println!("\n=== Hypothesis 5: Firth + Anisotropic Multi-Penalty ===");
    
    let n = 100;
    let p = 12;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let nullspace_dims = vec![1, 0, 0];
    
    // Anisotropic lambdas (same as original failing test)
    let rho = array![0.0, 4.6, -4.6];
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }), // Firth ON
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let offset = Array1::<f64>::zeros(n);
    let (analytic, fd) = evaluate_external_gradients(
        y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
    ).expect("Gradient evaluation failed");
    
    let (cos, rel, max_a, max_f) = gradient_metrics(&analytic, &fd);
    print_result("Firth+Aniso", cos, rel, max_a, max_f);
    
    let status = if cos > 0.99 && rel < 0.05 { "NEGATIVE" } else { "POSITIVE" };
    println!("  Hypothesis 5: {} (cos={:.4}, rel={:.3e})", status, cos, rel);
}

/// Hypothesis 6: Firth + rank-deficient penalty (difference matrix).
#[test]
fn hypothesis_firth_rank_deficient_penalty() {
    println!("\n=== Hypothesis 6: Firth + Rank-Deficient Penalty ===");
    
    let n = 100;
    let p = 12;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    // Use difference penalty which has a proper null space
    let s = create_difference_penalty_matrix(p, 2).expect("difference penalty");
    let s_list = vec![s];
    let nullspace_dims = vec![2];
    
    // High smoothing to stress the rank-deficiency
    let rho = array![6.0]; // λ ≈ 403
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let offset = Array1::<f64>::zeros(n);
    let (analytic, fd) = evaluate_external_gradients(
        y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
    ).expect("Gradient evaluation failed");
    
    let (cos, rel, max_a, max_f) = gradient_metrics(&analytic, &fd);
    print_result("Firth+RankDef", cos, rel, max_a, max_f);
    
    let status = if cos > 0.99 && rel < 0.05 { "NEGATIVE" } else { "POSITIVE" };
    println!("  Hypothesis 6: {} (cos={:.4}, rel={:.3e})", status, cos, rel);
}


/// Hypothesis 7: Full complexity scenario with overlapping penalties.
/// Firth + multi-penalty + anisotropic + high dimensionality.
#[test]
fn hypothesis_full_complexity_replication() {
    println!("\n=== Hypothesis 7: Full Complexity (Overlapping Penalties) ===");
    
    // Larger problem with overlapping penalties (simulating GAM structure)
    let n = 200;
    let p = 20;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    // Create overlapping penalties to simulate PGS + PC structure
    let s_list = vec![
        diagonal_penalty(p, 1, 8),   // "PGS main" effect
        diagonal_penalty(p, 6, 14),  // "PC1 main" effect (overlaps)
        diagonal_penalty(p, 12, 20), // "PC2 main" effect (overlaps)
    ];
    let nullspace_dims = vec![1, 0, 0];
    
    // Highly anisotropic lambdas (wide range, 4 orders of magnitude)
    let rho = array![3.0, 7.0, -1.0]; // Lambda = [20, 1097, 0.37]
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims: nullspace_dims.clone(),
    };
    
    let offset = Array1::<f64>::zeros(n);
    
    match evaluate_external_gradients(
        y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
    ) {
        Ok((analytic, fd)) => {
            let (cos, rel, max_a, max_f) = gradient_metrics(&analytic, &fd);
            print_result("FullComplex", cos, rel, max_a, max_f);
            
            let status = if cos > 0.99 && rel < 0.05 { "NEGATIVE" } else { "POSITIVE" };
            println!("  Hypothesis 7: {} (cos={:.4}, rel={:.3e})", status, cos, rel);
            
            // Print per-component breakdown
            if cos < 0.99 || rel > 0.05 {
                println!("\n  Per-component gradient comparison:");
                for k in 0..analytic.len().min(5) {
                    let diff = (analytic[k] - fd[k]).abs();
                    let scale = fd[k].abs().max(analytic[k].abs()).max(1e-10);
                    println!("    k={}: analytic={:.4e}, fd={:.4e}, rel_err={:.2e}", 
                             k, analytic[k], fd[k], diff/scale);
                }
            }
        }
        Err(e) => {
            println!("  ERROR: {:?}", e);
        }
    }
}

/// Hypothesis 8: Vary smoothing to find the breakpoint.
/// At what λ level does the gradient mismatch appear?
#[test]
fn hypothesis_smoothing_sweep() {
    println!("\n=== Hypothesis 8: Smoothing Sweep (Find Breakpoint) ===");
    
    let n = 100;
    let p = 10;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s = create_difference_penalty_matrix(p, 2).expect("difference penalty");
    let s_list = vec![s];
    let nullspace_dims = vec![2];
    
    let offset = Array1::<f64>::zeros(n);
    let rho_values = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
    
    println!("  rho     λ           cos      rel_err");
    println!("  ------- ----------- -------- --------");
    
    for &rho_val in &rho_values {
        let rho = array![rho_val];
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: nullspace_dims.clone(),
        };
        
        match evaluate_external_gradients(
            y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, _, _) = gradient_metrics(&analytic, &fd);
                let lambda = rho_val.exp();
                let status = if cos > 0.99 && rel < 0.05 { "✓" } else { "✗" };
                println!("  {:+6.1} {:11.2e} {:8.4} {:8.2e} {}", 
                         rho_val, lambda, cos, rel, status);
            }
            Err(_) => {
                println!("  {:+6.1} ERROR", rho_val);
            }
        }
    }
}

// ============================================================================
// Section 7: Deep Investigation of isolation_reparam_pgs_pc_mains_firth Failure
// ============================================================================

/// Hypothesis 9: Is it the number of penalties that causes the failure?
/// Test with 10 penalties but simpler structure.
#[test]
fn hypothesis_many_penalties_simple_structure() {
    println!("\n=== Hypothesis 9: Many Penalties (10) Simple Structure ===");
    
    let n = 100;
    let p = 25;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    // Create 10 non-overlapping diagonal penalties
    let mut s_list = Vec::new();
    for i in 0..10 {
        let start = 1 + (i * 2);
        let end = start + 3;
        s_list.push(diagonal_penalty(p, start.min(p), end.min(p)));
    }
    let nullspace_dims = vec![1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    
    // Use same extreme smoothing as failing test
    let rho = Array1::from_elem(10, 12.0);
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let offset = Array1::<f64>::zeros(n);
    
    match evaluate_external_gradients(
        y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
    ) {
        Ok((analytic, fd)) => {
            let (cos, rel, max_a, max_f) = gradient_metrics(&analytic, &fd);
            print_result("10 penalties", cos, rel, max_a, max_f);
            
            let status = if cos > 0.99 && rel < 0.05 { "NEGATIVE" } else { "POSITIVE" };
            println!("  Hypothesis 9: {} (cos={:.4}, rel={:.3e})", status, cos, rel);
            println!("  Gradient magnitude: analytic={:.2e}, fd={:.2e}", max_a, max_f);
        }
        Err(e) => {
            println!("  ERROR: {:?}", e);
        }
    }
}

/// Hypothesis 10: Is it the extreme smoothing (rho=12) that causes tiny gradients
/// to be unreliable?
#[test]
fn hypothesis_extreme_smoothing_tiny_gradients() {
    println!("\n=== Hypothesis 10: Extreme Smoothing -> Tiny Gradients ===");
    
    let n = 100;
    let p = 12;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let nullspace_dims = vec![1, 0, 0];
    
    // Sweep through extreme smoothing values
    let rho_values = [8.0, 10.0, 12.0, 14.0, 16.0];
    let offset = Array1::<f64>::zeros(n);
    
    println!("  rho    λ           cos       rel_err   |grad|");
    println!("  ------ ----------- --------- --------- ---------");
    
    for &rho_val in &rho_values {
        let rho = Array1::from_elem(3, rho_val);
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: nullspace_dims.clone(),
        };
        
        match evaluate_external_gradients(
            y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
                let lambda = rho_val.exp();
                let status = if cos > 0.99 && rel < 0.05 { "✓" } else { "✗" };
                println!("  {:+5.1} {:11.2e} {:9.4} {:9.2e} {:9.2e} {}", 
                         rho_val, lambda, cos, rel, max_a, status);
            }
            Err(_) => {
                println!("  {:+5.1} ERROR", rho_val);
            }
        }
    }
}

/// Hypothesis 11: Does using real GAM design matrices trigger the failure?
/// Use build_design_and_penalty_matrices with fewer penalties to isolate.
#[test]
fn hypothesis_real_gam_matrices_small() {
    println!("\n=== Hypothesis 11: Real GAM Matrices (Small Config) ===");
    
    let train = create_logistic_training_data(100, 1, 31); // Only 1 PC
    let config = logistic_model_config(true, false, &train); // PC mains only
    
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    
    println!("  Design dimensions: {} x {}", x.nrows(), x.ncols());
    println!("  Number of penalties: {}", s_list.len());
    
    let nullspace_dims = vec![0; s_list.len()];
    let offset = Array1::<f64>::zeros(train.y.len());
    
    // Test at moderate smoothing first
    for &rho_val in &[0.0, 6.0, 12.0] {
        let rho = Array1::from_elem(layout.num_penalties, rho_val);
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: nullspace_dims.clone(),
        };
        
        match evaluate_external_gradients(
            train.y.view(),
            train.weights.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.99 && rel < 0.05 { "✓" } else { "✗" };
                println!("  rho={:+5.1}: cos={:.4}, rel={:.2e}, |grad|={:.2e} {}", 
                         rho_val, cos, rel, max_a, status);
            }
            Err(e) => {
                println!("  rho={:+5.1}: ERROR {:?}", rho_val, e);
            }
        }
    }
}

/// Hypothesis 12: Test gradient quality as a function of gradient magnitude.
/// The hypothesis is that when |grad| < 1e-4, numerical precision dominates.
#[test]
fn hypothesis_gradient_magnitude_threshold() {
    println!("\n=== Hypothesis 12: Gradient Magnitude vs Quality ===");
    
    let n = 100;
    let p = 10;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    let s = create_difference_penalty_matrix(p, 2).expect("penalty");
    let s_list = vec![s];
    let nullspace_dims = vec![2];
    let offset = Array1::<f64>::zeros(n);
    
    // Collect (grad_magnitude, cosine, rel_error) across smoothing values
    println!("  |grad|       cos       rel_err   Quality");
    println!("  ----------   -------- --------- ---------");
    
    for rho_val in (-4..=16).step_by(2) {
        let rho = array![rho_val as f64];
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: nullspace_dims.clone(),
        };
        
        match evaluate_external_gradients(
            y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
                let quality = if cos > 0.999 && rel < 0.01 { 
                    "EXCELLENT" 
                } else if cos > 0.99 && rel < 0.05 { 
                    "GOOD" 
                } else if cos > 0.95 { 
                    "MARGINAL" 
                } else { 
                    "BAD" 
                };
                println!("  {:10.2e}   {:8.4} {:9.2e} {}", max_a, cos, rel, quality);
            }
            Err(_) => {
                println!("  rho={}  ERROR", rho_val);
            }
        }
    }
}

/// Hypothesis 13: The exact failing configuration - build_design_and_penalty_matrices
/// with 3 PCs and rho=12. What if we vary the number of PCs?
#[test]
fn hypothesis_varying_pc_count() {
    println!("\n=== Hypothesis 13: Varying PC Count ===");
    
    for num_pcs in [1, 2, 3, 4] {
        let train = create_logistic_training_data(100, num_pcs, 31);
        let config = logistic_model_config(true, false, &train);
        
        let (x, s_list, layout, ..) =
            build_design_and_penalty_matrices(&train, &config).expect("design");
        
        let rho = Array1::from_elem(layout.num_penalties, 12.0);
        let nullspace_dims = vec![0; s_list.len()];
        let offset = Array1::<f64>::zeros(train.y.len());
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims,
        };
        
        match evaluate_external_gradients(
            train.y.view(),
            train.weights.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.99 && rel < 0.05 { "✓" } else { "✗" };
                println!("  {} PCs -> {} penalties: cos={:.4}, rel={:.2e}, |grad|={:.2e} {}", 
                         num_pcs, layout.num_penalties, cos, rel, max_a, status);
            }
            Err(e) => {
                println!("  {} PCs: ERROR {:?}", num_pcs, e);
            }
        }
    }
}

/// Hypothesis 14: Is it the spline difference penalty structure that causes failure?
/// Use real spline penalty matrices but with simple synthetic data.
#[test]
fn hypothesis_spline_penalty_synthetic_data() {
    println!("\n=== Hypothesis 14: Spline Penalty + Synthetic Data ===");
    
    // Create difference penalties like real GAM uses
    let p = 20;
    let n = 120;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    // Use multiple difference penalties (order 2) like real splines
    let s1 = create_difference_penalty_matrix(8, 2).expect("penalty 1");
    let s2 = create_difference_penalty_matrix(6, 2).expect("penalty 2");  
    let s3 = create_difference_penalty_matrix(6, 2).expect("penalty 3");
    
    // Embed them in a block-diagonal structure within the full p dimensions
    let mut s1_full = Array2::<f64>::zeros((p, p));
    let mut s2_full = Array2::<f64>::zeros((p, p));
    let mut s3_full = Array2::<f64>::zeros((p, p));
    
    for i in 0..8 {
        for j in 0..8 {
            s1_full[[i, j]] = s1[[i, j]];
        }
    }
    for i in 0..6 {
        for j in 0..6 {
            s2_full[[8+i, 8+j]] = s2[[i, j]];
        }
    }
    for i in 0..6 {
        for j in 0..6 {
            s3_full[[14+i, 14+j]] = s3[[i, j]];
        }
    }
    
    let s_list = vec![s1_full, s2_full, s3_full];
    // Nullspace dims: order-2 difference has 2D null space each
    let nullspace_dims = vec![2, 2, 2];
    
    let offset = Array1::<f64>::zeros(n);
    
    for &rho_val in &[0.0, 6.0, 12.0] {
        let rho = Array1::from_elem(3, rho_val);
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: nullspace_dims.clone(),
        };
        
        match evaluate_external_gradients(
            y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.99 && rel < 0.05 { "✓" } else { "✗" };
                println!("  rho={:+5.1}: cos={:.4}, rel={:.2e}, |grad|={:.2e} {}", 
                         rho_val, cos, rel, max_a, status);
            }
            Err(e) => {
                println!("  rho={:+5.1}: ERROR {:?}", rho_val, e);
            }
        }
    }
}

/// Hypothesis 15: Does the problem come from specific penalty overlap patterns?
/// Create penalties that overlap like real GAM does.
#[test]
fn hypothesis_overlapping_penalty_structure() {
    println!("\n=== Hypothesis 15: Overlapping Penalty Structure ===");
    
    let p = 20;
    let n = 120;
    let (y, x, w) = generate_logit_data(n, p, 42);
    
    // Create OVERLAPPING penalties - this is what real GAM does
    // Penalty 1: covers indices 0-10
    // Penalty 2: covers indices 5-15 (overlaps with 1)
    // Penalty 3: covers indices 10-20 (overlaps with 1 and 2)
    let mut s1 = Array2::<f64>::zeros((p, p));
    let mut s2 = Array2::<f64>::zeros((p, p));
    let mut s3 = Array2::<f64>::zeros((p, p));
    
    // Fill with difference penalty structure in each block
    let d1 = create_difference_penalty_matrix(11, 2).expect("d1");
    let d2 = create_difference_penalty_matrix(11, 2).expect("d2");
    let d3 = create_difference_penalty_matrix(10, 2).expect("d3");
    
    for i in 0..11 {
        for j in 0..11 {
            s1[[i, j]] = d1[[i, j]];
        }
    }
    for i in 0..11 {
        for j in 0..11 {
            s2[[5+i, 5+j]] = d2[[i, j]];
        }
    }
    for i in 0..10 {
        for j in 0..10 {
            s3[[10+i, 10+j]] = d3[[i, j]];
        }
    }
    
    let s_list = vec![s1, s2, s3];
    let nullspace_dims = vec![2, 2, 2];
    
    let offset = Array1::<f64>::zeros(n);
    
    for &rho_val in &[0.0, 6.0, 12.0] {
        let rho = Array1::from_elem(3, rho_val);
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: nullspace_dims.clone(),
        };
        
        match evaluate_external_gradients(
            y.view(), w.view(), x.view(), offset.view(), &s_list, &opts, &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.99 && rel < 0.05 { "✓" } else { "✗" };
                println!("  rho={:+5.1}: cos={:.4}, rel={:.2e}, |grad|={:.2e} {}", 
                         rho_val, cos, rel, max_a, status);
            }
            Err(e) => {
                println!("  rho={:+5.1}: ERROR {:?}", rho_val, e);
            }
        }
    }
}

/// Hypothesis 16: Is it the reparameterization that triggers the failure?
/// Use actual reparameterization but skip constraint projection.
#[test]
fn hypothesis_reparameterization_trigger() {
    println!("\n=== Hypothesis 16: Reparameterization as Trigger ===");
    
    // Get actual reparameterized matrices from real GAM construction
    let train = create_logistic_training_data(100, 2, 31);
    let config = logistic_model_config(true, false, &train);
    
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    
    println!("  Design: {} x {}, {} penalties", x.nrows(), x.ncols(), s_list.len());
    
    // Test WITHOUT reparameterization - use raw matrices
    // Note: s_list is already the penalties, x is the design
    
    let nullspace_dims = vec![0; s_list.len()];
    let offset = Array1::<f64>::zeros(train.y.len());
    
    // Get the reparameterization to see its condition number
    let rs_list = compute_penalty_square_roots(&s_list).unwrap();
    let lambdas: Vec<f64> = vec![1.0; s_list.len()];
    let reparam = stable_reparameterization(&rs_list, &lambdas, &layout).unwrap();
    
    println!("  Reparam.s_transformed dims: {} x {}", 
             reparam.s_transformed.nrows(), reparam.s_transformed.ncols());
    println!("  Reparam.qs dims: {} x {}", 
             reparam.qs.nrows(), reparam.qs.ncols());
    
    // Check condition number of transformed penalty
    let s_diag: Vec<f64> = (0..reparam.s_transformed.nrows())
        .map(|i| reparam.s_transformed[[i, i]])
        .collect();
    let s_max = s_diag.iter().fold(0.0f64, |a, &b| a.max(b));
    let s_min = s_diag.iter().filter(|&&x| x > 1e-12).fold(f64::INFINITY, |a, &b| a.min(b));
    let cond = s_max / s_min;
    println!("  s_transformed condition (diag): {:.2e} / {:.2e} = {:.2e}", s_max, s_min, cond);
    
    // Now test at different rho values
    for &rho_val in &[0.0, 6.0, 12.0] {
        let rho = Array1::from_elem(layout.num_penalties, rho_val);
        
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: nullspace_dims.clone(),
        };
        
        match evaluate_external_gradients(
            train.y.view(),
            train.weights.view(),
            x.view(),
            offset.view(),
            &s_list,
            &opts,
            &rho,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.99 && rel < 0.05 { "✓" } else { "✗" };
                
                // Check if any FD components have very small magnitude
                let tiny_count = fd.iter().filter(|&&v| v.abs() < 1e-6).count();
                println!("  rho={:+5.1}: cos={:.4}, rel={:.2e}, |grad|={:.2e}, tiny={} {}", 
                         rho_val, cos, rel, max_a, tiny_count, status);
            }
            Err(e) => {
                println!("  rho={:+5.1}: ERROR {:?}", rho_val, e);
            }
        }
    }
}

/// Hypothesis 17: Check if the failure correlates with near-zero gradient components.
/// Components with |grad| < 1e-6 may be numerically unreliable.
#[test]
fn hypothesis_near_zero_components() {
    println!("\n=== Hypothesis 17: Near-Zero Component Analysis ===");
    
    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    
    let (x, s_list, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    
    let rho = Array1::from_elem(layout.num_penalties, 12.0);
    let nullspace_dims = vec![0; s_list.len()];
    let offset = Array1::<f64>::zeros(train.y.len());
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    let (analytic, fd) = evaluate_external_gradients(
        train.y.view(),
        train.weights.view(),
        x.view(),
        offset.view(),
        &s_list,
        &opts,
        &rho,
    ).expect("gradients");
    
    println!("  Per-component analysis:");
    println!("  k    analytic       fd             diff          rel_err");
    println!("  ---  ------------   ------------   ------------  --------");
    
    let mut large_error_tiny_grad = 0;
    let mut large_error_normal_grad = 0;
    
    for k in 0..analytic.len() {
        let a = analytic[k];
        let f = fd[k];
        let diff = (a - f).abs();
        let scale = a.abs().max(f.abs()).max(1e-10);
        let rel = diff / scale;
        
        let note = if f.abs() < 1e-6 {
            if rel > 0.1 {
                large_error_tiny_grad += 1;
            }
            "TINY"
        } else if rel > 0.1 {
            large_error_normal_grad += 1;
            "ERROR"
        } else {
            ""
        };
        
        println!("  {:2}   {:+12.4e}   {:+12.4e}   {:+12.4e}  {:8.2e}  {}", 
                 k, a, f, diff, rel, note);
    }
    
    println!("\n  Summary:");
    println!("    Components with large error AND tiny gradient: {}", large_error_tiny_grad);
    println!("    Components with large error AND normal gradient: {}", large_error_normal_grad);
    
    if large_error_normal_grad == 0 && large_error_tiny_grad > 0 {
        println!("  => Hypothesis SUPPORTED: Errors only in tiny gradient components");
    } else if large_error_normal_grad > 0 {
        println!("  => Hypothesis NOT SUPPORTED: Errors exist in normal gradient components");
    }
}


/// Hypothesis 18: Use GAM S matrices but with random X.
/// This tests if the problem is S-related or X-related.
#[test]
fn hypothesis_gam_s_random_x() {
    println!("\n=== Hypothesis 18: GAM S Matrices + Random X ===");
    
    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    
    let p = x_gam.ncols();
    let n = x_gam.nrows();
    
    // Create random X with same dimensions
    let (y_rand, x_rand, w_rand) = generate_logit_data(n, p, 42);
    
    let nullspace_dims = vec![0; s_list_gam.len()];
    let offset = Array1::<f64>::zeros(n);
    let rho = Array1::from_elem(layout.num_penalties, 12.0);
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    // Test: Use GAM penalties with random X and random y
    let (analytic, fd) = evaluate_external_gradients(
        y_rand.view(), w_rand.view(), x_rand.view(), offset.view(),
        &s_list_gam, &opts, &rho,
    ).expect("gradients");
    
    let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
    let status = if cos > 0.99 && rel < 0.05 { "✓ PASS" } else { "✗ FAIL" };
    println!("  GAM S + Random X: cos={:.4}, rel={:.2e}, |grad|={:.2e} {}", cos, rel, max_a, status);
    
    if cos > 0.99 {
        println!("  => S matrices alone do NOT trigger failure");
    } else {
        println!("  => S matrices DO trigger failure (even with random X)");
    }
}

/// Hypothesis 19: Use random S matrices but with GAM X and y.
#[test]
fn hypothesis_random_s_gam_xy() {
    println!("\n=== Hypothesis 19: Random S + GAM X and Y ===");
    
    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");
    
    let p = x_gam.ncols();
    
    // Create simple diagonal penalties matching the number of GAM penalties
    let mut s_list_diag = Vec::new();
    for k in 0..s_list_gam.len() {
        s_list_diag.push(diagonal_penalty(p, k * (p / s_list_gam.len()), (k+1) * (p / s_list_gam.len())));
    }
    
    let nullspace_dims = vec![0; s_list_gam.len()];
    let offset = Array1::<f64>::zeros(train.y.len());
    let rho = Array1::from_elem(layout.num_penalties, 12.0);
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    // Test: Use diagonal penalties with GAM X and y
    let (analytic, fd) = evaluate_external_gradients(
        train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
        &s_list_diag, &opts, &rho,
    ).expect("gradients");
    
    let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
    let status = if cos > 0.99 && rel < 0.05 { "✓ PASS" } else { "✗ FAIL" };
    println!("  Random S + GAM X: cos={:.4}, rel={:.2e}, |grad|={:.2e} {}", cos, rel, max_a, status);
    
    if cos > 0.99 {
        println!("  => GAM X/y alone do NOT trigger failure");
    } else {
        println!("  => GAM X/y DO trigger failure (even with diagonal S)");
    }
}

/// DIAGNOSTIC: Deep analysis of spectral vs expected gradient computation.
/// This test examines each component of the Firth gradient to find exact
/// point of divergence between analytic and finite-difference gradients.
#[test]
fn diagnostic_deep_gradient_analysis() {
    println!("\n=== DIAGNOSTIC: Deep Gradient Component Analysis ===");

    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");

    let n = x_gam.nrows();
    let p = x_gam.ncols();
    let k = s_list_gam.len();
    println!("  Dimensions: n={}, p={}, k={} penalties", n, p, k);

    // Compute eigenvalue analysis of penalty structure
    let rho_val = 12.0_f64;
    let rho = Array1::from_elem(layout.num_penalties, rho_val);

    // Build S_lambda = sum(exp(rho_k) * S_k)
    let mut s_lambda = Array2::<f64>::zeros((p, p));
    for (idx, s_k) in s_list_gam.iter().enumerate() {
        let lambda_k = rho[idx].exp();
        for i in 0..p {
            for j in 0..p {
                s_lambda[[i, j]] += lambda_k * s_k[[i, j]];
            }
        }
    }

    // Eigenvalue analysis of S_lambda
    let (s_eigvals, _): (Array1<f64>, _) = s_lambda.eigh(Side::Lower).expect("S eigh");
    let s_pos = s_eigvals.iter().filter(|v| **v > 1e-10).count();
    let s_neg = s_eigvals.iter().filter(|v| **v < -1e-10).count();
    let s_zero = p - s_pos - s_neg;
    println!("  S_lambda spectrum: pos={}, neg={}, zero={}", s_pos, s_neg, s_zero);
    println!("  S_lambda eigenrange: [{:.2e}, {:.2e}]",
        s_eigvals.iter().cloned().fold(f64::INFINITY, f64::min),
        s_eigvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // Test gradient at multiple rho values
    let offset = Array1::<f64>::zeros(n);
    for rho_test in [0.0_f64, 6.0, 12.0] {
        let rho_vec = Array1::from_elem(k, rho_test);
        println!("\n  --- rho = {:.1} ---", rho_test);

        // WITHOUT Firth
        let opts_no_firth = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: None,
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![0; k],
        };

        match evaluate_external_gradients(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &s_list_gam, &opts_no_firth, &rho_vec,
        ) {
            Ok((analytic, fd)) => {
                let (cos, ..) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.999 { "✓" } else { "✗" };
                println!("    No Firth: cos={:.6} {}", cos, status);
            }
            Err(e) => println!("    No Firth: ERROR {:?}", e),
        }

        // WITH Firth
        let opts_firth = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![0; k],
        };

        match evaluate_external_gradients(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &s_list_gam, &opts_firth, &rho_vec,
        ) {
            Ok((analytic, fd)) => {
                let (cos, rel, max_a, max_fd) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.999 { "✓" } else { "✗" };
                println!("    Firth: cos={:.6} {} (rel={:.2e}, |a|={:.2e}, |fd|={:.2e})",
                    cos, status, rel, max_a, max_fd);

                // If failing, show per-component analysis
                if cos < 0.999 && k <= 15 {
                    println!("    Per-component (analytic - fd):");
                    for (i, (a, f)) in analytic.iter().zip(fd.iter()).enumerate() {
                        let d = a - f;
                        let rel_err = if f.abs() > 1e-12 { (d / f).abs() } else { f64::INFINITY };
                        let flag = if rel_err > 0.1 { " <-- SUSPECT" } else { "" };
                        println!("      [{:2}] a={:+.4e}, fd={:+.4e}, diff={:+.4e}, rel={:.1}%{}",
                            i, a, f, d, 100.0 * rel_err, flag);
                    }
                }
            }
            Err(e) => println!("    Firth: ERROR {:?}", e),
        }
    }
}

/// DIAGNOSTIC: Map penalty indices to their structure and rank
#[test]
fn diagnostic_penalty_structure_analysis() {
    println!("\n=== DIAGNOSTIC: Penalty Structure Analysis ===");

    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");

    let p = x_gam.ncols();
    println!("  Total parameters: {}", p);
    println!("  Number of penalties: {}", s_list_gam.len());

    // Analyze each penalty matrix
    for (idx, s_k) in s_list_gam.iter().enumerate() {
        // Compute eigenvalues to understand structure
        let (eigvals, _): (Array1<f64>, _) = s_k.eigh(Side::Lower).expect("eigh");
        let pos_count = eigvals.iter().filter(|v| **v > 1e-10).count();
        let zero_count = eigvals.iter().filter(|v| v.abs() <= 1e-10).count();
        let neg_count = eigvals.iter().filter(|v| **v < -1e-10).count();

        let max_eig = eigvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_eig = eigvals.iter().cloned().fold(f64::INFINITY, f64::min);

        // Compute Frobenius norm
        let frob: f64 = s_k.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Check which parameters are penalized
        let diag: Vec<f64> = (0..p).map(|i| s_k[[i, i]]).collect();
        let penalized_range: Vec<usize> = diag.iter().enumerate()
            .filter(|(_, v)| **v > 1e-10)
            .map(|(i, _)| i)
            .collect();

        let start_idx = penalized_range.first().copied().unwrap_or(0);
        let end_idx = penalized_range.last().copied().unwrap_or(0);

        // Flag if this is a suspect component (1, 2, 4, 6)
        let suspect = matches!(idx, 1 | 2 | 4 | 6);
        let flag = if suspect { " <-- SUSPECT" } else { "" };

        println!("  Penalty [{}]{}:", idx, flag);
        println!("    Eigenvalues: pos={}, zero={}, neg={}", pos_count, zero_count, neg_count);
        println!("    Eigenrange: [{:.2e}, {:.2e}]", min_eig, max_eig);
        println!("    Frobenius norm: {:.2e}", frob);
        println!("    Penalized param range: [{}..{}]", start_idx, end_idx);
    }

    // Also print the layout info if available
    println!("\n  Layout info:");
    println!("    Intercept col: {}", layout.intercept_col);
    println!("    PGS main cols: {:?}", layout.pgs_main_cols);
    println!("    Num penalties: {}", layout.num_penalties);
}

/// DIAGNOSTIC: Test each penalty individually to isolate the problem
#[test]
fn diagnostic_individual_penalty_test() {
    println!("\n=== DIAGNOSTIC: Individual Penalty Test ===");

    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");

    let n = x_gam.nrows();
    let offset = Array1::<f64>::zeros(n);
    let rho_single = array![12.0_f64];

    println!("  Testing each penalty individually with Firth:");

    // Test each penalty individually
    for (idx, s_k) in s_list_gam.iter().enumerate() {
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![0],
        };

        match evaluate_external_gradients(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &[s_k.clone()], &opts, &rho_single,
        ) {
            Ok((analytic, fd)) => {
                let (cos, ..) = gradient_metrics(&analytic, &fd);
                let suspect = matches!(idx, 1 | 2 | 4 | 6);
                let flag = if suspect { " <-- was suspect in multi-penalty" } else { "" };
                let status = if cos > 0.999 { "✓" } else { "✗" };
                println!("    Penalty [{}]: cos={:.6} {}{}", idx, cos, status, flag);
            }
            Err(e) => {
                println!("    Penalty [{}]: ERROR {:?}", idx, e);
            }
        }
    }

    // Now test consecutive pairs to find interaction effects
    println!("\n  Testing consecutive penalty pairs with Firth:");
    for i in 0..s_list_gam.len()-1 {
        let s_pair = vec![s_list_gam[i].clone(), s_list_gam[i+1].clone()];
        let rho_pair = array![12.0_f64, 12.0_f64];
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![0, 0],
        };

        match evaluate_external_gradients(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &s_pair, &opts, &rho_pair,
        ) {
            Ok((analytic, fd)) => {
                let (cos, ..) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.999 { "✓" } else { "✗" };
                println!("    Pair [{}, {}]: cos={:.6} {}", i, i+1, cos, status);
            }
            Err(e) => {
                println!("    Pair [{}, {}]: ERROR {:?}", i, i+1, e);
            }
        }
    }
}

/// DIAGNOSTIC: Deep investigation of penalty [4] which has cos=-1
#[test]
fn diagnostic_penalty_4_deep_investigation() {
    println!("\n=== DIAGNOSTIC: Deep Investigation of Penalty [4] (cos=-1) ===");

    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");

    let n = x_gam.nrows();
    let p = x_gam.ncols();
    let offset = Array1::<f64>::zeros(n);

    println!("  Comparing penalty structures:");
    for idx in [3, 4, 5] {
        let s_k = &s_list_gam[idx];

        // Check for any negative elements
        let min_val = s_k.iter().fold(f64::INFINITY, |a, b| a.min(*b));
        let max_val = s_k.iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
        let has_neg = s_k.iter().any(|v| *v < -1e-15);

        // Sum of diagonal
        let diag_sum: f64 = (0..p).map(|i| s_k[[i, i]]).sum();

        // Off-diagonal sum
        let mut offdiag_sum = 0.0_f64;
        for i in 0..p {
            for j in 0..p {
                if i != j {
                    offdiag_sum += s_k[[i, j]];
                }
            }
        }

        let flag = if idx == 4 { " <-- PROBLEM PENALTY" } else { "" };
        println!("  Penalty [{}]{}:", idx, flag);
        println!("    Value range: [{:.4e}, {:.4e}]", min_val, max_val);
        println!("    Has negative: {}", has_neg);
        println!("    Diagonal sum: {:.4e}", diag_sum);
        println!("    Off-diag sum: {:.4e}", offdiag_sum);

        // Print the actual non-zero structure
        let mut nonzero_entries: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..p {
            for j in 0..p {
                if s_k[[i, j]].abs() > 1e-12 {
                    nonzero_entries.push((i, j, s_k[[i, j]]));
                }
            }
        }
        if nonzero_entries.len() <= 20 {
            println!("    Non-zero entries: {:?}", nonzero_entries);
        } else {
            println!("    Non-zero entries count: {}", nonzero_entries.len());
        }
    }

    // Test penalty 4 at different rho values
    println!("\n  Penalty [4] gradient at different rho:");
    for rho_val in [0.0_f64, 6.0, 12.0] {
        let rho_single = array![rho_val];
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![0],
        };

        match evaluate_external_gradients(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &[s_list_gam[4].clone()], &opts, &rho_single,
        ) {
            Ok((analytic, fd)) => {
                let (cos, ..) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.999 { "✓" } else if cos < -0.999 { "✗ OPPOSITE!" } else { "✗" };
                println!("    rho={:.1}: cos={:.6} {} a={:.4e} fd={:.4e}",
                    rho_val, cos, status, analytic[0], fd[0]);
            }
            Err(e) => println!("    rho={:.1}: ERROR {:?}", rho_val, e),
        }
    }

    // Compare to non-Firth
    println!("\n  Penalty [4] gradient WITHOUT Firth:");
    for rho_val in [0.0_f64, 6.0, 12.0] {
        let rho_single = array![rho_val];
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: None,  // No Firth
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![0],
        };

        match evaluate_external_gradients(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &[s_list_gam[4].clone()], &opts, &rho_single,
        ) {
            Ok((analytic, fd)) => {
                let (cos, ..) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.999 { "✓" } else { "✗" };
                println!("    rho={:.1}: cos={:.6} {} a={:.4e} fd={:.4e}",
                    rho_val, cos, status, analytic[0], fd[0]);
            }
            Err(e) => println!("    rho={:.1}: ERROR {:?}", rho_val, e),
        }
    }
}

/// DIAGNOSTIC: Find exact rho threshold where penalty 4 bug appears
#[test]
fn diagnostic_penalty_4_rho_threshold() {
    println!("\n=== DIAGNOSTIC: Finding Exact Rho Threshold for Penalty [4] Bug ===");

    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");

    let n = x_gam.nrows();
    let offset = Array1::<f64>::zeros(n);

    println!("  Sweeping rho values to find threshold:");

    // Test rho from 6 to 12 in steps
    for rho_val in [6.0, 7.0, 8.0, 9.0, 10.0, 10.5, 11.0, 11.5, 12.0, 13.0, 14.0, 15.0] {
        let rho_single = array![rho_val];
        let opts = ExternalOptimOptions {
            link: LinkFunction::Logit,
            firth: Some(FirthSpec { enabled: true }),
            tol: 1e-10,
            max_iter: 200,
            nullspace_dims: vec![0],
        };

        match evaluate_external_gradients(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &[s_list_gam[4].clone()], &opts, &rho_single,
        ) {
            Ok((analytic, fd)) => {
                let (cos, ..) = gradient_metrics(&analytic, &fd);
                let status = if cos > 0.99 {
                    "✓"
                } else if cos < -0.99 {
                    "✗ OPPOSITE!"
                } else if cos > 0.0 {
                    "~ degraded"
                } else {
                    "✗ wrong sign"
                };
                let lambda = rho_val.exp();
                println!("    rho={:5.1} (λ={:.2e}): cos={:+.6} {} a={:+.4e} fd={:+.4e}",
                    rho_val, lambda, cos, status, analytic[0], fd[0]);
            }
            Err(e) => {
                println!("    rho={:5.1}: ERROR {:?}", rho_val, e);
            }
        }
    }
}

/// DIAGNOSTIC: Verify gradient direction via cost function trend
/// At rho=12, which gradient sign is ACTUALLY correct?
#[test]
fn diagnostic_verify_gradient_via_cost_trend() {
    println!("\n=== DIAGNOSTIC: Verify Gradient Direction via Cost Trend ===");

    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");

    let n = x_gam.nrows();
    let offset = Array1::<f64>::zeros(n);

    // At rho=12, the bug shows analytic=-2.25e-6 (negative) but FD=+2.75e-6 (positive)
    // If the cost INCREASES with rho, the gradient should be POSITIVE
    // If the cost DECREASES with rho, the gradient should be NEGATIVE

    println!("  Evaluating cost at rho values around 12.0:");
    println!("  (If cost increases → gradient should be positive)");
    println!("  (If cost decreases → gradient should be negative)\n");

    // Use penalty 4 only
    let s_4 = s_list_gam[4].clone();

    // Evaluate cost at several rho values
    let rho_values: Vec<f64> = vec![11.9, 11.95, 12.0, 12.05, 12.1];

    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-12,  // Very tight tolerance
        max_iter: 500,
        nullspace_dims: vec![0],
    };

    let mut costs: Vec<(f64, f64)> = Vec::new();

    for &rho_val in &rho_values {
        let rho_single = array![rho_val];
        match evaluate_external_cost_and_ridge(
            train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
            &[s_4.clone()], &opts, &rho_single,
        ) {
            Ok((cost, ..)) => {
                costs.push((rho_val, cost));
                println!("    rho={:.4}: cost={:.12}", rho_val, cost);
            }
            Err(e) => {
                println!("    rho={:.4}: ERROR {:?}", rho_val, e);
            }
        }
    }

    // Calculate numerical gradient from costs
    if costs.len() >= 2 {
        println!("\n  Numerical gradient from cost differences:");
        for i in 1..costs.len() {
            let (rho1, c1) = costs[i-1];
            let (rho2, c2) = costs[i];
            let d_rho = rho2 - rho1;
            let d_cost = c2 - c1;
            let approx_grad = d_cost / d_rho;
            let sign_word = if approx_grad > 0.0 { "POSITIVE" } else { "NEGATIVE" };
            println!("    [{:.3}→{:.3}]: Δcost/Δrho = {:.6e} ({})",
                rho1, rho2, approx_grad, sign_word);
        }
    }

    // Now compare to our gradients
    println!("\n  Compare to computed gradients at rho=12.0:");
    let rho_12 = array![12.0_f64];
    match evaluate_external_gradients(
        train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
        &[s_4.clone()], &opts, &rho_12,
    ) {
        Ok((analytic, fd)) => {
            let a_sign = if analytic[0] > 0.0 { "POSITIVE" } else { "NEGATIVE" };
            let fd_sign = if fd[0] > 0.0 { "POSITIVE" } else { "NEGATIVE" };
            println!("    Analytic:  {:.6e} ({})", analytic[0], a_sign);
            println!("    FD:        {:.6e} ({})", fd[0], fd_sign);
            println!("\n    => If cost trend is POSITIVE, then ANALYTIC is WRONG");
            println!("    => If cost trend is NEGATIVE, then FD is WRONG");
        }
        Err(e) => println!("    ERROR: {:?}", e),
    }
}

/// Hypothesis 20: Full GAM combination (control - expected to fail).
#[test]
fn hypothesis_full_gam_control() {
    println!("\n=== Hypothesis 20: Full GAM (Control - Expected Fail) ===");

    let train = create_logistic_training_data(100, 3, 31);
    let config = logistic_model_config(true, false, &train);
    let (x_gam, s_list_gam, layout, ..) =
        build_design_and_penalty_matrices(&train, &config).expect("design");

    let nullspace_dims = vec![0; s_list_gam.len()];
    let offset = Array1::<f64>::zeros(train.y.len());
    let rho = Array1::from_elem(layout.num_penalties, 12.0);

    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
        firth: Some(FirthSpec { enabled: true }),
        tol: 1e-10,
        max_iter: 200,
        nullspace_dims,
    };
    
    // Test: Full GAM configuration
    let (analytic, fd) = evaluate_external_gradients(
        train.y.view(), train.weights.view(), x_gam.view(), offset.view(),
        &s_list_gam, &opts, &rho,
    ).expect("gradients");
    
    let (cos, rel, max_a, _) = gradient_metrics(&analytic, &fd);
    let status = if cos > 0.99 && rel < 0.05 { "✓ PASS" } else { "✗ FAIL" };
    println!("  Full GAM: cos={:.4}, rel={:.2e}, |grad|={:.2e} {}", cos, rel, max_a, status);
}
