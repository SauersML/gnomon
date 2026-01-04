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
use gnomon::calibrate::estimate::{evaluate_external_gradients, ExternalOptimOptions};
use gnomon::calibrate::faer_ndarray::FaerCholesky;
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
