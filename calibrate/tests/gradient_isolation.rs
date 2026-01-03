//! Gradient isolation tests to identify what triggers analytic vs FD mismatch.
//!
//! Each test toggles ONE factor while keeping others constant.
//! This identifies the root cause of gradient failures in complex models.

use ndarray::{array, Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::calibrate::estimate::{evaluate_external_gradients, ExternalOptimOptions};
use crate::calibrate::model::LinkFunction;
use crate::calibrate::calibrator::FirthSpec;

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
    
    // True coefficients (small to stay well-conditioned)
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

/// Run gradient check and return (cosine_similarity, relative_l2_error, max_analytic, max_fd).
fn check_gradient(
    y: &Array1<f64>,
    x: &Array2<f64>,
    weights: &Array1<f64>,
    s_list: &[Array2<f64>],
    rho: &Array1<f64>,
    firth: bool,
    nullspace_dims: Vec<usize>,
) -> Result<(f64, f64, f64, f64), String> {
    let n = x.nrows();
    let offset = Array1::<f64>::zeros(n);
    
    let opts = ExternalOptimOptions {
        link: LinkFunction::Logit,
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
            let dot = analytic.dot(&fd);
            let n_a = analytic.dot(&analytic).sqrt();
            let n_f = fd.dot(&fd).sqrt();
            let cosine = if n_a * n_f > 1e-12 { dot / (n_a * n_f) } else { 1.0 };
            
            let diff = &analytic - &fd;
            let rel_l2 = diff.dot(&diff).sqrt() / n_f.max(n_a).max(1e-12);
            
            let max_a = analytic.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
            let max_f = fd.iter().fold(0.0f64, |acc, &v| acc.max(v.abs()));
            
            Ok((cosine, rel_l2, max_a, max_f))
        }
        Err(e) => Err(format!("{:?}", e)),
    }
}

/// Test 1: Single penalty, no Firth — baseline that should pass.
#[test]
fn isolation_baseline_single_penalty_no_firth() {
    println!("\n=== ISOLATION: Single penalty, No Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 8, 42);
    let p = x.ncols();
    
    // Single penalty on non-intercept coefficients
    let s = diagonal_penalty(p, 1, p);
    let s_list = vec![s];
    let rho = array![0.0]; // λ = 1
    
    match check_gradient(&y, &x, &w, &s_list, &rho, false, vec![1]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            assert!(cos > 0.99, "Baseline should pass: cosine={}", cos);
            println!("  ✓ PASS");
        }
        Err(e) => panic!("Baseline failed: {}", e),
    }
}

/// Test 2: Single penalty, WITH Firth — isolates Firth alone.
#[test]
fn isolation_single_penalty_with_firth() {
    println!("\n=== ISOLATION: Single penalty, WITH Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 8, 42);
    let p = x.ncols();
    
    let s = diagonal_penalty(p, 1, p);
    let s_list = vec![s];
    let rho = array![0.0];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            if cos > 0.99 {
                println!("  ✓ PASS — Firth alone is fine");
            } else {
                println!("  ✗ FAIL — Firth alone triggers mismatch");
            }
        }
        Err(e) => println!("  ✗ ERROR: {}", e),
    }
}

/// Test 3: Multiple penalties (isotropic λ), no Firth.
#[test]
fn isolation_multi_penalty_isotropic_no_firth() {
    println!("\n=== ISOLATION: Multi-penalty (isotropic λ), No Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    
    // 3 overlapping penalties
    let s1 = diagonal_penalty(p, 1, 5);
    let s2 = diagonal_penalty(p, 4, 9);
    let s3 = diagonal_penalty(p, 8, 12);
    let s_list = vec![s1, s2, s3];
    let rho = array![0.0, 0.0, 0.0]; // all λ = 1 (isotropic)
    
    match check_gradient(&y, &x, &w, &s_list, &rho, false, vec![1, 0, 0]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            if cos > 0.99 {
                println!("  ✓ PASS — Multi-penalty isotropic is fine");
            } else {
                println!("  ✗ FAIL — Multi-penalty alone triggers mismatch");
            }
        }
        Err(e) => println!("  ✗ ERROR: {}", e),
    }
}

/// Test 4: Multiple penalties (ANISOTROPIC λ), no Firth.
#[test]
fn isolation_multi_penalty_anisotropic_no_firth() {
    println!("\n=== ISOLATION: Multi-penalty (ANISOTROPIC λ), No Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    
    let s1 = diagonal_penalty(p, 1, 5);
    let s2 = diagonal_penalty(p, 4, 9);
    let s3 = diagonal_penalty(p, 8, 12);
    let s_list = vec![s1, s2, s3];
    
    // Anisotropic: λ = [1, 100, 0.01]
    let rho = array![0.0, 4.6, -4.6];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, false, vec![1, 0, 0]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            if cos > 0.99 {
                println!("  ✓ PASS — Anisotropic λ is fine");
            } else {
                println!("  ✗ FAIL — Anisotropic λ triggers mismatch");
            }
        }
        Err(e) => println!("  ✗ ERROR: {}", e),
    }
}

/// Test 5: Multiple penalties + Firth (isotropic λ).
#[test]
fn isolation_multi_penalty_isotropic_with_firth() {
    println!("\n=== ISOLATION: Multi-penalty (isotropic) + Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    
    let s1 = diagonal_penalty(p, 1, 5);
    let s2 = diagonal_penalty(p, 4, 9);
    let s3 = diagonal_penalty(p, 8, 12);
    let s_list = vec![s1, s2, s3];
    let rho = array![0.0, 0.0, 0.0];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1, 0, 0]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            if cos > 0.99 {
                println!("  ✓ PASS — Multi + Firth isotropic is fine");
            } else {
                println!("  ✗ FAIL — Multi + Firth isotropic triggers mismatch");
            }
        }
        Err(e) => println!("  ✗ ERROR: {}", e),
    }
}

/// Test 6: Multiple penalties + Firth (ANISOTROPIC λ) — full complexity.
#[test]
fn isolation_multi_penalty_anisotropic_with_firth() {
    println!("\n=== ISOLATION: Multi-penalty (ANISOTROPIC) + Firth ===");
    
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    
    let s1 = diagonal_penalty(p, 1, 5);
    let s2 = diagonal_penalty(p, 4, 9);
    let s3 = diagonal_penalty(p, 8, 12);
    let s_list = vec![s1, s2, s3];
    let rho = array![0.0, 4.6, -4.6];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1, 0, 0]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            if cos > 0.99 {
                println!("  ✓ PASS");
            } else {
                println!("  ✗ FAIL — Full complexity triggers mismatch");
            }
        }
        Err(e) => println!("  ✗ ERROR: {}", e),
    }
}

/// Test 7: High dimensionality (like failing test: p=68).
#[test]
fn isolation_high_dimension_with_firth() {
    println!("\n=== ISOLATION: High dimension (p=40) + Firth ===");
    
    let (y, x, w) = generate_logit_data(150, 40, 42);
    let p = x.ncols();
    
    // 4 penalties covering different ranges
    let s1 = diagonal_penalty(p, 1, 10);
    let s2 = diagonal_penalty(p, 10, 20);
    let s3 = diagonal_penalty(p, 20, 30);
    let s4 = diagonal_penalty(p, 30, 40);
    let s_list = vec![s1, s2, s3, s4];
    let rho = array![0.0, 2.0, -2.0, 1.0]; // Mixed λ
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1, 0, 0, 0]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            if cos > 0.99 {
                println!("  ✓ PASS");
            } else {
                println!("  ✗ FAIL — High dimension triggers mismatch");
            }
        }
        Err(e) => println!("  ✗ ERROR: {}", e),
    }
}

/// Test 8: Disjoint null spaces (the critic's hypothesis).
#[test]
fn isolation_disjoint_null_spaces() {
    println!("\n=== ISOLATION: Disjoint null spaces ===");
    println!("  S1 penalizes [1,2,3], null space = [0,4,5]");
    println!("  S2 penalizes [4,5], null space = [0,1,2,3]");
    println!("  Combined null space = [0] only");
    
    let (y, x, w) = generate_logit_data(80, 6, 42);
    
    // S1: penalize cols 1,2,3
    let mut s1 = Array2::<f64>::zeros((6, 6));
    for j in 1..4 {
        s1[[j, j]] = 1.0;
    }
    
    // S2: penalize cols 4,5
    let mut s2 = Array2::<f64>::zeros((6, 6));
    for j in 4..6 {
        s2[[j, j]] = 1.0;
    }
    
    let s_list = vec![s1, s2];
    let rho = array![2.0, -2.0]; // Anisotropic: λ1=7.4, λ2=0.14
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1, 0]) {
        Ok((cos, rel, max_a, max_f)) => {
            println!("  cosine={:.6}, rel_l2={:.3e}, max|a|={:.3e}, max|fd|={:.3e}", cos, rel, max_a, max_f);
            if cos > 0.99 {
                println!("  ✓ PASS — Disjoint null spaces OK");
            } else {
                println!("  ✗ FAIL — Disjoint null spaces trigger mismatch");
                println!("  → This confirms the truncation/det1 hypothesis!");
            }
        }
        Err(e) => println!("  ✗ ERROR: {}", e),
    }
}
