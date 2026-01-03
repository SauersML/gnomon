//! Gradient isolation tests to identify what triggers analytic vs FD mismatch.
//!
//! Each test toggles ONE factor while keeping others constant.
//! This identifies the root cause of gradient failures in complex models.
//!
//! DIAGNOSTIC STRATEGY:
//! 
//! 1. Factor Isolation: Toggle single factors (Firth, multi-penalty, anisotropic λ)
//! 2. Frozen Beta Test: Disable re-optimization to test Envelope Theorem
//! 3. Component Breakout: Test each LAML term separately
//! 4. Ridge Scaling: Vary ridge to detect null-space amplification
//! 5. Identity Link: Control for Firth/logit-specific issues

use ndarray::{array, Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use gnomon::calibrate::estimate::{evaluate_external_gradients, ExternalOptimOptions};
use gnomon::calibrate::model::LinkFunction;
use gnomon::calibrate::calibrator::FirthSpec;

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

// ============================================================================
// SECTION 1: FACTOR ISOLATION TESTS
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

/// Test 7: Disjoint null spaces (critic's hypothesis).
#[test]
fn isolation_disjoint_null_spaces() {
    println!("\n=== ISOLATION: Disjoint null spaces ===");
    println!("  S1 penalizes [1,2,3], null=[0,4,5]");
    println!("  S2 penalizes [4,5], null=[0,1,2,3]");
    
    let (y, x, w) = generate_logit_data(80, 6, 42);
    
    let mut s1 = Array2::<f64>::zeros((6, 6));
    for j in 1..4 { s1[[j, j]] = 1.0; }
    
    let mut s2 = Array2::<f64>::zeros((6, 6));
    for j in 4..6 { s2[[j, j]] = 1.0; }
    
    let rho = array![2.0, -2.0];
    
    match check_gradient(&y, &x, &w, &[s1, s2], &rho, true, vec![1, 0], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => {
            print_result("Disjoint null", cos, rel, max_a, max_f);
            if cos < 0.99 {
                println!("  → Confirms truncation/det1 hypothesis!");
            }
        }
        Err(e) => println!("  ERROR: {}", e),
    }
}

// ============================================================================
// SECTION 2: IDENTITY LINK CONTROL (Critic's #6)
// ============================================================================

/// If Identity works but Logit fails → issue is Firth/W-derivative specific.
/// If Identity also fails → issue is in linear algebra (penalties, traces).
#[test]
fn isolation_identity_link_control() {
    println!("\n=== IDENTITY LINK CONTROL ===");
    println!("  Same penalties as logit test, but with Gaussian data + Identity link");
    
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
            if cos > 0.99 {
                println!("  → Identity PASSES: issue is Firth/logit-specific");
            } else {
                println!("  → Identity FAILS: issue is in linear algebra");
            }
        }
        Err(e) => println!("  ERROR: {}", e),
    }
}

// ============================================================================
// SECTION 3: RIDGE SCALING SWEEP (Critic's #2)
// ============================================================================

/// If error ∝ 1/ridge → confirms null-space amplification (Ghost Penalty).
/// If error is constant → structural logic error (sign, transpose, etc).
///
/// NOTE: This test conceptually documents the approach. The actual fixed ridge
/// is determined by FIXED_STABILIZATION_RIDGE in pirls.rs. To truly test this,
/// you'd need to modify that constant and rebuild.
#[test]
fn isolation_ridge_scaling_conceptual() {
    println!("\n=== RIDGE SCALING SWEEP (conceptual) ===");
    println!("  Varying ridge to detect null-space amplification");
    println!("  If error ∝ 1/ridge → Ghost Penalty confirmed");
    println!("  If error constant → structural logic error");
    println!();
    println!("  To test: modify FIXED_STABILIZATION_RIDGE in pirls.rs to:");
    println!("    1e-8, 1e-6, 1e-4");
    println!("  and compare gradient errors.");
    println!();
    
    // Run with current ridge as baseline measurement
    let (y, x, w) = generate_logit_data(100, 12, 42);
    let p = x.ncols();
    let s_list = vec![
        diagonal_penalty(p, 1, 5),
        diagonal_penalty(p, 4, 9),
        diagonal_penalty(p, 8, 12),
    ];
    let rho = array![0.0, 4.6, -4.6];
    
    match check_gradient(&y, &x, &w, &s_list, &rho, true, vec![1, 0, 0], LinkFunction::Logit) {
        Ok((cos, rel, max_a, max_f)) => {
            print_result("Current ridge", cos, rel, max_a, max_f);
            println!("  Record this. Repeat with different FIXED_STABILIZATION_RIDGE values.");
        }
        Err(e) => println!("  ERROR: {}", e),
    }
}

// ============================================================================
// SECTION 4: COMPONENT BREAKOUT (Critic's #1)
// ============================================================================

/// The LAML gradient is: ∂V/∂ρ = ∂D_p/∂ρ + 0.5·∂log|H|/∂ρ - 0.5·∂log|S|+/∂ρ
/// 
/// This test conceptually documents what to check:
/// - If D_p fails → Envelope Theorem violated (PIRLS didn't converge)
/// - If log|S| fails → truncation logic mismatch (Ghost Penalty)
/// - If log|H| fails → Firth Hessian mismatch
///
/// NOTE: Implementing this requires instrumenting compute_gradient to return
/// individual terms. This is a TODO for full diagnosis.
#[test]
fn isolation_component_breakout_conceptual() {
    println!("\n=== COMPONENT BREAKOUT (conceptual) ===");
    println!("  LAML gradient = ∂D_p/∂ρ + 0.5·∂log|H|/∂ρ - 0.5·∂log|S|/∂ρ");
    println!();
    println!("  To diagnose: instrument compute_gradient() to return each term.");
    println!("  Compare analytic vs FD for EACH term separately.");
    println!();
    println!("  Interpretation:");
    println!("    D_p fails    → PIRLS didn't converge (Envelope Theorem)");
    println!("    log|S| fails → truncation mismatch (Ghost Penalty)");
    println!("    log|H| fails → Firth Hessian wrong");
}

// ============================================================================
// SECTION 5: FROZEN BETA TEST (Critic's #3) - THE KILLER TEST
// ============================================================================

/// Frozen Beta Test: Perturb ρ but do NOT re-optimize β.
/// 
/// Standard FD: f(ρ+h) - f(ρ-h) where β is re-optimized at each ρ.
/// Frozen FD:   f(ρ+h, β_fixed) - f(ρ-h, β_fixed) with the SAME β.
///
/// If Analytic ≈ Frozen_FD → direct derivatives are correct, but we're 
///                           missing the implicit dβ/dρ term.
/// If Analytic ≠ Frozen_FD → the ∂/∂ρ formulas themselves are wrong.
///
/// NOTE: This requires modifying evaluate_external_gradients to accept a
/// frozen_beta parameter. This is a TODO for full diagnosis.
#[test]
fn isolation_frozen_beta_conceptual() {
    println!("\n=== FROZEN BETA TEST (conceptual) ===");
    println!("  Standard FD: re-optimize β at ρ±h");
    println!("  Frozen FD:   use SAME β at ρ±h");
    println!();
    println!("  If Analytic ≈ Frozen_FD:");
    println!("    → Direct derivatives correct, missing implicit dβ/dρ");
    println!("  If Analytic ≠ Frozen_FD:");
    println!("    → The ∂/∂ρ formulas are wrong");
    println!();
    println!("  To implement: modify evaluate_external_gradients to accept");
    println!("  an optional frozen_beta parameter for FD computation.");
}

// ============================================================================
// SECTION 6: CLEAN MATRIX INJECTION (Critic's #5)
// ============================================================================

/// Clean Matrix Injection: Project S_k onto the active subspace before
/// computing trace(H^{-1} S_k).
///
/// If S_clean fixes the mismatch → spectral bleed is the cause.
/// The cost function ignores null-space energy, but the trace captures it.
///
/// NOTE: This requires modifying the gradient code to accept projected
/// penalty matrices. This is a TODO for full diagnosis.
#[test]
fn isolation_clean_matrix_conceptual() {
    println!("\n=== CLEAN MATRIX INJECTION (conceptual) ===");
    println!("  Project S_k onto active subspace before computing trace");
    println!("  S_clean = P_range · S_k · P_range");
    println!();
    println!("  If S_clean fixes mismatch:");
    println!("    → Spectral bleed is cause (null-space energy captured by trace)");
    println!();
    println!("  To implement: modify gradient to project S_k before trace.");
}

// ============================================================================
// SUMMARY TEST: Run all isolation tests and print diagnosis
// ============================================================================

#[test]
fn isolation_summary() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║            GRADIENT ISOLATION DIAGNOSTIC SUMMARY             ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Run: cargo test isolation_ -- --nocapture --test-threads=1   ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Pattern Analysis:                                            ║");
    println!("║   • All pass except Firth → Firth derivative bug             ║");
    println!("║   • All pass except anisotropic → truncation bug             ║");
    println!("║   • All pass except disjoint null → Ghost Penalty confirmed  ║");
    println!("║   • Identity passes, Logit fails → Firth/W-derivative bug    ║");
    println!("║   • Everything fails → fundamental linear algebra bug        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
}
