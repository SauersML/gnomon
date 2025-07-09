// Test to verify the CRAZY IDEA about REML gradient sign error
// Hypothesis: beta_term and trace_term should be ADDED, not SUBTRACTED

use ndarray::Array1;
use ndarray::array;

// We'll need to import the necessary functions from estimate.rs
// This is a standalone test to isolate the sign issue

#[test]
fn test_reml_gradient_plus_vs_minus_sign() {
    // GOAL: Test if gradient = 0.5 * λ * (beta_term/σ² + trace_term) 
    // is more accurate than gradient = 0.5 * λ * (beta_term/σ² - trace_term)
    
    println!("=== Testing REML Gradient Sign: PLUS vs MINUS ===");
    println!("Current:     gradient = 0.5 * λ * (beta_term/σ² - trace_term)");
    println!("Hypothesis:  gradient = 0.5 * λ * (beta_term/σ² + trace_term)");
    println!();
    
    // This test will be implemented to:
    // 1. Calculate numerical gradient using finite differences
    // 2. Calculate analytical gradient with MINUS sign (current)
    // 3. Calculate analytical gradient with PLUS sign (hypothesis)
    // 4. Compare errors to see which is more accurate
    
    println!("Test structure:");
    println!("1. Setup simple REML problem");
    println!("2. Calculate numerical gradient (ground truth)");
    println!("3. Calculate gradient with MINUS sign (current implementation)");
    println!("4. Calculate gradient with PLUS sign (hypothesis)");
    println!("5. Compare absolute errors");
    println!();
    println!("If hypothesis is correct, PLUS sign should have smaller error.");
}

#[test] 
fn test_cost_function_return_negation() {
    // GOAL: Check if compute_gradient returns -gradient or +gradient
    // The IDEA mentions that the function returns Ok(-gradient)
    
    println!("=== Testing if compute_gradient returns negated values ===");
    println!("Need to check the actual return statement in compute_gradient");
    println!("If it returns Ok(-gradient), that would explain the sign error");
}

#[test]
fn test_cost_function_direction() {
    // GOAL: Test the intuitive relationship between penalty and cost
    // If we increase penalty λ, the cost should increase (worse fit)
    // The gradient should be positive (pointing toward higher cost)
    
    println!("=== Testing Cost vs Penalty Relationship ===");
    println!("Intuition: Higher penalty λ → Higher cost → Positive gradient");
    println!("We'll test this relationship at multiple penalty levels");
}
