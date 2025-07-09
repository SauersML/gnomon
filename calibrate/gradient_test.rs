use ndarray::Array1;

// Create a simple test to definitively determine which gradient formula is correct
fn test_gradient_formulations() {
    // Use the same setup as the failing test but test both formulations
    
    // From test output:
    let lambda = 0.367879;
    let beta_term_over_sigma_sq = 0.006785; // β̂ᵀS_kβ̂/σ²
    let trace_term = 2.537787; // tr(H⁻¹S_k)
    let numerical_gradient = 0.434215; // Ground truth from finite differences
    
    // Test formulation 1: (β̂ᵀS_kβ̂/σ² - tr(H⁻¹S_k)) - what the failing test uses
    let grad_1 = 0.5 * lambda * (beta_term_over_sigma_sq - trace_term);
    
    // Test formulation 2: (tr(H⁻¹S_k) - β̂ᵀS_kβ̂/σ²) - what my code uses  
    let grad_2 = 0.5 * lambda * (trace_term - beta_term_over_sigma_sq);
    
    println!("Numerical gradient (ground truth): {:.6}", numerical_gradient);
    println!("Formulation 1 (beta - trace): {:.6}", grad_1);
    println!("Formulation 2 (trace - beta): {:.6}", grad_2);
    
    println!("Error for formulation 1: {:.6}", (grad_1 - numerical_gradient).abs());
    println!("Error for formulation 2: {:.6}", (grad_2 - numerical_gradient).abs());
    
    // The correct formulation should have much smaller error
    if (grad_1 - numerical_gradient).abs() < (grad_2 - numerical_gradient).abs() {
        println!("Formulation 1 is correct: (β̂ᵀS_kβ̂/σ² - tr(H⁻¹S_k))");
    } else {
        println!("Formulation 2 is correct: (tr(H⁻¹S_k) - β̂ᵀS_kβ̂/σ²)");
    }
}

fn main() {
    test_gradient_formulations();
}
