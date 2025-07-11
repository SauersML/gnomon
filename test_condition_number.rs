// Simple test to verify the condition number calculation works
use ndarray::{Array2, array};
use ndarray_linalg::SVD;

fn calculate_condition_number(matrix: &Array2<f64>) -> Result<f64, ndarray_linalg::error::LinalgError> {
    // Compute SVD
    let (_u, s, _vt) = matrix.svd(false, false)?;
    
    // Get max and min singular values
    let max_sv = s.iter().fold(0.0_f64, |max, &val| max.max(val));
    let min_sv = s.iter().fold(f64::INFINITY, |min, &val| min.min(val));
    
    // Check for effective singularity
    if min_sv < 1e-12 {
        return Ok(f64::INFINITY);
    }
    
    Ok(max_sv / min_sv)
}

fn main() {
    println!("Testing condition number calculation...");
    
    // Test 1: Well-conditioned matrix (identity)
    let identity = Array2::<f64>::eye(3);
    match calculate_condition_number(&identity) {
        Ok(cond) => println!("Identity matrix condition number: {:.2e} (should be 1.0)", cond),
        Err(e) => println!("Error: {:?}", e),
    }
    
    // Test 2: Ill-conditioned matrix
    let ill_cond = array![
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0001],
        [1.0, 1.0001, 1.0]
    ];
    match calculate_condition_number(&ill_cond) {
        Ok(cond) => println!("Ill-conditioned matrix condition number: {:.2e} (should be large)", cond),
        Err(e) => println!("Error: {:?}", e),
    }
    
    // Test 3: Singular matrix
    let singular = array![
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ];
    match calculate_condition_number(&singular) {
        Ok(cond) => println!("Singular matrix condition number: {:.2e} (should be infinity)", cond),
        Err(e) => println!("Error: {:?}", e),
    }
    
    println!("\nAll tests completed!");
}
