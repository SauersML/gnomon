// test_constraint_debug.rs
use ndarray::Array;
use calibrate::basis::{create_bspline_basis, apply_sum_to_zero_constraint};

fn main() {
    // Create simple test data
    let data = Array::linspace(0.0, 1.0, 10);
    println!("Input data: {:?}", data);
    
    // Create basis without constraint
    let (basis_unc, _) = create_bspline_basis(
        data.view(),
        Some(data.view()), 
        (0.0, 1.0),
        2, // num_knots
        3  // degree
    ).unwrap();
    
    println!("Unconstrained basis shape: {:?}", basis_unc.dim());
    println!("Unconstrained basis:\n{:?}", basis_unc);
    
    // Apply sum-to-zero constraint
    let (basis_c, z_transform) = apply_sum_to_zero_constraint(basis_unc.view()).unwrap();
    
    println!("Constrained basis shape: {:?}", basis_c.dim());
    println!("Constrained basis:\n{:?}", basis_c);
    println!("Z transform shape: {:?}", z_transform.dim());
    println!("Z transform:\n{:?}", z_transform);
}
