// Simple test to verify the Cox-de Boor fix works correctly
use ndarray::array;

// Copy just the corrected function for testing
mod internal {
    use ndarray::{Array1, ArrayView1};

    pub(super) fn evaluate_splines_at_point(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
    ) -> Array1<f64> {
        let num_knots = knots.len();
        let num_basis = num_knots - degree - 1;

        // Find the knot interval `mu` that contains x, such that `knots[mu] <= x < knots[mu+1]`.
        let mu = match knots.iter().rposition(|&k| k <= x) {
            Some(pos) => {
                // Clamp to valid range
                pos.min(num_basis + degree - 1).max(degree)
            }
            None => degree, // Default to the first valid interval
        };

        // Initialize basis values for degree 0
        let mut b = Array1::zeros(degree + 1);
        b[0] = 1.0;

        // Apply the Cox-de Boor recurrence relation iteratively
        for d in 1..=degree {
            // Use a temporary buffer to read the "old" values from degree d-1
            let b_old = b.clone();
            // Reset the current buffer for the new degree's values
            b.fill(0.0);

            for i in 0..=d {
                // Index for knots array
                let idx = mu - d + i;

                // Left parent spline contribution
                if i < d && b_old[i] > 0.0 {
                    let denom = knots[idx + d] - knots[idx];
                    if denom > 1e-12 {
                        let w = (x - knots[idx]) / denom;
                        b[i] += w * b_old[i];
                    }
                }

                // Right parent spline contribution - FIXED DENOMINATOR
                if i > 0 && b_old[i - 1] > 0.0 {
                    let denom = knots[idx + d] - knots[idx - 1]; // <-- FIXED: use correct denominator
                    if denom > 1e-12 {
                        let w = (knots[idx + d] - x) / denom;
                        b[i] += w * b_old[i - 1];
                    }
                }
            }
        }

        // Place the non-zero values in the correct positions
        let mut basis_values = Array1::zeros(num_basis);
        let start_index = mu.saturating_sub(degree);
        
        for i in 0..=degree {
            let global_idx = start_index + i;
            if global_idx < num_basis {
                basis_values[global_idx] = b[i];
            }
        }
       
        basis_values
    }
}

fn main() {
    // Test the corrected implementation
    let knots = array![0.0, 0.0, 1.0, 2.0, 2.0];
    let x = 0.5;
    
    let values = internal::evaluate_splines_at_point(x, 1, knots.view());
    
    println!("Basis function values at x={}: {:?}", x, values);
    println!("Sum: {}", values.sum());
    
    // The values should sum to 1.0 and be [0.5, 0.5, 0.0] for this case
    assert!((values[0] - 0.5).abs() < 1e-9, "Expected B_0,1 to be 0.5, got {}", values[0]);
    assert!((values[1] - 0.5).abs() < 1e-9, "Expected B_1,1 to be 0.5, got {}", values[1]);
    assert!((values[2] - 0.0).abs() < 1e-9, "Expected B_2,1 to be 0.0, got {}", values[2]);
    assert!((values.sum() - 1.0).abs() < 1e-9, "Sum should be 1.0, got {}", values.sum());
    
    println!("✓ All tests passed!");
}
