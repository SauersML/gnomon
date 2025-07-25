use ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use approx::assert_abs_diff_eq;

/// Defines the strategy for placing the internal knots of a spline.
/// This is part of the public API and will be saved in the model configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnotStrategy {
    /// Place knots uniformly across the specified `data_range`.
    /// This is deterministic and suitable for prediction.
    Uniform,
    /// Place knots at the quantiles of the provided `training_data_for_quantiles`.
    /// This adapts to the data's distribution and is recommended for model training.
    Quantile,
}

/// A comprehensive error type for all operations within the basis module.
#[derive(Error, Debug)]
pub enum BasisError {
    #[error("Spline degree must be at least 1, but was {0}.")]
    InvalidDegree(usize),

    #[error("Data range is invalid: start ({0}) must be less than or equal to end ({1}).")]
    InvalidRange(f64, f64),

    #[error("Quantile strategy requires a non-empty training data set for quantile calculation.")]
    QuantileDataMissing,

    #[error("Cannot compute {num_quantiles} quantiles from only {num_points} data points.")]
    InsufficientDataForQuantiles {
        num_quantiles: usize,
        num_points: usize,
    },

    #[error(
        "Penalty order ({order}) must be positive and less than the number of basis functions ({num_basis})."
    )]
    InvalidPenaltyOrder { order: usize, num_basis: usize },

    #[error(
        "Insufficient knots for degree {degree} spline: need at least {required} knots but only {provided} were provided."
    )]
    InsufficientKnotsForDegree {
        degree: usize,
        required: usize,
        provided: usize,
    },

    #[error("QR decomposition failed while applying constraints: {0}")]
    LinalgError(#[from] ndarray_linalg::error::LinalgError),
}

/// Creates a B-spline basis expansion matrix and its corresponding knot vector.
///
/// This is the primary workhorse function, implementing a numerically stable
/// version of the Cox-de Boor algorithm for evaluation.
///
/// # Arguments
///
/// * `data`: A 1D view of the data points to be transformed (e.g., a single vector
///   of PGS values or a single Principal Component).
/// * `training_data_for_quantiles`: An `Option` containing a view of the *original,
///   full training dataset column*. This is **required** and must be `Some` when
///   `knot_strategy` is `Quantile`. It is ignored otherwise.
/// * `data_range`: A tuple `(min, max)` defining the boundaries for knot placement.
///   **Crucially, this must always be the range of the original training data**,
///   even when making predictions on new data, to ensure a consistent basis.
/// * `num_internal_knots`: The number of knots to place *between* the boundaries.
/// * `degree`: The degree of the B-spline polynomials (e.g., 3 for cubic).
///
/// # Returns
///
/// On success, returns a `Result` containing a tuple `(Array2<f64>, Array1<f64>)`:
/// 1.  The **basis matrix**, with shape `[data.len(), num_basis_functions]`.
///     The number of basis functions is `num_internal_knots + degree + 1`.
/// 2.  The **full knot vector** used to generate the basis. This is needed for the penalty matrix.
/// Creates a B-spline basis matrix using a pre-computed knot vector.
/// This function is used during prediction to ensure exact reproduction of training basis.
pub fn create_bspline_basis_with_knots(
    data: ArrayView1<f64>,
    knot_vector: ArrayView1<f64>,
    degree: usize,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }

    // Check that we have enough knots for the requested degree
    let required_knots = degree + 2;
    if knot_vector.len() < required_knots {
        return Err(BasisError::InsufficientKnotsForDegree {
            degree,
            required: required_knots,
            provided: knot_vector.len(),
        });
    }

    let num_basis_functions = knot_vector.len() - degree - 1;
    let mut basis_matrix = Array2::zeros((data.len(), num_basis_functions));

    // Evaluate the splines for each data point
    for (i, &x) in data.iter().enumerate() {
        let basis_row = internal::evaluate_splines_at_point(x, degree, knot_vector);
        basis_matrix.row_mut(i).assign(&basis_row);
    }

    Ok((basis_matrix, knot_vector.to_owned()))
}

pub fn create_bspline_basis(
    data: ArrayView1<f64>,
    training_data_for_quantiles: Option<ArrayView1<f64>>,
    data_range: (f64, f64),
    num_internal_knots: usize,
    degree: usize,
) -> Result<(Array2<f64>, Array1<f64>), BasisError> {
    if degree < 1 {
        return Err(BasisError::InvalidDegree(degree));
    }
    if data_range.0 > data_range.1 {
        return Err(BasisError::InvalidRange(data_range.0, data_range.1));
    }

    let knot_vector = internal::generate_full_knot_vector(
        data_range,
        num_internal_knots,
        degree,
        training_data_for_quantiles,
    )?;

    // The number of B-spline basis functions for a given knot vector and degree `d` is
    // n = k - d - 1, where k is the number of knots.
    // Our knot vector has k = num_internal_knots + 2 * (degree + 1) knots.
    // So, n = (num_internal_knots + 2*d + 2) - d - 1 = num_internal_knots + d + 1.
    let num_basis_functions = knot_vector.len() - degree - 1;

    let mut basis_matrix = Array2::zeros((data.len(), num_basis_functions));

    // Evaluate the splines for each data point.
    // This structure allows the inner loop (de Boor's) to be highly optimized
    // and cache-friendly for a single point `x`.
    for (i, &x) in data.iter().enumerate() {
        let basis_row = internal::evaluate_splines_at_point(x, degree, knot_vector.view());
        basis_matrix.row_mut(i).assign(&basis_row);
    }

    Ok((basis_matrix, knot_vector))
}

/// Creates a penalty matrix `S` for a B-spline basis from a difference matrix `D`.
/// The penalty is of the form `S = D' * D`, penalizing the squared `order`-th
/// differences of the spline coefficients. This is the core of P-splines.
///
/// # Arguments
/// * `num_basis_functions`: The number of basis functions (i.e., columns in the basis matrix).
/// * `order`: The order of the difference penalty (e.g., 2 for second differences).
///
/// # Returns
/// A square `Array2<f64>` of shape `[num_basis, num_basis]` representing the penalty `S`.
pub fn create_difference_penalty_matrix(
    num_basis_functions: usize,
    order: usize,
) -> Result<Array2<f64>, BasisError> {
    if order == 0 || order >= num_basis_functions {
        return Err(BasisError::InvalidPenaltyOrder {
            order,
            num_basis: num_basis_functions,
        });
    }

    // Start with the identity matrix
    let mut d = Array2::<f64>::eye(num_basis_functions);

    // Apply the differencing operation `order` times.
    // Each `diff` reduces the number of rows by 1.
    for _ in 0..order {
        // This calculates the difference between adjacent rows.
        d = &d.slice(s![1.., ..]) - &d.slice(s![..-1, ..]);
    }

    // The penalty matrix S = D' * D
    let s = d.t().dot(&d);
    Ok(s)
}

/// Applies a sum-to-zero constraint to a basis matrix for model identifiability.
///
/// This is achieved by reparameterizing the basis to be orthogonal to the intercept.
/// In GAMs, this constraint removes the confounding between the intercept and smooth functions.
///
/// # Arguments
/// * `basis_matrix`: An `ArrayView2<f64>` of the original, unconstrained basis matrix.
///
/// # Returns
/// A tuple containing:
/// 1. The new, constrained basis matrix (with one fewer column).
/// 2. The transformation matrix `Z` used to create it.
pub fn apply_sum_to_zero_constraint(
    basis_matrix: ArrayView2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let k = basis_matrix.ncols();
    if k < 2 {
        return Err(BasisError::InvalidDegree(k)); // cannot constrain a single column
    }

    // --- build a full-rank null-space basis for 1ᵀ --------------------------
    // Z has k rows and k-1 columns; each column sums to zero and the k-1 cols
    // are linearly independent, so B·Z drops exactly one dof (the mean).
    let mut z = Array2::<f64>::zeros((k, k - 1));
    for j in 0..k - 1 {
        z[[j, j]] = 1.0; // identity block
        z[[k - 1, j]] = -1.0; // last row makes column sum zero
    }

    // project the original basis
    let constrained = basis_matrix.dot(&z);
    Ok((constrained, z))
}

/// Internal module for implementation details not exposed in the public API.
mod internal {
    use super::*;

    /// Generates the full knot vector, including repeated boundary knots.
    pub(super) fn generate_full_knot_vector(
        data_range: (f64, f64),
        num_internal_knots: usize,
        degree: usize,
        training_data_for_quantiles: Option<ArrayView1<f64>>,
    ) -> Result<Array1<f64>, BasisError> {
        let (min_val, max_val) = data_range;

        let internal_knots = if let Some(training_data) = training_data_for_quantiles {
            // Quantile-based knots
            if training_data.is_empty() {
                return Err(BasisError::QuantileDataMissing);
            }
            if training_data.len() < num_internal_knots {
                return Err(BasisError::InsufficientDataForQuantiles {
                    num_quantiles: num_internal_knots,
                    num_points: training_data.len(),
                });
            }
            quantiles(training_data, num_internal_knots)?
        } else {
            // Uniformly spaced knots
            if num_internal_knots == 0 {
                Array1::from_vec(vec![])
            } else {
                let h = (max_val - min_val) / (num_internal_knots as f64 + 1.0);
                Array::from_iter((1..=num_internal_knots).map(|i| min_val + i as f64 * h))
            }
        };

        // B-splines require `degree + 1` repeated knots at each boundary.
        let min_knots = Array1::from_elem(degree + 1, min_val);
        let max_knots = Array1::from_elem(degree + 1, max_val);

        // Concatenate [boundary_min, internal, boundary_max] to form the full knot vector.
        Ok(ndarray::concatenate(
            Axis(0),
            &[min_knots.view(), internal_knots.view(), max_knots.view()],
        )
        .expect("Knot vector concatenation should never fail with correct inputs"))
    }

    /// Calculates quantiles from a data vector using linear interpolation (Type 7 in R).
    fn quantiles(data: ArrayView1<f64>, num_quantiles: usize) -> Result<Array1<f64>, BasisError> {
        if num_quantiles == 0 {
            return Ok(Array1::from_vec(vec![]));
        }

        let mut sorted_data = data.to_vec();
        // Use `sort_unstable_by` for performance and to handle non-total-order of f64.
        sorted_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_data.len();
        let quantiles_vec = (1..=num_quantiles)
            .map(|k| {
                let p = k as f64 / (num_quantiles as f64 + 1.0);
                let float_idx = (n as f64 - 1.0) * p;
                let lower_idx = float_idx.floor() as usize;
                let upper_idx = float_idx.ceil() as usize;

                if lower_idx == upper_idx {
                    sorted_data[lower_idx]
                } else {
                    let fraction = float_idx - lower_idx as f64;
                    sorted_data[lower_idx] * (1.0 - fraction) + sorted_data[upper_idx] * fraction
                }
            })
            .collect();

        Ok(Array1::from_vec(quantiles_vec))
    }

    /// Evaluates all B-spline basis functions at a single point `x`.
    /// This uses a numerically stable implementation of the Cox-de Boor algorithm,
    /// based on Algorithm A2.2 from "The NURBS Book" by Piegl and Tiller.
    pub(super) fn evaluate_splines_at_point(
        x: f64,
        degree: usize,
        knots: ArrayView1<f64>,
    ) -> Array1<f64> {
        let num_knots = knots.len();
        let num_basis = num_knots - degree - 1;

        // Clamp x to the valid domain defined by the knots.
        // The valid domain for a spline of degree `d` is between knot `t_d` and `t_{n+1}`
        // where n is the number of basis functions.
        let x_clamped = x.clamp(knots[degree], knots[num_basis]);

        // Find the knot span `mu` such that knots[mu] <= x < knots[mu+1].
        // This search is robust and correctly handles the half-open interval convention
        // and the special case for the upper boundary.
        let mu = {
            // Special case for the upper boundary, where the interval is closed.
            if x_clamped >= knots[num_basis] {
                // If x is at or beyond the last knot of the spline's support,
                // it belongs to the last valid span. num_basis = (num_knots - degree - 1)
                num_basis - 1
            } else {
                // Search for the span in the relevant part of the knot vector.
                // Can be optimized with binary search, but linear is fine and robust.
                let mut span = degree;
                while span < num_basis && x_clamped >= knots[span + 1] {
                    span += 1;
                }
                span
            }
        };

        // `n` will store the non-zero basis function values.
        // At any point x, at most `degree + 1` basis functions are non-zero.
        let mut n = Array1::zeros(degree + 1);
        let mut left = Array1::zeros(degree + 1);
        let mut right = Array1::zeros(degree + 1);

        // Base case (d=0)
        n[0] = 1.0;

        // Iteratively compute values for higher degrees (d=1 to degree)
        for d in 1..=degree {
            left[d] = x_clamped - knots[mu + 1 - d];
            right[d] = knots[mu + d] - x_clamped;

            let mut saved = 0.0;

            for r in 0..d {
                // This is an in-place update. n[r] on input is a value for degree d-1.
                let den = right[r + 1] + left[d - r];
                let temp = if den.abs() > 1e-12 { n[r] / den } else { 0.0 };

                // On output, n[r] will be a value for degree d.
                n[r] = saved + right[r + 1] * temp;
                saved = left[d - r] * temp;
            }
            n[d] = saved;
        }

        // `n` now contains the values of the `degree + 1` non-zero basis functions.
        // n[j] corresponds to B_{mu-degree+j, degree}.
        // Place them in the correct locations in the final full basis vector.
        let mut basis_values = Array1::zeros(num_basis);
        let start_index = mu.saturating_sub(degree);

        for i in 0..=degree {
            let global_idx = start_index + i;
            if global_idx < num_basis {
                basis_values[global_idx] = n[i];
            }
        }

        basis_values
    }
}

// Unit tests are crucial for a mathematical library like this.
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Independent recursive implementation of B-spline basis function evaluation.
    /// This implements the Cox-de Boor algorithm using recursion, following the
    /// canonical definition from De Boor's "A Practical Guide to Splines" (2001).
    /// This can be used to cross-validate the iterative implementation in evaluate_splines_at_point.
    fn evaluate_bspline(x: f64, knots: &Array1<f64>, i: usize, degree: usize) -> f64 {
        // Base case for degree 0
        if degree == 0 {
            // A degree-0 B-spline B_{i,0}(x) is an indicator function for the knot interval [knots[i], knots[i+1]).
            // It should be 1 if x is inside its corresponding knot interval, and 0 otherwise.
            // The key subtlety is the boundary condition needed to ensure partition of unity.

            // Clamp x to the spline's domain for consistent boundary handling
            let x_clamped = x.clamp(knots[0], knots[knots.len() - 1]);

            // 1. Guard against zero-length intervals (repeated knots)
            if (knots[i + 1] - knots[i]).abs() < 1e-12 {
                return 0.0;
            }

            // 2. Standard half-open interval check: [knots[i], knots[i+1])
            if x_clamped >= knots[i] && x_clamped < knots[i + 1] {
                return 1.0;
            }

            // 3. Critical boundary exception for partition of unity:
            // The rightmost non-zero-length interval must be treated as closed [t_i, t_{i+1}]
            // to ensure that the basis functions sum to 1 at the upper boundary.
            // Find the last non-zero-length interval
            let mut last_nonzero_interval = None;
            for j in (0..(knots.len() - 1)).rev() {
                if (knots[j + 1] - knots[j]).abs() >= 1e-12 {
                    last_nonzero_interval = Some(j);
                    break;
                }
            }

            if let Some(last_interval) = last_nonzero_interval {
                if i == last_interval && x_clamped == knots[i + 1] {
                    return 1.0;
                }
            }

            // 4. All other cases: x is outside this basis function's support
            return 0.0;
        }

        // Recursion for degree > 0
        let mut result = 0.0;

        // First term
        if (i + degree < knots.len()) && (knots[i + degree] - knots[i] > 1e-10) {
            result += (x - knots[i]) / (knots[i + degree] - knots[i])
                * evaluate_bspline(x, knots, i, degree - 1);
        }

        // Second term
        if (i + 1 < knots.len())
            && (i + degree + 1 < knots.len())
            && (knots[i + degree + 1] - knots[i + 1] > 1e-10)
        {
            result += (knots[i + degree + 1] - x) / (knots[i + degree + 1] - knots[i + 1])
                * evaluate_bspline(x, knots, i + 1, degree - 1);
        }

        return result;
    }

    #[test]
    fn test_knot_generation_uniform() {
        let knots = internal::generate_full_knot_vector((0.0, 10.0), 3, 2, None).unwrap();
        // 3 internal + 2 * (2+1) boundary = 9 knots
        assert_eq!(knots.len(), 9);
        assert_abs_diff_eq!(
            knots,
            array![0.0, 0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0],
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_knot_generation_quantile() {
        let training_data = array![0., 1., 2., 5., 8., 9., 10.]; // 7 points
        let knots =
            internal::generate_full_knot_vector((0.0, 10.0), 3, 2, Some(training_data.view()))
                .unwrap();
        // Quantiles at 1/4, 2/4, 3/4.
        // p=0.25 -> idx=(7-1)*0.25=1.5 -> (data[1]+data[2])/2 = (1+2)/2=1.5
        // p=0.50 -> idx=(7-1)*0.50=3.0 -> data[3] = 5.0
        // p=0.75 -> idx=(7-1)*0.75=4.5 -> (data[4]+data[5])/2 = (8+9)/2=8.5
        assert_abs_diff_eq!(
            knots,
            array![0.0, 0.0, 0.0, 1.5, 5.0, 8.5, 10.0, 10.0, 10.0],
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_penalty_matrix_creation() {
        let s = create_difference_penalty_matrix(5, 2).unwrap();
        assert_eq!(s.shape(), &[5, 5]);
        // D_2 for n=5 is [[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]]
        // s = d_2' * d_2
        let expected_s = array![
            [1., -2., 1., 0., 0.],
            [-2., 5., -4., 1., 0.],
            [1., -4., 6., -4., 1.],
            [0., 1., -4., 5., -2.],
            [0., 0., 1., -2., 1.]
        ];
        assert_abs_diff_eq!(s, expected_s, epsilon = 1e-9);
    }

    #[test]
    fn test_bspline_basis_sums_to_one() {
        let data = Array::linspace(0.1, 9.9, 100);
        let (basis, _) = create_bspline_basis(data.view(), None, (0.0, 10.0), 10, 3).unwrap();

        let sums = basis.sum_axis(Axis(1));

        // Every row should sum to 1.0 (with floating point tolerance)
        for &sum in sums.iter() {
            assert!(
                (sum - 1.0).abs() < 1e-9,
                "Basis did not sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_single_point_evaluation_degree_one() {
        // Test a single point where the value can be calculated by hand for a simple case.
        // Degree 1 (linear) splines with knots t = [0,0,1,2,2].
        // This gives 3 basis functions (n = k-d-1 = 5-1-1 = 3), B_{0,1}, B_{1,1}, B_{2,1}.
        let knots = array![0.0, 0.0, 1.0, 2.0, 2.0];
        let x = 0.5; // For x=0.5, the knot interval is mu=1, since t_1 <= x < t_2.

        let values = internal::evaluate_splines_at_point(x, 1, knots.view());
        assert_eq!(values.len(), 3);

        // Manual calculation for x=0.5:
        // The only non-zero basis function of degree 0 is B_{1,0} = 1.
        // Recurrence for degree 1:
        // B_{0,1}(x) = ( (x-t0)/(t1-t0) )*B_{0,0} + ( (t2-x)/(t2-t1) )*B_{1,0}
        //           = ( (0.5-0)/(0-0) )*0       + ( (1-0.5)/(1-0) )*1         = 0.5
        //           (Note: 0/0 division is taken as 0)
        // B_{1,1}(x) = ( (x-t1)/(t2-t1) )*B_{1,0} + ( (t3-x)/(t3-t2) )*B_{2,0}
        //           = ( (0.5-0)/(1-0) )*1       + ( (2-0.5)/(2-1) )*0         = 0.5
        // B_{2,1}(x) = ( (x-t2)/(t3-t2) )*B_{2,0} + ( (t4-x)/(t4-t3) )*B_{3,0}
        //           = ( (0.5-1)/(2-1) )*0       + ( (2-0.5)/(2-2) )*0         = 0.0

        assert!(
            (values[0] - 0.5).abs() < 1e-9,
            "Expected B_0,1 to be 0.5, got {}",
            values[0]
        );
        assert!(
            (values[1] - 0.5).abs() < 1e-9,
            "Expected B_1,1 to be 0.5, got {}",
            values[1]
        );
        assert!(
            (values[2] - 0.0).abs() < 1e-9,
            "Expected B_2,1 to be 0.0, got {}",
            values[2]
        );
    }

    #[test]
    fn test_cox_de_boor_higher_degree() {
        // Test that verifies the Cox-de Boor denominator handling for higher degree splines
        // Using non-uniform knots where numerical issues would be more apparent
        let knots = array![0.0, 0.0, 0.0, 1.0, 3.0, 4.0, 4.0, 4.0];
        let x = 2.0;

        let values = internal::evaluate_splines_at_point(x, 2, knots.view());

        // The basis functions should sum to 1.0 (partition of unity property)
        let sum = values.sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Basis functions should sum to 1.0, got {}",
            sum
        );

        // All values should be non-negative
        for (i, &val) in values.iter().enumerate() {
            assert!(
                val >= -1e-9,
                "Basis function {} should be non-negative, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_boundary_value_handling() {
        // Test for proper boundary value handling at the upper boundary.
        // This test ensures that evaluation at the upper boundary works correctly.

        // Test the internal function directly with the problematic case
        let knots = array![
            0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 10.0, 10.0, 10.0
        ];
        let x = 10.0; // This is the value that caused the panic
        let degree = 3;

        let basis_values = internal::evaluate_splines_at_point(x, degree, knots.view());

        // Should not panic and should return valid results
        assert_eq!(basis_values.len(), 8); // num_basis = 12 - 3 - 1 = 8

        let sum = basis_values.sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "Basis functions should sum to 1.0 at boundary, got {}",
            sum
        );
    }

    #[test]
    fn test_basis_boundary_values() {
        // Property-based test: Verify boundary conditions using mathematical properties
        // This complements the cross-validation test by testing fundamental B-spline properties

        // A cubic B-spline basis. Knots are [0,0,0,0, 1,2,3, 4,4,4,4].
        // The domain is [0, 4].
        let knots = array![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let degree = 3;
        let num_basis = knots.len() - degree - 1; // 11 - 3 - 1 = 7

        // Test at the lower boundary (x=0)
        let basis_at_start = internal::evaluate_splines_at_point(0.0, degree, knots.view());

        // At the very start of the domain, only the first basis function should be non-zero (and equal to 1).
        assert_abs_diff_eq!(basis_at_start[0], 1.0, epsilon = 1e-9);
        for i in 1..num_basis {
            assert_abs_diff_eq!(basis_at_start[i], 0.0, epsilon = 1e-9);
        }

        // Test at the upper boundary (x=4)
        let basis_at_end = internal::evaluate_splines_at_point(4.0, degree, knots.view());

        // At the very end of the domain, only the LAST basis function should be non-zero (and equal to 1).
        for i in 0..(num_basis - 1) {
            assert_abs_diff_eq!(basis_at_end[i], 0.0, epsilon = 1e-9);
        }
        assert_abs_diff_eq!(basis_at_end[num_basis - 1], 1.0, epsilon = 1e-9);

        // Test intermediate points for partition of unity
        let test_points = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        for &x in &test_points {
            let basis = internal::evaluate_splines_at_point(x, degree, knots.view());
            let sum: f64 = basis.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-9);
            if (sum - 1.0).abs() >= 1e-9 {
                panic!("Partition of unity failed at x={}", x);
            }
        }
    }

    #[test]
    fn test_degree_0_boundary_behavior() {
        let knots = array![0.0, 0.0, 1.0, 2.0, 2.0];
        let x = 2.0;

        println!("Testing degree-0 basis functions at x=2:");
        println!("Knots: {:?}", knots);
        println!();

        // Test each possible degree-0 basis function
        for i in 0..(knots.len() - 1) {
            let value = evaluate_bspline(x, &knots, i, 0);
            println!(
                "B_{},0 at x=2: {} (interval [{}, {}))",
                i,
                value,
                knots[i],
                knots[i + 1]
            );

            // Manual check
            let x_clamped = x.clamp(knots[0], knots[knots.len() - 1]);
            let in_interval = x_clamped >= knots[i] && x_clamped < knots[i + 1];
            let is_last_interval = i == knots.len() - 2;
            let at_upper_boundary = x_clamped == knots[i + 1];

            println!(
                "  x_clamped={}, in_interval={}, is_last_interval={}, at_upper_boundary={}",
                x_clamped, in_interval, is_last_interval, at_upper_boundary
            );

            if (knots[i + 1] - knots[i]).abs() < 1e-12 {
                println!("  Zero-length interval, should return 0");
            } else if in_interval {
                println!("  In interval, should return 1");
            } else if is_last_interval && at_upper_boundary {
                println!("  Last interval + upper boundary, should return 1");
            } else {
                println!("  Outside, should return 0");
            }
            println!();
        }
    }

    #[test]
    fn test_boundary_analysis() {
        // Test case from the failing test: knots [0, 0, 1, 2, 2], degree 1, x=2
        let knots = array![0.0, 0.0, 1.0, 2.0, 2.0];
        let degree = 1;
        let x = 2.0;

        println!("Testing knots: {:?}", knots);
        println!("Degree: {}", degree);
        println!("Point: x = {}", x);
        println!();

        // The number of basis functions for this case
        let num_basis = knots.len() - degree - 1;
        println!("Number of basis functions: {}", num_basis);

        // Test each basis function individually with recursive implementation
        for i in 0..num_basis {
            let value = evaluate_bspline(x, &knots, i, degree);
            println!("Recursive B_{},{} at x={}: {}", i, degree, x, value);

            // Also test degree 0 components to understand boundary handling
            if degree == 1 {
                let b_i_0 = evaluate_bspline(x, &knots, i, 0);
                let b_i1_0 = if i + 1 < knots.len() - 1 {
                    evaluate_bspline(x, &knots, i + 1, 0)
                } else {
                    0.0
                };
                println!(
                    "  Components: B_{},0={}, B_{},0={}",
                    i,
                    b_i_0,
                    i + 1,
                    b_i1_0
                );
            }
        }

        // Test with iterative implementation
        let iterative_basis = internal::evaluate_splines_at_point(x, degree, knots.view());
        println!();
        println!("Iterative implementation:");
        for i in 0..num_basis {
            println!(
                "Iterative B_{},{} at x={}: {}",
                i, degree, x, iterative_basis[i]
            );
        }

        // Check sum
        let recursive_sum: f64 = (0..num_basis)
            .map(|i| evaluate_bspline(x, &knots, i, degree))
            .sum();
        let iterative_sum = iterative_basis.sum();
        println!();
        println!("Recursive sum: {}", recursive_sum);
        println!("Iterative sum: {}", iterative_sum);

        // Test the specific failing case: basis function 2 at x=2
        println!();
        println!("Specific failing case:");
        println!(
            "Recursive B_{},{} at x={}: {}",
            2,
            degree,
            x,
            evaluate_bspline(x, &knots, 2, degree)
        );
        println!(
            "Iterative B_{},{} at x={}: {}",
            2, degree, x, iterative_basis[2]
        );

        // Let's manually trace the recursion for B_2,1(x=2)
        println!();
        println!("Manual trace of B_2,1(x=2):");
        println!("B_2,1(x) = (x - t_2)/(t_3 - t_2) * B_2,0(x) + (t_4 - x)/(t_4 - t_3) * B_3,0(x)");
        println!("t_2 = {}, t_3 = {}, t_4 = {}", knots[2], knots[3], knots[4]);

        let b20 = evaluate_bspline(x, &knots, 2, 0);
        let b30 = evaluate_bspline(x, &knots, 3, 0);

        println!("B_2,0(x=2) = {}", b20);
        println!("B_3,0(x=2) = {}", b30);

        let term1 = if (knots[3] - knots[2]).abs() > 1e-10 {
            (x - knots[2]) / (knots[3] - knots[2]) * b20
        } else {
            0.0
        };

        let term2 = if (knots[4] - knots[3]).abs() > 1e-10 {
            (knots[4] - x) / (knots[4] - knots[3]) * b30
        } else {
            0.0
        };

        println!(
            "Term 1: ({} - {}) / ({} - {}) * {} = {}",
            x, knots[2], knots[3], knots[2], b20, term1
        );
        println!(
            "Term 2: ({} - {}) / ({} - {}) * {} = {}",
            knots[4], x, knots[4], knots[3], b30, term2
        );
        println!("Total: {} + {} = {}", term1, term2, term1 + term2);
    }

    #[test]
    fn test_iterative_vs_recursive_bspline() {
        // This test validates our optimized iterative Cox-de Boor implementation against
        // a canonical recursive reference implementation. Both implementations should now
        // handle boundary conditions identically, including points that fall exactly on knots.
        // This cross-validation ensures mathematical correctness of our production code.

        // Test with various degrees and knot configurations
        let test_cases = vec![
            // (knots, degree)
            (array![0.0, 0.0, 1.0, 2.0, 2.0], 1), // Linear spline with 1 internal knot
            (array![0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0], 2), // Quadratic with 3 internal knots
            (array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0], 3), // Cubic with 1 internal knot
        ];

        // Test points including interior values, exact knot positions, and boundaries
        // This comprehensive set tests the half-open interval [t_i, t_{i+1}) convention
        let test_points = vec![
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
        ];

        for (knots, degree) in test_cases {
            for &x in &test_points {
                // Skip points outside the domain
                if x < knots[degree] || x > knots[knots.len() - degree - 1] {
                    continue;
                }

                // Get basis values from iterative implementation
                let iterative_basis = internal::evaluate_splines_at_point(x, degree, knots.view());

                // Calculate basis values using recursive implementation
                let num_basis = knots.len() - degree - 1;
                let mut recursive_basis = Array1::zeros(num_basis);

                for i in 0..num_basis {
                    recursive_basis[i] = evaluate_bspline(x, &knots, i, degree);
                }

                // Compare results - they should match exactly
                for i in 0..num_basis {
                    assert!(
                        (iterative_basis[i] - recursive_basis[i]).abs() < 1e-10,
                        "B-spline basis function {} at x={} doesn't match between implementations. \
                         Iterative: {}, Recursive: {}",
                        i,
                        x,
                        iterative_basis[i],
                        recursive_basis[i]
                    );
                }

                // Check sum-to-one property for both implementations
                let iterative_sum = iterative_basis.sum();
                let recursive_sum = recursive_basis.sum();
                assert!(
                    (iterative_sum - 1.0).abs() < 1e-9,
                    "Iterative implementation should sum to 1.0, got {}",
                    iterative_sum
                );
                assert!(
                    (recursive_sum - 1.0).abs() < 1e-9,
                    "Recursive implementation should sum to 1.0, got {}",
                    recursive_sum
                );
            }
        }
    }

    /// Validates the basis functions against Example 1 in Starkey's "Cox-deBoor" notes.
    ///
    /// This example is a linear spline (degree=1, order=2) with a uniform knot vector.
    /// We test the values of the blending functions at specific points to ensure they
    /// match the manually derived formulas in the literature.
    ///
    /// Reference: Denbigh Starkey, "Cox-deBoor Equations for B-Splines", pg. 8.
    #[test]
    fn test_starkey_notes_example_1() {
        let degree = 1;
        // The book uses knot vector (0, 1, 2, 3, 4, 5).
        // Our setup requires boundary knots. For num_internal_knots = 4, range (0,5),
        // we get internal knots {1,2,3,4}, full vector {0,0, 1,2,3,4, 5,5}.
        let knots = array![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 5.0];
        let num_basis = knots.len() - degree - 1; // 8 - 1 - 1 = 6 basis functions

        // Test case 1: u = 1.5, which is in the span [1, 2].
        // Expected: Two non-zero basis functions, each with value 0.5
        let basis_at_1_5 = internal::evaluate_splines_at_point(1.5, degree, knots.view());
        assert_eq!(basis_at_1_5.len(), num_basis);
        assert_abs_diff_eq!(basis_at_1_5.sum(), 1.0, epsilon = 1e-9);

        // Validate that exactly 2 basis functions are non-zero with value 0.5 each
        let non_zero_count = basis_at_1_5.iter().filter(|&&x| x > 1e-12).count();
        assert_eq!(
            non_zero_count, 2,
            "Should have exactly 2 non-zero basis functions at x=1.5"
        );

        // Check that the non-zero values are at indices 1 and 2 (as determined empirically)
        // and both have value 0.5 (from linear interpolation)
        assert_abs_diff_eq!(basis_at_1_5[1], 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(basis_at_1_5[2], 0.5, epsilon = 1e-9);

        // Test case 2: u = 2.5, which is in the span [2, 3].
        // Expected: Two non-zero basis functions, each with value 0.5
        let basis_at_2_5 = internal::evaluate_splines_at_point(2.5, degree, knots.view());
        assert_eq!(basis_at_2_5.len(), num_basis);
        assert_abs_diff_eq!(basis_at_2_5.sum(), 1.0, epsilon = 1e-9);

        // Validate that exactly 2 basis functions are non-zero with value 0.5 each
        let non_zero_count_2_5 = basis_at_2_5.iter().filter(|&&x| x > 1e-12).count();
        assert_eq!(
            non_zero_count_2_5, 2,
            "Should have exactly 2 non-zero basis functions at x=2.5"
        );

        // Check that the non-zero values are at indices 2 and 3 (as determined empirically)
        // and both have value 0.5 (from linear interpolation)
        assert_abs_diff_eq!(basis_at_2_5[2], 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(basis_at_2_5[3], 0.5, epsilon = 1e-9);
    }

    #[test]
    fn test_error_conditions() {
        match create_bspline_basis(array![].view(), None, (0.0, 10.0), 5, 0).unwrap_err() {
            BasisError::InvalidDegree(deg) => assert_eq!(deg, 0),
            _ => panic!("Expected InvalidDegree error"),
        }

        match create_bspline_basis(array![].view(), None, (10.0, 0.0), 5, 1).unwrap_err() {
            BasisError::InvalidRange(start, end) => {
                assert_eq!(start, 10.0);
                assert_eq!(end, 0.0);
            }
            _ => panic!("Expected InvalidRange error"),
        }

        match create_bspline_basis(
            array![].view(),
            Some(array![1., 2.].view()),
            (0.0, 10.0),
            3,
            1,
        )
        .unwrap_err()
        {
            BasisError::InsufficientDataForQuantiles {
                num_quantiles,
                num_points,
            } => {
                assert_eq!(num_quantiles, 3);
                assert_eq!(num_points, 2);
            }
            _ => panic!("Expected InsufficientDataForQuantiles error"),
        }

        match create_difference_penalty_matrix(5, 5).unwrap_err() {
            BasisError::InvalidPenaltyOrder { order, num_basis } => {
                assert_eq!(order, 5);
                assert_eq!(num_basis, 5);
            }
            _ => panic!("Expected InvalidPenaltyOrder error"),
        }
    }
}
