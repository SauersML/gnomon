use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use ndarray_linalg::QR;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

    #[error("Penalty order ({order}) must be positive and less than the number of basis functions ({num_basis}).")]
    InvalidPenaltyOrder { order: usize, num_basis: usize },

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
    let n_basis = basis_matrix.ncols();

    // 1. Construct the constraint vector `c = B' * 1`.
    // We want the sum over the data points for each basis function to be zero.
    // This is equivalent to requiring the basis functions to be orthogonal to the intercept.
    let constraint_vec = basis_matrix.sum_axis(Axis(0));

    // Reshape into a column vector for QR decomposition.
    let c = constraint_vec.to_shape((n_basis, 1)).unwrap();

    // 2. Find the null space basis `Z` using QR decomposition.
    // The QR decomposition of a vector `c` gives an orthogonal matrix `Q`
    // whose first column is proportional to `c` and whose remaining columns
    // are orthogonal to `c`, spanning its null space.
    let (q, _r) = c.qr()?;
    
    // The transformation matrix Z is composed of all columns of Q except the first.
    let z_transform = q.slice(s![.., 1..]).to_owned();

    // 3. Create the new, constrained basis by projecting the original basis.
    // B_constrained = B * Z
    let constrained_basis = basis_matrix.dot(&z_transform);

    Ok((constrained_basis, z_transform))
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
    /// This uses a corrected, stable implementation of the Cox-de Boor algorithm
    /// that properly handles the recurrence relation without in-place update issues.
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

                // Right parent spline contribution
                if i > 0 && b_old[i - 1] > 0.0 {
                    let denom = knots[idx + d] - knots[idx];
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

// Unit tests are crucial for a mathematical library like this.
#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_knot_generation_uniform() {
        let knots = internal::generate_full_knot_vector((0.0, 10.0), 3, 2, None).unwrap();
        // 3 internal + 2 * (2+1) boundary = 9 knots
        assert_eq!(knots.len(), 9);
        assert_eq!(
            knots,
            array![0.0, 0.0, 0.0, 2.5, 5.0, 7.5, 10.0, 10.0, 10.0]
        );
    }

    #[test]
    fn test_knot_generation_quantile() {
        let training_data = array![0., 1., 2., 5., 8., 9., 10.]; // 7 points
        let knots = internal::generate_full_knot_vector(
            (0.0, 10.0),
            3,
            2,
            Some(training_data.view()),
        )
        .unwrap();
        // Quantiles at 1/4, 2/4, 3/4.
        // p=0.25 -> idx=(7-1)*0.25=1.5 -> (data[1]+data[2])/2 = (1+2)/2=1.5
        // p=0.50 -> idx=(7-1)*0.50=3.0 -> data[3] = 5.0
        // p=0.75 -> idx=(7-1)*0.75=4.5 -> (data[4]+data[5])/2 = (8+9)/2=8.5
        assert_eq!(
            knots,
            array![0.0, 0.0, 0.0, 1.5, 5.0, 8.5, 10.0, 10.0, 10.0]
        );
    }

    #[test]
    fn test_penalty_matrix_creation() {
        let s = create_difference_penalty_matrix(5, 2).unwrap();
        assert_eq!(s.shape(), &[5, 5]);
        // D_2 for n=5 is [[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]]
        // S = D_2' * D_2
        let expected_s = array![
            [1., -2., 1., 0., 0.],
            [-2., 5., -4., 1., 0.],
            [1., -4., 6., -4., 1.],
            [0., 1., -4., 5., -2.],
            [0., 0., 1., -2., 1.]
        ];
        assert!(s.all_close(&expected_s, 1e-9));
    }

    #[test]
    fn test_bspline_basis_sums_to_one() {
        let data = Array::linspace(0.1, 9.9, 100);
        let (basis, _) =
            create_bspline_basis(data.view(), None, (0.0, 10.0), 10, 3).unwrap();

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

        assert!((values[0] - 0.5).abs() < 1e-9, "Expected B_0,1 to be 0.5, got {}", values[0]);
        assert!((values[1] - 0.5).abs() < 1e-9, "Expected B_1,1 to be 0.5, got {}", values[1]);
        assert!((values[2] - 0.0).abs() < 1e-9, "Expected B_2,1 to be 0.0, got {}", values[2]);
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
            },
            _ => panic!("Expected InvalidRange error"),
        }

        match create_bspline_basis(
            array![].view(),
            Some(array![1., 2.].view()),
            (0.0, 10.0),
            3,
            1
        )
        .unwrap_err() {
            BasisError::InsufficientDataForQuantiles { num_quantiles, num_points } => {
                assert_eq!(num_quantiles, 3);
                assert_eq!(num_points, 2);
            },
            _ => panic!("Expected InsufficientDataForQuantiles error"),
        }

        match create_difference_penalty_matrix(5, 5).unwrap_err() {
            BasisError::InvalidPenaltyOrder { order, num_basis } => {
                assert_eq!(order, 5);
                assert_eq!(num_basis, 5);
            },
            _ => panic!("Expected InvalidPenaltyOrder error"),
        }
    }
}
