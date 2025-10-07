use faer::linalg::matmul::matmul;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Accum, MatMut, MatRef, Par, Side};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::Normal; // Distribution trait is implicitly used by Normal + rng.sample()

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

// Helper function to calculate rank based on eigenvalue tolerance
fn calculate_rank_by_tolerance(
    sorted_desc_eigenvalues: &[f64],
    tolerance_fraction: Option<f64>,
    near_zero_threshold: f64, // Make sure this is passed, e.g., NEAR_ZERO_THRESHOLD
) -> usize {
    match tolerance_fraction {
        Some(tol_frac) => {
            let largest_eigval = sorted_desc_eigenvalues.first().copied().unwrap_or(0.0);

            if largest_eigval <= near_zero_threshold {
                return 0;
            }

            // Ensure tol_frac is clamped between 0.0 and 1.0
            let effective_tol_frac = tol_frac.max(0.0).min(1.0);
            let threshold_val = largest_eigval * effective_tol_frac;

            sorted_desc_eigenvalues
                .iter()
                .take_while(|&&val| val > threshold_val)
                .count()
        }
        None => sorted_desc_eigenvalues.len(),
    }
}

fn center_and_scale_columns(data_matrix: &mut Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let n_samples = data_matrix.nrows();
    let n_features = data_matrix.ncols();

    const PARALLEL_COLUMN_THRESHOLD: usize = 256;

    let mut mean_vector = Array1::<f64>::zeros(n_features);
    let mut scale_vector = Array1::<f64>::zeros(n_features);

    let mean_slice = mean_vector
        .as_slice_mut()
        .expect("mean vector should be contiguous");
    let scale_slice = scale_vector
        .as_slice_mut()
        .expect("scale vector should be contiguous");

    #[inline]
    fn process_column(
        mut column: ArrayViewMut1<'_, f64>,
        mean_slot: &mut f64,
        scale_slot: &mut f64,
        n_samples: usize,
    ) {
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &value in column.iter() {
            sum += value;
            sum_sq += value * value;
        }

        let n_samples_f64 = n_samples as f64;
        let mean = sum / n_samples_f64;
        let variance = if n_samples > 1 {
            let centered_sum_sq = (sum_sq - sum * sum / n_samples_f64).max(0.0);
            let var = centered_sum_sq / ((n_samples - 1) as f64);
            if var.is_finite() { var } else { 0.0 }
        } else {
            0.0
        };

        let std_dev = variance.sqrt();
        let sanitized_std = if !std_dev.is_finite() || std_dev <= NEAR_ZERO_THRESHOLD {
            1.0
        } else {
            std_dev
        };

        for value in column.iter_mut() {
            *value = (*value - mean) / sanitized_std;
        }

        *mean_slot = mean;
        *scale_slot = sanitized_std;
    }

    if n_features >= PARALLEL_COLUMN_THRESHOLD {
        data_matrix
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .zip(mean_slice.par_iter_mut())
            .zip(scale_slice.par_iter_mut())
            .for_each(|((column, mean_slot), scale_slot)| {
                process_column(column, mean_slot, scale_slot, n_samples);
            });
    } else {
        data_matrix
            .axis_iter_mut(Axis(1))
            .into_iter()
            .zip(mean_slice.iter_mut())
            .zip(scale_slice.iter_mut())
            .for_each(|((column, mean_slot), scale_slot)| {
                process_column(column, mean_slot, scale_slot, n_samples);
            });
    }

    (mean_vector, scale_vector)
}

fn compute_covariance_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let (n_samples_total, n_features) = data_matrix.dim();
    if n_features == 0 {
        return Array2::zeros((0, 0));
    }
    let scale = 1.0 / ((n_samples - 1) as f64);
    let mut covariance = Array2::<f64>::zeros((n_features, n_features).f());

    let mut apply_matmul = |mat_a: MatRef<'_, f64>| {
        let cov_slice = covariance
            .as_slice_memory_order_mut()
            .expect("Covariance matrix should provide contiguous storage");
        let cov_dst = MatMut::from_column_major_slice_mut(cov_slice, n_features, n_features);
        matmul(
            cov_dst,
            Accum::Replace,
            mat_a.transpose(),
            mat_a,
            scale,
            Par::rayon(0),
        );
    };

    if let Some(slice) = data_matrix.as_slice_memory_order() {
        let mat_a = if data_matrix.is_standard_layout() {
            MatRef::from_row_major_slice(slice, n_samples_total, n_features)
        } else {
            MatRef::from_column_major_slice(slice, n_samples_total, n_features)
        };
        apply_matmul(mat_a);
    } else {
        let owned = data_matrix.to_owned();
        let slice = owned
            .as_slice_memory_order()
            .expect("Owned copy should be contiguous");
        let mat_a = MatRef::from_row_major_slice(slice, n_samples_total, n_features);
        apply_matmul(mat_a);
    }

    covariance
}

#[cfg(not(feature = "backend_faer"))]
fn compute_covariance_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let mut cov_matrix = data_matrix.t().dot(data_matrix);
    cov_matrix /= (n_samples - 1) as f64;
    cov_matrix
}


fn compute_gram_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let (n_samples_total, n_features) = data_matrix.dim();
    if n_samples_total == 0 {
        return Array2::zeros((0, 0));
    }
    let scale = 1.0 / ((n_samples - 1) as f64);
    let mut gram = Array2::<f64>::zeros((n_samples_total, n_samples_total).f());

    let mut apply_matmul = |mat_a: MatRef<'_, f64>| {
        let gram_slice = gram
            .as_slice_memory_order_mut()
            .expect("Gram matrix should provide contiguous storage");
        let gram_dst =
            MatMut::from_column_major_slice_mut(gram_slice, n_samples_total, n_samples_total);
        matmul(
            gram_dst,
            Accum::Replace,
            mat_a,
            mat_a.transpose(),
            scale,
            Par::rayon(0),
        );
    };

    if let Some(slice) = data_matrix.as_slice_memory_order() {
        let mat_a = if data_matrix.is_standard_layout() {
            MatRef::from_row_major_slice(slice, n_samples_total, n_features)
        } else {
            MatRef::from_column_major_slice(slice, n_samples_total, n_features)
        };
        apply_matmul(mat_a);
    } else {
        let owned = data_matrix.to_owned();
        let slice = owned
            .as_slice_memory_order()
            .expect("Owned copy should be contiguous");
        let mat_a = MatRef::from_row_major_slice(slice, n_samples_total, n_features);
        apply_matmul(mat_a);
    }

    gram
}

#[cfg(not(feature = "backend_faer"))]
fn compute_gram_matrix(data_matrix: &Array2<f64>, n_samples: usize) -> Array2<f64> {
    let mut gram_matrix = data_matrix.dot(&data_matrix.t());
    gram_matrix /= (n_samples - 1) as f64;
    gram_matrix
}


fn compute_feature_space_projection(
    data_matrix: &Array2<f64>,
    u_subset: &Array2<f64>,
) -> Array2<f64> {
    let (n_samples_total, n_features) = data_matrix.dim();
    let (_, final_rank) = u_subset.dim();

    if n_features == 0 || final_rank == 0 {
        return Array2::zeros((n_features, final_rank));
    }

    let mut rotation = Array2::<f64>::zeros((n_features, final_rank).f());
    let mut apply_matmul = |mat_a: MatRef<'_, f64>, mat_u: MatRef<'_, f64>| {
        let rot_slice = rotation
            .as_slice_memory_order_mut()
            .expect("Rotation matrix should provide contiguous storage");
        let rot_dst = MatMut::from_column_major_slice_mut(rot_slice, n_features, final_rank);
        matmul(
            rot_dst,
            Accum::Replace,
            mat_a.transpose(),
            mat_u,
            1.0,
            Par::rayon(0),
        );
    };

    let mut call_with_u = |mat_a: MatRef<'_, f64>| {
        if let Some(u_slice) = u_subset.as_slice_memory_order() {
            let mat_u = if u_subset.is_standard_layout() {
                MatRef::from_row_major_slice(u_slice, n_samples_total, final_rank)
            } else {
                MatRef::from_column_major_slice(u_slice, n_samples_total, final_rank)
            };
            apply_matmul(mat_a, mat_u);
        } else {
            let owned_u = u_subset.to_owned();
            let slice = owned_u
                .as_slice_memory_order()
                .expect("Owned copy should be contiguous");
            let mat_u = MatRef::from_row_major_slice(slice, n_samples_total, final_rank);
            apply_matmul(mat_a, mat_u);
        }
    };

    if let Some(a_slice) = data_matrix.as_slice_memory_order() {
        let mat_a = if data_matrix.is_standard_layout() {
            MatRef::from_row_major_slice(a_slice, n_samples_total, n_features)
        } else {
            MatRef::from_column_major_slice(a_slice, n_samples_total, n_features)
        };
        call_with_u(mat_a);
    } else {
        let owned_a = data_matrix.to_owned();
        let slice = owned_a
            .as_slice_memory_order()
            .expect("Owned copy should be contiguous");
        let mat_a = MatRef::from_row_major_slice(slice, n_samples_total, n_features);
        call_with_u(mat_a);
    }

    rotation
}

#[cfg(not(feature = "backend_faer"))]
fn compute_feature_space_projection(
    data_matrix: &Array2<f64>,
    u_subset: &Array2<f64>,
) -> Array2<f64> {
    data_matrix.t().dot(u_subset)
}


fn faer_eigh_upper(matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>), Box<dyn Error>> {
    let (nrows, ncols) = matrix.dim();
    if nrows != ncols {
        return Err("faer_eigh_upper: matrix must be square".into());
    }
    if nrows == 0 {
        return Ok((Array1::zeros(0), Array2::zeros((0, 0))));
    }

    let eig = if let Some(slice) = matrix.as_slice_memory_order() {
        let mat_ref = if matrix.is_standard_layout() {
            MatRef::from_row_major_slice(slice, nrows, ncols)
        } else {
            MatRef::from_column_major_slice(slice, nrows, ncols)
        };
        SelfAdjointEigen::new(mat_ref, Side::Upper)
            .map_err(|e| format!("faer self_adjoint_eigen failed: {:?}", e))?
    } else {
        let owned = matrix.to_owned();
        let slice = owned
            .as_slice_memory_order()
            .expect("Owned matrix copy should be contiguous");
        let mat_ref = MatRef::from_row_major_slice(slice, nrows, ncols);
        SelfAdjointEigen::new(mat_ref, Side::Upper)
            .map_err(|e| format!("faer self_adjoint_eigen failed: {:?}", e))?
    };

    let mut eigenvalues = Array1::<f64>::zeros(ncols);
    for (dst, src) in eigenvalues.iter_mut().zip(eig.S().column_vector().iter()) {
        *dst = *src;
    }

    let mut eigenvectors = Array2::<f64>::zeros((nrows, ncols).f());
    if nrows > 0 {
        let slice_mut = eigenvectors
            .as_slice_memory_order_mut()
            .expect("column-major allocation must be contiguous");
        let mut dst = MatMut::from_column_major_slice_mut(slice_mut, nrows, ncols);
        dst.copy_from(eig.U());
    }

    Ok((eigenvalues, eigenvectors))
}

/// Principal component analysis (PCA) structure.
///
/// This struct holds the results of a PCA (mean, scale, and rotation matrix)
/// and can be used to transform data into the principal component space.
/// It supports both exact PCA computation and a faster, approximate randomized PCA.
/// Models can also be loaded from/saved to files.
#[derive(Serialize, Deserialize, Debug)]
pub struct PCA {
    /// The rotation matrix (principal components).
    /// Shape: (n_features, k_components)
    pub rotation: Option<Array2<f64>>,
    /// Mean vector of the original training data.
    /// Shape: (n_features)
    pub mean: Option<Array1<f64>>,
    /// Sanitized scale vector, representing standard deviations of the original training data.
    /// This vector is guaranteed to contain only positive values.
    /// When set via `with_model`, input `raw_standard_deviations` `s` where `!s.is_finite()` or `s <= 1e-9` are replaced by `1.0`.
    /// Loaded models are also validated so scale factors are positive.
    /// Shape: (n_features)
    pub scale: Option<Array1<f64>>,
    /// Explained variance for each principal component (eigenvalues of the covariance matrix).
    /// Shape: (k_components)
    pub explained_variance: Option<Array1<f64>>,
}

impl Default for PCA {
    fn default() -> Self {
        Self::new()
    }
}

// Public constants for thresholds and clamping values
pub const NEAR_ZERO_THRESHOLD: f64 = 1e-9;
pub const EIGENVALUE_CLAMP_MIN: f64 = 0.0;
pub const NORMALIZATION_THRESHOLD: f64 = 1e-9;
pub const SCALE_SANITIZATION_THRESHOLD: f64 = 1e-9;

/// Creates a new, empty PCA struct.
///
/// or loaded using `load_model` or `with_model`.
///
/// # Examples
///
/// ```
/// use efficient_pca::PCA; // Assuming efficient_pca is your crate name
/// let pca = PCA::new();
/// ```
pub fn new() -> Self {
    Self {
        rotation: None,
        mean: None,
        scale: None,
        explained_variance: None,
    }
}

/// Creates a new PCA instance from a pre-computed model.
///
/// This is useful for loading a PCA model whose components (rotation matrix,
/// mean, and original standard deviations) were computed externally or
/// previously. The library will sanitize the provided standard deviations
/// for consistent scaling.
///
/// * `rotation` - The rotation matrix (principal components), shape (d_features, k_components).
/// * `mean` - The mean vector of the original data used to compute the PCA, shape (d_features).
/// * `raw_standard_deviations` - The raw standard deviation vector of the original data,
///                               shape (d_features). Values that are not strictly positive
///                               (i.e., `s <= 1e-9`, zero, negative), or are non-finite,
///                               will be sanitized to `1.0` before being stored.
///                               If the original PCA did not involve scaling (e.g., data was
///                               already standardized, or only centering was desired),
///                               pass a vector of ones.
///
/// # Errors
/// Returns an error if feature dimensions are inconsistent or if `raw_standard_deviations`
/// contains non-finite values (this check is performed before sanitization).
pub fn with_model(
    rotation: Array2<f64>,
    mean: Array1<f64>,
    raw_standard_deviations: Array1<f64>,
) -> Result<Self, Box<dyn Error>> {
    let d_features_rotation = rotation.nrows();
    let k_components = rotation.ncols();
    let d_features_mean = mean.len();
    let d_features_raw_std = raw_standard_deviations.len();

    if !(d_features_rotation == d_features_mean && d_features_mean == d_features_raw_std) {
        if !(d_features_rotation == 0
            && k_components == 0
            && d_features_mean == 0
            && d_features_raw_std == 0)
        {
            return Err(format!(
                "PCA::with_model: Feature dimensions of rotation ({}), mean ({}), and raw_standard_deviations ({}) must match.",
                d_features_rotation, d_features_mean, d_features_raw_std
            ).into());
        }
    }

    if d_features_rotation == 0 && k_components > 0 {
        return Err(
            "PCA::with_model: Rotation matrix has 0 features but expects components.".into(),
        );
    }

    if raw_standard_deviations.iter().any(|&val| !val.is_finite()) {
        // Explicitly reject non-finite inputs early.
        return Err("PCA::with_model: raw_standard_deviations contains non-finite (NaN or infinity) values.".into());
    }

    // Sanitize scale factors:
    // All scale factors are positive. Values that are not strictly positive (<= SCALE_SANITIZATION_THRESHOLD),
    // or were non-finite (though checked above), are replaced with 1.0.
    let sanitized_scale_vector = raw_standard_deviations.mapv(|val| {
        if val.is_finite() && val > SCALE_SANITIZATION_THRESHOLD {
            val
        } else {
            1.0
        }
    });

    Ok(Self {
        rotation: Some(rotation),
        mean: Some(mean),
        scale: Some(sanitized_scale_vector),
        explained_variance: None, // Explained variance is not provided by this constructor directly
    })
}

/// Returns a reference to the mean vector of the original training data, if computed.
///
/// The mean vector has dimensions (n_features).
/// Returns `None` if the PCA model has not been fitted.
pub fn mean(&self) -> Option<&Array1<f64>> {
    self.mean.as_ref()
}

/// Returns a reference to the sanitized scale vector (standard deviations), if computed.
///
/// The scale vector has dimensions (n_features) and contains positive values.
/// Returns `None` if the PCA model has not been fitted.
pub fn scale(&self) -> Option<&Array1<f64>> {
    self.scale.as_ref()
}

/// Returns a reference to the rotation matrix (principal components), if computed.
///
/// The rotation matrix has dimensions (n_features, k_components).
/// Returns `None` if the PCA model has not been fitted, or if the rotation matrix
/// is not available (e.g., if fitting resulted in zero components).
pub fn rotation(&self) -> Option<&Array2<f64>> {
    self.rotation.as_ref()
}

/// Returns a reference to the explained variance for each principal component.
///
/// These are the eigenvalues of the covariance matrix of the scaled data,
/// ordered from largest to smallest.
/// Returns `None` if the PCA model has not been fitted or if variances are not available.
pub fn explained_variance(&self) -> Option<&Array1<f64>> {
    self.explained_variance.as_ref()
}

/// Fits the PCA model to the data using an exact covariance/Gram matrix approach.
///
/// This method computes the mean, (sanitized) scaling factors, and principal axes (rotation)
/// via an eigen-decomposition of the covariance matrix (if n_features <= n_samples)
/// or the Gram matrix (if n_features > n_samples, the "Gram trick").
/// The resulting principal components (columns of the rotation matrix) are normalized to unit length.
///
///
/// * `data_matrix` - Input data as a 2D array, shape (n_samples, n_features).
/// * `tolerance` - Optional: Tolerance for excluding low-variance components
///                 (fraction of the largest eigenvalue). If `None`, all components
///                 up to the effective rank of the matrix are kept.
///
/// # Errors
/// Returns an error if the input matrix has zero dimensions, fewer than 2 samples, or if
/// matrix operations (like eigen-decomposition) fail.
pub fn fit(
    &mut self,
    mut data_matrix: Array2<f64>,
    tolerance: Option<f64>,
) -> Result<(), Box<dyn Error>> {
    let n_samples = data_matrix.nrows();
    let n_features = data_matrix.ncols();

    if n_samples == 0 || n_features == 0 {
        return Err("PCA::fit: Input data_matrix has zero samples or zero features.".into());
    }
    if n_samples < 2 {
        return Err("PCA::fit: Input matrix must have at least 2 samples.".into());
    }

    let (mean_vector, sanitized_scale_vector) = center_and_scale_columns(&mut data_matrix);
    self.mean = Some(mean_vector);
    self.scale = Some(sanitized_scale_vector);

    #[cfg(not(feature = "backend_faer"))]
    let backend = LinAlgBackendProvider::<f64>::new();

    
    faer::set_global_parallelism(Par::rayon(0));

    if n_features <= n_samples {
        let cov_matrix = compute_covariance_matrix(&data_matrix, n_samples);

        
        let (eigenvalues, eigenvectors) = faer_eigh_upper(&cov_matrix).map_err(|e| {
            format!(
                "PCA::fit (Covariance path): Eigen decomposition of covariance matrix failed (via faer): {}",
                e
            )
        })?;
        #[cfg(not(feature = "backend_faer"))]
        let (eigenvalues, eigenvectors) = {
            let eigh_result = backend.eigh_upper(&cov_matrix).map_err(|e| {
                format!(
                    "PCA::fit (Covariance path): Eigen decomposition of covariance matrix failed (via backend): {}",
                    e
                )
            })?;
            (eigh_result.eigenvalues, eigh_result.eigenvectors)
        };

        let eigenvalues_desc: Vec<f64> = eigenvalues.iter().rev().copied().collect();
        let rank_limit =
            calculate_rank_by_tolerance(&eigenvalues_desc, tolerance, NEAR_ZERO_THRESHOLD);

        let final_rank = std::cmp::min(rank_limit, n_features);

        if final_rank == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
        } else {
            let mut explained_variance = Array1::<f64>::zeros(final_rank);
            let mut rotation_matrix = Array2::<f64>::zeros((n_features, final_rank));
            let total = eigenvalues.len();

            for component_idx in 0..final_rank {
                let eigen_idx = total - 1 - component_idx;
                let eigenvalue = eigenvalues[eigen_idx].max(EIGENVALUE_CLAMP_MIN);
                explained_variance[component_idx] = eigenvalue;
                rotation_matrix
                    .column_mut(component_idx)
                    .assign(&eigenvectors.column(eigen_idx));
            }

            self.rotation = Some(rotation_matrix);
            self.explained_variance = Some(explained_variance);
        }
    } else {
        let gram_matrix = compute_gram_matrix(&data_matrix, n_samples);

        
        let (gram_eigenvalues, gram_eigenvectors_u) = faer_eigh_upper(&gram_matrix).map_err(|e| {
            format!(
                "PCA::fit (Gram trick): Eigen decomposition of Gram matrix failed (via faer): {}",
                e
            )
        })?;
        #[cfg(not(feature = "backend_faer"))]
        let (gram_eigenvalues, gram_eigenvectors_u) = {
            let eigh_result_gram = backend.eigh_upper(&gram_matrix).map_err(|e| {
                format!(
                    "PCA::fit (Gram trick): Eigen decomposition of Gram matrix failed (via backend): {}",
                    e
                )
            })?;
            (eigh_result_gram.eigenvalues, eigh_result_gram.eigenvectors)
        };

        let eigenvalues_desc: Vec<f64> = gram_eigenvalues.iter().rev().copied().collect();
        let rank_limit =
            calculate_rank_by_tolerance(&eigenvalues_desc, tolerance, NEAR_ZERO_THRESHOLD);

        let final_rank = std::cmp::min(rank_limit, n_samples);

        if final_rank == 0 {
            self.rotation = Some(Array2::zeros((n_features, 0)));
            self.explained_variance = Some(Array1::zeros(0));
        } else {
            let mut explained_variance = Array1::<f64>::zeros(final_rank);
            let mut u_subset = Array2::<f64>::zeros((n_samples, final_rank));
            let total = gram_eigenvalues.len();

            for component_idx in 0..final_rank {
                let eigen_idx = total - 1 - component_idx;
                let eigenvalue = gram_eigenvalues[eigen_idx].max(EIGENVALUE_CLAMP_MIN);
                explained_variance[component_idx] = eigenvalue;
                u_subset
                    .column_mut(component_idx)
                    .assign(&gram_eigenvectors_u.column(eigen_idx));
            }

            let mut rotation_matrix = compute_feature_space_projection(&data_matrix, &u_subset);
            let scale_factors = explained_variance.map(|&lambda| {
                let denom_squared = (n_samples - 1) as f64 * lambda;
                if denom_squared > NEAR_ZERO_THRESHOLD * NEAR_ZERO_THRESHOLD {
                    1.0 / denom_squared.sqrt()
                } else {
                    0.0
                }
            });
            rotation_matrix *= &scale_factors;

            self.rotation = Some(rotation_matrix);
            self.explained_variance = Some(explained_variance);
        }
    }
    Ok(())
}



/// Applies the PCA transformation to the given data.
///
/// The data is centered and scaled using the mean and scale factors
/// learned during fitting (or loaded into the model), and then projected
/// onto the principal components.
///
/// * `x` - Input data to transform, shape (m_samples, d_features).
///         Can be a single sample (1 row) or multiple samples.
///         This matrix is modified in place.
///
/// # Errors
/// Returns an error if the PCA model is not fitted/loaded (i.e., missing mean,
/// scale, or rotation components), or if the input data's feature dimension
/// does not match the model's feature dimension.
pub fn transform(&self, mut x: Array2<f64>) -> Result<Array2<f64>, Box<dyn Error>> {
    // Retrieve model components, so they exist.
    let rotation_matrix = self.rotation.as_ref().ok_or_else(
        || "PCA::transform: PCA model: Rotation matrix not set. Fit or load a model first.",
    )?;
    let mean_vector = self.mean.as_ref().ok_or_else(
        || "PCA::transform: PCA model: Mean vector not set. Fit or load a model first.",
    )?;
    // self.scale is guaranteed to contain positive, finite values by model construction/loading.
    let scale_vector = self.scale.as_ref().ok_or_else(
        || "PCA::transform: PCA model: Scale vector not set. Fit or load a model first.",
    )?;

    let n_input_samples = x.nrows();
    let n_input_features = x.ncols();
    let n_model_features = mean_vector.len(); // Also self.scale.len() and self.rotation.nrows()

    // Validate dimensions
    if n_input_features != n_model_features {
        return Err(format!(
            "PCA::transform: Input data feature dimension ({}) does not match model's feature dimension ({}).",
            n_input_features, n_model_features
        ).into());
    }
    // Additional internal consistency checks (should hold if model was properly constructed/loaded)
    // These checks are defensive programming.
    if rotation_matrix.nrows() != n_model_features {
        return Err(format!(
            "PCA::transform: Model inconsistency: Rotation matrix feature dimension ({}) does not match model's feature dimension ({}).",
            rotation_matrix.nrows(), n_model_features
        ).into());
    }
    if scale_vector.len() != n_model_features {
        return Err(format!(
            "PCA::transform: Model inconsistency: Scale vector dimension ({}) does not match model's feature dimension ({}).",
            scale_vector.len(), n_model_features
        ).into());
    }

    // Handle empty input data (0 samples)
    if n_input_samples == 0 {
        let k_components = rotation_matrix.ncols();
        return Ok(Array2::zeros((0, k_components))); // Return 0-sample matrix with correct number of components
    }

    // Fuse centering and scaling in a single pass over the data `x`.
    // This modifies `x` in place.
    // Iterate over each row of x (which is an ArrayViewMut1).
    for mut row in x.axis_iter_mut(Axis(0)) {
        // Zip::from iterates over the elements of the row, mean_vector, and scale_vector simultaneously.
        // `row.view_mut()` provides the necessary IntoNdProducer.
        // `mean_vector.view()` and `scale_vector.view()` also provide IntoNdProducer.
        // ?????
    }

    // Project the centered and scaled data onto the principal components
    Ok(x.dot(rotation_matrix))
}

/// Saves the current PCA model to a file using bincode.
///
/// The model must contain rotation, mean, and scale components for saving.
/// The `explained_variance` field can be `None` (e.g., if the model was created
/// via `with_model` and eigenvalues were not supplied).
///
/// * `path` - The file path to save the model to.
///
/// # Errors
/// Returns an error if essential model components (rotation, mean, scale) are missing,
/// or if file I/O or serialization fails.
pub fn save_model<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
    // Rotation, mean, and scale are essential for a model to be usable for transformation.
    if self.rotation.is_none() || self.mean.is_none() || self.scale.is_none() {
        return Err("PCA::save_model: Cannot save a PCA model that is missing essential components (rotation, mean, or scale).".into());
    }
    // explained_variance being None is acceptable, for example, if the model was created
    // using `with_model` and eigenvalues were not provided or computed.
    // `load_model` contains further validation for consistency if explained_variance is Some.
    let file = File::create(path.as_ref()).map_err(|e| {
        format!(
            "PCA::save_model: Failed to create file at {:?}: {}",
            path.as_ref(),
            e
        )
    })?;
    let mut writer = BufWriter::new(file);

    bincode::serde::encode_into_std_write(self, &mut writer, bincode::config::standard())
        .map_err(|e| format!("PCA::save_model: Failed to serialize PCA model: {}", e))?;
    Ok(())
}

/// Loads a PCA model from a file previously saved with `save_model`.
///
/// * `path` - The file path to load the model from.
///
/// # Errors
/// Returns an error if file I/O or deserialization fails, or if the
/// loaded model is found to be incomplete, internally inconsistent (e.g., mismatched dimensions),
/// or contains non-positive scale factors.
pub fn load_model<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
    let file = File::open(path.as_ref()).map_err(|e| {
        format!(
            "PCA::load_model: Failed to open file at {:?}: {}",
            path.as_ref(),
            e
        )
    })?;
    let mut reader = BufReader::new(file);

    let pca_model: PCA =
        bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
            .map_err(|e| format!("PCA::load_model: Failed to deserialize PCA model: {}", e))?;

    let rotation = pca_model
        .rotation
        .as_ref()
        .ok_or("PCA::load_model: Loaded PCA model is missing rotation matrix.")?;
    let mean = pca_model
        .mean
        .as_ref()
        .ok_or("PCA::load_model: Loaded PCA model is missing mean vector.")?;
    let scale = pca_model
        .scale
        .as_ref()
        .ok_or("PCA::load_model: Loaded PCA model is missing scale vector.")?;

    let d_rot_features = rotation.nrows();
    let d_mean_features = mean.len();
    let d_scale_features = scale.len();

    if !(d_rot_features == d_mean_features && d_mean_features == d_scale_features) {
        if !(d_rot_features == 0
            && rotation.ncols() == 0
            && d_mean_features == 0
            && d_scale_features == 0)
        {
            return Err(format!(
                "PCA::load_model: Loaded PCA model has inconsistent feature dimensions: rotation_features={}, mean_features={}, scale_features={}",
                d_rot_features, d_mean_features, d_scale_features
            ).into());
        }
    }
    // Validate that loaded scale factors are positive, aligning with the contract for self.scale.
    // self.scale is expected to store sanitized, positive values (1.0 for original std devs <= SCALE_SANITIZATION_THRESHOLD, else the std dev itself).
    // Scale values must be strictly positive. EIGENVALUE_CLAMP_MIN is 0.0.
    if scale
        .iter()
        .any(|&val| !val.is_finite() || val <= EIGENVALUE_CLAMP_MIN)
    {
        return Err("PCA::load_model: Loaded PCA model's scale vector contains invalid (non-finite, zero, or negative) values. Scale values must be strictly positive.".into());
    }

    // Validate explained_variance if present
    if let Some(ev) = pca_model.explained_variance.as_ref() {
        if let Some(rot) = pca_model.rotation.as_ref() {
            if ev.len() != rot.ncols() {
                return Err(format!(
                    "PCA::load_model: Loaded PCA model has inconsistent dimensions: explained_variance length ({}) does not match rotation matrix number of components ({}).",
                    ev.len(), rot.ncols()
                ).into());
            }
        } else {
            // Should not happen if rotation is required for a valid model
            return Err("PCA::load_model: Loaded PCA model has explained_variance but no rotation matrix.".into());
        }
        if ev
            .iter()
            .any(|&val| !val.is_finite() || val < EIGENVALUE_CLAMP_MIN)
        {
            // Variances cannot be negative
            return Err("PCA::load_model: Loaded PCA model's explained_variance vector contains invalid (non-finite or negative) values.".into());
        }
    }
    // If rotation is Some and has components, but explained_variance is None (e.g. model from `with_model`),
    // this is an acceptable state. The `explained_variance()` accessor will simply return None.
    // If rotation itself is None or has no components (ncols == 0), then explained_variance being None is also consistent.

    Ok(pca_model)
}
