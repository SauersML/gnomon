use core::cmp::min;
use core::fmt;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};
use std::convert::Infallible;
use std::error::Error;
use std::ops::Range;

/// Minimum variance allowed when computing the Hardy–Weinberg scaling factor.
const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
const HWE_SCALE_FLOOR: f64 = 1.0e-6;
/// Eigenvalues below this threshold are discarded when selecting principal components.
const EIGENVALUE_EPSILON: f64 = 1.0e-9;
/// Number of variants to request per streaming chunk.
const DEFAULT_BLOCK_WIDTH: usize = 2_048;

/// Source of genotype data that can be streamed in feature-oriented blocks.
pub trait VariantBlockSource {
    /// Error type produced while reading the underlying storage.
    type Error;

    /// Returns the number of samples (rows) in the matrix.
    fn n_samples(&self) -> usize;
    /// Returns the number of variants (columns) in the matrix.
    fn n_variants(&self) -> usize;

    /// Rewinds the source so that the next call to [`next_block_into`] starts from the first
    /// variant again.
    fn reset(&mut self) -> Result<(), Self::Error>;

    /// Streams the next block of variants into `storage`.
    ///
    /// Implementations must write column-major data (sample-major in contiguous memory) for up to
    /// `max_variants` variants. The slice length is guaranteed to be
    /// `self.n_samples() * max_variants`.
    ///
    /// The return value indicates how many variants were written. Returning `0` signifies end of
    /// stream. Implementations must advance their internal cursor by the number of returned
    /// variants. Once the stream is exhausted, further calls should continue to return `0` until
    /// [`reset`] is invoked.
    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error>;
}

/// In-memory implementation of [`VariantBlockSource`].
///
/// The matrix is expected to be stored in **column-major** order with shape
/// `(n_samples, n_variants)` and genotype dosages encoded as `f64` values.
pub struct DenseBlockSource<'a> {
    data: &'a [f64],
    dims: (usize, usize),
    cursor: usize,
}

impl<'a> DenseBlockSource<'a> {
    /// Creates a new streaming source from column-major data.
    pub fn new(data: &'a [f64], n_samples: usize, n_variants: usize) -> Result<Self, HwePcaError> {
        if n_samples == 0 {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: n_samples must be positive",
            ));
        }
        if n_variants == 0 {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: n_variants must be positive",
            ));
        }
        let expected = n_samples
            .checked_mul(n_variants)
            .ok_or_else(|| HwePcaError::InvalidInput("DenseBlockSource: dimension overflow"))?;
        if data.len() != expected {
            return Err(HwePcaError::InvalidInput(
                "DenseBlockSource: data length does not match dimensions",
            ));
        }
        Ok(Self {
            data,
            dims: (n_samples, n_variants),
            cursor: 0,
        })
    }
}

impl<'a> VariantBlockSource for DenseBlockSource<'a> {
    type Error = Infallible;

    fn n_samples(&self) -> usize {
        self.dims.0
    }

    fn n_variants(&self) -> usize {
        self.dims.1
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.cursor = 0;
        Ok(())
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        let remaining = self.n_variants().saturating_sub(self.cursor);
        if remaining == 0 {
            return Ok(0);
        }
        let ncols = min(max_variants, remaining);
        let nrows = self.n_samples();
        let len = nrows * ncols;
        let start = self.cursor * nrows;
        let end = start + len;
        storage[..len].copy_from_slice(&self.data[start..end]);
        self.cursor += ncols;
        Ok(ncols)
    }
}

/// Errors produced during HWE-scaled PCA fitting.
#[derive(Debug)]
pub enum HwePcaError {
    /// Invalid argument or inconsistent state detected by the algorithm.
    InvalidInput(&'static str),
    /// Error originating from the underlying [`VariantBlockSource`].
    Source(Box<dyn Error + Send + Sync + 'static>),
    /// Eigenvalue decomposition failed or produced no usable eigenpairs.
    Eigen(String),
}

impl fmt::Display for HwePcaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HwePcaError::InvalidInput(msg) => f.write_str(msg),
            HwePcaError::Source(err) => write!(f, "source error: {err}"),
            HwePcaError::Eigen(msg) => write!(f, "eigendecomposition failed: {msg}"),
        }
    }
}

impl Error for HwePcaError {}

/// Result of the dual Gram-based PCA with HWE scaling.
pub struct HwePcaModel {
    /// Number of samples used for training.
    pub n_samples: usize,
    /// Number of variants processed during training.
    pub n_variants: usize,
    /// Per-variant allele frequencies estimated from the training cohort.
    pub allele_frequencies: Vec<f64>,
    /// Per-variant HWE standard deviations used for scaling new genotypes.
    pub variant_scales: Vec<f64>,
    /// Eigenvalues of the sample covariance matrix (descending order).
    pub eigenvalues: Vec<f64>,
    /// Singular values (sqrt((n_samples - 1) * eigenvalues)).
    pub singular_values: Vec<f64>,
    /// Sample eigenvectors (columns of the matrix `U_k`).
    pub sample_eigenvectors: Mat<f64>,
    /// Sample scores (`U_k * diag(singular_values)`).
    pub sample_scores: Mat<f64>,
    /// Variant loadings (`V_k`). Rows correspond to variants, columns to components.
    pub loadings: Mat<f64>,
}

impl HwePcaModel {
    /// Fits the PCA model using a streaming Gram decomposition with Hardy–Weinberg scaling.
    pub fn fit<S>(source: &mut S) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource,
        S::Error: Error + Send + Sync + 'static,
    {
        let n_samples = source.n_samples();
        let n_variants = source.n_variants();

        if n_samples < 2 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least two samples",
            ));
        }
        if n_variants == 0 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least one variant",
            ));
        }

        let block_capacity = min(DEFAULT_BLOCK_WIDTH.max(1), n_variants);
        let mut block_storage = vec![0.0f64; n_samples * block_capacity];

        // Step 1: allele frequency estimation (streaming pass)
        let mut allele_sums = vec![0.0f64; n_variants];
        let mut allele_counts = vec![0usize; n_variants];
        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        let mut processed = 0usize;
        loop {
            let filled = source
                .next_block_into(block_capacity, &mut block_storage)
                .map_err(|e| HwePcaError::Source(Box::new(e)))?;
            if filled == 0 {
                break;
            }
            if processed + filled > n_variants {
                return Err(HwePcaError::InvalidInput(
                    "VariantBlockSource returned more variants than reported",
                ));
            }
            let mut block = MatMut::from_column_major_slice_mut(
                &mut block_storage[..n_samples * filled],
                n_samples,
                filled,
            );
            for local_col in 0..filled {
                let variant_index = processed + local_col;
                let mut sum = 0.0f64;
                let mut calls = 0usize;
                for row in 0..n_samples {
                    let value = block[(row, local_col)];
                    if value.is_finite() {
                        sum += value;
                        calls += 1;
                    }
                }
                allele_sums[variant_index] += sum;
                allele_counts[variant_index] += calls;
            }
            processed += filled;
        }
        if processed != n_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early during allele counting",
            ));
        }

        let mut allele_frequencies = vec![0.0f64; n_variants];
        let mut variant_scales = vec![1.0f64; n_variants];
        for (idx, (&sum, &calls)) in allele_sums.iter().zip(&allele_counts).enumerate() {
            if calls == 0 {
                allele_frequencies[idx] = 0.0;
                variant_scales[idx] = HWE_SCALE_FLOOR;
                continue;
            }
            let mean_genotype = sum / (calls as f64);
            let freq = (mean_genotype / 2.0).clamp(0.0, 1.0);
            let variance = (2.0 * freq * (1.0 - freq)).max(HWE_VARIANCE_EPSILON);
            allele_frequencies[idx] = freq;
            let scale = variance.sqrt();
            variant_scales[idx] = if scale < HWE_SCALE_FLOOR {
                HWE_SCALE_FLOOR
            } else {
                scale
            };
        }

        // Step 2: accumulate Gram matrix with standardized blocks.
        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        let mut gram = Mat::zeros(n_samples, n_samples);
        let scale = 1.0 / ((n_samples - 1) as f64);
        processed = 0;
        loop {
            let filled = source
                .next_block_into(block_capacity, &mut block_storage)
                .map_err(|e| HwePcaError::Source(Box::new(e)))?;
            if filled == 0 {
                break;
            }
            if processed + filled > n_variants {
                return Err(HwePcaError::InvalidInput(
                    "VariantBlockSource returned more variants than reported",
                ));
            }
            {
                let mut block = MatMut::from_column_major_slice_mut(
                    &mut block_storage[..n_samples * filled],
                    n_samples,
                    filled,
                );
                standardize_block(
                    &mut block,
                    processed..processed + filled,
                    &allele_frequencies,
                    &variant_scales,
                );
            }
            let block_ref = MatRef::from_column_major_slice(
                &block_storage[..n_samples * filled],
                n_samples,
                filled,
            );
            matmul(
                gram.as_mut(),
                Accum::Add,
                block_ref,
                block_ref.transpose(),
                scale,
                Par::Seq,
            );
            processed += filled;
        }
        if processed != n_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early during Gram accumulation",
            ));
        }

        // Step 3: eigendecomposition of the Gram matrix.
        let eig = SelfAdjointEigen::new(gram.as_ref(), Side::Upper)
            .map_err(|err| HwePcaError::Eigen(format!("{err:?}")))?;
        let eigen_diag = eig.S().column_vector();
        let eigenvectors = eig.U();

        let mut ordering: Vec<(f64, usize)> = eigen_diag
            .iter()
            .enumerate()
            .map(|(idx, value)| (*value, idx))
            .collect();
        ordering.sort_by(|(lhs, _), (rhs, _)| rhs.partial_cmp(lhs).unwrap_or(std::cmp::Ordering::Equal));

        let mut eigenvalues = Vec::new();
        let mut selected_indices = Vec::new();
        for (value, idx) in ordering.iter().copied() {
            if value <= EIGENVALUE_EPSILON {
                break;
            }
            eigenvalues.push(value);
            selected_indices.push(idx);
        }
        if eigenvalues.is_empty() {
            return Err(HwePcaError::Eigen(
                "All eigenvalues are numerically zero; increase cohort size or review input data"
                    .into(),
            ));
        }

        let n_components = eigenvalues.len();
        let mut sample_eigenvectors = Mat::zeros(n_samples, n_components);
        for (target_col, src_col) in selected_indices.iter().copied().enumerate() {
            for row in 0..n_samples {
                sample_eigenvectors[(row, target_col)] = eigenvectors[(row, src_col)];
            }
        }

        let mut singular_values = Vec::with_capacity(n_components);
        let mut sample_scores = Mat::zeros(n_samples, n_components);
        let mut inverse_singular = vec![0.0f64; n_components];
        for (component, &lambda) in eigenvalues.iter().enumerate() {
            let sigma = ((n_samples - 1) as f64 * lambda).sqrt();
            singular_values.push(sigma);
            let inv = if sigma > 0.0 { 1.0 / sigma } else { 0.0 };
            inverse_singular[component] = inv;
            for row in 0..n_samples {
                let value = sample_eigenvectors[(row, component)];
                sample_scores[(row, component)] = value * sigma;
            }
        }

        // Step 4: stream again to build loadings (variant space projection).
        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        let mut loadings = Mat::zeros(n_variants, n_components);
        let mut chunk_storage = vec![0.0f64; block_capacity * n_components];
        processed = 0;
        loop {
            let filled = source
                .next_block_into(block_capacity, &mut block_storage)
                .map_err(|e| HwePcaError::Source(Box::new(e)))?;
            if filled == 0 {
                break;
            }
            if processed + filled > n_variants {
                return Err(HwePcaError::InvalidInput(
                    "VariantBlockSource returned more variants than reported",
                ));
            }
            {
                let mut block = MatMut::from_column_major_slice_mut(
                    &mut block_storage[..n_samples * filled],
                    n_samples,
                    filled,
                );
                standardize_block(
                    &mut block,
                    processed..processed + filled,
                    &allele_frequencies,
                    &variant_scales,
                );
            }
            let block_ref = MatRef::from_column_major_slice(
                &block_storage[..n_samples * filled],
                n_samples,
                filled,
            );
            let mut chunk = MatMut::from_column_major_slice_mut(
                &mut chunk_storage[..filled * n_components],
                filled,
                n_components,
            );
            matmul(
                chunk.as_mut(),
                Accum::Replace,
                block_ref.transpose(),
                sample_eigenvectors.as_ref(),
                1.0,
                Par::Seq,
            );
            for local_col in 0..filled {
                let global_variant = processed + local_col;
                for component in 0..n_components {
                    let value = chunk[(local_col, component)] * inverse_singular[component];
                    loadings[(global_variant, component)] = value;
                }
            }
            processed += filled;
        }
        if processed != n_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early while computing loadings",
            ));
        }

        Ok(Self {
            n_samples,
            n_variants,
            allele_frequencies,
            variant_scales,
            eigenvalues,
            singular_values,
            sample_eigenvectors,
            sample_scores,
            loadings,
        })
    }

    /// Projects a block of genotype dosages onto the pre-fitted principal components.
    ///
    /// The input slice must be column-major with shape `(self.n_samples, block_len)`. Missing
    /// values should be encoded as `NaN` and are imputed to zero after standardization.
    pub fn transform_block(
        &self,
        mut block: MatMut<'_, f64>,
        variant_offset: usize,
    ) -> Result<Mat<f64>, HwePcaError> {
        let block_len = block.ncols();
        if block.nrows() != self.n_samples {
            return Err(HwePcaError::InvalidInput(
                "transform_block: sample dimension mismatch",
            ));
        }
        if variant_offset + block_len > self.n_variants {
            return Err(HwePcaError::InvalidInput(
                "transform_block: variant range exceeds training dimensions",
            ));
        }
        if block_len == 0 {
            return Ok(Mat::zeros(0, self.eigenvalues.len()));
        }
        standardize_block(
            &mut block,
            variant_offset..variant_offset + block_len,
            &self.allele_frequencies,
            &self.variant_scales,
        );
        let mut transformed = Mat::zeros(block_len, self.eigenvalues.len());
        matmul(
            transformed.as_mut(),
            Accum::Replace,
            block.transpose(),
            self.sample_eigenvectors.as_ref(),
            1.0,
            Par::Seq,
        );
        for row in 0..block_len {
            for component in 0..self.eigenvalues.len() {
                let sigma = self.singular_values[component];
                transformed[(row, component)] = if sigma > 0.0 {
                    transformed[(row, component)] / sigma
                } else {
                    0.0
                };
            }
        }
        Ok(transformed)
    }
}

fn standardize_block(
    block: &mut MatMut<'_, f64>,
    variant_range: Range<usize>,
    allele_frequencies: &[f64],
    variant_scales: &[f64],
) {
    for (local_col, variant_index) in variant_range.enumerate() {
        let mean = 2.0 * allele_frequencies[variant_index];
        let denom = variant_scales[variant_index].max(HWE_SCALE_FLOOR);
        for row in 0..block.nrows() {
            let raw = block[(row, local_col)];
            block[(row, local_col)] = if raw.is_finite() {
                (raw - mean) / denom
            } else {
                0.0
            };
        }
    }
}
