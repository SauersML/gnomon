use core::cmp::min;
use core::fmt;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Accum, Mat, MatMut, MatRef, Par, Side};
use std::convert::Infallible;
use std::error::Error;
use std::ops::Range;

const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
const HWE_SCALE_FLOOR: f64 = 1.0e-6;
const EIGENVALUE_EPSILON: f64 = 1.0e-9;
const DEFAULT_BLOCK_WIDTH: usize = 2_048;

pub trait VariantBlockSource {
    type Error;

    fn n_samples(&self) -> usize;
    fn n_variants(&self) -> usize;
    fn reset(&mut self) -> Result<(), Self::Error>;
    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error>;
}

pub struct DenseBlockSource<'a> {
    data: &'a [f64],
    dims: (usize, usize),
    cursor: usize,
}

impl<'a> DenseBlockSource<'a> {
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

#[derive(Debug)]
pub enum HwePcaError {
    InvalidInput(&'static str),
    Source(Box<dyn Error + Send + Sync + 'static>),
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

#[derive(Clone, Debug)]
pub struct HweScaler {
    frequencies: Vec<f64>,
    scales: Vec<f64>,
}

impl HweScaler {
    fn new(frequencies: Vec<f64>, scales: Vec<f64>) -> Self {
        Self { frequencies, scales }
    }

    pub fn allele_frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    pub fn variant_scales(&self) -> &[f64] {
        &self.scales
    }

    fn standardize_block(&self, block: &mut MatMut<'_, f64>, variant_range: Range<usize>) {
        for (local_col, variant_index) in variant_range.enumerate() {
            let mean = 2.0 * self.frequencies[variant_index];
            let denom = self.scales[variant_index].max(HWE_SCALE_FLOOR);
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
}

pub struct HwePcaModel {
    pub n_samples: usize,
    pub n_variants: usize,
    pub scaler: HweScaler,
    pub eigenvalues: Vec<f64>,
    pub singular_values: Vec<f64>,
    pub sample_basis: Mat<f64>,
    pub sample_scores: Mat<f64>,
    pub loadings: Mat<f64>,
}

impl HwePcaModel {
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

        let scaler = stream_allele_statistics(source, block_capacity, &mut block_storage)?;

        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        let gram = accumulate_gram_matrix(source, &scaler, block_capacity, &mut block_storage)?;

        let decomposition = compute_eigenpairs(gram)?;
        if decomposition.values.is_empty() {
            return Err(HwePcaError::Eigen(
                "All eigenvalues are numerically zero; increase cohort size or review input data"
                    .into(),
            ));
        }

        let (singular_values, sample_scores) = build_sample_scores(
            n_samples,
            &decomposition.values,
            decomposition.vectors.as_ref(),
        );

        source
            .reset()
            .map_err(|e| HwePcaError::Source(Box::new(e)))?;
        let loadings = compute_variant_loadings(
            source,
            &scaler,
            block_capacity,
            &mut block_storage,
            decomposition.vectors.as_ref(),
            &singular_values,
        )?;

        Ok(Self {
            n_samples,
            n_variants,
            scaler,
            eigenvalues: decomposition.values,
            singular_values,
            sample_basis: decomposition.vectors,
            sample_scores,
            loadings,
        })
    }

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

        self.scaler
            .standardize_block(&mut block, variant_offset..variant_offset + block_len);

        let mut transformed = Mat::zeros(block_len, self.eigenvalues.len());
        matmul(
            transformed.as_mut(),
            Accum::Replace,
            block.transpose(),
            self.sample_basis.as_ref(),
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

struct EigenDecomposition {
    values: Vec<f64>,
    vectors: Mat<f64>,
}

fn stream_allele_statistics<S>(
    source: &mut S,
    block_capacity: usize,
    block_storage: &mut [f64],
) -> Result<HweScaler, HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
{
    let n_samples = source.n_samples();
    let n_variants = source.n_variants();
    let mut allele_sums = vec![0.0f64; n_variants];
    let mut allele_counts = vec![0usize; n_variants];
    let mut processed = 0usize;

    source
        .reset()
        .map_err(|e| HwePcaError::Source(Box::new(e)))?;

    loop {
        let filled = source
            .next_block_into(block_capacity, block_storage)
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

    let mut frequencies = vec![0.0f64; n_variants];
    let mut scales = vec![1.0f64; n_variants];

    for (idx, (&sum, &calls)) in allele_sums.iter().zip(&allele_counts).enumerate() {
        if calls == 0 {
            frequencies[idx] = 0.0;
            scales[idx] = HWE_SCALE_FLOOR;
            continue;
        }
        let mean_genotype = sum / (calls as f64);
        let freq = (mean_genotype / 2.0).clamp(0.0, 1.0);
        let variance = (2.0 * freq * (1.0 - freq)).max(HWE_VARIANCE_EPSILON);
        frequencies[idx] = freq;
        let scale = variance.sqrt();
        scales[idx] = if scale < HWE_SCALE_FLOOR {
            HWE_SCALE_FLOOR
        } else {
            scale
        };
    }

    Ok(HweScaler::new(frequencies, scales))
}

fn accumulate_gram_matrix<S>(
    source: &mut S,
    scaler: &HweScaler,
    block_capacity: usize,
    block_storage: &mut [f64],
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
{
    let n_samples = source.n_samples();
    let n_variants = source.n_variants();
    let mut gram = Mat::zeros(n_samples, n_samples);
    let mut processed = 0usize;
    let scale = 1.0 / ((n_samples - 1) as f64);

    loop {
        let filled = source
            .next_block_into(block_capacity, block_storage)
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
        scaler.standardize_block(&mut block, processed..processed + filled);

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

    Ok(gram)
}

fn compute_eigenpairs(gram: Mat<f64>) -> Result<EigenDecomposition, HwePcaError> {
    let eig = SelfAdjointEigen::new(gram.as_ref(), Side::Upper)
        .map_err(|err| HwePcaError::Eigen(format!("{err:?}")))?;
    let mut ordering: Vec<(f64, usize)> = eig
        .S()
        .column_vector()
        .iter()
        .enumerate()
        .map(|(idx, value)| (*value, idx))
        .collect();

    ordering.sort_by(|(lhs, _), (rhs, _)| rhs.partial_cmp(lhs).unwrap_or(core::cmp::Ordering::Equal));

    let mut values = Vec::new();
    let mut selected = Vec::new();
    for (value, idx) in ordering.iter().copied() {
        if value <= EIGENVALUE_EPSILON {
            break;
        }
        values.push(value);
        selected.push(idx);
    }

    let mut vectors = Mat::zeros(eig.U().nrows(), selected.len());
    for (target_col, src_col) in selected.iter().copied().enumerate() {
        for row in 0..eig.U().nrows() {
            vectors[(row, target_col)] = eig.U()[(row, src_col)];
        }
    }

    Ok(EigenDecomposition { values, vectors })
}

fn build_sample_scores(
    n_samples: usize,
    eigenvalues: &[f64],
    sample_basis: MatRef<'_, f64>,
) -> (Vec<f64>, Mat<f64>) {
    let mut singular_values = Vec::with_capacity(eigenvalues.len());
    let mut sample_scores = Mat::zeros(n_samples, eigenvalues.len());

    for (component, &lambda) in eigenvalues.iter().enumerate() {
        let sigma = ((n_samples - 1) as f64 * lambda).sqrt();
        singular_values.push(sigma);
        for row in 0..n_samples {
            sample_scores[(row, component)] = sample_basis[(row, component)] * sigma;
        }
    }

    (singular_values, sample_scores)
}

fn compute_variant_loadings<S>(
    source: &mut S,
    scaler: &HweScaler,
    block_capacity: usize,
    block_storage: &mut [f64],
    sample_basis: MatRef<'_, f64>,
    singular_values: &[f64],
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
{
    let n_samples = source.n_samples();
    let n_variants = source.n_variants();
    let n_components = singular_values.len();
    let mut loadings = Mat::zeros(n_variants, n_components);
    let mut processed = 0usize;
    let mut chunk_storage = vec![0.0f64; block_capacity * n_components];
    let inverse_singular: Vec<f64> = singular_values
        .iter()
        .map(|&sigma| if sigma > 0.0 { 1.0 / sigma } else { 0.0 })
        .collect();

    loop {
        let filled = source
            .next_block_into(block_capacity, block_storage)
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
        scaler.standardize_block(&mut block, processed..processed + filled);

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
            sample_basis,
            1.0,
            Par::Seq,
        );

        for local_col in 0..filled {
            let global_variant = processed + local_col;
            for component in 0..n_components {
                loadings[(global_variant, component)] =
                    chunk[(local_col, component)] * inverse_singular[component];
            }
        }

        processed += filled;
    }

    if processed != n_variants {
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource terminated early while computing loadings",
        ));
    }

    Ok(loadings)
}
