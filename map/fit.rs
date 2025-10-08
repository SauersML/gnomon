use core::cmp::min;
use core::fmt;
use core::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::{Accum, ColMut, Mat, MatMut, MatRef, Side};
use faer::{Unbind, unzip, zip};
use rayon::prelude::*;
use serde::de::Error as DeError;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::convert::Infallible;
use std::error::Error;
use std::ops::Range;

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
pub const DEFAULT_BLOCK_WIDTH: usize = 2_048;

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
            HwePcaError::Eigen(msg) => f.write_str(msg),
        }
    }
}

impl Error for HwePcaError {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HweScaler {
    frequencies: Vec<f64>,
    scales: Vec<f64>,
}

impl HweScaler {
    fn new(frequencies: Vec<f64>, scales: Vec<f64>) -> Self {
        Self {
            frequencies,
            scales,
        }
    }

    pub fn allele_frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    pub fn variant_scales(&self) -> &[f64] {
        &self.scales
    }

    pub(crate) fn standardize_block(
        &self,
        mut block: MatMut<'_, f64>,
        variant_range: Range<usize>,
    ) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let scales = &self.scales[start..end];
        let filled = freqs.len();

        debug_assert_eq!(filled, block.ncols());
        let mut block = block.subcols_mut(0, filled);

        let apply_standardization = |mut column: ColMut<'_, f64>, mean: f64, inv: f64| {
            if inv == 0.0 {
                column.fill(0.0);
                return;
            }

            let mut contiguous = column
                .try_as_col_major_mut()
                .expect("projection block column must be contiguous");
            let values = contiguous.as_slice_mut();
            standardize_column_simd(values, mean, inv);
        };

        if filled >= 32 && rayon::current_num_threads() > 1 {
            block
                .par_col_iter_mut()
                .enumerate()
                .for_each(|(idx, column)| {
                    let freq = freqs[idx];
                    let scale = scales[idx];
                    let mean = 2.0 * freq;
                    let denom = scale.max(HWE_SCALE_FLOOR);
                    let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                    apply_standardization(column, mean, inv);
                });
        } else {
            for (idx, column) in block.col_iter_mut().enumerate() {
                let freq = freqs[idx];
                let scale = scales[idx];
                let mean = 2.0 * freq;
                let denom = scale.max(HWE_SCALE_FLOOR);
                let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                apply_standardization(column, mean, inv);
            }
        }
    }
}

#[cfg(target_feature = "avx512f")]
const STANDARDIZATION_SIMD_LANES: usize = 8;

#[cfg(all(
    not(target_feature = "avx512f"),
    any(
        target_feature = "avx",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )
))]
const STANDARDIZATION_SIMD_LANES: usize = 4;

#[cfg(all(
    not(target_feature = "avx512f"),
    not(any(
        target_feature = "avx",
        target_arch = "aarch64",
        target_arch = "wasm32"
    ))
))]
const STANDARDIZATION_SIMD_LANES: usize = 2;

#[inline(always)]
fn standardize_column_simd(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_impl::<STANDARDIZATION_SIMD_LANES>(values, mean, inv);
}

#[inline(always)]
fn standardize_column_simd_impl<const LANES: usize>(values: &mut [f64], mean: f64, inv: f64)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mean_simd = Simd::<f64, LANES>::splat(mean);
    let inv_simd = Simd::<f64, LANES>::splat(inv);
    let zero = Simd::<f64, LANES>::splat(0.0);

    let (chunks, remainder) = values.as_chunks_mut::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_slice(chunk);
        let mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let result = mask.select(standardized, zero);
        result.write_to_slice(chunk);
    }

    for value in remainder {
        let raw = *value;
        *value = if raw.is_finite() {
            (raw - mean) * inv
        } else {
            0.0
        };
    }
}

#[derive(Clone, Debug)]
pub struct HwePcaModel {
    n_samples: usize,
    n_variants: usize,
    scaler: HweScaler,
    eigenvalues: Vec<f64>,
    singular_values: Vec<f64>,
    sample_basis: Mat<f64>,
    sample_scores: Mat<f64>,
    loadings: Mat<f64>,
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
        let decomposition = decompose_gram_matrix(gram)?;
        if decomposition.values.is_empty() {
            return Err(HwePcaError::Eigen(
                "All eigenvalues are numerically zero; increase cohort size or review input data"
                    .into(),
            ));
        }

        let (singular_values, sample_scores) = build_sample_scores(n_samples, &decomposition);

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

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn n_variants(&self) -> usize {
        self.n_variants
    }

    pub fn scaler(&self) -> &HweScaler {
        &self.scaler
    }

    pub fn components(&self) -> usize {
        self.eigenvalues.len()
    }

    pub fn explained_variance(&self) -> &[f64] {
        &self.eigenvalues
    }

    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        let total: f64 = self.eigenvalues.iter().copied().sum();
        if total > 0.0 {
            self.eigenvalues
                .iter()
                .map(|&lambda| lambda / total)
                .collect()
        } else {
            vec![0.0; self.eigenvalues.len()]
        }
    }

    pub fn sample_basis(&self) -> MatRef<'_, f64> {
        self.sample_basis.as_ref()
    }

    pub fn sample_scores(&self) -> MatRef<'_, f64> {
        self.sample_scores.as_ref()
    }

    pub fn variant_loadings(&self) -> MatRef<'_, f64> {
        self.loadings.as_ref()
    }
}

struct Eigenpairs {
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

        let block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        for (local_col, column) in block.as_ref().col_iter().enumerate() {
            let variant_index = processed + local_col;
            let mut sum = 0.0f64;
            let mut calls = 0usize;
            zip!(column).for_each(|unzip!(value)| {
                let raw = *value;
                if raw.is_finite() {
                    sum += raw;
                    calls += 1;
                }
            });
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
        scaler.standardize_block(block.as_mut(), processed..processed + filled);

        let block_ref = block.into_const();

        matmul(
            gram.as_mut(),
            Accum::Add,
            block_ref,
            block_ref.transpose(),
            scale,
            faer::get_global_parallelism(),
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

fn decompose_gram_matrix(gram: Mat<f64>) -> Result<Eigenpairs, HwePcaError> {
    let eig = SelfAdjointEigen::new(gram.as_ref(), Side::Upper)
        .map_err(|err| HwePcaError::Eigen(format!("{err:?}")))?;

    let mut ordering: Vec<(usize, f64)> = eig
        .S()
        .column_vector()
        .iter()
        .copied()
        .enumerate()
        .map(|(idx, value)| (idx, value))
        .collect();
    ordering
        .sort_by(|(_, lhs), (_, rhs)| rhs.partial_cmp(lhs).unwrap_or(core::cmp::Ordering::Equal));

    let mut values = Vec::new();
    let mut selected = Vec::new();
    for (idx, value) in ordering.iter().copied() {
        if value <= EIGENVALUE_EPSILON {
            break;
        }
        values.push(value);
        selected.push(idx);
    }

    let vectors = Mat::from_fn(eig.U().nrows(), selected.len(), |row_idx, col_idx| {
        let row = row_idx.unbound();
        let component = col_idx.unbound();
        let src_col = selected[component];
        eig.U()[(row, src_col)]
    });

    Ok(Eigenpairs { values, vectors })
}

fn build_sample_scores(n_samples: usize, decomposition: &Eigenpairs) -> (Vec<f64>, Mat<f64>) {
    let mut singular_values = Vec::with_capacity(decomposition.values.len());
    let mut sample_scores = decomposition.vectors.clone();

    for (&lambda, mut column) in decomposition
        .values
        .iter()
        .zip(sample_scores.col_iter_mut())
    {
        let sigma = ((n_samples - 1) as f64 * lambda).sqrt();
        singular_values.push(sigma);
        zip!(&mut column).for_each(|unzip!(value)| {
            *value *= sigma;
        });
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
        scaler.standardize_block(block.as_mut(), processed..processed + filled);

        let block_ref = block.into_const();

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
            faer::get_global_parallelism(),
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

#[derive(Serialize, Deserialize)]
struct MatrixData {
    nrows: usize,
    ncols: usize,
    data: Vec<f64>,
}

impl MatrixData {
    fn from_mat(mat: MatRef<'_, f64>) -> Self {
        let mut data = Vec::with_capacity(mat.nrows() * mat.ncols());
        for col in 0..mat.ncols() {
            for row in 0..mat.nrows() {
                data.push(mat[(row, col)]);
            }
        }
        Self {
            nrows: mat.nrows(),
            ncols: mat.ncols(),
            data,
        }
    }

    fn into_mat(self) -> Result<Mat<f64>, String> {
        let MatrixData { nrows, ncols, data } = self;
        if data.len() != nrows * ncols {
            return Err("matrix data length does not match dimensions".into());
        }
        let mut mat = Mat::zeros(nrows, ncols);
        for col in 0..ncols {
            for row in 0..nrows {
                mat[(row, col)] = data[col * nrows + row];
            }
        }
        Ok(mat)
    }
}

impl Serialize for HwePcaModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("HwePcaModel", 9)?;
        state.serialize_field("n_samples", &self.n_samples)?;
        state.serialize_field("n_variants", &self.n_variants)?;
        state.serialize_field("scaler", &self.scaler)?;
        state.serialize_field("eigenvalues", &self.eigenvalues)?;
        state.serialize_field("singular_values", &self.singular_values)?;
        state.serialize_field(
            "sample_basis",
            &MatrixData::from_mat(self.sample_basis.as_ref()),
        )?;
        state.serialize_field(
            "sample_scores",
            &MatrixData::from_mat(self.sample_scores.as_ref()),
        )?;
        state.serialize_field("loadings", &MatrixData::from_mat(self.loadings.as_ref()))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HwePcaModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ModelData {
            n_samples: usize,
            n_variants: usize,
            scaler: HweScaler,
            eigenvalues: Vec<f64>,
            singular_values: Vec<f64>,
            sample_basis: MatrixData,
            sample_scores: MatrixData,
            loadings: MatrixData,
        }

        let raw = ModelData::deserialize(deserializer)?;
        Ok(HwePcaModel {
            n_samples: raw.n_samples,
            n_variants: raw.n_variants,
            scaler: raw.scaler,
            eigenvalues: raw.eigenvalues,
            singular_values: raw.singular_values,
            sample_basis: raw.sample_basis.into_mat().map_err(DeError::custom)?,
            sample_scores: raw.sample_scores.into_mat().map_err(DeError::custom)?,
            loadings: raw.loadings.into_mat().map_err(DeError::custom)?,
        })
    }
}
