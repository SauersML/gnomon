use super::progress::{
    FitProgressObserver, FitProgressStage, NoopFitProgress, StageProgressHandle,
};
use super::variant_filter::VariantKey;
use core::cmp::{Ordering, min};
use core::fmt;
use core::marker::PhantomData;
use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::col::Col;
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::SelfAdjointEigen;
use faer::linalg::{temp_mat_scratch, temp_mat_uninit};
use faer::mat::AsMatMut;
use faer::matrix_free::LinOp;
use faer::matrix_free::eigen::{
    PartialEigenInfo, PartialEigenParams, partial_eigen_scratch, partial_self_adjoint_eigen,
};
use faer::prelude::ReborrowMut;
use faer::{
    Accum, ColMut, Mat, MatMut, MatRef, Par, Side, get_global_parallelism, set_global_parallelism,
    unzip, zip,
};
use rayon::prelude::*;
use serde::de::Error as DeError;
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::cell::UnsafeCell;
use std::convert::Infallible;
use std::error::Error;
use std::mem::ManuallyDrop;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr;
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SupportedLaneCount};
use std::sync::Arc;

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
pub const DEFAULT_BLOCK_WIDTH: usize = 2_048;
const DENSE_EIGEN_FALLBACK_THRESHOLD: usize = 64;
const INITIAL_PARTIAL_COMPONENTS: usize = 32;
const MAX_PARTIAL_COMPONENTS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CovarianceComputationMode {
    Dense,
    Partial,
}

fn covariance_computation_mode(n: usize) -> CovarianceComputationMode {
    if n <= DENSE_EIGEN_FALLBACK_THRESHOLD {
        return CovarianceComputationMode::Dense;
    }

    let max_rank = n.saturating_sub(1);
    if max_rank == 0 {
        return CovarianceComputationMode::Dense;
    }

    let max_partial = ((n - 1) / 2).min(MAX_PARTIAL_COMPONENTS);
    if max_partial == 0 {
        return CovarianceComputationMode::Dense;
    }

    CovarianceComputationMode::Partial
}

struct ParallelismGuard {
    previous: Par,
}

impl ParallelismGuard {
    fn new() -> Self {
        let previous = get_global_parallelism();
        let desired = Par::rayon(rayon::current_num_threads());
        set_global_parallelism(desired);
        Self { previous }
    }

    fn active_parallelism(&self) -> Par {
        get_global_parallelism()
    }
}

impl Drop for ParallelismGuard {
    fn drop(&mut self) {
        set_global_parallelism(self.previous);
    }
}

#[derive(Debug)]
struct OperatorError;

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
        block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        par: Par,
    ) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let scales = &self.scales[start..end];
        standardize_block_impl(block, freqs, scales, par);
    }

    pub(crate) fn standardize_block_with_mask(
        &self,
        block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        presence_out: MatMut<'_, f64>,
        par: Par,
    ) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let filled = freqs.len();

        debug_assert_eq!(filled, block.ncols());
        debug_assert_eq!(filled, presence_out.ncols());
        debug_assert_eq!(block.nrows(), presence_out.nrows());

        let block_ref = block.as_ref();

        for (presence_col, column) in presence_out
            .subcols_mut(0, filled)
            .col_iter_mut()
            .zip(block_ref.subcols(0, filled).col_iter())
        {
            zip!(presence_col, column).for_each(|unzip!(presence, raw)| {
                *presence = if raw.is_finite() { 1.0 } else { 0.0 };
            });
        }

        self.standardize_block(block, variant_range, par);
    }
}

fn standardize_block_impl(block: MatMut<'_, f64>, freqs: &[f64], scales: &[f64], par: Par) {
    let filled = freqs.len();

    debug_assert_eq!(filled, block.ncols());
    debug_assert_eq!(filled, scales.len());
    let block = block.subcols_mut(0, filled);

    let apply_standardization = |mut column: ColMut<'_, f64>, mean: f64, inv: f64| {
        if inv == 0.0 {
            column.fill(0.0);
            return;
        }

        let contiguous = column
            .try_as_col_major_mut()
            .expect("projection block column must be contiguous");
        let values = contiguous.as_slice_mut();
        standardize_column_simd(values, mean, inv);
    };

    let use_parallel = filled >= 32 && par.degree() > 1;

    if use_parallel {
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

struct VariantStatsCache {
    frequencies: Vec<f64>,
    scales: Vec<f64>,
    block_sums: Vec<f64>,
    block_calls: Vec<usize>,
    computed: bool,
}

impl VariantStatsCache {
    fn new(n_variants: usize, block_capacity: usize) -> Self {
        Self {
            frequencies: vec![0.0; n_variants],
            scales: vec![HWE_SCALE_FLOOR; n_variants],
            block_sums: vec![0.0; block_capacity],
            block_calls: vec![0usize; block_capacity],
            computed: false,
        }
    }

    fn is_computed(&self) -> bool {
        self.computed
    }

    fn ensure_statistics(&mut self, block: MatRef<'_, f64>, variant_range: Range<usize>, par: Par) {
        if self.computed {
            return;
        }

        let filled = block.ncols();
        let sums_slice = &mut self.block_sums[..filled];
        let calls_slice = &mut self.block_calls[..filled];

        let use_parallel = filled >= 32 && par.degree() > 1;

        if use_parallel {
            sums_slice
                .par_iter_mut()
                .zip(calls_slice.par_iter_mut())
                .zip(block.par_col_iter())
                .for_each(|((sum_slot, calls_slot), column)| {
                    let contiguous = column
                        .try_as_col_major()
                        .expect("variant block column must be contiguous");
                    let (sum, calls) = sum_and_count_finite(contiguous.as_slice());
                    *sum_slot = sum;
                    *calls_slot = calls;
                });
        } else {
            sums_slice
                .iter_mut()
                .zip(calls_slice.iter_mut())
                .zip(block.col_iter())
                .for_each(|((sum_slot, calls_slot), column)| {
                    let contiguous = column
                        .try_as_col_major()
                        .expect("variant block column must be contiguous");
                    let (sum, calls) = sum_and_count_finite(contiguous.as_slice());
                    *sum_slot = sum;
                    *calls_slot = calls;
                });
        }

        for (local_col, (&sum, &calls)) in sums_slice.iter().zip(calls_slice.iter()).enumerate() {
            let variant_index = variant_range.start + local_col;
            if calls == 0 {
                self.frequencies[variant_index] = 0.0;
                self.scales[variant_index] = HWE_SCALE_FLOOR;
                continue;
            }

            let mean_genotype = sum / (calls as f64);
            let allele_freq = (mean_genotype / 2.0).clamp(0.0, 1.0);
            let variance = (2.0 * allele_freq * (1.0 - allele_freq)).max(HWE_VARIANCE_EPSILON);
            self.frequencies[variant_index] = allele_freq;
            let derived_scale = variance.sqrt();
            self.scales[variant_index] = if derived_scale < HWE_SCALE_FLOOR {
                HWE_SCALE_FLOOR
            } else {
                derived_scale
            };
        }
    }

    fn standardize_block(&self, block: MatMut<'_, f64>, variant_range: Range<usize>, par: Par) {
        let start = variant_range.start;
        let end = variant_range.end;
        let freqs = &self.frequencies[start..end];
        let scales = &self.scales[start..end];
        standardize_block_impl(block, freqs, scales, par);
    }

    fn mark_computed(&mut self) {
        self.computed = true;
    }

    fn into_scaler(self) -> Option<HweScaler> {
        if self.computed {
            Some(HweScaler::new(self.frequencies, self.scales))
        } else {
            None
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
        *chunk = result.to_array();
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

#[inline(always)]
fn sum_and_count_finite(values: &[f64]) -> (f64, usize) {
    sum_and_count_finite_impl::<STANDARDIZATION_SIMD_LANES>(values)
}

#[inline(always)]
fn sum_and_count_finite_impl<const LANES: usize>(values: &[f64]) -> (f64, usize)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    let zero = Simd::<f64, LANES>::splat(0.0);

    let (chunks, remainder) = values.as_chunks::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_slice(chunk);
        let mask = lane.is_finite();
        let finite = mask.select(lane, zero);
        sum += finite.reduce_sum();
        count += mask
            .to_array()
            .into_iter()
            .map(|flag| flag as usize)
            .sum::<usize>();
    }

    for &value in remainder {
        if value.is_finite() {
            sum += value;
            count += 1;
        }
    }

    (sum, count)
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
    variant_keys: Option<Vec<VariantKey>>,
}

impl HwePcaModel {
    pub fn fit<S>(source: &mut S) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = Arc::new(NoopFitProgress::default());
        Self::fit_with_progress(source, &progress)
    }

    pub fn fit_with_progress<S, P>(source: &mut S, progress: &Arc<P>) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
        P: FitProgressObserver + 'static,
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

        let parallelism_guard = ParallelismGuard::new();
        let par = parallelism_guard.active_parallelism();
        let block_capacity = min(DEFAULT_BLOCK_WIDTH.max(1), n_variants);
        let block_storage = vec![0.0f64; n_samples * block_capacity];

        progress.on_stage_start(FitProgressStage::AlleleStatistics, n_variants);
        let stats_progress =
            StageProgressHandle::new(Arc::clone(progress), FitProgressStage::AlleleStatistics);

        let operator = StandardizedCovarianceOp::new(
            source,
            block_capacity,
            block_storage,
            Some(stats_progress),
        );

        let gram_mode = covariance_computation_mode(n_samples);
        let gram_progress_handle = match gram_mode {
            CovarianceComputationMode::Dense => {
                progress.on_stage_start(FitProgressStage::GramMatrix, n_variants);
                Some(StageProgressHandle::new(
                    Arc::clone(progress),
                    FitProgressStage::GramMatrix,
                ))
            }
            CovarianceComputationMode::Partial => {
                progress.on_stage_start(FitProgressStage::GramMatrix, 0);
                progress.on_stage_estimate(FitProgressStage::GramMatrix, n_variants);
                None
            }
        };

        let decomposition_result =
            compute_covariance_eigenpairs(&operator, par, gram_mode, gram_progress_handle.as_ref());
        let (source, mut block_storage, scaler_opt) = operator.into_parts();
        let decomposition = decomposition_result?;
        progress.on_stage_finish(FitProgressStage::GramMatrix);

        let scaler = scaler_opt.expect("covariance operator statistics missing after execution");

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
            progress,
            par,
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
            variant_keys: None,
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

    pub fn set_variant_keys(&mut self, keys: Option<Vec<VariantKey>>) {
        self.variant_keys = keys;
    }

    pub fn variant_keys(&self) -> Option<&[VariantKey]> {
        self.variant_keys.as_deref()
    }
}

struct Eigenpairs {
    values: Vec<f64>,
    vectors: Mat<f64>,
}

struct StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    source: *mut S,
    block_storage: UnsafeCell<Option<Vec<f64>>>,
    n_samples: usize,
    n_variants: usize,
    block_capacity: usize,
    scale: f64,
    stats: UnsafeCell<VariantStatsCache>,
    stats_progress: UnsafeCell<Option<StageProgressHandle<P>>>,
    error: UnsafeCell<Option<HwePcaError>>,
    marker: PhantomData<&'a mut S>,
}

impl<'a, S, P> StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    fn new(
        source: &'a mut S,
        block_capacity: usize,
        block_storage: Vec<f64>,
        stats_progress: Option<StageProgressHandle<P>>,
    ) -> Self {
        let n_samples = source.n_samples();
        let n_variants = source.n_variants();
        assert_eq!(
            block_storage.len(),
            n_samples * block_capacity,
            "block storage size must match block capacity",
        );
        let scale = 1.0 / ((n_samples - 1) as f64);
        Self {
            source: source as *mut S,
            block_storage: UnsafeCell::new(Some(block_storage)),
            n_samples,
            n_variants,
            block_capacity,
            scale,
            stats: UnsafeCell::new(VariantStatsCache::new(n_variants, block_capacity)),
            stats_progress: UnsafeCell::new(stats_progress),
            error: UnsafeCell::new(None),
            marker: PhantomData,
        }
    }

    fn into_parts(self) -> (&'a mut S, Vec<f64>, Option<HweScaler>) {
        let mut this = ManuallyDrop::new(self);
        let source = this.source;
        let storage = unsafe {
            (*this.block_storage.get())
                .take()
                .expect("block storage already taken")
        };
        let stats = unsafe { ptr::read(this.stats.get()) };
        let scaler = stats.into_scaler();
        if let Some(handle) = unsafe { (*this.stats_progress.get()).take() } {
            handle.finish();
        }
        let error = unsafe { ptr::read(this.error.get()) };
        drop(error);
        (unsafe { &mut *source }, storage, scaler)
    }

    fn take_error(&self) -> Option<HwePcaError> {
        unsafe { (*self.error.get()).take() }
    }

    fn fail_source(&self, err: S::Error) -> ! {
        self.record_error(HwePcaError::Source(Box::new(err)))
    }

    fn fail_invalid(&self, msg: &'static str) -> ! {
        self.record_error(HwePcaError::InvalidInput(msg))
    }

    fn record_error(&self, err: HwePcaError) -> ! {
        let slot = unsafe { &mut *self.error.get() };
        if slot.is_none() {
            *slot = Some(err);
        }
        std::panic::panic_any(OperatorError);
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn stats_computed(&self) -> bool {
        unsafe { (*self.stats.get()).is_computed() }
    }

    fn standardize_block_in_place(
        &self,
        block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        par: Par,
    ) {
        let stats = unsafe { &mut *self.stats.get() };
        stats.ensure_statistics(block.as_ref(), variant_range.clone(), par);
        stats.standardize_block(block, variant_range, par);
    }

    fn mark_stats_computed(&self) {
        let stats = unsafe { &mut *self.stats.get() };
        stats.mark_computed();
        let slot = unsafe { &mut *self.stats_progress.get() };
        if let Some(handle) = slot.take() {
            handle.finish();
        }
    }
}

impl<'a, S, P> fmt::Debug for StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StandardizedCovarianceOp")
            .field("n_samples", &self.n_samples)
            .field("n_variants", &self.n_variants)
            .field("block_capacity", &self.block_capacity)
            .finish()
    }
}

unsafe impl<'a, S, P> Sync for StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
}

impl<'a, S, P> Drop for StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    fn drop(&mut self) {
        let slot = unsafe { &mut *self.stats_progress.get() };
        if let Some(handle) = slot.take() {
            handle.finish();
        }
    }
}

impl<'a, S, P> LinOp<f64> for StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        match par {
            Par::Seq => {}
            _ => {}
        }
        temp_mat_scratch::<f64>(self.block_capacity, rhs_ncols)
    }

    fn nrows(&self) -> usize {
        self.n_samples
    }

    fn ncols(&self) -> usize {
        self.n_samples
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        match par {
            Par::Seq => {}
            _ => {}
        }
        debug_assert_eq!(out.nrows(), self.n_samples);
        debug_assert_eq!(rhs.nrows(), self.n_samples);

        out.fill(0.0);

        if rhs.ncols() == 0 {
            return;
        }

        let block_storage_opt = unsafe { &mut *self.block_storage.get() };
        let block_storage = block_storage_opt
            .as_mut()
            .expect("block storage missing during operator application");

        let source = unsafe { &mut *self.source };

        if let Err(err) = source.reset() {
            self.fail_source(err);
        }

        let (mut proj_uninit, _) =
            unsafe { temp_mat_uninit::<f64, _, _>(self.block_capacity, rhs.ncols(), stack) };
        let mut proj_storage = proj_uninit.as_mat_mut();

        let mut processed = 0usize;

        let stats_ready = self.stats_computed();

        loop {
            let filled = match source.next_block_into(self.block_capacity, &mut block_storage[..]) {
                Ok(filled) => filled,
                Err(err) => self.fail_source(err),
            };

            if filled == 0 {
                break;
            }

            if processed + filled > self.n_variants {
                self.fail_invalid("VariantBlockSource returned more variants than reported");
            }

            let block_slice = &mut block_storage[..self.n_samples * filled];
            let mut block =
                MatMut::from_column_major_slice_mut(block_slice, self.n_samples, filled);
            let variant_range = processed..processed + filled;
            self.standardize_block_in_place(block.rb_mut(), variant_range.clone(), par);

            let mut proj_block = proj_storage.rb_mut().subrows_mut(0, filled);

            matmul(
                proj_block.as_mut(),
                Accum::Replace,
                block.as_ref().transpose(),
                rhs,
                1.0,
                par,
            );

            matmul(
                out.rb_mut(),
                Accum::Add,
                block.as_ref(),
                proj_block.as_ref(),
                self.scale,
                par,
            );

            processed += filled;

            if !stats_ready {
                if let Some(handle) = unsafe { &*self.stats_progress.get() }.as_ref() {
                    handle.advance(processed);
                }
            }
        }

        if processed != self.n_variants {
            self.fail_invalid("VariantBlockSource terminated early during covariance accumulation");
        }

        if !stats_ready {
            self.mark_stats_computed();
        }
    }

    fn conj_apply(
        &self,
        out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        self.apply(out, rhs, par, stack);
    }
}

fn compute_covariance_eigenpairs<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    par: Par,
    mode: CovarianceComputationMode,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    let n = operator.n_samples();
    if n == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(0, 0),
        });
    }

    if matches!(mode, CovarianceComputationMode::Dense) {
        return compute_covariance_eigenpairs_dense(operator, par, progress);
    }

    let max_rank = n.saturating_sub(1);
    if max_rank == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    let mut max_partial = ((n - 1) / 2).min(MAX_PARTIAL_COMPONENTS);
    if max_partial == 0 {
        return compute_covariance_eigenpairs_dense(operator, par, progress);
    }
    max_partial = max_partial.min(max_rank);

    let mut target = INITIAL_PARTIAL_COMPONENTS.min(max_partial);
    if target == 0 {
        target = max_partial;
    }

    let normalization = (n as f64).sqrt();
    let v0 = Col::from_fn(n, |_| 1.0 / normalization);

    loop {
        let params = partial_solver_params(n, target);
        let (info, eigvals, eigvecs) = run_partial_eigensolver(operator, target, par, &v0, params)?;

        let n_converged = info.n_converged_eigen.min(target);
        let mut ordering = Vec::with_capacity(n_converged);
        for idx in 0..n_converged {
            ordering.push((idx, eigvals[idx]));
        }

        ordering.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

        let mut positive = 0usize;
        for &(_, value) in &ordering {
            if value <= EIGENVALUE_EPSILON {
                break;
            }
            positive += 1;
        }

        if positive == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        if positive == target && target < max_partial {
            let next_target = (target * 2).min(max_partial);
            if next_target > target {
                target = next_target;
                continue;
            }
        }

        let mut values = Vec::with_capacity(positive);
        let mut vectors = Mat::zeros(n, positive);
        for (out_idx, (src_idx, value)) in ordering.into_iter().take(positive).enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = eigvecs[(row, src_idx)];
            }
        }

        return Ok(Eigenpairs { values, vectors });
    }
}

fn compute_covariance_eigenpairs_dense<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    par: Par,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    let mut covariance = accumulate_covariance_matrix(operator, par, progress)?;
    let n = covariance.nrows();

    for col in 0..n {
        for row in 0..col {
            let avg = 0.5 * (covariance[(row, col)] + covariance[(col, row)]);
            covariance[(row, col)] = avg;
            covariance[(col, row)] = avg;
        }
    }

    let eig = SelfAdjointEigen::new(covariance.as_ref(), Side::Lower)
        .map_err(|err| HwePcaError::Eigen(format!("dense eigendecomposition failed: {err:?}")))?;

    let diag = eig.S();
    let basis = eig.U();

    let mut ordering = Vec::with_capacity(n);
    for idx in 0..n {
        ordering.push((idx, diag[idx]));
    }

    ordering.sort_by(|lhs, rhs| rhs.1.partial_cmp(&lhs.1).unwrap_or(Ordering::Equal));

    let mut positive = 0usize;
    for &(_, value) in &ordering {
        if value <= EIGENVALUE_EPSILON {
            break;
        }
        positive += 1;
    }

    if positive == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    let mut values = Vec::with_capacity(positive);
    let mut vectors = Mat::zeros(n, positive);
    for (out_idx, (src_idx, value)) in ordering.into_iter().take(positive).enumerate() {
        values.push(value);
        for row in 0..n {
            vectors[(row, out_idx)] = basis[(row, src_idx)];
        }
    }

    Ok(Eigenpairs { values, vectors })
}

fn accumulate_covariance_matrix<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    par: Par,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    let n_samples = operator.n_samples;
    let n_variants = operator.n_variants;
    let mut covariance = Mat::zeros(n_samples, n_samples);

    if n_samples == 0 || n_variants == 0 {
        return Ok(covariance);
    }

    let block_capacity = operator.block_capacity;

    let block_storage_opt = unsafe { &mut *operator.block_storage.get() };
    let block_storage = block_storage_opt
        .as_mut()
        .expect("block storage missing during covariance accumulation");

    let source = unsafe { &mut *operator.source };
    source
        .reset()
        .map_err(|err| HwePcaError::Source(Box::new(err)))?;

    let mut processed = 0usize;

    loop {
        let filled = source
            .next_block_into(block_capacity, &mut block_storage[..])
            .map_err(|err| HwePcaError::Source(Box::new(err)))?;

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

        let variant_range = processed..processed + filled;
        operator.standardize_block_in_place(block.rb_mut(), variant_range.clone(), par);

        matmul(
            covariance.as_mut(),
            Accum::Add,
            block.as_ref(),
            block.as_ref().transpose(),
            operator.scale,
            par,
        );

        processed += filled;

        if let Some(handle) = progress {
            handle.advance(processed);
        }
    }

    if processed != n_variants {
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource terminated early during covariance accumulation",
        ));
    }

    if !operator.stats_computed() {
        operator.mark_stats_computed();
    }

    Ok(covariance)
}

fn partial_solver_params(n: usize, target: usize) -> PartialEigenParams {
    let mut params = PartialEigenParams::default();
    let max_available = n.saturating_sub(1);
    let min_dim = ((2 * target).max(64)).min(max_available);
    let mut max_dim = ((4 * target).max(128)).min(max_available);
    if max_dim <= min_dim {
        max_dim = min_dim.min(max_available);
    }
    params.min_dim = min_dim.max(target);
    params.max_dim = max_dim.max(params.min_dim);
    params.max_restarts = 2048;
    params
}

fn run_partial_eigensolver<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    target: usize,
    par: Par,
    v0: &Col<f64>,
    params: PartialEigenParams,
) -> Result<(PartialEigenInfo, Vec<f64>, Mat<f64>), HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    let n = operator.n_samples();
    let mut eigvecs = Mat::zeros(n, target);
    let mut eigvals = vec![0.0f64; target];

    let scratch = partial_eigen_scratch(operator, target, par, params);
    let mut mem = MemBuffer::new(scratch);

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut stack = MemStack::new(&mut mem);
        partial_self_adjoint_eigen(
            eigvecs.as_mut(),
            &mut eigvals,
            operator,
            v0.as_ref(),
            f64::EPSILON * 128.0,
            par,
            &mut stack,
            params,
        )
    }));

    let info = match result {
        Ok(info) => info,
        Err(payload) => {
            if payload.downcast_ref::<OperatorError>().is_some() {
                let err = operator.take_error().unwrap_or_else(|| {
                    HwePcaError::Eigen(
                        "matrix-free eigensolver aborted with an internal error".into(),
                    )
                });
                return Err(err);
            }
            std::panic::resume_unwind(payload);
        }
    };

    if let Some(err) = operator.take_error() {
        return Err(err);
    }

    Ok((info, eigvals, eigvecs))
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

fn compute_variant_loadings<S, P>(
    source: &mut S,
    scaler: &HweScaler,
    block_capacity: usize,
    block_storage: &mut [f64],
    sample_basis: MatRef<'_, f64>,
    singular_values: &[f64],
    progress: &Arc<P>,
    par: Par,
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver,
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

    progress.on_stage_start(FitProgressStage::Loadings, n_variants);

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
        scaler.standardize_block(block.as_mut(), processed..processed + filled, par);

        let block_ref = block.as_ref();

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
            par,
        );

        {
            let chunk_view = chunk.as_mut();
            for (column, &inv_sigma) in chunk_view.col_iter_mut().zip(inverse_singular.iter()) {
                zip!(column).for_each(|unzip!(value)| {
                    *value *= inv_sigma;
                });
            }
        }

        loadings
            .submatrix_mut(processed, 0, filled, n_components)
            .copy_from(chunk.as_ref());

        processed += filled;
        progress.on_stage_advance(FitProgressStage::Loadings, processed);
    }

    if processed != n_variants {
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource terminated early while computing loadings",
        ));
    }

    progress.on_stage_finish(FitProgressStage::Loadings);

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
        state.serialize_field("variant_keys", &self.variant_keys)?;
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
            #[serde(default)]
            variant_keys: Option<Vec<VariantKey>>,
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
            variant_keys: raw.variant_keys,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::io::GenotypeDataset;
    use std::path::Path;

    const TEST_VCF_URL: &str = "https://raw.githubusercontent.com/SauersML/genomic_pca/refs/heads/main/tests/chr22_chunk.vcf.gz";
    const MAX_TEST_VARIANTS: usize = 32;
    const MAX_TEST_SAMPLES: usize = 8;

    struct LimitedBlockSource<T> {
        inner: T,
        sample_limit: usize,
        variant_limit: usize,
        remaining_variants: usize,
        inner_samples: usize,
        scratch: Vec<f64>,
    }

    impl<T> LimitedBlockSource<T>
    where
        T: VariantBlockSource,
    {
        fn new(inner: T, max_samples: usize, max_variants: usize) -> Self {
            let inner_samples = inner.n_samples();
            let inner_variants = inner.n_variants();
            let sample_limit = max_samples.max(2).min(inner_samples);
            let variant_limit = max_variants.max(1).min(inner_variants);
            let scratch = vec![0.0; inner_samples * variant_limit];
            Self {
                inner,
                sample_limit,
                variant_limit,
                remaining_variants: variant_limit,
                inner_samples,
                scratch,
            }
        }
    }

    impl<T> VariantBlockSource for LimitedBlockSource<T>
    where
        T: VariantBlockSource,
    {
        type Error = T::Error;

        fn n_samples(&self) -> usize {
            self.sample_limit
        }

        fn n_variants(&self) -> usize {
            self.variant_limit
        }

        fn reset(&mut self) -> Result<(), Self::Error> {
            self.inner.reset()?;
            self.remaining_variants = self.variant_limit;
            Ok(())
        }

        fn next_block_into(
            &mut self,
            max_variants: usize,
            storage: &mut [f64],
        ) -> Result<usize, Self::Error> {
            if self.remaining_variants == 0 {
                return Ok(0);
            }

            let request = max_variants.min(self.remaining_variants);
            if request == 0 {
                return Ok(0);
            }

            let inner_len = self.inner_samples * request;
            let read = self
                .inner
                .next_block_into(request, &mut self.scratch[..inner_len])?;
            let consumed = read.min(self.remaining_variants);
            self.remaining_variants -= consumed;

            let samples = self.sample_limit;
            for variant_idx in 0..read {
                let inner_offset = variant_idx * self.inner_samples;
                let outer_offset = variant_idx * samples;
                let src = &self.scratch[inner_offset..inner_offset + samples];
                let dst = &mut storage[outer_offset..outer_offset + samples];
                dst.copy_from_slice(src);
            }

            Ok(read)
        }
    }

    fn should_skip(message: &str) -> bool {
        let lower = message.to_lowercase();
        lower.contains("dns error")
            || lower.contains("timed out")
            || lower.contains("temporary failure")
            || lower.contains("could not resolve host")
            || lower.contains("connection refused")
            || lower.contains("network is unreachable")
    }

    #[test]
    fn fit_hwe_pca_from_http_vcf_stream() {
        let path = Path::new(TEST_VCF_URL);
        let dataset = match GenotypeDataset::open(path) {
            Ok(dataset) => dataset,
            Err(err) => {
                let msg = err.to_string();
                if should_skip(&msg) {
                    eprintln!("skipping HTTP PCA test: {msg}");
                    return;
                }
                panic!("Failed to open dataset: {msg}");
            }
        };

        let block_source = match dataset.block_source() {
            Ok(source) => source,
            Err(err) => {
                let msg = err.to_string();
                if should_skip(&msg) {
                    eprintln!("skipping HTTP PCA test: {msg}");
                    return;
                }
                panic!("Failed to create block source: {msg}");
            }
        };

        let mut limited_source =
            LimitedBlockSource::new(block_source, MAX_TEST_SAMPLES, MAX_TEST_VARIANTS);
        let expected_variants = limited_source.n_variants();
        let expected_samples = limited_source.n_samples();

        let model = match HwePcaModel::fit(&mut limited_source) {
            Ok(model) => model,
            Err(err) => {
                let msg = err.to_string();
                if should_skip(&msg) {
                    eprintln!("skipping HTTP PCA test: {msg}");
                    return;
                }
                panic!("Failed to fit PCA model: {msg}");
            }
        };

        assert_eq!(expected_samples, model.n_samples());
        assert_eq!(expected_variants, model.n_variants());
        assert!(model.components() > 0);
    }
}
