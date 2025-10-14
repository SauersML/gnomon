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
use faer::linalg::matmul::triangular as triangular_matmul;
use faer::linalg::solvers::{Llt as FaerLlt, Solve as FaerSolve};
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
use std::convert::Infallible;
use std::error::Error;
use std::ops::Range;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SupportedLaneCount};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex};
use std::thread;

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
pub const DEFAULT_BLOCK_WIDTH: usize = 2_048;
const DENSE_EIGEN_FALLBACK_THRESHOLD: usize = 64;
const MAX_PARTIAL_COMPONENTS: usize = 512;
const DEFAULT_GRAM_BUDGET_BYTES: usize = 8 * 1024 * 1024 * 1024;
const DEFAULT_LD_WINDOW: usize = 51;
const DEFAULT_LD_RIDGE: f64 = 1.0e-3;
const MIN_LD_WEIGHT: f64 = 1.0e-6;

#[inline]
fn select_top_k_desc(ordering: &mut [(usize, f64)], k: usize) -> usize {
    let mid = k.min(ordering.len());
    if mid == 0 {
        return 0;
    }
    let desc =
        |a: &(usize, f64), b: &(usize, f64)| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal);
    ordering.select_nth_unstable_by(mid - 1, desc);
    ordering[..mid].sort_by(desc);
    mid
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CovarianceComputationMode {
    Dense,
    Partial,
}

fn gram_matrix_budget_bytes() -> usize {
    match std::env::var("GNOMON_GRAM_BUDGET_BYTES") {
        Ok(value) => match value.parse::<u64>() {
            Ok(parsed) if parsed == 0 => usize::MAX,
            Ok(parsed) => usize::try_from(parsed).unwrap_or(usize::MAX),
            Err(_) => DEFAULT_GRAM_BUDGET_BYTES,
        },
        Err(_) => DEFAULT_GRAM_BUDGET_BYTES,
    }
}

fn gram_matrix_size_bytes(n: usize) -> Option<usize> {
    n.checked_mul(n)?.checked_mul(core::mem::size_of::<f64>())
}

fn covariance_computation_mode(n: usize, budget_bytes: usize) -> CovarianceComputationMode {
    match gram_matrix_size_bytes(n) {
        Some(bytes) if bytes <= budget_bytes => CovarianceComputationMode::Dense,
        _ => CovarianceComputationMode::Partial,
    }
}

#[derive(Clone, Debug, Default)]
pub struct FitOptions {
    pub ld: Option<LdConfig>,
}

#[derive(Clone, Debug, Default)]
pub struct LdConfig {
    pub window: Option<usize>,
    pub ridge: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LdWeights {
    pub weights: Vec<f64>,
    pub window: usize,
    pub ridge: f64,
}

#[derive(Clone, Copy, Debug)]
struct LdResolvedConfig {
    window: usize,
    ridge: f64,
}

impl FitOptions {
    fn resolved_ld(
        &self,
        observed_variants: usize,
    ) -> Result<Option<LdResolvedConfig>, HwePcaError> {
        match &self.ld {
            Some(cfg) if observed_variants > 0 => {
                let mut window = cfg.window.unwrap_or(DEFAULT_LD_WINDOW);
                if window == 0 {
                    return Err(HwePcaError::InvalidInput(
                        "LD weighting window must be at least one variant",
                    ));
                }
                window = window.min(observed_variants.max(1));
                if window == 0 {
                    window = 1;
                }
                if window % 2 == 0 {
                    window = window.saturating_sub(1);
                    if window == 0 {
                        window = 1;
                    }
                }

                let ridge = cfg.ridge.unwrap_or(DEFAULT_LD_RIDGE);
                if !(ridge.is_finite() && ridge > 0.0) {
                    return Err(HwePcaError::InvalidInput(
                        "LD weighting ridge must be positive and finite",
                    ));
                }

                Ok(Some(LdResolvedConfig { window, ridge }))
            }
            _ => Ok(None),
        }
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f64);

unsafe impl Send for SendPtr {}

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

    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        let _ = self;
        None
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        let _ = self;
        None
    }
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

fn apply_ld_weights(block: MatMut<'_, f64>, variant_range: Range<usize>, weights: &[f64]) {
    let start = variant_range.start.min(weights.len());
    let end = variant_range.end.min(weights.len());
    if end <= start {
        return;
    }
    let slice = &weights[start..end];
    let columns = block.subcols_mut(0, slice.len());
    for (column, &weight) in columns.col_iter_mut().zip(slice.iter()) {
        if (weight - 1.0).abs() < f64::EPSILON {
            continue;
        }
        zip!(column).for_each(|unzip!(value)| {
            *value *= weight;
        });
    }
}

struct VariantStatsCache {
    frequencies: Vec<f64>,
    scales: Vec<f64>,
    block_sums: Vec<f64>,
    block_calls: Vec<usize>,
    finalized_len: Option<usize>,
    write_pos: usize,
}

impl VariantStatsCache {
    fn new(block_capacity: usize, variant_capacity_hint: usize) -> Self {
        let frequencies = Vec::with_capacity(variant_capacity_hint);
        let scales = Vec::with_capacity(variant_capacity_hint);
        Self {
            frequencies,
            scales,
            block_sums: vec![0.0; block_capacity],
            block_calls: vec![0usize; block_capacity],
            finalized_len: None,
            write_pos: 0,
        }
    }

    fn is_finalized(&self) -> bool {
        self.finalized_len.is_some()
    }

    fn ensure_statistics(&mut self, block: MatRef<'_, f64>, variant_range: Range<usize>, par: Par) {
        if self.is_finalized() {
            return;
        }

        debug_assert!(variant_range.start == self.write_pos);

        let filled = block.ncols();
        {
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
        }

        let end = variant_range.end;
        self.ensure_capacity(end);

        let freq_slice = &mut self.frequencies[variant_range.clone()];
        let scale_slice = &mut self.scales[variant_range.clone()];
        let sums_slice = &self.block_sums[..filled];
        let calls_slice = &self.block_calls[..filled];

        for idx in 0..filled {
            let sum = sums_slice[idx];
            let calls = calls_slice[idx];
            if calls == 0 {
                freq_slice[idx] = 0.0;
                scale_slice[idx] = HWE_SCALE_FLOOR;

                continue;
            }

            let mean_genotype = sum / (calls as f64);
            let allele_freq = (mean_genotype / 2.0).clamp(0.0, 1.0);
            let variance = (2.0 * allele_freq * (1.0 - allele_freq)).max(HWE_VARIANCE_EPSILON);
            let derived_scale = variance.sqrt();

            freq_slice[idx] = allele_freq;
            scale_slice[idx] = if derived_scale < HWE_SCALE_FLOOR {
                HWE_SCALE_FLOOR
            } else {
                derived_scale
            };
        }

        self.write_pos = end;

        debug_assert!(self.frequencies.len() >= end);
        debug_assert!(self.scales.len() >= end);
    }

    fn finalize(&mut self) {
        self.frequencies.truncate(self.write_pos);
        self.scales.truncate(self.write_pos);
        self.finalized_len = Some(self.write_pos);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.finalized_len.unwrap_or(self.write_pos)
    }

    fn into_scaler(self) -> Option<HweScaler> {
        if self.finalized_len.is_some() {
            Some(HweScaler::new(self.frequencies, self.scales))
        } else {
            None
        }
    }

    fn ensure_capacity(&mut self, required: usize) {
        if self.frequencies.len() >= required {
            return;
        }

        let freq_capacity = self.frequencies.capacity();
        if freq_capacity < required {
            let block_capacity = self.block_sums.len();
            let growth_from_capacity = freq_capacity + freq_capacity / 2;
            let growth_from_block = self.write_pos.saturating_add(block_capacity);
            let mut target = required
                .max(growth_from_capacity)
                .max(growth_from_block)
                .max(1);

            if target <= freq_capacity {
                target = required;
            }

            let additional_capacity = target - freq_capacity;
            self.frequencies.reserve_exact(additional_capacity);
            self.scales.reserve_exact(additional_capacity);
        }

        let additional = required - self.frequencies.len();
        self.frequencies
            .extend(std::iter::repeat(0.0).take(additional));
        self.scales.extend(std::iter::repeat(0.0).take(additional));
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
        let lane = Simd::<f64, LANES>::from_array(*chunk);
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

#[cfg_attr(not(test), allow(dead_code))]
#[inline(always)]
fn standardize_column_simd_full(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_full_impl::<STANDARDIZATION_SIMD_LANES>(values, mean, inv);
}

#[cfg_attr(not(test), allow(dead_code))]
#[inline(always)]
fn standardize_column_simd_full_impl<const LANES: usize>(values: &mut [f64], mean: f64, inv: f64)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mean_simd = Simd::<f64, LANES>::splat(mean);
    let inv_simd = Simd::<f64, LANES>::splat(inv);

    let (chunks, remainder) = values.as_chunks_mut::<LANES>();
    for chunk in chunks {
        let lane = Simd::<f64, LANES>::from_array(*chunk);
        let standardized = (lane - mean_simd) * inv_simd;
        *chunk = standardized.to_array();
    }

    for value in remainder {
        *value = (*value - mean) * inv;
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
        let lane = Simd::<f64, LANES>::from_array(*chunk);
        let mask = lane.is_finite();
        let finite = mask.select(lane, zero);
        sum += finite.reduce_sum();
        count += mask.to_bitmask().count_ones() as usize;
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
    ld: Option<LdWeights>,
}

impl HwePcaModel {
    pub fn fit_k<S>(source: &mut S, components: usize) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = Arc::new(NoopFitProgress::default());
        Self::fit_k_with_options_and_progress(source, components, &FitOptions::default(), &progress)
    }

    pub fn fit_k_with_progress<S, P>(
        source: &mut S,
        components: usize,
        progress: &Arc<P>,
    ) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
        P: FitProgressObserver + 'static,
    {
        Self::fit_k_with_options_and_progress(source, components, &FitOptions::default(), progress)
    }

    pub fn fit_k_with_options_and_progress<S, P>(
        source: &mut S,
        components: usize,
        options: &FitOptions,
        progress: &Arc<P>,
    ) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
        P: FitProgressObserver + 'static,
    {
        let n_samples = source.n_samples();
        let n_variants_hint = source.n_variants();

        if n_samples < 2 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least two samples",
            ));
        }

        if components == 0 {
            return Err(HwePcaError::InvalidInput(
                "Requested component count must be at least one",
            ));
        }

        let max_rank = n_samples.saturating_sub(1);
        let target_components = components.min(max_rank);

        let parallelism_guard = ParallelismGuard::new();
        let par = parallelism_guard.active_parallelism();
        let block_capacity = if n_variants_hint > 0 {
            min(DEFAULT_BLOCK_WIDTH.max(1), n_variants_hint)
        } else {
            DEFAULT_BLOCK_WIDTH.max(1)
        };
        progress.on_stage_start(FitProgressStage::AlleleStatistics, n_variants_hint);
        let stats_progress =
            StageProgressHandle::new(Arc::clone(progress), FitProgressStage::AlleleStatistics);
        let (scaler, observed_variants) = compute_variant_statistics(
            source,
            block_capacity,
            par,
            stats_progress,
            n_variants_hint,
        )?;

        let ld_config = options.resolved_ld(observed_variants)?;
        let ld_weights = if let Some(ld_cfg) = ld_config {
            Some(compute_ld_weights(
                source,
                &scaler,
                observed_variants,
                block_capacity,
                ld_cfg,
                n_variants_hint,
                progress,
                par,
            )?)
        } else {
            None
        };

        let ld_weights_arc = ld_weights
            .as_ref()
            .map(|ld| Arc::<[f64]>::from(ld.weights.clone().into_boxed_slice()));

        let operator = StandardizedCovarianceOp::new(
            source,
            block_capacity,
            n_variants_hint,
            observed_variants,
            scaler,
            ld_weights_arc.clone(),
        );

        let gram_budget = gram_matrix_budget_bytes();
        let gram_bytes = gram_matrix_size_bytes(n_samples);
        let gram_mode = covariance_computation_mode(n_samples, gram_budget);
        let gram_progress_handle = match gram_mode {
            CovarianceComputationMode::Dense => {
                progress.on_stage_start(FitProgressStage::GramMatrix, n_variants_hint);
                Some(StageProgressHandle::new(
                    Arc::clone(progress),
                    FitProgressStage::GramMatrix,
                ))
            }
            CovarianceComputationMode::Partial => {
                progress.on_stage_start(FitProgressStage::GramMatrix, 0);
                if n_variants_hint > 0 {
                    progress.on_stage_estimate(FitProgressStage::GramMatrix, n_variants_hint);
                }
                None
            }
        };

        if matches!(gram_mode, CovarianceComputationMode::Partial) {
            if let Some(bytes) = gram_bytes {
                log::info!(
                    "Skipping explicit Gram matrix ({} bytes exceeds budget of {} bytes); using matrix-free solver",
                    bytes,
                    gram_budget
                );
            } else {
                log::info!(
                    "Skipping explicit Gram matrix due to size overflow; using matrix-free solver"
                );
            }
        }

        let decomposition_result = compute_covariance_eigenpairs(
            &operator,
            par,
            gram_mode,
            target_components,
            gram_progress_handle.as_ref(),
        );
        let (source, scaler) = operator.into_parts();
        let decomposition = decomposition_result?;

        let variant_count = scaler.variant_scales().len();
        debug_assert_eq!(variant_count, observed_variants);
        if variant_count == 0 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least one variant",
            ));
        }

        log::info!("Observed {} variants during first pass", variant_count);

        if gram_progress_handle.is_none() {
            progress.on_stage_total(FitProgressStage::GramMatrix, variant_count);
        }
        progress.on_stage_finish(FitProgressStage::GramMatrix);

        if decomposition.values.is_empty() {
            return Err(HwePcaError::Eigen(
                "All eigenvalues are numerically zero; increase cohort size or review input data"
                    .into(),
            ));
        }

        let (singular_values, sample_scores) = build_sample_scores(n_samples, &decomposition);

        let loadings = compute_variant_loadings(
            source,
            &scaler,
            variant_count,
            block_capacity,
            decomposition.vectors.as_ref(),
            &singular_values,
            ld_weights_arc.as_deref(),
            progress,
            par,
        )?;

        Ok(Self {
            n_samples,
            n_variants: variant_count,
            scaler,
            eigenvalues: decomposition.values,
            singular_values,
            sample_basis: decomposition.vectors,
            sample_scores,
            loadings,
            variant_keys: None,
            ld: ld_weights,
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

    pub fn ld(&self) -> Option<&LdWeights> {
        self.ld.as_ref()
    }
}

struct Eigenpairs {
    values: Vec<f64>,
    vectors: Mat<f64>,
}

#[derive(Debug)]
struct DenseSymmetricOp<'a> {
    matrix: MatRef<'a, f64>,
}

impl<'a> LinOp<f64> for DenseSymmetricOp<'a> {
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _ = (rhs_ncols, par);
        StackReq::empty()
    }

    fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    fn apply(
        &self,
        mut out: MatMut<'_, f64>,
        rhs: MatRef<'_, f64>,
        par: Par,
        stack: &mut MemStack,
    ) {
        let _ = stack;
        matmul(out.rb_mut(), Accum::Replace, self.matrix, rhs, 1.0, par);
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

struct StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    source: Mutex<&'a mut S>,
    n_samples: usize,
    n_variants_hint: usize,
    block_capacity: usize,
    scale: f64,
    scaler: HweScaler,
    observed_variants: usize,
    ld_weights: Option<Arc<[f64]>>,
    error: Mutex<Option<HwePcaError>>,
    marker: PhantomData<P>,
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
        n_variants_hint: usize,
        observed_variants: usize,
        scaler: HweScaler,
        ld_weights: Option<Arc<[f64]>>,
    ) -> Self {
        let n_samples = source.n_samples();
        let scale = 1.0 / ((n_samples - 1) as f64);
        Self {
            source: Mutex::new(source),
            n_samples,
            n_variants_hint,
            block_capacity,
            scale,
            scaler,
            observed_variants,
            ld_weights,
            error: Mutex::new(None),
            marker: PhantomData,
        }
    }

    fn into_parts(self) -> (&'a mut S, HweScaler) {
        let Self {
            source,
            n_samples: _,
            n_variants_hint: _,
            block_capacity: _,
            scale: _,
            scaler,
            observed_variants: _,
            ld_weights: _,
            error: _,
            marker: _,
        } = self;
        (
            source
                .into_inner()
                .expect("covariance source mutex poisoned during teardown"),
            scaler,
        )
    }

    fn take_error(&self) -> Option<HwePcaError> {
        self.error
            .lock()
            .expect("operator error mutex poisoned")
            .take()
    }

    fn fail_invalid(&self, msg: &'static str) -> ! {
        self.record_error(HwePcaError::InvalidInput(msg))
    }

    fn record_error(&self, err: HwePcaError) -> ! {
        let mut slot = self.error.lock().expect("operator error mutex poisoned");
        if slot.is_none() {
            *slot = Some(err);
        }
        std::panic::panic_any(OperatorError);
    }

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn standardize_block_in_place(
        &self,
        mut block: MatMut<'_, f64>,
        variant_range: Range<usize>,
        par: Par,
    ) {
        self.scaler
            .standardize_block(block.as_mut(), variant_range.clone(), par);
        if let Some(weights) = &self.ld_weights {
            apply_ld_weights(block, variant_range, weights);
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
            .field("n_variants_hint", &self.n_variants_hint)
            .field("observed_variants", &self.observed_variants)
            .field("block_capacity", &self.block_capacity)
            .finish()
    }
}

impl<'a, S, P> LinOp<f64> for StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    fn apply_scratch(&self, rhs_ncols: usize, par: Par) -> StackReq {
        let _ = par;
        let block_len = self.n_samples * self.block_capacity;
        let block_req = StackReq::new::<f64>(block_len);
        let proj_req = temp_mat_scratch::<f64>(self.block_capacity, rhs_ncols);
        block_req.and(block_req).and(proj_req)
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
        debug_assert_eq!(out.nrows(), self.n_samples);
        debug_assert_eq!(rhs.nrows(), self.n_samples);

        out.fill(0.0);

        if rhs.ncols() == 0 {
            return;
        }

        let block_len = self.n_samples * self.block_capacity;
        let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
        let buf0 = unsafe {
            std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len)
        };
        let (buf1_uninit, stack) = stack.make_uninit::<f64>(block_len);
        let buf1 = unsafe {
            std::slice::from_raw_parts_mut(buf1_uninit.as_mut_ptr() as *mut f64, block_len)
        };
        let mut buffer_slices = [buf0, buf1];
        let [first_slice, second_slice] = &mut buffer_slices;
        let buffer_ptrs = [
            SendPtr(first_slice.as_mut_ptr()),
            SendPtr(second_slice.as_mut_ptr()),
        ];

        let (mut proj_uninit, _) =
            unsafe { temp_mat_uninit::<f64, _, _>(self.block_capacity, rhs.ncols(), stack) };
        let mut proj_storage = proj_uninit.as_mat_mut();

        enum PrefetchMessage {
            Data {
                id: usize,
                filled: usize,
                start: usize,
            },
            End,
            Error(HwePcaError),
        }

        let buffer_count = buffer_ptrs.len();
        let (filled_tx, filled_rx) = sync_channel::<PrefetchMessage>(buffer_count);
        let (free_tx, free_rx) = sync_channel::<usize>(buffer_count);
        for id in 0..buffer_count {
            free_tx.send(id).expect("failed to seed prefetch buffers");
        }

        let source_mutex = &self.source;
        let n_samples = self.n_samples;
        let block_capacity = self.block_capacity;
        let observed_total = self.observed_variants;
        let scale = self.scale;
        let block_len = block_len;

        let processed = thread::scope(|scope| {
            let buffer_ptrs_prefetch = buffer_ptrs;
            let filled_sender = filled_tx;
            let free_receiver = free_rx;
            scope.spawn(move || {
                if let Err(err) = {
                    let mut guard = source_mutex
                        .lock()
                        .expect("covariance source mutex poisoned");
                    let source: &mut S = &mut **guard;
                    source.reset().map_err(|e| HwePcaError::Source(Box::new(e)))
                } {
                    let _ = filled_sender.send(PrefetchMessage::Error(err));
                    return;
                }

                let mut start = 0usize;
                while let Ok(id) = free_receiver.recv() {
                    let buffer_slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                    };
                    let filled_res = {
                        let mut guard = source_mutex
                            .lock()
                            .expect("covariance source mutex poisoned");
                        let source: &mut S = &mut **guard;
                        source.next_block_into(block_capacity, buffer_slice)
                    };

                    match filled_res {
                        Ok(filled) => {
                            if filled == 0 {
                                let _ = filled_sender.send(PrefetchMessage::End);
                                break;
                            }

                            let _ = filled_sender.send(PrefetchMessage::Data { id, filled, start });
                            start += filled;
                        }
                        Err(err) => {
                            let _ = filled_sender
                                .send(PrefetchMessage::Error(HwePcaError::Source(Box::new(err))));
                            break;
                        }
                    }
                }
            });

            let free_sender = free_tx;
            let mut processed = 0usize;
            let buffer_ptrs_compute = buffer_ptrs;
            while let Ok(message) = filled_rx.recv() {
                match message {
                    PrefetchMessage::Data { id, filled, start } => {
                        if start != processed {
                            self.fail_invalid("prefetch produced out-of-order variant ranges");
                        }
                        if start + filled > observed_total {
                            self.fail_invalid(
                                "VariantBlockSource returned more variants than observed",
                            );
                        }

                        let block_slice = unsafe {
                            std::slice::from_raw_parts_mut(buffer_ptrs_compute[id].0, block_len)
                        };
                        let mut block = MatMut::from_column_major_slice_mut(
                            &mut block_slice[..n_samples * filled],
                            n_samples,
                            filled,
                        );
                        let variant_range = start..start + filled;
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
                            scale,
                            par,
                        );

                        processed = start + filled;

                        if free_sender.send(id).is_err() {
                            break;
                        }
                    }
                    PrefetchMessage::End => {
                        break;
                    }
                    PrefetchMessage::Error(err) => {
                        self.record_error(err);
                    }
                }
            }

            processed
        });

        if processed != self.observed_variants {
            self.fail_invalid("VariantBlockSource terminated early during covariance accumulation");
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
    top_k: usize,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    let n = operator.n_samples();
    if n == 0 || top_k == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(0, 0),
        });
    }

    let max_rank = n.saturating_sub(1);
    if max_rank == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    let desired = top_k.min(max_rank);
    if desired == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    if matches!(mode, CovarianceComputationMode::Dense) {
        return compute_covariance_eigenpairs_dense(operator, par, desired, progress);
    }

    let upper_target = max_rank.min(MAX_PARTIAL_COMPONENTS.max(desired));
    if upper_target == 0 {
        return compute_covariance_eigenpairs_dense(operator, par, desired, progress);
    }

    let normalization = (n as f64).sqrt();
    let v0 = Col::from_fn(n, |_| 1.0 / normalization);
    let mut target = desired.min(upper_target).max(1);

    loop {
        let params = partial_solver_params(n, target);
        let (info, eigvals, eigvecs) = run_partial_eigensolver(operator, target, par, &v0, params)?;

        let n_converged = info.n_converged_eigen.min(target);

        let positive = eigvals
            .iter()
            .take(n_converged)
            .filter(|&&v| v > EIGENVALUE_EPSILON)
            .count();

        if positive == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        let keep = positive.min(desired);
        let mut ordering = Vec::with_capacity(n_converged);
        for idx in 0..n_converged {
            ordering.push((idx, eigvals[idx]));
        }

        let mid = select_top_k_desc(&mut ordering, keep);

        let mut values = Vec::with_capacity(mid);
        let mut vectors = Mat::zeros(n, mid);
        for (out_idx, (src_idx, value)) in ordering[..mid].iter().copied().enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = eigvecs[(row, src_idx)];
            }
        }

        if keep >= desired || target >= upper_target {
            return Ok(Eigenpairs { values, vectors });
        }

        let next_target = (target * 2).min(upper_target);
        if next_target == target {
            return Ok(Eigenpairs { values, vectors });
        }

        target = next_target;
    }
}

fn compute_covariance_eigenpairs_dense<S, P>(
    operator: &StandardizedCovarianceOp<'_, S, P>,
    par: Par,
    top_k: usize,
    progress: Option<&StageProgressHandle<P>>,
) -> Result<Eigenpairs, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    let mut covariance = accumulate_covariance_matrix(operator, par, progress)?;
    let n = covariance.nrows();

    if n == 0 || top_k == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
        });
    }

    if n <= DENSE_EIGEN_FALLBACK_THRESHOLD || top_k + 8 >= n {
        let eig = covariance.self_adjoint_eigen(Side::Lower).map_err(|err| {
            HwePcaError::Eigen(format!("dense eigendecomposition failed: {err:?}"))
        })?;

        let diag = eig.S();
        let basis = eig.U();

        let positive = (0..n).filter(|&i| diag[i] > EIGENVALUE_EPSILON).count();

        let keep = positive.min(top_k);
        if keep == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        let mut ordering = Vec::with_capacity(n);
        for idx in 0..n {
            ordering.push((idx, diag[idx]));
        }

        let mid = select_top_k_desc(&mut ordering, keep);

        let mut values = Vec::with_capacity(mid);
        let mut vectors = Mat::zeros(n, mid);
        for (out_idx, (src_idx, value)) in ordering[..mid].iter().copied().enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = basis[(row, src_idx)];
            }
        }

        return Ok(Eigenpairs { values, vectors });
    }

    mirror_lower_to_upper(&mut covariance);

    let gram = covariance.as_ref();
    let upper_target = n.min(MAX_PARTIAL_COMPONENTS.max(top_k));
    let mut target = top_k.min(upper_target).max(1);

    let normalization = (n as f64).sqrt();
    let v0 = Col::from_fn(n, |_| 1.0 / normalization);
    let op = DenseSymmetricOp { matrix: gram };

    loop {
        let params = partial_solver_params(n, target);
        let mut eigvecs = Mat::zeros(n, target);
        let mut eigvals = vec![0.0f64; target];
        let scratch = partial_eigen_scratch(&op, params.max_dim, par, params);
        let mut mem = MemBuffer::new(scratch);
        let info = {
            let mut stack = MemStack::new(&mut mem);
            partial_self_adjoint_eigen(
                eigvecs.as_mut(),
                &mut eigvals,
                &op,
                v0.as_ref(),
                f64::EPSILON * 128.0,
                par,
                &mut stack,
                params,
            )
        };

        let n_converged = info.n_converged_eigen.min(target);

        let positive = (0..n_converged)
            .filter(|&i| eigvals[i] > EIGENVALUE_EPSILON)
            .count();

        if positive == 0 {
            return Ok(Eigenpairs {
                values: Vec::new(),
                vectors: Mat::zeros(n, 0),
            });
        }

        let keep = positive.min(top_k);
        let mut ordering = Vec::with_capacity(n_converged);
        for idx in 0..n_converged {
            ordering.push((idx, eigvals[idx]));
        }

        let mid = select_top_k_desc(&mut ordering, keep);

        let mut values = Vec::with_capacity(mid);
        let mut vectors = Mat::zeros(n, mid);
        for (out_idx, (src_idx, value)) in ordering[..mid].iter().copied().enumerate() {
            values.push(value);
            for row in 0..n {
                vectors[(row, out_idx)] = eigvecs[(row, src_idx)];
            }
        }

        if keep >= top_k || target >= upper_target {
            return Ok(Eigenpairs { values, vectors });
        }

        let next_target = (target * 2).min(upper_target);
        if next_target == target {
            return Ok(Eigenpairs { values, vectors });
        }

        target = next_target;
    }
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
    let covariance = Mat::zeros(n_samples, n_samples);

    if n_samples == 0 {
        return Ok(covariance);
    }

    let block_capacity = operator.block_capacity;
    let block_len = n_samples * block_capacity;
    let buffer_req = StackReq::new::<f64>(block_len).and(StackReq::new::<f64>(block_len));
    let mut mem = MemBuffer::new(buffer_req);
    let stack = MemStack::new(&mut mem);
    let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
    let buf0 =
        unsafe { std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len) };
    let (buf1_uninit, _) = stack.make_uninit::<f64>(block_len);
    let buf1 =
        unsafe { std::slice::from_raw_parts_mut(buf1_uninit.as_mut_ptr() as *mut f64, block_len) };
    let mut buffer_slices = [buf0, buf1];
    let [first_slice, second_slice] = &mut buffer_slices;
    let buffer_ptrs = [
        SendPtr(first_slice.as_mut_ptr()),
        SendPtr(second_slice.as_mut_ptr()),
    ];

    enum PrefetchMessage {
        Data {
            id: usize,
            filled: usize,
            start: usize,
        },
        End,
        Error(HwePcaError),
    }

    let buffer_count = buffer_ptrs.len();
    let (filled_tx, filled_rx) = sync_channel::<PrefetchMessage>(buffer_count);
    let (free_tx, free_rx) = sync_channel::<usize>(buffer_count);
    for id in 0..buffer_count {
        free_tx.send(id).expect("failed to seed covariance buffers");
    }

    let source_mutex = &operator.source;
    let n_variants_hint = operator.n_variants_hint;
    let observed_total = operator.observed_variants;
    let scale = operator.scale;
    let block_capacity = operator.block_capacity;
    let block_len = block_len;
    let progress_handle = progress;

    let result = thread::scope(|scope| {
        let buffer_ptrs_prefetch = buffer_ptrs;
        let filled_sender = filled_tx;
        let free_receiver = free_rx;
        let progress_handle = progress_handle;
        scope.spawn(move || {
            if let Err(err) = {
                let mut guard = source_mutex
                    .lock()
                    .expect("covariance source mutex poisoned");
                let source: &mut S = &mut **guard;
                source.reset().map_err(|e| HwePcaError::Source(Box::new(e)))
            } {
                let _ = filled_sender.send(PrefetchMessage::Error(err));
                return;
            }

            let mut start = 0usize;
            let mut used_source_progress = false;
            while let Ok(id) = free_receiver.recv() {
                let buffer_slice = unsafe {
                    std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                };
                let (filled_res, progress_bytes, progress_variants) = {
                    let mut guard = source_mutex
                        .lock()
                        .expect("covariance source mutex poisoned");
                    let source: &mut S = &mut **guard;
                    let filled = source.next_block_into(block_capacity, buffer_slice);
                    let bytes = source.progress_bytes();
                    let variants = source.progress_variants();
                    (filled, bytes, variants)
                };

                match filled_res {
                    Ok(filled) => {
                        if filled == 0 {
                            if let Some(handle) = progress_handle {
                                if let Some((_, Some(total))) = progress_variants {
                                    handle.set_total(total);
                                } else if !used_source_progress {
                                    handle.set_total(start);
                                }
                            }
                            let _ = filled_sender.send(PrefetchMessage::End);
                            break;
                        }

                        if let Some(handle) = progress_handle {
                            if let Some((bytes_read, total_bytes)) = progress_bytes {
                                used_source_progress = true;
                                handle.advance_bytes(bytes_read, total_bytes);
                            } else if let Some((work_done, total_work)) = progress_variants {
                                used_source_progress = true;
                                if let Some(total) = total_work {
                                    handle.set_total(total);
                                } else if n_variants_hint > 0 {
                                    handle.estimate(n_variants_hint);
                                }
                                handle.advance(work_done);
                            } else {
                                handle.advance(start + filled);
                            }
                        }

                        let _ = filled_sender.send(PrefetchMessage::Data { id, filled, start });
                        start += filled;
                    }
                    Err(err) => {
                        let _ = filled_sender
                            .send(PrefetchMessage::Error(HwePcaError::Source(Box::new(err))));
                        break;
                    }
                }
            }
        });

        let free_sender = free_tx;
        let mut processed = 0usize;
        let buffer_ptrs_compute = buffer_ptrs;
        let mut covariance = covariance;
        while let Ok(message) = filled_rx.recv() {
            match message {
                PrefetchMessage::Data { id, filled, start } => {
                    if start != processed {
                        return Err(HwePcaError::InvalidInput(
                            "prefetch produced out-of-order variant ranges",
                        ));
                    }
                    if start + filled > observed_total {
                        return Err(HwePcaError::InvalidInput(
                            "VariantBlockSource returned more variants than observed",
                        ));
                    }

                    let block_slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer_ptrs_compute[id].0, block_len)
                    };
                    let mut block = MatMut::from_column_major_slice_mut(
                        &mut block_slice[..n_samples * filled],
                        n_samples,
                        filled,
                    );
                    let variant_range = start..start + filled;
                    operator.standardize_block_in_place(block.rb_mut(), variant_range.clone(), par);

                    triangular_matmul::matmul(
                        covariance.as_mut(),
                        triangular_matmul::BlockStructure::TriangularLower,
                        Accum::Add,
                        block.as_ref(),
                        triangular_matmul::BlockStructure::Rectangular,
                        block.as_ref().transpose(),
                        triangular_matmul::BlockStructure::Rectangular,
                        scale,
                        par,
                    );

                    processed = start + filled;

                    if free_sender.send(id).is_err() {
                        break;
                    }
                }
                PrefetchMessage::End => {
                    break;
                }
                PrefetchMessage::Error(err) => {
                    return Err(err);
                }
            }
        }

        if processed == 0 {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource yielded no variants",
            ));
        }

        if processed != observed_total {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early during covariance accumulation",
            ));
        }

        Ok(covariance)
    });

    result
}

fn mirror_lower_to_upper(matrix: &mut Mat<f64>) {
    debug_assert_eq!(matrix.nrows(), matrix.ncols());
    let n = matrix.nrows();
    for col in 0..n {
        for row in 0..col {
            let value = matrix[(col, row)];
            matrix[(row, col)] = value;
        }
    }
}

fn partial_solver_params(n: usize, target: usize) -> PartialEigenParams {
    let mut params = PartialEigenParams::default();
    let max_available = n.saturating_sub(1);
    params.min_dim = target.max(64).min(max_available); // let Faer clamp internally
    params.max_dim = (2 * target).max(128).min(max_available);
    if params.max_dim < params.min_dim {
        params.max_dim = params.min_dim;
    }
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

    // The partial eigensolver can increase its working subspace dimension up to
    // `params.max_dim` during restarts. The scratch allocator must therefore be
    // sized for the worst-case dimension rather than the requested target, or
    // faer's internal workspace requests will overflow the provided buffer.
    let scratch = partial_eigen_scratch(operator, params.max_dim, par, params);
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
        let scaled = (n_samples - 1) as f64 * lambda;
        let sigma = if scaled > 0.0 { scaled.sqrt() } else { 0.0 };
        singular_values.push(sigma);
        zip!(&mut column).for_each(|unzip!(value)| {
            *value *= sigma;
        });
    }

    (singular_values, sample_scores)
}

fn compute_variant_statistics<S, P>(
    source: &mut S,
    block_capacity: usize,
    par: Par,
    progress: StageProgressHandle<P>,
    n_variants_hint: usize,
) -> Result<(HweScaler, usize), HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver,
{
    let n_samples = source.n_samples();
    let mut stats = VariantStatsCache::new(block_capacity, n_variants_hint);
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];

    source
        .reset()
        .map_err(|err| HwePcaError::Source(Box::new(err)))?;

    let mut processed = 0usize;
    let mut used_source_progress = false;

    loop {
        let filled = source
            .next_block_into(block_capacity, &mut block_storage[..])
            .map_err(|err| HwePcaError::Source(Box::new(err)))?;

        if filled == 0 {
            break;
        }

        if n_variants_hint > 0 && processed + filled > n_variants_hint {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource returned more variants than reported hint",
            ));
        }

        let block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        let variant_range = processed..processed + filled;
        stats.ensure_statistics(block.as_ref(), variant_range.clone(), par);

        processed += filled;

        if let Some((bytes_read, total_bytes)) = source.progress_bytes() {
            used_source_progress = true;
            progress.advance_bytes(bytes_read, total_bytes);
        } else if let Some((work_done, total_work)) = source.progress_variants() {
            used_source_progress = true;
            if let Some(total) = total_work {
                progress.set_total(total);
            } else if n_variants_hint > 0 {
                progress.estimate(n_variants_hint);
            }
            progress.advance(work_done);
        } else {
            progress.advance(processed);
        }
    }

    if processed == 0 {
        progress.finish();
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource yielded no variants",
        ));
    }

    if let Some((_, Some(total))) = source.progress_variants() {
        progress.set_total(total);
    } else if !used_source_progress {
        progress.set_total(processed);
    }

    stats.finalize();
    let scaler = stats
        .into_scaler()
        .expect("finalized statistics must produce a scaler");
    progress.finish();

    Ok((scaler, processed))
}

struct LdRingBuffer {
    values: Mat<f64>,
    masks: Vec<u8>,
    n_samples: usize,
    indices: Vec<usize>,
    start: usize,
    len: usize,
}

impl LdRingBuffer {
    fn new(n_samples: usize, capacity: usize) -> Self {
        Self {
            values: Mat::zeros(n_samples, capacity),
            masks: vec![0u8; n_samples * capacity],
            n_samples,
            indices: vec![usize::MAX; capacity],
            start: 0,
            len: 0,
        }
    }

    fn capacity(&self) -> usize {
        self.indices.len()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn values(&self) -> MatRef<'_, f64> {
        self.values.as_ref()
    }

    fn values_mut(&mut self) -> MatMut<'_, f64> {
        self.values.as_mut()
    }

    fn mask_slice(&self, slot: usize) -> &[u8] {
        let start = slot * self.n_samples;
        &self.masks[start..start + self.n_samples]
    }

    fn mask_slice_mut(&mut self, slot: usize) -> &mut [u8] {
        let start = slot * self.n_samples;
        &mut self.masks[start..start + self.n_samples]
    }

    fn indices_mut(&mut self) -> &mut [usize] {
        &mut self.indices
    }

    fn push_slot(&mut self) -> usize {
        let capacity = self.capacity();
        if capacity == 0 {
            return 0;
        }
        let slot = if self.len < capacity {
            let slot = (self.start + self.len) % capacity;
            self.len += 1;
            slot
        } else {
            let slot = self.start;
            self.start = (self.start + 1) % capacity;
            slot
        };
        slot
    }

    fn position_of(&self, index: usize) -> Option<usize> {
        let capacity = self.capacity();
        if capacity == 0 {
            return None;
        }
        for offset in 0..self.len {
            let slot = (self.start + offset) % capacity;
            if self.indices[slot] == index {
                return Some(offset);
            }
        }
        None
    }

    fn slot_at(&self, offset: usize) -> usize {
        let capacity = self.capacity();
        if capacity == 0 {
            0
        } else {
            (self.start + offset) % capacity
        }
    }

    fn truncate_front(&mut self, keep_from: usize) {
        let capacity = self.capacity();
        if capacity == 0 {
            return;
        }
        while self.len > 0 {
            let slot = self.start;
            if self.indices[slot] < keep_from {
                self.indices[slot] = usize::MAX;
                self.start = (self.start + 1) % capacity;
                self.len -= 1;
            } else {
                break;
            }
        }
    }
}

fn compute_ld_weights<S, P>(
    source: &mut S,
    scaler: &HweScaler,
    observed_variants: usize,
    block_capacity: usize,
    config: LdResolvedConfig,
    n_variants_hint: usize,
    progress: &Arc<P>,
    par: Par,
) -> Result<LdWeights, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + 'static,
{
    let mut weights = vec![1.0; observed_variants];
    progress.on_stage_start(FitProgressStage::LdWeights, observed_variants);
    let stage_progress =
        StageProgressHandle::new(Arc::clone(progress), FitProgressStage::LdWeights);

    if observed_variants == 0 {
        stage_progress.finish();
        return Ok(LdWeights {
            weights,
            window: config.window,
            ridge: config.ridge,
        });
    }

    let n_samples = source.n_samples();
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];
    let mut presence_storage = vec![0.0f64; n_samples * block_capacity];
    let window_capacity = config.window.max(1);
    let mut ring = LdRingBuffer::new(n_samples, window_capacity);
    let mut next_weight = 0usize;

    let mut values_scratch = Mat::zeros(n_samples, window_capacity);
    let mut mask_scratch = Mat::zeros(n_samples, window_capacity);
    let mut gram_scratch = Mat::zeros(window_capacity, window_capacity);
    let mut count_scratch = Mat::zeros(window_capacity, window_capacity);
    let mut system_scratch = Mat::zeros(window_capacity, window_capacity);
    let mut rhs_scratch = Mat::zeros(window_capacity, 1);

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

        if n_variants_hint > 0 && processed + filled > n_variants_hint {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource returned more variants than reported hint",
            ));
        }

        let mut block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );
        let mut presence = MatMut::from_column_major_slice_mut(
            &mut presence_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        let variant_range = processed..processed + filled;
        scaler.standardize_block_with_mask(
            block.as_mut(),
            variant_range.clone(),
            presence.as_mut(),
            par,
        );

        for (local_idx, (column, present)) in block
            .as_ref()
            .col_iter()
            .zip(presence.as_ref().col_iter())
            .enumerate()
        {
            let slot = ring.push_slot();
            {
                let dst_col = ring.values_mut().col_mut(slot);
                zip!(dst_col, column).for_each(|unzip!(dst, src)| {
                    *dst = *src;
                });
            }
            {
                let mask_slice = ring.mask_slice_mut(slot);
                for (dst, &src) in mask_slice.iter_mut().zip(present.iter()) {
                    *dst = if src != 0.0 { 1u8 } else { 0u8 };
                }
            }
            ring.indices_mut()[slot] = processed + local_idx;
            assign_ready_weights(
                &mut ring,
                &mut weights,
                &mut next_weight,
                config,
                &mut values_scratch,
                &mut mask_scratch,
                &mut gram_scratch,
                &mut count_scratch,
                &mut system_scratch,
                &mut rhs_scratch,
                &stage_progress,
                par,
            )?;
        }

        processed += filled;
    }

    assign_ready_weights(
        &mut ring,
        &mut weights,
        &mut next_weight,
        config,
        &mut values_scratch,
        &mut mask_scratch,
        &mut gram_scratch,
        &mut count_scratch,
        &mut system_scratch,
        &mut rhs_scratch,
        &stage_progress,
        par,
    )?;

    stage_progress.set_total(observed_variants);
    stage_progress.finish();

    Ok(LdWeights {
        weights,
        window: config.window,
        ridge: config.ridge,
    })
}

fn assign_ready_weights<P: FitProgressObserver>(
    ring: &mut LdRingBuffer,
    weights: &mut [f64],
    next_weight: &mut usize,
    config: LdResolvedConfig,
    values_scratch: &mut Mat<f64>,
    mask_scratch: &mut Mat<f64>,
    gram_scratch: &mut Mat<f64>,
    count_scratch: &mut Mat<f64>,
    system_scratch: &mut Mat<f64>,
    rhs_scratch: &mut Mat<f64>,
    progress: &StageProgressHandle<P>,
    par: Par,
) -> Result<(), HwePcaError> {
    while *next_weight < weights.len() {
        let position = match ring.position_of(*next_weight) {
            Some(pos) => pos,
            None => break,
        };

        let available = ring.len();
        if available == 0 {
            break;
        }

        let window_size = config.window.min(available).max(1);
        let half_window = window_size / 2;
        let start = if available <= window_size {
            0
        } else if position <= half_window {
            0
        } else {
            let tail_start = available.saturating_sub(window_size);
            min(position - half_window, tail_start)
        };
        let end = min(start + window_size, available);
        let window_len = end - start;
        if window_len == 0 {
            break;
        }
        let center = position - start;

        let values_ref = ring.values();
        let first_slot = ring.slot_at(start);
        let contiguous = first_slot + window_len <= ring.capacity();
        let n_samples = values_ref.nrows();

        for col in 0..window_len {
            let slot = ring.slot_at(start + col);
            let src = ring.mask_slice(slot);
            let dst_col = mask_scratch.as_mut().col_mut(col);
            for (dst_value, &src_value) in dst_col.iter_mut().zip(src.iter()) {
                *dst_value = f64::from(src_value);
            }
        }
        let mask_view = mask_scratch.as_ref().submatrix(0, 0, n_samples, window_len);

        let values_view = if contiguous {
            values_ref.submatrix(0, first_slot, n_samples, window_len)
        } else {
            for col in 0..window_len {
                let slot = ring.slot_at(start + col);
                let src = values_ref.col(slot);
                let dst_col = values_scratch.as_mut().col_mut(col);
                zip!(dst_col, src).for_each(|unzip!(dst, src)| {
                    *dst = *src;
                });
            }
            values_scratch
                .as_ref()
                .submatrix(0, 0, n_samples, window_len)
        };

        let mut gram_view = gram_scratch
            .as_mut()
            .submatrix_mut(0, 0, window_len, window_len);
        matmul(
            gram_view.as_mut(),
            Accum::Replace,
            values_view.transpose(),
            values_view,
            1.0,
            par,
        );

        let mut count_view = count_scratch
            .as_mut()
            .submatrix_mut(0, 0, window_len, window_len);
        matmul(
            count_view.as_mut(),
            Accum::Replace,
            mask_view.transpose(),
            mask_view,
            1.0,
            par,
        );

        let mut system_view = system_scratch
            .as_mut()
            .submatrix_mut(0, 0, window_len, window_len);
        let mut rhs_view = rhs_scratch.as_mut().submatrix_mut(0, 0, window_len, 1);
        let weight = solve_ld_window_from_gram(
            gram_view.as_ref(),
            count_view.as_ref(),
            center,
            config.ridge,
            system_view.as_mut(),
            rhs_view.as_mut(),
        );
        weights[*next_weight] = weight;
        progress.advance(*next_weight + 1);
        *next_weight += 1;

        let keep_from = next_weight.saturating_sub(config.window / 2);
        ring.truncate_front(keep_from);
    }

    Ok(())
}

fn solve_ld_window_from_gram(
    gram: MatRef<'_, f64>,
    counts: MatRef<'_, f64>,
    center: usize,
    ridge: f64,
    mut system: MatMut<'_, f64>,
    mut rhs: MatMut<'_, f64>,
) -> f64 {
    let size = gram.nrows();
    if size == 0 || center >= size {
        return 1.0;
    }

    let mut adjusted_ridge = ridge;
    for attempt in 0..2 {
        for i in 0..size {
            system[(i, i)] = 1.0 + adjusted_ridge;
            for j in 0..i {
                let count = counts[(i, j)];
                let value = if count.is_finite() && count > 1.0 {
                    let den = count - 1.0;
                    if den <= 0.0 {
                        0.0
                    } else {
                        let dot = gram[(i, j)];
                        let corr = dot / den;
                        let noise = 1.0 / den;
                        let raw = corr * corr - noise;
                        if raw.is_finite() && raw > 0.0 {
                            raw
                        } else {
                            0.0
                        }
                    }
                } else {
                    0.0
                };
                system[(i, j)] = value;
                system[(j, i)] = value;
            }
        }

        for i in 0..size {
            rhs[(i, 0)] = 1.0;
        }

        match FaerLlt::new(system.as_ref(), Side::Lower) {
            Ok(factor) => {
                let solution = factor.solve(rhs.as_ref());
                let mut weight_sq = solution[(center, 0)];
                if !weight_sq.is_finite() || weight_sq <= 0.0 {
                    weight_sq = 1.0;
                }
                return weight_sq.sqrt().max(MIN_LD_WEIGHT);
            }
            Err(_) => {
                if attempt == 0 {
                    adjusted_ridge *= 10.0;
                    continue;
                } else {
                    return 1.0;
                }
            }
        }
    }

    1.0
}

fn compute_variant_loadings<S, P>(
    source: &mut S,
    scaler: &HweScaler,
    expected_variants: usize,
    block_capacity: usize,
    sample_basis: MatRef<'_, f64>,
    singular_values: &[f64],
    ld_weights: Option<&[f64]>,
    progress: &Arc<P>,
    par: Par,
) -> Result<Mat<f64>, HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver,
{
    let n_samples = source.n_samples();
    let n_components = singular_values.len();
    let loadings = Mat::zeros(expected_variants, n_components);
    let mut chunk_storage = vec![0.0f64; block_capacity * n_components];
    let inverse_singular: Vec<f64> = singular_values
        .iter()
        .map(|&sigma| if sigma > 0.0 { 1.0 / sigma } else { 0.0 })
        .collect();

    progress.on_stage_start(FitProgressStage::Loadings, expected_variants);

    let block_len = n_samples * block_capacity;
    let buffer_req = StackReq::new::<f64>(block_len).and(StackReq::new::<f64>(block_len));
    let mut mem = MemBuffer::new(buffer_req);
    let stack = MemStack::new(&mut mem);
    let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
    let buf0 =
        unsafe { std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len) };
    let (buf1_uninit, _) = stack.make_uninit::<f64>(block_len);
    let buf1 =
        unsafe { std::slice::from_raw_parts_mut(buf1_uninit.as_mut_ptr() as *mut f64, block_len) };
    let mut buffer_slices = [buf0, buf1];
    let [first_slice, second_slice] = &mut buffer_slices;
    let buffer_ptrs = [
        SendPtr(first_slice.as_mut_ptr()),
        SendPtr(second_slice.as_mut_ptr()),
    ];

    enum PrefetchMessage {
        Data {
            id: usize,
            filled: usize,
            start: usize,
        },
        End,
        Error(HwePcaError),
    }

    let buffer_count = buffer_ptrs.len();
    let (filled_tx, filled_rx) = sync_channel::<PrefetchMessage>(buffer_count);
    let (free_tx, free_rx) = sync_channel::<usize>(buffer_count);
    for id in 0..buffer_count {
        free_tx.send(id).expect("failed to seed loading buffers");
    }

    let observer = Arc::clone(progress);
    let block_capacity = block_capacity;
    let block_len = block_len;
    let expected_variants = expected_variants;

    let result = thread::scope(|scope| {
        let buffer_ptrs_prefetch = buffer_ptrs;
        let filled_sender = filled_tx;
        let free_receiver = free_rx;
        scope.spawn(move || {
            if let Err(err) = source.reset().map_err(|e| HwePcaError::Source(Box::new(e))) {
                let _ = filled_sender.send(PrefetchMessage::Error(err));
                return;
            }

            let mut start = 0usize;
            while let Ok(id) = free_receiver.recv() {
                let buffer_slice = unsafe {
                    std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                };
                let filled = match source.next_block_into(block_capacity, buffer_slice) {
                    Ok(filled) => filled,
                    Err(err) => {
                        let _ = filled_sender
                            .send(PrefetchMessage::Error(HwePcaError::Source(Box::new(err))));
                        break;
                    }
                };

                if filled == 0 {
                    let _ = filled_sender.send(PrefetchMessage::End);
                    break;
                }

                observer.on_stage_advance(
                    FitProgressStage::Loadings,
                    (start + filled).min(expected_variants),
                );

                if filled_sender
                    .send(PrefetchMessage::Data { id, filled, start })
                    .is_err()
                {
                    break;
                }
                start += filled;
            }
        });

        let free_sender = free_tx;
        let mut processed = 0usize;
        let buffer_ptrs_compute = buffer_ptrs;
        let mut loadings = loadings;
        while let Ok(message) = filled_rx.recv() {
            match message {
                PrefetchMessage::Data { id, filled, start } => {
                    if start != processed {
                        return Err(HwePcaError::InvalidInput(
                            "prefetch produced out-of-order variant ranges",
                        ));
                    }
                    if start + filled > expected_variants {
                        return Err(HwePcaError::InvalidInput(
                            "VariantBlockSource returned more variants than reported",
                        ));
                    }

                    let block_slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer_ptrs_compute[id].0, block_len)
                    };
                    let mut block = MatMut::from_column_major_slice_mut(
                        &mut block_slice[..n_samples * filled],
                        n_samples,
                        filled,
                    );
                    scaler.standardize_block(block.as_mut(), start..start + filled, par);
                    if let Some(weights) = ld_weights {
                        apply_ld_weights(block.as_mut(), start..start + filled, weights);
                    }

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
                        for (column, &inv_sigma) in
                            chunk_view.col_iter_mut().zip(inverse_singular.iter())
                        {
                            zip!(column).for_each(|unzip!(value)| {
                                *value *= inv_sigma;
                            });
                        }
                    }

                    loadings
                        .submatrix_mut(start, 0, filled, n_components)
                        .copy_from(chunk.as_ref());

                    processed = start + filled;

                    if free_sender.send(id).is_err() {
                        break;
                    }
                }
                PrefetchMessage::End => {
                    break;
                }
                PrefetchMessage::Error(err) => {
                    return Err(err);
                }
            }
        }

        if processed != expected_variants {
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource terminated early while computing loadings",
            ));
        }

        progress.on_stage_finish(FitProgressStage::Loadings);

        Ok(loadings)
    });

    result
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
        let mut state = serializer.serialize_struct("HwePcaModel", 10)?;
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
        state.serialize_field("ld", &self.ld)?;
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
            #[serde(default)]
            ld: Option<LdWeights>,
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
            ld: raw.ld,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::io::GenotypeDataset;
    use std::path::Path;

    #[test]
    fn fast_path_matches_masked_when_no_missingness() {
        let mut masked = (0..128)
            .map(|i| f64::from(i % 7) * 0.25)
            .collect::<Vec<_>>();
        let mut fast = masked.clone();
        let mean = 0.75;
        let inv = 0.5;

        standardize_column_simd(masked.as_mut_slice(), mean, inv);
        standardize_column_simd_full(fast.as_mut_slice(), mean, inv);

        for (lhs, rhs) in masked.iter().zip(fast.iter()) {
            assert!((lhs - rhs).abs() < 1e-15);
        }
    }

    #[test]
    fn negative_eigenvalues_do_not_produce_nan_scores() {
        let n_samples = 3;
        let eigenpairs = Eigenpairs {
            values: vec![-1.0e-12, 0.5],
            vectors: Mat::from_fn(n_samples, 2, |row, col| if row == col { 1.0 } else { 0.0 }),
        };

        let (singular_values, scores) = build_sample_scores(n_samples, &eigenpairs);

        assert_eq!(singular_values.len(), 2);
        assert!(singular_values.iter().all(|value| value.is_finite()));
        for row in 0..scores.nrows() {
            for col in 0..scores.ncols() {
                assert!(scores[(row, col)].is_finite());
            }
        }

        assert_eq!(singular_values[0], 0.0);
        for row in 0..scores.nrows() {
            assert_eq!(scores[(row, 0)], 0.0);
        }
    }

    #[test]
    fn variant_stats_cache_grows_lazily() {
        let block_capacity = 8;
        let hint = 1 << 15;
        let mut cache = VariantStatsCache::new(block_capacity, hint);
        let par = get_global_parallelism();
        let n_samples = 4;

        assert_eq!(cache.frequencies.len(), 0);
        assert_eq!(cache.scales.len(), 0);

        let first_block = Mat::from_fn(n_samples, 3, |row, col| (row + col) as f64);
        cache.ensure_statistics(first_block.as_ref(), 0..3, par);
        assert_eq!(cache.frequencies.len(), 3);
        assert_eq!(cache.scales.len(), 3);
        assert_eq!(cache.len(), 3);

        let second_block = Mat::from_fn(n_samples, 2, |row, col| (row + col + 1) as f64);
        cache.ensure_statistics(second_block.as_ref(), 3..5, par);
        assert_eq!(cache.frequencies.len(), 5);
        assert_eq!(cache.scales.len(), 5);
        assert_eq!(cache.len(), 5);

        cache.finalize();
        assert_eq!(cache.len(), 5);
        assert_eq!(cache.frequencies.len(), 5);
        assert_eq!(cache.scales.len(), 5);
    }

    #[test]
    fn variant_stats_cache_handles_zero_hint() {
        let block_capacity = 4;
        let mut cache = VariantStatsCache::new(block_capacity, 0);
        let par = get_global_parallelism();
        let n_samples = 3;
        let block = Mat::from_fn(n_samples, 2, |row, col| (row * 2 + col) as f64);

        cache.ensure_statistics(block.as_ref(), 0..2, par);
        assert_eq!(cache.frequencies.len(), 2);
        assert_eq!(cache.scales.len(), 2);

        cache.ensure_statistics(block.as_ref(), 2..4, par);
        assert_eq!(cache.frequencies.len(), 4);
        assert_eq!(cache.scales.len(), 4);
    }

    #[test]
    fn ld_weights_are_applied_during_standardization() {
        use std::sync::Arc;

        let scaler = HweScaler::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let dense_data = vec![0.0; 8];
        let mut source = DenseBlockSource::new(&dense_data, 4, 2).unwrap();
        let weights = Arc::from(vec![0.5, 2.0].into_boxed_slice());
        let operator: StandardizedCovarianceOp<'_, DenseBlockSource<'_>, NoopFitProgress> =
            StandardizedCovarianceOp::new(&mut source, 2, 2, 2, scaler, Some(weights));

        let mut block_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        {
            let mut block = MatMut::from_column_major_slice_mut(&mut block_data, 4, 2);
            operator.standardize_block_in_place(block.as_mut(), 0..2, get_global_parallelism());
        }

        let expected = vec![0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0];
        assert_eq!(block_data, expected);
    }

    #[test]
    fn ld_weights_are_ignored_when_absent() {
        let scaler = HweScaler::new(vec![0.0, 0.0], vec![1.0, 1.0]);
        let dense_data = vec![0.0; 8];
        let mut source = DenseBlockSource::new(&dense_data, 4, 2).unwrap();
        let operator: StandardizedCovarianceOp<'_, DenseBlockSource<'_>, NoopFitProgress> =
            StandardizedCovarianceOp::new(&mut source, 2, 2, 2, scaler, None);

        let mut block_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        {
            let mut block = MatMut::from_column_major_slice_mut(&mut block_data, 4, 2);
            operator.standardize_block_in_place(block.as_mut(), 0..2, get_global_parallelism());
        }

        let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        assert_eq!(block_data, expected);
    }

    const TEST_VCF_URL: &str = "https://raw.githubusercontent.com/SauersML/genomic_pca/refs/heads/main/tests/chr22_chunk.vcf.gz";
    const MAX_TEST_VARIANTS: usize = 32;
    const MAX_TEST_SAMPLES: usize = 8;
    const TEST_COMPONENTS: usize = 4;

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
            let variant_limit = if inner_variants == 0 {
                max_variants.max(1)
            } else {
                max_variants.max(1).min(inner_variants)
            };
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

    #[test]
    fn fit_hwe_pca_from_http_vcf_stream() {
        let path = Path::new(TEST_VCF_URL);
        let dataset = GenotypeDataset::open(path)
            .unwrap_or_else(|err| panic!("Failed to open dataset: {err}"));

        let block_source = dataset
            .block_source()
            .unwrap_or_else(|err| panic!("Failed to create block source: {err}"));

        let mut limited_source =
            LimitedBlockSource::new(block_source, MAX_TEST_SAMPLES, MAX_TEST_VARIANTS);
        let expected_variants = limited_source.n_variants();
        let expected_samples = limited_source.n_samples();

        let model = HwePcaModel::fit_k(&mut limited_source, TEST_COMPONENTS)
            .unwrap_or_else(|err| panic!("Failed to fit PCA model: {err}"));

        assert_eq!(expected_samples, model.n_samples());
        assert_eq!(expected_variants, model.n_variants());
        assert!(model.components() > 0);
    }
}
