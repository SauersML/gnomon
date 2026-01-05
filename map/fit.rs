use super::progress::{
    FitProgressObserver, FitProgressStage, NoopFitProgress, StageProgressHandle,
};
use super::variant_filter::VariantKey;
use core::cmp::{Ordering, min};
use core::fmt;
use dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::matmul::matmul;
use faer::linalg::solvers::{Llt as FaerLlt, Solve as FaerSolve};
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
use std::simd::num::SimdFloat;
use std::simd::{LaneCount, Simd, SupportedLaneCount};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::thread;

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
pub const DEFAULT_BLOCK_WIDTH: usize = 2_048;
pub const DEFAULT_LD_WINDOW: usize = 51;
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

#[derive(Clone, Debug, Default)]
pub struct FitOptions {
    pub ld: Option<LdConfig>,
}

#[derive(Clone, Debug)]
pub enum LdWindow {
    Sites(usize),
    BasePairs(u64),
}

#[derive(Clone, Debug, Default)]
pub struct LdConfig {
    pub window: Option<LdWindow>,
    pub ridge: Option<f64>,
    pub variant_keys: Option<Arc<Vec<VariantKey>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LdWeights {
    pub weights: Vec<f64>,
    pub window: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bp_window: Option<u64>,
    pub ridge: f64,
}

#[derive(Clone, Debug)]
struct LdBpWindowRange {
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
enum LdResolvedWindow {
    Sites {
        size: usize,
    },
    BasePairs {
        span_bp: u64,
        ranges: Arc<[LdBpWindowRange]>,
        capacity: usize,
    },
}

#[derive(Clone, Debug)]
struct LdResolvedConfig {
    window: LdResolvedWindow,
    ridge: f64,
}

impl LdResolvedConfig {
    fn window_capacity(&self) -> usize {
        match &self.window {
            LdResolvedWindow::Sites { size } => *size,
            LdResolvedWindow::BasePairs { capacity, .. } => *capacity,
        }
    }

    fn bp_window(&self) -> Option<u64> {
        match &self.window {
            LdResolvedWindow::Sites { .. } => None,
            LdResolvedWindow::BasePairs { span_bp, .. } => Some(*span_bp),
        }
    }
}

impl FitOptions {
    fn resolved_ld(
        &self,
        observed_variants: usize,
    ) -> Result<Option<LdResolvedConfig>, HwePcaError> {
        let Some(cfg) = &self.ld else {
            return Ok(None);
        };

        if observed_variants == 0 {
            return Ok(None);
        }

        let window_spec = cfg
            .window
            .clone()
            .unwrap_or(LdWindow::Sites(DEFAULT_LD_WINDOW));

        let ridge = cfg.ridge.unwrap_or(DEFAULT_LD_RIDGE);
        if !(ridge.is_finite() && ridge > 0.0) {
            return Err(HwePcaError::InvalidInput(
                "LD weighting ridge must be positive and finite",
            ));
        }

        let window = match window_spec {
            LdWindow::Sites(mut window) => {
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
                LdResolvedWindow::Sites { size: window }
            }
            LdWindow::BasePairs(span_bp) => {
                let keys = cfg.variant_keys.as_ref().ok_or_else(|| {
                    HwePcaError::InvalidInput("LD base-pair window requires variant positions")
                })?;
                if keys.len() != observed_variants {
                    return Err(HwePcaError::InvalidInput(
                        "LD base-pair window requires positions for all variants",
                    ));
                }
                let (ranges, capacity) = compute_ld_bp_ranges(keys, span_bp)?;
                LdResolvedWindow::BasePairs {
                    span_bp,
                    ranges,
                    capacity,
                }
            }
        };

        Ok(Some(LdResolvedConfig { window, ridge }))
    }
}

fn compute_ld_bp_ranges(
    keys: &[VariantKey],
    span_bp: u64,
) -> Result<(Arc<[LdBpWindowRange]>, usize), HwePcaError> {
    if keys.is_empty() {
        return Err(HwePcaError::InvalidInput(
            "LD base-pair window requires at least one variant",
        ));
    }

    let half_span = span_bp / 2;

    let mut left_bounds = vec![0usize; keys.len()];
    let mut start = 0usize;

    for (idx, key) in keys.iter().enumerate() {
        if idx == 0 {
            left_bounds[idx] = 0;
            continue;
        }

        if keys[idx - 1].chromosome != key.chromosome {
            start = idx;
        }

        while start < idx {
            let candidate = &keys[start];
            if candidate.chromosome != key.chromosome {
                start += 1;
                continue;
            }
            let delta = key.position.saturating_sub(candidate.position);
            if delta > half_span {
                start += 1;
                continue;
            }
            break;
        }

        left_bounds[idx] = start.min(idx);
    }

    let mut ranges = Vec::with_capacity(keys.len());
    let mut capacity = 1usize;
    let mut right = 0usize;

    for (idx, key) in keys.iter().enumerate() {
        if right < idx {
            right = idx;
        }

        while right < keys.len() {
            let candidate = &keys[right];
            if candidate.chromosome != key.chromosome {
                break;
            }
            let delta = candidate.position.saturating_sub(key.position);
            if delta > half_span {
                break;
            }
            right += 1;
        }

        let start_idx = left_bounds[idx].min(idx);
        let end_idx = right.max(idx + 1);
        let width = end_idx - start_idx;
        capacity = capacity.max(width.max(1));
        ranges.push(LdBpWindowRange {
            start: start_idx,
            end: end_idx,
        });
    }

    Ok((ranges.into_boxed_slice().into(), capacity.max(1)))
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f64);

// SAFETY: `SendPtr` is only ever constructed from buffers that are owned by the
// current thread and remain alive for the entire duration of the scoped thread
// in which the pointer is sent. We only move the raw pointer between threads to
// avoid borrow checker restrictions; the pointed-to memory is still uniquely
// owned, and channel coordination guarantees that at most one thread accesses a
// given buffer at a time.
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
        None
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        None
    }

    /// Returns per-variant imputation quality scores for the most recently fetched block.
    /// Quality values should be in [0, 1] range where:
    /// - 1.0 = perfectly genotyped (hard call, no imputation uncertainty)
    /// - 0.0 = completely uncertain (equivalent to missing)
    /// - 0.0-1.0 = imputed with INFO/DR2/R² quality score
    ///
    /// The storage slice should have at least `filled` elements (from last next_block_into).
    /// Default implementation returns 1.0 for all variants (assumes hard calls).
    fn variant_quality(&self, filled: usize, storage: &mut [f64]) {
        for value in storage.iter_mut().take(filled) {
            *value = 1.0;
        }
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
        let scales = &self.scales[start..end];
        standardize_block_with_mask_from_stats(block, presence_out, freqs, scales, par);
    }
}

fn standardize_block_with_mask_from_stats(
    block: MatMut<'_, f64>,
    presence_out: MatMut<'_, f64>,
    freqs: &[f64],
    scales: &[f64],
    par: Par,
) {
    let filled = freqs.len();

    debug_assert_eq!(filled, block.ncols());
    debug_assert_eq!(filled, presence_out.ncols());
    debug_assert_eq!(block.nrows(), presence_out.nrows());
    debug_assert_eq!(filled, scales.len());

    let mut block = block.subcols_mut(0, filled);
    let mut presence_out = presence_out.subcols_mut(0, filled);

    let apply_standardization =
        |column: ColMut<'_, f64>, presence_col: ColMut<'_, f64>, mean: f64, inv: f64| {
            let contiguous_values = column
                .try_as_col_major_mut()
                .expect("projection block column must be contiguous");
            let contiguous_mask = presence_col
                .try_as_col_major_mut()
                .expect("projection mask column must be contiguous");
            let values = contiguous_values.as_slice_mut();
            let mask = contiguous_mask.as_slice_mut();
            standardize_column_with_mask(values, mask, mean, inv);
        };

    let use_parallel = filled >= 32 && par.degree() > 1;

    if use_parallel {
        presence_out
            .par_col_iter_mut()
            .zip(block.par_col_iter_mut())
            .enumerate()
            .for_each(|(idx, (presence_col, column))| {
                let freq = freqs[idx];
                let scale = scales[idx];
                let mean = 2.0 * freq;
                let denom = scale.max(HWE_SCALE_FLOOR);
                let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
                apply_standardization(column, presence_col, mean, inv);
            });
    } else {
        for idx in 0..filled {
            let presence_col = presence_out.rb_mut().col_mut(idx);
            let column = block.rb_mut().col_mut(idx);
            let freq = freqs[idx];
            let scale = scales[idx];
            let mean = 2.0 * freq;
            let denom = scale.max(HWE_SCALE_FLOOR);
            let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
            apply_standardization(column, presence_col, mean, inv);
        }
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

pub(crate) fn apply_ld_weights(
    block: MatMut<'_, f64>,
    variant_range: Range<usize>,
    weights: &[f64],
) {
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

        assert!(variant_range.start == self.write_pos);

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

        assert!(self.frequencies.len() >= end);
        assert!(self.scales.len() >= end);
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
            .extend(std::iter::repeat_n(0.0, additional));
        self.scales.extend(std::iter::repeat_n(0.0, additional));
    }
}

#[derive(Clone, Copy, Debug)]
enum SimdLaneSelection {
    Lanes4,
    Lanes2,
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
const _: () = {
    // Ensure the two-lane variant stays in use on targets that never select it at runtime.
    let _ = SimdLaneSelection::Lanes2;
};

#[inline(always)]
fn record_simd_lane_diagnostic(
    stage: &'static str,
    selection: SimdLaneSelection,
) -> SimdLaneSelection {
    log::debug!("SIMD lane selection stage {stage} -> {selection:?}");
    selection
}

fn detected_simd_lane_selection() -> SimdLaneSelection {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        static DETECTED: OnceLock<SimdLaneSelection> = OnceLock::new();
        return *DETECTED.get_or_init(|| {
            let selection =
                if cfg!(target_feature = "avx") && std::arch::is_x86_feature_detected!("avx") {
                    SimdLaneSelection::Lanes4
                } else {
                    SimdLaneSelection::Lanes2
                };
            record_simd_lane_diagnostic("x86 runtime detection", selection)
        });
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        record_simd_lane_diagnostic(
            "default lanes4 architecture",
            SimdLaneSelection::Lanes4,
        )
    }

    #[cfg(not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        return record_simd_lane_diagnostic("portable fallback", SimdLaneSelection::Lanes2);
    }
}

#[inline(always)]
fn standardize_column_simd(values: &mut [f64], mean: f64, inv: f64) {
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            standardize_column_simd_lanes4(values, mean, inv);
        }
        _ => standardize_column_simd_impl::<2>(values, mean, inv),
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn standardize_column_simd_lanes4(values: &mut [f64], mean: f64, inv: f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: The AVX-specific implementation is only compiled on x86 targets
    // and this branch is taken after `detected_simd_lane_selection` confirmed
    // that the CPU supports the required feature set.
    unsafe {
        standardize_column_simd_avx(values, mean, inv);
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        standardize_column_simd_impl::<4>(values, mean, inv);
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
/// # Safety
/// The caller must ensure the current CPU supports AVX instructions. All call
/// sites guard this by checking `std::arch::is_x86_feature_detected!("avx")` or
/// by only invoking it in configurations where AVX is guaranteed to be present.
unsafe fn standardize_column_simd_avx(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_impl::<4>(values, mean, inv);
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

#[inline(always)]
fn standardize_column_with_mask(values: &mut [f64], mask: &mut [f64], mean: f64, inv: f64) {
    debug_assert_eq!(values.len(), mask.len());

    if inv == 0.0 {
        for (value, mask_value) in values.iter_mut().zip(mask.iter_mut()) {
            let raw = *value;
            *mask_value = if raw.is_finite() { 1.0 } else { 0.0 };
            *value = 0.0;
        }
        return;
    }

    standardize_column_with_mask_simd(values, mask, mean, inv);
}

#[inline(always)]
fn standardize_column_with_mask_simd(values: &mut [f64], mask: &mut [f64], mean: f64, inv: f64) {
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            standardize_column_with_mask_simd_lanes4(values, mask, mean, inv);
        }
        _ => standardize_column_with_mask_simd_impl::<2>(values, mask, mean, inv),
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn standardize_column_with_mask_simd_lanes4(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: On x86 we only reach this branch when runtime detection selects
    // the four-lane configuration, which implies AVX availability.
    unsafe {
        standardize_column_with_mask_simd_avx(values, mask, mean, inv);
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        standardize_column_with_mask_simd_impl::<4>(values, mask, mean, inv);
    }
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
/// # Safety
/// The caller must ensure the CPU supports AVX instructions. All invocations
/// are conditioned on runtime feature detection or target configurations that
/// guarantee AVX availability.
unsafe fn standardize_column_with_mask_simd_avx(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) {
    standardize_column_with_mask_simd_impl::<4>(values, mask, mean, inv);
}

#[inline(always)]
fn standardize_column_with_mask_simd_impl<const LANES: usize>(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mean_simd = Simd::<f64, LANES>::splat(mean);
    let inv_simd = Simd::<f64, LANES>::splat(inv);
    let zero = Simd::<f64, LANES>::splat(0.0);
    let one = Simd::<f64, LANES>::splat(1.0);

    let (value_chunks, value_remainder) = values.as_chunks_mut::<LANES>();
    let (mask_chunks, mask_remainder) = mask.as_chunks_mut::<LANES>();

    debug_assert_eq!(value_chunks.len(), mask_chunks.len());
    debug_assert_eq!(value_remainder.len(), mask_remainder.len());

    for (value_chunk, mask_chunk) in value_chunks.iter_mut().zip(mask_chunks.iter_mut()) {
        let lane = Simd::<f64, LANES>::from_array(*value_chunk);
        let finite_mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let result = finite_mask.select(standardized, zero);
        *value_chunk = result.to_array();
        let mask_values = finite_mask.select(one, zero);
        *mask_chunk = mask_values.to_array();
    }

    for (value, mask_value) in value_remainder.iter_mut().zip(mask_remainder.iter_mut()) {
        let raw = *value;
        if raw.is_finite() {
            *mask_value = 1.0;
            *value = (raw - mean) * inv;
        } else {
            *mask_value = 0.0;
            *value = 0.0;
        }
    }
}

#[cfg(test)]
#[inline(always)]
fn standardize_column_simd_full(values: &mut [f64], mean: f64, inv: f64) {
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            standardize_column_simd_full_lanes4(values, mean, inv);
        }
        _ => standardize_column_simd_full_impl::<2>(values, mean, inv),
    }
}

#[cfg(test)]
#[inline(always)]
fn standardize_column_simd_full_lanes4(values: &mut [f64], mean: f64, inv: f64) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    // SAFETY: The AVX path is only taken on x86 targets when the runtime lane
    // selection confirmed AVX support.
    unsafe {
        standardize_column_simd_full_avx(values, mean, inv);
    }

    #[cfg(all(
        not(any(target_arch = "x86", target_arch = "x86_64")),
        any(target_arch = "aarch64", target_arch = "wasm32")
    ))]
    {
        standardize_column_simd_full_impl::<4>(values, mean, inv);
    }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
/// # Safety
/// Callers must guarantee AVX availability; runtime dispatch ensures that the
/// function is only invoked when the CPU advertises the capability.
unsafe fn standardize_column_simd_full_avx(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_full_impl::<4>(values, mean, inv);
}

#[cfg(test)]
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
    match detected_simd_lane_selection() {
        #[cfg(any(
            target_feature = "avx",
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        SimdLaneSelection::Lanes4 => {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            // SAFETY: Runtime detection guaranteed AVX is present before
            // dispatching to the specialized implementation.
            unsafe {
                log::debug!("Invoking AVX sum_and_count_finite implementation");
                return sum_and_count_finite_avx(values);
            }

            #[cfg(all(
                not(any(target_arch = "x86", target_arch = "x86_64")),
                any(target_arch = "aarch64", target_arch = "wasm32")
            ))]
            {
                log::debug!(
                    "Using generic four-lane sum_and_count_finite implementation for non-x86 architecture"
                );
                sum_and_count_finite_impl::<4>(values)
            }

            #[cfg(not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            )))]
            {
                log::warn!(
                    "Falling back to two-lane sum_and_count_finite implementation despite four-lane selection"
                );
                return sum_and_count_finite_impl::<2>(values);
            }
        }
        _ => sum_and_count_finite_impl::<2>(values),
    }
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

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
#[target_feature(enable = "avx")]
/// # Safety
/// Callers must ensure AVX is supported by the running CPU. Runtime feature
/// checks protect all invocations of this function.
unsafe fn sum_and_count_finite_avx(values: &[f64]) -> (f64, usize) {
    sum_and_count_finite_impl::<4>(values)
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
    component_weighted_norms_sq: Vec<f64>,
    variant_keys: Option<Vec<VariantKey>>,
    ld: Option<LdWeights>,
    genome_build: Option<String>,
}

impl HwePcaModel {
    pub fn fit_k<S>(source: &mut S, components: usize) -> Result<Self, HwePcaError>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
    {
        let progress = Arc::new(NoopFitProgress);
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
        P: FitProgressObserver + Send + Sync + 'static,
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
        P: FitProgressObserver + Send + Sync + 'static,
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
        let ld_hint = if n_variants_hint > 0 {
            n_variants_hint
        } else if let Some(keys) = options
            .ld
            .as_ref()
            .and_then(|cfg| cfg.variant_keys.as_ref())
        {
            keys.len()
        } else {
            0
        };

        let ld_config = options.resolved_ld(ld_hint)?;

        // Use the new fused implementation for both LD and non-LD paths
        let (scaler, observed_variants, covariance, ld_weights) = if let Some(ld_cfg) = ld_config {
            // With LD weights - compute them first, then use fused pass
            let (_, _, ld_weights_computed) = compute_stats_and_ld_weights(
                source,
                block_capacity,
                ld_cfg,
                n_variants_hint,
                progress,
                par,
            )?;

            let ld_weights_arc = Arc::<[f64]>::from(ld_weights_computed.weights.clone().into_boxed_slice());

            // Reset source for fused pass
            source.reset().map_err(|err| HwePcaError::Source(Box::new(err)))?;

            let (scaler, observed, covariance) = compute_stats_and_covariance_blockwise(
                source,
                block_capacity,
                par,
                progress,
                n_variants_hint,
                Some(ld_weights_arc),
            )?;

            (scaler, observed, covariance, Some(ld_weights_computed))
        } else {
            // No LD weights - use fused pass directly
            let (scaler, observed, covariance) = compute_stats_and_covariance_blockwise(
                source,
                block_capacity,
                par,
                progress,
                n_variants_hint,
                None,
            )?;

            (scaler, observed, covariance, None)
        };

        let ld_weights_arc = ld_weights
            .as_ref()
            .map(|ld| Arc::<[f64]>::from(ld.weights.clone().into_boxed_slice()));

        // Now decompose the covariance matrix using faer's self-adjoint eigendecomposition
        let decomposition = {
            let eig_result = covariance.as_ref().self_adjoint_eigen(Side::Upper);
            match eig_result {
                Ok(eig) => {
                    let eigenvalues_diag = eig.S();
                    let eigenvectors_mat = eig.U();
                    let n_eig = eigenvalues_diag.dim();

                    // Create index-value pairs for sorting (descending by eigenvalue)
                    let mut indexed_values: Vec<(usize, f64)> = (0..n_eig)
                        .map(|i| (i, eigenvalues_diag[i]))
                        .collect();

                    // Select top k components by eigenvalue magnitude
                    let kept = select_top_k_desc(&mut indexed_values, target_components);

                    // Extract selected values and vectors
                    let selected_values: Vec<f64> = indexed_values[..kept].iter().map(|(_, val)| *val).collect();
                    let selected_vectors = Mat::from_fn(eigenvectors_mat.nrows(), kept, |row, col| {
                        let original_col = indexed_values[col].0;
                        eigenvectors_mat[(row, original_col)]
                    });

                    Eigenpairs {
                        values: selected_values,
                        vectors: selected_vectors,
                    }
                }
                Err(e) => {
                    return Err(HwePcaError::Eigen(format!(
                        "Eigendecomposition failed: {:?}",
                        e
                    )));
                }
            }
        };

        let variant_count = scaler.variant_scales().len();
        debug_assert_eq!(variant_count, observed_variants);
        if variant_count == 0 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least one variant",
            ));
        }

        log::info!("Observed {} variants during fused pass", variant_count);

        if decomposition.values.is_empty() {
            return Err(HwePcaError::Eigen(
                "All eigenvalues are numerically zero; increase cohort size or review input data"
                    .into(),
            ));
        }

        let (mut singular_values, mut sample_scores) =
            build_sample_scores(n_samples, &decomposition);

        let mut loadings = compute_variant_loadings(
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

        let component_weighted_norms_sq = renormalize_variant_loadings(
            loadings.as_mut(),
            &mut singular_values,
            sample_scores.as_mut(),
        );

        Ok(Self {
            n_samples,
            n_variants: variant_count,
            scaler,
            eigenvalues: decomposition.values,
            singular_values,
            sample_basis: decomposition.vectors,
            sample_scores,
            loadings,
            component_weighted_norms_sq,
            variant_keys: None,
            ld: ld_weights,
            genome_build: None,
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

    /// Returns the singular values after renormalizing the variant loadings.
    ///
    /// These values remain consistent with [`HwePcaModel::variant_loadings`] and
    /// [`HwePcaModel::sample_scores`] after the post-fit rescaling step. Use
    /// [`HwePcaModel::canonical_singular_values`] or [`HwePcaModel::explained_variance`]
    /// when deriving explained variance in the classical PCA metric.
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// Returns the canonical singular values that satisfy σ²/(n−1)=λ.
    pub fn canonical_singular_values(&self) -> Vec<f64> {
        let scale = self.n_samples.saturating_sub(1) as f64;
        if scale == 0.0 {
            return vec![0.0; self.eigenvalues.len()];
        }

        self.eigenvalues
            .iter()
            .map(|&lambda| (lambda * scale).max(0.0).sqrt())
            .collect()
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

    pub fn component_weighted_norms_sq(&self) -> &[f64] {
        &self.component_weighted_norms_sq
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

    pub fn genome_build(&self) -> Option<&str> {
        self.genome_build.as_deref()
    }

    pub fn set_genome_build(&mut self, build: Option<String>) {
        self.genome_build = build;
    }
}

struct Eigenpairs {
    values: Vec<f64>,
    vectors: Mat<f64>,
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

/// Computes both variant statistics and the covariance matrix in a single pass.
/// This is a mathematically exact optimization that eliminates the need for separate passes.
///
/// Note: LD weights require a sliding window buffer for proper computation across blocks.
/// The simple LD weight application here assumes weights are pre-computed or that blocks
/// are large enough to contain the full LD window.
fn compute_stats_and_covariance_blockwise<S, P>(
    source: &mut S,
    block_capacity: usize,
    par: Par,
    progress: &Arc<P>,
    n_variants_hint: usize,
    ld_weights: Option<Arc<[f64]>>,
) -> Result<(HweScaler, usize, Mat<f64>), HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync,
{
    let n_samples = source.n_samples();

    // Initialize statistics accumulator
    let mut stats = VariantStatsCache::new(block_capacity, n_variants_hint);

    // Initialize covariance matrix accumulator
    let mut covariance = Mat::<f64>::zeros(n_samples, n_samples);

    // Block storage - reused for each block
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];

    // Start progress tracking for combined pass
    progress.on_stage_start(FitProgressStage::AlleleStatistics, n_variants_hint);
    let stats_progress = StageProgressHandle::new(Arc::clone(progress), FitProgressStage::AlleleStatistics);

    source
        .reset()
        .map_err(|err| HwePcaError::Source(Box::new(err)))?;

    let mut processed = 0usize;
    let mut used_source_progress = false;

    // Single pass over all variants
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

        let variant_range = processed..processed + filled;

        // Step 1: Compute statistics for this block (handles NaN counting)
        stats.ensure_statistics(block.as_ref(), variant_range.clone(), par);

        // Step 2: Standardize block in-place using computed statistics
        // This transforms raw genotypes (0,1,2,NaN) to standardized values
        // The standardize_block_impl already handles NaN via SIMD masks (zeroing non-finite values)
        let freqs = &stats.frequencies[variant_range.clone()];
        let scales = &stats.scales[variant_range.clone()];
        standardize_block_impl(block.rb_mut(), freqs, scales, par);

        // Step 3: Apply LD weights if available
        // TODO: For proper LD weight computation across blocks, implement sliding window buffer
        if let Some(weights) = &ld_weights {
            apply_ld_weights(block.rb_mut(), variant_range.clone(), weights);
        }

        // Step 5: Accumulate covariance using optimized GEMM
        // Cov += Block × Block^T
        matmul(
            covariance.as_mut(),
            Accum::Add,
            block.as_ref(),
            block.as_ref().transpose(),
            1.0,
            par,
        );

        processed += filled;

        // Update progress
        if let Some((bytes_read, total_bytes)) = source.progress_bytes() {
            used_source_progress = true;
            stats_progress.advance_bytes(bytes_read, total_bytes);
        } else if let Some((work_done, total_work)) = source.progress_variants() {
            used_source_progress = true;
            if let Some(total) = total_work {
                stats_progress.set_total(total);
            } else if n_variants_hint > 0 {
                stats_progress.estimate(n_variants_hint);
            }
            stats_progress.advance(work_done);
        } else {
            stats_progress.advance(processed);
        }
    }

    if processed == 0 {
        stats_progress.finish();
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource yielded no variants",
        ));
    }

    // Finalize progress
    if let Some((_, Some(total))) = source.progress_variants() {
        stats_progress.set_total(total);
    } else if !used_source_progress {
        stats_progress.set_total(processed);
    }
    stats_progress.finish();

    // Finalize statistics
    stats.finalize();
    let scaler = stats
        .into_scaler()
        .expect("finalized statistics must produce a scaler");

    // Mark Gram matrix stage as complete (it was done during the combined pass)
    progress.on_stage_start(FitProgressStage::GramMatrix, 0);
    progress.on_stage_finish(FitProgressStage::GramMatrix);

    Ok((scaler, processed, covariance))
}

fn compute_stats_and_ld_weights<S, P>(
    source: &mut S,
    block_capacity: usize,
    config: LdResolvedConfig,
    n_variants_hint: usize,
    progress: &Arc<P>,
    par: Par,
) -> Result<(HweScaler, usize, LdWeights), HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let n_samples = source.n_samples();
    let mut stats = VariantStatsCache::new(block_capacity, n_variants_hint);
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];
    let mut presence_storage = vec![0.0f64; n_samples * block_capacity];
    let window_capacity = config.window_capacity().max(1);
    let mut ring = LdRingBuffer::new(n_samples, window_capacity);
    let mut weights: Vec<f64> = Vec::with_capacity(n_variants_hint.max(block_capacity));
    let mut next_weight = 0usize;

    progress.on_stage_start(FitProgressStage::AlleleStatistics, n_variants_hint);
    let stats_progress =
        StageProgressHandle::new(Arc::clone(progress), FitProgressStage::AlleleStatistics);

    progress.on_stage_start(FitProgressStage::LdWeights, n_variants_hint);
    let ld_progress = StageProgressHandle::new(Arc::clone(progress), FitProgressStage::LdWeights);

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
        stats.ensure_statistics(block.as_ref(), variant_range.clone(), par);

        let freqs = &stats.frequencies[variant_range.clone()];
        let scales = &stats.scales[variant_range.clone()];
        standardize_block_with_mask_from_stats(
            block.as_mut(),
            presence.as_mut(),
            freqs,
            scales,
            par,
        );

        weights.resize(processed + filled, 1.0);

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
                &config,
                &ld_progress,
                par,
            )?;
        }

        processed += filled;

        if let Some((bytes_read, total_bytes)) = source.progress_bytes() {
            used_source_progress = true;
            stats_progress.advance_bytes(bytes_read, total_bytes);
        } else if let Some((work_done, total_work)) = source.progress_variants() {
            used_source_progress = true;
            if let Some(total) = total_work {
                stats_progress.set_total(total);
            } else if n_variants_hint > 0 {
                stats_progress.estimate(n_variants_hint);
            }
            stats_progress.advance(work_done);
        } else {
            stats_progress.advance(processed);
        }
    }

    assign_ready_weights(
        &mut ring,
        &mut weights,
        &mut next_weight,
        &config,
        &ld_progress,
        par,
    )?;

    if processed == 0 {
        stats_progress.finish();
        ld_progress.finish();
        return Err(HwePcaError::InvalidInput(
            "VariantBlockSource yielded no variants",
        ));
    }

    if let Some((_, Some(total))) = source.progress_variants() {
        stats_progress.set_total(total);
    } else if !used_source_progress {
        stats_progress.set_total(processed);
    }

    stats.finalize();
    let scaler = stats
        .into_scaler()
        .expect("finalized statistics must produce a scaler");
    stats_progress.finish();

    ld_progress.set_total(processed);
    ld_progress.finish();

    let weights = LdWeights {
        weights,
        window: config.window_capacity().max(1),
        bp_window: config.bp_window(),
        ridge: config.ridge,
    };

    Ok((scaler, processed, weights))
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

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn len(&self) -> usize {
        self.len
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
        
        if self.len < capacity {
            let slot = (self.start + self.len) % capacity;
            self.len += 1;
            slot
        } else {
            let slot = self.start;
            self.start = (self.start + 1) % capacity;
            slot
        }
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

    fn window<'a>(
        &'a self,
        start: usize,
        len: usize,
        scratch: &'a mut LdWindowScratch,
    ) -> LdWindowView<'a> {
        assert!(start + len <= self.len);
        let capacity = self.capacity();
        if capacity == 0 || len == 0 {
            return LdWindowView {
                values: scratch.values.as_ref().submatrix(0, 0, self.n_samples, 0),
                masks: &[],
            };
        }

        let start_slot = self.slot_at(start);
        let contiguous = start_slot + len <= capacity;

        if contiguous {
            let mask_start = start_slot * self.n_samples;
            return LdWindowView {
                values: self
                    .values
                    .as_ref()
                    .submatrix(0, start_slot, self.n_samples, len),
                masks: &self.masks[mask_start..mask_start + self.n_samples * len],
            };
        }

        let mut dst_values = scratch
            .values
            .as_mut()
            .submatrix_mut(0, 0, self.n_samples, len);
        let dst_masks = &mut scratch.masks[..self.n_samples * len];

        for offset in 0..len {
            let slot = self.slot_at(start + offset);
            {
                let src = self.values.as_ref().col(slot);
                let dst = dst_values.rb_mut().col_mut(offset);
                zip!(dst, src).for_each(|unzip!(dst, src)| {
                    *dst = *src;
                });
            }
            {
                let src = self.mask_slice(slot);
                let dst = &mut dst_masks[offset * self.n_samples..(offset + 1) * self.n_samples];
                dst.copy_from_slice(src);
            }
        }

        let values_view = scratch.values.as_ref().submatrix(0, 0, self.n_samples, len);

        LdWindowView {
            values: values_view,
            masks: &scratch.masks[..self.n_samples * len],
        }
    }
}

struct LdWindowView<'a> {
    values: MatRef<'a, f64>,
    masks: &'a [u8],
}

struct LdWindowScratch {
    values: Mat<f64>,
    masks: Vec<u8>,
}

struct LdThreadScratch {
    window: LdWindowScratch,
    mask_f64: Mat<f64>,
    gram: Mat<f64>,
    counts: Mat<f64>,
    sums: Mat<f64>,
    squared_sums: Mat<f64>,
    system: Mat<f64>,
    rhs: Mat<f64>,
    sums_vec: Vec<f64>,
    squared_vec: Vec<f64>,
}

impl LdThreadScratch {
    fn new(n_samples: usize, window_capacity: usize) -> Self {
        Self {
            window: LdWindowScratch {
                values: Mat::zeros(n_samples, window_capacity),
                masks: vec![0u8; n_samples * window_capacity],
            },
            mask_f64: Mat::zeros(n_samples, window_capacity),
            gram: Mat::zeros(window_capacity, window_capacity),
            counts: Mat::zeros(window_capacity, window_capacity),
            sums: Mat::zeros(window_capacity, window_capacity),
            squared_sums: Mat::zeros(window_capacity, window_capacity),
            system: Mat::zeros(window_capacity, window_capacity),
            rhs: Mat::zeros(window_capacity, 1),
            sums_vec: vec![0.0; window_capacity],
            squared_vec: vec![0.0; window_capacity],
        }
    }
}

#[cfg(test)]
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
    P: FitProgressObserver + Send + Sync + 'static,
{
    let mut weights = vec![1.0; observed_variants];
    progress.on_stage_start(FitProgressStage::LdWeights, observed_variants);
    let stage_progress =
        StageProgressHandle::new(Arc::clone(progress), FitProgressStage::LdWeights);

    if observed_variants == 0 {
        stage_progress.finish();
        return Ok(LdWeights {
            weights,
            window: config.window_capacity().max(1),
            bp_window: config.bp_window(),
            ridge: config.ridge,
        });
    }

    let n_samples = source.n_samples();
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];
    let mut presence_storage = vec![0.0f64; n_samples * block_capacity];
    let window_capacity = config.window_capacity().max(1);
    let mut ring = LdRingBuffer::new(n_samples, window_capacity);
    let mut next_weight = 0usize;

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
                &config,
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
        &config,
        &stage_progress,
        par,
    )?;

    stage_progress.set_total(observed_variants);
    stage_progress.finish();

    Ok(LdWeights {
        weights,
        window: config.window_capacity().max(1),
        bp_window: config.bp_window(),
        ridge: config.ridge,
    })
}

#[derive(Clone, Copy)]
struct LdWeightJob {
    start_offset: usize,
    window_len: usize,
    center: usize,
    keep_from: usize,
}

fn assign_ready_weights<P: FitProgressObserver>(
    ring: &mut LdRingBuffer,
    weights: &mut [f64],
    next_weight: &mut usize,
    config: &LdResolvedConfig,
    progress: &StageProgressHandle<P>,
    par: Par,
) -> Result<(), HwePcaError> {
    let window_capacity = config.window_capacity().max(1);

    while *next_weight < weights.len() {
        if ring.len() == 0 {
            break;
        }

        let jobs = collect_ready_jobs(ring, *next_weight, config, window_capacity)?;
        if jobs.is_empty() {
            break;
        }

        let start_idx = *next_weight;
        let end_idx = start_idx + jobs.len();
        {
            let ring_ref: &LdRingBuffer = &*ring;
            let weight_slice = &mut weights[start_idx..end_idx];
            weight_slice
                .par_iter_mut()
                .zip_eq(jobs.clone().into_par_iter())
                .map_init(
                    || LdThreadScratch::new(ring_ref.n_samples(), window_capacity),
                    |scratch, (slot, job)| {
                        let weight = compute_ld_weight(
                            ring_ref,
                            job.start_offset,
                            job.window_len,
                            job.center,
                            config.ridge,
                            scratch,
                            par,
                        );
                        *slot = weight;
                    },
                )
                .for_each(|_| {});
        }

        progress.advance(end_idx);
        *next_weight = end_idx;
        if let Some(last_keep) = jobs.last().map(|job| job.keep_from) {
            ring.truncate_front(last_keep);
        }
    }

    Ok(())
}

fn compute_ld_weight(
    ring: &LdRingBuffer,
    start: usize,
    window_len: usize,
    center: usize,
    ridge: f64,
    scratch: &mut LdThreadScratch,
    par: Par,
) -> f64 {
    let window = ring.window(start, window_len, &mut scratch.window);
    let values = window.values;
    let masks = window.masks;

    let mut mask_mat = scratch
        .mask_f64
        .as_mut()
        .submatrix_mut(0, 0, ring.n_samples(), window_len);
    for col in 0..window_len {
        let src = &masks[col * ring.n_samples()..(col + 1) * ring.n_samples()];
        let dst = mask_mat.rb_mut().col_mut(col);
        for (dst, &src_val) in dst.iter_mut().zip(src.iter()) {
            *dst = src_val as f64;
        }
    }
    let mask_view = mask_mat.as_ref();

    let mut gram = scratch
        .gram
        .as_mut()
        .submatrix_mut(0, 0, window_len, window_len);
    gram.fill(0.0);
    matmul(
        gram.as_mut(),
        Accum::Replace,
        values.transpose(),
        values,
        1.0,
        par,
    );

    let mut counts = scratch
        .counts
        .as_mut()
        .submatrix_mut(0, 0, window_len, window_len);
    counts.fill(0.0);
    matmul(
        counts.as_mut(),
        Accum::Replace,
        mask_view.transpose(),
        mask_view,
        1.0,
        par,
    );

    let sums_vec = &mut scratch.sums_vec[..window_len];
    let squared_vec = &mut scratch.squared_vec[..window_len];

    for col in 0..window_len {
        let values_slice = values
            .col(col)
            .try_as_col_major()
            .expect("LD window column must be contiguous")
            .as_slice();
        let mask_slice = &masks[col * ring.n_samples()..(col + 1) * ring.n_samples()];
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        for (&value, &mask) in values_slice.iter().zip(mask_slice.iter()) {
            if mask != 0 {
                sum += value;
                sq_sum += value * value;
            }
        }
        sums_vec[col] = sum;
        squared_vec[col] = sq_sum;
    }

    let mut sums = scratch
        .sums
        .as_mut()
        .submatrix_mut(0, 0, window_len, window_len);
    let mut squared_sums = scratch
        .squared_sums
        .as_mut()
        .submatrix_mut(0, 0, window_len, window_len);
    for row in 0..window_len {
        for col in 0..window_len {
            sums[(row, col)] = sums_vec[col];
            squared_sums[(row, col)] = squared_vec[col];
        }
    }

    let mut system = scratch
        .system
        .as_mut()
        .submatrix_mut(0, 0, window_len, window_len);
    let mut rhs = scratch.rhs.as_mut().submatrix_mut(0, 0, window_len, 1);

    solve_ld_window_from_stats(
        gram.as_ref(),
        sums.as_ref(),
        squared_sums.as_ref(),
        counts.as_ref(),
        center,
        ridge,
        system.as_mut(),
        rhs.as_mut(),
    )
}

fn collect_ready_jobs(
    ring: &LdRingBuffer,
    start_index: usize,
    config: &LdResolvedConfig,
    window_capacity: usize,
) -> Result<Vec<LdWeightJob>, HwePcaError> {
    let mut jobs = Vec::new();
    let mut next = start_index;

    loop {
        let Some(position) = ring.position_of(next) else {
            break;
        };

        let window_params = match &config.window {
            LdResolvedWindow::Sites { size } => {
                let window_size = (*size).max(1).min(window_capacity);
                let available = ring.len();
                let window_size = window_size.min(available).max(1);
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
                    None
                } else {
                    let center = position - start;
                    let keep_from = next.saturating_sub(window_size / 2);
                    Some((start, window_len, center, keep_from))
                }
            }
            LdResolvedWindow::BasePairs { ranges, .. } => {
                let range = &ranges[next];
                if range.end <= range.start {
                    return Err(HwePcaError::InvalidInput(
                        "LD base-pair window produced an empty range",
                    ));
                }
                let last_index = range.end - 1;
                match (ring.position_of(range.start), ring.position_of(last_index)) {
                    (Some(start_offset), Some(end_offset)) => {
                        let window_len = end_offset.saturating_sub(start_offset) + 1;
                        if window_len == 0 {
                            None
                        } else {
                            let center = position.saturating_sub(start_offset);
                            Some((start_offset, window_len, center, range.start))
                        }
                    }
                    _ => None,
                }
            }
        };

        let Some((start, window_len, center, keep_from)) = window_params else {
            break;
        };

        jobs.push(LdWeightJob {
            start_offset: start,
            window_len,
            center,
            keep_from,
        });
        next += 1;
    }

    Ok(jobs)
}

fn solve_ld_window_from_stats(
    gram: MatRef<'_, f64>,
    sums: MatRef<'_, f64>,
    squared_sums: MatRef<'_, f64>,
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
                let value = if count.is_finite() && count >= 2.0 {
                    let sum_x = sums[(i, j)];
                    let sum_y = sums[(j, i)];
                    let cov = gram[(i, j)] - (sum_x * sum_y) / count;
                    let var_x = squared_sums[(i, j)] - (sum_x * sum_x) / count;
                    let var_y = squared_sums[(j, i)] - (sum_y * sum_y) / count;

                    if !cov.is_finite() || !var_x.is_finite() || !var_y.is_finite() {
                        0.0
                    } else if var_x <= 0.0 || var_y <= 0.0 {
                        0.0
                    } else {
                        let corr = (cov / (var_x * var_y).sqrt()).clamp(-1.0, 1.0);
                        if !corr.is_finite() {
                            0.0
                        } else if count <= 2.0 {
                            0.0
                        } else {
                            let corr_sq = corr * corr;
                            let numerator = (count - 1.0) * corr_sq - 1.0;
                            let denominator = count - 2.0;
                            if denominator <= 0.0 {
                                0.0
                            } else {
                                let estimate = numerator / denominator;
                                estimate.max(0.0).min(1.0)
                            }
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

fn compute_component_weighted_norms_sq(
    loadings: MatRef<'_, f64>,
    ld_weights: Option<&[f64]>,
) -> Vec<f64> {
    let weights = ld_weights.unwrap_or(&[]);
    let n_weights = weights.len();
    let n_components = loadings.ncols();
    let mut norms_sq = vec![0.0f64; n_components];

    for component in 0..n_components {
        let column_ref = loadings.col(component);
        let mut sum = 0.0f64;
        let mut compensation = 0.0f64;

        if n_weights > 0 {
            let mut idx = 0usize;
            zip!(column_ref).for_each(|unzip!(value)| {
                let weight = if idx < n_weights { weights[idx] } else { 1.0 };
                let weighted = weight * *value;
                let square = weighted * weighted;
                let y = square - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
                idx += 1;
            });
        } else {
            zip!(column_ref).for_each(|unzip!(value)| {
                let square = *value * *value;
                let y = square - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            });
        }

        let sum = if sum.is_finite() && sum >= 0.0 {
            sum
        } else {
            0.0
        };
        norms_sq[component] = sum;
    }

    norms_sq
}

fn renormalize_variant_loadings(
    mut loadings: MatMut<'_, f64>,
    singular_values: &mut [f64],
    mut sample_scores: MatMut<'_, f64>,
) -> Vec<f64> {
    // Note: We normalize loadings in Euclidean space so that Σ(L²) = 1.
    // This Euclidean orthonormality ensures that the WLS projection (which minimizes error in weighted space)
    // simplifies to the Standard Projection (s = Σ x·w·L) when q=1, because LHS = Σ L L^T = I.
    // We pass `None` for weights to force Euclidean normalization.
    let mut norms_sq = compute_component_weighted_norms_sq(loadings.as_ref(), None);

    for (component, norm_sq) in norms_sq.iter_mut().enumerate() {
        let norm = (*norm_sq).sqrt();
        if !(norm.is_finite() && norm > 0.0) {
            *norm_sq = 0.0;
            continue;
        }

        let inv = norm.recip();
        let column_mut = loadings.rb_mut().col_mut(component);
        zip!(column_mut).for_each(|unzip!(value)| {
            *value *= inv;
        });

        singular_values[component] *= norm;

        let score_col = sample_scores.rb_mut().col_mut(component);
        zip!(score_col).for_each(|unzip!(value)| {
            *value *= norm;
        });

        *norm_sq = 1.0;
    }

    norms_sq
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
    P: FitProgressObserver + Send + Sync + 'static,
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
        // SAFETY: `buf0_uninit` was allocated with exactly `block_len` elements
        // and lives until the end of this function. We convert it to `&mut [f64]`
        // solely to let `VariantBlockSource::next_block_into` fill the
        // `n_samples * filled` prefix before it is observed.
        unsafe { std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len) };
    let (buf1_uninit, _) = stack.make_uninit::<f64>(block_len);
    let buf1 =
        // SAFETY: Identical justification as for `buf0`; only the filled prefix
        // is ever accessed after writing.
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

    

    thread::scope(|scope| {
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
                // SAFETY: Each pointer came from a mutable slice backed by the
                // stack allocation above and remains valid throughout the
                // scoped thread. Distinct `id`s ensure no concurrent aliasing,
                // and we only consume the portion that `next_block_into`
                // initialized.
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

                    // SAFETY: The pointer corresponds to a unique buffer owned
                    // by this worker. It stays valid until the `id` is returned
                    // via `free_sender`, preventing simultaneous mutable
                    // borrows, and we only touch the prefix filled with new
                    // samples.
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
    })
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
        let mut state = serializer.serialize_struct("HwePcaModel", 12)?;
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
        state.serialize_field(
            "component_weighted_norms_sq",
            &self.component_weighted_norms_sq,
        )?;
        state.serialize_field("variant_keys", &self.variant_keys)?;
        state.serialize_field("ld", &self.ld)?;
        state.serialize_field("genome_build", &self.genome_build)?;
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
            component_weighted_norms_sq: Vec<f64>,
            #[serde(default)]
            variant_keys: Option<Vec<VariantKey>>,
            #[serde(default)]
            ld: Option<LdWeights>,
            #[serde(default)]
            genome_build: Option<String>,
        }

        let raw = ModelData::deserialize(deserializer)?;
        let singular_values_len = raw.singular_values.len();
        let sample_basis = raw.sample_basis.into_mat().map_err(DeError::custom)?;
        let sample_scores = raw.sample_scores.into_mat().map_err(DeError::custom)?;
        let loadings = raw.loadings.into_mat().map_err(DeError::custom)?;
        let ld = raw.ld;
        let component_weighted_norms_sq =
            if raw.component_weighted_norms_sq.len() == singular_values_len {
                raw.component_weighted_norms_sq
            } else {
                compute_component_weighted_norms_sq(
                    loadings.as_ref(),
                    ld.as_ref().map(|ld| ld.weights.as_slice()),
                )
            };

        Ok(HwePcaModel {
            n_samples: raw.n_samples,
            n_variants: raw.n_variants,
            scaler: raw.scaler,
            eigenvalues: raw.eigenvalues,
            singular_values: raw.singular_values,
            sample_basis,
            sample_scores,
            loadings,
            component_weighted_norms_sq,
            variant_keys: raw.variant_keys,
            ld,
            genome_build: raw.genome_build,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::io::GenotypeDataset;
    use std::path::Path;
    use std::sync::Arc;

    fn compute_reference_ld_weights(
        data: &[f64],
        n_samples: usize,
        scaler: &HweScaler,
        config: &LdResolvedConfig,
    ) -> Result<Vec<f64>, HwePcaError> {
        let observed_variants = data.len() / n_samples;
        assert_eq!(observed_variants * n_samples, data.len());

        let mut standardized = data.to_vec();
        let mut mask = vec![0.0f64; data.len()];
        {
            let mut block = MatMut::from_column_major_slice_mut(
                &mut standardized,
                n_samples,
                observed_variants,
            );
            let mut mask_mat =
                MatMut::from_column_major_slice_mut(&mut mask, n_samples, observed_variants);
            scaler.standardize_block_with_mask(
                block.as_mut(),
                0..observed_variants,
                mask_mat.as_mut(),
                Par::Seq,
            );
        }

        let capacity = config.window_capacity().max(1);
        let mut weights = vec![1.0; observed_variants];
        let mut ring = LdRingBuffer::new(n_samples, capacity);
        let mut next_weight = 0usize;
        let progress =
            StageProgressHandle::new(Arc::new(NoopFitProgress), FitProgressStage::LdWeights);

        let block =
            MatMut::from_column_major_slice_mut(&mut standardized, n_samples, observed_variants);
        let presence = MatMut::from_column_major_slice_mut(&mut mask, n_samples, observed_variants);

        for idx in 0..observed_variants {
            let column = block.as_ref().col(idx);
            let present = presence.as_ref().col(idx);
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
            ring.indices_mut()[slot] = idx;
            assign_ready_weights(
                &mut ring,
                &mut weights,
                &mut next_weight,
                config,
                &progress,
                Par::Seq,
            )?;
        }

        assign_ready_weights(
            &mut ring,
            &mut weights,
            &mut next_weight,
            config,
            &progress,
            Par::Seq,
        )?;

        Ok(weights)
    }

    fn make_simple_scaler(data: &[f64], n_samples: usize) -> HweScaler {
        let observed_variants = data.len() / n_samples;
        let mut freqs = Vec::with_capacity(observed_variants);
        let mut scales = Vec::with_capacity(observed_variants);
        for variant in 0..observed_variants {
            let mut sum = 0.0;
            for sample in 0..n_samples {
                sum += data[variant * n_samples + sample];
            }
            let mean = sum / (n_samples as f64);
            let freq = (mean / 2.0).clamp(0.0, 1.0);
            let variance = (2.0 * freq * (1.0 - freq)).max(HWE_VARIANCE_EPSILON);
            let scale = variance.sqrt().max(HWE_SCALE_FLOOR);
            freqs.push(freq);
            scales.push(scale);
        }
        HweScaler::new(freqs, scales)
    }

    #[test]
    fn ld_weights_sites_match_reference() {
        let n_samples = 4;
        let observed_variants = 6;
        let data: Vec<f64> = vec![
            0.0, 1.0, 2.0, 1.0, // variant 0
            1.0, 2.0, 0.0, 2.0, // variant 1
            2.0, 1.0, 1.0, 0.0, // variant 2
            0.0, 2.0, 2.0, 1.0, // variant 3
            1.0, 0.0, 2.0, 2.0, // variant 4
            2.0, 2.0, 1.0, 0.0, // variant 5
        ];
        assert_eq!(data.len(), n_samples * observed_variants);

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);
        let config = LdResolvedConfig {
            window: LdResolvedWindow::Sites { size: 3 },
            ridge: DEFAULT_LD_RIDGE,
        };

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn ld_weights_bp_window_matches_reference() {
        let n_samples = 4;
        let observed_variants = 5;
        let data: Vec<f64> = vec![
            0.0, 1.0, 2.0, 1.0, // variant 0
            1.0, 2.0, 0.0, 2.0, // variant 1
            2.0, 1.0, 1.0, 0.0, // variant 2
            0.0, 2.0, 2.0, 1.0, // variant 3
            1.0, 0.0, 2.0, 2.0, // variant 4
        ];
        assert_eq!(data.len(), n_samples * observed_variants);

        let keys = vec![
            VariantKey::new("1", 100),
            VariantKey::new("1", 140),
            VariantKey::new("1", 200),
            VariantKey::new("1", 260),
            VariantKey::new("1", 320),
        ];
        let (ranges, capacity) = compute_ld_bp_ranges(&keys, 120).expect("ranges");
        let window = LdResolvedWindow::BasePairs {
            span_bp: 120,
            ranges: Arc::clone(&ranges),
            capacity,
        };
        let config = LdResolvedConfig {
            window,
            ridge: DEFAULT_LD_RIDGE,
        };

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn ld_weights_sites_large_window_matches_reference() {
        let n_samples = 8;
        let observed_variants = 32;
        let mut data = Vec::with_capacity(n_samples * observed_variants);
        for variant in 0..observed_variants {
            for sample in 0..n_samples {
                let value = ((variant * 3 + sample * 5) % 4) as f64;
                data.push(value);
            }
        }

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);
        let config = LdResolvedConfig {
            window: LdResolvedWindow::Sites { size: 17 },
            ridge: DEFAULT_LD_RIDGE,
        };

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn ld_weights_bp_large_window_matches_reference() {
        let n_samples = 6;
        let observed_variants = 28;
        let mut data = Vec::with_capacity(n_samples * observed_variants);
        for variant in 0..observed_variants {
            for sample in 0..n_samples {
                let value = ((variant * 5 + sample * 7) % 6) as f64;
                data.push(value);
            }
        }

        let keys = (0..observed_variants)
            .map(|idx| VariantKey::new("1", 10 + (idx as u64) * 37))
            .collect::<Vec<_>>();
        let span_bp = 240;
        let (ranges, capacity) = compute_ld_bp_ranges(&keys, span_bp).expect("ranges");
        let window = LdResolvedWindow::BasePairs {
            span_bp,
            ranges: Arc::clone(&ranges),
            capacity,
        };
        let config = LdResolvedConfig {
            window,
            ridge: DEFAULT_LD_RIDGE,
        };

        let scaler = make_simple_scaler(&data, n_samples);
        let mut source =
            DenseBlockSource::new(&data, n_samples, observed_variants).expect("dense source");
        let progress = Arc::new(NoopFitProgress);

        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            observed_variants,
            observed_variants,
            config.clone(),
            observed_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let reference = compute_reference_ld_weights(&data, n_samples, &scaler, &config)
            .expect("reference weights");

        assert_eq!(weights.len(), reference.len());
        let max_diff = weights
            .iter()
            .zip(reference.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f64::max);
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn bp_window_treated_as_total_span() {
        let keys = vec![
            VariantKey::new("1", 100),
            VariantKey::new("1", 140),
            VariantKey::new("1", 180),
            VariantKey::new("1", 220),
        ];

        let (ranges, capacity) = compute_ld_bp_ranges(&keys, 100).expect("ranges");

        assert_eq!(capacity, 3);
        assert_eq!(ranges.len(), keys.len());

        assert_eq!((ranges[1].start, ranges[1].end), (0, 3));
        assert_eq!((ranges[2].start, ranges[2].end), (1, 4));
    }

    #[test]
    fn bp_window_uses_total_span_for_odds() {
        let keys = vec![VariantKey::new("1", 1), VariantKey::new("1", 52)];

        let (ranges, capacity) = compute_ld_bp_ranges(&keys, 101).expect("ranges");

        assert_eq!(capacity, 1);
        assert_eq!(ranges.len(), keys.len());

        assert_eq!((ranges[0].start, ranges[0].end), (0, 1));
        assert_eq!((ranges[1].start, ranges[1].end), (1, 2));
    }

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
        let weights = Arc::from(vec![0.5, 2.0].into_boxed_slice());

        let mut block_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        {
            let mut block = MatMut::from_column_major_slice_mut(&mut block_data, 4, 2);
            scaler.standardize_block(block.as_mut(), 0..2, get_global_parallelism());
            apply_ld_weights(block.as_mut(), 0..2, &weights);
        }

        let expected = vec![0.5, 1.0, 1.5, 2.0, 10.0, 12.0, 14.0, 16.0];
        assert_eq!(block_data, expected);
    }

    #[test]
    fn ld_weights_are_ignored_when_absent() {
        let scaler = HweScaler::new(vec![0.0, 0.0], vec![1.0, 1.0]);

        let mut block_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        {
            let mut block = MatMut::from_column_major_slice_mut(&mut block_data, 4, 2);
            scaler.standardize_block(block.as_mut(), 0..2, get_global_parallelism());
            // No LD weights applied
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
