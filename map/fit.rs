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
use std::simd::Simd;
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use sysinfo::System;

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
const POPCOUNT_MAX_COLS: usize = 8;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
pub const DEFAULT_BLOCK_WIDTH: usize = 2_048;
const DENSE_EIGEN_FALLBACK_THRESHOLD: usize = 64;
const MAX_PARTIAL_COMPONENTS: usize = 512;
const FALLBACK_GRAM_BUDGET_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MIN_GRAM_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CovarianceComputationMode {
    Dense,
    Partial,
}

fn default_gram_budget_usize() -> usize {
    default_gram_budget_bytes()
        .min(usize::MAX as u64)
        .try_into()
        .unwrap()
}

fn default_gram_budget_bytes() -> u64 {
    static DEFAULT_GRAM_BUDGET_BYTES: OnceLock<u64> = OnceLock::new();

    *DEFAULT_GRAM_BUDGET_BYTES.get_or_init(compute_default_gram_budget_bytes)
}

fn compute_default_gram_budget_bytes() -> u64 {
    match detect_total_memory_bytes() {
        Some(total) if total > 0 => {
            let target = total.saturating_mul(3) / 4;
            target.max(MIN_GRAM_BUDGET_BYTES).min(total).max(1)
        }
        _ => FALLBACK_GRAM_BUDGET_BYTES,
    }
}

fn detect_total_memory_bytes() -> Option<u64> {
    let mut system = System::new_all();
    system.refresh_memory();
    system.total_memory().checked_mul(1024)
}

fn gram_matrix_budget_bytes() -> usize {
    match std::env::var("GNOMON_GRAM_BUDGET_BYTES") {
        Ok(value) => match value.parse::<u64>() {
            Ok(parsed) if parsed == 0 => usize::MAX,
            Ok(parsed) => (parsed.min(usize::MAX as u64)) as usize,
            Err(_) => default_gram_budget_usize(),
        },
        Err(_) => default_gram_budget_usize(),
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

    /// Provides a packed 2-bit hard-call view of the data when available.
    /// The default implementation returns None.
    fn hard_call_packed(&mut self) -> Option<HardCallPacked<'_>> {
        None
    }
}

pub struct HardCallPacked<'a> {
    data: &'a [u8],
    bytes_per_variant: usize,
    n_variants: usize,
}

impl<'a> HardCallPacked<'a> {
    fn slice(&self, start: usize, count: usize) -> Option<&'a [u8]> {
        let byte_start = start.checked_mul(self.bytes_per_variant)?;
        let byte_len = count.checked_mul(self.bytes_per_variant)?;
        let byte_end = byte_start.checked_add(byte_len)?;
        if byte_end > self.data.len() {
            return None;
        }
        Some(&self.data[byte_start..byte_end])
    }
}

#[derive(Debug)]
enum CacheState {
    Disabled,
    BuildingHardCall {
        packed: Vec<u8>,
        bytes_per_variant: usize,
        observed_variants: usize,
    },
    BuildingDense {
        data: Vec<f64>,
        observed_variants: usize,
        max_bytes: usize,
    },
    ReadyHardCall {
        packed: Vec<u8>,
        bytes_per_variant: usize,
        n_variants: usize,
    },
    ReadyDense {
        data: Vec<f64>,
        n_variants: usize,
    },
}

/// A smart cache that opportunistically stores hard-call genotypes in 2-bit packed
/// form. If any non-hard-call values are encountered, caching is disabled and the
/// underlying source is used directly.
struct CachedVariantBlockSource<'a, S>
where
    S: VariantBlockSource,
{
    source: &'a mut S,
    n_samples: usize,
    n_variants: usize,
    cursor: usize,
    state: CacheState,
}

impl<'a, S> CachedVariantBlockSource<'a, S>
where
    S: VariantBlockSource,
{
    fn new(source: &'a mut S, enable_cache: bool) -> Self {
        let n_samples = source.n_samples();
        let n_variants = source.n_variants();
        let bytes_per_variant = bytes_per_variant(n_samples);
        let packed_capacity = bytes_per_variant.saturating_mul(n_variants);
        let state = if !enable_cache || n_samples == 0 {
            CacheState::Disabled
        } else {
            CacheState::BuildingHardCall {
                packed: Vec::with_capacity(packed_capacity),
                bytes_per_variant,
                observed_variants: 0,
            }
        };
        Self {
            source,
            n_samples,
            n_variants,
            cursor: 0,
            state,
        }
    }

    fn reset_cache_build(&mut self) {
        if self.n_samples == 0 {
            self.state = CacheState::Disabled;
            return;
        }
        let bytes_per_variant = bytes_per_variant(self.n_samples);
        let packed_capacity = bytes_per_variant.saturating_mul(self.n_variants);
        self.state = CacheState::BuildingHardCall {
            packed: Vec::with_capacity(packed_capacity),
            bytes_per_variant,
            observed_variants: 0,
        };
    }

    fn decode_from_cache(&self, start_variant: usize, filled: usize, storage: &mut [f64]) {
        match &self.state {
            CacheState::ReadyHardCall {
                packed,
                bytes_per_variant,
                n_variants,
            } => {
                let total_bytes = bytes_per_variant
                    .checked_mul(*n_variants)
                    .unwrap_or(0);
                if packed.len() < total_bytes {
                    return;
                }

                let table = hard_call_decode_table();
                for variant_idx in 0..filled {
                    let global_idx = start_variant + variant_idx;
                    let byte_start = global_idx * bytes_per_variant;
                    let byte_end = byte_start + bytes_per_variant;
                    if byte_end > packed.len() {
                        break;
                    }
                    let dest_offset = variant_idx * self.n_samples;
                    let dest = &mut storage[dest_offset..dest_offset + self.n_samples];
                    decode_packed_hard_calls(&packed[byte_start..byte_end], dest, self.n_samples, table);
                }
            }
            CacheState::ReadyDense { data, n_variants } => {
                let total = self.n_samples.saturating_mul(*n_variants);
                if data.len() < total {
                    return;
                }
                for variant_idx in 0..filled {
                    let global_idx = start_variant + variant_idx;
                    let src_offset = global_idx * self.n_samples;
                    let dest_offset = variant_idx * self.n_samples;
                    if src_offset + self.n_samples > data.len() {
                        break;
                    }
                    storage[dest_offset..dest_offset + self.n_samples]
                        .copy_from_slice(&data[src_offset..src_offset + self.n_samples]);
                }
            }
            _ => {}
        }
    }
}

impl<'a, S> VariantBlockSource for CachedVariantBlockSource<'a, S>
where
    S: VariantBlockSource,
{
    type Error = S::Error;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.n_variants
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.cursor = 0;
        match self.state {
            CacheState::ReadyHardCall { .. } | CacheState::ReadyDense { .. } => Ok(()),
            CacheState::Disabled => self.source.reset(),
            CacheState::BuildingHardCall { .. } | CacheState::BuildingDense { .. } => {
                self.reset_cache_build();
                self.source.reset()
            }
        }
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }

        if matches!(self.state, CacheState::ReadyHardCall { .. } | CacheState::ReadyDense { .. }) {
            let remaining = self.n_variants.saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(0);
            }
            let filled = remaining.min(max_variants);
            self.decode_from_cache(self.cursor, filled, storage);
            self.cursor += filled;
            return Ok(filled);
        }

        let filled = self.source.next_block_into(max_variants, storage)?;
        if filled == 0 {
            match &mut self.state {
                CacheState::BuildingHardCall {
                    packed,
                    bytes_per_variant,
                    observed_variants,
                } => {
                    if *observed_variants > 0 {
                        let packed = std::mem::take(packed);
                        let n_variants = *observed_variants;
                        self.n_variants = n_variants;
                        self.state = CacheState::ReadyHardCall {
                            packed,
                            bytes_per_variant: *bytes_per_variant,
                            n_variants,
                        };
                    } else {
                        self.state = CacheState::Disabled;
                    }
                }
                CacheState::BuildingDense {
                    data,
                    observed_variants,
                    ..
                } => {
                    if *observed_variants > 0 {
                        let data = std::mem::take(data);
                        let n_variants = *observed_variants;
                        self.n_variants = n_variants;
                        self.state = CacheState::ReadyDense { data, n_variants };
                    } else {
                        self.state = CacheState::Disabled;
                    }
                }
                _ => {}
            }
            return Ok(0);
        }

        match &mut self.state {
            CacheState::BuildingHardCall {
                packed,
                bytes_per_variant,
                observed_variants,
            } => {
                let base_observed = *observed_variants;
                let mut scratch = vec![0u8; *bytes_per_variant];
                let mut packed_count = 0usize;
                let mut hard_call_only = true;
                for variant_idx in 0..filled {
                    let src_offset = variant_idx * self.n_samples;
                    let src = &storage[src_offset..src_offset + self.n_samples];
                    if !pack_hard_calls_into(&mut scratch, src, self.n_samples) {
                        hard_call_only = false;
                        break;
                    }
                    packed.extend_from_slice(&scratch);
                    packed_count += 1;
                }
                *observed_variants = base_observed.saturating_add(packed_count);
                if !hard_call_only {
                    let packed_snapshot = std::mem::take(packed);
                    if let Some(mut dense) = DenseCacheBuilder::from_packed(
                        packed_snapshot,
                        *bytes_per_variant,
                        self.n_samples,
                        *observed_variants,
                    ) {
                        if packed_count < filled {
                            dense.push_block_range(
                                storage,
                                packed_count,
                                filled,
                                self.n_samples,
                            );
                            *observed_variants = base_observed.saturating_add(filled);
                        }
                        self.state = CacheState::BuildingDense {
                            data: dense.data,
                            observed_variants: *observed_variants,
                            max_bytes: dense.max_bytes,
                        };
                    } else {
                        self.state = CacheState::Disabled;
                    }
                } else {
                    *observed_variants = base_observed.saturating_add(filled);
                }
            }
            CacheState::BuildingDense {
                data,
                observed_variants,
                max_bytes,
            } => {
                let start = data.len();
                let needed = self.n_samples.saturating_mul(filled);
                data.resize(start + needed, 0.0);
                let src = &storage[..self.n_samples * filled];
                data[start..start + needed].copy_from_slice(src);
                *observed_variants = observed_variants.saturating_add(filled);
                if data.len().saturating_mul(std::mem::size_of::<f64>()) > *max_bytes {
                    self.state = CacheState::Disabled;
                }
            }
            CacheState::Disabled => {}
            CacheState::ReadyHardCall { .. } | CacheState::ReadyDense { .. } => {}
        }

        self.cursor += filled;
        Ok(filled)
    }

    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        match self.state {
            CacheState::ReadyHardCall { .. } | CacheState::ReadyDense { .. } => None,
            _ => self.source.progress_bytes(),
        }
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        match self.state {
            CacheState::ReadyHardCall { .. } | CacheState::ReadyDense { .. } => {
                Some((self.cursor.min(self.n_variants), Some(self.n_variants)))
            }
            _ => self.source.progress_variants(),
        }
    }

    fn variant_quality(&self, filled: usize, storage: &mut [f64]) {
        match self.state {
            CacheState::ReadyHardCall { .. } | CacheState::ReadyDense { .. } => {
                for value in storage.iter_mut().take(filled) {
                    *value = 1.0;
                }
            }
            _ => (&*self.source).variant_quality(filled, storage),
        }
    }

    fn hard_call_packed(&mut self) -> Option<HardCallPacked<'_>> {
        match &self.state {
            CacheState::ReadyHardCall {
                packed,
                bytes_per_variant,
                n_variants,
            } => Some(HardCallPacked {
                data: packed,
                bytes_per_variant: *bytes_per_variant,
                n_variants: *n_variants,
            }),
            _ => None,
        }
    }
}

fn bytes_per_variant(n_samples: usize) -> usize {
    (n_samples + 3) / 4
}

fn pack_hard_calls_into(dst: &mut [u8], src: &[f64], n_samples: usize) -> bool {
    for (byte_idx, out) in dst.iter_mut().enumerate() {
        let base = byte_idx * 4;
        let mut byte = 0u8;
        for offset in 0..4 {
            let sample_idx = base + offset;
            if sample_idx >= n_samples {
                break;
            }
            let val = src[sample_idx];
            let code = if val.is_nan() {
                1u8
            } else if val == 0.0 {
                0u8
            } else if val == 1.0 {
                2u8
            } else if val == 2.0 {
                3u8
            } else {
                return false;
            };
            byte |= code << (offset * 2);
        }
        *out = byte;
    }
    true
}

fn decode_packed_hard_calls(
    bytes: &[u8],
    dest: &mut [f64],
    n_samples: usize,
    table: &[[f64; 4]; 256],
) {
    let mut sample_idx = 0usize;
    for &byte in bytes {
        if sample_idx >= n_samples {
            break;
        }
        let decoded = &table[byte as usize];
        let remaining = n_samples - sample_idx;
        let take = remaining.min(4);
        dest[sample_idx..sample_idx + take].copy_from_slice(&decoded[..take]);
        sample_idx += take;
    }
}

fn hard_call_decode_table() -> &'static [[f64; 4]; 256] {
    static TABLE: OnceLock<[[f64; 4]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0f64; 4]; 256];
        for byte in 0u16..256 {
            for offset in 0..4 {
                let code = ((byte >> (offset * 2)) & 0b11) as u8;
                table[byte as usize][offset] = match code {
                    0 => 0.0,
                    1 => f64::NAN,
                    2 => 1.0,
                    3 => 2.0,
                    _ => unreachable!(),
                };
            }
        }
        table
    })
}

fn hard_call_code_table() -> &'static [[u8; 4]; 256] {
    static TABLE: OnceLock<[[u8; 4]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u8; 4]; 256];
        for byte in 0u16..256 {
            for offset in 0..4 {
                let code = ((byte >> (offset * 2)) & 0b11) as u8;
                table[byte as usize][offset] = code;
            }
        }
        table
    })
}

struct DenseCacheBuilder {
    data: Vec<f64>,
    max_bytes: usize,
}

impl DenseCacheBuilder {
    fn from_packed(
        packed: Vec<u8>,
        bytes_per_variant: usize,
        n_samples: usize,
        observed_variants: usize,
    ) -> Option<Self> {
        if n_samples == 0 {
            return None;
        }
        let max_bytes = cache_budget_bytes();
        if max_bytes == 0 {
            return None;
        }
        let needed = n_samples
            .checked_mul(observed_variants)?
            .checked_mul(std::mem::size_of::<f64>())?;
        if needed > max_bytes {
            return None;
        }
        let mut data = Vec::with_capacity(n_samples.saturating_mul(observed_variants));
        data.resize(n_samples.saturating_mul(observed_variants), 0.0);
        if observed_variants > 0 {
            let table = hard_call_decode_table();
            for variant_idx in 0..observed_variants {
                let byte_start = variant_idx * bytes_per_variant;
                let byte_end = byte_start + bytes_per_variant;
                if byte_end > packed.len() {
                    break;
                }
                let dest_offset = variant_idx * n_samples;
                let dest = &mut data[dest_offset..dest_offset + n_samples];
                decode_packed_hard_calls(&packed[byte_start..byte_end], dest, n_samples, table);
            }
        }
        Some(Self { data, max_bytes })
    }

    fn push_block_range(
        &mut self,
        storage: &[f64],
        start_variant: usize,
        end_variant: usize,
        n_samples: usize,
    ) {
        if end_variant <= start_variant {
            return;
        }
        let variant_count = end_variant - start_variant;
        let needed = n_samples.saturating_mul(variant_count);
        let offset = data_offset(start_variant, n_samples);
        let src = &storage[offset..offset + needed];
        self.data.extend_from_slice(src);
    }
}

fn data_offset(variant_idx: usize, n_samples: usize) -> usize {
    variant_idx.saturating_mul(n_samples)
}

fn cache_budget_bytes() -> usize {
    match detect_total_memory_bytes() {
        Some(total) if total > 0 => {
            let target = total.saturating_mul(1) / 3;
            target.min(usize::MAX as u64) as usize
        }
        _ => 0,
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
        _ => standardize_column_simd_impl_lanes2(values, mean, inv),
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
        standardize_column_simd_impl_lanes4(values, mean, inv);
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
    standardize_column_simd_impl_lanes4(values, mean, inv);
}

#[inline(always)]
fn standardize_column_simd_impl_lanes2(values: &mut [f64], mean: f64, inv: f64) {
    let mean_simd = Simd::<f64, 2>::splat(mean);
    let inv_simd = Simd::<f64, 2>::splat(inv);

    let (chunks, remainder) = values.as_chunks_mut::<2>();
    for chunk in chunks {
        let lane = Simd::<f64, 2>::from_array(*chunk);
        let mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let mut result = standardized.to_array();
        let finite_bits = mask.to_bitmask();
        for (lane, value) in result.iter_mut().enumerate() {
            if ((finite_bits >> lane) & 1) == 0 {
                *value = 0.0;
            }
        }
        *chunk = result;
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
fn standardize_column_simd_impl_lanes4(values: &mut [f64], mean: f64, inv: f64) {
    let mean_simd = Simd::<f64, 4>::splat(mean);
    let inv_simd = Simd::<f64, 4>::splat(inv);

    let (chunks, remainder) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        let lane = Simd::<f64, 4>::from_array(*chunk);
        let mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let mut result = standardized.to_array();
        let finite_bits = mask.to_bitmask();
        for (lane, value) in result.iter_mut().enumerate() {
            if ((finite_bits >> lane) & 1) == 0 {
                *value = 0.0;
            }
        }
        *chunk = result;
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
        _ => standardize_column_with_mask_simd_impl_lanes2(values, mask, mean, inv),
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
        standardize_column_with_mask_simd_impl_lanes4(values, mask, mean, inv);
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
    standardize_column_with_mask_simd_impl_lanes4(values, mask, mean, inv);
}

#[inline(always)]
fn standardize_column_with_mask_simd_impl_lanes2(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) {
    let mean_simd = Simd::<f64, 2>::splat(mean);
    let inv_simd = Simd::<f64, 2>::splat(inv);
    let one = Simd::<f64, 2>::splat(1.0);

    let (value_chunks, value_remainder) = values.as_chunks_mut::<2>();
    let (mask_chunks, mask_remainder) = mask.as_chunks_mut::<2>();

    debug_assert_eq!(value_chunks.len(), mask_chunks.len());
    debug_assert_eq!(value_remainder.len(), mask_remainder.len());

    for (value_chunk, mask_chunk) in value_chunks.iter_mut().zip(mask_chunks.iter_mut()) {
        let lane = Simd::<f64, 2>::from_array(*value_chunk);
        let finite_mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let finite_bits = finite_mask.to_bitmask();
        let mut result = standardized.to_array();
        let mut mask_values = one.to_array();
        for lane in 0..2 {
            if ((finite_bits >> lane) & 1) == 0 {
                result[lane] = 0.0;
                mask_values[lane] = 0.0;
            }
        }
        *value_chunk = result;
        *mask_chunk = mask_values;
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

#[inline(always)]
fn standardize_column_with_mask_simd_impl_lanes4(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) {
    let mean_simd = Simd::<f64, 4>::splat(mean);
    let inv_simd = Simd::<f64, 4>::splat(inv);
    let one = Simd::<f64, 4>::splat(1.0);

    let (value_chunks, value_remainder) = values.as_chunks_mut::<4>();
    let (mask_chunks, mask_remainder) = mask.as_chunks_mut::<4>();

    debug_assert_eq!(value_chunks.len(), mask_chunks.len());
    debug_assert_eq!(value_remainder.len(), mask_remainder.len());

    for (value_chunk, mask_chunk) in value_chunks.iter_mut().zip(mask_chunks.iter_mut()) {
        let lane = Simd::<f64, 4>::from_array(*value_chunk);
        let finite_mask = lane.is_finite();
        let standardized = (lane - mean_simd) * inv_simd;
        let finite_bits = finite_mask.to_bitmask();
        let mut result = standardized.to_array();
        let mut mask_values = one.to_array();
        for lane in 0..4 {
            if ((finite_bits >> lane) & 1) == 0 {
                result[lane] = 0.0;
                mask_values[lane] = 0.0;
            }
        }
        *value_chunk = result;
        *mask_chunk = mask_values;
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
        _ => standardize_column_simd_full_impl_lanes2(values, mean, inv),
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
        standardize_column_simd_full_impl_lanes4(values, mean, inv);
    }
}

#[cfg(test)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx")]
/// # Safety
/// Callers must guarantee AVX availability; runtime dispatch ensures that the
/// function is only invoked when the CPU advertises the capability.
unsafe fn standardize_column_simd_full_avx(values: &mut [f64], mean: f64, inv: f64) {
    standardize_column_simd_full_impl_lanes4(values, mean, inv);
}

#[cfg(test)]
#[inline(always)]
fn standardize_column_simd_full_impl_lanes2(values: &mut [f64], mean: f64, inv: f64) {
    let mean_simd = Simd::<f64, 2>::splat(mean);
    let inv_simd = Simd::<f64, 2>::splat(inv);

    let (chunks, remainder) = values.as_chunks_mut::<2>();
    for chunk in chunks {
        let lane = Simd::<f64, 2>::from_array(*chunk);
        let standardized = (lane - mean_simd) * inv_simd;
        *chunk = standardized.to_array();
    }

    for value in remainder {
        *value = (*value - mean) * inv;
    }
}

#[cfg(test)]
#[inline(always)]
fn standardize_column_simd_full_impl_lanes4(values: &mut [f64], mean: f64, inv: f64) {
    let mean_simd = Simd::<f64, 4>::splat(mean);
    let inv_simd = Simd::<f64, 4>::splat(inv);

    let (chunks, remainder) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        let lane = Simd::<f64, 4>::from_array(*chunk);
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
                sum_and_count_finite_impl_lanes4(values)
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
                return sum_and_count_finite_impl_lanes2(values);
            }
        }
        _ => sum_and_count_finite_impl_lanes2(values),
    }
}

#[inline(always)]
fn sum_and_count_finite_impl_lanes2(values: &[f64]) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0usize;

    let (chunks, remainder) = values.as_chunks::<2>();
    for chunk in chunks {
        let lane = Simd::<f64, 2>::from_array(*chunk);
        let mask = lane.is_finite();
        let lane_values = lane.to_array();
        let finite_bits = mask.to_bitmask();
        for (idx, &value) in lane_values.iter().enumerate() {
            if ((finite_bits >> idx) & 1) != 0 {
                sum += value;
            }
        }
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

#[inline(always)]
fn sum_and_count_finite_impl_lanes4(values: &[f64]) -> (f64, usize) {
    let mut sum = 0.0;
    let mut count = 0usize;

    let (chunks, remainder) = values.as_chunks::<4>();
    for chunk in chunks {
        let lane = Simd::<f64, 4>::from_array(*chunk);
        let mask = lane.is_finite();
        let lane_values = lane.to_array();
        let finite_bits = mask.to_bitmask();
        for (idx, &value) in lane_values.iter().enumerate() {
            if ((finite_bits >> idx) & 1) != 0 {
                sum += value;
            }
        }
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
    sum_and_count_finite_impl_lanes4(values)
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

        // Determine whether to use dense or matrix-free path based on memory budget
        let gram_budget = gram_matrix_budget_bytes();
        let gram_bytes = gram_matrix_size_bytes(n_samples);
        let gram_mode = covariance_computation_mode(n_samples, gram_budget);

        let mut cached_source = CachedVariantBlockSource::new(
            source,
            matches!(gram_mode, CovarianceComputationMode::Partial),
        );
        let source = &mut cached_source;

        // Compute LD weights first if requested (applies to both paths)
        let (ld_weights_arc, ld_weights) = if let Some(ld_cfg) = ld_config {
            let (_, _, ld_weights_computed) = compute_stats_and_ld_weights(
                source,
                block_capacity,
                ld_cfg,
                n_variants_hint,
                progress,
                par,
            )?;
            let ld_arc = Arc::<[f64]>::from(ld_weights_computed.weights.clone().into_boxed_slice());
            (Some(ld_arc), Some(ld_weights_computed))
        } else {
            (None, None)
        };

        // Reset source after LD computation
        if ld_weights_arc.is_some() {
            source.reset().map_err(|err| HwePcaError::Source(Box::new(err)))?;
        }

        // Choose between dense and matrix-free paths
        let (decomposition, scaler, observed_variants) = match gram_mode {
            CovarianceComputationMode::Dense => {
                // PATH A: Dense - Use fused pass for small/medium datasets
                let (scaler, observed_variants, covariance) = compute_stats_and_covariance_blockwise(
                    source,
                    block_capacity,
                    par,
                    progress,
                    n_variants_hint,
                    ld_weights_arc.clone(),
                )?;

                // Decompose the dense covariance matrix
                let eig_result = covariance.as_ref().self_adjoint_eigen(Side::Upper);
                let decomposition = match eig_result {
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
                };

                (decomposition, scaler, observed_variants)
            }
            CovarianceComputationMode::Partial => {
                // PATH B: Matrix-free - For biobank-scale datasets
                log::info!(
                    "Using matrix-free eigensolver (Gram matrix would require {} bytes)",
                    gram_bytes.unwrap_or(usize::MAX)
                );

                // First pass: compute statistics only
                progress.on_stage_start(FitProgressStage::AlleleStatistics, n_variants_hint);
                let stats_progress = StageProgressHandle::new(Arc::clone(progress), FitProgressStage::AlleleStatistics);
                let (scaler, observed_variants) = compute_variant_statistics(
                    source,
                    block_capacity,
                    par,
                    stats_progress,
                    n_variants_hint,
                )?;

                // Setup matrix-free operator
                let operator = StandardizedCovarianceOp::new(
                    source,
                    block_capacity,
                    n_variants_hint,
                    observed_variants,
                    scaler.clone(),
                    ld_weights_arc.clone(),
                );

                // Progress tracking for Gram matrix computation
                progress.on_stage_start(FitProgressStage::GramMatrix, n_variants_hint);
                let gram_progress_handle = Some(StageProgressHandle::new(
                    Arc::clone(progress),
                    FitProgressStage::GramMatrix,
                ));

                // Run matrix-free eigensolver
                let decomposition_result = compute_covariance_eigenpairs(
                    &operator,
                    par,
                    CovarianceComputationMode::Partial,
                    target_components,
                    gram_progress_handle.as_ref(),
                );

                // Extract source and scaler from operator (ownership handled)
                operator.into_parts();

                progress.on_stage_finish(FitProgressStage::GramMatrix);

                (decomposition_result?, scaler, observed_variants)
            }
        };

        let variant_count = scaler.variant_scales().len();
        debug_assert_eq!(variant_count, observed_variants);
        if variant_count == 0 {
            return Err(HwePcaError::InvalidInput(
                "HWE PCA requires at least one variant",
            ));
        }

        log::info!("Observed {} variants during PCA fitting", variant_count);

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

struct StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    apply_lock: Mutex<()>,
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
    P: FitProgressObserver + Send + Sync + 'static,
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
            apply_lock: Mutex::new(()),
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
            apply_lock: _,
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
    P: FitProgressObserver + Send + Sync + 'static,
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
    P: FitProgressObserver + Send + Sync + 'static,
{
    fn apply_scratch(&self, rhs_ncols: usize, _: Par) -> StackReq {
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
        let apply_guard = self
            .apply_lock
            .lock()
            .expect("covariance apply lock poisoned");
        let _ = &apply_guard;

        debug_assert_eq!(out.nrows(), self.n_samples);
        debug_assert_eq!(rhs.nrows(), self.n_samples);

        out.fill(0.0);

        if rhs.ncols() == 0 {
            return;
        }

        if rhs.ncols() <= POPCOUNT_MAX_COLS {
            if self.try_apply_hardcall_packed(out.rb_mut(), rhs) {
                return;
            }
        }

        let block_len = self.n_samples * self.block_capacity;
        let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
        // SAFETY: `buf0_uninit` was allocated with capacity `block_len` and lives
        // for the entire scope of `apply`. We immediately coerce it to
        // `&mut [f64]` so it can be passed to `VariantBlockSource::next_block_into`,
        // which writes the first `n_samples * filled` entries before we read
        // them. Any remaining slots stay uninitialized but are never observed.
        let buf0 = unsafe {
            std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len)
        };
        let (buf1_uninit, stack) = stack.make_uninit::<f64>(block_len);
        // SAFETY: Same reasoning as above for `buf0`; the buffer lives long
        // enough and only the initialized prefix is observed.
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
            // SAFETY: `temp_mat_uninit` returns an uninitialized matrix view that
            // must not be read before being written. We only use it as the
            // destination of GEMM calls with `Accum::Replace`, which overwrite
            // every element touched. The stack capacity was sized by
            // `apply_scratch`, so the backing storage stays live for the entire
            // duration of `apply`.
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
                    let source: &mut S = &mut guard;
                    source.reset().map_err(|e| HwePcaError::Source(Box::new(e)))
                } {
                    let _ = filled_sender.send(PrefetchMessage::Error(err));
                    return;
                }

                let mut start = 0usize;
                while let Ok(id) = free_receiver.recv() {
                    // SAFETY: The raw pointer originated from a mutable slice
                    // in `buffer_slices` and remains valid until the scoped
                    // thread exits. Channel ownership ensures each `id`
                    // corresponds to a single borrower, so the reconstructed
                    // slice is never aliased. Only the prefix written by
                    // `next_block_into` is observed after this call.
                    let buffer_slice = unsafe {
                        std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                    };
                    let filled_res = {
                        let mut guard = source_mutex
                            .lock()
                            .expect("covariance source mutex poisoned");
                        let source: &mut S = &mut guard;
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

                        // SAFETY: `buffer_ptrs_compute[id]` points to the same
                        // uniquely-owned buffer handed to the prefetch thread.
                        // Scoped threads guarantee the memory outlives this use,
                        // channel coordination prevents concurrent access, and we
                        // restrict all reads to the portion initialized by the
                        // source.
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

    fn apply(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>, par: Par, _: &mut MemStack) {
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
    P: FitProgressObserver + Send + Sync + 'static,
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
    P: FitProgressObserver + Send + Sync + 'static,
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
            let stack = MemStack::new(&mut mem);
            partial_self_adjoint_eigen(
                eigvecs.as_mut(),
                &mut eigvals,
                &op,
                v0.as_ref(),
                f64::EPSILON * 128.0,
                par,
                stack,
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
    P: FitProgressObserver + Send + Sync + 'static,
{
    let apply_guard = operator
        .apply_lock
        .lock()
        .expect("covariance apply lock poisoned");
    let _ = &apply_guard;

    let n_samples = operator.n_samples;
    let cross_products = Mat::zeros(n_samples, n_samples);

    if n_samples == 0 {
        return Ok(Mat::zeros(n_samples, n_samples));
    }

    let block_capacity = operator.block_capacity;
    let block_len = n_samples * block_capacity;
    let buffer_req = StackReq::new::<f64>(block_len).and(StackReq::new::<f64>(block_len));
    let mut mem = MemBuffer::new(buffer_req);
    let stack = MemStack::new(&mut mem);
    let (buf0_uninit, stack) = stack.make_uninit::<f64>(block_len);
    let buf0 =
        // SAFETY: `buf0_uninit` owns `block_len` contiguous `f64`s that live for
        // the duration of this function. We immediately coerce it to
        // `&mut [f64]` so `VariantBlockSource::next_block_into` can stream data
        // into the prefix `n_samples * filled`. That prefix is fully written
        // before being read; any trailing capacity remains untouched.
        unsafe { std::slice::from_raw_parts_mut(buf0_uninit.as_mut_ptr() as *mut f64, block_len) };
    let (buf1_uninit, _) = stack.make_uninit::<f64>(block_len);
    let buf1 =
        // SAFETY: Mirroring the reasoning for `buf0`, the allocation stays alive
        // for the entire call and only the written prefix is ever read.
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
                let source: &mut S = &mut guard;
                source.reset().map_err(|e| HwePcaError::Source(Box::new(e)))
            } {
                let _ = filled_sender.send(PrefetchMessage::Error(err));
                return;
            }

            let mut start = 0usize;
            let mut used_source_progress = false;
            while let Ok(id) = free_receiver.recv() {
                // SAFETY: The pointer was derived from a uniquely-owned slice
                // stored in `buffer_slices` and the scoped threads ensure the
                // backing storage outlives this reconstruction. Channel
                // ownership gives each `id` a single borrower, and we only
                // consume the prefix populated by `next_block_into`.
                let buffer_slice = unsafe {
                    std::slice::from_raw_parts_mut(buffer_ptrs_prefetch[id].0, block_len)
                };
                let (filled_res, progress_bytes, progress_variants) = {
                    let mut guard = source_mutex
                        .lock()
                        .expect("covariance source mutex poisoned");
                    let source: &mut S = &mut guard;
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
        let mut cross_products = cross_products;
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

                    // SAFETY: Each `id` corresponds to a single buffer whose
                    // ownership is passed through the channel. The pointer
                    // remains valid and uniquely borrowed until we send `id`
                    // back on the free list, and all reads stay within the
                    // initialized prefix.
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

                    let block_ref = block.as_ref();

                    triangular_matmul::matmul(
                        cross_products.as_mut(),
                        triangular_matmul::BlockStructure::TriangularLower,
                        Accum::Add,
                        block_ref,
                        triangular_matmul::BlockStructure::Rectangular,
                        block_ref.transpose(),
                        triangular_matmul::BlockStructure::Rectangular,
                        1.0,
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

        Ok(cross_products)
    });

    let mut covariance = result?;
    let scale = operator.scale;
    for col in 0..n_samples {
        for row in col..n_samples {
            covariance[(row, col)] *= scale;
        }
    }
    mirror_lower_to_upper(&mut covariance);

    Ok(covariance)
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

impl<'a, S, P> StandardizedCovarianceOp<'a, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    fn try_apply_hardcall_packed(&self, mut out: MatMut<'_, f64>, rhs: MatRef<'_, f64>) -> bool {
        let mut guard = self
            .source
            .lock()
            .expect("covariance source mutex poisoned");
        let source: &mut S = &mut guard;
        let _ = source.reset();
        let packed = match source.hard_call_packed() {
            Some(packed) => packed,
            None => return false,
        };

        let n_samples = self.n_samples;
        let ncols = rhs.ncols();
        let freqs = self.scaler.allele_frequencies();
        let scales = self.scaler.variant_scales();
        let max_variants = self.observed_variants.min(packed.n_variants);
        let max_variants = max_variants.min(freqs.len()).min(scales.len());
        let code_table = hard_call_code_table();

        for variant_idx in 0..max_variants {
            let mean = 2.0 * freqs[variant_idx];
            let denom = scales[variant_idx].max(HWE_SCALE_FLOOR);
            let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
            if inv == 0.0 {
                continue;
            }

            let z0 = (0.0 - mean) * inv;
            let z1 = (1.0 - mean) * inv;
            let z2 = (2.0 - mean) * inv;

            let weight_sq = if let Some(weights) = &self.ld_weights {
                let w = weights.get(variant_idx).copied().unwrap_or(1.0);
                w * w
            } else {
                1.0
            };
            let coeff = self.scale * weight_sq;

            let variant_bytes = match packed.slice(variant_idx, 1) {
                Some(slice) => slice,
                None => break,
            };

            let mut proj = [0.0f64; POPCOUNT_MAX_COLS];
            for col in 0..ncols {
                let mut sum0 = 0.0f64;
                let mut sum1 = 0.0f64;
                let mut sum2 = 0.0f64;

                let mut sample_idx = 0usize;
                for &byte in variant_bytes {
                    if sample_idx >= n_samples {
                        break;
                    }
                    let codes = &code_table[byte as usize];
                    for offset in 0..4 {
                        let idx = sample_idx + offset;
                        if idx >= n_samples {
                            break;
                        }
                        let val = rhs[(idx, col)];
                        match codes[offset] {
                            0 => sum0 += val,
                            2 => sum1 += val,
                            3 => sum2 += val,
                            _ => {}
                        }
                    }
                    sample_idx += 4;
                }

                let p = (z0 * sum0 + z1 * sum1 + z2 * sum2) * coeff;
                proj[col] = p;
            }

            let mut sample_idx = 0usize;
            for &byte in variant_bytes {
                if sample_idx >= n_samples {
                    break;
                }
                let codes = &code_table[byte as usize];
                for offset in 0..4 {
                    let idx = sample_idx + offset;
                    if idx >= n_samples {
                        break;
                    }
                    let z = match codes[offset] {
                        0 => z0,
                        2 => z1,
                        3 => z2,
                        _ => 0.0,
                    };
                    if z != 0.0 {
                        for col in 0..ncols {
                            out[(idx, col)] += z * proj[col];
                        }
                    }
                }
                sample_idx += 4;
            }
        }

        true
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
    P: FitProgressObserver + Send + Sync + 'static,
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
        let stack = MemStack::new(&mut mem);
        partial_self_adjoint_eigen(
            eigvecs.as_mut(),
            &mut eigvals,
            operator,
            v0.as_ref(),
            f64::EPSILON * 128.0,
            par,
            stack,
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
    P: FitProgressObserver + Send + Sync,
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

        if processed + filled < processed {
            return Err(HwePcaError::InvalidInput(
                "variant count overflow during statistics computation",
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

        // Step 3: Apply LD weights if available (pre-computed weights used directly)
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
