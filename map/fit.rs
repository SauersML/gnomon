use super::blocklanczos::{BlockKrylovError, BlockKrylovParams, BlockOperator, block_krylov_eigen};
use super::progress::{
    FitProgressObserver, FitProgressStage, NoopFitProgress, StageProgressHandle,
};
use super::variant_filter::{MatchKind, VariantKey};
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
    PartialEigenParams, partial_eigen_scratch, partial_self_adjoint_eigen,
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
use std::simd::Simd;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use sysinfo::System;

pub const HWE_VARIANCE_EPSILON: f64 = 1.0e-12;
pub const HWE_SCALE_FLOOR: f64 = 1.0e-6;
/// Right-hand-side columns handled per pass of the packed hard-call kernel.
///
/// Not a ceiling on block width: wider right-hand sides are processed in
/// chunks of this many columns. It bounds the small on-stack projection buffer
/// and how often the packed bytes are re-walked in the scatter phase.
///
/// `PACKED_RHS_MAX_COLS` keeps production below one chunk, so the chunk loop
/// runs exactly once there. It stays anyway: the kernel is correct at any
/// width, and `packed_hard_call_kernel_matches_the_general_path` exercises it
/// across a chunk boundary so the loop cannot rot behind the gate.
const PACKED_RHS_CHUNK_COLS: usize = 32;
/// Widest right-hand side that still takes the packed hard-call kernel.
///
/// A throughput gate, not a correctness one — see the dispatch in
/// [`StandardizedCovarianceOp::apply`].
const PACKED_RHS_MAX_COLS: usize = 8;
pub const EIGENVALUE_EPSILON: f64 = 1.0e-9;
/// Largest streamed variant tile. On 250k-sample PLINK microarray data, 512
/// matched 1,024's throughput while cutting the decode-buffer working set.
pub const DEFAULT_BLOCK_WIDTH: usize = 512;
const DENSE_EIGEN_FALLBACK_THRESHOLD: usize = 64;
const MAX_PARTIAL_COMPONENTS: usize = 512;
/// Pool assumed when the machine refuses to report its own memory.
const FALLBACK_MEMORY_POOL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MIN_GRAM_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const MIN_KRYLOV_BASIS_BYTES: u64 = 256 * 1024 * 1024;
const MIN_STREAMING_TILE_BYTES: u64 = 256 * 1024 * 1024;
/// Samples above which the dense reference is never selected, whatever the work
/// estimate says: past this its n³ eigendecomposition is minutes to hours and
/// its n×n allocation is hundreds of megabytes duplicating data the streaming
/// path never materializes at all.
const DENSE_REFERENCE_MAX_SAMPLES: usize = 4_096;
/// Flops per n³ for a symmetric eigendecomposition (tridiagonalization plus the
/// QR sweep, with eigenvectors). Order-of-magnitude is all this needs to be.
const DENSE_EIGEN_FLOPS_PER_CUBE: f64 = 10.0;
/// Genome passes a converged block-Krylov solve typically needs. `min_passes`
/// is 2 and `max_passes` 24; four is the middle of what actually converges.
const ESTIMATED_KRYLOV_PASSES: f64 = 4.0;
pub const DEFAULT_LD_WINDOW: usize = 51;
const DEFAULT_LD_RIDGE: f64 = 1.0e-3;
const MIN_LD_WEIGHT: f64 = 1.0e-6;

/// Raised when the LD schedule and the stream disagree about what variant an
/// index names. See [`LdResolvedConfig::range_count`] for what the two lists
/// are and how a filter between them pulls them apart.
const LD_RANGE_LIST_MISMATCH: &str = "LD windows were computed from a different variant list than the fit streamed: the window \
     ranges must be built from the retained, post-filter variants, in stream order";

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

/// One memory pool for the whole fit, cut into named shares.
///
/// Every consumer used to size itself against its own fraction of *total* RAM:
/// the Gram matrix took 75%, the source cache another 75%, the Krylov basis an
/// eighth, the streaming tiles another eighth. Each obeyed its own budget and
/// all four together could still exceed the machine — and none of them knew
/// that the sample basis, the scores, the loadings, the LD ring buffer, the
/// source mmaps, their page cache and allocator slack are resident at the same
/// time. Independent budgets that do not know about each other are not budgets.
///
/// So: one pool, apportioned once, from *available* rather than total memory —
/// what is free now is what a fit can actually spend — with a deliberate slice
/// left unapportioned for everything this plan does not name.
///
/// Measured once per process, deliberately. Availability drifts while a fit
/// runs, and a plan that re-measured would hand the same fit different tile
/// widths at different moments — different tilings mean different summation
/// order, and a fit that is not reproducible run to run is not a fit.
struct FitMemoryPlan {
    /// Dense n×n covariance. Only the small-problem reference path spends it.
    gram_bytes: usize,
    /// Opportunistic 2-bit or dense source cache. It buys speed and nothing
    /// else, so it takes the largest share and is the first to be denied.
    source_cache_bytes: usize,
    /// Retained block-Krylov basis before a restart is forced.
    krylov_basis_bytes: usize,
    /// The streaming operator's two decode tiles plus its projection temp.
    streaming_tile_bytes: usize,
}

/// Fraction of available memory the plan is allowed to apportion at all. The
/// remainder is headroom for the allocations no budget here can size.
const FIT_POOL_PERCENT_OF_AVAILABLE: u64 = 70;
// Shares of that pool, in percent, summing to 100.
const GRAM_SHARE_PERCENT: u64 = 15;
const SOURCE_CACHE_SHARE_PERCENT: u64 = 45;
const KRYLOV_BASIS_SHARE_PERCENT: u64 = 25;
const STREAMING_TILE_SHARE_PERCENT: u64 = 15;

fn fit_memory_plan() -> &'static FitMemoryPlan {
    static PLAN: OnceLock<FitMemoryPlan> = OnceLock::new();
    PLAN.get_or_init(compute_fit_memory_plan)
}

fn compute_fit_memory_plan() -> FitMemoryPlan {
    let (pool, cache_allowed) = match detect_memory_bytes() {
        Some((total, available)) => {
            // `available` already discounts what other processes hold; clamp it
            // to `total` in case the two are reported inconsistently.
            let usable = available.clamp(1, total);
            (
                usable.saturating_mul(FIT_POOL_PERCENT_OF_AVAILABLE) / 100,
                true,
            )
        }
        // Nothing measurable: assume a modest pool and refuse to cache the
        // source, the one share whose overshoot buys only speed.
        None => (FALLBACK_MEMORY_POOL_BYTES, false),
    };

    let share = |percent: u64, floor: u64| -> usize {
        let bytes = (pool.saturating_mul(percent) / 100).max(floor);
        bytes.min(usize::MAX as u64) as usize
    };

    FitMemoryPlan {
        gram_bytes: share(GRAM_SHARE_PERCENT, MIN_GRAM_BUDGET_BYTES),
        source_cache_bytes: if cache_allowed {
            share(SOURCE_CACHE_SHARE_PERCENT, 0)
        } else {
            0
        },
        krylov_basis_bytes: share(KRYLOV_BASIS_SHARE_PERCENT, MIN_KRYLOV_BASIS_BYTES),
        streaming_tile_bytes: share(STREAMING_TILE_SHARE_PERCENT, MIN_STREAMING_TILE_BYTES),
    }
}

/// `(total, available)` bytes, or `None` when the platform reports neither.
///
/// A platform that reports a total but no availability is treated as "all of it
/// is available": zero there means "unsupported", not "the machine is full".
fn detect_memory_bytes() -> Option<(u64, u64)> {
    let mut system = System::new_all();
    system.refresh_memory();
    let total = system.total_memory();
    if total == 0 {
        return None;
    }
    let available = system.available_memory();
    Some((total, if available == 0 { total } else { available }))
}

fn gram_matrix_budget_bytes() -> usize {
    fit_memory_plan().gram_bytes
}

fn cache_budget_bytes() -> usize {
    fit_memory_plan().source_cache_bytes
}

/// Byte budget for the retained block-Krylov basis.
///
/// The basis is `n × (block_width × depth)` f64s — for 500k samples, a 60-wide
/// block and depth 4 that is ~0.9 GiB, which is a good trade against re-reading
/// millions of variants. It is still bounded, because "a good trade" stops
/// being true if the basis becomes the largest object in the process; past the
/// budget the solver restarts from its current Ritz block instead of growing.
fn krylov_basis_budget_bytes() -> usize {
    fit_memory_plan().krylov_basis_bytes
}

fn streaming_tile_budget_bytes() -> usize {
    fit_memory_plan().streaming_tile_bytes
}

fn gram_matrix_size_bytes(n: usize) -> Option<usize> {
    n.checked_mul(n)?.checked_mul(core::mem::size_of::<f64>())
}

/// Chooses between forming the covariance and never forming it.
///
/// The old question — "does 8n² fit in a fraction of RAM?" — is the wrong one.
/// At 100k samples the Gram is 74.5 GiB, which "fits" a 128 GiB machine, and
/// the fit then spends O(p·n²) forming it and O(n³) decomposing it: an
/// allocation that succeeds and a computation that never finishes.
///
/// Dense is a *reference* path — the exact, direct solve that the iterative one
/// is checked against — so it is chosen on estimated **work**:
///
/// * dense costs `p·n²/2` to accumulate the symmetric Gram plus ~`10n³` to
///   eigendecompose it;
/// * the block solver never forms `C` at all: each pass streams the genome
///   through two GEMMs of width `b` (`Xᵀq`, then `X(Xᵀq)`), costing
///   `passes·2·p·n·b`.
///
/// Those cross at a few hundred samples for biobank-scale `p`, which is why
/// dense must stop being selected long before it stops fitting in memory.
/// Memory feasibility remains necessary — it is simply no longer sufficient.
fn covariance_computation_mode(
    n_samples: usize,
    n_variants_hint: usize,
    components: usize,
    gram_budget_bytes: usize,
) -> CovarianceComputationMode {
    let Some(gram_bytes) = gram_matrix_size_bytes(n_samples) else {
        return CovarianceComputationMode::Partial;
    };
    if gram_bytes > gram_budget_bytes || n_samples > DENSE_REFERENCE_MAX_SAMPLES {
        return CovarianceComputationMode::Partial;
    }

    // At this size the whole Gram is tens of kilobytes and its decomposition is
    // microseconds. The flop model below is asymptotic and says nothing useful
    // here — fixed per-pass overheads dominate both routes — so the direct
    // solve simply wins, and small fits stay off the knife edge where a change
    // of constant would flip them onto the iterative path.
    if n_samples <= DENSE_EIGEN_FALLBACK_THRESHOLD {
        return CovarianceComputationMode::Dense;
    }

    let n = n_samples as f64;
    // The same oversampling rule the solver will actually use, asked rather
    // than re-derived, so the estimate cannot drift away from the solver.
    let block_width = BlockKrylovParams::auto(components, n_samples, krylov_basis_budget_bytes())
        .block_width as f64;

    // Both streaming work and Gram accumulation are linear in the variant
    // count, so per-variant is the form in which `p` cancels.
    let dense_accumulation_per_variant = 0.5 * n * n;
    let streaming_per_variant = ESTIMATED_KRYLOV_PASSES * 2.0 * n * block_width;

    if n_variants_hint == 0 {
        // A source that will not say how many variants it has (a streaming VCF
        // before it has been counted) leaves the O(n³) eigendecomposition
        // unpriceable — but not the rest. Dropping that term can only flatter
        // dense, so deciding on the p-free half alone is the most generous
        // answer that is still honest.
        return if dense_accumulation_per_variant <= streaming_per_variant {
            CovarianceComputationMode::Dense
        } else {
            CovarianceComputationMode::Partial
        };
    }

    let p = n_variants_hint as f64;
    let dense_flops = p * dense_accumulation_per_variant + DENSE_EIGEN_FLOPS_PER_CUBE * n * n * n;
    let streaming_flops = p * streaming_per_variant;

    if dense_flops <= streaming_flops {
        CovarianceComputationMode::Dense
    } else {
        CovarianceComputationMode::Partial
    }
}

#[derive(Clone, Debug, Default)]
pub struct FitOptions {
    pub ld: Option<LdConfig>,
    pub cache_source: bool,
    pub(crate) precomputed_variant_statistics: Option<PrecomputedVariantStatistics>,
    /// Accept a fit whose eigensolver stopped short of its tolerance.
    ///
    /// The default refuses, and the default is the honest answer: an
    /// unconverged subspace is not a slightly noisy PCA, it is a set of
    /// components wrong by an amount nobody measured. Every score, loading and
    /// ancestry call derived from it inherits that error, and the artifact is
    /// otherwise indistinguishable from a finished one — which is precisely the
    /// failure [`FitDiagnostics`] exists to document rather than to excuse.
    ///
    /// A caller who genuinely wants the best available estimate — an
    /// exploratory run, a deliberately short pass budget — opts in here. That
    /// changes only whether the fit is refused: the diagnostics travel onto the
    /// model either way, so a best-effort artifact still records
    /// `converged: false` for whoever reads it later.
    pub allow_unconverged: bool,
}

/// Variant moments already computed against the exact sample and marker stream
/// the fit will consume. Indexed FIT QC produces these while deciding which
/// markers survive, so carrying them into the matrix-free fit removes a
/// redundant packed BED pass without changing any floating-point arithmetic.
#[derive(Clone, Debug)]
pub(crate) struct PrecomputedVariantStatistics {
    n_samples: usize,
    scaler: HweScaler,
    standardized_sums_sq: Vec<f64>,
}

impl PrecomputedVariantStatistics {
    pub(crate) fn from_moments(n_samples: usize, moments: &[(f64, f64, usize)]) -> Self {
        let mut frequencies = Vec::with_capacity(moments.len());
        let mut scales = Vec::with_capacity(moments.len());
        let mut standardized_sums_sq = Vec::with_capacity(moments.len());
        for &(sum, sum_sq, calls) in moments {
            let (frequency, scale, standardized_sum_sq) =
                finalize_variant_moments(sum, sum_sq, calls);
            frequencies.push(frequency);
            scales.push(scale);
            standardized_sums_sq.push(standardized_sum_sq);
        }
        Self {
            n_samples,
            scaler: HweScaler::new(frequencies, scales),
            standardized_sums_sq,
        }
    }

    fn matches(&self, n_samples: usize, n_variants: usize) -> bool {
        self.n_samples == n_samples
            && self.scaler.variant_scales().len() == n_variants
            && self.standardized_sums_sq.len() == n_variants
    }

    fn cloned_parts(&self) -> (HweScaler, Vec<f64>, usize) {
        (
            self.scaler.clone(),
            self.standardized_sums_sq.clone(),
            self.standardized_sums_sq.len(),
        )
    }
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

/// One variant's LD window, as a half-open range of **stream positions**.
///
/// Both window kinds resolve to this: a base-pair span becomes the indices
/// inside that span, a site count becomes the indices inside that count, and
/// either way the range is clipped to the variant's own chromosome so that no
/// window pairs the tail of one chromosome with the head of the next.
///
/// The indices are positions in the variant list the fit will stream, which is
/// the whole content of the invariant documented on
/// [`LdResolvedConfig::range_count`]: a range list computed against any other
/// ordering weights the wrong markers while every index stays in bounds.
#[derive(Clone, Debug)]
struct LdWindowRange {
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
enum LdResolvedWindow {
    Sites {
        size: usize,
        /// One chromosome-clipped window per streamed variant.
        ranges: Arc<[LdWindowRange]>,
    },
    BasePairs {
        span_bp: u64,
        ranges: Arc<[LdWindowRange]>,
        capacity: usize,
    },
}

#[derive(Clone, Debug)]
struct LdResolvedConfig {
    window: LdResolvedWindow,
    ridge: f64,
}

impl LdResolvedConfig {
    /// Ring-buffer size, and also the widest window the schedule can ask for.
    ///
    /// The two are the same number on purpose. A window is at most this many
    /// markers wide and its last marker is always the newest push, so the
    /// newest `window_capacity` markers the ring kept are exactly the ones any
    /// ready window can name — the buffer is bounded by the window, not by the
    /// genome, and nothing a window needs is ever evicted.
    fn window_capacity(&self) -> usize {
        match &self.window {
            LdResolvedWindow::Sites { size, .. } => *size,
            LdResolvedWindow::BasePairs { capacity, .. } => *capacity,
        }
    }

    fn bp_window(&self) -> Option<u64> {
        match &self.window {
            LdResolvedWindow::Sites { .. } => None,
            LdResolvedWindow::BasePairs { span_bp, .. } => Some(*span_bp),
        }
    }

    /// How many variants the precomputed windows were cut from.
    ///
    /// This is the checkable half of the index invariant. `ranges[i]` describes
    /// variant `i` *of the stream*, so it is only about the right marker if
    /// position `i` of the list the ranges came from is position `i` of the
    /// stream. Anything that drops variants between the two breaks exactly
    /// that — a MAF screen renumbers what it keeps `0, 1, 2, …` while the list
    /// it was numbered against still counts the rejects — and it breaks it
    /// silently, since every index stays in bounds and every weight stays
    /// plausible. The counts diverge the moment one variant is dropped, which
    /// makes them the cheap, total test for it.
    ///
    fn range_count(&self) -> usize {
        match &self.window {
            LdResolvedWindow::Sites { ranges, .. } | LdResolvedWindow::BasePairs { ranges, .. } => {
                ranges.len()
            }
        }
    }

    /// The window centred on `next`, or `None` while the stream still owes it
    /// markers.
    ///
    /// `newest` is the highest stream index the ring holds. A window is *ready*
    /// only once its last marker has been streamed. Sizing it against whatever
    /// the ring happened to hold instead is what let a 51-site request collapse
    /// to three markers, and `stream_ended` is the single legitimate exception:
    /// the file ran out and no further push will ever arrive. A chromosome
    /// ending mid-stream needs no exception, because the ranges are already
    /// clipped to it.
    fn window_range(
        &self,
        next: usize,
        newest: usize,
        stream_ended: bool,
    ) -> Result<Option<LdWindowRange>, HwePcaError> {
        let (start, end) = match &self.window {
            LdResolvedWindow::Sites { ranges, .. } | LdResolvedWindow::BasePairs { ranges, .. } => {
                let Some(range) = ranges.get(next) else {
                    return Ok(None);
                };
                (range.start, range.end)
            }
        };

        if end <= start {
            return Err(HwePcaError::InvalidInput(
                "LD window resolved to an empty range",
            ));
        }

        if end - 1 <= newest {
            return Ok(Some(LdWindowRange { start, end }));
        }

        if !stream_ended || newest < start {
            return Ok(None);
        }

        Ok(Some(LdWindowRange {
            start,
            end: newest + 1,
        }))
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

                let keys = cfg.variant_keys.as_ref().ok_or(HwePcaError::InvalidInput(
                    "LD site window requires chromosome keys for all variants",
                ))?;
                if keys.len() != observed_variants {
                    return Err(HwePcaError::InvalidInput(
                        "LD site window requires chromosome keys for all variants",
                    ));
                }
                let ranges = compute_ld_site_ranges(keys, window);
                LdResolvedWindow::Sites {
                    size: window,
                    ranges,
                }
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

/// Filters streamed variants by observed allele frequency and call rate.
///
/// Indexed hard-call sources perform the same screen directly on their packed
/// bytes before constructing this wrapper. This source is the bounded-memory
/// path for VCF/BCF and for any indexed source without a packed view.
pub struct StreamingVariantFilterSource<S>
where
    S: VariantBlockSource,
{
    inner: S,
    min_maf: Option<f64>,
    max_missing_rate: Option<f64>,
    n_samples: usize,
    observed_variants: usize,
    retained_variants_hint: Option<usize>,
    retained_indices: Vec<usize>,
    inner_storage: Vec<f64>,
    inner_quality: Vec<f64>,
    block_quality: Vec<f64>,
    retained_keys: Option<Vec<VariantKey>>,
}

impl<S> StreamingVariantFilterSource<S>
where
    S: VariantBlockSource,
{
    pub fn new(
        inner: S,
        min_maf: Option<f64>,
        max_missing_rate: Option<f64>,
    ) -> Result<Self, HwePcaError> {
        if min_maf.is_none() && max_missing_rate.is_none() {
            return Err(HwePcaError::InvalidInput(
                "variant filter requires a MAF or missing-call threshold",
            ));
        }
        if min_maf
            .is_some_and(|threshold| !(threshold.is_finite() && (0.0..=0.5).contains(&threshold)))
        {
            return Err(HwePcaError::InvalidInput(
                "MAF threshold must be finite and between 0 and 0.5",
            ));
        }
        if max_missing_rate
            .is_some_and(|threshold| !(threshold.is_finite() && (0.0..=1.0).contains(&threshold)))
        {
            return Err(HwePcaError::InvalidInput(
                "missing-call threshold must be finite and between 0 and 1",
            ));
        }
        let n_samples = inner.n_samples();
        let hint = inner.n_variants();
        Ok(Self {
            inner,
            min_maf,
            max_missing_rate,
            n_samples,
            observed_variants: 0,
            retained_variants_hint: None,
            retained_indices: Vec::with_capacity(hint),
            inner_storage: Vec::new(),
            inner_quality: Vec::new(),
            block_quality: Vec::new(),
            retained_keys: Some(Vec::with_capacity(hint)),
        })
    }

    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }

    pub fn retained_indices(&self) -> &[usize] {
        &self.retained_indices
    }

    fn ensure_workspaces(&mut self, max_variants: usize) {
        let capacity = self.n_samples.saturating_mul(max_variants.max(1));
        if self.inner_storage.len() < capacity {
            self.inner_storage.resize(capacity, 0.0);
        }
        if self.inner_quality.len() < max_variants {
            self.inner_quality.resize(max_variants, 1.0);
        }
        if self.block_quality.len() < max_variants {
            self.block_quality.resize(max_variants, 1.0);
        }
    }
}

impl<S> VariantBlockSource for StreamingVariantFilterSource<S>
where
    S: VariantBlockSource,
{
    type Error = S::Error;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.retained_variants_hint
            .unwrap_or_else(|| self.inner.n_variants())
    }

    fn block_storage_samples(&self) -> usize {
        self.inner.block_storage_samples().max(self.n_samples)
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.inner.reset()?;
        self.observed_variants = 0;
        self.retained_indices.clear();
        self.block_quality.clear();
        if let Some(keys) = self.retained_keys.as_mut() {
            keys.clear();
        }
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

        self.ensure_workspaces(max_variants);
        let mut filled = 0usize;

        while filled < max_variants {
            let request = max_variants - filled;
            let inner_filled = self
                .inner
                .next_block_into(request, &mut self.inner_storage[..])?;
            if inner_filled == 0 {
                self.retained_variants_hint = Some(self.retained_indices.len());
                break;
            }

            self.inner
                .variant_quality(inner_filled, &mut self.inner_quality[..inner_filled]);
            let inner_keys = self.inner.block_variant_keys();

            for local_idx in 0..inner_filled {
                let src_start = local_idx * self.n_samples;
                let src_end = src_start + self.n_samples;
                let values = &self.inner_storage[src_start..src_end];
                if !variant_passes_qc(values, self.min_maf, self.max_missing_rate) {
                    self.observed_variants += 1;
                    continue;
                }

                let dst_start = filled * self.n_samples;
                let dst_end = dst_start + self.n_samples;
                storage[dst_start..dst_end].copy_from_slice(values);
                self.block_quality[filled] = self.inner_quality[local_idx];
                self.retained_indices.push(self.observed_variants);
                if let (Some(keys), Some(inner_keys)) =
                    (self.retained_keys.as_mut(), inner_keys.as_ref())
                {
                    if let Some(key) = inner_keys.get(local_idx) {
                        keys.push(key.clone());
                    }
                }
                self.observed_variants += 1;
                filled += 1;
            }
        }

        Ok(filled)
    }

    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        self.inner.progress_bytes()
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        self.inner.progress_variants()
    }

    fn variant_quality(&self, filled: usize, storage: &mut [f64]) {
        let limit = filled.min(storage.len()).min(self.block_quality.len());
        storage[..limit].copy_from_slice(&self.block_quality[..limit]);
        for value in storage.iter_mut().take(filled).skip(limit) {
            *value = 1.0;
        }
    }

    fn take_variant_keys(&mut self) -> Option<Vec<VariantKey>> {
        self.retained_keys.take().filter(|keys| !keys.is_empty())
    }
}

fn variant_passes_qc(values: &[f64], min_maf: Option<f64>, max_missing_rate: Option<f64>) -> bool {
    let mut sum = 0.0;
    let mut calls = 0usize;
    for &value in values {
        if value.is_finite() {
            sum += value;
            calls += 1;
        }
    }

    variant_moments_pass_qc(sum, calls, values.len(), min_maf, max_missing_rate)
}

pub(crate) fn variant_moments_pass_qc(
    sum: f64,
    calls: usize,
    n_samples: usize,
    min_maf: Option<f64>,
    max_missing_rate: Option<f64>,
) -> bool {
    if max_missing_rate.is_some_and(|threshold| {
        let missing = n_samples.saturating_sub(calls);
        missing as f64 > threshold * n_samples as f64
    }) {
        return false;
    }

    if let Some(threshold) = min_maf {
        if calls == 0 {
            return false;
        }
        let allele_frequency = (sum / (2.0 * calls as f64)).clamp(0.0, 1.0);
        if allele_frequency.min(1.0 - allele_frequency) < threshold {
            return false;
        }
    }

    true
}

/// Restricts a [`VariantBlockSource`] to a subset of the dataset's sample rows.
///
/// Blocks are laid out column-major per variant
/// (`storage[variant * n_samples + sample]`), so a row subset is a per-variant
/// gather of the retained positions. Wrapping happens above the raw dataset
/// source and below the variant filter, which is what makes `--keep` mean
/// "fit the PCA on these samples only": allele frequencies, the MAF/call-rate
/// screen, LD weights and the covariance are all computed from the retained
/// rows alone.
///
/// [`SampleSubsetSource::passthrough`] is the identity case used when no subset
/// was requested; it forwards every call — including the packed hard-call fast
/// path — to the inner source untouched, so the unsubset fit is byte-identical
/// to one that never saw this wrapper.
pub struct SampleSubsetSource<S>
where
    S: VariantBlockSource,
{
    inner: S,
    /// `None` means "keep every sample", i.e. identity passthrough.
    indices: Option<Vec<usize>>,
    inner_n_samples: usize,
    n_samples: usize,
    inner_storage: Vec<f64>,
}

impl<S> SampleSubsetSource<S>
where
    S: VariantBlockSource,
{
    /// Wraps `inner` without dropping any sample.
    pub fn passthrough(inner: S) -> Self {
        let n_samples = inner.n_samples();
        Self {
            inner,
            indices: None,
            inner_n_samples: n_samples,
            n_samples,
            inner_storage: Vec::new(),
        }
    }

    /// Wraps `inner`, retaining only the rows at `indices`.
    ///
    /// `indices` are positions in the dataset's own sample order and must be
    /// strictly increasing, which both rules out duplicates and keeps the
    /// retained rows in dataset order.
    pub fn new(inner: S, indices: Vec<usize>) -> Result<Self, HwePcaError> {
        if indices.is_empty() {
            return Err(HwePcaError::InvalidInput(
                "sample subset must retain at least one sample",
            ));
        }
        if indices.windows(2).any(|pair| pair[1] <= pair[0]) {
            return Err(HwePcaError::InvalidInput(
                "sample subset indices must be strictly increasing",
            ));
        }
        let inner_n_samples = inner.n_samples();
        if indices.last().copied().unwrap_or(0) >= inner_n_samples {
            return Err(HwePcaError::InvalidInput(
                "sample subset index exceeds the dataset sample count",
            ));
        }
        let n_samples = indices.len();
        Ok(Self {
            inner,
            indices: Some(indices),
            inner_n_samples,
            n_samples,
            inner_storage: Vec::new(),
        })
    }

    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S> VariantBlockSource for SampleSubsetSource<S>
where
    S: VariantBlockSource,
{
    type Error = S::Error;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.inner.n_variants()
    }

    fn block_storage_samples(&self) -> usize {
        self.inner.block_storage_samples().max(self.inner_n_samples)
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.inner.reset()
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if self.indices.is_none() {
            return self.inner.next_block_into(max_variants, storage);
        }
        if max_variants == 0 {
            return Ok(0);
        }

        let capacity = self.inner_n_samples.saturating_mul(max_variants);
        if self.inner_storage.len() < capacity {
            self.inner_storage.resize(capacity, 0.0);
        }

        let filled = self
            .inner
            .next_block_into(max_variants, &mut self.inner_storage[..])?;

        let indices = match self.indices.as_deref() {
            Some(indices) => indices,
            None => return Ok(filled),
        };
        for local_idx in 0..filled {
            let src_start = local_idx * self.inner_n_samples;
            let src = &self.inner_storage[src_start..src_start + self.inner_n_samples];
            let dst_start = local_idx * self.n_samples;
            let dst = &mut storage[dst_start..dst_start + self.n_samples];
            for (out, &row) in dst.iter_mut().zip(indices.iter()) {
                *out = src[row];
            }
        }

        Ok(filled)
    }

    fn progress_bytes(&self) -> Option<(u64, Option<u64>)> {
        self.inner.progress_bytes()
    }

    fn progress_variants(&self) -> Option<(usize, Option<usize>)> {
        self.inner.progress_variants()
    }

    fn variant_quality(&self, filled: usize, storage: &mut [f64]) {
        self.inner.variant_quality(filled, storage);
    }

    fn block_variant_keys(&self) -> Option<&[VariantKey]> {
        self.inner.block_variant_keys()
    }

    fn take_variant_keys(&mut self) -> Option<Vec<VariantKey>> {
        self.inner.take_variant_keys()
    }

    fn next_standardized_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
        allele_frequencies: &[f64],
        variant_scales: &[f64],
        ld_weights: Option<&[f64]>,
    ) -> Result<Option<usize>, Self::Error> {
        if self.indices.is_some() {
            return Ok(None);
        }
        self.inner.next_standardized_block_into(
            max_variants,
            storage,
            allele_frequencies,
            variant_scales,
            ld_weights,
        )
    }

    fn hard_call_packed(&mut self) -> Option<HardCallPacked<'_>> {
        match self.indices.as_deref() {
            Some(indices) => self
                .inner
                .hard_call_packed()
                .map(|packed| packed.with_sample_selection(indices)),
            None => self.inner.hard_call_packed(),
        }
    }
}

/// Cuts one chromosome-clipped window per variant from a site count.
///
/// `window` is the number of markers a window holds, and it is odd on every
/// path the CLI can reach, so `left` and `right` below are the same half-width:
/// variant `i` wants `[i − h, i + h]`. Clipping is to the *contiguous run* of
/// records sharing a chromosome rather than to a global grouping, matching
/// [`compute_ld_bp_ranges`] — a file that returns to a chromosome later gets
/// windows within each run, and never a window that silently joins two runs
/// separated by half the genome.
fn compute_ld_site_ranges(keys: &[VariantKey], window: usize) -> Arc<[LdWindowRange]> {
    let left = window / 2;
    let right = window.saturating_sub(1) - left;

    let mut ranges = Vec::with_capacity(keys.len());
    let mut run_start = 0usize;
    while run_start < keys.len() {
        let mut run_end = run_start + 1;
        while run_end < keys.len() && keys[run_end].chromosome == keys[run_start].chromosome {
            run_end += 1;
        }

        for idx in run_start..run_end {
            ranges.push(LdWindowRange {
                start: idx.saturating_sub(left).max(run_start),
                end: idx.saturating_add(right).saturating_add(1).min(run_end),
            });
        }

        run_start = run_end;
    }

    ranges.into_boxed_slice().into()
}

fn compute_ld_bp_ranges(
    keys: &[VariantKey],
    span_bp: u64,
) -> Result<(Arc<[LdWindowRange]>, usize), HwePcaError> {
    if keys.is_empty() {
        return Err(HwePcaError::InvalidInput(
            "LD base-pair window requires at least one variant",
        ));
    }

    for pair in keys.windows(2) {
        if pair[0].chromosome == pair[1].chromosome && pair[1].position < pair[0].position {
            return Err(HwePcaError::InvalidInput(
                "LD base-pair window requires nondecreasing positions within each chromosome run",
            ));
        }
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
        ranges.push(LdWindowRange {
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

    /// Sample rows that one requested variant may occupy across this source's
    /// decode buffers. A row-subsetting wrapper still has to decode the
    /// physical cohort before gathering retained rows, so block planning must
    /// use this footprint rather than only the rows it ultimately yields.
    fn block_storage_samples(&self) -> usize {
        self.n_samples()
    }

    fn reset(&mut self) -> Result<(), Self::Error>;
    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error>;

    /// Decode and standardize a block in one pass when the source has a native
    /// representation that can apply the transform during decode.
    ///
    /// The statistic and optional weight slices begin at the source's current
    /// logical cursor. `None` means the source cannot fuse this operation and
    /// has not advanced; callers then use [`Self::next_block_into`].
    fn next_standardized_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
        allele_frequencies: &[f64],
        variant_scales: &[f64],
        ld_weights: Option<&[f64]>,
    ) -> Result<Option<usize>, Self::Error> {
        let _ = (
            max_variants,
            storage,
            allele_frequencies,
            variant_scales,
            ld_weights,
        );
        Ok(None)
    }

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

    fn block_variant_keys(&self) -> Option<&[VariantKey]> {
        None
    }

    fn take_variant_keys(&mut self) -> Option<Vec<VariantKey>> {
        None
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
    physical_n_variants: usize,
    selection: Option<Vec<usize>>,
    match_kinds: Option<Vec<MatchKind>>,
    sample_selection: Option<Vec<usize>>,
    sample_byte_masks: Option<Vec<u8>>,
    missing_variant: Option<Vec<u8>>,
}

impl<'a> HardCallPacked<'a> {
    const MISSING_VARIANT: usize = usize::MAX;

    pub(crate) fn new(data: &'a [u8], bytes_per_variant: usize, n_variants: usize) -> Self {
        Self {
            data,
            bytes_per_variant,
            physical_n_variants: n_variants,
            selection: None,
            match_kinds: None,
            sample_selection: None,
            sample_byte_masks: None,
            missing_variant: None,
        }
    }

    pub(crate) fn new_selected(
        data: &'a [u8],
        bytes_per_variant: usize,
        physical_n_variants: usize,
        selection: &'a [usize],
        match_kinds: Option<&'a [MatchKind]>,
    ) -> Self {
        Self {
            data,
            bytes_per_variant,
            physical_n_variants,
            selection: Some(selection.to_vec()),
            match_kinds: match_kinds.map(<[MatchKind]>::to_vec),
            sample_selection: None,
            sample_byte_masks: None,
            missing_variant: None,
        }
    }

    pub(crate) fn with_sample_selection(mut self, selection: &[usize]) -> Self {
        self.sample_selection = Some(selection.to_vec());
        self.sample_byte_masks = build_sample_byte_masks(self.bytes_per_variant, selection);
        self
    }

    pub(crate) fn with_sample_selection_and_masks(
        mut self,
        selection: &[usize],
        masks: &[u8],
    ) -> Self {
        self.sample_selection = Some(selection.to_vec());
        self.sample_byte_masks = Some(masks.to_vec());
        self
    }

    pub(crate) fn with_model_gaps(mut self, present_mask: &[bool]) -> Option<Self> {
        let matched = present_mask.iter().filter(|&&present| present).count();
        if matched != self.n_variants() {
            return None;
        }

        let mut selection = Vec::with_capacity(present_mask.len());
        let mut match_kinds = Vec::with_capacity(present_mask.len());
        let mut inner_variant = 0usize;
        for &present in present_mask {
            if present {
                selection.push(self.physical_variant(inner_variant)?);
                match_kinds.push(self.match_kind(inner_variant));
                inner_variant += 1;
            } else {
                selection.push(Self::MISSING_VARIANT);
                match_kinds.push(MatchKind::Exact);
            }
        }

        self.selection = Some(selection);
        self.match_kinds = Some(match_kinds);
        self.missing_variant = Some(vec![0x55; self.bytes_per_variant]);
        Some(self)
    }

    pub(crate) fn slice(&self, start: usize, count: usize) -> Option<&[u8]> {
        if count == 0 {
            return Some(&self.data[..0]);
        }
        let physical_start = if let Some(selection) = self.selection.as_deref() {
            let end = start.checked_add(count)?;
            if end > selection.len() {
                return None;
            }
            let base = *selection.get(start)?;
            if base == Self::MISSING_VARIANT {
                return if count == 1 {
                    self.missing_variant.as_deref()
                } else {
                    None
                };
            }
            for offset in 1..count {
                let physical = selection[start + offset];
                if physical == Self::MISSING_VARIANT || physical != base + offset {
                    return None;
                }
            }
            base
        } else {
            let end = start.checked_add(count)?;
            if end > self.physical_n_variants {
                return None;
            }
            start
        };
        let byte_start = physical_start.checked_mul(self.bytes_per_variant)?;
        let byte_len = count.checked_mul(self.bytes_per_variant)?;
        let byte_end = byte_start.checked_add(byte_len)?;
        if byte_end > self.data.len() {
            return None;
        }
        Some(&self.data[byte_start..byte_end])
    }

    pub(crate) fn n_variants(&self) -> usize {
        self.selection
            .as_deref()
            .map_or(self.physical_n_variants, <[usize]>::len)
    }

    pub(crate) fn match_kind(&self, logical_variant: usize) -> MatchKind {
        self.match_kinds
            .as_deref()
            .and_then(|kinds| kinds.get(logical_variant))
            .copied()
            .unwrap_or(MatchKind::Exact)
    }

    pub(crate) fn sample_selection(&self) -> Option<&[usize]> {
        self.sample_selection.as_deref()
    }

    pub(crate) fn sample_byte_masks(&self) -> Option<&[u8]> {
        self.sample_byte_masks.as_deref()
    }

    pub(crate) fn moments(
        &self,
        logical_variant: usize,
        n_samples: usize,
    ) -> Option<(f64, f64, usize)> {
        let bytes = self.slice(logical_variant, 1)?;
        let swapped = self.match_kind(logical_variant) == MatchKind::Swap;
        match (self.sample_selection(), self.sample_byte_masks()) {
            (Some(selection), Some(masks)) => {
                packed_variant_moments_selected(bytes, masks, selection.len(), swapped)
            }
            (Some(_), None) => None,
            (None, None) => Some(packed_variant_moments(bytes, n_samples, swapped)),
            (None, Some(_)) => None,
        }
    }

    /// Count missing hard calls for every logical sample across this view.
    ///
    /// Variant selections stay packed, and an optional sample selection is
    /// mapped from physical BED lanes back to its logical row number. Each
    /// Rayon worker owns one `n_samples` counter vector; reduction is linear in
    /// the cohort size rather than the genotype matrix size.
    pub(crate) fn sample_missing_counts<P: FitProgressObserver>(
        &self,
        n_samples: usize,
        progress: Option<&StageProgressHandle<P>>,
    ) -> Option<Vec<usize>> {
        let logical_by_physical = match self.sample_selection() {
            Some(selection) => {
                if selection.len() != n_samples {
                    return None;
                }
                let physical_capacity = self.bytes_per_variant.checked_mul(4)?;
                let mut logical = vec![usize::MAX; physical_capacity];
                for (logical_row, &physical_row) in selection.iter().enumerate() {
                    let slot = logical.get_mut(physical_row)?;
                    if *slot != usize::MAX {
                        return None;
                    }
                    *slot = logical_row;
                }
                Some(logical)
            }
            None => {
                if n_samples > self.bytes_per_variant.checked_mul(4)? {
                    return None;
                }
                None
            }
        };
        let missing_masks = hard_call_missing_mask_table();

        let n_variants = self.n_variants();
        let workers = rayon::current_num_threads().min(n_variants.max(1));
        (0..workers)
            .into_par_iter()
            .map(|worker| -> Option<Vec<usize>> {
                let mut counts = vec![0usize; n_samples];
                let start = worker * n_variants / workers;
                let end = (worker + 1) * n_variants / workers;
                let mut reported = 0usize;
                for variant in start..end {
                    let bytes = self.slice(variant, 1)?;
                    match logical_by_physical.as_deref() {
                        Some(logical) => {
                            if n_samples < bytes.len() {
                                for (logical_row, &physical_row) in self
                                    .sample_selection()
                                    .expect("sample selection was mapped above")
                                    .iter()
                                    .enumerate()
                                {
                                    let byte = bytes[physical_row / 4];
                                    let code = (byte >> ((physical_row % 4) * 2)) & 0b11;
                                    counts[logical_row] += usize::from(code == 0b01);
                                }
                            } else {
                                const LOW_BITS: u64 = 0x5555_5555_5555_5555;
                                let (chunks, remainder) = bytes.as_chunks::<8>();
                                for (word_idx, chunk) in chunks.iter().enumerate() {
                                    let word = u64::from_le_bytes(*chunk);
                                    let mut missing = word & !(word >> 1) & LOW_BITS;
                                    while missing != 0 {
                                        let bit = missing.trailing_zeros() as usize;
                                        missing &= missing - 1;
                                        let physical_row = word_idx * 32 + bit / 2;
                                        let logical_row = logical[physical_row];
                                        if logical_row != usize::MAX {
                                            counts[logical_row] += 1;
                                        }
                                    }
                                }
                                let byte_offset = chunks.len() * 8;
                                for (remainder_idx, &byte) in remainder.iter().enumerate() {
                                    let mut mask = missing_masks[byte as usize];
                                    while mask != 0 {
                                        let lane = mask.trailing_zeros() as usize;
                                        mask &= mask - 1;
                                        let physical_row = (byte_offset + remainder_idx) * 4 + lane;
                                        let logical_row = logical[physical_row];
                                        if logical_row != usize::MAX {
                                            counts[logical_row] += 1;
                                        }
                                    }
                                }
                            }
                        }
                        None => {
                            const LOW_BITS: u64 = 0x5555_5555_5555_5555;
                            let full_words = n_samples / 32;
                            let (words, remainder) = bytes[..full_words * 8].as_chunks::<8>();
                            debug_assert!(remainder.is_empty());
                            for (word_idx, word_bytes) in words.iter().enumerate() {
                                let word = u64::from_le_bytes(*word_bytes);
                                let mut missing = word & !(word >> 1) & LOW_BITS;
                                while missing != 0 {
                                    let bit = missing.trailing_zeros() as usize;
                                    missing &= missing - 1;
                                    counts[word_idx * 32 + bit / 2] += 1;
                                }
                            }
                            for sample in full_words * 32..n_samples {
                                let byte = bytes[sample / 4];
                                let code = (byte >> ((sample % 4) * 2)) & 0b11;
                                counts[sample] += usize::from(code == 0b01);
                            }
                        }
                    }
                    if variant + 1 - start - reported >= 1_024 {
                        if let Some(progress) = progress {
                            progress.increment(variant + 1 - start - reported);
                        }
                        reported = variant + 1 - start;
                    }
                }
                if let Some(progress) = progress {
                    progress.increment(end - start - reported);
                }
                Some(counts)
            })
            .try_reduce(
                || vec![0usize; n_samples],
                |mut left, right| {
                    for (total, value) in left.iter_mut().zip(right) {
                        *total += value;
                    }
                    Some(left)
                },
            )
    }

    fn physical_variant(&self, logical_variant: usize) -> Option<usize> {
        if let Some(selection) = self.selection.as_deref() {
            selection.get(logical_variant).copied()
        } else if logical_variant < self.physical_n_variants {
            Some(logical_variant)
        } else {
            None
        }
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
        // PLINK/PGEN hard calls already live in a packed representation owned
        // by the source. Decoding that data only to repack an identical second
        // copy costs a complete cohort traversal and, in profiles of the
        // 250k-sample fit, 15% of all cycles. Keep caching for streamed or dense
        // sources that have no packed view; an existing view is the cache.
        let source_is_packed = source.hard_call_packed().is_some();
        let state = if !enable_cache
            || source_is_packed
            || n_samples == 0
            || packed_capacity > cache_budget_bytes()
        {
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
        // The dense fallback has always had a byte budget; the 2-bit path had
        // none, so a biobank-scale fit would ask for the whole packed genotype
        // matrix up front -- 200k samples x 1M variants is ~50 GiB -- and die
        // in the allocator rather than degrade. Re-reading the source is slower
        // than caching it, but it finishes.
        if packed_capacity > cache_budget_bytes() {
            self.state = CacheState::Disabled;
            return;
        }
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
                let total_bytes = bytes_per_variant.checked_mul(*n_variants).unwrap_or(0);
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
                    decode_packed_hard_calls(
                        &packed[byte_start..byte_end],
                        dest,
                        self.n_samples,
                        table,
                    );
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

        if matches!(
            self.state,
            CacheState::ReadyHardCall { .. } | CacheState::ReadyDense { .. }
        ) {
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
                            dense.push_block_range(storage, packed_count, filled, self.n_samples);
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

    fn next_standardized_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
        allele_frequencies: &[f64],
        variant_scales: &[f64],
        ld_weights: Option<&[f64]>,
    ) -> Result<Option<usize>, Self::Error> {
        if !matches!(self.state, CacheState::Disabled) {
            return Ok(None);
        }
        let filled = self.source.next_standardized_block_into(
            max_variants,
            storage,
            allele_frequencies,
            variant_scales,
            ld_weights,
        )?;
        if let Some(filled) = filled {
            self.cursor += filled;
        }
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

    fn block_variant_keys(&self) -> Option<&[VariantKey]> {
        self.source.block_variant_keys()
    }

    fn take_variant_keys(&mut self) -> Option<Vec<VariantKey>> {
        self.source.take_variant_keys()
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
                physical_n_variants: *n_variants,
                selection: None,
                match_kinds: None,
                sample_selection: None,
                sample_byte_masks: None,
                missing_variant: None,
            }),
            CacheState::Disabled => self.source.hard_call_packed(),
            CacheState::BuildingHardCall { .. } | CacheState::BuildingDense { .. } => None,
            CacheState::ReadyDense { .. } => None,
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

fn hard_call_moment_table() -> &'static [[u8; 3]; 256] {
    static TABLE: OnceLock<[[u8; 3]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0u8; 3]; 256];
        for byte in 0u16..256 {
            let mut ones = 0u8;
            let mut twos = 0u8;
            let mut calls = 0u8;
            for offset in 0..4 {
                match ((byte >> (offset * 2)) & 0b11) as u8 {
                    0 => calls += 1,
                    1 => {}
                    2 => {
                        ones += 1;
                        calls += 1;
                    }
                    3 => {
                        twos += 1;
                        calls += 1;
                    }
                    _ => unreachable!(),
                }
            }
            table[byte as usize] = [ones, twos, calls];
        }
        table
    })
}

fn hard_call_missing_mask_table() -> &'static [u8; 256] {
    static TABLE: OnceLock<[u8; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [0u8; 256];
        for byte in 0u16..256 {
            let mut mask = 0u8;
            for lane in 0..4 {
                if ((byte >> (lane * 2)) & 0b11) == 0b01 {
                    mask |= 1 << lane;
                }
            }
            table[byte as usize] = mask;
        }
        table
    })
}

fn packed_variant_moments(bytes: &[u8], n_samples: usize, swap: bool) -> (f64, f64, usize) {
    debug_assert!(bytes.len() >= n_samples.div_ceil(4));

    let full_bytes = n_samples / 4;
    let table = hard_call_moment_table();
    let mut ones = 0usize;
    let mut twos = 0usize;
    let mut calls = 0usize;
    for &byte in &bytes[..full_bytes] {
        let [byte_ones, byte_twos, byte_calls] = table[byte as usize];
        ones += byte_ones as usize;
        twos += byte_twos as usize;
        calls += byte_calls as usize;
    }

    let tail = n_samples % 4;
    if tail > 0 {
        let byte = bytes[full_bytes];
        for offset in 0..tail {
            match (byte >> (offset * 2)) & 0b11 {
                0 => calls += 1,
                1 => {}
                2 => {
                    ones += 1;
                    calls += 1;
                }
                3 => {
                    twos += 1;
                    calls += 1;
                }
                _ => unreachable!(),
            }
        }
    }

    if swap {
        twos = calls - ones - twos;
    }
    let sum = ones + 2 * twos;
    let sum_sq = ones + 4 * twos;
    (sum as f64, sum_sq as f64, calls)
}

fn packed_variant_moments_selected(
    bytes: &[u8],
    sample_byte_masks: &[u8],
    n_samples: usize,
    swap: bool,
) -> Option<(f64, f64, usize)> {
    let mut ones = 0usize;
    let mut twos = 0usize;
    let mut calls = 0usize;
    let mut selected = 0usize;
    let moment_table = hard_call_moment_table();
    if sample_byte_masks.len() < bytes.len() {
        return None;
    }
    for (&byte, &selection_mask) in bytes.iter().zip(sample_byte_masks) {
        selected += selection_mask.count_ones() as usize;
        if selection_mask == 0b1111 {
            let [byte_ones, byte_twos, byte_calls] = moment_table[byte as usize];
            ones += byte_ones as usize;
            twos += byte_twos as usize;
            calls += byte_calls as usize;
        } else {
            let mut retained = selection_mask;
            while retained != 0 {
                let lane = retained.trailing_zeros() as usize;
                retained &= retained - 1;
                match (byte >> (lane * 2)) & 0b11 {
                    0 => calls += 1,
                    1 => {}
                    2 => {
                        ones += 1;
                        calls += 1;
                    }
                    3 => {
                        twos += 1;
                        calls += 1;
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
    if selected != n_samples {
        return None;
    }

    if swap {
        twos = calls - ones - twos;
    }
    let sum = ones + 2 * twos;
    let sum_sq = ones + 4 * twos;
    Some((sum as f64, sum_sq as f64, calls))
}

fn build_sample_byte_masks(bytes_per_variant: usize, selection: &[usize]) -> Option<Vec<u8>> {
    let mut masks = vec![0u8; bytes_per_variant];
    for &physical in selection {
        let mask = masks.get_mut(physical / 4)?;
        *mask |= 1 << (physical % 4);
    }
    Some(masks)
}

/// Visit a packed selected-sample view in physical byte order. A fully retained
/// byte—the dominant case after ordinary call-rate QC—takes one mask branch and
/// emits four consecutive logical rows without division, lookup, or gathering.
#[inline(always)]
pub(crate) fn for_each_packed_masked_code(
    bytes: &[u8],
    masks: &[u8],
    n_samples: usize,
    mut visit: impl FnMut(usize, u8),
) -> Option<()> {
    let mut logical = 0usize;
    for (&byte, &mask) in bytes.iter().zip(masks) {
        if mask == 0b1111 {
            visit(logical, byte & 0b11);
            visit(logical + 1, (byte >> 2) & 0b11);
            visit(logical + 2, (byte >> 4) & 0b11);
            visit(logical + 3, (byte >> 6) & 0b11);
            logical += 4;
        } else {
            let mut retained = mask;
            while retained != 0 {
                let lane = retained.trailing_zeros() as usize;
                retained &= retained - 1;
                visit(logical, (byte >> (lane * 2)) & 0b11);
                logical += 1;
            }
        }
    }
    (logical == n_samples).then_some(())
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
    standardized_sums_sq: Vec<f64>,
    block_sums: Vec<f64>,
    block_sums_sq: Vec<f64>,
    block_calls: Vec<usize>,
    finalized_len: Option<usize>,
    write_pos: usize,
}

fn finalize_variant_moments(sum: f64, sum_sq: f64, calls: usize) -> (f64, f64, f64) {
    if calls == 0 {
        return (0.0, HWE_SCALE_FLOOR, 0.0);
    }

    let mean_genotype = sum / (calls as f64);
    let allele_freq = (mean_genotype / 2.0).clamp(0.0, 1.0);
    let variance = (2.0 * allele_freq * (1.0 - allele_freq)).max(HWE_VARIANCE_EPSILON);
    let scale = variance.sqrt().max(HWE_SCALE_FLOOR);
    let centered_sum_sq = (sum_sq - sum * mean_genotype).max(0.0);
    let standardized_sum_sq = centered_sum_sq / (scale * scale);
    (allele_freq, scale, standardized_sum_sq)
}

impl VariantStatsCache {
    fn new(block_capacity: usize, variant_capacity_hint: usize) -> Self {
        let frequencies = Vec::with_capacity(variant_capacity_hint);
        let scales = Vec::with_capacity(variant_capacity_hint);
        let standardized_sums_sq = Vec::with_capacity(variant_capacity_hint);
        Self {
            frequencies,
            scales,
            standardized_sums_sq,
            block_sums: vec![0.0; block_capacity],
            block_sums_sq: vec![0.0; block_capacity],
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
            let sums_sq_slice = &mut self.block_sums_sq[..filled];
            let calls_slice = &mut self.block_calls[..filled];

            let use_parallel = filled >= 32 && par.degree() > 1;

            if use_parallel {
                sums_slice
                    .par_iter_mut()
                    .zip(sums_sq_slice.par_iter_mut())
                    .zip(calls_slice.par_iter_mut())
                    .zip(block.par_col_iter())
                    .for_each(|(((sum_slot, sum_sq_slot), calls_slot), column)| {
                        let contiguous = column
                            .try_as_col_major()
                            .expect("variant block column must be contiguous");
                        let (sum, sum_sq, calls) =
                            sum_sum_sq_and_count_finite(contiguous.as_slice());
                        *sum_slot = sum;
                        *sum_sq_slot = sum_sq;
                        *calls_slot = calls;
                    });
            } else {
                sums_slice
                    .iter_mut()
                    .zip(sums_sq_slice.iter_mut())
                    .zip(calls_slice.iter_mut())
                    .zip(block.col_iter())
                    .for_each(|(((sum_slot, sum_sq_slot), calls_slot), column)| {
                        let contiguous = column
                            .try_as_col_major()
                            .expect("variant block column must be contiguous");
                        let (sum, sum_sq, calls) =
                            sum_sum_sq_and_count_finite(contiguous.as_slice());
                        *sum_slot = sum;
                        *sum_sq_slot = sum_sq;
                        *calls_slot = calls;
                    });
            }
        }

        let end = variant_range.end;
        self.ensure_capacity(end);

        let freq_slice = &mut self.frequencies[variant_range.clone()];
        let scale_slice = &mut self.scales[variant_range.clone()];
        let standardized_sums_sq_slice = &mut self.standardized_sums_sq[variant_range.clone()];
        let sums_slice = &self.block_sums[..filled];
        let sums_sq_slice = &self.block_sums_sq[..filled];
        let calls_slice = &self.block_calls[..filled];

        for idx in 0..filled {
            let sum = sums_slice[idx];
            let sum_sq = sums_sq_slice[idx];
            let calls = calls_slice[idx];
            let (frequency, scale, standardized_sum_sq) =
                finalize_variant_moments(sum, sum_sq, calls);
            freq_slice[idx] = frequency;
            scale_slice[idx] = scale;
            standardized_sums_sq_slice[idx] = standardized_sum_sq;
        }

        self.write_pos = end;

        assert!(self.frequencies.len() >= end);
        assert!(self.scales.len() >= end);
    }

    fn finalize(&mut self) {
        self.frequencies.truncate(self.write_pos);
        self.scales.truncate(self.write_pos);
        self.standardized_sums_sq.truncate(self.write_pos);
        self.finalized_len = Some(self.write_pos);
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.finalized_len.unwrap_or(self.write_pos)
    }

    fn into_parts(self) -> Option<(HweScaler, Vec<f64>)> {
        if self.finalized_len.is_some() {
            Some((
                HweScaler::new(self.frequencies, self.scales),
                self.standardized_sums_sq,
            ))
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
            self.standardized_sums_sq.reserve_exact(additional_capacity);
        }

        let additional = required - self.frequencies.len();
        self.frequencies
            .extend(std::iter::repeat_n(0.0, additional));
        self.scales.extend(std::iter::repeat_n(0.0, additional));
        self.standardized_sums_sq
            .extend(std::iter::repeat_n(0.0, additional));
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
        record_simd_lane_diagnostic("default lanes4 architecture", SimdLaneSelection::Lanes4)
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
    let (chunks, remainder) = values.as_chunks_mut::<2>();
    for chunk in chunks {
        let lane = Simd::<f64, 2>::from_array(*chunk);
        let lane_values = lane.to_array();
        let mut result = [0.0; 2];
        if lane_values[0].is_finite() {
            result[0] = (lane_values[0] - mean) * inv;
        }
        if lane_values[1].is_finite() {
            result[1] = (lane_values[1] - mean) * inv;
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

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn standardize_column_simd_impl_lanes4(values: &mut [f64], mean: f64, inv: f64) {
    let (chunks, remainder) = values.as_chunks_mut::<4>();
    for chunk in chunks {
        let lane = Simd::<f64, 4>::from_array(*chunk);
        let lane_values = lane.to_array();
        let mut result = [0.0; 4];
        if lane_values[0].is_finite() {
            result[0] = (lane_values[0] - mean) * inv;
        }
        if lane_values[1].is_finite() {
            result[1] = (lane_values[1] - mean) * inv;
        }
        if lane_values[2].is_finite() {
            result[2] = (lane_values[2] - mean) * inv;
        }
        if lane_values[3].is_finite() {
            result[3] = (lane_values[3] - mean) * inv;
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
    let (value_chunks, value_remainder) = values.as_chunks_mut::<2>();
    let (mask_chunks, mask_remainder) = mask.as_chunks_mut::<2>();

    debug_assert_eq!(value_chunks.len(), mask_chunks.len());
    debug_assert_eq!(value_remainder.len(), mask_remainder.len());

    for (value_chunk, mask_chunk) in value_chunks.iter_mut().zip(mask_chunks.iter_mut()) {
        let lane = Simd::<f64, 2>::from_array(*value_chunk);
        let lane_values = lane.to_array();
        let mut result = [0.0; 2];
        let mut mask_values = [0.0; 2];
        if lane_values[0].is_finite() {
            result[0] = (lane_values[0] - mean) * inv;
            mask_values[0] = 1.0;
        }
        if lane_values[1].is_finite() {
            result[1] = (lane_values[1] - mean) * inv;
            mask_values[1] = 1.0;
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

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn standardize_column_with_mask_simd_impl_lanes4(
    values: &mut [f64],
    mask: &mut [f64],
    mean: f64,
    inv: f64,
) {
    let (value_chunks, value_remainder) = values.as_chunks_mut::<4>();
    let (mask_chunks, mask_remainder) = mask.as_chunks_mut::<4>();

    debug_assert_eq!(value_chunks.len(), mask_chunks.len());
    debug_assert_eq!(value_remainder.len(), mask_remainder.len());

    for (value_chunk, mask_chunk) in value_chunks.iter_mut().zip(mask_chunks.iter_mut()) {
        let lane = Simd::<f64, 4>::from_array(*value_chunk);
        let lane_values = lane.to_array();
        let mut result = [0.0; 4];
        let mut mask_values = [0.0; 4];
        if lane_values[0].is_finite() {
            result[0] = (lane_values[0] - mean) * inv;
            mask_values[0] = 1.0;
        }
        if lane_values[1].is_finite() {
            result[1] = (lane_values[1] - mean) * inv;
            mask_values[1] = 1.0;
        }
        if lane_values[2].is_finite() {
            result[2] = (lane_values[2] - mean) * inv;
            mask_values[2] = 1.0;
        }
        if lane_values[3].is_finite() {
            result[3] = (lane_values[3] - mean) * inv;
            mask_values[3] = 1.0;
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
#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
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
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
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
#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
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
fn sum_sum_sq_and_count_finite(values: &[f64]) -> (f64, f64, usize) {
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
                return sum_sum_sq_and_count_finite_avx(values);
            }

            #[cfg(all(
                not(any(target_arch = "x86", target_arch = "x86_64")),
                any(target_arch = "aarch64", target_arch = "wasm32")
            ))]
            {
                log::debug!(
                    "Using generic four-lane moment implementation for non-x86 architecture"
                );
                sum_sum_sq_and_count_finite_impl_lanes4(values)
            }

            #[cfg(not(any(
                target_arch = "x86",
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            )))]
            {
                log::warn!(
                    "Falling back to two-lane moment implementation despite four-lane selection"
                );
                return sum_sum_sq_and_count_finite_impl_lanes2(values);
            }
        }
        _ => sum_sum_sq_and_count_finite_impl_lanes2(values),
    }
}

#[inline(always)]
fn sum_sum_sq_and_count_finite_impl_lanes2(values: &[f64]) -> (f64, f64, usize) {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    let (chunks, remainder) = values.as_chunks::<2>();
    for chunk in chunks {
        let lane = Simd::<f64, 2>::from_array(*chunk);
        let lane_values = lane.to_array();
        if lane_values[0].is_finite() {
            sum += lane_values[0];
            sum_sq = lane_values[0].mul_add(lane_values[0], sum_sq);
            count += 1;
        }
        if lane_values[1].is_finite() {
            sum += lane_values[1];
            sum_sq = lane_values[1].mul_add(lane_values[1], sum_sq);
            count += 1;
        }
    }

    for &value in remainder {
        if value.is_finite() {
            sum += value;
            sum_sq = value.mul_add(value, sum_sq);
            count += 1;
        }
    }

    (sum, sum_sq, count)
}

#[cfg(any(
    target_feature = "avx",
    target_arch = "aarch64",
    target_arch = "wasm32"
))]
#[inline(always)]
fn sum_sum_sq_and_count_finite_impl_lanes4(values: &[f64]) -> (f64, f64, usize) {
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0usize;

    let (chunks, remainder) = values.as_chunks::<4>();
    for chunk in chunks {
        let lane = Simd::<f64, 4>::from_array(*chunk);
        let lane_values = lane.to_array();
        if lane_values[0].is_finite() {
            sum += lane_values[0];
            sum_sq = lane_values[0].mul_add(lane_values[0], sum_sq);
            count += 1;
        }
        if lane_values[1].is_finite() {
            sum += lane_values[1];
            sum_sq = lane_values[1].mul_add(lane_values[1], sum_sq);
            count += 1;
        }
        if lane_values[2].is_finite() {
            sum += lane_values[2];
            sum_sq = lane_values[2].mul_add(lane_values[2], sum_sq);
            count += 1;
        }
        if lane_values[3].is_finite() {
            sum += lane_values[3];
            sum_sq = lane_values[3].mul_add(lane_values[3], sum_sq);
            count += 1;
        }
    }

    for &value in remainder {
        if value.is_finite() {
            sum += value;
            sum_sq = value.mul_add(value, sum_sq);
            count += 1;
        }
    }

    (sum, sum_sq, count)
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx"
))]
#[target_feature(enable = "avx")]
/// # Safety
/// Callers must ensure AVX is supported by the running CPU. Runtime feature
/// checks protect all invocations of this function.
unsafe fn sum_sum_sq_and_count_finite_avx(values: &[f64]) -> (f64, f64, usize) {
    sum_sum_sq_and_count_finite_impl_lanes4(values)
}

#[derive(Clone, Debug)]
pub(crate) struct ProjectionModelCache {
    packed_score_vectors: Vec<f64>,
    global_info_packed: Vec<f64>,
}

fn projection_packed_tri_size(components: usize) -> usize {
    components.saturating_mul(components.saturating_add(1)) / 2
}

fn build_projection_model_cache(
    scaler: &HweScaler,
    loadings: MatRef<'_, f64>,
    ld: Option<&LdWeights>,
) -> ProjectionModelCache {
    let n_variants = loadings.nrows();
    let components = loadings.ncols();
    let packed_info_size = projection_packed_tri_size(components);
    let mut packed_score_vectors = vec![0.0; n_variants * components * 3];
    let mut global_info_packed = vec![0.0; packed_info_size];
    let freqs = scaler.allele_frequencies();
    let scales = scaler.variant_scales();
    let weights = ld.map(|weights| weights.weights.as_slice()).unwrap_or(&[]);

    for variant in 0..n_variants {
        let mean = 2.0 * freqs[variant];
        let denom = scales[variant];
        let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
        let weight = weights.get(variant).copied().unwrap_or(1.0);
        let coeffs = [
            (0.0 - mean) * inv * weight,
            (1.0 - mean) * inv * weight,
            (2.0 - mean) * inv * weight,
        ];
        let score_offset = variant * components * 3;

        for component in 0..components {
            let loading = loadings[(variant, component)];
            packed_score_vectors[score_offset + component] = coeffs[0] * loading;
            packed_score_vectors[score_offset + components + component] = coeffs[1] * loading;
            packed_score_vectors[score_offset + components * 2 + component] = coeffs[2] * loading;
        }

        let mut packed_idx = 0usize;
        for row in 0..components {
            let row_loading = loadings[(variant, row)];
            for col in row..components {
                global_info_packed[packed_idx] += row_loading * loadings[(variant, col)];
                packed_idx += 1;
            }
        }
    }

    ProjectionModelCache {
        packed_score_vectors,
        global_info_packed,
    }
}

/// Which route produced the eigenpairs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum FitSolver {
    /// The covariance was formed and decomposed exactly.
    Dense,
    /// Adaptive randomized block Lanczos over the streaming operator.
    BlockKrylov,
}

/// How the eigensolver terminated, recorded on the model and serialized with it.
///
/// A fit that ran out of passes short of its tolerance used to leave a line on
/// stderr and an artifact byte-indistinguishable from a converged one. In a
/// scientific pipeline that is the dangerous failure: the scores are wrong by an
/// unknown amount, the file says nothing, and whoever reads it a year later has
/// no way to find out. Everything needed to judge the solve therefore travels
/// with the solve.
///
/// The measured quantities are `Option` because a route that does not measure
/// one must say so. Reporting an unmeasured residual as `0.0` would read as
/// "perfectly converged", which is the same lie in a different font.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FitDiagnostics {
    pub solver: FitSolver,
    /// Whether every stopping criterion was met. `false` means the components
    /// are the best estimate available, not a finished answer.
    pub converged: bool,
    /// Passes over the genome, i.e. applications of the streaming operator.
    pub passes: usize,
    /// Worst relative Ritz residual `‖Cu − θu‖ / θ` over the returned pairs.
    #[serde(default)]
    pub max_relative_residual: Option<f64>,
    /// `1 − MEV` between the final two top-k Ritz subspaces.
    #[serde(default)]
    pub subspace_delta: Option<f64>,
    /// `(θ_k − θ_{k+1}) / θ_k` at the requested boundary, when it was observed.
    /// A small gap means PC k and PC k+1 are not individually identified, which
    /// no amount of extra iteration fixes.
    #[serde(default)]
    pub boundary_gap: Option<f64>,
    /// Times the Krylov basis was restarted under its memory budget.
    pub restarts: usize,
}

impl FitDiagnostics {
    /// The dense reference path decomposes the covariance directly: there is no
    /// iteration to report, nothing left to converge, and exactly one traversal
    /// of the genome behind the answer.
    fn exact_dense() -> Self {
        Self {
            solver: FitSolver::Dense,
            converged: true,
            passes: 1,
            max_relative_residual: None,
            subspace_delta: None,
            boundary_gap: None,
            restarts: 0,
        }
    }

    /// The in-core partial solve over an already-formed Gram: still one genome
    /// pass, but the eigenpairs themselves come from an iteration that can stop
    /// short of the requested count.
    fn dense_partial(converged: bool) -> Self {
        Self {
            solver: FitSolver::Dense,
            converged,
            ..Self::exact_dense()
        }
    }
}

/// Gate between a solve's termination record and a model built on it.
///
/// `converged: false` means the components are the best estimate available and
/// not a finished answer, and there is no way to tell from the artifact by how
/// much they are wrong — so by default they do not become an artifact.
/// `allow_unconverged` is the deliberate exception for a caller who wants the
/// estimate anyway; it suppresses only the refusal, never the record, and the
/// model it permits still reports `converged: false` in its diagnostics.
///
/// `None` diagnostics are not a refusal: they come from the degenerate exits
/// that return no eigenpairs at all, which the caller rejects on its own terms.
fn require_converged(
    diagnostics: Option<&FitDiagnostics>,
    allow_unconverged: bool,
) -> Result<(), HwePcaError> {
    let Some(diagnostics) = diagnostics else {
        return Ok(());
    };

    if diagnostics.converged || allow_unconverged {
        return Ok(());
    }

    // A route that did not measure a quantity says so rather than printing a
    // zero, for the same reason `FitDiagnostics` stores it as `Option`.
    let describe = |value: Option<f64>| {
        value.map_or_else(
            || String::from("unmeasured"),
            |value| format!("{value:.3e}"),
        )
    };

    Err(HwePcaError::Eigen(format!(
        "PCA eigensolver stopped after {} covariance passes without reaching its tolerance \
         (worst relative Ritz residual {}, subspace change {}); refusing to build a model on an \
         unconverged subspace. Pass --allow-unconverged in the CLI or set \
         FitOptions::allow_unconverged to accept the best available estimate, which is recorded \
         as unconverged in the model's fit diagnostics.",
        diagnostics.passes,
        describe(diagnostics.max_relative_residual),
        describe(diagnostics.subspace_delta),
    )))
}

#[derive(Clone, Debug)]
pub struct HwePcaModel {
    n_samples: usize,
    n_variants: usize,
    scaler: HweScaler,
    eigenvalues: Vec<f64>,
    /// Total variance of the standardized genotype matrix, i.e. the trace of the
    /// covariance (the sum of the *entire* eigenvalue spectrum, not just the
    /// retained components). Stored so `explained_variance_ratio` can normalize
    /// against the full variance rather than only the kept PCs.
    total_variance: f64,
    singular_values: Vec<f64>,
    sample_basis: Mat<f64>,
    sample_scores: Mat<f64>,
    loadings: Mat<f64>,
    component_weighted_norms_sq: Vec<f64>,
    variant_keys: Option<Vec<VariantKey>>,
    ld: Option<LdWeights>,
    genome_build: Option<String>,
    /// How the eigensolve terminated. `None` on projection-only stubs and on
    /// models deserialized from files written before this was recorded, which
    /// is exactly the population whose convergence is genuinely unknown.
    diagnostics: Option<FitDiagnostics>,
    projection_cache: Arc<ProjectionModelCache>,
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
        let block_capacity =
            adaptive_block_capacity(source.block_storage_samples(), n_variants_hint);
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

        // Dense or matrix-free is a question about work, not about whether an
        // n×n allocation happens to fit; see `covariance_computation_mode`.
        let gram_budget = gram_matrix_budget_bytes();
        let gram_bytes = gram_matrix_size_bytes(n_samples);
        let gram_mode =
            covariance_computation_mode(n_samples, n_variants_hint, target_components, gram_budget);

        let mut cached_source = CachedVariantBlockSource::new(
            source,
            matches!(gram_mode, CovarianceComputationMode::Partial) || options.cache_source,
        );
        let source = &mut cached_source;

        // Compute LD weights first if requested (applies to both paths).
        //
        // That pass streams every variant through a `VariantStatsCache`, so it
        // ends holding exactly the scaler and observed-variant count the fit
        // needs next — same source, same block width, same variant order, same
        // `ensure_statistics` arithmetic, and the source cache it warmed serves
        // both. Recomputing them costs a second full traversal of the genome to
        // arrive at a bit-identical answer, which at 500k variants is not a
        // micro-optimization but a whole redundant read of the dataset.
        let (ld_weights_arc, ld_weights, ld_pass_stats) = if let Some(ld_cfg) = ld_config {
            let (ld_scaler, ld_standardized_sums_sq, ld_observed_variants, ld_weights_computed) =
                compute_stats_and_ld_weights(
                    source,
                    block_capacity,
                    ld_cfg,
                    n_variants_hint,
                    progress,
                    par,
                )?;
            let ld_arc = Arc::<[f64]>::from(ld_weights_computed.weights.clone().into_boxed_slice());
            (
                Some(ld_arc),
                Some(ld_weights_computed),
                Some((ld_scaler, ld_standardized_sums_sq, ld_observed_variants)),
            )
        } else {
            (None, None, None)
        };

        // Reset source after LD computation
        if ld_weights_arc.is_some() {
            source
                .reset()
                .map_err(|err| HwePcaError::Source(Box::new(err)))?;
        }

        // Choose between dense and matrix-free paths
        let (decomposition, scaler, standardized_sums_sq, observed_variants) = match gram_mode {
            CovarianceComputationMode::Dense => {
                // PATH A: Dense - the exact reference solve, for problems small
                // enough that forming C is the cheaper way to get it.
                //
                // This path does not reuse `ld_pass_stats`: it recomputes the
                // statistics inside the traversal it has to make anyway to
                // accumulate the covariance, so there is no pass to save, and
                // the fused loop stays a single self-contained pass.
                let (scaler, standardized_sums_sq, observed_variants, covariance) =
                    compute_stats_and_covariance_blockwise(
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

                        // Only retain genuinely positive eigenvalues. Rank-deficient
                        // data yields zero or tiny-negative numerical artifacts that
                        // are not real principal components; the matrix-free and
                        // dense-operator paths apply the same `EIGENVALUE_EPSILON`
                        // filter, so the normal dense path must agree.
                        let positive = (0..n_eig)
                            .filter(|&i| eigenvalues_diag[i] > EIGENVALUE_EPSILON)
                            .count();
                        let keep = positive.min(target_components);

                        // Create index-value pairs for sorting (descending by eigenvalue)
                        let mut indexed_values: Vec<(usize, f64)> =
                            (0..n_eig).map(|i| (i, eigenvalues_diag[i])).collect();

                        // Select top `keep` positive components by eigenvalue magnitude
                        let kept = select_top_k_desc(&mut indexed_values, keep);

                        // Extract selected values and vectors
                        let selected_values: Vec<f64> =
                            indexed_values[..kept].iter().map(|(_, val)| *val).collect();
                        let selected_vectors =
                            Mat::from_fn(eigenvectors_mat.nrows(), kept, |row, col| {
                                let original_col = indexed_values[col].0;
                                eigenvectors_mat[(row, original_col)]
                            });

                        Eigenpairs {
                            values: selected_values,
                            vectors: selected_vectors,
                            diagnostics: Some(FitDiagnostics::exact_dense()),
                        }
                    }
                    Err(e) => {
                        return Err(HwePcaError::Eigen(format!(
                            "Eigendecomposition failed: {:?}",
                            e
                        )));
                    }
                };

                (
                    decomposition,
                    scaler,
                    standardized_sums_sq,
                    observed_variants,
                )
            }
            CovarianceComputationMode::Partial => {
                // PATH B: Matrix-free - For biobank-scale datasets
                log::info!(
                    "Using matrix-free eigensolver (forming the Gram matrix would take {} bytes \
                     and O(p·n²) work)",
                    gram_bytes.unwrap_or(usize::MAX)
                );

                // Statistics: reused from the LD pass when there was one,
                // computed in a pass of their own when there was not.
                let (scaler, standardized_sums_sq, observed_variants) = match (
                    ld_pass_stats,
                    options.precomputed_variant_statistics.as_ref(),
                ) {
                    (Some(stats), _) => stats,
                    (None, Some(stats)) if stats.matches(n_samples, n_variants_hint) => {
                        progress
                            .on_stage_start(FitProgressStage::AlleleStatistics, n_variants_hint);
                        progress
                            .on_stage_advance(FitProgressStage::AlleleStatistics, n_variants_hint);
                        progress.on_stage_finish(FitProgressStage::AlleleStatistics);
                        stats.cloned_parts()
                    }
                    (None, Some(_)) => {
                        return Err(HwePcaError::InvalidInput(
                            "precomputed variant statistics do not match the fit source",
                        ));
                    }
                    (None, None) => {
                        progress
                            .on_stage_start(FitProgressStage::AlleleStatistics, n_variants_hint);
                        let stats_progress = StageProgressHandle::new(
                            Arc::clone(progress),
                            FitProgressStage::AlleleStatistics,
                        );
                        compute_variant_statistics(
                            source,
                            block_capacity,
                            par,
                            stats_progress,
                            n_variants_hint,
                        )?
                    }
                };

                // Every adaptive solver application is one complete genome
                // pass. The handle lives in the operator so decoded-block
                // completion, rather than mere reads, drives the pass ETA.
                progress.on_stage_start(FitProgressStage::GramMatrix, observed_variants);
                let gram_progress_handle = Some(StageProgressHandle::new(
                    Arc::clone(progress),
                    FitProgressStage::GramMatrix,
                ));
                let operator = StandardizedCovarianceOp::new(
                    source,
                    block_capacity,
                    n_variants_hint,
                    observed_variants,
                    scaler.clone(),
                    ld_weights_arc.clone(),
                    gram_progress_handle.clone(),
                );

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

                (
                    decomposition_result?,
                    scaler,
                    standardized_sums_sq,
                    observed_variants,
                )
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

        // Refuse here, before the loadings pass: everything below is defined
        // relative to a subspace the solver itself declined to certify, and
        // refining it against the genotypes cannot recover what the solve did
        // not find — it only spreads an unmeasured error into more numbers that
        // look finished. Stopping now also saves the genome traversal those
        // numbers would have cost.
        require_converged(
            decomposition.diagnostics.as_ref(),
            options.allow_unconverged,
        )?;

        // One last traversal of the genome, and every number the model stores is
        // derived from what that traversal saw.
        //
        // The pass forms `B = Xᵀ·U`, the cross-product of each variant with the
        // sample basis the eigensolver returned, and — out of the same blocks,
        // for no extra genotype read — the small `k×k` Gram `BᵀB`, which is
        // `(n−1)·Uᵀ·C·U`: the covariance restricted to the returned subspace,
        // written in that subspace's own coordinates.
        let (mut loadings, restricted_gram) = compute_loading_cross_products(
            source,
            &scaler,
            variant_count,
            block_capacity,
            decomposition.vectors.as_ref(),
            ld_weights_arc.as_deref(),
            progress,
            par,
        )?;

        // Total variance = trace(covariance) = ‖X‖²_F / (n−1), where X is the
        // standardized (optionally LD-weighted) genotype matrix. Its per-variant
        // squared norms were accumulated during the mandatory allele-statistics
        // traversal, avoiding another scalar scan over every standardized call
        // in the loadings pass.
        let standardized_frobenius_sq =
            weighted_standardized_frobenius_sq(&standardized_sums_sq, ld_weights_arc.as_deref());
        let total_variance = standardized_frobenius_sq / (n_samples - 1) as f64;

        // Rayleigh-Ritz: diagonalize that restricted covariance and rotate the
        // basis onto its eigenvectors.
        //
        // An iterative solve stops at *a* basis for the top subspace, not at the
        // eigenvectors inside it; the dense route returns eigenvectors, but of a
        // covariance accumulated in a different pass from this one. Either way
        // `Uᵀ·C·U` is only approximately diagonal on arrival, and its
        // off-diagonal mass is precisely the error. Rotating by its eigenvectors
        // annihilates that mass against the genotypes streamed *here*, so the
        // eigenvalues, singular values, scores and loadings assembled below all
        // describe one and the same matrix instead of three nearby ones.
        let (eigenvalues, rotation) = rayleigh_ritz_rotation(restricted_gram.as_ref(), n_samples)?;
        if eigenvalues.is_empty() {
            return Err(HwePcaError::Eigen(
                "No component survived the final refinement against the streamed genotypes; \
                 increase cohort size or review input data"
                    .into(),
            ));
        }

        // The rotation refines the pairs the solver produced; it says nothing
        // about how that solve terminated, so its record travels unaltered.
        let diagnostics = decomposition.diagnostics;
        let refined = Eigenpairs {
            values: eigenvalues,
            vectors: rotate_columns(
                decomposition.vectors,
                rotation.as_ref(),
                block_capacity,
                par,
            ),
            diagnostics,
        };
        loadings = rotate_columns(loadings, rotation.as_ref(), block_capacity, par);

        // σ_i = √((n−1)·λ_i) and scores = U·Σ, both read off the refined
        // eigenvalues, so `λ_i = σ_i²/(n−1)` holds for the quantities stored
        // rather than for a canonical set computed on demand beside them.
        let (singular_values, sample_scores) = build_sample_scores(n_samples, &refined);

        // V_i = B_i/σ_i. `‖B_i‖² = (n−1)·λ_i = σ_i²` and `B_iᵀ·B_j = 0` off the
        // diagonal, both by construction, because `B` was just rotated into the
        // eigenbasis of `BᵀB`: the columns come out orthonormal, not merely unit
        // length, and there is nothing left for a rescaling pass to fix.
        //
        // The `EIGENVALUE_EPSILON` filter above guarantees `σ_i > 0`; the zero
        // branch is what stops a component that somehow slipped past it from
        // becoming a column of infinities.
        for (column, &sigma) in loadings.col_iter_mut().zip(singular_values.iter()) {
            let inverse = if sigma > 0.0 { sigma.recip() } else { 0.0 };
            zip!(column).for_each(|unzip!(value)| {
                *value *= inverse;
            });
        }

        // Euclidean, deliberately: this is the denominator the projector divides
        // the per-axis mass in `global_info_packed` by, and that matrix is
        // `Σ_j L_j·L_jᵀ` with no LD weight anywhere in it. Orthonormal columns
        // put every entry at 1 to rounding; measuring it rather than asserting it
        // keeps the pair consistent for any component the σ guard zeroed out.
        let component_weighted_norms_sq =
            compute_component_weighted_norms_sq(loadings.as_ref(), None);
        let projection_cache = Arc::new(build_projection_model_cache(
            &scaler,
            loadings.as_ref(),
            ld_weights.as_ref(),
        ));

        Ok(Self {
            n_samples,
            n_variants: variant_count,
            scaler,
            eigenvalues: refined.values,
            total_variance,
            singular_values,
            sample_basis: refined.vectors,
            sample_scores,
            loadings,
            component_weighted_norms_sq,
            variant_keys: None,
            ld: ld_weights,
            genome_build: None,
            diagnostics: refined.diagnostics,
            projection_cache,
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

    /// Singular values of the standardized genotype matrix: `σ_i = √((n−1)·λ_i)`.
    ///
    /// One notion of singular value, consistent with everything else the model
    /// carries. [`HwePcaModel::sample_scores`] is `sample_basis()·Σ`,
    /// [`HwePcaModel::variant_loadings`] scaled back up by `Σ` is the
    /// cross-product `Xᵀ·sample_basis()`, and `explained_variance()[i]` is
    /// exactly `σ_i²/(n−1)`.
    ///
    /// There was a second accessor here for the "canonical" values, because a
    /// post-fit rescaling of the loadings multiplied these and left the
    /// eigenvalues alone, breaking that last identity and forcing two
    /// incompatible answers to the same question to coexist. The final
    /// Rayleigh-Ritz rotation removed the need for the rescaling, and with it
    /// the need for the second answer.
    pub fn singular_values(&self) -> &[f64] {
        &self.singular_values
    }

    /// Total variance of the data, i.e. the trace of the covariance (sum of the
    /// full eigenvalue spectrum). Returns `0.0` for projection-only model stubs
    /// loaded from the binary cache, which do not carry fit statistics.
    pub fn total_variance(&self) -> f64 {
        self.total_variance
    }

    /// Fraction of the **total** data variance captured by each retained PC.
    ///
    /// Normalizes against [`HwePcaModel::total_variance`] (the full spectrum), so
    /// the ratios of a truncated fit sum to less than 1. Projection-only models
    /// carry no fit variance and therefore report zero ratios.
    pub fn explained_variance_ratio(&self) -> Vec<f64> {
        if self.total_variance > 0.0 {
            self.eigenvalues
                .iter()
                .map(|&lambda| lambda / self.total_variance)
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

    /// How the eigensolve that produced this model terminated.
    ///
    /// `None` means the record is genuinely unavailable — a projection-only
    /// stub, or a model serialized before fits carried one — and specifically
    /// *not* that the fit converged.
    pub fn fit_diagnostics(&self) -> Option<&FitDiagnostics> {
        self.diagnostics.as_ref()
    }

    pub fn set_genome_build(&mut self, build: Option<String>) {
        self.genome_build = build;
    }

    pub(crate) fn projection_packed_score_vectors(&self) -> &[f64] {
        &self.projection_cache.packed_score_vectors
    }

    pub(crate) fn projection_global_info_packed(&self) -> &[f64] {
        &self.projection_cache.global_info_packed
    }

    pub(crate) fn from_projection_binary_parts(
        n_samples: usize,
        n_variants: usize,
        frequencies: Vec<f64>,
        scales: Vec<f64>,
        loadings_col_major: Vec<f64>,
        components: usize,
        component_weighted_norms_sq: Vec<f64>,
        projection_packed_score_vectors: Vec<f64>,
        projection_global_info_packed: Vec<f64>,
        variant_keys: Option<Vec<VariantKey>>,
        ld: Option<LdWeights>,
        genome_build: Option<String>,
    ) -> Result<Self, String> {
        let scaler = HweScaler::new(frequencies, scales);
        let loadings = MatrixData {
            nrows: n_variants,
            ncols: components,
            data: loadings_col_major,
        }
        .into_mat()?;
        let component_weighted_norms_sq = if component_weighted_norms_sq.len() == components {
            component_weighted_norms_sq
        } else {
            compute_component_weighted_norms_sq(
                loadings.as_ref(),
                ld.as_ref().map(|weights| weights.weights.as_slice()),
            )
        };
        let projection_cache = if projection_packed_score_vectors.len()
            == n_variants * components * 3
            && projection_global_info_packed.len() == projection_packed_tri_size(components)
        {
            Arc::new(ProjectionModelCache {
                packed_score_vectors: projection_packed_score_vectors,
                global_info_packed: projection_global_info_packed,
            })
        } else {
            Arc::new(build_projection_model_cache(
                &scaler,
                loadings.as_ref(),
                ld.as_ref(),
            ))
        };

        Ok(Self {
            n_samples,
            n_variants,
            scaler,
            eigenvalues: vec![0.0; components],
            total_variance: 0.0,
            singular_values: vec![0.0; components],
            sample_basis: Mat::zeros(0, components),
            sample_scores: Mat::zeros(0, components),
            loadings,
            component_weighted_norms_sq,
            variant_keys,
            ld,
            genome_build,
            // A projection-only stub carries loadings and nothing about the fit
            // that produced them; inventing a convergence record here would be
            // asserting something this constructor cannot know.
            diagnostics: None,
            projection_cache,
        })
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
    progress: Option<StageProgressHandle<P>>,
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
        progress: Option<StageProgressHandle<P>>,
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
            progress,
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
            progress: _,
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

        // Narrow right-hand sides only — a throughput gate, not a correctness
        // one. The packed kernel is a serial walk over variants with no rayon
        // anywhere in it; what it buys is never materializing an f64 tile, and
        // that only pays while there are a handful of columns to project. The
        // general path below decodes a tile and hands it to faer's `matmul`
        // with `par`, which is parallel across every core.
        //
        // Past a few columns the trade reverses decisively: a block solver
        // asking for ~30 columns would run 30× the per-pass work on one core
        // while the tile path spreads the same work over all of them. A
        // measured baseline fit sat at 104% CPU — one core — for exactly this
        // reason. Enabling the packed kernel for wide blocks is a pessimization.
        //
        // The real fix is to parallelize the kernel over disjoint sample ranges
        // (a reduction to form the projection, then a disjoint scatter), after
        // which this gate can go. Until then it stays.
        if rhs.ncols() <= PACKED_RHS_MAX_COLS && self.try_apply_hardcall_packed(out.rb_mut(), rhs) {
            return;
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
                standardized: bool,
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
        let allele_frequencies = self.scaler.allele_frequencies();
        let variant_scales = self.scaler.variant_scales();
        let ld_weights = self.ld_weights.as_deref();

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
                        match source.next_standardized_block_into(
                            block_capacity,
                            buffer_slice,
                            &allele_frequencies[start..],
                            &variant_scales[start..],
                            ld_weights.map(|weights| &weights[start..]),
                        ) {
                            Ok(Some(filled)) => Ok((filled, true)),
                            Ok(None) => source
                                .next_block_into(block_capacity, buffer_slice)
                                .map(|filled| (filled, false)),
                            Err(error) => Err(error),
                        }
                    };

                    match filled_res {
                        Ok((filled, standardized)) => {
                            if filled == 0 {
                                let _ = filled_sender.send(PrefetchMessage::End);
                                break;
                            }

                            let _ = filled_sender.send(PrefetchMessage::Data {
                                id,
                                filled,
                                start,
                                standardized,
                            });
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
                    PrefetchMessage::Data {
                        id,
                        filled,
                        start,
                        standardized,
                    } => {
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
                        if !standardized {
                            self.standardize_block_in_place(
                                block.rb_mut(),
                                variant_range.clone(),
                                par,
                            );
                        }

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
                        if let Some(progress) = self.progress.as_ref() {
                            progress.advance(processed);
                        }

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
    /// How these pairs were obtained. `None` only on the degenerate returns
    /// where no solve was attempted at all — every one of which leaves the
    /// values empty and so ends the fit in an error before a model exists.
    diagnostics: Option<FitDiagnostics>,
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

/// Variants per streamed tile, sized from the sample count rather than fixed.
///
/// A tile is `n_samples × block_capacity` f64s, and the covariance operator
/// holds **two** of them (the prefetch double-buffer) plus a projection temp.
/// At the historical fixed 2048 that is ~3.3 GiB per buffer at 200k samples and
/// ~8 GiB at 500k — scratch alone, before the Krylov basis.
///
/// The tile exists only to amortize per-block overhead, and there is nothing
/// mathematically special about any particular width: the arithmetic is
/// identical whatever the tiling. So the budget decides, all the way down. A
/// former floor of 256 variants meant 2M samples still demanded ~7.6 GiB of
/// scratch on a machine whose plan said it could afford a tenth of that; the
/// only real floor is one variant, because a zero-width tile makes no progress.
///
/// The result is a pure function of `(n_samples, n_variants_hint)` and the fit
/// memory plan — deliberately *not* of measured throughput, so two machines
/// with the same memory produce the same tiling and the same arithmetic.
fn adaptive_block_capacity(n_samples: usize, n_variants_hint: usize) -> usize {
    let cap = if n_samples == 0 {
        DEFAULT_BLOCK_WIDTH
    } else {
        // Two decode buffers plus a projection temp, inside the streaming-tile
        // share of the one fit-level pool.
        let budget = streaming_tile_budget_bytes();
        let bytes_per_variant = n_samples.saturating_mul(std::mem::size_of::<f64>());
        let affordable = if bytes_per_variant == 0 {
            DEFAULT_BLOCK_WIDTH
        } else {
            (budget / 3) / bytes_per_variant
        };
        affordable.min(DEFAULT_BLOCK_WIDTH)
    };

    if n_variants_hint > 0 {
        min(cap.max(1), n_variants_hint)
    } else {
        cap.max(1)
    }
}

/// Adapts the streaming covariance operator to the block-Krylov solver.
///
/// `LinOp::apply` cannot return a `Result` (faer's signature has no room for
/// one), so the streaming operator signals failure by panicking with
/// [`OperatorError`] and stashing the real error. This adapter is where that
/// convention is converted back into a `Result` for the solver.
struct StreamingBlockOperator<'a, 'b, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    inner: &'a StandardizedCovarianceOp<'b, S, P>,
    par: Par,
    pass: AtomicUsize,
    max_passes: usize,
}

impl<S, P> BlockOperator for StreamingBlockOperator<'_, '_, S, P>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    type Error = HwePcaError;

    fn dim(&self) -> usize {
        self.inner.n_samples()
    }

    fn apply_block(&self, out: MatMut<'_, f64>, q: MatRef<'_, f64>) -> Result<(), Self::Error> {
        let pass = self.pass.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        if let Some(progress) = self.inner.progress.as_ref() {
            progress.begin_pass(pass, self.max_passes);
        }
        let scratch = self.inner.apply_scratch(q.ncols(), self.par);
        let mut mem = MemBuffer::new(scratch);

        let result = catch_unwind(AssertUnwindSafe(|| {
            let stack = MemStack::new(&mut mem);
            self.inner.apply(out, q, self.par, stack);
        }));

        match result {
            Ok(()) => Ok(()),
            Err(payload) => {
                if payload.downcast_ref::<OperatorError>().is_some() {
                    return Err(self.inner.take_error().unwrap_or_else(|| {
                        HwePcaError::Eigen(
                            "covariance operator aborted with an internal error".into(),
                        )
                    }));
                }
                std::panic::resume_unwind(payload);
            }
        }
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
            diagnostics: None,
        });
    }

    let max_rank = n.saturating_sub(1);
    if max_rank == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
            diagnostics: None,
        });
    }

    let desired = top_k.min(max_rank);
    if desired == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
            diagnostics: None,
        });
    }

    if matches!(mode, CovarianceComputationMode::Dense) {
        return compute_covariance_eigenpairs_dense(operator, par, desired, progress);
    }

    let upper_target = max_rank.min(MAX_PARTIAL_COMPONENTS.max(desired));
    if upper_target == 0 {
        return compute_covariance_eigenpairs_dense(operator, par, desired, progress);
    }

    // Block Krylov: one genome pass advances every requested component at once.
    //
    // The operator already applies `C` to an entire block of sample-space
    // vectors in a single streamed pass over the variants, so growing the
    // Krylov space by one column per pass — which is what a vector-at-a-time
    // Arnoldi does — spends the expensive resource (a genome traversal) to buy
    // the cheap one (a matrix-vector product). See `super::blocklanczos`.
    let requested = desired.min(upper_target).max(1);
    let params = BlockKrylovParams::auto(requested, n, krylov_basis_budget_bytes());
    let block_operator = StreamingBlockOperator {
        inner: operator,
        par,
        pass: AtomicUsize::new(0),
        max_passes: params.max_passes,
    };

    let outcome =
        block_krylov_eigen(&block_operator, requested, params, par).map_err(|err| match err {
            BlockKrylovError::Operator(inner) => inner,
            other => HwePcaError::Eigen(other.to_string()),
        })?;

    // The whole termination record, carried out with the eigenpairs so that the
    // serialized model can answer "did this converge?" long after the warning
    // below has scrolled off somebody's terminal.
    let diagnostics = Some(FitDiagnostics {
        solver: FitSolver::BlockKrylov,
        converged: outcome.converged,
        passes: outcome.passes,
        max_relative_residual: Some(outcome.max_relative_residual),
        subspace_delta: Some(outcome.subspace_delta),
        boundary_gap: outcome.boundary_gap,
        restarts: outcome.restarts,
    });

    if !outcome.converged {
        // Said once here where the numbers are, and enforced by
        // `require_converged` where the fit is assembled: unless the caller
        // opted in, this subspace does not become a model at all.
        eprintln!(
            "warning: PCA eigensolver stopped after {} covariance passes without reaching its \
             tolerance (worst relative Ritz residual {:.3e}, subspace change {:.3e}); the \
             reported components are the best available estimate.",
            outcome.passes, outcome.max_relative_residual, outcome.subspace_delta
        );
    }

    // Ritz values arrive in descending order, so the positive prefix is the
    // usable spectrum.
    let positive = outcome
        .values
        .iter()
        .take_while(|value| **value > EIGENVALUE_EPSILON)
        .count();
    let keep = positive.min(desired);
    if keep == 0 {
        return Ok(Eigenpairs {
            values: Vec::new(),
            vectors: Mat::zeros(n, 0),
            diagnostics,
        });
    }

    let mut values = Vec::with_capacity(keep);
    let mut vectors = Mat::zeros(n, keep);
    for idx in 0..keep {
        values.push(outcome.values[idx]);
        for row in 0..n {
            vectors[(row, idx)] = outcome.vectors[(row, idx)];
        }
    }

    Ok(Eigenpairs {
        values,
        vectors,
        diagnostics,
    })
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
            diagnostics: None,
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
                diagnostics: Some(FitDiagnostics::exact_dense()),
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

        return Ok(Eigenpairs {
            values,
            vectors,
            diagnostics: Some(FitDiagnostics::exact_dense()),
        });
    }

    mirror_lower_to_upper(&mut covariance);

    let gram = covariance.as_ref();
    let upper_target = n.min(MAX_PARTIAL_COMPONENTS.max(top_k));
    let mut target = top_k.min(upper_target).max(1);

    let v0 = krylov_seed_vector(n);
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
                diagnostics: Some(FitDiagnostics::dense_partial(false)),
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

        // Retrying with a wider target is what the loop is for; giving up
        // because the ceiling was reached is a fit that returned fewer
        // components than were asked for, and the model must say so.
        let diagnostics = Some(FitDiagnostics::dense_partial(keep >= top_k));

        if keep >= top_k || target >= upper_target {
            return Ok(Eigenpairs {
                values,
                vectors,
                diagnostics,
            });
        }

        let next_target = (target * 2).min(upper_target);
        if next_target == target {
            return Ok(Eigenpairs {
                values,
                vectors,
                diagnostics,
            });
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
        let max_variants = self.observed_variants.min(packed.n_variants());
        let max_variants = max_variants.min(freqs.len()).min(scales.len());
        let code_table = hard_call_code_table();
        let sample_selection = packed.sample_selection();
        let sample_byte_masks = packed.sample_byte_masks();
        debug_assert!(sample_selection.is_none_or(|selection| selection.len() == n_samples));
        debug_assert_eq!(sample_selection.is_some(), sample_byte_masks.is_some());

        for variant_idx in 0..max_variants {
            let mean = 2.0 * freqs[variant_idx];
            let denom = scales[variant_idx].max(HWE_SCALE_FLOOR);
            let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
            if inv == 0.0 {
                if (variant_idx + 1) % 1_024 == 0 || variant_idx + 1 == max_variants {
                    if let Some(progress) = self.progress.as_ref() {
                        progress.advance(variant_idx + 1);
                    }
                }
                continue;
            }

            let mut z0 = (0.0 - mean) * inv;
            let z1 = (1.0 - mean) * inv;
            let mut z2 = (2.0 - mean) * inv;
            if packed.match_kind(variant_idx) == MatchKind::Swap {
                // The scaler was estimated from the logically oriented stream,
                // while this view points at the physical BED codes. A swapped
                // match maps physical dosage 0 to logical dosage 2 and vice
                // versa; ignoring that here makes the packed operator disagree
                // with every decoded fit path.
                std::mem::swap(&mut z0, &mut z2);
            }

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

            // Wide right-hand sides are taken a chunk at a time so the
            // projection buffer stays on the stack and the scatter phase walks
            // the variant's bytes once per chunk rather than once per column.
            let mut chunk_start = 0usize;
            while chunk_start < ncols {
                let chunk = (ncols - chunk_start).min(PACKED_RHS_CHUNK_COLS);
                let mut proj = [0.0f64; PACKED_RHS_CHUNK_COLS];

                for local in 0..chunk {
                    let col = chunk_start + local;
                    let mut sum0 = 0.0f64;
                    let mut sum1 = 0.0f64;
                    let mut sum2 = 0.0f64;

                    if let Some(masks) = sample_byte_masks {
                        let valid = for_each_packed_masked_code(
                            variant_bytes,
                            masks,
                            n_samples,
                            |idx, code| {
                                let val = rhs[(idx, col)];
                                match code {
                                    0 => sum0 += val,
                                    2 => sum1 += val,
                                    3 => sum2 += val,
                                    _ => {}
                                }
                            },
                        );
                        debug_assert!(valid.is_some());
                    } else {
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
                    }

                    proj[local] = (z0 * sum0 + z1 * sum1 + z2 * sum2) * coeff;
                }

                if let Some(masks) = sample_byte_masks {
                    let valid = for_each_packed_masked_code(
                        variant_bytes,
                        masks,
                        n_samples,
                        |idx, code| {
                            let z = match code {
                                0 => z0,
                                2 => z1,
                                3 => z2,
                                _ => 0.0,
                            };
                            if z != 0.0 {
                                for local in 0..chunk {
                                    out[(idx, chunk_start + local)] += z * proj[local];
                                }
                            }
                        },
                    );
                    debug_assert!(valid.is_some());
                } else {
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
                                for local in 0..chunk {
                                    out[(idx, chunk_start + local)] += z * proj[local];
                                }
                            }
                        }
                        sample_idx += 4;
                    }
                }

                chunk_start += chunk;
            }
            if (variant_idx + 1) % 1_024 == 0 || variant_idx + 1 == max_variants {
                if let Some(progress) = self.progress.as_ref() {
                    progress.advance(variant_idx + 1);
                }
            }
        }

        true
    }
}

/// Deterministic starting vector for the Krylov eigensolvers.
///
/// The all-ones vector is **not** a usable seed here. Every variant column is
/// centered on its own observed mean (`allele_freq = mean(non-missing)/2`, with
/// missing calls landing on zero after centering), so each standardized column
/// sums to zero: `Xᵀ1 = 0`, and therefore `C·1 = X(Xᵀ1)/(n-1) = 0` exactly. The
/// all-ones vector is precisely the sample covariance's null direction, so in
/// exact arithmetic the Krylov sequence `1, C1, C²1, …` is identically zero and
/// carries no information about the leading eigenspace at all. Seeding with it
/// only ever worked because rounding leaves ~ε of noise for the first operator
/// application to amplify into a generic direction — a property of the floating
/// point error, not of the algorithm.
///
/// Instead: a fixed-stream pseudo-random vector with its mean removed, so the
/// known null direction is excluded by construction and the first application
/// does real work. The stream is an inline SplitMix64 rather than `rand`, so a
/// fit stays bit-for-bit reproducible across runs, machines and dependency
/// bumps — reproducibility is why this is not simply seeded from entropy.
fn krylov_seed_vector(n: usize) -> Col<f64> {
    let mut state: u64 = 0x243F_6A88_85A3_08D3; // fixed seed: reproducible fits
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 53 bits -> [0, 1) exactly representable, then shifted to [-1, 1).
        let unit = (z >> 11) as f64 / (1u64 << 53) as f64;
        values.push(unit.mul_add(2.0, -1.0));
    }

    if n > 0 {
        let mean = values.iter().sum::<f64>() / n as f64;
        for value in values.iter_mut() {
            *value -= mean;
        }
    }

    let norm_sq: f64 = values.iter().map(|v| v * v).sum();
    if norm_sq > 0.0 {
        let inv = norm_sq.sqrt().recip();
        for value in values.iter_mut() {
            *value *= inv;
        }
    } else if let Some(first) = values.first_mut() {
        // n == 1: mean removal zeroes the only entry. Any unit vector will do,
        // and this case never reaches a partial solve (max_rank is 0).
        *first = 1.0;
    }

    Col::from_fn(n, |idx| values[idx])
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

fn compute_variant_statistics<S, P>(
    source: &mut S,
    block_capacity: usize,
    par: Par,
    progress: StageProgressHandle<P>,
    n_variants_hint: usize,
) -> Result<(HweScaler, Vec<f64>, usize), HwePcaError>
where
    S: VariantBlockSource,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync,
{
    let n_samples = source.n_samples();

    source
        .reset()
        .map_err(|err| HwePcaError::Source(Box::new(err)))?;

    if let Some(packed) = source.hard_call_packed() {
        let n_variants = packed.n_variants();
        if n_variants == 0 {
            progress.finish();
            return Err(HwePcaError::InvalidInput(
                "VariantBlockSource yielded no variants",
            ));
        }

        let compute = |variant: usize| packed.moments(variant, n_samples);
        let moments: Option<Vec<_>> = if n_variants >= 32 && par.degree() > 1 {
            const BATCH: usize = 1_024;
            let batches = n_variants.div_ceil(BATCH);
            let chunks: Option<Vec<Vec<_>>> = (0..batches)
                .into_par_iter()
                .map(|batch| {
                    let start = batch * BATCH;
                    let end = (start + BATCH).min(n_variants);
                    let values: Option<Vec<_>> = (start..end).map(compute).collect();
                    if values.is_some() {
                        progress.increment(end - start);
                    }
                    values
                })
                .collect();
            chunks.map(|chunks| chunks.into_iter().flatten().collect())
        } else {
            let mut values = Vec::with_capacity(n_variants);
            for variant in 0..n_variants {
                let Some(moment) = compute(variant) else {
                    return Err(HwePcaError::InvalidInput(
                        "packed hard-call source contains an invalid variant selection",
                    ));
                };
                values.push(moment);
                if (variant + 1) % 1_024 == 0 || variant + 1 == n_variants {
                    progress.advance(variant + 1);
                }
            }
            Some(values)
        };
        let moments = moments.ok_or(HwePcaError::InvalidInput(
            "packed hard-call source contains an invalid variant selection",
        ))?;

        let mut frequencies = Vec::with_capacity(n_variants);
        let mut scales = Vec::with_capacity(n_variants);
        let mut standardized_sums_sq = Vec::with_capacity(n_variants);
        for (sum, sum_sq, calls) in moments {
            let (frequency, scale, standardized_sum_sq) =
                finalize_variant_moments(sum, sum_sq, calls);
            frequencies.push(frequency);
            scales.push(scale);
            standardized_sums_sq.push(standardized_sum_sq);
        }

        progress.set_total(n_variants);
        progress.finish();
        return Ok((
            HweScaler::new(frequencies, scales),
            standardized_sums_sq,
            n_variants,
        ));
    }

    let mut stats = VariantStatsCache::new(block_capacity, n_variants_hint);
    let mut block_storage = vec![0.0f64; n_samples * block_capacity];

    let mut processed = 0usize;
    let mut used_source_progress = false;

    loop {
        let filled = source
            .next_block_into(block_capacity, &mut block_storage[..])
            .map_err(|err| HwePcaError::Source(Box::new(err)))?;

        if filled == 0 {
            break;
        }

        let end = processed
            .checked_add(filled)
            .ok_or(HwePcaError::InvalidInput(
                "variant count overflow during statistics computation",
            ))?;

        let block = MatMut::from_column_major_slice_mut(
            &mut block_storage[..n_samples * filled],
            n_samples,
            filled,
        );

        let variant_range = processed..end;
        stats.ensure_statistics(block.as_ref(), variant_range.clone(), par);

        processed = end;

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
    let (scaler, standardized_sums_sq) = stats
        .into_parts()
        .expect("finalized statistics must produce a scaler");
    progress.finish();

    Ok((scaler, standardized_sums_sq, processed))
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
) -> Result<(HweScaler, Vec<f64>, usize, Mat<f64>), HwePcaError>
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
    let stats_progress =
        StageProgressHandle::new(Arc::clone(progress), FitProgressStage::AlleleStatistics);

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
    let (scaler, standardized_sums_sq) = stats
        .into_parts()
        .expect("finalized statistics must produce a scaler");

    // Convert the accumulated Gram matrix (X·Xᵀ) into the sample covariance
    // (X·Xᵀ / (n−1)). Without this, downstream eigenvalues would be inflated by
    // a factor of (n−1) and disagree with the matrix-free and dense-operator
    // paths, which scale via `operator.scale`.
    let covariance_scale = if n_samples > 1 {
        1.0 / ((n_samples - 1) as f64)
    } else {
        1.0
    };
    if covariance_scale != 1.0 {
        for col in 0..n_samples {
            for row in 0..n_samples {
                covariance[(row, col)] *= covariance_scale;
            }
        }
    }

    // Mark Gram matrix stage as complete (it was done during the combined pass)
    progress.on_stage_start(FitProgressStage::GramMatrix, 0);
    progress.on_stage_finish(FitProgressStage::GramMatrix);

    Ok((scaler, standardized_sums_sq, processed, covariance))
}

fn compute_stats_and_ld_weights<S, P>(
    source: &mut S,
    block_capacity: usize,
    config: LdResolvedConfig,
    n_variants_hint: usize,
    progress: &Arc<P>,
    par: Par,
) -> Result<(HweScaler, Vec<f64>, usize, LdWeights), HwePcaError>
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

        let expected = config.range_count();
        if processed + filled > expected {
            return Err(HwePcaError::InvalidInput(LD_RANGE_LIST_MISMATCH));
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
                false,
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

    if processed != config.range_count() {
        return Err(HwePcaError::InvalidInput(LD_RANGE_LIST_MISMATCH));
    }

    assign_ready_weights(
        &mut ring,
        &mut weights,
        &mut next_weight,
        &config,
        &ld_progress,
        true,
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
    let (scaler, standardized_sums_sq) = stats
        .into_parts()
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

    Ok((scaler, standardized_sums_sq, processed, weights))
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

    /// The highest stream index the ring holds, i.e. the most recent push.
    ///
    /// This is what tells the schedule whether a window's right flank has
    /// arrived, so it is read off the buffer rather than tracked beside it —
    /// a counter that drifted from the buffer would put the readiness test back
    /// where it started.
    fn newest_index(&self) -> Option<usize> {
        if self.len == 0 {
            return None;
        }
        Some(self.indices[self.slot_at(self.len - 1)])
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
    /// Elementwise squares of the window's standardized genotypes, so the
    /// pair-restricted sums of squares are one GEMM against the mask.
    values_sq: Mat<f64>,
    gram: Mat<f64>,
    counts: Mat<f64>,
    sums: Mat<f64>,
    squared_sums: Mat<f64>,
    system: Mat<f64>,
    rhs: Mat<f64>,
}

impl LdThreadScratch {
    fn new(n_samples: usize, window_capacity: usize) -> Self {
        Self {
            window: LdWindowScratch {
                values: Mat::zeros(n_samples, window_capacity),
                masks: vec![0u8; n_samples * window_capacity],
            },
            mask_f64: Mat::zeros(n_samples, window_capacity),
            values_sq: Mat::zeros(n_samples, window_capacity),
            gram: Mat::zeros(window_capacity, window_capacity),
            counts: Mat::zeros(window_capacity, window_capacity),
            sums: Mat::zeros(window_capacity, window_capacity),
            squared_sums: Mat::zeros(window_capacity, window_capacity),
            system: Mat::zeros(window_capacity, window_capacity),
            rhs: Mat::zeros(window_capacity, 1),
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
        let freqs = &scaler.allele_frequencies()[variant_range.clone()];
        let scales = &scaler.variant_scales()[variant_range.clone()];
        standardize_block_with_mask_from_stats(
            block.as_mut(),
            presence.as_mut(),
            freqs,
            scales,
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
                false,
                par,
            )?;
        }

        processed += filled;
    }

    if processed != config.range_count() {
        return Err(HwePcaError::InvalidInput(LD_RANGE_LIST_MISMATCH));
    }

    assign_ready_weights(
        &mut ring,
        &mut weights,
        &mut next_weight,
        &config,
        &stage_progress,
        true,
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

/// Solves every window the stream has now completed and slides the ring past
/// the markers none of them can need again.
///
/// `stream_ended` is set only by the flush after the source is exhausted. Until
/// then a centre whose right flank has not arrived simply waits: it is the
/// caller's job to keep pushing, and the ring is bounded by the window width
/// regardless of how long that takes.
fn assign_ready_weights<P: FitProgressObserver>(
    ring: &mut LdRingBuffer,
    weights: &mut [f64],
    next_weight: &mut usize,
    config: &LdResolvedConfig,
    progress: &StageProgressHandle<P>,
    stream_ended: bool,
    par: Par,
) -> Result<(), HwePcaError> {
    let window_capacity = config.window_capacity().max(1);

    while *next_weight < weights.len() {
        if ring.len() == 0 {
            break;
        }

        let jobs = collect_ready_jobs(ring, *next_weight, config, stream_ended)?;
        if jobs.is_empty() {
            break;
        }

        let start_idx = *next_weight;
        let end_idx = start_idx + jobs.len();
        {
            let ring_ref: &LdRingBuffer = &*ring;
            let weight_slice = &mut weights[start_idx..end_idx];
            const PROGRESS_BATCH: usize = 64;
            weight_slice
                .par_chunks_mut(PROGRESS_BATCH)
                .zip_eq(jobs.par_chunks(PROGRESS_BATCH))
                .map_init(
                    || LdThreadScratch::new(ring_ref.n_samples(), window_capacity),
                    |scratch, (slots, jobs)| {
                        for (slot, job) in slots.iter_mut().zip(jobs) {
                            *slot = compute_ld_weight(
                                ring_ref,
                                job.start_offset,
                                job.window_len,
                                job.center,
                                config.ridge,
                                scratch,
                                par,
                            );
                        }
                        progress.increment(slots.len());
                    },
                )
                .for_each(|_| {});
        }

        *next_weight = end_idx;
        if let Some(last_keep) = jobs.last().map(|job| job.keep_from) {
            ring.truncate_front(last_keep);
        }
    }

    Ok(())
}

/// LD weight for the window's centre variant, on the **complete-pairs**
/// convention: every statistic feeding a pair `(i, j)` is restricted to the
/// samples where `i` and `j` are *both* observed.
///
/// Missing standardized genotypes are exactly zero, which is what makes this
/// cheap. Writing `V` for the window's standardized values (zero where missing)
/// and `M` for its 0/1 presence mask, every quantity the correlation needs is a
/// GEMM whose sum silently drops the samples the pair does not share:
///
/// ```text
/// gram[(i, j)]         = (VᵀV)[i, j]      = Σ_{s ∈ Aᵢ∩Aⱼ} vᵢ(s)·vⱼ(s)
/// counts[(i, j)]       = (MᵀM)[i, j]      = |Aᵢ ∩ Aⱼ|
/// sums[(i, j)]         = (VᵀM)[i, j]      = Σ_{s ∈ Aᵢ∩Aⱼ} vᵢ(s)
/// squared_sums[(i, j)] = ((V∘V)ᵀM)[i, j]  = Σ_{s ∈ Aᵢ∩Aⱼ} vᵢ(s)²
/// ```
///
/// The last two used to be computed once per variant over *all* of that
/// variant's own observations and then reused unchanged for every partner. For
/// markers missing in the same samples that is the same number; for markers
/// missing in *different* samples it is not, and the result was a Pearson
/// correlation whose cross-product and count came from the intersection while
/// its centring and scaling came from the union — neither the complete-pairs
/// correlation nor the mean-imputed one, and biased toward zero in proportion
/// to how much the two markers' missingness patterns disagree.
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
    let mut values_sq =
        scratch
            .values_sq
            .as_mut()
            .submatrix_mut(0, 0, ring.n_samples(), window_len);
    for col in 0..window_len {
        let src = &masks[col * ring.n_samples()..(col + 1) * ring.n_samples()];
        let dst = mask_mat.rb_mut().col_mut(col);
        for (dst, &src_val) in dst.iter_mut().zip(src.iter()) {
            *dst = src_val as f64;
        }

        let values_slice = values
            .col(col)
            .try_as_col_major()
            .expect("LD window column must be contiguous")
            .as_slice();
        let dst = values_sq.rb_mut().col_mut(col);
        for (dst, &src_val) in dst.iter_mut().zip(values_slice.iter()) {
            *dst = src_val * src_val;
        }
    }
    let mask_view = mask_mat.as_ref();
    let values_sq_view = values_sq.as_ref();

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

    let mut sums = scratch
        .sums
        .as_mut()
        .submatrix_mut(0, 0, window_len, window_len);
    sums.fill(0.0);
    matmul(
        sums.as_mut(),
        Accum::Replace,
        values.transpose(),
        mask_view,
        1.0,
        par,
    );

    let mut squared_sums = scratch
        .squared_sums
        .as_mut()
        .submatrix_mut(0, 0, window_len, window_len);
    squared_sums.fill(0.0);
    matmul(
        squared_sums.as_mut(),
        Accum::Replace,
        values_sq_view.transpose(),
        mask_view,
        1.0,
        par,
    );

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

/// Collects the windows whose markers have all been streamed, in stream order,
/// starting at `start_index`.
///
/// Readiness is a statement about the *stream*, not about the buffer. The
/// schedule used to size each window against `ring.len()` and solve every
/// centre the instant it was pushed, so a window was whatever had arrived so
/// far: with a 51-site request the ring settled at three markers and every
/// weight came out of `[i−2, i−1, i]` with the centre pinned to the right edge.
/// The requested window was never once realised — not at the head of the
/// chromosome, where a truncated window is legitimate, but anywhere.
///
/// So a centre waits here until its last marker exists. `stream_ended` says no
/// further variant will ever arrive, and it is the only reason to solve a
/// window that is still short on the right; a chromosome boundary is already
/// baked into the range itself.
fn collect_ready_jobs(
    ring: &LdRingBuffer,
    start_index: usize,
    config: &LdResolvedConfig,
    stream_ended: bool,
) -> Result<Vec<LdWeightJob>, HwePcaError> {
    let mut jobs = Vec::new();
    let Some(newest) = ring.newest_index() else {
        return Ok(jobs);
    };

    let mut next = start_index;
    loop {
        // A centre that has not been pushed yet is where this run of the
        // schedule stops; it is also the bound that keeps `jobs` from running
        // past the weights vector when `stream_ended` extends windows to the
        // right of everything that exists.
        let Some(center_slot) = ring.position_of(next) else {
            break;
        };

        let Some(range) = config.window_range(next, newest, stream_ended)? else {
            break;
        };

        // Residency follows from the capacity argument on
        // `LdResolvedConfig::window_capacity`, so reaching either branch below
        // means that argument is broken. Checking it costs a walk of at most
        // `window_capacity` slots and converts a silent window over the wrong
        // markers into a stopped fit.
        let (Some(start_slot), Some(end_slot)) = (
            ring.position_of(range.start),
            ring.position_of(range.end - 1),
        ) else {
            return Err(HwePcaError::InvalidInput(
                "LD ring buffer no longer holds a marker its window needs",
            ));
        };

        let span = range.end - 1 - range.start;
        if end_slot.checked_sub(start_slot) != Some(span)
            || center_slot < start_slot
            || center_slot > end_slot
        {
            return Err(HwePcaError::InvalidInput(
                "LD ring buffer does not hold this window as one contiguous run",
            ));
        }

        jobs.push(LdWeightJob {
            start_offset: start_slot,
            window_len: span + 1,
            center: center_slot - start_slot,
            keep_from: range.start,
        });
        next += 1;
    }

    Ok(jobs)
}

/// Solves the ridge-regularized LD system for one window's centre variant.
///
/// Every argument is on the complete-pairs convention established by
/// [`compute_ld_weight`], and the recentring below is only valid on it:
/// `counts[(i, j)]` is the number of samples where `i` and `j` are both
/// observed, and `sums`/`squared_sums` are *asymmetric* — row `i`, column `j`
/// holds variant `i`'s statistic over exactly that shared set. Passing
/// whole-variant sums here instead would recentre an intersection cross-product
/// by union-scale moments.
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
                    // Row-major reading of the asymmetric statistics: `(i, j)`
                    // is variant `i` over the shared samples, `(j, i)` is
                    // variant `j` over the same shared samples.
                    let sum_i = sums[(i, j)];
                    let sum_j = sums[(j, i)];
                    let cov = gram[(i, j)] - (sum_i * sum_j) / count;
                    let var_i = squared_sums[(i, j)] - (sum_i * sum_i) / count;
                    let var_j = squared_sums[(j, i)] - (sum_j * sum_j) / count;

                    if !cov.is_finite() || !var_i.is_finite() || !var_j.is_finite() {
                        0.0
                    } else if var_i <= 0.0 || var_j <= 0.0 {
                        0.0
                    } else {
                        let corr = (cov / (var_i * var_j).sqrt()).clamp(-1.0, 1.0);
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
            // Indexed explicitly rather than counting inside a `zip!` closure.
            // Pairing weights[i] with the i-th entry by incrementing a counter
            // per visit is only correct if the traversal happens to run in
            // index order; nothing in the API promises that, and if it ever
            // vectorized or reordered, every LD weight would land on the wrong
            // variant — quietly, and with entirely plausible-looking output.
            let contiguous = column_ref
                .try_as_col_major()
                .expect("loading columns are contiguous");
            for (idx, value) in contiguous.as_slice().iter().enumerate() {
                let weight = if idx < n_weights { weights[idx] } else { 1.0 };
                let weighted = weight * *value;
                let square = weighted * weighted;
                let y = square - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
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

/// Diagonalizes the covariance restricted to the fitted subspace, returning its
/// eigenvalues and the rotation that carries the solver's basis onto them.
///
/// `restricted_gram` is `BᵀB` for `B = Xᵀ·U`, which is `(n−1)·Uᵀ·C·U`. Its
/// eigenvectors are the Ritz vectors of `C` within the subspace `U` spans, and
/// its eigenvalues are `(n−1)·λ`. Scaling a matrix by a positive constant moves
/// no eigenvector, so the Gram is decomposed exactly as accumulated and only the
/// *reported* eigenvalues are divided down: one multiply per component instead
/// of a `k×k` rescale, and no rounding inserted ahead of the decomposition.
///
/// Ordering and filtering follow the eigensolvers upstream exactly — descending,
/// keeping only components whose covariance eigenvalue clears
/// `EIGENVALUE_EPSILON`. A returned subspace really can be rank-deficient: more
/// components can be requested than the data has, and a Ritz value that was
/// positive against one pass's covariance can fail against this one. Such a
/// component has no variance behind it and therefore no singular value to divide
/// its loadings by, so it is dropped here rather than carried as a column of
/// noise scaled up by an arbitrarily small σ.
fn rayleigh_ritz_rotation(
    restricted_gram: MatRef<'_, f64>,
    n_samples: usize,
) -> Result<(Vec<f64>, Mat<f64>), HwePcaError> {
    let width = restricted_gram.ncols();
    let eig = restricted_gram
        .self_adjoint_eigen(Side::Lower)
        .map_err(|err| {
            HwePcaError::Eigen(format!("Rayleigh-Ritz eigendecomposition failed: {err:?}"))
        })?;
    let diag = eig.S();
    let basis = eig.U();

    // The fit rejects cohorts smaller than two samples long before this runs;
    // the floor is here only so the reciprocal is always defined.
    let inverse_scale = (n_samples.saturating_sub(1).max(1) as f64).recip();
    let mut ordering: Vec<(usize, f64)> = (0..width)
        .map(|idx| (idx, diag[idx] * inverse_scale))
        .collect();
    let keep = ordering
        .iter()
        .filter(|entry| entry.1 > EIGENVALUE_EPSILON)
        .count();
    let mid = select_top_k_desc(&mut ordering, keep);

    let mut values = Vec::with_capacity(mid);
    let mut rotation = Mat::zeros(width, mid);
    for (out_idx, (src_idx, value)) in ordering[..mid].iter().copied().enumerate() {
        values.push(value);
        for row in 0..width {
            rotation[(row, out_idx)] = basis[(row, src_idx)];
        }
    }

    Ok((values, rotation))
}

/// Applies `matrix ← matrix·rotation` in place, one row block at a time.
///
/// The two matrices a fit rotates are the two largest things it holds: the `n×k`
/// sample basis and the `p×k` variant cross-products. Multiplying either into a
/// freshly allocated result would double the larger of them at the one moment
/// both are resident, which at biobank `p` is gigabytes. A row block, though,
/// reads only the rows it is about to overwrite, so a `row_chunk×k` scratch copy
/// is the entire extra cost.
///
/// `rotation` never has more columns than `matrix` does, so the product is
/// written back over a prefix of the columns it was read from; anything past
/// `rotation.ncols()` is stale afterwards and the narrowing copy at the end
/// discards it. `row_chunk` is a blocking granularity and nothing more — any
/// positive value gives the same answer up to summation order inside the GEMM.
fn rotate_columns(
    mut matrix: Mat<f64>,
    rotation: MatRef<'_, f64>,
    row_chunk: usize,
    par: Par,
) -> Mat<f64> {
    let rows = matrix.nrows();
    let width = matrix.ncols();
    let kept = rotation.ncols();
    debug_assert_eq!(rotation.nrows(), width);
    debug_assert!(kept <= width);

    if rows > 0 && width > 0 && kept > 0 {
        let chunk = row_chunk.clamp(1, rows);
        let mut scratch = Mat::zeros(chunk, width);
        let mut start = 0usize;
        while start < rows {
            let take = chunk.min(rows - start);
            scratch
                .as_mut()
                .submatrix_mut(0, 0, take, width)
                .copy_from(matrix.as_ref().submatrix(start, 0, take, width));
            matmul(
                matrix.as_mut().submatrix_mut(start, 0, take, kept),
                Accum::Replace,
                scratch.as_ref().submatrix(0, 0, take, width),
                rotation,
                1.0,
                par,
            );
            start += take;
        }
    }

    if kept == width {
        return matrix;
    }

    let rotated = matrix.as_ref();
    Mat::from_fn(rows, kept, |row, col| rotated[(row, col)])
}

fn weighted_standardized_frobenius_sq(
    standardized_sums_sq: &[f64],
    ld_weights: Option<&[f64]>,
) -> f64 {
    let mut total = 0.0f64;
    let mut compensation = 0.0f64;
    for (variant, &sum_sq) in standardized_sums_sq.iter().enumerate() {
        let weight = ld_weights
            .and_then(|weights| weights.get(variant))
            .copied()
            .unwrap_or(1.0);
        let term = sum_sq * weight * weight;
        let adjusted = term - compensation;
        let next = total + adjusted;
        compensation = (next - total) - adjusted;
        total = next;
    }
    total
}

/// Streams the genotypes once to form `B = Xᵀ·U` and, from the same blocks,
/// `BᵀB`.
///
/// `B` is not yet the variant loadings: dividing its columns by σ is deferred
/// until after the Rayleigh-Ritz rotation, because the σ that makes the division
/// exact — the one for which `‖B_i‖ = σ_i` — is the one the rotation produces,
/// not the one the eigensolver arrived with. Scaling here and correcting later
/// is what forced the model to carry two different sets of singular values.
///
/// `BᵀB` accumulates block by block out of the chunk already computed for the
/// loadings, so the restricted covariance costs one `k`-wide GEMM per block and
/// not one additional read of the genome.
fn compute_loading_cross_products<S, P>(
    source: &mut S,
    scaler: &HweScaler,
    expected_variants: usize,
    block_capacity: usize,
    sample_basis: MatRef<'_, f64>,
    ld_weights: Option<&[f64]>,
    progress: &Arc<P>,
    par: Par,
) -> Result<(Mat<f64>, Mat<f64>), HwePcaError>
where
    S: VariantBlockSource + Send,
    S::Error: Error + Send + Sync + 'static,
    P: FitProgressObserver + Send + Sync + 'static,
{
    let n_samples = source.n_samples();
    let n_components = sample_basis.ncols();
    let loadings = Mat::zeros(expected_variants, n_components);
    let mut chunk_storage = vec![0.0f64; block_capacity * n_components];

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
            standardized: bool,
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

    let block_capacity = block_capacity;
    let block_len = block_len;
    let expected_variants = expected_variants;
    let allele_frequencies = scaler.allele_frequencies();
    let variant_scales = scaler.variant_scales();

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
                let (filled, standardized) = match source.next_standardized_block_into(
                    block_capacity,
                    buffer_slice,
                    &allele_frequencies[start..],
                    &variant_scales[start..],
                    ld_weights.map(|weights| &weights[start..]),
                ) {
                    Ok(Some(filled)) => (filled, true),
                    Ok(None) => match source.next_block_into(block_capacity, buffer_slice) {
                        Ok(filled) => (filled, false),
                        Err(err) => {
                            let _ = filled_sender
                                .send(PrefetchMessage::Error(HwePcaError::Source(Box::new(err))));
                            break;
                        }
                    },
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

                if filled_sender
                    .send(PrefetchMessage::Data {
                        id,
                        filled,
                        start,
                        standardized,
                    })
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
        // Accumulates BᵀB = (n−1)·Uᵀ·C·U, the covariance restricted to the
        // subspace the eigensolver returned.
        let mut restricted_gram = Mat::zeros(n_components, n_components);
        while let Ok(message) = filled_rx.recv() {
            match message {
                PrefetchMessage::Data {
                    id,
                    filled,
                    start,
                    standardized,
                } => {
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
                    if !standardized {
                        scaler.standardize_block(block.as_mut(), start..start + filled, par);
                        if let Some(weights) = ld_weights {
                            apply_ld_weights(block.as_mut(), start..start + filled, weights);
                        }
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

                    // Fold this block's contribution to BᵀB in while the chunk is
                    // still in cache, before it is copied out to `loadings`.
                    matmul(
                        restricted_gram.as_mut(),
                        Accum::Add,
                        chunk.as_ref().transpose(),
                        chunk.as_ref(),
                        1.0,
                        par,
                    );

                    loadings
                        .submatrix_mut(start, 0, filled, n_components)
                        .copy_from(chunk.as_ref());

                    processed = start + filled;
                    progress.on_stage_advance(FitProgressStage::Loadings, processed);

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

        Ok((loadings, restricted_gram))
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
        state.serialize_field("total_variance", &self.total_variance)?;
        state.serialize_field("singular_values", &self.singular_values)?;
        state.serialize_field("loadings", &MatrixData::from_mat(self.loadings.as_ref()))?;
        state.serialize_field(
            "component_weighted_norms_sq",
            &self.component_weighted_norms_sq,
        )?;
        state.serialize_field("variant_keys", &self.variant_keys)?;
        state.serialize_field("ld", &self.ld)?;
        state.serialize_field("genome_build", &self.genome_build)?;
        state.serialize_field("fit_diagnostics", &self.diagnostics)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HwePcaModel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct ModelData {
            n_samples: usize,
            n_variants: usize,
            scaler: HweScaler,
            eigenvalues: Vec<f64>,
            #[serde(default)]
            total_variance: f64,
            singular_values: Vec<f64>,
            loadings: MatrixData,
            #[serde(default)]
            component_weighted_norms_sq: Vec<f64>,
            #[serde(default)]
            variant_keys: Option<Vec<VariantKey>>,
            #[serde(default)]
            ld: Option<LdWeights>,
            #[serde(default)]
            genome_build: Option<String>,
            /// Absent from every `hwe.json` written before fits recorded their
            /// convergence; `default` is what keeps those files loading.
            #[serde(default)]
            fit_diagnostics: Option<FitDiagnostics>,
        }

        let raw = ModelData::deserialize(deserializer)?;
        let singular_values_len = raw.singular_values.len();
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
        let projection_cache = Arc::new(build_projection_model_cache(
            &raw.scaler,
            loadings.as_ref(),
            ld.as_ref(),
        ));

        Ok(HwePcaModel {
            n_samples: raw.n_samples,
            n_variants: raw.n_variants,
            scaler: raw.scaler,
            eigenvalues: raw.eigenvalues,
            total_variance: raw.total_variance,
            singular_values: raw.singular_values,
            // Training coordinates live only in `hwe_scores.bin`; duplicating
            // n_samples × components matrices in decimal JSON dominates fit
            // output at biobank scale and loses the row IDs the binary carries.
            sample_basis: Mat::zeros(0, singular_values_len),
            sample_scores: Mat::zeros(0, singular_values_len),
            loadings,
            component_weighted_norms_sq,
            variant_keys: raw.variant_keys,
            ld,
            genome_build: raw.genome_build,
            diagnostics: raw.fit_diagnostics,
            projection_cache,
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
            let freqs = &scaler.allele_frequencies()[0..observed_variants];
            let scales = &scaler.variant_scales()[0..observed_variants];
            standardize_block_with_mask_from_stats(
                block.as_mut(),
                mask_mat.as_mut(),
                freqs,
                scales,
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
                false,
                Par::Seq,
            )?;
        }

        assign_ready_weights(
            &mut ring,
            &mut weights,
            &mut next_weight,
            config,
            &progress,
            true,
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

    fn single_chromosome_keys(n_variants: usize) -> Vec<VariantKey> {
        (0..n_variants)
            .map(|idx| VariantKey::new("1", 1_000 + idx as u64 * 100))
            .collect()
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
            window: LdResolvedWindow::Sites {
                size: 3,
                ranges: compute_ld_site_ranges(&single_chromosome_keys(observed_variants), 3),
            },
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
    fn site_ld_refuses_a_stream_without_chromosome_keys() {
        let options = FitOptions {
            ld: Some(LdConfig {
                window: Some(LdWindow::Sites(3)),
                ridge: None,
                variant_keys: None,
            }),
            ..FitOptions::default()
        };

        let err = options
            .resolved_ld(4)
            .expect_err("chromosome-blind site windows must not be constructed");
        assert!(err.to_string().contains("chromosome keys"));
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
            window: LdResolvedWindow::Sites {
                size: 17,
                ranges: compute_ld_site_ranges(&single_chromosome_keys(observed_variants), 17),
            },
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

    /// Standardizes a raw genotype matrix the way the fit does, keeping the
    /// presence mask alongside: missing calls standardize to exactly zero, which
    /// is what lets a Gram carry pair-restricted cross-products.
    fn standardize_with_mask(
        data: &[f64],
        n_samples: usize,
        scaler: &HweScaler,
    ) -> (Vec<Vec<f64>>, Vec<Vec<bool>>) {
        let n_variants = data.len() / n_samples;
        let mut values = Vec::with_capacity(n_variants);
        let mut observed = Vec::with_capacity(n_variants);
        for variant in 0..n_variants {
            let mean = 2.0 * scaler.allele_frequencies()[variant];
            let denom = scaler.variant_scales()[variant].max(HWE_SCALE_FLOOR);
            let inv = if denom > 0.0 { denom.recip() } else { 0.0 };
            let mut column = Vec::with_capacity(n_samples);
            let mut mask = Vec::with_capacity(n_samples);
            for sample in 0..n_samples {
                let raw = data[variant * n_samples + sample];
                if raw.is_finite() {
                    column.push((raw - mean) * inv);
                    mask.push(true);
                } else {
                    column.push(0.0);
                    mask.push(false);
                }
            }
            values.push(column);
            observed.push(mask);
        }
        (values, observed)
    }

    /// Independent complete-pairs LD weight for one window.
    ///
    /// Deliberately shares no code with `compute_ld_weight`: it gathers each
    /// pair's jointly observed samples by hand and computes that pair's Pearson
    /// correlation from those samples alone. `compute_reference_ld_weights`
    /// above cannot serve here — it drives the very ring buffer and GEMMs under
    /// test, so it agrees with whatever convention the production statistics
    /// happen to use, including an inconsistent one.
    fn reference_complete_pairs_weight(
        standardized: &[Vec<f64>],
        observed: &[Vec<bool>],
        window: &[usize],
        center: usize,
        ridge: f64,
    ) -> f64 {
        let size = window.len();
        let mut system = Mat::<f64>::zeros(size, size);
        for i in 0..size {
            system[(i, i)] = 1.0 + ridge;
            for j in 0..i {
                let values_i = &standardized[window[i]];
                let values_j = &standardized[window[j]];
                let mask_i = &observed[window[i]];
                let mask_j = &observed[window[j]];
                let shared: Vec<usize> = (0..values_i.len())
                    .filter(|&sample| mask_i[sample] && mask_j[sample])
                    .collect();
                let count = shared.len() as f64;

                let value = if count > 2.0 {
                    let sum_i: f64 = shared.iter().map(|&s| values_i[s]).sum();
                    let sum_j: f64 = shared.iter().map(|&s| values_j[s]).sum();
                    let cross: f64 = shared.iter().map(|&s| values_i[s] * values_j[s]).sum();
                    let square_i: f64 = shared.iter().map(|&s| values_i[s] * values_i[s]).sum();
                    let square_j: f64 = shared.iter().map(|&s| values_j[s] * values_j[s]).sum();

                    let cov = cross - sum_i * sum_j / count;
                    let var_i = square_i - sum_i * sum_i / count;
                    let var_j = square_j - sum_j * sum_j / count;

                    if var_i <= 0.0 || var_j <= 0.0 {
                        0.0
                    } else {
                        let corr = (cov / (var_i * var_j).sqrt()).clamp(-1.0, 1.0);
                        let unbiased = ((count - 1.0) * corr * corr - 1.0) / (count - 2.0);
                        unbiased.clamp(0.0, 1.0)
                    }
                } else {
                    0.0
                };

                system[(i, j)] = value;
                system[(j, i)] = value;
            }
        }

        let rhs = Mat::<f64>::from_fn(size, 1, |_, _| 1.0);
        let factor = FaerLlt::new(system.as_ref(), Side::Lower)
            .expect("reference LD system must be positive definite");
        let solution = factor.solve(rhs.as_ref());
        let weight_sq = solution[(center, 0)];
        if !weight_sq.is_finite() || weight_sq <= 0.0 {
            1.0
        } else {
            weight_sq.sqrt().max(MIN_LD_WEIGHT)
        }
    }

    /// Runs `compute_ld_weights` over `data` and checks every weight against the
    /// complete-pairs reference, given the windows the streaming schedule
    /// produces.
    ///
    /// `windows[c]` is the set of variant indices that variant `c`'s weight is
    /// solved from. A window is clipped where the stream runs out — at the
    /// start of the genome and, for the callers below, at its end — so the
    /// leading variants are solved from short windows. That is the production
    /// schedule, and pinning it here keeps this test about the *statistics*
    /// rather than about the window geometry, which
    /// `sites_window_reaches_its_full_width_mid_genome` covers on its own.
    fn assert_ld_weights_match_complete_pairs(
        data: &[f64],
        n_samples: usize,
        window_size: usize,
        windows: &[&[usize]],
    ) {
        let n_variants = data.len() / n_samples;
        assert_eq!(windows.len(), n_variants);

        let mut source = DenseBlockSource::new(data, n_samples, n_variants).expect("dense source");
        let stats_progress = StageProgressHandle::new(
            Arc::new(NoopFitProgress),
            FitProgressStage::AlleleStatistics,
        );
        let (scaler, _, observed_variants) = compute_variant_statistics(
            &mut source,
            n_variants,
            Par::Seq,
            stats_progress,
            n_variants,
        )
        .expect("variant statistics");
        assert_eq!(observed_variants, n_variants);

        let config = LdResolvedConfig {
            window: LdResolvedWindow::Sites {
                size: window_size,
                ranges: compute_ld_site_ranges(&single_chromosome_keys(n_variants), window_size),
            },
            ridge: DEFAULT_LD_RIDGE,
        };
        let progress = Arc::new(NoopFitProgress);
        let weights = compute_ld_weights(
            &mut source,
            &scaler,
            n_variants,
            n_variants,
            config,
            n_variants,
            &progress,
            Par::Seq,
        )
        .expect("ld weights")
        .weights;

        let (standardized, observed) = standardize_with_mask(data, n_samples, &scaler);

        for (variant, &window) in windows.iter().enumerate() {
            let center = window
                .iter()
                .position(|&index| index == variant)
                .expect("a variant's window must contain that variant");
            let expected = reference_complete_pairs_weight(
                &standardized,
                &observed,
                window,
                center,
                DEFAULT_LD_RIDGE,
            );
            assert!(
                (weights[variant] - expected).abs() < 1.0e-9,
                "variant {variant}: weight {} but complete-pairs reference is {expected}",
                weights[variant]
            );
        }

        // Agreement is only worth something if some pair actually correlated:
        // a window whose off-diagonals are all zero solves to this baseline
        // whatever convention produced them, and would pass vacuously.
        let isolated = (1.0f64 / (1.0 + DEFAULT_LD_RIDGE)).sqrt();
        assert!(
            weights
                .iter()
                .any(|weight| (weight - isolated).abs() > 1.0e-6),
            "no pair in this dataset correlated, so the comparison proves nothing"
        );
    }

    #[test]
    fn ld_weights_use_only_jointly_observed_samples() {
        // Every marker is missing somewhere different, which is the whole
        // point: under a shared missingness pattern, per-variant sums and
        // pair-restricted sums coincide and nothing here could fail.
        //
        // Variants 0 and 1 carry the same genotypes on the six samples they
        // share, so their complete-pairs correlation is exactly ±1 whatever
        // their disjoint samples do. Statistics taken over each variant's own
        // observations cannot see that: they scale an intersection
        // cross-product by union-wide moments and report something smaller.
        const N_SAMPLES: usize = 10;
        let nan = f64::NAN;
        let data: Vec<f64> = vec![
            nan, nan, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, // variant 0
            2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, nan, nan, // variant 1
            0.0, 1.0, 2.0, 2.0, nan, 0.0, 1.0, 1.0, 2.0, 0.0, // variant 2
            1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, // variant 3
        ];
        assert_eq!(data.len(), N_SAMPLES * 4);

        // A two-site window keeps every solved system 2×2 — trivially positive
        // definite for any correlation, so the reference never has to guess at
        // a factorization failure. The ring holds two variants and slides.
        let windows: [&[usize]; 4] = [&[0], &[0, 1], &[1, 2], &[2, 3]];
        assert_ld_weights_match_complete_pairs(&data, N_SAMPLES, 2, &windows);
    }

    #[test]
    fn ld_weights_survive_barely_overlapping_markers() {
        // Variants 0 and 1 share only four samples out of twelve, and agree
        // exactly on those four. Statistics taken over each variant's own
        // observations describe two mostly disjoint sets of people and say
        // nothing about the four who are in both; the correlation between the
        // markers has to come from those four and nowhere else.
        const N_SAMPLES: usize = 12;
        let nan = f64::NAN;
        let data: Vec<f64> = vec![
            0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, nan, nan, nan, nan, // variant 0
            nan, nan, nan, nan, 2.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0, 2.0, // variant 1
            1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0, // variant 2
        ];
        assert_eq!(data.len(), N_SAMPLES * 3);

        let windows: [&[usize]; 3] = [&[0], &[0, 1], &[1, 2]];
        assert_ld_weights_match_complete_pairs(&data, N_SAMPLES, 2, &windows);
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

    /// Replays the streaming schedule and records, for each variant, the stream
    /// indices its window actually covered.
    ///
    /// Drives the production ring buffer and the production
    /// [`collect_ready_jobs`], so it measures the schedule rather than
    /// restating it; only the GEMM solve is left out, because window *geometry*
    /// is what these tests are about. Genotypes are therefore never written —
    /// the schedule reads indices and nothing else.
    fn replay_ld_windows(config: &LdResolvedConfig, n_variants: usize) -> Vec<Vec<usize>> {
        let capacity = config.window_capacity().max(1);
        let mut ring = LdRingBuffer::new(1, capacity);
        let mut windows: Vec<Vec<usize>> = vec![Vec::new(); n_variants];
        let mut next = 0usize;

        // One extra turn past the last variant is the flush the fit performs
        // once its source is exhausted.
        for idx in 0..=n_variants {
            let stream_ended = idx == n_variants;
            if !stream_ended {
                let slot = ring.push_slot();
                ring.indices_mut()[slot] = idx;
            }

            loop {
                let jobs =
                    collect_ready_jobs(&ring, next, config, stream_ended).expect("ready jobs");
                if jobs.is_empty() {
                    break;
                }

                for job in &jobs {
                    assert_eq!(
                        job.keep_from + job.center,
                        next,
                        "the job solved for variant {next} centres on a different variant"
                    );
                    windows[next] = (job.keep_from..job.keep_from + job.window_len).collect();
                    next += 1;
                }

                if let Some(last) = jobs.last() {
                    ring.truncate_front(last.keep_from);
                }
            }
        }

        assert_eq!(
            next, n_variants,
            "the schedule weighted {next} of {n_variants} variants"
        );
        windows
    }

    #[test]
    fn sites_window_reaches_its_full_width_mid_genome() {
        // The scheduler used to solve each centre the moment it was pushed and
        // size the window against whatever the ring held, which left a request
        // for many markers permanently satisfied by three — `[i-2, i-1, i]`,
        // centre at the right edge. Away from the ends of the stream every
        // window must now be exactly as wide as it was asked to be, with the
        // centre in the middle.
        const N_VARIANTS: usize = 40;
        const WINDOW: usize = 11;
        let half = WINDOW / 2;

        let config = LdResolvedConfig {
            window: LdResolvedWindow::Sites {
                size: WINDOW,
                ranges: compute_ld_site_ranges(&single_chromosome_keys(N_VARIANTS), WINDOW),
            },
            ridge: DEFAULT_LD_RIDGE,
        };
        let windows = replay_ld_windows(&config, N_VARIANTS);

        for centre in 0..N_VARIANTS {
            let expected: Vec<usize> =
                (centre.saturating_sub(half)..(centre + half + 1).min(N_VARIANTS)).collect();
            assert_eq!(windows[centre], expected, "window for variant {centre}");
        }

        for centre in half..N_VARIANTS - half {
            assert_eq!(
                windows[centre].len(),
                WINDOW,
                "interior variant {centre} was solved from a truncated window"
            );
            assert_eq!(windows[centre][half], centre);
        }
    }

    #[test]
    fn sites_window_never_spans_two_chromosomes() {
        // Twelve markers on chr1 followed by twelve on chr2, adjacent in the
        // stream and nowhere near each other in the genome. Cut from stream
        // order alone, the windows around index 11 and 12 would mix them and
        // report "LD" between chromosomes.
        const PER_CHROMOSOME: usize = 12;
        const WINDOW: usize = 9;
        let keys: Vec<VariantKey> = (0..PER_CHROMOSOME)
            .map(|idx| VariantKey::new("1", 1_000 + idx as u64 * 100))
            .chain((0..PER_CHROMOSOME).map(|idx| VariantKey::new("2", 1_000 + idx as u64 * 100)))
            .collect();

        let config = LdResolvedConfig {
            window: LdResolvedWindow::Sites {
                size: WINDOW,
                ranges: compute_ld_site_ranges(&keys, WINDOW),
            },
            ridge: DEFAULT_LD_RIDGE,
        };
        let windows = replay_ld_windows(&config, keys.len());

        for (centre, window) in windows.iter().enumerate() {
            assert!(window.contains(&centre), "variant {centre} left its window");
            for &member in window {
                assert_eq!(
                    keys[member].chromosome, keys[centre].chromosome,
                    "variant {centre} was weighted against variant {member} on another chromosome"
                );
            }
        }

        // Clipping is to the chromosome, not to the boundary's aftermath: a
        // centre with room on both sides of it still gets the full window.
        let interior = PER_CHROMOSOME + WINDOW / 2;
        assert_eq!(windows[interior].len(), WINDOW);
        assert_eq!(windows[interior][0], PER_CHROMOSOME);

        // The boundary itself: the last marker of chr1 keeps its left flank and
        // loses its right, and the first of chr2 the other way round.
        assert_eq!(
            windows[PER_CHROMOSOME - 1].last(),
            Some(&(PER_CHROMOSOME - 1))
        );
        assert_eq!(windows[PER_CHROMOSOME][0], PER_CHROMOSOME);
    }

    #[test]
    fn bp_window_never_spans_two_chromosomes() {
        // Positions deliberately collide across chromosomes: a range builder
        // comparing positions without chromosomes would merge all four markers
        // into one window.
        let keys = vec![
            VariantKey::new("1", 100),
            VariantKey::new("1", 140),
            VariantKey::new("2", 100),
            VariantKey::new("2", 140),
        ];

        let (ranges, capacity) = compute_ld_bp_ranges(&keys, 200).expect("ranges");

        assert_eq!(capacity, 2);
        assert_eq!((ranges[0].start, ranges[0].end), (0, 2));
        assert_eq!((ranges[1].start, ranges[1].end), (0, 2));
        assert_eq!((ranges[2].start, ranges[2].end), (2, 4));
        assert_eq!((ranges[3].start, ranges[3].end), (2, 4));
    }

    #[test]
    fn bp_window_rejects_decreasing_positions_within_a_chromosome() {
        let keys = vec![
            VariantKey::new("1", 100),
            VariantKey::new("1", 90),
            VariantKey::new("2", 10),
        ];
        let err = compute_ld_bp_ranges(&keys, 100)
            .expect_err("a distance window over decreasing positions is undefined");
        assert!(err.to_string().contains("nondecreasing positions"));
    }

    #[test]
    fn ld_ranges_must_describe_the_streamed_variant_list() {
        // Stands in for `--maf` on a PLINK fileset: the windows are cut from a
        // key list the filter has not been applied to, the stream delivers the
        // retained variants renumbered from zero, and every index in the
        // schedule then names a different marker than the one it was cut for —
        // with no index out of bounds and no weight out of range to show for
        // it. Both directions of the disagreement have to stop the fit.
        const N_SAMPLES: usize = 5;
        const WINDOW: usize = 3;

        let run = |streamed: usize, keyed: usize| -> Result<(), HwePcaError> {
            let data: Vec<f64> = (0..N_SAMPLES * streamed)
                .map(|idx| (idx % 3) as f64)
                .collect();
            let keys: Vec<VariantKey> = (0..keyed)
                .map(|idx| VariantKey::new("1", 1_000 + idx as u64 * 50))
                .collect();
            let config = LdResolvedConfig {
                window: LdResolvedWindow::Sites {
                    size: WINDOW,
                    ranges: compute_ld_site_ranges(&keys, WINDOW),
                },
                ridge: DEFAULT_LD_RIDGE,
            };

            let mut source =
                DenseBlockSource::new(&data, N_SAMPLES, streamed).expect("dense source");
            compute_stats_and_ld_weights(
                &mut source,
                streamed,
                config,
                streamed,
                &Arc::new(NoopFitProgress),
                Par::Seq,
            )
            .map(|_| ())
        };

        for (streamed, keyed) in [(7usize, 9usize), (9, 7)] {
            let err = run(streamed, keyed).expect_err(
                "a schedule cut from a different variant list must not produce weights",
            );
            assert!(
                err.to_string().contains("different variant list"),
                "streamed {streamed} against {keyed} keys reported: {err}"
            );
        }

        // The agreeing case is the control: the same machinery must weight a
        // stream whose length matches the list it was cut from.
        run(8, 8).expect("matched lists are the case this is meant to allow");
    }

    #[test]
    fn unconverged_solve_is_refused_unless_explicitly_allowed() {
        let unconverged = FitDiagnostics {
            solver: FitSolver::BlockKrylov,
            converged: false,
            passes: 24,
            max_relative_residual: Some(3.5e-2),
            subspace_delta: Some(1.1e-3),
            boundary_gap: None,
            restarts: 1,
        };

        let err = require_converged(Some(&unconverged), false)
            .expect_err("an unconverged subspace must not become a model by default");
        let message = err.to_string();
        assert!(
            message.contains("--allow-unconverged"),
            "the refusal must name its opt-in; it said: {message}"
        );
        assert!(
            message.contains("24"),
            "the refusal must carry the solve's own numbers; it said: {message}"
        );

        require_converged(Some(&unconverged), true)
            .expect("the opt-in exists so that a best-effort fit can be accepted deliberately");

        let converged = FitDiagnostics {
            converged: true,
            ..unconverged
        };
        require_converged(Some(&converged), false).expect("a converged solve is a fit");
        require_converged(None, false).expect("a route that records nothing is not a refusal");

        assert!(
            !FitOptions::default().allow_unconverged,
            "refusing is the default the whole gate depends on"
        );
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
            diagnostics: None,
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
    fn variant_statistics_reuse_standardized_column_norms() {
        let n_samples = 7;
        let n_variants = 4;
        let mut data = vec![
            0.0,
            1.0,
            2.0,
            f64::NAN,
            1.0,
            0.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            0.125,
            0.5,
            1.25,
            1.875,
            f64::NAN,
            0.75,
            1.5,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];
        let mut cache = VariantStatsCache::new(n_variants, n_variants);
        let block = MatRef::from_column_major_slice(&data, n_samples, n_variants);
        cache.ensure_statistics(block, 0..n_variants, Par::Seq);
        cache.finalize();
        let (scaler, standardized_sums_sq) = cache.into_parts().expect("finalized statistics");

        let mut standardized =
            MatMut::from_column_major_slice_mut(&mut data, n_samples, n_variants);
        scaler.standardize_block(standardized.rb_mut(), 0..n_variants, Par::Seq);
        let direct: Vec<f64> = standardized
            .as_ref()
            .col_iter()
            .map(|column| column.iter().map(|value| value * value).sum())
            .collect();

        for (variant, (&reused, &expected)) in
            standardized_sums_sq.iter().zip(direct.iter()).enumerate()
        {
            let tolerance = 1.0e-12 * expected.abs().max(1.0);
            assert!(
                (reused - expected).abs() <= tolerance,
                "variant {variant}: reused {reused}, direct {expected}"
            );
        }

        let weights = [0.5, 2.0, 1.25, 3.0];
        let reused = weighted_standardized_frobenius_sq(&standardized_sums_sq, Some(&weights));
        let expected: f64 = direct
            .iter()
            .zip(weights.iter())
            .map(|(&sum_sq, &weight)| sum_sq * weight * weight)
            .sum();
        assert!((reused - expected).abs() <= 1.0e-12 * expected.abs().max(1.0));
    }

    #[test]
    fn packed_hard_call_moments_match_decoded_values_for_every_tail() {
        let values = [0.0, 1.0, 2.0, f64::NAN, 2.0, 0.0, f64::NAN, 1.0, 2.0];

        for n_samples in 0..=values.len() {
            let mut packed = vec![0u8; n_samples.div_ceil(4)];
            assert!(pack_hard_calls_into(
                &mut packed,
                &values[..n_samples],
                n_samples,
            ));

            for swap in [false, true] {
                let expected: Vec<f64> = values[..n_samples]
                    .iter()
                    .copied()
                    .filter(|value| value.is_finite())
                    .map(|value| if swap { 2.0 - value } else { value })
                    .collect();
                let expected_sum = expected.iter().sum::<f64>();
                let expected_sum_sq = expected.iter().map(|value| value * value).sum::<f64>();
                let actual = packed_variant_moments(&packed, n_samples, swap);
                assert_eq!(actual, (expected_sum, expected_sum_sq, expected.len()));

                let selection: Vec<usize> =
                    (0..n_samples).filter(|sample| sample % 3 != 1).collect();
                let selected: Vec<f64> = selection
                    .iter()
                    .map(|&sample| values[sample])
                    .filter(|value| value.is_finite())
                    .map(|value| if swap { 2.0 - value } else { value })
                    .collect();
                let selected_sum = selected.iter().sum::<f64>();
                let selected_sum_sq = selected.iter().map(|value| value * value).sum::<f64>();
                let masks = build_sample_byte_masks(packed.len(), &selection).unwrap();
                assert_eq!(
                    packed_variant_moments_selected(&packed, &masks, selection.len(), swap),
                    Some((selected_sum, selected_sum_sq, selected.len()))
                );
            }
        }
    }

    #[test]
    fn packed_sample_missing_counts_match_selected_logical_rows() {
        const SAMPLES: usize = 7;
        const VARIANTS: usize = 4;
        let data = vec![
            f64::NAN,
            0.0,
            1.0,
            2.0,
            f64::NAN,
            0.0,
            2.0,
            0.0,
            f64::NAN,
            1.0,
            f64::NAN,
            2.0,
            0.0,
            2.0,
            0.0,
            1.0,
            2.0,
            0.0,
            1.0,
            2.0,
            0.0,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
            f64::NAN,
        ];
        let bytes_per_variant = SAMPLES.div_ceil(4);
        let mut bytes = vec![0u8; bytes_per_variant * VARIANTS];
        for variant in 0..VARIANTS {
            let source = &data[variant * SAMPLES..(variant + 1) * SAMPLES];
            let packed = &mut bytes[variant * bytes_per_variant..(variant + 1) * bytes_per_variant];
            assert!(pack_hard_calls_into(packed, source, SAMPLES));
        }

        let packed = HardCallPacked::new(&bytes, bytes_per_variant, VARIANTS);
        assert_eq!(
            packed.sample_missing_counts::<NoopFitProgress>(SAMPLES, None),
            Some(vec![2, 2, 1, 2, 2, 1, 1]),
            "the padded tail lane must not become a sample"
        );

        let selected = packed.with_sample_selection(&[0, 3, 5]);
        assert_eq!(
            selected.sample_missing_counts::<NoopFitProgress>(3, None),
            Some(vec![2, 2, 1])
        );
    }

    #[test]
    fn precomputed_variant_statistics_preserve_exact_moment_finalization() {
        let moments = [(3.0, 5.0, 2), (0.0, 0.0, 0), (4.0, 6.0, 3)];
        let statistics = PrecomputedVariantStatistics::from_moments(7, &moments);
        assert!(statistics.matches(7, moments.len()));
        assert!(!statistics.matches(8, moments.len()));

        let (scaler, standardized_sums_sq, observed) = statistics.cloned_parts();
        assert_eq!(observed, moments.len());
        for (variant, &(sum, sum_sq, calls)) in moments.iter().enumerate() {
            let (frequency, scale, standardized_sum_sq) =
                finalize_variant_moments(sum, sum_sq, calls);
            assert_eq!(scaler.allele_frequencies()[variant], frequency);
            assert_eq!(scaler.variant_scales()[variant], scale);
            assert_eq!(standardized_sums_sq[variant], standardized_sum_sq);
        }
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

    fn drain_source<S>(source: &mut S, block_width: usize) -> Vec<f64>
    where
        S: VariantBlockSource,
        S::Error: fmt::Debug,
    {
        let n_samples = source.n_samples();
        let mut block = vec![0.0f64; n_samples * block_width];
        let mut collected = Vec::new();
        loop {
            let filled = source
                .next_block_into(block_width, &mut block)
                .expect("block read");
            if filled == 0 {
                break;
            }
            collected.extend_from_slice(&block[..filled * n_samples]);
        }
        collected
    }

    /// Column-major genotype matrix with a fixed pseudo-random pattern, so both
    /// sides of an equivalence test see byte-identical inputs.
    fn synthetic_genotypes(n_samples: usize, n_variants: usize) -> Vec<f64> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut data = Vec::with_capacity(n_samples * n_variants);
        for _ in 0..n_samples * n_variants {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            data.push(((state >> 33) % 3) as f64);
        }
        data
    }

    struct DirectPackedSource<'a> {
        inner: DenseBlockSource<'a>,
        packed: Vec<u8>,
        bytes_per_variant: usize,
        n_variants: usize,
        match_kinds: Option<Vec<MatchKind>>,
        sample_selection: Option<Vec<usize>>,
    }

    impl<'a> DirectPackedSource<'a> {
        fn new(data: &'a [f64], n_samples: usize, n_variants: usize) -> Self {
            let inner = DenseBlockSource::new(data, n_samples, n_variants).expect("dense source");
            let bytes_per_variant = bytes_per_variant(n_samples);
            let mut packed = vec![0u8; bytes_per_variant * n_variants];
            for variant in 0..n_variants {
                let src = &data[variant * n_samples..(variant + 1) * n_samples];
                let dst =
                    &mut packed[variant * bytes_per_variant..(variant + 1) * bytes_per_variant];
                assert!(pack_hard_calls_into(dst, src, n_samples));
            }
            Self {
                inner,
                packed,
                bytes_per_variant,
                n_variants,
                match_kinds: None,
                sample_selection: None,
            }
        }

        fn with_match_kinds(mut self, match_kinds: Vec<MatchKind>) -> Self {
            assert_eq!(match_kinds.len(), self.n_variants);
            self.match_kinds = Some(match_kinds);
            self
        }

        fn with_sample_selection(mut self, sample_selection: Vec<usize>) -> Self {
            assert!(!sample_selection.is_empty());
            assert!(sample_selection.windows(2).all(|pair| pair[0] < pair[1]));
            assert!(sample_selection.last().copied().unwrap() < self.inner.n_samples());
            self.sample_selection = Some(sample_selection);
            self
        }
    }

    impl VariantBlockSource for DirectPackedSource<'_> {
        type Error = Infallible;

        fn n_samples(&self) -> usize {
            self.sample_selection
                .as_deref()
                .map_or_else(|| self.inner.n_samples(), <[usize]>::len)
        }

        fn n_variants(&self) -> usize {
            self.n_variants
        }

        fn reset(&mut self) -> Result<(), Self::Error> {
            self.inner.reset()
        }

        fn next_block_into(
            &mut self,
            max_variants: usize,
            storage: &mut [f64],
        ) -> Result<usize, Self::Error> {
            let Some(selection) = self.sample_selection.as_deref() else {
                return self.inner.next_block_into(max_variants, storage);
            };
            let physical_n_samples = self.inner.n_samples();
            let mut physical = vec![0.0; physical_n_samples * max_variants];
            let filled = self.inner.next_block_into(max_variants, &mut physical)?;
            for variant in 0..filled {
                for (logical, &sample) in selection.iter().enumerate() {
                    storage[variant * selection.len() + logical] =
                        physical[variant * physical_n_samples + sample];
                }
            }
            Ok(filled)
        }

        fn hard_call_packed(&mut self) -> Option<HardCallPacked<'_>> {
            Some(HardCallPacked {
                data: &self.packed,
                bytes_per_variant: self.bytes_per_variant,
                physical_n_variants: self.n_variants,
                selection: None,
                match_kinds: self.match_kinds.clone(),
                sample_selection: self.sample_selection.clone(),
                sample_byte_masks: self.sample_selection.as_deref().and_then(|selection| {
                    build_sample_byte_masks(self.bytes_per_variant, selection)
                }),
                missing_variant: None,
            })
        }
    }

    #[test]
    fn packed_sources_are_not_decoded_and_repacked_into_a_second_cache() {
        const N_SAMPLES: usize = 17;
        const N_VARIANTS: usize = 11;
        let data = synthetic_genotypes(N_SAMPLES, N_VARIANTS);
        let mut source = DirectPackedSource::new(&data, N_SAMPLES, N_VARIANTS);
        let mut cached = CachedVariantBlockSource::new(&mut source, true);

        assert!(matches!(cached.state, CacheState::Disabled));
        assert!(
            cached.hard_call_packed().is_some(),
            "the original packed view must pass through the disabled cache"
        );
        assert_eq!(drain_source(&mut cached, 4), data);
        assert!(matches!(cached.state, CacheState::Disabled));
    }

    #[test]
    fn direct_packed_operator_applies_selected_allele_swaps() {
        const N_SAMPLES: usize = 17;
        const N_VARIANTS: usize = 5;
        const SWAPPED: usize = 2;
        let physical = synthetic_genotypes(N_SAMPLES, N_VARIANTS);
        let mut oriented = physical.clone();
        for value in &mut oriented[SWAPPED * N_SAMPLES..(SWAPPED + 1) * N_SAMPLES] {
            *value = 2.0 - *value;
        }

        let mut stats_source =
            DenseBlockSource::new(&oriented, N_SAMPLES, N_VARIANTS).expect("stats source");
        let stats_progress = StageProgressHandle::new(
            Arc::new(NoopFitProgress),
            FitProgressStage::AlleleStatistics,
        );
        let (scaler, _, observed) = compute_variant_statistics(
            &mut stats_source,
            N_VARIANTS,
            Par::Seq,
            stats_progress,
            N_VARIANTS,
        )
        .expect("variant statistics");

        let mut general_source =
            DenseBlockSource::new(&oriented, N_SAMPLES, N_VARIANTS).expect("general source");
        let general =
            covariance_operator(&mut general_source, N_VARIANTS, observed, scaler.clone());

        let mut kinds = vec![MatchKind::Exact; N_VARIANTS];
        kinds[SWAPPED] = MatchKind::Swap;
        let mut direct =
            DirectPackedSource::new(&physical, N_SAMPLES, N_VARIANTS).with_match_kinds(kinds);
        let mut cached = CachedVariantBlockSource::new(&mut direct, true);
        let packed = covariance_operator(&mut cached, N_VARIANTS, observed, scaler);

        let rhs = Mat::<f64>::from_fn(N_SAMPLES, 5, |row, col| {
            ((row * 11 + col * 7) % 13) as f64 - 6.0
        });
        let mut expected = Mat::<f64>::zeros(N_SAMPLES, 5);
        let mut mem = MemBuffer::new(general.apply_scratch(rhs.ncols(), Par::Seq));
        general.apply(
            expected.as_mut(),
            rhs.as_ref(),
            Par::Seq,
            MemStack::new(&mut mem),
        );

        let mut actual = Mat::<f64>::zeros(N_SAMPLES, 5);
        assert!(packed.try_apply_hardcall_packed(actual.as_mut(), rhs.as_ref()));
        let mut max_diff = 0.0f64;
        for col in 0..rhs.ncols() {
            for row in 0..N_SAMPLES {
                max_diff = max_diff.max((expected[(row, col)] - actual[(row, col)]).abs());
            }
        }
        assert!(max_diff < 1.0e-9, "max difference was {max_diff}");
    }

    #[test]
    fn direct_packed_operator_applies_selected_sample_rows() {
        const PHYSICAL_SAMPLES: usize = 37;
        const N_VARIANTS: usize = 13;
        let sample_selection: Vec<usize> = (0..PHYSICAL_SAMPLES)
            .filter(|sample| sample % 3 != 1)
            .collect();
        let n_samples = sample_selection.len();
        let mut physical = synthetic_genotypes(PHYSICAL_SAMPLES, N_VARIANTS);
        for (index, value) in physical.iter_mut().enumerate() {
            if index % 29 == 7 {
                *value = f64::NAN;
            }
        }
        let mut selected = Vec::with_capacity(n_samples * N_VARIANTS);
        for variant in 0..N_VARIANTS {
            for &sample in &sample_selection {
                selected.push(physical[variant * PHYSICAL_SAMPLES + sample]);
            }
        }

        let mut stats_source =
            DenseBlockSource::new(&selected, n_samples, N_VARIANTS).expect("stats source");
        let stats_progress = StageProgressHandle::new(
            Arc::new(NoopFitProgress),
            FitProgressStage::AlleleStatistics,
        );
        let (scaler, _, observed) = compute_variant_statistics(
            &mut stats_source,
            N_VARIANTS,
            Par::Seq,
            stats_progress,
            N_VARIANTS,
        )
        .expect("variant statistics");

        let mut general_source =
            DenseBlockSource::new(&selected, n_samples, N_VARIANTS).expect("general source");
        let general =
            covariance_operator(&mut general_source, N_VARIANTS, observed, scaler.clone());

        let mut direct = DirectPackedSource::new(&physical, PHYSICAL_SAMPLES, N_VARIANTS)
            .with_sample_selection(sample_selection);
        let mut cached = CachedVariantBlockSource::new(&mut direct, true);
        let packed = covariance_operator(&mut cached, N_VARIANTS, observed, scaler);

        let rhs = Mat::<f64>::from_fn(n_samples, 5, |row, col| {
            ((row * 11 + col * 7) % 13) as f64 - 6.0
        });
        let mut expected = Mat::<f64>::zeros(n_samples, 5);
        let mut mem = MemBuffer::new(general.apply_scratch(rhs.ncols(), Par::Seq));
        general.apply(
            expected.as_mut(),
            rhs.as_ref(),
            Par::Seq,
            MemStack::new(&mut mem),
        );

        let mut actual = Mat::<f64>::zeros(n_samples, 5);
        assert!(packed.try_apply_hardcall_packed(actual.as_mut(), rhs.as_ref()));
        let mut scale = 0.0f64;
        let mut max_diff = 0.0f64;
        for col in 0..rhs.ncols() {
            for row in 0..n_samples {
                scale = scale.max(expected[(row, col)].abs());
                max_diff = max_diff.max((expected[(row, col)] - actual[(row, col)]).abs());
            }
        }
        assert!(
            max_diff <= 1.0e-10 * scale.max(1.0),
            "selected packed rows differ by {max_diff} (magnitude {scale})"
        );
    }

    /// Builds a streaming covariance operator over `source`, pinning the
    /// progress-observer type parameter for a test with no progress output.
    fn covariance_operator<S>(
        source: &mut S,
        block_capacity: usize,
        observed_variants: usize,
        scaler: HweScaler,
    ) -> StandardizedCovarianceOp<'_, S, NoopFitProgress>
    where
        S: VariantBlockSource + Send,
        S::Error: Error + Send + Sync + 'static,
    {
        StandardizedCovarianceOp::new(
            source,
            block_capacity,
            observed_variants,
            observed_variants,
            scaler,
            None,
            None,
        )
    }

    /// The packed 2-bit kernel must compute the same operator application as
    /// the general f64 tile path it is an optimization of.
    ///
    /// The kernel walks its right-hand side in chunks of
    /// `PACKED_RHS_CHUNK_COLS`, so the widths that matter straddle a chunk
    /// boundary: 32 fills a chunk exactly, 33 leaves a one-column remainder,
    /// and 1 is the single-column case the kernel has to keep reproducing.
    ///
    /// It is called here directly rather than through `apply`, which gates the
    /// packed path at `PACKED_RHS_MAX_COLS` for throughput. The gate decides
    /// *when* the kernel runs; this test decides whether it is right, and the
    /// two should not be entangled — a chunk-loop off-by-one that only shows up
    /// at width 33 must not be able to hide behind a gate at 8.
    #[test]
    fn packed_hard_call_kernel_matches_the_general_path() {
        const N_SAMPLES: usize = 37;
        const N_VARIANTS: usize = 23;
        const BLOCK_CAPACITY: usize = 8;

        // 37 samples is deliberately not a multiple of four, so the last packed
        // byte carries padding the kernel has to stop short of; the scattered
        // missing calls exercise the 2-bit "missing" code on both paths.
        let mut data = synthetic_genotypes(N_SAMPLES, N_VARIANTS);
        for (index, value) in data.iter_mut().enumerate() {
            if index % 17 == 5 {
                *value = f64::NAN;
            }
        }

        let mut stats_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("dense source");
        let stats_progress = StageProgressHandle::new(
            Arc::new(NoopFitProgress),
            FitProgressStage::AlleleStatistics,
        );
        let (scaler, _, observed_variants) = compute_variant_statistics(
            &mut stats_source,
            BLOCK_CAPACITY,
            Par::Seq,
            stats_progress,
            N_VARIANTS,
        )
        .expect("variant statistics");
        assert_eq!(observed_variants, N_VARIANTS);

        // General path: a source with no packed view at all, so the operator
        // decodes f64 tiles and multiplies them with faer.
        let mut general_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("dense source");
        let general = covariance_operator(
            &mut general_source,
            BLOCK_CAPACITY,
            observed_variants,
            scaler.clone(),
        );

        // Packed path: the same genotypes behind the 2-bit cache, warmed by one
        // full traversal so `hard_call_packed` has something to hand out.
        let mut packed_source =
            DenseBlockSource::new(&data, N_SAMPLES, N_VARIANTS).expect("dense source");
        let mut packed_cache = CachedVariantBlockSource::new(&mut packed_source, true);
        let _ = drain_source(&mut packed_cache, BLOCK_CAPACITY);
        assert!(
            packed_cache.hard_call_packed().is_some(),
            "the 2-bit cache must be warm, or this test compares the general path with itself"
        );
        let packed =
            covariance_operator(&mut packed_cache, BLOCK_CAPACITY, observed_variants, scaler);

        for &ncols in &[1usize, 8, 31, 32, 33, 40] {
            let rhs = Mat::<f64>::from_fn(N_SAMPLES, ncols, |row, col| {
                ((row * 7 + col * 13) % 11) as f64 - 5.0
            });

            let mut general_out = Mat::<f64>::zeros(N_SAMPLES, ncols);
            let mut mem = MemBuffer::new(general.apply_scratch(ncols, Par::Seq));
            {
                let stack = MemStack::new(&mut mem);
                general.apply(general_out.as_mut(), rhs.as_ref(), Par::Seq, stack);
            }

            // `apply` zeroes `out` before dispatching and the kernel only ever
            // accumulates, so the caller owns the zeroing either way.
            let mut packed_out = Mat::<f64>::zeros(N_SAMPLES, ncols);
            assert!(
                packed.try_apply_hardcall_packed(packed_out.as_mut(), rhs.as_ref()),
                "packed kernel refused a {ncols}-column right-hand side"
            );

            let mut scale = 0.0f64;
            let mut max_diff = 0.0f64;
            for row in 0..N_SAMPLES {
                for col in 0..ncols {
                    let expected = general_out[(row, col)];
                    scale = scale.max(expected.abs());
                    max_diff = max_diff.max((expected - packed_out[(row, col)]).abs());
                }
            }

            // Not bit-for-bit: faer's blocked GEMM and the kernel's sequential
            // accumulation sum the same terms in different orders.
            assert!(
                max_diff <= 1.0e-10 * scale.max(1.0),
                "width {ncols}: packed and general paths differ by {max_diff} (magnitude {scale})"
            );
        }
    }

    #[test]
    fn ld_weighted_norms_pair_each_weight_with_its_own_variant() {
        // This path is only reachable through a deserialization fallback, so it
        // had no test at all — and it used to pair weights with entries by
        // incrementing a counter inside a traversal closure, which is correct
        // only if the traversal runs in index order. A mispairing there is
        // invisible: every weight is plausible, the norms stay positive, and
        // only the projector's alignment is quietly wrong.
        //
        // The weights below are chosen so that any permutation gives a
        // different answer: with loadings [1, 2, 3] on one component and
        // weights [1, 10, 100], the weighted square sum is
        // 1 + 400 + 90000 = 90401, and no reordering of those weights
        // reproduces it.
        let mut loadings = Mat::<f64>::zeros(3, 1);
        loadings[(0, 0)] = 1.0;
        loadings[(1, 0)] = 2.0;
        loadings[(2, 0)] = 3.0;

        let weights = [1.0f64, 10.0, 100.0];
        let norms = compute_component_weighted_norms_sq(loadings.as_ref(), Some(&weights));

        assert_eq!(norms.len(), 1);
        assert!(
            (norms[0] - 90401.0).abs() < 1e-9,
            "weights must pair with their own variants: got {}",
            norms[0]
        );

        // Unweighted is the same sum with every weight at one, which is also
        // what the fit path itself asks for.
        let plain = compute_component_weighted_norms_sq(loadings.as_ref(), None);
        assert!((plain[0] - 14.0).abs() < 1e-12, "got {}", plain[0]);

        // Fewer weights than variants: the shortfall is treated as weight one
        // rather than reading out of bounds.
        let short = compute_component_weighted_norms_sq(loadings.as_ref(), Some(&weights[..2]));
        assert!(
            (short[0] - (1.0 + 400.0 + 9.0)).abs() < 1e-9,
            "got {}",
            short[0]
        );
    }

    #[test]
    fn tile_width_shrinks_as_the_cohort_grows() {
        // Small cohorts keep the wide tile: the buffers are trivial and wider
        // tiles amortize per-block overhead.
        assert_eq!(adaptive_block_capacity(1_000, 0), DEFAULT_BLOCK_WIDTH);

        // A biobank-scale cohort must not silently reserve gigabytes per
        // decode buffer just because the width was a compile-time constant.
        let wide = adaptive_block_capacity(1_000, 0);
        let narrow = adaptive_block_capacity(2_000_000, 0);
        assert!(
            narrow <= wide,
            "tile width must not grow with the sample count: {narrow} vs {wide}"
        );

        // The variant hint still bounds it.
        assert_eq!(adaptive_block_capacity(1_000, 7), 7);

        // And when the budget affords nothing at all the answer is one variant,
        // not a hard floor of 256 that would demand gigabytes of scratch the
        // plan just said the machine does not have. One is the only real floor:
        // a zero-width tile makes no progress.
        assert_eq!(adaptive_block_capacity(usize::MAX / 16, 0), 1);
    }

    #[test]
    fn dense_covariance_is_reserved_for_small_problems() {
        let budget = gram_matrix_budget_bytes();

        // The reference path is still chosen where it belongs: a handful of
        // samples, where forming C outright is the cheaper way to get it and
        // the exact solve is what the iterative path is checked against.
        assert_eq!(
            covariance_computation_mode(12, 40, 3, budget),
            CovarianceComputationMode::Dense
        );

        // 100k samples is the case that motivated this: 8n² is 74.5 GiB, which
        // "fits" a large machine, and forming it costs O(p·n²) followed by an
        // O(n³) eigendecomposition. Fitting is not a reason to choose it.
        assert_eq!(
            covariance_computation_mode(100_000, 500_000, 20, usize::MAX),
            CovarianceComputationMode::Partial
        );

        // Even well inside the memory budget, dense loses on work long before
        // it loses on bytes.
        assert_eq!(
            covariance_computation_mode(2_000, 500_000, 20, usize::MAX),
            CovarianceComputationMode::Partial
        );

        // A source that cannot report its variant count is still decided on
        // work, not waved through: the count cancels out of the accumulation
        // comparison, so the answer does not depend on knowing it.
        assert_eq!(
            covariance_computation_mode(12, 0, 3, budget),
            CovarianceComputationMode::Dense
        );
        assert_eq!(
            covariance_computation_mode(4_000, 0, 20, usize::MAX),
            CovarianceComputationMode::Partial
        );
    }

    #[test]
    fn sample_subset_source_gathers_requested_rows() {
        // 4 samples × 3 variants, column-major per variant.
        let data: Vec<f64> = vec![
            10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0, 30.0, 31.0, 32.0, 33.0,
        ];
        let inner = DenseBlockSource::new(&data, 4, 3).expect("dense source");
        let mut source = SampleSubsetSource::new(inner, vec![1, 3]).expect("subset source");

        assert_eq!(source.n_samples(), 2);
        assert_eq!(
            source.block_storage_samples(),
            4,
            "block planning must budget for physical decode rows before the gather"
        );
        assert_eq!(source.n_variants(), 3);

        // Block width 2 so the gather runs across more than one block.
        let gathered = drain_source(&mut source, 2);
        assert_eq!(
            gathered,
            vec![11.0, 13.0, 21.0, 23.0, 31.0, 33.0],
            "subset must keep the requested rows, in dataset order, per variant"
        );

        source.reset().expect("reset");
        assert_eq!(drain_source(&mut source, 3), gathered, "reset must rewind");
    }

    #[test]
    fn sample_subset_passthrough_is_the_identity() {
        let data = synthetic_genotypes(5, 7);
        let inner = DenseBlockSource::new(&data, 5, 7).expect("dense source");
        let mut source = SampleSubsetSource::passthrough(inner);

        assert_eq!(source.n_samples(), 5);
        assert_eq!(drain_source(&mut source, 4), data);
    }

    #[test]
    fn streaming_variant_filter_retains_its_post_filter_hint_across_resets() {
        let data = vec![
            0.0,
            0.0,
            0.0,
            0.0, // MAF 0.0: drop
            0.0,
            1.0,
            1.0,
            2.0, // MAF 0.5: retain
            0.0,
            1.0,
            f64::NAN,
            f64::NAN, // MAF 0.25 but 50% missing: drop
            0.0,
            0.0,
            0.0,
            1.0, // MAF 0.125: drop
        ];
        let inner = DenseBlockSource::new(&data, 4, 4).expect("dense source");
        let mut source = StreamingVariantFilterSource::new(inner, Some(0.2), Some(0.25))
            .expect("variant filter");

        let retained = drain_source(&mut source, 2);
        assert_eq!(retained, vec![0.0, 1.0, 1.0, 2.0]);
        assert_eq!(source.n_variants(), 1);

        source.reset().expect("reset");
        assert_eq!(
            source.n_variants(),
            1,
            "the solver must size LD to post-QC markers"
        );
        assert_eq!(drain_source(&mut source, 2), retained);
        assert_eq!(source.retained_indices(), &[1]);
    }

    #[test]
    fn call_rate_threshold_is_inclusive() {
        let values = [0.0, 1.0, f64::NAN, f64::NAN];
        assert!(variant_passes_qc(&values, Some(0.2), Some(0.5)));
        assert!(!variant_passes_qc(&values, Some(0.2), Some(0.499)));
    }

    #[test]
    fn sample_subset_rejects_malformed_index_sets() {
        let data = synthetic_genotypes(4, 2);
        let make = || DenseBlockSource::new(&data, 4, 2).expect("dense source");

        assert!(
            SampleSubsetSource::new(make(), Vec::new()).is_err(),
            "an empty subset leaves nothing to fit"
        );
        assert!(
            SampleSubsetSource::new(make(), vec![2, 1]).is_err(),
            "out-of-order indices would scramble the sample order"
        );
        assert!(
            SampleSubsetSource::new(make(), vec![1, 1]).is_err(),
            "a duplicated row would be counted twice in every statistic"
        );
        assert!(
            SampleSubsetSource::new(make(), vec![0, 4]).is_err(),
            "an index past the last sample must not read out of bounds"
        );
    }

    #[test]
    fn subset_fit_matches_a_fit_on_the_pre_subset_matrix() {
        const N_SAMPLES: usize = 12;
        const N_VARIANTS: usize = 40;
        const COMPONENTS: usize = 3;
        let keep: [usize; 7] = [0, 2, 3, 5, 8, 9, 11];

        let full = synthetic_genotypes(N_SAMPLES, N_VARIANTS);

        // What a caller would otherwise have to materialize with plink2 first.
        let mut pre_subset = Vec::with_capacity(keep.len() * N_VARIANTS);
        for variant in 0..N_VARIANTS {
            for &row in keep.iter() {
                pre_subset.push(full[variant * N_SAMPLES + row]);
            }
        }

        let mut subset_source = SampleSubsetSource::new(
            DenseBlockSource::new(&full, N_SAMPLES, N_VARIANTS).expect("dense source"),
            keep.to_vec(),
        )
        .expect("subset source");
        let subset_model =
            HwePcaModel::fit_k(&mut subset_source, COMPONENTS).expect("subset fit succeeds");

        let mut reference_source =
            DenseBlockSource::new(&pre_subset, keep.len(), N_VARIANTS).expect("dense source");
        let reference_model =
            HwePcaModel::fit_k(&mut reference_source, COMPONENTS).expect("reference fit succeeds");

        assert_eq!(subset_model.n_samples(), keep.len());
        assert_eq!(subset_model.components(), reference_model.components());

        for (subset, reference) in subset_model
            .explained_variance()
            .iter()
            .zip(reference_model.explained_variance())
        {
            assert!(
                (subset - reference).abs() <= 1e-10 * reference.abs().max(1.0),
                "eigenvalues diverged: {subset} vs {reference}"
            );
        }

        // Scores are compared per component up to an overall sign, which is the
        // only freedom an eigenvector has.
        let subset_scores = subset_model.sample_scores();
        let reference_scores = reference_model.sample_scores();
        assert_eq!(subset_scores.nrows(), reference_scores.nrows());
        for component in 0..subset_model.components() {
            let mut max_delta = f64::INFINITY;
            for sign in [1.0f64, -1.0] {
                let delta = (0..subset_scores.nrows())
                    .map(|row| {
                        (subset_scores[(row, component)]
                            - sign * reference_scores[(row, component)])
                            .abs()
                    })
                    .fold(0.0f64, f64::max);
                max_delta = max_delta.min(delta);
            }
            assert!(
                max_delta <= 1e-8,
                "component {component} scores diverged by {max_delta}"
            );
        }
    }

    /// A cohort small enough to check by hand and wide enough to be full rank.
    ///
    /// Standardization centres every variant column, so the row space of X lives
    /// in the `n−1` dimensions orthogonal to the all-ones vector; 40 columns
    /// drawn over 9 samples span all of it. Fitting `n−1` components therefore
    /// gives a decomposition that is exact rather than truncated, which is what
    /// lets the reconstruction below be an equality instead of a projection.
    const INVARIANT_SAMPLES: usize = 9;
    const INVARIANT_VARIANTS: usize = 40;

    fn invariant_fit() -> HwePcaModel {
        let data = synthetic_genotypes(INVARIANT_SAMPLES, INVARIANT_VARIANTS);
        let mut source = DenseBlockSource::new(&data, INVARIANT_SAMPLES, INVARIANT_VARIANTS)
            .expect("dense source");
        HwePcaModel::fit_k(&mut source, INVARIANT_SAMPLES - 1).expect("fit succeeds")
    }

    #[test]
    fn portable_model_json_does_not_duplicate_training_coordinates() {
        let model = invariant_fit();
        let value = serde_json::to_value(&model).expect("serialize fitted model");
        assert!(value.get("sample_basis").is_none());
        assert!(value.get("sample_scores").is_none());

        let loaded: HwePcaModel =
            serde_json::from_value(value.clone()).expect("load portable model");
        assert_eq!(loaded.sample_basis().nrows(), 0);
        assert_eq!(loaded.sample_scores().nrows(), 0);
        assert_eq!(loaded.sample_basis().ncols(), model.components());
        assert_eq!(loaded.sample_scores().ncols(), model.components());
        assert_eq!(loaded.explained_variance(), model.explained_variance());
        assert_eq!(
            loaded.variant_loadings().nrows(),
            model.variant_loadings().nrows()
        );
        assert_eq!(
            loaded.variant_loadings().ncols(),
            model.variant_loadings().ncols()
        );

        let mut stale = value;
        stale.as_object_mut().expect("model object").insert(
            "sample_scores".into(),
            serde_json::json!({"nrows": 1, "ncols": 1, "data": [0.0]}),
        );
        assert!(
            serde_json::from_value::<HwePcaModel>(stale).is_err(),
            "the obsolete cohort-matrix schema must fail rather than silently load"
        );
    }

    /// Rebuilds the standardized matrix the fit decomposed, from the same
    /// deterministic genotypes and through the model's own scaler, so both sides
    /// of every comparison below mean the same X.
    fn standardized_matrix(model: &HwePcaModel) -> Vec<f64> {
        let mut standardized = synthetic_genotypes(INVARIANT_SAMPLES, INVARIANT_VARIANTS);
        {
            let mut block = MatMut::from_column_major_slice_mut(
                &mut standardized,
                INVARIANT_SAMPLES,
                INVARIANT_VARIANTS,
            );
            model
                .scaler()
                .standardize_block(block.as_mut(), 0..INVARIANT_VARIANTS, Par::Seq);
        }
        standardized
    }

    /// `VᵀV = I` — off-diagonals included.
    ///
    /// This is the assertion the old fit could not have passed. Normalizing each
    /// loading column to unit length, which is what it did, constrains only the
    /// diagonal and says nothing whatever about the rest of the matrix; a doc
    /// comment there called the result "Euclidean orthonormality", which column
    /// normalization does not give you. It holds now because `V = B·Σ⁻¹` for a
    /// `B` already rotated into the eigenbasis of `BᵀB`, so the off-diagonals are
    /// zero by construction rather than by hope.
    #[test]
    fn variant_loadings_are_orthonormal_not_merely_unit_length() {
        let model = invariant_fit();
        let loadings = model.variant_loadings();
        let components = model.components();
        assert!(components > 0);

        for left in 0..components {
            for right in 0..components {
                let dot: f64 = (0..loadings.nrows())
                    .map(|row| loadings[(row, left)] * loadings[(row, right)])
                    .sum();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1.0e-9,
                    "VᵀV[{left},{right}] = {dot}, expected {expected}"
                );
            }
        }
    }

    /// `UᵀU = I`. The rotation is orthogonal, so it cannot have cost the sample
    /// basis the orthonormality the eigensolver handed it.
    #[test]
    fn sample_basis_stays_orthonormal_through_the_rotation() {
        let model = invariant_fit();
        let basis = model.sample_basis();
        let components = model.components();
        assert_eq!(basis.nrows(), INVARIANT_SAMPLES);

        for left in 0..components {
            for right in 0..components {
                let dot: f64 = (0..basis.nrows())
                    .map(|row| basis[(row, left)] * basis[(row, right)])
                    .sum();
                let expected = if left == right { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1.0e-9,
                    "UᵀU[{left},{right}] = {dot}, expected {expected}"
                );
            }
        }
    }

    /// `λ_i = σ_i²/(n−1)` for every retained component.
    ///
    /// The rescaling this replaced multiplied the singular values by a per-column
    /// norm and left the eigenvalues untouched, which is exactly how the model
    /// ended up needing two different accessors for "the singular values". One
    /// set now, and it satisfies the identity the other set existed to satisfy.
    #[test]
    fn eigenvalues_and_singular_values_describe_one_decomposition() {
        let model = invariant_fit();
        let scale = (model.n_samples() - 1) as f64;
        assert_eq!(model.singular_values().len(), model.components());
        assert_eq!(model.explained_variance().len(), model.components());

        for (idx, (&lambda, &sigma)) in model
            .explained_variance()
            .iter()
            .zip(model.singular_values().iter())
            .enumerate()
        {
            let implied = sigma * sigma / scale;
            assert!(
                (implied - lambda).abs() <= 1.0e-12 * lambda.abs().max(1.0),
                "component {idx}: λ = {lambda} but σ²/(n−1) = {implied}"
            );
        }
    }

    /// `Xᵀ·U = V·Σ`: the loadings really are the variant cross-products divided
    /// by the singular values.
    ///
    /// The right-hand side comes from the model; the left is multiplied out here
    /// from the genotypes, so this ties the stored pieces to the data rather than
    /// only to each other. It holds at any rank, unlike the reconstruction below.
    #[test]
    fn loadings_scale_back_to_the_variant_cross_products() {
        let model = invariant_fit();
        let standardized = standardized_matrix(&model);
        let basis = model.sample_basis();
        let loadings = model.variant_loadings();

        for variant in 0..INVARIANT_VARIANTS {
            for component in 0..model.components() {
                let cross: f64 = (0..INVARIANT_SAMPLES)
                    .map(|sample| {
                        let entry = standardized[variant * INVARIANT_SAMPLES + sample];
                        entry * basis[(sample, component)]
                    })
                    .sum();
                let stored = loadings[(variant, component)] * model.singular_values()[component];
                assert!(
                    (cross - stored).abs() <= 1.0e-9 * cross.abs().max(1.0),
                    "variant {variant}, component {component}: XᵀU = {cross} but V·Σ = {stored}"
                );
            }
        }
    }

    /// `U·Σ·Vᵀ = X`. With every component retained there is no residual left for
    /// the truncation to hide in, so scores, loadings and singular values have to
    /// reproduce the standardized matrix they were derived from, entry by entry.
    #[test]
    fn scores_and_loadings_reconstruct_the_standardized_matrix() {
        let model = invariant_fit();
        assert_eq!(
            model.components(),
            INVARIANT_SAMPLES - 1,
            "reconstruction is only an equality when the fit spans the whole row space"
        );

        let standardized = standardized_matrix(&model);
        let scores = model.sample_scores();
        let loadings = model.variant_loadings();

        for variant in 0..INVARIANT_VARIANTS {
            for sample in 0..INVARIANT_SAMPLES {
                let reconstructed: f64 = (0..model.components())
                    .map(|component| scores[(sample, component)] * loadings[(variant, component)])
                    .sum();
                let expected = standardized[variant * INVARIANT_SAMPLES + sample];
                assert!(
                    (reconstructed - expected).abs() <= 1.0e-9 * expected.abs().max(1.0),
                    "X[{sample},{variant}] reconstructed as {reconstructed}, expected {expected}"
                );
            }
        }
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
