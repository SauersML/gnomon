use crate::adapt_plink2::GenomeBuild;
use crate::pipeline_error::PipelineError;
use crate::score::batch::{self, DenseScratch, PersonLayout, VariantTerms};
use crate::score::complex::{ComplexVariantResolver, resolve_complex_variants};
use crate::score::decide::{self, DecisionContext, RunStrategy};
use crate::score::io;
use crate::score::types::{
    BimRowIndex, FilesetBoundary, PipelineKind, PreparationResult, ReconciledVariantIndex,
    WorkItem,
};
use ahash::AHashMap;
use crossbeam_channel::{Receiver, RecvTimeoutError, bounded};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use memmap2::{Mmap, MmapOptions};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::{BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// --- Pipeline Tuning Parameters ---

/// The number of dense variants to process in a single person-major batch.
/// Tuned for L3 cache efficiency.
const DENSE_BATCH_SIZE: usize = 256;
/// Kept cohorts at or below this size skip the full-row producer entirely.
const SMALL_KEEP_DIRECT_THRESHOLD: usize = 32;
/// The buffer size for complex variant spooling.
const SPOOL_BUFFER_SIZE: usize = 8 * 1024 * 1024;
const DEFAULT_RAM_FRACTION_NUMERATOR: u64 = 7;
const DEFAULT_RAM_FRACTION_DENOMINATOR: u64 = 10;
const MAX_IO_BUDGET_BYTES: usize = 512 * 1024 * 1024;

struct SpoolState {
    writer: BufWriter<File>,
    offsets: AHashMap<BimRowIndex, u64>,
    cursor: u64,
}

fn create_progress_bar(len: u64, message: &str) -> ProgressBar {
    let draw_target = if std::io::stderr().is_terminal() {
        ProgressDrawTarget::stderr_with_hz(20)
    } else {
        ProgressDrawTarget::hidden()
    };

    let pb = ProgressBar::with_draw_target(Some(len), draw_target);
    pb.set_style(
        ProgressStyle::with_template(
            "\n> [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message(message.to_string());

    pb
}

/// Emit periodic plain-text progress lines to stderr when stderr isn't
/// a TTY (the subprocess case: callers pipe gnomon's output and don't
/// see the indicatif bar). Returns whether a line was actually printed,
/// so the caller can update its bookkeeping.
fn maybe_emit_text_progress(
    processed: u64,
    total: u64,
    last_print: &mut Instant,
    last_pct: &mut u64,
    interval: Duration,
    pct_step: u64,
) -> bool {
    let pct = if total == 0 {
        100
    } else {
        processed.saturating_mul(100) / total
    };
    let now = Instant::now();
    let due_by_time = now.duration_since(*last_print) >= interval;
    let due_by_pct = pct >= last_pct.saturating_add(pct_step);
    if !(due_by_time || due_by_pct) {
        return false;
    }
    eprintln!("> Progress: {processed}/{total} variants ({pct}%)");
    *last_print = now;
    *last_pct = pct;
    true
}

/// The completion sender belongs to the pipeline scope, so dropping it wakes
/// this monitor on success, early error, or unwinding. Progress is not a lifetime
/// signal: a failed producer cannot reach its advertised variant count.
fn update_pipeline_progress(
    count: Arc<AtomicU64>,
    bar: ProgressBar,
    total: u64,
    completion: Receiver<()>,
) {
    let stderr_is_tty = std::io::stderr().is_terminal();
    let mut last_print = Instant::now();
    let initial = count.load(Ordering::Relaxed);
    let mut last_pct = if total == 0 {
        100
    } else {
        initial.saturating_mul(100) / total
    };
    if !stderr_is_tty {
        eprintln!("> Progress: {initial}/{total} variants ({last_pct}%)");
    }
    while count.load(Ordering::Relaxed) < total {
        let processed = count.load(Ordering::Relaxed);
        bar.set_position(processed);
        if !stderr_is_tty {
            maybe_emit_text_progress(
                processed,
                total,
                &mut last_print,
                &mut last_pct,
                Duration::from_secs(5),
                5,
            );
        }
        if !matches!(
            completion.recv_timeout(Duration::from_millis(200)),
            Err(RecvTimeoutError::Timeout)
        ) {
            break;
        }
    }
    let processed = count.load(Ordering::Relaxed);
    bar.set_position(processed);
    if !stderr_is_tty && processed >= total {
        eprintln!("> Progress: {total}/{total} variants (100%)");
    }
}

// ========================================================================================
//                          Public API, context & error handling
// ========================================================================================

/// An iterator that pulls items from a channel and groups them into batches.
///
/// This is a `Send`-compatible replacement for `itertools::chunks` on a channel
/// iterator, enabling true streaming processing with `rayon::par_bridge`. It is
/// the key to enabling simultaneous I/O and computation for the dense path.
struct ChannelBatcher<T> {
    rx: Receiver<Result<T, PipelineError>>,
    batch_size: usize,
}

impl<T> ChannelBatcher<T> {
    fn new(rx: Receiver<Result<T, PipelineError>>, batch_size: usize) -> Self {
        Self { rx, batch_size }
    }
}

// The implementation of the `Iterator` trait is what allows this to be used in loops
// and with adapters like `par_bridge`.
impl<T: Send> Iterator for ChannelBatcher<T> {
    // The iterator yields a `Result` containing either a `Vec` of items (a batch)
    // or a `PipelineError` if one was sent by the producer.
    type Item = Result<Vec<T>, PipelineError>;

    fn next(&mut self) -> Option<Self::Item> {
        // First, block waiting for one item. If the channel is empty and has been
        // closed by the producer, `recv()` will return an error, and we'll return `None`,
        // ending the iteration. This is the correct way to terminate the stream.
        match self.rx.recv() {
            // Happy path: We received a valid work item from the producer.
            Ok(Ok(first_item)) => {
                let mut batch = Vec::with_capacity(self.batch_size);
                batch.push(first_item);

                // Fill the batch before yielding it. Dense execution amortizes its
                // pivot and matrix setup over the entire tile; ending a batch merely
                // because the producer is briefly slower can turn remote input into
                // a stream of one-variant "batches" and destroy throughput.
                while batch.len() < self.batch_size {
                    match self.rx.recv() {
                        Ok(Ok(item)) => batch.push(item),
                        Ok(Err(e)) => return Some(Err(e)),
                        // The producer disconnected, so yield the final partial batch.
                        Err(_) => break,
                    }
                }
                // Return the completed (or partially-filled) batch.
                Some(Ok(batch))
            }
            // An error was sent down the channel as the first item. Propagate it.
            Ok(Err(e)) => Some(Err(e)),
            // The producer has disconnected the channel. End the iteration.
            Err(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryBudget {
    max_ram_bytes: usize,
    resident_bytes: usize,
}

impl MemoryBudget {
    fn auto() -> Self {
        let max_ram_bytes = default_max_ram_bytes().max(1);
        // Free pages the allocator kept from preparation are not held: return them first, so the
        // reading counts what this process holds, not what its allocator happened to keep.
        crate::memory::release_free_heap();
        let resident_bytes = usize::try_from(crate::memory::resident_bytes()).unwrap_or(usize::MAX);
        Self {
            max_ram_bytes,
            resident_bytes,
        }
    }

    #[inline]
    pub fn max_ram_bytes(self) -> usize {
        self.max_ram_bytes
    }

    /// What this process held when the budget was read, which the budget itself does not charge.
    #[inline]
    pub fn resident_bytes(self) -> usize {
        self.resident_bytes
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::auto()
    }
}

/// The budget: a fraction of this process's memory, and of its fair share.
///
/// Sizing a budget from free memory alone is a claim about the future -- that nothing
/// else will allocate -- and that claim is false whenever a caller scores several
/// chromosomes at once. Each sibling observes the same free memory, each takes its
/// fraction of the whole, and together they commit a multiple of what exists. The
/// contention is observable, so it is observed rather than described by the caller: the
/// siblings are the gnomon processes that plan against the same memory limit
/// ([`crate::memory::memory_share`]).
fn default_max_ram_bytes() -> usize {
    let (total, available, siblings) = crate::memory::memory_share("gnomon");
    memory_budget_from_system(available, total, siblings)
}

fn memory_budget_from_system(available: u64, total: u64, siblings: u64) -> usize {
    // TWO BOUNDS, AND THE SMALLER WINS.
    //
    // The first is the historical one: a fraction of what is free right now. Alone on a
    // machine that is the whole story.
    //
    // The second is this process's fair share of the machine as a whole. It exists
    // because the first is unstable exactly when it matters: siblings starting together
    // each see memory the others have not yet touched, so a free-memory reading taken at
    // startup licenses far more than the machine can honour once everyone is resident.
    // A share of TOTAL memory does not move as siblings warm up, so it holds the
    // aggregate at one machine's worth however the starts are staggered.
    //
    // A smaller budget selects bounded accumulation where it fits. Zero free
    // memory must never turn into permission to allocate another eight GiB.
    let by_free =
        available.saturating_mul(DEFAULT_RAM_FRACTION_NUMERATOR) / DEFAULT_RAM_FRACTION_DENOMINATOR;
    let by_fair_share = total.saturating_mul(DEFAULT_RAM_FRACTION_NUMERATOR)
        / DEFAULT_RAM_FRACTION_DENOMINATOR
        / siblings.max(1);
    let candidate = by_free.min(by_fair_share);
    usize::try_from(candidate).unwrap_or(usize::MAX).max(1)
}

pub fn preflight_memory(
    prep_result: &PreparationResult,
    memory_budget: MemoryBudget,
) -> Result<(), PipelineError> {
    ensure_memory_floor(prep_result, memory_budget, InputCharge::UNOPENED)?;
    let result_bytes = result_bytes(prep_result)?;
    let csr_bytes = csr_bytes(prep_result)?;
    let row_bytes = usize::try_from(prep_result.bytes_per_variant).map_err(|_| {
        PipelineError::Compute(format!(
            "PLINK row width {} does not fit on this platform.",
            prep_result.bytes_per_variant
        ))
    })?;
    let buffer_count = if should_use_small_keep_direct_for_prep(prep_result) {
        0
    } else {
        io_buffer_count(prep_result, memory_budget)?
    };
    let io_bytes = row_bytes
        .checked_mul(buffer_count)
        .and_then(|bytes| bytes.checked_add(io::local_prefetch_budget(prep_result, memory_budget)))
        .ok_or_else(|| {
            PipelineError::Compute(format!(
                "I/O buffer estimate overflow: row_bytes={row_bytes}, buffers={buffer_count}"
            ))
        })?;

    let consumer_threads = if should_use_small_keep_direct_for_prep(prep_result) {
        0
    } else {
        choose_consumer_threads(result_bytes, memory_budget)
    };
    let accumulator_copies = if should_use_small_keep_direct_for_prep(prep_result) {
        1usize
    } else {
        consumer_threads
            .checked_mul(2)
            .and_then(|v| v.checked_add(2))
            .ok_or_else(|| PipelineError::Compute("Accumulator estimate overflow.".to_string()))?
    };
    let accumulator_bytes = result_bytes
        .checked_mul(accumulator_copies)
        .ok_or_else(|| PipelineError::Compute("Accumulator byte estimate overflow.".to_string()))?;
    let estimated_bytes = accumulator_bytes
        .checked_add(io_bytes)
        .and_then(|v| v.checked_add(csr_bytes))
        .and_then(|v| {
            dense_scratch_bytes(prep_result, DENSE_BATCH_SIZE)
                .ok()
                .and_then(|scratch| scratch.checked_mul(consumer_threads))
                .and_then(|scratch| v.checked_add(scratch))
        })
        .ok_or_else(|| PipelineError::Compute("Memory estimate overflow.".to_string()))?;

    let max_ram = memory_budget.max_ram_bytes();
    if estimated_bytes > max_ram {
        eprintln!(
            "> Memory budget: {}; fast RAM plan estimate is {}, so gnomon will use the bounded accumulator plan.",
            format_bytes(max_ram),
            format_bytes(estimated_bytes)
        );
        return Ok(());
    }

    eprintln!(
        "> Memory budget: {}; estimated peak for selected RAM plan: {} ({} I/O buffer(s)).",
        format_bytes(max_ram),
        format_bytes(estimated_bytes),
        buffer_count
    );
    Ok(())
}

fn should_use_bounded_accumulator(context: &PipelineContext) -> Result<bool, PipelineError> {
    if should_use_small_keep_direct(context) {
        return Ok(false);
    }
    if context.force_bounded_accumulator {
        return Ok(true);
    }
    let prep_result = &context.prep_result;
    let result_bytes = result_bytes(prep_result)?;
    let csr_bytes = csr_bytes(prep_result)?;
    let row_bytes = usize::try_from(prep_result.bytes_per_variant).map_err(|_| {
        PipelineError::Compute(format!(
            "PLINK row width {} does not fit on this platform.",
            prep_result.bytes_per_variant
        ))
    })?;
    let buffer_count = context.io_buffer_count()?;
    let io_bytes = row_bytes
        .checked_mul(buffer_count)
        .and_then(|bytes| {
            bytes.checked_add(io::local_prefetch_budget(
                prep_result,
                context.memory_budget,
            ))
        })
        .ok_or_else(|| {
            PipelineError::Compute(format!(
                "I/O buffer estimate overflow: row_bytes={row_bytes}, buffers={buffer_count}"
            ))
        })?;
    let fast_threads = choose_consumer_threads(result_bytes, context.memory_budget);
    let fast_copies = fast_threads
        .checked_mul(2)
        .and_then(|v| v.checked_add(2))
        .ok_or_else(|| PipelineError::Compute("Accumulator estimate overflow.".to_string()))?;
    let fast_bytes = result_bytes
        .checked_mul(fast_copies)
        .and_then(|v| v.checked_add(csr_bytes))
        .and_then(|v| v.checked_add(io_bytes))
        .and_then(|v| {
            dense_scratch_bytes(prep_result, DENSE_BATCH_SIZE)
                .ok()
                .and_then(|scratch| scratch.checked_mul(fast_threads))
                .and_then(|scratch| v.checked_add(scratch))
        })
        .ok_or_else(|| PipelineError::Compute("Memory estimate overflow.".to_string()))?;
    Ok(fast_bytes > context.memory_budget.max_ram_bytes())
}

fn should_use_small_keep_direct(context: &PipelineContext) -> bool {
    should_use_small_keep_direct_for_prep(&context.prep_result)
}

fn should_use_small_keep_direct_for_prep(prep_result: &PreparationResult) -> bool {
    prep_result.num_people_to_score > 0
        && prep_result.num_people_to_score <= SMALL_KEEP_DIRECT_THRESHOLD
        && prep_result.complex_rules.is_empty()
}

fn open_scoring_bed_source(
    context: &PipelineContext,
    path: &Path,
) -> Result<io::BedSource, PipelineError> {
    io::open_bed_source_for_scoring(
        path,
        context.genome_build,
        &context.prep_result,
        context.memory_budget,
    )
}

/// One accumulator: people × stride exact i64 lanes and people × scores missing counts.
fn result_bytes(prep_result: &PreparationResult) -> Result<usize, PipelineError> {
    let cells = checked_cells_size(prep_result)?;
    let counts = checked_result_size(prep_result)?;
    cells
        .checked_mul(std::mem::size_of::<i64>())
        .and_then(|bytes| bytes.checked_add(counts.checked_mul(std::mem::size_of::<u32>())?))
        .ok_or_else(|| PipelineError::Compute("Result byte estimate overflow.".to_string()))
}

fn csr_bytes(prep_result: &PreparationResult) -> Result<usize, PipelineError> {
    // Each entry's exact weight, at most an i128, and its flags.
    let weights = prep_result
        .sparse_score_columns()
        .len()
        .checked_mul(std::mem::size_of::<i128>() + 1)
        .ok_or_else(|| PipelineError::Compute("CSR weight byte estimate overflow.".to_string()))?;
    let columns = prep_result
        .sparse_score_columns()
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| PipelineError::Compute("CSR column byte estimate overflow.".to_string()))?;
    let offsets = prep_result
        .sparse_row_offsets()
        .len()
        .checked_mul(std::mem::size_of::<u64>())
        .ok_or_else(|| {
            PipelineError::Compute("CSR row-offset byte estimate overflow.".to_string())
        })?;
    weights
        .checked_add(columns)
        .and_then(|v| v.checked_add(offsets))
        .ok_or_else(|| PipelineError::Compute("CSR byte estimate overflow.".to_string()))
}

fn io_buffer_count(
    prep_result: &PreparationResult,
    memory_budget: MemoryBudget,
) -> Result<usize, PipelineError> {
    let row_bytes = usize::try_from(prep_result.bytes_per_variant).map_err(|_| {
        PipelineError::Compute(format!(
            "PLINK row width {} does not fit on this platform.",
            prep_result.bytes_per_variant
        ))
    })?;
    if row_bytes == 0 {
        return Ok(1);
    }

    let max_ram = memory_budget.max_ram_bytes();
    let io_budget = (max_ram / 8).min(MAX_IO_BUDGET_BYTES).max(row_bytes);
    let by_budget = (io_budget / row_bytes).max(1);
    let by_parallelism = worker_ceiling()
        .saturating_mul(DENSE_BATCH_SIZE.saturating_add(64))
        .max(1);
    Ok(by_budget.min(by_parallelism).max(1))
}

pub fn format_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    const TIB: f64 = GIB * 1024.0;
    let b = bytes as f64;
    if b >= TIB {
        format!("{:.2} TiB", b / TIB)
    } else if b >= GIB {
        format!("{:.2} GiB", b / GIB)
    } else if b >= MIB {
        format!("{:.2} MiB", b / MIB)
    } else if b >= KIB {
        format!("{:.2} KiB", b / KIB)
    } else {
        format!("{bytes} B")
    }
}

/// Adjacent rows a local PGEN's producer reads with one call, per rayon thread.
const PGEN_READ_ROWS_PER_THREAD: usize = 64;

pub fn make_bed_buffer_pool(
    context: &PipelineContext,
) -> Result<Arc<io::RowBufferPool>, PipelineError> {
    let count = context.io_buffer_count()?;
    let row_bytes = usize::try_from(context.prep_result.bytes_per_variant).map_err(|_| {
        PipelineError::Compute(format!(
            "PLINK row width {} does not fit on this platform.",
            context.prep_result.bytes_per_variant
        ))
    })?;
    // A dense batcher keeps fewer than `DENSE_BATCH_SIZE` rows while it waits for the rest
    // of its batch, so the producer must not wait for those to come back.
    let buffer_pool = Arc::new(io::RowBufferPool::new(count, DENSE_BATCH_SIZE));
    for _ in 0..count {
        let mut buffer = Vec::new();
        buffer.try_reserve_exact(row_bytes).map_err(|e| {
            PipelineError::Compute(format!(
                "Failed to reserve {} for a PLINK row buffer: {e}",
                format_bytes(row_bytes)
            ))
        })?;
        buffer_pool.push(buffer).map_err(|_| {
            PipelineError::Compute("Failed to initialize PLINK row buffer pool.".to_string())
        })?;
    }
    Ok(buffer_pool)
}

/// Which compute path every variant takes. Production decides per variant; tests and
/// benchmarks force one path, which can change how fast a score is reached but never its value.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Dispatch {
    #[default]
    Decide,
    Dense,
    Sparse,
}

/// Owns shared resource pools and provides a handle to the read-only preparation results.
pub struct PipelineContext {
    pub prep_result: Arc<PreparationResult>,
    /// Where each scored person's calls sit in a packed row.
    pub person_layout: Arc<PersonLayout>,
    pub memory_budget: MemoryBudget,
    pub genome_build: Option<GenomeBuild>,
    /// The compute path of every variant; [`Dispatch::Decide`] outside tests and benchmarks.
    pub dispatch: Dispatch,
    /// Take the bounded accumulator whatever the memory budget; false outside tests.
    pub force_bounded_accumulator: bool,
}

impl PipelineContext {
    /// Creates a new `PipelineContext`, allocating all necessary memory pools.
    pub fn new(prep_result: Arc<PreparationResult>) -> Self {
        Self {
            person_layout: Arc::new(PersonLayout::new(&prep_result)),
            prep_result,
            memory_budget: MemoryBudget::default(),
            genome_build: None,
            dispatch: Dispatch::Decide,
            force_bounded_accumulator: false,
        }
    }

    pub fn with_budget(
        prep_result: Arc<PreparationResult>,
        memory_budget: MemoryBudget,
        genome_build: Option<GenomeBuild>,
    ) -> Self {
        Self {
            person_layout: Arc::new(PersonLayout::new(&prep_result)),
            prep_result,
            memory_budget,
            genome_build,
            dispatch: Dispatch::Decide,
            force_bounded_accumulator: false,
        }
    }

    pub fn io_buffer_count(&self) -> Result<usize, PipelineError> {
        io_buffer_count(&self.prep_result, self.memory_budget)
    }

    pub fn work_channel_bound(&self) -> Result<usize, PipelineError> {
        self.io_buffer_count().map(|count| count.max(1))
    }
}

/// Executes the entire concurrent compute pipeline.
///
/// This is the primary public entry point. It is synchronous and returns the
/// final aggregated scores and counts upon successful completion.
pub fn run(context: &PipelineContext) -> Result<(Vec<i64>, Vec<u32>), PipelineError> {
    ensure_memory_floor(
        &context.prep_result,
        context.memory_budget,
        InputCharge::UNOPENED,
    )?;

    // This match is a zero-cost abstraction. The compiler generates a simple jump
    // to the correct function based on the enum variant, and it's impossible
    // to call the wrong pipeline logic for a given configuration.
    match &context.prep_result.pipeline_kind {
        PipelineKind::SingleFile(bed_path) => run_single_file_pipeline(context, bed_path),
        PipelineKind::MultiFile(boundaries) => run_multi_file_pipeline(context, boundaries),
    }
}

// ========================================================================================
//                        Pipeline stage implementations
// ========================================================================================

/// The pipeline implementation for the common single-fileset case.
/// This function's body is effectively the same as the original `pipeline::run` function,
/// guaranteeing zero performance regression.
fn run_single_file_pipeline(
    context: &PipelineContext,
    bed_path: &Path,
) -> Result<(Vec<i64>, Vec<u32>), PipelineError> {
    // --- 1. Setup: Memory-map the file, create channels and a shared buffer pool ---
    let bed_source = open_scoring_bed_source(context, bed_path)?;
    let input = InputCharge::of(&context.prep_result, std::slice::from_ref(&bed_source));
    ensure_memory_floor(&context.prep_result, context.memory_budget, input)?;
    if should_use_small_keep_direct(context)
        || (context.prep_result.num_people_to_score <= SMALL_KEEP_DIRECT_THRESHOLD
            && bed_source.mmap().is_some())
    {
        return run_small_keep_direct_single_file(context, bed_source);
    }
    let shared_source = bed_source.byte_source();
    // A local PGEN decodes a long read of adjacent rows on the rayon pool, so with more than
    // one thread its producer reads rows in runs. Every other source reads one row per call.
    let read_batch = if crate::shared::files::is_pgen_path(bed_path)
        && bed_path.exists()
        && rayon::current_num_threads() > 1
    {
        PGEN_READ_ROWS_PER_THREAD * rayon::current_num_threads()
    } else {
        1
    };

    let channel_bound = context.work_channel_bound()?;
    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);

    let buffer_pool = make_bed_buffer_pool(context)?;

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let variants_processed_count = Arc::new(AtomicU64::new(0));
    let pb = create_progress_bar(variants_to_process, "Computing scores...");

    // --- 2. Pre-computation & STRATEGY SELECTION ---
    let prep_result = &context.prep_result;
    let run_ctx = DecisionContext {
        n_cohort: prep_result.total_people_in_fam as f32,
        k_scores: prep_result.score_names.len() as f32,
        subset_frac: prep_result.num_people_to_score as f32
            / prep_result.total_people_in_fam as f32,
        freq: 0.0,
    };
    let strategy = decide::RunStrategy::UseComplexTree;
    eprintln!("> Decision Engine Strategy: {strategy:?}");

    let use_bounded_accumulator = should_use_bounded_accumulator(context)?;
    if use_bounded_accumulator {
        announce_bounded_accumulator(prep_result, context.memory_budget, input)?;
    }
    let mut shared_accumulator = if use_bounded_accumulator {
        let (final_scores, final_counts) = initialize_cells(prep_result)?;
        Some(Arc::new(Mutex::new((final_scores, final_counts))))
    } else {
        None
    };

    let has_complex = !prep_result.complex_rules.is_empty();
    let is_remote = bed_source.mmap().is_none();
    let should_spool = has_complex && is_remote;
    let mut spool_state: Option<SpoolState> = None;
    let mut spool_path: Option<PathBuf> = None;
    if should_spool {
        let (spool_dir, spool_stem) = derive_spool_destination(bed_path);
        fs::create_dir_all(&spool_dir).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool directory {}: {e}",
                spool_dir.display()
            ))
        })?;
        let filename = unique_spool_filename(&spool_stem);
        let path = spool_dir.join(&filename);
        let file = File::create(&path).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool file {}: {e}",
                path.display()
            ))
        })?;
        let complex_variant_count = prep_result
            .required_is_complex()
            .iter()
            .filter(|&&flag| flag != 0)
            .count() as u64;
        let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
        let approx_mb = if spool_bytes_per_variant == 0 {
            0.0
        } else {
            (complex_variant_count * spool_bytes_per_variant) as f64 / (1024.0 * 1024.0)
        };
        eprintln!(
            "> Spooling complex genotypes locally: {} variants × {} B ≈ {:.2} MiB to {}",
            complex_variant_count,
            spool_bytes_per_variant,
            approx_mb,
            path.display()
        );
        let offsets_capacity = usize::try_from(complex_variant_count)
            .unwrap_or(usize::MAX / 2)
            .max(1);
        spool_state = Some(SpoolState {
            writer: BufWriter::with_capacity(SPOOL_BUFFER_SIZE, file),
            offsets: AHashMap::with_capacity(offsets_capacity),
            cursor: 0,
        });
        spool_path = Some(path);
    }

    // --- 3. Orchestration: Use a scoped thread for safe producer/consumer execution ---
    let run_ctx_for_closure = run_ctx;
    let strategy_for_closure = strategy;
    let final_result: Result<(Option<(Vec<i64>, Vec<u32>)>, Option<SpoolState>), PipelineError> =
        thread::scope(|s| {
            let (progress_lifetime, progress_completion) = bounded(0);
            let updater_thread_count = Arc::clone(&variants_processed_count);
            let updater_pb = pb.clone();
            s.spawn(move || {
                update_pipeline_progress(
                    updater_thread_count,
                    updater_pb,
                    variants_to_process,
                    progress_completion,
                );
            });

            let mut local_spool_state = spool_state.take();
            let producer_logic = {
                let source = Arc::clone(&shared_source);
                let prep_result = Arc::clone(&context.prep_result);
                let buffer_pool = Arc::clone(&buffer_pool);
                let producer_thread_count = Arc::clone(&variants_processed_count);
                let spool_enabled = should_spool;
                let run_ctx = run_ctx_for_closure;
                let strategy = strategy_for_closure;
                let dispatch = context.dispatch;

                move || -> Result<Option<SpoolState>, PipelineError> {
                    match strategy {
                        RunStrategy::UseSimpleTree => {
                            let global_path = decide::decide_path_without_freq(&run_ctx);
                            let path_decider = |_: &[u8]| global_path;
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::producer_thread_with_read_batch(
                                Arc::clone(&source),
                                Arc::clone(&prep_result),
                                Some(sparse_tx),
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                                read_batch,
                            );
                        }
                        RunStrategy::UseComplexTree => {
                            let path_decider = |variant_data: &[u8]| match dispatch {
                                Dispatch::Dense => decide::ComputePath::Pivot,
                                Dispatch::Sparse => decide::ComputePath::NoPivot,
                                Dispatch::Decide => {
                                    let current_freq = batch::assess_variant_density_for_dispatch(
                                        variant_data,
                                        run_ctx.n_cohort as usize,
                                    );
                                    let variant_ctx = DecisionContext {
                                        freq: current_freq,
                                        ..run_ctx
                                    };
                                    decide::decide_path_with_freq(&variant_ctx)
                                }
                            };
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::producer_thread_with_read_batch(
                                Arc::clone(&source),
                                Arc::clone(&prep_result),
                                Some(sparse_tx),
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                                read_batch,
                            );
                        }
                    }
                    Ok(local_spool_state)
                }
            };

            let producer_handle = s.spawn(producer_logic);
            let (sparse_result, dense_result) =
                if let Some(shared_accumulator) = shared_accumulator.as_ref() {
                    let sparse_accumulator = Arc::clone(shared_accumulator);
                    let dense_accumulator = Arc::clone(shared_accumulator);
                    rayon::join(
                        || {
                            process_sparse_stream_bounded(
                                sparse_rx,
                                context,
                                Arc::clone(&buffer_pool),
                                sparse_accumulator,
                            )
                        },
                        || {
                            process_dense_stream_bounded(
                                dense_rx,
                                context,
                                Arc::clone(&buffer_pool),
                                dense_accumulator,
                                input,
                            )
                        },
                    )
                } else {
                    rayon::join(
                        || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
                        || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
                    )
                };
            let local_spool_state = producer_handle
                .join()
                .map_err(|_| PipelineError::Producer("Producer thread panicked.".to_string()))??;

            let final_outputs = if use_bounded_accumulator {
                sparse_result?;
                dense_result?;
                None
            } else {
                // --- 4. Aggregate final results ---
                let (sparse_adjustments, sparse_counts) = sparse_result?;
                let (dense_adjustments, dense_counts) = dense_result?;
                let (mut final_scores, mut final_counts) = initialize_cells(prep_result)?;
                final_counts
                    .par_iter_mut()
                    .zip(sparse_counts)
                    .for_each(|(m, p)| *m += p);
                final_counts
                    .par_iter_mut()
                    .zip(dense_counts)
                    .for_each(|(m, p)| *m += p);
                final_scores
                    .par_iter_mut()
                    .zip(sparse_adjustments)
                    .for_each(|(m, p)| *m = m.wrapping_add(p));
                final_scores
                    .par_iter_mut()
                    .zip(dense_adjustments)
                    .for_each(|(m, p)| *m = m.wrapping_add(p));
                Some((final_scores, final_counts))
            };

            pb.finish_with_message("Computation complete.");
            drop(progress_lifetime);
            Ok((final_outputs, local_spool_state))
        });
    let (final_outputs, mut spool_state) = final_result?;
    let (mut final_scores, mut final_counts) = if let Some(outputs) = final_outputs {
        outputs
    } else {
        let accumulator = Arc::try_unwrap(shared_accumulator.take().ok_or_else(|| {
            PipelineError::Compute("Bounded accumulator missing after scoring.".to_string())
        })?)
        .map_err(|_| {
            PipelineError::Compute("Bounded accumulator still has outstanding owners.".to_string())
        })?;
        accumulator.into_inner().map_err(|_| {
            PipelineError::Compute("Bounded accumulator lock was poisoned.".to_string())
        })?
    };

    if !prep_result.complex_rules.is_empty() {
        let resolver_label = if should_spool {
            "spooled mmap"
        } else {
            "single-file mmap"
        };
        eprintln!(
            "> Resolving {} unique complex variant rule(s) with {} resolver...",
            prep_result.complex_rules.len(),
            resolver_label
        );
        if should_spool {
            let spool_file_path = spool_path
                .clone()
                .expect("spool path missing despite spooling enabled");
            let offsets = {
                let mut state = spool_state
                    .take()
                    .expect("spool state missing despite spooling enabled");
                state.writer.flush().map_err(|e| {
                    PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
                })?;
                state.offsets
            };
            let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
            with_spool(
                &spool_file_path,
                spool_bytes_per_variant,
                |mmap| {
                    let resolver = ComplexVariantResolver::from_spool(
                        Arc::new(mmap),
                        offsets,
                        spool_bytes_per_variant,
                        Arc::new(prep_result.spool_dense_map().to_vec()),
                    );
                    resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)
                },
                |path| fs::remove_file(path),
            )?;
        } else {
            let resolver = ComplexVariantResolver::from_single_source(bed_source.clone());
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

/// Maps the complex-variant spool at `path` read-only (one zero byte when no spooled row has
/// bytes), hands the mapping to `resolve`, and removes the spool with `remove` once `resolve` has
/// returned, whether or not it succeeded. `resolve` owns the mapping, so the mapping has ended when
/// the spool is unlinked: on NFS a file unlinked while it is still mapped is silly-renamed to a
/// `.nfsXXXX` file, which stayed in the output directory until the mapping ended, or for good if
/// the process died first (#2395).
fn with_spool<T>(
    path: &Path,
    spool_bytes_per_variant: u64,
    resolve: impl FnOnce(Mmap) -> Result<T, PipelineError>,
    remove: impl FnOnce(&Path) -> std::io::Result<()>,
) -> Result<T, PipelineError> {
    let resolved = map_spool(path, spool_bytes_per_variant).and_then(resolve);
    let removed = remove(path);
    let value = resolved?;
    if let Err(e) = removed {
        eprintln!(
            "> Warning: Failed to delete complex spool {}: {}",
            path.display(),
            e
        );
    }
    Ok(value)
}

fn map_spool(path: &Path, spool_bytes_per_variant: u64) -> Result<Mmap, PipelineError> {
    if spool_bytes_per_variant == 0 {
        let mut anon = MmapOptions::new().len(1).map_anon().map_err(|e| {
            PipelineError::Io(format!(
                "Failed to allocate anonymous mapping for empty complex spool: {e}"
            ))
        })?;
        anon.copy_from_slice(&[0u8]);
        return anon.make_read_only().map_err(|e| {
            PipelineError::Io(format!(
                "Failed to convert anonymous mapping to read-only: {e}"
            ))
        });
    }
    let spool_file = File::open(path).map_err(|e| {
        PipelineError::Io(format!(
            "Failed to open complex variant spool {}: {e}",
            path.display()
        ))
    })?;
    unsafe { Mmap::map(&spool_file) }.map_err(|e| {
        PipelineError::Io(format!(
            "Failed to memory-map complex variant spool {}: {e}",
            path.display()
        ))
    })
}

/// The pipeline implementation for the multi-fileset case.
fn run_multi_file_pipeline(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
) -> Result<(Vec<i64>, Vec<u32>), PipelineError> {
    let bed_sources: Vec<io::BedSource> = boundaries
        .iter()
        .map(|b| open_scoring_bed_source(context, &b.bed_path))
        .collect::<Result<_, _>>()?;
    let input = InputCharge::of(&context.prep_result, &bed_sources);
    ensure_memory_floor(&context.prep_result, context.memory_budget, input)?;
    let any_remote = bed_sources.iter().any(|s| s.mmap().is_none());
    if should_use_small_keep_direct(context)
        || (context.prep_result.num_people_to_score <= SMALL_KEEP_DIRECT_THRESHOLD && !any_remote)
    {
        return run_small_keep_direct_multi_file(context, boundaries, bed_sources);
    }
    let shared_sources = Arc::new(bed_sources);

    // --- 1. Setup: No mmap here. Producer manages its own. ---
    let channel_bound = context.work_channel_bound()?;
    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);
    let buffer_pool = make_bed_buffer_pool(context)?;

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let variants_processed_count = Arc::new(AtomicU64::new(0));
    let pb = create_progress_bar(variants_to_process, "Computing scores...");

    // --- 2. Pre-computation (same as single-file) ---
    let prep_result = &context.prep_result;
    let run_ctx = DecisionContext {
        n_cohort: prep_result.total_people_in_fam as f32,
        k_scores: prep_result.score_names.len() as f32,
        subset_frac: prep_result.num_people_to_score as f32
            / prep_result.total_people_in_fam as f32,
        freq: 0.0,
    };
    let strategy = decide::RunStrategy::UseComplexTree;
    eprintln!("> Decision Engine Strategy: {strategy:?}");
    let use_bounded_accumulator = should_use_bounded_accumulator(context)?;
    if use_bounded_accumulator {
        announce_bounded_accumulator(prep_result, context.memory_budget, input)?;
    }
    let mut shared_accumulator = if use_bounded_accumulator {
        let (final_scores, final_counts) = initialize_cells(prep_result)?;
        Some(Arc::new(Mutex::new((final_scores, final_counts))))
    } else {
        None
    };

    let has_complex = !prep_result.complex_rules.is_empty();
    let should_spool = has_complex && any_remote;
    let mut spool_state: Option<SpoolState> = None;
    let mut spool_path: Option<PathBuf> = None;
    if should_spool {
        let (spool_dir, spool_stem) = derive_spool_destination(&boundaries[0].bed_path);
        fs::create_dir_all(&spool_dir).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool directory {}: {e}",
                spool_dir.display()
            ))
        })?;
        let filename = unique_spool_filename(&spool_stem);
        let path = spool_dir.join(&filename);
        let file = File::create(&path).map_err(|e| {
            PipelineError::Io(format!(
                "Failed to create spool file {}: {e}",
                path.display()
            ))
        })?;
        let complex_variant_count = prep_result
            .required_is_complex()
            .iter()
            .filter(|&&flag| flag != 0)
            .count() as u64;
        let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
        let approx_mb = if spool_bytes_per_variant == 0 {
            0.0
        } else {
            (complex_variant_count * spool_bytes_per_variant) as f64 / (1024.0 * 1024.0)
        };
        eprintln!(
            "> Spooling complex genotypes locally: {} variants × {} B ≈ {:.2} MiB to {}",
            complex_variant_count,
            spool_bytes_per_variant,
            approx_mb,
            path.display()
        );
        let offsets_capacity = usize::try_from(complex_variant_count)
            .unwrap_or(usize::MAX / 2)
            .max(1);
        spool_state = Some(SpoolState {
            writer: BufWriter::with_capacity(SPOOL_BUFFER_SIZE, file),
            offsets: AHashMap::with_capacity(offsets_capacity),
            cursor: 0,
        });
        spool_path = Some(path);
    }

    // --- 3. Orchestration with multi-file producer ---
    let run_ctx_for_closure = run_ctx;
    let strategy_for_closure = strategy;
    let final_result: Result<(Option<(Vec<i64>, Vec<u32>)>, Option<SpoolState>), PipelineError> =
        thread::scope(|s| {
            let (progress_lifetime, progress_completion) = bounded(0);
            let updater_thread_count = Arc::clone(&variants_processed_count);
            let updater_pb = pb.clone();
            s.spawn(move || {
                update_pipeline_progress(
                    updater_thread_count,
                    updater_pb,
                    variants_to_process,
                    progress_completion,
                );
            });

            let mut local_spool_state = spool_state.take();
            let producer_logic = {
                let sources = Arc::clone(&shared_sources);
                let prep_result = Arc::clone(&context.prep_result);
                let buffer_pool = Arc::clone(&buffer_pool);
                let producer_thread_count = Arc::clone(&variants_processed_count);
                let spool_enabled = should_spool;
                let run_ctx = run_ctx_for_closure;
                let strategy = strategy_for_closure;
                let dispatch = context.dispatch;

                move || -> Result<Option<SpoolState>, PipelineError> {
                    match strategy {
                        RunStrategy::UseSimpleTree => {
                            let global_path = decide::decide_path_without_freq(&run_ctx);
                            let path_decider = |_: &[u8]| global_path;
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::multi_file_producer_thread(
                                Arc::clone(&prep_result),
                                boundaries,
                                sources.as_ref(),
                                Some(sparse_tx),
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                            );
                        }
                        RunStrategy::UseComplexTree => {
                            let path_decider = |variant_data: &[u8]| match dispatch {
                                Dispatch::Dense => decide::ComputePath::Pivot,
                                Dispatch::Sparse => decide::ComputePath::NoPivot,
                                Dispatch::Decide => {
                                    let current_freq = batch::assess_variant_density_for_dispatch(
                                        variant_data,
                                        run_ctx.n_cohort as usize,
                                    );
                                    let variant_ctx = DecisionContext {
                                        freq: current_freq,
                                        ..run_ctx
                                    };
                                    decide::decide_path_with_freq(&variant_ctx)
                                }
                            };
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::multi_file_producer_thread(
                                Arc::clone(&prep_result),
                                boundaries,
                                sources.as_ref(),
                                Some(sparse_tx),
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                spool_plan,
                            );
                        }
                    }
                    Ok(local_spool_state)
                }
            };

            let producer_handle = s.spawn(producer_logic);
            let (sparse_result, dense_result) =
                if let Some(shared_accumulator) = shared_accumulator.as_ref() {
                    let sparse_accumulator = Arc::clone(shared_accumulator);
                    let dense_accumulator = Arc::clone(shared_accumulator);
                    rayon::join(
                        || {
                            process_sparse_stream_bounded(
                                sparse_rx,
                                context,
                                Arc::clone(&buffer_pool),
                                sparse_accumulator,
                            )
                        },
                        || {
                            process_dense_stream_bounded(
                                dense_rx,
                                context,
                                Arc::clone(&buffer_pool),
                                dense_accumulator,
                                input,
                            )
                        },
                    )
                } else {
                    rayon::join(
                        || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
                        || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
                    )
                };
            let local_spool_state = producer_handle
                .join()
                .map_err(|_| PipelineError::Producer("Producer thread panicked.".to_string()))??;

            let final_outputs = if use_bounded_accumulator {
                sparse_result?;
                dense_result?;
                None
            } else {
                // --- 4. Aggregate final results (same as single-file) ---
                let (sparse_adjustments, sparse_counts) = sparse_result?;
                let (dense_adjustments, dense_counts) = dense_result?;
                let (mut final_scores, mut final_counts) = initialize_cells(prep_result)?;
                final_counts
                    .par_iter_mut()
                    .zip(sparse_counts)
                    .for_each(|(m, p)| *m += p);
                final_counts
                    .par_iter_mut()
                    .zip(dense_counts)
                    .for_each(|(m, p)| *m += p);
                final_scores
                    .par_iter_mut()
                    .zip(sparse_adjustments)
                    .for_each(|(m, p)| *m = m.wrapping_add(p));
                final_scores
                    .par_iter_mut()
                    .zip(dense_adjustments)
                    .for_each(|(m, p)| *m = m.wrapping_add(p));
                Some((final_scores, final_counts))
            };

            pb.finish_with_message("Computation complete.");
            drop(progress_lifetime);
            Ok((final_outputs, local_spool_state))
        });
    let (final_outputs, mut spool_state) = final_result?;
    let (mut final_scores, mut final_counts) = if let Some(outputs) = final_outputs {
        outputs
    } else {
        let accumulator = Arc::try_unwrap(shared_accumulator.take().ok_or_else(|| {
            PipelineError::Compute("Bounded accumulator missing after scoring.".to_string())
        })?)
        .map_err(|_| {
            PipelineError::Compute("Bounded accumulator still has outstanding owners.".to_string())
        })?;
        accumulator.into_inner().map_err(|_| {
            PipelineError::Compute("Bounded accumulator lock was poisoned.".to_string())
        })?
    };

    if !prep_result.complex_rules.is_empty() {
        let resolver_label = if should_spool {
            "spooled mmap"
        } else {
            "multi-file mmap"
        };
        eprintln!(
            "> Resolving {} unique complex variant rule(s) with {} resolver...",
            prep_result.complex_rules.len(),
            resolver_label
        );
        if should_spool {
            let spool_file_path = spool_path
                .clone()
                .expect("spool path missing despite spooling enabled");
            let offsets = {
                let mut state = spool_state
                    .take()
                    .expect("spool state missing despite spooling enabled");
                state.writer.flush().map_err(|e| {
                    PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
                })?;
                state.offsets
            };
            let spool_bytes_per_variant = prep_result.spool_bytes_per_variant();
            with_spool(
                &spool_file_path,
                spool_bytes_per_variant,
                |mmap| {
                    let resolver = ComplexVariantResolver::from_spool(
                        Arc::new(mmap),
                        offsets,
                        spool_bytes_per_variant,
                        Arc::new(prep_result.spool_dense_map().to_vec()),
                    );
                    resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)
                },
                |path| fs::remove_file(path),
            )?;
        } else {
            let resolver = ComplexVariantResolver::from_multi_sources(
                shared_sources.as_ref().clone(),
                boundaries.to_vec(),
            )?;
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

fn run_small_keep_direct_single_file(
    context: &PipelineContext,
    bed_source: io::BedSource,
) -> Result<(Vec<i64>, Vec<u32>), PipelineError> {
    eprintln!(
        "> Using small-keep direct PLINK path for {} kept individual(s).",
        context.prep_result.num_people_to_score
    );
    let prep_result = &context.prep_result;
    let (mut final_scores, mut final_counts) = initialize_cells(prep_result)?;
    let (stride, num_scores) = (prep_result.exact().stride(), prep_result.score_names.len());
    let mut terms = VariantTerms::default();

    let total = prep_result.num_reconciled_variants as u64;
    let pb = create_progress_bar(total, "Computing scores...");
    let mut scratch = [0u8; 1];
    let mut processed_since_update = 0u64;
    // One handle on the map for the whole loop: cloning it per genotype costs an
    // atomic reference-count update each time.
    let mmap = bed_source.mmap();
    let mapped = mmap.as_deref();

    for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
        let reconciled_idx = reconciled_index_from_usize(i)?;
        let row_base = 3u64
            .checked_add(
                bim_row_idx
                    .0
                    .checked_mul(prep_result.bytes_per_variant)
                    .ok_or_else(|| {
                        PipelineError::Compute("PLINK row offset overflow.".to_string())
                    })?,
            )
            .ok_or_else(|| PipelineError::Compute("PLINK row offset overflow.".to_string()))?;
        let row_end = row_base
            .checked_add(prep_result.bytes_per_variant)
            .ok_or_else(|| PipelineError::Compute("PLINK row end overflow.".to_string()))?;
        if row_end > bed_source.len() {
            return Err(PipelineError::Io(format!(
                "Fatal: Attempted to read past the end of the .bed source for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                bim_row_idx.0
            )));
        }

        terms.load(prep_result, reconciled_idx);
        for out_idx in 0..prep_result.num_people_to_score {
            let fam_idx = prep_result.output_idx_to_fam_idx[out_idx].0 as usize;
            let byte_offset = row_base
                .checked_add((fam_idx / 4) as u64)
                .ok_or_else(|| PipelineError::Compute("PLINK byte offset overflow.".to_string()))?;
            let byte = if let Some(mmap) = mapped {
                *mmap.get(byte_offset as usize).ok_or_else(|| {
                    PipelineError::Io(format!(
                        "Fatal: Attempted to read past the end of the .bed source for variant at BIM row {}.",
                        bim_row_idx.0
                    ))
                })?
            } else {
                bed_source.read_at(byte_offset, &mut scratch)?;
                scratch[0]
            };
            let packed = (byte >> ((fam_idx % 4) * 2)) & 0b11;
            terms.apply(
                packed,
                &mut final_scores[out_idx * stride..(out_idx + 1) * stride],
                &mut final_counts[out_idx * num_scores..(out_idx + 1) * num_scores],
            );
        }

        processed_since_update += 1;
        if processed_since_update == io::PROGRESS_UPDATE_BATCH_SIZE {
            pb.inc(processed_since_update);
            processed_since_update = 0;
        }
    }
    if processed_since_update > 0 {
        pb.inc(processed_since_update);
    }
    pb.finish_with_message("Computation complete.");
    if !prep_result.complex_rules.is_empty() {
        let resolver = ComplexVariantResolver::from_single_source(bed_source);
        resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
    }
    Ok((final_scores, final_counts))
}

fn run_small_keep_direct_multi_file(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
    bed_sources: Vec<io::BedSource>,
) -> Result<(Vec<i64>, Vec<u32>), PipelineError> {
    eprintln!(
        "> Using small-keep direct PLINK path for {} kept individual(s).",
        context.prep_result.num_people_to_score
    );
    let prep_result = &context.prep_result;
    let (mut final_scores, mut final_counts) = initialize_cells(prep_result)?;
    let (stride, num_scores) = (prep_result.exact().stride(), prep_result.score_names.len());
    let mut terms = VariantTerms::default();

    let total = prep_result.num_reconciled_variants as u64;
    let pb = create_progress_bar(total, "Computing scores...");
    let mut scratch = [0u8; 1];
    let mut processed_since_update = 0u64;
    let mut current_fileset_idx = 0usize;
    let mut next_boundary_start_idx = if boundaries.len() > 1 {
        boundaries[1].starting_global_index
    } else {
        u64::MAX
    };

    for (i, &global_bim_row_index) in prep_result.required_bim_indices.iter().enumerate() {
        while global_bim_row_index.0 >= next_boundary_start_idx {
            current_fileset_idx += 1;
            next_boundary_start_idx = if boundaries.len() > current_fileset_idx + 1 {
                boundaries[current_fileset_idx + 1].starting_global_index
            } else {
                u64::MAX
            };
        }

        let reconciled_idx = reconciled_index_from_usize(i)?;
        let local_index =
            global_bim_row_index.0 - boundaries[current_fileset_idx].starting_global_index;
        let row_base = 3u64
            .checked_add(
                local_index
                    .checked_mul(prep_result.bytes_per_variant)
                    .ok_or_else(|| {
                        PipelineError::Compute("PLINK row offset overflow.".to_string())
                    })?,
            )
            .ok_or_else(|| PipelineError::Compute("PLINK row offset overflow.".to_string()))?;
        let row_end = row_base
            .checked_add(prep_result.bytes_per_variant)
            .ok_or_else(|| PipelineError::Compute("PLINK row end overflow.".to_string()))?;
        let bed_source = &bed_sources[current_fileset_idx];
        let mmap = bed_source.mmap();
        let mapped = mmap.as_deref();
        if row_end > bed_source.len() {
            return Err(PipelineError::Io(format!(
                "Fatal: Read past end of .bed source '{}' for variant with global index {}. Source may be corrupt.",
                boundaries[current_fileset_idx].bed_path.display(),
                global_bim_row_index.0
            )));
        }

        terms.load(prep_result, reconciled_idx);
        for out_idx in 0..prep_result.num_people_to_score {
            let fam_idx = prep_result.output_idx_to_fam_idx[out_idx].0 as usize;
            let byte_offset = row_base
                .checked_add((fam_idx / 4) as u64)
                .ok_or_else(|| PipelineError::Compute("PLINK byte offset overflow.".to_string()))?;
            let byte = if let Some(mmap) = mapped {
                *mmap.get(byte_offset as usize).ok_or_else(|| {
                    PipelineError::Io(format!(
                        "Fatal: Read past end of .bed source '{}' for variant with global index {}.",
                        boundaries[current_fileset_idx].bed_path.display(),
                        global_bim_row_index.0
                    ))
                })?
            } else {
                bed_source.read_at(byte_offset, &mut scratch)?;
                scratch[0]
            };
            let packed = (byte >> ((fam_idx % 4) * 2)) & 0b11;
            terms.apply(
                packed,
                &mut final_scores[out_idx * stride..(out_idx + 1) * stride],
                &mut final_counts[out_idx * num_scores..(out_idx + 1) * num_scores],
            );
        }

        processed_since_update += 1;
        if processed_since_update == io::PROGRESS_UPDATE_BATCH_SIZE {
            pb.inc(processed_since_update);
            processed_since_update = 0;
        }
    }
    if processed_since_update > 0 {
        pb.inc(processed_since_update);
    }
    pb.finish_with_message("Computation complete.");
    if !prep_result.complex_rules.is_empty() {
        let resolver =
            ComplexVariantResolver::from_multi_sources(bed_sources, boundaries.to_vec())?;
        resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
    }
    Ok((final_scores, final_counts))
}

fn derive_spool_destination(base_path: &Path) -> (PathBuf, String) {
    let stem = base_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "gnomon_results".to_string());
    let path_str = base_path.to_string_lossy();
    if path_str.starts_with("gs://")
        || path_str.starts_with("http://")
        || path_str.starts_with("https://")
    {
        (Path::new(".").to_path_buf(), stem)
    } else {
        let dir = match base_path.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => Path::new(".").to_path_buf(),
        };
        (dir, stem)
    }
}

fn unique_spool_filename(stem: &str) -> String {
    let pid = process::id();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0));
    let timestamp = now.as_nanos();
    let random_component: u32 = rand::random();
    format!(
        "{}.{}.{}.{}.complex_spool.bin",
        stem, pid, timestamp, random_component
    )
}

fn create_spool_plan<'a>(
    prep_result: &'a PreparationResult,
    state: &'a mut SpoolState,
) -> Result<io::SpoolPlan<'a>, PipelineError> {
    let stride = prep_result.spool_bytes_per_variant();
    let stride_usize = usize::try_from(stride).map_err(|_| {
        PipelineError::Compute(format!(
            "spool stride of {} bytes does not fit on this platform",
            stride
        ))
    })?;
    Ok(io::SpoolPlan {
        is_complex_for_required: prep_result.required_is_complex(),
        compact_byte_index: prep_result.spool_compact_byte_index(),
        bytes_per_spooled_variant: stride,
        bytes_per_spooled_variant_usize: stride_usize,
        scratch: vec![0u8; stride_usize],
        file: &mut state.writer,
        offsets: &mut state.offsets,
        cursor: &mut state.cursor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spool is unlinked once no mapping of it is left, whether its rows resolved or not, and
    /// it is unlinked either way (#2395).
    #[cfg(target_os = "linux")]
    #[test]
    fn the_spool_is_unmapped_before_it_is_unlinked() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cohort.1.2.3.complex_spool.bin");
        for fails in [false, true] {
            std::fs::write(&path, [1u8, 2, 3, 4]).unwrap();
            let shown = std::fs::canonicalize(&path).unwrap();
            let mapped = || std::fs::read_to_string("/proc/self/maps").unwrap().contains(shown.to_str().unwrap());
            let result = with_spool(
                &path,
                4,
                |mmap| {
                    assert_eq!(&mmap[..], [1, 2, 3, 4]);
                    assert!(mapped(), "the resolver reads the spool through its mapping");
                    if fails { Err(PipelineError::Compute("resolution failed".into())) } else { Ok(()) }
                },
                |spool| {
                    assert!(!mapped(), "the spool is still mapped when it is unlinked");
                    std::fs::remove_file(spool)
                },
            );
            assert_eq!(result.is_ok(), !fails);
            assert!(!path.exists());
        }
    }

    fn memory_test_prep(people: usize, scores: usize) -> PreparationResult {
        let names: Vec<String> = (0..scores).map(|i| format!("S{i}")).collect();
        PreparationResult::new(
            crate::score::cells::ExactPlan::new(Vec::new(), &[], &[], &[0], &[], &names)
                .expect("empty plan"),
            vec![],
            vec![0],
            vec![],
            vec![],
            names,
            vec![0; scores],
            crate::score::types::PersonSubset::All,
            vec![],
            people,
            people,
            0,
            0,
            people.div_ceil(4) as u64,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            0,
            PipelineKind::SingleFile(PathBuf::from("memory-test")),
        )
    }

    #[test]
    fn zero_free_memory_never_grants_an_emergency_allocation() {
        assert_eq!(memory_budget_from_system(0, 512 << 30, 1), 1);
        assert_eq!(memory_budget_from_system(512 << 30, 0, 1), 1);
        assert_eq!(memory_budget_from_system(1000, 10000, 2), 700);
        assert_eq!(memory_budget_from_system(9000, 10000, 10), 700);
    }

    #[test]
    fn memory_floor_rejects_huge_outputs_without_allocating_them() {
        let budget = MemoryBudget {
            max_ram_bytes: 64 * 1024 * 1024,
            resident_bytes: 0,
        };
        let unopened = InputCharge::UNOPENED;
        assert!(ensure_memory_floor(&memory_test_prep(500_000, 1000), budget, unopened).is_err());
        assert!(ensure_memory_floor(&memory_test_prep(usize::MAX, 2), budget, unopened).is_err());
        assert!(ensure_memory_floor(&memory_test_prep(64, 4), budget, unopened).is_ok());
    }

    #[test]
    fn bounded_dense_batches_include_wide_weight_matrices() {
        let budget = MemoryBudget {
            max_ram_bytes: 64 * 1024 * 1024,
            resident_bytes: 0,
        };
        let wide = memory_test_prep(64, 10_000);
        let batch = bounded_dense_batch_size(&wide, budget).unwrap();
        assert!(batch < DENSE_BATCH_SIZE);
        assert!(dense_scratch_bytes(&wide, batch).unwrap() <= budget.max_ram_bytes() / 16);
        assert!(dense_scratch_bytes(&wide, batch + 1).unwrap() > budget.max_ram_bytes() / 16);
        assert_eq!(
            bounded_dense_batch_size(&memory_test_prep(64, 1), budget).unwrap(),
            DENSE_BATCH_SIZE
        );
    }

    /// Bounded consumers take what the budget leaves after the floor, the memory the process
    /// already holds beyond the plan and the input rows it maps, one private matrix, its scratch
    /// and its thread's stack each, and never more than the pool is wide. The CSR preparation
    /// left resident is charged once, by the floor.
    #[test]
    fn bounded_dense_consumers_fit_the_budget_left_after_the_floor() {
        let prep = memory_test_prep(10_000, 16);
        let csr = prep.csr_heap_bytes();
        for mebibytes in [1usize, 4, 16, 64, 256, 1024, 1 << 20] {
            for (held, mapped) in [
                (0usize, 0usize),
                (3 << 20, 0),
                (40 << 20, 0),
                (3 << 20, 5 << 20),
            ] {
                let budget = MemoryBudget {
                    max_ram_bytes: mebibytes << 20,
                    resident_bytes: csr + held,
                };
                let input = InputCharge {
                    mapped,
                    read_buffers: 0,
                    prefetch: true,
                };
                let consumers = bounded_dense_consumers(&prep, budget, input).unwrap();
                let private = result_bytes(&prep).unwrap()
                    + consumer_scratch_bytes(
                        &prep,
                        bounded_dense_batch_size(&prep, budget).unwrap(),
                    )
                    .unwrap()
                    + thread_stack_bytes();
                let charged = memory_floor_bytes(&prep, budget, input).unwrap() + held + mapped;
                let at = format!("{mebibytes} MiB, {held} held, {mapped} mapped");
                assert!(consumers >= 1 && consumers <= worker_ceiling(), "{at}");
                if consumers > 1 {
                    assert!(
                        charged + (consumers - 1) * private <= budget.max_ram_bytes(),
                        "{at}"
                    );
                }
                if consumers < worker_ceiling() {
                    assert!(
                        charged + consumers * private > budget.max_ram_bytes(),
                        "{at}"
                    );
                }
                if held == 0 {
                    let below = MemoryBudget {
                        resident_bytes: csr / 2,
                        ..budget
                    };
                    assert_eq!(
                        bounded_dense_consumers(&prep, below, input).unwrap(),
                        consumers,
                        "{at}"
                    );
                }
            }
        }
    }

    /// ensure_memory_floor charges what the bounded plan charges before any consumer: the floor,
    /// the memory held beyond the plan and the input rows read through a memory map. One byte
    /// below that sum it refuses and names each part, where the floor alone fit; at the sum and
    /// one byte above, it admits the run.
    #[test]
    fn memory_floor_refuses_one_byte_below_the_floor_held_and_mapped_memory() {
        let prep = memory_test_prep(10_000, 16);
        let csr = prep.csr_heap_bytes();
        // Held memory this large puts the boundary where the floor no longer grows with the
        // budget (I/O buffers, batch size and prefetch are all at their caps).
        let held = 64usize << 30;
        let mapped = 16usize << 20;
        let input = InputCharge {
            mapped,
            read_buffers: 0,
            prefetch: true,
        };
        let at = |max_ram_bytes| MemoryBudget {
            max_ram_bytes,
            resident_bytes: csr + held,
        };
        // The least budget that holds its own floor, the held memory and the mapped rows.
        let mut boundary = held + mapped;
        for _ in 0..64 {
            let next = memory_floor_bytes(&prep, at(boundary), input).unwrap() + held + mapped;
            if next == boundary {
                break;
            }
            boundary = next;
        }
        assert_eq!(
            memory_floor_bytes(&prep, at(boundary), input).unwrap() + held + mapped,
            boundary
        );
        assert!(ensure_memory_floor(&prep, at(boundary), input).is_ok());
        assert!(ensure_memory_floor(&prep, at(boundary + 1), input).is_ok());

        let below = at(boundary - 1);
        assert!(memory_floor_bytes(&prep, below, input).unwrap() <= below.max_ram_bytes());
        match ensure_memory_floor(&prep, below, input) {
            Err(PipelineError::Compute(refusal)) => {
                assert!(refusal.contains(&format_bytes(held)), "{refusal}");
                assert!(refusal.contains(&format_bytes(mapped)), "{refusal}");
                assert!(
                    refusal.contains(&format_bytes(below.max_ram_bytes())),
                    "{refusal}"
                );
            }
            _ => panic!(
                "a budget one byte below the floor, held memory and mapped rows was admitted"
            ),
        }
        // A resident reading of only the CSR holds nothing beyond the plan.
        let csr_only = MemoryBudget {
            resident_bytes: csr,
            ..below
        };
        assert!(ensure_memory_floor(&prep, csr_only, input).is_ok());
        // A run that maps no rows does not need their share.
        assert!(ensure_memory_floor(&prep, below, InputCharge::UNOPENED).is_ok());
    }

    /// The floor charges the CSR at the bytes its vectors hold: 13 an entry here (an i64 weight,
    /// its flags and a u32 column) where the i128 worst case charges 21, and a resident reading
    /// of exactly those bytes holds nothing beyond the plan.
    #[test]
    fn the_csr_is_charged_at_the_bytes_it_holds() {
        let (prep, ..) = bound_panel(64, 4096, [(1 << 53) - 1, 3]);
        let entries = prep.sparse_score_columns().len();
        let offsets = prep.sparse_row_offsets().len() * std::mem::size_of::<u64>();
        let entry = std::mem::size_of::<i64>() + 1 + std::mem::size_of::<u32>();
        let csr = prep.csr_heap_bytes();
        assert!(csr >= entries * entry + offsets, "{csr}");
        assert!(csr <= entries * (entry + 1) + offsets + 4096, "{csr}");
        assert!(csr < csr_bytes(&prep).unwrap());
        let at = |resident_bytes| MemoryBudget {
            max_ram_bytes: 64 << 20,
            resident_bytes,
        };
        assert_eq!(held_beyond_plan(&prep, at(csr)), 0);
        assert_eq!(held_beyond_plan(&prep, at(csr + 1)), 1);
    }

    /// The local prefetch budget is charged only while a source may still take it: before the
    /// sources open, and while one serves neither a map nor a read plan. A source reading through
    /// a plan is charged its reader's own window instead.
    #[test]
    fn the_prefetch_window_is_charged_only_while_a_source_may_take_one() {
        // 16,384 people pack into 4,096-byte rows; 4,096 needed rows take a local prefetch budget.
        let mut prep = memory_test_prep(16_384, 1);
        prep.required_bim_indices = (0..4096).map(crate::score::types::BimRowIndex).collect();
        let budget = MemoryBudget {
            max_ram_bytes: 1 << 30,
            resident_bytes: 0,
        };
        let window = io::local_prefetch_budget(&prep, budget);
        assert!(window > 0);
        let mapped = InputCharge {
            mapped: 0,
            read_buffers: 0,
            prefetch: false,
        };
        assert_eq!(
            memory_floor_bytes(&prep, budget, InputCharge::UNOPENED).unwrap(),
            memory_floor_bytes(&prep, budget, mapped).unwrap() + window
        );
        // A source reading through a plan is charged its reader's own window, here past the
        // budget's 16 MiB cap, in place of the budget.
        let planned = InputCharge {
            mapped: 0,
            read_buffers: 48 << 20,
            prefetch: false,
        };
        assert_eq!(
            memory_floor_bytes(&prep, budget, planned).unwrap(),
            memory_floor_bytes(&prep, budget, mapped).unwrap() + (48 << 20)
        );
    }

    /// A memory map is charged the rows a run needs, at most the mapped sources' length, and a
    /// source that does not serve reads from a map is charged nothing.
    #[test]
    fn mapped_row_bytes_charge_needed_rows_up_to_the_mapped_length() {
        // Eight people pack into two bytes a row; the file holds the header and three rows.
        let mut prep = memory_test_prep(8, 1);
        let file = tempfile::Builder::new().suffix(".bed").tempfile().unwrap();
        std::fs::write(file.path(), [0x6c, 0x1b, 0x01, 0, 0, 0, 0, 0, 0]).unwrap();
        let mapped = io::open_bed_source(file.path(), None).unwrap();
        assert!(mapped.mapped_rows().is_some());
        prep.required_bim_indices = (0..2).map(crate::score::types::BimRowIndex).collect();
        assert_eq!(mapped_row_bytes(&prep, std::slice::from_ref(&mapped)), 4);
        prep.required_bim_indices = (0..10).map(crate::score::types::BimRowIndex).collect();
        assert_eq!(mapped_row_bytes(&prep, std::slice::from_ref(&mapped)), 9);

        struct Unmapped;
        impl io::ByteRangeSource for Unmapped {
            fn len(&self) -> u64 {
                9
            }

            fn read_at(&self, _offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
                dst.fill(0);
                Ok(())
            }
        }
        let unmapped = io::BedSource::from_byte_source(Arc::new(Unmapped));
        assert!(unmapped.mapped_rows().is_none());
        assert_eq!(mapped_row_bytes(&prep, std::slice::from_ref(&unmapped)), 0);

        // Only a source that serves no map may take a prefetch window.
        let of = InputCharge::of(&prep, std::slice::from_ref(&mapped));
        assert_eq!((of.mapped, of.prefetch), (9, false));
        let of = InputCharge::of(&prep, &[mapped, unmapped]);
        assert_eq!((of.mapped, of.prefetch), (9, true));
    }

    /// Two scores over `rows` rows, every row weighing `unit[score]`: integers below 2^53, so each
    /// weight's shortest decimal is the integer itself and the oracle's arithmetic is the plan's.
    /// Person 0 carries two effect alleles everywhere, so its totals are the largest the plan
    /// bounds. Gives the plan, the packed rows, and each person's correctly rounded totals and
    /// missing counts.
    fn bound_panel(
        people: usize,
        rows: usize,
        unit: [i128; 2],
    ) -> (PreparationResult, Vec<u8>, Vec<u64>, Vec<u32>) {
        use crate::score::types::{OriginalPersonIndex, OutputPersonIndex, PersonSubset};
        let scores = 2;
        let weights: Vec<f64> = (0..rows).flat_map(|_| unit.map(|w| w as f64)).collect();
        let columns: Vec<u32> = (0..rows).flat_map(|_| [0u32, 1]).collect();
        let offsets: Vec<u64> = (0..=rows as u64).map(|row| 2 * row).collect();
        let names: Vec<String> = (0..scores).map(|s| format!("S{s}")).collect();
        let corrections = vec![0.0; weights.len()];
        let exact = crate::score::cells::ExactPlan::new(
            weights,
            &corrections,
            &columns,
            &offsets,
            &[],
            &names,
        )
        .expect("exact plan");
        let row_bytes = people.div_ceil(4);
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut data = vec![0u8; rows * row_bytes];
        let (mut sums, mut missing) = (vec![0i128; people * scores], vec![0u32; people * scores]);
        for row in 0..rows {
            for person in 0..people {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let code = if person == 0 {
                    3
                } else {
                    (state >> 32) as u8 & 3
                };
                data[row * row_bytes + person / 4] |= code << (2 * (person % 4));
                for (score, &weight) in unit.iter().enumerate() {
                    let cell = person * scores + score;
                    match code {
                        1 => missing[cell] += 1,
                        dose => sums[cell] += i128::from([0, 0, 1, 2][dose as usize]) * weight,
                    }
                }
            }
        }
        let rounded = sums
            .iter()
            .map(|sum| sum.to_string().parse::<f64>().unwrap().to_bits())
            .collect();
        let prep = PreparationResult::new(
            exact,
            columns,
            offsets,
            (0..rows as u64).map(BimRowIndex).collect(),
            Vec::new(),
            names,
            vec![rows as u32; scores],
            PersonSubset::All,
            (0..people).map(|p| format!("P{p}")).collect(),
            people,
            people,
            rows as u64,
            rows,
            row_bytes as u64,
            (0..people)
                .map(|p| Some(OutputPersonIndex(p as u32)))
                .collect(),
            (0..people).map(|p| OriginalPersonIndex(p as u32)).collect(),
            vec![0; rows],
            Vec::new(),
            Vec::new(),
            0,
            PipelineKind::SingleFile(PathBuf::from("panel.bed")),
        );
        (prep, data, rounded, missing)
    }

    fn rounded_totals(prep: &PreparationResult, cells: &[i64]) -> Vec<u64> {
        let (scores, stride) = (prep.score_names.len(), prep.exact().stride());
        (0..cells.len() / stride * scores)
            .map(|cell| {
                let person = cell / scores;
                prep.exact()
                    .sum(
                        cell % scores,
                        &cells[person * stride..(person + 1) * stride],
                    )
                    .to_bits()
            })
            .collect()
    }

    /// Adds a panel's rows serially, and again split at `cuts`: the first range into the shared
    /// cells, and each other range into a private consumer's cells merged into them. The merged
    /// cells must equal the serial ones bit for bit and give the exact oracle's totals and counts.
    fn check_private_merge(rows: usize, unit: [i128; 2], cuts: &[usize]) {
        let (prep, data, want_totals, want_missing) = bound_panel(37, rows, unit);
        let layout = PersonLayout::new(&prep);
        let row_bytes = prep.bytes_per_variant as usize;
        let indices: Vec<ReconciledVariantIndex> =
            (0..rows as u32).map(ReconciledVariantIndex).collect();
        let add = |range: std::ops::Range<usize>, cells: &mut (Vec<i64>, Vec<u32>)| {
            let mut scratch = DenseScratch::default();
            for start in range.clone().step_by(3) {
                let end = (start + 3).min(range.end);
                batch::run_dense_batch(
                    &data[start * row_bytes..end * row_bytes],
                    &indices[start..end],
                    &prep,
                    &layout,
                    &mut scratch,
                    &mut cells.0,
                    &mut cells.1,
                )
                .unwrap();
            }
        };
        let mut serial = initialize_cells(&prep).unwrap();
        add(0..rows, &mut serial);
        let shared = Mutex::new(initialize_cells(&prep).unwrap());
        let ends: Vec<usize> = cuts.iter().copied().chain([rows]).collect();
        add(0..ends[0], &mut *shared.lock().unwrap());
        for range in ends.windows(2) {
            let mut private = initialize_cells(&prep).unwrap();
            add(range[0]..range[1], &mut private);
            merge_bounded_cells(&shared, private.0, private.1).unwrap();
        }
        let merged = shared.into_inner().unwrap();
        assert_eq!(merged, serial);
        assert_eq!(rounded_totals(&prep, &merged.0), want_totals);
        assert_eq!(merged.1, want_missing);
    }

    /// At the edge of one limb: 512 rows of 2^53 - 1 sum, twice, to 2^63 - 1,024 in the first
    /// score, and rows of 2^53 take the second just past it into two limbs.
    #[test]
    fn private_consumers_merge_exactly_at_the_plan_bound() {
        check_private_merge(512, [(1 << 53) - 1, 1 << 53], &[128, 300]);
    }

    /// A stream whose total passes 2^63 many times over: 4,096 rows of 2^53 - 1 give person 0
    /// about 2^66, which the plan holds in two carry-free limbs, and each of four private
    /// consumers alone adds about 2^64 before it merges.
    #[test]
    fn private_consumers_merge_exactly_past_two_to_the_63() {
        check_private_merge(4096, [(1 << 53) - 1, 3], &[512, 1536, 2560, 3584]);
    }

    /// The bounded dense consumer, running every consumer an ample budget allows, reads a whole
    /// stream sent one row at a time, whose totals pass 2^63, and gives the exact totals.
    #[test]
    fn bounded_dense_consumers_score_a_stream_exactly() {
        let (people, rows) = (37, 4096);
        let (prep, data, want_totals, want_missing) = bound_panel(people, rows, [(1 << 53) - 1, 3]);
        let row_bytes = prep.bytes_per_variant as usize;
        let prep = Arc::new(prep);
        let budget = MemoryBudget {
            max_ram_bytes: 1 << 40,
            resident_bytes: 0,
        };
        assert_eq!(
            bounded_dense_consumers(&prep, budget, InputCharge::UNOPENED).unwrap(),
            worker_ceiling()
        );
        let context = PipelineContext::with_budget(Arc::clone(&prep), budget, None);
        let accumulator = Arc::new(Mutex::new(initialize_cells(&prep).unwrap()));
        let pool = Arc::new(io::RowBufferPool::new(rows, 0));
        let (tx, rx) = bounded(1);
        let result = thread::scope(|s| {
            s.spawn(move || {
                for row in 0..rows {
                    let item = WorkItem {
                        data: data[row * row_bytes..(row + 1) * row_bytes].to_vec(),
                        reconciled_variant_index: ReconciledVariantIndex(row as u32),
                    };
                    if tx.send(Ok(item)).is_err() {
                        break;
                    }
                }
            });
            process_dense_stream_bounded(
                rx,
                &context,
                pool,
                Arc::clone(&accumulator),
                InputCharge::UNOPENED,
            )
        });
        assert!(result.is_ok());
        let cells = accumulator.lock().unwrap();
        assert_eq!(rounded_totals(&prep, &cells.0), want_totals);
        assert_eq!(cells.1, want_missing);
    }

    #[test]
    fn small_cohort_complex_scoring_matches_manual_dosages_across_filesets() {
        let dir = tempfile::tempdir().unwrap();
        let fam = (0..7)
            .map(|i| format!("F I{i} 0 0 0 -9\n"))
            .collect::<String>();
        let rows = [
            "1 a 0 100 A G\n",
            "1 b 0 150 C T\n",
            "1 c 0 200 A C\n",
            "1 d 0 200 G C\n",
        ];
        let calls: [[u8; 7]; 4] = [
            [0, 1, 2, 3, 3, 2, 0],
            [3, 2, 1, 0, 1, 0, 2],
            [2, 3, 0, 1, 2, 1, 3],
            [0; 7],
        ];
        let weights = dir.path().join("weights.tsv");
        fs::write(&weights, "variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.25\n1:150\tC\tT\t0.5\n1:200\tC\tA\t0.125\n").unwrap();
        let keep = dir.path().join("keep.txt");
        fs::write(&keep, "I1\nI4\nI6\n").unwrap();
        for split in [false, true] {
            let mut prefixes = Vec::new();
            let ranges = if split { vec![0..2, 2..4] } else { vec![0..4] };
            for (part, range) in ranges.into_iter().enumerate() {
                let prefix = dir.path().join(format!("panel-{split}-{part}"));
                fs::write(prefix.with_extension("fam"), &fam).unwrap();
                fs::write(prefix.with_extension("bim"), rows[range.clone()].concat()).unwrap();
                let mut bed = vec![0x6c, 0x1b, 0x01];
                for row in &calls[range] {
                    for chunk in row.chunks(4) {
                        bed.push(
                            chunk
                                .iter()
                                .enumerate()
                                .fold(0, |byte, (i, call)| byte | (call << (2 * i))),
                        );
                    }
                }
                fs::write(prefix.with_extension("bed"), bed).unwrap();
                prefixes.push(prefix);
            }
            for subset in [None, Some(keep.as_path())] {
                let prep = Arc::new(
                    crate::score::prepare::prepare_for_computation(
                        &prefixes,
                        std::slice::from_ref(&weights),
                        subset,
                        None,
                    )
                    .unwrap(),
                );
                assert_eq!(prep.complex_rules.len(), 1);
                let context = PipelineContext::new(Arc::clone(&prep));
                let (scores, missing) = run(&context).unwrap();
                for (out, fam) in prep.output_idx_to_fam_idx.iter().enumerate() {
                    let mut expected = 0.0;
                    let mut expected_missing = 0;
                    for (row, weight) in [0.25, 0.5, 0.125].into_iter().enumerate() {
                        let packed = calls[row][fam.0 as usize];
                        if packed == 1 {
                            expected_missing += 1;
                        } else {
                            let dosage = match packed {
                                0 => 0.0,
                                2 => 1.0,
                                3 => 2.0,
                                _ => unreachable!(),
                            };
                            expected += weight * if row == 1 { 2.0 - dosage } else { dosage };
                        }
                    }
                    let stride = prep.exact().stride();
                    assert_eq!(
                        prep.exact()
                            .sum(0, &scores[out * stride..(out + 1) * stride]),
                        expected,
                        "split={split}, person={fam:?}"
                    );
                    assert_eq!(missing[out], expected_missing);
                }
            }
        }
    }

    #[test]
    fn progress_monitor_stops_when_pipeline_fails_before_total() {
        let (finished_tx, finished_rx) = bounded(1);
        let count = Arc::new(AtomicU64::new(1));
        let observed_count = Arc::clone(&count);
        let worker = thread::spawn(move || {
            let result = thread::scope(|scope| {
                let (progress_lifetime, completion) = bounded(0);
                scope.spawn(move || {
                    update_pipeline_progress(count, ProgressBar::hidden(), 2, completion)
                });
                let failure = Err::<(), _>("producer failed");
                failure?;
                drop(progress_lifetime);
                Ok(())
            });
            finished_tx.send(result).expect("report completion");
        });
        assert_eq!(
            finished_rx
                .recv_timeout(Duration::from_secs(2))
                .expect("progress thread must not hold the failed pipeline open"),
            Err("producer failed")
        );
        worker.join().expect("worker");
        assert_eq!(observed_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn derive_spool_destination_remote_paths_default_to_current_dir() {
        let (dir, stem) = derive_spool_destination(Path::new("gs://bucket/data/sample.bed"));
        assert_eq!(dir, Path::new("."));
        assert_eq!(stem, "sample");

        let (dir_http, stem_http) =
            derive_spool_destination(Path::new("https://example.com/study/run"));
        assert_eq!(dir_http, Path::new("."));
        assert_eq!(stem_http, "run");
    }

    #[test]
    fn derive_spool_destination_local_paths_use_parent_directory() {
        let path = Path::new("/tmp/project/cohort1.bed");
        let (dir, stem) = derive_spool_destination(path);
        assert_eq!(dir, Path::new("/tmp/project"));
        assert_eq!(stem, "cohort1");
    }

    #[test]
    fn channel_batcher_waits_for_a_full_dense_batch() {
        let (tx, rx) = bounded(1);
        tx.send(Ok(1u8)).unwrap();
        let producer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(10));
            tx.send(Ok(2u8)).unwrap();
            tx.send(Ok(3u8)).unwrap();
        });

        let mut batches = ChannelBatcher::new(rx, 3);
        assert_eq!(batches.next().unwrap().unwrap(), vec![1, 2, 3]);
        assert!(batches.next().is_none());
        producer.join().unwrap();
    }
}

/// A RAII guard that ensures a byte buffer is automatically returned to the shared
/// buffer pool when it goes out of scope. This is critical for preventing resource
// leaks in the consumer streams, especially when errors occur.
struct BufferGuard<'a> {
    /// The buffer being managed. Wrapped in an `Option` to allow ownership to be
    /// taken in the `drop` implementation.
    buffer: Option<Vec<u8>>,
    /// A reference to the shared pool where the buffer will be returned.
    pool: &'a io::RowBufferPool,
}

impl<'a> Drop for BufferGuard<'a> {
    fn drop(&mut self) {
        // When the guard is dropped, it returns its buffer to the pool at full length, so
        // the producer reuses it without a zero fill.
        if let Some(buf) = self.buffer.take() {
            let _ = self.pool.push(buf);
        }
    }
}

/// A general-purpose RAII guard that executes a closure when it goes out of scope.
///
/// This utility is crucial for ensuring that a specific action, such as releasing a
/// resource or signaling completion, is performed regardless of how a scope is exited
/// (e.g., normal completion, early return, or panic). It holds an optional closure,
/// which is taken and executed in the `drop` implementation, guaranteeing the
/// action runs exactly once.
pub struct ScopeGuard<F: FnOnce()> {
    /// The closure to execute on drop. `Option` is used to allow the closure
    /// to be taken and called, preventing multiple executions.
    action: Option<F>,
}

impl<F: FnOnce()> ScopeGuard<F> {
    /// Creates a new `ScopeGuard` with the given action.
    ///
    /// The action will be executed when the returned guard is dropped.
    #[inline(always)]
    pub fn new(action: F) -> Self {
        Self {
            action: Some(action),
        }
    }
}

impl<F: FnOnce()> Drop for ScopeGuard<F> {
    /// Executes the stored action when the guard goes out of scope.
    ///
    /// This method is called automatically by the Rust compiler. It takes the
    /// action out of the `Option`, ensuring it can only be run once, and then
    /// executes it.
    #[inline(always)]
    fn drop(&mut self) {
        if let Some(action) = self.action.take() {
            action();
        }
    }
}

type ConsumerResult = Result<(Vec<i64>, Vec<u32>), PipelineError>;

#[inline]
fn reconciled_index_from_usize(i: usize) -> Result<ReconciledVariantIndex, PipelineError> {
    let idx = u32::try_from(i).map_err(|_| {
        PipelineError::Compute(format!(
            "Reconciled variant index {i} exceeds u32::MAX; too many variants in one run."
        ))
    })?;
    Ok(ReconciledVariantIndex(idx))
}

#[inline]
fn checked_result_size(prep_result: &PreparationResult) -> Result<usize, PipelineError> {
    prep_result
        .num_people_to_score
        .checked_mul(prep_result.score_names.len())
        .ok_or_else(|| {
            PipelineError::Compute(format!(
                "Result size overflow: num_people_to_score={} * num_scores={}",
                prep_result.num_people_to_score,
                prep_result.score_names.len()
            ))
        })
}

#[inline]
fn checked_cells_size(prep_result: &PreparationResult) -> Result<usize, PipelineError> {
    prep_result
        .num_people_to_score
        .checked_mul(prep_result.exact().stride())
        .ok_or_else(|| {
            PipelineError::Compute(format!(
                "Cell matrix size overflow: num_people_to_score={} * stride={}",
                prep_result.num_people_to_score,
                prep_result.exact().stride()
            ))
        })
}

/// Zeroed exact cells (people × stride lanes) and missing counts (people × scores).
#[inline]
fn initialize_cells(
    prep_result: &PreparationResult,
) -> Result<(Vec<i64>, Vec<u32>), PipelineError> {
    let cells_size = checked_cells_size(prep_result)?;
    let result_size = checked_result_size(prep_result)?;
    let mut final_scores = Vec::new();
    final_scores.try_reserve_exact(cells_size).map_err(|e| {
        PipelineError::Compute(format!(
            "Failed to reserve final score matrix ({cells_size} lanes): {e}"
        ))
    })?;
    final_scores.resize(cells_size, 0i64);
    let mut final_counts = Vec::new();
    final_counts.try_reserve_exact(result_size).map_err(|e| {
        PipelineError::Compute(format!(
            "Failed to reserve final missing-count matrix ({result_size} cells): {e}"
        ))
    })?;
    final_counts.resize(result_size, 0u32);
    Ok((final_scores, final_counts))
}

/// The widest any worker set may be: the global rayon pool's size, which
/// follows `RAYON_NUM_THREADS` and otherwise the CPUs this process may run on
/// (its affinity mask and cgroup quota). The visible CPU count is neither.
#[inline]
fn worker_ceiling() -> usize {
    rayon::current_num_threads().max(1)
}

#[inline]
fn choose_consumer_threads(bytes_per_accumulator: usize, memory_budget: MemoryBudget) -> usize {
    let cpu_cap = worker_ceiling();

    if bytes_per_accumulator == 0 {
        return cpu_cap;
    }

    // Sparse and dense consumers run concurrently. Keep the combined thread-local
    // accumulator footprint to roughly half the user/system RAM budget.
    let thread_accumulator_budget = (memory_budget.max_ram_bytes() / 2).max(bytes_per_accumulator);
    let total_accumulators = (thread_accumulator_budget / bytes_per_accumulator).max(1);
    let by_mem = (total_accumulators / 2).max(1);

    by_mem.min(cpu_cap).max(1)
}

/// A contention-free consumer for the sparse variant stream, using Rayon's
/// fold/reduce pattern for maximum parallelism with no locks.
fn process_sparse_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<io::RowBufferPool>,
) -> ConsumerResult {
    // However this returns or unwinds, stop the producer taking buffers, so that it never
    // parks on a pool nothing will refill.
    let _stop_producer = ScopeGuard::new(|| buffer_pool.close());
    let prep_result = &context.prep_result;
    let result_size = checked_result_size(prep_result)?;
    let cells_size = checked_cells_size(prep_result)?;
    let consumer_threads = choose_consumer_threads(result_bytes(prep_result)?, context.memory_budget);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(consumer_threads)
        .build()
        .map_err(|e| {
            PipelineError::Compute(format!("Failed to build sparse consumer pool: {e}"))
        })?;

    // The fold/reduce pattern creates thread-local accumulators for scores and counts.
    // After processing a work item, its data buffer is immediately returned to the
    // shared pool, creating a true, continuous recycling system.
    let final_result = pool.install(|| {
        rx.into_iter() // Convert the channel to a blocking iterator.
            .par_bridge() // Bridge it to a Rayon parallel iterator.
            .try_fold(
                // Each thread gets its own accumulator and term scratch.
                || (vec![0i64; cells_size], vec![0u32; result_size], VariantTerms::default()),
                |mut acc, work_result| {
                    // The work_item and its buffer are processed within this scope.
                    // The `_guard` ensures the buffer is returned to the pool when this
                    // scope ends, whether by success or by `?` propagating an error.
                    {
                        let work_item = work_result?;
                        let guard = BufferGuard {
                            buffer: Some(work_item.data),
                            pool: &buffer_pool,
                        };

                        batch::run_variant_major_path(
                            // The guard holds the buffer, so we borrow it from there.
                            guard.buffer.as_ref().unwrap(),
                            prep_result,
                            &context.person_layout,
                            &mut acc.2,
                            &mut acc.0,
                            &mut acc.1,
                            work_item.reconciled_variant_index,
                        )?;
                    }
                    Ok::<_, PipelineError>(acc)
                },
            )
            .try_reduce(
                // Identity for the reduction.
                || (vec![0i64; cells_size], vec![0u32; result_size], VariantTerms::default()),
                |mut a, b| {
                    // Combine accumulators from two threads in parallel.
                    a.0.par_iter_mut()
                        .zip(b.0)
                        .for_each(|(v_a, v_b)| *v_a = v_a.wrapping_add(v_b));
                    a.1.par_iter_mut()
                        .zip(b.1)
                        .for_each(|(v_a, v_b)| *v_a += v_b);
                    Ok(a)
                },
            )
    })?;

    // `try_reduce` returns `Result<(scores, counts), PipelineError>`.
    // The `?` operator has already unwrapped the Result, leaving just the tuple.
    // With an identity function, try_reduce handles empty streams by returning the identity.
    let (cells, counts, _) = final_result;
    Ok((cells, counts))
}

fn process_sparse_stream_bounded(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<io::RowBufferPool>,
    accumulator: Arc<Mutex<(Vec<i64>, Vec<u32>)>>,
) -> ConsumerResult {
    let _stop_producer = ScopeGuard::new(|| buffer_pool.close());
    let prep_result = &context.prep_result;
    let mut terms = VariantTerms::default();
    for work_result in rx {
        let work_item = work_result?;
        let guard = BufferGuard {
            buffer: Some(work_item.data),
            pool: &buffer_pool,
        };
        {
            let mut locked = accumulator.lock().map_err(|_| {
                PipelineError::Compute("Bounded accumulator lock was poisoned.".to_string())
            })?;
            let (scores, counts) = &mut *locked;
            batch::run_variant_major_path(
                guard.buffer.as_ref().unwrap(),
                prep_result,
                &context.person_layout,
                &mut terms,
                scores,
                counts,
                work_item.reconciled_variant_index,
            )?;
        }
    }
    Ok((Vec::new(), Vec::new()))
}

/// A contention-free consumer for the dense variant stream. It uses a custom
/// batching iterator to group items, which are then processed in parallel by Rayon.
/// This implementation allows I/O and computation to run concurrently.
fn process_dense_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<io::RowBufferPool>,
) -> ConsumerResult {
    let _stop_producer = ScopeGuard::new(|| buffer_pool.close());
    let prep_result = &context.prep_result;
    let result_size = checked_result_size(prep_result)?;
    let cells_size = checked_cells_size(prep_result)?;
    let consumer_threads = choose_consumer_threads(result_bytes(prep_result)?, context.memory_budget);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(consumer_threads)
        .build()
        .map_err(|e| PipelineError::Compute(format!("Failed to build dense consumer pool: {e}")))?;

    // Instantiate our new Send-compatible batching iterator.
    let batch_iterator = ChannelBatcher::new(rx, DENSE_BATCH_SIZE);

    // Use the exact same fold/reduce pattern as the sparse stream, but on batches.
    let (cells, counts, _, _, _) = pool.install(|| {
        batch_iterator
            .par_bridge()
            .try_fold(
                || {
                    // Per-thread accumulator initializer
                    (
                        vec![0i64; cells_size],
                        vec![0u32; result_size],
                        Vec::with_capacity(
                            DENSE_BATCH_SIZE * (prep_result.bytes_per_variant as usize),
                        ),
                        Vec::<ReconciledVariantIndex>::with_capacity(DENSE_BATCH_SIZE),
                        DenseScratch::default(),
                    )
                },
                |mut acc, batch_result| {
                    // The `?` operator handles propagating errors from the channel.
                    let batch = batch_result?;
                    if batch.is_empty() {
                        return Ok(acc);
                    }
                    acc.3.clear();
                    acc.3
                        .extend(batch.iter().map(|wi| wi.reconciled_variant_index));
                    acc.2.clear();
                    // The kernel reads the concatenated copy, so each source buffer can return
                    // to the producer immediately after copying, letting I/O overlap compute.
                    for wi in batch {
                        acc.2.extend_from_slice(&wi.data);
                        drop(BufferGuard {
                            buffer: Some(wi.data),
                            pool: &buffer_pool,
                        });
                    }
                    batch::run_dense_batch(
                        &acc.2,
                        &acc.3,
                        prep_result,
                        &context.person_layout,
                        &mut acc.4,
                        &mut acc.0,
                        &mut acc.1,
                    )?;
                    Ok::<_, PipelineError>(acc)
                },
            )
            .try_reduce(
                || (Vec::new(), Vec::new(), Vec::new(), Vec::new(), DenseScratch::default()),
                |mut a, b| {
                    if a.0.is_empty() {
                        return Ok(b);
                    }
                    if !b.0.is_empty() {
                        a.0.par_iter_mut()
                            .zip(b.0)
                            .for_each(|(v_a, v_b)| *v_a = v_a.wrapping_add(v_b));
                        a.1.par_iter_mut()
                            .zip(b.1)
                            .for_each(|(v_a, v_b)| *v_a += v_b);
                    }
                    Ok(a)
                },
            )
    })?;
    // An empty stream reduces to the empty identity.
    if cells.is_empty() {
        return initialize_cells(prep_result);
    }
    Ok((cells, counts))
}

fn process_dense_stream_bounded(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<io::RowBufferPool>,
    accumulator: Arc<Mutex<(Vec<i64>, Vec<u32>)>>,
    input: InputCharge,
) -> ConsumerResult {
    let _stop_producer = ScopeGuard::new(|| buffer_pool.close());
    let prep_result = &context.prep_result;
    let batch_size = bounded_dense_batch_size(prep_result, context.memory_budget)?;
    let consumers = bounded_dense_consumers(prep_result, context.memory_budget, input)?;
    // The first consumer adds into the shared cells under the lock; every other one adds into
    // private cells the budget has room for and merges them when the stream ends.
    thread::scope(|s| {
        let private: Vec<_> = (1..consumers)
            .map(|_| {
                let rx = rx.clone();
                let buffer_pool = Arc::clone(&buffer_pool);
                let accumulator = Arc::clone(&accumulator);
                s.spawn(move || {
                    let _stop_producer = ScopeGuard::new(|| buffer_pool.close());
                    let mut cells = DenseCells::Private(None);
                    consume_dense_batches(rx, context, &buffer_pool, batch_size, &mut cells)?;
                    match cells {
                        DenseCells::Private(Some((scores, counts))) => {
                            merge_bounded_cells(&accumulator, scores, counts)
                        }
                        _ => Ok(()),
                    }
                })
            })
            .collect();
        let first = consume_dense_batches(
            rx,
            context,
            &buffer_pool,
            batch_size,
            &mut DenseCells::Shared(&accumulator),
        );
        // A stopped consumer closes the pool, so the producer and every other consumer end too.
        if first.is_err() {
            buffer_pool.close();
        }
        private.into_iter().fold(first, |result, handle| {
            let joined = handle
                .join()
                .map_err(|_| PipelineError::Compute("A dense consumer panicked.".to_string()))
                .and_then(|consumed| consumed);
            result.and(joined)
        })
    })?;
    Ok((Vec::new(), Vec::new()))
}

/// The bounded accumulator's memory plan: the floor every bounded run needs (of it, what the
/// score plan's tables hold), the memory this process already holds beyond the plan that floor
/// counts, the input rows it reads through memory maps, one private dense consumer's cells, batch
/// scratch and thread stack, and how many dense consumers the budget holds.
struct BoundedPlan {
    floor: usize,
    csr: usize,
    resident: usize,
    mapped: usize,
    private: usize,
    consumers: usize,
}

fn bounded_plan(
    prep: &PreparationResult,
    budget: MemoryBudget,
    input: InputCharge,
) -> Result<BoundedPlan, PipelineError> {
    let floor = memory_floor_bytes(prep, budget, input)?;
    let resident = held_beyond_plan(prep, budget);
    let private = result_bytes(prep)?
        .checked_add(consumer_scratch_bytes(
            prep,
            bounded_dense_batch_size(prep, budget)?,
        )?)
        .and_then(|v| v.checked_add(thread_stack_bytes()))
        .ok_or_else(|| PipelineError::Compute("Dense consumer size overflow.".into()))?;
    let spare = budget
        .max_ram_bytes()
        .saturating_sub(floor.saturating_add(resident).saturating_add(input.mapped));
    Ok(BoundedPlan {
        floor,
        csr: prep.csr_heap_bytes(),
        resident,
        mapped: input.mapped,
        private,
        consumers: (1 + spare / private.max(1)).min(worker_ceiling()),
    })
}

/// Dense consumers the bounded accumulator runs. The first adds into the shared cells; each
/// other one holds a private cell matrix, its batch scratch and its own thread's stack, paid from
/// what the budget leaves after the floor, the memory this process already holds and the input
/// rows it maps.
fn bounded_dense_consumers(
    prep: &PreparationResult,
    budget: MemoryBudget,
    input: InputCharge,
) -> Result<usize, PipelineError> {
    Ok(bounded_plan(prep, budget, input)?.consumers)
}

/// Says how a bounded run accumulates: one consumer on the shared cells, how many beside it the
/// budget gives private cells, and what the plan charged against the budget.
fn announce_bounded_accumulator(
    prep: &PreparationResult,
    budget: MemoryBudget,
    input: InputCharge,
) -> Result<(), PipelineError> {
    let plan = bounded_plan(prep, budget, input)?;
    match plan.consumers {
        1 => eprintln!(
            "> Using bounded RAM accumulator: one shared exact cell and count matrix, no per-thread full-matrix copies."
        ),
        consumers => eprintln!(
            "> Using bounded RAM accumulator: {consumers} dense consumers, the first adding into one shared exact cell and count matrix and each other into a private matrix merged when the stream ends."
        ),
    }
    eprintln!(
        "> Bounded plan: {} budget, {} floor ({} of it the score plan's tables), {} already resident beyond the plan, {} of input rows read through a memory map, {} per private dense consumer.",
        format_bytes(budget.max_ram_bytes()),
        format_bytes(plan.floor),
        format_bytes(plan.csr),
        format_bytes(plan.resident),
        format_bytes(plan.mapped),
        format_bytes(plan.private)
    );
    Ok(())
}

/// Adds a consumer's private cells and counts into the shared accumulator. Lanes add with
/// wrapping arithmetic: a lane of a partial sum is exact modulo 2^64 as the shared lane is, and
/// the plan bounds the lane's total over every row, so the merged lane is that exact total
/// whichever consumer added which rows.
fn merge_bounded_cells(
    accumulator: &Mutex<(Vec<i64>, Vec<u32>)>,
    cells: Vec<i64>,
    counts: Vec<u32>,
) -> Result<(), PipelineError> {
    let mut locked = accumulator.lock().map_err(|_| {
        PipelineError::Compute("Bounded accumulator lock was poisoned.".to_string())
    })?;
    let (scores, missing) = &mut *locked;
    scores
        .par_iter_mut()
        .zip(cells)
        .for_each(|(m, p)| *m = m.wrapping_add(p));
    missing
        .par_iter_mut()
        .zip(counts)
        .for_each(|(m, p)| *m += p);
    Ok(())
}

/// Where a bounded dense consumer adds its batches: the shared cells, under their lock, or
/// private cells it allocates at its first batch.
enum DenseCells<'a> {
    Shared(&'a Mutex<(Vec<i64>, Vec<u32>)>),
    Private(Option<(Vec<i64>, Vec<u32>)>),
}

/// Scores a dense stream's batches into `cells`, each batch's rows concatenated and every buffer
/// back in the pool first, so I/O overlaps compute. Shared and private consumers score through
/// this one call of the kernel.
fn consume_dense_batches(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: &io::RowBufferPool,
    batch_size: usize,
    cells: &mut DenseCells<'_>,
) -> Result<(), PipelineError> {
    let prep_result = &context.prep_result;
    let mut concatenated_data = Vec::new();
    let mut reconciled_indices = Vec::new();
    let mut scratch = DenseScratch::default();

    for batch_result in ChannelBatcher::new(rx, batch_size) {
        let batch = batch_result?;
        if batch.is_empty() {
            continue;
        }

        reconciled_indices.clear();
        reconciled_indices.extend(batch.iter().map(|wi| wi.reconciled_variant_index));
        concatenated_data.clear();
        let needed_len = batch
            .len()
            .checked_mul(prep_result.bytes_per_variant as usize)
            .ok_or_else(|| {
                PipelineError::Compute("Dense bounded batch byte length overflow.".to_string())
            })?;
        if concatenated_data.capacity() < needed_len {
            concatenated_data
                .try_reserve_exact(needed_len - concatenated_data.len())
                .map_err(|e| {
                    PipelineError::Compute(format!(
                        "Failed to reserve dense bounded batch buffer ({}): {e}",
                        format_bytes(needed_len)
                    ))
                })?;
        }
        for wi in batch {
            concatenated_data.extend_from_slice(&wi.data);
            drop(BufferGuard {
                buffer: Some(wi.data),
                pool: buffer_pool,
            });
        }
        let mut shared = None;
        let (scores, counts) = match cells {
            DenseCells::Shared(accumulator) => {
                let locked = shared.insert(accumulator.lock().map_err(|_| {
                    PipelineError::Compute("Bounded accumulator lock was poisoned.".to_string())
                })?);
                let (scores, counts) = &mut **locked;
                (scores, counts)
            }
            DenseCells::Private(private) => {
                let owned = match private.take() {
                    Some(owned) => owned,
                    None => initialize_cells(prep_result)?,
                };
                let (scores, counts) = private.insert(owned);
                (scores, counts)
            }
        };
        batch::run_dense_batch(
            &concatenated_data,
            &reconciled_indices,
            prep_result,
            &context.person_layout,
            &mut scratch,
            scores,
            counts,
        )?;
    }

    Ok(())
}

fn bounded_dense_batch_size(
    prep: &PreparationResult,
    budget: MemoryBudget,
) -> Result<usize, PipelineError> {
    let row_bytes = dense_scratch_bytes(prep, 1)?.max(1);
    Ok((budget.max_ram_bytes() / 16 / row_bytes).clamp(1, DENSE_BATCH_SIZE))
}

/// The most bytes one dense consumer holds for batches of at most `variants` rows: the batch's
/// rows concatenated, its work descriptors and indices, and the dense kernel's scratch as it
/// grows ([`batch::dense_scratch_bound`]). The memory plan charges this; batch sizing keeps
/// [`dense_scratch_bytes`].
fn consumer_scratch_bytes(
    prep: &PreparationResult,
    variants: usize,
) -> Result<usize, PipelineError> {
    let row = usize::try_from(prep.bytes_per_variant)
        .map_err(|_| PipelineError::Compute("PLINK row width overflow.".into()))?;
    let per_row = row
        .checked_add(
            std::mem::size_of::<WorkItem>() + std::mem::size_of::<ReconciledVariantIndex>(),
        )
        .and_then(|per_row| per_row.checked_mul(variants))
        .ok_or_else(|| PipelineError::Compute("Dense consumer scratch size overflow.".into()))?;
    Ok(per_row.saturating_add(batch::dense_scratch_bound(
        variants,
        row,
        prep.num_people_to_score,
        prep.exact().stride(),
    )))
}

fn dense_scratch_bytes(prep: &PreparationResult, variants: usize) -> Result<usize, PipelineError> {
    let row = usize::try_from(prep.bytes_per_variant).ok();
    // Packed calls, each variant's four term rows, its share of the four-variant tables (256
    // lane rows per group, sixteen groups at a time), the people's table keys and batched work
    // descriptors coexist. Score width matters as much as N.
    let lane_bytes = (4 + 16) * std::mem::size_of::<i64>();
    row.and_then(|row| {
        prep.exact()
            .stride()
            .checked_mul(lane_bytes)
            .and_then(|terms| row.checked_add(terms))
    })
    .and_then(|row| row.checked_add(prep.num_people_to_score.div_ceil(16)))
    .and_then(|row| {
        row.checked_add(
            std::mem::size_of::<WorkItem>() + std::mem::size_of::<ReconciledVariantIndex>(),
        )
    })
    .and_then(|row| row.checked_mul(variants))
    .ok_or_else(|| PipelineError::Compute("Dense scoring scratch size overflow.".into()))
}

/// The memory this process already holds beyond the plan the floor counts. The budget is read
/// after preparation, so what preparation left resident is not in its free memory but counts
/// toward the peak. The CSR is resident too, and the floor already charges it at the bytes it
/// holds.
fn held_beyond_plan(prep: &PreparationResult, budget: MemoryBudget) -> usize {
    budget
        .resident_bytes()
        .saturating_sub(prep.csr_heap_bytes())
}

/// What a run's input sources add to its memory: the rows it reads through memory maps, the
/// read buffers its sources hold (each read plan's window and the stacks of the threads it
/// fetches on, and each remote `.bed` read whole), and whether some other source that serves no
/// map may still take the local prefetch budget.
#[derive(Clone, Copy, Debug)]
struct InputCharge {
    mapped: usize,
    read_buffers: usize,
    prefetch: bool,
}

impl InputCharge {
    /// Before the sources open: no mapped rows, no read buffers yet, and room for a prefetch
    /// window.
    const UNOPENED: Self = Self {
        mapped: 0,
        read_buffers: 0,
        prefetch: true,
    };

    fn of(prep: &PreparationResult, sources: &[io::BedSource]) -> Self {
        Self {
            mapped: mapped_row_bytes(prep, sources),
            read_buffers: sources.iter().fold(0usize, |total, source| {
                total
                    .saturating_add(source.read_buffer_bytes())
                    .saturating_add(source.read_stack_bytes())
            }),
            prefetch: sources
                .iter()
                .any(|source| source.mapped_rows().is_none() && source.read_buffer_bytes() == 0),
        }
    }
}

/// The input bytes a run reads through memory maps: the rows it needs, at most the length of the
/// sources that serve reads from a map. Touched mapped pages count toward the process's resident
/// memory, and the budget, read before any row is, does not charge them.
fn mapped_row_bytes(prep: &PreparationResult, sources: &[io::BedSource]) -> usize {
    let mapped = sources
        .iter()
        .filter(|source| source.mapped_rows().is_some())
        .fold(0u64, |total, source| total.saturating_add(source.len()));
    let needed = u64::try_from(prep.required_bim_indices.len())
        .unwrap_or(u64::MAX)
        .saturating_mul(prep.bytes_per_variant);
    usize::try_from(needed.min(mapped)).unwrap_or(usize::MAX)
}

/// Refuses a run whose floor, held memory and mapped input rows together exceed the budget, the
/// same charge the bounded plan makes before it adds any consumer. Until the run's sources are
/// open, `input` maps no rows and leaves room for a prefetch window.
fn ensure_memory_floor(
    prep: &PreparationResult,
    budget: MemoryBudget,
    input: InputCharge,
) -> Result<(), PipelineError> {
    let floor = memory_floor_bytes(prep, budget, input)?;
    let held = held_beyond_plan(prep, budget);
    let required = floor.saturating_add(held).saturating_add(input.mapped);
    if required > budget.max_ram_bytes() {
        return Err(PipelineError::Compute(format!(
            "Scoring requires at least {} for {} people and {} scores ({} to score, {} this process already holds and {} of input rows read through a memory map), exceeding the {} memory budget even with bounded accumulation. Reduce the kept cohort or score panel.",
            format_bytes(required),
            prep.num_people_to_score,
            prep.score_names.len(),
            format_bytes(floor),
            format_bytes(held),
            format_bytes(input.mapped),
            format_bytes(budget.max_ram_bytes())
        )));
    }
    Ok(())
}

/// The least memory scoring needs: one output, the CSR at the bytes it holds, the I/O buffers,
/// the read buffers the sources hold (a read plan's window, a remote `.bed` read whole), the
/// local prefetch budget while some other source that serves no map may take it, one dense
/// consumer's batch scratch and the stacks of the threads scoring starts.
fn memory_floor_bytes(
    prep: &PreparationResult,
    budget: MemoryBudget,
    input: InputCharge,
) -> Result<usize, PipelineError> {
    let output = result_bytes(prep)?;
    let row = usize::try_from(prep.bytes_per_variant)
        .map_err(|_| PipelineError::Compute("PLINK row width overflow.".into()))?;
    let direct = should_use_small_keep_direct_for_prep(prep);
    let buffers = if direct {
        0
    } else {
        io_buffer_count(prep, budget)?
    };
    let scratch = if direct {
        0
    } else {
        consumer_scratch_bytes(prep, bounded_dense_batch_size(prep, budget)?)?
    };
    let prefetch = if input.prefetch {
        io::local_prefetch_budget(prep, budget)
    } else {
        0
    }
    .saturating_add(input.read_buffers);
    output
        .checked_add(prep.csr_heap_bytes())
        .and_then(|v| v.checked_add(row.checked_mul(buffers)?))
        .and_then(|v| v.checked_add(scratch))
        .and_then(|v| v.checked_add(prefetch))
        .and_then(|v| v.checked_add(SCORING_THREADS.checked_mul(thread_stack_bytes())?))
        .ok_or_else(|| PipelineError::Compute("Minimum scoring memory size overflow.".into()))
}

/// The threads every scoring run starts after the budget is read: the producer and the progress
/// monitor. Each private dense consumer is one more.
const SCORING_THREADS: usize = 2;

/// The stack a thread scoring starts may touch: std's default for a spawned thread, 2 MiB, or
/// `RUST_MIN_STACK` when it is set. Its pages count toward the peak like any other.
fn thread_stack_bytes() -> usize {
    std::env::var("RUST_MIN_STACK")
        .ok()
        .and_then(|bytes| bytes.parse().ok())
        .unwrap_or(2 << 20)
}
