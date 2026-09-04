use crate::adapt_plink2::GenomeBuild;
use crate::pipeline_error::PipelineError;
use crate::score::batch;
use crate::score::checkpoint::ScoreCheckpoint;
use crate::score::complex::{ComplexVariantResolver, resolve_complex_variants};
use crate::score::cuda_backend;
use crate::score::decide::{self, DecisionContext, RunStrategy};
use crate::score::io;
use crate::score::types::{
    BimRowIndex, EffectAlleleDosage, FilesetBoundary, PipelineKind, PreparationResult,
    ReconciledVariantIndex, WorkItem,
};
use ahash::AHashMap;
use crossbeam_channel::{Receiver, bounded};
use crossbeam_queue::ArrayQueue;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use memmap2::{Mmap, MmapOptions};
use num_cpus;
use rayon::prelude::*;
use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use sysinfo::System;

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
const FALLBACK_MAX_RAM_BYTES: usize = 8 * 1024 * 1024 * 1024;
/// Explicit memory budget for THIS process, in bytes.
///
/// Without it every gnomon process independently claims a fraction of the memory it
/// observes free, which is correct for one process on a machine and badly wrong for
/// several: N concurrent workers each size themselves to the whole box and together
/// commit N times what exists. Staggered starts make it worse rather than better,
/// since each new worker measures the memory the earlier ones have not yet touched.
///
/// A caller running workers in parallel knows the split and nothing else does, so it
/// sets this to its per-worker share. The budget is not merely advisory: exceeding it
/// selects the bounded-accumulator plan instead of the fast in-RAM one, so an honest
/// budget makes a large chromosome run SLOWER rather than die.
const MAX_RAM_ENV: &str = "GNOMON_MAX_RAM_BYTES";
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
}

impl MemoryBudget {
    fn auto() -> Self {
        let max_ram_bytes = default_max_ram_bytes().max(1);
        Self { max_ram_bytes }
    }

    #[inline]
    pub fn max_ram_bytes(self) -> usize {
        self.max_ram_bytes
    }
}

impl Default for MemoryBudget {
    fn default() -> Self {
        Self::auto()
    }
}

fn default_max_ram_bytes() -> usize {
    // An explicit budget wins outright: the caller partitioned the machine and this
    // process cannot see that partition by inspecting the system.
    if let Some(explicit) = env::var(MAX_RAM_ENV)
        .ok()
        .and_then(|raw| raw.trim().parse::<u64>().ok())
        .filter(|bytes| *bytes > 0)
    {
        return usize::try_from(explicit).unwrap_or(usize::MAX);
    }
    let mut system = System::new_all();
    system.refresh_memory();
    let available = system.available_memory();
    let candidate = if available > 0 {
        available.saturating_mul(DEFAULT_RAM_FRACTION_NUMERATOR) / DEFAULT_RAM_FRACTION_DENOMINATOR
    } else {
        FALLBACK_MAX_RAM_BYTES as u64
    };
    usize::try_from(candidate).unwrap_or(usize::MAX).max(1)
}

pub fn preflight_memory(
    prep_result: &PreparationResult,
    memory_budget: MemoryBudget,
) -> Result<(), PipelineError> {
    let result_size = checked_result_size(prep_result)?;
    let result_bytes = result_bytes(result_size)?;
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
    let io_bytes = row_bytes.checked_mul(buffer_count).ok_or_else(|| {
        PipelineError::Compute(format!(
            "I/O buffer estimate overflow: row_bytes={row_bytes}, buffers={buffer_count}"
        ))
    })?;

    let consumer_threads = if should_use_small_keep_direct_for_prep(prep_result) {
        0
    } else {
        choose_consumer_threads(result_size, memory_budget)
    };
    let accumulator_copies = if should_use_small_keep_direct_for_prep(prep_result) {
        1usize
    } else {
        consumer_threads
            .checked_mul(2)
            .and_then(|v| v.checked_add(1))
            .ok_or_else(|| PipelineError::Compute("Accumulator estimate overflow.".to_string()))?
    };
    let accumulator_bytes = result_bytes
        .checked_mul(accumulator_copies)
        .ok_or_else(|| PipelineError::Compute("Accumulator byte estimate overflow.".to_string()))?;
    let estimated_bytes = accumulator_bytes
        .checked_add(io_bytes)
        .and_then(|v| v.checked_add(csr_bytes))
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
    let prep_result = &context.prep_result;
    let result_size = checked_result_size(prep_result)?;
    let result_bytes = result_bytes(result_size)?;
    let csr_bytes = csr_bytes(prep_result)?;
    let row_bytes = usize::try_from(prep_result.bytes_per_variant).map_err(|_| {
        PipelineError::Compute(format!(
            "PLINK row width {} does not fit on this platform.",
            prep_result.bytes_per_variant
        ))
    })?;
    let buffer_count = context.io_buffer_count()?;
    let io_bytes = row_bytes.checked_mul(buffer_count).ok_or_else(|| {
        PipelineError::Compute(format!(
            "I/O buffer estimate overflow: row_bytes={row_bytes}, buffers={buffer_count}"
        ))
    })?;
    let fast_threads = choose_consumer_threads(result_size, context.memory_budget);
    let fast_copies = fast_threads
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| PipelineError::Compute("Accumulator estimate overflow.".to_string()))?;
    let fast_bytes = result_bytes
        .checked_mul(fast_copies)
        .and_then(|v| v.checked_add(csr_bytes))
        .and_then(|v| v.checked_add(io_bytes))
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
        context.prep_result.num_reconciled_variants,
        context.prep_result.total_variants_in_bim,
    )
}

fn result_bytes(result_size: usize) -> Result<usize, PipelineError> {
    result_size
        .checked_mul(std::mem::size_of::<f64>() + std::mem::size_of::<u32>())
        .ok_or_else(|| PipelineError::Compute("Result byte estimate overflow.".to_string()))
}

fn csr_bytes(prep_result: &PreparationResult) -> Result<usize, PipelineError> {
    let weights = prep_result
        .sparse_weights()
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| PipelineError::Compute("CSR weight byte estimate overflow.".to_string()))?;
    let missing = prep_result
        .sparse_missing_corrections()
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| {
            PipelineError::Compute("CSR missing-correction byte estimate overflow.".to_string())
        })?;
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
        .checked_add(missing)
        .and_then(|v| v.checked_add(columns))
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
    let by_parallelism = num_cpus::get()
        .max(1)
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

pub fn make_bed_buffer_pool(
    context: &PipelineContext,
) -> Result<Arc<ArrayQueue<Vec<u8>>>, PipelineError> {
    let count = context.io_buffer_count()?;
    let row_bytes = usize::try_from(context.prep_result.bytes_per_variant).map_err(|_| {
        PipelineError::Compute(format!(
            "PLINK row width {} does not fit on this platform.",
            context.prep_result.bytes_per_variant
        ))
    })?;
    let buffer_pool = Arc::new(ArrayQueue::new(count));
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

/// Owns shared resource pools and provides a handle to the read-only preparation results.
pub struct PipelineContext {
    pub prep_result: Arc<PreparationResult>,
    pub tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    pub checkpoint: Option<ScoreCheckpoint>,
    pub checkpoint_path: Option<PathBuf>,
    pub checkpoint_fingerprint: Option<[u8; 32]>,
    pub memory_budget: MemoryBudget,
    pub genome_build: Option<GenomeBuild>,
}

impl PipelineContext {
    /// Creates a new `PipelineContext`, allocating all necessary memory pools.
    pub fn new(prep_result: Arc<PreparationResult>) -> Self {
        Self {
            prep_result,
            tile_pool: Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 4)),
            checkpoint: None,
            checkpoint_path: None,
            checkpoint_fingerprint: None,
            memory_budget: MemoryBudget::default(),
            genome_build: None,
        }
    }

    pub fn with_checkpoint(
        prep_result: Arc<PreparationResult>,
        checkpoint: Option<ScoreCheckpoint>,
        checkpoint_path: PathBuf,
        checkpoint_fingerprint: [u8; 32],
        memory_budget: MemoryBudget,
        genome_build: Option<GenomeBuild>,
    ) -> Self {
        Self {
            prep_result,
            tile_pool: Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 4)),
            checkpoint,
            checkpoint_path: Some(checkpoint_path),
            checkpoint_fingerprint: Some(checkpoint_fingerprint),
            memory_budget,
            genome_build,
        }
    }

    #[inline]
    pub fn checkpoint_completed_variants(&self) -> usize {
        self.checkpoint
            .as_ref()
            .map(|checkpoint| checkpoint.completed_variants)
            .unwrap_or(0)
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
pub fn run(context: &PipelineContext) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    if should_use_small_keep_direct(context) {
        return match &context.prep_result.pipeline_kind {
            PipelineKind::SingleFile(bed_path) => {
                run_small_keep_direct_single_file(context, bed_path)
            }
            PipelineKind::MultiFile(boundaries) => {
                run_small_keep_direct_multi_file(context, boundaries)
            }
        };
    }

    if let Some(result) = cuda_backend::try_run_cuda(context)? {
        return Ok(result);
    }

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
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // --- 1. Setup: Memory-map the file, create channels and a shared buffer pool ---
    let bed_source = open_scoring_bed_source(context, bed_path)?;
    let shared_source = bed_source.byte_source();

    let channel_bound = context.work_channel_bound()?;
    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);

    let buffer_pool = make_bed_buffer_pool(context)?;

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let resume_from = context.checkpoint_completed_variants();
    if resume_from > 0 {
        eprintln!(
            "> Resuming score computation from checkpoint at {resume_from}/{variants_to_process} variants."
        );
    }
    let variants_processed_count = Arc::new(AtomicU64::new(resume_from as u64));
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

    let master_baseline = prep_result.baseline_missing_sum_by_score().to_vec();
    let use_bounded_accumulator = should_use_bounded_accumulator(context)?;
    if use_bounded_accumulator {
        eprintln!(
            "> Using bounded RAM accumulator: one shared f64/u32 output matrix, no per-thread full-matrix copies."
        );
    }
    let mut shared_accumulator = if use_bounded_accumulator {
        let (mut final_scores, mut final_counts) = initialize_final_output(
            prep_result.num_people_to_score,
            prep_result.score_names.len(),
            &master_baseline,
        )?;
        apply_checkpoint_initial_state(context, &mut final_scores, &mut final_counts)?;
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
    let final_result: Result<(Option<(Vec<f64>, Vec<u32>)>, Option<SpoolState>), PipelineError> =
        thread::scope(|s| {
            // Spawn the UI updater thread. This thread is responsible for polling the
            // atomic counter and updating the progress bar on the screen.
            let updater_thread_count = Arc::clone(&variants_processed_count);
            let updater_pb = pb.clone();
            let total_variants = variants_to_process;
            let stderr_is_tty = std::io::stderr().is_terminal();
            s.spawn(move || {
                // When stderr is a TTY, the indicatif bar handles user-facing
                // progress. When it isn't (e.g. gnomon is run as a subprocess
                // and stderr is a pipe), the bar's draw target is hidden, so
                // emit periodic plain-text lines instead — otherwise the
                // caller sees "> Decision Engine Strategy: ..." and then
                // silence for the whole run.
                let mut last_text_print = Instant::now();
                let initial_processed = updater_thread_count.load(Ordering::Relaxed);
                let initial_pct = if total_variants == 0 {
                    100
                } else {
                    initial_processed.saturating_mul(100) / total_variants
                };
                let mut last_text_pct: u64 = initial_pct;
                if !stderr_is_tty {
                    eprintln!(
                        "> Progress: {initial_processed}/{total_variants} variants ({initial_pct}%)"
                    );
                }
                // This loop terminates when the number of processed items reaches the
                // total, ensuring this thread finishes before the scope ends
                while updater_thread_count.load(Ordering::Relaxed) < total_variants {
                    let processed = updater_thread_count.load(Ordering::Relaxed);
                    updater_pb.set_position(processed);
                    if !stderr_is_tty {
                        maybe_emit_text_progress(
                            processed,
                            total_variants,
                            &mut last_text_print,
                            &mut last_text_pct,
                            Duration::from_secs(5),
                            5,
                        );
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                // Perform one final update to ensure the bar shows 100% completion.
                updater_pb.set_position(updater_thread_count.load(Ordering::Relaxed));
                if !stderr_is_tty {
                    eprintln!("> Progress: {total_variants}/{total_variants} variants (100%)");
                }
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
                            io::producer_thread(
                                Arc::clone(&source),
                                Arc::clone(&prep_result),
                                Some(sparse_tx),
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                resume_from,
                                spool_plan,
                            );
                        }
                        RunStrategy::UseComplexTree => {
                            let path_decider = |variant_data: &[u8]| {
                                let current_freq = batch::assess_variant_density_for_dispatch(
                                    variant_data,
                                    run_ctx.n_cohort as usize,
                                );
                                let variant_ctx = DecisionContext {
                                    freq: current_freq,
                                    ..run_ctx
                                };
                                decide::decide_path_with_freq(&variant_ctx)
                            };
                            let spool_plan = if spool_enabled {
                                let state = local_spool_state
                                    .as_mut()
                                    .expect("spool state missing despite spooling enabled");
                                Some(create_spool_plan(prep_result.as_ref(), state)?)
                            } else {
                                None
                            };
                            io::producer_thread(
                                Arc::clone(&source),
                                Arc::clone(&prep_result),
                                Some(sparse_tx),
                                dense_tx,
                                buffer_pool,
                                producer_thread_count,
                                path_decider,
                                resume_from,
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
                let num_people = prep_result.num_people_to_score;
                let num_scores = prep_result.score_names.len();
                let (mut final_scores, mut final_counts) =
                    initialize_final_output(num_people, num_scores, &master_baseline)?;
                apply_checkpoint_initial_state(context, &mut final_scores, &mut final_counts)?;
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
                    .for_each(|(m, p)| *m += p);
                final_scores
                    .par_iter_mut()
                    .zip(dense_adjustments)
                    .for_each(|(m, p)| *m += p);
                Some((final_scores, final_counts))
            };

            pb.finish_with_message("Computation complete.");
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
            let (offsets, spool_bytes_per_variant) = {
                let mut state = spool_state
                    .take()
                    .expect("spool state missing despite spooling enabled");
                state.writer.flush().map_err(|e| {
                    PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
                })?;
                (state.offsets, prep_result.spool_bytes_per_variant())
            };
            let mmap = if spool_bytes_per_variant == 0 {
                let mut anon = MmapOptions::new().len(1).map_anon().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to allocate anonymous mapping for empty complex spool: {e}"
                    ))
                })?;
                anon.copy_from_slice(&[0u8]);
                anon.make_read_only().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to convert anonymous mapping to read-only: {e}"
                    ))
                })?
            } else {
                let spool_file = File::open(&spool_file_path).map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to open complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?;
                unsafe { Mmap::map(&spool_file) }.map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to memory-map complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?
            };
            let dense_map = Arc::new(prep_result.spool_dense_map().to_vec());
            let resolver = ComplexVariantResolver::from_spool(
                Arc::new(mmap),
                offsets,
                spool_bytes_per_variant,
                dense_map,
            );
            if let Err(e) = resolve_complex_variants(
                &resolver,
                prep_result,
                &mut final_scores,
                &mut final_counts,
            ) {
                // Cleanup spool before propagating error
                let _ = fs::remove_file(&spool_file_path);
                return Err(e);
            }
            if let Err(e) = fs::remove_file(&spool_file_path) {
                eprintln!(
                    "> Warning: Failed to delete complex spool {}: {}",
                    spool_file_path.display(),
                    e
                );
            }
        } else {
            let resolver = ComplexVariantResolver::from_single_source(bed_source.clone());
            resolve_complex_variants(&resolver, prep_result, &mut final_scores, &mut final_counts)?;
        }
    }

    Ok((final_scores, final_counts))
}

/// The pipeline implementation for the multi-fileset case.
fn run_multi_file_pipeline(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let bed_sources: Vec<io::BedSource> = boundaries
        .iter()
        .map(|b| open_scoring_bed_source(context, &b.bed_path))
        .collect::<Result<_, _>>()?;
    let any_remote = bed_sources.iter().any(|s| s.mmap().is_none());
    let shared_sources = Arc::new(bed_sources);

    // --- 1. Setup: No mmap here. Producer manages its own. ---
    let channel_bound = context.work_channel_bound()?;
    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(channel_bound);
    let buffer_pool = make_bed_buffer_pool(context)?;

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let resume_from = context.checkpoint_completed_variants();
    if resume_from > 0 {
        eprintln!(
            "> Resuming score computation from checkpoint at {resume_from}/{variants_to_process} variants."
        );
    }
    let variants_processed_count = Arc::new(AtomicU64::new(resume_from as u64));
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
    let master_baseline = prep_result.baseline_missing_sum_by_score().to_vec();
    let use_bounded_accumulator = should_use_bounded_accumulator(context)?;
    if use_bounded_accumulator {
        eprintln!(
            "> Using bounded RAM accumulator: one shared f64/u32 output matrix, no per-thread full-matrix copies."
        );
    }
    let mut shared_accumulator = if use_bounded_accumulator {
        let (mut final_scores, mut final_counts) = initialize_final_output(
            prep_result.num_people_to_score,
            prep_result.score_names.len(),
            &master_baseline,
        )?;
        apply_checkpoint_initial_state(context, &mut final_scores, &mut final_counts)?;
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
    let final_result: Result<(Option<(Vec<f64>, Vec<u32>)>, Option<SpoolState>), PipelineError> =
        thread::scope(|s| {
            // Spawn the UI updater thread. This thread is responsible for polling the
            // atomic counter and updating the progress bar on the screen.
            let updater_thread_count = Arc::clone(&variants_processed_count);
            let updater_pb = pb.clone();
            let total_variants = variants_to_process;
            let stderr_is_tty = std::io::stderr().is_terminal();
            s.spawn(move || {
                // See first updater thread: emit plain-text progress when
                // stderr isn't a TTY so subprocess callers see something.
                let mut last_text_print = Instant::now();
                let initial_processed = updater_thread_count.load(Ordering::Relaxed);
                let initial_pct = if total_variants == 0 {
                    100
                } else {
                    initial_processed.saturating_mul(100) / total_variants
                };
                let mut last_text_pct: u64 = initial_pct;
                if !stderr_is_tty {
                    eprintln!(
                        "> Progress: {initial_processed}/{total_variants} variants ({initial_pct}%)"
                    );
                }
                // This loop terminates when the number of processed items reaches the
                // total, ensuring this thread finishes before the scope ends
                while updater_thread_count.load(Ordering::Relaxed) < total_variants {
                    let processed = updater_thread_count.load(Ordering::Relaxed);
                    updater_pb.set_position(processed);
                    if !stderr_is_tty {
                        maybe_emit_text_progress(
                            processed,
                            total_variants,
                            &mut last_text_print,
                            &mut last_text_pct,
                            Duration::from_secs(5),
                            5,
                        );
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                // Perform one final update to ensure the bar shows 100% completion.
                updater_pb.set_position(updater_thread_count.load(Ordering::Relaxed));
                if !stderr_is_tty {
                    eprintln!("> Progress: {total_variants}/{total_variants} variants (100%)");
                }
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
                                resume_from,
                                spool_plan,
                            );
                        }
                        RunStrategy::UseComplexTree => {
                            let path_decider = |variant_data: &[u8]| {
                                let current_freq = batch::assess_variant_density_for_dispatch(
                                    variant_data,
                                    run_ctx.n_cohort as usize,
                                );
                                let variant_ctx = DecisionContext {
                                    freq: current_freq,
                                    ..run_ctx
                                };
                                decide::decide_path_with_freq(&variant_ctx)
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
                                resume_from,
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
                let num_people = prep_result.num_people_to_score;
                let num_scores = prep_result.score_names.len();
                let (mut final_scores, mut final_counts) =
                    initialize_final_output(num_people, num_scores, &master_baseline)?;
                apply_checkpoint_initial_state(context, &mut final_scores, &mut final_counts)?;
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
                    .for_each(|(m, p)| *m += p);
                final_scores
                    .par_iter_mut()
                    .zip(dense_adjustments)
                    .for_each(|(m, p)| *m += p);
                Some((final_scores, final_counts))
            };

            pb.finish_with_message("Computation complete.");
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
            let (offsets, spool_bytes_per_variant) = {
                let mut state = spool_state
                    .take()
                    .expect("spool state missing despite spooling enabled");
                state.writer.flush().map_err(|e| {
                    PipelineError::Io(format!("Failed to flush complex variant spool: {e}"))
                })?;
                (state.offsets, prep_result.spool_bytes_per_variant())
            };
            let mmap = if spool_bytes_per_variant == 0 {
                let mut anon = MmapOptions::new().len(1).map_anon().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to allocate anonymous mapping for empty complex spool: {e}"
                    ))
                })?;
                anon.copy_from_slice(&[0u8]);
                anon.make_read_only().map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to convert anonymous mapping to read-only: {e}"
                    ))
                })?
            } else {
                let spool_file = File::open(&spool_file_path).map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to open complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?;
                unsafe { Mmap::map(&spool_file) }.map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to memory-map complex variant spool {}: {e}",
                        spool_file_path.display()
                    ))
                })?
            };
            let dense_map = Arc::new(prep_result.spool_dense_map().to_vec());
            let resolver = ComplexVariantResolver::from_spool(
                Arc::new(mmap),
                offsets,
                spool_bytes_per_variant,
                dense_map,
            );
            if let Err(e) = resolve_complex_variants(
                &resolver,
                prep_result,
                &mut final_scores,
                &mut final_counts,
            ) {
                // Cleanup spool before propagating error
                let _ = fs::remove_file(&spool_file_path);
                return Err(e);
            }
            if let Err(e) = fs::remove_file(&spool_file_path) {
                eprintln!(
                    "> Warning: Failed to delete complex spool {}: {}",
                    spool_file_path.display(),
                    e
                );
            }
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
    bed_path: &Path,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    eprintln!(
        "> Using small-keep direct PLINK path for {} kept individual(s).",
        context.prep_result.num_people_to_score
    );
    let bed_source = open_scoring_bed_source(context, bed_path)?;
    let prep_result = &context.prep_result;
    let master_baseline = prep_result.baseline_missing_sum_by_score().to_vec();
    let (mut final_scores, mut final_counts) = initialize_final_output(
        prep_result.num_people_to_score,
        prep_result.score_names.len(),
        &master_baseline,
    )?;
    apply_checkpoint_initial_state(context, &mut final_scores, &mut final_counts)?;

    let total = prep_result.num_reconciled_variants as u64;
    let resume_from = context.checkpoint_completed_variants();
    if resume_from > 0 {
        eprintln!("> Resuming direct score computation from {resume_from}/{total} variants.");
    }
    let pb = create_progress_bar(total, "Computing scores...");
    pb.set_position(resume_from as u64);
    let mut scratch = [0u8; 1];
    let mut processed_since_update = 0u64;

    for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
        if i < resume_from {
            continue;
        }
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

        for out_idx in 0..prep_result.num_people_to_score {
            let fam_idx = prep_result.output_idx_to_fam_idx[out_idx].0 as usize;
            let byte_offset = row_base
                .checked_add((fam_idx / 4) as u64)
                .ok_or_else(|| PipelineError::Compute("PLINK byte offset overflow.".to_string()))?;
            let byte = if let Some(mmap) = bed_source.mmap() {
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
            apply_packed_genotype(
                prep_result,
                reconciled_idx,
                out_idx,
                packed,
                &mut final_scores,
                &mut final_counts,
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
    Ok((final_scores, final_counts))
}

fn run_small_keep_direct_multi_file(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    eprintln!(
        "> Using small-keep direct PLINK path for {} kept individual(s).",
        context.prep_result.num_people_to_score
    );
    let bed_sources: Vec<io::BedSource> = boundaries
        .iter()
        .map(|b| open_scoring_bed_source(context, &b.bed_path))
        .collect::<Result<_, _>>()?;
    let prep_result = &context.prep_result;
    let master_baseline = prep_result.baseline_missing_sum_by_score().to_vec();
    let (mut final_scores, mut final_counts) = initialize_final_output(
        prep_result.num_people_to_score,
        prep_result.score_names.len(),
        &master_baseline,
    )?;
    apply_checkpoint_initial_state(context, &mut final_scores, &mut final_counts)?;

    let total = prep_result.num_reconciled_variants as u64;
    let resume_from = context.checkpoint_completed_variants();
    if resume_from > 0 {
        eprintln!("> Resuming direct score computation from {resume_from}/{total} variants.");
    }
    let pb = create_progress_bar(total, "Computing scores...");
    pb.set_position(resume_from as u64);
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
        if i < resume_from {
            continue;
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
        if row_end > bed_source.len() {
            return Err(PipelineError::Io(format!(
                "Fatal: Read past end of .bed source '{}' for variant with global index {}. Source may be corrupt.",
                boundaries[current_fileset_idx].bed_path.display(),
                global_bim_row_index.0
            )));
        }

        for out_idx in 0..prep_result.num_people_to_score {
            let fam_idx = prep_result.output_idx_to_fam_idx[out_idx].0 as usize;
            let byte_offset = row_base
                .checked_add((fam_idx / 4) as u64)
                .ok_or_else(|| PipelineError::Compute("PLINK byte offset overflow.".to_string()))?;
            let byte = if let Some(mmap) = bed_source.mmap() {
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
            apply_packed_genotype(
                prep_result,
                reconciled_idx,
                out_idx,
                packed,
                &mut final_scores,
                &mut final_counts,
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
    Ok((final_scores, final_counts))
}

fn apply_packed_genotype(
    prep_result: &PreparationResult,
    reconciled_idx: ReconciledVariantIndex,
    out_idx: usize,
    packed: u8,
    final_scores: &mut [f64],
    final_counts: &mut [u32],
) {
    let num_scores = prep_result.score_names.len();
    let scores_offset = out_idx * num_scores;
    let variant_view = prep_result.variant_csr_view(reconciled_idx);
    match packed {
        0b00 => {}
        0b01 => {
            for contribution in variant_view.iter() {
                let col = contribution.score_column.0;
                final_counts[scores_offset + col] += 1;
                final_scores[scores_offset + col] -= contribution.missing_correction as f64;
            }
        }
        0b10 | 0b11 => {
            let dosage = if packed == 0b10 { 1.0 } else { 2.0 };
            for contribution in variant_view.iter() {
                let col = contribution.score_column.0;
                final_scores[scores_offset + col] += contribution.weight as f64 * dosage;
            }
        }
        _ => unreachable!(),
    }
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
    pool: &'a ArrayQueue<Vec<u8>>,
}

struct DenseMiniBatchCanvas<'a> {
    weights: &'a mut [f32],
    missing_corrections: &'a mut [f32],
    stride: usize,
}

impl<'a> DenseMiniBatchCanvas<'a> {
    #[inline(always)]
    fn set(&mut self, batch_row: usize, score_col: usize, weight: f32, missing_correction: f32) {
        let idx = batch_row * self.stride + score_col;
        self.weights[idx] = weight;
        self.missing_corrections[idx] = missing_correction;
    }
}

impl<'a> Drop for BufferGuard<'a> {
    fn drop(&mut self) {
        // When the guard is dropped, it returns its buffer to the pool.
        if let Some(mut buf) = self.buffer.take() {
            buf.clear();
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

type ConsumerResult = Result<(Vec<f64>, Vec<u32>), PipelineError>;

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
fn initialize_final_output(
    num_people: usize,
    num_scores: usize,
    baseline: &[f64],
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    let result_size = num_people.checked_mul(num_scores).ok_or_else(|| {
        PipelineError::Compute(format!(
            "Final output size overflow: num_people={num_people} * num_scores={num_scores}"
        ))
    })?;
    if baseline.len() != num_scores {
        return Err(PipelineError::Compute(format!(
            "Baseline length mismatch: baseline={}, num_scores={num_scores}",
            baseline.len()
        )));
    }

    let mut final_scores = Vec::new();
    final_scores.try_reserve_exact(result_size).map_err(|e| {
        PipelineError::Compute(format!(
            "Failed to reserve final score matrix ({} cells): {e}",
            result_size
        ))
    })?;
    for _ in 0..num_people {
        final_scores.extend_from_slice(baseline);
    }
    let mut final_counts = Vec::new();
    final_counts.try_reserve_exact(result_size).map_err(|e| {
        PipelineError::Compute(format!(
            "Failed to reserve final missing-count matrix ({} cells): {e}",
            result_size
        ))
    })?;
    final_counts.resize(result_size, 0u32);
    Ok((final_scores, final_counts))
}

fn apply_checkpoint_initial_state(
    context: &PipelineContext,
    final_scores: &mut [f64],
    final_counts: &mut [u32],
) -> Result<(), PipelineError> {
    let Some(checkpoint) = context.checkpoint.as_ref() else {
        return Ok(());
    };
    if checkpoint.sum_scores.len() != final_scores.len()
        || checkpoint.missing_counts.len() != final_counts.len()
    {
        return Err(PipelineError::Compute(format!(
            "Checkpoint accumulator shape mismatch: scores {} vs {}, counts {} vs {}.",
            checkpoint.sum_scores.len(),
            final_scores.len(),
            checkpoint.missing_counts.len(),
            final_counts.len()
        )));
    }
    final_scores.copy_from_slice(&checkpoint.sum_scores);
    final_counts.copy_from_slice(&checkpoint.missing_counts);
    Ok(())
}

#[inline]
fn choose_consumer_threads(result_size: usize, memory_budget: MemoryBudget) -> usize {
    let cpu_cap = num_cpus::get().max(1);

    let bytes_per_accumulator = result_bytes(result_size).unwrap_or(usize::MAX);
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
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = checked_result_size(prep_result)?;
    let consumer_threads = choose_consumer_threads(result_size, context.memory_budget);
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
                || (vec![0.0f64; result_size], vec![0u32; result_size]), // Each thread gets its own accumulator.
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
                            &mut acc.0,
                            &mut acc.1,
                            work_item.reconciled_variant_index,
                        )?;
                    }
                    Ok::<_, PipelineError>(acc)
                },
            )
            .try_reduce(
                || (vec![0.0f64; result_size], vec![0u32; result_size]), // Identity for the reduction.
                |mut a, b| {
                    // Combine accumulators from two threads in parallel.
                    a.0.par_iter_mut()
                        .zip(b.0)
                        .for_each(|(v_a, v_b)| *v_a += v_b);
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
    Ok(final_result)
}

fn process_sparse_stream_bounded(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    accumulator: Arc<Mutex<(Vec<f64>, Vec<u32>)>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
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
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = checked_result_size(prep_result)?;
    let consumer_threads = choose_consumer_threads(result_size, context.memory_budget);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(consumer_threads)
        .build()
        .map_err(|e| PipelineError::Compute(format!("Failed to build dense consumer pool: {e}")))?;

    // Instantiate our new Send-compatible batching iterator.
    let batch_iterator = ChannelBatcher::new(rx, DENSE_BATCH_SIZE);

    // Use the exact same fold/reduce pattern as the sparse stream, but on batches.
    let final_result = pool.install(|| {
        batch_iterator
            .par_bridge() // This is now possible and correct.
            .try_fold(
                || {
                    // Per-thread accumulator initializer
                    (
                        vec![0.0f64; result_size],
                        vec![0u32; result_size],
                        Vec::with_capacity(
                            DENSE_BATCH_SIZE * (prep_result.bytes_per_variant as usize),
                        ),
                        Vec::<f32>::new(),
                        Vec::<f32>::new(),
                        Vec::<ReconciledVariantIndex>::with_capacity(DENSE_BATCH_SIZE),
                    )
                },
                |mut acc, batch_result| {
                    // The `?` operator handles propagating errors from the channel.
                    let batch = batch_result?;
                    if batch.is_empty() {
                        return Ok(acc);
                    }

                    acc.5.clear();
                    acc.5
                        .extend(batch.iter().map(|wi| wi.reconciled_variant_index));

                    let concatenated_data = &mut acc.2;
                    concatenated_data.clear();

                    {
                        // The dense kernel reads the concatenated copy, so each source
                        // buffer can return to the producer immediately after copying.
                        // This removes a per-batch allocation and lets I/O overlap the
                        // entire compute phase instead of waiting on 256 held buffers.
                        for wi in batch {
                            concatenated_data.extend_from_slice(&wi.data);
                            drop(BufferGuard {
                                buffer: Some(wi.data),
                                pool: &buffer_pool,
                            });
                        }

                        let stride = prep_result.stride();
                        let matrix_len = acc.5.len() * stride;
                        acc.3.resize(matrix_len, 0.0f32);
                        acc.3.fill(0.0);
                        acc.4.resize(matrix_len, 0.0f32);
                        acc.4.fill(0.0);
                        let mut canvas = DenseMiniBatchCanvas {
                            weights: &mut acc.3,
                            missing_corrections: &mut acc.4,
                            stride,
                        };
                        for (batch_row_idx, &reconciled_idx) in acc.5.iter().enumerate() {
                            let variant_view = prep_result.variant_csr_view(reconciled_idx);
                            for contribution in variant_view.iter() {
                                let col = contribution.score_column.0;
                                canvas.set(
                                    batch_row_idx,
                                    col,
                                    contribution.weight,
                                    contribution.missing_correction,
                                );
                            }
                        }

                        batch::run_person_major_path(
                            concatenated_data,
                            &acc.3,
                            &acc.4,
                            &acc.5,
                            prep_result,
                            &mut acc.0,
                            &mut acc.1,
                            &context.tile_pool,
                        )?;
                    }

                    Ok::<_, PipelineError>(acc)
                },
            )
            .try_reduce(
                || {
                    (
                        vec![0.0; result_size],
                        vec![0; result_size],
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                        Vec::new(),
                    )
                },
                |mut a, b| {
                    a.0.par_iter_mut()
                        .zip(b.0)
                        .for_each(|(v_a, v_b)| *v_a += v_b);
                    a.1.par_iter_mut()
                        .zip(b.1)
                        .for_each(|(v_a, v_b)| *v_a += v_b);
                    Ok(a)
                },
            )
    })?;

    // The `?` operator unwrapped the `Result` from the reduction. If the stream was
    // empty, `try_reduce` (on a `TryFold` iterator) returns the identity value, so
    // `final_result` correctly contains the initial empty vectors. We just need to destructure the tuple.
    let (scores, counts, _, _, _, _) = final_result;
    Ok((scores, counts))
}

fn process_dense_stream_bounded(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    accumulator: Arc<Mutex<(Vec<f64>, Vec<u32>)>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let batch_size = bounded_dense_batch_size(context);
    let mut batch_iterator = ChannelBatcher::new(rx, batch_size);
    let mut concatenated_data = Vec::new();
    let mut weights_for_batch = Vec::new();
    let mut missing_corrections_for_batch = Vec::new();

    for batch_result in &mut batch_iterator {
        let batch = batch_result?;
        if batch.is_empty() {
            continue;
        }

        let reconciled_indices: Vec<ReconciledVariantIndex> =
            batch.iter().map(|wi| wi.reconciled_variant_index).collect();
        concatenated_data.clear();
        let needed_len = batch
            .len()
            .checked_mul(prep_result.bytes_per_variant as usize)
            .ok_or_else(|| {
                PipelineError::Compute("Dense bounded batch byte length overflow.".to_string())
            })?;
        if concatenated_data.capacity() < needed_len {
            concatenated_data
                .try_reserve_exact(needed_len - concatenated_data.capacity())
                .map_err(|e| {
                    PipelineError::Compute(format!(
                        "Failed to reserve dense bounded batch buffer ({}): {e}",
                        format_bytes(needed_len)
                    ))
                })?;
        }

        {
            for wi in batch {
                concatenated_data.extend_from_slice(&wi.data);
                drop(BufferGuard {
                    buffer: Some(wi.data),
                    pool: &buffer_pool,
                });
            }

            let stride = prep_result.stride();
            let matrix_len = reconciled_indices
                .len()
                .checked_mul(stride)
                .ok_or_else(|| {
                    PipelineError::Compute(
                        "Dense bounded weight matrix length overflow.".to_string(),
                    )
                })?;
            if weights_for_batch.capacity() < matrix_len {
                weights_for_batch
                    .try_reserve_exact(matrix_len - weights_for_batch.capacity())
                    .map_err(|e| {
                        PipelineError::Compute(format!(
                            "Failed to reserve dense bounded weight matrix ({matrix_len} cells): {e}"
                        ))
                    })?;
            }
            weights_for_batch.resize(matrix_len, 0.0f32);
            weights_for_batch.fill(0.0);
            if missing_corrections_for_batch.capacity() < matrix_len {
                missing_corrections_for_batch
                    .try_reserve_exact(matrix_len - missing_corrections_for_batch.capacity())
                    .map_err(|e| {
                        PipelineError::Compute(format!(
                            "Failed to reserve dense bounded missing-correction matrix ({matrix_len} cells): {e}"
                        ))
                    })?;
            }
            missing_corrections_for_batch.resize(matrix_len, 0.0f32);
            missing_corrections_for_batch.fill(0.0);
            let mut canvas = DenseMiniBatchCanvas {
                weights: &mut weights_for_batch,
                missing_corrections: &mut missing_corrections_for_batch,
                stride,
            };
            for (batch_row_idx, &reconciled_idx) in reconciled_indices.iter().enumerate() {
                let variant_view = prep_result.variant_csr_view(reconciled_idx);
                for contribution in variant_view.iter() {
                    let col = contribution.score_column.0;
                    canvas.set(
                        batch_row_idx,
                        col,
                        contribution.weight,
                        contribution.missing_correction,
                    );
                }
            }

            {
                let mut locked = accumulator.lock().map_err(|_| {
                    PipelineError::Compute("Bounded accumulator lock was poisoned.".to_string())
                })?;
                let (scores, counts) = &mut *locked;
                batch::run_person_major_path(
                    &concatenated_data,
                    &weights_for_batch,
                    &missing_corrections_for_batch,
                    &reconciled_indices,
                    prep_result,
                    scores,
                    counts,
                    &context.tile_pool,
                )?;
            }
        }
    }

    Ok((Vec::new(), Vec::new()))
}

fn bounded_dense_batch_size(context: &PipelineContext) -> usize {
    let row_bytes = usize::try_from(context.prep_result.bytes_per_variant)
        .unwrap_or(usize::MAX)
        .max(1);
    let dense_budget = (context.memory_budget.max_ram_bytes() / 16).max(row_bytes);
    (dense_budget / row_bytes).clamp(1, DENSE_BATCH_SIZE)
}

