use crate::batch::{self, SparseIndexPool};
use crate::decide::{self, DecisionContext, RunStrategy};
use crate::io;
use crate::types::{
    BimRowIndex, EffectAlleleDosage, FilesetBoundary, PipelineKind, PreparationResult,
    ReconciledVariantIndex, ScoreColumnIndex, WorkItem,
};
use ahash::AHashSet;
use crossbeam_channel::{bounded, Receiver};
use crossbeam_queue::ArrayQueue;
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use num_cpus;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use std::sync::Mutex;
use dashmap::DashSet;

// --- Pipeline Tuning Parameters ---

/// The maximum number of sparse work items that can be buffered in the channel.
/// Provides backpressure against a fast producer.
const SPARSE_CHANNEL_BOUND: usize = 8192;
/// The maximum number of dense work items that can be buffered in the channel.
const DENSE_CHANNEL_BOUND: usize = 4096;
/// The number of dense variants to process in a single person-major batch.
/// Tuned for L3 cache efficiency.
const DENSE_BATCH_SIZE: usize = 256;
/// The number of reusable memory buffers for variant data.
const BUFFER_POOL_SIZE: usize = 16384;

/// A performant, read-only resolver for fetching complex variant genotypes.
///
/// This enum is initialized ONCE at the start of the pipeline run. It holds
/// either a single memory map or a collection of them, avoiding the massive
/// performance penalty of re-opening and re-mapping files inside a parallel loop.
enum ComplexVariantResolver {
    SingleFile(Arc<Mmap>),
    MultiFile {
        // A vector of all memory maps for the multi-file case.
        mmaps: Vec<Mmap>,
        // A copy of the boundaries to map a global index to the correct mmap.
        boundaries: Vec<FilesetBoundary>,
    },
}

impl ComplexVariantResolver {
    /// Creates a new resolver based on the pipeline strategy.
    fn new(prep_result: &PreparationResult) -> Result<Self, PipelineError> {
        match &prep_result.pipeline_kind {
            PipelineKind::SingleFile(bed_path) => {
                let file = File::open(bed_path).map_err(|e| {
                    PipelineError::Io(format!("Opening {}: {}", bed_path.display(), e))
                })?;
                let mmap = Arc::new(unsafe {
                    Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string()))?
                });
                Ok(Self::SingleFile(mmap))
            }
            PipelineKind::MultiFile(boundaries) => {
                let mmaps = boundaries
                    .iter()
                    .map(|b| {
                        let file = File::open(&b.bed_path).map_err(|e| {
                            PipelineError::Io(format!("Opening {}: {}", b.bed_path.display(), e))
                        })?;
                        unsafe { Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string())) }
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Self::MultiFile {
                    mmaps,
                    boundaries: boundaries.clone(),
                })
            }
        }
    }

    /// Fetches a packed genotype for a given person and global variant index.
    /// This is the fast, central lookup method used by the parallel resolver.
    #[inline(always)]
    fn get_packed_genotype(
        &self,
        bytes_per_variant: u64,
        bim_row_index: BimRowIndex,
        fam_index: u32,
    ) -> u8 {
        let (mmap, local_bim_index) = match self {
            ComplexVariantResolver::SingleFile(mmap) => (mmap.as_ref(), bim_row_index.0),
            ComplexVariantResolver::MultiFile { mmaps, boundaries } => {
                // Find which fileset contains this global index using a fast binary search.
                let fileset_idx =
                    boundaries.partition_point(|b| b.starting_global_index <= bim_row_index.0) - 1;
                let boundary = &boundaries[fileset_idx];
                let local_index = bim_row_index.0 - boundary.starting_global_index;
                // This unsafe is acceptable because the number of mmaps is tied to the
                // number of boundaries, and the index is derived from it.
                (unsafe { mmaps.get_unchecked(fileset_idx) }, local_index)
            }
        };

        // The +3 skips the PLINK .bed file magic number (0x6c, 0x1b, 0x01).
        let variant_start_offset = 3 + local_bim_index * bytes_per_variant;
        let person_byte_offset = fam_index as u64 / 4;
        let final_byte_offset = (variant_start_offset + person_byte_offset) as usize;

        let bit_offset_in_byte = (fam_index % 4) * 2;

        // This indexing is safe because the preparation phase guarantees all indices are valid.
        let packed_byte = unsafe { *mmap.get_unchecked(final_byte_offset) };
        (packed_byte >> bit_offset_in_byte) & 0b11
    }
}

// ========================================================================================
//                          PUBLIC API, CONTEXT & ERROR HANDLING
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

                // Greedily pull more items from the channel *without blocking* until
                // the batch is full or the channel is momentarily empty.
                while batch.len() < self.batch_size {
                    match self.rx.try_recv() {
                        Ok(Ok(item)) => batch.push(item),
                        // An error was sent down the channel. Stop and propagate it.
                        Ok(Err(e)) => return Some(Err(e)),
                        // The channel is empty or disconnected. The current batch is finished.
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

/// A specialized error type for the pipeline, allowing for robust, clonable error
/// propagation from any concurrent stage.
#[derive(Debug, Clone)]
pub enum PipelineError {
    Compute(String),
    Io(String),
    Producer(String),
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Compute(e) => write!(f, "{}", e),
            PipelineError::Io(e) => write!(f, "I/O error during pipeline execution: {}", e),
            PipelineError::Producer(e) => write!(f, "The data producer thread failed: {}", e),
        }
    }
}
impl Error for PipelineError {}

// Enables easy conversion from batch errors into a pipeline error.
impl From<Box<dyn Error + Send + Sync>> for PipelineError {
    fn from(e: Box<dyn Error + Send + Sync>) -> Self {
        PipelineError::Compute(e.to_string())
    }
}

/// Owns shared resource pools and provides a handle to the read-only preparation results.
pub struct PipelineContext {
    pub prep_result: Arc<PreparationResult>,
    pub tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    pub sparse_index_pool: Arc<SparseIndexPool>,
}

impl PipelineContext {
    /// Creates a new `PipelineContext`, allocating all necessary memory pools.
    pub fn new(prep_result: Arc<PreparationResult>) -> Self {
        Self {
            prep_result,
            tile_pool: Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 4)),
            sparse_index_pool: Arc::new(SparseIndexPool::new()),
        }
    }
}

/// Executes the entire concurrent compute pipeline.
///
/// This is the primary public entry point. It is synchronous and returns the
/// final aggregated scores and counts upon successful completion.
pub fn run(context: &PipelineContext) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // This match is a zero-cost abstraction. The compiler generates a simple jump
    // to the correct function based on the enum variant, and it's impossible
    // to call the wrong pipeline logic for a given configuration.
    match &context.prep_result.pipeline_kind {
        PipelineKind::SingleFile(bed_path) => run_single_file_pipeline(context, bed_path),
        PipelineKind::MultiFile(boundaries) => run_multi_file_pipeline(context, boundaries),
    }
}

// ========================================================================================
//                        PIPELINE STAGE IMPLEMENTATIONS
// ========================================================================================

/// The pipeline implementation for the common single-fileset case.
/// This function's body is effectively the same as the original `pipeline::run` function,
/// guaranteeing zero performance regression.
fn run_single_file_pipeline(
    context: &PipelineContext,
    bed_path: &Path,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // --- 1. Setup: Memory-map the file, create channels and a shared buffer pool ---
    let bed_file = File::open(bed_path)
        .map_err(|e| PipelineError::Io(format!("Opening {}: {}", bed_path.display(), e)))?;
    let mmap = Arc::new(
        unsafe { Mmap::map(&bed_file).map_err(|e| PipelineError::Io(e.to_string()))? },
    );
    mmap.advise(memmap2::Advice::Sequential)
        .map_err(|e| PipelineError::Io(e.to_string()))?;

    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(SPARSE_CHANNEL_BOUND);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(DENSE_CHANNEL_BOUND);

    let buffer_pool = Arc::new(ArrayQueue::new(BUFFER_POOL_SIZE));
    for _ in 0..BUFFER_POOL_SIZE {
        buffer_pool
            .push(Vec::with_capacity(
                context.prep_result.bytes_per_variant as usize,
            ))
            .unwrap();
    }

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let variants_processed_count = Arc::new(AtomicU64::new(0));
    let pb = ProgressBar::new(variants_to_process);
    pb.set_style(
        ProgressStyle::with_template(
            "\n> [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message("Computing scores...");

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
    eprintln!("> Decision Engine Strategy: {:?}", strategy);

    let num_scores = prep_result.score_names.len();
    let stride = prep_result.stride();
    let master_baseline: Vec<f64> = (0..prep_result.num_reconciled_variants)
        .into_par_iter()
        .fold(
            || vec![0.0f64; num_scores],
            |mut local_baseline, i| {
                let flip_row_offset = i * stride;
                let flip_row =
                    &prep_result.flip_mask_matrix()[flip_row_offset..flip_row_offset + stride];
                let weight_row =
                    &prep_result.weights_matrix()[flip_row_offset..flip_row_offset + stride];
                for k in 0..num_scores {
                    if flip_row[k] == 1 {
                        local_baseline[k] += 2.0 * weight_row[k] as f64;
                    }
                }
                local_baseline
            },
        )
        .reduce(
            || vec![0.0f64; num_scores],
            |mut a, b| {
                a.par_iter_mut().zip(b).for_each(|(v_a, v_b)| *v_a += v_b);
                a
            },
        );

// --- 3. Orchestration: Use a scoped thread for safe producer/consumer execution ---
        let final_result: Result<(Vec<f64>, Vec<u32>), PipelineError> = thread::scope(|s| {
            // Spawn the UI updater thread. This thread is responsible for polling the
            // atomic counter and updating the progress bar on the screen.
            let updater_thread_count = Arc::clone(&variants_processed_count);
            let updater_pb = pb.clone();
            let total_variants = variants_to_process;
            s.spawn(move || {
                // This loop terminates when the number of processed items reaches the
                // total, ensuring this thread finishes before the scope ends
                while updater_thread_count.load(Ordering::Relaxed) < total_variants {
                    let processed = updater_thread_count.load(Ordering::Relaxed);
                    updater_pb.set_position(processed);
                    thread::sleep(Duration::from_millis(200));
                }
                // Perform one final update to ensure the bar shows 100% completion.
                updater_pb.set_position(updater_thread_count.load(Ordering::Relaxed));
            });

        let producer_logic = {
            let mmap = Arc::clone(&mmap);
            let prep_result = Arc::clone(&context.prep_result);
            let buffer_pool = Arc::clone(&buffer_pool);
            let producer_thread_count = Arc::clone(&variants_processed_count);

            move || match strategy {
                RunStrategy::UseSimpleTree => {
                    let global_path = decide::decide_path_without_freq(&run_ctx);
                    let path_decider = |_variant_data: &[u8]| global_path;
                    io::producer_thread(
                        mmap,
                        prep_result,
                        sparse_tx,
                        dense_tx,
                        buffer_pool,
                        producer_thread_count,
                        path_decider,
                    );
                }
                RunStrategy::UseComplexTree => {
                    let path_decider = |variant_data: &[u8]| {
                        let current_freq =
                            batch::assess_variant_density(variant_data, run_ctx.n_cohort as usize);
                        let variant_ctx = DecisionContext {
                            freq: current_freq,
                            ..run_ctx
                        };
                        decide::decide_path_with_freq(&variant_ctx)
                    };
                    io::producer_thread(
                        mmap,
                        prep_result,
                        sparse_tx,
                        dense_tx,
                        buffer_pool,
                        producer_thread_count,
                        path_decider,
                    );
                }
            }
        };

        let producer_handle = s.spawn(producer_logic);
        let (sparse_result, dense_result) = rayon::join(
            || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
            || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
        );
        producer_handle
            .join()
            .map_err(|_| PipelineError::Producer("Producer thread panicked.".to_string()))?;

        // --- 4. Aggregate final results ---
        let (sparse_adjustments, sparse_counts) = sparse_result?;
        let (dense_adjustments, dense_counts) = dense_result?;
        let num_people = prep_result.num_people_to_score;
        let mut final_scores = Vec::with_capacity(num_people * num_scores);
        for _ in 0..num_people {
            final_scores.extend_from_slice(&master_baseline);
        }
        let mut final_counts = vec![0u32; num_people * num_scores];
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

        // --- 5. Resolve complex variants ---
        if !prep_result.complex_rules.is_empty() {
            eprintln!(
                "> Resolving {} unique complex variant rule(s)...",
                prep_result.complex_rules.len()
            );
            // The resolver is created once with the single mmap.
            let resolver = ComplexVariantResolver::SingleFile(Arc::clone(&mmap));
            resolve_complex_variants(
                &resolver,
                prep_result,
                &mut final_scores,
                &mut final_counts,
            )?;
        }
        pb.finish_with_message("Computation complete.");
        Ok((final_scores, final_counts))
    });
    final_result
}

/// The pipeline implementation for the multi-fileset case.
fn run_multi_file_pipeline(
    context: &PipelineContext,
    boundaries: &[FilesetBoundary],
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // --- 1. Setup: No mmap here. Producer manages its own. ---
    let (sparse_tx, sparse_rx) = bounded::<Result<WorkItem, PipelineError>>(SPARSE_CHANNEL_BOUND);
    let (dense_tx, dense_rx) = bounded::<Result<WorkItem, PipelineError>>(DENSE_CHANNEL_BOUND);
    let buffer_pool = Arc::new(ArrayQueue::new(BUFFER_POOL_SIZE));
    for _ in 0..BUFFER_POOL_SIZE {
        buffer_pool
            .push(Vec::with_capacity(
                context.prep_result.bytes_per_variant as usize,
            ))
            .unwrap();
    }

    // Progress Reporting Setup
    let variants_to_process = context.prep_result.num_reconciled_variants as u64;
    let variants_processed_count = Arc::new(AtomicU64::new(0));
    let pb = ProgressBar::new(variants_to_process);
    pb.set_style(
        ProgressStyle::with_template(
            "\n> [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}",
        )
        .unwrap()
        .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.set_message("Computing scores...");

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
    eprintln!("> Decision Engine Strategy: {:?}", strategy);
    let master_baseline = {
        let num_scores = prep_result.score_names.len();
        let stride = prep_result.stride();
        (0..prep_result.num_reconciled_variants)
            .into_par_iter()
            .fold(
                || vec![0.0f64; num_scores],
                |mut local_baseline, i| {
                    let flip_row_offset = i * stride;
                    let flip_row =
                        &prep_result.flip_mask_matrix()[flip_row_offset..flip_row_offset + stride];
                    let weight_row =
                        &prep_result.weights_matrix()[flip_row_offset..flip_row_offset + stride];
                    for k in 0..num_scores {
                        if flip_row[k] == 1 {
                            local_baseline[k] += 2.0 * weight_row[k] as f64;
                        }
                    }
                    local_baseline
                },
            )
            .reduce(
                || vec![0.0f64; num_scores],
                |mut a, b| {
                    a.par_iter_mut().zip(b).for_each(|(v_a, v_b)| *v_a += v_b);
                    a
                },
            )
    };

    // --- 3. Orchestration with multi-file producer ---
    let final_result: Result<(Vec<f64>, Vec<u32>), PipelineError> = thread::scope(|s| {
        // Spawn the UI updater thread. This thread is responsible for polling the
        // atomic counter and updating the progress bar on the screen.
        let updater_thread_count = Arc::clone(&variants_processed_count);
        let updater_pb = pb.clone();
        let total_variants = variants_to_process;
        s.spawn(move || {
            // This loop terminates when the number of processed items reaches the
            // total, ensuring this thread finishes before the scope ends
            while updater_thread_count.load(Ordering::Relaxed) < total_variants {
                let processed = updater_thread_count.load(Ordering::Relaxed);
                updater_pb.set_position(processed);
                thread::sleep(Duration::from_millis(200));
            }
            // Perform one final update to ensure the bar shows 100% completion.
            updater_pb.set_position(updater_thread_count.load(Ordering::Relaxed));
        });

        let producer_logic = {
            let prep_result = Arc::clone(&context.prep_result);
            let buffer_pool = Arc::clone(&buffer_pool);
            let producer_thread_count = Arc::clone(&variants_processed_count);

            move || match strategy {
                RunStrategy::UseSimpleTree => {
                    let global_path = decide::decide_path_without_freq(&run_ctx);
                    let path_decider = |_variant_data: &[u8]| global_path;
                    io::multi_file_producer_thread(
                        prep_result,
                        boundaries,
                        sparse_tx,
                        dense_tx,
                        buffer_pool,
                        producer_thread_count,
                        path_decider,
                    );
                }
                RunStrategy::UseComplexTree => {
                    let path_decider = |variant_data: &[u8]| {
                        let current_freq =
                            batch::assess_variant_density(variant_data, run_ctx.n_cohort as usize);
                        let variant_ctx = DecisionContext {
                            freq: current_freq,
                            ..run_ctx
                        };
                        decide::decide_path_with_freq(&variant_ctx)
                    };
                    io::multi_file_producer_thread(
                        prep_result,
                        boundaries,
                        sparse_tx,
                        dense_tx,
                        buffer_pool,
                        producer_thread_count,
                        path_decider,
                    );
                }
            }
        };

        let producer_handle = s.spawn(producer_logic);
        let (sparse_result, dense_result) = rayon::join(
            || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
            || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
        );
        producer_handle
            .join()
            .map_err(|_| PipelineError::Producer("Producer thread panicked.".to_string()))?;

        // --- 4. Aggregate final results (same as single-file) ---
        let (sparse_adjustments, sparse_counts) = sparse_result?;
        let (dense_adjustments, dense_counts) = dense_result?;
        let num_people = prep_result.num_people_to_score;
        let num_scores = prep_result.score_names.len();
        let mut final_scores = Vec::with_capacity(num_people * num_scores);
        for _ in 0..num_people {
            final_scores.extend_from_slice(&master_baseline);
        }
        let mut final_counts = vec![0u32; num_people * num_scores];
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

        // --- 5. Resolve complex variants ---
        if !prep_result.complex_rules.is_empty() {
            eprintln!(
                "> Resolving {} unique complex variant rule(s)...",
                prep_result.complex_rules.len()
            );
            // The resolver is created once with all necessary mmaps.
            let resolver = ComplexVariantResolver::new(prep_result)?;
            resolve_complex_variants(
                &resolver,
                prep_result,
                &mut final_scores,
                &mut final_counts,
            )?;
        }
        pb.finish_with_message("Computation complete.");
        Ok((final_scores, final_counts))
    });
    final_result
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
struct ScopeGuard<F: FnOnce()> {
    /// The closure to execute on drop. `Option` is used to allow the closure
    /// to be taken and called, preventing multiple executions.
    action: Option<F>,
}

impl<F: FnOnce()> ScopeGuard<F> {
    /// Creates a new `ScopeGuard` with the given action.
    ///
    /// The action will be executed when the returned guard is dropped.
    #[inline(always)]
    fn new(action: F) -> Self {
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

/// A contention-free consumer for the sparse variant stream, using Rayon's
/// fold/reduce pattern for maximum parallelism with no locks.
fn process_sparse_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = prep_result.num_people_to_score * prep_result.score_names.len();

    // The fold/reduce pattern creates thread-local accumulators for scores and counts.
    // After processing a work item, its data buffer is immediately returned to the
    // shared pool, creating a true, continuous recycling system.
    let final_result = rx
        .into_iter() // Convert the channel to a blocking iterator.
        .par_bridge() // Bridge it to a Rayon parallel iterator.
        .try_fold(
            || (vec![0.0f64; result_size], vec![0u32; result_size]), // Each thread gets its own accumulator.
            |mut acc, work_result| {
                // The work_item and its buffer are processed within this scope.
                // The `_guard` ensures the buffer is returned to the pool when this
                // scope ends, whether by success or by `?` propagating an error.
                {
                    let work_item = work_result?;
                    let _guard = BufferGuard {
                        buffer: Some(work_item.data),
                        pool: &buffer_pool,
                    };

                    batch::run_variant_major_path(
                        // The guard holds the buffer, so we borrow it from there.
                        _guard.buffer.as_ref().unwrap(),
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
                a.0.par_iter_mut().zip(b.0).for_each(|(v_a, v_b)| *v_a += v_b);
                a.1.par_iter_mut().zip(b.1).for_each(|(v_a, v_b)| *v_a += v_b);
                Ok(a)
            },
        )?;

    // `try_reduce` returns `Result<(scores, counts), PipelineError>`.
    // The `?` operator has already unwrapped the Result, leaving just the tuple.
    // With an identity function, try_reduce handles empty streams by returning the identity.
    Ok(final_result)
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
    let result_size = prep_result.num_people_to_score * prep_result.score_names.len();

    // Instantiate our new Send-compatible batching iterator.
    let batch_iterator = ChannelBatcher::new(rx, DENSE_BATCH_SIZE);

    // Use the exact same fold/reduce pattern as the sparse stream, but on batches.
    let final_result = batch_iterator
        .par_bridge() // This is now possible and correct.
        .try_fold(
            || { // Per-thread accumulator initializer
                (
                    vec![0.0f64; result_size],
                    vec![0u32; result_size],
                    Vec::with_capacity(DENSE_BATCH_SIZE * (prep_result.bytes_per_variant as usize)),
                )
            },
            |mut acc, batch_result| {
                // The `?` operator handles propagating errors from the channel.
                let batch = batch_result?;
                if batch.is_empty() { return Ok(acc); }

                let reconciled_indices: Vec<ReconciledVariantIndex> =
                    batch.iter().map(|wi| wi.reconciled_variant_index).collect();

                let concatenated_data = &mut acc.2;
                concatenated_data.clear();
                
                // BufferGuard ensures all variant data buffers are returned to the pool,
                // even if an error occurs during computation.
                let guards: Vec<_> = batch.into_iter().map(|wi| {
                    concatenated_data.extend_from_slice(&wi.data);
                    BufferGuard { buffer: Some(wi.data), pool: &buffer_pool }
                }).collect();

                let stride = prep_result.stride();
                let mut weights_for_batch = Vec::with_capacity(reconciled_indices.len() * stride);
                let mut flips_for_batch = Vec::with_capacity(reconciled_indices.len() * stride);
                for &reconciled_idx in &reconciled_indices {
                    let src_offset = reconciled_idx.0 as usize * stride;
                    weights_for_batch.extend_from_slice(&prep_result.weights_matrix()[src_offset..src_offset + stride]);
                    flips_for_batch.extend_from_slice(&prep_result.flip_mask_matrix()[src_offset..src_offset + stride]);
                }

                batch::run_person_major_path(
                    concatenated_data, &weights_for_batch, &flips_for_batch,
                    &reconciled_indices, prep_result, &mut acc.0, &mut acc.1,
                    &context.tile_pool, &context.sparse_index_pool,
                )?;
                
                drop(guards); // Explicitly drop to return buffers to the pool.

                Ok::<_, PipelineError>(acc)
            },
        )
        .try_reduce(
            || (vec![0.0; result_size], vec![0; result_size], Vec::new()),
            |mut a, b| {
                a.0.par_iter_mut().zip(b.0).for_each(|(v_a, v_b)| *v_a += v_b);
                a.1.par_iter_mut().zip(b.1).for_each(|(v_a, v_b)| *v_a += v_b);
                Ok(a)
            },
        )?;

    // The `?` operator unwrapped the `Result` from the reduction. If the stream was
    // empty, `try_reduce` (on a `TryFold` iterator) returns the identity value, so
    // `final_result` correctly contains the initial empty vectors. We just need to destructure the tuple.
    let (scores, counts, _) = final_result;
    Ok((scores, counts))
}


/// A private helper struct to hold the raw components of a warning message.
/// This avoids heap allocations (`format!`) inside the hot parallel loop.
struct WarningInfo {
    person_output_idx: usize,
    locus_id: BimRowIndex,
    winning_a1: String,
    winning_a2: String,
    score_col_idx: ScoreColumnIndex,
}

/// A private struct holding the raw data for one conflicting source of evidence.
/// This is used exclusively for building the final fatal error report.
struct ConflictSource {
    bim_row: BimRowIndex,
    alleles: (String, String),
    genotype_bits: u8,
}

/// A private struct holding the complete, raw payload for a fatal ambiguity error.
/// Collecting this data first and formatting it once at the end is a key optimization.
struct FatalAmbiguityData {
    iid: String,
    locus_chr_pos: (String, u32),
    score_name: String,
    conflicts: Vec<ConflictSource>,
}

/// A private struct holding the data for a critical but non-fatal integrity warning.
/// This is used when multiple data sources conflict but lead to a consistent outcome.
struct CriticalIntegrityWarningInfo {
    iid: String,
    locus_chr_pos: (String, u32),
    score_name: String,
    conflicts: Vec<ConflictSource>,
    consistent_dosage: f64,
}

/// A private helper enum to represent the outcome of processing one person for one rule.
/// This decouples the core logic from the side-effects (like I/O or setting global flags).
enum ResolutionOutcome {
    Success,
    Warning(WarningInfo),
    CriticalIntegrityWarning(CriticalIntegrityWarningInfo),
    Fatal(FatalAmbiguityData),
}

/// Processes a single person for a single complex rule.
///
/// This is a "pure" function: it contains only the core business logic and is free
/// of I/O, locks, or other side-effects, making it easy to test and reason about.
#[inline]
fn process_person_for_rule(
    resolver: &ComplexVariantResolver,
    prep_result: &Arc<PreparationResult>,
    group_rule: &crate::types::GroupedComplexRule,
    person_output_idx: usize,
    original_fam_idx: u32,
    person_scores_slice: &mut [f64],
    person_counts_slice: &mut [u32],
) -> ResolutionOutcome {
    // --- Step 1: Gather Evidence ---
    let mut valid_interpretations = Vec::with_capacity(group_rule.possible_contexts.len());
    for context in &group_rule.possible_contexts {
        let (bim_idx, ..) = context;
        let packed_geno =
            resolver.get_packed_genotype(prep_result.bytes_per_variant, *bim_idx, original_fam_idx);
        if packed_geno != 0b01 { // 0b01 is the PLINK "missing" code
            valid_interpretations.push((packed_geno, context));
        }
    }

    // --- Step 2: Apply Decision Policy ---
    match valid_interpretations.len() {
        0 => { // Case A: No valid genotypes found for this locus.
            let mut counted_cols: AHashSet<ScoreColumnIndex> = AHashSet::new();
            for score_info in &group_rule.score_applications {
                counted_cols.insert(score_info.score_column_index);
            }
            for score_col in counted_cols {
                person_counts_slice[score_col.0] += 1;
            }
            ResolutionOutcome::Success
        }
        1 => { // Case B: Exactly one valid genotype found. Unambiguous happy path.
            let (winning_geno, winning_context) = valid_interpretations[0];
            let (_bim_idx, winning_a1, winning_a2) = winning_context;
            for score_info in &group_rule.score_applications {
                let effect_allele = &score_info.effect_allele;
                let score_col = score_info.score_column_index.0;
                let weight = score_info.weight as f64;
                if effect_allele != winning_a1 && effect_allele != winning_a2 {
                    continue;
                }
                let dosage: f64 = if effect_allele == winning_a1 {
                    match winning_geno { 0b00 => 2.0, 0b10 => 1.0, 0b11 => 0.0, _ => unreachable!() }
                } else {
                    match winning_geno { 0b00 => 0.0, 0b10 => 1.0, 0b11 => 2.0, _ => unreachable!() }
                };
                person_scores_slice[score_col] += dosage * weight;
            }
            ResolutionOutcome::Success
        }
        _ => { // Case C: A data conflict was detected.
            for score_info in &group_rule.score_applications {
                let matching_interpretations: Vec<_> = valid_interpretations.iter().filter(|(_, context)| &score_info.effect_allele == &context.1 || &score_info.effect_allele == &context.2).collect();
                match matching_interpretations.len() {
                    1 => {
                        let (winning_geno, winning_context) = matching_interpretations[0];
                        let (locus_id, winning_a1, winning_a2) = &**winning_context;
                        let dosage: f64 = if &score_info.effect_allele == winning_a1 {
                            match *winning_geno { 0b00 => 2.0, 0b10 => 1.0, 0b11 => 0.0, _ => unreachable!() }
                        } else {
                            match *winning_geno { 0b00 => 0.0, 0b10 => 1.0, 0b11 => 2.0, _ => unreachable!() }
                        };
                        person_scores_slice[score_info.score_column_index.0] += dosage * score_info.weight as f64;
                        // Return the raw data for the warning, not the formatted string.
                        return ResolutionOutcome::Warning(WarningInfo {
                            person_output_idx,
                            locus_id: *locus_id,
                            winning_a1: winning_a1.clone(),
                            winning_a2: winning_a2.clone(),
                            score_col_idx: score_info.score_column_index,
                        });
                    }
                    0 => continue,
                    // This is the fatal ambiguity case. We collect all necessary
                    // This case handles multiple, conflicting, non-missing genotype
                    // records that are all relevant to the current score.
                    _ => {
                        // Instead of failing immediately, calculate the dosage from each
                        // conflicting source to see if they are consistent.
                        let mut dosages = Vec::with_capacity(matching_interpretations.len());
                        for (geno, context) in &matching_interpretations {
                            let a1 = &context.1;
                            let dosage: f64 = if &score_info.effect_allele == a1 {
                                match geno { 0b00 => 2.0, 0b10 => 1.0, 0b11 => 0.0, _ => unreachable!() }
                            } else {
                                match geno { 0b00 => 0.0, 0b10 => 1.0, 0b11 => 2.0, _ => unreachable!() }
                            };
                            dosages.push(dosage);
                        }

                        // Check if all calculated dosages are identical.
                        let first_dosage = dosages[0];
                        if dosages.iter().all(|&d| (d - first_dosage).abs() < 1e-9) {
                            // BENIGN AMBIGUITY: The data is messy (wrong or diploid), but the outcome is the same.
                            // We can apply the score and issue a critical warning.
                            person_scores_slice[score_info.score_column_index.0] += first_dosage * score_info.weight as f64;
                            
                            let conflicts: Vec<ConflictSource> = matching_interpretations
                                .iter()
                                .map(|(packed_geno, context)| ConflictSource {
                                    bim_row: context.0,
                                    alleles: (context.1.clone(), context.2.clone()),
                                    genotype_bits: *packed_geno,
                                })
                                .collect();

                            return ResolutionOutcome::CriticalIntegrityWarning(CriticalIntegrityWarningInfo {
                                iid: prep_result.final_person_iids[person_output_idx].clone(),
                                locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
                                conflicts,
                                consistent_dosage: first_dosage,
                            });

                        } else {
                            // MALIGNANT AMBIGUITY: The conflicting data leads to different
                            // results. The program must fail.
                            let conflicts: Vec<ConflictSource> = matching_interpretations
                                .iter()
                                .map(|(packed_geno, context)| ConflictSource {
                                    bim_row: context.0,
                                    alleles: (context.1.clone(), context.2.clone()),
                                    genotype_bits: *packed_geno,
                                })
                                .collect();

                            let data = FatalAmbiguityData {
                                iid: prep_result.final_person_iids[person_output_idx].clone(),
                                locus_chr_pos: group_rule.locus_chr_pos.clone(),
                                score_name: prep_result.score_names[score_info.score_column_index.0].clone(),
                                conflicts,
                            };
                            return ResolutionOutcome::Fatal(data);
                        }
                    }
                }
            }
            ResolutionOutcome::Success
        }
    }
}


/// The "slow path" resolver for complex, multiallelic variants.
///
/// This function runs *after* the main high-performance pipeline is complete. It
/// iterates through each person and resolves their score contributions for the small
/// set of variants that could not be handled by the fast path. It uses a rule-major
/// outer loop with a person-major parallel inner loop to provide granular progress.
fn resolve_complex_variants(
    resolver: &ComplexVariantResolver,
    prep_result: &Arc<PreparationResult>,
    final_scores: &mut [f64],
    final_missing_counts: &mut [u32],
) -> Result<(), PipelineError> {
    let num_rules = prep_result.complex_rules.len();
    if num_rules == 0 {
        return Ok(());
    }

    eprintln!("> Resolving {} complex variant rules...", num_rules);

    // This state must persist across all iterations of the rules loop. These are
    // thread-safe types that can be safely shared between threads.
    let fatal_error_occurred = Arc::new(AtomicBool::new(false));
    let fatal_error_storage = Mutex::new(None::<FatalAmbiguityData>);
    let warned_pairs = DashSet::<(usize, BimRowIndex)>::new();
    let all_warnings_to_print = Mutex::new(Vec::<WarningInfo>::new());
    let all_critical_warnings_to_print = Mutex::new(Vec::<CriticalIntegrityWarningInfo>::new());

    // Iterate through rules one by one to provide clear, sequential progress to the user.
    for (rule_idx, group_rule) in prep_result.complex_rules.iter().enumerate() {
        if fatal_error_occurred.load(Ordering::Relaxed) {
            break;
        }

        // This state is specific to the processing of a single rule.
        let pb = ProgressBar::new(prep_result.num_people_to_score as u64);
        let progress_style = ProgressStyle::with_template(&format!(
            ">  - Rule {:2}/{} [{{bar:40.cyan/blue}}] {{pos}}/{{len}} ({{eta}})",
            rule_idx + 1, num_rules
        )).expect("Internal Error: Invalid progress bar template string.");
        pb.set_style(progress_style.progress_chars("█▉▊▋▌▍▎▏ "));

        let progress_counter = Arc::new(AtomicU64::new(0));

        // Use a scoped thread block to ensure the main thread waits for both the
        // workers and the progress updater to finish before proceeding. `thread::scope`
        // guarantees that any threads spawned within it will complete before the scope
        // exits, allowing safe borrowing of data from the parent stack.
        thread::scope(|s| {
            // Spawner #1: The dedicated progress bar updater thread.
            // This closure uses `move` because it only needs to own its copies of the
            // `Arc` pointers, which is a cheap and correct way to pass them.
            s.spawn({
                let pb_updater = pb.clone();
                let counter_for_updater = Arc::clone(&progress_counter);
                let error_flag_for_updater = Arc::clone(&fatal_error_occurred);
                let total_people = prep_result.num_people_to_score as u64;

                move || {
                    // This loop terminates under two conditions:
                    // 1. All work is complete (counter reaches total).
                    // 2. A fatal error has been signaled by a worker thread.
                    // This prevents the updater from hanging if workers stop early.
                    while counter_for_updater.load(Ordering::Relaxed) < total_people
                        && !error_flag_for_updater.load(Ordering::Relaxed)
                    {
                        pb_updater.set_position(counter_for_updater.load(Ordering::Relaxed));
                        thread::sleep(Duration::from_millis(200));
                    }
                    // Perform one final update to show the terminal state. If an error
                    // occurred, this accurately reflects how many items were processed
                    // before the operation was aborted.
                    pb_updater.set_position(counter_for_updater.load(Ordering::Relaxed));
                }
            });

            // Spawner #2: The worker threads (managed by Rayon).
            // This closure does NOT use `move`. It correctly borrows data from the
            // parent scope. This is safe because `thread::scope` guarantees this thread
            // cannot outlive the borrowed data (like `final_scores`, `warned_pairs`, etc.).
            s.spawn(|| {
                final_scores
                    .par_chunks_mut(prep_result.score_names.len())
                    .zip(final_missing_counts.par_chunks_mut(prep_result.score_names.len()))
                    .enumerate()
                    .for_each(|(person_output_idx, (person_scores_slice, person_counts_slice))| {
                        // This guard ensures the progress counter is always incremented
                        // when the closure for a person finishes, regardless of how it exits
                        let _progress_guard = ScopeGuard::new(|| {
                            progress_counter.fetch_add(1, Ordering::Relaxed);
                        });

                        // This check provides a fast-fail mechanism, preventing new work
                        // from being done after a fatal error has been detected.
                        if fatal_error_occurred.load(Ordering::Relaxed) {
                            return;
                        }

                        let original_fam_idx = prep_result.output_idx_to_fam_idx[person_output_idx];

                        let outcome = process_person_for_rule(
                            resolver, prep_result, group_rule, person_output_idx,
                            original_fam_idx, person_scores_slice, person_counts_slice,
                        );

                          match outcome {
                            ResolutionOutcome::Success => {}
                            ResolutionOutcome::Warning(info) => {
                                if warned_pairs.insert((info.person_output_idx, info.locus_id)) {
                                    all_warnings_to_print.lock().unwrap().push(info);
                                }
                            }
                            ResolutionOutcome::CriticalIntegrityWarning(info) => {
                                // This is a non-fatal but severe warning. We collect it
                                // to report at the end. We do not set the fatal error flag.
                                all_critical_warnings_to_print.lock().unwrap().push(info);
                            }
                            ResolutionOutcome::Fatal(data) => {
                                // Use compare_exchange to ensure only the FIRST fatal error
                                // payload is stored. This prevents race conditions where multiple
                                // threads might fail on different individuals simultaneously.
                                if fatal_error_occurred.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                                    *fatal_error_storage.lock().unwrap() = Some(data);
                                }
                                // `return` ensures we stop processing for this thread.
                                return;
                            }
                        }
                    });
            });
        }); // Scope ends, all spawned threads are joined.

        pb.finish_with_message("Done.");
    }

    // After all rules are processed, drain and print a summary of warnings. This is
    // performed once by the main thread to avoid lock contention on stderr.
    let collected_warnings = std::mem::take(&mut *all_warnings_to_print.lock().unwrap());
    let collected_critical_warnings = std::mem::take(&mut *all_critical_warnings_to_print.lock().unwrap());
    const MAX_WARNINGS_TO_PRINT: usize = 10;

    let total_critical_warnings = collected_critical_warnings.len();
    if total_critical_warnings > 0 {
        eprintln!("\n\n========================= CRITICAL DATA INTEGRITY ISSUE =========================");
        eprintln!("Gnomon detected one or more loci with conflicting genotype data that were\n\
                  resolved using a heuristic. While computation was able to continue, the\n\
                  underlying genotype data is ambiguous and should be investigated.");
        eprintln!("---------------------------------------------------------------------------------");
        for (i, info) in collected_critical_warnings.into_iter().enumerate() {
            if i >= MAX_WARNINGS_TO_PRINT {
                break;
            }
            if i > 0 { eprintln!("---------------------------------------------------------------------------------"); }
            eprintln!("{}", format_critical_integrity_warning(&info));
        }
        if total_critical_warnings > MAX_WARNINGS_TO_PRINT {
             eprintln!("\n... and {} more similar critical warnings.", total_critical_warnings - MAX_WARNINGS_TO_PRINT);
        }
        eprintln!("=================================================================================\n");
    }

    let total_warnings = collected_warnings.len();
    if total_warnings > 0 {
        eprintln!("\n--- Data Inconsistency Warning Summary ---");
        for (i, info) in collected_warnings.into_iter().enumerate() {
            if i >= MAX_WARNINGS_TO_PRINT {
                break;
            }
            let iid = &prep_result.final_person_iids[info.person_output_idx];
            let score_name = &prep_result.score_names[info.score_col_idx.0];
            eprintln!(
                "WARNING: Resolved data inconsistency for IID '{}' at locus corresponding to BIM row {}. Multiple non-missing genotypes found. Used the one matching score '{}' (alleles: {}, {}).",
                iid, info.locus_id.0, score_name, info.winning_a1, info.winning_a2
            );
        }
        if total_warnings > MAX_WARNINGS_TO_PRINT {
            eprintln!("... and {} more similar warnings.", total_warnings - MAX_WARNINGS_TO_PRINT);
        }
    }

    // After all rules are processed, check if a fatal error was ever stored.
    // If so, retrieve it from the mutex, format the final report, and propagate it.
    if fatal_error_occurred.load(Ordering::Relaxed) {
        if let Ok(mut guard) = fatal_error_storage.lock() {
            if let Some(data) = guard.take() {
                let report = format_fatal_ambiguity_report(&data);
                return Err(PipelineError::Compute(report));
            }
        }
        // As a fallback, return a generic error if the specific one can't be retrieved.
        return Err(PipelineError::Compute(
            "A fatal, unspecified error occurred in a parallel task and the detailed report could not be generated.".to_string(),
        ));
    }

    eprintln!("> Complex variant resolution complete.");
    Ok(())
}

/// A private helper function to format the final, dense data report for a fatal ambiguity.
/// This is called only once, on the main thread, after a fatal error is confirmed.
fn format_fatal_ambiguity_report(data: &FatalAmbiguityData) -> String {
    use std::fmt::Write;
    let mut report = String::with_capacity(512);

    // Helper to interpret genotype bits and alleles into a human-readable string.
    let interpret_genotype = |bits: u8, a1: &str, a2: &str| -> String {
        match bits {
            0b00 => format!("{}/{}", a1, a1),
            0b01 => "Missing".to_string(),
            0b10 => format!("{}/{}", a1, a2),
            0b11 => format!("{}/{}", a2, a2),
            _ => "Invalid Bits".to_string(),
        }
    };

    // Build the final report string.
    writeln!(report, "Fatal: Unresolvable ambiguity for individual '{}'.\n", data.iid).unwrap();
    writeln!(report, "Individual:   {}", data.iid).unwrap();
    writeln!(report, "Locus:        {}:{}", data.locus_chr_pos.0, data.locus_chr_pos.1).unwrap();
    writeln!(report, "Score:        {}\n", data.score_name).unwrap();
    writeln!(report, "Conflicting Sources:").unwrap();

    for conflict in &data.conflicts {
        writeln!(report, "  - BIM Row: {}", conflict.bim_row.0).unwrap();
        writeln!(report, "    Alleles (A1,A2): ({}, {})", conflict.alleles.0, conflict.alleles.1).unwrap();

        let bits_str = match conflict.genotype_bits {
            0b00 => "00", 0b01 => "01", 0b10 => "10", 0b11 => "11", _ => "??"
        };
        let interpretation = interpret_genotype(conflict.genotype_bits, &conflict.alleles.0, &conflict.alleles.1);
        writeln!(report, "    Genotype Bits:   {} (Interpreted as {})", bits_str, interpretation).unwrap();
    }

    report
}

/// A private helper function to format a critical integrity warning.
/// This is called only once, on the main thread, to report benign ambiguities.
fn format_critical_integrity_warning(data: &CriticalIntegrityWarningInfo) -> String {
    use std::fmt::Write;
    let mut report = String::with_capacity(512);

    // Helper to interpret genotype bits and alleles into a human-readable string.
    let interpret_genotype = |bits: u8, a1: &str, a2: &str| -> String {
        match bits {
            0b00 => format!("{}/{}", a1, a1),
            0b01 => "Missing".to_string(),
            0b10 => format!("{}/{}", a1, a2),
            0b11 => format!("{}/{}", a2, a2),
            _ => "Invalid Bits".to_string(),
        }
    };

    // Build the final report string.
    writeln!(report, "Benign Ambiguity Resolved for Individual '{}'", data.iid).unwrap();
    writeln!(report, "  Locus:  {}:{}", data.locus_chr_pos.0, data.locus_chr_pos.1).unwrap();
    writeln!(report, "  Score:  {}", data.score_name).unwrap();
    writeln!(report, "  Outcome: All conflicting sources yielded a consistent dosage of {}, so computation continued.", data.consistent_dosage).unwrap();
    writeln!(report, "  \n  Conflicting Sources Found:").unwrap();

    for conflict in &data.conflicts {
        let interpretation = interpret_genotype(conflict.genotype_bits, &conflict.alleles.0, &conflict.alleles.1);
        writeln!(report, "    - BIM Row {}: Alleles=({}, {}), Genotype={} ({})",
            conflict.bim_row.0,
            conflict.alleles.0,
            conflict.alleles.1,
            conflict.genotype_bits,
            interpretation
        ).unwrap();
    }

    report.trim_end().to_string()
}
