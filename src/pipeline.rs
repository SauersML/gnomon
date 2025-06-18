use crate::batch::{self, SparseIndexPool};
use crate::io;
use crate::types::{EffectAlleleDosage, PreparationResult, ReconciledVariantIndex, WorkItem};
use crossbeam_channel::{bounded, Receiver};
use crossbeam_queue::ArrayQueue;
use itertools::Itertools;
use memmap2::Mmap;
use num_cpus;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::thread;

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

// ========================================================================================
//                          PUBLIC API, CONTEXT & ERROR HANDLING
// ========================================================================================

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
            PipelineError::Compute(e) => write!(f, "Computation error: {}", e),
            PipelineError::Io(e) => write!(f, "I/O error: {}", e),
            PipelineError::Producer(e) => write!(f, "Producer thread error: {}", e),
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
pub fn run(
    context: &PipelineContext,
    plink_prefix: &Path,
) -> Result<(Vec<f64>, Vec<u32>), PipelineError> {
    // --- 1. Setup: Memory-map the file, create channels and a shared buffer pool ---
    let bed_path = plink_prefix.with_extension("bed");
    let bed_file =
        File::open(&bed_path).map_err(|e| PipelineError::Io(format!("Opening {}: {}", bed_path.display(), e)))?;
    let mmap = Arc::new(unsafe { Mmap::map(&bed_file).map_err(|e| PipelineError::Io(e.to_string()))? });
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

    // --- 2. Orchestration: Use a scoped thread for safe producer/consumer execution ---
    let final_result: Result<(Vec<f64>, Vec<u32>), PipelineError> = thread::scope(|s| {
        let producer_handle = s.spawn({
            // Clone Arcs for the producer thread.
            let mmap = Arc::clone(&mmap);
            let prep_result = Arc::clone(&context.prep_result);
            let buffer_pool = Arc::clone(&buffer_pool);
            move || io::producer_thread(mmap, prep_result, sparse_tx, dense_tx, buffer_pool)
        });

        // Run both consumer streams in parallel on the Rayon thread pool.
        // The buffer pool is cloned again for each consumer to use for recycling.
        let (sparse_result, dense_result) = rayon::join(
            || process_sparse_stream(sparse_rx, context, Arc::clone(&buffer_pool)),
            || process_dense_stream(dense_rx, context, Arc::clone(&buffer_pool)),
        );

        // This join is critical for propagating panics from the producer.
        // It ensures we don't proceed with a partial pipeline.
        producer_handle.join().map_err(|_| {
            PipelineError::Producer("Producer thread panicked.".to_string())
        })?;

        // --- 3. Aggregate final results from both consumer streams ---
        let (mut final_scores, mut final_counts) = sparse_result?;
        let (dense_scores, dense_counts) = dense_result?;

        // Use a parallel zip to sum the dense results into the sparse results buffer.
        // This is a highly efficient, cache-friendly final reduction.
        final_scores
            .par_iter_mut()
            .zip(dense_scores)
            .for_each(|(master, partial)| *master += partial);
        final_counts
            .par_iter_mut()
            .zip(dense_counts)
            .for_each(|(master, partial)| *master += partial);

        Ok((final_scores, final_counts))
    });

    final_result
}

// ========================================================================================
//                        PIPELINE STAGE IMPLEMENTATIONS
// ========================================================================================

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
                let mut work_item = work_result?;
                batch::run_variant_major_path(
                    &work_item.data,
                    prep_result,
                    &mut acc.0,
                    &mut acc.1,
                    work_item.reconciled_variant_index,
                )?;
                // After processing, the buffer is immediately returned to the shared pool.
                work_item.data.clear();
                let _ = buffer_pool.push(work_item.data);
                Ok(acc)
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

    // Handle the case where the stream was empty.
    Ok(final_result.unwrap_or_else(|| (vec![0.0; result_size], vec![0; result_size])))
}

/// A contention-free consumer for the dense variant stream. It groups items from
/// the channel into batches, which are then processed in parallel by Rayon.
fn process_dense_stream(
    rx: Receiver<Result<WorkItem, PipelineError>>,
    context: &PipelineContext,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) -> ConsumerResult {
    let prep_result = &context.prep_result;
    let result_size = prep_result.num_people_to_score * prep_result.score_names.len();
    let bytes_per_variant = prep_result.bytes_per_variant as usize;

    let final_result = rx
        .into_iter()
        .chunks(DENSE_BATCH_SIZE) // From itertools: creates an iterator of batches.
        .into_iter()
        .map(|chunk| chunk.collect::<Result<Vec<_>, _>>()) // Collect each chunk into a Vec<WorkItem>.
        .par_bridge() // Make the iterator of batches parallel.
        .try_fold(
            || (vec![0.0f64; result_size], vec![0u32; result_size]),
            |mut acc, batch_result| {
                let batch = batch_result?;
                if batch.is_empty() {
                    return Ok(acc);
                }

                // For cache-efficiency in the compute kernel, we perform one fast, bulk
                // copy to make the variant data for this batch contiguous in memory.
                let mut concatenated_data = Vec::with_capacity(batch.len() * bytes_per_variant);
                let mut reconciled_indices = Vec::with_capacity(batch.len());
                for wi in &batch {
                    concatenated_data.extend_from_slice(&wi.data);
                    reconciled_indices.push(wi.reconciled_variant_index);
                }

                batch::run_person_major_path(
                    &concatenated_data,
                    &reconciled_indices,
                    prep_result,
                    &mut acc.0,
                    &mut acc.1,
                    &context.tile_pool,
                    &context.sparse_index_pool,
                )?;

                // After processing the entire batch, we recycle all of its buffers.
                for mut work_item in batch {
                    work_item.data.clear();
                    let _ = buffer_pool.push(work_item.data);
                }
                Ok(acc)
            },
        )
        .try_reduce(
            || (vec![0.0f64; result_size], vec![0u32; result_size]),
            |mut a, b| {
                a.0.par_iter_mut().zip(b.0).for_each(|(v_a, v_b)| *v_a += v_b);
                a.1.par_iter_mut().zip(b.1).for_each(|(v_a, v_b)| *v_a += v_b);
                Ok(a)
            },
        )?;

    Ok(final_result.unwrap_or_else(|| (vec![0.0; result_size], vec![0; result_size])))
}
