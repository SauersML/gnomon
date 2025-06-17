// ========================================================================================
//
//                            THE TACTICAL PIPELINE EXECUTOR
//
// ========================================================================================
//
// This module contains the "how" of the concurrent pipeline. It is architected to
// prevent resource-based deadlocks and resource exhaustion panics by enforcing a
// strict, semaphore-gated lifecycle for all compute tasks.

use crate::batch::{self, SparseIndexPool};
use crate::io::BedReader;
use crate::types::{
    ComputePath, DenseVariantBatch, DenseVariantBatchData, DirtyCounts, DirtyScores,
    EffectAlleleDosage, PreparationResult, ReconciledVariantIndex,
};
use cache_size;
use crossbeam_queue::ArrayQueue;
use futures::stream::{FuturesUnordered, StreamExt};
use num_cpus;
use std::error::Error;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::{self, JoinHandle};

// ========================================================================================
//                          PUBLIC API & CONTEXT
// ========================================================================================

/// The maximum number of concurrent compute tasks allowed to be in-flight. This value
/// is also used to size the `partial_result_pool`, and it determines the number of
/// permits in the `Semaphore` that gates task dispatch. It must be a value greater
/// than `PIPELINE_DEPTH` to prevent deadlocks.
const MAX_IN_FLIGHT_TASKS: usize = 32;

/// Owns shared resource pools and the final result buffers for the pipeline.
///
/// This struct acts as a state object for the entire computation. It holds sharable,
/// read-only resources (`Arc`s) and the mutable state that is exclusively owned and
/// modified by the pipeline.
pub struct PipelineContext {
    // Read-only configuration, sharable via Arc.
    pub prep_result: Arc<PreparationResult>,

    // Sharable, thread-safe resource pools.
    pub tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    pub sparse_index_pool: Arc<SparseIndexPool>,
    pub partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,

    // The final destination for all computed data, accessible after the pipeline runs.
    pub all_scores: Vec<f64>,
    pub all_missing_counts: Vec<u32>,
}

impl PipelineContext {
    /// Creates a new `PipelineContext`, allocating all necessary memory pools and
    /// final result buffers based on the blueprint from the preparation phase.
    pub fn new(prep_result: Arc<PreparationResult>) -> Self {
        let num_scores = prep_result.score_names.len();
        let result_buffer_size = prep_result.num_people_to_score * num_scores;
        let partial_result_pool = Arc::new(ArrayQueue::new(MAX_IN_FLIGHT_TASKS));
        for _ in 0..MAX_IN_FLIGHT_TASKS {
            partial_result_pool
                .push((
                    DirtyScores(vec![0.0f64; result_buffer_size]),
                    DirtyCounts(vec![0u32; result_buffer_size]),
                ))
                .unwrap();
        }

        Self {
            prep_result,
            tile_pool: Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 2)),
            sparse_index_pool: Arc::new(SparseIndexPool::new()),
            partial_result_pool,
            all_scores: vec![0.0f64; result_buffer_size],
            all_missing_counts: vec![0u32; result_buffer_size],
        }
    }
}

/// Executes the entire concurrent compute pipeline.
///
/// This is the primary public entry point into this module. It takes a mutable
/// context containing all necessary resources and a path to the PLINK data, then
/// orchestrates the I/O producer, compute dispatch, and result aggregation tasks.
/// The function returns once all processing is complete and all results have been
/// aggregated into the context's `all_scores` and `all_missing_counts` buffers.
pub async fn run(
    context: &mut PipelineContext,
    plink_prefix: &Path,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // The pipeline depth determines how many I/O requests can be in-flight.
    const PIPELINE_DEPTH: usize = 2;

    // Create the two channels that will connect the I/O producer and the main orchestrator.
    // The orchestrator sends empty buffers to the I/O task, which sends them back filled.
    let (full_buffer_tx, mut full_buffer_rx) = mpsc::channel(PIPELINE_DEPTH);
    let (empty_buffer_tx, empty_buffer_rx) = mpsc::channel(PIPELINE_DEPTH);

    // Pre-fill the pipeline with empty buffers. The I/O task will immediately
    // receive these and begin reading from the file.
    let single_variant_buffer_size = context.prep_result.bytes_per_variant as usize;
    for _ in 0..PIPELINE_DEPTH {
        empty_buffer_tx
            .send(vec![0u8; single_variant_buffer_size])
            .await?;
    }

    // Spawn the I/O producer task, moving the channel ends it owns into the task.
    let io_handle = spawn_io_producer_task(
        context.prep_result.clone(),
        plink_prefix.to_path_buf(),
        full_buffer_tx,
        empty_buffer_rx,
    );

    // Run the main orchestration loop, which receives data from the I/O task
    // and spawns compute tasks. This function now contains all dispatch, aggregation,
    // and backpressure logic.
    run_orchestration_loop(context, &mut full_buffer_rx, &empty_buffer_tx).await?;

    // Finally, await the I/O handle to ensure it has terminated cleanly and
    // propagate any errors it may have encountered.
    io_handle.await??;
    Ok(())
}

// ========================================================================================
//                             INTERNAL TYPES
// ========================================================================================

/// A self-contained, type-safe representation of a unit of computation.
///
/// This enum decouples the I/O transport mechanism (the buffer) from the compute
/// payload. By taking ownership of the necessary data, it guarantees that a compute
/// task has everything it needs and prevents I/O buffers from being accidentally
/// captured and held by long-running tasks.
enum WorkParcel {
    /// A single variant to be processed by the low-overhead variant-major path.
    Sparse {
        data: Vec<u8>,
        reconciled_variant_index: ReconciledVariantIndex,
    },
    /// A batch of dense variants to be processed by the high-throughput person-major path.
    Dense(DenseVariantBatchData),
}

// ========================================================================================
//                        PRIVATE IMPLEMENTATION
// ========================================================================================

/// The core orchestration loop that receives, dispatches, and aggregates work.
///
/// This function is the heart of the robust pipeline. It uses a `tokio::select!`
/// loop to concurrently handle two main activities:
/// 1. Receiving new variant data from the I/O producer.
/// 2. Receiving results from any completed compute task.
///
/// A `Semaphore` is used to enforce a strict limit on the number of concurrent
/// tasks, preventing resource exhaustion and providing natural backpressure.
async fn run_orchestration_loop(
    context: &mut PipelineContext,
    full_buffer_rx: &mut mpsc::Receiver<Vec<u8>>,
    empty_buffer_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- State for tracking progress through the .bed file and reconciled indices ---
    // Tracks the current row index in the full .bed file (0-based).
    let mut bed_row_cursor: u32 = 0;
    // Tracks the current index in the `prep_result.required_bim_indices` slice.
    let mut reconciled_variant_cursor: usize = 0;

    // --- State for batching and concurrency control ---
    let dense_batch_capacity =
        calculate_dense_batch_capacity(context.prep_result.bytes_per_variant);
    let mut dense_batch = DenseVariantBatch::Empty;
    let semaphore = Arc::new(Semaphore::new(MAX_IN_FLIGHT_TASKS));
    let mut in_flight_tasks = FuturesUnordered::new();

    loop {
        tokio::select! {
            // Biased selection prioritizes pulling completed work off the queue before
            // starting new work, which helps keep the pipeline flowing.
            biased;

            // --- Arm 1: Process results from completed compute tasks ---
            Some(task_result) = in_flight_tasks.next() => {
                // First, release the semaphore permit. This happens regardless of task outcome.
                semaphore.add_permits(1);

                // Unpack the result from the JoinHandle, propagating a panic.
                let (compute_result, scores, counts) = task_result?;

                // Check if the computation itself succeeded.
                if let Err(e) = compute_result {
                    // The task failed. Return the buffers to the pool before propagating the error.
                    context.partial_result_pool.push((scores, counts)).unwrap();
                    return Err(e);
                }

                // If we reach here, the task succeeded. Aggregate the results.
                for (master, &partial) in context.all_scores.iter_mut().zip(scores.deref()) {
                    *master += partial;
                }
                for (master, &partial) in context.all_missing_counts.iter_mut().zip(counts.deref()) {
                    *master += partial;
                }

                // AFTER aggregation, return the buffers to the pool for reuse.
                context.partial_result_pool.push((scores, counts)).unwrap();
            },

            // --- Arm 2: Receive new data from the I/O producer ---
            maybe_buffer = full_buffer_rx.recv() => {
                let mut filled_buffer = match maybe_buffer {
                    Some(buffer) => buffer,
                    None => break, // The I/O producer has finished.
                };

                // Check if we have processed all required variants. If so, all subsequent
                // buffers from the I/O producer are irrelevant.
                let is_relevant = if reconciled_variant_cursor < context.prep_result.required_bim_indices.len() {
                    // Get the next required variant index from the pre-computed list.
                    let next_required_bim_index = context.prep_result.required_bim_indices[reconciled_variant_cursor];
                    // Since both the .bed file and our required list are sorted, we can just
                    // check if the current .bed row is the one we're looking for.
                    bed_row_cursor == next_required_bim_index.0
                } else {
                    false
                };

                if is_relevant {
                    // This variant is needed for the calculation.
                    let reconciled_variant_index = ReconciledVariantIndex(reconciled_variant_cursor as u32);
                    let path_decision = batch::assess_path(&filled_buffer, context.prep_result.total_people_in_fam);

                    if let ComputePath::VariantMajor = path_decision {
                        // Sparse variant. First, flush any buffered dense batch to maintain order.
                        if let DenseVariantBatch::Buffering(data) = std::mem::replace(&mut dense_batch, DenseVariantBatch::Empty) {
                            let permit = semaphore.clone().acquire_owned().await.unwrap();
                            let handle = dispatch_work(WorkParcel::Dense(data), context, permit);
                            in_flight_tasks.push(handle);
                        }

                        // Now, dispatch the sparse variant.
                        let permit = semaphore.clone().acquire_owned().await.unwrap();
                        let parcel = WorkParcel::Sparse { data: filled_buffer.clone(), reconciled_variant_index };
                        let handle = dispatch_work(parcel, context, permit);
                        in_flight_tasks.push(handle);

                    } else { // PersonMajor
                        // Dense variant. Add it to the current dense batch.
                        dense_batch = match dense_batch {
                            DenseVariantBatch::Empty => {
                                let mut genotype_data = Vec::with_capacity(dense_batch_capacity);
                                genotype_data.extend_from_slice(&filled_buffer);

                                DenseVariantBatch::Buffering(DenseVariantBatchData {
                                    data: genotype_data,
                                    variant_count: 1,
                                    reconciled_variant_indices: vec![reconciled_variant_index],
                                })
                            }
                            DenseVariantBatch::Buffering(mut batch_data) => {
                                batch_data.data.extend_from_slice(&filled_buffer);
                                batch_data.reconciled_variant_indices.push(reconciled_variant_index);
                                batch_data.variant_count += 1;
                                DenseVariantBatch::Buffering(batch_data)
                            }
                        };

                        // If the dense batch is now full, dispatch it.
                        if let DenseVariantBatch::Buffering(data) = &dense_batch {
                            if data.data.len() >= dense_batch_capacity {
                               let permit = semaphore.clone().acquire_owned().await.unwrap();
                               let parcel = WorkParcel::Dense(std::mem::replace(&mut dense_batch, DenseVariantBatch::Empty).into_data());
                               let handle = dispatch_work(parcel, context, permit);
                               in_flight_tasks.push(handle);
                            }
                        }
                    }

                    // We've processed this relevant variant, so advance the cursor.
                    reconciled_variant_cursor += 1;
                    if reconciled_variant_cursor > 0 && reconciled_variant_cursor % 10000 == 0 {
                        eprintln!( "> Processed {}/{} relevant variants...", reconciled_variant_cursor, context.prep_result.num_reconciled_variants);
                    }
                }
                // If `is_relevant` was false, we do nothing with the buffer, effectively discarding it.

                // In ALL cases (relevant or not), increment the global .bed file cursor.
                bed_row_cursor += 1;

                // And in ALL cases, we must immediately recycle the I/O buffer to prevent deadlock.
                // We take the buffer, clear it to be polite, and send it back to the I/O producer.
                filled_buffer.clear();
                if empty_buffer_tx.send(filled_buffer).await.is_err() { break; }
            }
        }
    }

    // --- Finalization: Flush any remaining work ---

    // Flush the final dense batch if it's not empty.
    if let DenseVariantBatch::Buffering(data) = dense_batch {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let handle = dispatch_work(WorkParcel::Dense(data), context, permit);
        in_flight_tasks.push(handle);
    }

    // Wait for all remaining in-flight tasks to complete.
    while let Some(task_result) = in_flight_tasks.next().await {
        // Unpack the result, propagating panics.
        let (compute_result, scores, counts) = task_result?;

        // Check for compute errors.
        if let Err(e) = compute_result {
            // Even on error, we must return the buffers to the pool.
            context.partial_result_pool.push((scores, counts)).unwrap();
            return Err(e);
        }

        // On success, aggregate the final results.
        for (master, &partial) in context.all_scores.iter_mut().zip(scores.deref()) {
            *master += partial;
        }
        for (master, &partial) in context.all_missing_counts.iter_mut().zip(counts.deref()) {
            *master += partial;
        }

        // And return the final buffers to the pool.
        context.partial_result_pool.push((scores, counts)).unwrap();
    }

    Ok(())
}

/// Spawns the I/O producer task in the background.
///
/// This function is architected for maximum throughput. It spawns a single dedicated
/// OS thread to handle all blocking file I/O, preventing the async runtime from
/// being stalled. This producer's only responsibility is to read every variant from
/// the .bed file sequentially and send the raw data downstream. It performs zero
/// filtering or decision-making, ensuring a constant, predictable flow of data
/// to the orchestrator and preventing pipeline deadlocks.
fn spawn_io_producer_task(
    prep_result: Arc<PreparationResult>,
    plink_prefix: PathBuf,
    full_buffer_tx: mpsc::Sender<Vec<u8>>,
    mut empty_buffer_rx: mpsc::Receiver<Vec<u8>>,
) -> JoinHandle<Result<(), Box<dyn Error + Send + Sync>>> {
    tokio::spawn(async move {
        // task::spawn_blocking runs the provided closure on a dedicated thread pool
        // for blocking operations. We await its completion.
        match task::spawn_blocking(move || -> Result<(), Box<dyn Error + Send + Sync>> {
            // --- Setup Phase (inside the dedicated blocking thread) ---
            let bed_path = plink_prefix.with_extension("bed");
            let mut reader = BedReader::new(
                &bed_path,
                prep_result.bytes_per_variant,
                prep_result.total_variants_in_bim,
            )?;

            // --- Producer Loop ---
            // This loop will run until the orchestrator drops its receiver or we reach EOF.
            // We use `blocking_recv` because we are in a synchronous context.
            while let Some(mut buffer) = empty_buffer_rx.blocking_recv() {
                // Read the very next variant row from the memory-mapped file.
                let variant_data_opt = reader.read_next_variant(&mut buffer)?;

                if let Some(filled_buffer) = variant_data_opt {
                    // Immediately send the filled buffer to the orchestrator.
                    // If this fails, the orchestrator has shut down, so we can exit.
                    if full_buffer_tx.blocking_send(filled_buffer).is_err() {
                        break;
                    }
                } else {
                    // `None` signifies end-of-file. The producer's job is done.
                    break;
                }
            }
            Ok(())
        })
        .await
        {
            Ok(Ok(())) => Ok(()), // Blocking task succeeded.
            Ok(Err(e)) => Err(e), // Blocking task returned an error.
            Err(join_error) => Err(Box::new(join_error) as Box<dyn Error + Send + Sync>), // Blocking task panicked.
        }
    })
}

/// A unified dispatcher that takes a `WorkParcel`, spawns the appropriate compute
/// task, and returns a `JoinHandle` to its result. The provided `SemaphorePermit`
/// is consumed by the closure, ensuring it is not forgotten and is tied to the
/// lifecycle of the spawned task.
///
/// The return type is carefully structured to decouple the result of the computation
/// from the buffers used. This guarantees that buffers are always returned to the
/// orchestrator for recycling, even if the computation fails, preventing resource leaks.
fn dispatch_work(
    parcel: WorkParcel,
    context: &PipelineContext,
    permit: tokio::sync::OwnedSemaphorePermit,
) -> JoinHandle<(
    Result<(), Box<dyn Error + Send + Sync>>,
    DirtyScores,
    DirtyCounts,
)> {
    // Clone the necessary Arcs for the spawned task. This is a cheap reference count bump.
    let prep_result = context.prep_result.clone();
    let partial_result_pool = context.partial_result_pool.clone();
    let tile_pool = context.tile_pool.clone();
    let sparse_index_pool = context.sparse_index_pool.clone();

    task::spawn_blocking(move || {
        // This permit is held for the duration of the blocking task. When this
        // closure finishes, the permit is dropped, effectively releasing it
        // back to the semaphore. This happens even if the task panics.
        let _permit = permit;

        // Acquire a result buffer set. This will never panic because the semaphore
        // guarantees that a permit could only be acquired if a buffer set was available.
        let (dirty_scores, dirty_counts) = partial_result_pool.pop().unwrap();
        let mut clean_scores = dirty_scores.into_clean();
        let mut clean_counts = dirty_counts.into_clean();

        // Perform the computation, capturing the result without using `?` so that
        // we can guarantee the buffers are returned regardless of the outcome.
        let compute_result = match parcel {
            WorkParcel::Sparse {
                data,
                reconciled_variant_index,
            } => batch::run_variant_major_path(
                &data,
                &prep_result,
                &mut clean_scores,
                &mut clean_counts,
                reconciled_variant_index,
            ),
            WorkParcel::Dense(data) => batch::run_person_major_path(
                &data.data,
                &data.reconciled_variant_indices,
                &prep_result,
                &mut clean_scores,
                &mut clean_counts,
                &tile_pool,
                &sparse_index_pool,
            ),
        };

        // Return a tuple containing the computation's result AND the buffers.
        // The buffers are now considered "dirty" after being used.
        (
            compute_result,
            clean_scores.into_dirty(),
            clean_counts.into_dirty(),
        )
    })
}



/// Calculates a target batch size for the person-major path based on L3 cache size.
///
/// This heuristic aims to size the pivoted tile to fit comfortably in L3 cache, which
/// is critical for the performance of the pivot operation.
fn calculate_dense_batch_capacity(bytes_per_variant: u64) -> usize {
    const PERSON_BLOCK_SIZE: u64 = 4096; // Must match batch.rs
    const MIN_DENSE_BATCH_SIZE: usize = 1 * 1024 * 1024; // 1MB
    const MAX_DENSE_BATCH_SIZE: usize = 256 * 1024 * 1024; // 256MB

    let l3_cache_bytes = cache_size::l3_cache_size().unwrap_or(32 * 1024 * 1024);
    let max_variants_for_l3 = (l3_cache_bytes as u64) / PERSON_BLOCK_SIZE;

    ((max_variants_for_l3 * bytes_per_variant) as usize)
        .clamp(MIN_DENSE_BATCH_SIZE, MAX_DENSE_BATCH_SIZE)
}
