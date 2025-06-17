// ========================================================================================
//
//               THE TACTICAL PIPELINE EXECUTOR
//
// ========================================================================================
//
// This module contains the "how" of the concurrent pipeline. It is architected to
// prevent resource-based deadlocks by enforcing a strict and symmetrical lifecycle for
// all I/O buffers.

use crate::batch::{self, KernelInputBufferPool, SparseIndexPool};
use crate::io::BedReader;
use crate::types::{
    ComputePath, DenseVariantBatch, DenseVariantBatchData, DirtyCounts, DirtyScores,
    EffectAlleleDosage, PreparationResult, ReconciledVariantIndex,
};
use cache_size;
use crossbeam_queue::ArrayQueue;
use num_cpus;
use std::error::Error;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task;

// ========================================================================================
//                          PUBLIC API & CONTEXT
// ========================================================================================

/// Owns shared resource pools and the final result buffers for the pipeline.
///
/// This struct acts as a state object for the entire computation. It holds sharable,
/// read-only resources (`Arc`s) and the mutable state that is exclusively owned and
/// modified by the pipeline.
pub struct PipelineContext {
    // Read-only configuration, sharable via Arc.
    pub prep_result: Arc<PreparationResult>,

    // Sharable, thread-safe resource pools.
    tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    sparse_index_pool: Arc<SparseIndexPool>,
    kernel_input_buffer_pool: Arc<KernelInputBufferPool>,
    partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,

    // The final destination for all computed data, accessible after the pipeline runs.
    pub all_scores: Vec<f64>,
    pub all_missing_counts: Vec<u32>,

    // A list of all in-flight compute tasks, private to this module.
    join_handles:
        Vec<task::JoinHandle<Result<(DirtyScores, DirtyCounts), Box<dyn Error + Send + Sync>>>>,
}

impl PipelineContext {
    /// Creates a new `PipelineContext`, allocating all necessary memory pools and
    /// final result buffers based on the blueprint from the preparation phase.
    pub fn new(prep_result: Arc<PreparationResult>) -> Self {
        let num_scores = prep_result.score_names.len();
        let result_buffer_size = prep_result.num_people_to_score * num_scores;

        // The number of parallel compute tasks can be higher than the I/O depth.
        // Size the pool to avoid deadlocks. It must be at least as large as the
        // maximum number of tasks that can be running concurrently.
        const MAX_IN_FLIGHT_TASKS: usize = 32;
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
            kernel_input_buffer_pool: Arc::new(KernelInputBufferPool::new()),
            partial_result_pool,
            all_scores: vec![0.0f64; result_buffer_size],
            all_missing_counts: vec![0u32; result_buffer_size],
            join_handles: Vec::with_capacity(256),
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
    // and spawns compute tasks.
    run_orchestration_loop(context, &mut full_buffer_rx, &empty_buffer_tx).await?;

    // After the orchestration loop finishes, await all spawned compute tasks
    // and aggregate their results into the context's master buffers.
    aggregate_results(context).await?;

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
/// captured and held by long-running tasks, which was the source of the deadlock.
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

/// Spawns the I/O producer task in the background.
fn spawn_io_producer_task(
    prep_result: Arc<PreparationResult>,
    plink_prefix: PathBuf,
    full_buffer_tx: mpsc::Sender<Vec<u8>>,
    mut empty_buffer_rx: mpsc::Receiver<Vec<u8>>,
) -> task::JoinHandle<Result<(), Box<dyn Error + Send + Sync>>> {
    tokio::spawn(async move {
        // --- One-time setup within the spawned task ---
        let bed_path = plink_prefix.with_extension("bed");
        let mut reader = match BedReader::new(
            &bed_path,
            prep_result.bytes_per_variant,
            prep_result.total_variants_in_bim,
        ) {
            Ok(r) => r,
            Err(e) => return Err(Box::new(e) as Box<dyn Error + Send + Sync>),
        };

        let mut bed_row_cursor: u32 = 0;
        let mut required_indices_cursor: usize = 0;

        // --- Main I/O Producer Loop ---
        // This loop is driven by the orchestrator sending an empty buffer to fill.
        'producer: while let Some(mut buffer) = empty_buffer_rx.recv().await {
            // This inner loop ensures we hold onto the buffer until we find a relevant
            // variant to send to the orchestrator.
            'read_loop: loop {
                let (new_reader, filled_buffer_opt) = match task::spawn_blocking(move || {
                    let variant_data_opt = reader.read_next_variant(&mut buffer)?;
                    Ok::<_, io::Error>((reader, variant_data_opt))
                })
                .await
                {
                    Ok(Ok(res)) => res,
                    Ok(Err(e)) => return Err(Box::new(e)),
                    Err(e) => return Err(Box::new(e)),
                };

                reader = new_reader;

                if let Some(filled_buffer) = filled_buffer_opt {
                    // We must regain ownership of the buffer to reuse it in the next iteration.
                    buffer = filled_buffer;

                    let is_relevant = prep_result
                        .required_bim_indices
                        .get(required_indices_cursor)
                        .map_or(false, |&req_idx| req_idx.0 == bed_row_cursor);

                    // We must advance the cursor *after* the check to stay in sync.
                    bed_row_cursor += 1;

                    if is_relevant {
                        required_indices_cursor += 1;
                        if full_buffer_tx.send(buffer).await.is_err() {
                            // The orchestrator has hung up. Shut down.
                            break 'producer;
                        }
                        // We have successfully sent the buffer, so we must now break the
                        // inner loop to await a new empty buffer from the orchestrator.
                        break 'read_loop;
                    } else {
                        // This variant is not relevant. The loop continues, reusing the
                        // *same buffer* to read the next variant from the file.
                        continue 'read_loop;
                    }
                } else {
                    // End-Of-File reached. Terminate the producer task.
                    break 'producer;
                }
            }
        }
        Ok(())
    })
}

/// The core orchestration loop that receives and dispatches variants.
/// This function is responsible for managing the `DenseVariantBatch` state machine
/// and ensuring the I/O buffer pipeline remains full to prevent deadlocks.
async fn run_orchestration_loop(
    context: &mut PipelineContext,
    full_buffer_rx: &mut mpsc::Receiver<Vec<u8>>,
    empty_buffer_tx: &mpsc::Sender<Vec<u8>>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut required_indices_cursor: usize = 0;
    let dense_batch_capacity =
        calculate_dense_batch_capacity(context.prep_result.bytes_per_variant);
    let mut dense_batch = DenseVariantBatch::Empty;

    'orchestrator: while let Some(filled_buffer) = full_buffer_rx.recv().await {
        let reconciled_variant_index = ReconciledVariantIndex(required_indices_cursor as u32);
        required_indices_cursor += 1;

        let path_decision =
            batch::assess_path(&filled_buffer, context.prep_result.total_people_in_fam);

        match path_decision {
            ComputePath::VariantMajor => {
                // To preserve ordering, we must flush any pending dense batch first.
                dense_batch = dispatch_work(dense_batch, context);

                // Create a work parcel by cloning the data from the I/O buffer.
                // This is a small cost for the much less frequent sparse path,
                // and it guarantees the I/O buffer can be immediately recycled.
                let parcel = WorkParcel::Sparse {
                    data: filled_buffer.clone(),
                    reconciled_variant_index,
                };
                dispatch_work(parcel, context);
            }
            ComputePath::PersonMajor => {
                // This match handles the state transitions for the dense batch.
                dense_batch = match dense_batch {
                    DenseVariantBatch::Empty => {
                        let mut data = Vec::with_capacity(dense_batch_capacity);
                        data.extend_from_slice(&filled_buffer);
                        let batch_data = DenseVariantBatchData {
                            data,
                            reconciled_variant_indices: vec![reconciled_variant_index],
                        };
                        DenseVariantBatch::Buffering(batch_data)
                    }
                    DenseVariantBatch::Buffering(mut batch_data) => {
                        batch_data.data.extend_from_slice(&filled_buffer);
                        batch_data
                            .reconciled_variant_indices
                            .push(reconciled_variant_index);
                        DenseVariantBatch::Buffering(batch_data)
                    }
                };

                // After adding, check if the batch is now full and needs to be dispatched.
                if let DenseVariantBatch::Buffering(data) = &dense_batch {
                    if data.data.len() >= dense_batch_capacity {
                        dense_batch = dispatch_work(dense_batch, context);
                    }
                }
            }
        }

        // Symmetrical I/O Buffer Management: In all cases, the data from the filled
        // buffer has been handled (either cloned or copied). The buffer is now free
        // to be immediately returned to the I/O producer to be filled again.
        // This is the key to preventing the deadlock.
        if empty_buffer_tx.send(filled_buffer).await.is_err() {
            break 'orchestrator;
        }

        if required_indices_cursor > 0 && required_indices_cursor % 10000 == 0 {
            eprintln!(
                "> Processed {}/{} relevant variants...",
                required_indices_cursor, context.prep_result.num_reconciled_variants
            );
        }
    }

    // The I/O channel has closed. We must flush any final, partially-filled dense batch.
    dispatch_work(dense_batch, context);

    // The I/O producer will terminate automatically when the `empty_buffer_tx`
    // is dropped at the end of this function.

    Ok(())
}

/// Awaits all in-flight compute tasks and aggregates their results.
async fn aggregate_results(
    context: &mut PipelineContext,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Take ownership of the handles, leaving an empty Vec in the context.
    for handle in std::mem::take(&mut context.join_handles) {
        // Await the task's completion, propagating JoinErrors (e.g., panics) with `?`.
        let task_result = handle.await?;

        // Now match on the business-logic `Result` returned by the task itself.
        match task_result {
            Ok((scores, counts)) => {
                // 1. Aggregate scores and counts into the master buffers.
                // This `zip` iterator is highly performant and likely auto-vectorized.
                for (master, &partial) in context.all_scores.iter_mut().zip(&scores) {
                    *master += partial;
                }
                for (master, &partial) in context.all_missing_counts.iter_mut().zip(&counts)
                {
                    *master += partial;
                }

                // 2. Return the large partial result buffers to the shared pool for reuse.
                context
                    .partial_result_pool
                    .push((scores, counts))
                    .unwrap();

                // 3. I/O buffer recycling is no longer handled here. It is managed
                //    synchronously by the orchestrator, preventing deadlocks.
            }
            Err(e) => {
                // A task returned a business-logic error. Propagate it immediately.
                return Err(e);
            }
        }
    }
    Ok(())
}

/// A unified dispatcher that takes a `WorkParcel` or a `DenseVariantBatch`,
/// spawns the appropriate compute task, and manages the lifecycle of the work.
fn dispatch_work(
    work: impl Into<Option<WorkParcel>>,
    context: &mut PipelineContext,
) -> DenseVariantBatch {
    let maybe_parcel = work.into();
    if maybe_parcel.is_none() {
        return DenseVariantBatch::Empty;
    }
    let parcel = maybe_parcel.unwrap();

    // Clone the necessary Arcs for the spawned task. This is a cheap reference count bump.
    let prep_result = context.prep_result.clone();
    let partial_result_pool = context.partial_result_pool.clone();

    let handle = match parcel {
        WorkParcel::Sparse {
            data,
            reconciled_variant_index,
        } => task::spawn_blocking(move || {
            let (dirty_scores, dirty_counts) = partial_result_pool.pop().unwrap();
            let mut clean_scores = dirty_scores.into_clean();
            let mut clean_counts = dirty_counts.into_clean();
            batch::run_variant_major_path(
                &data,
                &prep_result,
                &mut clean_scores,
                &mut clean_counts,
                reconciled_variant_index,
            )?;
            Ok((clean_scores.into_dirty(), clean_counts.into_dirty()))
        }),
        WorkParcel::Dense(data) => {
            let tile_pool = context.tile_pool.clone();
            let sparse_index_pool = context.sparse_index_pool.clone();
            let kernel_input_buffer_pool = context.kernel_input_buffer_pool.clone();
            task::spawn_blocking(move || {
                let (dirty_scores, dirty_counts) = partial_result_pool.pop().unwrap();
                let mut clean_scores = dirty_scores.into_clean();
                let mut clean_counts = dirty_counts.into_clean();
                batch::run_person_major_path(
                    &data.data,
                    &data.reconciled_variant_indices,
                    &prep_result,
                    &mut clean_scores,
                    &mut clean_counts,
                    &tile_pool,
                    &sparse_index_pool,
                    &kernel_input_buffer_pool,
                )?;
                Ok((clean_scores.into_dirty(), clean_counts.into_dirty()))
            })
        }
    };

    context.join_handles.push(handle);
    // If we dispatched a dense batch, we return an empty one to reset the state.
    // If we dispatched a sparse one, the original batch was already empty.
    DenseVariantBatch::Empty
}

// Implement Into<Option<WorkParcel>> for DenseVariantBatch to use it in the dispatcher.
impl From<DenseVariantBatch> for Option<WorkParcel> {
    fn from(batch: DenseVariantBatch) -> Self {
        match batch {
            DenseVariantBatch::Empty => None,
            DenseVariantBatch::Buffering(data) => Some(WorkParcel::Dense(data)),
        }
    }
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
