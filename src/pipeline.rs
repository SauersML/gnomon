// ========================================================================================
//
//                          THE TACTICAL PIPELINE EXECUTOR
//
// ========================================================================================
//
// This module contains the "how" of the concurrent pipeline. It takes a "computation
// blueprint" from the `prepare` module and executes it, managing I/O, task dispatch,
// and result aggregation. It is completely decoupled from the strategic-level logic
// in `main.rs`.

use crate::batch::{self, SparseIndexPool};
use crate::io::BedReader;
use crate::types::PreparationResult;
use crate::types::{
    ComputePath, DenseVariantBatch, DirtyCounts, DirtyScores, EffectAlleleDosage, ReconciledVariantIndex,
    PackedVariantGenotypes,
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
/// modified by the pipeline (e.g., the final result buffers and the list of spawned
/// compute tasks).
pub struct PipelineContext {
    // Read-only configuration, sharable via Arc.
    pub prep_result: Arc<PreparationResult>,

    // Sharable, thread-safe resource pools.
    tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    sparse_index_pool: Arc<SparseIndexPool>,
    kernel_input_buffer_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,

    // The final destination for all computed data, accessible after the pipeline runs.
    pub all_scores: Vec<f64>,
    pub all_missing_counts: Vec<u32>,

    // A list of all in-flight compute tasks, private to this module.
    compute_handles: Vec<task::JoinHandle<Result<ComputeTaskResult, Box<dyn Error + Send + Sync>>>>,
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
        let kernel_input_buffer_pool = Arc::new(ArrayQueue::new(MAX_IN_FLIGHT_TASKS));
        for _ in 0..MAX_IN_FLIGHT_TASKS {
            kernel_input_buffer_pool
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
            kernel_input_buffer_pool,
            all_scores: vec![0.0f64; result_buffer_size],
            all_missing_counts: vec![0u32; result_buffer_size],
            compute_handles: Vec::with_capacity(256),
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
    let (full_buffer_tx, mut full_buffer_rx) = mpsc::channel(PIPELINE_DEPTH);
    let (empty_buffer_tx, empty_buffer_rx) = mpsc::channel(PIPELINE_DEPTH);

    // Pre-fill the pipeline with empty buffers. The I/O task will immediately
    // receive these and begin reading from the file.
    let single_variant_buffer_size = context.prep_result.bytes_per_variant as usize;
    for _ in 0..PIPELINE_DEPTH {
        empty_buffer_tx
            .send(IoCommand::Read(vec![0u8; single_variant_buffer_size]))
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
    aggregate_results(context, &empty_buffer_tx).await?;

    // Finally, await the I/O handle to ensure it has terminated cleanly and
    // propagate any errors it may have encountered.
    io_handle.await??;
    Ok(())
}

// ========================================================================================
//                             INTERNAL TYPES
// ========================================================================================

/// The unified result returned by any compute task, containing dirty buffers.
struct ComputeTaskResult {
    scores: DirtyScores,
    counts: DirtyCounts,
    // The I/O buffer from a sparse SNP task, to be recycled. `None` for dense batches.
    recycled_buffer: Option<Vec<u8>>,
}

/// A command sent from the orchestrator to the I/O producer thread.
enum IoCommand {
    /// A command to read the next SNP into the provided buffer.
    Read(Vec<u8>),
    /// A command to gracefully shut down the I/O thread.
    Shutdown,
}

// ========================================================================================
//                        PRIVATE IMPLEMENTATION
// ========================================================================================

/// Spawns the I/O producer task in the background.
fn spawn_io_producer_task(
    prep_result: Arc<PreparationResult>,
    plink_prefix: PathBuf,
    full_buffer_tx: mpsc::Sender<PackedVariantGenotypes>,
    mut empty_buffer_rx: mpsc::Receiver<IoCommand>,
) -> task::JoinHandle<Result<(), Box<dyn Error + Send + Sync>>> {
    tokio::spawn(async move {
        // --- One-time setup within the spawned task ---
        let bed_path = plink_prefix.with_extension("bed");
        let mut reader = match BedReader::new(
            &bed_path,
            prep_result.bytes_per_snp,
            prep_result.total_snps_in_bim,
        ) {
            Ok(r) => r,
            Err(e) => return Err(Box::new(e) as Box<dyn Error + Send + Sync>),
        };

        let mut bed_row_cursor: u32 = 0;
        let mut required_indices_cursor: usize = 0;

        // --- Main I/O Producer Loop ---
        // This loop is driven by the orchestrator, which sends commands.
        'producer: while let Some(command) = empty_buffer_rx.recv().await {
            let mut buffer_to_fill = match command {
                IoCommand::Read(buffer) => buffer,
                IoCommand::Shutdown => break 'producer,
            };

            // Perform the synchronous, potentially blocking file I/O in a dedicated thread
            // to avoid stalling the async runtime.
            let (new_reader, filled_buffer_opt) =
                match task::spawn_blocking(move || {
                    let variant_data_opt = reader.read_next_variant(&mut buffer_to_fill)?;
                    Ok::<_, io::Error>((reader, variant_data_opt))
                })
                .await
                {
                    Ok(Ok(res)) => res,
                    Ok(Err(e)) => return Err(Box::new(e)), // I/O error from within spawn_blocking
                    Err(e) => return Err(Box::new(e)), // JoinError (e.g., task panicked)
                };

            // The reader state is updated after the blocking operation completes.
            reader = new_reader;

            if let Some(filled_buffer) = filled_buffer_opt {
                // Check if the SNP we just read is one we actually need for the calculation.
                let is_relevant = prep_result
                    .required_bim_indices
                    .get(required_indices_cursor)
                    .map_or(false, |&req_idx| req_idx.0 == bed_row_cursor);

                if is_relevant {
                    // This SNP is needed. Send it to the orchestrator for processing.
                    required_indices_cursor += 1;
                    if full_buffer_tx.send(PackedVariantGenotypes(filled_buffer)).await.is_err() {
                        // The orchestrator has hung up (channel closed). Shut down.
                        break 'producer;
                        
                    }
                } else {
                    // This SNP is not relevant. The buffer and its data are simply
                    // dropped. The orchestrator is responsible for sending a new
                    // `IoCommand::Read` to keep the pipeline full.
                    drop(filled_buffer);
                }
                bed_row_cursor += 1;
            } else {
                // `read_next_snp` returned `None`, indicating End-Of-File.
                break 'producer;
            }
        }
        Ok(())
    })
}

/// The core orchestration loop that receives and dispatches SNPs.
async fn run_orchestration_loop(
    context: &mut PipelineContext,
    full_buffer_rx: &mut mpsc::Receiver<PackedVariantGenotypes>,
    empty_buffer_tx: &mpsc::Sender<IoCommand>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut required_indices_cursor: usize = 0;
    let dense_batch_capacity = calculate_dense_batch_capacity(context.prep_result.bytes_per_variant);
    let mut dense_batch = DenseVariantBatch::new_empty(dense_batch_capacity);

    // This is the main loop of the application's concurrent phase.
    'orchestrator: while let Some(packed_genotypes) = full_buffer_rx.recv().await {
        // The orchestrator receives a buffer only for SNPs that are required. Since the
        // I/O producer reads them in the sorted order defined by `required_bim_indices`,
        // the `required_indices_cursor` directly corresponds to the row index in the
        // dense compute matrices.
        let reconciled_variant_index = ReconciledVariantIndex(required_indices_cursor as u32);
        required_indices_cursor += 1;

        // The "brain" of the adaptive engine: make the dispatch decision.
        let path_decision =
            batch::assess_path(&packed_genotypes.0, context.prep_result.total_people_in_fam);

        match path_decision {
            ComputePath::SnpMajor => {
                // To preserve processing order, we must flush any pending dense SNPs
                // before processing this sparse one.
                dispatch_and_clear_dense_batch(&mut dense_batch, context);

                // Now, dispatch the sparse SNP for immediate computation.
                let handle = dispatch_variant_major_path(
                    packed_genotypes,
                    reconciled_variant_index,
                    context.prep_result.clone(),
                    context.kernel_input_buffer_pool.clone(),
                );
                context.compute_handles.push(handle);
            }
            ComputePath::PersonMajor => {
                // Add the dense SNP's data and metadata to the current batch.
                dense_batch.data.extend_from_slice(&packed_genotypes.0);
                dense_batch.reconciled_variant_indices.push(reconciled_variant_index);

                // Immediately recycle the now-empty buffer by sending a `Read`
                // command back to the I/O producer.
                if empty_buffer_tx
                    .send(IoCommand::Read(packed_genotypes.0))
                    .await
                    .is_err()
                {
                    // I/O thread has shut down unexpectedly. Terminate the loop.
                    break 'orchestrator;
                }

                // If the batch has reached its target capacity, dispatch it.
                if dense_batch.data.len() >= dense_batch.data.capacity() {
                    dispatch_and_clear_dense_batch(&mut dense_batch, context);
                }
            }
        }

        if required_indices_cursor > 0 && required_indices_cursor % 10000 == 0 {
            eprintln!(
                "> Processed {}/{} relevant variants...",
                required_indices_cursor, context.prep_result.num_reconciled_variants
            );
        }
    }

    // The loop has finished, meaning the I/O producer has closed its channel.
    // We must flush any final, partially-filled dense batch.
    dispatch_and_clear_dense_batch(&mut dense_batch, context);

    // Finally, tell the I/O producer to shut down gracefully.
    // It might have already terminated, so we ignore any potential send error.
    let _ = empty_buffer_tx.send(IoCommand::Shutdown).await;

    Ok(())
}

/// Awaits all in-flight compute tasks and aggregates their results.
async fn aggregate_results(
    context: &mut PipelineContext,
    empty_buffer_tx: &mpsc::Sender<IoCommand>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Take ownership of the handles, leaving an empty Vec in the context.
    for handle in std::mem::take(&mut context.compute_handles) {
        // Await the task's completion, propagating JoinErrors (e.g., panics) with `?`.
        let task_result = handle.await?;

        // Now match on the business-logic `Result` returned by the task itself.
        match task_result {
            Ok(result) => {
                // 1. Aggregate scores and counts into the master buffers.
                // This `zip` iterator is highly performant and likely auto-vectorized.
                for (master, &partial) in context.all_scores.iter_mut().zip(&result.scores) {
                    *master += partial;
                }
                for (master, &partial) in context.all_missing_counts.iter_mut().zip(&result.counts)
                {
                    *master += partial;
                }

                // 2. Return the large partial result buffers to the shared pool for reuse.
                context
                    .kernel_input_buffer_pool
                    .push((result.scores, result.counts))
                    .unwrap();

                // 3. If the task was for a sparse SNP, recycle its small I/O buffer.
                if let Some(recycled_buffer) = result.recycled_buffer {
                    // Use a non-blocking `try_send`. If the I/O task has already shut down,
                    // the channel will be closed, and this will return an `Err`. We gracefully
                    // ignore this error, as the buffer is no longer needed.
                    let _ = empty_buffer_tx.try_send(IoCommand::Read(recycled_buffer));
                }
            }
            Err(e) => {
                // A task returned a business-logic error. Propagate it immediately.
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Dispatches a dense batch and adds its handle to the context.
fn dispatch_and_clear_dense_batch(
    dense_batch: &mut DenseVariantBatch,
    context: &mut PipelineContext,
) {
    if dense_batch.reconciled_variant_indices.is_empty() {
        return;
    }

    // Take ownership of the batch and reset the original via a highly efficient swap.
    let batch_to_process = std::mem::replace(
        dense_batch,
        DenseVariantBatch::new_empty(dense_batch.data.capacity()),
    );

    // We clone the Arcs from the context, which is a cheap reference count bump.
    let handle = dispatch_person_major_batch(
        batch_to_process,
        context.prep_result.clone(),
        context.tile_pool.clone(),
        context.sparse_index_pool.clone(),
        context.kernel_input_buffer_pool.clone(),
    );

    context.compute_handles.push(handle);
}

/// Spawns a blocking task for the person-major path.
fn dispatch_person_major_batch(
    batch_to_process: DenseVariantBatch,
    prep_result: Arc<PreparationResult>,
    tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    sparse_index_pool: Arc<SparseIndexPool>,
    kernel_input_buffer_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,
) -> task::JoinHandle<Result<ComputeTaskResult, Box<dyn Error + Send + Sync>>> {
    task::spawn_blocking(move || {
        // Acquire a set of partial result buffers from the pool.
        let (dirty_scores, dirty_counts) = kernel_input_buffer_pool.pop().unwrap();

        // Transition the buffers to the "Clean" state, zeroing them.
        let mut clean_scores = dirty_scores.into_clean();
        let mut clean_counts = dirty_counts.into_clean();

        // Dispatch to the synchronous, CPU-bound compute function.
        batch::run_person_major_path(
            &batch_to_process.data,
            &batch_to_process.reconciled_variant_indices,
            &prep_result,
            &mut clean_scores,
            &mut clean_counts,
            &tile_pool,
            &sparse_index_pool,
        )?;

        // Package the results into the unified result type. The large `data`
        // buffer from `batch_to_process` is NOT recycled; it is dropped here.
        let result = ComputeTaskResult {
            scores: clean_scores.into_dirty(),
            counts: clean_counts.into_dirty(),
            recycled_buffer: None,
        };

        Ok(result)
    })
}

/// Spawns a blocking task for the SNP-major path.
fn dispatch_variant_major_path(
    packed_genotypes: PackedVariantGenotypes,
    reconciled_variant_index: ReconciledVariantIndex,
    prep_result: Arc<PreparationResult>,
    kernel_input_buffer_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,
) -> task::JoinHandle<Result<ComputeTaskResult, Box<dyn Error + Send + Sync>>> {
    task::spawn_blocking(move || {
        let (dirty_scores, dirty_counts) = kernel_input_buffer_pool.pop().unwrap();
        let mut clean_scores = dirty_scores.into_clean();
        let mut clean_counts = dirty_counts.into_clean();

        batch::run_variant_major_path(
            &packed_genotypes.0,
            &prep_result,
            &mut clean_scores,
            &mut clean_counts,
            reconciled_variant_index,
        )?;

        // The small I/O buffer is passed back for recycling.
        let result = ComputeTaskResult {
            scores: clean_scores.into_dirty(),
            counts: clean_counts.into_dirty(),
            recycled_buffer: Some(packed_genotypes.0),
        };
        Ok(result)
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

    ((max_variants_for_l3 * bytes_per_variant) as usize).clamp(MIN_DENSE_BATCH_SIZE, MAX_DENSE_BATCH_SIZE)
}
