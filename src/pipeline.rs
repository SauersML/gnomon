/// The unified result returned by any compute task.
struct ComputeTaskResult {
    scores: DirtyScores,
    counts: DirtyCounts,
    // The buffer from a sparse SNP task, to be recycled. `None` for dense batches.
    recycled_buffer: Option<Vec<u8>>,
}

/// Represents a command sent from the main orchestrator to the I/O producer thread.
/// Using an enum makes the control flow explicit and type-safe.
enum IoCommand {
    /// A command to read the next SNP into the provided buffer.
    Read(Vec<u8>),
    /// A command to gracefully shut down the I/O thread.
    Shutdown,
}

/// Owns shared resource pools and final result buffers for the pipeline.
///
/// This struct acts as a state object for the main orchestrator thread. It holds
/// sharable, read-only resources (`Arc`s) and the mutable state that is exclusively
/// owned and modified by the orchestrator (e.g., the final result buffers and the
/// list of spawned compute tasks).
///
/// Crucially, it does NOT own the communication channels, which are created and
/// managed by the `run_pipeline` function to enforce a clear ownership model
/// and data flow between asynchronous tasks.
struct PipelineContext {
    // Read-only configuration and resource pools, shared via Arc.
    prep_result: Arc<prepare::PreparationResult>,
    tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    sparse_index_pool: Arc<SparseIndexPool>,
    partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,

    // The final destination for all computed data, owned exclusively by the orchestrator.
    all_scores: Vec<f64>,
    all_missing_counts: Vec<u32>,

    // A list of all in-flight compute tasks.
    compute_handles: Vec<task::JoinHandle<Result<ComputeTaskResult, Box<dyn Error + Send + Sync>>>>,
}

impl PipelineContext {
    /// Creates a new `PipelineContext` by performing all necessary resource allocations.
    /// This function cleanly encapsulates the "Phase 3" setup logic from the original
    /// monolithic `main` function.
    fn new(prep_result: Arc<prepare::PreparationResult>) -> Self {
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
            partial_result_pool,
            all_scores: vec![0.0f64; result_buffer_size],
            all_missing_counts: vec![0u32; result_buffer_size],
            compute_handles: Vec::with_capacity(256),
        }
    }
}



/// **Helper 2:** The primary async orchestrator.
///
/// This function sets up and runs the entire concurrent pipeline. It owns the
/// communication channels, delegating the correct ends to the I/O task and the
/// main orchestration loop. It ensures all concurrent work is complete before
/// returning.
async fn run_pipeline(
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
    let single_snp_buffer_size = context.prep_result.bytes_per_snp as usize;
    for _ in 0..PIPELINE_DEPTH {
        empty_buffer_tx
            .send(IoCommand::Read(vec![0u8; single_snp_buffer_size]))
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

/// **Helper:** Spawns the I/O producer task in the background.
///
/// This function is the "factory" for the I/O producer thread. It's responsible for:
/// 1. Creating the `SnpReader` to access the `.bed` file.
/// 2. Spawning a new asynchronous `tokio` task.
/// 3. Moving all necessary state (channel ends, config) into the task.
/// 4. Returning a `JoinHandle` to the calling function, allowing it to await the
///    I/O task's completion and handle any errors.
///
/// The spawned task's core logic is a loop that:
/// - Awaits a command from the orchestrator via the `empty_buffer_rx` channel.
/// - Upon receiving a `Read` command with an empty buffer, it uses `spawn_blocking`
///   to perform a synchronous file read without blocking the `tokio` runtime.
/// - It filters the read SNP, sending only relevant data to the orchestrator via
///   the `full_buffer_tx` channel. Irrelevant SNP buffers are simply dropped,
///   and the task waits for a new `Read` command to replenish the pipeline.
/// - Upon receiving a `Shutdown` command, it terminates gracefully.
fn spawn_io_producer_task(
    prep_result: Arc<prepare::PreparationResult>,
    plink_prefix: PathBuf,
    full_buffer_tx: mpsc::Sender<SnpDataBuffer>,
    mut empty_buffer_rx: mpsc::Receiver<IoCommand>,
) -> task::JoinHandle<Result<(), Box<dyn Error + Send + Sync>>> {
    tokio::spawn(async move {
        // --- One-time setup within the spawned task ---
        let bed_path = plink_prefix.with_extension("bed");
        let mut reader = match SnpReader::new(
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
                    let snp_data_opt = reader.read_next_snp(&mut buffer_to_fill)?;
                    // The filled buffer is returned inside the Option, while the original
                    // buffer's (now empty) allocation is returned outside.
                    Ok::<_, io::Error>((reader, snp_data_opt))
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
                    if full_buffer_tx.send(SnpDataBuffer(filled_buffer)).await.is_err() {
                        // The orchestrator has hung up (channel closed). Shut down.
                        break 'producer;
                    }
                } else {
                    // This SNP is not relevant. The buffer and its data are simply
                    // dropped. The orchestrator is responsible for sending a new
                    // `IoCommand::Read` to keep the pipeline full. This creates a
                    // robust, single-direction flow for buffer management.
                    drop(filled_buffer);
                }
                bed_row_cursor += 1;
            } else {
                // `read_next_snp` returned `None`, indicating End-Of-File.
                break 'producer';
            }
        }
        Ok(())
    })
}

/// **Helper:** The core orchestration loop that receives and dispatches SNPs.
///
/// This function is the "Consumer-Dispatcher" of the pipeline. It runs in a loop,
/// receiving single, relevant SNPs from the I/O producer. For each SNP, it performs
/// the following actions:
///
/// 1.  Calls `batch::assess_path` to make a fast, data-driven decision on whether
///     the SNP is "dense" or "sparse".
/// 2.  **If Sparse:** It first dispatches any pending batch of dense SNPs to preserve
///     order, then immediately dispatches the sparse SNP for computation. The compute
///     task's `JoinHandle` is added to the context.
/// 3.  **If Dense:** It appends the SNP's data to a growing `DenseSnpBatch`. The SNP's
///     small buffer is immediately recycled. If the batch reaches its target capacity,
///     it is dispatched for computation and its `JoinHandle` is added to the context.
///
/// This function never `await`s a compute task, ensuring the orchestration loop
/// is never blocked and can continuously process the incoming stream of SNPs to
/// maximize pipeline throughput.
async fn run_orchestration_loop(
    context: &mut PipelineContext,
    full_buffer_rx: &mut mpsc::Receiver<SnpDataBuffer>,
    empty_buffer_tx: &mpsc::Sender<IoCommand>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut required_indices_cursor: usize = 0;

    // --- Determine a Target Batch Size for the Person-Major Path ---
    // This is calculated once, based on L3 cache size, to optimize the pivot operation.
    let dense_batch_capacity = {
        const PERSON_BLOCK_SIZE: u64 = 4096; // Must match batch.rs
        const MIN_DENSE_BATCH_SIZE: usize = 1 * 1024 * 1024; // 1MB
        const MAX_DENSE_BATCH_SIZE: usize = 256 * 1024 * 1024; // 256MB

        let l3_cache_bytes = cache_size::l3_cache_size().unwrap_or(32 * 1024 * 1024);
        let bytes_per_snp = context.prep_result.bytes_per_snp;
        let max_snps_for_l3 = (l3_cache_bytes as u64) / PERSON_BLOCK_SIZE;
        ((max_snps_for_l3 * bytes_per_snp) as usize)
            .clamp(MIN_DENSE_BATCH_SIZE, MAX_DENSE_BATCH_SIZE)
    };
    let mut dense_batch = DenseSnpBatch::new_empty(dense_batch_capacity);

    // This is the main loop of the application's concurrent phase.
    'orchestrator: while let Some(snp_buffer) = full_buffer_rx.recv().await {
        // Get the metadata for the SNP we just received.
        let matrix_row_index = context.prep_result.required_bim_indices[required_indices_cursor];
        required_indices_cursor += 1;

        // The "brain" of the adaptive engine: make the dispatch decision.
        let path_decision =
            batch::assess_path(&snp_buffer.0, context.prep_result.total_people_in_fam);

        match path_decision {
            ComputePath::SnpMajor => {
                // To preserve processing order, we must flush any pending dense SNPs
                // before processing this sparse one.
                dispatch_and_clear_dense_batch(&mut dense_batch, context);

                // Now, dispatch the sparse SNP for immediate computation.
                let handle = dispatch_snp_major_path(
                    snp_buffer,
                    matrix_row_index,
                    context.prep_result.clone(),
                    context.partial_result_pool.clone(),
                );
                context.compute_handles.push(handle);
            }
            ComputePath::PersonMajor => {
                // Add the dense SNP's data and metadata to the current batch.
                dense_batch.data.extend_from_slice(&snp_buffer.0);
                dense_batch.metadata.push(matrix_row_index);

                // Immediately recycle the now-empty buffer by sending a `Read`
                // command back to the I/O producer.
                if empty_buffer_tx
                    .send(IoCommand::Read(snp_buffer.0))
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

/// **Helper:** Awaits all in-flight compute tasks and aggregates their results.
///
/// This function is the synchronization point of the pipeline. It iterates through
/// all spawned `JoinHandle`s stored in the context, awaits their completion, and
/// performs three critical actions for each result:
///
/// 1.  **Aggregates** the partial scores and counts into the master result buffers.
///     This is done with efficient, auto-vectorizable iterators.
/// 2.  **Recycles** the large partial result vectors (`DirtyScores`, `DirtyCounts`)
///     back into their shared pool to be reused by future tasks.
/// 3.  **Recycles** the small `SnpDataBuffer` from sparse tasks by sending it back
///     to the I/O producer, if it is still running.
///
/// It robustly handles potential task panics and gracefully handles the case
/// where the I/O producer has already shut down.
async fn aggregate_results(
    context: &mut PipelineContext,
    empty_buffer_tx: &mpsc::Sender<IoCommand>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Take ownership of the handles, leaving an empty Vec in the context.
    // This is an efficient O(1) swap.
    for handle in std::mem::take(&mut context.compute_handles) {
        // Await the task's completion. The first `?` handles task panics (JoinError),
        // and the second `?` handles business-logic errors returned by the task itself.
        match handle.await? {
            Ok(result) => {
                // 1. Aggregate scores and counts into the master buffers.
                // This `zip` iterator is highly performant and will likely be auto-vectorized.
                for (master, &partial) in context.all_scores.iter_mut().zip(&result.scores) {
                    *master += partial;
                }
                for (master, &partial) in context.all_missing_counts.iter_mut().zip(&result.counts)
                {
                    *master += partial;
                }

                // 2. Return the large partial result buffers to the shared pool for reuse.
                // This unwrap is safe because we pop one set of buffers for every task we
                // spawn, so the pool cannot be full when we push them back.
                context
                    .partial_result_pool
                    .push((result.scores, result.counts))
                    .unwrap();

                // 3. If the task was for a sparse SNP, recycle its small I/O buffer.
                if let Some(recycled_buffer) = result.recycled_buffer {
                    // Use a non-blocking `try_send`. If the I/O task has already shut down,
                    // the channel will be closed, and this will return an `Err`.
                    // We gracefully ignore this error, as the buffer is no longer needed.
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


/// **Helper:** Dispatches a dense batch and adds its handle to the context.
///
/// This function encapsulates the logic for processing a batch of dense SNPs.
/// It takes the current `dense_batch` and a mutable reference to the main
/// `PipelineContext`. Its responsibilities are to:
///
/// 1.  Check if the batch is non-empty.
/// 2.  Use `std::mem::replace` to efficiently take ownership of the batch data,
///     leaving an empty, reset batch in its place.
/// 3.  Call `dispatch_person_major_batch`, cloning the necessary `Arc`s from the
///     context to move them into the new compute task.
/// 4.  Push the returned `JoinHandle` into the context's `compute_handles` vector
///     for later aggregation.
///
/// This approach simplifies the calling code in `run_orchestration_loop` and
/// centralizes the logic for handling dense SNP batches.
fn dispatch_and_clear_dense_batch(
    dense_batch: &mut DenseSnpBatch,
    context: &mut PipelineContext,
) {
    // --- Guard Clause: Do nothing if the batch is empty ---
    if dense_batch.metadata.is_empty() {
        return;
    }

    // --- Take ownership of the batch and reset the original ---
    // This is a highly efficient swap that moves the data without copying.
    // The original `dense_batch` is now empty and ready for the next set of SNPs.
    let batch_to_process = std::mem::replace(
        dense_batch,
        DenseSnpBatch::new_empty(dense_batch.data.capacity()),
    );

    // --- Dispatch the computation using the dedicated helper ---
    // We clone the Arcs from the context, which is a cheap reference count bump.
    let handle = dispatch_person_major_batch(
        batch_to_process,
        context.prep_result.clone(),
        context.tile_pool.clone(),
        context.sparse_index_pool.clone(),
        context.partial_result_pool.clone(),
    );

    // --- Add the handle to the context for later processing ---
    context.compute_handles.push(handle);
}


/// Dispatches a batch of dense SNPs to the person-major compute path.
///
/// This function takes ownership of a `DenseSnpBatch` and all necessary shared
/// resources as `Arc`s. It spawns a blocking task to run the CPU-bound computation
/// without stalling the async runtime. The spawned task performs the following:
///
/// 1.  Pops a set of reusable, partial result buffers from a shared pool.
/// 2.  Calls the high-throughput `run_person_major_path` kernel.
/// 3.  Packages the computed partial scores and counts into a `ComputeTaskResult`.
/// 4.  The `recycled_buffer` field of the result is `None`, as the large batch
///     data buffer is consumed and dropped, not recycled.
///
/// # Arguments
/// * `batch_to_process`: The `DenseSnpBatch` to be processed. The function takes
///   this by value to move it into the spawned task.
/// * `prep_result`: An `Arc` to the global `PreparationResult`.
/// * `tile_pool`: An `Arc` to the pool of reusable person-major tile buffers.
/// * `sparse_index_pool`: An `Arc` to the pool of thread-local sparse index buffers.
/// * `partial_result_pool`: An `Arc` to the pool of reusable partial score/count buffers.
///
/// # Returns
/// A `JoinHandle` to the spawned compute task. The handle resolves to a `Result`
/// containing the `ComputeTaskResult` on success.
fn dispatch_person_major_batch(
    batch_to_process: DenseSnpBatch,
    prep_result: Arc<prepare::PreparationResult>,
    tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    sparse_index_pool: Arc<SparseIndexPool>,
    partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,
) -> task::JoinHandle<Result<ComputeTaskResult, Box<dyn Error + Send + Sync>>> {
    task::spawn_blocking(move || {
        // 1. Acquire a set of partial result buffers from the pool.
        // This unwrap is safe; the pool is sized to prevent exhaustion.
        let (dirty_scores, dirty_counts) = partial_result_pool.pop().unwrap();

        // 2. Transition the buffers to the "Clean" state, zeroing them.
        let mut clean_scores = dirty_scores.into_clean();
        let mut clean_counts = dirty_counts.into_clean();

        // 3. Dispatch to the synchronous, CPU-bound compute function.
        batch::run_person_major_path(
            &batch_to_process.data,
            &batch_to_process.metadata,
            &prep_result,
            &mut clean_scores,
            &mut clean_counts,
            &tile_pool,
            &sparse_index_pool,
        )?;

        // 4. Package the results into the unified result type.
        let result = ComputeTaskResult {
            scores: clean_scores.into_dirty(),
            counts: clean_counts.into_dirty(),
            // The large `data` buffer from `batch_to_process` is NOT recycled.
            // It is dropped here when the task completes.
            recycled_buffer: None,
        };

        Ok(result)
    })
}

/// Dispatches a single sparse SNP to the SNP-major compute path.
fn dispatch_snp_major_path(
    snp_buffer: SnpDataBuffer,
    matrix_row_index: MatrixRowIndex,
    prep_result: Arc<prepare::PreparationResult>,
    partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,
) -> task::JoinHandle<Result<ComputeTaskResult, Box<dyn Error + Send + Sync>>> {
    task::spawn_blocking(move || {
        let (dirty_scores, dirty_counts) = partial_result_pool.pop().unwrap();
        let mut clean_scores = dirty_scores.into_clean();
        let mut clean_counts = dirty_counts.into_clean();

        batch::run_snp_major_path(
            &snp_buffer.0,
            &prep_result,
            &mut clean_scores,
            &mut clean_counts,
            matrix_row_index,
        )?;

        let result = ComputeTaskResult {
            scores: clean_scores.into_dirty(),
            counts: clean_counts.into_dirty(),
            recycled_buffer: Some(snp_buffer.0), // The buffer is passed back for recycling.
        };
        Ok(result)
    })
}

