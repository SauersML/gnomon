// ========================================================================================
//
//                               THE STRATEGIC ORCHESTRATOR: GNOMON
//
// ========================================================================================
//
// This module is the central nervous system and active conductor of the application.
// Its sole responsibility is to orchestrate the high-performance pipeline defined in
// the other modules. It owns all major resources and manages the application lifecycle
// from argument parsing to final output.

use clap::Parser;
use crossbeam_queue::ArrayQueue;
use gnomon::batch::{self, SparseIndexPool};
use gnomon::io::SnpReader;
use gnomon::reformat;
use gnomon::types::{
    BimRowIndex, ComputePath, DenseSnpBatch, DirtyCounts, DirtyScores, MatrixRowIndex, SnpDataBuffer,
};
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio::task;
use std::fs::File;
use cache_size;
use ryu;

// ========================================================================================
//                              COMMAND-LINE INTERFACE DEFINITION
// ========================================================================================

#[derive(Parser, Debug)]
#[clap(
    name = "gnomon",
    version,
    about = "A high-performance engine for polygenic score calculation."
)]
struct Args {
    /// Path to a single score file or a directory containing multiple score files.
    #[clap(long)]
    score: PathBuf,

    /// Path to a file containing a list of individual IDs (IIDs) to include.
    /// If not provided, all individuals in the .fam file will be scored.
    #[clap(long)]
    keep: Option<PathBuf>,

    /// Path to the PLINK .bed file, or a directory containing a single .bed file.
    input_path: PathBuf,
}

// ========================================================================================
//                              THE MAIN ORCHESTRATION LOGIC
// ========================================================================================

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let overall_start_time = Instant::now();
    let args = Args::parse();
    let plink_prefix = resolve_plink_prefix(&args.input_path)?;

    // --- Phase 1: Preparation ---
    // This phase is synchronous and CPU-bound. It parses all input files,
    // reconciles variants, and produces a "computation blueprint".
    let prep_result = run_preparation_phase(&plink_prefix, &args)?;

    // --- Phase 2: Resource Allocation ---
    // A new context is created, which allocates all necessary memory pools and
    // final result buffers based on the blueprint from the preparation phase.
    let mut context = PipelineContext::new(Arc::clone(&prep_result));
    eprintln!("> Resource allocation complete.");

    // --- Phase 3: Pipeline Execution ---
    // This is the primary asynchronous phase. It spawns the I/O producer and
    // orchestrator tasks, which run concurrently to maximize throughput.
    let computation_start = Instant::now();
    run_pipeline(&mut context, &plink_prefix).await?;
    eprintln!("> Computation finished. Total pipeline time: {:.2?}", computation_start.elapsed());

    // --- Phase 4: Finalization & Output ---
    // After all computation is complete and results are aggregated, this
    // synchronous phase writes the final scores to disk.
    finalize_and_write_output(&plink_prefix, &context)?;
    
    eprintln!("\nSuccess! Total execution time: {:.2?}", overall_start_time.elapsed());
    Ok(())
}

/// **Helper 1:** Encapsulates the entire preparation and file normalization phase.
///
/// This function is synchronous and CPU-bound. It takes the raw user arguments,
/// finds and normalizes all score files, and then calls the main preparation
/// logic to produce a "computation blueprint" (`PreparationResult`). All user-facing
/// console output for this phase is handled here.
fn run_preparation_phase(
    plink_prefix: &Path,
    args: &Args,
) -> Result<Arc<prepare::PreparationResult>, Box<dyn Error + Send + Sync>> {
    eprintln!("> Using PLINK prefix: {}", plink_prefix.display());
    let output_dir = plink_prefix.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;

    // --- Find and normalize all score files ---
    let score_files = if args.score.is_dir() {
        eprintln!("> Found directory for --score, locating all score files...");
        fs::read_dir(&args.score)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|p| p.is_file())
            .collect()
    } else {
        vec![args.score.clone()]
    };

    if score_files.is_empty() {
        return Err("No score files found in the specified path.".into());
    }

    eprintln!(
        "> Normalizing and preparing {} score file(s)...",
        score_files.len()
    );
    let prep_phase_start = Instant::now();

    let mut native_score_files = Vec::with_capacity(score_files.len());
    for score_file_path in &score_files {
        match reformat::is_gnomon_native_format(score_file_path) {
            Ok(true) => {
                native_score_files.push(score_file_path.clone());
            }
            Ok(false) => {
                eprintln!(
                    "> Info: Score file '{}' is not in native format. Attempting conversion...",
                    score_file_path.display()
                );
                match reformat::reformat_pgs_file(score_file_path) {
                    Ok(new_path) => {
                        eprintln!("> Success: Converted to '{}'.", new_path.display());
                        native_score_files.push(new_path);
                    }
                    Err(e) => {
                        return Err(format!(
                            "Failed to auto-reformat '{}': {}. Please ensure it is a valid PGS Catalog file or convert it to the gnomon-native format manually.",
                            score_file_path.display(), e
                        ).into());
                    }
                }
            }
            Err(e) => {
                return Err(format!(
                    "Error reading score file '{}': {}",
                    score_file_path.display(), e
                ).into());
            }
        }
    }

    // --- Run the main preparation logic ---
    let prep =
        prepare::prepare_for_computation(plink_prefix, &native_score_files, args.keep.as_deref())
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;

    eprintln!(
        "> Preparation complete in {:.2?}. Found {} individuals to score and {} overlapping variants across {} score(s).",
        prep_phase_start.elapsed(),
        prep.num_people_to_score,
        prep.num_reconciled_variants,
        prep.score_names.len()
    );

    Ok(Arc::new(prep))
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


/// **Helper:** Handles the final file writing.
///
/// This function is synchronous. It takes the final, aggregated results from the
/// `PipelineContext` and writes them to a `.sscore` file. The output path is
/// derived from the original PLINK prefix to ensure results are co-located with
/// the input data.
fn finalize_and_write_output(
    plink_prefix: &Path,
    context: &PipelineContext,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // The output directory should already exist from the preparation phase,
    // but we ensure it's there for robustness.
    let output_dir = plink_prefix.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;

    // Construct a self-describing output filename based on the input prefix.
    let out_filename = {
        let mut s = plink_prefix
            .file_name()
            .map_or_else(|| OsString::from("gnomon_results"), OsString::from);
        s.push(".sscore");
        s
    };
    let out_path = output_dir.join(&out_filename);

    eprintln!(
        "> Writing {} scores per person to {}",
        context.prep_result.score_names.len(),
        out_path.display()
    );
    let output_start = Instant::now();

    // Delegate to the existing file writer, passing all data from the context.
    write_scores_to_file(
        &out_path,
        &context.prep_result.final_person_iids,
        &context.prep_result.score_names,
        &context.prep_result.score_variant_counts,
        &context.all_scores,
        &context.all_missing_counts,
    )?;

    eprintln!("> Final output written in {:.2?}", output_start.elapsed());
    Ok(())
}

// ========================================================================================
//                                  HELPER FUNCTIONS
// ========================================================================================

/// Represents a command sent from the main orchestrator to the I/O producer thread.
/// Using an enum makes the control flow explicit and type-safe.
enum IoCommand {
    /// A command to read the next SNP into the provided buffer.
    Read(Vec<u8>),
    /// A command to gracefully shut down the I/O thread.
    Shutdown,
}

/// Represents the concrete action the orchestrator must take for a given SNP.
enum DispatchAction {
    /// The SNP is sparse and should be processed immediately.
    ProcessSparse {
        snp_buffer: SnpDataBuffer,
        matrix_row_index: MatrixRowIndex,
    },
    /// The SNP is dense and should be added to the pending batch.
    BufferDense {
        snp_buffer: SnpDataBuffer,
        matrix_row_index: MatrixRowIndex,
    },
}

// A type alias for the result of a compute task.
type ComputeResult = (DirtyScores, DirtyCounts, Option<Vec<u8>>);

/// The unified result returned by any compute task.
struct ComputeTaskResult {
    scores: DirtyScores,
    counts: DirtyCounts,
    // The buffer from a sparse SNP task, to be recycled. `None` for dense batches.
    recycled_buffer: Option<Vec<u8>>,
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

/// Intelligently resolves the user-provided input path to a PLINK prefix.
fn resolve_plink_prefix(path: &Path) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    if path.is_dir() {
        let bed_files: Vec<PathBuf> = fs::read_dir(path)?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "bed"))
            .collect();
        match bed_files.len() {
            0 => Err(format!("No .bed file found in the directory '{}'.", path.display()).into()),
            1 => Ok(bed_files[0].with_extension("")),
            _ => Err(format!("Ambiguous input: multiple .bed files found in directory '{}'.", path.display()).into()),
        }
    } else if path.is_file() {
        if path.extension().map_or(false, |ext| ext == "bed") {
            Ok(path.with_extension(""))
        } else {
            Err(format!("Input file '{}' must have a .bed extension.", path.display()).into())
        }
    } else {
        let bed_path_from_prefix = path.with_extension("bed");
        if bed_path_from_prefix.is_file() {
            Ok(path.to_path_buf())
        } else {
            Err(format!("Input '{}' is not a valid directory, .bed file, or PLINK prefix.", path.display()).into())
        }
    }
}

/// Writes the final calculated scores to a self-describing, tab-separated file.
/// This function now calculates the final per-variant average and missing percentage.
fn write_scores_to_file(
    path: &Path,
    person_iids: &[String],
    score_names: &[String],
    score_variant_counts: &[u32],
        sum_scores: &[f64],
    missing_counts: &[u32],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let num_scores = score_names.len();

    // Write the new, more descriptive, and correctly tab-separated header.
    write!(writer, "#IID")?;
    for name in score_names {
        write!(writer, "\t{}_AVG\t{}_MISSING_PCT", name, name)?;
    }
    writeln!(writer)?;

    let mut line_buffer =
        String::with_capacity(person_iids.get(0).map_or(128, |s| s.len() + num_scores * 24));
    let mut sum_score_chunks = sum_scores.chunks_exact(num_scores);
    let mut missing_count_chunks = missing_counts.chunks_exact(num_scores);
    let mut ryu_buffer_score = ryu::Buffer::new();
    let mut ryu_buffer_missing = ryu::Buffer::new();

    for iid in person_iids {
        let person_sum_scores = sum_score_chunks.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Mismatched number of persons and score rows during final write.",
            )
        })?;
        let person_missing_counts = missing_count_chunks.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Mismatched number of persons and missing count rows during final write.",
            )
        })?;

        line_buffer.clear();
        write!(&mut line_buffer, "{}", iid).unwrap();

        for i in 0..num_scores {
            let final_sum_score = person_sum_scores[i];
            let missing_count = person_missing_counts[i];
            let total_variants_for_score = score_variant_counts[i];

            // The score is calculated based on the number of non-missing variants.
            // This behavior matches standard tools when mean-imputation is disabled.
            let variants_used = total_variants_for_score.saturating_sub(missing_count);

            let avg_score = if variants_used > 0 {
                final_sum_score / (variants_used as f64)
            } else {
                0.0
            };

            let missing_pct = if total_variants_for_score > 0 {
                (missing_count as f32 / total_variants_for_score as f32) * 100.0
            } else {
                0.0
            };

            // Write the correctly tab-separated data columns.
            write!(&mut line_buffer, "\t{}\t{}", ryu_buffer_score.format(avg_score), ryu_buffer_missing.format(missing_pct)).unwrap();
        }
        writeln!(writer, "{}", line_buffer)?;
    }

    writer.flush()
}

/// Dispatches the current dense SNP batch for processing and resets the batch.
fn dispatch_and_clear_dense_batch(
    dense_batch: &mut DenseSnpBatch,
    prep_result: Arc<prepare::PreparationResult>,
    tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    sparse_index_pool: Arc<SparseIndexPool>,
    partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,
    handles: &mut Vec<task::JoinHandle<Result<ComputeTaskResult, Box<dyn Error + Send + Sync>>>>,
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
    // This avoids duplicating the task spawning logic.
    let handle = dispatch_person_major_batch(
        batch_to_process,
        prep_result,
        tile_pool,
        sparse_index_pool,
        partial_result_pool,
    );

    // Add the handle for the newly spawned task to the main list for later processing.
    handles.push(handle);
}
