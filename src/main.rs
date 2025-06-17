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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let overall_start_time = Instant::now();

    // --- Phase 1: Argument Parsing and Path Resolution ---
    let args = Args::parse();
    let plink_prefix = resolve_plink_prefix(&args.input_path)?;
    eprintln!("> Using PLINK prefix: {}", plink_prefix.display());
    let output_dir = plink_prefix.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;

    // --- Phase 2: The Preparation Phase ---
    // This logic now supports providing a directory to the --score argument.
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

    eprintln!("> Normalizing and preparing {} score file(s)...", score_files.len());
    let prep_phase_start = Instant::now();

    let mut native_score_files = Vec::with_capacity(score_files.len());
    for score_file_path in &score_files {
        // First, check if the file is already in the correct format.
        match reformat::is_gnomon_native_format(score_file_path) {
            Ok(true) => {
                // It's already native, so we can use it directly.
                native_score_files.push(score_file_path.clone());
            }
            Ok(false) => {
                // It's not in native format, so try to reformat it.
                eprintln!(
                    "> Info: Score file '{}' is not in native format. Attempting conversion...",
                    score_file_path.display()
                );
                match reformat::reformat_pgs_file(score_file_path) {
                    Ok(new_path) => {
                        eprintln!(
                            "> Success: Converted to '{}'.",
                            new_path.display()
                        );
                        native_score_files.push(new_path);
                    }
                    Err(e) => {
                        // Reformatting failed. This is a fatal error.
                        return Err(format!(
                            "Failed to auto-reformat '{}': {}. Please ensure it is a valid PGS Catalog file or convert it to the gnomon-native format manually.",
                            score_file_path.display(), e
                        ).into());
                    }
                }
            }
            Err(e) => {
                // An I/O error occurred while trying to check the file.
                return Err(format!(
                    "Error reading score file '{}': {}",
                    score_file_path.display(), e
                ).into());
            }
        }
    }

    // Now, run the preparation phase on the fully normalized list of files.
    let prep = prepare::prepare_for_computation(&plink_prefix, &native_score_files, args.keep.as_deref())
        .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
    let prep_result = Arc::new(prep);

    eprintln!(
        "> Preparation complete in {:.2?}. Found {} individuals to score and {} overlapping variants across {} score(s).",
        prep_phase_start.elapsed(),
        prep_result.num_people_to_score,
        prep_result.num_reconciled_variants,
        prep_result.score_names.len()
    );

    // --- Phase 3: Dynamic Runtime Resource Allocation ---
    let resource_alloc_start = Instant::now();

    let num_scores = prep_result.score_names.len();
    let result_buffer_size = prep_result.num_people_to_score * num_scores;
    let mut all_scores = vec![0.0f64; result_buffer_size];
    let mut all_missing_counts = vec![0u32; result_buffer_size];

    // These resource pools are shared across all compute tasks.
    let tile_pool = Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 2));
    let sparse_index_pool = Arc::new(SparseIndexPool::new());
    
    // --- Determine a Target Batch Size for the Person-Major Path ---
    // The goal is to make batches large enough to amortize the high fixed cost of
    // the pivot operation, while respecting L3 cache size to keep the tile hot.
    const PERSON_BLOCK_SIZE: u64 = 4096; // Must match batch.rs
    const MIN_DENSE_BATCH_SIZE: usize = 1 * 1024 * 1024; // 1 MB
    const MAX_DENSE_BATCH_SIZE: usize = 256 * 1024 * 1024; // 256 MB

    let dense_batch_target_size = {
        let l3_cache_bytes = cache_size::l3_cache_size().unwrap_or(32 * 1024 * 1024);
        let bytes_per_snp = prep_result.bytes_per_snp;

        // Calculate the number of SNPs whose pivoted tile would fill the L3 cache.
        // This is an ideal upper bound for the number of SNPs in a person-major batch.
        let max_snps_for_l3 = (l3_cache_bytes as u64) / PERSON_BLOCK_SIZE;

        // Convert that number of SNPs back into a raw data size.
        (max_snps_for_l3 * bytes_per_snp) as usize
    }
    .clamp(MIN_DENSE_BATCH_SIZE, MAX_DENSE_BATCH_SIZE);
    
    // The pipeline depth determines how many I/O and compute tasks can run in parallel.
    const PIPELINE_DEPTH: usize = 2;

    // A pool of reusable buffers for partial results. This avoids re-allocating
    // the large score/count vectors for every compute task.
    let partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>> =
        Arc::new(ArrayQueue::new(PIPELINE_DEPTH + 1));
    for _ in 0..(PIPELINE_DEPTH + 1) {
        partial_result_pool
            .push((
                DirtyScores(vec![0.0f64; result_buffer_size]),
                DirtyCounts(vec![0u32; result_buffer_size]),
            ))
            .unwrap();
    }

    // The pipeline now passes single-SNP data buffers between the I/O producer and
    // the main consumer/orchestrator thread.
    let (full_buffer_tx, mut full_buffer_rx) = mpsc::channel::<SnpDataBuffer>(PIPELINE_DEPTH);
    let (empty_buffer_tx, mut empty_buffer_rx) = mpsc::channel::<(Vec<u8>, bool)>(PIPELINE_DEPTH);

    eprintln!(
        "> Resource allocation complete in {:.2?}. Dense batch target: {} MB. Pipeline depth: {}",
        resource_alloc_start.elapsed(),
        dense_batch_target_size / 1024 / 1024,
        PIPELINE_DEPTH
    );

    // --- Phase 4: Adaptive Concurrent Pipeline Execution ---
    // Pre-fill the pipeline with empty buffers for the I/O producer to use.
    // Each buffer is sized to hold exactly one SNP's data.
    let single_snp_buffer_size = prep_result.bytes_per_snp as usize;
    for _ in 0..PIPELINE_DEPTH {
        empty_buffer_tx
            .send((vec![0u8; single_snp_buffer_size], true))
            .await?;
    }

    let bed_path = plink_prefix.with_extension("bed");
    let reader = SnpReader::new(
        &bed_path,
        prep_result.bytes_per_snp,
        prep_result.total_snps_in_bim,
    )?;

    // --- I/O Producer Task ---
    // This task reads SNPs from the .bed file and sends only the relevant ones
    // to the main orchestrator thread for processing.
    let prep_clone_for_io = Arc::clone(&prep_result);
    let io_handle = tokio::spawn(async move {
        let mut reader = reader;
        let mut bed_row_cursor: u32 = 0;
        let mut required_indices_cursor: usize = 0;

        'producer: while let Some(mut buffer) = empty_buffer_rx.recv().await {
            let snp_data_opt = match task::spawn_blocking(move || {
                let snp_data_opt = reader.read_next_snp(&mut buffer)?;
                Ok::<_, io::Error>((reader, buffer, snp_data_opt))
            })
            .await
            {
                Ok(Ok(res)) => res,
                Ok(Err(e)) | Err(_) => {
                    eprintln!("[I/O Task Error]: {}", e);
                    break 'producer;
                }
            };
            
            reader = snp_data_opt.0;
            let unused_buffer = snp_data_opt.1;

            if let Some(filled_buffer) = snp_data_opt.2 {
                // Check if the SNP we just read is one we actually need.
                let is_relevant = prep_clone_for_io
                    .required_bim_indices
                    .get(required_indices_cursor)
                    .map_or(false, |&req_idx| req_idx.0 == bed_row_cursor);

                if is_relevant {
                    required_indices_cursor += 1;
                    if full_buffer_tx.send(SnpDataBuffer(filled_buffer)).await.is_err() {
                        break 'producer'; // Consumer hung up.
                    }
                } else {
                    // This SNP is not needed, so recycle its buffer immediately.
                    if empty_buffer_tx.send(filled_buffer).await.is_err() {
                        break 'producer';
                    }
                }
                bed_row_cursor += 1;
            } else {
                // EOF. Drop the unused buffer and terminate.
                drop(unused_buffer);
                break 'producer';
            }
        }
    });

    // --- Main Orchestrator Loop ---
    // This loop receives a stream of *relevant* SNPs and dispatches them.
    let mut required_indices_cursor: usize = 0;
    let mut dense_batch = DenseSnpBatch::new_empty(dense_batch_target_size);
    
    let computation_start = Instant::now();
    while let Some(snp_buffer) = full_buffer_rx.recv().await {
        let matrix_row_index = prep_result.required_bim_indices[required_indices_cursor];
        required_indices_cursor += 1;
        
        // This is a placeholder for the adaptive logic.
        let path_decision = ComputePath::PersonMajor; 
        
        match path_decision {
            ComputePath::SnpMajor => {
                // To be implemented: dispatch sparse work.
            }
            ComputePath::PersonMajor => {
                // First SNP in a new dense batch? Set the starting metadata.
                if dense_batch.snp_count == 0 {
                    dense_batch.start_matrix_row = matrix_row_index;
                }
                dense_batch.data.extend_from_slice(&snp_buffer.0);
                dense_batch.snp_count += 1;

                // Recycle the single-SNP buffer immediately.
                if empty_buffer_tx.send(snp_buffer.0).await.is_err() {
                    break;
                }

                // If the batch is now full, dispatch it for processing.
                if dense_batch.data.len() >= dense_batch_target_size {
                    let batch_to_process = std::mem::replace(
                        &mut dense_batch,
                        DenseSnpBatch::new_empty(dense_batch_target_size),
                    );
                    
                    let prep_clone = Arc::clone(&prep_result);
                    let tile_pool_clone = Arc::clone(&tile_pool);
                    let partial_result_pool_clone = Arc::clone(&partial_result_pool);
                    let sparse_index_pool_clone = Arc::clone(&sparse_index_pool);
                    let (dirty_scores, dirty_counts) = partial_result_pool_clone.pop().unwrap();
                    
                    let compute_handle = task::spawn_blocking(move || {
                        let mut clean_scores = dirty_scores.into_clean();
                        let mut clean_counts = dirty_counts.into_clean();

                        // This will be renamed to `run_person_major_path`
                        batch::run_chunk_computation(
                            &batch_to_process.data,
                            &prep_clone,
                            &mut clean_scores,
                            &mut clean_counts,
                            &tile_pool_clone,
                            &sparse_index_pool_clone,
                            batch_to_process.start_matrix_row,
                            batch_to_process.snp_count,
                            0, // The chunk_bed_row_offset is no longer needed
                        )?;
                        
                        let result = (clean_scores.into_dirty(), clean_counts.into_dirty());
                        Ok::<_, Box<dyn Error + Send + Sync>>(result)
                    });
                    
                    let (partial_scores, partial_missing_counts) = match compute_handle.await? {
                        Ok(result) => result,
                        Err(e) => return Err(e),
                    };

                    for (master, &partial) in all_scores.iter_mut().zip(&partial_scores) { *master += partial; }
                    for (master, &partial) in all_missing_counts.iter_mut().zip(&partial_missing_counts) { *master += partial; }
                    partial_result_pool_clone.push((partial_scores, partial_missing_counts)).unwrap();
                }
            }
        }
        
        // Progress reporting
        if required_indices_cursor > 0 && required_indices_cursor % 10000 == 0 {
             eprintln!(
                "> Processed {}/{} relevant variants...",
                required_indices_cursor,
                prep_result.num_reconciled_variants
            );
        }
    }

    // After the loop, process any final, non-full dense batch.
    if dense_batch.snp_count > 0 {
        // This is a simplified version of the dispatch logic inside the loop.
        // In a full implementation, this would call the same dispatch helper.
        eprintln!("> Processing final batch of {} dense SNPs...", dense_batch.snp_count);
        let prep_clone = Arc::clone(&prep_result);
        let tile_pool_clone = Arc::clone(&tile_pool);
        let partial_result_pool_clone = Arc::clone(&partial_result_pool);
        let sparse_index_pool_clone = Arc::clone(&sparse_index_pool);
        let (dirty_scores, dirty_counts) = partial_result_pool_clone.pop().unwrap();
        let compute_handle = task::spawn_blocking(move || {
            let mut clean_scores = dirty_scores.into_clean();
            let mut clean_counts = dirty_counts.into_clean();
            batch::run_chunk_computation(&dense_batch.data, &prep_clone, &mut clean_scores, &mut clean_counts, &tile_pool_clone, &sparse_index_pool_clone, dense_batch.start_matrix_row, dense_batch.snp_count, 0)?;
            let result = (clean_scores.into_dirty(), clean_counts.into_dirty());
            Ok::<_, Box<dyn Error + Send + Sync>>(result)
        });
        let (partial_scores, partial_missing_counts) = compute_handle.await?.unwrap();
        for (master, &partial) in all_scores.iter_mut().zip(&partial_scores) { *master += partial; }
        for (master, &partial) in all_missing_counts.iter_mut().zip(&partial_missing_counts) { *master += partial; }
        partial_result_pool_clone.push((partial_scores, partial_missing_counts)).unwrap();
    }


    io_handle.await?;
    eprintln!("> Computation finished. Total pipeline time: {:.2?}", computation_start.elapsed());

    // --- Phase 5: Finalization and Output ---
    // The output filename is now robustly based on the PLINK prefix, not the score file path.
    let out_filename = {
        let mut s = plink_prefix.file_name().map_or_else(
            || OsString::from("gnomon_results"),
            OsString::from,
        );
        s.push(".sscore");
        s
    };
    let out_path = output_dir.join(&out_filename);
    eprintln!(
        "> Writing {} scores per person to {}",
        prep_result.score_names.len(),
        out_path.display()
    );

    let output_start = Instant::now();
    write_scores_to_file(
        &out_path,
        &prep_result.final_person_iids,
        &prep_result.score_names,
        &prep_result.score_variant_counts,
        &all_scores,
        &all_missing_counts,
    )?;
    eprintln!("> Final output written in {:.2?}", output_start.elapsed());

    eprintln!("\nSuccess! Total execution time: {:.2?}", overall_start_time.elapsed());

    Ok(())
}

// ========================================================================================
//                                  HELPER FUNCTIONS
// ========================================================================================

// A type alias for the result of a compute task.
type ComputeResult = (DirtyScores, DirtyCounts, Option<Vec<u8>>);

/// Dispatches a batch of dense SNPs to the person-major compute path.
fn dispatch_person_major_batch(
    batch_to_process: DenseSnpBatch,
    prep_result: Arc<prepare::PreparationResult>,
    tile_pool: Arc<ArrayQueue<Vec<EffectAlleleDosage>>>,
    sparse_index_pool: Arc<SparseIndexPool>,
    partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,
) -> task::JoinHandle<Result<ComputeResult, Box<dyn Error + Send + Sync>>> {
    task::spawn_blocking(move || {
        let (dirty_scores, dirty_counts) = partial_result_pool.pop().unwrap();
        let mut clean_scores = dirty_scores.into_clean();
        let mut clean_counts = dirty_counts.into_clean();

        batch::run_person_major_path(
            &batch_to_process.data,
            &prep_result,
            &mut clean_scores,
            &mut clean_counts,
            &tile_pool,
            &sparse_index_pool,
            batch_to_process.start_matrix_row,
            batch_to_process.snp_count,
        )?;
        
        let result = (clean_scores.into_dirty(), clean_counts.into_dirty(), None);
        Ok(result)
    })
}

/// Dispatches a single sparse SNP to the SNP-major compute path.
fn dispatch_snp_major_path(
    snp_buffer: SnpDataBuffer,
    matrix_row_index: MatrixRowIndex,
    prep_result: Arc<prepare::PreparationResult>,
    partial_result_pool: Arc<ArrayQueue<(DirtyScores, DirtyCounts)>>,
) -> task::JoinHandle<Result<ComputeResult, Box<dyn Error + Send + Sync>>> {
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

        // The SNP buffer is passed back so it can be recycled.
        let result = (clean_scores.into_dirty(), clean_counts.into_dirty(), Some(snp_buffer.0));
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
///
/// This function is the sole entry point for sending a batch of "dense" variants
/// to the high-throughput, person-major compute path. It is designed to be
/// non-blocking from the orchestrator's perspective.
///
/// # Key Behaviors:
/// 1.  **Idempotent:** If the batch is empty, it returns immediately with no action.
/// 2.  **Ownership Transfer:** It takes ownership of the batch's contents using
///     `std::mem::replace`, leaving the original `dense_batch` empty but with its
///     allocated capacity intact, preventing future reallocations.
/// 3.  **Asynchronous Execution:** It spawns the CPU-bound work onto a blocking
///     thread using `task::spawn_blocking`, freeing the main async orchestrator
///     to continue processing I/O.
/// 4.  **Resource Management:** It correctly pulls a set of partial result buffers
///     from the shared pool to be used by the compute task.
///
/// # Arguments
/// * `dense_batch` - A mutable reference to the `DenseSnpBatch` being assembled.
/// * `prep_result` - The `Arc`'d "computation blueprint".
/// * `tile_pool` - The `Arc`'d pool of reusable tile buffers.
/// * `sparse_index_pool` - The `Arc`'d pool of sparse index structures.
/// * `partial_result_pool` - The `Arc`'d pool of reusable partial result buffers.
/// * `handles` - The vector where the `JoinHandle` for the new task will be stored.
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

    // Clone the shared resource handles to move them into the async task.
    let prep_clone = Arc::clone(&prep_result);
    let tile_pool_clone = Arc::clone(&tile_pool);
    let sparse_index_pool_clone = Arc::clone(&sparse_index_pool);
    let partial_result_pool_clone = Arc::clone(&partial_result_pool);

    // --- Dispatch the computation to a blocking thread ---
    let handle = task::spawn_blocking(move || {
        // Pop a pre-allocated, "dirty" buffer set from the pool. This may block
        // briefly if the pipeline is full, which provides natural backpressure.
        let (dirty_scores, dirty_counts) = partial_result_pool_clone.pop().unwrap();

        // Zero out the buffers, transitioning them to the "Clean" state.
        let mut clean_scores = dirty_scores.into_clean();
        let mut clean_counts = dirty_counts.into_clean();

        // Execute the CPU-bound, person-major computation path.
        batch::run_person_major_path(
            &batch_to_process.data,
            &prep_clone,
            &mut clean_scores,
            &mut clean_counts,
            &tile_pool_clone,
            &sparse_index_pool_clone,
            &batch_to_process.metadata,
        )?;

        // Package the results into the unified result type.
        let result = ComputeTaskResult {
            scores: clean_scores.into_dirty(),
            counts: clean_counts.into_dirty(),
            // A dense batch consumes its data; there is no single-SNP I/O buffer to recycle.
            recycled_buffer: None,
        };

        Ok::<_, Box<dyn Error + Send + Sync>>(result)
    });

    // Add the handle for the newly spawned task to the main list for later processing.
    handles.push(handle);
}
