// ========================================================================================
//
//                      THE STRATEGIC ORCHESTRATOR: GNOMON
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
use gnomon::io::SnpChunkReader;
use gnomon::prepare::{self, PrepError};
use gnomon::reformat;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use sysinfo::System;
use tokio::sync::mpsc;
use tokio::task;
use std::fs::File;

// ========================================================================================
//                         COMMAND-LINE INTERFACE DEFINITION
// ========================================================================================

#[derive(Parser, Debug)]
#[clap(
    name = "gnomon",
    version,
    about = "A high-performance engine for polygenic score calculation."
)]
struct Args {
    /// Path to the score file.
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
//                           THE MAIN ORCHESTRATION LOGIC
// ========================================================================================

/// A message passed from the I/O producer task to the compute dispatcher.
///
/// This struct bundles the raw byte buffer from the .bed file with metadata
/// required for processing, such as the number of valid bytes and the number
/// of SNPs contained within that chunk.
struct IoMessage {
    buffer: Vec<u8>,
    bytes_read: usize,
    snps_in_chunk: usize,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    let start_time = Instant::now();

    // --- Phase 1: Argument Parsing and Path Resolution ---
    let args = Args::parse();
    let plink_prefix = resolve_plink_prefix(&args.input_path)?;
    eprintln!("> Using PLINK prefix: {}", plink_prefix.display());

    let output_dir = plink_prefix.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;

    // --- Phase 2: The Preparation Phase ---
    // This block attempts to prepare for computation. If it fails with a parsing
    // error, it falls back to an automatic reformatting attempt for the PGS Catalog format.
    let prep_result = {
        // The path to the score file to be used, which may be updated by the
        // fallback logic.
        let mut score_path_to_use = args.score.clone();

        eprintln!(
            "> Preparing data using score file: {}",
            score_path_to_use.display()
        );

        // --- First Attempt ---
        // Try to process the user-provided score file directly.
        match prepare::prepare_for_computation(
            &plink_prefix,
            &score_path_to_use,
            args.keep.as_deref(),
        ) {
            // Success on the first try means the file was already in the correct format.
            Ok(result) => Ok(result),

            // A recoverable parsing error triggers the fallback logic.
            Err(prep_error @ PrepError::Header(_)) | Err(prep_error @ PrepError::Parse(_)) => {
                eprintln!(
                    "\nWarning: Initial parsing failed. Checking for PGS Catalog format..."
                );

                // --- Reformat Attempt ---
                // Call the specialist reformatting module.
                match reformat::reformat_pgs_file(&score_path_to_use) {
                    Ok(new_path) => {
                        eprintln!(
                            "Success! Created compatible score file at: '{}'",
                            new_path.display()
                        );
                        eprintln!("Retrying computation with the new file...\n");

                        // The path to use is now the newly created, valid file.
                        score_path_to_use = new_path;

                        // --- Second (and Final) Attempt ---
                        // This result (Ok or Err) becomes the final result of the block.
                        prepare::prepare_for_computation(
                            &plink_prefix,
                            &score_path_to_use,
                            args.keep.as_deref(),
                        )
                    }
                    Err(reformat_error) => {
                        // The reformatting failed; the file is not a convertible format.
                        eprintln!("\nError: Automatic conversion failed: {}", reformat_error);
                        eprintln!("Please ensure the score file is in the gnomon-native format (snp_id, effect_allele, ...) or a valid PGS Catalog format.");
                        // Return the original, more relevant error.
                        Err(prep_error)
                    }
                }
            }
            // An unrecoverable error (e.g., I/O error) is passed through directly.
            Err(other_error) => Err(other_error),
        }
    };

    // The '?' operator propagates any final, unrecoverable error.
    // On success, the result is wrapped for concurrent sharing.
    let prep_result = Arc::new(prep_result?);

    eprintln!(
        "> Preparation complete. Found {} individuals to score and {} overlapping SNPs.",
        prep_result.num_people_to_score, prep_result.num_reconciled_snps
    );

    if prep_result.num_reconciled_snps == 0 {
        return Err("No overlapping SNPs found. Cannot proceed.".into());
    }

    // --- Phase 3: Dynamic Runtime Resource Allocation ---
    let mut all_scores =
        vec![0.0f32; prep_result.num_people_to_score * prep_result.score_names.len()];
    let tile_pool_capacity = num_cpus::get().max(1) * 2;
    let tile_pool = Arc::new(ArrayQueue::new(tile_pool_capacity));

    // The pool for sparse genotype indices is created once and shared safely
    // across all compute tasks using an Atomic Reference Counter.
    let sparse_index_pool = Arc::new(SparseIndexPool::new());

    // Determine a safe and effective I/O chunk size.
    const MIN_CHUNK_SIZE: u64 = 64 * 1024 * 1024;
    const MAX_CHUNK_SIZE: u64 = 1 * 1024 * 1024 * 1024;
    let mut sys = System::new_all();
    sys.refresh_memory();
    let available_mem = sys.available_memory();
    let chunk_size_bytes = (available_mem / 4).clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE) as usize;

    // --- Pipeline Resource Allocation ---
    const PIPELINE_DEPTH: usize = 2; // Classic double-buffering
    let partial_scores_size = prep_result.num_people_to_score * prep_result.score_names.len();
    let partial_scores_pool = Arc::new(ArrayQueue::new(PIPELINE_DEPTH + 1));
    for _ in 0..(PIPELINE_DEPTH + 1) {
        partial_scores_pool
            .push(vec![0.0f32; partial_scores_size])
            .unwrap();
    }
    let (full_buffer_tx, mut full_buffer_rx) = mpsc::channel::<IoMessage>(PIPELINE_DEPTH);
    let (empty_buffer_tx, mut empty_buffer_rx) = mpsc::channel::<Vec<u8>>(PIPELINE_DEPTH);

    eprintln!(
        "> Dynamically configured I/O chunk size to {} MB and pipeline depth to {}",
        chunk_size_bytes / 1024 / 1024,
        PIPELINE_DEPTH
    );

    // --- Phase 4: True Concurrent Pipeline Execution ---
    // Prime the pipeline by sending the empty I/O buffers to the I/O task.
    for _ in 0..PIPELINE_DEPTH {
        empty_buffer_tx
            .send(vec![0u8; chunk_size_bytes])
            .await?;
    }

    let bed_path = plink_prefix.with_extension("bed");
    let reader = SnpChunkReader::new(
        &bed_path,
        prep_result.total_people_in_fam,
        prep_result.total_snps_in_bim,
    )?;

    // Spawn the dedicated I/O Producer task.
    let io_handle = tokio::spawn(async move {
        // The `SnpChunkReader` is stateful and must be explicitly passed into and
        // returned from the blocking task on each iteration to update its cursor.
        // We re-bind `reader` as mutable to allow this update cycle.
        let mut reader = reader;

        // The I/O task owns the reader and the channels.
        while let Some(mut buffer) = empty_buffer_rx.recv().await {
            // The synchronous read_chunk call MUST be wrapped in spawn_blocking
            // to prevent it from blocking the async I/O task on a page fault.
            let read_result = task::spawn_blocking(move || {
                let bytes_read = reader.read_chunk(&mut buffer)?;
                let snps_in_chunk = reader.snps_in_bytes(bytes_read);
                // Return ownership of the reader and buffer for the next iteration.
                Ok::<_, io::Error>((reader, buffer, bytes_read, snps_in_chunk))
            })
            .await
            .unwrap(); // Panicking here is acceptable; a JoinError is unrecoverable.

            match read_result {
                // `reader` is destructured here to regain ownership.
                // EOF is reached. The I/O task's job is complete. We break the loop,
                // allowing the `reader` and `buffer` for this final iteration to be
                // dropped, which is the correct behavior during shutdown.
                Ok((_, _, 0, _)) => break,
                Ok((new_reader, buffer, bytes_read, snps_in_chunk)) => {
                    reader = new_reader;
                    if full_buffer_tx
                        .send(IoMessage {
                            buffer,
                            bytes_read,
                            snps_in_chunk,
                        })
                        .await
                        .is_err()
                    {
                        break; // Consumer has disconnected.
                    }
                }
                Err(e) => {
                    eprintln!("[I/O Task Error]: {}", e);
                    break;
                }
            }
        }
        // The reader is dropped here when the task finishes.
    });

    // The main task becomes the Consumer and Compute Dispatcher.
    // We must track two separate counters: one for the raw .bed file rows, and one
    // for the index into our compacted, reconciled SNP data structures.
    let mut bed_row_offset = 0;
    let mut reconciled_snps_processed = 0;

    while let Some(IoMessage {
        buffer: full_buffer,
        bytes_read,
        snps_in_chunk,
    }) = full_buffer_rx.recv().await
    {
        // The current raw I/O chunk corresponds to a specific range of rows in the .bed file.
        let bed_row_end = bed_row_offset + snps_in_chunk;

        // Using the sorted `required_snp_indices` list from the preparation phase,
        // we can efficiently find which (if any) of our reconciled SNPs fall
        // within the current raw I/O chunk. `partition_point` performs a binary search.
        let reconciled_indices_start = prep_result
            .required_snp_indices
            .partition_point(|&x| x < bed_row_offset);
        let reconciled_indices_end = prep_result
            .required_snp_indices
            .partition_point(|&x| x < bed_row_end);

        let num_reconciled_in_chunk = reconciled_indices_end - reconciled_indices_start;

        // If this chunk of the .bed file contains no SNPs relevant to our calculation,
        // we can skip it entirely and immediately recycle its I/O buffer.
        if num_reconciled_in_chunk == 0 {
            if empty_buffer_tx.send(full_buffer).await.is_err() {
                break;
            }
            bed_row_offset += snps_in_chunk;
            continue;
        }

        let prep_clone = Arc::clone(&prep_result);
        let tile_pool_clone = Arc::clone(&tile_pool);
        let partial_scores_pool_clone = Arc::clone(&partial_scores_pool);
        let sparse_index_pool_clone = Arc::clone(&sparse_index_pool);

        // Calculate the stride to correctly slice the padded weights buffer.
        // This logic must exactly match the padding logic in `prepare.rs`.
        const LANE_COUNT: usize = 8;
        let num_scores = prep_clone.score_names.len();
        let stride = (num_scores + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;

        // The indices are now based on the stride, not the original number of scores.
        let weights_start = reconciled_snps_processed * stride;
        let weights_end = (reconciled_snps_processed + num_reconciled_in_chunk) * stride;

        let mut partial_scores_buffer = partial_scores_pool_clone.pop().unwrap();

        // Dispatch the computation. We pass the raw buffer slice as before, but now also
        // include the specific sub-problem parameters for the compute engine.
        let compute_handle = task::spawn_blocking(move || {
            let buffer_slice = &full_buffer[..bytes_read];
            let weights_for_chunk = prep_clone
                .interleaved_weights
                .get(weights_start..weights_end)
                .ok_or("Internal error: Failed to slice weights buffer.")?;

            batch::run_chunk_computation(
                buffer_slice,
                weights_for_chunk,
                &prep_clone,
                &mut partial_scores_buffer,
                &tile_pool_clone,
                &sparse_index_pool_clone,
                reconciled_indices_start,
                bed_row_offset,
            )?;
            Ok::<_, Box<dyn Error + Send + Sync>>((full_buffer, partial_scores_buffer))
        });

        let (returned_io_buffer, mut returned_scores_buffer) = match compute_handle.await? {
            Ok(buffers) => buffers,
            Err(e) => return Err(e),
        };

        if empty_buffer_tx.send(returned_io_buffer).await.is_err() {
            break;
        }

        for (master_score, partial_score) in all_scores.iter_mut().zip(returned_scores_buffer.iter()) {
            *master_score += partial_score;
        }

        returned_scores_buffer.iter_mut().for_each(|s| *s = 0.0);
        partial_scores_pool_clone.push(returned_scores_buffer).unwrap();

        // Update both counters before processing the next chunk.
        bed_row_offset += snps_in_chunk;
        reconciled_snps_processed += num_reconciled_in_chunk;

        // The progress report is now accurate.
        eprintln!(
            "> Processed chunk: {}/{} SNPs ({:.1}%)",
            reconciled_snps_processed,
            prep_result.num_reconciled_snps,
            (reconciled_snps_processed as f32 / prep_result.num_reconciled_snps as f32) * 100.0
        );
    }

    // Await the I/O task to make sure it has finished.
    io_handle.await?;
    eprintln!("> Computation finished.");

    // --- Phase 5: Finalization and Output ---
    let out_filename = match args.score.file_name() {
        Some(name) => {
            let mut s = OsString::from(name);
            s.push(".sscore");
            Ok(s)
        }
        None => Err(format!(
            "Could not determine a base name from score file path '{}'.",
            args.score.display()
        )),
    }?;
    let out_path = output_dir.join(&out_filename);
    eprintln!(
        "> Writing {} scores per person to {}",
        prep_result.score_names.len(),
        out_path.display()
    );

    write_scores_to_file(
        &out_path,
        &prep_result.final_person_iids,
        &prep_result.score_names,
        &all_scores,
    )?;

    eprintln!(
        "\nSuccess! Total execution time: {:.2?}",
        start_time.elapsed()
    );

    Ok(())
}

// ========================================================================================
//                                  HELPER FUNCTIONS
// ========================================================================================

/// Intelligently resolves the user-provided input path to a PLINK prefix.
///
/// This function is designed to be flexible and supports three common ways of
/// specifying a PLINK fileset:
/// 1. By providing a path to the directory containing a single `.bed` file.
/// 2. By providing a direct path to the `.bed` file itself.
/// 3. By providing the fileset's prefix (e.g., `data/my_plink_files`), which
///    does not exist as a file or directory itself.
fn resolve_plink_prefix(path: &Path) -> Result<PathBuf, String> {
    // Case 1: The path is a directory. Search for a unique .bed file inside.
    if path.is_dir() {
        let bed_files: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| format!("Could not read directory '{}': {}", path.display(), e))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "bed"))
            .collect();

        match bed_files.len() {
            0 => Err(format!("No .bed file found in the directory '{}'.", path.display())),
            1 => Ok(bed_files[0].with_extension("")),
            _ => Err(format!(
                "Ambiguous input: multiple .bed files found in directory '{}': {:?}",
                path.display(),
                bed_files
            )),
        }
    }
    // Case 2: The path is a direct file path. Check if it's a .bed file.
    else if path.is_file() {
        if path.extension().map_or(false, |ext| ext == "bed") {
            Ok(path.with_extension(""))
        } else {
            // If it's a file but not a .bed, this is an explicit error, as the user
            // pointed to a specific, but incorrect, file type.
            Err(format!(
                "Input file '{}' must have a .bed extension.",
                path.display()
            ))
        }
    }
    // Case 3: The path does not exist as a file or directory. Treat it as a prefix string.
    // We construct the expected .bed file path and check for its existence.
    else {
        let bed_path_from_prefix = path.with_extension("bed");
        if bed_path_from_prefix.is_file() {
            Ok(path.to_path_buf())
        } else {
            // If all attempts fail, return a comprehensive error message.
            Err(format!(
                "Input '{}' is not a valid directory, .bed file, or PLINK prefix.",
                path.display()
            ))
        }
    }
}

/// Writes the final calculated scores to a self-describing, tab-separated file.
fn write_scores_to_file(
    path: &Path,
    person_iids: &[String],
    score_names: &[String],
    scores: &[f32],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let num_scores = score_names.len();

    // --- Write Header ---
    write!(writer, "#IID")?;
    for name in score_names {
        write!(writer, "\t{}", name)?;
    }
    writeln!(writer)?;

    // --- Write Data Rows ---
    let line_buffer_capacity = person_iids.get(0).map_or(128, |s| s.len() + num_scores * 12);
    let mut line_buffer = String::with_capacity(line_buffer_capacity);
    let mut score_chunks = scores.chunks_exact(num_scores);

    for iid in person_iids {
        let person_scores = score_chunks.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Mismatched number of persons and score rows during final write. This is a critical internal error.")
        })?;

        line_buffer.clear();
        write!(&mut line_buffer, "{}", iid).unwrap();

        for &score in person_scores {
            write!(&mut line_buffer, "\t{}", score).unwrap();
        }

        writeln!(writer, "{}", line_buffer)?;
    }

    writer.flush()
}
