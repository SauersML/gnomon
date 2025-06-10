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
use gnomon::prepare;
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
    /// Path to the PLINK .bed file, or a directory containing a single .bed file.
    input_path: PathBuf,

    /// Path to the score file.
    #[clap(long)]
    score: PathBuf,

    /// Path to a file containing a list of individual IDs (IIDs) to include.
    /// If not provided, all individuals in the .fam file will be scored.
    #[clap(long)]
    keep: Option<PathBuf>,
}

// ========================================================================================
//                           THE MAIN ORCHESTRATION LOGIC
// ========================================================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let start_time = Instant::now();

    // --- Phase 1: Argument Parsing and Path Resolution ---
    let args = Args::parse();
    let plink_prefix = resolve_plink_prefix(&args.input_path)?;
    eprintln!("> Using PLINK prefix: {}", plink_prefix.display());

    let output_dir = plink_prefix.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;

    // --- Phase 2: The Preparation Phase ---
    eprintln!(
        "> Preparing data using score file: {}",
        args.score.display()
    );
    let prep_result = Arc::new(prepare::prepare_for_computation(
        &plink_prefix,
        &args.score,
        args.keep.as_deref(),
    )?);

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
    let (full_buffer_tx, mut full_buffer_rx) =
        mpsc::channel::<(Vec<u8>, usize)>(PIPELINE_DEPTH);
    let (empty_buffer_tx, mut empty_buffer_rx) = mpsc::channel::<Vec<u8>>(PIPELINE_DEPTH);

    eprintln!(
        "> Dynamically configured I/O chunk size to {} MB and pipeline depth to {}",
        chunk_size_bytes / 1024 / 1024,
        PIPELINE_DEPTH
    );

    // --- Phase 4: True Concurrent Pipeline Execution ---
    // The message passed from the I/O task to the compute dispatcher.
    type IoMessage = (Vec<u8>, usize, usize); // (buffer, bytes_read, snps_in_chunk)

    // Prime the pipeline by sending the empty I/O buffers to the I/O task.
    for _ in 0..PIPELINE_DEPTH {
        empty_buffer_tx
            .send(vec![0u8; chunk_size_bytes])
            .await?;
    }

    let bed_path = plink_prefix.with_extension("bed");
    let mut reader = SnpChunkReader::new(
        &bed_path,
        prep_result.total_people_in_fam,
        prep_result.total_snps_in_bim,
    )?;

    // Spawn the dedicated I/O Producer task.
    let io_handle = tokio::spawn(async move {
        // The I/O task owns the reader and the channels.
        while let Some(mut buffer) = empty_buffer_rx.recv().await {
            // The synchronous read_chunk call MUST be wrapped in spawn_blocking
            // to prevent it from blocking the async I/O task on a page fault.
            let read_result = task::spawn_blocking(move || {
                let bytes_read = reader.read_chunk(&mut buffer)?;
                let snps_in_chunk = reader.snps_in_bytes(bytes_read);
                Ok::<_, io::Error>((buffer, bytes_read, snps_in_chunk))
            })
            .await
            .unwrap(); // Panicking here is acceptable; a JoinError is unrecoverable.

            match read_result {
                Ok((buffer, 0, _)) => break, // EOF
                Ok((buffer, bytes_read, snps_in_chunk)) => {
                    if full_buffer_tx.send((buffer, bytes_read, snps_in_chunk)).await.is_err() {
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
    let mut snps_processed_so_far = 0;
    while let Some((full_buffer, bytes_read, snps_in_chunk)) = full_buffer_rx.recv().await {
        if snps_in_chunk == 0 {
            // If the chunk was empty, immediately recycle the buffer.
            if empty_buffer_tx.send(full_buffer).await.is_err() {
                break;
            }
            continue;
        }

        let prep_clone = Arc::clone(&prep_result);
        let tile_pool_clone = Arc::clone(&tile_pool);
        let partial_scores_pool_clone = Arc::clone(&partial_scores_pool);
        let sparse_index_pool_clone = Arc::clone(&sparse_index_pool);

        let weights_start = snps_processed_so_far * prep_clone.score_names.len();
        let weights_end = (snps_processed_so_far + snps_in_chunk) * prep_clone.score_names.len();
        let weights_for_chunk = prep_clone.interleaved_weights.get(weights_start..weights_end)
            .ok_or("Internal error: Failed to slice weights buffer.")?;

        // Acquire a reusable buffer for partial scores to avoid allocation.
        let mut partial_scores_buffer = partial_scores_pool_clone.pop().unwrap();

        // Dispatch compute task to the blocking pool.
        let compute_handle = task::spawn_blocking(move || {
            let buffer_slice = &full_buffer[..bytes_read];
            batch::run_chunk_computation(
                buffer_slice,
                weights_for_chunk,
                &prep_clone,
                &mut partial_scores_buffer, // Mutates the buffer in-place
                &tile_pool_clone,
                &sparse_index_pool_clone,
            )?;
            // Return ownership of both buffers for reuse.
            Ok::<_, Box<dyn Error + Send + Sync>>((full_buffer, partial_scores_buffer))
        });

        // Await the results of the computation.
        let (returned_io_buffer, returned_scores_buffer) = compute_handle.await??;

        // Recycle the I/O buffer by sending it back to the I/O task.
        if empty_buffer_tx.send(returned_io_buffer).await.is_err() {
            // I/O task has shut down, we can't continue.
            break;
        }

        // Perform the sequential, non-contentious merge step.
        for (master_score, partial_score) in all_scores.iter_mut().zip(returned_scores_buffer.iter()) {
            *master_score += partial_score;
        }
        
        // Before returning the scores buffer to the pool, it must be cleared.
        returned_scores_buffer.iter_mut().for_each(|s| *s = 0.0);
        partial_scores_pool_clone.push(returned_scores_buffer).unwrap();

        snps_processed_so_far += snps_in_chunk;
        eprintln!(
            "> Processed chunk: {}/{} SNPs ({:.1}%)",
            snps_processed_so_far,
            prep_result.num_reconciled_snps,
            (snps_processed_so_far as f32 / prep_result.num_reconciled_snps as f32) * 100.0
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
fn resolve_plink_prefix(path: &Path) -> Result<PathBuf, String> {
    if !path.exists() {
        return Err(format!("Input path '{}' does not exist.", path.display()));
    }

    if path.is_dir() {
        let bed_files: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| format!("Could not read directory '{}': {}", path.display(), e))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|p| p.extension().map_or(false, |ext| ext == "bed"))
            .collect();

        match bed_files.len() {
            0 => Err("No .bed file found in the specified directory.".to_string()),
            1 => Ok(bed_files[0].with_extension("")),
            _ => Err(format!(
                "Ambiguous input: multiple .bed files found in directory: {:?}",
                bed_files
            )),
        }
    } else if path.is_file() {
        if path.extension().map_or(false, |ext| ext == "bed") {
            Ok(path.with_extension(""))
        } else {
            Err("Input file must have a .bed extension.".to_string())
        }
    } else {
        Err("Input path is not a file or directory.".to_string())
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
