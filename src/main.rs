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
#[cfg(debug_assertions)]
use std::cell::Cell;
use tokio::sync::mpsc::{error::SendError, Sender};

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
//                           THE MAIN ORCHESTRATION LOGIC
// ========================================================================================

/// A message passed from the I/O producer task to the compute dispatcher.
struct IoMessage {
    buffer: Vec<u8>,
    bytes_read: usize,
    snps_in_chunk: usize,
}

/// A structure that bundles the I/O buffer and the partial scores buffer for a
/// single computational chunk. This type enforces that the partial scores are
/// merged before the I/O buffer can be recycled, preventing data loss.
struct ChunkResult {
    io_buffer: Vec<u8>,
    partial_scores: Vec<f32>,
    #[cfg(debug_assertions)]
    committed: Cell<bool>,
}

impl ChunkResult {
    /// Creates a new `ChunkResult` from its constituent buffers.
    fn new(io_buffer: Vec<u8>, partial_scores: Vec<f32>) -> Self {
        Self {
            io_buffer,
            partial_scores,
            #[cfg(debug_assertions)]
            committed: Cell::new(false),
        }
    }

    /// Atomically merges partial scores, sends the I/O buffer back to the
    /// producer, and returns the now-cleared partial scores buffer for reuse.
    #[must_use = "The result of the commit must be handled to recycle buffers and check for errors."]
    async fn commit(
        self,
        all_scores: &mut [f32],
        empty_tx: &Sender<Vec<u8>>,
    ) -> (Vec<f32>, Result<(), SendError<Vec<u8>>>) {
        let this = std::mem::ManuallyDrop::new(self);
        let io_buffer = unsafe { std::ptr::read(&this.io_buffer) };
        let mut recycled_scores_buffer = unsafe { std::ptr::read(&this.partial_scores) };

        for (master_score, partial_score) in all_scores.iter_mut().zip(&recycled_scores_buffer) {
            *master_score += partial_score;
        }

        #[cfg(debug_assertions)]
        this.committed.set(true);

        let send_result = empty_tx.send(io_buffer).await;
        recycled_scores_buffer.fill(0.0);
        (recycled_scores_buffer, send_result)
    }
}

/// A debug-only RAII guard that panics if a `ChunkResult` is dropped without
/// being explicitly committed.
#[cfg(debug_assertions)]
impl Drop for ChunkResult {
    fn drop(&mut self) {
        if !self.committed.get() {
            if self.partial_scores.iter().any(|&score| score != 0.0) {
                panic!("CRITICAL BUG: A `ChunkResult` with non-zero scores was dropped without being committed. This indicates a silent data loss has occurred.");
            }
        }
    }
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

    eprintln!("> Preparing data using {} score file(s)...", score_files.len());

    // Call the preparation compiler, handling potential parsing errors by offering to reformat.
    let prep_result = match prepare::prepare_for_computation(
        &plink_prefix,
        &score_files, // Pass the entire slice of paths, fixing the E0308 error.
        args.keep.as_deref(),
    ) {
        Ok(result) => Ok(result),
        // If preparation fails on a single file, offer to reformat it.
        Err(prep_error @ PrepError::Header(_)) | Err(prep_error @ PrepError::Parse(_)) if score_files.len() == 1 => {
            let score_path_to_check = &score_files[0];
            eprintln!("\nWarning: Initial parsing of '{}' failed. Checking for PGS Catalog format...", score_path_to_check.display());
            match reformat::reformat_pgs_file(score_path_to_check) {
                Ok(new_path) => {
                    // HONEST UX: Do not auto-retry. Inform the user and exit successfully.
                    eprintln!("\nSuccess! A gnomon-compatible version of your score file has been created at:");
                    eprintln!("  {}", new_path.display());
                    eprintln!("\nPlease re-run your command using this new file path (or a directory containing it).");
                    return Ok(()); // Successful exit, no error code.
                }
                Err(reformat_error) => {
                    eprintln!("\nError: Automatic conversion failed: {}", reformat_error);
                    eprintln!("Please ensure the score file is in the gnomon-native format (snp_id, effect_allele, other_allele, ...) or a valid PGS Catalog format.");
                    Err(prep_error) // Propagate the original, more specific error.
                }
            }
        }
        // For any other error, or for errors with multiple files, just fail.
        Err(other_error) => Err(other_error),
    };

    let prep_result = Arc::new(prep_result?);

    eprintln!(
        "> Preparation complete. Found {} individuals to score and {} overlapping variants across {} score(s).",
        prep_result.num_people_to_score, prep_result.num_reconciled_variants, prep_result.score_names.len()
    );

    // --- Phase 3: Dynamic Runtime Resource Allocation ---
    let mut all_scores =
        vec![0.0f32; prep_result.num_people_to_score * prep_result.score_names.len()];
    let tile_pool = Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 2));
    let sparse_index_pool = Arc::new(SparseIndexPool::new());

    const MIN_CHUNK_SIZE: u64 = 64 * 1024 * 1024;
    const MAX_CHUNK_SIZE: u64 = 1 * 1024 * 1024 * 1024;
    let mut sys = System::new_all();
    sys.refresh_memory();
    let available_mem = sys.available_memory();
    let chunk_size_bytes = (available_mem / 4).clamp(MIN_CHUNK_SIZE, MAX_CHUNK_SIZE) as usize;

    const PIPELINE_DEPTH: usize = 2;
    let partial_scores_size = prep_result.num_people_to_score * prep_result.score_names.len();
    let partial_scores_pool = Arc::new(ArrayQueue::new(PIPELINE_DEPTH + 1));
    for _ in 0..(PIPELINE_DEPTH + 1) {
        partial_scores_pool.push(vec![0.0f32; partial_scores_size]).unwrap();
    }
    let (full_buffer_tx, mut full_buffer_rx) = mpsc::channel::<IoMessage>(PIPELINE_DEPTH);
    let (empty_buffer_tx, mut empty_buffer_rx) = mpsc::channel::<Vec<u8>>(PIPELINE_DEPTH);

    eprintln!(
        "> Dynamically configured I/O chunk size to {} MB and pipeline depth to {}",
        chunk_size_bytes / 1024 / 1024,
        PIPELINE_DEPTH
    );

    // --- Phase 4: True Concurrent Pipeline Execution ---
    for _ in 0..PIPELINE_DEPTH {
        empty_buffer_tx.send(vec![0u8; chunk_size_bytes]).await?;
    }

    let bed_path = plink_prefix.with_extension("bed");
    let reader = SnpChunkReader::new(
        &bed_path,
        prep_result.bytes_per_snp,
        prep_result.total_snps_in_bim,
    )?;

    let io_handle = tokio::spawn(async move {
        let mut reader = reader;
        while let Some(mut buffer) = empty_buffer_rx.recv().await {
            let read_result = task::spawn_blocking(move || {
                let bytes_read = reader.read_chunk(&mut buffer)?;
                let snps_in_chunk = reader.snps_in_bytes(bytes_read);
                Ok::<_, io::Error>((reader, buffer, bytes_read, snps_in_chunk))
            })
            .await.unwrap();

            match read_result {
                Ok((_, _, 0, _)) => break,
                Ok((new_reader, buffer, bytes_read, snps_in_chunk)) => {
                    reader = new_reader;
                    if full_buffer_tx.send(IoMessage { buffer, bytes_read, snps_in_chunk }).await.is_err() {
                        break;
                    }
                }
                Err(e) => {
                    eprintln!("[I/O Task Error]: {}", e);
                    break;
                }
            }
        }
    });

    let mut bed_row_offset: usize = 0;

    while let Some(IoMessage { buffer: full_buffer, bytes_read, snps_in_chunk }) = full_buffer_rx.recv().await {
        let bed_row_end = bed_row_offset + snps_in_chunk;

        let matrix_row_start = prep_result.required_bim_indices.partition_point(|&x| x < bed_row_offset);
        let matrix_row_end = prep_result.required_bim_indices.partition_point(|&x| x < bed_row_end);

        if matrix_row_end == matrix_row_start {
            if empty_buffer_tx.send(full_buffer).await.is_err() { break; }
            bed_row_offset += snps_in_chunk;
            continue;
        }

        let prep_clone = Arc::clone(&prep_result);
        let tile_pool_clone = Arc::clone(&tile_pool);
        let partial_scores_pool_clone = Arc::clone(&partial_scores_pool);
        let sparse_index_pool_clone = Arc::clone(&sparse_index_pool);
        let stride = prep_clone.stride;
        let matrix_slice_start = matrix_row_start * stride;
        let matrix_slice_end = matrix_row_end * stride;
        let mut partial_scores_buffer = partial_scores_pool_clone.pop().unwrap();

        let compute_handle = task::spawn_blocking(move || {
            let buffer_slice = &full_buffer[..bytes_read];
            
            let weights_for_chunk = prep_clone
                .aligned_weights_matrix
                .get(matrix_slice_start..matrix_slice_end)
                .ok_or_else(|| Box::<dyn Error + Send + Sync>::from("Internal error: Failed to slice aligned weights matrix."))?;
                
            let corrections_for_chunk = prep_clone
                .correction_constants_matrix
                .get(matrix_slice_start..matrix_slice_end)
                .ok_or_else(|| Box::<dyn Error + Send + Sync>::from("Internal error: Failed to slice correction constants matrix."))?;

            batch::run_chunk_computation(
                buffer_slice,
                weights_for_chunk,
                corrections_for_chunk,
                &prep_clone,
                &mut partial_scores_buffer,
                &tile_pool_clone,
                &sparse_index_pool_clone,
                matrix_row_start,
                bed_row_offset,
            )?;

            let result = ChunkResult::new(full_buffer, partial_scores_buffer);
            Ok::<_, Box<dyn Error + Send + Sync>>(result)
        });

        let chunk_result = match compute_handle.await? {
            Ok(result) => result,
            Err(e) => return Err(e),
        };

        let (recycled_scores_buffer, send_result) = chunk_result.commit(&mut all_scores, &empty_buffer_tx).await;
        partial_scores_pool_clone.push(recycled_scores_buffer).unwrap();

        if send_result.is_err() {
            break;
        }

        bed_row_offset += snps_in_chunk;

        eprintln!(
            "> Processed chunk: {}/{} variants ({:.1}%)",
            matrix_row_end,
            prep_result.num_reconciled_variants,
            (matrix_row_end as f32 / prep_result.num_reconciled_variants as f32) * 100.0
        );
    }

    io_handle.await?;
    eprintln!("> Computation finished.");

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

    write_scores_to_file(
        &out_path,
        &prep_result.final_person_iids,
        &prep_result.score_names,
        &all_scores,
    )?;

    eprintln!("\nSuccess! Total execution time: {:.2?}", start_time.elapsed());

    Ok(())
}

// ========================================================================================
//                                  HELPER FUNCTIONS
// ========================================================================================

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
fn write_scores_to_file(
    path: &Path,
    person_iids: &[String],
    score_names: &[String],
    scores: &[f32],
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let num_scores = score_names.len();

    write!(writer, "#IID")?;
    for name in score_names {
        write!(writer, "\t{}", name)?;
    }
    writeln!(writer)?;

    let mut line_buffer = String::with_capacity(person_iids.get(0).map_or(128, |s| s.len() + num_scores * 12));
    let mut score_chunks = scores.chunks_exact(num_scores);

    for iid in person_iids {
        let person_scores = score_chunks.next().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "Mismatched number of persons and score rows during final write.")
        })?;
        line_buffer.clear();
        write!(&mut line_buffer, "{}", iid).unwrap();
        for &score in person_scores {
            write!(&mut line_buffer, "\t{:.6}", score).unwrap();
        }
        writeln!(writer, "{}", line_buffer)?;
    }

    writer.flush()
}
