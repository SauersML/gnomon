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
use gnomon::io::SnpChunkReader;
use gnomon::io::SnpChunk;
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
use tokio::sync::mpsc;
use tokio::task;
use std::fs::File;
#[cfg(debug_assertions)]
use std::cell::Cell;
use tokio::sync::mpsc::{error::SendError, Sender};
use cache_size;

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

/// A structure that bundles the I/O buffer and the partial result buffers for a
/// single computational chunk. This type enforces that the partial results are
/// merged before the I/O buffer can be recycled, preventing data loss.
struct ChunkResult {
    chunk: SnpChunk,
    partial_scores: Vec<f32>,
    partial_missing_counts: Vec<u32>,
    partial_correction_sums: Vec<f32>,
    #[cfg(debug_assertions)]
    committed: Cell<bool>,
}

impl ChunkResult {
    /// Creates a new `ChunkResult` from its constituent buffers.
    fn new(
        chunk: SnpChunk,
        partial_scores: Vec<f32>,
        partial_missing_counts: Vec<u32>,
        partial_correction_sums: Vec<f32>,
    ) -> Self {
        Self {
            chunk,
            partial_scores,
            partial_missing_counts,
            partial_correction_sums,
            #[cfg(debug_assertions)]
            committed: Cell::new(false),
        }
    }

    /// Atomically merges all partial results, sends the I/O buffer back to the
    /// producer, and returns the now-cleared partial result buffers for reuse.
    #[must_use = "The result of the commit must be handled to recycle buffers and check for errors."]
    async fn commit(
        self,
        all_scores: &mut [f32],
        all_missing_counts: &mut [u32],
        all_correction_sums: &mut [f32],
        empty_tx: &Sender<Vec<u8>>,
    ) -> (
        (Vec<f32>, Vec<u32>, Vec<f32>),
        Result<(), SendError<Vec<u8>>>,
    ) {
        let this = std::mem::ManuallyDrop::new(self);
        // Unpack the guaranteed-valid chunk to retrieve the underlying buffer for recycling.
        let SnpChunk {
            buffer: io_buffer, ..
        } = unsafe { std::ptr::read(&this.chunk) };
        let mut recycled_scores_buffer = unsafe { std::ptr::read(&this.partial_scores) };
        let mut recycled_missing_counts_buffer =
            unsafe { std::ptr::read(&this.partial_missing_counts) };
        let mut recycled_correction_sums_buffer =
            unsafe { std::ptr::read(&this.partial_correction_sums) };

        // Aggregate all partial result buffers into their master counterparts.
        for (master, partial) in all_scores.iter_mut().zip(&recycled_scores_buffer) {
            *master += partial;
        }
        for (master, partial) in all_missing_counts.iter_mut().zip(&recycled_missing_counts_buffer)
        {
            *master += partial;
        }
        for (master, partial) in all_correction_sums.iter_mut().zip(&recycled_correction_sums_buffer)
        {
            *master += partial;
        }

        #[cfg(debug_assertions)]
        this.committed.set(true);

        let send_result = empty_tx.send(io_buffer).await;

        // Clear buffers for reuse.
        recycled_scores_buffer.fill(0.0);
        recycled_missing_counts_buffer.fill(0);
        recycled_correction_sums_buffer.fill(0.0);

        (
            (
                recycled_scores_buffer,
                recycled_missing_counts_buffer,
                recycled_correction_sums_buffer,
            ),
            send_result,
        )
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
    let overall_start_time = Instant::now();

    // --- Phase 1: Argument Parsing and Path Resolution ---
    let args = Args::parse();
    let plink_prefix = resolve_plink_prefix(&args.input_path)?;
    eprintln!("> Using PLINK prefix: {}", plink_prefix.display());
    let output_dir = plink_prefix.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;

    // --- Phase 2: The Preparation Phase ---
    // This logic now supports providing a directory to the --score argument.
    let mut score_files = if args.score.is_dir() {
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

    // This block of code adds detailed timing points for the resource allocation phase.
    let t0 = Instant::now();

    let num_scores = prep_result.score_names.len();
    let result_buffer_size = prep_result.num_people_to_score * num_scores;
    let mut all_scores = vec![0.0f32; result_buffer_size];
    let mut all_missing_counts = vec![0u32; result_buffer_size];
    let mut all_correction_sums = vec![0.0f32; result_buffer_size];

    let t1 = Instant::now();

    // Initialize the final scores buffer with the dosage-independent base scores.
    // The kernel will then compute the dosage-dependent deltas which are aggregated.
    for person_scores_slice in all_scores.chunks_mut(num_scores) {
        person_scores_slice.copy_from_slice(&prep_result.base_scores);
    }

    let t2 = Instant::now();

    let tile_pool = Arc::new(ArrayQueue::new(num_cpus::get().max(1) * 2));
    let sparse_index_pool = Arc::new(SparseIndexPool::new());

    let t3 = Instant::now();

    // GOAL: Find the largest I/O Chunk Size that produces a compute `tile` that
    //       fits within the L3 cache, thus co-optimizing I/O and compute.
    //
    // VARIABLES:
    // S_io_chunk_opt: Optimal I/O Chunk Size (bytes) - The value we want to find.
    // C_L3:           L3 Cache Size (bytes)
    // S_person_block: The `PERSON_BLOCK_SIZE` constant (4096)
    // N_people:       Total number of individuals in the study
    // N_total_snps:   Total number of SNPs in the genome file
    // N_score_snps:   Number of relevant SNPs used in the analysis
    // B_snp:          Bytes per SNP in the .bed file = CEIL(N_people / 4)
    // D_score:        Density of relevant SNPs = N_score_snps / N_total_snps
    //
    // DERIVATION:
    // 1. The L3 cache constraint: Size(tile) <= C_L3
    // 2. Tile size definition: Size(tile) = S_person_block * N_chunk_snps_max
    // 3. From (1) and (2), max SNPs per tile: N_chunk_snps_max = C_L3 / S_person_block
    // 4. Link tile to I/O chunk: N_chunk_snps_max = (S_io_chunk_opt / B_snp) * D_score
    // 5. Solving for S_io_chunk_opt: S_io_chunk_opt = (C_L3 * B_snp) / (S_person_block * D_score)
    //
    // FINAL FORMULA:
    // S_io_chunk_opt = (C_L3 * N_total_snps * CEIL(N_people / 4)) / (S_person_block * N_score_snps)
    //
    const PERSON_BLOCK_SIZE: u64 = 4096; // Must match the value in `batch.rs`

    // 1. Determine L3 Cache Size, with a robust fallback.
    const FALLBACK_L3_CACHE_BYTES: usize = 32 * 1024 * 1024; // 32 MiB
    const MINIMUM_PLAUSIBLE_L3_CACHE_BYTES: usize = 1 * 1024 * 1024; // 1 MiB

    // Attempt to detect the L3 cache size. If the detected size is too small to be a
    // plausible L3 cache (common in virtualized environments), treat it as a
    // detection failure and use the safe fallback value.
    let l3_cache_bytes = cache_size::l3_cache_size()
        .and_then(|size| {
            if size >= MINIMUM_PLAUSIBLE_L3_CACHE_BYTES {
                Some(size)
            } else {
                // A detected size that is non-zero but smaller than the minimum plausible
                // size is logged as a warning before reverting to the fallback.
                eprintln!(
                    "> Warning: Detected cache size ({} KiB) is implausibly small for L3. Reverting to fallback.",
                    size / 1024
                );
                None
            }
        })
        .unwrap_or_else(|| {
            eprintln!(
                "> L3 cache size not detected or is implausible. Using safe fallback: {} MiB.",
                FALLBACK_L3_CACHE_BYTES / 1024 / 1024
            );
            FALLBACK_L3_CACHE_BYTES
        });

    let t4 = Instant::now();

    // This prints the cache size actually being used for the calculation, using
    // floating-point division for accuracy and including the raw byte count for clarity.
    eprintln!(
        "> Using L3 cache size for optimization: {:.2} MiB ({} bytes)",
        l3_cache_bytes as f64 / (1024.0 * 1024.0),
        l3_cache_bytes
    );

    // 2. Get data parameters from the `prep_result`.
    let n_people = prep_result.total_people_in_fam as u64;
    let n_total_snps = prep_result.total_snps_in_bim as u64;
    let n_score_snps = prep_result.num_reconciled_variants as u64;

    // 3. Calculate the theoretically optimal chunk size using our formula.
    let mut optimal_chunk_size: u64 = 64 * 1024 * 1024; // Default to 64MB.

    if n_score_snps > 0 {
        let bytes_per_snp = (n_people + 3) / 4;

        // Use u128 for the intermediate multiplication to prevent overflow.
        let numerator = (l3_cache_bytes as u128)
            * (n_total_snps as u128)
            * (bytes_per_snp as u128);
        let denominator = (PERSON_BLOCK_SIZE as u128) * (n_score_snps as u128);

        if denominator > 0 {
            optimal_chunk_size = (numerator / denominator) as u64;
        }
    }

    let t5 = Instant::now();

    // 4. Apply practical guardrails to the optimal value. We assume sufficient RAM
    // and apply a fixed upper and lower bound for stability and I/O efficiency.
    const DYNAMIC_MIN_CHUNK_SIZE: u64 = 4 * 1024 * 1024; // 4 MB floor for I/O efficiency.
    const ABSOLUTE_MAX_CHUNK_SIZE: u64 = 512 * 1024 * 1024; // 512 MB fixed safety cap.

    let chunk_size_bytes = optimal_chunk_size
        .clamp(DYNAMIC_MIN_CHUNK_SIZE, ABSOLUTE_MAX_CHUNK_SIZE) as usize;

    let t6 = Instant::now();

    const PIPELINE_DEPTH: usize = 2;
    let partial_result_pool = Arc::new(ArrayQueue::new(PIPELINE_DEPTH + 1));
    for _ in 0..(PIPELINE_DEPTH + 1) {
        partial_result_pool
            .push((
                vec![0.0f32; result_buffer_size],
                vec![0u32; result_buffer_size],
                vec![0.0f32; result_buffer_size],
            ))
            .unwrap();
    }

    let t7 = Instant::now();

    let (full_buffer_tx, mut full_buffer_rx) = mpsc::channel::<SnpChunk>(PIPELINE_DEPTH);
    let (empty_buffer_tx, mut empty_buffer_rx) = mpsc::channel::<Vec<u8>>(PIPELINE_DEPTH);

    let t8 = Instant::now();

    {
        eprintln!("[Allocation Benchmark]");
        eprintln!("  - Main result buffer alloc:      {:.2?}", t1 - t0);
        eprintln!("  - Main result buffer init:       {:.2?}", t2 - t1);
        eprintln!("  - Basic worker pools alloc:      {:.2?}", t3 - t2);
        eprintln!("  - L3 cache detection:            {:.2?}", t4 - t3);
        eprintln!("  - Chunk size formula calc:       {:.2?}", t5 - t4);
        eprintln!("  - Chunk size clamping:           {:.2?}", t6 - t5);
        eprintln!("  - Partial result pool alloc:     {:.2?}", t7 - t6);
        eprintln!("  - Pipeline channels setup:       {:.2?}", t8 - t7);
    }

    eprintln!(
        "> Resource allocation complete in {:.2?}. Determined I/O chunk size to {} MB and pipeline depth to {}",
        resource_alloc_start.elapsed(),
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
        // The producer loop is now free of logic. It just calls the reader factory
        // and sends the resulting guaranteed-valid chunk to the consumer.
        'producer: while let Some(mut buffer) = empty_buffer_rx.recv().await {
            let read_result = task::spawn_blocking(move || {
                // The factory method returns a valid chunk or None at EOF.
                let chunk_opt = reader.read_next_chunk(&mut buffer)?;
                // The buffer is passed back, potentially empty if it was used.
                Ok::<_, io::Error>((reader, buffer, chunk_opt))
            })
            .await
            .unwrap();

            match read_result {
                Ok((new_reader, _cleared_buffer, Some(chunk))) => {
                    reader = new_reader;
                    // Send the validated chunk to the consumer.
                    if full_buffer_tx.send(chunk).await.is_err() {
                        // Consumer has hung up, no point in continuing.
                        break 'producer;
                    }
                }
                Ok((new_reader, unused_buffer, None)) => {
                    // EOF. The reader is now exhausted.
                    reader = new_reader;
                    // The buffer we received was not used by the factory. We drop it.
                    // The producer loop can now terminate.
                    drop(unused_buffer);
                    break 'producer;
                }
                Err(e) => {
                    eprintln!("[I/O Task Error]: {}", e);
                    break 'producer;
                }
            }
        }
        // The producer loop has finished (due to EOF, error, or consumer hang-up).
        // The `full_buffer_tx` is now dropped, which will cause the consumer
        // loop to terminate gracefully after processing any in-flight chunks.
    });

    let mut bed_row_offset: usize = 0;

    let computation_start = Instant::now();
    while let Some(chunk) = full_buffer_rx.recv().await {
        let snps_in_chunk = chunk.num_snps;
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
        let partial_result_pool_clone = Arc::clone(&partial_result_pool);
        let sparse_index_pool_clone = Arc::clone(&sparse_index_pool);
        let stride = prep_clone.stride;
        let matrix_slice_start = matrix_row_start * stride;
        let matrix_slice_end = matrix_row_end * stride;
        let (
            mut partial_scores_buffer,
            mut partial_missing_counts_buffer,
            mut partial_correction_sums_buffer,
        ) = partial_result_pool_clone.pop().unwrap();

        let compute_handle = task::spawn_blocking(move || {
            let weights_for_chunk = prep_clone
                .aligned_weights_matrix
                .get(matrix_slice_start..matrix_slice_end)
                .ok_or_else(|| {
                    Box::<dyn Error + Send + Sync>::from(
                        "Internal error: Failed to slice aligned weights matrix.",
                    )
                })?;

            // The buffer from the chunk is passed directly. No slicing is needed as its
            // length is guaranteed to be correct.
            batch::run_chunk_computation(
                &chunk.buffer,
                weights_for_chunk,
                &prep_clone,
                &mut partial_scores_buffer,
                &mut partial_missing_counts_buffer,
                &mut partial_correction_sums_buffer,
                &tile_pool_clone,
                &sparse_index_pool_clone,
                matrix_row_start,
                bed_row_offset,
            )?;

            let result = ChunkResult::new(
                chunk,
                partial_scores_buffer,
                partial_missing_counts_buffer,
                partial_correction_sums_buffer,
            );
            Ok::<_, Box<dyn Error + Send + Sync>>(result)
        });

        let chunk_result = match compute_handle.await? {
            Ok(result) => result,
            Err(e) => return Err(e),
        };

        let (recycled_buffers, send_result) = chunk_result
            .commit(
                &mut all_scores,
                &mut all_missing_counts,
                &mut all_correction_sums,
                &empty_buffer_tx,
            )
            .await;
        partial_result_pool_clone.push(recycled_buffers).unwrap();

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
        &all_correction_sums,
    )?;
    eprintln!("> Final output written in {:.2?}", output_start.elapsed());

    eprintln!("\nSuccess! Total execution time: {:.2?}", overall_start_time.elapsed());

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
/// This function now calculates the final per-variant average and missing percentage.
fn write_scores_to_file(
    path: &Path,
    person_iids: &[String],
    score_names: &[String],
    score_variant_counts: &[u32],
    sum_scores: &[f32],
    missing_counts: &[u32],
    correction_sums: &[f32],
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
    let mut correction_sum_chunks = correction_sums.chunks_exact(num_scores);

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
        let person_correction_sums = correction_sum_chunks.next().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "Mismatched number of persons and correction sum rows during final write.",
            )
        })?;

        line_buffer.clear();
        write!(&mut line_buffer, "{}", iid).unwrap();

        for i in 0..num_scores {
            let provisional_sum_score = person_sum_scores[i];
            let missing_count = person_missing_counts[i];
            let correction_for_missing = person_correction_sums[i];
            let total_variants_for_score = score_variant_counts[i];

            // This is the mathematically correct final sum after subtracting the
            // pre-added constants for variants that turned out to be missing.
            let final_sum_score = provisional_sum_score - correction_for_missing;

            let variants_used = total_variants_for_score.saturating_sub(missing_count);

            let avg_score = if variants_used > 0 {
                final_sum_score / (variants_used as f32)
            } else {
                0.0
            };

            let missing_pct = if total_variants_for_score > 0 {
                (missing_count as f32 / total_variants_for_score as f32) * 100.0
            } else {
                0.0
            };

            // Write the correctly tab-separated data columns.
            write!(&mut line_buffer, "\t{:.6}\t{:.4}", avg_score, missing_pct).unwrap();
        }
        writeln!(writer, "{}", line_buffer)?;
    }

    writer.flush()
}
