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
//
// ### The Orchestration Mandate ###
//
// 1.  **Smart Configuration:** The orchestrator presents a minimal, zero-friction
//     interface to the user. Complex tuning parameters (chunk sizes, buffer counts)
//     are internalized as developer-chosen constants, not exposed as user-facing knobs.
//
// 2.  **Intelligent Path Resolution:** It handles user-provided paths intelligently,
//     resolving directory paths to specific files and creating a deterministic output
//     structure without requiring boilerplate configuration from the user.
//
// 3.  **Resource Ownership:** `main` is the sole owner of all major resources: the
//     in-memory weight matrix, the pool of I/O pivot buffers, the pool of thread-local
//     kernel buffers, and the communication channels.
//
// 4.  **Pipeline Conduction:** It constructs the producer-consumer pipeline, spawns
//     the dedicated I/O thread, and executes the parallel compute consumer on the
//     main thread, driving the engine to completion.

use clap::Parser;
use gnomon::batch::{self, KernelDataPool, PipelineBuffer};
use gnomon::io::MmapBedReader;
use gnomon::prepare;
use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

// ========================================================================================
//                           APPLICATION-LEVEL CONSTANTS
// ========================================================================================

/// The number of SNPs to be read and pivoted in each pipeline chunk.
/// This value is a large power of two, which is efficient for memory alignment. It is
/// large enough to amortize the overhead of thread communication but small enough to
// likely fit the pivoted data for one person within the CPU's L2 cache.
const PIPELINE_CHUNK_SIZE_SNPS: usize = 8192;

/// The number of large, reusable pivot buffers used in the pipeline.
/// A value of 3 enables a classic "triple buffering" strategy, which is optimal for
/// pipeline throughput. At any given time, one buffer can be actively consumed by the
/// CPU cores, one can be actively produced by the I/O thread, and one remains free,
/// so neither the producer nor the consumer is ever starved for data.
const PIPELINE_BUFFER_COUNT: usize = 3;

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
}

// ========================================================================================
//                           THE MAIN ORCHESTRATION LOGIC
// ========================================================================================

fn main() {
    let start_time = Instant::now();

    // --- Phase 1: Argument Parsing and Path Resolution ---
    let args = Args::parse();

    let plink_prefix = match resolve_plink_prefix(&args.input_path) {
        Ok(prefix) => prefix,
        Err(e) => {
            eprintln!("Error resolving input path: {}", e);
            process::exit(1);
        }
    };
    eprintln!("> Using PLINK prefix: {}", plink_prefix.display());

    let output_dir = plink_prefix.parent().unwrap_or_else(|| Path::new("."));
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!(
            "Error creating output directory '{}': {}",
            output_dir.display(),
            e
        );
        process::exit(1);
    }

    // --- Phase 2: The Preparation Phase ---
    eprintln!(
        "> Preparing data using score file: {}",
        args.score.display()
    );
    // NOTE: This now passes paths directly, avoiding fragile string conversions.
    let prep_result = match prepare::prepare_for_computation(&plink_prefix, &args.score) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Fatal error during data preparation: {}", e);
            process::exit(1);
        }
    };
    eprintln!(
        "> Preparation complete. Found {} individuals and {} overlapping SNPs.",
        prep_result.num_people, prep_result.num_reconciled_snps
    );

    // --- Phase 3: Resource Allocation ---
    let mut all_scores =
        vec![0.0f32; prep_result.num_people * prep_result.num_scores];
    let kernel_data_pool = KernelDataPool::new();
    let pivot_buffer_capacity = prep_result.num_people * PIPELINE_CHUNK_SIZE_SNPS;
    let pipeline_buffers: Vec<_> = (0..PIPELINE_BUFFER_COUNT)
        .map(|_| PipelineBuffer::new(pivot_buffer_capacity))
        .collect();

    // --- Phase 4: Pipeline Construction and Execution ---
    eprintln!("> Starting pipeline with {} buffers...", PIPELINE_BUFFER_COUNT);
    let (free_buffers_tx, free_buffers_rx) =
        mpsc::sync_channel(PIPELINE_BUFFER_COUNT);
    let (filled_buffers_tx, filled_buffers_rx) =
        mpsc::sync_channel(PIPELINE_BUFFER_COUNT);

    thread::scope(|s| {
        // --- 4a: Spawn the Producer Thread ---
        let bed_path = plink_prefix.with_extension("bed");
        let producer_handle = s.spawn(move || {
            let bed_reader = MmapBedReader::new(
                &bed_path,
                prep_result.num_people,
                prep_result.total_snps_in_bim,
                prep_result.required_snp_indices,
                prep_result.reconciliation_instructions,
            )
            .expect("Failed to create .bed file reader"); // Panic is acceptable here per review.

            batch::producer_task(
                bed_reader,
                free_buffers_rx,
                filled_buffers_tx,
                PIPELINE_CHUNK_SIZE_SNPS,
                prep_result.num_people,
            );
        });

        // --- 4b: Execute the Consumer on the Main Thread ---
        for buffer in pipeline_buffers {
            let _ = free_buffers_tx.send(buffer);
        }

        let mut snps_processed = 0;
        while let Ok(filled_buffer) = filled_buffers_rx.recv() {
            let snps_in_chunk = filled_buffer.snps_in_chunk();
            let pivoted_data = filled_buffer.as_pivoted_data(prep_result.num_people);

            // This slicing logic is INTENTIONALLY here. The orchestrator owns the
            // loop state (`snps_processed`) and prepares a self-contained work
            // package for the stateless consumer function.
            let weights_chunk = &prep_result.interleaved_weights
                [snps_processed * prep_result.num_scores..];

            batch::process_pivoted_chunk(
                pivoted_data,
                weights_chunk,
                &mut all_scores,
                prep_result.num_people,
                snps_in_chunk,
                prep_result.num_scores,
                0, // No person-chunking implemented yet.
                &kernel_data_pool,
            );

            snps_processed += snps_in_chunk;
            eprintln!("> Processed chunk: {}/{} SNPs", snps_processed, prep_result.num_reconciled_snps);

            let empty_buffer = filled_buffer.release_for_reuse();
            if free_buffers_tx.send(empty_buffer).is_err() {
                break;
            }
        }

        if let Err(e) = producer_handle.join() {
            eprintln!("Fatal error: Producer thread panicked.");
            std::panic::resume_unwind(e);
        }
    });
    eprintln!("> Pipeline finished.");

    // --- Phase 5: Finalization and Output ---
    // The output filename is derived from the score file's name. This logic correctly
    // handles non-UTF8 paths by operating on OsString.
    let out_filename = match args.score.file_name() {
        Some(name) => {
            let mut os_string = OsString::from(name);
            os_string.push(".sscore");
            os_string
        }
        None => {
            eprintln!(
                "Error: Could not determine a base name from score file path '{}'.",
                args.score.display()
            );
            process::exit(1);
        }
    };
    let out_path = output_dir.join(&out_filename);
    eprintln!("> Writing {} scores per person to {}", prep_result.num_scores, out_path.display());

    if let Err(e) = write_scores_to_file(&out_path, &all_scores, prep_result.num_scores)
    {
        eprintln!("Error writing output file: {}", e);
        process::exit(1);
    }

    eprintln!(
        "\nSuccess! Total execution time: {:.2?}",
        start_time.elapsed()
    );
}

// ========================================================================================
//                                  HELPER FUNCTIONS
// ========================================================================================

/// Intelligently resolves the user-provided input path to a PLINK prefix.
fn resolve_plink_prefix(path: &Path) -> Result<PathBuf, String> {
    if !path.exists() {
        return Err("Input path does not exist.".to_string());
    }

    if path.is_dir() {
        let bed_files: Vec<PathBuf> = fs::read_dir(path)
            .map_err(|e| format!("Could not read directory: {}", e))?
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

/// Writes the final calculated scores to a tab-separated file.
///
/// This function minimizes I/O overhead. It formats an entire line into a reusable
/// in-memory buffer before writing it to the `BufWriter` in a single operation.
fn write_scores_to_file(
    path: &Path,
    scores: &[f32],
    num_scores: usize,
) -> io::Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Fix later
    let mut line_buffer = String::with_capacity(num_scores * 12);

    for person_scores in scores.chunks_exact(num_scores) {
        line_buffer.clear();

        // This pattern efficiently joins numbers into a string without a trailing
        // separator. `write!` on a String is fast and infallible.
        if let Some((last_score, head_scores)) = person_scores.split_last() {
            for score in head_scores {
                // The `unwrap` is safe because writing to a `String` cannot fail.
                write!(&mut line_buffer, "{}\t", score).unwrap();
            }
            write!(&mut line_buffer, "{}", last_score).unwrap();
        }

        // Write the fully-formed line to the buffer in a single operation.
        writeln!(writer, "{}", line_buffer)?;
    }
    Ok(())
}
