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
    let prep_result = match prepare::prepare_for_computation(&plink_prefix, &args.score) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("Fatal error during data preparation: {}", e);
            process::exit(1);
        }
    };

    // Derive counts from the length of the metadata vectors. This is the single
    // source of truth for the dimensions of the problem.
    let num_people = prep_result.person__ids.len();
    let num_scores = prep_result.score_names.len();

    eprintln!(
        "> Preparation complete. Found {} individuals and {} overlapping SNPs.",
        num_people, prep_result.num_reconciled_snps
    );

    // --- Phase 3: Resource Allocation ---
    let mut all_scores = vec![0.0f32; num_people * num_scores];
    let kernel_data_pool = KernelDataPool::new();
    let pivot_buffer_capacity = num_people * PIPELINE_CHUNK_SIZE_SNPS;
    let pipeline_buffers: Vec<_> = (0..PIPELINE_BUFFER_COUNT)
        .map(|_| PipelineBuffer::new(pivot_buffer_capacity))
        .collect();

    // --- Phase 4: Pipeline Construction and Execution ---
    eprintln!("> Starting pipeline with {} buffers...", PIPELINE_BUFFER_COUNT);
    let (free_buffers_tx, free_buffers_rx) = mpsc::sync_channel(PIPELINE_BUFFER_COUNT);
    let (filled_buffers_tx, filled_buffers_rx) = mpsc::sync_channel(PIPELINE_BUFFER_COUNT);

    thread::scope(|s| {
        // --- 4a: Spawn the Producer Thread ---
        let bed_path = plink_prefix.with_extension("bed");
        let producer_handle = s.spawn(move || {
            let bed_reader = MmapBedReader::new(
                &bed_path,
                num_people,
                prep_result.total_snps_in_bim,
                prep_result.required_snp_indices,
                prep_result.reconciliation_instructions,
            )
            .expect("Failed to create .bed file reader");

            batch::producer_task(
                bed_reader,
                free_buffers_rx,
                filled_buffers_tx,
                PIPELINE_CHUNK_SIZE_SNPS,
                num_people,
            );
        });

        // --- 4b: Execute the Consumer on the Main Thread ---
        for buffer in pipeline_buffers {
            let _ = free_buffers_tx.send(buffer);
        }

        let mut snps_processed = 0;
        while let Ok(filled_buffer) = filled_buffers_rx.recv() {
            let snps_in_chunk = filled_buffer.snps_in_chunk();
            if snps_in_chunk == 0 { break; } // Producer signals EOF with an empty buffer.

            let pivoted_data = filled_buffer.as_pivoted_data(num_people);
            let weights_chunk = &prep_result.interleaved_weights[snps_processed * num_scores..];

            batch::process_pivoted_chunk(
                pivoted_data,
                weights_chunk,
                &mut all_scores,
                num_people,
                snps_in_chunk,
                num_scores,
                0, // No person-chunking implemented yet.
                &kernel_data_pool,
            );

            snps_processed += snps_in_chunk;
            eprintln!(
                "> Processed chunk: {}/{} SNPs",
                snps_processed, prep_result.num_reconciled_snps
            );

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
    let out_filename = match args.score.file_name() {
        Some(name) => {
            let mut s = OsString::from(name);
            s.push(".sscore");
            s
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
    eprintln!(
        "> Writing {} scores per person to {}",
        num_scores,
        out_path.display()
    );

    if let Err(e) = write_scores_to_file(&out_path, &prep_result.person_ids, &prep_result.score_names, &all_scores) {
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

/// Writes the final calculated scores to a self-describing, tab-separated file.
///
/// The output format includes a header line with the Individual ID (IID) and all
/// score names. This makes the file human-readable and easily machine-parsable.
/// This function is optimized to minimize I/O overhead by formatting each line
/// into a reusable buffer before performing a single write call per line.
fn write_scores_to_file(
    path: &Path,
    person_ids: &[String],
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
    // A reusable buffer avoids repeated memory allocations in this hot loop.
    let mut line_buffer = String::new();
    let mut score_chunks = scores.chunks_exact(num_scores);

    // Iterate through the person IDs, pulling the corresponding chunk of scores.
    for iid in person_ids {
        let person_scores = score_chunks.next().ok_or_else(|| {
            // This error indicates a catastrophic logic bug upstream.
            io::Error::new(io::ErrorKind::InvalidData, "Mismatched number of persons and score rows during final write.")
        })?;

        line_buffer.clear();
        // The `unwrap` is safe because writing to a `String` via `fmt::Write` cannot fail.
        write!(&mut line_buffer, "{}", iid).unwrap();

        for score in person_scores {
            write!(&mut line_buffer, "\t{}", score).unwrap();
        }

        writeln!(writer, "{}", line_buffer)?;
    }

    Ok(())
}
