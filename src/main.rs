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
use gnomon::pipeline::{self, PipelineContext};
use gnomon::prepare;
use gnomon::reformat;
use gnomon::types::PreparationResult;
use ryu;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Write as FmtWrite;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

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

fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    // Initialize the Rayon global thread pool to use all available cores.
    // This is critical for the performance of the data-parallel compute pipeline.
    rayon::ThreadPoolBuilder::new()
        .build_global()
        .unwrap();

    let overall_start_time = Instant::now();
    let args = Args::parse();
    let plink_prefix = resolve_plink_prefix(&args.input_path)?;

    // --- Phase 1: Preparation ---
    // This phase is synchronous and CPU-bound. It parses all input files,
    // reconciles variants, and produces a "computation blueprint".
    let prep_result = run_preparation_phase(&plink_prefix, &args)?;

    // --- Phase 2: Resource Allocation ---
    // A read-only context is created, which allocates all necessary memory pools
    // for the pipeline to use.
    let context = PipelineContext::new(Arc::clone(&prep_result));
    eprintln!("> Resource allocation complete.");

    // --- Phase 3: Pipeline Execution ---
    // This is the primary compute phase. It is a synchronous, blocking call that
    // returns the final, aggregated results upon completion.
    let computation_start = Instant::now();
    let (final_scores, final_counts) = pipeline::run(&context, &plink_prefix)?;
    eprintln!("> Computation finished. Total pipeline time: {:.2?}", computation_start.elapsed());

    // --- Phase 4: Finalization & Output ---
    // After all computation is complete, this synchronous phase writes the
    // final scores to disk, passing the results directly.
    finalize_and_write_output(&plink_prefix, &prep_result, &final_scores, &final_counts)?;
    
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
) -> Result<Arc<PreparationResult>, Box<dyn Error + Send + Sync>> {
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
    eprintln!("> Starting core preparation (parsing, reconciling, compiling)...");
    let actual_prep_start = Instant::now();
    let prep =
        prepare::prepare_for_computation(plink_prefix, &native_score_files, args.keep.as_deref())
            .map_err(|e| Box::new(e) as Box<dyn Error + Send + Sync>)?;
    eprintln!("> TIMING: Core preparation logic took {:.2?}", actual_prep_start.elapsed());

    eprintln!(
        "> Preparation complete in {:.2?}. Found {} individuals to score and {} overlapping variants across {} score(s).",
        prep_phase_start.elapsed(),
        prep.num_people_to_score,
        prep.num_reconciled_variants,
        prep.score_names.len()
    );

    Ok(Arc::new(prep))
}

/// **Helper:** Handles the final file writing.
///
/// This function is synchronous. It now takes the final results directly,
/// making it a more focused and decoupled utility.
fn finalize_and_write_output(
    plink_prefix: &Path,
    prep_result: &Arc<PreparationResult>,
    final_scores: &[f64],
    final_counts: &[u32],
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
        prep_result.score_names.len(),
        out_path.display()
    );
    let output_start = Instant::now();

    // Delegate to the file writer, passing all necessary data.
    write_scores_to_file(
        &out_path,
        &prep_result.final_person_iids,
        &prep_result.score_names,
        &prep_result.score_variant_counts,
        final_scores,
        final_counts,
    )?;

    eprintln!("> Final output written in {:.2?}", output_start.elapsed());
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
