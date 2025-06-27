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
use gnomon::download;
use natord::compare;
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
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    let overall_start_time = Instant::now();
    let args = Args::parse();
    let fileset_prefixes = resolve_filesets(&args.input_path)?;

    // --- Phase 1a: Score File Resolution ---
    // This block resolves the --score argument into a definitive list of files.
    let resolved_score_files: Vec<PathBuf> = {
        let score_arg_str = args.score.to_string_lossy();

        // HEURISTIC: If the path doesn't exist AND it contains "PGS",
        // assume it's a list of IDs to download. This avoids accidentally
        // triggering a download for a simple typo in a file path.
        if !args.score.exists() && score_arg_str.contains("PGS") {
            // Define the permanent cache directory. Placing it relative to the
            // output is a good strategy, keeping all generated files together.
            let output_parent_dir = fileset_prefixes[0].parent().unwrap_or_else(|| Path::new("."));
            let scores_cache_dir = output_parent_dir.join("gnomon_score_cache");

            download::resolve_and_download_scores(&score_arg_str, &scores_cache_dir)?
        } else {
            // --- This is the original, local file handling logic ---
            // The path exists locally, so we handle it as a file or directory.
            if args.score.is_dir() {
                fs::read_dir(&args.score)?
                    .filter_map(Result::ok)
                    .map(|entry| entry.path())
                    .filter(|p| p.is_file())
                    .collect()
            } else {
                // It's a single file.
                vec![args.score.clone()]
            }
        }
    };

    if resolved_score_files.is_empty() {
        return Err("No score files were found or resolved.".into());
    }

    // --- Phase 1b: Preparation ---
    // This phase is synchronous and CPU-bound. It parses all input files,
    // reconciles variants, and produces a "computation blueprint".
    let prep_result =
        run_preparation_phase(&fileset_prefixes, &resolved_score_files, args.keep.as_deref())?;

    // --- Phase 2: Resource Allocation ---
    // A read-only context is created, which allocates all necessary memory pools
    // for the pipeline to use.
    let context = PipelineContext::new(Arc::clone(&prep_result));
    eprintln!("> Resource allocation complete.");

    // --- Phase 3: Pipeline Execution ---
    // This is the primary compute phase. It is a synchronous, blocking call that
    // returns the final, aggregated results upon completion.
    let computation_start = Instant::now();
    let (final_scores, final_counts) = pipeline::run(&context)?;
    eprintln!(
        "> Computation finished. Total pipeline time: {:.2?}",
        computation_start.elapsed()
    );

    // --- Phase 4: Finalization & Output ---
    // After all computation is complete, this synchronous phase writes the
    // final scores to disk. The first fileset's prefix determines the output name.
    finalize_and_write_output(
        &fileset_prefixes[0],
        &prep_result,
        &final_scores,
        &final_counts,
    )?;

    eprintln!(
        "\nSuccess! Total execution time: {:.2?}",
        overall_start_time.elapsed()
    );
    Ok(())
}

/// **Helper 1:** Encapsulates the entire preparation and file normalization phase.
///
/// This function is synchronous and CPU-bound. It takes a definitive list of
/// resolved score files, normalizes them, and then calls the main preparation
/// logic to produce a "computation blueprint" (`PreparationResult`). All user-facing
/// console output for this phase is handled here.
fn run_preparation_phase(
    fileset_prefixes: &[PathBuf],
    score_files: &[PathBuf],
    keep: Option<&Path>,
) -> Result<Arc<PreparationResult>, Box<dyn Error + Send + Sync>> {
    if fileset_prefixes.len() > 1 {
        eprintln!(
            "> Found {} PLINK filesets, starting with: {}",
            fileset_prefixes.len(),
            fileset_prefixes[0].display()
        );
    } else {
        eprintln!("> Using PLINK prefix: {}", fileset_prefixes[0].display());
    }

    eprintln!(
        "> Normalizing and preparing {} score file(s)...",
        score_files.len()
    );
    let prep_phase_start = Instant::now();

    let mut native_score_files = Vec::with_capacity(score_files.len());
    for score_file_path in score_files {
        match reformat::is_gnomon_native_format(score_file_path) {
            Ok(true) => {
                native_score_files.push(score_file_path.clone());
            }
            Ok(false) => {
                // Determine the output path for the converted file. It should be
                // placed in the same directory as the original.
                let output_dir = score_file_path.parent().unwrap_or_else(|| Path::new("."));
                fs::create_dir_all(output_dir)?;

                eprintln!(
                    "> Info: Score file '{}' is not in native format. Attempting conversion...",
                    score_file_path.display()
                );
                let new_path = score_file_path.with_extension("gnomon.tsv");
                match reformat::reformat_pgs_file(score_file_path, &new_path) {
                    Ok(_) => {
                        eprintln!("> Success: Converted to '{}'.", new_path.display());
                        native_score_files.push(new_path);
                    }
                    Err(e) => {
                        return Err(format!(
                            "Failed to auto-reformat '{}': {}. Please ensure it is a valid PGS Catalog file or convert it to the gnomon-native format manually.",
                            score_file_path.display(), e
                        )
                        .into());
                    }
                }
            }
            Err(e) => {
                return Err(format!(
                    "Error reading score file '{}': {}",
                    score_file_path.display(),
                    e
                )
                .into());
            }
        }
    }

    // --- Run the main preparation logic ---
    let prep = prepare::prepare_for_computation(fileset_prefixes, &native_score_files, keep)
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

/// **Helper:** Handles the final file writing.
///
/// This function is synchronous and takes the final results directly.
fn finalize_and_write_output(
    output_prefix: &Path,
    prep_result: &Arc<PreparationResult>,
    final_scores: &[f64],
    final_counts: &[u32],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let output_dir = output_prefix.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;

    // Construct a self-describing output filename based on the primary input prefix.
    let out_filename = {
        let mut s = output_prefix
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

/// Discovers and validates all PLINK filesets from a given path.
///
/// Handles three cases:
/// 1. Path is a prefix (`/path/to/data` -> `data.bed`, `data.bim`, `data.fam`).
/// 2. Path is a single `.bed` file (`/path/to/data.bed`).
/// 3. Path is a directory containing one or more complete filesets.
fn resolve_filesets(path: &Path) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    if !path.is_dir() {
        // --- SINGLE-FILESET PATH: Treat as a prefix, validate, and wrap in a Vec ---
        // This handles both --input-path my_file.bed and --input-path my_file
        let prefix = if path.extension().is_some() {
            path.with_extension("")
        } else {
            path.to_path_buf()
        };

        if !prefix.with_extension("bed").is_file()
            || !prefix.with_extension("bim").is_file()
            || !prefix.with_extension("fam").is_file()
        {
            return Err(format!(
                "Input prefix '{}' does not correspond to a complete PLINK fileset (.bed, .bim, .fam).",
                prefix.display()
            )
            .into());
        }
        return Ok(vec![prefix]);
    }

    // --- MULTI-FILESET PATH: Scan, sort, and validate directory ---
    let mut bed_files: Vec<PathBuf> = fs::read_dir(path)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| p.is_file() && p.extension().map_or(false, |ext| ext == "bed"))
        .collect();

    if bed_files.is_empty() {
        return Err(format!("No .bed files found in directory '{}'.", path.display()).into());
    }

    // Natural sort ensures chr2 comes before chr10. Critical for correctness.
    bed_files.sort_by(|a, b| compare(&a.to_string_lossy(), &b.to_string_lossy()));

    let mut prefixes = Vec::with_capacity(bed_files.len());
    for bed_path in bed_files {
        let prefix = bed_path.with_extension("");
        if !prefix.with_extension("bim").is_file() || !prefix.with_extension("fam").is_file() {
            return Err(format!(
                "Incomplete fileset for prefix '{}'. All .bed files in a directory must have corresponding .bim and .fam files.",
                prefix.display()
            )
            .into());
        }
        prefixes.push(prefix);
    }
    Ok(prefixes)
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
