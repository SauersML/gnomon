// ========================================================================================
//
//                               The strategic orchestrator: Gnomon
//
// ========================================================================================
//
// This module is the central nervous system and active conductor of the application.
// Its sole responsibility is to orchestrate the high-performance pipeline defined in
// the other modules. It owns all major resources and manages the application lifecycle
// from argument parsing to final output.

#![deny(dead_code)]
#![deny(unused_imports)]

use clap::{Parser, ValueEnum};
use gnomon::adapt_plink2::GenomeBuild;
use gnomon::score::download;
use gnomon::score::genotype_convert;
use gnomon::score::genotype_convert::{EnsurePlinkOptions, InputFormat, detect_input_format};
use gnomon::score::io::{gcs_billing_project_from_env, get_shared_runtime, load_adc_credentials};
use gnomon::score::native_vcf::{self, NativeVcfScoreResult};
use gnomon::score::pipeline::{self, MemoryBudget, PipelineContext};
use gnomon::score::prepare;
use gnomon::score::prepare::blocks::BlockPartition;
use gnomon::score::reformat;
use gnomon::score::types::{GenomicRegion, PreparationResult};
use natord::compare;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, UNIX_EPOCH};

// ========================================================================================
//                              Command-line interface definition
// ========================================================================================

/// Pre-computed sex from an upstream pipeline step. When supplied on the
/// command line via `--inferred-sex`, `gnomon score` skips its own full-VCF
/// sex-inference scan during the VCF→PLINK conversion (a ~4min pass on a
/// whole-genome imputed VCF) and writes the supplied value into the FAM file
/// directly. Intended for callers (e.g. pgsEngine's `gather` stage) that
/// already ran sex inference on a smaller upstream VCF and would otherwise
/// pay the cost twice.
#[derive(Copy, Clone, Debug, ValueEnum)]
#[clap(rename_all = "lower")]
pub enum InferredSexArg {
    Male,
    Female,
    Unknown,
}

#[derive(Parser, Debug)]
#[clap(
    name = "gnomon",
    version,
    about = "A high-performance engine for polygenic score calculation."
)]
struct Args {
    /// Path to a single score file or a directory containing multiple score files.
    #[clap(value_name = "SCORE_PATH")]
    score: PathBuf,

    /// Path to a file containing a list of individual IDs (IIDs) to include.
    /// If not provided, all individuals in the .fam file will be scored.
    #[clap(long)]
    keep: Option<PathBuf>,

    /// Path to genotype data (PLINK .bed/.bim/.fam prefix, VCF, BCF, or DTC text file)
    #[clap(value_name = "GENOTYPE_PATH")]
    input_path: PathBuf,

    /// Reference genome FASTA (optional; auto-downloaded if not provided for DTC files)
    #[clap(long)]
    reference: Option<PathBuf>,

    /// Genome build (37 or 38); required for PLINK 2 PGEN input
    #[clap(long)]
    build: Option<String>,

    /// Reference panel VCF for strand harmonization
    #[clap(long)]
    panel: Option<PathBuf>,

    /// Pre-computed sample sex (`male`, `female`, or `unknown`). When set,
    /// skips the VCF→PLINK step's internal `infer_first_sample_sex` scan
    /// and writes the supplied value into the FAM file. Use this to avoid
    /// a redundant ~4min whole-VCF sex scan when sex has already been
    /// inferred upstream (e.g. on a smaller pre-imputed VCF).
    #[clap(long, value_enum, value_name = "SEX")]
    inferred_sex: Option<InferredSexArg>,

    /// Emit sufficient statistics for aggregation across scored regions.
    #[clap(long)]
    emit_components: bool,

    /// Output prefix: write PREFIX.sscore, and keep score-file caches under
    /// PREFIX's directory, instead of beside the inputs.
    #[clap(long, value_name = "PREFIX")]
    out: Option<PathBuf>,

    /// Also emit per-block partial scores: `chrom` splits every score by
    /// chromosome; a BED file (0-based, half-open, non-overlapping rows) names
    /// the blocks. Columns `<SCORE>_b<ID>_AVG` and `_MISSING_PCT` join the
    /// unsplit ones, and `<OUTPUT>.blocks.tsv` maps block ids to intervals.
    #[clap(long, value_name = "chrom|BED")]
    blocks: Option<String>,

    /// The most blocks --blocks may name; required above 500.
    #[clap(long, value_name = "N", requires = "blocks")]
    blocks_max: Option<usize>,

    /// Write each weight of a score row that adds no variant to its score to PATH, with why and the
    /// alleles the genotypes hold at its position.
    #[clap(long, value_name = "PATH")]
    unmatched_report: Option<PathBuf>,
}

// ========================================================================================
//                              The main orchestration logic
// ========================================================================================

// Main function removed as it's now called through the main binary's subcommand system
// and was causing dead_code warnings.

/// Public interface for calling gnomon with explicit arguments, including a
/// pre-computed sex to skip the internal VCF-scan sex inference. Pass `None`
/// for `inferred_sex` to preserve the original full-scan behavior. `out` is
/// `--out PREFIX`; `None` writes beside the inputs as before. `blocks` and
/// `blocks_max` are `--blocks` and `--blocks-max`, and `unmatched_report` is
/// `--unmatched-report PATH`.
#[allow(clippy::too_many_arguments)]
pub fn run_gnomon_with_args(
    input_path: PathBuf,
    score: PathBuf,
    keep: Option<PathBuf>,
    reference: Option<PathBuf>,
    build: Option<String>,
    panel: Option<PathBuf>,
    inferred_sex: Option<InferredSexArg>,
    emit_components: bool,
    out: Option<PathBuf>,
    blocks: Option<String>,
    blocks_max: Option<usize>,
    unmatched_report: Option<PathBuf>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args {
        score,
        keep,
        input_path,
        reference,
        build,
        panel,
        inferred_sex,
        emit_components,
        out,
        blocks,
        blocks_max,
        unmatched_report,
    };
    run_gnomon_impl(args)
}

fn inline_pgs_output_suffix(score_arg: &str) -> String {
    let ids: Vec<&str> = score_arg
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| item.split('|').next().unwrap_or("").trim())
        .filter(|id| id.starts_with("PGS"))
        .collect();
    let count = ids.len();
    let hash = fnv1a64_hex8(score_arg.as_bytes());
    format!("pgs{count}_{hash}")
}

fn fnv1a64_hex8(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= u64::from(*b);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:08x}", (hash & 0xffff_ffff) as u32)
}

/// The primary application logic
// Function removed to eliminate dead code warnings

/// Core implementation that takes args as parameter
fn run_gnomon_impl(args: Args) -> Result<(), Box<dyn Error + Send + Sync>> {
    // Initialize the Rayon global thread pool to use all available cores.
    // Routed through the shared, idempotent helper so that the multi-phase
    // `gnomon all` driver (where the VCF→PLINK conversion may have already
    // brought up the global pool lazily) cannot abort on a racing
    // `build_global()`.
    gnomon::parallel::init_global_thread_pool();

    let overall_start_time = Instant::now();
    let score_arg_str = args.score.to_string_lossy().to_string();

    // --- Output Naming & Safety Check ---
    // For inline PGS lists, use a compact deterministic suffix to keep output names short.
    let out_suffix = if !args.score.exists() && score_arg_str.contains("PGS") {
        inline_pgs_output_suffix(&score_arg_str)
    } else {
        args.score
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "scores".to_string())
    };
    if let Some(prefix) = args.out.as_deref() {
        gnomon::output::validate_out_prefix(prefix)?;
    }
    if let Some(report) = args.unmatched_report.as_deref() {
        ensure_output_absent(report)?;
    }
    let blocks = match args.blocks.as_deref() {
        Some(arg) => Some(BlockPartition::parse_arg(arg).map_err(|e| format!("--blocks: {e}"))?),
        None => None,
    };
    if let Some(partition) = &blocks {
        eprintln!(
            "> Per-block partial scores over {}; block b0000 holds variants outside every block.",
            partition.describe()
        );
    }
    // Nothing behind the genotype path is not a format problem, so say that
    // before anything else runs.
    if !genotype_input_exists(&args.input_path) {
        return Err(format!(
            "no such file or PLINK/PGEN fileset: {}",
            args.input_path.display()
        )
        .into());
    }
    let cache_dir = score_cache_dir(args.out.as_deref(), &args.score);
    if let Some(dir) = cache_dir.as_deref() {
        fs::create_dir_all(dir).map_err(|e| {
            format!(
                "Could not create the score-file cache directory '{}': {e}",
                dir.display()
            )
        })?;
    }

    let input_format = detect_input_format(&args.input_path).ok_or_else(|| {
        format!(
            "Could not determine input format for '{}'. Expected PLINK (.bed/.bim/.fam), VCF (.vcf, .vcf.gz), BCF (.bcf), or DTC text (.txt).",
            args.input_path.display()
        )
    })?;

    let variant_file = matches!(input_format, InputFormat::Vcf | InputFormat::Bcf);
    let use_native_vcf = variant_file && args.panel.is_none();
    if variant_file && args.panel.is_some() {
        eprintln!(
            "> --panel supplied; using PLINK conversion path so panel harmonization is applied."
        );
    }

    if use_native_vcf {
        if blocks.is_some() {
            return Err(format!(
                "--blocks scores PLINK and PGEN filesets, and VCF/BCF through --panel; convert '{}' to PLINK first (plink2 --vcf ... --make-bed).",
                args.input_path.display()
            )
            .into());
        }
        let output_path = match args.out.as_deref() {
            Some(prefix) => gnomon::output::prefixed_path(prefix, "sscore"),
            None => score_output_path(&args.input_path, Some(&out_suffix)),
        };
        ensure_output_absent(&output_path)?;
        let (resolved_score_files, score_regions_map) = resolve_score_files(
            &args.score,
            &score_arg_str,
            args.out.as_deref().unwrap_or(&args.input_path),
        )?;

        if resolved_score_files.is_empty() {
            return Err("No score files were found or resolved.".into());
        }

        eprintln!("> Using native noodles streaming scorer for VCF/BCF input.");
        eprintln!(
            "> Normalizing and preparing {} score file(s)...",
            resolved_score_files.len()
        );
        let prep_start = Instant::now();
        let native_score_files = normalize_score_files(&resolved_score_files, cache_dir.as_deref())?;
        let native_result = native_vcf::score_vcf_streaming_reporting(
            &args.input_path,
            &native_score_files,
            args.keep.as_deref(),
            if score_regions_map.is_empty() {
                None
            } else {
                Some(&score_regions_map)
            },
            args.unmatched_report.as_deref(),
        )?;
        eprintln!(
            "> Native VCF scoring complete in {:.2?}. Found {} individuals to score and {} overlapping score variants across {} score(s).",
            prep_start.elapsed(),
            native_result.person_iids.len(),
            native_result.matched_variants,
            native_result.score_names.len()
        );

        let score_regions_ref = if score_regions_map.is_empty() {
            None
        } else {
            Some(&score_regions_map)
        };
        finalize_and_write_native_output(
            &output_path,
            &native_result,
            score_regions_ref,
            args.emit_components,
        )?;

        eprintln!(
            "\nSuccess! Total execution time: {:.2?}",
            overall_start_time.elapsed()
        );
        return Ok(());
    }

    // --- Genotype Format Conversion (if needed) ---
    // PLINK inputs continue directly; DTC inputs, and VCF/BCF inputs with --panel,
    // use the conversion path.
    let inferred_sex_override = args.inferred_sex.map(|s| match s {
        InferredSexArg::Male => genotype_convert::ConvertSex::Male,
        InferredSexArg::Female => genotype_convert::ConvertSex::Female,
        InferredSexArg::Unknown => genotype_convert::ConvertSex::Unknown,
    });
    // A converted input without --out keeps its conversion and its results in a
    // cache directory beside the input, so refuse before converting when that
    // location is not writable.
    if args.out.is_none()
        && let Some(naming_prefix) = genotype_convert::default_output_prefix(&args.input_path)
    {
        gnomon::output::ensure_output_writable(&fileset_output_path(
            &naming_prefix,
            Some(&out_suffix),
        ))?;
    }
    // Under --out the conversion cache joins the score-file caches under PREFIX's
    // directory, so nothing is written beside the genotypes.
    let effective_input_path = genotype_convert::ensure_plink_format_in(
        &args.input_path,
        args.reference.as_deref(),
        args.build.as_deref(),
        args.panel.as_deref(),
        EnsurePlinkOptions {
            skip_sex_inference: false,
            inferred_sex: inferred_sex_override,
        },
        args.out.as_deref().and(cache_dir.as_deref()),
    )?;
    let genome_build = args.build.as_deref().map(GenomeBuild::parse).transpose()?;

    let fileset_prefixes = resolve_filesets(&effective_input_path)?;
    // A converted input's default results and downloads keep their place in its cache
    // directory, outside the generation directories that later conversions replace.
    let naming_prefix = genotype_convert::default_output_prefix(&args.input_path)
        .unwrap_or_else(|| fileset_prefixes[0].clone());
    let output_path = match args.out.as_deref() {
        Some(prefix) => gnomon::output::prefixed_path(prefix, "sscore"),
        None => fileset_output_path(&naming_prefix, Some(&out_suffix)),
    };
    ensure_output_absent(&output_path)?;
    let sidecar_path = blocks
        .as_ref()
        .map(|_| blocks_sidecar_path(&output_path));
    if let Some(sidecar) = sidecar_path.as_deref() {
        ensure_output_absent(sidecar)?;
    }

    let (resolved_score_files, score_regions_map) = resolve_score_files(
        &args.score,
        &score_arg_str,
        args.out.as_deref().unwrap_or(&naming_prefix),
    )?;

    if resolved_score_files.is_empty() {
        return Err("No score files were found or resolved.".into());
    }

    let score_regions_ref = if score_regions_map.is_empty() {
        None
    } else {
        Some(&score_regions_map)
    };

    let prep_result = run_preparation_phase(
        &fileset_prefixes,
        &resolved_score_files,
        args.keep.as_deref(),
        score_regions_ref,
        cache_dir.as_deref(),
        blocks.as_ref(),
        args.blocks_max,
        args.unmatched_report.as_deref(),
    )?;
    let memory_budget = MemoryBudget::default();
    pipeline::preflight_memory(&prep_result, memory_budget)?;

    // --- Phase 2: Resource Allocation ---
    // A read-only context is created, which allocates all necessary memory pools
    // for the pipeline to use.
    let context =
        PipelineContext::with_budget(Arc::clone(&prep_result), memory_budget, genome_build);
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
    // final scores to the output path chosen before preparation.
    let block_columns = blocks
        .as_ref()
        .map(|partition| partition.block_column_flags(prep_result.score_names.len()));
    finalize_and_write_output(
        &output_path,
        &prep_result,
        &final_scores,
        &final_counts,
        score_regions_ref,
        args.emit_components,
        block_columns.as_deref(),
    )?;
    if let (Some(partition), Some(sidecar)) = (&blocks, sidecar_path.as_deref()) {
        gnomon::output::write_atomically(sidecar, |writer| partition.write_sidecar(writer))?;
        eprintln!("> Block intervals written to {}", sidecar.display());
    }

    eprintln!(
        "\nSuccess! Total execution time: {:.2?}",
        overall_start_time.elapsed()
    );
    Ok(())
}

/// The `--blocks` sidecar beside an `.sscore`: `<stem>.blocks.tsv`.
fn blocks_sidecar_path(output_path: &Path) -> PathBuf {
    output_path.with_extension("blocks.tsv")
}

fn ensure_output_absent(output_path: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    if output_path.exists() {
        return Err(format!(
            "Output file '{}' already exists. Gnomon will not overwrite it. Please remove it or rename it before running.",
            output_path.display()
        )
        .into());
    }
    // Refuse now, before preparation, when the results could not be saved: typically
    // a default location beside read-only inputs.
    gnomon::output::ensure_output_writable(output_path)?;
    Ok(())
}

/// Remote inputs have no local parent directory; their outputs and caches live
/// in the working directory.
/// Whether anything exists at a genotype path: a file or directory, a PLINK or
/// PGEN fileset with this prefix, or a remote location, which only its reader
/// can check.
fn genotype_input_exists(path: &Path) -> bool {
    if is_remote_prefix(path) || path.exists() {
        return true;
    }
    ["bed", "pgen"].iter().any(|extension| {
        let mut candidate = path.as_os_str().to_os_string();
        candidate.push(".");
        candidate.push(extension);
        Path::new(&candidate).exists()
    })
}

fn is_remote_prefix(path: &Path) -> bool {
    let raw = path.to_string_lossy();
    raw.starts_with("gs://") || raw.starts_with("http://") || raw.starts_with("https://")
}

fn score_output_path(output_prefix: &Path, name_suffix: Option<&str>) -> PathBuf {
    let (output_dir, mut out_stem) = if is_remote_prefix(output_prefix) {
        let stem = output_prefix
            .file_name()
            .map_or_else(|| OsString::from("gnomon_results"), OsString::from);
        (Path::new(".").to_path_buf(), stem)
    } else {
        let parent = output_prefix.parent();
        let dir = match parent {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => Path::new(".").to_path_buf(),
        };
        let stem = score_output_stem(output_prefix);
        (dir, stem)
    };

    if let Some(suffix) = name_suffix {
        out_stem.push("_");
        out_stem.push(suffix);
    }
    let mut out_filename = out_stem;
    out_filename.push(".sscore");
    output_dir.join(out_filename)
}

fn score_output_stem(path: &Path) -> OsString {
    const COMPOUND_SUFFIXES: &[&str] = &[".vcf.bgz", ".vcf.gz", ".bcf.bgz", ".bcf.gz"];
    const SIMPLE_SUFFIXES: &[&str] = &[".bed", ".vcf", ".bcf"];

    let Some(file_name) = path.file_name() else {
        return OsString::from("gnomon_results");
    };
    let file_name = file_name.to_string_lossy();
    let lower = file_name.to_ascii_lowercase();
    for suffix in COMPOUND_SUFFIXES.iter().chain(SIMPLE_SUFFIXES.iter()) {
        if lower.ends_with(suffix) && file_name.len() > suffix.len() {
            return OsString::from(&file_name[..file_name.len() - suffix.len()]);
        }
    }

    file_name.as_ref().into()
}

fn resolve_score_files(
    score_arg: &Path,
    score_arg_str: &str,
    cache_anchor: &Path,
) -> Result<(Vec<PathBuf>, HashMap<String, GenomicRegion>), Box<dyn Error + Send + Sync>> {
    if !score_arg.exists() && score_arg_str.contains("PGS") {
        let parent_local = is_remote_prefix(cache_anchor)
            .then(|| Path::new(".").to_path_buf())
            .unwrap_or_else(|| match cache_anchor.parent() {
                Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
                _ => Path::new(".").to_path_buf(),
            });
        let scores_cache_dir = parent_local.join("gnomon_score_cache");
        let resolved = download::resolve_and_download_scores(score_arg_str, &scores_cache_dir)?;
        return Ok((resolved.paths, resolved.regions));
    }

    let files: Vec<PathBuf> = if score_arg.is_dir() {
        let mut source_files: Vec<(PathBuf, String)> = Vec::new();
        let mut cache_files: Vec<(PathBuf, String, fs::Metadata)> = Vec::new();
        let mut source_metadata: std::collections::HashMap<String, fs::Metadata> =
            std::collections::HashMap::new();

        // Each entry costs a metadata round trip, so every entry is looked up at once.
        let entries: Vec<PathBuf> = fs::read_dir(score_arg)?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .collect();
        let described: Vec<(PathBuf, Option<fs::Metadata>)> = entries
            .into_par_iter()
            .map(|path| {
                let metadata = fs::metadata(&path).ok();
                (path, metadata)
            })
            .collect();

        for (path, metadata) in described {
            let Some(metadata) = metadata.filter(fs::Metadata::is_file) else {
                continue;
            };

            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            if name.ends_with(".gnomon.tsv") {
                if let Some(stem) = name.strip_suffix(".gnomon.tsv") {
                    cache_files.push((path, stem.to_string(), metadata));
                }
            } else if let Some(stem) = path.file_stem().map(|s| s.to_string_lossy().to_string()) {
                source_metadata.insert(stem.clone(), metadata);
                source_files.push((path, stem));
            }
        }

        let mut final_files = Vec::with_capacity(source_files.len() + cache_files.len());
        let mut covered_stems = std::collections::HashSet::new();

        // A previous run left each score file's sorted copy beside it as
        // `<stem>.sorted.gnomon.tsv`, which also ends in `.gnomon.tsv`. Only
        // `sorted_native_score_path` writes that name, so every such entry is
        // gnomon's own derivative, of a native `<stem>.tsv` as much as of a
        // `<stem>.gnomon.tsv` conversion, and the scorer derives it again from
        // the source it belongs to. Counting it as a score file made every repeat
        // run over a directory of native files fail with a duplicate score ID.
        cache_files.retain(|(_, stem, _)| !stem.ends_with(".sorted"));

        // A copy an older gnomon converted is converted again from its source. Each
        // stamp costs a round trip, so every copy is checked at once.
        let cache_files: Vec<(PathBuf, String, bool)> = cache_files
            .into_par_iter()
            .map(|(path, stem, cache_metadata)| {
                let keep_cache = match source_metadata.get(&stem) {
                    Some(source) => {
                        derived_copy_is_fresh(source, &cache_metadata)
                            && reformat::conversion_is_current(&path)
                    }
                    None => true,
                };
                (path, stem, keep_cache)
            })
            .collect();
        for (path, stem, keep_cache) in cache_files {
            if keep_cache {
                final_files.push(path);
                covered_stems.insert(stem);
            }
        }

        for (path, stem) in source_files {
            if !covered_stems.contains(&stem) {
                final_files.push(path);
            }
        }

        final_files
    } else {
        vec![score_arg.to_path_buf()]
    };

    Ok((files, HashMap::new()))
}

fn read_label_from_cached_file(path: &Path) -> Result<String, Box<dyn Error + Send + Sync>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 4 {
            return Ok(cols[3].to_string());
        }
        return Err(format!("Invalid header in cached file '{}'", path.display()).into());
    }
    Err(format!("Empty cached file '{}'", path.display()).into())
}

fn read_score_names_from_cached_file(
    path: &Path,
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() >= 4
            && cols[0] == "variant_id"
            && cols[1] == "effect_allele"
            && cols[2] == "other_allele"
        {
            return Ok(cols[3..].iter().map(|s| s.to_string()).collect());
        }
        return Err(format!("Invalid header in cached file '{}'", path.display()).into());
    }
    Err(format!("Empty cached file '{}'", path.display()).into())
}

/// Where normalized and sorted score-file caches go. `None` keeps each beside its
/// score file, as always. `--out` moves them under the prefix's directory, and a
/// score directory this process cannot write sends them to the user's cache
/// directory, so neither case writes beside the score files.
fn score_cache_dir(out: Option<&Path>, score_arg: &Path) -> Option<PathBuf> {
    if let Some(prefix) = out {
        let out_dir = prefix
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        return Some(out_dir.join("gnomon_score_cache"));
    }
    // PGS Catalog IDs download into a cache directory gnomon creates itself.
    if !score_arg.exists() {
        return None;
    }
    let score_dir = if score_arg.is_dir() {
        score_arg
    } else {
        score_arg
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
    };
    if directory_is_writable(score_dir) {
        return None;
    }
    let cache_dir = dirs::cache_dir()?.join("gnomon").join("score_cache");
    eprintln!(
        "> Score directory '{}' is not writable; caching converted score files under '{}'.",
        score_dir.display(),
        cache_dir.display()
    );
    Some(cache_dir)
}

/// Whether this process can create files in `dir`, found by creating one:
/// permission bits alone miss read-only mounts and ACLs.
fn directory_is_writable(dir: &Path) -> bool {
    let probe = dir.join(format!(".gnomon-write-probe.{}", std::process::id()));
    match OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(file) => {
            drop(file);
            let _ = fs::remove_file(&probe);
            true
        }
        Err(e) => e.kind() == io::ErrorKind::AlreadyExists,
    }
}

/// Names a cache entry for `source` inside a dedicated cache directory. Runs on
/// different score files share that directory, so the name carries a key over
/// this build and the source's identity (canonical path, size, modification
/// time): an edited, replaced or re-pointed source, or another gnomon build,
/// misses instead of reading an entry made from something else.
fn keyed_cache_path(cache_dir: &Path, source: &Path, suffix: &str) -> PathBuf {
    fn update_field(hasher: &mut Sha256, bytes: &[u8]) {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    }

    let mut hasher = Sha256::new();
    update_field(&mut hasher, env!("CARGO_PKG_VERSION").as_bytes());
    update_field(&mut hasher, env!("GNOMON_BUILD_TIMESTAMP").as_bytes());
    let canonical = fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    update_field(&mut hasher, canonical.as_os_str().as_encoded_bytes());
    if let Ok(metadata) = fs::metadata(source) {
        hasher.update(metadata.len().to_le_bytes());
        let modified_nanos = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |since| since.as_nanos());
        hasher.update(modified_nanos.to_le_bytes());
    }
    let key = hex::encode(&hasher.finalize()[..8]);
    let stem = source
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "score".into());
    cache_dir.join(format!("{stem}.{key}.{suffix}"))
}

fn sorted_native_score_path(path: &Path, cache_dir: Option<&Path>) -> PathBuf {
    if let Some(dir) = cache_dir {
        return keyed_cache_path(dir, path, "sorted.gnomon.tsv");
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy())
        .unwrap_or_else(|| "score".into());
    parent.join(format!("{stem}.sorted.gnomon.tsv"))
}

/// Whether a copy gnomon derived from `source`, a converted or a sorted score file,
/// was written after the source last changed.
///
/// On Unix that is the source's change time. The kernel sets it on every write and
/// no copy tool can restore it, so a file replaced under an older modification time,
/// as `cp -p` and `rsync -t` leave it, still reads as changed. Timestamps advance in
/// coarse ticks, so a copy stamped in the same tick as a change is not trusted.
fn derived_copy_is_fresh(source: &fs::Metadata, derived: &fs::Metadata) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        (derived.mtime(), derived.mtime_nsec()) > (source.ctime(), source.ctime_nsec())
    }
    #[cfg(not(unix))]
    {
        match (source.modified(), derived.modified()) {
            (Ok(changed), Ok(written)) => written > changed,
            _ => false,
        }
    }
}

/// Writes the sorted copy of each native score file that lacks a fresh one.
///
/// `freshly_converted` names the files this run just wrote with
/// `reformat_pgs_file`. Those can neither warn nor fail to parse, so each
/// maximal run of them is sorted in parallel; every other file is sorted on its
/// own, in order, so warnings and the first error come out as they always did
/// and no file after a failing one is touched.
fn sort_native_score_files(
    native_score_files: Vec<PathBuf>,
    freshly_converted: &HashSet<PathBuf>,
    cache_dir: Option<&Path>,
) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    let needs_sort =
        |path: &Path, sorted_path: &Path| match (fs::metadata(path), fs::metadata(sorted_path)) {
            (Ok(source), Ok(sorted)) => !derived_copy_is_fresh(&source, &sorted),
            _ => true,
        };
    let pairs: Vec<(PathBuf, PathBuf)> = native_score_files
        .into_iter()
        .map(|path| {
            let sorted_path = sorted_native_score_path(&path, cache_dir);
            (path, sorted_path)
        })
        .collect();

    // Deciding every file up front is only the same as deciding each just before
    // sorting it when no sort writes a path another file reads or writes.
    let mut touched = HashSet::with_capacity(2 * pairs.len());
    let independent = pairs.iter().all(|(_, sorted_path)| touched.insert(sorted_path))
        && pairs.iter().all(|(path, _)| !touched.contains(path));
    if !independent {
        for (path, sorted_path) in &pairs {
            if needs_sort(path, sorted_path) {
                reformat::sort_native_file(path, sorted_path)?;
            }
        }
        let mut sorted_files: Vec<PathBuf> =
            pairs.into_iter().map(|(_, sorted_path)| sorted_path).collect();
        sorted_files.sort();
        sorted_files.dedup();
        return Ok(sorted_files);
    }

    // Each decision is a few metadata round trips, made for every file at once.
    let plan: Vec<(PathBuf, PathBuf, bool)> = pairs
        .into_par_iter()
        .map(|(path, sorted_path)| {
            let should_sort = needs_sort(&path, &sorted_path);
            (path, sorted_path, should_sort)
        })
        .collect();
    let is_parallel = |(path, _, should_sort): &(PathBuf, PathBuf, bool)| {
        !should_sort || freshly_converted.contains(path)
    };

    let mut start = 0;
    while start < plan.len() {
        let run = plan[start..]
            .iter()
            .position(|item| !is_parallel(item))
            .unwrap_or(plan.len() - start);
        if run == 0 {
            let (path, sorted_path, should_sort) = &plan[start];
            if *should_sort {
                reformat::sort_native_file(path, sorted_path)?;
            }
            start += 1;
            continue;
        }
        let results: Vec<Result<(), reformat::ReformatError>> = plan[start..start + run]
            .par_iter()
            .map(|(path, sorted_path, should_sort)| {
                if *should_sort {
                    reformat::sort_native_file(path, sorted_path)
                } else {
                    Ok(())
                }
            })
            .collect();
        for result in results {
            result?;
        }
        start += run;
    }

    let mut sorted_files: Vec<PathBuf> = plan
        .into_iter()
        .map(|(_, sorted_path, _)| sorted_path)
        .collect();
    sorted_files.sort();
    sorted_files.dedup();
    Ok(sorted_files)
}

fn normalize_score_files(
    score_files: &[PathBuf],
    cache_dir: Option<&Path>,
) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    type BoxError = Box<dyn Error + Send + Sync>;
    enum Prep {
        Resolved(PathBuf),
        Pending(PathBuf, PathBuf),
    }
    enum Class {
        Native,
        Cached(PathBuf),
        Pending(PathBuf),
    }

    // Every file is classified at once: a header sniff and, for a file that needs
    // converting, a look at its cache. Each is a round trip on a network filesystem,
    // so a directory of many small score files waits on them in parallel. Messages
    // and the first error still come out in file order.
    let classified: Vec<Result<(Class, Result<(), BoxError>), String>> = score_files
        .par_iter()
        .map(
            |score_file_path| match reformat::is_gnomon_native_format(score_file_path) {
                Ok(true) => Ok((
                    Class::Native,
                    read_label_from_cached_file(score_file_path).map(drop),
                )),
                Ok(false) => {
                    let new_path = match cache_dir {
                        Some(dir) => keyed_cache_path(dir, score_file_path, "gnomon.tsv"),
                        None => score_file_path.with_extension("gnomon.tsv"),
                    };
                    let fresh = match (fs::metadata(score_file_path), fs::metadata(&new_path)) {
                        (Ok(source), Ok(cache)) => {
                            derived_copy_is_fresh(&source, &cache)
                                && reformat::conversion_is_current(&new_path)
                        }
                        _ => false,
                    };
                    if fresh {
                        let header = read_label_from_cached_file(&new_path).map(drop);
                        Ok((Class::Cached(new_path), header))
                    } else {
                        Ok((Class::Pending(new_path), Ok(())))
                    }
                }
                Err(e) => Err(format!(
                    "Error reading score file '{}': {}",
                    score_file_path.display(),
                    e
                )),
            },
        )
        .collect();

    let mut prep_items: Vec<(PathBuf, Prep)> = Vec::with_capacity(score_files.len());
    for (score_file_path, class) in score_files.iter().zip(classified) {
        let (class, header) = class?;
        let item = match class {
            Class::Native => Prep::Resolved(score_file_path.clone()),
            Class::Cached(new_path) => {
                eprintln!(
                    "> Info: Using cached converted file '{}'.",
                    new_path.display()
                );
                Prep::Resolved(new_path)
            }
            Class::Pending(new_path) => Prep::Pending(score_file_path.clone(), new_path),
        };
        header?;
        prep_items.push((score_file_path.clone(), item));
    }

    type ReformatRow = Option<(PathBuf, Option<reformat::SkipSummary>)>;
    let reformat_results: Vec<Result<ReformatRow, reformat::ReformatError>> = prep_items
        .par_iter()
        .map(|(_, item)| match item {
            Prep::Resolved(path) => Ok(Some((path.clone(), None))),
            Prep::Pending(src, dst) => {
                eprintln!(
                    "> Info: Score file '{}' is not in native format. Attempting conversion...",
                    src.display()
                );
                let outcome = reformat::reformat_pgs_file(src, dst)?;
                if outcome.wrote_output {
                    eprintln!("> Success: Converted to '{}'.", dst.display());
                    if outcome.score_label.is_none() {
                        return Err(reformat::ReformatError::Io(io::Error::other(
                            "Internal error: missing score label after successful conversion.",
                        )));
                    }
                    Ok(Some((dst.clone(), outcome.skip_summary)))
                } else {
                    if let Some(warning) = outcome.warning {
                        eprintln!("> Warning: {warning}");
                    } else {
                        eprintln!(
                            "> Warning: Unsupported score format for '{}'; skipping.",
                            src.display()
                        );
                    }
                    Ok(None)
                }
            }
        })
        .collect();

    // The native headers are read in parallel as well; duplicates are then checked
    // in file order.
    let names: Vec<Option<Result<Vec<String>, BoxError>>> = reformat_results
        .par_iter()
        .map(|row| match row {
            Ok(Some((out_path, _))) => Some(read_score_names_from_cached_file(out_path)),
            _ => None,
        })
        .collect();

    let mut native_score_files = Vec::with_capacity(prep_items.len());
    let mut label_to_path: HashMap<String, PathBuf> = HashMap::new();
    let mut skip_summaries = Vec::new();
    let mut freshly_converted = HashSet::new();
    for (((src_path, item), row), names) in prep_items.iter().zip(reformat_results).zip(names) {
        let (Some((out_path, skip_summary)), Some(names)) = (row.map_err(Box::new)?, names) else {
            continue;
        };
        for label in names? {
            if let Some(existing_path) = label_to_path.get(&label) {
                return Err(format!(
                    "Duplicate Score ID '{}' detected!\n  File 1: '{}'\n  File 2: '{}'\nPlease ensure each score column has a unique identifier.",
                    label,
                    existing_path.display(),
                    src_path.display()
                )
                .into());
            }
            label_to_path.insert(label, src_path.clone());
        }
        if let Some(summary) = skip_summary {
            skip_summaries.push(summary);
        }
        if matches!(item, Prep::Pending(..)) {
            freshly_converted.insert(out_path.clone());
        }
        native_score_files.push(out_path);
    }

    reformat::emit_overall_skip_summary(&skip_summaries);
    native_score_files.sort();
    native_score_files.dedup();
    if native_score_files.is_empty() {
        return Err("No compatible score files remained after normalization. Scores that only provide dosage-specific weights ('dosage_0_weight', 'dosage_1_weight', 'dosage_2_weight') are currently unsupported.".into());
    }

    sort_native_score_files(native_score_files, &freshly_converted, cache_dir)
}

/// **Helper 1:** Encapsulates the entire preparation and file normalization phase.
///
/// This function is synchronous and CPU-bound. It takes a definitive list of
/// resolved score files, normalizes them into a consistent, sorted, gnomon-native
/// format, and then calls the main preparation logic to produce a "computation
// blueprint" (`PreparationResult`). All user-facing console output for this phase
// is handled here.
#[allow(clippy::too_many_arguments)]
fn run_preparation_phase(
    fileset_prefixes: &[PathBuf],
    score_files: &[PathBuf],
    keep: Option<&Path>,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    cache_dir: Option<&Path>,
    blocks: Option<&BlockPartition>,
    blocks_max: Option<usize>,
    unmatched_report: Option<&Path>,
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
    let native_score_files = normalize_score_files(score_files, cache_dir)?;

    // --- Run the main preparation logic with the fully normalized and sorted files ---
    let prep = prepare::prepare_for_computation_with_blocks(
        fileset_prefixes,
        &native_score_files,
        keep,
        score_regions,
        blocks,
        blocks_max,
        unmatched_report,
    )
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

fn finalize_and_write_native_output(
    out_path: &Path,
    result: &NativeVcfScoreResult,
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    emit_components: bool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    eprintln!(
        "> Writing {} scores per person to {}",
        result.score_names.len(),
        out_path.display()
    );
    let output_start = Instant::now();

    write_scores_to_file(
        out_path,
        &result.person_iids,
        &result.score_names,
        &result.score_variant_counts,
        &NativeCells { result },
        &result.missing_counts,
        score_regions,
        emit_components,
        None,
    )?;

    eprintln!("> Final output written in {:.2?}", output_start.elapsed());
    Ok(())
}

/// Default `.sscore` path for PLINK filesets: `<prefix>_<suffix>.sscore` beside
/// the first fileset, or in the working directory when it is remote. Unlike
/// [`score_output_path`], the fileset name is used whole.
fn fileset_output_path(fileset_prefix: &Path, name_suffix: Option<&str>) -> PathBuf {
    let (output_dir, mut out_stem) = if is_remote_prefix(fileset_prefix) {
        let stem = fileset_prefix
            .file_name()
            .map_or_else(|| OsString::from("gnomon_results"), OsString::from);
        (Path::new(".").to_path_buf(), stem)
    } else {
        let parent = fileset_prefix.parent();
        // Handle both None and empty parent (Path::new("arrays").parent() == Some(""))
        let dir = match parent {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => Path::new(".").to_path_buf(),
        };
        let stem = fileset_prefix
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("gnomon_results"))
            .to_os_string();
        (dir, stem)
    };

    if let Some(suffix) = name_suffix {
        out_stem.push("_");
        out_stem.push(suffix);
    }
    let mut out_filename = out_stem;
    out_filename.push(".sscore");
    output_dir.join(out_filename)
}

/// **Helper:** Handles the final file writing.
///
/// This function is synchronous and takes the final results directly.
#[allow(clippy::too_many_arguments)]
fn finalize_and_write_output(
    out_path: &Path,
    prep_result: &Arc<PreparationResult>,
    final_scores: &[i64],
    final_counts: &[u32],
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    emit_components: bool,
    block_columns: Option<&[bool]>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    eprintln!(
        "> Writing {} scores per person to {}",
        prep_result.score_names.len(),
        out_path.display()
    );
    let output_start = Instant::now();

    write_scores_to_file(
        out_path,
        &prep_result.final_person_iids,
        &prep_result.score_names,
        &prep_result.score_variant_counts,
        &ExactCells {
            exact: prep_result.exact(),
            lanes: final_scores,
            num_scores: prep_result.score_names.len(),
        },
        final_counts,
        score_regions,
        emit_components,
        block_columns,
    )?;

    eprintln!("> Final output written in {:.2?}", output_start.elapsed());
    Ok(())
}

// ========================================================================================
//                                  Helper functions
// ========================================================================================

/// Discovers and validates all PLINK filesets from a given path.
///
/// Local (unchanged):
///   1) /path/to/prefix        -> prefix.{bed,bim,fam}
///   2) /path/to/prefix.bed    -> prefix.{bed,bim,fam}
///   3) /path/to/dir/          -> scan for *.bed in that dir, validate triads
///
/// Remote (new):
///   A) gs://bucket/prefix           -> prefix.{bed,bim,fam}
///   B) gs://bucket/prefix.bed       -> prefix.{bed,bim,fam}
///   C) gs://bucket/dir/             -> list all *.bed under that prefix
///   D) gs://bucket/dir/*            -> same as (C) (star is treated as “all under dir/”)
fn resolve_filesets(path: &Path) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    // --- GCS handling first ---
    if let Some(s) = path.to_str() {
        if is_gcs_uri_str(s) {
            return resolve_gcs_filesets(s);
        }
        // An HTTP fileset is named by one of its members; each member is
        // validated when it is opened, with the server's own error.
        if s.starts_with("http://") || s.starts_with("https://") {
            return Ok(vec![fileset_prefix(path)]);
        }
    }

    // --- ORIGINAL LOCAL LOGIC (unchanged) ---
    if !path.is_dir() {
        // Strip only a genuine fileset extension: prefixes such as
        // `acaf_threshold.chr22` carry dots of their own, and lopping off the
        // last dot segment would look for `acaf_threshold.bed`.
        let prefix = fileset_prefix(path);

        // APPEND the member extension; never `with_extension`, which REPLACES the
        // last dot segment. The comment above is right that `acaf_threshold.chr22`
        // carries a dot of its own -- and `with_extension("pgen")` on it yields
        // `acaf_threshold.pgen`, precisely the file the comment warns against
        // looking for. Every All of Us per-chromosome fileset has this shape, so
        // scoring one exits in under a second with "does not correspond to a
        // complete PLINK fileset" while all three members are sitting right there.
        let has_bed = ["bed", "bim", "fam"]
            .iter()
            .all(|ext| fileset_member(&prefix, ext).is_file());
        let has_pgen = ["pgen", "pvar", "psam"]
            .iter()
            .all(|ext| fileset_member(&prefix, ext).is_file());
        if !has_bed && !has_pgen {
            return Err(format!(
                "Input prefix '{}' does not correspond to a complete PLINK fileset \
                 (.bed/.bim/.fam or .pgen/.pvar/.psam).",
                prefix.display()
            )
            .into());
        }
        return Ok(vec![prefix]);
    }

    // A directory may hold either flavour of fileset (e.g. one .pgen per
    // chromosome, which is how All of Us ships its callsets). `.bed` wins when
    // both are present so existing layouts keep their current behaviour.
    let mut bed_files: Vec<PathBuf> = collect_fileset_members(path, "bed")?;
    let genotype_ext = if bed_files.is_empty() {
        bed_files = collect_fileset_members(path, "pgen")?;
        "pgen"
    } else {
        "bed"
    };

    if bed_files.is_empty() {
        return Err(format!(
            "No .bed or .pgen files found in directory '{}'.",
            path.display()
        )
        .into());
    }

    bed_files.sort_by(|a, b| compare(&a.to_string_lossy(), &b.to_string_lossy()));

    let mut prefixes = Vec::with_capacity(bed_files.len());
    let (meta_a, meta_b) = if genotype_ext == "pgen" {
        ("pvar", "psam")
    } else {
        ("bim", "fam")
    };
    for bed_path in bed_files {
        let prefix = fileset_prefix(&bed_path);
        if !fileset_member(&prefix, meta_a).is_file() || !fileset_member(&prefix, meta_b).is_file()
        {
            return Err(format!(
                "Incomplete fileset for prefix '{}'. Every .{genotype_ext} file in a directory must have corresponding .{meta_a} and .{meta_b} files.",
                prefix.display()
            )
            .into());
        }
        prefixes.push(prefix);
    }
    Ok(prefixes)
}

/// Drops a trailing fileset extension, leaving the prefix its members share.
/// `prefix` + `.ext`, by APPENDING. Never `Path::with_extension`, which REPLACES
/// the last dot segment: a per-chromosome fileset prefix such as
/// `acaf_threshold.chr22` would become `acaf_threshold.pgen` and the fileset
/// would be reported missing while all three members sit on disk. Every All of
/// Us callset has that shape.
fn fileset_member(prefix: &Path, ext: &str) -> PathBuf {
    let mut s = prefix.to_path_buf().into_os_string();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn fileset_prefix(path: &Path) -> PathBuf {
    match path.to_str() {
        Some(s) => PathBuf::from(gnomon::score::prepare::strip_fileset_extension(s)),
        None => path.to_path_buf(),
    }
}

/// Lists the genotype-table files of one flavour in a directory, skipping the
/// `.sorted` intermediates gnomon writes itself.
fn collect_fileset_members(
    dir: &Path,
    extension: &str,
) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    Ok(fs::read_dir(dir)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|p| {
            p.is_file()
                && p.extension().is_some_and(|ext| ext == extension)
                && !p
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .is_some_and(|stem| stem.ends_with(".sorted"))
        })
        .collect())
}

fn is_gcs_uri_str(s: &str) -> bool {
    s.starts_with("gs://")
}

fn split_gcs_uri_dir_and_leaf(raw: &str) -> Result<(String, String), Box<dyn Error + Send + Sync>> {
    let without = raw.trim_start_matches("gs://");
    let mut it = without.splitn(2, '/');
    let bucket = it.next().unwrap_or_default();
    let object = it.next().unwrap_or_default();
    if bucket.is_empty() {
        return Err(format!("Malformed GCS URI '{raw}': missing bucket").into());
    }
    Ok((bucket.to_string(), object.to_string()))
}

/// Robust GCS resolver that supports: exact triad prefix, *.bed in a "directory",
/// trailing slash, and star suffix.
fn resolve_gcs_filesets(uri: &str) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    use google_cloud_auth::credentials::{Credentials, anonymous::Builder as AnonymousCredentials};
    use google_cloud_storage::client::StorageControl;

    let wants_dir_scan = uri.ends_with("/*") || uri.ends_with('/');
    let normalized = if uri.ends_with("/*") {
        &uri[..uri.len() - 1]
    } else {
        uri
    };

    let (bucket, object) = split_gcs_uri_dir_and_leaf(normalized)?;

    let runtime = get_shared_runtime().map_err(|e| format!("{e}"))?;
    let user_project = gcs_billing_project_from_env();

    let make_control =
        |creds: Option<Credentials>| -> Result<StorageControl, Box<dyn Error + Send + Sync>> {
            let credentials = match creds {
                Some(existing) => existing,
                None => load_adc_credentials()
                    .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("{e}").into() })?,
            };

            runtime.block_on(async move {
                StorageControl::builder()
                    .with_credentials(credentials)
                    .build()
                    .await
                    .map_err(|e| -> Box<dyn Error + Send + Sync> {
                        format!("Failed to create Cloud Storage control client: {e}").into()
                    })
            })
        };
    let try_list_objects = |control: &StorageControl,
                            prefix: &str,
                            mut page_token: Option<String>|
     -> Result<
        (Vec<google_cloud_storage::model::Object>, Option<String>),
        Box<dyn Error + Send + Sync>,
    > {
        let mut req = control
            .list_objects()
            .set_parent(format!("projects/_/buckets/{bucket}"))
            .set_prefix(prefix.to_string());
        if let Some(tok) = page_token.take() {
            req = req.set_page_token(tok);
        }
        let resp = runtime
            .block_on(req.send())
            .map_err(|e| {
                let msg = e.to_string();
                if user_project.is_none() && msg.to_lowercase().contains("requester pays") {
                    format!("This is a Requester Pays bucket. Set GOOGLE_PROJECT (or `gcloud config set project ...`) and re-run. Original error while listing gs://{bucket}/{prefix}: {msg}")
                } else {
                    format!("Error listing gs://{bucket}/{prefix}: {msg}")
                }
            })?;
        let next = (!resp.next_page_token.is_empty()).then(|| resp.next_page_token.clone());
        Ok((resp.objects, next))
    };

    let try_head = |control: &StorageControl,
                    object_name: &str|
     -> Result<
        google_cloud_storage::model::Object,
        Box<dyn Error + Send + Sync>,
    > {
        let req = control
            .get_object()
            .set_bucket(format!("projects/_/buckets/{bucket}"))
            .set_object(object_name.to_string());
        runtime
            .block_on(req.send())
            .map_err(|e| {
                let msg = e.to_string();
                if user_project.is_none() && msg.to_lowercase().contains("requester pays") {
                    format!("This is a Requester Pays bucket. Set GOOGLE_PROJECT (or `gcloud config set project ...`) and re-run. Original error while fetching metadata for gs://{bucket}/{object_name}: {msg}").into()
                } else {
                    format!("Failed to fetch metadata for gs://{bucket}/{object_name}: {msg}").into()
                }
            })
    };

    let control = match make_control(None) {
        Ok(control) => control,
        Err(e) => {
            let e_msg = e.to_string();
            // Do NOT downgrade to anonymous credentials when a billing project is
            // configured. Anonymous credentials carry no quota project, so they cannot
            // satisfy a Requester Pays bucket under any circumstances -- the retry can
            // only fail, and it fails LATER, during a list or a metadata fetch, where the
            // error names the object and says nothing about credentials having been
            // silently swapped. That turns a clear authentication failure into an
            // apparent genotype problem. The fallback is still the right behaviour for a
            // public bucket, which is exactly the case where no project is set.
            if user_project.is_some() {
                return Err(Box::<dyn Error + Send + Sync>::from(format!(
                    "Unable to initialize Cloud Storage clients with ADC credentials, and \
                     a billing project is set (GOOGLE_PROJECT), so retrying anonymously \
                     cannot succeed against a Requester Pays bucket: {e_msg}"
                )));
            }
            let anonymous_creds = AnonymousCredentials::new().build();
            match make_control(Some(anonymous_creds)) {
                Ok(control) => control,
                Err(e2) => {
                    return Err(Box::<dyn Error + Send + Sync>::from(format!(
                        "Unable to initialize Cloud Storage clients: {e_msg} / {e2}"
                    )));
                }
            }
        }
    };

    if !wants_dir_scan && !object.ends_with('/') {
        let triad_prefix = match object.rsplit_once('.') {
            Some((base, "bed" | "pgen")) => base.to_string(),
            _ => object.clone(),
        };

        // Probe PLINK 1.9 first, then PLINK 2. A HEAD per member is cheap and
        // avoids a bucket listing, which requester-pays buckets charge for.
        let mut last_err = None;
        for triple in [["bed", "bim", "fam"], ["pgen", "pvar", "psam"]] {
            let mut ok = true;
            for ext in triple {
                let name = format!("{triad_prefix}.{ext}");
                if let Err(e) = try_head(&control, &name) {
                    last_err = Some(e);
                    ok = false;
                    break;
                }
            }
            if ok {
                return Ok(vec![PathBuf::from(format!("gs://{bucket}/{triad_prefix}"))]);
            }
        }
        return Err(last_err.unwrap_or_else(|| {
            format!(
                "gs://{bucket}/{triad_prefix} is not a complete PLINK fileset \
                 (.bed/.bim/.fam or .pgen/.pvar/.psam)"
            )
            .into()
        }));
    }

    let scan_prefix = if object.is_empty() || object.ends_with('/') {
        object.clone()
    } else {
        format!("{object}/")
    };

    let mut page_token: Option<String> = None;
    let mut objects: Vec<google_cloud_storage::model::Object> = Vec::new();
    loop {
        let (mut items, next) = try_list_objects(&control, &scan_prefix, page_token)?;
        objects.append(&mut items);
        if next.is_none() {
            break;
        }
        page_token = next;
    }

    if objects.is_empty() {
        return Err(format!("No objects found under gs://{bucket}/{scan_prefix}").into());
    }

    use std::collections::{HashMap, HashSet};
    let mut by_prefix: HashMap<String, HashSet<String>> = HashMap::new();
    for obj in objects.into_iter().filter(|o| !o.name.ends_with('/')) {
        let name = obj.name;
        if !name.starts_with(&scan_prefix) {
            continue;
        }
        if let Some((base, ext)) = name.rsplit_once('.') {
            if ["bed", "bim", "fam", "pgen", "pvar", "psam"].contains(&ext) {
                by_prefix
                    .entry(base.to_string())
                    .or_default()
                    .insert(ext.to_string());
            }
        }
    }

    let mut complete: Vec<String> = by_prefix
        .into_iter()
        .filter_map(|(base, exts)| {
            let has = |t: [&str; 3]| t.iter().all(|e| exts.contains(*e));
            (has(["bed", "bim", "fam"]) || has(["pgen", "pvar", "psam"])).then_some(base)
        })
        .collect();

    if complete.is_empty() {
        return Err(format!(
            "No complete PLINK filesets (.bed/.bim/.fam or .pgen/.pvar/.psam) under gs://{bucket}/{scan_prefix}"
        )
        .into());
    }

    complete.sort_by(|a, b| compare(a, b));

    Ok(complete
        .into_iter()
        .map(|base| PathBuf::from(format!("gs://{bucket}/{base}")))
        .collect())
}

/// Writes the final calculated scores to a self-describing, tab-separated file.
/// This function now calculates the final per-variant average and missing percentage.
/// `block_columns[i]` marks a `--blocks` partial column, whose `_MISSING_PCT`
/// is 100 when its block holds no variant of the score; an unsplit column
/// without variants reports 0.
#[allow(clippy::too_many_arguments)]
fn write_scores_to_file<V: ScoreValues>(
    path: &Path,
    person_iids: &[String],
    score_names: &[String],
    score_variant_counts: &[u32],
    values: &V,
    missing_counts: &[u32],
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    emit_components: bool,
    block_columns: Option<&[bool]>,
) -> io::Result<()> {
    let num_scores = score_names.len();

    gnomon::output::write_atomically(path, |writer| {
        write_sscore_header(
            writer,
            score_names,
            score_variant_counts,
            score_regions,
            emit_components,
        )?;
        write_score_rows(
            writer,
            person_iids,
            score_variant_counts,
            values,
            missing_counts,
            num_scores,
            emit_components,
            score_rows_per_block(person_iids, num_scores),
            block_columns,
        )
    })
}

/// The `#SCORE_VARIANT_COUNT` and `#REGION` metadata rows, then the `#IID` header.
fn write_sscore_header<W: Write>(
    writer: &mut W,
    score_names: &[String],
    score_variant_counts: &[u32],
    score_regions: Option<&HashMap<String, GenomicRegion>>,
    emit_components: bool,
) -> io::Result<()> {
    if emit_components {
        writeln!(writer, "#SCORE_VARIANT_COUNT\tSCORE\tCOUNT")?;
        for (name, count) in score_names.iter().zip(score_variant_counts) {
            writeln!(writer, "#SCORE_VARIANT_COUNT\t{name}\t{count}")?;
        }
    }

    if let Some(regions) = score_regions {
        let mut wrote_metadata_header = false;
        for name in score_names {
            if let Some(region) = regions.get(name) {
                if !wrote_metadata_header {
                    writeln!(writer, "#REGION\tSCORE\tINTERVAL")?;
                    wrote_metadata_header = true;
                }
                writeln!(writer, "#REGION\t{name}\t{region}")?;
            }
        }
    }

    // Write the new, more descriptive, and correctly tab-separated header.
    write!(writer, "#IID")?;
    for name in score_names {
        if emit_components {
            write!(writer, "\t{name}_SUM\t{name}_MISSING_CT")?;
        } else {
            write!(writer, "\t{name}_AVG\t{name}_MISSING_PCT")?;
        }
    }
    writeln!(writer)
}

/// The finished values of `.sscore` cells, laid out person × score.
trait ScoreValues: Sync {
    /// How many cells there are.
    fn cells(&self) -> usize;
    /// A cell's sum.
    fn sum(&self, cell: usize) -> f64;
    /// A cell's average over the variants it used; 0 when it used none.
    fn average(&self, cell: usize, variants_used: u32) -> f64;
}

/// A compiled plan's exact integer cells: every value is the exact rational sum or average,
/// rounded once.
struct ExactCells<'a> {
    exact: &'a gnomon::score::cells::ExactPlan,
    lanes: &'a [i64],
    num_scores: usize,
}

impl ExactCells<'_> {
    #[inline]
    fn score_and_lanes(&self, cell: usize) -> (usize, &[i64]) {
        let stride = self.exact.stride();
        let person = cell / self.num_scores;
        (
            cell % self.num_scores,
            &self.lanes[person * stride..(person + 1) * stride],
        )
    }
}

impl ScoreValues for ExactCells<'_> {
    fn cells(&self) -> usize {
        self.lanes.len() / self.exact.stride() * self.num_scores
    }

    fn sum(&self, cell: usize) -> f64 {
        let (score, lanes) = self.score_and_lanes(cell);
        self.exact.sum(score, lanes)
    }

    fn average(&self, cell: usize, variants_used: u32) -> f64 {
        let (score, lanes) = self.score_and_lanes(cell);
        self.exact.average(score, lanes, variants_used)
    }
}

/// The native VCF scorer's exact cells, each rounded once.
struct NativeCells<'a> {
    result: &'a NativeVcfScoreResult,
}

impl ScoreValues for NativeCells<'_> {
    fn cells(&self) -> usize {
        self.result.missing_counts.len()
    }

    fn sum(&self, cell: usize) -> f64 {
        self.result.sum(cell)
    }

    fn average(&self, cell: usize, variants_used: u32) -> f64 {
        self.result.average(cell, variants_used)
    }
}

/// f64 sums averaged in f64: the finished values the output tests write.
#[cfg(test)]
struct F64Sums<'a> {
    sums: &'a [f64],
}

#[cfg(test)]
impl ScoreValues for F64Sums<'_> {
    fn cells(&self) -> usize {
        self.sums.len()
    }

    fn sum(&self, cell: usize) -> f64 {
        self.sums[cell]
    }

    fn average(&self, cell: usize, variants_used: u32) -> f64 {
        // Based on the number of non-missing variants, as standard tools compute it when
        // mean imputation is disabled.
        if variants_used > 0 {
            self.sums[cell] / (variants_used as f64)
        } else {
            0.0
        }
    }
}

/// Text formatted per block of `.sscore` rows. At most one block per worker
/// thread is held at once, so the rows cost a few MiB beyond the scores
/// themselves whatever the cohort size.
const SCORE_ROW_BLOCK_BYTES: usize = 1 << 20;

/// Rows per formatting block, from the width of a typical row: the first IID,
/// then per score a tab, a shortest-round-trip f64 of at most 24 characters,
/// a tab and a missing column.
fn score_rows_per_block(person_iids: &[String], num_scores: usize) -> usize {
    let row_bytes = person_iids.first().map_or(16, String::len) + 1 + num_scores * 36;
    (SCORE_ROW_BLOCK_BYTES / row_bytes).max(1)
}

/// Writes one `.sscore` row per person from whole-cohort sums. Blocks of
/// `rows_per_block` rows are formatted in parallel and written in order, so the
/// bytes are exactly those of formatting the rows one at a time.
#[allow(clippy::too_many_arguments)]
fn write_score_rows<W: Write, V: ScoreValues>(
    writer: &mut W,
    person_iids: &[String],
    score_variant_counts: &[u32],
    values: &V,
    missing_counts: &[u32],
    num_scores: usize,
    emit_components: bool,
    rows_per_block: usize,
    block_columns: Option<&[bool]>,
) -> io::Result<()> {
    let n_persons = person_iids.len();
    let needed = n_persons.saturating_mul(num_scores);
    if values.cells() < needed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Mismatched number of persons and score rows during final write.",
        ));
    }
    if missing_counts.len() < needed {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Mismatched number of persons and missing count rows during final write.",
        ));
    }
    if block_columns.is_some_and(|flags| flags.len() != num_scores) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Block column flags do not cover every score column during final write.",
        ));
    }

    let rows_per_block = rows_per_block.max(1);
    // The formatted text of one block per worker thread.
    let mut blocks = vec![Vec::new(); rayon::current_num_threads().max(1)];
    let batch_rows = rows_per_block.saturating_mul(blocks.len());
    let mut batch_start = 0;
    while batch_start < n_persons {
        let batch_end = batch_start.saturating_add(batch_rows).min(n_persons);
        blocks.par_iter_mut().enumerate().for_each(|(block, text)| {
            text.clear();
            let start = batch_start + block * rows_per_block;
            let end = start.saturating_add(rows_per_block).min(batch_end);
            if start < end {
                format_score_rows(
                    text,
                    person_iids,
                    start..end,
                    score_variant_counts,
                    values,
                    missing_counts,
                    num_scores,
                    emit_components,
                    block_columns,
                );
            }
        });
        for text in blocks.iter() {
            writer.write_all(text)?;
        }
        batch_start = batch_end;
    }
    Ok(())
}

/// Appends the `.sscore` rows of `persons` to `text`.
#[allow(clippy::too_many_arguments)]
fn format_score_rows<V: ScoreValues>(
    text: &mut Vec<u8>,
    person_iids: &[String],
    persons: std::ops::Range<usize>,
    score_variant_counts: &[u32],
    values: &V,
    missing_counts: &[u32],
    num_scores: usize,
    emit_components: bool,
    block_columns: Option<&[bool]>,
) {
    let mut ryu_buffer_score = ryu::Buffer::new();
    let mut ryu_buffer_missing = ryu::Buffer::new();
    for person in persons {
        text.extend_from_slice(person_iids[person].as_bytes());
        let row = person * num_scores;
        for i in 0..num_scores {
            let missing_count = missing_counts[row + i];
            let total_variants_for_score = score_variant_counts[i];

            text.push(b'\t');
            if emit_components {
                text.extend_from_slice(ryu_buffer_score.format(values.sum(row + i)).as_bytes());
                text.push(b'\t');
                write!(text, "{missing_count}").unwrap();
            } else {
                let variants_used = total_variants_for_score.saturating_sub(missing_count);
                let avg_score = values.average(row + i, variants_used);
                let missing_pct = if total_variants_for_score > 0 {
                    (missing_count as f32 / total_variants_for_score as f32) * 100.0
                } else if block_columns.is_some_and(|flags| flags[i]) {
                    // A block without any variant of the score: nothing is known.
                    100.0
                } else {
                    0.0
                };
                text.extend_from_slice(ryu_buffer_score.format(avg_score).as_bytes());
                text.push(b'\t');
                text.extend_from_slice(ryu_buffer_missing.format(missing_pct).as_bytes());
            }
        }
        text.push(b'\n');
    }
}

#[cfg(test)]
mod output_tests {
    use super::{F64Sums, write_score_rows, write_scores_to_file};
    use std::fs;

    /// A directory scored once holds each file's `<stem>.sorted.gnomon.tsv`
    /// beside it. Scoring it again must see the same score files as the first
    /// run: the sorted copies are derivatives, whether their source is a native
    /// `.tsv` or an earlier `.gnomon.tsv` conversion, and a fresh conversion
    /// still stands in for its source.
    #[test]
    fn repeat_directory_discovery_skips_sorted_copies() {
        let dir = tempfile::tempdir().unwrap();
        let header = "variant_id\teffect_allele\tother_allele\tSCORE\n";
        for name in [
            "a.tsv",
            "a.sorted.gnomon.tsv",
            "b.gnomon.tsv",
            "b.sorted.gnomon.tsv",
            "c.txt",
        ] {
            fs::write(dir.path().join(name), header).unwrap();
        }
        let (files, regions) =
            super::resolve_score_files(dir.path(), &dir.path().to_string_lossy(), dir.path())
                .unwrap();
        let mut names: Vec<String> = files
            .iter()
            .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["a.tsv", "b.gnomon.tsv", "c.txt"]);
        assert!(regions.is_empty());
    }

    #[test]
    fn directory_discovery_converts_an_older_conversion_again() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("c.txt"), "#pgs_id=C\n").unwrap();
        let cache = dir.path().join("c.gnomon.tsv");
        let discover = || {
            // Written well after its source, so only the stamp can make it stale.
            fs::File::options()
                .write(true)
                .open(&cache)
                .unwrap()
                .set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(60))
                .unwrap();
            let (files, _) =
                super::resolve_score_files(dir.path(), &dir.path().to_string_lossy(), dir.path())
                    .unwrap();
            files
                .iter()
                .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
        };
        let rows = "variant_id\teffect_allele\tother_allele\tC\n1:100\tA\tG\t0.5\n";
        fs::write(&cache, rows).unwrap();
        assert_eq!(discover(), ["c.txt"]);
        fs::write(
            &cache,
            format!("{}\n{rows}", super::reformat::CONVERSION_STAMP),
        )
        .unwrap();
        assert_eq!(discover(), ["c.gnomon.tsv"]);
    }

    /// The one-row-at-a-time writer that `write_score_rows` replaced, kept verbatim as
    /// the byte-identity reference.
    fn serial_rows(
        person_iids: &[String],
        score_variant_counts: &[u32],
        sum_scores: &[f64],
        missing_counts: &[u32],
        num_scores: usize,
        emit_components: bool,
    ) -> Vec<u8> {
        use std::fmt::Write as _;
        use std::io::Write as _;

        let mut writer = Vec::new();
        let mut line_buffer = String::new();
        let mut sum_score_chunks = sum_scores.chunks_exact(num_scores);
        let mut missing_count_chunks = missing_counts.chunks_exact(num_scores);
        let mut ryu_buffer_score = ryu::Buffer::new();
        let mut ryu_buffer_missing = ryu::Buffer::new();

        for iid in person_iids {
            let person_sum_scores = sum_score_chunks.next().unwrap();
            let person_missing_counts = missing_count_chunks.next().unwrap();

            line_buffer.clear();
            write!(&mut line_buffer, "{iid}").unwrap();

            for i in 0..num_scores {
                let final_sum_score = person_sum_scores[i];
                let missing_count = person_missing_counts[i];
                let total_variants_for_score = score_variant_counts[i];
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
                if emit_components {
                    write!(
                        &mut line_buffer,
                        "\t{}\t{missing_count}",
                        ryu_buffer_score.format(final_sum_score)
                    )
                    .unwrap();
                } else {
                    write!(
                        &mut line_buffer,
                        "\t{}\t{}",
                        ryu_buffer_score.format(avg_score),
                        ryu_buffer_missing.format(missing_pct)
                    )
                    .unwrap();
                }
            }
            writeln!(writer, "{line_buffer}").unwrap();
        }
        writer
    }

    /// Persons with short and long IIDs; scores with zero, few and many variants;
    /// sums that include NaN, infinities, signed zeros, a subnormal and extremes.
    fn cohort(persons: usize, num_scores: usize) -> (Vec<String>, Vec<u32>, Vec<f64>, Vec<u32>) {
        let iids = (0..persons)
            .map(|person| {
                if person % 5 == 0 {
                    format!("p{person}")
                } else {
                    format!("person-with-a-longer-identifier-{person}")
                }
            })
            .collect();
        let counts = (0..num_scores).map(|score| [0, 7, 1_000_003][score % 3]).collect();
        let specials = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            5e-324,
            1e300,
            -2.5e-8,
        ];
        let sums = (0..persons * num_scores)
            .map(|slot| {
                if slot % 11 == 0 {
                    specials[(slot / 11) % specials.len()]
                } else {
                    (slot as f64).sin() * 1e-3
                }
            })
            .collect();
        let missing = (0..persons * num_scores).map(|slot| (slot % 9) as u32).collect();
        (iids, counts, sums, missing)
    }

    #[test]
    fn score_rows_match_the_serial_writer_byte_for_byte() {
        let (iids, counts, sums, missing) = cohort(1_001, 3);
        for emit_components in [false, true] {
            let expected = serial_rows(&iids, &counts, &sums, &missing, 3, emit_components);
            for rows_per_block in [1, 2, 7, 1_000, 5_000] {
                let mut written = Vec::new();
                write_score_rows(
                    &mut written,
                    &iids,
                    &counts,
                    &F64Sums { sums: &sums },
                    &missing,
                    3,
                    emit_components,
                    rows_per_block,
                    None,
                )
                .expect("rows should be written");
                assert!(
                    written == expected,
                    "emit_components={emit_components} rows_per_block={rows_per_block}"
                );
            }
        }
    }

    #[test]
    fn score_rows_refuse_fewer_values_than_persons() {
        let (iids, counts, sums, missing) = cohort(10, 2);
        for (sums, missing) in [(&sums[..19], &missing[..]), (&sums[..], &missing[..19])] {
            let error = write_score_rows(
                &mut Vec::new(),
                &iids,
                &counts,
                &F64Sums { sums },
                missing,
                2,
                false,
                4,
                None,
            )
            .expect_err("too few values must be refused");
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        }
    }

    /// Scores that stream in person blocks are written by calling write_score_rows once
    /// per block with that block's slices; the concatenation must be the serial bytes.
    #[test]
    fn rows_written_block_by_block_match_the_serial_writer() {
        let (iids, counts, sums, missing) = cohort(1_001, 3);
        for emit_components in [false, true] {
            let expected = serial_rows(&iids, &counts, &sums, &missing, 3, emit_components);
            for block in [1, 7, 333, 4_096] {
                let mut written = Vec::new();
                let mut first = 0;
                while first < iids.len() {
                    let n = block.min(iids.len() - first);
                    write_score_rows(
                        &mut written,
                        &iids[first..first + n],
                        &counts,
                        &F64Sums {
                            sums: &sums[first * 3..(first + n) * 3],
                        },
                        &missing[first * 3..(first + n) * 3],
                        3,
                        emit_components,
                        5,
                        None,
                    )
                    .expect("block rows should be written");
                    first += n;
                }
                assert!(
                    written == expected,
                    "emit_components={emit_components} block={block}"
                );
            }
        }
    }

    #[test]
    fn component_output_contains_raw_sum_and_exact_counts() {
        let path = std::env::temp_dir().join(format!(
            "gnomon-components-{}-{}.sscore",
            std::process::id(),
            std::thread::current().name().unwrap_or("unnamed")
        ));

        write_scores_to_file(
            &path,
            &["person-1".to_string()],
            &["PGS000001".to_string()],
            &[4],
            &F64Sums { sums: &[6.0] },
            &[1],
            None,
            true,
            None,
        )
        .expect("component output should be written");

        let output = fs::read_to_string(&path).expect("component output should be readable");
        fs::remove_file(&path).expect("component output should be removable");
        assert_eq!(
            output,
            "#SCORE_VARIANT_COUNT\tSCORE\tCOUNT\n\
#SCORE_VARIANT_COUNT\tPGS000001\t4\n\
#IID\tPGS000001_SUM\tPGS000001_MISSING_CT\n\
person-1\t6.0\t1\n"
        );
    }

    /// An unsplit score without variants reports 0% missing as before; a block
    /// column without variants reports 100%, and `_SUM`/`_MISSING_CT` are unchanged.
    #[test]
    fn an_empty_block_column_is_fully_missing() {
        let dir = tempfile::tempdir().expect("temporary directory");
        let iids = ["p1".to_string()];
        let names = ["S".to_string(), "S_b0000".to_string(), "S_b0001".to_string()];
        let flags = [false, true, true];
        let averaged = dir.path().join("averaged.sscore");
        write_scores_to_file(
            &averaged,
            &iids,
            &names,
            &[0, 0, 2],
            &F64Sums {
                sums: &[0.0, 0.0, 3.0],
            },
            &[0, 0, 1],
            None,
            false,
            Some(&flags),
        )
        .expect("averaged output should be written");
        assert_eq!(
            fs::read_to_string(&averaged).unwrap(),
            "#IID\tS_AVG\tS_MISSING_PCT\tS_b0000_AVG\tS_b0000_MISSING_PCT\tS_b0001_AVG\tS_b0001_MISSING_PCT\n\
p1\t0.0\t0.0\t0.0\t100.0\t3.0\t50.0\n"
        );
        let components = dir.path().join("components.sscore");
        write_scores_to_file(
            &components,
            &iids,
            &names,
            &[0, 0, 2],
            &F64Sums {
                sums: &[0.0, 0.0, 3.0],
            },
            &[0, 0, 1],
            None,
            true,
            Some(&flags),
        )
        .expect("component output should be written");
        assert!(
            fs::read_to_string(&components)
                .unwrap()
                .ends_with("p1\t0.0\t0\t0.0\t0\t3.0\t1\n")
        );
    }
}
