//! `gnomon all` unified driver.
//!
//! Runs the existing `score`, `project`, and `terms` pipelines against a
//! single VCF/BCF input while only parsing the VCF *once*. Under the hood
//! the VCF is converted to a PLINK fileset (via the score pipeline's own
//! `ensure_plink_format` cache) and that cached PLINK is then reused as the
//! genotype source for project and terms. Outputs are written at the paths
//! the standalone subcommands would have used (keyed off the original VCF
//! path), so downstream consumers like pgsEngine need no changes.
//!
//! Included from `cli/main.rs` via `#[path = "all.rs"] mod all_cmd;` so that
//! it shares the CLI's access to the `score_main` CLI-local module.
//!
//! Only compiled when all four of `map`, `score`, `calibrate`, and `terms`
//! are active (the same cfg as the full `gnomon` binary entry point).

use std::path::PathBuf;
use std::time::Instant;

use gnomon::map::io::derive_local_output_path;
use gnomon::map::main::run_project_with_output;
use gnomon::score::genotype_convert::{
    EnsurePlinkOptions, InputFormat, detect_input_format, ensure_plink_format_with_options,
};
use gnomon::terms::infer_sex_to_tsv_at;

/// Fully-expanded arguments for the `gnomon all` subcommand. Kept as a plain
/// struct (rather than a `clap::Args`) so the CLI layer owns parsing and the
/// `all` driver owns orchestration.
#[derive(Debug, Clone)]
pub struct AllOptions {
    /// `SCORE_PATH` positional: path to one score file, a directory of score
    /// files, or a comma-separated `PGS...` identifier list.
    pub score: PathBuf,
    /// `GENOTYPE_PATH` positional: VCF/BCF strongly preferred (that's where
    /// the single-scan win applies); PLINK also accepted.
    pub input_path: PathBuf,
    /// Built-in HWE-PCA model name used for projection
    /// (e.g. `hwe_1kg_hgdp_gsa_v3`).
    pub model: String,
    /// Optional PLINK `--keep` sample list forwarded to score.
    pub keep: Option<PathBuf>,
    /// Optional reference genome FASTA forwarded to score (DTC only).
    pub reference: Option<PathBuf>,
    /// Optional genome build override ("37" / "38" / "GRCh38" / ...).
    pub build: Option<String>,
    /// Optional strand-harmonization panel VCF forwarded to score.
    pub panel: Option<PathBuf>,
    /// Optional output manifest path for project's JSON summary.
    pub output_manifest: Option<PathBuf>,
}

/// Run the unified `gnomon all` pipeline.
///
/// Phase layout:
///   1. `ensure_plink_format` once. For VCF/BCF inputs this is the SINGLE
///      whole-file scan; for already-PLINK inputs it is a no-op.
///   2. `score` (reuses the cache transparently, so zero extra scans).
///   3. `project` against the cached PLINK, outputs written at the
///      VCF-keyed path.
///   4. `terms --sex` against the cached PLINK, sex.tsv at the VCF-keyed
///      path.
pub fn run(opts: AllOptions) -> Result<(), Box<dyn std::error::Error>> {
    let overall = Instant::now();
    let vcf_path = opts.input_path.clone();

    println!("=== gnomon all: unified score + project + terms ===");
    println!("Input genotype path: {}", vcf_path.display());
    println!("Score path: {}", opts.score.display());
    println!("Projection model: {}", opts.model);

    // --- Phase 1: ensure PLINK fileset (single VCF scan, cached for reuse) ---
    let format = detect_input_format(&vcf_path).ok_or_else(|| -> Box<dyn std::error::Error> {
        format!(
            "Could not determine input format for '{}' (expected PLINK/VCF/BCF/DTC).",
            vcf_path.display()
        )
        .into()
    })?;
    let input_is_vcf_like = matches!(format, InputFormat::Vcf | InputFormat::Bcf);

    let conv_start = Instant::now();
    let plink_prefix = ensure_plink_format_with_options(
        &vcf_path,
        opts.reference.as_deref(),
        opts.build.as_deref(),
        opts.panel.as_deref(),
        EnsurePlinkOptions {
            // Skip the in-conversion sex inference scan — phase 4's `terms`
            // pass computes per-sample sex directly from the cached PLINK
            // fileset, so the VCF-level pre-inference pass would be pure
            // duplicated work. (For non-VCF inputs the flag is a no-op.)
            skip_sex_inference: input_is_vcf_like,
            // We don't carry a caller-provided sex in `gnomon all`; phase 4
            // recomputes the real per-sample sex from the cached PLINK.
            inferred_sex: None,
        },
    )
    .map_err(|err| -> Box<dyn std::error::Error> { err })?;
    println!(
        "[all] ensure_plink_format: {:.2}s (prefix = {})",
        conv_start.elapsed().as_secs_f64(),
        plink_prefix.display()
    );

    // --- Phase 2: score ---
    // Score is passed the *original* input path so that its own output-
    // naming logic (derived from the VCF file stem) is unchanged. Internally
    // it calls ensure_plink_format again, which hits the cache produced in
    // phase 1 and short-circuits.
    let score_start = Instant::now();
    super::score_main::run_gnomon_with_args(
        vcf_path.clone(),
        opts.score.clone(),
        opts.keep.clone(),
        opts.reference.clone(),
        opts.build.clone(),
        opts.panel.clone(),
    )
    .map_err(|err| err as Box<dyn std::error::Error>)?;
    println!(
        "[all] score phase: {:.2}s",
        score_start.elapsed().as_secs_f64()
    );

    // --- Phase 3: project (against cached PLINK, outputs keyed to VCF) ---
    let project_start = Instant::now();
    let projection_scores_path = if input_is_vcf_like {
        derive_local_output_path(&vcf_path, "projection_scores.bin")
    } else {
        derive_local_output_path(&plink_prefix, "projection_scores.bin")
    };
    run_project_with_output(
        &plink_prefix,
        Some(&opts.model),
        opts.output_manifest.as_deref(),
        &projection_scores_path,
    )
    .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    println!(
        "[all] project phase: {:.2}s",
        project_start.elapsed().as_secs_f64()
    );

    // --- Phase 4: terms (sex inference against cached PLINK, sex.tsv keyed to VCF) ---
    let terms_start = Instant::now();
    let sex_tsv_path = if input_is_vcf_like {
        derive_local_output_path(&vcf_path, "sex.tsv")
    } else {
        derive_local_output_path(&plink_prefix, "sex.tsv")
    };
    let written = infer_sex_to_tsv_at(&plink_prefix, None, &sex_tsv_path)
        .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)?;
    println!(
        "[all] terms phase: {:.2}s (sex.tsv = {})",
        terms_start.elapsed().as_secs_f64(),
        written.display()
    );

    println!(
        "[all] total wall time: {:.2}s",
        overall.elapsed().as_secs_f64()
    );

    Ok(())
}
