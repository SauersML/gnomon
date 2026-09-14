//! `gnomon all` unified driver.
//!
//! Runs `score`, `project`, and `terms --sex` against one genotype input. Every
//! phase calls its subcommand's own entry point on the original input, so each
//! output is the file that subcommand writes, at the path it uses by default or
//! under `--out PREFIX`. Nothing is converted first: an input a phase cannot read
//! fails that phase, as it fails the subcommand, while the other phases still run
//! and write their outputs.
//!
//! Included from `cli/main.rs` via `#[path = "all.rs"] mod all_cmd;` so that
//! it shares the CLI's argument types and subcommand entry points.
//!
//! Only compiled when all four of `map`, `score`, `calibrate`, and `terms`
//! are active (the same cfg as the full `gnomon` binary entry point).

use std::path::{Path, PathBuf};
use std::thread::{self, ScopedJoinHandle};
use std::time::{Duration, Instant};

use gnomon::adapt_plink2::GenomeBuild;
use gnomon::map::main::run_project_with_output;
use gnomon::map::prefit;
use gnomon::output::{prefixed_path, validate_out_prefix};

/// Fully-expanded arguments for the `gnomon all` subcommand. Kept as a plain
/// struct (rather than a `clap::Args`) so the CLI layer owns parsing and the
/// `all` driver owns orchestration.
#[derive(Debug, Clone)]
pub struct AllOptions {
    /// `SCORE_PATH` positional: path to one score file, a directory of score
    /// files, or a comma-separated `PGS...` identifier list.
    pub score: PathBuf,
    /// `GENOTYPE_PATH` positional, given unchanged to every phase.
    pub input_path: PathBuf,
    /// Built-in HWE-PCA model name used for projection (e.g.
    /// `hwe_1kg_hgdp_gsa_v3`). Without one, projection reads the
    /// `<GENOTYPE>.hwe.json` beside the genotype data, as `gnomon project` does.
    pub model: Option<String>,
    /// Optional PLINK `--keep` sample list forwarded to score.
    pub keep: Option<PathBuf>,
    /// Optional reference genome FASTA forwarded to score (DTC only).
    pub reference: Option<PathBuf>,
    /// Optional genome build override ("37" / "38" / "GRCh38" / ...), forwarded
    /// to every phase.
    pub build: Option<String>,
    /// Optional strand-harmonization panel VCF forwarded to score.
    pub panel: Option<PathBuf>,
    /// Optional output manifest path for project's JSON summary.
    pub output_manifest: Option<PathBuf>,
    /// Optional output prefix shared by every phase: `<PREFIX>.sscore`,
    /// `<PREFIX>.projection_scores.bin` and `<PREFIX>.sex.tsv`.
    pub out: Option<PathBuf>,
}

/// How one phase ended, and how long it ran.
struct PhaseOutcome {
    name: &'static str,
    elapsed: Duration,
    result: Result<(), String>,
}

/// Runs a phase to completion and records how it ended. The error becomes text
/// here because a phase on its own thread must hand back something `Send`.
fn run_phase(
    name: &'static str,
    phase: impl FnOnce() -> Result<(), Box<dyn std::error::Error>>,
) -> PhaseOutcome {
    let start = Instant::now();
    let result = phase().map_err(|err| err.to_string());
    PhaseOutcome {
        name,
        elapsed: start.elapsed(),
        result,
    }
}

/// Run the unified `gnomon all` pipeline.
///
/// `score`, `project` and `terms --sex` each run on the original input through
/// the subcommand's own entry point, concurrently when [`phase_overlap`] allows
/// it and in that order otherwise. Every phase runs to completion even when
/// another fails; the status of each is printed at the end, and the run fails
/// naming every phase that failed.
pub fn run(opts: AllOptions) -> Result<(), Box<dyn std::error::Error>> {
    let overall = Instant::now();

    println!("=== gnomon all: unified score + project + terms ===");
    println!("Input genotype path: {}", opts.input_path.display());
    println!("Score path: {}", opts.score.display());
    match opts.model.as_deref() {
        Some(model) => println!("Projection model: {model}"),
        None => println!("Projection model: the .hwe.json beside the genotype data"),
    }

    // Bring up Rayon's global thread pool ONCE, up front, before any phase
    // touches a parallel primitive, so that no phase's explicit `build_global()`
    // races another phase's lazy initialization (the historical `gnomon all`
    // SIGABRT). This idempotent helper makes every later init a no-op.
    gnomon::parallel::init_global_thread_pool();

    // A prefix that cannot name local files fails every phase the same way, so
    // reject it before any of them starts.
    if let Some(prefix) = opts.out.as_deref() {
        validate_out_prefix(prefix)?;
    }

    // --- score: `gnomon score SCORE GENOTYPE [--out PREFIX]` ---
    let score_phase = || {
        run_phase("score", || {
            super::run_score(super::ScoreArgs {
                score: opts.score.clone(),
                keep: opts.keep.clone(),
                input_path: opts.input_path.clone(),
                out: opts.out.clone(),
                reference: opts.reference.clone(),
                build: opts.build.clone(),
                panel: opts.panel.clone(),
                // `gnomon all` carries no caller-provided sex; the terms phase
                // infers it.
                inferred_sex: None,
                emit_components: false,
            })
        })
    };

    // --- project: `gnomon project GENOTYPE`, which has no --out of its own ---
    let project_phase = || {
        run_phase("project", || match opts.out.as_deref() {
            None => super::run_map_project(super::ProjectArgs {
                genotype_path: opts.input_path.clone(),
                build: opts.build.clone(),
                model: opts.model.clone(),
                output_manifest: opts.output_manifest.clone(),
            }),
            // Under a prefix the scores take the name score and terms give their
            // outputs.
            Some(prefix) => {
                let genome_build = opts.build.as_deref().map(GenomeBuild::parse).transpose()?;
                run_project_with_output(
                    &opts.input_path,
                    genome_build,
                    opts.model.as_deref(),
                    opts.output_manifest.as_deref(),
                    &prefixed_path(prefix, "projection_scores.bin"),
                )
                .map_err(|err| Box::new(err) as Box<dyn std::error::Error>)
            }
        })
    };

    // --- terms: `gnomon terms GENOTYPE --sex [--out PREFIX]` ---
    let terms_phase = || {
        run_phase("terms", || {
            super::run_terms(super::TermsArgs {
                genotype_path: opts.input_path.clone(),
                build: opts.build.clone(),
                sex: true,
                out: opts.out.clone(),
            })
        })
    };

    let outcomes = match phase_overlap(&opts.input_path, opts.model.as_deref()) {
        Ok(threads) => {
            println!("[all] running score, project and terms concurrently on {threads} threads");
            thread::scope(|scope| {
                let project = scope.spawn(&project_phase);
                let terms = scope.spawn(&terms_phase);
                let score = score_phase();
                [score, join_phase(project), join_phase(terms)]
            })
        }
        Err(reason) => {
            println!("[all] running score, project and terms in order: {reason}");
            [score_phase(), project_phase(), terms_phase()]
        }
    };

    let mut failures = Vec::new();
    for outcome in &outcomes {
        let seconds = outcome.elapsed.as_secs_f64();
        match &outcome.result {
            Ok(()) => println!("[all] {} phase: {seconds:.2}s", outcome.name),
            Err(error) => {
                println!(
                    "[all] {} phase: FAILED after {seconds:.2}s: {error}",
                    outcome.name
                );
                failures.push(format!("{}: {error}", outcome.name));
            }
        }
    }
    println!(
        "[all] total wall time: {:.2}s",
        overall.elapsed().as_secs_f64()
    );

    if failures.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} of 3 gnomon all phases failed ({})",
            failures.len(),
            failures.join("; ")
        )
        .into())
    }
}

/// Whether score, project and terms may run at the same time: the number of
/// threads they would share, or why they run in order.
///
/// None of the three fills the pool by itself on small inputs (score file
/// preparation and model loading are mostly serial, sex inference accumulates
/// on one thread), so overlapping them saves their fixed costs. On a large
/// fileset projection uses every core and overlapping gains nothing, so the
/// question is memory. Measured on local PLINK filesets from 1 to 51,200
/// samples, a run in order peaked within 3% of the fileset plus the projection
/// model on disk, and overlapping peaked at 1.44 times that sum at most, so the
/// phases overlap only when available memory covers twice the sum. An input
/// without that measured bound (VCF, BCF, PGEN, DTC text, a remote fileset) or a
/// model that is not on disk yet runs in order.
fn phase_overlap(genotypes: &Path, model: Option<&str>) -> Result<usize, String> {
    let threads = rayon::current_num_threads();
    if threads < 2 {
        return Err("one thread".to_string());
    }
    let prefix = match genotypes.extension().and_then(|ext| ext.to_str()) {
        Some("bed" | "bim" | "fam") => genotypes.with_extension(""),
        _ => genotypes.to_path_buf(),
    };
    let genotype_bytes = std::fs::metadata(prefixed_path(&prefix, "bed"))
        .map_err(|_| format!("{} is not a local PLINK fileset", genotypes.display()))?
        .len();
    let model_json = match model {
        Some(name) => {
            let info =
                prefit::lookup_model(name).ok_or_else(|| format!("unknown model '{name}'"))?;
            prefit::cached_model_path(info).map_err(|err| err.to_string())?
        }
        None => prefixed_path(&prefix, "hwe.json"),
    };
    let model_bytes = std::fs::metadata(&model_json)
        .map_err(|_| format!("{} is not on disk yet", model_json.display()))?
        .len()
        + std::fs::metadata(model_json.with_extension("project.bin")).map_or(0, |meta| meta.len());
    let needed = genotype_bytes.saturating_add(model_bytes).saturating_mul(2);
    let (_, available) = gnomon::memory::memory_bytes();
    if available < needed {
        return Err(format!(
            "{} available, {} needed to overlap them",
            gib(available),
            gib(needed)
        ));
    }
    Ok(threads)
}

fn gib(bytes: u64) -> String {
    format!("{:.1} GiB", bytes as f64 / f64::from(1u32 << 30))
}

/// Wait for a phase thread, re-raising its panic here so that a phase which
/// crashes fails `gnomon all` the same way it fails its own subcommand.
fn join_phase(handle: ScopedJoinHandle<'_, PhaseOutcome>) -> PhaseOutcome {
    handle
        .join()
        .unwrap_or_else(|payload| std::panic::resume_unwind(payload))
}
