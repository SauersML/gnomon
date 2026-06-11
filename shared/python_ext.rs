//! In-process Python bindings for the gnomon core.
//!
//! This module exposes the same entry points the `gnomon` CLI dispatches to
//! (`score`, `project`, `terms`, `model`) as a native `#[pymodule]`, plus one
//! standalone helper (`infer_sex`) that the subprocess wrapper cannot offer
//! without running a full CLI invocation and re-parsing the written TSV. PyO3
//! loads the compiled `cdylib` directly into the host Python process, so
//! `import gnomon` runs the Rust engine in-process: there is no subprocess to
//! the `gnomon` binary and no `install.sh` step.
//!
//! Every function here is a thin adapter over an existing `pub` function in the
//! gnomon library (`crate::score_main`, `crate::map`, `crate::terms`); no
//! pipeline logic is reimplemented, and the Python arguments map onto exactly
//! the parameters the matching CLI subcommand passes. Heavy work runs with the
//! GIL released (`py.detach`) so other Python threads keep running, and Rust
//! errors surface as Python exceptions.

use std::path::PathBuf;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::map::main as map_cli;
use crate::score_main;
use crate::terms::{infer_first_sample_sex, infer_sex_to_tsv};

/// Translate a caller-supplied sex string into the same `InferredSexArg` the
/// CLI's `--inferred-sex` flag accepts. `None` preserves the full-VCF scan.
fn parse_inferred_sex(value: Option<&str>) -> PyResult<Option<score_main::InferredSexArg>> {
    match value {
        None => Ok(None),
        Some(raw) => match raw.to_ascii_lowercase().as_str() {
            "male" => Ok(Some(score_main::InferredSexArg::Male)),
            "female" => Ok(Some(score_main::InferredSexArg::Female)),
            "unknown" => Ok(Some(score_main::InferredSexArg::Unknown)),
            other => Err(PyValueError::new_err(format!(
                "inferred_sex must be one of 'male', 'female', 'unknown' (got '{other}')"
            ))),
        },
    }
}

/// Compute raw polygenic scores in-process.
///
/// Mirrors `gnomon score <SCORE_PATH> <GENOTYPE_PATH> [flags]`. Returns the
/// resolved `input_path` the scoring engine wrote its outputs alongside, as a
/// string, so callers can locate the produced `.sscore` files.
#[pyfunction]
#[pyo3(signature = (
    score,
    input_path,
    keep = None,
    reference = None,
    build = None,
    panel = None,
    inferred_sex = None,
))]
#[allow(clippy::too_many_arguments)]
fn score(
    py: Python<'_>,
    score: String,
    input_path: String,
    keep: Option<String>,
    reference: Option<String>,
    build: Option<String>,
    panel: Option<String>,
    inferred_sex: Option<String>,
) -> PyResult<String> {
    let input_path_buf = PathBuf::from(&input_path);
    let inferred = parse_inferred_sex(inferred_sex.as_deref())?;
    let score_buf = PathBuf::from(score);
    let keep_buf = keep.map(PathBuf::from);
    let reference_buf = reference.map(PathBuf::from);
    let panel_buf = panel.map(PathBuf::from);

    // Heavy work: release the GIL so other Python threads can run.
    py.detach(|| {
        score_main::run_gnomon_with_args(
            input_path_buf,
            score_buf,
            keep_buf,
            reference_buf,
            build,
            panel_buf,
            inferred,
        )
    })
    .map_err(|err| PyRuntimeError::new_err(format!("gnomon score failed: {err}")))?;

    Ok(input_path)
}

/// Project samples onto a built-in (or default) HWE-PCA model in-process.
///
/// Mirrors `gnomon project <GENOTYPE_PATH> [--model NAME] [--output-manifest
/// PATH]`. Returns the `output_manifest` path when one was requested, otherwise
/// the `genotype_path` the projection outputs were written next to.
#[pyfunction]
#[pyo3(signature = (genotype_path, model = None, output_manifest = None))]
fn project(
    py: Python<'_>,
    genotype_path: String,
    model: Option<String>,
    output_manifest: Option<String>,
) -> PyResult<String> {
    let genotype_buf = PathBuf::from(&genotype_path);
    let manifest_buf = output_manifest.clone().map(PathBuf::from);

    py.detach(|| {
        map_cli::run(map_cli::MapCommand::Project {
            genotype_path: genotype_buf,
            model,
            output_manifest: manifest_buf,
        })
    })
    .map_err(|err| PyRuntimeError::new_err(format!("gnomon project failed: {err}")))?;

    Ok(output_manifest.unwrap_or(genotype_path))
}

/// Infer sample metadata terms (currently sex) in-process, writing the sex TSV.
///
/// Mirrors `gnomon terms <GENOTYPE_PATH> --sex`. Returns the path of the
/// written sex TSV. `sex` must be `True` (the only term inference available),
/// matching the CLI's requirement that `--sex` be passed.
#[pyfunction]
#[pyo3(signature = (genotype_path, sex = true))]
fn terms(py: Python<'_>, genotype_path: String, sex: bool) -> PyResult<String> {
    if !sex {
        return Err(PyValueError::new_err(
            "No term inference selected. Pass sex=True to run sex inference.",
        ));
    }

    let genotype_buf = PathBuf::from(genotype_path);
    let output_path = py
        .detach(|| infer_sex_to_tsv(&genotype_buf, None))
        .map_err(|err| PyRuntimeError::new_err(format!("gnomon terms failed: {err}")))?;

    Ok(output_path.display().to_string())
}

/// Infer the first sample's biological sex in-process, returning the call
/// directly (no TSV written).
///
/// This is the standalone analog of the `infer_sex`/`detect_build` helpers in
/// the convert_genome spike: the subprocess wrapper cannot return a sex call
/// without running the full `terms` subcommand and re-parsing the written sex
/// TSV. Returns `"male"`, `"female"`, `"indeterminate"`, or `None` when the
/// dataset has no samples.
#[pyfunction]
#[pyo3(name = "infer_sex", signature = (genotype_path))]
fn infer_sex_first(py: Python<'_>, genotype_path: String) -> PyResult<Option<&'static str>> {
    // Leading `::` disambiguates the `infer_sex` crate from this module's
    // `infer_sex`-named pyfunction wrapper.
    use ::infer_sex::InferredSex;

    let genotype_buf = PathBuf::from(genotype_path);
    let call = py
        .detach(|| infer_first_sample_sex(&genotype_buf, None))
        .map_err(|err| PyRuntimeError::new_err(format!("gnomon infer_sex failed: {err}")))?;

    Ok(call.map(|sex| match sex {
        InferredSex::Male => "male",
        InferredSex::Female => "female",
        InferredSex::Indeterminate => "indeterminate",
    }))
}

/// Return a built-in projection model's variant keys as a JSON string.
///
/// Mirrors `gnomon model-keys <MODEL_NAME>` (the CLI's `ModelKeys` subcommand),
/// downloading the model on demand. Returns the JSON document as a `str`.
#[pyfunction]
#[pyo3(signature = (model))]
fn model(py: Python<'_>, model: String) -> PyResult<String> {
    py.detach(|| map_cli::model_variant_keys_json(&model))
        .map_err(|err| PyRuntimeError::new_err(format!("gnomon model-keys failed: {err}")))
}

/// The native `gnomon` extension module. Registered under the name `_gnomon`
/// (see `python/pyproject.toml` `module-name`) and re-exported by the Python
/// package's `__init__.py`.
#[pymodule]
#[pyo3(name = "_gnomon")]
fn gnomon_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__doc__", "Native in-process gnomon core (PyO3).")?;
    module.add_function(wrap_pyfunction!(score, module)?)?;
    module.add_function(wrap_pyfunction!(project, module)?)?;
    module.add_function(wrap_pyfunction!(terms, module)?)?;
    module.add_function(wrap_pyfunction!(infer_sex_first, module)?)?;
    module.add_function(wrap_pyfunction!(model, module)?)?;
    Ok(())
}
