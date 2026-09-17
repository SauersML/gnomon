#!/usr/bin/env python3
"""Submit the pilot's bounded workspace runs; no participant data is read locally.

Every command stages its files with verified media uploads and starts its run
through Workbench.run_workflow, which tolerates Workbench hanging after it has
acted. Deployment state stays in the git-ignored .aou-workflow directory: the
portable scorer, configurations, the current checkpoint URI, and one folder per
submission holding its inputs, workflow record and receipt.

  score-training           hypertension score-and-survival pilot
                           (AOU_ANALYSIS_CONFIG selects the analysis, AOU_FULL_ANALYSIS=1
                           the confirmatory analysis)
  benchmark                score-method benchmark from the workspace sscore cache
                           (aou_benchmark.json; AOU_BENCHMARK_CONFIG selects another)
  scoring-diagnostic       scoring resource counters of the current checkpoint
  fit-diagnostic           survival fit progress of the current checkpoint
  stderr-diagnostic URI    one failed task's stderr, scanned with its checkpoint
  results-digest URI       a completed run's metrics as aggregate result tokens
  finish FOLDER            start a staged submission whose run did not start
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
import time

from aou_checkpoint import result_identity
from submit_aou import HERE, Workbench, required_env

STAGING = HERE / ".aou-workflow"
MICROARRAY = "gs://vwb-aou-datasets-controlled/v8/microarray/plink/arrays"
PGS004525_WEIGHTS_SHA256 = "017127773cc569986aa6e0de6c433920ab9b833c3cf8cec7c242a3d9e66718d8"
BENCHMARK_MODULES = ["aou_benchmark.py", "aou_checkpoint.py", "aou_evaluation.py", "aou_identity.py",
                     "aou_projection.py", "aou_score_transform.py", "aou_status.py", "aou_survival.py",
                     "disease_selection.py", "reference_ctn.py", "aou_requirements.txt"]


def new_submission(wb, stem):
    name = f"{stem}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    folder = STAGING / name
    folder.mkdir(parents=True)
    return name, folder, f"{wb.bucket}/workflows/{name}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def current_checkpoint(args):
    return args.checkpoint or (STAGING / "current-checkpoint-uri").read_text().strip()


def score_training(wb, args):
    portable = STAGING / "portable-scorer"
    scorer_sha = (portable / "scorer.sha256").read_text().split()[0]
    archive = portable / "scorer.tar.gz"
    if sha256(archive) != (portable / "archive.sha256").read_text().split()[0]:
        raise ValueError("Portable scorer archive checksum mismatch")
    name, folder, uri = new_submission(wb, "aou-score-training")
    config_path = os.environ.get("AOU_ANALYSIS_CONFIG") or STAGING / "hypertension-native-analysis.json"
    config = json.loads(Path(config_path).read_text())
    config.update(google_project=wb.project, workspace_cdr=required_env("WORKSPACE_CDR"))
    (folder / "analysis.json").write_text(json.dumps(config))
    spec = dict(endpoint="hypertension", genotype_prefix=MICROARRAY, scorer_sha256=scorer_sha,
                weights_sha256=PGS004525_WEIGHTS_SHA256, timeout_seconds=600)
    (folder / "scoring.json").write_text(json.dumps(spec))
    sources = STAGING / "score-training-sources.tar"
    wdl = HERE / "aou_score_training.wdl"
    checkpoint_key = hashlib.sha256(sources.read_bytes() + json.dumps(
        {"analysis": result_identity(config), "scoring": result_identity(spec)},
        sort_keys=True).encode()).hexdigest()[:20]
    # The scorer archive is content-addressed; a staged copy with the same MD5 is reused.
    scorer_uri = f"{wb.bucket}/artifacts/aou-training/gnomon-score-{scorer_sha}.tar.gz"
    wb.stage_as(archive, scorer_uri, reuse=True)
    wb.stage([sources, wdl, folder / "analysis.json", folder / "scoring.json"], uri)
    values = dict(smoke_only=os.environ.get("AOU_FULL_ANALYSIS") != "1",
                  sources=f"{uri}/{sources.name}", analysis_config=f"{uri}/analysis.json",
                  scoring_config=f"{uri}/scoring.json",
                  score_weights=f"{wb.bucket}/workflows/aou-score-training-20260911-224803/PGS004525.gnomon.tsv",
                  genotype_fam=MICROARRAY + ".fam",
                  scorer_archive=scorer_uri,
                  prior_shared_features_uri=required_env("AOU_SHARED_FEATURES_URI"),
                  ancestry_predictions=required_env("AOU_ANCESTRY_URI"),
                  relatedness_prune=required_env("AOU_RELATEDNESS_PRUNE_URI"),
                  phenotype_library_archive=required_env("AOU_PHENOTYPE_LIBRARY_URI"),
                  reference_ctn=required_env("AOU_REFERENCE_CTN_URIS").split(),
                  wheelhouse_archive=required_env("AOU_WHEELHOUSE_URI"),
                  runtime_image=required_env("AOU_RUNTIME_IMAGE"),
                  checkpoint_uri=f"{wb.bucket}/workflow-checkpoints/hypertension-{checkpoint_key}.tar.gz")
    inputs = {"aou_score_training." + key: value for key, value in values.items()}
    wb.run_workflow(name, f"{uri}/{wdl.name}", "Hypertension score and native survival pilot",
                    inputs, folder, storage_capacity=50)


def newest_cached_scores(objects, prefix, pgs_ids):
    """The newest cached .sscore object per score; cache keys change with the scorer binary."""
    newest = {}
    pattern = re.compile(re.escape(prefix) + r"/[^/]+/(PGS\d+)\.sscore")
    for item in objects:
        match = pattern.fullmatch(item["name"])
        if match and match.group(1) in pgs_ids:
            if match.group(1) not in newest or item["updated"] > newest[match.group(1)]["updated"]:
                newest[match.group(1)] = item
    missing = sorted(set(pgs_ids) - set(newest))
    if missing:
        raise ValueError(f"{', '.join(missing)} has no cached score in the workspace")
    return [newest[pgs]["name"] for pgs in sorted(pgs_ids)]


def benchmark(wb, args):
    name, folder, uri = new_submission(wb, "aou-benchmark")
    config = json.loads(Path(os.environ.get("AOU_BENCHMARK_CONFIG") or HERE / "aou_benchmark.json").read_text())
    config.update(google_project=wb.project, workspace_cdr=required_env("WORKSPACE_CDR"))
    (folder / "config.json").write_text(json.dumps(config, indent=2))
    sources = folder / "benchmark-sources.tar"
    with tarfile.open(sources, "w") as archive:
        for module in BENCHMARK_MODULES:
            archive.add(HERE / module, arcname=module)
    cache = "artifacts/aou-training/sscore_cache"
    pgs_ids = {pgs for disease in config["diseases"].values() for pgs in disease["scores"]}
    names = newest_cached_scores(wb.list_objects(f"{wb.bucket}/{cache}/"), cache, pgs_ids)
    score_files = [f"{wb.bucket}/{object_name}" for object_name in names]
    print(f"Using {len(score_files)} cached scores", flush=True)
    wdl = HERE / "aou_benchmark.wdl"
    wb.stage([sources, wdl, folder / "config.json"], uri)
    key = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    inputs = {"aou_benchmark." + k: v for k, v in dict(
        sources=f"{uri}/{sources.name}", config=f"{uri}/config.json",
        wheelhouse_archive=required_env("AOU_WHEELHOUSE_URI"),
        ancestry_predictions=required_env("AOU_ANCESTRY_URI"),
        relatedness_prune=required_env("AOU_RELATEDNESS_PRUNE_URI"),
        features_uri=required_env("AOU_SHARED_FEATURES_URI"),
        score_files=score_files, runtime_image=required_env("AOU_RUNTIME_IMAGE"),
        status_uri=f"{wb.bucket}/workflow-checkpoints/benchmark-{key}-{name[-15:]}.tar.gz").items()}
    wb.run_workflow(name, f"{uri}/{wdl.name}", "Score method benchmark", inputs, folder,
                    storage_capacity=50)


def diagnostic(wb, stem, display_name, values):
    name, folder, uri = new_submission(wb, stem)
    wdl = HERE / "aou_diagnostic.wdl"
    wb.stage([wdl, HERE / "aou_identity.py", HERE / "aou_status.py"], uri)
    values = dict(values, identity_guard=f"{uri}/aou_identity.py", status_module=f"{uri}/aou_status.py",
                  runtime_image=required_env("AOU_RUNTIME_IMAGE"))
    inputs = {"aou_diagnostic." + key: value for key, value in values.items()}
    wb.run_workflow(name, f"{uri}/{wdl.name}", display_name, inputs, folder)


def scoring_diagnostic(wb, args):
    checkpoint = current_checkpoint(args)
    diagnostic(wb, "aou-score-resources", "Scoring resource diagnostic",
               dict(checkpoint=checkpoint + ".scoring", status_uri=checkpoint))


def fit_diagnostic(wb, args):
    checkpoint = current_checkpoint(args)
    diagnostic(wb, "aou-fit-diagnostic", "Survival fit progress diagnostic",
               dict(checkpoint=checkpoint, status_uri=checkpoint))


def stderr_diagnostic(wb, args):
    checkpoint = current_checkpoint(args)
    diagnostic(wb, "aou-stderr-diagnostic", "Task stderr diagnostic",
               dict(task_stderr=args.uri, checkpoint=checkpoint, status_uri=checkpoint))


def results_digest(wb, args):
    name, folder, uri = new_submission(wb, "aou-results-digest")
    wdl = HERE / "aou_results_digest.wdl"
    wb.stage([wdl, HERE / "aou_identity.py"], uri)
    inputs = {"aou_results_digest." + key: value for key, value in dict(
        metrics=args.uri, identity_guard=f"{uri}/aou_identity.py",
        runtime_image=required_env("AOU_RUNTIME_IMAGE")).items()}
    wb.run_workflow(name, f"{uri}/{wdl.name}", "Aggregate results digest", inputs, folder)


def finish(wb, args):
    folder = args.folder.resolve()
    workflow = json.loads((folder / "workflow.json").read_text())
    if workflow["name"] != folder.name:
        raise ValueError("workflow.json names a different submission than its folder")
    inputs = json.loads((folder / "inputs.json").read_text())
    wb.run_workflow(workflow["name"], workflow["wdl"], workflow["display_name"], inputs, folder,
                    storage_capacity=workflow["storage_capacity"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, handler in (("score-training", score_training), ("benchmark", benchmark)):
        sub.add_parser(command).set_defaults(handler=handler)
    for command, handler in (("scoring-diagnostic", scoring_diagnostic), ("fit-diagnostic", fit_diagnostic)):
        child = sub.add_parser(command)
        child.add_argument("--checkpoint", help="checkpoint URI instead of .aou-workflow/current-checkpoint-uri")
        child.set_defaults(handler=handler)
    child = sub.add_parser("stderr-diagnostic")
    child.add_argument("uri", help="gs:// URI of the failed task's stderr")
    child.add_argument("--checkpoint", help="checkpoint URI instead of .aou-workflow/current-checkpoint-uri")
    child.set_defaults(handler=stderr_diagnostic)
    child = sub.add_parser("results-digest")
    child.add_argument("uri", help="gs:// URI of a completed run's metrics.json")
    child.set_defaults(handler=results_digest)
    child = sub.add_parser("finish")
    child.add_argument("folder", type=Path, help="the submission folder under .aou-workflow")
    child.set_defaults(handler=finish)
    args = parser.parse_args()
    args.handler(Workbench(), args)


if __name__ == "__main__":
    main()
