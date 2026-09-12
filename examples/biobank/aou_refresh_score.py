"""Recompute one bounded cohort's score inside its authorized AoU workspace."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tarfile
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

from aou_identity import task_account, require_spot_amd
from aou_checkpoint import StudyCheckpoint, result_identity
from aou_projection import source_identity, stream_projection
from aou_status import failure_label, publish_status, score_progress_label
from aou_survival import (BoundedClient, bounded_fit, build_cohort, case_dates,
                         digest, load_cached_score, load_score_panel, person_times,
                         read_ancestry, unpack_phenotypes,
                         validate_config, write_json)
from disease_selection import select_runtime_diseases


def microarray_prefix(value):
    uri = urlsplit(value)
    if (uri.scheme != "gs" or not uri.netloc or uri.query or uri.fragment
            or not uri.path.endswith("/microarray/plink/arrays")):
        raise ValueError("scoring requires the AoU microarray PLINK arrays prefix")
    return value


def component_scores(path, pgs):
    """Retain exact native components; never turn an absent score into zero."""
    with Path(path).open() as handle:
        if handle.readline().rstrip() != "#SCORE_VARIANT_COUNT\tSCORE\tCOUNT":
            raise ValueError("native score components lack their variant-count header")
        fields = handle.readline().rstrip().split("\t")
        if len(fields) != 3 or fields[:2] != ["#SCORE_VARIANT_COUNT", pgs]:
            raise ValueError("native score components do not identify the requested score")
        total = int(fields[2])
        header = handle.readline().rstrip().split("\t")
        if total <= 0 or header != ["#IID", f"{pgs}_SUM", f"{pgs}_MISSING_CT"]:
            raise ValueError("native score components have no matched variants or an invalid schema")
        frame = pd.read_csv(handle, sep="\t", names=header, dtype={"#IID": str})
    sums = frame[f"{pgs}_SUM"].to_numpy(dtype=float)
    missing = frame[f"{pgs}_MISSING_CT"].to_numpy(dtype=float)
    if (frame.empty or frame["#IID"].isna().any() or frame["#IID"].duplicated().any()
            or not np.isfinite(sums).all() or not np.isfinite(missing).all()
            or (missing < 0).any() or (missing > total).any()
            or (missing != np.floor(missing)).any()):
        raise ValueError("invalid native score components")
    usable = missing < total
    scores = frame.loc[usable, ["#IID"]].copy()
    scores[f"{pgs}_AVG"] = sums[usable] / (total - missing[usable])
    scores[f"{pgs}_MISSING_PCT"] = 100.0 * missing[usable] / total
    if scores.empty:
        raise ValueError("no participants have an observed genetic score")
    return scores, total, int((~usable).sum()), set(frame["#IID"])


def save_scoring_state(checkpoint, score_dir, log, *, complete, status_uri):
    """Keep native continuation state and outputs, without copying genotype spools."""
    for stale in checkpoint.root.glob("*.gnomon-checkpoint.bin"):
        stale.unlink()
    files = [*score_dir.glob("*.gnomon-checkpoint.bin"), *score_dir.glob("*.sscore")]
    for source in files:
        shutil.copyfile(source, checkpoint.root / source.name)
    for source, name in ((log, "fit.log"), (log.with_suffix(".resources.json"), "fit.resources.json")):
        if source.is_file():
            shutil.copyfile(source, checkpoint.root / name)
    if complete:
        checkpoint.complete_step(checkpoint.root, [path.name for path in files])
    else:
        checkpoint.publish()
    if log.is_file():
        label = score_progress_label(log)
        if label is not None:
            publish_status(status_uri, label)


def refresh(args):
    account = task_account()
    config = json.loads(args.config.read_text())
    validate_config(config)
    specification = json.loads(args.scoring_config.read_text())
    if set(specification) != {"endpoint", "genotype_prefix", "scorer_sha256", "weights_sha256",
                              "timeout_seconds"}:
        raise ValueError("scoring configuration has missing or unknown keys")
    runtime = require_spot_amd()
    if (digest(args.scorer) != specification["scorer_sha256"]
            or digest(args.weights) != specification["weights_sha256"]):
        raise ValueError("scorer or scoring weights do not match their pinned hashes")
    microarray_prefix(specification["genotype_prefix"])
    if type(specification["timeout_seconds"]) is not int or not 1 <= specification["timeout_seconds"] <= 600:
        raise ValueError("score preparation must have a wall budget of at most 600 seconds")
    panel = load_score_panel(args.score_panel, exploratory=True)
    endpoint = specification["endpoint"]
    pgs, = panel["endpoints"][endpoint]["candidates"]
    args.output.mkdir(parents=True, exist_ok=True)
    publish_status(args.status_uri, "restoring_score_checkpoint")
    features = source_identity(args.features_uri, config["google_project"], account)
    checkpoint = StudyCheckpoint(args.output / "scoring_state", args.status_uri + ".scoring",
                                 config["google_project"], account,
                                 {"scoring": result_identity(specification),
                                  "analysis": result_identity(config),
                                  "inputs": {name: digest(getattr(args, name)) for name in
                                             ("fam", "ancestry", "prune", "phenotypes", "score_panel")},
                                  "features": features,
                                  "runner_sha256": digest(__file__)},
                                 resume=args.resume_scoring_checkpoint, resume_latest=True)
    artifacts = checkpoint.root / "artifacts"
    if checkpoint.step_is_complete(artifacts):
        for name in ("shared_features.tar.gz", "score_manifest.json"):
            shutil.copyfile(artifacts / name, args.output / name)
        publish_status(args.status_uri, "score_artifact_ready")
        return
    publish_status(args.status_uri, "preparing_score_cohort")
    projection_dir = checkpoint.root / "projection"
    projection_dir.mkdir(exist_ok=True)
    projection = projection_dir / "projection_pcs.parquet"
    if not checkpoint.step_is_complete(projection_dir):
        publish_status(args.status_uri, "reading_projection")
        receipt = stream_projection(features, projection, config["google_project"], account)
        write_json(projection_dir / "source.json", receipt)
        checkpoint.complete_step(projection_dir, ["projection_pcs.parquet", "source.json"])
    publish_status(args.status_uri, "projection_ready")
    client = BoundedClient(config, account)
    preparation = checkpoint.root / "preparation"
    keep = preparation / "keep.txt"
    if not checkpoint.step_is_complete(preparation):
        ancestry = read_ancestry(args.ancestry, args.prune, config["num_pcs"], projection)
        fam = pd.read_csv(args.fam, sep=r"\s+", header=None, usecols=[1], dtype=str)
        samples = fam.rename(columns={1: "person_id"})
        if samples.person_id.isna().any() or samples.person_id.duplicated().any():
            raise ValueError("genotype sample IDs must be present and unique")
        phenotypes = unpack_phenotypes(args.phenotypes, args.output / "phenotypes")
        diseases = select_runtime_diseases(client, config["workspace_cdr"], config["top_n_diseases"], phenotypes)
        if endpoint not in diseases:
            raise ValueError("requested endpoint did not pass the existing disease selection rule")
        base = ancestry.merge(person_times(client, config["workspace_cdr"]),
                              on="person_id", validate="one_to_one")
        cohort = build_cohort(base, samples,
                              case_dates(client, config["workspace_cdr"], diseases[endpoint]["concept_id"]), config)
        if cohort.empty:
            raise ValueError("no eligible genotyped participants for score preparation")
        preparation.mkdir(exist_ok=True)
        keep.write_text("\n".join(cohort.person_id) + "\n")
        write_json(preparation / "queries.json", [job.job_id for job in client.jobs])
        checkpoint.complete_step(preparation, ["keep.txt", "queries.json"])
    requested_ids = set(keep.read_text().splitlines())
    score_dir = args.output / "scoring"
    score_dir.mkdir()
    completed = checkpoint.step_is_complete(checkpoint.root)
    for pattern in (["*.sscore"] if completed else ["*.gnomon-checkpoint.bin"]):
        for saved in checkpoint.root.glob(pattern):
            shutil.copyfile(saved, score_dir / saved.name)
    log = args.output / "score.log"
    publish_status(args.status_uri, "scoring")
    previous = Path.cwd()
    try:
        os.chdir(score_dir)
        try:
            if not completed:
                bounded_fit([str(args.scorer), str(args.weights),
                             specification["genotype_prefix"] + ".bed", "--keep", str(keep), "--emit-components"],
                            specification["timeout_seconds"], log,
                            checkpoint_callback=lambda: save_scoring_state(
                                checkpoint, score_dir, log, complete=False, status_uri=args.status_uri))
        except BaseException:
            # Preserve the native error in workspace storage even when WDL
            # cannot delocalize successful outputs. Never export log text.
            save_scoring_state(checkpoint, score_dir, log, complete=False, status_uri=args.status_uri)
            message = log.read_text(errors="replace").lower()
            for phrases, label in (
                (("permission denied", "http 403", "request violates vpc"), "failed_scoring_permissions"),
                (("failed to load adc credentials", "unauthenticated", "http 401"), "failed_scoring_credentials"),
                (("no such file", "not found", "http 404", "no filesets", "unsupported input",
                  "could not determine input format"), "failed_scoring_input"),
                (("unexpected argument", "unrecognized", "usage:"), "failed_scoring_cli"),
                (("certificate", "tls", "ssl"), "failed_scoring_tls"),
                (("panicked", "symbol lookup", "glibc"), "failed_scoring_runtime"),
            ):
                if any(phrase in message for phrase in phrases):
                    publish_status(args.status_uri, label)
            raise
    finally:
        os.chdir(previous)
    files = list(score_dir.glob("*.sscore"))
    if len(files) != 1:
        raise ValueError("expected exactly one native score component artifact")
    scores, matched, excluded, observed_ids = component_scores(files[0], pgs)
    if observed_ids != requested_ids:
        raise ValueError("native scorer did not return exactly the requested cohort")
    if not completed:
        save_scoring_state(checkpoint, score_dir, log, complete=True, status_uri=args.status_uri)
    score_file = args.output / f"{pgs}.sscore"
    scores.to_csv(score_file, sep="\t", index=False, float_format="%.17g")
    score_tar = args.output / "scores.tar"
    with tarfile.open(score_tar, "w") as archive:
        archive.add(score_file, arcname=score_file.name)
    load_cached_score(score_tar, pgs)
    write_json(args.output / "score_manifest.json", {
        "pgs_id": pgs, "endpoint": endpoint, "genotype_prefix": specification["genotype_prefix"],
        "runtime": runtime,
        "scorer_sha256": digest(args.scorer), "weights_sha256": digest(args.weights),
        "sample_file_sha256": digest(args.fam), "projection_table_sha256": digest(projection),
        "matched_variant_count": matched, "scored_participants": len(observed_ids),
        "unobserved_scores_excluded": excluded, "score_file_sha256": digest(score_file),
        "definition": "native score sum divided by nonmissing matched-variant count",
        "eligibility": "baseline disease-free cohort; score availability required; no outcome balancing",
        "reference_effective_variant_equivalence": "not established",
        "query_job_ids": json.loads((preparation / "queries.json").read_text()),
    })
    with tarfile.open(args.output / "shared_features.tar.gz", "w:gz") as archive:
        for path in (projection, score_tar, args.output / "score_manifest.json"):
            archive.add(path, arcname=f"shared_features/{path.name}")
    artifacts.mkdir(exist_ok=True)
    for name in ("shared_features.tar.gz", "score_manifest.json"):
        shutil.copyfile(args.output / name, artifacts / name)
    checkpoint.complete_step(artifacts, ["shared_features.tar.gz", "score_manifest.json"])
    publish_status(args.status_uri, "score_artifact_ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "scoring-config", "scorer", "weights", "fam",
                 "ancestry", "prune", "phenotypes", "score-panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--status-uri", required=True)
    parser.add_argument("--features-uri", required=True)
    parser.add_argument("--resume-scoring-checkpoint", type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        refresh(args)
    except Exception as error:
        publish_status(args.status_uri, failure_label(error))
        raise


if __name__ == "__main__":
    main()
