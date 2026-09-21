#!/usr/bin/env python3
"""Submit the AoU study (study.wdl) and read back its aggregate tokens.

Only the lead submits. No participant data is read locally: the run stages its
code, config and the workspace's cached scores, and its only outputs are the
digest's aggregate token names.

  submit_study.py submit [--config study.json] [--cpu 16 --memory-gb 64 --timeout-minutes 110]
                                                 stage and start one study run
  submit_study.py tokens RUN                     print a finished run's token names
                                                 (pipe into a file for tabulate_study.py)
  submit_study.py finish FOLDER                  start a staged submission whose run did not start

Environment: the Workbench variables submit_aou.Workbench reads, plus
WORKSPACE_CDR, AOU_STUDY_WHEELHOUSE_URI, AOU_SHARED_FEATURES_URI, AOU_ANCESTRY_URI,
AOU_RELATEDNESS_PRUNE_URI and AOU_RUNTIME_IMAGE. Every submission appends to the
local ledger of outer-test looks (SPEC section 8), keyed by the config's hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.request

from study import digest
from study.checkpoint import config_hash
from submit_aou import HERE, Workbench, required_env

STAGING = HERE / ".aou-workflow"
LOOKS = STAGING / "study-looks.jsonl"
SCORE_CACHE = "artifacts/aou-training/sscore_cache"
DIGEST = "study-digest"
LOOKS_PREFIX = "workflow-checkpoints/study-looks"
VCPU_HOUR_BUDGET = 32
SOURCES = ["study.py", "aou_checkpoint.py", "aou_identity.py", "aou_projection.py", "aou_status.py",
           "study/__init__.py", "study/checkpoint.py", "study/cohort.py", "study/digest.py", "study/disclosure.py",
           "study/evaluate.py",
           "study/models/__init__.py", "study/models/binary.py", "study/phenotypes.py",
           "study/pool.py", "study/requirements.txt"]


def newest_cached_scores(objects, prefix, pgs_ids):
    """{pgs: newest cached .sscore object} for the scores the workspace has cached;
    cache keys change with the scorer binary."""
    newest = {}
    pattern = re.compile(re.escape(prefix) + r"/[^/]+/(PGS\d+)\.sscore")
    for item in objects:
        match = pattern.fullmatch(item["name"])
        if match and match.group(1) in pgs_ids:
            if match.group(1) not in newest or item["updated"] > newest[match.group(1)]["updated"]:
                newest[match.group(1)] = item
    return {pgs: item["name"] for pgs, item in sorted(newest.items())}


def catalog_scoring_file(pgs):
    """The PGS Catalog's harmonized GRCh38 scoring file, downloaded here (the
    workspace has no egress) and checked against the Catalog's own md5."""
    name = f"{pgs}_hmPOS_GRCh38.txt.gz"
    target = STAGING / "scoring-files" / name
    base = f"https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs}/ScoringFiles/Harmonized/{name}"
    with urllib.request.urlopen(base + ".md5", timeout=60) as response:
        expected = response.read().decode().split()[0]
    if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != expected:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        with urllib.request.urlopen(base, timeout=300) as response, partial.open("wb") as handle:
            handle.write(response.read())
        if hashlib.md5(partial.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} does not match the Catalog's md5")
        partial.replace(target)
    return target


def resolved_config(path, wb):
    """study.json with its disease list inlined and the deployment filled in."""
    config = json.loads(Path(path).read_text())
    if config["data"].get("google_project") or config["data"].get("workspace_cdr"):
        raise ValueError("the deployment project and CDR come from the environment, not the config")
    if "diseases" not in config:
        config["diseases"] = json.loads((Path(path).parent / config.pop("diseases_file")).read_text())
    config.pop("diseases_file", None)
    config["data"] = dict(config["data"], source="bigquery", google_project=wb.project,
                          workspace_cdr=required_env("WORKSPACE_CDR"))
    block = config["diseases"]
    block = block["diseases"] if isinstance(block, dict) and "diseases" in block else block
    entries = list(block.values()) if isinstance(block, dict) else list(block)
    if not entries or any(not re.fullmatch(r"PGS\d{6}", str(entry.get("pgs"))) for entry in entries):
        raise ValueError("every disease needs one PGS Catalog score")
    return config, entries


def submit(wb, args):
    digest.check_caveats(args.caveat)  # refused here, not after the task has started
    config, diseases = resolved_config(args.config, wb)
    # SPEC section 8 (S8): the outer test is evaluated only under the frozen config.
    if config.get("frozen_config_sha256") != config_hash(config):
        raise ValueError(f"freeze the config first: frozen_config_sha256 is {config.get('frozen_config_sha256')}, "
                         f"the config hashes to {config_hash(config)} (study.py hash)")
    # SPEC section 7a: a run may not cost more than its vCPU-hour budget.
    if args.cpu * args.timeout_minutes / 60 > VCPU_HOUR_BUDGET or args.memory_gb > args.memory_limit_gb:
        raise ValueError(f"{args.cpu} vCPU x {args.timeout_minutes} min exceeds {VCPU_HOUR_BUDGET} vCPU-hours, "
                         f"or {args.memory_gb} GiB exceeds {args.memory_limit_gb} GiB")
    name = f"study-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    folder = STAGING / name
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True) + "\n")
    sources = folder / "study-sources.tar"
    with tarfile.open(sources, "w") as archive:
        for relative in SOURCES:
            archive.add(HERE / relative, arcname=relative)
    config_sha = config_hash(config)
    # One checkpoint per code and config: a resubmission of both resumes it.
    key = hashlib.sha256(sources.read_bytes() + config_sha.encode()).hexdigest()[:20]
    pgs_ids = {entry["pgs"] for entry in diseases}
    objects = newest_cached_scores(wb.list_objects(f"{wb.bucket}/{SCORE_CACHE}/"), SCORE_CACHE, pgs_ids)
    uncached = sorted(pgs_ids - set(objects))
    print(f"Using {len(objects)} cached scores; scoring {uncached or 'none'} in the workspace", flush=True)
    # Content-addressed artifacts: a staged copy with the same MD5 is reused.
    portable = STAGING / "portable-scorer"
    scorer = portable / "scorer.tar.gz"
    if hashlib.sha256(scorer.read_bytes()).hexdigest() != (portable / "archive.sha256").read_text().split()[0]:
        raise ValueError("portable scorer archive checksum mismatch")
    scorer_sha = (portable / "scorer.sha256").read_text().split()[0]
    scorer_uri = wb.stage_as(scorer, f"{wb.bucket}/artifacts/aou-training/gnomon-score-{scorer_sha}.tar.gz",
                             reuse=True)
    weights = []
    for pgs in uncached:
        path = catalog_scoring_file(pgs)
        weights.append(wb.stage_as(path, f"{wb.bucket}/artifacts/aou-study/scoring-files/{path.name}", reuse=True))
    looks = [json.loads(line) for line in LOOKS.read_text().splitlines()] if LOOKS.exists() else []
    look = 1 + sum(entry["config_sha256"] == config_sha for entry in looks)
    uri = f"{wb.bucket}/workflows/{name}"
    wdl = HERE / "study.wdl"
    wb.stage([sources, wdl, folder / "config.json"], uri)
    checkpoint = f"{wb.bucket}/workflow-checkpoints/study-{key}"
    inputs = {"study." + field: value for field, value in dict(
        sources=f"{uri}/{sources.name}", config=f"{uri}/config.json",
        wheelhouse_archive=required_env("AOU_STUDY_WHEELHOUSE_URI"), scorer_archive=scorer_uri,
        score_files=[f"{wb.bucket}/{object_name}" for object_name in objects.values()], score_weights=weights,
        ancestry_predictions=required_env("AOU_ANCESTRY_URI"),
        relatedness_prune=required_env("AOU_RELATEDNESS_PRUNE_URI"),
        features_uri=required_env("AOU_SHARED_FEATURES_URI"),
        runtime_image=required_env("AOU_RUNTIME_IMAGE"),
        checkpoint_uri=checkpoint + "/", status_uri=checkpoint,
        digest_uri=f"{wb.bucket}/{DIGEST}/{name}/", looks_uri=f"{wb.bucket}/{LOOKS_PREFIX}/{config_sha[:20]}/",
        cpu=args.cpu, memory_gb=args.memory_gb, timeout_minutes=args.timeout_minutes, caveats=args.caveat).items()}
    record = wb.run_workflow(name, f"{uri}/{wdl.name}", "AoU single study", inputs, folder, storage_capacity=100)
    with LOOKS.open("a") as ledger:
        ledger.write(json.dumps({"run": name, "run_id": record.get("runId"), "config_sha256": config_sha,
                                 "checkpoint_key": key, "look": look,
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    print(f"outer-test look {look} for config {config_sha[:12]}; status labels under {checkpoint}.status/",
          flush=True)


def tokens(wb, args):
    for item in wb.list_objects(f"{wb.bucket}/{DIGEST}/{args.run}/"):
        leaf = item["name"].rsplit("/", 1)[-1]
        if leaf.startswith(("d__", "o__")):
            print(leaf)


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
    child = sub.add_parser("submit")
    child.add_argument("--config", type=Path, default=HERE / "study.json")
    child.add_argument("--cpu", type=int, default=16, help="task vCPUs (size from the MSI end-to-end run)")
    child.add_argument("--memory-gb", type=int, default=64)
    child.add_argument("--memory-limit-gb", type=int, default=64,
                       help="raise only with a measured need (SPEC 7a)")
    child.add_argument("--timeout-minutes", type=int, default=110)
    child.add_argument("--caveat", action="append", default=[], metavar="LABEL",
                       help="a fixed label the run's digest carries, e.g. shipped_survival_refused_gam2945 (repeatable)")
    child.set_defaults(handler=submit)
    child = sub.add_parser("tokens")
    child.add_argument("run", help="the submission name, e.g. study-20260918-230000")
    child.set_defaults(handler=tokens)
    child = sub.add_parser("finish")
    child.add_argument("folder", type=Path, help="the submission folder under .aou-workflow")
    child.set_defaults(handler=finish)
    args = parser.parse_args()
    args.handler(Workbench(), args)


if __name__ == "__main__":
    main()
