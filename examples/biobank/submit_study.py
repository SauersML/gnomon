#!/usr/bin/env python3
"""Submit the AoU study (study.wdl) and read back its aggregate tokens.

Only the lead submits. No participant data is read locally: the run stages its
code, config and the workspace's cached scores, and its only outputs are the
digest's aggregate token names.

  submit_study.py submit [--config study.json] [--cpu 180 --memory-gb 128 --timeout-minutes 180]
                                                 stage one study run as a C3D Spot Batch job
                                                 (--engine cromwell: run study.wdl instead)
  submit_study.py status KEY                     the status labels a run has published
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
import os
from pathlib import Path
import re
import tarfile
import time
import urllib.request

from study import digest
from study.checkpoint import config_hash
import aou_batch
from submit_aou import HERE, Workbench, required_env

STAGING = HERE / ".aou-workflow"
LOOKS = STAGING / "study-looks.jsonl"
# The workspace's WGS score bank (one gnomon .sscore per PGS over the whole short-read
# callset), under one callset fingerprint; the study scores nothing in the task.
SCORE_CACHE = os.environ.get("AOU_SCORE_CACHE_PREFIX", "wgs_scores/2ab675e0fe9f7382af2e980285bab338")
DIGEST = "study-digest"
LOOKS_PREFIX = "workflow-checkpoints/study-looks"
VCPU_HOUR_BUDGET = 540
SOURCES = ["study.py", "aou_checkpoint.py", "aou_identity.py", "aou_projection.py", "aou_status.py",
           "study/__init__.py", "study/checkpoint.py", "study/cohort.py", "study/digest.py", "study/disclosure.py",
           "study/evaluate.py",
           "study/models/__init__.py", "study/models/binary.py", "study/models/survival.py", "study/phenotypes.py",
           "study/pool.py", "study/requirements.txt"]


def newest_cached_scores(objects, prefix, pgs_ids):
    """{pgs: newest cached .sscore object} for the scores the workspace has cached;
    cache keys change with the scorer binary."""
    newest = {}
    # <prefix>/<key>/<PGS>.sscore, or <prefix>/<PGS>.sscore when the prefix names the key itself.
    pattern = re.compile(re.escape(prefix) + r"/(?:[^/]+/)?(PGS\d+)\.sscore")
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
    # ...and per deployment (project, CDR): a store's manifest signs the deployment too,
    # so a run against another CDR must not restore, and be refused by, this one's store.
    from study.checkpoint import deployment_identity
    key = hashlib.sha256(sources.read_bytes() + config_sha.encode()
                         + json.dumps(deployment_identity(config), sort_keys=True).encode()).hexdigest()[:20]
    pgs_ids = {entry["pgs"] for entry in diseases}
    objects = newest_cached_scores(wb.list_objects(f"{wb.bucket}/{SCORE_CACHE}/"), SCORE_CACHE, pgs_ids)
    uncached = sorted(pgs_ids - set(objects))
    if uncached and not args.allow_scoring:
        raise ValueError(f"no cached score under {SCORE_CACHE} for {uncached}: the study runs on cached scores "
                         f"only (switch the disease's score in diseases.json, or pass --allow-scoring)")
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
    if args.engine == "batch":
        record = stage_batch_job(wb, name, wdl, inputs, folder, args, diseases)
    else:
        record = wb.run_workflow(name, f"{uri}/{wdl.name}", "AoU single study", inputs, folder,
                                 storage_capacity=100)
    with LOOKS.open("a") as ledger:
        ledger.write(json.dumps({"run": name, "run_id": record.get("runId"), "config_sha256": config_sha,
                                 "checkpoint_key": key, "look": look,
                                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    print(f"outer-test look {look} for config {config_sha[:12]}; status labels under {checkpoint}.status/",
          flush=True)


def task_service_account(wb):
    """The workspace pet service account the task runs as: AOU_TASK_SERVICE_ACCOUNT,
    or the one `wb auth status` reports for the current workspace."""
    email = os.environ.get("AOU_TASK_SERVICE_ACCOUNT", "").strip()
    if not email:
        for line in wb.wb("auth", "status").splitlines():
            if line.startswith("Service account email for current workspace:"):
                email = line.split(":", 1)[1].strip()
    if not re.fullmatch(r"pet-[0-9a-f]+@" + re.escape(wb.project) + r"\.iam\.gserviceaccount\.com", email):
        raise RuntimeError(f"the task service account must be the workspace pet account, not {email!r}")
    return email


ORCHESTRATOR = "orchestrator"


def stage_batch_job(wb, name, wdl, inputs, folder, args, diseases):
    """Write the study's Batch job(s) beside its staged inputs and hand the submit
    command to the in-perimeter orchestrator (the app VM's loop reads
    orchestrator/cmd/<id>.sh from the bucket and runs it as the pet account).
    --split: one shard job per disease on --shard-cpu vCPUs, then the whole-study
    job that gathers them (study.py --shard); otherwise one job for the whole study."""
    fields = {key.split(".", 1)[1]: value for key, value in inputs.items()}
    account = task_service_account(wb)
    (folder / "inputs.json").write_text(json.dumps(inputs, indent=2))
    prefix = f"{wb.bucket}/workflows/{name}"
    documents = {}
    if args.split:
        # A split run keeps its own store and status prefix: a whole-study run of the same
        # code and config must never delete the store its shards are writing.
        fields = dict(fields, checkpoint_uri=fields["checkpoint_uri"].rstrip("/") + "-split/",
                      status_uri=fields["status_uri"] + "-split")
        shard_status = {}
        for entry in diseases:
            slug = entry["slug"]
            shard = f"{name}-{slug.replace('_', '-')}"[:63].rstrip("-")
            own = dict(fields, status_uri=f"{fields['status_uri']}-{slug}")
            documents[shard] = aou_batch.job(shard, wdl, own, wb.project, account, args.shard_cpu,
                                             min(args.memory_gb, aou_batch.machine_type(args.shard_cpu, 1)[1]),
                                             args.timeout_minutes, shard=slug)  # a shard always keeps the store
            shard_status[own["status_uri"]] = documents[shard]
    documents[name] = aou_batch.job(name, wdl, fields, wb.project, account, args.cpu, args.memory_gb,
                                    args.timeout_minutes, keep_store=args.keep_store,
                                    wait_for=list(shard_status) if args.split else ())
    uris = {}
    for job_name, document in documents.items():
        path = folder / f"batch-{job_name}.json"
        path.write_text(json.dumps(document, indent=1) + "\n")
        uris[job_name] = wb.stage([path], prefix)[0]
    if args.split:
        shards = [(job_name, uris[job_name]) for job_name in documents if job_name != name]
        command = aou_batch.split_command(name, shards, (name, uris[name]), wb.project)
    else:
        command = aou_batch.paste_command(name, uris[name], wb.project).replace("/tmp/", "/w/")
    (folder / "orchestrator-cmd.sh").write_text(command if command.endswith("\n") else command + "\n")
    command_uri = wb.stage_as(folder / "orchestrator-cmd.sh", f"{wb.bucket}/{ORCHESTRATOR}/cmd/{name}.sh")
    record = {"runId": name, "engine": "batch", "job_uris": uris, "orchestrator_command": command_uri,
              "machines": {job_name: d["allocationPolicy"]["instances"][0]["policy"]["machineType"]
                           for job_name, d in documents.items()}}
    (folder / "submission.json").write_text(json.dumps(record, indent=2))
    print(f"{len(documents)} Batch job(s) staged under {prefix}; the orchestrator runs {command_uri} "
          f"(machines: {sorted(set(record['machines'].values()))}). Read its reply with: "
          f"submit_study.py orch {name}", flush=True)
    return record


def orch(wb, args):
    """Decode an orchestrator command's reply, which it emits as object names
    (orchestrator/out/<id>/t/<seq>__<base64url chunk>) because objects cannot be read
    from outside the perimeter; a DONE name marks the end."""
    import base64
    prefix = f"{wb.bucket}/{ORCHESTRATOR}/out/{args.id}/"
    names = sorted(item["name"].rsplit("/", 1)[-1] for item in wb.list_objects(prefix + "t/"))
    text = "".join(name.split("__", 1)[1] for name in names if "__" in name)
    print(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)).decode("utf-8", "replace"), end="")
    done = any(item["name"].endswith("/DONE") for item in wb.list_objects(prefix))
    print(f"[{args.id}: {'finished' if done else 'still running or not started'}]")
    beats = sorted(item["name"].rsplit("/", 1)[-1] for item in wb.list_objects(f"{wb.bucket}/{ORCHESTRATOR}/hb/"))
    print(f"[orchestrator heartbeat: {beats[-1] if beats else 'none yet'}]")


def status(wb, args):
    """The status labels a run's task has published (listing is allowed from outside the perimeter)."""
    prefix = f"{wb.bucket}/workflow-checkpoints/study-{args.key}.status/"
    for item in sorted(wb.list_objects(prefix), key=lambda item: item["updated"]):
        print(item["updated"], item["name"].rsplit("/", 1)[-1].removesuffix(".txt"))


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
    child.add_argument("--cpu", type=int, default=180,
                       help="task vCPUs: one c3d-standard-180 Spot box (AMD Genoa). The 40k simulator run measured "
                            "4.4 CPU-hours of binary fits per disease at 16k training rows on gamfit 0.1.276 (Rome "
                            "cores), about 300 CPU-hours for the whole study at 300k participants; the ~50 fits per "
                            "disease are independent, so one big box finishes in about 1.5 hours (MSI, 2026-09-24)")
    child.add_argument("--memory-gb", type=int, default=128)
    child.add_argument("--memory-limit-gb", type=int, default=128,
                       help="raise only with a measured need (SPEC 7a)")
    child.add_argument("--timeout-minutes", type=int, default=180)
    child.add_argument("--keep-store", action=argparse.BooleanOptionalAction, default=True,
                       help="keep the bucket checkpoint after the run, so the next run of the same code and config "
                            "restores its cohort and features instead of querying them (SPEC 7a asks for deletion; "
                            "time is the deliverable, user 2026-09-24; --no-keep-store deletes)")
    child.add_argument("--allow-scoring", action="store_true",
                       help="score uncached PGS in the task (default: refuse; the study runs on the score bank)")
    child.add_argument("--split", action="store_true",
                       help="one Spot Batch job per disease (--shard-cpu vCPUs each), then the whole-study "
                            "job gathers them: the fits are the cost, and the diseases are independent")
    child.add_argument("--shard-cpu", type=int, default=60, help="vCPUs of each disease shard (C3D high-CPU shape)")
    child.add_argument("--engine", choices=["batch", "cromwell"], default="batch",
                       help="batch: one C3D Spot Batch job, submitted from the workspace terminal (the "
                            "managed Cromwell has no C3D); cromwell: the study.wdl run as before")
    child.add_argument("--caveat", action="append", default=[], metavar="LABEL",
                       help="a fixed label the run's digest carries, e.g. shipped_survival_refused_gam2945 (repeatable)")
    child.set_defaults(handler=submit)
    child = sub.add_parser("orch")
    child.add_argument("id", help="the orchestrator command id (the run name)")
    child.set_defaults(handler=orch)
    child = sub.add_parser("status")
    child.add_argument("key", help="the checkpoint key the submission printed (study-<key>)")
    child.set_defaults(handler=status)
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
