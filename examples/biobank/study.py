#!/usr/bin/env python3
"""The AoU study in one resumable run: scores -> cohort -> features -> fits ->
predict -> evaluate -> digest.

The cohort stage produces the SCHEMA.md tables (BigQuery and release files in
the workspace; the simulator writes the same tables on MSI), so every later
stage runs identically on synthetic data and in AoU. Scores come first because
the tables' score cache must hold every study score before they are exported:
scoring needs no cohort, since it covers every array sample.

Each unit of work is a checkpoint step (study/checkpoint.py), so a Spot
preemption costs only the steps in flight. Fits, predictions and evaluations
run as separate processes under one thread budget (study/pool.py): the gamfit
hot loop is partly serial, so concurrency across fits is what fills the task. A
failed fit is a result, recorded and reported, never a reason to stop the
others. Only aggregate tokens leave the workspace (study/digest.py).

  study.py run --work DIR --source parquet --input tables=DIR            (MSI, simulator tables)
  study.py run --work DIR --input NAME=PATH ... --checkpoint gs://BUCKET/PREFIX \\
               --status-uri gs://BUCKET/OBJECT                          (AoU task)
  study.py worker                                                        (pool worker, internal)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from aou_checkpoint import file_hash  # noqa: E402
from aou_status import STUDY_STAGES as STAGES  # noqa: E402
from study import digest  # noqa: E402
from study.checkpoint import (Checkpoint, GcsStore, LocalStore, config_hash, deployment_identity,  # noqa: E402
                              study_identity)
from study.pool import Job, run_jobs, task_memory_bytes  # noqa: E402

KINDS = ("binary", "survival")
# Outcome columns: in a training frame, never in a prediction frame. n_dates
# counts the disease's qualifying records, so it is an outcome too.
OUTCOMES = {"binary": ("y", "n_dates"), "survival": ("exit_age", "event", "followup", "n_dates")}
LOGO_AXES = ("ancestry", "region", "ehr_site")
# Binary y; survival event: 0 censored (an exclusion match censors too), 1 disease, 2 death. Nothing else.
OUTCOME_CODES = {"binary": {0, 1}, "survival": {0, 1, 2}}
# study.json primary_censoring_rule -> phenotypes.build_frames(censor=...).
CENSORING = {"ehr_end": "ehr_end", "min_death_cutoff": "cutoff"}
UNKNOWN = "unknown"
# The fit label of a pooled fit refitted from a second start (convergence gate).
RESTART = "restart"
# SPEC section 8, minimum events: a fit below the bar is this result, not a failure.
INSUFFICIENT_EVENTS = "insufficient_events"
# The event code each survival component models.
SURVIVAL_CAUSES = {"disease": 1, "death": 2}
# A config's label: "production" for the shipped study.json, a named dev variant otherwise.
LABEL = re.compile(r"[a-z0-9_]{1,40}")
LOG_TAIL = 256 * 1024


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, sort_keys=True, allow_nan=False, default=_plain) + "\n")


def _plain(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON")


def read_json(path):
    return json.loads(Path(path).read_text())


def fit_slug(fit):
    return fit.replace(":", "__")


def person_set_hash(person_ids):
    ids = np.sort(np.asarray(person_ids, dtype=np.int64))
    return hashlib.sha256(ids.tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
# configuration and identity
# --------------------------------------------------------------------------- #
def load_config(path, source=None):
    """The study config with its disease list resolved into config["diseases"]."""
    from study import phenotypes
    path = Path(path).resolve()
    config = read_json(path)
    if source is not None:
        config["data"] = dict(config["data"], source=source)
    required = {"study", "data", "cohort", "logo", "models", "compute", "report"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"study config lacks {sorted(missing)}")
    if "horizons_years" in config:
        raise ValueError("horizons come from the prespecified outcome-blind rule (cohort horizon_candidates)")
    if "diseases" not in config:
        config["diseases"] = read_json(path.parent / config["diseases_file"])
    if re.search(r"standard[-_ ]normal", json.dumps(config["models"]), re.IGNORECASE):
        raise ValueError("a shipped study config may not declare a standard-normal latent law (SPEC section 4)")
    check_reasons(config)
    if config["report"]["small_cell_max"] != digest.LIMIT:
        raise ValueError(f"the small-cell maximum is AoU policy's {digest.LIMIT}, one constant everywhere")
    if not config.get("variants") or not set(config["logo"]["variants"]) <= set(config["variants"]):
        raise ValueError("study.json lists its variants, and the LOGO variants are among them")
    convergence = config.get("convergence") or {}
    if (not set(convergence.get("variants", ())) <= set(config["variants"])
            or convergence.get("start") not in ("permuted_rows", "warm")
            or not isinstance(convergence.get("max_delta_sd"), (int, float)) or not convergence["max_delta_sd"] > 0):
        raise ValueError("study.json convergence needs variants among the study's, a start (permuted_rows or warm) "
                         "and a positive max_delta_sd (SPEC section 8)")
    minimum = (config.get("fit_gate") or {}).get("min_events")
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("study.json fit_gate.min_events must be a positive count (SPEC section 8)")
    check_claims(config)
    check_declared_refusals(config)
    if not LABEL.fullmatch(str(config.get("label", ""))):
        raise ValueError("study.json needs a label (production, or a named dev configuration such as dev_k6_offspec)")
    unknown = set(config["logo"]["axes"]) - set(LOGO_AXES)
    if unknown:
        raise ValueError(f"LOGO axes are {LOGO_AXES}, not {sorted(unknown)}")
    diseases = phenotypes.load_diseases(config["diseases"])
    if not diseases:
        raise ValueError("the disease list is empty")
    phenotypes.CohortConfig.from_json(config["cohort"])
    # SPEC section 8 (N2b): the primary survival censoring, chosen before the claim run.
    if config.get("primary_censoring_rule") not in CENSORING:
        raise ValueError(f"primary_censoring_rule must be one of {sorted(CENSORING)}")
    return config, diseases


def leaf_settings(settings, prefix):
    """Dotted paths of every leaf value under a settings dict."""
    for key, value in settings.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict) and value:
            yield from leaf_settings(value, path)
        else:
            yield path


def check_reasons(config):
    """User order (SPEC section 4, 23:05Z): set nothing without a reason, and
    never hard-code a length scale. Every model setting study.json makes (the
    settings that differ from the fitting library's defaults) carries a
    one-line reason in study.json "reasons", keyed by its dotted path; a
    reason on a parent key covers its children. A reason that names no
    setting is refused too, so the block cannot go stale."""
    if re.search(r"length[_ ]?scale|kappa|κ", json.dumps(config["models"], ensure_ascii=False), re.IGNORECASE):
        raise ValueError("no length scale or kappa may be set in study.json (SPEC section 4)")
    reasons = config.get("reasons", {})
    settings = [path for kind, block in config["models"].items() for path in leaf_settings(block, f"models.{kind}")]
    # The shipped arm is gnomon calibrate pinned at a version: its pin needs a reason too,
    # and so does the number of PCs every model sees.
    settings += list(leaf_settings(config.get("shipped", {}), "shipped"))
    settings += list(leaf_settings(config.get("fit_gate", {}), "fit_gate"))
    if "cohort" in config:
        settings.append("cohort.num_pcs")
    if "data" in config:
        settings.append("data.maximum_bytes_billed")  # the one BigQuery cap (SPEC section 7a)
    # Each fit's thread budget is measured, not guessed (sp-threads), so it needs its reason too.
    settings += list(leaf_settings(config.get("compute", {}).get("threads", {}), "compute.threads"))
    if "memory_headroom_fraction" in config.get("compute", {}):
        settings.append("compute.memory_headroom_fraction")
    for path in settings:
        parts = path.split(".")
        covering = [".".join(parts[:end]) for end in range(len(parts), 1, -1)]
        if not any(isinstance(reasons.get(key), str) and reasons[key].strip() for key in covering):
            raise ValueError(f"study.json sets {path} without a reason in its reasons block")
    stale = [key for key in reasons if not any(path == key or path.startswith(key + ".") for path in settings)]
    if stale:
        raise ValueError(f"study.json reasons name no setting: {sorted(stale)}")


# SPEC section 8 (23:18Z, 23:35Z): the prespecified non-inferiority margins,
# against the simulator's true probabilities, frozen with the config.
CLAIM_MARGINS = ("rmse_true", "oe_true", "cal_slope_true", "auc_true", "slope_recovery")


def check_claims(config, claim_run=False):
    """The claim rule is prespecified and frozen with the config: every margin
    present (a positive delta, its scale, an optional positive floor, a reason);
    the two-start convergence rule (R1), the same threshold the pipeline's
    convergence gate applies; runs at production n. A claim run also needs its
    replicate counts (R4): every planned (scenario, metric) cell sized from the
    paired dev-seed spread of converged fits, R >= max(minimum, ceil((2 t sd/delta)^2)),
    with the job that measured it. There is no fallback count."""
    claims = config.get("claims") or {}
    margins = claims.get("margins") or {}
    replicates = claims.get("replicates") or {}
    if set(margins) != set(CLAIM_MARGINS) or not claims.get("rule") or claims.get("run_n") != "production":
        raise ValueError(f"study.json claims needs its rule, run_n production and exactly the margins "
                         f"{list(CLAIM_MARGINS)}")
    for name, margin in margins.items():
        if (not isinstance(margin.get("delta"), (int, float)) or not margin["delta"] > 0
                or margin.get("scale") not in ("relative", "absolute") or not str(margin.get("reason", "")).strip()
                or ("floor" in margin and not (isinstance(margin["floor"], (int, float)) and margin["floor"] > 0))):
            raise ValueError(f"claims margin {name} needs a positive delta, a relative or absolute scale, "
                             "a positive floor if any, and a reason")
    rule = claims.get("convergence") or {}
    if (rule.get("starts") != 2 or rule.get("max_dRisk_over_SD") != config["convergence"]["max_delta_sd"]
            or rule.get("on_fail") != "inconclusive_not_converged" or not str(rule.get("reason", "")).strip()):
        raise ValueError("claims.convergence needs 2 starts, the convergence gate's max_delta_sd, "
                         "on_fail inconclusive_not_converged and a reason (SPEC section 8, R1)")
    minimum, t = replicates.get("minimum", 0), replicates.get("t", 0)
    if not isinstance(minimum, int) or minimum < 5 or not isinstance(t, (int, float)) or not t > 0:
        raise ValueError("claims replicates need a minimum of at least 5 and a positive t")
    if not claim_run:
        return
    cells = [f"{scenario}.{metric}" for scenario in claims.get("scenarios") or [] for metric in CLAIM_MARGINS]
    if not cells:
        raise ValueError("a claim run needs its planned scenarios in study.json claims.scenarios")
    sized = replicates.get("by_scenario_metric") or {}
    for cell in cells:
        entry = sized.get(cell)
        if not entry:
            raise ValueError(f"claim cell {cell} has no replicate count sized from dev-seed spreads (R4)")
        sd, delta = entry.get("sd_dev"), entry.get("delta")
        if not (isinstance(sd, (int, float)) and sd >= 0 and isinstance(delta, (int, float)) and delta > 0):
            raise ValueError(f"claim cell {cell} needs its dev-seed paired spread sd_dev and the margin delta used")
        needed = max(minimum, math.ceil((2 * t * sd / delta) ** 2))
        if (entry.get("converged_starts") is not True or not str(entry.get("source_job", "")).strip()
                or not isinstance(entry.get("R"), int) or entry["R"] < needed):
            raise ValueError(f"claim cell {cell} needs R >= {needed} from converged dev-seed fits and its source job")


def frame_options(config, build_frames):
    """build_frames' censoring argument for the config's primary rule. A
    build_frames without the argument censors at ehr_end, so any other rule is
    refused rather than silently ignored."""
    censor = CENSORING[config["primary_censoring_rule"]]
    if "censor" in inspect.signature(build_frames).parameters:
        return {"censor": censor}
    if censor != "ehr_end":
        raise ValueError("this phenotypes.build_frames cannot censor other than at ehr_end")
    return {}


def check_frozen(config):
    """SPEC section 8 (S8): the AoU outer test is evaluated only under the config
    whose hash was frozen in study.json before the run."""
    frozen, actual = config.get("frozen_config_sha256"), config_hash(config)
    if frozen != actual:
        raise ValueError(f"study.json frozen_config_sha256 is {frozen}, the config hashes to {actual}: "
                         "freeze the final config before evaluating the AoU outer test")


def code_identity():
    files = [HERE / "study.py", HERE / "aou_checkpoint.py", *sorted((HERE / "study").rglob("*.py"))]
    return {path.relative_to(HERE).as_posix(): file_hash(path) for path in files}


def engine_identity():
    """The installed gamfit wheel: its version, its native library bytes and
    the gam commit they were built from. Any other build is another signature,
    so a checkpoint's fits are never reloaded under a different engine."""
    try:
        distribution = importlib.metadata.distribution("gamfit")
    except importlib.metadata.PackageNotFoundError:
        return {"gamfit": None}
    native = sorted(str(f) for f in distribution.files or () if str(f).endswith((".so", ".pyd")))
    hashes = {name: file_hash(distribution.locate_file(name)) for name in native}
    return {"gamfit": distribution.version, "native": hashes,
            "gam_commit": built_from(hashes, Path(sys.prefix) / "PROVENANCE.json")}


def built_from(hashes, record):
    """The gam commit in the venv's build record (study-wheel's PROVENANCE.json;
    gamfit.build_info() carries none), believed only when the record's engine
    bytes are the installed extension's."""
    record = Path(record)
    if not record.is_file():
        raise RuntimeError(f"{record} is missing: a study venv carries the build record of its gamfit")
    provenance = read_json(record)
    extension = provenance["extension"]
    if hashes.get(extension["member"]) != extension["engine_sha256"]:
        raise RuntimeError(f"{record} records another gamfit build than the one installed")
    return provenance["gam_commit"]


def input_identity(path):
    """Content identity of one input; a directory is every file in it. Files
    over 2 GiB are identified by size plus their first and last 16 MiB."""
    path = Path(path)
    if path.is_dir():
        return {p.relative_to(path).as_posix(): input_identity(p) for p in sorted(path.rglob("*")) if p.is_file()}
    size = path.stat().st_size
    if size <= 2 * 1024**3:
        return file_hash(path)
    sampled = hashlib.sha256()
    with path.open("rb") as handle:
        sampled.update(handle.read(16 * 1024**2))
        handle.seek(size - 16 * 1024**2)
        sampled.update(handle.read())
    return {"size": size, "sampled_sha256": sampled.hexdigest()}


class Status:
    """Fixed public labels beside the checkpoint in AoU; a log line elsewhere."""
    def __init__(self, uri):
        self.uri = uri

    def __call__(self, label):
        print(f"study_status {label} {time.strftime('%H:%M:%S', time.gmtime())}Z", flush=True)
        if self.uri:
            from aou_status import publish_status
            try:
                publish_status(self.uri, label)
            except Exception as error:  # a status label must never end the run
                print(f"study_status_publish_failed {type(error).__name__}", flush=True)


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #
class Study:
    def __init__(self, args):
        self.args = args
        self.config, self.diseases = load_config(args.config, args.source)
        self.inputs = {}
        for item in args.input or []:
            name, _, value = item.partition("=")
            if not name or not value or name in self.inputs:
                raise ValueError(f"inputs are unique NAME=PATH pairs, not {item!r}")
            self.inputs[name] = Path(value).resolve()
        self.parquet = self.config["data"]["source"] == "parquet"
        # A validation run may scope itself to some kinds and diseases (lead, 09-19);
        # an AoU run is always the whole study.
        if (args.kinds or args.diseases) and not self.parquet:
            raise ValueError("--kinds and --diseases scope a simulator validation run; an AoU run is the whole study")
        if not self.parquet:
            check_frozen(self.config)
        if args.claim:
            # A claim is made on the simulator, from sized replicates (SPEC section 8, R4).
            if not self.parquet or self.config["label"] != "production":
                raise ValueError("a claim run is a simulator run of the production config")
            check_claims(self.config, claim_run=True)
        self.kinds = tuple(kind for kind in KINDS if kind in (args.kinds or KINDS))
        if args.diseases:
            unknown = set(args.diseases) - {disease.slug for disease in self.diseases}
            if unknown:
                raise ValueError(f"--diseases names no study disease: {sorted(unknown)}")
            self.diseases = [disease for disease in self.diseases if disease.slug in args.diseases]
        from study import models
        self.models = models
        check_single_sex([definition(disease) for disease in self.diseases],
                         {kind: [("shared", c) for c in self.shared_components(kind)]
                          + [(v, c) for v in self.variants(kind) for c in self.own_components(kind, v)]
                          for kind in self.kinds},
                         lambda kind, variant, component, disease: models.covariates(
                             kind, variant, component, self.settings(kind), disease))
        digest.check_caveats(args.caveat)
        self.run_kind = "claim" if args.claim else "aou" if not self.parquet else "dev"
        self.work = Path(args.work).resolve()
        self.root = self.work / "steps"
        self.status = Status(args.status_uri)
        self.started = time.time()
        # Cost is charged on the CPUs the task holds (a VM's vCPUs, a Slurm job's cores),
        # whatever share of them the pool uses; os.cpu_count() is the whole host.
        self.vcpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
        self.threads = args.threads or self.config["compute"].get("total_threads") or self.vcpus
        # The memory the pool may fill: the task's (its cgroup limit or the host's;
        # --memory-gb on a shared node, whose host memory is not this run's), less headroom.
        held = args.memory_gb * 2**30 if args.memory_gb else task_memory_bytes()
        self.memory = int(held * (1 - self.config["compute"]["memory_headroom_fraction"]))
        self.pool_stats = {}
        self.signature = {
            "study": study_identity(self.config),
            "deployment": deployment_identity(self.config),
            "code": code_identity(),
            "engine": engine_identity(),
            "scope": {"kinds": list(self.kinds), "diseases": [disease.slug for disease in self.diseases]},
            "inputs": {name: input_identity(path) for name, path in sorted(self.inputs.items())},
        }
        store = self.store(args.checkpoint, self.work / "store")
        self.checkpoint = Checkpoint(self.root, store, self.signature,
                                     min_interval=self.config["compute"].get("checkpoint_interval_seconds", 20))
        # Outer-test looks (SPEC section 8, S8): one marker per checkpoint that
        # reached evaluation, counted over every run of this config.
        self.looks = self.store(args.looks, self.work / "looks")
        self.look_marker = (args.checkpoint or "local").rstrip("/").rsplit("/", 1)[-1] + ".txt"
        self.digest_store = self.store(args.digest_uri, None) if args.digest_uri else None
        self.timings = {}
        self.source_handle = None

    def store(self, uri, local):
        if uri and uri.startswith("gs://"):
            from aou_identity import task_account
            return GcsStore(uri, self.config["data"]["google_project"], task_account())
        return LocalStore(uri or local)

    # ------------------------------------------------------------ helpers
    def path(self, step):
        return self.checkpoint.path(step)

    def pgs_ids(self):
        return sorted({disease.pgs for disease in self.diseases})

    def settings(self, kind):
        return self.config["models"].get(kind, {})

    def disease(self, slug):
        return next(disease for disease in self.diseases if disease.slug == slug)

    def variants(self, kind):
        """study.json's methods, every one of which the kind's model module must
        offer: a configured method is never silently dropped."""
        offered = self.models.VARIANTS[kind]
        missing = [variant for variant in self.config["variants"] if variant not in offered]
        if missing:
            raise ValueError(f"study/models/{kind}.py does not offer the configured variants {missing}")
        return list(self.config["variants"])

    def own_components(self, kind, variant):
        return list(self.models.components(kind, variant, self.settings(kind)))

    def shared_components(self, kind):
        shared = getattr(self.models, "shared_components", None)
        return list(shared(kind, self.settings(kind))) if shared else []

    def fits(self, slug, kind, variant=None):
        """"pooled" plus one LOGO refit per reportable held-out group."""
        plan = read_json(self.path(f"features/{slug}") / "plan.json")[kind]
        logo_variants = self.config["logo"]["variants"]
        fits = ["pooled"]
        if variant is None or variant in logo_variants:
            fits += [f"logo:{axis}:{group}" for axis in self.config["logo"]["axes"]
                     for group in plan["logo"].get(axis, [])]
        return fits

    def fit_step(self, slug, kind, variant, fit, component):
        return f"fits/{slug}/{kind}/{variant}/{fit_slug(fit)}/{component}"

    def model_dirs(self, slug, kind, variant, fit):
        """Every component a variant's prediction needs: its own and the shared
        ones. A convergence restart pairs its own components with the pooled
        shared fits, so the check isolates the variant's own solution."""
        dirs = {c: self.fit_step(slug, kind, variant, fit, c) for c in self.own_components(kind, variant)}
        shared = "pooled" if fit == RESTART else fit
        dirs.update({c: self.fit_step(slug, kind, "shared", shared, c) for c in self.shared_components(kind)})
        return dirs

    def certified(self, slug, kind, variant, fit):
        """certification() of the component fits one (variant, fit) model predicts with."""
        paths = [self.path(step) / "fit.json" for step in self.model_dirs(slug, kind, variant, fit).values()]
        return certification([read_json(path) for path in paths if path.is_file()])

    def checked(self, kind):
        """The variants whose pooled fits get a second-start convergence check (SPEC section 8)."""
        return [v for v in self.config["convergence"]["variants"] if v in self.variants(kind)]

    def job(self, step, spec, threads, priority=0.0, deps=()):
        """A pool job for `step`, whose directory must already be begun. Its log
        lands in the step itself, so a job leaves no files outside its step."""
        spec = dict(spec, step=step, root=str(self.root), config=str(Path(self.args.config).resolve()),
                    source=self.args.source, horizons=self.horizons(),
                    **({"disease_definition": definition(self.disease(spec["disease"]))} if "disease" in spec else {}))
        # Jobs of one stage, kind, variant and component peak alike, in proportion to their frame.
        memory_class = "/".join(str(spec.get(key, "-")) for key in ("type", "kind", "variant", "component"))
        size = (read_json(self.path(f"features/{spec['disease']}") / "plan.json")[spec["kind"]]["rows"]
                if "disease" in spec and "kind" in spec else 0)
        return Job(key=step, spec=spec, threads=threads, log=self.path(step) / "job.log", deps=tuple(deps),
                   priority=priority, timeout=self.config["compute"]["job_timeout_seconds"],
                   affinity=spec.get("frame", ""), memory_class=memory_class, size=size)

    def seal_job(self, job, outcome, record_name):
        """Seal a pool job's step as a result, successful or not, with its log tail."""
        directory = self.path(job.key)
        log = Path(job.log).read_bytes()
        if len(log) > LOG_TAIL:
            Path(job.log).write_bytes(log[-LOG_TAIL:])
        record = directory / record_name
        written = read_json(record) if record.is_file() else {}
        if outcome.status == "ok" and written.get("status") != "ok":
            raise ValueError(f"{job.key} exited 0 without its {record_name}")
        if outcome.status != "ok":
            # A failed step keeps only its log and its record: never a partial model.
            for stale in directory.iterdir():
                if stale.name != "job.log":
                    shutil.rmtree(stale) if stale.is_dir() else stale.unlink()
            written = {"status": outcome.status, "category": failure_category(log)}
            if "invalid_output" in outcome.extra:
                written["category"] = "invalid_output"
            issue = declared_refusal(self.config, job.spec, log)
            if issue:
                written["declared_refusal"] = issue
        written.update(wall_seconds=round(outcome.seconds, 3), cpu_seconds=round(outcome.cpu_seconds, 3),
                       max_rss_mb=round(outcome.max_rss_mb, 1), threads=outcome.threads,
                       restarted_after_signal=outcome.restarted, over_budget=bool(outcome.extra.get("over_budget")))
        write_json(record, written)
        self.checkpoint.complete(job.key, info={"status": written["status"]})

    def pool(self, jobs, stage, record_name):
        if not jobs:
            return {}
        marks = {max(1, len(jobs) * q // 4): q * 25 for q in (1, 2, 3)}

        def progress(done, total):
            if stage == "fits" and done in marks:
                self.status(f"study_fits_{marks[done]}")
        print(f"study_pool {stage} jobs={len(jobs)} threads={self.threads} memory_gb={self.memory / 2**30:.1f}",
              flush=True)
        stats = {}
        outcomes = run_jobs(jobs, self.threads, lambda job, outcome: self.seal_job(job, outcome, record_name),
                            lambda threads: [sys.executable, str(HERE / "study.py"), "worker"], progress=progress,
                            memory=self.memory, stats=stats)
        self.pool_stats[stage] = stats
        print(f"study_pool {stage} max_held_gb={stats['max_bytes'] / 2**30:.2f} memory_waits={stats['memory_waits']}",
              flush=True)
        return outcomes

    def horizons(self):
        """The survival horizons the prespecified outcome-blind rule chose in this run."""
        return read_json(self.path("features/base") / "base.json")["horizons"]

    # ------------------------------------------------------------- stages
    def run(self):
        self.status("study_started")
        if self.checkpoint.restored_batches:
            self.status("study_resumed")
        self.attempt = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(self.started)) + f"-{os.getpid()}"
        finished = False
        try:
            for stage in STAGES:
                self.status(f"study_{stage}_started")
                started = time.monotonic()
                try:
                    getattr(self, "stage_" + stage)()
                    self.checkpoint.sync()
                except BaseException:
                    self.status(f"failed_study_{stage}")
                    raise
                self.timings[stage] = time.monotonic() - started
                print(f"study_stage {stage} wall_seconds={self.timings[stage]:.1f}", flush=True)
                self.record_attempt()
                self.status(f"study_{stage}_complete")
                if stage == self.args.stop_after:
                    break
            else:
                finished = True
        finally:
            write_json(self.work / "timings.json", self.timings)
            if finished and isinstance(self.checkpoint.store, GcsStore) and not self.args.keep_checkpoint:
                # The tokens are out; no participant-level intermediate stays in the bucket (SPEC 7a).
                self.checkpoint.delete_store()
            else:
                self.checkpoint.close()
        # A failure no declared refusal owns fails the run, after its digest (no masking).
        failures = unexpected_failures(self.step_records()) if finished else {}
        if failures:
            counts = sorted(collections.Counter(failures.values()).items())
            print(f"study_unexpected_errors n={len(failures)} " + " ".join(f"{c}={n}" for c, n in counts), flush=True)
            self.status("failed_study_unexpected_errors")
            raise SystemExit(3)
        self.status("study_completed")

    def step_records(self):
        """Every sealed fit, predict and evaluate record of this run, by step."""
        return {path.parent.relative_to(self.root).as_posix(): read_json(path)
                for stage, name in (("fits", "fit.json"), ("predict", "predict.json"), ("evaluate", "evaluate.json"))
                for path in sorted((self.root / stage).glob(f"**/{name}"))}

    def record_attempt(self):
        """This attempt's vCPUs and wall so far, kept across preemptions so the
        digest can charge every attempt (a preempted one up to its last stage)."""
        step = f"attempts/{self.attempt}"
        directory = self.checkpoint.begin(step)
        write_json(directory / "attempt.json", {"vcpus": self.vcpus, "stages": list(self.timings),
                                                "wall_seconds": round(time.time() - self.started, 1)})
        self.checkpoint.complete(step)

    def attempts(self):
        """Every attempt's record, this one's measured now."""
        directory = self.root / "attempts"
        records = {path.name: read_json(path / "attempt.json") for path in sorted(directory.iterdir())
                   if (path / "attempt.json").is_file()} if directory.is_dir() else {}
        records[self.attempt] = {"vcpus": self.vcpus, "wall_seconds": round(time.time() - self.started, 1)}
        return records

    def stage_scores(self):
        """Every study score in the cache; in AoU, the uncached ones are scored here."""
        if self.checkpoint.done("scores"):
            return
        directory = self.checkpoint.begin("scores")
        if self.parquet:
            manifest = read_json(self.inputs["tables"] / "manifest.json")
            absent = sorted(set(self.pgs_ids()) - set(manifest["scores"]))
            if absent:
                raise ValueError(f"the simulator tables lack scores {absent}")
            record = {pgs: "cached" for pgs in self.pgs_ids()}
        else:
            from study import cohort
            cached = set()
            for name, opener in cohort._sscore_members(self.inputs["score_cache"]):
                with opener() as handle:
                    _, header = cohort._sscore_header(handle)
                cached |= {pgs for pgs in self.pgs_ids() if f"{pgs}_AVG" in header}
            record = {pgs: "cached" if pgs in cached else "scored" for pgs in self.pgs_ids()}
            for pgs in sorted(set(self.pgs_ids()) - cached):
                self.score(pgs)
        write_json(directory / "scores.json", record)
        self.checkpoint.complete("scores")

    def score(self, pgs):
        """Score one uncached PGS on every array sample with the pinned gnomon
        scorer and its staged Catalog scoring file, then write it in the cache's
        own form ({pgs}_AVG, {pgs}_MISSING_PCT) for the cohort export."""
        from study import cohort
        step = f"scores/{pgs}"
        if self.checkpoint.done(step):
            return
        directory = self.checkpoint.begin(step)
        raw = self.work / "scoring" / pgs
        shutil.rmtree(raw, ignore_errors=True)
        raw.mkdir(parents=True)
        command = [str(self.inputs["scorer"]), str(self.inputs[f"weights_{pgs}"]),
                   self.config["data"]["genotype_prefix"] + ".bed", "--out", str(raw / pgs)]
        with (directory / "score.log").open("wb") as log:
            started = time.monotonic()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=raw,
                           timeout=self.config["compute"]["scoring_timeout_seconds"],
                           env=dict(os.environ, RAYON_NUM_THREADS=str(self.threads)))
        produced = sorted(raw.glob("*.sscore"))
        if len(produced) != 1:
            raise ValueError(f"the scorer wrote {len(produced)} score files for {pgs}")
        with produced[0].open("rb") as handle:
            skipped, header = cohort._sscore_header(handle)
        iid = next(field for field in header[:2] if field.lstrip("#") == "IID")
        averages = [f for f in header if f.endswith("_AVG")]
        missing = [f for f in header if f.endswith("_MISSING_PCT")]
        if len(averages) != 1 or len(missing) != 1:
            raise ValueError(f"the scorer's output for {pgs} is not one score")
        frame = pd.read_csv(produced[0], sep="\t", skiprows=skipped, usecols=[iid, averages[0], missing[0]],
                            dtype={iid: str})
        frame.columns = ["#IID", f"{pgs}_AVG", f"{pgs}_MISSING_PCT"]
        frame.to_csv(directory / f"{pgs}.sscore", sep="\t", index=False, float_format="%.17g")
        shutil.rmtree(raw)
        self.checkpoint.complete(step, info={"seconds": round(time.monotonic() - started, 1)})

    def source(self):
        if self.source_handle is None:
            from study import cohort
            tables = self.inputs["tables"] if self.parquet else self.path("cohort") / "tables"
            self.source_handle = cohort.ParquetSource(tables)
        return self.source_handle

    def stage_cohort(self):
        """The SCHEMA.md tables: the simulator's as given, or exported from AoU."""
        if not self.checkpoint.done("cohort"):
            directory = self.checkpoint.begin("cohort")
            if not self.parquet:
                self.export_tables(directory / "tables")
            manifest = dict(self.source().manifest)
            # One hash names the exact tables (every table's own sha256) this run read.
            manifest["tables_sha256"] = hashlib.sha256(
                json.dumps(manifest.pop("tables", {}), sort_keys=True).encode()).hexdigest()
            write_json(directory / "manifest.json", manifest)
            self.checkpoint.complete("cohort")
        self.source()

    def export_tables(self, directory):
        from google.auth.compute_engine import Credentials
        from google.cloud import bigquery
        from aou_identity import task_account
        from study import cohort, phenotypes
        data = self.config["data"]
        # Every query is dry-run against the remaining byte budget before it bills (SPEC 7a).
        client = cohort.BoundedClient(
            bigquery.Client(project=data["google_project"], credentials=Credentials(service_account_email=task_account())),
            data["maximum_bytes_billed"], timeout_seconds=data["query_timeout_seconds"])
        codes = phenotypes.phenotype_codes(self.diseases)
        # The staged cache plus the scores computed in the scores stage, one directory.
        cache = self.work / "score_cache"
        shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir()
        fresh = [self.path(f"scores/{pgs}") / f"{pgs}.sscore" for pgs in self.pgs_ids()
                 if (self.path(f"scores/{pgs}") / f"{pgs}.sscore").is_file()]
        for source in [*sorted(Path(self.inputs["score_cache"]).glob("*.sscore")), *fresh]:
            (cache / source.name).symlink_to(source.resolve())
        cohort.AouSource(client, data["workspace_cdr"], snomed_codes=codes["snomed_codes"],
                         excluded_branches=codes["excluded_branches"], scores=self.pgs_ids(),
                         ancestry=self.inputs["ancestry"], prune=self.inputs["prune"],
                         projection=self.inputs["projection"], score_cache=cache).export(directory)

    def stage_features(self):
        from study import phenotypes
        pending = [d for d in self.diseases if not self.checkpoint.done(f"features/{d.slug}")]
        if not pending and self.checkpoint.done("features/base"):
            return
        config = phenotypes.CohortConfig.from_json(self.config["cohort"])
        base, frames = phenotypes.build_frames(self.source(), self.diseases, config,
                                               **frame_options(self.config, phenotypes.build_frames))
        sites = self.site_labels(base.frame.ehr_site)
        if not self.checkpoint.done("features/base"):
            directory = self.checkpoint.begin("features/base")
            followup = phenotypes.followup_distribution(base, config)
            # The horizons come from the prespecified outcome-blind rule, in this
            # same run and before any outcome is read (SPEC section 3, 7a).
            write_json(directory / "base.json", {"flow": base.flow, "cuts": base.cuts, "followup": followup,
                                                 "horizons": list(phenotypes.choose_horizons(followup, config)),
                                                 "sites": len([s for s in sites.values() if s.startswith("site")]),
                                                 # The PC scale the models see (study-sim: confirms the geometry).
                                                 "pc_sd": {f"PC{i}": float(base.frame[f"PC{i}"].std(ddof=1))
                                                           for i in range(1, config.num_pcs + 1)}})
            self.checkpoint.complete("features/base")
        truth_path = self.inputs.get("tables", Path("/nonexistent")) / "truth.parquet"
        truth = pd.read_parquet(truth_path) if self.parquet and truth_path.is_file() else None
        # The simulator's truth slopes are per its own z (score / z_sd); evaluate rescales them.
        z_scale = read_json(truth_path.parent / "manifest.json")["simulator"]["z_scale"] if truth is not None else None
        for disease in pending:
            directory = self.checkpoint.begin(f"features/{disease.slug}")
            plan = {}
            for kind in KINDS:
                frame = getattr(frames[disease.slug], kind).copy()
                frame["ehr_site"] = frame.ehr_site.astype(str).map(sites).astype("category")
                if frame.person_id.duplicated().any() or frame.empty:
                    raise ValueError(f"{disease.slug} {kind} frame is empty or repeats a person")
                codes = set(frame.y.unique() if kind == "binary" else frame.event.unique())
                if not codes <= OUTCOME_CODES[kind]:
                    raise ValueError(f"{disease.slug} {kind} frame has unknown outcome codes {sorted(codes)}")
                frame.to_parquet(directory / f"{kind}.parquet", index=False)
                plan[kind] = {"rows": len(frame), "test_rows": int(frame.test.sum()),
                              "logo": self.logo_groups(frame, kind)}
            write_json(directory / "flow.json", frames[disease.slug].flow)
            write_json(directory / "plan.json", plan)
            if truth is not None:
                truth.loc[truth.disease.astype(str).eq(disease.slug)].to_parquet(directory / "truth.parquet",
                                                                                 index=False)
                write_json(directory / "truth.json", {"z_sd": float(z_scale[disease.slug]["z_sd"])})
            self.checkpoint.complete(f"features/{disease.slug}")

    def site_labels(self, sites):
        """EHR sites by size rank, never by identifier: the largest keep a rank
        label (the LOGO groups), the rest pool into "other"."""
        sites = sites.astype(str)
        ranked = sites[sites.ne(UNKNOWN)].value_counts(sort=True)
        top = self.config["logo"]["max_sites"]
        labels = {site: f"site{rank + 1:02d}" if rank < top else "other" for rank, site in enumerate(ranked.index)}
        labels[UNKNOWN] = UNKNOWN
        return labels

    def logo_groups(self, frame, kind):
        """Held-out groups whose outer-test rows can carry a publishable cell:
        more than the small-cell maximum of cases and of non-cases (SPEC section
        8: ancestries, the 4 regions, and only the largest `logo.sites` sites)."""
        limit = self.config["report"]["small_cell_max"]
        case = (frame.y.eq(1) if kind == "binary" else frame.event.eq(1)).to_numpy()
        test = frame.test.to_numpy()
        largest = {f"site{rank:02d}" for rank in range(1, self.config["logo"]["sites"] + 1)}
        groups = {}
        for axis in self.config["logo"]["axes"]:
            values = frame[axis].astype(str).to_numpy()
            chosen = []
            for group in sorted(set(values) - {UNKNOWN, "other"}):
                if axis == "ehr_site" and group not in largest:
                    continue
                inside = values == group
                if (min(int((inside & test & case).sum()), int((inside & test & ~case).sum())) > limit
                        and (~inside & ~test).any()):
                    chosen.append(group)
            groups[axis] = chosen
        return groups

    def budget(self, stage, kind, variant):
        """A job's thread budget from study.json compute.threads[stage][kind]: the
        variant's own entry ("shared" for the shared competing fits), else
        "default". The speed lanes tune these without code changes."""
        table = self.config["compute"]["threads"][stage][kind]
        return int(table.get(variant, table["default"]))

    def stage_fits(self):
        jobs, scheduled = [], set()
        for disease in self.diseases:
            slug = disease.slug
            for kind in self.kinds:
                rows = read_json(self.path(f"features/{slug}") / "plan.json")[kind]["rows"]
                frame = None
                plan = [("shared", c) for c in self.shared_components(kind)]
                plan += [(v, c) for v in self.variants(kind) for c in self.own_components(kind, v)]
                for variant, component in plan:
                    threads = self.budget("fit", kind, variant)
                    for fit in self.fits(slug, kind, None if variant == "shared" else variant):
                        step = self.fit_step(slug, kind, variant, fit, component)
                        if self.checkpoint.done(step):
                            continue
                        if frame is None:
                            frame = pd.read_parquet(self.path(f"features/{slug}") / f"{kind}.parquet")
                        if not self.enough_events(frame, kind, variant, fit, component, step):
                            continue
                        pooled = self.fit_step(slug, kind, variant, "pooled", component)
                        reuse = fit != "pooled" and self.config["logo"].get("reuse_pooled", False)
                        self.checkpoint.begin(step)
                        scheduled.add(step)
                        # Longest first: the budget stands for the fit's cost (marginal
                        # slope over location-scale and competitors), then survival,
                        # pooled over LOGO, and more rows.
                        priority = rows * threads * (2 if kind == "survival" else 1) * (2 if fit == "pooled" else 1)
                        jobs.append(self.job(step, {
                            "type": "fit", "disease": slug, "kind": kind, "variant": variant, "fit": fit,
                            "component": component, "frame": f"features/{slug}/{kind}.parquet",
                            "reference": pooled if reuse else None,
                        }, threads, priority, deps=[pooled] if reuse and pooled in scheduled else ()))
                    if variant in self.checked(kind):
                        jobs += self.restart_jobs(slug, kind, variant, component, threads, rows, scheduled)
        self.pool(jobs, "fits", "fit.json")

    def enough_events(self, frame, kind, variant, fit, component, step):
        """SPEC section 8, minimum events: a fit is attempted only if its training
        rows carry fit_gate.min_events of what it models (binary: cases and
        non-cases alike; survival: its own cause). A shared competing cause with
        no events at all is a zero hazard, which the model stores without fitting.
        A fit below the bar is sealed as insufficient_events: a result, never a
        failure. The count stays in the workspace record."""
        minimum = self.config["fit_gate"]["min_events"]
        train = frame.loc[training_rows(frame, fit)]
        if kind == "binary":
            events = int(min((train.y == 1).sum(), (train.y == 0).sum()))
        else:
            if component not in SURVIVAL_CAUSES:
                raise ValueError(f"survival component {component!r} names no event code the gate can count")
            events = int((train.event == SURVIVAL_CAUSES[component]).sum())
            if variant == "shared" and events == 0:
                return True
        if events >= minimum:
            return True
        write_json(self.checkpoint.begin(step) / "fit.json",
                   {"status": INSUFFICIENT_EVENTS, "events": events, "min_events": minimum,
                    "wall_seconds": 0.0, "cpu_seconds": 0.0, "max_rss_mb": 0.0, "threads": 0})
        self.checkpoint.complete(step, info={"status": INSUFFICIENT_EVENTS})
        return False

    def restart_jobs(self, slug, kind, variant, component, threads, rows, scheduled):
        """The pooled fit again from a second start (SPEC section 8, convergence
        gate): a warm restart from its own solution ("warm", passed to the model
        as `reference`), or its training rows in a seeded permuted order
        ("permuted_rows") until gamfit's warm_start_from exists."""
        step = self.fit_step(slug, kind, variant, RESTART, component)
        if self.checkpoint.done(step):
            return []
        start = self.config["convergence"]["start"]
        pooled = self.fit_step(slug, kind, variant, "pooled", component)
        record = self.path(pooled) / "fit.json"
        if record.is_file() and read_json(record).get("status") == INSUFFICIENT_EVENTS:
            return []  # nothing was fitted, so there is nothing to restart
        self.checkpoint.begin(step)
        scheduled.add(step)
        return [self.job(step, {
            "type": "fit", "disease": slug, "kind": kind, "variant": variant, "fit": "pooled",
            "component": component, "frame": f"features/{slug}/{kind}.parquet",
            "reference": pooled if start == "warm" else None,
            "restart": {"start": start, "seed": self.config["cohort"]["seed"]},
        }, threads, rows * threads * (2 if kind == "survival" else 1) * 2,
            deps=[pooled] if start == "warm" and pooled in scheduled else ())]

    def fit_ok(self, slug, kind, variant, fit):
        records = [self.path(step) / "fit.json" for step in self.model_dirs(slug, kind, variant, fit).values()]
        return all(record.is_file() and read_json(record)["status"] == "ok" for record in records)

    def restarted(self, slug, kind, variant):
        """[RESTART] when this variant's pooled fit has a second start (none
        when the pooled fit itself was skipped for insufficient events)."""
        if variant not in self.checked(kind):
            return []
        steps = [self.fit_step(slug, kind, variant, RESTART, c) for c in self.own_components(kind, variant)]
        return [RESTART] if all((self.path(s) / "fit.json").is_file() for s in steps) else []

    def predict_step(self, slug, kind, variant, fit):
        return f"predict/{slug}/{kind}/{variant}/{fit_slug(fit)}"

    def stage_predict(self):
        """One job per fitted model: it predicts that model's rows once, every
        horizon in one call, and evaluation reuses the saved arrays (a survival
        marginal-slope prediction is the costliest step per row)."""
        jobs = []
        for disease in self.diseases:
            for kind in self.kinds:
                rows = read_json(self.path(f"features/{disease.slug}") / "plan.json")[kind]["test_rows"]
                for variant in self.variants(kind):
                    threads = self.budget("predict", kind, variant)
                    restart = self.restarted(disease.slug, kind, variant)
                    for fit in self.fits(disease.slug, kind, variant) + restart:
                        step = self.predict_step(disease.slug, kind, variant, fit)
                        if self.checkpoint.done(step) or not self.fit_ok(disease.slug, kind, variant, fit):
                            continue
                        self.checkpoint.begin(step)
                        jobs.append(self.job(step, {"type": "predict", "disease": disease.slug, "kind": kind,
                                                    "variant": variant, "fit": fit,
                                                    "models": self.model_dirs(disease.slug, kind, variant, fit),
                                                    "frame": f"features/{disease.slug}/{kind}.parquet"},
                                             threads, priority=threads * (rows if fit in ("pooled", RESTART) else rows / 4)))
        self.pool(jobs, "predict", "predict.json")

    def check_convergence(self):
        """The convergence gate (SPEC section 8): each checked pooled fit against
        its second start, as max|delta risk| / SD of the pooled risk over the
        outer-test rows, the worst horizon for survival. Above the threshold the
        variant's pooled cells are "not converged" (numerical error, not
        sampling noise). Reads saved predictions only; no outcome is read."""
        threshold = self.config["convergence"]["max_delta_sd"]
        for disease in self.diseases:
            for kind in self.kinds:
                for variant in self.checked(kind):
                    step = f"convergence/{disease.slug}/{kind}/{variant}"
                    if self.checkpoint.done(step):
                        continue
                    record = {"start": self.config["convergence"]["start"], "threshold": threshold}
                    steps = [self.path(self.predict_step(disease.slug, kind, variant, fit)) for fit in ("pooled", RESTART)]
                    outcomes = [read_json(s / "predict.json") if (s / "predict.json").is_file() else {"status": "missing"}
                                for s in steps]
                    pooled_fits = [self.path(s) / "fit.json" for s in self.model_dirs(disease.slug, kind, variant, "pooled").values()]
                    if any(f.is_file() and read_json(f)["status"] == INSUFFICIENT_EVENTS for f in pooled_fits):
                        record["status"] = INSUFFICIENT_EVENTS
                    elif any(o["status"] != "ok" for o in outcomes):
                        record["status"] = "unchecked"
                    else:
                        pooled, restart = (np.load(s / "predictions.npz") for s in steps)
                        if not np.array_equal(pooled["index"], restart["index"]):
                            raise ValueError(f"{disease.slug} {kind} {variant}: the restart predicted other rows")
                        record["max_delta_sd"] = delta_over_sd(pooled["risk"], restart["risk"])
                        record["certification"] = [self.certified(disease.slug, kind, variant, fit)
                                                   for fit in ("pooled", RESTART)]
                        # A start its engine did not certify is never counted as converged,
                        # however closely the two starts agree.
                        record["status"] = ("not_certified" if "not_certified" in record["certification"]
                                            else "converged" if record["max_delta_sd"] <= threshold
                                            else "not_converged")
                    write_json(self.checkpoint.begin(step) / "convergence.json", record)
                    self.checkpoint.complete(step, info={"status": record["status"]})

    def convergence(self, slug, kind, variant):
        return read_json(self.path(f"convergence/{slug}/{kind}/{variant}") / "convergence.json")

    def stage_evaluate(self):
        self.check_convergence()
        # This checkpoint's outer-test look, recorded before any outer-test outcome is read.
        if self.look_marker not in self.looks.names():
            self.looks.put_bytes(self.look_marker, (config_hash(self.config) + "\n").encode())
        jobs = []
        for disease in self.diseases:
            for kind in self.kinds:
                step = f"evaluate/{disease.slug}/{kind}"
                if self.checkpoint.done(step):
                    continue
                predictions = {}
                for variant in self.variants(kind):
                    for fit in self.fits(disease.slug, kind, variant):
                        step_ = self.predict_step(disease.slug, kind, variant, fit)
                        record = self.path(step_) / "predict.json"
                        if record.is_file() and read_json(record)["status"] == "ok":
                            predictions.setdefault(variant, {})[fit] = step_
                self.checkpoint.begin(step)
                jobs.append(self.job(step, {"type": "evaluate", "disease": disease.slug, "kind": kind,
                                            "predictions": predictions,
                                            "frame": f"features/{disease.slug}/{kind}.parquet"},
                                     self.budget("evaluate", kind, "default"),
                                     priority=2 if kind == "survival" else 1))
        self.pool(jobs, "evaluate", "evaluate.json")


    def stage_digest(self):
        """Result names (participant aggregates, suppressed and audited) and
        operation names (timings, fit outcomes, cost: no participant data).
        Always rebuilt: it is cheap and carries this attempt's timings. In AoU
        the names are written straight to the digest prefix, one empty object
        each; the step keeps them as one text file."""
        directory = self.checkpoint.begin("digest")
        rows = []
        for disease in self.diseases:
            flow = read_json(self.path(f"features/{disease.slug}") / "flow.json")
            # SPEC section 2: the single-record count is reported (they stay in the analysis).
            subgroups = {"binary": {"single_record": flow["binary"]["single_record"]},
                         "survival": {"single_record_at_risk": flow["survival"]["single_record_at_risk"],
                                      # Rows an exclusion match censors (study-audit: a dependent
                                      # censoring caveat where they pass 1% of the frame).
                                      "exclusion_exits": flow["survival"]["exclusion_exits"]}}
            rows += digest.flow_rows(disease.slug, {name: flow[name] for name in ("disease", "binary", "survival")},
                                     digest.LIMIT, subgroups)
            # Whole-cohort counts by ancestry beside their totals, so the audit partitions them.
            if "by_ancestry" in flow:
                rows += digest.cohort_rows(disease.slug, flow["by_ancestry"])
            for kind in self.kinds:
                record = read_json(self.path(f"evaluate/{disease.slug}/{kind}") / "evaluate.json")
                # Every pooled cell of a checked variant says whether its fit converged,
                # and every cell whether its engines certified the fits behind it.
                status = {v: self.convergence(disease.slug, kind, v)["status"] for v in self.checked(kind)}
                certified = {(v, fit): self.certified(disease.slug, kind, v, fit)
                             for v in self.variants(kind) for fit in self.fits(disease.slug, kind, v)}
                for row in record.get("rows", []):
                    if row.get("fit") == "pooled" and row.get("variant") in status:
                        row = {**row, "convergence": status[row["variant"]]}
                    if (row.get("variant"), row.get("fit")) in certified:
                        row = {**row, "certification": certified[row["variant"], row["fit"]]}
                    rows.append(row)
                rows += self.outcome_rows(disease.slug, kind)
        base = read_json(self.path("features/base") / "base.json")
        rows += digest.flow_rows("base", {"base": base["flow"]}, digest.LIMIT)
        rows += digest.followup_rows(base["followup"]["administrative"], digest.LIMIT)
        rows += digest.pc_scale_rows(base["pc_sd"])
        if "ehr" in base["followup"]:
            rows += digest.ehr_rows(base["followup"]["ehr"], digest.LIMIT)
        rows += digest.ehr_domain_rows(read_json(self.path("cohort") / "manifest.json"), digest.LIMIT)
        descendants = self.source().directory / "descendants.parquet"
        if descendants.is_file():
            rows += digest.descendant_rows(pd.read_parquet(descendants), digest.LIMIT)
        # Survival nests inside binary only for diseases without exclusion roots (SPEC C3).
        nested = {digest.slug(d.slug) for d in self.diseases if not d.exclusions}
        results, operations = digest.encode(rows, self.operation_rows(base), nested=nested)
        written = [*results, *operations]
        (directory / "tokens.txt").write_text("\n".join(written) + "\n")
        shutil.copyfile(directory / "tokens.txt", self.work / "tokens.txt")
        if self.digest_store is not None:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(16) as pool:
                list(pool.map(lambda name: self.digest_store.put_bytes(name, b""), written))
        print(f"study_digest result_names={len(results)} operation_names={len(operations)}", flush=True)
        self.checkpoint.complete("digest")

    def outcome_rows(self, slug, kind):
        """A result row for every fit or prediction that did not produce
        predictions (a fit timeout at the 900 s cap, an error, a signal): the
        table counts them per cell instead of leaving the variant silently out.
        Statuses only, so nothing here is a count."""
        rows = []
        for variant in self.variants(kind):
            for fit in self.fits(slug, kind, variant):
                dirs = self.model_dirs(slug, kind, variant, fit)
                failed = [read_json(self.path(step) / "fit.json")["status"] for step in dirs.values()
                          if read_json(self.path(step) / "fit.json")["status"] != "ok"]
                predict = self.path(self.predict_step(slug, kind, variant, fit)) / "predict.json"
                status = (INSUFFICIENT_EVENTS if INSUFFICIENT_EVENTS in failed else f"fit_{failed[0]}" if failed else
                          f"predict_{read_json(predict)['status']}" if predict.is_file()
                          and read_json(predict)["status"] != "ok" else None)
                if status:
                    rows.append({"disease": slug, "model": kind, "variant": variant, "fit": fit,
                                 "stratum": "overall", "horizon": None, "outcome": status})
        return rows

    def operation_rows(self, base):
        """Aggregates of how the run went: no participant data, so not audited."""
        attempts = self.attempts()
        manifest = read_json(self.path("cohort") / "manifest.json")
        study = {"scope": "study", "item": "run", "config_sha256_12": config_hash(self.config)[:12],
                 "label": self.config["label"], "run_kind": self.run_kind,
                 "gam_commit_12": "g" + str(self.signature["engine"].get("gam_commit") or "none")[:12],
                 "scope_kinds": "_".join(self.kinds), "scope_diseases": len(self.diseases),
                 "unexpected_errors": len(unexpected_failures(self.step_records())),
                 "declared_refusals": sum(bool(r.get("declared_refusal")) for r in self.step_records().values()),
                 **({"caveats": "_and_".join(self.args.caveat)} if self.args.caveat else {}),
                 "vcpus": self.vcpus, "threads": self.threads, "memory_budget_gb": round(self.memory / 2**30, 2),
                 "attempts": len(attempts),
                 "vcpu_hours": round(sum(a["vcpus"] * a["wall_seconds"] for a in attempts.values()) / 3600, 3),
                 "outer_test_looks": len([n for n in self.looks.names() if n.endswith(".txt")]),
                 "horizons": "_".join("h" + digest.token(float(h)) for h in base["horizons"]),
                 **manifest_fields(manifest)}
        rows = [study]
        # Which EHR domains moved ehr_end later: exact audited fractions when the
        # manifest carries their denominator (ehr_people, see stage_digest); without
        # it a share leaves only as a coarse bucket, which pins no count.
        extended = manifest.get("ehr_extended_by") or {}
        if extended and not manifest.get("ehr_people"):
            rows.append({"scope": "ehr_extended_by", "item": "domains",
                         **{digest.label(domain, 20): share_bucket(share) for domain, share in extended.items()}})
        for disease in self.diseases:
            for kind in self.kinds:
                for variant in self.checked(kind):
                    record = self.convergence(disease.slug, kind, variant)
                    rows.append({"scope": "convergence", "item": f"{disease.slug}.{kind}.{variant}",
                                 "status": record["status"], "start": record["start"],
                                 **({"max_delta_sd": record["max_delta_sd"]} if "max_delta_sd" in record else {})})
        rows += [{"scope": "timing", "item": stage, "wall_seconds": round(seconds, 1)}
                 for stage, seconds in self.timings.items()]
        # What the pool held at once against its memory budget, per stage (this attempt's).
        rows += [{"scope": "pool", "item": stage, "max_held_gb": round(stats["max_bytes"] / 2**30, 2),
                  "memory_waits": stats["memory_waits"]} for stage, stats in self.pool_stats.items()]
        for disease in self.diseases:
            for kind in self.kinds:
                plan = [("shared", c) for c in self.shared_components(kind)]
                plan += [(v, c) for v in self.variants(kind) for c in self.own_components(kind, v)]
                for variant, component in plan:
                    records = [read_json(self.path(self.fit_step(disease.slug, kind, variant, fit, component))
                                         / "fit.json")
                               for fit in self.fits(disease.slug, kind, None if variant == "shared" else variant)
                               + (self.restarted(disease.slug, kind, variant) if variant != "shared" else [])]
                    seconds = sorted(r.get("fit_seconds", r["wall_seconds"]) for r in records)
                    ok = [r for r in records if r["status"] == "ok"]
                    skipped = [r for r in records if r["status"] == INSUFFICIENT_EVENTS]
                    certificates = [certification([r]) for r in ok]
                    rows.append({"scope": "fits", "item": f"{disease.slug}.{kind}.{variant}.{component}",
                                 "fits": len(records), "ok": len(ok), "insufficient_events": len(skipped),
                                 "failed": len(records) - len(ok) - len(skipped),
                                 "not_certified": certificates.count("not_certified"),
                                 "no_certificate": certificates.count("no_certificate"),
                                 "median_seconds": seconds[len(seconds) // 2], "max_seconds": seconds[-1],
                                 "cpu_seconds": round(sum(r["cpu_seconds"] for r in records), 1),
                                 "threads": records[0]["threads"],
                                 "max_rss_mb": max(r["max_rss_mb"] for r in records),
                                 "over_budget": sum(bool(r.get("over_budget")) for r in records)})
                    categories = {}
                    for record in records:
                        if record["status"] not in ("ok", INSUFFICIENT_EVENTS):
                            label = f"{record['status']}_{record.get('category', 'unclassified')}"
                            categories[label] = categories.get(label, 0) + 1
                    if categories:
                        rows.append({"scope": "fit_failures", "item": f"{disease.slug}.{kind}.{variant}.{component}",
                                     **{digest.label(name): count for name, count in categories.items()}})
                for variant in self.variants(kind):
                    records = [read_json(self.path(self.predict_step(disease.slug, kind, variant, fit)) / "predict.json")
                               for fit in self.fits(disease.slug, kind, variant)
                               if (self.path(self.predict_step(disease.slug, kind, variant, fit)) / "predict.json").is_file()]
                    seconds = sorted(r.get("predict_seconds", r["wall_seconds"]) for r in records) or [0.0]
                    rows.append({"scope": "predict", "item": f"{disease.slug}.{kind}.{variant}",
                                 "ok": sum(r["status"] == "ok" for r in records),
                                 "failed": sum(r["status"] != "ok" for r in records),
                                 "max_seconds": seconds[-1], "cpu_seconds": round(sum(r["cpu_seconds"] for r in records), 1),
                                 "max_rss_mb": max((r["max_rss_mb"] for r in records), default=0.0),
                                 "over_budget": sum(bool(r.get("over_budget")) for r in records)})
                record = read_json(self.path(f"evaluate/{disease.slug}/{kind}") / "evaluate.json")
                rows.append({"scope": "evaluate", "item": f"{disease.slug}.{kind}", "status": record["status"],
                             "category": record.get("category", "none"), "wall_seconds": record["wall_seconds"],
                             "max_rss_mb": record["max_rss_mb"], "over_budget": int(bool(record.get("over_budget")))})
        return rows



def definition(disease):
    """A disease as the model modules see it: its slug and its sex restriction."""
    return {"slug": disease.slug, "sex": disease.sex}


def check_single_sex(definitions, plans, covariates):
    """The single-sex rule (lead, 09-19): sex enters no fit of a disease whose
    definition restricts it to one sex. `plans` is {kind: [(variant, component)]}
    and `covariates(kind, variant, component, definition)` a fit's design columns."""
    for disease in definitions:
        if disease["sex"] is None:
            continue
        for kind, plan in plans.items():
            for variant, component in plan:
                if "sex" in covariates(kind, variant, component, disease):
                    raise ValueError(f"{disease['slug']} is {disease['sex']}-only, but sex is in the design of "
                                     f"its {kind} {variant} {component} fit")


def certification(records):
    """Whether the engines certified every fitted component behind one model
    (its fit.json records): "certified", "not_certified" when any reports
    converged false (gam keeps some fits uncertified rather than refusing
    them), "no_certificate" when an engine reports none, "no_fit" when none
    fitted. A converged flag that is not a bool is refused, not guessed."""
    flags = []
    for record in records:
        if record["status"] != "ok":
            continue
        flag = (record.get("info") or {}).get("converged")
        if flag is not None and not isinstance(flag, bool):
            raise ValueError(f"a fit reports converged={flag!r}, not a bool")
        flags.append(flag)
    if not flags:
        return "no_fit"
    if False in flags:
        return "not_certified"
    return "certified" if all(flags) else "no_certificate"


def manifest_fields(manifest):
    """The run row's facts about its tables: where they came from, the BigQuery
    plan and bill, and the derived CDR cutoff. No participant data."""
    bigquery = manifest.get("bigquery") or {}
    fields = {"bigquery_bytes_billed": int(bigquery.get("bytes_billed", 0)),
              "tables_source": digest.label(manifest.get("source", "unknown")),
              "tables_sha256_12": "t" + str(manifest.get("tables_sha256", ""))[:12]}
    if manifest.get("seed") is not None:
        fields["tables_seed"] = int(manifest["seed"])
    if "simulator" in manifest:  # a simulated world names its scenario (study-sim's manifest)
        fields["tables_scenario"] = digest.label(manifest["simulator"]["scenario"])
    if "plan_bytes" in bigquery:
        # plan() dry-runs every query, {query name: bytes}; the budget gates the total.
        fields["bigquery_plan_bytes"] = int(sum(bigquery["plan_bytes"].values()))
    for key in ("cdr_cutoff", "cdr_cutoff_source"):
        if manifest.get(key):
            fields[key] = digest.label(manifest[key])
    if "ehr_domains" in manifest:
        fields["ehr_domains"] = "_".join(digest.label(d, 12) for d in manifest["ehr_domains"]) or "none"
    return fields


def share_bucket(share):
    """A share as a coarse bucket label, which pins no count; null is "unknown"."""
    if share is None:
        return "unknown"
    for bound, name in ((0.0, "zero"), (0.001, "under_0_1pct"), (0.01, "under_1pct"), (0.1, "under_10pct")):
        if share <= bound:
            return name
    return "over_10pct"


# --------------------------------------------------------------------------- #
# failure categories: fixed vocabulary, never data
# --------------------------------------------------------------------------- #
FAILURE_PHRASES = (
    ("no candidate seeds passed", "startup_seeds"),
    ("non-finite", "nonfinite"),
    ("failed to converge", "nonconvergence"),
    ("did not converge", "nonconvergence"),
    ("integration", "integration"),
    ("resource policy", "resource_policy"),
    ("refusing to densify", "resource_policy"),
    ("identifiab", "identifiability"),
    ("singular", "singular"),
    ("memory", "memory"),
    ("provenance", "provenance"),
    ("timed out", "timeout"),
)


def check_declared_refusals(config):
    """study.json declared_refusals: the only failures a run may end with and
    still succeed. Each names the kind and variant it applies to, a phrase of
    the engine's refusal message, the gam issue that owns it and a reason; any
    other failed step makes the run exit non-zero (no masking, SPEC section 5)."""
    for entry in config.get("declared_refusals", []):
        if (set(entry) != {"kind", "variant", "phrase", "issue", "reason"} or entry["kind"] not in KINDS
                or entry["variant"] not in config["variants"] or not str(entry["phrase"]).strip()
                or entry["phrase"] != entry["phrase"].lower() or not re.fullmatch(r"gam#\d+", str(entry["issue"]))
                or not str(entry["reason"]).strip()):
            raise ValueError("a declared refusal is {kind, variant, phrase (lower case), issue gam#N, reason}, "
                             f"for a configured kind and variant: {entry}")


def declared_refusal(config, spec, log):
    """The gam issue of the declared refusal this failed job's log shows, else None."""
    text = log[-65536:].decode("utf-8", errors="replace").lower()
    for entry in config.get("declared_refusals", []):
        if entry["kind"] == spec.get("kind") and entry["variant"] == spec.get("variant") and entry["phrase"] in text:
            return entry["issue"]
    return None


def unexpected_failures(records):
    """The failed steps a run may not pass over, as {step: category}: every
    record neither ok, nor below the events gate, nor a declared refusal."""
    return {step: record.get("category", "unclassified") for step, record in records.items()
            if record["status"] not in ("ok", INSUFFICIENT_EVENTS) and not record.get("declared_refusal")}


def failure_category(log):
    """The exception class a failed job raised, with the fixed category of its
    message when one applies: a code name, never data."""
    text = log[-65536:].decode("utf-8", errors="replace")
    found = re.findall(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt))\b(.*)$", text, re.MULTILINE)
    if not found:
        return "unclassified"
    exception, message = found[-1]
    label = re.sub(r"[^a-z0-9]+", "_", exception.rsplit(".", 1)[-1].lower()).strip("_")
    for phrase, category in FAILURE_PHRASES:
        if phrase in message.lower():
            return f"{label}_{category}"
    return label


# --------------------------------------------------------------------------- #
# pool jobs (each its own process)
# --------------------------------------------------------------------------- #
FRAMES = {}


def load_frame(path):
    """A sealed features frame, kept for the worker's next jobs (the pool gives a
    worker jobs on the frame it last read when it can). Jobs only read it;
    every subset they make is a new frame."""
    path = str(path)
    if path not in FRAMES:
        while len(FRAMES) >= 2:
            FRAMES.pop(next(iter(FRAMES)))
        FRAMES[path] = pd.read_parquet(path)
    return FRAMES[path]


def delta_over_sd(reference, other):
    """max |other - reference| / SD(reference) over rows, the worst horizon for
    a rows x horizons risk: the convergence gate's measure (SPEC section 8)."""
    a = np.asarray(reference, dtype=float).reshape(len(reference), -1)
    b = np.asarray(other, dtype=float).reshape(a.shape)
    sd = a.std(axis=0)
    delta = np.abs(b - a).max(axis=0)
    return float(np.where(sd > 0, delta / np.where(sd > 0, sd, 1.0), np.where(delta > 0, np.inf, 0.0)).max())

def held_out(frame, fit):
    """The rows of a LOGO fit's held-out group (none for the pooled fit)."""
    if fit == "pooled":
        return np.zeros(len(frame), dtype=bool)
    _, axis, group = fit.split(":", 2)
    return frame[axis].astype(str).to_numpy() == group


def training_rows(frame, fit):
    """A fit's training rows: development rows outside its held-out group."""
    return ~frame.test.to_numpy() & ~held_out(frame, fit)


def standardize(pgs):
    pgs = np.asarray(pgs, dtype=float)
    return {"mean": float(pgs.mean()), "sd": float(pgs.std(ddof=1)), "n": int(len(pgs))}


def to_pooled_z(slope, fit_sd, pooled_sd):
    """A slope per one z (score / fit_sd) as a slope per the pooled fit's z
    (score / pooled_sd): a LOGO fit and the simulator standardize differently."""
    return np.asarray(slope, dtype=float) * (pooled_sd / fit_sd)


def model_frame(rows, kind, standardization, *, predict):
    """Model inputs: z standardized on the fit's own training rows (affine, no
    CTN) in place of the raw score. A prediction frame carries no outcome."""
    drop = ["pgs", "test", *(OUTCOMES[kind] if predict else ("n_dates",))]
    data = rows.drop(columns=[c for c in drop if c in rows.columns]).reset_index(drop=True)
    data["z"] = (rows.pgs.to_numpy(dtype=float) - standardization["mean"]) / standardization["sd"]
    return data


def run_fit(spec, config, models):
    root = Path(spec["root"])
    out = root / spec["step"]
    kind, fit = spec["kind"], spec["fit"]
    frame = load_frame(root / spec["frame"])
    train = frame.loc[training_rows(frame, fit)]
    standardization = standardize(train.pgs)
    if not standardization["sd"] > 0:
        raise ValueError("the training score has no spread")
    reference = None
    if spec["reference"] is not None and read_json(root / spec["reference"] / "fit.json")["status"] == "ok":
        reference = root / spec["reference"]
    data = model_frame(train, kind, standardization, predict=False)
    restart = spec.get("restart")
    if restart and restart["start"] == "permuted_rows":
        # The same rows and standardization in another order: a second start
        # that needs nothing from the model (convergence gate, SPEC section 8).
        order = np.random.default_rng(restart["seed"]).permutation(len(data))
        data = data.iloc[order].reset_index(drop=True)
    print("study_fit_started", flush=True)
    started = time.perf_counter()
    info = models.fit(kind, spec["variant"], spec["component"], data, config["models"].get(kind, {}), out, reference,
                      disease=spec["disease_definition"])
    seconds = time.perf_counter() - started
    print("study_fit_saved", flush=True)
    # Provenance (SPEC section 8, LOGO): the person set this fit saw, checked at predict.
    write_json(out / "fit.json", {
        "status": "ok", "fit_seconds": round(seconds, 3), "standardization": standardization,
        "train_rows": int(len(train)), "train_sha256": person_set_hash(train.person_id),
        "held_out_in_train": int(held_out(train, fit).sum()), "warm_reference": reference is not None,
        "restart": restart["start"] if restart else None,
        "info": json.loads(json.dumps(info or {}, default=str))})


def verify_provenance(frame, fit, record):
    """Refuse a fit whose recorded person set or standardization is not its own
    training set: a planted pooled standardization in a LOGO fit fires here."""
    train = frame.loc[training_rows(frame, fit)]
    expected = standardize(train.pgs)
    if (record["train_rows"] != len(train) or record["train_sha256"] != person_set_hash(train.person_id)
            or record["held_out_in_train"] != 0
            or not np.isclose(record["standardization"]["mean"], expected["mean"], rtol=0, atol=1e-12)
            or not np.isclose(record["standardization"]["sd"], expected["sd"], rtol=1e-12, atol=0)):
        raise ValueError(f"fit provenance mismatch for {fit}")
    return record["standardization"]


def run_predict(spec, config, models):
    """One fitted model predicts its rows (outer test; LOGO: its held-out
    group's outer test) from its saved components, as a deployment would, every
    horizon in one call. Evaluation reuses these arrays and never predicts."""
    root = Path(spec["root"])
    out = root / spec["step"]
    kind, variant = spec["kind"], spec["variant"]
    # A convergence restart is the pooled fit again: the same rows, the same checks.
    fit = "pooled" if spec["fit"] == RESTART else spec["fit"]
    frame = load_frame(root / spec["frame"])
    test = frame.loc[frame.test].reset_index(drop=True)
    horizons = spec["horizons"]
    standardizations = [verify_provenance(frame, fit, read_json(root / step / "fit.json"))
                        for step in spec["models"].values()]
    index = np.flatnonzero(held_out(test, fit) if fit != "pooled" else np.ones(len(test), dtype=bool))
    data = model_frame(test.iloc[index], kind, standardizations[0], predict=True)
    started = time.perf_counter()
    prediction = models.predict(kind, variant, {c: root / s for c, s in spec["models"].items()}, data,
                                config["models"].get(kind, {}), horizons, disease=spec["disease_definition"])
    seconds = time.perf_counter() - started
    risk = np.asarray(prediction["risk"], dtype=float)
    shape = (len(index),) if kind == "binary" else (len(index), len(horizons))
    if risk.shape != shape or not np.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
        raise ValueError("predicted risks must be finite probabilities of the expected shape")
    # z_sd: the score scale this model's z (and so its slope) is per.
    np.savez(out / "predictions.npz", index=index, z_sd=standardizations[0]["sd"],
             **{name: np.asarray(value, dtype=float) for name, value in prediction.items()})
    write_json(out / "predict.json", {"status": "ok", "predict_seconds": round(seconds, 3), "rows": int(len(index))})


def run_evaluate(spec, config, models):
    from study import evaluate
    root = Path(spec["root"])
    out = root / spec["step"]
    kind = spec["kind"]
    frame = load_frame(root / spec["frame"])
    test = frame.loc[frame.test].reset_index(drop=True)
    train = frame.loc[~frame.test].reset_index(drop=True)
    pooled_sd = standardize(train.pgs)["sd"]
    truth_path = root / f"features/{spec['disease']}/truth.parquet"
    truth = None
    if truth_path.is_file():
        truth = test[["person_id"]].merge(pd.read_parquet(truth_path).drop(columns=["disease"]),
                                          on="person_id", how="left", validate="one_to_one")
        # Every slope here is per the pooled model's z (score / pooled sd), truth's included.
        for column in [c for c in truth.columns if c == "slope" or c.startswith("slope_cif_")]:
            truth[column] = to_pooled_z(truth[column], read_json(truth_path.parent / "truth.json")["z_sd"],
                                        pooled_sd)
    horizons = spec["horizons"]
    shape = (len(test),) if kind == "binary" else (len(test), len(horizons))
    # Every model's saved prediction, predicted once: pooled over all outer-test
    # rows, a LOGO fit over its held-out group's.
    predictions, slopes = {}, {}
    for variant, fits in spec["predictions"].items():
        for fit, step in fits.items():
            saved = np.load(root / step / "predictions.npz")
            predictions[(variant, fit)] = np.full(shape, np.nan)
            predictions[(variant, fit)][saved["index"]] = saved["risk"]
            if "slope" in saved:
                slopes[(variant, fit)] = np.full(shape, np.nan)
                slopes[(variant, fit)][saved["index"]] = to_pooled_z(saved["slope"], float(saved["z_sd"]), pooled_sd)
    # Per-person slopes (d probit risk / dz) feed slope recovery where evaluate takes them.
    extra = {"slopes": slopes} if slopes and "slopes" in inspect.signature(evaluate.evaluate).parameters else {}
    rows = evaluate.evaluate(kind, test, predictions, horizons, config, train=train, truth=truth, **extra)
    for row in rows:
        row.setdefault("disease", spec["disease"])
        row.setdefault("model", kind)
    write_json(out / "evaluate.json", {"status": "ok", "rows": rows})


def run_job(spec):
    config, _ = load_config(spec["config"], spec.get("source"))
    from study import models
    print(f"study_job_started {spec['type']} {spec['step']}", flush=True)
    {"fit": run_fit, "predict": run_predict, "evaluate": run_evaluate}[spec["type"]](spec, config, models)
    print("study_job_complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume the whole study")
    run.add_argument("--config", type=Path, default=HERE / "study.json")
    run.add_argument("--work", type=Path, required=True, help="local work directory")
    run.add_argument("--source", choices=["bigquery", "parquet"], help="override the config's data source")
    run.add_argument("--input", action="append", metavar="NAME=PATH", help="a staged input (repeatable)")
    run.add_argument("--checkpoint", help="gs:// prefix, or a local directory standing in for one")
    run.add_argument("--status-uri", help="gs:// object whose .status/ labels report progress")
    run.add_argument("--threads", type=int, help="total thread budget (default: the CPUs this process may use)")
    run.add_argument("--looks", help="gs:// prefix (or local directory) counting this config's outer-test looks")
    run.add_argument("--digest-uri", help="gs:// prefix the digest writes its token names to (AoU)")
    run.add_argument("--caveat", action="append", default=[], metavar="LABEL",
                     help="a fixed label the digest carries, e.g. gam2990_survival_ls_not_interpretable (repeatable)")
    run.add_argument("--claim", action="store_true",
                     help="a claim run: refuse unless every planned claim cell has its sized replicate count (R4)")
    run.add_argument("--keep-checkpoint", action="store_true",
                     help="keep the bucket checkpoint after a complete run (default: delete it, SPEC 7a)")
    run.add_argument("--memory-gb", type=float,
                     help="the memory this run holds on a shared node (default: the task's cgroup limit or host memory)")
    run.add_argument("--kinds", nargs="+", choices=KINDS,
                     help="a validation run's kinds (simulator only; recorded in the signature and run row)")
    run.add_argument("--diseases", nargs="+", metavar="SLUG",
                     help="a validation run's diseases (simulator only; recorded in the signature and run row)")
    run.add_argument("--stop-after", choices=STAGES)
    sub.add_parser("worker", help="serve pool jobs from stdin (internal)")
    frozen = sub.add_parser("hash", help="print the config hash to freeze as frozen_config_sha256")
    frozen.add_argument("--config", type=Path, default=HERE / "study.json")
    args = parser.parse_args()
    if args.command == "worker":
        from study.pool import serve
        serve(run_job)
    elif args.command == "hash":
        print(config_hash(load_config(args.config, "bigquery")[0]))
    else:
        Study(args).run()


if __name__ == "__main__":
    main()
