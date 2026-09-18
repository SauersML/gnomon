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
import hashlib
import importlib.metadata
import inspect
import json
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
from study.pool import Job, run_jobs  # noqa: E402

KINDS = ("binary", "survival")
# Outcome columns: in a training frame, never in a prediction frame. n_dates
# counts the disease's qualifying records, so it is an outcome too.
OUTCOMES = {"binary": ("y", "n_dates"), "survival": ("exit_age", "event", "followup", "n_dates")}
LOGO_AXES = ("ancestry", "region", "ehr_site")
# Binary y; survival event: 0 censored, 1 disease, 2 death, 3 exclusion (competing). Nothing else.
OUTCOME_CODES = {"binary": {0, 1}, "survival": {0, 1, 2, 3}}
# study.json primary_censoring_rule -> phenotypes.build_frames(censor=...).
CENSORING = {"ehr_end": "ehr_end", "min_death_cutoff": "cutoff"}
UNKNOWN = "unknown"
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
    if config["report"]["small_cell_max"] != digest.LIMIT:
        raise ValueError(f"the small-cell maximum is AoU policy's {digest.LIMIT}, one constant everywhere")
    if not config.get("variants") or not set(config["logo"]["variants"]) <= set(config["variants"]):
        raise ValueError("study.json lists its variants, and the LOGO variants are among them")
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
    """The installed gamfit wheel: its version and native library bytes."""
    try:
        distribution = importlib.metadata.distribution("gamfit")
    except importlib.metadata.PackageNotFoundError:
        return {"gamfit": None}
    native = sorted(str(f) for f in distribution.files or () if str(f).endswith((".so", ".pyd")))
    return {"gamfit": distribution.version,
            "native": {name: file_hash(distribution.locate_file(name)) for name in native}}


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
        if not self.parquet:
            check_frozen(self.config)
        self.work = Path(args.work).resolve()
        self.root = self.work / "steps"
        self.status = Status(args.status_uri)
        self.started = time.time()
        # Cost is charged on the task's vCPUs, whatever share of them the pool uses.
        self.vcpus = os.cpu_count()
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else self.vcpus
        self.threads = args.threads or self.config["compute"].get("total_threads") or available
        self.signature = {
            "study": study_identity(self.config),
            "deployment": deployment_identity(self.config),
            "code": code_identity(),
            "engine": engine_identity(),
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
        from study import models
        self.models = models

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
        """Every component a variant's prediction needs: its own and the shared ones."""
        dirs = {c: self.fit_step(slug, kind, variant, fit, c) for c in self.own_components(kind, variant)}
        dirs.update({c: self.fit_step(slug, kind, "shared", fit, c) for c in self.shared_components(kind)})
        return dirs

    def job(self, step, spec, threads, priority=0.0, deps=()):
        """A pool job for `step`, whose directory must already be begun. Its log
        lands in the step itself, so a job leaves no files outside its step."""
        spec = dict(spec, step=step, root=str(self.root), config=str(Path(self.args.config).resolve()),
                    source=self.args.source, horizons=self.horizons())
        return Job(key=step, spec=spec, threads=threads, log=self.path(step) / "job.log", deps=tuple(deps),
                   priority=priority, timeout=self.config["compute"]["job_timeout_seconds"],
                   affinity=spec.get("frame", ""))

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
        written.update(wall_seconds=round(outcome.seconds, 3), cpu_seconds=round(outcome.cpu_seconds, 3),
                       max_rss_mb=round(outcome.max_rss_mb, 1), threads=outcome.threads,
                       restarted_after_signal=outcome.restarted)
        write_json(record, written)
        self.checkpoint.complete(job.key, info={"status": written["status"]})

    def pool(self, jobs, stage, record_name):
        if not jobs:
            return {}
        marks = {max(1, len(jobs) * q // 4): q * 25 for q in (1, 2, 3)}

        def progress(done, total):
            if stage == "fits" and done in marks:
                self.status(f"study_fits_{marks[done]}")
        print(f"study_pool {stage} jobs={len(jobs)} threads={self.threads}", flush=True)
        return run_jobs(jobs, self.threads, lambda job, outcome: self.seal_job(job, outcome, record_name),
                        lambda threads: [sys.executable, str(HERE / "study.py"), "worker"], progress=progress)

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
        self.status("study_completed")

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
            manifest.pop("tables", None)
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
                                                 "sites": len([s for s in sites.values() if s.startswith("site")])})
            self.checkpoint.complete("features/base")
        truth_path = self.inputs.get("tables", Path("/nonexistent")) / "truth.parquet"
        truth = pd.read_parquet(truth_path) if self.parquet and truth_path.is_file() else None
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

    def stage_fits(self):
        threads = self.config["compute"]["threads"]
        jobs, scheduled = [], set()
        for disease in self.diseases:
            slug = disease.slug
            for kind in KINDS:
                rows = read_json(self.path(f"features/{slug}") / "plan.json")[kind]["rows"]
                plan = [("shared", c) for c in self.shared_components(kind)]
                plan += [(v, c) for v in self.variants(kind) for c in self.own_components(kind, v)]
                for variant, component in plan:
                    ours = variant in ("ours", "shared")
                    for fit in self.fits(slug, kind, None if variant == "shared" else variant):
                        step = self.fit_step(slug, kind, variant, fit, component)
                        if self.checkpoint.done(step):
                            continue
                        pooled = self.fit_step(slug, kind, variant, "pooled", component)
                        reuse = fit != "pooled" and self.config["logo"].get("reuse_pooled", False)
                        self.checkpoint.begin(step)
                        scheduled.add(step)
                        # Longest first: our model, survival, pooled, most rows.
                        priority = rows * (8 if ours else 1) * (2 if kind == "survival" else 1) \
                            * (2 if fit == "pooled" else 1)
                        jobs.append(self.job(step, {
                            "type": "fit", "disease": slug, "kind": kind, "variant": variant, "fit": fit,
                            "component": component, "frame": f"features/{slug}/{kind}.parquet",
                            "reference": pooled if reuse else None,
                        }, threads["ours" if ours else "competitor"][kind], priority,
                            deps=[pooled] if reuse and pooled in scheduled else ()))
        self.pool(jobs, "fits", "fit.json")

    def fit_ok(self, slug, kind, variant, fit):
        return all(read_json(self.path(step) / "fit.json")["status"] == "ok"
                   for step in self.model_dirs(slug, kind, variant, fit).values())

    def stage_predict(self):
        jobs = []
        for disease in self.diseases:
            for kind in KINDS:
                for variant in self.variants(kind):
                    step = f"predict/{disease.slug}/{kind}/{variant}"
                    if self.checkpoint.done(step):
                        continue
                    fits = {fit: self.model_dirs(disease.slug, kind, variant, fit)
                            for fit in self.fits(disease.slug, kind, variant)
                            if self.fit_ok(disease.slug, kind, variant, fit)}
                    self.checkpoint.begin(step)
                    jobs.append(self.job(step, {"type": "predict", "disease": disease.slug, "kind": kind,
                                                "variant": variant, "fits": fits,
                                                "frame": f"features/{disease.slug}/{kind}.parquet"},
                                         self.config["compute"]["threads"]["predict"],
                                         priority=len(fits) * (4 if variant == "ours" else 1)))
        self.pool(jobs, "predict", "predict.json")

    def stage_evaluate(self):
        # This checkpoint's outer-test look, recorded before any outer-test outcome is read.
        if self.look_marker not in self.looks.names():
            self.looks.put_bytes(self.look_marker, (config_hash(self.config) + "\n").encode())
        jobs = []
        for disease in self.diseases:
            for kind in KINDS:
                step = f"evaluate/{disease.slug}/{kind}"
                if self.checkpoint.done(step):
                    continue
                self.checkpoint.begin(step)
                jobs.append(self.job(step, {"type": "evaluate", "disease": disease.slug, "kind": kind,
                                            "variants": self.variants(kind),
                                            "frame": f"features/{disease.slug}/{kind}.parquet"},
                                     self.config["compute"]["threads"]["evaluate"],
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
                         "survival": {"single_record_at_risk": flow["survival"]["single_record_at_risk"]}}
            rows += digest.flow_rows(disease.slug, {name: flow[name] for name in ("disease", "binary", "survival")},
                                     digest.LIMIT, subgroups)
            # Whole-cohort counts by ancestry beside their totals, so the audit partitions them.
            if "by_ancestry" in flow:
                rows += digest.cohort_rows(disease.slug, flow["by_ancestry"])
            for kind in KINDS:
                record = read_json(self.path(f"evaluate/{disease.slug}/{kind}") / "evaluate.json")
                rows += record.get("rows", [])
        base = read_json(self.path("features/base") / "base.json")
        rows += digest.flow_rows("base", {"base": base["flow"]}, digest.LIMIT)
        rows += digest.followup_rows(base["followup"]["administrative"], digest.LIMIT)
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

    def operation_rows(self, base):
        """Aggregates of how the run went: no participant data, so not audited."""
        attempts = self.attempts()
        manifest = read_json(self.path("cohort") / "manifest.json")
        study = {"scope": "study", "item": "run", "config_sha256_12": config_hash(self.config)[:12],
                 "vcpus": self.vcpus, "threads": self.threads, "attempts": len(attempts),
                 "vcpu_hours": round(sum(a["vcpus"] * a["wall_seconds"] for a in attempts.values()) / 3600, 3),
                 "outer_test_looks": len([n for n in self.looks.names() if n.endswith(".txt")]),
                 "bigquery_bytes_billed": int((manifest.get("bigquery") or {}).get("bytes_billed", 0)),
                 "horizons": "_".join("h" + digest.token(float(h)) for h in base["horizons"])}
        bigquery = manifest.get("bigquery") or {}
        if "plan_bytes" in bigquery:
            study["bigquery_plan_bytes"] = int(bigquery["plan_bytes"])
        for key in ("ehr_domains", "ehr_domains_skipped"):
            if key in manifest:
                study[key] = "_".join(digest.label(d, 12) for d in manifest[key]) or "none"
        rows = [study]
        # Which EHR domains moved ehr_end later: exact audited fractions when the
        # manifest carries their denominator (ehr_people, see stage_digest); without
        # it a share leaves only as a coarse bucket, which pins no count.
        extended = manifest.get("ehr_extended_by") or {}
        if extended and not manifest.get("ehr_people"):
            rows.append({"scope": "ehr_extended_by", "item": "domains",
                         **{digest.label(domain, 20): share_bucket(share) for domain, share in extended.items()}})
        rows += [{"scope": "timing", "item": stage, "wall_seconds": round(seconds, 1)}
                 for stage, seconds in self.timings.items()]
        for disease in self.diseases:
            for kind in KINDS:
                plan = [("shared", c) for c in self.shared_components(kind)]
                plan += [(v, c) for v in self.variants(kind) for c in self.own_components(kind, v)]
                for variant, component in plan:
                    records = [read_json(self.path(self.fit_step(disease.slug, kind, variant, fit, component))
                                         / "fit.json")
                               for fit in self.fits(disease.slug, kind, None if variant == "shared" else variant)]
                    seconds = sorted(r.get("fit_seconds", r["wall_seconds"]) for r in records)
                    ok = [r for r in records if r["status"] == "ok"]
                    rows.append({"scope": "fits", "item": f"{disease.slug}.{kind}.{variant}.{component}",
                                 "fits": len(records), "ok": len(ok), "failed": len(records) - len(ok),
                                 "median_seconds": seconds[len(seconds) // 2], "max_seconds": seconds[-1],
                                 "cpu_seconds": round(sum(r["cpu_seconds"] for r in records), 1),
                                 "threads": records[0]["threads"],
                                 "max_rss_mb": max(r["max_rss_mb"] for r in records)})
                    categories = {}
                    for record in records:
                        if record["status"] != "ok":
                            label = f"{record['status']}_{record.get('category', 'unclassified')}"
                            categories[label] = categories.get(label, 0) + 1
                    if categories:
                        rows.append({"scope": "fit_failures", "item": f"{disease.slug}.{kind}.{variant}.{component}",
                                     **{digest.label(name): count for name, count in categories.items()}})
                for variant in self.variants(kind):
                    record = read_json(self.path(f"predict/{disease.slug}/{kind}/{variant}") / "predict.json")
                    outcomes = list(record.get("fits", {}).values())
                    rows.append({"scope": "predict", "item": f"{disease.slug}.{kind}.{variant}",
                                 "status": record["status"], "wall_seconds": record["wall_seconds"],
                                 "ok": sum(o["status"] == "ok" for o in outcomes),
                                 "failed": sum(o["status"] != "ok" for o in outcomes)})
                record = read_json(self.path(f"evaluate/{disease.slug}/{kind}") / "evaluate.json")
                rows.append({"scope": "evaluate", "item": f"{disease.slug}.{kind}", "status": record["status"],
                             "category": record.get("category", "none"), "wall_seconds": record["wall_seconds"]})
        return rows



def share_bucket(share):
    """A share as a coarse bucket label, which pins no count."""
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
    print("study_fit_started", flush=True)
    started = time.perf_counter()
    info = models.fit(kind, spec["variant"], spec["component"], data, config["models"].get(kind, {}), out, reference)
    seconds = time.perf_counter() - started
    print("study_fit_saved", flush=True)
    # Provenance (SPEC section 8, LOGO): the person set this fit saw, checked at predict.
    write_json(out / "fit.json", {
        "status": "ok", "fit_seconds": round(seconds, 3), "standardization": standardization,
        "train_rows": int(len(train)), "train_sha256": person_set_hash(train.person_id),
        "held_out_in_train": int(held_out(train, fit).sum()), "warm_reference": reference is not None,
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
    """Every fit of one disease, model and variant predicts its outer-test rows
    from the saved models, as a deployment would; all arrays go in one file."""
    root = Path(spec["root"])
    out = root / spec["step"]
    kind, variant = spec["kind"], spec["variant"]
    frame = load_frame(root / spec["frame"])
    test = frame.loc[frame.test].reset_index(drop=True)
    horizons = spec["horizons"]
    record, arrays = {"status": "ok", "fits": {}}, {}
    for fit, steps in spec["fits"].items():
        started = time.perf_counter()
        try:
            standardizations = [verify_provenance(frame, fit, read_json(root / step / "fit.json"))
                                for step in steps.values()]
            index = np.flatnonzero(held_out(test, fit) if fit != "pooled" else np.ones(len(test), dtype=bool))
            data = model_frame(test.iloc[index], kind, standardizations[0], predict=True)
            prediction = models.predict(kind, variant, {c: root / s for c, s in steps.items()}, data,
                                        config["models"].get(kind, {}), horizons)
            risk = np.asarray(prediction["risk"], dtype=float)
            shape = (len(index),) if kind == "binary" else (len(index), len(horizons))
            if risk.shape != shape or not np.isfinite(risk).all() or (risk < 0).any() or (risk > 1).any():
                raise ValueError("predicted risks must be finite probabilities of the expected shape")
        except Exception as error:
            text = traceback.format_exc()
            print(text, flush=True)
            record["fits"][fit] = {"status": "error", "category": failure_category(text.encode())}
            print(f"study_predict_failed {fit} {type(error).__name__}", flush=True)
            continue
        arrays[f"{fit_slug(fit)}__index"] = index
        for name, value in prediction.items():
            arrays[f"{fit_slug(fit)}__{name}"] = np.asarray(value, dtype=float)
        record["fits"][fit] = {"status": "ok", "seconds": round(time.perf_counter() - started, 3)}
    np.savez(out / "predictions.npz", **arrays)
    write_json(out / "predict.json", record)


def run_evaluate(spec, config, models):
    from study import evaluate
    root = Path(spec["root"])
    out = root / spec["step"]
    kind = spec["kind"]
    frame = load_frame(root / spec["frame"])
    test = frame.loc[frame.test].reset_index(drop=True)
    train = frame.loc[~frame.test].reset_index(drop=True)
    truth_path = root / f"features/{spec['disease']}/truth.parquet"
    truth = None
    if truth_path.is_file():
        truth = test[["person_id"]].merge(pd.read_parquet(truth_path).drop(columns=["disease"]),
                                          on="person_id", how="left", validate="one_to_one")
    horizons = spec["horizons"]
    shape = (len(test),) if kind == "binary" else (len(test), len(horizons))
    predictions, slopes = {}, {}
    for variant in spec["variants"]:
        directory = root / f"predict/{spec['disease']}/{kind}/{variant}"
        record = read_json(directory / "predict.json")
        if record["status"] != "ok":
            continue
        saved = np.load(directory / "predictions.npz")
        for fit, outcome in record["fits"].items():
            if outcome["status"] != "ok":
                continue
            index = saved[f"{fit_slug(fit)}__index"]
            predictions[(variant, fit)] = np.full(shape, np.nan)
            predictions[(variant, fit)][index] = saved[f"{fit_slug(fit)}__risk"]
            if f"{fit_slug(fit)}__slope" in saved:
                slopes[(variant, fit)] = np.full(len(test), np.nan)
                slopes[(variant, fit)][index] = saved[f"{fit_slug(fit)}__slope"]
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
    run.add_argument("--keep-checkpoint", action="store_true",
                     help="keep the bucket checkpoint after a complete run (default: delete it, SPEC 7a)")
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
