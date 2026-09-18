#!/usr/bin/env python3
"""Bounded AoU incident-survival pilot; execute only inside the AoU workspace.

Uses cached Gnomon scores, the shared disease selector, and the formula-first
gamfit API. Does not invoke Gnomon's different calibration adapter model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

from aou_score_transform import (baseline_columns, declared_law_diagnostics,
                                 score_diagnostics, transformed_score)
from reference_ctn import load_reference
from aou_checkpoint import StudyCheckpoint, result_identity
from aou_evaluation import audit_groups, loss_summary
from aou_status import failure_label, publish_status


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_config(c):
    expected = {
        "google_project", "workspace_cdr", "gamfit_version", "top_n_diseases",
        "disease_limit", "num_pcs", "baseline_centers", "slope_centers",
        "time_num_internal_knots",
        "max_rows_per_disease", "train_fraction", "seed", "horizons_years",
        "grid_intervals", "fit_timeout_seconds", "query_timeout_seconds",
        "maximum_bytes_billed", "min_train_events_per_cause", "min_report_count",
        "lookback_days", "projection_model_sha256", "landmark_days", "fit_no_score_comparator",
        "survival_time_anchor", "fit_budget", "score_law",
    }
    if set(c) != expected:
        raise ValueError("analysis configuration has missing or unknown keys")
    if c["score_law"] not in {"declared_empirical", "reference_ctn_gaussian"}:
        raise ValueError("score_law must be declared_empirical or reference_ctn_gaussian")
    if not re.fullmatch(r"[a-z][a-z0-9-]+\.[A-Za-z0-9_]+", c["workspace_cdr"]):
        raise ValueError("workspace_cdr must be a concrete project.dataset")
    if not re.fullmatch(r"[a-z][a-z0-9-]+", c["google_project"]):
        raise ValueError("google_project must be a concrete billing project")
    positive = expected - {"google_project", "workspace_cdr", "gamfit_version",
                           "train_fraction", "seed", "horizons_years", "projection_model_sha256",
                           "landmark_days", "fit_no_score_comparator", "survival_time_anchor", "fit_budget",
                           "score_law"}
    anchor = c["survival_time_anchor"]
    if anchor is not None and (type(anchor) not in (int, float) or not 0 <= anchor <= 10):
        raise ValueError("survival_time_anchor must be null or a follow-up time in years within a decade")
    if type(c["landmark_days"]) is not int or not 0 <= c["landmark_days"] <= 730:
        raise ValueError("landmark_days must be a whole number of days within two years")
    if type(c["fit_no_score_comparator"]) is not bool:
        raise ValueError("fit_no_score_comparator must be true or false")
    if any(type(c[k]) is not int or c[k] <= 0 for k in positive):
        raise ValueError("resource and sample budgets must be positive integers")
    if not 1 <= c["num_pcs"] <= 16 or not 0.5 <= c["train_fraction"] <= 0.9:
        raise ValueError("unsupported PC count or training fraction")
    budget = c["fit_budget"]
    if not isinstance(budget, dict) or set(budget) != {
            "gamfit_version", "engine_sha256", "solver_threads", "training_rows", "wall_seconds",
            "exponent", "budget_seconds", "source"}:
        raise ValueError("fit_budget has missing or unknown keys")
    engine = budget["engine_sha256"]
    if (any(type(budget[k]) is not int or budget[k] <= 0 for k in ("solver_threads", "training_rows", "budget_seconds"))
            or type(budget["wall_seconds"]) not in (int, float) or not budget["wall_seconds"] > 0
            or type(budget["exponent"]) not in (int, float) or not 1 <= budget["exponent"] <= 3
            or not (engine is None or (isinstance(engine, str) and re.fullmatch(r"[0-9a-f]{64}", engine)))
            or not isinstance(budget["source"], str) or not budget["source"].strip()):
        raise ValueError("fit_budget needs a positive measured cost, an exponent in [1, 3], "
                         "the measured engine's SHA-256 or null, and its source")
    if budget["gamfit_version"] != c["gamfit_version"]:
        raise ValueError("fit_budget was measured with a different gamfit version")
    if budget["budget_seconds"] > c["fit_timeout_seconds"]:
        raise ValueError("fit_budget cannot allow more than the per-stage fit timeout")
    if c["max_rows_per_disease"] > fit_budget_rows(c):
        raise ValueError("max_rows_per_disease exceeds what the measured fit budget can finish")
    if min(c["baseline_centers"], c["slope_centers"]) <= c["num_pcs"] + 1:
        raise ValueError("Duchon basis sizes must exceed the PC count plus one")
    if c["time_num_internal_knots"] < 2:
        raise ValueError("survival time basis requires at least two internal knots")
    if not re.fullmatch(r"[0-9a-f]{64}", c["projection_model_sha256"]):
        raise ValueError("the external PC projection model must be pinned by SHA-256")
    if c["min_report_count"] < 20 or c["min_train_events_per_cause"] < 20:
        raise ValueError("pilot requires at least 20 observations per reported cell/event class")
    if type(c["seed"]) is not int or not c["gamfit_version"]:
        raise ValueError("seed and gamfit version are required")
    h = np.asarray(c["horizons_years"], dtype=float)
    if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all() or h[0] <= 0 or (np.diff(h) <= 0).any():
        raise ValueError("horizons must be finite, positive, and strictly increasing")


def fit_budget_rows(c):
    """The largest cohort whose final-stage fits finish within the budget, by the
    measured stage wall scaled as training rows ** exponent. The final stage trains
    on `train_fraction` of the cohort; development trains on less."""
    budget = c["fit_budget"]
    rows = budget["training_rows"] * (budget["budget_seconds"] / budget["wall_seconds"]) ** (1 / budget["exponent"])
    return int(rows / c["train_fraction"])


class BoundedClient:
    """Apply query budgets even to the existing selector's query calls."""
    def __init__(self, config, account):
        from google.cloud import bigquery
        from google.auth.compute_engine import Credentials
        self.client = bigquery.Client(project=config["google_project"],
                                      credentials=Credentials(service_account_email=account))
        self.config = config
        self.jobs = []
        self.remaining_bytes = config["maximum_bytes_billed"]

    def query(self, sql, job_config=None):
        from google.cloud import bigquery
        job_config = job_config or bigquery.QueryJobConfig()
        if self.remaining_bytes <= 0:
            raise RuntimeError("the cumulative BigQuery byte budget is exhausted")
        job_config.maximum_bytes_billed = self.remaining_bytes
        job_config.use_query_cache = True
        job_config.job_timeout_ms = self.config["query_timeout_seconds"] * 1000
        job = self.client.query(sql, job_config=job_config)
        self.jobs.append(job)
        try:
            job.result(timeout=self.config["query_timeout_seconds"])
        except BaseException:
            job.cancel()
            raise
        billed = 0 if job.cache_hit else job.total_bytes_billed
        if billed is None or billed < 0:
            raise RuntimeError("BigQuery did not report billable bytes; refusing further queries")
        self.remaining_bytes -= billed
        return job


def unpack_phenotypes(archive, destination):
    """Stage only the reference metadata/JSON used by the shared selector."""
    destination.mkdir()
    with zipfile.ZipFile(archive) as z:
        metadata = [n for n in z.namelist() if n.endswith("/inst/Cohorts.csv")]
        if len(metadata) != 1:
            raise ValueError("expected one PhenotypeLibrary inst/Cohorts.csv")
        prefix = metadata[0][:-len("inst/Cohorts.csv")]
        for name in z.namelist():
            relative = name.removeprefix(prefix)
            if name == metadata[0] or (relative.startswith("inst/cohorts/") and relative.endswith(".json")):
                rel = Path(relative)
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("unsafe phenotype archive member")
                target = destination / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
    return destination


def read_ancestry(ancestry, prune, num_pcs, projections):
    ancestry_columns = pd.read_csv(ancestry, sep="\t", nrows=0).columns
    if not {"research_id", "ancestry_pred"}.issubset(ancestry_columns):
        raise ValueError("ancestry file lacks required research_id/ancestry_pred columns")
    df = pd.read_csv(ancestry, sep="\t", dtype=str,
                     usecols=["research_id", "ancestry_pred"])
    # The release's published flagged-sample file uses sample_id. These IDs
    # join to research_id in the ancestry predictions; require the real schema.
    excluded = pd.read_csv(prune, sep="\t", dtype=str,
                           keep_default_na=False, skip_blank_lines=False)
    if list(excluded.columns) != ["sample_id"]:
        raise ValueError("relatedness prune must be a single sample_id column")
    excluded.columns = ["research_id"]
    if not excluded.research_id.str.fullmatch(r"[0-9]+").all():
        raise ValueError("relatedness prune contains invalid research IDs")
    if excluded.empty or excluded.research_id.isna().any():
        raise ValueError("AoU relatedness prune list is missing or empty")
    if df.research_id.isna().any() or df.research_id.duplicated().any():
        raise ValueError("ancestry IDs must be present and unique")
    df = df.loc[~df.research_id.isin(excluded.research_id)].copy()
    pc_columns = [f"PC{i + 1}" for i in range(num_pcs)]
    pcs = pd.read_parquet(projections)
    if not {"IID", *pc_columns}.issubset(pcs.columns):
        raise ValueError("cached projection lacks IID or required PC columns")
    pcs = pcs[["IID", *pc_columns]].rename(columns={"IID": "person_id"}).copy()
    pcs["person_id"] = pcs.person_id.astype("string")
    if pcs.person_id.isna().any() or pcs.person_id.duplicated().any() or not np.isfinite(pcs[pc_columns]).all().all():
        raise ValueError("invalid cached projection IDs or PCs")
    labels = df.rename(columns={"research_id": "person_id", "ancestry_pred": "ancestry"})
    out = labels.merge(pcs, on="person_id", validate="one_to_one")
    if out.empty or out.ancestry.isna().any():
        raise ValueError("projected reference coordinates have no labelled participants")
    # Published max-IS pruning supplies the independent units for this pilot.
    # Explicit group labels remain the interface for the development/test split.
    out["split_group"] = out.person_id
    return out


def unpack_score_cache(archive, output, projection_output):
    """Read pgsEngine's real shared-feature artifact without extracting genotypes."""
    output = Path(output)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix="scores-", suffix=".partial",
                                     delete=False) as target:
        temporary = Path(target.name)
        try:
            count = 0
            projection_count = 0
            projection_temporary = Path(str(projection_output) + ".partial")
            # Streaming avoids scanning the multi-GB gzip and then decompressing
            # it again to seek back to the score member. Still inspect the rest
            # of the archive to reject duplicate score members.
            with tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    if member.isfile() and Path(member.name).name == "projection_pcs.parquet":
                        projection_count += 1
                        if projection_count > 1:
                            raise ValueError("expected exactly one cached projection PC table")
                        with tar.extractfile(member) as source, projection_temporary.open("wb") as destination:
                            shutil.copyfileobj(source, destination, length=1024 * 1024)
                    if not member.isfile() or Path(member.name).name != "scores.tar":
                        continue
                    count += 1
                    if count > 1:
                        raise ValueError("expected exactly one scores.tar in the pgsEngine shared-feature archive")
                    with tar.extractfile(member) as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            if count != 1:
                raise ValueError("expected exactly one scores.tar in the pgsEngine shared-feature archive")
            if projection_count != 1:
                raise ValueError("expected exactly one cached projection PC table")
            target.close()
            temporary.replace(output)
            projection_temporary.replace(projection_output)
        finally:
            temporary.unlink(missing_ok=True)
            projection_temporary.unlink(missing_ok=True)
    return output


def load_cached_score(archive, pgs):
    column = f"{pgs}_AVG"
    missing_column = f"{pgs}_MISSING_PCT"
    result = None
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".sscore"):
                continue
            with tar.extractfile(member) as handle:
                header = handle.readline().decode("utf-8").rstrip().split("\t")
            if column not in header:
                continue
            if missing_column not in header:
                raise ValueError(f"cached score {pgs} lacks per-participant missingness")
            ids = [name for name in header if name.lstrip("#") == "IID"]
            if len(ids) != 1 or result is not None:
                raise ValueError(f"ambiguous cached score source for {pgs}")
            with tar.extractfile(member) as handle:
                result = pd.read_csv(handle, sep="\t", usecols=[ids[0], column, missing_column],
                                     dtype={ids[0]: str})
            result = result.rename(columns={ids[0]: "person_id", column: "PGS"})
    if result is None:
        raise ValueError(f"selected score {pgs} is absent from scores archive; populate that cache first")
    if result.person_id.isna().any() or result.person_id.duplicated().any():
        raise ValueError("score IDs must be present and unique")
    if not np.isfinite(result.PGS.to_numpy(dtype=float)).all():
        raise ValueError("cached score contains non-finite values")
    missing = result[missing_column].to_numpy(dtype=float)
    if not np.isfinite(missing).all() or (missing < 0).any() or (missing > 100).any():
        raise ValueError("cached score has invalid missingness percentages")
    if (missing == 100).any():
        raise ValueError("completely missing scores cannot enter CTN as numerical scores")
    return result


def cached_score_ids(archive):
    ids = set()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if member.isfile() and member.name.endswith(".sscore"):
                with tar.extractfile(member) as handle:
                    header = handle.readline().decode("utf-8").rstrip().split("\t")
                ids.update(name[:-4] for name in header if re.fullmatch(r"PGS\d{6}_AVG", name))
    return ids


def endpoint_scores(archive, disease):
    first = disease["candidates"][0]
    scores = load_cached_score(archive, first)
    scores[first] = scores.PGS
    return scores


def load_score_panel(path, *, exploratory, endpoints=None):
    """Validate the prespecified panel. A final analysis demands a completed
    development audit for the scores of the endpoints it analyses; other
    endpoints in the panel wait for their own audits."""
    panel = json.loads(Path(path).read_text())
    if endpoints is not None and any(name not in panel["endpoints"] for name in endpoints):
        raise ValueError("requested endpoint is not in the prespecified panel")
    for name, endpoint in panel["endpoints"].items():
        candidates = endpoint["candidates"]
        if len(candidates) != 1:
            raise ValueError("each endpoint requires exactly one prespecified score")
        for pgs in candidates:
            if pgs == "PGS004787" or pgs in panel["excluded"] or pgs not in panel["scores"]:
                raise ValueError("excluded or unlisted PGS in candidate panel")
            if not exploratory and (endpoints is None or name in endpoints):
                audit = panel["scores"][pgs].get("development_audit") or {}
                for stage in ("discovery", "components", "tuning"):
                    record = audit.get(stage, {})
                    sources = record.get("sources", [])
                    if (record.get("status") != "no_documented_aou_development"
                            or not isinstance(sources, list) or not sources
                            or not all(isinstance(s, str) and s.startswith("https://") for s in sources)):
                        raise ValueError(f"final analysis requires completed {stage} provenance for {pgs}")
    return panel


def development_partition(df, config):
    """Outer-test rows are removed before score/model selection can see data."""
    development = df.loc[df.is_train].copy().reset_index(drop=True)
    development["is_train"] = development.split_group.map(lambda group:
        int(hashlib.sha256(f"{config['seed']}:development:{group}".encode()).hexdigest()[:16], 16)
        / 2**64 < .75)
    return development


def fit_support(train, test, config):
    errors = []
    for cause in (1, 2):
        if (train.event_code == cause).sum() < config["min_train_events_per_cause"]:
            errors.append(f"insufficient training events for cause {cause}")
    if len(test) < config["min_report_count"]:
        errors.append("too few held-out participants")
    return errors


def partition_support(train, test, config):
    errors = fit_support(train, test, config)
    for horizon in config["horizons_years"]:
        try:
            ipcw_weights(train, test, horizon, {})
        except ValueError as error:
            errors.append(f"horizon {horizon:g}: {error}")
    return errors


def person_times(client, cdr):
    # AoU's primary-consent enrollment proxy, not EHR consent or first care.
    # A single interval must cover enrollment and the prespecified lookback.
    return client.query(f"""
      WITH enrollment AS (
        SELECT o.person_id, MIN(o.observation_date) AS baseline
        FROM `{cdr}.concept` c
        JOIN `{cdr}.concept_ancestor` ca ON c.concept_id = ca.ancestor_concept_id
        JOIN `{cdr}.observation` o ON ca.descendant_concept_id = o.observation_concept_id
        WHERE c.concept_name = 'Consent PII' AND c.concept_class_id = 'Module'
        GROUP BY o.person_id
      ), periods AS (
        SELECT e.person_id, e.baseline, op.observation_period_start_date AS obs_start,
               op.observation_period_end_date AS obs_end,
               ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY
                 observation_period_start_date, observation_period_end_date DESC,
                 observation_period_id) AS seq
        FROM `{cdr}.observation_period` op JOIN enrollment e USING(person_id)
        WHERE op.observation_period_start_date <= e.baseline
          AND op.observation_period_end_date >= e.baseline
      ), deaths AS (
        SELECT person_id, MIN(death_date) AS death_date FROM `{cdr}.aou_death`
        WHERE primary_death_record = TRUE
        GROUP BY person_id
      )
      SELECT CAST(p.person_id AS STRING) AS person_id,
             DATE(p.birth_datetime) AS birth_date,
             p.sex_at_birth_concept_id, o.baseline, o.obs_start, o.obs_end, d.death_date
      FROM `{cdr}.person` p JOIN periods o USING(person_id)
      LEFT JOIN deaths d USING(person_id)
      WHERE o.seq = 1
    """).to_dataframe(create_bqstorage_client=False)


def case_dates(client, cdr, concept_id):
    from google.cloud import bigquery
    return client.query(f"""
      SELECT CAST(co.person_id AS STRING) AS person_id,
             MIN(co.condition_start_date) AS disease_date
      FROM `{cdr}.condition_occurrence` co
      JOIN `{cdr}.concept_ancestor` ca
        ON ca.descendant_concept_id = co.condition_concept_id
      WHERE ca.ancestor_concept_id = @root GROUP BY co.person_id
    """, bigquery.QueryJobConfig(query_parameters=[
        bigquery.ScalarQueryParameter("root", "INT64", concept_id)
    ])).to_dataframe(create_bqstorage_client=False)


def build_cohort(base, scores, cases, config):
    df = base.merge(scores, on="person_id", validate="one_to_one").merge(
        cases, on="person_id", how="left", validate="one_to_one")
    for name in ["birth_date", "baseline", "obs_start", "obs_end", "death_date", "disease_date"]:
        df[name] = pd.to_datetime(df[name])
    df["sex"] = df.sex_at_birth_concept_id.map({8507: 1, 8532: 0, 45880669: 1, 45878463: 0})
    # Diagnoses made at the enrollment visits record prevalent disease, so
    # follow-up starts at a landmark after consent and anyone diagnosed,
    # dead or lost before it is not an incident-risk participant.
    df["landmark"] = df.baseline + pd.to_timedelta(config["landmark_days"], unit="D")
    df["age0"] = (df.landmark - df.birth_date).dt.days / 365.25
    eligible = (df.sex.notna() & df.age0.ge(18) & df.obs_end.gt(df.landmark)
                & ((df.baseline - df.obs_start).dt.days >= config["lookback_days"])
                & (df.disease_date.isna() | df.disease_date.gt(df.landmark))
                & (df.death_date.isna() | df.death_date.gt(df.landmark)))
    df = df.loc[eligible].copy()
    # Recorded diagnosis takes precedence for same-day ties. Retain the
    # participant; future event ordering must not select the baseline cohort.
    df["disease_death_same_day"] = (df.disease_date.notna()
        & df.disease_date.eq(df.death_date) & df.disease_date.le(df.obs_end))
    end = df[["obs_end", "disease_date", "death_date"]].min(axis=1)
    df["event_code"] = np.select([df.disease_date.eq(end), df.death_date.eq(end)], [1, 2], default=0)
    df["followup"] = (end - df.landmark).dt.days / 365.25
    df["entry"] = 0.0
    # Outcome-blind participant sampling and split after the published max-IS
    # prune. Do not balance events/censors or choose the seed by event counts.
    def hash_value(identifier, purpose):
        return hashlib.sha256(f"{config['seed']}:{purpose}:{identifier}".encode()).hexdigest()
    df["sample_order"] = df.person_id.map(lambda x: hash_value(x, "sample"))
    df = df.sort_values("sample_order").head(config["max_rows_per_disease"]).copy()
    df["is_train"] = df.split_group.map(
        lambda x: int(hash_value(x, "split")[:16], 16) / 2**64 < config["train_fraction"])
    return df.reset_index(drop=True)


def cif_from_hazards(hazards):
    """Conditional-on-entry CIF, with constant hazard proportions per interval."""
    h = np.asarray(hazards, dtype=float)  # cause, participant, grid
    if h.ndim != 3 or not np.isfinite(h).all() or (h < -1e-10).any():
        raise ValueError("invalid cumulative hazard surface")
    increments = np.diff(h, axis=2)
    if (increments < -1e-8).any():
        raise ValueError("cumulative hazard decreases")
    increments = np.maximum(increments, 0.0)  # only numerical roundoff
    total = increments.sum(axis=0)
    survival_before = np.exp(-np.concatenate([np.zeros_like(total[:, :1]),
                                             np.cumsum(total, axis=1)[:, :-1]], axis=1))
    mass = survival_before * -np.expm1(-total)
    shares = np.divide(increments, total[None, :, :], out=np.zeros_like(increments),
                       where=total[None, :, :] > 0)
    cif = np.concatenate([np.zeros_like(h[:, :, :1]), np.cumsum(shares * mass, axis=2)], axis=2)
    if (cif.sum(axis=0) > 1 + 1e-10).any():
        raise ValueError("cause probabilities exceed one")
    return cif


def censor_km(train):
    t = train.followup.to_numpy(float)
    code = train.event_code.to_numpy(int)
    times, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    censored = np.bincount(inverse, weights=code == 0, minlength=len(times))
    at_risk = np.cumsum(counts[::-1])[::-1]
    denominator = at_risk - (counts - censored)
    fraction = np.divide(censored, denominator, out=np.zeros_like(censored), where=denominator > 0)
    return times, np.cumprod(1 - fraction)


# The censoring model's constants, fixed before its Monte Carlo (gnomon#2338): prior
# strengths, in censorings at the pooled rate, for a stratum's level over its whole
# follow-up and for each interval's ratio around that level; the number of intervals
# of equal pooled censoring mass; the positivity floor and the one-sided 95% normal
# quantile of its upper bound.
CENSORING_LEVEL_PRIOR, CENSORING_INTERVAL_PRIOR, CENSORING_INTERVALS = 1.0, 1.0, 20
POSITIVITY_FLOOR, ONE_SIDED_95 = 0.05, 1.6448536269514722


def censor_km_at(reference, horizon):
    km_t, km_g = censor_km(reference)
    return float(np.r_[1.0, km_g][np.searchsorted(km_t, horizon, side="right")])


def censoring_model(train):
    """Every training ancestry stratum's censoring survival over the pooled training set.

    The pooled reverse Kaplan-Meier increments dL_k are grouped into CENSORING_INTERVALS
    intervals of equal pooled censoring mass. A stratum's censorings O_j in interval j,
    against E_j, the censorings its own risk set would show at the pooled rate, give its
    censoring ratio with two levels of Poisson-gamma shrinkage: the level
    R = (O + A) / (E + A) over its whole follow-up, and theta_j = (O_j + a R) / (E_j + a),
    A and a the level and interval priors. Its survival is
    G(t) = prod over t_k <= t of (1 - dL_k) ** theta_j(k). A well-supported interval takes
    about the stratum's own ratio; a sparse one borrows the stratum's own level, not the
    pooled set's, and inside an interval the pooled curve's shape."""
    censored = train.event_code.to_numpy(int) == 0
    times, inverse, counts = np.unique(train.followup.to_numpy(float), return_inverse=True, return_counts=True)
    def at_risk(rows, censorings):
        # Events at a censoring time leave the risk set first, as in censor_km.
        return np.cumsum(rows[::-1])[::-1] - (rows - censorings)
    d_all = np.bincount(inverse, weights=censored, minlength=len(times))
    n_all = at_risk(counts, d_all)
    dl = np.divide(d_all, n_all, out=np.zeros_like(d_all), where=n_all > 0)
    with np.errstate(divide="ignore"):
        ell = np.where(dl > 0, np.log1p(-dl), 0.0)
    interval = np.minimum(((np.cumsum(d_all) - d_all) * CENSORING_INTERVALS // max(d_all.sum(), 1)).astype(int),
                          CENSORING_INTERVALS - 1)
    one_hot = np.zeros((len(times), CENSORING_INTERVALS))
    one_hot[np.arange(len(times)), interval] = ell
    remaining = n_all - d_all
    model = {"times": times, "interval": interval, "d": d_all,
             "inv_remaining": np.divide(1.0, remaining, out=np.zeros_like(remaining), where=remaining > 0),
             # cumulative[m, j]: log(1 - dL_k) summed over the first m times that fall in interval j.
             "cumulative": np.vstack([np.zeros(CENSORING_INTERVALS), np.cumsum(one_hot, axis=0)]),
             "log_g_pooled": np.r_[0.0, np.cumsum(ell)], "strata": {}}
    labels = train.ancestry.to_numpy()
    for label in np.unique(labels):
        mine = labels == label
        d_s = np.bincount(inverse, weights=mine & censored, minlength=len(times))
        n_s = at_risk(np.bincount(inverse, weights=mine, minlength=len(times)), d_s)
        observed = np.bincount(interval, weights=d_s, minlength=CENSORING_INTERVALS)
        expected = np.bincount(interval, weights=n_s * dl, minlength=CENSORING_INTERVALS)
        level = (observed.sum() + CENSORING_LEVEL_PRIOR) / (expected.sum() + CENSORING_LEVEL_PRIOR)
        scale = expected + CENSORING_INTERVAL_PRIOR
        ratio = (observed + CENSORING_INTERVAL_PRIOR * level) / scale
        model["strata"][label] = {
            "d": d_s, "level": level, "ratio": ratio, "inv_scale": 1.0 / scale,
            "kappa": CENSORING_INTERVAL_PRIOR / (expected.sum() + CENSORING_LEVEL_PRIOR),
            "share": np.divide(n_s, n_all, out=np.zeros_like(n_s), where=n_all > 0),
            "log_g": np.r_[0.0, np.cumsum(ratio[interval] * ell)]}
    return model


def censoring_log_survival_variance(model, ancestry, position):
    """Delta-method variance of a stratum's log censoring survival at a time-grid position,
    over the training censoring counts d_{u,k}, each taken as Poisson with variance d_{u,k}
    and carried through the pooled increments, the expected counts, the level and the
    interval ratios."""
    stratum, j = model["strata"][ancestry], model["interval"]
    theta = stratum["ratio"][j]
    base = np.where(np.arange(len(j)) < position, -theta * model["inv_remaining"], 0.0)
    level = model["cumulative"][position] * stratum["inv_scale"]
    spread = stratum["kappa"] * level.sum()
    own = level[j] + spread
    cross = stratum["share"] * (theta * level[j] + stratum["level"] * spread)
    return float(np.sum((base + own - cross)**2 * stratum["d"] + (base - cross)**2 * (model["d"] - stratum["d"])))


def censoring_standard_error(model, test, positions, contribution, count):
    """Standard error the censoring model's estimation adds to sum(contribution) / count,
    contribution a participant's IPCW weight times their loss and zero outside the cell,
    by the delta method over the same counts as censoring_log_survival_variance."""
    j, size = model["interval"], len(model["times"])
    labels = test.ancestry.to_numpy()
    base, own = np.zeros(size), {}
    for ancestry in np.unique(labels):
        rows = labels == ancestry
        c = contribution[rows]
        # after[k]: the contribution of participants whose weight includes time k.
        after = np.cumsum(np.bincount(positions[rows], weights=c, minlength=size + 1)[::-1])[::-1][1:]
        stratum = model["strata"].get(ancestry)
        if stratum is None:
            base += after * model["inv_remaining"]
            continue
        level = (c[:, None] * model["cumulative"][positions[rows]]).sum(axis=0) * stratum["inv_scale"]
        spread = stratum["kappa"] * level.sum()
        theta = stratum["ratio"][j]
        base += (theta * (after * model["inv_remaining"] + stratum["share"] * level[j])
                 + stratum["share"] * stratum["level"] * spread)
        own[ancestry] = level[j] + spread
    variance = sum(float(np.sum((base - own.get(ancestry, 0.0))**2 * stratum["d"]))
                   for ancestry, stratum in model["strata"].items())
    return float(np.sqrt(variance)) / count


def censoring_support(train, ancestry, horizon):
    """(problem, reason) where a stratum's own reverse Kaplan-Meier cannot carry a horizon,
    else None: under 20 training rows, or own censoring survival below the floor there or
    nobody followed past it."""
    reference = train.loc[train.ancestry == ancestry]
    if len(reference) < 20:
        return "insufficient training support for ancestry-specific censoring", "training_rows"
    if censor_km_at(reference, horizon) < POSITIVITY_FLOOR or not (reference.followup > horizon).any():
        return "evaluation horizon lacks censoring support in an ancestry stratum", "horizon_support"
    return None


class CensoringRefusal(ValueError):
    """A horizon refused on positivity: in each stratum of `strata`, a record of its
    ancestry, reason, modelled censoring survival at the horizon and that value's upper
    bound, the censoring model can stand in for no stable weight."""
    def __init__(self, message, strata):
        super().__init__(message)
        self.strata = strata


def censoring_weights(model, train, test, horizon, pooled=None):
    """IPCW weights at a horizon from the censoring model, and each participant's position
    on its time grid, which the censoring standard errors need.

    A horizon the pooled training set cannot support is refused. So is one where, for any
    test stratum, the upper one-sided 95% bound on the modelled censoring survival at the
    horizon is below POSITIVITY_FLOOR: weights there would pass 20 with no stable
    estimator. Refusing on the bound rather than the point estimate refuses only where
    positivity confidently fails, so a stratum just under the floor by noise reports
    instead of the refusal selecting the replicates where its estimate came out high.
    Every refused stratum is named with its reason and both values. A stratum without
    its own reverse Kaplan-Meier support is recorded in `pooled` against its reason, the
    caveat the report carries; with `pooled` None that shortfall is fatal."""
    if censor_km_at(train, horizon) < POSITIVITY_FLOOR or not (train.followup > horizon).any():
        raise ValueError("evaluation horizon lacks censoring support in the pooled training set")
    t = test.followup.to_numpy(float)
    event = (test.event_code.to_numpy(int) != 0) & (t <= horizon)
    past = t > horizon
    at_horizon = int(np.searchsorted(model["times"], horizon, side="right"))
    positions = np.zeros(len(test), int)
    positions[event] = np.searchsorted(model["times"], t[event], side="left")
    positions[past] = at_horizon
    weights, refused = np.zeros(len(test)), []
    for ancestry in test.ancestry.unique():
        support = censoring_support(train, ancestry, horizon)
        if support is not None and pooled is None:
            raise ValueError(support[0])
        stratum = model["strata"].get(ancestry)
        log_g = stratum["log_g"] if stratum else model["log_g_pooled"]
        log_h = float(log_g[at_horizon])
        bound = log_h + (ONE_SIDED_95 * np.sqrt(censoring_log_survival_variance(model, ancestry, at_horizon))
                         if stratum else 0.0)
        if not np.exp(bound) >= POSITIVITY_FLOOR:
            refused.append({"ancestry": str(ancestry), "reason": support[1] if support else "modelled_survival",
                            "censoring_survival": float(np.exp(log_h)),
                            "censoring_survival_upper": float(np.exp(bound))})
            continue
        if support is not None:
            pooled[str(ancestry)] = support[1]
        rows = test.ancestry.eq(ancestry).to_numpy() & (event | past)
        weights[rows] = np.exp(-log_g[positions[rows]])
    if refused:
        raise CensoringRefusal("evaluation horizon lacks censoring support in an ancestry stratum: the upper 95% "
                               "bound on its modelled censoring survival is below 0.05",
                               sorted(refused, key=lambda s: s["ancestry"]))
    return weights, positions


def ipcw_weights(train, test, horizon, pooled=None):
    return censoring_weights(censoring_model(train), train, test, horizon, pooled)[0]


def weighted_auc(score, target, weights):
    """Weighted probability that a case outranks a control; ties count half."""
    score, target, weights = (np.asarray(v, dtype=float) for v in (score, target, weights))
    order = np.argsort(score, kind="mergesort")
    s, case_w, ctrl_w = score[order], (weights * target)[order], (weights * (1 - target))[order]
    total = case_w.sum() * ctrl_w.sum()
    if total <= 0:
        raise ValueError("AUC needs weighted cases and controls")
    starts = np.concatenate([[0], np.flatnonzero(np.diff(s)) + 1])
    ctrl_below = np.concatenate([[0.0], np.cumsum(ctrl_w)])[starts]
    case_group, ctrl_group = np.add.reduceat(case_w, starts), np.add.reduceat(ctrl_w, starts)
    return float(np.sum(case_group * (ctrl_below + 0.5 * ctrl_group)) / total)


def evaluation_cells(train, test, risk, horizons, min_count):
    """Yield one (horizon index, horizon, label, mask, weights, targets, censoring_se) per
    reportable audit cell, censoring_se(values) the standard error the censoring model's
    estimation adds to the mean of values over the cell, or a status row for cells that
    cannot be reported."""
    def reportable(ancestry):
        # Numbers drawn from a stratum's own rows are withheld under the reporting minimum.
        return min(test.ancestry.astype(str).eq(ancestry).sum(),
                   train.ancestry.astype(str).eq(ancestry).sum()) >= min_count
    model = censoring_model(train)
    for j, horizon in enumerate(horizons):
        pooled = {}
        try:
            weights, positions = censoring_weights(model, train, test, horizon, pooled)
        except CensoringRefusal as refusal:
            # Refused on positivity: each refused stratum is named with its reason, its
            # modelled censoring survival at the horizon and that value's upper bound.
            yield {"group": "overall", "horizon": horizon, "status": "insufficient_support",
                   "reason": str(refusal),
                   "strata": [dict(s, **{key: s[key] if reportable(s["ancestry"]) else None
                                         for key in ("censoring_survival", "censoring_survival_upper")})
                              for s in refusal.strata]}
            continue
        except ValueError as error:
            yield {"group": "overall", "horizon": horizon,
                   "status": "insufficient_support", "reason": str(error)}
            continue
        if pooled:
            # The horizon is reported, with the strata that lack their own reverse
            # Kaplan-Meier support named beside it: their censoring rests on the model
            # with little of their own data there. Under the right censoring model a
            # stratum's mean IPCW weight is one in expectation; well below one, its
            # censoring curve is too high at the horizon and its cells are biased low.
            # The mean weight and the stratum's censoring level, its hazard ratio over
            # its whole follow-up, are withheld under the reporting minimum.
            strata = []
            for ancestry, reason in sorted(pooled.items()):
                mask = test.ancestry.astype(str).eq(ancestry).to_numpy()
                stratum = model["strata"].get(ancestry)
                strata.append({"ancestry": ancestry, "reason": reason,
                               "ipcw_weight_mass": float(weights[mask].mean())
                               if mask.sum() >= min_count else None,
                               "censoring_hazard_ratio": float(stratum["level"])
                               if stratum is not None and reportable(ancestry) else None})
            yield {"group": "overall", "horizon": horizon, "status": "pooled_censoring",
                   "strata": strata}
        y = ((test.event_code == 1) & (test.followup <= horizon)).to_numpy(float)
        groups = audit_groups(train, test)
        # Fixed probability intervals, not test-outcome-selected bins.
        for lo, hi in [(0, .01), (.01, .05), (.05, .1), (.1, .2), (.2, 1.0000001)]:
            groups.append((f"risk:{lo:g}-{min(hi, 1):g}", (risk[:, j] >= lo) & (risk[:, j] < hi)))
        for label, mask in groups:
            n = int(mask.sum())
            known = weights[mask] > 0
            disease = int(np.sum(y[mask][known]))
            noncase = int(known.sum()) - disease
            if min(n, disease, noncase) < min_count:
                yield {"group": label, "horizon": horizon, "status": "insufficient_support"}
                continue
            def censoring_se(values, mask=mask, positions=positions):
                contribution = np.zeros(len(test))
                contribution[mask] = values
                return censoring_standard_error(model, test, positions, contribution, int(mask.sum()))
            yield j, horizon, label, mask, weights[mask], y[mask], censoring_se


def interval_bounds(name, mean, test_standard_error, censoring_standard_error):
    """The 95% interval of a cell's value from its test-sampling and censoring standard errors."""
    half = 1.959963984540054 * float(np.hypot(test_standard_error, censoring_standard_error))
    return {f"{name}_95_lower": mean - half, f"{name}_95_upper": mean + half}


def evaluate(train, test, risk, horizons, min_count):
    rows = []
    for cell in evaluation_cells(train, test, risk, horizons, min_count):
        if isinstance(cell, dict):
            rows.append(cell)
            continue
        j, horizon, label, mask, w, target, censoring_se = cell
        p = risk[mask, j]
        groups = test.loc[mask, "split_group"]
        brier = loss_summary(w * (target - p)**2, groups)
        observed = loss_summary(w * target, groups)
        brier_censoring, observed_censoring = censoring_se(w * (target - p)**2), censoring_se(w * target)
        rows.append({"group": label, "horizon": horizon, "status": "ok", "n": int(mask.sum()),
                     "observed_disease_events": int(np.sum(target[w > 0])),
                     **brier, "brier_censoring_standard_error": brier_censoring,
                     **interval_bounds("brier", brier["brier"], brier["brier_standard_error"], brier_censoring),
                     "ipcw_auc": weighted_auc(p, target, w),
                     "mean_predicted_risk": float(p.mean()),
                     "ipcw_observed_risk": observed["brier"],
                     "ipcw_observed_risk_standard_error": observed["brier_standard_error"],
                     "ipcw_observed_risk_censoring_standard_error": observed_censoring,
                     **interval_bounds("ipcw_observed_risk", observed["brier"], observed["brier_standard_error"],
                                       observed_censoring),
                     "mean_risk_discrepancy": float(p.mean() - observed["brier"]),
                     "ipcw_weight_n_eff": float(w.sum()**2 / np.sum(w**2))})
    return rows


def incremental_value(train, test, risk_full, risk_null, horizons, min_count):
    """Paired comparison of the score model against the no-score model on the
    same held-out participants: the Brier difference with its group-robust and
    censoring standard errors and 95% interval, and the IPCW AUC difference. Cells
    follow the full model."""
    rows = []
    for cell in evaluation_cells(train, test, risk_full, horizons, min_count):
        if isinstance(cell, dict):
            rows.append(cell)
            continue
        j, horizon, label, mask, w, target, censoring_se = cell
        p_full, p_null = risk_full[mask, j], risk_null[mask, j]
        difference = w * ((target - p_full)**2 - (target - p_null)**2)
        paired = loss_summary(difference, test.loc[mask, "split_group"])
        difference_censoring = censoring_se(difference)
        rows.append({"group": label, "horizon": horizon, "status": "ok", "n": int(mask.sum()),
                     "brier_difference": paired["brier"],
                     "brier_difference_standard_error": paired["brier_standard_error"],
                     "brier_difference_censoring_standard_error": difference_censoring,
                     **interval_bounds("brier_difference", paired["brier"], paired["brier_standard_error"],
                                       difference_censoring),
                     "auc_difference": weighted_auc(p_full, target, w) - weighted_auc(p_null, target, w)})
    return rows


def predict_bundle(directory, baseline_data, times):
    """Replay both frozen components on new baseline observations, without fitting."""
    import gamfit
    directory = Path(directory)
    spec = json.loads((directory / "spec.json").read_text())
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (times < 0).any() or (np.diff(times) <= 0).any():
        raise ValueError("prediction times must be finite, nonnegative and increasing")
    columns = baseline_columns(spec)
    data = baseline_data[columns].copy()
    data["entry"], data["followup"], data["event"] = 0., times[-1], 0
    data["PGS"] = baseline_data.PGS
    model = gamfit.load(directory / "model.gamfit")
    return np.asarray(model.predict(data).cumulative_hazard_at(times))


def fit_worker(frame_path, config_path, cause, output, transform_path=None, variant="pc_varying"):
    import gamfit
    config = json.loads(Path(config_path).read_text())
    df = pd.read_parquet(frame_path)
    pc_cols = [f"PC{i + 1}" for i in range(config["num_pcs"])]
    columns = ["entry", "followup", "age0", "sex", *pc_cols]
    data = df[columns].copy()
    data["PGS"] = df.PGS
    data["event"] = (df.event_code == cause).astype(int)
    train = data.loc[df.is_train].copy()
    test = data.loc[~df.is_train].copy()
    pc_args = ", ".join(pc_cols)
    baseline = f"s(age0, k=8) + sex + duchon({pc_args}, centers={config['baseline_centers']}, scale_dims=true)"
    slope = f"1 + duchon({pc_args}, centers={config['slope_centers']}, scale_dims=true)"
    native = getattr(gamfit, "_rust", None)
    if hasattr(native, "set_log_level"):
        # Solver progress stays in the private fit log; the workspace
        # diagnostic reduces it to fixed categories.
        native.set_log_level("info")
    print("worker_fit_started", flush=True)
    # Both variants centre their baseline time basis at the same follow-up
    # time, so the smoothing selection sees the same frame; otherwise each
    # likelihood picks its own anchor and their early hazards differ.
    anchor = {} if config["survival_time_anchor"] is None else {"survival_time_anchor": float(config["survival_time_anchor"])}
    knots = config["time_num_internal_knots"]
    if variant == "no_score":
        # The same baseline hazard model without any score term: the
        # comparator that prices what the polygenic score adds. The solver's
        # baseline integration can fail to converge with a flexible time
        # basis on some partitions; the comparator then retries with fewer
        # internal knots and records the basis it used.
        while True:
            try:
                model = gamfit.fit(train, f"Surv(entry, followup, event) ~ {baseline}",
                                   survival_likelihood="transformation",
                                   config={"time_num_internal_knots": knots},
                                   persistent_warm_start_root=output / "warm", **anchor)
                break
            except Exception as error:
                if not type(error).__name__.endswith("IntegrationError") or knots <= 2:
                    raise
                print(f"worker_comparator_retry knots={knots}->{knots - 2}", flush=True)
                knots -= 2
        model.save(output / "model.gamfit")
        print("worker_fit_saved", flush=True)
    elif config["score_law"] == "declared_empirical":
        # The score enters as given, and the index is anchored on the weighted
        # empirical law of these training rows' scores: the same rows and
        # eligibility the outcome model sees. No transform is fitted.
        model = gamfit.fit(train, f"Surv(entry, followup, event) ~ {baseline}",
                           survival_likelihood="marginal-slope",
                           z_column="PGS",
                           slope_formula=slope,
                           config={"time_num_internal_knots": config["time_num_internal_knots"],
                                   "latent_measure": "global-empirical"},
                           persistent_warm_start_root=output / "warm", **anchor)
        model.save(output / "model.gamfit")
        print("worker_fit_saved", flush=True)
    else:
        transformer = gamfit.load(transform_path)
        model = gamfit.fit(train, f"Surv(entry, followup, event) ~ {baseline}",
                           survival_likelihood="marginal-slope",
                           transformation_normal_stage1=transformer,
                           slope_formula=slope,
                           config={"time_num_internal_knots": config["time_num_internal_knots"]},
                           persistent_warm_start_root=output / "warm", **anchor)
        model.save(output / "model.gamfit")
        print("worker_fit_saved", flush=True)
        replay_z = model.transformation_score(test)
        if not np.allclose(df.loc[~df.is_train, "Z_ctn"], replay_z, rtol=1e-8, atol=1e-10):
            raise ValueError("saved native CTN disagrees with held-out score artifact")
    horizons = np.asarray(config["horizons_years"], dtype=float)
    # Every horizon is an exact grid point and the spacing stays uniform:
    # inserting a horizon between two linspace points leaves an interval of
    # about a day, on which the cumulative-hazard quadrature cannot converge.
    edges = np.concatenate([[0.0], horizons])
    coarse = np.unique(np.concatenate([
        np.linspace(a, b, max(2, int(round(config["grid_intervals"] * (b - a) / horizons[-1])) + 1))
        for a, b in zip(edges[:-1], edges[1:])]))
    grid = np.sort(np.r_[coarse, (coarse[:-1] + coarse[1:]) / 2])
    # Do not send held-out event/censoring observations as prediction features.
    test["event"] = 0
    test["followup"] = horizons[-1]
    prediction = model.predict(test)
    print("worker_grid_started", flush=True)
    h = np.asarray(prediction.cumulative_hazard_at(grid))
    print("worker_grid_complete", flush=True)
    payload = json.loads((output / "model.gamfit").read_text())["payload"]
    if variant != "no_score" and (payload["latent_z_rank_int_calibration"] is not None
                                  or payload["latent_z_conditional_calibration"] is not None):
        raise ValueError("outcome fit changed the frozen latent score")
    declared = variant != "no_score" and config["score_law"] == "declared_empirical"
    if declared and (payload.get("latent_measure") or {}).get("kind") != "global-empirical":
        raise ValueError("outcome fit did not anchor on the declared latent law")
    if variant == "no_score":
        score_spec = {"slope": None, "kind": "no_score", "normalizer": None, "score_path": "no score term"}
    elif declared:
        score_spec = {"slope": slope, "kind": "pc_varying", "score_law": "declared_empirical", "normalizer": None,
                      "score_path": "raw score as given; anchored on the weighted empirical law of the training rows"}
    else:
        score_spec = {"slope": slope, "kind": "pc_varying", "score_law": "reference_ctn_gaussian",
                      "normalizer": "ctn",
                      "score_path": "external reference CTN declared standard normal; frozen deployment transform"}
    write_json(output / "spec.json", {"baseline": baseline, "cause": cause, "num_pcs": config["num_pcs"],
                                      "time_num_internal_knots": knots,
                                      "orthogonality_claim": False, **score_spec})
    replayed = predict_bundle(output, df.loc[~df.is_train], grid)
    if h.shape != (len(test), len(grid)) or not np.allclose(h, replayed, rtol=1e-7, atol=1e-9):
        raise ValueError("combined transform/outcome save/load predictions disagree")
    held = df.loc[~df.is_train]
    small = np.arange(min(3, len(held)))
    batch = predict_bundle(output, held.iloc[small[::-1]], grid)
    alone = predict_bundle(output, held.iloc[[0]], grid)
    if not (np.allclose(batch, h[small[::-1]], rtol=1e-7, atol=1e-9)
            and np.allclose(alone, h[[0]], rtol=1e-7, atol=1e-9)):
        raise ValueError("prediction changes with batch composition or row ordering")
    np.savez(output / "hazards.npz", hazards=h, grid=grid, coarse=coarse)
    print("worker_validation_complete", flush=True)


def solver_threads():
    return int(os.environ.get("RAYON_NUM_THREADS") or os.cpu_count() or 1)


def signal_session(child, signum):
    """Signal the session a worker leads; its pid is also its process group id.

    killpg(1) is kill(-1), which reaches every process this account owns on
    the node, so refuse any id a real child could not have.
    """
    pid = int(child.pid)
    if pid <= 1:
        raise ValueError(f"refusing to signal process group {pid}")
    os.killpg(pid, signum)


def bounded_fits(jobs, timeout_seconds, checkpoint_callback=None, threads=None):
    """Run fit workers side by side under one wall bound; every child is reaped.

    `jobs` pairs each command with its private log. The checkpoint callback
    runs about every 30 seconds while any worker is alive, and one failing
    worker stops the others at once. A worker killed by a signal (a
    preempted or oversubscribed VM) is restarted once from its persistent
    warm start; a worker that exits with an error is not. `threads` gives
    each worker its own solver thread count, in job order, instead of the
    inherited count. Returns the exit codes in job order.
    """
    def terminate(signum, frame):
        raise InterruptedError("fit controller was terminated")
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    envs = ([None] * len(jobs) if threads is None else
            [dict(os.environ, RAYON_NUM_THREADS=str(count)) for count in threads])
    handles = [Path(log).open("ab") for _, log in jobs]
    children = []
    results = {}
    restarted = set()
    try:
        for (command, _), handle, env in zip(jobs, handles, envs):
            children.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                             start_new_session=True, env=env))
        deadline = started + timeout_seconds
        published = started
        while True:
            now = time.monotonic()
            if now >= deadline:
                raise subprocess.TimeoutExpired([command for command, _ in jobs], timeout_seconds)
            for index, child in enumerate(children):
                if index not in results and child.poll() is not None:
                    if child.returncode < 0 and index not in restarted:
                        restarted.add(index)
                        handles[index].write(f"worker_restarted_after_signal {-child.returncode}\n".encode())
                        handles[index].flush()
                        children[index] = subprocess.Popen(jobs[index][0], stdout=handles[index],
                                                           stderr=subprocess.STDOUT,
                                                           start_new_session=True, env=envs[index])
                        continue
                    results[index] = child.returncode
            if any(code != 0 for code in results.values()):
                raise RuntimeError("fit failed; inspect the private task fit log")
            if len(results) == len(children):
                break
            if checkpoint_callback and now - published >= 30:
                checkpoint_callback()
                published = time.monotonic()
            time.sleep(min(0.5, max(0., deadline - time.monotonic())))
    except BaseException:
        for child in children:
            if child.poll() is None:
                signal_session(child, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_session(child, signal.SIGKILL)
                child.wait()
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        for handle in handles:
            handle.close()
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        elapsed = max(time.monotonic() - started, 1e-9)
        cpu = (usage_after.ru_utime + usage_after.ru_stime
               - usage_before.ru_utime - usage_before.ru_stime)
        allotted = [solver_threads()] * len(jobs) if threads is None else list(threads)
        for index, ((_, log), own) in enumerate(zip(jobs, allotted)):
            write_json(Path(log).with_suffix(".resources.json"), {
                "wall_seconds": elapsed, "cpu_seconds": cpu,
                "average_cpu_cores": cpu / elapsed, "concurrent_fits": len(jobs),
                "solver_threads": own, "allotted_threads": sum(allotted),
                "exit_code": results.get(index), "restarted_after_signal": index in restarted,
            })
    return [results.get(index) for index in range(len(jobs))]


def bounded_fit(command, timeout_seconds, log, checkpoint_callback=None):
    bounded_fits([(command, log)], timeout_seconds, checkpoint_callback)


def checkpointed_fits(jobs, timeout_seconds, checkpoint, threads=None):
    """Retain private failure logs and partial fits without a completion receipt,
    and say in a fixed label whether a worker errored or was killed."""
    try:
        bounded_fits(jobs, timeout_seconds, checkpoint_callback=checkpoint.publish, threads=threads)
    except BaseException as error:
        checkpoint.publish()
        if isinstance(error, RuntimeError):
            codes = [json.loads(Path(log).with_suffix(".resources.json").read_text()).get("exit_code")
                     for _, log in jobs if Path(log).with_suffix(".resources.json").is_file()]
            label = ("failed_fit_worker_signal" if any(code is not None and code < 0 for code in codes)
                     else "failed_fit_worker_error")
            try:
                publish_status(f"gs://{checkpoint.bucket}/{checkpoint.object}", label)
            except Exception:
                pass
        raise


def checkpointed_fit(command, timeout_seconds, log, checkpoint):
    checkpointed_fits([(command, log)], timeout_seconds, checkpoint)


def analyze_partition(df, config, args, disease_dir, checkpoint, pgs):
    """Fit the prespecified disease/death components; never select a model."""
    disease_dir.mkdir(parents=True, exist_ok=True)
    slug = disease_dir.name
    train, test = df.loc[df.is_train], df.loc[~df.is_train]
    support_check = fit_support if args.smoke_only else partition_support
    errors = support_check(train, test, config)
    if errors:
        raise ValueError("; ".join(errors))
    groups = audit_groups(train, test)
    transform_args = []
    if config["score_law"] == "declared_empirical":
        # The law is declared from the training rows the outcome model fits;
        # its adequacy within each held-out stratum is reported, not assumed.
        diagnostics = {"declared_law": declared_law_diagnostics(
            train.PGS.to_numpy(), test.PGS.to_numpy(), groups, config["min_report_count"])}
        publish_status(args.checkpoint_uri, "score_law_declared")
    else:
        model, transform_path, manifest = load_reference(
            args.reference_ctn, disease_dir / "reference_ctn",
            pgs, config["num_pcs"], config["projection_model_sha256"])
        publish_status(args.checkpoint_uri, "applying_reference_ctn")
        df["Z_ctn"] = transformed_score(model, "ctn", df, config["num_pcs"])
        publish_status(args.checkpoint_uri, "score_transform_ready")
        diagnostics = {"ctn": score_diagnostics(
            df.loc[~df.is_train, "Z_ctn"].to_numpy(), groups, config["min_report_count"]),
            "reference": manifest}
        transform_args = ["--transform-model", str(transform_path)]
    frame = disease_dir / "transformed.parquet"
    df.to_parquet(frame, index=False)
    variants = ["pc_varying"] + (["no_score"] if config["fit_no_score_comparator"] else [])
    pending = []
    for variant in variants:
        for cause in (1, 2):
            fit_dir = disease_dir / f"{variant}_{cause}"
            fit_dir.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "fit",
                       "--frame", str(frame), "--config", str(args.config.resolve()),
                       "--cause", str(cause), "--output", str(fit_dir),
                       *transform_args, "--variant", variant]
            if not checkpoint.step_is_complete(fit_dir, model=True):
                pending.append((variant, cause, fit_dir, command))
    if pending:
        # Every fit runs side by side, so the wall time is the slowest fit
        # rather than the sum, and every fit gets an equal share of the solver
        # threads. With a release engine the disease fits are as slow as the
        # death fits: giving death three quarters of 16 threads left the
        # 2-thread disease fits unfinished at 900 s where equal shares finished
        # the stage in 500 s (gnomon#2338, 6,000 training rows).
        for variant, cause, _, _ in pending:
            print(f"Fitting {slug}: {variant} cause {cause}", flush=True)
            if variant == "pc_varying":
                publish_status(args.checkpoint_uri, "fitting_disease" if cause == 1 else "fitting_death")
        threads = [max(1, solver_threads() // len(pending))] * len(pending)
        checkpointed_fits([(command, fit_dir / "fit.log") for _, _, fit_dir, command in pending],
                          config["fit_timeout_seconds"], checkpoint, threads=threads)
        for _, _, fit_dir, _ in pending:
            checkpoint.complete_step(fit_dir, ["hazards.npz", "model.gamfit", "spec.json"], model=True)
    risks, models = {}, {}
    for variant in variants:
        hazards = []
        for cause in (1, 2):
            with np.load(disease_dir / f"{variant}_{cause}" / "hazards.npz") as saved:
                hazards.append(saved["hazards"])
                grid, coarse = saved["grid"], saved["coarse"]
        fine_cif = cif_from_hazards(hazards)
        coarse_indices = np.searchsorted(grid, coarse)
        coarse_cif = cif_from_hazards(np.asarray(hazards)[:, :, coarse_indices])
        error = float(np.max(np.abs(fine_cif[:, :, coarse_indices] - coarse_cif)))
        if error > 0.001:
            raise ValueError("CIF grid refinement differs by >0.001; increase grid_intervals")
        indices = np.searchsorted(grid, config["horizons_years"])
        risks[variant] = fine_cif[0][:, indices]
        models[variant] = {"cif_grid_error": error,
                           "time_num_internal_knots": [
                               json.loads((disease_dir / f"{variant}_{cause}" / "spec.json").read_text())
                               .get("time_num_internal_knots") for cause in (1, 2)],
                           "metrics": evaluate(train, test, risks[variant], config["horizons_years"],
                                               config["min_report_count"])}
    report = {"models": models, "score_diagnostics": diagnostics}
    if "no_score" in risks:
        report["incremental"] = incremental_value(train, test, risks["pc_varying"], risks["no_score"],
                                                  config["horizons_years"], config["min_report_count"])
    return report


def analyze_development(df, disease, config, args, disease_dir, checkpoint):
    """Fit the one prespecified score using development observations only."""
    development = development_partition(df, config)
    pgs, = disease["candidates"]
    development["PGS"] = development[pgs]
    report = analyze_partition(development, config, args,
        disease_dir / "development" / pgs, checkpoint, pgs)
    return {pgs: report}


def run(args):
    from aou_identity import task_account
    execution_account = task_account()
    from disease_selection import select_runtime_diseases
    config = json.loads(args.config.read_text())
    validate_config(config)
    if solver_threads() != config["fit_budget"]["solver_threads"]:
        raise ValueError("the fit budget was measured at a different solver thread count")
    endpoint = json.loads(args.endpoint_config.read_text())
    panel = load_score_panel(args.score_panel, exploratory=args.smoke_only or args.prepare_only,
                             endpoints=[endpoint] if endpoint else None)
    import gamfit
    if gamfit.__version__ != config["gamfit_version"]:
        raise ValueError("gamfit version does not match the analysis configuration")
    engine_hash = digest(importlib.util.find_spec("gamfit._rust").origin)
    if config["fit_budget"]["engine_sha256"] != engine_hash:
        # A version names source, not a build: a dev-profile wheel of the same
        # version fits many times slower, so only the measured binary counts.
        raise ValueError("the fit budget was not measured with this gamfit engine build")
    declared = config["score_law"] == "declared_empirical"
    if declared == bool(args.reference_ctn):
        raise ValueError("stage reference CTN archives exactly when the analysis declares a reference CTN score")
    if not args.prepare_only and not declared:
        # Validate all requested external models before accessing cohort data.
        requested = [panel["endpoints"][endpoint]] if endpoint else panel["endpoints"].values()
        for disease in requested:
            for pgs in disease["candidates"]:
                load_reference(args.reference_ctn, args.output / "references" / pgs,
                               pgs, config["num_pcs"], config["projection_model_sha256"])
    image = json.loads(args.runtime_image.read_text())
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("runtime_image must use an immutable digest")
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ("aou_identity.py", "aou_score_transform.py", "aou_checkpoint.py",
                "aou_evaluation.py", "aou_status.py", "disease_selection.py", "reference_ctn.py")]]
    signature = {"config": result_identity(config), "endpoint": endpoint,
                 "score_scope": "prespecified_single_score",
                 "sources": {p.name: digest(p) for p in sources},
                 "inputs": {key: digest(getattr(args, key)) for key in
                            ("scores", "ancestry", "prune", "phenotypes", "score_panel")},
                 "reference_ctn": [digest(path) for path in args.reference_ctn]}
    checkpoint = StudyCheckpoint(args.output, args.checkpoint_uri, config["google_project"],
                                 execution_account, signature, args.resume,
                                 engine_hash=engine_hash,
                                 resume_latest=args.resume_latest)
    prepared_path = args.output / "prepared.json"
    client = BoundedClient(config, execution_account)
    if prepared_path.exists():
        prepared = json.loads(prepared_path.read_text())
        diseases = prepared["diseases"]
        for slug in diseases:
            if diseases[slug].get("missing_scores"):
                continue
            if not checkpoint.step_is_complete(args.output / slug):
                raise ValueError("prepared cohort checkpoint is incomplete")
    else:
        phenotypes = unpack_phenotypes(args.phenotypes, args.output / "phenotypes")
        diseases = select_runtime_diseases(client, config["workspace_cdr"], config["top_n_diseases"], phenotypes)
        if not diseases:
            raise ValueError("the existing disease selector found no eligible mapped disease")
        diseases = {slug: {**disease, "candidates": panel["endpoints"][slug]["candidates"]}
                    for slug, disease in diseases.items() if slug in panel["endpoints"]}
        if endpoint:
            if endpoint not in diseases:
                raise ValueError("requested endpoint did not pass the existing disease selection rule")
            diseases = {endpoint: diseases[endpoint]}
        diseases = dict(list(diseases.items())[:config["disease_limit"]])
        if not diseases:
            raise ValueError("no prespecified endpoint passed the existing disease selection rule")
        projection_path = args.output / "projection_pcs.parquet"
        score_cache = unpack_score_cache(args.scores, args.output / "scores.tar", projection_path)
        available_scores = cached_score_ids(score_cache)
        for disease in diseases.values():
            required = disease["candidates"]
            disease["missing_scores"] = sorted(set(required) - available_scores)
            for pgs in disease["missing_scores"]:
                publish_status(args.checkpoint_uri, "missing_" + pgs.lower())
        publish_status(args.checkpoint_uri, "reading_ancestry")
        ancestry = read_ancestry(args.ancestry, args.prune, config["num_pcs"], projection_path)
        publish_status(args.checkpoint_uri, "loading_person_times")
        base = ancestry.merge(person_times(client, config["workspace_cdr"]), on="person_id", validate="one_to_one")
        publish_status(args.checkpoint_uri, "preparing_cohort")
        for slug, disease in diseases.items():
            if disease["missing_scores"]:
                continue
            scores = endpoint_scores(score_cache, disease)
            cases = case_dates(client, config["workspace_cdr"], disease["concept_id"])
            df = build_cohort(base, scores, cases, config)
            disease_dir = args.output / slug
            disease_dir.mkdir(exist_ok=True)
            df.to_parquet(disease_dir / "cohort.parquet", index=False)
            checkpoint.complete_step(disease_dir, ["cohort.parquet"])
        prepared = {"diseases": diseases, "query_job_ids": [job.job_id for job in client.jobs]}
        write_json(prepared_path, prepared)
        checkpoint.publish()
    results = {}
    for slug, disease in diseases.items():
        print(f"Preparing {slug} ({', '.join(disease['candidates'])})", flush=True)
        if disease["missing_scores"]:
            publish_status(args.checkpoint_uri, "scores_missing")
            if not args.prepare_only:
                raise ValueError(f"{slug}: prespecified scores missing from pgsEngine cache: {disease['missing_scores']}")
            results[slug] = {"status": "missing_scores", "missing_scores": disease["missing_scores"]}
            continue
        disease_dir = args.output / slug
        frame = disease_dir / "cohort.parquet"
        df = pd.read_parquet(frame)
        train, test = df.loc[df.is_train], df.loc[~df.is_train]
        support_errors = [] if args.smoke_only else partition_support(train, test, config)
        development = development_partition(df, config)
        support_errors += [f"development: {error}" for error in partition_support(
            development.loc[development.is_train], development.loc[~development.is_train], config)]
        if args.prepare_only:
            publish_status(args.checkpoint_uri, "cohort_unsupported" if support_errors else "cohort_ready")
            counts = {"training": len(train), "held_out": len(test),
                      "training_disease_events": int((train.event_code == 1).sum()),
                      "training_deaths": int((train.event_code == 2).sum()),
                      "same_day_disease_death": int(df.disease_death_same_day.sum())}
            results[slug] = {"status": "unsupported" if support_errors else "cohort_ready",
                             "candidates": disease["candidates"], "support_errors": support_errors,
                             "counts": {key: value if value >= config["min_report_count"] else None
                                        for key, value in counts.items()}}
            continue
        required_errors = (fit_support(development.loc[development.is_train],
                                       development.loc[~development.is_train], config)
                           if args.smoke_only else support_errors)
        if required_errors:
            publish_status(args.checkpoint_uri, "cohort_unsupported")
            raise ValueError("; ".join(required_errors))
        if support_errors:
            publish_status(args.checkpoint_uri, "evaluation_unsupported")
        publish_status(args.checkpoint_uri, "cohort_ready")
        development_reports = analyze_development(df, disease, config, args, disease_dir, checkpoint)
        if args.smoke_only:
            results[slug] = {"status": "development_smoke_completed",
                             "prespecified_pgs": disease["candidates"][0],
                             "evaluation_support_errors": support_errors,
                             "development": development_reports}
            checkpoint.publish()
            publish_status(args.checkpoint_uri, "smoke_completed")
            continue
        selected = disease["candidates"][0]
        choice = {"selected_pgs": selected,
                  "criterion": "prespecified before development; no score or model search"}
        write_json(disease_dir / "score_selection.json", choice)
        checkpoint.publish()
        df["PGS"] = df[selected]
        report = analyze_partition(df, config, args, disease_dir / "final", checkpoint, selected)
        results[slug] = {"concept_id": disease["concept_id"], "score_selection": choice,
                         "evaluation_support_errors": support_errors,
                         "development": development_reports, **report}
    write_json(args.output / "metrics.json", results)
    write_json(args.output / "provenance.json", {
        "config": config, "fit_budget_rows": fit_budget_rows(config),
        "runtime_image": image, "gamfit_build": gamfit.build_info(),
        "requested_endpoint": endpoint,
        "execution_account": execution_account,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "input_sha256": {key: digest(getattr(args, key)) for key in ["config", "scores", "ancestry", "prune", "phenotypes"]},
        "runner_sha256": digest(__file__), "selector_sha256": digest(Path(__file__).with_name("disease_selection.py")),
        "query_job_ids": prepared["query_job_ids"],
        "analysis_status": ("cohort_only" if args.prepare_only else
                            "development_smoke" if args.smoke_only else "completed"),
        "target": "first qualifying recorded disease after primary consent, competing death",
        "same_day_event_rule": "recorded diagnosis takes precedence over death; no within-day order inferred",
        "baseline": f"landmark {config['landmark_days']} days after the AoU primary-consent date "
                    "(Consent PII Module descendants); continuous EHR lookback before consent; "
                    "participants diagnosed, dead or lost before the landmark are excluded",
        "comparator": ("the same hazard model without any score term, fitted alongside; "
                       "incremental value is the paired IPCW Brier difference with a group-robust "
                       "standard error and the IPCW AUC difference"
                       if config["fit_no_score_comparator"] else "none"),
        "validation": "group holdout after published relatedness prune; 75/25 development split for the one prespecified model; outer test remains locked during development",
        "score_panel": panel,
        "score_law": config["score_law"],
        "score_transform": ("none; the raw score is anchored on the weighted empirical law of the training rows"
                            if config["score_law"] == "declared_empirical" else
                            "externally fitted PC-conditional CTN declared standard normal; frozen latent scores"),
        "reference_ctn_sha256": [digest(path) for path in args.reference_ctn],
        "orthogonality_claim": False,
        "uncertainty": "group-robust test-sampling standard errors conditional on fitted models, and the "
                       "censoring model's delta-method standard error; 95% intervals combine both",
        "censoring_model": "training-only two-level Poisson-gamma censoring model by reported genetic ancestry "
                           "over the pooled reverse Kaplan-Meier, 20 intervals of equal pooled censoring mass: "
                           "level R = (O + 1) / (E + 1), interval ratio (O_j + R) / (E_j + 1), survival "
                           "prod (1 - dL)^ratio; a horizon is refused where the upper one-sided 95% bound on a "
                           "stratum's modelled censoring survival is below 0.05, each refused stratum named with "
                           "its reason, survival and bound; strata without their own reverse Kaplan-Meier support "
                           "are named in the horizon's pooled_censoring row with their mean IPCW weight and level",
    })
    checkpoint.publish()
    if not args.prepare_only and not args.smoke_only:
        publish_status(args.checkpoint_uri, "analysis_completed")
    print("Completed bounded pilot; aggregate metrics and provenance written", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    for name in ["config", "phenotypes", "scores", "ancestry", "prune", "runtime-image", "score-panel", "endpoint-config", "output", "reference-ctn-list"]:
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--checkpoint-uri", required=True)
    run_parser.add_argument("--resume", type=Path)
    run_parser.add_argument("--resume-latest", action="store_true")
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--smoke-only", action="store_true")
    fit_parser = sub.add_parser("fit")
    for name in ["frame", "config", "output"]:
        fit_parser.add_argument(f"--{name}", type=Path, required=True)
    fit_parser.add_argument("--transform-model", type=Path)
    fit_parser.add_argument("--cause", type=int, choices=[1, 2], required=True)
    fit_parser.add_argument("--variant", choices=["pc_varying", "no_score"], default="pc_varying")
    args = parser.parse_args()
    if args.command == "run":
        reference_paths = json.loads(args.reference_ctn_list.read_text())
        if not isinstance(reference_paths, list) or not all(isinstance(p, str) for p in reference_paths):
            parser.error("reference CTN list must be a list of staged model archive paths")
        args.reference_ctn = [Path(path) for path in reference_paths]
        try:
            run(args)
        except Exception as error:
            try:
                publish_status(args.checkpoint_uri, failure_label(error))
            except Exception:
                print("Could not publish the fixed failure label; inspect workspace logs", file=sys.stderr)
            raise
    else:
        fit_worker(args.frame, args.config, args.cause, args.output, args.transform_model, args.variant)


if __name__ == "__main__":
    main()
