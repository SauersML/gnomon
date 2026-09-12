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

from aou_score_transform import (baseline_columns,
                                 score_diagnostics, transformed_score)
from reference_ctn import load_reference
from aou_checkpoint import StudyCheckpoint
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
        "lookback_days", "projection_model_sha256",
    }
    if set(c) != expected:
        raise ValueError("analysis configuration has missing or unknown keys")
    if not re.fullmatch(r"[a-z][a-z0-9-]+\.[A-Za-z0-9_]+", c["workspace_cdr"]):
        raise ValueError("workspace_cdr must be a concrete project.dataset")
    if not re.fullmatch(r"[a-z][a-z0-9-]+", c["google_project"]):
        raise ValueError("google_project must be a concrete billing project")
    positive = expected - {"google_project", "workspace_cdr", "gamfit_version",
                           "train_fraction", "seed", "horizons_years", "projection_model_sha256"}
    if any(type(c[k]) is not int or c[k] <= 0 for k in positive):
        raise ValueError("resource and sample budgets must be positive integers")
    if not 1 <= c["num_pcs"] <= 16 or not 0.5 <= c["train_fraction"] <= 0.9:
        raise ValueError("unsupported PC count or training fraction")
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


def load_score_panel(path, *, exploratory):
    panel = json.loads(Path(path).read_text())
    for endpoint in panel["endpoints"].values():
        candidates = endpoint["candidates"]
        if len(candidates) != 1:
            raise ValueError("each endpoint requires exactly one prespecified score")
        for pgs in candidates:
            if pgs == "PGS004787" or pgs in panel["excluded"] or pgs not in panel["scores"]:
                raise ValueError("excluded or unlisted PGS in candidate panel")
            if not exploratory:
                audit = panel["scores"][pgs].get("development_audit", {})
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
            ipcw_weights(train, test, horizon)
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
    df["age0"] = (df.baseline - df.birth_date).dt.days / 365.25
    eligible = (df.sex.notna() & df.age0.ge(18) & df.obs_end.gt(df.baseline)
                & ((df.baseline - df.obs_start).dt.days >= config["lookback_days"])
                & (df.disease_date.isna() | df.disease_date.gt(df.baseline))
                & (df.death_date.isna() | df.death_date.gt(df.baseline)))
    df = df.loc[eligible].copy()
    # Recorded diagnosis takes precedence for same-day ties. Retain the
    # participant; future event ordering must not select the baseline cohort.
    df["disease_death_same_day"] = (df.disease_date.notna()
        & df.disease_date.eq(df.death_date) & df.disease_date.le(df.obs_end))
    end = df[["obs_end", "disease_date", "death_date"]].min(axis=1)
    df["event_code"] = np.select([df.disease_date.eq(end), df.death_date.eq(end)], [1, 2], default=0)
    df["followup"] = (end - df.baseline).dt.days / 365.25
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


def ipcw_weights(train, test, horizon):
    result = np.zeros(len(test))
    for ancestry in test.ancestry.unique():
        reference = train.loc[train.ancestry == ancestry]
        if len(reference) < 20:
            raise ValueError("insufficient training support for ancestry-specific censoring")
        km_t, km_g = censor_km(reference)
        def g_at(t, side):
            idx = np.searchsorted(km_t, t, side=side)
            return np.r_[1.0, km_g][idx]
        mask = test.ancestry.eq(ancestry).to_numpy()
        t = test.followup.to_numpy(float)
        observed = (test.event_code.to_numpy(int) != 0) & (t <= horizon)
        g_horizon = float(g_at(horizon, "right"))
        if g_horizon < 0.05 or not (reference.followup > horizon).any():
            raise ValueError("evaluation horizon lacks censoring support in an ancestry stratum")
        g_event = g_at(t[mask & observed], "left")
        if (g_event < 0.05).any():
            raise ValueError("event-time censoring weights are unstable")
        result[mask & observed] = 1 / g_event
        result[mask & (t > horizon)] = 1 / g_horizon
    return result


def evaluate(train, test, risk, horizons, min_count):
    rows = []
    for j, horizon in enumerate(horizons):
        try:
            weights = ipcw_weights(train, test, horizon)
        except ValueError as error:
            rows.append({"group": "overall", "horizon": horizon,
                         "status": "insufficient_support", "reason": str(error)})
            continue
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
                rows.append({"group": label, "horizon": horizon, "status": "insufficient_support"})
                continue
            w, target, p = weights[mask], y[mask], risk[mask, j]
            rows.append({"group": label, "horizon": horizon, "status": "ok", "n": n,
                         "observed_disease_events": disease,
                         **loss_summary(w * (target - p)**2, test.loc[mask, "split_group"]),
                         "mean_predicted_risk": float(p.mean()),
                         "ipcw_observed_risk": float(np.mean(w * target)),
                         "mean_risk_discrepancy": float(p.mean() - np.mean(w * target))})
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


def fit_worker(frame_path, config_path, cause, output, transform_path):
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
    print("worker_fit_started", flush=True)
    transformer = gamfit.load(transform_path)
    model = gamfit.fit(train, f"Surv(entry, followup, event) ~ {baseline}",
                       survival_likelihood="marginal-slope",
                       transformation_normal_stage1=transformer,
                       slope_formula=slope,
                       config={"time_num_internal_knots": config["time_num_internal_knots"]},
                       persistent_warm_start_root=output / "warm")
    model.save(output / "model.gamfit")
    print("worker_fit_saved", flush=True)
    replay_z = model.transformation_score(test)
    if not np.allclose(df.loc[~df.is_train, "Z_ctn"], replay_z, rtol=1e-8, atol=1e-10):
        raise ValueError("saved native CTN disagrees with held-out score artifact")
    horizons = np.asarray(config["horizons_years"])
    coarse = np.unique(np.r_[np.linspace(0, horizons[-1], config["grid_intervals"] + 1), horizons])
    grid = np.sort(np.r_[coarse, (coarse[:-1] + coarse[1:]) / 2])
    # Do not send held-out event/censoring observations as prediction features.
    test["event"] = 0
    test["followup"] = horizons[-1]
    prediction = model.predict(test)
    print("worker_grid_started", flush=True)
    h = np.asarray(prediction.cumulative_hazard_at(grid))
    print("worker_grid_complete", flush=True)
    payload = json.loads((output / "model.gamfit").read_text())["payload"]
    if payload["latent_z_rank_int_calibration"] is not None or payload["latent_z_conditional_calibration"] is not None:
        raise ValueError("outcome fit changed the frozen latent score")
    write_json(output / "spec.json", {"baseline": baseline, "slope": slope, "cause": cause,
                                      "kind": "pc_varying", "normalizer": "ctn",
                                      "num_pcs": config["num_pcs"],
                                      "score_path": "external reference CTN; frozen deployment transform",
                                      "orthogonality_claim": False})
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


def bounded_fit(command, timeout_seconds, log, checkpoint_callback=None):
    def terminate(signum, frame):
        raise InterruptedError("fit controller was terminated")
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    with Path(log).open("wb") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = started + timeout_seconds
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    result = child.wait(timeout=min(30, remaining) if checkpoint_callback else remaining)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
                    checkpoint_callback()
        except BaseException:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
            elapsed = time.monotonic() - started
            cpu = (usage_after.ru_utime + usage_after.ru_stime
                   - usage_before.ru_utime - usage_before.ru_stime)
            write_json(Path(log).with_suffix(".resources.json"), {
                "wall_seconds": elapsed, "cpu_seconds": cpu,
                "average_cpu_cores": cpu / elapsed,
            })
        if result != 0:
            raise RuntimeError("fit failed; inspect the private task fit log")


def checkpointed_fit(command, timeout_seconds, log, checkpoint):
    """Retain private failure logs and partial fits without a completion receipt."""
    try:
        bounded_fit(command, timeout_seconds, log, checkpoint_callback=checkpoint.publish)
    except BaseException:
        checkpoint.publish()
        raise


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
    model, transform_path, manifest = load_reference(
        args.reference_ctn, disease_dir / "reference_ctn",
        pgs, config["num_pcs"], config["projection_model_sha256"])
    publish_status(args.checkpoint_uri, "applying_reference_ctn")
    df["Z_ctn"] = transformed_score(model, "ctn", df, config["num_pcs"])
    publish_status(args.checkpoint_uri, "score_transform_ready")
    diagnostics = {"ctn": score_diagnostics(
        df.loc[~df.is_train, "Z_ctn"].to_numpy(), groups, config["min_report_count"]),
        "reference": manifest}
    frame = disease_dir / "transformed.parquet"
    df.to_parquet(frame, index=False)
    hazards = []
    for cause in (1, 2):
        fit_dir = disease_dir / f"pc_varying_ctn_{cause}"
        fit_dir.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "fit",
                   "--frame", str(frame), "--config", str(args.config.resolve()),
                   "--cause", str(cause), "--output", str(fit_dir),
                   "--transform-model", str(transform_path)]
        print(f"Fitting {slug}: cause {cause}", flush=True)
        if not checkpoint.step_is_complete(fit_dir, model=True):
            publish_status(args.checkpoint_uri, "fitting_disease" if cause == 1 else "fitting_death")
            checkpointed_fit(command, config["fit_timeout_seconds"], fit_dir / "fit.log", checkpoint)
            checkpoint.complete_step(fit_dir, ["hazards.npz", "model.gamfit", "spec.json"], model=True)
        with np.load(fit_dir / "hazards.npz") as saved:
            hazards.append(saved["hazards"])
            grid, coarse = saved["grid"], saved["coarse"]
    fine_cif = cif_from_hazards(hazards)
    coarse_indices = np.searchsorted(grid, coarse)
    coarse_cif = cif_from_hazards(np.asarray(hazards)[:, :, coarse_indices])
    error = float(np.max(np.abs(fine_cif[:, :, coarse_indices] - coarse_cif)))
    if error > 0.001:
        raise ValueError("CIF grid refinement differs by >0.001; increase grid_intervals")
    indices = np.searchsorted(grid, config["horizons_years"])
    risk = fine_cif[0][:, indices]
    return {"models": {"pc_varying_ctn": {
        "cif_grid_error": error,
        "metrics": evaluate(train, test, risk, config["horizons_years"], config["min_report_count"])
    }}, "score_diagnostics": diagnostics}


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
    panel = load_score_panel(args.score_panel, exploratory=args.smoke_only or args.prepare_only)
    endpoint = json.loads(args.endpoint_config.read_text())
    if endpoint != "" and endpoint not in panel["endpoints"]:
        raise ValueError("requested endpoint is not in the prespecified panel")
    import gamfit
    if gamfit.__version__ != config["gamfit_version"]:
        raise ValueError("gamfit version does not match the analysis configuration")
    if not args.prepare_only:
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
    signature = {"config": config, "endpoint": endpoint,
                 "score_scope": "prespecified_single_score",
                 "sources": {p.name: digest(p) for p in sources},
                 "inputs": {key: digest(getattr(args, key)) for key in
                            ("scores", "ancestry", "prune", "phenotypes", "score_panel")},
                 "reference_ctn": [digest(path) for path in args.reference_ctn]}
    checkpoint = StudyCheckpoint(args.output, args.checkpoint_uri, config["google_project"],
                                 execution_account, signature, args.resume,
                                 engine_hash=digest(importlib.util.find_spec("gamfit._rust").origin),
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
                         "development": development_reports, **report}
    write_json(args.output / "metrics.json", results)
    write_json(args.output / "provenance.json", {
        "config": config, "runtime_image": image, "gamfit_build": gamfit.build_info(),
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
        "baseline": "AoU primary-consent date (Consent PII Module descendants); continuous EHR lookback",
        "validation": "group holdout after published relatedness prune; 75/25 development split for the one prespecified model; outer test remains locked during development",
        "score_panel": panel,
        "score_transform": "externally fitted PC-conditional CTN; frozen latent scores",
        "reference_ctn_sha256": [digest(path) for path in args.reference_ctn],
        "orthogonality_claim": False,
        "uncertainty": "group-robust Brier intervals conditional on fitted models and training censoring estimates",
        "censoring_model": "training-only reverse Kaplan-Meier stratified by reported genetic ancestry",
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
    fit_parser.add_argument("--transform-model", type=Path, required=True)
    fit_parser.add_argument("--cause", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    if args.command == "run":
        reference_paths = json.loads(args.reference_ctn_list.read_text())
        if not isinstance(reference_paths, list) or not reference_paths or not all(isinstance(p, str) for p in reference_paths):
            parser.error("reference CTN list must contain staged model archive paths")
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
        fit_worker(args.frame, args.config, args.cause, args.output, args.transform_model)


if __name__ == "__main__":
    main()
