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
import signal
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import numpy as np
import pandas as pd

from aou_score_transform import (assemble_scores, baseline_columns, fit_transform,
                                 grouped_folds, score_diagnostics, transformed_score)
from aou_checkpoint import StudyCheckpoint
from aou_evaluation import audit_groups, paired_loss_summary
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
        "lookback_days", "crossfit_folds", "stage1_centers", "stage1_age_k", "stage1_response_knots", "stage1_timeout_seconds",
    }
    if set(c) != expected:
        raise ValueError("analysis configuration has missing or unknown keys")
    if not re.fullmatch(r"[a-z][a-z0-9-]+\.[A-Za-z0-9_]+", c["workspace_cdr"]):
        raise ValueError("workspace_cdr must be a concrete project.dataset")
    if not re.fullmatch(r"[a-z][a-z0-9-]+", c["google_project"]):
        raise ValueError("google_project must be a concrete billing project")
    positive = expected - {"google_project", "workspace_cdr", "gamfit_version",
                           "train_fraction", "seed", "horizons_years"}
    if any(type(c[k]) is not int or c[k] <= 0 for k in positive):
        raise ValueError("resource and sample budgets must be positive integers")
    if not 1 <= c["num_pcs"] <= 16 or not 0.5 <= c["train_fraction"] <= 0.9:
        raise ValueError("unsupported PC count or training fraction")
    if min(c["baseline_centers"], c["slope_centers"]) < 4:
        raise ValueError("smooths require at least four centers")
    if c["time_num_internal_knots"] < 2:
        raise ValueError("survival time basis requires at least two internal knots")
    if not 2 <= c["crossfit_folds"] <= 5 or c["stage1_centers"] < 4 or c["stage1_age_k"] < 4:
        raise ValueError("stage one requires 2–5 folds and basis sizes of at least four")
    if c["stage1_response_knots"] < 2:
        raise ValueError("stage one requires at least two response knots")
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

    def query(self, sql, job_config=None):
        from google.cloud import bigquery
        job_config = job_config or bigquery.QueryJobConfig()
        job_config.maximum_bytes_billed = self.config["maximum_bytes_billed"]
        job_config.job_timeout_ms = self.config["query_timeout_seconds"] * 1000
        job = self.client.query(sql, job_config=job_config)
        self.jobs.append(job)
        try:
            job.result(timeout=self.config["query_timeout_seconds"])
        except BaseException:
            job.cancel()
            raise
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


def read_ancestry(ancestry, prune, num_pcs):
    ancestry_columns = pd.read_csv(ancestry, sep="\t", nrows=0).columns
    if not {"research_id", "pca_features", "ancestry_pred"}.issubset(ancestry_columns):
        raise ValueError("ancestry file lacks required research_id/pca_features/ancestry_pred columns")
    df = pd.read_csv(ancestry, sep="\t", dtype=str,
                     usecols=["research_id", "pca_features", "ancestry_pred"])
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
    arrays = df.pca_features.map(json.loads)
    if not arrays.map(lambda row: isinstance(row, list) and len(row) >= num_pcs).all():
        raise ValueError("invalid AoU pca_features")
    pcs = np.asarray([row[:num_pcs] for row in arrays], dtype=float)
    if not np.isfinite(pcs).all() or df.ancestry_pred.isna().any():
        raise ValueError("non-finite PCs or missing ancestry labels")
    out = pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(num_pcs)])
    out["person_id"] = df.research_id.to_numpy()
    out["ancestry"] = df.ancestry_pred.to_numpy()
    # Published max-IS pruning supplies the independent units for this pilot.
    # Explicit group labels remain the split interface, so related units cannot
    # be separated by inner folds if a broader group mapping is supplied.
    out["split_group"] = out.person_id
    return out


def unpack_score_cache(archive, output):
    """Read pgsEngine's real shared-feature artifact without extracting genotypes."""
    output = Path(output)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix="scores-", suffix=".partial",
                                     delete=False) as target:
        temporary = Path(target.name)
        try:
            count = 0
            # Streaming avoids scanning the multi-GB gzip and then decompressing
            # it again to seek back to the score member. Still inspect the rest
            # of the archive to reject duplicate score members.
            with tarfile.open(archive, "r|gz") as tar:
                for member in tar:
                    if not member.isfile() or Path(member.name).name != "scores.tar":
                        continue
                    count += 1
                    if count > 1:
                        raise ValueError("expected exactly one scores.tar in the pgsEngine shared-feature archive")
                    with tar.extractfile(member) as source:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
            if count != 1:
                raise ValueError("expected exactly one scores.tar in the pgsEngine shared-feature archive")
            target.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def load_cached_score(archive, pgs):
    column = f"{pgs}_AVG"
    result = None
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".sscore"):
                continue
            with tar.extractfile(member) as handle:
                header = handle.readline().decode("utf-8").rstrip().split("\t")
            if column not in header:
                continue
            ids = [name for name in header if name.lstrip("#") == "IID"]
            if len(ids) != 1 or result is not None:
                raise ValueError(f"ambiguous cached score source for {pgs}")
            with tar.extractfile(member) as handle:
                result = pd.read_csv(handle, sep="\t", usecols=[ids[0], column], dtype={ids[0]: str})
            result = result.rename(columns={ids[0]: "person_id", column: "PGS"})
    if result is None:
        raise ValueError(f"selected score {pgs} is absent from scores archive; populate that cache first")
    if result.person_id.isna().any() or result.person_id.duplicated().any():
        raise ValueError("score IDs must be present and unique")
    if not np.isfinite(result.PGS.to_numpy(dtype=float)).all():
        raise ValueError("cached score contains non-finite values")
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


def endpoint_scores(archive, disease, *, primary_only):
    first = disease["candidates"][0]
    scores = load_cached_score(archive, first)
    scores[first] = scores.PGS
    if not primary_only:
        second = disease["candidates"][1]
        scores = scores.merge(load_cached_score(archive, second).rename(columns={"PGS": second}),
                              on="person_id", validate="one_to_one")
    return scores


def load_score_panel(path):
    panel = json.loads(Path(path).read_text())
    for endpoint in panel["endpoints"].values():
        candidates = endpoint["candidates"]
        if len(candidates) != 2 or len(set(candidates)) != 2:
            raise ValueError("each endpoint requires exactly two prespecified scores")
        for pgs in candidates:
            if pgs == "PGS004787" or pgs in panel["excluded"] or pgs not in panel["scores"]:
                raise ValueError("excluded or unaudited PGS in candidate panel")
    return panel


def development_partition(df, config):
    """Outer-test rows are removed before score/model selection can see data."""
    development = df.loc[df.is_train].copy().reset_index(drop=True)
    development["is_train"] = development.split_group.map(lambda group:
        int(hashlib.sha256(f"{config['seed']}:development:{group}".encode()).hexdigest()[:16], 16)
        / 2**64 < .75)
    development["inner_fold"] = -1
    mask = development.is_train
    development.loc[mask, "inner_fold"] = grouped_folds(
        development.loc[mask, "split_group"], config["crossfit_folds"], config["seed"])
    return development


def select_development_score(reports, horizons):
    losses = {}
    for pgs, report in reports.items():
        rows = report["models"]["pc_varying_ctn"]["metrics"]
        overall = {row["horizon"]: row for row in rows if row["group"] == "overall"}
        if any(h not in overall or overall[h]["status"] != "ok" for h in horizons):
            raise ValueError("score selection needs supported development Brier scores at every horizon")
        losses[pgs] = float(np.mean([overall[h]["brier"] for h in horizons]))
    if len(losses) != 2 or not all(np.isfinite(list(losses.values()))):
        raise ValueError("score selection requires two finite candidate losses")
    return min(losses, key=lambda pgs: (losses[pgs], pgs)), losses


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
    # Same-day disease/death lacks within-day ordering; exclude explicitly.
    tied = df.disease_date.notna() & df.disease_date.eq(df.death_date) & df.disease_date.le(df.obs_end)
    df = df.loc[~tied].copy()
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
    df["inner_fold"] = -1
    df.loc[df.is_train, "inner_fold"] = grouped_folds(
        df.loc[df.is_train, "split_group"], config["crossfit_folds"], config["seed"])
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
                         "brier": float(np.mean(w * (target - p)**2)),
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
    if spec["kind"] != "baseline":
        if spec["normalizer"] == "ctn" and spec["kind"] in ("constant", "pc_varying"):
            data["PGS"] = baseline_data.PGS.to_numpy()
            chain = gamfit.load(directory / "chain.gamfit")
            return np.asarray(chain.predict(data).cumulative_hazard_at(times))
        transform = gamfit.load(directory / "transform.gamfit")
        data["Z"] = transformed_score(transform, spec["normalizer"], baseline_data[["PGS", *columns]])
    model = gamfit.load(directory / "model.gamfit")
    return np.asarray(model.predict(data).cumulative_hazard_at(times))


def fit_worker(frame_path, config_path, model_kind, cause, output, normalizer, transform_path):
    import gamfit
    config = json.loads(Path(config_path).read_text())
    df = pd.read_parquet(frame_path)
    pc_cols = [f"PC{i + 1}" for i in range(config["num_pcs"])]
    columns = ["entry", "followup", "age0", "sex", *pc_cols]
    data = df[columns].copy()
    if model_kind != "baseline":
        data["Z"] = df[f"Z_{normalizer}"]
    data["event"] = (df.event_code == cause).astype(int)
    train = data.loc[df.is_train].copy()
    test = data.loc[~df.is_train].copy()
    pc_args = ", ".join(pc_cols)
    baseline = f"s(age0, k=8) + sex + duchon({pc_args}, centers={config['baseline_centers']}, scale_dims=true)"
    slope = "1" if model_kind == "constant" else f"1 + duchon({pc_args}, centers={config['slope_centers']}, scale_dims=true)"
    print("worker_fit_started", flush=True)
    if model_kind in ("baseline", "ordinary"):
        rhs = baseline
        if model_kind == "ordinary":
            rhs += f" + duchon({pc_args}, centers={config['slope_centers']}, scale_dims=true, by=Z)"
        model = gamfit.fit(train, f"Surv(entry, followup, event) ~ {rhs}",
                           survival_likelihood="location-scale", noise_formula="1",
                           config={"time_num_internal_knots": config["time_num_internal_knots"]},
                           persistent_warm_start_root=output / "warm")
    else:
        model = gamfit.fit(train, f"Surv(entry, followup, event) ~ {baseline}",
                           survival_likelihood="marginal-slope", z_column="Z",
                           slope_formula=slope, config={"frozen_score": True,
                               "time_num_internal_knots": config["time_num_internal_knots"]},
                           persistent_warm_start_root=output / "warm")
    model.save(output / "model.gamfit")
    print("worker_fit_saved", flush=True)
    if model_kind != "baseline":
        transformer = gamfit.load(transform_path)
        replay_z = transformed_score(transformer, normalizer,
                                     df.loc[~df.is_train, ["PGS", *baseline_columns(config)]])
        if not np.allclose(test.Z, replay_z, rtol=1e-8, atol=1e-10):
            raise ValueError("deployment transform disagrees with held-out score artifact")
        test["Z"] = replay_z
        shutil.copyfile(transform_path, output / "transform.gamfit")
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
    if normalizer == "ctn" and model_kind in ("constant", "pc_varying"):
        from aou_score_transform import stage1_rhs
        recipe = gamfit.CtnStage1("PGS", stage1_rhs(config), fold_column="inner_fold",
                                 group_column="split_group",
                                 response_num_internal_knots=config["stage1_response_knots"])
        fold_digest = hashlib.sha256(df.loc[df.is_train, "inner_fold"].to_numpy("<i8").tobytes()).hexdigest()
        gamfit.CtnMarginalSlopeModel(transformer, model, recipe, fold_digest,
                                     score_column="Z").save(output / "chain.gamfit")
    if model_kind in ("constant", "pc_varying"):
        payload = json.loads((output / "model.gamfit").read_text())["payload"]
        if payload["latent_z_rank_int_calibration"] is not None or payload["latent_z_conditional_calibration"] is not None:
            raise ValueError("outcome fit changed the frozen latent score; matched comparison is invalid")
    write_json(output / "spec.json", {"baseline": baseline, "slope": slope, "cause": cause,
                                      "kind": model_kind, "normalizer": normalizer,
                                      "num_pcs": config["num_pcs"],
                                      "score_path": "explicit cross-fitted scores; frozen deployment transform",
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


def bounded_fit(command, timeout_seconds, log):
    def terminate(signum, frame):
        raise InterruptedError("fit controller was terminated")
    previous_handler = signal.signal(signal.SIGTERM, terminate)
    with Path(log).open("wb") as handle:
        child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            result = child.wait(timeout=timeout_seconds)
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
        if result != 0:
            raise RuntimeError("fit failed; inspect the private task fit log")


def checkpointed_fit(command, timeout_seconds, log, checkpoint):
    """Retain private failure logs and partial fits without a completion receipt."""
    try:
        bounded_fit(command, timeout_seconds, log)
    except BaseException:
        checkpoint.publish()
        raise


def analyze_partition(df, config, args, disease_dir, checkpoint, candidates):
    """Analyze an explicit development or outer-test partition with bounded steps."""
    df = df.copy()
    disease_dir.mkdir(parents=True, exist_ok=True)
    slug = disease_dir.name
    frame = disease_dir / "cohort.parquet"
    df.to_parquet(frame, index=False)
    train, test = df.loc[df.is_train], df.loc[~df.is_train]
    support_check = fit_support if args.smoke_only else partition_support
    errors = support_check(train, test, config)
    if errors:
        raise ValueError("; ".join(errors))
    transforms = {}
    transform_diagnostics = {}
    groups = audit_groups(train, test)
    for normalizer in sorted({normalizer for _, normalizer in candidates} - {"none"}):
        artifacts = []
        for fold in [*range(config["crossfit_folds"]), -1]:
            stage_dir = disease_dir / f"{normalizer}_fold_{fold}"
            stage_dir.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "transform",
                       "--frame", str(frame), "--config", str(args.config.resolve()),
                       "--normalizer", normalizer, "--fold", str(fold), "--output", str(stage_dir)]
            print(f"Transforming {slug}: {normalizer}, fold {fold}", flush=True)
            if not checkpoint.step_is_complete(stage_dir, model=True):
                publish_status(args.checkpoint_uri, "transforming_score")
                checkpointed_fit(command, config["stage1_timeout_seconds"], stage_dir / "fit.log", checkpoint)
                files = ["scores.npz", "transform.gamfit"]
                if fold < 0:
                    files.append("training_replay.npy")
                checkpoint.complete_step(stage_dir, files, model=True)
            artifacts.append(stage_dir / "scores.npz")
            if fold == -1:
                transforms[normalizer] = stage_dir / "transform.gamfit"
        df[f"Z_{normalizer}"] = assemble_scores(df, artifacts)
        publish_status(args.checkpoint_uri, "score_transform_ready")
        transform_diagnostics[normalizer] = score_diagnostics(
            df.loc[~df.is_train, f"Z_{normalizer}"].to_numpy(), groups, config["min_report_count"])
        full_train_z = np.load(disease_dir / f"{normalizer}_fold_-1" / "training_replay.npy")
        oof_z = df.loc[df.is_train, f"Z_{normalizer}"].to_numpy()
        transform_diagnostics[f"{normalizer}_oof_deployment_rms_difference"] = float(
            np.sqrt(np.mean((oof_z - full_train_z)**2)))
    frame = disease_dir / "transformed.parquet"
    df.to_parquet(frame, index=False)
    models = {}
    candidate_risks = {}
    for kind, normalizer in candidates:
        candidate = f"{kind}_{normalizer}"
        hazards = []
        for cause in (1, 2):
            fit_dir = disease_dir / f"{candidate}_{cause}"
            fit_dir.mkdir(exist_ok=True)
            command = [sys.executable, str(Path(__file__).resolve()), "fit",
                       "--frame", str(frame), "--config", str(args.config.resolve()),
                       "--kind", kind, "--normalizer", normalizer,
                       "--cause", str(cause), "--output", str(fit_dir)]
            if kind != "baseline":
                command += ["--transform-model", str(transforms[normalizer])]
            print(f"Fitting {slug}: {candidate}, cause {cause}", flush=True)
            if not checkpoint.step_is_complete(fit_dir, model=True):
                publish_status(args.checkpoint_uri, "fitting_disease" if cause == 1 else "fitting_death")
                checkpointed_fit(command, config["fit_timeout_seconds"], fit_dir / "fit.log", checkpoint)
                files = ["hazards.npz", "model.gamfit", "spec.json"]
                if kind != "baseline":
                    files.append("transform.gamfit")
                if normalizer == "ctn" and kind in ("constant", "pc_varying"):
                    files.append("chain.gamfit")
                checkpoint.complete_step(fit_dir, files, model=True)
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
        candidate_risks[candidate] = risk
        models[candidate] = {"cif_grid_error": error, "metrics": evaluate(
            train, test, risk, config["horizons_years"], config["min_report_count"])}
    comparisons = []
    for j, horizon in enumerate(config["horizons_years"] if len(candidate_risks) > 1 else []):
        weights = ipcw_weights(train, test, horizon)
        target = ((test.event_code == 1) & (test.followup <= horizon)).to_numpy(float)
        full_loss = weights * (target - candidate_risks["pc_varying_ctn"][:, j])**2
        for candidate, risk in candidate_risks.items():
            if candidate == "pc_varying_ctn":
                continue
            reference_loss = weights * (target - risk[:, j])**2
            for label, mask in groups:
                known = weights[mask] > 0
                events = int(target[mask][known].sum())
                if min(events, int(known.sum()) - events,
                       test.loc[mask, "split_group"].nunique()) < config["min_report_count"]:
                    comparisons.append({"reference": candidate, "group": label,
                                        "horizon": horizon, "status": "insufficient_support"})
                    continue
                comparisons.append({"reference": candidate, "model": "pc_varying_ctn",
                                    "group": label, "horizon": horizon, "status": "ok",
                                    **paired_loss_summary(reference_loss, full_loss, mask, test.split_group)})
    return {"models": models, "score_diagnostics": transform_diagnostics,
            "paired_comparisons": comparisons}, candidate_risks


def analyze_development(df, disease, config, args, disease_dir, checkpoint):
    """A smoke fit reuses the first development candidate without opening test data."""
    development = development_partition(df, config)
    candidates = disease["candidates"][:1] if args.smoke_only else disease["candidates"]
    reports = {}
    for pgs in candidates:
        development["PGS"] = development[pgs]
        report, _ = analyze_partition(development, config, args,
            disease_dir / "development" / pgs, checkpoint, [("pc_varying", "ctn")])
        reports[pgs] = report
    return reports


def run(args):
    from aou_identity import task_account
    execution_account = task_account()
    from disease_selection import select_runtime_diseases
    config = json.loads(args.config.read_text())
    validate_config(config)
    panel = load_score_panel(args.score_panel)
    endpoint = json.loads(args.endpoint_config.read_text())
    if endpoint != "" and endpoint not in panel["endpoints"]:
        raise ValueError("requested endpoint is not in the prespecified panel")
    import gamfit
    if gamfit.__version__ != config["gamfit_version"]:
        raise ValueError("gamfit version does not match the analysis configuration")
    if not args.prepare_only:
        # Reject an old wheel before queries or expensive score-model fits.
        validation = gamfit.validate_formula(
            pd.DataFrame({"entry": [0.] * 6, "followup": [1., 2., 3., 4., 5., 6.],
                          "event": [0, 1, 0, 1, 0, 1], "Z": [-2., -1., -.5, .5, 1., 2.]}),
            "Surv(entry, followup, event) ~ 1", survival_likelihood="marginal-slope",
            z_column="Z", slope_formula="1", config={"frozen_score": True})
        if not validation.supported_by_python:
            raise ValueError("native engine does not support the frozen-score survival contract")
    image = json.loads(args.runtime_image.read_text())
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("runtime_image must use an immutable digest")
    sources = [Path(__file__), *[Path(__file__).with_name(name) for name in
               ("aou_identity.py", "aou_score_transform.py", "aou_checkpoint.py",
                "aou_evaluation.py", "aou_status.py", "disease_selection.py")]]
    signature = {"config": config, "endpoint": endpoint,
                 "score_scope": "primary" if args.smoke_only else "comparison",
                 "sources": {p.name: digest(p) for p in sources},
                 "inputs": {key: digest(getattr(args, key)) for key in
                            ("scores", "ancestry", "prune", "phenotypes", "score_panel")}}
    checkpoint = StudyCheckpoint(args.output, args.checkpoint_uri, config["google_project"],
                                 execution_account, signature, args.resume,
                                 engine_hash=digest(importlib.util.find_spec("gamfit._rust").origin))
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
        score_cache = unpack_score_cache(args.scores, args.output / "scores.tar")
        available_scores = cached_score_ids(score_cache)
        for disease in diseases.values():
            required = disease["candidates"][:1] if args.smoke_only else disease["candidates"]
            disease["missing_scores"] = sorted(set(required) - available_scores)
            for pgs in disease["missing_scores"]:
                publish_status(args.checkpoint_uri, "missing_" + pgs.lower())
        publish_status(args.checkpoint_uri, "reading_ancestry")
        ancestry = read_ancestry(args.ancestry, args.prune, config["num_pcs"])
        publish_status(args.checkpoint_uri, "loading_person_times")
        base = ancestry.merge(person_times(client, config["workspace_cdr"]), on="person_id", validate="one_to_one")
        publish_status(args.checkpoint_uri, "preparing_cohort")
        for slug, disease in diseases.items():
            if disease["missing_scores"]:
                continue
            scores = endpoint_scores(score_cache, disease, primary_only=args.smoke_only)
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
                      "training_deaths": int((train.event_code == 2).sum())}
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
        selection_reports = analyze_development(df, disease, config, args, disease_dir, checkpoint)
        if args.smoke_only:
            results[slug] = {"status": "development_smoke_completed",
                             "prespecified_pgs": disease["candidates"][0],
                             "evaluation_support_errors": support_errors,
                             "development": selection_reports}
            checkpoint.publish()
            publish_status(args.checkpoint_uri, "smoke_completed")
            continue
        selected, losses = select_development_score(selection_reports, config["horizons_years"])
        # Commit the choice before any outer-test fit/evaluation. Every method
        # below receives this same score and exactly the same cohort rows.
        choice = {"selected_pgs": selected, "development_brier": losses,
                  "criterion": "unweighted mean of horizon-specific development Brier scores"}
        write_json(disease_dir / "score_selection.json", choice)
        checkpoint.publish()
        df["PGS"] = df[selected]
        report, _ = analyze_partition(df, config, args, disease_dir / "final", checkpoint,
            [("baseline", "none"), ("constant", "ctn"), ("pc_varying", "ctn"),
             ("pc_varying", "location_scale"), ("ordinary", "ctn")])
        results[slug] = {"concept_id": disease["concept_id"], "score_selection": choice,
                         "development": selection_reports, **report}
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
        "baseline": "AoU primary-consent date (Consent PII Module descendants); continuous EHR lookback",
        "validation": "group holdout after published relatedness prune; 75/25 development split within outer training selects PGS; outer test evaluates matched methods",
        "score_panel": panel,
        "score_transform": "explicit cross-fitted CTN and location-scale models; frozen latent scores",
        "orthogonality_claim": False,
        "paired_uncertainty": "group-robust normal intervals conditional on fitted models and training censoring estimates",
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
    for name in ["config", "phenotypes", "scores", "ancestry", "prune", "runtime-image", "score-panel", "endpoint-config", "output"]:
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    run_parser.add_argument("--checkpoint-uri", required=True)
    run_parser.add_argument("--resume", type=Path)
    mode = run_parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--smoke-only", action="store_true")
    fit_parser = sub.add_parser("fit")
    for name in ["frame", "config", "output"]:
        fit_parser.add_argument(f"--{name}", type=Path, required=True)
    fit_parser.add_argument("--kind", choices=["baseline", "constant", "pc_varying", "ordinary"], required=True)
    fit_parser.add_argument("--normalizer", choices=["ctn", "location_scale", "none"], required=True)
    fit_parser.add_argument("--transform-model", type=Path)
    fit_parser.add_argument("--cause", type=int, choices=[1, 2], required=True)
    transform_parser = sub.add_parser("transform")
    for name in ["frame", "config", "output"]:
        transform_parser.add_argument(f"--{name}", type=Path, required=True)
    transform_parser.add_argument("--normalizer", choices=["ctn", "location_scale"], required=True)
    transform_parser.add_argument("--fold", type=int, required=True)
    args = parser.parse_args()
    if args.command == "run":
        try:
            run(args)
        except Exception as error:
            try:
                publish_status(args.checkpoint_uri, failure_label(error))
            except Exception:
                print("Could not publish the fixed failure label; inspect workspace logs", file=sys.stderr)
            raise
    elif args.command == "transform":
        fit_transform(args.frame, args.config, args.normalizer, args.fold, args.output)
    else:
        if (args.kind == "baseline") != (args.normalizer == "none"):
            parser.error("only the no-score baseline uses normalizer=none")
        if args.kind != "baseline" and args.transform_model is None:
            parser.error("score models require a saved deployment transform")
        fit_worker(args.frame, args.config, args.kind, args.cause, args.output,
                   args.normalizer, args.transform_model)


if __name__ == "__main__":
    main()
