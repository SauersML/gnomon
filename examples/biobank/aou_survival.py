#!/usr/bin/env python3
"""Bounded AoU incident-survival pilot; execute only inside the AoU workspace.

Uses cached Gnomon scores, the shared disease selector, and the formula-first
gamfit API. Does not invoke Gnomon's different calibration adapter model.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tarfile
import zipfile

import numpy as np
import pandas as pd


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
        "max_rows_per_disease", "train_fraction", "seed", "horizons_years",
        "grid_intervals", "fit_timeout_seconds", "query_timeout_seconds",
        "maximum_bytes_billed", "min_train_events_per_cause", "min_report_count",
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
    if c["min_report_count"] < 20 or c["min_train_events_per_cause"] < 20:
        raise ValueError("pilot requires at least 20 observations per reported cell/event class")
    if type(c["seed"]) is not int or not c["gamfit_version"]:
        raise ValueError("seed and gamfit version are required")
    h = np.asarray(c["horizons_years"], dtype=float)
    if h.ndim != 1 or h.size == 0 or not np.isfinite(h).all() or h[0] <= 0 or (np.diff(h) <= 0).any():
        raise ValueError("horizons must be finite, positive, and strictly increasing")


class BoundedClient:
    """Apply query budgets even to the existing selector's query calls."""
    def __init__(self, config):
        from google.cloud import bigquery
        self.client = bigquery.Client(project=config["google_project"])
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
    df = pd.read_csv(ancestry, sep="\t", dtype=str,
                     usecols=["research_id", "pca_features", "ancestry_pred"])
    excluded = pd.read_csv(prune, sep="\t", dtype=str, usecols=["research_id"])
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
    return out


def unpack_score_cache(archive, output):
    """Read pgsEngine's real shared-feature artifact without extracting genotypes."""
    with tarfile.open(archive, "r:*") as tar:
        members = [member for member in tar if member.isfile() and Path(member.name).name == "scores.tar"]
        if len(members) != 1:
            raise ValueError("expected exactly one scores.tar in the pgsEngine shared-feature archive")
        with tar.extractfile(members[0]) as source, Path(output).open("wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
    return Path(output)


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


def person_times(client, cdr):
    # A single observed interval: never bridge unobserved gaps with MIN/MAX.
    return client.query(f"""
      WITH periods AS (
        SELECT person_id, observation_period_start_date AS baseline,
               observation_period_end_date AS obs_end,
               ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY
                 observation_period_start_date, observation_period_end_date DESC,
                 observation_period_id) AS seq
        FROM `{cdr}.observation_period`
      ), deaths AS (
        SELECT person_id, MIN(death_date) AS death_date FROM `{cdr}.death` GROUP BY person_id
      )
      SELECT CAST(p.person_id AS STRING) AS person_id,
             DATE(p.birth_datetime) AS birth_date,
             p.sex_at_birth_concept_id, o.baseline, o.obs_end, d.death_date
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
    for name in ["birth_date", "baseline", "obs_end", "death_date", "disease_date"]:
        df[name] = pd.to_datetime(df[name])
    df["sex"] = df.sex_at_birth_concept_id.map({8507: 1, 8532: 0, 45880669: 1, 45878463: 0})
    df["age0"] = (df.baseline - df.birth_date).dt.days / 365.25
    eligible = (df.sex.notna() & df.age0.ge(18) & df.obs_end.gt(df.baseline)
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
    df["is_train"] = df.person_id.map(
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
        weights = ipcw_weights(train, test, horizon)
        y = ((test.event_code == 1) & (test.followup <= horizon)).to_numpy(float)
        groups = [("overall", np.ones(len(test), dtype=bool))]
        groups += [(f"ancestry:{a}", test.ancestry.eq(a).to_numpy()) for a in sorted(test.ancestry.unique())]
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
                         "ipcw_observed_risk": float(np.mean(w * target))})
    return rows


def fit_worker(frame_path, config_path, model_kind, cause, output):
    import gamfit
    config = json.loads(Path(config_path).read_text())
    df = pd.read_parquet(frame_path)
    pc_cols = [f"PC{i + 1}" for i in range(config["num_pcs"])]
    columns = ["entry", "followup", "age0", "sex", "PGS", *pc_cols]
    data = df[columns].copy()
    data["event"] = (df.event_code == cause).astype(int)
    train = data.loc[df.is_train].copy()
    test = data.loc[~df.is_train].copy()
    pc_args = ", ".join(pc_cols)
    baseline = f"s(age0, k=8) + sex + duchon({pc_args}, centers={config['baseline_centers']}, scale_dims=true)"
    slope = "1" if model_kind == "constant" else f"1 + duchon({pc_args}, centers={config['slope_centers']}, scale_dims=true)"
    model = gamfit.fit(train, f"Surv(entry, followup, event) ~ {baseline}",
                       survival_likelihood="marginal-slope", z_column="PGS",
                       slope_formula=slope, persistent_warm_start_root=output / "warm")
    horizons = np.asarray(config["horizons_years"])
    coarse = np.unique(np.r_[np.linspace(0, horizons[-1], config["grid_intervals"] + 1), horizons])
    grid = np.sort(np.r_[coarse, (coarse[:-1] + coarse[1:]) / 2])
    # Do not send held-out event/censoring observations as prediction features.
    test["event"] = 0
    test["followup"] = horizons[-1]
    prediction = model.predict(test)
    h = np.asarray(prediction.cumulative_hazard_at(grid))
    model.save(output / "model.gamfit")
    restored = gamfit.load(output / "model.gamfit")
    replayed = np.asarray(restored.predict(test).cumulative_hazard_at(grid))
    if h.shape != (len(test), len(grid)) or not np.allclose(h, replayed, rtol=1e-7, atol=1e-9):
        raise ValueError("save/load cumulative-hazard predictions disagree")
    np.savez(output / "hazards.npz", hazards=h, grid=grid, coarse=coarse)
    write_json(output / "spec.json", {"baseline": baseline, "slope": slope, "cause": cause,
                                      "score_path": "z_column fitted latent-score gate",
                                      "cross_fitted_ctn": False})


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


def run(args):
    from aou_identity import task_account
    execution_account = task_account()
    from disease_selection import select_runtime_diseases
    config = json.loads(args.config.read_text())
    validate_config(config)
    import gamfit
    if gamfit.__version__ != config["gamfit_version"]:
        raise ValueError("gamfit version does not match the analysis configuration")
    image = json.loads(args.runtime_image.read_text())
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("runtime_image must use an immutable digest")
    args.output.mkdir(parents=True)
    client = BoundedClient(config)
    phenotypes = unpack_phenotypes(args.phenotypes, args.output / "phenotypes")
    diseases = select_runtime_diseases(client, config["workspace_cdr"], config["top_n_diseases"], phenotypes)
    if not diseases:
        raise ValueError("the existing disease selector found no eligible mapped disease")
    diseases = dict(list(diseases.items())[:config["disease_limit"]])
    score_cache = unpack_score_cache(args.scores, args.output / "scores.tar")
    ancestry = read_ancestry(args.ancestry, args.prune, config["num_pcs"])
    base = ancestry.merge(person_times(client, config["workspace_cdr"]), on="person_id", validate="one_to_one")
    results = {}
    for slug, disease in diseases.items():
        print(f"Preparing {slug} ({disease['pgs']})", flush=True)
        scores = load_cached_score(score_cache, disease["pgs"])
        cases = case_dates(client, config["workspace_cdr"], disease["concept_id"])
        df = build_cohort(base, scores, cases, config)
        train, test = df.loc[df.is_train], df.loc[~df.is_train]
        for cause in (1, 2):
            if (train.event_code == cause).sum() < config["min_train_events_per_cause"]:
                raise ValueError("insufficient training events in pilot; increase the outcome-blind sample budget")
        if len(test) < config["min_report_count"]:
            raise ValueError("pilot has too few held-out participants")
        # Verify evaluation support before spending time fitting.
        for horizon in config["horizons_years"]:
            ipcw_weights(train, test, horizon)
        disease_dir = args.output / slug
        disease_dir.mkdir()
        frame = disease_dir / "cohort.parquet"
        df.to_parquet(frame, index=False)
        models = {}
        for kind in ("constant", "pc_varying"):
            hazards = []
            for cause in (1, 2):
                fit_dir = disease_dir / f"{kind}_{cause}"
                fit_dir.mkdir()
                command = [sys.executable, str(Path(__file__).resolve()), "fit",
                           "--frame", str(frame), "--config", str(args.config.resolve()),
                           "--kind", kind, "--cause", str(cause), "--output", str(fit_dir)]
                print(f"Fitting {slug}: {kind}, cause {cause}", flush=True)
                bounded_fit(command, config["fit_timeout_seconds"], fit_dir / "fit.log")
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
            models[kind] = {"cif_grid_error": error, "metrics": evaluate(
                train, test, risk, config["horizons_years"], config["min_report_count"])}
        results[slug] = {"pgs": disease["pgs"], "concept_id": disease["concept_id"], "models": models}
    write_json(args.output / "metrics.json", results)
    write_json(args.output / "provenance.json", {
        "config": config, "runtime_image": image, "gamfit_build": gamfit.build_info(),
        "execution_account": execution_account,
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "input_sha256": {key: digest(getattr(args, key)) for key in ["config", "scores", "ancestry", "prune", "phenotypes"]},
        "runner_sha256": digest(__file__), "selector_sha256": digest(Path(__file__).with_name("disease_selection.py")),
        "query_job_ids": [job.job_id for job in client.jobs],
        "target": "first recorded disease within the first continuous EHR observation period, competing death",
        "baseline": "first observation_period_start_date; not recruitment",
        "validation": "outcome-blind holdout after AoU published relatedness prune; no model selection claim",
        "score_transform": "gamfit z_column gate; no claim of cross-fitted CTN or conditional Gaussian adequacy",
        "censoring_model": "training-only reverse Kaplan-Meier stratified by reported genetic ancestry",
    })
    print("Completed bounded pilot; aggregate metrics and provenance written", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    for name in ["config", "phenotypes", "scores", "ancestry", "prune", "runtime-image", "output"]:
        run_parser.add_argument(f"--{name}", type=Path, required=True)
    fit_parser = sub.add_parser("fit")
    for name in ["frame", "config", "output"]:
        fit_parser.add_argument(f"--{name}", type=Path, required=True)
    fit_parser.add_argument("--kind", choices=["constant", "pc_varying"], required=True)
    fit_parser.add_argument("--cause", type=int, choices=[1, 2], required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        fit_worker(args.frame, args.config, args.kind, args.cause, args.output)


if __name__ == "__main__":
    main()
