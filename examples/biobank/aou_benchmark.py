"""Benchmark polygenic score methods on ever-diagnosed EHR outcomes in AoU.

Each disease is a case-control outcome: a participant is a case with any
recorded descendant of the disease concept, and a control otherwise, among
participants with at least a year of EHR history before consent. Every score
already cached in the workspace feeds the same training and held-out
participants into five methods, from covariates alone to gnomon's
ancestry-aware marginal-slope model. Held-out metrics leave the workspace only
as aggregate tokens over at least the reporting minimum of participants,
cases and controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

METHODS = ("covariates", "standard", "ancestry_z", "pc_adjusted", "gnomon")
REFERENCES = ("covariates", "standard")
# E[log chi-square(1)]: the bias of a log squared residual as a log variance.
LOG_CHI2_1_MEAN = -1.2703628454614782


def validate_config(c):
    expected = {"google_project", "workspace_cdr", "seed", "num_pcs", "max_rows_per_disease",
                "test_fraction", "min_report_count", "maximum_bytes_billed",
                "query_timeout_seconds", "lookback_days", "gnomon_timeout_seconds",
                "gnomon_centers", "diseases"}
    if set(c) != expected:
        raise ValueError("benchmark configuration has missing or unknown keys")
    if not 1 <= c["num_pcs"] <= 16 or not 0.1 <= c["test_fraction"] <= 0.5:
        raise ValueError("unsupported PC count or test fraction")
    if c["min_report_count"] < 20:
        raise ValueError("aggregate cells need at least 20 participants, cases and controls")
    if not 1000 <= c["max_rows_per_disease"] <= 250000:
        raise ValueError("participants per disease must be between 1,000 and 250,000")
    if not 60 <= c["gnomon_timeout_seconds"] <= 3600 or c["gnomon_centers"] <= c["num_pcs"] + 1:
        raise ValueError("gnomon fit bound or Duchon basis size is invalid")
    diseases = c["diseases"]
    if not isinstance(diseases, dict) or not 1 <= len(diseases) <= 5:
        raise ValueError("benchmark one to five diseases")
    for name, disease in diseases.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,39}", name) or set(disease) != {"snomed_code", "scores"}:
            raise ValueError("each disease needs a slug, a SNOMED code and its scores")
        if not re.fullmatch(r"[0-9]{6,18}", disease["snomed_code"]):
            raise ValueError("SNOMED codes are numeric concept codes")
        scores = disease["scores"]
        if not 1 <= len(scores) <= 3 or not all(re.fullmatch(r"PGS[0-9]{6}", s) for s in scores):
            raise ValueError("each disease benchmarks one to three PGS Catalog scores")


def stable_hash(seed, purpose, identifier):
    return hashlib.sha256(f"{seed}:{purpose}:{identifier}".encode()).hexdigest()


# --------------------------------------------------------------------------- #
# cohort
# --------------------------------------------------------------------------- #
def eligible_participants(ancestry, times, config):
    df = ancestry.merge(times, on="person_id", validate="one_to_one")
    for name in ("birth_date", "baseline", "obs_start", "obs_end"):
        df[name] = pd.to_datetime(df[name])
    df["sex"] = df.sex_at_birth_concept_id.map({8507: 1, 8532: 0, 45880669: 1, 45878463: 0})
    df["age0"] = (df.baseline - df.birth_date).dt.days / 365.25
    keep = (df.sex.notna() & df.age0.ge(18) & df.obs_end.gt(df.baseline)
            & ((df.baseline - df.obs_start).dt.days >= config["lookback_days"]))
    pcs = [f"PC{i + 1}" for i in range(config["num_pcs"])]
    return df.loc[keep, ["person_id", "ancestry", "sex", "age0", *pcs]].reset_index(drop=True)


def case_sets(client, cdr, roots):
    from google.cloud import bigquery
    frame = client.query(f"""
      SELECT ca.ancestor_concept_id AS root, CAST(co.person_id AS STRING) AS person_id
      FROM `{cdr}.condition_occurrence` co
      JOIN `{cdr}.concept_ancestor` ca ON ca.descendant_concept_id = co.condition_concept_id
      WHERE ca.ancestor_concept_id IN UNNEST(@roots)
      GROUP BY root, person_id
    """, bigquery.QueryJobConfig(query_parameters=[
        bigquery.ArrayQueryParameter("roots", "INT64", [int(r) for r in roots])
    ])).to_dataframe(create_bqstorage_client=False)
    return {int(root): set(group.person_id) for root, group in frame.groupby("root")}


def load_sscore(path, pgs):
    """One cached score: its average and, where present, its missingness."""
    with Path(path).open() as handle:
        lines = [handle.readline() for _ in range(64)]
    header_index = next((i for i, line in enumerate(lines)
                         if line.split("\t", 1)[0].lstrip("#").strip() == "IID"), None)
    if header_index is None:
        raise ValueError(f"cached score {pgs} has no IID header")
    header = lines[header_index].rstrip("\n").split("\t")
    iid, avg, missing = header[0], f"{pgs}_AVG", f"{pgs}_MISSING_PCT"
    if avg not in header:
        raise ValueError(f"cached score {pgs} lacks its average column")
    columns = [iid, avg] + ([missing] if missing in header else [])
    scores = pd.read_csv(path, sep="\t", skiprows=header_index, usecols=columns, dtype={iid: str})
    scores = scores.rename(columns={iid: "person_id", avg: pgs})
    if missing in scores:
        scores = scores.loc[scores[missing] < 100].drop(columns=missing)
    scores = scores.loc[np.isfinite(scores[pgs].to_numpy(float))]
    if scores.person_id.duplicated().any():
        raise ValueError(f"cached score {pgs} repeats participants")
    return scores


def disease_cohort(base, case_ids, scores, config):
    df = base
    for pgs, frame in scores.items():
        df = df.merge(frame, on="person_id", validate="one_to_one")
    df = df.assign(y=df.person_id.isin(case_ids).astype(float))
    order = df.person_id.map(lambda x: stable_hash(config["seed"], "benchmark-sample", x))
    df = df.assign(_order=order).sort_values("_order").head(config["max_rows_per_disease"])
    df["is_test"] = df.person_id.map(
        lambda x: int(stable_hash(config["seed"], "benchmark-split", x)[:16], 16) / 2**64 < config["test_fraction"])
    return df.drop(columns="_order").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #
def expit(x):
    return np.exp(-np.logaddexp(0.0, -x))


def logistic_fit(X, y, ridge=1e-8, iterations=100):
    """Newton-Raphson logistic regression with step halving."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    def loglik(b):
        eta = X @ b
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)) - 0.5 * ridge * b @ b)
    current = loglik(beta)
    for _ in range(iterations):
        p = expit(X @ beta)
        w = p * (1 - p)
        hessian = X.T @ (X * w[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hessian, X.T @ (y - p) - ridge * beta)
        scale = 1.0
        while scale > 1e-6:
            candidate = beta + scale * step
            value = loglik(candidate)
            if value >= current - 1e-12:
                break
            scale /= 2
        beta, previous, current = candidate, current, value
        if np.max(np.abs(scale * step)) < 1e-9 or abs(current - previous) < 1e-10 * (1 + abs(current)):
            return beta
    raise ValueError("logistic regression did not converge")


def midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, float)[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(xs)) + 1])
    ends = np.concatenate([starts[1:], [len(xs)]])
    ranks = np.empty(len(xs))
    ranks[order] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
    return ranks


def auc(y, p):
    y = np.asarray(y).astype(bool)
    m, n = int(y.sum()), int((~y).sum())
    if m == 0 or n == 0:
        raise ValueError("AUC needs cases and controls")
    ranks = midrank(p)
    return float((ranks[y].sum() - m * (m + 1) / 2) / (m * n))


def delong(y, predictions):
    """AUCs of paired predictions and their covariance (Sun and Xu 2014)."""
    y = np.asarray(y).astype(bool)
    predictions = np.atleast_2d(np.asarray(predictions, float))
    positive, negative = predictions[:, y], predictions[:, ~y]
    m, n = positive.shape[1], negative.shape[1]
    if m < 2 or n < 2:
        raise ValueError("DeLong needs at least two cases and two controls")
    tx = np.vstack([midrank(row) for row in positive])
    ty = np.vstack([midrank(row) for row in negative])
    tz = np.vstack([midrank(np.concatenate([a, b])) for a, b in zip(positive, negative)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, covariance


def cell_metrics(y, p):
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    logit = np.log(p) - np.log1p(-p)
    slope_intercept = logistic_fit(np.column_stack([np.ones(len(p)), logit]), y)
    return {"auc": auc(y, p), "brier": float(np.mean((y - p) ** 2)),
            "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p))),
            "mean_predicted": float(p.mean()), "observed": float(y.mean()),
            "calibration_intercept": float(slope_intercept[0]),
            "calibration_slope": float(slope_intercept[1])}


def paired_deltas(y, p_method, p_reference):
    y = np.asarray(y, float)
    p1 = np.clip(np.asarray(p_method, float), 1e-12, 1 - 1e-12)
    p0 = np.clip(np.asarray(p_reference, float), 1e-12, 1 - 1e-12)
    aucs, cov = delong(y, np.vstack([p1, p0]))
    d_brier = (y - p1) ** 2 - (y - p0) ** 2
    d_logloss = -(y * np.log(p1) + (1 - y) * np.log1p(-p1)) + (y * np.log(p0) + (1 - y) * np.log1p(-p0))
    root_n = math.sqrt(len(y))
    return {"auc_difference": float(aucs[0] - aucs[1]),
            "auc_difference_se": float(math.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0))),
            "brier_difference": float(d_brier.mean()),
            "brier_difference_se": float(d_brier.std(ddof=1) / root_n),
            "logloss_difference": float(d_logloss.mean()),
            "logloss_difference_se": float(d_logloss.std(ddof=1) / root_n)}


# --------------------------------------------------------------------------- #
# methods
# --------------------------------------------------------------------------- #
def standardise(values, train):
    center, scale = values[train].mean(axis=0), values[train].std(axis=0)
    if np.any(scale <= 0):
        raise ValueError("a covariate does not vary in the training rows")
    return (values - center) / scale


def covariate_matrix(df, train, num_pcs):
    pcs = df[[f"PC{i + 1}" for i in range(num_pcs)]].to_numpy(float)
    age = standardise(df.age0.to_numpy(float)[:, None], train)
    return np.column_stack([np.ones(len(df)), age, age ** 2, df.sex.to_numpy(float),
                            standardise(pcs, train)])


def score_transforms(df, pgs, train, num_pcs, minimum_group=50):
    """The score three ways, all from training rows only: standardised overall,
    within each reported ancestry, and against a linear model of its mean and
    log variance in PC space."""
    s = df[pgs].to_numpy(float)
    out = {"standard": (s - s[train].mean()) / s[train].std()}
    z = np.empty(len(s))
    overall = (s[train].mean(), s[train].std())
    for label in df.ancestry.unique():
        rows = (df.ancestry == label).to_numpy()
        fit = rows & train
        center, scale = (s[fit].mean(), s[fit].std()) if fit.sum() >= minimum_group and s[fit].std() > 0 else overall
        z[rows] = (s[rows] - center) / scale
    out["ancestry_z"] = z
    P = np.column_stack([np.ones(len(df)), standardise(df[[f"PC{i + 1}" for i in range(num_pcs)]].to_numpy(float), train)])
    mean_coef = np.linalg.lstsq(P[train], s[train], rcond=None)[0]
    residual = s - P @ mean_coef
    log_var_coef = np.linalg.lstsq(P[train], np.log(residual[train] ** 2 + 1e-300), rcond=None)[0]
    out["pc_adjusted"] = residual / np.exp(0.5 * (P @ log_var_coef - LOG_CHI2_1_MEAN))
    return out


def logistic_predictions(df, pgs, config):
    train = ~df.is_test.to_numpy()
    y = df.y.to_numpy(float)
    X = covariate_matrix(df, train, config["num_pcs"])
    transforms = score_transforms(df, pgs, train, config["num_pcs"])
    predictions, odds_ratios = {}, {}
    beta = logistic_fit(X[train], y[train])
    predictions["covariates"] = expit(X @ beta)
    for method in ("standard", "ancestry_z", "pc_adjusted"):
        Xm = np.column_stack([X, transforms[method]])
        beta = logistic_fit(Xm[train], y[train])
        predictions[method] = expit(Xm @ beta)
        odds_ratios[method] = float(np.exp(beta[-1]))
    return predictions, odds_ratios, transforms["pc_adjusted"]


def gnomon_worker(frame_path, params_path, output_path):
    import gamfit
    params = json.loads(Path(params_path).read_text())
    df = pd.read_parquet(frame_path)
    pcs = [f"PC{i + 1}" for i in range(params["num_pcs"])]
    surface = f"duchon({', '.join(pcs)}, centers={params['centers']})"
    columns = ["age0", "sex", *pcs]
    train, test = df.loc[~df.is_test], df.loc[df.is_test]
    fit_data = {"event": train.y.to_numpy(float), "z": train.z.to_numpy(float),
                **{c: train[c].to_numpy(float) for c in columns}}
    test_data = {"z": test.z.to_numpy(float), **{c: test[c].to_numpy(float) for c in columns}}
    print("gnomon_fit_started", flush=True)
    model = gamfit.fit(fit_data, formula=f"event ~ s(age0, k=6) + sex + {surface}",
                       family="bernoulli-marginal-slope", link="probit", z_column="z",
                       slope_formula=f"1 + {surface}")
    print("gnomon_fit_saved", flush=True)
    p = np.asarray(model.predict(test_data), dtype=float)
    if p.shape != (len(test),) or not np.isfinite(p).all():
        raise ValueError("gnomon predictions are invalid")
    np.save(output_path, p)
    print("gnomon_predict_complete", flush=True)


def run_bounded(jobs, timeout_seconds, threads):
    """Run every job at once, each isolated: one failure never stops another."""
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    running = {}
    for key, command, log in jobs:
        handle = Path(log).open("wb")
        running[key] = (subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                         start_new_session=True, env=env), handle, time.monotonic())
    results, deadline = {}, time.monotonic() + timeout_seconds
    while len(results) < len(running):
        for key, (child, handle, started) in running.items():
            if key not in results and child.poll() is not None:
                handle.close()
                results[key] = ("ok" if child.returncode == 0 else "error", time.monotonic() - started)
        if len(results) < len(running) and time.monotonic() >= deadline:
            for key, (child, handle, started) in running.items():
                if key not in results:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    handle.close()
                    results[key] = ("timeout", time.monotonic() - started)
        time.sleep(0.5)
    return results


# Fixed categories for the first recognizable phrase of a solver failure, so a
# generic exception class (gamfit raises GamError for every solver outcome) still
# separates a startup-seed rejection from a resource refusal. Vocabulary, never
# data: a message that matches none stays at the class name alone.
FAILURE_PHRASES = (
    ("no candidate seeds passed", "startup_seeds"),
    ("non-finite cost", "nonfinite_cost"),
    ("failed to converge", "nonconvergence"),
    ("did not converge", "nonconvergence"),
    ("resource policy", "resource_policy"),
    ("refusing to densify", "resource_policy"),
    ("identifiab", "identifiability"),
    ("singular", "singular"),
    ("timed out", "timeout"),
)


def failure_class(log):
    """The exception class a failed worker raised, with the fixed category of
    its message when one applies: a code name, never data."""
    text = Path(log).read_bytes()[-65536:].decode("utf-8", errors="replace")
    found = re.findall(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception))\b(.*)$", text, re.MULTILINE)
    if not found:
        return "unclassified"
    exception, message = found[-1]
    label = re.sub(r"[^a-z0-9]+", "_", exception.rsplit(".", 1)[-1].lower())
    lowered = message.lower()
    for phrase, category in FAILURE_PHRASES:
        if phrase in lowered:
            return f"{label}_{category}"
    return label


# --------------------------------------------------------------------------- #
# aggregate tokens
# --------------------------------------------------------------------------- #
def token(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("digest values must be numbers")
    if isinstance(value, (int, np.integer)):
        text = str(int(value))
    else:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("digest values must be finite")
        text = f"{value:.4g}"
    return text.replace("-", "m").replace("+", "p")


def slug(text):
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if not cleaned:
        raise ValueError("empty digest label")
    return cleaned[:40]


class Digest:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def emit(self, *parts):
        # Words and tokens pass through; anything else is reduced to a slug.
        name = "digest__" + "__".join(p if isinstance(p, str) and re.fullmatch(r"[a-z0-9_.]+", p) else slug(p)
                                        for p in parts) + ".txt"
        (self.directory / name).write_text("\n")


def audit_groups(test):
    groups = [("overall", np.ones(len(test), dtype=bool))]
    groups += [(f"ancestry_{a}", test.ancestry.eq(a).to_numpy()) for a in sorted(test.ancestry.unique())]
    groups += [(f"sex_{s}", test.sex.eq(s).to_numpy()) for s in (0, 1)]
    groups += [(f"age_{lo}_{hi}", ((test.age0 >= lo) & (test.age0 < hi)).to_numpy())
               for lo, hi in ((18, 40), (40, 60), (60, 200))]
    return groups


def report(digest, disease, pgs, test, predictions, config):
    y = test.y.to_numpy(float)
    minimum = config["min_report_count"]
    for label, mask in audit_groups(test):
        n, cases = int(mask.sum()), int(y[mask].sum())
        if min(n, cases, n - cases) < minimum:
            digest.emit(disease, slug(pgs), "all", label, "insufficient_support")
            continue
        for method, p in predictions.items():
            for key, value in cell_metrics(y[mask], p[mask]).items():
                digest.emit(disease, slug(pgs), method, label, key, token(value))
            digest.emit(disease, slug(pgs), method, label, "n", token(n))
            digest.emit(disease, slug(pgs), method, label, "cases", token(cases))
        for reference in REFERENCES:
            # Each method is paired only against the simpler references before it.
            for method, p in predictions.items():
                if reference not in predictions or METHODS.index(method) <= METHODS.index(reference):
                    continue
                for key, value in paired_deltas(y[mask], p[mask], predictions[reference][mask]).items():
                    digest.emit(disease, slug(pgs), "delta", method, "vs", reference, label, key, token(value))


# --------------------------------------------------------------------------- #
# workflow
# --------------------------------------------------------------------------- #
def run(args):
    from aou_identity import task_account
    from aou_projection import source_identity, stream_projection
    from aou_status import publish_status
    from aou_survival import BoundedClient, person_times, read_ancestry
    from disease_selection import resolve_snomed_codes

    account = task_account()
    config = json.loads(args.config.read_text())
    validate_config(config)
    status = args.status_uri
    project, cdr = config["google_project"], config["workspace_cdr"]
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    digest = Digest(args.output)

    projection = work / "projection_pcs.parquet"
    stream_projection(source_identity(args.features_uri, project, account), projection, project, account)
    publish_status(status, "benchmark_projection_ready")
    ancestry = read_ancestry(args.ancestry, args.prune, config["num_pcs"], projection)
    client = BoundedClient(config, account)
    base = eligible_participants(ancestry, person_times(client, cdr), config)
    codes = {name: d["snomed_code"] for name, d in config["diseases"].items()}
    resolved = resolve_snomed_codes(client, cdr, sorted(set(codes.values())))
    code_to_id = dict(zip(resolved.concept_code.astype(str), resolved.concept_id.astype(int)))
    if any(code not in code_to_id for code in codes.values()):
        raise ValueError("a benchmark SNOMED code did not resolve to a standard concept")
    cases = case_sets(client, cdr, sorted(set(code_to_id.values())))
    publish_status(status, "benchmark_cohort_ready")

    staged = {}
    for disease, spec in config["diseases"].items():
        scores = {pgs: load_sscore(args.scores / f"{pgs}.sscore", pgs) for pgs in spec["scores"]}
        cohort = disease_cohort(base, cases.get(code_to_id[spec["snomed_code"]], set()), scores, config)
        n, n_cases = len(cohort), int(cohort.y.sum())
        if min(n_cases, n - n_cases) < config["min_report_count"]:
            digest.emit(disease, "cohort", "insufficient_support")
            continue
        digest.emit(disease, "cohort", "n", token(n))
        digest.emit(disease, "cohort", "cases", token(n_cases))
        for pgs in spec["scores"]:
            predictions, odds_ratios, z_pc = logistic_predictions(cohort, pgs, config)
            for method, value in odds_ratios.items():
                digest.emit(disease, slug(pgs), method, "odds_ratio_per_sd", token(value))
            frame = work / f"{disease}__{pgs}.parquet"
            pcs = [f"PC{i + 1}" for i in range(config["num_pcs"])]
            cohort[["y", "is_test", "age0", "sex", *pcs]].assign(z=z_pc).to_parquet(frame, index=False)
            staged[(disease, pgs)] = (cohort.loc[cohort.is_test].reset_index(drop=True),
                                      {m: p[cohort.is_test.to_numpy()] for m, p in predictions.items()})

    publish_status(status, "benchmark_fitting")
    params = work / "gnomon_params.json"
    params.write_text(json.dumps({"num_pcs": config["num_pcs"], "centers": config["gnomon_centers"]}))
    jobs = [((disease, pgs), [sys.executable, str(Path(__file__).resolve()), "gnomon-fit",
                              "--frame", str(work / f"{disease}__{pgs}.parquet"), "--params", str(params),
                              "--output", str(work / f"{disease}__{pgs}.npy")],
             work / f"{disease}__{pgs}.log") for disease, pgs in staged]
    threads = max(1, (os.cpu_count() or 2) // max(1, len(jobs)))
    outcomes = run_bounded(jobs, config["gnomon_timeout_seconds"], threads) if jobs else {}
    for (disease, pgs), (test, predictions) in staged.items():
        outcome, wall = outcomes[(disease, pgs)]
        bucket = "under_2min" if wall < 120 else "under_10min" if wall < 600 else "over_10min"
        digest.emit(disease, slug(pgs), "gnomon", "fit_wall", bucket)
        if outcome == "ok":
            predictions["gnomon"] = np.load(work / f"{disease}__{pgs}.npy")
            digest.emit(disease, slug(pgs), "gnomon", "status", "ok")
        else:
            label = outcome if outcome == "timeout" else "error_" + failure_class(work / f"{disease}__{pgs}.log")
            digest.emit(disease, slug(pgs), "gnomon", "status", label)
        report(digest, disease, pgs, test, predictions, config)
    publish_status(status, "benchmark_completed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("run")
    for name in ("config", "ancestry", "prune", "scores", "output", "work"):
        bench.add_argument(f"--{name}", type=Path, required=True)
    bench.add_argument("--features-uri", required=True)
    bench.add_argument("--status-uri", required=True)
    worker = sub.add_parser("gnomon-fit")
    for name in ("frame", "params", "output"):
        worker.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        gnomon_worker(args.frame, args.params, args.output)


if __name__ == "__main__":
    main()
