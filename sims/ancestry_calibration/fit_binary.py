"""Fit binary real-P+T ancestry-calibration models and compute truth-based metrics.

Inputs are datasets from gen_real_pt.py with the canonical split:
pgs_train, model_train, test. This script never uses pgs_train for model
training and never uses model_train for reported metrics.

Methods:
- gamfit marginal-slope, probit, Matern PC surface, default centers=20
- PGS + PCs linear logistic model
- znorm2: continuous PC-adjusted PGS z-score (mean and variance modelled as
  linear functions of the PCs, no discrete population labels), then logistic

Metrics:
- global held-out discrimination: AUC, Brier, liability-scale R2
- group held-out calibration/error by distance, all against p_true
- individual absolute error for Figure 2b
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import norm
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "sims" / "results_hpc" / "ancestry_calibration" / "results"

METHODS = ("gamfit", "linpc", "znorm2")
CENTERS = 20
FIT_JOBS = 6

DATASET_RE = re.compile(r"^(serial1d|grid2d|admix)_(phenoA|phenoB)_realpt_s(\d+)$")


def clip_prob(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def probit(p: np.ndarray) -> np.ndarray:
    return norm.ppf(clip_prob(p))


def pc_columns(df: pd.DataFrame) -> list[str]:
    cols = sorted([c for c in df.columns if c.startswith("PC")], key=lambda c: int(c[2:]))
    if not cols:
        raise ValueError("dataset has no PC columns")
    return cols


def require_columns(df: pd.DataFrame) -> None:
    required = {
        "iid",
        "deme",
        "dist_from_train",
        "PGS_raw",
        "PGS_z",
        "y_binary",
        "true_liab",
        "true_slope_deme",
        "intercept_deme",
        "p_true",
        "split",
        "pgs_mode",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    got = set(df["split"].astype(str).unique())
    want = {"pgs_train", "model_train", "test"}
    if got != want:
        raise ValueError(f"expected split values {sorted(want)}, got {sorted(got)}")
    if set(df["pgs_mode"].astype(str)) != {"realpt"}:
        raise ValueError("fit_binary only accepts pgs_mode == realpt")


def logistic_fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    model = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000)
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def fit_linpc(train: pd.DataFrame, test: pd.DataFrame, pcs: list[str]) -> np.ndarray:
    features = ["PGS_z", *pcs]
    return logistic_fit_predict(train[features].to_numpy(), train["y_binary"].to_numpy(), test[features].to_numpy())


def fit_znorm2(train: pd.DataFrame, test: pd.DataFrame, pcs: list[str]) -> np.ndarray:
    """Continuous PC-adjusted PGS z-score (PGS Catalog / Khera-2019 full adjustment).

    Ancestry is treated as a continuum via the PCs: the raw PGS mean and variance
    are each modelled as linear functions of the PCs (fit on model_train only), and
    every individual is standardized by their PC-predicted mean and SD --
    Z = (PGS_raw - mu_hat(PC)) / sd_hat(PC). No discrete population labels are used.
    The standardized score is then passed through a logistic model of the outcome.
    """
    Xtr = train[pcs].to_numpy(dtype=float)
    Xte = test[pcs].to_numpy(dtype=float)
    raw_tr = train["PGS_raw"].to_numpy(dtype=float)
    raw_te = test["PGS_raw"].to_numpy(dtype=float)

    mean_model = LinearRegression().fit(Xtr, raw_tr)
    mu_tr = mean_model.predict(Xtr)
    mu_te = mean_model.predict(Xte)

    resid_tr = raw_tr - mu_tr
    # variance as a (non-negative) linear function of the PCs; floor to a small
    # positive fraction of the global residual variance for numerical safety.
    var_floor = max(float(np.var(resid_tr)) * 1e-4, 1e-12)
    var_model = LinearRegression().fit(Xtr, resid_tr ** 2)
    sd_tr = np.sqrt(np.clip(var_model.predict(Xtr), var_floor, None))
    sd_te = np.sqrt(np.clip(var_model.predict(Xte), var_floor, None))

    z_train = ((raw_tr - mu_tr) / sd_tr).reshape(-1, 1)
    z_test = ((raw_te - mu_te) / sd_te).reshape(-1, 1)
    return logistic_fit_predict(z_train, train["y_binary"].to_numpy(), z_test)


def extract_gamfit_mean(prediction: object) -> np.ndarray:
    if isinstance(prediction, pd.DataFrame):
        if "mean" not in prediction.columns:
            raise ValueError(f"gamfit prediction DataFrame lacks 'mean': {list(prediction.columns)}")
        return prediction["mean"].to_numpy(dtype=float)
    arr = np.asarray(prediction, dtype=float)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr[:, 1]
    raise ValueError(f"unexpected gamfit prediction shape {arr.shape}")


def fit_gamfit(train: pd.DataFrame, test: pd.DataFrame, pcs: list[str]) -> np.ndarray:
    import gamfit

    pc_term = ", ".join(pcs)
    surface = f"matern({pc_term}, centers={CENTERS})"
    formula = f"y_binary ~ {surface}"
    cols = ["y_binary", "PGS_z", *pcs]
    model = gamfit.fit(
        train[cols],
        formula,
        family="bernoulli-marginal-slope",
        link="probit",
        z_column="PGS_z",
        logslope_formula=surface,
    )
    return extract_gamfit_mean(model.predict(test[cols]))


def global_metrics(y: np.ndarray, p: np.ndarray, p_true: np.ndarray, true_liab: np.ndarray) -> dict[str, float]:
    if not np.isfinite(p).all():   # method failed upstream (NaN sentinel) -> all metrics NaN
        return {k: float("nan") for k in ("brier", "mae_true_risk", "auc", "liability_pseudo_r2")}
    eta = probit(p)   # probit() clips internally only for finiteness; metrics use raw p
    out: dict[str, float] = {
        "brier": float(brier_score_loss(y, p)),
        "mae_true_risk": float(mean_absolute_error(p_true, p)),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    if len(np.unique(eta)) > 1 and len(np.unique(true_liab)) > 1:
        out["liability_pseudo_r2"] = float(pearsonr(true_liab, eta).statistic ** 2)
    else:
        out["liability_pseudo_r2"] = float("nan")
    return out


def group_metrics(frame: pd.DataFrame, method: str, p: np.ndarray, group_col: str, group_kind: str) -> list[dict[str, object]]:
    if not np.isfinite(p).all():   # method failed upstream -> no group rows
        return []
    work = frame.copy()
    work["p_hat"] = clip_prob(p)
    rows: list[dict[str, object]] = []
    for group, sub in work.groupby(group_col, sort=True):
        y = sub["y_binary"].to_numpy()
        p_hat = sub["p_hat"].to_numpy()
        eta_hat = probit(p_hat)
        true_liab = sub["true_liab"].to_numpy()
        pred_prev = float(sub["p_hat"].mean())
        true_prev = float(sub["p_true"].mean())
        auc = float(roc_auc_score(y, p_hat)) if len(np.unique(y)) == 2 else float("nan")
        if len(np.unique(eta_hat)) > 1 and len(np.unique(true_liab)) > 1:
            liability_pseudo_r2 = float(pearsonr(true_liab, eta_hat).statistic ** 2)
        else:
            liability_pseudo_r2 = float("nan")
        rows.append(
            {
                "method": method,
                "group_kind": group_kind,
                "group": group,
                "dist_from_train": float(sub["dist_from_train"].mean()),
                "n": int(len(sub)),
                "known_true_prevalence": true_prev,
                "derived_prediction_prevalence": pred_prev,
                "prevalence_error": pred_prev - true_prev,
                "abs_prevalence_error": abs(pred_prev - true_prev),
                "auc": auc,
                "liability_pseudo_r2": liability_pseudo_r2,
                "mae_true_risk": float(mean_absolute_error(sub["p_true"], sub["p_hat"])),
                "brier": float(brier_score_loss(sub["y_binary"], sub["p_hat"])),
            }
        )
    return rows


def evaluate_dataset(path: Path, outdir: Path) -> dict[str, Path]:
    df = pd.read_parquet(path)
    require_columns(df)
    pcs = pc_columns(df)
    train = df[df["split"] == "model_train"].copy()
    test = df[df["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError(f"{path}: model_train/test split empty")

    def _safe(name, fn):
        # one method failing (e.g. gamfit _rust.IntegrationError at large N) must not
        # kill the whole pool via an unpicklable exception; record NaN and continue.
        try:
            return np.asarray(fn(), dtype=float)
        except Exception as e:
            print(f"WARN {path.name}: method {name} failed -> NaN: {type(e).__name__}: {e}", flush=True)
            return np.full(len(test), np.nan)

    predictions = {
        "gamfit": _safe("gamfit", lambda: fit_gamfit(train, test, pcs)),
        "linpc": _safe("linpc", lambda: fit_linpc(train, test, pcs)),
        "znorm2": _safe("znorm2", lambda: fit_znorm2(train, test, pcs)),
    }

    stem = path.stem.replace("_realpt", "")
    match = DATASET_RE.fullmatch(path.stem)
    if not match:
        raise ValueError(f"unexpected real-P+T dataset name: {path.name}")
    dem_name, pheno, seed_text = match.groups()
    seed = int(seed_text)

    accuracy_rows = []
    group_rows = []
    pred_frames = []
    for method, p in predictions.items():
        gm = global_metrics(
            test["y_binary"].to_numpy(),
            p,
            test["p_true"].to_numpy(),
            test["true_liab"].to_numpy(),
        )
        for metric, value in gm.items():
            accuracy_rows.append(
                {"dem": dem_name, "pheno": pheno, "method": method, "metric": metric, "value": value}
            )
        group_rows.extend(group_metrics(test, method, p, "dist_from_train", "distance"))
        pf = test[
            ["iid", "deme", "dist_from_train", "PGS_z", "y_binary", "p_true", "true_liab", "true_slope_deme"]
        ].copy()
        pf["method"] = method
        pf["p_hat"] = clip_prob(p)
        pf["abs_error_true_risk"] = np.abs(pf["p_hat"] - pf["p_true"])
        pred_frames.append(pf)

    common = {"dem": dem_name, "pheno": pheno, "seed": seed, "pgs_mode": "realpt", "centers": CENTERS}
    acc = pd.DataFrame(accuracy_rows).assign(**common)
    groups = pd.DataFrame(group_rows).assign(**common)
    preds = pd.concat(pred_frames, ignore_index=True).assign(**common)

    outdir.mkdir(parents=True, exist_ok=True)
    acc_path = outdir / f"{stem}_accuracy.csv"
    group_path = outdir / f"{stem}_group_metrics.csv"
    pred_path = outdir / f"{stem}_individual_errors.csv"
    acc.to_csv(acc_path, index=False)
    groups.to_csv(group_path, index=False)
    preds.to_csv(pred_path, index=False)
    return {"accuracy": acc_path, "group": group_path, "individual": pred_path}


def main() -> None:
    global CENTERS              # reassigned below from --centers
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", type=Path)
    parser.add_argument("--run-tag", default="", help="output subdir for sweep isolation")
    parser.add_argument("--centers", type=int, default=CENTERS, help="marginal-slope surface centers")
    args = parser.parse_args()

    CENTERS = args.centers      # picked up by forked pool workers (surface formula)
    if args.run_tag:
        outdir = REPO_ROOT / "sims" / "results_hpc" / "ancestry_calibration" / args.run_tag / "results"
    else:
        outdir = RESULTS_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = max(1, min(FIT_JOBS, len(args.datasets)))
    all_acc = []
    all_group = []
    all_indiv = []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(evaluate_dataset, path, outdir): path for path in args.datasets}
        for fut in as_completed(futures):
            path = futures[fut]
            paths = fut.result()
            print(f"done fit {path}", flush=True)
            all_acc.append(pd.read_csv(paths["accuracy"]))
            all_group.append(pd.read_csv(paths["group"]))
            all_indiv.append(pd.read_csv(paths["individual"]))

    pd.concat(all_acc, ignore_index=True).to_csv(outdir / "accuracy_realpt_binary.csv", index=False)
    pd.concat(all_group, ignore_index=True).to_csv(outdir / "group_metrics_realpt_binary.csv", index=False)
    pd.concat(all_indiv, ignore_index=True).to_csv(outdir / "individual_errors_realpt_binary.csv", index=False)
    (outdir / "fit_config.json").write_text(
        json.dumps({"centers": CENTERS, "methods": METHODS, "fit_jobs": jobs}, indent=2)
    )


if __name__ == "__main__":
    main()
