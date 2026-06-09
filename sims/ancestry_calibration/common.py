"""Shared utilities for the ancestry-calibration study: dataset schema
normalization, PC/PGS helpers, the z-norm ancestry adjustment, ancestry strata,
and the ground-truth calibration metrics (calibration-vs-truth + Brier Skill
Score with the Murphy reliability/resolution decomposition).

Both the binary and survival evaluators import from here, so the metric
definitions live in exactly one place. Discrimination is arm-specific (binary
AUC vs Harrell's C) and stays in each evaluator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

EPS = 1e-6


# --------------------------------------------------------------------------- #
# dataset schema
# --------------------------------------------------------------------------- #
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Add a canonical ``split_role`` in {fit, test, train_only} and a derived
    ``surv_risk_true`` (admin-horizon cumulative incidence), without disturbing
    the producer columns. gen_real_pt writes the 4-level split
    (train-deme-fit / train-deme-test / other-deme-fit / other-deme-test):
    every ``*-fit`` row (spanning ancestries) is recalibration-training; every
    ``*-test`` row is held-out evaluation; the internal P+T 'GWAS' rows are
    carved privately inside gen_real_pt and never appear as a split here."""
    df = df.copy()
    if "split" in df.columns:
        s = df["split"].astype(str)

        def role(v: str) -> str:
            if v.endswith("-test") or v == "test":
                return "test"
            if v.endswith("-fit") or v == "cal":
                return "fit"
            return "train_only"

        df["split_role"] = s.map(role)
    if "is_train" not in df.columns and "dist_from_train" in df.columns:
        df["is_train"] = df["dist_from_train"].astype(float) == 0.0
    if "surv_risk_true" not in df.columns and "true_surv_at_admin" in df.columns:
        df["surv_risk_true"] = 1.0 - df["true_surv_at_admin"].astype(float)
    return df


def load_normalized(path: str) -> pd.DataFrame:
    return normalize(pd.read_parquet(path))


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def pc_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.startswith("PC") and c[2:].isdigit()],
                  key=lambda c: int(c[2:]))


def clip01(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def logit(p) -> np.ndarray:
    p = clip01(p)
    return np.log(p / (1.0 - p))


def probit(p) -> np.ndarray:
    """Inverse-normal CDF; matches the probit-liability generative model."""
    return norm.ppf(clip01(p))


def _slope_probit(p_true, p_pred) -> float:
    """OLS slope of probit(p_true) on probit(p_pred); 1.0 iff p_pred == p_true."""
    a, b = probit(p_true), probit(p_pred)
    if np.std(b) < 1e-9:
        return np.nan
    return float(np.cov(b, a, bias=True)[0, 1] / np.var(b))


# --------------------------------------------------------------------------- #
# z-norm ancestry adjustment (continuous PC regression of PGS mean + log-var)
# --------------------------------------------------------------------------- #
def znorm_fit(pgs_raw: np.ndarray, PC: np.ndarray) -> dict:
    X = np.column_stack([np.ones(len(pgs_raw)), PC])
    beta_mean, *_ = np.linalg.lstsq(X, pgs_raw, rcond=None)
    resid = pgs_raw - X @ beta_mean
    beta_var, *_ = np.linalg.lstsq(X, np.log(resid ** 2 + EPS), rcond=None)
    return {"beta_mean": beta_mean, "beta_var": beta_var}


def znorm_apply(coefs: dict, pgs_raw: np.ndarray, PC: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(pgs_raw)), PC])
    mu = X @ coefs["beta_mean"]
    sd = np.exp(0.5 * (X @ coefs["beta_var"]))
    sd = np.where(sd < EPS, EPS, sd)
    return (pgs_raw - mu) / sd


# --------------------------------------------------------------------------- #
# ancestry strata
# --------------------------------------------------------------------------- #
def _deme_key(x):
    s = str(x)
    digits = "".join(ch for ch in s if ch.isdigit())
    return (0, int(digits)) if digits else (1, s)


def ancestry_bins(test_df: pd.DataFrame, n_dist_bins: int = 5):
    """Yield (bin_kind, bin_label, boolean_mask) over the test set:
    the held-out training-ancestry vs the other ancestries, each deme, and
    genetic-distance quantile bins from the training deme."""
    out = []
    if "is_train" in test_df.columns:
        is_tr = test_df["is_train"].values.astype(bool)
        out.append(("train_ancestry", "train_deme", is_tr))
        out.append(("train_ancestry", "other_deme", ~is_tr))
    if "deme" in test_df.columns:
        for d in sorted(test_df["deme"].dropna().unique(), key=_deme_key):
            out.append(("deme", str(d), (test_df["deme"] == d).values))
    if "dist_from_train" in test_df.columns and test_df["dist_from_train"].notna().any():
        dist = test_df["dist_from_train"].values.astype(float)
        qs = np.unique(np.quantile(dist, np.linspace(0, 1, n_dist_bins + 1)))
        if len(qs) >= 3:
            labels = np.digitize(dist, qs[1:-1])
            for b in range(len(qs) - 1):
                m = labels == b
                if m.any():
                    out.append(("dist_bin", f"q{b}:[{qs[b]:.3g},{qs[b + 1]:.3g}]", m))
    return out


# --------------------------------------------------------------------------- #
# ground-truth calibration metrics (shared by both arms)
# --------------------------------------------------------------------------- #
def calib_vs_truth(p_true, p_pred) -> tuple[dict, int]:
    """Calibration of predicted risk against the KNOWN generative risk on one
    stratum (we control the simulation, so we never fit to the noisy outcome).
    Probit slope is the principled summary (1.0 = calibrated); bias/rmse/mae
    are reported alongside."""
    p_true = clip01(p_true)
    p_pred = clip01(p_pred)
    n = len(p_true)
    keys = ("calib_bias", "calib_slope_probit", "abs_slope_minus_1_probit", "rmse", "mae")
    if n < 10:
        return {k: np.nan for k in keys}, n
    slope = _slope_probit(p_true, p_pred)
    return {
        "calib_bias": float(np.mean(p_pred) - np.mean(p_true)),
        "calib_slope_probit": slope,
        "abs_slope_minus_1_probit": abs(slope - 1.0) if np.isfinite(slope) else np.nan,
        "rmse": float(np.sqrt(np.mean((p_pred - p_true) ** 2))),
        "mae": float(np.mean(np.abs(p_pred - p_true))),
    }, n


def calib_skill(y, p, n_bins: int = 10) -> dict:
    """Brier Skill Score (vs the stratum base rate) and the Murphy decomposition
    of the Brier score: Brier = reliability - resolution + uncertainty.
    reliability = miscalibration (lower better); resolution = how far bin
    outcome rates move from the base rate (higher better); BSS > 0 means the
    predictions beat the stratum base rate."""
    y = np.asarray(y, dtype=float)
    p = clip01(p)
    n = len(y)
    keys = ("brier_y", "bss", "reliability", "resolution", "uncertainty")
    if n < 20 or y.min() == y.max():
        return {k: np.nan for k in keys}
    ybar = float(y.mean())
    unc = ybar * (1.0 - ybar)
    bs = float(np.mean((p - y) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            nb = int(m.sum())
            rel += nb / n * (p[m].mean() - y[m].mean()) ** 2
            res += nb / n * (y[m].mean() - ybar) ** 2
    return {
        "brier_y": bs,
        "bss": float(1.0 - bs / unc) if unc > 0 else np.nan,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
    }
