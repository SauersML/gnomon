"""Shared utilities for the ancestry-calibration study: dataset schema
normalization, PC/PGS helpers, the z-norm ancestry adjustment, ancestry strata,
and the ground-truth risk metrics.

Both the binary and survival evaluators import from here, so the metric
definitions live in exactly one place. Discrimination is arm-specific (binary
AUC vs Harrell's C) and stays in each evaluator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import binomtest, norm, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

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


def _probit_risk_slope_ratio(p_true, p_pred) -> float:
    """Direct slope ratio on the known probit-risk scale; 1.0 is ideal."""
    true_z = probit(p_true)
    pred_z = probit(p_pred)
    true_sd = float(np.std(true_z))
    if true_sd < 1e-9:
        return np.nan
    return float(np.std(pred_z) / true_sd)


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
# ground-truth risk metrics (shared by both arms)
# --------------------------------------------------------------------------- #
def _corr(fn, a, b) -> float:
    """Correlation guarding against constant input (undefined => nan)."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    c = fn(a, b)[0]
    return float(c) if c == c else np.nan


def _tail_rmse_true(p_true, p_pred, q: float = 0.9) -> float:
    """RMSE of predicted vs true risk among the highest-true-risk individuals
    (>= q-quantile of p_true) -- accuracy where it matters clinically."""
    thr = float(np.quantile(p_true, q))
    m = p_true >= thr
    if m.sum() < 3:
        return np.nan
    return float(np.sqrt(np.mean((p_pred[m] - p_true[m]) ** 2)))


def _rr_top_true(p_true, p_pred, q: float) -> float:
    """True-risk ratio of the top (1-q) fraction by PREDICTED score vs the rest:
    does flagging the top X% by score actually enrich for genuinely high true
    risk? Higher = better high-risk identification (the canonical clinical use)."""
    flag = p_pred >= float(np.quantile(p_pred, q))
    if flag.sum() < 3 or (~flag).sum() < 3:
        return np.nan
    return float((p_true[flag].mean() + EPS) / (p_true[~flag].mean() + EPS))


def _rr_top_obs(y, p_pred, q: float) -> float:
    """OBSERVED risk ratio of the top (1-q) fraction by score vs the rest -- the
    clinical headline ('top X% carry N-fold the event rate'). Uses outcomes."""
    y = np.asarray(y, dtype=float)
    flag = p_pred >= float(np.quantile(p_pred, q))
    if flag.sum() < 3 or (~flag).sum() < 3:
        return np.nan
    return float((y[flag].mean() + EPS) / (y[~flag].mean() + EPS))


def _frac_ge_kx(p, k: float) -> float:
    """Fraction of the stratum whose risk is >= k x the mean risk -- the size of
    the >=k-fold-risk group. Invariant to multiplicative miscalibration; sensitive
    to the spread of the risk distribution (which attenuates out-of-ancestry)."""
    p = np.asarray(p, dtype=float)
    m = float(np.mean(p))
    if m <= 0:
        return np.nan
    return float(np.mean(p >= k * m))


def calibration(y, p_pred) -> tuple[float, float]:
    """(calibration slope, calibration-in-the-large intercept). Slope = b from the
    logistic fit y ~ a + b*logit(p): 1 = right spread, <1 = predictions too
    extreme, >1 = too compressed. Intercept = the offset a* solving
    mean(sigmoid(a*+logit(p)))=mean(y) (slope fixed at 1): 0 = right average level,
    >0 = under-predicts on average. nan if a stratum has one class."""
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan
    lp = logit(p_pred)
    slope = np.nan
    if float(np.std(lp)) > 1e-9:
        try:
            slope = float(LogisticRegression(C=1e6, max_iter=2000)
                          .fit(lp.reshape(-1, 1), y).coef_[0][0])
        except Exception:  # noqa: BLE001
            slope = np.nan
    ybar = float(y.mean())
    try:
        intercept = float(brentq(lambda a: float(np.mean(expit(a + lp))) - ybar, -25.0, 25.0))
    except Exception:  # noqa: BLE001
        intercept = np.nan
    return slope, intercept


def or_per_sd(y, p_pred) -> float:
    """Odds ratio per 1 SD of the standardized risk score (logit scale) -- the
    canonical PGS effect-size metric. Higher = stronger per-SD risk
    stratification. Needs the outcome, so it is discrimination-flavoured (not a
    vs-p_true oracle metric); nan if a stratum has one class or a constant score."""
    y = np.asarray(y).astype(int)
    z = logit(p_pred)
    sd = float(np.std(z))
    if len(np.unique(y)) < 2 or sd < 1e-9:
        return np.nan
    z = ((z - z.mean()) / sd).reshape(-1, 1)
    coef = LogisticRegression(C=1e6, max_iter=2000).fit(z, y).coef_[0][0]
    return float(np.exp(coef))


def risk_vs_truth(p_true, p_pred) -> tuple[dict, int]:
    """Prediction error against the known generative risk on one stratum.

    The probit slope ratio is derived directly from the known true risk and
    predicted risk distributions. It is not a fitted regression slope.

    Beyond the aggregate-calibration metrics (avg error, slope ratio, rmse, mae),
    we report rank/linear fidelity to the true per-individual risk and accuracy
    in the high-risk tail: out-of-ancestry these separate gamfit (which preserves
    the individual risk gradient) from recalibrators that only fix the mean/scale.
    """
    p_true = clip01(p_true)
    p_pred = clip01(p_pred)
    n = len(p_true)
    keys = ("avg_pred_minus_true_risk", "probit_risk_slope_ratio", "rmse", "mae",
            "spearman_true", "pearson_true", "r2_true", "tail_rmse_true",
            "rr_top10_true", "rr_top20_true",
            "frac_ge_2x_true", "frac_ge_3x_true", "frac_ge_2x_pred", "frac_ge_3x_pred")
    if n < 10:
        return {k: np.nan for k in keys}, n
    ss_tot = float(np.sum((p_true - p_true.mean()) ** 2))
    return {
        "avg_pred_minus_true_risk": float(np.mean(p_pred) - np.mean(p_true)),
        "probit_risk_slope_ratio": _probit_risk_slope_ratio(p_true, p_pred),
        "rmse": float(np.sqrt(np.mean((p_pred - p_true) ** 2))),
        "mae": float(np.mean(np.abs(p_pred - p_true))),
        "spearman_true": _corr(spearmanr, p_pred, p_true),
        "pearson_true": _corr(pearsonr, p_pred, p_true),
        "r2_true": float(1.0 - np.sum((p_pred - p_true) ** 2) / ss_tot) if ss_tot > 0 else np.nan,
        "tail_rmse_true": _tail_rmse_true(p_true, p_pred),
        # clinical risk-stratification -- two dual views, vs the ORACLE true risk:
        "rr_top10_true": _rr_top_true(p_true, p_pred, 0.90),  # top 10% true-risk enrichment
        "rr_top20_true": _rr_top_true(p_true, p_pred, 0.80),  # top 20%
        "frac_ge_2x_true": _frac_ge_kx(p_true, 2.0),   # true size of the >=2x-risk group
        "frac_ge_3x_true": _frac_ge_kx(p_true, 3.0),
        "frac_ge_2x_pred": _frac_ge_kx(p_pred, 2.0),   # this method's estimate of it
        "frac_ge_3x_pred": _frac_ge_kx(p_pred, 3.0),
    }, n


def _wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for a binomial rate -- valid at small n. Delegates to
    scipy.stats.binomtest(...).proportion_ci(method='wilson'), which is the
    closed-form Wilson score interval; equivalent to the prior hand-coded formula
    to machine precision. The ``z`` arg is retained for signature compatibility and
    is mapped to the corresponding two-sided confidence level."""
    if n <= 0 or k < 0 or k > n:
        return np.nan, np.nan
    conf = float(2.0 * norm.cdf(z) - 1.0)
    ci = binomtest(int(k), int(n)).proportion_ci(method="wilson", confidence_level=conf)
    return float(ci.low), float(ci.high)


def calibration_coverage(y, p_pred, bins: int = 10, min_per_bin: int = 20) -> float:
    """Interval-coverage of calibration: split into equal-count predicted-risk
    bins; a bin is 'covered' if its mean predicted risk lies inside the Wilson
    95%% CI of its OBSERVED event rate. Returns the fraction of bins covered
    (target ~0.95). Needs only outcomes + predictions (no model posterior).
    Out-of-ancestry, overconfident/mis-scaled risks fall outside -> coverage drops."""
    y = np.asarray(y).astype(int)
    p = clip01(p_pred)
    n = len(y)
    if n < bins * min_per_bin or y.min() == y.max():
        return np.nan
    order = np.argsort(p)
    edges = np.linspace(0, n, bins + 1).astype(int)
    covered = total = 0
    for b in range(bins):
        idx = order[edges[b]:edges[b + 1]]
        if len(idx) < min_per_bin:
            continue
        lo, hi = _wilson_ci(int(y[idx].sum()), len(idx))
        if lo <= float(p[idx].mean()) <= hi:
            covered += 1
        total += 1
    return float(covered / total) if total > 0 else np.nan


def outcome_metrics(y, p_pred) -> dict:
    """Outcome-based accuracy/stratification on one stratum (needs the realized
    outcome, so these match what the biobank can compute on real data):
    calibration slope + intercept, the OBSERVED top-10%/20% risk ratios, and
    calibration interval-coverage."""
    y = np.asarray(y).astype(int)
    keys = ("calibration_slope", "calibration_intercept", "rr_top10_obs",
            "rr_top20_obs", "calibration_coverage")
    if len(y) < 20 or y.min() == y.max():
        return {k: np.nan for k in keys}
    slope, intercept = calibration(y, p_pred)
    return {
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "rr_top10_obs": _rr_top_obs(y, p_pred, 0.90),
        "rr_top20_obs": _rr_top_obs(y, p_pred, 0.80),
        "calibration_coverage": calibration_coverage(y, p_pred),
    }


def brier_skill(y, p) -> dict:
    """Brier Skill Score versus the stratum base rate."""
    y = np.asarray(y, dtype=float)
    p = clip01(p)
    n = len(y)
    keys = ("bss",)
    if n < 20 or y.min() == y.max():
        return {k: np.nan for k in keys}
    ybar = float(y.mean())
    unc = ybar * (1.0 - ybar)
    bs = float(brier_score_loss(y, p, pos_label=1))
    return {
        "bss": float(1.0 - bs / unc) if unc > 0 else np.nan,
    }
