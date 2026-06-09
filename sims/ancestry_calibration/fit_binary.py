#!/usr/bin/env python3
"""Binary recalibration fit + ground-truth evaluation for one dataset.

Recalibration methods, in display order:
  - gamfit  : bernoulli marginal-slope, link=probit, z=PGS_z, marginal and
              log-slope surfaces = matern(PC1..PCk, centers=C). No wiggle.
  - linpc   : logistic on PGS_z + linear PCs (no z x PC interaction).
  - znorm   : ancestry mean+log-variance adjustment of PGS_raw as OLS on the
              PCs, then logistic on the adjusted z-score.
  - calpred : CalPred (Hou et al. 2024) binary analog -- heteroscedastic probit
              with the location AND the log-scale linear in the PCs.
  - rawpgs  : logistic on PGS_z alone (unadjusted reference).

TEST-SET DISCIPLINE: models are FIT on the recalibration rows (split_role=='fit',
spanning all ancestries) and ALL metrics are computed on the held-out rows
(split_role=='test'). The P+T 'GWAS' rows never appear here.

DISCRIMINATION (AUC, Brier, Nagelkerke and Lee-2011 liability R2) is on the real
outcome y_binary, GLOBAL only -- never within an ancestry stratum.
CALIBRATION is against the known generative risk p_true, per ancestry stratum:
calibration-vs-truth (probit slope, bias, rmse, mae) and the Brier Skill Score
with the Murphy reliability/resolution decomposition.
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import ndtri
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import common

try:
    import gamfit
except Exception:  # noqa: BLE001
    gamfit = None

# gamfit runs LAST and results flush after each method, so a slow/failed gamfit
# (gam#979) still leaves the four baselines on disk. Plot display order (gamfit
# first) is set in the plotter, independent of this execution order.
METHODS = ["linpc", "znorm", "calpred", "rawpgs", "gamfit"]


# --------------------------------------------------------------------------- #
# method fits -> predicted P(y=1) on the test rows
# --------------------------------------------------------------------------- #
def fit_rawpgs(fit_df, test_df, pccols):
    lr = LogisticRegression(max_iter=1000).fit(fit_df[["PGS_z"]].values, fit_df["y_binary"].values)
    return lr.predict_proba(test_df[["PGS_z"]].values)[:, 1]


def fit_linpc(fit_df, test_df, pccols):
    cols = ["PGS_z"] + pccols
    lr = LogisticRegression(max_iter=1000).fit(fit_df[cols].values, fit_df["y_binary"].values)
    return lr.predict_proba(test_df[cols].values)[:, 1]


def fit_znorm(fit_df, test_df, pccols):
    coefs = common.znorm_fit(fit_df["PGS_raw"].values, fit_df[pccols].values)
    z_fit = common.znorm_apply(coefs, fit_df["PGS_raw"].values, fit_df[pccols].values).reshape(-1, 1)
    z_te = common.znorm_apply(coefs, test_df["PGS_raw"].values, test_df[pccols].values).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000).fit(z_fit, fit_df["y_binary"].values)
    return lr.predict_proba(z_te)[:, 1]


def fit_calpred(fit_df, test_df, pccols):
    """CalPred binary analog: heteroscedastic probit. Location eta = X b with
    X = [1, PGS_z, PCs]; log-scale = Z g with Z = [1, PCs]; p = Phi(eta / exp(Z g))
    -- the PGS->risk mean AND its spread vary with ancestry (Hou et al. 2024)."""
    def design(d):
        X = np.column_stack([np.ones(len(d)), d["PGS_z"].values.astype(float),
                             d[pccols].values.astype(float)])
        Z = np.column_stack([np.ones(len(d)), d[pccols].values.astype(float)])
        return X, Z

    Xf, Zf = design(fit_df)
    yf = fit_df["y_binary"].values.astype(float)
    px, pz = Xf.shape[1], Zf.shape[1]

    def negll(th):
        b, g = th[:px], th[px:]
        s = np.exp(np.clip(Zf @ g, -4, 4))
        p = common.clip01(norm.cdf((Xf @ b) / s))
        return -float(np.sum(yf * np.log(p) + (1 - yf) * np.log(1 - p)))

    th0 = np.zeros(px + pz)
    try:
        lr = LogisticRegression(max_iter=1000).fit(Xf[:, 1:], yf)
        th0[0] = float(lr.intercept_[0]) * 0.5
        th0[1:px] = lr.coef_.ravel() * 0.5
    except Exception:  # noqa: BLE001
        pass
    res = minimize(negll, th0, method="L-BFGS-B", options={"maxiter": 800})
    b, g = res.x[:px], res.x[px:]
    Xt, Zt = design(test_df)
    st = np.exp(np.clip(Zt @ g, -4, 4))
    return common.clip01(norm.cdf((Xt @ b) / st))


def fit_gamfit(fit_df, test_df, pccols, centers):
    if gamfit is None:
        raise RuntimeError("gamfit not importable")
    terms = f"matern({', '.join(pccols)}, centers={centers})"
    data_fit = {"event": fit_df["y_binary"].astype(float).values, "z": fit_df["PGS_z"].astype(float).values}
    data_te = {"z": test_df["PGS_z"].astype(float).values}
    for c in pccols:
        data_fit[c] = fit_df[c].astype(float).values
        data_te[c] = test_df[c].astype(float).values
    model = gamfit.fit(data_fit, formula=f"event ~ {terms}", family="bernoulli-marginal-slope",
                       link="probit", z_column="z", logslope_formula=terms)
    pred = model.predict(data_te)
    if hasattr(pred, "columns") and "mean" in getattr(pred, "columns", []):
        return common.clip01(np.asarray(pred["mean"], dtype=float).ravel())
    arr = np.asarray(getattr(pred, "to_numpy", lambda: pred)(), dtype=float)
    return common.clip01(arr[:, -1].ravel() if arr.ndim == 2 else arr.ravel())


_FITTERS = {"linpc": fit_linpc, "znorm": fit_znorm, "calpred": fit_calpred, "rawpgs": fit_rawpgs}


# --------------------------------------------------------------------------- #
# global discrimination on the real outcome
# --------------------------------------------------------------------------- #
def _nagelkerke_r2(y, p):
    p = common.clip01(p)
    y = np.asarray(y, dtype=float)
    n = len(y)
    pbar = y.mean()
    if pbar <= 0 or pbar >= 1:
        return np.nan
    ll_full = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    ll_null = np.sum(y * np.log(pbar) + (1 - y) * np.log(1 - pbar))
    cox = 1.0 - np.exp((2.0 / n) * (ll_null - ll_full))
    rmax = 1.0 - np.exp((2.0 / n) * ll_null)
    return float(cox / rmax) if rmax > 0 else np.nan


def _lee_liability_r2(y, p):
    """Lee et al. 2011 liability-scale R2; population sampling (P==K)."""
    p = common.clip01(p)
    y = np.asarray(y, dtype=float)
    K = float(y.mean())
    if K <= 0 or K >= 1:
        return np.nan
    r2_obs = float(np.corrcoef(p, y)[0, 1] ** 2)
    z = norm.pdf(ndtri(1 - K))
    return float(r2_obs * K * (1 - K) / (z * z)) if z > 0 else np.nan


def global_discrimination(y, p):
    out = {}
    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:  # noqa: BLE001
        out["auc"] = np.nan
    try:
        out["brier"] = float(brier_score_loss(y, common.clip01(p)))
    except Exception:  # noqa: BLE001
        out["brier"] = np.nan
    out["nagelkerke_r2"] = _nagelkerke_r2(y, p)
    out["liability_r2"] = _lee_liability_r2(y, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--pgs-mode", default="realpt")
    ap.add_argument("--centers", type=int, default=20)
    ap.add_argument("--out-acc", required=True)
    ap.add_argument("--out-cal", required=True)
    ap.add_argument("--out-pred", default=None)
    args = ap.parse_args()

    df = common.load_normalized(args.data)
    pccols = common.pc_cols(df)
    if not pccols or "p_true" not in df.columns:
        print("FATAL: need PC columns and a p_true ground-truth column", file=sys.stderr)
        sys.exit(2)
    fit_df = df[df["split_role"] == "fit"].copy()
    test_df = df[df["split_role"] == "test"].copy()
    if fit_df.empty or test_df.empty:
        print(f"FATAL: empty split (fit={len(fit_df)} test={len(test_df)})", file=sys.stderr)
        sys.exit(2)

    strata = common.ancestry_bins(test_df)
    y_te = test_df["y_binary"].values
    p_true_te = test_df["p_true"].values
    acc_rows, cal_rows, preds = [], [], {}

    for method in METHODS:
        try:
            p = (fit_gamfit(fit_df, test_df, pccols, args.centers) if method == "gamfit"
                 else _FITTERS[method](fit_df, test_df, pccols))
        except Exception as exc:  # noqa: BLE001
            print(f"[{method}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue
        p = np.asarray(p, dtype=float)
        preds[method] = p
        for metric, val in global_discrimination(y_te, p).items():
            acc_rows.append(dict(dem=args.dem, pheno=args.pheno, pgs_mode=args.pgs_mode,
                                 method=method, metric=metric, value=val))
        for bk, bl, mask in strata:
            cm, n = common.calib_vs_truth(p_true_te[mask], p[mask])
            sk = common.calib_skill(y_te[mask], p[mask])
            for metric, val in {**cm, **sk}.items():
                cal_rows.append(dict(dem=args.dem, pheno=args.pheno, pgs_mode=args.pgs_mode,
                                     method=method, ancestry_bin_kind=bk, ancestry_bin=bl,
                                     n=int(n), metric=metric, value=val))
        a = dict((r["metric"], r["value"]) for r in acc_rows if r["method"] == method)
        print(f"[{method}] AUC={a['auc']:.4f} Brier={a['brier']:.4f} liabR2={a['liability_r2']:.4f}")
        # Flush after each method so a slow/killed gamfit still leaves baselines on disk.
        pd.DataFrame(acc_rows).to_csv(args.out_acc, index=False)
        pd.DataFrame(cal_rows).to_csv(args.out_cal, index=False)

    if args.out_pred and preds:
        idc = [c for c in ["iid", "deme", "dist_from_train", "is_train"] if c in test_df.columns]
        pw = test_df[idc].reset_index(drop=True)
        pw["dem"], pw["pheno"] = args.dem, args.pheno
        pw["y_binary"], pw["p_true"] = y_te, p_true_te
        for m, pp in preds.items():
            pw["p_" + m] = pp
        pw.to_parquet(args.out_pred, index=False)
    print(f"WROTE {args.out_acc} ({len(acc_rows)}), {args.out_cal} ({len(cal_rows)})")


if __name__ == "__main__":
    main()
