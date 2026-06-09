#!/usr/bin/env python3
"""Survival recalibration fit + ground-truth evaluation for one dataset.

Mirrors fit_binary.py with Cox PH baselines and the gamfit survival
marginal-slope:
  - gamfit  : gamfit survival marginal-slope, Surv(time,event), z=PGS_z,
              surfaces = matern(PCs, centers=C). Known to stall/crash on some
              builds (SauersML/gam#979); wrapped so its failure only drops its
              own rows. When it runs we report its global C-index.
  - linpc   : Cox on PGS_z + linear PCs.
  - znorm   : Cox on the ancestry-adjusted z-score.
  - rawpgs  : Cox on PGS_z alone (unadjusted reference).

The data has ONLY administrative censoring at a single horizon c_admin (=
max(surv_time)), so surv_event is exactly "failed by the admin horizon" and is
fully observed. That lets the survival arm reuse the binary calibration suite at
the admin horizon: predicted admin-horizon risk vs the known true admin risk
(surv_risk_true), scored with calibration-vs-truth AND the Brier Skill Score /
Murphy decomposition against the observed surv_event.

DISCRIMINATION: Harrell's C on (surv_time, surv_event), GLOBAL only.
CALIBRATION: per ancestry stratum, against ground truth (never within-stratum C).
"""
from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

import common

try:
    import gamfit
except Exception:  # noqa: BLE001
    gamfit = None

# gamfit survival MS can stall indefinitely (gam#979); a try/except cannot catch a
# native hang. So the reliable Cox baselines run FIRST and results are flushed to disk
# after EACH method -- an external timeout killing a gamfit hang then leaves the
# baseline CSVs intact. (Plot display order is set in the plotter, not here.)
METHODS = ["linpc", "znorm", "rawpgs", "gamfit"]


def _cox(fit_df, test_df, feat_cols, c_admin):
    """Fit Cox on feat_cols; return (prognostic index on test, admin-horizon
    cumulative-incidence risk on test = 1 - S(c_admin))."""
    cph = CoxPHFitter(penalizer=0.0)
    cph.fit(fit_df[feat_cols + ["surv_time", "surv_event"]], duration_col="surv_time", event_col="surv_event")
    pi = cph.predict_log_partial_hazard(test_df[feat_cols]).values.ravel()
    surv = cph.predict_survival_function(test_df[feat_cols], times=[c_admin]).iloc[0].values.ravel()
    return pi, common.clip01(1.0 - surv)


def fit_rawpgs(fit_df, test_df, pccols, c_admin):
    return _cox(fit_df, test_df, ["PGS_z"], c_admin)


def fit_linpc(fit_df, test_df, pccols, c_admin):
    return _cox(fit_df, test_df, ["PGS_z"] + pccols, c_admin)


def fit_znorm(fit_df, test_df, pccols, c_admin):
    coefs = common.znorm_fit(fit_df["PGS_raw"].values, fit_df[pccols].values)
    f = fit_df.copy()
    t = test_df.copy()
    f["Z"] = common.znorm_apply(coefs, fit_df["PGS_raw"].values, fit_df[pccols].values)
    t["Z"] = common.znorm_apply(coefs, test_df["PGS_raw"].values, test_df[pccols].values)
    return _cox(f, t, ["Z"], c_admin)


def fit_gamfit(fit_df, test_df, pccols, centers, c_admin):
    """gamfit survival marginal-slope. predict() semantics for survival are not
    a stable probability, so we use its output only as a risk-ordering prognostic
    index (-> C-index) and leave admin-horizon risk undefined (calibration NaN)."""
    if gamfit is None:
        raise RuntimeError("gamfit not importable")
    terms = f"matern({', '.join(pccols)}, centers={centers})"
    data_fit = {"time": fit_df["surv_time"].astype(float).values,
                "event": fit_df["surv_event"].astype(float).values,
                "z": fit_df["PGS_z"].astype(float).values}
    data_te = {"z": test_df["PGS_z"].astype(float).values}
    for c in pccols:
        data_fit[c] = fit_df[c].astype(float).values
        data_te[c] = test_df[c].astype(float).values
    model = gamfit.fit(data_fit, formula=f"Surv(time, event) ~ {terms}",
                       survival_likelihood="marginal-slope", z_column="z", logslope_formula=terms)
    pred = model.predict(data_te)
    pi = np.asarray(getattr(pred, "to_numpy", lambda: pred)(), dtype=float).ravel()[:len(test_df)]
    return pi, None


_FITTERS = {"linpc": fit_linpc, "znorm": fit_znorm, "rawpgs": fit_rawpgs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dem", required=True)
    ap.add_argument("--pheno", required=True)
    ap.add_argument("--pgs-mode", default="realpt")
    ap.add_argument("--centers", type=int, default=20)
    ap.add_argument("--out-acc", required=True)
    ap.add_argument("--out-cal", required=True)
    args = ap.parse_args()

    df = common.load_normalized(args.data)
    pccols = common.pc_cols(df)
    for need in ("surv_time", "surv_event"):
        if need not in df.columns:
            print(f"FATAL: missing {need}", file=sys.stderr)
            sys.exit(2)
    fit_df = df[df["split_role"] == "fit"].copy()
    test_df = df[df["split_role"] == "test"].copy()
    if fit_df.empty or test_df.empty:
        print(f"FATAL: empty split (fit={len(fit_df)} test={len(test_df)})", file=sys.stderr)
        sys.exit(2)

    # Administrative horizon: everyone is observed to c_admin = max(surv_time), so
    # surv_event is the fully-observed "failed by c_admin" indicator.
    c_admin = float(df["surv_time"].max())
    strata = common.ancestry_bins(test_df)
    t_te = test_df["surv_time"].values
    e_te = test_df["surv_event"].values
    risk_true_te = test_df["surv_risk_true"].values if "surv_risk_true" in test_df.columns else None
    acc_rows, cal_rows = [], []

    for method in METHODS:
        try:
            if method == "gamfit":
                pi, risk = fit_gamfit(fit_df, test_df, pccols, args.centers, c_admin)
            else:
                pi, risk = _FITTERS[method](fit_df, test_df, pccols, c_admin)
        except Exception as exc:  # noqa: BLE001
            print(f"[{method}] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            traceback.print_exc()
            continue

        try:
            cindex = float(concordance_index(t_te, -pi, e_te))  # higher PI -> shorter time
        except Exception:  # noqa: BLE001
            cindex = np.nan
        acc_rows.append(dict(dem=args.dem, pheno=args.pheno, pgs_mode=args.pgs_mode,
                             method=method, metric="cindex", value=cindex))

        for bk, bl, mask in strata:
            if risk is None or risk_true_te is None:
                continue  # gamfit survival: no calibrated risk (C-index only)
            cm, n = common.calib_vs_truth(risk_true_te[mask], risk[mask])
            sk = common.calib_skill(e_te[mask], risk[mask])
            for metric, val in {**cm, **sk}.items():
                cal_rows.append(dict(dem=args.dem, pheno=args.pheno, pgs_mode=args.pgs_mode,
                                     method=method, ancestry_bin_kind=bk, ancestry_bin=bl,
                                     n=int(n), metric=metric, value=val))
        print(f"[{method}] C-index={cindex:.4f}")
        # Flush after each method so a later gamfit hang (killed by an external
        # timeout) still leaves the completed baselines on disk.
        pd.DataFrame(acc_rows).to_csv(args.out_acc, index=False)
        pd.DataFrame(cal_rows).to_csv(args.out_cal, index=False)

    print(f"WROTE {args.out_acc} ({len(acc_rows)}), {args.out_cal} ({len(cal_rows)})")


if __name__ == "__main__":
    main()
