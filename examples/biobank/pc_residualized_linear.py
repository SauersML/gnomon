#!/usr/bin/env python3
"""Standard PRS survival baseline: pgscatalog-style Z_norm2 ancestry adjustment
on train, then Cox PH on (entry_age, exit_age, event) ~ z_norm2.

Z_norm2 is the two-step continuous PC-based normalization implemented in
eMERGE / pgsc_calc:
  - Step 1: mu_hat = OLS(PGS ~ PCs);  resid = PGS - mu_hat
  - Step 2: var_hat = OLS(resid^2 ~ PCs);  z_norm2 = resid / sqrt(var_hat)
Both regressions are fit on train and applied to test using train-only
coefficients. No population labels are required — this is the continuous-
ancestry adjustment, not a stratified one.

Reuses the cohort/loader/metric machinery from marginal_slope_diseases.py so
the splits and metric are bit-for-bit comparable with the GAM marginal-slope
survival run. Only the model differs: linear log-HR on a single Z_norm2
score vs. the GAM's smooth log-HR over PCs.

Reported per disease: held-out Harrell's C.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from lifelines import CoxPHFitter
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marginal_slope_diseases import (  # noqa: E402
    DISEASES,
    NUM_PCS,
    PGS_ID_PATTERN,
    RNG_SEED,
    TRAIN_FRACTION,
    ensure_scored,
    fetch_cases,
    fetch_person_times,
    load_one_pgs,
    load_pcs,
    load_sex,
    lookup_snomed_concept,
    metrics,
)


def z_norm2(
    train_pgs: np.ndarray,
    train_pcs: np.ndarray,
    test_pgs: np.ndarray,
    test_pcs: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """pgscatalog two-step PC-based normalization.

    Step 1 (mean):     mu_hat = OLS(PGS ~ PCs);    resid = PGS - mu_hat
    Step 2 (variance): var_hat = OLS(resid^2 ~ PCs), clipped at `eps`;
                       z      = resid / sqrt(var_hat)

    Both regressions are fit on train and applied to test using the train
    coefficients only — no test leakage. No population labels are used.
    """
    mean_reg = LinearRegression().fit(train_pcs, train_pgs)
    train_resid = train_pgs - mean_reg.predict(train_pcs)
    test_resid = test_pgs - mean_reg.predict(test_pcs)
    var_reg = LinearRegression().fit(train_pcs, train_resid ** 2)
    train_var = np.maximum(var_reg.predict(train_pcs), eps)
    test_var = np.maximum(var_reg.predict(test_pcs), eps)
    return train_resid / np.sqrt(train_var), test_resid / np.sqrt(test_var)


def main() -> None:
    diseases = {k: v for k, v in DISEASES.items() if PGS_ID_PATTERN.match(v["pgs"])}
    print(f"diseases: {list(diseases)}")
    ensure_scored([cfg["pgs"] for cfg in diseases.values()])

    print("loading PCs and sex ...")
    pcs = load_pcs(NUM_PCS)
    sex = load_sex()
    base = pcs.merge(sex, on="person_id")
    print(f"base: n={len(base):,}")

    cdr = os.environ["WORKSPACE_CDR"]
    client = bigquery.Client()
    rng = np.random.default_rng(RNG_SEED)
    pc_cols = [f"PC{i+1}" for i in range(NUM_PCS)]

    print("loading person times ...")
    times = fetch_person_times(client, cdr)
    base = base.merge(times, on="person_id")
    print(f"base (with times): n={len(base):,}")

    days_per_year = 365.25

    for name, cfg in diseases.items():
        print(f"\n=== {name.upper()} (Cox PH on Z_norm2 PRS) ===")
        pgs_df = load_one_pgs(cfg["pgs"])
        df_full = base.merge(pgs_df, on="person_id")
        ancestor = lookup_snomed_concept(client, cdr, cfg["snomed_name"])
        case_dates = fetch_cases(client, cdr, ancestor)
        df_full = df_full.merge(case_dates, on="person_id", how="left")
        df_full["event"] = df_full["event_date"].notna().astype(int)
        df_full["entry_age"] = (
            (df_full["obs_start"] - df_full["birth_datetime"]).dt.days / days_per_year
        )
        exit_date = df_full["event_date"].fillna(df_full["obs_end"])
        df_full["exit_age"] = (
            (exit_date - df_full["birth_datetime"]).dt.days / days_per_year
        )
        df_full = df_full.dropna(subset=["entry_age", "exit_age"]).copy()
        df_full = df_full[df_full["exit_age"] > df_full["entry_age"]].copy()
        df_full = df_full[df_full["entry_age"] >= 0].copy()

        n_event = int(df_full["event"].sum())
        n_censor = len(df_full) - n_event
        K = n_event / max(1, len(df_full))
        print(
            f"  snomed={cfg['snomed_name']!r}  concept_id={ancestor}  "
            f"events={n_event:,}  censors={n_censor:,}  K(crude)={K:.6f}"
        )

        event_idx = rng.permutation(df_full.index[df_full["event"] == 1].to_numpy())
        censor_idx = rng.permutation(df_full.index[df_full["event"] == 0].to_numpy())
        n_train_event = int(round(len(event_idx) * TRAIN_FRACTION))
        n_train_censor = int(round(len(censor_idx) * TRAIN_FRACTION))
        train_pick = np.concatenate([event_idx[:n_train_event], censor_idx[:n_train_censor]])
        test_pick = np.concatenate([event_idx[n_train_event:], censor_idx[n_train_censor:]])
        print(
            f"  split: n={len(df_full):,}  "
            f"train_events={n_train_event:,} train_censors={n_train_censor:,}  "
            f"test_events={len(event_idx) - n_train_event:,} "
            f"test_censors={len(censor_idx) - n_train_censor:,}"
        )
        train = df_full.loc[train_pick].reset_index(drop=True)
        test = df_full.loc[test_pick].reset_index(drop=True)

        z_train, z_test = z_norm2(
            train["pgs"].to_numpy(),
            train[pc_cols].to_numpy(),
            test["pgs"].to_numpy(),
            test[pc_cols].to_numpy(),
        )
        train["z_norm2"] = z_train
        test["z_norm2"] = z_test

        print(
            f"  fit_spec: model=Cox PH  formula='Surv(entry_age, exit_age, event) ~ "
            f"z_norm2(PGS | {' + '.join(pc_cols)})'"
        )
        cph = CoxPHFitter()
        cph.fit(
            train[["entry_age", "exit_age", "event", "z_norm2"]],
            duration_col="exit_age",
            entry_col="entry_age",
            event_col="event",
        )
        coef = float(cph.params_["z_norm2"])
        print(f"  coef:   log_HR={coef:+.4f}  HR/SD={np.exp(coef):.4f}")

        risk = cph.predict_partial_hazard(test[["z_norm2"]]).to_numpy()
        m = metrics(
            test["exit_age"].to_numpy(),
            test["event"].to_numpy(),
            risk,
        )

        print(
            f"  PGS={cfg['pgs']}  train_n={len(train):,}  test_n={m['n']:,}  "
            f"test_events={m['n_events']:,}  K(crude)={K:.6f}  "
            f"median_exit_age={m['median_exit_age']:.2f}"
        )
        print(
            f"  held-out  concordance={m['concordance']:.4f}"
        )


if __name__ == "__main__":
    main()
