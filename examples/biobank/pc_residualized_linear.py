#!/usr/bin/env python3
"""Standard PRS baseline: residualize PGS against PCs + sex on train,
then fit case ~ residualized_prs_z with logistic regression.

Reuses the cohort/loader machinery from marginal_slope_diseases.py so the
splits, K, and metrics are bit-for-bit comparable with the marginal-slope
run. The only difference is the model.

Reported per disease: AUROC, Nagelkerke R^2, Lee-2011 liability R^2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from marginal_slope_diseases import (  # noqa: E402
    DISEASES,
    NUM_PCS,
    PGS_ID_PATTERN,
    RNG_SEED,
    TRAIN_FRACTION,
    ensure_scored,
    fetch_cases,
    load_one_pgs,
    load_pcs,
    load_sex,
    lookup_snomed_concept,
    metrics,
)


def residualize(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Fit prs_z ~ cols on train; return train + test residuals.

    Standard PRS practice: project ancestry/sex effects out of the score before
    using it as a predictor, so the downstream coefficient reflects PGS effect
    net of stratification.
    """
    reg = LinearRegression().fit(train[cols].to_numpy(), train["prs_z"].to_numpy())
    return (
        train["prs_z"].to_numpy() - reg.predict(train[cols].to_numpy()),
        test["prs_z"].to_numpy() - reg.predict(test[cols].to_numpy()),
    )


def fit_logistic(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Plain logistic regression on a single residualized score (+ intercept)."""
    return LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(
        x_train.reshape(-1, 1), y_train
    )


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
    covar_cols = [f"PC{i+1}" for i in range(NUM_PCS)] + ["sex"]

    for name, cfg in diseases.items():
        print(f"\n=== {name.upper()} (PC-residualized linear) ===")
        pgs_df = load_one_pgs(cfg["pgs"])
        df_full = base.merge(pgs_df, on="person_id")
        ancestor = lookup_snomed_concept(client, cdr, cfg["snomed_name"])
        cases = fetch_cases(client, cdr, ancestor)
        df_full["case"] = df_full["person_id"].isin(cases).astype(int)
        case_idx = rng.permutation(df_full.index[df_full["case"] == 1].to_numpy())
        ctrl_idx = rng.permutation(df_full.index[df_full["case"] == 0].to_numpy())
        K = len(case_idx) / max(1, len(case_idx) + len(ctrl_idx))
        print(
            f"  snomed={cfg['snomed_name']!r}  concept_id={ancestor}  "
            f"cases={len(case_idx):,}  controls={len(ctrl_idx):,}  K={K:.6f}"
        )

        n_cases = len(case_idx)
        n_ctrl_sampled = min(n_cases, len(ctrl_idx))
        ctrl_sampled = ctrl_idx[:n_ctrl_sampled]
        n_train_per_class = int(round(n_cases * TRAIN_FRACTION))
        n_train_ctrl = int(round(n_ctrl_sampled * TRAIN_FRACTION))
        train_pick = np.concatenate([case_idx[:n_train_per_class], ctrl_sampled[:n_train_ctrl]])
        test_pick = np.concatenate([case_idx[n_train_per_class:], ctrl_sampled[n_train_ctrl:]])
        print(
            f"  split: balanced n={n_cases + n_ctrl_sampled:,}  "
            f"train_cases={n_train_per_class:,} train_controls={n_train_ctrl:,}  "
            f"test_cases={n_cases - n_train_per_class:,} test_controls={n_ctrl_sampled - n_train_ctrl:,}"
        )
        train = df_full.loc[train_pick].reset_index(drop=True)
        test = df_full.loc[test_pick].reset_index(drop=True)

        pgs_mean = float(train["pgs"].mean())
        pgs_std = float(train["pgs"].std(ddof=0))
        train["prs_z"] = (train["pgs"] - pgs_mean) / pgs_std
        test["prs_z"] = (test["pgs"] - pgs_mean) / pgs_std

        x_train, x_test = residualize(train, test, covar_cols)
        # Re-standardize on train so the logistic intercept/slope are on a
        # familiar scale (z-score), without re-leaking test stats.
        rmu = float(x_train.mean())
        rsd = float(x_train.std(ddof=0)) or 1.0
        x_train = (x_train - rmu) / rsd
        x_test = (x_test - rmu) / rsd

        print(
            f"  fit_spec: family=bernoulli  link=logit  formula='case ~ resid(prs_z | {' + '.join(covar_cols)})'"
        )
        model = fit_logistic(x_train, train["case"].to_numpy())
        coef = float(model.coef_[0, 0])
        intercept = float(model.intercept_[0])
        print(f"  coef:   intercept={intercept:+.4f}  slope_resid_prs={coef:+.4f} (logit / SD)")

        p_test = model.predict_proba(x_test.reshape(-1, 1))[:, 1]
        m = metrics(test["case"].to_numpy(), p_test, K)

        print(
            f"  PGS={cfg['pgs']}  train_n={len(train):,}  test_n={m['n']:,}  "
            f"test_cases={m['cases']:,}  P={m['P']:.4f}  K={K:.6f}"
        )
        print(
            f"  held-out  AUROC={m['auroc']:.4f}  "
            f"Nagelkerke R^2={m['nagelkerke_r2']:.4f}  liability R^2={m['liability_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
