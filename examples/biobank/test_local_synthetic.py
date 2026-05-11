#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gamfit",
#     "numpy",
#     "pandas",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""Local sanity check for the Bernoulli marginal-slope fit used by
marginal_slope_diseases.py — no AoU, no gnomon, no BigQuery.

Generates a synthetic cohort (PCs, sex, PGS, case label) with plausible
structure (PGS depends on PCs and sex; liability is a probit mix of PCs,
sex, and PGS), subsamples 100 cases + 100 controls per scenario, and
fits the same `case ~ duchon(PC1..PC10) + sex` GAM with `prs_z` driving
the marginal-slope log-slope channel through the same Duchon smooth.

Run:  uv run examples/biobank/test_local_synthetic.py
"""

from __future__ import annotations

import gamfit
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

NUM_PCS = 10
DUCHON_CENTERS = 4 * NUM_PCS
N_TOTAL = 5000
N_CASES = 100
N_CONTROLS = 100
RNG_SEED = 0

SCENARIOS = {
    "copd-like":          {"prevalence": 0.06, "pgs_effect": 0.6},
    "hypertension-like":  {"prevalence": 0.45, "pgs_effect": 0.3},
    "obesity-like":       {"prevalence": 0.42, "pgs_effect": 0.5},
}


def synth_cohort(rng: np.random.Generator, n: int, num_pcs: int,
                 pgs_effect: float, prevalence: float) -> pd.DataFrame:
    """Synthesize one cohort with PGS that depends on PCs+sex (stratification)
    and a probit-scale liability that mixes PCs, sex, and the true PGS effect."""
    pcs = rng.standard_normal((n, num_pcs))
    sex = rng.integers(0, 2, size=n)

    pgs_loadings = rng.standard_normal(num_pcs) * 0.3
    pgs_raw = pcs @ pgs_loadings + 0.15 * sex + rng.standard_normal(n)
    pgs = (pgs_raw - pgs_raw.mean()) / pgs_raw.std()

    pc_effects = rng.standard_normal(num_pcs) * 0.2
    liability = (
        pcs @ pc_effects
        + 0.1 * sex
        + pgs_effect * pgs
        + rng.standard_normal(n)
    )
    threshold = float(np.quantile(liability, 1.0 - prevalence))
    case = (liability > threshold).astype(int)

    return pd.DataFrame({
        "case": case,
        "sex": sex,
        "pgs": pgs,
        **{f"PC{i+1}": pcs[:, i] for i in range(num_pcs)},
    })


def fit_marginal_slope(df: pd.DataFrame, num_pcs: int) -> gamfit.Model:
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    duchon = f"duchon({pcs}, centers={DUCHON_CENTERS}, order=1, power=2, length_scale=1.0)"
    df["prs_z"] = (df["pgs"] - df["pgs"].mean()) / df["pgs"].std(ddof=0)
    cols = ["case", "sex", "prs_z"] + [f"PC{i+1}" for i in range(num_pcs)]
    return gamfit.fit(
        df[cols],
        f"case ~ {duchon} + sex",
        link="probit",
        z_column="prs_z",
        logslope_formula=duchon,
        scale_dimensions=True,
    )


def metrics(y: np.ndarray, p: np.ndarray, K: float) -> dict[str, float]:
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll_full = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    P = float(y.mean())
    ll_null = float(np.sum(y * np.log(P) + (1 - y) * np.log1p(-P)))
    n = len(y)
    cox_snell = 1.0 - np.exp(2 * (ll_null - ll_full) / n)
    nagelkerke = cox_snell / (1.0 - np.exp(2 * ll_null / n))
    z = float(norm.pdf(norm.ppf(K)))
    liability_r2 = nagelkerke * K * (1 - K) / (z ** 2 * P * (1 - P))
    return {
        "n": n,
        "cases": int(y.sum()),
        "P": P,
        "auroc": float(roc_auc_score(y, p)),
        "nagelkerke_r2": float(nagelkerke),
        "liability_r2": float(liability_r2),
    }


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    for name, cfg in SCENARIOS.items():
        print(f"\n=== {name.upper()} (synthetic) ===")
        cohort = synth_cohort(rng, N_TOTAL, NUM_PCS, cfg["pgs_effect"], cfg["prevalence"])
        case_idx = cohort.index[cohort["case"] == 1].to_numpy()
        ctrl_idx = cohort.index[cohort["case"] == 0].to_numpy()
        print(f"  synth: n={N_TOTAL:,}  cases={len(case_idx):,}  controls={len(ctrl_idx):,}")

        pick = np.concatenate([
            rng.choice(case_idx, N_CASES, replace=False),
            rng.choice(ctrl_idx, N_CONTROLS, replace=False),
        ])
        df = cohort.loc[pick].reset_index(drop=True)
        model = fit_marginal_slope(df, NUM_PCS)
        p_hat = np.asarray(model.predict(df.drop(columns=["case"])), dtype=float)
        m = metrics(df["case"].to_numpy(), p_hat, cfg["prevalence"])

        print(
            f"  PGS effect (true)={cfg['pgs_effect']:.2f}  "
            f"n={m['n']}  cases={m['cases']}  P={m['P']:.4f}  K={cfg['prevalence']:.3f}"
        )
        print(
            f"  AUROC={m['auroc']:.4f}  Nagelkerke R^2={m['nagelkerke_r2']:.4f}  "
            f"liability R^2={m['liability_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
