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
"""Local sanity check for the biobank Bernoulli marginal-slope fit — no
AoU, no gnomon, no BigQuery.

For each scenario the script:
  1. synthesizes a cohort (PGS depends on PCs + sex; liability is a probit
     mix of PCs, sex, and a configurable true PGS effect),
  2. takes 100 cases + 100 controls for *training* (no overlap with test),
  3. takes a disjoint 100 cases + 100 controls for *held-out test*,
  4. standardizes the PGS using training stats only (no leakage),
  5. fits `case ~ duchon(PC1..PC10) + sex` with `prs_z` driving the
     marginal-slope log-slope channel through the same Duchon smooth,
  6. reports held-out AUROC, Nagelkerke R^2, and Lee-2011 liability R^2.

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
N_TOTAL = 20_000
N_TRAIN_CASES = 200
N_TRAIN_CONTROLS = 200
N_TEST_CASES = 200
N_TEST_CONTROLS = 200
RNG_SEED = 0

SCENARIOS = {
    # pgs_effect is the true liability-scale slope per SD of standardized PGS.
    # Under a probit liability model OR/SD ≈ exp(1.65 * pgs_effect), so:
    #   copd-like         pgs_effect=0.30  -> OR/SD ≈ 1.65
    #   hypertension-like pgs_effect=0.20  -> OR/SD ≈ 1.39
    "copd-like":          {"prevalence": 0.06, "pgs_effect": 0.30},
    "hypertension-like":  {"prevalence": 0.45, "pgs_effect": 0.20},
}

# Rough All-of-Us-like ancestry composition for the four "EUR/AFR/AMR/EAS" buckets.
ANCESTRY_FRACTIONS = np.array([0.70, 0.15, 0.10, 0.05])


def synth_cohort(rng: np.random.Generator, n: int, num_pcs: int,
                 pgs_effect: float, prevalence: float) -> pd.DataFrame:
    """Multi-cluster ancestry mixture with realistic PC + PGS structure.

    Top PCs separate the four ancestry clusters; deeper PCs decay to nearly
    pure within-cluster noise. PGS has cluster-dependent mean (population
    stratification) plus a per-person noise term. Liability is a probit-scale
    mix of cluster baseline risk, PC effects that decay with PC index, sex,
    the true PGS effect, and N(0,1) residual.
    """
    n_clusters = len(ANCESTRY_FRACTIONS)
    cluster = rng.choice(n_clusters, size=n, p=ANCESTRY_FRACTIONS)

    # Cluster centroids in PC space: PC1/PC2 strongly separate, PC3/PC4
    # weakly separate, PC5+ essentially no ancestry signal.
    centroids = np.zeros((n_clusters, num_pcs))
    centroids[:, 0] = [1.5, -2.5, 0.8, -0.5]
    if num_pcs >= 2:
        centroids[:, 1] = [0.0, 1.2, -1.8, 0.3]
    if num_pcs >= 3:
        centroids[:, 2] = [0.3, -0.4, 0.1, 0.2]
    if num_pcs >= 4:
        centroids[:, 3] = [-0.1, 0.2, 0.3, -0.2]
    if num_pcs >= 5:
        centroids[:, 4:] = rng.standard_normal((n_clusters, num_pcs - 4)) * 0.05

    # Within-cluster spread shrinks as PC index grows.
    pc_sd = 1.0 / np.sqrt(np.arange(1, num_pcs + 1))
    pc_sd /= pc_sd.mean()
    pcs = centroids[cluster] + rng.standard_normal((n, num_pcs)) * pc_sd

    sex = rng.integers(0, 2, size=n)

    # PGS: cluster-dependent mean (stratification) + sex tilt + noise, then
    # standardized so prs_z is genuinely centered on the cohort.
    cluster_pgs_mean = np.array([0.30, -0.40, 0.10, -0.20])
    pgs_raw = cluster_pgs_mean[cluster] + 0.10 * sex + rng.standard_normal(n)
    pgs = (pgs_raw - pgs_raw.mean()) / pgs_raw.std()

    # Liability: cluster baseline + PC effects with decay + sex + PGS + noise.
    cluster_liab_offset = np.array([0.0, 0.15, -0.05, 0.25])
    pc_effect_scale = 1.0 / np.sqrt(np.arange(1, num_pcs + 1))
    pc_effects = rng.standard_normal(num_pcs) * 0.20 * pc_effect_scale
    liability = (
        cluster_liab_offset[cluster]
        + pcs @ pc_effects
        + 0.10 * sex
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


def fit_marginal_slope(train_df: pd.DataFrame, num_pcs: int) -> gamfit.Model:
    """Bernoulli marginal-slope probit GAM with joint Duchon over PCs in
    both the location and log-slope channels; sex linear; prs_z is the
    latent score (z_column) so its slope varies in PC space.

    No linkwiggle(...) in either formula -> engine's score-warp / link-dev
    deviation blocks remain inactive in the protocol that this triggers.
    """
    pcs = ", ".join(f"PC{i+1}" for i in range(num_pcs))
    duchon = f"duchon({pcs}, centers={DUCHON_CENTERS}, order=1, power=2, length_scale=1.0)"
    cols = ["case", "sex", "prs_z"] + [f"PC{i+1}" for i in range(num_pcs)]
    return gamfit.fit(
        train_df[cols],
        f"case ~ {duchon} + sex",
        link="probit",
        z_column="prs_z",
        logslope_formula=duchon,
    )


def metrics(y: np.ndarray, p: np.ndarray, K: float) -> dict[str, float]:
    """Held-out AUROC + Nagelkerke + Lee-2011 liability-scale R^2.

    Lee, Wray, Goddard, Visscher (AJHG 2011, eq. 23) for ascertained case-control:

        R^2_l = R^2_O * K^2 * (1-K)^2 / (z^2 * P * (1-P))
    """
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll_full = float(np.sum(y * np.log(p) + (1 - y) * np.log1p(-p)))
    P = float(y.mean())
    ll_null = float(np.sum(y * np.log(P) + (1 - y) * np.log1p(-P)))
    n = len(y)
    cox_snell = 1.0 - np.exp(2 * (ll_null - ll_full) / n)
    nagelkerke = cox_snell / (1.0 - np.exp(2 * ll_null / n))
    z = float(norm.pdf(norm.ppf(K)))
    liability_r2 = nagelkerke * K**2 * (1 - K) ** 2 / (z**2 * P * (1 - P))
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
        case_idx = rng.permutation(cohort.index[cohort["case"] == 1].to_numpy())
        ctrl_idx = rng.permutation(cohort.index[cohort["case"] == 0].to_numpy())
        print(f"  synth: n={N_TOTAL:,}  cases={len(case_idx):,}  controls={len(ctrl_idx):,}")

        n_te_case = min(N_TEST_CASES, max(0, len(case_idx) - N_TRAIN_CASES))
        n_te_ctrl = min(N_TEST_CONTROLS, max(0, len(ctrl_idx) - N_TRAIN_CONTROLS))
        train_pick = np.concatenate([case_idx[:N_TRAIN_CASES], ctrl_idx[:N_TRAIN_CONTROLS]])
        test_pick = np.concatenate([
            case_idx[N_TRAIN_CASES : N_TRAIN_CASES + n_te_case],
            ctrl_idx[N_TRAIN_CONTROLS : N_TRAIN_CONTROLS + n_te_ctrl],
        ])
        train = cohort.loc[train_pick].reset_index(drop=True)
        test = cohort.loc[test_pick].reset_index(drop=True)

        pgs_mean = float(train["pgs"].mean())
        pgs_std = float(train["pgs"].std(ddof=0))
        train["prs_z"] = (train["pgs"] - pgs_mean) / pgs_std
        test["prs_z"] = (test["pgs"] - pgs_mean) / pgs_std

        model = fit_marginal_slope(train, NUM_PCS)
        predict_cols = ["sex", "prs_z"] + [f"PC{i+1}" for i in range(NUM_PCS)]
        pred = model.predict(test[predict_cols], return_type="dict")
        p_test = np.asarray(pred["mean"], dtype=float)
        m = metrics(test["case"].to_numpy(), p_test, cfg["prevalence"])

        print(
            f"  PGS effect (true)={cfg['pgs_effect']:.2f}  "
            f"train_n={len(train)}  test_n={m['n']}  test_cases={m['cases']}  "
            f"P={m['P']:.4f}  K={cfg['prevalence']:.3f}"
        )
        print(
            f"  held-out  AUROC={m['auroc']:.4f}  "
            f"Nagelkerke R^2={m['nagelkerke_r2']:.4f}  liability R^2={m['liability_r2']:.4f}"
        )


if __name__ == "__main__":
    main()
