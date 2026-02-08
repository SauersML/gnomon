#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import ttest_1samp, ttest_rel
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Config:
    n_ind: int
    seq_len: int
    recomb: float
    mut: float
    ne: int
    n_causal: int
    n_pca_sites: int
    prevalence: float
    h2: float


def std(v: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(v.reshape(-1, 1)).ravel()


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def _p_one_sided_from_ttest(p_two_sided: float, t_stat: float, direction: str) -> float:
    if np.isnan(p_two_sided) or np.isnan(t_stat):
        return np.nan
    if direction == "greater":
        return float(p_two_sided / 2.0) if t_stat > 0 else float(1.0 - p_two_sided / 2.0)
    if direction == "less":
        return float(p_two_sided / 2.0) if t_stat < 0 else float(1.0 - p_two_sided / 2.0)
    raise ValueError(direction)


def mean_ci(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if len(x) < 2:
        return m, m, m
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    z = 1.96
    return m, m - z * se, m + z * se


def simulate_seed(cfg: Config, seed: int) -> Dict[str, np.ndarray]:
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(cfg.n_ind, ploidy=2)],
        sequence_length=cfg.seq_len,
        recombination_rate=cfg.recomb,
        ploidy=2,
        population_size=cfg.ne,
        random_seed=seed,
        model="hudson",
    )
    ts = msprime.sim_mutations(ts, rate=cfg.mut, random_seed=seed + 1)

    m = ts.num_sites
    if m < 300:
        raise RuntimeError(f"Too few sites: {m}")

    X = np.empty((cfg.n_ind, m), dtype=np.int8)
    maf = np.empty(m, dtype=np.float64)

    for j, var in enumerate(ts.variants()):
        gh = var.genotypes.astype(np.int16)
        gd = gh[0::2] + gh[1::2]
        X[:, j] = gd.astype(np.int8)
        p = gd.mean() / 2.0
        maf[j] = min(p, 1.0 - p)

    rng = np.random.default_rng(seed + 7)
    elig = np.where(maf >= 0.01)[0]
    if len(elig) < 100:
        raise RuntimeError("Too few eligible variants")

    cidx = np.sort(rng.choice(elig, size=min(cfg.n_causal, len(elig)), replace=False))
    pidx = np.sort(rng.choice(elig, size=min(cfg.n_pca_sites, len(elig)), replace=False))

    betas = rng.normal(0.0, math.sqrt(cfg.h2 / len(cidx)), size=len(cidx))
    g = std(X[:, cidx].astype(float) @ betas)

    b0 = brentq(lambda b: expit(b + g).mean() - cfg.prevalence, -20, 20)
    y = rng.binomial(1, expit(b0 + g)).astype(np.int8)

    idx = np.arange(cfg.n_ind)
    train, test = train_test_split(idx, test_size=0.5, random_state=42)

    # Intentionally weak PRS to surface PC-tagging artifact.
    prs_weak = std(g[test] + rng.normal(0.0, 5.5, size=len(test)))

    # Strong oracle-like PRS control.
    prs_strong = std(g[test] + rng.normal(0.0, 0.25, size=len(test)))

    # PCA: fit on train only, project test.
    Xp_train = X[np.ix_(train, pidx)].astype(float)
    Xp_test = X[np.ix_(test, pidx)].astype(float)
    mu_p = Xp_train.mean(axis=0)
    sd_p = Xp_train.std(axis=0)
    sd_p[sd_p == 0] = 1.0
    Xp_train = np.nan_to_num((Xp_train - mu_p) / sd_p)
    Xp_test = np.nan_to_num((Xp_test - mu_p) / sd_p)

    pca = PCA(n_components=min(20, Xp_train.shape[0] - 1, Xp_train.shape[1]), random_state=seed)
    pca.fit(Xp_train)
    pcs_test = pca.transform(Xp_test)
    pcs_test = StandardScaler().fit_transform(pcs_test)[:, :5]

    return {
        "y_test": y[test],
        "g_test": g[test],
        "prs_weak": prs_weak,
        "prs_strong": prs_strong,
        "pcs_test": pcs_test,
    }


def build_features(prs: np.ndarray, pcs: np.ndarray, method: str) -> np.ndarray:
    p = prs.reshape(-1, 1)
    if method == "Raw":
        return p
    if method == "Additive":
        return np.hstack([p, pcs])
    if method == "Linear":
        return np.hstack([p, pcs, p * pcs])
    raise ValueError(method)


def calibrated_auc_with_optional_orth(prs: np.ndarray, pcs: np.ndarray, g: np.ndarray, y: np.ndarray, orth: bool) -> Dict[str, float]:
    idx = np.arange(len(y))
    cal, val = train_test_split(idx, test_size=0.5, random_state=42)

    pcs_cal = pcs[cal].copy()
    pcs_val = pcs[val].copy()

    if orth:
        for k in range(pcs.shape[1]):
            reg = LinearRegression().fit(g[cal].reshape(-1, 1), pcs_cal[:, k])
            pcs_cal[:, k] = pcs_cal[:, k] - reg.predict(g[cal].reshape(-1, 1))
            pcs_val[:, k] = pcs_val[:, k] - reg.predict(g[val].reshape(-1, 1))

    scaler = StandardScaler().fit(pcs_cal)
    pcs_cal = scaler.transform(pcs_cal)
    pcs_val = scaler.transform(pcs_val)

    out: Dict[str, float] = {
        "r2_pc_g_cal": float(LinearRegression().fit(pcs_cal, g[cal]).score(pcs_cal, g[cal])),
        "r2_pc_g_val": float(LinearRegression().fit(pcs_val, g[val]).score(pcs_val, g[val])),
    }
    for method in ["Raw", "Additive", "Linear"]:
        X_cal = build_features(prs[cal], pcs_cal, method)
        X_val = build_features(prs[val], pcs_val, method)
        clf = LogisticRegression(max_iter=4000)
        clf.fit(X_cal, y[cal])
        p = clf.predict_proba(X_val)[:, 1]
        out[f"auc_{method.lower()}"] = safe_auc(y[val], p)
    return out


def run(cfg: Config, seeds: List[int], out_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for seed in seeds:
        sim = simulate_seed(cfg, seed)
        y = sim["y_test"]
        g = sim["g_test"]
        pcs = sim["pcs_test"]

        weak_orig = calibrated_auc_with_optional_orth(sim["prs_weak"], pcs, g, y, orth=False)
        weak_orth = calibrated_auc_with_optional_orth(sim["prs_weak"], pcs, g, y, orth=True)
        strong_orig = calibrated_auc_with_optional_orth(sim["prs_strong"], pcs, g, y, orth=False)

        rows.append({"seed": seed, "condition": "weak_original", **weak_orig})
        rows.append({"seed": seed, "condition": "weak_orthogonalized", **weak_orth})
        rows.append({"seed": seed, "condition": "strong_original", **strong_orig})

    df = pd.DataFrame(rows)
    df["gap_additive_minus_raw"] = df["auc_additive"] - df["auc_raw"]
    df["gap_linear_minus_raw"] = df["auc_linear"] - df["auc_raw"]

    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "gen0_orthogonalization_results.csv", index=False)
    return df


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    plt.rcParams.update({
        "figure.dpi": 180,
        "axes.facecolor": "#fafafa",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d0d0d0",
        "grid.alpha": 0.5,
        "axes.titleweight": "bold",
    })

    order = ["weak_original", "weak_orthogonalized", "strong_original"]
    labels = {
        "weak_original": "Weak PRS + Original PCs",
        "weak_orthogonalized": "Weak PRS + Orthogonalized PCs",
        "strong_original": "Strong PRS + Original PCs",
    }
    colors = {
        "weak_original": "#1f77b4",
        "weak_orthogonalized": "#ff7f0e",
        "strong_original": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    for i, cond in enumerate(order):
        sub = df[df["condition"] == cond]
        x = np.full(len(sub), i, dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(sub)) if len(sub) > 1 else np.array([0.0])
        ax.scatter(x + jitter, sub["gap_additive_minus_raw"], s=36, alpha=0.8, color=colors[cond], edgecolor="white", linewidth=0.4)
        m, lo, hi = mean_ci(sub["gap_additive_minus_raw"].values)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", color="black", capsize=4, linewidth=1.3)
    ax.axhline(0, color="gray", ls=":")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([labels[c] for c in order], rotation=15, ha="right")
    ax.set_ylabel("AUC gain (Additive - Raw)")
    ax.set_title("Gen0 artifact: additive PC gain")
    ax.grid(True)

    ax = axes[1]
    for i, cond in enumerate(order):
        sub = df[df["condition"] == cond]
        x = np.full(len(sub), i, dtype=float)
        jitter = np.linspace(-0.08, 0.08, len(sub)) if len(sub) > 1 else np.array([0.0])
        ax.scatter(x + jitter, sub["r2_pc_g_val"], s=36, alpha=0.8, color=colors[cond], edgecolor="white", linewidth=0.4)
        m, lo, hi = mean_ci(sub["r2_pc_g_val"].values)
        ax.errorbar([i], [m], yerr=[[m - lo], [hi - m]], fmt="o", color="black", capsize=4, linewidth=1.3)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([labels[c] for c in order], rotation=15, ha="right")
    ax.set_ylabel("R²(PCs, G_true) on validation")
    ax.set_title("Orthogonalization removes PC-G coupling")
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_gen0_orthogonalization.png", dpi=220)
    plt.close(fig)


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = df.groupby("condition").agg(
        n=("seed", "count"),
        auc_raw_mean=("auc_raw", "mean"),
        auc_additive_mean=("auc_additive", "mean"),
        auc_linear_mean=("auc_linear", "mean"),
        gap_additive_mean=("gap_additive_minus_raw", "mean"),
        gap_additive_std=("gap_additive_minus_raw", "std"),
        gap_linear_mean=("gap_linear_minus_raw", "mean"),
        r2_pc_g_val_mean=("r2_pc_g_val", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "gen0_orthogonalization_summary.csv", index=False)

    weak_orig = df[df["condition"] == "weak_original"].sort_values("seed")
    weak_orth = df[df["condition"] == "weak_orthogonalized"].sort_values("seed")
    strong_orig = df[df["condition"] == "strong_original"].sort_values("seed")

    g1 = weak_orig["gap_additive_minus_raw"].values
    g2 = weak_orth["gap_additive_minus_raw"].values
    g3 = strong_orig["gap_additive_minus_raw"].values

    t1 = ttest_1samp(g1, popmean=0.0, nan_policy="omit")
    p1 = _p_one_sided_from_ttest(float(t1.pvalue), float(t1.statistic), "greater")

    t2 = ttest_rel(g1, g2, nan_policy="omit")
    p2 = _p_one_sided_from_ttest(float(t2.pvalue), float(t2.statistic), "greater")

    t3 = ttest_1samp(g3, popmean=0.0, nan_policy="omit")

    m1, l1, u1 = mean_ci(g1)
    m2, l2, u2 = mean_ci(g2)
    m3, l3, u3 = mean_ci(g3)

    make_plots(df, out_dir)

    tests = pd.DataFrame([
        {
            "test_id": "H1",
            "contrast": "weak_original_gap_additive_minus_raw_gt_0",
            "estimate_mean": m1,
            "estimate_ci_lo": l1,
            "estimate_ci_hi": u1,
            "t_stat": float(t1.statistic),
            "p_value": p1,
            "p_value_type": "one-sided",
        },
        {
            "test_id": "H2",
            "contrast": "weak_original_gap_gt_weak_orthogonalized_gap",
            "estimate_mean": float(np.mean(g1 - g2)),
            "estimate_ci_lo": np.nan,
            "estimate_ci_hi": np.nan,
            "t_stat": float(t2.statistic),
            "p_value": p2,
            "p_value_type": "one-sided",
        },
        {
            "test_id": "H3",
            "contrast": "strong_original_gap_additive_minus_raw_ne_0",
            "estimate_mean": m3,
            "estimate_ci_lo": l3,
            "estimate_ci_hi": u3,
            "t_stat": float(t3.statistic),
            "p_value": float(t3.pvalue),
            "p_value_type": "two-sided",
        },
    ])
    tests.to_csv(out_dir / "gen0_hypothesis_tests.csv", index=False)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Rigorous replication of Gen0 PC orthogonalization artifact")
    ap.add_argument("--seed-start", type=int, default=1)
    ap.add_argument("--seed-end", type=int, default=20)
    ap.add_argument("--n-ind", type=int, default=2200)
    ap.add_argument("--seq-len", type=int, default=5_000_000)
    ap.add_argument("--n-causal", type=int, default=5000)
    ap.add_argument("--n-pca-sites", type=int, default=2000)
    ap.add_argument("--out-dir", type=Path, default=Path("sims/results_gen0"))
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--input-csv", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir

    if args.summary_only:
        if args.input_csv is None:
            raise SystemExit("--summary-only requires --input-csv")
        df = pd.read_csv(args.input_csv)
        summarize(df, out_dir)
        print(f"Summary complete: {out_dir}")
        return

    seeds = list(range(args.seed_start, args.seed_end + 1))
    cfg = Config(
        n_ind=args.n_ind,
        seq_len=args.seq_len,
        recomb=1e-8,
        mut=1e-8,
        ne=10_000,
        n_causal=args.n_causal,
        n_pca_sites=args.n_pca_sites,
        prevalence=0.10,
        h2=0.50,
    )

    print(f"Running Gen0 orthogonalization: seeds={seeds[0]}..{seeds[-1]} n={len(seeds)}")
    df = run(cfg, seeds, out_dir)
    summarize(df, out_dir)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
