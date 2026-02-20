#!/usr/bin/env python3
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import run_mechanism_tests as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "model_decomposition.csv"


def _std(v):
    return StandardScaler().fit_transform(v.reshape(-1, 1)).ravel()


def _fit_auc(X, y, idx_cal, idx_val):
    m = LogisticRegression(max_iter=2000)
    m.fit(X[idx_cal], y[idx_cal])
    p = m.predict_proba(X[idx_val])[:, 1]
    return float(roc_auc_score(y[idx_val], p))


def run(seed: int, n_ind: int, seq_len: int, n_causal: int, n_pca: int):
    sim = base._simulate_genotypes(seed=70_000 + seed, recomb_rate=1e-8, n_ind=n_ind, sequence_length=seq_len, n_causal=n_causal)
    eligible_causal = np.where(sim.maf >= 0.01)[0]
    rng = np.random.default_rng(71_000 + seed)
    causal_idx = rng.choice(eligible_causal, size=min(n_causal, len(eligible_causal)), replace=False)

    pca_idx = base._select_pca_sites(sim, seed=72_000 + seed, n_pca=n_pca, mode="all", causal_idx=causal_idx)
    pcs = base._compute_pcs(sim, pca_idx)[:, :5]

    test = sim.test_idx
    y = sim.y[test]
    g = sim.g_true[test]
    pcs_t = pcs[test]
    weak = base._weak_gwas_prs(sim)
    strong = base._make_synthetic_prs(g, sigma=0.25, seed=73_000 + seed)
    noise = _std(np.random.default_rng(74_000 + seed).normal(size=len(g)))

    idx = np.arange(len(y))
    idx_cal, idx_val = train_test_split(idx, test_size=0.5, random_state=42)

    out = []
    for prs_name, prs in [("noise", noise), ("weak_gwas", weak), ("strong_oracle", strong)]:
        X_raw = prs.reshape(-1, 1)
        X_pc = pcs_t
        X_add = np.hstack([prs.reshape(-1, 1), pcs_t])
        X_lin = np.hstack([prs.reshape(-1, 1), pcs_t, prs.reshape(-1, 1) * pcs_t])

        out.append({
            "seed": seed,
            "prs_type": prs_name,
            "corr_prs_g": float(np.corrcoef(prs, g)[0, 1]),
            "r2_pc_g": float(LinearRegression().fit(pcs_t, g).score(pcs_t, g)),
            "auc_raw": _fit_auc(X_raw, y, idx_cal, idx_val),
            "auc_pc_only": _fit_auc(X_pc, y, idx_cal, idx_val),
            "auc_additive": _fit_auc(X_add, y, idx_cal, idx_val),
            "auc_interaction": _fit_auc(X_lin, y, idx_cal, idx_val),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-ind", type=int, default=400)
    ap.add_argument("--seq-len", type=int, default=800000)
    ap.add_argument("--n-causal", type=int, default=300)
    ap.add_argument("--n-pca", type=int, default=200)
    args = ap.parse_args()

    rows = []
    for s in range(1, args.seeds + 1):
        rows.extend(run(s, args.n_ind, args.seq_len, args.n_causal, args.n_pca))

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(df.groupby("prs_type")[["corr_prs_g", "r2_pc_g", "auc_raw", "auc_pc_only", "auc_additive", "auc_interaction"]].mean())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
