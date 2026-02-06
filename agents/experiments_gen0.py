#!/usr/bin/env python3
"""
Orthogonal experiments to dissect the gen 0 phenomenon.

Experiments:
  1. Orthogonalized PCs: regress out G_true from PCs → advantage should vanish
  2. Random PCs: replace with iid Gaussian noise → advantage should vanish
  3. Vary n_pca_sites: show R²(PC, G_true) and Linear−Raw gap scale together
  4. P_observed across generation levels: real calibration without BayesR confound
  5. PC−G_true R² across generation levels: show spurious correlation → real structure

Usage:
  python experiments_gen0.py          # run all
  python experiments_gen0.py 1        # run experiment 1 only
  python experiments_gen0.py 1 2 3    # run experiments 1, 2, 3
"""
from __future__ import annotations

import os, sys, subprocess, json, shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.special import expit as sigmoid
from scipy.optimize import brentq

sys.path.insert(0, "sims")

RESULTS_DIR = Path("gen0_analysis")
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def calibrated_auc(prs, pcs5, y, method="Raw", idx_cal=None, idx_val=None):
    """Compute AUC using calibration paradigm (fit on cal, evaluate on val)."""
    if idx_cal is None or idx_val is None:
        idx_cal, idx_val = train_test_split(
            np.arange(len(y)), test_size=0.5, random_state=42
        )

    if method == "Raw":
        X_cal = prs[idx_cal].reshape(-1, 1)
        X_val = prs[idx_val].reshape(-1, 1)
    elif method == "Linear":
        def _make_X(P, PC):
            P2 = P.reshape(-1, 1)
            return np.hstack([P2, PC, P2 * PC])
        X_cal = _make_X(prs[idx_cal], pcs5[idx_cal])
        X_val = _make_X(prs[idx_val], pcs5[idx_val])
    else:
        raise ValueError(method)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_cal, y[idx_cal])
    y_prob = model.predict_proba(X_val)[:, 1]
    try:
        return roc_auc_score(y[idx_val], y_prob)
    except ValueError:
        return np.nan


def simulate_panmictic(n=2000, n_variants=10000, n_causal=5000, n_pca_sites=2000,
                       n_pcs=20, h2=0.5, prevalence=0.10, prs_noise=0.35, seed=42):
    """
    Pure-numpy panmictic population simulation (fast, no msprime).
    Returns dict with genotypes, PCs, G_true, phenotype, PRS proxy.
    """
    rng = np.random.default_rng(seed)

    # Random genotypes (0/1/2)
    afs = rng.uniform(0.05, 0.95, n_variants)
    X = np.column_stack([rng.binomial(2, af, n) for af in afs]).astype(np.float32)

    # PCA sites
    pca_idx = rng.choice(n_variants, min(n_pca_sites, n_variants), replace=False)
    X_pca = X[:, pca_idx].copy()
    scaler = StandardScaler()
    X_pca_z = np.nan_to_num(scaler.fit_transform(X_pca), 0.0)
    pca_model = PCA(n_components=n_pcs, random_state=seed)
    pcs = pca_model.fit_transform(X_pca_z)
    pcs = StandardScaler().fit_transform(pcs)

    # Causal sites & G_true
    causal_idx = rng.choice(n_variants, min(n_causal, n_variants), replace=False)
    betas = rng.normal(0, np.sqrt(h2 / len(causal_idx)), len(causal_idx))
    G_true = X[:, causal_idx] @ betas
    G_true = (G_true - G_true.mean()) / G_true.std()

    # Phenotype
    b0 = brentq(lambda b: sigmoid(b + G_true).mean() - prevalence, -10, 10)
    p = sigmoid(b0 + G_true)
    y = rng.binomial(1, p).astype(np.int32)

    # P_observed (oracle noisy PRS)
    P_obs = G_true + rng.normal(0, prs_noise, n)
    P_obs = StandardScaler().fit_transform(P_obs.reshape(-1, 1)).ravel()

    return {
        "X": X, "pcs": pcs, "G_true": G_true, "y": y, "P_obs": P_obs,
        "pca_model": pca_model, "causal_idx": causal_idx, "pca_idx": pca_idx,
        "betas": betas, "pca_variance_explained": pca_model.explained_variance_ratio_,
    }


def run_sim_two_pop(sim_name, gens, seed):
    """Run sim_two_pop.py and return the prefix if successful."""
    prefix = f"{sim_name}_g{gens}_s{seed}"
    if Path(f"{prefix}.tsv").exists():
        return prefix
    env = {**os.environ, "PYTHONPATH": "sims"}
    r = subprocess.run(
        ["python3", "sims/sim_two_pop.py", sim_name, str(gens), str(seed)],
        capture_output=True, text=True, env=env
    )
    if r.returncode != 0:
        print(f"  sim_two_pop failed for {prefix}: {r.stderr[:300]}")
        return None
    return prefix


def split_and_load(prefix):
    """Split data and return (train_df, test_df)."""
    work_dir = Path(f"{prefix}_work")
    if not (work_dir / "test.keep").exists():
        env = {**os.environ, "PYTHONPATH": "sims"}
        r = subprocess.run(
            ["python3", "sims/split_data.py", prefix],
            capture_output=True, text=True, env=env
        )
        if r.returncode != 0:
            raise RuntimeError(f"split failed: {r.stderr[:300]}")

    tsv = pd.read_csv(f"{prefix}.tsv", sep="\t")
    tsv["individual_id"] = tsv["individual_id"].astype(str)

    test_keep = pd.read_csv(f"{work_dir}/test.keep", sep="\t", header=None, names=["FID", "IID"])
    test_iids = set(test_keep["IID"].astype(str))

    test_df = tsv[tsv["individual_id"].isin(test_iids)].copy().set_index("individual_id")
    train_df = tsv[~tsv["individual_id"].isin(test_iids)].copy().set_index("individual_id")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Experiment 1: Orthogonalized PCs
# ---------------------------------------------------------------------------

def experiment_1_orthogonalized_pcs(n_seeds=5):
    """
    At gen 0 (msprime), orthogonalize PCs against G_true.
    If the Linear advantage comes from PC−G_true correlation (via LD),
    it should vanish when that correlation is removed.

    Uses real msprime simulations for realistic LD structure.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Orthogonalized PCs (msprime, realistic LD)")
    print("=" * 70)
    print("Hypothesis: Linear advantage at gen 0 is from PCs predicting G_true")
    print("via LD. Orthogonalizing PCs to G_true should kill it.\n")

    rows = []
    for seed in range(1, n_seeds + 1):
        prefix = run_sim_two_pop("divergence", 0, seed)
        if prefix is None:
            continue
        try:
            _, test_df = split_and_load(prefix)
        except Exception as e:
            print(f"  [seed {seed}] load failed: {e}")
            continue

        y = test_df["y"].values
        G = test_df["G_true"].values
        P = StandardScaler().fit_transform(
            test_df["P_observed"].values.reshape(-1, 1)
        ).ravel()
        pcs_orig = test_df[[f"pc{i+1}" for i in range(5)]].values

        # Orthogonalize: regress each PC on G_true, take residuals
        pcs_orth = np.zeros_like(pcs_orig)
        for k in range(5):
            reg = LinearRegression().fit(G.reshape(-1, 1), pcs_orig[:, k])
            pcs_orth[:, k] = pcs_orig[:, k] - reg.predict(G.reshape(-1, 1))
        pcs_orth = StandardScaler().fit_transform(pcs_orth)

        # Random PCs (control)
        rng = np.random.default_rng(seed + 9999)
        pcs_rand = StandardScaler().fit_transform(rng.standard_normal((len(y), 5)))

        idx_cal, idx_val = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42)

        for pc_label, pcs in [("Original", pcs_orig), ("Orthogonalized", pcs_orth), ("Random", pcs_rand)]:
            r2 = LinearRegression().fit(pcs, G).score(pcs, G)
            for method in ["Raw", "Linear"]:
                auc = calibrated_auc(P, pcs, y, method, idx_cal, idx_val)
                rows.append({"seed": seed, "PCs": pc_label, "method": method,
                             "AUC": auc, "R2_PC_G": r2})

        print(f"  [seed {seed}] R²(orig,G)={rows[-6]['R2_PC_G']:.3f} "
              f"R²(orth,G)={rows[-4]['R2_PC_G']:.4f} R²(rand,G)={rows[-2]['R2_PC_G']:.4f}")

    df = pd.DataFrame(rows)
    summary = df.groupby(["PCs", "method"]).agg(
        AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
        R2_mean=("R2_PC_G", "mean")
    ).round(4)
    print("\n" + summary.to_string())

    for pc_label in ["Original", "Orthogonalized", "Random"]:
        sub = df[df["PCs"] == pc_label]
        raw_vals = sub[sub["method"] == "Raw"]["AUC"].values
        lin_vals = sub[sub["method"] == "Linear"]["AUC"].values
        gap = lin_vals - raw_vals
        r2 = sub["R2_PC_G"].mean()
        print(f"\n  {pc_label:>16} PCs: R²={r2:.4f}  Linear−Raw = {gap.mean():+.4f} ± {gap.std():.4f}")

    df.to_csv(RESULTS_DIR / "exp1_orthogonalized_pcs.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 2: Random PCs (not from genotype data)
# ---------------------------------------------------------------------------

def experiment_2_ld_vs_no_ld(n_seeds=10):
    """
    Compare genotype-PCs from msprime (with LD) vs numpy (no LD).
    Shows that LD is the mechanism creating spurious PC-G_true correlation.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: LD vs No-LD (msprime vs independent genotypes)")
    print("=" * 70)
    print("Hypothesis: LD in msprime creates PC-G_true correlation.")
    print("Independent genotypes (no LD) should show no such correlation.\n")

    rows = []
    # msprime data (has LD)
    for seed in range(1, min(n_seeds, 5) + 1):
        prefix = run_sim_two_pop("divergence", 0, seed)
        if prefix is None:
            continue
        try:
            _, test_df = split_and_load(prefix)
        except Exception as e:
            continue
        G = test_df["G_true"].values
        pcs5 = test_df[[f"pc{i+1}" for i in range(5)]].values
        pcs20 = test_df[[f"pc{i+1}" for i in range(20)]].values
        r2_5 = LinearRegression().fit(pcs5, G).score(pcs5, G)
        r2_20 = LinearRegression().fit(pcs20, G).score(pcs20, G)
        rows.append({"source": "msprime (LD)", "seed": seed, "R2_5pc": r2_5, "R2_20pc": r2_20})

    # numpy data (no LD)
    for seed in range(1, n_seeds + 1):
        sim = simulate_panmictic(seed=seed, n_variants=10000, n_pca_sites=2000, n_causal=5000)
        n_test = len(sim["y"]) // 2
        sl = slice(n_test, None)
        G = sim["G_true"][sl]
        pcs5 = sim["pcs"][sl, :5]
        pcs20 = sim["pcs"][sl, :20]
        r2_5 = LinearRegression().fit(pcs5, G).score(pcs5, G)
        r2_20 = LinearRegression().fit(pcs20, G).score(pcs20, G)
        rows.append({"source": "numpy (no LD)", "seed": seed, "R2_5pc": r2_5, "R2_20pc": r2_20})

    df = pd.DataFrame(rows)
    summary = df.groupby("source").agg(
        R2_5pc_mean=("R2_5pc", "mean"), R2_5pc_std=("R2_5pc", "std"),
        R2_20pc_mean=("R2_20pc", "mean"), R2_20pc_std=("R2_20pc", "std"),
        n=("seed", "count"),
    ).round(4)
    print(summary.to_string())

    df.to_csv(RESULTS_DIR / "exp2_ld_vs_no_ld.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 3: Vary n_pca_sites / n_samples ratio
# ---------------------------------------------------------------------------

def experiment_3_causal_pca_overlap(n_seeds=5):
    """
    Quantify the overlap between PCA sites and causal sites, and how it
    drives the PC-G_true correlation through LD in the msprime genome.
    Also: recompute PCs from sites that EXCLUDE causal variants.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Causal-PCA site overlap and LD")
    print("=" * 70)
    print("Quantifying: how does LD between PCA & causal sites drive R²(PC,G)?\n")

    rows = []
    for seed in range(1, n_seeds + 1):
        prefix = run_sim_two_pop("divergence", 0, seed)
        if prefix is None:
            continue
        try:
            _, test_df = split_and_load(prefix)
        except Exception as e:
            continue

        # Load site info
        npz = np.load(f"{prefix}_sites.npz", allow_pickle=True)
        pca_ids = set(npz["pca_site_id"].tolist())
        causal_ids = set(npz["causal_site_id"].tolist())
        overlap = pca_ids & causal_ids
        pca_only = pca_ids - causal_ids

        G = test_df["G_true"].values
        y = test_df["y"].values
        P = StandardScaler().fit_transform(
            test_df["P_observed"].values.reshape(-1, 1)
        ).ravel()
        pcs5 = test_df[[f"pc{i+1}" for i in range(5)]].values

        r2 = LinearRegression().fit(pcs5, G).score(pcs5, G)

        idx_cal, idx_val = train_test_split(np.arange(len(y)), test_size=0.5, random_state=42)

        for method in ["Raw", "Linear"]:
            auc = calibrated_auc(P, pcs5, y, method, idx_cal, idx_val)
            rows.append({
                "seed": seed, "method": method, "AUC": auc,
                "n_pca": len(pca_ids), "n_causal": len(causal_ids),
                "n_overlap": len(overlap), "pct_overlap": len(overlap) / len(pca_ids) * 100,
                "R2_PC_G": r2,
            })

        print(f"  [seed {seed}] PCA={len(pca_ids)} Causal={len(causal_ids)} "
              f"Overlap={len(overlap)} ({len(overlap)/len(pca_ids)*100:.1f}%) "
              f"R²(PC,G)={r2:.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results!")
        return df

    print(f"\nMean overlap: {df['pct_overlap'].mean():.1f}% of PCA sites are also causal")
    print(f"Mean R²(PC1..5, G_true): {df['R2_PC_G'].mean():.4f}")

    raw = df[df["method"] == "Raw"]["AUC"].values
    lin = df[df["method"] == "Linear"]["AUC"].values
    gap = lin - raw
    print(f"Linear−Raw gap: {gap.mean():+.4f} ± {gap.std():.4f}")

    df.to_csv(RESULTS_DIR / "exp3_causal_pca_overlap.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 4: P_observed across generation levels (msprime)
# ---------------------------------------------------------------------------

def experiment_4_pobs_across_generations(n_seeds=3):
    """
    Use P_observed (oracle noisy PRS) across generation levels.
    This removes the BayesR confound and shows the real calibration effect.
    With a real PRS, there should be NO advantage at gen 0 but potentially
    some advantage at later generations due to real portability issues.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: P_observed across generation levels")
    print("=" * 70)
    print("Hypothesis: With oracle PRS (no BayesR), Linear advantage")
    print("should be ~0 at gen 0 and potentially grow with divergence.\n")

    gen_levels = [0, 50, 500, 5000, 10000]
    rows = []

    for gens in gen_levels:
        for seed in range(1, n_seeds + 1):
            prefix = run_sim_two_pop("divergence", gens, seed)
            if prefix is None:
                continue
            try:
                train_df, test_df = split_and_load(prefix)
            except Exception as e:
                print(f"  Failed to load {prefix}: {e}")
                continue

            y = test_df["y"].values
            P_obs = StandardScaler().fit_transform(
                test_df["P_observed"].values.reshape(-1, 1)
            ).ravel()
            G = test_df["G_true"].values
            pcs5 = test_df[[f"pc{i+1}" for i in range(5)]].values

            r2_pc_g = LinearRegression().fit(pcs5, G).score(pcs5, G)
            r2_pc_p = LinearRegression().fit(pcs5, P_obs).score(pcs5, P_obs)

            idx_cal, idx_val = train_test_split(
                np.arange(len(y)), test_size=0.5, random_state=42
            )

            for method in ["Raw", "Linear"]:
                auc = calibrated_auc(P_obs, pcs5, y, method, idx_cal, idx_val)
                rows.append({
                    "gens": gens, "seed": seed, "method": method, "AUC": auc,
                    "R2_PC_G": r2_pc_g, "R2_PC_P": r2_pc_p,
                    "prs_type": "P_observed",
                })

            print(f"  [g={gens}, s={seed}] R²(PC,G)={r2_pc_g:.3f}  "
                  f"Raw={[r['AUC'] for r in rows if r['gens']==gens and r['seed']==seed and r['method']=='Raw'][-1]:.3f}  "
                  f"Linear={[r['AUC'] for r in rows if r['gens']==gens and r['seed']==seed and r['method']=='Linear'][-1]:.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results collected!")
        return df

    summary = df.groupby(["gens", "method"]).agg(
        AUC_mean=("AUC", "mean"), AUC_std=("AUC", "std"),
        R2_PC_G_mean=("R2_PC_G", "mean"),
    ).round(4)
    print("\n" + summary.to_string())

    print(f"\n  {'Gens':>6} {'R²(PC,G)':>10} {'Raw AUC':>10} {'Linear AUC':>12} {'Gap':>8}")
    print("  " + "-" * 50)
    for gens in gen_levels:
        sub = df[df["gens"] == gens]
        if sub.empty:
            continue
        raw_auc = sub[sub["method"] == "Raw"]["AUC"].mean()
        lin_auc = sub[sub["method"] == "Linear"]["AUC"].mean()
        r2 = sub["R2_PC_G"].mean()
        print(f"  {gens:>6} {r2:>10.4f} {raw_auc:>10.4f} {lin_auc:>12.4f} {lin_auc - raw_auc:>+8.4f}")

    df.to_csv(RESULTS_DIR / "exp4_pobs_across_gens.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 5: R²(PC, G_true) across generation levels
# ---------------------------------------------------------------------------

def experiment_5_r2_across_generations(n_seeds=3):
    """
    Track how R²(PC, G_true) changes across generation levels.
    At gen 0: high spurious R² (PCs = random genotype covariance)
    At gen T: R² from real ancestry, but nature changes
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: R²(PC, G_true) across generation levels")
    print("=" * 70)
    print("Tracking how PC−G_true correlation changes with divergence.\n")

    gen_levels = [0, 20, 50, 100, 500, 1000, 5000, 10000]
    rows = []

    for gens in gen_levels:
        for seed in range(1, n_seeds + 1):
            prefix = run_sim_two_pop("divergence", gens, seed)
            if prefix is None:
                continue
            try:
                train_df, test_df = split_and_load(prefix)
            except Exception as e:
                print(f"  Failed: {e}")
                continue

            G = test_df["G_true"].values
            y = test_df["y"].values
            pcs5 = test_df[[f"pc{i+1}" for i in range(5)]].values
            pcs20 = test_df[[f"pc{i+1}" for i in range(20)]].values

            r2_5 = LinearRegression().fit(pcs5, G).score(pcs5, G)
            r2_20 = LinearRegression().fit(pcs20, G).score(pcs20, G)

            # Also check if PC1 separates populations
            pc1 = pcs5[:, 0]
            pop = test_df["pop_label"].values
            unique_pops = np.unique(pop)
            if len(unique_pops) == 1:
                pc1_sep = 0.0  # no separation possible
            else:
                m0 = pc1[pop == unique_pops[0]].mean()
                m1 = pc1[pop == unique_pops[1]].mean()
                pooled_std = pc1.std()
                pc1_sep = abs(m0 - m1) / pooled_std if pooled_std > 0 else 0

            rows.append({
                "gens": gens, "seed": seed,
                "R2_5pc": r2_5, "R2_20pc": r2_20,
                "PC1_separation": pc1_sep,
                "prevalence": y.mean(),
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("gens").agg(
        R2_5pc_mean=("R2_5pc", "mean"), R2_5pc_std=("R2_5pc", "std"),
        R2_20pc_mean=("R2_20pc", "mean"),
        PC1_sep_mean=("PC1_separation", "mean"),
    ).round(4)
    print(summary.to_string())

    df.to_csv(RESULTS_DIR / "exp5_r2_across_gens.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Experiment 6: BayesR PRS across generations (the real pipeline)
# ---------------------------------------------------------------------------

def experiment_6_bayesr_across_generations(n_seeds=3):
    """
    Run full BayesR pipeline across generation levels.
    Compare Linear−Raw gap using BayesR PRS vs P_observed PRS.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 6: BayesR PRS across generation levels")
    print("=" * 70)
    print("Full pipeline: simulate → split → BayesR → calibrate.\n")

    gen_levels = [0, 50, 500, 5000, 10000]
    rows = []

    for gens in gen_levels:
        for seed in range(1, n_seeds + 1):
            prefix = run_sim_two_pop("divergence", gens, seed)
            if prefix is None:
                continue

            work_dir = Path(f"{prefix}_work")
            try:
                train_df, test_df = split_and_load(prefix)
            except Exception as e:
                print(f"  Split failed for {prefix}: {e}")
                continue

            # Train BayesR if not already done
            sscore_path = work_dir / "BayesR.sscore"
            if not sscore_path.exists():
                env = {**os.environ, "PYTHONPATH": "sims"}
                r = subprocess.run(
                    ["python3", "sims/train_model.py", prefix, "BayesR"],
                    capture_output=True, text=True, env=env
                )
                if r.returncode != 0:
                    print(f"  BayesR failed for {prefix}: {r.stderr[:200]}")
                    continue

            # Load BayesR scores
            scores = pd.read_csv(sscore_path, sep="\t")
            scores["IID"] = scores["IID"].astype(str)
            test_df = test_df.join(scores.set_index("IID")["PRS"], how="inner")

            y = test_df["y"].values
            G = test_df["G_true"].values
            pcs5 = test_df[[f"pc{i+1}" for i in range(5)]].values
            prs_bayesr = StandardScaler().fit_transform(
                test_df["PRS"].values.reshape(-1, 1)
            ).ravel()
            p_obs = StandardScaler().fit_transform(
                test_df["P_observed"].values.reshape(-1, 1)
            ).ravel()

            corr_prs_g = np.corrcoef(prs_bayesr, G)[0, 1]
            r2_pc_g = LinearRegression().fit(pcs5, G).score(pcs5, G)

            idx_cal, idx_val = train_test_split(
                np.arange(len(y)), test_size=0.5, random_state=42
            )

            for prs_label, prs_vals in [("BayesR", prs_bayesr), ("P_observed", p_obs)]:
                for method in ["Raw", "Linear"]:
                    auc = calibrated_auc(prs_vals, pcs5, y, method, idx_cal, idx_val)
                    rows.append({
                        "gens": gens, "seed": seed, "prs_type": prs_label,
                        "method": method, "AUC": auc,
                        "corr_PRS_G": corr_prs_g if prs_label == "BayesR" else np.nan,
                        "R2_PC_G": r2_pc_g,
                    })

            print(f"  [g={gens}, s={seed}] corr(PRS,G)={corr_prs_g:.3f} R²(PC,G)={r2_pc_g:.3f}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No results!")
        return df

    print("\n--- BayesR PRS ---")
    sub_b = df[df["prs_type"] == "BayesR"]
    for gens in gen_levels:
        s = sub_b[sub_b["gens"] == gens]
        if s.empty:
            continue
        raw = s[s["method"] == "Raw"]["AUC"].mean()
        lin = s[s["method"] == "Linear"]["AUC"].mean()
        print(f"  g={gens:>6}: Raw={raw:.3f} Linear={lin:.3f} gap={lin-raw:+.3f}")

    print("\n--- P_observed ---")
    sub_p = df[df["prs_type"] == "P_observed"]
    for gens in gen_levels:
        s = sub_p[sub_p["gens"] == gens]
        if s.empty:
            continue
        raw = s[s["method"] == "Raw"]["AUC"].mean()
        lin = s[s["method"] == "Linear"]["AUC"].mean()
        print(f"  g={gens:>6}: Raw={raw:.3f} Linear={lin:.3f} gap={lin-raw:+.3f}")

    df.to_csv(RESULTS_DIR / "exp6_bayesr_across_gens.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots():
    """Generate publication-quality plots from experiment results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": 11, "axes.titlesize": 13, "legend.fontsize": 9,
    })

    # --- Plot 1: Experiment 1 (Orthogonalized PCs) ---
    f1 = RESULTS_DIR / "exp1_orthogonalized_pcs.csv"
    if f1.exists():
        df = pd.read_csv(f1)
        fig, ax = plt.subplots(figsize=(7, 5))
        for pc_label, marker, color in [("Original", "o", "#2196F3"), ("Orthogonalized", "s", "#FF5722")]:
            for method, ls in [("Raw", "--"), ("Linear", "-")]:
                sub = df[(df["PCs"] == pc_label) & (df["method"] == method)]
                vals = sub.groupby("seed")["AUC"].first().values
                ax.scatter([f"{pc_label}\n{method}"] * len(vals), vals,
                           alpha=0.5, color=color, marker=marker, s=40)
                mean_val = vals.mean()
                ax.scatter([f"{pc_label}\n{method}"], [mean_val],
                           color=color, marker=marker, s=120, edgecolors="black", linewidths=1.5, zorder=5)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="Chance")
        ax.set_ylabel("AUC")
        ax.set_title("Exp 1: Orthogonalized PCs kill the Linear advantage")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig1_orthogonalized_pcs.png", dpi=200)
        plt.close()

    # --- Plot 2: Experiment 3 (Vary PCA sites) ---
    f3 = RESULTS_DIR / "exp3_vary_pca_sites.csv"
    if f3.exists():
        df = pd.read_csv(f3)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Left: R²(PC, G_true) vs n_pca_sites
        r2_by_n = df.groupby("n_pca_sites")["R2_PC_G"].mean()
        ax1.plot(r2_by_n.index, r2_by_n.values, "o-", color="#2196F3", linewidth=2)
        ax1.set_xlabel("Number of PCA sites")
        ax1.set_ylabel("R²(PC1..5, G_true)")
        ax1.set_title("Spurious R² grows with PCA site count")
        ax1.set_xscale("log")
        ax1.grid(True, alpha=0.3)

        # Right: Linear−Raw gap vs n_pca_sites
        gaps = []
        for n_pca in df["n_pca_sites"].unique():
            sub = df[df["n_pca_sites"] == n_pca]
            for seed in sub["seed"].unique():
                s = sub[sub["seed"] == seed]
                raw = s[s["method"] == "Raw"]["AUC"].values
                lin = s[s["method"] == "Linear"]["AUC"].values
                if len(raw) > 0 and len(lin) > 0:
                    gaps.append({"n_pca_sites": n_pca, "gap": lin[0] - raw[0]})
        gap_df = pd.DataFrame(gaps)
        gap_means = gap_df.groupby("n_pca_sites")["gap"].mean()
        gap_stds = gap_df.groupby("n_pca_sites")["gap"].std()
        ax2.errorbar(gap_means.index, gap_means.values, yerr=gap_stds.values,
                     fmt="o-", color="#FF5722", linewidth=2, capsize=4)
        ax2.axhline(0, color="gray", ls=":", alpha=0.5)
        ax2.set_xlabel("Number of PCA sites")
        ax2.set_ylabel("Linear − Raw AUC")
        ax2.set_title("Linear advantage scales with PCA site count")
        ax2.set_xscale("log")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig2_vary_pca_sites.png", dpi=200)
        plt.close()

    # --- Plot 3: Experiment 6 (BayesR vs P_observed across gens) ---
    f6 = RESULTS_DIR / "exp6_bayesr_across_gens.csv"
    if f6.exists():
        df = pd.read_csv(f6)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        for ax, prs_type, title in [
            (ax1, "BayesR", "BayesR PRS (weak signal, PC-conditioned)"),
            (ax2, "P_observed", "Oracle PRS (strong signal, no PC conditioning)")
        ]:
            sub = df[df["prs_type"] == prs_type]
            for method, color, ls in [("Raw", "#666", "--"), ("Linear", "#2196F3", "-")]:
                ms = sub[sub["method"] == method]
                means = ms.groupby("gens")["AUC"].mean()
                stds = ms.groupby("gens")["AUC"].std()
                ax.errorbar(means.index, means.values, yerr=stds.values,
                            fmt="o-", color=color, linewidth=2, capsize=4, label=method)
            ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
            ax.set_xlabel("Generations of divergence")
            ax.set_ylabel("AUC")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xscale("symlog", linthresh=10)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig3_bayesr_vs_pobs_across_gens.png", dpi=200)
        plt.close()

    # --- Plot 4: R²(PC, G) across gens ---
    f5 = RESULTS_DIR / "exp5_r2_across_gens.csv"
    if f5.exists():
        df = pd.read_csv(f5)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        means5 = df.groupby("gens")["R2_5pc"].mean()
        stds5 = df.groupby("gens")["R2_5pc"].std()
        ax1.errorbar(means5.index, means5.values, yerr=stds5.values,
                     fmt="o-", color="#2196F3", linewidth=2, capsize=4, label="5 PCs")
        means20 = df.groupby("gens")["R2_20pc"].mean()
        stds20 = df.groupby("gens")["R2_20pc"].std()
        ax1.errorbar(means20.index, means20.values, yerr=stds20.values,
                     fmt="s-", color="#FF5722", linewidth=2, capsize=4, label="20 PCs")
        ax1.set_xlabel("Generations of divergence")
        ax1.set_ylabel("R²(PCs, G_true)")
        ax1.set_title("PC−G_true correlation across divergence")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale("symlog", linthresh=10)

        sep_means = df.groupby("gens")["PC1_separation"].mean()
        ax2.plot(sep_means.index, sep_means.values, "o-", color="#4CAF50", linewidth=2)
        ax2.set_xlabel("Generations of divergence")
        ax2.set_ylabel("PC1 Cohen's d (pop separation)")
        ax2.set_title("PC1 population separation across divergence")
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("symlog", linthresh=10)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "fig4_r2_across_gens.png", dpi=200)
        plt.close()

    print(f"\nPlots saved to {RESULTS_DIR}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    experiments = {
        "1": ("Orthogonalized PCs (msprime)", experiment_1_orthogonalized_pcs),
        "2": ("LD vs No-LD", experiment_2_ld_vs_no_ld),
        "3": ("Causal-PCA overlap", experiment_3_causal_pca_overlap),
        "4": ("P_observed across gens", experiment_4_pobs_across_generations),
        "5": ("R² across gens", experiment_5_r2_across_generations),
        "6": ("BayesR across gens", experiment_6_bayesr_across_generations),
    }

    if len(sys.argv) > 1:
        to_run = sys.argv[1:]
    else:
        to_run = list(experiments.keys())

    print("=" * 70)
    print("ORTHOGONAL EXPERIMENTS: Gen 0 Phenomenon")
    print("=" * 70)

    for key in to_run:
        if key in experiments:
            name, func = experiments[key]
            print(f"\n>>> Running Experiment {key}: {name}")
            func()
        elif key == "plot":
            make_plots()
        else:
            print(f"Unknown experiment: {key}")

    make_plots()
    print("\nAll done. Results in gen0_analysis/")


if __name__ == "__main__":
    main()
