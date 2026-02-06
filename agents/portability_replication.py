#!/usr/bin/env python3
"""
PRS Portability Replication in msprime Two-Population Simulations.

Tests whether the PRS portability problem (PRS trained in one population
performs worse in a diverged population) can be replicated using simple
msprime demographic models.

Mechanism: GWAS discovers tag SNPs in LD with causal variants. When LD
patterns differ between populations (due to drift/bottleneck), tag-based
PRS loses accuracy. This experiment tests whether that LD divergence is
large enough on a 5 Mb genome to produce detectable portability loss.

Usage:
    python agents/portability_replication.py
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Add sims/ to path so we can reuse demography builders
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sims"))

import msprime

from sim_two_pop import _build_demography_divergence, _build_demography_bottleneck
from sim_pops import _diploid_index_pairs, _solve_intercept_for_prevalence


# ---------------------------------------------------------------------------
# Section A: Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    scenario: str  # "divergence" or "bottleneck"
    divergence_gens: int
    n_causal: int
    n_per_pop: int
    seq_len: int
    recomb_rate: float
    mut_rate: float
    ne: int
    h2: float
    prevalence: float
    n_array_sites: int  # 0 = use all variants (full sequence)
    seed: int


# Default sweep parameters
N_CAUSAL_LEVELS = [50, 200, 1000, 5000]
DIVERGENCE_LEVELS = [0, 100, 500, 1000, 5000, 10_000]
SCENARIOS = ["divergence", "bottleneck"]
SEEDS = [1, 2, 3]
N_PER_POP = 2000
SEQ_LEN = 5_000_000
RECOMB_RATE = 1e-8
MUT_RATE = 1e-8
NE = 10_000
H2 = 0.50
PREVALENCE = 0.10
N_ARRAY_SITES = 2000  # for array variant

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "portability_results")
N_WORKERS = max(1, min(6, (os.cpu_count() or 1)))


# ---------------------------------------------------------------------------
# Section B: Two-population simulation
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    """Holds all data from a single simulation run."""
    X: np.ndarray        # (n_ind, n_variants) diploid dosage matrix, int8
    pos: np.ndarray      # (n_variants,) variant positions
    maf: np.ndarray      # (n_variants,) minor allele frequencies
    pop_idx: np.ndarray  # (n_ind,) 0=POP0, 1=POP1
    G_true: np.ndarray   # (n_ind,) standardized true genetic value
    y: np.ndarray        # (n_ind,) binary phenotype
    causal_idx: np.ndarray  # indices into variant axis for causal variants
    causal_betas: np.ndarray  # true effect sizes


def simulate_two_pop(cfg: Config) -> SimResult:
    """Simulate two populations and build genotype matrix + trait."""
    n0 = cfg.n_per_pop
    n1 = cfg.n_per_pop
    total = n0 + n1

    # --- Simulate tree sequence ---
    if cfg.divergence_gens == 0:
        ts = msprime.sim_ancestry(
            samples=[msprime.SampleSet(total, ploidy=2)],
            sequence_length=cfg.seq_len,
            recombination_rate=cfg.recomb_rate,
            ploidy=2,
            population_size=cfg.ne,
            random_seed=cfg.seed,
            model="hudson",
        )
        ts = msprime.sim_mutations(ts, rate=cfg.mut_rate, random_seed=cfg.seed + 1)
        # Arbitrary split into two "populations"
        pop_idx = np.array([0] * n0 + [1] * n1, dtype=np.int32)
    else:
        split_time = cfg.divergence_gens
        bottle_ne = max(100, cfg.ne // 10)

        if cfg.scenario == "divergence":
            dem = _build_demography_divergence(split_time, cfg.ne)
        else:
            dem = _build_demography_bottleneck(split_time, cfg.ne, bottle_ne)

        ts = msprime.sim_ancestry(
            samples={"pop0": n0, "pop1": n1},
            demography=dem,
            sequence_length=cfg.seq_len,
            recombination_rate=cfg.recomb_rate,
            ploidy=2,
            random_seed=cfg.seed,
            model="hudson",
        )
        ts = msprime.sim_mutations(ts, rate=cfg.mut_rate, random_seed=cfg.seed + 1)

        a_idx, b_idx, pi, _ = _diploid_index_pairs(ts)
        pop_idx = pi.copy()
        if pop_idx.shape[0] != total:
            pop_idx = np.array([0] * n0 + [1] * n1, dtype=np.int32)

    # --- Build full genotype matrix ---
    n_ind = total
    m = ts.num_sites
    if m < 100:
        raise RuntimeError(f"Too few variant sites: {m}")

    X = np.empty((n_ind, m), dtype=np.int8)
    pos = np.empty(m, dtype=np.float64)
    maf_arr = np.empty(m, dtype=np.float64)

    for j, var in enumerate(ts.variants()):
        g_h = var.genotypes.astype(np.int16)
        g_d = g_h[0::2] + g_h[1::2]
        X[:, j] = g_d.astype(np.int8)
        pos[j] = ts.site(var.site.id).position
        p = g_d.mean() / 2.0
        maf_arr[j] = min(p, 1.0 - p)

    # --- Select causal variants and build G_true ---
    rng = np.random.default_rng(cfg.seed + 7)
    eligible = np.where(maf_arr >= 0.01)[0]
    n_causal = min(cfg.n_causal, len(eligible))
    causal_idx = np.sort(rng.choice(eligible, size=n_causal, replace=False))

    beta_sd = math.sqrt(cfg.h2 / n_causal)
    causal_betas = rng.normal(0.0, beta_sd, size=n_causal).astype(np.float64)

    G_raw = X[:, causal_idx].astype(np.float64) @ causal_betas
    G_mean = G_raw.mean()
    G_std = G_raw.std()
    if G_std < 1e-12:
        G_std = 1.0
    G_true = (G_raw - G_mean) / G_std

    # --- Binary phenotype ---
    b0 = _solve_intercept_for_prevalence(cfg.prevalence, G_true)
    y = rng.binomial(1, expit(b0 + G_true)).astype(np.int8)

    return SimResult(
        X=X, pos=pos, maf=maf_arr, pop_idx=pop_idx,
        G_true=G_true, y=y,
        causal_idx=causal_idx, causal_betas=causal_betas,
    )


# ---------------------------------------------------------------------------
# Section C: PRS construction
# ---------------------------------------------------------------------------

def build_marginal_prs(
    X: np.ndarray,
    G_true: np.ndarray,
    train_idx: np.ndarray,
    score_idx: np.ndarray,
    variant_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build marginal-regression PRS: correlate each variant with G_true in
    training set, then score on score_idx.

    Returns (PRS_scores, beta_hat).
    """
    if variant_mask is not None:
        Xt = X[np.ix_(train_idx, variant_mask)].astype(np.float64)
        Xs = X[np.ix_(score_idx, variant_mask)].astype(np.float64)
    else:
        Xt = X[train_idx].astype(np.float64)
        Xs = X[score_idx].astype(np.float64)

    # Standardize using training statistics
    mu = Xt.mean(axis=0)
    sd = Xt.std(axis=0)
    sd[sd == 0] = 1.0

    Xt_z = (Xt - mu) / sd
    Xs_z = (Xs - mu) / sd

    # Marginal regression: beta_hat_j = X_z_j' @ G_true_train / n_train
    G_train = G_true[train_idx]
    G_centered = G_train - G_train.mean()
    beta_hat = Xt_z.T @ G_centered / len(train_idx)

    # Score
    prs = Xs_z @ beta_hat
    return prs, beta_hat


def build_oracle_prs(
    X: np.ndarray,
    causal_idx: np.ndarray,
    causal_betas: np.ndarray,
    score_idx: np.ndarray,
) -> np.ndarray:
    """
    Oracle PRS using true causal variant indices and betas.
    No LD dependence — should show zero portability loss.
    """
    return X[np.ix_(score_idx, causal_idx)].astype(np.float64) @ causal_betas


# ---------------------------------------------------------------------------
# Section D: Evaluation
# ---------------------------------------------------------------------------

@dataclass
class PortabilityResult:
    seed: int
    scenario: str
    n_causal: int
    divergence_gens: int
    use_array: bool
    n_variants: int
    n_variants_used: int
    n_train: int

    # GWAS (marginal regression) PRS
    r2_gwas_holdout: float
    r2_gwas_pop1: float
    portability_ratio_gwas: float

    # Thresholded GWAS PRS (top 5% of variants by |beta|)
    r2_thresh_holdout: float
    r2_thresh_pop1: float
    portability_ratio_thresh: float
    n_variants_thresh: int

    # Oracle PRS
    r2_oracle_holdout: float
    r2_oracle_pop1: float
    portability_ratio_oracle: float

    # Diagnostic: median tag-causal distance for array variant
    median_tag_causal_dist_bp: float


def safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson R^2, handling degenerate cases."""
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def _score_given_variant_mask(
    sim: SimResult,
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    pop1: np.ndarray,
    variant_mask: Optional[np.ndarray],
    use_array: bool,
    cfg: Config,
) -> PortabilityResult:
    """Score one simulated dataset under either full-sequence or array mask."""
    n_variants_used = len(variant_mask) if variant_mask is not None else sim.X.shape[1]

    prs_holdout, beta_hat = build_marginal_prs(
        sim.X, sim.G_true, train_idx, holdout_idx, variant_mask
    )
    prs_pop1, _ = build_marginal_prs(
        sim.X, sim.G_true, train_idx, pop1, variant_mask
    )

    r2_gwas_h = safe_r2(prs_holdout, sim.G_true[holdout_idx])
    r2_gwas_p1 = safe_r2(prs_pop1, sim.G_true[pop1])
    ratio_gwas = r2_gwas_p1 / r2_gwas_h if r2_gwas_h > 1e-6 else float("nan")

    abs_beta = np.abs(beta_hat)
    thresh = np.percentile(abs_beta, 95)
    keep_mask = abs_beta >= thresh
    n_thresh = int(keep_mask.sum())

    if variant_mask is not None:
        thresh_variant_mask = variant_mask[keep_mask]
    else:
        thresh_variant_mask = np.where(keep_mask)[0]

    if n_thresh > 0:
        prs_t_holdout, _ = build_marginal_prs(
            sim.X, sim.G_true, train_idx, holdout_idx, thresh_variant_mask
        )
        prs_t_pop1, _ = build_marginal_prs(
            sim.X, sim.G_true, train_idx, pop1, thresh_variant_mask
        )
        r2_t_h = safe_r2(prs_t_holdout, sim.G_true[holdout_idx])
        r2_t_p1 = safe_r2(prs_t_pop1, sim.G_true[pop1])
    else:
        r2_t_h = 0.0
        r2_t_p1 = 0.0
    ratio_thresh = r2_t_p1 / r2_t_h if r2_t_h > 1e-6 else float("nan")

    oracle_holdout = build_oracle_prs(sim.X, sim.causal_idx, sim.causal_betas, holdout_idx)
    oracle_pop1 = build_oracle_prs(sim.X, sim.causal_idx, sim.causal_betas, pop1)
    r2_o_h = safe_r2(oracle_holdout, sim.G_true[holdout_idx])
    r2_o_p1 = safe_r2(oracle_pop1, sim.G_true[pop1])
    ratio_oracle = r2_o_p1 / r2_o_h if r2_o_h > 1e-6 else float("nan")

    if use_array and variant_mask is not None and len(variant_mask) > 0:
        causal_positions = sim.pos[sim.causal_idx]
        array_positions = sim.pos[variant_mask]
        median_dist = float(np.median([np.min(np.abs(causal_positions - ap)) for ap in array_positions]))
    else:
        median_dist = 0.0

    return PortabilityResult(
        seed=cfg.seed,
        scenario=cfg.scenario,
        n_causal=cfg.n_causal,
        divergence_gens=cfg.divergence_gens,
        use_array=use_array,
        n_variants=sim.X.shape[1],
        n_variants_used=n_variants_used,
        n_train=len(train_idx),
        r2_gwas_holdout=r2_gwas_h,
        r2_gwas_pop1=r2_gwas_p1,
        portability_ratio_gwas=ratio_gwas,
        r2_thresh_holdout=r2_t_h,
        r2_thresh_pop1=r2_t_p1,
        portability_ratio_thresh=ratio_thresh,
        n_variants_thresh=n_thresh,
        r2_oracle_holdout=r2_o_h,
        r2_oracle_pop1=r2_o_p1,
        portability_ratio_oracle=ratio_oracle,
        median_tag_causal_dist_bp=median_dist,
    )


def run_single_config(cfg: Config) -> List[PortabilityResult]:
    """Run one simulation and produce both full-seq and array PRS results."""
    try:
        sim = simulate_two_pop(cfg)
    except Exception as e:
        print(f"  [SKIP] {cfg.scenario} g={cfg.divergence_gens} nc={cfg.n_causal} s={cfg.seed}: {e}")
        return []

    pop0 = np.where(sim.pop_idx == 0)[0]
    pop1 = np.where(sim.pop_idx == 1)[0]

    # Split POP0 into train (70%) and holdout (30%)
    rng_split = np.random.default_rng(cfg.seed + 100)
    shuffled = rng_split.permutation(pop0)
    n_train = int(0.7 * len(pop0))
    train_idx = shuffled[:n_train]
    holdout_idx = shuffled[n_train:]

    n_total = sim.X.shape[1]
    step = max(1, n_total // cfg.n_array_sites)
    array_mask = np.arange(0, n_total, step)[:cfg.n_array_sites]

    full_res = _score_given_variant_mask(
        sim, train_idx, holdout_idx, pop1, None, False, cfg
    )
    array_res = _score_given_variant_mask(
        sim, train_idx, holdout_idx, pop1, array_mask, True, cfg
    )
    return [full_res, array_res]


# ---------------------------------------------------------------------------
# Section E: Sweep orchestration
# ---------------------------------------------------------------------------

def build_all_configs() -> List[Config]:
    """Build one simulation config per scenario/divergence/n_causal/seed."""
    configs = []
    for scenario in SCENARIOS:
        for n_causal in N_CAUSAL_LEVELS:
            for div_gens in DIVERGENCE_LEVELS:
                for seed in SEEDS:
                    configs.append(Config(
                        scenario=scenario,
                        divergence_gens=div_gens,
                        n_causal=n_causal,
                        n_per_pop=N_PER_POP,
                        seq_len=SEQ_LEN,
                        recomb_rate=RECOMB_RATE,
                        mut_rate=MUT_RATE,
                        ne=NE,
                        h2=H2,
                        prevalence=PREVALENCE,
                        n_array_sites=N_ARRAY_SITES,
                        seed=seed,
                    ))
    return configs


def run_sweep() -> pd.DataFrame:
    """Run the full sweep, return DataFrame of results."""
    configs = build_all_configs()
    print(f"Total simulations: {len(configs)} (each yields full-seq + array)")
    print(f"Workers: {N_WORKERS}")

    results = []
    t0 = time.time()
    completed = 0
    with cf.ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        fut_to_cfg = {ex.submit(run_single_config, cfg): cfg for cfg in configs}
        for fut in cf.as_completed(fut_to_cfg):
            cfg = fut_to_cfg[fut]
            completed += 1
            elapsed = time.time() - t0
            eta = elapsed / completed * (len(configs) - completed) if completed > 0 else 0
            print(
                f"[{completed}/{len(configs)}] {cfg.scenario} g={cfg.divergence_gens} "
                f"nc={cfg.n_causal} s={cfg.seed} (elapsed={elapsed:.0f}s, ETA={eta:.0f}s)"
            )
            try:
                res_list = fut.result()
                if res_list:
                    results.extend(res_list)
            except Exception as e:
                print(
                    f"  [FAIL] {cfg.scenario} g={cfg.divergence_gens} nc={cfg.n_causal} "
                    f"s={cfg.seed}: {e}"
                )

    elapsed = time.time() - t0
    print(f"\nSweep complete: {len(results)} scored rows in {elapsed:.0f}s")

    rows = []
    for r in results:
        rows.append({
            "seed": r.seed,
            "scenario": r.scenario,
            "n_causal": r.n_causal,
            "divergence_gens": r.divergence_gens,
            "use_array": r.use_array,
            "n_variants": r.n_variants,
            "n_variants_used": r.n_variants_used,
            "n_train": r.n_train,
            "r2_gwas_holdout": r.r2_gwas_holdout,
            "r2_gwas_pop1": r.r2_gwas_pop1,
            "portability_ratio_gwas": r.portability_ratio_gwas,
            "r2_thresh_holdout": r.r2_thresh_holdout,
            "r2_thresh_pop1": r.r2_thresh_pop1,
            "portability_ratio_thresh": r.portability_ratio_thresh,
            "n_variants_thresh": r.n_variants_thresh,
            "r2_oracle_holdout": r.r2_oracle_holdout,
            "r2_oracle_pop1": r.r2_oracle_pop1,
            "portability_ratio_oracle": r.portability_ratio_oracle,
            "median_tag_causal_dist_bp": r.median_tag_causal_dist_bp,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section F: Figures
# ---------------------------------------------------------------------------

def _apply_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "grid.color": "#d0d0d0",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "legend.fontsize": 9,
    })


def _mean_ci(vals: np.ndarray) -> Tuple[float, float, float]:
    """Mean and 95% CI."""
    m = np.nanmean(vals)
    if len(vals) < 2:
        return m, m, m
    se = np.nanstd(vals, ddof=1) / np.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se


def fig1_portability_ratio(df: pd.DataFrame, out_dir: str):
    """Portability ratio vs divergence, panels for sequence vs array."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_i, (use_array, title) in enumerate([(False, "Full Sequence"), (True, "Array (2K sites)")]):
        ax = axes[ax_i]
        sub = df[(df["use_array"] == use_array) & (df["scenario"] == "divergence")]

        colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(N_CAUSAL_LEVELS)))
        for nc, color in zip(N_CAUSAL_LEVELS, colors):
            nc_sub = sub[sub["n_causal"] == nc]
            means, lo, hi = [], [], []
            gens_plot = []
            for g in DIVERGENCE_LEVELS:
                vals = nc_sub[nc_sub["divergence_gens"] == g]["portability_ratio_gwas"].values
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    m, l, h = _mean_ci(vals)
                    means.append(m)
                    lo.append(l)
                    hi.append(h)
                    gens_plot.append(g)
            if means:
                ax.plot(gens_plot, means, "o-", color=color, label=f"n_causal={nc}", linewidth=2, markersize=5)
                ax.fill_between(gens_plot, lo, hi, alpha=0.15, color=color)

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="No loss")
        ax.set_xlabel("Divergence (generations)")
        ax.set_ylabel("Portability Ratio (R\u00b2 POP1 / R\u00b2 holdout)")
        ax.set_title(f"GWAS PRS Portability \u2014 {title}")
        ax.set_xscale("symlog", linthresh=50)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig1_portability_ratio.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig2_absolute_r2(df: pd.DataFrame, out_dir: str):
    """Absolute R^2 in each population across divergence."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, use_array in enumerate([False, True]):
        for col, nc in enumerate([200, 5000]):
            ax = axes[row][col]
            sub = df[(df["use_array"] == use_array) & (df["n_causal"] == nc)
                      & (df["scenario"] == "divergence")]

            for pop_label, r2_col, color, marker in [
                ("POP0 holdout", "r2_gwas_holdout", "#2196F3", "o"),
                ("POP1", "r2_gwas_pop1", "#F44336", "s"),
                ("Oracle holdout", "r2_oracle_holdout", "#2196F3", "^"),
                ("Oracle POP1", "r2_oracle_pop1", "#F44336", "v"),
            ]:
                means, gens_plot = [], []
                ls = "-" if "Oracle" not in pop_label else "--"
                for g in DIVERGENCE_LEVELS:
                    vals = sub[sub["divergence_gens"] == g][r2_col].values
                    vals = vals[np.isfinite(vals)]
                    if len(vals) > 0:
                        means.append(np.nanmean(vals))
                        gens_plot.append(g)
                if means:
                    ax.plot(gens_plot, means, f"{marker}{ls}", color=color,
                            label=pop_label, linewidth=1.5, markersize=5, alpha=0.8)

            array_label = "Array" if use_array else "Full Seq"
            ax.set_title(f"n_causal={nc}, {array_label}")
            ax.set_xlabel("Divergence (generations)")
            ax.set_ylabel("R\u00b2(PRS, G_true)")
            ax.set_xscale("symlog", linthresh=50)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig2_absolute_r2.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig3_divergence_vs_bottleneck(df: pd.DataFrame, out_dir: str):
    """Compare divergence and bottleneck scenarios."""
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax_i, use_array in enumerate([False, True]):
        ax = axes[ax_i]
        array_label = "Array" if use_array else "Full Sequence"

        for scenario, color, ls in [("divergence", "#2196F3", "-"), ("bottleneck", "#F44336", "--")]:
            sub = df[(df["use_array"] == use_array) & (df["scenario"] == scenario)
                      & (df["n_causal"] == 200)]

            means, lo, hi, gens_plot = [], [], [], []
            for g in DIVERGENCE_LEVELS:
                vals = sub[sub["divergence_gens"] == g]["portability_ratio_gwas"].values
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    m, l, h = _mean_ci(vals)
                    means.append(m)
                    lo.append(l)
                    hi.append(h)
                    gens_plot.append(g)
            if means:
                ax.plot(gens_plot, means, f"o{ls}", color=color,
                        label=f"{scenario}", linewidth=2, markersize=5)
                ax.fill_between(gens_plot, lo, hi, alpha=0.15, color=color)

        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Divergence (generations)")
        ax.set_ylabel("Portability Ratio")
        ax.set_title(f"Divergence vs Bottleneck (n_causal=200) \u2014 {array_label}")
        ax.set_xscale("symlog", linthresh=50)
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig3_div_vs_bottleneck.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def fig4_oracle_control(df: pd.DataFrame, out_dir: str):
    """Oracle PRS portability ratio (should stay near 1.0 everywhere)."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    sub = df[(df["scenario"] == "divergence")]
    colors = plt.cm.tab10(np.linspace(0, 1, len(N_CAUSAL_LEVELS)))

    for nc, color in zip(N_CAUSAL_LEVELS, colors):
        for use_array, ls in [(False, "-"), (True, "--")]:
            nc_sub = sub[(sub["n_causal"] == nc) & (sub["use_array"] == use_array)]
            means, gens_plot = [], []
            for g in DIVERGENCE_LEVELS:
                vals = nc_sub[nc_sub["divergence_gens"] == g]["portability_ratio_oracle"].values
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    means.append(np.nanmean(vals))
                    gens_plot.append(g)
            if means:
                lbl = f"nc={nc} {'array' if use_array else 'seq'}"
                ax.plot(gens_plot, means, f"o{ls}", color=color, label=lbl,
                        linewidth=1.5, markersize=4, alpha=0.7)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Divergence (generations)")
    ax.set_ylabel("Oracle Portability Ratio")
    ax.set_title("Oracle PRS (True Causal Betas) \u2014 Should Stay Near 1.0")
    ax.set_xscale("symlog", linthresh=50)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0.5, 1.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "fig4_oracle_control.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Section G: Summary table
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame):
    """Print human-readable summary table."""
    print("\n" + "=" * 100)
    print("PORTABILITY REPLICATION SUMMARY")
    print("=" * 100)

    for scenario in SCENARIOS:
        for use_array in [False, True]:
            array_label = "Array (2K)" if use_array else "Full Seq"
            print(f"\n--- {scenario.upper()} | {array_label} ---")
            print(f"{'n_causal':>10} {'div_gens':>10} {'R2_hold':>10} {'R2_pop1':>10} "
                  f"{'ratio_gwas':>12} {'R2_orc_h':>10} {'R2_orc_p1':>10} {'ratio_orc':>12}")

            sub = df[(df["scenario"] == scenario) & (df["use_array"] == use_array)]
            for nc in N_CAUSAL_LEVELS:
                for g in DIVERGENCE_LEVELS:
                    row = sub[(sub["n_causal"] == nc) & (sub["divergence_gens"] == g)]
                    if len(row) == 0:
                        continue
                    r2h = row["r2_gwas_holdout"].mean()
                    r2p = row["r2_gwas_pop1"].mean()
                    rg = row["portability_ratio_gwas"].mean()
                    oh = row["r2_oracle_holdout"].mean()
                    op = row["r2_oracle_pop1"].mean()
                    ro = row["portability_ratio_oracle"].mean()
                    print(f"{nc:>10} {g:>10} {r2h:>10.4f} {r2p:>10.4f} "
                          f"{rg:>12.4f} {oh:>10.4f} {op:>10.4f} {ro:>12.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Parameters: n_per_pop={N_PER_POP}, seq_len={SEQ_LEN/1e6:.0f}Mb, "
          f"h2={H2}, Ne={NE}")
    n_sims = len(N_CAUSAL_LEVELS) * len(DIVERGENCE_LEVELS) * len(SCENARIOS) * len(SEEDS)
    print(
        f"Sweep: {len(N_CAUSAL_LEVELS)} n_causal x {len(DIVERGENCE_LEVELS)} div_gens "
        f"x {len(SCENARIOS)} scenarios x {len(SEEDS)} seeds = {n_sims} simulations"
    )
    print(f"Expected scored rows: {n_sims * 2} (full-seq + array)")
    print()

    df = run_sweep()

    csv_path = os.path.join(OUTPUT_DIR, "portability_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    print("\nGenerating figures...")
    fig1_portability_ratio(df, OUTPUT_DIR)
    fig2_absolute_r2(df, OUTPUT_DIR)
    fig3_divergence_vs_bottleneck(df, OUTPUT_DIR)
    fig4_oracle_control(df, OUTPUT_DIR)

    print_summary(df)

    # Sanity checks
    print("\n--- SANITY CHECKS ---")
    gen0 = df[df["divergence_gens"] == 0]
    if len(gen0) > 0:
        mean_ratio_gen0 = gen0["portability_ratio_gwas"].mean()
        print(f"Gen 0 mean portability ratio (GWAS): {mean_ratio_gen0:.4f} (expect ~1.0)")

    oracle = df[df["portability_ratio_oracle"].notna()]
    if len(oracle) > 0:
        mean_oracle = oracle["portability_ratio_oracle"].mean()
        print(f"Overall mean oracle portability ratio: {mean_oracle:.4f} (expect ~1.0)")

    # Key result: portability at max divergence with array + few causal
    key = df[(df["divergence_gens"] == 10000) & (df["use_array"] == True)
             & (df["n_causal"] == 50) & (df["scenario"] == "divergence")]
    if len(key) > 0:
        kr = key["portability_ratio_gwas"].mean()
        print(f"Key test (nc=50, g=10K, array, divergence): portability ratio = {kr:.4f}")
        if kr < 0.8:
            print("  -> PORTABILITY LOSS DETECTED!")
        else:
            print("  -> No clear portability loss at this configuration.")

    print("\nDone.")


if __name__ == "__main__":
    main()
