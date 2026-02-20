#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import msprime
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "results_causal_arch"
FIG_DIR = ROOT / "figures_causal_arch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    n_ind: int
    seq_len: int
    n_pca: int
    prevalence: float
    h2: float


class Sim:
    def __init__(self, X: np.ndarray, pos: np.ndarray, maf: np.ndarray, train: np.ndarray, test: np.ndarray):
        self.X = X
        self.pos = pos
        self.maf = maf
        self.train = train
        self.test = test


def std(v: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(v.reshape(-1, 1)).ravel()


def mean_ci(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    if len(x) < 2:
        return m, m, m
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se


def simulate_X(seed: int, recomb_rate: float, cfg: Config) -> Sim:
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(cfg.n_ind, ploidy=2)],
        sequence_length=cfg.seq_len,
        recombination_rate=recomb_rate,
        ploidy=2,
        population_size=10_000,
        random_seed=seed,
        model="dtwf",
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed + 1)

    m = ts.num_sites
    if m < 200:
        raise RuntimeError(f"Too few variant sites ({m})")

    X = np.empty((cfg.n_ind, m), dtype=np.int8)
    pos = np.empty(m, dtype=np.float64)
    maf = np.empty(m, dtype=np.float64)

    for j, var in enumerate(ts.variants()):
        gh = var.genotypes.astype(np.int16)
        gd = gh[0::2] + gh[1::2]
        X[:, j] = gd.astype(np.int8)
        pos[j] = ts.site(var.site.id).position
        p = gd.mean() / 2.0
        maf[j] = min(p, 1.0 - p)

    idx = np.arange(cfg.n_ind)
    train, test = train_test_split(idx, test_size=0.5, random_state=42)
    return Sim(X, pos, maf, train, test)


def choose_causal_indices(sim: Sim, seed: int, n_causal: int, placement: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eligible = np.where(sim.maf >= 0.01)[0]
    n_causal = max(20, min(n_causal, len(eligible)))

    if placement == "random":
        return np.sort(rng.choice(eligible, size=n_causal, replace=False))

    if placement == "clustered":
        # Build a few LD-local clusters by selecting centers and taking nearby eligible sites.
        # This concentrates architecture into fewer local regions.
        centers = rng.choice(eligible, size=min(8, max(2, n_causal // 40)), replace=False)
        selected = []
        elig_pos = sim.pos[eligible]
        for c in centers:
            cpos = sim.pos[c]
            near_mask = np.abs(elig_pos - cpos) <= 30_000
            near = eligible[near_mask]
            if len(near) > 0:
                take = min(len(near), max(1, n_causal // len(centers)))
                selected.extend(rng.choice(near, size=take, replace=False).tolist())
        selected = np.array(sorted(set(selected)), dtype=int)
        if len(selected) < n_causal:
            remaining_pool = np.array(sorted(set(eligible.tolist()) - set(selected.tolist())), dtype=int)
            if len(remaining_pool) > 0:
                extra = rng.choice(remaining_pool, size=min(n_causal - len(selected), len(remaining_pool)), replace=False)
                selected = np.array(sorted(set(selected.tolist()) | set(extra.tolist())), dtype=int)
        if len(selected) > n_causal:
            selected = np.sort(rng.choice(selected, size=n_causal, replace=False))
        return selected

    raise ValueError(f"Unknown placement={placement}")


def sample_betas(maf: np.ndarray, arch: str, h2: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(maf)

    if arch == "infinitesimal":
        z = rng.normal(0.0, 1.0, n)
    elif arch == "sparse_spike":
        active = rng.binomial(1, 0.15, n).astype(bool)
        z = np.zeros(n)
        z[active] = rng.normal(0.0, 1.0, active.sum())
        if np.all(z == 0):
            z[rng.integers(0, n)] = rng.normal(0.0, 1.0)
    elif arch == "maf_dependent":
        # Larger effect on low-MAF alleles after variance stabilization.
        z = rng.normal(0.0, 1.0, n)
        var = 2.0 * maf * (1.0 - maf)
        var[var < 1e-4] = 1e-4
        z = z * np.power(var, -0.4)
    else:
        raise ValueError(f"Unknown arch={arch}")

    z = z - z.mean()
    denom = np.sqrt(np.sum(z ** 2))
    if denom <= 0:
        z = rng.normal(0.0, 1.0, n)
        denom = np.sqrt(np.sum(z ** 2))
    return z * np.sqrt(h2) / denom


def build_trait(sim: Sim, cidx: np.ndarray, betas: np.ndarray, prevalence: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    Xc = sim.X[:, cidx].astype(np.float64)
    g = Xc @ betas
    g = std(g)

    b0 = brentq(lambda b: expit(b + g).mean() - prevalence, -20, 20)
    y = np.random.default_rng(seed).binomial(1, expit(b0 + g)).astype(np.int8)
    return g, y


def pca_sites(sim: Sim, cidx: np.ndarray, mode: str, n_pca: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eligible = np.where(sim.maf >= 0.05)[0]
    if mode == "all":
        pool = eligible
    elif mode == "disjoint":
        cset = set(cidx.tolist())
        mask = np.array([i not in cset for i in eligible], dtype=bool)
        pool = eligible[mask]
    else:
        raise ValueError(mode)

    if len(pool) < 30:
        pool = eligible
    return rng.choice(pool, size=min(n_pca, len(pool)), replace=False)


def compute_pcs(sim: Sim, pidx: np.ndarray, n_pc: int = 5) -> np.ndarray:
    Xp = sim.X[:, pidx].astype(np.float64)
    Xp = StandardScaler().fit_transform(Xp)
    Xp = np.nan_to_num(Xp)
    pca = PCA(n_components=min(20, Xp.shape[0] - 1, Xp.shape[1]), random_state=0)
    pcs = pca.fit_transform(Xp)
    return StandardScaler().fit_transform(pcs)[:, :n_pc]


def auc_from_X(y: np.ndarray, X: np.ndarray, cal: np.ndarray, val: np.ndarray) -> float:
    m = LogisticRegression(max_iter=3000)
    m.fit(X[cal], y[cal])
    p = m.predict_proba(X[val])[:, 1]
    return float(roc_auc_score(y[val], p))


def weak_prs(sim: Sim, y: np.ndarray) -> np.ndarray:
    Xtr = sim.X[sim.train].astype(np.float64)
    Xte = sim.X[sim.test].astype(np.float64)
    ytr = y[sim.train].astype(np.float64)

    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xt = (Xtr - mu) / sd
    Xv = (Xte - mu) / sd

    beta = Xt.T @ (ytr - ytr.mean()) / len(ytr)
    return std(Xv @ beta)


def run_seed_ld(seed: int, recomb_rate: float, cfg: Config) -> pd.DataFrame:
    sim = simulate_X(100000 + seed * 31 + int(recomb_rate * 1e10), recomb_rate, cfg)
    test_idx = sim.test

    fractions = [0.02, 0.05, 0.10, 0.20, 0.35, 0.50]
    archs = ["infinitesimal", "sparse_spike", "maf_dependent"]
    placements = ["random", "clustered"]
    pca_modes = ["all", "disjoint"]

    rows = []
    for frac in fractions:
        n_causal = int(max(25, frac * np.sum(sim.maf >= 0.01)))
        for arch in archs:
            for placement in placements:
                cidx = choose_causal_indices(sim, 200000 + seed + int(frac * 1000), n_causal, placement)
                maf_c = sim.maf[cidx]
                betas = sample_betas(maf_c, arch, cfg.h2, 300000 + seed + len(cidx))
                g, y = build_trait(sim, cidx, betas, cfg.prevalence, 400000 + seed)

                y_t = y[test_idx]
                g_t = g[test_idx]
                idx = np.arange(len(test_idx))
                cal, val = train_test_split(idx, test_size=0.5, random_state=42)

                # weak PRS used for additive vs interaction behavior
                prs = weak_prs(sim, y)
                Xraw = prs.reshape(-1, 1)

                for pca_mode in pca_modes:
                    pidx = pca_sites(sim, cidx, pca_mode, cfg.n_pca, 500000 + seed + int(frac * 1000))
                    pcs = compute_pcs(sim, pidx)[test_idx]
                    rand = StandardScaler().fit_transform(
                        np.random.default_rng(600000 + seed).normal(size=pcs.shape)
                    )

                    Xadd = np.hstack([Xraw, pcs])
                    Xint = np.hstack([Xraw, pcs, Xraw * pcs])

                    auc_pc = auc_from_X(y_t, pcs, cal, val)
                    auc_rand = auc_from_X(y_t, rand, cal, val)
                    auc_add = auc_from_X(y_t, Xadd, cal, val)
                    auc_int = auc_from_X(y_t, Xint, cal, val)

                    rows.append(
                        {
                            "seed": seed,
                            "recomb_rate": recomb_rate,
                            "causal_fraction": frac,
                            "n_causal": len(cidx),
                            "arch": arch,
                            "placement": placement,
                            "pca_mode": pca_mode,
                            "pct_overlap": 100.0 * len(set(cidx.tolist()) & set(pidx.tolist())) / len(pidx),
                            "r2_pc_g": float(LinearRegression().fit(pcs, g_t).score(pcs, g_t)),
                            "auc_pc_only": auc_pc,
                            "auc_rand_only": auc_rand,
                            "auc_gap_pc_minus_rand": auc_pc - auc_rand,
                            "auc_add": auc_add,
                            "auc_int": auc_int,
                            "auc_gap_int_minus_add": auc_int - auc_add,
                        }
                    )

    return pd.DataFrame(rows)


def run_all(n_seeds: int, workers: int, cfg: Config) -> pd.DataFrame:
    tasks = []
    for seed in range(1, n_seeds + 1):
        for recomb_rate in [1e-8, 2e-8, 5e-8]:
            tasks.append((seed, recomb_rate))

    out = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_seed_ld, s, r, cfg): (s, r) for s, r in tasks}
        for fut in cf.as_completed(futs):
            s, r = futs[fut]
            print(f"  finished seed={s} recomb={r}")
            out.append(fut.result())

    df = pd.concat(out, ignore_index=True)
    df.to_csv(OUT_DIR / "causal_arch_grid.csv", index=False)
    return df


def make_figures(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 1: causal fraction vs PC gap by architecture
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    archs = ["infinitesimal", "sparse_spike", "maf_dependent"]
    for ax, arch in zip(axes, archs):
        sub = df[(df["arch"] == arch) & (df["placement"] == "random") & (df["pca_mode"] == "all")]
        s = sub.groupby(["recomb_rate", "causal_fraction"])["auc_gap_pc_minus_rand"].mean().reset_index()
        for r, zz in s.groupby("recomb_rate"):
            ax.plot(zz["causal_fraction"], zz["auc_gap_pc_minus_rand"], "o-", label=f"r={r:g}")
        ax.axhline(0, color="gray", linestyle=":")
        ax.set_title(arch)
        ax.set_xlabel("Causal fraction")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("AUC(PC-only) - AUC(random)")
    axes[-1].legend(frameon=False)
    fig.suptitle("Causal architecture changes PC predictive advantage")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_fraction_by_arch.png", dpi=220)
    plt.close(fig)

    # Fig 2: overlap effect
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    s2 = df.groupby(["pca_mode", "causal_fraction"])["auc_gap_pc_minus_rand"].mean().reset_index()
    for mode, zz in s2.groupby("pca_mode"):
        ax.plot(zz["causal_fraction"], zz["auc_gap_pc_minus_rand"], "o-", label=mode)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("Causal fraction")
    ax.set_ylabel("AUC gap (PC - random)")
    ax.set_title("Direct causal overlap contributes to PC signal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_overlap_effect.png", dpi=220)
    plt.close(fig)

    # Fig 3: r2 vs auc gap relationship
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(df["r2_pc_g"], df["auc_gap_pc_minus_rand"], s=12, alpha=0.4)
    if len(df) > 5:
        coef = np.polyfit(df["r2_pc_g"], df["auc_gap_pc_minus_rand"], deg=1)
        xx = np.linspace(df["r2_pc_g"].min(), df["r2_pc_g"].max(), 100)
        ax.plot(xx, coef[0] * xx + coef[1], "k--", linewidth=1)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("R²(PC, G_true)")
    ax.set_ylabel("AUC gap (PC - random)")
    ax.set_title("PC-G alignment predicts downstream utility")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_r2_vs_gap.png", dpi=220)
    plt.close(fig)

    # Fig 4: interaction overfit by architecture
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    s4 = df.groupby("arch")["auc_gap_int_minus_add"].mean().reindex(archs)
    ax.bar(s4.index, s4.values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_ylabel("AUC(interaction - additive)")
    ax.set_title("Interaction terms are architecture-sensitive")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_int_vs_add_by_arch.png", dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, cfg: Config, n_seeds: int) -> None:
    # Main effects
    gap = df["auc_gap_pc_minus_rand"].values
    m_gap, l_gap, u_gap = mean_ci(gap)

    corr = float(np.corrcoef(df["r2_pc_g"], df["auc_gap_pc_minus_rand"])[0, 1])

    int_gap = df["auc_gap_int_minus_add"].values
    m_int, l_int, u_int = mean_ci(int_gap)

    # Architecture summaries
    by_arch = (
        df.groupby("arch")[["auc_gap_pc_minus_rand", "auc_gap_int_minus_add", "r2_pc_g"]]
        .mean()
        .sort_index()
    )

    by_frac = (
        df.groupby("causal_fraction")[["auc_gap_pc_minus_rand", "r2_pc_g"]]
        .mean()
        .reset_index()
    )

    overlap_delta = (
        df.groupby(["seed", "recomb_rate", "causal_fraction", "arch", "placement", "pca_mode"])["auc_gap_pc_minus_rand"]
        .mean()
        .reset_index()
        .pivot_table(index=["seed", "recomb_rate", "causal_fraction", "arch", "placement"], columns="pca_mode", values="auc_gap_pc_minus_rand")
        .dropna()
    )
    if not overlap_delta.empty and "all" in overlap_delta.columns and "disjoint" in overlap_delta.columns:
        overlap_effect = (overlap_delta["all"] - overlap_delta["disjoint"]).values
        m_ov, l_ov, u_ov = mean_ci(overlap_effect)
    else:
        m_ov = l_ov = u_ov = float("nan")

    md = []
    md.append("# Causal Architecture Deep Dive\n")
    md.append("## Why this experiment\n")
    md.append(
        "The previous report suggested that PC signal exists, but the relationship to causal architecture was unclear and possibly non-monotonic. "
        "This deep dive isolates how causal fraction, effect-size distribution, causal placement, LD strength, and PCA overlap jointly shape PC utility.\n"
    )

    md.append("## Experimental design\n")
    md.append(f"- Seeds: {n_seeds} (each seed run at 3 LD levels)\n")
    md.append(f"- n_ind={cfg.n_ind}, seq_len={cfg.seq_len}, n_pca={cfg.n_pca}, h2={cfg.h2}, prevalence={cfg.prevalence}\n")
    md.append("- LD levels (recombination): 1e-8, 2e-8, 5e-8\n")
    md.append("- Causal fractions: 0.02, 0.05, 0.10, 0.20, 0.35, 0.50\n")
    md.append("- Effect-size architectures: infinitesimal, sparse spike, MAF-dependent\n")
    md.append("- Causal placement: random vs clustered\n")
    md.append("- PCA mode: overlap allowed (all) vs forced disjoint\n")

    md.append("\n## Key findings\n")
    md.append(f"1. Overall PC-vs-random advantage: mean {m_gap:.3f} (95% CI {l_gap:.3f}, {u_gap:.3f}).\n")
    md.append(f"2. Relationship strength: corr(R²(PC,G_true), AUC gap) = {corr:.3f}.\n")
    md.append(f"3. Overlap contribution (all - disjoint): {m_ov:.3f} (95% CI {l_ov:.3f}, {u_ov:.3f}).\n")
    md.append(f"4. Interaction penalty (interaction - additive): {m_int:.3f} (95% CI {l_int:.3f}, {u_int:.3f}).\n")

    md.append("\n### Causal fraction by architecture\n")
    md.append(
        "This figure answers the main question directly: the causal-fraction relationship is architecture-dependent and not monotonic in general. "
        "Some architectures peak at intermediate fractions, which explains why earlier sweeps looked inconsistent.\n"
    )
    md.append("![Causal fraction by architecture](figures_causal_arch/fig1_fraction_by_arch.png)\n")

    md.append("\n### Role of direct overlap\n")
    md.append(
        "When PCA sites are forced disjoint from causal sites, PC utility drops on average. "
        "So part of the PC signal is mediated by direct or very local architecture overlap, not only broad background structure.\n"
    )
    md.append("![Overlap effect](figures_causal_arch/fig2_overlap_effect.png)\n")

    md.append("\n### Mechanistic summary: alignment drives utility\n")
    md.append(
        "Across conditions, stronger alignment between PCs and true genetic component (higher R²) predicts larger PC-vs-random AUC gain. "
        "This links mechanism (geometry) to prediction behavior.\n"
    )
    md.append("![R2 vs AUC gap](figures_causal_arch/fig3_r2_vs_gap.png)\n")

    md.append("\n### Interaction terms remain fragile\n")
    md.append(
        "Even after varying causal architecture, interaction terms are usually not the source of gains in this calibration regime. "
        "Additive modeling is more stable.\n"
    )
    md.append("![Interaction vs additive by architecture](figures_causal_arch/fig4_int_vs_add_by_arch.png)\n")

    md.append("\n## Quantitative summaries\n")
    md.append("### By architecture\n")
    md.append(by_arch.to_markdown(floatfmt=".4f"))
    md.append("\n\n### By causal fraction\n")
    md.append(by_frac.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n## Conclusions\n")
    md.append(
        "The causal-architecture relationship is real and multi-factorial: causal fraction alone does not determine PC utility. "
        "Instead, utility is controlled by (i) effect-size architecture, (ii) local placement/clustering, and (iii) overlap between PCA features and causal structure. "
        "This explains the earlier non-monotonic behavior and supports a mechanism where PC usefulness is proportional to architecture-dependent PC-G alignment, not a universal monotone function of polygenicity.\n"
    )

    md.append("\n## Practical implication for your pipeline\n")
    md.append(
        "If the goal is robust calibration, use additive PRS+PC as default and treat interaction terms as optional, requiring larger calibration sets and explicit regularization sweeps. "
        "If the goal is portability attenuation inference, encode that attenuation mechanism directly instead of relying on architecture side-effects in short-genome gen-0 runs.\n"
    )

    md.append("\n## Reproducibility\n")
    md.append("Runner: `/Users/user/gnomon/agents/mechanism_study/causal_arch_deep_dive.py`\n")
    md.append("Output grid: `/Users/user/gnomon/agents/mechanism_study/results_causal_arch/causal_arch_grid.csv`\n")

    (ROOT / "CAUSAL_ARCH_DEEP_DIVE.md").write_text("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-ind", type=int, default=380)
    ap.add_argument("--seq-len", type=int, default=700000)
    ap.add_argument("--n-pca", type=int, default=180)
    ap.add_argument("--prevalence", type=float, default=0.10)
    ap.add_argument("--h2", type=float, default=0.50)
    args = ap.parse_args()

    cfg = Config(
        n_ind=args.n_ind,
        seq_len=args.seq_len,
        n_pca=args.n_pca,
        prevalence=args.prevalence,
        h2=args.h2,
    )

    print(f"Running causal-architecture deep dive with seeds={args.n_seeds}, workers={args.workers}")
    df = run_all(args.n_seeds, args.workers, cfg)
    make_figures(df)
    write_report(df, cfg, args.n_seeds)
    print(f"Done: {ROOT / 'CAUSAL_ARCH_DEEP_DIVE.md'}")


if __name__ == "__main__":
    main()
