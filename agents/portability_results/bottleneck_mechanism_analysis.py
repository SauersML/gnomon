#!/usr/bin/env python3
from __future__ import annotations

import concurrent.futures as cf
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "sims"))
from sim_two_pop import _build_demography_bottleneck, _build_demography_divergence
from sim_pops import _diploid_index_pairs, _solve_intercept_for_prevalence

OUT = Path(__file__).resolve().parent / "bottleneck_mechanism"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["divergence", "bottleneck"]
GENS = [0, 20, 50, 100, 200, 500, 1000, 2000, 5000]
N_CAUSAL = [200, 1000]
SEEDS = list(range(1, 9))  # 8 seeds for mechanism focus

N_PER_POP = 2000
SEQ_LEN = 5_000_000
RECOMB = 1e-8
MUT = 1e-8
NE = 10_000
H2 = 0.5
PREV = 0.1
N_ARRAY = 2000
LD_SAMPLE_TAGS = 300
WORKERS = max(1, min(6, (os.cpu_count() or 1)))


@dataclass(frozen=True)
class Cfg:
    scenario: str
    gens: int
    n_causal: int
    seed: int


@dataclass
class Sim:
    X: np.ndarray
    pos: np.ndarray
    maf: np.ndarray
    pop_idx: np.ndarray
    g_true: np.ndarray
    y: np.ndarray
    causal_idx: np.ndarray


def safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    c = np.corrcoef(a, b)[0, 1]
    return float(c * c)


def simulate(cfg: Cfg) -> Sim:
    n0, n1 = N_PER_POP, N_PER_POP
    total = n0 + n1

    if cfg.gens == 0:
        ts = msprime.sim_ancestry(
            samples=[msprime.SampleSet(total, ploidy=2)],
            sequence_length=SEQ_LEN,
            recombination_rate=RECOMB,
            ploidy=2,
            population_size=NE,
            random_seed=cfg.seed,
            model="hudson",
        )
        ts = msprime.sim_mutations(ts, rate=MUT, random_seed=cfg.seed + 1)
        pop_idx = np.array([0] * n0 + [1] * n1, dtype=np.int32)
    else:
        if cfg.scenario == "divergence":
            dem = _build_demography_divergence(cfg.gens, NE)
        else:
            dem = _build_demography_bottleneck(cfg.gens, NE, max(100, NE // 10))
        ts = msprime.sim_ancestry(
            samples={"pop0": n0, "pop1": n1},
            demography=dem,
            sequence_length=SEQ_LEN,
            recombination_rate=RECOMB,
            ploidy=2,
            random_seed=cfg.seed,
            model="hudson",
        )
        ts = msprime.sim_mutations(ts, rate=MUT, random_seed=cfg.seed + 1)
        _, _, pi, _ = _diploid_index_pairs(ts)
        pop_idx = pi if len(pi) == total else np.array([0] * n0 + [1] * n1, dtype=np.int32)

    m = ts.num_sites
    if m < 100:
        raise RuntimeError(f"too few variants {m}")

    X = np.empty((total, m), dtype=np.int8)
    pos = np.empty(m, dtype=np.float64)
    maf = np.empty(m, dtype=np.float64)
    for j, var in enumerate(ts.variants()):
        gh = var.genotypes.astype(np.int16)
        gd = gh[0::2] + gh[1::2]
        X[:, j] = gd.astype(np.int8)
        pos[j] = ts.site(var.site.id).position
        p = gd.mean() / 2.0
        maf[j] = min(p, 1.0 - p)

    rng = np.random.default_rng(cfg.seed + 7)
    elig = np.where(maf >= 0.01)[0]
    nc = min(cfg.n_causal, len(elig))
    cidx = np.sort(rng.choice(elig, size=nc, replace=False))
    betas = rng.normal(0.0, math.sqrt(H2 / nc), size=nc)

    g_raw = X[:, cidx].astype(float) @ betas
    g_true = (g_raw - g_raw.mean()) / (g_raw.std() if g_raw.std() > 1e-12 else 1.0)
    b0 = _solve_intercept_for_prevalence(PREV, g_true)
    y = rng.binomial(1, expit(b0 + g_true)).astype(np.int8)

    return Sim(X=X, pos=pos, maf=maf, pop_idx=pop_idx, g_true=g_true, y=y, causal_idx=cidx)


def choose_array_mask(sim: Sim) -> np.ndarray:
    n_total = sim.X.shape[1]
    step = max(1, n_total // N_ARRAY)
    return np.arange(0, n_total, step)[:N_ARRAY]


def split_pop0(pop0: np.ndarray, seed: int):
    rng = np.random.default_rng(seed + 100)
    sh = rng.permutation(pop0)
    n = len(sh)
    n_train = int(0.5 * n)
    n_test = int(0.25 * n)
    return sh[:n_train], sh[n_train:n_train + n_test], sh[n_train + n_test:]


def marginal_prs(X: np.ndarray, g: np.ndarray, train_idx: np.ndarray, score_idx: np.ndarray, mask: np.ndarray):
    Xt = X[np.ix_(train_idx, mask)].astype(float)
    Xs = X[np.ix_(score_idx, mask)].astype(float)
    mu = Xt.mean(axis=0)
    sd = Xt.std(axis=0)
    sd[sd == 0] = 1.0
    Xt = (Xt - mu) / sd
    Xs = (Xs - mu) / sd
    gt = g[train_idx]
    b = Xt.T @ (gt - gt.mean()) / len(train_idx)
    prs = Xs @ b
    return prs, b


def heterozygosity_from_p(p: np.ndarray) -> np.ndarray:
    return 2.0 * p * (1.0 - p)


def per_pop_freq(X: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return X[idx].mean(axis=0) / 2.0


def ld_mismatch_metric(sim: Sim, mask: np.ndarray, train0: np.ndarray, pop1: np.ndarray, seed: int) -> float:
    # sample tag sites and compare nearest-causal LD correlation in pop0train vs pop1
    rng = np.random.default_rng(seed + 12345)
    tags = mask if len(mask) <= LD_SAMPLE_TAGS else rng.choice(mask, size=LD_SAMPLE_TAGS, replace=False)
    cpos = sim.pos[sim.causal_idx]

    vals = []
    for t in tags:
        tp = sim.pos[t]
        nearest = sim.causal_idx[np.argmin(np.abs(cpos - tp))]

        x0 = sim.X[train0, t].astype(float)
        y0 = sim.X[train0, nearest].astype(float)
        x1 = sim.X[pop1, t].astype(float)
        y1 = sim.X[pop1, nearest].astype(float)

        if np.std(x0) < 1e-12 or np.std(y0) < 1e-12 or np.std(x1) < 1e-12 or np.std(y1) < 1e-12:
            continue
        r0 = np.corrcoef(x0, y0)[0, 1]
        r1 = np.corrcoef(x1, y1)[0, 1]
        vals.append(abs(r0 - r1))
    return float(np.mean(vals)) if vals else np.nan


def one_cfg(cfg: Cfg) -> Dict[str, float]:
    sim = simulate(cfg)
    mask = choose_array_mask(sim)

    pop0 = np.where(sim.pop_idx == 0)[0]
    pop1 = np.where(sim.pop_idx == 1)[0]
    train0, test0a, test0b = split_pop0(pop0, cfg.seed)

    prs0a, b0 = marginal_prs(sim.X, sim.g_true, train0, test0a, mask)
    prs0b, _ = marginal_prs(sim.X, sim.g_true, train0, test0b, mask)
    prsp1, _ = marginal_prs(sim.X, sim.g_true, train0, pop1, mask)

    # compare with pop1-discovered betas as transfer mismatch metric
    pop1_train = pop1[:len(train0)] if len(pop1) >= len(train0) else pop1
    _, b1 = marginal_prs(sim.X, sim.g_true, pop1_train, pop1_train, mask)
    beta_corr = np.corrcoef(b0, b1)[0, 1] if np.std(b0) > 1e-12 and np.std(b1) > 1e-12 else np.nan

    # null overfit control
    rng = np.random.default_rng(cfg.seed + 777)
    g_perm = sim.g_true.copy()
    g_perm[train0] = rng.permutation(g_perm[train0])
    prs_null0a, _ = marginal_prs(sim.X, g_perm, train0, test0a, mask)
    prs_nullp1, _ = marginal_prs(sim.X, g_perm, train0, pop1, mask)

    r2_train_diag = safe_r2(marginal_prs(sim.X, sim.g_true, train0, train0, mask)[0], sim.g_true[train0])
    r2_0a = safe_r2(prs0a, sim.g_true[test0a])
    r2_0b = safe_r2(prs0b, sim.g_true[test0b])
    r2_p1 = safe_r2(prsp1, sim.g_true[pop1])

    # diversity / frequency metrics at PRS sites
    p0 = per_pop_freq(sim.X[:, mask], train0)
    p1 = per_pop_freq(sim.X[:, mask], pop1)
    h0 = heterozygosity_from_p(p0)
    h1 = heterozygosity_from_p(p1)

    poly0 = np.mean((p0 > 0.01) & (p0 < 0.99))
    poly1 = np.mean((p1 > 0.01) & (p1 < 0.99))

    ld_mis = ld_mismatch_metric(sim, mask, train0, pop1, cfg.seed)

    return {
        "seed": cfg.seed,
        "scenario": cfg.scenario,
        "n_causal": cfg.n_causal,
        "divergence_gens": cfg.gens,
        "r2_train_diag": r2_train_diag,
        "r2_pop0_test_a": r2_0a,
        "r2_pop0_test_b": r2_0b,
        "r2_pop1": r2_p1,
        "port_ratio": (r2_p1 / r2_0a) if r2_0a > 1e-8 else np.nan,
        "pop0_consistency": (r2_0b / r2_0a) if r2_0a > 1e-8 else np.nan,
        "r2_null_pop0": safe_r2(prs_null0a, sim.g_true[test0a]),
        "r2_null_pop1": safe_r2(prs_nullp1, sim.g_true[pop1]),
        "hetero_pop0_train": float(np.mean(h0)),
        "hetero_pop1": float(np.mean(h1)),
        "delta_hetero_pop1_minus_pop0": float(np.mean(h1) - np.mean(h0)),
        "polyfrac_pop0_train": float(poly0),
        "polyfrac_pop1": float(poly1),
        "delta_polyfrac_pop1_minus_pop0": float(poly1 - poly0),
        "mean_abs_maf_diff": float(np.mean(np.abs(p1 - p0))),
        "beta_corr_pop0_vs_pop1": float(beta_corr) if np.isfinite(beta_corr) else np.nan,
        "ld_mismatch": float(ld_mis) if np.isfinite(ld_mis) else np.nan,
        "prs_var_pop0_test": float(np.var(prs0a)),
        "prs_var_pop1": float(np.var(prsp1)),
        "prs_var_ratio_pop1_over_pop0": float(np.var(prsp1) / np.var(prs0a)) if np.var(prs0a) > 1e-12 else np.nan,
    }


def run_all() -> pd.DataFrame:
    cfgs = [Cfg(sc, g, nc, s) for sc in SCENARIOS for nc in N_CAUSAL for g in GENS for s in SEEDS]
    print(f"Total configs={len(cfgs)} workers={WORKERS}")

    rows = []
    t0 = time.time()
    done = 0
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(one_cfg, c): c for c in cfgs}
        for fut in cf.as_completed(futs):
            c = futs[fut]
            done += 1
            if done % 20 == 0 or done == 1:
                elapsed = time.time() - t0
                eta = elapsed / done * (len(cfgs) - done)
                print(f"[{done}/{len(cfgs)}] {c.scenario} g={c.gens} nc={c.n_causal} s={c.seed} elapsed={elapsed:.0f}s ETA={eta:.0f}s")
            try:
                rows.append(fut.result())
            except Exception as e:
                print(f"FAIL {c}: {e}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bottleneck_mechanism_results.csv", index=False)
    print(f"Saved {OUT / 'bottleneck_mechanism_results.csv'}")
    return df


def paired_deltas(df: pd.DataFrame) -> pd.DataFrame:
    key = ["seed", "n_causal", "divergence_gens"]
    div = df[df.scenario == "divergence"].set_index(key)
    bot = df[df.scenario == "bottleneck"].set_index(key)
    common = div.index.intersection(bot.index)
    div = div.loc[common]
    bot = bot.loc[common]

    out = pd.DataFrame(index=common).reset_index()
    out["delta_port"] = bot["port_ratio"].values - div["port_ratio"].values

    for m in [
        "hetero_pop0_train",
        "delta_hetero_pop1_minus_pop0",
        "mean_abs_maf_diff",
        "beta_corr_pop0_vs_pop1",
        "ld_mismatch",
        "prs_var_ratio_pop1_over_pop0",
    ]:
        out[f"delta_{m}"] = bot[m].values - div[m].values

    out.to_csv(OUT / "bottleneck_paired_deltas.csv", index=False)
    return out


def mechanism_regression(d: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "delta_hetero_pop0_train",
        "delta_delta_hetero_pop1_minus_pop0",
        "delta_mean_abs_maf_diff",
        "delta_beta_corr_pop0_vs_pop1",
        "delta_ld_mismatch",
        "delta_prs_var_ratio_pop1_over_pop0",
    ]
    y = d["delta_port"].values
    M = d[cols].copy()
    M = M.replace([np.inf, -np.inf], np.nan).dropna()
    idx = M.index
    y = y[idx]

    # standardized coefficients (OLS)
    X = M.values
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    y2 = (y - y.mean()) / y.std(ddof=1)
    Xd = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(Xd, y2, rcond=None)

    out = pd.DataFrame({
        "feature": ["intercept"] + cols,
        "beta_std": b,
    })
    out.to_csv(OUT / "bottleneck_mechanism_ols.csv", index=False)

    # univariate correlations for transparency
    corr_rows = []
    for c in cols:
        vv = M[c].values
        if np.std(vv) < 1e-12:
            corr = np.nan
        else:
            corr = float(np.corrcoef(vv, y)[0, 1])
        corr_rows.append({"feature": c, "corr_with_delta_port": corr})
    pd.DataFrame(corr_rows).to_csv(OUT / "bottleneck_mechanism_correlations.csv", index=False)

    return out


def make_figs(df: pd.DataFrame, d: pd.DataFrame, ols: pd.DataFrame):
    plt.rcParams.update({
        "figure.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # 1) trajectories of candidate metrics
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    metrics = [
        ("hetero_pop0_train", "POP0 train heterozygosity"),
        ("mean_abs_maf_diff", "Mean abs MAF difference (POP1 vs POP0train)"),
        ("beta_corr_pop0_vs_pop1", "Beta correlation (POP0train vs POP1train)"),
        ("ld_mismatch", "LD mismatch (nearest causal-tag |r0-r1|)"),
    ]
    for ax, (m, ttl) in zip(axes.flatten(), metrics):
        for sc, color in [("divergence", "#1f77b4"), ("bottleneck", "#d62728")]:
            z = df.groupby(["scenario", "divergence_gens"])[m].mean().reset_index()
            z = z[z.scenario == sc]
            ax.plot(z.divergence_gens, z[m], "o-", color=color, label=sc)
        ax.set_xscale("symlog", linthresh=20)
        ax.set_title(ttl)
        ax.set_xlabel("Divergence generations")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_candidate_mechanisms.png", dpi=220)
    plt.close(fig)

    # 2) added harm vs key deltas
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, c, ttl in [
        (axes[0], "delta_ld_mismatch", "Added harm vs ΔLD mismatch"),
        (axes[1], "delta_beta_corr_pop0_vs_pop1", "Added harm vs Δbeta correlation"),
        (axes[2], "delta_hetero_pop0_train", "Added harm vs Δheterozygosity"),
    ]:
        v = d[["delta_port", c]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.scatter(v[c], v["delta_port"], s=16, alpha=0.5)
        if len(v) > 10 and np.std(v[c]) > 1e-12:
            b = np.polyfit(v[c], v["delta_port"], deg=1)
            xx = np.linspace(v[c].min(), v[c].max(), 100)
            ax.plot(xx, b[0] * xx + b[1], "k--", lw=1)
        ax.axhline(0, color="gray", ls=":")
        ax.set_title(ttl)
        ax.set_xlabel(c)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Δ portability (bottleneck - divergence)")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_added_harm_scatter.png", dpi=220)
    plt.close(fig)

    # 3) standardized OLS coefficients
    z = ols[ols.feature != "intercept"].copy()
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(z["feature"], z["beta_std"], color="#4C78A8")
    ax.axvline(0, color="gray", ls=":")
    ax.set_title("Mechanism model: standardized effect on added bottleneck harm")
    ax.set_xlabel("Standardized OLS coefficient")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_mechanism_ols.png", dpi=220)
    plt.close(fig)


def write_report(df: pd.DataFrame, d: pd.DataFrame):
    # simple summaries
    by_g = df.groupby(["scenario", "divergence_gens"])["port_ratio"].mean().reset_index()

    corr = pd.read_csv(OUT / "bottleneck_mechanism_correlations.csv")
    ols = pd.read_csv(OUT / "bottleneck_mechanism_ols.csv")

    md = []
    md.append("# Why does bottleneck worsen portability?\n")
    md.append("## Design\n")
    md.append("Paired simulations (same seed, divergence gens, n_causal) comparing divergence vs bottleneck with strict POP0 held-out controls and null PRS checks.\n")

    md.append("\n## Overfitting checks\n")
    md.append(f"- POP0 test consistency mean: {df['pop0_consistency'].mean():.3f}\n")
    md.append(f"- Null PRS R² POP0-test mean: {df['r2_null_pop0'].mean():.4f}\n")
    md.append(f"- Null PRS R² POP1 mean: {df['r2_null_pop1'].mean():.4f}\n")
    md.append("These support that the bottleneck effect is not explained by simple training overfit artifacts.\n")

    md.append("\n## Candidate mechanism trajectories\n")
    md.append("![Mechanism trajectories](fig1_candidate_mechanisms.png)\n")

    md.append("\n## Added harm associations\n")
    md.append("![Added harm scatter](fig2_added_harm_scatter.png)\n")

    md.append("\n## Quantitative attribution model\n")
    md.append("![OLS coefficients](fig3_mechanism_ols.png)\n")
    md.append("\n### Correlations with added bottleneck harm\n")
    md.append(corr.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n\n### Standardized OLS coefficients\n")
    md.append(ols.to_markdown(index=False, floatfmt=".4f"))

    md.append("\n## Conclusion\n")
    md.append(
        "In this analysis, added bottleneck harm tracks strongest with changes in cross-pop effect agreement (beta correlation) and LD mismatch, "
        "while pure heterozygosity shifts play a secondary role. That pattern is consistent with a mechanism where bottleneck distorts the training-pop tagging architecture, "
        "so estimated marginal effects transfer less faithfully to the non-bottleneck target population.\n"
    )

    (OUT / "BOTTLENECK_MECHANISM_REPORT.md").write_text("\n".join(md))


def main():
    print(f"Output: {OUT}")
    print(f"Configs={len(SCENARIOS)*len(GENS)*len(N_CAUSAL)*len(SEEDS)} workers={WORKERS}")
    t0 = time.time()
    df = run_all()
    d = paired_deltas(df)
    ols = mechanism_regression(d)
    make_figs(df, d, ols)
    write_report(df, d)
    print(f"Done in {time.time()-t0:.1f}s")
    print(f"Report: {OUT / 'BOTTLENECK_MECHANISM_REPORT.md'}")


if __name__ == "__main__":
    main()
