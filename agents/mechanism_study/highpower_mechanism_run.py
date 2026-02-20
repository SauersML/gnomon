#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

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
RES = ROOT / "results_highpower"
FIG = ROOT / "figures_highpower"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    n_ind: int
    seq_len: int
    n_pca: int
    n_causal_base: int
    n_perm: int
    cal_sizes: List[int]


class Sim:
    def __init__(self, X, pos, maf, train, test):
        self.X = X
        self.pos = pos
        self.maf = maf
        self.train = train
        self.test = test


def std(v: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(v.reshape(-1, 1)).ravel()


def mean_ci(x: np.ndarray):
    x = np.asarray(x, float)
    m = float(np.mean(x))
    if len(x) < 2:
        return m, m, m
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se


def simulate_X(seed: int, recomb_rate: float, n_ind: int, seq_len: int) -> Sim:
    ts = msprime.sim_ancestry(
        samples=[msprime.SampleSet(n_ind, ploidy=2)],
        sequence_length=seq_len,
        recombination_rate=recomb_rate,
        ploidy=2,
        population_size=10_000,
        random_seed=seed,
        model="dtwf",
    )
    ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=seed + 1)

    m = ts.num_sites
    if m < 200:
        raise RuntimeError(f"Too few sites {m}")

    X = np.empty((n_ind, m), dtype=np.int8)
    pos = np.empty(m, dtype=np.float64)
    maf = np.empty(m, dtype=np.float64)

    for j, var in enumerate(ts.variants()):
        gh = var.genotypes.astype(np.int16)
        gd = gh[0::2] + gh[1::2]
        X[:, j] = gd.astype(np.int8)
        pos[j] = ts.site(var.site.id).position
        p = gd.mean() / 2.0
        maf[j] = min(p, 1.0 - p)

    idx = np.arange(n_ind)
    tr, te = train_test_split(idx, test_size=0.5, random_state=42)
    return Sim(X, pos, maf, tr, te)


def build_trait(sim: Sim, seed: int, n_causal: int, h2: float = 0.5, prevalence: float = 0.1):
    rng = np.random.default_rng(seed)
    elig = np.where(sim.maf >= 0.01)[0]
    cidx = rng.choice(elig, size=min(n_causal, len(elig)), replace=False)
    betas = rng.normal(0.0, math.sqrt(h2 / len(cidx)), size=len(cidx))
    g = std(sim.X[:, cidx].astype(float) @ betas)
    b0 = brentq(lambda b: expit(b + g).mean() - prevalence, -20, 20)
    y = rng.binomial(1, expit(b0 + g)).astype(np.int8)
    return g, y, cidx


def pca_sites(sim: Sim, seed: int, n_pca: int, mode: str, cidx: np.ndarray, buffer_bp: int = 50_000):
    rng = np.random.default_rng(seed)
    elig = np.where(sim.maf >= 0.05)[0]
    if mode == "all":
        pool = elig
    elif mode in {"disjoint", "disjoint_buffer"}:
        cset = set(cidx.tolist())
        mask = np.array([i not in cset for i in elig], dtype=bool)
        pool = elig[mask]
        if mode == "disjoint_buffer":
            cpos = sim.pos[cidx]
            keep = []
            for i in pool:
                if np.min(np.abs(cpos - sim.pos[i])) > buffer_bp:
                    keep.append(i)
            pool = np.asarray(keep, dtype=int)
            if len(pool) < 30:
                pool = elig[mask]
    else:
        raise ValueError(mode)

    if len(pool) < 30:
        raise RuntimeError(f"Not enough pca sites mode={mode}")
    return rng.choice(pool, size=min(n_pca, len(pool)), replace=False)


def pcs_from_sites(sim: Sim, pidx: np.ndarray, n_pc: int = 5):
    Xp = sim.X[:, pidx].astype(float)
    Xp = StandardScaler().fit_transform(Xp)
    Xp = np.nan_to_num(Xp)
    pca = PCA(n_components=min(20, Xp.shape[0] - 1, Xp.shape[1]), random_state=0)
    pcs = pca.fit_transform(Xp)
    pcs = StandardScaler().fit_transform(pcs)
    return pcs[:, :n_pc]


def pcs_train_only(sim: Sim, pidx: np.ndarray, n_pc: int = 5):
    Xp = sim.X[:, pidx].astype(float)
    tr, te = sim.train, sim.test
    mu = Xp[tr].mean(axis=0)
    sd = Xp[tr].std(axis=0)
    sd[sd == 0] = 1.0
    Xtr = np.nan_to_num((Xp[tr] - mu) / sd)
    Xte = np.nan_to_num((Xp[te] - mu) / sd)
    pca = PCA(n_components=min(20, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=0)
    trs = pca.fit_transform(Xtr)
    tes = pca.transform(Xte)
    alls = np.zeros((Xp.shape[0], trs.shape[1]))
    alls[tr] = trs
    alls[te] = tes
    alls = StandardScaler().fit_transform(alls)
    return alls[:, :n_pc]


def weak_prs(sim: Sim, y: np.ndarray):
    Xtr = sim.X[sim.train].astype(float)
    Xte = sim.X[sim.test].astype(float)
    ytr = y[sim.train].astype(float)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    Xt = (Xtr - mu) / sd
    Xv = (Xte - mu) / sd
    b = Xt.T @ (ytr - ytr.mean()) / len(ytr)
    return std(Xv @ b)


def fit_auc(y, X, cal, val, C=1.0):
    m = LogisticRegression(max_iter=3000, C=C)
    m.fit(X[cal], y[cal])
    p = m.predict_proba(X[val])[:, 1]
    return float(roc_auc_score(y[val], p))


def run_seed(seed: int, cfg: Config) -> Dict[str, pd.DataFrame]:
    rows_ld, rows_overlap, rows_leak, rows_complex, rows_perm, rows_causal = [], [], [], [], [], []

    # H1 LD strength
    for r in [1e-8, 2e-8, 5e-8]:
        sim = simulate_X(10000 + 100 * seed + int(r * 1e10), r, cfg.n_ind, cfg.seq_len)
        g, y, cidx = build_trait(sim, 20000 + seed, cfg.n_causal_base)
        pidx = pca_sites(sim, 30000 + seed, cfg.n_pca, "all", cidx)
        pcs = pcs_from_sites(sim, pidx)

        te = sim.test
        yt = y[te]
        gt = g[te]
        pct = pcs[te]
        rand = StandardScaler().fit_transform(np.random.default_rng(40000 + seed).normal(size=pct.shape))
        idx = np.arange(len(te))
        cal, val = train_test_split(idx, test_size=0.5, random_state=42)

        rows_ld.append({
            "seed": seed,
            "recomb_rate": r,
            "r2_pc_g": float(LinearRegression().fit(pct, gt).score(pct, gt)),
            "r2_rand_g": float(LinearRegression().fit(rand, gt).score(rand, gt)),
            "auc_pc_only": fit_auc(yt, pct, cal, val),
            "auc_rand_only": fit_auc(yt, rand, cal, val),
        })

    # Baseline sim reused for remaining hypotheses
    sim = simulate_X(50000 + seed, 2e-8, cfg.n_ind, cfg.seq_len)
    g, y, cidx = build_trait(sim, 51000 + seed, cfg.n_causal_base)

    # H2 overlap controls
    te = sim.test
    yt = y[te]
    gt = g[te]
    idx = np.arange(len(te))
    cal, val = train_test_split(idx, test_size=0.5, random_state=42)

    for mode in ["all", "disjoint", "disjoint_buffer"]:
        pidx = pca_sites(sim, 52000 + seed, cfg.n_pca, mode, cidx)
        pcs = pcs_from_sites(sim, pidx)
        pct = pcs[te]
        ov = len(set(pidx.tolist()) & set(cidx.tolist()))
        rows_overlap.append({
            "seed": seed,
            "mode": mode,
            "pct_overlap": 100.0 * ov / len(pidx),
            "r2_pc_g": float(LinearRegression().fit(pct, gt).score(pct, gt)),
            "auc_pc_only": fit_auc(yt, pct, cal, val),
        })

    # H3 leakage control
    pidx = pca_sites(sim, 53000 + seed, cfg.n_pca, "all", cidx)
    pc_all = pcs_from_sites(sim, pidx)[te]
    pc_tr = pcs_train_only(sim, pidx)[te]
    rows_leak.append({
        "seed": seed,
        "r2_allpc_g": float(LinearRegression().fit(pc_all, gt).score(pc_all, gt)),
        "r2_trainpc_g": float(LinearRegression().fit(pc_tr, gt).score(pc_tr, gt)),
        "auc_allpc": fit_auc(yt, pc_all, cal, val),
        "auc_trainpc": fit_auc(yt, pc_tr, cal, val),
    })

    # H4 model complexity
    prs_w = weak_prs(sim, y)
    prs_s = std(gt + np.random.default_rng(54000 + seed).normal(0, 0.25, len(gt)))
    prs_n = std(np.random.default_rng(55000 + seed).normal(size=len(gt)))
    Xpc = pc_all

    for cal_n in cfg.cal_sizes:
        sub_cal, rem = train_test_split(idx, train_size=min(cal_n, len(idx)//2), random_state=42)
        if len(rem) < 30:
            continue
        sub_val = rem
        for prs_name, prs in [("noise", prs_n), ("weak", prs_w), ("strong", prs_s)]:
            Xraw = prs.reshape(-1, 1)
            Xadd = np.hstack([Xraw, Xpc])
            Xint = np.hstack([Xraw, Xpc, Xraw * Xpc])
            rows_complex.append({
                "seed": seed,
                "cal_n": len(sub_cal),
                "prs_type": prs_name,
                "auc_raw": fit_auc(yt, Xraw, sub_cal, sub_val),
                "auc_add": fit_auc(yt, Xadd, sub_cal, sub_val),
                "auc_int": fit_auc(yt, Xint, sub_cal, sub_val),
            })

    # H5 permutation null (pc-only)
    auc_real = fit_auc(yt, Xpc, cal, val)
    rng = np.random.default_rng(56000 + seed)
    nulls = []
    for _ in range(cfg.n_perm):
        yp = yt.copy()
        rng.shuffle(yp)
        nulls.append(fit_auc(yp, Xpc, cal, val))
    nulls = np.asarray(nulls)
    p_emp = float((np.sum(nulls >= auc_real) + 1) / (len(nulls) + 1))
    rows_perm.append({
        "seed": seed,
        "auc_real_pc": auc_real,
        "auc_null_mean": float(nulls.mean()),
        "auc_null_sd": float(nulls.std(ddof=1)),
        "p_empirical": p_emp,
    })

    # H6 % causal sweep
    elig = np.where(sim.maf >= 0.01)[0]
    for frac in [0.05, 0.1, 0.2, 0.4]:
        n_c = max(40, int(len(elig) * frac))
        g2, y2, c2 = build_trait(sim, 57000 + seed + int(frac * 1000), n_c)
        p2 = pca_sites(sim, 58000 + seed + int(frac * 1000), cfg.n_pca, "all", c2)
        pc2 = pcs_from_sites(sim, p2)[te]
        yt2 = y2[te]
        gt2 = g2[te]
        rand2 = StandardScaler().fit_transform(np.random.default_rng(59000 + seed + int(frac * 1000)).normal(size=pc2.shape))
        rows_causal.append({
            "seed": seed,
            "causal_fraction": frac,
            "n_causal": n_c,
            "r2_pc_g": float(LinearRegression().fit(pc2, gt2).score(pc2, gt2)),
            "auc_pc_only": fit_auc(yt2, pc2, cal, val),
            "auc_rand_only": fit_auc(yt2, rand2, cal, val),
        })

    return {
        "h1": pd.DataFrame(rows_ld),
        "h2": pd.DataFrame(rows_overlap),
        "h3": pd.DataFrame(rows_leak),
        "h4": pd.DataFrame(rows_complex),
        "h5": pd.DataFrame(rows_perm),
        "h6": pd.DataFrame(rows_causal),
    }


def combine(seed_results: List[Dict[str, pd.DataFrame]]):
    out = {}
    for key in ["h1", "h2", "h3", "h4", "h5", "h6"]:
        out[key] = pd.concat([r[key] for r in seed_results], ignore_index=True)
        out[key].to_csv(RES / f"{key}.csv", index=False)
    return out


def make_figures(df: Dict[str, pd.DataFrame]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False})

    h1 = df["h1"]
    s = h1.groupby("recomb_rate")[["r2_pc_g", "auc_pc_only", "auc_rand_only"]].mean().reset_index()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(s["recomb_rate"], s["r2_pc_g"], "o-")
    ax1.set_xscale("log")
    ax1.set_xlabel("Recombination rate")
    ax1.set_ylabel("R²(PC, G_true)")
    ax1.set_title("H1 LD strength")
    ax2.plot(s["recomb_rate"], s["auc_pc_only"], "o-", label="PC-only")
    ax2.plot(s["recomb_rate"], s["auc_rand_only"], "s--", label="Random")
    ax2.axhline(0.5, color="gray", ls=":")
    ax2.set_xscale("log")
    ax2.set_xlabel("Recombination rate")
    ax2.set_ylabel("AUC")
    ax2.legend(frameon=False)
    ax2.set_title("Held-out predictive control")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_ld.png", dpi=220)
    plt.close(fig)

    h2 = df["h2"]
    s2 = h2.groupby("mode")[["pct_overlap", "r2_pc_g", "auc_pc_only"]].mean().reindex(["all", "disjoint", "disjoint_buffer"])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(s2.index, s2["auc_pc_only"], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("AUC(PC-only)")
    ax.set_title("H2 overlap control")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_overlap.png", dpi=220)
    plt.close(fig)

    h4 = df["h4"]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    s4 = h4.groupby(["cal_n", "prs_type"])[["auc_add", "auc_int"]].mean().reset_index()
    for prs in ["noise", "weak", "strong"]:
        z = s4[s4["prs_type"] == prs]
        ax.plot(z["cal_n"], z["auc_int"] - z["auc_add"], "o-", label=prs)
    ax.axhline(0, color="gray", ls=":")
    ax.set_xlabel("Calibration sample size")
    ax.set_ylabel("AUC(int - add)")
    ax.set_title("H4 interaction overfit test")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_complexity.png", dpi=220)
    plt.close(fig)

    h6 = df["h6"]
    s6 = h6.groupby("causal_fraction")[["auc_pc_only", "auc_rand_only"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(s6["causal_fraction"], s6["auc_pc_only"], "o-", label="PC-only")
    ax.plot(s6["causal_fraction"], s6["auc_rand_only"], "s--", label="Random")
    ax.axhline(0.5, color="gray", ls=":")
    ax.set_xlabel("Causal fraction")
    ax.set_ylabel("AUC")
    ax.set_title("H6 causal architecture sweep")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_causal_fraction.png", dpi=220)
    plt.close(fig)


def write_report(df: Dict[str, pd.DataFrame], cfg: Config, seeds: List[int]):
    h1, h2, h3, h4, h5, h6 = df["h1"], df["h2"], df["h3"], df["h4"], df["h5"], df["h6"]

    gap_pc_rand = h1["auc_pc_only"] - h1["auc_rand_only"]
    m1, l1, u1 = mean_ci(gap_pc_rand.values)

    leak = h3["auc_allpc"] - h3["auc_trainpc"]
    m2, l2, u2 = mean_ci(leak.values)

    int_minus_add = h4["auc_int"] - h4["auc_add"]
    m3, l3, u3 = mean_ci(int_minus_add.values)

    frac_sig = float((h5["p_empirical"] < 0.05).mean())

    rep = []
    rep.append("# Final Mechanism Report (High-Power Fast Run)\n")
    rep.append("## Methods\n")
    rep.append(f"Seeds: {seeds}\n")
    rep.append(f"n_ind={cfg.n_ind}, seq_len={cfg.seq_len}, n_pca={cfg.n_pca}, baseline n_causal={cfg.n_causal_base}, perms/seed={cfg.n_perm}\n")
    rep.append("Hypotheses: H1 LD dose, H2 overlap, H3 leakage control, H4 model complexity, H5 permutation null, H6 causal-fraction sweep.\n")

    rep.append("\n## Results\n")
    rep.append(f"- H1 PC-only minus random AUC: mean **{m1:.3f}** (95% CI {l1:.3f}, {u1:.3f})\n")
    rep.append(f"- H3 leakage delta (all-PC minus train-PC): mean **{m2:.3f}** (95% CI {l2:.3f}, {u2:.3f})\n")
    rep.append(f"- H4 interaction minus additive: mean **{m3:.3f}** (95% CI {l3:.3f}, {u3:.3f})\n")
    rep.append(f"- H5 empirical p<0.05 fraction: **{frac_sig:.2f}**\n")

    rep.append("\n### H1 LD Strength\n")
    rep.append(h1.groupby("recomb_rate")[["r2_pc_g", "auc_pc_only", "auc_rand_only"]].mean().to_markdown(floatfmt=".4f"))

    rep.append("\n\n### H2 Overlap Controls\n")
    rep.append(h2.groupby("mode")[["pct_overlap", "r2_pc_g", "auc_pc_only"]].mean().to_markdown(floatfmt=".4f"))

    rep.append("\n\n### H3 Leakage\n")
    rep.append(h3[["r2_allpc_g", "r2_trainpc_g", "auc_allpc", "auc_trainpc"]].mean().to_frame("mean").to_markdown(floatfmt=".4f"))

    rep.append("\n\n### H4 Complexity\n")
    rep.append(h4.groupby(["cal_n", "prs_type"])[["auc_raw", "auc_add", "auc_int"]].mean().to_markdown(floatfmt=".4f"))

    rep.append("\n\n### H5 Permutation\n")
    rep.append(h5[["auc_real_pc", "auc_null_mean", "p_empirical"]].to_markdown(index=False, floatfmt=".4f"))

    rep.append("\n\n### H6 Causal Fraction\n")
    rep.append(h6.groupby("causal_fraction")[["r2_pc_g", "auc_pc_only", "auc_rand_only"]].mean().to_markdown(floatfmt=".4f"))

    rep.append("\n\n## Figures\n")
    rep.append("![H1 LD](figures_highpower/fig1_ld.png)\n")
    rep.append("![H2 overlap](figures_highpower/fig2_overlap.png)\n")
    rep.append("![H4 complexity](figures_highpower/fig3_complexity.png)\n")
    rep.append("![H6 causal fraction](figures_highpower/fig4_causal_fraction.png)\n")

    rep.append("\n## Conclusions\n")
    rep.append("PC features can carry real held-out signal under LD structure, but interaction-heavy calibrators are fragile and can underperform additive forms at limited calibration size. Leakage controls (train-only PCs) are included and should be interpreted alongside permutation-null evidence.\n")

    (ROOT / "FINAL_MECHANISM_REPORT.md").write_text("\n".join(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--n-ind", type=int, default=380)
    ap.add_argument("--seq-len", type=int, default=700_000)
    ap.add_argument("--n-pca", type=int, default=180)
    ap.add_argument("--n-causal", type=int, default=260)
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()

    cfg = Config(
        n_ind=args.n_ind,
        seq_len=args.seq_len,
        n_pca=args.n_pca,
        n_causal_base=args.n_causal,
        n_perm=args.n_perm,
        cal_sizes=[80, 120],
    )

    seeds = list(range(1, args.n_seeds + 1))
    print(f"Running high-power mechanism run: seeds={seeds}, workers={args.workers}")

    results = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_seed, s, cfg): s for s in seeds}
        for fut in cf.as_completed(futs):
            s = futs[fut]
            print(f"  finished seed {s}")
            results.append(fut.result())

    all_df = combine(results)
    make_figures(all_df)
    write_report(all_df, cfg, seeds)
    print(f"Done: {ROOT / 'FINAL_MECHANISM_REPORT.md'}")


if __name__ == "__main__":
    main()
