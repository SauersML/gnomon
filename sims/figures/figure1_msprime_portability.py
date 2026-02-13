from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import msprime
from scipy.stats import gaussian_kde
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler

# Avoid noisy rpy2 API-mode import warnings on this environment.
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

THIS_DIR = Path(__file__).resolve().parent
SIMS_DIR = THIS_DIR.parent
if str(SIMS_DIR) not in sys.path:
    sys.path.append(str(SIMS_DIR))

try:
    from .common import (
        ensure_pgs003725,
        load_pgs003725_effects,
        diploid_index_pairs,
        sample_site_ids_for_maf,
        pcs_from_sites,
        genetic_risk_from_real_pgs_effect_distribution,
        solve_intercept_for_prevalence,
        get_chr22_recomb_map,
    )
except ImportError:
    from common import (
    ensure_pgs003725,
    load_pgs003725_effects,
    diploid_index_pairs,
    sample_site_ids_for_maf,
    pcs_from_sites,
    genetic_risk_from_real_pgs_effect_distribution,
    solve_intercept_for_prevalence,
    get_chr22_recomb_map,
    )
from methods.raw_pgs import RawPGSMethod
from methods.linear_interaction import LinearInteractionMethod
from methods.normalization import NormalizationMethod
from methods.gam_mgcv import GAMMethod
from plink_utils import run_plink_conversion
from prs_tools import BayesR


DIVERGENCE_GENS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Requested scale-down by 100 from original design.
N_AFR = 100
N_OOA_TRAIN = 100
N_OOA_TEST = 100
N_PCS = 5

CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "black": "#111111",
}


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#2a2a2a",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d4d9de",
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def _style_axes(ax) -> None:
    ax.grid(True, alpha=0.55)
    ax.set_axisbelow(True)


def _build_demography(split_gens: int) -> msprime.Demography:
    dem = msprime.Demography()
    dem.add_population(name="afr", initial_size=10_000)
    # OoA bottleneck at split followed by forward-time growth to present N=10k.
    growth_rate = math.log(10_000 / 2_000) / float(split_gens)
    dem.add_population(name="ooa", initial_size=10_000, growth_rate=growth_rate)
    dem.add_population(name="ancestral", initial_size=10_000)
    dem.add_population_split(time=split_gens, derived=["afr", "ooa"], ancestral="ancestral")
    dem.sort_events()
    return dem


def _simulate_for_generation(gens: int, seed: int, pgs_effects: np.ndarray, out_dir: Path) -> pd.DataFrame:
    prefix = f"fig1_g{gens}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    recomb_map = get_chr22_recomb_map()
    dem = _build_demography(gens)

    ts = msprime.sim_ancestry(
        samples={"afr": N_AFR, "ooa": N_OOA_TRAIN + N_OOA_TEST},
        demography=dem,
        recombination_rate=recomb_map,
        ploidy=2,
        random_seed=seed,
        model="dtwf",
    )
    ts = msprime.sim_mutations(
        ts,
        rate=1e-8,
        random_seed=seed + 1,
        discrete_genome=True,
    )

    a_idx, b_idx, pop_idx, ts_ind_id = diploid_index_pairs(ts)
    pop_map = {0: "AFR", 1: "OOA"}
    pop = np.array([pop_map.get(int(x), "UNK") for x in pop_idx], dtype=object)

    ooa_idx = np.where(pop == "OOA")[0]
    afr_idx = np.where(pop == "AFR")[0]
    if len(ooa_idx) < (N_OOA_TRAIN + N_OOA_TEST) or len(afr_idx) < N_AFR:
        raise RuntimeError("Unexpected sample counts after simulation")

    ooa_train_idx = ooa_idx[:N_OOA_TRAIN]
    ooa_test_idx = ooa_idx[N_OOA_TRAIN:N_OOA_TRAIN + N_OOA_TEST]

    pca_sites = sample_site_ids_for_maf(ts, a_idx, b_idx, n_sites=2000, maf_min=0.05, seed=seed + 7)
    pcs = pcs_from_sites(ts, a_idx, b_idx, pca_sites, seed=seed + 11, n_components=N_PCS)

    causal_sites = sample_site_ids_for_maf(ts, a_idx, b_idx, n_sites=min(5000, int(ts.num_sites)), maf_min=0.01, seed=seed + 17)
    G_true = genetic_risk_from_real_pgs_effect_distribution(
        ts,
        a_idx,
        b_idx,
        causal_sites,
        pgs_effects,
        seed=seed + 19,
    )

    b0 = solve_intercept_for_prevalence(0.10, G_true)
    p = sigmoid(b0 + G_true)
    rng = np.random.default_rng(seed + 23)
    y = rng.binomial(1, p).astype(np.int32)

    rows = []
    ooa_train_set = set(ooa_train_idx.tolist())
    ooa_test_set = set(ooa_test_idx.tolist())
    for i in range(len(pop)):
        group = "AFR_test"
        if i in ooa_train_set:
            group = "OOA_train"
        elif i in ooa_test_set:
            group = "OOA_test"
        iid = f"ind_{i+1}"
        row = {
            "IID": iid,
            "individual_id": iid,
            "tskit_individual_id": int(ts_ind_id[i]),
            "pop_label": pop[i],
            "group": group,
            "y": int(y[i]),
            "G_true": float(G_true[i]),
            "gen": int(gens),
            "seed": int(seed),
        }
        for k in range(N_PCS):
            row[f"pc{k+1}"] = float(pcs[i, k])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Write VCF + PLINK files for BayesR.
    vcf_path = out_dir / f"{prefix}.vcf"
    with open(vcf_path, "w") as f:
        names = [f"ind_{i+1}" for i in range(ts.num_individuals)]
        ts.write_vcf(f, individual_names=names, position_transform=lambda x: np.asarray(x) + 1)
    run_plink_conversion(str(vcf_path), str(out_dir / prefix), cm_map_path=None)
    vcf_path.unlink(missing_ok=True)

    df.to_csv(out_dir / f"{prefix}.tsv", sep="\t", index=False)
    return df


def _prepare_bayesr_files(df: pd.DataFrame, prefix: str, work_dir: Path) -> tuple[str, str, str, str]:
    work_dir.mkdir(parents=True, exist_ok=True)

    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, names=["FID", "IID", "PID", "MID", "SEX", "PHENO"], dtype=str)
    iid_to_fid = dict(zip(fam["IID"], fam["FID"]))

    train_ids = df.loc[df["group"] == "OOA_train", "IID"].astype(str).tolist()
    test_ids = df.loc[df["group"].isin(["OOA_test", "AFR_test"]), "IID"].astype(str).tolist()

    train_keep = work_dir / "train.keep"
    test_keep = work_dir / "test.keep"
    pd.DataFrame({"FID": [iid_to_fid[i] for i in train_ids], "IID": train_ids}).to_csv(train_keep, sep="\t", header=False, index=False)
    pd.DataFrame({"FID": [iid_to_fid[i] for i in test_ids], "IID": test_ids}).to_csv(test_keep, sep="\t", header=False, index=False)

    plink = shutil.which("plink2") or "plink2"

    import subprocess
    subprocess.run([plink, "--bfile", prefix, "--freq", "--out", str(work_dir / "ref")], check=True)
    subprocess.run([plink, "--bfile", prefix, "--keep", str(train_keep), "--make-bed", "--out", str(work_dir / "train")], check=True)
    subprocess.run([plink, "--bfile", prefix, "--keep", str(test_keep), "--make-bed", "--out", str(work_dir / "test")], check=True)

    train_df = df[df["IID"].isin(train_ids)].copy()
    test_df = df[df["IID"].isin(test_ids)].copy()

    pd.DataFrame({
        "FID": [iid_to_fid[i] for i in train_df["IID"].astype(str)],
        "IID": train_df["IID"].astype(str),
        "y": train_df["y"].astype(int),
    }).to_csv(work_dir / "train.phen", sep=" ", header=False, index=False)

    covar_cols = [f"pc{i+1}" for i in range(N_PCS)]
    pd.DataFrame({
        "FID": [iid_to_fid[i] for i in train_df["IID"].astype(str)],
        "IID": train_df["IID"].astype(str),
        **{c: train_df[c].to_numpy() for c in covar_cols},
    }).to_csv(work_dir / "train.covar", sep=" ", header=False, index=False)

    return (
        str(work_dir / "train"),
        str(work_dir / "test"),
        str(work_dir / "train.phen"),
        str(work_dir / "ref.afreq"),
    )


def _fit_and_predict_methods(train_df: pd.DataFrame, test_df: pd.DataFrame, train_prs: np.ndarray, test_prs: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    P_train = StandardScaler(with_mean=True, with_std=True).fit_transform(train_prs.reshape(-1, 1)).ravel()
    P_test = StandardScaler(with_mean=True, with_std=True).fit_transform(test_prs.reshape(-1, 1)).ravel()
    PC_train = train_df[[f"pc{i+1}" for i in range(N_PCS)]].to_numpy()
    PC_test = test_df[[f"pc{i+1}" for i in range(N_PCS)]].to_numpy()
    y_train = train_df["y"].to_numpy(dtype=np.int32)

    methods = [
        ("raw", RawPGSMethod(max_iter=1000)),
        ("linear", LinearInteractionMethod(max_iter=1000)),
        ("normalized", NormalizationMethod(n_pcs=N_PCS, max_iter=1000)),
    ]

    for name, method in methods:
        if isinstance(method, NormalizationMethod):
            method.set_pop_labels(train_df["pop_label"].to_numpy())
            method.fit(P_train, PC_train, y_train)
            method.set_pop_labels(test_df["pop_label"].to_numpy())
            out[name] = method.predict_proba(P_test, PC_test)
            continue

        method.fit(P_train, PC_train, y_train)
        out[name] = method.predict_proba(P_test, PC_test)

    gam_attempts = [
        dict(n_pcs=2, k_pgs=4, k_pc=4, use_ti=False),
        dict(n_pcs=1, k_pgs=4, k_pc=4, use_ti=False),
        dict(n_pcs=1, k_pgs=3, k_pc=3, use_ti=False),
    ]
    last_gam_error = None
    for cfg in gam_attempts:
        try:
            gm = GAMMethod(**cfg)
            gm.fit(P_train, PC_train, y_train)
            out["gam"] = gm.predict_proba(P_test, PC_test)
            last_gam_error = None
            break
        except Exception as e:
            last_gam_error = e
    if last_gam_error is not None:
        raise RuntimeError(f"mgcv GAM failed for all tiny-sample configs: {last_gam_error}")

    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def _plot_main(df: pd.DataFrame, out_dir: Path) -> None:
    _apply_plot_style()
    plt.figure(figsize=(10, 6))
    method_colors = {
        "raw": CB["blue"],
        "normalized": CB["green"],
        "linear": CB["orange"],
        "gam": CB["purple"],
    }
    for m in sorted(df["method"].unique()):
        sub = df[df["method"] == m].sort_values("gens")
        plt.plot(
            sub["gens"],
            sub["auc_ratio"],
            marker="o",
            markersize=4.5,
            linewidth=2.0,
            label=m,
            color=method_colors.get(m, CB["black"]),
        )
    plt.xscale("log")
    plt.xlabel("Generations of divergence")
    plt.ylabel("AUC ratio (OoA test / AFR test)")
    ax = plt.gca()
    _style_axes(ax)
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "figure1_auc_ratio.png", dpi=240)
    plt.close()


def _plot_demography(out_dir: Path) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot([0, 0], [0, 1], color=CB["black"], linewidth=2.1)
    ax.plot([0, 1], [1, 1.8], color=CB["blue"], linewidth=2.2)
    ax.plot([0, 1], [1, 0.2], color=CB["orange"], linewidth=2.2)
    ax.text(1.02, 1.8, "AFR-like (Ne=10k)", va="center")
    ax.text(1.02, 0.2, "OoA-like (Ne 2k -> 10k)", va="center")
    ax.text(-0.02, 1.02, "Split", ha="right", va="bottom")
    ax.set_xlim(-0.1, 1.6)
    ax.set_ylim(-0.1, 2.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_demography.png", dpi=240)
    plt.close(fig)


def _plot_prs_grid(prs_rows: list[dict[str, object]], out_dir: Path) -> None:
    _apply_plot_style()
    def _draw_kde(ax, x: np.ndarray, color: str, label: str) -> None:
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return
        if x.size < 2 or float(np.std(x)) < 1e-8:
            xv = float(np.mean(x))
            ax.axvline(xv, color=color, alpha=0.9, linewidth=2, label=label)
            return
        lo, hi = float(np.min(x)), float(np.max(x))
        pad = max(1e-6, 0.25 * (hi - lo))
        grid = np.linspace(lo - pad, hi + pad, 200)
        kde = gaussian_kde(x)
        dens = kde(grid)
        ax.plot(grid, dens, color=color, linewidth=2, label=label)
        ax.fill_between(grid, 0.0, dens, color=color, alpha=0.22)

    df = pd.DataFrame(prs_rows)
    gens = sorted(df["gens"].unique())
    ncols = 4
    nrows = int(math.ceil(len(gens) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, g in enumerate(gens):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["gens"] == g]
        for pop, color in [("OOA_test", CB["blue"]), ("AFR_test", CB["orange"])]:
            x = sub.loc[sub["group"] == pop, "prs"].to_numpy(dtype=float)
            if x.size:
                _draw_kde(ax, x, color=color, label=pop)
        ax.text(0.03, 0.95, f"g={g}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        _style_axes(ax)
    for j in range(len(gens), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].legend(frameon=False, ncol=2, loc="upper right")
    fig.supxlabel("PRS", y=0.02)
    fig.supylabel("Density", x=0.01)
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_prs_distributions_grid.png", dpi=240)
    plt.close(fig)


def _plot_pc_grid(pc_rows: list[dict[str, object]], out_dir: Path) -> None:
    _apply_plot_style()
    df = pd.DataFrame(pc_rows)
    gens = sorted(df["gens"].unique())
    ncols = 4
    nrows = int(math.ceil(len(gens) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    for i, g in enumerate(gens):
        ax = axes[i // ncols][i % ncols]
        sub = df[df["gens"] == g]
        for pop, color in [("OOA_train", CB["sky"]), ("OOA_test", CB["blue"]), ("AFR_test", CB["orange"])]:
            s = sub[sub["group"] == pop]
            if len(s) > 0:
                ax.scatter(s["pc1"], s["pc2"], s=16, alpha=0.75, label=pop, color=color, edgecolors="none")
        ax.text(0.03, 0.95, f"g={g}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
        _style_axes(ax)
    for j in range(len(gens), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    axes[0][0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.supxlabel("PC1", y=0.02)
    fig.supylabel("PC2", x=0.01)
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_pc12_grid.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sims/results_figure1_local", help="Output directory")
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--gens", nargs="*", type=int, default=DIVERGENCE_GENS)
    parser.add_argument("--cache", default="sims/.cache")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_path = ensure_pgs003725(Path(args.cache))
    pgs_effects = load_pgs003725_effects(score_path, chr_filter="22")

    results = []
    prs_rows: list[dict[str, object]] = []
    pc_rows: list[dict[str, object]] = []

    for i, g in enumerate(args.gens):
        base_seed = args.seed + i * 101
        seed = None
        df = None
        prefix = None
        work = None
        train_prefix = None
        test_prefix = None
        train_phen = None
        freq_file = None
        train_scores = None
        test_scores = None
        max_retries = 8
        last_err = None
        for r in range(max_retries):
            seed = base_seed + r * 10_000
            try:
                df = _simulate_for_generation(g, seed, pgs_effects, out_dir)
                prefix = str(out_dir / f"fig1_g{g}_s{seed}")
                work = out_dir / f"fig1_g{g}_s{seed}_work"
                train_prefix, test_prefix, train_phen, freq_file = _prepare_bayesr_files(df, prefix, work)
                br = BayesR()
                eff_file = br.fit(train_prefix, train_phen, str(work / "bayesr"), covar_file=str(work / "train.covar"))
                test_scores = br.predict(test_prefix, eff_file, str(work / "bayesr_test"), freq_file=freq_file)
                train_scores = br.predict(train_prefix, eff_file, str(work / "bayesr_train"), freq_file=freq_file)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise RuntimeError(f"Figure1 failed for gens={g} after {max_retries} retries: {last_err}")

        train_df = df[df["group"] == "OOA_train"].copy()
        test_df = df[df["group"].isin(["OOA_test", "AFR_test"])].copy()

        train_df = train_df.set_index("IID").loc[train_scores["IID"].astype(str)].reset_index()
        test_df = test_df.set_index("IID").loc[test_scores["IID"].astype(str)].reset_index()

        pred = _fit_and_predict_methods(
            train_df,
            test_df,
            train_scores["PRS"].to_numpy(dtype=float),
            test_scores["PRS"].to_numpy(dtype=float),
        )

        for method, y_prob in pred.items():
            o = test_df["group"] == "OOA_test"
            a = test_df["group"] == "AFR_test"
            auc_o = _auc(test_df.loc[o, "y"].to_numpy(), y_prob[o.to_numpy()])
            auc_a = _auc(test_df.loc[a, "y"].to_numpy(), y_prob[a.to_numpy()])
            ratio = float(auc_o / auc_a) if np.isfinite(auc_o) and np.isfinite(auc_a) and auc_a > 0 else np.nan
            results.append({"gens": int(g), "seed": int(seed), "method": method, "auc_ooa": auc_o, "auc_afr": auc_a, "auc_ratio": ratio})

        for _, row in test_df.iterrows():
            prs_rows.append({
                "gens": int(g),
                "group": row["group"],
                "prs": float(test_scores.loc[test_scores["IID"].astype(str) == row["IID"], "PRS"].iloc[0]),
            })
        for _, row in df.iterrows():
            pc_rows.append({"gens": int(g), "group": row["group"], "pc1": float(row["pc1"]), "pc2": float(row["pc2"])})

    res_df = pd.DataFrame(results).sort_values(["method", "gens"]) 
    res_df.to_csv(out_dir / "figure1_auc_ratio.tsv", sep="\t", index=False)

    _plot_main(res_df, out_dir)
    _plot_demography(out_dir)
    _plot_prs_grid(prs_rows, out_dir)
    _plot_pc_grid(pc_rows, out_dir)


if __name__ == "__main__":
    main()
