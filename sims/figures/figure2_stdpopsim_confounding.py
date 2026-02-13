from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stdpopsim
import tskit
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
        pop_names_from_ts,
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
    pop_names_from_ts,
    )
from methods.raw_pgs import RawPGSMethod
from methods.linear_interaction import LinearInteractionMethod
from methods.normalization import NormalizationMethod
from methods.gam_mgcv import GAMMethod
from plink_utils import run_plink_conversion
from prs_tools import BayesR


# Requested scale-down by 100.
N_TRAIN_EUR = 120
N_TEST_PER_POP = 30
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


def _style_axes(ax, y_grid: bool = True) -> None:
    if y_grid:
        ax.grid(True, axis="y", alpha=0.55)
    else:
        ax.grid(True, axis="both", alpha=0.45)
    ax.set_axisbelow(True)


def _true_ancestry_proportions(ts, a_idx: np.ndarray, b_idx: np.ndarray, pop_lookup: dict[int, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sample_nodes = ts.samples()
    n_hap = len(sample_nodes)
    src_names = ["AFR", "EUR", "ASIA"]
    src_to_col = {name: j for j, name in enumerate(src_names)}
    acc = np.zeros((n_hap, len(src_names)), dtype=np.float64)
    total_len = float(ts.sequence_length)

    for tree in ts.trees():
        left, right = tree.interval
        seg_len = float(right - left)
        if seg_len <= 0:
            continue
        for h in range(n_hap):
            u = int(sample_nodes[h])
            src = None
            while u != tskit.NULL:
                pname = pop_lookup.get(int(ts.node(u).population), "")
                if pname in src_to_col:
                    src = pname
                    break
                u = tree.parent(u)
            if src is not None:
                acc[h, src_to_col[src]] += seg_len

    dip = (acc[a_idx] + acc[b_idx]) / 2.0
    denom = np.where(total_len > 0, total_len, 1.0)
    props = dip / denom
    afr = props[:, src_to_col["AFR"]]
    eur = props[:, src_to_col["EUR"]]
    asia = props[:, src_to_col["ASIA"]]
    s = afr + eur + asia
    bad = s <= 0
    if np.any(bad):
        afr[bad] = 1.0 / 3.0
        eur[bad] = 1.0 / 3.0
        asia[bad] = 1.0 / 3.0
        s = afr + eur + asia
    afr /= s
    eur /= s
    asia /= s
    return afr, eur, asia


def _simulate(seed: int, pgs_effects: np.ndarray, out_dir: Path, n_train_eur: int, n_test_per_pop: int) -> pd.DataFrame:
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("AmericanAdmixture_4B18")
    contig = species.get_contig("chr22", genetic_map="HapMapII_GRCh38", mutation_rate=model.mutation_rate)
    engine = stdpopsim.get_engine("msprime")

    samples = {
        "AFR": n_test_per_pop,
        "EUR": n_train_eur + n_test_per_pop,
        "ASIA": n_test_per_pop,
        "ADMIX": n_test_per_pop,
    }

    ts = engine.simulate(
        model,
        contig,
        samples,
        seed=seed,
        msprime_model="dtwf",
        msprime_change_model=[(50, "hudson")],
        record_migrations=True,
    )

    a_idx, b_idx, pop_idx, ts_ind_id = diploid_index_pairs(ts)
    pop_lookup = pop_names_from_ts(ts)
    pop_label = np.array([pop_lookup.get(int(p), f"pop_{int(p)}") for p in pop_idx], dtype=object)

    pca_sites = sample_site_ids_for_maf(ts, a_idx, b_idx, n_sites=2000, maf_min=0.05, seed=seed + 11)
    pcs = pcs_from_sites(ts, a_idx, b_idx, pca_sites, seed=seed + 13, n_components=N_PCS)

    causal_sites = sample_site_ids_for_maf(ts, a_idx, b_idx, n_sites=min(5000, int(ts.num_sites)), maf_min=0.01, seed=seed + 17)
    G_true = genetic_risk_from_real_pgs_effect_distribution(ts, a_idx, b_idx, causal_sites, pgs_effects, seed=seed + 19)

    rng = np.random.default_rng(seed + 23)
    afr_prop, eur_prop, asia_prop = _true_ancestry_proportions(ts, a_idx, b_idx, pop_lookup)

    env = 0.8 * afr_prop + 0.3 * asia_prop + 0.1 * eur_prop
    eta = 0.75 * G_true + env
    b0 = solve_intercept_for_prevalence(0.10, eta)
    y = rng.binomial(1, sigmoid(b0 + eta)).astype(np.int32)

    rows = []
    eur_idx = np.where(pop_label == "EUR")[0]
    eur_train = eur_idx[:n_train_eur]
    eur_test = eur_idx[n_train_eur:n_train_eur + n_test_per_pop]

    eur_train_set = set(eur_train.tolist())
    eur_test_set = set(eur_test.tolist())
    group_arr = np.empty(len(pop_label), dtype=object)
    for i in range(len(pop_label)):
        grp = "test"
        if i in eur_train_set:
            grp = "EUR_train"
        elif i in eur_test_set:
            grp = "EUR_test"
        elif pop_label[i] in {"AFR", "ASIA", "ADMIX"}:
            grp = f"{pop_label[i]}_test"
        group_arr[i] = grp

    for i in range(len(pop_label)):
        grp = str(group_arr[i])
        iid = f"ind_{i+1}"
        row = {
            "IID": iid,
            "individual_id": iid,
            "tskit_individual_id": int(ts_ind_id[i]),
            "pop_label": str(pop_label[i]),
            "group": grp,
            "y": int(y[i]),
            "G_true": float(G_true[i]),
            "afr_prop": float(afr_prop[i]),
            "eur_prop": float(eur_prop[i]),
            "asia_prop": float(asia_prop[i]),
            "seed": int(seed),
        }
        for k in range(N_PCS):
            row[f"pc{k+1}"] = float(pcs[i, k])
        rows.append(row)

    df = pd.DataFrame(rows)

    prefix = out_dir / f"fig2_s{seed}"
    vcf = prefix.with_suffix(".vcf")
    with open(vcf, "w") as f:
        names = [f"ind_{i+1}" for i in range(ts.num_individuals)]
        ts.write_vcf(f, individual_names=names, position_transform=lambda x: np.asarray(x) + 1)
    run_plink_conversion(str(vcf), str(prefix), cm_map_path=None)
    vcf.unlink(missing_ok=True)

    df.to_csv(prefix.with_suffix(".tsv"), sep="\t", index=False)
    return df


def _run_bayesr_and_predict(df: pd.DataFrame, prefix: str, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    import shutil
    import subprocess

    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, names=["FID", "IID", "PID", "MID", "SEX", "PHENO"], dtype=str)
    iid_to_fid = dict(zip(fam["IID"], fam["FID"]))

    train_ids = df.loc[df["group"] == "EUR_train", "IID"].astype(str).tolist()
    test_ids = df.loc[df["group"].str.endswith("_test"), "IID"].astype(str).tolist()

    pd.DataFrame({"FID": [iid_to_fid[i] for i in train_ids], "IID": train_ids}).to_csv(work / "train.keep", sep="\t", header=False, index=False)
    pd.DataFrame({"FID": [iid_to_fid[i] for i in test_ids], "IID": test_ids}).to_csv(work / "test.keep", sep="\t", header=False, index=False)

    plink = shutil.which("plink2") or "plink2"
    subprocess.run([plink, "--bfile", prefix, "--freq", "--out", str(work / "ref")], check=True)
    subprocess.run([plink, "--bfile", prefix, "--keep", str(work / "train.keep"), "--make-bed", "--out", str(work / "train")], check=True)
    subprocess.run([plink, "--bfile", prefix, "--keep", str(work / "test.keep"), "--make-bed", "--out", str(work / "test")], check=True)

    train_df = df[df["IID"].isin(train_ids)].copy()
    pd.DataFrame({
        "FID": [iid_to_fid[i] for i in train_df["IID"].astype(str)],
        "IID": train_df["IID"].astype(str),
        "y": train_df["y"].astype(int),
    }).to_csv(work / "train.phen", sep=" ", header=False, index=False)

    pd.DataFrame({
        "FID": [iid_to_fid[i] for i in train_df["IID"].astype(str)],
        "IID": train_df["IID"].astype(str),
        **{f"pc{i+1}": train_df[f"pc{i+1}"].to_numpy() for i in range(N_PCS)},
    }).to_csv(work / "train.covar", sep=" ", header=False, index=False)

    br = BayesR()
    eff = br.fit(str(work / "train"), str(work / "train.phen"), str(work / "bayesr"), covar_file=str(work / "train.covar"))
    train_scores = br.predict(str(work / "train"), eff, str(work / "bayesr_train"), freq_file=str(work / "ref.afreq"))
    test_scores = br.predict(str(work / "test"), eff, str(work / "bayesr_test"), freq_file=str(work / "ref.afreq"))
    return train_scores, test_scores


def _cleanup_seed_artifacts(prefix: Path, seed_work_dir: Path) -> None:
    for ext in (".bed", ".bim", ".fam", ".log", ".tsv", ".vcf"):
        p = Path(f"{prefix}{ext}")
        if p.exists():
            p.unlink(missing_ok=True)
    if seed_work_dir.exists():
        shutil.rmtree(seed_work_dir, ignore_errors=True)


def _method_preds(train_df: pd.DataFrame, test_df: pd.DataFrame, train_prs: np.ndarray, test_prs: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    P_train = StandardScaler().fit_transform(train_prs.reshape(-1, 1)).ravel()
    P_test = StandardScaler().fit_transform(test_prs.reshape(-1, 1)).ravel()
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

    # Fixed GAM specification for reproducibility.
    gm = GAMMethod(n_pcs=2, k_pgs=4, k_pc=4, use_ti=False)
    gm.fit(P_train, PC_train, y_train)
    out["gam"] = gm.predict_proba(P_test, PC_test)

    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def _plot_main(df: pd.DataFrame, out_dir: Path) -> None:
    _apply_plot_style()
    methods = [m for m in ["raw", "normalized", "linear", "gam"] if m in set(df["method"])]
    pops = ["EUR", "AFR", "ASIA", "ADMIX"]
    pop_colors = {"EUR": CB["blue"], "AFR": CB["orange"], "ASIA": CB["green"], "ADMIX": CB["purple"]}

    x = np.arange(len(methods))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, pop in enumerate(pops):
        vals = []
        for m in methods:
            sub = df[(df["method"] == m) & (df["population"] == pop)]
            vals.append(float(sub["auc"].iloc[0]) if len(sub) else np.nan)
        ax.bar(
            x + (i - 1.5) * width,
            vals,
            width=width,
            label=pop,
            color=pop_colors[pop],
            edgecolor="#222222",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("AUC")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False)
    _style_axes(ax, y_grid=True)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_auc_by_method_population.png", dpi=240)
    plt.close(fig)


def _plot_prs_distribution(test_df: pd.DataFrame, prs: np.ndarray, out_dir: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(8, 5))
    for pop, color in [("EUR", CB["blue"]), ("AFR", CB["orange"]), ("ASIA", CB["green"]), ("ADMIX", CB["purple"])]:
        mask = test_df["pop_label"] == pop
        x = prs[mask.to_numpy()]
        if x.size:
            _draw_kde(ax, x, color=color, label=pop)
    ax.set_xlabel("PRS")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    _style_axes(ax, y_grid=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_prs_distributions.png", dpi=240)
    plt.close(fig)


def _plot_pcs(df: pd.DataFrame, out_dir: Path) -> None:
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    for pop, color in [("EUR", CB["blue"]), ("AFR", CB["orange"]), ("ASIA", CB["green"]), ("ADMIX", CB["purple"])]:
        s = df[df["pop_label"] == pop]
        if len(s):
            ax.scatter(s["pc1"], s["pc2"], s=24, alpha=0.75, label=pop, color=color, edgecolors="none")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(frameon=False)
    _style_axes(ax, y_grid=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_pc12.png", dpi=240)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sims/results_figure2_local")
    parser.add_argument("--seed", type=int, default=99173)
    parser.add_argument("--cache", default="sims/.cache")
    parser.add_argument("--n-train-eur", type=int, default=N_TRAIN_EUR)
    parser.add_argument("--n-test-per-pop", type=int, default=N_TEST_PER_POP)
    parser.add_argument("--work-root", default=None, help="Directory for heavy transient work files (e.g., RAM disk).")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep PLINK/GCTB intermediate files.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root) if args.work_root else out_dir
    work_root.mkdir(parents=True, exist_ok=True)

    score_path = ensure_pgs003725(Path(args.cache))
    pgs_effects = load_pgs003725_effects(score_path, chr_filter="22")

    df = None
    prefix = None
    train_scores = None
    test_scores = None
    seed = int(args.seed)
    df = _simulate(
        seed,
        pgs_effects,
        out_dir,
        n_train_eur=int(args.n_train_eur),
        n_test_per_pop=int(args.n_test_per_pop),
    )
    prefix = str(out_dir / f"fig2_s{seed}")
    seed_work = work_root / f"work_s{seed}"
    train_scores, test_scores = _run_bayesr_and_predict(df, prefix, seed_work)

    train_df = df[df["group"] == "EUR_train"].copy()
    test_df = df[df["group"].str.endswith("_test")].copy()
    train_df = train_df.set_index("IID").loc[train_scores["IID"].astype(str)].reset_index()
    test_df = test_df.set_index("IID").loc[test_scores["IID"].astype(str)].reset_index()

    preds = _method_preds(
        train_df,
        test_df,
        train_scores["PRS"].to_numpy(dtype=float),
        test_scores["PRS"].to_numpy(dtype=float),
    )

    rows = []
    for method, y_prob in preds.items():
        for pop in ["EUR", "AFR", "ASIA", "ADMIX"]:
            mask = test_df["pop_label"] == pop
            rows.append({"method": method, "population": pop, "auc": _auc(test_df.loc[mask, "y"].to_numpy(), y_prob[mask.to_numpy()])})

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "figure2_auc_by_method_population.tsv", sep="\t", index=False)

    _plot_main(res, out_dir)
    _plot_prs_distribution(test_df, test_scores["PRS"].to_numpy(dtype=float), out_dir)
    _plot_pcs(df, out_dir)

    if not args.keep_intermediates:
        _cleanup_seed_artifacts(prefix=Path(prefix), seed_work_dir=seed_work)


if __name__ == "__main__":
    main()
