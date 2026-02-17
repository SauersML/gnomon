from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import msprime
from scipy.stats import gaussian_kde
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler

# mgcv path is required; prefer rpy2 API mode.
os.environ.setdefault("RPY2_CFFI_MODE", "API")

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
        summarize_true_effect_site_diagnostics,
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
    summarize_true_effect_site_diagnostics,
    )
from methods.raw_pgs import RawPGSMethod
from methods.linear_interaction import LinearInteractionMethod
from methods.normalization import NormalizationMethod
from methods.gam_mgcv import GAMMethod
from plink_utils import run_plink_conversion
from prs_tools import BayesR, PPlusT


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


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


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


def _set_runtime_thread_env(threads: int) -> None:
    t = str(max(1, int(threads)))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GCTB_THREADS",
    ):
        os.environ[k] = t


def _default_total_threads() -> int:
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "OMP_NUM_THREADS"):
        val = os.environ.get(key)
        if val:
            try:
                return max(1, int(val))
            except Exception:
                pass
    return max(1, int(os.cpu_count() or 1))


def _detect_total_mem_mb() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(1, kb // 1024)
    except Exception:
        return None
    return None


def _default_work_root(out_dir: Path) -> Path:
    for base in (Path("/dev/shm"), Path("/tmp")):
        if base.exists():
            return (base / "gnomon_sims_work" / "figure1").resolve()
    return out_dir / "work"


def _truncate_ratemap(rate_map: msprime.RateMap, bp_cap: int | None) -> msprime.RateMap:
    if bp_cap is None:
        return rate_map
    cap = int(bp_cap)
    if cap <= 0:
        raise ValueError("bp_cap must be positive")
    pos = np.asarray(rate_map.position, dtype=float)
    rates = np.asarray(rate_map.rate, dtype=float)
    if cap >= int(pos[-1]):
        return rate_map
    i = int(np.searchsorted(pos, float(cap), side="right") - 1)
    i = max(0, min(i, len(rates) - 1))
    new_pos = np.concatenate([pos[: i + 1], np.asarray([float(cap)])])
    new_rates = rates[: i + 1]
    return msprime.RateMap(position=new_pos, rate=new_rates)


def _resource_plan(
    n_tasks: int,
    total_threads_override: int | None = None,
    total_mem_mb_override: int | None = None,
) -> tuple[int, int, int | None]:
    total_threads = (
        max(1, int(total_threads_override))
        if total_threads_override is not None
        else _default_total_threads()
    )
    total_mem_mb = (
        max(1024, int(total_mem_mb_override))
        if total_mem_mb_override is not None
        else _detect_total_mem_mb()
    )
    usable_mem_mb = int(0.85 * total_mem_mb) if total_mem_mb is not None else None
    max_jobs = max(1, min(int(n_tasks), total_threads))

    best = None
    for jobs in range(1, max_jobs + 1):
        threads_per_job = max(1, total_threads // jobs)
        mem_per_job_mb = None if usable_mem_mb is None else max(2048, usable_mem_mb // jobs)
        # Favor more parallel tasks, but cap per-task thread benefit around 8.
        score = jobs * min(threads_per_job, 8)
        if threads_per_job < 2:
            score -= 0.25
        cand = (score, jobs, threads_per_job, mem_per_job_mb)
        if best is None or cand > best:
            best = cand

    assert best is not None
    _, jobs, threads_per_job, mem_per_job_mb = best
    return int(jobs), int(threads_per_job), mem_per_job_mb


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


def _simulate_for_generation(
    gens: int,
    seed: int,
    pgs_effects: np.ndarray,
    out_dir: Path,
    n_afr: int,
    n_ooa_train: int,
    n_ooa_test: int,
    plink_threads: int,
    plink_memory_mb: int | None,
    msprime_model: str,
    mut_rate: float,
    pca_n_sites: int,
    causal_max_sites: int,
    bp_cap: int | None,
) -> pd.DataFrame:
    prefix = f"fig1_g{gens}_s{seed}"
    _log(
        f"[{prefix}] Start generation simulation "
        f"(model={msprime_model}, n_afr={n_afr}, n_ooa_train={n_ooa_train}, n_ooa_test={n_ooa_test})"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[{prefix}] Loading/truncating recombination map (bp_cap={bp_cap})")
    recomb_map = _truncate_ratemap(get_chr22_recomb_map(), bp_cap=bp_cap)
    _log(f"[{prefix}] Building demography for split_gens={gens}")
    dem = _build_demography(gens)

    _log(f"[{prefix}] Running msprime.sim_ancestry")
    ts = msprime.sim_ancestry(
        samples={"afr": n_afr, "ooa": n_ooa_train + n_ooa_test},
        demography=dem,
        recombination_rate=recomb_map,
        ploidy=2,
        random_seed=seed,
        model=msprime_model,
    )
    _log(f"[{prefix}] Ancestry complete (nodes={ts.num_nodes}, edges={ts.num_edges}, trees={ts.num_trees})")
    _log(f"[{prefix}] Running msprime.sim_mutations (mut_rate={mut_rate})")
    ts = msprime.sim_mutations(
        ts,
        rate=mut_rate,
        random_seed=seed + 1,
        discrete_genome=True,
    )
    _log(f"[{prefix}] Mutations complete (sites={ts.num_sites})")

    _log(f"[{prefix}] Extracting diploid sample mappings")
    a_idx, b_idx, pop_idx, ts_ind_id = diploid_index_pairs(ts)
    pop_map = {0: "AFR", 1: "OOA"}
    pop = np.array([pop_map.get(int(x), "UNK") for x in pop_idx], dtype=object)

    ooa_idx = np.where(pop == "OOA")[0]
    afr_idx = np.where(pop == "AFR")[0]
    if len(ooa_idx) < (n_ooa_train + n_ooa_test) or len(afr_idx) < n_afr:
        raise RuntimeError("Unexpected sample counts after simulation")

    ooa_train_idx = ooa_idx[:n_ooa_train]
    ooa_test_idx = ooa_idx[n_ooa_train:n_ooa_train + n_ooa_test]

    _log(f"[{prefix}] Sampling PCA sites (n={pca_n_sites}, maf_min=0.05)")
    pca_sites = sample_site_ids_for_maf(
        ts,
        a_idx,
        b_idx,
        n_sites=pca_n_sites,
        maf_min=0.05,
        seed=seed + 7,
        log_fn=_log,
        progress_label=f"{prefix} pca_site_scan",
    )
    _log(f"[{prefix}] Computing PCs (n_components={N_PCS})")
    pcs = pcs_from_sites(ts, a_idx, b_idx, pca_sites, seed=seed + 11, n_components=N_PCS)

    _log(f"[{prefix}] Sampling causal sites (max_n={causal_max_sites}, maf_min=0.01)")
    causal_sites = sample_site_ids_for_maf(
        ts,
        a_idx,
        b_idx,
        n_sites=min(causal_max_sites, int(ts.num_sites)),
        maf_min=0.01,
        seed=seed + 17,
        log_fn=_log,
        progress_label=f"{prefix} causal_site_scan",
    )
    _log(f"[{prefix}] Building genetic risk from causal sites (n={len(causal_sites)})")
    G_true = genetic_risk_from_real_pgs_effect_distribution(
        ts,
        a_idx,
        b_idx,
        causal_sites,
        pgs_effects,
        seed=seed + 19,
        log_fn=_log,
        progress_label=f"{prefix} genetic_risk",
    )
    ts_sites, causal_overlap, het_by_pop, causal_pos_1based = summarize_true_effect_site_diagnostics(
        ts,
        a_idx,
        b_idx,
        pop,
        causal_sites,
        log_fn=_log,
        progress_label=f"{prefix} true_effect_diagnostics",
    )
    het_bits = ", ".join(f"{k}={v:.4f}" for k, v in sorted(het_by_pop.items()))
    _log(
        f"[{prefix}] Variant diagnostics: ts_sites={ts_sites}, "
        f"true_effect_sites={len(causal_sites)}, overlap={causal_overlap}"
    )
    _log(f"[{prefix}] Mean heterozygosity at true-effect sites by pop: {het_bits}")

    _log(f"[{prefix}] Sampling phenotypes from liability model")
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

    _log(f"[{prefix}] Building dataframe (n_rows={len(rows)})")
    df = pd.DataFrame(rows)

    # Write VCF + PLINK files for downstream PRS training/scoring.
    vcf_path = out_dir / f"{prefix}.vcf"
    _log(f"[{prefix}] Writing VCF to {vcf_path}")
    with open(vcf_path, "w") as f:
        names = [f"ind_{i+1}" for i in range(ts.num_individuals)]
        ts.write_vcf(f, individual_names=names, position_transform=lambda x: np.asarray(x) + 1)
    _log(f"[{prefix}] Converting VCF to PLINK")
    run_plink_conversion(
        str(vcf_path),
        str(out_dir / prefix),
        cm_map_path=None,
        threads=plink_threads,
        memory_mb=plink_memory_mb,
    )
    bim_path = out_dir / f"{prefix}.bim"
    bim_n_variants = sum(1 for _ in open(bim_path, "r", encoding="utf-8", errors="replace"))
    bim_pos = set()
    with open(bim_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4:
                try:
                    bim_pos.add(int(parts[3]))
                except ValueError:
                    pass
    overlap_bim_true_effect = len(bim_pos & causal_pos_1based)
    _log(
        f"[{prefix}] PLINK conversion complete "
        f"(bim_variants={bim_n_variants}, overlap_true_effect_positions={overlap_bim_true_effect})"
    )
    vcf_path.unlink(missing_ok=True)

    _log(f"[{prefix}] Writing simulation table to {out_dir / f'{prefix}.tsv'}")
    df.to_csv(out_dir / f"{prefix}.tsv", sep="\t", index=False)
    _log(f"[{prefix}] Generation simulation complete")
    return df


def _prepare_bayesr_files(
    df: pd.DataFrame,
    prefix: str,
    work_dir: Path,
    plink_threads: int,
    plink_memory_mb: int | None,
) -> tuple[str, str, str, str]:
    _log(f"[{Path(prefix).name}] Preparing PRS input files in {work_dir}")
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
    common = ["--threads", str(plink_threads)]
    if plink_memory_mb is not None and plink_memory_mb > 0:
        common.extend(["--memory", str(plink_memory_mb)])
    _log(f"[{Path(prefix).name}] Running plink2 --freq")
    subprocess.run([plink, "--bfile", prefix, "--freq", *common, "--out", str(work_dir / "ref")], check=True)
    _log(f"[{Path(prefix).name}] Running plink2 train split")
    subprocess.run([plink, "--bfile", prefix, "--keep", str(train_keep), "--make-bed", *common, "--out", str(work_dir / "train")], check=True)
    _log(f"[{Path(prefix).name}] Running plink2 test split")
    subprocess.run([plink, "--bfile", prefix, "--keep", str(test_keep), "--make-bed", *common, "--out", str(work_dir / "test")], check=True)

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


def _cleanup_generation_artifacts(prefix: Path, work_dir: Path) -> None:
    # Keep only final aggregated figure outputs in the top-level output directory.
    _log(f"[{prefix.name}] Cleaning generation artifacts")
    for ext in (".bed", ".bim", ".fam", ".log", ".tsv", ".vcf"):
        p = Path(f"{prefix}{ext}")
        if p.exists():
            p.unlink(missing_ok=True)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


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

    gm = GAMMethod(n_pcs=3, k_pgs=4, k_pc=4, k_interaction=3, use_ti=True)
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


def _plot_demography(out_dir: Path, split_gens: int) -> None:
    _apply_plot_style()
    import demes
    import demesdraw

    b = demes.Builder(time_units="generations")
    b.add_deme(
        "ancestral",
        epochs=[dict(start_size=10_000, end_size=10_000, end_time=float(split_gens))],
    )
    b.add_deme(
        "AFR_like",
        ancestors=["ancestral"],
        start_time=float(split_gens),
        epochs=[dict(start_size=10_000, end_size=10_000, end_time=0)],
    )
    b.add_deme(
        "OoA_like",
        ancestors=["ancestral"],
        start_time=float(split_gens),
        epochs=[dict(start_size=2_000, end_size=10_000, end_time=0)],
    )
    graph = b.resolve()

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    demesdraw.tubes(graph, ax=ax, labels="xticks-mid")
    ax.set_title(f"Figure 1 Demography Schematic (split = {int(split_gens)} generations)")
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


def _log_results_table(title: str, df: pd.DataFrame) -> None:
    if df.empty:
        _log(f"{title}: <empty>")
        return
    txt = df.to_string(index=False, justify="left")
    _log(f"{title}\n{txt}")


def _plot_pt_train_accuracy(metrics_df: pd.DataFrame, out_path: Path, title: str) -> None:
    if metrics_df.empty:
        return
    _apply_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = metrics_df["p_threshold"].to_numpy(dtype=float)
    y = metrics_df["train_accuracy"].to_numpy(dtype=float)
    ax.plot(x, y, marker="o", linewidth=2, color=CB["blue"])
    ax.set_xscale("log")
    ax.set_xlabel("P+T p-value threshold")
    ax.set_ylabel("Train accuracy")
    ax.set_title(title)
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _run_generation_task(task: dict[str, object]) -> dict[str, object]:
    g = int(task["g"])
    seed = int(task["seed"])
    out_dir = Path(str(task["out_dir"]))
    work_root = Path(str(task["work_root"]))
    n_afr = int(task["n_afr"])
    n_ooa_train = int(task["n_ooa_train"])
    n_ooa_test = int(task["n_ooa_test"])
    threads_per_job = int(task["threads_per_job"])
    memory_mb_per_job = task["memory_mb_per_job"]
    keep_intermediates = bool(task["keep_intermediates"])
    msprime_model = str(task["msprime_model"])
    mut_rate = float(task["mut_rate"])
    pca_n_sites = int(task["pca_n_sites"])
    causal_max_sites = int(task["causal_max_sites"])
    bp_cap = task["bp_cap"]
    pgs_effects = np.asarray(task["pgs_effects"], dtype=float)
    use_existing = bool(task.get("use_existing", False))
    use_existing_dir_raw = task.get("use_existing_dir")
    use_existing_dir = Path(str(use_existing_dir_raw)) if use_existing_dir_raw is not None else None
    _log(f"[fig1_g{g}_s{seed}] Task started")

    _set_runtime_thread_env(threads_per_job)
    if memory_mb_per_job is not None:
        os.environ["PLINK_MEMORY_MB"] = str(int(memory_mb_per_job))
    _log(f"[fig1_g{g}_s{seed}] Runtime resources set (threads={threads_per_job}, memory_mb={memory_mb_per_job})")

    prefix = str(out_dir / f"fig1_g{g}_s{seed}")
    source_base = use_existing_dir if use_existing_dir is not None else out_dir
    source_prefix = source_base / f"fig1_g{g}_s{seed}"
    sim_tsv = source_base / f"fig1_g{g}_s{seed}.tsv"
    if use_existing:
        needed = [Path(f"{source_prefix}.bed"), Path(f"{source_prefix}.bim"), Path(f"{source_prefix}.fam"), sim_tsv]
        miss = [str(p) for p in needed if not p.exists()]
        if miss:
            raise FileNotFoundError(
                f"[fig1_g{g}_s{seed}] --use-existing requested, but missing files: {miss}"
            )
        _log(f"[fig1_g{g}_s{seed}] Reusing existing simulation/PLINK files from {source_base}")
        df = pd.read_csv(sim_tsv, sep="\t")
    else:
        df = _simulate_for_generation(
            g,
            seed,
            pgs_effects,
            out_dir,
            n_afr=n_afr,
            n_ooa_train=n_ooa_train,
            n_ooa_test=n_ooa_test,
            plink_threads=threads_per_job,
            plink_memory_mb=memory_mb_per_job,
            msprime_model=msprime_model,
            mut_rate=mut_rate,
            pca_n_sites=pca_n_sites,
            causal_max_sites=causal_max_sites,
            bp_cap=(int(bp_cap) if bp_cap is not None else None),
        )
    work = work_root / f"fig1_g{g}_s{seed}_work"
    train_prefix, test_prefix, train_phen, freq_file = _prepare_bayesr_files(
        df,
        str(source_prefix),
        work,
        plink_threads=threads_per_job,
        plink_memory_mb=memory_mb_per_job,
    )
    use_bayesr = bool(task.get("use_bayesr", False))
    if use_bayesr:
        _log(f"[fig1_g{g}_s{seed}] BayesR fit starting")
        br = BayesR(threads=threads_per_job, plink_memory_mb=memory_mb_per_job)
        eff_file = br.fit(train_prefix, train_phen, str(work / "bayesr"), covar_file=str(work / "train.covar"))
        _log(f"[fig1_g{g}_s{seed}] BayesR fit complete; scoring test/train")
        test_scores = br.predict(test_prefix, eff_file, str(work / "bayesr_test"), freq_file=freq_file)
        train_scores = br.predict(train_prefix, eff_file, str(work / "bayesr_train"), freq_file=freq_file)
    else:
        _log(f"[fig1_g{g}_s{seed}] P+T fit/scoring starting")
        pt = PPlusT(threads=threads_per_job, plink_memory_mb=memory_mb_per_job)
        train_scores, test_scores, pt_meta = pt.fit_and_predict(
            bfile_train=train_prefix,
            bfile_test=test_prefix,
            pheno_file=train_phen,
            covar_file=str(work / "train.covar"),
            freq_file=freq_file,
            out_prefix=str(work / "pt"),
        )
        pt_metrics_df = pt_meta["threshold_metrics"].copy()
        pt_metrics_df.insert(0, "gens", int(g))
        pt_metrics_df.insert(1, "seed", int(seed))
        pt_metrics_df["is_selected"] = pt_metrics_df["p_threshold"] == float(pt_meta["best_p_threshold"])
        pt_tsv_path = out_dir / f"fig1_g{g}_s{seed}.pt_thresholds.tsv"
        pt_plot_path = out_dir / f"fig1_g{g}_s{seed}.pt_train_accuracy.png"
        pt_metrics_df.to_csv(pt_tsv_path, sep="\t", index=False)
        _plot_pt_train_accuracy(pt_metrics_df, pt_plot_path, title=f"fig1 g={g} seed={seed} train accuracy")
        _log(
            f"[fig1_g{g}_s{seed}] P+T selected p={float(pt_meta['best_p_threshold']):g} "
            f"(train_accuracy={float(pt_meta['best_train_accuracy']):.4f}, n_snps={int(pt_meta['best_n_snps'])})"
        )
        _log(f"[fig1_g{g}_s{seed}] P+T fit/scoring complete")

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
    _log(f"[fig1_g{g}_s{seed}] Method fitting/prediction complete")
    test_prs = test_scores["PRS"].to_numpy(dtype=float)

    result_rows = []
    for method, y_prob in pred.items():
        o = test_df["group"] == "OOA_test"
        a = test_df["group"] == "AFR_test"
        auc_o = _auc(test_df.loc[o, "y"].to_numpy(), y_prob[o.to_numpy()])
        auc_a = _auc(test_df.loc[a, "y"].to_numpy(), y_prob[a.to_numpy()])
        ratio = float(auc_o / auc_a) if np.isfinite(auc_o) and np.isfinite(auc_a) and auc_a > 0 else np.nan
        result_rows.append(
            {
                "gens": int(g),
                "seed": int(seed),
                "method": method,
                "n_train": int(len(train_df)),
                "n_test_total": int(len(test_df)),
                "n_test_ooa": int(np.count_nonzero(o.to_numpy())),
                "n_test_afr": int(np.count_nonzero(a.to_numpy())),
                "auc_ooa": auc_o,
                "auc_afr": auc_a,
                "auc_ratio": ratio,
            }
        )

    prs_df = test_df[["IID", "group"]].copy()
    prs_df["IID"] = prs_df["IID"].astype(str)
    prs_df = prs_df.merge(
        test_scores[["IID", "PRS"]].assign(IID=test_scores["IID"].astype(str)),
        on="IID",
        how="left",
    )
    prs_df.insert(0, "gens", int(g))
    prs_df = prs_df.rename(columns={"PRS": "prs"})

    pc_df = df[["group", "pc1", "pc2"]].copy()
    pc_df.insert(0, "gens", int(g))

    pred_rows = []
    for method, y_prob in pred.items():
        part = test_df[["IID", "group", "pop_label", "y", "G_true", "pc1", "pc2", "pc3", "pc4", "pc5"]].copy()
        part.insert(0, "seed", int(seed))
        part.insert(0, "gens", int(g))
        part["method"] = method
        part["prs"] = test_prs
        part["y_prob"] = np.asarray(y_prob, dtype=float)
        pred_rows.append(part)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    res_path = out_dir / f"fig1_g{g}_s{seed}.results.tsv"
    prs_path = out_dir / f"fig1_g{g}_s{seed}.prs.tsv"
    pc_path = out_dir / f"fig1_g{g}_s{seed}.pc.tsv"
    pred_path = out_dir / f"fig1_g{g}_s{seed}.predictions.tsv"
    pd.DataFrame(result_rows).to_csv(res_path, sep="\t", index=False)
    prs_df.to_csv(prs_path, sep="\t", index=False)
    pc_df.to_csv(pc_path, sep="\t", index=False)
    pred_df.to_csv(pred_path, sep="\t", index=False)
    _log(f"[fig1_g{g}_s{seed}] Wrote per-generation outputs")

    if not keep_intermediates and not use_existing:
        _cleanup_generation_artifacts(prefix=Path(prefix), work_dir=Path(work))
    elif not keep_intermediates and use_existing:
        shutil.rmtree(Path(work), ignore_errors=True)
    _log(f"[fig1_g{g}_s{seed}] Task complete")

    return {
        "gens": int(g),
        "seed": int(seed),
        "res_path": str(res_path),
        "prs_path": str(prs_path),
        "pc_path": str(pc_path),
        "pred_path": str(pred_path),
        "pt_thresholds_path": str(pt_tsv_path) if not use_bayesr else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="sims/results_figure1_local", help="Output directory")
    parser.add_argument("--seed", type=int, default=90210)
    parser.add_argument("--gens", nargs="*", type=int, default=DIVERGENCE_GENS)
    parser.add_argument("--cache", default="sims/.cache")
    parser.add_argument("--n-afr", type=int, default=N_AFR)
    parser.add_argument("--n-ooa-train", type=int, default=N_OOA_TRAIN)
    parser.add_argument("--n-ooa-test", type=int, default=N_OOA_TEST)
    parser.add_argument(
        "--msprime-model",
        choices=["dtwf", "hudson"],
        default="hudson",
        help="msprime ancestry model (hudson is usually faster).",
    )
    parser.add_argument("--mut-rate", type=float, default=1e-8)
    parser.add_argument("--pca-sites", type=int, default=2000)
    parser.add_argument("--causal-sites", type=int, default=5000)
    parser.add_argument("--bp-cap", type=int, default=None)
    parser.add_argument(
        "--total-threads",
        type=int,
        default=None,
        help="Total thread budget for Figure 1 auto-planner.",
    )
    parser.add_argument(
        "--memory-mb-total",
        type=int,
        default=None,
        help="Total memory budget (MB) for Figure 1 auto-planner.",
    )
    parser.add_argument("--work-root", default=None, help="Directory for heavy transient work files (e.g., RAM disk).")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep per-generation PLINK/GCTB intermediate files.")
    parser.add_argument("--bayesr", action="store_true", help="Use BayesR backend (default is fast P+T).")
    parser.add_argument("--use-existing", action="store_true", help="Reuse existing fig1_g*_s* PLINK/TSV files; skip simulation.")
    parser.add_argument("--use-existing-dir", default=None, help="Directory containing existing fig1_g*_s* PLINK/TSV files.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root) if args.work_root else _default_work_root(out_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    _log("Figure1 GAM backend: mgcv")
    _log(f"Figure1 PRS backend: {'BayesR' if args.bayesr else 'P+T'}")

    score_path = ensure_pgs003725(Path(args.cache))
    pgs_effects = load_pgs003725_effects(score_path, chr_filter="22")
    # Prewarm stdpopsim chr22 map once in the parent process to avoid
    # concurrent cache initialization races across workers.
    _ = get_chr22_recomb_map()
    jobs, threads_per_job, memory_mb_per_job = _resource_plan(
        n_tasks=len(args.gens),
        total_threads_override=args.total_threads,
        total_mem_mb_override=args.memory_mb_total,
    )
    _set_runtime_thread_env(threads_per_job)
    _log(
        f"Figure1 resources: jobs={jobs} threads_per_job={threads_per_job} "
        f"memory_mb_per_job={memory_mb_per_job}"
    )

    tasks = []
    for i, g in enumerate(args.gens):
        tasks.append(
            {
                "g": int(g),
                "seed": int(args.seed + i * 101),
                "out_dir": str(out_dir),
                "work_root": str(work_root),
                "n_afr": int(args.n_afr),
                "n_ooa_train": int(args.n_ooa_train),
                "n_ooa_test": int(args.n_ooa_test),
                "threads_per_job": int(threads_per_job),
                "memory_mb_per_job": memory_mb_per_job,
                "keep_intermediates": bool(args.keep_intermediates),
                "msprime_model": str(args.msprime_model),
                "mut_rate": float(args.mut_rate),
                "pca_n_sites": int(args.pca_sites),
                "causal_max_sites": int(args.causal_sites),
                "bp_cap": args.bp_cap,
                "pgs_effects": pgs_effects,
                "use_bayesr": bool(args.bayesr),
                "use_existing": bool(args.use_existing),
                "use_existing_dir": args.use_existing_dir,
            }
        )
    _log(f"Figure1 queued {len(tasks)} generation tasks")

    done = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(_run_generation_task, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futs):
            rec = fut.result()
            done.append(rec)
            _log(f"Completed generation g={rec['gens']} seed={rec['seed']}")

    done = sorted(done, key=lambda x: int(x["gens"]))
    res_df = pd.concat([pd.read_csv(str(r["res_path"]), sep="\t") for r in done], ignore_index=True).sort_values(["method", "gens"])
    prs_df = pd.concat([pd.read_csv(str(r["prs_path"]), sep="\t") for r in done], ignore_index=True)
    pc_df = pd.concat([pd.read_csv(str(r["pc_path"]), sep="\t") for r in done], ignore_index=True)
    pred_df = pd.concat([pd.read_csv(str(r["pred_path"]), sep="\t") for r in done], ignore_index=True)
    res_df.to_csv(out_dir / "figure1_auc_ratio.tsv", sep="\t", index=False)
    pred_df.to_csv(out_dir / "figure1_predictions.tsv", sep="\t", index=False)
    pt_paths = [str(r.get("pt_thresholds_path", "")) for r in done if str(r.get("pt_thresholds_path", ""))]
    if pt_paths:
        pt_all = pd.concat([pd.read_csv(p, sep="\t") for p in pt_paths], ignore_index=True)
        pt_all.to_csv(out_dir / "figure1_pt_thresholds.tsv", sep="\t", index=False)
        _apply_plot_style()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for g in sorted(pt_all["gens"].unique()):
            sub = pt_all[pt_all["gens"] == g].sort_values("p_threshold")
            ax.plot(
                sub["p_threshold"].to_numpy(dtype=float),
                sub["train_accuracy"].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                label=f"g={int(g)}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("P+T p-value threshold")
        ax.set_ylabel("Train accuracy")
        ax.set_title("Figure1 P+T train accuracy across thresholds")
        ax.legend(frameon=False, ncol=2)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "figure1_pt_train_accuracy_by_generation.png", dpi=240)
        plt.close(fig)
    detailed = (
        pred_df.groupby(["gens", "method", "group"], as_index=False)
        .agg(
            n=("y", "size"),
            prevalence=("y", "mean"),
            mean_prs=("prs", "mean"),
            sd_prs=("prs", "std"),
            mean_g_true=("G_true", "mean"),
            mean_y_prob=("y_prob", "mean"),
        )
        .sort_values(["gens", "method", "group"])
    )
    detailed.to_csv(out_dir / "figure1_detailed_metrics.tsv", sep="\t", index=False)
    _log("Figure1 wrote aggregated results table")
    _log_results_table(
        "[fig1] AUC ratio results (gens/method)",
        res_df.sort_values(["gens", "method"]).reset_index(drop=True),
    )
    _log_results_table(
        "[fig1] Detailed grouped stats",
        detailed.sort_values(["gens", "method", "group"]).reset_index(drop=True),
    )

    _log("Figure1 plotting outputs")
    _plot_main(res_df, out_dir)
    _plot_demography(out_dir, split_gens=int(sorted(args.gens)[len(args.gens) // 2]))
    _plot_prs_grid(prs_df.to_dict("records"), out_dir)
    _plot_pc_grid(pc_df.to_dict("records"), out_dir)
    _log("Figure1 run complete")


if __name__ == "__main__":
    main()
