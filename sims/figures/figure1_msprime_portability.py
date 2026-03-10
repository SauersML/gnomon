from __future__ import annotations

import argparse
import concurrent.futures
import math
import multiprocessing as mp
import os
import re
import signal
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import msprime
from scipy.stats import gaussian_kde
from scipy.special import expit as sigmoid
from sklearn.preprocessing import StandardScaler

THIS_DIR = Path(__file__).resolve().parent
SIMS_DIR = THIS_DIR.parent
if str(SIMS_DIR) not in sys.path:
    sys.path.append(str(SIMS_DIR))

from .common import (
    simulate_effect_size_distribution,
    diploid_index_pairs,
    sample_two_site_sets_for_maf,
    compute_pcs_risk_and_diagnostics,
    solve_intercept_for_prevalence,
    get_chr22_recomb_map,
    plink_safe_individual_id,
    stable_prs_covar_frame,
)
from methods.raw_pgs import RawPGSMethod
from methods.linear_interaction import LinearInteractionMethod
from methods.normalization import NormalizationMethod
from methods.gam_mgcv import GAMMethod
from methods.thinplate_mgcv import ThinPlateMethod
from plink_utils import run_plink_conversion
from prs_tools import BayesR, PPlusT


DIVERGENCE_GENS = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Requested scale-down by 10 from original design.
N_AFR = 1000
N_OOA_TRAIN = 1000
N_OOA_TEST = 1000
N_PCS = 5
PRS_COVAR_PCS = 3
FIG1_TASK_TIMEOUT_S = 43_200.0
FIG1_POOL_STALL_TIMEOUT_S = 57_600.0
FIG1_POOL_POLL_S = 15.0

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
    candidates: list[int] = []
    if hasattr(os, "sched_getaffinity"):
        try:
            candidates.append(max(1, len(os.sched_getaffinity(0))))  # type: ignore[attr-defined]
        except Exception:
            pass
    candidates.append(max(1, int(os.cpu_count() or 1)))
    return max(1, min(candidates))


def _detect_physical_mem_mb() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return max(1, kb // 1024)
    except Exception:
        return None
    return None


def _detect_cgroup_mem_limit_mb() -> int | None:
    limit_bytes: int | None = None
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            txt = Path(p).read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not txt or txt == "max":
            continue
        try:
            b = int(txt)
        except Exception:
            continue
        if b <= 0 or b >= (1 << 60):
            continue
        limit_bytes = b if limit_bytes is None else min(limit_bytes, b)
    if limit_bytes is None:
        return None
    return max(1, limit_bytes // (1024 * 1024))


def _detect_total_mem_mb() -> int | None:
    caps = [
        v
        for v in (
            _detect_physical_mem_mb(),
            _detect_cgroup_mem_limit_mb(),
        )
        if v is not None and v > 0
    ]
    if not caps:
        return None
    return max(1, min(caps))


def _default_work_root(out_dir: Path) -> Path:
    for base in (Path("/dev/shm"), Path("/tmp")):
        if base.exists():
            return (base / "gnomon_sims_work" / "figure1").resolve()
    return out_dir / "work"


def _install_task_termination_logging(run_id: str) -> dict[int, object]:
    previous: dict[int, object] = {}

    def _handler(signum: int, _frame: object) -> None:
        signame = signal.Signals(signum).name
        _log(f"[{run_id}] Received {signame}; terminating task")
        raise SystemExit(128 + int(signum))

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, _handler)
        except Exception:
            continue
    return previous


def _restore_task_termination_logging(previous: dict[int, object]) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except Exception:
            continue


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
    usable_mem_mb = int(0.80 * total_mem_mb) if total_mem_mb is not None else None
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


def _choose_mp_context() -> mp.context.BaseContext:
    methods = set(mp.get_all_start_methods())
    if "spawn" in methods:
        return mp.get_context("spawn")
    return mp.get_context()


def _pool_workers(ex: concurrent.futures.ProcessPoolExecutor) -> dict[int, object]:
    workers = getattr(ex, "_processes", {}) or {}
    out: dict[int, object] = {}
    for p in workers.values():
        if p is None:
            continue
        pid = getattr(p, "pid", None)
        if pid is None:
            continue
        out[int(pid)] = p
    return out


def _terminated_workers(ex: concurrent.futures.ProcessPoolExecutor) -> list[tuple[int, int | None]]:
    out: list[tuple[int, int | None]] = []
    for pid, proc in _pool_workers(ex).items():
        exitcode = getattr(proc, "exitcode", None)
        if exitcode is not None:
            out.append((pid, int(exitcode)))
    return out


def _shutdown_pool_hard(ex: concurrent.futures.ProcessPoolExecutor, reason: str) -> None:
    _log(f"Figure1 pool hard-stop: {reason}")
    workers = _pool_workers(ex)
    for proc in workers.values():
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 5.0
    for proc in workers.values():
        try:
            remaining = max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)
        except Exception:
            pass
    for proc in workers.values():
        try:
            if proc.is_alive():
                proc.kill()
        except Exception:
            pass
    ex.shutdown(wait=False, cancel_futures=True)


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
    work_root: Path,
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
    if bp_cap is not None:
        if int(bp_cap) <= 0:
            raise ValueError("bp_cap must be positive")
        # For short debug runs, use a simple capped map instead of truncating chr22's
        # HapMap map into its leading no-data interval.
        recomb_map = msprime.RateMap(position=[0.0, float(int(bp_cap))], rate=[1e-8])
    else:
        recomb_map = get_chr22_recomb_map()
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

    rng = np.random.default_rng(seed + 23)
    ooa_perm = rng.permutation(ooa_idx)
    ooa_train_idx = ooa_perm[:n_ooa_train]
    ooa_holdout_idx = ooa_perm[n_ooa_train:n_ooa_train + n_ooa_test]
    ooa_n_cal = int(len(ooa_holdout_idx) // 2)
    ooa_cal_idx = ooa_holdout_idx[:ooa_n_cal]
    ooa_test_idx = ooa_holdout_idx[ooa_n_cal:]
    afr_perm = rng.permutation(afr_idx)
    afr_n_cal = int(len(afr_perm) // 2)
    afr_cal_idx = afr_perm[:afr_n_cal]
    afr_test_idx = afr_perm[afr_n_cal:]

    _log(f"[{prefix}] Sampling PCA+causal sites in one pass")
    pca_sites, causal_sites = sample_two_site_sets_for_maf(
        ts,
        a_idx,
        b_idx,
        pca_n_sites=pca_n_sites,
        pca_maf_min=0.05,
        pca_seed=seed + 7,
        causal_n_sites=min(causal_max_sites, int(ts.num_sites)),
        causal_maf_min=0.01,
        causal_seed=seed + 17,
        log_fn=_log,
        progress_label=f"{prefix} site_scan",
    )
    _log(f"[{prefix}] Computing PCs+risk+diagnostics in one pass")
    pcs, G_true, ts_sites, causal_overlap, het_by_pop, causal_pos_1based = compute_pcs_risk_and_diagnostics(
        ts,
        a_idx,
        b_idx,
        pop,
        pca_site_ids=pca_sites,
        n_pcs=N_PCS,
        pca_seed=seed + 11,
        causal_site_ids=causal_sites,
        real_effects=pgs_effects,
        causal_seed=seed + 19,
        log_fn=_log,
        progress_label=f"{prefix} feature_build",
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
    y = rng.binomial(1, p).astype(np.int32)

    rows = []
    ooa_train_set = set(ooa_train_idx.tolist())
    ooa_cal_set = set(ooa_cal_idx.tolist())
    ooa_test_set = set(ooa_test_idx.tolist())
    afr_cal_set = set(afr_cal_idx.tolist())
    afr_test_set = set(afr_test_idx.tolist())
    for i in range(len(pop)):
        group = "unused"
        if i in ooa_train_set:
            group = "OOA_train"
        elif i in ooa_cal_set:
            group = "OOA_cal"
        elif i in ooa_test_set:
            group = "OOA_test"
        elif i in afr_cal_set:
            group = "AFR_cal"
        elif i in afr_test_set:
            group = "AFR_test"
        iid = plink_safe_individual_id(i)
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

    def _write_vcf_stream(handle) -> None:
        _log(f"[{prefix}] Streaming VCF to PLINK")
        names = [plink_safe_individual_id(i) for i in range(ts.num_individuals)]
        ts.write_vcf(handle, individual_names=names, position_transform=lambda x: np.asarray(x) + 1)

    _log(f"[{prefix}] Converting simulated variants to PLINK")
    run_plink_conversion(
        _write_vcf_stream,
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
    pred_ids = df.loc[df["group"].isin(["OOA_cal", "AFR_cal", "OOA_test", "AFR_test"]), "IID"].astype(str).tolist()

    train_keep = work_dir / "train.keep"
    test_keep = work_dir / "test.keep"
    pd.DataFrame({"FID": [iid_to_fid[i] for i in train_ids], "IID": train_ids}).to_csv(train_keep, sep="\t", header=False, index=False)
    pd.DataFrame({"FID": [iid_to_fid[i] for i in pred_ids], "IID": pred_ids}).to_csv(test_keep, sep="\t", header=False, index=False)

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

    pd.DataFrame({
        "FID": [iid_to_fid[i] for i in train_df["IID"].astype(str)],
        "IID": train_df["IID"].astype(str),
        "y": train_df["y"].astype(int),
    }).to_csv(work_dir / "train.phen", sep=" ", header=False, index=False)

    covar_df, kept_cols, dropped_cols = stable_prs_covar_frame(
        train_df,
        iid_to_fid=iid_to_fid,
        max_pcs=PRS_COVAR_PCS,
    )
    if dropped_cols:
        _log(f"[{Path(prefix).name}] PRS covariates dropped: {', '.join(dropped_cols)}")
    _log(f"[{Path(prefix).name}] PRS covariates kept: {', '.join(kept_cols)}")
    covar_df.to_csv(work_dir / "train.covar", sep=" ", header=False, index=False)

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


def _reuse_cache_dir(base_dir: Path) -> Path:
    return base_dir / "_reuse_cache"


def _cache_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _publish_reuse_cache(prefix: Path, out_dir: Path, run_id: str) -> Path:
    cache_dir = _reuse_cache_dir(out_dir)
    needed = [Path(f"{prefix}.bed"), Path(f"{prefix}.bim"), Path(f"{prefix}.fam"), Path(f"{prefix}.tsv")]
    miss = [str(p) for p in needed if not p.exists()]
    if miss:
        raise FileNotFoundError(f"[{run_id}] Cannot publish reuse cache; missing files: {miss}")
    for src in needed:
        _cache_link_or_copy(src, cache_dir / src.name)
    _log(f"[{run_id}] Reuse cache updated at {cache_dir}")
    return cache_dir


def _resolve_existing_source(g: int, seed: int, out_dir: Path, use_existing_dir: Path | None) -> tuple[Path, Path]:
    stem = f"fig1_g{g}_s{seed}"
    candidates: list[Path] = []
    if use_existing_dir is not None:
        candidates.append(use_existing_dir)
        candidates.append(_reuse_cache_dir(use_existing_dir))
    candidates.append(_reuse_cache_dir(out_dir))
    candidates.append(out_dir)

    missing_paths: list[str] = []
    for base in candidates:
        prefix = base / stem
        sim_tsv = base / f"{stem}.tsv"
        needed = [Path(f"{prefix}.bed"), Path(f"{prefix}.bim"), Path(f"{prefix}.fam"), sim_tsv]
        if all(p.exists() for p in needed):
            return prefix, sim_tsv
        missing_paths.extend(str(p) for p in needed if not p.exists())

    uniq_missing = sorted(set(missing_paths))
    raise FileNotFoundError(
        f"[fig1_g{g}_s{seed}] --use-existing requested, but required files were not found "
        f"in searched locations. Missing candidates include: {uniq_missing}"
    )


def _fit_and_predict_methods(train_df: pd.DataFrame, test_df: pd.DataFrame, train_prs: np.ndarray, test_prs: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    prs_scaler = StandardScaler(with_mean=True, with_std=True)
    P_train = prs_scaler.fit_transform(train_prs.reshape(-1, 1)).ravel()
    P_test = prs_scaler.transform(test_prs.reshape(-1, 1)).ravel()
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

    gm = GAMMethod(k_pgs=4, k_pc=4, k_interaction=3, use_ti=True)
    gm.fit(P_train, PC_train, y_train)
    out["pspline"] = gm.predict_proba(P_test, PC_test)
    tp = ThinPlateMethod(k_joint=80)
    tp.fit(P_train, PC_train, y_train)
    out["duchon"] = tp.predict_proba(P_test, PC_test)

    return out


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import brier_score_loss

    if y.size == 0:
        return np.nan
    return float(brier_score_loss(y.astype(float), np.clip(p.astype(float), 0.0, 1.0)))


def _assert_noncollapsed_prs(labels: np.ndarray, prs: np.ndarray, context: str) -> None:
    lab = np.asarray(labels, dtype=object)
    x = np.asarray(prs, dtype=float)
    if lab.shape[0] != x.shape[0]:
        raise RuntimeError(f"{context}: label/score length mismatch ({lab.shape[0]} vs {x.shape[0]})")
    if x.size == 0:
        raise RuntimeError(f"{context}: empty PRS vector")

    finite = np.isfinite(x)
    if not np.all(finite):
        raise RuntimeError(f"{context}: non-finite PRS values detected")

    rows: list[dict[str, object]] = []
    collapsed: list[str] = []
    for g in sorted(pd.unique(lab)):
        mask = lab == g
        s = x[mask]
        n = int(s.size)
        n_unique = int(np.unique(s).size) if n > 0 else 0
        sd = float(np.std(s, ddof=1)) if n > 1 else float("nan")
        rows.append({"group": str(g), "n": n, "n_unique_prs": n_unique, "sd_prs": sd})
        if n > 1 and n_unique <= 1:
            collapsed.append(str(g))

    overall_unique = int(np.unique(x).size)
    _log_results_table(
        f"[{context}] PRS variation by group",
        pd.DataFrame(rows).sort_values("group").reset_index(drop=True),
    )
    _log(f"[{context}] Overall PRS unique values: {overall_unique}")

    if overall_unique <= 1 or collapsed:
        raise RuntimeError(
            f"{context}: collapsed PRS detected "
            f"(overall_unique={overall_unique}, collapsed_groups={collapsed})"
        )


def _plot_main(df: pd.DataFrame, out_dir: Path) -> None:
    _apply_plot_style()
    method_colors = {
        "raw": CB["blue"],
        "normalized": CB["green"],
        "linear": CB["orange"],
        "pspline": CB["purple"],
        "duchon": CB["vermillion"],
    }
    prs_sources = sorted(df["prs_source"].unique()) if "prs_source" in df.columns else ["estimated"]
    for prs_source in prs_sources:
        sub_df = df[df["prs_source"] == prs_source] if "prs_source" in df.columns else df
        source_suffix = "" if prs_source == "estimated" else f"_{prs_source}"
        source_title = f" ({prs_source} PRS)" if len(prs_sources) > 1 else ""

        plt.figure(figsize=(10, 6))
        for m in sorted(sub_df["method"].unique()):
            sub = sub_df[sub_df["method"] == m].sort_values("gens")
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
        plt.ylabel(f"AUC ratio (OoA test / AFR test){source_title}")
        ax = plt.gca()
        _style_axes(ax)
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"figure1_auc_ratio{source_suffix}.png", dpi=240)
        plt.close()

        plt.figure(figsize=(10, 6))
        for m in sorted(sub_df["method"].unique()):
            sub = sub_df[sub_df["method"] == m].sort_values("gens")
            plt.plot(
                sub["gens"],
                sub["brier_ratio"],
                marker="o",
                markersize=4.5,
                linewidth=2.0,
                label=m,
                color=method_colors.get(m, CB["black"]),
            )
        plt.xscale("log")
        plt.xlabel("Generations of divergence")
        plt.ylabel(f"Brier ratio (OoA test / AFR test){source_title}")
        ax = plt.gca()
        _style_axes(ax)
        plt.legend(frameon=False, loc="best")
        plt.tight_layout()
        plt.savefig(out_dir / f"figure1_brier_ratio{source_suffix}.png", dpi=240)
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
    prs_sources = sorted(df["prs_source"].unique()) if "prs_source" in df.columns else ["estimated"]
    for prs_source in prs_sources:
        sub_df = df[df["prs_source"] == prs_source] if "prs_source" in df.columns else df
        gens = sorted(sub_df["gens"].unique())
        ncols = 4
        nrows = int(math.ceil(len(gens) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for i, g in enumerate(gens):
            ax = axes[i // ncols][i % ncols]
            gsub = sub_df[sub_df["gens"] == g]
            for pop, color in [("OOA_test", CB["blue"]), ("AFR_test", CB["orange"])]:
                x = gsub.loc[gsub["group"] == pop, "prs"].to_numpy(dtype=float)
                if x.size:
                    _draw_kde(ax, x, color=color, label=pop)
            ax.text(0.03, 0.95, f"g={g}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
            _style_axes(ax)
        for j in range(len(gens), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        axes[0][0].legend(frameon=False, ncol=2, loc="upper right")
        fig.supxlabel(f"PRS ({prs_source})", y=0.02)
        fig.supylabel("Density", x=0.01)
        fig.tight_layout()
        suffix = "" if prs_source == "estimated" else f"_{prs_source}"
        fig.savefig(out_dir / f"figure1_prs_distributions_grid{suffix}.png", dpi=240)
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


def _ensure_calibration_groups(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Backfill *_cal groups for legacy cached simulations that only have *_test."""
    if "group" not in df.columns:
        raise RuntimeError("Simulation table must include group column.")
    has_cal = bool(df["group"].astype(str).str.endswith("_cal").any())
    if has_cal:
        return df

    out = df.copy()
    rng = np.random.default_rng(int(seed) + 29)
    for pop in ("OOA", "AFR"):
        test_label = f"{pop}_test"
        cal_label = f"{pop}_cal"
        idx = out.index[out["group"].astype(str) == test_label].to_numpy()
        if idx.size == 0:
            continue
        perm = rng.permutation(idx)
        n_cal = int(perm.size // 2)
        out.loc[perm[:n_cal], "group"] = cal_label
        out.loc[perm[n_cal:], "group"] = test_label

    if not out["group"].astype(str).str.endswith("_cal").any():
        raise RuntimeError("Could not derive calibration groups from cached simulation table.")
    _log(
        "[fig1] Upgraded cached split to include calibration groups: "
        f"OOA_cal={int(np.count_nonzero(out['group'].astype(str) == 'OOA_cal'))}, "
        f"AFR_cal={int(np.count_nonzero(out['group'].astype(str) == 'AFR_cal'))}"
    )
    return out


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
    ax.plot(x, metrics_df["train_accuracy"].to_numpy(dtype=float), marker="o", linewidth=2, color=CB["blue"], label="Accuracy")
    if "train_balanced_accuracy" in metrics_df.columns:
        ax.plot(
            x,
            metrics_df["train_balanced_accuracy"].to_numpy(dtype=float),
            marker="s",
            linewidth=2,
            color=CB["orange"],
            label="Balanced Accuracy",
        )
    if "train_auc" in metrics_df.columns:
        ax.plot(
            x,
            metrics_df["train_auc"].to_numpy(dtype=float),
            marker="^",
            linewidth=2,
            color=CB["green"],
            label="AUC",
        )
    ax.set_xscale("log")
    ax.set_xlabel("P+T p-value threshold")
    ax.set_ylabel("Train metric")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def _collect_existing_generation_records(out_dir: Path) -> list[dict[str, object]]:
    patt = re.compile(r"^fig1_g(?P<g>\d+)_s(?P<s>\d+)\.results\.tsv$")
    recs: list[dict[str, object]] = []
    for res_path in sorted(out_dir.glob("fig1_g*_s*.results.tsv")):
        m = patt.match(res_path.name)
        if m is None:
            continue
        g = int(m.group("g"))
        s = int(m.group("s"))
        stem = out_dir / f"fig1_g{g}_s{s}"
        prs_path = Path(f"{stem}.prs.tsv")
        pc_path = Path(f"{stem}.pc.tsv")
        pred_path = Path(f"{stem}.predictions.tsv")
        if not (prs_path.exists() and pc_path.exists() and pred_path.exists()):
            continue
        pt_path = Path(f"{stem}.pt_thresholds.tsv")
        recs.append(
            {
                "gens": g,
                "seed": s,
                "res_path": str(res_path),
                "prs_path": str(prs_path),
                "pc_path": str(pc_path),
                "pred_path": str(pred_path),
                "pt_thresholds_path": str(pt_path) if pt_path.exists() else "",
            }
        )
    recs.sort(key=lambda x: (int(x["gens"]), int(x["seed"])))
    return recs


def _discover_recoverable_existing_generations(out_dir: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    patt = re.compile(r"^fig1_g(?P<g>\d+)_s(?P<s>\d+)\.bed$")
    recoverable: list[tuple[int, int]] = []
    missing_sim_tsv: list[tuple[int, int]] = []
    for bed_path in sorted(out_dir.glob("fig1_g*_s*.bed")):
        m = patt.match(bed_path.name)
        if m is None:
            continue
        g = int(m.group("g"))
        s = int(m.group("s"))
        stem = out_dir / f"fig1_g{g}_s{s}"
        needed_plink = [Path(f"{stem}.bed"), Path(f"{stem}.bim"), Path(f"{stem}.fam")]
        if not all(p.exists() for p in needed_plink):
            continue
        sim_tsv = Path(f"{stem}.tsv")
        if sim_tsv.exists():
            recoverable.append((g, s))
        else:
            missing_sim_tsv.append((g, s))
    return recoverable, missing_sim_tsv


def _aggregate_and_plot(done: list[dict[str, object]], out_dir: Path, demography_split_gens: int | None = None) -> None:
    if not done:
        raise RuntimeError("No per-generation outputs available to aggregate.")
    done = sorted(done, key=lambda x: (int(x["gens"]), int(x["seed"])))
    res_df = pd.concat([pd.read_csv(str(r["res_path"]), sep="\t") for r in done], ignore_index=True).sort_values(["prs_source", "method", "gens"])
    prs_df = pd.concat([pd.read_csv(str(r["prs_path"]), sep="\t") for r in done], ignore_index=True)
    pc_df = pd.concat([pd.read_csv(str(r["pc_path"]), sep="\t") for r in done], ignore_index=True)
    pred_df = pd.concat([pd.read_csv(str(r["pred_path"]), sep="\t") for r in done], ignore_index=True)
    res_df.to_csv(out_dir / "figure1_auc_ratio.tsv", sep="\t", index=False)
    res_df.to_csv(out_dir / "figure1_auc_brier_ratio.tsv", sep="\t", index=False)
    _log("Figure1 wrote figure1_auc_ratio.tsv and figure1_auc_brier_ratio.tsv")
    pred_df.to_csv(out_dir / "figure1_predictions.tsv", sep="\t", index=False)
    pt_paths = [str(r.get("pt_thresholds_path", "")) for r in done if str(r.get("pt_thresholds_path", ""))]
    if pt_paths:
        pt_all = pd.concat([pd.read_csv(p, sep="\t") for p in pt_paths], ignore_index=True)
        pt_all.to_csv(out_dir / "figure1_pt_thresholds.tsv", sep="\t", index=False)
        _apply_plot_style()
        fig, ax = plt.subplots(figsize=(9, 5.5))
        metric_col = "train_auc" if "train_auc" in pt_all.columns else "train_accuracy"
        metric_label = "Train AUC" if metric_col == "train_auc" else "Train accuracy"
        for g in sorted(pt_all["gens"].unique()):
            sub = pt_all[pt_all["gens"] == g].sort_values("p_threshold")
            ax.plot(
                sub["p_threshold"].to_numpy(dtype=float),
                sub[metric_col].to_numpy(dtype=float),
                marker="o",
                linewidth=1.8,
                label=f"g={int(g)}",
            )
        ax.set_xscale("log")
        ax.set_xlabel("P+T p-value threshold")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Figure1 P+T {metric_label.lower()} across thresholds")
        ax.legend(frameon=False, ncol=2)
        _style_axes(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "figure1_pt_train_accuracy_by_generation.png", dpi=240)
        plt.close(fig)
    detailed = (
        pred_df.groupby(["gens", "prs_source", "method", "group"], as_index=False)
        .agg(
            n=("y", "size"),
            prevalence=("y", "mean"),
            mean_prs=("prs", "mean"),
            sd_prs=("prs", "std"),
            mean_g_true=("G_true", "mean"),
            mean_y_prob=("y_prob", "mean"),
        )
        .sort_values(["gens", "prs_source", "method", "group"])
    )
    detailed.to_csv(out_dir / "figure1_detailed_metrics.tsv", sep="\t", index=False)
    _log("Figure1 wrote aggregated results table")
    _log_results_table(
        "[fig1] AUC/Brier ratio results (gens/method)",
        res_df.sort_values(["gens", "prs_source", "method"]).reset_index(drop=True),
    )
    _log_results_table(
        "[fig1] Detailed grouped stats",
        detailed.sort_values(["gens", "prs_source", "method", "group"]).reset_index(drop=True),
    )
    _log("Figure1 plotting outputs")
    _plot_main(res_df, out_dir)
    if demography_split_gens is None:
        avail = sorted({int(r["gens"]) for r in done})
        demography_split_gens = int(avail[len(avail) // 2])
    _plot_demography(out_dir, split_gens=int(demography_split_gens))
    _plot_prs_grid(prs_df.to_dict("records"), out_dir)
    _plot_pc_grid(pc_df.to_dict("records"), out_dir)
    _log("Figure1 run complete")


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
    _log(f"[fig1_g{g}_s{seed}] Runtime resources set (threads={threads_per_job}, memory_mb={memory_mb_per_job})")

    run_id = f"fig1_g{g}_s{seed}"
    previous_signal_handlers = _install_task_termination_logging(run_id)
    prefix_path = out_dir / run_id
    prefix = str(prefix_path)
    source_prefix = prefix_path
    try:
        def _simulate_and_publish() -> pd.DataFrame:
            df_local = _simulate_for_generation(
                g,
                seed,
                pgs_effects,
                out_dir,
                work_root,
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
            _publish_reuse_cache(prefix=prefix_path, out_dir=out_dir, run_id=run_id)
            return df_local

        reused_existing = False
        if use_existing:
            try:
                source_prefix, sim_tsv = _resolve_existing_source(g, seed, out_dir, use_existing_dir)
                _log(f"[{run_id}] Reusing existing simulation/PLINK files from {source_prefix.parent}")
                df = pd.read_csv(sim_tsv, sep="\t")
                reused_existing = True
            except Exception as e:
                _log(f"[{run_id}] Existing artifacts unavailable/unreadable; regenerating this generation ({type(e).__name__}: {e})")
                source_prefix = prefix_path
                df = _simulate_and_publish()
        else:
            source_prefix = prefix_path
            df = _simulate_and_publish()
        df = _ensure_calibration_groups(df, seed=seed)
        work = work_root / f"fig1_g{g}_s{seed}_work"
        try:
            train_prefix, test_prefix, train_phen, freq_file = _prepare_bayesr_files(
                df,
                str(source_prefix),
                work,
                plink_threads=threads_per_job,
                plink_memory_mb=memory_mb_per_job,
            )
        except Exception as e:
            if use_existing and reused_existing:
                _log(
                    f"[{run_id}] Existing PLINK/TSV artifacts failed during PRS prep; "
                    f"regenerating and retrying ({type(e).__name__}: {e})"
                )
                source_prefix = prefix_path
                df = _ensure_calibration_groups(_simulate_and_publish(), seed=seed)
                train_prefix, test_prefix, train_phen, freq_file = _prepare_bayesr_files(
                    df,
                    str(source_prefix),
                    work,
                    plink_threads=threads_per_job,
                    plink_memory_mb=memory_mb_per_job,
                )
            else:
                raise
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
                f"(train_accuracy={float(pt_meta['best_train_accuracy']):.4f}, "
                f"train_balanced_accuracy={float(pt_meta.get('best_train_balanced_accuracy', float('nan'))):.4f}, "
                f"train_auc={float(pt_meta.get('best_train_auc', float('nan'))):.4f}, "
                f"n_snps={int(pt_meta['best_n_snps'])})"
            )
            _log(f"[fig1_g{g}_s{seed}] P+T fit/scoring complete")
    except BaseException as e:
        _log(f"[{run_id}] Task aborted ({type(e).__name__}: {e})")
        raise
    finally:
        _restore_task_termination_logging(previous_signal_handlers)

    train_df = df[df["group"] == "OOA_train"].copy()
    pred_df = df[df["group"].isin(["OOA_cal", "AFR_cal", "OOA_test", "AFR_test"])].copy()

    train_df = train_df.set_index("IID").loc[train_scores["IID"].astype(str)].reset_index()
    pred_scores = test_scores.copy()
    pred_df = pred_df.set_index("IID").loc[pred_scores["IID"].astype(str)].reset_index()
    cal_df = pred_df[pred_df["group"].isin(["OOA_cal", "AFR_cal"])].copy()
    test_df = pred_df[pred_df["group"].isin(["OOA_test", "AFR_test"])].copy()
    cal_scores = pred_scores[pred_scores["IID"].astype(str).isin(cal_df["IID"].astype(str))].copy()
    test_scores = pred_scores[pred_scores["IID"].astype(str).isin(test_df["IID"].astype(str))].copy()
    cal_df = cal_df.set_index("IID").loc[cal_scores["IID"].astype(str)].reset_index()
    test_df = test_df.set_index("IID").loc[test_scores["IID"].astype(str)].reset_index()
    _assert_noncollapsed_prs(
        train_df["group"].to_numpy(),
        train_scores["PRS"].to_numpy(dtype=float),
        context=f"fig1_g{g}_s{seed} train",
    )
    _assert_noncollapsed_prs(
        pred_df["group"].to_numpy(),
        pred_scores["PRS"].to_numpy(dtype=float),
        context=f"fig1_g{g}_s{seed} pred",
    )
    _assert_noncollapsed_prs(
        cal_df["group"].to_numpy(),
        cal_scores["PRS"].to_numpy(dtype=float),
        context=f"fig1_g{g}_s{seed} cal",
    )
    _assert_noncollapsed_prs(
        test_df["group"].to_numpy(),
        test_scores["PRS"].to_numpy(dtype=float),
        context=f"fig1_g{g}_s{seed} test",
    )

    result_rows = []
    prs_sources = {
        "estimated": {
            "train": train_scores["PRS"].to_numpy(dtype=float),
            "pred": pred_scores["PRS"].to_numpy(dtype=float),
            "cal": cal_scores["PRS"].to_numpy(dtype=float),
            "test": test_scores["PRS"].to_numpy(dtype=float),
        },
        "oracle": {
            "train": train_df["G_true"].to_numpy(dtype=float),
            "pred": pred_df["G_true"].to_numpy(dtype=float),
            "cal": cal_df["G_true"].to_numpy(dtype=float),
            "test": test_df["G_true"].to_numpy(dtype=float),
        },
    }

    predictions_by_source: dict[str, dict[str, np.ndarray]] = {}
    prs_parts: list[pd.DataFrame] = []
    for prs_source, prs_vals in prs_sources.items():
        _assert_noncollapsed_prs(
            train_df["group"].to_numpy(),
            prs_vals["train"],
            context=f"fig1_g{g}_s{seed} train ({prs_source})",
        )
        _assert_noncollapsed_prs(
            pred_df["group"].to_numpy(),
            prs_vals["pred"],
            context=f"fig1_g{g}_s{seed} pred ({prs_source})",
        )
        _assert_noncollapsed_prs(
            cal_df["group"].to_numpy(),
            prs_vals["cal"],
            context=f"fig1_g{g}_s{seed} cal ({prs_source})",
        )
        _assert_noncollapsed_prs(
            test_df["group"].to_numpy(),
            prs_vals["test"],
            context=f"fig1_g{g}_s{seed} test ({prs_source})",
        )
        pred = _fit_and_predict_methods(cal_df, test_df, prs_vals["cal"], prs_vals["test"])
        predictions_by_source[prs_source] = pred
        _log(f"[fig1_g{g}_s{seed}] Method fitting/prediction complete (prs_source={prs_source})")

        o = test_df["group"] == "OOA_test"
        a = test_df["group"] == "AFR_test"
        for method, y_prob in pred.items():
            auc_o = _auc(test_df.loc[o, "y"].to_numpy(), y_prob[o.to_numpy()])
            auc_a = _auc(test_df.loc[a, "y"].to_numpy(), y_prob[a.to_numpy()])
            auc_ratio = float(auc_o / auc_a) if np.isfinite(auc_o) and np.isfinite(auc_a) and auc_a > 0 else np.nan
            brier_o = _brier(test_df.loc[o, "y"].to_numpy(), y_prob[o.to_numpy()])
            brier_a = _brier(test_df.loc[a, "y"].to_numpy(), y_prob[a.to_numpy()])
            brier_ratio = float(brier_o / brier_a) if np.isfinite(brier_o) and np.isfinite(brier_a) and brier_a > 0 else np.nan
            _log(
                f"[fig1_g{g}_s{seed}] prs_source={prs_source} method={method} "
                f"auc_ooa={auc_o:.4f} auc_afr={auc_a:.4f} auc_ratio={auc_ratio:.4f} "
                f"brier_ooa={brier_o:.4f} brier_afr={brier_a:.4f} brier_ratio={brier_ratio:.4f}"
            )
            result_rows.append(
                {
                    "gens": int(g),
                    "seed": int(seed),
                    "prs_source": prs_source,
                    "method": method,
                    "n_train_prs": int(len(train_df)),
                    "n_calibration": int(len(cal_df)),
                    "n_test_total": int(len(test_df)),
                    "n_test_ooa": int(np.count_nonzero(o.to_numpy())),
                    "n_test_afr": int(np.count_nonzero(a.to_numpy())),
                    "auc_ooa": auc_o,
                    "auc_afr": auc_a,
                    "auc_ratio": auc_ratio,
                    "brier_ooa": brier_o,
                    "brier_afr": brier_a,
                    "brier_ratio": brier_ratio,
                }
            )

        part = test_df[["IID", "group"]].copy()
        part["IID"] = part["IID"].astype(str)
        part.insert(0, "gens", int(g))
        part["prs_source"] = prs_source
        part["prs"] = prs_vals["test"]
        prs_parts.append(part)

    prs_df = pd.concat(prs_parts, ignore_index=True)

    pc_df = df[["group", "pc1", "pc2"]].copy()
    pc_df.insert(0, "gens", int(g))

    pred_rows = []
    for prs_source, pred in predictions_by_source.items():
        test_prs = prs_sources[prs_source]["test"]
        for method, y_prob in pred.items():
            part = test_df[["IID", "group", "pop_label", "y", "G_true", "pc1", "pc2", "pc3", "pc4", "pc5"]].copy()
            part.insert(0, "seed", int(seed))
            part.insert(0, "gens", int(g))
            part["prs_source"] = prs_source
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
    parser.add_argument("--cache", default="sims/.cache", help="Deprecated: external PGS cache is no longer used.")
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
    parser.add_argument(
        "--task-timeout-s",
        type=float,
        default=FIG1_TASK_TIMEOUT_S,
        help="Per-generation hard timeout in seconds (default: 43200 = 12h).",
    )
    parser.add_argument(
        "--pool-stall-timeout-s",
        type=float,
        default=FIG1_POOL_STALL_TIMEOUT_S,
        help="Fail the run if no task completes for this many seconds (default: 57600 = 16h).",
    )
    parser.add_argument(
        "--pool-poll-s",
        type=float,
        default=FIG1_POOL_POLL_S,
        help="Executor polling interval in seconds (default: 15).",
    )
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep per-generation PLINK/GCTB intermediate files.")
    parser.add_argument("--bayesr", action="store_true", help="Use BayesR backend (default is fast P+T).")
    parser.add_argument("--use-existing", action="store_true", help="Reuse existing fig1_g*_s* PLINK/TSV files; skip simulation.")
    parser.add_argument("--use-existing-dir", default=None, help="Directory containing existing fig1_g*_s* PLINK/TSV files.")
    parser.add_argument(
        "--force-figs",
        action="store_true",
        help="Force regeneration of Figure1 aggregate tables/plots from whatever per-generation outputs already exist.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root) if args.work_root else _default_work_root(out_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    _log("Figure1 spline backends: pspline + duchon (mgcv)")
    _log(f"Figure1 PRS backend: {'BayesR' if args.bayesr else 'P+T'}")

    if bool(args.force_figs):
        existing = _collect_existing_generation_records(out_dir)
        if not existing:
            recoverable, missing_sim_tsv = _discover_recoverable_existing_generations(out_dir)
            if recoverable:
                _log(
                    f"Figure1 force-figs mode: recovering downstream outputs from "
                    f"{len(recoverable)} existing PLINK+TSV generation(s)."
                )
                pgs_effects = simulate_effect_size_distribution(
                    n_effects=200000,
                    seed=int(args.seed) + 4049,
                )
                jobs, threads_per_job, memory_mb_per_job = _resource_plan(
                    n_tasks=len(recoverable),
                    total_threads_override=args.total_threads,
                    total_mem_mb_override=args.memory_mb_total,
                )
                _set_runtime_thread_env(threads_per_job)
                recovered: list[dict[str, object]] = []
                for g, s in recoverable:
                    rec = _run_generation_task(
                        {
                            "g": int(g),
                            "seed": int(s),
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
                            "use_existing": True,
                            "use_existing_dir": str(out_dir),
                        }
                    )
                    recovered.append(rec)
                    _log(f"Figure1 force-figs recovered generation g={g} seed={s}")
                existing = _collect_existing_generation_records(out_dir)
                if existing:
                    _log(
                        f"Figure1 force-figs mode: recovered {len(recovered)} generation(s); regenerating aggregate outputs."
                    )
                    _aggregate_and_plot(existing, out_dir=out_dir, demography_split_gens=None)
                    return
            missing_bits = (
                f" Missing simulation tables for: {', '.join([f'g={g},seed={s}' for g, s in missing_sim_tsv])}."
                if missing_sim_tsv
                else ""
            )
            raise RuntimeError(
                f"No recoverable Figure1 per-generation outputs found in {out_dir}."
                " Full regeneration requires existing fig1_g*_s*.results.tsv/.prs.tsv/.pc.tsv/.predictions.tsv, "
                "or recovery inputs fig1_g*_s*.bed/.bim/.fam plus fig1_g*_s*.tsv."
                f"{missing_bits}"
            )
        _log(f"Figure1 force-figs mode: found {len(existing)} generation result sets; regenerating aggregate outputs.")
        _aggregate_and_plot(existing, out_dir=out_dir, demography_split_gens=None)
        return

    pgs_effects = simulate_effect_size_distribution(
        n_effects=200000,
        seed=int(args.seed) + 4049,
    )
    _log(f"Figure1 using runtime-simulated effect-size distribution (n={len(pgs_effects)})")
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
    mp_ctx = _choose_mp_context()
    _log(f"Figure1 process start method: {mp_ctx.get_start_method()}")
    ex = concurrent.futures.ProcessPoolExecutor(
        max_workers=jobs,
        mp_context=mp_ctx,
        max_tasks_per_child=1,
    )
    pool_failed = False
    try:
        future_to_task = {ex.submit(_run_generation_task, t): t for t in tasks}
        pending = set(future_to_task.keys())
        submitted_at = {fut: time.monotonic() for fut in pending}
        last_completion = time.monotonic()
        task_timeout_s = max(1.0, float(args.task_timeout_s))
        stall_timeout_s = max(1.0, float(args.pool_stall_timeout_s))
        poll_s = max(1.0, float(args.pool_poll_s))

        while pending:
            finished, pending = concurrent.futures.wait(
                pending,
                timeout=poll_s,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            now = time.monotonic()
            if finished:
                for fut in finished:
                    rec = fut.result()
                    done.append(rec)
                    _log(f"Completed generation g={rec['gens']} seed={rec['seed']}")
                    last_completion = now
                continue

            dead_workers = _terminated_workers(ex)
            if dead_workers:
                pool_failed = True
                _shutdown_pool_hard(
                    ex,
                    reason=(
                        "detected terminated worker(s): "
                        + ", ".join([f"pid={pid},exit={code}" for pid, code in dead_workers])
                    ),
                )
                raise RuntimeError(
                    "Figure1 worker process exited unexpectedly; aborted to prevent deadlock. "
                    f"terminated_workers={dead_workers}"
                )

            overdue = []
            for fut in pending:
                age_s = now - submitted_at[fut]
                if age_s > task_timeout_s:
                    task = future_to_task[fut]
                    overdue.append((int(task["g"]), int(task["seed"]), age_s))
            if overdue:
                pool_failed = True
                _shutdown_pool_hard(
                    ex,
                    reason=(
                        "generation task timeout(s): "
                        + ", ".join([f"g={g},seed={s},age_s={age:.1f}" for g, s, age in overdue])
                    ),
                )
                raise TimeoutError(
                    "Figure1 task timeout hit; aborted to prevent indefinite hang. "
                    + ", ".join([f"g={g},seed={s},age_s={age:.1f}" for g, s, age in overdue])
                )

            idle_s = now - last_completion
            if idle_s > stall_timeout_s:
                pool_failed = True
                _shutdown_pool_hard(
                    ex,
                    reason=f"no completed tasks for {idle_s:.1f}s (stall timeout={stall_timeout_s:.1f}s)",
                )
                raise TimeoutError(
                    "Figure1 pool stalled with no completed tasks; "
                    f"idle_for_s={idle_s:.1f}, stall_timeout_s={stall_timeout_s:.1f}"
                )
    except BaseException as e:
        _log(f"Figure1 main loop aborted ({type(e).__name__}: {e})")
        raise
    finally:
        ex.shutdown(wait=not pool_failed, cancel_futures=pool_failed)

    _aggregate_and_plot(done, out_dir=out_dir, demography_split_gens=int(sorted(args.gens)[len(args.gens) // 2]))


if __name__ == "__main__":
    main()
