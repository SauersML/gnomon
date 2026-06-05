"""Generate real-P+T portability datasets for the ancestry-calibration study.

Hard rules:
- only serial1d and grid2d custom msprime demographies
- real P+T only: GWAS + LD clump + p-threshold + score
- binary outcome only
- split is explicit and disjoint:
  * pgs_train: large cohort from one training deme; trains P+T only
  * model_train: all demes; fits risk models only
  * test: all demes; reported results only
- GWAS + P+T weights AND PGS standardization use split == "pgs_train" ONLY
- binary ground truth is p_true
- no within-deme/within-stratum AUC computed here (only GLOBAL pooled)

Real P+T needs GWAS power:
- large single-deme PGS training cohort
- ~100 Mb of sequence as CHUNKS x CHUNK_BP independent segments (real LD for clumping)
- PGS signal decay is EMERGENT from LD/AF drift (no rho floor); we verify it stays
  non-degenerate and report it honestly.

Usage: gen_real_pt.py <dem: serial1d|grid2d> [seed]
Writes to sims/results_hpc/ancestry_calibration/data/:
<dem>_{phenoA,phenoB}_realpt_s<seed>.parquet (+ .csv, .sanity.tsv)
and sidecar JSON files (GLOBAL/pooled diagnostics only).
"""
from __future__ import annotations

import argparse
import json
import os
# Single-threaded BLAS, set before numpy/BLAS loads: the genotype streaming forks a
# ProcessPool, which corrupts a multithreaded BLAS threadpool in the parent and
# deadlocks the subsequent PCA SVD. Fork-safety fix (plink threads unaffected; PCA
# single-threaded is plenty fast), not a configuration knob.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import shutil
import subprocess
import time
from pathlib import Path

import msprime
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "sims" / "results_hpc" / "ancestry_calibration" / "data"
PLINK1 = "/common/software/install/migrated/plink/1.90b6.10/bin/plink"
PLINK2 = "/users/0/sauer354/bin/plink2"

MU = 1.25e-8
RECOMB = 1.0e-8
N_PGS_TRAIN = 100000       # large single-deme cohort for GWAS/P+T only (20x)
N_MODEL_TRAIN = 500        # calibration/model-training rows per deme (2x)
N_TEST = 500               # reported-test rows per deme (2x)
PREV = 0.15
SIGMA_E = 1.0              # environmental SD on the liability scale

# Many small independent-LD chunks, processed in PARALLEL across the task's cores
# (stream_geno auto-scales workers to the allocation). A chunk's genotype_matrix is
# int32, so ~200 kb (~9e3 sites, ~10 GiB) lets several workers run concurrently.
# 500 x 200 kb = the same ~100 Mb of sequence.
DEFAULT_CHUNKS = 800
DEFAULT_CHUNK_BP = 125_000
DEFAULT_THREADS = 16      # plink threads; matches the per-task core allocation (-c16)
DEFAULT_MEM_MB = 64_000   # plink --memory cap/job
N_CAUSAL = 4500

P_THRESHOLDS = [5e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
CLUMP_R2 = 0.1
CLUMP_KB = 250
MIN_CLUMP_SNPS = 20


# ---------------- demographies ----------------
# These are stress-test human-scale demographies. The far-deme divergence is not
# assigned directly; it is measured from the simulated common variants and written
# to the sidecar as fst_train_vs_far.
def dem_serial1d(D=10, N=3000, Nanc=10000, m=5e-4, split_step=500, T0=250, migration_scale=1.0):
    m = m * migration_scale     # <1 = more isolation (harder portability), >1 = less
    d = msprime.Demography()
    for k in range(D):
        d.add_population(name=f"d{k}", initial_size=N)
    d.add_population(name="ANC", initial_size=Nanc)
    for i in range(D - 1):
        d.set_migration_rate(f"d{i}", f"d{i+1}", m)
        d.set_migration_rate(f"d{i+1}", f"d{i}", m)
    for k in range(D - 1, 0, -1):
        t = T0 + (D - 1 - k) * split_step
        d.add_migration_rate_change(time=t, rate=0, source=f"d{k-1}", dest=f"d{k}")
        d.add_migration_rate_change(time=t, rate=0, source=f"d{k}", dest=f"d{k-1}")
        d.add_mass_migration(time=t + 1, source=f"d{k}", dest=f"d{k-1}", proportion=1.0)
    tanc = T0 + (D - 1) * split_step + 500
    d.add_population_split(time=tanc, derived=["d0"], ancestral="ANC")
    d.sort_events()
    train_deme = 0
    npc = 2
    samples = {
        f"d{k}": (N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST if k == train_deme else N_MODEL_TRAIN + N_TEST)
        for k in range(D)
    }
    coord_list, dist_list = [], []
    for k in range(D):
        n = N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST if k == train_deme else N_MODEL_TRAIN + N_TEST
        coord_list += [k] * n
        dist_list += [abs(k - train_deme)] * n
    coord = np.array(coord_list, dtype=float)
    dist = np.array(dist_list, dtype=float)
    return d, samples, coord, dist, train_deme, npc


def dem_grid2d(side=6, N=3000, Nanc=10000, m=2e-4, split_time=15000, migration_scale=1.0):
    m = m * migration_scale     # <1 = more isolation (harder portability), >1 = less
    d = msprime.Demography()
    nm = lambda r, c: f"d_{r}_{c}"
    for r in range(side):
        for c in range(side):
            d.add_population(name=nm(r, c), initial_size=N)
    d.add_population(name="ANC", initial_size=Nanc)
    for r in range(side):
        for c in range(side):
            for dr, dc in [(1, 0), (0, 1)]:
                r2, c2 = r + dr, c + dc
                if r2 < side and c2 < side:
                    d.set_migration_rate(nm(r, c), nm(r2, c2), m)
                    d.set_migration_rate(nm(r2, c2), nm(r, c), m)
    d.add_population_split(
        time=split_time,
        derived=[nm(r, c) for r in range(side) for c in range(side)],
        ancestral="ANC",
    )
    d.sort_events()
    train_deme = 0
    npc = 3
    nd = side * side
    rc = np.array([(r, c) for r in range(side) for c in range(side)])
    dist_per_deme = rc[:, 0] + rc[:, 1]
    samples = {
        nm(rc[i][0], rc[i][1]): (
            N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST if i == train_deme else N_MODEL_TRAIN + N_TEST
        )
        for i in range(nd)
    }
    coord_list, dist_list = [], []
    for i in range(nd):
        n = N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST if i == train_deme else N_MODEL_TRAIN + N_TEST
        coord_list += [i] * n
        dist_list += [int(dist_per_deme[i])] * n
    coord = np.array(coord_list, dtype=float)
    dist = np.array(dist_list, dtype=float)
    return d, samples, coord, dist, train_deme, npc


def dem_admix(migration_scale=1.0):
    """stdpopsim HomSap AmericanAdmixture model: AFR/EUR/ASIA/ADMIX present-day
    populations. GWAS/training population is EUR; all four are sampled. ADMIX is the
    admixed (interpolation) target. migration_scale accepted for interface parity (the
    published model is fixed; ignored)."""
    import stdpopsim
    m = stdpopsim.get_species("HomSap").get_demographic_model("AmericanAdmixture_4B18")
    d = m.model
    train_pop = "EUR"
    npc = 3
    # rough divergence-from-EUR ordering, only used for distance-bin grouping/plots
    distmap = {"EUR": 0.0, "ADMIX": 1.0, "ASIA": 2.0, "AFR": 3.0}
    samples, coord_list, dist_list = {}, [], []
    for idx, p in enumerate(d.populations):          # demography population order = sample order
        if p.name not in distmap:
            continue
        n = N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST if p.name == train_pop else N_MODEL_TRAIN + N_TEST
        samples[p.name] = n
        coord_list += [idx] * n
        dist_list += [distmap[p.name]] * n
    coord = np.array(coord_list, dtype=float)
    dist = np.array(dist_list, dtype=float)
    train_deme = next(i for i, p in enumerate(d.populations) if p.name == train_pop)
    return d, samples, coord, dist, train_deme, npc


DEMS = {"serial1d": dem_serial1d, "grid2d": dem_grid2d, "admix": dem_admix}

DEMOGRAPHY_NOTES = {
    "serial1d": {
        "model": "1-D serial-founder / nearest-neighbor stepping-stone",
        "deme_size": 3000,
        "ancestral_size": 10000,
        "nearest_neighbor_migration": 5e-4,
        "founder_interval_generations": 500,
        "first_founder_time_generations": 250,
    },
    "grid2d": {
        "model": "2-D nearest-neighbor stepping-stone grid",
        "side_length": 6,
        "deme_size": 3000,
        "ancestral_size": 10000,
        "nearest_neighbor_migration": 2e-4,
        "common_ancestor_time_generations": 15000,
    },
    "admix": {
        "model": "stdpopsim HomSap AmericanAdmixture_4B18 (AFR/EUR/ASIA/ADMIX)",
        "training_population": "EUR",
        "admixed_population": "ADMIX",
        "source": "Browning et al. 2018 via stdpopsim 0.3.0",
    },
}


def standardize(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-12)


def fst_pair(dos_a, dos_b):
    p_a = dos_a.mean(0) / 2.0
    p_b = dos_b.mean(0) / 2.0
    n_a, n_b = dos_a.shape[0], dos_b.shape[0]
    num = (p_a - p_b) ** 2 - p_a * (1 - p_a) / (2 * n_a - 1) - p_b * (1 - p_b) / (2 * n_b - 1)
    den = p_a * (1 - p_b) + p_b * (1 - p_a)
    keep = den > 1e-9
    return float(num[keep].sum() / den[keep].sum())


def make_split(coord, train_deme, rng):
    """Canonical disjoint split.

    pgs_train: one large training-deme cohort for GWAS/P+T only.
    model_train: all demes, used to fit risk models/calibration models.
    test: all demes, used only for reported metrics.
    """
    split = np.empty(coord.shape[0], dtype=object)
    split[:] = "unset"
    for dd in np.unique(coord):
        idx = np.where(coord == dd)[0]
        rng.shuffle(idx)
        if dd == train_deme:
            if len(idx) < N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST:
                raise RuntimeError(f"training deme has {len(idx)} rows, expected at least "
                                   f"{N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST}")
            split[idx[:N_PGS_TRAIN]] = "pgs_train"
            split[idx[N_PGS_TRAIN:N_PGS_TRAIN + N_MODEL_TRAIN]] = "model_train"
            split[idx[N_PGS_TRAIN + N_MODEL_TRAIN:N_PGS_TRAIN + N_MODEL_TRAIN + N_TEST]] = "test"
        else:
            if len(idx) < N_MODEL_TRAIN + N_TEST:
                raise RuntimeError(f"non-training deme {dd} has {len(idx)} rows, expected at least "
                                   f"{N_MODEL_TRAIN + N_TEST}")
            split[idx[:N_MODEL_TRAIN]] = "model_train"
            split[idx[N_MODEL_TRAIN:N_MODEL_TRAIN + N_TEST]] = "test"
    if np.any(split == "unset"):
        raise RuntimeError("unset splits remain")
    return split


def log(msg):
    print(msg, flush=True)


def run(cmd, desc):
    log(f"[run] {desc}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{desc} failed rc={r.returncode}\nCMD: {' '.join(cmd)}\n"
                           f"STDERR:\n{r.stderr[-3000:]}\nSTDOUT:\n{r.stdout[-1500:]}")
    return r


def real_pt(geno_prefix, n_ind, coord, train_deme, y_binary, pcs, npc, split,
            workdir, threads, mem_mb):
    """Real GWAS + LD clump + P+T on the pre-built <geno_prefix> PLINK fileset.
    GWAS in pgs_train rows only, on y_binary with PC covariates; LD clump with
    plink1.9; choose best p-threshold on an internal pgs_train validation slice;
    score genome-wide. model_train/test rows are untouched by P+T fitting.
    Returns PGS_raw (n_ind,) and meta dict."""
    os.makedirs(workdir, exist_ok=True)
    prefix = geno_prefix
    common = ["--threads", str(threads)]
    if mem_mb:
        common += ["--memory", str(mem_mb)]

    # The real PGS uses pgs_train ONLY. P+T needs a held-out slice for p-threshold
    # selection (in-sample selection overfits to the most-SNPs threshold), so we carve an
    # internal 80/20 partition PRIVATELY from pgs_train: GWAS on the inner 80%,
    # threshold selection on the inner 20%. model_train/test are fully untouched here.
    tf_idx = np.where(split == "pgs_train")[0]
    if tf_idx.size < 100:
        raise RuntimeError(f"Too few pgs_train rows ({tf_idx.size})")
    rng_int = np.random.default_rng(90210 + int(tf_idx.size))
    perm = rng_int.permutation(tf_idx)
    n_inner_fit = int(round(0.80 * perm.size))
    fit_idx = np.sort(perm[:n_inner_fit])        # inner GWAS rows
    sel_idx = np.sort(perm[n_inner_fit:])        # inner threshold-selection rows
    if sel_idx.size < 20:
        raise RuntimeError(f"Too few inner threshold-selection rows ({sel_idx.size})")
    log(f"[P+T] pgs_train={tf_idx.size}: inner GWAS={fit_idx.size}, "
        f"inner threshold-selection={sel_idx.size}")
    keep_path = os.path.join(workdir, "fit.keep")
    with open(keep_path, "w") as f:
        f.writelines(f"i{i}\ti{i}\n" for i in fit_idx)

    phe_path = os.path.join(workdir, "fit.pheno")
    cov_path = os.path.join(workdir, "fit.covar")
    with open(phe_path, "w") as f:
        f.write("FID\tIID\ty\n")
        f.writelines(f"i{i}\ti{i}\t{int(y_binary[i])}\n" for i in fit_idx)
    with open(cov_path, "w") as f:
        f.write("FID\tIID\t" + "\t".join(f"PC{k+1}" for k in range(npc)) + "\n")
        f.writelines("i{0}\ti{0}\t".format(i) + "\t".join(f"{pcs[i,k]:.6f}" for k in range(npc)) + "\n"
                     for i in fit_idx)

    # GWAS: plink2 logistic, pgs_train inner rows, PC covariates
    gwas_prefix = os.path.join(workdir, "gwas")
    run([PLINK2, "--bfile", prefix, "--keep", keep_path, "--pheno", phe_path,
         "--pheno-name", "y", "--1", "--covar", cov_path, "--covar-variance-standardize",
         "--glm", "hide-covar", "--allow-no-sex", "--allow-extra-chr",
         *common, "--out", gwas_prefix], "GWAS (pgs_train inner)")
    glm = [p for p in os.listdir(workdir) if p.startswith("gwas.") and ".glm.logistic" in p]
    if not glm:
        raise RuntimeError(f"No GWAS logistic output; dir={os.listdir(workdir)}")
    gwas = pd.read_csv(os.path.join(workdir, glm[0]), sep=r"\s+")
    if "TEST" in gwas.columns:
        gwas = gwas[gwas["TEST"].astype(str) == "ADD"].copy()
    if "OR" in gwas.columns:
        eff = np.log(pd.to_numeric(gwas["OR"], errors="coerce").where(lambda v: v > 0))
    elif "BETA" in gwas.columns:
        eff = pd.to_numeric(gwas["BETA"], errors="coerce")
    else:
        raise RuntimeError(f"GWAS missing OR/BETA; cols={list(gwas.columns)}")
    gdf = pd.DataFrame({
        "SNP": gwas["ID"].astype(str), "A1": gwas["A1"].astype(str),
        "P": pd.to_numeric(gwas["P"], errors="coerce"), "EFF": eff,
    }).dropna(subset=["P", "EFF"])
    log(f"[P+T] GWAS usable rows = {len(gdf)}")

    # LD clump with plink1.9 on the pgs_train inner-GWAS genotypes
    fit_bfile = os.path.join(workdir, "fit_geno")
    run([PLINK1, "--bfile", prefix, "--keep", keep_path, "--make-bed",
         "--allow-extra-chr", *common, "--out", fit_bfile], "subset fit bfile")
    assoc = os.path.join(workdir, "assoc.txt")
    gdf[["SNP", "P"]].to_csv(assoc, sep="\t", index=False)
    clump_prefix = os.path.join(workdir, "clump")
    run([PLINK1, "--bfile", fit_bfile, "--clump", assoc,
         "--clump-p1", "1.0", "--clump-p2", "1.0",
         "--clump-r2", str(CLUMP_R2), "--clump-kb", str(CLUMP_KB),
         "--clump-snp-field", "SNP", "--clump-field", "P",
         "--allow-extra-chr", *common, "--out", clump_prefix], "LD clump (plink1.9)")
    clumped = set()
    cf = f"{clump_prefix}.clumped"
    if os.path.exists(cf):
        clumped = set(pd.read_csv(cf, sep=r"\s+")["SNP"].astype(str))
    log(f"[P+T] clumped index SNPs = {len(clumped)}")
    if len(clumped) < MIN_CLUMP_SNPS:
        raise RuntimeError(f"Too few clumped SNPs ({len(clumped)}); need >= {MIN_CLUMP_SNPS}")
    gdf = gdf[gdf["SNP"].isin(clumped)].copy().sort_values("P")

    # Score genome-wide at all thresholds in one pass (q-score-range)
    score_all = os.path.join(workdir, "all.score")
    gdf[["SNP", "A1", "EFF"]].to_csv(score_all, sep="\t", header=False, index=False)
    qfile = os.path.join(workdir, "qvals.txt")
    gdf[["SNP", "P"]].to_csv(qfile, sep="\t", header=False, index=False)
    rfile = os.path.join(workdir, "ranges.txt")
    labels = {}
    with open(rfile, "w") as f:
        for thr in P_THRESHOLDS:
            lab = f"T{str(thr).replace('.', 'p').replace('-', 'm')}"
            labels[thr] = lab
            f.write(f"{lab} 0 {thr}\n")
    score_prefix = os.path.join(workdir, "scoreall")
    run([PLINK2, "--bfile", prefix, "--score", score_all, "1", "2", "3",
         "cols=+scoresums", "--q-score-range", rfile, qfile, "1", "2",
         "--allow-no-sex", "--allow-extra-chr", *common, "--out", score_prefix], "score all thresholds")

    yb_sel = y_binary[sel_idx].astype(int)
    yb_fit = y_binary[fit_idx].astype(int)
    order = {f"i{i}": i for i in range(n_ind)}
    best = None
    pgs_by_thr = {}
    sel_aucs = {}
    for thr in P_THRESHOLDS:
        sp = f"{score_prefix}.{labels[thr]}.sscore"
        if not os.path.exists(sp):
            continue
        sdf = pd.read_csv(sp, sep=r"\s+")
        iidc = "IID" if "IID" in sdf.columns else "#IID"
        sc = "SCORE1_AVG" if "SCORE1_AVG" in sdf.columns else "SCORE1_SUM"
        sdf[iidc] = sdf[iidc].astype(str)
        sdf["__o"] = sdf[iidc].map(order)
        sdf = sdf.dropna(subset=["__o"])
        full = np.full(n_ind, np.nan)
        full[sdf["__o"].astype(int).to_numpy()] = sdf[sc].to_numpy(dtype=float)
        pgs_by_thr[thr] = full
        psel = full[sel_idx]
        n_snps = int((gdf["P"] <= thr).sum())
        if np.std(psel) < 1e-9 or len(np.unique(yb_sel)) < 2 or n_snps < 1:
            continue
        auc = roc_auc_score(yb_sel, psel)
        # orient per-threshold so AUC>=0.5 (sign learned on fit rows below)
        auc = max(auc, 1 - auc)
        sel_aucs[thr] = (round(auc, 4), n_snps)
        if best is None or auc > best[1]:
            best = (thr, auc, n_snps)
    if best is None:
        raise RuntimeError("No valid P+T threshold produced a non-degenerate cal PGS")
    best_thr, best_auc_sel, best_nsnps = best
    log(f"[P+T] selected p<={best_thr:g} (internal pgs_train validation AUC={best_auc_sel:.4f}) n_snps={best_nsnps}")
    log(f"[P+T] per-threshold internal validation AUC / n_snps: "
        + ", ".join(f"{t:g}:{a}/{n}" for t, (a, n) in sorted(sel_aucs.items())))
    pgs_raw = pgs_by_thr[best_thr].copy()
    # orient so higher PGS = higher risk, using the GWAS (fit) rows
    if np.corrcoef(pgs_raw[fit_idx], yb_fit)[0, 1] < 0:
        pgs_raw = -pgs_raw
    if not np.isfinite(pgs_raw).all():
        raise ValueError(f"non-finite PGS for {int((~np.isfinite(pgs_raw)).sum())} individuals")
    meta = dict(best_p_threshold=float(best_thr), best_cal_auc=round(float(best_auc_sel), 4),
                n_snps=int(best_nsnps), n_clumped=int(len(clumped)),
                n_gwas_usable=int(len(gdf)), clump_r2=CLUMP_R2, clump_kb=CLUMP_KB,
                n_fit=int(fit_idx.size), n_sel=int(sel_idx.size),
                per_threshold_cal_auc={f"{t:g}": a for t, (a, n) in sorted(sel_aucs.items())})
    return pgs_raw, meta


def main():
    global SIGMA_E, N_CAUSAL, DATA_DIR, N_MODEL_TRAIN     # reassigned below from CLI overrides
    ap = argparse.ArgumentParser()
    ap.add_argument("dem", choices=["serial1d", "grid2d", "admix"])
    ap.add_argument("seed", nargs="?", type=int, default=7)
    # generation/plink knobs (CLI, not env vars) so chunk size / parallelism can be
    # swept without editing constants; all default to the module DEFAULT_* values.
    ap.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    ap.add_argument("--chunk-bp", type=int, default=DEFAULT_CHUNK_BP)
    ap.add_argument("--workers", type=int, default=0, help="0 = auto (allocated cores)")
    ap.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    ap.add_argument("--mem-mb", type=int, default=DEFAULT_MEM_MB)
    # scientific sweep knobs (CLI, not env vars); default to the module constants.
    ap.add_argument("--sigma-e", type=float, default=SIGMA_E, help="env SD; h2=1/(1+sigma_e^2)")
    ap.add_argument("--n-causal", type=int, default=N_CAUSAL)
    ap.add_argument("--baseline-scale", type=float, default=1.0,
                    help="multiplies the deme-varying environmental baseline (confounding strength)")
    ap.add_argument("--migration-scale", type=float, default=1.0,
                    help="multiplies nearest-neighbour migration (<1 = more isolation)")
    ap.add_argument("--npc", type=int, default=0,
                    help="override # PCs used by PGS+PCs and the gamfit surface (0 = demography default)")
    ap.add_argument("--run-tag", default="", help="output subdir for isolating sweep experiments")
    ap.add_argument("--n-model-train", type=int, default=N_MODEL_TRAIN,
                    help="calibration-model training rows per deme")
    args = ap.parse_args()

    # apply scientific overrides (module globals so all downstream references pick them up)
    SIGMA_E = args.sigma_e
    N_MODEL_TRAIN = args.n_model_train
    N_CAUSAL = args.n_causal
    if args.run_tag:
        DATA_DIR = REPO_ROOT / "sims" / "results_hpc" / "ancestry_calibration" / args.run_tag / "data"

    dem_name = args.dem
    outdir = DATA_DIR
    seed = args.seed
    TAG = f"_s{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    work_root = (outdir.parent / "_work").resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    tag_clean = TAG.replace("/", "_") if TAG else "base"
    rng = np.random.default_rng(1000 + seed * 17)

    # STREAM the genome to <geno_prefix>.bed/.bim/.fam without ever holding the full
    # dense dosage matrix in RAM. Only PCA- and causal-candidate columns are retained
    # dense (a few thousand cols), so peak RAM is bounded regardless of total Mb.
    from stream_geno import simulate_stream
    dG, samples, coord, dist, train_deme, npc = DEMS[dem_name](migration_scale=args.migration_scale)
    if args.npc:
        npc = min(args.npc, 6)              # PCA computes up to 6 components
    n_demes = len(np.unique(coord))
    geno_dir = work_root / f"{dem_name}_seed{seed}_{tag_clean}_geno"
    shutil.rmtree(geno_dir, ignore_errors=True)
    geno_dir.mkdir(parents=True, exist_ok=True)
    geno_prefix = os.path.join(geno_dir, "geno")
    st = simulate_stream(dG, samples, args.chunks, args.chunk_bp, RECOMB, MU, seed,
                         geno_prefix, n_pca_keep=5000, n_causal_keep=max(N_CAUSAL * 8, 2000),
                         log=lambda m: log(f"[{dem_name}] {m}"),
                         workers=(args.workers or None))
    nI = st["n_ind"]
    if nI != coord.shape[0]:
        raise RuntimeError(f"individual count mismatch: bed {nI} vs coord {coord.shape[0]}")
    if st["Xpca"].shape[1] < 500:
        raise RuntimeError(f"{dem_name}: too few common SNPs retained ({st['Xpca'].shape[1]})")
    log(f"[{dem_name}] nInd={nI} nLoci={st['n_loci']} pca_kept={st['Xpca'].shape[1]} causal_pool={st['Xcausal'].shape[1]}")

    # PCs from genotypes (the reservoir-sampled common loci)
    _t = time.time()
    pcs = PCA(min(6, st["Xpca"].shape[1]), svd_solver="randomized", random_state=0).fit_transform(
        StandardScaler().fit_transform(st["Xpca"]))
    pcs = StandardScaler().fit_transform(pcs)[:, :npc]
    log(f"[{dem_name}] PCA done in {time.time()-_t:.0f}s")

    # True genetic liability. Synth uses a highly polygenic oracle (2000 candidates) and
    # fixes home accuracy by fiat (rho0=0.8). A REAL P+T PGS at finite GWAS n only carries
    # comparable home-deme signal if the architecture is GWAS-recoverable, so we use a
    # moderately concentrated architecture (n_causal causal SNPs). Liability is still
    # standardized to N(0,1), so heritability semantics are stable; only the detectability
    # differs -- which is exactly what makes real-P+T the rigorous track. Causal SNPs are
    # drawn from the retained causal pool (a representative genome-wide common-variant set).
    _t = time.time()
    Xc = st["Xcausal"]
    nca = min(N_CAUSAL, Xc.shape[1])
    caus_local = rng.choice(Xc.shape[1], nca, replace=False)
    Gmat = StandardScaler(with_std=False).fit_transform(Xc[:, caus_local])
    beta = rng.normal(0, 1, nca)
    true_liab = standardize(Gmat @ beta)
    log(f"[{dem_name}] liability done in {time.time()-_t:.0f}s")

    _t = time.time()
    split = make_split(coord, train_deme, rng)
    # per-deme baselines: phenoA varies by deme, phenoB is constant
    uq = np.unique(coord)
    udeme = (uq - uq.min()) / (uq.max() - uq.min() + 1e-12)
    base_by_deme = args.baseline_scale * (0.45 * np.sin(2 * np.pi * udeme) + 0.25 * (udeme - 0.5))
    base_by_deme -= base_by_deme.mean()
    deme_to_baseA = {dd: b for dd, b in zip(uq, base_by_deme)}
    baseA = np.array([deme_to_baseA[c] for c in coord])
    baseB = np.zeros(nI)

    # ---- REAL P+T: build PGS once (on phenoA's y_binary, the realistic phenotype) ----
    # PGS is a property of genotype + the GWAS; reused for phenoB (same as deploying a
    # trained PGS regardless of which baseline-shift defines the outcome.
    linA = baseA + true_liab
    c0A = brentq(lambda c: norm.cdf((c + linA) / SIGMA_E).mean() - PREV, -20, 20)
    yA = ((c0A + linA + rng.normal(0, SIGMA_E, nI)) > 0).astype(int)
    log(f"[{dem_name}] split+pheno done in {time.time()-_t:.0f}s; entering P+T")
    work = work_root / f"{dem_name}_seed{seed}_{tag_clean}_pt"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    try:
        PGS_raw, pt_meta = real_pt(geno_prefix, nI, coord, train_deme, yA, pcs, npc,
                                   split, str(work), args.threads, args.mem_mb)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(geno_dir, ignore_errors=True)  # drop the big .bed once scored
    # standardize PGS_z on PGS-training rows ONLY; no model_train/test leakage.
    # PGS_raw is left as the raw plink score (no pooled standardization).
    fitm = (split == "pgs_train")
    tr_mean = float(PGS_raw[fitm].mean())
    tr_sd = float(PGS_raw[fitm].std())
    PGS_z = (PGS_raw - tr_mean) / (tr_sd + 1e-12)

    # GROUND TRUTH: generative per-deme PGS_z->liability slope. Synth uses the analytic
    # oracle value rho(d)*tr_sd; for a real PGS there is no rho, so the honest generative
    # slope is the population regression coefficient Cov(true_liab, PGS_z)/Var(PGS_z)
    # within each deme (all individuals, not just test). Same column + meaning.
    true_slope_by_deme = {}
    true_slope_deme = np.zeros(nI, dtype=float)
    for dd in uq:
        mk = coord == dd
        z = PGS_z[mk]
        vz = float(np.var(z))
        b = float(np.cov(z, true_liab[mk])[0, 1] / vz) if vz > 1e-12 else 0.0
        true_slope_by_deme[int(dd)] = b
        true_slope_deme[mk] = b

    far_deme = uq[int(np.argmax([dist[coord == c][0] for c in uq]))]
    near_deme = uq[1] if len(uq) > 1 else uq[0]
    sidecar = {
        "dem": dem_name, "seed": seed, "train_deme": int(train_deme),
        "n_total": int(nI),
        "n_pgs_train": int(N_PGS_TRAIN),
        "n_model_train_per_deme": int(N_MODEL_TRAIN),
        "n_test_per_deme": int(N_TEST),
        "split_counts": {str(k): int(v) for k, v in pd.Series(split).value_counts().sort_index().items()},
        "npc": int(npc), "sigma_e": SIGMA_E, "prev_target": PREV,
        "demography": DEMOGRAPHY_NOTES[dem_name],
        "pgs_train_mean": tr_mean, "pgs_train_sd": tr_sd, "pgs_mode": "realpt",
        "pgs_method": "real GWAS (plink2 logistic, PC covars) + plink1.9 LD clump "
                      f"(r2<{CLUMP_R2}, {CLUMP_KB}kb) + p-value thresholding; "
                      "GWAS+weights on pgs_train only (inner 80/20 GWAS/threshold-select); "
                      "model_train/test held out from P+T",
        "params": {"mu": MU, "recomb": RECOMB, "n_chunks": int(args.chunks),
                   "chunk_bp": int(args.chunk_bp), "seqlen_bp": int(args.chunks * args.chunk_bp),
                   "n_loci_total": int(st["n_loci"]), "n_loci_common": int(st["Xpca"].shape[1]),
                   "n_causal": int(N_CAUSAL)},
        "pt": pt_meta,
        "true_slope_by_deme": {int(d): round(v, 4) for d, v in true_slope_by_deme.items()},
        "fst_train_vs_near": round(fst_pair(st["Xpca"][coord == train_deme], st["Xpca"][coord == near_deme]), 5),
        "fst_train_vs_far": round(fst_pair(st["Xpca"][coord == train_deme], st["Xpca"][coord == far_deme]), 5),
        "far_deme": int(far_deme), "near_deme": int(near_deme),
        "phenos": {},
        "truth_columns": ["true_liab", "true_slope_deme", "intercept_deme", "p_true"],
        "metric_rule": "Use split == test rows only for reported metrics. Never compute within-bin AUC; "
                       "stratify calibration/group metrics only.",
    }

    for pheno, base in [("phenoA", baseA), ("phenoB", baseB)]:
        # Binary outcome via probit liability with environmental noise.
        lin = base + true_liab
        c0 = brentq(lambda c: norm.cdf((c + lin) / SIGMA_E).mean() - PREV, -20, 20)
        intercept_deme = c0 + base                       # generative per-deme intercept
        p_true = norm.cdf((c0 + lin) / SIGMA_E)          # EXACT generative P(y=1)
        e = rng.normal(0, SIGMA_E, nI)
        L = c0 + lin + e
        y_binary = (L > 0).astype(int)

        df = pd.DataFrame({
            "iid": np.arange(nI),
            "deme": coord.astype(int),
            "coord": coord,
            "dist_from_train": dist,
            "PGS_raw": PGS_raw,
            "PGS_z": PGS_z,
            "y_binary": y_binary,
            "true_liab": true_liab,
            "true_slope_deme": true_slope_deme,
            "intercept_deme": intercept_deme,
            "p_true": p_true,
            "split": split,
            "pgs_mode": "realpt",
        })
        for i in range(npc):
            df.insert(5 + i, f"PC{i+1}", pcs[:, i])

        path = outdir / f"{dem_name}_{pheno}_realpt{TAG}.parquet"
        df.to_parquet(path, index=False)
        df.to_csv(outdir / f"{dem_name}_{pheno}_realpt{TAG}.csv", index=False)

        test = df[df["split"].astype(str) == "test"].copy()
        pooled_auc = (roc_auc_score(test["y_binary"], test["PGS_raw"])
                      if test["y_binary"].nunique() > 1 else float("nan"))
        per_deme = []
        for dd in uq:
            sub = test[test["deme"] == int(dd)]
            if sub.empty:
                continue
            corr = (float(np.corrcoef(sub["PGS_raw"], sub["true_liab"])[0, 1])
                    if len(sub) > 3 else float("nan"))
            per_deme.append({
                "deme": int(dd), "dist": float(dist[coord == dd][0]),
                "n_test": int(len(sub)), "is_training_ancestry": bool(dd == train_deme),
                "true_slope_deme": float(true_slope_by_deme[int(dd)]),
                "corr_pgs_true_liab_test": corr,
                "mean_pgs_raw_test": float(sub["PGS_raw"].mean()),
                "sd_pgs_raw_test": float(sub["PGS_raw"].std()),
                "mean_p_true": float(sub["p_true"].mean()),
                "binary_prevalence_test": float(sub["y_binary"].mean()),
            })
        sanity = pd.DataFrame(per_deme).sort_values("dist")
        sanity_path = outdir / f"{dem_name}_{pheno}_realpt{TAG}.sanity.tsv"
        sanity.to_csv(sanity_path, sep="\t", index=False)
        sidecar["phenos"][pheno] = {
            "path": str(path), "sanity_path": str(sanity_path), "n_test": int(len(test)),
            "p_true_range_test": [float(test["p_true"].min()), float(test["p_true"].max())],
            "binary_prevalence_test": float(test["y_binary"].mean()),
            "global_test_raw_pgs_auc_binary": float(pooled_auc),
        }
        log(f"[{dem_name}/{pheno}] wrote {path}")
        log(f"  test n={len(test)} prev={test['y_binary'].mean():.3f} "
            f"raw_auc={pooled_auc:.3f}")
        log(sanity[["deme", "n_test", "is_training_ancestry", "dist", "true_slope_deme",
                    "corr_pgs_true_liab_test", "mean_pgs_raw_test", "binary_prevalence_test"]].to_string(index=False))

    with open(outdir / f"{dem_name}_realpt{TAG}.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    log(f"[{dem_name}] sidecar -> {outdir / f'{dem_name}_realpt{TAG}.json'}  "
        f"P+T p<={pt_meta['best_p_threshold']:g} n_snps={pt_meta['n_snps']} "
        f"cal_auc={pt_meta['best_cal_auc']} fst_far={sidecar['fst_train_vs_far']}")


if __name__ == "__main__":
    main()
