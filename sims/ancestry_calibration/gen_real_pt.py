"""Generate the real-P+T ancestry-portability datasets for the study.

Two custom non-affine msprime demographies (serial1d serial-founder chain, grid2d
2-D stepping-stone), a shared-effect genetic liability, and a real PGS built by an
actual GWAS + LD clumping (plink1.9 --clump r2<0.1, 250kb) + p-value thresholding
(P+T) trained in ONE randomly chosen training deme, then scored genome-wide.

Hard rules:
- only serial1d and grid2d custom msprime demographies
- the training deme is chosen at RANDOM per seed (a real biobank's majority ancestry
  is an arbitrary derived population, not the high-diversity founder)
- the training ancestry has held-out test rows; downstream metrics use the *-deme-test
  rows only (incl. the held-out training-ancestry test)
- GWAS + P+T weights AND PGS standardization use split == "train-deme-fit" ONLY, with
  the GWAS-vs-threshold-selection partition carved privately inside real_pt() so the
  train-deme-test rows never leak
- binary ground truth is p_true; survival ground truth is the admin-horizon risk
- causal effects are SHARED across demes, so any portability decay is EMERGENT from
  LD/allele-frequency drift -- never manufactured -- and is reported honestly

Three phenotypes per dataset exercise distinct confounds:
- phenoA: deme-varying baseline risk (1-D linear gradient / 2-D per-deme random offset)
- phenoB: constant baseline (drift in the genetic liability still shifts per-deme prevalence)
- phenoC: drift-proof -- the liability is centered WITHIN each deme, so every deme has an
  identical mean liability and prevalence; only the calibration slope can differ.

Usage: gen_real_pt.py <dem: serial1d|grid2d> <outdir> [seed]
       [--chunks N] [--chunk-bp BP] [--threads T] [--mem-mb MB]
Writes <outdir>/<dem>_{phenoA,phenoB,phenoC}_realpt.parquet (+ .csv, .sanity.tsv)
  phenoA = deme-varying baseline (1-D linear / 2-D random); phenoB = constant baseline;
  phenoC = drift-proof (per-deme-centered liability => identical prevalence across demes)
and a sidecar <outdir>/<dem>_realpt.json (GLOBAL/pooled diagnostics only).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import msprime
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PLINK1 = "/common/software/install/migrated/plink/1.90b6.10/bin/plink"
PLINK2 = "/users/0/sauer354/bin/plink2"

MU = 1.25e-8
RECOMB = 1.0e-8
NPER = 250                 # non-training demes (match synth)
NPER_TRAIN = 5000          # large training deme for GWAS power (team-lead spec)
PREV = 0.15
SIGMA_E = 1.0              # environmental SD on the liability scale (matches synth "normal")

DEFAULT_CHUNKS = 20
DEFAULT_CHUNK_BP = 5_000_000

P_THRESHOLDS = [5e-8, 1e-6, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
CLUMP_R2 = 0.1
CLUMP_KB = 250
MIN_CLUMP_SNPS = 20


# ---------------- demographies (serial-founder chain / stepping-stone grid) ----------------
def dem_serial1d(train_deme=0, D=10, N=3000, Nanc=10000, m=1e-3, split_step=400, T0=200):
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
    npc = 2
    samples = {f"d{k}": (NPER_TRAIN if k == train_deme else NPER) for k in range(D)}
    coord_list, dist_list = [], []
    for k in range(D):
        n = NPER_TRAIN if k == train_deme else NPER
        coord_list += [k] * n
        dist_list += [abs(k - train_deme)] * n
    coord = np.array(coord_list, dtype=float)
    dist = np.array(dist_list, dtype=float)
    return d, samples, coord, dist, train_deme, npc


def dem_grid2d(train_deme=0, side=6, N=3000, Nanc=10000, m=3e-4):
    # m lowered 1e-3 -> 3e-4 so far-corner Fst rises from ~0.09 toward the
    # target far-Fst 0.2-0.25 (human "sorta distant" range). serial1d m=1e-3
    # already gives far-Fst ~0.21. Achieved Fst is reported in the sidecar.
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
        time=4 * N,
        derived=[nm(r, c) for r in range(side) for c in range(side)],
        ancestral="ANC",
    )
    d.sort_events()
    npc = 3
    nd = side * side
    rc = np.array([(r, c) for r in range(side) for c in range(side)])
    tr, tc = rc[train_deme]
    dist_per_deme = np.abs(rc[:, 0] - tr) + np.abs(rc[:, 1] - tc)   # Manhattan from train deme
    samples = {nm(rc[i][0], rc[i][1]): (NPER_TRAIN if i == train_deme else NPER) for i in range(nd)}
    coord_list, dist_list = [], []
    for i in range(nd):
        n = NPER_TRAIN if i == train_deme else NPER
        coord_list += [i] * n
        dist_list += [int(dist_per_deme[i])] * n
    coord = np.array(coord_list, dtype=float)
    dist = np.array(dist_list, dtype=float)
    return d, samples, coord, dist, train_deme, npc


DEMS = {"serial1d": dem_serial1d, "grid2d": dem_grid2d}


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


def harrell_c_index(time, event, score):
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=bool)
    s = np.asarray(score, dtype=float)
    conc = 0.0
    comp = 0.0
    for i in range(len(t)):
        if not e[i]:
            continue
        m = t[i] < t
        if not np.any(m):
            continue
        diff = s[i] - s[m]
        conc += float(np.sum(diff > 0)) + 0.5 * float(np.sum(diff == 0))
        comp += float(np.sum(m))
    return float(conc / comp) if comp else float("nan")


def make_split(coord, train_deme, rng):
    """CANONICAL 4-level split (matches synth): every deme is split 50/50 into
    fit/test -> {train-deme-fit, train-deme-test, other-deme-fit, other-deme-test}.
    The real PGS (GWAS + P+T weights + PGS_z standardization) uses train-deme-fit ONLY.
    All downstream metrics use the *-deme-test rows (incl. a held-out training-ancestry
    test = train-deme-test). The internal GWAS-vs-threshold-selection partition is carved
    privately from train-deme-fit inside real_pt(), so train-deme-test never leaks."""
    split = np.empty(coord.shape[0], dtype=object)
    split[:] = "unset"
    for dd in np.unique(coord):
        idx = np.where(coord == dd)[0]
        rng.shuffle(idx)
        nfit = int(round(0.5 * len(idx)))
        fit_idx, test_idx = idx[:nfit], idx[nfit:]
        tag = "train" if dd == train_deme else "other"
        split[fit_idx] = f"{tag}-deme-fit"
        split[test_idx] = f"{tag}-deme-test"
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
    GWAS in train-deme-fit (inner) rows only, on y_binary with PC covariates; LD clump
    with plink1.9; choose best p-threshold on the held-out TRAIN-DEME cal rows (NOT the
    GWAS rows -- in-sample selection overfits to the most-SNPs threshold); score
    genome-wide. The 'test' rows of the training deme stay fully held out.
    Returns PGS_raw (n_ind,) and meta dict."""
    os.makedirs(workdir, exist_ok=True)
    prefix = geno_prefix
    common = ["--threads", str(threads)]
    if mem_mb:
        common += ["--memory", str(mem_mb)]

    # The real PGS uses train-deme-fit ONLY. P+T needs a held-out slice for p-threshold
    # selection (in-sample selection overfits to the most-SNPs threshold), so we carve an
    # internal 80/20 partition PRIVATELY from train-deme-fit: GWAS on the inner 80%,
    # threshold selection on the inner 20%. train-deme-test (and all *-deme-test) stay
    # fully held out for downstream modeling -- nothing here ever touches them.
    tf_idx = np.where(split == "train-deme-fit")[0]
    if tf_idx.size < 100:
        raise RuntimeError(f"Too few train-deme-fit rows ({tf_idx.size})")
    rng_int = np.random.default_rng(90210 + int(tf_idx.size))
    perm = rng_int.permutation(tf_idx)
    n_inner_fit = int(round(0.80 * perm.size))
    fit_idx = np.sort(perm[:n_inner_fit])        # inner GWAS rows
    sel_idx = np.sort(perm[n_inner_fit:])        # inner threshold-selection rows
    if sel_idx.size < 20:
        raise RuntimeError(f"Too few inner threshold-selection rows ({sel_idx.size})")
    log(f"[P+T] train-deme-fit={tf_idx.size}: inner GWAS={fit_idx.size}, "
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

    # GWAS: plink2 logistic, train-deme-fit inner rows, PC covariates, firth fallback
    gwas_prefix = os.path.join(workdir, "gwas")
    run([PLINK2, "--bfile", prefix, "--keep", keep_path, "--pheno", phe_path,
         "--pheno-name", "y", "--1", "--covar", cov_path, "--covar-variance-standardize",
         "--glm", "hide-covar", "firth-fallback", "--allow-no-sex",
         *common, "--out", gwas_prefix], "GWAS (train-deme-fit inner)")
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

    # LD clump with plink1.9 on the train-deme-fit (inner GWAS) genotypes
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
         "--allow-no-sex", *common, "--out", score_prefix], "score all thresholds")

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
        try:
            auc = roc_auc_score(yb_sel, psel)
        except Exception:
            continue
        # orient per-threshold so AUC>=0.5 (sign learned on fit rows below)
        auc = max(auc, 1 - auc)
        sel_aucs[thr] = (round(auc, 4), n_snps)
        if best is None or auc > best[1]:
            best = (thr, auc, n_snps)
    if best is None:
        raise RuntimeError("No valid P+T threshold produced a non-degenerate cal PGS")
    best_thr, best_auc_sel, best_nsnps = best
    log(f"[P+T] selected p<={best_thr:g} (train-deme cal AUC={best_auc_sel:.4f}) n_snps={best_nsnps}")
    log(f"[P+T] per-threshold train-deme-cal AUC / n_snps: "
        + ", ".join(f"{t:g}:{a}/{n}" for t, (a, n) in sorted(sel_aucs.items())))
    pgs_raw = pgs_by_thr[best_thr].copy()
    # orient so higher PGS = higher risk, using the GWAS (fit) rows
    if np.corrcoef(pgs_raw[fit_idx], yb_fit)[0, 1] < 0:
        pgs_raw = -pgs_raw
    pgs_raw = np.where(np.isfinite(pgs_raw), pgs_raw, 0.0)
    meta = dict(best_p_threshold=float(best_thr), best_cal_auc=round(float(best_auc_sel), 4),
                n_snps=int(best_nsnps), n_clumped=int(len(clumped)),
                n_gwas_usable=int(len(gdf)), clump_r2=CLUMP_R2, clump_kb=CLUMP_KB,
                n_fit=int(fit_idx.size), n_sel=int(sel_idx.size),
                per_threshold_cal_auc={f"{t:g}": a for t, (a, n) in sorted(sel_aucs.items())})
    return pgs_raw, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dem", choices=["serial1d", "grid2d"])
    ap.add_argument("outdir")
    ap.add_argument("seed", nargs="?", type=int, default=7)
    ap.add_argument("--chunks", type=int, default=DEFAULT_CHUNKS)
    ap.add_argument("--chunk-bp", type=int, default=DEFAULT_CHUNK_BP)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mem-mb", type=int, default=8000)
    ap.add_argument("--n-causal", type=int, default=150,
                    help="number of causal SNPs (concentrated => GWAS-recoverable PGS)")
    ap.add_argument("--tag", default="",
                    help="filename suffix, e.g. '_s1' -> <dem>_<pheno>_realpt_s1.parquet "
                         "(for multi-seed; empty = canonical unsuffixed files)")
    args = ap.parse_args()
    TAG = str(args.tag)

    dem_name = args.dem
    outdir = Path(args.outdir)
    seed = args.seed
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1000 + seed * 17)

    # STREAM the genome to <geno_prefix>.bed/.bim/.fam without ever holding the full
    # dense dosage matrix in RAM. Only PCA- and causal-candidate columns are retained
    # dense (a few thousand cols), so peak RAM is bounded regardless of total Mb.
    from stream_geno import simulate_stream
    # Training deme is RANDOM per seed (not the founder/source deme): a real biobank's
    # majority ancestry is an arbitrary derived population, not the high-diversity founder.
    n_demes_total = {"serial1d": 10, "grid2d": 36}[dem_name]
    train_deme_pick = int(rng.integers(0, n_demes_total))
    dG, samples, coord, dist, train_deme, npc = DEMS[dem_name](train_deme=train_deme_pick)
    log(f"[{dem_name}] random training deme = {train_deme} (of {n_demes_total})")
    n_demes = len(np.unique(coord))
    geno_dir = tempfile.mkdtemp(prefix=f"realpt_{dem_name}_geno_", dir=str(outdir))
    geno_prefix = os.path.join(geno_dir, "geno")
    st = simulate_stream(dG, samples, args.chunks, args.chunk_bp, RECOMB, MU, seed,
                         geno_prefix, n_pca_keep=5000, n_causal_keep=max(args.n_causal * 8, 2000),
                         log=lambda m: log(f"[{dem_name}] {m}"))
    nI = st["n_ind"]
    if nI != coord.shape[0]:
        raise RuntimeError(f"individual count mismatch: bed {nI} vs coord {coord.shape[0]}")
    if st["Xpca"].shape[1] < 500:
        raise RuntimeError(f"{dem_name}: too few common SNPs retained ({st['Xpca'].shape[1]})")
    log(f"[{dem_name}] nInd={nI} nLoci={st['n_loci']} pca_kept={st['Xpca'].shape[1]} causal_pool={st['Xcausal'].shape[1]}")

    # PCs from genotypes (the reservoir-sampled common loci)
    pcs = PCA(min(6, st["Xpca"].shape[1]), svd_solver="randomized", random_state=0).fit_transform(
        StandardScaler().fit_transform(st["Xpca"]))
    pcs = StandardScaler().fit_transform(pcs)[:, :npc]

    # True genetic liability. Synth uses a highly polygenic oracle (2000 candidates) and
    # fixes home accuracy by fiat (rho0=0.8). A REAL P+T PGS at finite GWAS n only carries
    # comparable home-deme signal if the architecture is GWAS-recoverable, so we use a
    # moderately concentrated architecture (n_causal causal SNPs). Liability is still
    # standardized to N(0,1), so heritability semantics match synth; only the detectability
    # differs -- which is exactly what makes real-P+T the rigorous track. Causal SNPs are
    # drawn from the retained causal pool (a representative genome-wide common-variant set).
    Xc = st["Xcausal"]
    nca = min(args.n_causal, Xc.shape[1])
    caus_local = rng.choice(Xc.shape[1], nca, replace=False)
    Gmat = StandardScaler(with_std=False).fit_transform(Xc[:, caus_local])
    beta = rng.normal(0, 1, nca)
    true_liab = standardize(Gmat @ beta)

    split = make_split(coord, train_deme, rng)
    is_train = coord == train_deme

    # Per-deme baselines. phenoA = deme-varying baseline risk: for the 1-D chain a smooth
    # LINEAR gradient across demes; for the 2-D grid an arbitrary per-deme RANDOM offset
    # (a grid has no single ordering, so a gradient would be artificial). phenoB = constant
    # baseline. phenoC (drift-proof) is built below. All baselines are mean-zero so the
    # GLOBAL prevalence target is met by a single intercept solve.
    uq = np.unique(coord)
    udeme = (uq - uq.min()) / (uq.max() - uq.min() + 1e-12)
    if dem_name == "serial1d":
        base_by_deme = 0.8 * (udeme - 0.5)
    else:
        base_by_deme = rng.normal(0, 0.4, len(uq))
    base_by_deme = base_by_deme - base_by_deme.mean()
    deme_to_baseA = {dd: b for dd, b in zip(uq, base_by_deme)}
    baseA = np.array([deme_to_baseA[c] for c in coord])
    baseB = np.zeros(nI)

    # phenoC drift-proof liability: center the genetic liability WITHIN each deme so every
    # deme has an identical mean liability (=> identical prevalence after one intercept
    # solve). Drift in allele frequencies can still shift the PER-DEME mean genetic
    # liability under phenoA/phenoB; phenoC removes that mean shift entirely, so the only
    # thing that can differ across ancestries is the PGS->liability calibration SLOPE --
    # never the base rate. (Demeaning by a per-deme constant leaves every within-deme
    # covariance/variance unchanged, so true_slope_deme is identical across phenos.)
    liab_demean = true_liab.copy()
    for dd in uq:
        mk = coord == dd
        liab_demean[mk] = true_liab[mk] - true_liab[mk].mean()

    # ---- REAL P+T: build PGS once (on phenoA's y_binary, the realistic phenotype) ----
    # PGS is a property of genotype + the GWAS; reused for phenoB (same as deploying a
    # trained PGS regardless of which baseline-shift defines the outcome). Use synth's
    # probit liability for the GWAS phenotype so the trait the GWAS sees matches synth.
    linA = baseA + true_liab
    c0A = brentq(lambda c: norm.cdf((c + linA) / SIGMA_E).mean() - PREV, -20, 20)
    yA = ((c0A + linA + rng.normal(0, SIGMA_E, nI)) > 0).astype(int)
    work = tempfile.mkdtemp(prefix=f"realpt_{dem_name}_", dir=str(outdir))
    try:
        PGS_raw, pt_meta = real_pt(geno_prefix, nI, coord, train_deme, yA, pcs, npc,
                                   split, work, args.threads, args.mem_mb)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(geno_dir, ignore_errors=True)  # drop the big .bed once scored
    PGS_raw = standardize(PGS_raw)
    # standardize PGS_z on TRAIN-DEME-FIT rows ONLY (matches synth; no test leak)
    fitm = (split == "train-deme-fit")
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
        "n_total": int(nI), "n_per_deme": int(NPER), "n_per_deme_train": int(NPER_TRAIN),
        "npc": int(npc), "sigma_e": SIGMA_E, "prev_target": PREV,
        "pgs_train_mean": tr_mean, "pgs_train_sd": tr_sd, "pgs_mode": "realpt",
        "pgs_method": "real GWAS (plink2 logistic, PC covars) + plink1.9 LD clump "
                      f"(r2<{CLUMP_R2}, {CLUMP_KB}kb) + p-value thresholding; "
                      "GWAS+weights on train-deme-fit only (inner 80/20 GWAS/threshold-select); "
                      "train-deme-test held out",
        "params": {"mu": MU, "recomb": RECOMB, "n_chunks": int(args.chunks),
                   "chunk_bp": int(args.chunk_bp), "seqlen_bp": int(args.chunks * args.chunk_bp),
                   "n_loci_total": int(st["n_loci"]), "n_loci_common": int(st["Xpca"].shape[1])},
        "pt": pt_meta,
        "true_slope_by_deme": {int(d): round(v, 4) for d, v in true_slope_by_deme.items()},
        "fst_train_vs_near": round(fst_pair(st["Xpca"][coord == train_deme], st["Xpca"][coord == near_deme]), 5),
        "fst_train_vs_far": round(fst_pair(st["Xpca"][coord == train_deme], st["Xpca"][coord == far_deme]), 5),
        "far_deme": int(far_deme), "near_deme": int(near_deme),
        "phenos": {},
        "truth_columns": ["true_liab", "true_slope_deme", "intercept_deme", "p_true",
                          "true_hazard", "true_surv_at_admin", "surv_rate_true",
                          "surv_risk_5y_true", "surv_risk_10y_true"],
        "metric_rule": "Use *-deme-test rows only. Never compute within-deme AUC/C-index; "
                       "stratify calibration only.",
    }

    for pheno, base, liab in [("phenoA", baseA, true_liab),
                              ("phenoB", baseB, true_liab),
                              ("phenoC", np.zeros(nI), liab_demean)]:
        # Binary outcome via probit liability with environmental noise (matches synth).
        # phenoA: deme-varying baseline + genetic liability; phenoB: constant baseline +
        # genetic liability; phenoC: no baseline + drift-proof (per-deme-centered) liability.
        lin = base + liab
        c0 = brentq(lambda c: norm.cdf((c + lin) / SIGMA_E).mean() - PREV, -20, 20)
        intercept_deme = c0 + base                       # generative per-deme intercept
        p_true = norm.cdf((c0 + lin) / SIGMA_E)          # EXACT generative P(y=1)
        e = rng.normal(0, SIGMA_E, nI)
        L = c0 + lin + e
        y_binary = (L > 0).astype(int)

        # Survival from the SAME liability: exponential baseline, hazard ~ exp(eta_surv).
        eta_surv = lin - lin.mean()
        lam = 0.02 * np.exp(eta_surv)                    # true individual hazard rate
        true_haz = lam
        t_event = rng.exponential(1.0 / np.clip(lam, 1e-9, None))
        c_admin = float(np.quantile(t_event, 0.65))      # ~35% admin censoring
        surv_time = np.minimum(t_event, c_admin)
        surv_event = (t_event <= c_admin).astype(int)
        true_surv_at_admin = np.exp(-lam * c_admin)
        h5 = float(np.quantile(t_event, 0.30))
        h10 = float(np.quantile(t_event, 0.60))
        surv_rate_true = lam
        surv_risk_5y_true = 1.0 - np.exp(-lam * h5)
        surv_risk_10y_true = 1.0 - np.exp(-lam * h10)

        df = pd.DataFrame({
            "iid": np.arange(nI),
            "deme": coord.astype(int),
            "coord": coord,
            "dist_from_train": dist,
            "PGS_raw": PGS_raw,
            "PGS_z": PGS_z,
            "y_binary": y_binary,
            "surv_time": surv_time,
            "surv_event": surv_event,
            "true_liab": liab,
            "true_slope_deme": true_slope_deme,
            "intercept_deme": intercept_deme,
            "p_true": p_true,
            "true_hazard": true_haz,
            "true_surv_at_admin": true_surv_at_admin,
            "surv_rate_true": surv_rate_true,
            "surv_risk_5y_true": surv_risk_5y_true,
            "surv_risk_10y_true": surv_risk_10y_true,
            "is_train": is_train,
            "split": split,
            "pgs_mode": "realpt",
        })
        for i in range(npc):
            df.insert(5 + i, f"PC{i+1}", pcs[:, i])

        path = outdir / f"{dem_name}_{pheno}_realpt{TAG}.parquet"
        df.to_parquet(path, index=False)
        df.to_csv(outdir / f"{dem_name}_{pheno}_realpt{TAG}.csv", index=False)

        test = df[df["split"].astype(str).str.endswith("-deme-test")].copy()
        pooled_auc = (roc_auc_score(test["y_binary"], test["PGS_raw"])
                      if test["y_binary"].nunique() > 1 else float("nan"))
        pooled_c = harrell_c_index(test["surv_time"], test["surv_event"], test["PGS_raw"])
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
                "surv_risk_5y_mean_test": float(sub["surv_risk_5y_true"].mean()),
                "surv_risk_10y_mean_test": float(sub["surv_risk_10y_true"].mean()),
                "surv_event_rate_test": float(sub["surv_event"].mean()),
            })
        sanity = pd.DataFrame(per_deme).sort_values("dist")
        sanity_path = outdir / f"{dem_name}_{pheno}_realpt{TAG}.sanity.tsv"
        sanity.to_csv(sanity_path, sep="\t", index=False)
        sidecar["phenos"][pheno] = {
            "path": str(path), "sanity_path": str(sanity_path), "n_test": int(len(test)),
            "p_true_range_test": [float(test["p_true"].min()), float(test["p_true"].max())],
            "binary_prevalence_test": float(test["y_binary"].mean()),
            "surv_risk_5y_pooled": round(float(surv_risk_5y_true.mean()), 4),
            "surv_risk_10y_pooled": round(float(surv_risk_10y_true.mean()), 4),
            "surv_event_rate_test": float(test["surv_event"].mean()),
            "global_test_raw_pgs_auc_binary": float(pooled_auc),
            "global_test_raw_pgs_cindex_survival": float(pooled_c),
        }
        log(f"[{dem_name}/{pheno}] wrote {path}")
        log(f"  test n={len(test)} prev={test['y_binary'].mean():.3f} "
            f"surv_event={test['surv_event'].mean():.3f} raw_auc={pooled_auc:.3f} raw_c={pooled_c:.3f}")
        log(sanity[["deme", "n_test", "is_training_ancestry", "dist", "true_slope_deme",
                    "corr_pgs_true_liab_test", "mean_pgs_raw_test", "binary_prevalence_test"]].to_string(index=False))

    with open(outdir / f"{dem_name}_realpt{TAG}.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    log(f"[{dem_name}] sidecar -> {outdir / f'{dem_name}_realpt{TAG}.json'}  "
        f"P+T p<={pt_meta['best_p_threshold']:g} n_snps={pt_meta['n_snps']} "
        f"cal_auc={pt_meta['best_cal_auc']} fst_far={sidecar['fst_train_vs_far']}")


if __name__ == "__main__":
    main()
