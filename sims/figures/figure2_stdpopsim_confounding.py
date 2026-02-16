from __future__ import annotations

import argparse
import os
import time
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stdpopsim
import tskit
try:
    import numba as nb
except Exception:
    nb = None
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


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _rss_gb() -> float:
    try:
        import resource

        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux ru_maxrss is KiB; macOS/BSD is bytes.
        if rss > 10_000_000:
            return rss / (1024.0 ** 3)
        return rss / (1024.0 ** 2)
    except Exception:
        return float("nan")


def _stage_done(prefix: str, stage: str, started_at: float, extra: str = "") -> None:
    elapsed = time.perf_counter() - started_at
    tail = f" {extra}" if extra else ""
    _log(f"[{prefix}] {stage} done in {elapsed:.1f}s (rss_gb={_rss_gb():.2f}){tail}")


if nb is not None:
    @nb.njit(cache=True)
    def _accumulate_link_edges_numba(
        child_idx: np.ndarray,
        parent_src: np.ndarray,
        span: np.ndarray,
        hap_to_dip: np.ndarray,
        out_dip: np.ndarray,
    ) -> None:
        for i in range(child_idx.shape[0]):
            c = int(child_idx[i])
            s = int(parent_src[i])
            if c >= 0 and s >= 0:
                d = int(hap_to_dip[c])
                if d >= 0:
                    out_dip[d, s] += 0.5 * span[i]
else:
    def _accumulate_link_edges_numba(
        child_idx: np.ndarray,
        parent_src: np.ndarray,
        span: np.ndarray,
        hap_to_dip: np.ndarray,
        out_dip: np.ndarray,
    ) -> None:
        valid = (child_idx >= 0) & (parent_src >= 0)
        if np.any(valid):
            dip_idx = hap_to_dip[child_idx[valid]]
            w = 0.5 * span[valid]
            np.add.at(out_dip, (dip_idx, parent_src[valid]), w)


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
            return (base / "gnomon_sims_work" / "figure2").resolve()
    return out_dir / "work"


def _true_ancestry_proportions(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    pop_lookup: dict[int, str],
    ancestry_census_time: float,
    log_prefix: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _source_index_from_pop_name(name: str) -> int:
        u = str(name).upper()
        if "AFR" in u:
            return 0
        if "EUR" in u:
            return 1
        if "ASIA" in u or "EAS" in u:
            return 2
        return -1

    sample_nodes = ts.samples()
    n_hap = len(sample_nodes)
    n_dip = int(a_idx.shape[0])
    src_names = ["AFR", "EUR", "ASIA"]
    src_to_col = {name: i for i, name in enumerate(src_names)}
    dip_acc = np.zeros((n_dip, len(src_names)), dtype=np.float64)
    total_len = float(ts.sequence_length)
    sample_nodes_arr = np.asarray(sample_nodes, dtype=np.int32)
    n_nodes = int(ts.num_nodes)
    sample_lookup = np.full(n_nodes, -1, dtype=np.int64)
    sample_lookup[sample_nodes_arr.astype(np.int64)] = np.arange(n_hap, dtype=np.int64)
    hap_to_dip = np.full(n_hap, -1, dtype=np.int64)
    dip_ids = np.arange(n_dip, dtype=np.int64)
    hap_to_dip[np.asarray(a_idx, dtype=np.int64)] = dip_ids
    hap_to_dip[np.asarray(b_idx, dtype=np.int64)] = dip_ids

    node_pop = np.asarray(ts.tables.nodes.population, dtype=np.int32)
    node_time = np.asarray(ts.tables.nodes.time, dtype=np.float64)
    max_pop = max(pop_lookup.keys()) if len(pop_lookup) > 0 else -1
    pop_to_src = np.full(max_pop + 1, -1, dtype=np.int8) if max_pop >= 0 else np.full(0, -1, dtype=np.int8)
    for pop_id, pop_name in pop_lookup.items():
        src_idx = _source_index_from_pop_name(str(pop_name))
        if src_idx >= 0 and int(pop_id) >= 0:
            pop_to_src[int(pop_id)] = int(src_idx)
    node_src_from_pop = np.full(n_nodes, -1, dtype=np.int8)
    if pop_to_src.size > 0:
        ok = (node_pop >= 0) & (node_pop < pop_to_src.shape[0])
        node_src_from_pop[ok] = pop_to_src[node_pop[ok]]

    candidate_mask = (node_time > 0.0) & (node_src_from_pop >= 0)
    candidate_times = node_time[candidate_mask]
    if candidate_times.size == 0:
        prefix = log_prefix if log_prefix is not None else "fig2"
        _log(f"[{prefix}] WARNING: no source-labeled ancestor nodes found; ancestry proportions set to 0.0")
        return np.zeros(n_dip, dtype=np.float64), np.zeros(n_dip, dtype=np.float64), np.zeros(n_dip, dtype=np.float64)

    # Use the requested census time; if unavailable, use nearest available source-node time.
    uniq_times = np.unique(candidate_times)
    requested = float(ancestry_census_time)
    near_idx = int(np.argmin(np.abs(uniq_times - requested)))
    selected_time = float(uniq_times[near_idx])
    exact_match = bool(np.any(np.isclose(candidate_times, requested, rtol=0.0, atol=1e-9)))
    if (not exact_match) and log_prefix:
        _log(
            f"[{log_prefix}] Requested census_time={requested:.6f} not present; "
            f"using nearest available census_time={selected_time:.6f}"
        )

    anc_mask = np.isclose(node_time, selected_time, rtol=0.0, atol=1e-9) & (node_src_from_pop >= 0)
    anc_nodes = np.nonzero(anc_mask)[0].astype(np.int32, copy=False)
    if anc_nodes.size == 0:
        prefix = log_prefix if log_prefix is not None else "fig2"
        _log(
            f"[{prefix}] WARNING: no ancestors found at selected census_time={selected_time:.6f}; "
            "ancestry proportions set to 0.0"
        )
        return np.zeros(n_dip, dtype=np.float64), np.zeros(n_dip, dtype=np.float64), np.zeros(n_dip, dtype=np.float64)

    # Only census-time ancestors are valid ancestry sources for attribution.
    node_src = np.full(n_nodes, -1, dtype=np.int8)
    node_src[anc_nodes.astype(np.int64)] = node_src_from_pop[anc_nodes.astype(np.int64)]

    if log_prefix:
        _log(
            f"[{log_prefix}] ancestry proportions via link_ancestors "
            f"(haplotypes={n_hap}, diploids={n_dip}, trees={int(getattr(ts, 'num_trees', 0))}, "
            f"census_time={selected_time:.6f}, ancestors={anc_nodes.size})"
        )
        _log(f"[{log_prefix}] numba enabled for edge accumulation={nb is not None}")

    t0 = time.perf_counter()
    edges = ts.link_ancestors(sample_nodes_arr, anc_nodes)
    parents = np.asarray(edges.parent, dtype=np.int64)
    children = np.asarray(edges.child, dtype=np.int64)
    left = np.asarray(edges.left, dtype=np.float64)
    right = np.asarray(edges.right, dtype=np.float64)
    span = right - left
    child_idx = sample_lookup[children]
    parent_src = node_src[parents]
    _accumulate_link_edges_numba(child_idx, parent_src, span, hap_to_dip, dip_acc)
    valid = (child_idx >= 0) & (parent_src >= 0)
    if log_prefix:
        _stage_done(
            log_prefix,
            "link_ancestors combined",
            t0,
            extra=f"(edges={len(edges)}, matched_edges={int(np.count_nonzero(valid))})",
        )

    denom = np.where(total_len > 0, total_len, 1.0)
    props = dip_acc / denom
    afr = props[:, src_to_col["AFR"]]
    eur = props[:, src_to_col["EUR"]]
    asia = props[:, src_to_col["ASIA"]]
    s = afr + eur + asia
    bad = s <= 0
    if np.any(bad):
        n_bad = int(np.count_nonzero(bad))
        prefix = log_prefix if log_prefix is not None else "fig2"
        _log(
            f"[{prefix}] WARNING: unresolved ancestry for {n_bad} diploids "
            "(no AFR/EUR/ASIA source found on traced paths); leaving proportions as 0.0"
        )
        afr[bad] = 0.0
        eur[bad] = 0.0
        asia[bad] = 0.0
    good = ~bad
    if np.any(good):
        afr[good] /= s[good]
        eur[good] /= s[good]
        asia[good] /= s[good]
    return afr, eur, asia


def _simulate(
    seed: int,
    pgs_effects: np.ndarray,
    out_dir: Path,
    n_train_eur: int,
    n_test_per_pop: int,
    plink_threads: int,
    plink_memory_mb: int | None,
    msprime_model: str,
    pca_n_sites: int,
    causal_max_sites: int,
    bp_cap: int | None,
    ancestry_census_time: float,
) -> pd.DataFrame:
    run_id = f"fig2_s{seed}"
    run_t0 = time.perf_counter()
    _log(
        f"[{run_id}] Simulation start "
        f"(model={msprime_model}, n_train_eur={n_train_eur}, n_test_per_pop={n_test_per_pop}, "
        f"bp_cap={bp_cap}, ancestry_census_time={ancestry_census_time})"
    )
    _log(f"[{run_id}] Loading stdpopsim species/model")
    t0 = time.perf_counter()
    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("AmericanAdmixture_4B18")
    if bp_cap is not None:
        # For short debug runs, switch to a generic capped contig.
        contig = species.get_contig(
            chromosome=None,
            length=int(bp_cap),
            mutation_rate=model.mutation_rate,
            recombination_rate=1e-8,
        )
    else:
        contig = species.get_contig("chr22", genetic_map="HapMapII_GRCh38", mutation_rate=model.mutation_rate)
    engine = stdpopsim.get_engine("msprime")
    _stage_done(run_id, "stdpopsim setup", t0, extra=f"(sequence_length={int(contig.length)})")

    samples = {
        "AFR": n_test_per_pop,
        "EUR": n_train_eur + n_test_per_pop,
        "ASIA": n_test_per_pop,
        "ADMIX": n_test_per_pop,
    }

    simulate_kwargs = {
        "seed": seed,
        "msprime_model": msprime_model,
        "record_migrations": True,
    }
    _log(f"[{run_id}] Running stdpopsim/msprime simulate")
    t0 = time.perf_counter()
    ts = engine.simulate(model, contig, samples, **simulate_kwargs)
    _stage_done(
        run_id,
        "stdpopsim simulate",
        t0,
        extra=f"(nodes={ts.num_nodes}, edges={ts.num_edges}, sites={ts.num_sites}, trees={ts.num_trees})",
    )

    _log(f"[{run_id}] Extracting diploid sample mappings")
    t0 = time.perf_counter()
    a_idx, b_idx, pop_idx, ts_ind_id = diploid_index_pairs(ts)
    pop_lookup = pop_names_from_ts(ts)
    pop_label = np.array([pop_lookup.get(int(p), f"pop_{int(p)}") for p in pop_idx], dtype=object)
    _stage_done(run_id, "sample mapping", t0, extra=f"(diploids={len(pop_label)}, haploids={len(ts.samples())})")

    _log(f"[{run_id}] Sampling PCA sites (n={pca_n_sites}, maf_min=0.05)")
    t0 = time.perf_counter()
    pca_sites = sample_site_ids_for_maf(ts, a_idx, b_idx, n_sites=pca_n_sites, maf_min=0.05, seed=seed + 11)
    _stage_done(run_id, "sample PCA sites", t0, extra=f"(selected={len(pca_sites)})")
    _log(f"[{run_id}] Computing PCs (n_components={N_PCS})")
    t0 = time.perf_counter()
    pcs = pcs_from_sites(ts, a_idx, b_idx, pca_sites, seed=seed + 13, n_components=N_PCS)
    _stage_done(run_id, "compute PCs", t0)

    _log(f"[{run_id}] Sampling causal sites (max_n={causal_max_sites}, maf_min=0.01)")
    t0 = time.perf_counter()
    causal_sites = sample_site_ids_for_maf(
        ts,
        a_idx,
        b_idx,
        n_sites=min(causal_max_sites, int(ts.num_sites)),
        maf_min=0.01,
        seed=seed + 17,
    )
    _stage_done(run_id, "sample causal sites", t0, extra=f"(selected={len(causal_sites)})")
    _log(f"[{run_id}] Building genetic risk from causal sites (n={len(causal_sites)})")
    t0 = time.perf_counter()
    G_true = genetic_risk_from_real_pgs_effect_distribution(ts, a_idx, b_idx, causal_sites, pgs_effects, seed=seed + 19)
    _stage_done(run_id, "build genetic risk", t0)

    _log(f"[{run_id}] Computing true ancestry proportions")
    rng = np.random.default_rng(seed + 23)
    t0 = time.perf_counter()
    afr_prop, eur_prop, asia_prop = _true_ancestry_proportions(
        ts,
        a_idx,
        b_idx,
        pop_lookup,
        ancestry_census_time=ancestry_census_time,
        log_prefix=run_id,
    )
    _stage_done(run_id, "ancestry proportions", t0)

    _log(f"[{run_id}] Sampling phenotypes from liability model")
    t0 = time.perf_counter()
    env = 0.8 * afr_prop + 0.3 * asia_prop + 0.1 * eur_prop
    eta = 0.75 * G_true + env
    b0 = solve_intercept_for_prevalence(0.10, eta)
    y = rng.binomial(1, sigmoid(b0 + eta)).astype(np.int32)
    _stage_done(run_id, "sample phenotypes", t0)

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

    _log(f"[{run_id}] Building dataframe (n_rows={len(rows)})")
    t0 = time.perf_counter()
    df = pd.DataFrame(rows)
    _stage_done(run_id, "dataframe build", t0)

    prefix = out_dir / f"fig2_s{seed}"
    vcf = prefix.with_suffix(".vcf")
    _log(f"[{run_id}] Writing VCF to {vcf}")
    t0 = time.perf_counter()
    with open(vcf, "w") as f:
        names = [f"ind_{i+1}" for i in range(ts.num_individuals)]
        ts.write_vcf(f, individual_names=names, position_transform=lambda x: np.asarray(x) + 1)
    _stage_done(run_id, "write VCF", t0)
    _log(f"[{run_id}] Converting VCF to PLINK")
    t0 = time.perf_counter()
    run_plink_conversion(
        str(vcf),
        str(prefix),
        cm_map_path=None,
        threads=plink_threads,
        memory_mb=plink_memory_mb,
    )
    _stage_done(run_id, "PLINK conversion", t0)
    vcf.unlink(missing_ok=True)

    _log(f"[{run_id}] Writing simulation table to {prefix.with_suffix('.tsv')}")
    t0 = time.perf_counter()
    df.to_csv(prefix.with_suffix(".tsv"), sep="\t", index=False)
    _stage_done(run_id, "write simulation table", t0)
    _stage_done(run_id, "simulation stage total", run_t0)
    return df


def _run_bayesr_and_predict(
    df: pd.DataFrame,
    prefix: str,
    out_dir: Path,
    plink_threads: int,
    plink_memory_mb: int | None,
    bayesr_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import shutil
    import subprocess

    run_id = Path(prefix).name
    run_t0 = time.perf_counter()
    _log(f"[{run_id}] Preparing BayesR files in {out_dir}")
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, names=["FID", "IID", "PID", "MID", "SEX", "PHENO"], dtype=str)
    iid_to_fid = dict(zip(fam["IID"], fam["FID"]))

    train_ids = df.loc[df["group"] == "EUR_train", "IID"].astype(str).tolist()
    test_ids = df.loc[df["group"].str.endswith("_test"), "IID"].astype(str).tolist()

    pd.DataFrame({"FID": [iid_to_fid[i] for i in train_ids], "IID": train_ids}).to_csv(work / "train.keep", sep="\t", header=False, index=False)
    pd.DataFrame({"FID": [iid_to_fid[i] for i in test_ids], "IID": test_ids}).to_csv(work / "test.keep", sep="\t", header=False, index=False)
    _stage_done(run_id, "BayesR file prep", t0, extra=f"(n_train={len(train_ids)}, n_test={len(test_ids)})")

    plink = shutil.which("plink2") or "plink2"
    common = ["--threads", str(plink_threads)]
    if plink_memory_mb is not None and plink_memory_mb > 0:
        common.extend(["--memory", str(plink_memory_mb)])
    _log(f"[{run_id}] Running plink2 --freq")
    t0 = time.perf_counter()
    subprocess.run([plink, "--bfile", prefix, "--freq", *common, "--out", str(work / "ref")], check=True)
    _stage_done(run_id, "plink2 --freq", t0)
    _log(f"[{run_id}] Running plink2 train split")
    t0 = time.perf_counter()
    subprocess.run([plink, "--bfile", prefix, "--keep", str(work / "train.keep"), "--make-bed", *common, "--out", str(work / "train")], check=True)
    _stage_done(run_id, "plink2 train split", t0)
    _log(f"[{run_id}] Running plink2 test split")
    t0 = time.perf_counter()
    subprocess.run([plink, "--bfile", prefix, "--keep", str(work / "test.keep"), "--make-bed", *common, "--out", str(work / "test")], check=True)
    _stage_done(run_id, "plink2 test split", t0)

    train_df = df[df["IID"].isin(train_ids)].copy()
    t0 = time.perf_counter()
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
    _stage_done(run_id, "write BayesR phenotype/covariate files", t0)

    br = BayesR(threads=bayesr_threads, plink_memory_mb=plink_memory_mb)
    _log(f"[{run_id}] BayesR fit starting")
    t0 = time.perf_counter()
    eff = br.fit(str(work / "train"), str(work / "train.phen"), str(work / "bayesr"), covar_file=str(work / "train.covar"))
    _stage_done(run_id, "BayesR fit", t0)
    _log(f"[{run_id}] BayesR scoring train")
    t0 = time.perf_counter()
    train_scores = br.predict(str(work / "train"), eff, str(work / "bayesr_train"), freq_file=str(work / "ref.afreq"))
    _stage_done(run_id, "BayesR score train", t0)
    _log(f"[{run_id}] BayesR scoring test")
    t0 = time.perf_counter()
    test_scores = br.predict(str(work / "test"), eff, str(work / "bayesr_test"), freq_file=str(work / "ref.afreq"))
    _stage_done(run_id, "BayesR score test", t0)
    _stage_done(run_id, "BayesR/predict stage total", run_t0)
    return train_scores, test_scores


def _cleanup_seed_artifacts(prefix: Path, seed_work_dir: Path) -> None:
    _log(f"[{prefix.name}] Cleaning seed artifacts")
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
    parser.add_argument(
        "--msprime-model",
        choices=["dtwf", "hudson"],
        default="hudson",
        help="msprime ancestry model (hudson is usually faster).",
    )
    parser.add_argument("--pca-sites", type=int, default=2000)
    parser.add_argument("--causal-sites", type=int, default=5000)
    parser.add_argument("--bp-cap", type=int, default=None)
    parser.add_argument(
        "--ancestry-census-time",
        type=float,
        default=100.0,
        help="Census time (generations ago) used to define ancestry source nodes.",
    )
    parser.add_argument("--threads", type=int, default=None, help="Thread count for PLINK/BayesR. Default uses node allocation.")
    parser.add_argument("--memory-mb", type=int, default=None, help="PLINK memory limit (MB). Default auto-sizes from node RAM.")
    parser.add_argument("--work-root", default=None, help="Directory for heavy transient work files (e.g., RAM disk).")
    parser.add_argument("--keep-intermediates", action="store_true", help="Keep PLINK/GCTB intermediate files.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_root) if args.work_root else _default_work_root(out_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    _log("Figure2 GAM backend: mgcv")
    threads = max(1, int(args.threads)) if args.threads is not None else _default_total_threads()
    if args.memory_mb is not None:
        memory_mb = max(512, int(args.memory_mb))
    else:
        total_mb = _detect_total_mem_mb()
        memory_mb = max(2048, int(0.85 * total_mb)) if total_mb is not None else None
    _log(f"Figure2 resources: threads={threads} memory_mb={memory_mb}")
    _set_runtime_thread_env(threads)
    if memory_mb is not None:
        os.environ["PLINK_MEMORY_MB"] = str(memory_mb)

    score_path = ensure_pgs003725(Path(args.cache))
    _log(f"Figure2 loading PGS effects from cache={args.cache}")
    pgs_effects = load_pgs003725_effects(score_path, chr_filter="22")

    df = None
    prefix = None
    train_scores = None
    test_scores = None
    seed = int(args.seed)
    _log(f"[fig2_s{seed}] Starting full Figure2 pipeline")
    df = _simulate(
        seed,
        pgs_effects,
        out_dir,
        n_train_eur=int(args.n_train_eur),
        n_test_per_pop=int(args.n_test_per_pop),
        plink_threads=threads,
        plink_memory_mb=memory_mb,
        msprime_model=str(args.msprime_model),
        pca_n_sites=int(args.pca_sites),
        causal_max_sites=int(args.causal_sites),
        bp_cap=args.bp_cap,
        ancestry_census_time=float(args.ancestry_census_time),
    )
    prefix = str(out_dir / f"fig2_s{seed}")
    seed_work = work_root / f"work_s{seed}"
    _log(f"[fig2_s{seed}] Starting BayesR/prediction stage")
    train_scores, test_scores = _run_bayesr_and_predict(
        df,
        prefix,
        seed_work,
        plink_threads=threads,
        plink_memory_mb=memory_mb,
        bayesr_threads=threads,
    )

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
    _log(f"[fig2_s{seed}] Method fitting/prediction complete")

    rows = []
    for method, y_prob in preds.items():
        for pop in ["EUR", "AFR", "ASIA", "ADMIX"]:
            mask = test_df["pop_label"] == pop
            rows.append({"method": method, "population": pop, "auc": _auc(test_df.loc[mask, "y"].to_numpy(), y_prob[mask.to_numpy()])})

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "figure2_auc_by_method_population.tsv", sep="\t", index=False)
    _log("[fig2] Wrote figure2_auc_by_method_population.tsv")

    _log("[fig2] Plotting outputs")
    _plot_main(res, out_dir)
    _plot_prs_distribution(test_df, test_scores["PRS"].to_numpy(dtype=float), out_dir)
    _plot_pcs(df, out_dir)

    if not args.keep_intermediates:
        _cleanup_seed_artifacts(prefix=Path(prefix), seed_work_dir=seed_work)
    _log("[fig2] Figure2 run complete")


if __name__ == "__main__":
    main()
