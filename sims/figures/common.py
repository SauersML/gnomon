from __future__ import annotations

import gzip
import time
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import msprime
import stdpopsim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PGS003725_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS003725/"
    "ScoringFiles/Harmonized/PGS003725_hmPOS_GRCh38.txt.gz"
)


def ensure_pgs003725(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / "PGS003725_hmPOS_GRCh38.txt.gz"
    if dst.exists():
        return dst
    urllib.request.urlretrieve(PGS003725_URL, dst)
    return dst


def load_pgs003725_effects(score_path: Path, chr_filter: str = "22") -> np.ndarray:
    with gzip.open(score_path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep="\t", comment="#")

    chr_col = "hm_chr"
    eff_col = "effect_weight"
    missing = [c for c in (chr_col, eff_col) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"PGS003725 missing required columns {missing}. Available={list(df.columns)}"
        )

    chr_series = df[chr_col].astype(str).str.replace("chr", "", regex=False)
    chr_df = df.loc[chr_series == str(chr_filter)].copy()
    if chr_df.empty:
        raise RuntimeError("PGS003725 has no chr22 rows after parsing")

    effects = pd.to_numeric(chr_df[eff_col], errors="coerce").dropna().to_numpy(dtype=np.float64)
    effects = effects[np.isfinite(effects)]
    if effects.size == 0:
        raise RuntimeError("No finite effect weights found in PGS003725 chr22")
    return effects


def diploid_index_pairs(ts) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_nodes = ts.samples()
    node_to_sample_col = {int(n): i for i, n in enumerate(sample_nodes)}

    a_list: list[int] = []
    b_list: list[int] = []
    pop_list: list[int] = []
    ind_list: list[int] = []

    if ts.num_individuals > 0:
        for ind_id in range(ts.num_individuals):
            ind = ts.individual(ind_id)
            nodes = list(ind.nodes)
            if len(nodes) != 2:
                continue
            n0, n1 = int(nodes[0]), int(nodes[1])
            if n0 not in node_to_sample_col or n1 not in node_to_sample_col:
                continue
            a_list.append(node_to_sample_col[n0])
            b_list.append(node_to_sample_col[n1])
            p0 = int(ts.node(n0).population)
            p1 = int(ts.node(n1).population)
            pop_list.append(p0 if p0 == p1 else p0)
            ind_list.append(ind_id)

    if ts.num_individuals == 0 or len(a_list) == 0:
        raise RuntimeError(
            "Tree sequence has no valid diploid individuals; sequential-node pairing fallback is disallowed."
        )

    return (
        np.asarray(a_list, dtype=np.int64),
        np.asarray(b_list, dtype=np.int64),
        np.asarray(pop_list, dtype=np.int32),
        np.asarray(ind_list, dtype=np.int64),
    )


def sample_two_site_sets_for_maf(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    pca_n_sites: int,
    pca_maf_min: float,
    pca_seed: int,
    causal_n_sites: int,
    causal_maf_min: float,
    causal_seed: int,
    log_fn: Callable[[str], None] | None = None,
    progress_label: str | None = None,
    progress_every_variants: int = 50000,
    progress_every_seconds: float = 20.0,
) -> tuple[list[int], list[int]]:
    """Single-pass site sampling for PCA and causal streams with separate reservoirs."""
    rng_pca = np.random.default_rng(pca_seed)
    rng_causal = np.random.default_rng(causal_seed)
    pca_cap = max(1, int(pca_n_sites))
    causal_cap = max(1, int(causal_n_sites))
    pca_res = np.empty(pca_cap, dtype=np.int32)
    causal_res = np.empty(causal_cap, dtype=np.int32)
    pca_fill = 0
    causal_fill = 0
    pca_seen = 0
    causal_seen = 0
    denom = 2.0 * float(a_idx.shape[0])
    total_sites = int(getattr(ts, "num_sites", 0))
    last_log_t = time.perf_counter()
    every_n = max(1, int(progress_every_variants))
    every_s = float(progress_every_seconds)

    if log_fn is not None:
        label = progress_label or "dual site scan"
        log_fn(
            f"[{label}] start: pca_target={pca_n_sites} (maf>={pca_maf_min}), "
            f"causal_target={causal_n_sites} (maf>={causal_maf_min}), total_sites={total_sites}"
        )

    for idx, var in enumerate(ts.variants(), start=1):
        alleles = getattr(var, "alleles", None)
        if alleles is not None and len(alleles) != 2:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        dos = g[a_idx] + g[b_idx]
        af = float(dos.sum()) / denom
        maf = af if af <= 0.5 else (1.0 - af)
        sid = int(var.site.id)

        if maf >= pca_maf_min:
            pca_seen += 1
            if pca_fill < pca_cap:
                pca_res[pca_fill] = sid
                pca_fill += 1
            else:
                j = int(rng_pca.integers(0, pca_seen))
                if j < pca_cap:
                    pca_res[j] = sid

        if maf >= causal_maf_min:
            causal_seen += 1
            if causal_fill < causal_cap:
                causal_res[causal_fill] = sid
                causal_fill += 1
            else:
                j = int(rng_causal.integers(0, causal_seen))
                if j < causal_cap:
                    causal_res[j] = sid

        if log_fn is not None:
            now_t = time.perf_counter()
            if (idx % every_n == 0) or (every_s > 0 and (now_t - last_log_t) >= every_s):
                pct = f"{(100.0 * idx / total_sites):.1f}%" if total_sites > 0 else "n/a"
                label = progress_label or "dual site scan"
                log_fn(
                    f"[{label}] progress: processed={idx}"
                    + (f"/{total_sites}" if total_sites > 0 else "")
                    + f" ({pct}), pca_pass={pca_seen}, pca_res={pca_fill}, "
                    f"causal_pass={causal_seen}, causal_res={causal_fill}"
                )
                last_log_t = now_t

    if pca_fill == 0:
        raise RuntimeError("No sites passed PCA MAF filter")
    if causal_fill == 0:
        raise RuntimeError("No sites passed causal MAF filter")
    if log_fn is not None:
        label = progress_label or "dual site scan"
        log_fn(
            f"[{label}] complete: pca_selected={pca_fill} from pass={pca_seen}; "
            f"causal_selected={causal_fill} from pass={causal_seen}"
        )
    return (
        pca_res[:pca_fill].astype(np.int64, copy=False).tolist(),
        causal_res[:causal_fill].astype(np.int64, copy=False).tolist(),
    )


def compute_pcs_risk_and_diagnostics(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    pop_labels: np.ndarray,
    pca_site_ids: list[int],
    n_pcs: int,
    pca_seed: int,
    causal_site_ids: list[int],
    real_effects: np.ndarray,
    causal_seed: int,
    log_fn: Callable[[str], None] | None = None,
    progress_label: str | None = None,
    progress_every_variants: int = 50000,
    progress_every_seconds: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, int, int, dict[str, float], set[int]]:
    """
    Single pass to compute:
    - PCA matrix for selected PCA sites, then PCs
    - standardized genetic risk (G_true) for selected causal sites
    - causal overlap/heterozygosity diagnostics
    """
    n_ind = int(a_idx.shape[0])
    total_sites = int(ts.num_sites)
    pca_col_by_site = np.full(total_sites, -1, dtype=np.int32)
    for j, sid in enumerate(pca_site_ids):
        pca_col_by_site[int(sid)] = int(j)
    X = np.empty((n_ind, len(pca_site_ids)), dtype=np.float32)
    pca_filled = 0

    rng = np.random.default_rng(causal_seed)
    draw = rng.choice(real_effects, size=len(causal_site_ids), replace=True).astype(np.float64)
    betas = draw - float(np.mean(draw))
    sd = float(np.std(betas))
    if sd > 0:
        betas = betas / sd
    beta_by_site = np.zeros(total_sites, dtype=np.float64)
    causal_mask_by_site = np.zeros(total_sites, dtype=np.bool_)
    for s, b in zip(causal_site_ids, betas):
        sid = int(s)
        beta_by_site[sid] = float(b)
        causal_mask_by_site[sid] = True
    beta_mask_by_site = beta_by_site != 0.0
    G = np.zeros(n_ind, dtype=np.float64)

    overlap_true_effect_sites = 0
    causal_pos_1based: set[int] = set()
    pop_labels_obj = np.asarray(pop_labels, dtype=object)
    pop_names = [str(x) for x in pd.unique(pop_labels_obj)]
    pop_to_code = {p: i for i, p in enumerate(pop_names)}
    pop_codes = np.array([pop_to_code[str(p)] for p in pop_labels_obj], dtype=np.int32)
    n_pop = len(pop_names)
    pop_counts = np.bincount(pop_codes, minlength=n_pop).astype(np.float64)
    pop_has_samples = pop_counts > 0
    pop_het_sum = np.zeros(n_pop, dtype=np.float64)
    pop_het_n = np.zeros(n_pop, dtype=np.int64)

    last_log_t = time.perf_counter()
    every_n = max(1, int(progress_every_variants))
    every_s = float(progress_every_seconds)
    if log_fn is not None:
        label = progress_label or "feature build"
        log_fn(
            f"[{label}] start: total_sites={total_sites}, pca_sites={len(pca_site_ids)}, "
            f"causal_sites={len(causal_site_ids)}"
        )

    for idx, var in enumerate(ts.variants(), start=1):
        sid = int(var.site.id)
        pca_col = int(pca_col_by_site[sid])
        beta_present = bool(beta_mask_by_site[sid])
        needs_diag = bool(causal_mask_by_site[sid])
        if pca_col < 0 and (not beta_present) and (not needs_diag):
            if log_fn is not None:
                now_t = time.perf_counter()
                if (idx % every_n == 0) or (every_s > 0 and (now_t - last_log_t) >= every_s):
                    pct = f"{(100.0 * idx / total_sites):.1f}%"
                    label = progress_label or "feature build"
                    log_fn(
                        f"[{label}] progress: processed={idx}/{total_sites} ({pct}), "
                        f"pca_filled={pca_filled}/{len(pca_site_ids)}, "
                        f"causal_overlap={overlap_true_effect_sites}/{len(causal_site_ids)}"
                    )
                    last_log_t = now_t
            continue

        alleles = getattr(var, "alleles", None)
        if alleles is not None and len(alleles) != 2:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        dos = g[a_idx] + g[b_idx]

        if pca_col >= 0:
            X[:, pca_col] = dos.astype(np.float32, copy=False)
            pca_filled += 1

        if beta_present:
            G += float(beta_by_site[sid]) * dos.astype(np.float64, copy=False)

        if needs_diag:
            overlap_true_effect_sites += 1
            causal_pos_1based.add(int(var.site.position) + 1)
            het = (dos == 1)
            het_sum = np.bincount(pop_codes, weights=het.astype(np.float64, copy=False), minlength=n_pop)
            het_mean = np.divide(het_sum, pop_counts, out=np.zeros_like(het_sum), where=pop_has_samples)
            pop_het_sum += het_mean
            pop_het_n += pop_has_samples.astype(np.int64, copy=False)

        if log_fn is not None:
            now_t = time.perf_counter()
            if (idx % every_n == 0) or (every_s > 0 and (now_t - last_log_t) >= every_s):
                pct = f"{(100.0 * idx / total_sites):.1f}%"
                label = progress_label or "feature build"
                log_fn(
                    f"[{label}] progress: processed={idx}/{total_sites} ({pct}), "
                    f"pca_filled={pca_filled}/{len(pca_site_ids)}, "
                    f"causal_overlap={overlap_true_effect_sites}/{len(causal_site_ids)}"
                )
                last_log_t = now_t

    if pca_filled != len(pca_site_ids):
        raise RuntimeError(f"PCA fill mismatch: expected {len(pca_site_ids)}, got {pca_filled}")

    Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=n_pcs, svd_solver="randomized", random_state=pca_seed)
    pcs = pca.fit_transform(Xz)
    pcs = StandardScaler(with_mean=True, with_std=True).fit_transform(pcs)

    G = StandardScaler(with_mean=True, with_std=True).fit_transform(G.reshape(-1, 1)).ravel()
    mean_het_by_pop = {}
    for i, p in enumerate(pop_names):
        mean_het_by_pop[p] = float(pop_het_sum[i] / pop_het_n[i]) if pop_het_n[i] > 0 else float("nan")
    if log_fn is not None:
        label = progress_label or "feature build"
        log_fn(
            f"[{label}] complete: pca_filled={pca_filled}/{len(pca_site_ids)}, "
            f"causal_overlap={overlap_true_effect_sites}/{len(causal_site_ids)}"
        )
    return (
        pcs.astype(np.float64),
        G.astype(np.float64),
        total_sites,
        overlap_true_effect_sites,
        mean_het_by_pop,
        causal_pos_1based,
    )


def solve_intercept_for_prevalence(target_prev: float, eta_no_intercept: np.ndarray) -> float:
    from scipy.optimize import brentq
    from scipy.special import expit as sigmoid

    if not (0.0 < target_prev < 1.0):
        raise ValueError("target_prev must be in (0,1)")

    def f(b0: float) -> float:
        return float(np.mean(sigmoid(b0 + eta_no_intercept)) - target_prev)

    return float(brentq(f, -60.0, 60.0, maxiter=300))


def get_chr22_recomb_map() -> msprime.RateMap:
    species = stdpopsim.get_species("HomSap")
    contig = species.get_contig("chr22", genetic_map="HapMapII_GRCh38")
    return contig.recombination_map


def pop_names_from_ts(ts) -> dict[int, str]:
    out: dict[int, str] = {}
    for p in ts.populations():
        md = p.metadata
        name = None
        if isinstance(md, dict):
            name = md.get("name")
        out[p.id] = str(name) if name else f"pop_{p.id}"
    return out
