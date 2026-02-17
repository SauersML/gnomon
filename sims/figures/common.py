from __future__ import annotations

import gzip
import time
import urllib.request
from pathlib import Path
from typing import Callable
from typing import Iterable, Sequence

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


def _pick_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower_to_real = {c.lower(): c for c in df.columns}
    for c in candidates:
        got = lower_to_real.get(c.lower())
        if got is not None:
            return got
    raise RuntimeError(f"Missing required column; tried={candidates}, available={list(df.columns)}")


def load_pgs003725_effects(score_path: Path, chr_filter: str = "22") -> np.ndarray:
    with gzip.open(score_path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, sep="\t", comment="#")

    chr_col = _pick_col(df, ["hm_chr", "chr_name", "chr", "chromosome"])
    eff_col = _pick_col(df, ["effect_weight", "effect", "beta", "weight"])

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

    if len(a_list) == 0:
        if len(sample_nodes) % 2 != 0:
            raise RuntimeError("Odd number of sample nodes; cannot pair into diploids")
        for j in range(0, len(sample_nodes), 2):
            n0 = int(sample_nodes[j])
            n1 = int(sample_nodes[j + 1])
            a_list.append(j)
            b_list.append(j + 1)
            pop_list.append(int(ts.node(n0).population))
            ind_list.append(-1)

    return (
        np.asarray(a_list, dtype=np.int64),
        np.asarray(b_list, dtype=np.int64),
        np.asarray(pop_list, dtype=np.int32),
        np.asarray(ind_list, dtype=np.int64),
    )


def sample_site_ids_for_maf(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    n_sites: int,
    maf_min: float,
    seed: int,
    log_fn: Callable[[str], None] | None = None,
    progress_label: str | None = None,
    progress_every_variants: int = 50000,
    progress_every_seconds: float = 20.0,
) -> list[int]:
    rng = np.random.default_rng(seed)
    reservoir: list[int] = []
    seen = 0
    denom = 2.0 * float(a_idx.shape[0])
    total_sites = int(getattr(ts, "num_sites", 0))
    last_log_t = time.perf_counter()
    every_n = max(1, int(progress_every_variants))
    every_s = float(progress_every_seconds)

    if log_fn is not None:
        label = progress_label or "site scan"
        log_fn(
            f"[{label}] start: target_sites={n_sites}, maf_min={maf_min}, "
            + (
                f"total_sites={total_sites}, progress_every={every_n} variants/{every_s:.0f}s"
                if total_sites > 0
                else f"progress_every={every_n} variants/{every_s:.0f}s"
            )
        )

    for idx, var in enumerate(ts.variants(), start=1):
        alleles = getattr(var, "alleles", None)
        if alleles is not None and len(alleles) != 2:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        dos = g[a_idx] + g[b_idx]
        af = float(dos.sum()) / denom
        maf = af if af <= 0.5 else (1.0 - af)
        if maf < maf_min:
            continue
        seen += 1
        sid = int(var.site.id)
        if len(reservoir) < n_sites:
            reservoir.append(sid)
        else:
            j = int(rng.integers(0, seen))
            if j < n_sites:
                reservoir[j] = sid
        if log_fn is not None:
            now_t = time.perf_counter()
            if (idx % every_n == 0) or (every_s > 0 and (now_t - last_log_t) >= every_s):
                pct = f"{(100.0 * idx / total_sites):.1f}%" if total_sites > 0 else "n/a"
                label = progress_label or "site scan"
                log_fn(
                    f"[{label}] progress: processed={idx}"
                    + (f"/{total_sites}" if total_sites > 0 else "")
                    + f" ({pct}), passing_maf={seen}, reservoir={len(reservoir)}"
                )
                last_log_t = now_t

    if not reservoir:
        raise RuntimeError("No sites passed MAF filter")
    if log_fn is not None:
        label = progress_label or "site scan"
        log_fn(f"[{label}] complete: selected={len(reservoir)} from passing_maf={seen}")
    return reservoir


def pcs_from_sites(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    site_ids: list[int],
    seed: int,
    n_components: int,
) -> np.ndarray:
    n_ind = a_idx.shape[0]
    site_to_col = {sid: j for j, sid in enumerate(site_ids)}
    X = np.empty((n_ind, len(site_ids)), dtype=np.float32)

    filled = 0
    for var in ts.variants():
        sid = int(var.site.id)
        j = site_to_col.get(sid)
        if j is None:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        X[:, j] = (g[a_idx] + g[b_idx]).astype(np.float32, copy=False)
        filled += 1
        if filled == len(site_ids):
            break

    if filled != len(site_ids):
        raise RuntimeError(f"PCA fill mismatch: expected {len(site_ids)}, got {filled}")

    Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pcs = pca.fit_transform(Xz)
    pcs = StandardScaler(with_mean=True, with_std=True).fit_transform(pcs)
    return pcs.astype(np.float64)


def genetic_risk_from_real_pgs_effect_distribution(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    causal_site_ids: list[int],
    real_effects: np.ndarray,
    seed: int,
    log_fn: Callable[[str], None] | None = None,
    progress_label: str | None = None,
    progress_every_variants: int = 50000,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draw = rng.choice(real_effects, size=len(causal_site_ids), replace=True).astype(np.float64)
    betas = draw - float(np.mean(draw))
    sd = float(np.std(betas))
    if sd > 0:
        betas = betas / sd

    beta_by_site = {int(s): float(b) for s, b in zip(causal_site_ids, betas)}
    G = np.zeros(a_idx.shape[0], dtype=np.float64)

    total_sites = int(getattr(ts, "num_sites", 0))
    used = 0
    last_log_t = time.perf_counter()
    every_n = max(1, int(progress_every_variants))
    if log_fn is not None:
        label = progress_label or "risk build"
        log_fn(
            f"[{label}] start: causal_sites={len(causal_site_ids)}, "
            + (f"total_sites={total_sites}, " if total_sites > 0 else "")
            + f"progress_every={every_n} variants/20s"
        )
    for idx, var in enumerate(ts.variants(), start=1):
        b = beta_by_site.get(int(var.site.id))
        if b is None:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        dos = (g[a_idx] + g[b_idx]).astype(np.float64, copy=False)
        G += b * dos
        used += 1
        if log_fn is not None:
            now_t = time.perf_counter()
            if (idx % every_n == 0) or (now_t - last_log_t >= 20.0):
                pct = f"{(100.0 * idx / total_sites):.1f}%" if total_sites > 0 else "n/a"
                label = progress_label or "risk build"
                log_fn(
                    f"[{label}] progress: processed={idx}"
                    + (f"/{total_sites}" if total_sites > 0 else "")
                    + f" ({pct}), matched_causal={used}/{len(causal_site_ids)}"
                )
                last_log_t = now_t

    G = StandardScaler(with_mean=True, with_std=True).fit_transform(G.reshape(-1, 1)).ravel()
    if log_fn is not None:
        label = progress_label or "risk build"
        log_fn(f"[{label}] complete: matched_causal={used}/{len(causal_site_ids)}")
    return G.astype(np.float64)


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


def summarize_true_effect_site_diagnostics(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    pop_labels: np.ndarray,
    causal_site_ids: list[int],
    log_fn: Callable[[str], None] | None = None,
    progress_label: str | None = None,
    progress_every_variants: int = 50000,
    progress_every_seconds: float = 20.0,
) -> tuple[int, int, dict[str, float], set[int]]:
    """
    Summarize overlap and heterozygosity diagnostics for true-effect (causal) sites.

    Returns:
      total_sites_in_ts: number of variant sites in the simulated sample.
      overlap_true_effect_sites: number of causal site IDs present in this ts.
      mean_het_by_pop: per-population mean heterozygosity across overlapping causal sites.
      causal_pos_1based: set of 1-based positions for overlapping causal sites.
    """
    total_sites_in_ts = int(ts.num_sites)
    causal_set = set(int(s) for s in causal_site_ids)
    pop_labels = np.asarray(pop_labels, dtype=object)
    pop_names = [str(x) for x in pd.unique(pop_labels)]
    pop_masks = {p: (pop_labels == p) for p in pop_names}
    pop_het_sum = {p: 0.0 for p in pop_names}
    pop_het_n = {p: 0 for p in pop_names}
    overlap_true_effect_sites = 0
    causal_pos_1based: set[int] = set()
    seen = 0
    every_n = max(1, int(progress_every_variants))
    every_s = float(progress_every_seconds)
    last_log_t = time.perf_counter()
    if log_fn is not None:
        label = progress_label or "true effect diagnostics"
        log_fn(
            f"[{label}] start: causal_sites={len(causal_set)}, total_sites={total_sites_in_ts}, "
            f"progress_every={every_n} variants/{every_s:.0f}s"
        )

    for idx, var in enumerate(ts.variants(), start=1):
        sid = int(var.site.id)
        if sid not in causal_set:
            if log_fn is not None:
                now_t = time.perf_counter()
                if (idx % every_n == 0) or (every_s > 0 and (now_t - last_log_t) >= every_s):
                    pct = f"{(100.0 * idx / total_sites_in_ts):.1f}%" if total_sites_in_ts > 0 else "n/a"
                    label = progress_label or "true effect diagnostics"
                    log_fn(
                        f"[{label}] progress: processed={idx}/{total_sites_in_ts} ({pct}), "
                        f"matched_causal={seen}"
                    )
                    last_log_t = now_t
            continue
        seen += 1
        overlap_true_effect_sites += 1
        causal_pos_1based.add(int(var.site.position) + 1)
        alleles = getattr(var, "alleles", None)
        if alleles is not None and len(alleles) != 2:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        dos = g[a_idx] + g[b_idx]
        het = (dos == 1)
        for p in pop_names:
            mask = pop_masks[p]
            n = int(np.count_nonzero(mask))
            if n == 0:
                continue
            pop_het_sum[p] += float(np.mean(het[mask]))
            pop_het_n[p] += 1
        if log_fn is not None:
            now_t = time.perf_counter()
            if (idx % every_n == 0) or (every_s > 0 and (now_t - last_log_t) >= every_s):
                pct = f"{(100.0 * idx / total_sites_in_ts):.1f}%" if total_sites_in_ts > 0 else "n/a"
                label = progress_label or "true effect diagnostics"
                log_fn(
                    f"[{label}] progress: processed={idx}/{total_sites_in_ts} ({pct}), "
                    f"matched_causal={seen}"
                )
                last_log_t = now_t

    mean_het_by_pop: dict[str, float] = {}
    for p in pop_names:
        n = pop_het_n[p]
        mean_het_by_pop[p] = float(pop_het_sum[p] / n) if n > 0 else float("nan")

    if log_fn is not None:
        label = progress_label or "true effect diagnostics"
        log_fn(
            f"[{label}] complete: matched_causal={seen}, "
            f"overlap_true_effect_sites={overlap_true_effect_sites}"
        )

    return total_sites_in_ts, overlap_true_effect_sites, mean_het_by_pop, causal_pos_1based
