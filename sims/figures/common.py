from __future__ import annotations

import gzip
import math
import urllib.request
from pathlib import Path
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
) -> list[int]:
    rng = np.random.default_rng(seed)
    reservoir: list[int] = []
    seen = 0
    denom = 2.0 * float(a_idx.shape[0])

    for var in ts.variants():
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

    if not reservoir:
        raise RuntimeError("No sites passed MAF filter")
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
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draw = rng.choice(real_effects, size=len(causal_site_ids), replace=True).astype(np.float64)
    betas = draw - float(np.mean(draw))
    sd = float(np.std(betas))
    if sd > 0:
        betas = betas / sd

    beta_by_site = {int(s): float(b) for s, b in zip(causal_site_ids, betas)}
    G = np.zeros(a_idx.shape[0], dtype=np.float64)

    for var in ts.variants():
        b = beta_by_site.get(int(var.site.id))
        if b is None:
            continue
        g = (var.genotypes == 1).astype(np.int8, copy=False)
        dos = (g[a_idx] + g[b_idx]).astype(np.float64, copy=False)
        G += b * dos

    G = StandardScaler(with_mean=True, with_std=True).fit_transform(G.reshape(-1, 1)).ravel()
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
