from __future__ import annotations

import sys
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import stdpopsim
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid
from scipy.optimize import brentq
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class GenomeSpec:
    # Human + full chr22 with HapMapII map (GRCh38).
    species_id: str = "HomSap"
    model_id: str = "AmericanAdmixture_4B18"
    contig_id: str = "chr22"
    genetic_map: str = "HapMapII_GRCh38"


@dataclass(frozen=True)
class TraitSpec:
    # "Polygenic" architecture for the *latent genetic component*.
    n_causal: int
    h2_liability: float
    prevalence: float


@dataclass(frozen=True)
class PCSpec:
    # PCA is computed on a random subset of common variants to avoid huge matrices.
    n_pca_sites: int
    maf_min: float


@dataclass(frozen=True)
class SiteSelectionSpec:
    # Separate MAF filter for causal sites (keeps effect sites reasonably informative).
    maf_min_causal: float


@dataclass(frozen=True)
class SimulationConfig:
    sim_name: str
    sim_prefix: str
    seed: int
    samples: Dict[str, int]  # population_name -> number of diploids
    genome: GenomeSpec
    trait: TraitSpec
    pc: PCSpec
    sites: SiteSelectionSpec
    msprime_recent_gens: int = 50  # dtwf for recent generations, then switch to hudson


# --------------------------------------------------------------------
# Simulation knobs (edit HERE if you want different cohort sizes)
# --------------------------------------------------------------------

GENOME = GenomeSpec()

SIM_CONFIGS: Dict[str, SimulationConfig] = {
    # Simulation 1: Ancestry confounding - signal preservation when mean liability shifts with ancestry (PC)
    "confounding": SimulationConfig(
        sim_name="confounding",
        sim_prefix="confounding",
        seed=101,
        samples={"AFR": 1300, "EUR": 1300, "ASIA": 1300, "ADMIX": 1300},
        genome=GENOME,
        trait=TraitSpec(n_causal=20_000, h2_liability=0.50, prevalence=0.10),
        pc=PCSpec(n_pca_sites=5_000, maf_min=0.05),
        sites=SiteSelectionSpec(maf_min_causal=0.01),
        msprime_recent_gens=50,
    ),
    # Simulation 3: Portability - train in EUR, test across ancestries with no PC-driven liability shifts.
    # EUR is 5x each other ancestry so that 20% EUR test ~= each other ancestry size.
    "portability": SimulationConfig(
        sim_name="portability",
        sim_prefix="portability",
        seed=303,
        samples={"AFR": 700, "EUR": 3500, "ASIA": 700, "ADMIX": 700},
        genome=GENOME,
        trait=TraitSpec(n_causal=20_000, h2_liability=0.50, prevalence=0.10),
        pc=PCSpec(n_pca_sites=5_000, maf_min=0.05),
        sites=SiteSelectionSpec(maf_min_causal=0.01),
        msprime_recent_gens=50,
    ),
}

# --------------------------------------------------------------------
# Core helpers (tskit-idiomatic, but memory-safe for full chr22)
# --------------------------------------------------------------------

def _simulate_tree_sequence(cfg: SimulationConfig):
    """
    Run stdpopsim using msprime engine for full chr22 with an empirical genetic map.
    """
    species = stdpopsim.get_species(cfg.genome.species_id)
    model = species.get_demographic_model(cfg.genome.model_id)

    # Full chr22; set mutation_rate to model's calibrated rate for realism.
    contig = species.get_contig(
        cfg.genome.contig_id,
        genetic_map=cfg.genome.genetic_map,
        mutation_rate=model.mutation_rate,
    )

    engine = stdpopsim.get_engine("msprime")

    ts = engine.simulate(
        model,
        contig,
        cfg.samples,
        seed=cfg.seed,
        msprime_model="dtwf",
        msprime_change_model=[(cfg.msprime_recent_gens, "hudson")],
    )
    return ts, model, contig


def _diploid_index_pairs(ts) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Map diploid individuals -> (haploid_sample_index_a, haploid_sample_index_b).

    Why this exists:
    - ts.variants() gives haploid genotypes over ts.samples().
    - We want *diploid dosages* (0/1/2) per individual without building ts.genotype_matrix().

    Returns
      a_idx, b_idx : int arrays of length n_diploid
      pop_idx      : population index per diploid
      ind_id       : tskit individual id per diploid (or -1 if synthetic)
    """
    sample_nodes = ts.samples()
    node_to_sample_col = {int(n): i for i, n in enumerate(sample_nodes)}

    a_list: List[int] = []
    b_list: List[int] = []
    pop_list: List[int] = []
    ind_list: List[int] = []

    # Preferred path: use tskit individuals (stdpopsim/msprime generally creates diploids).
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

    # Fallback: pair samples sequentially (still produces correct diploid dosages).
    if len(a_list) == 0:
        if len(sample_nodes) % 2 != 0:
            raise RuntimeError("Odd number of sample nodes; cannot pair into diploids.")
        for j in range(0, len(sample_nodes), 2):
            n0 = int(sample_nodes[j])
            n1 = int(sample_nodes[j + 1])
            a_list.append(j)
            b_list.append(j + 1)
            pop_list.append(int(ts.node(n0).population))
            ind_list.append(-1)

    a_idx = np.asarray(a_list, dtype=np.int64)
    b_idx = np.asarray(b_list, dtype=np.int64)
    pop_idx = np.asarray(pop_list, dtype=np.int32)
    ind_id = np.asarray(ind_list, dtype=np.int64)
    return a_idx, b_idx, pop_idx, ind_id


def _reservoir_sample_sites_for_diploids(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    n_sites: int,
    maf_min: float,
    seed: int,
) -> List[int]:
    """
    Uniform random sample of biallelic sites passing a diploid MAF filter, without
    storing all eligible sites (reservoir sampling).

    This scans through ts.variants() once and keeps only n_sites site IDs.
    """
    rng = np.random.default_rng(seed)

    reservoir: List[int] = []
    seen = 0

    n_ind = a_idx.shape[0]
    denom = 2.0 * float(n_ind)

    for var in ts.variants():
        # Keep PCA/causal selection stable: only simple biallelic sites.
        alleles = getattr(var, "alleles", None)
        if alleles is not None and len(alleles) != 2:
            continue

        g = var.genotypes
        # Treat any non-reference allele as 1 (biallelic path).
        # (msprime/stdpopsim typically encodes derived allele as 1.)
        g01 = (g == 1).astype(np.int8, copy=False)

        dos = g01[a_idx] + g01[b_idx]
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

    if len(reservoir) == 0:
        raise RuntimeError(f"No sites passed maf_min={maf_min}. Lower maf_min or change model/samples.")

    return reservoir


def _solve_intercept_for_prevalence(target_prev: float, eta_no_intercept: np.ndarray) -> float:
    """
    Find intercept b0 so that mean(sigmoid(b0 + eta)) = target_prev.
    Uses scipy brentq when available (robust root-finding).
    """
    if target_prev <= 0.0 or target_prev >= 1.0:
        raise ValueError("prevalence must be between 0 and 1")

    def f(b0: float) -> float:
        p = sigmoid(b0 + eta_no_intercept)
        return float(np.mean(p) - target_prev)

    # Wide bracket; sigmoid saturates hard, so this is safe in practice.
    lo, hi = -60.0, 60.0
    return float(brentq(f, lo, hi, maxiter=200))


def _compute_pcs_from_sites(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    pca_site_ids: List[int],
    seed: int,
    n_components: int = 20,
) -> List[np.ndarray]:
    """
    Compute PCs from diploid dosages at a subset of sites using scikit-learn PCA.
    This avoids building a genotype matrix over all chr22 variants.

    Returns
      List of PC arrays (length n_components), each standardized (mean 0, sd 1)
    """
    n_ind = a_idx.shape[0]
    n_feat = len(pca_site_ids)

    site_to_col = {sid: j for j, sid in enumerate(pca_site_ids)}
    X = np.empty((n_ind, n_feat), dtype=np.float32)
    pos = np.empty((n_feat,), dtype=np.float64)
    pos.fill(np.nan)

    filled = 0
    for var in ts.variants():
        sid = int(var.site.id)
        j = site_to_col.get(sid)
        if j is None:
            continue

        g = var.genotypes
        g01 = (g == 1).astype(np.int8, copy=False)
        dos = g01[a_idx] + g01[b_idx]
        X[:, j] = dos.astype(np.float32, copy=False)
        pos[j] = float(var.site.position)
        filled += 1
        if filled == n_feat:
            break

    if filled != n_feat:
        # This should be rare (only if some selected site IDs are missing due to filters).
        # Still, better to fail loudly than silently change the design.
        raise RuntimeError(f"PCA site fill failed: expected {n_feat}, filled {filled}.")

    # Standardize SNP columns then PCA.
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pcs = pca.fit_transform(Xz).astype(np.float64)

    # Standardize PC axes to make downstream functions stable.
    pc_scaler = StandardScaler(with_mean=True, with_std=True)
    pcs_z = pc_scaler.fit_transform(pcs)

    # Return all PCs as a list of arrays
    return [pcs_z[:, k].astype(np.float64) for k in range(n_components)]


def _make_genetic_component(
    ts,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    causal_site_ids: List[int],
    h2_liability: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a standardized 'true genetic component' G_true from n_causal sites.
    Effect sizes are Normal(0, sqrt(h2 / n_causal)).

    Returns
      G_true : standardized (mean 0, sd 1)
      causal_betas : effect sizes aligned with causal_site_ids
      causal_pos : positions aligned with causal_site_ids
    """
    rng = np.random.default_rng(seed)

    n_ind = a_idx.shape[0]
    n_causal = len(causal_site_ids)
    if n_causal == 0:
        raise RuntimeError("No causal sites provided.")

    beta_sd = math.sqrt(float(h2_liability) / float(n_causal))
    causal_betas = rng.normal(loc=0.0, scale=beta_sd, size=n_causal).astype(np.float64)

    beta_by_site = {int(sid): float(b) for sid, b in zip(causal_site_ids, causal_betas)}
    pos_by_site: Dict[int, float] = {}

    G = np.zeros((n_ind,), dtype=np.float64)

    for var in ts.variants():
        sid = int(var.site.id)
        b = beta_by_site.get(sid)
        if b is None:
            continue

        g = var.genotypes
        g01 = (g == 1).astype(np.int8, copy=False)
        dos = (g01[a_idx] + g01[b_idx]).astype(np.float64, copy=False)
        G += b * dos
        pos_by_site[sid] = float(var.site.position)

    # Standardize G_true for stable interpretation in later steps.
    G = StandardScaler(with_mean=True, with_std=True).fit_transform(G.reshape(-1, 1)).ravel()

    causal_pos = np.asarray([pos_by_site.get(int(sid), np.nan) for sid in causal_site_ids], dtype=np.float64)
    return G.astype(np.float64), causal_betas.astype(np.float64), causal_pos.astype(np.float64)


def _pop_label(model, pop_index: int) -> str:
    """
    stdpopsim models expose populations with names; fall back to pop_{k}.
    """
    try:
        return str(model.populations[pop_index].name)
    except Exception:
        return f"pop_{pop_index}"


def _write_tsv(path: str, header: List[str], rows: List[List[object]]) -> None:
    """
    Simple TSV writer with explicit header.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def _plot_pcs(
    pc1: np.ndarray,
    pc2: np.ndarray,
    pop_labels: np.ndarray,
    output_path: str,
) -> None:
    """
    Create a scatter plot of PC1 vs PC2 colored by population.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique populations and assign colors
    unique_pops = sorted(set(pop_labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pops)))
    color_map = {pop: colors[i] for i, pop in enumerate(unique_pops)}
    
    # Plot each population separately for proper legend
    for pop in unique_pops:
        mask = pop_labels == pop
        ax.scatter(
            pc1[mask],
            pc2[mask],
            c=[color_map[pop]],
            label=pop,
            alpha=0.6,
            s=30,
            edgecolors='white',
            linewidth=0.5,
        )
    
    ax.set_xlabel("PC1", fontsize=12, fontweight='bold')
    ax.set_ylabel("PC2", fontsize=12, fontweight='bold')
    ax.set_title("Population Structure (PC1 vs PC2)", fontsize=14, fontweight='bold')
    ax.legend(title="Population", loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {output_path}")


# --------------------------------------------------------------------
# Three simulation designs (paper-aligned)
# --------------------------------------------------------------------

def _simulate_dataset(cfg: SimulationConfig) -> None:
    """
    Shared pipeline:
      1) simulate full chr22 tree sequence
      2) construct diploid mapping
      3) select PCA and causal sites (MAF-filtered, reservoir sampling)
      4) compute PC1/PC2 (sklearn PCA on subset)
      5) compute G_true (polygenic component)
      6) generate P_observed and y according to sim 1/2/3
      7) write outputs
    """
    print(f"[{cfg.sim_prefix}] Simulating stdpopsim/msprime tree sequence for full {cfg.genome.contig_id} ...")
    ts, model, contig = _simulate_tree_sequence(cfg)
    print(f"[{cfg.sim_prefix}] Done. ts.num_samples={ts.num_samples}, ts.num_sites={ts.num_sites}")

    a_idx, b_idx, pop_idx, ts_ind_id = _diploid_index_pairs(ts)
    n_ind = int(a_idx.shape[0])
    print(f"[{cfg.sim_prefix}] Diploids={n_ind} (from {len(cfg.samples)} populations)")

    # --- Select sites for PCA and causal effects (streaming, memory-safe) ---
    # Separate seeds so design components don't unintentionally couple.
    print(f"[{cfg.sim_prefix}] Selecting PCA sites (n={cfg.pc.n_pca_sites}, maf>={cfg.pc.maf_min}) ...")
    pca_site_ids = _reservoir_sample_sites_for_diploids(
        ts=ts,
        a_idx=a_idx,
        b_idx=b_idx,
        n_sites=cfg.pc.n_pca_sites,
        maf_min=cfg.pc.maf_min,
        seed=cfg.seed + 11,
    )

    n_causal_req = min(cfg.trait.n_causal, max(1, int(ts.num_sites)))
    print(f"[{cfg.sim_prefix}] Selecting causal sites (n={n_causal_req}, maf>={cfg.sites.maf_min_causal}) ...")
    causal_site_ids = _reservoir_sample_sites_for_diploids(
        ts=ts,
        a_idx=a_idx,
        b_idx=b_idx,
        n_sites=n_causal_req,
        maf_min=cfg.sites.maf_min_causal,
        seed=cfg.seed + 17,
    )

    # --- Compute PCs ---
    N_PCS = 20  # Compute 20 PCs for use as BayesR covariates
    print(f"[{cfg.sim_prefix}] Computing {N_PCS} PCs with scikit-learn PCA (features={len(pca_site_ids)}) ...")
    pcs = _compute_pcs_from_sites(ts, a_idx, b_idx, pca_site_ids, seed=cfg.seed + 23, n_components=N_PCS)
    pc1 = pcs[0] # Keep pc1 handy for simulation logic

    # --- Compute true genetic component ---
    print(f"[{cfg.sim_prefix}] Building G_true from {len(causal_site_ids)} causal sites ...")
    G_true, causal_betas, causal_pos = _make_genetic_component(
        ts=ts,
        a_idx=a_idx,
        b_idx=b_idx,
        causal_site_ids=causal_site_ids,
        h2_liability=cfg.trait.h2_liability,
        seed=cfg.seed + 29,
    )

    # --- Generate observed PGS and phenotype y per simulation design ---
    rng = np.random.default_rng(cfg.seed + 31)

    if cfg.sim_name == "confounding":
        # Simulation 1:
        # - Liability mean shifts along ancestry axis (PC1), so naive ancestry-only normalization can remove signal.
        # - P_observed is a noisy proxy for G_true (like an imperfect PGS).
        eta = G_true + pc1

        b0 = _solve_intercept_for_prevalence(cfg.trait.prevalence, eta)
        p = sigmoid(b0 + eta)
        y = rng.binomial(1, p).astype(np.int32)

        P_obs = G_true + rng.normal(0.0, 0.35, size=n_ind).astype(np.float64)
        P_obs = StandardScaler(with_mean=True, with_std=True).fit_transform(P_obs.reshape(-1, 1)).ravel()

        extra_cols = ["attenuation_a", "noise_sigma", "center_majority_pc1"]
        extra_vals = (np.full(n_ind, np.nan), np.full(n_ind, np.nan), np.full(n_ind, np.nan))

    elif cfg.sim_name == "portability":
        # Simulation 3:
        # - No ancestry-dependent liability shift (pure genetic liability).
        eta = G_true

        b0 = _solve_intercept_for_prevalence(cfg.trait.prevalence, eta)
        p = sigmoid(b0 + eta)
        y = rng.binomial(1, p).astype(np.int32)

        P_obs = G_true + rng.normal(0.0, 0.35, size=n_ind).astype(np.float64)
        P_obs = StandardScaler(with_mean=True, with_std=True).fit_transform(P_obs.reshape(-1, 1)).ravel()

        extra_cols = ["attenuation_a", "noise_sigma", "center_majority_pc1"]
        extra_vals = (np.full(n_ind, np.nan), np.full(n_ind, np.nan), np.full(n_ind, np.nan))
    else:
        raise RuntimeError(f"Unknown sim_name in config: {cfg.sim_name}")

    # --- Build TSV table ---
    pc_cols = [f"pc{i+1}" for i in range(N_PCS)]
    header = [
        "individual_id",
        "tskit_individual_id",
        "pop_index",
        "pop_label",
    ] + pc_cols + [
        "G_true",
        "P_observed",
        "y",
        "sim",
        "seed",
        "model_id",
        "contig_id",
        "genetic_map",
        "sequence_length_bp",
        "n_causal",
        "h2_liability",
        "prevalence",
        "n_pca_sites",
        "pca_maf_min",
        "causal_maf_min",
    ] + extra_cols

    # Create population labels array for plotting
    pop_labels = np.array([_pop_label(model, int(pop_idx[i])) for i in range(n_ind)], dtype=object)

    rows: List[List[object]] = []
    for i in range(n_ind):
        pop_label = pop_labels[i]
        pc_values = [float(pcs[k][i]) for k in range(N_PCS)]
        base = [
            f"ind_{i+1}",  # Match VCF sample ID format (1-indexed with prefix)
            int(ts_ind_id[i]),
            int(pop_idx[i]),
            pop_label,
        ] + pc_values + [
            float(G_true[i]),
            float(P_obs[i]),
            int(y[i]),
            cfg.sim_name,
            cfg.seed,
            cfg.genome.model_id,
            cfg.genome.contig_id,
            cfg.genome.genetic_map,
            int(ts.sequence_length),
            int(len(causal_site_ids)),
            float(cfg.trait.h2_liability),
            float(cfg.trait.prevalence),
            int(len(pca_site_ids)),
            float(cfg.pc.maf_min),
            float(cfg.sites.maf_min_causal),
        ]
        base += [float(extra_vals[j][i]) for j in range(len(extra_vals))]
        rows.append(base)

    # --- Write outputs ---
    trees_path = f"{cfg.sim_prefix}.trees"
    tsv_path = f"{cfg.sim_prefix}.tsv"
    vcf_path = f"{cfg.sim_prefix}.vcf"
    npz_path = f"{cfg.sim_prefix}_sites.npz"

    print(f"[{cfg.sim_prefix}] Writing {trees_path} ...")
    ts.dump(trees_path)

    print(f"[{cfg.sim_prefix}] Writing {vcf_path} (for PLINK conversion) ...")
    # Generate individual names matching TSV individual_id so PLINK filtering works
    # PLINK2 doesn't allow sample ID '0', so we use 1-indexed IDs: "ind_1", "ind_2", etc.
    individual_names = [f"ind_{i+1}" for i in range(ts.num_individuals)]
    with open(vcf_path, "w") as f:
        # Ensure VCF positions are 1-based and avoid invalid position 0 records.
        ts.write_vcf(f, individual_names=individual_names, position_transform=lambda x: np.asarray(x) + 1)

    # --- Extract Genetic Map for Consistency ---
    # stdpopsim/msprime used a specific map (GRCh38). We must export this so downstream tools
    # use the EXACT same map, rather than approximating it from GRCh37.
    map_path = f"{cfg.sim_prefix}.cm"
    print(f"[{cfg.sim_prefix}] Writing {map_path} (Variant ID -> cM) ...")
    
    # contig.recombination_map is an msprime.RateMap
    # get_cumulative_mass(x) returns mass in Morgans. Convert to cM.
    rmap = contig.recombination_map
    
    with open(map_path, "w") as f:
        for var in ts.variants():
            pos = var.site.position
            # SNP ID in VCF is usually just "." but we need to match what PLINK will produce.
            # PLINK usually assigns IDs like "22:POS:REF:ALT" if missing, or we can trust the order.
            # However, run_plink_conversion ensures we are working with the same VCF.
            # Ideally, we map by Position.
            cm = rmap.get_cumulative_mass(pos) * 100.0
            # We'll write POS -> cM map.
            f.write(f"{int(pos)}\t{cm:.6f}\n")


    print(f"[{cfg.sim_prefix}] Writing {tsv_path} ...")
    _write_tsv(tsv_path, header, rows)

    print(f"[{cfg.sim_prefix}] Writing {npz_path} (reproducibility: causal + PCA site IDs) ...")
    np.savez_compressed(
        npz_path,
        causal_site_id=np.asarray(causal_site_ids, dtype=np.int64),
        causal_position=causal_pos.astype(np.float64),
        causal_beta=causal_betas.astype(np.float64),
        pca_site_id=np.asarray(pca_site_ids, dtype=np.int64),
        seed=np.int64(cfg.seed),
        model_id=np.asarray([cfg.genome.model_id]),
        contig_id=np.asarray([cfg.genome.contig_id]),
        genetic_map=np.asarray([cfg.genome.genetic_map]),
        samples=np.asarray([str(cfg.samples)], dtype=object),
    )

    # --- Generate PC plot ---
    plot_path = f"{cfg.sim_prefix}_pcs.png"
    print(f"[{cfg.sim_prefix}] Plotting PC1 vs PC2 -> {plot_path} ...")
    _plot_pcs(pcs[0], pcs[1], pop_labels, plot_path)

    print(f"[{cfg.sim_prefix}] Done.")


def _final_outputs_exist(sim_prefix: str) -> bool:
    required = [
        f"{sim_prefix}.tsv",
        f"{sim_prefix}_sites.npz",
        f"{sim_prefix}_pcs.png",
        f"{sim_prefix}.bed",
        f"{sim_prefix}.bim",
        f"{sim_prefix}.fam",
    ]
    return all(os.path.exists(p) for p in required)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python sims/sim_pops.py <confounding|portability> [seed] [--force]")

    which = sys.argv[1].strip().lower()
    if which not in SIM_CONFIGS:
        raise SystemExit("Unknown simulation. Use 'confounding' or 'portability'.")

    force = False
    seed: Optional[int] = None
    for arg in sys.argv[2:]:
        arg = arg.strip().lower()
        if arg in ("--force", "-f"):
            force = True
            continue
        if seed is not None:
            raise SystemExit("Usage: python sims/sim_pops.py <confounding|portability> [seed] [--force]")
        seed = int(arg)

    base_cfg = SIM_CONFIGS[which]
    final_seed = seed if seed is not None else base_cfg.seed
    sim_prefix = which if seed is None else f"{which}_s{final_seed}"

    cfg = SimulationConfig(
        sim_name=base_cfg.sim_name,
        sim_prefix=sim_prefix,
        seed=final_seed,
        samples=base_cfg.samples,
        genome=base_cfg.genome,
        trait=base_cfg.trait,
        pc=base_cfg.pc,
        sites=base_cfg.sites,
        msprime_recent_gens=base_cfg.msprime_recent_gens,
    )

    if not force:
        if _final_outputs_exist(cfg.sim_prefix):
            print(f"[{cfg.sim_prefix}] Cached outputs found. Skipping simulation.")
            return

    _simulate_dataset(cfg)
    
    # Convert to PLINK format and inject the correct genetic map
    from plink_utils import run_plink_conversion
    run_plink_conversion(f"{cfg.sim_prefix}.vcf", cfg.sim_prefix, cm_map_path=f"{cfg.sim_prefix}.cm")

    # Clean up large intermediate files (fail hard if deletion fails)
    if os.path.exists(f"{cfg.sim_prefix}.trees"):
        os.remove(f"{cfg.sim_prefix}.trees")
    if os.path.exists(f"{cfg.sim_prefix}.vcf"):
        os.remove(f"{cfg.sim_prefix}.vcf")
    if os.path.exists(f"{cfg.sim_prefix}.cm"):
        os.remove(f"{cfg.sim_prefix}.cm")
    print(f"[{cfg.sim_prefix}] Cleaned up intermediate files (.trees, .vcf, .cm).")


if __name__ == "__main__":
    main()
