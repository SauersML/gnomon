"""
Two-population simulations without stdpopsim.

Modes:
  - divergence: split into two populations, no gene flow
  - bottleneck: split, then one population undergoes a bottleneck + recovery

Usage:
  python sims/sim_two_pop.py <divergence|bottleneck> <divergence_gens>
"""
from __future__ import annotations

import sys
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import msprime
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid

from sim_pops import (
    _diploid_index_pairs,
    _reservoir_sample_sites_for_diploids,
    _compute_pcs_from_sites,
    _make_genetic_component,
    _solve_intercept_for_prevalence,
    _write_tsv,
    _plot_pcs,
)
from plink_utils import run_plink_conversion


@dataclass(frozen=True)
class TwoPopConfig:
    sim_name: str
    divergence_gens: int
    seed: int
    samples: Dict[str, int]  # pop label -> diploid count
    sequence_length: int
    recomb_rate: float
    mut_rate: float
    n_causal: int
    n_pca_sites: int
    maf_min_pca: float
    maf_min_causal: float
    h2_liability: float
    prevalence: float


GENS_LEVELS = [0, 20, 50, 100, 500, 1000, 5000, 10_000]


def _build_demography_divergence(split_time: int, ne: int) -> msprime.Demography:
    dem = msprime.Demography()
    dem.add_population(name="pop0", initial_size=ne)
    dem.add_population(name="pop1", initial_size=ne)
    dem.add_population(name="ancestral", initial_size=ne)
    dem.add_population_split(time=split_time, derived=["pop0", "pop1"], ancestral="ancestral")
    return dem


def _build_demography_bottleneck(split_time: int, ne: int, bottle_ne: int) -> msprime.Demography:
    dem = msprime.Demography()
    dem.add_population(name="pop0", initial_size=ne)
    dem.add_population(name="pop1", initial_size=ne)
    dem.add_population(name="ancestral", initial_size=ne)
    dem.add_population_split(time=split_time, derived=["pop0", "pop1"], ancestral="ancestral")

    # Bottleneck after split (more recent than split), then recover to NE by present.
    if split_time > 0:
        start = max(1, int(round(split_time * 0.5)))
        end = max(0, int(round(split_time * 0.1)))
        if end >= start:
            end = max(0, start - 1)
        dem.add_population_parameters_change(time=start, population="pop0", initial_size=bottle_ne)
        dem.add_population_parameters_change(time=end, population="pop0", initial_size=ne)
    # msprime requires events to be time-sorted; our inserts can be out of order.
    dem.sort_events()
    return dem


def _simulate_ts(cfg: TwoPopConfig) -> Tuple[msprime.TreeSequence, np.ndarray]:
    n0 = cfg.samples["POP0"]
    n1 = cfg.samples["POP1"]
    total = n0 + n1

    if cfg.divergence_gens == 0:
        ts = msprime.sim_ancestry(
            samples=[msprime.SampleSet(total, ploidy=2)],
            sequence_length=cfg.sequence_length,
            recombination_rate=cfg.recomb_rate,
            ploidy=2,
            population_size=10_000,
            random_seed=cfg.seed,
            model="dtwf",
        )
        ts = msprime.sim_mutations(ts, rate=cfg.mut_rate, random_seed=cfg.seed + 1)
        pop_labels = np.array(["POP0"] * n0 + ["POP1"] * n1, dtype=object)
        return ts, pop_labels

    split_time = int(cfg.divergence_gens)
    ne = 10_000
    bottle_ne = 1_000

    if cfg.sim_name == "divergence":
        dem = _build_demography_divergence(split_time, ne)
    else:
        dem = _build_demography_bottleneck(split_time, ne, bottle_ne)

    ts = msprime.sim_ancestry(
        samples={"pop0": n0, "pop1": n1},
        demography=dem,
        sequence_length=cfg.sequence_length,
        recombination_rate=cfg.recomb_rate,
        ploidy=2,
        random_seed=cfg.seed,
        model="dtwf",
    )
    ts = msprime.sim_mutations(ts, rate=cfg.mut_rate, random_seed=cfg.seed + 1)

    a_idx, b_idx, pop_idx, _ = _diploid_index_pairs(ts)
    labels = np.where(pop_idx == 0, "POP0", "POP1")
    if labels.shape[0] != (n0 + n1):
        labels = np.array(["POP0"] * n0 + ["POP1"] * n1, dtype=object)
    return ts, labels.astype(object)


def _simulate_dataset(cfg: TwoPopConfig) -> str:
    sim_prefix = f"{cfg.sim_name}_g{cfg.divergence_gens}_s{cfg.seed}"
    print(f"[{sim_prefix}] Simulating msprime two-pop model ...")

    ts, pop_labels = _simulate_ts(cfg)
    print(f"[{sim_prefix}] Done. ts.num_samples={ts.num_samples}, ts.num_sites={ts.num_sites}")

    a_idx, b_idx, _, ts_ind_id = _diploid_index_pairs(ts)
    n_ind = int(a_idx.shape[0])

    print(f"[{sim_prefix}] Diploids={n_ind} (POP0={cfg.samples['POP0']} POP1={cfg.samples['POP1']})")

    print(f"[{sim_prefix}] Selecting PCA sites (n={cfg.n_pca_sites}, maf>={cfg.maf_min_pca}) ...")
    pca_site_ids = _reservoir_sample_sites_for_diploids(
        ts=ts,
        a_idx=a_idx,
        b_idx=b_idx,
        n_sites=cfg.n_pca_sites,
        maf_min=cfg.maf_min_pca,
        seed=cfg.seed + 11,
    )

    n_causal_req = min(cfg.n_causal, max(1, int(ts.num_sites)))
    print(f"[{sim_prefix}] Selecting causal sites (n={n_causal_req}, maf>={cfg.maf_min_causal}) ...")
    causal_site_ids = _reservoir_sample_sites_for_diploids(
        ts=ts,
        a_idx=a_idx,
        b_idx=b_idx,
        n_sites=n_causal_req,
        maf_min=cfg.maf_min_causal,
        seed=cfg.seed + 17,
    )

    N_PCS = 20
    print(f"[{sim_prefix}] Computing {N_PCS} PCs (features={len(pca_site_ids)}) ...")
    pcs = _compute_pcs_from_sites(ts, a_idx, b_idx, pca_site_ids, seed=cfg.seed + 23, n_components=N_PCS)

    print(f"[{sim_prefix}] Building G_true from {len(causal_site_ids)} causal sites ...")
    G_true, causal_betas, causal_pos = _make_genetic_component(
        ts=ts,
        a_idx=a_idx,
        b_idx=b_idx,
        causal_site_ids=causal_site_ids,
        h2_liability=cfg.h2_liability,
        seed=cfg.seed + 29,
    )

    # Liability is purely genetic (no explicit PC effects).
    eta = G_true
    b0 = _solve_intercept_for_prevalence(cfg.prevalence, eta)
    p = sigmoid(b0 + eta)
    rng = np.random.default_rng(cfg.seed + 31)
    y = rng.binomial(1, p).astype(np.int32)

    P_obs = G_true + rng.normal(0.0, 0.35, size=n_ind).astype(np.float64)
    P_obs = StandardScaler(with_mean=True, with_std=True).fit_transform(P_obs.reshape(-1, 1)).ravel()

    extra_cols = ["attenuation_a", "noise_sigma", "center_majority_pc1"]
    extra_vals = (np.full(n_ind, np.nan), np.full(n_ind, np.nan), np.full(n_ind, np.nan))

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

    pop_index = np.where(pop_labels == "POP0", 0, 1).astype(int)

    rows: List[List[object]] = []
    for i in range(n_ind):
        pc_values = [float(pcs[k][i]) for k in range(N_PCS)]
        base = [
            f"ind_{i+1}",
            int(ts_ind_id[i]),
            int(pop_index[i]),
            str(pop_labels[i]),
        ] + pc_values + [
            float(G_true[i]),
            float(P_obs[i]),
            int(y[i]),
            sim_prefix,
            int(cfg.seed),
            cfg.sim_name,
            "chr22",
            "uniform",
            int(cfg.sequence_length),
            int(len(causal_site_ids)),
            float(cfg.h2_liability),
            float(cfg.prevalence),
            int(len(pca_site_ids)),
            float(cfg.maf_min_pca),
            float(cfg.maf_min_causal),
        ]
        base += [float(extra_vals[j][i]) for j in range(len(extra_vals))]
        rows.append(base)

    tsv_path = f"{sim_prefix}.tsv"
    vcf_path = f"{sim_prefix}.vcf"
    npz_path = f"{sim_prefix}_sites.npz"

    print(f"[{sim_prefix}] Writing {tsv_path} ...")
    _write_tsv(tsv_path, header, rows)

    print(f"[{sim_prefix}] Writing {vcf_path} (for PLINK conversion) ...")
    individual_names = [f"ind_{i+1}" for i in range(ts.num_individuals)]
    with open(vcf_path, "w") as f:
        # Ensure VCF positions are 1-based and avoid invalid position 0 records.
        ts.write_vcf(f, individual_names=individual_names, position_transform=lambda x: np.asarray(x) + 1)

    print(f"[{sim_prefix}] Writing {npz_path} (reproducibility: causal + PCA site IDs) ...")
    np.savez_compressed(
        npz_path,
        causal_site_id=np.asarray(causal_site_ids, dtype=np.int64),
        causal_position=causal_pos.astype(np.float64),
        causal_beta=causal_betas.astype(np.float64),
        pca_site_id=np.asarray(pca_site_ids, dtype=np.int64),
        seed=np.int64(cfg.seed),
        model_id=np.asarray([cfg.sim_name]),
        contig_id=np.asarray(["chr22"]),
        genetic_map=np.asarray(["uniform"]),
        samples=np.asarray([str(cfg.samples)], dtype=object),
    )

    plot_path = f"{sim_prefix}_pcs.png"
    print(f"[{sim_prefix}] Plotting PC1 vs PC2 -> {plot_path} ...")
    _plot_pcs(pcs[0], pcs[1], pop_labels, plot_path)

    run_plink_conversion(vcf_path, sim_prefix, cm_map_path=None)

    for path in [vcf_path]:
        try:
            import os
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    print(f"[{sim_prefix}] Done.")
    return sim_prefix


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


def _build_config(sim_name: str, divergence_gens: int, seed: int | None) -> TwoPopConfig:
    if sim_name not in ("divergence", "bottleneck"):
        raise ValueError("sim_name must be 'divergence' or 'bottleneck'")
    if divergence_gens not in GENS_LEVELS:
        raise ValueError(f"divergence_gens must be one of {GENS_LEVELS}")

    base_seed = 101 + divergence_gens + (0 if sim_name == "divergence" else 7)
    final_seed = seed if seed is not None else base_seed

    return TwoPopConfig(
        sim_name=sim_name,
        divergence_gens=divergence_gens,
        seed=final_seed,
        samples={"POP0": 1000, "POP1": 1000},
        sequence_length=5_000_000,
        recomb_rate=1e-8,
        mut_rate=1e-8,
        n_causal=5_000,
        n_pca_sites=2_000,
        maf_min_pca=0.05,
        maf_min_causal=0.01,
        h2_liability=0.50,
        prevalence=0.10,
    )


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Usage: python sims/sim_two_pop.py <divergence|bottleneck> <divergence_gens> [seed]")

    sim_name = sys.argv[1].strip().lower()
    divergence_gens = int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) == 4 else None
    cfg = _build_config(sim_name, divergence_gens, seed)

    if os.environ.get("SIM_FORCE", "").lower() not in ("1", "true", "yes"):
        sim_prefix = f"{cfg.sim_name}_g{cfg.divergence_gens}_s{cfg.seed}"
        if _final_outputs_exist(sim_prefix):
            print(f"[{sim_prefix}] Cached outputs found. Skipping simulation.")
            return

    _simulate_dataset(cfg)


if __name__ == "__main__":
    main()
