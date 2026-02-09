#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_pops import _diploid_index_pairs
from sim_two_pop import _build_demography_bottleneck, _build_demography_divergence


@dataclass(frozen=True)
class Config:
    scenario: str
    gens: int
    seed: int
    n_per_pop: int
    seq_len: int
    recomb: float
    mut: float
    ne: int
    n_causal: int
    h2: float
    bottleneck_frac: float = 0.10


@dataclass
class Sim:
    X: np.ndarray
    maf: np.ndarray
    pop_idx: np.ndarray
    g_true: np.ndarray
    causal_idx: np.ndarray


def safe_r2(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1] ** 2)


def mean_ci(x: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(np.mean(x))
    if len(x) < 2:
        return m, m, m
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    return m, m - 1.96 * se, m + 1.96 * se


def fisher_ci(r: float, n: int) -> Tuple[float, float]:
    if not np.isfinite(r) or n <= 3:
        return np.nan, np.nan
    r = max(min(r, 0.999999), -0.999999)
    z = np.arctanh(r)
    se = 1.0 / math.sqrt(n - 3)
    z_lo = z - 1.96 * se
    z_hi = z + 1.96 * se
    return float(np.tanh(z_lo)), float(np.tanh(z_hi))


def simulate(cfg: Config) -> Sim:
    n0 = cfg.n_per_pop
    n1 = cfg.n_per_pop
    total = n0 + n1

    if cfg.gens == 0:
        ts = msprime.sim_ancestry(
            samples=[msprime.SampleSet(total, ploidy=2)],
            sequence_length=cfg.seq_len,
            recombination_rate=cfg.recomb,
            ploidy=2,
            population_size=cfg.ne,
            random_seed=cfg.seed,
            model="dtwf",
        )
        ts = msprime.sim_mutations(ts, rate=cfg.mut, random_seed=cfg.seed + 1)
        pop_idx = np.array([0] * n0 + [1] * n1, dtype=np.int32)
    else:
        if cfg.scenario == "divergence":
            dem = _build_demography_divergence(cfg.gens, cfg.ne)
        elif cfg.scenario == "bottleneck":
            bottle_ne = max(100, int(round(cfg.ne * float(cfg.bottleneck_frac))))
            dem = _build_demography_bottleneck(cfg.gens, cfg.ne, bottle_ne)
        else:
            raise ValueError(cfg.scenario)

        ts = msprime.sim_ancestry(
            samples={"pop0": n0, "pop1": n1},
            demography=dem,
            sequence_length=cfg.seq_len,
            recombination_rate=cfg.recomb,
            ploidy=2,
            random_seed=cfg.seed,
            model="dtwf",
        )
        ts = msprime.sim_mutations(ts, rate=cfg.mut, random_seed=cfg.seed + 1)
        _, _, pi, _ = _diploid_index_pairs(ts)
        pop_idx = pi if len(pi) == total else np.array([0] * n0 + [1] * n1, dtype=np.int32)

    m = ts.num_sites
    if m < 300:
        raise RuntimeError(f"Too few variants ({m})")

    X = np.empty((total, m), dtype=np.int8)
    maf = np.empty(m, dtype=np.float64)
    for j, var in enumerate(ts.variants()):
        gh = var.genotypes.astype(np.int16)
        gd = gh[0::2] + gh[1::2]
        X[:, j] = gd.astype(np.int8)
        p = gd.mean() / 2.0
        maf[j] = min(p, 1.0 - p)

    rng = np.random.default_rng(cfg.seed + 7)
    eligible = np.where(maf >= 0.01)[0]
    cidx = np.sort(rng.choice(eligible, size=min(cfg.n_causal, len(eligible)), replace=False))
    betas = rng.normal(0.0, math.sqrt(cfg.h2 / len(cidx)), size=len(cidx))

    g_raw = X[:, cidx].astype(float) @ betas
    g_true = (g_raw - g_raw.mean()) / (g_raw.std() if g_raw.std() > 1e-12 else 1.0)
    return Sim(X=X, maf=maf, pop_idx=pop_idx.astype(np.int32), g_true=g_true, causal_idx=cidx)


def split_pop0(pop0: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 100)
    sh = rng.permutation(pop0)
    n_train = int(0.6 * len(sh))
    return sh[:n_train], sh[n_train:]


def build_marginal_prs(
    X: np.ndarray,
    g: np.ndarray,
    train_idx: np.ndarray,
    score_idx: np.ndarray,
    tags: np.ndarray,
    use_score_stats: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    Xt = X[np.ix_(train_idx, tags)].astype(float)
    Xs = X[np.ix_(score_idx, tags)].astype(float)

    mu_t = Xt.mean(axis=0)
    sd_t = Xt.std(axis=0)
    sd_t[sd_t == 0] = 1.0

    Xt_z = (Xt - mu_t) / sd_t
    if use_score_stats:
        mu_s = Xs.mean(axis=0)
        sd_s = Xs.std(axis=0)
        sd_s[sd_s == 0] = 1.0
        Xs_z = (Xs - mu_s) / sd_s
    else:
        Xs_z = (Xs - mu_t) / sd_t

    gt = g[train_idx]
    betas = Xt_z.T @ (gt - gt.mean()) / len(train_idx)
    prs = Xs_z @ betas
    return prs, betas


def select_tag_panel(sim: Sim, n_array_sites: int) -> np.ndarray:
    n_total = int(sim.X.shape[1])
    if n_total < n_array_sites:
        raise RuntimeError(
            f"Insufficient variants for array-like panel: requested {n_array_sites}, available {n_total}"
        )
    step = max(1, n_total // n_array_sites)
    tags = np.arange(0, n_total, step, dtype=int)[:n_array_sites]
    if len(tags) != n_array_sites:
        raise RuntimeError(
            f"Array-like marker panel selection failed: requested {n_array_sites}, selected {len(tags)}"
        )
    return tags


def ld_destroyed_matrix(X: np.ndarray, score_idx: np.ndarray, tags: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 300)
    Xs = X[np.ix_(score_idx, tags)].copy().astype(float)
    for j in range(Xs.shape[1]):
        rng.shuffle(Xs[:, j])
    return Xs


def _mean_heterozygosity_from_matrix(X: np.ndarray) -> float:
    p = X.mean(axis=0) / 2.0
    return float(np.mean(2.0 * p * (1.0 - p)))


def _heterozygosity_resample_probs(
    X: np.ndarray, strength: float, increase: bool
) -> np.ndarray:
    X = np.asarray(X, dtype=np.int8)
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=float)
    if strength <= 0.0:
        return np.full(n, 1.0 / n, dtype=float)

    h = np.mean(X == 1, axis=1).astype(float)
    sd = float(np.std(h))
    if sd <= 1e-12:
        return np.full(n, 1.0 / n, dtype=float)

    z = (h - float(np.mean(h))) / sd
    logits = strength * z if increase else -strength * z
    logits = logits - float(np.max(logits))
    w = np.exp(logits)
    return w / float(np.sum(w))


def _effective_sample_size(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    if probs.size == 0:
        return 0.0
    denom = float(np.sum(probs ** 2))
    if denom <= 0.0:
        return 0.0
    return float(1.0 / denom)


def _resample_stability(
    X: np.ndarray,
    strength: float,
    increase: bool,
    min_ess_frac: float,
    min_ess_count: int,
) -> Tuple[bool, float, float]:
    probs = _heterozygosity_resample_probs(X, strength, increase)
    n = probs.size
    ess = _effective_sample_size(probs)
    min_ess_required = max(float(min_ess_count), float(min_ess_frac) * float(n))
    return ess >= min_ess_required, ess, min_ess_required


def _heterozygosity_biased_resample(
    X: np.ndarray, strength: float, increase: bool, rng: np.random.Generator
) -> np.ndarray:
    X = np.asarray(X, dtype=np.int8)
    n = X.shape[0]
    if n == 0:
        return X.copy()
    probs = _heterozygosity_resample_probs(X, strength, increase)
    idx = rng.choice(n, size=n, replace=True, p=probs)
    return X[idx].copy()


def _ld_corr_change(X_before: np.ndarray, X_after: np.ndarray) -> float:
    if X_before.shape[1] < 2 or X_before.shape[0] < 4:
        return np.nan

    def corr_no_divide_warnings(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, p = X.shape
        out = np.full((p, p), np.nan, dtype=float)
        sd = X.std(axis=0, ddof=1)
        valid = sd > 1e-12
        if not np.any(valid):
            return out

        Xv = X[:, valid]
        Xv = Xv - Xv.mean(axis=0)
        cov = (Xv.T @ Xv) / max(1, n - 1)
        denom = np.outer(sd[valid], sd[valid])
        corr = cov / denom
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)

        idx = np.where(valid)[0]
        out[np.ix_(idx, idx)] = corr
        return out

    c0 = corr_no_divide_warnings(X_before)
    c1 = corr_no_divide_warnings(X_after)
    iu = np.triu_indices(c0.shape[0], k=1)
    d = np.abs(c1[iu] - c0[iu])
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    return float(np.mean(d))


def one_run(
    cfg: Config,
    n_array_sites: int,
    min_resample_ess_frac: float,
    min_resample_ess_count: int,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    sim = simulate(cfg)

    pop0 = np.where(sim.pop_idx == 0)[0]
    pop1 = np.where(sim.pop_idx == 1)[0]
    train0, test0 = split_pop0(pop0, cfg.seed)

    tags = select_tag_panel(sim, n_array_sites=n_array_sites)
    prs_all_0, b_all = build_marginal_prs(sim.X, sim.g_true, train0, test0, tags, use_score_stats=False)
    prs_all_1, _ = build_marginal_prs(sim.X, sim.g_true, train0, pop1, tags, use_score_stats=False)

    # LD destruction: preserve per-marker distribution but scramble marker relationships in target.
    Xt = sim.X[np.ix_(train0, tags)].astype(float)
    mu_t = Xt.mean(axis=0)
    sd_t = Xt.std(axis=0)
    sd_t[sd_t == 0] = 1.0
    Xld = ld_destroyed_matrix(sim.X, pop1, tags, cfg.seed)
    prs_all_1_ld = ((Xld - mu_t) / sd_t) @ b_all

    rng = np.random.default_rng(cfg.seed + 500)
    g_perm = sim.g_true.copy()
    g_perm[train0] = rng.permutation(g_perm[train0])
    prs_null_0, _ = build_marginal_prs(sim.X, g_perm, train0, test0, tags, use_score_stats=False)
    prs_null_1, _ = build_marginal_prs(sim.X, g_perm, train0, pop1, tags, use_score_stats=False)

    # Compare marginal betas trained in pop0 vs pop1 on the same tag panel.
    pop1_train = pop1[:len(train0)] if len(pop1) >= len(train0) else pop1
    _, b_all_pop1 = build_marginal_prs(sim.X, sim.g_true, pop1_train, pop1_train, tags, use_score_stats=False)
    beta_corr_all = (
        float(np.corrcoef(b_all, b_all_pop1)[0, 1])
        if np.std(b_all) > 1e-12 and np.std(b_all_pop1) > 1e-12
        else np.nan
    )

    p0_all = sim.X[np.ix_(train0, tags)].mean(axis=0) / 2.0
    p1_all = sim.X[np.ix_(pop1, tags)].mean(axis=0) / 2.0
    h0_all = float(np.mean(2 * p0_all * (1 - p0_all)))
    h1_all = float(np.mean(2 * p1_all * (1 - p1_all)))

    r2_all_0 = safe_r2(prs_all_0, sim.g_true[test0])
    r2_all_1 = safe_r2(prs_all_1, sim.g_true[pop1])
    r2_all_1_ld = safe_r2(prs_all_1_ld, sim.g_true[pop1])

    ratio_all = r2_all_1 / r2_all_0 if r2_all_0 > 1e-8 else np.nan
    ratio_all_ld = r2_all_1_ld / r2_all_0 if r2_all_0 > 1e-8 else np.nan

    base_row = {
        "scenario": cfg.scenario,
        "divergence_gens": cfg.gens,
        "seed": cfg.seed,
        "bottleneck_frac": float(cfg.bottleneck_frac),
        "n_tags": len(tags),
        "r2_alltags_training_holdout": r2_all_0,
        "r2_alltags_target_population": r2_all_1,
        "ratio_alltags": ratio_all,
        "r2_alltags_target_with_destroyed_linkage": r2_all_1_ld,
        "ratio_alltags_ld_destroy": ratio_all_ld,
        "delta_lddestroy_minus_baseline": ratio_all_ld - ratio_all if np.isfinite(ratio_all_ld) and np.isfinite(ratio_all) else np.nan,
        "r2_null_training_holdout": safe_r2(prs_null_0, sim.g_true[test0]),
        "r2_null_target_population": safe_r2(prs_null_1, sim.g_true[pop1]),
        "beta_corr_alltags_training_vs_target": beta_corr_all,
        "delta_heterozygosity_alltags_target_minus_training": h1_all - h0_all,
        "mean_abs_maf_diff_alltags": float(np.mean(np.abs(p1_all - p0_all))),
    }

    sweep_rows: List[Dict[str, float]] = []
    # Conservative sweep: avoid extreme resampling weights that collapse ESS.
    strengths = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    strengths_run = 0
    strengths_skipped_unstable = 0

    X_train_all = sim.X[np.ix_(train0, tags)].astype(float)
    X_test_all = sim.X[np.ix_(test0, tags)].astype(float)
    X_target_all = sim.X[np.ix_(pop1, tags)].astype(float)
    X_train_all_i = sim.X[np.ix_(train0, tags)].astype(np.int8)
    X_test_all_i = sim.X[np.ix_(test0, tags)].astype(np.int8)
    X_target_all_i = sim.X[np.ix_(pop1, tags)].astype(np.int8)
    rng_sweep = np.random.default_rng(cfg.seed + 900)

    baseline_heterozygosity_training = _mean_heterozygosity_from_matrix(X_train_all_i)
    baseline_heterozygosity_target = _mean_heterozygosity_from_matrix(X_target_all_i)

    # Divergence scenario: lower target heterozygosity across a broad range.
    if cfg.scenario == "divergence":
        Xt = X_train_all
        mu_t = Xt.mean(axis=0)
        sd_t = Xt.std(axis=0)
        sd_t[sd_t == 0] = 1.0
        Xt_z = (Xt - mu_t) / sd_t
        gt = sim.g_true[train0]
        b = Xt_z.T @ (gt - gt.mean()) / len(train0)
        r2_holdout = safe_r2(((X_test_all - mu_t) / sd_t) @ b, sim.g_true[test0])

        for strength in strengths:
            stable, ess_target, min_ess_target = _resample_stability(
                X_target_all_i,
                strength,
                increase=False,
                min_ess_frac=min_resample_ess_frac,
                min_ess_count=min_resample_ess_count,
            )
            if not stable:
                strengths_skipped_unstable += 1
                continue
            X_target_new_i = _heterozygosity_biased_resample(X_target_all_i, strength, increase=False, rng=rng_sweep)
            X_target_new = X_target_new_i.astype(float)
            prs_target = ((X_target_new - mu_t) / sd_t) @ b
            r2_target = safe_r2(prs_target, sim.g_true[pop1])
            ratio = r2_target / r2_holdout if r2_holdout > 1e-8 else np.nan
            strengths_run += 1
            sweep_rows.append(
                {
                    "scenario": cfg.scenario,
                    "divergence_gens": cfg.gens,
                    "seed": cfg.seed,
                    "intervention": "decrease_target_heterozygosity",
                    "resample_strength": strength,
                    "baseline_heterozygosity_training": baseline_heterozygosity_training,
                    "baseline_heterozygosity_target": baseline_heterozygosity_target,
                    "mean_heterozygosity_training": _mean_heterozygosity_from_matrix(X_train_all_i),
                    "mean_heterozygosity_target": _mean_heterozygosity_from_matrix(X_target_new_i),
                    "heterozygosity_shift_target_minus_training": _mean_heterozygosity_from_matrix(X_target_new_i) - _mean_heterozygosity_from_matrix(X_train_all_i),
                    "r2_training_holdout": r2_holdout,
                    "r2_target_population": r2_target,
                    "transfer_ratio": ratio,
                    "ld_absolute_change": _ld_corr_change(X_target_all_i.astype(float), X_target_new),
                    "resample_ess_target": ess_target,
                    "resample_min_ess_required": min_ess_target,
                }
            )

    # Bottleneck scenario: raise training heterozygosity across a broad range.
    if cfg.scenario == "bottleneck":
        for strength in strengths:
            stable_train, ess_train, min_ess_train = _resample_stability(
                X_train_all_i,
                strength,
                increase=True,
                min_ess_frac=min_resample_ess_frac,
                min_ess_count=min_resample_ess_count,
            )
            stable_test, ess_test, min_ess_test = _resample_stability(
                X_test_all_i,
                strength,
                increase=True,
                min_ess_frac=min_resample_ess_frac,
                min_ess_count=min_resample_ess_count,
            )
            if not (stable_train and stable_test):
                strengths_skipped_unstable += 1
                continue
            X_train_new_i = _heterozygosity_biased_resample(X_train_all_i, strength, increase=True, rng=rng_sweep)
            X_test_new_i = _heterozygosity_biased_resample(X_test_all_i, strength, increase=True, rng=rng_sweep)
            X_train_new = X_train_new_i.astype(float)
            X_test_new = X_test_new_i.astype(float)

            mu_t = X_train_new.mean(axis=0)
            sd_t = X_train_new.std(axis=0)
            sd_t[sd_t == 0] = 1.0
            Xt_z = (X_train_new - mu_t) / sd_t
            gt = sim.g_true[train0]
            b = Xt_z.T @ (gt - gt.mean()) / len(train0)

            prs_holdout = ((X_test_new - mu_t) / sd_t) @ b
            prs_target = ((X_target_all - mu_t) / sd_t) @ b
            r2_holdout = safe_r2(prs_holdout, sim.g_true[test0])
            r2_target = safe_r2(prs_target, sim.g_true[pop1])
            ratio = r2_target / r2_holdout if r2_holdout > 1e-8 else np.nan
            strengths_run += 1

            sweep_rows.append(
                {
                    "scenario": cfg.scenario,
                    "divergence_gens": cfg.gens,
                    "seed": cfg.seed,
                    "intervention": "increase_training_heterozygosity",
                    "resample_strength": strength,
                    "baseline_heterozygosity_training": baseline_heterozygosity_training,
                    "baseline_heterozygosity_target": baseline_heterozygosity_target,
                    "mean_heterozygosity_training": _mean_heterozygosity_from_matrix(X_train_new_i),
                    "mean_heterozygosity_target": _mean_heterozygosity_from_matrix(X_target_all_i),
                    "heterozygosity_shift_target_minus_training": _mean_heterozygosity_from_matrix(X_target_all_i) - _mean_heterozygosity_from_matrix(X_train_new_i),
                    "r2_training_holdout": r2_holdout,
                    "r2_target_population": r2_target,
                    "transfer_ratio": ratio,
                    "ld_absolute_change": _ld_corr_change(X_train_all_i.astype(float), X_train_new),
                    "resample_ess_training": ess_train,
                    "resample_ess_holdout": ess_test,
                    "resample_min_ess_required_training": min_ess_train,
                    "resample_min_ess_required_holdout": min_ess_test,
                }
            )

    base_row["sweep_strengths_considered"] = float(len(strengths))
    base_row["sweep_strengths_run"] = float(strengths_run)
    base_row["sweep_strengths_skipped_unstable"] = float(strengths_skipped_unstable)
    return base_row, sweep_rows


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def run_chunk(args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = out_dir.parent / "hetero_sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [args.scenario] if args.scenario != "both" else ["divergence", "bottleneck"]
    seeds = parse_int_list(args.seeds)
    gens = parse_int_list(args.gens)

    cfgs: List[Config] = []
    for sc in scenarios:
        for g in gens:
            for s in seeds:
                cfgs.append(
                    Config(
                        scenario=sc,
                        gens=g,
                        seed=s,
                        n_per_pop=args.n_per_pop,
                        seq_len=args.seq_len,
                        recomb=1e-8,
                        mut=1e-8,
                        ne=10_000,
                        n_causal=args.n_causal,
                        h2=0.5,
                        bottleneck_frac=float(args.bottleneck_frac),
                    )
                )

    rows: List[Dict[str, float]] = []
    sweep_rows: List[Dict[str, float]] = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                one_run,
                cfg,
                args.n_array_sites,
                args.min_resample_ess_frac,
                args.min_resample_ess_count,
            )
            for cfg in cfgs
        ]
        for fut in cf.as_completed(futs):
            base_row, sweep_part = fut.result()
            rows.append(base_row)
            sweep_rows.extend(sweep_part)

    df = pd.DataFrame(rows).sort_values(
        ["scenario", "divergence_gens", "bottleneck_frac", "seed"]
    ).reset_index(drop=True)

    chunk_name = args.chunk_name or f"chunk_{args.scenario}_g{'-'.join(map(str, gens))}_s{'-'.join(map(str, seeds))}"
    path = out_dir / f"{chunk_name}.csv"
    df.to_csv(path, index=False)
    sweep_path = sweep_dir / f"{chunk_name}_hetero_sweep.csv"
    pd.DataFrame(sweep_rows).sort_values(["scenario", "divergence_gens", "seed", "resample_strength"]).to_csv(sweep_path, index=False)
    return path, sweep_path


def run_strength_chunk(args: argparse.Namespace) -> Tuple[Path, Path]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_dir = out_dir.parent / "strength_sweeps"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    bottleneck_fracs = parse_float_list(args.bottleneck_fracs)
    if not bottleneck_fracs:
        raise SystemExit("No bottleneck fractions provided")

    cfgs: List[Config] = []
    for frac in bottleneck_fracs:
        for seed in seeds:
            cfgs.append(
                Config(
                    scenario="bottleneck",
                    gens=int(args.gens),
                    seed=seed,
                    n_per_pop=args.n_per_pop,
                    seq_len=args.seq_len,
                    recomb=1e-8,
                    mut=1e-8,
                    ne=10_000,
                    n_causal=args.n_causal,
                    h2=0.5,
                    bottleneck_frac=float(frac),
                )
            )

    rows: List[Dict[str, float]] = []
    sweep_rows: List[Dict[str, float]] = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                one_run,
                cfg,
                args.n_array_sites,
                args.min_resample_ess_frac,
                args.min_resample_ess_count,
            )
            for cfg in cfgs
        ]
        for fut in cf.as_completed(futs):
            base_row, sweep_part = fut.result()
            rows.append(base_row)
            sweep_rows.extend(sweep_part)

    df = pd.DataFrame(rows).sort_values(["bottleneck_frac", "seed"]).reset_index(drop=True)
    chunk_name = args.chunk_name or f"strength_g{int(args.gens)}_s{'-'.join(map(str, seeds))}"
    path = out_dir / f"{chunk_name}.csv"
    df.to_csv(path, index=False)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_path = sweep_dir / f"{chunk_name}_strength_sweep.csv"
    if len(sweep_df) > 0:
        sweep_df = sweep_df.sort_values(["bottleneck_frac", "seed", "resample_strength"]).reset_index(drop=True)
    sweep_df.to_csv(sweep_path, index=False)
    return path, sweep_path


def summarize(df: pd.DataFrame, sweep_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "bottleneck_ld_mechanism_results.csv", index=False)

    summary = df.groupby(["scenario", "divergence_gens"]).agg(
        n=("seed", "count"),
        ratio_alltags_mean=("ratio_alltags", "mean"),
        ratio_alltags_ld_destroy_mean=("ratio_alltags_ld_destroy", "mean"),
        beta_corr_alltags_mean=("beta_corr_alltags_training_vs_target", "mean"),
    ).reset_index()
    summary.to_csv(out_dir / "bottleneck_ld_mechanism_summary.csv", index=False)

    # Pair divergence vs bottleneck by seed+gens
    key = ["seed", "divergence_gens"]
    div = df[df["scenario"] == "divergence"].set_index(key)
    bot = df[df["scenario"] == "bottleneck"].set_index(key)
    common = div.index.intersection(bot.index)
    div = div.loc[common]
    bot = bot.loc[common]

    paired = pd.DataFrame(index=common).reset_index()
    paired["added_harm_alltags"] = div["ratio_alltags"].values - bot["ratio_alltags"].values
    paired["beta_decorrelation"] = (
        div["beta_corr_alltags_training_vs_target"].values - bot["beta_corr_alltags_training_vs_target"].values
    )
    paired["hetero_shift_delta"] = (
        div["delta_heterozygosity_alltags_target_minus_training"].values
        - bot["delta_heterozygosity_alltags_target_minus_training"].values
    )
    paired.to_csv(out_dir / "bottleneck_paired_deltas.csv", index=False)

    corr = np.nan
    corr_lo, corr_hi = np.nan, np.nan
    tmp = paired[["added_harm_alltags", "beta_decorrelation"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) > 3 and np.std(tmp["added_harm_alltags"]) > 1e-12 and np.std(tmp["beta_decorrelation"]) > 1e-12:
        corr = float(np.corrcoef(tmp["added_harm_alltags"], tmp["beta_decorrelation"])[0, 1])
        corr_lo, corr_hi = fisher_ci(corr, len(tmp))

    tmp2 = paired[["added_harm_alltags", "hetero_shift_delta"]].replace([np.inf, -np.inf], np.nan).dropna()
    corr_het = np.nan
    if len(tmp2) > 3 and np.std(tmp2["added_harm_alltags"]) > 1e-12 and np.std(tmp2["hetero_shift_delta"]) > 1e-12:
        corr_het = float(np.corrcoef(tmp2["added_harm_alltags"], tmp2["hetero_shift_delta"])[0, 1])

    # Tests for intervention effects on the unified tag panel.
    d_ld = df["delta_lddestroy_minus_baseline"].dropna().values
    t_ld = ttest_1samp(d_ld, 0.0, nan_policy="omit")

    null0 = df["r2_null_training_holdout"].mean()
    null1 = df["r2_null_target_population"].mean()

    # Figures
    plt.rcParams.update({
        "figure.dpi": 180,
        "axes.facecolor": "#fafafa",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d0d0d0",
        "grid.alpha": 0.5,
        "axes.titleweight": "bold",
    })

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 5.0), constrained_layout=True)
    max_gen = 0
    y_lo_vals: List[float] = []
    y_hi_vals: List[float] = []
    for sc, color in [("divergence", "#1f77b4"), ("bottleneck", "#d62728")]:
        z = df[df["scenario"] == sc]
        gens_sorted = sorted(z["divergence_gens"].unique())
        if gens_sorted:
            max_gen = max(max_gen, int(max(gens_sorted)))
        xs, ms, lo, hi = [], [], [], []
        for g in gens_sorted:
            vals = z[z["divergence_gens"] == g]["ratio_alltags"].dropna().values
            if len(vals) == 0:
                continue
            m, l, h = mean_ci(vals)
            xs.append(g)
            ms.append(m)
            lo.append(l)
            hi.append(h)
        if not xs:
            continue
        ax.plot(xs, ms, "o-", color=color, linewidth=2.2, markersize=6, label=sc)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12)
        y_lo_vals.append(float(np.nanmin(np.asarray(lo, dtype=float))))
        y_hi_vals.append(float(np.nanmax(np.asarray(hi, dtype=float))))
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xscale("symlog", linthresh=20)
    if max_gen > 0:
        ticks = sorted(df["divergence_gens"].unique())
        ax.set_xlim(left=0, right=max(1.0, float(max_gen) * 1.1))
        ax.set_xticks(ticks)
    if y_lo_vals and y_hi_vals:
        y_min = float(np.nanmin(np.asarray(y_lo_vals, dtype=float)))
        y_max = float(np.nanmax(np.asarray(y_hi_vals, dtype=float)))
        y_span = y_max - y_min
        y_pad = 0.06 * y_span if y_span > 1e-12 else max(1e-6, abs(y_min) * 0.03)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title("Transfer performance using unified marker panel")
    ax.set_xlabel("Population split age (generations)")
    ax.set_ylabel("Transfer ratio\n(target population R^2 / same-ancestry holdout R^2)")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out_dir / "fig1_transfer_by_scenario.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 5.0), constrained_layout=True)
    max_gen = 0
    for sc, color in [("divergence", "#1f77b4"), ("bottleneck", "#d62728")]:
        z = df[df["scenario"] == sc]
        gens_sorted = sorted(z["divergence_gens"].unique())
        if gens_sorted:
            max_gen = max(max_gen, int(max(gens_sorted)))
        for metric, style, label in [
            ("ratio_alltags", "o-", "baseline"),
            ("ratio_alltags_ld_destroy", "d:", "target linkage structure destroyed"),
        ]:
            xs, ms, lo, hi = [], [], [], []
            for g in gens_sorted:
                vals = z[z["divergence_gens"] == g][metric].dropna().values
                if len(vals) == 0:
                    continue
                m, l, h = mean_ci(vals)
                xs.append(g)
                ms.append(m)
                lo.append(l)
                hi.append(h)
            if not xs:
                continue
            full_label = f"{sc}, {label}"
            ax.plot(xs, ms, style, color=color, linewidth=2.1, markersize=6, label=full_label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.08)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xscale("symlog", linthresh=20)
    if max_gen > 0:
        ticks = sorted(df["divergence_gens"].unique())
        ax.set_xlim(left=0, right=max(1.0, float(max_gen) * 1.1))
        ax.set_xticks(ticks)
    ax.set_title("Intervention comparison for unified marker panel")
    ax.set_xlabel("Population split age (generations)")
    ax.set_ylabel("Transfer ratio\n(target population R^2 / same-ancestry holdout R^2)")
    ax.grid(True)
    ax.set_ylim(bottom=-0.05)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out_dir / "fig2_ld_vs_hetero_interventions.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), constrained_layout=True)
    t = paired[["added_harm_alltags", "beta_decorrelation"]].replace([np.inf, -np.inf], np.nan).dropna()
    axes[0].scatter(t["beta_decorrelation"], t["added_harm_alltags"], s=28, alpha=0.75, color="#1f77b4", edgecolor="white", linewidth=0.4)
    if len(t) > 3 and np.std(t["beta_decorrelation"]) > 1e-12:
        b = np.polyfit(t["beta_decorrelation"], t["added_harm_alltags"], 1)
        xx = np.linspace(t["beta_decorrelation"].min(), t["beta_decorrelation"].max(), 200)
        axes[0].plot(xx, b[0] * xx + b[1], color="#1f77b4", ls="--", linewidth=1.8)
    axes[0].set_xlabel("Difference in marker-effect correlation\n(divergence - bottleneck)")
    axes[0].set_ylabel("Additional bottleneck harm\n(divergence all-tag ratio - bottleneck all-tag ratio)")
    axes[0].set_title(f"Additional bottleneck harm vs effect-size decorrelation\n(correlation={corr:.3f})")
    axes[0].grid(True)

    t = paired[["added_harm_alltags", "hetero_shift_delta"]].replace([np.inf, -np.inf], np.nan).dropna()
    axes[1].scatter(t["hetero_shift_delta"], t["added_harm_alltags"], s=28, alpha=0.75, color="#d62728", edgecolor="white", linewidth=0.4)
    if len(t) > 3 and np.std(t["hetero_shift_delta"]) > 1e-12:
        b = np.polyfit(t["hetero_shift_delta"], t["added_harm_alltags"], 1)
        xx = np.linspace(t["hetero_shift_delta"].min(), t["hetero_shift_delta"].max(), 200)
        axes[1].plot(xx, b[0] * xx + b[1], color="#d62728", ls="--", linewidth=1.8)
    axes[1].set_xlabel("Difference in heterozygosity shift\n(divergence - bottleneck)")
    axes[1].set_ylabel("Additional bottleneck harm")
    axes[1].set_title(f"Additional bottleneck harm vs heterozygosity shift\n(correlation={corr_het:.3f})")
    axes[1].grid(True)

    fig.savefig(out_dir / "fig3_added_harm_correlates.png", dpi=220)
    plt.close(fig)

    # Single-curve summary analogous to rigorous fig3 (no n_causal stratification).
    harm_rows = []
    for g in sorted(paired["divergence_gens"].unique()):
        vals = paired.loc[paired["divergence_gens"] == g, "added_harm_alltags"].dropna().values
        if len(vals) == 0:
            continue
        m, l, h = mean_ci(vals)
        harm_rows.append((g, m, l, h))

    if harm_rows:
        xs = [r[0] for r in harm_rows]
        ms = [r[1] for r in harm_rows]
        lo = [r[2] for r in harm_rows]
        hi = [r[3] for r in harm_rows]

        fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
        ax.plot(xs, ms, "o-", color="#2ca02c", linewidth=2.2, markersize=6)
        ax.fill_between(xs, lo, hi, color="#2ca02c", alpha=0.12)
        ax.axhline(0.0, color="gray", ls=":")
        ax.set_xscale("symlog", linthresh=20)
        ax.set_xlabel("Divergence generations")
        ax.set_ylabel("Added bottleneck harm\n(divergence ratio - bottleneck ratio)")
        ax.set_title("Bottleneck added harm across divergence")
        ax.grid(True)
        fig.savefig(out_dir / "fig3_bottleneck_added_harm.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(7.8, 4.8), constrained_layout=True)
    for sc, color in [("divergence", "#1f77b4"), ("bottleneck", "#d62728")]:
        vals = (
            df[df["scenario"] == sc]["delta_heterozygosity_alltags_target_minus_training"]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .values
        )
        if len(vals) < 2:
            continue
        sd = float(np.std(vals, ddof=1))
        if sd <= 1e-12:
            continue
        n = len(vals)
        bw = 1.06 * sd * (n ** (-1.0 / 5.0))
        xs = np.linspace(float(vals.min() - 3 * bw), float(vals.max() + 3 * bw), 400)
        dens = np.exp(-0.5 * ((xs[:, None] - vals[None, :]) / bw) ** 2).sum(axis=1)
        dens /= (n * bw * np.sqrt(2 * np.pi))
        ax.plot(xs, dens, color=color, linewidth=2.2, label=sc)
        ax.fill_between(xs, 0, dens, color=color, alpha=0.12)
    ax.set_xlabel("Heterozygosity shift (target population - training population)")
    ax.set_ylabel("Density")
    ax.set_title("Heterozygosity distribution by scenario")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_dir / "fig4_heterozygosity_distributions.png", dpi=220)
    plt.close(fig)

    if len(sweep_df) > 0:
        sweep_df.to_csv(out_dir / "bottleneck_heterozygosity_sweep_results.csv", index=False)

        fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.0), constrained_layout=True)
        x_lo_vals: List[float] = []
        x_hi_vals: List[float] = []
        y_lo_vals: List[float] = []
        y_hi_vals: List[float] = []
        for (sc, intervention), color, marker in [
            (("divergence", "decrease_target_heterozygosity"), "#1f77b4", "o"),
            (("bottleneck", "increase_training_heterozygosity"), "#d62728", "s"),
        ]:
            z = sweep_df[(sweep_df["scenario"] == sc) & (sweep_df["intervention"] == intervention)]
            if len(z) == 0:
                continue
            if intervention == "increase_training_heterozygosity":
                z = z[z["mean_heterozygosity_training"] >= z["baseline_heterozygosity_training"]]
                grp = z.groupby("resample_strength", as_index=False).agg(
                    mean_heterozygosity=("mean_heterozygosity_training", "mean"),
                    transfer_ratio_mean=("transfer_ratio", "mean"),
                    transfer_ratio_sd=("transfer_ratio", "std"),
                    n=("transfer_ratio", "count"),
                )
            else:
                z = z[z["mean_heterozygosity_target"] <= z["baseline_heterozygosity_target"]]
                grp = z.groupby("resample_strength", as_index=False).agg(
                    mean_heterozygosity=("mean_heterozygosity_target", "mean"),
                    transfer_ratio_mean=("transfer_ratio", "mean"),
                    transfer_ratio_sd=("transfer_ratio", "std"),
                    n=("transfer_ratio", "count"),
                )

            if len(grp) == 0:
                continue
            grp = grp.sort_values("mean_heterozygosity")

            sem = grp["transfer_ratio_sd"].fillna(0.0) / np.sqrt(np.maximum(grp["n"], 1))
            lo = grp["transfer_ratio_mean"] - 1.96 * sem
            hi = grp["transfer_ratio_mean"] + 1.96 * sem
            label = (
                "divergence: lower target heterozygosity"
                if sc == "divergence"
                else "bottleneck: raise training heterozygosity"
            )
            ax.plot(
                grp["mean_heterozygosity"],
                grp["transfer_ratio_mean"],
                marker=marker,
                color=color,
                linewidth=2.2,
                markersize=6,
                label=label,
            )
            ax.fill_between(grp["mean_heterozygosity"], lo, hi, color=color, alpha=0.12)
            x_lo_vals.append(float(np.nanmin(grp["mean_heterozygosity"].values)))
            x_hi_vals.append(float(np.nanmax(grp["mean_heterozygosity"].values)))
            y_lo_vals.append(float(np.nanmin(lo.values)))
            y_hi_vals.append(float(np.nanmax(hi.values)))
        ax.axhline(1.0, color="gray", ls=":")
        if x_lo_vals and x_hi_vals:
            x_min = float(np.nanmin(np.asarray(x_lo_vals, dtype=float)))
            x_max = float(np.nanmax(np.asarray(x_hi_vals, dtype=float)))
            x_span = x_max - x_min
            x_pad = 0.03 * x_span if x_span > 1e-12 else max(1e-6, abs(x_min) * 0.03)
            ax.set_xlim(x_min - x_pad, x_max + x_pad)
        if y_lo_vals and y_hi_vals:
            y_min = float(np.nanmin(np.asarray(y_lo_vals, dtype=float)))
            y_max = float(np.nanmax(np.asarray(y_hi_vals, dtype=float)))
            y_span = y_max - y_min
            y_pad = 0.06 * y_span if y_span > 1e-12 else max(1e-6, abs(y_min) * 0.03)
            ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.set_xlabel("Mean heterozygosity of manipulated population")
        ax.set_ylabel("Transfer ratio\n(target population R^2 / same-ancestry holdout R^2)")
        ax.set_title("Heterozygosity-manipulation sweep")
        ax.grid(True)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(out_dir / "fig5_heterozygosity_manipulation_sweep.png", dpi=220)
        plt.close(fig)

    m_ld, l_ld, u_ld = mean_ci(d_ld)
    tests = pd.DataFrame([
        {
            "test_id": "H2",
            "contrast": "ld_destroyed_alltags_minus_baseline_alltags_ne_0",
            "estimate_mean": m_ld,
            "estimate_ci_lo": l_ld,
            "estimate_ci_hi": u_ld,
            "t_stat": float(t_ld.statistic),
            "p_value": float(t_ld.pvalue),
            "p_value_type": "two-sided",
        },
    ])
    tests.to_csv(out_dir / "bottleneck_hypothesis_tests.csv", index=False)

    diagnostics = pd.DataFrame([
        {
            "metric": "corr_added_harm_vs_beta_decorrelation",
            "value": corr,
            "ci_lo": corr_lo,
            "ci_hi": corr_hi,
        },
        {
            "metric": "corr_added_harm_vs_heterozygosity_shift_delta",
            "value": corr_het,
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        },
        {
            "metric": "null_r2_training_holdout_mean",
            "value": float(null0),
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        },
        {
            "metric": "null_r2_target_population_mean",
            "value": float(null1),
            "ci_lo": np.nan,
            "ci_hi": np.nan,
        },
    ])
    diagnostics.to_csv(out_dir / "bottleneck_diagnostics.csv", index=False)


def summarize_strength(df: pd.DataFrame, sweep_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    req_cols = {"scenario", "divergence_gens", "seed", "bottleneck_frac"}
    missing = sorted(req_cols - set(df.columns))
    if missing:
        raise SystemExit(f"Strength summary missing required columns: {missing}")

    z = df[(df["scenario"] == "bottleneck") & (df["divergence_gens"] == 300)].copy()
    if len(z) == 0:
        raise SystemExit("No bottleneck-strength rows at divergence_gens=300")

    z = z.sort_values(["bottleneck_frac", "seed"]).reset_index(drop=True)
    z["bottleneck_pct"] = z["bottleneck_frac"] * 100.0
    z.to_csv(out_dir / "bottleneck_strength_portability_results.csv", index=False)

    summary_rows: List[Dict[str, float]] = []
    metric_defs = [
        ("ratio_alltags", "Transfer ratio (target R^2 / holdout R^2)"),
        ("r2_alltags_target_population", "Target R^2"),
        ("r2_alltags_training_holdout", "Training holdout R^2"),
        ("beta_corr_alltags_training_vs_target", "Marker-effect correlation"),
        ("delta_heterozygosity_alltags_target_minus_training", "Heterozygosity shift (target - training)"),
        ("delta_lddestroy_minus_baseline", "Destroyed-linkage delta"),
    ]
    for frac in sorted(z["bottleneck_frac"].dropna().unique()):
        grp = z[z["bottleneck_frac"] == frac]
        for metric, label in metric_defs:
            vals = grp[metric].replace([np.inf, -np.inf], np.nan).dropna().values
            m, lo, hi = mean_ci(vals)
            summary_rows.append(
                {
                    "bottleneck_frac": float(frac),
                    "bottleneck_pct": float(frac) * 100.0,
                    "metric": metric,
                    "metric_label": label,
                    "n": int(len(vals)),
                    "mean": m,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["metric", "bottleneck_frac"]).reset_index(drop=True)
    summary.to_csv(out_dir / "bottleneck_strength_portability_summary.csv", index=False)

    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "axes.facecolor": "#fafafa",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#d0d0d0",
            "grid.alpha": 0.5,
            "axes.titleweight": "bold",
        }
    )

    x_vals = sorted(summary["bottleneck_frac"].dropna().unique())
    x_labels = [f"{int(round(x * 100.0))}%" for x in x_vals]

    def _plot_single(metric: str, y_label: str, out_name: str, ref_value: float | None = None) -> None:
        s = summary[summary["metric"] == metric].copy()
        s = s.set_index("bottleneck_frac").reindex(x_vals).reset_index()
        y = s["mean"].astype(float).values
        lo = s["ci_lo"].astype(float).values
        hi = s["ci_hi"].astype(float).values
        err_lo = y - lo
        err_hi = hi - y
        x = np.arange(len(x_vals))
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), constrained_layout=True)
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([err_lo, err_hi]),
            fmt="o-",
            color="#1f77b4",
            ecolor="#1f77b4",
            elinewidth=1.4,
            capsize=3.5,
            markersize=5.5,
            linewidth=2.0,
        )
        if ref_value is not None:
            ax.axhline(float(ref_value), color="gray", linestyle=":", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Bottleneck size as fraction of baseline population size")
        ax.set_ylabel(y_label)
        ax.grid(True)
        fig.savefig(out_dir / out_name, dpi=220)
        plt.close(fig)

    _plot_single(
        "ratio_alltags",
        "Transfer ratio\n(target R^2 / holdout R^2)",
        "fig_strength_transfer_ratio.png",
        ref_value=1.0,
    )
    _plot_single(
        "beta_corr_alltags_training_vs_target",
        "Marker-effect correlation",
        "fig_strength_beta_correlation.png",
        ref_value=1.0,
    )
    _plot_single(
        "delta_heterozygosity_alltags_target_minus_training",
        "Heterozygosity shift\n(target - training)",
        "fig_strength_heterozygosity_shift.png",
        ref_value=0.0,
    )
    _plot_single(
        "delta_lddestroy_minus_baseline",
        "Destroyed-linkage ratio delta\n(destroyed-linkage - baseline)",
        "fig_strength_ld_destroy_delta.png",
        ref_value=0.0,
    )

    rr = summary[summary["metric"].isin(["r2_alltags_training_holdout", "r2_alltags_target_population"])].copy()
    rr = rr.sort_values(["metric", "bottleneck_frac"])
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), constrained_layout=True)
    for metric, color, label in [
        ("r2_alltags_training_holdout", "#1f77b4", "Holdout R^2"),
        ("r2_alltags_target_population", "#d62728", "Target R^2"),
    ]:
        s = rr[rr["metric"] == metric].set_index("bottleneck_frac").reindex(x_vals).reset_index()
        y = s["mean"].astype(float).values
        lo = s["ci_lo"].astype(float).values
        hi = s["ci_hi"].astype(float).values
        x = np.arange(len(x_vals))
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([y - lo, hi - y]),
            fmt="o-",
            color=color,
            ecolor=color,
            elinewidth=1.3,
            capsize=3.2,
            markersize=5.0,
            linewidth=1.9,
            label=label,
        )
    ax.set_xticks(np.arange(len(x_vals)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Bottleneck size as fraction of baseline population size")
    ax.set_ylabel("R^2")
    ax.grid(True)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out_dir / "fig_strength_target_vs_holdout_r2.png", dpi=220)
    plt.close(fig)

    if len(sweep_df) > 0:
        sweep = sweep_df[
            (sweep_df["scenario"] == "bottleneck")
            & (sweep_df["divergence_gens"] == 300)
            & sweep_df["bottleneck_frac"].notna()
        ].copy()
        if len(sweep) > 0:
            sweep.to_csv(out_dir / "bottleneck_strength_sweep_results.csv", index=False)
            grp = (
                sweep.groupby(["bottleneck_frac", "resample_strength"], as_index=False)
                .agg(
                    mean_transfer_ratio=("transfer_ratio", "mean"),
                    sd_transfer_ratio=("transfer_ratio", "std"),
                    n=("transfer_ratio", "count"),
                )
                .sort_values(["bottleneck_frac", "resample_strength"])
            )
            fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.0), constrained_layout=True)
            cmap = plt.get_cmap("viridis", len(x_vals))
            for idx, frac in enumerate(x_vals):
                g = grp[grp["bottleneck_frac"] == frac]
                if len(g) == 0:
                    continue
                sem = g["sd_transfer_ratio"].fillna(0.0) / np.sqrt(np.maximum(g["n"], 1))
                lo = g["mean_transfer_ratio"] - 1.96 * sem
                hi = g["mean_transfer_ratio"] + 1.96 * sem
                label = f"{int(round(frac * 100.0))}%"
                ax.plot(
                    g["resample_strength"],
                    g["mean_transfer_ratio"],
                    marker="o",
                    linewidth=1.8,
                    markersize=4.5,
                    color=cmap(idx),
                    label=label,
                )
                ax.fill_between(g["resample_strength"], lo, hi, color=cmap(idx), alpha=0.12)
            ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.1)
            ax.set_xlabel("Heterozygosity resampling strength")
            ax.set_ylabel("Transfer ratio\n(target R^2 / holdout R^2)")
            ax.grid(True)
            ax.legend(title="Bottleneck size", frameon=False, fontsize=8)
            fig.savefig(out_dir / "fig_strength_heterozygosity_sweep.png", dpi=220)
            plt.close(fig)
            return

    pd.DataFrame(
        columns=[
            "scenario",
            "divergence_gens",
            "seed",
            "bottleneck_frac",
            "intervention",
            "resample_strength",
            "transfer_ratio",
        ]
    ).to_csv(out_dir / "bottleneck_strength_sweep_results.csv", index=False)
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.0), constrained_layout=True)
    ax.axis("off")
    ax.text(0.5, 0.5, "No sweep data available", ha="center", va="center")
    fig.savefig(out_dir / "fig_strength_heterozygosity_sweep.png", dpi=220)
    plt.close(fig)


def merge_and_summarize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pattern = args.glob
    csvs = sorted(Path(".").glob(pattern))
    if not csvs:
        raise SystemExit(f"No files matched: {pattern}")

    dfs = [pd.read_csv(p) for p in csvs]
    df = pd.concat(dfs, ignore_index=True)
    sweep_csvs = sorted(Path(".").glob(args.sweep_glob))
    sweep_df = pd.concat([pd.read_csv(p) for p in sweep_csvs], ignore_index=True) if sweep_csvs else pd.DataFrame()
    summarize(df, sweep_df, out_dir)


def merge_and_summarize_strength(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    pattern = args.glob
    csvs = sorted(Path(".").glob(pattern))
    if not csvs:
        raise SystemExit(f"No files matched: {pattern}")

    dfs = [pd.read_csv(p) for p in csvs]
    df = pd.concat(dfs, ignore_index=True)
    sweep_csvs = sorted(Path(".").glob(args.sweep_glob))
    sweep_df = pd.concat([pd.read_csv(p) for p in sweep_csvs], ignore_index=True) if sweep_csvs else pd.DataFrame()
    summarize_strength(df, sweep_df, out_dir)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Bottleneck portability mechanism analysis")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_run = sub.add_parser("run-chunk")
    ap_run.add_argument("--scenario", type=str, default="both", choices=["divergence", "bottleneck", "both"])
    ap_run.add_argument("--gens", type=str, default="0,20,50,100,200,500,1000,2000,5000")
    ap_run.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10")
    ap_run.add_argument("--chunk-name", type=str, default="")
    ap_run.add_argument("--out-dir", type=str, default="sims/results_bottleneck_ld_mechanism/chunks")
    ap_run.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 1))))
    ap_run.add_argument("--n-per-pop", type=int, default=2000)
    ap_run.add_argument("--seq-len", type=int, default=5_000_000)
    ap_run.add_argument("--n-causal", type=int, default=400)
    ap_run.add_argument("--n-array-sites", type=int, default=2000)
    ap_run.add_argument("--bottleneck-frac", type=float, default=0.10)
    ap_run.add_argument(
        "--min-resample-ess-frac",
        type=float,
        default=0.6,
        help="Skip sweep strengths when ESS < max(min_resample_ess_count, min_resample_ess_frac * n).",
    )
    ap_run.add_argument(
        "--min-resample-ess-count",
        type=int,
        default=300,
        help="Absolute ESS floor used by the heterozygosity resampling stability gate.",
    )

    ap_merge = sub.add_parser("summarize")
    ap_merge.add_argument("--glob", type=str, default="sims/results_bottleneck_ld_mechanism/chunks/*.csv")
    ap_merge.add_argument("--sweep-glob", type=str, default="sims/results_bottleneck_ld_mechanism/hetero_sweeps/*_hetero_sweep.csv")
    ap_merge.add_argument("--out-dir", type=str, default="sims/results_bottleneck_ld_mechanism")

    ap_run_strength = sub.add_parser("run-strength-chunk")
    ap_run_strength.add_argument("--gens", type=int, default=300)
    ap_run_strength.add_argument("--bottleneck-fracs", type=str, default="0.1,0.2,0.5,0.7,1.0")
    ap_run_strength.add_argument("--seeds", type=str, default="1,2,3,4,5,6,7,8,9,10")
    ap_run_strength.add_argument("--chunk-name", type=str, default="")
    ap_run_strength.add_argument("--out-dir", type=str, default="sims/results_bottleneck_strength/chunks")
    ap_run_strength.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 1))))
    ap_run_strength.add_argument("--n-per-pop", type=int, default=2000)
    ap_run_strength.add_argument("--seq-len", type=int, default=5_000_000)
    ap_run_strength.add_argument("--n-causal", type=int, default=400)
    ap_run_strength.add_argument("--n-array-sites", type=int, default=2000)
    ap_run_strength.add_argument("--min-resample-ess-frac", type=float, default=0.6)
    ap_run_strength.add_argument("--min-resample-ess-count", type=int, default=300)

    ap_merge_strength = sub.add_parser("summarize-strength")
    ap_merge_strength.add_argument("--glob", type=str, default="sims/results_bottleneck_strength/chunks/*.csv")
    ap_merge_strength.add_argument(
        "--sweep-glob",
        type=str,
        default="sims/results_bottleneck_strength/strength_sweeps/*_strength_sweep.csv",
    )
    ap_merge_strength.add_argument("--out-dir", type=str, default="sims/results_bottleneck_strength")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "run-chunk":
        path, sweep_path = run_chunk(args)
        print(f"Wrote chunk: {path}")
        print(f"Wrote heterozygosity sweep chunk: {sweep_path}")
        return

    if args.cmd == "summarize":
        merge_and_summarize(args)
        print(f"Summary complete in {args.out_dir}")
        return

    if args.cmd == "run-strength-chunk":
        path, sweep_path = run_strength_chunk(args)
        print(f"Wrote strength chunk: {path}")
        print(f"Wrote strength sweep chunk: {sweep_path}")
        return

    if args.cmd == "summarize-strength":
        merge_and_summarize_strength(args)
        print(f"Strength summary complete in {args.out_dir}")
        return

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
