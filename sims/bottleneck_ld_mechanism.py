#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ttest_1samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sim_pops import _diploid_index_pairs, _solve_intercept_for_prevalence
from sim_two_pop import _build_demography_bottleneck, _build_demography_divergence


NEAR_MAX_BP = 5_000
FAR_MIN_BP = 50_000
FAR_MAX_BP = 250_000


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
    prevalence: float


@dataclass
class Sim:
    X: np.ndarray
    pos: np.ndarray
    maf: np.ndarray
    pop_idx: np.ndarray
    g_true: np.ndarray
    y: np.ndarray
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
            dem = _build_demography_bottleneck(cfg.gens, cfg.ne, max(100, cfg.ne // 10))
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
    pos = np.empty(m, dtype=np.float64)
    maf = np.empty(m, dtype=np.float64)
    for j, var in enumerate(ts.variants()):
        gh = var.genotypes.astype(np.int16)
        gd = gh[0::2] + gh[1::2]
        X[:, j] = gd.astype(np.int8)
        pos[j] = ts.site(var.site.id).position
        p = gd.mean() / 2.0
        maf[j] = min(p, 1.0 - p)

    rng = np.random.default_rng(cfg.seed + 7)
    eligible = np.where(maf >= 0.01)[0]
    cidx = np.sort(rng.choice(eligible, size=min(cfg.n_causal, len(eligible)), replace=False))
    betas = rng.normal(0.0, math.sqrt(cfg.h2 / len(cidx)), size=len(cidx))

    g_raw = X[:, cidx].astype(float) @ betas
    g_true = (g_raw - g_raw.mean()) / (g_raw.std() if g_raw.std() > 1e-12 else 1.0)
    b0 = _solve_intercept_for_prevalence(cfg.prevalence, g_true)
    y = np.random.default_rng(cfg.seed + 31).binomial(1, expit(b0 + g_true)).astype(np.int8)

    return Sim(X=X, pos=pos, maf=maf, pop_idx=pop_idx.astype(np.int32), g_true=g_true, y=y, causal_idx=cidx)


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


def match_near_far_tags(sim: Sim, seed: int, min_tags: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 200)

    all_idx = np.arange(sim.X.shape[1])
    noncausal = np.setdiff1d(all_idx, sim.causal_idx, assume_unique=False)

    near = []
    far = []

    nc_pos = sim.pos[noncausal]

    for c in sim.causal_idx:
        cp = sim.pos[c]
        d = np.abs(nc_pos - cp)

        near_pool = noncausal[d <= NEAR_MAX_BP]
        far_pool = noncausal[(d >= FAR_MIN_BP) & (d <= FAR_MAX_BP)]
        if len(near_pool) == 0 or len(far_pool) == 0:
            continue

        # MAF match near/far per-causal to reduce heterozygosity confounding
        cm = sim.maf[c]
        near_choice = near_pool[np.argmin(np.abs(sim.maf[near_pool] - cm))]
        far_choice = far_pool[np.argmin(np.abs(sim.maf[far_pool] - cm))]

        near.append(int(near_choice))
        far.append(int(far_choice))

    near = np.array(sorted(set(near)), dtype=int)
    far = np.array(sorted(set(far)), dtype=int)
    n = min(len(near), len(far))
    if n < min_tags:
        raise RuntimeError(f"Insufficient matched tags: near={len(near)} far={len(far)}")

    near_sel = np.sort(rng.choice(near, size=n, replace=False))
    far_sel = np.sort(rng.choice(far, size=n, replace=False))
    return near_sel, far_sel


def ld_destroyed_matrix(X: np.ndarray, score_idx: np.ndarray, tags: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 300)
    Xs = X[np.ix_(score_idx, tags)].copy().astype(float)
    for j in range(Xs.shape[1]):
        rng.shuffle(Xs[:, j])
    return Xs


def _mean_heterozygosity_from_matrix(X: np.ndarray) -> float:
    p = X.mean(axis=0) / 2.0
    return float(np.mean(2.0 * p * (1.0 - p)))


def _heterozygosity_biased_resample(
    X: np.ndarray, strength: float, increase: bool, rng: np.random.Generator
) -> np.ndarray:
    X = np.asarray(X, dtype=np.int8)
    n = X.shape[0]
    if n == 0:
        return X.copy()
    if strength <= 0.0:
        idx = rng.choice(n, size=n, replace=True)
        return X[idx].copy()

    h = np.mean(X == 1, axis=1).astype(float)
    sd = float(np.std(h))
    if sd <= 1e-12:
        idx = rng.choice(n, size=n, replace=True)
        return X[idx].copy()

    z = (h - float(np.mean(h))) / sd
    logits = strength * z if increase else -strength * z
    logits = logits - float(np.max(logits))
    w = np.exp(logits)
    probs = w / float(np.sum(w))
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


def one_run(cfg: Config, min_tags: int) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    sim = simulate(cfg)

    pop0 = np.where(sim.pop_idx == 0)[0]
    pop1 = np.where(sim.pop_idx == 1)[0]
    train0, test0 = split_pop0(pop0, cfg.seed)

    near_tags, far_tags = match_near_far_tags(sim, cfg.seed, min_tags=min_tags)
    all_tags = np.sort(np.unique(np.concatenate([near_tags, far_tags]))).astype(int)

    prs_near_0, b_near = build_marginal_prs(sim.X, sim.g_true, train0, test0, near_tags, use_score_stats=False)
    prs_near_1, _ = build_marginal_prs(sim.X, sim.g_true, train0, pop1, near_tags, use_score_stats=False)

    prs_far_0, b_far = build_marginal_prs(sim.X, sim.g_true, train0, test0, far_tags, use_score_stats=False)
    prs_far_1, _ = build_marginal_prs(sim.X, sim.g_true, train0, pop1, far_tags, use_score_stats=False)
    prs_all_0, b_all = build_marginal_prs(sim.X, sim.g_true, train0, test0, all_tags, use_score_stats=False)
    prs_all_1, _ = build_marginal_prs(sim.X, sim.g_true, train0, pop1, all_tags, use_score_stats=False)

    prs_far_1_het, _ = build_marginal_prs(sim.X, sim.g_true, train0, pop1, far_tags, use_score_stats=True)

    # LD destruction: preserve marginal distribution but scramble LD at target.
    Xt = sim.X[np.ix_(train0, far_tags)].astype(float)
    mu_t = Xt.mean(axis=0)
    sd_t = Xt.std(axis=0)
    sd_t[sd_t == 0] = 1.0
    Xld = ld_destroyed_matrix(sim.X, pop1, far_tags, cfg.seed)
    prs_far_1_ld = ((Xld - mu_t) / sd_t) @ b_far

    rng = np.random.default_rng(cfg.seed + 500)
    g_perm = sim.g_true.copy()
    g_perm[train0] = rng.permutation(g_perm[train0])
    prs_null_0, _ = build_marginal_prs(sim.X, g_perm, train0, test0, far_tags, use_score_stats=False)
    prs_null_1, _ = build_marginal_prs(sim.X, g_perm, train0, pop1, far_tags, use_score_stats=False)

    # Compare marginal betas trained in pop0 vs pop1 (same panel)
    pop1_train = pop1[:len(train0)] if len(pop1) >= len(train0) else pop1
    _, b_far_pop1 = build_marginal_prs(sim.X, sim.g_true, pop1_train, pop1_train, far_tags, use_score_stats=False)
    _, b_all_pop1 = build_marginal_prs(sim.X, sim.g_true, pop1_train, pop1_train, all_tags, use_score_stats=False)
    beta_corr_far = (
        float(np.corrcoef(b_far, b_far_pop1)[0, 1])
        if np.std(b_far) > 1e-12 and np.std(b_far_pop1) > 1e-12
        else np.nan
    )
    beta_corr_all = (
        float(np.corrcoef(b_all, b_all_pop1)[0, 1])
        if np.std(b_all) > 1e-12 and np.std(b_all_pop1) > 1e-12
        else np.nan
    )

    p0_far = sim.X[np.ix_(train0, far_tags)].mean(axis=0) / 2.0
    p1_far = sim.X[np.ix_(pop1, far_tags)].mean(axis=0) / 2.0
    h0_far = float(np.mean(2 * p0_far * (1 - p0_far)))
    h1_far = float(np.mean(2 * p1_far * (1 - p1_far)))
    p0_all = sim.X[np.ix_(train0, all_tags)].mean(axis=0) / 2.0
    p1_all = sim.X[np.ix_(pop1, all_tags)].mean(axis=0) / 2.0
    h0_all = float(np.mean(2 * p0_all * (1 - p0_all)))
    h1_all = float(np.mean(2 * p1_all * (1 - p1_all)))

    r2_near_0 = safe_r2(prs_near_0, sim.g_true[test0])
    r2_near_1 = safe_r2(prs_near_1, sim.g_true[pop1])
    r2_far_0 = safe_r2(prs_far_0, sim.g_true[test0])
    r2_far_1 = safe_r2(prs_far_1, sim.g_true[pop1])
    r2_all_0 = safe_r2(prs_all_0, sim.g_true[test0])
    r2_all_1 = safe_r2(prs_all_1, sim.g_true[pop1])
    r2_far_1_het = safe_r2(prs_far_1_het, sim.g_true[pop1])
    r2_far_1_ld = safe_r2(prs_far_1_ld, sim.g_true[pop1])

    ratio_near = r2_near_1 / r2_near_0 if r2_near_0 > 1e-8 else np.nan
    ratio_far = r2_far_1 / r2_far_0 if r2_far_0 > 1e-8 else np.nan
    ratio_all = r2_all_1 / r2_all_0 if r2_all_0 > 1e-8 else np.nan
    ratio_far_het = r2_far_1_het / r2_far_0 if r2_far_0 > 1e-8 else np.nan
    ratio_far_ld = r2_far_1_ld / r2_far_0 if r2_far_0 > 1e-8 else np.nan

    base_row = {
        "scenario": cfg.scenario,
        "divergence_gens": cfg.gens,
        "seed": cfg.seed,
        "n_near_tags": len(near_tags),
        "n_far_tags": len(far_tags),
        "r2_near_training_holdout": r2_near_0,
        "r2_near_target_population": r2_near_1,
        "ratio_near": ratio_near,
        "r2_distant_training_holdout": r2_far_0,
        "r2_distant_target_population": r2_far_1,
        "ratio_far": ratio_far,
        "r2_alltags_training_holdout": r2_all_0,
        "r2_alltags_target_population": r2_all_1,
        "ratio_alltags": ratio_all,
        "r2_distant_target_standardized_by_target": r2_far_1_het,
        "ratio_far_het": ratio_far_het,
        "r2_distant_target_with_destroyed_linkage": r2_far_1_ld,
        "ratio_far_ld_destroy": ratio_far_ld,
        "delta_near_minus_far": ratio_near - ratio_far if np.isfinite(ratio_near) and np.isfinite(ratio_far) else np.nan,
        "delta_het_minus_baseline": ratio_far_het - ratio_far if np.isfinite(ratio_far_het) and np.isfinite(ratio_far) else np.nan,
        "delta_lddestroy_minus_baseline": ratio_far_ld - ratio_far if np.isfinite(ratio_far_ld) and np.isfinite(ratio_far) else np.nan,
        "r2_null_training_holdout": safe_r2(prs_null_0, sim.g_true[test0]),
        "r2_null_target_population": safe_r2(prs_null_1, sim.g_true[pop1]),
        "beta_corr_distant_training_vs_target": beta_corr_far,
        "beta_corr_alltags_training_vs_target": beta_corr_all,
        "delta_heterozygosity_distant_target_minus_training": h1_far - h0_far,
        "delta_heterozygosity_alltags_target_minus_training": h1_all - h0_all,
        "mean_abs_maf_diff_far": float(np.mean(np.abs(p1_far - p0_far))),
    }

    sweep_rows: List[Dict[str, float]] = []
    strengths = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]

    X_train_all = sim.X[np.ix_(train0, all_tags)].astype(float)
    X_test_all = sim.X[np.ix_(test0, all_tags)].astype(float)
    X_target_all = sim.X[np.ix_(pop1, all_tags)].astype(float)
    X_train_all_i = sim.X[np.ix_(train0, all_tags)].astype(np.int8)
    X_test_all_i = sim.X[np.ix_(test0, all_tags)].astype(np.int8)
    X_target_all_i = sim.X[np.ix_(pop1, all_tags)].astype(np.int8)
    rng_sweep = np.random.default_rng(cfg.seed + 900)

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
            X_target_new_i = _heterozygosity_biased_resample(X_target_all_i, strength, increase=False, rng=rng_sweep)
            X_target_new = X_target_new_i.astype(float)
            prs_target = ((X_target_new - mu_t) / sd_t) @ b
            r2_target = safe_r2(prs_target, sim.g_true[pop1])
            ratio = r2_target / r2_holdout if r2_holdout > 1e-8 else np.nan
            sweep_rows.append(
                {
                    "scenario": cfg.scenario,
                    "divergence_gens": cfg.gens,
                    "seed": cfg.seed,
                    "intervention": "decrease_target_heterozygosity",
                    "resample_strength": strength,
                    "mean_heterozygosity_training": _mean_heterozygosity_from_matrix(X_train_all_i),
                    "mean_heterozygosity_target": _mean_heterozygosity_from_matrix(X_target_new_i),
                    "heterozygosity_shift_target_minus_training": _mean_heterozygosity_from_matrix(X_target_new_i) - _mean_heterozygosity_from_matrix(X_train_all_i),
                    "r2_training_holdout": r2_holdout,
                    "r2_target_population": r2_target,
                    "transfer_ratio": ratio,
                    "ld_absolute_change": _ld_corr_change(X_target_all_i.astype(float), X_target_new),
                }
            )

    # Bottleneck scenario: raise training heterozygosity across a broad range.
    if cfg.scenario == "bottleneck":
        for strength in strengths:
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

            sweep_rows.append(
                {
                    "scenario": cfg.scenario,
                    "divergence_gens": cfg.gens,
                    "seed": cfg.seed,
                    "intervention": "increase_training_heterozygosity",
                    "resample_strength": strength,
                    "mean_heterozygosity_training": _mean_heterozygosity_from_matrix(X_train_new_i),
                    "mean_heterozygosity_target": _mean_heterozygosity_from_matrix(X_target_all_i),
                    "heterozygosity_shift_target_minus_training": _mean_heterozygosity_from_matrix(X_target_all_i) - _mean_heterozygosity_from_matrix(X_train_new_i),
                    "r2_training_holdout": r2_holdout,
                    "r2_target_population": r2_target,
                    "transfer_ratio": ratio,
                    "ld_absolute_change": _ld_corr_change(X_train_all_i.astype(float), X_train_new),
                }
            )

    return base_row, sweep_rows


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
                        prevalence=0.1,
                    )
                )

    rows: List[Dict[str, float]] = []
    sweep_rows: List[Dict[str, float]] = []
    with cf.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(one_run, cfg, args.min_tags) for cfg in cfgs]
        for fut in cf.as_completed(futs):
            base_row, sweep_part = fut.result()
            rows.append(base_row)
            sweep_rows.extend(sweep_part)

    df = pd.DataFrame(rows).sort_values(["scenario", "divergence_gens", "seed"]).reset_index(drop=True)

    chunk_name = args.chunk_name or f"chunk_{args.scenario}_g{'-'.join(map(str, gens))}_s{'-'.join(map(str, seeds))}"
    path = out_dir / f"{chunk_name}.csv"
    df.to_csv(path, index=False)
    sweep_path = sweep_dir / f"{chunk_name}_hetero_sweep.csv"
    pd.DataFrame(sweep_rows).sort_values(["scenario", "divergence_gens", "seed", "resample_strength"]).to_csv(sweep_path, index=False)
    return path, sweep_path


def summarize(df: pd.DataFrame, sweep_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "bottleneck_ld_mechanism_results.csv", index=False)

    summary = df.groupby(["scenario", "divergence_gens"]).agg(
        n=("seed", "count"),
        ratio_near_mean=("ratio_near", "mean"),
        ratio_far_mean=("ratio_far", "mean"),
        ratio_far_het_mean=("ratio_far_het", "mean"),
        ratio_far_ld_destroy_mean=("ratio_far_ld_destroy", "mean"),
        beta_corr_far_mean=("beta_corr_distant_training_vs_target", "mean"),
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

    # Tests for requested claims
    d_nf = df["delta_near_minus_far"].dropna().values
    d_het = df["delta_het_minus_baseline"].dropna().values
    d_ld = df["delta_lddestroy_minus_baseline"].dropna().values

    t_nf = ttest_1samp(d_nf, 0.0, nan_policy="omit")
    t_het = ttest_1samp(d_het, 0.0, nan_policy="omit")
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
    for sc, color in [("divergence", "#1f77b4"), ("bottleneck", "#d62728")]:
        z = df[df["scenario"] == sc]
        gens_sorted = sorted(z["divergence_gens"].unique())
        if gens_sorted:
            max_gen = max(max_gen, int(max(gens_sorted)))
        for metric, style, tag_label in [
            ("ratio_near", "o-", "near tags (within 5,000 base pairs)"),
            ("ratio_far", "s--", "distant tags (50,000 to 250,000 base pairs)"),
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
            label = f"{sc}, {tag_label}"
            ax.plot(xs, ms, style, color=color, linewidth=2.2, markersize=6, label=label)
            ax.fill_between(xs, lo, hi, color=color, alpha=0.12 if metric == "ratio_near" else 0.08)
    ax.axhline(1.0, color="gray", ls=":")
    ax.set_xscale("symlog", linthresh=20)
    if max_gen > 0:
        ticks = sorted(df["divergence_gens"].unique())
        ax.set_xlim(left=0, right=max(1.0, float(max_gen) * 1.1))
        ax.set_xticks(ticks)
    ax.set_title("Nearby vs distant marker-tag transfer")
    ax.set_xlabel("Population split age (generations)")
    ax.set_ylabel("Transfer ratio\n(target population R^2 / same-ancestry holdout R^2)")
    ax.grid(True)
    ax.set_ylim(bottom=0.15)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out_dir / "fig1_near_vs_far.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12.5, 5.0), constrained_layout=True)
    max_gen = 0
    for sc, color in [("divergence", "#1f77b4"), ("bottleneck", "#d62728")]:
        z = df[df["scenario"] == sc]
        gens_sorted = sorted(z["divergence_gens"].unique())
        if gens_sorted:
            max_gen = max(max_gen, int(max(gens_sorted)))
        for metric, style, label in [
            ("ratio_far", "o-", "baseline"),
            ("ratio_far_het", "s--", "target standardized by target distribution"),
            ("ratio_far_ld_destroy", "d:", "target linkage structure destroyed"),
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
    ax.set_title("Intervention comparison for distant marker tags")
    ax.set_xlabel("Population split age (generations)")
    ax.set_ylabel("Transfer ratio for distant marker tags\n(target population R^2 / same-ancestry holdout R^2)")
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
        for (sc, intervention), color, marker in [
            (("divergence", "decrease_target_heterozygosity"), "#1f77b4", "o"),
            (("bottleneck", "increase_training_heterozygosity"), "#d62728", "s"),
        ]:
            z = sweep_df[(sweep_df["scenario"] == sc) & (sweep_df["intervention"] == intervention)]
            if len(z) == 0:
                continue
            grp = (
                z.groupby("resample_strength", as_index=False)
                .agg(
                    mean_heterozygosity=("mean_heterozygosity_target", "mean"),
                    transfer_ratio_mean=("transfer_ratio", "mean"),
                    transfer_ratio_sd=("transfer_ratio", "std"),
                    n=("transfer_ratio", "count"),
                    ld_abs_change_mean=("ld_absolute_change", "mean"),
                )
                .sort_values("mean_heterozygosity")
            )
            if intervention == "increase_training_heterozygosity":
                grp = (
                    z.groupby("resample_strength", as_index=False)
                    .agg(
                        mean_heterozygosity=("mean_heterozygosity_training", "mean"),
                        transfer_ratio_mean=("transfer_ratio", "mean"),
                        transfer_ratio_sd=("transfer_ratio", "std"),
                        n=("transfer_ratio", "count"),
                        ld_abs_change_mean=("ld_absolute_change", "mean"),
                    )
                    .sort_values("mean_heterozygosity")
                )
            else:
                grp = (
                    z.groupby("resample_strength", as_index=False)
                    .agg(
                        mean_heterozygosity=("mean_heterozygosity_target", "mean"),
                        transfer_ratio_mean=("transfer_ratio", "mean"),
                        transfer_ratio_sd=("transfer_ratio", "std"),
                        n=("transfer_ratio", "count"),
                        ld_abs_change_mean=("ld_absolute_change", "mean"),
                    )
                    .sort_values("mean_heterozygosity")
                )

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
        ax.axhline(1.0, color="gray", ls=":")
        ax.set_xlabel("Mean heterozygosity of manipulated population")
        ax.set_ylabel("Transfer ratio\n(target population R^2 / same-ancestry holdout R^2)")
        ax.set_title("Heterozygosity-manipulation sweep with linkage structure preserved")
        ax.grid(True)
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(out_dir / "fig5_heterozygosity_manipulation_sweep.png", dpi=220)
        plt.close(fig)

    m_nf, l_nf, u_nf = mean_ci(d_nf)
    m_het, l_het, u_het = mean_ci(d_het)
    m_ld, l_ld, u_ld = mean_ci(d_ld)
    tests = pd.DataFrame([
        {
            "test_id": "H1",
            "contrast": "near_ratio_minus_far_ratio_ne_0",
            "estimate_mean": m_nf,
            "estimate_ci_lo": l_nf,
            "estimate_ci_hi": u_nf,
            "t_stat": float(t_nf.statistic),
            "p_value": float(t_nf.pvalue),
            "p_value_type": "two-sided",
        },
        {
            "test_id": "H2",
            "contrast": "heterozygosity_normalized_far_minus_baseline_far_ne_0",
            "estimate_mean": m_het,
            "estimate_ci_lo": l_het,
            "estimate_ci_hi": u_het,
            "t_stat": float(t_het.statistic),
            "p_value": float(t_het.pvalue),
            "p_value_type": "two-sided",
        },
        {
            "test_id": "H3",
            "contrast": "ld_destroyed_far_minus_baseline_far_ne_0",
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
    ap_run.add_argument("--min-tags", type=int, default=80)

    ap_merge = sub.add_parser("summarize")
    ap_merge.add_argument("--glob", type=str, default="sims/results_bottleneck_ld_mechanism/chunks/*.csv")
    ap_merge.add_argument("--sweep-glob", type=str, default="sims/results_bottleneck_ld_mechanism/hetero_sweeps/*_hetero_sweep.csv")
    ap_merge.add_argument("--out-dir", type=str, default="sims/results_bottleneck_ld_mechanism")

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

    raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
