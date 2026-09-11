"""Explicit conditional score models and group-aware cross-fitting.

Only baseline covariates and the raw PGS enter these fits. Outcomes, exit times,
and held-out outer-test participants never enter a training transform.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def baseline_columns(config):
    return ["age0", "sex", *[f"PC{i + 1}" for i in range(config["num_pcs"])]]


def stage1_rhs(config):
    pcs = ", ".join(baseline_columns(config)[2:])
    return (f"s(age0, k={config['stage1_age_k']}) + sex + "
            f"duchon({pcs}, centers={config['stage1_centers']}, scale_dims=true)")


def grouped_folds(groups, count, seed):
    """Assign whole groups; stable under row permutation, with no outcome input."""
    groups = pd.Series(groups, dtype="string")
    if groups.isna().any() or (groups.str.len() == 0).any():
        raise ValueError("split groups must be present")
    unique = sorted(groups.unique(), key=lambda g: hashlib.sha256(
        f"{seed}:inner:{g}".encode()).digest())
    if len(unique) < count or count < 2:
        raise ValueError("too few independent groups for cross-fitting")
    mapping = {group: i % count for i, group in enumerate(unique)}
    return groups.map(mapping).to_numpy(int)


def transformed_score(model, kind, data):
    if kind == "ctn":
        z = np.asarray(model.transformation_score(data), dtype=float)
    elif kind == "location_scale":
        prediction = model.predict(data, return_type="dict")
        mu = np.asarray(prediction["mean_plugin"], dtype=float)
        sigma = np.asarray(prediction["noise_scale"], dtype=float)
        if not np.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError("conditional score scale is not finite and positive")
        z = (data.PGS.to_numpy(float) - mu) / sigma
    else:
        raise ValueError("unknown conditional score model")
    if z.shape != (len(data),) or not np.isfinite(z).all():
        raise ValueError("score transformation produced invalid latent scores")
    return z


def fit_transform(frame_path, config_path, kind, fold, output):
    """One bounded-worker unit: a fold-complement fit or deployment fit."""
    import gamfit
    config = json.loads(Path(config_path).read_text())
    frame = pd.read_parquet(frame_path)
    training = frame.is_train.to_numpy(bool)
    if fold >= 0:
        fit_rows = training & (frame.inner_fold.to_numpy(int) != fold)
        score_rows = training & (frame.inner_fold.to_numpy(int) == fold)
    else:
        fit_rows, score_rows = training, ~training
    columns = ["PGS", *baseline_columns(config)]
    train, held = frame.loc[fit_rows, columns], frame.loc[score_rows, columns]
    if train.empty or held.empty:
        raise ValueError("empty score-model training or held-out partition")
    rhs = stage1_rhs(config)
    options = {"transformation_normal": True, "config": {"transformation_normal_config": {
        "response_num_internal_knots": config["stage1_response_knots"]}}} if kind == "ctn" else {
        "family": "gaussian", "noise_formula": rhs}
    model = gamfit.fit(train, f"PGS ~ {rhs}", persistent_warm_start_root=output / "warm", **options)
    z = transformed_score(model, kind, held)
    model.save(output / "transform.gamfit")
    restored = gamfit.load(output / "transform.gamfit")
    replayed = transformed_score(restored, kind, held)
    if not np.allclose(z, replayed, rtol=1e-8, atol=1e-10):
        raise ValueError("score-transform save/load replay disagrees")
    np.savez(output / "scores.npz", rows=np.flatnonzero(score_rows), z=z)
    # Deployment-versus-OOF shift is diagnostic only; never replace OOF values.
    if fold < 0:
        np.save(output / "training_replay.npy", transformed_score(restored, kind, frame.loc[training, columns]))


def assemble_scores(frame, artifacts):
    z = np.full(len(frame), np.nan)
    seen = np.zeros(len(frame), dtype=bool)
    for path in artifacts:
        with np.load(path) as saved:
            rows, values = saved["rows"], saved["z"]
        if rows.ndim != 1 or values.shape != rows.shape or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("malformed cross-fit score artifact")
        if (rows < 0).any() or (rows >= len(frame)).any() or len(np.unique(rows)) != len(rows) or seen[rows].any():
            raise ValueError("overlapping or invalid cross-fit score rows")
        z[rows], seen[rows] = values, True
    if not seen.all() or not np.isfinite(z).all():
        raise ValueError("cross-fitting must score every row exactly once")
    return z


def score_diagnostics(z, groups, min_count):
    """Held-out distribution summaries, not a conditional-normality certificate."""
    results = []
    for label, mask in groups:
        x = np.asarray(z)[mask]
        if len(x) < min_count:
            results.append({"group": label, "status": "insufficient_support"})
            continue
        centered = x - x.mean()
        sd = float(x.std())
        results.append({"group": label, "status": "ok", "n": len(x),
                        "mean": float(x.mean()), "sd": sd,
                        "skew": float(np.mean((centered / sd)**3)) if sd else None,
                        "excess_kurtosis": float(np.mean((centered / sd)**4) - 3) if sd else None,
                        "central_95_fraction": float(np.mean(np.abs(x) <= 1.959963984540054))})
    return results
