"""Apply frozen external CTNs and report score-distribution diagnostics."""
from __future__ import annotations

import numpy as np


def baseline_columns(config):
    return ["age0", "sex", *[f"PC{i + 1}" for i in range(config["num_pcs"])]]


def transformed_score(model, kind, data, num_pcs):
    if kind != "ctn":
        raise ValueError("the workflow requires a frozen external CTN")
    # Cohort frames also contain dates, outcomes and identifiers. The external
    # CTN only consumes its observed score and frozen PC coordinates.
    predictors = data[["PGS", *[f"PC{i + 1}" for i in range(num_pcs)]]]
    z = np.asarray(model.transformation_score(predictors), dtype=float)
    if z.shape != (len(data),) or not np.isfinite(z).all():
        raise ValueError("score transformation produced invalid latent scores")
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
