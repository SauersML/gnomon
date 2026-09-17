"""Report the declared score law's adequacy; apply a frozen external CTN when declared."""
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


def declared_law_diagnostics(train_score, test_score, groups, min_count):
    """Adequacy of the declared law within each held-out context stratum.

    The outcome model anchors on one law: the empirical law of the training
    rows' scores, standardized as gam stores it. A stratum whose held-out scores
    depart from that pooled law is where the anchor is approximate; these
    summaries measure the departure, they do not certify its absence.
    """
    train_score = np.asarray(train_score, dtype=float)
    mean, sd = train_score.mean(), train_score.std()
    pooled = np.sort((train_score - mean) / sd)
    standardized = (np.asarray(test_score, dtype=float) - mean) / sd
    strata = []
    for summary, (_, mask) in zip(score_diagnostics(standardized, groups, min_count), groups):
        if summary["status"] == "ok":
            held = np.sort(standardized[mask])
            points = np.concatenate([held, pooled])
            summary["ks_distance_to_pooled_training_law"] = float(np.max(np.abs(
                np.searchsorted(held, points, side="right") / len(held)
                - np.searchsorted(pooled, points, side="right") / len(pooled))))
        strata.append(summary)
    pooled_summary, = score_diagnostics(pooled, [("training", np.ones(len(pooled), dtype=bool))], 1)
    return {"pooled_training_law": pooled_summary, "strata": strata}
