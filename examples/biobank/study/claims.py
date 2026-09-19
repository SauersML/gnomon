"""The simulator claim classifier (SPEC section 8): ours against each competitor, cell by cell, on the TRUE
probabilities, as not worse, inconclusive, worse (WORSE = BUG: reproduce on >= 3 seeds, then file) or not
converged.

Input rows are evaluate.evaluate()'s pooled rows over replicate seeds, each carrying `scenario` and `seed`, with
the truth metrics (rmse_true, oe_true, cal_slope_true, auc_true, slope_bias_true, slope_rmse_true). The paired
difference per seed is d = error(ours) - error(competitor), oriented so that positive is worse: the RMSE, the
distance of the expected O/E and calibration slope from 1, the expected AUC's loss (competitor minus ours) and
the slope recovery's absolute bias and RMSE. With R usable seeds, a t interval with R - 1 degrees of freedom on
mean(d) against the margin delta = max(relative x the competitor's error, floor) (study.json claims.margins)
classifies the cell:
- not worse: the upper 95% bound <= delta;
- worse: the lower bound > delta;
- inconclusive: otherwise (more seeds, never a pass);
- not converged: fewer than the planned R seeds pass the two-start convergence test for both fits (R1); such
  seeds never enter d or the dev-seed spreads that size R (R4).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Each classified metric: the row column it reads, how that becomes an error (larger worse) and its margin's
# study.json name.
ERRORS = {
    "rmse_true": ("rmse_true", lambda v: v, "rmse_true"),
    "oe_true": ("oe_true", lambda v: np.abs(v - 1), "oe_true"),
    "cal_slope_true": ("cal_slope_true", lambda v: np.abs(v - 1), "cal_slope_true"),
    "auc_true": ("auc_true", lambda v: -v, "auc_true"),
    "slope_bias": ("slope_bias_true", lambda v: np.abs(v), "slope_recovery"),
    "slope_rmse": ("slope_rmse_true", lambda v: v, "slope_recovery"),
}
CELL = ["scenario", "disease", "model", "stratum", "horizon"]
CLASSES = ("not_worse", "inconclusive", "worse", "not_converged")
# The largest R the sizing rule reports; a cell needing more is unattainable at that margin.
MAX_REPLICATES = 1000


def t_quantile(df, p=0.975):
    from scipy.stats import t
    return float(t.ppf(p, df))


def converged_seeds(convergence, limit):
    """(scenario, seed, disease, model, variant) of every fit whose two starts agree: max|drisk|/SD <= limit.
    A fit with no record never passed the test."""
    ok = convergence.loc[convergence.max_drisk_over_sd <= limit]
    return set(map(tuple, ok[["scenario", "seed", "disease", "model", "variant"]].to_numpy()))


def paired_differences(rows, ours, competitors, convergence=None, limit=0.01):
    """One row per (cell, metric, variant, competitor, seed): d, the competitor's error and whether both fits of
    that seed passed the convergence test (convergence: a frame of scenario, seed, disease, model, variant and
    max_drisk_over_sd; None marks every seed converged)."""
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame.fit == "pooled"].copy()
    # A label per horizon, so binary (None) and survival (years) cells group and sort together.
    frame["horizon"] = frame.horizon.map(lambda h: "none" if h is None or pd.isna(h) else f"{float(h):g}")
    passed = None if convergence is None else converged_seeds(pd.DataFrame(convergence), limit)
    keys = CELL + ["seed"]
    out = []
    for metric, (column, error, _) in ERRORS.items():
        if column not in frame:
            continue
        wide = frame.dropna(subset=[column]).pivot_table(index=keys, columns="variant", values=column,
                                                         aggfunc="first")
        for variant in ours:
            for competitor in competitors:
                if variant not in wide or competitor not in wide:
                    continue
                pair = wide[[variant, competitor]].dropna()
                part = pair.index.to_frame(index=False)
                part["metric"], part["variant"], part["competitor"] = metric, variant, competitor
                part["d"] = (error(pair[variant]) - error(pair[competitor])).to_numpy(float)
                part["competitor_error"] = error(pair[competitor]).to_numpy(float)
                part["converged"] = True if passed is None else [
                    all((s, seed, dis, mod, v) in passed for v in (variant, competitor))
                    for s, seed, dis, mod in zip(part.scenario, part.seed, part.disease, part.model)]
                out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def margin(metric, competitor_error, margins):
    """delta = max(relative x the competitor's mean error, floor), or the absolute margin."""
    spec = margins[ERRORS[metric][2]]
    if spec["scale"] == "relative":
        return max(spec["delta"] * float(competitor_error), spec.get("floor", 0.0))
    return float(spec["delta"])


def classify(differences, margins, planned, minimum=5):
    """Every cell's class from its per-seed paired differences. planned: (scenario, metric) -> R, the replicate
    count written into study.json before the claim run (study.json claims.replicates.by_scenario_metric), with
    `minimum` for any pair it lacks. Returns (cells, counts)."""
    group = CELL + ["metric", "variant", "competitor"]
    cells = []
    for key, part in differences.groupby(group, dropna=False, sort=True):
        record = dict(zip(group, key))
        R = int(planned.get((record["scenario"], record["metric"]), minimum))
        usable = part.loc[part.converged]
        delta = margin(record["metric"], usable.competitor_error.mean() if len(usable) else np.nan, margins)
        row = {**record, "planned": R, "seeds": len(part), "converged_seeds": len(usable), "delta": delta}
        if len(usable) < R:
            cells.append({**row, "class": "not_converged" if len(usable) < len(part) else "inconclusive",
                          "reason": "seeds failing the two-start test" if len(usable) < len(part)
                          else "fewer seeds than planned"})
            continue
        d = usable.d.to_numpy(float)
        half = t_quantile(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
        lower, upper = d.mean() - half, d.mean() + half
        verdict = "not_worse" if upper <= delta else "worse" if lower > delta else "inconclusive"
        cells.append({**row, "mean_d": float(d.mean()), "lower": float(lower), "upper": float(upper),
                      "class": verdict})
    table = pd.DataFrame(cells)
    counts = {c: int((table["class"] == c).sum()) if len(table) else 0 for c in CLASSES}
    return table, counts


def replicates_for(sd, delta, minimum=5):
    """The smallest R >= minimum with R >= (2 t_{R-1} sd / delta)^2 (R4), or MAX_REPLICATES when none up to it is.
    study.py's check_claims calls it too, so a claim run's planned R and this module's sizing are one rule."""
    R = minimum
    while R < MAX_REPLICATES and R < math.ceil((2 * t_quantile(R - 1) * sd / delta) ** 2):
        R += 1
    return R


def replicates_needed(dev_differences, margins, minimum=5, sources=()):
    """R per (scenario, metric) = max(minimum, ceil((2 t_{R-1} sd_dev / delta)^2)), the smallest R meeting it,
    from the paired spreads on development seeds of fits that passed the two-start test (R4), taking the cell
    that needs the most. Returns one row per (scenario, metric) with sd_dev, delta, the deciding cell, R and the
    dev sources (job ids) it rests on."""
    usable = dev_differences.loc[dev_differences.converged]
    out = []
    for (scenario, metric), part in usable.groupby(["scenario", "metric"], sort=True):
        worst = None
        for key, cell in part.groupby(CELL[1:] + ["variant", "competitor"], dropna=False):
            if len(cell) < 2:
                continue
            sd = float(cell.d.std(ddof=1))
            delta = margin(metric, cell.competitor_error.mean(), margins)
            R = replicates_for(sd, delta, minimum)
            if worst is None or R > worst["replicates"]:
                worst = {"scenario": scenario, "metric": metric, "sd_dev": sd, "delta": delta,
                         "cell": dict(zip(CELL[1:] + ["variant", "competitor"], key)), "replicates": R,
                         "attainable": R < MAX_REPLICATES, "dev_seeds": int(len(cell))}
        if worst is not None:
            out.append({**worst, "sources": list(sources)})
    return pd.DataFrame(out)
