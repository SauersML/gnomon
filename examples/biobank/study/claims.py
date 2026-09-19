"""The simulator claim classifier (SPEC section 8): ours against each competitor, cell by cell, on the TRUE
probabilities, as not worse, inconclusive, worse (WORSE = BUG: reproduce on >= 3 seeds, then file), not
converged or not compared.

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
  seeds never enter d or the dev-seed spreads that size R (R4);
- not compared: in some seed the comparison could not be made, and the cell says why by name: a competitor whose
  fit was not certified ("not compared: competitor not_certified (<name>)"; study.py's result rows show such a
  fit with no metric), or a side with no row or no value for the metric (a method not fitted, say). Such a cell
  is never a pass or a tie. by_disease lists every missing comparison of a disease's claim.
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
CLASSES = ("not_worse", "inconclusive", "worse", "not_converged", "not_compared")
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
    """One row per (cell, metric, variant, competitor, seed): d, the competitor's error, whether both fits of that
    seed passed the convergence test (convergence: a frame of scenario, seed, disease, model, variant and
    max_drisk_over_sd; None marks every seed converged), and `missing`, empty where d exists and otherwise the
    named reason the comparison could not be made: "competitor not_certified (<name>)" (or "<ours> not_certified"),
    "no <name> row" or "<name> has no <column>". A seed where neither side has a row or a value is not there."""
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame.fit == "pooled"].copy()
    if frame.empty:
        return pd.DataFrame()
    # A label per horizon, so binary (None) and survival (years) cells group and sort together.
    frame["horizon"] = frame.horizon.map(lambda h: "none" if h is None or pd.isna(h) else f"{float(h):g}")
    certification = frame["certification"] if "certification" in frame else pd.Series("", index=frame.index)
    frame["_uncertified"] = certification.eq("not_certified").to_numpy()
    passed = None if convergence is None else converged_seeds(pd.DataFrame(convergence), limit)
    keys = CELL + ["seed"]
    out = []
    for metric, (column, error, _) in ERRORS.items():
        if column not in frame:
            continue
        for variant in ours:
            for competitor in competitors:
                sides = {}
                for name in (variant, competitor):
                    side = frame.loc[frame.variant == name, keys + [column, "_uncertified"]].set_index(keys)
                    if side.index.has_duplicates:
                        raise ValueError(f"{name} has two pooled rows for one cell and seed")
                    sides[name] = side
                pair = sides[variant].join(sides[competitor], how="outer", lsuffix="_a", rsuffix="_b")
                if pair.empty:
                    continue
                a, b = pair[f"{column}_a"], pair[f"{column}_b"]
                has_a, has_b = pair["_uncertified_a"].notna(), pair["_uncertified_b"].notna()
                missing = np.select(
                    [pair["_uncertified_b"].eq(True), pair["_uncertified_a"].eq(True), ~has_a, ~has_b,
                     a.isna() & b.isna(), a.isna(), b.isna()],
                    [f"competitor not_certified ({competitor})", f"{variant} not_certified", f"no {variant} row",
                     f"no {competitor} row", "neither", f"{variant} has no {column}",
                     f"{competitor} has no {column}"], "")
                keep = missing != "neither"
                pair, a, b, missing = pair.loc[keep], a[keep], b[keep], missing[keep]
                part = pair.index.to_frame(index=False)
                part["metric"], part["variant"], part["competitor"] = metric, variant, competitor
                compared = missing == ""
                part["d"] = np.where(compared, error(a.fillna(0)).to_numpy(float) - error(b.fillna(0)).to_numpy(float),
                                     np.nan)
                part["competitor_error"] = np.where(compared, error(b.fillna(0)).to_numpy(float), np.nan)
                part["converged"] = True if passed is None else [
                    all((s, seed, dis, mod, v) in passed for v in (variant, competitor))
                    for s, seed, dis, mod in zip(part.scenario, part.seed, part.disease, part.model)]
                part["missing"] = missing
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
    `minimum` for any pair it lacks. A cell with any seed whose comparison is missing is not compared, by name,
    whatever its other seeds say. Returns (cells, counts)."""
    group = CELL + ["metric", "variant", "competitor"]
    cells = []
    for key, part in differences.groupby(group, dropna=False, sort=True):
        record = dict(zip(group, key))
        R = int(planned.get((record["scenario"], record["metric"]), minimum))
        absent = part.loc[part.missing != ""]
        if len(absent):
            cells.append({**record, "planned": R, "seeds": len(part), "converged_seeds": np.nan, "delta": np.nan,
                          "class": "not_compared",
                          "reason": "not compared: " + "; ".join(sorted(set(absent.missing))),
                          "seeds_not_compared": ",".join(str(s) for s in sorted(absent.seed))})
            continue
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
    usable = dev_differences.loc[dev_differences.converged & dev_differences.missing.eq("")]
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


def by_disease(table):
    """One row per (scenario, disease, model, variant, competitor) of classify's cells: how many fall in each class,
    and every comparison the claim could not make, by name ("<metric> <stratum> <horizon>: <reason>"). A
    disease's claim is complete only where that list is empty."""
    group = ["scenario", "disease", "model", "variant", "competitor"]
    out = []
    for key, part in table.groupby(group, sort=True):
        absent = part.loc[part["class"] == "not_compared"]
        out.append({**dict(zip(group, key)), **{c: int((part["class"] == c).sum()) for c in CLASSES},
                    "complete": absent.empty,
                    "missing": [f"{r.metric} {r.stratum} {r.horizon}: {r.reason}" for r in absent.itertuples()]})
    return pd.DataFrame(out)
