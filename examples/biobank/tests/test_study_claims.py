"""study/claims.py: the simulator claim classifier's classes, margins, convergence gate and replicate sizing,
each on a planted case whose class is known."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import claims  # noqa: E402

MARGINS = {"rmse_true": {"delta": 0.02, "scale": "relative", "floor": 0.0005},
           "oe_true": {"delta": 0.02, "scale": "absolute"},
           "cal_slope_true": {"delta": 0.05, "scale": "absolute"},
           "auc_true": {"delta": 0.002, "scale": "absolute"},
           "slope_recovery": {"delta": 0.05, "scale": "relative", "floor": 0.01}}


def rows(values, seeds=5, scenario="realistic", stratum="overall"):
    """Pooled rows for ours and one competitor over seeds: values maps metric column -> (ours per seed,
    competitor per seed)."""
    out = []
    for s in range(seeds):
        for v, i in (("ours", 0), ("standard", 1)):
            row = {"scenario": scenario, "seed": s, "disease": "t2d", "model": "binary", "variant": v,
                   "fit": "pooled", "stratum": stratum, "horizon": None}
            row.update({column: float(pair[i][s]) for column, pair in values.items()})
            out.append(row)
    return out


def verdict(values, planned=None, convergence=None, metric=None):
    d = claims.paired_differences(rows(values), ["ours"], ["standard"], convergence)
    table, counts = claims.classify(d, MARGINS, planned or {})
    if metric is not None:
        table = table.loc[table.metric == metric]
    return table.iloc[0], counts


def test_a_clearly_better_and_a_clearly_worse_cell():
    competitor = np.full(5, 0.0100)
    better = verdict({"rmse_true": (competitor - 0.001 + 1e-6 * np.arange(5), competitor)})[0]
    assert better["class"] == "not_worse" and better["delta"] == pytest.approx(0.0005)
    # Ours worse by 2 delta with a small spread: the lower bound clears delta.
    worse = verdict({"rmse_true": (competitor + 0.001 + 1e-6 * np.arange(5), competitor)})[0]
    assert worse["class"] == "worse"


def test_a_mean_exactly_at_the_margin_is_inconclusive():
    competitor = np.full(5, 0.05)
    delta = 0.02 * 0.05  # the relative margin exceeds the floor here
    d = delta + np.array([-2, -1, 0, 1, 2]) * 1e-4
    cell = verdict({"rmse_true": (competitor + d, competitor)})[0]
    assert cell["delta"] == pytest.approx(delta)
    assert cell["mean_d"] == pytest.approx(delta)
    assert cell["class"] == "inconclusive"


def test_the_relative_margin_floors():
    tiny = np.full(5, 1e-4)
    cell = verdict({"rmse_true": (tiny + 0.0003 + 1e-7 * np.arange(5), tiny)})[0]
    # 2% of 1e-4 is 2e-6; the floor 0.0005 governs, and 0.0003 worse is within it.
    assert cell["delta"] == pytest.approx(0.0005) and cell["class"] == "not_worse"
    slope = verdict({"slope_rmse_true": (tiny + 0.005 + 1e-7 * np.arange(5), tiny)})[0]
    assert slope["delta"] == pytest.approx(0.01) and slope["class"] == "not_worse"


def test_orientations_are_worse_when_positive():
    # AUC: competitor minus ours. A lower expected AUC by 0.005 is worse.
    auc = verdict({"auc_true": (np.full(5, 0.700) + 1e-6 * np.arange(5), np.full(5, 0.705))})[0]
    assert auc["mean_d"] == pytest.approx(0.005, abs=1e-5) and auc["class"] == "worse"
    # O/E and calibration slope: the distance from 1, on either side.
    oe = verdict({"oe_true": (np.full(5, 0.95) + 1e-6 * np.arange(5), np.full(5, 1.04))})[0]
    assert oe["mean_d"] == pytest.approx(0.01, abs=1e-5) and oe["class"] == "not_worse"
    slope = verdict({"cal_slope_true": (np.full(5, 1.10) + 1e-6 * np.arange(5), np.full(5, 0.99))})[0]
    assert slope["class"] == "worse"
    bias = verdict({"slope_bias_true": (np.full(5, -0.05) + 1e-6 * np.arange(5), np.full(5, 0.01))})[0]
    assert bias["mean_d"] == pytest.approx(0.04, abs=1e-5) and bias["class"] == "worse"


def test_a_seed_failing_the_two_start_test_leaves_the_classification():
    competitor = np.full(5, 0.0100)
    values = {"rmse_true": (competitor - 0.001 + 1e-6 * np.arange(5), competitor)}
    convergence = pd.DataFrame([{"scenario": "realistic", "seed": s, "disease": "t2d", "model": "binary",
                                 "variant": v, "max_drisk_over_sd": 0.02 if (s, v) == (3, "ours") else 0.001}
                                for s in range(5) for v in ("ours", "standard")])
    cell, counts = verdict(values, convergence=convergence)
    assert cell["class"] == "not_converged" and cell["converged_seeds"] == 4
    assert counts["not_converged"] == 1 and counts["not_worse"] == 0
    # With one seed more than planned remaining, the four converged seeds classify.
    cell, _ = verdict(values, planned={("realistic", "rmse_true"): 4}, convergence=convergence)
    assert cell["class"] == "not_worse" and cell["converged_seeds"] == 4
    # A fit with no convergence record never passed.
    cell, _ = verdict(values, convergence=convergence.loc[convergence.seed != 0])
    assert cell["class"] == "not_converged"


def test_binary_and_survival_cells_classify_together():
    competitor = np.full(5, 0.0100)
    binary = rows({"rmse_true": (competitor - 0.001 + 1e-6 * np.arange(5), competitor)})
    survival = [dict(r, model="survival", horizon=h) for r in binary for h in (1.0, 3.0)]
    d = claims.paired_differences(binary + survival, ["ours"], ["standard"])
    table, counts = claims.classify(d, MARGINS, {})
    assert sorted(set(table.horizon)) == ["1", "3", "none"] and counts["not_worse"] == 3
    assert len(claims.replicates_needed(d, MARGINS)) == 1


def test_replicates_sized_from_converged_dev_spreads():
    competitor = np.full(8, 0.05)
    rng = np.random.default_rng(1)
    d = rng.normal(0, 0.0004, size=8)
    dev = claims.paired_differences(rows({"rmse_true": (competitor + d, competitor)}, seeds=8), ["ours"],
                                    ["standard"])
    sizing = claims.replicates_needed(dev, MARGINS, sources=["job1"]).iloc[0]
    sd, delta = float(np.std(d, ddof=1)), 0.001
    R = sizing["replicates"]
    assert sizing["sd_dev"] == pytest.approx(sd) and sizing["delta"] == pytest.approx(delta)
    assert R >= math.ceil((2 * claims.t_quantile(R - 1) * sd / delta) ** 2) and R >= 5
    assert R == 5 or R - 1 < math.ceil((2 * claims.t_quantile(R - 2) * sd / delta) ** 2)
    assert sizing["sources"] == ["job1"]
    # Planted: a dev seed whose fit failed the two-start test, with a wild d, must not move sd_dev.
    wild = dev.copy()
    wild.loc[wild.seed == 7, "d"] = 0.05
    wild.loc[wild.seed == 7, "converged"] = False
    kept = wild.loc[wild.seed != 7]
    assert claims.replicates_needed(wild, MARGINS).iloc[0]["sd_dev"] == pytest.approx(kept.d.std(ddof=1))


def test_replicates_for_uses_t_with_r_minus_one_degrees_of_freedom():
    # sd / delta = 1/2: R = 5 needs (2 x 2.776 / 2)^2 = 7.7 and R = 6 needs 6.6, so R = 7 (t_6 = 2.447 gives 6.0).
    assert claims.replicates_for(0.001, 0.002) == 7
    # Planted: a fixed t = 1.96 would stop at the minimum, (2 x 1.96 / 2)^2 = 3.8 <= 5.
    assert math.ceil((2 * 1.96 * 0.001 / 0.002) ** 2) <= 5
    assert claims.replicates_for(0.001, 0.002, minimum=10) == 10
    assert claims.replicates_for(1.0, 1e-6) == claims.MAX_REPLICATES


def test_a_cell_where_ours_or_the_competitor_has_no_row_is_not_compared_by_name():
    """study-pipe's time_scale_not_identified: a method that is not fitted has no rows. A cell (disease x scenario)
    where ours or that competitor lacks a row in some seed is never read as a difference: it is not compared, and
    says which row is missing; the differences that do exist are unchanged. A seed with neither side (never run)
    only leaves the cell short of seeds."""
    competitor = np.full(5, 0.0100)
    values = {"rmse_true": (competitor - 0.001 + 1e-6 * np.arange(5), competitor)}
    both = rows(values) + [dict(r, disease="cad") for r in rows(values)]
    # Ours is not identified for CAD in any seed, and the competitor not for T2D in seed 4.
    partial = [r for r in both if not (r["disease"] == "cad" and r["variant"] == "ours")
               and not (r["disease"] == "t2d" and r["variant"] == "standard" and r["seed"] == 4)]
    d = claims.paired_differences(partial, ["ours"], ["standard"])
    compared = d.loc[d.missing == ""].drop(columns="missing").reset_index(drop=True)
    assert set(compared.disease) == {"t2d"} and sorted(compared.seed) == [0, 1, 2, 3]
    full = claims.paired_differences(both, ["ours"], ["standard"])
    assert (full.missing == "").all()
    kept = full.loc[(full.disease == "t2d") & (full.seed != 4)].drop(columns="missing").reset_index(drop=True)
    pd.testing.assert_frame_equal(compared, kept)
    table, counts = claims.classify(d, MARGINS, {})
    by = table.set_index("disease")
    assert by.loc["t2d", "class"] == "not_compared" and by.loc["t2d", "reason"] == "not compared: no standard row"
    assert by.loc["t2d", "seeds_not_compared"] == "4"
    assert by.loc["cad", "class"] == "not_compared" and by.loc["cad", "reason"] == "not compared: no ours row"
    assert counts["not_compared"] == 2 and counts["not_worse"] == 0
    # A fifth seed never run on either side: four seeds against a planned five, inconclusive, not a failure.
    four = [r for r in both if r["disease"] == "t2d" and r["seed"] != 4]
    table, _ = claims.classify(claims.paired_differences(four, ["ours"], ["standard"]), MARGINS, {})
    assert table.iloc[0]["class"] == "inconclusive" and table.iloc[0]["reason"] == "fewer seeds than planned"


def test_a_competitor_not_certified_is_not_compared_and_never_a_pass():
    """study.py shows a competitor cell whose fit was not certified with certification not_certified and no
    metric (35378b3f). The cell, which would be not worse against the certified competitor, is not compared by
    name, in any seed it happens, and the disease's claim lists it as missing."""
    competitor = np.full(5, 0.0100)
    values = {"rmse_true": (competitor - 0.001 + 1e-6 * np.arange(5), competitor),
              "oe_true": (np.full(5, 1.001), np.full(5, 1.0))}
    certified = rows(values)
    table, counts = claims.classify(claims.paired_differences(certified, ["ours"], ["standard"]), MARGINS, {})
    assert counts["not_worse"] >= 1 and counts["not_compared"] == 0
    # The competitor's seed-2 fit is not certified: its row keeps its keys and reason, and shows no metric.
    strip = lambda r: {k: v for k, v in r.items() if k not in values} | {"certification": "not_certified"}
    uncertified = [strip(r) if r["variant"] == "standard" and r["seed"] == 2 else r for r in certified]
    d = claims.paired_differences(uncertified, ["ours"], ["standard"])
    table, counts = claims.classify(d, MARGINS, {})
    assert set(table["class"]) == {"not_compared"} and counts["not_worse"] == counts["inconclusive"] == 0
    assert set(table.reason) == {"not compared: competitor not_certified (standard)"}
    assert set(table.seeds_not_compared) == {"2"}
    summary = claims.by_disease(table).iloc[0]
    assert not summary.complete and summary.not_compared == len(table)
    assert all(m.endswith("not compared: competitor not_certified (standard)") for m in summary.missing)
    assert sorted(m.split()[0] for m in summary.missing) == ["oe_true", "rmse_true"]
    # The certified run's claim is complete.
    table, _ = claims.classify(claims.paired_differences(certified, ["ours"], ["standard"]), MARGINS, {})
    assert claims.by_disease(table).iloc[0].complete
