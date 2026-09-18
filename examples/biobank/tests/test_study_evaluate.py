"""study/evaluate.py against brute-force definitions, each check with a planted error that must break it.

The independent reference implementations (R, pROC, lifelines, scikit-survival, ...) are in
test_study_evaluate_reference.py; these are the fast in-process checks of the O(n log n) algorithms.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import evaluate as ev  # noqa: E402


def brute_auc(score, target, weights):
    """The weighted AUC and its fixed-weight influence variance, pair by pair."""
    score, target, weights = map(np.asarray, (score, target, weights))
    case, ctrl = (target == 1) & (weights > 0), (target == 0) & (weights > 0)
    kernel = (score[case, None] > score[None, ctrl]) + 0.5 * (score[case, None] == score[None, ctrl])
    wc, wk = weights[case], weights[ctrl]
    auc = float(wc @ kernel @ wk / (wc.sum() * wk.sum()))
    v10 = kernel @ wk / wk.sum()
    v01 = wc @ kernel / wc.sum()
    variance = np.sum((wc * (v10 - auc)) ** 2) / wc.sum() ** 2 + np.sum((wk * (v01 - auc)) ** 2) / wk.sum() ** 2
    return auc, variance


def brute_wolbers(t, code, risk, horizon, g=None):
    """Wolbers' competing-risk C by its pair definition; g(j, s) row j's censoring survival just before s,
    for Uno's weights."""
    concordant = comparable = 0.0
    for i in np.flatnonzero((code == 1) & (t <= horizon)):
        gi = 1.0 if g is None else g(i, t[i])
        for j in range(len(t)):
            if j == i:
                continue
            outlives = t[j] > t[i] or (t[j] == t[i] and code[j] != 1)
            died_first = code[j] == 2 and t[j] < t[i]
            if not (outlives or died_first):
                continue
            w = 1.0 if g is None else 1 / (gi * g(j, t[i] if outlives else t[j]))
            comparable += w
            concordant += w * ((risk[i] > risk[j]) + 0.5 * (risk[i] == risk[j]))
    return concordant / comparable


def survival_sample(n, seed, ties=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t1 = rng.exponential(1 / (0.15 * np.exp(0.7 * x)))
    t2 = rng.exponential(1 / 0.05, size=n)
    c = rng.exponential(1 / (0.2 * np.exp(0.5 * (x > 0))))
    t = np.minimum.reduce([t1, t2, c])
    code = np.select([t == t1, t == t2], [1, 2], 0)
    if ties:
        t = np.ceil(t * 12) / 12
    risk = 1 / (1 + np.exp(-(x + rng.normal(scale=0.5, size=n))))
    return t, code, risk, x


# --------------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------------- #
def test_weighted_auc_matches_pairs_and_delong():
    rng = np.random.default_rng(1)
    score = np.round(rng.normal(size=400), 1)  # ties
    target = (rng.random(400) < 0.3).astype(float)
    weights = rng.choice([0.0, 1.0, 1.7, 3.0], size=400)
    auc, influence = ev.weighted_auc(score, target, weights)
    ref_auc, ref_var = brute_auc(score, target, weights)
    assert abs(auc - ref_auc) < 1e-12
    assert abs(np.sum(influence ** 2) / len(score) ** 2 - ref_var) < 1e-12
    # Unit weights: DeLong's AUC, and its variance up to the m/(m-1), n/(n-1) factors.
    ones = np.ones(400)
    auc1, influence1 = ev.weighted_auc(score, target, ones)
    aucs, cov = ev.delong(target, score[None, :])
    m, k = int(target.sum()), int(400 - target.sum())
    assert abs(auc1 - aucs[0]) < 1e-12
    y = target.astype(bool)
    v10 = np.array([np.mean((s > score[~y]) + 0.5 * (s == score[~y])) for s in score[y]])
    v01 = np.array([np.mean((score[y] > s) + 0.5 * (score[y] == s)) for s in score[~y]])
    assert abs(cov[0, 0] - (v10.var(ddof=1) / m + v01.var(ddof=1) / k)) < 1e-12
    assert abs(np.sum(influence1 ** 2) / 400 ** 2 - (v10.var() / m + v01.var() / k)) < 1e-12


def test_planted_tie_error_breaks_the_auc_check():
    rng = np.random.default_rng(2)
    score = np.round(rng.normal(size=300), 1)
    target = (rng.random(300) < 0.4).astype(float)
    weights = rng.choice([1.0, 2.0], size=300)
    reference = brute_auc(score, target, weights)[0]
    assert abs(ev.weighted_auc(score, target, weights)[0] - reference) < 1e-12
    # Ties broken at random instead of counted half: the same check must fail.
    jittered = score + 1e-9 * rng.normal(size=300)
    assert abs(ev.weighted_auc(jittered, target, weights)[0] - reference) > 1e-4


# --------------------------------------------------------------------------- #
# concordance
# --------------------------------------------------------------------------- #
def test_earlier_below_matches_brute_force():
    rng = np.random.default_rng(3)
    rank = rng.integers(0, 37, size=500)
    weight = rng.random(500)
    group = rng.integers(0, 4, size=500)
    below, equal = ev._earlier_below(rank, weight, group)
    for i in range(0, 500, 7):
        earlier = np.arange(i)[group[:i] == group[i]]
        assert abs(below[i] - weight[earlier][rank[earlier] < rank[i]].sum()) < 1e-9
        assert abs(equal[i] - weight[earlier][rank[earlier] == rank[i]].sum()) < 1e-9


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("kind", ["km", "cox", "strata"])
def test_wolbers_concordance_matches_pair_definition(ties, kind):
    t, code, risk, x = survival_sample(300, 4, ties=ties)
    risk = np.round(risk, 2)  # risk ties too
    horizon = float(np.quantile(t, 0.7))
    assert abs(ev.wolbers_concordance(t, code, risk, horizon) - brute_wolbers(t, code, risk, horizon)) < 1e-12
    frame = pd.DataFrame({"followup": t, "event_code": code, "group": np.where(x > 0, "a", "b"), "x": x})
    model = ev.Censoring(frame, horizon, kind, ("group",) if kind == "strata" else ("x",))
    g = lambda j, s: float(model.subset(np.arange(len(t)) == j).at(np.array([s]), left=True)[0])
    assert abs(ev.wolbers_concordance(t, code, risk, horizon, model)
               - brute_wolbers(t, code, risk, horizon, g)) < 1e-12


def test_concordance_of_several_variants_is_each_variant_alone():
    t, code, risk, _ = survival_sample(2000, 17, ties=True)
    frame = pd.DataFrame({"followup": t, "event_code": code, "x": risk})
    model = ev.Censoring(frame, 2.0, "cox", ("x",))
    risks = np.vstack([risk, np.round(risk, 1), 1 - risk])
    for censoring in (None, model):
        together = ev.wolbers_concordance(t, code, risks, 2.0, censoring)
        alone = [ev.wolbers_concordance(t, code, r, 2.0, censoring) for r in risks]
        np.testing.assert_allclose(together, alone, rtol=0, atol=1e-13)
    # The blocked sum does not depend on the block size.
    saved = ev.CONCORDANCE_BLOCK
    try:
        ev.CONCORDANCE_BLOCK = 2000 * 7
        small = ev.wolbers_concordance(t, code, risks, 2.0, model)
    finally:
        ev.CONCORDANCE_BLOCK = saved
    np.testing.assert_allclose(small, ev.wolbers_concordance(t, code, risks, 2.0, model), rtol=0, atol=1e-13)


def test_planted_missing_competing_branch_breaks_the_concordance_check():
    t, code, risk, _ = survival_sample(300, 5)
    horizon = float(np.quantile(t, 0.7))
    # Deaths treated as censorings (the cause-specific C) is not Wolbers' C.
    planted = ev.wolbers_concordance(t, np.where(code == 2, 0, code), risk, horizon)
    assert abs(planted - brute_wolbers(t, code, risk, horizon)) > 1e-3


# --------------------------------------------------------------------------- #
# Aalen-Johansen and censoring
# --------------------------------------------------------------------------- #
def test_aalen_johansen_without_censoring_is_the_empirical_incidence():
    t, code, _, _ = survival_sample(2000, 6)
    code = np.where(code == 0, 2, code)  # no censoring
    horizon = 3.0
    cif, se = ev.aalen_johansen(t, code, horizon)
    empirical = np.mean((code == 1) & (t <= horizon))
    assert abs(cif - empirical) < 1e-12
    assert abs(se - np.sqrt(empirical * (1 - empirical) / len(t))) < 1e-12


def test_aalen_johansen_influence_matches_a_finite_difference_of_case_weights():
    t, code, _, _ = survival_sample(60, 7, ties=True)
    horizon = float(np.quantile(t, 0.8))

    def weighted_cif(w):
        times, k = np.unique(t, return_inverse=True)
        d1 = np.bincount(k, weights=w * (code == 1), minlength=len(times))
        d = np.bincount(k, weights=w * (code != 0), minlength=len(times))
        at_risk = np.cumsum(np.bincount(k, weights=w, minlength=len(times))[::-1])[::-1]
        before = np.r_[1.0, np.cumprod(1 - d / at_risk)[:-1]]
        return float(np.sum(np.where(times <= horizon, before * d1 / at_risk, 0.0)))

    eps = 1e-7
    derivative = np.array([(weighted_cif(np.ones(60) + eps * np.eye(60)[l]) - weighted_cif(np.ones(60))) / eps
                           for l in range(60)])
    cif, se = ev.aalen_johansen(t, code, horizon)
    assert abs(cif - weighted_cif(np.ones(60))) < 1e-12
    assert abs(se - np.sqrt(np.sum(derivative ** 2))) < 1e-5 * se


def test_reverse_km_puts_events_before_censorings():
    # Two rows at t=1: an event and a censoring. The censoring's risk set excludes the event.
    times, g, _ = ev.reverse_km([1.0, 1.0, 2.0, 3.0], [1, 0, 0, 1])
    np.testing.assert_allclose(g, [1 - 1 / 3, (1 - 1 / 3) * (1 - 1 / 2), (1 - 1 / 3) * (1 - 1 / 2)])


def test_cox_censoring_model_solves_its_score_equation_and_reduces_to_nelson_aalen():
    rng = np.random.default_rng(8)
    n = 3000
    z = rng.integers(0, 3, size=n)
    a = rng.normal(size=n)
    frame = pd.DataFrame({"followup": np.round(rng.exponential(1 / (0.3 * np.exp(0.4 * z - 0.3 * a))), 2) + 0.01,
                          "event_code": rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1], size=n),
                          "site": z.astype(str), "entry_age": a})
    X = ev.censoring_design(frame, ("site", "entry_age"))
    t, d = frame.followup.to_numpy(), (frame.event_code == 0).to_numpy(float)
    beta, times, base = ev.cox_fit(t, d, X, ridge=0.0)
    Xc = X - X.mean(axis=0)
    r = np.exp(Xc @ beta)
    score = sum(d[i] * (Xc[i] - (r[t >= t[i]] @ Xc[t >= t[i]]) / r[t >= t[i]].sum()) for i in range(0, n)
                if d[i])
    assert np.max(np.abs(score)) < 1e-6
    # No covariates: the Breslow baseline is the Nelson-Aalen of censoring with every row at risk at its time.
    _, times0, base0 = ev.cox_fit(t, d, np.zeros((n, 0)), ridge=0.0)
    u = np.unique(t[d > 0])
    nelson = np.cumsum([np.sum((t == s) & (d > 0)) / np.sum(t >= s) for s in u])
    np.testing.assert_allclose(times0, u)
    np.testing.assert_allclose(base0, nelson, rtol=1e-12)


def test_ipcw_weights_are_one_without_censoring_and_zero_for_censored_rows():
    frame = pd.DataFrame({"followup": [0.5, 1.5, 2.5, 0.7], "event_code": [1, 2, 0, 0]})
    model = ev.Censoring(frame, 2.0, "km")
    w = ev.ipcw(frame, 2.0, model)
    # Censored at 0.7 before h: weight 0; the others divide by G = 1 - 1/3 after the censoring at 0.7.
    np.testing.assert_allclose(w, [1.0, 1.5, 1.5, 0.0])


@pytest.mark.parametrize("kind", ["km", "cox", "strata"])
def test_a_censoring_at_the_horizon_keeps_rows_followed_past_it_at_risk(kind):
    # Rows followed past h were at risk of censoring at h: G(h) = 1 - 1/3, not 0.
    frame = pd.DataFrame({"followup": [1.0, 2.0, 3.0, 3.0], "event_code": [1, 0, 0, 2], "site": ["a"] * 4})
    model = ev.Censoring(frame, 2.0, kind, ("site",) if kind != "km" else ())
    np.testing.assert_allclose(ev.ipcw(frame, 2.0, model), [1.0, 0.0, 1.5, 1.5], rtol=1e-12)
    followed, g_upper = ev.cell_support(frame.followup.to_numpy(), frame.event_code.to_numpy(), 2.0)
    assert followed == 2 and g_upper > 2 / 3


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_calibration_recovers_known_miscalibration():
    rng = np.random.default_rng(9)
    lp = rng.normal(-1, 1.2, size=200000)
    y = (rng.random(len(lp)) < 1 / (1 + np.exp(-(0.3 + 0.8 * lp)))).astype(float)
    cal = ev.calibration(y, 1 / (1 + np.exp(-lp)))
    assert abs(cal["cal_slope"] - 0.8) < 4 * cal["cal_slope_se"]
    # The intercept with slope fixed at one is the calibration-in-the-large: mean(y) matched on the logit scale.
    p_fitted = 1 / (1 + np.exp(-(lp + cal["cal_int"])))
    assert abs(p_fitted.mean() - y.mean()) < 1e-9


def test_integrated_calibration_index_of_a_calibrated_model_is_small_and_of_a_shifted_one_is_not():
    rng = np.random.default_rng(10)
    p = rng.beta(2, 8, size=20000)
    y = (rng.random(len(p)) < p).astype(float)
    assert ev.integrated_calibration_index(y, p) < 0.01
    assert ev.integrated_calibration_index(y, np.clip(p + 0.05, 0, 1)) > 0.04


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def table_frame(n, seed, kind):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "ancestry": rng.choice(["afr", "amr", "eur", "sas", "mid"], p=[0.2, 0.15, 0.6, 0.045, 0.005], size=n),
        "region": rng.choice(["Northeast", "South", "West", "unknown"], size=n),
        "division": rng.choice(["New England", "Pacific"], size=n),
        "ehr_site": rng.choice([f"s{i}" for i in range(6)], size=n),
        "sex": rng.integers(0, 2, size=n),
        "age_band": rng.choice(["18-39", "40-59", "60-74", "75+"], size=n),
        "ses_quartile": pd.Series(rng.choice(["q1", "q2", "q3", "q4", None], size=n)),
        "lookback_tertile": rng.choice(["t1", "t2", "t3"], size=n),
        "entry_age": rng.uniform(20, 80, size=n),
    })
    x = rng.normal(size=n)
    if kind == "binary":
        frame["baseline_year"] = rng.choice([2018, 2019, 2020], size=n)
        frame["y"] = (rng.random(n) < 1 / (1 + np.exp(-(x - 1.5)))).astype(int)
    else:
        frame["entry_year"] = rng.choice([2018, 2019, 2020], size=n)
        t, code, _, x = survival_sample(n, seed)
        frame["followup"], frame["event"] = t, code
        frame["max_followup"] = rng.uniform(1.5, 6, size=n)
    return frame, x


def predictions_for(frame, x, kind, horizons, logo=None):
    rng = np.random.default_rng(11)
    out = {}
    for k, variant in enumerate(("ours", "covariates", "standard")):
        base = 1 / (1 + np.exp(-(x * (1 - 0.3 * k) - 1.5 + rng.normal(scale=0.3, size=len(x)))))
        risk = base if kind == "binary" else np.column_stack([1 - (1 - base) ** h for h in horizons])
        out[(variant, "pooled")] = risk
        if logo:
            axis, group = logo.split(":")[1:]
            held = (frame[axis].astype(str) == group).to_numpy()
            logo_risk = np.where(held if risk.ndim == 1 else held[:, None], risk * 0.9, np.nan)
            out[(variant, logo)] = logo_risk
    return out


def truth_for(frame, kind, horizons, seed=18):
    rng = np.random.default_rng(seed)
    if kind == "binary":
        return pd.DataFrame({"p_ever": rng.beta(2, 8, size=len(frame))})
    truth = pd.DataFrame({f"cif_{h:g}y": rng.beta(2, 8, size=len(frame)) for h in horizons})
    truth["uncensored_event"] = np.where(frame.event == 0, rng.integers(0, 3, size=len(frame)), frame.event)
    truth["uncensored_exit_age"] = frame.entry_age + frame.followup + np.where(frame.event == 0, 1.0, 0.0)
    return truth


@pytest.mark.parametrize("kind", ["binary", "survival"])
def test_every_released_field_is_in_the_registry_and_every_axis_partitions(kind):
    horizons = [1.0, 2.0]
    frame, x = table_frame(6000, 12, kind)
    predictions = predictions_for(frame, x, kind, horizons, logo="logo:ancestry:afr")
    rows = ev.evaluate(kind, frame, predictions, horizons, {"report": {"small_cell_max": 20}},
                       truth=truth_for(frame, kind, horizons))
    assert any("rmse_true" in r for r in rows)
    if kind == "survival":
        assert any("auc_unc" in r and "c_uno" in r for r in rows)
    for row in rows:
        for key, value in row.items():
            if key in ev.KEYS or isinstance(value, str):
                continue
            assert key in ev.METRICS, key
    pooled = [r for r in rows if r["fit"] == "pooled" and r["variant"] == "ours"]
    for horizon in ({None} if kind == "binary" else set(horizons)):
        at = {r["stratum"]: r for r in pooled if r["horizon"] == horizon}
        for axis in ev.AXES[kind]:
            members = [r for s, r in at.items() if s.startswith(axis + "_")]
            assert sum(r["n"] for r in members) == at["overall"]["n"], axis
    logo = [r for r in rows if r["fit"] != "pooled"]
    assert logo and all(r["stratum"] == "overall" for r in logo)
    assert all("d_auc_pooled" in r for r in logo if r["status"] == "ok")
    # A cell under the minimum carries only its counts and status.
    small = [r for r in rows if r["status"] == ev.INSUFFICIENT]
    assert small and all(not any(k in r for k in ("auc", "brier", "oe")) for r in small)


def test_a_cell_with_twenty_cases_is_withheld_and_a_planted_minimum_releases_it():
    frame, x = table_frame(3000, 13, "binary")
    frame["ancestry"] = "eur"
    frame.loc[:59, "ancestry"] = "mid"
    frame.loc[:59, "y"] = np.r_[np.ones(20, int), np.zeros(40, int)]
    predictions = predictions_for(frame, x, "binary", [])
    status = lambda rows: {r["status"] for r in rows if r["stratum"] == "ancestry_mid"}
    rows = ev.evaluate("binary", frame, predictions, [], {"report": {"small_cell_max": 20}})
    assert status(rows) == {ev.INSUFFICIENT}
    assert all(min(r["n"], r["cases"], r["n"] - r["cases"]) > 20 for r in rows if r["status"] == "ok")
    planted = ev.evaluate("binary", frame, predictions, [], {"report": {"small_cell_max": 4}})
    assert status(planted) == {"ok"}


def test_logo_transport_difference_is_logo_minus_pooled_on_the_held_out_rows():
    frame, x = table_frame(6000, 16, "binary")
    rows = ev.evaluate("binary", frame, predictions_for(frame, x, "binary", [], logo="logo:ancestry:afr"), [],
                       {})
    pooled = {r["variant"]: r for r in rows if r["fit"] == "pooled" and r["stratum"] == "ancestry_afr"}
    for row in (r for r in rows if r["fit"] == "logo:ancestry:afr"):
        assert abs(row["d_auc_pooled"] - (row["auc"] - pooled[row["variant"]]["auc"])) < 1e-12
        assert abs(row["d_brier_pooled"] - (row["brier"] - pooled[row["variant"]]["brier"])) < 1e-12
        assert row["n"] == pooled[row["variant"]]["n"]


def test_logo_predictions_must_cover_exactly_the_held_out_group():
    frame, x = table_frame(2000, 14, "binary")
    predictions = predictions_for(frame, x, "binary", [], logo="logo:ancestry:afr")
    leaked = dict(predictions)
    risk = leaked[("ours", "logo:ancestry:afr")].copy()
    risk[np.flatnonzero(frame.ancestry != "afr")[0]] = 0.5
    leaked[("ours", "logo:ancestry:afr")] = risk
    with pytest.raises(ValueError, match="held-out group"):
        ev.evaluate("binary", frame, leaked, [], {})


def test_a_horizon_evaluates_only_rows_whose_administrative_follow_up_reaches_it():
    frame, x = table_frame(6000, 19, "survival")
    frame = frame.drop(columns="max_followup")
    frame["admin_years"] = np.random.default_rng(20).uniform(1.0, 4.0, size=len(frame))
    horizons = [1.0, 2.0]
    rows = ev.evaluate("survival", frame, predictions_for(frame, x, "survival", horizons), horizons,
                       {"cohort": {"landmark_days": 180}})
    for horizon in horizons:
        overall = [r for r in rows if r["stratum"] == "overall" and r["horizon"] == horizon][0]
        assert overall["n"] == int(np.sum(frame.admin_years - 180 / 365.25 >= horizon))


def test_paired_differences_are_differences_of_the_rows_own_metrics():
    horizons = [1.0, 2.0]
    frame, x = table_frame(8000, 15, "survival")
    rows = ev.evaluate("survival", frame, predictions_for(frame, x, "survival", horizons), horizons, {})
    at = {(r["variant"], r["stratum"], r["horizon"]): r for r in rows if r["status"] == "ok"}
    for (variant, stratum, horizon), row in at.items():
        for ref in ev.REFERENCES:
            if variant != ref and (ref, stratum, horizon) in at:
                other = at[(ref, stratum, horizon)]
                assert abs(row[f"d_auc_{ref}"] - (row["auc"] - other["auc"])) < 1e-12
                assert abs(row[f"d_brier_{ref}"] - (row["brier"] - other["brier"])) < 1e-12
