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
            died_first = code[j] >= 2 and t[j] < t[i]
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
def aalen_johansen(t, code, horizon):
    """The Aalen-Johansen incidence of cause 1 by its product-limit definition (every other exit competing)."""
    times, k = np.unique(t, return_inverse=True)
    d1 = np.bincount(k, weights=code == 1, minlength=len(times))
    d = np.bincount(k, weights=code != 0, minlength=len(times))
    at_risk = np.cumsum(np.bincount(k, minlength=len(times))[::-1])[::-1]
    before = np.r_[1.0, np.cumprod(1 - d / at_risk)[:-1]]
    return float(np.sum(np.where(times <= horizon, before * d1 / at_risk, 0.0)))


@pytest.mark.parametrize("ties", [False, True])
def test_ipcw_observed_risk_with_a_km_g_is_the_aalen_johansen_incidence(ties):
    # Events leave the censoring risk set first, so n S(u-) G(u-) = Y(u) even at tied times, and the IPCW mean
    # of the outcome is the Aalen-Johansen incidence exactly.
    t, code, _, _ = survival_sample(3000, 6, ties=ties)
    code = np.where((code == 2) & (np.arange(len(code)) % 3 == 0), 3, code)
    horizon = 2.5
    frame = pd.DataFrame({"followup": t, "event_code": code})
    w = ev.ipcw(frame, horizon, ev.Censoring(frame, horizon, "km"))
    y = (code == 1) & (t <= horizon)
    assert abs(np.mean(w * y) - aalen_johansen(t, code, horizon)) < 1e-12
    # Planted: 1 - KM with every competing exit censored is not the competing-risk incidence.
    times, km, _ = ev.reverse_km(t, np.where(code == 1, 0, 1))
    assert abs(1 - float(ev.step_at(times, km, horizon)) - aalen_johansen(t, code, horizon)) > 1e-3


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
    beta, times, base, _ = ev.cox_fit(t, d, X, ridge=0.0)
    Xc = X - X.mean(axis=0)
    r = np.exp(Xc @ beta)
    score = sum(d[i] * (Xc[i] - (r[t >= t[i]] @ Xc[t >= t[i]]) / r[t >= t[i]].sum()) for i in range(0, n)
                if d[i])
    assert np.max(np.abs(score)) < 1e-6
    # No covariates: the Breslow baseline is the Nelson-Aalen of censoring with every row at risk at its time.
    _, times0, base0, _ = ev.cox_fit(t, d, np.zeros((n, 0)), ridge=0.0)
    u = np.unique(t[d > 0])
    nelson = np.cumsum([np.sum((t == s) & (d > 0)) / np.sum(t >= s) for s in u])
    np.testing.assert_allclose(times0, u)
    np.testing.assert_allclose(base0, nelson, rtol=1e-12)


@pytest.mark.parametrize("kind", ["cox", "km", "strata"])
def test_total_influence_is_the_jackknife_of_the_whole_estimator(kind):
    """The total influence (given G, plus G's own estimation) of the IPCW AUC, Brier, observed risk and
    calibration intercept and slope is the first-order change when one row is dropped and G refitted without it;
    the influence given G alone is not."""
    rng = np.random.default_rng(24)
    n, horizon = 3600, 2.0
    site, x = rng.integers(0, 3, size=n), rng.normal(size=n)
    t1 = rng.exponential(1 / (0.15 * np.exp(0.7 * x)))
    t2 = rng.exponential(1 / 0.05, size=n)
    c = rng.exponential(1 / (0.2 * np.exp(0.6 * site)))
    t = np.minimum.reduce([t1, t2, c])
    frame = pd.DataFrame({"followup": t, "event_code": np.select([t == t1, t == t2], [1, 2], 0),
                          "site": site.astype(str), "age": rng.normal(size=n)})
    p = 1 / (1 + np.exp(-(x - 1)))
    covariates = ("site",) if kind == "strata" else ("site", "age")

    def metrics(rows):
        # Each metric's value, its influence given G, and its derivative in each row's log weight.
        part = frame.loc[rows].reset_index(drop=True)
        model = ev.Censoring(part, horizon, kind, covariates)
        w = ev.ipcw(part, horizon, model)
        y = ((part.event_code == 1) & (part.followup <= horizon)).to_numpy(float)
        m = len(rows)
        auc, influence = ev.weighted_auc(p[rows], y, w)
        loss, observed = w * (y - p[rows]) ** 2, w * y
        known = w > 0
        lp = ev.clipped_logit(p[rows][known])
        out = {"auc": (auc, influence / m, influence / m),
               "brier": (loss.mean(), (loss - loss.mean()) / m, loss / m),
               "risk": (observed.mean(), (observed - observed.mean()) / m, observed / m)}
        for name, X, offset, k in (("cal_int", np.ones((known.sum(), 1)), lp, 0),
                                   ("cal_slope", np.column_stack([np.ones(known.sum()), lp]), None, 1)):
            beta, _, _, row_influence = ev.logistic_fit(X, y[known], w[known], offset=offset)
            padded = np.zeros(m)
            padded[known] = row_influence[:, k]
            out[name] = (beta[k], padded, padded)
        return model, out

    everyone = np.arange(n)
    model, full = metrics(everyone)
    fixed, total = {}, {}
    for name, (value, given, sensitivity) in full.items():
        fixed[name], total[name] = given, model.influence(everyone, given, sensitivity)
        variance = model.variance(everyone, given, sensitivity)
        assert abs(variance - np.sum(total[name] ** 2)) < 1e-10 * variance
    sample = rng.choice(n, size=40, replace=False)
    jack = {name: [] for name in full}
    for i in sample:
        _, dropped = metrics(everyone[everyone != i])
        for name in full:
            jack[name].append((n - 1) / n * (full[name][0] - dropped[name][0]))
    # The agreement is judged in the norm the variance uses, the root sum of squares over the sampled rows; the
    # largest single-row error, dominated by high-leverage rows' second-order terms, is printed beside it.
    errors = {}
    for name in full:
        jackknife = np.array(jack[name])
        norm = np.linalg.norm(jackknife)
        errors[name] = {"total": np.linalg.norm(total[name][sample] - jackknife) / norm,
                        "given_g": np.linalg.norm(fixed[name][sample] - jackknife) / norm,
                        "total_max": np.max(np.abs(total[name][sample] - jackknife)) / np.max(np.abs(jackknife))}
    print(f"{kind}: relative |influence - jackknife| {errors}")
    for name, error in errors.items():
        assert error["total"] < 0.02, name
    # Planted: for a mean of weighted terms the influence given G alone misses G's own estimation by far more
    # (a censored row moves G, not the terms). For the AUC the censoring term is small at the row level.
    for name in ("brier", "risk"):
        assert errors[name]["given_g"] > 3 * errors[name]["total"], name


def test_a_cell_with_a_row_weighted_above_twenty_is_unsupported():
    # A small site loses almost everyone to follow-up by h, so its few rows followed past h carry weights
    # 1/G_i(h) > 20 under the Cox G, while the cell's own marginal G(h) is far above the floor.
    rng = np.random.default_rng(26)
    n, horizon = 3000, 2.0
    site = np.where(rng.random(n) < 0.1, "b", "a")
    t1 = rng.exponential(1 / 0.2, size=n)
    t2 = rng.exponential(1 / 0.05, size=n)
    c = rng.exponential(1 / np.where(site == "b", 2.0, 0.05))
    t = np.minimum.reduce([t1, t2, c])
    frame = pd.DataFrame({"followup": t, "event_code": np.select([t == t1, t == t2], [1, 2], 0), "site": site})
    p = np.clip(rng.beta(2, 5, size=n), 0.01, 0.99)[None, :]

    def status(rows):
        part = frame.loc[rows].reset_index(drop=True)
        model = ev.Censoring(part, horizon, "cox", ("site",))
        w = ev.ipcw(part, horizon, model)
        cell = ev.survival_cell(part.followup.to_numpy(), part.event_code.to_numpy(), w, model,
                                np.arange(len(part)), p[:, rows], ["ours"], "pooled", "overall", horizon, 21)
        return cell[0]["status"], float(w.max()), ev.cell_support(part.followup.to_numpy(),
                                                                  part.event_code.to_numpy(), horizon)[1]

    everyone, weight, g_upper = status(np.arange(n))
    assert weight > 1 / ev.POSITIVITY_FLOOR and g_upper > ev.POSITIVITY_FLOOR
    assert everyone == ev.UNSUPPORTED
    # The same cell without the small site is reported.
    assert status(np.flatnonzero(site == "a"))[0] == "ok"


@pytest.mark.parametrize("kind", ["cox", "km", "strata"])
def test_a_window_without_censorings_weighs_every_row_one(kind):
    # Censoring only at the CDR cutoff, past the horizon for every eligible row (the censor="cutoff" frames).
    t, code, risk, x = survival_sample(2000, 25)
    code = np.where(code == 0, 2, code)
    frame = pd.DataFrame({"followup": t, "event_code": code, "site": np.where(x > 0, "a", "b"), "x": x})
    horizon = 2.0
    model = ev.Censoring(frame, horizon, kind, ("site",) if kind == "strata" else ("site", "x"))
    w = ev.ipcw(frame, horizon, model)
    np.testing.assert_array_equal(w, np.ones(len(t)))
    y = ((code == 1) & (t <= horizon)).astype(float)
    rows = np.arange(len(t))
    # With nothing to estimate in G the total variance is the one given G.
    given = (y - y.mean()) / len(t)
    assert abs(model.variance(rows, given, y / len(t)) - np.sum(given ** 2)) < 1e-15


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


def test_expected_auc_is_the_pair_definition_under_the_true_probabilities():
    rng = np.random.default_rng(27)
    p_true = rng.beta(2, 6, size=300)
    p = np.round(p_true + rng.normal(scale=0.05, size=300), 2)  # ties
    kernel = (p[:, None] > p[None, :]) + 0.5 * (p[:, None] == p[None, :])
    pair = p_true[:, None] * (1 - p_true[None, :])
    np.fill_diagonal(pair, 0.0)
    reference = float(np.sum(pair * kernel) / pair.sum())
    assert abs(ev.expected_auc(p, p_true) - reference) < 1e-12
    # Planted: keeping the self pairs (a row as its own control) is not the expected AUC.
    with_self = p_true[:, None] * (1 - p_true[None, :])
    assert abs(float(np.sum(with_self * kernel) / with_self.sum()) - reference) > 1e-4


def test_truth_metrics_recover_known_miscalibration_and_slope_error():
    rng = np.random.default_rng(28)
    true_logit = rng.normal(-1.5, 1.0, size=20000)
    true_risk = 1 / (1 + np.exp(-true_logit))
    # A prediction whose logit is 0.2 + 1.25 x the true logit: the slope of true on predicted logit is 1 / 1.25.
    p = 1 / (1 + np.exp(-(0.2 + 1.25 * true_logit)))
    true_slope = rng.normal(0.3, 0.05, size=20000)
    S = (true_slope + 0.02)[None, :]
    row = ev.truth_metrics(p[None, :], true_risk, S, true_slope)[0]
    assert abs(row["cal_slope_true"] - 0.8) < 1e-9
    assert abs(row["oe_true"] - true_risk.mean() / p.mean()) < 1e-15
    assert abs(row["slope_bias_true"] - 0.02) < 1e-12 and abs(row["slope_rmse_true"] - 0.02) < 1e-12
    assert row["auc_true"] == ev.expected_auc(p, true_risk)
    # A perfect prediction has expected O/E and calibration slope 1.
    perfect = ev.truth_metrics(true_risk[None, :], true_risk)[0]
    assert abs(perfect["oe_true"] - 1) < 1e-15 and abs(perfect["cal_slope_true"] - 1) < 1e-12


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
        return pd.DataFrame({"p_ever": rng.beta(2, 8, size=len(frame)),
                             "slope": rng.normal(0.3, 0.1, size=len(frame))})
    truth = pd.DataFrame({f"cif_{h:g}y": rng.beta(2, 8, size=len(frame)) for h in horizons})
    for h in horizons:
        truth[f"slope_cif_{h:g}y"] = rng.normal(0.3, 0.1, size=len(frame))
    truth["uncensored_event"] = np.where(frame.event == 0, rng.integers(0, 3, size=len(frame)), frame.event)
    truth["uncensored_exit_age"] = frame.entry_age + frame.followup + np.where(frame.event == 0, 1.0, 0.0)
    return truth


@pytest.mark.parametrize("kind", ["binary", "survival"])
def test_every_released_field_is_in_the_registry_and_every_axis_partitions(kind):
    horizons = [1.0, 2.0]
    frame, x = table_frame(6000, 12, kind)
    predictions = predictions_for(frame, x, kind, horizons, logo="logo:ancestry:afr")
    rows = ev.evaluate(kind, frame, predictions, horizons, {"report": {"small_cell_max": 20}},
                       truth=truth_for(frame, kind, horizons),
                       slopes={key: np.full_like(risk, 0.3) for key, risk in predictions.items()})
    assert any("rmse_true" in r and "slope_rmse_true" in r and "auc_true" in r for r in rows)
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
    # Without a report block the AoU rule (counts 1 to 20 withheld) still holds.
    assert status(ev.evaluate("binary", frame, predictions, [], {})) == {ev.INSUFFICIENT}
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


def test_an_event_code_other_than_censored_disease_or_death_is_refused():
    """study-cohort's survival frames carry events 0, 1 and 2 only (4a9bd621): a stray 3, the removed
    exclusion-rule exit, is refused rather than taken as a competing event."""
    horizons = [1.0]
    frame, x = table_frame(3000, 22, "survival")
    predictions = predictions_for(frame, x, "survival", horizons)
    assert any(r["status"] == "ok" for r in ev.evaluate("survival", frame, predictions, horizons, {}))
    three = frame.assign(event=np.where(frame.event == 2, 3, frame.event))
    assert (three.event == 3).sum() > 0
    with pytest.raises(ValueError, match="events 0, 1 or 2"):
        ev.evaluate("survival", three, predictions, horizons, {})


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


def test_the_uncensored_difference_se_counts_every_case_chance_of_censoring():
    """The simulator gate's SEs of released minus uncensored observed risk count every case's chance of censoring,
    given the uncensored outcomes: sum u^2 (1/G - 1) with G known, less with G's estimation. Planted: a cell of
    40 cases that lost none of them to censoring (about 2 expected), as the N2a gate's small cells often do. The
    rows' own differences are then all (w - 1) y >= 0, and their sample SD, the SE of 64074309's gate, puts that
    ordinary cell near z = sqrt(40)."""
    rng = np.random.default_rng(7)
    n, cases, censored, h = 4000, 40, 200, 1.0
    t = np.full(n, 2.0)
    code = np.zeros(n, int)
    t[:cases], code[:cases] = rng.uniform(0.01, 0.99, cases), 1
    t[cases:cases + censored] = rng.uniform(0.01, 0.99, censored)
    entry = rng.uniform(40, 70, n)
    frame = pd.DataFrame({"followup": t, "event": code, "entry_age": entry,
                          **{axis: "all" for axis in ev.AXES["survival"]}})
    p = rng.uniform(0.005, 0.02, n)
    # Truth: the censored rows were event-free past h in the uncensored world.
    t_unc = np.where(code == 1, t, 2.0)
    truth = pd.DataFrame({"cif_1y": p, "uncensored_event": code, "uncensored_exit_age": entry + t_unc})
    rows = ev.evaluate("survival", frame, {("ours", "pooled"): p[:, None]}, [h], {"evaluate": {"censoring": "km"}},
                       truth=truth)
    row = next(r for r in rows if r["stratum"] == "overall" and r["status"] == "ok")
    # G(T-) of each case by the reverse Kaplan-Meier (no ties: the risk set at u is every row with t >= u).
    cuts = np.sort(t[cases:cases + censored])
    at_risk = np.array([(t >= u).sum() for u in cuts])
    g = np.array([np.prod(1 - 1 / at_risk[cuts < T]) for T in t[:cases]])
    expected = np.sqrt(np.sum((1 / n) ** 2 * (1 / g - 1)))
    assert row["risk_unc_diff_se_known_g"] == pytest.approx(expected, rel=1e-10)
    assert row["risk_unc_diff_se"] < row["risk_unc_diff_se_known_g"]
    diff = row["obs_risk"] - row["risk_unc"]
    assert diff == pytest.approx(np.sum(1 / g - 1) / n, rel=1e-10)
    assert abs(diff / row["risk_unc_diff_se_known_g"]) < 2 and abs(diff / row["risk_unc_diff_se"]) < 2
    w = np.zeros(n)
    w[:cases] = 1 / g
    w[cases + censored:] = 1 / np.prod(1 - 1 / at_risk)
    sample = np.sqrt(np.sum(((w * (code == 1) - (code == 1)) / n - diff / n) ** 2))
    assert diff / sample > 4


@pytest.mark.parametrize("kind", ["km", "cox"])
def test_the_uncensored_difference_se_matches_censoring_redrawn_given_the_truth(kind):
    """Given one uncensored world, over 1000 redraws of the censoring alone (site-dependent for the Cox model), the
    released-minus-uncensored observed risk's SE with G's estimation matches the spread of that difference with G
    refitted each time, and the known-G SE the spread of the difference weighted by the true G (the ratio's own
    noise is about 2%)."""
    rng = np.random.default_rng(1)
    n, cases, h = 4000, 200, 1.0
    site = rng.integers(0, 2, n)
    rate = np.where(site == 1, 0.6, 0.2) if kind == "cox" else np.full(n, 0.3)
    t_unc = np.full(n, 2.0)
    t_unc[:cases] = rng.uniform(0.01, 0.99, cases)
    y_unc = (np.arange(n) < cases).astype(float)
    event = y_unc == 1
    decided = np.where(event, t_unc, h)
    covariates = ("ehr_site",) if kind == "cox" else ()
    released, true_g, with_g, known_g = [], [], [], []
    for _ in range(1000):
        c = rng.exponential(1 / rate)
        frame = pd.DataFrame({"followup": np.minimum(t_unc, c), "event_code": np.where(event & (c >= t_unc), 1, 0),
                              "ehr_site": site.astype(str)})
        model = ev.Censoring(frame, h, kind, covariates)
        w = ev.ipcw(frame, h, model)
        y = ((frame.event_code == 1) & (frame.followup <= h)).to_numpy(float)
        released.append(np.mean(w * y) - y_unc.mean())
        observed = (c >= decided).astype(float)
        true_g.append(np.mean(y_unc * (observed / np.exp(-rate * decided) - 1)))
        g = np.where(event, model.at(t_unc, left=True), model.at(h))
        u = y_unc / n
        with_g.append(np.sqrt(model.conditional_variance(np.arange(n), u, g, t_unc, event, h)))
        known_g.append(np.sqrt(np.sum(u ** 2 * (1 / g - 1))))
    ratio_with, ratio_known = np.mean(with_g) / np.std(released, ddof=1), np.mean(known_g) / np.std(true_g, ddof=1)
    print(f"{kind}: SE with G's estimation / SD of the released difference {ratio_with:.3f}; known-G SE / SD of "
          f"the true-G difference {ratio_known:.3f}; known-G / with-G SE {np.mean(known_g) / np.mean(with_g):.3f}")
    assert 0.9 < ratio_with < 1.1 and 0.9 < ratio_known < 1.1
    assert np.mean(with_g) < np.mean(known_g)


@pytest.mark.parametrize("kind", ["binary", "survival"])
@pytest.mark.parametrize("missing", ["ours", "covariates", "standard@logo"])
def test_an_arm_without_predictions_is_omitted_with_its_comparisons_and_nothing_else_changes(kind, missing):
    """A method with no predictions for a fit (study-pipe's time_scale_not_identified: not fitted, so never
    predicted) has no rows for it, and no row carries a paired difference against it; every other row is what
    it is with the arm present. `ours` and `covariates` miss every fit; `standard@logo` misses only its LOGO fit."""
    horizons = [1.0, 2.0]
    logo = "logo:ancestry:afr"
    frame, x = table_frame(6000, 31, kind)
    full = predictions_for(frame, x, kind, horizons, logo=logo)
    variant = missing.split("@")[0]
    drop = {(variant, logo)} if missing.endswith("@logo") else {(variant, "pooled"), (variant, logo)}
    partial = {k: v for k, v in full.items() if k not in drop}
    key = lambda r: (r["variant"], r["fit"], r["stratum"], r["horizon"])
    with_arm = {key(r): r for r in ev.evaluate(kind, frame, full, horizons, {})}
    without = {key(r): r for r in ev.evaluate(kind, frame, partial, horizons, {})}
    assert set(without) == {k for k in with_arm if (k[0], k[1]) not in drop}
    for k, row in without.items():
        against = (variant, k[1]) in drop
        expected = {f: v for f, v in with_arm[k].items()
                    if not (against and (f.startswith(f"d_auc_{variant}") or f.startswith(f"d_brier_{variant}")))}
        assert row.keys() == expected.keys(), k
        for f, value in expected.items():
            if isinstance(value, float):
                assert np.isclose(row[f], value, rtol=1e-12, atol=1e-13, equal_nan=True), (k, f)
            else:
                assert row[f] == value, (k, f)
    assert any(f.startswith(f"d_auc_{variant}") for r in with_arm.values() for f in r) == (variant != "ours")
