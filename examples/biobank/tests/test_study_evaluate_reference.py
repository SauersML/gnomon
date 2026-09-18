"""study/evaluate.py against independent reference implementations (MSI only).

R (STUDY_RSCRIPT, default Rscript): base stats (glm, loess) and survival (survfit Aalen-Johansen, concordance
with Harrell and Uno weights, coxph), plus pROC, timeROC and riskRegression::Score from STUDY_RLIB when they
are installed there. Python: scikit-learn, scikit-survival, lifelines and statsmodels. Each comparison is
repeated on a planted error in our implementation and must fail there; a reference that is not installed
skips its test by name, so a run shows exactly what was verified.

Continuous times must agree to rounding. On tied (weekly) times the reverse Kaplan-Meier conventions differ:
evaluate.reverse_km lets events leave the risk set before censorings at a tied time, R's flipped-status
survfit does not; those comparisons report the difference, and every metric definition is also checked on
the tied data with our own G passed to the reference.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from study import evaluate as ev  # noqa: E402

H = 2.0
N_BINARY, N_SURVIVAL = 4000, 3000


def binary_data(seed=21):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=N_BINARY)
    y = (rng.random(N_BINARY) < 1 / (1 + np.exp(-(-1 + 1.2 * x)))).astype(int)
    p1 = np.round(1 / (1 + np.exp(-(-1.2 + x + rng.normal(scale=0.4, size=N_BINARY)))), 3)  # ties
    p2 = 1 / (1 + np.exp(-(-1 + 0.7 * x + rng.normal(scale=0.6, size=N_BINARY))))
    return pd.DataFrame({"y": y, "p1": np.clip(p1, 1e-3, 1 - 1e-3), "p2": p2})


def survival_data(seed=22, ties=False, competing=True):
    rng = np.random.default_rng(seed)
    n = N_SURVIVAL
    x = rng.normal(size=n)
    site = rng.integers(0, 4, size=n)
    age = rng.uniform(20, 80, size=n)
    t1 = rng.exponential(1 / (0.15 * np.exp(0.7 * x)))
    t2 = rng.exponential(1 / (0.06 * np.exp(0.02 * (age - 50)))) if competing else np.full(n, np.inf)
    c = rng.exponential(1 / (0.25 * np.exp(0.4 * site - 0.01 * (age - 50))))
    t = np.minimum.reduce([t1, t2, c])
    code = np.select([t == t1, t == t2], [1, 2], 0)
    if ties:
        t = np.ceil(t * 52) / 52
    p1 = 1 / (1 + np.exp(-(-1.5 + x + rng.normal(scale=0.5, size=n))))
    p2 = 1 / (1 + np.exp(-(-1.4 + 0.5 * x + rng.normal(scale=0.7, size=n))))
    return pd.DataFrame({"t": t, "code": code, "p1": p1, "p2": p2, "site": site.astype(str), "age": age})


def frame_of(s):
    return s.rename(columns={"t": "followup", "code": "event_code"})


def km_g(s):
    """Our marginal reverse Kaplan-Meier over [0, H]: G(T-) per row and G(H)."""
    model = ev.Censoring(frame_of(s), H, "km")
    return model.at(s.t.to_numpy(), left=True), float(model.at(H)[0])


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """Write every dataset, run the R script once, and return R's values and vectors."""
    directory = tmp_path_factory.mktemp("evaluate_reference")
    b = binary_data()
    b.to_csv(directory / "binary.csv", index=False)
    for kind in ("cont", "tied"):
        s = survival_data(ties=kind == "tied")
        g_left, g_h = km_g(s)
        s.assign(g_left=g_left, g_h=g_h).to_csv(directory / f"surv_{kind}.csv", index=False)
    s = survival_data()
    frame = frame_of(s)
    w = ev.ipcw(frame, H, ev.Censoring(frame, H, "km"))
    known = w > 0
    y = ((s.code == 1) & (s.t <= H)).to_numpy(float)
    pd.DataFrame({"y": y[known], "p": s.p1[known], "w": w[known]}).to_csv(directory / "weighted.csv", index=False)
    X = ev.censoring_design(frame, ("site", "age"))
    cox = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    cox.insert(0, "d", ((s.t <= H) & (s.code == 0)).astype(int))
    cox.insert(0, "t", np.minimum(s.t, H))
    cox.to_csv(directory / "cox.csv", index=False)
    env = dict(os.environ)
    if os.environ.get("STUDY_RLIB"):
        env["R_LIBS_USER"] = os.environ["STUDY_RLIB"]
    result = subprocess.run([os.environ.get("STUDY_RSCRIPT", "Rscript"), str(HERE / "study_evaluate_reference.R"),
                             str(directory), str(H)], capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    values = dict(pd.read_csv(directory / "r_values.csv").itertuples(index=False))
    vectors = {name: pd.read_csv(directory / f"r_{name}.csv") for name in ("loess", "wloess", "gleft", "cox")}
    return values, vectors


def need(values, name):
    if name not in values:
        pytest.skip(f"{name}: its R package is not installed in STUDY_RLIB")
    return values[name]


def fails(error, tolerance):
    """A planted error must move the compared value by more than the comparison's tolerance."""
    assert error > tolerance, f"the planted error moved the value by only {error:.3g} <= {tolerance:.3g}"


# --------------------------------------------------------------------------- #
# binary
# --------------------------------------------------------------------------- #
def test_auc_matches_sklearn_and_proc(ref):
    from sklearn.metrics import roc_auc_score
    b = binary_data()
    aucs, cov = ev.delong(b.y, np.vstack([b.p1, b.p2]))
    reference = roc_auc_score(b.y, b.p1)
    assert abs(aucs[0] - reference) < 1e-12
    jittered = ev.delong(b.y, (b.p1.to_numpy() + 1e-9 * np.random.default_rng(0).normal(size=len(b)))[None, :])[0][0]
    fails(abs(jittered - reference), 1e-12)
    values, _ = ref
    assert abs(aucs[0] - need(values, "auc")) < 1e-12


def test_delong_variance_and_paired_difference_match_proc(ref):
    values, _ = ref
    b = binary_data()
    aucs, cov = ev.delong(b.y, np.vstack([b.p1, b.p2]))
    assert abs(cov[0, 0] - need(values, "auc_var")) < 1e-12 * max(1.0, cov[0, 0])
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    assert abs((aucs[0] - aucs[1]) - values["d_auc"]) < 1e-12
    assert abs((aucs[0] - aucs[1]) / se - values["d_auc_z"]) < 1e-9
    # Planted: the paired SE without the covariance term.
    fails(abs((aucs[0] - aucs[1]) / np.sqrt(cov[0, 0] + cov[1, 1]) - values["d_auc_z"]), 1e-9)


def test_brier_matches_sklearn():
    from sklearn.metrics import brier_score_loss
    b = binary_data()
    rows = ev.binary_cell(b.y.to_numpy(), np.vstack([b.p1, b.p2]), ["ours", "standard"], "pooled", "overall", 21)
    assert abs(rows[0]["brier"] - brier_score_loss(b.y, b.p1)) < 1e-15
    fails(abs(np.mean(np.abs(b.y - b.p1)) - brier_score_loss(b.y, b.p1)), 1e-15)


def test_calibration_intercept_and_slope_match_r_glm_and_statsmodels(ref):
    import statsmodels.api as sm
    values, _ = ref
    b = binary_data()
    cal = ev.calibration(b.y, b.p1)
    for name in ("cal_int", "cal_int_se", "cal_slope", "cal_slope_se"):
        assert abs(cal[name] - values[name]) < 1e-8 * max(1.0, abs(values[name])), name
    lp = ev.clipped_logit(b.p1)
    slope = sm.GLM(b.y, sm.add_constant(lp), family=sm.families.Binomial()).fit(tol=1e-14)
    assert abs(cal["cal_slope"] - slope.params[1]) < 1e-8
    # Planted: the intercept of the joint fit is not the calibration intercept (slope fixed at one).
    joint = ev.logistic_fit(np.column_stack([np.ones(len(lp)), lp]), b.y)[0][0]
    fails(abs(joint - values["cal_int"]), 1e-8)


def test_weighted_calibration_matches_r_and_its_sandwich_matches_statsmodels(ref):
    import statsmodels.api as sm
    values, _ = ref
    s = survival_data()
    frame = frame_of(s)
    w = ev.ipcw(frame, H, ev.Censoring(frame, H, "km"))
    known = w > 0
    y = ((s.code == 1) & (s.t <= H)).to_numpy(float)[known]
    p, wk = s.p1.to_numpy()[known], w[known]
    cal = ev.calibration(y, p, wk, robust=True)
    assert abs(cal["cal_int"] - values["wcal_int"]) < 1e-8
    assert abs(cal["cal_slope"] - values["wcal_slope"]) < 1e-8
    lp = ev.clipped_logit(p)
    fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial(), var_weights=wk).fit(tol=1e-14,
                                                                                             cov_type="HC0")
    assert abs(cal["cal_slope_se"] - fit.bse[1]) < 1e-6 * fit.bse[1]
    # Planted: the model-based SE, which treats IPCW weights as frequencies.
    fails(abs(ev.calibration(y, p, wk)["cal_slope_se"] - fit.bse[1]), 1e-6 * fit.bse[1])


def test_loess_matches_r_loess_direct_and_the_ici_grid_is_close(ref):
    values, vectors = ref
    b = binary_data()
    fitted = ev.loess(b.p1.to_numpy(), b.y.to_numpy(float), b.p1.to_numpy())
    r_fit = vectors["loess"].loess_fit.to_numpy()
    assert np.max(np.abs(fitted - r_fit)) < 1e-8
    planted = ev.loess(b.p1.to_numpy(), b.y.to_numpy(float), b.p1.to_numpy(), span=0.7)
    fails(np.max(np.abs(planted - r_fit)), 1e-8)
    # The ICI fits the smooth on a grid of 201 quantiles and interpolates.
    ici = ev.integrated_calibration_index(b.y.to_numpy(float), b.p1.to_numpy())
    print(f"ici grid {ici:.8f} R direct {values['ici']:.8f} difference {ici - values['ici']:.2e}")
    assert abs(ici - values["ici"]) < 1e-4


def test_weighted_loess_matches_r(ref):
    _, vectors = ref
    s = survival_data()
    frame = frame_of(s)
    w = ev.ipcw(frame, H, ev.Censoring(frame, H, "km"))
    known = w > 0
    y = ((s.code == 1) & (s.t <= H)).to_numpy(float)[known]
    p = s.p1.to_numpy()[known]
    fitted = ev.loess(p, y, p, w[known])
    r_fit = vectors["wloess"].wloess_fit.to_numpy()
    assert np.max(np.abs(fitted - r_fit)) < 1e-8
    fails(np.max(np.abs(ev.loess(p, y, p) - r_fit)), 1e-8)


# --------------------------------------------------------------------------- #
# survival
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["cont", "tied"])
def test_aalen_johansen_matches_r_survfit(ref, kind):
    values, _ = ref
    s = survival_data(ties=kind == "tied")
    cif, se = ev.aalen_johansen(s.t, s.code, H)
    assert abs(cif - values[f"aj_{kind}"]) < 1e-12
    assert abs(se - values[f"aj_se_{kind}"]) < 1e-9 * values[f"aj_se_{kind}"]
    # Planted: 1 - KM with deaths censored, and the binomial SE that ignores censoring.
    times, km, _ = ev.reverse_km(s.t, np.where(s.code == 1, 0, 1))
    fails(abs(1 - float(ev.step_at(times, km, H)) - values[f"aj_{kind}"]), 1e-12)
    fails(abs(np.sqrt(cif * (1 - cif) / len(s)) - values[f"aj_se_{kind}"]), 1e-9 * values[f"aj_se_{kind}"])


def test_aalen_johansen_matches_lifelines():
    from lifelines import AalenJohansenFitter
    s = survival_data()
    fitter = AalenJohansenFitter(calculate_variance=False).fit(s.t, s.code, event_of_interest=1)
    curve = fitter.cumulative_density_
    reference = float(curve.loc[curve.index <= H].iloc[-1, 0])
    assert abs(ev.aalen_johansen(s.t, s.code, H)[0] - reference) < 1e-10


@pytest.mark.parametrize("kind", ["cont", "tied"])
def test_harrell_wolbers_concordance_matches_r_and_sksurv(ref, kind):
    from sksurv.metrics import concordance_index_censored
    values, _ = ref
    s = survival_data(ties=kind == "tied")
    t, code, p = s.t.to_numpy(), s.code.to_numpy(), s.p1.to_numpy()
    c = ev.wolbers_concordance(t, code, p, H)
    assert abs(c - values[f"c_harrell_{kind}"]) < 1e-12
    # scikit-survival's Harrell on the same recoding: deaths censored after every time, truncation at H.
    tt = np.where(code == 2, t.max() + 1, np.where(t > H, H, t))
    event = (code == 1) & (t <= H)
    if kind == "cont":
        assert abs(c - concordance_index_censored(event, tt, p)[0]) < 1e-12
    fails(abs(ev.wolbers_concordance(t, np.where(code == 2, 0, code), p, H) - values[f"c_harrell_{kind}"]), 1e-12)


@pytest.mark.parametrize("kind", ["cont", "tied"])
def test_single_event_uno_matches_r_and_sksurv(ref, kind):
    from sksurv.metrics import concordance_index_ipcw
    from sksurv.util import Surv
    values, _ = ref
    s = survival_data(ties=kind == "tied")
    t, p = s.t.to_numpy(), s.p1.to_numpy()
    code = np.where(s.code == 2, 0, s.code)  # deaths are censorings in the single-event C
    marginal = ev.Censoring(pd.DataFrame({"followup": t, "event_code": code}), H, "km")
    harrell = ev.wolbers_concordance(t, code, p, H)
    uno = ev.wolbers_concordance(t, code, p, H, marginal)
    assert abs(harrell - values[f"c_single_harrell_{kind}"]) < 1e-12
    print(f"uno {kind}: ours {uno:.10f} R n/G2 {values[f'c_single_uno_{kind}']:.10f}")
    if kind == "cont":
        assert abs(uno - values[f"c_single_uno_{kind}"]) < 1e-9
        y = Surv.from_arrays(code == 1, t)
        assert abs(uno - concordance_index_ipcw(y, y, p, tau=H)[0]) < 1e-9
        fails(abs(harrell - values[f"c_single_uno_{kind}"]), 1e-9)
    else:
        assert abs(uno - values[f"c_single_uno_{kind}"]) < 1e-3


@pytest.mark.parametrize("kind", ["cont", "tied"])
def test_reverse_km_matches_r(ref, kind):
    values, vectors = ref
    s = survival_data(ties=kind == "tied")
    g_left, g_h = km_g(s)
    difference = max(abs(g_h - values[f"g_h_{kind}"]),
                     np.max(np.abs(g_left - vectors["gleft"][f"g_left_{kind}"].to_numpy())))
    print(f"reverse KM {kind}: max |ours - R| = {difference:.3g}")
    # On tied times events leave the risk set first here, not in R's flipped-status survfit.
    assert difference < (1e-12 if kind == "cont" else 1e-2)


@pytest.mark.parametrize("kind", ["cont", "tied"])
def test_competing_risk_ipcw_auc_and_brier_match_the_definitions_in_r(ref, kind):
    values, _ = ref
    s = survival_data(ties=kind == "tied")
    frame = frame_of(s)
    w = ev.ipcw(frame, H, ev.Censoring(frame, H, "km"))
    y = ((s.code == 1) & (s.t <= H)).to_numpy(float)
    auc = ev.weighted_auc(s.p1, y, w)[0]
    brier = float(np.mean(w * (y - s.p1) ** 2))
    assert abs(auc - values[f"brute_auc_{kind}"]) < 1e-12
    assert abs(brier - values[f"brute_brier_{kind}"]) < 1e-12
    known = w > 0
    fails(abs(ev.weighted_auc(s.p1[known], y[known], np.ones(known.sum()))[0] - values[f"brute_auc_{kind}"]),
          1e-12)
    fails(abs(np.mean((y - s.p1)[known] ** 2) - values[f"brute_brier_{kind}"]), 1e-12)


def test_single_event_ipcw_auc_and_brier_match_sksurv():
    from sksurv.metrics import brier_score, cumulative_dynamic_auc
    from sksurv.util import Surv
    s = survival_data(competing=False)
    frame = frame_of(s)
    w = ev.ipcw(frame, H, ev.Censoring(frame, H, "km"))
    y = ((s.code == 1) & (s.t <= H)).to_numpy(float)
    ys = Surv.from_arrays(s.code == 1, s.t)
    auc = ev.weighted_auc(s.p1, y, w)[0]
    reference = float(cumulative_dynamic_auc(ys, ys, s.p1.to_numpy(), [H])[0][0])
    assert abs(auc - reference) < 1e-10
    brier = float(np.mean(w * (y - s.p1) ** 2))
    reference_brier = float(brier_score(ys, ys, (1 - s.p1.to_numpy())[:, None], [H])[1][0])
    assert abs(brier - reference_brier) < 1e-10
    fails(abs(float(np.mean((y - s.p1)[w > 0] ** 2)) - reference_brier), 1e-10)


@pytest.mark.parametrize("kind", ["cont", "tied"])
def test_competing_risk_auc_and_se_match_timeroc_and_riskregression(ref, kind):
    values, _ = ref
    s = survival_data(ties=kind == "tied")
    frame = frame_of(s)
    w = ev.ipcw(frame, H, ev.Censoring(frame, H, "km"))
    y = ((s.code == 1) & (s.t <= H)).to_numpy(float)
    auc, influence = ev.weighted_auc(s.p1, y, w)
    reference = need(values, f"timeroc_auc2_{kind}")
    tolerance = 1e-10 if kind == "cont" else 1e-3
    assert abs(auc - reference) < tolerance
    se = np.sqrt(np.sum(influence ** 2)) / len(s)
    print(f"IPCW AUC SE {kind}: ours (fixed G) {se:.6f} timeROC iid {values[f'timeroc_auc2_se_{kind}']:.6f}")
    assert abs(se / values[f"timeroc_auc2_se_{kind}"] - 1) < 0.1
    if f"rr_auc_{kind}" in values:
        assert abs(auc - values[f"rr_auc_{kind}"]) < tolerance
        assert abs(float(np.mean(w * (y - s.p1) ** 2)) - values[f"rr_brier_{kind}"]) < tolerance


@pytest.mark.parametrize("ridge", [0, 1])
def test_cox_censoring_model_matches_r_coxph(ref, ridge):
    _, vectors = ref
    s = survival_data()
    model = ev.Censoring(frame_of(s), H, "cox", ("site", "age"), ridge=float(ridge))
    g = model.at(H)
    reference = vectors["cox"][f"cox_g_{ridge}"].to_numpy()
    assert np.max(np.abs(g - reference)) < 1e-8
    fails(np.max(np.abs(ev.Censoring(frame_of(s), H, "km").at(H) - reference)), 1e-8)


# --------------------------------------------------------------------------- #
# standard errors and the IPCW estimator gate, by simulation
# --------------------------------------------------------------------------- #
def gate_sample(rng, n):
    """Site drives both the disease hazard and loss to follow-up; given site, censoring is independent."""
    site = rng.integers(0, 4, size=n)
    ancestry = np.where(rng.random(n) < 0.5 + 0.1 * site, "a", "b")
    x = rng.normal(size=n)
    t1 = rng.exponential(1 / (0.08 * np.exp(0.8 * x + 0.5 * site)))
    t2 = rng.exponential(1 / 0.04, size=n)
    c = rng.exponential(1 / (0.05 * np.exp(0.9 * site)))
    t_full = np.minimum(t1, t2)
    code_full = np.where(t1 < t2, 1, 2)
    t = np.minimum(t_full, c)
    code = np.where(c < t_full, 0, code_full)
    p = 1 / (1 + np.exp(-(-2.5 + 0.8 * x + 0.5 * site)))
    q = 1 / (1 + np.exp(-(-2.3 + 0.4 * x + 0.5 * site)))
    frame = pd.DataFrame({"followup": t, "event_code": code, "ehr_site": site.astype(str), "ancestry": ancestry})
    return frame, t_full, code_full, p, q


def ipcw_metrics(frame, p, q, model):
    w = ev.ipcw(frame, H, model)
    y = ((frame.event_code == 1) & (frame.followup <= H)).to_numpy(float)
    auc_p, inf_p = ev.weighted_auc(p, y, w)
    auc_q, inf_q = ev.weighted_auc(q, y, w)
    n = len(y)
    return {"auc": auc_p, "auc_se": np.sqrt(np.sum(inf_p ** 2)) / n, "d_auc": auc_p - auc_q,
            "d_auc_se": np.sqrt(np.sum((inf_p - inf_q) ** 2)) / n, "brier": float(np.mean(w * (y - p) ** 2)),
            "risk": float(np.mean(w * y))}


def truth_metrics(t_full, code_full, p, q):
    y = ((code_full == 1) & (t_full <= H)).astype(float)
    ones = np.ones(len(y))
    return {"auc": ev.weighted_auc(p, y, ones)[0], "d_auc": ev.weighted_auc(p, y, ones)[0]
            - ev.weighted_auc(q, y, ones)[0], "brier": float(np.mean((y - p) ** 2)), "risk": float(y.mean())}


def test_ipcw_estimator_gate_and_se_calibration_under_site_dependent_censoring():
    """SPEC section 8 (audit N2a): under independent, site-dependent censoring, every IPCW metric agrees with the
    same metric on the uncensored outcomes (|mean difference| < 2 SE over replicate seeds), and the influence
    standard errors match the replicate spread. The planted ancestry-only G must fire the gate."""
    rng = np.random.default_rng(31)
    reps, n = 60, 3000
    diffs = {"cox": [], "ancestry_only": []}
    spread = []
    for _ in range(reps):
        frame, t_full, code_full, p, q = gate_sample(rng, n)
        truth = truth_metrics(t_full, code_full, p, q)
        for name, model in (("cox", ev.Censoring(frame, H, "cox", ("ehr_site",))),
                            ("ancestry_only", ev.Censoring(frame, H, "strata", ("ancestry",)))):
            m = ipcw_metrics(frame, p, q, model)
            diffs[name].append({k: m[k] - truth[k] for k in truth})
            if name == "cox":
                spread.append(m)
    for name, expect_fire in (("cox", False), ("ancestry_only", True)):
        table = pd.DataFrame(diffs[name])
        z = table.mean() / (table.std(ddof=1) / np.sqrt(reps))
        print(f"gate {name}: mean IPCW - truth {table.mean().to_dict()} z {z.to_dict()}")
        fired = bool((z.abs() > 2).any())
        assert fired == expect_fire, name
    # The influence SEs (conditional on G) against the replicate standard deviations of the estimates.
    spread = pd.DataFrame(spread)
    for metric in ("auc", "d_auc"):
        ratio = spread[f"{metric}_se"].mean() / spread[metric].std(ddof=1)
        print(f"{metric}: mean SE / replicate SD = {ratio:.3f}")
        assert 0.8 < ratio < 1.25, metric
        planted = (spread[f"{metric}_se"] / 2).mean() / spread[metric].std(ddof=1)
        assert not 0.8 < planted < 1.25


def test_binary_oe_interval_and_brier_standard_errors_cover_the_truth():
    """No package computes these intervals, so their reference is the known truth: over replicate cells of
    calibrated predictions, the O/E interval covers 1 and the Brier and paired-Brier intervals cover their
    population values at about 95%. A planted half-width SE must break the coverage."""
    rng = np.random.default_rng(41)
    reps, n = 400, 3000
    covered = {"oe": 0, "brier": 0, "d_brier": 0, "oe_half": 0}
    # Population Brier values of the two predictors, from one large draw of the same law.
    big = rng.normal(size=2_000_000)
    p_big = 1 / (1 + np.exp(-(-1.5 + big)))
    q_big = 1 / (1 + np.exp(-(-1.5 + 0.6 * big)))
    brier_true = float(np.mean(p_big * (1 - p_big)))
    d_true = float(np.mean(p_big * (1 - p_big) - (p_big * (1 - q_big) ** 2 + (1 - p_big) * q_big ** 2)))
    for _ in range(reps):
        x = rng.normal(size=n)
        p = 1 / (1 + np.exp(-(-1.5 + x)))
        q = 1 / (1 + np.exp(-(-1.5 + 0.6 * x)))
        y = (rng.random(n) < p).astype(int)
        row = ev.binary_cell(y, np.vstack([p, q]), ["ours", "standard"], "pooled", "overall", 21)[0]
        covered["oe"] += row["oe_lo"] <= 1 <= row["oe_hi"]
        half = (row["oe_hi"] - row["oe_lo"]) / 4
        covered["oe_half"] += row["oe"] - half <= 1 <= row["oe"] + half
        covered["brier"] += abs(row["brier"] - brier_true) <= ev.Z95 * row["brier_se"]
        covered["d_brier"] += abs(row["d_brier_standard"] - d_true) <= ev.Z95 * row["d_brier_standard_se"]
    rates = {k: v / reps for k, v in covered.items()}
    print(f"coverage over {reps} cells: {rates}")
    for name in ("oe", "brier", "d_brier"):
        assert 0.92 <= rates[name] <= 0.985, name
    assert rates["oe_half"] < 0.85
