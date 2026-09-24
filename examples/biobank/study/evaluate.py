"""Held-out metrics for the study's single results table.

One row per (disease, model, variant, fit, stratum, horizon), the digest's KEYS; the metrics a row can carry
are the registry METRICS, and the layout is written up in the study's TABLE.md. Every metric here is checked
against an independent reference implementation by tests/test_study_evaluate_reference.py (R survival/stats
and pROC, timeROC and riskRegression where installed; lifelines, scikit-survival, scikit-learn and statsmodels),
each with a planted error that must make it fail.

Binary cells: AUC with its DeLong standard error, Brier, O/E, the calibration intercept (slope fixed at one) and
slope, the integrated calibration index, and paired AUC and Brier differences against the covariates-only and
standard competitors.

Survival cells, at each horizon h on time since entry, with death a competing event:
- the IPCW AUC with controls including deaths (Blanche definition 2);
- the IPCW Brier score, a death before h being a known non-case weighted 1/G(T-);
- the observed risk, the IPCW incidence (the Aalen-Johansen estimator when G is the cell's reverse KM), and
  O/E against the mean predicted CIF;
- the IPCW calibration intercept, slope and integrated calibration index;
- Wolbers' competing-risk concordance truncated at h, unweighted (Harrell) and IPCW (Uno);
- paired AUC and Brier differences with influence-function standard errors.

G, the censoring survival, is an evaluation nuisance fitted on the evaluation rows themselves: a Cox model for
loss to follow-up over [0, h] on the rows whose potential follow-up reaches h, given site, region, ancestry,
entry age and entry year. Deaths and disease events end follow-up and are not censorings. Every IPCW standard
error (AUC, Brier, observed risk, calibration intercept and slope, and the paired differences) includes G's own
estimation (the Cox model's score residuals and the Breslow or Kaplan-Meier hazard's martingale). The
simulator-only SEs of each released-minus-uncensored difference are instead the censoring's variance given the
uncensored outcomes, with G's estimation (and, in the _known_g columns, without it: conservative).

Release (SPEC section 3, 17:50Z): a cell's metrics are released only when its development rows, scaled to its
test rows, hold at least RELEASE_MARGIN x the smallest releasable count of cases and of known non-cases, and its
test counts pass the floor; a decision on the test counts alone would select cells on their own outcomes and bias
every released value upward (the winner's curse, +4-10% in 5 simulated worlds). A withheld cell carries only its
test counts and insufficient_support, whichever side withheld it.

Ties at the horizon: an event at exactly h is a case by h, as in the cumulative incidence F(h) = P(T <= h);
a row followed past h is a known non-case and one censored at exactly h is unknown. (timeROC counts cases T < t
strictly, so the two differ only where events fall exactly on the horizon.)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

Z95 = 1.959963984540054
ONE_SIDED_95 = 1.6448536269514722
# The AoU dissemination rule: no released count, and no count derivable from released ones, in 1..20. A cell
# needs more than config["report"]["small_cell_max"] (default this) cases, non-cases and rows.
SMALL_CELL_MAX = 20
# The release rule (SPEC section 3, 17:50Z): a cell's metrics are released only when its development rows at the
# horizon, scaled to its test rows, hold at least this many times the smallest releasable count of cases and of
# known non-cases. The decision reads no test outcome, so the released values carry no winner's curse; the test
# counts must still pass the floor, which at this margin almost never decides (0 of 2,115 cells in 5 simulated
# worlds; at 1.5x it decided 14).
RELEASE_MARGIN = 2
# A censoring-model category level with fewer rows is merged with the other small levels.
CENSORING_LEVEL_MIN = 20
# The prespecified support rule for a horizon in a cell: the one-sided 95% upper bound on the cell's own
# reverse Kaplan-Meier G(h) at or above this floor, which caps its IPCW weights near 1/floor.
POSITIVITY_FLOOR = 0.05
# The integrated calibration index smooths with R's loess defaults (span 0.75, local quadratic, tricube) fitted
# exactly at this many quantiles of the predictions and interpolated linearly between them.
LOESS_SPAN, LOESS_GRID = 0.75, 201
# Predictions are clipped this far from 0 and 1 before any logit.
PROBABILITY_CLIP = 1e-12


class MetricRefusal(ValueError):
    """A metric that has no stable value in a cell, named by the refusal's first argument."""


# --------------------------------------------------------------------------- #
# ranks, AUC and DeLong
# --------------------------------------------------------------------------- #
def midrank(x):
    order = np.argsort(x, kind="mergesort")
    xs = np.asarray(x, float)[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(xs)) + 1])
    ends = np.concatenate([starts[1:], [len(xs)]])
    ranks = np.empty(len(xs))
    ranks[order] = np.repeat((starts + ends - 1) / 2 + 1, ends - starts)
    return ranks


def delong(y, predictions):
    """AUCs of paired predictions and their DeLong covariance, by Sun and Xu's midrank algorithm (2014)."""
    y = np.asarray(y).astype(bool)
    predictions = np.atleast_2d(np.asarray(predictions, float))
    positive, negative = predictions[:, y], predictions[:, ~y]
    m, n = positive.shape[1], negative.shape[1]
    if m < 2 or n < 2:
        raise MetricRefusal("auc", "DeLong needs at least two cases and two controls")
    tx = np.vstack([midrank(row) for row in positive])
    ty = np.vstack([midrank(row) for row in negative])
    tz = np.vstack([midrank(np.concatenate([a, b])) for a, b in zip(positive, negative)])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    covariance = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, covariance


def weighted_auc(score, target, weights):
    """The weighted probability that a case outranks a control, ties counting half, and its influence
    function: one value per row, zero for rows of zero weight, with Var(AUC) ~ sum(IF**2) / n**2 given the
    weights. With unit weights the variance is DeLong's with m and n in place of m - 1 and n - 1."""
    score, target, weights = (np.asarray(v, dtype=float) for v in (score, target, weights))
    case_w, ctrl_w = weights * target, weights * (1 - target)
    total_case, total_ctrl = case_w.sum(), ctrl_w.sum()
    if total_case <= 0 or total_ctrl <= 0:
        raise MetricRefusal("auc", "AUC needs weighted cases and controls")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    starts = np.concatenate([[0], np.flatnonzero(np.diff(s)) + 1])
    group = np.repeat(np.arange(len(starts)), np.diff(np.concatenate([starts, [len(s)]])))
    case_group, ctrl_group = np.add.reduceat(case_w[order], starts), np.add.reduceat(ctrl_w[order], starts)
    ctrl_below = np.concatenate([[0.0], np.cumsum(ctrl_group)])[:-1]
    case_above = total_case - np.cumsum(case_group)
    auc = float(np.sum(case_group * (ctrl_below + 0.5 * ctrl_group)) / (total_case * total_ctrl))
    v10 = np.empty(len(s))
    v01 = np.empty(len(s))
    v10[order] = (ctrl_below + 0.5 * ctrl_group)[group] / total_ctrl
    v01[order] = (case_above + 0.5 * case_group)[group] / total_case
    n = len(score)
    influence = n * (case_w * (v10 - auc) / total_case + ctrl_w * (v01 - auc) / total_ctrl)
    return auc, influence


# --------------------------------------------------------------------------- #
# logistic calibration and the integrated calibration index
# --------------------------------------------------------------------------- #
def clipped_logit(p):
    p = np.clip(np.asarray(p, float), PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return np.log(p) - np.log1p(-p)


def expit(x):
    return np.exp(-np.logaddexp(0.0, -x))


def logistic_fit(X, y, weights=None, offset=None, iterations=100):
    """Weighted Newton-Raphson logistic regression with an optional offset: the coefficients, their
    model-based covariance (R glm's for unit weights), their sandwich covariance (HC0), which is the one to use
    when the weights are IPCW weights rather than frequencies, and each row's influence (rows x coefficients,
    summing to beta's error), which is also beta's derivative in the row's log weight."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, float)
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    beta = np.zeros(X.shape[1])

    def loglik(b):
        eta = off + X @ b
        return float(np.sum(w * (y * eta - np.logaddexp(0.0, eta))))

    current = loglik(beta)
    for _ in range(iterations):
        p = expit(off + X @ beta)
        information = X.T @ (X * (w * p * (1 - p))[:, None])
        try:
            step = np.linalg.solve(information, X.T @ (w * (y - p)))
        except np.linalg.LinAlgError as error:
            raise MetricRefusal("calibration", "singular logistic information") from error
        scale = 1.0
        while True:
            candidate = beta + scale * step
            value = loglik(candidate)
            if value >= current - 1e-12 * (1 + abs(current)) or scale < 1e-6:
                break
            scale /= 2
        beta, previous, current = candidate, current, value
        if np.max(np.abs(scale * step)) < 1e-10 or abs(current - previous) < 1e-12 * (1 + abs(current)):
            break
    else:
        raise MetricRefusal("calibration", "logistic calibration did not converge")
    p = expit(off + X @ beta)
    information = X.T @ (X * (w * p * (1 - p))[:, None])
    bread = np.linalg.inv(information)
    influence = (X * (w * (y - p))[:, None]) @ bread
    return beta, bread, influence.T @ influence, influence


def calibration(target, p, weights=None, robust=False, censoring=None, rows=None):
    """Calibration intercept (logit P(y) = a + logit p, slope fixed at one) and slope (logit P(y) = a + b logit p),
    each with its standard error: model-based, or the HC0 sandwich with robust=True. With IPCW weights from
    `censoring`, and `rows` these rows' indices among its evaluation rows, the SE also carries the censoring
    model's own estimation (Censoring.variance)."""
    lp = clipped_logit(p)
    ones = np.ones((len(lp), 1))
    a, model_a, robust_a, influence_a = logistic_fit(ones, target, weights, offset=lp)
    b, model_b, robust_b, influence_b = logistic_fit(np.column_stack([ones, lp]), target, weights)
    if censoring is not None:
        se_a = math.sqrt(censoring.variance(rows, influence_a[:, 0], influence_a[:, 0]))
        se_b = math.sqrt(censoring.variance(rows, influence_b[:, 1], influence_b[:, 1]))
    else:
        cov_a, cov_b = (robust_a, robust_b) if robust else (model_a, model_b)
        se_a, se_b = math.sqrt(cov_a[0, 0]), math.sqrt(cov_b[1, 1])
    return {"cal_int": float(a[0]), "cal_int_se": float(se_a), "cal_slope": float(b[1]), "cal_slope_se": float(se_b)}


def loess(x, y, at, weights=None, span=LOESS_SPAN):
    """R's loess(y ~ x, span, degree = 2, family = "gaussian", surface = "direct") evaluated at `at`: at each
    point a weighted local quadratic over the floor(n * span) nearest x, with tricube weights on the distance
    over the q-th nearest distance, times the prior weights."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, float)
    order = np.argsort(x, kind="mergesort")
    x, y, w = x[order], y[order], w[order]
    n, q = len(x), int(math.floor(len(x) * min(span, 1.0)))
    if q < 3:
        raise MetricRefusal("ici", "loess needs at least three neighbours")
    at = np.asarray(at, float)
    # The q nearest points form a window [lo, lo + q) of the sorted x: bisect for the first lo from which
    # moving the window right no longer brings a nearer point in.
    low = np.clip(np.searchsorted(x, at) - q, 0, n - q)
    high = np.clip(np.searchsorted(x, at), 0, n - q)
    while np.any(low < high):
        mid = (low + high) // 2
        right = x[np.minimum(mid + q, n - 1)]
        move = (mid + q < n) & (at - x[mid] > right - at)
        low, high = np.where(move, mid + 1, low), np.where(move, high, mid)
    radius = np.maximum(at - x[low], x[low + q - 1] - at) * max(span, 1.0)
    if (radius <= 0).any():
        raise MetricRefusal("ici", "loess neighbourhood has zero width")
    fitted = np.empty(len(at))
    for k, (x0, r, lo) in enumerate(zip(at, radius, low)):
        # The weighted local quadratic in u = (x - x0) / r over the window (points at distance r weigh 0).
        u = (x[lo:lo + q] - x0) / r
        u2 = u * u
        local = w[lo:lo + q] * np.clip(1 - np.abs(u) * u2, 0, None) ** 3
        lu, lu2, yy = local * u, local * u2, y[lo:lo + q]
        s0, s1, s2, s3, s4 = local.sum(), lu.sum(), lu2.sum(), lu2 @ u, lu2 @ u2
        gram = np.array([[s0, s1, s2], [s1, s2, s3], [s2, s3, s4]])
        try:
            fitted[k] = np.linalg.solve(gram, np.array([local @ yy, lu @ yy, lu2 @ yy]))[0]
        except np.linalg.LinAlgError as error:
            raise MetricRefusal("ici", "singular local quadratic") from error
    return fitted


def integrated_calibration_index(target, p, weights=None, grid=LOESS_GRID):
    """Austin and Steyerberg's ICI: the mean over the cell of |p - loess(y ~ p)(p)|. The smooth is fitted on
    the rows of positive weight (all rows when weights is None) and averaged over every row of the cell."""
    p = np.asarray(p, float)
    target = np.asarray(target, float)
    known = np.ones(len(p), bool) if weights is None else np.asarray(weights, float) > 0
    w = None if weights is None else np.asarray(weights, float)[known]
    at = np.unique(np.quantile(p, np.linspace(0, 1, grid)))
    curve = loess(p[known], target[known], at, w)
    return float(np.mean(np.abs(p - np.interp(p, at, curve))))


# --------------------------------------------------------------------------- #
# censoring survival G
# --------------------------------------------------------------------------- #
def reverse_km(time, code):
    """The reverse Kaplan-Meier of censoring (code 0): distinct times, G at each (right-continuous) and the
    Greenwood variance of log G. At a time with both, events leave the risk set before censorings."""
    t, code = np.asarray(time, float), np.asarray(code, int)
    times, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    censored = np.bincount(inverse, weights=code == 0, minlength=len(times))
    at_risk = np.cumsum(counts[::-1])[::-1] - (counts - censored)
    hazard = np.divide(censored, at_risk, out=np.zeros_like(censored), where=at_risk > 0)
    remaining = at_risk - censored
    greenwood = np.cumsum(np.divide(censored, at_risk * remaining, out=np.zeros_like(censored),
                                    where=(at_risk > 0) & (remaining > 0)))
    return times, np.cumprod(1 - hazard), greenwood


def step_at(times, values, t, left=False, start=1.0):
    """A right-continuous step function at t, or its left limit with left=True."""
    return np.r_[start, values][np.searchsorted(times, t, side="left" if left else "right")]


def censoring_window(time, code, horizon):
    """Follow-up over [0, horizon] for the censoring process: (time, censored). A row followed past the horizon
    is held just after it, so it stays at risk of censoring at the horizon itself."""
    t = np.asarray(time, float)
    return np.where(t > horizon, np.nextafter(horizon, np.inf), t), (t <= horizon) & (np.asarray(code) == 0)


def cox_fit(time, event, X, ridge=1.0, iterations=50):
    """Breslow-tie Cox partial likelihood with the penalty ridge/2 |beta|^2 on centred covariates (R:
    coxph(ties = "breslow") with ridge(..., theta = ridge, scale = FALSE)), by damped Newton-Raphson.
    Returns beta, the distinct event times, the Breslow cumulative baseline hazard at the covariate means and
    the penalised information at beta."""
    t, d, X = np.asarray(time, float), np.asarray(event, float), np.asarray(X, float)
    order = np.argsort(t, kind="mergesort")
    t, d = t[order], d[order]
    X = X[order] - X[order].mean(axis=0)
    times, first, counts = np.unique(t, return_index=True, return_counts=True)
    deaths = np.add.reduceat(d, first)
    events = deaths > 0
    row_time = np.repeat(np.arange(len(times)), counts)
    x_events = d @ X

    def pieces(b):
        # Risk-set sums at each distinct time, on a shifted scale that cancels in every ratio.
        eta = X @ b
        shift = eta.max()
        r = np.exp(eta - shift)
        s0 = np.cumsum(r[::-1])[::-1][first]
        s1 = np.cumsum((X * r[:, None])[::-1], axis=0)[::-1][first]
        value = float(d @ eta - deaths[events] @ (np.log(s0[events]) + shift) - 0.5 * ridge * b @ b)
        return r, s0, s1, shift, value

    def information_at(r, s0, s1):
        m = s1[events] / s0[events, None]
        cumulative = np.cumsum(np.where(events, deaths / s0, 0.0))[row_time]
        return m, (X.T @ (X * (r * cumulative)[:, None]) - m.T @ (m * deaths[events, None])
                   + ridge * np.eye(X.shape[1]))

    beta = np.zeros(X.shape[1])
    r, s0, s1, shift, current = pieces(beta)
    for _ in range(iterations if X.shape[1] else 0):
        m, information = information_at(r, s0, s1)
        gradient = x_events - deaths[events] @ m - ridge * beta
        step = np.linalg.solve(information, gradient)
        scale = 1.0
        while True:
            candidate = pieces(beta + scale * step)
            if candidate[-1] >= current - 1e-12 * (1 + abs(current)) or scale < 1e-6:
                break
            scale /= 2
        beta = beta + scale * step
        previous, (r, s0, s1, shift, current) = current, candidate
        if np.max(np.abs(scale * step)) < 1e-9 or abs(current - previous) < 1e-13 * (1 + abs(current)):
            break
    else:
        if X.shape[1]:
            raise MetricRefusal("censoring", "Cox censoring model did not converge")
    return (beta, times[events], np.cumsum(deaths[events] / (s0[events] * np.exp(shift))),
            information_at(r, s0, s1)[1])


def censoring_design(frame, covariates, minimum=CENSORING_LEVEL_MIN, categorical_levels=12):
    """The censoring model's design: one-hot categories with the first level dropped, levels of fewer than
    `minimum` rows merged into one; for a numeric column with more than `categorical_levels` distinct values
    (entry age, not entry year) its standardised value and centred square."""
    blocks = []
    for name in covariates:
        column = frame[name]
        if (pd.api.types.is_numeric_dtype(column) and not pd.api.types.is_bool_dtype(column)
                and column.nunique() > categorical_levels):
            v = column.to_numpy(float)
            if not np.isfinite(v).all():
                raise MetricRefusal("censoring", f"censoring covariate {name} is not finite")
            sd = v.std()
            z = (v - v.mean()) / sd if sd > 0 else np.zeros(len(v))
            blocks.append(np.column_stack([z, z * z - np.mean(z * z)]))
            continue
        labels = column.astype(str).to_numpy()
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        merged = np.where(counts[inverse] < minimum, "~merged", labels)
        levels, inverse = np.unique(merged, return_inverse=True)
        if len(levels) > 1:
            blocks.append(np.eye(len(levels))[inverse][:, 1:])
    return np.column_stack(blocks) if blocks else np.zeros((len(frame), 0))


class Censoring:
    """G_i(t), the probability that row i is still under observation at t on time since entry, fitted on the
    evaluation rows over [0, horizon], in one form for every kind: G_i(t) = exp(-risk_i * L_c(i)(t)) with L_c a
    cumulative hazard curve. kind "cox": a Cox model for censoring given `covariates` (one Breslow curve, risk
    exp(x'beta)); "km": the marginal reverse Kaplan-Meier (one curve, risk 1); "strata": a reverse
    Kaplan-Meier within each level of covariates[0] (a curve per level, risk 1).

    `variance` adds G's own estimation to a metric's influence function: for an IPCW metric with sensitivity
    d_j = d(metric)/d(log w_j), row i's influence gains sum_m C_m dM_i(u_m) / D_m + b' psi_beta_i, with
    log w_j = risk_j L(T*_j) (T*_j = T_j- for an event by h, h for a row followed past it), C_m the total
    d_j risk_j of the rows whose L(T*_j) includes the censoring time u_m, dM_i row i's censoring martingale
    increment, D_m the curve's risk-set denominator (the Breslow S0, or the reverse KM's Y - dN with events
    leaving first) and, for the Cox model, psi_beta = I^-1 U (U the score residuals) and
    b = sum_j d_j risk_j (L(T*_j) x_j - sum_{u_m <= T*_j} xbar(u_m) dL_m)."""

    def __init__(self, frame, horizon, kind="cox", covariates=(), ridge=1.0):
        followup, code = frame.followup.to_numpy(float), frame.event_code.to_numpy(int)
        t, censored = censoring_window(followup, code, horizon)
        n = len(t)
        self.risk, self.curve, self.psi_beta = np.ones(n), np.zeros(n, np.int64), None
        # Nobody censored in the window (the censor-at-cutoff frames, whose potential follow-up reaches h): G = 1.
        X = censoring_design(frame, covariates) if kind == "cox" and censored.any() else np.zeros((n, 0))
        if kind == "strata":
            _, self.curve = np.unique(frame[covariates[0]].astype(str).to_numpy(), return_inverse=True)
        elif kind not in ("cox", "km"):
            raise ValueError(f"unknown censoring model {kind}")
        # Per curve: the censoring times u, the increments a of the hazard in dM, the denominators D, and the
        # cumulative hazard L at each u that G is built from.
        self.curves, self.pieces = [], []
        if X.shape[1]:
            beta, times, base, information = cox_fit(t, censored, X, ridge)
            Xc = X - X.mean(axis=0)
            self.risk = np.exp(Xc @ beta)
            increments = np.diff(np.r_[0.0, base])
            order = np.argsort(t, kind="mergesort")
            tail = np.cumsum(self.risk[order][::-1])[::-1]
            start = np.searchsorted(t[order], times, side="left")
            s0 = tail[start]
            xbar = np.cumsum((Xc * self.risk[:, None])[order][::-1], axis=0)[::-1][start] / s0[:, None]
            self.curves.append((times, base))
            self.pieces.append((times, increments, s0, np.searchsorted(times, t, side="right")))
            # Score residuals U_i = int (x_i - xbar) dM_i and psi_beta = U I^-1, with the Breslow risk set.
            at_risk = self.pieces[0][3]
            cumulative = np.r_[0.0, base][at_risk]
            xbar_lambda = np.vstack([np.zeros(Xc.shape[1]), np.cumsum(xbar * increments[:, None], axis=0)])
            own = np.searchsorted(times, t, side="left")
            U = (np.where(censored[:, None], Xc - xbar[np.minimum(own, len(times) - 1)], 0.0)
                 - self.risk[:, None] * (Xc * cumulative[:, None] - xbar_lambda[at_risk]))
            self.psi_beta = np.linalg.solve(information, U.T).T
            self.design, self.xbar, self.xbar_lambda, self.information = Xc, xbar, xbar_lambda, information
            self.psi_gram = self.psi_beta.T @ self.psi_beta
            # psi_beta summed by each row's own censoring time and by its at-risk count, so that psi_beta' first
            # costs O(censoring times x p) per metric instead of O(rows x p).
            self.psi_by_own = np.zeros((len(times), Xc.shape[1]))
            np.add.at(self.psi_by_own, own[censored], self.psi_beta[censored])
            self.psi_by_risk = np.zeros((len(times) + 1, Xc.shape[1]))
            np.add.at(self.psi_by_risk, at_risk, self.risk[:, None] * self.psi_beta)
        else:
            for c in range(self.curve.max() + 1):
                rows = self.curve == c
                self.curves.append(self._km(t[rows], censored[rows]))
                times = np.unique(t[rows][censored[rows]])
                k = np.searchsorted(times, t[rows], side="left")
                d = np.bincount(k[censored[rows]], minlength=len(times)).astype(float)
                ahead = len(t[rows]) - np.searchsorted(np.sort(t[rows]), times, side="right")
                # Events leave first, so the censoring risk set at u is {T > u} plus the censorings at u.
                self.pieces.append((times, d / (ahead + d), ahead.astype(float), k + censored[rows]))
        # Each row's position in its curve: how many censoring times its weight's L(T*) includes, and its own.
        event = (code != 0) & (followup <= horizon)
        self.weight_position, self.own, self.censored = np.zeros(n, np.int64), np.zeros(n, np.int64), censored
        self.at_risk = np.zeros(n, np.int64)
        for c, (times, _, _, at_risk) in enumerate(self.pieces):
            rows = self.curve == c
            position = np.where(event[rows], np.searchsorted(times, followup[rows], side="left"), len(times))
            self.weight_position[rows] = position
            self.own[rows] = np.searchsorted(times, t[rows], side="left")
            self.at_risk[rows] = at_risk

    @staticmethod
    def _km(t, censored):
        times, g, _ = reverse_km(t, np.where(censored, 0, 1))
        return times, -np.log(np.maximum(g, 1e-300))

    def subset(self, mask):
        """The same fitted curves for a subset of the rows (for Uno's weights; no variance)."""
        out = object.__new__(Censoring)
        out.risk, out.curve, out.curves = self.risk[mask], self.curve[mask], self.curves
        return out

    def hazard_before(self, t):
        """(curves x len(t)): each curve's cumulative hazard just before each time."""
        return np.vstack([step_at(times, base, t, left=True, start=0.0) for times, base in self.curves])

    def at(self, t, left=False):
        """G_i(t_i) for every row i, or the left limits G_i(t_i-)."""
        t = np.broadcast_to(np.asarray(t, float), self.risk.shape)
        cumulative = np.empty(len(t))
        for c, (times, base) in enumerate(self.curves):
            rows = self.curve == c
            cumulative[rows] = step_at(times, base, t[rows], left, start=0.0)
        return np.exp(-self.risk * cumulative)

    def _parts(self, rows, fixed, sensitivity):
        """The total influence of a metric over a cell as (first, b, projection): first per evaluation row (the
        influence given G on the cell's rows plus each row's censoring martingale term); for the Cox model the
        vector b with which psi_beta enters and projection = psi_beta' first (both None otherwise). rows: the
        cell's indices among the evaluation rows; fixed: its influence given G on them (summing to the metric's
        error, e.g. (x - mean) / n); sensitivity: its derivative in each of their log weights."""
        first = np.zeros(len(self.risk))
        first[rows] = fixed
        b = projection = None
        for c, (times, increments, denominator, _) in enumerate(self.pieces):
            chosen = self.curve[rows] == c
            if not len(times) or not chosen.any():
                continue
            members = rows[chosen]
            weight = sensitivity[chosen] * self.risk[members]
            position = self.weight_position[members]
            mass = np.bincount(position, weights=weight, minlength=len(times) + 1)
            # C_m: the weight of the rows whose L(T*) includes censoring time m.
            C = weight.sum() - np.cumsum(mass)[:-1]
            term = np.r_[0.0, np.cumsum(C * increments / denominator)]
            curve_rows = np.flatnonzero(self.curve == c)
            own = np.minimum(self.own[curve_rows], len(times) - 1)
            first[curve_rows] += (np.where(self.censored[curve_rows], C[own] / denominator[own], 0.0)
                                  - self.risk[curve_rows] * term[self.at_risk[curve_rows]])
            if self.psi_beta is not None:
                cumulative = np.r_[0.0, np.cumsum(increments)]
                b = self.design[members].T @ (weight * cumulative[position]) - self.xbar_lambda.T @ mass
                projection = (self.psi_beta[rows].T @ fixed + self.psi_by_own.T @ (C / denominator)
                              - self.psi_by_risk.T @ term)
        return first, b, projection

    def influence(self, rows, fixed, sensitivity):
        """Each evaluation row's total influence on a metric over a cell (see _parts)."""
        first, b, _ = self._parts(rows, fixed, sensitivity)
        return first if b is None else first + self.psi_beta @ b

    def variance(self, rows, fixed, sensitivity):
        """The sum of squared total influences, without forming psi_beta @ b row by row."""
        first, b, projection = self._parts(rows, fixed, sensitivity)
        total = float(first @ first)
        if b is not None:
            total += 2 * float(b @ projection) + float(b @ self.psi_gram @ b)
        return total

    def conditional_variance(self, rows, uncensored, g, decided, event, horizon):
        """Simulator only: the variance of an IPCW sum's released-minus-uncensored difference over a cell, given
        the uncensored outcomes, with G's estimation. rows: the cell's indices among the evaluation rows;
        uncensored: each one's uncensored contribution u_i (its term in the metric when observed); g: G_i where
        its outcome is decided, at `decided` (T_unc for an event of either cause by h, `event`, else h).

        Row i's total influence is u_i (observed_i / G_i - 1) + int h_i dM_i, with h_i the metric's
        censoring-martingale integrand (see the class; C_m and b from the expected sensitivities u_j at the decided
        times). Its conditional mean square is u_i^2 (1/G_i - 1) - 2 u_i H_i + int h_i^2 dA_i: H_i the compensator
        of h_i up to the decided time and A_i row i's cumulative censoring intensity, taken at its realized risk
        set (a sum over every row of the curve, which does not collapse where no case was lost)."""
        total = float(np.sum(uncensored ** 2 * (1 / g - 1)))
        window = np.where(event, decided, np.nextafter(horizon, np.inf))
        for c, (times, increments, denominator, _) in enumerate(self.pieces):
            chosen = self.curve[rows] == c
            if not len(times) or not chosen.any():
                continue
            members, u = rows[chosen], uncensored[chosen]
            weight = u * self.risk[members]
            position = np.where(event[chosen], np.searchsorted(times, decided[chosen], side="left"), len(times))
            mass = np.bincount(position, weights=weight, minlength=len(times) + 1)
            a = (weight.sum() - np.cumsum(mass)[:-1]) / denominator
            curve_rows = np.flatnonzero(self.curve == c)
            # The Cox model's psi_beta term enters h_i as slope' (x_i - xbar(u)), slope = I^-1 b.
            beta, own_beta, e = np.zeros(len(curve_rows)), np.zeros(len(members)), np.zeros(len(times))
            if self.psi_beta is not None:
                cumulative = np.r_[0.0, np.cumsum(increments)]
                b = self.design[members].T @ (weight * cumulative[position]) - self.xbar_lambda.T @ mass
                slope = np.linalg.solve(self.information, b)
                beta, own_beta, e = self.design[curve_rows] @ slope, self.design[members] @ slope, self.xbar @ slope
                # The Breslow risk set at u is every row with t >= u.
                reach = np.searchsorted(times, window[chosen], side="right")
            else:
                # The reverse Kaplan-Meier's: events leave first.
                reach = np.searchsorted(times, window[chosen], side="left")
            p0, p1, p2 = (np.r_[0.0, np.cumsum(increments * v)] for v in (1.0, a - e, (a - e) ** 2))
            k = self.at_risk[curve_rows]
            total += float(np.sum(self.risk[curve_rows] * (p2[k] + 2 * beta * p1[k] + beta ** 2 * p0[k])))
            total -= 2 * float(np.sum(weight * (p1[reach] + own_beta * p0[reach])))
        return total


def ipcw(frame, horizon, censoring):
    """IPCW weights at a horizon: 1/G(T-) for an event of either cause by h, 1/G(h) for a row followed past h,
    0 for a row censored by h."""
    t = frame.followup.to_numpy(float)
    code = frame.event_code.to_numpy(int)
    event = (code != 0) & (t <= horizon)
    known = event | (t > horizon)
    g = np.where(event, censoring.at(t, left=True), censoring.at(horizon))
    if (g[known] <= 0).any():
        raise MetricRefusal("censoring", "a row observed at the horizon has zero censoring survival")
    return np.where(known, 1.0 / np.where(known, g, 1.0), 0.0)


# --------------------------------------------------------------------------- #
# Wolbers' concordance
# --------------------------------------------------------------------------- #
def _earlier_below(rank, weight, group=None):
    """For each element, in the given order: the total weight of earlier elements of its group whose rank is
    smaller, and whose rank is equal. Ranks are integers from 0; O(n log n), one stable sort per bit."""
    n = len(rank)
    rank = np.asarray(rank, np.int64)
    weight = np.asarray(weight, float)
    group = np.zeros(n, np.int64) if group is None else np.asarray(group, np.int64)
    if n == 0:
        return np.zeros(0), np.zeros(0)

    def earlier_same_key(key, mass):
        order = np.argsort(key, kind="stable")
        k, m = key[order], mass[order]
        running = np.cumsum(m) - m
        starts = np.r_[0, np.flatnonzero(np.diff(k)) + 1]
        out = np.empty(n)
        out[order] = running - np.repeat(running[starts], np.diff(np.r_[starts, n]))
        return out

    bits = max(int(rank.max()).bit_length(), 1)
    below = np.zeros(n)
    for level in range(bits):
        # A smaller earlier rank first differs from this one at exactly one bit, where it has 0 and this has 1.
        zero = ((rank >> level) & 1) == 0
        key = (group << (bits - level)) | (rank >> (level + 1))
        below += np.where(zero, 0.0, earlier_same_key(key, np.where(zero, weight, 0.0)))
    return below, earlier_same_key(group * (int(rank.max()) + 1) + rank, weight)


# Pairs (cases x rows) the IPCW concordance holds in memory at once, as float64: about 80 MB per array.
CONCORDANCE_BLOCK = 10_000_000


def wolbers_concordance(time, code, risks, horizon, censoring=None):
    """Wolbers' competing-risk concordance truncated at a horizon, on time since entry, for one risk per row or
    for several (variants x rows) at once. A case i (cause 1 at T_i <= h) is comparable with j when j outlives
    it (T_j > T_i, or T_j = T_i and j is not a case) or when j died (code 2) before
    T_i; the pair is concordant when risk_i > risk_j, ties counting half. Without `censoring` every pair weighs
    one (Harrell).
    With a Censoring of the same rows every pair weighs the inverse probability that it is seen (Uno's weights,
    each row with its own censoring survival): 1/(G_i(T_i-) G_j(T_i-)) when j outlives i, and
    1/(G_i(T_i-) G_j(T_j-)) when j died first. The outliving pairs are then summed exactly, block by block."""
    t, code = np.asarray(time, float), np.asarray(code, int)
    risks = np.asarray(risks, float)
    single = risks.ndim == 1
    risks = np.atleast_2d(risks)
    n = len(t)
    case = (code == 1) & (t <= horizon)
    if not case.any():
        raise MetricRefusal("concordance", "no cases by the horizon")
    if censoring is None:
        inverse_left = np.ones(n)
    else:
        g_left = censoring.at(t, left=True)
        if (g_left[case] <= 0).any():
            raise MetricRefusal("concordance", "zero censoring survival at a case time")
        inverse_left = np.divide(1.0, g_left, out=np.zeros(n), where=g_left > 0)
    # j died before the case: ascending time, cases before deaths at a tied time (a death at the case's own time
    # outlives it instead).
    died_order = np.lexsort((code == 2, t))
    mass = np.where(code == 2, inverse_left, 0.0)[died_order]
    died_total = np.empty(n)
    died_total[died_order] = np.cumsum(mass) - mass
    comparable = float(np.sum((inverse_left * died_total)[case]))
    concordant = np.zeros(len(risks))
    ranks = [np.unique(risk, return_inverse=True)[1] for risk in risks]
    for v, rank in enumerate(ranks):
        below, equal = np.empty(n), np.empty(n)
        below[died_order], equal[died_order] = _earlier_below(rank[died_order], mass)
        concordant[v] += np.sum((inverse_left * (below + 0.5 * equal))[case])
    if censoring is None:
        # j outlives the case: descending time, non-cases before cases at a tied time; tied cases removed.
        order = np.lexsort((case, -t))
        tied = order[case[order]]
        time_group = np.unique(t, return_inverse=True)[1][tied]
        later_total = np.empty(n)
        later_total[order] = np.arange(n, dtype=float)
        later_total[tied] -= _earlier_below(np.zeros(len(tied), np.int64), np.ones(len(tied)), time_group)[1]
        comparable += float(np.sum(later_total[case]))
        for v, rank in enumerate(ranks):
            below, equal = np.empty(n), np.empty(n)
            below[order], equal[order] = _earlier_below(rank[order], np.ones(n))
            tied_below, tied_equal = _earlier_below(rank[tied], np.ones(len(tied)), time_group)
            below[tied] -= tied_below
            equal[tied] -= tied_equal
            concordant[v] += np.sum((below + 0.5 * equal)[case])
    else:
        cases = np.flatnonzero(case)
        size = max(1, CONCORDANCE_BLOCK // n)
        for start in range(0, len(cases), size):
            rows = cases[start:start + size]
            outlives = (t[None, :] > t[rows, None]) | ((t[None, :] == t[rows, None]) & (code[None, :] != 1))
            hazard = censoring.hazard_before(t[rows])[censoring.curve].T
            with np.errstate(over="ignore"):
                weight = np.where(outlives, np.exp(censoring.risk[None, :] * hazard), 0.0) * inverse_left[rows, None]
            comparable += float(weight.sum())
            for v, risk in enumerate(risks):
                kernel = (risk[None, :] < risk[rows, None]) + 0.5 * (risk[None, :] == risk[rows, None])
                concordant[v] += float(np.sum(weight * kernel))
    if not comparable > 0 or not np.isfinite(comparable):
        raise MetricRefusal("concordance", "no comparable pairs with finite weight")
    c = concordant / comparable
    return float(c[0]) if single else c


# --------------------------------------------------------------------------- #
# the results table
# --------------------------------------------------------------------------- #
# Heterogeneity axes: study-cohort's frame columns. Entry year is the baseline year in the binary cohort and the
# landmark year in the survival cohort, two different partitions, so the two keep different names.
AXES = {"binary": ("ancestry", "region", "division", "ehr_site", "baseline_year", "sex", "age_band",
                   "ses_quartile", "lookback_tertile"),
        "survival": ("ancestry", "region", "division", "ehr_site", "entry_year", "sex", "age_band",
                     "ses_quartile", "lookback_tertile")}
# Every variant carries its paired difference from competitor (a), covariates only, and (b), standard; a LOGO
# row also carries its difference from the same variant's pooled fit on the same rows ("pooled").
REFERENCES = ("covariates", "standard")
CENSORING_COVARIATES = ("ehr_site", "region", "ancestry", "entry_age", "entry_year")
KEYS = ("disease", "model", "variant", "fit", "stratum", "horizon")
PAIRED = tuple(f"d_{metric}_{ref}{se}" for ref in REFERENCES + ("pooled",) for metric in ("auc", "brier")
               for se in ("", "_se"))
# The digest's registry (study/digest.py): a count; a proportion, which times a count gives a count (or a
# count-like quantity) and so goes wherever its cell's counts go; or a score, which never inverts to a count.
METRICS = {
    "n": "count", "cases": "count",
    "mean_risk": "proportion", "obs_risk": "proportion", "obs_risk_se": "proportion", "oe": "proportion",
    "oe_lo": "proportion", "oe_hi": "proportion", "n_eff": "proportion",
    "auc": "score", "auc_se": "score", "brier": "score", "brier_se": "score", "cal_int": "score",
    "cal_int_se": "score", "cal_slope": "score", "cal_slope_se": "score", "ici": "score", "c_harrell": "score",
    "c_uno": "score", "w_max": "score", **{name: "score" for name in PAIRED},
    # Simulator only: accuracy against the true risk, and each metric on the uncensored outcomes.
    "rmse_true": "score", "mae_true": "score", "bias_true": "score", "corr_true": "score",
    "mean_true": "proportion", "risk_unc": "proportion", "auc_unc": "score", "brier_unc": "score",
    "n_unc": "count", "auc_unc_diff_se": "score", "brier_unc_diff_se": "score", "risk_unc_diff_se": "score",
    "auc_unc_diff_se_known_g": "score", "brier_unc_diff_se_known_g": "score", "risk_unc_diff_se_known_g": "score",
    "oe_true": "score", "cal_slope_true": "score", "auc_true": "score", "slope_bias_true": "score",
    "slope_rmse_true": "score",
}
INSUFFICIENT = "insufficient_support"
UNSUPPORTED = "horizon_unsupported"


def stratum_labels(column):
    """An axis as strings, missing values as their own level, so every axis partitions its cell."""
    labels = column.astype(object).where(column.notna(), "unknown")
    return labels.astype(str).to_numpy()


def cells(frame, axes):
    """(stratum, mask): "overall", then "<axis>_<level>" for every level of every axis."""
    yield "overall", np.ones(len(frame), bool)
    for axis in axes:
        labels = stratum_labels(frame[axis])
        for level in sorted(set(labels)):
            yield f"{axis}_{level}", labels == level


def _refused(row, name, refusal):
    row[f"{name}_status"] = "refused"
    print(f"study_evaluate_refused {row['model']} {row['variant']} {row['fit']} {row['stratum']} "
          f"{row['horizon']} {name}: {refusal.args[-1]}", flush=True)


def _guarded(row, name, compute):
    """Add a metric group to a row, or mark it refused and log why."""
    try:
        row.update(compute())
    except MetricRefusal as refusal:
        _refused(row, name, refusal)


def _paired(row, ref, auc_difference, auc_difference_se, loss_difference):
    row[f"d_auc_{ref}"] = float(auc_difference)
    row[f"d_auc_{ref}_se"] = float(auc_difference_se)
    row[f"d_brier_{ref}"] = float(loss_difference.mean())
    row[f"d_brier_{ref}_se"] = float(loss_difference.std(ddof=1) / np.sqrt(len(loss_difference)))


def expected_auc(p, p_true):
    """The AUC of predictions p in expectation over outcomes drawn from the true probabilities:
    sum over pairs i != j of pi_i (1 - pi_j) K(p_i, p_j) / sum of pi_i (1 - pi_j), K counting a tie half. Every
    row is a case with weight pi and a control with weight 1 - pi, and the self pairs are removed."""
    p, pi = np.asarray(p, float), np.asarray(p_true, float)
    n = len(p)
    both, _ = weighted_auc(np.r_[p, p], np.r_[np.ones(n), np.zeros(n)], np.r_[pi, 1 - pi])
    total, own = pi.sum() * (1 - pi).sum(), float(np.sum(pi * (1 - pi)))
    return float((both * total - 0.5 * own) / (total - own))


def truth_metrics(P, true_risk, S=None, true_slope=None):
    """Accuracy against the simulator's true probabilities (SPEC sections 5 and 8), per variant: the RMSE, mean
    absolute error (the ICI against truth), bias and correlation of predicted against true risk; the expected
    O/E_true = mean(p_true) / mean(p); the expected calibration slope, the least-squares slope of logit(p_true)
    on logit(p); the expected AUC under p_true; and with S (variants x rows) the slope recovery against
    true_slope, both on the pooled model's z scale: bias and RMSE of the predicted slope."""
    rows = []
    true_logit = clipped_logit(true_risk)
    for a, p in enumerate(P):
        error = p - true_risk
        spread = p.std() * true_risk.std()
        logit = clipped_logit(p)
        centred = logit - logit.mean()
        row = {"rmse_true": float(np.sqrt(np.mean(error ** 2))), "mae_true": float(np.mean(np.abs(error))),
               "bias_true": float(error.mean()),
               "corr_true": float(np.mean((p - p.mean()) * (true_risk - true_risk.mean())) / spread)
               if spread > 0 else float("nan"),
               "oe_true": float(true_risk.mean() / p.mean()), "auc_true": expected_auc(p, true_risk)}
        if centred @ centred > 0:
            row["cal_slope_true"] = float(centred @ (true_logit - true_logit.mean()) / (centred @ centred))
        if S is not None:
            known = np.isfinite(S[a]) & np.isfinite(true_slope)
            if known.any():
                slope_error = S[a][known] - true_slope[known]
                row.update(slope_bias_true=float(slope_error.mean()),
                           slope_rmse_true=float(np.sqrt(np.mean(slope_error ** 2))))
        rows.append(row)
    return rows


def binary_cell(y, P, variants, fit, stratum, minimum, *, released, references=REFERENCES, pooled=None,
                true_risk=None, slopes=None):
    """Rows for one binary cell: y the outcomes, P the (variants x rows) predictions, released the release rule's
    development side (release_by_development), pooled the pooled fit's predictions of the same variants for a LOGO
    cell, true_risk the simulator's p_ever, and slopes (simulator only) the (variants x rows) predicted PGS slopes
    with the true slopes, both on the pooled model's z scale. A cell the release rule withholds carries its counts
    and the same status as one under the test floor."""
    n, cases = len(y), int(y.sum())
    base = {"model": "binary", "fit": fit, "stratum": stratum, "horizon": None, "n": n, "cases": cases}
    if not released or min(n, cases, n - cases) < minimum:
        return [dict(base, variant=v, status=INSUFFICIENT) for v in variants]
    stacked = P if pooled is None else np.vstack([P, pooled])
    aucs, cov = delong(y, stacked)
    losses = (y[None, :] - stacked) ** 2
    truth = (truth_metrics(P, true_risk, *(slopes or (None, None))) if true_risk is not None
             else [{} for _ in variants])
    rows = []
    for a, variant in enumerate(variants):
        p = P[a]
        oe, half = cases / p.sum(), Z95 * math.sqrt(1 / cases - 1 / n)
        row = dict(base, variant=variant, status="ok", auc=float(aucs[a]), auc_se=float(np.sqrt(cov[a, a])),
                   brier=float(losses[a].mean()), brier_se=float(losses[a].std(ddof=1) / np.sqrt(n)),
                   mean_risk=float(p.mean()), oe=oe, oe_lo=oe * math.exp(-half), oe_hi=oe * math.exp(half),
                   **truth[a])
        if true_risk is not None:
            row["mean_true"] = float(true_risk.mean())
        _guarded(row, "calibration", lambda: calibration(y, p))
        _guarded(row, "ici", lambda: {"ici": integrated_calibration_index(y, p)})
        others = [(ref, variants.index(ref)) for ref in references if ref in variants and ref != variant]
        others += [("pooled", len(variants) + a)] if pooled is not None else []
        for ref, b in others:
            _paired(row, ref, aucs[a] - aucs[b], np.sqrt(max(cov[a, a] + cov[b, b] - 2 * cov[a, b], 0.0)),
                    losses[a] - losses[b])
        rows.append(row)
    return rows


def cell_support(time, code, horizon):
    """A cell's rows followed past the horizon and the one-sided 95% upper bound on its own reverse
    Kaplan-Meier G(h) (Greenwood, log scale): the prespecified support rule's inputs."""
    window, censored = censoring_window(time, code, horizon)
    times, g, var = reverse_km(window, np.where(censored, 0, 1))
    g_h, var_h = float(step_at(times, g, horizon)), float(step_at(times, var, horizon, start=0.0))
    upper = min(math.exp(math.log(g_h) + ONE_SIDED_95 * math.sqrt(var_h)), 1.0) if g_h > 0 else 0.0
    return int(np.sum(time > horizon)), upper


def survival_cell(t, code, w, model, rows, P, variants, fit, stratum, horizon, minimum, *, released,
                  references=REFERENCES, pooled=None, truth=None, slopes=None):
    """Rows for one survival cell at a horizon: t the follow-up from entry; code 0 censored, 1 disease, 2 death
    (competing); w the IPCW weights of `model`, the evaluation's censoring model,
    and `rows` the cell's indices among its rows; P the (variants x rows) CIFs at the horizon.
    truth (simulator only): (true CIF at h, uncensored follow-up, uncensored event code, G where the
    uncensored outcome is decided: T- for an event of either cause by h, else h, and whether the row has an
    uncensored outcome at all) per row.

    Every IPCW standard error carries the censoring model's own estimation (Censoring.variance). Reported only
    where the release rule's development side passes (`released`, release_by_development) and the prespecified
    support rule holds: at least `minimum` cases, known non-cases and rows followed past h; the cell's own G(h)
    upper bound at or above POSITIVITY_FLOOR; and every known row's weight at most 1/POSITIVITY_FLOOR. A cell the
    release rule withholds carries its counts and the same status as one under the test floor."""
    y = ((code == 1) & (t <= horizon)).astype(float)
    known = w > 0
    n, cases, controls = len(t), int(y.sum()), int(np.sum(known & (y == 0)))
    base = {"model": "survival", "fit": fit, "stratum": stratum, "horizon": horizon, "n": n, "cases": cases}
    if not released or min(n, cases, controls) < minimum:
        return [dict(base, variant=v, status=INSUFFICIENT) for v in variants]
    followed, g_upper = cell_support(t, code, horizon)
    if followed < minimum or g_upper < POSITIVITY_FLOOR or w.max() > 1 / POSITIVITY_FLOOR:
        return [dict(base, variant=v, status=UNSUPPORTED) for v in variants]

    def mean_se(x):
        # A mean of weighted row terms: its influence given G, and its sensitivity to each log weight.
        return float(np.sqrt(model.variance(rows, (x - x.mean()) / n, x / n)))

    def auc_se(influence):
        # The AUC's influence given G is also its derivative in each log weight.
        return float(np.sqrt(model.variance(rows, influence / n, influence / n)))

    # The observed risk is the IPCW mean of the outcome, with the same weights as every other metric. With the
    # cell's own reverse Kaplan-Meier as G it is exactly the Aalen-Johansen incidence (n S(u-) G(u-) = Y(u));
    # a marginal Aalen-Johansen per cell would be biased wherever censoring depends on a covariate.
    observed = float(np.mean(w * y))
    observed_se = mean_se(w * y)
    shared = {"obs_risk": observed, "obs_risk_se": observed_se, "w_max": float(w.max()),
              "n_eff": float(w.sum() ** 2 / np.sum(w ** 2))}
    stacked = P if pooled is None else np.vstack([P, pooled])
    fits = [weighted_auc(p, y, w) for p in stacked]
    losses = w[None, :] * (y[None, :] - stacked) ** 2
    concordance = {}
    for name, censoring in (("c_harrell", None), ("c_uno", model.subset(rows))):
        try:
            concordance[name] = wolbers_concordance(t, code, P, horizon, censoring)
        except MetricRefusal as refusal:
            concordance[name] = refusal
    truth_rows = [{} for _ in variants]
    if truth is not None:
        true_cif, t_unc, code_unc, g_unc, defined = truth
        truth_rows = truth_metrics(P, true_cif, *(slopes or (None, None)))
        # The uncensored comparisons run over the m rows that have an uncensored outcome (see evaluate()); the
        # released values count all n rows. The k0 others censored at entry sit inside G's jump at 0, so with no
        # bias a row with an outcome is observed with probability n / (n - k0) G_i, and that is its G in the
        # paired SEs: under the marginal G it is exactly those rows' own G. (Others followed past entry, as the
        # cutoff frame's are, are observed non-cases and stay in the released values.)
        m = int(defined.sum())
        at_entry = int(np.sum(~defined & (t == 0)))
        t_unc, code_unc, own = t_unc[defined], code_unc[defined], rows[defined]
        g_unc = np.minimum(g_unc[defined] * n / max(n - at_entry, 1), 1.0)
        y_unc = ((code_unc == 1) & (t_unc <= horizon)).astype(float)
        event_unc = (code_unc != 0) & (t_unc <= horizon)

        def paired_se(name, uncensored):
            # The SEs of a released-minus-uncensored difference over the same rows, given the uncensored outcomes
            # (the IPCW estimator gate's): row i's term is its uncensored contribution u_i times
            # (observed_i / G_i - 1), with G_i at T_unc- for an event of either cause by h and at h otherwise. They
            # count every row's chance of censoring, not the censorings that happened: the rows' own differences
            # are (w - 1) u >= 0 in a cell that lost no case, and their sample SD collapses there (z ~ sqrt(cases);
            # study-eval's N2a oracle, 09-19). The first carries G's estimation (Censoring.conditional_variance);
            # the second, sum u_i^2 (1 / G_i - 1), holds G known, which is conservative while G is well estimated.
            with_g = model.conditional_variance(own, uncensored, g_unc, t_unc, event_unc, horizon)
            return {f"{name}_diff_se": float(np.sqrt(with_g)),
                    f"{name}_diff_se_known_g": float(np.sqrt(np.sum(uncensored ** 2 * (1 / g_unc - 1))))}

        for a, p in enumerate(P[:, defined] if m else ()):
            unc_loss = (y_unc - p) ** 2
            truth_rows[a].update(
                n_unc=m, mean_true=float(true_cif[defined].mean()), risk_unc=float(y_unc.mean()),
                brier_unc=float(unc_loss.mean()), **paired_se("risk_unc", y_unc / m),
                **paired_se("brier_unc", unc_loss / m))
            if 0 < y_unc.sum() < m:
                auc_unc, influence_unc = weighted_auc(p, y_unc, np.ones(m))
                truth_rows[a].update(auc_unc=auc_unc, **paired_se("auc_unc", influence_unc / m))
    out = []
    for a, variant in enumerate(variants):
        p, (auc, influence) = P[a], fits[a]
        row = dict(base, variant=variant, status="ok", auc=auc, auc_se=auc_se(influence),
                   brier=float(losses[a].mean()), brier_se=mean_se(losses[a]), mean_risk=float(p.mean()),
                   **shared, **truth_rows[a])
        if observed > 0:
            oe, half = observed / row["mean_risk"], Z95 * observed_se / observed
            row.update(oe=oe, oe_lo=oe * math.exp(-half), oe_hi=oe * math.exp(half))
        _guarded(row, "calibration", lambda: calibration(y[known], p[known], w[known], censoring=model,
                                                         rows=rows[known]))
        _guarded(row, "ici", lambda: {"ici": integrated_calibration_index(y, p, w)})
        for name, value in concordance.items():
            if isinstance(value, MetricRefusal):
                _refused(row, name, value)
            else:
                row[name] = float(value[a])
        others = [(ref, variants.index(ref)) for ref in references if ref in variants and ref != variant]
        others += [("pooled", len(variants) + a)] if pooled is not None else []
        for ref, b in others:
            difference = losses[a] - losses[b]
            row[f"d_auc_{ref}"] = float(auc - fits[b][0])
            row[f"d_auc_{ref}_se"] = auc_se(influence - fits[b][1])
            row[f"d_brier_{ref}"] = float(difference.mean())
            row[f"d_brier_{ref}_se"] = mean_se(difference)
        out.append(row)
    return out


def development_counts(kind, dev, strata, config, horizon=None):
    """stratum -> (cases, known non-cases, rows) of the development rows (study.py's train: every row outside the
    outer test) in each cell: the counts the release rule reads. For survival, at a horizon, over the rows whose
    potential follow-up reaches it, with the test cells' definitions: a case is a disease event by h, and a known
    non-case a row followed past h or dead by it."""
    if kind == "binary":
        y = dev.y.to_numpy(int)
        if not np.isin(y, (0, 1)).all():
            raise ValueError("binary development outcome must be 0/1")
        case, control = y == 1, y == 0
    else:
        potential = potential_followup(dev, config)
        dev = dev.loc[np.ones(len(dev), bool) if potential is None else potential >= horizon].reset_index(drop=True)
        t, code = dev.followup.to_numpy(float), dev.event.to_numpy(int)
        if (t < 0).any() or not np.isin(code, (0, 1, 2)).all() or ((t == 0) & (code != 0)).any():
            raise ValueError("development follow-up must be non-negative, events 0, 1 or 2, censored at entry")
        case = (code == 1) & (t <= horizon)
        control = (t > horizon) | ((code == 2) & (t <= horizon))
    return {stratum: (int(case[mask].sum()), int(control[mask].sum()), int(mask.sum()))
            for stratum, mask in cells(dev, strata)}


def release_by_development(counts, n, minimum):
    """The release rule's development side (SPEC section 3, 17:50Z): a cell's development cases and known
    non-cases, scaled to its n test rows, both at least RELEASE_MARGIN x minimum. It reads no test outcome."""
    cases, controls, rows = counts
    return rows > 0 and min(cases, controls) * n / rows >= RELEASE_MARGIN * minimum


def _fits(predictions):
    """variant -> fits, "pooled" first, from the (variant, fit) keys of the predictions."""
    out = {}
    for variant, fit in predictions:
        out.setdefault(variant, []).append(fit)
    return {v: sorted(f, key=lambda fit: (fit != "pooled", fit)) for v, f in out.items()}


def _held_out(test, fit, predictions, variants):
    """The rows a LOGO fit predicts: its held-out group's outer-test rows. Every variant's LOGO prediction must
    cover exactly those rows, and the axis column must agree."""
    _, axis, group = fit.split(":", 2)
    rows = stratum_labels(test[axis]) == group
    for variant in variants:
        predicted = np.isfinite(np.asarray(predictions[(variant, fit)], float).reshape(len(test), -1)).all(axis=1)
        if not np.array_equal(predicted, rows):
            raise ValueError(f"{variant} {fit} predicts rows other than its held-out group")
    return rows


def potential_followup(frame, config):
    """Years from entry to the CDR cutoff, known at baseline: `max_followup` if the frame has it, else
    admin_years (cutoff - baseline) less the landmark; None when the frame has neither."""
    if "max_followup" in frame:
        return frame.max_followup.to_numpy(float)
    if "admin_years" in frame:
        landmark_days = config.get("cohort", {}).get("landmark_days", 180)
        return frame.admin_years.to_numpy(float) - landmark_days / 365.25
    return None


def evaluate(kind, test, predictions, horizons, config, train, truth=None, slopes=None):
    """Every row of the single results table for one disease and model, from its outer-test rows.

    kind: "binary" or "survival". test: the outer-test frame (binary: `y`; survival: `followup`, `event` (0
    censored, 1 disease, 2 death, competing; a follow-up of 0 only censored at entry) and `entry_age`; both: the
    axis and censoring columns, and admin_years). train: the development rows, the same columns; their counts
    decide which cells are released (release_by_development), and nothing else reads them. predictions:
    (variant, fit) ->
    risk per test row (binary) or rows x horizons CIF (survival), NaN outside the fit's rows; fit is "pooled"
    or "logo:<axis>:<group>". A LOGO row is the pooled cell of its held-out group, with stratum "overall".
    truth (simulator only): rows aligned with test carrying p_ever (binary) or cif_<h>y, uncensored_event and
    uncensored_exit_age (survival; the two uncensored columns are null on a row the no-exit world would not admit,
    which the frame censors at entry, and those rows drop out of the uncensored comparisons alone, counted by
    n_unc). slopes (simulator only): (variant, fit) -> the predicted PGS slope per test row
    (binary) or rows x horizons (survival), set beside truth's slope (binary) or slope_cif_<h>y (survival); both
    must be per the POOLED model's z, score / sd over all development rows, LOGO fits included (study.py rescales
    truth by sd_pooled / z_sd and each fit's slope by sd_pooled / sd_fit).

    At a survival horizon only rows whose potential follow-up reaches it are evaluated, and the censoring model
    is fitted on exactly those rows, separately for each LOGO group (SPEC section 8). A cell's metrics are
    released only when its development counts pass the release rule and its test counts the floor; otherwise it
    carries its test counts and insufficient_support, whichever side failed.
    config["evaluate"] may set "censoring" ("cox", "km" or "strata") and "censoring_covariates";
    config["report"]["small_cell_max"] is the largest count withheld."""
    settings = config.get("evaluate", {})
    minimum = int(config.get("report", {}).get("small_cell_max", SMALL_CELL_MAX)) + 1
    fits = _fits(predictions)
    variants = [v for v in ("ours", "covariates", "standard", "znorm2", "z_pc", "calpred") if v in fits]
    variants += sorted(set(fits) - set(variants))
    fit_names = sorted({f for fs in fits.values() for f in fs}, key=lambda fit: (fit != "pooled", fit))
    if truth is not None and len(truth) != len(test):
        raise ValueError("truth rows must align with the test rows")
    rows = []
    for fit in fit_names:
        have = [v for v in variants if fit in fits[v]]
        if fit == "pooled":
            mask_all, strata = np.ones(len(test), bool), AXES[kind]
        else:
            mask_all, strata = _held_out(test, fit, predictions, have), ()
        sub = test.loc[mask_all].reset_index(drop=True)
        # The development rows of the same cells: all of them for the pooled fit, the held-out group's for LOGO.
        dev = train if fit == "pooled" else train.loc[
            stratum_labels(train[fit.split(":", 2)[1]]) == fit.split(":", 2)[2]].reset_index(drop=True)
        sub_truth = None if truth is None else truth.loc[mask_all].reset_index(drop=True)
        take = lambda variant, fit_name: np.asarray(predictions[(variant, fit_name)], float)[mask_all]
        pooled = None
        if fit != "pooled":
            pooled = {v: take(v, "pooled") for v in have if (v, "pooled") in predictions}
            have = [v for v in have if v in pooled]
        S_fit = None
        if slopes is not None and sub_truth is not None and all((v, fit) in slopes for v in have):
            S_fit = [np.asarray(slopes[(v, fit)], float)[mask_all] for v in have]
        if kind == "binary":
            y = sub.y.to_numpy(int)
            if not np.isin(y, (0, 1)).all():
                raise ValueError("binary outcome must be 0/1")
            P_all = np.vstack([take(v, fit) for v in have])
            Q_all = None if pooled is None else np.vstack([pooled[v] for v in have])
            true_all = None if sub_truth is None else sub_truth.p_ever.to_numpy(float)
            slope_all = None
            if S_fit is not None and "slope" in sub_truth:
                slope_all = (np.vstack(S_fit), sub_truth.slope.to_numpy(float))
            dev_cells = development_counts(kind, dev, strata, config)
            for stratum, mask in cells(sub, strata):
                released = release_by_development(dev_cells.get(stratum, (0, 0, 0)), int(mask.sum()), minimum)
                rows += binary_cell(y[mask], P_all[:, mask], have, fit, stratum, minimum, released=released,
                                    pooled=None if Q_all is None else Q_all[:, mask],
                                    true_risk=None if true_all is None else true_all[mask],
                                    slopes=None if slope_all is None else (slope_all[0][:, mask], slope_all[1][mask]))
            continue
        potential = potential_followup(sub, config)
        for j, horizon in enumerate(float(h) for h in horizons):
            eligible = np.ones(len(sub), bool) if potential is None else potential >= horizon
            frame = sub.loc[eligible].rename(columns={"event": "event_code"}).reset_index(drop=True)
            t, code = frame.followup.to_numpy(float), frame.event_code.to_numpy(int)
            if (t < 0).any():
                raise ValueError("survival follow-up must be non-negative")
            if not np.isin(code, (0, 1, 2)).all():
                raise ValueError("a survival event code outside events 0, 1 or 2 (censored, disease, death)")
            # A row may exit at entry only censored: study-cohort's frames censor an EHR with no record after the
            # landmark there (gnomon#2400), and a disease event or death at the landmark cannot occur in a kept row.
            if ((t == 0) & (code != 0)).any():
                raise ValueError("a survival row exiting at entry must be censored, not an event")
            model = Censoring(frame, horizon, settings.get("censoring", "cox"),
                              tuple(c for c in settings.get("censoring_covariates", CENSORING_COVARIATES)
                                    if c in frame))
            w = ipcw(frame, horizon, model)
            P_all = np.vstack([take(v, fit).reshape(len(sub), -1)[eligible, j] for v in have])
            Q_all = None if pooled is None else np.vstack([pooled[v].reshape(len(sub), -1)[eligible, j]
                                                           for v in have])
            truth_all = slope_all = None
            if sub_truth is not None:
                known = sub_truth.loc[eligible].reset_index(drop=True)
                true_cif = known[f"cif_{horizon:g}y"].to_numpy(dtype=float, na_value=np.nan)
                if not np.isfinite(true_cif).all():
                    raise ValueError(f"the truth lacks cif_{horizon:g}y on {int((~np.isfinite(true_cif)).sum())} rows")
                # A frame row the no-exit world would not admit (its EHR exit came by the landmark, and a latent
                # code after that exit and by the landmark was never recorded) has no uncensored outcome. The
                # last-contact frame censors it at entry; the cutoff frame follows it, event-free or to a death, as
                # no record can follow its EHR exit. Its uncensored terms are skipped by name, never cast. A null
                # on a recorded disease event, or on one of the two columns only, is a truth defect.
                exit_unc = known.uncensored_exit_age.to_numpy(dtype=float, na_value=np.nan)
                event_unc = known.uncensored_event.to_numpy(dtype=float, na_value=np.nan)
                defined = np.isfinite(exit_unc) & np.isfinite(event_unc)
                partial = np.isfinite(exit_unc) != np.isfinite(event_unc)
                if partial.any() or (~defined & (code == 1)).any():
                    raise ValueError("a truth row's uncensored outcome is partly null, or null on a recorded disease "
                                     "event")
                t_unc = np.where(defined, exit_unc - frame.entry_age.to_numpy(float), np.nan)
                code_unc = np.where(defined, event_unc, 0.0).astype(int)
                # G where each row's uncensored outcome is decided: T- for an event of either cause by h, else h.
                decided = defined & (code_unc != 0) & (t_unc <= horizon)
                g_unc = np.where(decided, model.at(np.where(decided, t_unc, horizon), left=True), model.at(horizon))
                truth_all = (true_cif, t_unc, code_unc, g_unc, defined)
                if S_fit is not None and f"slope_cif_{horizon:g}y" in known:
                    slope_all = (np.vstack([s.reshape(len(sub), -1)[eligible, j] for s in S_fit]),
                                 known[f"slope_cif_{horizon:g}y"].to_numpy(float))
            dev_cells = development_counts(kind, dev, strata, config, horizon)
            for stratum, mask in cells(frame, strata):
                released = release_by_development(dev_cells.get(stratum, (0, 0, 0)), int(mask.sum()), minimum)
                rows += survival_cell(t[mask], code[mask], w[mask], model, np.flatnonzero(mask), P_all[:, mask],
                                      have, fit, stratum, horizon, minimum, released=released,
                                      pooled=None if Q_all is None else Q_all[:, mask],
                                      truth=None if truth_all is None else tuple(v[mask] for v in truth_all),
                                      slopes=None if slope_all is None
                                      else (slope_all[0][:, mask], slope_all[1][mask]))
    return rows
