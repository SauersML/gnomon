"""Held-out metrics for the study's single results table.

One row per (disease, model, variant, fit, stratum, horizon), the digest's KEYS; the metrics a row can carry
are the registry METRICS, and the layout is written up in the study's TABLE.md. Every metric here is checked
against an independent reference implementation by tests/test_study_evaluate_reference.py (R survival/stats
and pROC, timeROC and riskRegression where installed; lifelines, scikit-survival, scikit-learn and statsmodels),
each with a planted error that must make it fail.

Binary cells: AUC with its DeLong standard error, Brier, O/E, the calibration intercept (slope fixed at one) and
slope, the integrated calibration index, and paired AUC and Brier differences against the covariates-only and
standard competitors.

Survival cells, at each horizon h on time since entry, with death (and an exclusion-rule exit, when study-cohort
makes it one) a competing event:
- the IPCW AUC with controls including deaths (Blanche definition 2);
- the IPCW Brier score, a death before h being a known non-case weighted 1/G(T-);
- the Aalen-Johansen observed risk and O/E against the mean predicted CIF;
- the IPCW calibration intercept, slope and integrated calibration index;
- Wolbers' competing-risk concordance truncated at h, unweighted (Harrell) and IPCW (Uno);
- paired AUC and Brier differences with influence-function standard errors.

G, the censoring survival, is an evaluation nuisance fitted on the evaluation rows themselves: a Cox model for
loss to follow-up over [0, h] on the rows whose potential follow-up reaches h, given site, region, ancestry,
entry age and entry year. Deaths and disease events end follow-up and are not censorings. Standard errors are
conditional on the fitted G.
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
    model-based covariance (R glm's for unit weights) and their sandwich covariance (HC0), which is the one
    to use when the weights are IPCW weights rather than frequencies."""
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
    score = X * (w * (y - p))[:, None]
    return beta, bread, bread @ (score.T @ score) @ bread


def calibration(target, p, weights=None, robust=False):
    """Calibration intercept (logit P(y) = a + logit p, slope fixed at one) and slope (logit P(y) = a + b logit p),
    each with its standard error."""
    lp = clipped_logit(p)
    ones = np.ones((len(lp), 1))
    a, model_a, robust_a = logistic_fit(ones, target, weights, offset=lp)
    b, model_b, robust_b = logistic_fit(np.column_stack([ones, lp]), target, weights)
    cov_a, cov_b = (robust_a, robust_b) if robust else (model_a, model_b)
    return {"cal_int": float(a[0]), "cal_int_se": float(np.sqrt(cov_a[0, 0])),
            "cal_slope": float(b[1]), "cal_slope_se": float(np.sqrt(cov_b[1, 1]))}


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
    Returns beta, the distinct event times and the Breslow cumulative baseline hazard at the covariate means."""
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

    beta = np.zeros(X.shape[1])
    r, s0, s1, shift, current = pieces(beta)
    for _ in range(iterations if X.shape[1] else 0):
        m = s1[events] / s0[events, None]
        gradient = x_events - deaths[events] @ m - ridge * beta
        cumulative = np.cumsum(np.where(events, deaths / s0, 0.0))[row_time]
        information = (X.T @ (X * (r * cumulative)[:, None]) - m.T @ (m * deaths[events, None])
                       + ridge * np.eye(X.shape[1]))
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
    return beta, times[events], np.cumsum(deaths[events] / (s0[events] * np.exp(shift)))


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
    Kaplan-Meier within each level of covariates[0] (a curve per level, risk 1)."""

    def __init__(self, frame, horizon, kind="cox", covariates=(), ridge=1.0):
        t, censored = censoring_window(frame.followup, frame.event_code, horizon)
        n = len(t)
        self.risk, self.curve = np.ones(n), np.zeros(n, np.int64)
        X = censoring_design(frame, covariates) if kind == "cox" else np.zeros((n, 0))
        if kind == "strata":
            _, self.curve = np.unique(frame[covariates[0]].astype(str).to_numpy(), return_inverse=True)
            self.curves = [self._km(t[self.curve == c], censored[self.curve == c]) for c in range(self.curve.max() + 1)]
        elif X.shape[1]:
            beta, times, base = cox_fit(t, censored, X, ridge)
            self.risk = np.exp((X - X.mean(axis=0)) @ beta)
            self.curves = [(times, base)]
        elif kind in ("cox", "km"):
            # Without covariates the censoring survival is the product-limit reverse Kaplan-Meier.
            self.curves = [self._km(t, censored)]
        else:
            raise ValueError(f"unknown censoring model {kind}")

    @staticmethod
    def _km(t, censored):
        times, g, _ = reverse_km(t, np.where(censored, 0, 1))
        return times, -np.log(np.maximum(g, 1e-300))

    def subset(self, mask):
        """The same fitted curves for a subset of the rows."""
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
# Aalen-Johansen and Wolbers' concordance
# --------------------------------------------------------------------------- #
def aalen_johansen(time, code, horizon):
    """The Aalen-Johansen cumulative incidence of cause 1 at a horizon, every other exit (death, 2, or an
    exclusion-rule exit, 3) competing, and its infinitesimal-jackknife standard error (R:
    survfit(Surv(time, factor(code)) ~ 1)$std.err).

    With h1, h the cause-1 and all-cause hazards, Y the risk set, d the events and S the all-cause Kaplan-Meier,
    F1(h) = sum_{u <= h} S(u-) h1(u). The derivative of F1(h) in row l's weight is
    sum_{v <= min(T_l, h)} a(v) B(v) - [l ended by h] B(T_l) / (Y - d)(T_l) + [l a case by h] S(T_l-) / Y(T_l)
    - sum_{u <= min(T_l, h)} S(u-) h1(u) / Y(u), with a = h / (Y - d) and B(v) = F1(h) - F1(v); the variance
    is the sum of its squares."""
    t, code = np.asarray(time, float), np.asarray(code, int)
    times, k, counts = np.unique(t, return_inverse=True, return_counts=True)
    d1 = np.bincount(k, weights=code == 1, minlength=len(times))
    d = np.bincount(k, weights=code != 0, minlength=len(times))
    at_risk = np.cumsum(counts[::-1])[::-1].astype(float)
    within = times <= horizon
    before = np.r_[1.0, np.cumprod(1 - d / at_risk)[:-1]]
    increment = np.where(within, before * d1 / at_risk, 0.0)
    cif = np.cumsum(increment)
    total = float(cif[-1])
    after = at_risk - d
    a = np.divide(d / at_risk, after, out=np.zeros(len(times)), where=after > 0)
    remaining = np.divide(total - cif, after, out=np.zeros(len(times)), where=after > 0)
    influence = (np.cumsum(np.where(within, a * (total - cif), 0.0))[k]
                 - np.cumsum(increment / at_risk)[k]
                 - np.where((code != 0) & within[k], remaining[k], 0.0)
                 + np.where((code == 1) & within[k], before[k] / at_risk[k], 0.0))
    return total, float(np.sqrt(np.sum(influence ** 2)))


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
    it (T_j > T_i, or T_j = T_i and j is not a case) or when j left by a competing exit (code 2 or 3) before
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
    # j left by a competing exit before the case: ascending time, cases before competing exits at a tied time
    # (one at the case's own time outlives it instead).
    died_order = np.lexsort((code >= 2, t))
    mass = np.where(code >= 2, inverse_left, 0.0)[died_order]
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


def truth_metrics(P, true_risk):
    """Accuracy against the simulator's true risk (SPEC section 5): per variant the RMSE, mean absolute error
    (the ICI against truth), bias and correlation of predicted against true risk."""
    rows = []
    for p in P:
        error = p - true_risk
        spread = p.std() * true_risk.std()
        rows.append({"rmse_true": float(np.sqrt(np.mean(error ** 2))), "mae_true": float(np.mean(np.abs(error))),
                     "bias_true": float(error.mean()),
                     "corr_true": float(np.mean((p - p.mean()) * (true_risk - true_risk.mean())) / spread)
                     if spread > 0 else float("nan")})
    return rows


def binary_cell(y, P, variants, fit, stratum, minimum, references=REFERENCES, pooled=None, true_risk=None):
    """Rows for one binary cell: y the outcomes, P the (variants x rows) predictions, pooled the pooled fit's
    predictions of the same variants for a LOGO cell, true_risk the simulator's p_ever."""
    n, cases = len(y), int(y.sum())
    base = {"model": "binary", "fit": fit, "stratum": stratum, "horizon": None, "n": n, "cases": cases}
    if min(n, cases, n - cases) < minimum:
        return [dict(base, variant=v, status=INSUFFICIENT) for v in variants]
    stacked = P if pooled is None else np.vstack([P, pooled])
    aucs, cov = delong(y, stacked)
    losses = (y[None, :] - stacked) ** 2
    truth = truth_metrics(P, true_risk) if true_risk is not None else [{} for _ in variants]
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


def survival_cell(t, code, w, censoring, P, variants, fit, stratum, horizon, minimum, references=REFERENCES,
                  pooled=None, truth=None):
    """Rows for one survival cell at a horizon: t the follow-up from entry; code 0 censored, 1 disease, 2 death,
    3 an exclusion-rule exit (2 and 3 compete); w the IPCW weights and `censoring` the evaluation's censoring
    model for these rows; P the (variants x rows) CIFs at the horizon.
    truth (simulator only): (true CIF at h, uncensored follow-up, uncensored event code) per row.

    Reported only where the prespecified support rule holds: at least `minimum` cases, known non-cases and
    rows followed past h; the cell's own G(h) upper bound at or above POSITIVITY_FLOOR; and every known row's
    weight at most 1/POSITIVITY_FLOOR."""
    y = ((code == 1) & (t <= horizon)).astype(float)
    known = w > 0
    n, cases, controls = len(t), int(y.sum()), int(np.sum(known & (y == 0)))
    base = {"model": "survival", "fit": fit, "stratum": stratum, "horizon": horizon, "n": n, "cases": cases}
    if min(n, cases, controls) < minimum:
        return [dict(base, variant=v, status=INSUFFICIENT) for v in variants]
    followed, g_upper = cell_support(t, code, horizon)
    if followed < minimum or g_upper < POSITIVITY_FLOOR or w.max() > 1 / POSITIVITY_FLOOR:
        return [dict(base, variant=v, status=UNSUPPORTED) for v in variants]
    observed, observed_se = aalen_johansen(t, code, horizon)
    shared = {"obs_risk": observed, "obs_risk_se": observed_se, "w_max": float(w.max()),
              "n_eff": float(w.sum() ** 2 / np.sum(w ** 2))}
    stacked = P if pooled is None else np.vstack([P, pooled])
    fits = [weighted_auc(p, y, w) for p in stacked]
    losses = w[None, :] * (y[None, :] - stacked) ** 2
    concordance = {}
    for name, model in (("c_harrell", None), ("c_uno", censoring)):
        try:
            concordance[name] = wolbers_concordance(t, code, P, horizon, model)
        except MetricRefusal as refusal:
            concordance[name] = refusal
    truth_rows = [{} for _ in variants]
    if truth is not None:
        true_cif, t_unc, code_unc = truth
        y_unc = ((code_unc == 1) & (t_unc <= horizon)).astype(float)
        truth_rows = truth_metrics(P, true_cif)
        for a, p in enumerate(P):
            truth_rows[a].update(mean_true=float(true_cif.mean()), risk_unc=float(y_unc.mean()),
                                 brier_unc=float(np.mean((y_unc - p) ** 2)))
            if 0 < y_unc.sum() < n:
                truth_rows[a]["auc_unc"] = weighted_auc(p, y_unc, np.ones(n))[0]
    rows = []
    for a, variant in enumerate(variants):
        p, (auc, influence) = P[a], fits[a]
        row = dict(base, variant=variant, status="ok", auc=auc, auc_se=float(np.sqrt(np.sum(influence ** 2)) / n),
                   brier=float(losses[a].mean()), brier_se=float(losses[a].std(ddof=1) / np.sqrt(n)),
                   mean_risk=float(p.mean()), **shared, **truth_rows[a])
        if observed > 0:
            oe, half = observed / row["mean_risk"], Z95 * observed_se / observed
            row.update(oe=oe, oe_lo=oe * math.exp(-half), oe_hi=oe * math.exp(half))
        _guarded(row, "calibration", lambda: calibration(y[known], p[known], w[known], robust=True))
        _guarded(row, "ici", lambda: {"ici": integrated_calibration_index(y, p, w)})
        for name, value in concordance.items():
            if isinstance(value, MetricRefusal):
                _refused(row, name, value)
            else:
                row[name] = float(value[a])
        others = [(ref, variants.index(ref)) for ref in references if ref in variants and ref != variant]
        others += [("pooled", len(variants) + a)] if pooled is not None else []
        for ref, b in others:
            _paired(row, ref, auc - fits[b][0], np.sqrt(np.sum((influence - fits[b][1]) ** 2)) / n,
                    losses[a] - losses[b])
        rows.append(row)
    return rows


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


def evaluate(kind, test, predictions, horizons, config, train=None, truth=None):
    """Every row of the single results table for one disease and model, from its outer-test rows.

    kind: "binary" or "survival". test: the outer-test frame (binary: `y`; survival: `followup`, `event` (0
    censored, 1 disease, 2 death, 3 an exclusion-rule exit; 2 and 3 compete) and `entry_age`; both: the axis and
    censoring columns, and admin_years). predictions: (variant, fit) ->
    risk per test row (binary) or rows x horizons CIF (survival), NaN outside the fit's rows; fit is "pooled"
    or "logo:<axis>:<group>". A LOGO row is the pooled cell of its held-out group, with stratum "overall".
    truth (simulator only): rows aligned with test carrying p_ever (binary) or cif_<h>y, uncensored_event and
    uncensored_exit_age (survival).

    At a survival horizon only rows whose potential follow-up reaches it are evaluated, and the censoring model
    is fitted on exactly those rows, separately for each LOGO group (SPEC section 8), so `train` is not used.
    config["evaluate"] may set "censoring" ("cox", "km" or "strata") and "censoring_covariates";
    config["report"]["small_cell_max"] is the largest count withheld."""
    settings = config.get("evaluate", {})
    minimum = int(config.get("report", {}).get("small_cell_max", SMALL_CELL_MAX)) + 1
    fits = _fits(predictions)
    variants = [v for v in ("ours", "covariates", "standard", "z_pc", "calpred") if v in fits]
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
        sub_truth = None if truth is None else truth.loc[mask_all].reset_index(drop=True)
        take = lambda variant, fit_name: np.asarray(predictions[(variant, fit_name)], float)[mask_all]
        pooled = None
        if fit != "pooled":
            pooled = {v: take(v, "pooled") for v in have if (v, "pooled") in predictions}
            have = [v for v in have if v in pooled]
        if kind == "binary":
            y = sub.y.to_numpy(int)
            if not np.isin(y, (0, 1)).all():
                raise ValueError("binary outcome must be 0/1")
            P_all = np.vstack([take(v, fit) for v in have])
            Q_all = None if pooled is None else np.vstack([pooled[v] for v in have])
            true_all = None if sub_truth is None else sub_truth.p_ever.to_numpy(float)
            for stratum, mask in cells(sub, strata):
                rows += binary_cell(y[mask], P_all[:, mask], have, fit, stratum, minimum,
                                    pooled=None if Q_all is None else Q_all[:, mask],
                                    true_risk=None if true_all is None else true_all[mask])
            continue
        potential = potential_followup(sub, config)
        for j, horizon in enumerate(float(h) for h in horizons):
            eligible = np.ones(len(sub), bool) if potential is None else potential >= horizon
            frame = sub.loc[eligible].rename(columns={"event": "event_code"}).reset_index(drop=True)
            t, code = frame.followup.to_numpy(float), frame.event_code.to_numpy(int)
            if not ((t > 0).all() and np.isin(code, (0, 1, 2, 3)).all()):
                raise ValueError("follow-up must be positive and events 0, 1, 2 or 3")
            model = Censoring(frame, horizon, settings.get("censoring", "cox"),
                              tuple(c for c in settings.get("censoring_covariates", CENSORING_COVARIATES)
                                    if c in frame))
            w = ipcw(frame, horizon, model)
            P_all = np.vstack([take(v, fit).reshape(len(sub), -1)[eligible, j] for v in have])
            Q_all = None if pooled is None else np.vstack([pooled[v].reshape(len(sub), -1)[eligible, j]
                                                           for v in have])
            truth_all = None
            if sub_truth is not None:
                known = sub_truth.loc[eligible].reset_index(drop=True)
                truth_all = (known[f"cif_{horizon:g}y"].to_numpy(float),
                             known.uncensored_exit_age.to_numpy(float) - frame.entry_age.to_numpy(float),
                             known.uncensored_event.to_numpy(int))
            for stratum, mask in cells(frame, strata):
                rows += survival_cell(t[mask], code[mask], w[mask], model.subset(mask), P_all[:, mask], have, fit,
                                      stratum, horizon, minimum, pooled=None if Q_all is None else Q_all[:, mask],
                                      truth=None if truth_all is None else tuple(v[mask] for v in truth_all))
    return rows
