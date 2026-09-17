"""The marginal anchor on a declared finite predictor law, read off Descent.

A marginally interpretable binary model fixes the population risk ``pi`` of a
context and lets the predictor shape ``h(v)`` move each person around it,
``p(v) = L(a + h(v))``.  The intercept ``a`` is defined by the anchoring
equation ``E_p[L(a + h(v))] = pi`` under the declared law ``p`` of the
predictor in that context.  This file is a literal numpy reading of
``Descent.Portability.MarginalAnchor`` (any link), ``ProbitAnchor`` (the probit
instance) and ``GaussianAnchor`` (the closed form under ``N(mu, sigma^2)``),
short enough to be checked by hand against those docstrings.  Run it as a
script to execute the four checks the design needs and write
``results/anchor.json``.

A finite law is a pair ``(nodes, weights)``: ``weights`` are non-negative and sum
to one (``FiniteReportLaw``), ``nodes`` are the predictor values ``v`` (any
shape whose first axis indexes the nodes).  A predictor shape ``h`` is a
callable on ``nodes`` returning one number per node (``h : V -> R``).
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, log_ndtr, ndtr, ndtri

SQRT_2PI = math.sqrt(2.0 * math.pi)


def _phi(x):
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


def _logit_weight(x):
    s = expit(x)
    return s * (1.0 - s)


def _logit_quantile(p):
    return np.log(p) - np.log1p(-p)


class Link:
    """A link ``L`` with its derivative ``W = L'`` (``hasDerivAt``), its inverse,
    and the Bernoulli Fisher weight ``W^2 / (L (1 - L))``.  For the logit link
    ``W`` is the Bernoulli weight ``sigma (1 - sigma)`` and coincides with the
    Fisher weight (``MarginalAnchor.bernoulliWeight``); for the probit link ``W``
    is the standard normal density ``phi`` and does not (``ProbitAnchor``)."""

    def __init__(self, name, cdf, deriv, quantile, fisher):
        self.name = name
        self.cdf = cdf
        self.deriv = deriv
        self.quantile = quantile
        self._fisher = fisher

    def fisher_weight(self, x):
        return self._fisher(x)


def _probit_fisher(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-np.square(x) - math.log(2.0 * math.pi) - log_ndtr(x) - log_ndtr(-x))


LOGIT = Link("logit", expit, _logit_weight, _logit_quantile, _logit_weight)
PROBIT = Link("probit", ndtr, _phi, ndtri, _probit_fisher)
LINKS = {"logit": LOGIT, "probit": PROBIT}


def anchored_mean(link, nodes, weights, h, a):
    """``MarginalAnchor.anchoredMean``: ``sum_v p_v L(a + h(v))``."""
    return float(np.dot(weights, link.cdf(a + h(nodes))))


def solve_anchor(link, nodes, weights, h, pi, max_iter=100):
    """The unique root of ``anchored_mean(.) = pi`` (``exists_unique_anchor``,
    ``anchor_spec``).  ``L(a + min h) <= anchored mean <= L(a + max h)`` gives
    the exact bracket ``[L^{-1}(pi) - max h, L^{-1}(pi) - min h]``; inside it the
    anchored mean is continuous and strictly increasing (``anchoredMean_strictMono``)
    with derivative ``sum_v p_v W(a + h(v)) > 0`` (``sum_mass_weight_pos``), so
    Newton steps kept inside the bracket, with bisection when one leaves it,
    converge to the root."""
    if not 0.0 < pi < 1.0:
        raise ValueError("pi must lie in (0, 1)")
    hv = np.asarray(h(nodes), dtype=float)
    w = np.asarray(weights, dtype=float)
    lq = float(link.quantile(pi))
    lo, hi = lq - float(hv.max()), lq - float(hv.min())
    a = 0.5 * (lo + hi)
    for _ in range(max_iter):
        eta = a + hv
        f = float(np.dot(w, link.cdf(eta))) - pi
        if f == 0.0:
            return a
        if f < 0.0:
            lo = a
        else:
            hi = a
        slope = float(np.dot(w, link.deriv(eta)))
        a_new = a - f / slope if slope > 0.0 else 0.5 * (lo + hi)
        if not lo < a_new < hi:
            a_new = 0.5 * (lo + hi)
        if abs(a_new - a) <= 1e-15 * (1.0 + abs(a)):
            return a_new
        a = a_new
    return a


def anchor_deriv(link, nodes, weights, h, a, dh):
    """``MarginalAnchor.anchor_deriv_eq`` (``anchor_deriv_eq_logit``,
    ``ProbitAnchor.anchor_deriv_eq_probit``): along a path on which the anchor
    holds, ``a' = -sum_v p_v W(a + h(v)) h'(v) / sum_v p_v W(a + h(v))``.  ``dh``
    is ``h'`` evaluated at the nodes."""
    wv = np.asarray(weights, dtype=float) * link.deriv(a + h(nodes))
    return -float(np.dot(wv, dh)) / float(wv.sum())


def baseline_anchor_deriv(link, nodes, weights, h, a, q):
    """``MarginalAnchor.baseline_anchor_deriv``: moving the target risk
    ``pi = L(q)`` with ``h`` fixed, ``a_q sum_v p_v W(a + h(v)) = W(q)``."""
    return float(link.deriv(q)) / float(np.dot(weights, link.deriv(a + h(nodes))))


def cross_information(weight_fn, weights, eta, s1, s2):
    """``MarginalAnchor.crossInformation``: ``sum_v p_v W(eta_v) s1(v) s2(v)``
    for a weight function ``W`` (the link derivative in the theorems; the
    Bernoulli Fisher weight is a different ``W`` for the probit link)."""
    return float(np.sum(np.asarray(weights) * weight_fn(eta) * s1 * s2))


def gaussian_anchor_closed_form(q, b, mu, sigma2):
    """``GaussianAnchor.gaussian_anchor_closed_form`` / ``gaussian_anchor_unique``:
    under ``z ~ N(mu, sigma^2)`` the probit anchor of ``h(z) = b z`` at target
    ``Phi(q)`` is ``q sqrt(1 + b^2 sigma^2) - b mu``.  ``drive_anchor_closed_form``
    is the same formula with ``b mu`` and ``b^2 sigma^2`` read as the mean and
    variance of the whole Gaussian drive."""
    return q * np.sqrt(1.0 + b * b * sigma2) - b * mu


def gauss_hermite(mu, sigma, m):
    """An ``m``-node Gauss-Hermite law for ``N(mu, sigma^2)``."""
    x, w = hermegauss(m)
    return mu + sigma * x, w / w.sum()


def empirical_grid(z, m=65):
    """Equal-mass compression of a sample into an ``m``-node law, as gam's
    ``EmpiricalZGrid`` builder does: sort, cut into ``m`` equal-count bins, one
    node per bin at the bin mean.  ``m >= len(z)`` returns the sample itself."""
    z = np.sort(np.asarray(z, dtype=float))
    if m >= z.size:
        return z, np.full(z.size, 1.0 / z.size)
    edges = np.linspace(0, z.size, m + 1).round().astype(int)
    nodes = np.array([z[lo:hi].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    counts = np.diff(edges).astype(float)
    return nodes, counts / counts.sum()


# ---------------------------------------------------------------------------
# Checks.  Each returns a JSON-serialisable dict of the numbers it measured.


def random_law(rng):
    m = int(rng.integers(2, 200))
    kind = rng.integers(3)
    if kind == 0:
        nodes = rng.normal(0.0, 2.0, m)
    elif kind == 1:
        nodes = rng.standard_t(2.0, m)
    else:
        nodes = np.concatenate([rng.normal(-3, 0.3, m // 2), rng.normal(2, 1.0, m - m // 2)])
    weights = rng.dirichlet(np.full(m, 0.5))
    return nodes, weights


def random_shape(rng):
    b0, b1, b2 = rng.uniform(-2, 2), rng.uniform(-1, 1), rng.uniform(-0.5, 0.5)
    return lambda v: b0 * v + b1 * np.tanh(v) + b2 * v * v


def check_existence_uniqueness(rng, cases=2000):
    """(i) ``exists_unique_anchor``: on random finite laws the bracketed Newton
    root satisfies the equation to rounding, and an independent pure bisection
    on the same bracket lands on the same number."""
    out = {}
    for name, link in LINKS.items():
        worst_resid, worst_gap, worst_mono = 0.0, 0.0, 0.0
        for _ in range(cases):
            nodes, weights = random_law(rng)
            h = random_shape(rng)
            pi = float(rng.uniform(0.001, 0.999))
            a = solve_anchor(link, nodes, weights, h, pi)
            resid = abs(anchored_mean(link, nodes, weights, h, a) - pi)
            hv = h(nodes)
            lo, hi = link.quantile(pi) - hv.max(), link.quantile(pi) - hv.min()
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if anchored_mean(link, nodes, weights, h, mid) < pi:
                    lo = mid
                else:
                    hi = mid
            gap = abs(0.5 * (lo + hi) - a)
            delta = 1e-6 * (1.0 + abs(a))
            mono = (anchored_mean(link, nodes, weights, h, a - delta) < pi
                    < anchored_mean(link, nodes, weights, h, a + delta))
            worst_resid = max(worst_resid, resid)
            worst_gap = max(worst_gap, gap)
            worst_mono = max(worst_mono, 0.0 if mono else 1.0)
        out[name] = {"cases": cases, "max_abs_residual": worst_resid,
                     "max_abs_gap_vs_bisection": worst_gap,
                     "strict_monotone_violations": worst_mono}
    return out


def central_difference(f, x, eps):
    """Five-point central difference, truncation error ``O(eps^4)``."""
    return (f(x - 2 * eps) - 8 * f(x - eps) + 8 * f(x + eps) - f(x + 2 * eps)) / (12 * eps)


def check_anchor_deriv_fd(rng, cases=300, eps=1e-4):
    """(ii) ``anchor_deriv_eq`` against central finite differences of
    ``solve_anchor`` along random paths ``h(theta, v) = h0 + theta h1 + sin(theta) h2``."""
    out = {}
    for name, link in LINKS.items():
        worst = 0.0
        for _ in range(cases):
            nodes, weights = random_law(rng)
            h0, h1, h2 = random_shape(rng), random_shape(rng), random_shape(rng)
            pi = float(rng.uniform(0.01, 0.99))
            theta = float(rng.uniform(-1, 1))
            path = lambda t: (lambda v: h0(v) + t * h1(v) + math.sin(t) * h2(v))
            a = solve_anchor(link, nodes, weights, path(theta), pi)
            analytic = anchor_deriv(link, nodes, weights, path(theta), a,
                                    h1(nodes) + math.cos(theta) * h2(nodes))
            fd = central_difference(lambda t: solve_anchor(link, nodes, weights, path(t), pi), theta, eps)
            worst = max(worst, abs(analytic - fd) / max(1e-3, abs(fd)))
        out[name] = {"cases": cases, "fd_step": eps, "max_rel_err": worst}
    return out


def check_gaussian_closed_form(rng, cases=400, sizes=(16, 32, 64, 128)):
    """(iii) ``gaussian_anchor_closed_form``: on a Gauss-Hermite grid of
    ``N(mu, sigma^2)`` the solved probit anchor equals ``q sqrt(1 + b^2 sigma^2) - b mu``
    to quadrature tolerance.  The tolerance is governed by the drive's standard
    deviation ``|b| sigma`` (a steep drive makes ``Phi(a + b z)`` step-like on the
    grid), so it is reported per grid size and per ``|b| sigma`` bucket."""
    draws = [(rng.uniform(-2.5, 2.5), rng.uniform(-2, 2), rng.uniform(-1, 1), rng.uniform(0.5, 2.0))
             for _ in range(cases)]
    scale = np.array([abs(b) * sigma for _, b, _, sigma in draws])
    buckets = [(0.0, 1.0), (1.0, 2.0), (2.0, 4.0)]
    out = {}
    for m in sizes:
        errs = []
        for q, b, mu, sigma in draws:
            nodes, weights = gauss_hermite(mu, sigma, m)
            a = solve_anchor(PROBIT, nodes, weights, lambda v: b * v, float(ndtr(q)))
            errs.append(abs(a - gaussian_anchor_closed_form(q, b, mu, sigma * sigma)))
        errs = np.array(errs)
        out[str(m)] = {"max_abs_err": float(errs.max()), "median_abs_err": float(np.median(errs))}
        for lo, hi in buckets:
            sel = (scale > lo) & (scale <= hi)
            out[str(m)][f"max_abs_err_drive_sd_{lo:g}_{hi:g}"] = float(errs[sel].max())
    return out


def check_cross_information(rng, cases=300, eps=1e-5):
    """(iv) ``crossInformation_baseline_shape_zero`` (logit: ``fisher_baseline_shape_zero``;
    probit: ``crossInformation_baseline_shape_zero_probit``): the ``W``-weighted
    cross-information between the constant baseline direction ``a_q`` and the
    anchored shape direction ``a' + h'`` is zero.  For the probit link the true
    Bernoulli Fisher weight is ``phi^2 / (Phi (1 - Phi))``, not ``phi``, and the
    Fisher cross block is not zero; its size relative to the shape direction's
    own Fisher norm is reported."""
    out = {}
    for name, link in LINKS.items():
        link_w, link_w_fd, fisher_w = 0.0, 0.0, 0.0
        for _ in range(cases):
            nodes, weights = random_law(rng)
            h0, h1 = random_shape(rng), random_shape(rng)
            q = float(rng.uniform(-2, 2))
            pi = float(link.cdf(q))
            path = lambda t: (lambda v: h0(v) + t * h1(v))
            a = solve_anchor(link, nodes, weights, path(0.0), pi)
            eta = a + h0(nodes)
            a_q = baseline_anchor_deriv(link, nodes, weights, h0, a, q)
            shape = anchor_deriv(link, nodes, weights, h0, a, h1(nodes)) + h1(nodes)
            fd = (solve_anchor(link, nodes, weights, path(eps), pi)
                  - solve_anchor(link, nodes, weights, path(-eps), pi)) / (2 * eps)
            shape_fd = fd + h1(nodes)
            base = np.full(nodes.size, a_q)
            norm = math.sqrt(cross_information(link.fisher_weight, weights, eta, shape, shape)
                             * cross_information(link.fisher_weight, weights, eta, base, base))
            link_w = max(link_w, abs(cross_information(link.deriv, weights, eta, base, shape)))
            link_w_fd = max(link_w_fd, abs(cross_information(link.deriv, weights, eta, base, shape_fd)))
            fisher_w = max(fisher_w, abs(cross_information(link.fisher_weight, weights, eta, base, shape)) / norm)
        out[name] = {"cases": cases,
                     "max_abs_link_weighted_cross_information": link_w,
                     "max_abs_link_weighted_cross_information_fd_deriv": link_w_fd,
                     "max_rel_bernoulli_fisher_cross_information": fisher_w}
    return out


def main(argv):
    out_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260916)
    results = {}
    for key, fn in [("existence_uniqueness", check_existence_uniqueness),
                    ("anchor_deriv_vs_fd", check_anchor_deriv_fd),
                    ("gaussian_closed_form_vs_gauss_hermite", check_gaussian_closed_form),
                    ("cross_information", check_cross_information)]:
        t0 = time.perf_counter()
        results[key] = fn(rng)
        results[key]["wall_seconds"] = time.perf_counter() - t0
        print(key, json.dumps(results[key], indent=1))
    (out_dir / "anchor.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv)
