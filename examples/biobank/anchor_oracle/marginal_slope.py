"""An anchored probit marginal-slope model, fitted by Fisher scoring in numpy.

The model of gam's Bernoulli marginal-slope family, stripped to what the
anchor theorems say: a person in context ``a`` with score ``z`` has

    P(Y = 1 | a, z) = Phi(alpha(a) + b(a) z),

where ``q(a)`` is the marginal index (``pi(a) = Phi(q(a))`` is the population
risk of the context), ``b(a)`` the slope, and ``alpha(a)`` is *defined* by the
anchoring equation on the declared law of ``z | a``
(``MarginalAnchor.exists_unique_anchor`` / ``ProbitAnchor``):

    sum_k w_k(a) Phi(alpha(a) + b(a) z_k(a)) = Phi(q(a)).

``q`` and ``b`` are linear in small bases of the context; the fit is the
unpenalised maximum-likelihood estimate of their coefficients.  Four anchors
are compared on the same data:

* ``gaussian-standard``: the closed form for ``z | a ~ N(0, 1)``,
  ``alpha = q sqrt(1 + b^2)`` -- gam's current lowering;
* ``gaussian-moments``: the closed form for ``N(mu_a, sigma_a^2)`` with the
  context's sample moments (``GaussianAnchor.gaussian_anchor_closed_form``);
* ``empirical``: gam's 65-node equal-mass grid of the context's training scores;
* ``empirical-full``: every training score of the context as a node;
* ``gauss-hermite-64``: a declared 64-node Gauss-Hermite law of ``N(0, 1)`` --
  the same law the closed form integrates, so this fit must reproduce
  ``gaussian-standard`` to quadrature tolerance (gam#2923, deliverable 3).

Data are generated with ``z | a`` standard normal, non-standard normal, or a
standardised shifted log-normal whose skew moves with the context, so the
``gaussian-standard`` anchor is exact, biased by the closed-form amount, or
biased by an amount only the anchoring equation itself predicts.  Running the
file writes ``results/marginal_slope.json``.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri

from anchor import (PROBIT, anchor_deriv, baseline_anchor_deriv, empirical_grid,
                    gauss_hermite, gaussian_anchor_closed_form, solve_anchor)

SQRT_2PI = math.sqrt(2.0 * math.pi)


def phi(x):
    return np.exp(-0.5 * np.square(x)) / SQRT_2PI


# ---------------------------------------------------------------------------
# Contexts, bases and the truth.

GRID = np.linspace(-1.0, 1.0, 4)
CONTEXTS = np.array([(a1, a2) for a1 in GRID for a2 in GRID])  # 16 contexts


def poly2(ctx):
    a1, a2 = ctx[:, 0], ctx[:, 1]
    return np.column_stack([np.ones_like(a1), a1, a2, a1 * a1, a2 * a2, a1 * a2])


def saturated(ctx):
    return np.eye(len(ctx))


BASES = {"poly2": poly2, "saturated": saturated}

THETA_Q_TRUE = np.array([-1.2, 0.4, -0.3, 0.0, 0.0, 0.2])
THETA_B_TRUE = np.array([0.5, 0.3, -0.2, 0.15, 0.0, 0.0])


def true_q_b():
    return poly2(CONTEXTS) @ THETA_Q_TRUE, poly2(CONTEXTS) @ THETA_B_TRUE


# ---------------------------------------------------------------------------
# Generators for z | a.  Each is a (sampler, law) pair; ``law(c, m)`` is the
# m-node equal-mass quantile law of z in context c (midpoint rule in probability
# space), used with m = 10^6 to compute the true anchor.

def quantile_law(quantile, m):
    u = (np.arange(m) + 0.5) / m
    return quantile(u), np.full(m, 1.0 / m)


def gen_standard(rng, c, n):
    return rng.standard_normal(n)


def law_standard(c, m):
    return quantile_law(ndtri, m)


def moments_nonstandard(c):
    a1, a2 = CONTEXTS[c]
    return 0.5 * a1 - 0.3 * a2, math.exp(0.3 * a1 + 0.2 * a2)


def gen_nonstandard(rng, c, n):
    mu, sigma = moments_nonstandard(c)
    return mu + sigma * rng.standard_normal(n)


def law_nonstandard(c, m):
    mu, sigma = moments_nonstandard(c)
    return quantile_law(lambda u: mu + sigma * ndtri(u), m)


def lognormal_shape(c):
    return 0.4 + 0.3 * CONTEXTS[c, 0]


def skewed_from_normal(g, s):
    """Standardised shifted log-normal: mean 0, variance 1, skew growing with s."""
    mean = math.exp(0.5 * s * s)
    sd = math.sqrt((math.exp(s * s) - 1.0) * math.exp(s * s))
    return (np.exp(s * g) - mean) / sd


def gen_skewed(rng, c, n):
    return skewed_from_normal(rng.standard_normal(n), lognormal_shape(c))


def law_skewed(c, m):
    return quantile_law(lambda u: skewed_from_normal(ndtri(u), lognormal_shape(c)), m)


GENERATORS = {"standard": (gen_standard, law_standard),
              "nonstandard": (gen_nonstandard, law_nonstandard),
              "skewed": (gen_skewed, law_skewed)}


def true_anchor(law, c, q, b, m=1_000_000):
    """The true intercept: ``exists_unique_anchor_probit`` on the generator's own
    law of ``z | a``, integrated on its ``m``-node quantile law."""
    nodes, weights = law(c, m)
    return solve_anchor(PROBIT, nodes, weights, lambda v: b * v, float(ndtr(q)))


def generate(rng, gen, n):
    sampler, law = GENERATORS[gen]
    q, b = true_q_b()
    ctx = rng.integers(len(CONTEXTS), size=n)
    z = np.empty(n)
    alpha = np.array([true_anchor(law, c, q[c], b[c]) for c in range(len(CONTEXTS))])
    for c in range(len(CONTEXTS)):
        sel = ctx == c
        z[sel] = sampler(rng, c, int(sel.sum()))
    p = ndtr(alpha[ctx] + b[ctx] * z)
    y = (rng.random(n) < p).astype(float)
    return ctx, z, y, p, alpha


# ---------------------------------------------------------------------------
# Anchors: each maps (q_c, b_c) to (alpha_c, d alpha/d q, d alpha/d b) on the law
# declared for context c from the training scores.

class GaussianAnchor:
    """``gaussian_anchor_closed_form``: ``alpha = q sqrt(1 + b^2 s^2) - b mu`` with
    its two partial derivatives, ``mu``/``s`` per context (zero/one for the
    standard form)."""

    def __init__(self, mu, sigma2):
        self.mu, self.sigma2 = mu, sigma2

    def __call__(self, c, q, b):
        mu, s2 = self.mu[c], self.sigma2[c]
        root = math.sqrt(1.0 + b * b * s2)
        return gaussian_anchor_closed_form(q, b, mu, s2), root, q * b * s2 / root - mu


class EmpiricalAnchor:
    """``solve_anchor`` on the context's declared grid, with ``baseline_anchor_deriv``
    for ``d alpha / d q`` and ``anchor_deriv`` (``h' = z``) for ``d alpha / d b``."""

    def __init__(self, grids):
        self.grids = grids

    def __call__(self, c, q, b):
        nodes, weights = self.grids[c]
        h = lambda v: b * v
        alpha = solve_anchor(PROBIT, nodes, weights, h, float(ndtr(q)))
        return (alpha, baseline_anchor_deriv(PROBIT, nodes, weights, h, alpha, q),
                anchor_deriv(PROBIT, nodes, weights, h, alpha, nodes))


def declare_anchor(kind, ctx, z):
    per = [z[ctx == c] for c in range(len(CONTEXTS))]
    if kind == "gaussian-standard":
        return GaussianAnchor(np.zeros(len(CONTEXTS)), np.ones(len(CONTEXTS)))
    if kind == "gaussian-moments":
        return GaussianAnchor(np.array([v.mean() for v in per]), np.array([v.var() for v in per]))
    if kind == "empirical":
        return EmpiricalAnchor([empirical_grid(v, 65) for v in per])
    if kind == "empirical-full":
        return EmpiricalAnchor([empirical_grid(v, v.size) for v in per])
    if kind == "gauss-hermite-64":
        return EmpiricalAnchor([gauss_hermite(0.0, 1.0, 64)] * len(CONTEXTS))
    raise ValueError(kind)


ANCHORS = ("gaussian-standard", "gaussian-moments", "empirical", "empirical-full", "gauss-hermite-64")


# ---------------------------------------------------------------------------
# The fit.

def log_likelihood(eta, y):
    return float(np.sum(np.where(y > 0.5, log_ndtr(eta), log_ndtr(-eta))))


def fit(ctx, z, y, anchor, Xq, Xb, max_iter=100, tol=1e-9):
    """Fisher scoring (IRLS) for theta = (theta_q, theta_b).  Per row
    ``eta = alpha_c + b_c z`` and ``d eta / d theta = (alpha_q X_q[c], (alpha_b + z) X_b[c])``;
    the expected information uses the Bernoulli probit weight ``phi^2 / (Phi (1 - Phi))``.
    Converges when the largest score component is below ``tol * n`` and the step
    is below ``tol`` in the maximum norm."""
    n, C = y.size, len(CONTEXTS)
    pq, pb = Xq.shape[1], Xb.shape[1]
    theta = np.zeros(pq + pb)
    theta[:pq] = np.linalg.lstsq(Xq, np.full(C, ndtri(max(1e-3, min(1 - 1e-3, y.mean())))), rcond=None)[0]

    def state(theta):
        q, b = Xq @ theta[:pq], Xb @ theta[pq:]
        parts = [anchor(c, q[c], b[c]) for c in range(C)]
        alpha = np.array([p[0] for p in parts])
        aq = np.array([p[1] for p in parts])
        ab = np.array([p[2] for p in parts])
        eta = alpha[ctx] + b[ctx] * z
        J = np.hstack([aq[ctx, None] * Xq[ctx], (ab[ctx] + z)[:, None] * Xb[ctx]])
        return eta, J, q, b, alpha

    eta, J, q, b, alpha = state(theta)
    ll = log_likelihood(eta, y)
    iters = 0
    for iters in range(1, max_iter + 1):
        p = ndtr(eta)
        d = phi(eta)
        var = np.exp(log_ndtr(eta) + log_ndtr(-eta))
        score = J.T @ ((y - p) * d / var)
        info = (J * (d * d / var)[:, None]).T @ J
        step = np.linalg.solve(info, score)
        for _ in range(40):
            cand = theta + step
            eta_c, J_c, q_c, b_c, alpha_c = state(cand)
            ll_c = log_likelihood(eta_c, y)
            if ll_c >= ll - 1e-10 * abs(ll):
                break
            step *= 0.5
        theta, eta, J, q, b, alpha, ll = cand, eta_c, J_c, q_c, b_c, alpha_c, ll_c
        if np.abs(score).max() < tol * n and np.abs(step).max() < tol:
            break
    p = ndtr(eta)
    d = phi(eta)
    var = np.exp(log_ndtr(eta) + log_ndtr(-eta))
    score = J.T @ ((y - p) * d / var)
    return {"theta": theta, "q": q, "b": b, "alpha": alpha, "loglik": ll, "iterations": iters,
            "max_abs_score": float(np.abs(score).max())}


def predict(fitted, ctx, z):
    return ndtr(fitted["alpha"][ctx] + fitted["b"][ctx] * z)


def evaluate(p_hat, ctx, y, p_true, q_true):
    """Held-out calibration-in-the-large per context, against the true marginal
    risk ``pi(a) = Phi(q(a))``, and log-loss / Brier of p_hat and of the truth."""
    pi = ndtr(q_true)
    mean_p = np.array([p_hat[ctx == c].mean() for c in range(len(CONTEXTS))])
    eps = 1e-300
    return {"mean_phat_minus_pi_by_context": (mean_p - pi).tolist(),
            "max_abs_mean_phat_minus_pi": float(np.abs(mean_p - pi).max()),
            "logloss": float(-np.mean(y * np.log(p_hat + eps) + (1 - y) * np.log(1 - p_hat + eps))),
            "logloss_true": float(-np.mean(y * np.log(p_true + eps) + (1 - y) * np.log(1 - p_true + eps))),
            "brier": float(np.mean(np.square(p_hat - y))),
            "brier_true": float(np.mean(np.square(p_true - y))),
            "rmse_vs_true_prob": float(math.sqrt(np.mean(np.square(p_hat - p_true))))}


def predicted_bias(gen):
    """What the theorems say the ``gaussian-standard`` fit's saturated baseline
    converges to.  The true intercept is ``alpha*_c`` (the anchor on the true
    law); the standard-form fit represents it as ``qhat_c sqrt(1 + b_c^2)``, so
    ``qhat_c - q_c = alpha*_c / sqrt(1 + b_c^2) - q_c``.  For a Gaussian law
    ``alpha*_c - q_c sqrt(1 + b_c^2) = q_c (sqrt(1 + b_c^2 s_c^2) - sqrt(1 + b_c^2)) - b_c mu_c``
    is the intercept-scale leak ``GaussianAnchor`` names; for the skewed law it is
    the same difference with ``alpha*_c`` from ``exists_unique_anchor`` on that law."""
    q, b = true_q_b()
    law = GENERATORS[gen][1]
    alpha = np.array([true_anchor(law, c, q[c], b[c]) for c in range(len(CONTEXTS))])
    root = np.sqrt(1.0 + b * b)
    out = {"alpha_true": alpha.tolist(), "intercept_leak": (alpha - q * root).tolist(),
           "predicted_q_bias": (alpha / root - q).tolist()}
    if gen == "nonstandard":
        mu = np.array([moments_nonstandard(c)[0] for c in range(len(CONTEXTS))])
        sigma = np.array([moments_nonstandard(c)[1] for c in range(len(CONTEXTS))])
        closed = q * (np.sqrt(1 + b * b * sigma * sigma) - root) - b * mu
        out["intercept_leak_closed_form"] = closed.tolist()
        out["max_abs_closed_form_minus_quadrature"] = float(np.abs(closed - (alpha - q * root)).max())
    return out


def main(argv):
    out_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(float(s)) for s in argv[2].split(",")] if len(argv) > 2 else [20000, 300000]
    n_test = 200000
    q_true, b_true = true_q_b()
    results = {"contexts": CONTEXTS.tolist(), "q_true": q_true.tolist(), "b_true": b_true.tolist(),
               "n_test": n_test, "runs": []}
    for gen in GENERATORS:
        results.setdefault("predicted_bias", {})[gen] = predicted_bias(gen)
        rng = np.random.default_rng(20260916)
        ctx_te, z_te, y_te, p_te, _ = generate(rng, gen, n_test)
        for n in sizes:
            ctx, z, y, _, _ = generate(rng, gen, n)
            for basis in BASES:
                Xq, Xb = BASES[basis](CONTEXTS), poly2(CONTEXTS)
                fits = {}
                for kind in ANCHORS:
                    t0 = time.perf_counter()
                    anchor = declare_anchor(kind, ctx, z)
                    fitted = fit(ctx, z, y, anchor, Xq, Xb)
                    wall = time.perf_counter() - t0
                    fits[kind] = fitted
                    ev = evaluate(predict(fitted, ctx_te, z_te), ctx_te, y_te, p_te, q_true)
                    run = {"generator": gen, "n": n, "basis": basis, "anchor": kind,
                           "wall_seconds": wall, "iterations": fitted["iterations"],
                           "max_abs_score": fitted["max_abs_score"], "loglik": fitted["loglik"],
                           "theta": fitted["theta"].tolist(),
                           "qhat_minus_q_by_context": (fitted["q"] - q_true).tolist(),
                           "max_abs_qhat_minus_q": float(np.abs(fitted["q"] - q_true).max()),
                           "max_abs_bhat_minus_b": float(np.abs(fitted["b"] - b_true).max()),
                           "Phi_qhat_minus_pi_by_context": (ndtr(fitted["q"]) - ndtr(q_true)).tolist(),
                           "max_abs_Phi_qhat_minus_pi": float(np.abs(ndtr(fitted["q"]) - ndtr(q_true)).max()),
                           "heldout": ev}
                    if basis == "saturated" and kind == "gaussian-standard":
                        pred = np.array(results["predicted_bias"][gen]["predicted_q_bias"])
                        run["max_abs_qbias_minus_predicted"] = float(np.abs(fitted["q"] - q_true - pred).max())
                    if kind != "gaussian-standard":
                        ref = fits["gaussian-standard"]
                        run["vs_gaussian_standard"] = {
                            "max_abs_dtheta": float(np.abs(fitted["theta"] - ref["theta"]).max()),
                            "max_abs_dq": float(np.abs(fitted["q"] - ref["q"]).max()),
                            "max_abs_dalpha": float(np.abs(fitted["alpha"] - ref["alpha"]).max()),
                            "abs_dloglik": abs(fitted["loglik"] - ref["loglik"]),
                            "max_abs_dpred": float(np.abs(predict(fitted, ctx_te, z_te)
                                                          - predict(ref, ctx_te, z_te)).max())}
                    results["runs"].append(run)
                    print(f"{gen:12s} n={n:7d} {basis:9s} {kind:17s} it={fitted['iterations']:3d} "
                          f"{wall:6.2f}s max|q^-q|={run['max_abs_qhat_minus_q']:.4f} "
                          f"max|b^-b|={run['max_abs_bhat_minus_b']:.4f} "
                          f"max|E[p^|a]-pi|={ev['max_abs_mean_phat_minus_pi']:.5f} "
                          f"max|Phi(q^)-pi|={run['max_abs_Phi_qhat_minus_pi']:.5f} "
                          f"ll={ev['logloss']:.6f} (true {ev['logloss_true']:.6f})", flush=True)
    (out_dir / "marginal_slope.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv)
