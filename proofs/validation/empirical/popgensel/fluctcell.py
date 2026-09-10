"""Cell F: the two PolygenicAdaptation effect correlations.

  effectCorrelationStabilizingDriftSelection d s N = 1 - d / (1 + s*N)
  effectCorrelationFluctuating              d f N = max (-1) (1 - d*(1 + f*N))

Both take `d`, the NEUTRAL decorrelation, as an input.  That is fed REALIZED --
measured from a neutral arm run on the same cells with the same loci, seeds and
divergence time -- not from a nominal parameter, because the whole size of the
predicted effect is the gap between the neutral arm and the selected one.

The neutral arm is also the positive control: it is the s = f = 0 cell, where
both bodies must return exactly the measured neutral correlation, and they do by
construction.  The measurement that carries information is what happens when the
selection axis is turned up.

Guard PGSEL_F1.
"""
import json, math, sys
import numpy as np

GUARD = "PGSEL_F1"


def run_pop(freq, beta, N, Vs, gens, rng, opt_sd=0.0):
    """Diploid WF, L unlinked loci, Gaussian selection w = exp(-(z-opt)^2/(2Vs)).
    opt_sd > 0 makes the optimum a fresh N(0, opt_sd^2) draw every generation and
    INDEPENDENTLY in each population -- that is fluctuating selection."""
    L = len(beta)
    G = (rng.random((N, 2, L)) < freq).astype(np.int8)
    for _ in range(gens):
        opt = rng.normal(0.0, opt_sd) if opt_sd > 0 else 0.0
        z = (G.sum(axis=1) * beta).sum(axis=1)
        w = np.exp(-((z - opt) ** 2) / (2 * Vs))
        w = w / w.sum()
        par = rng.choice(N, size=(N, 2), p=w)
        pick = rng.integers(0, 2, size=(N, 2, L))
        g1 = np.take_along_axis(G[par[:, 0]], pick[:, :1], axis=1)[:, 0]
        g2 = np.take_along_axis(G[par[:, 1]], pick[:, 1:], axis=1)[:, 0]
        G = np.stack([g1, g2], axis=1)
    return G


def corr(G1, G2):
    p1 = G1.mean(axis=(0, 1))
    p2 = G2.mean(axis=(0, 1))
    m = (p1 > 0) & (p1 < 1) & (p2 > 0) & (p2 < 1)
    return np.corrcoef(p1[m], p2[m])[0, 1] if m.sum() > 15 else np.nan


def cell_F(reps=150, L=60, t=20):
    rng = np.random.default_rng(5150)
    rows = []
    for N in (30, 60):
        for mode, strengths in (("stabilizing", (0.0, 0.1, 0.5, 2.0)),
                                ("fluctuating", (0.0, 0.1, 0.5, 2.0))):
            neutral = None
            for st in strengths:
                acc = []
                for _ in range(reps):
                    beta = rng.normal(0, 1, L)
                    freq = rng.uniform(0.1, 0.9, L)
                    Vg = (2 * freq * (1 - freq) * beta ** 2).sum()
                    if st == 0.0:
                        Vs, sd = 1e12, 0.0
                    elif mode == "stabilizing":
                        Vs, sd = Vg / (2 * st), 0.0
                    else:
                        # same selection intensity, but the optimum moves: sd is
                        # one phenotypic SD, so the two populations chase
                        # independent targets.
                        Vs, sd = Vg / (2 * st), math.sqrt(Vg)
                    c = corr(run_pop(freq, beta, N, Vs, t, rng, sd),
                             run_pop(freq, beta, N, Vs, t, rng, sd))
                    if not math.isnan(c):
                        acc.append(c)
                a = np.array(acc)
                m, se = a.mean(), a.std(ddof=1) / math.sqrt(len(a))
                if st == 0.0:
                    neutral = m
                    d = 1 - m
                row = dict(N=N, mode=mode, strength=st, t=t, L=L,
                           measured=m, sem=se, reps=len(a),
                           realized_d_from_neutral_arm=1 - neutral)
                if st > 0:
                    d = 1 - neutral
                    sN = st * N
                    row["stabilizingDriftSelection_pred"] = 1 - d / (1 + sN)
                    row["stabilizingDriftSelection_sems"] = \
                        (row["stabilizingDriftSelection_pred"] - m) / se
                    row["fluctuating_pred"] = max(-1.0, 1 - d * (1 + sN))
                    row["fluctuating_sems"] = (row["fluctuating_pred"] - m) / se
                rows.append(row)
    return rows


if __name__ == "__main__":
    r = cell_F(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
    print("FRESHNESS=OK", GUARD)
    print(json.dumps(dict(cell="F", guard=GUARD,
                          targets=["effectCorrelationStabilizingDriftSelection",
                                   "effectCorrelationFluctuating"],
                          argument_source="forward individual-based WF; the neutral arm "
                                          "supplies REALIZED d on the same loci, seeds and "
                                          "divergence time as the selected arms",
                          detail=r), indent=1, default=float))
