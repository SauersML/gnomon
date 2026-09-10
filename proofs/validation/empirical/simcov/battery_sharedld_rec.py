"""Shared LD between demes is set by migration racing recombination.

`sharedLD_from_equilibrium` used to read `1 - F_ST`, and `battery_bulk34.py`
falsified that at 35 sems: measured shared LD stayed near 0.9 where `1 - F_ST`
predicted 0.44. No function of `F_ST` alone can be right, and this battery shows
why in one line: at FIXED `F_ST` -- one cell, one pair of demes, one replicate
set -- the measured shared fraction runs from 0.97 down to 0.06 as the SNP pairs
are sorted by how far apart they sit. The missing argument is the recombination
rate between the two sites, and no exponent on `F_ST` can supply it.

THE DERIVATION (Sved 2009 eqns 5-6, 9, 15; the same recursion as Ohta 1982).
Write `sigma_W` for the within-deme second moment of the disequilibrium `D` and
`sigma_B` for the between-deme one. Both decay by `(1-c)^2` per generation --
one factor of `(1-c)` for each deme -- and drift feeds only the within-deme
moment, since drift in one deme is independent of drift in the other:

    sigma_W' = (1-c)^2 [(1-alpha) sigma_W + alpha sigma_B] + 1/(2Ne)
    sigma_B' = (1-c)^2 [beta sigma_W + (1-beta) sigma_B]

`beta = 2m(1-m)` is the chance that two gametes now in different demes sat in
the same deme one generation ago. The second line has no source term, and that
is the whole content: between demes there is nothing to renew the association,
so at stationarity

    sharedLD = sigma_B / sigma_W = (1-c)^2 beta / [1 - (1-c)^2 (1-beta)]
             -> 2m / (2m + 2c) = m / (m + c)

for small `m` and `c`. The coalescent reading is the same race: two lineages in
different demes cannot coalesce at all until migration puts them together, and
recombination -- rate `2c`, one lineage each -- is running against that
migration. `Ne` cancels: it sets how much LD there is, not how much of it is
shared. `m` is the backward rate between one ORDERED pair of demes, so the deme
count drops out too (a move to a third deme leaves the lineages apart).

Limits, all four of which `1 - F_ST` gets wrong: `c -> 0` gives 1 (loci that
never recombine apart stay shared however far the frequencies drift), `c >> m`
gives 0, `m -> 0` gives 0, `m -> infinity` gives 1.

THE OBSERVABLE. The recursion is a moment system in `D`, so the oracle is the
`D`-ratio, and the `r`-normalised version is reported beside it because they are
NOT the same number: dividing by the frequency term inflates the shared fraction
by 5-25 percent, which is the size of the effect being measured. Sved's step
`E[r^2] = L` is the approximate one and this is where its cost shows.

THE INSTRUMENT. A plain correlation of per-deme estimates is not an estimate of
`sigma_B/sigma_W`: the denominator carries sampling noise the numerator does
not, and the attenuation grows exactly where the signal decays, so a naive
version falls off with distance whatever the truth is. Each deme's samples are
therefore split into two disjoint halves and

    ratio = mean(D_0a D_1a) / sqrt(mean(D_0a D_0b) * mean(D_1a D_1b))

Every product is between INDEPENDENT sample sets, so each estimates a noise-free
expectation. The positive control -- one panmictic population split four ways --
must return 1.00 at every distance and CAN fail; under the old correlation
estimator it returned 0.9945 and the shortfall was pure attenuation.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simlib
from battery_core import RESULTS, record, dump_results

GUARD = "SHAREDLD-REC-v5-Dratio-splithalf"

SEQ = 1e7
RHO = 1e-8
MU = 1e-8
N_DIP = 100          # diploid individuals per deme -> 200 nodes -> two halves
REPS = 12
MAX_SITES = 900
BIN_EDGES = np.array([2e3, 8e3, 3e4, 1e5, 3e5, 1e6, 3e6])

# Ne varies threefold at fixed m: the derivation says the shared fraction does
# not move, and every F_ST-based form says it moves a long way.
CELLS = [(2000, 1e-3), (2000, 4e-3), (2000, 2.5e-4), (1000, 1e-3), (4000, 1e-3)]


def moments(gm, cols):
    """Gametic covariance `D` and correlation `r` between every pair of sites."""
    G = gm[:, cols].astype(np.float64)
    G -= G.mean(axis=1, keepdims=True)
    D = (G @ G.T) / G.shape[1]
    nrm = np.sqrt(np.diag(D)).copy()
    nrm[nrm <= 0] = np.inf
    return D, D / np.outer(nrm, nrm)


def replicate(Ne, m, seed, panmictic=False, scheme="both"):
    """One replicate -> per-bin sums of the six products, plus F_ST."""
    import msprime
    if panmictic:
        ts = msprime.sim_ancestry(
            samples=2 * N_DIP, population_size=Ne, sequence_length=SEQ,
            recombination_rate=RHO, random_seed=seed)
    else:
        dem = msprime.Demography.island_model([Ne, Ne], migration_rate=m)
        ts = msprime.sim_ancestry(
            samples={"pop_0": N_DIP, "pop_1": N_DIP}, demography=dem,
            sequence_length=SEQ, recombination_rate=RHO, random_seed=seed)
    mts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 500000)
    gm = mts.genotype_matrix()
    pos = np.asarray(mts.tables.sites.position)
    if panmictic:
        allnodes = np.arange(4 * N_DIP)
        A, B = allnodes[:2 * N_DIP], allnodes[2 * N_DIP:]
    else:
        A, B = mts.samples(population=0), mts.samples(population=1)
    fa, fb = gm[:, A].mean(axis=1), gm[:, B].mean(axis=1)
    if scheme == "both":
        keep = np.where((fa > 0.05) & (fa < 0.95) & (fb > 0.05) & (fb < 0.95))[0]
    else:                                     # pooled: not selected per deme
        fp = gm.mean(axis=1)
        keep = np.where((fp > 0.05) & (fp < 0.95))[0]
    if keep.size < 100:
        return None
    rng = np.random.default_rng(seed)
    if keep.size > MAX_SITES:
        keep = np.sort(rng.choice(keep, MAX_SITES, replace=False))
    fst = simlib.hudson_fst(gm[keep][:, A].sum(1).astype(float), len(A),
                            gm[keep][:, B].sum(1).astype(float), len(B))
    # disjoint halves WITHIN each deme: a denominator that shares samples with
    # itself re-imports the sampling noise the split is there to remove.
    pa, pb = rng.permutation(len(A)), rng.permutation(len(B))
    A0, A1 = A[pa[:len(A) // 2]], A[pa[len(A) // 2:]]
    B0, B1 = B[pb[:len(B) // 2]], B[pb[len(B) // 2:]]
    G = gm[keep]
    Da0, Ra0 = moments(G, A0)
    Da1, Ra1 = moments(G, A1)
    Db0, Rb0 = moments(G, B0)
    Db1, Rb1 = moments(G, B1)
    d = np.abs(pos[keep][:, None] - pos[keep][None, :])
    iu = np.triu_indices(len(keep), k=1)
    p = [(Da0 * Db0)[iu], (Da0 * Da1)[iu], (Db0 * Db1)[iu],
         (Ra0 * Rb0)[iu], (Ra0 * Ra1)[iu], (Rb0 * Rb1)[iu]]
    dd = d[iu]
    ok = np.all([np.isfinite(v) for v in p], axis=0)
    idx = np.digitize(dd, BIN_EDGES) - 1
    out = np.zeros((len(BIN_EDGES) - 1, 8))
    for b in range(len(BIN_EDGES) - 1):
        s = ok & (idx == b)
        if s.sum():
            out[b] = tuple(v[s].sum() for v in p) + (s.sum(),
                                                     (RHO * dd[s]).sum())
    return out, fst


def ratio_from(sums, off):
    tot = sums.sum(axis=0)
    n, a, b = tot[off], tot[off + 1], tot[off + 2]
    if a <= 0 or b <= 0:
        return float("nan")
    return float(n / math.sqrt(a * b))


def jackknife(sums, off=0):
    """Ratio of averages, with a delete-one-replicate jackknife error bar."""
    R = sums.shape[0]
    full = ratio_from(sums, off)
    if R < 2:
        return full, float("nan")
    part = np.array([ratio_from(np.delete(sums, i, axis=0), off)
                     for i in range(R)])
    part = part[np.isfinite(part)]
    if part.size < 2:
        return full, float("nan")
    var = (part.size - 1) / part.size * ((part - part.mean()) ** 2).sum()
    return full, math.sqrt(var)


def run_cells(scheme):
    """Every (cell, distance bin) under one ascertainment."""
    rows = []
    for (Ne, m) in CELLS:
        acc, fsts = [], []
        for rep in range(REPS):
            got = replicate(Ne, m, 91000 + 137 * rep + int(1e7 * m) + Ne,
                            scheme=scheme)
            if got is None:
                continue
            acc.append(got[0])
            fsts.append(got[1])
        if len(acc) < 3:
            continue
        acc = np.array(acc)
        F = float(np.mean(fsts))
        for b in range(len(BIN_EDGES) - 1):
            sub = acc[:, b, :]
            if sub[:, 6].sum() < 500:
                continue
            dval, dsem = jackknife(sub, 0)
            rval, _ = jackknife(sub, 3)
            if not np.isfinite(dval) or not np.isfinite(dsem):
                continue
            # REALIZED recombination distance: the mean over the pairs that
            # actually landed in the bin, not the bin's nominal centre.
            c = float(sub[:, 7].sum() / sub[:, 6].sum())
            rows.append(dict(Ne=Ne, m=m, c=c, F=F, D=dval, sem=max(dsem, 1e-4),
                             r=rval, pairs=int(sub[:, 6].sum())))
    return rows


def main():
    print("GUARD=%s" % GUARD)
    rows = run_cells("both")
    alt = {(r["Ne"], r["m"], round(math.log10(r["c"]), 2)): r["D"]
           for r in run_cells("pooled")}
    inreg, outreg = [], []
    comps = {k: [] for k in ("1-F", "M/(1+M)", "2m/(2m+c)", "1/(1+4Nec)")}
    spread = []
    for r in rows:
        Ne, m, c = r["Ne"], r["m"], r["c"]
        pred = m / (m + c)
        M = 4 * Ne * m
        lab = ("Ne=%d m=%.2e c=%.2e c/m=%.2f (n=%d)"
               % (Ne, m, c, c / m, r["pairs"]))
        cell = dict(design=lab, lean=pred, truth=r["D"], sem=r["sem"])
        key = (Ne, m, round(math.log10(c), 2))
        pooled = alt.get(key, float("nan"))
        if np.isfinite(pooled) and r["D"] > 0:
            spread.append(abs(pooled - r["D"]) / r["D"])
        print("  %-46s D=%.4f ± %.4f  r=%.4f  pooled-D=%.4f | m/(m+c)=%.4f  "
              "2m/(2m+c)=%.4f  1-F=%.4f  M/(1+M)=%.4f"
              % (lab, r["D"], r["sem"], r["r"], pooled, pred,
                 2 * m / (2 * m + c), 1 - r["F"], M / (1 + M)))
        outreg.append(cell)
        if c <= m:
            inreg.append(cell)
            comps["1-F"].append(dict(design=lab, lean=1 - r["F"],
                                     truth=r["D"], sem=r["sem"]))
            comps["M/(1+M)"].append(dict(design=lab, lean=M / (1 + M),
                                         truth=r["D"], sem=r["sem"]))
            comps["2m/(2m+c)"].append(dict(design=lab,
                                           lean=2 * m / (2 * m + c),
                                           truth=r["D"], sem=r["sem"]))
            comps["1/(1+4Nec)"].append(dict(design=lab,
                                            lean=1 / (1 + 4 * Ne * c),
                                            truth=r["D"], sem=r["sem"]))
    # positive control: one panmictic population split four ways. Migration is
    # effectively infinite, so the ratio is 1 at EVERY distance and the
    # estimator has nowhere to hide an attenuation.
    cacc = []
    for rep in range(6):
        got = replicate(1000, 0.0, 777000 + 137 * rep, panmictic=True)
        if got is not None:
            cacc.append(got[0])
    csums = np.array(cacc).sum(axis=1)
    cval, csem = jackknife(csums, 0)
    print("  CONTROL panmictic, split four ways: D-ratio=%.4f ± %.4f (expect 1)"
          % (cval, csem))
    control = dict(design="one panmictic population split four ways "
                          "[no structure, shared fraction = 1 at every "
                          "distance]", lean=1.0, truth=cval,
                   sem=max(csem, 1e-4))
    asc = float(np.mean(spread)) if spread else float("nan")
    reg = ("island model, two demes, Ne in {1000, 2000, 4000} and m in "
           "{2.5e-4, 1e-3, 4e-3} so that Ne varies threefold at fixed m, 10 Mb "
           "at recombination 1e-8 and mutation 1e-8, 12 replicates, 100 "
           "diploids per deme. The observable is sigma_B/sigma_W: the "
           "cross-deme product of the gametic covariance D over SNP pairs "
           "divided by the geometric mean of the two within-deme products, "
           "every product taken between DISJOINT sample halves so no sampling "
           "noise enters numerator or denominator. Pairs are binned by physical "
           "separation and c is the REALIZED mean RHO*distance over the pairs "
           "in the bin; error bars are a delete-one-replicate jackknife on the "
           "ratio of averages. Sites are common in both demes; the same run "
           "under a pooled-frequency filter moves the ratio by %.0f%% on "
           "average, which is the ascertainment systematic" % (100 * asc))
    note = ("%s; mean |pooled - both| / both = %.3f, the ascertainment "
            "systematic" % (GUARD, asc))
    record("sharedLD_from_equilibrium [m/(m+c), c <= m]", "PortabilityDrift.lean",
           "m / (m + c)", inreg, regime=reg, control=control,
           argument_source="model", realised_inputs=True, note=note)
    record("sharedLD_from_equilibrium [m/(m+c), all c to c = 80 m]",
           "PortabilityDrift.lean", "m / (m + c)", outreg, regime=reg,
           control=control, argument_source="model", realised_inputs=True,
           note=note)
    for k, cells in comps.items():
        record("sharedLD [%s, competing, c <= m]" % k, "PortabilityDrift.lean",
               k, cells, regime=reg, control=control,
               argument_source=("sample" if k == "1-F" else "model"),
               realised_inputs=True)
    for r in RESULTS:
        r["guard"] = GUARD
        r["ascertainment_systematic"] = asc
    dump_results("battery_sharedld_rec_results.json")
    print("\n================ SUMMARY (GUARD=%s) ================" % GUARD)
    for r in RESULTS:
        w = r.get("worst", {})
        print("%-36s %-48s worst %9.2f sems, %7.2f%% rel"
              % (r["verdict"], r["name"], w.get("sems_off", float("nan")),
                 100 * w.get("rel_err", float("nan"))))


if __name__ == "__main__":
    main()
