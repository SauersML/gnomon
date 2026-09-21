"""Truth-math gates for study/simulate.py (MSI only; needs numpy, scipy, pandas and pyarrow).

    pytest examples/biobank/tests/test_study_simulate.py
    python examples/biobank/tests/test_study_simulate.py --n 400000 --json report.json

The Monte Carlo gate compares the analytic truth with observed outcomes, cell by cell: disease x ancestry, age band
and sex, and deciles of the truth. The outcomes come from two sources:
  - the generator's own latent draws;
  - an INDEPENDENT re-simulation that uses different samplers: bisection on F for onset, bisection on the Gompertz
    survival for death, and two homogeneous Poisson pieces for the codes.
In every cell z = (O - E) / sqrt(sum p (1 - p)). The gate passes when max |z| is below the Bonferroni bound at a
family alpha of 1e-3. Each planted error in the truth formulas must make the same gate fire.

Set STUDY_SIM_REFERENCE to the real 1KG projection parquet; without it, the synthetic mixture is used.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import special

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import simulate as sim  # noqa: E402

REFERENCE = os.environ.get("STUDY_SIM_REFERENCE")
ALPHA = 1e-3
PLANTS = {"no_competing_death": "survival", "entry_at_baseline": "survival", "first_code_event": "survival",
          "no_undiagnosed_prevalent": "survival", "no_slope_attenuation": "both", "no_exit": "binary"}


@functools.lru_cache(maxsize=6)
def world_and_sample(n: int, seed: int, scenario: str = "realistic", full: bool = True):
    """full: the truth for every person it is defined for (the gates need it); False is the fixtures' mode."""
    ref, src = sim.load_reference(REFERENCE, 20260918)
    world = sim.make_world(20260918, scenario, sim.load_diseases(sim.DEFAULT_DISEASES), ref, src)
    return world, sim.simulate(world, n, seed, workers=sim.default_workers(), log=lambda *_: None, full_truth=full)


def _bisect(fn, lo, hi, iters=64):
    lo, hi = lo.copy(), hi.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = fn(mid) > 0
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi)


def resample(world, s, seed):
    """Independent latent re-simulation (see the module docstring). Returns per-disease latent arrays."""
    rng = np.random.default_rng([seed, 99])
    p = s["people"]
    n = p["n"]
    birth = p["birth"]
    e0 = (p["e0"] - birth) / sim.DAYS
    a_b = (p["baseline"] - birth) / sim.DAYS
    c = sim.DEATH_AGE_SLOPE
    target = rng.exponential(1.0, n)

    def death_gap(a):
        return np.exp(p["omega"]) / c * (np.exp(c * (a - 60.0)) - np.exp(c * (a_b - 60.0))) - target

    t_m = _bisect(death_gap, a_b, a_b + 150.0)
    rate = world.sites.exit_rate.to_numpy()[p["site"]]
    known = (p["x_exit"] - birth) / sim.DAYS
    x_exit = np.where(known < a_b, known, a_b + rng.exponential(1.0, n) / rate)   # a pre-consent exit is known
    cut = (sim.day(sim.CDR_CUTOFF) + 1 - birth) / sim.DAYS - _lag(world)
    out = {}
    for rec in s["diseases"]:
        dp = rec["dp"]
        if dp.spec.pgs is None:
            continue
        terms = rec["terms"]
        scale = terms["scale"]
        lin = scale * terms["eta"] + terms["beta"] * rec["s"]
        u = rng.random(n)
        top = np.full(n, 200.0)
        f_top = world.link.cdf(scale * sim.alpha(dp, top)[0] + lin)
        t_d = _bisect(lambda a: world.link.cdf(scale * sim.alpha(dp, a)[0] + lin) - u, np.zeros(n), top)
        t_d = np.where(u >= f_top, np.inf, t_d)
        mu0, mu1 = terms["mu0"], terms["mu1"]
        e = rng.exponential(1.0, (4, n))
        end1 = np.maximum(t_d, e0)
        p1a = e0 + e[0] / mu0
        p1b = p1a + e[1] / mu0
        p1a = np.where(p1a < end1, p1a, np.inf)
        p1b = np.where(p1b < end1, p1b, np.inf)
        start2 = np.where(np.isfinite(t_d), np.maximum(t_d, e0), np.inf)
        p2a = start2 + e[2] / mu1
        p2b = p2a + e[3] / mu1
        pts = np.sort(np.stack([p1a, p1b, p2a, p2b]), axis=0)
        t1 = np.where(pts[0] < t_m, pts[0], np.inf)
        t2 = np.where(pts[1] < t_m, pts[1], np.inf)
        out[dp.spec.slug] = dict(t1=t1, t2=t2, t_m=t_m, label=t2 < np.minimum(x_exit, cut),
                                 label_noexit=t2 < cut)
    return out


def _cells(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = frame.groupby(by, observed=True)[["o", "e", "v"]].sum()
    g = g[g.v > 4.0]
    g["z"] = (g.o - g.e) / np.sqrt(g.v)
    return g


def calibration_z(obs, pred, strata: dict) -> pd.Series:
    """z per cell over several partitions (each partition is a list of stratum column names)."""
    f = pd.DataFrame({"o": obs.astype(float), "e": pred, "v": pred * (1.0 - pred), **strata})
    parts = [["all"], ["age_band"], ["site"], ["disease"], ["disease", "ancestry"], ["disease", "age_band"],
             ["disease", "sex"], ["disease", "decile"], ["ancestry", "age_band", "sex"], ["ancestry", "decile"],
             ["age_band", "decile"]]
    zs = []
    for by in parts:
        g = _cells(f, by)
        zs.append(pd.Series(g.z.to_numpy(), index=["|".join(by) + "=" + "|".join(map(str, k if isinstance(k, tuple)
                                                                                    else (k,))) for k in g.index]))
    return pd.concat(zs)


def gate(z: pd.Series) -> dict:
    crit = float(special.ndtri(1.0 - ALPHA / (2.0 * len(z))))
    worst = z.abs().idxmax()
    return {"cells": int(len(z)), "max_abs_z": float(z.abs().max()), "worst": worst, "crit": crit,
            "passed": bool(z.abs().max() < crit)}


def _strata(s, disease, idx):
    p = s["people"]
    age = p["a_base"][idx]
    return dict(all=np.zeros(len(idx), np.int8), disease=np.full(len(idx), disease), ancestry=p["label"][idx],
                age_band=np.digitize(age, [40, 60, 75]), sex=p["sex_code"][idx], site=p["site"][idx])


def monte_carlo(world, s, truths, latent, horizons_used=(1, 3, 5)):
    """Build the survival (per horizon), entry and binary O/E cells from latent outcomes."""
    p = s["people"]
    birth = p["birth"]
    a_l = (p["baseline"] + sim.LANDMARK_DAYS + 1 - birth) / sim.DAYS
    lag = _lag(world)
    blocks = {f"cif_{h}y": [] for h in horizons_used}
    blocks.update(entry=[], p_ever=[], p_ever_noexit=[])
    hs = list(s["horizons"])
    for rec in s["diseases"]:
        slug = rec["dp"].spec.slug
        if slug not in latent:
            continue
        t = truths[slug]
        lat = latent[slug]
        ok = np.isfinite(t["p_ever"])
        alive = ok & (lat["t_m"] >= a_l)
        entry = alive & (lat["t1"] >= a_l)
        idx = np.flatnonzero(alive)
        st = _strata(s, slug, idx)
        st["decile"] = pd.qcut(t["den"][idx], 10, labels=False, duplicates="drop")
        blocks["entry"].append((entry[idx], t["den"][idx], st))
        idx = np.flatnonzero(entry)
        for h in horizons_used:
            i = hs.index(h)
            end = a_l[idx] + np.floor(sim.DAYS * h) / sim.DAYS - lag
            obs = (lat["t2"][idx] < end) & (lat["t2"][idx] < lat["t_m"][idx])
            pred = t["cif"][idx, i]
            st = _strata(s, slug, idx)
            st["decile"] = pd.qcut(pred, 10, labels=False, duplicates="drop")
            blocks[f"cif_{h}y"].append((obs, pred, st))
        idx = np.flatnonzero(ok)
        for key, lab in (("p_ever", "label"), ("p_ever_noexit", "label_noexit")):
            pred = t[key][idx]
            st = _strata(s, slug, idx)
            st["decile"] = pd.qcut(pred, 10, labels=False, duplicates="drop")
            blocks[key].append((lat[lab][idx], pred, st))
    result = {}
    for key, parts in blocks.items():
        obs = np.concatenate([b[0] for b in parts])
        pred = np.concatenate([b[1] for b in parts])
        strata = {k: np.concatenate([b[2][k] for b in parts]) for k in parts[0][2]}
        result[key] = gate(calibration_z(obs, pred, strata))
    return result


def _lag(world):
    """Instant coding puts the event date one day after onset (the distinct-date rule); the truth applies it."""
    return 1.0 / sim.DAYS if sim.SCENARIOS[world.scenario]["coding"] == "instant" else 0.0


def own_latent(s, lag=0.0):
    """The generator's own latent outcomes, in the resample() layout."""
    p = s["people"]
    birth = p["birth"]
    cut = (sim.day(sim.CDR_CUTOFF) + 1 - birth) / sim.DAYS - lag
    out = {}
    for rec in s["diseases"]:
        if rec["dp"].spec.pgs is None:
            continue
        # the generator stores only the recorded count with exit; the no-exit label comes from its latent t2
        out[rec["dp"].spec.slug] = dict(t1=rec["t1"], t2=rec["t2"], t_m=p["t_m"],
                                        label=np.where(p["ehr"], rec["n_rec"] >= 2, False),
                                        label_noexit=rec["t2"] < cut)
    return out


def truths_of(world, s, plant=None, quad=None):
    if plant is None and quad is None:
        return {rec["dp"].spec.slug: rec["truth"] for rec in s["diseases"] if rec["dp"].spec.pgs is not None}
    out = sim.truths(world, s["people"], s["diseases"], s["horizons"], quad, plant, sim.default_workers())
    return {s["diseases"][i]["dp"].spec.slug: t for i, t in out.items()}


def subset(s, idx):
    """A copy of the sample restricted to people idx (for the refinement and derivative checks)."""
    people = {k: (v[idx] if isinstance(v, np.ndarray) and v.shape[:1] == (s["people"]["n"],) else v)
              for k, v in s["people"].items()}
    people["n"] = len(idx)
    diseases = []
    for rec in s["diseases"]:
        r = dict(rec)
        r["terms"] = {k: v[idx] for k, v in rec["terms"].items()}
        r["s"] = rec["s"][idx]
        diseases.append(r)
    return dict(s, people=people, diseases=diseases)


def refinement(world, s, m=3000):
    small = subset(s, np.arange(min(m, s["people"]["n"])))
    a = truths_of(world, small, quad=sim.Quad())
    b = truths_of(world, small, quad=sim.Quad(refine=2))
    worst = {}
    for key in ("p_ever", "p_ever_noexit", "dp_ever", "cif", "dcif", "den"):
        d = max(float(np.nanmax(np.abs(a[k][key] - b[k][key]))) for k in a)
        worst[key] = d
    return worst


def derivatives(world, s, m=2000, h=1e-4):
    small = subset(s, np.arange(min(m, s["people"]["n"])))
    base = truths_of(world, small, quad=sim.Quad())
    worst = {"dp_ever": 0.0, "dcif": 0.0}
    for rec in small["diseases"]:
        if rec["dp"].spec.pgs is None:
            continue
        vals = []
        for sign in (1.0, -1.0):
            vals.append(sim.compute_truth(world, rec["dp"], rec["terms"], rec["s"] + sign * h, small["people"],
                                          small["horizons"]))
        t = base[rec["dp"].spec.slug]
        for key, dkey in (("p_ever", "dp_ever"), ("cif", "dcif")):
            fd = (vals[0][key] - vals[1][key]) / (2.0 * h)
            scale = np.nanmax(np.abs(t[dkey])) + 1e-12
            worst[dkey] = max(worst[dkey], float(np.nanmax(np.abs(fd - t[dkey]))) / scale)
    return worst


def run_all(n, seed, plants=True, scenario="realistic"):
    t0 = time.time()
    world, s = world_and_sample(n, seed, scenario)
    gen_seconds = time.time() - t0
    truths = truths_of(world, s)
    report = {"n": n, "seed": seed, "scenario": scenario, "generate_seconds": round(gen_seconds, 1),
              "reference": world.ref_source}
    own = own_latent(s, _lag(world))
    indep = resample(world, s, seed)
    report["own_draws"] = monte_carlo(world, s, truths, own)
    report["independent_draws"] = monte_carlo(world, s, truths, indep)
    if plants:
        report["plants"] = {}
        for plant, kind in PLANTS.items():
            res = monte_carlo(world, s, truths_of(world, s, plant=plant), indep)
            keys = [k for k in res if (kind in ("survival", "both") and (k.startswith("cif") or k == "entry"))
                    or (kind in ("binary", "both") and k.startswith("p_ever"))]
            report["plants"][plant] = {"fired": any(not res[k]["passed"] for k in keys),
                                       "max_abs_z": max(res[k]["max_abs_z"] for k in keys)}
    report["refinement_max_abs_diff"] = refinement(world, s)
    report["derivative_max_rel_err"] = derivatives(world, s)
    report["seconds"] = round(time.time() - t0, 1)
    return report


# ------------------------------------------------------------------------------------------------ pytest
N_TEST = int(os.environ.get("STUDY_SIM_TEST_N", "100000"))


def test_sinh_arcsinh_moments_match_quadrature():
    x, w = np.polynomial.hermite_e.hermegauss(200)
    w = w / np.sqrt(2.0 * np.pi)
    for eps, tau in [(0.0, 1.0), (0.3, 0.8), (-0.2, 0.7), (0.45, 0.95), (0.1, 1.0)]:
        v = np.sinh((np.arcsinh(x) + eps) / tau)
        mean = float(np.sum(w * v))
        var = float(np.sum(w * v * v)) - mean ** 2
        m, s2 = sim.sas_moments(eps, tau)
        assert abs(m - mean) < 1e-9 and abs(s2 - var) < 1e-9, (eps, tau, m, mean, s2, var)


def test_link_functions_are_consistent():
    for link in (sim.Link("ao", 0.5), sim.Link("probit"), sim.Link("cloglog"), sim.Link("ao", 1.0)):
        p = np.linspace(1e-6, 1 - 1e-6, 1001)
        assert np.max(np.abs(link.cdf(link.ppf(p)) - p)) < 1e-10
        u = np.linspace(-8, 4, 2001)
        h = 1e-5
        assert np.max(np.abs((link.cdf(u + h) - link.cdf(u - h)) / (2 * h) - link.pdf(u))) < 1e-7
        assert np.max(np.abs((link.pdf(u + h) - link.pdf(u - h)) / (2 * h) - link.dpdf(u))) < 1e-7
    assert abs(sim.Link("ao", 1.0).sd - np.pi / np.sqrt(3.0)) < 1e-6
    # the survival function keeps its relative precision where the CDF rounds to 1
    for link, u, exact in ((sim.Link("cloglog"), 5.0, np.exp(-np.exp(5.0))),
                           (sim.Link("probit"), 12.0, 1.7764821120776e-33),
                           (sim.Link("ao", 0.5), 40.0, (1.0 + 0.5 * np.exp(40.0)) ** -2.0)):
        assert abs(link.sf(np.array([u]))[0] / exact - 1.0) < 1e-10, link.name
        assert abs(link.sf(np.array([-2.0]))[0] + link.cdf(np.array([-2.0]))[0] - 1.0) < 1e-15


def test_projection_reader_roundtrip(tmp_path):
    rows, cols = 5, 3
    m = np.arange(rows * cols, dtype="<f8").reshape(rows, cols)
    ids = [f"S{i}" for i in range(rows)]
    blob = bytearray(b"GNPSPC01" + (1).to_bytes(4, "little") + rows.to_bytes(8, "little")
                     + cols.to_bytes(8, "little") + (1).to_bytes(4, "little"))
    blob += m.T.astype("<f8").tobytes()
    strings = "".join(ids).encode()
    offs = np.cumsum([0] + [len(i) for i in ids]).astype("<u8")
    blob += b"GNPSID01" + (1).to_bytes(4, "little") + (0).to_bytes(4, "little") + rows.to_bytes(8, "little")
    blob += len(strings).to_bytes(8, "little") + offs.tobytes() + strings
    path = tmp_path / "x.bin"
    path.write_bytes(bytes(blob))
    f = sim.read_projection_bin(path)
    assert f.IID.tolist() == ids and np.array_equal(f[["PC1", "PC2", "PC3"]].to_numpy(), m)


def test_quadrature_is_converged():
    world, s = world_and_sample(N_TEST, 7)
    worst = refinement(world, s)
    assert worst["p_ever"] < 1e-8 and worst["cif"] < 1e-8 and worst["den"] < 1e-8, worst
    assert worst["dp_ever"] < 1e-7 and worst["dcif"] < 1e-7, worst


def test_derivatives_match_finite_differences():
    world, s = world_and_sample(N_TEST, 7)
    worst = derivatives(world, s)
    assert worst["dp_ever"] < 1e-5 and worst["dcif"] < 1e-5, worst


def test_truth_matches_monte_carlo():
    world, s = world_and_sample(N_TEST, 7)
    truths = truths_of(world, s)
    for name, latent in (("own", own_latent(s)), ("independent", resample(world, s, 7))):
        res = monte_carlo(world, s, truths, latent)
        bad = {k: v for k, v in res.items() if not v["passed"]}
        assert not bad, (name, bad)


def test_competitor_true_survival_world_matches_monte_carlo():
    """The calpred survival world exercises what the realistic world does not: a probit link, log-age alpha, a
    PC-dependent index scale and instant coding (event the day after onset), with death and exit on."""
    world, s = world_and_sample(60000, 9, "true_calpred_survival")
    truths = truths_of(world, s)
    for name, latent in (("own", own_latent(s, _lag(world))), ("independent", resample(world, s, 9))):
        res = monte_carlo(world, s, truths, latent)
        bad = {k: v for k, v in res.items() if not v["passed"]}
        assert not bad, (name, bad)


def test_planted_errors_fire():
    world, s = world_and_sample(N_TEST, 7)
    latent = resample(world, s, 7)
    silent = []
    for plant, kind in PLANTS.items():
        res = monte_carlo(world, s, truths_of(world, s, plant=plant), latent)
        keys = [k for k in res if (kind in ("survival", "both") and (k.startswith("cif") or k == "entry"))
                or (kind in ("binary", "both") and k.startswith("p_ever"))]
        if all(res[k]["passed"] for k in keys):
            silent.append((plant, {k: round(res[k]["max_abs_z"], 2) for k in keys}))
    assert not silent, silent


def test_competitor_true_worlds_have_closed_forms():
    """Without death, exit and rule-out codes, and with a next-day second code, the truth reduces to G of the index:
    p_ever = G(u(cutoff)), P(entry) = 1 - F(aL), and cif_h = (F(t_h) - F(aL)) / (1 - F(aL)). The binary worlds then
    lie exactly in the named competitor's family: G^-1(p_ever) is linear in its covariates (probit for the true_*
    worlds, the pipeline's link; logit for the logittrue_* misspecification world)."""
    for scenario in ("true_standard_binary", "true_zpc_binary", "true_covariates_binary", "true_calpred_binary",
                     "logittrue_standard_binary"):
        world, s = world_and_sample(5000, 5, scenario)
        p = s["people"]
        birth = p["birth"]
        lag = 1.0 / sim.DAYS
        a_c = (sim.day(sim.CDR_CUTOFF) + 1 - birth) / sim.DAYS - lag
        a_l = (p["baseline"] + sim.LANDMARK_DAYS + 1 - birth) / sim.DAYS
        for rec in s["diseases"]:
            if rec["dp"].spec.pgs is None:
                continue
            t, terms, dp = rec["truth"], rec["terms"], rec["dp"]
            lin = terms["scale"] * terms["eta"] + terms["beta"] * rec["s"]

            def big(a):
                return world.link.cdf(terms["scale"] * sim.alpha(dp, a)[0] + lin)

            ok = np.isfinite(t["p_ever"])
            assert np.max(np.abs(t["p_ever"][ok] - big(a_c)[ok])) < 1e-7, (scenario, dp.spec.slug)
            assert np.max(np.abs(t["den"][ok] - (1.0 - big(a_l))[ok])) < 1e-7, (scenario, dp.spec.slug)
            for i, h in enumerate(s["horizons"]):
                end = a_l + np.floor(sim.DAYS * h) / sim.DAYS - lag
                closed = (big(end) - big(a_l)) / (1.0 - big(a_l))
                assert np.max(np.abs(t["cif"][ok, i] - closed[ok])) < 1e-7, (scenario, dp.spec.slug, h)
            if scenario == "true_calpred_binary":
                continue
            # G^-1(p_ever) is linear in the covariates the competitor uses (z x PC terms for zpc)
            z = (rec["pgs"] - rec["pgs"][ok].mean()) / rec["pgs"][ok].std()
            age = (p["baseline"] - birth) / sim.DAYS
            admin = (sim.day(sim.CDR_CUTOFF) - p["baseline"]) / sim.DAYS
            cols = [np.ones(len(z)), age, admin, p["male"], *p["pcs"][:, :6].T]
            if "covariates" not in scenario:
                cols.append(z)
            if "zpc" in scenario:
                cols += [z * p["pcs"][:, k] for k in range(6)]
            x = np.column_stack(cols)[ok]
            y = world.link.ppf(t["p_ever"][ok])
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            assert np.max(np.abs(y - x @ coef)) < 1e-6, (scenario, dp.spec.slug, np.max(np.abs(y - x @ coef)))


def test_tables_follow_schema_and_latent_world():
    world, s = world_and_sample(20000, 3)
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        out = Path(d) / "t"
        man = sim.write_tables(s, world, out, "independent", {"scenario": "realistic"})
        cohort = sim._cohort_module()
        if cohort is not None:
            cohort.validate_tables(out)
        t = {k: pq.read_table(out / f"{k}.parquet").to_pandas(date_as_object=False) for k in
             ("person", "condition", "root", "ancestry", "pcs", "scores", "truth")}
        assert man["schema_version"] == 2 and set(man["snomed_codes"]) == set(t["root"].snomed_code)
        person, cond, truth = t["person"], t["condition"], t["truth"]
        assert person.person_id.is_unique and not cond.duplicated(["snomed_code", "person_id"]).any()
        cov = person.obs_start.notna()
        assert (cov == person.obs_end.notna()).all()
        assert (person.obs_start[cov] <= person.baseline_date[cov]).all()
        assert (person.baseline_date[cov] <= person.obs_end[cov]).all()
        assert person.ehr_site[person.baseline_date.isna()].isna().all()
        assert (person.zip3.isna() == person.zip3_post_baseline.isna()).all()
        two = cond.second_date.notna()
        assert ((cond.n_dates >= 2) == two).all() and (cond.first_date[two] < cond.second_date[two]).all()
        assert not truth.duplicated(["disease", "person_id"]).any() and truth.p_ever.notna().all()
        num = truth.select_dtypes("number")
        assert np.isfinite(num.to_numpy(float)[~num.isna().to_numpy()]).all()
        assert (truth.cif_1y.notna() == truth.in_survival).all()
        # Only a frame row whose EHR ended by the landmark lacks the uncensored outcome, and only when the latent
        # world has a code of it by the landmark (so the no-exit world would not admit the row).
        unknown = truth.in_survival & truth.uncensored_event.isna()
        assert (truth.uncensored_event.notna() <= truth.in_survival).all() and unknown.any()
        assert (truth.cutoff_event.isna() == truth.uncensored_event.isna()).all()
        lost = truth[unknown].merge(person, on="person_id")
        landmark = lost.baseline_date.to_numpy("datetime64[D]").astype(np.int64) + sim.LANDMARK_DAYS
        assert (lost.ehr_end.to_numpy("datetime64[D]").astype(np.int64) <= landmark).all()
        birth_day = lost.birth_date.to_numpy("datetime64[D]").astype(np.int64)
        assert (lost.t1_age.to_numpy(float) < (landmark + 1 - birth_day) / sim.DAYS).all()  # by the landmark's end

        # The observed survival outcome must be the latent one wherever observation did not end first.
        dis = sim.load_diseases(sim.DEFAULT_DISEASES)[0]
        rows = truth[(truth.disease == dis.slug) & truth.in_survival].merge(person, on="person_id")
        c = cond[cond.snomed_code == dis.root][["person_id", "second_date"]]
        rows = rows.merge(c, on="person_id", how="left")
        day = np.timedelta64(1, "D")
        birth = rows.birth_date.to_numpy("datetime64[D]")
        ev = rows.second_date.to_numpy("datetime64[D]")
        death = rows.death_date.to_numpy("datetime64[D]")
        end = rows.ehr_end.to_numpy("datetime64[D]")
        big = np.datetime64("2999-01-01")
        ev, death = np.where(np.isnat(ev), big, ev), np.where(np.isnat(death), big, death)
        exit_ = np.minimum(np.minimum(ev, death), end)
        event = np.where(ev == exit_, 1, np.where(death == exit_, 2, 0))
        admin = (rows.baseline_date.to_numpy("datetime64[D]") + sim.LANDMARK_DAYS * day
                 + int(np.floor(sim.DAYS * max(s["horizons"]))) * day)
        seen = (event > 0) & (exit_ <= admin)
        assert seen.sum() > 50
        assert (rows.uncensored_event.to_numpy()[seen] == event[seen]).all()
        age = (exit_ - birth) / day / sim.DAYS
        assert np.allclose(rows.uncensored_exit_age.to_numpy()[seen], age[seen], atol=1e-9)


def test_truth_frames_match_study_phenotypes():
    """truth.parquet's rows and in_survival must be exactly phenotypes.build_frames' disease population and survival
    frame, on the simulator's own tables (the truth is nulled with an independent implementation). The sample uses
    the fixtures' mode, whose truth is computed only on those rows, so a missed row shows up as a null."""
    from study import cohort, phenotypes
    world, s = world_and_sample(20000, 3, "realistic", False)
    kept = {}
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        for censoring in sim.CENSORING:
            out = Path(d) / censoring
            sim.write_tables(s, world, out, censoring, {"scenario": "realistic"})
            diseases = phenotypes.load_diseases(json.loads(Path(sim.DEFAULT_DISEASES).read_text()))
            base, frames = phenotypes.build_frames(cohort.ParquetSource(out), diseases,
                                                   phenotypes.CohortConfig(seed=4242))
            truth = pq.read_table(out / "truth.parquet").to_pandas()
            person = pq.read_table(out / "truth_person.parquet").to_pandas()
            assert set(base.frame.person_id) == set(person.person_id[person.in_base]), censoring
            for disease in diseases:
                rows = truth[truth.disease == disease.slug]
                surv = frames[disease.slug].survival
                binary = frames[disease.slug].binary
                assert set(surv.person_id) == set(rows.person_id[rows.in_survival]), (censoring, disease.slug)
                assert set(binary.person_id) <= set(rows.person_id), (censoring, disease.slug)
                assert (surv.event != 3).all(), (censoring, disease.slug)   # exclusions censor (below 1%)
                assert rows.p_ever.notna().all() and rows.cif_1y[rows.in_survival].notna().all()
                kept[censoring, disease.slug] = set(surv.person_id)
    # Survival eligibility reads only what is known at the landmark, and the two censoring rules share one latent
    # world, so they keep the same rows; they differ only in where follow-up ends.
    for disease in diseases:
        assert kept["independent", disease.slug] == kept["lastcontact", disease.slug], disease.slug


def test_publish_records_each_size_seed(tmp_path):
    """publish takes a seed per size (name=n:seed; the k-th size defaults to 1000 (k + 1)) and records it."""
    root = tmp_path / "v"
    try:
        sim.main(["publish", "--root", str(root), "--sizes", "tiny=600:4321,wee=500", "--scenarios", "realistic",
                  "--workers", "1", "--git-sha", "test"])
        listing = json.loads((root / "MANIFEST.json").read_text())
        assert {(s["dir"], s["n"], s["seed"]) for s in listing["sets"]} == {
            (f"{size}/realistic_{rule}", n, seed) for size, n, seed in (("tiny", 600, 4321), ("wee", 500, 2000))
            for rule in sim.CENSORING}
        for s in listing["sets"]:
            assert json.loads((root / s["dir"] / "manifest.json").read_text())["seed"] == s["seed"]
    finally:  # published fixtures are read-only
        for path in [root, *root.rglob("*")] if root.exists() else []:
            path.chmod(0o750 if path.is_dir() else 0o640)


def test_truth_short_exits_non_zero_on_a_refused_set(tmp_path):
    """truth_short refuses a set whose own 1-y truth disagrees with the recomputed one, writes nothing for it, and
    exits non-zero (it used to print REFUSED and exit 0)."""
    # The reference PCs the module was given, else the synthetic mixture on both sides (an
    # empty STUDY_SIM_REFERENCE is no path to truth_short too).
    out = tmp_path / "g"
    sim.main(["generate", "--out", str(out), "--n", "600", "--seed", "5", "--workers", "1", "--git-sha", "test",
              *(["--reference", REFERENCE] if REFERENCE else [])])
    bad = out / "realistic_lastcontact" / "truth.parquet"
    table = pq.read_table(bad)
    column = table.schema.get_field_index("cif_1y")
    cif = table.column(column).to_numpy(zero_copy_only=False)
    table = table.set_column(column, "cif_1y", pa.array(cif + 1e-6, pa.float64(), mask=~np.isfinite(cif)))
    pq.write_table(table, bad, compression="zstd")
    src = Path(__file__).resolve().parents[3]
    done = subprocess.run([sys.executable, str(src / "examples" / "biobank" / "study" / "truth_short.py"), str(src),
                           str(out), "realistic"], env={**os.environ, "STUDY_SIM_REFERENCE": REFERENCE or ""},
                          capture_output=True, text=True, timeout=300)
    assert done.returncode != 0 and "REFUSED: realistic_lastcontact" in done.stderr, (done.returncode,
                                                                                       done.stderr[-500:])
    assert (out / "realistic_independent" / "truth_short.parquet").exists()
    assert not (out / "realistic_lastcontact" / "truth_short.parquet").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--json")
    ap.add_argument("--no-plants", action="store_true")
    ap.add_argument("--scenario", default="realistic", choices=sorted(sim.SCENARIOS))
    a = ap.parse_args()
    report = run_all(a.n, a.seed, plants=not a.no_plants, scenario=a.scenario)
    text = json.dumps(report, indent=1, default=str)
    print(text)
    if a.json:
        Path(a.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
