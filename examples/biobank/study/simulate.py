#!/usr/bin/env python3
"""AoU-shaped truth simulator for the single study (SPEC section 6, amended by section 8).

It writes the SCHEMA v1 tables (manifest.json, person, condition, root, ancestry, pcs, scores) exactly as the
BigQuery cohort stage does, plus ``truth.parquet`` (key disease, person_id) and ``truth_person.parquet``. Every
stage after the cohort reads the same files on MSI and in AoU, so the synthetic tables exercise the real
pipeline, and the truth columns let every lane score predictions against the true risk.

The generative model
====================
Time is age in years (days / 365.25). A person has birth day b, consent (baseline) age aB, an EHR start E0 (the
first encounter, ``obs_start``) and, per disease k, the following latent processes. They are independent given
the person's covariates x, which include the latent standardized score S_k.

Clinical onset. ``P(T_D <= a | x) = F(a) = G(u(a))`` with ``u(a) = alpha(a) + eta(x) + beta(x) S``.
  - ``alpha(a) = a_inf - delta exp(-kappa (a - 50))`` is increasing, and ``G(a_inf + ...)`` < 1 leaves a never-onset
    fraction.  G is the Aranda-Ordaz asymmetric link ``G(u) = 1 - (1 + lam e^u)^(-1/lam)`` with lam = 0.5, which is
    neither probit, logit nor cloglog (audit S3: data are generated outside the fitted families).
  - Onset runs from birth, and everybody enrolled is alive at consent, so the disease-free survivors who enter the
    survival frame are depleted of high-S people as entry age grows (audit M10).
  - eta(x) holds sex, genetic-ancestry components, Census region, EHR site, zip3 deprivation and a PC term.
  - beta(x) = beta_EUR * sd_G * r(d) * exp(zeta (male - 0.4)), where d is the genetic distance (raw PC1-6
    Euclidean) from the EUR centroid, scaled so the AFR centroid is at d = 1, and
    ``r(d) = r_min + (1 - r_min) / (1 + (d / d50)^4)``: a sigmoid that no Duchon or linear PC basis represents
    exactly.
  - S is the ancestry-standardized score. The raw score is ``pgs = m + s (mu(x) + sigma(x) S)``, where
    ``S = (X - E X) / sd X`` and ``X = sinh((asinh(Z) + eps(x)) / tau(x))`` for Z ~ N(0, 1): skewed (eps) and
    heavy tailed (tau < 1). mu, sigma, eps and tau all depend on the ancestry mixture, so the law of z given the
    PCs changes with ancestry in location, scale AND shape (audit M10).  The standardization uses the closed-form
    sinh-arcsinh moments (Jones and Pewsey 2009).

Qualifying records. From E0, qualifying codes are a Poisson process of rate mu0 (rule-out codes) before onset and
mu1 (> mu0) after onset. Both rates scale with the EHR site's coding depth. A person who was diagnosed before E0 is
coded from E0 onwards at rate mu1; with a sparse site, they can look incident after a short lookback (audit M11).
Codes stop at death. In the data they are also cut at the EHR exit X and at the CDR cutoff.

Death. Gompertz, ``lambda_M(a) = exp(omega(x) + c (a - 60))`` with c = 0.085, conditioned on being alive at
consent. Death is independent of the disease given x.

Observation. The EHR runs from E0 (its first record, person.ehr_start) to person.ehr_end. X - aB ~ Exp(r_site) is
the EHR exit; 5% of EHR histories end before consent, which is known at baseline. Two censoring rules share one
latent world, so their tables differ only in ehr_end and obs_end:
  - ``independent`` (SPEC section 8 N2a): ehr_end = min(X, death, cutoff). The censoring is independent of the
    events given site and region.
  - ``lastcontact`` (N2b): ehr_end = the last EHR encounter before min(X, death, cutoff). The encounter rate rises
    after any onset and in the year before death, and every code is itself an encounter.
The observation period (every domain, as AoU builds it) is [min(E0, consent), max(ehr_end, last survey)], and 55%
complete a later survey; survival follow-up ends at ehr_end, as SCHEMA says.

The truth
=========
The truth is the probability of the OBSERVABLE event under the phenotype rule (audit S3). It is defined for
people with at least the base cohort's 365 days of lookback. Every integral is one-dimensional in the onset age s
or in the window end: Gauss-Legendre pieces clustered on the code-rate scale 1 / mu1 (and at most 2 years long),
running recurrences over a time grid, and Boole's rule for the outer integrals. The refinement test doubles every
node count and requires agreement to 1e-8. Write ``q(L) = 1 - e^-L (1 + L)`` for P(Poisson(L) >= 2), and
``Lam(s, w) = mu0 (s - E0) + mu1 (w - s)`` for the code intensity over [E0, w) given onset s.

p_ever (binary; SCHEMA truth.p_ever). Recorded codes run over [E0, W), with W = min(X, T_M, C), and
C = the end of the cutoff day. P(N[E0, w) >= 2) is
    P2(w) = F(E0) q(mu1 (w - E0)) + int_E0^w f(s) q(Lam(s, w)) ds + (1 - F(w)) q(mu0 (w - E0)),
and
    p_ever = int_aB^aC P2(w) (r_X + lambda_M(w)) S_W(w) dw + P2(aC) S_W(aC),
where ``S_W(w) = exp(-r_X (w - aB)) exp(-(H_M(w) - H_M(aB)))``. For an EHR that ended before consent, W = X is
known at baseline and p_ever = P2(X). ``p_ever_noexit`` is the same with r_X = 0 (nobody leaves the EHR before the
cutoff).

cif_{h}y (survival). Entry is at tau0, the end of the landmark day, which gives entry age aL = (landmark + 1 - b) /
365.25. The frame keeps people with no record dated on or before the landmark who are alive past it; the event is
the SECOND qualifying date. There is no exit and no cutoff (both are censoring). In the equations below, tau = t -
aL and P0 = exp(-mu0 (aL - E0)).
    A    = F(E0) e^{-mu1 (aL - E0)} + int_E0^aL f(s) e^{-mu0 (s - E0) - mu1 (aL - s)} ds     onset < aL, no code yet
    Den  = A + P0 (1 - F(aL))                                                               = P(entry | alive)
    Q(t) = A q(mu1 tau) + P0 [(1 - F(t)) q(mu0 tau) + int_aL^t f(s) q(mu0 (s - aL) + mu1 (t - s)) ds]
                                                                          = P(entry, 2nd code by t), no death
    cif_h = [Q(t_h) R(t_h) + int_aL^t_h Q(t) lambda_M(t) R(t) dt] / Den,   R(t) = exp(-(H_M(t) - H_M(aL)))
Here t_h = aL + floor(365.25 h) / 365.25, which is exactly the pipeline's "second_date - landmark <= 365.25 h".
``death_{h}y = 1 - R(t_h)`` is the true death risk by the horizon for someone who is alive at entry. Death is
independent of the disease, so it is also the cause-specific hazard truth for the death fit.

Slopes. ``slope_liab`` = beta(x) dS/dz is the generating link-scale slope. ``slope`` (and ``slope_cif_{h}y``) is
the local probit slope dPhi^-1(p)/dz of p_ever (and cif_h) at the person's own score, with exact derivatives
(the integrals are linear in F and f). Here ``z = (pgs - mean) / sd`` over the disease's truth rows (``z_mean``
and ``z_sd`` are in the manifest), and ``dS/dz = sd / (s sigma(x))``. A model lane that standardizes on its own
training rows rescales by sd_train / sd.

Exclusion roots (T1D for T2D, bipolar for MDD) have independent generic processes (no score). Because they are
independent, removing or censoring the people who meet them leaves every truth value above unchanged.

Usage
=====
    simulate.py reference --projection data.projection_scores.bin --labels kg_pop.tsv --out reference_pcs.parquet
    simulate.py generate --out DIR --n 20000 --seed 1 [--scenario realistic] [--reference reference_pcs.parquet]
    simulate.py publish --root /scratch.global/sauer354/aou-study/sim/v1 --reference ... --git-sha SHA
The scenarios are ``realistic`` (a), ``null_slope`` (b, constant slope), and ``gaussian_pgs`` (c, a Gaussian
score law in every ancestry). The ``true_*`` worlds make each competitor the true model (see SCENARIOS), and
``holdout`` is the claim run's scenario, never to be used for selection. Each writes one table directory per
censoring rule. ``--world-seed`` fixes the data-
generating process (sites, disease parameters, ancestry shifts), and ``--seed`` draws a fresh sample from it, so
replicate seeds are samples of one world. ``publish`` uses one seed per size for every scenario, so the scenarios
are paired: the same people, scores and uniforms, differing only in the scenario's knobs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import special

GENERATOR_VERSION = "study-sim/1"
SCHEMA_VERSION = 2
EPOCH = dt.date(1970, 1, 1)
CDR_CUTOFF = dt.date(2023, 10, 1)
DAYS = 365.25
LANDMARK_DAYS = 180
HORIZONS = (1, 2, 3, 5)
NUM_PCS = 16
DEATH_AGE_SLOPE = 0.085
MALE, FEMALE = 45880669, 45878463
COMPONENTS = ("AFR", "EUR", "EAS", "SAS", "AMR")
GROUP_SHARE = {"eur": 0.52, "afr": 0.22, "amr": 0.19, "eas": 0.026, "sas": 0.009, "mid": 0.005}
DEFAULT_DISEASES = Path(__file__).resolve().parent / "diseases.json"


def day(date: dt.date) -> int:
    return (date - EPOCH).days


# --------------------------------------------------------------------------------------------------------------
# links
# --------------------------------------------------------------------------------------------------------------
class Link:
    """CDF G of the onset liability noise, with density, density derivative, quantile and standard deviation."""

    def __init__(self, name: str, lam: float | None = None):
        self.name, self.lam = name, lam
        if name == "ao":
            if not lam or lam <= 0:
                raise ValueError("Aranda-Ordaz link needs lam > 0")
            self.loglam = float(np.log(lam))
        elif name not in ("probit", "cloglog"):
            raise ValueError(f"unknown link {name}")
        grid = np.linspace(-60.0, 60.0, 240001)
        dens = self.pdf(grid)
        step = grid[1] - grid[0]
        mean = float(np.sum(grid * dens) * step)
        self.sd = float(np.sqrt(np.sum((grid - mean) ** 2 * dens) * step))

    def describe(self) -> dict:
        return {"name": self.name, "lam": self.lam, "sd": self.sd}

    def cdf(self, u):
        if self.name == "probit":
            return special.ndtr(u)
        if self.name == "cloglog":
            return -np.expm1(-np.exp(np.minimum(u, 700.0)))
        return -np.expm1(-np.logaddexp(0.0, u + self.loglam) / self.lam)

    def pdf(self, u):
        if self.name == "probit":
            return np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
        if self.name == "cloglog":
            return np.exp(u - np.exp(np.minimum(u, 700.0)))
        return np.exp(u - (1.0 / self.lam + 1.0) * np.logaddexp(0.0, u + self.loglam))

    def dpdf(self, u):
        return self.evaluate(u)[3]

    def sf(self, u):
        return self.evaluate(u)[1]

    def evaluate(self, u, cdf=True):
        """(G, 1 - G, g, g') at u in one pass. The survival function 1 - G is computed directly, so it keeps its
        relative precision where G rounds to 1. G and 1 - G are None when cdf is False."""
        if self.name == "probit":
            g = np.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
            return (special.ndtr(u) if cdf else None), (special.ndtr(-u) if cdf else None), g, -u * g
        if self.name == "cloglog":
            e = np.exp(np.minimum(u, 700.0))
            g = np.exp(u - e)
            return (-np.expm1(-e) if cdf else None), (np.exp(-e) if cdf else None), g, g * (1.0 - e)
        v = u + self.loglam
        log1p_ev = np.logaddexp(0.0, v)
        g = np.exp(u - (1.0 / self.lam + 1.0) * log1p_ev)
        dg = g * (1.0 - (1.0 / self.lam + 1.0) * np.exp(v - log1p_ev))
        if not cdf:
            return None, None, g, dg
        return -np.expm1(-log1p_ev / self.lam), np.exp(-log1p_ev / self.lam), g, dg

    def ppf(self, p):
        p = np.asarray(p, dtype=float)
        if self.name == "probit":
            return special.ndtri(p)
        if self.name == "cloglog":
            return np.log(-np.log1p(-p))
        return np.log(np.expm1(-self.lam * np.log1p(-p))) - self.loglam


def sas_moments(eps, tau):
    """Mean and variance of X = sinh((asinh(Z) + eps) / tau), Z ~ N(0, 1) (Jones and Pewsey 2009)."""
    eps, tau = np.asarray(eps, float), np.asarray(tau, float)

    def p(q):
        return np.exp(0.25) / np.sqrt(8.0 * np.pi) * (special.kv((q + 1.0) / 2.0, 0.25)
                                                      + special.kv((q - 1.0) / 2.0, 0.25))

    mean = np.sinh(eps / tau) * p(1.0 / tau)
    second = 0.5 * (np.cosh(2.0 * eps / tau) * p(2.0 / tau) - 1.0)
    return mean, second - mean * mean


# --------------------------------------------------------------------------------------------------------------
# reference PCs (real 1KG projection)
# --------------------------------------------------------------------------------------------------------------
def read_projection_bin(path: str | Path) -> pd.DataFrame:
    """Read gnomon's ``*.projection_scores.bin`` (column-major f64 matrix plus an embedded IID section)."""
    raw = Path(path).read_bytes()
    rows = int.from_bytes(raw[12:20], "little")
    cols = int.from_bytes(raw[20:28], "little")
    if int.from_bytes(raw[28:32], "little") != 1:
        raise ValueError("projection matrix is not float64")
    end = 32 + rows * cols * 8
    matrix = np.frombuffer(raw, dtype="<f8", count=rows * cols, offset=32).reshape(cols, rows).T
    if raw[end:end + 8] != b"GNPSID01":
        raise ValueError("projection has no embedded row IDs")
    count = int.from_bytes(raw[end + 16:end + 24], "little")
    if count != rows:
        raise ValueError("projection row-ID count differs from the matrix")
    offsets = np.frombuffer(raw, dtype="<u8", count=count + 1, offset=end + 32).astype(np.int64)
    strings = raw[end + 32 + 8 * (count + 1):]
    ids = [strings[offsets[i]:offsets[i + 1]].decode() for i in range(count)]
    frame = pd.DataFrame(matrix, columns=[f"PC{i + 1}" for i in range(cols)])
    frame.insert(0, "IID", ids)
    return frame


def build_reference(projection: str, labels: str, out: str) -> pd.DataFrame:
    pcs = read_projection_bin(projection)
    lab = pd.read_csv(labels, sep="\t", dtype=str).drop_duplicates("IID")
    ref = pcs.merge(lab, on="IID", how="inner", validate="one_to_one")
    ref = ref[ref.superpop.isin(COMPONENTS)].reset_index(drop=True)
    if len(ref) < 0.9 * len(pcs):
        raise ValueError(f"only {len(ref)} of {len(pcs)} projected samples have a superpopulation label")
    ref = ref[["IID", "pop", "superpop"] + [c for c in pcs.columns if c.startswith("PC")]]
    ref.to_parquet(out, index=False)
    return ref


def synthetic_reference(rng: np.random.Generator, k: int = 20) -> pd.DataFrame:
    """Documented fallback when no real projection is given: five Gaussian clusters placed like the 1KG
    superpopulations on gnomon's projection scale (AFR far out on PC1, EAS on PC2, SAS and AMR between)."""
    centers = {"AFR": (-220, 20, 0, 0), "EUR": (80, 60, -20, 10), "EAS": (60, -180, 10, -20),
               "SAS": (40, 10, 90, 30), "AMR": (60, 20, -40, 110)}
    scale = np.array([22, 20, 18, 16] + [10.0] * (k - 4))
    rows = []
    for sp, c in centers.items():
        m = np.zeros(k)
        m[:4] = c
        x = m + scale * rng.standard_normal((600, k))
        for i in range(600):
            rows.append([f"SYN_{sp}_{i}", sp, sp, *x[i]])
    return pd.DataFrame(rows, columns=["IID", "pop", "superpop"] + [f"PC{i + 1}" for i in range(k)])


# --------------------------------------------------------------------------------------------------------------
# the fixed world: sites, geography, diseases
# --------------------------------------------------------------------------------------------------------------
REGION = {}
for _r, _states in {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE FL GA MD NC SC VA DC WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}.items():
    for _s in _states.split():
        REGION[_s] = _r

# USPS 3-digit ZIP prefix ranges per state (inclusive); synthetic zip3s are drawn from them.
ZIP3 = {"MA": (10, 27), "RI": (28, 29), "NH": (30, 38), "ME": (39, 49), "VT": (50, 59), "CT": (60, 69),
        "NJ": (70, 89), "NY": (100, 149), "PA": (150, 196), "DE": (197, 199), "DC": (200, 205), "MD": (206, 219),
        "VA": (220, 246), "WV": (247, 268), "NC": (270, 289), "SC": (290, 299), "GA": (300, 319),
        "FL": (320, 349), "AL": (350, 369), "TN": (370, 385), "MS": (386, 397), "KY": (400, 427),
        "OH": (430, 458), "IN": (460, 479), "MI": (480, 499), "IA": (500, 528), "WI": (530, 549),
        "MN": (550, 567), "SD": (570, 577), "ND": (580, 588), "MT": (590, 599), "IL": (600, 629),
        "MO": (630, 658), "KS": (660, 679), "NE": (680, 693), "LA": (700, 714), "AR": (716, 729),
        "OK": (730, 749), "TX": (750, 799), "CO": (800, 816), "WY": (820, 831), "ID": (832, 838),
        "UT": (840, 847), "AZ": (850, 865), "NM": (870, 884), "NV": (889, 898), "CA": (900, 961),
        "HI": (967, 968), "OR": (970, 979), "WA": (980, 994), "AK": (995, 999)}

# Rough US population weights for residence outside a site's state.
STATE_WEIGHT = {"CA": 12, "TX": 9, "FL": 7, "NY": 6, "PA": 4, "IL": 4, "OH": 3.5, "GA": 3.2, "NC": 3.2,
                "MI": 3, "NJ": 2.8, "VA": 2.6, "WA": 2.3, "AZ": 2.2, "MA": 2.1, "TN": 2.1, "IN": 2, "MD": 1.8,
                "MO": 1.8, "WI": 1.8, "CO": 1.7, "MN": 1.7, "SC": 1.6, "AL": 1.5, "LA": 1.4, "KY": 1.3,
                "OR": 1.3, "OK": 1.2, "CT": 1.1, "UT": 1, "IA": 1, "NV": 0.9, "AR": 0.9, "MS": 0.9, "KS": 0.9,
                "NM": 0.6, "NE": 0.6, "ID": 0.6, "WV": 0.5, "HI": 0.4, "NH": 0.4, "ME": 0.4, "MT": 0.3,
                "RI": 0.3, "DE": 0.3, "SD": 0.3, "ND": 0.2, "AK": 0.2, "DC": 0.2, "VT": 0.2, "WY": 0.2}

SITES = [("101", "NY", 0.09), ("102", "NY", 0.05), ("103", "PA", 0.07), ("104", "MA", 0.06), ("105", "IL", 0.06),
         ("106", "IL", 0.03), ("107", "WI", 0.07), ("108", "MI", 0.03), ("109", "AZ", 0.08), ("110", "CA", 0.05),
         ("111", "CA", 0.04), ("112", "TX", 0.04), ("113", "FL", 0.04), ("114", "FL", 0.03), ("115", "GA", 0.04),
         ("116", "AL", 0.03), ("117", "TN", 0.02), ("118", "MN", 0.02), ("119", "MO", 0.015), ("120", "LA", 0.02),
         ("121", "NM", 0.01), ("122", "WA", 0.015), ("123", "CO", 0.01), ("124", "NC", 0.02), ("125", "NJ", 0.02),
         ("126", "OH", 0.02)]

# Where each generation group enrolls, relative to the site weights.
ENRICH = {"afr": {"GA": 3, "AL": 3, "LA": 3, "NC": 2, "TN": 2, "FL": 1.5, "IL": 1.8, "MI": 2, "NY": 1.6, "MO": 1.5,
                  "OH": 1.4},
          "amr": {"AZ": 3, "CA": 3, "TX": 3.5, "NM": 3, "FL": 2.5, "NY": 1.8, "CO": 2, "IL": 1.3, "NJ": 1.5},
          "eas": {"CA": 4, "WA": 3, "NY": 2.2, "MA": 2, "IL": 1.2, "NJ": 1.5},
          "sas": {"NY": 2, "CA": 2, "TX": 1.8, "IL": 1.5, "MA": 1.5, "NJ": 2.5},
          "eur": {"WI": 2, "MN": 2, "PA": 1.5, "MA": 1.3, "MI": 1.2, "TN": 1.2, "OH": 1.3},
          "mid": {"MI": 3, "NY": 1.5, "CA": 1.5, "NJ": 1.5}}

# Deprivation preference (in within-state SD units) of each group's zip3 choice.
SES_PULL = {"afr": 0.8, "amr": 0.6, "eur": -0.1, "eas": -0.2, "sas": -0.1, "mid": 0.2}


@dataclass
class DiseaseSpec:
    """Generating parameters of one root. Effects are in probit-equivalent units (multiplied by the link sd)."""
    slug: str
    root: str
    sex: str | None
    pgs: str | None
    p50: float
    p80: float
    pinf: float
    beta: float = 0.0
    male: float = 0.0
    anc: dict = field(default_factory=dict)
    ses: float = 0.0
    south: float = 0.0
    code_rate: float = 4.0
    ruleout_rate: float = 0.02
    zeta: float = 0.0
    r_afr: float = 0.6
    exclusion_of: str | None = None
    branches: tuple = ()     # excluded branches (codes under the root that do not qualify); synthetic records never
                             # fall under them, so the tables are net of them as diseases.json declares


# Per-slug defaults. Targets are the reference person's cumulative onset (EUR female, S = 0, average SES), set so
# the simulated any-record prevalence is near the AoU Data Browser rates of study/diseases.json.
DISEASE_DEFAULTS = {
    "hypertension": dict(p50=0.30, p80=0.65, pinf=0.80, beta=0.30, male=0.12, anc={"AFR": 0.35}, ses=0.08,
                         south=0.10, code_rate=6.0, ruleout_rate=0.006, r_afr=0.60),
    "type_2_diabetes": dict(p50=0.10, p80=0.28, pinf=0.38, beta=0.32, male=0.10,
                            anc={"AFR": 0.25, "AMR": 0.30, "SAS": 0.40, "EAS": 0.10}, ses=0.12, south=0.08,
                            code_rate=5.0, ruleout_rate=0.004, r_afr=0.55),
    "atrial_fibrillation": dict(p50=0.012, p80=0.12, pinf=0.25, beta=0.30, male=0.25,
                                anc={"AFR": -0.25, "AMR": -0.10, "EAS": -0.15}, ses=0.03, south=0.02,
                                code_rate=4.0, ruleout_rate=0.006, r_afr=0.60),
    "coronary_artery_disease": dict(p50=0.03, p80=0.18, pinf=0.30, beta=0.28, male=0.35,
                                    anc={"SAS": 0.20, "EAS": -0.10}, ses=0.10, south=0.06, code_rate=4.0,
                                    ruleout_rate=0.008, zeta=0.15, r_afr=0.50),
    "breast_cancer": dict(p50=0.03, p80=0.10, pinf=0.13, beta=0.35, anc={"AFR": -0.03, "EAS": -0.10}, ses=-0.02,
                          code_rate=4.0, ruleout_rate=0.003, r_afr=0.55),
    "prostate_cancer": dict(p50=0.01, p80=0.13, pinf=0.18, beta=0.45, anc={"AFR": 0.30, "EAS": -0.20},
                            code_rate=4.0, ruleout_rate=0.003, r_afr=0.55),
    "asthma": dict(p50=0.14, p80=0.19, pinf=0.22, beta=0.20, male=-0.10, anc={"AFR": 0.12}, ses=0.10,
                   code_rate=3.0, ruleout_rate=0.006, r_afr=0.65),
    "copd": dict(p50=0.025, p80=0.12, pinf=0.17, beta=0.15, male=0.05, ses=0.20, south=0.05, code_rate=3.0,
                 ruleout_rate=0.006, r_afr=0.65),
    "major_depressive_disorder": dict(p50=0.24, p80=0.32, pinf=0.35, beta=0.20, male=-0.30,
                                      anc={"AFR": -0.10, "EAS": -0.20}, ses=0.10, code_rate=3.0,
                                      ruleout_rate=0.006, r_afr=0.60),
    "chronic_kidney_disease": dict(p50=0.03, p80=0.20, pinf=0.32, beta=0.20, male=0.02, anc={"AFR": 0.25},
                                   ses=0.10, south=0.05, code_rate=4.0, ruleout_rate=0.005, r_afr=0.60),
    "gout": dict(p50=0.012, p80=0.055, pinf=0.075, beta=0.35, male=0.50, anc={"AFR": 0.10, "EAS": 0.10},
                 ses=0.03, code_rate=2.5, ruleout_rate=0.003, r_afr=0.55),
    "primary_open_angle_glaucoma": dict(p50=0.004, p80=0.025, pinf=0.05, beta=0.35, male=0.05, anc={"AFR": 0.30},
                                        ses=0.02, code_rate=3.0, ruleout_rate=0.004, r_afr=0.60),
}
GENERIC_DISEASE = dict(p50=0.05, p80=0.15, pinf=0.25, beta=0.25, code_rate=3.0, ruleout_rate=0.005, r_afr=0.6)
EXCLUSION_DEFAULTS = {"46635009": dict(p50=0.009, p80=0.011, pinf=0.012, code_rate=4.0, ruleout_rate=0.002),
                      "13746004": dict(p50=0.030, p80=0.040, pinf=0.045, code_rate=3.0, ruleout_rate=0.002)}


def load_diseases(path: str | Path) -> list[DiseaseSpec]:
    """Read study/diseases.json (or a study.json diseases block): slug, root/snomed_code, sex, pgs, exclusions."""
    data = json.loads(Path(path).read_text())
    items = data.get("diseases", data)
    if isinstance(items, dict):
        items = [dict(v, slug=k) for k, v in items.items()]
    out, seen = [], set()
    for item in items:
        root = str(item.get("root") or item.get("snomed_code"))
        slug = item["slug"]
        branches = tuple(sorted(str(b.get("root") or b.get("snomed_code")) for b in item.get("excluded_branches") or []))
        spec = DiseaseSpec(slug=slug, root=root, sex=item.get("sex"), pgs=item.get("pgs") or item.get("pgs_id"),
                           branches=branches, **DISEASE_DEFAULTS.get(slug, GENERIC_DISEASE))
        out.append(spec)
        seen.add(root)
        for exc in item.get("exclusions") or []:
            code = str(exc.get("root") or exc.get("snomed_code"))
            if code in seen:
                continue
            seen.add(code)
            out.append(DiseaseSpec(slug=f"exclusion_{code}", root=code, sex=None, pgs=None,
                                   exclusion_of=slug, **EXCLUSION_DEFAULTS.get(code, dict(
                                       p50=0.01, p80=0.02, pinf=0.03))))
    return out


REALISTIC = dict(link=("ao", 0.5), pgs_shape=True, slope="distance", eta="full", law="ancestry", coding="ehr",
                 death=True, exit=True, alpha="saturating", scale="one", site_scale=1.0)
# Competitor-true worlds (audit S3): the named competitor's family holds the truth exactly, so ours must tie. No
# unobserved heterogeneity (eta = sex + linear PC1-6), one affine law of z in every ancestry (still skewed and
# heavy tailed), no rule-out codes and a second code on the day after onset.
#   binary (no death, no exit, linear alpha): logit P(y) is linear in age at baseline, admin years, sex, PCs, and z
#     (covariates: no z; standard: + beta z; zpc: + (beta + zeta'PC) z); calpred: probit with the whole index
#     divided by exp(nu'PC), i.e. PC-dependent mean and log-variance of the liability;
#   survival (cloglog, saturating alpha): proportional hazards on the age scale, so Cox with delayed entry is true
#     (covariates, standard, zpc as above; death competes); calpred: probit location-scale in log age.
_BIN = dict(REALISTIC, link=("ao", 1.0), eta="linear", law="affine", coding="instant", death=False, exit=False,
            alpha="linear")
_SURV = dict(REALISTIC, link=("cloglog",), eta="linear", law="affine", coding="instant")
SCENARIOS = {
    "realistic": REALISTIC,
    "null_slope": dict(REALISTIC, slope="constant"),
    "gaussian_pgs": dict(REALISTIC, pgs_shape=False),
    "true_covariates_binary": dict(_BIN, slope="zero"),
    "true_standard_binary": dict(_BIN, slope="constant"),
    "true_zpc_binary": dict(_BIN, slope="linear_pc"),
    "true_calpred_binary": dict(_BIN, link=("probit",), slope="constant", scale="calpred"),
    "true_covariates_survival": dict(_SURV, slope="zero"),
    "true_standard_survival": dict(_SURV, slope="constant"),
    "true_zpc_survival": dict(_SURV, slope="linear_pc"),
    "true_calpred_survival": dict(_SURV, link=("probit",), slope="constant", scale="calpred", alpha="log"),
    # The claim run's scenario, never used for selection (SPEC section 8, S4): a heavier-tailed link, a slope that also
    # weakens with age at consent, and larger site effects. Draw it only at claim time, with a fresh world seed.
    "holdout": dict(REALISTIC, link=("ao", 2.0), slope="distance_age", site_scale=1.5),
}
PUBLISHED = ("realistic", "null_slope", "gaussian_pgs")
CENSORING = ("independent", "lastcontact")


@dataclass
class DiseaseParams:
    spec: DiseaseSpec
    a_inf: float
    delta: float
    kappa: float
    law: dict            # component -> (mu, log_sigma, eps, tau)
    scale: float         # s in pgs = m + s (mu + sigma S)
    offset: float        # m
    d50: float
    r_min: float
    site_effect: np.ndarray
    region_effect: dict
    alpha_kind: str = "saturating"
    pc_effect: np.ndarray = field(default_factory=lambda: np.zeros(6))
    pc_slope: np.ndarray = field(default_factory=lambda: np.zeros(6))
    pc_scale: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def describe(self) -> dict:
        s = self.spec
        return {"slug": s.slug, "root": s.root, "sex": s.sex, "pgs": s.pgs, "exclusion_of": s.exclusion_of,
                "targets": [s.p50, s.p80, s.pinf], "alpha": [self.a_inf, self.delta, self.kappa],
                "beta_eur_probit": s.beta, "male": s.male, "ancestry": s.anc, "ses": s.ses, "south": s.south,
                "code_rate": s.code_rate, "ruleout_rate": s.ruleout_rate, "zeta": s.zeta,
                "r_afr": s.r_afr, "r_min": self.r_min, "d50": self.d50, "pgs_scale": self.scale,
                "pgs_offset": self.offset, "law": {k: list(v) for k, v in self.law.items()},
                "region_effect": self.region_effect, "alpha_kind": self.alpha_kind,
                "pc_effect": self.pc_effect.tolist(), "pc_slope": self.pc_slope.tolist(),
                "pc_scale": self.pc_scale.tolist(), "site_effect": [float(v) for v in self.site_effect]}


@dataclass
class World:
    scenario: str
    link: Link
    sites: pd.DataFrame
    zip3: pd.DataFrame
    diseases: list
    ref: pd.DataFrame
    ref_source: str
    c_eur: np.ndarray
    d_afr: float
    within_sd: np.ndarray
    pc_center: np.ndarray
    pc_sd: np.ndarray


def make_world(world_seed: int, scenario: str, specs: list[DiseaseSpec], ref: pd.DataFrame,
               ref_source: str) -> World:
    cfg = SCENARIOS[scenario]
    rng = np.random.default_rng([world_seed, 7])
    link = Link(*cfg["link"])
    pc = [c for c in ref.columns if c.startswith("PC")]
    if len(pc) < NUM_PCS:
        raise ValueError(f"reference has {len(pc)} PCs; need {NUM_PCS}")
    x = ref[pc].to_numpy(float)
    sp = ref.superpop.to_numpy()
    c_eur = x[sp == "EUR", :6].mean(0)
    d_afr = float(np.linalg.norm(x[sp == "AFR", :6].mean(0) - c_eur))
    within = np.sqrt(np.mean([x[sp == g].var(0) for g in COMPONENTS], axis=0))

    sites = pd.DataFrame(SITES, columns=["code", "state", "weight"])
    sites["src_id"] = "EHR site " + sites.code
    sites["region"] = sites.state.map(REGION)
    ns = len(sites)
    sites["depth"] = np.exp(rng.normal(0.0, 0.35, ns))
    sites["lookback_scale"] = rng.uniform(2.5, 7.0, ns)
    sites["exit_rate"] = rng.uniform(0.03, 0.14, ns) * np.where(sites.region == "South", 1.25, 1.0)
    sites["encounter_rate"] = rng.uniform(2.0, 6.0, ns)
    sites["death_effect"] = rng.normal(0.0, 0.08, ns)
    sites["common_effect"] = rng.normal(0.0, 0.06, ns)

    zrows = []
    for st, (lo, hi) in ZIP3.items():
        mean = rng.normal(0.31, 0.025)
        for z in range(lo, hi + 1):
            zrows.append((st, z, float(np.clip(rng.normal(mean, 0.045), 0.1, 0.7))))
    zip3 = pd.DataFrame(zrows, columns=["state", "zip3", "dep"])
    zip3["in_map"] = rng.random(len(zip3)) > 0.01

    diseases = []
    for spec in specs:
        g = link.ppf
        if cfg["alpha"] == "saturating":
            a_inf = float(g(spec.pinf))
            delta = float(a_inf - g(spec.p50))
            kappa = float(-np.log((a_inf - g(spec.p80)) / delta) / 30.0)
        else:   # linear: alpha = c0 + c1 (a - 50); log: alpha = c0 + c1 log(a / 50); stored as (c0, c1, 0)
            a_inf = float(g(spec.p50))
            delta = float((g(spec.p80) - a_inf) / (30.0 if cfg["alpha"] == "linear" else np.log(1.6)))
            kappa = 0.0
        if not (delta > 0 and kappa >= 0):
            raise ValueError(f"{spec.slug}: targets must satisfy p50 < p80 < pinf")
        law = {}
        eps_eur, tau_eur = rng.uniform(0.15, 0.45), rng.uniform(0.70, 0.90)
        span = {"AFR": 1.2, "EAS": 0.9, "SAS": 0.6, "AMR": 0.5}
        for comp in COMPONENTS:
            if comp == "EUR":
                mu, ls, eps, tau = 0.0, 0.0, eps_eur, tau_eur
            else:
                mu = rng.uniform(-span[comp], span[comp])
                ls = rng.uniform(-0.3, 0.25)
                eps = eps_eur + rng.uniform(-0.3, 0.1)
                tau = rng.uniform(0.75, 1.0)
            if not cfg["pgs_shape"]:
                eps, tau = 0.0, 1.0
            if cfg["law"] == "affine":
                mu, ls, eps, tau = 0.0, 0.0, eps_eur, tau_eur
            law[comp] = (float(mu), float(ls), float(eps), float(tau))
        r_min = 0.3
        r_afr = min(max(spec.r_afr, r_min + 0.02), 0.98)
        d50 = float(((1.0 - r_min) / (r_afr - r_min) - 1.0) ** -0.25)
        diseases.append(DiseaseParams(
            spec=spec, a_inf=a_inf, delta=delta, kappa=kappa, law=law,
            scale=float(10.0 ** rng.uniform(-4.0, -2.5)), offset=float(rng.normal(0.0, 1e-3)),
            d50=d50, r_min=r_min, site_effect=rng.normal(0.0, 0.10, ns),
            region_effect={"South": spec.south, "West": float(rng.normal(0.0, 0.04)),
                           "Midwest": float(rng.normal(0.0, 0.04)), "Northeast": 0.0},
            alpha_kind=cfg["alpha"], pc_effect=rng.normal(0.0, 0.12, 6), pc_slope=rng.normal(0.0, 0.15, 6),
            pc_scale=rng.normal(0.0, 0.10, 6)))
    return World(scenario, link, sites, zip3, diseases, ref, ref_source, c_eur, d_afr, within,
                 x[:, :6].mean(0), x[:, :6].std(0))


# --------------------------------------------------------------------------------------------------------------
# people
# --------------------------------------------------------------------------------------------------------------
def _pick(rng, weights: np.ndarray) -> np.ndarray:
    """One categorical draw per row of a (n, k) weight matrix."""
    c = np.cumsum(weights, axis=1)
    c /= c[:, -1:]
    return (rng.random((len(c), 1)) > c).sum(1)


def _uniform_between(rng, edges, probs, n):
    edges, probs = np.asarray(edges, float), np.asarray(probs, float)
    k = rng.choice(len(probs), n, p=probs / probs.sum())
    return edges[k] + (edges[k + 1] - edges[k]) * rng.random(n)


def sample_people(world: World, n: int, rng: np.random.Generator) -> dict:
    ref = world.ref
    pc = [c for c in ref.columns if c.startswith("PC")]
    x = ref[pc].to_numpy(float)
    pools = {}
    for comp in COMPONENTS:
        mask = ref.superpop.to_numpy() == comp
        if comp == "AFR":
            mask &= ~ref["pop"].isin(["ASW", "ACB"]).to_numpy()
        w = np.where(ref["pop"].to_numpy() == "FIN", 0.3, 1.0)[mask]
        pools[comp] = (np.flatnonzero(mask), w / w.sum())

    groups = np.array(list(GROUP_SHARE))
    share = np.array(list(GROUP_SHARE.values()))
    grp = groups[rng.choice(len(groups), n, p=share / share.sum())]
    w = np.zeros((n, len(COMPONENTS)))
    ci = {c: i for i, c in enumerate(COMPONENTS)}
    for g in groups:
        m = grp == g
        k = int(m.sum())
        if not k:
            continue
        if g == "eur":
            a = rng.beta(0.5, 30.0, k)
            other = rng.choice(["AFR", "AMR", "EAS", "SAS"], k, p=[0.4, 0.4, 0.1, 0.1])
            ww = np.zeros((k, 5))
            ww[:, ci["EUR"]] = 1 - a
            for o in ("AFR", "AMR", "EAS", "SAS"):
                ww[other == o, ci[o]] += a[other == o]
        elif g == "afr":
            a, b = rng.beta(2.5, 10.0, k), rng.beta(0.3, 30.0, k)
            ww = np.zeros((k, 5))
            ww[:, ci["AFR"]], ww[:, ci["EUR"]], ww[:, ci["AMR"]] = 1 - a - b, a, b
        elif g == "amr":
            a1, a2 = rng.beta(1.0, 6.0, k), rng.beta(0.7, 10.0, k)
            s = np.maximum(1.0, a1 + a2 + 0.05)
            a1, a2 = a1 / s, a2 / s
            ww = np.zeros((k, 5))
            ww[:, ci["AMR"]], ww[:, ci["EUR"]], ww[:, ci["AFR"]] = 1 - a1 - a2, a1, a2
        elif g == "eas":
            a = rng.beta(0.4, 12.0, k)
            ww = np.zeros((k, 5))
            ww[:, ci["EAS"]], ww[:, ci["EUR"]] = 1 - a, a
        elif g == "sas":
            a = rng.beta(0.4, 20.0, k)
            ww = np.zeros((k, 5))
            ww[:, ci["SAS"]], ww[:, ci["EUR"]] = 1 - a, a
        else:
            e = rng.uniform(0.5, 0.7, k)
            f = rng.uniform(0.0, 0.1, k)
            ww = np.zeros((k, 5))
            ww[:, ci["EUR"]], ww[:, ci["SAS"]], ww[:, ci["AFR"]] = e, 1 - e - f, f
        w[m] = ww
    # Projection is linear in dosage, so an admixed genome projects near the mixture of its sources.
    pcs = np.zeros((n, x.shape[1]))
    for comp in COMPONENTS:
        idx, pw = pools[comp]
        pick = idx[rng.choice(len(idx), n, p=pw)]
        pcs += w[:, [ci[comp]]] * x[pick]
    pcs += 0.15 * world.within_sd * rng.standard_normal(pcs.shape)
    dist = np.linalg.norm(pcs[:, :6] - world.c_eur, axis=1) / world.d_afr
    present = np.array([g for g in groups if (grp == g).any()])
    cent = np.stack([pcs[grp == g, :6].mean(0) for g in present])
    label = present[np.argmin(((pcs[:, None, :6] - cent[None]) ** 2).sum(-1), axis=1)]

    # demographics and dates
    cut = day(CDR_CUTOFF)
    edges = [day(dt.date(2017, 5, 6)), day(dt.date(2018, 5, 6)), day(dt.date(2019, 1, 1)), day(dt.date(2020, 1, 1)),
             day(dt.date(2021, 1, 1)), day(dt.date(2022, 1, 1)), day(dt.date(2023, 1, 1)), cut]
    baseline = np.floor(_uniform_between(rng, edges, [0.04, 0.16, 0.22, 0.09, 0.14, 0.20, 0.15], n)).astype(np.int64)
    age_b = _uniform_between(rng, [18, 30, 40, 50, 60, 70, 80, 90],
                             [0.13, 0.14, 0.15, 0.19, 0.21, 0.14, 0.04], n)
    young = rng.random(n) < 0.003
    age_b[young] = rng.uniform(15.0, 18.0, young.sum())
    birth = baseline - np.floor(age_b * DAYS).astype(np.int64) - rng.integers(0, 2, n)
    sex_draw = rng.random(n)
    sex_code = np.where(sex_draw < 0.60, FEMALE, np.where(sex_draw < 0.99, MALE,
                        np.where(sex_draw < 0.996, 903096, 1585849))).astype(np.int64)
    male = (sex_code == MALE).astype(float)
    has_baseline = rng.random(n) > 0.005

    # EHR site, residence, zip3 and deprivation
    sites = world.sites
    ehr = rng.random(n) < 0.72
    sw = np.tile(sites.weight.to_numpy(), (n, 1))
    for g in groups:
        m = grp == g
        mult = sites.state.map(ENRICH[g]).fillna(1.0).to_numpy()
        sw[m] *= mult
    site = _pick(rng, sw)
    states = np.array(list(STATE_WEIGHT))
    national = np.array(list(STATE_WEIGHT.values()), float)
    other_state = states[rng.choice(len(states), n, p=national / national.sum())]
    site_state = sites.state.to_numpy()[site]
    u = rng.random(n)
    state = np.where(ehr & (u < 0.88), site_state, other_state).astype(object)
    live_state = state.copy()
    state_null = rng.random(n) < 0.03
    zip3 = np.zeros(n, np.int64)
    dep = np.zeros(n)
    zt = world.zip3
    for st, block in zt.groupby("state"):
        m = live_state == st
        k = int(m.sum())
        if not k:
            continue
        dz = (block.dep.to_numpy() - block.dep.mean()) / 0.045
        pull = pd.Series(grp[m]).map(SES_PULL).to_numpy()
        pick = _pick(rng, np.exp(pull[:, None] * dz[None, :]))
        zip3[m] = block.zip3.to_numpy()[pick]
        dep[m] = block.dep.to_numpy()[pick]
    zip_known = rng.random(n) > 0.05
    zip_post = rng.random(n) < 0.10
    in_map = zt.set_index("zip3").in_map.reindex(zip3).fillna(False).to_numpy(bool)
    ses_z = (dep - 0.31) / 0.05
    region = pd.Series(live_state).map(REGION).fillna("unknown").to_numpy()

    # EHR history: start E0 (first encounter) and exit X, in days
    scale = sites.lookback_scale.to_numpy()[site]
    mix = rng.random(n)
    look = np.where(mix < 0.08, -rng.uniform(0.02, 3.0, n),
                    np.where(mix < 0.18, rng.uniform(0.0, 1.0, n), 1.0 + rng.gamma(1.4, scale)))
    look = np.minimum(look, np.maximum(age_b - 1.0, 0.0))
    look = np.minimum(look, 30.0)
    e0 = baseline - np.floor(look * DAYS)
    e0 = e0 + rng.random(n)
    ehr &= e0 < cut + 1          # an EHR that would start after the cutoff has no rows in the CDR
    cfg = SCENARIOS[world.scenario]
    rate_x = sites.exit_rate.to_numpy()[site] * (1.0 if cfg["exit"] else 0.0)
    pre_end = (rng.random(n) < 0.05) & (e0 < baseline) & cfg["exit"]
    with np.errstate(divide="ignore"):
        x_exit = baseline + np.minimum(rng.exponential(1.0, n) / rate_x, 1e4) * DAYS
    x_exit = np.where(pre_end, e0 + (baseline - e0) * rng.random(n), x_exit)

    # death: Gompertz from age 60, conditional on being alive at consent
    comp = dict(zip(COMPONENTS, w.T))
    omega = (np.log(0.0095) + 0.40 * male + 0.12 * ses_z + 0.15 * comp["AFR"] - 0.15 * comp["EAS"]
             + 0.05 * (region == "South") + np.where(ehr, sites.death_effect.to_numpy()[site], 0.0))
    if not cfg["death"]:
        omega = np.full(n, -50.0)   # hazard ~1e-22: nobody dies within any horizon
    a_base = (baseline - birth) / DAYS
    e_draw = rng.exponential(1.0, n)
    t_m = 60.0 + np.log(np.exp(DEATH_AGE_SLOPE * (a_base - 60.0))
                        + DEATH_AGE_SLOPE * e_draw * np.exp(-omega)) / DEATH_AGE_SLOPE

    race = np.select([grp == "eur", grp == "afr", np.isin(grp, ["eas", "sas"]), grp == "mid"],
                     [8527, 8516, 8515, 8527], default=2000000001)
    race = np.where(rng.random(n) < 0.04, 0, race).astype(np.int64)
    hisp = np.where(grp == "amr", rng.random(n) < 0.85, rng.random(n) < 0.03)
    eth = np.where(hisp, 38003563, 38003564)
    eth = np.where(rng.random(n) < 0.03, 903096, eth).astype(np.int64)

    ids = 1_000_000 + rng.choice(9_000_000, n, replace=False).astype(np.int64)
    in_ancestry = rng.random(n) > 0.05
    related = in_ancestry & (rng.random(n) < 0.05)
    genotyped = rng.random(n) > 0.02
    return dict(n=n, person_id=ids, group=grp, w=w, pcs=pcs, dist=dist, label=label, birth=birth,
                baseline=baseline, has_baseline=has_baseline, a_base=a_base, sex_code=sex_code, male=male,
                ehr=ehr, site=site, state=state, state_null=state_null, region=region, zip3=zip3,
                zip_known=zip_known, zip_post=zip_post, in_map=in_map, dep=dep, ses_z=ses_z, e0=e0, x_exit=x_exit,
                rate_x=rate_x, omega=omega, t_m=t_m, race=race, eth=eth, in_ancestry=in_ancestry, related=related, genotyped=genotyped,
                u_enc=rng.gamma(1.2, 1.0 / 1.2, n))


# --------------------------------------------------------------------------------------------------------------
# diseases: score, onset, codes
# --------------------------------------------------------------------------------------------------------------
def person_disease_terms(world: World, dp: DiseaseParams, people: dict) -> dict:
    """Per-person law of the score (mu, sigma, eps, tau), slope beta(x), index eta(x), index scale and code rates
    for one disease. The onset index is u(a) = scale (alpha(a) + eta) + beta S."""
    cfg = SCENARIOS[world.scenario]
    w = people["w"]
    n = people["n"]
    law = np.array([dp.law[c] for c in COMPONENTS])            # (5, 4)
    mu, log_sigma, eps, tau = (w @ law).T
    pcs = people["pcs"]
    pcz = (pcs[:, :6] - world.pc_center) / world.pc_sd
    if cfg["law"] == "ancestry":
        mu = mu + 0.15 * np.tanh(pcz[:, 2])
    sigma = np.exp(log_sigma)
    spec = dp.spec
    sd = world.link.sd
    scale = np.exp(-pcz @ dp.pc_scale) if cfg["scale"] == "calpred" else np.ones(n)
    base_slope = spec.beta * sd
    if cfg["slope"] in ("distance", "distance_age"):
        r = dp.r_min + (1.0 - dp.r_min) / (1.0 + (people["dist"] / dp.d50) ** 4)
        beta = base_slope * r * np.exp(spec.zeta * (people["male"] - 0.4))
        if cfg["slope"] == "distance_age":
            beta = beta * np.exp(-0.12 * (people["a_base"] - 55.0) / 10.0)
    elif cfg["slope"] == "constant":
        beta = np.full(n, base_slope)
    elif cfg["slope"] == "linear_pc":
        beta = base_slope * (1.0 + pcz @ dp.pc_slope)
    else:
        beta = np.zeros(n)
    beta = beta * scale
    site = people["site"]
    if cfg["eta"] == "full":
        anc = sum(spec.anc.get(c, 0.0) * w[:, i] for i, c in enumerate(COMPONENTS))
        site_eff = cfg["site_scale"] * np.where(people["ehr"], dp.site_effect[site]
                                                + world.sites.common_effect.to_numpy()[site], 0.0)
        reg = pd.Series(people["region"]).map(dp.region_effect).fillna(0.0).to_numpy()
        eta = sd * (spec.male * people["male"] + anc + spec.ses * people["ses_z"] + reg + site_eff
                    + 0.05 * np.tanh(pcz[:, 3]))
    else:
        eta = sd * (spec.male * people["male"] + pcz @ dp.pc_effect)
    depth = world.sites.depth.to_numpy()[site]
    if cfg["coding"] == "instant":
        mu1, mu0 = np.full(n, 1e7), np.full(n, 1e-9)
    else:
        mu1 = np.clip(spec.code_rate * depth, 0.3, 12.0)
        mu0 = np.clip(spec.ruleout_rate * depth, 1e-4, 0.2)
    allowed = np.ones(people["n"], bool)
    if spec.sex == "female":
        allowed = people["sex_code"] == FEMALE
    elif spec.sex == "male":
        allowed = people["sex_code"] == MALE
    return dict(mu=mu, sigma=sigma, eps=eps, tau=tau, beta=beta, eta=eta, scale=scale, mu0=mu0, mu1=mu1,
                allowed=allowed)


def alpha(dp: DiseaseParams, a):
    """alpha(a) and alpha'(a): saturating a_inf - delta e^{-kappa (a - 50)} (a never-onset fraction remains),
    linear c0 + c1 (a - 50), or log c0 + c1 log(a / 50); (c0, c1) are stored in (a_inf, delta)."""
    if dp.alpha_kind == "linear":
        return dp.a_inf + dp.delta * (a - 50.0), np.full(np.shape(a), dp.delta)
    if dp.alpha_kind == "log":
        a = np.maximum(a, 1e-9)
        return dp.a_inf + dp.delta * np.log(a / 50.0), dp.delta / a
    e = np.exp(-dp.kappa * (a - 50.0))
    return dp.a_inf - dp.delta * e, dp.delta * dp.kappa * e


def sample_onset(world, dp, lin, scale, rng):
    """Inverse-CDF draw of the clinical onset age from u(T) = scale alpha(T) + lin = G^-1(U); +inf for people who
    never develop the disease."""
    y = (world.link.ppf(rng.random(len(lin))) - lin) / scale
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if dp.alpha_kind == "linear":
            age = 50.0 + (y - dp.a_inf) / dp.delta
        elif dp.alpha_kind == "log":
            age = 50.0 * np.exp(np.minimum((y - dp.a_inf) / dp.delta, 700.0))
        else:
            age = 50.0 - np.log((dp.a_inf - y) / dp.delta) / dp.kappa
            age = np.where(y >= dp.a_inf, np.inf, age)
    return np.maximum(age, 0.0)


def _inverse_intensity(gam, e0, onset, mu0, mu1):
    """Age at which the code process (rate mu0 before onset, mu1 after, from E0) reaches cumulative gam."""
    pre = mu0 * np.clip(onset - e0, 0.0, None)
    before = np.where(onset > e0, e0 + gam / mu0, e0)
    after = np.maximum(onset, e0) + np.maximum(gam - pre, 0.0) / mu1
    return np.where(gam <= pre, before, after)


def _intensity(t, e0, onset, mu0, mu1):
    t = np.maximum(t, e0)
    return mu0 * (np.minimum(t, np.maximum(onset, e0)) - e0) + mu1 * np.clip(t - np.maximum(onset, e0), 0.0, None)


def realize_codes(t_d, e0, w_rec, t_m, mu0, mu1, rng):
    """First/second code ages of the latent (no-exit) world and the codes recorded before w_rec (ages)."""
    n = len(t_d)
    g1 = rng.exponential(1.0, n)
    g2 = g1 + rng.exponential(1.0, n)
    t1 = _inverse_intensity(g1, e0, t_d, mu0, mu1)
    t2 = _inverse_intensity(g2, e0, t_d, mu0, mu1)
    t1 = np.where(t1 < t_m, t1, np.inf)
    t2 = np.where(t2 < t_m, t2, np.inf)
    lam_w = _intensity(w_rec, e0, t_d, mu0, mu1)
    n_rec = np.where(lam_w < g1, 0, np.where(lam_w < g2, 1, 2 + rng.poisson(np.maximum(lam_w - g2, 0.0))))
    extra = n_rec - 2
    frac = np.where(extra > 0, rng.random(n) ** (1.0 / np.maximum(extra, 1)), 0.0)
    last_gam = np.where(n_rec >= 3, g2 + (lam_w - g2) * frac, np.where(n_rec == 2, g2, g1))
    last = np.where(n_rec >= 1, _inverse_intensity(last_gam, e0, t_d, mu0, mu1), -np.inf)
    return t1, t2, n_rec, last


def last_encounter(people, onset_any, w_rec_age, e0_age, world, rng):
    """Last background encounter age in [E0, W) for the last-contact rule (reverse-time inversion)."""
    site = people["site"]
    base = world.sites.encounter_rate.to_numpy()[site] * people["u_enc"]
    base = base * np.clip(1.0 + 0.02 * (people["a_base"] - 50.0), 0.4, None)
    t_m = people["t_m"]
    c1 = onset_any
    c2 = t_m - 1.0

    def rate(t):
        return base * (1.0 + 1.5 * (t >= c1)) * (1.0 + 3.0 * (t >= c2))

    lo = e0_age
    hi = w_rec_age
    pts = np.sort(np.stack([np.clip(c1, lo, hi), np.clip(c2, lo, hi)], 1), axis=1)
    b = np.column_stack([hi, pts[:, 1], pts[:, 0], lo])       # descending boundaries
    e = rng.exponential(1.0, len(lo))
    out = np.full(len(lo), -np.inf)
    acc = np.zeros(len(lo))
    for j in range(3):
        top, bot = b[:, j], b[:, j + 1]
        width = np.maximum(top - bot, 0.0)
        r = rate(0.5 * (top + bot))
        seg = r * width
        hit = (out == -np.inf) & (acc + seg >= e) & (width > 0)
        out = np.where(hit, top - (e - acc) / np.where(r > 0, r, 1.0), out)
        acc = acc + seg
    return out


# --------------------------------------------------------------------------------------------------------------
# truth: quadrature engine
# --------------------------------------------------------------------------------------------------------------
@dataclass
class Quad:
    """Quadrature resolution. ``refine`` multiplies every node count; the refinement test compares 1 with 2."""
    gl: int = 6          # Gauss-Legendre order in the clustered pieces
    gl_inc: int = 4      # and in the running recurrences' increments (4 and 6 agree to 1e-10 there)
    first_steps: int = 16
    steps_per_year: int = 8
    binary_steps: int = 56
    refine: int = 1
    abs_step: float = 2.0

    def nodes(self, increments=False):
        x, w = np.polynomial.legendre.leggauss((self.gl_inc if increments else self.gl) * self.refine)
        return (x + 1.0) / 2.0, w / 2.0


CLUSTER = np.array([0.0, 0.5, 1.5, 4.0, 10.0, 20.0, 40.0])
CLUSTER_ABS_SPAN = 32.0


def _clustered(span, mu, quad: Quad, absolute=True):
    """Nodes v in [0, min(span, 40 / mu)] clustered near 0 on the scale 1 / mu; beyond 40 / mu the kernel e^-mu v
    is below 5e-18. With ``absolute`` the pieces are also at most quad.abs_step / refine years long, so an onset
    density's own age scale is resolved; integrands without the onset density don't need that."""
    x, w = quad.nodes()
    top = np.minimum(40.0 / mu, span)[:, None]
    b = CLUSTER[None, :] / mu[:, None]
    if absolute:
        step = quad.abs_step / quad.refine
        grid = np.arange(step, CLUSTER_ABS_SPAN + 1e-9, step)
        b = np.concatenate([b, np.broadcast_to(grid, (len(mu), len(grid)))], axis=1)
    b = np.sort(np.minimum(b, top), axis=1)
    lo, hi = b[:, :-1], b[:, 1:]
    v = lo[:, :, None] + (hi - lo)[:, :, None] * x
    wv = (hi - lo)[:, :, None] * w
    return v.reshape(len(mu), -1), wv.reshape(len(mu), -1)


class Index:
    """Per-person onset index u(a) = scale alpha(a) + lin, with du/dS = beta."""

    def __init__(self, lin, beta, scale):
        self.lin, self.beta, self.scale = lin, beta, scale

    def take(self, j):
        return Index(self.lin[j], self.beta[j], self.scale[j])

    def at(self, dp, link, a, cdf=True):
        """F, f and the onset survival 1 - F at ages a (person axis first), each stacked as (value, d/dS); F and
        1 - F are None when cdf is False."""
        shape = (slice(None),) + (None,) * (np.ndim(a) - 1)
        lin, beta, scale = self.lin[shape], self.beta[shape], self.scale[shape]
        al, dal = alpha(dp, a)
        big, surv, g, dg = link.evaluate(scale * al + lin, cdf)
        dal = scale * dal
        f = np.stack([dal * g, dal * beta * dg])
        if not cdf:
            return None, f, None
        return np.stack([big, beta * g]), f, np.stack([surv, -beta * g])


def _q(x):
    return -np.expm1(-x) - x * np.exp(-x)


def _death(omega, a, a0):
    """Hazard at a and survival from a0 to a under the Gompertz death law."""
    c = DEATH_AGE_SLOPE
    lam = np.exp(omega + c * (a - 60.0))
    surv = np.exp(-np.exp(omega) / c * (np.exp(c * (a - 60.0)) - np.exp(c * (a0 - 60.0))))
    return lam, surv


def _running_corr(dp, link, ix, a0_rate, grid, mu0, mu1, quad, start=None):
    """Corr(t_j) = int_{a0}^{t_j} f(s) e^{-L} (1 + L) ds at every grid point, L = mu0 (s - a0) + mu1 (t_j - s).
    grid: (n, K + 1) ages from a0 upward. start: optional (M0, M1) at grid[:, 0] (a piece before the grid)."""
    x, w = quad.nodes(increments=True)
    n, kp1 = grid.shape
    dt_ = np.diff(grid, axis=1)                                         # (n, K)
    s = grid[:, :-1, None] + dt_[:, :, None] * x                        # (n, K, m)
    _, f, _ = ix.at(dp, link, s, cdf=False)
    lam = mu0[:, None, None] * (s - a0_rate[:, None, None]) + mu1[:, None, None] * dt_[:, :, None] * (1.0 - x)
    e = np.exp(-lam) * dt_[:, :, None] * w
    inc0 = (f * e).sum(-1)
    inc1 = (f * e * (1.0 + lam)).sum(-1)
    decay = np.exp(-mu1[:, None] * dt_)
    m0, m1 = start if start is not None else (np.zeros((2, n)), np.zeros((2, n)))
    corr = np.empty((2, n, kp1))
    corr[..., 0] = m1
    for j in range(kp1 - 1):
        m1 = decay[:, j] * (m1 + mu1 * dt_[:, j] * m0) + inc1[..., j]
        m0 = decay[:, j] * m0 + inc0[..., j]
        corr[..., j + 1] = m1
    return corr


def _boole_weights(k):
    """Composite Boole (5-point Newton-Cotes, error O(h^6)) weights for k intervals of unit width, k % 4 == 0."""
    if k % 4:
        raise ValueError("Boole's rule needs a multiple of 4 intervals")
    wts = np.zeros(k + 1)
    for start in range(0, k, 4):
        wts[start:start + 5] += np.array([7.0, 32.0, 12.0, 32.0, 7.0]) * 2.0 / 45.0
    return wts


def _boole(values, step):
    """Composite Boole over the last axis; step broadcasts."""
    return (values * _boole_weights(values.shape[-1] - 1)).sum(-1) * step


def survival_grid(horizons, quad: Quad, lag=0.0):
    """Common tau grid (years after entry) with the horizon ends as grid points, plus cumulative Boole weights
    (with the Jacobian of the quadratic grading on the first segment). ``lag`` moves every horizon end earlier by
    the fixed delay from onset to the event date (one day under instant coding)."""
    ends = [np.floor(DAYS * h) / DAYS - lag for h in horizons]
    n1 = quad.first_steps * quad.refine
    xs = np.linspace(0.0, 1.0, n1 + 1)
    taus = [ends[0] * xs ** 2]
    segs = [(0, n1, "x")]
    for i in range(1, len(ends)):
        k = 4 * int(np.ceil(quad.steps_per_year * quad.refine * (ends[i] - ends[i - 1]) / 4.0))
        seg = np.linspace(ends[i - 1], ends[i], k + 1)[1:]
        segs.append((len(np.concatenate(taus)) - 1, k, "t"))
        taus.append(seg)
    tau = np.concatenate(taus)
    weights = np.zeros((len(ends), len(tau)))
    acc = np.zeros(len(tau))
    for i, (s0, k, kind) in enumerate(segs):
        wts = _boole_weights(k)
        if kind == "x":
            wts = wts * (1.0 / k) * 2.0 * ends[0] * xs
        else:
            wts = wts * (tau[s0 + k] - tau[s0]) / k
        acc = acc.copy()
        acc[s0:s0 + k + 1] += wts
        weights[i] = acc
    index = [int(np.argmin(np.abs(tau - e))) for e in ends]
    return tau, weights, index, np.array(ends)


def survival_truth(dp, link, ix, e0, a_l, omega, mu0, mu1, horizons, quad: Quad, plant=None, lag=0.0):
    """cif at each horizon, its derivative in S, P(entry | alive), and net death risk; see the module docstring."""
    n = len(e0)
    span = a_l - e0
    v, wv = _clustered(span, mu1, quad)
    _, f_s, _ = ix.at(dp, link, a_l[:, None] - v, cdf=False)
    kern = np.exp(-mu0[:, None] * (span[:, None] - v) - mu1[:, None] * v) * wv
    big_e0, _, _ = ix.at(dp, link, e0)
    a_term = big_e0 * np.exp(-mu1 * span) + (f_s * kern).sum(-1)
    if plant == "no_undiagnosed_prevalent":
        a_term = np.zeros_like(a_term)
    p0 = np.exp(-mu0 * span)
    _, _, s_al = ix.at(dp, link, a_l)
    den = a_term + p0 * s_al

    tau, simp, index, ends = survival_grid(horizons, quad, lag)
    grid = a_l[:, None] + tau[None, :]
    corr = _running_corr(dp, link, ix, a_l, grid, mu0, mu1, quad)
    _, _, s_g = ix.at(dp, link, grid)
    # F(t) - F(aL) is written S(aL) - S(t): both survivals keep their precision when onset is nearly certain
    b = s_g * _q(mu0[:, None] * tau) + (s_al[..., None] - s_g) - corr
    q_grid = a_term[..., None] * _q(mu1[:, None] * tau) + p0[:, None] * b
    if plant == "first_code_event":
        # wrong estimand: onset at the FIRST qualifying date
        q_grid = (a_term[..., None] * -np.expm1(-mu1[:, None] * tau)
                  + p0[:, None] * (s_g * -np.expm1(-mu0[:, None] * tau) + s_al[..., None] - s_g))
    lam_g, r_g = _death(omega[:, None], grid, a_l[:, None])
    if plant == "no_competing_death":
        lam_g, r_g = np.zeros_like(lam_g), np.ones_like(r_g)
    ib = (b * lam_g * r_g) @ simp.T
    num = []
    death = []
    for i, h in enumerate(ends):
        vv, ww = _clustered(np.full(n, h), mu1, quad, absolute=False)
        lam_v, r_v = _death(omega[:, None], a_l[:, None] + vv, a_l[:, None])
        if plant == "no_competing_death":
            lam_v, r_v = np.zeros_like(lam_v), np.ones_like(r_v)
        r_h = r_g[:, index[i]]
        ia = (1.0 - r_h) - (np.exp(-mu1[:, None] * vv) * (1.0 + mu1[:, None] * vv) * lam_v * r_v * ww).sum(-1)
        if plant == "first_code_event":
            ia = (1.0 - r_h) - (np.exp(-mu1[:, None] * vv) * lam_v * r_v * ww).sum(-1)
            ib_i = ((s_g * -np.expm1(-mu0[:, None] * tau) + s_al[..., None] - s_g)
                    * lam_g * r_g) @ simp[i]
        else:
            ib_i = ib[..., i]
        num.append(q_grid[..., index[i]] * r_h + a_term * ia + p0 * ib_i)
        death.append(1.0 - r_h)
    num = np.stack(num, -1)                                            # (2, n, H)
    cif = num[0] / den[0][:, None]
    dcif = (num[1] * den[0][:, None] - num[0] * den[1][:, None]) / den[0][:, None] ** 2
    return cif, dcif, den[0], np.stack(death, -1)


def binary_truth(dp, link, ix, e0, a_b, a_c, omega, mu0, mu1, rate_x, quad: Quad, plant=None):
    """p_ever with exit, p_ever without exit, and d p_ever / dS (see the module docstring)."""
    n = len(e0)
    span = a_b - e0
    v, wv = _clustered(span, mu1, quad)
    s = a_b[:, None] - v
    _, f_s, _ = ix.at(dp, link, s, cdf=False)
    lam = mu0[:, None] * (s - e0[:, None]) + mu1[:, None] * v
    e = np.exp(-lam) * wv
    m0, m1 = (f_s * e).sum(-1), (f_s * e * (1.0 + lam)).sum(-1)
    k = quad.binary_steps * quad.refine
    step = (a_c - a_b) / k
    grid = a_b[:, None] + step[:, None] * np.arange(k + 1)
    corr = _running_corr(dp, link, ix, e0, grid, mu0, mu1, quad, start=(m0, m1))
    _, _, s_g = ix.at(dp, link, grid)
    big_e0, _, s_e0 = ix.at(dp, link, e0)
    since = grid - e0[:, None]
    p2 = (big_e0[..., None] * _q(mu1[:, None] * since) + (s_e0[..., None] - s_g) - corr
          + s_g * _q(mu0[:, None] * since))
    lam_g, r_g = _death(omega[:, None], grid, a_b[:, None])
    out = []
    for rx in (rate_x, np.zeros(n)):
        if plant == "no_exit":
            rx = np.zeros(n)
        s_w = np.exp(-rx[:, None] * (grid - a_b[:, None])) * r_g
        dens = (rx[:, None] + lam_g) * s_w
        out.append(_boole(p2 * dens, step) + p2[..., -1] * s_w[:, -1])
    return out[0][0], out[1][0], out[0][1]


def compute_truth(world, dp, terms, s_latent, people, horizons, quad=None, plant=None, chunk=8192):
    quad = quad or Quad()
    n = people["n"]
    beta = terms["beta"]
    if plant == "no_slope_attenuation":
        beta = np.full(n, dp.spec.beta * world.link.sd) * terms["scale"]
    ix = Index(terms["scale"] * terms["eta"] + beta * s_latent, beta, terms["scale"])
    birth = people["birth"]
    lag = 1.0 / DAYS if SCENARIOS[world.scenario]["coding"] == "instant" else 0.0
    e0 = (people["e0"] - birth) / DAYS
    a_b = (people["baseline"] - birth) / DAYS
    a_c = (day(CDR_CUTOFF) + 1 - birth) / DAYS - lag
    x_age = (people["x_exit"] - birth) / DAYS
    a_l = (people["baseline"] + LANDMARK_DAYS + 1 - birth) / DAYS
    if plant == "entry_at_baseline":
        a_l = (people["baseline"] + 1 - birth) / DAYS
    # defined where the base cohort's 365-day lookback holds (the truth is published for base participants only)
    ok = terms["allowed"] & people["ehr"] & (people["baseline"] - np.floor(people["e0"]) >= 365)
    out = {k: np.full(n, np.nan) for k in ("p_ever", "p_ever_noexit", "dp_ever", "den")}
    for k in ("cif", "dcif", "death"):
        out[k] = np.full((n, len(horizons)), np.nan)
    idx = np.flatnonzero(ok)
    for lo in range(0, len(idx), chunk):
        j = idx[lo:lo + chunk]
        sub = ix.take(j)
        p, p_no, dp_ = binary_truth(dp, world.link, sub, e0[j], a_b[j], a_c[j], people["omega"][j],
                                    terms["mu0"][j], terms["mu1"][j], people["rate_x"][j], quad, plant)
        # An EHR that ended before consent is known at baseline: its records stop at that exit.
        pre = np.flatnonzero(x_age[j] < a_b[j])
        if len(pre) and plant != "no_exit":
            k = j[pre]
            p[pre], _, dp_[pre] = binary_truth(dp, world.link, ix.take(k), e0[k], x_age[k], x_age[k],
                                               people["omega"][k], terms["mu0"][k], terms["mu1"][k],
                                               people["rate_x"][k], quad, plant)
        out["p_ever"][j], out["p_ever_noexit"][j], out["dp_ever"][j] = p, p_no, dp_
        cif, dcif, den, death = survival_truth(dp, world.link, sub, e0[j], a_l[j], people["omega"][j],
                                               terms["mu0"][j], terms["mu1"][j], horizons, quad, plant, lag)
        out["cif"][j], out["dcif"][j], out["den"][j], out["death"][j] = cif, dcif, den, death
    return out


def probit_slope(p, dp_ds, ds_dz):
    """d Phi^-1(p) / dz from dp/dS (the local probit slope); null within 1e-12 of 0 or 1."""
    ok = (p > 1e-12) & (p < 1.0 - 1e-12)     # nearer 0 or 1 the probit slope has no reliable digits
    z = special.ndtri(np.where(ok, p, 0.5))
    dens = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
    return np.where(ok, dp_ds * ds_dz / dens, np.nan)


# --------------------------------------------------------------------------------------------------------------
# one simulated world sample
# --------------------------------------------------------------------------------------------------------------
def simulate(world: World, n: int, seed: int, horizons=HORIZONS, quad=None, workers=1, log=print) -> dict:
    rng = np.random.default_rng([seed, 11])
    t0 = time.time()
    people = sample_people(world, n, rng)
    log(f"  people: {time.time() - t0:.1f}s")
    birth = people["birth"]
    e0_age = (people["e0"] - birth) / DAYS
    cut_age = (day(CDR_CUTOFF) + 1 - birth) / DAYS
    w_rec = np.minimum(np.minimum((people["x_exit"] - birth) / DAYS, people["t_m"]), cut_age)
    people["ehr"] = people["ehr"] & (e0_age < w_rec)   # an EHR that would start after death or exit has no rows
    diseases = []
    onset_any = np.full(n, np.inf)
    last_code = np.full(n, -np.inf)
    for dp in world.diseases:
        terms = person_disease_terms(world, dp, people)
        z = rng.standard_normal(n)
        mean, var = sas_moments(terms["eps"], terms["tau"])
        s_lat = (np.sinh((np.arcsinh(z) + terms["eps"]) / terms["tau"]) - mean) / np.sqrt(var)
        if dp.spec.pgs is None:
            terms["beta"] = np.zeros(n)
        lin = terms["scale"] * terms["eta"] + terms["beta"] * s_lat
        t_d = sample_onset(world, dp, lin, terms["scale"], rng)
        t_d = np.where(terms["allowed"], t_d, np.inf)
        mu0 = np.where(terms["allowed"], terms["mu0"], 1e-12)
        codes = realize_codes(t_d, e0_age, w_rec, people["t_m"], mu0, terms["mu1"], rng)
        n_rec = np.where(people["ehr"], codes[2], 0)
        onset_any = np.minimum(onset_any, t_d)
        last_code = np.maximum(last_code, np.where(n_rec > 0, codes[3], -np.inf))
        pgs = dp.offset + dp.scale * (terms["mu"] + terms["sigma"] * s_lat)
        pgs_null = rng.random(n) < 0.002
        missing = np.where(pgs_null, 100.0, np.round(rng.beta(1.0, 60.0, n) * 100.0, 4))
        diseases.append(dict(dp=dp, terms=terms, s=s_lat, pgs=pgs, pgs_null=pgs_null, missing_pct=missing,
                             t_d=t_d, t1=codes[0], t2=codes[1], n_rec=n_rec, last=codes[3]))
    log(f"  draws: {time.time() - t0:.1f}s")
    ehr_end = {"independent": w_rec}
    last_bg = last_encounter(people, onset_any, w_rec, e0_age, world, rng)
    ehr_end["lastcontact"] = np.maximum(np.maximum(last_bg, last_code), e0_age)
    # PPI surveys and physical measurements: at consent, and for 55% a later survey before death and the cutoff
    later = rng.random(n) < 0.55
    horizon = np.minimum(day(CDR_CUTOFF) + 1.0, birth + people["t_m"] * DAYS)
    people["survey_end"] = np.floor(people["baseline"] + later * rng.random(n)
                                    * np.maximum(horizon - people["baseline"], 0.0)).astype(np.int64)
    t1 = time.time()
    parallel_truth(world, people, diseases, horizons, quad, workers)
    log(f"  truth: {time.time() - t1:.1f}s on {workers} worker(s)")
    return dict(people=people, diseases=diseases, ehr_end_age=ehr_end, w_rec=w_rec, horizons=tuple(horizons),
                seed=seed)


_TRUTH_CONTEXT = None


def _truth_task(i):
    world, people, diseases, horizons, quad, plant = _TRUTH_CONTEXT
    rec = diseases[i]
    return i, compute_truth(world, rec["dp"], rec["terms"], rec["s"], people, horizons, quad, plant)


def truths(world, people, diseases, horizons, quad=None, plant=None, workers=1):
    """{index: truth} for every scored disease, one disease per forked worker (inputs shared copy-on-write)."""
    global _TRUTH_CONTEXT
    todo = [i for i, rec in enumerate(diseases) if rec["dp"].spec.pgs is not None]
    _TRUTH_CONTEXT = (world, people, diseases, horizons, quad, plant)
    try:
        if workers <= 1:
            return dict(map(_truth_task, todo))
        with multiprocessing.get_context("fork").Pool(workers) as pool:
            return dict(pool.imap_unordered(_truth_task, todo))
    finally:
        _TRUTH_CONTEXT = None


def parallel_truth(world, people, diseases, horizons, quad, workers):
    for i, truth in truths(world, people, diseases, horizons, quad, None, workers).items():
        diseases[i]["truth"] = truth


def default_workers() -> int:
    return max(1, min(4, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1))


# --------------------------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------------------------
def _date(values, mask=None):
    arr = np.asarray(values, dtype=np.int64)
    return pa.array(arr.astype(np.int32), type=pa.date32(), mask=None if mask is None else ~np.asarray(mask))


def _floor_day(age, birth):
    return np.floor(birth + age * DAYS).astype(np.int64)


def _write(table: pa.Table, path: Path) -> dict:
    pq.write_table(table, path, compression="zstd")
    return {"rows": table.num_rows, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _second_date(rec, birth):
    """Day of the second distinct recorded code (a same-day second code moves one day later; see the docstring)."""
    first = _floor_day(np.where(np.isfinite(rec["t1"]), rec["t1"], 0.0), birth)
    second = _floor_day(np.where(np.isfinite(rec["t2"]), rec["t2"], 0.0), birth)
    return first, np.maximum(second, first + 1)


def build_frames(sim, censoring):
    """Reference implementation of the SCHEMA v2 base cohort and per-disease survival frame, used to null the truth
    outside the frames and by the tests. Returns (base mask, {slug: (disease rows, survival rows)}, periods), where
    periods holds the EHR range (EHR rows only) and the observation period (every domain, consent and surveys
    included), as SCHEMA defines them."""
    p = sim["people"]
    birth, base = p["birth"], p["baseline"]
    ehr = p["ehr"]
    ehr_start = np.floor(p["e0"]).astype(np.int64)
    ehr_end = np.ceil(birth + sim["ehr_end_age"][censoring] * DAYS).astype(np.int64) - 1
    obs_start = np.where(ehr, np.minimum(ehr_start, base), base)
    obs_end = np.maximum(np.where(ehr, ehr_end, base), p["survey_end"])
    covering = p["has_baseline"].copy()     # the consent record itself lies in the one observation period
    age = (base - birth) / DAYS
    base_mask = (p["in_ancestry"] & ~p["related"] & p["genotyped"] & np.isin(p["sex_code"], [MALE, FEMALE])
                 & p["has_baseline"] & (age >= 18) & covering & (base - obs_start >= 365))
    death_day = np.floor(birth + p["t_m"] * DAYS).astype(np.int64)
    death_rec = death_day <= day(CDR_CUTOFF)
    landmark = base + LANDMARK_DAYS
    never = np.iinfo(np.int64).max
    met = {}
    for rec in sim["diseases"]:
        first, second = _second_date(rec, birth)
        rec["first_day"] = np.where(rec["n_rec"] >= 1, first, never)
        rec["second_day"] = np.where(rec["n_rec"] >= 2, second, never)
        met[rec["dp"].spec.root] = rec["second_day"]
    frames = {}
    for rec in sim["diseases"]:
        spec = rec["dp"].spec
        if spec.pgs is None:
            continue
        pop = base_mask & rec["terms"]["allowed"] & ~rec["pgs_null"]
        excluded = np.zeros(p["n"], bool)
        for other in sim["diseases"]:
            if other["dp"].spec.exclusion_of == spec.slug:
                excluded |= met[other["dp"].spec.root] <= landmark
        surv = (pop & ~excluded & (rec["first_day"] > landmark) & ~(death_rec & (death_day <= landmark))
                & ehr & (np.minimum(ehr_end, day(CDR_CUTOFF)) > landmark))
        frames[spec.slug] = (pop, surv)
    periods = dict(ehr_start=ehr_start, ehr_end=ehr_end, obs_start=obs_start, obs_end=obs_end, covering=covering)
    return base_mask, frames, periods


def _outcome(rec, p, end_day):
    """(event, exit age) in the no-exit latent world, administratively ended at end_day: 1 disease (the second
    code), 2 death, 0 event-free at end_day. Same-day disease and death go to the disease, as in the frames."""
    birth = p["birth"]
    ev = np.where(np.isfinite(rec["t2"]), _floor_day(np.where(np.isfinite(rec["t2"]), rec["t2"], 0.0), birth),
                  np.iinfo(np.int64).max)
    ev = np.where(np.isfinite(rec["t2"]), np.maximum(ev, rec["first_day_latent"] + 1), ev)
    death = np.floor(birth + p["t_m"] * DAYS).astype(np.int64)
    exit_day = np.minimum(np.minimum(ev, death), end_day)
    event = np.where(ev == exit_day, 1, np.where(death == exit_day, 2, 0)).astype(np.int8)
    return event, (exit_day - birth) / DAYS


def _arrow_frame(frame: pd.DataFrame) -> pa.Table:
    arrays = {}
    for c in frame.columns:
        col = frame[c]
        if col.dtype == object or pd.api.types.is_string_dtype(col):
            arrays[c] = pa.array(col.astype(object), pa.string())
        elif col.dtype == bool:
            arrays[c] = pa.array(col, pa.bool_())
        elif c == "person_id":
            arrays[c] = pa.array(col, pa.int64())
        elif c.endswith("_event"):
            v = col.to_numpy(float)
            arrays[c] = pa.array(np.nan_to_num(v).astype(np.int8), pa.int8(), mask=~np.isfinite(v))
        else:
            v = col.to_numpy(float)
            arrays[c] = pa.array(v, pa.float64(), mask=~np.isfinite(v))
    return pa.table(arrays)


def truth_table(sim, frames, censoring):
    """truth.parquet: one row per (disease, person) of the disease population (base, sex, score present)."""
    p = sim["people"]
    birth = p["birth"]
    landmark = p["baseline"] + LANDMARK_DAYS
    horizons = sim["horizons"]
    admin_end = landmark + int(np.floor(DAYS * max(horizons)))
    cutoff_end = np.minimum(admin_end, day(CDR_CUTOFF))
    rows, z_stats = [], {}
    for rec in sim["diseases"]:
        spec = rec["dp"].spec
        if spec.pgs is None:
            continue
        pop, surv = frames[spec.slug]
        t = rec["truth"]
        j = np.flatnonzero(pop)
        pgs = rec["pgs"][j]
        mean, sd = float(pgs.mean()), float(pgs.std())
        z_stats[spec.slug] = {"z_mean": mean, "z_sd": sd}
        ds_dz = sd / (rec["dp"].scale * rec["terms"]["sigma"][j])
        rec["first_day_latent"] = _floor_day(np.where(np.isfinite(rec["t1"]), rec["t1"], 0.0), birth)
        ev_u, age_u = _outcome(rec, p, admin_end)
        ev_c, age_c = _outcome(rec, p, cutoff_end)
        sj = surv[j]
        row = {"disease": spec.slug, "person_id": p["person_id"][j], "p_ever": t["p_ever"][j],
               "p_ever_noexit": t["p_ever_noexit"][j],
               "slope": probit_slope(t["p_ever"][j], t["dp_ever"][j], ds_dz),
               "slope_liab": rec["terms"]["beta"][j] * ds_dz, "z_sim": (pgs - mean) / sd,
               "S": rec["s"][j], "eta": rec["terms"]["eta"][j],
               "lin": rec["terms"]["scale"][j] * rec["terms"]["eta"][j] + rec["terms"]["beta"][j] * rec["s"][j],
               "index_scale": rec["terms"]["scale"][j],
               "onset_age": rec["t_d"][j], "t1_age": rec["t1"][j], "t2_age": rec["t2"][j],
               "p_entry": t["den"][j], "in_survival": sj,
               "uncensored_event": np.where(sj, ev_u[j], np.nan),
               "uncensored_exit_age": np.where(sj, age_u[j], np.nan),
               "cutoff_event": np.where(sj, ev_c[j], np.nan), "cutoff_exit_age": np.where(sj, age_c[j], np.nan)}
        for i, h in enumerate(horizons):
            row[f"cif_{h}y"] = np.where(sj, t["cif"][j, i], np.nan)
            row[f"slope_cif_{h}y"] = np.where(sj, probit_slope(t["cif"][j, i], t["dcif"][j, i], ds_dz), np.nan)
            row[f"death_{h}y"] = np.where(sj, t["death"][j, i], np.nan)
        rows.append(pd.DataFrame(row))
    return _arrow_frame(pd.concat(rows, ignore_index=True)), z_stats


def schema_tables(sim, world, censoring):
    """The six SCHEMA v2 tables as Arrow tables, plus the frames used to null the truth."""
    p = sim["people"]
    birth, base = p["birth"], p["baseline"]
    base_mask, frames, periods = build_frames(sim, censoring)
    covering = periods["covering"]
    e0 = np.floor(p["e0"]).astype(np.int64)
    death_day = np.floor(birth + p["t_m"] * DAYS).astype(np.int64)
    death_rec = death_day <= day(CDR_CUTOFF)
    pre_base = p["ehr"] & p["has_baseline"] & (e0 < base)
    zip_known = p["zip_known"] & p["has_baseline"]
    tables = {"person": pa.table({
        "person_id": pa.array(p["person_id"], pa.int64()),
        "birth_date": _date(birth),
        "sex_at_birth_concept_id": pa.array(p["sex_code"], pa.int64()),
        "race_concept_id": pa.array(p["race"], pa.int64()),
        "ethnicity_concept_id": pa.array(p["eth"], pa.int64()),
        "baseline_date": _date(base, p["has_baseline"]),
        "obs_start": _date(periods["obs_start"], covering),
        "obs_end": _date(periods["obs_end"], covering),
        "ehr_start": _date(periods["ehr_start"], p["ehr"]),
        "ehr_end": _date(periods["ehr_end"], p["ehr"]),
        "death_date": _date(death_day, death_rec),
        "state": pa.array(list(np.where(p["state_null"], None, p["state"])), pa.string()),
        "ehr_site": pa.array(list(np.where(pre_base, world.sites.src_id.to_numpy()[p["site"]], None)), pa.string()),
        "zip3": pa.array(p["zip3"].astype(np.int32), pa.int32(), mask=~zip_known),
        "zip3_post_baseline": pa.array(p["zip_post"], pa.bool_(), mask=~zip_known),
        "deprivation_index": pa.array(p["dep"], pa.float64(), mask=~(zip_known & p["in_map"])),
    })}
    cond = []
    for rec in sim["diseases"]:
        m = rec["n_rec"] > 0
        if m.any():
            cond.append(pd.DataFrame({"snomed_code": rec["dp"].spec.root, "person_id": p["person_id"][m],
                                      "first_date": rec["first_day"][m], "second_date": rec["second_day"][m],
                                      "n_dates": rec["n_rec"][m].astype(np.int32)}))
    cond = pd.concat(cond, ignore_index=True)
    has2 = cond.n_dates.to_numpy() >= 2
    tables["condition"] = pa.table({
        "snomed_code": pa.array(cond.snomed_code.astype(object), pa.string()),
        "person_id": pa.array(cond.person_id, pa.int64()),
        "first_date": _date(cond.first_date),
        "second_date": _date(np.where(has2, cond.second_date, 0), has2),
        "n_dates": pa.array(cond.n_dates, pa.int32())})
    codes = [rec["dp"].spec.root for rec in sim["diseases"]]
    tables["root"] = pa.table({
        "snomed_code": pa.array(codes, pa.string()),
        "concept_id": pa.array([2_000_000_000 + i for i in range(len(codes))], pa.int64()),
        "concept_name": pa.array([f"simulated {rec['dp'].spec.slug}" for rec in sim["diseases"]], pa.string()),
        "n_descendants": pa.array([1] * len(codes), pa.int64())})
    anc = p["in_ancestry"]
    tables["ancestry"] = pa.table({"person_id": pa.array(p["person_id"][anc], pa.int64()),
                                   "ancestry_pred": pa.array(p["label"][anc], pa.string()),
                                   "related_excluded": pa.array(p["related"][anc], pa.bool_())})
    g = p["genotyped"]
    pcs = {"person_id": pa.array(p["person_id"][g], pa.int64())}
    for i in range(NUM_PCS):
        pcs[f"PC{i + 1}"] = pa.array(p["pcs"][g, i], pa.float64())
    tables["pcs"] = pa.table(pcs)
    scores = {"person_id": pa.array(p["person_id"][g], pa.int64())}
    for rec in sim["diseases"]:
        code = rec["dp"].spec.pgs
        if code is not None:
            scores[code] = pa.array(rec["pgs"][g], pa.float64(), mask=rec["pgs_null"][g])
            scores[f"{code}_missing_pct"] = pa.array(rec["missing_pct"][g], pa.float64())
    tables["scores"] = pa.table(scores)
    return tables, frames, base_mask


def _cohort_module():
    """study/cohort.py (study-cohort's contract implementation) when it is importable, else None."""
    try:
        from . import cohort  # type: ignore[attr-defined]
        return cohort
    except ImportError:
        try:
            import cohort  # type: ignore[no-redef]
            return cohort
        except ImportError:
            return None


def write_tables(sim, world, out: Path, censoring: str, meta: dict) -> dict:
    """Write one extraction directory: truth files first, then the six tables and manifest.json last (through
    cohort.write_tables when it is importable, which validates the whole contract)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").unlink(missing_ok=True)
    tables, frames, base_mask = schema_tables(sim, world, censoring)
    truth, z_stats = truth_table(sim, frames, censoring)
    p = sim["people"]
    extra = {"truth": _write(truth, out / "truth.parquet")}
    person_truth = pa.table({
        "person_id": pa.array(p["person_id"], pa.int64()),
        "group": pa.array(p["group"], pa.string()),
        **{f"w_{c.lower()}": pa.array(p["w"][:, i], pa.float64()) for i, c in enumerate(COMPONENTS)},
        "genetic_distance": pa.array(p["dist"], pa.float64()),
        "site": pa.array(list(np.where(p["ehr"], world.sites.src_id.to_numpy()[p["site"]], None)), pa.string()),
        "region_true": pa.array(p["region"], pa.string()),
        "deprivation_true": pa.array(p["dep"], pa.float64()),
        "death_age": pa.array(p["t_m"], pa.float64()),
        "exit_age": pa.array((p["x_exit"] - p["birth"]) / DAYS, pa.float64(), mask=~p["ehr"]),
        "in_base": pa.array(base_mask, pa.bool_())})
    extra["truth_person"] = _write(person_truth, out / "truth_person.parquet")
    manifest = {
        "source": "simulator", "seed": sim["seed"], "snomed_codes": [rec["dp"].spec.root for rec in sim["diseases"]],
        "scores": [rec["dp"].spec.pgs for rec in sim["diseases"] if rec["dp"].spec.pgs],
        "num_pcs": NUM_PCS, "ses_available": True, "cdr_cutoff": CDR_CUTOFF.isoformat(),
        "cdr_cutoff_source": "simulator", "prune_unmatched": 0,
        "excluded_branches": {rec["dp"].spec.root: list(rec["dp"].spec.branches) for rec in sim["diseases"]
                              if rec["dp"].spec.branches},
        "simulator": dict(meta, censoring=censoring, horizons=list(sim["horizons"]), landmark_days=LANDMARK_DAYS,
                          admin_end_years=max(sim["horizons"]), z_scale=z_stats, link=world.link.describe(),
                          reference=world.ref_source, files=extra,
                          diseases=[d.describe() for d in world.diseases],
                          sites=json.loads(world.sites.to_json(orient="records"))),
    }
    cohort = _cohort_module()
    if cohort is not None:
        return cohort.write_tables(out, tables, manifest)
    manifest["schema_version"] = SCHEMA_VERSION
    manifest["created_utc"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["tables"] = {name: _write(tables[name], out / f"{name}.parquet") for name in tables}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_reference(path: str | None, world_seed: int):
    if path:
        return pd.read_parquet(path), f"{path} sha256={file_sha(Path(path))}"
    return synthetic_reference(np.random.default_rng([world_seed, 3])), "synthetic mixture (no projection given)"


def generate(out: Path, n: int, seed: int, scenario: str, world_seed: int, reference: str | None,
             diseases: str, sha: str, workers: int = 1, log=print) -> list[dict]:
    ref, ref_source = load_reference(reference, world_seed)
    world = make_world(world_seed, scenario, load_diseases(diseases), ref, ref_source)
    t0 = time.time()
    sim = simulate(world, n, seed, workers=workers, log=log)
    meta = {"generator": GENERATOR_VERSION, "scenario": scenario, "world_seed": world_seed, "n": n,
            "git_sha": sha, "script_sha256": file_sha(Path(__file__)), "diseases_file": str(diseases),
            "diseases_sha256": file_sha(Path(diseases)), "simulate_seconds": round(time.time() - t0, 2)}
    manifests = []
    for censoring in CENSORING:
        manifests.append(write_tables(sim, world, out / f"{scenario}_{censoring}", censoring, meta))
    log(f"  wrote {scenario} in {time.time() - t0:.1f}s")
    return manifests


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--projection", required=True)
    r.add_argument("--labels", required=True)
    r.add_argument("--out", required=True)
    for name in ("generate", "publish"):
        g = sub.add_parser(name)
        g.add_argument("--reference")
        g.add_argument("--diseases", default=str(DEFAULT_DISEASES))
        g.add_argument("--world-seed", type=int, default=20260918)
        g.add_argument("--git-sha", default=None)
        g.add_argument("--workers", type=int, default=default_workers())
        if name == "generate":
            g.add_argument("--out", required=True)
            g.add_argument("--n", type=int, required=True)
            g.add_argument("--seed", type=int, required=True)
            g.add_argument("--scenario", choices=sorted(SCENARIOS), default="realistic")
        else:
            g.add_argument("--root", required=True)
            g.add_argument("--sizes", default="small=20000,medium=100000,large=400000")
            g.add_argument("--scenarios", default=",".join(PUBLISHED))
    a = ap.parse_args(argv)
    if a.cmd == "reference":
        ref = build_reference(a.projection, a.labels, a.out)
        print(ref.superpop.value_counts().to_dict())
        return
    sha = a.git_sha or git_sha()
    if a.cmd == "generate":
        generate(Path(a.out), a.n, a.seed, a.scenario, a.world_seed, a.reference, a.diseases, sha, a.workers)
        return
    root = Path(a.root)
    if root.exists() and any(root.iterdir()):
        raise SystemExit(f"{root} exists and is not empty; published fixtures are read-only, use a new version")
    listing = {"generator": GENERATOR_VERSION, "git_sha": sha, "world_seed": a.world_seed, "sets": []}
    for k, item in enumerate(a.sizes.split(",")):
        size, n = item.split("=")
        for scenario in a.scenarios.split(","):
            seed = 1000 * (k + 1)
            print(f"{size} {scenario} n={n} seed={seed}", flush=True)
            for m in generate(root / size, int(n), seed, scenario, a.world_seed, a.reference, a.diseases, sha,
                              a.workers):
                listing["sets"].append({"dir": f"{size}/{scenario}_{m['simulator']['censoring']}", "n": int(n),
                                        "seed": seed, "scenario": scenario,
                                        "censoring": m["simulator"]["censoring"],
                                        "files": {f"{t}.parquet": v["sha256"] for t, v in
                                                  {**m["tables"], **m["simulator"]["files"]}.items()}})
    for s in listing["sets"]:
        s["files"]["manifest.json"] = file_sha(root / s["dir"] / "manifest.json")
    (root / "MANIFEST.json").write_text(json.dumps(listing, indent=1) + "\n")
    for path in sorted(root.rglob("*"), reverse=True) + [root]:
        path.chmod(0o550 if path.is_dir() else 0o440)


if __name__ == "__main__":
    main()
