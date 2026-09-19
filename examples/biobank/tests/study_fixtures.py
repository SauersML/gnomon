"""Synthetic SCHEMA.md tables for the cohort and phenotype tests.

This is a test fixture, not the study simulator: it draws dates, codes and
records with no disease model, only to cover every table rule and tie at any
size. `grid_days` rounds every event date to a coarse grid so same-day ties
between records, deaths and observation ends are common.
"""
from __future__ import annotations

import datetime

import numpy as np
import pyarrow as pa

EPOCH = datetime.date(1970, 1, 1)
CUTOFF = datetime.date(2023, 10, 1)
ROOTS = ("38341003", "44054006", "49436004", "254837009", "399068003", "46635009")
SCORES = ("PGS000011", "PGS000022", "PGS000033", "PGS000044")
STATES = ("CA", "NY", "TX", "MN", "FL", "IL", "PA", "WA", "MA", "AZ", "PR", "GU", "DC", "AK")
SEX_CODES = (45880669, 45878463, 8507, 8532, 1177221, 0)
SEX_WEIGHTS = (0.40, 0.54, 0.02, 0.02, 0.01, 0.01)
LABELS = ("afr", "amr", "eas", "eur", "mid", "sas")


def day(value):
    return (value - EPOCH).days


def date_array(days_):
    """float day numbers (NaN = null) -> Arrow date32."""
    days_ = np.asarray(days_, dtype=np.float64)
    null = np.isnan(days_)
    return pa.array(np.where(null, 0, days_).astype(np.int32), pa.int32(), mask=null).cast(pa.date32())


def manifest(roots=ROOTS, scores=SCORES, num_pcs=8, ses_available=True, cutoff=CUTOFF):
    return {"source": "simulator", "seed": 0, "snomed_codes": list(roots), "scores": list(scores),
            "num_pcs": num_pcs, "ses_available": ses_available, "cdr_cutoff": cutoff.isoformat(),
            "cdr_cutoff_source": "simulator"}


def synthetic_tables(n, seed=0, roots=ROOTS, scores=SCORES, num_pcs=8, grid_days=None, record_rate=0.2,
                     late_ehr=0.0):
    """Every contract table for n people: a mix of eligible and ineligible, cases and single records.

    late_ehr is the fraction of EHRs that begin inside the observation period after baseline, some of them
    after the landmark."""
    rng = np.random.default_rng(seed)
    cutoff = day(CUTOFF)
    snap = (lambda d: np.floor(d / grid_days) * grid_days) if grid_days else (lambda d: np.floor(d))
    person_id = rng.choice(10 ** 9, size=n, replace=False).astype(np.int64) + 1_000_000
    birth = np.floor(day(datetime.date(1932, 1, 1)) + rng.uniform(0, 72 * 365.25, n))
    enrol = day(datetime.date(2017, 6, 1))
    baseline = np.floor(enrol + rng.uniform(0, cutoff - 60 - enrol, n))
    baseline[rng.random(n) < 0.03] = np.nan
    covered = ~np.isnan(baseline) & (rng.random(n) < 0.95)
    obs_start = np.where(covered, baseline - np.floor(rng.exponential(6 * 365.25, n)), np.nan)
    obs_end = np.where(covered, np.minimum(cutoff, snap(baseline + rng.uniform(0, 7 * 365.25, n))), np.nan)
    obs_end = np.where(covered, np.maximum(obs_end, baseline), np.nan)
    # The EHR ends at or before the observation period, which also counts surveys and measurements.
    has_ehr = covered & (rng.random(n) < 0.97)
    ehr_start = np.where(has_ehr, obs_start, np.nan)
    if late_ehr:
        late = has_ehr & (rng.random(n) < late_ehr)
        ehr_start = np.where(late, np.floor(baseline + rng.random(n) * (obs_end - baseline)), ehr_start)
    ehr_end = np.where(rng.random(n) < 0.6, obs_end, snap(obs_end - rng.exponential(200, n)))
    ehr_end = np.where(has_ehr, np.maximum(ehr_end, ehr_start), np.nan)
    death = np.where(rng.random(n) < 0.08, snap(np.nan_to_num(baseline, nan=cutoff - 400)
                                                + rng.uniform(1, 6 * 365.25, n)), np.nan)
    death = np.where(death > cutoff, np.nan, death)
    death = np.where(~np.isnan(baseline) & (death <= baseline), baseline + 1, death)
    sex = rng.choice(SEX_CODES, size=n, p=SEX_WEIGHTS)
    state = rng.choice(np.array(STATES + (None,), dtype=object), size=n)
    sites = np.array([f"EHR site {100 + i}" for i in range(25)] + [None], dtype=object)
    ehr_site = np.where(np.isnan(baseline), None, rng.choice(sites, size=n))
    zip3 = rng.integers(0, 1000, n)
    zip3_null = (rng.random(n) < 0.05) | np.isnan(baseline)
    by_zip = rng.normal(0.35, 0.08, 1000)
    person = pa.table({
        "person_id": pa.array(person_id, pa.int64()),
        "birth_date": date_array(birth),
        "sex_at_birth_concept_id": pa.array(sex, pa.int64()),
        "race_concept_id": pa.array(rng.choice([8527, 8516, 0], size=n), pa.int64()),
        "ethnicity_concept_id": pa.array(rng.choice([38003563, 38003564, 0], size=n), pa.int64()),
        "baseline_date": date_array(baseline),
        "obs_start": date_array(obs_start),
        "obs_end": date_array(obs_end),
        "ehr_start": date_array(ehr_start),
        "ehr_end": date_array(ehr_end),
        "death_date": date_array(death),
        "state": pa.array(state, pa.string()),
        "ehr_site": pa.array(ehr_site, pa.string()),
        "zip3": pa.array(zip3, pa.int32(), mask=zip3_null),
        "zip3_post_baseline": pa.array(rng.random(n) < 0.02, pa.bool_(), mask=zip3_null),
        "deprivation_index": pa.array(by_zip[zip3], pa.float64(), mask=zip3_null),
    })

    in_ancestry = rng.random(n) < 0.95
    ancestry = pa.table({
        "person_id": pa.array(person_id[in_ancestry], pa.int64()),
        "ancestry_pred": pa.array(rng.choice(LABELS, size=int(in_ancestry.sum())), pa.string()),
        "related_excluded": pa.array(rng.random(int(in_ancestry.sum())) < 0.03, pa.bool_()),
    })
    with_pcs = rng.random(n) < 0.97
    pcs = {"person_id": pa.array(person_id[with_pcs], pa.int64())}
    for i in range(1, num_pcs + 1):
        pcs[f"PC{i}"] = pa.array(rng.normal(0, 1 / i, int(with_pcs.sum())), pa.float64())
    scored = rng.random(n) < 0.97
    m = int(scored.sum())
    columns = {"person_id": pa.array(person_id[scored], pa.int64())}
    for pgs in scores:
        missing = np.round(rng.uniform(0, 6, m), 2)
        missing[rng.random(m) < 0.004] = 100.0
        unscored = rng.random(m) < 0.01
        columns[pgs] = pa.array(rng.gamma(2.0, 1.0, m), pa.float64(), mask=unscored | (missing == 100))
        columns[f"{pgs}_missing_pct"] = pa.array(missing, pa.float64(), mask=unscored)
    score_table = pa.table(columns)

    parts = []
    anchor = np.where(np.isnan(obs_start), birth + 30 * 365.25, obs_start)
    horizon = np.where(np.isnan(obs_end), cutoff, obs_end) + 400
    for root in roots:
        has = rng.random(n) < record_rate
        k = int(has.sum())
        first = snap(anchor[has] + rng.uniform(0, 1, k) * (np.minimum(horizon[has], cutoff) - anchor[has]))
        first = np.clip(first, day(datetime.date(1990, 1, 1)), cutoff - 1)
        count = np.minimum(1 + rng.geometric(0.45, k) - 1 + (rng.random(k) < 0.6), 30).astype(np.int32)
        gap = np.maximum(grid_days or 1, snap(rng.exponential(300, k)))
        second = np.where(count >= 2, np.minimum(first + gap, cutoff), np.nan)
        count = np.where(second <= first, 1, count).astype(np.int32)
        second = np.where(count >= 2, second, np.nan)
        parts.append(pa.table({
            "snomed_code": pa.array(np.full(k, root, dtype=object), pa.string()),
            "person_id": pa.array(person_id[has], pa.int64()),
            "first_date": date_array(first), "second_date": date_array(second),
            "n_dates": pa.array(count, pa.int32()),
        }))
    condition = pa.concat_tables(parts)
    root = pa.table({"snomed_code": pa.array(list(roots), pa.string()),
                     "concept_id": pa.array(np.arange(len(roots)) + 3_000_000, pa.int64()),
                     "concept_name": pa.array([f"synthetic root {r}" for r in roots], pa.string()),
                     "n_descendants": pa.array(np.full(len(roots), 12), pa.int64())})
    tables = {"person": person, "condition": condition, "root": root, "ancestry": ancestry,
              "pcs": pa.table(pcs), "scores": score_table}
    return tables, manifest(roots, scores, num_pcs)
