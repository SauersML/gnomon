"""Analysis frames (SPEC sections 2-3, as amended by the audit).

Hand-built people (one per rule), an independent per-person reference on tie-heavy random tables, the
split, and the 400k-person scale.
"""
from __future__ import annotations

import bisect
import datetime
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import cohort, phenotypes  # noqa: E402
from study_fixtures import CUTOFF, synthetic_tables  # noqa: E402

D = datetime.date
CONFIG = phenotypes.CohortConfig(seed=918273645, num_pcs=6)
DIABETES = phenotypes.Disease("t2d", "44054006", "PGS000011", None, (("46635009", "type 1 diabetes"),))
BREAST = phenotypes.Disease("breast", "254837009", "PGS000022", "female")
PROSTATE = phenotypes.Disease("prostate", "399068003", "PGS000033", "male")
HYPERTENSION = phenotypes.Disease("htn", "38341003", "PGS000044")
DISEASES = (DIABETES, BREAST, PROSTATE, HYPERTENSION)


# --------------------------------------------------------------------------- #
# hand-built people: one per rule
# --------------------------------------------------------------------------- #
MALE, FEMALE = 45880669, 45878463
TEMPLATE = dict(birth=D(1960, 1, 1), sex=MALE, baseline=D(2019, 1, 1), obs_start=D(2010, 1, 1),
                obs_end=D(2023, 6, 1), death=None, state="MN", site="EHR site 7", zip3=553, deprivation=0.3,
                ancestry="eur", related=False, pcs=True, scored=True, missing={}, records={})
LANDMARK = D(2019, 6, 30)  # baseline + 180 days
T2D, T1D, BRCA = "44054006", "46635009", "254837009"


def people():
    rows = {
        101: dict(ancestry=None), 102: dict(related=True), 103: dict(pcs=False), 104: dict(scored=False),
        105: dict(sex=1177221),
        106: dict(baseline=None, obs_start=None, obs_end=None, site=None, zip3=None, deprivation=None),
        107: dict(birth=D(2002, 1, 2)), 108: dict(obs_start=None, obs_end=None),
        109: dict(obs_start=D(2019, 1, 1) - datetime.timedelta(days=300)),
        1: {},
        2: dict(records={T2D: (D(2020, 1, 1), D(2020, 3, 1), 3)}),
        3: dict(sex=FEMALE, records={T2D: (D(2020, 5, 5), None, 1)}),
        4: dict(records={T2D: (D(2018, 5, 1), D(2020, 1, 1), 2)}),
        5: dict(records={T2D: (LANDMARK, D(2021, 1, 1), 2)}),
        6: dict(death=D(2019, 3, 1)),
        7: dict(obs_end=D(2019, 5, 1)),
        8: dict(death=D(2021, 1, 1), records={T2D: (D(2020, 6, 1), D(2021, 6, 1), 2)}),
        9: dict(death=D(2021, 2, 2), records={T2D: (D(2020, 2, 2), D(2021, 2, 2), 2)}),
        10: dict(records={T2D: (D(2022, 1, 1), D(2023, 8, 1), 2)}),
        11: dict(death=D(2023, 6, 1)),
        12: dict(sex=FEMALE, records={BRCA: (D(2020, 1, 1), D(2020, 2, 1), 2)}),
        13: dict(records={T2D: (D(2020, 1, 1), D(2020, 2, 1), 2), T1D: (D(2015, 1, 1), D(2016, 1, 1), 2)}),
        14: dict(records={T1D: (D(2015, 1, 1), None, 1)}),
        15: dict(missing={"PGS000011": 100.0}),
        16: dict(obs_end=D(2024, 1, 1)),
        17: dict(state="PR", site=None),
        18: dict(state=None, zip3=None, deprivation=None, ancestry="afr"),
        19: dict(records={T2D: (D(2020, 1, 1), D(2020, 1, 10), 2)}),
        # Audit C3: the exclusion rule met after the landmark ends survival follow-up there.
        20: dict(records={T2D: (D(2021, 1, 1), D(2021, 6, 1), 2), T1D: (D(2020, 1, 1), D(2020, 9, 1), 2)}),
        21: dict(records={T2D: (D(2020, 1, 1), D(2020, 5, 5), 2), T1D: (D(2020, 2, 2), D(2020, 5, 5), 2)}),
        22: dict(death=D(2021, 3, 3), records={T1D: (D(2020, 1, 1), D(2021, 3, 3), 2)}),
        # SPEC section 3 (21:22Z): follow-up ends at the last EHR record, not the observation period.
        23: dict(ehr_end=D(2021, 12, 31)),
        24: dict(ehr_end=D(2019, 5, 5)),
        # Survival eligibility is known at the landmark: an EHR begun after it cannot enter.
        25: dict(ehr_start=D(2019, 9, 1)),
    }
    people = {pid: {**TEMPLATE, **row} for pid, row in rows.items()}
    for row in people.values():  # by default the EHR spans the observation period
        row.setdefault("ehr_start", row["obs_start"])
        row.setdefault("ehr_end", row["obs_end"])
    return people


def to_tables(rows, scores=("PGS000011", "PGS000022", "PGS000033", "PGS000044"), prune_unmatched=0):
    ids = list(rows)
    column = lambda key, kind: pa.array([rows[i][key] for i in ids], kind)  # noqa: E731
    person = pa.table({
        "person_id": pa.array(ids, pa.int64()), "birth_date": column("birth", pa.date32()),
        "sex_at_birth_concept_id": column("sex", pa.int64()),
        "race_concept_id": pa.array([0] * len(ids), pa.int64()),
        "ethnicity_concept_id": pa.array([0] * len(ids), pa.int64()),
        "baseline_date": column("baseline", pa.date32()), "obs_start": column("obs_start", pa.date32()),
        "obs_end": column("obs_end", pa.date32()), "ehr_start": column("ehr_start", pa.date32()),
        "ehr_end": column("ehr_end", pa.date32()), "death_date": column("death", pa.date32()),
        "state": column("state", pa.string()), "ehr_site": column("site", pa.string()),
        "zip3": column("zip3", pa.int32()),
        "zip3_post_baseline": pa.array([None if rows[i]["zip3"] is None else False for i in ids], pa.bool_()),
        "deprivation_index": column("deprivation", pa.float64()),
    })
    in_ancestry = [i for i in ids if rows[i]["ancestry"]]
    ancestry = pa.table({"person_id": pa.array(in_ancestry, pa.int64()),
                         "ancestry_pred": pa.array([rows[i]["ancestry"] for i in in_ancestry], pa.string()),
                         "related_excluded": pa.array([rows[i]["related"] for i in in_ancestry], pa.bool_())})
    with_pcs = [i for i in ids if rows[i]["pcs"]]
    pcs = pa.table({"person_id": pa.array(with_pcs, pa.int64()),
                    **{f"PC{k}": pa.array([0.01 * k * (i % 7) for i in with_pcs], pa.float64())
                       for k in range(1, 9)}})
    scored = [i for i in ids if rows[i]["scored"]]
    columns = {"person_id": pa.array(scored, pa.int64())}
    for pgs in scores:
        missing = [rows[i]["missing"].get(pgs, 1.0) for i in scored]
        columns[pgs] = pa.array([None if m == 100 else 0.1 * i for i, m in zip(scored, missing)], pa.float64())
        columns[f"{pgs}_missing_pct"] = pa.array(missing, pa.float64())
    records = [(code, i, *record) for i in ids for code, record in rows[i]["records"].items()]
    condition = pa.table({"snomed_code": pa.array([r[0] for r in records], pa.string()),
                          "person_id": pa.array([r[1] for r in records], pa.int64()),
                          "first_date": pa.array([r[2] for r in records], pa.date32()),
                          "second_date": pa.array([r[3] for r in records], pa.date32()),
                          "n_dates": pa.array([r[4] for r in records], pa.int32())})
    codes = ["38341003", T2D, BRCA, "399068003", T1D]
    root = pa.table({"snomed_code": pa.array(codes, pa.string()),
                     "concept_id": pa.array(range(len(codes)), pa.int64()),
                     "concept_name": pa.array(codes, pa.string()),
                     "n_descendants": pa.array([1] * len(codes), pa.int64())})
    spec = {"source": "simulator", "seed": 1, "snomed_codes": codes, "scores": list(scores), "num_pcs": 8,
            "ses_available": True, "cdr_cutoff": CUTOFF.isoformat(), "cdr_cutoff_source": "simulator",
            "prune_unmatched": prune_unmatched}
    return {"person": person, "condition": condition, "root": root, "ancestry": ancestry, "pcs": pcs,
            "scores": pa.table(columns)}, spec


@pytest.fixture(scope="module")
def handmade(tmp_path_factory):
    directory = tmp_path_factory.mktemp("handmade")
    cohort.write_tables(directory, *to_tables(people()))
    return cohort.ParquetSource(directory)


def age(on, birth=D(1960, 1, 1)):
    return (on - birth).days / 365.25


def by_id(frame):
    return frame.set_index("person_id")


def steps(flow):
    return [(s["step"], s["n"]) for s in flow]


def test_base_removes_one_person_per_rule(handmade):
    base = phenotypes.base_cohort(handmade, CONFIG)
    assert [(s["step"], s["n"], s["removed"]) for s in base.flow] == [
        ("cdr_persons", 34, 0), ("in_ancestry", 33, 1), ("not_related_excluded", 32, 1), ("has_pcs", 31, 1),
        ("scored", 30, 1), ("sex_male_or_female", 29, 1), ("has_baseline", 28, 1), ("adult_at_baseline", 27, 1),
        ("covering_observation_period", 26, 1), ("lookback", 25, 1)]
    assert sorted(base.frame.person_id) == list(range(1, 26))
    frame = by_id(base.frame)
    assert (frame.loc[1, "division"], frame.loc[1, "region"]) == ("West North Central", "Midwest")
    assert (frame.loc[17, "division"], frame.loc[17, "region"], frame.loc[17, "ehr_site"]) == (
        "unknown", "unknown", "unknown")
    assert frame.loc[18, "region"] == "unknown" and frame.loc[18, "ses_quartile"] == "unknown"
    assert frame.loc[1, "ses_quartile"] == "Q1" and frame.loc[1, "ehr_site"] == "EHR site 7"
    assert frame.loc[1, "age_band"] == "40-59" and frame.loc[1, "age_baseline"] == age(D(2019, 1, 1))
    assert frame.loc[1, "lookback_days"] == (D(2019, 1, 1) - D(2010, 1, 1)).days
    # Audit N3: window covariates known at baseline.
    assert frame.loc[1, "lookback_years"] == (D(2019, 1, 1) - D(2010, 1, 1)).days / 365.25
    assert frame.loc[1, "admin_years"] == (CUTOFF - D(2019, 1, 1)).days / 365.25
    assert frame.loc[1, "baseline_year"] == 2019 and frame.loc[18, "ancestry"] == "afr"


def test_disease_rules_and_binary_outcome(handmade):
    base, frames = phenotypes.build_frames(handmade, [DIABETES, BREAST], CONFIG)
    t2d = frames["t2d"]
    assert steps(t2d.flow["disease"]) == [("base", 25), ("pgs_present", 24)]
    assert steps(t2d.flow["binary"]["steps"]) == [("disease_rows", 24), ("exclusion_46635009", 20)]
    binary = by_id(t2d.binary)
    assert 15 not in binary.index and 14 in binary.index
    assert not {13, 20, 21, 22} & set(binary.index)  # the exclusion rule met by the cutoff removes
    assert sorted(binary.index[binary.y == 1]) == [2, 4, 5, 8, 9, 10, 19]
    assert binary.loc[3, "y"] == 0 and binary.loc[3, "n_dates"] == 1  # a single record is a non-case
    assert {k: t2d.flow["binary"][k] for k in ("cases", "non_cases", "single_record")} == {
        "cases": 7, "non_cases": 13, "single_record": 1}
    # Audit S2: nothing measured after baseline enters the binary frame.
    assert not {"age_last", "exit_age", "entry_age", "event", "followup"} & set(binary.columns)
    breast = frames["breast"]
    assert steps(breast.flow["disease"]) == [("base", 25), ("pgs_present", 25), ("sex_female", 2)]
    assert dict(zip(breast.binary.person_id, breast.binary.y)) == {3: 0, 12: 1}
    assert t2d.flow["by_ancestry"]["eur"]["binary_cases"] == 7


def test_survival_entry_exit_competing_death_and_late_exclusions(handmade):
    base, frames = phenotypes.build_frames(handmade, [DIABETES], CONFIG)
    survival = frames["t2d"].flow["survival"]
    assert steps(survival["steps"]) == [
        ("disease_rows", 24), ("exclusion_46635009_by_landmark", 23), ("no_record_by_landmark", 21),
        ("alive_at_landmark", 20), ("ehr_by_landmark", 19), ("cutoff_past_landmark", 19)]
    frame = by_id(frames["t2d"].survival)
    assert sorted(frame.index) == [1, 2, 3, 7, 8, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    expected = {1: (0, D(2023, 6, 1)), 2: (1, D(2020, 3, 1)), 3: (0, D(2023, 6, 1)), 8: (2, D(2021, 1, 1)),
                9: (1, D(2021, 2, 2)), 10: (0, D(2023, 6, 1)), 11: (2, D(2023, 6, 1)),
                16: (0, CUTOFF), 19: (1, D(2020, 1, 10)),
                20: (0, D(2020, 9, 1)), 21: (1, D(2020, 5, 5)), 22: (2, D(2021, 3, 3)),
                23: (0, D(2021, 12, 31)),  # censored at the EHR end, before the observation period's
                7: (0, LANDMARK), 24: (0, LANDMARK)}  # no EHR record after the landmark: censored at entry
    for pid, (event, exit_date) in expected.items():
        assert frame.loc[pid, "event"] == event, pid
        assert frame.loc[pid, "exit_age"] == age(exit_date), pid
    assert (frame.entry_age == age(LANDMARK)).all()  # entry at the landmark, not at baseline (audit M12)
    assert (frame.entry_year == 2019).all()
    assert np.allclose(frame.followup, frame.exit_age - frame.entry_age)
    assert sorted(frame.index[frame.followup == 0]) == [7, 24] and (frame.followup >= 0).all()
    # 20 is censored at its later exclusion match; 21's match falls on its event day, and the event counts.
    assert survival["events"] == {"censored": 12, "disease": 4, "death": 3}
    assert survival["exclusion_exits"] == 1 and survival["single_record_at_risk"] == 1
    assert survival["censored_at_entry"] == 2


def test_sensitivity_variants(handmade):
    base = phenotypes.base_cohort(handmade, CONFIG)
    rows = phenotypes.disease_rows(base, handmade, DIABETES)
    first, _ = phenotypes.survival_frame(rows, base, CONFIG, onset="first")
    first = by_id(first)
    for pid, when in {2: D(2020, 1, 1), 8: D(2020, 6, 1), 9: D(2020, 2, 2), 10: D(2022, 1, 1),
                      21: D(2020, 1, 1)}.items():
        assert (first.loc[pid, "event"], first.loc[pid, "exit_age"]) == (1, age(when)), pid
    assert (first.loc[20, "event"], first.loc[20, "exit_age"]) == (0, age(D(2020, 9, 1)))
    cutoff, counts = phenotypes.survival_frame(rows, base, CONFIG, censor="cutoff")
    cutoff = by_id(cutoff)
    assert (cutoff.loc[10, "event"], cutoff.loc[10, "exit_age"]) == (1, age(D(2023, 8, 1)))
    assert (cutoff.loc[1, "event"], cutoff.loc[1, "exit_age"]) == (0, age(CUTOFF))
    # Censoring at the cutoff alone, the EHR end no longer ends follow-up.
    assert (cutoff.loc[23, "event"], cutoff.loc[23, "exit_age"]) == (0, age(CUTOFF))
    assert (cutoff.loc[24, "event"], cutoff.loc[24, "exit_age"]) == (0, age(CUTOFF))
    assert counts["censored_at_entry"] == 0 and 25 not in cutoff.index
    assert cutoff.loc[11, "event"] == 2
    assert counts["steps"][-1]["step"] == "cutoff_past_landmark"
    with pytest.raises(ValueError):
        phenotypes.survival_frame(rows, base, CONFIG, onset="last")


def test_tables_must_exclude_exactly_the_declared_branches(tmp_path, handmade):
    branched = phenotypes.Disease("t2d", T2D, "PGS000011", None, DIABETES.exclusions, (("123456789", "stage"),))
    with pytest.raises(ValueError, match="exclude branches"):
        phenotypes.build_frames(handmade, [branched], CONFIG)
    tables, spec = to_tables(people())
    cohort.write_tables(tmp_path, tables, {**spec, "excluded_branches": {T2D: ["123456789"]}})
    extracted = cohort.ParquetSource(tmp_path)
    assert len(phenotypes.build_frames(extracted, [branched], CONFIG)[1]["t2d"].binary)
    with pytest.raises(ValueError, match="exclude branches"):
        phenotypes.build_frames(extracted, [DIABETES], CONFIG)


def test_the_prespecified_horizon_rule(handmade):
    base = phenotypes.base_cohort(handmade, CONFIG)
    distribution = phenotypes.followup_distribution(base, CONFIG)
    # Everyone enters on 2019-06-30, 4.25 years before the cutoff.
    assert distribution["administrative"]["quantiles"][0] == (CUTOFF - LANDMARK).days / 365.25
    assert phenotypes.choose_horizons(distribution, CONFIG) == [1, 2, 3, 4]
    strict = phenotypes.CohortConfig(seed=CONFIG.seed, horizon_candidates=[5, 6])
    with pytest.raises(ValueError, match="no horizon"):
        phenotypes.choose_horizons(phenotypes.followup_distribution(base, strict), strict)


def test_prune_outside_the_ancestry_universe_is_refused(tmp_path):
    cohort.write_tables(tmp_path, *to_tables(people(), prune_unmatched=3))
    source = cohort.ParquetSource(tmp_path)
    with pytest.raises(ValueError, match="outside the ancestry universe"):
        phenotypes.base_cohort(source, CONFIG)


# --------------------------------------------------------------------------- #
# an independent per-person reference on tie-heavy random tables
# --------------------------------------------------------------------------- #
MASK = (1 << 64) - 1


def reference_unit(person_id, seed, purpose):
    def mix(x):
        x = (x + 0x9E3779B97F4A7C15) & MASK
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
        return x ^ (x >> 31)
    h = 0xCBF29CE484222325
    for byte in purpose.encode():
        h = ((h ^ byte) * 0x100000001B3) & MASK
    return (mix((person_id & MASK) ^ mix((seed & MASK) ^ h)) >> 11) / 2 ** 53


def reference_quantile(values, p):
    values = sorted(values)
    h = (len(values) - 1) * p
    lo = math.floor(h)
    return values[lo] if lo + 1 >= len(values) else values[lo] + (h - lo) * (values[lo + 1] - values[lo])


def reference_frames(tables, spec, disease, config):
    """Row by row, with calendar dates: the SPEC rules written a second, independent way."""
    ancestry = {r["person_id"]: r for r in tables["ancestry"].to_pylist()}
    pcs = {r["person_id"]: r for r in tables["pcs"].to_pylist()}
    scores = {r["person_id"]: r for r in tables["scores"].to_pylist()}
    records = {(r["snomed_code"], r["person_id"]): r for r in tables["condition"].to_pylist()}
    cutoff = D.fromisoformat(spec["cdr_cutoff"])
    sexes = {45880669: 1, 8507: 1, 45878463: 0, 8532: 0}
    base = []
    for p in tables["person"].to_pylist():
        pid, birth, baseline = p["person_id"], p["birth_date"], p["baseline_date"]
        if pid not in ancestry or ancestry[pid]["related_excluded"] or pid not in pcs or pid not in scores:
            continue
        if p["sex_at_birth_concept_id"] not in sexes or baseline is None:
            continue
        if (baseline - birth).days / 365.25 < config.min_age or p["obs_start"] is None:
            continue
        if (baseline - p["obs_start"]).days < config.lookback_days:
            continue
        base.append(p)
    deprivations = [p["deprivation_index"] for p in base if p["deprivation_index"] is not None]
    ses_cuts = [reference_quantile(deprivations, q) for q in (0.25, 0.5, 0.75)]
    lookbacks = [(p["baseline_date"] - p["obs_start"]).days for p in base]
    lookback_cuts = [reference_quantile(lookbacks, q) for q in (1 / 3, 2 / 3)]
    binary, survival, ties = {}, {}, 0
    for p in base:
        pid, birth, baseline = p["person_id"], p["birth_date"], p["baseline_date"]
        sex = sexes[p["sex_at_birth_concept_id"]]
        if scores[pid][disease.pgs] is None:
            continue
        if disease.sex is not None and sex != (1 if disease.sex == "male" else 0):
            continue
        record = records.get((disease.snomed_code, pid))
        n = record["n_dates"] if record else 0
        met = [records[(code, pid)]["second_date"] for code, _ in disease.exclusions
               if (code, pid) in records and records[(code, pid)]["n_dates"] >= 2]
        excluded = min(met) if met else None
        age_baseline = (baseline - birth).days / 365.25
        division = next((d for d, states in phenotypes.DIVISIONS.items() if p["state"] in states), "unknown")
        dep = p["deprivation_index"]
        lookback = (baseline - p["obs_start"]).days
        common = {
            "n_dates": n, "sex": sex, "age_baseline": age_baseline, "baseline_year": baseline.year,
            "lookback_days": lookback, "lookback_years": lookback / 365.25,
            "admin_years": (cutoff - baseline).days / 365.25,
            "division": division, "region": phenotypes.DIVISION_REGION.get(division, "unknown"),
            "ehr_site": p["ehr_site"] or "unknown",
            "age_band": "18-39" if age_baseline < 40 else "40-59" if age_baseline < 60 else
                        "60-74" if age_baseline < 75 else "75+",
            "ses_quartile": "unknown" if dep is None else f"Q{bisect.bisect_left(ses_cuts, dep) + 1}",
            "lookback_tertile": f"T{bisect.bisect_left(lookback_cuts, lookback) + 1}",
            "test": reference_unit(pid, config.seed, "test") < config.test_fraction,
        }
        common["fold"] = -1 if common["test"] else min(int(reference_unit(pid, config.seed, "fold")
                                                             * config.dev_folds), config.dev_folds - 1)
        if excluded is None or excluded > cutoff:
            binary[pid] = {**common, "y": int(n >= 2 and record["second_date"] <= cutoff)}
        landmark = baseline + datetime.timedelta(days=config.landmark_days)
        if excluded is not None and excluded <= landmark:
            continue
        if record and record["first_date"] <= landmark:
            continue
        if p["death_date"] is not None and p["death_date"] <= landmark:
            continue
        if p["ehr_start"] is None or p["ehr_start"] > landmark or cutoff <= landmark:
            continue
        observed_to = min(p["ehr_end"], cutoff)
        event_date = record["second_date"] if n >= 2 else None
        dates = [d for d in (event_date, p["death_date"], observed_to, excluded) if d is not None]
        exit_date = min(dates)
        ties += dates.count(exit_date) > 1
        exit_date = max(exit_date, landmark)  # an EHR silent after the landmark is censored at entry
        if event_date == exit_date:
            event = 1
        elif p["death_date"] == exit_date:
            event = 2
        else:
            event = 0
        survival[pid] = {**common, "entry_age": (landmark - birth).days / 365.25, "entry_year": landmark.year,
                         "exit_age": (exit_date - birth).days / 365.25, "event": event}
    return binary, survival, ties


def assert_frame_matches(frame, reference, columns):
    assert sorted(frame.person_id) == sorted(reference)
    for column in columns:
        got = dict(zip(frame.person_id, frame[column].tolist()))
        mismatched = [pid for pid in reference if got[pid] != reference[pid][column]]
        assert not mismatched, (column, len(mismatched), mismatched[:3])


@pytest.mark.parametrize("seed", [11, 12])
def test_frames_match_the_reference(tmp_path, seed):
    tabs, spec = synthetic_tables(3000, seed=seed, grid_days=30, record_rate=0.35, late_ehr=0.1)
    cohort.write_tables(tmp_path, tabs, spec)
    source = cohort.ParquetSource(tmp_path)
    base, frames = phenotypes.build_frames(source, DISEASES, CONFIG)
    shared = ["n_dates", "sex", "age_baseline", "baseline_year", "lookback_days", "lookback_years",
              "admin_years", "division", "region", "ehr_site", "age_band", "ses_quartile", "lookback_tertile",
              "test", "fold"]
    ties = exclusion_exits = at_entry = late_start = 0
    for disease in DISEASES:
        binary, survival, tied = reference_frames(tabs, spec, disease, CONFIG)
        ties += tied
        assert_frame_matches(frames[disease.slug].binary, binary, ["y", *shared])
        assert_frame_matches(frames[disease.slug].survival, survival,
                             ["entry_age", "entry_year", "exit_age", "event", *shared])
        assert set(frames[disease.slug].survival.event) == {0, 1, 2}
        exclusion_exits += frames[disease.slug].flow["survival"]["exclusion_exits"]
        at_entry += frames[disease.slug].flow["survival"]["censored_at_entry"]
        remaining = dict(steps(frames[disease.slug].flow["survival"]["steps"]))
        late_start += remaining["alive_at_landmark"] - remaining["ehr_by_landmark"]
    assert ties > 0  # same-day disease, death, exclusion and censoring exits were exercised
    assert exclusion_exits > 0  # and so were exclusions met after the landmark
    assert at_entry > 0 and late_start > 0  # EHRs silent after the landmark, and EHRs begun after it


def test_followup_distribution_is_outcome_blind(tmp_path):
    tabs, spec = synthetic_tables(3000, seed=5, record_rate=0.35, late_ehr=0.1)
    cohort.write_tables(tmp_path, tabs, spec)
    source = cohort.ParquetSource(tmp_path)
    base = phenotypes.base_cohort(source, CONFIG)
    result = phenotypes.followup_distribution(base, CONFIG)
    cutoff = D.fromisoformat(spec["cdr_cutoff"])
    ids = set(base.frame.person_id)
    administrative, observed, years, after, with_ehr = [], [], {}, [], []
    for p in tabs["person"].to_pylist():
        if p["person_id"] not in ids:
            continue
        landmark = p["baseline_date"] + datetime.timedelta(days=CONFIG.landmark_days)
        after.append(p["ehr_end"] is not None and p["obs_end"] > p["ehr_end"])
        with_ehr.append(p["ehr_end"] is not None)
        if p["ehr_end"] is None:
            continue
        observed_to = min(p["ehr_end"], cutoff)
        alive = p["death_date"] is None or p["death_date"] > landmark
        if alive and p["ehr_start"] <= landmark < cutoff:
            administrative.append((cutoff - landmark).days / 365.25)
            observed.append(max((observed_to - landmark).days, 0) / 365.25)
            years[landmark.year] = years.get(landmark.year, 0) + 1
    for name, spans in (("administrative", administrative), ("observed", observed)):
        summary = result[name]
        assert summary["n"] == len(spans)
        assert summary["fraction_reaching"]["2"] == sum(s >= 2 for s in spans) / len(spans)
        assert np.allclose(summary["quantiles"], [reference_quantile(spans, q) for q in result["quantiles"]])
    assert {y: v["n"] for y, v in result["by_entry_year"].items()} == years
    ehr = result["ehr"]
    assert ehr["n"] == len(with_ehr) and ehr["fraction_without_ehr"] == with_ehr.count(False) / len(with_ehr)
    assert ehr["fraction_obs_end_after_ehr_end"] == sum(after) / sum(with_ehr)
    assert 0 < ehr["fraction_obs_end_after_ehr_end"] < 1 and ehr["median_positive_gap_years"] > 0
    # No condition table is read: the same numbers come back with every record removed.
    empty = {**tabs, "condition": tabs["condition"].slice(0, 0)}
    cohort.write_tables(tmp_path / "empty", empty, spec)
    again = phenotypes.base_cohort(cohort.ParquetSource(tmp_path / "empty"), CONFIG)
    assert phenotypes.followup_distribution(again, CONFIG) == result


# --------------------------------------------------------------------------- #
# the split
# --------------------------------------------------------------------------- #
def test_split_matches_its_reference_and_proportions():
    rng = np.random.default_rng(0)
    ids = np.concatenate([rng.integers(-2 ** 63, 2 ** 63 - 1, 500, dtype=np.int64),
                          np.arange(1_000_000, 1_000_500, dtype=np.int64)])
    for seed, purpose in ((0, "test"), (918273645, "fold"), (2 ** 63 + 5, "test")):
        got = phenotypes.unit_hash(ids, seed, purpose)
        assert got.tolist() == [reference_unit(int(i), seed, purpose) for i in ids]
    ids = rng.choice(10 ** 10, size=200_000, replace=False).astype(np.int64)
    test, fold = phenotypes.split(ids, CONFIG)
    sd = math.sqrt(0.2 * 0.8 / len(ids))
    assert abs(test.mean() - 0.2) < 5 * sd
    counts = np.bincount(fold[~test], minlength=5)
    assert (np.abs(counts / counts.sum() - 0.2) < 5 * math.sqrt(0.2 * 0.8 / counts.sum())).all()
    other, _ = phenotypes.split(ids, phenotypes.CohortConfig(seed=CONFIG.seed + 1))
    assert 0.18 < (other & test).sum() / test.sum() < 0.22  # a new seed redraws the test set independently
    shuffled = rng.permutation(len(ids))
    assert (phenotypes.split(ids[shuffled], CONFIG)[0] == test[shuffled]).all()


def test_frames_do_not_depend_on_row_order(tmp_path):
    tabs, spec = synthetic_tables(2000, seed=8, grid_days=30, record_rate=0.3)
    rng = np.random.default_rng(1)
    shuffled = {name: table.take(rng.permutation(table.num_rows)) for name, table in tabs.items()}
    results = []
    for label, tables in (("a", tabs), ("b", shuffled)):
        cohort.write_tables(tmp_path / label, tables, spec)
        _, frames = phenotypes.build_frames(cohort.ParquetSource(tmp_path / label), DISEASES, CONFIG)
        results.append({slug: (f.binary.sort_values("person_id").reset_index(drop=True),
                               f.survival.sort_values("person_id").reset_index(drop=True))
                        for slug, f in frames.items()})
    text = {c: str for c in phenotypes.STRATA}
    for slug in results[0]:
        for a, b in zip(results[0][slug], results[1][slug]):
            assert a.astype(text).equals(b.astype(text))


def test_config_and_disease_parsing():
    assert phenotypes.CohortConfig.from_json({"seed": 7, "landmark_days": 180}).landmark_days == 180
    with pytest.raises(ValueError, match="unknown"):
        phenotypes.CohortConfig.from_json({"seed": 7, "row_cap": 20000})
    for spent in phenotypes.SPENT_SEEDS:
        with pytest.raises(ValueError, match="already seen"):
            phenotypes.CohortConfig(seed=spent)
    diseases = phenotypes.load_diseases({"locked": True, "diseases": [
        {"slug": "type_2_diabetes", "snomed_code": "44054006", "omop_concept_id": 201826, "pgs": "PGS002308",
         "sex": None, "concept_name": "Type 2 diabetes mellitus",
         "exclusions": [{"snomed_code": "46635009", "omop_concept_id": 201254,
                         "reason": "type 1 diabetes is a different disease"}]},
        {"slug": "breast_cancer", "snomed_code": 254837009, "pgs": "PGS000004", "sex": "female"}]})
    assert diseases[0].snomed_codes == ("44054006", "46635009") and diseases[1].snomed_code == "254837009"
    assert phenotypes.phenotype_codes(diseases) == {
        "snomed_codes": {"44054006": 201826, "46635009": 201254, "254837009": None}, "excluded_branches": {}}
    with pytest.raises(ValueError, match="does not implement"):
        phenotypes.load_diseases([{"slug": "x", "snomed_code": "44054006", "pgs": "PGS000011", "onset": "first"}])
    with pytest.raises(ValueError, match="does not implement"):
        phenotypes.load_diseases([{"slug": "x", "snomed_code": "44054006", "pgs": "PGS000011",
                                   "exclusions": [{"snomed_code": "46635009", "reason": "r", "window": "ever"}]}])
    with pytest.raises(ValueError, match="written reason"):
        phenotypes.load_diseases([{"slug": "x", "snomed_code": "44054006", "pgs": "PGS000011",
                                   "exclusions": [{"snomed_code": "1"}]}])
    with pytest.raises(ValueError, match="sex restriction"):
        phenotypes.load_diseases([{"slug": "x", "snomed_code": "44054006", "pgs": "PGS000011", "sex": "f"}])


def test_the_locked_disease_list_loads():
    path = Path(__file__).resolve().parents[1] / "study" / "diseases.json"
    if not path.exists():
        pytest.fail(f"{path} is missing: the locked disease list must ship with the study")
    diseases = phenotypes.load_diseases(json.loads(path.read_text()))
    assert len(diseases) == 9  # asthma, gout and glaucoma left with no cached score (2026-09-24)
    assert {d.slug: d.sex for d in diseases if d.sex} == {"breast_cancer": "female", "prostate_cancer": "male"}
    codes = phenotypes.phenotype_codes(diseases)
    assert all(concept is not None for concept in codes["snomed_codes"].values())
    assert codes["excluded_branches"] == {"709044004": {"431855005": 443614, "431856006": 443601}}
    ckd = next(d for d in diseases if d.slug == "chronic_kidney_disease")
    assert ckd.branch_codes == ("431855005", "431856006") and ckd.snomed_codes == ("709044004",)


# --------------------------------------------------------------------------- #
# scale: the full AoU-shaped cohort
# --------------------------------------------------------------------------- #
def test_build_frames_at_400k(tmp_path):
    tabs, spec = synthetic_tables(400_000, seed=21, record_rate=0.15)
    cohort.write_tables(tmp_path, tabs, spec)
    start = time.perf_counter()
    source = cohort.ParquetSource(tmp_path)
    opened = time.perf_counter()
    diseases = [phenotypes.Disease(f"{d.slug}{k}", d.snomed_code, d.pgs, d.sex, d.exclusions)
                for k in range(3) for d in DISEASES]  # 12 diseases, as in the study
    base, frames = phenotypes.build_frames(source, diseases, CONFIG)
    built = time.perf_counter()
    rows = sum(len(f.binary) + len(f.survival) for f in frames.values())
    print(f"\n400k: open+validate {opened - start:.2f} s, base+12 diseases x 2 frames {built - opened:.2f} s, "
          f"{len(base.frame)} base rows, {rows} frame rows")
    assert len(base.frame) > 200_000  # the fixture draws about 65% eligible (ancestry, prune, PCs, lookback ...)
    assert built - opened < 60, "frame construction is no longer vectorized"


def test_the_shipped_study_config_loads():
    """study.json's cohort block is exactly CohortConfig's fields (from_json refuses unknown keys)."""
    config = json.loads((Path(__file__).resolve().parents[1] / "study.json").read_text())
    cohort_config = phenotypes.CohortConfig.from_json(config["cohort"])
    assert cohort_config.seed not in phenotypes.SPENT_SEEDS
