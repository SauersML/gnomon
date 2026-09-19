"""The AoU SQL, executed: `AouSource` against a miniature OMOP CDR in DuckDB.

Each BigQuery statement is transpiled to DuckDB with sqlglot and run on
hand-built CDR tables with one row per rule: the consent baseline, the covering
observation period, primary death, PIIState, the modal pre-baseline EHR site,
the zip3 nearest baseline and its SES vintage, qualifying records net of
excluded branches, and the descendant log. This checks the queries' semantics;
BigQuery syntax and bytes are checked by `AouSource.plan()`'s dry runs in AoU.
Needs duckdb and sqlglot (MSI: the study-cohort pylib).
"""
from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pytest
import sqlglot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import cohort  # noqa: E402

D = datetime.date
CDR = "study.main"
T2D, CKD, STAGE1, STAGE2 = "44054006", "709044004", "431855005", "431856006"
EXTRA = list(range(1001, 1026))  # 25 more people, so descendant counts pass the >20 rule


class DuckClient:
    """The BoundedClient interface over DuckDB: each BigQuery statement is transpiled and executed."""

    def __init__(self, con):
        self.con, self.remaining, self.billed, self.job_ids, self.statements = con, 50 * 10 ** 9, 0, [], []

    def _duck(self, sql):
        return sqlglot.transpile(sql, read="bigquery", write="duckdb")[0]

    def estimate(self, sql, parameters=None):
        self._duck(sql)
        return 0

    def query(self, sql, parameters=None):
        duck = self._duck(sql)
        self.statements.append(duck)
        values = {name: value for name, (_, value) in (parameters or {}).items()
                  if re.search(rf"\${name}\b", duck)}
        result = self.con.execute(duck, values)
        return result.to_arrow_table() if hasattr(result, "to_arrow_table") else result.fetch_arrow_table()

    def columns(self, table_id):
        catalog, schema, table = table_id.split(".")
        rows = self.con.execute("SELECT column_name FROM information_schema.columns WHERE table_catalog = ? "
                                "AND table_schema = ? AND table_name = ?", [catalog, schema, table]).fetchall()
        return {row[0] for row in rows} or None


def create(con, name, **columns):
    con.register("staged", pa.table(columns))
    con.execute(f"CREATE TABLE {CDR}.{name} AS SELECT * FROM staged")
    con.unregister("staged")


def dates(values):
    return pa.array(values, pa.date32())


@pytest.fixture(scope="module")
def cdr():
    con = duckdb.connect()
    con.execute("ATTACH ':memory:' AS study")
    concepts = [  # (id, code, name, vocabulary, standard, class)
        (1000, "PIIModule", "Consent PII", "PPI", None, "Module"), (1001, "ConsentQ", "consent question", "PPI",
                                                                        None, "Question"),
        (2010, T2D, "Type 2 diabetes mellitus", "SNOMED", "S", "Clinical Finding"),
        (2011, "1111111", "Type 2 diabetes without complication", "SNOMED", "S", "Clinical Finding"),
        (709, CKD, "Chronic kidney disease", "SNOMED", "S", "Clinical Finding"),
        (7091, STAGE1, "Chronic kidney disease stage 1", "SNOMED", "S", "Clinical Finding"),
        (7092, STAGE2, "Chronic kidney disease stage 2", "SNOMED", "S", "Clinical Finding"),
        (7093, "433146000", "Chronic kidney disease stage 3", "SNOMED", "S", "Clinical Finding"),
        (7094, CKD, "a non-standard duplicate code", "SNOMED", None, "Clinical Finding"),
        (3001, "PIIState_MN", "MN", "PPI", None, "Answer"),
        (2000000011, "GeneralizedForPrivacy", "suppressed", "PPI", None, "Answer"),
    ]
    create(con, "concept", concept_id=[c[0] for c in concepts], concept_code=[c[1] for c in concepts],
           concept_name=[c[2] for c in concepts], vocabulary_id=[c[3] for c in concepts],
           standard_concept=[c[4] for c in concepts], concept_class_id=[c[5] for c in concepts])
    ancestry = [(1000, 1000), (1000, 1001), (2010, 2010), (2010, 2011), (709, 709), (709, 7091), (709, 7092),
                (709, 7093), (7091, 7091), (7092, 7092), (7093, 7093)]
    create(con, "concept_ancestor", ancestor_concept_id=[a for a, _ in ancestry],
           descendant_concept_id=[d for _, d in ancestry])
    people = [1, 2, 3, 4, *EXTRA]
    create(con, "person", person_id=people,
           birth_datetime=pa.array([datetime.datetime(1960, 1, 1, 13, 30)] + [datetime.datetime(1975, 5, 5)] * 3
                                   + [datetime.datetime(1970, 1, 1)] * len(EXTRA), pa.timestamp("us")),
           sex_at_birth_concept_id=pa.array([45880669, None, 45878463, 8532] + [45878463] * len(EXTRA),
                                            pa.int64()),
           race_concept_id=pa.array([8527, None, 0, 0] + [0] * len(EXTRA), pa.int64()),
           ethnicity_concept_id=pa.array([38003564, None, 0, 0] + [0] * len(EXTRA), pa.int64()))
    observations = [  # (person, concept, date, source concept, value)
        (1, 1001, D(2019, 3, 1), 0, None), (1, 1001, D(2019, 1, 1), 0, None),
        (1, 0, D(2018, 6, 1), 1585250, "553**"), (1, 0, D(2020, 1, 1), 1585250, "554**"),
        (2, 1001, D(2020, 5, 5), 0, None), (2, 0, D(2021, 1, 1), 1585250, "100**"),
        (2, 0, D(2022, 1, 1), 1585250, "101**"), (2, 0, D(2020, 1, 1), 1585250, "Response removed"),
        (3, 0, D(2020, 1, 1), 1585250, "200**"), (2, 0, D(2021, 8, 8), 0, None),
        *((p, 1001, D(2019, 2, 2), 0, None) for p in EXTRA),
    ]
    create(con, "observation_ext", observation_id=list(range(len(observations))),
           src_id=["PPI/PM"] * (len(observations) - len(EXTRA) - 1) + ["EHR site 3"] + ["PPI/PM"] * len(EXTRA))
    create(con, "observation", observation_id=list(range(len(observations))),
           person_id=[o[0] for o in observations], observation_concept_id=[o[1] for o in observations],
           observation_date=dates([o[2] for o in observations]),
           observation_source_concept_id=[o[3] for o in observations],
           value_as_string=pa.array([o[4] for o in observations], pa.string()))
    periods = [(1, D(2015, 1, 1), D(2023, 9, 1)), (1, D(2010, 1, 1), D(2023, 6, 1)),
               (2, D(2020, 6, 1), D(2022, 1, 1)), *((p, D(2012, 1, 1), D(2023, 5, 5)) for p in EXTRA)]
    create(con, "observation_period", observation_period_id=list(range(len(periods))),
           person_id=[p[0] for p in periods], observation_period_start_date=dates([p[1] for p in periods]),
           observation_period_end_date=dates([p[2] for p in periods]))
    create(con, "aou_death", person_id=[1, 1], death_date=dates([D(2023, 7, 1), D(2023, 5, 1)]),
           primary_death_record=[True, False])
    create(con, "person_ext", person_id=[1, 2], state_of_residence_concept_id=[3001, 2000000011])
    visits = [(1, D(2018, 1, 1), None, "EHR site 7"), (1, D(2018, 2, 1), None, "EHR site 7"),
              (1, D(2018, 3, 1), None, "EHR site 7"), (1, D(2018, 4, 1), None, "EHR site 9"),
              (1, D(2018, 5, 1), None, "PPI/PM"), *((1, D(2020, m, 1), None, "EHR site 9") for m in range(1, 6)),
              (2, D(2019, 11, 11), D(2019, 11, 1), "EHR site 3"),  # an end before the start is ignored
              (1003, D(2021, 6, 1), D(2021, 6, 20), "EHR site 5"),  # an inpatient stay lasts to discharge
              (1004, D(2021, 6, 1), D(2099, 12, 31), "EHR site 5")]  # a placeholder end counts one year
    create(con, "visit_occurrence", visit_occurrence_id=list(range(len(visits))),
           person_id=[v[0] for v in visits], visit_start_date=dates([v[1] for v in visits]),
           visit_end_date=dates([v[2] for v in visits]))
    create(con, "visit_occurrence_ext", visit_occurrence_id=list(range(len(visits))),
           src_id=[v[3] for v in visits])
    conditions = [  # (person, concept, date, src_id)
        (1, 2011, D(2020, 1, 1), "EHR site 9"), (1, 2011, D(2020, 1, 1), "EHR site 9"),
        (1, 2011, D(2020, 3, 1), "EHR site 9"), (1, 2011, D(2021, 1, 1), "EHR site 9"),
        (1, 7091, D(2019, 5, 5), "EHR site 9"), (1, 7093, D(2021, 2, 2), "EHR site 9"),
        (2, 9999, D(2019, 12, 12), "EHR site 3"), (2, 7092, D(2020, 9, 9), "EHR site 3"),
        (2, 2011, D(1850, 1, 1), "EHR site 8"), (2, 2011, D(2021, 5, 5), "EHR site 4"),
        # Audit: an unspecified CKD code and a stage-2 code on the same day; that day still qualifies once.
        (4, 709, D(2021, 3, 3), "EHR site 6"), (4, 7092, D(2021, 3, 3), "EHR site 6"),
        (4, 7092, D(2021, 4, 4), "EHR site 6"),
        *((p, 2011, D(2020, 1, 1), "EHR site 5") for p in EXTRA),
        *((p, 2011, D(2020, 6, 6), "EHR site 5") for p in EXTRA),
        *((p, 7091, D(2020, 2, 2), "EHR site 5") for p in EXTRA),
        *((p, 7093, D(2021, 1, 1), "EHR site 5") for p in EXTRA),
    ]
    create(con, "condition_occurrence", condition_occurrence_id=list(range(len(conditions))),
           person_id=[c[0] for c in conditions], condition_concept_id=[c[1] for c in conditions],
           condition_start_date=dates([c[2] for c in conditions]))
    create(con, "condition_occurrence_ext", condition_occurrence_id=list(range(len(conditions))),
           src_id=[c[3] for c in conditions])
    # The other EHR domains: only rows whose _ext src_id names an EHR site count.
    domains = {"procedure_occurrence": ("procedure_date", [(1001, D(2022, 3, 3), "EHR site 5")]),
               "drug_exposure": ("drug_exposure_start_date", [(1, D(2017, 1, 1), "EHR site 7"),
                                                              (2, D(2023, 1, 1), "PPI/PM")]),
               "measurement": ("measurement_date", [(1, D(2022, 2, 2), "EHR site 9")])}
    for table, (day, rows_) in domains.items():
        create(con, table, **{f"{table}_id": list(range(len(rows_))), "person_id": [r[0] for r in rows_],
                              day: dates([r[1] for r in rows_])})
        create(con, f"{table}_ext", **{f"{table}_id": list(range(len(rows_))), "src_id": [r[2] for r in rows_]})
    create(con, "zip3_ses_map", zip3=[553, 553, 100], zip3_as_string=["553", "553", "100"],
           deprivation_index=[0.29, 0.31, 0.40], acs=["2016", "2017", "2017"])
    return con


@pytest.fixture(scope="module")
def exported(cdr, tmp_path_factory):
    directory = tmp_path_factory.mktemp("sql")
    people = [1, 2, 3, *EXTRA]
    pd.DataFrame({"research_id": people, "ancestry_pred": ["eur"] * len(people)}).to_csv(
        directory / "ancestry.tsv", sep="\t", index=False)
    pd.DataFrame({"sample_id": [3]}).to_csv(directory / "prune.tsv", sep="\t", index=False)
    pd.DataFrame({"IID": [str(p) for p in people], **{f"PC{i}": [0.1 * i] * len(people) for i in range(1, 7)}}
                 ).to_parquet(directory / "pcs.parquet", index=False)
    (directory / "cache").mkdir()
    rows = "".join(f"{p}\t0.5\t0\n" for p in people)
    (directory / "cache" / "a.sscore").write_text(f"#IID\tPGS000001_AVG\tPGS000001_MISSING_PCT\n{rows}")
    client = DuckClient(cdr)
    source = cohort.AouSource(client, CDR, snomed_codes={T2D: 2010, CKD: 709}, scores=["PGS000001"],
                              excluded_branches={CKD: {STAGE1: 7091, STAGE2: 7092}},
                              ancestry=directory / "ancestry.tsv", prune=directory / "prune.tsv",
                              projection=directory / "pcs.parquet", score_cache=directory / "cache")
    return source.export(directory / "tables"), client


def rows(table, key):
    return {row[key]: row for row in table.to_pylist()}


def test_person_rules(exported):
    source, _ = exported
    person = rows(source.table("person"), "person_id")
    assert len(person) == 4 + len(EXTRA)  # every CDR person, eligible or not
    one, two, three = person[1], person[2], person[3]
    assert one["birth_date"] == D(1960, 1, 1)
    assert one["baseline_date"] == D(2019, 1, 1)  # the earliest consent-module record
    assert (one["obs_start"], one["obs_end"]) == (D(2010, 1, 1), D(2023, 6, 1))  # earliest covering start
    assert one["death_date"] == D(2023, 7, 1)  # primary records only
    assert (one["ehr_start"], one["ehr_end"]) == (D(2017, 1, 1), D(2022, 2, 2))  # drug and measurement rows
    assert one["state"] == "MN" and one["ehr_site"] == "EHR site 7"  # pre-baseline visits only
    assert (one["zip3"], one["zip3_post_baseline"], one["deprivation_index"]) == (553, False, 0.31)  # latest acs
    assert (two["sex_at_birth_concept_id"], two["race_concept_id"]) == (0, 0)
    assert two["baseline_date"] == D(2020, 5, 5) and two["obs_start"] is None  # no period covers baseline
    assert two["state"] is None and two["death_date"] is None
    assert two["ehr_site"] == "EHR site 3"  # the pre-baseline condition row; the later site does not count
    # The EHR-sourced observation extends the range; the PPI/PM drug row and the 1850 record do not.
    assert (two["ehr_start"], two["ehr_end"]) == (D(2019, 11, 11), D(2021, 8, 8))
    assert (two["zip3"], two["zip3_post_baseline"], two["deprivation_index"]) == (100, True, 0.40)
    assert three["baseline_date"] is None and three["zip3"] is None and three["ehr_site"] is None
    assert three["ehr_end"] is None
    assert person[1001]["ehr_end"] == D(2022, 3, 3) and person[1002]["ehr_end"] == D(2021, 1, 1)
    assert person[1003]["ehr_end"] == D(2021, 6, 20) and person[1004]["ehr_end"] == D(2022, 6, 1)


def test_condition_dates_net_of_excluded_branches(exported):
    source, _ = exported
    condition = {(r["snomed_code"], r["person_id"]): r for r in source.table("condition").to_pylist()}
    assert {k: (r["first_date"], r["second_date"], r["n_dates"]) for k, r in condition.items()
            if k[1] in (1, 2, 4)} == {
        (T2D, 1): (D(2020, 1, 1), D(2020, 3, 1), 3),  # a repeated day counts once
        (T2D, 2): (D(2021, 5, 5), None, 1),  # the 1850 record is out of range
        (CKD, 1): (D(2021, 2, 2), None, 1),  # the stage-1 record does not qualify
        (CKD, 4): (D(2021, 3, 3), None, 1),  # the unspecified code qualifies its day; stage 2 adds none
    }  # person 2's only CKD record is stage 2: no row
    for p in EXTRA:
        assert (condition[(CKD, p)]["first_date"], condition[(CKD, p)]["n_dates"]) == (D(2021, 1, 1), 1)
        assert (condition[(T2D, p)]["second_date"], condition[(T2D, p)]["n_dates"]) == (D(2020, 6, 6), 2)


def test_roots_descendants_and_manifest(exported):
    source, client = exported
    root = rows(source.table("root"), "snomed_code")
    assert {code: (r["concept_id"], r["n_descendants"]) for code, r in root.items()} == {
        T2D: (2010, 2), CKD: (709, 4)}  # the non-standard duplicate of the CKD code is ignored
    descendants = pd.read_parquet(source.directory / "descendants.parquet")
    got = {(r.snomed_code, r.role, r.concept_id): r.n_persons for r in descendants.itertuples()}
    assert got == {(T2D, "root", 2011): 27, (CKD, "root", 7093): 26, (STAGE1, "excluded_branch", 7091): 26}
    assert set(descendants.root_code) == {T2D, CKD}  # stage 2 (1 person) is under the >20 rule
    manifest = source.manifest
    assert manifest["cdr_cutoff"] == "2023-09-01"  # the latest observation-period end
    assert manifest["cdr_cutoff_source"] == "max(observation_period_end_date)"
    assert manifest["excluded_branches"] == {CKD: [STAGE1, STAGE2]} and manifest["prune_unmatched"] == 0
    assert manifest["ses_available"] is True
    # One query per table for all roots, the cutoff, and one grouped pass per extra EHR domain.
    assert len(client.statements) == 9
    # Each extra domain moved these people's ehr_end later, of the 28 with EHR: procedure 1001,
    # observation 2 and measurement 1; the drug rows are earlier or not EHR-sourced.
    assert manifest["ehr_extended_by"] == {"procedure": 1 / 28, "drug": 0.0, "observation": 1 / 28,
                                           "measurement": 1 / 28}
    assert manifest["ehr_end_from_long_visit"] == 1 / 28  # person 1004's capped placeholder stay
    assert manifest["ehr_people"] == 28  # the denominator of both shares
