"""The table contract (SCHEMA.md): round trip, planted violations, release-file readers, AoU source."""
from __future__ import annotations

import datetime
import io
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import cohort  # noqa: E402
from study_fixtures import date_array, manifest, synthetic_tables  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return synthetic_tables(4000, seed=3, grid_days=30)


def replace(table, name, array):
    return table.set_column(table.schema.get_field_index(name), name, array)


def test_round_trip_preserves_every_table(tmp_path, tables):
    tabs, spec = tables
    written = cohort.write_tables(tmp_path, tabs, spec)
    source = cohort.ParquetSource(tmp_path)
    assert cohort.validate_tables(tmp_path)["tables"] == written["tables"]
    for name in cohort.TABLES:
        assert source.table(name).equals(cohort.conform(name, tabs[name], spec)), name
        assert written["tables"][name]["rows"] == tabs[name].num_rows
    person = source.frame("person")
    assert str(person.birth_date.dtype).startswith("datetime64")
    assert set(source.condition(["44054006"]).column("snomed_code").to_pylist()) == {"44054006"}
    assert source.scores("PGS000022").column_names == ["person_id", "PGS000022", "PGS000022_missing_pct"]


def test_pandas_input_conforms_and_a_time_of_day_is_refused(tmp_path, tables):
    tabs, spec = tables
    frames = {name: table.to_pandas(date_as_object=False) for name, table in tabs.items()}
    frames["person"]["zip3"] = frames["person"].zip3.astype("Int64")
    cohort.write_tables(tmp_path / "pandas", frames, spec)
    assert cohort.ParquetSource(tmp_path / "pandas").table("person").equals(
        cohort.conform("person", tabs["person"], spec))
    shifted = frames["person"].copy()
    shifted["birth_date"] = shifted.birth_date + pd.Timedelta(hours=6)
    with pytest.raises(cohort.SchemaError, match="time of day"):
        cohort.conform("person", pa.Table.from_pandas(shifted, preserve_index=False), spec)


def _first_with(table, column, predicate):
    values = table.column(column).to_pandas()
    return int(np.flatnonzero(predicate(values))[0])


def _set(table, column, index, value, kind=None):
    values = table.column(column).to_pylist()
    values[index] = value
    return replace(table, column, pa.array(values, kind or table.schema.field(column).type))


def mutations(tabs):
    """(label, table name, mutated table, expected message): each must make validation fire."""
    person, condition, scores = tabs["person"], tabs["condition"], tabs["scores"]
    multi = _first_with(condition, "n_dates", lambda v: v >= 2)
    single = _first_with(condition, "n_dates", lambda v: v == 1)
    covered = _first_with(person, "obs_start", lambda v: v.notna())
    no_baseline = _first_with(person, "baseline_date", lambda v: v.isna())
    zipless = _first_with(person, "zip3", lambda v: v.isna())
    pct = _first_with(scores, "PGS000011_missing_pct", lambda v: v.notna() & (v < 100))
    first = condition.column("first_date")[multi].as_py()
    return [
        ("duplicate person", "person", pa.concat_tables([person, person.slice(0, 1)]), "repeats its key"),
        ("duplicate condition key", "condition", pa.concat_tables([condition, condition.slice(multi, 1)]),
         "repeats its key"),
        ("second date on a single record", "condition",
         _set(condition, "second_date", single, condition.column("first_date")[single].as_py()),
         "second_date must be present exactly"),
        ("second date missing on a multi record", "condition", _set(condition, "second_date", multi, None),
         "second_date must be present exactly"),
        ("second date not after first", "condition", _set(condition, "second_date", multi, first),
         "must follow first_date"),
        ("zero dates", "condition", _set(condition, "n_dates", single, 0), "at least 1"),
        ("record before 1900", "condition",
         _set(condition, "first_date", single, pd.Timestamp("1850-01-01").date()), "outside 1900"),
        ("unknown root", "condition", _set(condition, "snomed_code", single, "123456789"),
         "snomed_code the manifest lacks"),
        ("root table short", "root", tabs["root"].slice(1), "every manifest snomed_code"),
        ("zip3 without baseline", "person", _set(person, "zip3", no_baseline, 553), "zip3 but no baseline"),
        ("zip3 flag without zip3", "person", _set(person, "zip3_post_baseline", zipless, False),
         "zip3_post_baseline must be present"),
        ("period misses baseline", "person",
         _set(person, "obs_start", covered, person.column("baseline_date")[covered].as_py()
              + datetime.timedelta(days=1)), "does not cover"),
        ("site without baseline", "person", _set(person, "ehr_site", no_baseline, "EHR site 1"),
         "ehr_site but no baseline"),
        ("deprivation without zip3", "person", _set(person, "deprivation_index", zipless, 0.3),
         "deprivation_index but no zip3"),
        ("lowercase state", "person", _set(person, "state", 0, "ca"), "two-letter"),
        ("infinite PC", "pcs", _set(tabs["pcs"], "PC2", 0, float("inf")), "non-finite"),
        ("null PC", "pcs", _set(tabs["pcs"], "PC1", 0, None), "has nulls"),
        ("score at 100% missing", "scores", _set(scores, "PGS000011_missing_pct", pct, 100.0),
         "null exactly when unscored"),
        ("missingness above 100", "scores", _set(scores, "PGS000011_missing_pct", pct, 101.0),
         "outside \\[0, 100\\]"),
        ("string person_id", "ancestry",
         replace(tabs["ancestry"], "person_id", pc.cast(tabs["ancestry"].column("person_id"), pa.string())),
         "contract says int64"),
        ("extra column", "root", tabs["root"].append_column("note", pa.array(["x"] * tabs["root"].num_rows)),
         "unexpected columns"),
        ("missing column", "person", person.drop_columns(["ehr_site"]), "missing columns"),
    ]


def test_every_planted_violation_fires(tmp_path, tables):
    tabs, spec = tables
    cohort.write_tables(tmp_path / "clean", tabs, spec)  # the unmutated tables pass
    for label, name, mutated, message in mutations(tabs):
        with pytest.raises(cohort.SchemaError, match=message):
            cohort.write_tables(tmp_path / "bad", {**tabs, name: mutated}, spec)
        assert not (tmp_path / "bad" / "manifest.json").exists(), label


def test_manifest_rules_fire(tmp_path, tables):
    tabs, spec = tables
    for key, value, message in [("cdr_cutoff", "2023-13-01", "ISO date"), ("num_pcs", 5, "num_pcs"),
                                ("snomed_codes", ["38341003", "38341003"], "distinct SNOMED"),
                                ("scores", ["PGS11"], "PGS Catalog"), ("source", "aou", "source")]:
        with pytest.raises(cohort.SchemaError, match=message):
            cohort.write_tables(tmp_path / key, tabs, {**spec, key: value})
    person = tabs["person"]
    with pytest.raises(cohort.SchemaError, match="SES is unavailable"):
        cohort.write_tables(tmp_path / "ses", tabs, {**spec, "ses_available": False})
    cleared = replace(person, "deprivation_index", pa.nulls(person.num_rows, pa.float64()))
    cohort.write_tables(tmp_path / "ses2", {**tabs, "person": cleared}, {**spec, "ses_available": False})


def test_a_changed_file_fails_its_manifest_hash(tmp_path, tables):
    tabs, spec = tables
    cohort.write_tables(tmp_path, tabs, spec)
    pq.write_table(tabs["root"].slice(0, tabs["root"].num_rows), tmp_path / "root.parquet", compression="none")
    with pytest.raises(cohort.SchemaError, match="manifest hash"):
        cohort.ParquetSource(tmp_path)
    (tmp_path / "manifest.json").unlink()
    with pytest.raises(cohort.SchemaError, match="no manifest"):
        cohort.ParquetSource(tmp_path)


# --------------------------------------------------------------------------- #
# release files
# --------------------------------------------------------------------------- #
def test_ancestry_and_prune(tmp_path):
    (tmp_path / "anc.tsv").write_text("research_id\tancestry_pred\tpca_features\n"
                                      "1001\teur\t[0]\n1002\tafr\t[1]\n1003\tamr\t[2]\n")
    (tmp_path / "prune.tsv").write_text("sample_id\n1002\n9999\n")
    table, unmatched = cohort.read_ancestry(tmp_path / "anc.tsv", tmp_path / "prune.tsv")
    assert table.column("person_id").to_pylist() == [1001, 1002, 1003]
    assert table.column("related_excluded").to_pylist() == [False, True, False]
    assert table.column("ancestry_pred").to_pylist() == ["eur", "afr", "amr"]
    assert unmatched == 1
    (tmp_path / "bad.tsv").write_text("person_id\n1002\n")
    with pytest.raises(cohort.SchemaError, match="sample_id"):
        cohort.read_ancestry(tmp_path / "anc.tsv", tmp_path / "bad.tsv")
    (tmp_path / "anc2.tsv").write_text("research_id\tancestry_pred\nA1\teur\n")
    with pytest.raises(cohort.SchemaError, match="non-numeric"):
        cohort.read_ancestry(tmp_path / "anc2.tsv", tmp_path / "prune.tsv")


def test_projection_pcs(tmp_path):
    frame = pd.DataFrame({"IID": ["11", "12"], **{f"PC{i}": [0.1 * i, -0.1 * i] for i in range(1, 9)}})
    frame.to_parquet(tmp_path / "p.parquet", index=False)
    table = cohort.read_pcs(tmp_path / "p.parquet", 6)
    assert table.column_names == ["person_id", *(f"PC{i}" for i in range(1, 7))]
    assert table.column("person_id").to_pylist() == [11, 12]
    assert cohort.read_pcs(tmp_path / "p.parquet").num_columns == 9
    with pytest.raises(cohort.SchemaError, match="PC1..PC9"):
        cohort.read_pcs(tmp_path / "p.parquet", 9)


SSCORE_A = ("#SCORE_VARIANT_COUNT\tSCORE\tCOUNT\n#SCORE_VARIANT_COUNT\tPGS000011\t10\n"
            "#REGION\tSCORE\tINTERVAL\n#REGION\tPGS000011\tchr1:1-9\n"
            "#IID\tPGS000011_AVG\tPGS000011_MISSING_PCT\tPGS000099_AVG\tPGS000099_MISSING_PCT\n"
            "5\t0.5\t0\t9\t0\n6\t0.25\t100\t9\t0\n7\t-1.5\t2.5\t9\t0\n")
SSCORE_B = "#IID\tPGS000022_AVG\tPGS000022_MISSING_PCT\n5\t1.0\t0\n8\t2.0\t50\n"


def test_score_cache_directory_and_tar(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "a.sscore").write_text(SSCORE_A)
    (directory / "b.sscore").write_text(SSCORE_B)
    archive = tmp_path / "scores.tar"
    with tarfile.open(archive, "w") as tar:
        for name, text in (("x/a.sscore", SSCORE_A), ("x/b.sscore", SSCORE_B)):
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    for cache in (directory, archive):
        frame = cohort.read_scores(cache, ["PGS000011", "PGS000022"]).to_pandas().set_index("person_id")
        assert list(frame.columns) == ["PGS000011", "PGS000011_missing_pct", "PGS000022", "PGS000022_missing_pct"]
        assert sorted(frame.index) == [5, 6, 7, 8]
        assert frame.loc[5, "PGS000011"] == 0.5 and frame.loc[7, "PGS000011_missing_pct"] == 2.5
        assert np.isnan(frame.loc[6, "PGS000011"]) and frame.loc[6, "PGS000011_missing_pct"] == 100
        assert np.isnan(frame.loc[8, "PGS000011"]) and np.isnan(frame.loc[8, "PGS000011_missing_pct"])
        assert np.isnan(frame.loc[6, "PGS000022"]) and frame.loc[8, "PGS000022"] == 2.0
    with pytest.raises(cohort.SchemaError, match="absent"):
        cohort.read_scores(directory, ["PGS000033"])
    (directory / "c.sscore").write_text(SSCORE_B)
    with pytest.raises(cohort.SchemaError, match="appears in both"):
        cohort.read_scores(directory, ["PGS000022"])
    (directory / "c.sscore").write_text("#IID\tPGS000044_AVG\n5\t1\n")
    # The WGS score bank writes no per-participant missingness: the column is then null.
    bare = cohort.read_scores(directory, ["PGS000044"]).to_pandas().set_index("person_id")
    assert bare["PGS000044_missing_pct"].isna().all() and bare["PGS000044"].notna().any()


def test_scores_are_read_by_name_whatever_the_file_order(tmp_path):
    """One file holding two scores in the opposite order to the request, its missingness
    columns apart from their scores: every value lands under its own name."""
    (tmp_path / "ab.sscore").write_text(
        "#IID\tPGS000022_MISSING_PCT\tPGS000022_AVG\tPGS000011_AVG\tPGS000099_AVG\tPGS000011_MISSING_PCT\n"
        "101\t3\t8\t2\t7\t1\n")
    for wanted in (["PGS000011", "PGS000022"], ["PGS000022", "PGS000011"]):
        row = cohort.read_scores(tmp_path, wanted).to_pylist()[0]
        assert row == {"person_id": 101, "PGS000011": 2.0, "PGS000011_missing_pct": 1.0,
                       "PGS000022": 8.0, "PGS000022_missing_pct": 3.0}
        assert list(row)[1::2] == wanted


# --------------------------------------------------------------------------- #
# the AoU source against a recording fake client
# --------------------------------------------------------------------------- #
class FakeClient:
    """Answers the source's queries from fixture tables and records what it was asked."""

    def __init__(self, tabs, spec, ses=True, branches=None, estimate=1000):
        self.tabs, self.spec = tabs, spec
        self.ses = ses
        self.pairs = [f"{root}:{b}" for root, bs in (branches or {}).items() for b in sorted(bs)]
        self.queries, self.estimates = [], []
        self.estimate_bytes = estimate
        self.billed = 123
        self.remaining = 50 * 10 ** 9
        self.job_ids = ["job-1"]
        self.descendants = pa.table({"snomed_code": ["44054006"], "role": ["root"], "root_code": ["44054006"],
                                     "concept_id": pa.array([201826], pa.int64()),
                                     "concept_name": ["Type 2 diabetes mellitus"],
                                     "n_persons": pa.array([4321], pa.int64())})

    def columns(self, table_id):
        if table_id.endswith(".zip3_ses_map"):
            return {"zip3", "zip3_as_string", "deprivation_index", "acs"} if self.ses else None
        raise AssertionError(table_id)

    def estimate(self, sql, parameters=None):
        self.estimates.append((sql, parameters))
        return self.estimate_bytes(sql) if callable(self.estimate_bytes) else self.estimate_bytes

    def query(self, sql, parameters=None):
        assert len(self.estimates) == 9, "a query ran before the whole plan was estimated"
        self.queries.append((sql, parameters))
        if "GROUP BY t.person_id" in sql:  # an extra EHR domain: nobody widens the fixture's range
            return pa.table({"person_id": pa.array([], pa.int64()), "ehr_start": pa.array([], pa.date32()),
                             "ehr_end": pa.array([], pa.date32())})
        if "MAX(observation_period_end_date)" in sql:
            return pa.table({"cutoff": pa.array([pd.Timestamp(self.spec["cdr_cutoff"]).date()], pa.date32())})
        if "Consent PII" in sql:
            person = self.tabs["person"]
            if not self.ses:
                person = replace(person, "deprivation_index", pa.nulls(person.num_rows, pa.float64()))
            # BigQuery returns INT64 for every integer column, plus the long-visit diagnostic column.
            person = person.append_column("ehr_long_end", pa.nulls(person.num_rows, pa.date32()))
            return replace(person, "zip3", person.column("zip3").cast(pa.int64()))
        members = {"codes": ("STRING", self.spec["snomed_codes"]), "branch_pairs": ("STRING", self.pairs)}
        if "n_persons > 20" in sql:
            assert parameters == members
            return self.descendants
        if "n_descendants" in sql:
            branch_codes = sorted({pair.split(":")[1] for pair in self.pairs})
            assert parameters == {"codes": ("STRING", self.spec["snomed_codes"] + branch_codes)}
            extra = pa.table({"snomed_code": pa.array(branch_codes, pa.string()),
                              "concept_id": pa.array(range(9_000_000, 9_000_000 + len(branch_codes)), pa.int64()),
                              "concept_name": pa.array(branch_codes, pa.string()),
                              "n_descendants": pa.array([3] * len(branch_codes), pa.int64())})
            return pa.concat_tables([self.tabs["root"], extra])
        assert parameters == members
        condition = self.tabs["condition"]
        return replace(condition, "n_dates", condition.column("n_dates").cast(pa.int64()))


def by_key(name, table):
    return table.sort_by([(key, "ascending") for key in cohort.KEYS[name]])


def aou_source(tmp_path, tabs, spec, client, codes=None, branches=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame = tabs["ancestry"].to_pandas()
    frame.rename(columns={"person_id": "research_id"})[["research_id", "ancestry_pred"]].to_csv(
        tmp_path / "anc.tsv", sep="\t", index=False)
    pruned = frame.loc[frame.related_excluded, "person_id"]
    pd.DataFrame({"sample_id": pruned}).to_csv(tmp_path / "prune.tsv", sep="\t", index=False)
    pcs = tabs["pcs"].to_pandas().rename(columns={"person_id": "IID"})
    pcs["IID"] = pcs.IID.astype(str)
    pcs.to_parquet(tmp_path / "pcs.parquet", index=False)
    cache = tmp_path / "cache"
    cache.mkdir()
    scores = tabs["scores"].to_pandas()
    for pgs in spec["scores"]:
        body = scores[["person_id", pgs, f"{pgs}_missing_pct"]].copy()
        body = body.loc[body[f"{pgs}_missing_pct"].notna()]
        body.columns = ["#IID", f"{pgs}_AVG", f"{pgs}_MISSING_PCT"]
        body[f"{pgs}_AVG"] = body[f"{pgs}_AVG"].fillna(0.0)  # a 100%-missing person scores 0 in the file
        body.to_csv(cache / f"{pgs}.sscore", sep="\t", index=False, float_format="%.17g")
    return cohort.AouSource(client, "fc-aou-cdr-prod-ct.C2024Q3R5", snomed_codes=codes or spec["snomed_codes"],
                            scores=spec["scores"], ancestry=tmp_path / "anc.tsv", prune=tmp_path / "prune.tsv",
                            projection=tmp_path / "pcs.parquet", score_cache=cache, excluded_branches=branches)


def test_aou_source_exports_the_contract(tmp_path, tables):
    tabs, spec = tables
    client = FakeClient(tabs, spec)
    source = aou_source(tmp_path, tabs, spec, client)
    exported = source.export(tmp_path / "out")
    manifest = exported.manifest
    assert manifest["source"] == "bigquery" and manifest["ses_available"] is True
    assert manifest["cdr_cutoff"] == spec["cdr_cutoff"]
    assert manifest["cdr_cutoff_source"] == "max(observation_period_end_date)"
    assert manifest["bigquery"] == {"bytes_billed": 123, "job_ids": ["job-1"], "plan_bytes": dict.fromkeys(
        ["person", "condition", "root", "descendants", "cutoff", "ehr_procedure", "ehr_drug", "ehr_observation",
         "ehr_measurement"], 1000)}
    assert manifest["ehr_domains"] == ["visit", "condition", "procedure", "drug", "observation", "measurement"]
    assert manifest["prune_unmatched"] == 0
    assert manifest["descendants"]["rows"] == 1
    assert pq.read_table(tmp_path / "out" / "descendants.parquet").equals(client.descendants)
    for name in cohort.TABLES:  # row order is not part of the contract
        expected = cohort.conform(name, tabs[name], spec)
        assert by_key(name, exported.table(name)).equals(by_key(name, expected)), name
    # Roots travel as a query parameter, never as SQL text; one query covers every root.
    condition_queries = [q for q, p in client.queries if "qualifying" in q]
    assert len(condition_queries) == 1
    assert not any(root in sql for sql, _ in client.queries for root in spec["snomed_codes"])
    person_sql = next(q for q, _ in client.queries if "Consent PII" in q)
    assert "zip3_ses_map" in person_sql and "acs DESC" in person_sql
    assert "COUNTIF(r.day < e.baseline_date) AS pre_baseline" in person_sql  # sites from pre-baseline rows
    assert "IF(pre_baseline > 0, src_id, NULL)" in person_sql


def test_aou_source_without_ses(tmp_path, tables):
    tabs, spec = tables
    client = FakeClient(tabs, spec, ses=False)
    manifest = aou_source(tmp_path, tabs, spec, client).manifest
    assert manifest["ses_available"] is False
    person_sql = next(q for q, _ in client.queries if "Consent PII" in q)
    assert "zip3_ses_map" not in person_sql and "CAST(NULL AS FLOAT64)" in person_sql


def test_aou_source_refuses_an_unresolved_root(tmp_path, tables):
    tabs, spec = tables
    client = FakeClient({**tabs, "root": tabs["root"].slice(1)}, spec)
    with pytest.raises(cohort.SchemaError, match="unresolved"):
        aou_source(tmp_path, tabs, spec, client).manifest


def test_a_plan_over_budget_is_refused_before_anything_bills(tmp_path, tables):
    tabs, spec = tables
    client = FakeClient(tabs, spec, estimate=11 * 10 ** 9)  # five queries at 11 GB each exceed 50 GB
    with pytest.raises(RuntimeError, match="plan would bill"):
        aou_source(tmp_path, tabs, spec, client).manifest
    assert len(client.estimates) == 9 and client.queries == []


def test_the_ehr_definition_must_fit_the_budget_whole(tmp_path, tables):
    tabs, spec = tables
    costs = lambda sql: 45 * 10 ** 9 if "measurement" in sql else 10 ** 9  # noqa: E731  (8 + 45 > 50 GB)
    client = FakeClient(tabs, spec, estimate=costs)
    with pytest.raises(RuntimeError, match="plan would bill"):  # no domain is dropped to fit
        aou_source(tmp_path, tabs, spec, client).manifest
    assert client.queries == []


def test_excluded_branches_reach_the_queries_and_the_manifest(tmp_path, tables):
    tabs, spec = tables
    branches = {"44054006": {"123456789": 9_000_000}}
    client = FakeClient(tabs, spec, branches=branches)
    exported = aou_source(tmp_path, tabs, spec, client, branches=branches).export(tmp_path / "out")
    assert exported.manifest["excluded_branches"] == {"44054006": ["123456789"]}
    assert exported.table("root").column("snomed_code").to_pylist() == spec["snomed_codes"]
    condition_sql = next(sql for sql, _ in client.queries if "qualifying" in sql)
    assert "UNNEST(@branch_pairs)" in condition_sql and "LOGICAL_AND(NOT k.is_branch)" in condition_sql
    with pytest.raises(cohort.SchemaError, match="other concepts than declared"):
        wrong = {"44054006": {"123456789": 1}}
        aou_source(tmp_path / "w", tabs, spec, FakeClient(tabs, spec, branches=wrong), branches=wrong).manifest
    with pytest.raises(cohort.SchemaError, match="excluded_branches"):
        cohort.write_tables(tmp_path / "bad", tabs, {**spec, "excluded_branches": {"44054006": []}})


def test_aou_source_checks_declared_concepts(tmp_path, tables):
    tabs, spec = tables
    declared = dict(zip(spec["snomed_codes"], range(3_000_000, 3_000_000 + len(spec["snomed_codes"]))))
    assert aou_source(tmp_path / "a", tabs, spec, FakeClient(tabs, spec), codes=declared).manifest
    wrong = {**declared, spec["snomed_codes"][2]: 201826}
    with pytest.raises(cohort.SchemaError, match="other concepts than declared"):
        aou_source(tmp_path / "b", tabs, spec, FakeClient(tabs, spec), codes=wrong).manifest


def test_cdr_name_is_checked_before_it_reaches_sql():
    with pytest.raises(ValueError, match="project.dataset"):
        cohort.person_sql("x`; DROP TABLE y; --.z", None)


# --------------------------------------------------------------------------- #
# BoundedClient against a stand-in for google.cloud.bigquery
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_bigquery(monkeypatch):
    module = types.ModuleType("google.cloud.bigquery")

    class Parameter:
        def __init__(self, *args):
            self.args = args

    class QueryJobConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    module.ScalarQueryParameter = module.ArrayQueryParameter = Parameter
    module.QueryJobConfig = QueryJobConfig
    google, cloud = types.ModuleType("google"), types.ModuleType("google.cloud")
    google.cloud, cloud.bigquery = cloud, module
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.bigquery", module)
    return module


class Job:
    def __init__(self, billed, fail=False, cache_hit=False):
        self.job_id, self.total_bytes_billed, self.cache_hit = f"j{billed}", billed, cache_hit
        self.fail, self.cancelled = fail, False

    def result(self, timeout):
        if self.fail:
            raise TimeoutError("slow")
        return types.SimpleNamespace(to_arrow=lambda create_bqstorage_client: pa.table({"x": [1]}))

    def cancel(self):
        self.cancelled = True


def test_bounded_client_spends_one_budget(fake_bigquery):
    jobs, configs = [Job(600), Job(300), Job(0, cache_hit=True), Job(500)], []

    class Raw:
        def query(self, sql, job_config):
            configs.append(job_config)
            return jobs[len(configs) - 1]

    client = cohort.BoundedClient(Raw(), 1000)
    client.query("select 1", {"codes": ("STRING", ["1"]), "n": ("INT64", 3)})
    assert configs[0].maximum_bytes_billed == 1000 and len(configs[0].query_parameters) == 2
    client.query("select 2")
    assert configs[1].maximum_bytes_billed == 400
    client.query("select 3")
    assert client.remaining == 100 and client.billed == 900
    client.query("select 4")
    with pytest.raises(RuntimeError, match="exhausted"):
        client.query("select 5")
    assert client.job_ids == ["j600", "j300", "j0", "j500"]


def test_bounded_client_cancels_a_failed_job_and_refuses_unknown_bytes(fake_bigquery):
    failing, unknown = Job(1, fail=True), Job(None)
    raw = types.SimpleNamespace(query=lambda sql, job_config: failing)
    with pytest.raises(TimeoutError):
        cohort.BoundedClient(raw, 1000).query("select 1")
    assert failing.cancelled
    raw = types.SimpleNamespace(query=lambda sql, job_config: unknown)
    with pytest.raises(RuntimeError, match="billable bytes"):
        cohort.BoundedClient(raw, 1000).query("select 1")


def test_date_array_helper_round_trips():
    array = date_array([0.0, np.nan, 19000.0])
    assert array.null_count == 1 and array.type == pa.date32()
    assert cohort.days(pa.chunked_array([array])).tolist()[::2] == [0.0, 19000.0]
