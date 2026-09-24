"""The study's data-access seam (SPEC section 1; the contract is SCHEMA.md).

The cohort stage produces one directory of named parquet tables: person,
condition, root, ancestry, pcs, scores and a manifest. In AoU, `AouSource`
builds them from BigQuery and the release files; the simulator writes the same
tables with `write_tables`. Every later stage reads them through
`ParquetSource`, so everything after the cohort runs identically on synthetic
data and in AoU. `validate_tables` enforces the contract on both sides.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCHEMA_VERSION = 2
TABLES = ("person", "condition", "root", "ancestry", "pcs", "scores")
SOURCES = ("bigquery", "simulator")
PGS_ID = re.compile(r"PGS\d{6}")
SNOMED_CODE = re.compile(r"\d{6,18}")
DATE_FLOOR = datetime.date(1900, 1, 1)
EPOCH = datetime.date(1970, 1, 1)
MIN_PCS = 6

# Column -> (Arrow type, nullable) for the tables whose columns do not depend
# on the manifest. pcs and scores are built by `table_columns`.
FIXED_COLUMNS = {
    "person": {
        "person_id": (pa.int64(), False),
        "birth_date": (pa.date32(), False),
        "sex_at_birth_concept_id": (pa.int64(), False),
        "race_concept_id": (pa.int64(), False),
        "ethnicity_concept_id": (pa.int64(), False),
        "baseline_date": (pa.date32(), True),
        "obs_start": (pa.date32(), True),
        "obs_end": (pa.date32(), True),
        "ehr_start": (pa.date32(), True),
        "ehr_end": (pa.date32(), True),
        "death_date": (pa.date32(), True),
        "state": (pa.string(), True),
        "ehr_site": (pa.string(), True),
        "zip3": (pa.int32(), True),
        "zip3_post_baseline": (pa.bool_(), True),
        "deprivation_index": (pa.float64(), True),
    },
    "condition": {
        "snomed_code": (pa.string(), False),
        "person_id": (pa.int64(), False),
        "first_date": (pa.date32(), False),
        "second_date": (pa.date32(), True),
        "n_dates": (pa.int32(), False),
    },
    "root": {
        "snomed_code": (pa.string(), False),
        "concept_id": (pa.int64(), False),
        "concept_name": (pa.string(), False),
        "n_descendants": (pa.int64(), False),
    },
    "ancestry": {
        "person_id": (pa.int64(), False),
        "ancestry_pred": (pa.string(), False),
        "related_excluded": (pa.bool_(), False),
    },
}
KEYS = {"person": ["person_id"], "condition": ["snomed_code", "person_id"], "root": ["snomed_code"],
        "ancestry": ["person_id"], "pcs": ["person_id"], "scores": ["person_id"]}


class SchemaError(ValueError):
    """A table or manifest breaks the SCHEMA.md contract."""


def table_columns(name, manifest):
    if name == "pcs":
        return {"person_id": (pa.int64(), False),
                **{f"PC{i}": (pa.float64(), False) for i in range(1, manifest["num_pcs"] + 1)}}
    if name == "scores":
        columns = {"person_id": (pa.int64(), False)}
        for pgs in manifest["scores"]:
            columns[pgs] = (pa.float64(), True)
            columns[f"{pgs}_missing_pct"] = (pa.float64(), True)
        return columns
    return FIXED_COLUMNS[name]


# --------------------------------------------------------------------------- #
# conformance and validation
# --------------------------------------------------------------------------- #
def _conform_column(name, column, source, target):
    if source == target:
        return column
    if pa.types.is_dictionary(source):
        return _conform_column(name, column.cast(source.value_type), source.value_type, target)
    if pa.types.is_date(target):
        if pa.types.is_date(source):
            return column.cast(target)
        if pa.types.is_timestamp(source) and source.tz is None:
            day = pc.floor_temporal(column, unit="day").cast(target)
            if pc.any(pc.not_equal(day.cast(source), column)).as_py():
                raise SchemaError(f"{name} holds a time of day; dates are calendar days")
            return day
    elif pa.types.is_string(target) and (pa.types.is_string(source) or pa.types.is_large_string(source)
                                         or getattr(pa.types, "is_string_view", lambda _: False)(source)):
        return column.cast(target)
    elif pa.types.is_integer(target) and pa.types.is_integer(source):
        return column.cast(target)  # a safe cast refuses values that do not fit
    elif pa.types.is_floating(target) and (pa.types.is_floating(source) or pa.types.is_integer(source)):
        return column.cast(target)
    elif pa.types.is_boolean(target) and pa.types.is_boolean(source):
        return column
    raise SchemaError(f"{name} is {source}, the contract says {target}")


def conform(name, table, manifest):
    """Select and order the contract's columns, casting only where no value can change."""
    columns = table_columns(name, manifest)
    missing = [c for c in columns if c not in table.column_names]
    unexpected = [c for c in table.column_names if c not in columns]
    if missing or unexpected:
        raise SchemaError(f"{name}: missing columns {missing}, unexpected columns {unexpected}")
    arrays = []
    for column, (kind, _) in columns.items():
        try:
            arrays.append(_conform_column(f"{name}.{column}", table.column(column),
                                          table.schema.field(column).type, kind))
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
            raise SchemaError(f"{name}.{column}: {error}") from error
    return pa.table(arrays, names=list(columns))


def days(column):
    """date32 column -> float64 days since 1970-01-01, NaN where null."""
    values = pc.fill_null(column.cast(pa.int32()), 0).to_numpy().astype(np.float64)
    values[column.is_null().to_numpy()] = np.nan
    return values


def _numbers(column):
    values = pc.fill_null(column.cast(pa.float64()), 0.0).to_numpy()
    return values, column.is_null().to_numpy()


def _require(condition, message):
    if not condition:
        raise SchemaError(message)


def validate_manifest(manifest):
    _require(manifest.get("schema_version") == SCHEMA_VERSION,
             f"manifest schema_version must be {SCHEMA_VERSION}")
    _require(manifest.get("source") in SOURCES, f"manifest source must be one of {SOURCES}")
    codes = manifest.get("snomed_codes")
    _require(isinstance(codes, list) and codes and all(isinstance(c, str) and SNOMED_CODE.fullmatch(c)
                                                       for c in codes) and len(set(codes)) == len(codes),
             "manifest snomed_codes must be distinct SNOMED concept codes")
    scores = manifest.get("scores")
    _require(isinstance(scores, list) and scores and all(isinstance(s, str) and PGS_ID.fullmatch(s)
                                                         for s in scores) and len(set(scores)) == len(scores),
             "manifest scores must be distinct PGS Catalog IDs")
    _require(isinstance(manifest.get("num_pcs"), int) and manifest["num_pcs"] >= MIN_PCS,
             f"manifest num_pcs must be an integer >= {MIN_PCS}")
    _require(isinstance(manifest.get("ses_available"), bool), "manifest ses_available must be a boolean")
    try:
        datetime.date.fromisoformat(manifest.get("cdr_cutoff"))
    except (TypeError, ValueError):
        raise SchemaError("manifest cdr_cutoff must be an ISO date") from None
    _require(isinstance(manifest.get("cdr_cutoff_source"), str), "manifest cdr_cutoff_source is required")
    branches = manifest.get("excluded_branches", {})
    _require(isinstance(branches, dict) and set(branches) <= set(codes)
             and all(isinstance(b, list) and b and len(set(b)) == len(b)
                     and all(isinstance(c, str) and SNOMED_CODE.fullmatch(c) for c in b)
                     for b in branches.values()),
             "manifest excluded_branches must map queried roots to distinct SNOMED codes")


def validate(tables, manifest):
    """Check conformed tables against the contract and each other; raise SchemaError."""
    validate_manifest(manifest)
    for name in TABLES:
        _require(name in tables, f"table {name} is missing")
        table = tables[name]
        columns = table_columns(name, manifest)
        _require(table.schema.names == list(columns), f"{name} is not conformed")
        for column, (kind, nullable) in columns.items():
            array = table.column(column)
            _require(array.type == kind, f"{name}.{column} is {array.type}, the contract says {kind}")
            if not nullable:
                _require(array.null_count == 0, f"{name}.{column} has nulls")
            if pa.types.is_floating(kind):
                values, null = _numbers(array)
                _require(np.isfinite(values[~null]).all(), f"{name}.{column} has non-finite values")
        keys = table.select(KEYS[name]).to_pandas()
        _require(not keys.duplicated().any(), f"{name} repeats its key {KEYS[name]}")

    # Person dates are CDR values: implausible ones (a birth after consent) are
    # filtered by the eligibility rules, not refused here. The rules below are
    # the ones every extraction guarantees by construction.
    person = tables["person"]
    baseline = days(person.column("baseline_date"))
    start, end = days(person.column("obs_start")), days(person.column("obs_end"))
    _require((np.isnan(start) == np.isnan(end)).all(), "person obs_start and obs_end must be null together")
    _require(not (~np.isnan(start) & np.isnan(baseline)).any(), "person has a covering period but no baseline")
    covered = ~np.isnan(start)
    _require(((start[covered] <= baseline[covered]) & (baseline[covered] <= end[covered])).all(),
             "person's observation period does not cover its baseline")
    ehr_start, ehr_end = days(person.column("ehr_start")), days(person.column("ehr_end"))
    _require((np.isnan(ehr_start) == np.isnan(ehr_end)).all(),
             "person ehr_start and ehr_end must be null together")
    both = ~np.isnan(ehr_start)
    _require((ehr_start[both] <= ehr_end[both]).all(), "person ehr_start must not follow ehr_end")
    site_null = person.column("ehr_site").is_null().to_numpy()
    _require(site_null[np.isnan(baseline)].all(),
             "person has an ehr_site but no baseline (sites are pre-baseline)")
    state = person.column("state").drop_null().to_pylist()
    _require(all(re.fullmatch(r"[A-Z]{2}", s) for s in state), "person.state must be a two-letter code")
    zip3, zip3_null = _numbers(person.column("zip3"))
    _require(((zip3[~zip3_null] >= 0) & (zip3[~zip3_null] <= 999)).all(), "person.zip3 is outside 0..999")
    _require(zip3_null[np.isnan(baseline)].all(), "person has a zip3 but no baseline (zip3 is chosen by baseline)")
    _require((person.column("zip3_post_baseline").is_null().to_numpy() == zip3_null).all(),
             "person.zip3_post_baseline must be present exactly when zip3 is")
    deprivation_null = person.column("deprivation_index").is_null().to_numpy()
    _require(deprivation_null[zip3_null].all(), "person has a deprivation_index but no zip3")
    if not manifest["ses_available"]:
        _require(deprivation_null.all(), "manifest says SES is unavailable but deprivation_index is present")

    condition = tables["condition"]
    count, _ = _numbers(condition.column("n_dates"))
    first, second = days(condition.column("first_date")), days(condition.column("second_date"))
    _require((count >= 1).all(), "condition.n_dates must be at least 1")
    _require(((count >= 2) == ~np.isnan(second)).all(),
             "condition.second_date must be present exactly when n_dates >= 2")
    paired = ~np.isnan(second)
    _require((first[paired] < second[paired]).all(), "condition.second_date must follow first_date")
    # The extraction keeps only records dated 1900-01-01 .. the extraction day.
    floor = (DATE_FLOOR - EPOCH).days
    ceiling = (datetime.date.today() - EPOCH).days
    for label, values in (("first_date", first), ("second_date", second)):
        known = values[~np.isnan(values)]
        _require(((known >= floor) & (known <= ceiling)).all(), f"condition.{label} lies outside 1900..today")
    codes = set(manifest["snomed_codes"])
    _require(set(pc.unique(condition.column("snomed_code")).to_pylist()) <= codes,
             "condition has a snomed_code the manifest lacks")
    _require(set(tables["root"].column("snomed_code").to_pylist()) == codes,
             "root must list every manifest snomed_code once")

    scores = tables["scores"]
    for pgs in manifest["scores"]:
        value_null = scores.column(pgs).is_null().to_numpy()
        missing, missing_null = _numbers(scores.column(f"{pgs}_missing_pct"))
        _require(((missing[~missing_null] >= 0) & (missing[~missing_null] <= 100)).all(),
                 f"scores.{pgs}_missing_pct is outside [0, 100]")
        # A score file that reports no missingness at all (the WGS score bank) leaves the whole
        # column null: then a null says "not reported", not "not scored".
        if not missing_null.all():
            _require(value_null[missing_null].all(), f"scores.{pgs} is present where the person was not scored")
            _require((value_null == (missing_null | (missing == 100))).all(),
                     f"scores.{pgs} must be null exactly when unscored or 100% missing")

    for name in TABLES:
        recorded = manifest.get("tables", {}).get(name, {}).get("rows")
        if recorded is not None:
            _require(recorded == tables[name].num_rows, f"manifest row count for {name} is stale")


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_arrow(value):
    if isinstance(value, pa.Table):
        return value
    return pa.Table.from_pandas(value, preserve_index=False)


def write_tables(directory, tables, manifest):
    """Conform, validate and write every table plus manifest.json.

    `tables` maps each name in TABLES to a pyarrow Table or pandas DataFrame;
    `manifest` holds everything but "schema_version" and "tables", which this
    fills. The manifest is written last, so a directory with a manifest is
    complete. Returns the written manifest.
    """
    directory = Path(directory)
    manifest = {**manifest, "schema_version": SCHEMA_VERSION}
    manifest.pop("tables", None)
    validate_manifest(manifest)
    conformed = {name: conform(name, _as_arrow(tables[name]), manifest) for name in TABLES}
    validate(conformed, manifest)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").unlink(missing_ok=True)
    manifest["tables"] = {}
    for name in TABLES:
        path = directory / f"{name}.parquet"
        pq.write_table(conformed[name], path, compression="zstd")
        manifest["tables"][name] = {"rows": conformed[name].num_rows, "sha256": _file_sha256(path)}
    manifest.setdefault("created_utc", datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"))
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# --------------------------------------------------------------------------- #
# sources: one interface over the parquet tables and over AoU
# --------------------------------------------------------------------------- #
class Source:
    """`manifest`, `table(name)` (validated pyarrow) and `frame(name)` (pandas)."""

    def table(self, name):
        raise NotImplementedError

    def frame(self, name, columns=None):
        table = self.table(name)
        if columns is not None:
            table = table.select(list(columns))
        return table.to_pandas(date_as_object=False)

    def condition(self, snomed_codes=None):
        table = self.table("condition")
        if snomed_codes is not None:
            wanted = pa.array(list(snomed_codes), pa.string())
            table = table.filter(pc.is_in(table.column("snomed_code"), value_set=wanted))
        return table

    def scores(self, pgs):
        return self.table("scores").select(["person_id", pgs, f"{pgs}_missing_pct"])


class ParquetSource(Source):
    """The tables of one extraction directory, validated once on open."""

    def __init__(self, directory, *, check_hashes=True):
        self.directory = Path(directory)
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.exists():
            raise SchemaError(f"{self.directory} has no manifest.json (an incomplete extraction)")
        self.manifest = json.loads(manifest_path.read_text())
        validate_manifest(self.manifest)
        self._tables = {}
        for name in TABLES:
            path = self.directory / f"{name}.parquet"
            recorded = self.manifest.get("tables", {}).get(name, {}).get("sha256")
            if check_hashes and recorded is not None and _file_sha256(path) != recorded:
                raise SchemaError(f"{path} does not match its manifest hash")
            self._tables[name] = conform(name, pq.read_table(path), self.manifest)
        validate(self._tables, self.manifest)

    def table(self, name):
        return self._tables[name]


def validate_tables(directory):
    """Check an extraction directory against the contract; returns its manifest or raises SchemaError."""
    return ParquetSource(directory).manifest


# --------------------------------------------------------------------------- #
# AoU: BigQuery
# --------------------------------------------------------------------------- #
class BoundedClient:
    """Run every query under the remaining share of one cumulative byte budget.

    `estimate` dry-runs a query (BigQuery bills nothing for it), so a caller can
    refuse a plan before any query runs (SPEC 7a).
    """

    # BigQuery bills at least 10 MB for every table a query references.
    TABLE_MINIMUM = 10 * 1024 ** 2

    def __init__(self, client, maximum_bytes_billed, *, timeout_seconds=1800, use_storage_api=False):
        if maximum_bytes_billed <= 0:
            raise ValueError("the BigQuery byte budget must be positive")
        self.client = client
        self.remaining = int(maximum_bytes_billed)
        self.billed = 0
        self.timeout_seconds = timeout_seconds
        self.use_storage_api = use_storage_api
        self.job_ids = []

    @staticmethod
    def _config(parameters, **settings):
        """`parameters` maps a name to (BigQuery type, scalar or list)."""
        from google.cloud import bigquery
        query_parameters = []
        for name, (kind, value) in (parameters or {}).items():
            if isinstance(value, (list, tuple)):
                query_parameters.append(bigquery.ArrayQueryParameter(name, kind, list(value)))
            else:
                query_parameters.append(bigquery.ScalarQueryParameter(name, kind, value))
        return bigquery.QueryJobConfig(query_parameters=query_parameters, **settings)

    def estimate(self, sql, parameters=None):
        """Bytes the query would bill: its dry-run bytes plus the per-table minimum, an upper bound."""
        job = self.client.query(sql, job_config=self._config(parameters, dry_run=True, use_query_cache=False))
        processed = job.total_bytes_processed
        if processed is None or processed < 0:
            raise RuntimeError("the BigQuery dry run reported no byte estimate")
        tables = len(set(re.findall(r"`([^`]+)`", sql)))
        return int(processed) + self.TABLE_MINIMUM * tables

    def query(self, sql, parameters=None):
        if self.remaining <= 0:
            raise RuntimeError("the cumulative BigQuery byte budget is exhausted")
        config = self._config(parameters, maximum_bytes_billed=self.remaining, use_query_cache=True,
                              job_timeout_ms=self.timeout_seconds * 1000)
        job = self.client.query(sql, job_config=config)
        self.job_ids.append(job.job_id)
        try:
            rows = job.result(timeout=self.timeout_seconds)
        except BaseException:
            job.cancel()
            raise
        billed = 0 if job.cache_hit else job.total_bytes_billed
        if billed is None or billed < 0:
            raise RuntimeError("BigQuery did not report billable bytes; refusing further queries")
        self.remaining -= billed
        self.billed += billed
        return rows.to_arrow(create_bqstorage_client=self.use_storage_api)

    def columns(self, table_id):
        """The column names of a table, or None when it does not exist (no bytes billed)."""
        from google.api_core.exceptions import NotFound
        try:
            return {field.name for field in self.client.get_table(table_id).schema}
        except NotFound:
            return None


def _check_cdr(cdr):
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,62}\.[A-Za-z0-9_]{1,1024}", cdr):
        raise ValueError(f"not a BigQuery project.dataset: {cdr!r}")
    return cdr


def person_sql(cdr, ses_columns):
    """One row per CDR person. Every date rule matches SCHEMA.md person.parquet.

    `ses_columns` are zip3_ses_map's columns, or None when the CDR lacks it.
    Should the map carry several ACS vintages of one zip3, the latest is used.
    One pass over the EHR-sourced visit and condition rows gives both the
    modal pre-baseline site and the EHR date range of these two domains
    (`ehr_sql` adds the others).
    """
    cdr = _check_cdr(cdr)
    if ses_columns:
        vintage = "acs DESC, " if "acs" in ses_columns else ""
        ses_join = f"""LEFT JOIN (SELECT zip3, deprivation_index FROM `{cdr}.zip3_ses_map` WHERE TRUE
                         QUALIFY ROW_NUMBER() OVER (PARTITION BY zip3 ORDER BY {vintage}deprivation_index) = 1
                       ) m ON m.zip3 = z.zip3"""
        deprivation = "m.deprivation_index"
    else:
        ses_join, deprivation = "", "CAST(NULL AS FLOAT64)"
    return f"""
      WITH consent AS (
        SELECT o.person_id, MIN(o.observation_date) AS baseline_date
        FROM `{cdr}.concept` c
        JOIN `{cdr}.concept_ancestor` ca ON ca.ancestor_concept_id = c.concept_id
        JOIN `{cdr}.observation` o ON o.observation_concept_id = ca.descendant_concept_id
        WHERE c.concept_name = 'Consent PII' AND c.concept_class_id = 'Module'
        GROUP BY o.person_id
      ), covering AS (
        SELECT op.person_id, op.observation_period_start_date AS obs_start,
               op.observation_period_end_date AS obs_end
        FROM `{cdr}.observation_period` op JOIN consent e ON e.person_id = op.person_id
        WHERE op.observation_period_start_date <= e.baseline_date
          AND op.observation_period_end_date >= e.baseline_date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY op.person_id ORDER BY op.observation_period_start_date,
                                   op.observation_period_end_date DESC, op.observation_period_id) = 1
      ), death AS (
        SELECT person_id, MIN(death_date) AS death_date
        FROM `{cdr}.aou_death`
        WHERE primary_death_record = TRUE AND death_date BETWEEN DATE '1900-01-01' AND CURRENT_DATE()
        GROUP BY person_id
      ), state AS (
        SELECT pe.person_id, NULLIF(REGEXP_EXTRACT(c.concept_code, r'^PIIState_([A-Z]{{2}})$'), '') AS state
        FROM `{cdr}.person_ext` pe JOIN `{cdr}.concept` c ON c.concept_id = pe.state_of_residence_concept_id
      ), ehr_rows AS (
        -- A visit lasts to its end date (an inpatient stay to discharge), at most a year: a junk end before
        -- the start is ignored, and a placeholder end (2099-12-31, an open stay) cannot grant follow-up.
        SELECT vo.person_id, vo.visit_start_date AS day,
               LEAST(GREATEST(vo.visit_start_date, IFNULL(vo.visit_end_date, vo.visit_start_date)),
                     DATE_ADD(vo.visit_start_date, INTERVAL 365 DAY)) AS day_end,
               IF(IFNULL(vo.visit_end_date, vo.visit_start_date) > DATE_ADD(vo.visit_start_date, INTERVAL 30 DAY),
                  LEAST(vo.visit_end_date, DATE_ADD(vo.visit_start_date, INTERVAL 365 DAY)), NULL) AS long_end,
               ve.src_id
        FROM `{cdr}.visit_occurrence` vo
        JOIN `{cdr}.visit_occurrence_ext` ve ON ve.visit_occurrence_id = vo.visit_occurrence_id
        WHERE REGEXP_CONTAINS(ve.src_id, r'(?i)EHR site')
          AND vo.visit_start_date BETWEEN DATE '1900-01-01' AND CURRENT_DATE()
        UNION ALL
        SELECT co.person_id, co.condition_start_date AS day, co.condition_start_date AS day_end,
               CAST(NULL AS DATE) AS long_end, ce.src_id
        FROM `{cdr}.condition_occurrence` co
        JOIN `{cdr}.condition_occurrence_ext` ce ON ce.condition_occurrence_id = co.condition_occurrence_id
        WHERE REGEXP_CONTAINS(ce.src_id, r'(?i)EHR site')
          AND co.condition_start_date BETWEEN DATE '1900-01-01' AND CURRENT_DATE()
      ), ehr_sites AS (
        SELECT r.person_id, r.src_id, COUNTIF(r.day < e.baseline_date) AS pre_baseline,
               MIN(r.day) AS first_day, MAX(r.day_end) AS last_day, MAX(r.long_end) AS long_day
        FROM ehr_rows r LEFT JOIN consent e ON e.person_id = r.person_id
        GROUP BY r.person_id, r.src_id
      ), ehr AS (
        SELECT person_id, IF(pre_baseline > 0, src_id, NULL) AS ehr_site, ehr_start, ehr_end, ehr_long_end
        FROM (SELECT person_id, src_id, pre_baseline,
                     MIN(first_day) OVER (PARTITION BY person_id) AS ehr_start,
                     MAX(last_day) OVER (PARTITION BY person_id) AS ehr_end,
                     MAX(long_day) OVER (PARTITION BY person_id) AS ehr_long_end,
                     ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY pre_baseline DESC, src_id) AS k
              FROM ehr_sites)
        WHERE k = 1
      ), zip AS (
        -- The address nearest baseline: the latest at or before it, else the earliest after it (flagged).
        SELECT person_id, zip3, post_baseline
        FROM (SELECT o.person_id, o.zip3, o.observation_date > e.baseline_date AS post_baseline,
                     ABS(DATE_DIFF(o.observation_date, e.baseline_date, DAY)) AS distance
              FROM (SELECT person_id, observation_date,
                           SAFE_CAST(SUBSTR(value_as_string, 1, STRPOS(value_as_string, '*') - 1) AS INT64) AS zip3
                    FROM `{cdr}.observation`
                    WHERE observation_source_concept_id = 1585250 AND value_as_string NOT LIKE 'Res%'
                      AND STRPOS(value_as_string, '*') > 1) o
              JOIN consent e ON e.person_id = o.person_id
              WHERE o.zip3 IS NOT NULL)
        WHERE TRUE
        QUALIFY ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY post_baseline, distance, zip3) = 1
      )
      SELECT p.person_id, DATE(p.birth_datetime) AS birth_date,
             IFNULL(p.sex_at_birth_concept_id, 0) AS sex_at_birth_concept_id,
             IFNULL(p.race_concept_id, 0) AS race_concept_id,
             IFNULL(p.ethnicity_concept_id, 0) AS ethnicity_concept_id,
             e.baseline_date, cv.obs_start, cv.obs_end, h.ehr_start, h.ehr_end, h.ehr_long_end,
             d.death_date, s.state,
             IF(e.baseline_date IS NULL, NULL, h.ehr_site) AS ehr_site,
             CAST(z.zip3 AS INT64) AS zip3, z.post_baseline AS zip3_post_baseline,
             {deprivation} AS deprivation_index
      FROM `{cdr}.person` p
      LEFT JOIN consent e ON e.person_id = p.person_id
      LEFT JOIN covering cv ON cv.person_id = p.person_id
      LEFT JOIN death d ON d.person_id = p.person_id
      LEFT JOIN state s ON s.person_id = p.person_id
      LEFT JOIN ehr h ON h.person_id = p.person_id
      LEFT JOIN zip z ON z.person_id = p.person_id
      {ses_join}
    """


# The EHR domains beyond visits and conditions (which `person_sql` reads):
# name -> (table, id column, date column). Their _ext rows mark EHR-sourced rows.
EHR_DOMAINS = {
    "procedure": ("procedure_occurrence", "procedure_occurrence_id", "procedure_date"),
    "drug": ("drug_exposure", "drug_exposure_id", "drug_exposure_start_date"),
    "observation": ("observation", "observation_id", "observation_date"),
    "measurement": ("measurement", "measurement_id", "measurement_date"),
}


def ehr_sql(cdr, domain):
    """Each person's first and last EHR-sourced date in one more domain: one grouped pass over it."""
    cdr = _check_cdr(cdr)
    table, key, day = EHR_DOMAINS[domain]
    return f"""
      SELECT t.person_id, MIN(t.{day}) AS ehr_start, MAX(t.{day}) AS ehr_end
      FROM `{cdr}.{table}` t JOIN `{cdr}.{table}_ext` x ON x.{key} = t.{key}
      WHERE REGEXP_CONTAINS(x.src_id, r'(?i)EHR site') AND t.{day} BETWEEN DATE '1900-01-01' AND CURRENT_DATE()
      GROUP BY t.person_id
    """


def merge_ehr(person, extra):
    """The person table with its EHR range widened by each extra domain's (person_id, ehr_start, ehr_end).

    `extra` maps a domain to its table. Also returns, per domain in order, the
    fraction of people with EHR whose ehr_end that domain moved later: an
    outcome-blind check of whether a skipped domain would matter."""
    index = pd.Index(person.column("person_id").to_numpy())
    start, end = days(person.column("ehr_start")), days(person.column("ehr_end"))
    extended = {}
    for domain, table in extra.items():
        at = index.get_indexer(table.column("person_id").to_numpy())
        found = at >= 0
        later = days(table.column("ehr_end"))[found]
        moved = ~(later <= end[at[found]])  # also counts people whose only EHR is in this domain
        start[at[found]] = np.fmin(start[at[found]], days(table.column("ehr_start"))[found])
        end[at[found]] = np.fmax(end[at[found]], later)
        with_ehr = int((~np.isnan(end)).sum())
        extended[domain] = float(moved.sum() / with_ehr) if with_ehr else None

    def column(values):
        null = np.isnan(values)
        return pa.array(np.where(null, 0, values).astype(np.int32), pa.int32(), mask=null).cast(pa.date32())

    for name, values in (("ehr_start", start), ("ehr_end", end)):
        person = person.set_column(person.schema.get_field_index(name), name, column(values))
    return person, extended


# The disease and exclusion roots with their excluded branches, from one
# concept scan and one concept_ancestor scan. @codes are the roots;
# @branch_pairs are "root:branch" codes. A descendant qualifies for a root
# unless an excluded branch of that root also covers it (the branch included).
_MEMBERS = """
      codes AS (
        SELECT c.concept_code AS code, c.concept_id, p.root_code, p.is_branch
        FROM `{cdr}.concept` c
        JOIN (SELECT code, code AS root_code, FALSE AS is_branch FROM UNNEST(@codes) AS code
              UNION ALL
              SELECT SPLIT(pair, ':')[OFFSET(1)], SPLIT(pair, ':')[OFFSET(0)], TRUE
              FROM UNNEST(@branch_pairs) AS pair) p ON p.code = c.concept_code
        WHERE c.vocabulary_id = 'SNOMED' AND c.standard_concept = 'S'
      ), reached AS (
        SELECT k.root_code, ca.descendant_concept_id AS concept_id, LOGICAL_AND(NOT k.is_branch) AS kept,
               MIN(IF(k.is_branch, k.code, NULL)) AS branch
        FROM codes k JOIN `{cdr}.concept_ancestor` ca ON ca.ancestor_concept_id = k.concept_id
        GROUP BY k.root_code, ca.descendant_concept_id
      )"""


def condition_sql(cdr):
    """Every root in one query: first and second distinct qualifying dates, and their count.

    Records under an excluded branch are dropped before the dates are taken."""
    cdr = _check_cdr(cdr)
    return f"""
      WITH {_MEMBERS.format(cdr=cdr)}, qualifying AS (
        SELECT DISTINCT r.root_code AS snomed_code, co.person_id, co.condition_start_date AS day
        FROM `{cdr}.condition_occurrence` co
        JOIN reached r ON r.concept_id = co.condition_concept_id AND r.kept
        WHERE co.condition_start_date BETWEEN DATE '1900-01-01' AND CURRENT_DATE()
      ), ranked AS (
        SELECT snomed_code, person_id, day,
               ROW_NUMBER() OVER (PARTITION BY snomed_code, person_id ORDER BY day) AS k
        FROM qualifying
      )
      SELECT snomed_code, person_id, MIN(day) AS first_date, MIN(IF(k = 2, day, NULL)) AS second_date,
             COUNT(*) AS n_dates
      FROM ranked GROUP BY snomed_code, person_id
    """


def root_sql(cdr):
    """Each root and excluded branch (@codes): its standard SNOMED concept and descendant count."""
    cdr = _check_cdr(cdr)
    return f"""
      SELECT c.concept_code AS snomed_code, c.concept_id, c.concept_name,
             COUNT(ca.descendant_concept_id) AS n_descendants
      FROM `{cdr}.concept` c
      LEFT JOIN `{cdr}.concept_ancestor` ca ON ca.ancestor_concept_id = c.concept_id
      WHERE c.vocabulary_id = 'SNOMED' AND c.standard_concept = 'S' AND c.concept_code IN UNNEST(@codes)
      GROUP BY c.concept_code, c.concept_id, c.concept_name
    """


def descendant_sql(cdr):
    """Outcome-blind phenotype check: the ten descendant concepts recorded for the most people.

    One list per root, over its qualifying concepts, and one per excluded
    branch, over the concepts it removes. Only concepts recorded for more
    than 20 people are listed (the AoU small-cell rule)."""
    cdr = _check_cdr(cdr)
    return f"""
      WITH {_MEMBERS.format(cdr=cdr)}, counts AS (
        SELECT IF(r.kept, r.root_code, r.branch) AS snomed_code,
               IF(r.kept, 'root', 'excluded_branch') AS role, r.root_code,
               co.condition_concept_id AS concept_id, COUNT(DISTINCT co.person_id) AS n_persons
        FROM `{cdr}.condition_occurrence` co
        JOIN reached r ON r.concept_id = co.condition_concept_id AND (r.kept OR r.branch IS NOT NULL)
        GROUP BY 1, 2, 3, 4
      )
      SELECT n.snomed_code, n.role, n.root_code, n.concept_id, c.concept_name, n.n_persons
      FROM counts n JOIN `{cdr}.concept` c ON c.concept_id = n.concept_id
      WHERE n.n_persons > 20
      QUALIFY ROW_NUMBER() OVER (PARTITION BY n.snomed_code, n.role ORDER BY n.n_persons DESC, n.concept_id) <= 10
    """

# --------------------------------------------------------------------------- #
# AoU: release files
# --------------------------------------------------------------------------- #
def _person_ids(values, what):
    text = pd.Series(values, dtype="string").str.strip()
    if text.isna().any() or not text.str.fullmatch(r"[0-9]{1,18}").all():
        raise SchemaError(f"{what} has missing or non-numeric person IDs")
    return text.astype("int64").to_numpy()


def read_ancestry(ancestry_path, prune_path):
    """The release ancestry predictions with the published relatedness prune marked.

    Returns (table, prune IDs absent from the ancestry file)."""
    header = pd.read_csv(ancestry_path, sep="\t", nrows=0).columns
    if not {"research_id", "ancestry_pred"} <= set(header):
        raise SchemaError("ancestry file lacks research_id/ancestry_pred")
    labels = pd.read_csv(ancestry_path, sep="\t", dtype=str, usecols=["research_id", "ancestry_pred"],
                         keep_default_na=False)
    person_id = _person_ids(labels.research_id, "ancestry file")
    ancestry = labels.ancestry_pred.str.strip()
    if ancestry.eq("").any():
        raise SchemaError("ancestry file has empty ancestry_pred labels")
    prune = pd.read_csv(prune_path, sep="\t", dtype=str, keep_default_na=False, skip_blank_lines=False)
    if list(prune.columns) != ["sample_id"] or prune.empty:
        raise SchemaError("relatedness prune must be one non-empty sample_id column")
    pruned = np.unique(_person_ids(prune.sample_id, "relatedness prune"))
    table = pa.table({"person_id": pa.array(person_id, pa.int64()),
                      "ancestry_pred": pa.array(ancestry.to_numpy(), pa.string()),
                      "related_excluded": pa.array(np.isin(person_id, pruned), pa.bool_())})
    return table, int((~np.isin(pruned, person_id)).sum())


def read_pcs(projection_path, num_pcs=None):
    """pgsEngine's projection_pcs.parquet: IID plus contiguous PC1..PCk."""
    table = pq.read_table(projection_path)
    available = 0
    while f"PC{available + 1}" in table.column_names:
        available += 1
    num_pcs = available if num_pcs is None else num_pcs
    if "IID" not in table.column_names or num_pcs < MIN_PCS or num_pcs > available:
        raise SchemaError(f"projection needs IID and PC1..PC{max(num_pcs, MIN_PCS)}; it has {available} PCs")
    ids = _person_ids(table.column("IID").to_pandas().astype(str), "projection")
    columns = {"person_id": pa.array(ids, pa.int64())}
    for i in range(1, num_pcs + 1):
        columns[f"PC{i}"] = table.column(f"PC{i}").cast(pa.float64())
    return pa.table(columns)


def _sscore_header(handle):
    """Skip #SCORE_VARIANT_COUNT/#REGION metadata; return (lines before header, header fields)."""
    for skipped, raw in enumerate(handle):
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        fields = line.rstrip("\r\n").split("\t")
        if any(field.lstrip("#") == "IID" for field in fields[:2]):
            return skipped, fields
        if not line.startswith("#"):
            break
    raise SchemaError("sscore file has no IID header")


def _sscore_members(score_cache):
    """(name, opener) for every .sscore in a directory or a (compressed) tar."""
    path = Path(score_cache)
    if path.is_dir():
        for member in sorted(path.glob("*.sscore")):
            yield member.name, (lambda member=member: member.open("rb"))
        return
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if member.isfile() and member.name.endswith(".sscore"):
                yield member.name, (lambda member=member: archive.extractfile(member))


def read_scores(score_cache, pgs_ids):
    """Wide score table: each PGS's _AVG and _MISSING_PCT, found exactly once in the cache."""
    wanted = list(dict.fromkeys(pgs_ids))
    found = {}
    frames = []
    for name, opener in _sscore_members(score_cache):
        with opener() as handle:
            skipped, header = _sscore_header(handle)
        present = [pgs for pgs in wanted if f"{pgs}_AVG" in header]
        if not present:
            continue
        for pgs in present:
            if pgs in found:
                raise SchemaError(f"{pgs} appears in both {found[pgs]} and {name}")
            found[pgs] = name
        # The workspace's WGS score bank writes each score's average without its per-participant
        # missingness: that column is then null (unknown), never a number the file did not carry.
        with_missing = [pgs for pgs in present if f"{pgs}_MISSING_PCT" in header]
        iid = next(field for field in header[:2] if field.lstrip("#") == "IID")
        columns = [iid] + [f"{pgs}_AVG" for pgs in present] + [f"{pgs}_MISSING_PCT" for pgs in with_missing]
        with opener() as handle:
            frame = pd.read_csv(handle, sep="\t", skiprows=skipped, usecols=columns, dtype={iid: str},
                                float_precision="round_trip")  # the exact double each value spells
        # By name: read_csv returns usecols in the file's column order, not the requested one.
        frame = frame.rename(columns={iid: "person_id", **{f"{pgs}_AVG": pgs for pgs in present},
                                      **{f"{pgs}_MISSING_PCT": f"{pgs}_missing_pct" for pgs in present}})
        for pgs in present:
            if pgs not in with_missing:
                frame[f"{pgs}_missing_pct"] = np.nan
        frame["person_id"] = _person_ids(frame.person_id, name)
        if frame.person_id.duplicated().any():
            raise SchemaError(f"{name} repeats participants")
        frames.append(frame.set_index("person_id"))
    absent = [pgs for pgs in wanted if pgs not in found]
    if absent:
        raise SchemaError(f"scores absent from the cache: {absent}")
    scores = pd.concat(frames, axis=1, join="outer") if len(frames) > 1 else frames[0]
    for pgs in wanted:
        scores.loc[scores[f"{pgs}_missing_pct"].eq(100), pgs] = np.nan
    return pa.Table.from_pandas(scores.reset_index()[["person_id", *[c for pgs in wanted
                                for c in (pgs, f"{pgs}_missing_pct")]]], preserve_index=False)


class AouSource(Source):
    """The contract's tables built in AoU: BigQuery for the CDR, release files for the rest.

    `client` is a BoundedClient. `snomed_codes` maps each disease and exclusion
    root to its declared OMOP concept_id (or None); `excluded_branches` maps a
    root to {branch: declared concept_id or None}, the records that do not
    qualify for it. Resolved concepts must match the declarations
    (`phenotypes.phenotype_codes` builds both from diseases.json).

    ehr_end is the EHR range over visits, conditions and every EHR_DOMAINS
    table: one definition, which the plan must fit whole.

    Before any query bills, every query is dry-run and the plan is refused if
    it would exceed the client's remaining budget (SPEC 7a). Tables are built
    on first use; `export` writes them, plus the outcome-blind descendant log,
    so later stages read a ParquetSource.
    """

    def __init__(self, client, cdr, *, snomed_codes, scores, ancestry, prune, projection, score_cache,
                 excluded_branches=None, num_pcs=None):
        self.client = client
        self.cdr = _check_cdr(cdr)
        declared = snomed_codes if isinstance(snomed_codes, dict) else dict.fromkeys(snomed_codes)
        self.snomed_codes = [str(code) for code in declared]
        self.branches = {str(root): {str(b): c for b, c in (branches if isinstance(branches, dict)
                                                           else dict.fromkeys(branches)).items()}
                         for root, branches in (excluded_branches or {}).items()}
        if not set(self.branches) <= set(self.snomed_codes):
            raise ValueError("excluded branches name a root that is not queried")
        self.declared = {str(code): concept for code, concept in declared.items()}
        for branches in self.branches.values():
            self.declared.update(branches)
        self.pgs_ids = list(dict.fromkeys(scores))
        self.paths = {"ancestry": ancestry, "prune": prune, "projection": projection, "score_cache": score_cache}
        self.num_pcs = num_pcs
        self._tables = {}
        self._facts = {}
        self._manifest = None

    def _ses_columns(self):
        """zip3_ses_map's columns when it has what the SES stratum needs, else None."""
        if "ses" not in self._facts:
            columns = self.client.columns(f"{self.cdr}.zip3_ses_map")
            usable = columns is not None and {"zip3", "deprivation_index"} <= columns
            self._facts["ses"] = columns if usable else None
        return self._facts["ses"]

    def _cutoff_sql(self):
        """The CDR's data cutoff: AoU's curation caps every observation period at ehr_cutoff_date."""
        return f"SELECT MAX(observation_period_end_date) AS cutoff FROM `{self.cdr}.observation_period`"

    def _members(self):
        pairs = [f"{root}:{branch}" for root, branches in self.branches.items() for branch in sorted(branches)]
        return {"codes": ("STRING", self.snomed_codes), "branch_pairs": ("STRING", pairs)}

    def _queries(self):
        """Every billed query of an extraction: {name: (sql, parameters)}."""
        every_code = self.snomed_codes + sorted({b for branches in self.branches.values() for b in branches})
        return {"person": (person_sql(self.cdr, self._ses_columns()), None),
                "condition": (condition_sql(self.cdr), self._members()),
                "root": (root_sql(self.cdr), {"codes": ("STRING", every_code)}),
                "descendants": (descendant_sql(self.cdr), self._members()),
                "cutoff": (self._cutoff_sql(), None),
                **{f"ehr_{domain}": (ehr_sql(self.cdr, domain), None) for domain in EHR_DOMAINS}}

    def plan(self):
        """Dry-run every query and refuse a plan over the remaining budget. {name: bytes}."""
        if "plan" not in self._facts:
            estimates = {name: self.client.estimate(sql, parameters)
                         for name, (sql, parameters) in self._queries().items()}
            total = sum(estimates.values())
            if total > self.client.remaining:
                raise RuntimeError(f"the BigQuery plan would bill up to {total:,} bytes, over the remaining "
                                   f"budget of {self.client.remaining:,}: {estimates}")
            self._facts["plan"] = estimates
        return self._facts["plan"]

    def _query(self, name):
        self.plan()
        sql, parameters = self._queries()[name]
        return self.client.query(sql, parameters)

    def _build(self, name):
        if name == "person":
            person = self._query("person")
            for name in ("ehr_start", "ehr_end", "ehr_long_end"):  # as calendar days, whatever the engine returns
                column = person.column(name)
                person = person.set_column(person.schema.get_field_index(name), name,
                                           _conform_column(f"person.{name}", column, column.type, pa.date32()))
            extra = {domain: self._query(f"ehr_{domain}") for domain in EHR_DOMAINS}
            person, self._facts["ehr_extended"] = merge_ehr(person, extra)
            # Outcome-blind check: how often the EHR end is a visit end more than 30 days after its start.
            end, long_end = days(person.column("ehr_end")), days(person.column("ehr_long_end"))
            known = ~np.isnan(end)
            if not known.any():
                raise SchemaError("no person in the CDR has an EHR-sourced record")
            self._facts["ehr_end_from_long_visit"] = float((long_end[known] == end[known]).mean())
            self._facts["ehr_people"] = int(known.sum())  # the two shares' denominator
            return person.drop_columns(["ehr_long_end"])
        if name == "condition":
            return self._query(name)
        if name == "root":
            table = self._query("root")
            resolved = table.column("snomed_code").to_pylist()
            unresolved = sorted(set(self.declared) - set(resolved))
            repeated = sorted({code for code in resolved if resolved.count(code) > 1})
            if unresolved or repeated:
                raise SchemaError(f"SNOMED codes unresolved {unresolved} or ambiguous {repeated}")
            found = dict(zip(resolved, table.column("concept_id").to_pylist()))
            wrong = {code: (found[code], concept) for code, concept in self.declared.items()
                     if concept is not None and found[code] != concept}
            if wrong:
                raise SchemaError("SNOMED codes resolve to other concepts than declared "
                                  f"(found, declared): {wrong}")
            roots = pc.is_in(table.column("snomed_code"), value_set=pa.array(self.snomed_codes, pa.string()))
            return table.filter(roots)
        if name == "ancestry":
            table, self._facts["prune_unmatched"] = read_ancestry(self.paths["ancestry"], self.paths["prune"])
            return table
        if name == "pcs":
            return read_pcs(self.paths["projection"], self.num_pcs)
        if name == "scores":
            return read_scores(self.paths["score_cache"], self.pgs_ids)
        raise KeyError(name)

    def _raw(self, name):
        if name not in self._tables:
            self._tables[name] = self._build(name)
        return self._tables[name]

    def descendants(self):
        """The ten most-recorded descendant concepts per root and per excluded branch (more than 20 people)."""
        if "descendants" not in self._facts:
            self._facts["descendants"] = self._query("descendants")
        return self._facts["descendants"]

    @property
    def manifest(self):
        if self._manifest is None:
            plan = self.plan()
            raw = {name: self._raw(name) for name in TABLES}
            cutoff = self._query("cutoff").column("cutoff")[0].as_py()
            if isinstance(cutoff, str):
                cutoff = datetime.date.fromisoformat(cutoff[:10])
            manifest = {"schema_version": SCHEMA_VERSION, "source": "bigquery", "cdr": self.cdr,
                        "snomed_codes": self.snomed_codes, "scores": self.pgs_ids,
                        "excluded_branches": {root: sorted(b) for root, b in self.branches.items()},
                        "num_pcs": len(raw["pcs"].column_names) - 1,
                        "ses_available": self._ses_columns() is not None,
                        "cdr_cutoff": cutoff.isoformat(), "cdr_cutoff_source": "max(observation_period_end_date)",
                        "prune_unmatched": self._facts["prune_unmatched"],
                        "ehr_domains": ["visit", "condition", *EHR_DOMAINS],
                        "ehr_extended_by": self._facts["ehr_extended"],
                        "ehr_end_from_long_visit": self._facts["ehr_end_from_long_visit"],
                        "ehr_people": self._facts["ehr_people"],
                        "bigquery": {"plan_bytes": plan, "bytes_billed": self.client.billed,
                                     "job_ids": list(self.client.job_ids)}}
            self._tables = {name: conform(name, raw[name], manifest) for name in TABLES}
            validate(self._tables, manifest)
            self._manifest = manifest
        return self._manifest

    def table(self, name):
        self.manifest
        return self._tables[name]

    def export(self, directory):
        manifest = dict(self.manifest)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        descendants = self.descendants()
        pq.write_table(descendants, directory / "descendants.parquet", compression="zstd")
        manifest["descendants"] = {"rows": descendants.num_rows,
                                   "sha256": _file_sha256(directory / "descendants.parquet")}
        manifest["bigquery"] = {**manifest["bigquery"], "bytes_billed": self.client.billed,
                                "job_ids": list(self.client.job_ids)}
        write_tables(directory, self._tables, manifest)
        return ParquetSource(directory)
