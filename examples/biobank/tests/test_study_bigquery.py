"""BoundedClient against the real google-cloud-bigquery classes (the AoU wheelhouse's version).

A recording stand-in replaces only the network client, so every job config the
study sends is built by the library itself: dry runs, byte limits, and array
and scalar parameters, including an empty STRING array (no excluded branches).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pyarrow as pa
import pytest

bigquery = pytest.importorskip("google.cloud.bigquery")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from study import cohort  # noqa: E402


class Job:
    def __init__(self, config):
        self.config, self.job_id, self.cache_hit = config, f"job{id(config)}", False
        self.total_bytes_processed, self.total_bytes_billed = 12345, 20 * 1024 ** 2

    def result(self, timeout):
        return types.SimpleNamespace(to_arrow=lambda create_bqstorage_client: pa.table({"x": [1]}))


class Raw:
    def __init__(self):
        self.configs = []

    def get_table(self, table_id):  # every table exists, with the columns the SES probe looks for
        return types.SimpleNamespace(schema=[types.SimpleNamespace(name=n) for n in ("zip3", "deprivation_index")])

    def query(self, sql, job_config):
        self.configs.append(job_config)
        return Job(job_config)


def test_job_configs_are_built_by_the_library():
    raw = Raw()
    client = cohort.BoundedClient(raw, 50 * 10 ** 9)
    parameters = {"codes": ("STRING", ["44054006", "709044004"]), "branch_pairs": ("STRING", [])}
    sql = "SELECT 1 FROM `p-proj.d.concept` JOIN `p-proj.d.concept_ancestor` USING (x)"
    assert client.estimate(sql, parameters) == 12345 + 2 * cohort.BoundedClient.TABLE_MINIMUM
    dry = raw.configs[-1]
    assert isinstance(dry, bigquery.QueryJobConfig) and dry.dry_run and not dry.use_query_cache
    assert [(p.name, p.array_type, p.values) for p in dry.query_parameters] == [
        ("codes", "STRING", ["44054006", "709044004"]), ("branch_pairs", "STRING", [])]
    assert client.remaining == 50 * 10 ** 9  # a dry run bills nothing
    client.query(sql, {"n": ("INT64", 3), **parameters})
    run = raw.configs[-1]
    assert run.maximum_bytes_billed == 50 * 10 ** 9 and run.use_query_cache and not run.dry_run
    assert isinstance(run.query_parameters[0], bigquery.ScalarQueryParameter)
    assert client.remaining == 50 * 10 ** 9 - 20 * 1024 ** 2 and client.billed == 20 * 1024 ** 2


def test_every_study_query_builds_a_job():
    """Each query the AoU source plans builds a real job config with its parameters."""
    raw = Raw()
    client = cohort.BoundedClient(raw, 100 * 10 ** 9)
    source = cohort.AouSource(client, "fc-aou-cdr-prod-ct.C2024Q3R5", snomed_codes={"709044004": 443597},
                              excluded_branches={"709044004": {"431855005": 443614}}, scores=["PGS002237"],
                              ancestry="unused", prune="unused", projection="unused", score_cache="unused")
    plan = source.plan()
    assert list(plan) == ["person", "condition", "root", "descendants", "cutoff",
                          *(f"ehr_{domain}" for domain in cohort.EHR_DOMAINS)]
    assert len(raw.configs) == 9 and all(config.dry_run for config in raw.configs)
    branch = [p for config in raw.configs for p in config.query_parameters if p.name == "branch_pairs"]
    assert {tuple(p.values) for p in branch} == {("709044004:431855005",)}
