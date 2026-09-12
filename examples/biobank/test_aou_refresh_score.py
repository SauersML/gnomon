"""Exact component handoff from the native scorer, without participant data."""
import pytest
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from aou_refresh_score import component_scores, save_scoring_state, microarray_prefix
from aou_identity import require_spot_amd
from aou_status import score_progress_label
from aou_checkpoint import StudyCheckpoint
from aou_survival import bounded_fit, BoundedClient


def test_microarray_scoring_rejects_wgs_and_other_sources():
    prefix = "gs://controlled/v8/microarray/plink/arrays"
    assert microarray_prefix(prefix) == prefix
    for invalid in ("gs://controlled/v8/wgs/arrays", prefix + ".bed", prefix + "?wgs=1", "/local/arrays"):
        with pytest.raises(ValueError, match="microarray"):
            microarray_prefix(invalid)


@pytest.mark.parametrize("spot,vendor,accepted", [(b"TRUE", "AuthenticAMD", True),
                                                    (b"FALSE", "AuthenticAMD", False),
                                                    (b"TRUE", "GenuineIntel", False)])
def test_runtime_refuses_nonspot_or_nonamd(spot, vendor, accepted):
    from unittest.mock import MagicMock
    replies = []
    for value in (spot, b"projects/example/machineTypes/n2d-standard-4"):
        reply = MagicMock()
        reply.__enter__.return_value.read.return_value = value
        replies.append(reply)
    with patch("aou_identity.urllib.request.urlopen", side_effect=replies), patch("pathlib.Path.read_text", return_value=vendor):
        if accepted:
            assert require_spot_amd()["preemptible"] is True
        else:
            with pytest.raises(RuntimeError, match="requires an AMD"):
                require_spot_amd()


def test_running_child_publishes_checkpoint_before_completion(tmp_path):
    child = MagicMock()
    child.wait.side_effect = [subprocess.TimeoutExpired(["scorer"], 30), 0]
    published = MagicMock()
    with patch("aou_survival.subprocess.Popen", return_value=child), \
         patch("aou_survival.time.monotonic", side_effect=[0., 0., 30., 30., 31.]):
        bounded_fit(["scorer"], 60, tmp_path / "fit.log", checkpoint_callback=published)
    published.assert_called_once_with()
    assert child.wait.call_count == 2
    assert (tmp_path / "fit.resources.json").is_file()


def test_query_budget_is_cumulative_and_cached_queries_are_free():
    client = BoundedClient.__new__(BoundedClient)
    client.config = {"query_timeout_seconds": 10}
    client.remaining_bytes = 100
    client.jobs = []
    client.client = MagicMock()
    first = MagicMock(cache_hit=False, total_bytes_billed=60)
    cached = MagicMock(cache_hit=True, total_bytes_billed=None)
    last = MagicMock(cache_hit=False, total_bytes_billed=40)
    client.client.query.side_effect = [first, cached, last]
    sdk = SimpleNamespace(QueryJobConfig=lambda: SimpleNamespace())
    with patch.dict(sys.modules, {"google.cloud.bigquery": sdk}):
        for query in ("first", "cached", "last"):
            client.query(query)
    configs = [call.kwargs["job_config"] for call in client.client.query.call_args_list]
    assert [c.maximum_bytes_billed for c in configs] == [100, 40, 40]
    assert all(c.use_query_cache for c in configs)
    with patch.dict(sys.modules, {"google.cloud.bigquery": sdk}), pytest.raises(RuntimeError, match="budget is exhausted"):
        client.query("not submitted")
    assert client.client.query.call_count == 3


def test_vm_retry_restores_latest_workspace_checkpoint(tmp_path):
    original = StudyCheckpoint(tmp_path / "first", "gs://workspace/checkpoint", "project",
                               "service@example.org", {"inputs": "frozen"})
    (original.root / "score.sscore").write_bytes(b"completed scores")
    with patch.object(original, "publish"):
        original.complete_step(original.root, ["score.sscore"])
    import io
    import tarfile
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in original.root.iterdir():
            archive.add(path, arcname=path.name)
    response = MagicMock()
    response.status_code = 200
    response.iter_content.return_value = [buffer.getvalue()]
    session = MagicMock()
    session.__enter__.return_value.get.return_value.__enter__.return_value = response
    with patch("google.auth.transport.requests.AuthorizedSession", return_value=session):
        restored = StudyCheckpoint(tmp_path / "retry", "gs://workspace/checkpoint", "project",
                                   "service@example.org", {"inputs": "frozen"}, resume_latest=True)
    assert restored.step_is_complete(restored.root)
    assert (restored.root / "score.sscore").read_bytes() == b"completed scores"


@pytest.mark.parametrize("status", [404, 403])
def test_checkpoint_missing_is_distinct_from_denied_access(tmp_path, status):
    response = MagicMock()
    response.status_code = status
    response.raise_for_status.side_effect = RuntimeError("denied")
    session = MagicMock()
    session.__enter__.return_value.get.return_value.__enter__.return_value = response
    with patch("google.auth.transport.requests.AuthorizedSession", return_value=session):
        if status == 404:
            cp = StudyCheckpoint(tmp_path / "new", "gs://workspace/checkpoint", "project",
                                 "service@example.org", {}, resume_latest=True)
            assert not cp.step_is_complete(cp.root)
            response.raise_for_status.assert_not_called()
        else:
            with pytest.raises(RuntimeError, match="denied"):
                StudyCheckpoint(tmp_path / "denied", "gs://workspace/checkpoint", "project",
                                "service@example.org", {}, resume_latest=True)


def test_components_preserve_denominator_and_exclude_unobserved(tmp_path):
    path = tmp_path / "score.sscore"
    path.write_text("#SCORE_VARIANT_COUNT\tSCORE\tCOUNT\n"
                    "#SCORE_VARIANT_COUNT\tPGS004525\t4\n"
                    "#IID\tPGS004525_SUM\tPGS004525_MISSING_CT\n"
                    "observed\t6\t1\nmissing\t0\t4\nzero\t0\t0\n")
    scores, total, excluded, ids = component_scores(path, "PGS004525")
    assert total == 4 and excluded == 1
    assert ids == {"observed", "missing", "zero"}
    assert scores["#IID"].tolist() == ["observed", "zero"]
    assert scores["PGS004525_AVG"].tolist() == [2., 0.]
    assert scores["PGS004525_MISSING_PCT"].tolist() == [25., 0.]


@pytest.mark.parametrize("missing", ["-1", "5", "1.5", "nan"])
def test_invalid_native_missing_counts_fail(tmp_path, missing):
    path = tmp_path / "score.sscore"
    path.write_text("#SCORE_VARIANT_COUNT\tSCORE\tCOUNT\n"
                    "#SCORE_VARIANT_COUNT\tPGS004525\t4\n"
                    "#IID\tPGS004525_SUM\tPGS004525_MISSING_CT\n"
                    f"sample\t0\t{missing}\n")
    with pytest.raises(ValueError):
        component_scores(path, "PGS004525")


def test_scoring_checkpoint_preserves_continuation_without_genotype_spools(tmp_path):
    score_dir = tmp_path / "scoring"
    score_dir.mkdir()
    native = score_dir / "score.sscore.gnomon-checkpoint.bin"
    native.write_bytes(b"native continuation state")
    (score_dir / "genotype-spool.bin").write_bytes(b"must stay outside checkpoint")
    log = tmp_path / "score.log"
    log.write_text("native progress")
    state = StudyCheckpoint(tmp_path / "state", "gs://workspace/checkpoint", "project",
                            "service@example.org", {"inputs": "frozen"})
    with patch.object(state, "publish") as publish:
        save_scoring_state(state, score_dir, log, complete=False, status_uri="gs://workspace/status")
        publish.assert_called_once()
    assert (state.root / native.name).read_bytes() == native.read_bytes()
    assert not (state.root / "genotype-spool.bin").exists()
    assert not state.step_is_complete(state.root)
    native.unlink()  # the native scorer removes its continuation file on success
    output = score_dir / "score.sscore"
    output.write_text("completed native score components")
    with patch.object(state, "publish"):
        save_scoring_state(state, score_dir, log, complete=True, status_uri="gs://workspace/status")
    assert not (state.root / native.name).exists()
    assert state.step_is_complete(state.root)
    (state.root / output.name).write_text("corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        state.step_is_complete(state.root)


def test_progress_reports_only_fixed_categories(tmp_path):
    log = tmp_path / "score.log"
    log.write_text("private details\n> Progress: 1/4 variants (25%)\n")
    assert score_progress_label(log) == "score_progress_25_49"
    log.write_text("private error contains 100% but is not a progress line")
    assert score_progress_label(log) is None
    log.write_text("> Progress: 9/4 variants (225%)\n")
    with pytest.raises(ValueError, match="progress"):
        score_progress_label(log)
