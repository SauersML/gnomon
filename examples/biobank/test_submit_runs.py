"""Workbench staging and submission recovery, without Workbench or Cloud Storage."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
from unittest.mock import MagicMock, patch
import urllib.error
import urllib.parse

import pytest

import submit_aou
import submit_runs
from submit_aou import Workbench, WorkbenchTimeout


def workbench():
    wb = Workbench.__new__(Workbench)
    wb.workspace, wb.project, wb.bucket_id, wb.bucket = "workspace", "project", "bucket-id", "gs://bucket"
    wb.env = dict(os.environ)
    return wb


def response(record):
    reply = MagicMock()
    reply.__enter__.return_value = io.BytesIO(json.dumps(record).encode())
    return reply


def md5(body):
    return base64.b64encode(hashlib.md5(body).digest()).decode()


def test_stage_posts_each_file_once_and_checks_md5_and_size(tmp_path):
    files = [tmp_path / "sources.tar", tmp_path / "workflow.wdl"]
    for index, path in enumerate(files):
        path.write_bytes(bytes([index]) * (1000 + index))
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        name = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)["name"][0]
        return response({"name": name, "md5Hash": md5(request.data), "size": str(len(request.data))})

    wb = workbench()
    with patch.object(Workbench, "assert_identity") as identity, \
         patch.object(Workbench, "command", return_value="token\n"), \
         patch("submit_aou.urllib.request.urlopen", side_effect=urlopen):
        uris = wb.stage(files, "gs://bucket/workflows/run-1/")
    identity.assert_called_once_with()
    assert uris == ["gs://bucket/workflows/run-1/sources.tar", "gs://bucket/workflows/run-1/workflow.wdl"]
    assert [request.method for request in requests] == ["POST", "POST"]
    assert [request.data for request in requests] == [path.read_bytes() for path in files]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requests[0].full_url).query)
    assert requests[0].full_url.startswith("https://storage.googleapis.com/upload/storage/v1/b/bucket/o?")
    assert query == {"uploadType": ["media"], "name": ["workflows/run-1/sources.tar"], "userProject": ["project"]}
    assert requests[0].get_header("Authorization") == "Bearer token"


@pytest.mark.parametrize("field,value", [("md5Hash", "AAAA"), ("size", "1"), ("name", "workflows/other")])
def test_stage_rejects_an_object_that_differs_from_the_file(tmp_path, field, value):
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    meta = {"name": "workflows/run/config.json", "md5Hash": md5(b"{}"), "size": "2", field: value}
    with patch.object(Workbench, "assert_identity"), \
         patch.object(Workbench, "command", return_value="token"), \
         patch("submit_aou.urllib.request.urlopen", return_value=response(meta)):
        with pytest.raises(RuntimeError, match="does not match"):
            workbench().stage([path], "gs://bucket/workflows/run")


def test_stage_refuses_objects_outside_the_workspace_bucket(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"x")
    with patch.object(Workbench, "assert_identity"), \
         patch.object(Workbench, "command", return_value="token"), \
         patch("submit_aou.urllib.request.urlopen") as urlopen:
        with pytest.raises(ValueError, match="workspace bucket"):
            workbench().stage([path], "gs://other-bucket/workflows/run")
    urlopen.assert_not_called()


def test_reused_object_with_the_same_md5_is_not_uploaded_again(tmp_path):
    archive = tmp_path / "scorer.tar.gz"
    archive.write_bytes(b"scorer")
    meta = {"name": "artifacts/scorer.tar.gz", "md5Hash": md5(b"scorer"), "size": "6"}
    with patch.object(Workbench, "assert_identity"), \
         patch.object(Workbench, "command", return_value="token"), \
         patch("submit_aou.urllib.request.urlopen", return_value=response({"items": [meta]})) as urlopen:
        workbench().stage_as(archive, "gs://bucket/artifacts/scorer.tar.gz", reuse=True)
    requests = [call.args[0] for call in urlopen.call_args_list]
    assert [request.method for request in requests] == ["GET"]
    # A listing, not a metadata GET of the object, which the workspace perimeter refuses with HTTP 403.
    assert requests[0].full_url.startswith("https://storage.googleapis.com/storage/v1/b/bucket/o?")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(requests[0].full_url).query)
    assert query == {"prefix": ["artifacts/scorer.tar.gz"], "fields": ["items(name,md5Hash,size)"],
                     "userProject": ["project"]}


def test_reuse_needs_the_exact_name_not_a_listed_prefix_sibling(tmp_path):
    archive = tmp_path / "scorer.tar.gz"
    archive.write_bytes(b"scorer")
    sibling = {"name": "artifacts/scorer.tar.gz.partial", "md5Hash": md5(b"scorer"), "size": "6"}
    replies = [response({"items": [sibling]}),
               response({"name": "artifacts/scorer.tar.gz", "md5Hash": md5(b"scorer"), "size": "6"})]
    with patch.object(Workbench, "assert_identity"), \
         patch.object(Workbench, "command", return_value="token"), \
         patch("submit_aou.urllib.request.urlopen", side_effect=replies) as urlopen:
        workbench().stage_as(archive, "gs://bucket/artifacts/scorer.tar.gz", reuse=True)
    assert [call.args[0].method for call in urlopen.call_args_list] == ["GET", "POST"]


def test_storage_retries_server_errors_but_not_policy_denials():
    wb = workbench()
    denied = urllib.error.HTTPError("https://storage", 403, "denied", {}, io.BytesIO(b""))
    with patch("submit_aou.urllib.request.urlopen", side_effect=denied) as urlopen:
        with pytest.raises(RuntimeError, match="HTTP 403"):
            wb.storage("POST", "https://storage", "token", b"x")
    assert urlopen.call_count == 1
    busy = urllib.error.HTTPError("https://storage", 503, "busy", {}, io.BytesIO(b""))
    with patch("submit_aou.urllib.request.urlopen", side_effect=[busy, response({"ok": 1})]), \
         patch("submit_aou.time.sleep"):
        assert wb.storage("POST", "https://storage", "token", b"x") == {"ok": 1}


def test_list_objects_follows_every_page():
    pages = [response({"items": [{"name": "a"}], "nextPageToken": "next"}), response({"items": [{"name": "b"}]})]
    with patch.object(Workbench, "assert_identity"), \
         patch.object(Workbench, "command", return_value="token"), \
         patch("submit_aou.urllib.request.urlopen", side_effect=pages) as urlopen:
        assert workbench().list_objects("gs://bucket/artifacts/cache/") == [{"name": "a"}, {"name": "b"}]
    second = urllib.parse.parse_qs(urllib.parse.urlsplit(urlopen.call_args_list[1].args[0].full_url).query)
    assert second["pageToken"] == ["next"] and second["prefix"] == ["artifacts/cache/"]


class FakeWorkbenchCli:
    """Plays `wb workflow` for run_workflow: each command pops its next scripted result."""

    def __init__(self, **script):
        self.script = {key: list(values) for key, values in script.items()}
        self.calls = []

    def __call__(self, command, timeout=None):
        verb = " ".join(command[1:4]) if command[1:3] == ["workflow", "job"] else " ".join(command[1:3])
        self.calls.append(verb)
        result = self.script[verb].pop(0)
        if isinstance(result, Exception):
            raise result
        return result if isinstance(result, str) else json.dumps(result)


def run(tmp_path, cli):
    with patch.object(Workbench, "assert_identity"), \
         patch.object(Workbench, "command", side_effect=cli), \
         patch("submit_aou.time.sleep"):
        return workbench().run_workflow("aou-run-1", "gs://bucket/workflows/aou-run-1/flow.wdl",
                                        "Flow", {"flow.x": 1}, tmp_path)


def test_create_that_timed_out_after_creating_the_workflow_still_starts_one_run(tmp_path):
    receipt = {"runId": "r1", "displayName": "aou-run-1", "status": "PENDING"}
    cli = FakeWorkbenchCli(**{"workflow create": [WorkbenchTimeout("hung")],
                              "workflow describe": [RuntimeError("not yet"), {"id": "aou-run-1"}],
                              "workflow job list": ["[]"], "workflow job run": [receipt]})
    assert run(tmp_path, cli) == receipt
    assert cli.calls == ["workflow create", "workflow describe", "workflow describe",
                         "workflow job list", "workflow job run"]
    assert json.loads((tmp_path / "submission.json").read_text()) == receipt
    assert json.loads((tmp_path / "inputs.json").read_text()) == {"flow.x": 1}


def test_run_created_by_a_timed_out_submission_is_recorded_not_resubmitted(tmp_path):
    listed = {"runId": "r1", "displayName": "aou-run-1", "status": "RUNNING"}
    cli = FakeWorkbenchCli(**{"workflow create": ["{}"],
                              "workflow job list": ["[]", "[]", [listed]],
                              "workflow job run": [WorkbenchTimeout("hung after creating the run")]})
    assert run(tmp_path, cli) == listed
    assert cli.calls.count("workflow job run") == 1
    assert json.loads((tmp_path / "submission.json").read_text()) == listed


def test_finishing_a_folder_whose_run_exists_never_submits_another(tmp_path):
    listed = {"runId": "r1", "displayName": "aou-run-1", "status": "RUNNING"}
    cli = FakeWorkbenchCli(**{"workflow create": [RuntimeError("workflow already exists")],
                              "workflow describe": [{"id": "aou-run-1"}], "workflow job list": [[listed]]})
    assert run(tmp_path, cli) == listed
    assert "workflow job run" not in cli.calls


def test_a_run_that_never_appears_fails_loudly_after_polling(tmp_path):
    cli = FakeWorkbenchCli(**{"workflow create": ["{}"], "workflow job list": ["[]"] * 9,
                              "workflow job run": [RuntimeError("engine refused the inputs")]})
    with pytest.raises(RuntimeError, match="no run is listed"):
        run(tmp_path, cli)
    assert cli.calls.count("workflow job run") == 1
    assert cli.calls.count("workflow job list") == 9
    assert not (tmp_path / "submission.json").exists()


def test_a_second_listed_run_stops_submission(tmp_path):
    runs = [{"runId": "r1"}, {"runId": "r2"}]
    cli = FakeWorkbenchCli(**{"workflow create": ["{}"], "workflow job list": [runs]})
    with pytest.raises(RuntimeError, match="already has 2 runs"):
        run(tmp_path, cli)


def test_command_timeout_ends_the_cli_and_everything_it_started(tmp_path):
    pid_file = tmp_path / "grandchild.pid"
    code = ("import subprocess, sys, time; "
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            f"open({str(pid_file)!r}, 'w').write(str(child.pid)); time.sleep(60)")
    wb = workbench()
    grandchild = None
    try:
        started = time.monotonic()
        with pytest.raises(WorkbenchTimeout):
            wb.command([sys.executable, "-c", code], timeout=3)
        assert time.monotonic() - started < 10
        grandchild = int(pid_file.read_text())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            # A killed grandchild is reparented and reaped by init; wait for it.
            time.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild, 0)
    finally:
        if grandchild is not None and grandchild > 1:
            try:
                os.kill(grandchild, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_newest_cached_score_per_pgs_from_one_listing():
    cache = "artifacts/aou-training/sscore_cache"
    objects = [{"name": f"{cache}/old/PGS000014.sscore", "updated": "2026-09-10T01:00:00.000Z"},
               {"name": f"{cache}/new/PGS000014.sscore", "updated": "2026-09-12T01:00:00.000Z"},
               {"name": f"{cache}/new/PGS005110.sscore", "updated": "2026-09-12T01:00:00.000Z"},
               {"name": f"{cache}/new/nested/PGS005110.sscore", "updated": "2026-09-13T01:00:00.000Z"},
               {"name": f"{cache}/new/PGS999999.sscore", "updated": "2026-09-14T01:00:00.000Z"}]
    assert submit_runs.newest_cached_scores(objects, cache, {"PGS005110", "PGS000014"}) == [
        f"{cache}/new/PGS000014.sscore", f"{cache}/new/PGS005110.sscore"]
    with pytest.raises(ValueError, match="PGS000331 has no cached score"):
        submit_runs.newest_cached_scores(objects, cache, {"PGS000014", "PGS000331"})


def test_finish_restarts_from_the_staged_folder(tmp_path):
    folder = tmp_path / "aou-run-1"
    folder.mkdir()
    (folder / "workflow.json").write_text(json.dumps(dict(
        name="aou-run-1", wdl="gs://bucket/workflows/aou-run-1/flow.wdl", display_name="Flow",
        storage_capacity=50)))
    (folder / "inputs.json").write_text(json.dumps({"flow.x": 1}))
    wb = MagicMock()
    submit_runs.finish(wb, MagicMock(folder=folder))
    wb.run_workflow.assert_called_once_with("aou-run-1", "gs://bucket/workflows/aou-run-1/flow.wdl", "Flow",
                                            {"flow.x": 1}, folder.resolve(), storage_capacity=50)


def test_no_submitter_stages_through_gsutil():
    for path in (Path(submit_aou.__file__), Path(submit_runs.__file__)):
        assert '"gsutil", "cp"' not in path.read_text()
