#!/usr/bin/env python3
"""Submit one bounded WDL run using a locally configured Workbench context.

Deployment values are required environment variables, never example strings.
Use --check to verify identities, workspace bindings and input URI syntax.
Input objects are read by the workspace task, never by a local read preflight.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from aou_identity import check_account

HERE = Path(__file__).resolve().parent
STORAGE = "https://storage.googleapis.com"


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is unset")
    return value


def split_uri(uri):
    match = re.fullmatch(r"gs://([a-z0-9][a-z0-9._-]+)/(.+)", uri)
    if not match:
        raise ValueError(f"{uri} does not name a gs:// object")
    return match.group(1), match.group(2)


def md5_base64(body):
    return base64.b64encode(hashlib.md5(body).digest()).decode()


class WorkbenchTimeout(RuntimeError):
    """A Workbench command outlived its bound; what it did remotely is unknown."""


class Workbench:
    def __init__(self):
        self.workspace = required_env("AOU_WORKSPACE_ID")
        self.project = required_env("GOOGLE_PROJECT")
        self.bucket_id = required_env("AOU_BUCKET_ID")
        self.bucket = required_env("WORKSPACE_BUCKET").rstrip("/")
        if not re.fullmatch(r"gs://[a-z0-9][a-z0-9.-]+", self.bucket):
            raise ValueError("WORKSPACE_BUCKET must name a bucket, without an object prefix")
        self.expected_account = check_account(required_env("AOU_EXPECTED_ACCOUNT"))
        self.profile = required_env("AOU_GCLOUD_CONFIGURATION")
        self.env = dict(os.environ)
        self.env["WORKBENCH_CONTEXT_PARENT_DIR"] = str(Path(
            required_env("WORKBENCH_CONTEXT_PARENT_DIR")).expanduser())
        self.env["CLOUDSDK_ACTIVE_CONFIG_NAME"] = self.profile
        self.assert_identity()

    def command(self, command, timeout=None):
        # Workbench acknowledges a submission after creating its engine run;
        # killing that handshake early can leave a real job without its receipt,
        # which is why run_workflow confirms a failed step by listing.
        if timeout is None:
            timeout = 180 if command[:4] == ["wb", "workflow", "job", "run"] else 60
        # In its own session, a timeout ends the CLI and everything it started,
        # so no orphan can finish a submission after the caller gave up on it.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, env=self.env, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise WorkbenchTimeout(f"{' '.join(command[:4])} did not finish within {timeout} s") from None
        if process.returncode:
            raise RuntimeError(stderr.strip() or stdout.strip()
                               or f"Workbench command exited {process.returncode}")
        return stdout

    def assert_identity(self):
        active = self.command(["gcloud", f"--configuration={self.profile}", "auth", "list",
                               "--filter=status:ACTIVE", "--format=value(account)"]).strip()
        check_account(active, expected=self.expected_account)
        status = json.loads(self.command(["wb", "status", "--format=JSON"]))
        check_account(status.get("user", {}).get("email"), expected=self.expected_account)
        workspace = status.get("workspace", {})
        if workspace.get("id") != self.workspace or workspace.get("googleProjectId") != self.project:
            raise RuntimeError("Workbench workspace/project do not match the requested environment")

    def wb(self, *args):
        self.assert_identity()  # includes every create and submission
        if args[0] == "gsutil":
            args = ("gsutil", "-u", self.project, *args[1:])
        return self.command(["wb", *args])

    def access_token(self):
        return self.command(["gcloud", "auth", "application-default", "print-access-token"]).strip()

    def storage(self, method, url, token, body=None):
        """One Cloud Storage JSON API request; None when the object does not exist."""
        request = urllib.request.Request(url, data=body, method=method, headers={
            "Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if error.code < 500 or attempt == 2:
                    raise RuntimeError(f"Cloud Storage {method} returned HTTP {error.code}") from error
            except OSError:
                if attempt == 2:
                    raise
            time.sleep(2 ** attempt)

    def stage_as(self, path, uri, token=None, reuse=False):
        """Upload one file with a single media POST and verify its MD5 and size.

        `wb gsutil cp` stalled indefinitely on a 12 MB archive that one POST
        uploaded in 18 s. With `reuse`, an object whose MD5 already matches is
        left in place."""
        bucket, name = split_uri(uri)
        if not uri.startswith(self.bucket + "/"):
            raise ValueError("objects are staged only in the workspace bucket")
        if token is None:
            self.assert_identity()
            token = self.access_token()
        body = Path(path).read_bytes()
        quoted = f"{STORAGE}/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}"
        project = urllib.parse.quote(self.project, safe="")
        if reuse:
            meta = self.storage("GET", f"{quoted}?userProject={project}", token)
            if meta and meta.get("md5Hash") == md5_base64(body):
                return uri
        meta = self.storage("POST", f"{STORAGE}/upload/storage/v1/b/{bucket}/o?uploadType=media"
                            f"&name={urllib.parse.quote(name, safe='')}&userProject={project}",
                            token, body)
        if (not meta or meta.get("name") != name or meta.get("md5Hash") != md5_base64(body)
                or str(meta.get("size")) != str(len(body))):
            raise RuntimeError(f"staged {uri} does not match {Path(path).name}")
        return uri

    def stage(self, paths, prefix):
        """Stage files under a gs:// prefix by name; returns their URIs in order."""
        self.assert_identity()
        token = self.access_token()
        return [self.stage_as(path, f"{prefix.rstrip('/')}/{Path(path).name}", token)
                for path in paths]

    def list_objects(self, prefix):
        """Every object under a gs:// prefix, as JSON API records (name, size, updated).

        A gsutil wildcard over the workspace score cache took over 180 s; one
        listing request covers a thousand objects."""
        bucket, name = split_uri(prefix)
        self.assert_identity()
        token = self.access_token()
        items, page = [], None
        while True:
            query = {"prefix": name, "fields": "items(name,size,updated),nextPageToken",
                     "userProject": self.project}
            if page:
                query["pageToken"] = page
            listing = self.storage("GET", f"{STORAGE}/storage/v1/b/{bucket}/o?"
                                   + urllib.parse.urlencode(query), token) or {}
            items += listing.get("items", [])
            page = listing.get("nextPageToken")
            if not page:
                return items

    def describe_workflow(self, name):
        return json.loads(self.command(["wb", "workflow", "describe", f"--workflow={name}",
                                        f"--workspace={self.workspace}", "--format=JSON"]))

    def listed_run(self, name):
        """The run already started for workflow `name`, or None."""
        runs = json.loads(self.command(["wb", "workflow", "job", "list", f"--workflow={name}",
                                        "--limit=10", f"--workspace={self.workspace}",
                                        "--format=JSON"]) or "[]")
        if len(runs) > 1:
            raise RuntimeError(f"workflow {name} already has {len(runs)} runs")
        return runs[0] if runs else None

    @staticmethod
    def confirmed(check, attempts=8, interval=15.):
        """Poll a read-only check until it finds something; Workbench lags its own writes."""
        for attempt in range(attempts):
            try:
                found = check()
            except (RuntimeError, ValueError):
                found = None
            if found:
                return found
            if attempt + 1 < attempts:
                time.sleep(interval)
        return None

    def run_workflow(self, name, wdl_uri, display_name, inputs, folder, storage_capacity=10):
        """Create workflow `name` from a staged WDL and start one run of it.

        Workbench has hung after creating a workflow and after creating a run,
        so neither step is judged by its exit: a failed create is accepted once
        the workflow can be described, and a run is started only while the
        workflow lists none. Two runs of one submission would race on the same
        checkpoint. Calling this again for a folder whose run exists only
        records that run. Returns the run record saved as submission.json."""
        folder = Path(folder)
        if not wdl_uri.startswith(self.bucket + "/"):
            raise ValueError("the workflow definition must be staged in the workspace bucket")
        (folder / "workflow.json").write_text(json.dumps(dict(
            name=name, wdl=wdl_uri, display_name=display_name,
            storage_capacity=storage_capacity), indent=2))
        (folder / "inputs.json").write_text(json.dumps(inputs, indent=2))
        try:
            self.wb("workflow", "create", f"--bucket-id={self.bucket_id}",
                    f"--path={wdl_uri[len(self.bucket) + 1:]}", f"--workflow={name}",
                    "--workflow-type=WDL", f"--display-name={display_name}",
                    f"--workspace={self.workspace}", "--format=JSON")
        except RuntimeError as error:
            if not self.confirmed(lambda: self.describe_workflow(name)):
                raise
            print(f"Workflow {name} exists; create reported: {str(error).splitlines()[0][:200]}",
                  flush=True)
        record = self.listed_run(name)
        if record is None:
            try:
                record = json.loads(self.wb(
                    "workflow", "job", "run", f"--workflow={name}", f"--job-id={name}",
                    f"--output-bucket-id={self.bucket_id}", f"--output-path=workflow-runs/{name}",
                    "--storage-type=STATIC", f"--storage-capacity={storage_capacity}",
                    f"--inputs={json.dumps(inputs)}", f"--workspace={self.workspace}",
                    "--format=JSON"))
            except (RuntimeError, ValueError) as error:
                record = self.confirmed(lambda: self.listed_run(name))
                if record is None:
                    raise RuntimeError(f"no run is listed for workflow {name} after the run step "
                                       f"failed; finishing this folder later starts it once") from error
        else:
            print(f"Workflow {name} already has run {record.get('runId')}; not submitting another",
                  flush=True)
        (folder / "submission.json").write_text(json.dumps(record, indent=2))
        print(json.dumps({key: record.get(key) for key in ("runId", "displayName", "status")}),
              flush=True)
        return record


def prepare_inputs():
    fields = {
        "runtime_image": "AOU_RUNTIME_IMAGE",
        "wheelhouse_archive": "AOU_WHEELHOUSE_URI",
        "phenotype_library_archive": "AOU_PHENOTYPE_LIBRARY_URI",
        "shared_features_archive": "AOU_SHARED_FEATURES_URI",
        "ancestry_predictions": "AOU_ANCESTRY_URI",
        "relatedness_prune": "AOU_RELATEDNESS_PRUNE_URI",
    }
    inputs = {f"aou_survival.{key}": required_env(env) for key, env in fields.items()}
    # Reference CTN archives exist only for an analysis that declares a reference CTN score.
    inputs["aou_survival.reference_ctn"] = os.environ.get("AOU_REFERENCE_CTN_URIS", "").split()
    image = inputs["aou_survival.runtime_image"]
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("AOU_RUNTIME_IMAGE must be pinned by digest")
    for key, value in inputs.items():
        if not key.endswith("runtime_image"):
            for uri in value if isinstance(value, list) else [value]:
                if not re.fullmatch(r"gs://[^\s]+/.+", uri):
                    raise ValueError(f"{key} must name staged gs:// objects")
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, default=HERE / "aou_analysis.json")
    parser.add_argument("--check", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--smoke-only", action="store_true",
                      help="Fit the first prespecified score on development data only")
    parser.add_argument("--endpoint", choices=["copd", "hypertension", "obesity"])
    parser.add_argument("--resume", help="Workspace gs:// checkpoint object from a previous run")
    args = parser.parse_args()
    wb = Workbench()
    inputs = prepare_inputs()
    if args.resume:
        if not args.resume.startswith(wb.bucket + "/"):
            raise ValueError("resume checkpoint must be in the selected workspace bucket")
        inputs["aou_survival.resume_checkpoint"] = args.resume
    config = json.loads(args.analysis.read_text())
    if "google_project" in config or "workspace_cdr" in config:
        raise ValueError("deployment project/CDR must come from the environment")
    config["google_project"] = wb.project
    config["workspace_cdr"] = required_env("WORKSPACE_CDR")
    # Resource resolution is checked against the selected workspace, rather
    # than trusting a bucket URL copied from a different workspace.
    resolved = wb.wb("resource", "resolve", f"--id={wb.bucket_id}",
                     f"--workspace={wb.workspace}").strip().rstrip("/")
    if resolved != wb.bucket:
        raise RuntimeError("WORKSPACE_BUCKET does not match AOU_BUCKET_ID in this workspace")
    cdr_resource = required_env("AOU_CDR_RESOURCE_ID")
    resolved_cdr = wb.wb("resource", "resolve", f"--id={cdr_resource}",
                         f"--workspace={wb.workspace}").strip()
    if resolved_cdr.removeprefix("bq://").replace(":", ".") != config["workspace_cdr"]:
        raise RuntimeError("WORKSPACE_CDR does not match the workspace CDR resource")
    if args.check:
        print("Verified configured identities, workspace resources, and input URI syntax.")
        return
    sources = {"runner": HERE / "aou_survival.py",
               "disease_selector": HERE / "disease_selection.py",
               "identity_guard": HERE / "aou_identity.py",
               "status_code": HERE / "aou_status.py",
               "score_transform": HERE / "aou_score_transform.py",
               "reference_code": HERE / "reference_ctn.py",
               "checkpoint_code": HERE / "aou_checkpoint.py",
               "evaluation_code": HERE / "aou_evaluation.py",
               "score_panel": HERE / "aou_pgs_panel.json",
               "requirements": HERE / "aou_requirements.txt"}
    encoded = (json.dumps(config, sort_keys=True, indent=2) + "\n").encode()
    checksum = hashlib.sha256(encoded + (HERE / "aou_survival.wdl").read_bytes())
    for path in sources.values():
        checksum.update(path.read_bytes())
    run_id = f"aou-survival-{checksum.hexdigest()[:12]}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
    folder = HERE / ".aou-workflow" / run_id
    folder.mkdir(parents=True)
    config_path = folder / f"{run_id}.json"
    config_path.write_bytes(encoded)
    sources["analysis_config"] = config_path
    prefix = f"{wb.bucket}/workflows/{run_id}"
    inputs["aou_survival.checkpoint_uri"] = f"{wb.bucket}/workflow-checkpoints/{run_id}.tar.gz"
    inputs["aou_survival.prepare_only"] = args.prepare_only
    inputs["aou_survival.smoke_only"] = args.smoke_only
    inputs["aou_survival.endpoint"] = args.endpoint or ""
    wb.stage([*sources.values(), HERE / "aou_survival.wdl"], prefix)
    for name, path in sources.items():
        inputs[f"aou_survival.{name}"] = f"{prefix}/{path.name}"
    wb.run_workflow(run_id, f"{prefix}/aou_survival.wdl", "AoU PC-varying survival pilot",
                    inputs, folder, storage_capacity=50)


if __name__ == "__main__":
    main()
