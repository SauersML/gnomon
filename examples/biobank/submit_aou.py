#!/usr/bin/env python3
"""Submit one bounded WDL run using a locally configured Workbench context.

Deployment values are required environment variables, never example strings.
Use --check to verify identities, workspace bindings and input URI syntax.
Input objects are read by the workspace task, never by a local read preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from aou_identity import check_account

HERE = Path(__file__).resolve().parent


def required_env(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable {name} is unset")
    return value


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

    def command(self, command):
        # Workbench acknowledges a submission after creating its engine run;
        # killing that handshake early can leave a real job without its receipt.
        timeout = 180 if command[:4] == ["wb", "workflow", "job", "run"] else 60
        try:
            return subprocess.run(command, check=True, text=True, capture_output=True,
                                  env=self.env, timeout=timeout).stdout
        except subprocess.CalledProcessError as error:
            raise RuntimeError(error.stderr.strip() or error.stdout.strip()
                               or f"Workbench command exited {error.returncode}") from error

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
        self.assert_identity()  # includes every upload, create, and submission
        if args[0] == "gsutil":
            args = ("gsutil", "-u", self.project, *args[1:])
        return self.command(["wb", *args])


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
    inputs["aou_survival.reference_ctn"] = required_env("AOU_REFERENCE_CTN_URIS").split()
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
    staging = HERE / ".aou-workflow"
    staging.mkdir(exist_ok=True)
    config_path = staging / f"{run_id}.json"
    config_path.write_bytes(encoded)
    sources["analysis_config"] = config_path
    prefix = f"workflows/{run_id}"
    inputs["aou_survival.checkpoint_uri"] = f"{wb.bucket}/workflow-checkpoints/{run_id}.tar.gz"
    inputs["aou_survival.prepare_only"] = args.prepare_only
    inputs["aou_survival.smoke_only"] = args.smoke_only
    inputs["aou_survival.endpoint"] = args.endpoint or ""
    wb.wb("gsutil", "cp", *[str(path) for path in sources.values()],
          str(HERE / "aou_survival.wdl"), f"{wb.bucket}/{prefix}/")
    for name, path in sources.items():
        inputs[f"aou_survival.{name}"] = f"{wb.bucket}/{prefix}/{path.name}"
    wb.wb("workflow", "create", f"--bucket-id={wb.bucket_id}",
          f"--path={prefix}/aou_survival.wdl", f"--workflow={run_id}",
          "--workflow-type=WDL", "--display-name=AoU PC-varying survival pilot",
          f"--workspace={wb.workspace}", "--format=JSON")
    result = wb.wb("workflow", "job", "run", f"--workflow={run_id}",
                   f"--job-id={run_id}", f"--output-bucket-id={wb.bucket_id}",
                   f"--output-path=workflow-runs/{run_id}", "--storage-type=STATIC",
                   "--storage-capacity=50", f"--inputs={json.dumps(inputs)}",
                   f"--workspace={wb.workspace}", "--format=JSON")
    (staging / f"{run_id}.submission.json").write_text(result)
    print(result)


if __name__ == "__main__":
    main()
