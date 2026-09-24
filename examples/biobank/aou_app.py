#!/usr/bin/env python3
"""The workspace app VM as the study's in-perimeter orchestrator.

Google Batch answers only from inside the workspace perimeter, and the workspace
policy lets no new VM be created from the command line. What the command line
can do is set the startup script of the one app VM the workspace has, through
the Workbench resource API, and start or stop that app. The startup script
(orchestrator/startup.sh) fetches orchestrator/orch.sh from the bucket and runs
it on the app's host: a loop that runs every orchestrator/cmd/<id>.sh the
submitter drops in the bucket, as the app's pet service account, and reports
through object names (orchestrator/out/<id>/...), since objects cannot be read
from outside.

  aou_app.py install [--app ID]     stage the orchestrator, set the startup script, restart the app
  aou_app.py describe [--app ID]    the app resource as the Workbench API reports it
"""
from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKBENCH = "https://workbench.verily.com/api/wsm"
DEFAULT_APP = "AoU_Jupyter_ComputeEngine_20260924_1"
ORCHESTRATOR = "orchestrator"


def wb(*args, timeout=180):
    done = subprocess.run(["wb", *args], capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"wb {' '.join(args)} failed: {done.stderr.strip()[-400:] or done.stdout.strip()[-400:]}")
    return done.stdout


def wb_json(*args):
    return json.loads(wb(*args, "--format=JSON"))


def token():
    value = wb("auth", "print-access-token").strip()
    if not value:
        raise RuntimeError("wb gave no access token")
    return value


def request(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token()}", "Accept": "application/json", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            text = response.read().decode()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{method} {url.rsplit('/', 1)[-1]}: HTTP {error.code} {error.read().decode()[:300]}") from error
    return json.loads(text) if text else {}


class App:
    def __init__(self, app_id):
        workspace = wb_json("workspace", "describe")
        self.workspace_uuid, self.project = workspace["uuid"], workspace["googleProjectId"]
        resource = wb_json("resource", "describe", f"--id={app_id}")
        if resource.get("resourceType") != "GCE_INSTANCE":
            raise RuntimeError(f"{app_id} is not a GCE app")
        self.app_id, self.resource_uuid = app_id, resource["uuid"]
        self.url = (f"{WORKBENCH}/api/workspaces/v1/{self.workspace_uuid}/resources/controlled/gcp/"
                    f"gce-instances/{self.resource_uuid}")

    def describe(self):
        return request("GET", self.url)

    def status(self):
        for row in wb_json("app", "list"):
            if row.get("id") == self.app_id:
                return row.get("status")
        raise RuntimeError(f"{self.app_id} is not listed")

    def set_startup_script(self, script):
        return request("PATCH", self.url, {"updateParameters": {"metadata": {"startup-script": script}}})

    def restart(self):
        if self.status() != "TERMINATED":
            wb("app", "stop", f"--id={self.app_id}", timeout=600)
        wb("app", "start", f"--id={self.app_id}", timeout=900)
        return self.status()


def stage_orchestrator(bucket, project):
    """orch.sh to the bucket, by one media POST each, verified by size."""
    import base64
    import hashlib
    import urllib.parse
    body = (HERE / ORCHESTRATOR / "orch.sh").read_bytes()
    name = f"{ORCHESTRATOR}/orch.sh"
    url = (f"https://storage.googleapis.com/upload/storage/v1/b/{bucket}/o?uploadType=media"
           f"&name={urllib.parse.quote(name, safe='')}&userProject={project}")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Authorization": f"Bearer {token()}", "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=120) as response:
        meta = json.load(response)
    if meta.get("md5Hash") != base64.b64encode(hashlib.md5(body).digest()).decode():
        raise RuntimeError("orch.sh did not stage intact")
    return f"gs://{bucket}/{name}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["install", "describe"])
    parser.add_argument("--app", default=DEFAULT_APP)
    parser.add_argument("--bucket", default="aou-train-work-wb-amiable-carrot-1173")
    args = parser.parse_args()
    app = App(args.app)
    if args.command == "describe":
        print(json.dumps(app.describe(), indent=1)[:3000])
        return
    staged = stage_orchestrator(args.bucket, app.project)
    script = (HERE / ORCHESTRATOR / "startup.sh").read_text()
    if staged.rsplit("/", 1)[0] not in script:
        raise RuntimeError("startup.sh does not fetch the staged orchestrator")
    app.set_startup_script(script)
    print(f"startup script set on {args.app}; orchestrator at {staged}; restarting", flush=True)
    print("app", app.restart(), flush=True)


if __name__ == "__main__":
    main()
