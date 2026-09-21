"""Fixed software/status labels in workspace storage; never copy exception text."""
from __future__ import annotations

import json
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from aou_identity import task_account

LABELS = frozenset({
    "task_started", "runtime_verified", "installing_dependencies", "dependencies_ready",
    "failed_runtime_policy", "failed_runtime_nonspot", "failed_runtime_nonamd",
    "failed_task_setup", "reading_projection", "projection_ready",
})
# study.py: one started/complete/failed label per stage, plus fit progress.
STUDY_STAGES = ("scores", "cohort", "features", "fits", "predict", "evaluate", "digest")
LABELS = LABELS | {f"{prefix}{stage}{suffix}" for stage in STUDY_STAGES
                   for prefix, suffix in (("study_", "_started"), ("study_", "_complete"), ("failed_study_", ""))} \
    | {"study_started", "study_resumed", "study_completed", "study_fits_25", "study_fits_50", "study_fits_75",
       "failed_study_unexpected_errors"}


def publish_status(checkpoint_uri, label):
    if label not in LABELS:
        raise ValueError("status must be a fixed public label")
    uri = urlsplit(checkpoint_uri)
    if uri.scheme != "gs" or not uri.netloc or not uri.path.strip("/") or uri.query or uri.fragment:
        raise ValueError("status requires a workspace checkpoint URI")
    task_account()
    request = Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token",
        headers={"Metadata-Flavor": "Google"})
    with urlopen(request, timeout=10) as response:
        token = json.load(response)["access_token"]
    name = uri.path.lstrip("/") + ".status/" + label + ".txt"
    request = Request(
        f"https://storage.googleapis.com/upload/storage/v1/b/{quote(uri.netloc, safe='')}/o"
        f"?uploadType=media&name={quote(name, safe='')}",
        data=(label + "\n").encode(), method="POST",
        headers={"Authorization": "Bearer " + token, "Content-Type": "text/plain"})
    with urlopen(request, timeout=20) as response:
        response.read()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Publish a fixed workspace task status.")
    parser.add_argument("checkpoint_uri")
    parser.add_argument("label", choices=sorted(LABELS))
    args = parser.parse_args()
    publish_status(args.checkpoint_uri, args.label)
