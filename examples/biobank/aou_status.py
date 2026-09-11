"""Fixed software/status labels in workspace storage; never copy exception text."""
from __future__ import annotations

import json
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

from aou_identity import task_account

LABELS = frozenset({
    "reading_ancestry", "loading_person_times", "preparing_cohort",
    "cohort_ready", "cohort_unsupported", "evaluation_unsupported", "scores_missing", "smoke_completed",
    "analysis_completed", "failed_ancestry_schema", "failed_prune_schema",
    "failed_query", "failed_timeout", "failed_other",
    "missing_pgs004536", "missing_pgs001783", "missing_pgs004525",
    "missing_pgs004603", "missing_pgs005199", "missing_pgs005331",
    "transforming_score", "applying_reference_ctn", "score_transform_ready", "fitting_disease", "fitting_death",
    "failed_score_missingness_absent", "failed_score_missingness_invalid",
    "failed_score_completely_missing", "failed_score_source_ambiguous",
})


def failure_label(error):
    message = str(error).lower()
    for phrase, label in (
        ("lacks per-participant missingness", "failed_score_missingness_absent"),
        ("cached score has invalid missingness percentages", "failed_score_missingness_invalid"),
        ("completely missing scores cannot enter ctn", "failed_score_completely_missing"),
        ("ambiguous cached score source", "failed_score_source_ambiguous"),
    ):
        if phrase in message:
            return label
    if "ancestry file lacks" in message:
        return "failed_ancestry_schema"
    if "relatedness prune" in message:
        return "failed_prune_schema"
    if type(error).__module__.startswith("google.api_core.exceptions"):
        return "failed_query"
    if isinstance(error, TimeoutError) or "timed out" in message:
        return "failed_timeout"
    return "failed_other"


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
