"""Account checks shared by submission and the in-workspace task."""
from __future__ import annotations

import os
import urllib.request

def check_account(email, *, expected=None):
    if not isinstance(email, str) or "@" not in email or email != email.strip():
        raise RuntimeError("cannot establish the execution account")
    normalized = email.lower()
    if "user" in normalized:
        raise RuntimeError("refusing an execution account whose email contains 'user'")
    if expected is not None and normalized != expected.lower():
        raise RuntimeError(f"execution account must be {expected}")
    return normalized


def task_account():
    # WDL uses the workspace VM service account. Credential-file overrides could
    # change BigQuery's identity independently from that account; reject them.
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("credential-file overrides are not allowed in the AoU task")
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        email = check_account(response.read().decode().strip())
    if not email.endswith(".gserviceaccount.com"):
        raise RuntimeError("AoU task requires the workspace VM service account")
    return email
