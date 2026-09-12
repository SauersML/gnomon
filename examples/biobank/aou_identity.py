"""Account checks shared by submission and the in-workspace task."""
from __future__ import annotations

import os
from pathlib import Path
import urllib.request


class RuntimePolicyError(RuntimeError):
    def __init__(self, label, message):
        super().__init__(message)
        self.label = label


def require_spot_amd(required_cpu_flags):
    """Check the actual VM before setup, scoring, or fitting can spend resources."""
    def metadata(path):
        request = urllib.request.Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/" + path,
            headers={"Metadata-Flavor": "Google"})
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode().strip()
    if not isinstance(required_cpu_flags, list) or not all(
            isinstance(flag, str) and flag and flag.replace("_", "").isalnum()
            for flag in required_cpu_flags):
        raise ValueError("scorer CPU requirements must be Linux CPU flag names")
    preemptible = metadata("scheduling/preemptible")
    machine = metadata("machine-type").rsplit("/", 1)[-1]
    cpuinfo = Path("/proc/cpuinfo").read_text()
    if preemptible.upper() != "TRUE":
        raise RuntimePolicyError("failed_runtime_nonspot", "this pilot requires an AMD Spot/preemptible VM")
    if "AuthenticAMD" not in cpuinfo:
        raise RuntimePolicyError("failed_runtime_nonamd", "this pilot requires an AMD Spot/preemptible VM")
    flag_sets = [set(line.split(":", 1)[1].split()) for line in cpuinfo.splitlines()
                 if line.split(":", 1)[0].strip() == "flags"]
    if not flag_sets or any(not set(required_cpu_flags).issubset(flags) for flags in flag_sets):
        raise RuntimePolicyError("failed_runtime_instructions", "VM CPU does not support the pinned native scorer instruction set")
    return {"machine_type": machine, "preemptible": True, "cpu_vendor": "AMD",
            "verified_cpu_flags": sorted(required_cpu_flags)}

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
