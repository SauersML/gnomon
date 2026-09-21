"""What identifies a checkpointed result: its result-determining settings and file hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path


# Settings that bound cost, not results. Tuning a wall, query or byte budget
# must never discard checkpointed scoring or partially fitted models.
COMPUTE_BOUNDS = frozenset({"fit_timeout_seconds", "query_timeout_seconds",
                            "maximum_bytes_billed", "timeout_seconds", "fit_budget"})


def result_identity(settings):
    """The subset of a configuration that determines results."""
    return {key: value for key, value in settings.items() if key not in COMPUTE_BOUNDS}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
