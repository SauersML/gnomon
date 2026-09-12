"""Read the frozen PC table without localizing the unrelated score bank."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
from urllib.parse import quote, urlsplit

from aou_checkpoint import file_hash


def object_url(uri):
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
        raise ValueError("projection source must be a workspace gs:// object")
    return (f"https://storage.googleapis.com/storage/v1/b/{quote(parsed.netloc, safe='')}/o/"
            f"{quote(parsed.path.lstrip('/'), safe='')}")


def source_identity(uri, project, account):
    from google.auth.compute_engine import Credentials
    from google.auth.transport.requests import AuthorizedSession
    with AuthorizedSession(Credentials(service_account_email=account)) as session:
        with session.get(object_url(uri), params={"userProject": project,
                         "fields": "generation,size,crc32c"}, timeout=30) as response:
            response.raise_for_status()
            metadata = response.json()
    if not str(metadata["generation"]).isdigit() or int(metadata["size"]) <= 0 or not metadata["crc32c"]:
        raise ValueError("projection archive lacks immutable object metadata")
    return {"uri": uri, **metadata}


class PrefixReader:
    """Bound compressed input even if an archive places a large member first."""
    def __init__(self, stream):
        self.stream = stream
        self.remaining = 256 * 1024**2

    def read(self, size=-1):
        amount = self.remaining + 1 if size < 0 else min(size, self.remaining + 1)
        block = self.stream.read(amount)
        self.remaining -= len(block)
        if self.remaining < 0:
            raise ValueError("projection exceeds the 256-MiB compressed-prefix budget")
        return block


def extract_projection(stream, output):
    """Select the leading PC member of pgsEngine's sorted feature archive.

    The source object generation fixes the archive. Deliberately stop at this
    member: reading the subsequent score bank adds no information to the PCs.
    The extracted bytes receive their own checksum and checkpoint receipt.
    """
    output = Path(output)
    temporary = output.with_suffix(".partial")
    manifest = None
    root = None
    try:
        with tarfile.open(fileobj=PrefixReader(stream), mode="r|gz") as archive:
            for member in archive:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not (member.isdir() or member.isfile()):
                    raise ValueError("projection source contains an unsafe archive member")
                if path.name == "scores.tar":
                    raise ValueError("projection must precede the score bank in the shared-feature archive")
                if not member.isfile():
                    continue
                if member.size > 512 * 1024**2:
                    raise ValueError("projection prefix contains an oversized member")
                if path.name == "manifest.json":
                    if manifest is not None or member.size > 1024**2:
                        raise ValueError("projection source has an ambiguous or oversized manifest")
                    manifest = json.load(archive.extractfile(member))
                    if manifest.get("schema_version") != 1 or not manifest.get("git_sha"):
                        raise ValueError("projection source has an unsupported manifest")
                    root = path.parent
                elif path.name == "projection_pcs.parquet":
                    if manifest is None or path.parent != root or member.size <= 0:
                        raise ValueError("projection member does not match its source manifest")
                    with archive.extractfile(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024**2)
                    if temporary.stat().st_size != member.size:
                        raise ValueError("projection member is truncated")
                    temporary.replace(output)
                    return {"member": str(path), "producer_revision": manifest["git_sha"],
                            "sha256": file_hash(output)}
        raise ValueError("shared-feature archive has no projection PC table")
    finally:
        temporary.unlink(missing_ok=True)


def stream_projection(identity, output, project, account):
    from google.auth.compute_engine import Credentials
    from google.auth.transport.requests import AuthorizedSession
    with AuthorizedSession(Credentials(service_account_email=account)) as session:
        with session.get(object_url(identity["uri"]), params={"alt": "media", "userProject": project,
                         "generation": identity["generation"]}, stream=True, timeout=30) as response:
            response.raise_for_status()
            receipt = extract_projection(response.raw, output)
    return {"source": identity, **receipt}
