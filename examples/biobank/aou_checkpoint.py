"""Workspace-only checkpoints for bounded analysis steps, including failed runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tarfile
from urllib.parse import quote, urlsplit


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class StudyCheckpoint:
    def __init__(self, root, uri, project, account, signature, resume=None, engine_hash=None):
        self.root = Path(root)
        parsed = urlsplit(uri)
        if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
            raise ValueError("checkpoint URI must name a workspace gs:// object")
        self.bucket, self.object = parsed.netloc, parsed.path.lstrip("/")
        self.project, self.account = project, account
        self.engine_hash = engine_hash
        self.archive = self.root.parent / "checkpoint.tar.gz"
        self.root.mkdir(parents=True, exist_ok=True)
        if resume is not None:
            self.restore(resume, signature)
        (self.root / "checkpoint_manifest.json").write_text(json.dumps(signature, sort_keys=True))

    def restore(self, archive, signature):
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            if sum(m.size for m in members) > 2 * 1024**3:
                raise ValueError("checkpoint exceeds the two-GiB expanded budget")
            for member in members:
                path = Path(member.name)
                if not member.isfile() or path.is_absolute() or ".." in path.parts:
                    raise ValueError("checkpoint must contain only relative regular files")
            manifests = [m for m in members if m.name == "checkpoint_manifest.json"]
            if len(manifests) != 1 or json.load(source.extractfile(manifests[0])) != signature:
                raise ValueError("checkpoint inputs, configuration, or analysis code do not match")
            source.extractall(self.root, members=members, filter="data")

    def step_is_complete(self, directory, *, model=False):
        directory = Path(directory)
        receipt = directory / "completed.json"
        if not receipt.exists():
            return False
        recorded = json.loads(receipt.read_text())
        if not recorded["files"]:
            raise ValueError("empty completed-step receipt")
        if recorded["engine_hash"] != (self.engine_hash if model else None):
            raise ValueError("native engine differs from the completed model checkpoint")
        for name, digest in recorded["files"].items():
            if Path(name).name != name or not (directory / name).is_file() or file_hash(directory / name) != digest:
                raise ValueError("completed checkpoint artifact is missing or corrupt")
        return True

    def complete_step(self, directory, filenames, *, model=False):
        directory = Path(directory)
        receipt = {"files": {name: file_hash(directory / name) for name in filenames},
                   "engine_hash": self.engine_hash if model else None}
        (directory / "completed.json").write_text(json.dumps(receipt, sort_keys=True))
        self.publish()

    def publish(self):
        # Large localized score/phenotype sources are independently supplied as
        # WDL inputs. Persist prepared cohorts, fits, folds, warm starts and logs.
        with tarfile.open(self.archive, "w:gz", compresslevel=1) as archive:
            for path in sorted(self.root.rglob("*")):
                relative = path.relative_to(self.root)
                if path.is_symlink():
                    raise ValueError("checkpoint cannot contain symlinks")
                if path.is_file() and relative.parts[0] not in {"phenotypes", "scores.tar"}:
                    archive.add(path, arcname=str(relative), recursive=False)
        # Explicit VM credentials: never consult local gcloud/ADC credential
        # files, which could name a different identity from the metadata check.
        from google.auth.compute_engine import Credentials
        from google.auth.transport.requests import AuthorizedSession
        session = AuthorizedSession(Credentials(service_account_email=self.account))
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{quote(self.bucket, safe='')}/o"
        with self.archive.open("rb") as body:
            response = session.post(url, params={"uploadType": "media", "name": self.object,
                                                "userProject": self.project},
                                    data=body, headers={"Content-Type": "application/gzip"}, timeout=60)
        response.raise_for_status()
        session.close()
