"""Resumable steps for study.py: every completed step survives a Spot preemption.

A step is a directory under the work root holding a `completed.json` receipt
(the sha256 of each of its files). Completed steps are uploaded in batches, each
batch its own object under the checkpoint prefix, so the upload cost is linear
in the work done: republishing one growing archive after every one of hundreds
of fits would not be. Restoring replays the batches in order, and a receipt
whose files do not hash to it is a step that was never done.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import threading
import time
from urllib.parse import quote, urlsplit

from aou_checkpoint import file_hash, result_identity

RECEIPT = "completed.json"
MANIFEST = "manifest.json"
BATCH_PREFIX = "batch-"
DEPLOYMENT = frozenset({"google_project", "workspace_cdr"})
RESTORE_BUDGET = 16 * 1024**3


class LocalStore:
    """A directory standing in for the workspace bucket (MSI runs and tests)."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def names(self):
        return sorted(path.name for path in self.directory.iterdir()
                      if path.is_file() and not path.name.endswith(".partial"))

    def get(self, name, target):
        shutil.copyfile(self.directory / name, target)

    def put(self, name, source):
        partial = self.directory / (name + ".partial")
        shutil.copyfile(source, partial)
        partial.replace(self.directory / name)

    def put_bytes(self, name, body):
        partial = self.directory / (name + ".partial")
        partial.write_bytes(body)
        partial.replace(self.directory / name)

    def delete_all(self):
        shutil.rmtree(self.directory)


class GcsStore:
    """A gs:// prefix in the workspace bucket, written as the VM service account."""
    def __init__(self, uri, project, account):
        parsed = urlsplit(uri)
        if parsed.scheme != "gs" or not parsed.netloc or not parsed.path.strip("/") or parsed.query or parsed.fragment:
            raise ValueError("checkpoint URI must name a workspace gs:// prefix")
        self.bucket, self.prefix = parsed.netloc, parsed.path.strip("/") + "/"
        self.project, self.account = project, account
        self.local = threading.local()

    def session(self):
        # Explicit VM credentials: never local gcloud/ADC files, which could
        # name a different identity from the task account. One session per
        # thread, since the uploader runs beside the driver.
        if getattr(self.local, "session", None) is None:
            from google.auth.compute_engine import Credentials
            from google.auth.transport.requests import AuthorizedSession
            self.local.session = AuthorizedSession(Credentials(service_account_email=self.account))
        return self.local.session

    def objects(self):
        return f"https://storage.googleapis.com/storage/v1/b/{quote(self.bucket, safe='')}/o"

    def names(self):
        names, page = [], None
        while True:
            params = {"prefix": self.prefix, "fields": "items(name),nextPageToken", "userProject": self.project}
            if page:
                params["pageToken"] = page
            with self.session().get(self.objects(), params=params, timeout=60) as response:
                response.raise_for_status()
                listing = response.json()
            names += [item["name"][len(self.prefix):] for item in listing.get("items", [])]
            page = listing.get("nextPageToken")
            if not page:
                return sorted(name for name in names if name and "/" not in name)

    def get(self, name, target):
        url = f"{self.objects()}/{quote(self.prefix + name, safe='')}"
        with self.session().get(url, params={"alt": "media", "userProject": self.project},
                                stream=True, timeout=120) as response:
            response.raise_for_status()
            with Path(target).open("wb") as handle:
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)

    def delete_all(self):
        for name in self.names():
            url = f"{self.objects()}/{quote(self.prefix + name, safe='')}"
            response = self.session().delete(url, params={"userProject": self.project}, timeout=60)
            if response.status_code != 404:
                response.raise_for_status()

    def put(self, name, source):
        self.put_bytes(name, Path(source).read_bytes())

    def put_bytes(self, name, body):
        url = f"https://storage.googleapis.com/upload/storage/v1/b/{quote(self.bucket, safe='')}/o"
        params = {"uploadType": "media", "name": self.prefix + name, "userProject": self.project}
        for attempt in range(4):
            try:
                response = self.session().post(url, params=params, data=body, timeout=300,
                                               headers={"Content-Type": "application/x-tar"})
            except OSError:
                # Connection-level failures only; an HTTP status is judged below.
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
                continue
            if response.status_code >= 500 and attempt < 3:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            if response.json().get("md5Hash") != base64.b64encode(hashlib.md5(body).digest()).decode():
                raise RuntimeError("stored checkpoint batch does not match its bytes")
            return


def study_identity(config):
    """The part of a study config that determines results. Compute bounds
    (threads, timeouts, query budgets) do not, so retuning one never discards
    finished steps. The disease list counts as the file content it resolves to,
    and the frozen hash is excluded because it is this identity's own hash."""
    identity = {key: value for key, value in config.items()
                if key not in ("compute", "diseases_file", "frozen_config_sha256")}
    # The workspace fills in its project and CDR at submission; the checkpoint
    # signature records them (deployment_identity), the frozen hash does not.
    identity["data"] = {key: value for key, value in result_identity(config["data"]).items()
                        if key not in DEPLOYMENT}
    return identity


def deployment_identity(config):
    return {key: config["data"].get(key) for key in sorted(DEPLOYMENT)}


def config_hash(config):
    return hashlib.sha256(json.dumps(study_identity(config), sort_keys=True).encode()).hexdigest()


def step_files(directory):
    """Every regular file under a step directory, as sorted POSIX relative paths."""
    directory = Path(directory)
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("a checkpoint step cannot contain symlinks")
        if path.is_file() and path.name != RECEIPT:
            files.append(path.relative_to(directory).as_posix())
    return files


class Checkpoint:
    """Completed steps under `root`, mirrored to `store` by a background uploader.

    `signature` is everything that determines results (configuration, disease
    list, code, engine and input identities). A store written under another
    signature is refused rather than mixed in. Uploads are batched: at most one
    per `min_interval` seconds unless `sync` asks for one now.
    """
    def __init__(self, root, store, signature, *, min_interval=20.0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = store
        self.signature = json.loads(json.dumps(signature, sort_keys=True))
        self.min_interval = min_interval
        self.pending = []
        self.uploading = False
        self.urgent = False
        self.closed = False
        self.error = None
        self.sequence = 0
        self.condition = threading.Condition()
        self.restored_batches = self.restore()
        self.thread = threading.Thread(target=self.upload_loop, name="checkpoint-upload", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ restore
    def restore(self):
        names = self.store.names()
        manifest = self.root / MANIFEST
        if MANIFEST in names:
            self.store.get(MANIFEST, manifest)
            if json.loads(manifest.read_text()) != self.signature:
                raise ValueError("checkpoint inputs, configuration, engine or code do not match this run")
        else:
            manifest.write_text(json.dumps(self.signature, sort_keys=True, indent=1))
            self.store.put(MANIFEST, manifest)
        batches = sorted(name for name in names if name.startswith(BATCH_PREFIX) and name.endswith(".tar"))
        staging = self.root.parent / (self.root.name + "-restore")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        restored = 0
        for name in batches:
            local = staging / name
            self.store.get(name, local)
            restored += local.stat().st_size
            if restored > RESTORE_BUDGET:
                raise ValueError("checkpoint exceeds its restore budget")
            self.extract(local)
            local.unlink()
            self.sequence = max(self.sequence, int(name[len(BATCH_PREFIX):-len(".tar")]) + 1)
        shutil.rmtree(staging)
        return len(batches)

    def extract(self, archive):
        with tarfile.open(archive, "r") as source:
            members = source.getmembers()
            index = [m for m in members if m.name == "index.json"]
            if len(index) != 1:
                raise ValueError("checkpoint batch lacks its index")
            steps = json.load(source.extractfile(index[0]))
            for step in steps:
                self.check_step_name(step)
            files = [m for m in members if m.name != "index.json"]
            for member in files:
                path = PurePosixPath(member.name)
                if (not member.isfile() or path.is_absolute() or ".." in path.parts
                        or not any(member.name.startswith(step + "/") for step in steps)):
                    raise ValueError("checkpoint batch must hold only its steps' regular files")
            # A later batch holds the newer copy of a redone step: replace, never merge.
            for step in steps:
                shutil.rmtree(self.root / step, ignore_errors=True)
            source.extractall(self.root, members=files, filter="data")

    @staticmethod
    def check_step_name(step):
        path = PurePosixPath(step)
        if (not isinstance(step, str) or not step or path.is_absolute() or ".." in path.parts
                or path.as_posix() != step or step == MANIFEST):
            raise ValueError(f"invalid checkpoint step name {step!r}")

    # --------------------------------------------------------------- steps
    def path(self, step):
        self.check_step_name(step)
        return self.root / step

    def done(self, step):
        """Whether `step` completed; a partial or corrupt step is cleared for a rerun."""
        directory = self.path(step)
        receipt = directory / RECEIPT
        if receipt.is_file():
            files = json.loads(receipt.read_text()).get("files") or {}
            if files and step_files(directory) == sorted(files) and all(
                    file_hash(directory / name) == digest for name, digest in files.items()):
                return True
        shutil.rmtree(directory, ignore_errors=True)
        return False

    def begin(self, step):
        """A fresh, empty directory for `step`."""
        directory = self.path(step)
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        return directory

    def complete(self, step, info=None):
        """Seal `step` with its receipt and queue it for upload."""
        directory = self.path(step)
        files = step_files(directory)
        if not files:
            raise ValueError(f"step {step} produced no files")
        receipt = {"files": {name: file_hash(directory / name) for name in files}}
        if info is not None:
            receipt["info"] = info
        (directory / RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=1))
        with self.condition:
            self.raise_upload_error()
            if self.closed:
                raise RuntimeError("checkpoint is closed")
            self.pending.append(step)
            self.condition.notify_all()

    def info(self, step):
        return json.loads((self.path(step) / RECEIPT).read_text()).get("info")

    # -------------------------------------------------------------- upload
    def raise_upload_error(self):
        if self.error is not None:
            raise RuntimeError("checkpoint upload failed") from self.error

    def upload_loop(self):
        last = float("-inf")
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending or self.closed)
                if not self.pending:
                    return
                wait = self.min_interval - (time.monotonic() - last)
                if wait > 0:
                    self.condition.wait_for(lambda: self.closed or self.urgent, timeout=wait)
                steps, self.pending = self.pending, []
                sequence, self.sequence = self.sequence, self.sequence + 1
                self.uploading = True
            try:
                self.upload(steps, sequence)
            except BaseException as error:
                with self.condition:
                    self.error, self.closed, self.uploading = error, True, False
                    self.condition.notify_all()
                return
            last = time.monotonic()
            with self.condition:
                self.uploading = False
                self.condition.notify_all()

    def snapshot(self, step):
        """{relative path: bytes} of a sealed step exactly as its receipt names
        it, or None when the step has changed or gone since it was sealed (it is
        then sealed and queued again, or it was never done)."""
        directory = self.path(step)
        try:
            receipt = (directory / RECEIPT).read_bytes()
            files = json.loads(receipt)["files"]
            contents = {name: (directory / name).read_bytes() for name in files}
        except (FileNotFoundError, NotADirectoryError, KeyError, ValueError):
            return None
        if any(hashlib.sha256(body).hexdigest() != files[name] for name, body in contents.items()):
            return None
        return {**contents, RECEIPT: receipt}

    def upload(self, steps, sequence):
        snapshots = {step: self.snapshot(step) for step in sorted(set(steps))}
        snapshots = {step: files for step, files in snapshots.items() if files is not None}
        if not snapshots:
            return
        name = f"{BATCH_PREFIX}{sequence:06d}.tar"
        archive = self.root.parent / (self.root.name + "-" + name)
        with tarfile.open(archive, "w") as target:
            for member_name, body in [("index.json", json.dumps(sorted(snapshots)).encode()),
                                      *((f"{step}/{relative}", body) for step, files in snapshots.items()
                                        for relative, body in sorted(files.items()))]:
                member = tarfile.TarInfo(member_name)
                member.size = len(body)
                target.addfile(member, io.BytesIO(body))
        self.store.put(name, archive)
        archive.unlink()

    def delete_store(self):
        """Remove every stored object once the run's outputs are safe (SPEC 7a)."""
        self.close()
        self.store.delete_all()

    def sync(self):
        """Block until every completed step is stored; raise if an upload failed."""
        with self.condition:
            self.urgent = True
            self.condition.notify_all()
            self.condition.wait_for(lambda: self.error is not None or not (self.pending or self.uploading))
            self.urgent = False
            self.raise_upload_error()

    def close(self):
        self.sync()
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()
        self.raise_upload_error()
