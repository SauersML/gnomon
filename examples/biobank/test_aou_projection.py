"""Public fixtures for bounded, generation-pinned PC extraction."""
import hashlib
import io
import json
import os
import tarfile
from unittest.mock import MagicMock, patch

import pytest

from aou_projection import extract_projection, PrefixReader, source_identity, stream_projection


def archive_bytes(members):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in members:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


MANIFEST = ("shared/manifest.json", json.dumps({"schema_version": 1, "git_sha": "a" * 40}).encode())
PROJECTION = ("shared/projection_pcs.parquet", b"public test PC bytes")


def test_projection_stops_before_unrelated_score_bank(tmp_path):
    data = archive_bytes([MANIFEST, PROJECTION, ("shared/scores.tar", os.urandom(1024**2))])
    stream = io.BytesIO(data)
    output = tmp_path / "projection.parquet"
    receipt = extract_projection(stream, output)
    assert output.read_bytes() == PROJECTION[1]
    assert receipt["sha256"] == hashlib.sha256(PROJECTION[1]).hexdigest()
    assert stream.tell() < len(data) // 2


@pytest.mark.parametrize("members", [
    [PROJECTION],
    [MANIFEST, ("../projection_pcs.parquet", b"invalid")],
    [MANIFEST, ("different/projection_pcs.parquet", b"invalid")],
    [MANIFEST, MANIFEST, PROJECTION],
    [MANIFEST, ("shared/scores.tar", b"unneeded"), PROJECTION],
])
def test_invalid_projection_layout_never_creates_output(tmp_path, members):
    output = tmp_path / "projection.parquet"
    with pytest.raises(ValueError):
        extract_projection(io.BytesIO(archive_bytes(members)), output)
    assert not output.exists()
    assert not output.with_suffix(".partial").exists()


def test_prefix_reader_cannot_consume_an_unbounded_stream():
    reader = PrefixReader(io.BytesIO(b"12345"))
    reader.remaining = 3
    with pytest.raises(ValueError, match="prefix budget"):
        reader.read()


def test_projection_download_is_pinned_to_the_recorded_generation(tmp_path):
    metadata = MagicMock()
    metadata.json.return_value = {"generation": "123", "size": "4096", "crc32c": "AAAAAA=="}
    media = MagicMock()
    media.raw = io.BytesIO(archive_bytes([MANIFEST, PROJECTION]))
    session = MagicMock()
    session.__enter__.return_value = session
    responses = []
    for response in (metadata, media):
        context = MagicMock()
        context.__enter__.return_value = response
        responses.append(context)
    session.get.side_effect = responses
    with patch("google.auth.transport.requests.AuthorizedSession", return_value=session):
        identity = source_identity("gs://workspace/cache.tar.gz", "billing", "vm@example.org")
        receipt = stream_projection(identity, tmp_path / "pcs.parquet", "billing", "vm@example.org")
    request = session.get.call_args_list[1]
    assert request.kwargs["params"] == {"alt": "media", "userProject": "billing", "generation": "123"}
    assert request.kwargs["stream"] is True
    assert receipt["source"] == identity
    metadata.raise_for_status.assert_called_once()
    media.raise_for_status.assert_called_once()
