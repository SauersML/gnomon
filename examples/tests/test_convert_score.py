from __future__ import annotations

import gzip
import io
import subprocess
from pathlib import Path

import pytest

from examples.misc import convert_score


def test_download_pgs_score_uses_cached_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache_dir = tmp_path
    cached = cache_dir / "PGS000001_hmPOS_GRCh37.txt"
    cached.write_text("cached", encoding="utf-8")

    called = False

    def fake_download(url: str, destination: Path) -> None:  # pragma: no cover - safety
        nonlocal called
        called = True
        raise AssertionError("download should not be invoked when cache exists")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    result = convert_score.download_pgs_score("PGS000001", cache_dir, assembly="GRCh37")

    assert result == cached
    assert called is False


def test_download_pgs_score_decompresses_fresh_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path
    target_gz = cache_dir / "PGS000002_hmPOS_GRCh37.txt.gz"

    def fake_download(url: str, destination: Path) -> None:
        with gzip.open(destination, "wb") as handle:
            handle.write(b"hm_chr\thm_pos\n")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    result = convert_score.download_pgs_score("PGS000002", cache_dir, assembly="GRCh37")

    assert result.read_text(encoding="utf-8") == "hm_chr\thm_pos\n"
    assert not target_gz.exists()


def test_download_pgs_score_raises_when_assembly_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path

    def fake_download(url: str, destination: Path) -> None:
        raise RuntimeError("404")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    with pytest.raises(RuntimeError) as excinfo:
        convert_score.download_pgs_score("PGS000003", cache_dir, assembly="GRCh37")

    assert "PGS000003" in str(excinfo.value)
    assert "GRCh37" in str(excinfo.value)


def test_download_pgs_score_does_not_mix_cached_assemblies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path
    other_assembly = cache_dir / "PGS000004_hmPOS_GRCh38.txt"
    other_assembly.write_text("other", encoding="utf-8")

    def fake_download(url: str, destination: Path) -> None:
        raise RuntimeError("404")

    monkeypatch.setattr(convert_score, "stream_download", fake_download)

    with pytest.raises(RuntimeError):
        convert_score.download_pgs_score("PGS000004", cache_dir, assembly="GRCh37")

    assert other_assembly.exists()


@pytest.mark.parametrize("assembly", ["GRCh37", "GRCh38"])
def test_convert_genome_to_vcf_preserves_requested_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, assembly: str
) -> None:
    genome_path = tmp_path / "genome.txt"
    genome_path.write_text("genome", encoding="utf-8")
    reference = tmp_path / "ref.fa"
    reference.write_text(">ref\nACGT\n", encoding="utf-8")
    output_dir = tmp_path / "vcf"

    recorded: list[list[str]] = []

    def fake_run_command(argv, cwd=None, **kwargs):
        recorded.append(list(map(str, argv)))
        Path(argv[2]).write_text("##fileformat=VCFv4.2\n", encoding="utf-8")

    monkeypatch.setattr(convert_score, "run_command", fake_run_command)

    vcf_path = convert_score.convert_genome_to_vcf(
        Path("convert_genome"),
        genome_path,
        "Sample",
        reference,
        output_dir,
        assembly,
        "GRCh37",
    )

    assert vcf_path.exists()
    assert recorded
    assert "--format" in recorded[0]
    assert "vcf" in recorded[0]
    assert "--assembly" not in recorded[0]
    assert recorded[0][recorded[0].index("--output-build") + 1] == assembly
    assert recorded[0][recorded[0].index("--input-build") + 1] == "GRCh37"


def test_failed_conversion_does_not_publish_partial_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_conversion(argv, **kwargs):
        Path(argv[2]).write_text("partial", encoding="utf-8")
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(convert_score, "run_command", failed_conversion)
    output_dir = tmp_path / "output"
    with pytest.raises(RuntimeError, match="conversion failed"):
        convert_score.convert_genome_to_vcf(
            Path("converter"), tmp_path / "input.txt", "Sample", None, output_dir, "GRCh38", "GRCh37"
        )
    assert list(output_dir.iterdir()) == []


def test_failed_download_does_not_publish_partial_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenResponse(io.BytesIO):
        headers = {}

        def read(self, size=-1):
            if self.tell():
                raise OSError("connection lost")
            return super().read(3)

    monkeypatch.setattr(convert_score, "urlopen", lambda *args, **kwargs: BrokenResponse(b"payload"))
    destination = tmp_path / "genome.txt"
    with pytest.raises(RuntimeError, match="connection lost"):
        convert_score.stream_download("https://example.test/genome", destination)
    assert list(tmp_path.iterdir()) == []


def test_truncated_response_does_not_publish_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class TruncatedResponse(io.BytesIO):
        headers = {"Content-Length": "100"}

    monkeypatch.setattr(convert_score, "urlopen", lambda *args, **kwargs: TruncatedResponse(b"short"))
    with pytest.raises(RuntimeError, match="Incomplete download"):
        convert_score.stream_download("https://example.test/genome", tmp_path / "genome.txt")
    assert list(tmp_path.iterdir()) == []


def test_corrupt_gzip_does_not_poison_decompressed_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def truncated_download(url: str, destination: Path) -> None:
        destination.write_bytes(gzip.compress(b"hm_chr\thm_pos\n" * 1000)[:-8])

    monkeypatch.setattr(convert_score, "stream_download", truncated_download)
    with pytest.raises(EOFError):
        convert_score.download_pgs_score("PGS000005", tmp_path, "GRCh37")
    assert list(tmp_path.iterdir()) == []


def test_parser_exposes_assembly_flag() -> None:
    parser = convert_score.build_parser()
    default_args = parser.parse_args([])
    assert default_args.assembly == "GRCh37"

    custom_args = parser.parse_args(["--assembly", "GRCh38"])
    assert custom_args.assembly == "GRCh38"


@pytest.mark.parametrize("binary", ["/bin/true", "/bin/false"])
def test_parity_harness_rejects_missing_output_and_command_failure(tmp_path: Path, binary: str) -> None:
    fixture = tmp_path / "fixture.vcf"
    fixture.write_text("##fileformat=VCFv4.2\n", encoding="utf-8")
    score = tmp_path / "score.tsv"
    score.write_text("fixture", encoding="utf-8")
    result = subprocess.run(
        ["bash", str(convert_score.REPO_ROOT / "tests/gnomon_all_parity.sh"), binary,
         str(fixture), str(score), "required-model", str(tmp_path / "results"), "37"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0
    assert "PASS" not in result.stdout
