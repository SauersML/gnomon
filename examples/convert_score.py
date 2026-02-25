#!/usr/bin/env python3
"""Score direct-to-consumer genotype exports using convert_genome + gnomon.

Pipeline:
1. Convert DTC text -> VCF (via convert_genome)
2. Score converted VCF with gnomon

Required dependencies:
- gnomon
- convert_genome

Usage:
    python examples/convert_score.py
    python examples/convert_score.py --assembly GRCh38
"""

from __future__ import annotations

import argparse
import gzip
import selectors
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "convert_score_output"

PGS_IDS = ("PGS000007", "PGS000317", "PGS004869", "PGS000507")
PGS_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"

GENOME_FILES = (
    ("genome_Joshua_Yoakem_v5_Full_20250129211749.txt", "Joshua_Yoakem"),
    ("autosomal.txt", "LivingDNA_Autosomal"),
)


@dataclass
class ScoreResult:
    pgs_id: str
    average: float
    missing_pct: float


def debug(msg: str) -> None:
    print(f"[convert_score] {msg}", flush=True)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f}s"


def run_command(
    argv: list[str],
    cwd: Path | None = None,
    *,
    live: bool = False,
    heartbeat_seconds: int = 30,
    heartbeat_label: str | None = None,
) -> subprocess.CompletedProcess[str]:
    debug(f"$ {' '.join(map(str, argv))}")

    if not live:
        result = subprocess.run(argv, capture_output=True, text=True, cwd=cwd)
        if result.returncode != 0:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            raise RuntimeError(f"Command failed: {' '.join(map(str, argv))}")
        return result

    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None  # for type-checkers
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    captured: list[str] = []
    last_heartbeat = start
    label = heartbeat_label or Path(argv[0]).name

    while True:
        now = time.monotonic()
        timeout = max(0.0, heartbeat_seconds - (now - last_heartbeat))
        events = selector.select(timeout)

        if events:
            line = proc.stdout.readline()
            if line:
                line = line.rstrip("\n")
                captured.append(line)
                print(line, flush=True)
                continue
            if proc.poll() is not None:
                break
        else:
            elapsed = _format_seconds(time.monotonic() - start)
            debug(f"{label} still running ({elapsed})...")
            last_heartbeat = time.monotonic()

        if proc.poll() is not None:
            break

    remainder = proc.stdout.read()
    if remainder:
        for line in remainder.splitlines():
            captured.append(line)
            print(line, flush=True)

    return_code = proc.wait()
    elapsed = _format_seconds(time.monotonic() - start)
    if return_code != 0:
        tail = "\n".join(captured[-30:])
        if tail:
            print(tail, file=sys.stderr)
        raise RuntimeError(
            f"Command failed after {elapsed}: {' '.join(map(str, argv))}"
        )

    debug(f"Completed in {elapsed}: {Path(argv[0]).name}")
    result = subprocess.CompletedProcess(
        argv,
        returncode=0,
        stdout="\n".join(captured),
        stderr="",
    )
    return result


def stream_download(url: str, destination: Path) -> None:
    if destination.exists():
        debug(f"Using cached {destination.name}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    debug(f"Downloading {url}")
    try:
        with urlopen(url) as response, open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


def require_binary(name: str, install_hint: str, preferred: tuple[Path, ...] = ()) -> Path:
    for candidate in preferred:
        if candidate.exists() and candidate.is_file():
            debug(f"Using local {name} at {candidate}")
            return candidate
    found = shutil.which(name)
    if found:
        debug(f"Using {name} at {found}")
        return Path(found)
    raise RuntimeError(f"{name} not found. Install with:\n  {install_hint}")


def download_pgs_score(pgs_id: str, cache_dir: Path, assembly: str) -> Path:
    filename = f"{pgs_id}_hmPOS_{assembly}.txt"
    target = cache_dir / filename

    if target.exists():
        debug(f"Using cached {target.name}")
        return target

    gz_path = cache_dir / f"{filename}.gz"
    url = f"{PGS_BASE_URL}/{pgs_id}/ScoringFiles/Harmonized/{filename}.gz"

    try:
        stream_download(url, gz_path)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not download harmonized score for {pgs_id} ({assembly})."
        ) from exc

    debug(f"Decompressing {gz_path.name}")
    with gzip.open(gz_path, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_path.unlink(missing_ok=True)
    return target


def convert_genome_to_vcf(
    convert_genome_bin: Path,
    genome_path: Path,
    sample_id: str,
    reference: Path | None,
    output_dir: Path,
    assembly: str,
) -> Path:
    start = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    vcf_path = output_dir / f"{sample_id}.vcf"

    if vcf_path.exists():
        debug(f"Using cached {vcf_path.name}")
        return vcf_path

    cmd = [
        str(convert_genome_bin),
        str(genome_path),
        str(vcf_path),
        "--format",
        "vcf",
        "--assembly",
        assembly,
    ]
    if reference is not None:
        cmd.extend(["--reference", str(reference)])

    debug(f"Starting conversion for {sample_id} ({genome_path.name})")
    run_command(
        cmd,
        live=True,
        heartbeat_seconds=30,
        heartbeat_label=f"convert_genome:{sample_id}",
    )

    if not vcf_path.exists():
        raise RuntimeError(f"convert_genome did not produce output: {vcf_path}")
    debug(f"Finished conversion for {sample_id} in {_format_seconds(time.monotonic() - start)}")
    return vcf_path


def parse_sscore(path: Path) -> list[ScoreResult]:
    results: list[ScoreResult] = []
    with open(path, encoding="utf-8") as handle:
        header: list[str] | None = None
        for line in handle:
            if line.startswith("#REGION"):
                continue
            if header is None:
                header = line.strip().lstrip("#").split("\t")
                continue
            values = line.strip().split("\t")
            i = 1
            while i < len(values):
                if header[i].endswith("_AVG"):
                    results.append(
                        ScoreResult(
                            pgs_id=header[i].replace("_AVG", ""),
                            average=float(values[i]),
                            missing_pct=float(values[i + 1]),
                        )
                    )
                    i += 2
                else:
                    i += 1
    return results


def score_vcf(gnomon_bin: Path, score_file: Path, vcf_path: Path, assembly: str) -> Path:
    start = time.monotonic()
    build = "37" if assembly == "GRCh37" else "38"
    score_name = score_file.stem.replace(f"_hmPOS_{assembly}", "")
    debug(f"Scoring {vcf_path.name} with {score_name}")
    run_command(
        [str(gnomon_bin), "score", str(score_file), str(vcf_path), "--build", build],
        live=True,
        heartbeat_seconds=30,
        heartbeat_label=f"gnomon-score:{score_name}",
    )

    sscore = vcf_path.parent / f"{vcf_path.stem}_{score_file.stem}.sscore"
    if not sscore.exists():
        raise RuntimeError(f"Score output not found: {sscore}")
    debug(f"Finished scoring {score_name} in {_format_seconds(time.monotonic() - start)}")
    return sscore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assembly",
        choices=["GRCh37", "GRCh38"],
        default="GRCh37",
        help="Target assembly for convert_genome output and harmonized score downloads.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Optional FASTA reference path passed to convert_genome.",
    )
    return parser


def main() -> None:
    run_start = time.monotonic()
    args = build_parser().parse_args()

    gnomon = require_binary(
        "gnomon",
        "curl -fsSL https://raw.githubusercontent.com/SauersML/gnomon/main/install.sh | bash",
        preferred=(
            REPO_ROOT / "target" / "release" / "gnomon",
            REPO_ROOT / "target" / "debug" / "gnomon",
        ),
    )
    convert_genome = require_binary(
        "convert_genome",
        "curl -fsSL https://raw.githubusercontent.com/SauersML/convert_genome/main/install.sh | bash",
    )

    output_dir = OUTPUT_DIR / args.assembly
    output_dir.mkdir(parents=True, exist_ok=True)
    score_cache = output_dir / "scores"
    score_cache.mkdir(parents=True, exist_ok=True)
    converted_dir = output_dir / "converted"
    converted_dir.mkdir(parents=True, exist_ok=True)

    genomes = [(DATA_DIR / file_name, sample_id) for file_name, sample_id in GENOME_FILES if (DATA_DIR / file_name).exists()]
    if len(genomes) < 2:
        raise RuntimeError(f"Need at least 2 genome files in {DATA_DIR}")

    all_results: dict[str, dict[str, ScoreResult]] = {}

    total_samples = len(genomes)
    total_scores = len(PGS_IDS)
    for sample_index, (genome_path, sample_id) in enumerate(genomes, start=1):
        sample_start = time.monotonic()
        debug(f"[sample {sample_index}/{total_samples}] Processing {sample_id} ({genome_path.name})")
        vcf_path = convert_genome_to_vcf(
            convert_genome,
            genome_path,
            sample_id,
            args.reference,
            converted_dir,
            args.assembly,
        )

        sample_scores: dict[str, ScoreResult] = {}
        for score_index, pgs_id in enumerate(PGS_IDS, start=1):
            debug(f"[sample {sample_index}/{total_samples}] [score {score_index}/{total_scores}] {pgs_id}")
            score_file = download_pgs_score(pgs_id, score_cache, args.assembly)
            sscore_path = score_vcf(gnomon, score_file, vcf_path, args.assembly)
            for result in parse_sscore(sscore_path):
                sample_scores[result.pgs_id] = result
        all_results[sample_id] = sample_scores
        debug(
            f"[sample {sample_index}/{total_samples}] Completed {sample_id} in "
            f"{_format_seconds(time.monotonic() - sample_start)}"
        )

    print("\n" + "=" * 60)
    print(f"SCORE SUMMARY ({args.assembly})")
    print("=" * 60)
    for sample_id, scores in all_results.items():
        print(f"\n{sample_id}:")
        for pgs_id, result in sorted(scores.items()):
            print(f"  {pgs_id}: avg={result.average:.6f}, missing={result.missing_pct:.1f}%")

    for sample_id, scores in all_results.items():
        missing = set(PGS_IDS) - set(scores)
        if missing:
            raise AssertionError(f"{sample_id} missing scores: {missing}")

    debug(f"All assertions passed. Total runtime: {_format_seconds(time.monotonic() - run_start)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
