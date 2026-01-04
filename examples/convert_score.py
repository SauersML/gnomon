#!/usr/bin/env python3
"""Example workflow that scores direct-to-consumer genotype exports using gnomon.

gnomon natively supports DTC text files (23andMe, AncestryDNA, etc.) and will
automatically download the required reference genome on first run.

Steps performed:
    1. Download a configurable set of PGS catalog scores.
    2. Run gnomon directly on the raw .txt genomes in ``data/``.
    3. Print the final score matrix and assert a few high-level invariants so the script
       can be dropped into continuous integration.

Usage (after building gnomon with ``cargo build --release``)::

    python examples/convert_score.py \
        --scores PGS004696 PGS003725 \
        --output-dir examples/convert_score_output

The script caches downloaded artifacts inside ``examples/convert_score_output`` so repeat
runs are quick.  Delete that directory to force a full re-run.
"""

from __future__ import annotations

import argparse
import gzip
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# --------------------------------------------------------------------------------------
# Configuration constants
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "convert_score_output"
DEFAULT_PGS_IDS = (
    "PGS000007",
    "PGS000317",
    "PGS004869",
    "PGS000507",
)
PGS_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"

GENOME_SOURCES: Tuple[Tuple[str, str], ...] = (
    ("genome_Joshua_Yoakem_v5_Full_20250129211749.txt", "Joshua_Yoakem"),
    ("autosomal.txt", "LivingDNA_Autosomal"),
)

# --------------------------------------------------------------------------------------
# Dataclasses and small utilities
# --------------------------------------------------------------------------------------


@dataclass
class ScoreResult:
    pgs_id: str
    average: float
    missing_pct: float


def debug(msg: str) -> None:
    """Small helper to print progress messages with a common prefix."""
    print(f"[convert_score] {msg}", flush=True)


class CommandError(RuntimeError):
    """Raised when a subprocess finishes unsuccessfully."""


def run_command(argv: Sequence[os.PathLike[str] | str], cwd: Path | None = None) -> None:
    """Execute ``argv`` and raise :class:`CommandError` on failure."""
    printable = " ".join(map(str, argv))
    if cwd is not None:
        printable = f"(cd {cwd} && {printable})"
    debug(f"$ {printable}")
    completed = subprocess.run(list(map(str, argv)), cwd=cwd)
    if completed.returncode != 0:
        raise CommandError(
            f"Command failed with exit code {completed.returncode}: {printable}"
        )


def ensure_binary(name: str, explicit: str | None = None) -> Path:
    """Return the resolved path to ``name`` or raise a clear error."""
    candidate = Path(explicit) if explicit else shutil.which(name)
    if not candidate:
        raise FileNotFoundError(
            f"Required binary '{name}' was not found in PATH. Install it or pass ``--{name.replace('-', '_')}-bin``."
        )
    path = Path(candidate).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Binary '{name}' resolved to {path}, but the file does not exist.")
    return path


# --------------------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------------------


def stream_download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` using urllib with streaming."""
    debug(f"Downloading {url} -> {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as response, open(destination, "wb") as handle:
            shutil.copyfileobj(response, handle)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def download_pgs_score(
    pgs_id: str, cache_dir: Path, assembly: str = "GRCh37"
) -> Path:
    """Download the harmonized PGS scoring file for ``assembly``."""
    valid_assemblies = {"GRCh37", "GRCh38"}
    if assembly not in valid_assemblies:
        raise ValueError(
            f"Unsupported assembly '{assembly}'. Expected one of: {sorted(valid_assemblies)}."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{pgs_id}_hmPOS_{assembly}.txt.gz"
    url = f"{PGS_BASE_URL}/{pgs_id}/ScoringFiles/Harmonized/{filename}"
    target_gz = cache_dir / filename
    target_txt = cache_dir / filename[:-3]

    if target_txt.exists():
        debug(f"Using cached score {target_txt}")
        return target_txt

    if not target_gz.exists():
        try:
            stream_download(url, target_gz)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Unable to download harmonized score for {pgs_id} ({assembly})."
            ) from exc

    debug(f"Decompressing {target_gz}")
    with gzip.open(target_gz, "rb") as src, open(target_txt, "wb") as dst:
        shutil.copyfileobj(src, dst)
    target_gz.unlink(missing_ok=True)

    if not target_txt.exists():
        raise RuntimeError(
            f"Failed to prepare harmonized score for {pgs_id} ({assembly}); expected {target_txt}."
        )

    return target_txt


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def run_gnomon_score(
    tool: Path,
    score_path: Path,
    genome_path: Path,
    assembly: str,
) -> Path:
    """Execute ``gnomon score`` directly on a DTC text file.

    gnomon will automatically download the reference genome if needed.
    """
    # gnomon score <SCORE_PATH> <GENOTYPE_PATH> [--build <BUILD>]
    cmd = [tool, "score", score_path, genome_path]
    if assembly == "GRCh37":
        cmd.extend(["--build", "37"])

    run_command(cmd)

    # Output naming: genome_stem + "_" + score_stem + ".sscore"
    # But gnomon creates a cache dir, so output is in genome_path.parent / cache / ...
    # Actually, gnomon outputs relative to the converted PLINK prefix
    # The cache is at genome_path.parent / genome_stem.gnomon_cache / genotypes
    cache_dir = genome_path.parent / f"{genome_path.stem}.gnomon_cache"
    score_stem = score_path.stem
    sscore_path = cache_dir / f"genotypes_{score_stem}.sscore"

    if not sscore_path.exists():
        # Try alternate location (if gnomon changes output location)
        alt_path = genome_path.parent / f"{genome_path.stem}_{score_stem}.sscore"
        if alt_path.exists():
            sscore_path = alt_path
        else:
            raise RuntimeError(f"gnomon score did not create {sscore_path} or {alt_path}")

    return sscore_path


# --------------------------------------------------------------------------------------
# Parsing helpers
# --------------------------------------------------------------------------------------


def parse_sscore(path: Path) -> Tuple[List[str], List[ScoreResult]]:
    """Parse a gnomon ``.sscore`` file into a row-wise structure."""
    with path.open("r", encoding="utf-8") as handle:
        header: List[str] | None = None
        rows: List[ScoreResult] = []
        for raw in handle:
            if raw.startswith("#REGION"):
                continue
            if header is None:
                header = [col.lstrip("#") for col in raw.rstrip().split("\t")]
                continue
            values = raw.rstrip().split("\t")
            idx = 1
            while idx < len(values):
                score_name = header[idx]
                if not score_name.endswith("_AVG"):
                    idx += 1
                    continue
                avg = float(values[idx])
                missing = float(values[idx + 1])
                rows.append(
                    ScoreResult(
                        pgs_id=score_name.replace("_AVG", ""),
                        average=avg,
                        missing_pct=missing,
                    )
                )
                idx += 2
        if header is None:
            raise RuntimeError(f"No header found in {path}")
    return header, rows


def derive_sample_id(filename: str, override: str | None = None) -> str:
    """Generate a stable sample ID for the provided genome filename."""
    if override:
        return override
    name = Path(filename).stem
    name = re.sub(r"^genome_", "", name)
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return sanitized or "Sample"


# --------------------------------------------------------------------------------------
# Main orchestration
# --------------------------------------------------------------------------------------


def orchestrate(args: argparse.Namespace) -> None:
    gnomon = ensure_binary(
        "gnomon",
        args.gnomon_bin or str(REPO_ROOT / "target" / "release" / "gnomon"),
    )

    output_dir = Path(args.output_dir).resolve()
    score_cache = output_dir / "scores"

    genome_inputs: List[Tuple[Path, str]] = []
    for filename, override in GENOME_SOURCES:
        path = DATA_DIR / filename
        if not path.exists():
            debug(f"Skipping missing genome file: {path}")
            continue
        genome_inputs.append((path, derive_sample_id(filename, override)))

    if len(genome_inputs) < 2:
        raise RuntimeError(
            "Expected at least two genome files in the data directory; cannot proceed."
        )

    all_results: Dict[str, Dict[str, ScoreResult]] = defaultdict(dict)

    for genome_path, sample_id in genome_inputs:
        debug(f"Processing genome {genome_path.name} (sample {sample_id})")

        for pgs_id in args.scores:
            score_path = download_pgs_score(pgs_id, score_cache, assembly=args.assembly)
            sscore_path = run_gnomon_score(gnomon, score_path, genome_path, args.assembly)
            print()
            print(
                f"==== Contents of {sscore_path.name} for sample {sample_id} / {pgs_id} ===="
            )
            sscore_contents = sscore_path.read_text(encoding="utf-8")
            print(sscore_contents, end="")
            if not sscore_contents.endswith("\n"):
                print()
            print("==== End of", sscore_path.name, "====\n")
            header, scores = parse_sscore(sscore_path)
            assert scores, f"No score rows parsed from {sscore_path}"
            for score in scores:
                assert 0.0 <= score.missing_pct <= 100.0
                all_results[sample_id][score.pgs_id] = score

    print("\nFinal score summary:\n======================")
    for sample, sample_scores in all_results.items():
        print(f"Sample: {sample}")
        for pgs_id, score in sorted(sample_scores.items()):
            print(
                f"  {pgs_id}: average={score.average:.6f}, missing_pct={score.missing_pct:.2f}"
            )
        print()

    expected_scores = set(args.scores)
    for sample, sample_scores in all_results.items():
        assert (
            expected_scores <= set(sample_scores)
        ), f"Sample {sample} missing expected scores"

    debug("All assertions passed. Scoring succeeded.")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scores",
        nargs="+",
        default=list(DEFAULT_PGS_IDS),
        help="PGS catalog identifiers to download and score (default: %(default)s)",
    )
    parser.add_argument(
        "--assembly",
        default="GRCh37",
        choices=("GRCh37", "GRCh38"),
        help="Genome assembly for PGS downloads (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Working directory for downloads and intermediate files",
    )
    parser.add_argument(
        "--gnomon-bin",
        default=None,
        help="Path to the gnomon executable (defaults to target/release/gnomon)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    orchestrate(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
