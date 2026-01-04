#!/usr/bin/env python3
"""Score direct-to-consumer genotype exports using gnomon.

gnomon natively supports DTC text files (23andMe, AncestryDNA, etc.) and will
automatically download the required reference genome on first run.

Usage (after building gnomon with ``cargo build --release``):

    python examples/convert_score.py
"""

from __future__ import annotations

import gzip
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "convert_score_output"
GNOMON_BIN = REPO_ROOT / "target" / "release" / "gnomon"

PGS_IDS = ("PGS000007", "PGS000317", "PGS004869", "PGS000507")
PGS_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores"
ASSEMBLY = "GRCh37"

GENOME_FILES = (
    ("genome_Joshua_Yoakem_v5_Full_20250129211749.txt", "Joshua_Yoakem"),
    ("autosomal.txt", "LivingDNA_Autosomal"),
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


@dataclass
class ScoreResult:
    pgs_id: str
    average: float
    missing_pct: float


def debug(msg: str) -> None:
    print(f"[convert_score] {msg}", flush=True)


def run(cmd: list) -> None:
    debug(f"$ {' '.join(map(str, cmd))}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def download(url: str, dest: Path) -> None:
    debug(f"Downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urlopen(url) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except (HTTPError, URLError) as e:
        raise RuntimeError(f"Download failed: {e}") from e


def get_pgs_score(pgs_id: str, cache_dir: Path) -> Path:
    """Download PGS score file if not cached."""
    filename = f"{pgs_id}_hmPOS_{ASSEMBLY}.txt"
    target = cache_dir / filename

    if target.exists():
        debug(f"Using cached {target.name}")
        return target

    gz_file = cache_dir / f"{filename}.gz"
    url = f"{PGS_BASE_URL}/{pgs_id}/ScoringFiles/Harmonized/{filename}.gz"
    download(url, gz_file)

    debug(f"Decompressing {gz_file.name}")
    with gzip.open(gz_file, "rb") as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst)
    gz_file.unlink()

    return target


def parse_sscore(path: Path) -> list[ScoreResult]:
    """Parse gnomon .sscore file."""
    results = []
    with open(path) as f:
        header = None
        for line in f:
            if line.startswith("#REGION"):
                continue
            if header is None:
                header = line.strip().lstrip("#").split("\t")
                continue
            values = line.strip().split("\t")
            i = 1
            while i < len(values):
                if header[i].endswith("_AVG"):
                    results.append(ScoreResult(
                        pgs_id=header[i].replace("_AVG", ""),
                        average=float(values[i]),
                        missing_pct=float(values[i + 1]),
                    ))
                    i += 2
                else:
                    i += 1
    return results


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main() -> None:
    if not GNOMON_BIN.exists():
        raise RuntimeError(f"gnomon not found at {GNOMON_BIN}. Run: cargo build --release")

    score_cache = OUTPUT_DIR / "scores"
    all_results: dict[str, dict[str, ScoreResult]] = defaultdict(dict)

    # Find genome files
    genomes = [(DATA_DIR / f, name) for f, name in GENOME_FILES if (DATA_DIR / f).exists()]
    if len(genomes) < 2:
        raise RuntimeError("Need at least 2 genome files in data/")

    for genome_path, sample_id in genomes:
        debug(f"Processing {genome_path.name}")

        for pgs_id in PGS_IDS:
            score_file = get_pgs_score(pgs_id, score_cache)

            # Run gnomon directly on .txt file
            run([GNOMON_BIN, "score", score_file, genome_path, "--build", "37"])

            # Find output
            cache_dir = genome_path.parent / f"{genome_path.stem}.gnomon_cache"
            sscore = cache_dir / f"genotypes_{score_file.stem}.sscore"
            if not sscore.exists():
                raise RuntimeError(f"Expected output not found: {sscore}")

            # Parse and store results
            for result in parse_sscore(sscore):
                all_results[sample_id][result.pgs_id] = result

    # Print summary
    print("\n" + "=" * 50)
    print("SCORE SUMMARY")
    print("=" * 50)
    for sample, scores in all_results.items():
        print(f"\n{sample}:")
        for pgs_id, r in sorted(scores.items()):
            print(f"  {pgs_id}: avg={r.average:.6f}, missing={r.missing_pct:.1f}%")

    # Verify all scores present
    for sample, scores in all_results.items():
        assert set(PGS_IDS) <= set(scores), f"{sample} missing scores"

    debug("All assertions passed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
