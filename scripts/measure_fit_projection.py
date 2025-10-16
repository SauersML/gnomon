#!/usr/bin/env python3
"""Measure fit and projection timings for the gnomon CLI."""
from __future__ import annotations

import argparse
import io
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from typing import Iterable

import matplotlib.pyplot as plt
import requests
import zipfile


ARCHIVES = {
    "chr22_subset50.fam": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.fam.zip",
    "chr22_subset50.bim": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.bim.zip",
    "chr22_subset50.bed": "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/chr22_subset50.bed.zip",
}


class CommandError(RuntimeError):
    """Raised when an external command fails."""


def download_archive(url: str, expected: str, destination: pathlib.Path) -> pathlib.Path:
    """Download a zip archive, extract the expected file, and return its path."""
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    members = archive.namelist()
    if expected not in members:
        raise RuntimeError(
            f"Archive {url} did not contain {expected!r}; found {members}"
        )

    archive.extract(expected, path=destination)
    return destination / expected


def download_dataset(output_dir: pathlib.Path) -> pathlib.Path:
    """Download the PLINK dataset used for fit/project integration tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bed_path: pathlib.Path | None = None

    for expected, url in ARCHIVES.items():
        extracted = download_archive(url, expected, output_dir)
        if expected.endswith(".bed"):
            bed_path = extracted

    if bed_path is None:
        raise RuntimeError("Failed to download PLINK dataset; .bed file missing")

    return bed_path


def run_command(cmd: Iterable[str], cwd: pathlib.Path) -> tuple[float, subprocess.CompletedProcess[str]]:
    """Run a command, timing the elapsed wall time in seconds."""
    start = time.perf_counter()
    result = subprocess.run(
        list(cmd),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        raise CommandError(
            f"Command {' '.join(cmd)} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return elapsed, result


def write_plot(output_dir: pathlib.Path, fit_seconds: float, project_seconds: float) -> pathlib.Path:
    """Create a simple bar chart summarizing the timings."""
    labels = ["fit", "project"]
    durations = [fit_seconds, project_seconds]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, durations, color=["#1f77b4", "#ff7f0e"])
    ax.set_ylabel("seconds")
    ax.set_title("gnomon fit/project timing")
    ax.bar_label(bars, fmt="{:.2f}")
    fig.tight_layout()

    plot_path = output_dir / "timing.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=pathlib.Path, help="Path to gnomon binary")
    parser.add_argument(
        "--components",
        type=int,
        default=8,
        help="Number of principal components to fit",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("timing-results"),
        help="Directory where timing artifacts will be stored",
    )
    parser.add_argument(
        "--working-dir",
        type=pathlib.Path,
        default=None,
        help="Optional working directory for downloads and outputs",
    )
    parser.add_argument(
        "--ld",
        action="store_true",
        help="Enable LD normalization during fitting",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    binary = args.binary.resolve()
    if not binary.exists():
        raise SystemExit(f"Binary not found at {binary}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    working_dir = args.working_dir.resolve() if args.working_dir else output_dir
    working_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = working_dir / "dataset"
    bed_path = download_dataset(dataset_dir)

    fit_cmd = [str(binary), "fit", str(bed_path), "--components", str(args.components)]
    if args.ld:
        fit_cmd.append("--ld")
    fit_seconds, fit_result = run_command(fit_cmd, cwd=working_dir)

    project_cmd = [str(binary), "project", str(bed_path)]
    project_seconds, project_result = run_command(project_cmd, cwd=working_dir)

    metadata = {
        "ld_enabled": bool(args.ld),
        "components": int(args.components),
        "fit_seconds": fit_seconds,
        "project_seconds": project_seconds,
        "platform": platform.platform(),
        "python_version": sys.version,
        "runner_name": os.environ.get("RUNNER_NAME"),
        "binary": str(binary),
    }

    (output_dir / "fit_stdout.txt").write_text(fit_result.stdout)
    (output_dir / "fit_stderr.txt").write_text(fit_result.stderr)
    (output_dir / "project_stdout.txt").write_text(project_result.stdout)
    (output_dir / "project_stderr.txt").write_text(project_result.stderr)

    timing_path = output_dir / "timing.json"
    timing_path.write_text(json.dumps(metadata, indent=2))

    plot_path = write_plot(output_dir, fit_seconds, project_seconds)

    print(json.dumps({"timing": metadata, "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
