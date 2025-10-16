#!/usr/bin/env python3
"""Aggregate timing results from multiple benchmark runs."""
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable

import matplotlib.pyplot as plt


def load_timings(paths: Iterable[pathlib.Path]) -> list[dict]:
    results = []
    for path in paths:
        data = json.loads(path.read_text())
        data["source"] = path.name
        results.append(data)
    if not results:
        raise SystemExit("No timing files provided")
    return results


def write_combined_plot(results: list[dict], output_dir: pathlib.Path) -> pathlib.Path:
    labels = []
    fit = []
    project = []

    for entry in results:
        label = "ld" if entry.get("ld_enabled") else "no-ld"
        platform_label = entry.get("platform", "unknown")
        labels.append(f"{label}\n{platform_label}")
        fit.append(entry.get("fit_seconds", 0.0))
        project.append(entry.get("project_seconds", 0.0))

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([p - width / 2 for p in x], fit, width, label="fit", color="#1f77b4")
    ax.bar([p + width / 2 for p in x], project, width, label="project", color="#ff7f0e")
    ax.set_ylabel("seconds")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("gnomon fit/project timings across runners")
    ax.legend()

    for idx, (xf, xp, fval, pval) in enumerate(zip(x, x, fit, project)):
        ax.text(xf - width / 2, fval, f"{fval:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(xp + width / 2, pval, f"{pval:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "timing-comparison.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timings", nargs="+", type=pathlib.Path, help="timing.json files to aggregate")
    parser.add_argument(
        "--output-dir",
        default=pathlib.Path("combined-results"),
        type=pathlib.Path,
        help="Directory for aggregated artifacts",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_timings(args.timings)
    combined_path = output_dir / "combined.json"
    combined_path.write_text(json.dumps(results, indent=2))

    plot_path = write_combined_plot(results, output_dir)
    print(json.dumps({"combined": str(combined_path), "plot": str(plot_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
