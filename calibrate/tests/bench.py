#!/usr/bin/env python3
"""
Calibrate end-to-end benchmark + perf harness (no arguments).

What it does:
- Imports data generators from mgcv.py and creates three datasets:
  1) Non-linear (default), 2) Linear, 3) Pure-noise.
- Builds the profiling binary (`cargo build --profile profiling`).
- Trains the GAM via `gnomon train` for each dataset.
- Profiles each training run with `perf record -g` and prints a short summary.

Outputs are written under calibrate/tests and calibrate/tests/bench_workdir.
Run simply as:  python3 calibrate/tests/bench.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
import webbrowser
from html import escape

import pandas as pd

# Import data-generation utilities from the sibling mgcv.py module
import mgcv  # same directory import


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # calibrate/
WORKSPACE_ROOT = PROJECT_ROOT.parent  # repo root
WORKDIR = SCRIPT_DIR / "bench_workdir"
WORKDIR.mkdir(exist_ok=True)

EXECUTABLE_PATH = WORKSPACE_ROOT / "target" / "profiling" / "gnomon"


def build_profiling_binary():
    if not EXECUTABLE_PATH.exists():
        print("--- Building profiling binary (symbols kept) ---")
        # Build from workspace root so cargo.toml resolves correctly
        run_or_die(["cargo", "build", "--profile", "profiling"], cwd=WORKSPACE_ROOT)
    else:
        print("--- Found existing profiling binary. ---")


def run_or_die(cmd, cwd=None, env=None, stream=True):
    print(f"$ {' '.join(map(str, cmd))}")
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        if stream:
            sys.stdout.write(line)
    ret = proc.wait()
    if ret != 0:
        sys.exit(ret)
    return ret


def run_capture(cmd, cwd=None, env=None):
    """Run a command and capture stdout into a list of lines without exiting on non-zero."""
    proc = subprocess.Popen(
        [str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    assert proc.stdout is not None
    lines = proc.stdout.read().splitlines()
    ret = proc.wait()
    return ret, lines


def prepare_training_tsv_from_df(df: pd.DataFrame, out_path: Path):
    # Map mgcv columns -> calibrate training schema
    mapping = {
        "variable_one": "score",   # unpenalized PGS-like
        "variable_two": "PC1",     # penalized PC-like
        "outcome": "phenotype",
    }
    required = set(mapping.keys())
    if not required.issubset(df.columns):
        raise RuntimeError(f"Input DF missing required columns: {sorted(required - set(df.columns))}")
    df2 = df.rename(columns=mapping)
    df2[["phenotype", "score", "PC1"]].to_csv(out_path, sep="\t", index=False)


def train_with_perf(train_tsv: Path, tag: str):
    # Compose the train command with realistic defaults
    cmd = [
        str(EXECUTABLE_PATH), "train",
        "--num-pcs", "1",
        "--pgs-knots", "8", "--pgs-degree", "3",
        "--pc-knots", "8", "--pc-degree", "3",
        str(train_tsv),
    ]

    perf_data = WORKDIR / f"perf_{tag}.data"
    cmd = ["perf", "record", "-g", "-o", str(perf_data), "--"] + cmd

    # Use fixed threads for stability if user hasn't set it
    env = os.environ.copy()
    env.setdefault("RAYON_NUM_THREADS", "1")

    print(f"\n=== Training [{tag}] with perf ===")
    run_or_die(cmd, cwd=WORKSPACE_ROOT, env=env)

    # Collect perf reports (flat and call-graph) for HTML embedding
    flat_cmd = [
        "perf", "report", "--stdio", "--no-children", "--percent-limit", "0.5",
        "-i", str(perf_data)
    ]
    graph_cmd = [
        "perf", "report", "--stdio", "--call-graph=graph", "--percent-limit", "2",
        "-i", str(perf_data)
    ]
    rc_flat, flat_lines = run_capture(flat_cmd, cwd=WORKSPACE_ROOT)
    rc_graph, graph_lines = run_capture(graph_cmd, cwd=WORKSPACE_ROOT)

    return {
        "tag": tag,
        "perf_data": perf_data,
        "flat_ok": (rc_flat == 0),
        "flat": "\n".join(flat_lines),
        "graph_ok": (rc_graph == 0),
        "graph": "\n".join(graph_lines),
    }


def main():
    # 1) Build profiling binary
    build_profiling_binary()

    # 2) Generate three datasets using mgcv's generator:
    #    default (non-linear), linear mode, and pure noise.
    datasets = [
        ("nonlinear", mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=False, noise_mode=False)),
        ("linear",    mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=True,  noise_mode=False)),
        ("noise",     mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=False, noise_mode=True)),
    ]

    # 3) Convert each to the Rust training TSV schema and run training under perf
    report_sections = []
    for tag, df in datasets:
        train_tsv = SCRIPT_DIR / f"rust_train_{tag}.tsv"
        prepare_training_tsv_from_df(df, train_tsv)
        section = train_with_perf(train_tsv, tag)
        report_sections.append(section)

    # 4) Build a simple HTML report and open it automatically
    html_path = WORKDIR / "report.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Calibrate Bench Perf Report</title>\n")
        f.write("<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px} pre{background:#f7f7f7;padding:12px;overflow:auto;border:1px solid #eee} h2{margin-top:1.6em}</style>\n")
        f.write("<h1>Calibrate Bench Perf Report</h1>\n")
        f.write("<p>This report embeds perf outputs for each workload. Generated by calibrate/tests/bench.py</p>")
        for sec in report_sections:
            f.write(f"<h2>Workload: {escape(sec['tag'])}</h2>\n")
            f.write(f"<p>perf.data: {escape(str(sec['perf_data']))}</p>\n")
            f.write("<h3>Flat profile (no children)</h3>\n")
            flat = escape(sec.get("flat", ""))
            f.write(f"<pre>{flat}</pre>\n")
            f.write("<h3>Call graph (graph)</h3>\n")
            graph = escape(sec.get("graph", ""))
            f.write(f"<pre>{graph}</pre>\n")

    print(f"\nHTML report -> {html_path}")
    try:
        webbrowser.open(html_path.resolve().as_uri())
        print("Opened report in default browser.")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

    print("\nAll profiling runs complete.")


if __name__ == "__main__":
    main()
