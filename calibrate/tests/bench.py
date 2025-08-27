#!/usr/bin/env python3
"""
Calibrate end-to-end benchmark + perf harness (no arguments).

What it does:
- Generates one non-linear dataset via mgcv.py.
- Builds the profiling binary (`cargo build --profile profiling`).
- Trains the GAM via `gnomon train` and profiles with `perf record -g`.
- Produces a single HTML report with a flamegraph (if inferno is installed)
  and a filtered raw-text perf call-graph report under it.

Outputs are written under calibrate/tests and calibrate/tests/bench_workdir.
Run simply as:  python3 calibrate/tests/bench.py
"""

import os
import subprocess
import sys
from pathlib import Path
import shutil as _sh
import webbrowser
from html import escape

import pandas as pd

# Import data-generation utilities from the sibling mgcv.py module
import mgcv  # same directory import


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
# repo root = calibrate/.. from tests/
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
WORKDIR = SCRIPT_DIR / "bench_workdir"
WORKDIR.mkdir(exist_ok=True)

EXECUTABLE_PATH = WORKSPACE_ROOT / "target" / "profiling" / "gnomon"
# Minimum percent to include in perf text reports. Override with BENCH_PERF_PERCENT=1 for more detail.
PERF_PERCENT_LIMIT = int(os.environ.get("BENCH_PERF_PERCENT", "5"))


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
    # Time the perf-wrapped run
    import time as _time
    t0 = _time.perf_counter()
    run_or_die(cmd, cwd=WORKSPACE_ROOT, env=env)
    dt = _time.perf_counter() - t0

    # Collect perf reports (flat and call-graph) for HTML embedding
    graph_cmd = [
        "perf", "report", "--stdio", "--call-graph=graph", "--percent-limit", str(PERF_PERCENT_LIMIT),
        "-i", str(perf_data)
    ]
    rc_graph, graph_lines = run_capture(graph_cmd, cwd=WORKSPACE_ROOT)

    return {
        "tag": tag,
        "perf_data": perf_data,
        "runtime_sec": dt,
        "graph_ok": (rc_graph == 0),
        "graph": "\n".join(graph_lines),
    }


def generate_flamegraph(perf_data_path: Path, tag: str) -> Path | None:
    """Generate a flamegraph SVG via inferno toolchain.
    Returns SVG path or None if tools are not installed.
    """
    svg_path = WORKDIR / f"flame_{tag}.svg"
    has_collapse = _sh.which("inferno-collapse-perf") is not None
    has_flame = _sh.which("inferno-flamegraph") is not None
    if has_collapse and has_flame:
        collapsed = WORKDIR / f"collapsed_{tag}.txt"
        shell_cmd = f"perf script -i {perf_data_path} | inferno-collapse-perf > {collapsed} && inferno-flamegraph {collapsed} > {svg_path}"
        rc, _ = run_capture(["bash", "-lc", shell_cmd], cwd=WORKSPACE_ROOT)
        if rc == 0 and svg_path.exists():
            return svg_path
    return None


def main():
    # 1) Build profiling binary
    build_profiling_binary()

    # 2) Generate a single dataset (default non-linear) to keep report minimal
    datasets = [
        ("nonlinear", False, False, mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=False, noise_mode=False)),
    ]

    # 3) Convert to the Rust training TSV schema and run training under perf
    report_sections = []
    for tag, _linear_mode, _noise_mode, df in datasets:
        train_tsv = SCRIPT_DIR / f"rust_train_{tag}.tsv"
        prepare_training_tsv_from_df(df, train_tsv)
        section = train_with_perf(train_tsv, tag)
        # Generate a flamegraph SVG if possible
        section["flame_svg"] = generate_flamegraph(section["perf_data"], tag)
        report_sections.append(section)

    # 4) Build a simple HTML report and open it automatically
    html_path = WORKDIR / "report.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Calibrate Flamegraph</title>\n")
        f.write("<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:18px} h1{margin:0 0 8px} .meta{color:#444;margin:4px 0 16px} .frame{border:1px solid #ddd}</style>\n")
        f.write("<h1>Calibrate Flamegraph</h1>\n")
        # only one section
        sec = report_sections[0]
        f.write(f"<div class='meta'>Runtime: {sec['runtime_sec']:.3f} s</div>\n")
        if sec.get('flame_svg') and sec['flame_svg']:
            # Inline the SVG content so it always renders
            try:
                svg_text = Path(sec['flame_svg']).read_text(encoding='utf-8')
                f.write(svg_text)
            except Exception as e:
                f.write(f"<p>Could not inline flamegraph SVG: {escape(str(e))}</p>")
        else:
            f.write("<p>Flamegraph not available (missing tooling). Install inferno-collapse-perf and inferno-flamegraph.</p>")

        # Raw text perf report (call graph) under the plot, filtered by percent threshold
        f.write("<h2 style='margin-top:18px'>Perf Text Report</h2>\n")
        f.write(f"<div class='meta'>Showing entries with ≥{PERF_PERCENT_LIMIT}%</div>\n")
        f.write("<h3>Call Graph</h3>\n")
        if sec.get('graph_ok') and sec.get('graph'):
            f.write("<pre style='white-space:pre-wrap;max-height:500px;overflow:auto;border:1px solid #eee;padding:8px;background:#fafafa'>")
            f.write(escape(sec['graph']))
            f.write("</pre>")
        else:
            f.write("<p>Call-graph report unavailable.</p>")

    print(f"\nHTML report -> {html_path}")
    try:
        webbrowser.open(html_path.resolve().as_uri())
        print("Opened report in default browser.")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

    print("\nAll profiling runs complete.")


if __name__ == "__main__":
    main()
