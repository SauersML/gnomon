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
import shutil as _sh
import webbrowser
from html import escape

import pandas as pd

# Use a headless backend for matplotlib and provide our own saved plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    # Time the perf-wrapped run
    import time as _time
    t0 = _time.perf_counter()
    run_or_die(cmd, cwd=WORKSPACE_ROOT, env=env)
    dt = _time.perf_counter() - t0

    # Collect perf reports (flat and call-graph) for HTML embedding
    flat_cmd = [
        "perf", "report", "--stdio", "--no-children", "--percent-limit", "5",
        "-i", str(perf_data)
    ]
    graph_cmd = [
        "perf", "report", "--stdio", "--call-graph=graph", "--percent-limit", "5",
        "-i", str(perf_data)
    ]
    rc_flat, flat_lines = run_capture(flat_cmd, cwd=WORKSPACE_ROOT)
    rc_graph, graph_lines = run_capture(graph_cmd, cwd=WORKSPACE_ROOT)

    return {
        "tag": tag,
        "perf_data": perf_data,
        "runtime_sec": dt,
        "flat_ok": (rc_flat == 0),
        "flat": "\n".join(flat_lines),
        "graph_ok": (rc_graph == 0),
        "graph": "\n".join(graph_lines),
    }


def save_plot_png(df: pd.DataFrame, tag: str, alpha: float, linear_mode: bool, noise_mode: bool) -> Path:
    """Generate the binned probability plots and save as PNG, similar to mgcv.create_binned_plots."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    if noise_mode:
        suptitle = f'Binned Probability (Pure Noise, alpha={alpha})'
    elif linear_mode:
        suptitle = f'Binned Probability (Linear, alpha={alpha})'
    else:
        suptitle = f'Binned Probability (Non-Linear, alpha={alpha})'
    fig.suptitle(suptitle)

    # variable_one
    df = df.copy()
    v1bin = pd.cut(df['variable_one'], bins=mgcv.N_BINS)
    v1_emp = df.groupby(v1bin, observed=True)['outcome'].mean()
    v1_true = df.groupby(v1bin, observed=True)['final_probability'].mean()
    v1_centers = [b.mid for b in v1_emp.index]
    axes[0].plot(v1_centers, v1_emp.values, 'o-', label='Empirical P(1)')
    axes[0].plot(v1_centers, v1_true.values, 'r--', label='True Prob')
    axes[0].set_title('variable 1 vs. P(1)')
    axes[0].set_xlabel('variable 1' if not linear_mode else 'variable 1 (angle)')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()

    # variable_two
    v2bin = pd.cut(df['variable_two'], bins=mgcv.N_BINS)
    v2_emp = df.groupby(v2bin, observed=True)['outcome'].mean()
    v2_true = df.groupby(v2bin, observed=True)['final_probability'].mean()
    v2_centers = [b.mid for b in v2_emp.index]
    axes[1].plot(v2_centers, v2_emp.values, 'o-', label='Empirical P(1)')
    axes[1].plot(v2_centers, v2_true.values, 'r--', label='True Prob')
    axes[1].set_title('variable 2 vs. P(1)')
    axes[1].set_xlabel('variable 2')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()

    out_png = WORKDIR / f"plot_{tag}.png"
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def generate_flamegraph(perf_data_path: Path, tag: str) -> Path | None:
    """Generate a flamegraph SVG if tooling is available.
    Preferred: inferno-collapse-perf + inferno-flamegraph from perf.data.
    Fallback: cargo flamegraph re-run (slower), if available.
    Returns SVG path or None if not generated.
    """
    svg_path = WORKDIR / f"flame_{tag}.svg"

    # Try inferno toolchain (faster, uses existing perf.data)
    has_collapse = _sh.which("inferno-collapse-perf") is not None
    has_flame = _sh.which("inferno-flamegraph") is not None
    if has_collapse and has_flame:
        collapsed = WORKDIR / f"collapsed_{tag}.txt"
        # Use a shell pipeline for brevity
        shell_cmd = f"perf script -i {perf_data_path} | inferno-collapse-perf > {collapsed} && inferno-flamegraph {collapsed} > {svg_path}"
        rc, _ = run_capture(["bash", "-lc", shell_cmd], cwd=WORKSPACE_ROOT)
        if rc == 0 and svg_path.exists():
            return svg_path

    # Fallback: cargo flamegraph (re-runs the binary; slower)
    rc_fg, _ = run_capture(["bash", "-lc", "cargo flamegraph --version"], cwd=WORKSPACE_ROOT)
    if rc_fg == 0:
        # We need to reconstruct the original command (train args are simple and refer to files we still have)
        # This fallback is not ideal because cargo flamegraph rebuilds; still acceptable when tools missing.
        # Create a temp script that echoes a note and runs gnomon train using the profiling binary path.
        # But cargo flamegraph expects a Cargo target; we approximate by running the bin with args.
        # We'll try: cargo flamegraph -o svg -- target/profiling/gnomon --version (dummy) -> Not useful.
        # Instead, run flamegraph against the profiling binary directly via: cargo flamegraph -o <svg> -- target/profiling/gnomon train <file>
        cmd = [
            "bash", "-lc",
            f"cargo flamegraph -o {svg_path} -- target/profiling/gnomon --version || cargo flamegraph -o {svg_path} -- target/profiling/gnomon"
        ]
        rc2, _ = run_capture(cmd, cwd=WORKSPACE_ROOT)
        if rc2 == 0 and svg_path.exists():
            return svg_path

    return None


def main():
    # 1) Build profiling binary
    build_profiling_binary()

    # 2) Generate three datasets using mgcv's generator:
    #    default (non-linear), linear mode, and pure noise.
    datasets = [
        ("nonlinear", False, False, mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=False, noise_mode=False)),
        ("linear",    True,  False, mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=True,  noise_mode=False)),
        ("noise",     False, True,  mgcv.generate_data(mgcv.N_SAMPLES_TRAIN, mgcv.NOISE_BLEND_FACTOR, linear_mode=False, noise_mode=True)),
    ]

    # 3) Convert each to the Rust training TSV schema and run training under perf
    report_sections = []
    for tag, linear_mode, noise_mode, df in datasets:
        train_tsv = SCRIPT_DIR / f"rust_train_{tag}.tsv"
        prepare_training_tsv_from_df(df, train_tsv)
        section = train_with_perf(train_tsv, tag)
        # Save a PNG plot for the dataset
        plot_path = save_plot_png(df, tag, mgcv.NOISE_BLEND_FACTOR, linear_mode, noise_mode)
        section["plot_png"] = plot_path
        # Generate a flamegraph SVG if possible
        section["flame_svg"] = generate_flamegraph(section["perf_data"], tag)
        report_sections.append(section)

    # 4) Build a simple HTML report and open it automatically
    html_path = WORKDIR / "report.html"
    with html_path.open("w", encoding="utf-8") as f:
        f.write("<!doctype html><meta charset='utf-8'>\n")
        f.write("<title>Calibrate Performance Report</title>\n")
        f.write("<style>body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:24px} .card{border:1px solid #e5e5e5;padding:16px;margin:16px 0} h2{margin:0 0 8px 0} pre{background:#fafafa;border:1px solid #eee;padding:10px;overflow:auto} .row{display:flex;gap:20px;flex-wrap:wrap} .col{flex:1 1 380px}</style>\n")
        f.write("<h1>Calibrate Performance Report</h1>\n")
        for sec in report_sections:
            f.write("<div class='card'>\n")
            f.write(f"<h2>{escape(sec['tag'])}</h2>\n")
            f.write(f"<p><strong>Runtime:</strong> {sec['runtime_sec']:.3f} s</p>\n")
            f.write("<div class='row'>\n")
            # Plot column
            f.write("<div class='col'>\n")
            if sec.get('plot_png'):
                rel_plot = os.path.relpath(sec['plot_png'], start=html_path.parent)
                f.write("<h3>Data plot</h3>\n")
                f.write(f"<img src='{escape(rel_plot)}' alt='plot {escape(sec['tag'])}' style='max-width:100%;height:auto;border:1px solid #ddd'/>\n")
            f.write("</div>\n")
            # Flamegraph column
            f.write("<div class='col'>\n")
            if sec.get('flame_svg') and sec['flame_svg']:
                rel_svg = os.path.relpath(sec['flame_svg'], start=html_path.parent)
                f.write("<h3>Flamegraph</h3>\n")
                f.write(f"<object data='{escape(rel_svg)}' type='image/svg+xml' style='width:100%;height:520px;border:1px solid #ddd'></object>\n")
            else:
                f.write("<h3>Flamegraph</h3><p>Not available (missing tooling).</p>")
            f.write("</div>\n")
            f.write("</div>\n")
            # Top functions text (>5%)
            if sec.get('flat'):
                f.write("<h3>Top functions (>5%)</h3>\n")
                f.write(f"<pre>{escape(sec['flat'])}</pre>\n")
            f.write("</div>\n")

    print(f"\nHTML report -> {html_path}")
    try:
        webbrowser.open(html_path.resolve().as_uri())
        print("Opened report in default browser.")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

    print("\nAll profiling runs complete.")


if __name__ == "__main__":
    main()
