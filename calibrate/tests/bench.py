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

# Import data-generation utilities from the sibling mgcv.py module
import mgcv  # same directory import


# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # calibrate/
WORKSPACE_ROOT = PROJECT_ROOT.parent  # repo root
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


# NOTE: Removed PNG plot generation to keep report speed-focused and single-plot (flamegraph only).


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

    # Fallback 2: generate a minimal SVG flamegraph in pure Python from `perf script` output
    try:
        svg_text = _render_flamegraph_inline(perf_data_path)
        if svg_text:
            svg_path.write_text(svg_text, encoding='utf-8')
            return svg_path
    except Exception as _e:
        pass

    return None


def _render_flamegraph_inline(perf_data_path: Path) -> str:
    """Minimal flamegraph generator.

    Parses `perf script -i <data>` output into collapsed stacks and renders
    a simple SVG. This is a best-effort fallback to avoid external tools.
    """
    rc, lines = run_capture(["perf", "script", "-i", str(perf_data_path)], cwd=WORKSPACE_ROOT)
    if rc != 0 or not lines:
        return ""

    stacks: list[list[str]] = []
    cur: list[str] = []
    import re
    header_re = re.compile(r"\S.*:\s")
    for raw in lines:
        line = raw.rstrip("\n")
        if not line:
            if cur:
                stacks.append(cur[:])
                cur.clear()
            continue
        # Start of sample block
        if header_re.search(line) and not line.startswith(" ") and not line.startswith("\t"):
            # header line (event), flush any pending
            if cur:
                stacks.append(cur[:])
                cur.clear()
            continue
        # stack frame line (indented)
        if line.startswith(" ") or line.startswith("\t"):
            sym = line.strip()
            # Try to strip addresses and DSOs, keep symbol name
            # Common formats include: "ffffffff foo (lib.so)" or "7f.. symbol ([unknown])"
            # Prefer substring before first " (" if present
            if " (" in sym:
                sym = sym.split(" (", 1)[0]
            # Drop leading hex address tokens
            parts = sym.split()
            if parts and re.fullmatch(r"0x?[0-9a-fA-F]+", parts[0]):
                parts = parts[1:]
            sym = " ".join(parts) if parts else sym
            # Filter overly noisy frames (optional): keep userland focus if available
            cur.append(sym)
            continue
        # other lines ignored
    if cur:
        stacks.append(cur[:])

    # Convert to collapsed stack format root;...;leaf with count 1 per sample
    # perf script typically prints leaf->root; reverse to root->leaf if main present near end
    collapsed: dict[str, int] = {}
    for s in stacks:
        if not s:
            continue
        stack = s[::-1]  # assume reverse to root->leaf
        key = ";".join(stack)
        collapsed[key] = collapsed.get(key, 0) + 1

    if not collapsed:
        return ""

    # Build tree
    class Node:
        __slots__ = ("name", "children", "value")
        def __init__(self, name: str):
            self.name = name
            self.children: dict[str, Node] = {}
            self.value = 0

    root = Node("root")
    for key, count in collapsed.items():
        node = root
        node.value += count
        for part in key.split(";"):
            child = node.children.get(part)
            if child is None:
                child = Node(part)
                node.children[part] = child
            child.value += count
            node = child

    # Layout
    width_px = 1200
    frame_h = 18
    pad_x = 1
    pad_y = 2
    total = max(root.value, 1)
    scale = width_px / total

    # Color by hash
    import hashlib
    def color(name: str) -> str:
        h = hashlib.sha1(name.encode('utf-8')).hexdigest()
        # map to warm palette
        r = int(h[0:2], 16)
        g = int(h[2:4], 16) // 2
        b = int(h[4:6], 16) // 4
        return f"rgb({200 + r%55},{80 + g%100},{b%80})"

    # Traverse to compute rectangles
    rects: list[tuple[float,int,float,str]] = []  # x, depth, width, name

    def walk(node: Node, x0: float, depth: int):
        x = x0
        for child in node.children.values():
            w = child.value * scale
            rects.append((x, depth, w, child.name))
            walk(child, x, depth + 1)
            x += w

    walk(root, 0.0, 0)

    max_depth = 0
    if rects:
        max_depth = max(d for _, d, _, _ in rects)

    height_px = int((max_depth + 1) * (frame_h + pad_y) + 30)

    # Render SVG
    def esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
                 .replace('"', "&quot;"))

    out = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width_px}' height='{height_px}' style='font-family:monospace;font-size:11px'>",
        "<style>rect{stroke:#eee;stroke-width:0.5} .lbl{pointer-events:none;fill:#000}</style>",
        f"<text x='4' y='14'>Flamegraph (samples={total})</text>",
    ]
    for x, depth, w, name in rects:
        y = 20 + depth * (frame_h + pad_y)
        out.append(f"<g>")
        out.append(f"<title>{esc(name)} — {w/scale:.0f} samples</title>")
        out.append(f"<rect x='{x:.2f}' y='{y}' width='{w:.2f}' height='{frame_h}' fill='{color(name)}'></rect>")
        # label if wide enough
        if w >= 40:
            tx = x + 3
            ty = y + frame_h - 5
            out.append(f"<text class='lbl' x='{tx:.2f}' y='{ty}'>{esc(name[:80])}</text>")
        out.append("</g>")
    out.append("</svg>")
    return "\n".join(out)


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
