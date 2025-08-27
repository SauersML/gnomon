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

def _maybe_set_affinity():
    cores = os.environ.get("BENCH_PIN_CORES", "").strip()
    if not cores:
        return
    try:
        cpus = {int(x) for x in cores.split(",") if x.strip()}
        if cpus:
            os.sched_setaffinity(0, cpus)
            print(f"Affinity pinned to CPUs: {sorted(cpus)}")
    except Exception as e:
        print(f"Could not set affinity: {e}")


def build_profiling_binary():
    env = os.environ.copy()
    # Ensure good debug/unwind quality in profiling builds
    rf = env.get("RUSTFLAGS", "").strip()
    flags = "-C force-frame-pointers=yes -C debuginfo=2 -C link-dead-code=yes"
    env["RUSTFLAGS"] = (rf + " " + flags).strip() if rf else flags
    if not EXECUTABLE_PATH.exists():
        print("--- Building profiling binary (symbols kept) ---")
        # Build from workspace root so cargo.toml resolves correctly
        run_or_die(["cargo", "build", "--profile", "profiling"], cwd=WORKSPACE_ROOT, env=env)
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


def _cache_dsos_for_perf():
    exe = str(EXECUTABLE_PATH)
    rc, out = run_capture(["bash", "-lc", f"ldd {exe} || otool -L {exe}"], cwd=WORKSPACE_ROOT)
    if rc != 0:
        return
    import re as _re
    paths: list[str] = []
    for line in out:
        m = _re.search(r"(/[^ )]+?\.(?:so(?:\.[0-9]+)?|dylib))", line)
        if m:
            paths.append(m.group(1))
    for p in set(paths):
        run_capture(["perf", "buildid-cache", "-r", p], cwd=WORKSPACE_ROOT)


def annotate_blas_server(perf_data: Path) -> str:
    # Try a few common OpenBLAS/BLAS soname variants
    dsos = "libopenblas.so,libopenblas.so.0,libblas.so.3,libopenblas.dylib"
    cmd = [
        "perf", "annotate", "--stdio", "-i", str(perf_data),
        "--dsos", dsos,
        "--symbol", "blas_thread_server",
    ]
    rc, lines = run_capture(cmd, cwd=WORKSPACE_ROOT)
    return "\n".join(lines) if rc == 0 and lines else ""


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


def _run_perf_record(app_cmd: list[str], perf_data: Path, env: dict) -> float:
    """Run perf record with sensible defaults and automatic fallback for mmap limit.
    Returns wall time in seconds. Exits on non-recoverable errors.
    """
    import time as _time
    # Build candidate mmap sizes: env override first, then fallbacks, finally no -m
    mm_env = os.environ.get("BENCH_MMAP_PAGES")
    candidates: list[int | None] = []
    if mm_env and mm_env.isdigit():
        candidates.append(int(mm_env))
    # Preferred default smaller to avoid mlock errors
    for v in (256, 128, 64):
        if mm_env is None or str(v) != mm_env:
            candidates.append(v)
    candidates.append(None)  # last resort: let perf choose

    freq = os.environ.get("BENCH_FREQ", "2000")
    t0 = None
    last_err = None
    for mmap_pages in candidates:
        record = [
            "perf", "record",
            "-e", "cycles:u",
            "--call-graph", "dwarf,16384",
            "-F", freq,
            "-o", str(perf_data),
        ]
        if mmap_pages is not None:
            record += ["-m", str(mmap_pages)]
        record += ["--"] + app_cmd

        print(f"$ {' '.join(map(str, record))}")
        t0 = _time.perf_counter()
        rc, lines = run_capture(record, cwd=WORKSPACE_ROOT, env=env)
        dt = _time.perf_counter() - t0
        # Echo output for transparency
        if lines:
            sys.stdout.write("\n".join(lines) + ("\n" if lines and not lines[-1].endswith("\n") else ""))
        if rc == 0:
            return dt
        text = "\n".join(lines) if lines else ""
        last_err = (rc, text)
        if "Permission error mapping pages" in text or "mmap" in text.lower():
            print("perf record failed due to mmap/lock limits; retrying with smaller -m…")
            continue
        # Non-recoverable error
        sys.exit(rc)
    # Exhausted candidates
    if last_err is not None:
        rc, _ = last_err
        sys.exit(rc)
    return 0.0


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

    # Use fixed threads for stability if user hasn't set it
    env = os.environ.copy()
    env.setdefault("RAYON_NUM_THREADS", "1")
    # BLAS threading defaults; overridable
    oblas_threads = os.environ.get("BENCH_OBLAS_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", oblas_threads)
    env.setdefault("OMP_NUM_THREADS", oblas_threads)
    env.setdefault("MKL_NUM_THREADS", oblas_threads)
    env.setdefault("OPENBLAS_WAIT_POLICY", os.environ.get("BENCH_OBLAS_WAIT", "PASSIVE"))

    print(f"\n=== Training [{tag}] with perf ===")
    # Run perf record with automatic fallback for mmap limits
    dt = _run_perf_record(cmd, perf_data, env)

    # Collect perf reports (call-graph) for HTML embedding
    graph_cmd = [
        "perf", "report", "--stdio", "--call-graph=graph", "--percent-limit", str(PERF_PERCENT_LIMIT),
        "--max-stack", os.environ.get("BENCH_MAX_STACK", "1024"),
        "-i", str(perf_data)
    ]
    rc_graph, graph_lines = run_capture(graph_cmd, cwd=WORKSPACE_ROOT)
    annotate_text = annotate_blas_server(perf_data)

    return {
        "tag": tag,
        "perf_data": perf_data,
        "runtime_sec": dt,
        "graph_ok": (rc_graph == 0),
        "graph": "\n".join(graph_lines),
        "annotate": annotate_text,
        "env_snapshot": {
            "RAYON_NUM_THREADS": env.get("RAYON_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", ""),
            "OPENBLAS_WAIT_POLICY": env.get("OPENBLAS_WAIT_POLICY", ""),
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", ""),
            "BENCH_FREQ": os.environ.get("BENCH_FREQ", "2000"),
            "BENCH_MAX_STACK": os.environ.get("BENCH_MAX_STACK", "1024"),
        },
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
    # Optional CPU affinity for reproducibility
    _maybe_set_affinity()
    # 1) Build profiling binary
    build_profiling_binary()
    # Prime perf with DSO build-ids to improve annotation
    _cache_dsos_for_perf()

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
        # Environment snapshot for reproducibility
        es = sec.get('env_snapshot') or {}
        f.write("<div class='meta'>Env: "
                f"OPENBLAS_NUM_THREADS={escape(es.get('OPENBLAS_NUM_THREADS',''))}, "
                f"OPENBLAS_WAIT_POLICY={escape(es.get('OPENBLAS_WAIT_POLICY',''))}, "
                f"RAYON_NUM_THREADS={escape(es.get('RAYON_NUM_THREADS',''))}, "
                f"FREQ={escape(es.get('BENCH_FREQ',''))} Hz, "
                f"MAX_STACK={escape(es.get('BENCH_MAX_STACK',''))}"
                "</div>\n")
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

        # Annotate the OpenBLAS thread server if available
        f.write("<h3>OpenBLAS blas_thread_server annotate</h3>\n")
        if sec.get('annotate'):
            f.write("<pre style='white-space:pre-wrap;max-height:400px;overflow:auto;border:1px solid #eee;padding:8px;background:#fafafa'>")
            f.write(escape(sec['annotate']))
            f.write("</pre>")
        else:
            f.write("<p>No annotate output (symbols or data missing).</p>")

    print(f"\nHTML report -> {html_path}")
    try:
        webbrowser.open(html_path.resolve().as_uri())
        print("Opened report in default browser.")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")

    print("\nAll profiling runs complete.")


if __name__ == "__main__":
    main()
