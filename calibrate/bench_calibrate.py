#!/usr/bin/env python3
"""
Single-file benchmark + profiler harness for the calibrate pipeline.

It runs the compiled `gnomon` binary `train` subcommand on a realistic TSV
and optionally profiles it with Linux `perf`, summarizing hot calibrate symbols.

Usage examples:
  # Build a profiling binary first
  #   cargo build --profile profiling

  # Quick run on 2 workloads, without perf
  python3 calibrate/bench_calibrate.py

  # Include perf recording and hot function summary
  python3 calibrate/bench_calibrate.py --perf

Options:
  --binary <path>       Path to compiled gnomon binary (default: ./target/profiling/gnomon)
  --data <path>         TSV with columns: phenotype, score, PC1[, PC2..] (default: calibrate/tests/rust_formatted_training_data.tsv)
  --pcs <N>             Number of PCs to use (default: 1)
  --out <path>          Output CSV for timings (default: calibrate/bench_results.csv)
  --reps <N>            Repetitions per workload (default: 1)
  --perf                Enable `perf record` and post-run `perf report` summary
  --threads <N>         Set RAYON_NUM_THREADS for stability (default: unset)
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple


DEFAULT_BINARY = Path("./target/profiling/gnomon").resolve()
DEFAULT_DATA = Path("calibrate/tests/rust_formatted_training_data.tsv").resolve()
DEFAULT_OUT = Path("calibrate/bench_results.csv").resolve()
WORKDIR = Path("calibrate/bench_workdir").resolve()


@dataclass
class Workload:
    name: str
    n_rows: int
    pgs_knots: int = 8
    pgs_degree: int = 3
    pc_knots: int = 8
    pc_degree: int = 3
    penalty_order: int = 2
    pirls_max_iter: int = 50
    reml_max_iter: int = 80
    reml_tol: float = 1e-3


DEFAULT_WORKLOADS: List[Workload] = [
    Workload(name="N1k_K8_logit", n_rows=1000, pgs_knots=8, pc_knots=8),
    Workload(name="N2k_K8_logit", n_rows=2000, pgs_knots=8, pc_knots=8),
]


def ensure_workdir() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)


def subset_tsv(input_path: Path, output_path: Path, n_rows: int) -> int:
    """Create a random subset with exactly n_rows data rows (keeps header)."""
    with input_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise RuntimeError(f"Empty input file: {input_path}")
    header, rows = lines[0], lines[1:]
    if n_rows >= len(rows):
        # Copy entire file
        shutil.copyfile(input_path, output_path)
        return len(rows)
    sampled = random.sample(rows, n_rows)
    with output_path.open("w", encoding="utf-8") as out:
        out.write(header)
        out.writelines(sampled)
    return n_rows


def run(cmd: List[str], env=None, cwd: Path = None, stream=True) -> Tuple[int, List[str]]:
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines: List[str] = []
    assert proc.stdout is not None
    for line in iter(proc.stdout.readline, ""):
        if stream:
            sys.stdout.write(line)
            sys.stdout.flush()
        lines.append(line)
    ret = proc.wait()
    return ret, lines


def perf_hot_functions(lines: List[str], n: int = 5) -> List[str]:
    """Extract top hot symbols from `perf report --stdio --no-children` output."""
    out: List[Tuple[float, str]] = []
    for line in lines:
        # Try formats like: "  3.21%  [.] gnomon::calibrate::..."
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0].endswith("%"):
            try:
                pct = float(parts[0].rstrip("%"))
            except ValueError:
                continue
            sym = line.split("  ")[-1].strip()
            if "gnomon::calibrate::" in sym:
                out.append((pct, sym))
    out.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in out[:n]]


def main():
    ap = argparse.ArgumentParser(description="Calibrate benchmark and perf harness")
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--pcs", type=int, default=1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--perf", action="store_true")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()

    bin_path = Path(args.binary)
    data_path = Path(args.data)
    out_csv = Path(args.out)

    if not bin_path.exists():
        print(f"❌ Missing binary at {bin_path}. Build with: cargo build --profile profiling", file=sys.stderr)
        sys.exit(1)
    if not data_path.exists():
        print(f"❌ Missing data at {data_path}", file=sys.stderr)
        sys.exit(1)

    ensure_workdir()
    results_rows: List[dict] = []

    print("\n=== Calibrate Benchmarks ===")
    print(f"Binary : {bin_path}")
    print(f"Data   : {data_path}")
    print(f"Workdir: {WORKDIR}")
    print(f"PCs    : {args.pcs}")

    for wl in DEFAULT_WORKLOADS:
        print(f"\n--- Workload: {wl.name} (n_rows={wl.n_rows}) ---")

        subset_path = WORKDIR / f"subset_{wl.name}.tsv"
        actual_n = subset_tsv(data_path, subset_path, wl.n_rows)
        print(f"Subset -> {subset_path} (rows={actual_n})")

        base_cmd = [
            str(bin_path), "train", str(subset_path),
            "--num-pcs", str(args.pcs),
            "--pgs-knots", str(wl.pgs_knots),
            "--pgs-degree", str(wl.pgs_degree),
            "--pc-knots", str(wl.pc_knots),
            "--pc-degree", str(wl.pc_degree),
            "--penalty-order", str(wl.penalty_order),
            "--max-iterations", str(wl.pirls_max_iter),
            "--reml-max-iterations", str(wl.reml_max_iter),
            "--reml-convergence-tolerance", str(wl.reml_tol),
        ]

        env = os.environ.copy()
        if args.threads is not None:
            env["RAYON_NUM_THREADS"] = str(args.threads)

        for rep in range(1, args.reps + 1):
            print(f"\n[Run {rep}/{args.reps}] Executing training...\n")

            cmd = base_cmd
            perf_file = WORKDIR / f"perf_{wl.name}_rep{rep}.data"
            if args.perf:
                cmd = ["perf", "record", "-g", "-o", str(perf_file), "--"] + cmd

            t0 = time.perf_counter()
            rc, lines = run(cmd, env=env)
            dt = time.perf_counter() - t0

            success = (rc == 0)
            print(f"\n=> Result: {'OK' if success else 'FAIL'} | Time: {dt:.3f}s")

            hot_syms: List[str] = []
            if success and args.perf and perf_file.exists():
                print("\nAnalyzing perf report (flat, no-children)...\n")
                rc2, rep_lines = run(["perf", "report", "--stdio", "--no-children", "--percent-limit", "0.5", "-i", str(perf_file)], stream=True)
                if rc2 == 0:
                    hot_syms = perf_hot_functions(rep_lines, n=5)
                    if hot_syms:
                        print("\nTop hot calibrate symbols:")
                        for s in hot_syms:
                            print(f"  - {s}")

            results_rows.append({
                "workload": wl.name,
                "rep": rep,
                "n_rows": actual_n,
                "pgs_knots": wl.pgs_knots,
                "pc_knots": wl.pc_knots,
                "time_sec": f"{dt:.6f}",
                "ok": int(success),
                "hot_syms": "; ".join(hot_syms) if hot_syms else "",
            })

    # Write CSV summary
    print(f"\nWriting results -> {out_csv}")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results_rows[0].keys()))
        writer.writeheader()
        for row in results_rows:
            writer.writerow(row)

    print("\nDone.")


if __name__ == "__main__":
    main()

