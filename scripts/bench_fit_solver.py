#!/usr/bin/env python3
"""Measure what a `gnomon fit` actually costs, in wall clock and in genome passes.

The fit's expensive resource is a traversal of the genotype data, not arithmetic,
so the question this answers is: how much wall clock does a solver change buy on
a cohort large enough to take the streaming path?

Sizing is the whole point and is easy to get wrong. `gnomon` picks a dense
covariance whenever the n*n Gram fits its memory budget, so a benchmark run at
n=10k silently measures the *dense* path and says nothing about the streaming
solver. This script therefore defaults to a sample count large enough that the
Gram cannot fit (8*n^2 bytes against a 500 GiB machine needs n well past 200k),
which is the regime the streaming solver exists for.

The genotypes carry real population structure (Balding-Nichols), so the spectrum
has genuine leading eigenvalues to find. A uniform-random panel has no structure
and every iterative solver "converges" on noise, which would flatter any change.

Usage:
  bench_fit_solver.py generate --out DIR [--samples N] [--variants M]
  bench_fit_solver.py run --binary PATH --data DIR/prefix [--components K]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Five populations with an Fst that is subtle but real: continental-scale
# structure is trivially recoverable and would not discriminate between solvers.
DEFAULT_POPULATIONS = 5
DEFAULT_FST = 0.02
# Enough samples that the n*n Gram (8*n^2 bytes) exceeds any sane memory budget,
# forcing the streaming path on both old and new builds.
DEFAULT_SAMPLES = 250_000
DEFAULT_VARIANTS = 20_000

PLINK_BED_MAGIC = bytes([0x6C, 0x1B, 0x01])


def write_plink(prefix: Path, samples: int, variants: int, populations: int,
                fst: float, seed: int) -> None:
    """Write a .bed/.bim/.fam trio with Balding-Nichols population structure."""
    rng = np.random.default_rng(seed)
    assignment = rng.integers(0, populations, size=samples)

    prefix.parent.mkdir(parents=True, exist_ok=True)

    with open(f"{prefix}.fam", "w") as fam:
        for idx in range(samples):
            # FID IID PAT MAT SEX PHENO; population lands in FID so a fitted
            # model can be checked against the truth it was generated from.
            fam.write(f"pop{assignment[idx]} s{idx} 0 0 0 -9\n")

    with open(f"{prefix}.bim", "w") as bim:
        for idx in range(variants):
            # One chromosome, strictly increasing positions: enough for the fit,
            # and LD windows stay well defined.
            bim.write(f"1\trs{idx}\t0\t{idx + 1}\tA\tG\n")

    # PLINK .bed is variant-major: one row per variant, 2 bits per sample,
    # padded to whole bytes. Codes: 00 hom-A1, 10 het, 11 hom-A2, 01 missing.
    bytes_per_variant = (samples + 3) // 4
    # Genotype -> 2-bit code, indexed by dosage 0/1/2.
    code_for_dosage = np.array([0b00, 0b10, 0b11], dtype=np.uint8)

    t0 = time.time()
    with open(f"{prefix}.bed", "wb") as bed:
        bed.write(PLINK_BED_MAGIC)
        for variant in range(variants):
            ancestral = rng.uniform(0.05, 0.95)
            # Balding-Nichols: population frequencies drawn around the ancestral
            # frequency with dispersion set by Fst.
            alpha = ancestral * (1.0 - fst) / fst
            beta = (1.0 - ancestral) * (1.0 - fst) / fst
            pop_freq = rng.beta(alpha, beta, size=populations)
            freq = pop_freq[assignment]
            dosage = rng.binomial(2, freq).astype(np.uint8)

            codes = code_for_dosage[dosage]
            # Pack four samples per byte, lowest sample in the lowest bit pair.
            padded = np.zeros(bytes_per_variant * 4, dtype=np.uint8)
            padded[:samples] = codes
            grouped = padded.reshape(bytes_per_variant, 4)
            packed = (grouped[:, 0]
                      | (grouped[:, 1] << 2)
                      | (grouped[:, 2] << 4)
                      | (grouped[:, 3] << 6)).astype(np.uint8)
            bed.write(packed.tobytes())

            if variant and variant % 2000 == 0:
                done = variant / variants
                print(f"  {variant}/{variants} variants "
                      f"({done:.0%}, {time.time() - t0:.0f}s)", file=sys.stderr)

    size_gb = (3 + bytes_per_variant * variants) / 1e9
    print(f"wrote {prefix}.bed  {samples} samples x {variants} variants "
          f"({size_gb:.2f} GB) in {time.time() - t0:.0f}s", file=sys.stderr)


def run_fit(binary: Path, data: Path, components: int, extra: list[str]) -> dict:
    """Time one fit and report wall clock plus what the run said about itself."""
    command = [str(binary), "fit", "--components", str(components), *extra, f"{data}.bed"]
    print(f"$ {' '.join(command)}", file=sys.stderr)

    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.time() - started

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-15:]
        raise SystemExit(f"fit failed ({completed.returncode}):\n" + "\n".join(tail))

    # Surface anything the fit said about convergence: a faster run that quietly
    # stopped short is not a faster run.
    notes = [line for line in (completed.stdout + completed.stderr).splitlines()
             if "warning" in line.lower() or "converg" in line.lower()
             or "Retained" in line]
    return {"seconds": elapsed, "notes": notes}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic structured PLINK fileset")
    gen.add_argument("--out", required=True, type=Path, help="path prefix to write")
    gen.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    gen.add_argument("--variants", type=int, default=DEFAULT_VARIANTS)
    gen.add_argument("--populations", type=int, default=DEFAULT_POPULATIONS)
    gen.add_argument("--fst", type=float, default=DEFAULT_FST)
    gen.add_argument("--seed", type=int, default=20260819)

    run = sub.add_parser("run", help="time one fit")
    run.add_argument("--binary", required=True, type=Path)
    run.add_argument("--data", required=True, type=Path, help="PLINK prefix")
    run.add_argument("--components", type=int, default=20)
    run.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    args = parser.parse_args(argv)

    if args.command == "generate":
        write_plink(args.out, args.samples, args.variants,
                    args.populations, args.fst, args.seed)
        return 0

    result = run_fit(args.binary, args.data, args.components, args.extra)
    print(f"seconds={result['seconds']:.1f}")
    for note in result["notes"]:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
