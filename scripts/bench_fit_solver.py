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



def read_scores_bin(path: Path):
    """Read a gnomon GNPRJ001 score matrix: (row_ids, scores).

    The container is self-describing on purpose — the row IDs travel inside the
    same artifact as the numbers, so a score matrix can never be silently paired
    with the wrong sample list.
    """
    raw = path.read_bytes()
    if raw[:8] != b"GNPRJ001":
        raise ValueError(f"{path} is not a GNPRJ001 matrix")
    rows = int.from_bytes(raw[12:20], "little")
    cols = int.from_bytes(raw[20:28], "little")
    header_len = 32
    data_bytes = rows * cols * 8
    scores = np.frombuffer(raw[header_len:header_len + data_bytes], dtype="<f8")
    # Column-major on disk.
    scores = scores.reshape(cols, rows).T

    ids_offset = header_len + data_bytes
    if raw[ids_offset:ids_offset + 8] != b"GNPSID01":
        raise ValueError("row-id section missing or malformed")
    count = int.from_bytes(raw[ids_offset + 16:ids_offset + 24], "little")
    offsets_at = ids_offset + 32
    offsets = np.frombuffer(raw[offsets_at:offsets_at + 8 * (count + 1)], dtype="<u8")
    strings_at = offsets_at + 8 * (count + 1)
    blob = raw[strings_at:]
    row_ids = [blob[offsets[i]:offsets[i + 1]].decode() for i in range(count)]
    return row_ids, scores


def verify_structure(prefix: Path, components: int) -> int:
    """Check that the fit recovered the structure the data was built with.

    A solver can converge, report tiny residuals, and still be wrong — every
    numerical certificate in the fit is a statement about the operator it was
    handed, not about whether the answer means anything. The generated cohort has
    known population labels, so the leading PCs must separate them. This is the
    end-to-end check that unit tests on synthetic operators cannot make.

    The statistic is the between-population share of variance along each PC
    (a one-way ANOVA eta-squared). Under the generated Fst the leading axes
    should carry a large share; a PC that separates nothing has a share near the
    noise floor of 1/(populations-1).
    """
    scores_path = prefix.parent / f"{prefix.name}.hwe_scores.bin"
    if not scores_path.exists():
        # `fit` writes next to the genotype file; tolerate either layout.
        scores_path = prefix.parent / "hwe_scores.bin"
    row_ids, scores = read_scores_bin(scores_path)

    labels = {}
    with open(f"{prefix}.fam") as fam:
        for line in fam:
            fields = line.split()
            labels[fields[1]] = fields[0]

    groups = np.array([labels[iid] for iid in row_ids])
    distinct = sorted(set(groups))
    print(f"{len(row_ids)} samples, {scores.shape[1]} PCs, "
          f"{len(distinct)} populations")

    recovered = 0
    for pc in range(min(components, scores.shape[1])):
        column = scores[:, pc]
        grand = column.mean()
        between = sum(np.sum(groups == g) * (column[groups == g].mean() - grand) ** 2
                      for g in distinct)
        total = np.sum((column - grand) ** 2)
        share = between / total if total > 0 else 0.0
        flag = "structure" if share > 0.25 else "noise"
        if share > 0.25:
            recovered += 1
        print(f"  PC{pc + 1}: between-population variance share {share:6.3f}  {flag}")

    # K populations span a (K-1)-dimensional mean space, so that many PCs should
    # carry real structure and the rest should not.
    expected = len(distinct) - 1
    print(f"structured PCs recovered: {recovered} (expected about {expected})")
    return 0 if recovered >= expected else 1

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

    check = sub.add_parser("verify", help="check the fit recovered the planted structure")
    check.add_argument("--data", required=True, type=Path, help="PLINK prefix")
    check.add_argument("--components", type=int, default=20)

    args = parser.parse_args(argv)

    if args.command == "generate":
        write_plink(args.out, args.samples, args.variants,
                    args.populations, args.fst, args.seed)
        return 0

    if args.command == "verify":
        return verify_structure(args.data, args.components)

    result = run_fit(args.binary, args.data, args.components, args.extra)
    print(f"seconds={result['seconds']:.1f}")
    for note in result["notes"]:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
