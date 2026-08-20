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
                               [--missing-rate RATE]
                               [--high-missing-variants N --high-missing-rate RATE]
                               [--high-missing-samples N --high-sample-missing-rate RATE]
  bench_fit_solver.py inject-missing --data SOURCE --out DEST
                               [--missing-rate RATE]
                               [--high-missing-variants N --high-missing-rate RATE]
                               [--high-missing-samples N --high-sample-missing-rate RATE]
  bench_fit_solver.py run --binary PATH --data DIR/prefix [--components K]
  bench_fit_solver.py compare-plink --data DIR/prefix --eigenvec PATH
                               [--components K]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
PLINK_BYTE_HAS_MISSING = np.array([
    any(((byte >> (lane * 2)) & 0b11) == 0b01 for lane in range(4))
    for byte in range(256)
], dtype=np.bool_)


def validate_missing_profile(samples: int, variants: int, missing_rate: float,
                             high_missing_variants: int,
                             high_missing_rate: float | None,
                             high_missing_samples: int,
                             high_sample_missing_rate: float | None) -> None:
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    if not 0 <= high_missing_variants <= variants:
        raise ValueError("high_missing_variants must be between 0 and variants")
    if high_missing_variants == 0 and high_missing_rate is not None:
        raise ValueError("high_missing_rate requires high_missing_variants")
    if high_missing_variants > 0:
        if high_missing_rate is None:
            raise ValueError("high_missing_variants requires high_missing_rate")
        if not missing_rate < high_missing_rate < 1.0:
            raise ValueError("high_missing_rate must exceed missing_rate and be below 1")
    if not 0 <= high_missing_samples <= samples:
        raise ValueError("high_missing_samples must be between 0 and samples")
    if high_missing_samples == 0 and high_sample_missing_rate is not None:
        raise ValueError("high_sample_missing_rate requires high_missing_samples")
    if high_missing_samples > 0:
        if high_sample_missing_rate is None:
            raise ValueError("high_missing_samples requires high_sample_missing_rate")
        if not missing_rate < high_sample_missing_rate < 1.0:
            raise ValueError(
                "high_sample_missing_rate must exceed missing_rate and be below 1"
            )


def high_missing_ranks(variants: int, count: int) -> np.ndarray:
    ranks = np.full(variants, -1, dtype=np.int64)
    if count > 0:
        slots = (np.arange(count, dtype=np.int64) * variants) // count
        ranks[slots] = np.arange(count, dtype=np.int64)
    return ranks


def sample_missing_groups(samples: int, high_missing_samples: int):
    ranks = high_missing_ranks(samples, high_missing_samples)
    high = np.flatnonzero(ranks >= 0)
    ordinary = np.flatnonzero(ranks < 0)
    return ordinary, high


def rotated_group_rows(group: np.ndarray, rate: float, variant: int,
                       salt: int) -> np.ndarray:
    count = round(group.size * rate)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    positions = (
        (np.arange(count, dtype=np.int64) * group.size) // count
        + variant * 37
        + salt * 17
    ) % group.size
    return group[positions]


def missing_rows_for_variant(ordinary_samples: np.ndarray,
                             high_missing_samples: np.ndarray,
                             variant: int, high_variant_rank: int,
                             baseline_rate: float,
                             high_variant_rate: float | None,
                             high_sample_rate: float | None) -> np.ndarray:
    marker_rate = baseline_rate
    if high_variant_rank >= 0:
        if high_variant_rate is None:
            raise AssertionError("validated high missing rate disappeared")
        marker_rate = high_variant_rate
    ordinary = rotated_group_rows(
        ordinary_samples, marker_rate, variant, max(high_variant_rank, 0)
    )
    sample_rate = marker_rate
    if high_variant_rank < 0 and high_missing_samples.size > 0:
        if high_sample_rate is None:
            raise AssertionError("validated high sample missing rate disappeared")
        sample_rate = high_sample_rate
    high = rotated_group_rows(
        high_missing_samples, sample_rate, variant, max(high_variant_rank, 0) + 1
    )
    return np.concatenate((ordinary, high))


def write_plink(prefix: Path, samples: int, variants: int, populations: int,
                fst: float, seed: int, missing_rate: float,
                high_missing_variants: int,
                high_missing_rate: float | None,
                high_missing_samples: int,
                high_sample_missing_rate: float | None) -> None:
    """Write a .bed/.bim/.fam trio with Balding-Nichols population structure."""
    validate_missing_profile(samples, variants, missing_rate,
                             high_missing_variants, high_missing_rate,
                             high_missing_samples, high_sample_missing_rate)
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
    # Spread failing markers across the genome instead of making one bad
    # chromosome segment. This makes --geno the only intended difference
    # between the complete and filtered fits.
    high_missing_rank = high_missing_ranks(variants, high_missing_variants)
    ordinary_samples, poor_samples = sample_missing_groups(
        samples, high_missing_samples
    )

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
            high_rank = int(high_missing_rank[variant])
            if (missing_rate > 0.0 or high_rank >= 0
                    or high_missing_samples > 0):
                # An exact, deterministic per-variant rate makes two benchmark
                # runs comparable without letting Bernoulli noise change the
                # number of observed calls. Evenly spaced rows avoid deleting
                # one contiguous population block; the rotating offset prevents
                # every marker from dropping the same samples. This touches only
                # the packed codes—the genotype RNG stream stays identical to a
                # complete-call panel generated with the same seed.
                missing_rows = missing_rows_for_variant(
                    ordinary_samples, poor_samples, variant, high_rank,
                    missing_rate, high_missing_rate, high_sample_missing_rate
                )
                codes[missing_rows] = 0b01
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
          f"({size_gb:.2f} GB, baseline missing rate {missing_rate:.3%}, "
          f"{high_missing_variants} variants at "
          f"{high_missing_rate if high_missing_rate is not None else 0.0:.3%}) "
          f"and {high_missing_samples} high-missing samples at "
          f"{high_sample_missing_rate if high_sample_missing_rate is not None else 0.0:.3%} "
          f"in {time.time() - t0:.0f}s", file=sys.stderr)


def inject_missing(source: Path, out: Path, missing_rate: float,
                   high_missing_variants: int,
                   high_missing_rate: float | None,
                   high_missing_samples: int,
                   high_sample_missing_rate: float | None) -> None:
    """Copy a complete-call PLINK trio while injecting exact missingness."""
    with open(f"{source}.fam") as fam:
        samples = sum(1 for _ in fam)
    with open(f"{source}.bim") as bim:
        variants = sum(1 for _ in bim)
    validate_missing_profile(samples, variants, missing_rate,
                             high_missing_variants, high_missing_rate,
                             high_missing_samples, high_sample_missing_rate)
    ranks = high_missing_ranks(variants, high_missing_variants)
    ordinary_samples, poor_samples = sample_missing_groups(
        samples, high_missing_samples
    )
    bytes_per_variant = (samples + 3) // 4

    out.parent.mkdir(parents=True, exist_ok=True)
    outputs = [Path(f"{out}{suffix}") for suffix in (".bed", ".bim", ".fam")]
    parts = [Path(f"{path}.part") for path in outputs]
    occupied = [path for path in outputs + parts if path.exists()]
    if occupied:
        raise FileExistsError(f"refusing to overwrite {occupied[0]}")

    started = time.time()
    try:
        with open(f"{source}.bed", "rb") as src, open(parts[0], "xb") as dst:
            magic = src.read(3)
            if magic != PLINK_BED_MAGIC:
                raise ValueError(f"{source}.bed is not a variant-major PLINK BED")
            dst.write(magic)
            for variant in range(variants):
                packed = np.frombuffer(src.read(bytes_per_variant),
                                       dtype=np.uint8).copy()
                if packed.size != bytes_per_variant:
                    raise ValueError(f"short BED row at variant {variant}")
                full_bytes = samples // 4
                tail_lanes = samples % 4
                tail_missing = (
                    tail_lanes > 0
                    and any(
                        ((int(packed[full_bytes]) >> (lane * 2)) & 0b11) == 0b01
                        for lane in range(tail_lanes)
                    )
                )
                if PLINK_BYTE_HAS_MISSING[packed[:full_bytes]].any() or tail_missing:
                    raise ValueError(
                        f"source already has missing calls at variant {variant}"
                    )

                rank = int(ranks[variant])
                rows = missing_rows_for_variant(
                    ordinary_samples, poor_samples, variant, rank,
                    missing_rate, high_missing_rate, high_sample_missing_rate
                )
                if rows.size > 0:
                    byte_indices = rows >> 2
                    shifts = ((rows & 3) << 1).astype(np.uint8)
                    clear = np.bitwise_not(np.left_shift(np.uint8(3), shifts))
                    missing = np.left_shift(np.uint8(1), shifts)
                    np.bitwise_and.at(packed, byte_indices, clear)
                    np.bitwise_or.at(packed, byte_indices, missing)
                dst.write(packed.tobytes())

            if src.read(1):
                raise ValueError(f"{source}.bed has trailing bytes")

        shutil.copyfile(f"{source}.bim", parts[1])
        shutil.copyfile(f"{source}.fam", parts[2])
        os.replace(parts[1], outputs[1])
        os.replace(parts[2], outputs[2])
        os.replace(parts[0], outputs[0])
    except BaseException:
        for part in parts:
            part.unlink(missing_ok=True)
        raise

    print(f"wrote {out}.bed  {samples} samples x {variants} variants "
          f"({high_missing_variants} variants at "
          f"{high_missing_rate if high_missing_rate is not None else 0.0:.3%}, "
          f"baseline {missing_rate:.3%}; {high_missing_samples} samples at "
          f"{high_sample_missing_rate if high_sample_missing_rate is not None else 0.0:.3%}) "
          f"in {time.time() - started:.1f}s",
          file=sys.stderr)


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


def read_plink_eigenvec(path: Path, rows: int, components: int):
    """Read IID and PC columns from a PLINK2 .eigenvec file."""
    scores = np.empty((rows, components), dtype=np.float64)
    row_ids = []
    with open(path) as stream:
        header = stream.readline().split()
        expected = [f"PC{idx + 1}" for idx in range(components)]
        if len(header) < 2 + components or header[1] != "IID":
            raise ValueError(f"{path} has no IID/PC columns")
        if header[2:2 + components] != expected:
            raise ValueError(f"{path} PC columns are not {expected}")
        for row, line in enumerate(stream):
            if row >= rows:
                raise ValueError(f"{path} has more than {rows} score rows")
            fields = line.split()
            row_ids.append(fields[1])
            scores[row, :] = fields[2:2 + components]
    if len(row_ids) != rows:
        raise ValueError(f"{path} has {len(row_ids)} rows, expected {rows}")
    return row_ids, scores


def population_variance_shares(prefix: Path, row_ids: list[str],
                               scores: np.ndarray) -> np.ndarray:
    labels = {}
    with open(f"{prefix}.fam") as fam:
        for line in fam:
            fields = line.split()
            labels[fields[1]] = fields[0]
    groups = np.array([labels[iid] for iid in row_ids])
    distinct = sorted(set(groups))
    shares = np.empty(scores.shape[1], dtype=np.float64)
    for component in range(scores.shape[1]):
        column = scores[:, component]
        grand = column.mean()
        between = sum(
            np.sum(groups == group)
            * (column[groups == group].mean() - grand) ** 2
            for group in distinct
        )
        total = np.sum((column - grand) ** 2)
        shares[component] = between / total if total > 0 else 0.0
    return shares


def compare_plink(prefix: Path, eigenvec: Path, components: int) -> int:
    """Compare gnomon scores with a matched PLINK2 PCA score space."""
    scores_path = prefix.parent / f"{prefix.name}.hwe_scores.bin"
    gnomon_ids, gnomon_scores = read_scores_bin(scores_path)
    if gnomon_scores.shape[1] < components:
        raise ValueError(
            f"gnomon has {gnomon_scores.shape[1]} PCs, requested {components}"
        )
    gnomon_scores = gnomon_scores[:, :components].copy()
    plink_ids, plink_scores = read_plink_eigenvec(
        eigenvec, len(gnomon_ids), components
    )

    model_path = prefix.parent / f"{prefix.name}.hwe.json"
    with open(model_path) as model_stream:
        model_keys = json.load(model_stream)["variant_keys"]
    bim_keys = {}
    with open(f"{prefix}.bim") as bim:
        for line in bim:
            chromosome, marker_id, _, position, allele1, allele2 = line.split()
            bim_keys[marker_id] = (
                chromosome, int(position), (allele1, allele2)
            )
    allele_path = Path(f"{eigenvec}.allele")
    plink_marker_keys = []
    with open(allele_path) as allele_stream:
        allele_stream.readline()
        previous_marker_id = None
        for line in allele_stream:
            marker_id = line.split()[1]
            marker_key = bim_keys[marker_id]
            # PLINK2 allele-wts emits one row per allele, so each biallelic
            # marker appears twice. The physical marker stream appears in the
            # same order on both rows.
            if marker_id != previous_marker_id:
                plink_marker_keys.append(marker_key)
                previous_marker_id = marker_id
    gnomon_marker_keys = [
        (key["chromosome"], int(key["position"]), tuple(key["alleles"]))
        for key in model_keys
    ]
    marker_keys_exact = gnomon_marker_keys == plink_marker_keys
    if not marker_keys_exact:
        mismatch = next(
            (idx for idx, pair in enumerate(zip(gnomon_marker_keys,
                                                plink_marker_keys))
             if pair[0] != pair[1]),
            min(len(gnomon_marker_keys), len(plink_marker_keys)),
        )
        print(f"marker_key_mismatch_at={mismatch} "
              f"gnomon={gnomon_marker_keys[mismatch:mismatch + 1]} "
              f"plink={plink_marker_keys[mismatch:mismatch + 1]}")

    if plink_ids != gnomon_ids:
        plink_rows = {iid: row for row, iid in enumerate(plink_ids)}
        if len(plink_rows) != len(plink_ids):
            raise ValueError("PLINK2 score rows contain duplicate IIDs")
        try:
            order = [plink_rows[iid] for iid in gnomon_ids]
        except KeyError as error:
            raise ValueError(f"PLINK2 scores are missing IID {error.args[0]}") from error
        plink_scores = plink_scores[order, :]

    gnomon_scores -= gnomon_scores.mean(axis=0)
    plink_scores -= plink_scores.mean(axis=0)
    gnomon_basis, _ = np.linalg.qr(gnomon_scores, mode="reduced")
    plink_basis, _ = np.linalg.qr(plink_scores, mode="reduced")
    canonical = np.linalg.svd(gnomon_basis.T @ plink_basis,
                              compute_uv=False)
    axis_correlations = np.abs(np.sum(gnomon_scores * plink_scores, axis=0))
    axis_correlations /= (
        np.linalg.norm(gnomon_scores, axis=0)
        * np.linalg.norm(plink_scores, axis=0)
    )

    gnomon_shares = population_variance_shares(prefix, gnomon_ids,
                                                gnomon_scores)
    plink_shares = population_variance_shares(prefix, gnomon_ids,
                                              plink_scores)
    print("canonical=" + " ".join(f"{value:.12f}" for value in canonical))
    print(f"marker_keys_exact={str(marker_keys_exact).lower()} "
          f"markers={len(gnomon_marker_keys)}")
    print("axis_correlations="
          + " ".join(f"{value:.12f}" for value in axis_correlations))
    print("gnomon_population_shares="
          + " ".join(f"{value:.9f}" for value in gnomon_shares))
    print("plink_population_shares="
          + " ".join(f"{value:.9f}" for value in plink_shares))
    return 0 if marker_keys_exact and canonical.min() >= 0.999999 else 1


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
    gen.add_argument("--missing-rate", type=float, default=0.0,
                     help="exact deterministic missing-call fraction per variant")
    gen.add_argument("--high-missing-variants", type=int, default=0,
                     help="evenly spaced variants assigned the high missing rate")
    gen.add_argument("--high-missing-rate", type=float,
                     help="exact missing-call fraction for high-missing variants")
    gen.add_argument("--high-missing-samples", type=int, default=0,
                     help="evenly spaced samples assigned the high missing rate")
    gen.add_argument("--high-sample-missing-rate", type=float,
                     help="exact missing-call fraction for high-missing samples")

    inject = sub.add_parser("inject-missing",
                            help="copy an existing PLINK trio with exact missing calls")
    inject.add_argument("--data", required=True, type=Path, help="source PLINK prefix")
    inject.add_argument("--out", required=True, type=Path, help="output PLINK prefix")
    inject.add_argument("--missing-rate", type=float, default=0.0,
                        help="exact baseline missing-call fraction per variant")
    inject.add_argument("--high-missing-variants", type=int, default=0,
                        help="evenly spaced variants assigned the high missing rate")
    inject.add_argument("--high-missing-rate", type=float,
                        help="exact missing-call fraction for high-missing variants")
    inject.add_argument("--high-missing-samples", type=int, default=0,
                        help="evenly spaced samples assigned the high missing rate")
    inject.add_argument("--high-sample-missing-rate", type=float,
                        help="exact missing-call fraction for high-missing samples")

    run = sub.add_parser("run", help="time one fit")
    run.add_argument("--binary", required=True, type=Path)
    run.add_argument("--data", required=True, type=Path, help="PLINK prefix")
    run.add_argument("--components", type=int, default=20)
    run.add_argument("--extra", nargs=argparse.REMAINDER, default=[])

    check = sub.add_parser("verify", help="check the fit recovered the planted structure")
    check.add_argument("--data", required=True, type=Path, help="PLINK prefix")
    check.add_argument("--components", type=int, default=20)

    compare = sub.add_parser("compare-plink",
                             help="compare gnomon scores with PLINK2 eigenvectors")
    compare.add_argument("--data", required=True, type=Path, help="PLINK prefix")
    compare.add_argument("--eigenvec", required=True, type=Path)
    compare.add_argument("--components", type=int, default=4)

    args = parser.parse_args(argv)

    if args.command == "generate":
        write_plink(args.out, args.samples, args.variants,
                    args.populations, args.fst, args.seed, args.missing_rate,
                    args.high_missing_variants, args.high_missing_rate,
                    args.high_missing_samples, args.high_sample_missing_rate)
        return 0

    if args.command == "inject-missing":
        inject_missing(args.data, args.out, args.missing_rate,
                       args.high_missing_variants, args.high_missing_rate,
                       args.high_missing_samples, args.high_sample_missing_rate)
        return 0

    if args.command == "verify":
        return verify_structure(args.data, args.components)

    if args.command == "compare-plink":
        return compare_plink(args.data, args.eigenvec, args.components)

    result = run_fit(args.binary, args.data, args.components, args.extra)
    print(f"seconds={result['seconds']:.1f}")
    for note in result["notes"]:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
