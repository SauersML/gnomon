#!/usr/bin/env bash
# Self-contained reproducer for the mid-run SIGABRT in `gnomon score`.
#
# This script depends on NOTHING in your filesystem except `gnomon`
# itself. It generates a small synthetic PLINK 1.9 fileset + a synthetic
# PGS-Catalog-format scoring file in a fresh temp directory, runs
# `gnomon score` against them, and reports the exit code.
#
# Edit the SHAPE knobs below to chase the bug. Start tiny, scale up,
# narrow on the smallest config that still aborts.
#
# Usage:
#   bash examples/repro_score_sigabrt.sh
#
# Optional diagnostic flags (pass-through to the gnomon child only):
#   --mcheck      glibc aborts at the first sign of corruption
#   --no-tcache   disables per-thread cache (exposes hidden double-frees)
#   --scribble    fills freed memory so use-after-free fails loudly

set -euo pipefail

# ---- SHAPE knobs ----------------------------------------------------------
# Default is roughly the shape of the original failing run (~310k variants
# × ~450k samples, 3 scores) but synthetic. Drop these in half each time
# to find the smallest config that still crashes.
N_VARIANTS=${N_VARIANTS:-310448}
N_SAMPLES=${N_SAMPLES:-447278}
N_SCORES=${N_SCORES:-3}
SEED=${SEED:-1}
# --------------------------------------------------------------------------

GNOMON=gnomon

EXTRA_ENV=()
for arg in "$@"; do
    case "$arg" in
        --mcheck)    EXTRA_ENV+=(MALLOC_CHECK_=3) ;;
        --no-tcache) EXTRA_ENV+=(GLIBC_TUNABLES=glibc.malloc.tcache_count=0) ;;
        --scribble)  EXTRA_ENV+=(GLIBC_TUNABLES=glibc.malloc.perturb=170) ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "unknown flag: $arg (try --help)" >&2
            exit 2
            ;;
    esac
done

WORK_DIR="$(mktemp -d -t gnomon-repro-XXXXXX)"
trap 'rm -rf "$WORK_DIR"' EXIT
echo "work dir: $WORK_DIR"
echo "shape: variants=$N_VARIANTS samples=$N_SAMPLES scores=$N_SCORES seed=$SEED"

PLINK_PREFIX="$WORK_DIR/synth"
SCORE_DIR="$WORK_DIR/scores"
mkdir -p "$SCORE_DIR"

# ---- Synthetic data generation -------------------------------------------
# Inlined Python keeps this script a single file with no helper deps. We
# fix variant chromosomes to 1..22 (autosomes only — no complex-variant
# resolution path), positions are dense and increasing, alleles are a
# tiny A/C/G/T cycle so the score file's REF/ALT will always match.
python3 - "$PLINK_PREFIX" "$SCORE_DIR" \
        "$N_VARIANTS" "$N_SAMPLES" "$N_SCORES" "$SEED" <<'PY'
import os, random, struct, sys
from pathlib import Path

prefix = Path(sys.argv[1])
score_dir = Path(sys.argv[2])
n_var = int(sys.argv[3])
n_samp = int(sys.argv[4])
n_scores = int(sys.argv[5])
seed = int(sys.argv[6])

random.seed(seed)

bases = ['A', 'C', 'G', 'T']

# --- .bim: chrom rsid cm pos a1 a2 -----------------------------------------
bim_path = prefix.with_suffix('.bim')
with bim_path.open('w') as f:
    for i in range(n_var):
        chrom = 1 + (i % 22)
        pos = 1000 + i  # dense, increasing within a chrom (gnomon doesn't care)
        a1 = bases[i % 4]
        a2 = bases[(i + 1) % 4]
        f.write(f"{chrom}\trs{i}\t0\t{pos}\t{a1}\t{a2}\n")

# --- .fam: FID IID PID MID SEX PHENO ---------------------------------------
fam_path = prefix.with_suffix('.fam')
with fam_path.open('w') as f:
    for i in range(n_samp):
        f.write(f"FAM\tS{i}\t0\t0\t0\t-9\n")

# --- .bed: variant-major, magic 6c 1b 01 -----------------------------------
# Each variant occupies (n_samp + 3) // 4 bytes. We fill with 0x00 (all
# samples = homozygous A1 = 00 bits), which is the cheapest possible
# write and keeps results reproducible.
bytes_per_var = (n_samp + 3) // 4
bed_path = prefix.with_suffix('.bed')
with bed_path.open('wb') as f:
    f.write(b'\x6c\x1b\x01')
    chunk = b'\x00' * bytes_per_var
    for _ in range(n_var):
        f.write(chunk)

print(f"  bim: {bim_path}  ({n_var:,} variants)")
print(f"  fam: {fam_path}  ({n_samp:,} samples)")
print(f"  bed: {bed_path}  ({bed_path.stat().st_size:,} bytes)")

# --- Score files in PGS Catalog format -------------------------------------
# Each score file references a random N% of the variants in the BIM.
# Header gives PGS-Catalog-ish metadata so gnomon's reformat layer is happy.
import datetime
ts = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
SCORE_FRACTION = 0.5  # half the BIM is referenced per score
indices = list(range(n_var))
for s in range(n_scores):
    pgs_id = f"PGS{10**6 + s:07d}"[:9]   # PGS0000001-ish; fits the PGS\d{6,} shape
    score_path = score_dir / f"{pgs_id}_hmPOS_GRCh38.txt"
    rng = random.Random(seed * 100 + s)
    rng.shuffle(indices)
    chosen = sorted(indices[: max(1, int(n_var * SCORE_FRACTION))])
    with score_path.open('w') as f:
        f.write(f"###PGS CATALOG SCORING FILE - see https://www.pgscatalog.org/downloads/#dl_ftp_scoring\n")
        f.write(f"##pgs_id={pgs_id}\n")
        f.write(f"##pgs_name=synth_{s}\n")
        f.write(f"##trait_reported=Synthetic test trait {s}\n")
        f.write(f"##genome_build=GRCh38\n")
        f.write(f"##variants_number={len(chosen)}\n")
        f.write(f"##license=No restrictions\n")
        f.write(f"#format_version=2.0\n")
        f.write("chr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n")
        for i in chosen:
            chrom = 1 + (i % 22)
            pos = 1000 + i
            a1 = bases[i % 4]
            a2 = bases[(i + 1) % 4]
            # effect = a2 (the "minor" allele in our setup); weight in [-1, 1]
            f.write(f"{chrom}\t{pos}\t{a2}\t{a1}\t{rng.uniform(-1, 1):.6f}\n")
    print(f"  score: {score_path}  ({len(chosen):,} variants)")

PY

# Pick up exactly which PGS IDs we just wrote (in alpha order — same as
# the directory).
SCORE_ARG=$(ls "$SCORE_DIR"/*.txt | xargs -n 1 basename | sed -E 's/_.*//' | paste -sd, -)
echo
echo "score arg: $SCORE_ARG"
echo "command: gnomon score $SCORE_DIR $PLINK_PREFIX"
[[ ${#EXTRA_ENV[@]} -gt 0 ]] && echo "extra env: ${EXTRA_ENV[*]}"
echo

start=$EPOCHSECONDS
set +e
env "${EXTRA_ENV[@]}" "$GNOMON" score "$SCORE_DIR" "$PLINK_PREFIX"
rc=$?
set -e
echo
echo "=========================================================="
echo "exit=$rc elapsed=$(( EPOCHSECONDS - start ))s"
echo "shape: variants=$N_VARIANTS samples=$N_SAMPLES scores=$N_SCORES"
if [[ $rc -eq 134 ]]; then
    echo "(134 = 128 + SIGABRT — REPRODUCED the abort)"
fi
echo "=========================================================="
exit "$rc"
