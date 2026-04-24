#!/usr/bin/env bash
# gnomon_all_parity.sh
#
# Parity harness for the `gnomon all` unified subcommand.
#
# Runs the three subcommands (score / project / terms) independently on a
# tiny synthetic VCF fixture, records sha256 hashes of their outputs, then
# runs `gnomon all` on the same fixture and compares. Outputs should be
# byte-identical (same deterministic pipeline, same inputs).
#
# Requires:
#   - A `gnomon` binary on $PATH or pointed to by $GNOMON_BIN.
#   - A projection model name in $GNOMON_MODEL (defaults to
#     `hwe_1kg_hgdp_gsa_v3` but the tiny synthetic fixture won't have
#     variant overlap with the real model, so the project phase is expected
#     to fail unless $GNOMON_MODEL is a tiny test model. The score and
#     terms phases are still parity-tested.).
#
# Usage:
#   tests/gnomon_all_parity.sh [fixture_vcf]
#
# If `fixture_vcf` is omitted, a tiny synthetic VCF is generated under the
# script's temp working directory.

set -euo pipefail

GNOMON_BIN="${GNOMON_BIN:-gnomon}"
GNOMON_MODEL="${GNOMON_MODEL:-hwe_1kg_hgdp_gsa_v3}"

if ! command -v "$GNOMON_BIN" >/dev/null 2>&1; then
    echo "ERROR: gnomon binary '$GNOMON_BIN' not found on PATH." >&2
    exit 2
fi

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# --- Fixture VCF ---------------------------------------------------------
# Either a caller-provided VCF or a synthetic one covering autosomes + X + Y
# with three samples. Small enough that every subcommand finishes in <60s.
if [[ $# -ge 1 ]]; then
    FIXTURE="$1"
else
    FIXTURE="$WORK/fixture.vcf"
    cat >"$FIXTURE" <<'VCF'
##fileformat=VCFv4.2
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	S1	S2	S3
1	1000	rs1	A	G	.	PASS	.	GT	0/0	0/1	1/1
1	2000	rs2	C	T	.	PASS	.	GT	0/1	1/1	0/0
2	3000	rs3	G	A	.	PASS	.	GT	1/1	0/1	0/0
X	3500000	rsX1	T	C	.	PASS	.	GT	0/1	0/0	1/1
Y	2800000	rsY1	A	G	.	PASS	.	GT	0/0	1/1	0/0
VCF
fi

SERIAL_DIR="$WORK/serial"
ALL_DIR="$WORK/all"
mkdir -p "$SERIAL_DIR" "$ALL_DIR"

# Copy fixture into both run dirs so outputs land there and are comparable.
SERIAL_VCF="$SERIAL_DIR/$(basename "$FIXTURE")"
ALL_VCF="$ALL_DIR/$(basename "$FIXTURE")"
cp "$FIXTURE" "$SERIAL_VCF"
cp "$FIXTURE" "$ALL_VCF"

sha256_of() {
    # macOS has `shasum -a 256`, Linux has `sha256sum`.
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# You'll need a small score file that matches the fixture variants. The
# harness assumes $GNOMON_SCORE points at a PGS-native score TSV
# ("variant_id\teffect_allele\tother_allele\tSCORE") or to a PGS catalog
# directory. Edit this path before running:
#   export GNOMON_SCORE=/abs/path/to/score.tsv
GNOMON_SCORE="${GNOMON_SCORE:-$WORK/fake_score.tsv}"
if [[ ! -e "$GNOMON_SCORE" ]]; then
    cat >"$WORK/fake_score.tsv" <<'TSV'
variant_id	effect_allele	other_allele	FIXTURE_SCORE
rs1	G	A	0.5
rs2	T	C	-0.25
rs3	A	G	0.75
TSV
    GNOMON_SCORE="$WORK/fake_score.tsv"
fi

echo "[parity] fixture VCF   : $FIXTURE"
echo "[parity] gnomon bin    : $GNOMON_BIN"
echo "[parity] score file    : $GNOMON_SCORE"
echo "[parity] project model : $GNOMON_MODEL"
echo

# --- Serial baseline -----------------------------------------------------
echo "[parity] === serial baseline ==="
(cd "$SERIAL_DIR" && "$GNOMON_BIN" score "$GNOMON_SCORE" "$(basename "$SERIAL_VCF")") || {
    echo "[parity] WARN: serial gnomon score failed (may be expected if fixture has no overlap)." >&2
}
(cd "$SERIAL_DIR" && "$GNOMON_BIN" project "$(basename "$SERIAL_VCF")" --model "$GNOMON_MODEL") || {
    echo "[parity] WARN: serial gnomon project failed (often due to model-variant mismatch on fixture)." >&2
}
(cd "$SERIAL_DIR" && "$GNOMON_BIN" terms --sex "$(basename "$SERIAL_VCF")") || {
    echo "[parity] WARN: serial gnomon terms failed." >&2
}

# --- `gnomon all` --------------------------------------------------------
echo
echo "[parity] === gnomon all ==="
(cd "$ALL_DIR" && "$GNOMON_BIN" all "$GNOMON_SCORE" "$(basename "$ALL_VCF")" --model "$GNOMON_MODEL") || {
    echo "[parity] WARN: gnomon all failed; see stderr above." >&2
}

# --- Compare -------------------------------------------------------------
echo
echo "[parity] === output comparison ==="
status=0

compare_pair() {
    local name="$1" serial="$2" allrun="$3"
    if [[ ! -e "$serial" && ! -e "$allrun" ]]; then
        echo "[parity] $name: both absent (skip)."
        return
    fi
    if [[ ! -e "$serial" || ! -e "$allrun" ]]; then
        echo "[parity] $name: MISMATCH — only one of the two runs produced it."
        echo "    serial: $serial  (exists=$( [[ -e "$serial" ]] && echo yes || echo no ))"
        echo "    all   : $allrun  (exists=$( [[ -e "$allrun" ]] && echo yes || echo no ))"
        status=1
        return
    fi
    local h1 h2
    h1=$(sha256_of "$serial")
    h2=$(sha256_of "$allrun")
    if [[ "$h1" == "$h2" ]]; then
        echo "[parity] $name: byte-identical ($h1)"
    else
        echo "[parity] $name: MISMATCH"
        echo "    serial sha256 : $h1"
        echo "    all    sha256 : $h2"
        echo "    diff -u (context 3):"
        diff -u "$serial" "$allrun" | head -60 || true
        status=1
    fi
}

# Score outputs: .sscore
for f in "$SERIAL_DIR"/*.sscore; do
    [[ -e "$f" ]] || continue
    name=$(basename "$f")
    compare_pair "score/$name" "$f" "$ALL_DIR/$name"
done

# Project outputs: *.projection_scores.bin + *.projection_scores.metadata.json
for f in "$SERIAL_DIR"/*.projection_scores.bin "$SERIAL_DIR"/*.projection_scores.metadata.json; do
    [[ -e "$f" ]] || continue
    name=$(basename "$f")
    compare_pair "project/$name" "$f" "$ALL_DIR/$name"
done

# Terms outputs: *.sex.tsv
for f in "$SERIAL_DIR"/*.sex.tsv; do
    [[ -e "$f" ]] || continue
    name=$(basename "$f")
    compare_pair "terms/$name" "$f" "$ALL_DIR/$name"
done

if [[ "$status" -eq 0 ]]; then
    echo
    echo "[parity] PASS — all compared outputs are byte-identical."
else
    echo
    echo "[parity] FAIL — see mismatches above."
fi
exit "$status"
