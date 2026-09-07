#!/usr/bin/env bash
# Compare standalone score/project/terms with `gnomon all` using matching,
# caller-provided inputs. Keep results for inspection; never accept failed
# commands or missing output as evidence of parity.
# Usage: gnomon_all_parity.sh GNOMON FIXTURE_VCF SCORE MODEL WORK_DIR BUILD
set -euo pipefail
shopt -s nullglob

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 GNOMON FIXTURE_VCF SCORE MODEL WORK_DIR BUILD" >&2
    echo "Provide a VCF and score with overlap in the specified cached model; BUILD is 37 or 38." >&2
    exit 2
fi

absolute_path() {
    local parent
    parent=$(cd "$(dirname "$1")" && pwd)
    printf '%s/%s\n' "$parent" "$(basename "$1")"
}

GNOMON_BIN=$(command -v "$1")
GNOMON_BIN=$(absolute_path "$GNOMON_BIN")
FIXTURE=$(absolute_path "$2")
SCORE=$(absolute_path "$3")
MODEL=$4
WORK=$5
BUILD=$6
[[ -x "$GNOMON_BIN" && -s "$FIXTURE" && -e "$SCORE" ]]
[[ "$BUILD" == 37 || "$BUILD" == 38 ]]

mkdir -p "$WORK"
WORK=$(cd "$WORK" && pwd)
SERIAL_DIR="$WORK/serial"
ALL_DIR="$WORK/all"
# Existing directories are an error: stale output must never satisfy a check.
mkdir "$SERIAL_DIR" "$ALL_DIR"
cp "$FIXTURE" "$SERIAL_DIR/fixture.vcf"
cp "$FIXTURE" "$ALL_DIR/fixture.vcf"

echo "[parity] serial baseline"
(cd "$SERIAL_DIR" && "$GNOMON_BIN" score "$SCORE" fixture.vcf --build "$BUILD")
(cd "$SERIAL_DIR" && "$GNOMON_BIN" project fixture.vcf --model "$MODEL" --build "$BUILD")
(cd "$SERIAL_DIR" && "$GNOMON_BIN" terms --sex fixture.vcf --build "$BUILD")

echo "[parity] unified pipeline"
(cd "$ALL_DIR" && "$GNOMON_BIN" all "$SCORE" fixture.vcf --model "$MODEL" --build "$BUILD")

compare_group() {
    local suffix=$1 name output
    local serial=("$SERIAL_DIR"/*"$suffix")
    local unified=("$ALL_DIR"/*"$suffix")
    if [[ ${#serial[@]} -eq 0 || ${#serial[@]} -ne ${#unified[@]} ]]; then
        echo "[parity] FAIL: $suffix requires matching nonempty output sets (serial=${#serial[@]}, all=${#unified[@]})." >&2
        exit 1
    fi
    for output in "${serial[@]}"; do
        name=$(basename "$output")
        if [[ ! -s "$output" || ! -s "$ALL_DIR/$name" ]]; then
            echo "[parity] FAIL: missing or empty output $name." >&2
            exit 1
        fi
        if ! cmp -s "$output" "$ALL_DIR/$name"; then
            echo "[parity] FAIL: output differs: $name (results retained in $WORK)." >&2
            exit 1
        fi
        echo "[parity] byte-identical: $name"
    done
}

compare_group .sscore
compare_group projection_scores.bin
compare_group projection_scores.metadata.json
compare_group sex.tsv
echo "[parity] PASS: all required outputs are present and byte-identical."
