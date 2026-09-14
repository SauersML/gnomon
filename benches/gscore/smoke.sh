#!/usr/bin/env bash
# Exactness smoke test for gscore.
#
#   benches/gscore/smoke.sh [work dir]
#
# Builds gscore, writes a small random fixture (make_fixture.py) and checks that every
# kernel (scalar per-call i128, lookup-table, catalog), thread count, partition and I/O
# mode writes byte-identical .sscore files, for native, PGS Catalog and many-score inputs,
# and that a repeat run from the prepared-plan cache matches the first run. With
# SCORE_ORACLE pointing at a score_oracle binary, it also requires the native-file
# output to be within 1e-12 of the exact oracle. Needs a nightly toolchain (portable_simd).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
work=${1:-$(mktemp -d)}
mkdir -p "$work"
cargo build --release --manifest-path "$here/Cargo.toml" ${CARGO_FLAGS:-}
target=${CARGO_TARGET_DIR:-$here/target}
gscore=$target/release/gscore
python3 "$here/make_fixture.py" "$work"

fail() { echo "FAIL: $*" >&2; exit 1; }
run() {  # run <out> <args...>
  local out=$1
  shift
  "$gscore" score "$@" --out "$out" 2> "$out.err" || { cat "$out.err" >&2; fail "gscore exited nonzero: $*"; }
}

for input in native.tsv pgs.txt catalog; do
  ref=$work/ref-$input.sscore
  run "$ref" "$work/$input" "$work/panel" --no-cache --kernel scalar --threads 1
  count=0
  for kernel in lut scalar; do
    for threads in 1 3; do
      for partition in rows people; do
        for io in mmap pread; do
          out=$work/$input-$kernel-t$threads-$partition-$io.sscore
          run "$out" "$work/$input" "$work/panel" --no-cache --kernel "$kernel" --threads "$threads" --partition "$partition" --io "$io"
          cmp -s "$ref" "$out" || fail "$input: $kernel t$threads $partition $io differs from the scalar reference"
          count=$((count + 1))
        done
      done
    done
  done
  mkdir -p "$work/cache-$input"
  run "$work/$input-first.sscore" "$work/$input" "$work/panel" --cache-dir "$work/cache-$input" --threads 2
  run "$work/$input-repeat.sscore" "$work/$input" "$work/panel" --cache-dir "$work/cache-$input" --threads 2
  grep -q "loaded from cache" "$work/$input-repeat.sscore.err" || fail "$input: repeat run did not use the plan cache"
  cmp -s "$ref" "$work/$input-repeat.sscore" || fail "$input: repeat run from the plan cache differs"
  echo "ok $input: $((count + 2)) runs byte-identical to the scalar reference"
done
grep -q "complex applications" "$work/ref-native.tsv.sscore.err" || fail "no summary line"
grep -Eq "[1-9][0-9]* complex applications" "$work/ref-native.tsv.sscore.err" || fail "fixture has no complex loci"

if [ -n "${SCORE_ORACLE:-}" ]; then
  "$SCORE_ORACLE" score "$work/native.tsv" "$work/panel" --out "$work/oracle.sscore" --threads 2 2> "$work/oracle.err"
  "$SCORE_ORACLE" compare "$work/oracle.sscore" "$work/ref-native.tsv.sscore" --tol 1e-12 > "$work/compare.tsv" \
    || { cat "$work/compare.tsv" >&2; fail "native output is not within 1e-12 of score_oracle"; }
  echo "ok score_oracle: $(awk -F'\t' 'NR > 1 && $NF == "PASS"' "$work/compare.tsv" | wc -l) columns PASS"
fi
echo "PASS"
