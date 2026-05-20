#!/usr/bin/env bash
# Minimal reproducer for the mid-run SIGABRT in `gnomon score`.
#
# Reported behaviour:
#   > CUDA progress: 303104/310448 (97.6%)
#   > CUDA progress: 306176/310448 (98.6%)
#   double free or corruption (!prev)
#   <SIGABRT>
#
# This script runs ONE `gnomon score` call with ONE PGS and reports the
# exit code. Nothing else. Edit the variables at the top if your paths
# differ; everything is plain CLI args, no env vars.
#
# Usage:
#   bash examples/repro_score_sigabrt.sh
#
# Optional diagnostic flags:
#   --mcheck      MALLOC_CHECK_=3 — glibc aborts at the first sign of
#                 corruption instead of much later, so the abort points
#                 closer to the offending malloc/free.
#   --no-tcache   GLIBC_TUNABLES tcache_count=0 — disables glibc's per-
#                 thread cache, which sometimes hides double-frees.
#   --scribble    GLIBC_TUNABLES perturb=170 — fills freed memory so
#                 use-after-free fails loudly.
#
# These three flags are the only knobs and they set glibc tunables on
# the child process only.

set -u

# Edit these if your paths differ. No env vars.
GNOMON=gnomon
PLINK_PREFIX="$HOME/.sv_pgs_cache/aou_array_plink/arrays"
PGS=PGS004536   # one score, smallest case that still hits the dense pipeline

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

echo "gnomon score $PGS $PLINK_PREFIX"
[[ ${#EXTRA_ENV[@]} -gt 0 ]] && echo "extra env: ${EXTRA_ENV[*]}"
start=$EPOCHSECONDS
env "${EXTRA_ENV[@]}" "$GNOMON" score "$PGS" "$PLINK_PREFIX"
rc=$?
echo "exit=$rc elapsed=$(( EPOCHSECONDS - start ))s"
if [[ $rc -eq 134 ]]; then
    echo "(134 = 128 + SIGABRT — matches the reported crash)"
fi
exit "$rc"
