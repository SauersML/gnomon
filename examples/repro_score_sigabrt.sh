#!/usr/bin/env bash
# Minimal reproducer for the in-flight SIGABRT in `gnomon score` on the AoU
# array PLINK fileset.
#
# Reported behaviour:
#   > CUDA progress: 303104/310448 (97.6%)
#   > CUDA progress: 306176/310448 (98.6%)
#   double free or corruption (!prev)
#   <SIGABRT>
#
# The earlier abort we fixed was an *at-exit* teardown problem in
# cudarc/cuBLAS — this one is different: it fires while CUDA progress is
# still climbing, so the heap is being corrupted by the running pipeline
# itself, not by destructor ordering. The fix in cli/main.rs (`_exit` to
# skip atexit handlers) does NOT help this case; we need the heap
# corruption itself caught.
#
# Run with:
#   bash examples/repro_score_sigabrt.sh
#
# Useful env knobs (override on the command line, e.g.
# `GNOMON_REPRO_MODE=mcheck bash examples/repro_score_sigabrt.sh`):
#
#   GNOMON_BIN          path to the gnomon binary (defaults to `gnomon` on PATH)
#   PLINK_PREFIX        PLINK 1.9 prefix (.bed/.bim/.fam without extension)
#   PGS_ARG             comma-separated PGS Catalog IDs
#   TRIALS              how many times to loop the score call
#   GNOMON_REPRO_MODE   diagnostic to layer on:
#                         off       - run plain (default)
#                         mcheck    - glibc malloc consistency checks
#                         scribble  - fill freed memory with garbage to fail
#                                     loud on use-after-free
#                         tcache_off- disable glibc per-thread cache (a known
#                                     "smoke screen" over heap bugs)
#                         tcache_v  - print tcache verbose info on abort

set -u

: "${GNOMON_BIN:=gnomon}"
: "${PLINK_PREFIX:=$HOME/.sv_pgs_cache/aou_array_plink/arrays}"
: "${PGS_ARG:=PGS004536,PGS001320,PGS005331}"
: "${TRIALS:=5}"
: "${GNOMON_REPRO_MODE:=off}"

case "$GNOMON_REPRO_MODE" in
    off)
        EXTRA_ENV=()
        ;;
    mcheck)
        # MALLOC_CHECK_=3 makes glibc abort *immediately* on the first sign
        # of heap corruption (instead of printing the warning much later
        # and continuing into deeper undefined behaviour). Crash log will
        # then point closer to the actual bug.
        EXTRA_ENV=(MALLOC_CHECK_=3)
        ;;
    scribble)
        # Tunable that fills freed memory with 0x41. Any subsequent
        # read-after-free turns into operating on `AAAA…` which usually
        # produces an obvious deref or NaN downstream.
        EXTRA_ENV=(GLIBC_TUNABLES=glibc.malloc.perturb=170)
        ;;
    tcache_off)
        # The per-thread cache hides some double-frees behind a "valid"
        # cached return; turning it off makes the abort fire at the real
        # call site.
        EXTRA_ENV=(GLIBC_TUNABLES=glibc.malloc.tcache_count=0)
        ;;
    tcache_v)
        EXTRA_ENV=(MALLOC_CHECK_=3 GLIBC_TUNABLES=glibc.malloc.tcache_count=0)
        ;;
    *)
        echo "Unknown GNOMON_REPRO_MODE=$GNOMON_REPRO_MODE"
        exit 2
        ;;
esac

echo "=========================================================="
echo "gnomon score SIGABRT reproducer"
echo "  binary:       $GNOMON_BIN"
echo "  PLINK prefix: $PLINK_PREFIX"
echo "  PGS arg:      $PGS_ARG"
echo "  trials:       $TRIALS"
echo "  diag mode:    $GNOMON_REPRO_MODE  (env extras: ${EXTRA_ENV[*]:-<none>})"
echo "  gnomon ver:   $("$GNOMON_BIN" --version 2>&1 || echo '<unknown>')"
echo "=========================================================="

failures=0
for trial in $(seq 1 "$TRIALS"); do
    echo
    echo "----- trial $trial / $TRIALS -----"
    start=$EPOCHSECONDS
    # `env` injects MALLOC_CHECK_/GLIBC_TUNABLES per-trial so the child
    # gnomon process sees them.
    env "${EXTRA_ENV[@]}" "$GNOMON_BIN" score "$PGS_ARG" "$PLINK_PREFIX"
    rc=$?
    elapsed=$(( EPOCHSECONDS - start ))
    echo "trial $trial: exit=$rc elapsed=${elapsed}s"
    if [[ $rc -ne 0 ]]; then
        failures=$((failures + 1))
        echo
        echo "CRASH on trial $trial (rc=$rc). Captured state:"
        # The crash deposits a .sscore.tmp or partial file next to the PLINK
        # prefix — listing the directory once may show the partial output.
        ls -lah "$(dirname "$PLINK_PREFIX")"/ 2>/dev/null | head -20 || true
        echo
        # On Linux, exit code 134 == 128 + 6 (SIGABRT). Surface that.
        if [[ $rc -eq 134 ]]; then
            echo "exit 134 = 128 + SIGABRT(6) — matches the reported abort."
        fi
        # Don't break — keep looping so we can see whether the crash is
        # deterministic at the same progress fraction or whether it
        # wanders. Reproducer value is higher with the timing data.
    fi
done

echo
echo "=========================================================="
echo "summary: $failures / $TRIALS trials failed"
echo "=========================================================="
exit $(( failures > 0 ? 1 : 0 ))
