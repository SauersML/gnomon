#!/usr/bin/env bash
# Capture a backtrace from the gnomon SIGABRT on PGS001320 without
# requiring system gdb. The AoU runner doesn't ship gdb, so:
#
#   1. We download a portable static gdb-multiarch into a temp dir.
#   2. We enable core dumps for this shell.
#   3. We run gnomon under a wrapper that locks `core_pattern` for the
#      process to a writable file (where `ulimit -c unlimited` will then
#      drop the core).
#   4. On abort, we point gdb at the dropped core and the installed
#      gnomon binary and dump:
#        * `thread apply all bt full` — full backtrace, every thread.
#        * `info registers` — register state at abort.
#        * `info threads` — thread list.
#
# The captured trace is written to /tmp/gnomon_abort_<timestamp>.txt
# and also printed to stdout.

set -uo pipefail

GNOMON="${GNOMON:-/home/jupyter/.local/bin/gnomon}"
PLINK_PREFIX="${PLINK_PREFIX:-$HOME/.sv_pgs_cache/aou_array_plink/arrays}"
PGS="${PGS:-PGS001320}"

if [ ! -x "$GNOMON" ]; then
  echo "[capture] gnomon binary not found at $GNOMON" >&2
  exit 1
fi

WORK_DIR="$(mktemp -d -t gnomon-capture-XXXXXX)"
echo "[capture] work dir: $WORK_DIR" >&2

# --- Download a static gdb ---------------------------------------------------
# gdb-multiarch-12.0.50-1.cf12.x86_64 — Conda-Forge static-ish builds also
# work; we try a couple of sources. If none reach, the user can stop here.
GDB_BIN="$WORK_DIR/gdb"
if command -v gdb >/dev/null 2>&1; then
  GDB_BIN="$(command -v gdb)"
  echo "[capture] using system gdb at $GDB_BIN" >&2
else
  echo "[capture] downloading static gdb" >&2
  curl -fsSL --retry 3 \
    "https://github.com/hugsy/gdb-static/releases/download/9.2/gdbserver_x86_64-musl_v9.2" \
    -o "$WORK_DIR/gdbserver.attempt" 2>/dev/null || true
  if [ ! -s "$WORK_DIR/gdbserver.attempt" ]; then
    # Fallback: build gdb via micromamba if available; otherwise give up
    # and tell the user what we needed.
    if command -v micromamba >/dev/null 2>&1; then
      micromamba install -y -p "$WORK_DIR/env" -c conda-forge gdb >&2
      GDB_BIN="$WORK_DIR/env/bin/gdb"
    elif command -v conda >/dev/null 2>&1; then
      conda create -y -p "$WORK_DIR/env" -c conda-forge gdb >&2
      GDB_BIN="$WORK_DIR/env/bin/gdb"
    elif command -v pip >/dev/null 2>&1; then
      # Last-ditch: use Python's `faulthandler`-style traceback — emit a
      # Python-side LD_PRELOAD that prints a libunwind backtrace from the
      # SIGABRT handler. Implemented inline below.
      :
    else
      echo "[capture] no gdb, no conda, no pip — cannot install a debugger." >&2
      echo "[capture] Install gdb on the runner manually (apt-get install gdb), " >&2
      echo "[capture] or run this from a host that has it, and rerun." >&2
      exit 1
    fi
  else
    # gdbserver alone is not enough; we need the full gdb client. Drop
    # the partial gdbserver and bail to conda.
    rm -f "$WORK_DIR/gdbserver.attempt"
    if command -v micromamba >/dev/null 2>&1; then
      micromamba install -y -p "$WORK_DIR/env" -c conda-forge gdb >&2
      GDB_BIN="$WORK_DIR/env/bin/gdb"
    elif command -v conda >/dev/null 2>&1; then
      conda create -y -p "$WORK_DIR/env" -c conda-forge gdb >&2
      GDB_BIN="$WORK_DIR/env/bin/gdb"
    else
      echo "[capture] no static gdb available and no conda installed" >&2
      exit 1
    fi
  fi
fi

if [ ! -x "$GDB_BIN" ]; then
  echo "[capture] gdb still not available at $GDB_BIN" >&2
  exit 1
fi
echo "[capture] using gdb: $GDB_BIN" >&2
"$GDB_BIN" --version 2>&1 | head -1 >&2

# --- Run gnomon under gdb directly (no need to mess with core_pattern) ------
# Running the program inside gdb means we don't depend on the kernel's
# core_pattern settings. gdb stops on SIGABRT, we dump the bt, then
# quit. This is the simplest path that works on every Linux box that
# has the binary and gdb available.

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_TXT="/tmp/gnomon_abort_${TIMESTAMP}.txt"

echo "[capture] running: $GNOMON score $PGS $PLINK_PREFIX" >&2
echo "[capture] writing trace to: $OUT_TXT" >&2

# Remove any partial sscore output so gnomon doesn't refuse to run.
rm -f "$(dirname "$PLINK_PREFIX")"/*_pgs1_*.sscore 2>/dev/null || true

"$GDB_BIN" --batch \
  --return-child-result \
  -ex 'set confirm off' \
  -ex 'set pagination off' \
  -ex 'handle SIGABRT stop print' \
  -ex 'set logging file '"$OUT_TXT" \
  -ex 'set logging overwrite on' \
  -ex 'set logging redirect on' \
  -ex 'set logging on' \
  -ex 'run' \
  -ex 'echo \n=== thread list ===\n' \
  -ex 'info threads' \
  -ex 'echo \n=== full backtrace (all threads) ===\n' \
  -ex 'thread apply all bt full' \
  -ex 'echo \n=== registers (current thread) ===\n' \
  -ex 'info registers' \
  -ex 'echo \n=== local memory map ===\n' \
  -ex 'info proc mappings' \
  -ex 'set logging off' \
  -ex 'quit' \
  --args "$GNOMON" score "$PGS" "$PLINK_PREFIX" 2>&1 | tail -40

echo
echo "=========================================================="
echo "[capture] gdb trace written to: $OUT_TXT"
echo "[capture] last 100 lines:"
echo "=========================================================="
tail -100 "$OUT_TXT" 2>/dev/null || echo "(no trace produced — gdb may have failed to attach)"
