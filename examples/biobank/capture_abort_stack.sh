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

# --- Get a working gdb -------------------------------------------------------
# The AoU Jupyter image ships no gdb. Conda's cache is unwritable for
# the jupyter@ user, and the Ubuntu .deb depends on libbabeltrace +
# libsource-highlight + libipt + libdebuginfod which aren't installed.
# Use the truly-static x86_64 build from github.com/guyush1/gdb-static
# instead — it bundles every dependency into one self-contained binary.
GDB_BIN=""
if command -v gdb >/dev/null 2>&1; then
  GDB_BIN="$(command -v gdb)"
  echo "[capture] using system gdb at $GDB_BIN" >&2
fi

if [ -z "$GDB_BIN" ]; then
  STATIC_GDB_URL="https://github.com/guyush1/gdb-static/releases/download/v17.1-static/gdb-static-slim-x86_64.tar.gz"
  echo "[capture] downloading static gdb from $STATIC_GDB_URL" >&2
  mkdir -p "$WORK_DIR/gdb-static"
  if ! curl -fsSL --retry 3 --connect-timeout 10 "$STATIC_GDB_URL" \
      -o "$WORK_DIR/gdb-static.tar.gz"; then
    echo "[capture] could not download static gdb tarball" >&2
    exit 1
  fi
  tar -xzf "$WORK_DIR/gdb-static.tar.gz" -C "$WORK_DIR/gdb-static" || {
    echo "[capture] static gdb tarball did not extract" >&2
    exit 1
  }
  # The tarball lays out gdb/gdbserver binaries somewhere inside;
  # grep for the gdb executable.
  GDB_BIN="$(find "$WORK_DIR/gdb-static" -maxdepth 5 -type f -name 'gdb' -perm -u+x 2>/dev/null | head -n 1)"
  if [ -z "$GDB_BIN" ]; then
    GDB_BIN="$(find "$WORK_DIR/gdb-static" -maxdepth 5 -type f -name 'gdb' 2>/dev/null | head -n 1)"
  fi
  if [ -z "$GDB_BIN" ] || [ ! -f "$GDB_BIN" ]; then
    echo "[capture] static gdb tarball did not contain a gdb binary; contents:" >&2
    find "$WORK_DIR/gdb-static" -maxdepth 5 -type f 2>/dev/null | head -20 >&2
    exit 1
  fi
  chmod +x "$GDB_BIN"
fi

echo "[capture] using gdb: $GDB_BIN" >&2
# Probe gdb without piping into head — `set -o pipefail` + head closing
# the pipe early would SIGPIPE gdb and make this look like a failure
# even when gdb is healthy. Capture full output and grep for the
# version string instead.
GDB_VERSION_OUT="$("$GDB_BIN" --version 2>&1)" || true
if ! printf '%s\n' "$GDB_VERSION_OUT" | head -1 | grep -q 'GNU gdb'; then
  echo "[capture] gdb refuses to run; output was:" >&2
  printf '%s\n' "$GDB_VERSION_OUT" | head -10 >&2
  exit 1
fi
printf '%s\n' "$GDB_VERSION_OUT" | head -1 >&2

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
