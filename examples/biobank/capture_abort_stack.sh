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
# The AoU Jupyter image is Ubuntu 22.04 (glibc 2.35) and ships with no
# gdb on PATH. Conda's cache directory is unwritable for our user, so
# `conda install gdb` fails before it can fetch anything. Workaround:
# grab the official Ubuntu `gdb` .deb from archive.ubuntu.com (or its
# mirror), extract it locally with `dpkg-deb -x` — that doesn't need
# root and skips all package-management bookkeeping.
GDB_BIN=""
if command -v gdb >/dev/null 2>&1; then
  GDB_BIN="$(command -v gdb)"
  echo "[capture] using system gdb at $GDB_BIN" >&2
fi

if [ -z "$GDB_BIN" ]; then
  if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "[capture] no gdb on PATH and no dpkg-deb to extract one — bailing." >&2
    exit 1
  fi
  echo "[capture] fetching Ubuntu gdb .deb and extracting into $WORK_DIR/gdb-deb" >&2
  # Pin to a known 22.04 (jammy) build. The actual binary version varies
  # by minor updates; try the latest mirror copy, fall back to a known
  # archive URL.
  GDB_DEB="$WORK_DIR/gdb.deb"
  GDB_LIBC6="$WORK_DIR/libc6.deb"
  # Try the 22.04 ports first (also serves x86_64 via the main archive).
  for url in \
    "http://archive.ubuntu.com/ubuntu/pool/main/g/gdb/gdb_12.1-0ubuntu1~22.04.2_amd64.deb" \
    "http://archive.ubuntu.com/ubuntu/pool/main/g/gdb/gdb_12.1-0ubuntu1~22.04_amd64.deb" \
    "http://mirrors.kernel.org/ubuntu/pool/main/g/gdb/gdb_12.1-0ubuntu1~22.04.2_amd64.deb" \
    ; do
    if curl -fsSL --retry 3 --connect-timeout 10 "$url" -o "$GDB_DEB"; then
      echo "[capture] downloaded gdb deb from $url" >&2
      break
    fi
    rm -f "$GDB_DEB"
  done
  if [ ! -s "$GDB_DEB" ]; then
    echo "[capture] could not download an Ubuntu gdb .deb" >&2
    exit 1
  fi
  mkdir -p "$WORK_DIR/gdb-deb"
  dpkg-deb -x "$GDB_DEB" "$WORK_DIR/gdb-deb"
  GDB_BIN="$WORK_DIR/gdb-deb/usr/bin/gdb"
  if [ ! -x "$GDB_BIN" ]; then
    echo "[capture] extracted .deb did not contain usr/bin/gdb:" >&2
    ls -la "$WORK_DIR/gdb-deb/usr/bin/" 2>/dev/null >&2 || true
    exit 1
  fi
  # gdb depends on libsource-highlight + libgmp + libpython3.10; those
  # are installed on every Ubuntu 22.04 base image. If the binary can't
  # find them we'd error out loudly below.
fi

echo "[capture] using gdb: $GDB_BIN" >&2
"$GDB_BIN" --version 2>&1 | head -1 >&2 || {
  echo "[capture] gdb refuses to run — likely missing a shared library." >&2
  ldd "$GDB_BIN" 2>&1 | grep -v '=>' | head -10 >&2 || true
  ldd "$GDB_BIN" 2>&1 | grep 'not found' >&2 || true
  exit 1
}

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
