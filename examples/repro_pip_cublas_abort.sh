#!/usr/bin/env bash
# Show that the AoU CUDA heap-corruption abort can be reproduced
# without gnomon, just by mixing pip-wheel CUDA libs with the system
# CUDA driver.
#
# Runs examples/repro_pip_cublas_abort.py twice:
#
#   1. "system-first": LD_LIBRARY_PATH unchanged. The loader picks the
#      system /usr/local/cuda/lib64 cuBLAS, matching the driver. We
#      expect this to run to completion.
#
#   2. "wheels-first": LD_LIBRARY_PATH gets the pip nvidia-*-cu12
#      wheel lib dirs prepended. The loader picks the pip cuBLAS,
#      which is a different patch version than the kernel-pinned
#      libcuda. We expect "double free or corruption (!prev)" or a
#      similar abort at teardown.
#
# If wheels-first aborts and system-first doesn't, the ABI-mismatch
# hypothesis is confirmed and the gnomon fix (don't inject pip lib
# dirs into LD_LIBRARY_PATH) is correct.

set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/repro_pip_cublas_abort.py"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[repro] python3 not on PATH" >&2
  exit 1
fi

# Discover pip-wheel nvidia lib dirs. If the wheels aren't installed
# there's nothing to test on the wheels-first side.
NV_LIBS="$(python3 - <<'PY' 2>/dev/null || true
import sys
try:
    import nvidia
except ImportError:
    sys.exit(0)
from pathlib import Path
parts = []
for parent in nvidia.__path__:
    for child in Path(parent).iterdir():
        lib = child / "lib"
        if lib.is_dir():
            parts.append(str(lib))
print(":".join(parts))
PY
)"

SYSTEM_LD="${LD_LIBRARY_PATH:-}"

run_one() {
  local label="$1"
  local ld="$2"
  echo
  echo "=========================================================="
  echo "[$label] LD_LIBRARY_PATH=$ld"
  echo "=========================================================="
  LD_LIBRARY_PATH="$ld" python3 "$SCRIPT" "$label"
  local rc=$?
  echo "[$label] exit_code=$rc"
  return $rc
}

run_one "system-first" "$SYSTEM_LD"
SYS_RC=$?

if [ -z "$NV_LIBS" ]; then
  echo
  echo "[repro] no nvidia-*-cu12 wheels found on this Python; cannot"
  echo "[repro] run the wheels-first variant. (pip install"
  echo "[repro] nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 if you"
  echo "[repro] want to test the bad path.)"
  exit "$SYS_RC"
fi

run_one "wheels-first" "$NV_LIBS:$SYSTEM_LD"
WHEELS_RC=$?

echo
echo "=========================================================="
echo "[repro] summary"
echo "=========================================================="
echo "[repro] system-first exit: $SYS_RC"
echo "[repro] wheels-first exit: $WHEELS_RC"
if [ "$SYS_RC" -eq 0 ] && [ "$WHEELS_RC" -ne 0 ]; then
  echo "[repro] HYPOTHESIS CONFIRMED: pip-wheel cuBLAS shadowing the"
  echo "[repro] system cuBLAS reliably aborts at teardown, while the"
  echo "[repro] same script with system libs runs clean."
elif [ "$SYS_RC" -eq 0 ] && [ "$WHEELS_RC" -eq 0 ]; then
  echo "[repro] both variants ran clean — either the wheel and system"
  echo "[repro] versions are compatible on this host, or the gnomon"
  echo "[repro] abort is not the pure pip-wheel-shadowing mode."
elif [ "$SYS_RC" -ne 0 ] && [ "$WHEELS_RC" -ne 0 ]; then
  echo "[repro] both variants failed — neither library set is healthy"
  echo "[repro] on this host. Check the per-variant logs above."
else
  echo "[repro] only system-first failed; that's the opposite of the"
  echo "[repro] hypothesis. Investigate what's wrong with the system"
  echo "[repro] CUDA libs."
fi
