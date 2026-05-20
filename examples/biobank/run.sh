#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

if [ "$#" -ne 0 ]; then
  echo "run.sh takes no arguments; it always runs the full biobank validation." >&2
  exit 2
fi

# --- kill any older biobank run still in flight -----------------------------
# Match by the python entrypoint, not by run.sh -- prior runs may still be
# blocking on a long gamfit fit even if their wrapper shell has exited.
if [ -z "${GNOMON_RUN_REEXEC:-}" ]; then
  MY_PID=$$
  PIDS_TO_KILL=$(pgrep -f "python.*examples/biobank/marginal_slope_diseases\.py" 2>/dev/null \
    | grep -v "^${MY_PID}$" || true)
  if [ -n "$PIDS_TO_KILL" ]; then
    echo "[run.sh] killing prior marginal_slope_diseases.py processes: $PIDS_TO_KILL" >&2
    kill $PIDS_TO_KILL 2>/dev/null || true
    sleep 2
    for pid in $PIDS_TO_KILL; do
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
  fi
fi

# --- self-update: pull latest, then re-exec the refreshed script ------------
# `exec bash "$0"` reloads run.sh from disk *and* re-imports the python script
# fresh on each python launch, so the latest code is what actually runs.
if [ -z "${GNOMON_RUN_REEXEC:-}" ] && git -C "$SCRIPT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
  echo "[run.sh] git pull --ff-only in $REPO_ROOT" >&2
  if git -C "$REPO_ROOT" pull --ff-only; then
    export GNOMON_RUN_REEXEC=1
    exec bash "$0" "$@"
  else
    echo "[run.sh] git pull failed; continuing with on-disk code" >&2
  fi
fi

command -v gnomon >/dev/null 2>&1 || bash "$HOME/gnomon/install.sh"
AOU_DISK_ROOT="$HOME/aou-gpu-baremetal"
RESULTS_DIR="$AOU_DISK_ROOT/biobank_results"
RUN_STATE_DIR="$AOU_DISK_ROOT/gnomon_runtime/biobank"
UV_CACHE_DIR="$RUN_STATE_DIR/uv"
UV_PYTHON_INSTALL_DIR="$RUN_STATE_DIR/uv_python"
XDG_CACHE_HOME="$RUN_STATE_DIR/xdg_cache"
XDG_DATA_HOME="$RUN_STATE_DIR/xdg_data"
TMPDIR="$RUN_STATE_DIR/tmp"
MIN_RUN_STATE_FREE_GIB=30
MIN_RUN_STATE_FREE_INODES=200000
mkdir -p "$RESULTS_DIR" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$XDG_CACHE_HOME" "$XDG_DATA_HOME" "$TMPDIR"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR XDG_CACHE_HOME XDG_DATA_HOME TMPDIR
TS=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS="$RESULTS_DIR/biobank_run_${TS}.log"

if ! command -v uv >/dev/null 2>&1; then
  echo "[run.sh] uv is not on PATH after PATH=$PATH" | tee -a "$RESULTS" >&2
  exit 127
fi

path_usage() {
  local label="$1"
  local path="$2"
  local existing="$path"
  while [ ! -e "$existing" ] && [ "$existing" != "/" ]; do
    existing="$(dirname "$existing")"
  done
  if [ -e "$existing" ]; then
    printf '%-18s path=%s\n' "$label:" "$path"
    df -hP "$existing" | sed 's/^/  df: /'
    df -ihP "$existing" 2>/dev/null | sed 's/^/  inode: /' || true
  else
    printf '%-18s path=%s (not found)\n' "$label:" "$path"
  fi
}

available_kib() {
  local path="$1"
  df -Pk "$path" | awk 'NR == 2 {print $4}'
}

available_inodes() {
  local path="$1"
  df -Pi "$path" | awk 'NR == 2 {print $4}'
}

format_kib() {
  awk -v kib="$1" 'BEGIN {printf "%.1fGiB", kib / 1024 / 1024}'
}

require_run_state_capacity() {
  local label="$1"
  local path="$2"
  local required_gib="$3"
  local required_inodes="$4"
  local free_kib
  local free_inodes
  free_kib="$(available_kib "$path" 2>/dev/null || true)"
  free_inodes="$(available_inodes "$path" 2>/dev/null || true)"
  if [[ ! "$free_kib" =~ ^[0-9]+$ ]]; then
    echo "[run.sh] could not determine free space for $label at $path" | tee -a "$RESULTS" >&2
    failure_diagnostics 1
  fi
  local required_kib=$((required_gib * 1024 * 1024))
  if [ "$free_kib" -lt "$required_kib" ]; then
    {
      echo "[run.sh] insufficient free space for $label at $path"
      echo "[run.sh] required: ${required_gib}GiB; available: $(format_kib "$free_kib")"
      df -hP "$path"
    } | tee -a "$RESULTS" >&2
    failure_diagnostics 1
  fi
  if [[ "$free_inodes" =~ ^[0-9]+$ ]] && [ "$free_inodes" -lt "$required_inodes" ]; then
    {
      echo "[run.sh] insufficient free inodes for $label at $path"
      echo "[run.sh] required: ${required_inodes}; available: ${free_inodes}"
      df -ihP "$path"
    } | tee -a "$RESULTS" >&2
    failure_diagnostics 1
  fi
}

cache_du() {
  local label="$1"
  local path="$2"
  if [ -e "$path" ]; then
    printf '%-18s ' "$label:"
    du -sh "$path" 2>/dev/null || true
  else
    printf '%-18s missing %s\n' "$label:" "$path"
  fi
}

failure_diagnostics() {
  local exit_code="$1"
  trap - ERR
  {
    echo
    echo "=========================================================================="
    echo "=== FAILURE DIAGNOSTICS"
    echo "=========================================================================="
    echo "exit_code:         $exit_code"
    echo "timestamp_utc:     $(date -u +%Y%m%dT%H%M%SZ)"
    echo
    echo "--- filesystem at failure ---"
    path_usage "home" "$HOME"
    path_usage "aou_disk" "$AOU_DISK_ROOT"
    path_usage "results" "$RESULTS_DIR"
    path_usage "run_state" "$RUN_STATE_DIR"
    path_usage "uv_cache" "$UV_CACHE_DIR"
    path_usage "uv_python" "$UV_PYTHON_INSTALL_DIR"
    path_usage "xdg_cache" "$XDG_CACHE_HOME"
    path_usage "xdg_data" "$XDG_DATA_HOME"
    path_usage "tmpdir" "$TMPDIR"
    path_usage "system_tmp" "/tmp"
    echo
    echo "--- cache sizes ---"
    cache_du "legacy_uv_cache" "$HOME/.cache/uv"
    cache_du "legacy_gam_cache" "$HOME/.cache/gam"
    cache_du "uv_cache" "$UV_CACHE_DIR"
    cache_du "uv_python" "$UV_PYTHON_INSTALL_DIR"
    cache_du "xdg_cache" "$XDG_CACHE_HOME"
    cache_du "gamfit_cache" "$XDG_CACHE_HOME/gam"
    cache_du "xdg_data" "$XDG_DATA_HOME"
    cache_du "tmpdir" "$TMPDIR"
    echo "=========================================================================="
  } | tee -a "$RESULTS" >&2
  exit "$exit_code"
}

trap 'failure_diagnostics $?' ERR

# `uv run --with ...` pulls several large wheels, including CUDA runtime
# packages, and writes temporary files inside the client cache.  AoU home
# directories can have much less free space than the attached workspace disk,
# so keep both uv and XDG-style caches under RESULTS_DIR and fail before the
# resolver starts if that volume is not large enough.
require_run_state_capacity \
  "biobank run state" \
  "$RUN_STATE_DIR" \
  "$MIN_RUN_STATE_FREE_GIB" \
  "$MIN_RUN_STATE_FREE_INODES"

# --- preamble ---------------------------------------------------------------
{
  echo "=========================================================================="
  echo "biobank marginal-slope PRS validation run"
  echo "=========================================================================="
  echo "timestamp_utc:    $TS"
  echo "host:             $(hostname)"
  echo "user:             $(whoami)"
  echo "git_rev:          $(git -C "$HOME/gnomon" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_subject:      $(git -C "$HOME/gnomon" log -1 --format=%s 2>/dev/null || echo unknown)"
  echo "gnomon_bin:       $(command -v gnomon)"
  echo "gnomon_version:   $(gnomon --version 2>/dev/null | head -1 || echo unknown)"
  echo "uv_bin:           $(command -v uv)"
  echo "uv_version:       $(uv --version 2>/dev/null || echo unknown)"
  echo "uv_cache_dir:     $UV_CACHE_DIR"
  echo "uv_effective_cache: $(uv cache dir 2>/dev/null || echo unknown)"
  echo "uv_python_dir:    $UV_PYTHON_INSTALL_DIR"
  echo "xdg_cache_home:   $XDG_CACHE_HOME"
  echo "xdg_data_home:    $XDG_DATA_HOME"
  echo "tmpdir:           $TMPDIR"
  echo "run_state_dir:    $RUN_STATE_DIR"
  echo "script:           $SCRIPT_DIR/marginal_slope_diseases.py"
  echo "results_file:     $RESULTS"
  echo
  echo "--- filesystem ---"
  path_usage "home" "$HOME"
  path_usage "results" "$RESULTS_DIR"
  path_usage "run_state" "$RUN_STATE_DIR"
  path_usage "uv_cache" "$UV_CACHE_DIR"
  path_usage "uv_python" "$UV_PYTHON_INSTALL_DIR"
  path_usage "xdg_cache" "$XDG_CACHE_HOME"
  path_usage "xdg_data" "$XDG_DATA_HOME"
  path_usage "tmpdir" "$TMPDIR"
  path_usage "system_tmp" "/tmp"
  echo
  echo "--- fit configuration (from script) ---"
  grep -E '^(NUM_PCS|DUCHON_CENTERS|TRAIN_FRACTION|RNG_SEED|MAX_LOSO_CARE_SITES|MIN_LOSO_|BOOTSTRAP_) *=' \
      "$SCRIPT_DIR/marginal_slope_diseases.py"
  echo
  echo "--- diseases ---"
  awk '/^DISEASES = \{/,/^\}/' "$SCRIPT_DIR/marginal_slope_diseases.py"
  echo
  echo "--- env ---"
  echo "WORKSPACE_CDR:    ${WORKSPACE_CDR:-<unset>}"
  echo "GOOGLE_PROJECT:   ${GOOGLE_PROJECT:-<unset>}"
  echo "=========================================================================="
  echo
} | tee "$RESULTS"

# --- the actual run ---------------------------------------------------------
uv --cache-dir "$UV_CACHE_DIR" run \
    --python 3.11 \
    --refresh-package gamfit \
    --upgrade-package gamfit \
    --with gamfit \
    --with numpy \
    --with pandas \
    --with pyarrow \
    --with scipy \
    --with scikit-learn \
    --with scikit-survival \
    --with statsmodels \
    --with patsy \
    --with tzdata \
    --with pgscatalog.calc \
    --with gcsfs \
    --with google-cloud-bigquery \
    --with google-cloud-bigquery-storage \
    --with db-dtypes \
    --with nvidia-cuda-nvrtc-cu12 \
    --with nvidia-cuda-runtime-cu12 \
    --with nvidia-cublas-cu12 \
    --with nvidia-cusolver-cu12 \
    --with nvidia-cusparse-cu12 \
    --with nvidia-curand-cu12 \
    --with nvidia-nvjitlink-cu12 \
    -- python -u "$SCRIPT_DIR/marginal_slope_diseases.py" 2>&1 | tee -a "$RESULTS"

# --- extract just the summary lines ----------------------------------------
{
  echo
  echo "=========================================================================="
  echo "=== SUMMARY (extracted)"
  echo "=========================================================================="
  grep -E "^gamfit |^loso_axes:|^=== |^cohort:|^  pcs:|^  sex:|^  pgs:|^  times:|^  context:|^  ancestry:|^  snomed=|^  split:|^  fit_spec:|^  binary_fit_spec:|^  baseline_spec:|^  baselineA|^  baselineB|^  binary_baseline|^  PGS=|^  binary PGS=|^  OOD:|^  LOSO |^  GAM |^  baseline |^  binaryGAM |^  binaryN |^  binaryA |^  binaryB |^  delta |^  binary_delta |^  survival_metrics |^  RESULT |^  BINARY_RESULT |^  save:" "$RESULTS" || echo "(no summary lines matched — fit likely failed; see full log above)"
  echo "=========================================================================="
  echo "Full log: $RESULTS"
} | tee -a "$RESULTS"
