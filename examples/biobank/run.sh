#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

for arg in "$@"; do
  case "$arg" in
    --reexecuted) ;;
    *)
      echo "run.sh takes no arguments; it always runs the full biobank validation." >&2
      exit 2
      ;;
  esac
done

REEXEC_FLAG="--reexecuted"
REEXECUTED=0
for arg in "$@"; do
  [ "$arg" = "$REEXEC_FLAG" ] && REEXECUTED=1
done

# --- kill any older biobank run still in flight -----------------------------
# Match by the python entrypoint, not by run.sh -- prior runs may still be
# blocking on a long gamfit fit even if their wrapper shell has exited.
if [ "$REEXECUTED" -eq 0 ]; then
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
# `exec bash "$0" --reexecuted` reloads run.sh from disk so the latest
# code is what actually runs. The `--reexecuted` flag (not an env var)
# prevents an infinite re-exec loop.
if [ "$REEXECUTED" -eq 0 ] && git -C "$SCRIPT_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
  echo "[run.sh] git pull --ff-only in $REPO_ROOT" >&2
  if git -C "$REPO_ROOT" pull --ff-only; then
    exec bash "$0" "$REEXEC_FLAG"
  else
    echo "[run.sh] git pull failed; continuing with on-disk code" >&2
  fi
fi

# --- Install the scorer ------------------------------------------------------
# Download the most recently published linux-x64 release binary directly
# from the GitHub Releases API and install it to ~/.local/bin/gnomon.
# This path requires no cargo / rustc toolchain on the runner. We do not
# pin to the checked-out HEAD SHA — releases are published asynchronously
# (~20 min after each push while the build workflow runs), and pinning to
# an unpublished SHA hard-fails. Accepting the latest published release
# is the right trade-off for a runner that doesn't carry a Rust build
# toolchain.
INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR" "$HOME/.local/share/gnomon"
RELEASE_API="https://api.github.com/repos/SauersML/gnomon/releases/latest"
RELEASE_JSON="$(curl -sL --retry 5 --retry-delay 2 --connect-timeout 5 --max-time 30 "$RELEASE_API")"
RELEASE_TAG="$(printf '%s' "$RELEASE_JSON" \
  | sed -n 's/^[[:space:]]*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' \
  | head -n 1)"
ASSET_URL="$(printf '%s' "$RELEASE_JSON" \
  | grep -F 'browser_download_url' \
  | grep -F 'gnomon-linux-x64-v3.tar.gz' \
  | cut -d '"' -f 4 \
  | head -n 1)"
if [ -z "$ASSET_URL" ]; then
  ASSET_URL="$(printf '%s' "$RELEASE_JSON" \
    | grep -F 'browser_download_url' \
    | grep -F 'gnomon-linux-x64.tar.gz' \
    | cut -d '"' -f 4 \
    | head -n 1)"
fi
if [ -z "$ASSET_URL" ]; then
  echo "[run.sh] could not find a linux-x64 asset on the latest release" >&2
  exit 1
fi
echo "[run.sh] installing gnomon release $RELEASE_TAG from $ASSET_URL" >&2
TMP_INSTALL="$(mktemp -d)"
trap 'rm -rf "$TMP_INSTALL"' EXIT
curl -fsSL --retry 5 --retry-delay 2 "$ASSET_URL" -o "$TMP_INSTALL/gnomon.tar.gz"
tar -xzf "$TMP_INSTALL/gnomon.tar.gz" -C "$TMP_INSTALL"
INSTALLED_BIN="$(find "$TMP_INSTALL" -maxdepth 3 -type f -name gnomon -perm -u+x | head -n 1)"
if [ -z "$INSTALLED_BIN" ]; then
  echo "[run.sh] extracted release did not contain a gnomon binary" >&2
  exit 1
fi
install -m 0755 "$INSTALLED_BIN" "$INSTALL_DIR/gnomon"
printf '%s\n' "$RELEASE_TAG" > "$HOME/.local/share/gnomon/installed-release"
git -C "$HOME/gnomon" rev-parse HEAD > "$HOME/.local/share/gnomon/installed-sha"
hash -r
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

mount_of() {
  local path="$1"
  while [ ! -e "$path" ] && [ "$path" != "/" ]; do
    path="$(dirname "$path")"
  done
  df -P "$path" 2>/dev/null | awk 'NR == 2 {print $NF}'
}

same_mount() {
  local left
  local right
  left="$(mount_of "$1")"
  right="$(mount_of "$2")"
  [ -n "$left" ] && [ "$left" = "$right" ]
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
      # If a pre-redirection uv cache shares this filesystem, point at it
      # explicitly: this run uses UV_CACHE_DIR=$RUN_STATE_DIR/uv, so anything
      # under $HOME/.cache/uv is genuine stale state and is safe to drop.
      if [ -d "$HOME/.cache/uv" ] && same_mount "$HOME/.cache/uv" "$path"; then
        echo
        echo "[run.sh] legacy uv cache at \$HOME/.cache/uv shares this filesystem:"
        du -sh "$HOME/.cache/uv" 2>/dev/null || true
        echo "[run.sh] this run does not use it (UV_CACHE_DIR=$UV_CACHE_DIR); free it with:"
        echo "[run.sh]     UV_CACHE_DIR=\"\$HOME/.cache/uv\" uv cache clean"
      fi
      echo
      echo "[run.sh] largest entries under \$HOME/.cache (cleanup candidates):"
      du -sh "$HOME"/.cache/* 2>/dev/null | sort -hr | head -10 || true
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
  echo "gnomon_head_sha:  $(git -C "$HOME/gnomon" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "uv_bin:           $(command -v uv)"
  echo "uv_version:       $(uv --version)"
  echo "uv_default_cache: $(env -u UV_CACHE_DIR uv cache dir 2>/dev/null || echo unknown)"
  echo "uv_cache_dir:     $UV_CACHE_DIR"
  echo "uv_effective_cache: $(uv cache dir 2>/dev/null || echo unknown)"
  echo "uv_python_dir:    $UV_PYTHON_INSTALL_DIR"
  echo "xdg_cache_home:   $XDG_CACHE_HOME"
  echo "xdg_data_home:    $XDG_DATA_HOME"
  echo "tmpdir:           $TMPDIR"
  echo "run_state_dir:    $RUN_STATE_DIR"
  echo "min_run_state_free: ${MIN_RUN_STATE_FREE_GIB}GiB and ${MIN_RUN_STATE_FREE_INODES} inodes"
  echo "script:           $SCRIPT_DIR/marginal_slope_diseases.py"
  echo "results_file:     $RESULTS"
  echo
  echo "--- storage mechanism ---"
  echo "uv defaults its client cache to HOME/.cache/uv; UV_CACHE_DIR moves all uv cache writes."
  echo "uv-managed Python installs are moved with UV_PYTHON_INSTALL_DIR."
  echo "gamfit warm-starts use XDG_CACHE_HOME/gam; XDG_CACHE_HOME moves the warm-start cache."
  echo "TMPDIR keeps wheel/build temporary files on the same runtime disk."
  echo
  echo "--- filesystem ---"
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

# `uv run --with ...` pulls several large wheels, including CUDA runtime
# packages, and writes temporary files inside the client cache. AoU home
# directories can have much less free space than the attached workspace disk,
# so keep uv, uv-managed Python, XDG-style caches, and temp files under
# RUN_STATE_DIR and fail before the resolver starts if that volume is tight.
{
  echo "--- uv cache maintenance ---"
  if ! uv cache prune; then
    echo "[run.sh] warning: uv cache prune failed; continuing to capacity check"
  fi
  echo
} 2>&1 | tee -a "$RESULTS"

require_run_state_capacity \
  "biobank run state" \
  "$RUN_STATE_DIR" \
  "$MIN_RUN_STATE_FREE_GIB" \
  "$MIN_RUN_STATE_FREE_INODES"

# --- the actual run ---------------------------------------------------------
# UV_CACHE_DIR is exported above instead of repeated as a flag. The original
# failure happened before Python started because uv used its default
# HOME/.cache/uv client cache; the env var is authoritative for every uv cache
# path in this process and any child uv process.
uv run \
    --no-project \
    --python 3.11 \
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
