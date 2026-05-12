#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

# --- self-update: pull latest, then re-exec the refreshed script ------------
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
RESULTS_DIR="$HOME/aou-gpu-baremetal/biobank_results"
mkdir -p "$RESULTS_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS="$RESULTS_DIR/biobank_run_${TS}.log"

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
  echo "uv_version:       $(uv --version 2>/dev/null || echo unknown)"
  echo "script:           $SCRIPT_DIR/marginal_slope_diseases.py"
  echo "results_file:     $RESULTS"
  echo
  echo "--- fit configuration (from script) ---"
  grep -E '^(NUM_PCS|DUCHON_CENTERS|TRAIN_FRACTION|RNG_SEED) *=' \
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
uv run \
    --python 3.11 \
    --refresh-package gamfit \
    --upgrade-package gamfit \
    --with gamfit \
    --with numpy \
    --with pandas \
    --with pyarrow \
    --with scipy \
    --with scikit-learn \
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
  grep -E "^gamfit |^=== |^cohort:|^  pcs:|^  sex:|^  pgs:|^  snomed=|^  split:|^  fit_spec:|^  PGS=|^  held-out" "$RESULTS" || echo "(no summary lines matched — fit likely failed; see full log above)"
  echo "=========================================================================="
  echo "Full log: $RESULTS"
} | tee -a "$RESULTS"
