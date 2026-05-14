#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

command -v gnomon >/dev/null 2>&1 || bash "$HOME/gnomon/install.sh"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
RESULTS_DIR="$HOME/aou-gpu-baremetal/biobank_results"
mkdir -p "$RESULTS_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS="$RESULTS_DIR/biobank_linear_${TS}.log"

{
  echo "=========================================================================="
  echo "biobank PC-residualized linear PRS baseline"
  echo "=========================================================================="
  echo "timestamp_utc:    $TS"
  echo "host:             $(hostname)"
  echo "git_rev:          $(git -C "$HOME/gnomon" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  echo "git_subject:      $(git -C "$HOME/gnomon" log -1 --format=%s 2>/dev/null || echo unknown)"
  echo "script:           $SCRIPT_DIR/pc_residualized_linear.py"
  echo "results_file:     $RESULTS"
  echo
  echo "--- env ---"
  echo "WORKSPACE_CDR:    ${WORKSPACE_CDR:-<unset>}"
  echo "GOOGLE_PROJECT:   ${GOOGLE_PROJECT:-<unset>}"
  echo "=========================================================================="
  echo
} | tee "$RESULTS"

uv run \
    --python 3.11 \
    --with numpy \
    --with pandas \
    --with pyarrow \
    --with scipy \
    --with scikit-learn \
    --with lifelines \
    --with tzdata \
    --with google-cloud-bigquery \
    --with google-cloud-bigquery-storage \
    --with db-dtypes \
    -- python -u "$SCRIPT_DIR/pc_residualized_linear.py" 2>&1 | tee -a "$RESULTS"

{
  echo
  echo "=========================================================================="
  echo "=== SUMMARY (extracted)"
  echo "=========================================================================="
  grep -E "^=== |^  snomed=|^  split:|^  fit_spec:|^  coef:|^  PGS=|^  held-out" "$RESULTS" \
      || echo "(no summary lines matched)"
  echo "=========================================================================="
  echo "Full log: $RESULTS"
} | tee -a "$RESULTS"
