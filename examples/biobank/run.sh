#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

command -v gnomon >/dev/null 2>&1 || bash "$HOME/gnomon/install.sh"

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
RESULTS="$HOME/aou-gpu-baremetal/biobank_run_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "$(dirname "$RESULTS")"

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
    -- python -u "$SCRIPT_DIR/marginal_slope_diseases.py" 2>&1 | tee "$RESULTS"

echo
echo "=========================================================================="
echo "=== RESULTS FILE: $RESULTS"
echo "=========================================================================="
cat "$RESULTS"
