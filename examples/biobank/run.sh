#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

command -v gnomon >/dev/null 2>&1 || bash "$HOME/gnomon/install.sh"

uv run \
    --python 3.11 \
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
    -- python -u "$(dirname -- "${BASH_SOURCE[0]}")/marginal_slope_diseases.py"
