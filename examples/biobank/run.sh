#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

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
    "$(dirname -- "${BASH_SOURCE[0]}")/marginal_slope_diseases.py"
