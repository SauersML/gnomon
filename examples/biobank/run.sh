#!/usr/bin/env bash
# Bootstrap uv (if missing) and run the AoU marginal-slope disease script
# with all required Python dependencies resolved on the fly.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"

uv run \
    --python 3.11 \
    --with gamfit \
    --with numpy \
    --with pandas \
    --with pyarrow \
    --with scipy \
    --with scikit-learn \
    --with google-cloud-bigquery \
    --with db-dtypes \
    "$SCRIPT_DIR/marginal_slope_diseases.py" "$@"
