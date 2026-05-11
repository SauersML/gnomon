#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
export RUST_LOG="${RUST_LOG:-info}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"

# Make libnvrtc.so dlopen-able so gnomon score's CUDA path can JIT-compile
# kernels. AoU baremetal ships CUDA but does not put it on the default
# ld search path for ad-hoc shells.
nvrtc=$(find /usr/local /usr/lib /opt -maxdepth 6 -name 'libnvrtc.so*' 2>/dev/null | head -1)
export LD_LIBRARY_PATH="$(dirname "${nvrtc:-/dev/null}"):/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

command -v gnomon >/dev/null 2>&1 || bash "$HOME/gnomon/install.sh"

exec 2>&1
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
    -- python -u "$(dirname -- "${BASH_SOURCE[0]}")/marginal_slope_diseases.py"
