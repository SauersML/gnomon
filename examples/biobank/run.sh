#!/usr/bin/env bash
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

# Build gnomon from this checkout so the panic-safe CUDA fallback in
# score/cuda_backend.rs is what actually runs. Incremental rebuilds
# are sub-second when nothing changed.
command -v cargo >/dev/null 2>&1 \
    || curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal
(cd "$HOME/gnomon" && cargo build --release --features 'map score calibrate terms' --bin gnomon)
export GNOMON_BIN="$HOME/gnomon/target/release/gnomon"

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
