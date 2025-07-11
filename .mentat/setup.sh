#!/bin/bash

# Install Rust nightly and set project override
echo "Installing Rust nightly..."
rustup install nightly
rustup override set nightly

# Install OpenBLAS development libraries
echo "Installing OpenBLAS development libraries..."
apt-get update
apt-get install -y libopenblas-dev

# Install Python dependencies for testing
echo "Installing Python dependencies..."
pip3 install pandas numpy requests psutil tabulate polars pyarrow gmpy2

# Install Lean toolchain
echo "Installing Lean toolchain..."
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y
export PATH="$HOME/.elan/bin:$PATH"

# Build Rust project
echo "Building Rust project..."
cargo build --release

# Build Lean proofs
echo "Building Lean proofs..."
lake update
lake build
