#!/bin/bash

# Format Rust code
cargo fmt

# Fix Rust linting issues
cargo clippy --fix --allow-dirty --allow-staged
