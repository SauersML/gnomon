#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
python_bin="/projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/python3"
out_dir="${repo_root}/sims/results_hpc/ancestry_calibration"
log_dir="${out_dir}/logs"
mkdir -p "${log_dir}"

log_file="${log_dir}/run_$(date +%Y%m%d_%H%M%S).log"
cd "${repo_root}"
exec "${python_bin}" sims/main.py 2>&1 | tee "${log_file}"
