#!/bin/bash
#SBATCH --job-name=anc_calib
#SBATCH --partition=amd512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/anc_calib_%j.out
set -euo pipefail
cd /projects/standard/hsiehph/sauer354/gnomon
export ANC_RUN_SURVIVAL=0          # survival is gam#979-stalled; binary-only run
export TMPDIR=/scratch.global/sauer354/anctmp
mkdir -p "$TMPDIR"
PY=/projects/standard/hsiehph/sauer354/gamfit-sweep-venv/bin/python3
echo "host=$(hostname) start=$(date) head=$(git log --oneline -1)"
exec "$PY" sims/main.py
