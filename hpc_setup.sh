#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIMS_DIR="$REPO_ROOT/sims"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
USER_BIN="${USER_BIN:-$HOME/bin}"
PIP_USER_BIN="$HOME/.local/bin"

export PATH="$USER_BIN:$PIP_USER_BIN:$PATH"
export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-API}"
export PYTHONUNBUFFERED=1

log() { printf '[hpc_setup] %s\n' "$*"; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

maybe_load_modules() {
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck source=/dev/null
    source /etc/profile.d/modules.sh || true
  fi
  if has_cmd module; then
    export MODULES_PAGER=cat
    local rmods=(
      "${R_MODULE:-}"
      "R/4.3.0-openblas"
      "R/4.2.2-gcc-8.2.0-vp7tyde"
      "R/4.2.2-openblas"
      "r/4.2.2-gcc-8.2.0-vp7tyde"
    )
    local loaded_r=0
    local m
    for m in "${rmods[@]}"; do
      [[ -z "$m" ]] && continue
      if module load "$m" >/dev/null 2>&1; then
        if Rscript -e 'library(mgcv)' >/dev/null 2>&1; then
          loaded_r=1
          log "Loaded working R module: $m"
          break
        fi
        module unload "$m" >/dev/null 2>&1 || true
      fi
    done
    if [[ "$loaded_r" -eq 0 ]]; then
      log "WARNING: could not load an R module via environment modules"
    fi
    module load plink/1.90b6.10 >/dev/null 2>&1 || true
    module load plink/1.90b6 >/dev/null 2>&1 || true
  fi
}

ensure_python() {
  if ! has_cmd "$PYTHON_BIN"; then
    log "ERROR: $PYTHON_BIN not found. Set PYTHON_BIN to a valid interpreter."
    exit 1
  fi

  if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    log "Bootstrapping pip via ensurepip"
    "$PYTHON_BIN" -m ensurepip --upgrade
  fi

  # Fast path: skip pip work when all required imports already resolve.
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib
mods = [
    "numpy", "pandas", "scipy", "sklearn", "matplotlib",
    "msprime", "stdpopsim", "tskit",
    "seaborn", "tabulate", "h5py", "statsmodels",
    "demes", "demesdraw", "rpy2",
]
for m in mods:
    importlib.import_module(m)
PY
  then
    log "Python dependencies already installed; skipping pip install"
    return
  fi

  log "Installing Python dependencies"
  "$PYTHON_BIN" -m pip install --user --upgrade 'pip<25' setuptools wheel
  "$PYTHON_BIN" -m pip install --user \
    "numpy<2" "pandas<3" scipy scikit-learn matplotlib \
    "msprime==1.3.4" "stdpopsim==0.3.0" "tskit<1" \
    pgscatalog-calc seaborn tabulate h5py statsmodels demes demesdraw rpy2
}

ensure_r_mgcv() {
  if ! has_cmd Rscript; then
    log "ERROR: Rscript not found. Load an R module (set R_MODULE=...) and rerun."
    exit 1
  fi

  if has_cmd R; then
    export R_HOME="$(R RHOME)"
    export LD_LIBRARY_PATH="$R_HOME/lib:${LD_LIBRARY_PATH:-}"
  fi

  Rscript -e 'library(mgcv)' >/dev/null

  "$PYTHON_BIN" - <<'PY'
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
importr("mgcv")
print("rpy2+mgcv OK")
PY
}

install_plink2() {
  if has_cmd plink2; then
    log "Found plink2: $(command -v plink2)"
    return
  fi

  log "Installing plink2 into $USER_BIN"
  mkdir -p "$USER_BIN"
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  curl -fsSL -o "$tmp/plink2.zip" https://s3.amazonaws.com/plink2-assets/plink2_linux_x86_64_latest.zip
  unzip -q -o "$tmp/plink2.zip" -d "$tmp"
  cp "$tmp/plink2" "$USER_BIN/plink2"
  chmod +x "$USER_BIN/plink2"

  log "Installed plink2: $USER_BIN/plink2"
}

install_gctb() {
  if has_cmd gctb; then
    log "Found gctb: $(command -v gctb)"
    return
  fi

  log "Installing gctb into $USER_BIN"
  mkdir -p "$USER_BIN"
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' RETURN

  curl -fsSL -o "$tmp/gctb.zip" https://cnsgenomics.com/software/gctb/download/gctb_2.5.4_Linux.zip
  unzip -q -o "$tmp/gctb.zip" -d "$tmp"
  cp "$tmp/gctb_2.5.4_Linux/gctb" "$USER_BIN/gctb"
  chmod +x "$USER_BIN/gctb"

  log "Installed gctb: $USER_BIN/gctb"
}

run_main() {
  local -a main_args
  if [[ "$#" -eq 0 ]]; then
    main_args=(run --figure both)
  else
    main_args=("$@")
  fi

  log "Running sims/main.py ${main_args[*]}"
  exec "$PYTHON_BIN" -u "$SIMS_DIR/main.py" "${main_args[@]}"
}

main() {
  maybe_load_modules
  ensure_python
  ensure_r_mgcv
  install_plink2
  install_gctb

  log "Tool check: python=$(command -v "$PYTHON_BIN")"
  log "Tool check: plink2=$(command -v plink2 || true)"
  log "Tool check: gctb=$(command -v gctb || true)"
  log "Tool check: Rscript=$(command -v Rscript || true)"

  run_main "$@"
}

main "$@"
