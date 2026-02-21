#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIMS_DIR="$REPO_ROOT/sims"
RESULTS_DIR="$SIMS_DIR/results_hpc"
LOG_DIR="$RESULTS_DIR/logs"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
USER_BIN="${USER_BIN:-$HOME/bin}"
PIP_USER_BIN="$HOME/.local/bin"

export PATH="$USER_BIN:$PIP_USER_BIN:$PATH"
export RPY2_CFFI_MODE="${RPY2_CFFI_MODE:-API}"
export PYTHONUNBUFFERED=1

mkdir -p "$LOG_DIR"

PID_FILE="$LOG_DIR/full_run.pid"
LOG_FILE="$LOG_DIR/full_run_$(date +%Y%m%d_%H%M%S).log"

cmd="${1:-start}"
shift || true

log() { printf '[hpc_run] %s\n' "$*"; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  rm -f "$PID_FILE"
  return 1
}

find_live_pipeline_pid() {
  # Match actual python pipeline process, not incidental shell command strings.
  ps -eo pid=,comm=,args= \
    | awk '$2 ~ /^python/ && $0 ~ /sims\/main.py([[:space:]]|$)/ {print $1; exit}' \
    || true
}

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

  maybe_load_modules
  ensure_python
  ensure_r_mgcv
  install_plink2
  install_gctb

  log "Tool check: python=$(command -v "$PYTHON_BIN")"
  log "Tool check: plink2=$(command -v plink2 || true)"
  log "Tool check: gctb=$(command -v gctb || true)"
  log "Tool check: Rscript=$(command -v Rscript || true)"

  log "Running sims/main.py ${main_args[*]}"
  exec "$PYTHON_BIN" -u "$SIMS_DIR/main.py" "${main_args[@]}"
}

case "$cmd" in
  _run-main)
    run_main "$@"
    ;;
  start)
    if is_running; then
      echo "A run is already active (pid=$(cat "$PID_FILE"))."
      exit 0
    fi
    # Default to full Figure1+Figure2 run. Extra args are forwarded to the internal runner.
    local_args=("$@")
    if [[ "${#local_args[@]}" -eq 0 ]]; then
      local_args=(run --figure both)
    fi
    ts="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="$LOG_DIR/full_run_${ts}.log"
    EXIT_FILE="$LOG_DIR/full_run_${ts}.exitcode"
    WRAP_FILE="$LOG_DIR/full_run_${ts}.sh"
    {
      echo '#!/usr/bin/env bash'
      echo 'set -uo pipefail'
      printf 'cd %q\n' "$REPO_ROOT"
      printf 'args=('
      for a in "${local_args[@]}"; do
        printf ' %q' "$a"
      done
      echo ' )'
      printf '%q _run-main "${args[@]}"\n' "$SIMS_DIR/hpc_run_detached.sh"
      echo 'rc=$?'
      printf 'echo "$rc" > %q\n' "$EXIT_FILE"
      echo 'exit "$rc"'
    } >"$WRAP_FILE"
    chmod +x "$WRAP_FILE"

    nohup setsid "$WRAP_FILE" >"$LOG_FILE" 2>&1 < /dev/null &
    echo $! >"$PID_FILE"
    echo "$LOG_FILE" >"$LOG_DIR/latest.log.path"
    echo "$EXIT_FILE" >"$LOG_DIR/latest.exit.path"
    sleep 1
    if ! is_running; then
      echo "Failed to start detached run. Check log: $LOG_FILE"
      exit 1
    fi
    echo "Started detached run."
    echo "  pid: $(cat "$PID_FILE")"
    echo "  log: $LOG_FILE"
    ;;
  status)
    if is_running; then
      echo "RUNNING pid=$(cat "$PID_FILE") (tracked)"
    else
      live_pid="$(find_live_pipeline_pid)"
      if [[ -n "$live_pid" ]]; then
        echo "RUNNING pid=$live_pid (untracked)"
      else
        exit_path="$(cat "$LOG_DIR/latest.exit.path" 2>/dev/null || true)"
        if [[ -n "$exit_path" && -f "$exit_path" ]]; then
          code="$(cat "$exit_path" 2>/dev/null || true)"
          echo "NOT RUNNING (last exit code: ${code:-unknown})"
        else
          echo "NOT RUNNING"
        fi
      fi
    fi
    ;;
  attach-status)
    live_pid="$(find_live_pipeline_pid)"
    if [[ -n "$live_pid" ]]; then
      echo "RUNNING pid=$live_pid"
    else
      echo "NOT RUNNING"
    fi
    ;;
  tail)
    latest_log="$(cat "$LOG_DIR/latest.log.path" 2>/dev/null || true)"
    if [[ -z "$latest_log" || ! -f "$latest_log" ]]; then
      latest_log="$(ls -1t "$LOG_DIR"/full_run_*.log 2>/dev/null | head -n1 || true)"
    fi
    if [[ -z "$latest_log" ]]; then
      echo "No log files found in $LOG_DIR"
      exit 1
    fi
    tail -n 80 "$latest_log"
    ;;
  stop)
    if is_running; then
      pid="$(cat "$PID_FILE")"
      kill "$pid" || true
      echo "Stopped pid=$pid"
      rm -f "$PID_FILE"
    else
      echo "No running detached job."
    fi
    ;;
  *)
    echo "Usage: $0 {start|status|attach-status|tail|stop} [sim args...]"
    exit 2
    ;;
esac
