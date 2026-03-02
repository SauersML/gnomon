#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SIMS_DIR="$REPO_ROOT/sims"
SELF_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
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

log() { printf '[hpc_run] %s\n' "$*"; }

has_cmd() { command -v "$1" >/dev/null 2>&1; }

_cgroup_memory_events_path() {
  local pid="$1"
  [[ -r "/proc/$pid/cgroup" ]] || return 1
  local cgv2_path
  cgv2_path="$(awk -F: '$1=="0" && $2=="" {print $3; exit}' "/proc/$pid/cgroup" 2>/dev/null || true)"
  [[ -n "$cgv2_path" ]] || return 1
  local events="/sys/fs/cgroup${cgv2_path}/memory.events"
  [[ -r "$events" ]] || return 1
  printf '%s\n' "$events"
}

_oom_kill_count() {
  local events_file="${1:-}"
  [[ -n "$events_file" && -r "$events_file" ]] || {
    printf '0\n'
    return 0
  }
  awk '$1=="oom_kill" {print $2; found=1} END{if (!found) print 0}' "$events_file" 2>/dev/null
}

_disk_space_line() {
  local path="$1"
  df -Pk "$path" 2>/dev/null | awk -v p="$path" 'NR==2{printf "[hpc_run] disk-space path=%s size_kb=%s used_kb=%s avail_kb=%s use=%s mount=%s\n", p, $2, $3, $4, $5, $6}'
}

_inode_space_line() {
  local path="$1"
  df -Pi "$path" 2>/dev/null | awk -v p="$path" 'NR==2{printf "[hpc_run] inode-space path=%s inodes=%s iused=%s ifree=%s iuse=%s mount=%s\n", p, $2, $3, $4, $5, $6}'
}

_cgroup_memory_value() {
  local events_file="${1:-}"
  local key="${2:-}"
  [[ -n "$events_file" && -r "$events_file" && -n "$key" ]] || {
    printf 'NA\n'
    return 0
  }
  local cg_dir
  cg_dir="$(dirname "$events_file")"
  local f="$cg_dir/$key"
  if [[ -r "$f" ]]; then
    head -n 1 "$f" 2>/dev/null || printf 'NA\n'
  else
    printf 'NA\n'
  fi
}

_memory_events_line() {
  local events_file="${1:-}"
  [[ -n "$events_file" && -r "$events_file" ]] || return 0
  local line
  line="$(tr '\n' ' ' < "$events_file" 2>/dev/null | sed 's/[[:space:]]\+/ /g' | sed 's/ $//')"
  [[ -n "$line" ]] && printf '[hpc_run] cgroup-memory-events %s\n' "$line"
}

_cpu_total_available() {
  if has_cmd nproc; then
    nproc
    return 0
  fi
  if has_cmd getconf; then
    getconf _NPROCESSORS_ONLN
    return 0
  fi
  printf '1\n'
}

_read_cpu_jiffies() {
  [[ -r /proc/stat ]] || return 1
  awk '
    $1=="cpu" {
      idle=$5+$6;
      total=0;
      for (i=2;i<=NF;i++) total+=$i;
      printf "%s %s\n", total, idle;
      exit
    }
  ' /proc/stat
}

_cpu_util_pct_from_delta() {
  local prev_total="$1"
  local prev_idle="$2"
  local curr_total="$3"
  local curr_idle="$4"
  awk -v pt="$prev_total" -v pi="$prev_idle" -v ct="$curr_total" -v ci="$curr_idle" '
    BEGIN {
      dt = ct - pt;
      di = ci - pi;
      if (dt <= 0) { print "NA"; exit }
      util = ((dt - di) / dt) * 100.0;
      if (util < 0) util = 0;
      if (util > 100) util = 100;
      printf "%.2f", util;
    }
  '
}

_append_abnormal_diagnostics() {
  local log_file="$1"
  local events_file="${2:-}"
  {
    printf '[hpc_run] abnormal-termination-diagnostics begin\n'
    _disk_space_line "$RESULTS_DIR" || true
    _inode_space_line "$RESULTS_DIR" || true
    _disk_space_line "$LOG_DIR" || true
    _inode_space_line "$LOG_DIR" || true
    _disk_space_line "/tmp" || true
    _inode_space_line "/tmp" || true
    if [[ -d "/dev/shm" ]]; then
      _disk_space_line "/dev/shm" || true
      _inode_space_line "/dev/shm" || true
    fi
    _memory_events_line "$events_file" || true
    printf '[hpc_run] cgroup-memory-current=%s\n' "$(_cgroup_memory_value "$events_file" "memory.current")"
    printf '[hpc_run] cgroup-memory-max=%s\n' "$(_cgroup_memory_value "$events_file" "memory.max")"
    printf '[hpc_run] cgroup-memory-peak=%s\n' "$(_cgroup_memory_value "$events_file" "memory.peak")"
    printf '[hpc_run] abnormal-termination-diagnostics end\n'
  } >>"$log_file"
}

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
    "numba",
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
    numba \
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

if [[ "${1:-}" == "_run-main" ]]; then
  shift
  run_main "$@"
fi

if [[ "$#" -ne 0 ]]; then
  echo "Usage: $0"
  echo "Run with zero args; this starts the detached full run."
  exit 2
fi

if is_running; then
  echo "A run is already active (pid=$(cat "$PID_FILE"))."
  exit 0
fi

local_args=(run --figure both)
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
  # Re-enter this script in internal mode to avoid drift with helper filenames.
  printf 'bash %q _run-main "${args[@]}"\n' "$SELF_SCRIPT"
  echo 'rc=$?'
  printf 'echo "$rc" > %q\n' "$EXIT_FILE"
  echo 'exit "$rc"'
} >"$WRAP_FILE"
chmod +x "$WRAP_FILE"

nohup setsid "$WRAP_FILE" >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"
echo "$LOG_FILE" >"$LOG_DIR/latest.log.path"
echo "$EXIT_FILE" >"$LOG_DIR/latest.exit.path"

# Sidecar monitor: if the run is externally killed (OOM, timeout, preemption),
# write a clear marker to the log and synthesize an exit code file.
(
  run_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$run_pid" ]] || exit 0

  heartbeat_s=60
  total_cpus="$(_cpu_total_available)"
  run_started_ts="$(date +%s)"
  last_hb_ts="$run_started_ts"
  prev_total=""
  prev_idle=""
  if read -r prev_total prev_idle < <(_read_cpu_jiffies 2>/dev/null); then
    :
  else
    prev_total=""
    prev_idle=""
  fi

  events_file="$(_cgroup_memory_events_path "$run_pid" || true)"
  oom_before="$(_oom_kill_count "$events_file")"

  while kill -0 "$run_pid" 2>/dev/null; do
    sleep 5
    now_ts="$(date +%s)"

    curr_total=""
    curr_idle=""
    cpu_system_util_pct="NA"
    if read -r curr_total curr_idle < <(_read_cpu_jiffies 2>/dev/null); then
      if [[ -n "$prev_total" && -n "$prev_idle" ]]; then
        cpu_system_util_pct="$(_cpu_util_pct_from_delta "$prev_total" "$prev_idle" "$curr_total" "$curr_idle")"
      fi
      prev_total="$curr_total"
      prev_idle="$curr_idle"
    fi

    if (( now_ts - last_hb_ts >= heartbeat_s )); then
      elapsed_s=$((now_ts - run_started_ts))
      cpu_proc_pct="$(ps -p "$run_pid" -o %cpu= 2>/dev/null | awk '{print $1; exit}' || true)"
      cpu_proc_pct="${cpu_proc_pct:-NA}"
      cpu_threads="$(ps -p "$run_pid" -o nlwp= 2>/dev/null | awk '{print $1; exit}' || true)"
      cpu_threads="${cpu_threads:-NA}"
      cpu_system_used_est="NA"
      cpu_proc_used_est="NA"
      if [[ "$cpu_system_util_pct" != "NA" ]]; then
        cpu_system_used_est="$(awk -v pct="$cpu_system_util_pct" -v total="$total_cpus" 'BEGIN{printf "%.2f", (pct/100.0)*total}')"
      fi
      if [[ "$cpu_proc_pct" != "NA" ]]; then
        cpu_proc_used_est="$(awk -v pct="$cpu_proc_pct" 'BEGIN{printf "%.2f", pct/100.0}')"
      fi
      printf '[hpc_run] heartbeat elapsed_s=%s pid=%s cpu_total=%s cpu_system_util_pct=%s cpu_system_used_est=%s cpu_proc_pct=%s cpu_proc_used_est=%s proc_threads=%s\n' \
        "$elapsed_s" "$run_pid" "$total_cpus" "$cpu_system_util_pct" "$cpu_system_used_est" "$cpu_proc_pct" "$cpu_proc_used_est" "$cpu_threads" >>"$LOG_FILE"
      last_hb_ts="$now_ts"
    fi
  done

  if [[ ! -f "$EXIT_FILE" ]]; then
    oom_after="$(_oom_kill_count "$events_file")"
    if [[ "$oom_after" =~ ^[0-9]+$ && "$oom_before" =~ ^[0-9]+$ && "$oom_after" -gt "$oom_before" ]]; then
      printf '[hpc_run] Run ended without normal exit file; cgroup oom_kill increased (%s -> %s), likely OOM kill\n' "$oom_before" "$oom_after" >>"$LOG_FILE"
    else
      printf '[hpc_run] Run ended without normal exit file; likely external kill (OOM/timeout/preemption)\n' >>"$LOG_FILE"
    fi
    _append_abnormal_diagnostics "$LOG_FILE" "$events_file"
    printf '137\n' >"$EXIT_FILE"
  fi
  rm -f "$PID_FILE"
) >/dev/null 2>&1 &

sleep 1
if ! is_running; then
  echo "Failed to start detached run. Check log: $LOG_FILE"
  exit 1
fi
echo "Started detached run."
echo "  pid: $(cat "$PID_FILE")"
echo "  log: $LOG_FILE"
