#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO_ROOT/sims/results_hpc"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$LOG_DIR"

PID_FILE="$LOG_DIR/full_run.pid"
LOG_FILE="$LOG_DIR/full_run_$(date +%Y%m%d_%H%M%S).log"

cmd="${1:-start}"
shift || true

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
    | awk '$2 ~ /^python/ && $0 ~ /sims\/main.py run --figure both/ {print $1; exit}' \
    || true
}

case "$cmd" in
  start)
    if is_running; then
      echo "A run is already active (pid=$(cat "$PID_FILE"))."
      exit 0
    fi
    # Default to full Figure1+Figure2 run. Extra args are forwarded to hpc_setup.sh.
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
      printf '%q "${args[@]}"\n' "$REPO_ROOT/hpc_setup.sh"
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
    echo "Usage: $0 {start|status|tail|stop} [hpc_setup args...]"
    exit 2
    ;;
esac
