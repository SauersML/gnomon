#!/usr/bin/env bash
set -uo pipefail
cd /Users/user/gnomon
args=( run --figure both )
bash /Users/user/gnomon/sims/hpc_run.sh _run-main "${args[@]}"
rc=$?
echo "$rc" > /Users/user/gnomon/sims/results_hpc/logs/full_run_20260302_153554.exitcode
exit "$rc"
