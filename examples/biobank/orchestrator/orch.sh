#!/bin/bash
# In-perimeter orchestrator: runs on the app VM host as root; every command runs inside the
# google-cloud-cli container with the VM's (pet) service account. Commands arrive as objects
# under orchestrator/cmd/<id>.sh; each runs once; its log is copied to orchestrator/out/<id>/log
# and also emitted as object NAMES (out/<id>/t/<seq>__<base64url chunk>) since the laptop may
# list but not read objects. A heartbeat name orchestrator/hb/<host>__<epoch> is written each cycle.
B=gs://aou-train-work-wb-amiable-carrot-1173
CLI=gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
W=/var/orch; mkdir -p $W/done $W/run; : > $W/empty
H=$(hostname)
run() { docker run --rm --network host -v $W:/w -e B=$B -e H=$H $CLI bash -c "$1"; }
n=0
while true; do
  n=$((n+1))
  # self-update: a newer orch.sh in the bucket replaces this loop in place
  run "cd /w && gcloud storage cp \$B/orchestrator/orch.sh orch.new -q" >/dev/null 2>&1 && [ -s $W/orch.new ] && ! cmp -s $W/orch.new $W/orch.sh && { mv $W/orch.new $W/orch.sh; echo "[orch] reloading $(date -u +%FT%TZ)"; exec bash $W/orch.sh; }
  run "cd /w && printf '' > empty && ( [ \$(( $n % 6 )) -eq 1 ] && gcloud storage cp empty \$B/orchestrator/hb/\${H}__\$(date +%s) -q || true ) && gcloud storage ls \$B/orchestrator/cmd/ 2>/dev/null | sed 's#.*/##' > cmds.txt" || true
  for c in $(cat $W/cmds.txt 2>/dev/null | grep '\.sh$'); do
    id=${c%.sh}; [ -e $W/done/$id ] && continue; touch $W/done/$id
    ( run "cd /w && gcloud storage cp \$B/orchestrator/cmd/$c run/$c -q && bash run/$c > run/$id.log 2>&1; echo \"[orch] rc=\$? \$(date -u +%FT%TZ)\" >> run/$id.log; gcloud storage cp run/$id.log \$B/orchestrator/out/$id/log -q; i=0; base64 -w0 run/$id.log | tr '+/' '-_' | tr -d '=' | fold -w 900 | while read -r chunk; do i=\$((i+1)); gcloud storage cp empty \$B/orchestrator/out/$id/t/\$(printf %04d \$i)__\$chunk -q; done; gcloud storage cp empty \$B/orchestrator/out/$id/DONE -q" || true ) &
  done
  sleep 20
done
