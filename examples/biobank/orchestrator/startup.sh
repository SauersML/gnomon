#!/bin/bash
mkdir -p /var/orch
docker run --rm --network host -v /var/orch:/w gcr.io/google.com/cloudsdktool/google-cloud-cli:slim bash -c "gcloud storage cp gs://aou-train-work-wb-amiable-carrot-1173/orchestrator/orch.sh /w/orch.sh -q"
chmod +x /var/orch/orch.sh
nohup bash /var/orch/orch.sh > /var/orch/orch.log 2>&1 &
