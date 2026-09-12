version 1.0

# Recompute one eligible cohort's score and run the native survival pilot.
# All genotypes, scores, models and outputs remain in the AoU workspace.
workflow aou_score_training {
  input {
    File sources
    File analysis_config
    File scoring_config
    File scorer_archive
    File score_weights
    File genotype_fam
    String prior_shared_features_uri
    File ancestry_predictions
    File relatedness_prune
    File phenotype_library_archive
    Array[File] reference_ctn
    File wheelhouse_archive
    String runtime_image
    String checkpoint_uri
    File? resume_scoring_checkpoint
  }
  call train { input:
    sources=sources, analysis_config=analysis_config, scoring_config=scoring_config,
    scorer_archive=scorer_archive, score_weights=score_weights, genotype_fam=genotype_fam,
    prior_shared_features_uri=prior_shared_features_uri, ancestry_predictions=ancestry_predictions,
    relatedness_prune=relatedness_prune, phenotype_library_archive=phenotype_library_archive,
    reference_ctn=reference_ctn, wheelhouse_archive=wheelhouse_archive,
    runtime_image=runtime_image, checkpoint_uri=checkpoint_uri,
    resume_scoring_checkpoint=resume_scoring_checkpoint
  }
  output {
    File metrics=train.metrics
    File provenance=train.provenance
    File checkpoint=train.checkpoint
    File scores=train.scores
    File score_manifest=train.score_manifest
  }
}

task train {
  input {
    File sources
    File analysis_config
    File scoring_config
    File scorer_archive
    File score_weights
    File genotype_fam
    String prior_shared_features_uri
    File ancestry_predictions
    File relatedness_prune
    File phenotype_library_archive
    Array[File] reference_ctn
    File wheelhouse_archive
    String runtime_image
    String checkpoint_uri
    File? resume_scoring_checkpoint
  }
  command <<<
    set -euo pipefail
    mkdir -p wheels work/tmp work/cache
    tar -xf "~{sources}"
    python aou_status.py "~{checkpoint_uri}" task_started
    python - "~{checkpoint_uri}" <<'PY'
    import sys
    from aou_identity import require_spot_amd
    from aou_status import publish_status
    try:
        require_spot_amd()
    except Exception:
        publish_status(sys.argv[1], "failed_runtime_policy")
        raise
    publish_status(sys.argv[1], "runtime_verified")
    PY
    tar -xf "~{scorer_archive}"
    chmod +x gnomon-score
    cp "~{write_json(reference_ctn)}" reference_ctn.json
    cp "~{write_json(runtime_image)}" runtime_image.json
    export TMPDIR="$PWD/work/tmp" XDG_CACHE_HOME="$PWD/work/cache"
    export RAYON_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
    cat > work/setup.sh <<'SH'
    set -euo pipefail
    tar -xf "$1" -C wheels
    python -m venv work/venv
    work/venv/bin/python -m pip install --disable-pip-version-check --no-compile --no-cache-dir \
      --no-index --only-binary=:all: --find-links wheels -r aou_requirements.txt
    SH
    timeout --kill-after=10s 20m bash -euo pipefail -c '
      python aou_status.py "${10}" installing_dependencies
      if timeout --kill-after=10s 120s bash work/setup.sh "$1"; then
        python aou_status.py "${10}" dependencies_ready
      else
        setup_rc=$?
        python aou_status.py "${10}" failed_task_setup
        exit "$setup_rc"
      fi
      export GOOGLE_PROJECT=$(work/venv/bin/python -c "import json,sys; print(json.load(open(sys.argv[1]))[\"google_project\"])" "$2")
      work/venv/bin/python -c "import json,sys; json.dump(json.load(open(sys.argv[1]))[\"endpoint\"],open(\"endpoint.json\",\"w\"))" "$3"
      resume_args=()
      if [[ -n "${11}" ]]; then resume_args=(--resume-scoring-checkpoint "${11}"); fi
      work/venv/bin/python aou_refresh_score.py --config "$2" --scoring-config "$3" \
        --scorer "$PWD/gnomon-score" --weights "$4" --fam "$5" --features-uri "$6" \
        --ancestry "$7" --prune "$8" --phenotypes "$9" --score-panel aou_pgs_panel.json \
        --output work/score --status-uri "${10}" "${resume_args[@]}"
      exec work/venv/bin/python aou_survival.py run --config "$2" --phenotypes "$9" \
        --scores work/score/shared_features.tar.gz --ancestry "$7" --prune "$8" \
        --output work/results --runtime-image runtime_image.json --checkpoint-uri "${10}" \
        --endpoint-config endpoint.json --score-panel aou_pgs_panel.json \
        --reference-ctn-list reference_ctn.json --smoke-only --resume-latest
    ' bash "~{wheelhouse_archive}" "~{analysis_config}" "~{scoring_config}" \
      "~{score_weights}" "~{genotype_fam}" "~{prior_shared_features_uri}" \
      "~{ancestry_predictions}" "~{relatedness_prune}" "~{phenotype_library_archive}" "~{checkpoint_uri}" \
      "~{default="" resume_scoring_checkpoint}"
  >>>
  output {
    File metrics="work/results/metrics.json"
    File provenance="work/results/provenance.json"
    File checkpoint="work/checkpoint.tar.gz"
    File scores="work/score/shared_features.tar.gz"
    File score_manifest="work/score/score_manifest.json"
  }
  runtime {
    docker: runtime_image
    cpu: 4
    memory: "16 GiB"
    predefinedMachineType: "n2d-standard-4"
    cpuPlatform: "AMD Milan"
    zones: "us-central1-a us-central1-b us-central1-c us-central1-f"
    disks: "local-disk 50 SSD"
    preemptible: 3
    maxRetries: 0
  }
}
