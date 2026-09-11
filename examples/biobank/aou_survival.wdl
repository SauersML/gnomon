version 1.0

# One bounded pilot task. Reuses scores already computed by pgsEngine.
# All data and task outputs stay in the authorized AoU workspace.
workflow aou_survival {
  input {
    File runner
    File disease_selector
    File identity_guard
    File status_code
    File score_transform
    File reference_code
    Array[File] reference_ctn
    File checkpoint_code
    File evaluation_code
    File score_panel
    File requirements
    File analysis_config
    File wheelhouse_archive
    File phenotype_library_archive
    File shared_features_archive
    File ancestry_predictions
    File relatedness_prune
    String runtime_image
    String checkpoint_uri
    File? resume_checkpoint
    Boolean prepare_only = false
    Boolean smoke_only = false
    String endpoint = ""
    Int cpu = 4
    Int memory_gb = 16
    Int disk_gb = 50
    Int wall_minutes = 30
  }

  call analyze {
    input:
      runner = runner,
      disease_selector = disease_selector,
      identity_guard = identity_guard,
      status_code = status_code,
      score_transform = score_transform,
      reference_code = reference_code,
      reference_ctn = reference_ctn,
      checkpoint_code = checkpoint_code,
      evaluation_code = evaluation_code,
      score_panel = score_panel,
      requirements = requirements,
      analysis_config = analysis_config,
      wheelhouse_archive = wheelhouse_archive,
      phenotype_library_archive = phenotype_library_archive,
      shared_features_archive = shared_features_archive,
      ancestry_predictions = ancestry_predictions,
      relatedness_prune = relatedness_prune,
      runtime_image = runtime_image,
      checkpoint_uri = checkpoint_uri,
      resume_checkpoint = resume_checkpoint,
      prepare_only = prepare_only,
      smoke_only = smoke_only,
      endpoint = endpoint,
      cpu = cpu, memory_gb = memory_gb, disk_gb = disk_gb,
      wall_minutes = wall_minutes
  }

  output {
    File metrics = analyze.metrics
    File provenance = analyze.provenance
    File checkpoint = analyze.checkpoint
    # Checkpoints contain participant data and models; workspace storage only.
  }
}

task analyze {
  input {
    File runner
    File disease_selector
    File identity_guard
    File status_code
    File score_transform
    File reference_code
    Array[File] reference_ctn
    File checkpoint_code
    File evaluation_code
    File score_panel
    File requirements
    File analysis_config
    File wheelhouse_archive
    File phenotype_library_archive
    File shared_features_archive
    File ancestry_predictions
    File relatedness_prune
    String runtime_image
    String checkpoint_uri
    File? resume_checkpoint
    Boolean prepare_only
    Boolean smoke_only
    String endpoint
    Int cpu
    Int memory_gb
    Int disk_gb
    Int wall_minutes
  }

  command <<<
    set -euo pipefail
    if [[ "~{prepare_only}" == true && "~{smoke_only}" == true ]]; then
      echo 'prepare_only and smoke_only are mutually exclusive' >&2
      exit 2
    fi
    mkdir -p wheels work/tmp work/cache
    cp "~{runner}" runner.py
    cp "~{disease_selector}" disease_selection.py
    cp "~{identity_guard}" aou_identity.py
    cp "~{status_code}" aou_status.py
    cp "~{score_transform}" aou_score_transform.py
    cp "~{reference_code}" reference_ctn.py
    cp "~{write_json(reference_ctn)}" reference_ctn.json
    cp "~{checkpoint_code}" aou_checkpoint.py
    cp "~{evaluation_code}" aou_evaluation.py
    cp "~{score_panel}" score_panel.json
    # Read WDL strings from JSON instead of interpolating them as shell code.
    cp "~{write_json(runtime_image)}" runtime_image.json
    cp "~{write_json(endpoint)}" endpoint.json
    export TMPDIR="$PWD/work/tmp"
    export XDG_CACHE_HOME="$PWD/work/cache"
    export RAYON_NUM_THREADS=~{cpu}
    export OPENBLAS_NUM_THREADS=1
    export OMP_NUM_THREADS=1
    export PYTHONUNBUFFERED=1
    timeout --kill-after=10s ~{wall_minutes}m bash -euo pipefail -c '
      tar -xf "$1" -C wheels
      python -m venv work/venv
      work/venv/bin/python -m pip install --disable-pip-version-check \
        --no-index --only-binary=:all: --find-links wheels -r "$2"
      resume_args=()
      if [[ -n "$9" ]]; then resume_args=(--resume "$9"); fi
      mode_args=()
      if [[ "${10}" == true ]]; then mode_args=(--prepare-only); fi
      if [[ "${11}" == true ]]; then mode_args=(--smoke-only); fi
      exec work/venv/bin/python runner.py run \
        --config "$3" --phenotypes "$4" --scores "$5" \
        --ancestry "$6" --prune "$7" --output work/results \
        --runtime-image runtime_image.json --checkpoint-uri "$8" \
        --endpoint-config endpoint.json \
        --score-panel score_panel.json \
        --reference-ctn-list reference_ctn.json \
        "${resume_args[@]}" "${mode_args[@]}"
    ' bash "~{wheelhouse_archive}" "~{requirements}" "~{analysis_config}" \
      "~{phenotype_library_archive}" "~{shared_features_archive}" \
      "~{ancestry_predictions}" "~{relatedness_prune}" "~{checkpoint_uri}" \
      "~{default="" resume_checkpoint}" "~{prepare_only}" "~{smoke_only}"
  >>>

  output {
    File metrics = "work/results/metrics.json"
    File provenance = "work/results/provenance.json"
    File checkpoint = "work/checkpoint.tar.gz"
  }

  runtime {
    docker: runtime_image
    cpu: cpu
    memory: "~{memory_gb} GiB"
    disks: "local-disk ~{disk_gb} SSD"
    preemptible: 0
    maxRetries: 0
  }
}
