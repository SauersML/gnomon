version 1.0

# The whole AoU study in one Spot task: scores, cohort, features, fits,
# predictions, evaluation and the aggregate digest (study.py). Every
# participant-level artifact stays in the workspace: the checkpoint prefix holds
# completed steps so a preempted attempt resumes where it stopped, and only the
# digest's aggregate token names are workflow outputs.
workflow study {
  input {
    File sources
    File config
    File wheelhouse_archive
    File scorer_archive
    Array[File] score_files
    Array[File] score_weights
    File ancestry_predictions
    File relatedness_prune
    String features_uri
    String runtime_image
    String checkpoint_uri
    String status_uri
    String digest_uri
    String looks_uri
    Int cpu = 16
    Int memory_gb = 64
    Int timeout_minutes = 110
    # Fixed labels the digest carries (study.py --caveat), e.g. a known-refused arm.
    Array[String] caveats = []
  }
  call analyze { input:
    sources=sources, config=config, wheelhouse_archive=wheelhouse_archive, scorer_archive=scorer_archive,
    score_files=score_files, score_weights=score_weights,
    ancestry_predictions=ancestry_predictions, relatedness_prune=relatedness_prune,
    features_uri=features_uri, runtime_image=runtime_image, checkpoint_uri=checkpoint_uri,
    status_uri=status_uri, digest_uri=digest_uri, looks_uri=looks_uri, cpu=cpu, memory_gb=memory_gb,
    timeout_minutes=timeout_minutes, caveats=caveats
  }
  output {
    File tokens = analyze.tokens
  }
}

task analyze {
  input {
    File sources
    File config
    File wheelhouse_archive
    File scorer_archive
    Array[File] score_files
    Array[File] score_weights
    File ancestry_predictions
    File relatedness_prune
    String features_uri
    String runtime_image
    String checkpoint_uri
    String status_uri
    String digest_uri
    String looks_uri
    Int cpu
    Int memory_gb
    Int timeout_minutes
    Array[String] caveats
  }
  command <<<
    set -euo pipefail
    mkdir -p wheels work/tmp work/cache work/score_cache
    tar -xf "~{sources}"
    python aou_status.py "~{status_uri}" task_started
    python - "~{status_uri}" <<'PY'
    import sys
    from aou_identity import require_spot_amd
    from aou_status import publish_status
    try:
        require_spot_amd()
    except Exception as error:
        publish_status(sys.argv[1], getattr(error, "label", "failed_runtime_policy"))
        raise
    publish_status(sys.argv[1], "runtime_verified")
    PY
    export TMPDIR="$PWD/work/tmp" XDG_CACHE_HOME="$PWD/work/cache" PYTHONUNBUFFERED=1
    # study.py gives each fit process its own thread budget; the driver itself stays single-threaded.
    export RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    python aou_status.py "~{status_uri}" installing_dependencies
    if timeout --kill-after=10s 180s bash -euo pipefail -c '
        tar -xf "$1" -C wheels
        python -m venv work/venv
        work/venv/bin/python -m pip install --disable-pip-version-check --no-compile --no-cache-dir \
          --no-index --only-binary=:all: --find-links wheels -r study/requirements.txt
        # The wheelhouse build record: study.py reads the gam commit from it.
        cp wheels/PROVENANCE.json work/venv/PROVENANCE.json' bash "~{wheelhouse_archive}"; then
      python aou_status.py "~{status_uri}" dependencies_ready
    else
      rc=$?
      python aou_status.py "~{status_uri}" failed_task_setup
      exit "$rc"
    fi
    # The cached scores the submitter selected, one .sscore per cached study score.
    for score in ~{sep=" " score_files}; do ln -s "$score" work/score_cache/; done
    # The pinned portable scorer and a Catalog scoring file for each uncached score.
    mkdir -p work/scorer
    tar -xf "~{scorer_archive}" -C work/scorer
    scorer=$(find work/scorer -type f -name gnomon-score | head -n 1)
    chmod +x "$scorer"
    weights=()
    for file in ~{sep=" " score_weights}; do
      pgs=$(basename "$file" | grep -oE '^PGS[0-9]{6}')
      weights+=(--input "weights_${pgs}=${file}")
    done
    python aou_status.py "~{status_uri}" reading_projection
    work/venv/bin/python - "~{config}" "~{features_uri}" <<'PY'
    import json, sys
    from aou_identity import task_account
    from aou_projection import source_identity, stream_projection
    project = json.load(open(sys.argv[1]))["data"]["google_project"]
    account = task_account()
    stream_projection(source_identity(sys.argv[2], project, account), "work/projection_pcs.parquet", project, account)
    PY
    python aou_status.py "~{status_uri}" projection_ready
    exec timeout --kill-after=30s ~{timeout_minutes}m work/venv/bin/python study.py run --config "~{config}" --work work/study \
      --input ancestry="~{ancestry_predictions}" --input prune="~{relatedness_prune}" \
      --input projection=work/projection_pcs.parquet --input score_cache=work/score_cache \
      --input scorer="$scorer" "${weights[@]}" \
      --checkpoint "~{checkpoint_uri}" --status-uri "~{status_uri}" --digest-uri "~{digest_uri}" \
      --looks "~{looks_uri}" ~{sep=" " prefix("--caveat ", caveats)}
  >>>
  output {
    File tokens = "work/study/tokens.txt"
  }
  runtime {
    docker: runtime_image
    cpu: cpu
    memory: "~{memory_gb} GiB"
    # c3d (AMD Genoa): half the Spot price per vCPU of n2d (Rome) in us-central1 and faster cores;
    # the task's identity check only needs AMD + preemptible (aou_identity.py).
    cpuPlatform: "AMD Genoa"
    zones: "us-central1-a us-central1-b us-central1-c us-central1-f"
    disks: "local-disk 100 SSD"
    preemptible: 3
    maxRetries: 0
  }
}
