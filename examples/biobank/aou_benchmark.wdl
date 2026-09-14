version 1.0

# Benchmark polygenic score methods on ever-diagnosed EHR outcomes inside the
# AoU workspace, from already cached scores. Only aggregate result tokens over
# at least the reporting minimum leave the task.
workflow aou_benchmark {
  input {
    File sources
    File config
    File wheelhouse_archive
    File ancestry_predictions
    File relatedness_prune
    String features_uri
    Array[File] score_files
    String runtime_image
    String status_uri
  }
  call bench { input:
    sources=sources, config=config, wheelhouse_archive=wheelhouse_archive,
    ancestry_predictions=ancestry_predictions, relatedness_prune=relatedness_prune,
    features_uri=features_uri, score_files=score_files, runtime_image=runtime_image,
    status_uri=status_uri
  }
  output { Array[File] tokens = bench.tokens }
}

task bench {
  input {
    File sources
    File config
    File wheelhouse_archive
    File ancestry_predictions
    File relatedness_prune
    String features_uri
    Array[File] score_files
    String runtime_image
    String status_uri
  }
  command <<<
    set -euo pipefail
    mkdir -p wheels work/tmp work/cache work/scores work/bench digest
    tar -xf "~{sources}"
    python aou_status.py "~{status_uri}" benchmark_started
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
    for score in ~{sep=" " score_files}; do ln -s "$score" work/scores/; done
    export TMPDIR="$PWD/work/tmp" XDG_CACHE_HOME="$PWD/work/cache"
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
    tar -xf "~{wheelhouse_archive}" -C wheels
    timeout --kill-after=10s 60s python -m venv work/venv
    timeout --kill-after=10s 180s work/venv/bin/python -m pip install --disable-pip-version-check \
      --no-compile --no-cache-dir --no-index --only-binary=:all: --find-links wheels -r aou_requirements.txt
    if ! timeout --kill-after=10s 40m work/venv/bin/python aou_benchmark.py run \
        --config "~{config}" --ancestry "~{ancestry_predictions}" --prune "~{relatedness_prune}" \
        --features-uri "~{features_uri}" --scores work/scores --status-uri "~{status_uri}" \
        --output digest --work work/bench; then
      python aou_status.py "~{status_uri}" failed_benchmark
      exit 1
    fi
  >>>
  output { Array[File] tokens = glob("digest/digest__*.txt") }
  runtime {
    docker: runtime_image
    cpu: 16
    memory: "32 GiB"
    cpuPlatform: "AMD Rome"
    zones: "us-central1-a us-central1-b us-central1-c us-central1-f"
    disks: "local-disk 50 SSD"
    preemptible: 3
    maxRetries: 0
  }
}
