version 1.0

# Reduce a completed pilot's aggregate metrics file to fixed-form result
# tokens inside the workspace. Every emitted token is an aggregate over at
# least the configured minimum number of participants; small counts,
# identifiers and free text never leave.
workflow aou_results_digest {
  input {
    File metrics
    File identity_guard
    String runtime_image
  }
  call digest { input: metrics=metrics, identity_guard=identity_guard, runtime_image=runtime_image }
  output { Array[File] tokens = digest.digest }
}

task digest {
  input {
    File metrics
    File identity_guard
    String runtime_image
  }
  command <<<
    set -euo pipefail
    cp "~{identity_guard}" aou_identity.py
    timeout --kill-after=5s 60s python - "~{metrics}" <<'PY'
    import json
    import math
    from pathlib import Path
    import re
    import sys
    from aou_identity import task_account

    task_account()
    MINIMUM_COUNT = 20
    ALLOWED = ("n", "observed_disease_events", "brier", "brier_standard_error", "ipcw_auc",
               "mean_predicted_risk", "ipcw_observed_risk", "mean_risk_discrepancy",
               "brier_difference", "brier_difference_standard_error", "auc_difference")

    def token(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("digest values must be finite numbers")
        text = str(value) if isinstance(value, int) else f"{value:.4g}"
        return text.replace("-", "m").replace("+", "p")

    def slug(text):
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
        if not cleaned:
            raise ValueError("empty digest label")
        return cleaned[:40]

    def emit(parts):
        Path("digest__" + "__".join(parts) + ".txt").write_text("\n")

    def metric_rows(rows, stage):
        for row in rows:
            base = [stage, slug(row["group"]), "h" + token(float(row["horizon"]))]
            if row.get("status") != "ok" or int(row.get("n", 0)) < MINIMUM_COUNT:
                emit(base + ["insufficient_support"])
                continue
            for key in ALLOWED:
                if key not in row:
                    continue
                if key == "observed_disease_events" and int(row[key]) < MINIMUM_COUNT:
                    continue
                emit(base + [key, token(row[key])])

    def report_rows(report, stage):
        for variant, model in sorted((report.get("models") or {}).items()):
            if "cif_grid_error" in model:
                emit([stage, slug(variant), "cif_grid_error", token(float(model["cif_grid_error"]))])
            metric_rows(model.get("metrics", []), f"{stage}__{slug(variant)}")
        metric_rows(report.get("incremental") or [], f"{stage}__incremental")

    results = json.loads(Path(sys.argv[1]).read_text())
    if len(results) > 8:
        raise ValueError("digest covers at most eight endpoints")
    for disease, result in results.items():
        name = slug(disease)
        for pgs, report in (result.get("development") or {}).items():
            report_rows(report, f"{name}__development__{slug(pgs)}")
        report_rows(result, f"{name}__final")
        emit([name, "status", slug(result.get("status", "completed"))])
    PY
  >>>
  output { Array[File] digest = glob("digest__*.txt") }
  runtime {
    docker: runtime_image
    cpu: 2
    memory: "2 GiB"
    cpuPlatform: "AMD Rome"
    zones: "us-central1-a us-central1-b us-central1-c us-central1-f"
    disks: "local-disk 10 SSD"
    preemptible: 3
    maxRetries: 0
  }
}
