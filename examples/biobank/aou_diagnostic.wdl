version 1.0

# Inspect a failed task inside its workspace. Only fixed diagnostic labels are
# outputs; raw log text, identifiers, query results, and credentials stay inside.
workflow aou_diagnostic {
  input {
    File task_stderr
    File identity_guard
    String runtime_image
  }
  call diagnose {
    input: task_stderr = task_stderr, identity_guard = identity_guard,
           runtime_image = runtime_image
  }
  output { Array[File] diagnostics = diagnose.diagnostics }
}

task diagnose {
  input {
    File task_stderr
    File identity_guard
    String runtime_image
  }
  command <<<
    set -euo pipefail
    cp "~{identity_guard}" aou_identity.py
    timeout --kill-after=5s 90s python - "~{task_stderr}" <<'PY'
    from pathlib import Path
    import sys
    from aou_identity import task_account

    task_account()
    with Path(sys.argv[1]).open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 262144))
        log = handle.read().decode("utf-8", errors="replace").lower()
    signatures = {
        "credential_override": "credential-file overrides are not allowed",
        "forbidden_identity": "refusing an execution account",
        "invalid_identity": "cannot establish the execution account",
        "metadata_account_type": "requires the workspace vm service account",
        "permission_denied": "permission denied",
        "access_denied": "access denied",
        "forbidden": "forbidden",
        "missing_libgomp": "libgomp.so",
        "missing_libopenblas": "libopenblas",
        "module_missing": "modulenotfounderror:",
        "import_failure": "importerror:",
        "value_error": "valueerror:",
        "key_error": "keyerror:",
        "type_error": "typeerror:",
        "runtime_error": "runtimeerror:",
        "attribute_error": "attributeerror:",
        "file_missing": "filenotfounderror:",
        "query_bad_request": "google.api_core.exceptions.badrequest:",
        "query_column_missing": "unrecognized name:",
        "query_table_missing": "not found: table",
        "query_budget": "exceeded limit for bytes billed",
        "timeout": "timeouterror",
        "pip_no_distribution": "no matching distribution found",
        "table_columns": "usecols do not match columns",
        "no_selected_disease": "no prespecified endpoint passed",
        "no_mapped_disease": "no eligible mapped disease",
        "score_archive": "expected exactly one scores.tar",
        "phenotype_archive": "expected one phenotypelibrary",
        "gamfit_version": "gamfit version does not match",
        "checkpoint_upload": "checkpoint upload failed",
        "no_space": "no space left on device",
    }
    matched = [label for label, phrase in signatures.items() if phrase in log]
    if not matched:
        matched = ["empty_log" if not log.strip() else "unclassified_failure"]
    for label in matched:
        Path(f"diagnostic__{label}.txt").write_text(label + "\n")
    PY
  >>>
  output { Array[File] diagnostics = glob("diagnostic__*.txt") }
  runtime {
    docker: runtime_image
    cpu: 1
    memory: "2 GiB"
    disks: "local-disk 10 SSD"
    preemptible: 0
    maxRetries: 0
  }
}
