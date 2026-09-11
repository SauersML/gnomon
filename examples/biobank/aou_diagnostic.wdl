version 1.0

# Inspect a failed task inside its workspace. Only fixed diagnostic labels are
# outputs; raw log text, identifiers, query results, and credentials stay inside.
workflow aou_diagnostic {
  input {
    File task_stderr
    File identity_guard
    String runtime_image
    File? relatedness_prune
    File? checkpoint
  }
  call diagnose {
    input: task_stderr = task_stderr, identity_guard = identity_guard,
           runtime_image = runtime_image, relatedness_prune = relatedness_prune,
           checkpoint = checkpoint
  }
  output { Array[File] diagnostics = diagnose.diagnostics }
}

task diagnose {
  input {
    File task_stderr
    File identity_guard
    String runtime_image
    File? relatedness_prune
    File? checkpoint
  }
  command <<<
    set -euo pipefail
    cp "~{identity_guard}" aou_identity.py
    timeout --kill-after=5s 90s python - "~{task_stderr}" "~{default="" relatedness_prune}" "~{default="" checkpoint}" <<'PY'
    from pathlib import Path
    import re
    import sys
    import tarfile
    from aou_identity import task_account

    task_account()
    if sys.argv[2]:
        with Path(sys.argv[2]).open() as handle:
            fields = handle.readline(8192).strip().split("\t")
        known_headers = {"research_id", "s", "sample_id", "IID", "#IID"}
        found = [name for name in fields if name in known_headers]
        if found:
            for name in found:
                label = "prune_header_" + name.lstrip("#")
                Path(f"diagnostic__{label}.txt").write_text(label + "\n")
        elif len(fields) == 1 and fields[0].isdigit():
            Path("diagnostic__prune_headerless_numeric.txt").write_text("headerless numeric\n")
        else:
            Path("diagnostic__prune_header_unrecognized.txt").write_text("unrecognized header\n")
        if len(fields) > 1:
            Path("diagnostic__prune_multiple_columns.txt").write_text("multiple columns\n")
    with Path(sys.argv[1]).open("rb") as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 262144))
        log = handle.read().decode("utf-8", errors="replace").lower()
    if sys.argv[3]:
        with tarfile.open(sys.argv[3], "r:gz") as archive:
            members = archive.getmembers()
            if sum(member.size for member in members) > 2 * 1024**3:
                raise ValueError("checkpoint exceeds diagnostic size budget")
            names = {member.name for member in members}
            unfinished = [member for member in members if member.isfile()
                          and member.name.endswith("/fit.log")
                          and str(Path(member.name).parent / "completed.json") not in names]
            if len(unfinished) > 16:
                raise ValueError("checkpoint exceeds diagnostic worker budget")
            for member in unfinished:
                with archive.extractfile(member) as handle:
                    handle.seek(max(0, member.size - 262144))
                    log += "\n" + handle.read().decode("utf-8", errors="replace").lower()
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
        "merge_error": "pandas.errors.mergeerror:",
        "merge_right_duplicates": "merge keys are not unique in right dataset",
        "merge_left_duplicates": "merge keys are not unique in left dataset",
        "merge_both_duplicates": "merge keys are not unique in either left or right dataset",
        "arrow_invalid": "pyarrow.lib.arrowinvalid:",
        "memory_error": "memoryerror:",
        "zero_division": "zerodivisionerror:",
        "assertion_error": "assertionerror:",
        "fit_nonconvergence": "did not converge",
        "nonfinite_value": "non-finite",
        "singular_system": "singular",
        "date_out_of_bounds": "outofboundsdatetime",
        "date_parse_error": "dateparseerror",
        "datetime_resolution": "cannot subtract",
        "score_ids_invalid": "score ids must be present and unique",
        "score_values_invalid": "cached score contains non-finite values",
        "censoring_training_support": "insufficient training support for ancestry-specific censoring",
        "censoring_horizon_support": "evaluation horizon lacks censoring support",
        "censoring_weights_unstable": "event-time censoring weights are unstable",
        "disease_events_insufficient": "insufficient training events for cause 1",
        "death_events_insufficient": "insufficient training events for cause 2",
        "heldout_size_insufficient": "too few held-out participants",
        "worker_fit_started": "worker_fit_started",
        "worker_fit_saved": "worker_fit_saved",
        "worker_grid_started": "worker_grid_started",
        "worker_grid_complete": "worker_grid_complete",
        "worker_validation_complete": "worker_validation_complete",
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
        "http_error": "urllib.error.httperror:",
        "http_404": "http error 404",
        "http_403": "http error 403",
        "syntax_error": "syntaxerror:",
    }
    matched = [label for label, phrase in signatures.items() if phrase in log]
    if not matched:
        matched = ["empty_log" if not log.strip() else "unclassified_failure"]
    for label in matched:
        Path(f"diagnostic__{label}.txt").write_text(label + "\n")
    # Only known public function names, never raw traceback paths or messages.
    functions = ("run", "read_ancestry", "unpack_phenotypes", "unpack_score_cache",
                 "cached_score_ids", "person_times", "case_dates", "build_cohort",
                 "task_account", "publish_status", "validate_config", "load_score_panel",
                 "select_runtime_diseases", "query", "publish", "prepare_inputs",
                 "fit_worker", "transform_worker", "fit_ctn", "transformed_score",
                 "analyze_partition", "checkpointed_fit", "bounded_fit")
    for function in functions:
        if re.search(r", in " + re.escape(function) + r"\s*\n", log):
            Path(f"diagnostic__function_{function}.txt").write_text(function + "\n")
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
