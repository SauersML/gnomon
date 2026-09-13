version 1.0

# Inspect a failed task inside its workspace. Only fixed diagnostic labels are
# outputs; raw log text, identifiers, query results, and credentials stay inside.
workflow aou_diagnostic {
  input {
    File? task_stderr
    File identity_guard
    File status_module
    String status_uri
    String runtime_image
    File? relatedness_prune
    File? checkpoint
    Array[File] score_files = []
    String score_id = ""
  }
  call diagnose {
    input: task_stderr = task_stderr, identity_guard = identity_guard,
           status_module = status_module, status_uri = status_uri,
           runtime_image = runtime_image, relatedness_prune = relatedness_prune,
           checkpoint = checkpoint, score_files = score_files, score_id = score_id
  }
  output { Array[File] diagnostics = diagnose.diagnostics }
}

task diagnose {
  input {
    File? task_stderr
    File identity_guard
    File status_module
    String status_uri
    String runtime_image
    File? relatedness_prune
    File? checkpoint
    Array[File] score_files
    String score_id
  }
  command <<<
    set -euo pipefail
    cp "~{identity_guard}" aou_identity.py
    cp "~{status_module}" aou_status.py
    cp "~{write_json(score_files)}" score_files.json
    cp "~{write_json(score_id)}" score_id.json
    timeout --kill-after=5s 90s python - "~{default="" task_stderr}" "~{default="" relatedness_prune}" "~{default="" checkpoint}" "~{status_uri}" <<'PY'
    from pathlib import Path
    import re
    import sys
    import tarfile
    import json
    from aou_identity import task_account, require_spot_amd
    from aou_status import fit_cpu_label, publish_status, scoring_cpu_label

    task_account()
    require_spot_amd()
    score_files = json.loads(Path("score_files.json").read_text())
    score_id = json.loads(Path("score_id.json").read_text())
    if len(score_files) > 4 or (score_files and not re.fullmatch(r"PGS\d{6}", score_id)):
        raise ValueError("score schema inspection needs at most four files and one Catalog ID")
    for index, score_file in enumerate(score_files):
        with Path(score_file).open() as handle:
            header = []
            for _ in range(32):
                fields = handle.readline(8192).rstrip().split("\t")
                if fields[0] in ("#IID", "IID"):
                    header = fields
                    break
                if not fields[0].startswith("#"):
                    break
        for suffix, label in (("_AVG", "average"), ("_MISSING_PCT", "missingness"),
                              ("_SUM", "sum"), ("_MISSING_CT", "missing_count")):
            state = "present" if score_id + suffix in header else "absent"
            name = f"score_file_{index}_{label}_{state}"
            Path(f"diagnostic__{name}.txt").write_text(name + "\n")
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
    if not sys.argv[1] and not sys.argv[3]:
        raise ValueError("a private log or checkpoint is required")
    log = ""
    if sys.argv[1]:
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
            resources = [member for member in members
                         if member.isfile() and member.name == "fit.resources.json"]
            if len(resources) == 1:
                if resources[0].size > 4096:
                    raise ValueError("oversized resource diagnostics")
                metrics = json.load(archive.extractfile(resources[0]))
                label = scoring_cpu_label(metrics)
                publish_status(sys.argv[4], label)
                Path(f"diagnostic__{label}.txt").write_text(label + "\n")
            fit_resources = [member for member in members if member.isfile()
                             and Path(member.name).name == "fit.resources.json"
                             and member.name != "fit.resources.json"]
            if len(fit_resources) > 16:
                raise ValueError("checkpoint exceeds diagnostic fit budget")
            for member in fit_resources:
                if member.size > 4096:
                    raise ValueError("oversized fit resource diagnostics")
                metrics = json.load(archive.extractfile(member))
                if "allotted_threads" in metrics:
                    label = fit_cpu_label(metrics)
                    Path(f"diagnostic__{label}.txt").write_text(label + "\n")
                code = metrics.get("exit_code")
                if isinstance(code, int) and code != 0:
                    # Fixed categories: which signal killed a worker, or that it
                    # errored; never the log text.
                    label = (f"fit_exit_signal_{-code}" if -31 <= code < 0 else
                             "fit_exit_error" if 0 < code < 128 else "fit_exit_other")
                    Path(f"diagnostic__{label}.txt").write_text(label + "\n")
                if metrics.get("restarted_after_signal"):
                    Path("diagnostic__fit_worker_restarted.txt").write_text("fit_worker_restarted\n")
            unfinished = [member for member in members if member.isfile()
                          and Path(member.name).name == "fit.log"
                          and str(Path(member.name).parent / "completed.json") not in names]
            if len(unfinished) > 16:
                raise ValueError("checkpoint exceeds diagnostic worker budget")
            for member in unfinished:
                with archive.extractfile(member) as handle:
                    handle.seek(max(0, member.size - 262144))
                    log += "\n" + handle.read().decode("utf-8", errors="replace").lower()
    signatures = {
        "score_input_format": "could not determine input format",
        "score_indexing": "stage 1: indexing subject data",
        "score_columns": "stage 2: discovering all score columns",
        "score_preparing": "stage 3: streaming and collecting data",
        "score_matrices": "stage 4: verifying data",
        "score_computing": "resource allocation complete",
        "score_computed": "computation finished",
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
        "score_missingness_absent": "lacks per-participant missingness",
        "score_missingness_invalid": "cached score has invalid missingness percentages",
        "score_completely_missing": "completely missing scores cannot enter ctn",
        "score_source_ambiguous": "ambiguous cached score source",
        "censoring_training_support": "insufficient training support for ancestry-specific censoring",
        "censoring_horizon_support": "evaluation horizon lacks censoring support",
        "censoring_weights_unstable": "event-time censoring weights are unstable",
        "disease_events_insufficient": "insufficient training events for cause 1",
        "death_events_insufficient": "insufficient training events for cause 2",
        "heldout_size_insufficient": "too few held-out participants",
        "fit_warm_start_restored": "[warm-start-cache] restored",
        "worker_restarted_after_signal": "worker_restarted_after_signal",
        "worker_fit_started": "worker_fit_started",
        "worker_fit_saved": "worker_fit_saved",
        "worker_grid_started": "worker_grid_started",
        "worker_grid_complete": "worker_grid_complete",
        "worker_validation_complete": "worker_validation_complete",
        "runtime_error": "runtimeerror:",
        "gam_error": "gamerror:",
        "gam_basis_error": "basiserror:",
        "gam_linear_solve_error": "linearsystemsolveerror:",
        "gam_constraint_error": "parameterconstrainterror:",
        "gam_inner_convergence_error": "pirlsconvergenceerror:",
        "gam_outer_convergence_error": "remlconvergenceerror:",
        "gam_hessian_error": "hessiannotpositivedefiniteerror:",
        "gam_geometry_error": "geometryerror:",
        "gam_input_error": "invalidinputerror:",
        "gam_monotone_root_error": "monotonerooterror:",
        "gam_cache_error": "cachestoreerror:",
        "gam_integration_error": "integrationerror:",
        "gam_ctn_error": "transformationnormalerror:",
        "gam_custom_family_error": "customfamilyerror:",
        "gam_penalty_error": "jointpenaltyerror:",
        "gam_identifiability_error": "identifiabilitycompilererror:",
        "gam_matrix_error": "matrixerror:",
        "gam_linalg_error": "linearalgebraerror:",
        "gam_smooth_error": "smootherror:",
        "gam_term_error": "termbuildererror:",
        "gam_data_error": "dataerror:",
        "gam_config_error": "invalidconfigurationerror:",
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
    # Each certified inner solve is one outer evaluation the solver completed;
    # fixed buckets only, never the count or any log text.
    certified = log.count("mode certificate]")
    if certified:
        matched.append("fit_inner_solves_" + ("1_9" if certified < 10 else "10_49" if certified < 50
                                              else "50_199" if certified < 200 else "200_plus"))
    progress = re.findall(r"> progress: \d+/\d+ variants \((\d+)%\)", log)
    if progress:
        percent = int(progress[-1])
        if not 0 <= percent <= 100:
            raise ValueError("native progress percentage is outside its contract")
        # Fixed performance categories only; never emit participant/variant
        # counts, identifiers, arbitrary log text, or exception messages.
        bucket = ("0" if percent == 0 else "1_24" if percent < 25 else
                  "25_49" if percent < 50 else "50_74" if percent < 75 else
                  "75_99" if percent < 100 else "100")
        matched.append("score_progress_" + bucket)
    if not matched:
        matched = ["empty_log" if not log.strip() else "unclassified_failure"]
    for label in matched:
        Path(f"diagnostic__{label}.txt").write_text(label + "\n")
    # Only known public function names, never raw traceback paths or messages.
    functions = ("run", "read_ancestry", "unpack_phenotypes", "unpack_score_cache",
                 "cached_score_ids", "load_cached_score", "endpoint_scores",
                 "person_times", "case_dates", "build_cohort",
                 "task_account", "publish_status", "validate_config", "load_score_panel",
                 "select_runtime_diseases", "query", "publish", "prepare_inputs",
                 "fit_worker", "fit_transform", "transformed_score",
                 "analyze_partition", "checkpointed_fit", "bounded_fit")
    for function in functions:
        if re.search(r", in " + re.escape(function) + r"\s*\n", log):
            Path(f"diagnostic__function_{function}.txt").write_text(function + "\n")
    PY
  >>>
  output { Array[File] diagnostics = glob("diagnostic__*.txt") }
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
