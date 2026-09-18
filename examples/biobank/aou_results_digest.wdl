version 1.0

# Reduce a completed pilot's aggregate metrics file to fixed-form result
# tokens inside the workspace. Every emitted token is an aggregate over at
# least the configured minimum number of participants; small counts,
# identifiers and free text never leave, and no withheld count can be
# recovered by subtracting the shown ones.
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
               "brier_difference", "brier_difference_standard_error", "auc_difference",
               "brier_censoring_standard_error", "brier_95_lower", "brier_95_upper",
               "ipcw_observed_risk_standard_error", "ipcw_observed_risk_censoring_standard_error",
               "ipcw_observed_risk_95_lower", "ipcw_observed_risk_95_upper", "ipcw_weight_n_eff",
               "brier_difference_censoring_standard_error", "brier_difference_95_lower",
               "brier_difference_95_upper")

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

    # Counts that add over disjoint groups, or nearly so (Kish's effective size of a
    # group's weights). Withheld from one member of a partition of the overall cell,
    # they would follow from the overall cell's minus the other members'.
    COUNTS = ("n", "observed_disease_events", "ipcw_weight_n_eff")

    def family(group):
        group = str(group)
        return "pc_neighborhood" if group == "pc_outside_training_support" else group.split(":", 1)[0]

    def shown(row):
        return row.get("status") == "ok" and int(row.get("n", 0)) >= MINIMUM_COUNT

    def partitions_withheld(rows):
        """The (horizon, family) partitions whose members keep their counts inside: a
        member is withheld, or the shown counts do not add up to the overall cell's,
        which leaves a remainder cell that subtraction would disclose."""
        overall, members = {}, {}
        for row in rows:
            if row.get("status") == "pooled_censoring":
                continue
            horizon = float(row["horizon"])
            if row["group"] == "overall":
                overall[horizon] = row
            else:
                members.setdefault((horizon, family(row["group"])), []).append(row)
        withheld = set()
        for (horizon, name), group_rows in members.items():
            whole = overall.get(horizon)
            if whole is None or not shown(whole) or not all(shown(row) for row in group_rows):
                withheld.add((horizon, name))
                continue
            for key in ("n", "observed_disease_events"):
                values = [row.get(key) for row in group_rows]
                if key not in whole and all(value is None for value in values):
                    continue
                if (key not in whole or any(value is None or int(value) < MINIMUM_COUNT for value in values)
                        or sum(int(value) for value in values) != int(whole[key])):
                    withheld.add((horizon, name))
        return withheld

    def metric_rows(rows, stage):
        withheld = partitions_withheld(rows)
        for row in rows:
            base = [stage, slug(row["group"]), "h" + token(float(row["horizon"]))]
            if row.get("status") == "pooled_censoring":
                # A caveat beside the horizon's cells, not a cell: stratum labels,
                # reasons, and a mean weight and censoring hazard ratio the runner
                # withholds under the minimum.
                for stratum in row.get("strata") or []:
                    pooled = [stage, "censoring_pooled", base[2], slug(stratum["ancestry"]), slug(stratum["reason"])]
                    emit(pooled)
                    for key in ("ipcw_weight_mass", "censoring_hazard_ratio"):
                        if stratum.get(key) is not None:
                            emit(pooled + [key, token(float(stratum[key]))])
                continue
            if row.get("status") == "insufficient_support":
                # A horizon refused on positivity names each refused stratum, its
                # reason, and a modelled censoring survival and upper bound withheld
                # under the minimum.
                for stratum in row.get("strata") or []:
                    refused = [stage, "censoring_refused", base[2], slug(stratum["ancestry"]), slug(stratum["reason"])]
                    emit(refused)
                    for key in ("censoring_survival", "censoring_survival_upper"):
                        if stratum.get(key) is not None:
                            emit(refused + [key, token(float(stratum[key]))])
            if row.get("status") != "ok" or int(row.get("n", 0)) < MINIMUM_COUNT:
                emit(base + ["insufficient_support"])
                continue
            counts_withheld = (float(row["horizon"]), family(row["group"])) in withheld
            if counts_withheld:
                emit(base + ["counts_withheld"])
            for key in ALLOWED:
                if key not in row or (counts_withheld and key in COUNTS):
                    continue
                if key == "observed_disease_events" and int(row[key]) < MINIMUM_COUNT:
                    continue
                emit(base + [key, token(row[key])])

    def report_rows(report, stage):
        for variant, model in sorted((report.get("models") or {}).items()):
            if "cif_grid_error" in model:
                emit([stage, slug(variant), "cif_grid_error", token(float(model["cif_grid_error"]))])
            knots = model.get("time_num_internal_knots") or []
            if len(knots) == 2 and all(isinstance(k, int) for k in knots):
                emit([stage, slug(variant), "time_knots", token(knots[0]), token(knots[1])])
            metric_rows(model.get("metrics", []), f"{stage}__{slug(variant)}")
        metric_rows(report.get("incremental") or [], f"{stage}__incremental")

    SUPPORT = (("insufficient training events for cause 1", "disease_events_insufficient"),
               ("insufficient training events for cause 2", "death_events_insufficient"),
               ("too few held-out participants", "heldout_size_insufficient"),
               ("lacks censoring support", "censoring_horizon_support"),
               ("censoring weights are unstable", "censoring_weights_unstable"),
               ("insufficient training support for ancestry-specific censoring", "censoring_training_support"))

    def support_rows(messages, name):
        if len(messages) > 32:
            raise ValueError("too many support messages to digest")
        for message in messages:
            partition = "development" if str(message).startswith("development:") else "outer"
            horizon = re.search(r"horizon ([0-9.]+)", str(message))
            when = "h" + token(float(horizon.group(1))) if horizon else "all"
            for phrase, label in SUPPORT:
                if phrase in str(message):
                    emit([name, "support", partition, when, label])

    results = json.loads(Path(sys.argv[1]).read_text())
    if len(results) > 8:
        raise ValueError("digest covers at most eight endpoints")
    for disease, result in results.items():
        name = slug(disease)
        for pgs, report in (result.get("development") or {}).items():
            report_rows(report, f"{name}__development__{slug(pgs)}")
        report_rows(result, f"{name}__final")
        support_rows(result.get("evaluation_support_errors") or [], name)
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
