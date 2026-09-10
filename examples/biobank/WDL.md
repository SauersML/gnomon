# AoU survival WDL

`aou_survival.wdl` runs a bounded real-data pilot inside an authorized AoU
Workbench workspace. Its launcher follows the `wb workflow create` / `wb
workflow job run` pattern in `pgsEngine/pipeline/aou/workbench`.

## Data and disease selection

The workflow imports the **same** `disease_selection.py` as the existing
`marginal_slope_diseases.py` example. It resolves the curated SNOMED/PGS map,
extracts single-root reference disease concepts from a staged OHDSI Phenotype
Library snapshot, ranks all canonical disease roots in the active CDR, then
keeps PGS-mapped diseases within the top 20. `disease_limit=1` initially runs the
first eligible disease in that ordering. No disease name is supplied manually.
The reference-root rule is a first-recorded-condition phenotype, not execution
of the complete OHDSI cohort definition with all its eligibility criteria.

Cached scores come from pgsEngine's existing **`shared_features.tar.gz`**.
The task extracts only its inner `scores.tar` containing Gnomon `.sscore`
files. Each selected PGS must have exactly one `PGSnnnnnn_AVG` column source and
an `IID`/`#IID` column. The runner refuses missing or ambiguous scores. It does
not localize the genotype callset or repeat scoring. The existing disease map
and pgsEngine's disease registry differ: check that the selected score is in
the cache rather than silently substituting another PGS.

PCs and ancestry labels come from the CDR's `ancestry_preds.tsv` columns
`research_id`, `pca_features`, and `ancestry_pred`. The published
`samples_relatedness_flagged_samples.tsv` (`research_id` header) is required.
Participants on that prune list are removed before the outcome-blind sample
and train/test split. This uses AoU's published relatedness threshold; it is
not a claim that all remaining distant relatives are independent.

Follow-up starts at the **first EHR observation period**, matching the existing
example's available time source, not at recruitment. Only that single interval
is used; observation gaps are not bridged. Adults with disease/death on or
before entry are excluded. Follow-up ends at disease, death, or the interval's
end, whichever comes first. Same-day disease/death ties are excluded because
their order is unresolved. This estimates first recorded disease under EHR
ascertainment; a recruitment-based prospective study needs a recruitment-date
cohort definition instead. Sex uses the explicit AoU sex-at-birth concept map.

## Model and outputs

Both candidates use `Surv(entry, followup, event)` with time since entry,
baseline age, sex, and a joint six-PC Duchon baseline surface (32 centers).
The constant-slope candidate has `slope_formula="1"`; the PC-varying candidate
has an intercept and a smaller joint Duchon PC surface (16 centers). Each is
fit separately for disease and competing death. The score is supplied only as
`z_column="PGS"`, never in the baseline/slope formulas. There is no frailty or
follow-up-time slope margin in this specification.

This uses gamfit's fitted/replayed latent-score gate. It **does not claim** to
repair or invoke the integrated cross-fitted CTN entry point. The conditional
normal score assumption still needs held-out distribution diagnostics before
claiming a marginal interpretation. These fits also use the formula API,
not the current Gnomon calibration adapter's different term construction.

Each fit must reproduce cumulative hazards after save/load. The runner checks
finite, monotone hazards, combines disease/death hazard increments into CIFs
conditional on event-free entry, and requires the fine/coarse grid difference
to be at most 0.001. It reports horizon-specific disease Brier scores and
mean-predicted versus IPCW-observed risks overall, by ancestry, and within
fixed predicted-risk intervals. Training-only ancestry-stratified reverse KM
estimates censoring. This assumes sufficient censoring independence within
those strata; it does not adjust censoring for every clinical predictor.

Cells with fewer than 20 observed disease events or known noncases are marked
`insufficient_support`. That is not evidence of calibration. The same holdout
compares both fixed candidates; it is not a locked evaluation of a selected or
recalibrated winner. No post-hoc calibration layer is fitted in this pilot.

Workflow outputs are `metrics.json` and `provenance.json`. Models, participant
frames, predictions and raw fit logs remain internal workspace task artifacts.
Aggregate output declarations do not constitute approval to export artifacts
from AoU; the existing workspace's data-use controls still apply.

## Environment and submission

`submit_aou.py` reads these required environment variables. They must describe
real resources accessible to the selected workspace; no placeholder JSON is
used:

| Variable | Value supplied by the workspace/operator |
| --- | --- |
| `GOOGLE_PROJECT` | Workspace billing project |
| `WORKSPACE_CDR` | Resolved BigQuery `project.dataset` |
| `AOU_WORKSPACE_ID` | Workbench workspace ID |
| `WORKSPACE_BUCKET` | Existing `gs://` workspace bucket |
| `AOU_BUCKET_ID` | Workbench resource ID of that same bucket |
| `AOU_CDR_RESOURCE_ID` | Workbench resource ID of that CDR |
| `AOU_EXPECTED_ACCOUNT` | Authorized human email, configured locally |
| `AOU_GCLOUD_CONFIGURATION` | Local gcloud configuration name |
| `WORKBENCH_CONTEXT_PARENT_DIR` | Local isolated Workbench context directory |
| `AOU_RUNTIME_IMAGE` | Accessible Linux Python 3.12 image pinned with `@sha256:` |
| `AOU_WHEELHOUSE_URI` | Staged tar of Linux CPython 3.12 dependency wheels |
| `AOU_PHENOTYPE_LIBRARY_URI` | Staged OHDSI PhenotypeLibrary ZIP snapshot |
| `AOU_SHARED_FEATURES_URI` | Existing pgsEngine `shared_features.tar.gz` |
| `AOU_ANCESTRY_URI` | Published ancestry-predictions TSV for this CDR |
| `AOU_RELATEDNESS_PRUNE_URI` | Published relatedness-prune TSV for this CDR |

The launcher requires an explicitly configured isolated context and the exact
locally configured human account in both CLIs. It refuses **any email containing
`user`**, case-insensitively, including in `AOU_EXPECTED_ACCOUNT`.
It repeats the checks before every upload or workflow mutation. The runtime
also checks its VM service-account email before querying the CDR.

Refresh the dedicated Workbench login when necessary:

```bash
wb auth login --mode=BROWSER
```

Once the environment is populated, run from the authenticated Workbench
environment (the launcher only performs CLI/file operations):

```bash
python examples/biobank/submit_aou.py --check
python examples/biobank/submit_aou.py
```

For the configured local workspace, the resolved values are saved in the
git-ignored `examples/biobank/.aou-workflow/environment.sh`; source that file
before invoking the launcher. It contains resource identifiers, not login
credentials. Refresh resource values there when changing workspaces or CDRs.

`--check` verifies both identities, workspace/CDR/bucket resolution, and the
existence of the staged objects without uploading or submitting. Submission
uses unique source/config-hashed paths and writes a submission receipt under
`examples/biobank/.aou-workflow/`. It submits exactly one task and does not
start a local polling process. Use Workbench's job UI to inspect/cancel it.

## Runtime and iteration budget

Stage wheels satisfying `aou_requirements.txt` once, including the pinned
gamfit wheel and all its transitive dependencies. The task installs only
binary wheels with `--no-index`; it never compiles or downloads dependencies.
Use a Linux Python 3.12 runtime image that supplies the system libraries those
wheels require. The runtime image digest, installed versions, gamfit build
information, source/input hashes, and query IDs are recorded in provenance.

The initial budget is one disease, at most 5,000 outcome-blind sampled rows,
four sequential fits, four CPUs, 16 GiB RAM, and 50 GiB disk. Each fit has a
180-second wall cap, each query a 120-second cap and a billed-byte ceiling;
the command has a 30-minute cap and zero automatic retries. Temporary storage
and solver caches use the attached task disk. A child timeout terminates its
process group. Insufficient events/support fail before fitting; increase the
sample budget only after inspecting that signal. Population event frequencies
are never replaced with a balanced case/control sample.

Run static and synthetic contract validation on MSI using the existing warm
Python dependencies; no participant data needs to leave AoU:

```bash
miniwdl check examples/biobank/aou_survival.wdl
python -m unittest discover -s examples/biobank -p test_aou_survival.py
```

The native acceptance check also exercises a real PC-varying fit, held-out
cumulative hazards, and save/load prediction equivalence on synthetic data:

```bash
python examples/biobank/test_aou_runtime.py --output .validation-logs/aou/native
```

Keep that output directory between development attempts so the native fit can
reuse its persistent warm cache. A timeout is a failed acceptance check even
if partial artifacts exist. The deterministic contract tests check termination
of a fit when its controller receives SIGTERM.
