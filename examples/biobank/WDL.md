# AoU prospective survival workflow

This workflow runs in the real AoU workspace and reuses pgsEngine's existing
`shared_features.tar.gz`. Workspace identifiers and the authorized account
come from local environment variables; neither belongs in the repository.

## Cohort and endpoint selection

The shared selector ranks all canonical OHDSI disease roots by recorded case
count, then intersects the top 20 with its disease map. The survival experiment
further restricts that set to the COPD, hypertension and obesity endpoints in
[aou_pgs_panel.json](aou_pgs_panel.json). This is a first-recorded-condition
phenotype using those roots and descendants, not the complete OHDSI algorithm.

Baseline is the first primary-consent date from observations descended from
the `Consent PII` Module concept, following the
[AoU enrollment documentation](https://support.researchallofus.org/hc/en-us/articles/13176125767188-How-to-find-participant-enrollment-data).
A single EHR observation interval must cover baseline and the prespecified
365-day lookback. Adults with recorded disease or death at/before baseline
are excluded. We do not require future disease-free observation to enter.

Follow-up ends at the earliest qualifying diagnosis, primary death date
from `aou_death`, or the covering observation interval's end. Multiple primary
death reports are reduced to the earliest date per person before joining. Observation
gaps are not bridged. Same-day disease/death ties retain the participant and
give precedence to the recorded diagnosis; this is a day-level endpoint convention,
not an inference of within-day ordering. Their count is reported subject to
the same small-cell suppression as other counts. This predicts recorded
diagnosis, not biological onset.
The resolved CDR release is recorded; horizons must be supported by that
release's actual follow-up.

PCs come from pgsEngine's `projection_pcs.parquet`, using the pinned external
`hwe_1kg_hgdp_gsa_v3` projection. The CTN reference must use that same model
(uncompressed model SHA-256 is recorded in `aou_analysis.json`). AoU's published
PC coordinates are a separate coordinate system and are not substituted.
Ancestry labels still use `research_id` and `ancestry_pred` from the release's
ancestry file. The published relatedness-prune file's `sample_id` column is
matched to ancestry `research_id` and applied before sampling
or splitting. The `sample_id` header and numeric IDs are required; unknown
headers, extra columns, and malformed rows are rejected.
Remaining person IDs are the pilot's split groups. This is not
full pedigree reconstruction or a claim that distant relatives are independent.

## Prespecified model

The prespecified scores are COPD PGS004536, hypertension PGS004525,
and obesity PGS005199. The panel records exact
Catalog sources and pending component-provenance audits. PGS004787 is excluded
because its documented score development includes AoU. Public cohort metadata
does not establish participant-level non-overlap.

Every required score must exist in the real cache as a unique
`PGSnnnnnn_AVG` column with participant IDs and `PGSnnnnnn_MISSING_PCT`.
Completely missing scores are rejected, including those represented as zero.
Missing scores are reported by preflight and stop fitting; no score is substituted.
Final analysis requires completed discovery, component and tuning provenance.

A seeded group split reserves 20% as outer test. The remaining development
sample is split 75/25 by group for development checks of the single
PC-varying marginal-slope predictor. The score and model are prespecified;
there is no challenger search. After development acceptance, the outcome model
is fitted on outer training and evaluated on the locked test set.

The baseline has age, sex and a joint six-PC Duchon surface (32 centers).
The score surface has 16 centers and a time-constant signed slope. No frailty,
ensemble, manifold or post-hoc calibration stack is enabled. The baseline and
score-effect surfaces are separately penalized and jointly fitted.

CTN is fitted once per PGS on an external genetic reference panel, conditional
on PCs only. It is never fitted or updated on AoU rows. There are no internal
AoU CTN folds. CTN uses `transformation_score`, never its conditional-mean
`predict` operation. The model and manifest are required staged WDL inputs;
missing, duplicate, mismatched or corrupt reference transforms stop the run.
GAM's native model embeds the saved CTN and replays it at prediction, with
save/load and batch-invariance checks. No second
normalization or influence absorber is fitted.

`reference_ctn.py` trains and packages the external model from a real reference
table containing `sample_id`, the matching `PGSnnnnnn_AVG`, and projected PCs.
Its metadata names the actual panel, score-file hash and projection-model hash.
Training and preprocessing run on MSI with external data; the resulting model
is staged into the workspace. The reference panel's PGS calculation must use
the same allele, weight and score-scaling conventions as the target cache.
Variant coverage and projection-marker overlap require a transport audit.

This estimates the reference-panel score distribution. It does not establish
normality in AoU conditional on age, sex or baseline eligibility. Consequently,
the exact conditional-normal marginal identity is an assumption to assess, not
a target-population calibration guarantee. Keep age, sex and PCs in the outcome
model and evaluate held-out score-distribution and risk diagnostics.

## Evaluation, persistence and limits

Separate disease and death components produce disease cumulative incidence.
This is not disease-only `1-S`. Saved predictions must be finite, monotone,
stable under save/load, row ordering and batch membership. The CIF grid
refinement error must be at most 0.001. Prediction horizons are explicit;
observed diagnosis/censoring times are excluded from prediction inputs.

Metrics include horizon-specific IPCW Brier score with group-robust uncertainty,
mean-risk discrepancy, and fixed risk-bin calibration. Audits cover ancestry,
sex, age bands and training-defined PC neighborhoods, including points outside
their support. Sparse cells are suppressed, not certified as calibrated.
Loss uncertainty is conditional on fitted models and censoring estimates.

Censoring uses training-only ancestry-stratified reverse Kaplan–Meier. A
stratum with fewer than 20 training rows, or whose own censoring survival at a
horizon is below 0.05 or has nobody followed past it, takes a proportional-hazards
censoring model over the pooled training set at that horizon: the pooled reverse
Kaplan–Meier raised to the stratum's censoring hazard ratio. The ratio is
(observed + 20) / (expected + 20), where expected counts the censorings the
stratum would show over its own follow-up at the pooled Breslow rate: the
posterior mean under a gamma prior worth 20 censorings at the pooled rate. The 20
is the support minimum, so a thin stratum stays near the pooled curve and one with
many censorings takes its own level; a weaker prior let a stratum of about 16 rows
under shared censoring cross the floor below by noise alone. The metrics carry a
`pooled_censoring` row per horizon naming each such stratum, its reason
(`training_rows`, `horizon_support`), its mean IPCW weight and its censoring hazard
ratio (both withheld under the reporting minimum); the digest emits them as
`…__censoring_pooled__h<h>__<stratum>__<reason>`.

Positivity: a horizon is refused where a modelled censoring survival is below
0.05, as where an own one is, and where the pooled set cannot support it. Below
that floor IPCW weights exceed 20 and the horizon has no stable estimator, so a
refusal is the correct output and a biased estimate reported as usable would be
worse. The refusal row names each refused stratum with its reason and modelled
censoring survival (withheld under the minimum); the digest emits
`…__censoring_refused__h<h>__<stratum>__<reason>`. The model takes the pooled
curve's shape: where a stratum's censoring is not proportional to the pooled
set's its cells keep a residual bias, and where nobody in it is followed to the
horizon its curve there is an extrapolation no censoring model can check. A
stratum whose follow-up truly ends before the horizon is censored at a higher
rate over the follow-up it has, so its modelled survival falls below the floor
and the horizon is refused; nobody in a small stratum reaching the horizon by
chance is not refused on that ground alone. The mean IPCW weight is one in
expectation under the right censoring model; well below one it shows the
stratum's censoring curve is too high. The stratified model
assumes sufficient independence within those strata and does not account for
all site, calendar-period or clinical dependence. No result from this pilot
establishes optimal deployment accuracy. The marginal identity is a model
property, not observed-outcome calibration.

The first run should use `--prepare-only`: it checks real score availability,
cohort fields, event counts and horizon support without fitting models.
After that and native acceptance pass, `--smoke-only` fits the first
prespecified score using only the development split: frozen external CTN and two
cause-specific outcome models. Only that primary score must be cached for the
smoke run and final analysis.
It performs the same persistence, batching,
monotonicity and CIF checks as the full analysis. It neither selects a score
nor evaluates the outer test set. Checkpoint signatures bind the model and input definitions.
The pilot caps rows, CPUs, query bytes/time and each fit's wall time.
A failed step raises an error; its process group is stopped.

Completed steps publish a workspace checkpoint so a later bounded job can
resume with `--resume gs://...`. Source, input and configuration hashes must
match; fitted-step receipts also require the same native engine. Checkpoints
contain participant data, scores, models and logs and must remain inside the
authorized workspace. WDL outputs are aggregate metrics, provenance **and
the sensitive checkpoint archive**; none is automatically approved for export.

The external-CTN revision passes 27 deterministic workflow contract tests and
WDL validation on MSI. Native survival acceptance and the real cohort run
remain separate checks; see [VALIDATION.md](VALIDATION.md) for observed results.

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
| `AOU_REFERENCE_CTN_URIS` | Space-separated staged external CTN archives, one per requested PGS |
| `AOU_PHENOTYPE_LIBRARY_URI` | Staged OHDSI PhenotypeLibrary ZIP snapshot |
| `AOU_SHARED_FEATURES_URI` | Existing pgsEngine `shared_features.tar.gz` |
| `AOU_ANCESTRY_URI` | Published ancestry-predictions TSV for this CDR |
| `AOU_RELATEDNESS_PRUNE_URI` | Published relatedness-prune TSV for this CDR |

The launcher requires an explicitly configured isolated context and the exact
locally configured human account in both CLIs. It refuses **any email containing
`user`**, case-insensitively, including in `AOU_EXPECTED_ACCOUNT`.
It repeats the checks before every staging batch or workflow mutation. The runtime
also checks its VM service-account email before querying the CDR.

Refresh the dedicated Workbench login when necessary:

```bash
wb auth login --mode=BROWSER
```

Once the environment is populated, run from the authenticated Workbench
environment (the launcher only performs CLI/file operations):

```bash
python examples/biobank/submit_aou.py --check
python examples/biobank/submit_aou.py --endpoint hypertension --prepare-only
# After native acceptance and cohort preflight succeed:
python examples/biobank/submit_aou.py --endpoint hypertension --smoke-only
# After the development smoke passes:
python examples/biobank/submit_aou.py
```

For the configured local workspace, the resolved values are saved in the
git-ignored `examples/biobank/.aou-workflow/environment.sh`; source that file
before invoking the launcher. It contains resource identifiers, not login
credentials. Refresh resource values there when changing workspaces or CDRs.

`--check` verifies both identities, workspace/CDR/bucket resolution, and the
existence of the staged objects without uploading or submitting. Submission
uses unique source/config-hashed paths and writes its inputs, workflow record and
receipt to one folder under `examples/biobank/.aou-workflow/`. It submits exactly
one task and does not start a local polling process. Use Workbench's job UI to
inspect/cancel it.

The pilot's other runs are submitted with `submit_runs.py`: `score-training`,
`benchmark`, `scoring-diagnostic`, `fit-diagnostic`, `stderr-diagnostic URI` and
`results-digest URI`. Deployment state for them (the portable scorer, analysis
configurations, the current checkpoint URI) stays in `.aou-workflow/`.

The benchmark's panel is tracked in `aou_benchmark.json`. Each disease leads with
multi-ancestry scores (discovery GWAS or tuning drawn from several ancestries,
none with All of Us participants in development) and keeps a European-trained
score as the comparator; the outcome-blind sample is capped per ancestry so
African and admixed American ancestry keep the held-out support their gains
need. Tabulate a finished run from a listing of its token objects, headline
ancestries first:

```bash
python examples/biobank/aou_benchmark_table.py tokens.txt
```

Every launcher stages files with one Cloud Storage media upload per file and
compares the stored MD5 and size with the local bytes; `wb gsutil cp` has stalled
indefinitely on a 12 MB archive that a single upload finished in 18 seconds.
Workbench has also hung after it had already created a workflow or a run, so
neither step is judged by its exit status. A failed create is accepted once the
workflow can be described, a run is started only while the workflow lists no
run, and a failed run step is confirmed by listing. If no run appears, rerun
the submission from its folder, which starts the run once and never starts a
second:

```bash
python examples/biobank/submit_runs.py finish examples/biobank/.aou-workflow/<submission>
```

Do not wrap a launcher in a short outer `timeout`, and never resubmit under a
new name after a reported failure without listing the workspace's runs first:
two runs of one analysis race on the same checkpoint.
`--endpoint` narrows execution after the existing selector; it never overrides
eligibility. Run endpoints separately when a combined run would exceed its
wall budget. Resume a checkpoint only for the same endpoint, score scope,
inputs and configuration. Comparison preflight checkpoints can feed the full
comparison; primary-score smoke checkpoints resume that smoke. Both endpoint
and score scope are included in the checkpoint signature.

When the AoU perimeter blocks raw log downloads, `aou_diagnostic.wdl` can
inspect a failed task's stderr inside the same workspace. It emits only fixed
software-failure labels, never exception text or participant information. Its
runtime checks the VM identity, uses one CPU, and has a 90-second command cap.
It does not grant local access to the underlying log.
The analysis also writes fixed stage/failure labels under its checkpoint
object's `.status/` prefix. Those labels distinguish input parsing, cohort
support, missing scores and completion without exposing exception text.

## Runtime and iteration budget

Stage wheels satisfying `aou_requirements.txt` once, including the pinned
gamfit wheel and all its transitive dependencies. The task installs only
binary wheels with `--no-index`; it never compiles or downloads dependencies.
Use a Linux Python 3.12 runtime image that supplies the system libraries those
wheels require. The runtime image digest, installed versions, gamfit build
information, source/input hashes, and query IDs are recorded in provenance.

The default run contains one selected disease, at most `max_rows_per_disease`
outcome-blind sampled rows each, four CPUs, 16 GiB RAM, and 50 GiB disk.
The cap must fit the measured fit budget. `fit_budget` records one measured
stage (gamfit version, solver threads, training rows, the slowest concurrent
fit's wall seconds, and where it was measured), a cost exponent and the stage
budget, at most `fit_timeout_seconds`. Validation refuses a cap whose final
stage (`train_fraction` of the cap) would need more than `budget_seconds` at
`wall_seconds × (rows / training_rows) ** exponent`, and the runner refuses a
task whose solver thread count differs from the measurement or whose
`gamfit._rust` binary does not have the measured `engine_sha256`: a version
names source, not a build, and a dev-profile wheel fits many times slower than
a release one (`null` records a measurement no engine matches). An exponent of 1
is the linear lower bound and over-states the cap when the cost is
superlinear; measure it at two sizes before raising the cap far past the
measurement. Provenance records the largest allowed cap as `fit_budget_rows`.
The budget is a compute bound and stays out of the checkpoint identity; the
cap itself does not.
One reference CTN is trained externally per endpoint. Development and final
evaluation require four cause-specific fits per endpoint and no AoU CTN fits.
The external trainer uses an explicit two-interior-knot CTN response basis.
The outcome fits use the configured time basis and run
sequentially with a checkpoint after each completed unit. Each fit has a
180-second wall cap, each query a 120-second cap and a billed-byte ceiling;
the command has a 30-minute cap and zero automatic retries. Temporary storage
and solver caches use the attached task disk. A child timeout terminates its
process group. Insufficient training events fail before fitting. The primary-score
smoke run records unsupported censoring-adjusted evaluation separately and emits
no accuracy estimate for those horizons; finite predictions do not establish
calibration. Final evaluation still requires censoring
support, pooled where a stratum lacks its own, before fitting. Increase the sample budget only after inspecting that
signal. Failed-worker logs and partial models are checkpointed privately without
a completion receipt. Population event frequencies
are never replaced with a balanced case/control sample.

The hypertension pilot's original 5,000-row cap produced insufficient development
death events for the prespecified competing-death model. The 20,000-row cap uses
the same outcome-blind hash order and seed; it does not balance events or relax
the minimum event count. Resource and wall limits are unchanged.

Run static and synthetic contract validation on MSI using the existing warm
Python dependencies; no participant data needs to leave AoU. Run every test
module and every WDL, on a compute node pinned to your own cores rather than
the login node, whose CPU watchdog kills sustained work:

```bash
cd examples/biobank
taskset -c <cores> env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=2 \
  TMPDIR=<scratch dir> python -m pytest -q -p no:cacheprovider test_*.py
for wdl in *.wdl; do miniwdl check "$wdl"; done
```

The suite takes under a minute (93 tests on MSI acl42). GitHub Actions runs the
same commands on every push that touches `examples/biobank`.

The native acceptance check applies the saved external CTN to real public
reference predictors and fits synthetic survival outcomes. It checks a
PC-varying fit, held-out cumulative hazards, save/load and batch equivalence:

```bash
python examples/biobank/test_aou_runtime.py \
  --reference-ctn reference/ctn_primary/reference_ctn.tar.gz \
  --reference-table reference/reference.parquet \
  --projection-sha256 75ce487f80eb4c386abd21f8168ba547c3292e77db812eb4a7c65289171cd5e0 \
  --output reference/survival_acceptance
```

Keep that output directory between development attempts so the native fit can
reuse its persistent warm cache. A timeout is a failed acceptance check even
if partial artifacts exist. The deterministic contract tests check termination
of a fit when its controller receives SIGTERM.
