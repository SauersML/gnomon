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

Follow-up ends at the earliest qualifying diagnosis, primary death record
from `aou_death`, or the covering observation interval's end. Observation
gaps are not bridged. Same-day disease/death ties are excluded because their
ordering is unknown. This predicts recorded diagnosis, not biological onset.
The resolved CDR release is recorded; horizons must be supported by that
release's actual follow-up.

PCs use `research_id`, `pca_features`, and `ancestry_pred` from the release's
ancestry file. The published relatedness-prune list is applied before sampling
or splitting. Its single-column numeric IDs may have the `research_id` header
or start on the first row; unknown headers and malformed rows are rejected.
Remaining person IDs are the pilot's split groups. This is not
full pedigree reconstruction or a claim that distant relatives are independent.

## Score selection and matched models

The prespecified pairs are COPD PGS004536/PGS001783, hypertension
PGS004525/PGS004603, and obesity PGS005199/PGS005331. The panel records exact
Catalog sources and pending component-provenance audits. PGS004787 is excluded
because its documented score development includes AoU. Public cohort metadata
does not establish participant-level non-overlap.

Every required score must exist in the real cache as a unique
`PGSnnnnnn_AVG` column with participant IDs. Missing scores are reported by
preflight and stop model fitting; no score is substituted. Score pairs use the
same complete-case cohort.

A seeded group split reserves 20% as outer test. The remaining development
sample is split 75/25 by group. Each candidate score gets the same CTN plus
PC-varying marginal-slope predictor, fitted only on development-training rows.
Mean development Brier score across the prespecified horizons selects the
score. The choice is recorded before outer-test evaluation. Selection fails
when either candidate lacks supported development metrics.

The selected PGS is then shared by all five methods refitted on outer training:

- Flexible baseline without PGS.
- Cross-fitted CTN with constant marginal slope.
- Cross-fitted CTN with PC-varying marginal slope.
- Conditional Gaussian location–scale normalization with the same PC-varying outcome.
- CTN with an ordinary varying-coefficient Gaussian transformation-survival model.

The baseline has age, sex and a joint six-PC Duchon surface (32 centers).
The score surface has 16 centers and a time-constant signed slope. No frailty,
ensemble, manifold or post-hoc calibration stack is enabled. The ordinary
comparator uses Gaussian location–scale survival; its time representation and
penalties differ, so this is not a pure unrestricted reparameterization test.

CTN and location–scale transforms condition on baseline age, sex and PCs.
Each internal fold fits its own transformation on the complement. A separate
full-training transform is saved for deployment. CTN uses
`transformation_score`, never its conditional-mean `predict` operation.
The outcome consumes frozen latent scores; no second normalization or influence
absorber is fitted. [GAM PR #2882](https://github.com/SauersML/gam/pull/2882)
supplies `frozen_score` and the saved
`CtnMarginalSlopeModel` used by the CTN marginal-slope bundles. A stock wheel
without that contract is rejected before analysis queries.

## Evaluation, persistence and limits

Separate disease and death components produce disease cumulative incidence.
This is not disease-only `1-S`. Saved predictions must be finite, monotone,
stable under save/load, row ordering and batch membership. The CIF grid
refinement error must be at most 0.001. Prediction horizons are explicit;
observed diagnosis/censoring times are excluded from prediction inputs.

Metrics include horizon-specific IPCW Brier score, mean-risk discrepancy,
fixed risk-bin calibration, and paired loss differences. Audits cover ancestry,
sex, age bands and training-defined PC neighborhoods, including points outside
their support. Sparse cells are suppressed, not certified as calibrated.
Paired uncertainty is conditional on fitted models and censoring estimates.

Censoring uses training-only ancestry-stratified reverse Kaplan–Meier. This
assumes sufficient independence within those strata and does not account for
all site, calendar-period or clinical dependence. No result from this pilot
establishes optimal deployment accuracy. The marginal identity is a model
property, not observed-outcome calibration.

The first run should use `--prepare-only`: it checks real score availability,
cohort fields, event counts and horizon support without fitting models.
After that and native acceptance pass, `--smoke-only` fits the first
prespecified score using only the development split: cross-fitted CTN and two
cause-specific outcome models. It performs the same persistence, batching,
monotonicity and CIF checks as the full analysis. It neither selects a score
nor evaluates the outer test set. Its completed development fits are reusable
by the full comparison through the same checkpoint.
The pilot caps rows, CPUs, query bytes/time and each fit's wall time.
A failed step raises an error; its process group is stopped.

Completed steps publish a workspace checkpoint so a later bounded job can
resume with `--resume gs://...`. Source, input and configuration hashes must
match; fitted-step receipts also require the same native engine. Checkpoints
contain participant data, scores, models and logs and must remain inside the
authorized workspace. WDL outputs are aggregate metrics, provenance **and
the sensitive checkpoint archive**; none is automatically approved for export.

Validation so far: 24 deterministic workflow contract tests and WDL validation
pass on MSI. The updated native CTN survival acceptance test and real cohort
preflight remain separate checks; synthetic contracts are not AoU results.

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
uses unique source/config-hashed paths and writes a submission receipt under
`examples/biobank/.aou-workflow/`. It submits exactly one task and does not
start a local polling process. Use Workbench's job UI to inspect/cancel it.
`--endpoint` narrows execution after the existing selector; it never overrides
eligibility. Run endpoints separately when a combined run would exceed its
wall budget. A checkpoint can move from preflight to smoke to full comparison
for the same endpoint, inputs and configuration. Keep the endpoint fixed when
resuming; its name is included in the checkpoint signature.

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

The configured panel contains up to three selected diseases, at most 5,000
outcome-blind sampled rows each, four CPUs, 16 GiB RAM, and 50 GiB disk.
With two internal folds, development selection and final matched comparisons
require 12 transform fits and 14 cause-specific fits per endpoint. They run
sequentially with a checkpoint after each completed unit. Each fit has a
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

The native acceptance check exercises cross-fitted CTN, a PC-varying fit,
held-out cumulative hazards, and save/load and batch equivalence on synthetic data:

```bash
python examples/biobank/test_aou_runtime.py --output .validation-logs/aou/native
```

Keep that output directory between development attempts so the native fit can
reuse its persistent warm cache. A timeout is a failed acceptance check even
if partial artifacts exist. The deterministic contract tests check termination
of a fit when its controller receives SIGTERM.
