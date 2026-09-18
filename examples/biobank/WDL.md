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
`hwe_1kg_hgdp_gsa_v3` projection. A reference CTN, when one is declared (below),
must use that same model (uncompressed model SHA-256 is recorded in
`aou_analysis.json`). AoU's published PC coordinates are a separate coordinate
system and are not substituted.
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

The baseline has age, sex and a joint six-PC Duchon surface (8 centers).
The score surface has 8 centers and a time-constant signed slope. No frailty,
ensemble, manifold or post-hoc calibration stack is enabled. The baseline and
score-effect surfaces are separately penalized and jointly fitted.

## Score law

The score enters the outcome model as given. Each marginal-slope fit declares
the law of the score its marginal index is anchored on, and the saved native
model records that law and replays it at prediction. The anchoring equation
`Σ_k w_k Φ(α(a) + b(a)·u_k) = Φ(q(a))` has one solution on any finite declared
law, and the Gaussian closed form is its standard-normal instance, so the
marginal identity is a property of the law the model consumed. Nothing about
it requires the score to be normal.

With `score_law: declared_empirical` (the default in `aou_analysis.json`) the
declared law is the weighted empirical law of the training rows the outcome
model sees: the same development-training or outer-training span, the same
eligibility and landmark, the same rows. The two cause-specific fits pass the
raw `PGS` column as `z_column` with `latent_measure: global-empirical`; gamfit
compresses the law to at most 65 equal-mass nodes and standardizes it. No
transform of the score is fitted, on AoU rows or anywhere else, and the worker
refuses a saved model that did not anchor on that law or that fitted a latent
score transform. The `latent_measure` key needs a gamfit engine built from gam
at or after d2f73efe17 (gam#2923): gamfit 0.1.268 as released from gam v0.3.157
has no such key, so `fit_budget.engine_sha256` must name a build measured on
this declared fit. The anchored survival kernel is rigid: a baseline time wiggle,
a score-warp or link-deviation block, a follow-up-varying slope or several
scores have only the closed form, so gam refuses them together with a declared
law before fitting. The pilot's outcome model uses none of them, and gnomon's
calibrate refuses `--survival-time-wiggle` with the empirical law up front.

The declared law is pooled over the context (age, sex and PCs). Its adequacy
within a context stratum is measured rather than assumed: the report's
`score_diagnostics.declared_law` gives, for the pooled training law and for
each held-out audit group, the standardized score's mean, SD, skewness, excess
kurtosis, central-95 fraction and KS distance to the pooled training law. The
per-group `mean_risk_discrepancy` in the evaluation is the held-out
`E[p̂ | a]` check the design uses for acceptance. Per-context laws follow when
gamfit anchors on local empirical laws.

The reference-panel CTN remains only as an explicit Gaussian declaration,
`score_law: reference_ctn_gaussian`. It is fitted once per PGS on an external
genetic reference panel, conditional on PCs only, and never on AoU rows. There
are no internal AoU CTN folds. CTN uses `transformation_score`, never its
conditional-mean `predict` operation, and its output is declared standard
normal. GAM's native model embeds the saved CTN and replays it at prediction,
with save/load and batch-invariance checks; no second normalization or
influence absorber is fitted. That declaration estimates the reference-panel
score distribution; it does not establish normality in AoU conditional on age,
sex or baseline eligibility, so the same diagnostics assess it. Reference
archives are staged exactly when this declaration is selected; missing,
duplicate, mismatched or corrupt reference transforms stop the run.

`reference_ctn.py` trains and packages the external model from a real reference
table containing `sample_id`, the matching `PGSnnnnnn_AVG`, and projected PCs.
Its metadata names the actual panel, score-file hash and projection-model hash.
Training and preprocessing run on MSI with external data; the resulting model
is staged into the workspace. The reference panel's PGS calculation must use
the same allele, weight and score-scaling conventions as the target cache.
Variant coverage and projection-marker overlap require a transport audit.

Keep age, sex and PCs in the outcome model under either declaration and
evaluate the held-out score-distribution and risk diagnostics.

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
Loss uncertainty is conditional on fitted models; the censoring model's own
estimation error is carried as a separate standard error in every reported cell.

Censoring uses a training-only model of every reported-ancestry stratum over the
pooled training set. The pooled reverse Kaplan–Meier increments are grouped into
20 intervals of equal pooled censoring mass. In each, a stratum's censorings O_j
are set against E_j, the censorings its own risk set would show at the pooled
rate, with two levels of Poisson–gamma shrinkage: the stratum's level
R = (O + A) / (E + A) over its whole follow-up, and the interval ratio
θ_j = (O_j + a·R) / (E_j + a), with A = a = 1 censoring at the pooled rate. Its
censoring survival is G(t) = ∏ (1 − dΛ_pool)^θ_j over the pooled times up to t. A
well-supported interval takes about the stratum's own ratio; a sparse one borrows
the stratum's own level rather than the pooled set's, so a large stratum censored
differently is not pulled toward everyone else, and a thin one still has a defined
curve. A, a and the 20 intervals were fixed before the Monte Carlo below and not
tuned to it.

Positivity: a horizon is refused where the pooled training set cannot support it
(pooled censoring survival below 0.05, or nobody followed past it), or where, for
any test stratum, the upper one-sided 95% bound on its modelled censoring survival
at the horizon is below 0.05. Below that floor IPCW weights exceed 20 and the
horizon has no stable estimator, so a refusal is the correct output. Refusing on
the upper bound rather than the point estimate refuses only where positivity
confidently fails: a stratum whose estimate sits just under the floor by noise
reports, instead of the refusal selecting the samples where its estimate came out
high and biasing every reported cell low. The refusal row names each refused
stratum with its reason, modelled censoring survival and upper bound (both withheld
under the reporting minimum); the digest emits
`…__censoring_refused__h<h>__<stratum>__<reason>`. A reported horizon carries a
`pooled_censoring` row naming the strata without their own reverse Kaplan–Meier
support (under 20 training rows, own censoring survival below 0.05 or nobody
followed past the horizon), whose censoring rests on the model with little of their
own data there, with the reason, mean IPCW weight and censoring level R (both
withheld under the minimum); the digest emits
`…__censoring_pooled__h<h>__<stratum>__<reason>`. The mean IPCW weight is one in
expectation under the right censoring model; well below one it shows the stratum's
censoring curve is too high.

Every reported cell carries the standard error the censoring model's estimation
adds, by a delta method over every training censoring count taken as Poisson and
carried through the pooled increments, the expected counts, the level and the
interval ratios; the 95% interval that combines it with the group-robust
test-sampling error; and the Kish effective sample size of its IPCW weights. The
metrics and digest carry them as `brier_censoring_standard_error`,
`brier_95_lower`, `brier_95_upper`, the same three for `ipcw_observed_risk` beside
its `ipcw_observed_risk_standard_error`, the same three for `brier_difference`,
and `ipcw_weight_n_eff`.

Validation (gnomon#2338): a planted-censoring Monte Carlo of 400 replicates of a
100,000-row cohort with the pilot's ancestry shares, in three scenarios (shared
censoring; two strata losing contact three times as fast; two strata with no
follow-up past 4.5 years), set the model against the planted censoring survival on
the same replicates in 48 reportable cells, each for the Brier score and observed
risk. The censoring interval covered the oracle value in 92.3–96.8% of replicates
and the total interval the population value in 91.4–96.8%, with ratios of empirical
to delta-method standard deviation 0.90–1.07. The no-follow-up 5-year horizon was
refused in 400 of 400 replicates, and shared censoring refused 2 of 400. The
5-year bias of the fast-losing stratum fell from −0.0063 under the previous rule
(own reverse Kaplan–Meier where supported, a pooled stand-in refused on its point
estimate otherwise) to −0.0006, within Monte Carlo error, and its refusals from 248
to 61. Squared error was no worse than the previous rule's beyond 2 Monte Carlo
errors except in the four cells described below.

Acceptance was on a multiple-testing basis. Bias against the oracle exceeded 2
Monte Carlo errors in 6 of the 96 comparisons, at 2.0–2.3 errors, against about
4–5 expected by chance at that threshold. Three were shared-censoring 5-year
cells at +2.0–2.1, the overall one exactly where the previous rule sits on the same
draws. The other three are a known small bias: where a stratum's censoring ratio
climbs toward a follow-up cutoff (the planted end of follow-up at 4.5 years),
shrinking its intervals toward its level smooths the climb, so its 3-year observed
risk comes out about +0.0003 above the oracle (+2.3 Monte Carlo errors, about 0.25%
of the risk) and the overall cell by +2.1–2.2 errors, and its squared error at 1
and 3 years exceeds its own reverse Kaplan–Meier's by 3.0–3.7 errors (about 0.4% in
root-mean-square error at 3 years). The bias scales with the interval prior a: one
such cell at a = 0.5, six at a = 2. The stratified model
assumes sufficient independence within those strata and does not account for
all site, calendar-period or clinical dependence. No result from this pilot
establishes optimal deployment accuracy. The marginal identity is a model
property, not observed-outcome calibration.

The first run should use `--prepare-only`: it checks real score availability,
cohort fields, event counts and horizon support without fitting models.
After that and native acceptance pass, `--smoke-only` fits the first
prespecified score using only the development split: two cause-specific outcome
models anchored on the declared law of the development-training scores (or on
a frozen external CTN under the Gaussian declaration), with the declared law's
per-stratum diagnostics. Only that primary score must be cached for the
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

`submit_aou.py` reads these environment variables. They must describe
real resources accessible to the selected workspace; no placeholder JSON is
used. All are required except `AOU_REFERENCE_CTN_URIS`, which is set only under
the Gaussian declaration:

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
| `AOU_REFERENCE_CTN_URIS` | Space-separated staged external CTN archives, one per requested PGS; only with `score_law: reference_ctn_gaussian` |
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
gamfit wheel and all its transitive dependencies. The declared-law anchor needs
a gamfit release that carries the declared-law survival anchor
(SauersML/gam#2923); `gamfit_version` in the analysis configuration must match
the pinned wheel. The task installs only
binary wheels with `--no-index`; it never compiles or downloads dependencies.
Use a Linux Python 3.12 runtime image that supplies the system libraries those
wheels require. The runtime image digest, installed versions, gamfit build
information, source/input hashes, and query IDs are recorded in provenance.

The default run contains one selected disease, at most `max_rows_per_disease`
outcome-blind sampled rows each, 16 vCPUs, 32 GiB RAM, and 50 GiB disk.
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
Development and final evaluation require four cause-specific fits per endpoint
and no AoU CTN fits. Under the Gaussian declaration one reference CTN is
trained externally per endpoint with an explicit two-interior-knot CTN
response basis.
The outcome fits use the configured time basis. A stage's four fits run side
by side under one `fit_timeout_seconds` bound and publish a checkpoint about
every 30 seconds and after the stage completes. Each query has a 120-second cap
and a billed-byte ceiling; the command has a 150-minute cap and zero automatic
retries after a failure. Temporary storage
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
death events for the prespecified competing-death model. Later caps use the same
outcome-blind hash order and seed; they do not balance events or relax the
minimum event count.

The 25,000-row cap and `fit_budget` come from one measurement. The release-pypi
gamfit build of gam 18b1a6e353 ran the declared-law stage (four concurrent fits,
equal thread shares, 16 solver threads) on 16 EPYC 7702 (Rome) cores with one
thread per core, on a synthetic partition shaped like the pilot's. Medians of
three rounds: 349.2 s at 4,000 training rows and 695.2 s at 8,000 (exponent 0.99,
recorded as the linear bound 1). With 16 threads on 8 of those cores the stage
took 1.365 times as long (median of three paired rounds, used as 1.37), the bound
for a VM whose 16 vCPUs are 8 cores' hyperthreads. The cap bounds eligible cohort
rows; the final stage trains on `train_fraction` 0.8 of them and development on
0.75 of that. The worst case counts every cohort row as a training row.

| Step | Final stage | Development stage |
| --- | --- | --- |
| (1) Training rows at the 25,000-row cap | 20,000 (worst 25,000) | 15,000 (worst 18,750) |
| (2) Validator scaling, 695.2 s × rows / 8,000, 16 MSI cores | 1,738 s (worst 2,173 s) | 1,304 s (worst 1,629 s) |
| (3) × 1.37 for the 16-vCPU AoU VM | 2,381 s (worst 2,976 s) | 1,786 s (worst 2,232 s) |
| (4) Against the 4,500 s fit timeout | 1.89× (worst 1.51×) | |

Both stages plus about 10 minutes of setup, scoring and queries take about 79 of
the command's 150 minutes (worst 97). `budget_seconds` = floor(4,500 / 1.5 / 1.37)
= 2,189. The 1.5 is a declared timeout safety margin: an operating policy for
extrapolating past the measured sizes and from a synthetic to a real partition,
not an accuracy constant. The validator then allows 8,000 × 2,189 / 695.2 / 0.8 =
31,487 cohort rows. The task runs 16 vCPUs at `RAYON_NUM_THREADS=16` with 32 GiB
(the largest fit worker peaked at 298 MiB at 8,000 training rows; scoring and
cohort preparation ran in a 16 GiB task), and every fit of a stage gets an equal
share of the threads: the death fits' former three quarters left the 2-thread
disease fits unfinished at 900 s where equal shares finished the stage in 500 s.

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

The native acceptance check fits synthetic survival outcomes on real public
reference predictors twice: anchored on the declared law of the training
scores, and under the frozen reference-CTN Gaussian declaration. Each arm
checks a PC-varying fit, held-out cumulative hazards, save/load and batch
equivalence:

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
