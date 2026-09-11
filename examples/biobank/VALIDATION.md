# AoU workflow validation

Recorded 2026-09-11 UTC. Synthetic checks run on MSI; participant computation
and all participant artifacts remain inside the AoU workspace.

## External-reference revision

CTN training has moved out of AoU. The workflow only loads frozen reference
models and applies them; its former CTN fit/cross-fit command was removed.
PCs now use pgsEngine's reference projection, rather than AoU's separate
published PC coordinate system. Prior participant checkpoints therefore cannot
be resumed under this changed model specification.

All 27 current deterministic contract tests pass on MSI, including external
model checksum, score-ID and PC-projection checks. The updated WDL passes
miniwdl validation. Native acceptance now requires a real external reference
bundle and uses synthetic survival outcomes; this is separate from an AoU fit.

[GAM PR #2885](https://github.com/SauersML/gam/pull/2885) was merged after removing
test-only solver timing instrumentation rejected by the release scanner.
The exact-main runtime from commit `994668d561f1c71d075ad388974e86aed7f4b971`
was built by [workflow 34604995982](https://github.com/SauersML/gam/actions/runs/34604995982).
Its wheel SHA-256 is
`d681f6fe9cabef56f390e5a7f155d6f7f6df424edab2dc3bf0c9b940c2e48e50`.
The small-score six-PC CTN regression passed with this wheel in approximately
eight seconds. The previous synthetic survival pipeline exceeded its bounded
90-second outcome-fit budget, so that pipeline is not recorded as passing with
this runtime. No successful AoU outcome training is claimed.

## Real external CTN fit

The hypertension PGS004525 CTN completed on MSI in approximately 38 seconds
using 2,583 founders from the public GRCh38 1000 Genomes panel. This run used
1000 Genomes alone, not the combined HGDP+1000 Genomes panel. The frozen PC
projection matched 562,225 of 570,709 model markers. Gnomon normalized
1,059,364 of 1,059,939 Catalog score variants and matched 844,993 to the panel.
Per-sample zero missingness refers to those matched variants, not full Catalog
coverage. Target-score coverage still needs comparison inside AoU.

The six-PC, eight-center Duchon CTN produced finite scores and passed saved-model
replay, single-row/batch agreement and monotonicity over the observed score
range. Its model SHA-256 is
`fbdb249819a1a253c9a847cc1ddd18029c88d964dbe59e1a7b1e1182a26c6ebb`.
In-sample transformed scores have mean approximately zero, SD 0.9976, and
94.81% within ±1.96. Across the five superpopulation labels, means range from
−0.0234 to 0.0206 and SDs from 0.9905 to 1.0182. These diagnostics are not
held-out reference validation or a target-population normality certificate.

GAM returned a converged constrained mode but declined posterior covariance
because of seven flat directions. The artifact supports the tested point
transformation; coefficient uncertainty is not certified. No covariance
estimate was fabricated or substituted.

The external-transform survival handoff test initially rejected an invalid
four-center basis for six PCs. The test now uses eight centers, and the workflow
rejects undersized Duchon bases before fitting. With that correction, the
PC-varying synthetic survival fit exceeded its 90-second child limit. Its
process group was terminated and no completion was accepted. The reference
transform has been staged in the authorized workspace.

## Bounded real training run

On 2026-09-11, the real hypertension run with PGS004525 and the frozen 1000
Genomes CTN was submitted and reached Workbench `RUNNING`. It uses GAM commit
`994668d561f1c71d075ad388974e86aed7f4b971` and workflow source commit
`3044701910c66833193118fa9d033a704329a4b6`. The run is a development-only primary
score fit, with the outer test set untouched, at most 20,000 sampled cohort
rows, four CPUs, 16 GiB RAM, 180 seconds per fit and a 30-minute task limit.
It fits disease and competing-death components and records checkpoints inside
the authorized workspace. It passed real cohort preparation, then failed while
applying the external CTN, before the disease-model fit.

A public-reference MSI reproducer showed that passing the whole cohort frame
to CTN includes Arrow timestamp columns unsupported by GAM. The corrected
score adapter selects only PGS and the configured PCs at every transformation
call, excluding dates, outcomes, and identifiers. The saved real CTN replayed
successfully through that corrected adapter; all 27 contract tests and WDL
validation passed. This does not replace the separate survival convergence
check or establish successful AoU training.

## Prior native artifact and historical checks

- Release: `gamfit-0.1.267-cp310-abi3-manylinux_2_28_x86_64.whl`.
- SHA-256: `a908af29ff97ee5f5ee825f4cee7bd3f237189cdc0d529bb78e040b04b1290bd`.
- Build: [GAM workflow 34545871333](https://github.com/SauersML/gam/actions/runs/34545871333),
  from merged commit `945ab32903eaa7cee2c96ae810d06dc49d95e36a`.
- Runtime tested: Linux, CPython 3.12, two CPU threads.

`test_aou_runtime.py` passed using the persistent warm cache. It checks two
out-of-fold CTN transforms, the deployment transform, the frozen-score
PC-varying survival fit, finite monotone cumulative hazards, coherent
single-cause incidence, save/load agreement, and batch/order invariance.

The initial cold survival fit exceeded its 90-second limit. Its bounded
continuation reused the saved solver state and passed; the complete pipeline
replay then passed. This is evidence for checked warm execution, not evidence
that cold fits always meet that limit. Failed units must not receive completion
receipts or be substituted for fitted models.

The separate GAM `test_survival_marginal_slope_clustered_pc_808.py` raw-score,
attained-age Matérn regression still exceeded its 90-second child-fit limit
with this wheel. Invoking the test directly confirmed the numerical timeout
independently of pytest collection. It is not marked fixed or passing. The
AoU pilot uses the explicitly transformed, frozen-score Duchon pipeline above;
its checked completion does not certify the other input route.

## Workflow contracts

All 28 deterministic contract tests passed on MSI. Both WDL definitions passed
static validation. These checks include identity rejection, cohort construction,
family/group folds, frozen transforms, checkpoint integrity, worker termination,
and preserving failed-worker evidence without a completion receipt.
The workspace diagnostic can classify unfinished checkpoint logs while ignoring
completed workers and emitting only fixed labels; raw logs remain in AoU.

The real hypertension cohort preparation exposed inadequate ancestry-specific
censoring support in the 5,000-person pilot. The smoke run now separates this
evaluation limitation from training-event support: unsupported horizons receive
no Brier estimate. Score selection still rejects unsupported development
metrics. Smoke support audits and model fits use development observations only.

The subsequent real primary-score pilot stopped at the training-event guard;
the workspace diagnostic identified insufficient development death events.
No outcome fit was accepted. A subsequent pilot increased the cap to 20,000
using the same hash order and seed, retaining the event minimum and resource
caps. It passed cohort preparation and reached score transformation, then
failed in the first CTN worker. The workspace diagnostic reported fixed
`gam_integration_error`, `singular_system`, and `function_fit_transform` labels.
The failed worker evidence remains in its private workspace checkpoint.

A separate synthetic six-PC CTN fit completed on MSI in 6.87 seconds. With the
same generated observations and configuration, multiplying PCs by `1e-3` and
the score by `1e-5` caused GAM's identifiability audit to reject the fit (52 of
65 joint columns retained). This demonstrates sensitivity to numerical units
in this synthetic case. Follow-up isolation passed with only PCs rescaled
(6.41 seconds), but failed with only the score rescaled. The raw structural
penalty sum in GAM's rank audit depends on those response units.

[GAM PR #2884](https://github.com/SauersML/gam/pull/2884) normalizes the audit's
individual PSD penalties without changing fitted penalties or rank tolerances.
All 65 identifiability library tests passed on MSI, including extreme independent
penalty units and a genuinely unidentified control. The small-score native
regression subsequently passed on the runtime above. This remains a candidate
explanation for the workspace failure, not proof that the two failures have
the same cause; no successful AoU training completion is claimed.

These are software and numerical checks. They do not establish disease-risk
calibration, predictive superiority, or a successful completed AoU training run.
