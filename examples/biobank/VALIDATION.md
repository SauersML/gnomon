# AoU workflow validation

Recorded 2026-09-11 UTC. Synthetic checks run on MSI; participant computation
and all participant artifacts remain inside the AoU workspace.

## Native artifact

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
penalty units and a genuinely unidentified control. Native CTN validation and
the exact-commit deployable runtime are still pending. This remains a candidate
explanation for the workspace failure, not proof that the two failures have
the same cause; no successful AoU training completion is claimed.

These are software and numerical checks. They do not establish disease-risk
calibration, predictive superiority, or a successful completed AoU training run.
