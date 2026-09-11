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

## Workflow contracts

All 27 deterministic contract tests passed on MSI. Both WDL definitions passed
static validation. These checks include identity rejection, cohort construction,
family/group folds, frozen transforms, checkpoint integrity, worker termination,
and preserving failed-worker evidence without a completion receipt.

The real hypertension cohort preparation exposed inadequate ancestry-specific
censoring support in the 5,000-person pilot. The smoke run now separates this
evaluation limitation from training-event support: unsupported horizons receive
no Brier estimate. Score selection still rejects unsupported development
metrics. Smoke support audits and model fits use development observations only.

These are software and numerical checks. They do not establish disease-risk
calibration, predictive superiority, or a successful completed AoU training run.
