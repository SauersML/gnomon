# LAML Stationarity Test Failure (Binomial)

## Symptom
Running `cargo test calibrate::calibrator::tests::laml_stationary_at_optimizer_solution_binom` panics because the KKT residual check reports `Inner KKT residual norm should be small, got 1.071710e2`.【a4e34c†L1-L86】

## What the test is doing
The stationarity test builds the calibrator design matrix and also receives the identity offset that must be added to the smooth terms during scoring.【F:calibrate/calibrator.rs†L5794-L5838】
Later in the same test it recomputes the working response and the KKT residual. However, it forms the linear predictor as `eta = X beta` and never adds the stored offset before evaluating the logistic mean, weights, and residuals.【F:calibrate/calibrator.rs†L5852-L5899】

## Why that is wrong
During normal fitting the offset is always included in the model: `optimize_external_design` forwards the data and offset into the penalized IRLS machinery (`RemlState::new_with_offset`), so the optimizer solves for `eta = offset + X beta`.【F:calibrate/estimate.rs†L1133-L1176】
Without the offset the recalculated probabilities collapse toward 0.5, producing a large mismatch between `z` and `X beta` and therefore an apparent KKT violation even though the optimizer converged.

## Confirming the hypothesis
Instrumenting the test to compare both formulations shows the issue directly: using the raw `X beta` gives a residual norm of roughly `1.072e2`, while including the offset lets the assertion pass (residual below `1e-4`).【b12cbc†L1-L126】【b12cbc†L127-L129】

## Root cause
The KKT residual verification inside `laml_stationary_at_optimizer_solution_binom` ignores the calibrator offset when reconstructing the linear predictor. This mis-specified predictor makes the residual appear huge and triggers the test failure even though the fitted solution is valid.
