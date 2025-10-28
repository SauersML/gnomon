# Royston–Parmar Survival Model Integration Plan

## 1. Purpose and scope
Establish a first-class Royston–Parmar survival model family in `calibrate/` that fits a full-likelihood Fine–Gray subdistribution hazard using a single PIRLS implementation shared with existing GAM families. The plan covers data ingestion, basis construction, penalized optimization, prediction, calibration, testing, and artifact serialization needed for production-ready absolute risk estimation.

## 2. Architectural alignment
- Introduce `ModelFamily::Survival(SurvivalSpec)` alongside the existing GAM option. Dispatch solely on this family flag; do not route survival through link selections.
- Define a `WorkingModel` trait returning `eta`, gradient, dense Hessian, and deviance. Implementors (logistic, Gaussian, survival) all satisfy the trait so `pirls::run_pirls` maintains a single code path.
- The survival working model produces full gradients/Hessians and its own deviance. PIRLS consumes the dense Hessian directly; diagonal shortcuts remain optional optimizations for other families but survival does not depend on link-driven branching.

## 3. Data schema and ingestion
- Expect TSV columns: `age_entry`, `age_exit`, `event_target`, `event_competing`, `sample_weight` (default 1.0), plus covariates (`pgs`, `sex`, `pcs`, ...).
- Implement `SurvivalTrainingData` with two mutually exclusive event flags and per-subject sampling weights only. Drop the categorical `event_type` and any risk-set or pseudo-weight preprocessing.
- Add validation ensuring `age_entry < age_exit`, ages finite, indicators in {0,1}, and `event_target + event_competing ≤ 1`.
- Scoring inputs mirror training schema with optional horizons for batch risk evaluation.

## 4. Age transform and basis construction
- Use the guarded log-age transform `u = ln(age - a_min + δ)`, where `a_min = min(age_entry)` and `δ > 0` (e.g., 0.1). Store `(a_min, δ)` inside the model artifact and reuse during scoring to avoid drift.
- Evaluate baseline spline bases and derivatives on the transformed scale. Apply the chain rule factor `1 / (age - a_min + δ)` when converting to derivatives with respect to age.
- Apply a reference constraint to the baseline log cumulative hazard spline (e.g., anchor at a reference age). Persist the constraint transform so scoring reconstructs the same null-space removal.
- Maintain anisotropic penalties for the PGS×age interaction. Center interaction terms to prevent leakage into main effects.

## 5. Layout caching
- Extend `ModelLayoutKind::Survival` to hold:
  - `baseline_entry`, `baseline_exit` value matrices.
  - `baseline_derivative_exit` with chain-rule scaling applied.
  - Optional time-varying entry/exit value blocks and exit derivatives.
  - Covariate blocks for proportional hazards effects.
  - `AgeTransform { a_min, delta }` and the stored reference-constraint transform.
- Remove derivative storage at entry; hazard evaluation only uses exit derivatives.

## 6. Likelihood, gradient, and Hessian
- For each subject `i`, the log-likelihood is
  `ℓ_i = w_i [δ_i (η_i(a_exit) + ln(∂η_i/∂a (a_exit))) - (H_i(a_exit) - H_i(a_entry))]`,
  where `H_i(t) = exp(η_i(t))` and `δ_i = event_target_i`.
- Compute gradients via boundary evaluations using cached entry/exit designs. The Hessian sums dense outer products from event and cumulative hazard terms, leveraging the precomputed matrices.
- No risk-set, pseudo-weight, or Kaplan–Meier components are required. Left truncation is handled entirely by the entry evaluation.
- Add a soft monotonicity barrier: evaluate `∂η/∂u` on a dense age grid and penalize negative values with a smooth hinge (e.g., `softplus(-∂η/∂u)`). Include the barrier contribution in the penalized Hessian. Remove all hard derivative clamps.

## 7. PIRLS and REML integration
- `pirls::run_pirls` requests `WorkingModel::update` each iteration, receiving `WorkingState { eta, gradient, hessian, deviance }`. The solver adds penalties and solves `(H + S) Δβ = g` with Faer for all families.
- The survival deviance feeds directly into REML/LAML updates. Trace calculations reuse the dense Hessian supplied by the working state.
- Eliminate survival-specific shim functions such as `update_glm_vectors`; survival uses the same PIRLS loop as other families via the trait.

## 8. Prediction API
- Expose two primitives:
  1. `H(t)` for any age given covariates (`exp(η(t))` evaluated via cached designs).
  2. Conditional absolute risk over `[t0, t1]` using `CIF(t) = 1 - exp(-H(t))` and
     `risk = [CIF_target(t1) - CIF_target(t0)] / max(ε, 1 - CIF_target(t0) - CIF_competing(t0))`.
- Emphasize that `CIF_competing` must come from either a companion Royston–Parmar model or caller-supplied values. Remove any suggestion of Kaplan–Meier proxies.
- Quadrature is optional diagnostics only; default predictions rely solely on boundary evaluations of `H(t)`.

## 9. Calibration
- Calibrate on the logit of the conditional absolute risk (or CIF) using out-of-fold predictions.
- Include as features: base prediction, delta-method standard error derived from the penalized Hessian, and a bounded leverage metric. Drop KM-derived diagnostics and age hull distances.
- Keep the calibrator minimal to avoid overfitting while improving transportability.

## 10. Testing and diagnostics
- Unit tests: derivative correctness (values and chain-rule scaling), deviance monotonicity, left-truncation handling, and prediction monotonicity in horizon.
- Grid diagnostic: verify the soft monotonicity barrier activates on only a small fraction of ages; emit warnings with remediation hints when it does.
- Integration tests: CIF comparisons at named ages against trusted tooling, Brier scores with and without calibrator, and monotonicity of `H(t)` on observed ages.
- Remove benchmarks tied to quadrature or risk-set linear algebra—they no longer apply.

## 11. Artifact serialization
- Persist in the trained model artifact:
  - Baseline knot vector and spline degree.
  - Reference-constraint transform.
  - `AgeTransform { a_min, delta }`.
  - Column ranges for baseline, covariates, and interactions.
  - Penalized Hessian (or its factorization) for delta-method variances.
  - Optional handle to a companion competing-risk model when present.
- Exclude any caches related to censoring survival, risk sets, or pseudo-weights.

## 12. Implementation sequence
1. Update data loaders and schema validations for the new survival dataset shape.
2. Extend basis evaluation utilities with guarded log-age transform support and derivative computation including chain-rule scaling.
3. Build `ModelLayoutKind::Survival` with cached entry/exit designs, exit derivatives, age transform metadata, and reference constraint.
4. Implement the survival `WorkingModel`, including the soft monotonicity barrier penalty, and integrate with PIRLS.
5. Adapt REML/LAML to consume the survival deviance and dense Hessian trace contributions.
6. Extend the trained model artifact and scoring pipeline for boundary-evaluated predictions and conditional risk outputs.
7. Update calibration features to operate on logit risk and hook into the existing calibrator machinery.
8. Add unit and integration tests covering derivatives, truncation, monotonicity, CIF accuracy, and calibrator performance.
9. Provide CLI and documentation updates describing the survival workflow, schema, prediction API, and companion model expectations.
