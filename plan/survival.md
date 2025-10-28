# Survival Model Architecture Plan

This plan consolidates the survival implementation into a first-class Royston–Parmar model family that optimizes the full subdistribution likelihood with dense curvature, explicit age transforms, and consistent prediction/calibration behavior. It replaces all link-driven forks, risk-set helpers, and quadrature-dependent scoring paths.

## 1. Model family and PIRLS integration
- Extend `ModelFamily` with a `Survival(SurvivalSpec)` variant. Survival no longer routes through link functions.
- Implement a single `WorkingModel` trait returning `eta`, dense gradient, dense Hessian, and deviance. Logistic and Gaussian paths reuse it; survival supplies full curvature each iteration.
- The PIRLS core consumes the `WorkingState` directly. When the Hessian is dense (survival), solve `(H + S) Δβ = g` with Faer; when diagonal (logistic/Gaussian) the existing optimizations still apply because the WorkingModel encodes them.

## 2. Data schema
```rust
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,      // 0/1, mutually exclusive with event_competing
    pub event_competing: Array1<u8>,   // 0/1, mutually exclusive with event_target
    pub sample_weight: Array1<f64>,    // optional; defaults to 1.0
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
}
```
- Loaders validate `age_entry < age_exit`, indicator exclusivity, and positive weights.
- Prediction inputs mirror the same column names: `age_entry`, `age_exit`, `event_target`, `event_competing`, `sample_weight` (optional), covariates.

## 3. Age transform and basis layout
- Guarded log-age transform: `u = ln(age - a_min + δ)` with global `(a_min, δ)` stored in the model artifact.
- `AgeTransform` structure recorded in artifacts:
  ```rust
  pub struct AgeTransform { pub a_min: f64, pub delta: f64 }
  ```
- All basis evaluations (training and scoring) use the stored transform; chain rule factors contribute `1 / (age - a_min + δ)` wherever the hazard derivative appears.
- Baseline log-cumulative hazard spline uses this transformed variable. Cache entry/exit basis values and exit derivatives only (`baseline_entry`, `baseline_exit`, `baseline_derivative_exit`).
- Interaction blocks (e.g., PGS×age) reuse the same transform; cache derivative rows at exit with the chain rule applied.

## 4. Identifiability and constraints
- Apply an explicit reference constraint to the baseline spline to remove the null direction (e.g., anchor `η` at a reference age). Store the reference-transform matrix alongside basis metadata so scoring reconstructs the constrained design exactly.
- Maintain anisotropic penalties for tensor interactions and center them so interactions do not leak into main effects.

## 5. Likelihood, gradient, and Hessian
- Per-subject contribution for target indicator `δ_i` and weight `w_i`:
  ```
  ℓ_i = w_i [ δ_i (η_i(a_exit) + ln(∂η_i/∂a (a_exit))) - (H_i(a_exit) - H_i(a_entry)) ]
  H_i(t) = exp(η_i(t))
  ```
- `η` evaluated with cached entry/exit designs; `∂η/∂a` uses derivative rows scaled by the transform Jacobian.
- Gradient and Hessian accumulate per subject from boundary evaluations; the WorkingModel returns dense curvature (event term outer products plus cumulative-hazard contributions at entry/exit).
- Sampling weights multiply the likelihood terms directly. No pseudo-risk, risk-set, or Kaplan–Meier preprocessing exists anywhere in the pipeline.

## 6. Monotonicity enforcement
- Remove derivative clamps. Introduce a soft inequality penalty: evaluate `(∂η/∂u)` on a dense grid of ages; penalize negative slopes with a smooth hinge (e.g., `softplus(-∂η/∂u)` scaled by a small coefficient). This penalty contributes to the Hessian like any other smoothness penalty.
- Add training diagnostics that flag if the penalty activates on more than a tiny fraction of grid points; emit guidance to increase knots or smoothness when violated.

## 7. Layout and artifact contents
```rust
pub struct SurvivalLayout {
    pub baseline_entry: Array2<f64>,
    pub baseline_exit: Array2<f64>,
    pub baseline_derivative_exit: Array2<f64>,
    pub time_varying_entry: Option<Array2<f64>>,
    pub time_varying_exit: Option<Array2<f64>>,
    pub time_varying_derivative_exit: Option<Array2<f64>>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
}
```
- Model artifact persists baseline knot vector and spline degree, the reference-constraint transform, the guarded age transform, coefficient ranges for covariates/interactions, and the penalized Hessian (or factorization) for delta-method standard errors.
- Drop derivative-at-entry caches and any risk-set/pseudo-weight metadata.

## 8. Prediction API
- Provide primitives:
  1. `H(t)` evaluation for any age using stored layout.
  2. Conditional absolute risk over `[t0, t1]`:
     ```
     let H0 = H(t0);
     let H1 = H(t1);
     let cif_target = |t| 1.0 - (-H(t)).exp();
     risk = [cif_target(t1) - cif_target(t0)] /
            max(eps, 1.0 - cif_target(t0) - cif_competing(t0));
     ```
- Computation relies solely on boundary evaluations (`H(t1) - H(t0)`); Gauss–Kronrod or other quadrature stays disabled by default and exists only as an optional diagnostic toggle.
- Competing-risk denominator requires individualized `CIF_competing(t0)` from either a companion RP model or caller-supplied CIFs. No Kaplan–Meier proxies.
- Extrapolation guards clip ages to training support, flagging scores when outside hull/age ranges.

## 9. Calibrator
- Calibrate on the logit of the conditional absolute risk using:
  - Base prediction (`logit` of risk),
  - Delta-method standard error derived from the stored penalized Hessian,
  - Optional bounded leverage score for covariates (excluding brittle age-hull distances).
- Remove KM-derived diagnostics and hull-over-age features. Keep calibrator minimalist to avoid overfitting.

## 10. Testing and diagnostics
- Unit tests: derivative correctness (against finite differences), deviance monotonicity, left-truncation handling, prediction monotonicity in horizon.
- Grid check ensures the soft monotonicity barrier rarely activates; if it does, surface warnings with remediation steps.
- Integration tests compare CIFs at named ages and Brier scores (with/without calibrator) against external tooling.
- No benchmarks or diagnostics reference quadrature, risk sets, or pseudo weights.

## 11. Implementation sequence
1. Introduce survival data loaders with two-flag schema and `sample_weight` handling.
2. Add guarded log-age transform infrastructure and persist `(a_min, δ)` with reference-constraint metadata.
3. Build `ModelLayoutKind::Survival` with cached entry/exit values and derivatives at exit only.
4. Refactor PIRLS to consume `WorkingModel` outputs and share the dense-solver path across families.
5. Implement survival likelihood, gradient, Hessian, deviance, and monotonicity penalty inside the survival WorkingModel.
6. Serialize artifacts (knots, transform, constraint, Hessian factors) and update scoring to use boundary evaluations.
7. Expose prediction APIs (`H(t)`, conditional risk) and document reliance on companion competing-risk models.
8. Trim calibrator features to the minimalist set and update calibration/testing harnesses.
9. Add diagnostic tests for monotonicity, left truncation, and conditional risk bounds; ensure warnings fire when the monotonicity barrier activates excessively.

With these edits survival becomes a coherent, full-likelihood model family: no link-driven branches, no risk-set shims, exact age transforms, stable monotonicity enforcement, and prediction APIs grounded in cumulative hazards.
