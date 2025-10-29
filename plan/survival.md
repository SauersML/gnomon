# Survival Royston–Parmar Model Architecture

## 1. Purpose
Deliver a first-class survival model family built on the Royston–Parmar (RP) parameterisation of the cause-specific cumulative hazard. The design must:

- share the existing basis, penalty, and PIRLS infrastructure with the GAM families while contributing its own gradient, Hessian, and deviance;
- expose a clean per-subject full-likelihood objective that respects delayed entry and competing risks without any risk-set or pseudo-weight preprocessing; and
- provide reproducible scoring with a guarded age transform, stored constraints, and prediction APIs that operate entirely through cumulative hazard evaluations.

## 2. Architecture Overview
### 2.1 Model family dispatch
- Extend `ModelFamily` to include `ModelFamily::Survival(SurvivalSpec)` and reuse `ModelFamily::Gam(LinkFunction)` for existing paths.
- Implement a single `pirls::WorkingModel` trait:
  ```rust
  pub trait WorkingModel {
      fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
  }

  pub struct WorkingState {
      pub eta: Array1<f64>,
      pub gradient: Array1<f64>,
      pub hessian: Array2<f64>,
      pub deviance: f64,
  }
  ```
- Logistic and Gaussian models continue to supply diagonal Hessians through this trait. The RP survival model returns a dense Hessian and its own deviance. `pirls::run_pirls` consumes `WorkingState` without branching on link functions.

### 2.2 Survival working model
- Implement `WorkingModel` for `WorkingModelSurvival`, which reads a `SurvivalLayout` and produces `η`, score, Hessian, and deviance each iteration.
- PIRLS adds the penalty Hessians and solves `(H + S) Δβ = g` using the existing Faer linear algebra. No alternate update loops or GLM-specific vectors are required.

## 3. Data schema and ingestion
### 3.1 Required columns
Expect TSV/Parquet columns (names fixed):
- `age_entry`, `age_exit` (years, `age_entry < age_exit`),
- `event_target`, `event_competing` (0/1 integers, mutually exclusive, both zero for censoring),
- `sample_weight` (optional, defaults to 1.0 and multiplies log-likelihood contributions directly),
- covariates: `pgs`, `sex`, `pc1..pcK`, plus optional additional columns already supported by the GAM path.

### 3.2 Training and scoring bundles
```rust
pub struct SurvivalTrainingData {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<u8>,
    pub event_competing: Array1<u8>,
    pub sample_weight: Array1<f64>,
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    // optional additional covariates handled in ModelLayout
}
```

```rust
pub struct SurvivalPredictionInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub sample_weight: ArrayView1<'a, f64>,
    pub covariates: CovariateViews<'a>,
}
```
- Loaders validate ordering, exclusivity, and finiteness. There is no construction of inverse-probability weights or risk-set slices.

## 4. Basis, transforms, and constraints
### 4.1 Guarded age transform
- Compute `a_min = min(age_entry)` and choose a small guard `δ > 0` (e.g., `0.1`).
- Map ages to `u = log(age - a_min + δ)` for both training and scoring.
- Store `AgeTransform { a_min, delta }` in the trained artifact and reuse it verbatim at prediction time.
- Apply the chain rule factor `∂u/∂age = 1/(age - a_min + δ)` wherever derivatives of `η(u)` are converted back to age derivatives.

### 4.2 Baseline spline and reference constraint
- Build a B-spline basis over `u` for the baseline log cumulative hazard `η_0(u)`.
- Apply a reference constraint via an explicit linear transform `ReferenceConstraint` that removes the null direction (e.g., fix `η_0(u_ref) = 0`). Store this transform alongside the basis metadata so scoring can reconstruct the exact constrained basis.
- Penalize the baseline spline with the existing difference-penalty machinery (order configurable).

### 4.3 Time-varying effects
- Optional tensor-product smooth for `PGS × age` reuses the same log-age marginal and includes anisotropic penalties.
- Center the interaction to prevent leakage into main effects; cache and serialize the centering transform.

### 4.4 Stored layout pieces
`SurvivalLayout` aggregates the cached designs:
```rust
pub struct SurvivalLayout {
    pub baseline_entry: Array2<f64>,
    pub baseline_exit: Array2<f64>,
    pub baseline_derivative_exit: Array2<f64>,
    pub time_varying_entry: Option<Array2<f64>>,
    pub time_varying_exit: Option<Array2<f64>>,
    pub time_varying_derivative_exit: Option<Array2<f64>>,
    pub static_covariates: Array2<f64>,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub penalties: PenaltyBlocks,
}
```
- Derivative matrices apply the chain-rule scaling so they already represent `(∂η/∂age)` contributions at exit. No derivative-at-entry cache is required.

## 5. Likelihood, score, and Hessian
### 5.1 Per-subject quantities
- `η_exit = X_exit β`, `η_entry = X_entry β`.
- `H_exit = exp(η_exit)`, `H_entry = exp(η_entry)`.
- `ΔH = H_exit - H_entry` (non-negative by construction of the cumulative hazard).
- `dη_exit = D_exit β` already on the age scale.
- Target event indicator `d = event_target`, sample weight `w = sample_weight`.

### 5.2 Log-likelihood
For subject `i`:
```
ℓ_i = w_i [ d_i (η_exit_i + log(dη_exit_i)) - ΔH_i ].
```
Competing and censored records have `d_i = 0` but still subtract `ΔH_i`. There is no auxiliary risk set.

### 5.3 Score and Hessian
- Define `x_exit` and `x_entry` as the full design rows (baseline + time-varying + static covariates).
- Let `x̃_exit = x_exit + D_exit / dη_exit` where the division is elementwise after broadcasting the scalar derivative.
- Score contribution:
```
U += w_i [ d_i x̃_exit - H_exit_i x_exit + H_entry_i x_entry ].
```
(The `H_entry` term enters with a positive sign because the derivative of `-H_entry` contributes `+x_entry`.)
- Hessian contribution:
```
H += w_i [ d_i x̃_exit^T x̃_exit + H_exit_i x_exit^T x_exit + H_entry_i x_entry^T x_entry ].
```
- `WorkingState::eta` returns `η_exit` so diagnostics (calibrator, standard errors) can reuse it.
- Devianee `D = -2 Σ_i ℓ_i` feeds REML/LAML.

### 5.4 Monotonicity penalty
- Add a soft inequality penalty to discourage negative `dη_exit`. Evaluate `dη` on a dense grid of ages (e.g., 200 points across training support). Accumulate `penalty += λ_soft Σ softplus(-dη_grid)` with a small weight (`λ_soft ≈ 1e-4`).
- Add the barrier Hessian/gradient to the working state like any other smoothness penalty. Remove any ad-hoc derivative clamping.

## 6. REML / smoothing integration
- The outer REML loop is unchanged. It now receives `WorkingState` with dense Hessians when the survival family is active.
- The penalty trace term uses the provided Hessian: compute `solve_cholesky(H + Σ λ S)` as already done for GAMs.
- No special-case link logic remains in `estimate.rs`; branching is solely on `ModelFamily`.

## 7. Prediction APIs
### 7.1 Stored artifacts
`SurvivalModelArtifacts` persist:
```rust
pub struct SurvivalModelArtifacts {
    pub coefficients: Array1<f64>,
    pub age_basis: BasisDescriptor,
    pub time_varying_basis: Option<BasisDescriptor>,
    pub static_covariate_layout: CovariateLayout,
    pub penalties: PenaltyDescriptor,
    pub age_transform: AgeTransform,
    pub reference_constraint: ReferenceConstraint,
    pub hessian_factor: Option<CholeskyFactor>,
}
```
- The Hessian factor enables delta-method standard errors.
- Column ranges for covariates and interactions are recorded for scoring-time guards.

### 7.2 Hazard, survival, and cumulative incidence
- Evaluate the cause-specific log cumulative hazard `η_target(t)` by reconstructing the constrained basis at age `t` using the
  stored transform, and set `H_target(t) = exp(η_target(t))`.
- Differentiate to obtain `h_target(t) = dH_target/dt` using the cached derivative basis on the age scale.
- Gather cumulative hazards `H_c^{(j)}(t)` and hazards `h_c^{(j)}(t)` for every competing cause `j` active in scoring (see
  Section 7.3). Combine them to form the all-cause survival `S(t) = exp(-H_total(t))` where `H_total(t) = H_target(t) + Σ_j
  H_c^{(j)}(t)`.
- Compute the target cumulative incidence along a monotone age grid `t_0 < … < t_m` that spans the requested horizon by
  numerically integrating `S(t) h_target(t)`. Use adaptive Gauss–Kronrod or composite Simpson rules so the integral resolves
  rapidly changing hazards. For example,
  ```
  CIF_target(t_k) = Σ_{r=1}^k ∫_{t_{r-1}}^{t_r} S(u) h_target(u) du.
  ```
- Absolute risk between `t0` and `t1` is `ΔF = CIF_target(t1) - CIF_target(t0)`.
- Default `ε = 1e-12` to maintain numeric stability when forming conditional risks (see Section 7.4).

### 7.3 Competing risks
- Encourage fitting companion RP models for key competing causes. Scoring requires cumulative hazard and hazard evaluations for
  each such model so that `S(t)` reflects all causes. Accept either:
  - handles to `SurvivalModelArtifacts` for competing causes, exposing `H_c^{(j)}(t)` and `h_c^{(j)}(t)`; or
  - user-supplied cumulative hazard grids when external tooling provides them.
- Warn that providing only a single cause-specific model forces the system to assume `H_total = H_target`, which inflates the
  CIF and should be limited to sensitivity checks.
- Remove any suggestion of Kaplan–Meier proxies.

### 7.4 Conditioned scoring API
Expose:
```rust
fn cumulative_hazard(age: f64, covariates: &Covariates) -> f64;
fn cumulative_incidence(age: f64, covariates: &Covariates, competing: &CompetingContext) -> f64;
fn conditional_absolute_risk(t0: f64, t1: f64, covariates: &Covariates, competing: &CompetingContext) -> f64;
```
where `CompetingContext` supplies the age grid, cumulative hazards, and hazards for all relevant competing causes (Section 7.3).

## 8. Calibration
- Calibrate on the logit of the conditional absolute risk (or CIF at a fixed horizon).
- Features: base prediction, delta-method standard error derived from the stored Hessian factor, optional bounded leverage score.
- Use out-of-fold predictions during training to avoid optimism.
- Remove age-hull or KM-based diagnostics from calibrator features.

## 9. Testing and diagnostics
- Unit tests:
  - gradient/Hessian correctness via finite differences on small synthetic data;
  - deviance decreases monotonically under PIRLS iterations;
  - left-truncation: confirm `ΔH` equals the difference of endpoint evaluations;
  - prediction monotonicity in horizon (risk between `t0` and `t1` is non-negative and increases with `t1`).
- Grid diagnostic: monitor the fraction of grid ages where the soft barrier activates. If it exceeds a small threshold (e.g., 5%), emit a warning suggesting more knots or stronger smoothing.
- Compare with reference tooling (`rstpm2` or `flexsurv`) on CIFs at named ages and Brier scores with/without calibration.
- Add benchmarks that compare numerical integration accuracy against analytic solutions for simple hazard shapes.

## 10. Implementation roadmap
1. **Model family plumbing**: add survival variant to `ModelFamily`, update CLI flags, and ensure serialization handles the new branch.
2. **Data loaders**: implement survival-specific loaders with the two-flag event schema and guarded age transform metadata.
3. **Basis updates**: extend basis evaluation to emit values and derivatives on the log-age scale plus reference constraints.
4. **Layout builder**: construct `SurvivalLayout` with entry/exit caches and derivative matrices; serialize transforms.
5. **Working model**: implement the RP likelihood, gradient, Hessian, and soft barrier contributions.
6. **PIRLS integration**: refactor the solver to consume dense Hessians from `WorkingState` while preserving the GAM path.
7. **Artifact + scoring**: persist age transforms, constraints, and Hessian factors; implement scoring APIs that perform the
   survival-weighted quadrature for cumulative incidence.
8. **Calibrator**: adapt calibrator feature extraction to the survival outputs and ensure logit-risk calibration works end-to-end.
9. **Testing**: add unit/integration tests described above, including monotonicity and left-truncation checks.

## 11. Persisted metadata checklist
Store in the trained model artifact:
- baseline knot vector and spline degree;
- reference constraint transform (matrix or factorisation);
- `AgeTransform { a_min, delta }`;
- centering transforms for interactions and covariate ranges for guard rails;
- penalized Hessian (or its Cholesky factor) for delta-method standard errors;
- optional handles to companion competing-risk models.

With this plan the survival implementation is unified, risk-set-free, and fully reproducible across training and serving, while remaining compatible with the existing PIRLS and calibrator infrastructure.
