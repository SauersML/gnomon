# Royston–Parmar Fine–Gray Survival Integration Plan

## 1. Purpose and scope
This document defines the end-to-end design for adding Royston–Parmar flexible parametric survival models with Fine–Gray competing-risk support to the `gnomon` codebase. It addresses data ingestion, basis construction, penalized optimization, competing-risk likelihood evaluation, prediction APIs, and calibrator integration. The intended implementers are Rust developers familiar with the existing penalized GAM engine in `calibrate/`.

## 2. Current architecture audit
- **Data ingestion** (`calibrate::data`, `calibrate/data.rs`): Enforces a fixed TSV schema and returns `TrainingData`/`PredictionData` bundles with phenotype, PGS, PCs, sex, and weights.
- **Basis & penalties** (`calibrate::basis`, `calibrate/basis.rs`): Generates B-spline bases, applies constraints, and caches results. Penalty roots are assembled here.
- **Design assembly** (`calibrate::construction`, `calibrate/construction.rs`): Converts `ModelConfig` + `TrainingData` into dense design matrices and aligned penalty blocks via `ModelLayout`.
- **Optimization** (`calibrate::pirls`, `calibrate::estimate`): P-IRLS solves penalized normal equations using working responses/weights derived by `update_glm_vectors`; REML/LAML outer loop in `estimate.rs` updates smoothing parameters.
- **Model artifact & scoring** (`calibrate::model`): Serializes configuration/coefficients and reconstructs designs during prediction; orchestrates optional calibrator application.
- **Calibrator** (`calibrate::calibrator`): Fits secondary GAM using diagnostics from the base fit, reusing the same penalty and REML infrastructure.

The survival extension must respect these boundaries: reuse basis caching, hook into `ModelLayout`, leverage existing Faer-based solves, and emit metadata compatible with `TrainedModel`.

## 3. Statistical specification
### 3.1 Baseline model
- Time scale: attained age (years) with log transform `u = log(age)` to stabilize tail behavior.
- Baseline cumulative subdistribution hazard for event of interest: `H_0^*(a) = exp(f_0(u))`, with `f_0` represented via penalized B-spline basis.
- Individual subdistribution cumulative hazard: `H_i^*(a) = H_0^*(a) * exp(x_i^T γ + g_{pgs}(PGS_i, u))`, where:
  - `x_i` contains static covariates (PGS, PCs, sex) with proportional hazards coefficients `γ`.
  - `g_{pgs}` is a smooth varying-coefficient term for `PGS × age` built from tensor-product basis (PGS marginal basis ⊗ age basis without intercept column so proportional hazard is recovered when smooth is zero).
- Subdistribution hazard: `λ_i^*(a) = d/da H_i^*(a)` = `H_i^*(a) * (df_0/da + ∂g_{pgs}/∂a)`.

### 3.2 Likelihood with left truncation and competing risks
- Observed tuple per subject: `(a_entry, a_exit, δ_target, δ_competing, weight)`, where `δ_target` = 1 if event-of-interest by `a_exit`, `δ_competing` = 1 if competing event occurred first, and right-censor when both zero.
- Optimize the Fine–Gray **full likelihood** with the Royston–Parmar baseline supplying `η_i(a) = log H_i^*(a)`. Each subject contributes

  `ℓ_i = w_i [δ_i log λ_i^*(a_exit) - (H_i^*(a_exit) - H_i^*(a_entry))]`,

  where `δ_i = δ_target` and the cumulative hazard difference accounts for left truncation when `a_entry > 0`. This treats the subdistribution time-to-event as a fully parameterized model instead of relying on weighted risk-set ratios.
- Competing events correspond to `δ_target = 0` but still subtract their cumulative hazard through `H_i^*(a_exit) - H_i^*(a_entry)`, retaining Fine–Gray semantics while maximizing a bona fide likelihood.
- Left truncation is automatically enforced via the subtraction of `H_i^*(a_entry)`; no separate risk-set pruning is required.
- Denote by `d_i^{exit}` the derivative design row extracted from `D_baseline_exit` (and augmented with time-varying derivatives) so `(∂η_i/∂a)(t) = d_i^{exit} β̃` when evaluating hazards at `t = a_exit_i`. This derivative feeds both the hazard term and its gradient contribution.

### 3.3 Penalization
- Baseline smooth `f_0` penalized with difference penalty of order `ModelConfig::penalty_order` on the age spline coefficients.
- Varying-coefficient tensor product `g_{pgs}` penalized anisotropically (reuse `InteractionPenaltyKind`) combining PGS marginal penalty and age-direction penalty.
- Covariate main effects `γ` are unpenalized (enter penalty nullspace).

## 4. Data pipeline extensions
### 4.1 Schema changes (`calibrate/data.rs`)
- Extend TSV expectations to include columns: `age_entry`, `age_exit`, `event_target`, `event_competing`, `censoring_weight` (optional), `age_current` (for scoring), `horizon` (for evaluation dataset only).
- Update `TrainingData` with fields:
  ```rust
  pub struct SurvivalTrainingData {
      pub age_entry: Array1<f64>,
      pub age_exit: Array1<f64>,
      pub event_target: Array1<f64>, // 0/1
      pub event_competing: Array1<f64>, // 0/1
      pub censoring_weights: Array1<f64>, // default 1.0 if absent
      pub pgs: Array1<f64>,
      pub sex: Array1<f64>,
      pub pcs: Array2<f64>,
      pub prior_weights: Array1<f64>,
  }
  ```
- Provide survival-specific loaders `load_survival_training_data` and `load_survival_prediction_data` that validate monotone ages (`age_entry < age_exit`), finite values, event indicator exclusivity, and precompute Fine–Gray risk weights if provided externally.
- Maintain backward compatibility by keeping existing GAM loader; CLI will select survival loader when new flag `--survival` is passed.

### 4.2 Fine–Gray weight preprocessing
- Add helper to compute pseudo-risk weights `ω_i`:
  - Sort by `age_exit`.
  - Compute Kaplan–Meier estimate of censoring survival `G(a)` using `censoring_weights`.
  - For each record, assign `ω_i = prior_weights[i] * I(age_entry < age_exit) / G(a_exit)` with adjustments for left truncation `G(a_entry)`.
  - Store both `ω_exit` and `ω_entry` to reuse during iterations.

## 5. Basis & design construction updates
### 5.1 Baseline age spline
- Extend `basis::create_bspline_basis` to support evaluating both basis values and their derivatives with respect to log-age.
- Implement `basis::evaluate_derivative` returning `(B(u), B'(u))` for a vector of `u` values.
- Cache derivative evaluations keyed by `(knots_hash, degree, penalty_order, derivative=1)` using the existing LRU infrastructure.

### 5.2 Survival model layout
- Introduce new builder `construction::build_survival_design` alongside the existing GAM builder. Responsibilities:
  1. Transform ages to log scale with centering/rescaling stored in `ModelConfig`.
  2. Construct baseline spline matrix `X_baseline_exit`/`X_baseline_entry` and derivative matrices `D_baseline_exit` for hazard term.
  3. Build PGS marginal spline and PC smooths exactly as current GAM to maintain reuse of `BasisConfig`/penalty assembly.
  4. Create tensor product for `PGS × age` varying coefficient using `basis::tensor_product` with age marginal identical to baseline but drop the first column to avoid non-identifiable intercept.
  5. Package the matrices into an augmented `ModelLayout::Survival` variant encapsulating:
     ```rust
     pub enum ModelLayoutKind {
         Gam,
         Survival(SurvivalLayout {
             baseline_exit: Array2<f64>,
             baseline_entry: Array2<f64>,
             baseline_derivative_exit: Array2<f64>,
             age_scale: (f64, f64), // mean, std for log-age
             time_varying_exit: Array2<f64>,
             time_varying_entry: Array2<f64>,
         }),
     }
     ```
  6. Share penalty assembly logic: baseline smooth penalty uses `PenaltyMatrix` with derivative order from config; time-varying smooth uses anisotropic penalty.
- Ensure the `ModelLayout` retains ability to produce overall dense design for `pirls` (for linear algebra) while storing survival-specific matrices for gradient computations.

## 6. Survival log-likelihood, gradient, and Hessian
### 6.1 Linear predictors
For subject `i` define:
- `η_i^{exit} = X_i^{exit} β`, where `X_i^{exit}` is the concatenation `[B_exit, PGS_i, sex_i, PC_i, TP_exit]`.
- `η_i^{entry} = X_i^{entry} β`, sharing coefficients but with baseline/time-varying blocks evaluated at `age_entry`.
- `exp_term_i = exp(η_i^{exit})`, `exp_entry_i = exp(η_i^{entry})`.
- `ΔH_i = exp_term_i - exp_entry_i` (always ≥ 0 with monotone ages).

### 6.2 Event contributions
- Event density term: `δ_i log λ_i^*(a_exit)` requires derivative of baseline spline: `log λ_i^* = η_i^{exit} + log( d/da f_0(u_exit) + ∂g_{pgs}/∂a )`.
- Precompute `∂f_0/∂u` via derivative basis and scale by `1/age` from chain rule. Similarly compute derivative for time-varying term `PGS × ∂T/∂u`.
- Implement safe guard ensuring `d/da f_0 + ... > 0` by exponentiating a log-derivative parameterization or clamping to small positive constant (e.g., `1e-6`) before taking log.

### 6.3 Gradient/Hessian formulas
- Define the augmented design row `\tilde{x}_i^{exit}(t) = x_i^{exit} + d_i^{exit} / (∂η_i/∂a)(t)` so that derivatives of `log λ_i^*(t)` are evaluated consistently for both baseline and time-varying pieces.
- The score under the full likelihood becomes

  `U = Σ_i w_i [δ_i \tilde{x}_i^{exit}(a_exit) - (exp(η_i^{exit}) x_i^{exit} - exp(η_i^{entry}) x_i^{entry})]`,

  making the cumulative hazard contribution explicit through boundary-evaluated design rows. The implementation must therefore cache both `X_i^{exit}` and `X_i^{entry}` (for baseline and time-varying blocks) together with their exponentiated linear predictors `exp(η_i^{exit})`, `exp(η_i^{entry})` each iteration, rather than attempting to reuse a single pre-integrated design vector.
- Build the negative Hessian by differentiating the score directly: add `w_i δ_i (\tilde{x}_i^{exit})^⊤ \tilde{x}_i^{exit}` for the event term and subtract the integral of `H_i^*(t)` times the outer product of the design rows over the interval `[a_entry, a_exit]`. In practice approximate the integral with the cached cumulative hazard difference, yielding

  `H = Σ_i w_i [δ_i (\tilde{x}_i^{exit})^⊤ \tilde{x}_i^{exit} + ΔH_i x_i^{integral ⊤} x_i^{integral}]`,

  where `ΔH_i = H_i^*(a_exit) - H_i^*(a_entry)` still reuses the exponentiated boundary terms and `x_i^{integral}` denotes the design row used for the cumulative hazard evaluation constructed from the cached entry/exit matrices. Reuse dense cross-product helpers so PIRLS sees a familiar penalized normal-equation structure.
- The derivative of the time-varying smooth enters through `(∂z_i/∂a)`; cache these derivatives alongside basis evaluations so that hazard diagnostics and derivative-based penalties remain well defined.

### 6.4 Mapping to P-IRLS
- Augment `pirls::WorkingModel` abstraction:
  ```rust
  pub trait WorkingModel {
      fn update(&mut self, beta: &Array1<f64>) -> WorkingState;
  }
  pub struct WorkingState {
      pub eta: Array1<f64>,
      pub gradient: Array1<f64>,
      pub hessian: Array2<f64>, // dense p×p before penalties
      pub deviance: f64,
  }
  ```
- Implement `WorkingModel` for logistic, Gaussian (defer to current code), and new `RoystonParmarFineGray`. For GAM links, compute diagonal Hessian and map to existing `update_glm_vectors`. For survival, compute dense Hessian. This retains a single P-IRLS loop while letting each model supply `gradient`/`hessian`.
- Inside `pirls::run_pirls`, replace diagonal `weights` logic with general Hessian accumulation: solve `(H + S) Δβ = g` using Faer; if Hessian is diagonal, we can form `X^T W X` as today; otherwise reuse gradient/Hessian directly to build the penalized system without constructing pseudo responses.
- Maintain workspace reuse by storing `H` in `PirslWorkspace::xtwx_buf`; ensure symmetry and add ridge if eigenvalues approach zero.

### 6.5 Deviance for REML/LAML
- Define survival deviance as `-2ℓ` using the Fine–Gray log-likelihood with weights and penalty adjustments. Provide `calculate_survival_deviance` called by REML to keep gradient consistency.
- Update `estimate.rs` to branch on `ModelLayoutKind` for deviance, gradient, and `PirslResult` extraction.

## 7. REML/LAML integration
- Extend `ModelConfig` with `enum ModelFamily { Gam(LinkFunction), Survival(SurvivalSpec) }`. `SurvivalSpec` stores:
  ```rust
  pub struct SurvivalSpec {
      pub age_basis: BasisConfig,
      pub num_age_knots: usize,
      pub time_varying_basis: BasisConfig,
      pub competing: bool,
      pub firth_bias_reduction: bool, // not used but kept for API parity
  }
  ```
- Update `estimate::train_model` to dispatch:
  1. Build appropriate design layout via `ModelFamily`.
  2. Instantiate `pirls::WorkingModel` accordingly.
  3. Compute REML objective using survival deviance and penalty traces. Derive trace term `tr(W^{-1}S)` from Hessian as in GAM but using eigen decomposition of Hessian (Faer already available).
- Ensure smoothing parameter gradient uses new Hessian as base; reuse `compute_penalty_square_roots`.

## 8. Prediction and scoring API
### 8.1 Baseline survival reconstruction
- Extend `TrainedModel` with survival metadata:
  ```rust
  pub struct SurvivalModelArtifacts {
      pub age_knots: Array1<f64>,
      pub age_transform: (f64, f64),
      pub baseline_coeffs: Vec<f64>,
      pub time_varying_coeffs: Vec<f64>,
      pub gamma: Vec<f64>,
      pub competing: bool,
  }
  ```
- Provide helper `evaluate_cumulative_hazard(age, covariates)` computing `H_i^*(age)` and derivative via stored bases.

- Add method `TrainedModel::predict_survival_risk(age_current, horizon, pgs, pcs, sex, calibrate: bool)` returning conditional cumulative incidence between `age_current` and `age_current + horizon`:
  ```
  let cif_target = |t| 1.0 - (-H_i^*(t)).exp();
  let cif_comp = competing_cif(t); // recovered from Fine–Gray preprocessing
  let numer = cif_target(age_current + horizon) - cif_target(age_current);
  let denom = (1.0 - cif_target(age_current) - cif_comp(age_current)).max(1e-12);
  let risk = numer / denom;
  ```
  - Accept vectorized inputs for batch scoring. Provide CLI subcommand `score survival` accepting TSV with columns `age_current`, `horizon`, `score`, `sex`, `PC*`.

### 8.3 Calibration integration
- Extend calibrator feature computation to survival context:
  - Baseline predictor: log cumulative incidence ratio at requested horizon.
  - Standard error: delta method using Hessian inverse; reuse `pirls::penalized_hessian_transformed` to compute variance of log cumulative hazard difference.
  - Distance-to-hull: reuse existing `PeeledHull` but ensure age dimension added? (Option: keep PGS/PC hull only; age handled via safe range check.)
- Calibrator link remains `Logit` but on absolute risk; calibrator design uses same code path with new feature adapter bridging survival outputs to calibrator inputs.

## 9. Competing risk handling
- Represent event state via two indicator vectors; compute Fine–Gray weights once during preprocessing.
- Provide optional cause-specific hazard output for debugging by exposing `predict_cause_specific_hazard(age)` (target cause only); but core API returns subdistribution risk.
- Store metadata about competing-risk handling (e.g., `competing_events_present: bool`, `num_competing_events: usize`) in `SurvivalModelArtifacts` for downstream compatibility checks.

## 10. CLI & configuration updates
- Update `cli/main.rs` to accept `--model-family survival` and optional age knot configuration flags.
- Provide default knot placements (e.g., 5 internal knots) with quantile spacing on log-age range.
- Extend CLI scoring to read survival models and output risk across user-specified horizons with calibrator applied when available.
- Ensure serialization/deserialization in `model.rs` handles new `ModelFamily` variant while preserving backward compatibility for existing GAM models.

## 11. Testing & validation strategy
### 11.1 Unit tests
- `basis::tests`: verify derivative evaluations against finite differences for log-age basis.
- `construction::tests`: ensure survival layout builds consistent baseline/time-varying matrices and penalties (dimensions, null spaces).
- `pirls::tests`: create synthetic survival dataset with known parameters; compare estimated coefficients against analytic solution (simulate exponential hazards where Royston–Parmar reduces to linear model).
- `estimate::tests`: end-to-end training on toy data; check REML convergence and monotonic cumulative incidence.
- `model::tests`: verify serialization/deserialization of `SurvivalModelArtifacts` and risk predictions.
- `calibrator::tests`: ensure calibrator attaches to survival models and outputs probabilities within [0,1].

### 11.2 Integration tests
- Compare risk predictions against reference implementation (`rstpm2::stpm2cr`) on matched dataset (load offline results into fixtures). Validate log-likelihood within tolerance and horizon-specific risk differences < 1e-3.
- Stress-test left truncation: simulate cohort with delayed entry and confirm monotonicity of predicted risk vs. horizon.
- Competing-risk edge cases: dataset with only censoring, only target events, or only competing events; ensure training handles gracefully (guard rails + informative errors).

### 11.3 Numerical diagnostics
- Evaluate Hessian eigenvalues across iterations; assert minimum eigenvalue > -1e-8 after ridge to catch instability.
- Confirm `ΔH_i` remains non-negative; if not, log warning and apply floor.
- Validate calibrator inputs by checking mean absolute calibration residual on held-out dataset.

## 12. Performance considerations
- Precompute `log-age` vectors and share across baseline/time-varying evaluations to avoid repeated `ln`.
- Cache `exp(η)` terms and Fine–Gray weights between gradient/Hessian computations within an iteration.
- Use Faer batched GEMM to form Hessian contributions: treat baseline/entry matrices as separate blocks and accumulate into shared buffer.
- Exploit sparsity: baseline/time-varying design matrices can remain dense but low-rank; ensure `PirslWorkspace::xtwx_buf` sized accordingly.
- For large cohorts, compute pseudo-risk weights using streaming approach to avoid storing full risk set matrix (O(n) memory).

## 13. Implementation sequence & checkpoints
1. **Data layer**: introduce survival loaders, update CLI gating. Validate TSV parsing via unit tests.
2. **Basis derivative support**: extend `basis.rs`, add tests verifying derivative accuracy.
3. **Survival layout**: implement `ModelLayoutKind::Survival`, build baseline/time-varying design matrices, extend penalty assembly.
4. **Working model abstraction**: refactor `pirls` to use trait-based likelihood interface; ensure logistic/Gaussian paths unchanged (add regression tests to compare coefficients pre/post refactor).
5. **Survival likelihood implementation**: code gradient/Hessian, integrate with `pirls` iteration, add synthetic data tests.
6. **REML integration**: adapt outer loop for survival deviance, confirm smoothing parameter optimization converges on synthetic dataset.
7. **Model artifact & scoring**: extend serialization, implement risk prediction API, create CLI surfaces.
8. **Calibrator support**: adapt diagnostic extraction, fit calibrator on simulated survival outputs, verify probability bounds.
9. **Competing-risk validation**: compare against reference (imported CSV of benchmarks), write regression tests for absolute risk at multiple horizons.
10. **Documentation**: update `calibrate/README.md` and CLI docs to describe survival workflow, expected columns, and example commands.

## 14. Risks & mitigations
- **Non-concave likelihood**: enforce age monotonicity (`ΔH_i ≥ 0`) and apply adaptive ridge if Hessian loses definiteness.
- **Derivative underflow**: floor `d/da f_0` contributions and consider modeling log-derivative via separate smooth to guarantee positivity.
- **Pseudo-weight instability**: guard against small censoring survival `G(a)` by applying floor (e.g., 1e-6) and dropping observations beyond support.
- **Calibration mismatch**: provide switch to disable calibrator until survival diagnostics validated; default to disabled until tests confirm stability.

This plan provides all architectural decisions, mathematical details, and implementation checkpoints required to integrate Royston–Parmar Fine–Gray survival modeling into `gnomon` while respecting existing design principles.
