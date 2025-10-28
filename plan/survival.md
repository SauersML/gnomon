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
- Provide survival-specific loaders `load_survival_training_data` and `load_survival_prediction_data` that validate monotone ages (`age_entry < age_exit`), finite values, event indicator exclusivity, and pass through optional user-supplied sampling weights without transformation.
- Maintain backward compatibility by keeping existing GAM loader; CLI will select survival loader when new flag `--survival` is passed.

### 4.2 Optional sampling weights
- Accept user-provided sampling or censoring weights via `censoring_weights` and default to ones when absent.
- Validate that supplied weights are positive and finite, then pass them through to likelihood assembly without additional preprocessing.
- Ensure downstream code multiplies likelihood contributions by these weights directly; no pseudo-risk or inverse-censoring weights are cached.

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
- Build the negative Hessian by differentiating the score directly: add `w_i δ_i (\tilde{x}_i^{exit})^⊤ \tilde{x}_i^{exit}` for the event term and accumulate the cumulative-hazard part as the sum of boundary outer products,

  `H = Σ_i w_i [δ_i (\tilde{x}_i^{exit})^⊤ \tilde{x}_i^{exit} + exp(η_i^{exit}) x_i^{exit} x_i^{exit ⊤} + exp(η_i^{entry}) x_i^{entry} x_i^{entry ⊤}]`.

  Cache the boundary design rows `x_i^{exit}`/`x_i^{entry}` so PIRLS can add their contributions independently instead of scaling a single integral row.
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
- When assembling the survival Hessian, accumulate `exp(η_i^{exit}) x_i^{exit} x_i^{exit ⊤}` and `exp(η_i^{entry}) x_i^{entry} x_i^{entry ⊤}` as separate rank-one updates so no shared integral row needs to be scaled in-place.
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
  let cif_comp = competing_cif(t); // computed from modelled competing-risk subdistribution
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
- Represent event state via two indicator vectors; rely solely on provided sampling weights (default ones) during likelihood evaluation.
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
- Cache `exp(η)` terms and other likelihood intermediates (e.g., cumulative hazard differences) between gradient/Hessian computations within an iteration.
- Use Faer batched GEMM to form Hessian contributions: treat baseline/entry matrices as separate blocks and accumulate into shared buffer.
- Exploit sparsity: baseline/time-varying design matrices can remain dense but low-rank; ensure `PirslWorkspace::xtwx_buf` sized accordingly.
- No additional preprocessing buffers are required for censoring adjustments beyond the direct likelihood inputs, keeping memory bounded by design matrix storage.

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
# Royston–Parmar Fine–Gray Survival Integration Plan

## 1. Objectives and Scope
1. Extend the `calibrate/` crate to support a Royston–Parmar flexible parametric survival model that uses attained age as the timescale and accounts for competing risks via a Fine–Gray subdistribution hazard formulation.
2. Preserve and reuse the existing penalized GAM infrastructure—basis generation (`basis.rs`), design assembly (`construction.rs`), P-IRLS solver (`pirls.rs`), and REML/LAML smoothing selection (`estimate.rs`)—while introducing survival-specific data structures, likelihood computations, and prediction logic.
3. Deliver an inference API capable of producing calibrated absolute risk between an individual’s current age and arbitrary future horizons, integrating with the existing post-hoc calibrator when enabled.
4. Maintain backward compatibility with binary and Gaussian models by isolating survival functionality behind new model types, configuration flags, and serialization fields.

## 2. Current Infrastructure Survey
- **Data handling (`calibrate/data.rs`):** Strict schema loader for phenotype, score, sex, PCs, weights. Needs extension to ingest survival-specific columns (entry/exit age, event type) and to produce survival-specific data bundles.
- **Basis and penalties (`calibrate/basis.rs`):** Provides B-spline basis construction, difference penalties, tensor product structures, and null-space constraints. We can reuse this machinery for the log-age baseline spline and for the PGS×age interaction smooth, but must add convenience wrappers for log-age transformations and monotonicity constraints.
- **Design layout (`calibrate/construction.rs`):** Builds the block-structured design matrix and penalty map recorded in `ModelLayout`. Survival mode will require new block descriptors (baseline age spline, hazard shift covariates, optional tensor interactions) and updated null-space handling.
- **Optimization (`calibrate/pirls.rs` & `calibrate/estimate.rs`):** Implements penalized iteratively reweighted least squares and nested REML optimization keyed off `ModelConfig::link_function`. We must add a new link variant with survival-specific working response and deviance calculations, plus support for the additional sufficient statistics demanded by the Fine–Gray full likelihood (exit/entry cumulative hazards, hazard derivatives, Jacobian factors).
- **Model artifact (`calibrate/model.rs`):** Serializes `ModelConfig`, stores design metadata, and executes prediction with optional calibrator adjustments. Survival models must capture age transformations, baseline spline coefficients, competing-risk metadata, and a horizon-aware prediction path.
- **Calibrator (`calibrate/calibrator.rs`):** Builds a secondary GAM using diagnostics from the base fit. Survival predictions will use different diagnostics (e.g., CIF standard errors, age-based hull distances) but can reuse the same P-IRLS machinery with tailored feature construction.
- **Hull guard (`calibrate/hull.rs`):** Currently handles convex hulls in PGS/PC space; we will extend to include attained age for extrapolation safeguards.

## 3. Mathematical Specification
### 3.1 Time scale and transformations
- Let `a` denote attained age in years. Training data provide `a_entry` (left truncation) and `a_exit` (event or censoring age). Define transformation constants: `a_min = min(a_entry)` and `δ = 0.1` years (guard). Map each age to `u = log(a - a_min + δ)` so the spline basis operates on a stabilized log-age scale. Record `(a_min, δ)` and `a_max = max(a_exit)` for prediction.
- Evaluate baseline basis functions at both entry and exit ages to compute cumulative hazard differences, a requirement for Fine–Gray risk increments.

### 3.2 Baseline cumulative hazard
- Represent the log cumulative baseline hazard `ℓ(a) = log H_0(a)` as a penalized B-spline smooth over `u`. Let `B(u)` be the basis matrix (rows = observations, columns = spline coefficients). The baseline contribution to the linear predictor for subject `i` at age `a_exit_i` is `η_{0,i} = B(u_exit_i) θ`. For left truncation we subtract the baseline evaluated at entry age: `η_{0,i}^{entry} = B(u_entry_i) θ`. The effective cumulative hazard contribution for the observation interval is `exp(η_{0,i}) - exp(η_{0,i}^{entry})`.
- Enforce identifiability via one of two equivalent constraints: (a) impose `ℓ(a_ref) = 0` at reference age `a_ref` (e.g., weighted median of `a_exit`) by subtracting `B(u_ref)` from each basis evaluation; or (b) drop the first column of the constrained basis (`basis.rs` already supports sum-to-zero transformations). Option (a) preserves interpretability and will be implemented by augmenting the basis builder to subtract `B(u_ref)` before applying difference penalties.
- Apply a second-order difference penalty on `θ` (wiggliness control) using existing `difference_penalty` function. Provide a configuration knob for penalty order in `ModelConfig` (default 2, optional 3 for extra smoothness).

### 3.3 Covariate effects
- The hazard multiplicative shifts are additive on the log cumulative hazard scale:
  - `β_pgs` for the raw polygenic score.
  - Optional `β_sex` (binary).
  - Linear coefficients `β_pc_j` for each principal component (default), with option to promote to smooths if future work requires.
  - Optional smooth interaction `f_{pgs×age}(PGS, u)` capturing non-proportional hazards. Implement as a tensor-product smooth with marginal bases: univariate B-spline in `PGS` (as already built for calibration) and the log-age spline `B(u)` with anisotropic penalties (existing `InteractionPenaltyKind::Anisotropic`). Center the interaction to ensure pure `ti()` semantics (requires extending current orthogonalization to include the new log-age marginal).
- Whenever covariate effects vary with age, cache both the basis values and their derivatives with respect to log-age so that hazard derivatives include the term `(∂z_i(u)/∂u)^T θ`. The derivative is needed for subdistribution hazard evaluation, deviance diagnostics, and any variance calculations that involve `d/dt` of the cumulative hazard.

### 3.4 Fine–Gray full likelihood
- For each subject `i`, define indicators and weights: `d_i = 1` when the target event occurs at `a_exit_i`, `0` otherwise; `c_i = 1` for competing events; and sample weight `w_i` (default 1).
- Let `η_i(t)` denote the log cumulative subdistribution hazard evaluated with the Royston–Parmar basis. The subdistribution hazard on the age scale is `λ_i^*(t) = exp(η_i(t)) (∂η_i/∂t)(t)`, so derivative design matrices must supply `(∂η_i/∂t)(t)` at the exit age. Left truncation is handled by subtracting the cumulative hazard at entry rather than by pruning risk sets.
- The per-subject log-likelihood for the full model is

  `ℓ_i = w_i [d_i (η_i(a_exit_i) + log (∂η_i/∂t)(a_exit_i)) - (H_i(a_exit_i) - H_i(a_entry_i))]`,

  with `H_i(a) = exp(η_i(a))`. Competing events set `d_i = 0` but still subtract the cumulative hazard increment `ΔH_i = H_i(a_exit_i) - H_i(a_entry_i)` so Fine–Gray semantics are preserved inside a true likelihood objective.
- Implementation steps:
  1. Precompute design matrices `X_exit`, `X_entry` for `η` at exit and entry ages, alongside derivative matrices `D_exit` that encode `(∂η/∂t)` via the chain rule on the log-age scale.
  2. During each IRLS iteration evaluate `η_exit = X_exit β̃`, `η_entry = X_entry β̃`, and `dη_exit = D_exit β̃`; form `H_exit = exp(η_exit)` and `H_entry = exp(η_entry)` with safeguards for numerical underflow.
  3. Compute cumulative hazard differences `ΔH = H_exit - H_entry` (clamped at zero) and the event hazard `λ_i^*(a_exit_i) = H_exit ⊙ dη_exit / a_exit_i` using cached Jacobians.
  4. Accumulate score and Hessian contributions per observation rather than per risk set, reusing cached designs for both event and integral terms.
- Define the augmented design row `\tilde{X}_i^{exit} = X_i^{exit} + D_i^{exit} / (∂η_i/∂t)(a_exit_i)` so differentiation of `log λ_i^*(a_exit_i)` collects both baseline and time-varying pieces.
- Score vector for coefficient block `β̃` becomes `U = Σ_i w_i [d_i \tilde{X}_i^{exit} - ΔH_i X_i^{integral}]`, where `X_i^{integral}` reuses entry/exit design rows to approximate the cumulative hazard gradient.
- Observed negative Hessian is `H = Σ_i w_i [d_i \tilde{X}_i^{exit ⊤} \tilde{X}_i^{exit} + ΔH_i X_i^{integral ⊤} X_i^{integral}]` plus penalty matrices. This dense matrix drops neatly into the PIRLS linear solves.
- Deviance for diagnostics is `D = -2 Σ_i w_i [d_i log λ_i^*(a_exit_i) - ΔH_i]`; track it per iteration for convergence checks and REML updates.

## 4. Data Schema and Preprocessing
### 4.1 Survival data structs
```rust
pub struct SurvivalTrainingData {
    pub pgs: Array1<f64>,
    pub sex: Array1<f64>,
    pub pcs: Array2<f64>,
    pub weights: Array1<f64>,
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_type: Array1<u8>, // 0=censor,1=target,2=competing
}
```
- Add `SurvivalPredictionInputs` for inference:
```rust
pub struct SurvivalPredictionInputs<'a> {
    pub pgs: ArrayView1<'a, f64>,
    pub sex: ArrayView1<'a, f64>,
    pub pcs: ArrayView2<'a, f64>,
    pub current_age: ArrayView1<'a, f64>,
    pub horizon_age: ArrayView1<'a, f64>, // same length as individuals or length-1 broadcast
}
```

### 4.2 Loader enhancements (`calibrate/data.rs`)
- Introduce `load_survival_training_data(path, num_pcs)` alongside existing loaders. Required columns: `score`, `sex`, `PC*`, `weights`, `age_entry`, `age_exit`, `event_type`. Optional `phenotype` is ignored in survival mode to prevent confusion.
- Validation logic:
  - Confirm `age_entry`, `age_exit` numeric, positive, finite.
  - Enforce `age_exit >= age_entry` with tolerance `1e-6`.
  - Map `event_type` strings/numbers to `u8`. Accept synonyms: `{0, "censor", "none"}`, `{1, "event", "case"}`, `{2, "compete", "death"}`.
  - Disallow all-zero events; raise informative error prompting user to verify event coding.
  - Ensure at least one primary event and one censored observation for identifiability.
- Compute and return sorted indices by `age_exit`; store for reuse in survival preprocessor.

### 4.3 Survival preprocessor module
- Create `calibrate/survival/mod.rs` housing:
  - `struct SurvivalStats` containing sorted indices, log-age transforms, Jacobian logs, precomputed design matrices for exit/entry (`X_exit`, `X_entry`) and their derivative counterparts (`D_exit`, `D_entry`), event indicators, and sampling weights.
  - `fn build_survival_stats(data: &SurvivalTrainingData, prior_weights: &Array1<f64>, layout: &ModelLayout) -> SurvivalStats` performing:
    1. Sort indices by `age_exit` (for reproducible derivative validation) but retain original ordering for design rows.
    2. Compute log-age transforms `u_exit`, `u_entry`, Jacobian factors `log_j_exit`, `log_j_entry`.
    3. Evaluate baseline and interaction bases at `u_exit` and `u_entry` to produce dense `X_exit`, `X_entry`, `D_exit`, `D_entry` aligned with coefficient ordering in `ModelLayout`.
    4. Cache boolean masks for `d_i` (target events) and `left_truncated_i` for branching within the IRLS loop.
- Provide methods on `SurvivalStats` to evaluate `η` and its derivative from a coefficient vector, returning a struct with `eta_exit`, `eta_entry`, `deta_exit`, `deta_entry`, `H_exit`, `H_entry`, and scratch buffers for scores/Hessian components.

## 5. Design and Penalty Construction
### 5.1 Extending `ModelLayout`
- Add new enum variant `ModelFamily::SurvivalFineGray` stored in `ModelConfig`. Extend `ModelLayout` to track:
  - `baseline_exit_design: Array2<f64>` and `baseline_entry_design: Array2<f64>`.
  - Column ranges for baseline smooth, covariate shifts, tensor interactions.
  - Penalty metadata: `baseline_penalty_id`, `pgs_age_penalty_id`, etc.
- Modify `construction::build_design_and_penalty_matrices` to branch on `ModelFamily`. In survival mode:
  - Build baseline spline using new helper `basis::log_age_spline(age_values, knots, degree, reference_age)`.
  - Append covariate columns (PGS, sex, PCs) as dense columns.
  - If `pgs_age_interaction` enabled, use existing `tensor_product_basis` function with log-age basis and PGS basis; ensure orthogonalization w.r.t. main effects (`interaction_orth_alpha` map in `ModelConfig`).
  - Generate penalties using `difference_penalty` for baseline, `tensor_penalty` for interactions, and `null_shrinkage` as needed.
- Update `ModelLayout::total_coeffs` and penalty assembly to include the new blocks. Provide functions to retrieve baseline column indices for downstream prediction routines.

### 5.2 Null-space handling
- Baseline smooth must have one degree of freedom removed to anchor the cumulative hazard. Implement via `basis::apply_reference_constraint` returning `(design_matrix, z_transform)` to drop the null direction. Store `z_transform` in `ModelConfig::sum_to_zero_constraints` under key `"baseline_age"` to reproduce predictions.
- Ensure PGS and PC columns remain unpenalized and orthogonal to the baseline by subtracting weighted means (existing code already handles centering for parametric columns; confirm and extend if necessary).
- For tensor interactions, reuse existing `interaction_orth_alpha` logic to ensure no leakage into main effects.

## 6. Solver Integration
### 6.1 Link function expansion
- In `calibrate/model.rs`, add:
```rust
pub enum LinkFunction {
    Logit,
    Identity,
    FineGrayRp,
}
```
- Update all `match` statements (e.g., default iteration counts, scale estimation) to handle the new variant. For survival, set `min_iterations = 5` to guard against premature convergence due to complex likelihood surfaces.

### 6.2 Working response computation
- Add module `calibrate/survival/irls.rs` with helper
```rust
pub fn assemble_fine_gray_iteration(
    coeffs: &Array1<f64>,
    stats: &SurvivalStats,
) -> FineGrayIterationState
```
  that evaluates `η`, `∂η/∂t`, `H`, score vector, and Hessian blocks described in Section 3.4, storing them in `FineGrayIterationState` for reuse during REML updates.
- Modify `pirls::update_glm_vectors` to delegate to survival helper when `link == FineGrayRp`. Provide access to `SurvivalStats` via new field in `ModelLayout` or via closure capturing from `estimate::train_model`.
- Ensure P-IRLS workspace caches working vectors sized to survival data. Some arrays (e.g., `delta_eta`) remain applicable without change.

### 6.3 Deviance and gradient tracking
- Add `calculate_survival_deviance(iter_state: &FineGrayIterationState)` to compute `-2 ℓ`. Integrate into `calculate_deviance` by pattern matching on link. Store deviance per iteration for convergence diagnostics and REML objective.
- Update gradient norm computation to use survival-specific residuals `U_i`. Provide fallback if all weights zero (e.g., due to zero events) with descriptive error.

### 6.4 REML/LAML adjustments
- In `estimate.rs`, treat `FineGrayRp` like other non-Gaussian links: LAML objective with log determinant adjustments. Ensure derivative of penalized deviance with respect to smoothing parameters uses the survival Hessian; `pirls.rs` already returns `penalized_hessian` and `edf`, so as long as survival branch populates them correctly no extra work is needed.
- Disable Firth bias reduction in survival mode by erroring if `firth_bias_reduction=true && link==FineGrayRp` (document limitation).

## 7. Prediction Pipeline
### 7.1 Baseline evaluation
- Store in `TrainedModel`:
  - `age_transform: AgeTransform { a_min, delta, reference_age }`.
  - `baseline_knots` and `baseline_degree` for reproducing spline basis.
  - `baseline_z_transform` for constraint reproduction.
  - `baseline_coefficients` subset of overall coefficient vector (extract via column ranges recorded in `ModelLayout`).
  - Optional precomputed `baseline_cumulative_grid` (Array1) evaluated on a fine log-age grid to speed up interpolation.
- Provide helper `fn evaluate_baseline_cumhaz(&self, age: f64) -> f64` performing: transform age to `u`, evaluate constrained basis, compute `H_0(age) = exp(B(u) θ)`, optionally using interpolation if grid available.

### 7.2 Absolute risk computation
- Extend `TrainedModel::predict` signature or add `predict_survival` method that accepts `SurvivalPredictionInputs` and returns `Array1<f64>` of absolute risk over the specified horizon(s).
- Steps per individual:
  1. Clamp `current_age` and `horizon_age` within `[a_min + δ, a_max + margin]`, using hull guard to detect extrapolation.
  2. Evaluate `η(t)` and `∂η/∂t` for a Gauss–Kronrod grid spanning `[current_age, horizon_age]`. Use cached design matrices to avoid reallocations.
  3. Integrate the subdistribution hazard `h(t) = H(t) ⋅ ∂η/∂t` over the grid to obtain `ΔH = ∫ h(t) dt`. When no time-varying effects are present the integrand reduces to `s ⋅ dH_0`, allowing reuse of pre-tabulated baseline increments; keep both branches.
  4. Form cumulative incidences `CIF_target(current) = 1 - exp(-H(current))` and `CIF_target(horizon) = 1 - exp(-(H(current) + ΔH))`.
  5. Acquire competing subdistribution cumulative incidence at `current_age` by evaluating the companion model (or Kaplan–Meier proxy for MVP). Apply renormalisation `conditional = (CIF_target(horizon) - CIF_target(current)) / max(1e-12, 1 - CIF_target(current) - CIF_competing(current))`.
  6. Return the conditional risk, optionally along with `linear_predictor` and `std_error` if caller requests. For the latter propagate uncertainty using the stored Hessian inverse and quadrature weights (delta method on `ΔH`).
- Provide convenience for multiple horizons: accept `horizon_grid: &[f64]` and vectorize evaluation by reusing baseline grid and covariate shifts.

### 7.3 Integration with calibrator
- When calibrator enabled, compute diagnostics in survival context:
  - Baseline absolute risk (`CIF`),
  - Approximate variance of `η` using diagonal of inverse penalized Hessian and delta method to get variance of `CIF`,
  - Hull distance in extended feature space (PGS, PCs, age).
- Build calibrator design using existing routines but mark link as `Logit` to keep outputs in (0,1). Provide new calibrator feature generator `compute_survival_alo_features` mirroring `compute_alo_features` but using survival diagnostics.
- Apply calibrator at prediction time by mapping base CIF to logits, adding calibrator correction, and mapping back via logistic transform.

## 8. CLI and Configuration
- Add CLI enum `ModelKind { Binary, Continuous, Survival }` in `cli` crate. Update argument parser to accept `--survival` or `--model-kind survival`.
- In training command:
  - Route to survival loader and builder when `ModelKind::Survival`.
  - Expose additional flags: `--baseline-knots`, `--baseline-degree`, `--pgs-age-interaction`, `--no-competing` (if user wants to drop competing events), `--no-calibrator` (already supported), `--horizon` for evaluation summary.
- In inference command:
  - Require input columns `current_age` and `horizon_age` (or `years_ahead`), verifying numeric and finite.
  - Output CSV with columns `sample_id`, `current_age`, `horizon_age`, `absolute_risk`, `calibrated_absolute_risk` (if calibrator active), `std_error` (optional).

## 9. Implementation Sequence & Milestones
1. **Design groundwork**
   - Introduce `ModelFamily::SurvivalFineGray` and `LinkFunction::FineGrayRp` scaffolding without behavior. Update serialization/tests to ensure round-trip.
2. **Data ingestion**
   - Implement survival data loader, validations, and unit tests covering missing columns, invalid codes, degenerate datasets.
3. **Survival statistics module**
   - Build `SurvivalStats` with precomputed design/derivative matrices and Jacobian logs. Validate by finite-difference checking scores on toy datasets.
4. **Baseline spline support**
   - Extend `basis.rs` with log-age helper and reference constraints. Add tests verifying monotonic cumulative hazard when coefficients increase.
5. **Model layout adjustments**
   - Modify `construction.rs` to assemble survival design/penalties. Ensure compatibility with existing logistic/Gaussian cases through regression tests.
6. **P-IRLS integration**
   - Implement `update_fine_gray_vectors`, hook into `pirls.rs`, and verify convergence on toy survival dataset (simulate via Python/R). Inspect deviance trajectory for monotonic decrease.
7. **REML coupling**
   - Enable smoothing parameter optimization; ensure gradients finite using numerical checks. Compare λ estimates to R `rstpm2` on matched dataset.
8. **Prediction API**
   - Add survival scoring path, baseline evaluation, CIF computation, and hull extension. Provide unit tests for monotonicity in age and horizon.
9. **Calibrator adjustments**
   - Extend feature extraction and calibrator design for survival. Validate that calibrator training converges and reduces Brier score on held-out simulation.
10. **CLI integration & documentation**
    - Wire CLI options, update README, and provide usage examples.
11. **End-to-end validation**
    - Full training/inference pipeline on synthetic dataset with known CIF; compare predictions vs. R reference to <1e-3 absolute difference.
    - Stress tests: heavy censoring, all competing events, extremely late entry ages.

## 10. Testing Strategy
- **Unit tests**
  - `data::load_survival_training_data`: invalid event labels, negative ages, exit < entry, no events.
- `survival::build_survival_stats`: confirm derivative matrices match analytical derivatives on synthetic spline (≤5 subjects).
- `survival::assemble_fine_gray_iteration`: compare gradient/Hessian to finite differences.
  - `basis::log_age_spline`: ensure evaluation at reference age yields zero contribution after constraint.
- **Property tests**
  - Simulate data from known RP model (use Python or Rust to generate). Fit survival model and assert estimated coefficients within tolerance (e.g., |β_est - β_true| < 0.05) and CIF predictions within ±0.01.
  - Randomly permute input order to ensure invariance (design matrices reference original ordering; stats builder must be stable).
- **Integration tests**
  - CLI-driven training/inference with sample TSVs; confirm output schema and calibration.
  - Compare to R `rstpm2`/`cmprsk` by exporting training data and verifying log-likelihood and CIF at several ages.
- **Performance benchmarks**
- Benchmark training on synthetic dataset with 100k subjects to ensure runtime scales sub-quadratically and memory remains within reasonable bounds. Profile derivative assembly to identify bottlenecks.
- **Calibrator validation**
  - Evaluate Brier score and calibration plots before/after calibrator on validation split. Confirm calibrator does not violate monotonicity with age by checking CIF increases with horizon after calibration.

## 11. Performance and Numerical Stability Considerations
- Precompute and reuse `exp(η)` and derivative vectors within each IRLS iteration to avoid repeated exponentiation and basis multiplies.
- Implement fused multiply–add loops for rank-one derivative corrections to reduce floating-point error when events are numerous.
- Guard against negative or extremely small `∂η/∂t` by clipping at `1e-12` before logging; report descriptive error if clipping activates for >1% of events (indicative of spline undersmoothing).
- Apply step-halving when deviance increases or when cumulative hazard becomes non-monotonic (detected via negative increments). Reuse existing step-halving infrastructure in `pirls.rs`.
- Extend hull to include age axis: build 3D convex hull over `(PGS, PC1..PCk, age_exit)` or compute axis-aligned guard with signed distance function to detect extrapolation. Reuse `hull::PeeledHull` by augmenting input matrix with age column.

## 12. Documentation & Communication
- Update `calibrate/README.md` with a dedicated survival section covering mathematical formulation, data requirements, CLI usage, and prediction semantics.
- Add Rustdoc comments to new structs (`SurvivalTrainingData`, `SurvivalStats`, `AgeTransform`, etc.) clarifying the Fine–Gray formulation and referencing key equations.
- Provide example workflow in `examples/` (e.g., `examples/survival/README.md`) demonstrating training on synthetic data and scoring multiple horizons.
- Coordinate with maintainers to ensure release notes highlight survival support and note limitations (single competing event type, static covariates).

## 13. Future Extensions (Post-MVP)
- Multi-state extension supporting >1 competing risk by fitting separate subdistribution hazards with shared baseline or cause-specific baselines.
- Support for time-varying covariates by interval-splitting each subject and reusing the counting-process representation.
- Stratified baselines (e.g., by sex) by adding additional baseline smooth blocks with shared penalties.
- Explore alternative link functions (e.g., log cumulative odds) for different risk interpretations.
- GPU-accelerated derivative/Hessian assembly if profiling reveals bottlenecks.

# Royston–Parmar Fine–Gray Survival Integration Plan

## 1. Context, Goals, and High-Level Design Principles
- **Objective.** Extend the penalized GAM infrastructure in `calibrate/` to fit Royston–Parmar flexible parametric survival models with Fine–Gray subdistribution hazards, using attained age as analysis time and supporting absolute risk predictions between a current age and user-specified horizons.
- **Constraints.**
  - Reuse the existing P-IRLS + REML machinery in `calibrate/pirls.rs` and `calibrate/estimate.rs` to optimize smoothing parameters jointly with regression coefficients.
  - Maintain backwards compatibility for existing logistic and Gaussian workflows (`LinkFunction::{Logit, Identity}`) while introducing survival as an additional analysis mode.
  - Keep serialization/deserialization stable via `calibrate/model.rs::TrainedModel`; new fields must be optional/defaulted for previously trained models.
  - Integrate with the existing post-hoc calibrator (`calibrate/calibrator.rs`) so survival risk outputs can be optionally calibrated in the same pipeline.
- **Design heuristics.**
  - Introduce a **family abstraction** that decouples the P-IRLS core from distribution-specific math. Logistic and Gaussian become implementations; survival is a third implementation.
  - Treat survival as a **new model variant** with its own response schema, design blocks, and prediction interface, rather than overloading `LinkFunction` semantics.
  - Ensure the baseline log cumulative hazard spline is handled with the same penalty infrastructure (difference penalties, null/range splits) for consistency.
  - Preserve numerical stability by carefully evaluating spline basis and derivatives on the log-age scale and by caching baseline evaluations for scoring.

## 2. Mathematical Specification
### 2.1 Parameterization
- Let attained age be the time scale. For subject *i*: entry age `a_i`, exit age `b_i`, event indicator `d_i ∈ {0,1}`, competing-risk indicator `c_i ∈ {0,1}` (1 if competing event occurred at `b_i`).
- Define log-time variable `u = log(b_i)`; for left-truncated contributions use both `log(a_i)` and `log(b_i)`.
- Baseline cumulative subdistribution hazard: `log H_0(u) = Σ_j γ_j B_j(u)` where `{B_j}` is a B-spline basis on the log-age scale. Penalty is applied to γ via difference matrix as in existing smooth terms.
- Linear predictor: `η_i(u) = log H_0(u) + x_i^T β + z_i(u)^T θ`, where
  - `x_i` includes baseline covariates (PGS main effect, PCs, optional sex) with proportional hazard interpretation.
  - `z_i(u)` can encode time-varying effects such as PGS×age (basis for interaction). Begin with optional PGS×log-age smooth.
- Cumulative subdistribution hazard: `H_i(u) = exp(η_i(u))`.
- Subdistribution hazard: `h_i(u) = H_i(u) * (dη_i(u)/du) / exp(u)` where `dη_i(u)/du = Σ_j γ_j B'_j(u) + (∂z_i(u)/∂u)^T θ`. Time-varying covariate bases must therefore supply their derivatives with respect to `u = log(age)`; when no time-varying terms are used the second summand vanishes.

### 2.2 Likelihood Contributions
- Adopt a Fine–Gray style **full likelihood** with the Royston–Parmar spline supplying the parametric subdistribution hazard. Each individual contributes the log-likelihood of an interval-censored subdistribution time-to-event model rather than just a risk-set ratio.
- For subject `i` with entry age `a_i` and exit age `b_i`, evaluate the cumulative subdistribution hazard difference `ΔH_i = H_i(b_i) - H_i(a_i)` (with `H_i(a_i) = 0` when no left truncation). The weighted log-likelihood contribution is

  `ℓ_i = w_i [d_i log h_i(b_i) - ΔH_i]`,

  where `d_i` is 1 only when the event of interest occurs at `b_i`. Competing events correspond to `d_i = 0` but still subtract their cumulative hazard through `ΔH_i`, preserving the Fine–Gray subdistribution semantics while keeping a true likelihood objective.
- The derivative term `(∂η_i/∂t)(t)` remains essential because `h_i(t) = H_i(t) (∂η_i/∂t)(t) / exp(t)`; cache both `H_i` and the derivative so that gradient/Hessian code reuses them across REML iterations.
- Left truncation is handled directly via `H_i(a_i)`: subtract the cumulative hazard evaluated at entry so the likelihood conditions on survival up to `a_i`.
- Multiply all contributions by sample weights before accumulating the score or Hessian.

### 2.3 Gradients / IRLS quantities
- Define the augmented design row `\tilde{X}_i^{exit}(t) = X_i^{exit} + D_i^{exit} / (∂η_i/∂t)(t)` so that differentiation of `log h_i(t)` collects both baseline and time-varying pieces.
- Express the score per subject: `U_i = w_i [d_i \tilde{X}_i^{exit}(b_i) - ΔH_i X_i^{integral}]`, where `X_i^{integral}` combines the entry/exit design rows used to evaluate the cumulative hazard difference. Cache these rows alongside `ΔH_i` so PIRLS iterations can reuse them without regenerating spline evaluations.
- Assemble the negative Hessian as `H = Σ_i w_i [d_i \tilde{X}_i^{exit ⊤} \tilde{X}_i^{exit} + ΔH_i X_i^{integral ⊤} X_i^{integral}]` and add penalty blocks. This retains the off-diagonal curvature needed for REML and mirrors the structure expected by the existing linear algebra routines.
- Left truncation already appears through `ΔH_i = H_i(b_i) - H_i(a_i)`; ensure monotonicity of `H_i` to keep `ΔH_i ≥ 0` and avoid negative working weights.
- The working response continues to use `z = η + H^{-1} U` inside the penalized Newton update, leveraging the same solver infrastructure as other families.

### 2.4 Absolute Risk Predictions
- Absolute risk between current age `t0` and horizon `t1` requires conditioning on being event-free (for all causes) at `t0`. Evaluate `CIF_target(t) = 1 - exp(-H_i(t))` and retain the Fine–Gray-derived competing incidence `CIF_competing(t)` from the censoring weights.
- Conditional probability of the target event in `(t0, t1]` is `(CIF_target(t1) - CIF_target(t0)) / max(1e-12, 1 - CIF_target(t0) - CIF_competing(t0))`.
- Provide helper routines to evaluate both the cumulative hazards and the derivative-based hazard when time-varying effects are present; cache basis evaluations at requested ages for efficiency.

## 3. Data & Schema Extensions (`calibrate/data.rs`)
### 3.1 Input Columns
- Extend training TSV schema to include:
  - `age_start`: optional (defaults to baseline age if not provided). Required when left truncation occurs.
  - `age_stop`: required (event/censor age).
  - `event`: 1 if event of interest, 0 otherwise.
  - `competing`: 1 if competing event occurred at `age_stop`, 0 otherwise.
- Validate mutual exclusivity: `event` and `competing` cannot both be 1.
- Ensure `age_stop > age_start` and both are finite; allow inclusive start for immediate entry.
- Optionally support `left_trunc = false` path where `age_start` column missing → default to minimum age in dataset minus small ε for stability.
- For prediction scoring API, add `current_age` column optional (if not provided, inference caller passes age per query) and no event/censor columns.

### 3.2 Data Structures
- Introduce `enum AnalysisMode` in `calibrate/data.rs` or new module to indicate standard vs survival dataset.
- Add `SurvivalTrainingData` struct with fields:
  ```rust
  pub struct SurvivalTrainingData {
      pub age_start: Array1<f64>,
      pub age_stop: Array1<f64>,
      pub event: Array1<f64>, // binary 0/1
      pub competing: Array1<f64>,
      pub p: Array1<f64>,
      pub sex: Array1<f64>,
      pub pcs: Array2<f64>,
      pub weights: Array1<f64>,
  }
  ```
- Modify `load_training_data` to detect presence of survival columns and return `TrainingDataVariant::Standard(TrainingData)` or `TrainingDataVariant::Survival(SurvivalTrainingData)`.
- Update CLI (`cli/main.rs::train`) to branch on analysis mode. Survival-specific CLI options (baseline knot count, etc.) should become flags.

## 4. Model Configuration & Serialization (`calibrate/model.rs`)
### 4.1 Extended Config
- Introduce `enum ModelFamily { Standard(LinkFunction), Survival(SurvivalSpec) }` within `ModelConfig`.
- `SurvivalSpec` should contain:
  - `baseline_basis: BasisConfig` (knots/degree on log-age scale).
  - `time_range: (f64, f64)` storing training age range for knot placement.
  - `pgs_time_interaction: Option<BasisConfig>` and metadata for optional time-varying PGS effect.
  - `penalty_order_baseline: usize` for difference penalty on baseline spline.
  - `hazard_offset: Option<f64>` reserved for future (set 0 now).
  - `fine_gray: bool` (default true) to allow toggling to cause-specific if needed.
- Maintain `link_function` for backward compatibility but mark deprecated path when `family` is Standard; when `family` is Survival, ignore `link_function` and set to sentinel (maybe keep but set to Identity?). Add serde attributes to keep old models loading (`#[serde(default)]`).
- Update `TrainedModel::predict*` to branch: standard predictions use existing path; survival predictions delegate to new functions (see Section 8).
- Store baseline spline coefficients separately in `MappedCoefficients` or extend structure with `Option<SurvivalCoefficients>` containing:
  ```rust
  pub struct SurvivalCoefficients {
      pub baseline: Vec<f64>,
      pub baseline_knot_vector: Array1<f64>,
      pub baseline_penalty_transform: Array2<f64>,
      pub pgs_main: Vec<f64>,
      pub pcs: HashMap<String, Vec<f64>>,
      pub pgs_time: Option<Vec<f64>>,
  }
  ```
  Keep logistic fields for compatibility; restructure mapping helpers to populate whichever variant is active.

### 4.2 Layout Consistency Checks
- Extend `TrainedModel::rebuild_layout_from_config` to understand survival layout: baseline block first, PGS/PC blocks next, optional interaction block. Ensure `ModelLayout` is enhanced (Section 5) to represent survival terms.
- When loading older models without survival info, maintain existing behavior. When survival model missing calibrator/hull, allow optional.

## 5. Design & Penalty Construction (`calibrate/construction.rs`)
### 5.1 New Layout Support
- Introduce new `TermKind::BaselineSpline`, `TermKind::SurvivalCovariate`, `TermKind::TimeInteraction` in `ModelLayout` to describe survival-specific column spans.
- Extend `build_design_and_penalty_matrices` to accept `TrainingDataVariant` and `ModelConfig::family`:
  - For survival: evaluate baseline B-spline on `log(age_stop)` and `log(age_start)`; also compute derivative basis `B'(u)` for log-hazard derivative needed later (store for reuse by `pirls` in `ModelLayout` or extra arrays).
  - Apply sum-to-zero constraint for baseline? For log cumulative hazard, identifiability is provided by intercept; need to fix intercept by centering or by dropping final coefficient. Strategy: enforce mean-zero on baseline to avoid confounding with intercept; store transformation matrices (Z) in config.
  - Build penalty for baseline via `create_difference_penalty_matrix` using `penalty_order_baseline` and convert to block in `PenaltyMatrix`.
  - For PGS main effect and PCs: reuse existing basis + penalty infrastructure (range/null transforms). For survival, these effects multiply hazard; treat them as linear terms (no additional smoothing) or re-use smoothing as before. Most likely keep PGS/PC smooths same as logistic but interpret as log-H scale; ensure penalty structures identical.
  - Optional PGS×age interaction: build tensor-product between PGS range basis and baseline log-age basis; use anisotropic penalty to allow separate smoothing on time dimension vs PGS dimension. Represent as new penalty block in layout.
- Provide derivative basis matrix `baseline_deriv` to PIRLS (maybe store inside `ModelLayout` or `SurvivalWorkspace` struct). This matrix is evaluation of `∂B/∂u` on log-age. Need ability to evaluate at both `age_start` and `age_stop`; store both arrays.

### 5.2 Balanced Penalty Roots / Constraints
- Extend `compute_penalty_square_roots` and `create_balanced_penalty_root` to handle survival baseline penalty. No conceptual change; just ensure new block included.
- Ensure sum-to-zero constraint logic doesn’t drop baseline intercept inadvertently; the baseline needs intercept to capture cumulative hazard scale. Implement `apply_sum_to_zero_constraint` with caution: probably maintain intercept but impose roughness penalty only on higher-order components (like difference penalty). Document reasoning inside plan.

## 6. PIRLS & REML Adaptation (`calibrate/pirls.rs`, `calibrate/estimate.rs`)
### 6.1 Family Abstraction
- Introduce trait in `calibrate/pirls.rs`:
  ```rust
  pub trait PirlsFamily {
      fn initialize_eta(
          y: ArrayView1<f64>,
          offset: ArrayView1<f64>,
      ) -> Array1<f64>;
      fn mu_from_eta(eta: ArrayView1<f64>) -> Array1<f64>;
      fn compute_working_quantities(
          eta: ArrayView1<f64>,
          y: ArrayView1<f64>,
          offset: ArrayView1<f64>,
          weights: ArrayView1<f64>,
          aux: &FamilyWorkspace,
      ) -> (Array1<f64>, Array1<f64>, f64, Array1<f64>);
      fn deviance(
          y: ArrayView1<f64>,
          mu: ArrayView1<f64>,
          offset: ArrayView1<f64>,
      ) -> f64;
  }
  ```
- `FamilyWorkspace` can hold scratch buffers per family. For survival, needs `baseline_stop`, `baseline_start`, derivative arrays, hazard caches.
- Refactor logistic and identity flows to implement trait; existing code for weights/z/residuals migrates into `BinomialFamily` and `GaussianFamily` modules.

### 6.2 Survival Family Implementation
- Implement `RoystonParmarFineGrayFamily` with:
  - Inputs: `ModelLayout` should provide arrays `baseline_stop_design`, `baseline_start_design`, `baseline_stop_deriv`, `baseline_start_deriv` to compute `H(b)` and `H(a)` for each subject using current β.
  - Additional data: `event`, `competing`, `weights` from `SurvivalTrainingData`.
  - `initialize_eta`: start from log cumulative hazard based on Nelson-Aalen style estimator or start with `η = log(-log(1 - empirical CIF))`. Simpler: start from baseline-only fit (γ initial) by solving for intercept using events/weights or using log of (events / risk). Provide fallback constant.
  - `compute_working_quantities`: For each observation compute `H_stop = exp(η_stop)` and `H_start = exp(η_start)` using matrix multiplication `X_stop dot β`, `X_start dot β`. Keep both because design matrix is same for stop and start but offset differs (left truncation subtract). Evaluate derivatives `s'(u)` using derivative design times coefficients for baseline block only; combine to compute hazard `h_stop`. Provide stable evaluation by clamping `η` to [-700, 700].
  - Compute gradient `g_i` and weights `W_i` per Section 2.3. Return `z = eta + g/W`, `sqrt_w = sqrt(weights * sample_weight)`. Provide `mu` analog as `H_stop` for deviance computation.
  - Provide log-likelihood contributions for deviance (deviance = -2 * weighted log-likelihood). Use aggregated sums to ensure numerical stability (Kahan summation).
  - Provide Firth-like adjustments? Not required initially but keep path for potential future bias reduction.

### 6.3 REML Outer Loop Changes (`calibrate/estimate.rs`)
- When `ModelFamily::Survival`, `internal::RemlState::new` should accept survival-specific data (two design matrices for stop/start?). Options:
  - Build single combined design `X` on stop times; incorporate left truncation via offset vector stored separately; treat contributions inside family functions rather than in `X`. Keep `X` as the standard design for stop times. Provide extra arrays in `RemlState` for start times.
- Update REML log-likelihood computation to use survival deviance and penalty term; derivative/hessian computations rely on `PirlsResult` outputs unaffected by family abstraction.
- Ensure Gaussian scale profiling is skipped in survival branch.
- Update gradient check/test harness to handle `ModelFamily::Survival` by verifying finite values.

## 7. Training Pipeline Integration (`calibrate/estimate.rs::train_model`)
- Accept `TrainingDataVariant`. For survival branch:
  1. Build design/penalty via new path.
  2. Initialize `RemlState` with survival-specific data.
  3. Run REML optimization as usual.
  4. Map coefficients into `SurvivalCoefficients` using new layout mapping helper.
  5. Skip PHC hull (not meaningful in survival) or redesign hull to operate on `(PGS, PCs)` ignoring age; consider storing for calibrator features (PGS, PCs) only if beneficial.
  6. Compute penalized Hessian and store for inference (used for variance of η if needed). Provide optional ability to store baseline evaluation grid for faster scoring.
- Update logging messages to mention survival mode and dataset size.

## 8. Prediction & Scoring API (`calibrate/model.rs`)
### 8.1 Survival Prediction Methods
- Add methods:
  ```rust
  impl TrainedModel {
      pub fn predict_survival_eta(
          &self,
          p: ArrayView1<f64>,
          sex: ArrayView1<f64>,
          pcs: ArrayView2<f64>,
          ages: ArrayView1<f64>,
      ) -> Result<Array1<f64>, ModelError>;

      pub fn absolute_risk(
          &self,
          p: ArrayView1<f64>,
          sex: ArrayView1<f64>,
          pcs: ArrayView2<f64>,
          current_age: f64,
          horizon_age: f64,
      ) -> Result<f64, ModelError>;
  }
  ```
- Implement helper to evaluate baseline spline and derivative at arbitrary ages using saved knots and coefficient transforms.
- `predict_detailed` should branch: in survival mode, return tuple `(eta_stop, risk, Array1::zeros, None)` where `risk` is e.g. `1 - exp(-ΔH)` over a default horizon; document difference. For compatibility, maybe keep method returning CIF at `age_stop` for training data predictions; provide new method for horizon-specific risk.

### 8.2 Batch Scoring API
- Add struct in new module `calibrate/survival/predict.rs` exposing function to compute vectorized absolute risks for arrays of horizons. Accept arrays of `current_age`, `horizon`, `pgs`, `pcs`. Provide ability to reuse basis evaluation caching for repeated queries.
- Update CLI `infer` to support `--current-age` and `--horizon` flags (or TSV columns) for survival models. Output should include baseline CIF and optionally hazard.

### 8.3 Calibration Integration
- For survival, calibrator should operate on absolute risk predictions (bounded [0,1]). Reuse logistic calibrator by applying logit transform of risk as `pred`. Provide SEs via delta method using Hessian (if stored) or fallback to zero. Update calibrator input building to handle survival-specific features (PGS, age, risk) and to optionally include `current_age` for hull distance.
- Ensure `set_calibrator_enabled` flag applies uniformly.

## 9. CLI & User-Facing Changes (`cli/main.rs`)
- Add CLI flags to specify survival mode explicitly (`--analysis survival`) or auto-detect based on columns.
- Introduce survival-specific hyperparameters: `--baseline-knots`, `--baseline-degree`, `--baseline-penalty-order`, `--pgs-age-knots`, etc.
- Update `train` command to build `SurvivalSpec` from CLI options. When survival, skip auto-detecting `LinkFunction` and set `ModelFamily::Survival`.
- Update `infer` command to support survival models: accept `--current-age`, `--horizon`, or TSV with `current_age` column. Print risk estimates with optional calibration.
- Provide helpful error messages if user requests survival scoring from non-survival model or vice versa.

## 10. Testing & Validation Strategy
### 10.1 Unit Tests
- **Basis derivatives:** Add tests in `calibrate/basis.rs` verifying new derivative evaluator against finite differences on known spline (log-age grid).
- **Family math:** Unit-test `RoystonParmarFineGrayFamily` weight/gradient computation on synthetic data with known hazard to ensure gradient/Hessian match numerical differentiation.
- **Prediction identity:** Confirm `absolute_risk` integrates to zero horizon and matches expected CIF for simple exponential baseline (use known parameters).
- **Serialization:** Round-trip survival `TrainedModel` through TOML to ensure optional fields default correctly.

### 10.2 Integration Tests
- Add tests under `calibrate/tests/` comparing fitted survival model against reference (R `rstpm2` or `flexsurv`). Use small dataset with known CIF to verify parameter recovery within tolerance.
- Validate Fine–Gray competing risks by simulating dataset where competing events dominate; confirm predictions match numerical integration of CIF.
- Regression tests for existing logistic/Gaussian flows to ensure no changes in outputs; reuse existing harness (e.g., `calibrate/tests/run_calibrate.py`).

### 10.3 Performance & Numerical Diagnostics
- Benchmark training runtime vs dataset size; ensure derivative evaluation and hazard computation vectorized and cache-friendly.
- Add logging of min/max `η`, `H`, weights to detect overflow; clamp values to avoid NaNs.
- Consider precomputing baseline basis for unique ages to reduce repeated evaluation.

## 11. Performance Considerations
- Baseline B-spline evaluation on log-age: use existing caching infrastructure in `basis.rs` keyed by knot vector + derivative order (extend cache key to include derivative flag).
- For left truncation, maintain separate design matrices for start and stop; share coefficient vector to avoid duplication. Represent as `Array2` views inside `RemlState` to prevent repeated allocation.
- Use `ndarray::Zip` to compute gradient/weights in survival family to maximize SIMD and minimize temporaries.
- For scoring multiple horizons, precompute basis per unique horizon (store in `HashMap<f64, Arc<Array1<f64>>>`) or use vector evaluation to avoid repeated B-spline evaluation per individual.

## 12. Documentation & Examples
- Update `README.md` and `calibrate/README.md` to describe survival mode, required columns, CLI usage, and prediction semantics.
- Provide worked example in `examples/` showing training survival model on synthetic dataset, scoring absolute risk, and comparing to standard logistic model.
- Document mathematical derivations (Section 2) in new markdown under `docs/` or extend plan after implementation.

## 13. Validation Checkpoints & Rollout Sequence
1. **Refactor PIRLS family abstraction** (ensure existing models still fit).
2. **Implement basis derivative support** with tests.
3. **Add survival data schema and config structures**; ensure CLI can detect mode.
4. **Implement survival design construction** and extend `ModelLayout`.
5. **Implement survival family** and integrate with REML.
6. **Fit basic survival model without competing risks**; validate vs exponential baseline.
7. **Add Fine–Gray contributions**; validate on simulated data.
8. **Wire prediction API** for absolute risk and verify with training-time predictions.
9. **Integrate calibrator** and update CLI outputs.
10. **Expand tests/documentation** and run full CI/regression suite.

## 14. Future Extensions Considered
- Time-varying covariates: design can be extended by splitting intervals; plan preserves ability by structuring start/stop arrays and by not assuming single record per person.
- Stratified baselines: extend `SurvivalSpec` with optional factor variable that replicates baseline spline per stratum.
- Alternative survival families (cause-specific hazards, accelerated failure time) can implement additional `PirlsFamily` variants without disturbing infrastructure.

