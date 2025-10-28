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
- For event-of-interest at `b_i` (Fine–Gray): log-likelihood term `log h_i(u_i) + log S_i(u_i^-)`, with `S_i(u) = exp(-H_i(u))` being the survival function for the subdistribution hazard.
- For censored or competing events: contribution `log S_i(u_i)`.
- Left truncation at `a_i`: subtract `log S_i(v_i)` evaluated at `v_i = log(a_i)` from each individual's log-likelihood (i.e., condition on survival up to entry age).
- Competing risks (Fine–Gray) keep individuals with competing events in the risk set; their contributions use the same `S_i(u_i)` term but without `log h_i`. A weight update step must ensure risk set weights reflect subdistribution hazard definition (weights stay 1 after competing event until evaluation time because hazard integrates over all individuals).
- Weighted log-likelihood includes sample weights (`TrainingData::weights`).

### 2.3 Gradients / IRLS quantities
- Need gradient and Hessian of log-likelihood w.r.t. `η`. For each observation, derivative of contribution w.r.t. `η`:
  - Event: `∂ℓ/∂η = 1 - H_i(b_i)` (because `log h = η + log[s'(u)/exp(u)]` → derivative w.r.t. η is 1) minus derivative from survival term `H_i(b_i)`.
  - Censor/competing: `∂ℓ/∂η = -H_i(b_i)`.
- Left truncation subtracts `-H_i(a_i)` contributions, so its score contribution is `+H_i(a_i)` because derivative of `-log S(a_i)` equals `+H_i(a_i)`.
- Second derivatives (for IRLS weights):
  - Event: `∂²ℓ/∂η² = -H_i(b_i)`.
  - Censor/competing: `∂²ℓ/∂η² = -H_i(b_i)`.
- Left truncation adds `+H_i(a_i)`.
- Implement P-IRLS by treating `W_i = -∂²ℓ/∂η²` and `z_i = η_i - g_i/∂g/∂η`, where `g_i = ∂ℓ/∂η`. Because link is identity in η-space (η is natural parameter), the working response simplifies to `z_i = η_i + g_i/W_i`.
- Need to ensure weights remain positive; `W_i = H_i(b_i) - H_i(a_i)` for truncated observations, which stays non-negative because `H_i` is monotone increasing in time. Guard against overflow by clamping exponentials.
- The derivative of log hazard `log[s'(u)/exp(u)]` influences the gradient; precompute `s'(u)` via derivative basis evaluation.

### 2.4 Absolute Risk Predictions
- Absolute risk between current age `t0` and horizon `t1` (on age scale) for covariates x:
  - Evaluate `H_i(u)` at both `u0 = log(t0)` and `u1 = log(t1)`.
  - Subdistribution cumulative incidence function (CIF): `CIF(t1 | t0) = 1 - exp(-(H_i(u1) - H_i(u0)))`.
  - For survival probability `S(t1 | t0) = exp(-(H_i(u1) - H_i(u0)))`.
- Need to precompute `H_0(u)` spline basis for any requested age grid; store knots and basis degree in model config.
- For predictions over multiple horizons, evaluate basis matrix for vector of ages to avoid recomputation.

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

