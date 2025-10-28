# Royston–Parmar Attained-Age Survival Integration Plan

## 1. Vision and Constraints
- **Goal**: Add a Royston–Parmar flexible parametric survival engine (including Fine–Gray competing risks) that reuses the penalized GAM infrastructure (`calibrate::*`) to fit age-as-time hazard models and expose calibrated absolute risk predictions.
- **Branch Context**: Target branch `feature/royston-parmar-survival` already contains GAM tooling optimized around binary/continuous outcomes with B-spline bases, tensor-product penalties, and REML-driven smoothing selection.
- **Non-negotiables**:
  - Retain REML/P-IRLS as the inner/outer optimization with minimal duplication.
  - Preserve calibration pipeline (optionally disable via `set_calibrator_enabled`).
  - Avoid forcing per-horizon refits; predictions must integrate hazard over arbitrary future intervals using one fitted model.
  - Support left truncation (delayed entry), right censoring, and cause-specific competing risks within the Fine–Gray subdistribution framework.
  - Maintain compatibility with existing CLI/config APIs (no breaking changes for non-survival models unless explicitly version-gated).

## 2. Mathematical Blueprint
### 2.1 Baseline Structure
- Use attained age \(a\) as analysis time. Let \(t = \log(a)\) for spline modeling stability.
- Baseline cumulative hazard on original time scale: \(H_0(a) = \exp\{ s(t) \}\), where \(s(t)\) is a penalized B-spline.
- Baseline hazard: \(h_0(a) = H'_0(a) = H_0(a) \cdot s'(t) / a\) (needed for Fine–Gray likelihood and optional diagnostics).
- Model linear predictor: \( \eta_i(a) = s(\log a) + x_i^\top \beta + z_i(a)^\top \gamma \), where
  - `x_i`: time-invariant covariates (PGS, PCs, sex, optional offsets).
  - `z_i(a)`: time-varying effects (PGS×age smooth, optional PC×age). Start with PGS×age (tensor-product of PGS basis with age basis sharing baseline knots to ease penalty reuse).
- Hazard for subject `i`: \( h_i(a) = h_0(a) \, \exp( x_i^\top \beta + z_i(a)^\top \gamma ) \).

### 2.2 Likelihoods
- **Primary endpoint**: Fine–Gray subdistribution hazard for competing risks.
  - Use counting-process notation with cumulative incidence \(F_k(a)\) for cause `k`.
  - Subdistribution hazard: \( \tilde{h}_k(a|X) = h_0(a) \exp( X\beta ) \).
  - Construct pseudo-observations using Fine–Gray weighting (see Gray (1999)). Implementation should follow the data augmentation strategy from Andersen et al., aligning with the Poisson trick enabling P-IRLS.
- **IRLS Formulation**:
  - Represent survival log-likelihood via Poisson GLM on disaggregated intervals (Andersen-Gill). For Royston–Parmar, we can avoid data splitting by using analytical gradient/Hessian for log cumulative hazard basis. Use weights derived from risk set contributions at unique event ages.
  - Proposed approach: derive log-likelihood contributions
    \[
    \ell = \sum_i \delta_i \{ \eta_i(a_i) + \log w_i \} - w_i \exp(\eta_i(a_i))
    \]
    where `w_i` encodes cumulative hazard exposure (requires computing interval integrals). For Fine–Gray, modify `w_i` using subdistribution weights.
  - Implement as custom `LinkFunction::RoystonParmar` that supplies `WorkingResponse` & `FisherWeights` to PIRLS given current `eta`.
  - Because existing PIRLS expects canonical links with known variance, extend PIRLS to accept trait-specific closures (see §5.2).

### 2.3 Absolute Risk Prediction
- After fitting, compute cumulative incidence for horizon \(h\) at current age \(a_0\):
  \[
  \Pr(T \le a_0+h, \text{cause}=k \mid T > a_0) = 1 - \exp\{- [H_k(a_0+h) - H_k(a_0)] \}
  \]
  - With subdistribution hazard, ensure conversion from subdistribution cumulative hazard to CIF is consistent with Fine–Gray (requires recasting to cause-specific CIF if necessary).
- Provide scoring API: `predict_absolute_risk(current_age, horizon, pgs, pcs, sex, optional offsets)` returning CIF along with baseline survival, cause-specific hazard, and optionally standard errors via delta method (if feasible).

## 3. Data Layer Extensions (`calibrate::data`)
### 3.1 Training Schema
- Augment TSV schema to include survival columns:
  - `age_entry`: left-truncation age (continuous, same units as horizon input).
  - `age_exit`: observed age at event or censoring.
  - `event_primary`: {0,1} indicator for target event.
  - `event_competing`: {0,1} indicator for competing event (death). Additional columns for multiple competing risks should be specifiable (see §3.4).
  - `calendar_time` optional? Not required for attained-age models; ignore for now.
- All ages must be >0 and `age_exit >= age_entry`. Validate monotonicity, finite values, minimal spread.
- Provide `TrainingDataSurvival` struct separate from existing `TrainingData` to avoid breaking binary regression users.
  ```rust
  pub struct TrainingDataSurvival {
      pub age_entry: Array1<f64>,
      pub age_exit: Array1<f64>,
      pub event_primary: Array1<f64>,
      pub event_competing: Array1<f64>,
      pub y_pseudo: Array1<f64>,           // filled later by Fine–Gray augmentation
      pub weight_pseudo: Array1<f64>,      // event weights for PIRLS
      pub offsets: Array1<f64>,            // log exposure offsets if Poisson trick used
      pub baseline_knots: Array1<f64>,     // saved for ModelConfig
      pub p: Array1<f64>,
      pub pcs: Array2<f64>,
      pub sex: Array1<f64>,
      pub weights: Array1<f64>,
  }
  ```
- Extend `load_training_data` with mode switch `TrainingKind::Standard | TrainingKind::Survival`. Introduce enum to reuse internal parsing pipeline but map to either struct.
- Update CLI/config to request survival mode explicitly (e.g., `--survival`).

### 3.2 Prediction Schema
- Introduce `PredictionDataSurvival` with `current_age`, `pgs`, `pcs`, `sex`, optionally `horizon` vector (or pass at scoring time). If horizon column provided, allow per-person horizon.

### 3.3 Risk-Set Preprocessing
- Precompute unique sorted event ages (including competing events) for baseline spline support.
- Derive log-age transformation arrays and Jacobians (1/a) for derivative calculations.
- Compute Fine–Gray weights before PIRLS: apply cumulative incidence weighting using Kaplan–Meier of censoring/competing risk (requires partial sorting). Use `ndarray` operations; consider `faer` for prefix sums if necessary.

### 3.4 Multiple Competing Risks (Future-proofing)
- For now restrict to a single competing event column. Document extension path: generalize to `Vec<Array1<f64>>` with event type codes.

## 4. Model Representation (`calibrate::model`)
### 4.1 LinkFunction Extension
- Add variant `LinkFunction::RoystonParmarSurvival` capturing metadata:
  ```rust
  pub enum LinkFunction {
      Logit,
      Identity,
      RoystonParmarSurvival(SurvivalLinkConfig),
  }
  
  pub struct SurvivalLinkConfig {
      pub attained_age_origin: f64,
      pub log_age_scale: f64,
      pub baseline_knot_count: usize,
      pub baseline_degree: usize,
      pub fine_gray: bool,
      pub log_time_offset: bool,
  }
  ```
- `SurvivalLinkConfig` stored in `ModelConfig` for inference. Contains baseline spline metadata, integration step size, penalty order(s), mapping between design blocks and covariates.

### 4.2 New ModelConfig Fields
- Add survival-specific configuration: baseline knot placement (quantiles of event ages), penalty structure for baseline/time-varying effects, smoothing parameter priors, offset usage.
- Expand `MappedCoefficients` with survival-specific surfaces (baseline, PGS×age). Provide names for new terms to maintain serialization compatibility.
- Document serialization changes in README.

### 4.3 Null Space & Constraints
- Ensure baseline smooth satisfies identifiability: enforce \(H_0(a_{ref}) = 0\) or \(s(\log a_{ref}) = 0\) to avoid non-identifiable intercept. Choose anchor at median attained age or `log_age = 0` via offset subtraction. Represent via constraint matrix `Z_baseline`. Hook into `sum_to_zero_constraints` map.

## 5. Design & Penalty Construction (`calibrate::construction`)
### 5.1 Baseline B-spline
- Reuse `create_bspline_basis` with log-age input and baseline knots/degree from config.
- Build penalty matrix using existing difference penalty utilities; ensure first derivative penalty for baseline cumulative hazard to enforce smooth log-H0.
- Save baseline basis matrix `B_age` and penalty block `S_age`.

### 5.2 Time-Varying Interactions
- Construct tensor-product basis between log-age and PGS basis for PGS×age effect.
- Use anisotropic penalties by default: `S = λ_age S_age ⊗ I + λ_pgs I ⊗ S_pgs`. Reuse `create_balanced_penalty_root` for null-space handling.
- Guarantee that PGS main effect remains orthogonal to age interaction via `interaction_orth_alpha` infrastructure (existing pure-ti code path). Confirm compatibility; adjust if `ModelLayout` expects logistic identity (examine `build_design_and_penalty_matrices`).

### 5.3 ModelLayout Enhancements
- Extend `ModelLayout` to tag columns belonging to baseline/time-varying blocks. Provide metadata for PIRLS to compute hazard integrals (e.g., `baseline_column_range` stored in `ModelLayout`).
- Provide ability to supply custom offset vector (log exposure) along with design matrix.

## 6. PIRLS & REML Core Changes (`calibrate::pirls`, `calibrate::estimate`)
### 6.1 PIRLS Input Generalization
- Introduce trait `LikelihoodFamily` encapsulating link-specific operations: working response, weights, deviance, gradient, Hessian contributions. Implementations for `Logit`, `Identity`, and new `RoystonParmar`.
  ```rust
  pub trait LikelihoodFamily {
      fn update_working_quantities(&self, eta: ArrayView1<f64>, y: ArrayView1<f64>,
                                   prior_weights: ArrayView1<f64>,
                                   work: &mut PirlsWorkspace) -> Result<PirlsIterationState, EstimationError>;
      fn deviance(&self, ...);
      fn gradient_hessian(&self, ...);
  }
  ```
- Modify `run_pirls` to dispatch on `ModelConfig.link_function` by obtaining boxed `LikelihoodFamily`. This isolates survival-specific math without polluting logistic implementation.

### 6.2 Survival Working Quantities
- `y` in survival context becomes Fine–Gray pseudo response; `prior_weights` incorporate risk-set weights.
- For each subject/time, compute `mu = exp(eta + offset)` (hazard). Working response: `z = eta + (y - mu)/mu` (Poisson canonical). Fisher weight: `w = mu` times prior weights.
- Provide deviance/residual calculations consistent with Poisson log-likelihood (for monitoring but not exported to user).
- Ensure offset is fed into PIRLS pipeline (currently supported? check `run_pirls` signature). If absent, extend to accept optional offset array.

### 6.3 REML Gradient/Hessian
- REML objective requires log determinant of penalized Hessian. With new link, Hessian is `X^T W X + Sλ` as before. Ensure gradient contributions from `family.deviance` align with Poisson.
- Validate that existing `RemlState` logic (kahan sums, `compute_penalty_square_roots`) remains applicable.
- Provide guard rails for cases with near-zero events (add ridge, drop terms?). Possibly incorporate Firth-like correction for separation analog (rare events) by reusing existing Firth toggles? Document as future work.

## 7. Fine–Gray Specific Machinery
### 7.1 Risk Set Construction Module
- Add new module `calibrate::survival::fine_gray` to encapsulate data prep independent of PIRLS.
  - Input: `(age_entry, age_exit, event_primary, event_competing, weights)`.
  - Output: `pseudo_y`, `pseudo_weights`, `offset_log_exposure`, `ordering_indices`, `baseline_knots` suggestions.
  - Steps:
    1. Sort by `age_exit`.
    2. Compute Kaplan–Meier for censoring/competing risk.
    3. Derive Fine–Gray weights `w_i = G(age_exit_i -)`, where `G` is survivor function of censoring/competing events.
    4. Compute `y_i = event_primary_i` (since FG uses subdistribution hazard) but ensure weight `w_i` zeroes out competing events past event time.
    5. Build offsets representing exposure length: `offset_i = log( cumulative_hazard_increment(age_entry_i, age_exit_i) )`. For Royston–Parmar this is integral of baseline hazard; approximate using quadrature with baseline basis evaluation once per iteration (requires storing basis integrals; see §8.2).
- Provide ability to recompute offsets each iteration if baseline hazard changes. To avoid expensive recomputation, express hazard integral analytically: integrate `exp(s(t))` over interval using pre-integrated basis functions. Precompute matrix `I_age` where each row corresponds to subject integral of basis over `[log age_entry, log age_exit]`. Then offset update each iteration reduces to `offset_i = log( I_age[i] · exp(coefficients_baseline) )`. See §8.2 for details.

### 7.2 Baseline Integral Representation
- Represent baseline smooth as \(s(t) = B(t) \theta\), where `B` is basis row.
- Need integral \( \int_{a_{start}}^{a_{end}} \exp(B(t) \theta) \frac{1}{a} dt \) (since dt = da/a). Hard to integrate analytically; choose Gauss-Legendre quadrature on log-age scale using precomputed nodes/weights for each interval (3-5 nodes). Implementation steps:
  - Precompute quadrature nodes (e.g., 7-point Gauss-Legendre) on reference interval [0,1].
  - For each subject, map to `[log a_start, log a_end]`, evaluate `B(t_j)` at nodes, compute `s(t_j)`; hazard integral approximated as `∑ w_j exp(s(t_j))` times `Δt`.
  - Store matrix of basis evaluations at nodes to avoid repeated `create_bspline_basis` calls each iteration (heavy). Reuse basis cache (ensured by `basis::` module).
- Use `offset_i = log( integral )` to feed PIRLS.
- Because offsets depend on current `theta_baseline`, we need iterative update inside PIRLS: after each `eta` update, recompute offsets (since baseline coefficients change). To integrate with PIRLS, treat offset as function of coefficients; we can re-express the score and Hessian to incorporate integral without offset recomputation by using derivative-of-integral. Implementation detail: treat baseline coefficients as part of `Xβ` and include `I_age θ` rows in design matrix. Derive gradient contributions explicitly (requires customizing PIRLS). Document two options and pick one (see §8.3).

## 8. Survival-Specific Linear Algebra
### 8.1 Design Matrix Augmentation
- Extend design builder to compute:
  - `B_baseline`: n × m_b matrix of baseline basis at exit age.
  - `B_pgs`: n × m_p for PGS main effect.
  - `B_pcs`: as existing.
  - `B_pgs_age`: n × (m_p * m_age) for interaction.
  - `I_baseline`: n × m_b matrix representing integrals over interval (for hazard exposures).
- Compose linear predictor for event contributions: `eta = B_baseline θ + X β + ...`.
- Exposure integral enters log-likelihood as `exp(eta_offset)`, so we must adjust `z`/`w` accordingly. Implementation options:
  1. Introduce iteration step to recompute `mu` using `integral = exp(I_baseline θ)` without storing as offset.
  2. Expand dataset with quadrature nodes so that standard Poisson GLM approximates integral via repeated rows (Simpler but increases data). Evaluate trade-off: more rows but avoids custom derivatives. Considering performance (~29k lines), prefer quadrature expansion because `faer` handles large matrices and logistic models already support big n. Document in §8.3.

### 8.2 Quadrature Expansion Strategy
- For each subject interval `[a_entry, a_exit]`, create `Q` pseudo-rows (Q=5-7) with design rows evaluated at quadrature nodes `a_j`. Response `y_j = event_primary_i * δ_{j=last}`? Instead, treat hazard integral as sum of exposures `μ_j = w_j * exp(η(a_j))`. We can restructure as Poisson regression where each pseudo-row has `offset = log(w_j Δt)` and event count `y_j=0`, except final row representing event with `y=1`. Validate correctness with log-likelihood equivalence (should approximate integral). Document theoretical justification referencing Bender et al. (2005). Provide plan to implement helper generating expanded arrays (n×Q). Manage memory by streaming into `Array2` with parallel loops.
- Evaluate performance: With Q=7 and n ~200k, we get 1.4M rows, still manageable given `faer` HPC but ensure workspace scaling (update `PirlsWorkspace::new`).

### 8.3 Alternative: Analytical Gradients (Future Option)
- Document in plan but not immediate: using expectation of derivative to avoid data expansion. Provide stub for future to convert to closed-form integrals once base implementation validated.

## 9. Calibration Layer Integration (`calibrate::calibrator`)
- Survival predictions output absolute risk (probability). Feed calibrator with features similar to logistic case: predicted CIF, standard error (approx), hull distance.
- Need to supply `pred_identity` as base CIF (bounded 0-1). Since calibrator expects real-valued link, convert CIF to logit for calibrator fit. Provide `LinkFunction::Logit` to calibrator while storing survival context for documentation.
- Ensure calibrator training uses same sample weights as survival risk set (use `FineGray` weights aggregated per subject). Provide aggregator mapping from quadrature pseudo-rows back to individuals for calibrator features (since calibration should operate at individual horizon predictions, not pseudo-rows). Possibly reuse `peeled hull` infrastructure for features.

## 10. Scoring API & CLI
### 10.1 Rust API (`calibrate::model::TrainedModel`)
- Extend `TrainedModel` with survival-specific inference method:
  ```rust
  impl TrainedModel {
      pub fn predict_survival(&self, request: SurvivalRequest) -> Result<SurvivalPrediction, PredictionError>;
  }
  ```
  - `SurvivalRequest` contains arrays for `current_age`, `horizon`, `pgs`, `pcs`, `sex`, `weights` (optional), `calibration: bool`.
  - `SurvivalPrediction` returns CIFs, baseline survival, log cumulative hazard, optionally hazard ratios.
- Ensure compatibility with existing logistic `predict` by gating on `link_function` variant.

### 10.2 CLI
- Add `score` subcommand options: `--survival`, `--horizon YEARS`, `--per-row-horizon-column horizon_col`.
- Provide training CLI toggle to ingest survival schema, choose baseline knots (# internal knots default 8) and degree (cubic).
- Document new CLI flags in README.

## 11. Testing Strategy
### 11.1 Unit Tests
- `calibrate::survival::fine_gray` unit tests: verify weight computation on synthetic dataset vs reference R `cmprsk::crr` outputs.
- Baseline basis integrals: check quadrature approximations converge for known analytic `s(t)` (simulate simple log-linear hazard).
- PIRLS dispatch: ensure logistic/Gaussian unchanged via regression tests (existing suite). Add `LinkFunction::RoystonParmarSurvival` branch coverage.

### 11.2 Integration Tests
- Fit small dataset replicating published example (e.g., ovarian cancer data). Compare predicted CIF at ages 50, 60 with `rstpm2` or `flexsurv` references.
- Competing risk scenario: dataset where all events are competing; verify CIF = 0 and model remains stable.
- Left-truncation: create dataset with `age_entry > 0`, ensure risk set excludes earlier times (simulate by verifying log-likelihood vs manual calculation).

### 11.3 Calibration Tests
- Run calibrator on simulated data with known truth to ensure calibrated CIF matches empirical incidence (Kolmogorov–Smirnov test).
- Regression tests for serialization/deserialization of survival models (round-trip via TOML file, ensure predictions preserved).

## 12. Performance & Stability Considerations
- Precompute and cache B-spline basis for log-age evaluations (both event ages and quadrature nodes) to minimize repeated computation. Reuse `basis::` caching with deterministic keys (include node positions and baseline knots in hash).
- Monitor condition numbers in PIRLS: baseline block may become ill-conditioned due to quadrature rows. Use existing `calculate_condition_number` to adapt ridge.
- Provide configuration to adjust quadrature order for accuracy/performance trade-offs.
- Ensure memory usage of `Array2` in quadrature expansion remains manageable (maybe chunked assembly). Use `ndarray::parallel` loops to build design matrix chunk-by-chunk.
- For extremely old ages (tails), add prior penalty on baseline slope to prevent divergence (increase `penalty_order` or add boundary knots repeated).

## 13. Documentation Deliverables
- Update `calibrate/README.md` with survival overview, CLI usage, mathematical derivations.
- Provide example pipeline in `examples/` directory (e.g., `examples/survival.rs` or notebook) demonstrating training + scoring + calibration.
- Document API in Rustdoc for new structs/enums.
- Add developer doc in `plan/` referencing this plan for historical context.

## 14. Implementation Sequencing & Validation Checkpoints
1. **Data Layer**: Implement survival schema parsing (`TrainingDataSurvival`, CLI toggles). Validation: unit test verifying parsing & validation.
2. **Fine–Gray Preprocessing Module**: Build risk set weights, quadrature expansion scaffolding. Validation: compare to R reference.
3. **Design/Penalty Updates**: Extend `ModelLayout`, `build_design_and_penalty_matrices` to produce survival design. Validation: ensure logistic path unaffected (existing tests) and new survival-specific tests compile.
4. **LikelihoodFamily Abstraction**: Refactor PIRLS to use trait. Regression test logistic/Gaussian to ensure identical outputs (floating tolerance).
5. **Survival Family Implementation**: Hook survival-specific working response and quadrature offsets. Validation: simple dataset with constant hazard yields closed-form solution; compare to expected cumulative hazard.
6. **REML Integration**: Ensure smoothing selection works with new family; check gradient/Hessian finite and monotonic. Validate with small dataset (monitor `rho` convergence).
7. **Scoring API**: Implement `predict_survival`. Validation: ensure outputs monotone in horizon, bounded 0-1, consistent with training data by comparing to Monte Carlo simulation.
8. **Calibration Layer**: Adapt calibrator inputs for survival predictions; ensure toggles respect `calibrator_enabled`. Validation: run calibrator training pipeline.
9. **CLI + Examples + Docs**: Expose features and document usage. Validation: CLI integration test on sample dataset.
10. **Performance Profiling**: Benchmark on realistic dataset; tune quadrature order, caching, and parallelism. Provide metrics.

## 15. Future Extensions (Document but Out-of-Scope)
- Multiple competing risks generalization using stacked Fine–Gray pseudo-data.
- Non-proportional hazards for PCs or sex via additional interactions.
- Time-varying covariates implemented via counting process rows.
- Stratified baselines (sex-specific) implemented by block-diagonal penalties.
- Analytical integral approach to replace quadrature for improved speed.
- Bayesian smoothing priors to encode prior knowledge of hazard shape.

