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
- For any time-varying effect, cache both the basis evaluations and their derivatives with respect to `t = log(a)` so that hazard and score computations include `(∂z_i/∂t)^T γ` alongside the baseline derivative `s'(t)`.

### 2.2 Likelihoods
- **Primary endpoint**: Fine–Gray subdistribution hazard for competing risks.
  - Maximize a **full likelihood** built from the parametric subdistribution hazard. For subject `i` with entry age `a_i^{entry}` and exit age `a_i^{exit}`, the contribution is

    `ℓ_i = w_i [δ_i log h_i(a_i^{exit}) - (H_i(a_i^{exit}) - H_i(a_i^{entry}))]`,

    where `δ_i` is the event-of-interest indicator evaluated at exit. Competing events have `δ_i = 0` but still subtract the cumulative hazard so the joint likelihood respects Fine–Gray semantics.
  - Cache value and derivative designs at entry/exit ages so the subdistribution hazard `h_i(a) = exp(η_i(a)) (∂η_i/∂a)(a)` and cumulative hazard `H_i(a) = exp(η_i(a))` are both available. Left truncation is handled through `H_i(a_i^{entry})`, eliminating the need for risk-set bookkeeping.
- **IRLS Formulation**:
- Provide survival-specific working updates that return the score vector and the full negative Hessian derived from the log-likelihood above. The score for observation `i` is `w_i [δ_i \tilde{X}_i^{exit} - ΔH_i X_i^{integral}]` with `ΔH_i = H_i(a_i^{exit}) - H_i(a_i^{entry})` and `\tilde{X}_i^{exit} = X_i^{exit} + D_i^{exit} / (∂η_i/∂a)(a_i^{exit})`. The integral design `X_i^{integral}` reuses the stored entry/exit basis evaluations to avoid numerical quadrature.
  - Extend PIRLS to accept a family implementation that supplies `(U, H, deviance)` constructed from these per-observation quantities, mirroring the existing penalized Newton solver but using dense Hessians that incorporate both event and cumulative-hazard curvature.

### 2.3 Absolute Risk Prediction
- After fitting, compute cumulative incidence for horizon \(h\) at current age \(a_0\):
  \[
  \Pr(T \le a_0+h, \text{cause}=k \mid T > a_0) = \frac{F_k(a_0+h) - F_k(a_0)}{1 - \sum_j F_j(a_0)}
  \]
  where \(F_k\) denotes the Fine–Gray cumulative incidence for cause \(k\). Because the subdistribution hazard references the original time-zero risk set, conditioning on survival to \(a_0\) requires renormalizing by the remaining survival mass \(1 - \sum_j F_j(a_0)\). If a cause-specific conditioning workflow is needed, convert the fitted subdistribution hazards into cause-specific hazards before evaluating the CIF.
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

### 3.3 Hazard Cache Preprocessing
- Precompute unique sorted entry/exit ages for baseline spline support and to drive potential quadrature grids.
- Derive log-age transformation arrays and Jacobians (1/a) for derivative calculations.
- Build reusable caches of `H_i(a)` and `(∂η_i/∂a)(a)` at both entry and exit ages so PIRLS iterations can evaluate `ΔH_i` and event hazards without re-running spline evaluations. Store optional Gauss–Kronrod weights if higher-order integration between entry/exit ages becomes necessary.

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
- `y` remains the event indicator at exit age; censoring/competing adjustments are absorbed through the cumulative hazard difference `ΔH_i`.
- The survival family uses the cached `SurvivalStats` to assemble the per-observation event hazard and cumulative hazard. From these it forms the score `U` and full negative Hessian `H` described in §2.2 and streams them back to PIRLS.
- Working responses are derived from the penalized Newton system `H δ = U`; reuse the existing solver infrastructure to obtain `δ` and update `η` without ever forming diagonal Fisher weights.
- Deviance is the negative twice log-likelihood `-2 Σ_i w_i [δ_i log h_i(a_i^{exit}) - ΔH_i]`. Cache `ΔH_i` and hazard terms for reuse across REML iterations.

### 6.3 REML Gradient/Hessian
- REML objective still requires the log determinant of the penalized Hessian. For the survival family this Hessian is the dense matrix assembled from risk-set cross-products plus penalties; feed it directly into the existing Faer solves and determinant routines.
- Ensure the survival branch populates the trace terms (`tr(W^{-1}S)`) using the same helper functions; they only require access to the assembled penalized Hessian, which the survival family now provides.
- Guard against near-singular Hessians (e.g., few events) by injecting a small ridge (`1e-8`) before factorization, mirroring the safeguards in the other plans.

## 7. Fine–Gray Specific Machinery
### 7.1 Risk Set Construction Module
- Add new module `calibrate::survival::fine_gray` to encapsulate data prep independent of PIRLS.
  - Input: `(age_entry, age_exit, event_primary, event_competing, weights)`.
  - Output: `SurvivalStats` containing sorted indices, Kaplan–Meier censoring survivals `G(t)`, event-time slices, cumulative hazard entry/exit design matrices, and helper arrays for accumulating risk-set weights.
  - Steps:
    1. Sort by `age_exit` and record the permutation.
    2. Compute Kaplan–Meier for censoring/competing risk; cache `G_i(t_k)` evaluated at each event age.
    3. For each event time `t_k`, record the set of at-risk indices (including individuals with prior competing events) and precompute normalized weights numerator `w_i G_i(t_k)`.
    4. Precompute baseline basis evaluations at both entry and exit ages so that each iteration can form `exp(η_exit)` / `exp(η_entry)` with simple dot products.
    5. Store cumulative contribution buffers `Σ_k s_i(t_k)` to avoid recomputing risk-set traversals on every iteration.

### 7.2 Baseline Increment Handling
- Represent the baseline smooth as \(s(t) = B(t) \theta\) and cache both exit and entry evaluations. Because `H_i = exp(s(t) + x_i^T β + …)`, the cumulative hazard increment for each observation is simply `exp(η_exit) - exp(η_entry)`; no numerical quadrature is required.
- Maintain these exponentiated values inside `SurvivalStats` so that each PIRLS iteration can reuse them when forming risk-set denominators and left-truncation adjustments.

## 8. Survival-Specific Linear Algebra
### 8.1 Design Matrix Augmentation
- Extend the design builder to compute and cache:
  - `B_exit`: n × m_b matrix of baseline basis at exit age.
  - `B_entry`: n × m_b matrix at entry age (for left truncation adjustments).
  - `B_pgs`, `B_pcs`, and any additional parametric columns as in existing GAM builds.
  - `TP_exit` / `TP_entry` for time-varying interactions, along with their log-age derivatives.
- Compose the linear predictor at exit and entry ages via shared coefficient vector; expose helpers on `ModelLayout` to retrieve the relevant column spans for prediction and diagnostics.
- Provide lightweight structs in `SurvivalStats` that hold `B_exit · θ`, `B_entry · θ`, and their exponentials to minimize repeated matrix multiplications during PIRLS iterations.

### 8.2 Risk-Set Linear Algebra
- Reuse existing dense cross-product utilities to accumulate `X_{R(t)}^⊤ [diag(s(t)) - s(t)s(t)^⊤] X_{R(t)}` slice by slice. Because each slice references only the rows active in that risk set, process them sequentially to control memory.
- Ensure penalized Hessian assembly reuses existing buffers; the only difference from logistic/Gaussian paths is that the per-slice weight matrix is dense rather than diagonal.

### 8.3 Future Enhancements
- If performance profiling shows the risk-set accumulation dominating runtime, investigate block-sparse representations or low-rank updates. Document this as a future extension rather than part of the MVP.

## 9. Calibration Layer Integration (`calibrate::calibrator`)
- Survival predictions output absolute risk (probability). Feed calibrator with features similar to logistic case: predicted CIF, standard error (approx), hull distance.
- Supply the base CIF on the logit scale to the calibrator while retaining survival-specific metadata so downstream consumers know predictions are conditional absolute risks.
- Calibrator training should aggregate weights at the individual level using the same Fine–Gray sample weights employed during fitting; no pseudo-row expansion is required because predictions are evaluated per individual per horizon.

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
- `calibrate::survival::fine_gray` unit tests: verify risk-set weight computation on synthetic dataset vs reference R `cmprsk::crr` outputs.
- Baseline basis caching: confirm exit-entry evaluations and exponentials remain monotone and match manual calculations on toy datasets.
- PIRLS dispatch: ensure logistic/Gaussian unchanged via regression tests (existing suite). Add `LinkFunction::RoystonParmarSurvival` branch coverage.

### 11.2 Integration Tests
- Fit small dataset replicating published example (e.g., ovarian cancer data). Compare predicted CIF at ages 50, 60 with `rstpm2` or `flexsurv` references.
- Competing risk scenario: dataset where all events are competing; verify CIF = 0 and model remains stable.
- Left-truncation: create dataset with `age_entry > 0`, ensure risk set excludes earlier times (simulate by verifying log-likelihood vs manual calculation).

### 11.3 Calibration Tests
- Run calibrator on simulated data with known truth to ensure calibrated CIF matches empirical incidence (Kolmogorov–Smirnov test).
- Regression tests for serialization/deserialization of survival models (round-trip via TOML file, ensure predictions preserved).

## 12. Performance & Stability Considerations
- Precompute and cache B-spline basis for log-age evaluations at both entry and exit ages to minimize repeated computation. Reuse `basis::` caching with deterministic keys (include knot placement and transform parameters in the hash).
- Monitor condition numbers in PIRLS: dense risk-set Hessians can become ill-conditioned when events are rare. Inject small ridge adjustments and enable step-halving when deviance increases.
- Profile risk-set accumulation; if it dominates runtime, consider batching event times or parallelizing over slices. Document these options for future optimization.
- For extremely old ages (tails), add prior penalty on baseline slope to prevent divergence (increase `penalty_order` or add repeated boundary knots).

## 13. Documentation Deliverables
- Update `calibrate/README.md` with survival overview, CLI usage, mathematical derivations.
- Provide example pipeline in `examples/` directory (e.g., `examples/survival.rs` or notebook) demonstrating training + scoring + calibration.
- Document API in Rustdoc for new structs/enums.
- Add developer doc in `plan/` referencing this plan for historical context.

## 14. Implementation Sequencing & Validation Checkpoints
1. **Data Layer**: Implement survival schema parsing (`TrainingDataSurvival`, CLI toggles). Validation: unit test verifying parsing & validation.
2. **Fine–Gray Preprocessing Module**: Build risk set weights and cumulative contribution caches. Validation: compare to R reference.
3. **Design/Penalty Updates**: Extend `ModelLayout`, `build_design_and_penalty_matrices` to produce survival design. Validation: ensure logistic path unaffected (existing tests) and new survival-specific tests compile.
4. **LikelihoodFamily Abstraction**: Refactor PIRLS to use trait. Regression test logistic/Gaussian to ensure identical outputs (floating tolerance).
5. **Survival Family Implementation**: Hook survival-specific working updates returning risk-set score/Hessian. Validation: simple dataset with constant hazard yields closed-form solution; compare to expected cumulative hazard.
6. **REML Integration**: Ensure smoothing selection works with new family; check gradient/Hessian finite and monotonic. Validate with small dataset (monitor `rho` convergence).
7. **Scoring API**: Implement `predict_survival`. Validation: ensure outputs monotone in horizon, bounded 0-1, consistent with training data by comparing to Monte Carlo simulation.
8. **Calibration Layer**: Adapt calibrator inputs for survival predictions; ensure toggles respect `calibrator_enabled`. Validation: run calibrator training pipeline.
9. **CLI + Examples + Docs**: Expose features and document usage. Validation: CLI integration test on sample dataset.
10. **Performance Profiling**: Benchmark on realistic dataset; profile risk-set assembly, caching, and linear solves. Provide metrics.

## 15. Future Extensions (Document but Out-of-Scope)
- Multiple competing risks generalization using stacked Fine–Gray pseudo-data.
- Non-proportional hazards for PCs or sex via additional interactions.
- Time-varying covariates implemented via counting process rows.
- Stratified baselines (sex-specific) implemented by block-diagonal penalties.
- Low-rank or block-sparse approximations of risk-set cross-products for improved speed.
- Bayesian smoothing priors to encode prior knowledge of hazard shape.

