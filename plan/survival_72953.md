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
- **Optimization (`calibrate/pirls.rs` & `calibrate/estimate.rs`):** Implements penalized iteratively reweighted least squares and nested REML optimization keyed off `ModelConfig::link_function`. We must add a new link variant with survival-specific working response and deviance calculations, plus support for the additional sufficient statistics the survival likelihood requires (risk-set denominators, counting-process weights).
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

### 3.4 Fine–Gray subdistribution hazard
- For each subject `i`, define:
  - `d_i = 1` if the target event occurred at `a_exit_i`, `0` otherwise.
  - `c_i = 1` if a competing event occurred, `0` otherwise.
  - Weight `w_i` from training data (defaults to 1).
- The subdistribution hazard for subject `i` at age `t` is `λ_s(t | x_i) = d/dt Λ_s(t | x_i)`, where `Λ_s(t | x_i) = H_0(t) ⋅ exp(x_i^⊤ β)` with `x_i` representing covariates evaluated at age `t`. Because RP models parameterize `log Λ_s`, the linear predictor `η_i(t) = log Λ_s(t | x_i)`.
- The Fine–Gray log-likelihood is:
  
  `ℓ(β, θ) = Σ_{i: d_i=1} w_i [η_i(a_exit_i) - log(Σ_{j ∈ R(a_exit_i)} w_j G_j(a_exit_i) exp(η_j(a_exit_i)))]`

  where `R(a_exit_i)` is the Fine–Gray risk set evaluated just before the event time, `G_j` is the censoring weight, and `w_i` is the subject weight from the training data. The log-likelihood depends on the shared risk-set denominator; there is no separate `Δ H_0` factor because the Royston–Parmar parameterization already expresses the cumulative hazard through `η`.
- Implement the computational recipe used by Beyersmann et al.: maintain risk-set weights `W_j(t) = ℓ( max(t, a_exit_j) )` and apply the Fine–Gray cumulative incidence weighting factor `G_j(t)` (Kaplan–Meier of censoring/competing). Practical plan:
  1. Sort subjects by `a_exit`.
  2. Compute censoring weights `G_j(t)` by fitting Kaplan–Meier on the union of target + competing events, treating the target event as failure and competing as censoring for `G`.
  3. For each event time, accumulate `risk_sum = Σ_j w_j G_j(t) exp(η_j(t))` over all subjects with `a_entry_j ≤ t`.
  4. Score contribution for event `i`: `w_i [η_i(a_exit_i) - log(risk_sum)]`.
- Because RP models provide `η_i(a_exit)` directly, we avoid time-derivative computations; however, we need `exp(η_i)` for risk-set sums and `∂ℓ/∂η` for P-IRLS.

- Denote `r_i = w_i` if `d_i=1`, else 0. Let `s_i = w_i G_i(a_exit_i) exp(η_i(a_exit_i)) / risk_sum(a_exit_i)` for all subjects with `a_entry_i ≤ a_exit_i`. The score with respect to `η_i` is `U_i = r_i - Σ_{k: events at t_k} s_i(t_k)` where the sum covers event times `t_k` where subject `i` is in the Fine–Gray risk set. For efficiency we precompute `Σ_{k} s_i(t_k)` using cumulative sums over the sorted time axis.
- Assemble the negative Hessian as the full risk-set cross-product: for each event time `t_k`, form `W_k = diag(s(t_k)) - s(t_k) s(t_k)^⊤` over the risk-set vector `s(t_k)` and accumulate `H = Σ_k X_{R(t_k)}^⊤ W_k X_{R(t_k)}`. This preserves the off-diagonal curvature induced by shared risk denominators and prevents the Newton step from collapsing when multiple individuals share an event time. Implement the accumulation via existing dense cross-product utilities, processing one event slice at a time to control memory.
- Solve the linear system `H δ = U` with the penalized normal-equation solver to obtain the Newton step, then set the working response `z = η + δ`. Inject a small ridge (`1e-8`) into `H` if it becomes near-singular. Observations with zero risk-set involvement contribute only through penalties.
- Provide helper `update_fine_gray_vectors(eta, survival_stats) -> (mu, hessian, z)` returning `mu = exp(η)` for reuse in prediction-standard-error calculations and the assembled Hessian blocks for REML updates.
- Deviance for monitoring: `D = -2 Σ_{events} w_i [η_i(a_exit_i) - log(risk_sum(a_exit_i))]` plus constant; store unpenalized deviance for diagnostics.

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
  - `struct SurvivalStats` containing sorted ages, entry/exit indices, Fine–Gray risk weights, censoring KM estimates, event indicators, and helper arrays for cumulative sums.
  - `fn build_survival_stats(data: &SurvivalTrainingData, prior_weights: &Array1<f64>) -> SurvivalStats` performing:
    1. Sort indices by `age_exit`.
    2. Compute Kaplan–Meier censoring weights `G(t)` treating competing events as censoring.
    3. Precompute for each subject the set of event time positions they influence; encode as `Vec<Range<usize>>` or compressed sparse row arrays to evaluate `Σ s_i(t_k)` quickly.
    4. Precompute `baseline_entry_matrix = B(u_entry)` and `baseline_exit_matrix = B(u_exit)`; store to avoid recomputation per IRLS iteration.
- Provide methods on `SurvivalStats` to update risk-set denominators given current `eta`, returning `RiskContributions` with per-event denominators, subject-level cumulative `s_i` sums, and the working weights `W_i`.

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
- Add module `calibrate/survival/irls.rs` with:
```rust
pub fn update_fine_gray_vectors(
    eta: &Array1<f64>,
    stats: &SurvivalStats,
) -> (Array1<f64>, Array1<f64>, Array1<f64>, FineGrayIterationState)
```
  returning `mu`, `weights`, `z`, and auxiliary state (risk denominators per event, cumulative hazard increments) used later for deviance and smoothing gradient calculations.
- Modify `pirls::update_glm_vectors` to delegate to survival helper when `link == FineGrayRp`. Provide access to `SurvivalStats` via new field in `ModelLayout` or via closure capturing from `estimate::train_model`.
- Ensure P-IRLS workspace caches working vectors sized to survival data. Some arrays (e.g., `delta_eta`) remain applicable without change.

### 6.3 Deviance and gradient tracking
- Add `calculate_survival_deviance(mu, stats, iter_state)` to compute `-2 ℓ`. Integrate into `calculate_deviance` by pattern matching on link. Store deviance per iteration for convergence diagnostics and REML objective.
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
  2. Evaluate the target-event baseline cumulative hazard at both ages: `Λ0_target(current)` and `Λ0_target(horizon)`. Reuse the stored Kaplan–Meier curve for competing events to obtain `CIF_competing(current)`.
  3. Compute covariate shift `s = exp(x^⊤ β)` using stored coefficients (PGS, sex, PCs, optional PGS×age evaluated at `horizon_age`).
  4. Convert to subject-specific cumulative incidences: `CIF_target(age) = 1 - exp(-Λ0_target(age) ⋅ s)` for both `age = current` and `age = horizon`.
  5. Conditional absolute risk for the interval is `(CIF_target(horizon) - CIF_target(current)) / max(1e-12, 1 - CIF_target(current) - CIF_competing(current))`, ensuring the probability is conditioned on having avoided both the target and competing events up to `current_age`.
  6. Return the conditional risk, optionally along with `linear_predictor` and `std_error` if caller requests.
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
   - Build `SurvivalStats` with Kaplan–Meier weights and risk-set structures. Validate via tests comparing to manual calculations on toy datasets.
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
  - `survival::build_survival_stats`: confirm risk-set counts match brute-force enumeration on small dataset (≤5 subjects).
  - `survival::update_fine_gray_vectors`: compare gradient/Hessian to finite differences.
  - `basis::log_age_spline`: ensure evaluation at reference age yields zero contribution after constraint.
- **Property tests**
  - Simulate data from known RP model (use Python or Rust to generate). Fit survival model and assert estimated coefficients within tolerance (e.g., |β_est - β_true| < 0.05) and CIF predictions within ±0.01.
  - Randomly permute input order to ensure invariance (risk sets rely on sorting; tests ensure stable results).
- **Integration tests**
  - CLI-driven training/inference with sample TSVs; confirm output schema and calibration.
  - Compare to R `rstpm2`/`cmprsk` by exporting training data and verifying log-likelihood and CIF at several ages.
- **Performance benchmarks**
  - Benchmark training on synthetic dataset with 100k subjects to ensure runtime scales sub-quadratically and memory remains within reasonable bounds. Profile risk-set preprocessing to identify bottlenecks.
- **Calibrator validation**
  - Evaluate Brier score and calibration plots before/after calibrator on validation split. Confirm calibrator does not violate monotonicity with age by checking CIF increases with horizon after calibration.

## 11. Performance and Numerical Stability Considerations
- Precompute and reuse `exp(η)` and risk-set denominators within each IRLS iteration to avoid repeated exponentiation.
- Implement Kahan summation or pairwise summation for risk-set accumulations to reduce floating-point error when risk sets are large.
- Guard against zero denominators by adding floor `1e-12` to risk sums; report descriptive error if all weights vanish (indicative of malformed data).
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
- GPU-accelerated risk-set computations if profiling reveals bottlenecks.

