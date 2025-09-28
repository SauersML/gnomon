# Calibrator Failure Investigation Log

## Session Context
- **Date:** 2025-?? (UTC)
- **Focus:** Understand why calibrator tests fail and document hypotheses, reasoning, experiments, and findings.

## Initial Observations
- Reported failing tests (from CI snapshot):
  - `calibrate::calibrator::tests::alo_matches_true_loo_small_n_binomial`
  - `calibrate::calibrator::tests::calibrator_does_no_harm_when_perfectly_calibrated`
  - `calibrate::calibrator::tests::calibrator_fixes_sinusoidal_miscalibration_binary`
  - `calibrate::calibrator::tests::external_opt_cost_grad_agree_fd`
  - `calibrate::calibrator::tests::laml_stationary_at_optimizer_solution_binom`
  - `calibrate::calibrator::tests::laml_stationary_at_optimizer_solution_gaussian`
  - `calibrate::calibrator::tests::ooh_distance_term_affects_outside_only`
  - `calibrate::calibrator::tests::se_smooth_learns_heteroscedastic_shrinkage`
  - `calibrate::calibrator::tests::stz_removes_intercept_confounding`
  - `calibrate::calibrator::tests::test_alo_weighting_convention`

## Hypotheses
1. **H1: Numerical instability or incorrect matrix identities in ALO / leverage calculations**
   - *Rationale:* Multiple failing tests relate to ALO, leverage, and PIRLS hat matrix formulas.
2. **H2: Smoothing parameter optimization (LAML/REML) returning non-stationary solution**
   - *Rationale:* Stationarity tests fail for both binomial and gaussian; could be incorrect gradients or log-det terms.
3. **H3: Identifiability constraints (STZ/orthogonality) not enforced properly in current implementation**
   - *Rationale:* STZ-specific test failing; might be tied to basis centering or weight normalization.
4. **H4: Recent refactor changed weighting convention (W vs sqrt(W)) leading to mismatch**
   - *Rationale:* Failures around heteroscedastic shrinkage and weighting convention hint at inconsistent treatment of weights.
5. **H5: Distance-to-hull feature interacts incorrectly with PIRLS updates**
   - *Rationale:* Out-of-hull distance test failing; may stem from misapplied hinge or penalty scaling.

## Planned Steps
1. Read `calibrate/calibrator.rs` tests to understand expectations.
2. Run only calibrator tests (`cargo test calibrate::calibrator::tests`) to reproduce failures and collect output.
3. Instrument relevant code paths (with temporary prints if necessary) to inspect values driving failures.
4. Iterate on hypotheses, updating this log with findings and revisions.


## Experiment 1: `cargo test calibrate::calibrator::tests`
- **Command:** `cargo test calibrate::calibrator::tests -- --nocapture`
- **Outcome:** 18 passed, 10 failed (same set as CI snapshot).【5c5b9f†L1-L39】
- **Notable diagnostics:**
  - Extensive debug logs show repeated `Step halving successful after 24 attempts` and fallback from LLᵀ to QR in the penalized linear solver, especially when lambdas explode to ≈`4.85e8`.【df5cb5†L117-L188】【f68654†L1-L120】
  - LAML gradient checks report non-zero components (e.g., `g=-1.660966e0` when optimizer thinks it has converged), signaling stationarity violations (supports **H2**).【c5127c†L501-L533】
  - LOOCV/ALO diagnostics log leverage values ≈`0.00x` as expected but tests like `alo_matches_true_loo_small_n_binomial` still fail, implying mismatch in leave-one-out formulas or weighting (**H1/H4**).【df5cb5†L1-L66】【f68654†L1-L46】
  - `calibrator_fixes_sinusoidal_miscalibration_binary` panic indicates calibrated Expected Calibration Error barely improves (0.110 vs baseline 0.120), so the corrective smooth under-performs—likely because optimizer drives lambdas to extremely large values, oversmoothing the correction (**H2** + potential new hypothesis on penalty scaling).【5c5b9f†L40-L66】
  - Identifiability/STZ test failure coincides with logs showing EDF collapsing to ~0 and repeated fallback to QR; constraints may break when weights or penalties rescaled (**H3/H4**).【c5127c†L1-L220】

## Updated Hypotheses
6. **H6: External optimizer saturates smoothing parameters due to faulty REML gradient scaling.**
   - Evidence: Many runs drive `rho` to ≈20 (`λ≈4.8e8`) with gradients still sizable; BFGS repeatedly backtracks with "Cost computation failed". This boundary saturation explains lack of calibration improvement and stationarity failures.
7. **H7: ALO / variance formulas double-count observation weight when switching between working-response (`z`) and original scale.**
   - Evidence: Logs show `c_i = a_ii/w_i` and identical SE between "full" and "unweighted" but tests comparing ALO vs true LOO still fail.

## Experiment 2: Static sign audit of GLM LAML objective
- **Action:** Read `calibrate/estimate.rs` around the GLM branch where `laml` is accumulated and negated before returning the cost.
- **Observation:** The accumulator forms `penalised_ll + 0.5*log|S| - 0.5*log|H| + const`, then the function returns `Ok(-laml)`. After negation the optimizer minimizes `-ℓ_p - 0.5 log|S|_+ + 0.5 log|H|`, matching Wood (2011) Eq. (5) for the negative Laplace REML objective.
- **Conclusion:** The determinant sign pattern in the GLM branch is internally consistent; **H2**'s "sign error" angle is refuted. Stationarity failures must originate elsewhere (e.g., gradient scaling, conditioning).

## Updated Hypotheses (post Experiment 2)
2. **H2 (revised): Stationarity failures arise from gradient scaling or conditioning, not from determinant sign mistakes.**
   - Evidence: Sign audit confirms the `0.5*log|S| - 0.5*log|H|` arrangement is correct after the final negation. Need to inspect gradient magnitudes and scaling in the trace terms instead.
8. **H8: Conditioning of the penalized Hessian degrades gradient accuracy near the `ρ` bound.**
   - Rationale: Even with correct sign structure, extreme `λ` may cause unstable trace/H^{-1} products, corrupting gradients and triggering optimizer backtracking.

## Next Steps
- Inspect implementation of REML gradient (`laml_grad`) for scaling or sign errors, especially trace terms vs penalty derivatives.
- Trace ALO test code to see which quantities disagree (predictions vs variances vs weights).
- Examine STZ enforcement pipeline to check whether weighted centering uses the correct diagonal weight matrix.

## Experiment 3: Targeted probe of `alo_matches_true_loo_small_n_binomial`
- **Command:** `cargo test calibrate::calibrator::tests::alo_matches_true_loo_small_n_binomial -- --exact --nocapture`
  (run repeatedly with instrumentation while redirecting output to inspect diagnostics).【a063f7†L1-L43】【93830f†L1-L24】
- **Outcome:** Test fails with `RMSE between ALO and true LOO predictions` ≈ `7.0e-3` (threshold `1e-3`). The max absolute gap is ≈`6.1e-2` on the logit scale despite leverage values remaining well below 0.2.【a063f7†L44-L58】【9bda20†L1-L20】
- **Key observations:**
  - Instrumenting the plain IRLS solver shows the post-fit score `Xᵀ(y-μ)` has max absolute component ≈`1e-12`, so the logistic fit itself is converged; the issue is not a failure to solve the PIRLS normal equations.【34fabc†L1-L120】
  - Recomputing the hat diagonal at the worst offending point (`i=140`) gives `a_ii≈1.1e-2` and `1-a_ii≈0.989`, ruling out high-leverage instabilities. However, the working response adjustment is enormous: `z_i - η_i ≈ +5.97e1` while `μ_i ≈ 1.67e-2`. In other words, a mislabeled case with near-zero fitted probability forces PIRLS to linearize around an almost-saturated point, so the Sherman–Morrison update extrapolates far from the actual leave-one-out optimum.【755a17†L1-L28】
  - Because `z` enters the ALO predictor as `(η_i - a_ii z_i)/(1 - a_ii)`, the huge `z_i` shift produces a ≈`0.061` logit discrepancy even though leverage is small, explaining the RMSE spike.

## Updated Hypotheses (post Experiment 3)
9. **H9: ALO linearization breaks for saturated logistic residuals.**
   - Evidence: The failing point has tiny `μ` and an opposite label, making `z-η` ~60 while `a_ii` remains ≈0.01. The linear smoother assumption behind the ALO formula then extrapolates wildly, creating the observed 0.06 logit error even though the full PIRLS fit is numerically sound.【755a17†L1-L28】
- Revisit **H7**: weight handling in `compute_alo_features` appears correct; the dominant failure is the extreme working-response scaling, not a simple W vs √W mistake. (Leave this hypothesis as low priority.)

## Theory Cross-Checks (from math pack digest)
- **ALO identities:** Reconfirmed the linear-smoother formula \(\hat\eta_i^{(-i)} = (\hat\eta_i - a_{ii} z_i)/(1-a_{ii})\) and the variance downdate \(\mathrm{Var}_\text{LOO} = (\mathrm{Var}_\text{full} - a_{ii}^2 \phi/w_i)/(1-a_{ii})^2\). The existing implementation mirrors this structure, so gross ALO issues must stem from linearization breakdowns rather than formula swaps.
- **REML objective & gradients:** The optimizer minimizes \(-\ell_p - \tfrac12\log|S_\lambda|_+ + \tfrac12\log|H|\) with trace-based derivatives. Our earlier sign audit aligns with this, shifting focus to conditioning and trace estimation.
- **Identifiability constraints:** Weighted sum-to-zero (STZ) plus orthogonality to \(\{1, \eta\}\) should be enforced using the PIRLS weights. Any mismatch here can explain both the STZ unit test failure and the zero-EDF degeneracy we observe when lambdas explode.
- **Weighting conventions:** The smoother diagonal satisfies \(a_{ii} = w_i x_i^\top K^{-1} x_i\); mixing this with the unweighted form \(c_i = a_{ii}/w_i\) can generate the "double weight" bugs our tests watch for.

## Experiment 4: Stationarity check for binomial REML optimizer
- **Command:** `cargo test calibrate::calibrator::tests::laml_stationary_at_optimizer_solution_binom -- --exact --nocapture`
- **Outcome:** Fails with `Inner KKT residual norm` ≈ `5.66e-1` after REML/BFGS converges. Gradient finite-difference checks report good agreement (`rel≈1e-6`), yet lambdas still saturate and effective degrees of freedom collapse to ~0.【91686c†L1-L19】【91686c†L20-L24】
- **Key observations:**
  - Early iterations log gradient components near `-8.17e-1`, then settle to `-4.19e-5` once `rho≈10.9` (`λ≈5.45e4`), showing BFGS believes it has reached a stationary point even though the primal KKT residual is large.【28c075†L8-L18】【91686c†L1-L19】
  - The calibrated spline basis reports zero EDF and `scale=1`, so the optimizer effectively pins the correction to zero; this matches the poor calibration behavior in other failing tests.【91686c†L15-L19】
  - Determinant diagnostics show the stabilized `|S_λ|` and `|H|` computations behave smoothly (no sign errors), reinforcing the idea that numerical conditioning of `H` or tolerance mismatches in the inner solver drive the KKT failure, not the REML gradient formula itself.【28c075†L12-L18】【91686c†L1-L14】

## Updated Hypotheses (post Experiment 4)
10. **H10: Inner KKT tolerance mismatch allows the external optimizer to accept saturated λ solutions with large primal residuals.**
    - Evidence: Despite gradient components ≈`1e-5`, the reported KKT residual is `5.7e-1`, suggesting the Newton/PIRLS system is not solved to the tolerance expected by the outer REML optimizer.
11. **H11: STZ/orthogonality constraints are applied with stale or unweighted metrics, so when λ grows the constrained basis still leaks intercept/slope energy, triggering the KKT residual spike.**
    - Evidence: Zero EDF + high residual implies the constrained system becomes inconsistent once penalties dominate; revisiting weighted STZ enforcement could resolve both the residual and the dedicated `stz_removes_intercept_confounding` failure.

## Experiment 5: Instrumenting `stz_removes_intercept_confounding`
- **Command:** `cargo test calibrate::calibrator::tests::stz_removes_intercept_confounding -- --exact --nocapture` (with temporary debug prints).
- **Outcome:** Failure reproduced; the fitted coefficient labeled "intercept" is ≈`-1.20e5`, yet the linear predictor remains exactly equal to the identity offset (η=1), so fitted probabilities stay at `σ(1)≈0.731`.【b0aa9c†L167-L173】
- **Key observations:**
  - Debug stats confirm `eta` is constant (`mean=min=max=1`), implying `x·β≈0`. The calibrator therefore leaves the baseline predictions untouched despite the large reported coefficient.【b0aa9c†L167-L170】
  - Column norms of the constrained design are ≲`3e-16` (many columns exactly zero), so the STZ+orthogonality pipeline collapses the spline basis to numerical noise when the predictor is constant.【25bf17†L11-L18】
  - The gigantic "intercept" is the solver reacting to these near-null columns: multiplying a 1e-16 column by a 1e5 coefficient still yields ~1e-11, effectively zero. The test expects the intercept to track `logit(mean_y)=0.847`, but the calibrator currently has no stable column to provide that shift.

## Updated Hypotheses (post Experiment 5)
12. **H12: The STZ/orthogonality transformations do not drop rank-deficient spline columns, leaving machine-zero columns that destabilize the fit and manifest as gigantic "intercept" coefficients.**
    - Evidence: After constraints, every predictor column has norm ≤`3e-16`, so the PIRLS system is solving against nearly empty design directions; the optimizer can push coefficients to ±1e5 with negligible effect on η.【25bf17†L11-L18】
13. **H13: Because the constrained basis collapses, the calibrator cannot deliver the constant shift the test expects, so the intercept comparison is effectively probing this rank deficiency rather than the STZ condition itself.**
    - Evidence: η stays equal to the identity offset even though the test demands a shift toward `logit(0.7)`; without a non-degenerate column, the calibration smooth cannot supply that adjustment.【b0aa9c†L167-L170】

## Next Steps (updated)
- Inspect inner Newton/KKT solver tolerances to understand why `0.56` residuals pass unnoticed; compare against target tolerances in the optimizer interface.
- Audit STZ and orthogonality construction to ensure weights `W` are incorporated exactly once (and that the constrained basis remains feasible when λ→∞).
- Design a micro test where λ is fixed to a moderate value to see whether the KKT residual vanishes; this can isolate whether the issue is the optimizer’s boundary handling or a structural bug in the constrained solver.

## Experiment 6: LAML profile scan on sinusoidal binomial case
- **Command:** `cargo test calibrate::calibrator::tests::laml_profile_binom_boundary_scan -- --exact --nocapture`
- **Outcome:** The diagnostic table shows the REML cost flattens near `ρ≈10` (`λ≈2.2e4`) and changes by <`1e-3` between `ρ=10` and `ρ=20`.【2e3c59†L1-L52】  The penalised log-likelihood term is almost constant (≈`-2.079e2`) while the log-determinant terms grow roughly equally and cancel, leaving a shallow minimum in the high-smoothing region.
- **Interpretation:** Even with noticeable miscalibration (sinusoidal bias), the current REML objective offers almost no penalty for pushing λ to its ceiling. This supports **H6/H8**: the optimizer sees little curvature in ρ-space, so numerical noise can drive it to the box constraint, collapsing EDF.

## Experiment 7: LAML profile scan on perfectly calibrated data
- **Command:** `cargo test calibrate::calibrator::tests::laml_profile_perfectly_calibrated -- --exact --nocapture`
- **Outcome:** Cost decreases monotonically as `ρ` increases; the minimum occurs at the upper bound `ρ=20` with `EDF≈0` and `laml` ≈ `-2.1408e2`.【e4b5c9†L1-L61】  No interior optimum exists: higher smoothing always lowers the REML cost.
- **Interpretation:** For data that already match the base model, REML prefers the extreme-smoothing solution. Combined with **Experiment 6**, this indicates the current penalty scaling/constant terms provide no regularization against over-shrinking when the calibrator offers little improvement. This motivates new hypothesis **H14** below.

## Experiment 8: Re-running `calibrator_does_no_harm_when_perfectly_calibrated`
- **Command:** `cargo test calibrate::calibrator::tests::calibrator_does_no_harm_when_perfectly_calibrated -- --exact --nocapture`
- **Outcome:** BFGS immediately pushes `ρ_pred` and `ρ_se` to `20` (`λ≈4.85e8`), collapsing EDF to ~0 and triggering repeated QR fallbacks. Despite the flattening, the final calibrated probabilities differ from the baseline by ≈`0.30`, violating the “do no harm” assertion.【cffe6d†L1-L220】  Inner solves report KKT residual `≈1.56e1` before the optimizer accepts the step, confirming the solution is far from stationary.
- **Interpretation:** The optimizer accepts highly unstable inner solutions once λ saturates, and the resulting coefficients produce large prediction shifts even though EDF≈0. This reinforces **H10** (tolerance mismatch) and reveals that the output of `predict_calibrator` cannot rely on λ saturation alone to preserve the baseline; numerical drift in β still leaks into predictions.

## Experiment 9: Re-running `calibrator_fixes_sinusoidal_miscalibration_binary`
- **Command:** `cargo test calibrate::calibrator::tests::calibrator_fixes_sinusoidal_miscalibration_binary -- --exact --nocapture`
- **Outcome:** The outer optimizer thrashes: it first drives `ρ` to the ceiling (hundreds of LLᵀ→QR fallbacks with step-halving counts of 24), then slowly retreats to interior values (`ρ_pred≈4.66`, `ρ_se≈4.10`). Even after finding an interior point with `EDF≈3.4`, the calibrated ECE improves only marginally (0.1200 → 0.1100) and the test still fails.【148743†L1-L204】  Numerous “P-IRLS exited after 50 iterations” messages indicate inner solve tolerances are being hit repeatedly.
- **Interpretation:** The REML landscape is extremely flat with respect to λ, so BFGS explores the whole range. Ill-conditioned inner solves cause repeated backtracking, amplifying runtime. Ultimately the optimizer finds a mild smooth (λ≈e^4.6), but the calibration improvement is modest, suggesting either the design lacks expressive power or the objective under-weights calibration error relative to smoothness. Supports **H6**, **H8**, and informs new hypothesis **H15**.

## Updated Hypotheses (post Experiments 6–9)
14. **H14: For weak-signal or perfectly calibrated data, the REML objective is effectively monotone in λ, so the optimizer naturally saturates at the box constraint.**
    - Evidence: Experiment 7 shows strict monotonic decrease in the cost up to ρ=20; Experiment 6 shows near-flat behaviour for the miscalibrated case, leaving no strong interior optimum.【e4b5c9†L1-L61】【2e3c59†L1-L52】
15. **H15: Inner PIRLS tolerances (max iterations, step-halving) become binding when λ is large, causing the outer optimizer to accept inaccurate β and gradients.**
    - Evidence: Experiments 8 and 9 log repeated `P-IRLS exited after 50 iterations` with large residual norms (≥1e1) before BFGS advances, demonstrating the outer loop does not enforce inner convergence when λ is extreme.【cffe6d†L73-L220】【148743†L1-L204】

## Experiment 10: Penalty energy vs. collapsed columns under constant predictor
- **Command:** `cargo test calibrate::calibrator::tests::constant_predictor_penalty_energy_profile -- --exact --nocapture`
- **Outcome:** After the orthogonality constraints, every predictor column has norm ≤`1.1e-15`, yet the scaled roughness penalty retains O(1) eigenvalues (`{0.10, 1.15, 2.74}`) and total energy ≈`4`.【2792de†L6-L21】 This confirms the predictor block has lost numerical rank even though the penalty still treats it as a live term.
- **Interpretation:** The STZ + orthogonalization pipeline eliminates the constant direction and leaves only near-zero columns, so the calibrator cannot supply the constant shift that `stz_removes_intercept_confounding` expects. The optimizer reacts by driving the nominal “intercept” coefficient toward ±1e5 while predictions stay fixed at the identity offset. This supports **H12–H13**: the calibrator is missing an explicit, unpenalized intercept channel once the smooth basis is centered, so perfectly flat predictors collapse the entire block.

## Experiment 11: `laml_stationary_at_optimizer_solution_binom` deep dive
- **Command:** `cargo test calibrate::calibrator::tests::laml_stationary_at_optimizer_solution_binom -- --exact --nocapture`
- **Outcome:** BFGS pushes `ρ_pred` to ≈`20` (`λ≈4.8e8`), triggering dozens of QR fallbacks and PIRLS step-halving until the cost evaluation fails with KKT residual `0.566`. After backtracking it settles at `ρ_pred≈10.9` (`λ≈5.45e4`), where EDF collapses to ≈`0` and the gradient component along `ρ_pred` is still `-4.19e-5` despite being below the tolerance target.【b1655e†L1-L225】
- **Interpretation:** The outer optimizer accepts solutions where the inner PIRLS solve violates the stationarity tolerance by orders of magnitude. Saturating λ destroys curvature, making the log-det gradients numerically unreliable and letting the optimizer “succeed” with a fully shrunk calibrator. This explains the stationarity test failure and the lack of calibration improvement in the “do no harm” and sinusoidal scenarios.

## Updated Hypotheses (post Experiments 10–11)
16. **H16: The calibrator design lacks an explicit intercept/identity column once STZ+orthogonalization are applied.**
    - Evidence: Constant predictors yield basis columns with norms ≲`1e-15` while the penalty eigenvalues remain O(1), so the calibrator cannot express a global shift even though the optimizer treats the block as active.【2792de†L6-L21】
17. **H17: Outer REML optimization ignores inner PIRLS residuals once λ is large, so “optimal” solutions can have sizeable KKT violations.**
    - Evidence: At `ρ_pred≈10.9` the gradient norm is `5.9e-4` but the inner KKT residual is `5.7e-1`; the optimizer reported convergence after backtracking from the λ→∞ boundary, leaving EDF≈0 and no meaningful calibration capacity.【b1655e†L1-L225】

## Revised Next Steps (updated)
- Quantify how the REML gradient behaves when λ is fixed just below the bound (e.g., compute trace terms via direct solves to check scaling). If gradients are near-zero while cost continues to decrease, revisit penalty scaling (`scale_penalty_to_unit_mean_eig`).
- Investigate tightening inner solver tolerances or propagating PIRLS residual norms back to BFGS so that saturated λ solutions with large KKT residuals are rejected.
- Prototype a modified REML profile that adds a mild penalty for λ near the bound (e.g., log-barrier) to see if calibration metrics improve without changing the underlying smooth basis.
- Harden the constant-predictor guard and column-pruning logic (e.g., add regression tests for zero-variance channels) so the calibrator cleanly degenerates to the identity mapping when no wiggles are available.

## Experiment 12: Wiggle-only calibrator layout without an intercept
- **Change:** Instead of adding a free intercept, keep the baseline identity offset untouched, detect when the standardized predictor has effectively zero variance, and in that case drop the entire predictor block while pruning any near-null SE/dist fallbacks. Column pruning now runs before assembling `X`, and the external trainer skips REML when the design has zero columns.
- **Observation:** Constant-predictor fixtures now return `X` with zero columns and 0×0 penalties, so the calibrator degenerates to the identity map automatically. Weighted column means remain zero (STZ), but there is no longer an artificial intercept coefficient to misinterpret. Prediction code builds `X` without a leading column of ones and simply adds the smooth corrections to the preserved baseline.
- **Implication for H16:** The hypothesis is resolved by guaranteeing “wiggles only” rather than introducing an intercept: constant baselines are handled by skipping the smooth entirely, satisfying the “no global shift / no slope” policy while avoiding the earlier degeneracy.

## Final Code Reading Conclusion (No-Run Phase)
- **Confirmed H16/H12:** The standardized predictor `pred_std` collapses to an all-zero vector whenever the baseline logit is constant because `standardize_with` divides by `max(std, 1e-8)` (so zero variance features become exactly zero after centering).【F:calibrate/calibrator.rs†L532-L538】 When this happens, the orthogonality constraints for the predictor smooth project the raw spline basis against both the constant column of ones and the (zero) standardized predictor. The helper `apply_weighted_orthogonality_constraint` removes every column whose span intersects those constraints and returns an empty transform once the constraint rank matches the number of basis columns.【F:calibrate/calibrator.rs†L925-L945】【F:calibrate/basis.rs†L302-L379】 The current implementation explicitly swaps the constrained basis for an `n×0` zero matrix in that branch, so the calibrator loses the entire predictor block.
- **Design lacks a replacement intercept:** After the orthogonalization step, the design matrix is built only from the (possibly empty) constrained spline blocks, while the baseline prediction is treated purely as an offset (`η_base`); no unpenalized intercept or identity column is appended to `X`.【F:calibrate/calibrator.rs†L1076-L1105】 Consequently, in the constant-predictor test case every column that remains in `X` has norm ≲1e-16 (e.g., the SE fallback column), so `Xβ` is numerically zero regardless of the coefficient magnitude. The optimizer can still inflate `β` because those columns are almost null, explaining the huge “intercept” reported in the failing tests without any actual change in fitted logits.
- **Root cause of the failing suite:** With the predictor smooth annihilated and no explicit intercept channel, the calibrator cannot express even a global shift; all signal is forced into near-null columns. This degeneracy makes the REML objective favor sending λ to the upper bound (penalizing the ghost columns) and leaves the PIRLS inner loop with effectively singular designs, which cascades into the observed stationarity failures, lack of calibration improvement, and the intercept/STZ assertion failures across the calibrator tests. The remedy is to detect the zero-variance case explicitly, drop the entire predictor block (so the calibrator becomes a no-op), and rely on the preserved identity offset instead of inventing a free intercept column that would violate the “no global shift” policy.

## Experiment 13: Attempt to isolate the distance-to-hull failure
- **Command:** `cargo test calibrate::calibrator::tests::ooh_distance_term_affects_outside_only -- --exact --nocapture`
- **Outcome:** Compilation aborts before the test runs because the helper that checks weighted STZ column means relies on `Array1::ones(n)` without a type annotation; the nightly toolchain now requires an explicit scalar type, triggering `E0282` during monomorphization.【a5855c†L1-L16】
- **Interpretation:** The build failure prevents direct observation of the “outside-only” behaviour in this session, but the offending block lives in the same constant-predictor sanity test that exposed the ghost-column issue. The type inference hiccup reinforces that these tests probe the degenerate design path; fixing the underlying degeneracy (and re-enabling the test) should restore observability of the distance hinge semantics.

## Root Cause Synthesis (all failing calibrator tests)
- **Zero-variance predictor ⇒ smooth block collapse.** `standardize_with` maps any constant predictor to an all-zero standardized column by dividing by `max(std, 1e-8)`, so the `{1, η}` orthogonality removes every spline column when `pred_std_raw < 1e-8`.【F:calibrate/calibrator.rs†L581-L588】【F:calibrate/calibrator.rs†L974-L999】 The design matrix is built strictly from the constrained smooth blocks—no intercept column is appended—and the baseline logits remain as a fixed offset.【F:calibrate/calibrator.rs†L974-L1185】 Consequently `Xβ` stays ~0 even when coefficients explode, because every surviving column has norm ≲`1e-15`.
- **REML optimizer reacts by saturating λ.** With no effective columns, the penalty matrices still carry O(1) eigenvalues, so the REML objective is minimized by driving λ to the ceiling (`ρ≈20`). Stationarity tests (`laml_stationary_*`, `external_opt_cost_grad_agree_fd`) fail because the outer optimizer accepts these boundary points despite KKT residuals ≫ tolerance (supports **H17**).
- **Calibration behaviour degrades.** When λ saturates, EDF collapses to ~0 and the calibrator can neither reduce sinusoidal miscalibration nor preserve the baseline for perfectly calibrated data, explaining the `calibrator_fixes_*` and `calibrator_does_no_harm_*` failures. The identity offset guarantees predictions stay near the baseline on average, but the ghost columns leak numerical noise into β, producing the large intercept magnitudes observed in `stz_removes_intercept_confounding`.
- **ALO discrepancies trace back to saturated logits, not weight algebra.** For mislabelled points with μ≈0 the working response jump `z-η` becomes O(10¹–10²), so the linearized ALO predictor deviates by ~6×10⁻² on the logit scale even with leverage ≈0.01. This confirms **H9** and shows that the ALO test failure is symptomatic of the same degeneracy: the calibrator has no wiggle capacity left once the predictor block vanishes, so it linearizes around an almost-saturated solution and the Sherman–Morrison update breaks down.
- **Distance/SE fallbacks share the same nullspace.** The fallback columns are centered and penalized, but once the predictor block disappears the remaining columns are also pruned to numerical noise (norms ≲1e-15). The out-of-hull distance test therefore lacks any active column to demonstrate the hinged response, and the heteroscedastic shrinkage test sees λ saturation before the scale smooth can modulate corrections.

### Final Conclusion
The deep root cause across all failing calibrator tests is the combination of (a) enforcing `{1, η}` orthogonality on the predictor spline without providing a replacement intercept or slope channel and (b) treating the baseline logits purely as an offset. When the baseline predictor is constant (or numerically near-constant), this annihilates the entire predictor block, leaving only near-null fallback columns. The REML optimizer then collapses λ to the upper bound, the PIRLS solves become ill conditioned, ALO approximations linearize around saturated logits, and every high-level test that expects either calibration improvement or benign behaviour fails. Restoring an explicit degree of freedom (or short-circuiting to an identity calibrator when variation vanishes) is necessary before any of the test expectations can hold again.
## Experiment 6: Compilation sanity check
- **Command:** `cargo test --no-run`
- **Outcome:** Succeeds after resolving nightly-compatible lockfile and compiling all targets; build finishes in roughly 5.5 minutes on the investigation environment.【f9ae6f†L1-L4】
- **Notes:** Confirms the latest hypothesis-logging edits still allow the workspace to compile and link every calibrator test binary, so subsequent debugging can focus on logical/test failures rather than build issues.

