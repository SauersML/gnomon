# Calibrator Failure Investigation

## Overview
- **Objective:** Understand why `calibrator_fixes_sinusoidal_miscalibration_binary` fails and identify the deep root cause.
- **Date:** 2025-09-29 (updates continue prior 09-28 work).
- **Context:** The unit test expects the calibrator to cut Expected Calibration Error (ECE) in half on a synthetic sinusoidal scenario. Earlier investigations pointed to (a) a GLM Laplace objective bug that drives the optimizer to extreme smoothing, and (b) the possibility that the synthetic fixture is already Bayes-optimal. I revisited both with focused experiments aimed at separating proximal symptoms from the underlying cause.

## Baseline reproduction
- Re-running the test on the unmodified code reproduces the runaway smoothing: the optimizer slams `rho_pred` and `rho_se` into the +20 cap (λ≈6×10⁵), the stabilized EDF collapses to ≈1, and the calibrated ECE stalls at 0.0658 while the baseline is 0.0765.【e50fe3†L1-L218】
- The panic is triggered by the `cal_ece < 0.5 * base_ece` assertion, so understanding why the calibrator cannot move the ECE is the central question.【e50fe3†L219-L233】

## Hypotheses
1. **H1 – GLM Laplace sign inversion (confirmed proximal bug).** The non-Gaussian branch assembles `laml = penalised_ll + ½ log|S| − ½ log|H|`, the opposite of Wood’s expression. Minimizing `-laml` therefore *rewards* large λ, explaining the EDF collapse.【F:calibrate/estimate.rs†L2124-L2374】
2. **H2 – Fixture is Bayes-optimal (implementation-intent gap).** The current test samples labels from the already-wiggly `base_probs`, so the base logits equal `E[y|x]` and the identity mapping is optimal—even though the intent was a straight-line truth with a sinusoidal *prediction* error.【F:calibrate/calibrator.rs†L3372-L3491】
3. **H3 – Sample noise limits observable improvement.** Even though the true miscalibration is zero, the finite-sample ECE (50 bins on 500 points) hovers around 0.07. The calibrator cannot reliably cut that random fluctuation in half without overfitting.
4. **H4 – Identity backbone locks in the base spline.** `compute_alo_features` feeds the PIRLS fit’s logits into `pred_identity`, so the calibrator is explicitly encouraged to stay close to the distorted base curve unless the smooth picks up substantial EDF.【F:calibrate/calibrator.rs†L436-L471】

## Experiments
- **E1: Baseline diagnostics.** Confirmed the saturated λ behavior and logged the failing assertion (see Baseline reproduction).【e50fe3†L1-L233】
- **E2: Objective sign flip.** Temporarily corrected the determinant signs (and gradient) to match Wood’s formula, then re-ran the test. The optimizer now dives to `rho≈-20` (λ≈2×10⁻⁹), restoring full EDF (pred=8, se=5). Despite the added flexibility the calibrated ECE only nudges from 0.0765 to 0.0724, far short of the required 50% drop.【cb8fec†L1-L207】
- **E3: Large-sample Monte Carlo.** Reproduced the fixture in Python with 200 000 points sampled from the distorted probabilities. The empirical ECE shrinks to ≈0.006, demonstrating that as sample size grows the base predictions converge to perfect calibration—the remaining 0.07 in the test is sampling noise, not systematic bias.【fe113a†L1-L28】
- **E4: Oracle curve construction.** Because `base_probs` feed both the evaluation metrics and the Bernoulli sampling step, the oracle mapping is the identity—any spline bump would force predictions away from the true conditional mean.【F:calibrate/calibrator.rs†L3372-L3384】
- **E5: Intent-aligned resample (temporary hack).** Resampled labels from the straight-line logits while keeping the distorted predictions and identity backbone. Base ECE climbed to 0.1006 but the calibrated curve stuck at 0.0763 and λ still maxed out, so the 50 % target remained unreachable until the sign bug is fixed.【0714aa†L482-L551】【0714aa†L558-L564】

## Findings
- **F1: Proximal failure mechanism.** H1 is real—the sign inversion makes the optimizer prefer λ→∞, producing an almost constant calibrator and guaranteeing the assertion failure.【e50fe3†L1-L218】【F:calibrate/estimate.rs†L2124-L2374】
- **F2: Implementation vs. intent.** The code path uses the distorted logits both to sample labels and to seed the identity backbone, so the system is *already* Bayes-optimal unless we change the data generator. This directly conflicts with the intended “true straight line + sine-wave prediction error” story.【F:calibrate/calibrator.rs†L3372-L3491】【F:calibrate/calibrator.rs†L436-L471】
- **F3: Noisy target exacerbates the assertion.** Finite-sample ECE ≈0.07-0.10 is sampling variance, so demanding a 50 % drop forces the optimizer to chase noise, which it cannot do without breaking the identity penalty.【fe113a†L1-L28】【0714aa†L558-L564】
- **F4: Even with the intended DGP, the sign bug blocks success.** I reran the test after temporarily sampling labels from the straight-line logits while keeping the wiggly predictions. The base ECE rose to 0.1006, but the calibrated curve only reached 0.0763 before the λ runaway resurfaced, still missing the 50 % target.【0714aa†L558-L564】【0714aa†L482-L551】

## Recommendations
1. **Fix the GLM Laplace signs.** The bug uncovered in H1 destabilizes every calibrator test and must be corrected for the optimizer to behave sensibly.
2. **Realign the sinusoidal fixture with its intent.** Generate labels from the straight-line logits, keep the distorted predictions as inputs, and ensure the identity backbone references the miscalibrated logits so the spline can “un-bump” them without contradicting the backbone.【F:calibrate/calibrator.rs†L3372-L3491】【F:calibrate/calibrator.rs†L436-L471】
3. **Revisit the assertion strategy.** After fixing the DGP, validate on a hold-out split or compare against an oracle to quantify expected improvement instead of hardcoding a 50 % in-sample drop that is smaller than the observed sampling noise.【fe113a†L1-L28】【0714aa†L558-L564】

## Update — 2025-09-30
- Implemented Recommendation 1 by swapping the determinant signs in the GLM Laplace objective and its gradient, restoring the λ penalty that prevents runaway smoothing.【F:calibrate/estimate.rs†L2200-L2294】【F:calibrate/estimate.rs†L2811-L2930】
- Implemented Recommendation 2 within the unit test: labels now come from the straight-line logits, the base probabilities remain wiggly, and the identity backbone is pinned to the distorted logits so the calibrator can learn the corrective spline.【F:calibrate/calibrator.rs†L3370-L3496】
