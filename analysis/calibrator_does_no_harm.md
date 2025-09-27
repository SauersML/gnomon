# Why `calibrator_does_no_harm_when_perfectly_calibrated` Fails

## What data flow into the calibrator
The calibrator is a three-smooth GAM built on the upstream model’s leave-one-out
prediction (`pred`), its leave-one-out standard error (`se`), and an optional
peeled-hull distance (`dist`). We also retain the original in-sample logits as
`pred_identity` so inference can fall back to the baseline scores without paying
any penalty. `compute_alo_features` populates all four channels by replaying the
P-IRLS fit and solving the Sherman–Morrison adjustments, so every training row
comes with both the high-variance ALO predictor and the deterministic backbone
logit.【F:calibrate/calibrator.rs†L24-L475】【F:calibrate/calibrator.rs†L435-L475】

## Where training and inference diverged
`build_calibrator_design` historically standardised the spline block off
`features.pred` (ALO) while the unpenalised identity column drew from
`features.pred_identity`. At inference, however, `predict_calibrator` only
accepts a single predictor array, and every call site—including
`calibrator_does_no_harm_when_perfectly_calibrated`—feeds it the preserved
baseline logits (`alo_features.pred_identity`).【F:calibrate/calibrator.rs†L552-L567】【F:calibrate/calibrator.rs†L1253-L1338】【F:calibrate/calibrator.rs†L1387-L1548】【F:calibrate/calibrator.rs†L3671-L3706】

That mismatch meant the knots, standardisation parameters, and STZ transform
captured the distribution of the noisy ALO channel, yet the spline basis was
evaluated on the much tighter in-sample logits at serve time. The columns the
REML optimiser had tuned against `pred` were therefore evaluated on shifted,
compressed inputs, so the fitted coefficients no longer represented the
identity-plus-small-perturbation shape they had during training.

## How the mismatch breaks “do no harm”
Because the spline basis was centred and scaled for the high-variance ALO
predictor, replaying it on the smoother baseline logits produced large offsets
even when the base model was perfectly calibrated. In the failing test the
identity backbone kept the intercept near the baseline, but the mis-evaluated
penalised columns dumped sizeable adjustments on top—enough to push already
calibrated 0.51 probabilities to ~0.70 for the same case, tripping the
assertions that the calibrator should leave perfect data untouched.【F:calibrate/calibrator.rs†L3671-L3706】

## Root cause and remedy
The failure was not a lack of regularisation; it was the inconsistent predictor
channel between training and inference. Aligning the spline to the same
`pred_identity` inputs the backbone uses (and that inference can actually
provide) removes the train/serve skew: the design matrix now standardises and
builds knots from the baseline logits, and the stored parameters drive
`predict_calibrator` on exactly that channel.【F:calibrate/calibrator.rs†L552-L567】【F:calibrate/calibrator.rs†L1253-L1338】【F:calibrate/calibrator.rs†L1387-L1548】

With the spline and identity column seeing the same predictor end-to-end, the
calibrator regains a zero-cost path to the baseline, so perfectly calibrated data
remain near the identity and `calibrator_does_no_harm_when_perfectly_calibrated`
passes.
