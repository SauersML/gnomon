# Allele-frequency variance ratios enter covariance kernels twice

Audit of `Calibrator/PortabilityDrift.lean`, 2026-09-07. The numerical example
below records the defect before the correction. The file already had independent
edits when the audit began; the correction preserves them.

Before the correction, the definitions composed as follows:

1. `tagAlleleFreqRetentionAt` returns the genotype-variance ratio
   `R_i = 2 p_target,i (1-p_target,i) / (2 p_source,i (1-p_source,i))`.
2. `jointTagLDKernelAt` multiplies `R_i * R_j` together with LD, mutation,
   and migration factors.
3. `sigmaTagTargetAt` multiplies each source matrix entry by that kernel.

With the other factors equal to one, the diagonal therefore receives `R_i²`.
For genotype covariance in raw dosage units, or standardized using the source
population's scales, the required diagonal factor is `R_i`.

An exact example, requiring no fitted model or simulation:

| Quantity | Exact value | Decimal |
| --- | --- | --- |
| Source frequency | 1/5 | 0.2 |
| Target frequency | 1/2 | 0.5 |
| Source genotype variance | 8/25 | 0.32 |
| Target genotype variance | 1/2 | 0.5 |
| Variance ratio `R` | 25/16 | 1.5625 |
| Old diagonal kernel `R²` | 625/256 | 2.44140625 |
| Old predicted target variance | 25/32 | 0.78125 |
| Covariance amplitude `sqrt(R)` | 5/4 | 1.25 |

The old prediction is 56.25% above the HWE target variance. This is an
algebraic composition error under the covariance interpretation, independently
of the empirical rejection of the earlier exponential retention formula.

The correction preserves variance-retention functions as
variance ratios and introduces a separate covariance-amplitude factor
`a_i = sqrt(R_i)`. Covariance transport uses `a_i * a_j`; its diagonal is
then `R_i`, and a correlation-preserving transport is `D * Sigma * D`, where
`D = diag(a)`. A covariance contraction or gain is not required to lie below
one: a frequency moving toward one half can increase genotype variance.

The correction makes the coordinate contract explicit. If both
populations independently standardize their genotypes, a correlation matrix
instead has diagonal one, so these frequency factors do not belong in its
diagonal. Existing descriptions mix LD operators, second moments, and
standardized variants. Cross-covariance consumers also need their coordinate
units specified before their tag and causal factors are changed.

Proof implications:

- Preserve the measured variance-ratio definitions and their empirical claims.
- For the covariance interpretation, prove the diagonal identity using
  nonnegative ratios and `Real.sq_sqrt`.
- State source polymorphism (`0 < p_source < 1`) and target frequencies in
  `[0,1]`. Lean's total division at source fixation must not manufacture a
  transport claim from an absent source variance.
- Update the explicit product statements for `jointTagLDKernelAt` and audit
  the direct, proxy, and novel tag-to-causal kernels separately.
- Retain a regression on the composed matrix diagonal. A test of the scalar
  variance ratio alone cannot detect this error.

The old differential check
`alleleFreqMismatchPenalty-is-not-the-variance-retention` still describes the
obsolete exponential function. It squares its callable and expects a MODEL
disagreement, so nonagreement could not certify that this composition was
correct. It is replaced by
`frequency-covariance-retention-preserves-hwe-diagonal`, which executes the
source-extracted covariance-amplitude helper and variance-retention scalar,
and compares their composed diagonal with the independent HWE target variance.
`sigmaTagTargetAt_diagonal` binds the helper's identity to the actual matrix
constructor. Verification results are recorded on GitHub issue #2331.

The same unit error occurs in the shared direct-causal and proxy-tagging
kernels. Their outputs are summed as tag-to-causal cross-covariances and then
contracted with the causal effect vector to obtain `Cov(tag, outcome)`.
Therefore they also receive one standard-deviation factor from each coordinate.
For an identical tag and causal allele changing from 0.2 to 0.5, the old
cross-covariance is 0.78125 while each target variance is 0.5. The joint matrix
has eigenvalues 1.28125 and **-0.28125**, violating positive semidefiniteness.
Even a source correlation of 0.8 becomes an impossible target correlation of
1.25 under that product. The correction preserves the correlation and the
two-coordinate covariance bound. Its helper's squared factor is the product of
the two variance ratios, as proved by
`covarianceRetentionFromVarianceRatios_sq`. A separate 48-point regression
covers unequal coordinate ratios and both signs of perfect correlation.

The existing frequency-path witness also changes quantitatively: for source
frequency 0.5 and target frequency 0.75 its covariance and score variance are
3/4, replacing 9/16. Its asserted target score fraction is consequently
`(3/4)/(2 + 2*(1 - 3/4)^2)`. The algebraic witness is updated with the model;
its previously squared variance factor is not retained as a reference oracle.

The target-only novel kernels had an additional coordinate defect:
they divided by the source causal variance even when the causal allele was absent
in the source. At source frequency zero, total division forces both novel
contributions to zero regardless of the target frequency or positive innovation.
All existing concrete witnesses used zero novel templates, so they could not
exercise the failure. The replacement inputs are explicitly time-indexed
target-coordinate covariance templates. They carry the target scales and are
multiplied only by the existing innovation, migration and proxy-LD modifiers.
The model does not pretend to infer new covariance from absent source variation.
No source-frequency ratio enters these paths. Two source-independence theorems
bind this contract to the actual constructors, and their magnitude bounds are
proved from a supplied template bound and a modifier whose absolute value is
at most one. These premises do not claim arbitrary free matrices are valid
covariances. The extracted-constructor regression exercises both source fixation
points and a polymorphic control, with target frequency 0.5, nonzero templates,
three generations and both signs of covariance.

MSI validation of the initial tag-diagonal correction on 2026-09-07, one pinned
CPU per process (the subsequent direct/proxy and novelty extension awaits its
own constructor regression and Lean compilation):

- Source-extracted covariance regression: all eight frequency pairs agree
  exactly with the HWE target variance, including 0.2 to 0.5 and target fixation.
- Full differential gate calibration: passed in 80.36 seconds, including its
  deliberately broken-grid and empty-instrument controls.
- Fresh extraction/parser ground truth: passed in 2.97 seconds.
- Symbolic regression: passed in 5.75 seconds after generating all required
  artifacts. Mutation testing reached 291 definitions and demonstrated a
  rejecting check for 281, above the regression's 150-definition threshold.

Logs are persisted under the canonical MSI repository's `.validation-logs/`
as `map-audit-final-test_battery_gate.py.log`,
`map-audit-final-test_parser.py.log`, and
`map-audit-symbolic-test_regressions.py.log`.

Follow-up validation on 2026-09-10 covered the direct/proxy and novelty
extension, using the issue fix isolated from the other workspace changes:

- Lean 4.24 compiled `Calibrator.PortabilityDrift` and
  `Calibrator.MechanisticPortabilityWitnesses` successfully against the restored
  project cache, using two pinned CPUs and under 3.6 GB peak RSS. The modules
  took 125.58 and 122.14 seconds respectively, including shared-storage imports.
- Three constructor regressions passed: eight HWE diagonal pairs, 24 unequal
  tag/causal frequency scenarios in raw and source-standardized coordinates,
  and nine target-template novelty scenarios including source fixation.
- Fresh extraction/parser checks and the 73-check differential gate passed;
  the extraction cross-check compared 59 definitions at 162,785 points with
  no discrepancies.

These logs are `PortabilityDrift.log`, `MechanisticPortabilityWitnesses.log`,
`final-extract.log`, `final-parser.log`, `final-covariance.log`, and
`final-gate.log` in the canonical MSI checkout's `.validation-logs/` directory.
This module-level Lean validation does not certify the separate migration and
population-model edits subsequently gathered onto `main`; those require the
full project CI build.
