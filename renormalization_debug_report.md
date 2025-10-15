# Renormalization Alignment Debug Report

## Failing Tests Observed
- `map::project::tests::dropping_variant_matches_manual_renormalization`
- `map::project::tests::renormalization_matches_baseline_without_missingness`
- `map::project::tests::zero_alignment_behavior_respected`

## Diagnostic Instrumentation
- Added temporary logging in the projection tests to print:
  - Squared-norm sums of each loading column.
  - Alignment values and score comparisons when assertions failed.
- The instrumentation exposed that each loading column had a squared norm of 1.5 and that alignments in the renormalized path were returning `sqrt(1.5)` instead of the expected `1.0` when no data were missing.

## Root Cause Analysis
- The renormalization logic accumulated per-sample alignment mass as the sum of squared loadings across observed variants, but it never normalized by the total squared mass of each loading vector.
- Because the PCA loadings are not unit vectors (their squared entries sum to 1.5 in the fixture), the algorithm always produced alignment norms greater than one even when no variants were missing.
- Downstream, manual expectations in the variant-dropping test assumed unit-norm loadings and therefore computed an incorrect expected renormalization factor, leading to score mismatches.

## Fix Implemented
- Compute a per-component normalization factor equal to the total squared mass of that component’s loadings.
- During renormalization, divide each accumulated mass by the corresponding normalization factor before taking the square root.
- Update the manual test expectations to derive the retained mass from these total factors.
- Remove the temporary instrumentation once the bug was verified.

## Verification
- Re-ran only the affected tests with `--nocapture` to confirm the fix and ensure no other regressions:
  - `map::project::tests::renormalization_matches_baseline_without_missingness`
  - `map::project::tests::zero_alignment_behavior_respected`
  - `map::project::tests::dropping_variant_matches_manual_renormalization`

All three now pass with alignments equal to 1.0 in the baseline case and correct renormalization factors when variants are missing.
