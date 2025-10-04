# Mistake in `test_model_realworld_metrics`

## Summary
- The cross-validation split implemented inside `test_model_realworld_metrics` in `calibrate/estimate.rs` attempts to remove the validation fold from the shuffled index list, but it filters on the *values* of the indices instead of their *positions* within the shuffled vector.
- Because the comparison uses the fold boundary positions (`start`/`end`) against raw sample indices, many validation observations leak back into the training set, and unrelated training samples may be omitted.
- This data leakage invalidates the reported metrics (AUC, calibration, etc.) and explains why the test takes so long and produces overly optimistic statistics.

## Evidence
```rust
let val_idx: Vec<usize> = idx[start..end].to_vec();
let train_idx: Vec<usize> = idx
    .iter()
    .cloned()
    .filter(|i| *i < start || *i >= end)
    .collect();
```
- `start`/`end` are **positions** inside the shuffled `idx` vector, not label values. However, the predicate `*i < start || *i >= end` compares the *sample IDs* (values inside `idx`) against those positional bounds.
- Any validation ID whose numeric value happens to lie outside `[start, end)` (most of them) survives the filter and is still used for training; likewise, training IDs whose numeric values fall within `[start, end)` are wrongly excluded from training.

## Impact
- The model is effectively trained with access to most of the hold-out labels, causing target leakage that inflates discrimination and calibration metrics.
- Because the fold construction does not actually reduce the sample count, the P-IRLS refit and subsequent diagnostics are run on nearly the full dataset each time, significantly increasing runtime.
- Assertions about CV behaviour are therefore checking corrupted statistics and can pass or fail unpredictably, masking real regressions.

## Fix Implemented
- `train_idx` is now constructed by skipping entries whose **positions** fall in the validation range using `enumerate()` to keep the positional context while retaining the shuffled sample IDs.
- This change ensures that every validation observation is excluded from the training fold and vice versa, eliminating the leakage that previously corrupted the metrics and inflated runtimes.
