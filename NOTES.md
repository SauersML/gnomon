# Investigation: `test_save_load_functionality` Failure

## Failure Reproduction
- Command: `cargo test calibrate::model::tests::test_save_load_functionality`
- Panic message: `Mismatch: stored vs rebuilt interaction width for f(PGS,PC1) (stored 81, expected 49 = 7×7)`

## Observations
- The interaction coefficient block for `f(PGS,PC1)` is populated with `layout.penalty_map[layout.interaction_block_idx[0]].col_range.len()` coefficients.
- During matrix construction with `InteractionPenaltyKind::Anisotropic`, the *training* layout allocates **81** columns for that interaction block (9×9 tensor product) because it uses the **unwhitened** tensor-product basis dimensions (`pgs_main_basis_unc.ncols()` and `pc_unconstrained_bases_main[0].ncols()`).
- The saved `range_transforms` for `"pgs"` and `"PC1"` come from `basis::null_range_whiten`, which returns the **whitened range** subspace. Each has only **7** columns (penalized range dimension after removing the degree-2 polynomial null space).
- The loader rebuild path (`rebuild_layout_from_config`) currently multiplies those whitened widths (7×7=49) to reconstruct the interaction block, so the *reloaded* layout expects 49 columns and disagrees with both the stored coefficients (81) and the training layout.
- `assert_layout_consistency_with_layout` uses the same whitened multiplication, so it panics even though the serialized coefficients are internally consistent with the anisotropic training layout.
- The prediction path (`construct_design_matrix`) also multiplies the whitened transforms in both branches, so even bypassing the assertion leaves a 49-column interaction basis that no longer aligns with the stored anisotropic tensor coefficients.

## Root Cause
The consistency check assumes that tensor-product interactions always occupy `range_transforms["pgs"].ncols() × range_transforms[pc].ncols()` columns. That assumption holds for isotropic interactions (which operate in the whitened range space) but is false for anisotropic interactions, where the design uses the **full unwhitened** marginal bases instead. Consequently, the assertion compares incompatible dimensions (whitened vs. unwhitened bases) and fails even though the saved model is internally consistent.

