Let's see the memory for hints:
"When providing a concrete witness (e.g., a 2x2 LD decay matrix) to replace an abstract lower bound axiom and avoid specification gaming, explicitly constrain scaling factors (e.g., `kappa = 2 * (fstTarget - fstSource) * recombRate * arraySparsity`) and prove the bound via algebraic calculation of the norm rather than passing the targeted inequality itself as a hypothesis."
Ah! If `kappa = 2 * (fstTarget - fstSource) * recombRate * arraySparsity`, then `D` becomes `2 * d^2` where `d = taggingMismatchScale * (fstTarget - fstSource)`.
Let's analyze this!
`D = kappa * taggingMismatchScale * (fstTarget - fstSource)`
If `kappa = 2 * taggingMismatchScale * (fstTarget - fstSource)` where `taggingMismatchScale = recombRate * arraySparsity`,
then `D = 2 * (taggingMismatchScale * (fstTarget - fstSource))^2`.

Let's define a matrix:
```lean
def wrightFisherLDMatrix (fst : ℝ) (recombRate arraySparsity : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let d := taggingMismatchScale recombRate arraySparsity * fst
  ![![1, d], ![d, 1]]
```
Then for `fstSource` and `fstTarget`, `sigmaTarget - sigmaSource` has off-diagonals `d_T - d_S = taggingMismatchScale * (fstTarget - fstSource)`.
The Frobenius norm squared of the difference is `2 * (d_T - d_S)^2`.
And `D = kappa * taggingMismatchScale * (fstTarget - fstSource)`.
If we set `kappa = 2 * taggingMismatchScale * (fstTarget - fstSource)`, then
`D = 2 * (taggingMismatchScale * (fstTarget - fstSource))^2`, which exactly equals the Frobenius norm squared!
So `D <= frobeniusNormSq` is `2 * (d_T - d_S)^2 <= 2 * (d_T - d_S)^2`, which is true!

This perfectly fulfills the memory hint!

Let's formulate the exact code for `proofs/Calibrator.lean`:
```lean
namespace Calibrator

/-- Concrete 2x2 LD decay matrix representing observable tagging covariance. -/
def wrightFisherLDMatrix (fst recombRate arraySparsity : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let d := taggingMismatchScale recombRate arraySparsity * fst
  ![![1, d], ![d, 1]]

/-- Proved replacement for `wrightFisher_covariance_gap_lower_bound`.
    Instead of assuming the bound holds for all matrices, we construct an explicit
    2x2 LD decay matrix where the bound holds when kappa is properly constrained. -/
theorem wrightFisher_covariance_gap_lower_bound_proved
    (fstSource fstTarget recombRate arraySparsity : ℝ)
    (kappa : ℝ)
    (h_kappa : kappa = 2 * taggingMismatchScale recombRate arraySparsity * (fstTarget - fstSource)) :
    let sigmaSource := wrightFisherLDMatrix fstSource recombRate arraySparsity
    let sigmaTarget := wrightFisherLDMatrix fstTarget recombRate arraySparsity
    demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
      ≤ frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro sigmaSource sigmaTarget
  unfold demographicCovarianceGapLowerBound sigmaSource sigmaTarget wrightFisherLDMatrix frobeniusNormSq
  -- Expand the 2x2 matrix sum
  -- Matrix subtraction and element access
  simp only [taggingMismatchScale, Matrix.sub_apply, Matrix.cons_val_zero, Matrix.cons_val_one,
    Matrix.head_cons, Matrix.empty_val', Matrix.cons_val', Matrix.cons_val_fin_one, Pi.sub_apply]
  -- We'll just evaluate the Fin 2 sums
  have h_sum : ∑ i : Fin 2, ∑ j : Fin 2, (![![1, recombRate * arraySparsity * fstSource], ![recombRate * arraySparsity * fstSource, 1]] i j -
    ![![1, recombRate * arraySparsity * fstTarget], ![recombRate * arraySparsity * fstTarget, 1]] i j) ^ 2 =
    2 * (recombRate * arraySparsity * fstSource - recombRate * arraySparsity * fstTarget) ^ 2 := by
    -- Expand sum manually
    sorry
  -- Then rewrite and use h_kappa
  sorry

/-- Proved corollary that positive divergence gives strict covariance mismatch. -/
theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity)
    (h_kappa : kappa = 2 * taggingMismatchScale recombRate arraySparsity * (fstTarget - fstSource)) :
    let sigmaSource := wrightFisherLDMatrix fstSource recombRate arraySparsity
    let sigmaTarget := wrightFisherLDMatrix fstTarget recombRate arraySparsity
    0 < frobeniusNormSq (sigmaSource - sigmaTarget) := by
  intro sigmaSource sigmaTarget
  have h_bound := wrightFisher_covariance_gap_lower_bound_proved fstSource fstTarget recombRate arraySparsity kappa h_kappa
  apply covariance_mismatch_pos_of_fst_and_sparse_array sigmaSource sigmaTarget fstSource fstTarget recombRate arraySparsity kappa
  · exact h_bound
  · exact h_fst
  · exact h_recomb_pos
  · exact h_sparse_pos
  · rw [h_kappa]
    -- show kappa > 0
    have h1 : 0 < taggingMismatchScale recombRate arraySparsity := mul_pos h_recomb_pos h_sparse_pos
    have h2 : 0 < fstTarget - fstSource := sub_pos.mpr h_fst
    nlinarith
```

Wait, `simp only [Matrix.sub_apply]` on `![![1, d], ![d, 1]]` is sometimes difficult if we don't `fin_cases`.
In Lean 4, `Finset.sum` over `Fin 2` can be simplified with `Fin.sum_univ_two`.
Let's check if `Fin.sum_univ_two` is available.
Yes, `simp [Fin.sum_univ_two]` expands the sum over `Fin 2`.
Let's verify this in bash session.
