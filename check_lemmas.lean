import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.Submodule

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

lemma test_norm_sub_sq (x y : E) : ‖x - y‖^2 = ‖x‖^2 - 2 * inner x y + ‖y‖^2 := by
  rw [norm_sub_pow_two_real]
  sorry

lemma test_sq_le_sq (a b : ℝ) (h : 0 ≤ a) (h' : 0 ≤ b) : a^2 ≤ b^2 ↔ a ≤ b := by
  exact sq_le_sq
