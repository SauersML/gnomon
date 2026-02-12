import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2

open scoped InnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
variable {K : Submodule ℝ E} [CompleteSpace K]

example (y p : E) (h_mem : p ∈ K) (h_min : ∀ w ∈ K, dist y p ≤ dist y w) :
  p = K.subtype (orthogonalProjection K y) := by
  -- try library_search or exact?
  -- But I can't run tactics interactively.
  -- I'll check known lemma names.
  apply eq_orthogonalProjection_of_mem_of_minimize_dist
  sorry
