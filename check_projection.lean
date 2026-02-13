import Mathlib.Analysis.InnerProductSpace.Projection.Basic
variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
variable (K : Submodule ℝ E) (y p : E)
#check Submodule.eq_orthogonalProjection_of_dist_le
