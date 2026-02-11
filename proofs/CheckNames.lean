import Mathlib.Analysis.InnerProductSpace.Projection.Submodule
open InnerProductSpace
open Submodule

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
variable (K : Submodule ℝ E) [K.HasOrthogonalProjection] (u : E)

#check norm_sub_sq_eq_norm_sq_add_norm_sq_sub_two_inner
#check Submodule.orthogonalProjection K u
example : Submodule.orthogonalProjection K u = K.starProjection u := rfl
