import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Set

lemma test_sig (f : ℝ → ℝ) (s : Set ℝ) (h : StrictConcaveOn ℝ s f) : True := trivial

theorem test_concave : True := by
  have : StrictConcaveOn ℝ (Ioi 0) (fun x => -x^2) := by
    apply strictConcaveOn_of_deriv2_neg
    sorry
  trivial
