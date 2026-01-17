import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Convex.Strict
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Probability.Notation
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.MeasureTheory.Integral.Jensen
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Comp

open scoped InnerProductSpace
open MeasureTheory

noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

lemma sigmoid_deriv (x : ℝ) :
  HasDerivAt sigmoid (sigmoid x * (1 - sigmoid x)) x := by
  unfold sigmoid
  set f := fun x => 1 + Real.exp (-x)
  have hf : ∀ x, f x ≠ 0 := by
    intro x
    have : Real.exp (-x) > 0 := Real.exp_pos _
    linarith
  have hf_deriv : HasDerivAt f (-Real.exp (-x)) x := by
    apply HasDerivAt.add_const
    apply HasDerivAt.exp_neg
    apply hasDerivAt_id
  apply HasDerivAt.inv (hf x) hf_deriv
  -- The derivative of 1/f is -f'/f^2
  -- = -(-exp(-x)) / (1+exp(-x))^2 = exp(-x) / (1+exp(-x))^2
  -- sigmoid * (1-sigmoid) = (1/(1+e)) * (1 - 1/(1+e)) = (1/(1+e)) * (e/(1+e)) = e/(1+e)^2
  -- So they match.
  -- Prove the equality:
  rw [inv_eq_one_div]
  field_simp
  ring

lemma sigmoid_deriv2 (x : ℝ) :
  HasDerivAt (fun x => sigmoid x * (1 - sigmoid x)) (sigmoid x * (1 - sigmoid x) * (1 - 2 * sigmoid x)) x := by
  apply HasDerivAt.mul
  · exact sigmoid_deriv x
  · apply HasDerivAt.const_sub
    exact sigmoid_deriv x
  · ring_nf

lemma sigmoid_two_deriv_neg_on_pos (x : ℝ) (hx : 0 < x) :
  sigmoid x * (1 - sigmoid x) * (1 - 2 * sigmoid x) < 0 := by
  have h1 : 0 < sigmoid x := by
    unfold sigmoid
    have : 0 < Real.exp (-x) := Real.exp_pos _
    nlinarith
  have h2 : sigmoid x < 1 := by
    unfold sigmoid
    have : 0 < Real.exp (-x) := Real.exp_pos _
    rw [div_lt_one] <;> linarith
  have h3 : 1 / 2 < sigmoid x := by
    unfold sigmoid
    have : Real.exp (-x) < 1 := by rw [Real.exp_lt_one_iff]; linarith
    have : 1 + Real.exp (-x) < 2 := by linarith
    have : 0 < 1 + Real.exp (-x) := by have := Real.exp_pos (-x); linarith
    rw [lt_div_iff this, one_mul]
    linarith
  have term1 : 0 < sigmoid x := h1
  have term2 : 0 < 1 - sigmoid x := sub_pos.mpr h2
  have term3 : 1 - 2 * sigmoid x < 0 := by linarith
  nlinarith

lemma sigmoid_strictConcaveOn_Ici : StrictConcaveOn ℝ (Set.Ici 0) sigmoid := by
  apply strictConcaveOn_of_deriv2_neg' (convex_Ici 0) (continuousOn_id.sigmoid)
  intro x hx
  -- Need strict inequality on interior? Set.Ici interior is Set.Ioi.
  -- strictConcaveOn_of_deriv2_neg' usually takes interior.
  -- Let's check mathlib's API.
  sorry

