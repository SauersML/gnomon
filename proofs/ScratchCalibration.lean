import Mathlib.Tactic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Convex.Strict
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Analysis.Convex.SpecificFunctions.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Probability.Notation
import Mathlib.MeasureTheory.Integral.Bochner
import Mathlib.Analysis.Calculus.MeanValue

open MeasureTheory Real Set Filter

noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + Real.exp (-x))

lemma sigmoid_eq (x : ℝ) : sigmoid x = (1 + Real.exp (-x))⁻¹ := by
  unfold sigmoid
  rw [inv_eq_one_div]

lemma sigmoid_diff {x : ℝ} : DifferentiableAt ℝ sigmoid x := by
  unfold sigmoid
  apply DifferentiableAt.div
  · exact differentiableAt_const _
  · apply DifferentiableAt.add
    · exact differentiableAt_const _
    · apply DifferentiableAt.exp
      exact differentiableAt_neg x
  · apply ne_of_gt
    have : Real.exp (-x) > 0 := Real.exp_pos _
    linarith

lemma deriv_sigmoid (x : ℝ) : deriv sigmoid x = sigmoid x * (1 - sigmoid x) := by
  unfold sigmoid
  rw [deriv_div]
  · simp
    rw [deriv_add, deriv_const, zero_add, deriv_exp, deriv_neg, deriv_id]
    · simp
      set e := Real.exp (-x)
      have h : (1 + e) ≠ 0 := ne_of_gt (by have := Real.exp_pos (-x); linarith)
      field_simp
      ring
    · exact differentiableAt_const _
    · apply DifferentiableAt.exp; exact differentiableAt_neg _
    · exact differentiableAt_const _
    · apply DifferentiableAt.exp; exact differentiableAt_neg _
    · exact differentiableAt_neg _
  · exact differentiableAt_const _
  · apply DifferentiableAt.add
    · exact differentiableAt_const _
    · apply DifferentiableAt.exp; exact differentiableAt_neg _
  · apply ne_of_gt
    have : Real.exp (-x) > 0 := Real.exp_pos _
    linarith

lemma second_deriv_sigmoid (x : ℝ) : deriv (deriv sigmoid) x = sigmoid x * (1 - sigmoid x) * (1 - 2 * sigmoid x) := by
  conv => lhs; congr; ext y; rw [deriv_sigmoid y]
  rw [deriv_mul]
  · rw [deriv_sigmoid]
    rw [deriv_sub]
    · rw [deriv_const, deriv_sigmoid]
      ring
    · exact differentiableAt_const _
    · exact sigmoid_diff
  · exact sigmoid_diff
  · apply DifferentiableAt.sub
    · exact differentiableAt_const _
    · exact sigmoid_diff

lemma sigmoid_pos (x : ℝ) : 0 < sigmoid x := by
  unfold sigmoid
  apply div_pos one_pos
  have : Real.exp (-x) > 0 := Real.exp_pos _
  linarith

lemma sigmoid_lt_one (x : ℝ) : sigmoid x < 1 := by
  unfold sigmoid
  rw [div_lt_one]
  · have : Real.exp (-x) > 0 := Real.exp_pos _
    linarith
  · have : Real.exp (-x) > 0 := Real.exp_pos _
    linarith

lemma sigmoid_strict_concave_on_Ioi : StrictConcaveOn ℝ (Set.Ioi 0) sigmoid := by
  apply strictConcaveOn_of_deriv_deriv_neg
  · exact convex_Ioi 0
  · exact ContinuousOn.div continuousOn_const (ContinuousOn.add continuousOn_const (ContinuousOn.exp (ContinuousOn.neg continuousOn_id))) (fun x _ => ne_of_gt (by have := Real.exp_pos (-x); linarith))
  · intro x hx
    rw [second_deriv_sigmoid]
    have h_sig_pos : 0 < sigmoid x := sigmoid_pos x
    have h_sig_lt_one : sigmoid x < 1 := sigmoid_lt_one x
    have h_sig_gt_half : sigmoid x > 1/2 := by
      unfold sigmoid
      have h : Real.exp (-x) < 1 := by rw [Real.exp_lt_one_iff]; linarith
      have : 1 + Real.exp (-x) < 2 := by linarith
      apply one_div_lt_one_div_of_lt
      · norm_num
      · linarith
    have term1 : sigmoid x > 0 := h_sig_pos
    have term2 : 1 - sigmoid x > 0 := sub_pos.mpr h_sig_lt_one
    have term3 : 1 - 2 * sigmoid x < 0 := by linarith
    nlinarith

lemma sigmoid_strict_concave_on_Ici : StrictConcaveOn ℝ (Set.Ici 0) sigmoid := by
  apply StrictConcaveOn.of_deriv_deriv_neg'
  · exact convex_Ici 0
  · apply Continuous.continuousOn
    exact sigmoid_diff.continuous
  · intro x hx
    simp at hx
    rw [second_deriv_sigmoid]
    have h_sig_pos : 0 < sigmoid x := sigmoid_pos x
    have h_sig_lt_one : sigmoid x < 1 := sigmoid_lt_one x
    have h_sig_gt_half : sigmoid x > 1/2 := by
      unfold sigmoid
      have h : Real.exp (-x) < 1 := by rw [Real.exp_lt_one_iff]; linarith
      have : 1 + Real.exp (-x) < 2 := by linarith
      apply one_div_lt_one_div_of_lt
      · norm_num
      · linarith
    have term1 : sigmoid x > 0 := h_sig_pos
    have term2 : 1 - sigmoid x > 0 := sub_pos.mpr h_sig_lt_one
    have term3 : 1 - 2 * sigmoid x < 0 := by linarith
    nlinarith

variable {Ω : Type*} [MeasureSpace Ω] {P : Measure Ω} [IsProbabilityMeasure P]

theorem calibration_shrinkage (μ : ℝ) (hμ_pos : μ > 0)
    (X : Ω → ℝ)
    (h_measurable : Measurable X) (h_integrable : Integrable X P)
    (h_mean : ∫ ω, X ω ∂P = μ)
    (h_support : ∀ᵐ ω ∂P, X ω > 0)
    (h_non_degenerate : ¬ ∀ᵐ ω ∂P, X ω = μ) :
    (∫ ω, sigmoid (X ω) ∂P) < sigmoid μ := by
  have h_concave : StrictConcaveOn ℝ (Set.Ici 0) sigmoid := sigmoid_strict_concave_on_Ici
  have h_mem : ∀ᵐ ω ∂P, X ω ∈ Set.Ici 0 := by
    filter_upwards [h_support] with ω h
    exact le_of_lt h

  -- The signature of StrictConcaveOn.ae_eq_const_or_lt_map_average
  -- StrictConcaveOn.ae_eq_const_or_lt_map_average
  -- {E : Type u_1} {s : Set E} {f : E → ℝ} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E] {α : Type u_2} [MeasurableSpace α] {μ : Measure α} [IsFiniteMeasure μ]
  -- (hf : StrictConcaveOn ℝ s f) (hc : ContinuousOn f s) (hs : IsClosed s)
  -- (hmem : ∀ᵐ (x : α) ∂μ, g x ∈ s) (hg : Measurable g) (hgi : Integrable g μ)
  -- : (g =ᵐ[μ] fun x => average μ g) ∨ average μ (fun x => f (g x)) < f (average μ g)

  -- Note: average μ g = (μ univ).toReal⁻¹ • ∫ x, g x ∂μ. For prob measure, it's just the integral.
  have h_avg : average P X = μ := by
    rw [average_eq_integral]
    exact h_mean

  have h_res := StrictConcaveOn.ae_eq_const_or_lt_map_average h_concave
    (sigmoid_diff.continuous.continuousOn)
    (isClosed_Ici)
    h_mem h_measurable h_integrable

  rw [h_avg] at h_res
  cases h_res with
  | inl h_eq =>
      -- Contradiction with h_non_degenerate
      exfalso
      apply h_non_degenerate
      filter_upwards [h_eq] with ω h
      rw [h]
  | inr h_lt =>
      rw [average_eq_integral] at h_lt
      exact h_lt
