import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

theorem BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proved
    {α : Type*} [TopologicalSpace α]
    (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α)
    (h_cont : Continuous R)
    (h_attain : ∃ a ∈ closure Fsmall, BayesRisk R Fsmall = R a)
    (h_truth_big : truth ∈ closure Fbig)
    (h_truth_not_small : truth ∉ closure Fsmall)
    (h_bdd_big : BddBelow (R '' Fbig))
    (h_strict_min : ∀ a, a ∈ closure Fsmall → a ≠ truth → 0 < R a - R truth) :
    BayesRisk R Fbig < BayesRisk R Fsmall := by
  rcases h_attain with ⟨a, ha_closure, ha_eq⟩
  have ha_neq_truth : a ≠ truth := by
    intro h_eq
    rw [h_eq] at ha_closure
    exact h_truth_not_small ha_closure
  have h_strict : 0 < R a - R truth := h_strict_min a ha_closure ha_neq_truth
  have h_Ra_gt_Rtruth : R truth < R a := sub_pos.mp h_strict
  rw [ha_eq]
  have h_inf_le_truth : BayesRisk R Fbig ≤ R truth := by
    unfold BayesRisk oracleRisk
    apply le_of_forall_gt
    intro c hc
    have h_truth_lt_c : R truth < c := hc
    have h_open : IsOpen {x | R x < c} := h_cont.isOpen_preimage (Set.Iio c) isOpen_Iio
    have h_truth_mem : truth ∈ {x | R x < c} := h_truth_lt_c
    have h_inter_nonempty := mem_closure_iff.mp h_truth_big {x | R x < c} h_open h_truth_mem
    rcases h_inter_nonempty with ⟨y, hy_mem_preimage, hy_mem_fbig⟩
    have hy_lt_c : R y < c := hy_mem_preimage
    have h_inf_le_y : sInf (R '' Fbig) ≤ R y := csInf_le h_bdd_big (Set.mem_image_of_mem R hy_mem_fbig)
    exact h_inf_le_y.trans_lt hy_lt_c
  exact h_inf_le_truth.trans_lt h_Ra_gt_Rtruth

theorem logBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proved
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : MeasureTheory.Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z))
    (h_cont : Continuous (logRisk μ η))
    (h_attain : ∃ a ∈ closure Fbase, logBayesRisk μ η Fbase = logRisk μ η a)
    (h_truth_big : η ∈ closure Ffull)
    (h_truth_not_small : η ∉ closure Fbase)
    (h_bdd_big : BddBelow ((logRisk μ η) '' Ffull))
    (h_strict_min : ∀ q, q ∈ closure Fbase → q ≠ η → 0 < logRisk μ η q - logRisk μ η η) :
    logBayesRisk μ η Ffull < logBayesRisk μ η Fbase := by
  apply BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proved (logRisk μ η) η Fbase Ffull
  · exact h_cont
  · exact h_attain
  · exact h_truth_big
  · exact h_truth_not_small
  · exact h_bdd_big
  · exact h_strict_min

theorem brierBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proved
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : MeasureTheory.Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z))
    (h_cont : Continuous (brierRisk μ η))
    (h_attain : ∃ a ∈ closure Fbase, brierBayesRisk μ η Fbase = brierRisk μ η a)
    (h_truth_big : η ∈ closure Ffull)
    (h_truth_not_small : η ∉ closure Fbase)
    (h_bdd_big : BddBelow ((brierRisk μ η) '' Ffull))
    (h_strict_min : ∀ q, q ∈ closure Fbase → q ≠ η → 0 < brierRisk μ η q - brierRisk μ η η) :
    brierBayesRisk μ η Ffull < brierBayesRisk μ η Fbase := by
  apply BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proved (brierRisk μ η) η Fbase Ffull
  · exact h_cont
  · exact h_attain
  · exact h_truth_big
  · exact h_truth_not_small
  · exact h_bdd_big
  · exact h_strict_min

end Calibrator
