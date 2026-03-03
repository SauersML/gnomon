import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

open Real Set Topology Filter

theorem BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof
    {α : Type*} [TopologicalSpace α]
    (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α) :
    truth ∈ closure Fbig →
    truth ∉ closure Fsmall →
    BddBelow (R '' Fbig) →
    (R '' Fsmall).Nonempty →
    Continuous R →
    (∀ a, 0 ≤ R a - R truth) →
    (∀ a, a ∈ closure Fsmall → a ≠ truth → 0 < R a - R truth) →
    (∃ a ∈ closure Fsmall, BayesRisk R Fsmall = R a) →
    BayesRisk R Fbig < BayesRisk R Fsmall := by
  intro h1 h2 h3 _ h_cont _ h6 h8
  have ⟨a, ha_mem, ha_eq⟩ := h8
  have a_ne : a ≠ truth := fun h_eq => h2 (h_eq ▸ ha_mem)
  have ha_gt : 0 < R a - R truth := h6 a ha_mem a_ne
  have R_truth_lt_R_a : R truth < R a := sub_pos.mp ha_gt
  have truth_R_mem : R truth ∈ closure (R '' Fbig) := by
    have h_sub := image_closure_subset_closure_image h_cont (s := Fbig)
    exact h_sub (Set.mem_image_of_mem R h1)
  have le_truth : BayesRisk R Fbig ≤ R truth := by
    have h_Ici : IsClosed (Set.Ici (BayesRisk R Fbig)) := isClosed_Ici
    have h_sub : R '' Fbig ⊆ Set.Ici (BayesRisk R Fbig) := fun y hy => csInf_le h3 hy
    have h_closure_sub : closure (R '' Fbig) ⊆ Set.Ici (BayesRisk R Fbig) := closure_minimal h_sub h_Ici
    exact h_closure_sub truth_R_mem
  rw [ha_eq]
  exact lt_of_le_of_lt le_truth R_truth_lt_R_a

theorem logBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proof
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : MeasureTheory.Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z)) :
    η ∈ closure Ffull →
    η ∉ closure Fbase →
    BddBelow ((logRisk μ η) '' Ffull) →
    ((logRisk μ η) '' Fbase).Nonempty →
    Continuous (logRisk μ η) →
    (∀ q, 0 ≤ logRisk μ η q - logRisk μ η η) →
    (∀ q, q ∈ closure Fbase → q ≠ η → 0 < logRisk μ η q - logRisk μ η η) →
    (∃ a ∈ closure Fbase, BayesRisk (logRisk μ η) Fbase = logRisk μ η a) →
    logBayesRisk μ η Ffull < logBayesRisk μ η Fbase := by
  intro h1 h2 h3 h4 h_cont h5 h6 h8
  exact BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof (logRisk μ η) η Fbase Ffull h1 h2 h3 h4 h_cont h5 h6 h8

theorem brierBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proof
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : MeasureTheory.Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z)) :
    η ∈ closure Ffull →
    η ∉ closure Fbase →
    BddBelow ((brierRisk μ η) '' Ffull) →
    ((brierRisk μ η) '' Fbase).Nonempty →
    Continuous (brierRisk μ η) →
    (∀ q, 0 ≤ brierRisk μ η q - brierRisk μ η η) →
    (∀ q, q ∈ closure Fbase → q ≠ η → 0 < brierRisk μ η q - brierRisk μ η η) →
    (∃ a ∈ closure Fbase, BayesRisk (brierRisk μ η) Fbase = brierRisk μ η a) →
    brierBayesRisk μ η Ffull < brierBayesRisk μ η Fbase := by
  intro h1 h2 h3 h4 h_cont h5 h6 h8
  exact BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof (brierRisk μ η) η Fbase Ffull h1 h2 h3 h4 h_cont h5 h6 h8

end Calibrator
