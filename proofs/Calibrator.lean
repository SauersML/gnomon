import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator
open MeasureTheory

theorem BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof
    {α : Type*} [TopologicalSpace α]
    (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α)
    (h_cont : Continuous R)
    (h_inf_attained : ∃ a ∈ closure Fsmall, BayesRisk R Fsmall = R a)
    (h_truth_in_big : truth ∈ closure Fbig)
    (h_truth_not_in_small : truth ∉ closure Fsmall)
    (h_bdd_big : BddBelow (R '' Fbig))
    (_h_nonempty_small : (R '' Fsmall).Nonempty)
    (_h_nonneg : ∀ a, 0 ≤ R a - R truth)
    (h_strict : ∀ a, a ∈ closure Fsmall → a ≠ truth → 0 < R a - R truth) :
    BayesRisk R Fbig < BayesRisk R Fsmall := by
  rcases h_inf_attained with ⟨a, ha_mem, ha_eq⟩
  have h_a_neq : a ≠ truth := by
    intro h_eq
    subst h_eq
    exact h_truth_not_in_small ha_mem
  have h_strict_pos : 0 < R a - R truth := h_strict a ha_mem h_a_neq
  have h_R_a_gt : R truth < R a := by linarith
  have h_R_truth_le_inf_big : BayesRisk R Fbig ≤ R truth := by
    unfold BayesRisk oracleRisk
    apply le_of_forall_gt
    intro c hc
    have h_open : IsOpen {x | R x < c} := h_cont.isOpen_preimage _ isOpen_Iio
    have h_inter := (mem_closure_iff.mp h_truth_in_big) {x | R x < c} h_open hc
    rcases h_inter with ⟨b, hb_mem, hb_open⟩
    have h_inf_le : sInf (R '' Fbig) ≤ R b := csInf_le h_bdd_big ⟨b, hb_open, rfl⟩
    exact lt_of_le_of_lt h_inf_le hb_mem
  calc
    BayesRisk R Fbig ≤ R truth := h_R_truth_le_inf_big
    _ < R a := h_R_a_gt
    _ = BayesRisk R Fsmall := ha_eq.symm

theorem logBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proof
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z))
    (h_cont : Continuous (logRisk μ η))
    (h_inf_attained : ∃ a ∈ closure Fbase, logBayesRisk μ η Fbase = logRisk μ η a)
    (h_eta_in_full : η ∈ closure Ffull)
    (h_eta_not_in_base : η ∉ closure Fbase)
    (h_bdd_full : BddBelow ((logRisk μ η) '' Ffull))
    (h_nonempty_base : ((logRisk μ η) '' Fbase).Nonempty)
    (h_nonneg : ∀ q, 0 ≤ logRisk μ η q - logRisk μ η η)
    (h_strict : ∀ q, q ∈ closure Fbase → q ≠ η → 0 < logRisk μ η q - logRisk μ η η) :
    logBayesRisk μ η Ffull < logBayesRisk μ η Fbase := by
  exact BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof (logRisk μ η) η Fbase Ffull h_cont h_inf_attained h_eta_in_full h_eta_not_in_base h_bdd_full h_nonempty_base h_nonneg h_strict

theorem brierBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proof
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z))
    (h_cont : Continuous (brierRisk μ η))
    (h_inf_attained : ∃ a ∈ closure Fbase, brierBayesRisk μ η Fbase = brierRisk μ η a)
    (h_eta_in_full : η ∈ closure Ffull)
    (h_eta_not_in_base : η ∉ closure Fbase)
    (h_bdd_full : BddBelow ((brierRisk μ η) '' Ffull))
    (h_nonempty_base : ((brierRisk μ η) '' Fbase).Nonempty)
    (h_nonneg : ∀ q, 0 ≤ brierRisk μ η q - brierRisk μ η η)
    (h_strict : ∀ q, q ∈ closure Fbase → q ≠ η → 0 < brierRisk μ η q - brierRisk μ η η) :
    brierBayesRisk μ η Ffull < brierBayesRisk μ η Fbase := by
  exact BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof (brierRisk μ η) η Fbase Ffull h_cont h_inf_attained h_eta_in_full h_eta_not_in_base h_bdd_full h_nonempty_base h_nonneg h_strict

end Calibrator
