import Mathlib.Topology.Basic
import Mathlib.Topology.ContinuousOn
import Mathlib.Topology.MetricSpace.Basic
import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

namespace Calibrator

theorem covariance_mismatch_pos_of_fst_and_sparse_array_wf_proved
    {t : ℕ}
    (sigmaSource sigmaTarget : Matrix (Fin t) (Fin t) ℝ)
    (fstSource fstTarget recombRate arraySparsity kappa : ℝ)
    (h_cov_lb :
      demographicCovarianceGapLowerBound fstSource fstTarget recombRate arraySparsity kappa
        ≤ frobeniusNormSq (sigmaSource - sigmaTarget))
    (h_fst : fstSource < fstTarget)
    (h_recomb_pos : 0 < recombRate)
    (h_sparse_pos : 0 < arraySparsity)
    (h_kappa_pos : 0 < kappa) :
    0 < frobeniusNormSq (sigmaSource - sigmaTarget) := by
  exact covariance_mismatch_pos_of_fst_and_sparse_array
    sigmaSource sigmaTarget fstSource fstTarget recombRate arraySparsity kappa
    h_cov_lb h_fst h_recomb_pos h_sparse_pos h_kappa_pos

open Set

theorem BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proved
    {α : Type*} [TopologicalSpace α] (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α)
    (h_cont : Continuous R)
    (h_truth_in_big : truth ∈ closure Fbig)
    (h_truth_not_in_small : truth ∉ closure Fsmall)
    (h_strict_min : ∀ a ≠ truth, R truth < R a)
    (h_inf_attained : ∃ a ∈ closure Fsmall, BayesRisk R Fsmall = R a)
    (h_bdd_big : BddBelow (R '' Fbig)) :
    BayesRisk R Fbig < BayesRisk R Fsmall := by
  rcases h_inf_attained with ⟨a, ha, h_eq⟩
  have ha_neq : a ≠ truth := by
    intro h_eq_truth
    rw [h_eq_truth] at ha
    exact h_truth_not_in_small ha
  have h1 : R truth < R a := h_strict_min a ha_neq
  have h2 : BayesRisk R Fbig ≤ R truth := by
    unfold BayesRisk oracleRisk
    have h_mem : R truth ∈ closure (R '' Fbig) :=
      image_closure_subset_closure_image h_cont ⟨truth, h_truth_in_big, rfl⟩
    have h_subset : R '' Fbig ⊆ Ici (sInf (R '' Fbig)) := by
      intro y hy
      exact csInf_le h_bdd_big hy
    have h_closed : IsClosed (Ici (sInf (R '' Fbig))) := isClosed_Ici
    have h_closure_subset : closure (R '' Fbig) ⊆ Ici (sInf (R '' Fbig)) :=
      closure_minimal h_subset h_closed
    exact h_closure_subset h_mem
  rw [h_eq]
  exact lt_of_le_of_lt h2 h1

end Calibrator
