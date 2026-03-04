import Mathlib.Topology.Basic
import Mathlib.Topology.Instances.Real
import Calibrator.Conclusions

open Set Calibrator

theorem BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure
    {α : Type*} [TopologicalSpace α] (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α)
    (h_cont : Continuous R)
    (h_truth_in_big : truth ∈ closure Fbig)
    (h_truth_not_in_small : truth ∉ closure Fsmall)
    (h_strict_min : ∀ a ≠ truth, R truth < R a)
    (h_inf_attained : ∃ a ∈ closure Fsmall, BayesRisk R Fsmall = R a)
    (h_bdd_big : BddBelow (R '' Fbig)) :
    BayesRisk R Fbig < BayesRisk R Fsmall := by
  -- We know BayesRisk R Fbig ≤ R truth.
  -- R is continuous, so R '' (closure Fbig) ⊆ closure (R '' Fbig)
  -- wait, csInf (R '' Fbig) is the BayesRisk.
  sorry
