import Mathlib

noncomputable def oracleRisk {α : Type*} (R : α → ℝ) (F : Set α) : ℝ :=
  sInf (R '' F)

noncomputable def BayesRisk {α : Type*} (R : α → ℝ) (F : Set α) : ℝ :=
  oracleRisk R F

theorem BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure
    {α : Type*} [TopologicalSpace α]
    (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α)
    (h_cont : Continuous R)
    (h_attain : ∃ a ∈ closure Fsmall, BayesRisk R Fsmall = R a)
    (h_truth_big : truth ∈ closure Fbig)
    (h_truth_not_small : truth ∉ closure Fsmall)
    (h_bdd_big : BddBelow (R '' Fbig))
    (h_nonempty_small : (R '' Fsmall).Nonempty)
    (h_min : ∀ a, 0 ≤ R a - R truth)
    (h_strict_min : ∀ a, a ∈ closure Fsmall → a ≠ truth → 0 < R a - R truth) :
    BayesRisk R Fbig < BayesRisk R Fsmall := by
  sorry
