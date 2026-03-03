import Mathlib

noncomputable def logRisk {Z : Type*} [MeasurableSpace Z] (μ : MeasureTheory.Measure Z) (p q : Z → ℝ) : ℝ := sorry

noncomputable def logBayesRisk {Z : Type*} [MeasurableSpace Z] (μ : MeasureTheory.Measure Z) (η : Z → ℝ) (F : Set (Z → ℝ)) : ℝ := sorry

axiom BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure
    {α : Type*} [TopologicalSpace α]
    (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α)
    (h_cont : Continuous R)
    (h_attain : ∃ a ∈ closure Fsmall, sInf (R '' Fsmall) = R a) :
    truth ∈ closure Fbig →
    truth ∉ closure Fsmall →
    BddBelow (R '' Fbig) →
    (R '' Fsmall).Nonempty →
    (∀ a, 0 ≤ R a - R truth) →
    (∀ a, a ∈ closure Fsmall → a ≠ truth → 0 < R a - R truth) →
    sInf (R '' Fbig) < sInf (R '' Fsmall)

theorem logBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (Z → ℝ)]
    (μ : MeasureTheory.Measure Z) (η : Z → ℝ) (Fbase Ffull : Set (Z → ℝ))
    (h_cont : Continuous (logRisk μ η))
    (h_attain : ∃ a ∈ closure Fbase, logBayesRisk μ η Fbase = logRisk μ η a) :
    η ∈ closure Ffull →
    η ∉ closure Fbase →
    BddBelow ((logRisk μ η) '' Ffull) →
    ((logRisk μ η) '' Fbase).Nonempty →
    (∀ q, 0 ≤ logRisk μ η q - logRisk μ η η) →
    (∀ q, q ∈ closure Fbase → q ≠ η → 0 < logRisk μ η q - logRisk μ η η) →
    logBayesRisk μ η Ffull < logBayesRisk μ η Fbase := by
  sorry
