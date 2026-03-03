import Calibrator.Probability
import Calibrator.DGP
import Calibrator.Models
import Calibrator.Conclusions
import Calibrator.PortabilityDrift

open Filter Topology Set

namespace Calibrator

theorem BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof
    {α : Type*} [TopologicalSpace α]
    (R : α → ℝ) (truth : α) (Fsmall Fbig : Set α)
    (h_big : truth ∈ closure Fbig)
    (h_small : truth ∉ closure Fsmall)
    (h_bdd : BddBelow (R '' Fbig))
    (h_ne : (R '' Fsmall).Nonempty)
    (_h_nonneg : ∀ a, 0 ≤ R a - R truth)
    (h_strict : ∀ a, a ∈ closure Fsmall → a ≠ truth → 0 < R a - R truth)
    (h_cont : Continuous R)
    (h_compact : IsCompact (closure Fsmall)) :
    BayesRisk R Fbig < BayesRisk R Fsmall := by
  unfold BayesRisk oracleRisk
  have hz : ∀ z ∈ closure Fsmall, R truth < R z := fun z hz => by
    have hnz : z ≠ truth := by
      intro h_eq
      rw [h_eq] at hz
      exact h_small hz
    have h_strict_z := h_strict z hz hnz
    exact sub_pos.mp h_strict_z

  have h_ne_clos : (closure Fsmall).Nonempty := by
    rcases h_ne with ⟨yr, hyr⟩
    rcases hyr with ⟨y, hy, _⟩
    use y
    exact subset_closure hy

  have hm : ∃ m ∈ closure Fsmall, ∀ z ∈ closure Fsmall, R m ≤ R z := by
    rcases IsCompact.exists_isMinOn h_compact h_ne_clos h_cont.continuousOn with ⟨m, hm_clos, hm_min⟩
    use m, hm_clos
    exact hm_min

  rcases hm with ⟨m, hm_clos, hm_min⟩
  have h_min_strict : R truth < R m := hz m hm_clos

  have ht_inf : sInf (R '' Fbig) ≤ R truth := by
    have h_mem : R truth ∈ closure (R '' Fbig) := mem_closure_image (h_cont.continuousAt) h_big
    have hd : ∀ x ∈ closure (R '' Fbig), sInf (R '' Fbig) ≤ x := by
      intro b hb
      have is_clos : IsClosed { x | sInf (R '' Fbig) ≤ x } := isClosed_Ici
      have h_sub : R '' Fbig ⊆ { x | sInf (R '' Fbig) ≤ x } := fun x hx => csInf_le h_bdd hx
      exact is_clos.closure_subset_iff.mpr h_sub hb
    exact hd (R truth) h_mem

  have ht_small_inf : R m ≤ sInf (R '' Fsmall) := by
    have hd2 : R m ∈ lowerBounds (R '' Fsmall) := by
      intro y hy
      rcases hy with ⟨x, hx, hxy⟩
      rw [←hxy]
      exact hm_min x (subset_closure hx)
    exact le_csInf h_ne hd2

  calc sInf (R '' Fbig)
    _ ≤ R truth := ht_inf
    _ < R m := h_min_strict
    _ ≤ sInf (R '' Fsmall) := ht_small_inf

theorem logBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proof
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : MeasureTheory.Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z))
    (h_full : η ∈ closure Ffull)
    (h_base : η ∉ closure Fbase)
    (h_bdd : BddBelow ((logRisk μ η) '' Ffull))
    (h_ne : ((logRisk μ η) '' Fbase).Nonempty)
    (h_nonneg : ∀ q, 0 ≤ logRisk μ η q - logRisk μ η η)
    (h_strict : ∀ q, q ∈ closure Fbase → q ≠ η → 0 < logRisk μ η q - logRisk μ η η)
    (h_cont : Continuous (logRisk μ η))
    (h_compact : IsCompact (closure Fbase)) :
    logBayesRisk μ η Ffull < logBayesRisk μ η Fbase := by
  unfold logBayesRisk
  exact BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof
    (logRisk μ η) η Fbase Ffull h_full h_base h_bdd h_ne h_nonneg h_strict h_cont h_compact

theorem brierBayesRisk_strict_of_eta_in_closure_not_in_baseline_closure_proof
    {Z : Type*} [MeasurableSpace Z] [TopologicalSpace (ProbPredictor Z)]
    (μ : MeasureTheory.Measure Z) (η : ProbPredictor Z) (Fbase Ffull : Set (ProbPredictor Z))
    (h_full : η ∈ closure Ffull)
    (h_base : η ∉ closure Fbase)
    (h_bdd : BddBelow ((brierRisk μ η) '' Ffull))
    (h_ne : ((brierRisk μ η) '' Fbase).Nonempty)
    (h_nonneg : ∀ q, 0 ≤ brierRisk μ η q - brierRisk μ η η)
    (h_strict : ∀ q, q ∈ closure Fbase → q ≠ η → 0 < brierRisk μ η q - brierRisk μ η η)
    (h_cont : Continuous (brierRisk μ η))
    (h_compact : IsCompact (closure Fbase)) :
    brierBayesRisk μ η Ffull < brierBayesRisk μ η Fbase := by
  unfold brierBayesRisk
  exact BayesRisk_strict_of_truth_in_closure_not_in_baseline_closure_proof
    (brierRisk μ η) η Fbase Ffull h_full h_base h_bdd h_ne h_nonneg h_strict h_cont h_compact

end Calibrator
