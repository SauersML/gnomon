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
  rcases h_attain with ⟨a, ha_closure, ha_eq⟩
  have ha_neq_truth : a ≠ truth := by
    intro h_eq
    rw [h_eq] at ha_closure
    exact h_truth_not_small ha_closure
  have h_strict : 0 < R a - R truth := h_strict_min a ha_closure ha_neq_truth
  have h_Ra_gt_Rtruth : R truth < R a := sub_pos.mp h_strict
  rw [ha_eq]
  have h_min_val : ∀ b ∈ Fbig, R truth ≤ R b := by
    intro b _
    exact sub_nonneg.mp (h_min b)
  have h_inf_le_truth : BayesRisk R Fbig ≤ R truth := by
    unfold BayesRisk oracleRisk
    apply le_of_forall_lt'
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
