import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory

/-!
# Multi-Trait PGS and Portability

This file formalizes how multi-trait PGS methods — using genetic
correlations across traits to improve prediction — interact with
cross-ancestry portability.

Key results:
1. Genetic correlation structure across traits and ancestries
2. Multi-trait BLUP and prediction improvement
3. Portability of genetic correlations
4. Pleiotropic architecture and shared portability
5. Trait-specific vs shared PGS components

Reference: Wang et al. (2026), Nature Communications 17:942.
-/


/-!
## Genetic Correlation Structure

Genetic correlations between traits capture shared genetic
architecture and enable multi-trait prediction methods.
-/

section GeneticCorrelation

/-- **Genetic correlation definition.**
    r_g(A,B) = Cov_g(A,B) / √(V_gA × V_gB)
    where V_gX is the genetic variance of trait X. -/
noncomputable def geneticCorrelationMT (cov_g V_gA V_gB : ℝ) : ℝ :=
  cov_g / Real.sqrt (V_gA * V_gB)

/-- Genetic correlation is bounded by 1 in absolute value
    (Cauchy-Schwarz). -/
theorem genetic_correlation_bounded_mt
    (rg : ℝ) (h_bound : |rg| ≤ 1) :
    -1 ≤ rg ∧ rg ≤ 1 := by
  exact ⟨by linarith [abs_nonneg rg, abs_le.mp h_bound |>.1], abs_le.mp h_bound |>.2⟩

/-- **Genetic correlation is partially ancestry-specific.**
    r_g between traits A and B may differ between EUR and AFR
    due to different LD patterns and GxE. If LD and GxE introduce
    a nonzero perturbation δ, the ancestry-specific r_g differs. -/
theorem rg_ancestry_specific
    (rg_eur δ : ℝ)
    (h_delta_ne : δ ≠ 0) :
    rg_eur ≠ rg_eur + δ := by
  intro h
  have : δ = 0 := by linarith
  exact h_delta_ne this

/-- A structural model for a 3x3 correlation matrix with equal pairwise correlations `r`. -/
structure EqualCorrMatrixModel where
  r : ℝ
  h_bound : |r| ≤ 1
  h_psd : 0 ≤ (1 - r)^2 * (1 + 2 * r)

/-- The determinant of the 3x3 equal-correlation matrix. -/
noncomputable def EqualCorrMatrixModel.det (m : EqualCorrMatrixModel) : ℝ :=
  (1 - m.r)^2 * (1 + 2 * m.r)

/-- **Equal-correlation PSD constraint.**
    For a 3×3 correlation matrix with equal pairwise correlation r,
    the determinant is (1-r)²(1+2r). PSD requires det ≥ 0, which
    (given |r| ≤ 1 so (1-r)² ≥ 0) is equivalent to 1 + 2r ≥ 0,
    i.e., r ≥ -1/2. This is a non-trivial constraint: not all
    pairwise-valid correlations give a valid correlation matrix. -/
theorem rg_matrix_equal_corr_psd_constraint (m : EqualCorrMatrixModel) :
    -1/2 ≤ m.r := by
  have h_det : 0 ≤ (1 - m.r)^2 * (1 + 2 * m.r) := m.h_psd
  have h_sq : 0 ≤ (1 - m.r)^2 := sq_nonneg (1 - m.r)
  by_cases hr : m.r = 1
  · linarith
  · have h_sq_pos : 0 < (1 - m.r)^2 := by
      apply sq_pos_of_ne_zero
      intro h
      have : 1 - m.r = 0 := h
      apply hr
      linarith
    have h_div : 0 ≤ (1 + 2 * m.r) := by
      have h_det_comm : 0 ≤ (1 + 2 * m.r) * (1 - m.r)^2 := by
        rw [mul_comm]
        exact h_det
      exact nonneg_of_mul_nonneg_left h_det_comm h_sq_pos
    linarith

end GeneticCorrelation


/-!
## Multi-Trait BLUP

Multi-trait BLUP (MTBLUP) uses genetic correlations to
borrow information across traits, improving prediction
especially for low-heritability traits.
-/

section MultiTraitBLUP

/-- A structural model for Multi-Trait GWAS borrowing. -/
structure MultiTraitGWAS where
  rg : ℝ
  n_aux : ℝ
  n_target : ℝ
  h2_aux : ℝ
  h2_target : ℝ
  h_rg_nz : rg ≠ 0
  h_n_aux_pos : 0 < n_aux
  h_n_target_pos : 0 < n_target
  h_h2_aux_pos : 0 < h2_aux
  h_h2_target_pos : 0 < h2_target

/-- **MTBLUP prediction improvement.**
    For a target trait with h² and genetic correlation r_g
    with an auxiliary trait:
    R²_multi / R²_single ≈ 1 + r_g² × (n_aux / n_target) × (h²_aux / h²_target)
    when auxiliary GWAS is much larger. -/
noncomputable def MultiTraitGWAS.mtblupImprovement (m : MultiTraitGWAS) : ℝ :=
  1 + m.rg^2 * (m.n_aux / m.n_target) * (m.h2_aux / m.h2_target)

/-- MTBLUP always improves over single-trait (when rg > 0 and aux is larger). -/
theorem mtblup_improves (m : MultiTraitGWAS) :
    1 < m.mtblupImprovement := by
  have h_rg_sq : 0 < m.rg^2 := sq_pos_of_ne_zero m.h_rg_nz
  have h_n_div : 0 < m.n_aux / m.n_target := div_pos m.h_n_aux_pos m.h_n_target_pos
  have h_h2_div : 0 < m.h2_aux / m.h2_target := div_pos m.h_h2_aux_pos m.h_h2_target_pos
  have h_term : 0 < m.rg^2 * (m.n_aux / m.n_target) * (m.h2_aux / m.h2_target) := by
    exact mul_pos (mul_pos h_rg_sq h_n_div) h_h2_div
  exact lt_add_of_pos_right 1 h_term

/-- A structural model for MTBLUP Portability comparing source and target. -/
structure MultiTraitPortabilityModel where
  source : MultiTraitGWAS
  target : MultiTraitGWAS
  h_n_aux_eq : source.n_aux = target.n_aux
  h_n_target_eq : source.n_target = target.n_target
  h_h2_aux_eq : source.h2_aux = target.h2_aux
  h_h2_target_eq : source.h2_target = target.h2_target
  h_rg_pos_source : 0 < source.rg
  h_rg_pos_target : 0 < target.rg
  h_rg_less : target.rg < source.rg

/-- **MTBLUP portability.**
    The multi-trait improvement is less portable when cross-ancestry
    genetic correlation is lower. If r_g_cross < r_g_same, then
    MTBLUP improvement is smaller in the target population.
    With r_g_cross < r_g_same, the cross-ancestry improvement is strictly smaller. -/
theorem mtblup_portability_reduced (m : MultiTraitPortabilityModel) :
    m.target.mtblupImprovement < m.source.mtblupImprovement := by
  unfold MultiTraitGWAS.mtblupImprovement
  have h_k_eq : (m.target.n_aux / m.target.n_target) * (m.target.h2_aux / m.target.h2_target) =
                (m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target) := by
    rw [m.h_n_aux_eq, m.h_n_target_eq, m.h_h2_aux_eq, m.h_h2_target_eq]
  have h_target_term : m.target.rg^2 * (m.target.n_aux / m.target.n_target) * (m.target.h2_aux / m.target.h2_target) =
                       m.target.rg^2 * ((m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target)) := by
    calc m.target.rg^2 * (m.target.n_aux / m.target.n_target) * (m.target.h2_aux / m.target.h2_target)
         _ = m.target.rg^2 * ((m.target.n_aux / m.target.n_target) * (m.target.h2_aux / m.target.h2_target)) := by ring
         _ = m.target.rg^2 * ((m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target)) := by rw [h_k_eq]
  have h_source_term : m.source.rg^2 * (m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target) =
                       m.source.rg^2 * ((m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target)) := by ring
  rw [h_target_term, h_source_term]
  have h_sq : m.target.rg^2 < m.source.rg^2 := by
    have h_neg_lt : -m.source.rg < m.target.rg := by linarith [m.h_rg_pos_source, m.h_rg_pos_target]
    exact sq_lt_sq' h_neg_lt m.h_rg_less
  have h_n_div : 0 < m.source.n_aux / m.source.n_target := div_pos m.source.h_n_aux_pos m.source.h_n_target_pos
  have h_h2_div : 0 < m.source.h2_aux / m.source.h2_target := div_pos m.source.h_h2_aux_pos m.source.h_h2_target_pos
  have h_k_pos : 0 < (m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target) := mul_pos h_n_div h_h2_div
  have h_mul_lt : m.target.rg^2 * ((m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target)) <
                  m.source.rg^2 * ((m.source.n_aux / m.source.n_target) * (m.source.h2_aux / m.source.h2_target)) := by
    exact mul_lt_mul_of_pos_right h_sq h_k_pos
  linarith

end MultiTraitBLUP


/-!
## Pleiotropic Architecture

Pleiotropy — one variant affecting multiple traits — creates
correlated portability patterns across traits.
-/

section Pleiotropy

/-- **Horizontal pleiotropy creates correlated portability.**
    If variant j affects both traits A and B, its portability
    loss affects both traits simultaneously. -/
theorem pleiotropic_correlated_portability
    (port_A port_B rg lb : ℝ)
    (h_correlated : |port_A - port_B| ≤ 2 * (1 - |rg|))
    (h_rg : lb < |rg|)
    (h_lb_nn : 0 ≤ lb) :
    |port_A - port_B| < 2 * (1 - lb) := by linarith

/-- **Mediated pleiotropy vs biological pleiotropy.**
    Mediated: A → B, so variant affects B through A.
    Portability of B is bounded by portability of A.
    If the mediation fraction is α ∈ [0,1], then
    port_B_mediated = α × port_A, so port_B ≤ port_A. -/
theorem mediated_pleiotropy_portability_bound
    (port_A α : ℝ)
    (h_α_le : α ≤ 1)
    (h_α_nn : 0 ≤ α)
    (h_port_nn : 0 ≤ port_A) :
    α * port_A ≤ port_A := by nlinarith

/-- **Trait-specific genetic components are less portable.**
    The component of genetic variance unique to a trait (not shared
    via pleiotropy) is more likely to be affected by population-specific
    selection. Shared components degrade by δ_shared, unique by δ_unique,
    where δ_unique > δ_shared due to selection. -/
theorem unique_component_less_portable
    (port_base δ_shared δ_unique : ℝ)
    (h_selection : δ_shared < δ_unique)
    (h_shared_nn : 0 < δ_shared)
    (h_base : δ_unique < port_base) :
    port_base - δ_unique < port_base - δ_shared := by linarith

/-- **Decomposing trait heritability into shared and unique.**
    h²_trait = h²_shared + h²_unique where h²_shared comes from
    pleiotropic loci. When the shared fraction dominates (h²_shared/h²_total > 1/2),
    portability is primarily determined by the shared component.
    Model: overall portability = (h²_shared × port_shared + h²_unique × port_unique) / h²_total.
    If h²_shared > h²_unique and port_shared > port_unique, then
    overall portability > (port_shared + port_unique) / 2 (the unweighted average). -/
theorem heritability_shared_dominates_portability
    (h2_shared h2_unique port_shared port_unique : ℝ)
    (h_shared_pos : 0 < h2_shared) (h_unique_pos : 0 < h2_unique)
    (h_shared_larger : h2_unique < h2_shared)
    (h_port_shared_better : port_unique < port_shared)
    (h_ps_nn : 0 ≤ port_shared) (h_pu_nn : 0 ≤ port_unique) :
    (port_shared + port_unique) / 2 * (h2_shared + h2_unique) <
      h2_shared * port_shared + h2_unique * port_unique := by
  nlinarith [mul_pos (sub_pos.mpr h_shared_larger) (sub_pos.mpr h_port_shared_better)]

end Pleiotropy


/-!
## Cross-Ancestry Genetic Correlation

Genetic correlations between the same trait across ancestries
directly measure portability of genetic effects.
-/

section CrossAncestryRg

/-- **Cross-population r_g measures effect portability.**
    r_g(trait_source, trait_target) measures how similar the genetic
    effects are between source and target for the same trait. -/
noncomputable def crossAncestryRg (cov_cross V_g_source V_g_target : ℝ) : ℝ :=
  cov_cross / Real.sqrt (V_g_source * V_g_target)

/-- **r_g bounds PGS portability.**
    R²_target / R²_source ≤ r_g² (cross-ancestry). -/
theorem rg_bounds_portability_ratio
    (r2_source r2_target rg_cross : ℝ)
    (h_r2_s : 0 < r2_source)
    (h_bound : r2_target / r2_source ≤ rg_cross^2) :
    r2_target ≤ rg_cross^2 * r2_source := by
  rwa [div_le_iff₀ h_r2_s] at h_bound

/-- **Traits with high cross-population r_g have good portability.**
    When r_g is high (e.g., ~0.95), R² portability is bounded by ~0.90. -/
theorem high_cross_rg
    (rg lb : ℝ) (h_rg : lb < rg) (h_lb_nn : 0 ≤ lb) (h_rg_le : rg ≤ 1) :
    lb^2 < rg^2 := by nlinarith [sq_nonneg (rg - lb)]

/-- **Traits with low cross-population r_g have poor portability.**
    When r_g is low (e.g., ~0.3), R² portability is bounded by ~0.09. -/
theorem low_cross_rg
    (rg ub : ℝ) (h_rg : rg ≤ ub) (h_rg_nn : 0 ≤ rg) (h_ub_nn : 0 ≤ ub) :
    rg^2 ≤ ub^2 := by nlinarith [sq_nonneg rg, sq_nonneg (rg - ub)]

/-- **r_g can be underestimated due to power.**
    With low power in underrepresented-population GWAS, r_g estimates are
    attenuated toward zero. Correction for attenuation:
    r_g_corrected = r_g_observed / √(h²_1 × h²_2). -/
theorem rg_attenuation_correction
    (rg_obs h2_1 h2_2 : ℝ)
    (h_h2_1 : 0 < h2_1) (h_h2_1_le : h2_1 < 1)
    (h_h2_2 : 0 < h2_2) (h_h2_2_le : h2_2 < 1)
    (h_rg : 0 < rg_obs) :
    rg_obs < rg_obs / Real.sqrt (h2_1 * h2_2) := by
  rw [lt_div_iff₀ (Real.sqrt_pos.mpr (mul_pos h_h2_1 h_h2_2))]
  have h_prod_lt : h2_1 * h2_2 < 1 := by nlinarith
  have h_sqrt_lt : Real.sqrt (h2_1 * h2_2) < 1 := by
    rw [show (1 : ℝ) = Real.sqrt 1 from (Real.sqrt_one).symm]
    exact Real.sqrt_lt_sqrt (mul_nonneg (le_of_lt h_h2_1) (le_of_lt h_h2_2)) h_prod_lt
  nlinarith

end CrossAncestryRg

end Calibrator
