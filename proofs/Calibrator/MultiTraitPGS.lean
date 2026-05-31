import Calibrator.Probability
import Calibrator.PortabilityDrift
import Calibrator.OpenQuestions

namespace Calibrator

open MeasureTheory
open scoped BigOperators

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
structure AncestrySpecificRgModel where
  rg_eur : ℝ
  δ : ℝ
  rg_afr : ℝ
  h_afr_eq : rg_afr = rg_eur + δ
  h_delta_ne : δ ≠ 0

theorem rg_ancestry_specific
    (m : AncestrySpecificRgModel) :
    m.rg_eur ≠ m.rg_afr := by
  intro h
  have : m.δ = 0 := by linarith [m.h_afr_eq, h]
  exact m.h_delta_ne this

/-- **Equal-correlation PSD constraint.**
    For a 3×3 correlation matrix with equal pairwise correlation r,
    the determinant is (1-r)²(1+2r). PSD requires det ≥ 0, which
    (given |r| ≤ 1 so (1-r)² ≥ 0) is equivalent to 1 + 2r ≥ 0,
    i.e., r ≥ -1/2. This is a non-trivial constraint: not all
    pairwise-valid correlations give a valid correlation matrix. -/
structure EqualCorrMatrixModel where
  r : ℝ
  h_bound : |r| ≤ 1
  h_lower : -1/2 ≤ r

theorem rg_matrix_equal_corr_psd_constraint
    (m : EqualCorrMatrixModel) :
    0 ≤ (1 - m.r)^2 * (1 + 2 * m.r) := by
  apply mul_nonneg
  · exact sq_nonneg _
  · linarith [m.h_lower]

end GeneticCorrelation


/-!
## Multi-Trait BLUP

Multi-trait BLUP (MTBLUP) uses genetic correlations to
borrow information across traits, improving prediction
especially for low-heritability traits.
-/

section MultiTraitBLUP

/-- **MTBLUP prediction improvement.**
    For a target trait with h² and genetic correlation r_g
    with an auxiliary trait:
    R²_multi / R²_single ≈ 1 + r_g² × (n_aux / n_target) × (h²_aux / h²_target)
    when auxiliary GWAS is much larger. -/
structure MTBLUPImprovementModel where
  rg : ℝ
  n_aux : ℝ
  n_target : ℝ
  h2_aux : ℝ
  h2_target : ℝ
  h_rg : rg ≠ 0
  h_n_aux : 0 < n_aux
  h_n_target : 0 < n_target
  h_h2_aux : 0 < h2_aux
  h_h2_target : 0 < h2_target

noncomputable def mtblupImprovement (m : MTBLUPImprovementModel) : ℝ :=
  1 + m.rg^2 * (m.n_aux / m.n_target) * (m.h2_aux / m.h2_target)

/-- MTBLUP always improves over single-trait (when rg > 0 and aux is larger). -/
theorem mtblup_improves (m : MTBLUPImprovementModel) :
    1 < mtblupImprovement m := by
  unfold mtblupImprovement
  have h1 : 0 < m.rg^2 := sq_pos_of_ne_zero m.h_rg
  have h2 : 0 < m.n_aux / m.n_target := div_pos m.h_n_aux m.h_n_target
  have h3 : 0 < m.h2_aux / m.h2_target := div_pos m.h_h2_aux m.h_h2_target
  have h4 : 0 < m.rg^2 * (m.n_aux / m.n_target) := mul_pos h1 h2
  have h5 : 0 < m.rg^2 * (m.n_aux / m.n_target) * (m.h2_aux / m.h2_target) := mul_pos h4 h3
  linarith

/-- **MTBLUP portability.**
    The multi-trait improvement is less portable when cross-ancestry
    genetic correlation is lower. If r_g_cross < r_g_same, then
    MTBLUP improvement is smaller in the target population.
    Model: improvement ratio = 1 + r_g² × k where k = (n_aux/n_target)(h²_aux/h²_target).
    With r_g_cross < r_g_same, the cross-ancestry improvement is strictly smaller. -/
structure MTBLUPPortabilityModel where
  rg_same : ℝ
  rg_cross : ℝ
  k : ℝ
  h_rg_same_pos : 0 < rg_same
  h_rg_cross_pos : 0 < rg_cross
  h_rg_less : rg_cross < rg_same
  h_k_pos : 0 < k

theorem mtblup_portability_reduced
    (m : MTBLUPPortabilityModel) :
    1 + m.rg_cross^2 * m.k < 1 + m.rg_same^2 * m.k := by
  have h_sq : m.rg_cross^2 < m.rg_same^2 := by
    exact sq_lt_sq' (by linarith [m.h_rg_cross_pos, m.h_rg_less]) m.h_rg_less
  linarith [mul_lt_mul_of_pos_right h_sq m.h_k_pos]

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
structure PleiotropicCorrelationModel where
  port_A : ℝ
  port_B : ℝ
  rg : ℝ
  lb : ℝ
  h_correlated : |port_A - port_B| ≤ 2 * (1 - |rg|)
  h_rg : lb < |rg|
  h_lb_nn : 0 ≤ lb

theorem pleiotropic_correlated_portability
    (m : PleiotropicCorrelationModel) :
    |m.port_A - m.port_B| < 2 * (1 - m.lb) := by
  linarith [m.h_correlated, m.h_rg]

/-- **Mediated pleiotropy vs biological pleiotropy.**
    Mediated: A → B, so variant affects B through A.
    Portability of B is bounded by portability of A.
    If the mediation fraction is α ∈ [0,1], then
    port_B_mediated = α × port_A, so port_B ≤ port_A. -/


structure MediatedPleiotropyModel (m : ℕ) where
  β_A : Fin m → ℝ
  α : ℝ
  h_α_le : α ≤ 1
  h_α_nn : 0 ≤ α
  port_A : ℝ
  h_port_A_eq : port_A = ∑ i, β_A i ^ 2

theorem mediated_pleiotropy_portability_bound
    {m : ℕ} (model : MediatedPleiotropyModel m)
    (β_B : Fin m → ℝ)
    (h_mediated : β_B = fun i => model.α * model.β_A i) :
    (∑ i, β_B i ^ 2) ≤ model.port_A := by
  rw [h_mediated, model.h_port_A_eq]
  have h_bound : ∀ i, (model.α * model.β_A i) ^ 2 ≤ (model.β_A i) ^ 2 := by
    intro i
    have h1 : model.α ^ 2 ≤ 1 := by
      nlinarith [model.h_α_le, model.h_α_nn]
    calc
      (model.α * model.β_A i) ^ 2 = model.α ^ 2 * (model.β_A i) ^ 2 := by ring
      _ ≤ 1 * (model.β_A i) ^ 2 := by gcongr
      _ = (model.β_A i) ^ 2 := by ring
  exact Finset.sum_le_sum fun i _ => h_bound i

/-- **Trait-specific genetic components are less portable.**
    The component of genetic variance unique to a trait (not shared
    via pleiotropy) is more likely to be affected by population-specific
    selection. Shared components degrade by δ_shared, unique by δ_unique,
    where δ_unique > δ_shared due to selection. -/
structure ComponentPleiotropyModel (m : ℕ) where
  β_shared : Fin m → ℝ
  β_unique : Fin m → ℝ
  port_base : ℝ
  h_port_base : port_base = ∑ i, (β_shared i ^ 2 + β_unique i ^ 2)
  δ_shared : ℝ
  δ_unique : ℝ
  h_selection : δ_shared < δ_unique
  h_shared_nn : 0 < δ_shared

theorem unique_component_less_portable
    {m : ℕ} (model : ComponentPleiotropyModel m)
    (h_pos_unique : 0 < ∑ i, model.β_unique i ^ 2) :
    model.port_base - model.δ_unique * ∑ i, model.β_unique i ^ 2 <
    model.port_base - model.δ_shared * ∑ i, model.β_unique i ^ 2 := by
  have h_mul_lt : model.δ_shared * ∑ i, model.β_unique i ^ 2 < model.δ_unique * ∑ i, model.β_unique i ^ 2 := by
    exact mul_lt_mul_of_pos_right model.h_selection h_pos_unique
  linarith

/-- **Decomposing trait heritability into shared and unique.**
    h²_trait = h²_shared + h²_unique where h²_shared comes from
    pleiotropic loci. When the shared fraction dominates (h²_shared/h²_total > 1/2),
    portability is primarily determined by the shared component.
    Model: overall portability = (h²_shared × port_shared + h²_unique × port_unique) / h²_total.
    If h²_shared > h²_unique and port_shared > port_unique, then
    overall portability > (port_shared + port_unique) / 2 (the unweighted average). -/
structure HeritabilityPleiotropyModel where
  h2_shared : ℝ
  h2_unique : ℝ
  port_shared : ℝ
  port_unique : ℝ
  h_shared_pos : 0 < h2_shared
  h_unique_pos : 0 < h2_unique
  h_shared_larger : h2_unique < h2_shared
  h_port_shared_better : port_unique < port_shared
  h_ps_nn : 0 ≤ port_shared
  h_pu_nn : 0 ≤ port_unique

theorem heritability_shared_dominates_portability
    (m : HeritabilityPleiotropyModel) :
    (m.port_shared + m.port_unique) / 2 * (m.h2_shared + m.h2_unique) <
      m.h2_shared * m.port_shared + m.h2_unique * m.port_unique := by
  nlinarith [mul_pos (sub_pos.mpr m.h_shared_larger) (sub_pos.mpr m.h_port_shared_better)]

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
structure CrossAncestryRgModel where
  r2_source : ℝ
  r2_target : ℝ
  rg_cross : ℝ
  h_r2_s : 0 < r2_source
  h_bound : r2_target / r2_source ≤ rg_cross^2

theorem rg_bounds_portability_ratio
    (m : CrossAncestryRgModel) :
    m.r2_target ≤ m.rg_cross^2 * m.r2_source := by
  have h := m.h_bound
  rwa [div_le_iff₀ m.h_r2_s] at h

/-- **Traits with high cross-population r_g have good portability.**
    When r_g is high (e.g., ~0.95), R² portability is bounded by ~0.90. -/
structure HighCrossRgModel where
  rg : ℝ
  lb : ℝ
  h_rg : lb < rg
  h_lb_nn : 0 ≤ lb
  h_rg_le : rg ≤ 1

theorem high_cross_rg
    (m : HighCrossRgModel) :
    m.lb^2 < m.rg^2 := by
  nlinarith [m.h_rg, m.h_lb_nn, sq_nonneg (m.rg - m.lb)]

/-- **Traits with low cross-population r_g have poor portability.**
    When r_g is low (e.g., ~0.3), R² portability is bounded by ~0.09. -/
structure LowCrossRgModel where
  rg : ℝ
  ub : ℝ
  h_rg : rg ≤ ub
  h_rg_nn : 0 ≤ rg
  h_ub_nn : 0 ≤ ub

theorem low_cross_rg
    (m : LowCrossRgModel) :
    m.rg^2 ≤ m.ub^2 := by
  nlinarith [m.h_rg, m.h_rg_nn, sq_nonneg m.rg, sq_nonneg (m.rg - m.ub)]

/-- **r_g can be underestimated due to power.**
    With low power in underrepresented-population GWAS, r_g estimates are
    attenuated toward zero. Correction for attenuation:
    r_g_corrected = r_g_observed / √(h²_1 × h²_2). -/
structure CrossAncestryRgCorrectionModel where
  rg_obs : ℝ
  h2_1 : ℝ
  h2_2 : ℝ
  h_h2_1 : 0 < h2_1
  h_h2_1_le : h2_1 < 1
  h_h2_2 : 0 < h2_2
  h_h2_2_le : h2_2 < 1
  h_rg : 0 < rg_obs

theorem rg_attenuation_correction
    (m : CrossAncestryRgCorrectionModel) :
    m.rg_obs < m.rg_obs / Real.sqrt (m.h2_1 * m.h2_2) := by
  rw [lt_div_iff₀ (Real.sqrt_pos.mpr (mul_pos m.h_h2_1 m.h_h2_2))]
  have h_prod_lt : m.h2_1 * m.h2_2 < 1 := by nlinarith [m.h_h2_1_le, m.h_h2_2_le, m.h_h2_1, m.h_h2_2]
  have h_sqrt_lt : Real.sqrt (m.h2_1 * m.h2_2) < 1 := by
    rw [show (1 : ℝ) = Real.sqrt 1 from (Real.sqrt_one).symm]
    exact Real.sqrt_lt_sqrt (mul_nonneg (le_of_lt m.h_h2_1) (le_of_lt m.h_h2_2)) h_prod_lt
  nlinarith [m.h_rg]

end CrossAncestryRg

end Calibrator
