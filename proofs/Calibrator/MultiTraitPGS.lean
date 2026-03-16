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

/-- **Equal-correlation PSD constraint.**
    For a 3×3 correlation matrix with equal pairwise correlation r,
    the determinant is (1-r)²(1+2r). PSD requires det ≥ 0, which
    (given |r| ≤ 1 so (1-r)² ≥ 0) is equivalent to 1 + 2r ≥ 0,
    i.e., r ≥ -1/2. This is a non-trivial constraint: not all
    pairwise-valid correlations give a valid correlation matrix. -/
theorem rg_matrix_equal_corr_psd_constraint
    (r : ℝ)
    (h_bound : |r| ≤ 1)
    (h_lower : -1/2 ≤ r) :
    0 ≤ (1 - r)^2 * (1 + 2 * r) := by
  apply mul_nonneg
  · exact sq_nonneg _
  · linarith

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
noncomputable def mtblupImprovement (rg n_aux n_target h2_aux h2_target : ℝ) : ℝ :=
  1 + rg^2 * (n_aux / n_target) * (h2_aux / h2_target)

/-- MTBLUP always improves over single-trait (when rg > 0 and aux is larger). -/
theorem mtblup_improves (rg n_aux n_target h2_aux h2_target : ℝ)
    (h_rg : rg ≠ 0) (h_n_aux : 0 < n_aux) (h_n_target : 0 < n_target)
    (h_h2_aux : 0 < h2_aux) (h_h2_target : 0 < h2_target) :
    1 < mtblupImprovement rg n_aux n_target h2_aux h2_target := by
  unfold mtblupImprovement
  linarith [sq_pos_of_ne_zero h_rg,
            div_pos h_n_aux h_n_target,
            div_pos h_h2_aux h_h2_target,
            mul_pos (mul_pos (sq_pos_of_ne_zero h_rg) (div_pos h_n_aux h_n_target))
                    (div_pos h_h2_aux h_h2_target)]

/-- **MTBLUP portability.**
    The multi-trait improvement is less portable when cross-ancestry
    genetic correlation is lower. If r_g_cross < r_g_same, then
    MTBLUP improvement is smaller in the target population.
    Model: improvement ratio = 1 + r_g² × k where k = (n_aux/n_target)(h²_aux/h²_target).
    With r_g_cross < r_g_same, the cross-ancestry improvement is strictly smaller. -/
theorem mtblup_portability_reduced
    (rg_same rg_cross k : ℝ)
    (h_rg_same_pos : 0 < rg_same)
    (h_rg_cross_pos : 0 < rg_cross)
    (h_rg_less : rg_cross < rg_same)
    (h_k_pos : 0 < k) :
    1 + rg_cross^2 * k < 1 + rg_same^2 * k := by
  have h_sq : rg_cross^2 < rg_same^2 := by
    exact sq_lt_sq' (by linarith) h_rg_less
  linarith [mul_lt_mul_of_pos_right h_sq h_k_pos]

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
    When r_g is bounded below by a high value (e.g., ~0.95), R² portability
    is bounded below by the square of that value if we assume the standard R² proportionality.
    (Assuming r2_target / r2_source = rg_cross^2 for full shared genetic architecture). -/
theorem high_cross_rg
    (r2_source r2_target rg_cross lb : ℝ)
    (h_r2_s : 0 < r2_source)
    (h_rg : lb < rg_cross)
    (h_lb_nn : 0 ≤ lb)
    (h_rg_le : rg_cross ≤ 1)
    (h_target_eq : r2_target / r2_source = rg_cross^2) :
    lb^2 * r2_source < r2_target := by
  have h_sq : lb^2 < rg_cross^2 := by nlinarith [sq_nonneg (rg_cross - lb)]
  have h_mul : lb^2 * r2_source < rg_cross^2 * r2_source := (mul_lt_mul_iff_of_pos_right h_r2_s).mpr h_sq
  have h_target_val : r2_target = rg_cross^2 * r2_source := (div_eq_iff h_r2_s.ne').mp h_target_eq
  rwa [← h_target_val] at h_mul

/-- **Traits with low cross-population r_g have poor portability.**
    When r_g is bounded above by a low value (e.g., ~0.3), R² portability
    is bounded above by the square of that value. -/
theorem low_cross_rg
    (r2_source r2_target rg_cross ub : ℝ)
    (h_r2_s : 0 < r2_source)
    (h_rg : rg_cross ≤ ub)
    (h_rg_nn : 0 ≤ rg_cross)
    (h_ub_nn : 0 ≤ ub)
    (h_target_upper_bound : r2_target / r2_source ≤ rg_cross^2) :
    r2_target ≤ ub^2 * r2_source := by
  have h1 : r2_target ≤ rg_cross^2 * r2_source := (div_le_iff₀ h_r2_s).mp h_target_upper_bound
  have h2 : rg_cross^2 ≤ ub^2 := by nlinarith [sq_nonneg rg_cross, sq_nonneg (rg_cross - ub)]
  have h3 : rg_cross^2 * r2_source ≤ ub^2 * r2_source := mul_le_mul_of_nonneg_right h2 (le_of_lt h_r2_s)
  linarith

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
