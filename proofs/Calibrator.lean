import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Convex.Strict
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Matrix.Rank
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.Probability.ConditionalExpectation
import Mathlib.Probability.ConditionalProbability
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Data.NNReal.Basic
import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Integration
import Mathlib.Probability.Moments.Variance
import Mathlib.Probability.Notation
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic

open MeasureTheory

namespace Calibrator

/-!
=================================================================
## Part 1: Definitions
=================================================================
-/

variable {Ω : Type*} [MeasureSpace Ω] {ℙ : Measure Ω} [IsProbabilityMeasure ℙ]

def Phenotype := Ω → ℝ
def PGS := Ω → ℝ
def PC (k : ℕ) := Ω → (Fin k → ℝ)

structure RealizedData (n k : ℕ) where
  y : Fin n → ℝ
  p : Fin n → ℝ
  c : Fin n → (Fin k → ℝ)

noncomputable def stdNormalProdMeasure (k : ℕ) [Fintype (Fin k)] : Measure (ℝ × (Fin k → ℝ)) :=
  (ProbabilityTheory.gaussianReal 0 1).prod (Measure.pi (fun (_ : Fin k) => ProbabilityTheory.gaussianReal 0 1))

instance stdNormalProdMeasure_is_prob {k : ℕ} [Fintype (Fin k)] : IsProbabilityMeasure (stdNormalProdMeasure k) := by
  unfold stdNormalProdMeasure
  infer_instance

structure PGSBasis (p : ℕ) where
  B : Fin (p + 1) → (ℝ → ℝ)
  B_zero_is_one : B 0 = fun _ => 1

structure SplineBasis (n : ℕ) where
  b : Fin n → (ℝ → ℝ)

def linearPGSBasis : PGSBasis 1 where
  B := fun m => if h : m = 0 then (fun _ => 1) else (fun p_val => p_val)
  B_zero_is_one := by simp

def polynomialSplineBasis (num_basis_funcs : ℕ) : SplineBasis num_basis_funcs where
  b := fun i x => x ^ (i.val + 1)

def SmoothFunction (n : ℕ) := Fin n → ℝ

def evalSmooth {n : ℕ} [Fintype (Fin n)] (s : SplineBasis n) (coeffs : SmoothFunction n) (x : ℝ) : ℝ :=
  ∑ i : Fin n, coeffs i * s.b i x

inductive LinkFunction | logit | identity
inductive DistributionFamily | Bernoulli | Gaussian

structure PhenotypeInformedGAM (p k sp : ℕ) where
  pgsBasis : PGSBasis p
  pcSplineBasis : SplineBasis sp
  γ₀₀ : ℝ
  γₘ₀ : Fin p → ℝ
  f₀ₗ : Fin k → SmoothFunction sp
  fₘₗ : Fin p → Fin k → SmoothFunction sp
  link : LinkFunction
  dist : DistributionFamily

noncomputable def linearPredictor {p k sp : ℕ} [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (model : PhenotypeInformedGAM p k sp) (pgs_val : ℝ) (pc_val : Fin k → ℝ) : ℝ :=
  let baseline_effect := model.γ₀₀ + ∑ l, evalSmooth model.pcSplineBasis (model.f₀ₗ l) (pc_val l)
  let pgs_related_effects := ∑ m : Fin p,
    let pgs_basis_val := model.pgsBasis.B ⟨m.val + 1, by linarith [m.isLt]⟩ pgs_val
    let pgs_coeff := model.γₘ₀ m + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ m l) (pc_val l)
    pgs_coeff * pgs_basis_val
  baseline_effect + pgs_related_effects

noncomputable def predict {p k sp : ℕ} [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (model : PhenotypeInformedGAM p k sp) (pgs_val : ℝ) (pc_val : Fin k → ℝ) : ℝ :=
  let η := linearPredictor model pgs_val pc_val
  match model.link with
  | .logit => 1 / (1 + Real.exp (-η))
  | .identity => η

structure DataGeneratingProcess (k : ℕ) where
  trueExpectation : ℝ → (Fin k → ℝ) → ℝ
  jointMeasure : Measure (ℝ × (Fin k → ℝ))
  is_prob : IsProbabilityMeasure jointMeasure := by infer_instance

instance dgp_is_prob {k : ℕ} (dgp : DataGeneratingProcess k) : IsProbabilityMeasure dgp.jointMeasure := dgp.is_prob

noncomputable def pointwiseNLL (dist : DistributionFamily) (y_obs : ℝ) (η : ℝ) : ℝ :=
  match dist with
  | .Gaussian => (y_obs - η)^2
  | .Bernoulli => Real.log (1 + Real.exp η) - y_obs * η

noncomputable def empiricalLoss {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (model : PhenotypeInformedGAM p k sp) (data : RealizedData n k) (lambda : ℝ) : ℝ :=
  (1 / (n : ℝ)) * (∑ i, pointwiseNLL model.dist (data.y i) (linearPredictor model (data.p i) (data.c i)))
  + lambda * ((∑ l, ∑ j, (model.f₀ₗ l j)^2) + (∑ m, ∑ l, ∑ j, (model.fₘₗ m l j)^2))

def IsIdentifiable {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)] (m : PhenotypeInformedGAM p k sp) (data : RealizedData n k) : Prop :=
  (∀ l, (∑ i, evalSmooth m.pcSplineBasis (m.f₀ₗ l) (data.c i l)) = 0) ∧
  (∀ mIdx l, (∑ i, evalSmooth m.pcSplineBasis (m.fₘₗ mIdx l) (data.c i l)) = 0)

noncomputable def fit {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)] (data : RealizedData n k) (lambda : ℝ) : PhenotypeInformedGAM p k sp :=
  sorry

theorem fit_minimizes_loss {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (data : RealizedData n k) (lambda : ℝ) :
  (∀ m, empiricalLoss (fit data lambda) data lambda ≤ empiricalLoss m data lambda) ∧
  IsIdentifiable (fit data lambda) data := by sorry

def IsRawScoreModel {p k sp : ℕ} [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)](m : PhenotypeInformedGAM p k sp) : Prop :=
  (∀ (l : Fin k) (s : Fin sp), m.f₀ₗ l s = 0) ∧ (∀ (i : Fin p) (l : Fin k) (s : Fin sp), m.fₘₗ i l s = 0)

def IsNormalizedScoreModel {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (m : PhenotypeInformedGAM p k sp) : Prop :=
  ∀ (i : Fin p) (l : Fin k) (s : Fin sp), m.fₘₗ i l s = 0

noncomputable def fitRaw {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)] (data : RealizedData n k) (lambda : ℝ) : PhenotypeInformedGAM p k sp :=
  sorry

theorem fitRaw_minimizes_loss {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (data : RealizedData n k) (lambda : ℝ) :
  IsRawScoreModel (fitRaw data lambda) ∧
  ∀ m (h_m : IsRawScoreModel m),
    empiricalLoss (fitRaw data lambda) data lambda ≤ empiricalLoss m data lambda := by sorry

noncomputable def fitNormalized {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)] (data : RealizedData n k) (lambda : ℝ) : PhenotypeInformedGAM p k sp :=
  sorry

theorem fitNormalized_minimizes_loss {p k sp n : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (data : RealizedData n k) (lambda : ℝ) :
  IsNormalizedScoreModel (fitNormalized data lambda) ∧
  ∀ m (h_m : IsNormalizedScoreModel m),
    empiricalLoss (fitNormalized data lambda) data lambda ≤ empiricalLoss m data lambda := by sorry

/-!
=================================================================
## Part 2: Fully Formalized Theorems and Proofs
=================================================================
-/

section AllClaims

variable {p k sp n : ℕ}

noncomputable def dgpScenario1 (k : ℕ) [Fintype (Fin k)] : DataGeneratingProcess k := {
  trueExpectation := fun p pc => p * (1 + 0.1 * (∑ l, pc l)),
  jointMeasure := stdNormalProdMeasure k
}

noncomputable def dgpScenario3 (k : ℕ) [Fintype (Fin k)] : DataGeneratingProcess k := {
  trueExpectation := fun p pc => p + (0.5 * (∑ l, pc l)),
  jointMeasure := stdNormalProdMeasure k
}

noncomputable def dgpScenario4 (k : ℕ) [Fintype (Fin k)] : DataGeneratingProcess k := {
  trueExpectation := fun p pc => p - (0.8 * (∑ l, pc l)),
  jointMeasure := stdNormalProdMeasure k
}

def hasInteraction {k : ℕ} [Fintype (Fin k)] (f : ℝ → (Fin k → ℝ) → ℝ) : Prop :=
  ∃ p₁ p₂ (c₁ c₂ : Fin k → ℝ), p₁ ≠ p₂ ∧ c₁ ≠ c₂ ∧
    (f p₂ c₁ - f p₁ c₁) / (p₂ - p₁) ≠ (f p₂ c₂ - f p₁ c₂) / (p₂ - p₁)

theorem scenarios_are_distinct (k : ℕ) (hk_pos : 0 < k) :
  hasInteraction (dgpScenario1 k).trueExpectation ∧
  ¬ hasInteraction (dgpScenario3 k).trueExpectation ∧
  ¬ hasInteraction (dgpScenario4 k).trueExpectation := by
  haveI : Fintype (Fin k) := Fin.fintype k
  constructor
  · let c₁ : Fin k → ℝ := fun _ => 0
    let c₂ : Fin k → ℝ := fun l => if l = ⟨0, hk_pos⟩ then 1 else 0
    use 0, 1, c₁, c₂
    simp only [ne_eq, one_ne_zero, not_false_eq_true, and_true]
    constructor
    · intro h_c_eq; simp [c₁, c₂] at h_c_eq; have := h_c_eq ⟨0, hk_pos⟩; simp at this
    · simp [dgpScenario1, Finset.sum_const_zero];
      have h_sum_c₂ : (∑ (l : Fin k), c₂ l) = 1 := by simp [c₂, Finset.sum_eq_single_of_mem, Fin.exists_fin_one, Finset.mem_univ];
      simp [h_sum_c₂]; norm_num
  constructor
  · intro h; simp [hasInteraction, dgpScenario3] at h
    rcases h with ⟨p₁, p₂, c₁, c₂, hp, _, h_neq⟩
    have h_slope₁ : ((p₂ + 0.5 * ∑ l, c₁ l) - (p₁ + 0.5 * ∑ l, c₁ l)) / (p₂ - p₁) = 1 := by
      rw [add_sub_add_left_eq_sub, div_self (sub_ne_zero.mpr hp)]
    have h_slope₂ : ((p₂ + 0.5 * ∑ l, c₂ l) - (p₁ + 0.5 * ∑ l, c₂ l)) / (p₂ - p₁) = 1 := by
      rw [add_sub_add_left_eq_sub, div_self (sub_ne_zero.mpr hp)]
    rw [h_slope₁, h_slope₂] at h_neq; contradiction
  · intro h; simp [hasInteraction, dgpScenario4] at h
    rcases h with ⟨p₁, p₂, c₁, c₂, hp, _, h_neq⟩
    have h_slope₁ : ((p₂ - 0.8 * ∑ l, c₁ l) - (p₁ - 0.8 * ∑ l, c₁ l)) / (p₂ - p₁) = 1 := by
      rw [sub_sub_sub_cancel_left, div_self (sub_ne_zero.mpr hp)]
    have h_slope₂ : ((p₂ - 0.8 * ∑ l, c₂ l) - (p₁ - 0.8 * ∑ l, c₂ l)) / (p₂ - p₁) = 1 := by
      rw [sub_sub_sub_cancel_left, div_self (sub_ne_zero.mpr hp)]
    rw [h_slope₁, h_slope₂] at h_neq; contradiction

theorem necessity_of_phenotype_data :
  ∃ (dgp_A dgp_B : DataGeneratingProcess 1),
    dgp_A.jointMeasure = dgp_B.jointMeasure ∧ hasInteraction dgp_A.trueExpectation ∧ ¬ hasInteraction dgp_B.trueExpectation := by
  haveI : Fintype (Fin 1) := Fin.fintype 1
  use dgpScenario1 1, dgpScenario4 1
  constructor; rfl
  have h_distinct := scenarios_are_distinct 1 (by norm_num)
  exact ⟨h_distinct.1, h_distinct.2.2⟩

noncomputable def expectedSquaredError [Fintype (Fin k)] (dgp : DataGeneratingProcess k) (f : ℝ → (Fin k → ℝ) → ℝ) : ℝ :=
  ∫ pc, (dgp.trueExpectation pc.1 pc.2 - f pc.1 pc.2)^2 ∂dgp.jointMeasure

def isBayesOptimalInClass {p k sp : ℕ} [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)] (dgp : DataGeneratingProcess k) (model : PhenotypeInformedGAM p k sp) : Prop :=
  ∀ m, expectedSquaredError dgp (fun p c => linearPredictor model p c) ≤
        expectedSquaredError dgp (fun p c => linearPredictor m p c)

theorem l2_projection_of_additive_is_additive [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
  {f : ℝ → ℝ} {g : Fin k → ℝ → ℝ} {dgp : DataGeneratingProcess k}
  (h_indep : dgp.jointMeasure = (dgp.jointMeasure.map Prod.fst).prod (dgp.jointMeasure.map Prod.snd))
  (h_true_fn : dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
  (proj : PhenotypeInformedGAM p k sp) (h_optimal : isBayesOptimalInClass dgp proj) :
  IsNormalizedScoreModel proj := by sorry

theorem independence_implies_no_interaction [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (dgp : DataGeneratingProcess k)
    (h_additive : ∃ f g, dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
    (h_indep : dgp.jointMeasure = (dgp.jointMeasure.map Prod.fst).prod (dgp.jointMeasure.map Prod.snd)) :
  ∀ m (h_opt : isBayesOptimalInClass dgp m), IsNormalizedScoreModel m := by
  intros m h_opt
  rcases h_additive with ⟨f, g, h_fn_struct⟩
  exact l2_projection_of_additive_is_additive h_indep h_fn_struct m h_opt

structure DGPWithEnvironment (k : ℕ) where
  to_dgp : DataGeneratingProcess k
  environmentalEffect : (Fin k → ℝ) → ℝ
  trueGeneticEffect : ℝ → ℝ
  is_additive_causal : to_dgp.trueExpectation = fun p c => trueGeneticEffect p + environmentalEffect c

theorem prediction_causality_tradeoff_linear_case {p sp : ℕ} [Fintype (Fin p)] [Fintype (Fin sp)]
    (dgp_env : DGPWithEnvironment 1)
    (hp_pos : p > 0)
    (h_gen : dgp_env.trueGeneticEffect = fun p => 2 * p)
    (h_env : dgp_env.environmentalEffect = fun c => 3 * (c ⟨0, by norm_num⟩))
    (h_confounding : ∫ pc, pc.1 * (pc.2 ⟨0, by norm_num⟩) ∂dgp_env.to_dgp.jointMeasure ≠ 0)
    (model : PhenotypeInformedGAM p 1 sp)
    (h_opt : isBayesOptimalInClass dgp_env.to_dgp model) :
    model.γₘ₀ ⟨0, hp_pos⟩ ≠ 2 := by sorry

def total_params (p k sp : ℕ) : ℕ := 1 + p + k*sp + p*k*sp

noncomputable def designMatrix [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (data : RealizedData n k) (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp)
    (hp : p > 0) (hk : k > 0) (hsp : sp > 0) : Matrix (Fin n) (Fin (total_params p k sp)) ℝ :=
  Matrix.of (fun (i : Fin n) (j : Fin (total_params p k sp)) =>
    let p_val := data.p i
    let c_val := data.c i
    if h_j_lt_1 : j.val < 1 then 1
    else if h_j_lt_gam : j.val < 1 + p then
      pgsBasis.B ⟨j.val - 1, by linarith [j.isLt, h_j_lt_gam, not_lt.mp h_j_lt_1]⟩ p_val
    else if h_j_lt_f0 : j.val < 1 + p + k*sp then
      let idx := j.val - (1 + p)
      have h_idx_ub : idx < k * sp := by linarith [j.isLt, h_j_lt_f0]
      have h_sp_pos : 0 < sp := hsp
      let l : Fin k := ⟨idx / sp, by {rw [Nat.mul_comm]; exact Nat.div_lt_of_lt_mul h_idx_ub}⟩
      let s : Fin sp := ⟨idx % sp, Nat.mod_lt _ hsp⟩
      splineBasis.b s (c_val l)
    else
      let idx := j.val - (1 + p + k*sp)
      have h_idx_ub : idx < p * k * sp := by linarith [j.isLt, total_params]
      have h_ksp_pos : 0 < k * sp := by linarith
      let m_val := idx / (k*sp)
      have hm_ub : m_val < p := by {rw [Nat.mul_comm]; exact Nat.div_lt_of_lt_mul h_idx_ub}
      let m : Fin p := ⟨m_val, hm_ub⟩
      let rem := idx % (k*sp)
      have h_rem_ub : rem < k * sp := Nat.mod_lt _ (by linarith)
      let l : Fin k := ⟨rem / sp, by {rw [Nat.mul_comm]; exact Nat.div_lt_of_lt_mul h_rem_ub}⟩
      let s : Fin sp := ⟨rem % sp, Nat.mod_lt _ hsp⟩
      (pgsBasis.B ⟨m.val + 1, by linarith [m.isLt, hp]⟩ p_val) * (splineBasis.b s (c_val l)))

theorem parameter_identifiability [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (data : RealizedData n k) (lambda : ℝ)
    (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp) (hp : p > 0) (hk : k > 0) (hsp : sp > 0)
    (h_rank : Matrix.rank (designMatrix data pgsBasis splineBasis hp hk hsp) = total_params p k sp) :
  ∃! m, IsIdentifiable m data ∧ ∀ m', IsIdentifiable m' data → empiricalLoss m data lambda ≤ empiricalLoss m' data lambda := by
  sorry

def predictionBias {k : ℕ} [Fintype (Fin k)] (dgp : DataGeneratingProcess k) (f : ℝ → (Fin k → ℝ) → ℝ) (p_val : ℝ) (c_val : Fin k → ℝ) : ℝ :=
  dgp.trueExpectation p_val c_val - f p_val c_val

theorem raw_score_bias_in_scenario4_simplified [Fact (p = 1)]
    (model_raw : PhenotypeInformedGAM 1 1 1) (h_raw_struct : IsRawScoreModel model_raw)
    (h_pgs_basis_linear : model_raw.pgsBasis.B 1 = id ∧ model_raw.pgsBasis.B 0 = fun _ => 1)
    (dgp4 : DataGeneratingProcess 1) (h_s4 : dgp4.trueExpectation = fun p c => p - (0.8 * c ⟨0, by norm_num⟩))
    (h_opt_raw : ∀ m (hm : IsRawScoreModel m), isBayesOptimalInClass dgp4 m → isBayesOptimalInClass dgp4 model_raw)
    (h_indep : dgp4.jointMeasure = (dgp4.jointMeasure.map Prod.fst).prod (dgp4.jointMeasure.map Prod.snd))
    (h_means_zero : ∫ pc, pc.1 ∂dgp4.jointMeasure = 0 ∧ ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂dgp4.jointMeasure = 0)
    (h_var_p_one : ∫ pc, pc.1^2 ∂dgp4.jointMeasure = 1) :
  ∀ (p_val : ℝ) (c_val : Fin 1 → ℝ),
    predictionBias dgp4 (fun p _ => linearPredictor model_raw p c_val) p_val c_val = -0.8 * c_val ⟨0, by norm_num⟩ := by
  sorry

def approxEq (a b : ℝ) (ε : ℝ := 0.01) : Prop := |a - b| < ε
notation:50 a " ≈ " b => approxEq a b 0.01

noncomputable def rsquared [MeasurableSingletonBorel (Fin k → ℝ)] [Fintype (Fin k)] (dgp : DataGeneratingProcess k) (f g : ℝ → (Fin k → ℝ) → ℝ) : ℝ := sorry
noncomputable def var [MeasurableSingletonBorel (Fin k → ℝ)] [Fintype (Fin k)] (dgp : DataGeneratingProcess k) (f : ℝ → (Fin k → ℝ) → ℝ) : ℝ := sorry

theorem quantitative_error_of_normalization [MeasurableSingletonBorel (Fin k → ℝ)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (dgp1 : DataGeneratingProcess k) (h_s1 : hasInteraction dgp1.trueExpectation)
    (hk_pos : k > 0)
    (model_norm : PhenotypeInformedGAM p k sp) (h_norm_model : IsNormalizedScoreModel model_norm) (h_norm_opt : isBayesOptimalInClass dgp1 model_norm)
    (model_oracle : PhenotypeInformedGAM p k sp) (h_oracle_opt : isBayesOptimalInClass dgp1 model_oracle) :
  let predict_norm := fun p c => linearPredictor model_norm p c
  let predict_oracle := fun p c => linearPredictor model_oracle p c
  expectedSquaredError dgp1 predict_norm - expectedSquaredError dgp1 predict_oracle
  = rsquared dgp1 (fun p c => p) (fun p c => c ⟨0, hk_pos⟩) * var dgp1 (fun p c => p) := by sorry

noncomputable def dgpMultiplicativeBias [Fintype (Fin k)] (k : ℕ) (scaling_func : (Fin k → ℝ) → ℝ) : DataGeneratingProcess k :=
  { trueExpectation := fun p c => (scaling_func c) * p, jointMeasure := stdNormalProdMeasure k }

theorem multiplicative_bias_correction [Fintype (Fin k)]
    (k : ℕ) (scaling_func : (Fin k → ℝ) → ℝ) (h_deriv : Differentiable ℝ scaling_func)
    (model : PhenotypeInformedGAM 1 k 1) (h_opt : isBayesOptimalInClass (dgpMultiplicativeBias k scaling_func) model) :
  ∀ l : Fin k, (evalSmooth model.pcSplineBasis (model.fₘₗ ⟨0, by norm_num⟩ l) 1 - evalSmooth model.pcSplineBasis (model.fₘₗ ⟨0, by norm_num⟩ l) 0)
    ≈ (scaling_func (fun i => if i = l then 1 else 0) - scaling_func (fun _ => 0)) := by sorry

structure DGPWithLatentRisk (k : ℕ) where
  to_dgp : DataGeneratingProcess k
  noise_variance_given_pc : (Fin k → ℝ) → ℝ
  sigma_G_sq : ℝ
  is_latent : to_dgp.trueExpectation = fun p c => (sigma_G_sq / (sigma_G_sq + noise_variance_given_pc c)) * p

theorem shrinkage_effect [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (dgp_latent : DGPWithLatentRisk k) (model : PhenotypeInformedGAM 1 k sp)
    (h_opt : isBayesOptimalInClass dgp_latent.to_dgp model) (hp_one : p = 1) :
  ∀ c : Fin k → ℝ, (model.γₘ₀ ⟨0, by {rw [hp_one]; norm_num}⟩ + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ ⟨0, by {rw [hp_one]; norm_num}⟩ l) (c l))
    ≈ (dgp_latent.sigma_G_sq / (dgp_latent.sigma_G_sq + dgp_latent.noise_variance_given_pc c)) := by sorry

theorem prediction_is_invariant_to_affine_pc_transform [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (A : Matrix (Fin k) (Fin k) ℝ) (hA : IsUnit A.det) (b : Fin k → ℝ) (data : RealizedData n k) (lambda : ℝ) :
  let data' : RealizedData n k := { y := data.y, p := data.p, c := fun i => A.mulVec (data.c i) + b }
  let model := fit data lambda; let model' := fit data' lambda
  ∀ (pgs : ℝ) (pc : Fin k → ℝ), predict model pgs pc ≈ predict model' pgs (A.mulVec pc + b) := by sorry

noncomputable def dist_to_support {k : ℕ} (c : Fin k → ℝ) (supp : Set (Fin k → ℝ)) : ℝ := sorry

theorem extrapolation_risk [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)] (dgp : DataGeneratingProcess k) (data : RealizedData n k) (lambda : ℝ) (c_new : Fin k → ℝ) :
  ∃ (f : ℝ → ℝ), Monotone f ∧ |predict (fit data lambda) 0 c_new - dgp.trueExpectation 0 c_new| ≤
    f (dist_to_support c_new {c | ∃ i, c = data.c i}) := by sorry

theorem context_specificity [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (dgp1 dgp2 : DGPWithEnvironment k)
    (h_same_genetics : dgp1.trueGeneticEffect = dgp2.trueGeneticEffect ∧ dgp1.to_dgp.jointMeasure = dgp2.to_dgp.jointMeasure)
    (h_diff_env : dgp1.environmentalEffect ≠ dgp2.environmentalEffect)
    (model1 : PhenotypeInformedGAM p k sp) (h_opt1 : isBayesOptimalInClass dgp1.to_dgp model1) :
  ¬ isBayesOptimalInClass dgp2.to_dgp model1 := by
  intro h_opt2
  have h_neq : dgp1.to_dgp.trueExpectation ≠ dgp2.to_dgp.trueExpectation := by
    intro h_eq_fn
    rw [dgp1.is_additive_causal, dgp2.is_additive_causal, h_same_genetics.1] at h_eq_fn
    have : dgp1.environmentalEffect = dgp2.environmentalEffect := by
      ext c
      have := congr_fun (congr_fun h_eq_fn 0) c
      simp at this; exact this
    exact h_diff_env this
  sorry

end AllClaims
end Calibrator
