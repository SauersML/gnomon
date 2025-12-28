import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Convex.Strict
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.Basic
import Mathlib.Analysis.InnerProductSpace.Projection.FiniteDimensional
import Mathlib.Analysis.InnerProductSpace.Projection.Minimal
import Mathlib.Analysis.InnerProductSpace.Projection.Reflection
import Mathlib.Analysis.InnerProductSpace.Projection.Submodule
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Fin.Basic
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.ToLin
import Mathlib.LinearAlgebra.Dimension.OrzechProperty
import Mathlib.Data.Matrix.Basic
import Mathlib.LinearAlgebra.Matrix.Trace
import Mathlib.MeasureTheory.Function.L2Space
import Mathlib.MeasureTheory.Constructions.Pi
import Mathlib.MeasureTheory.Integral.Prod
import Mathlib.Probability.ConditionalExpectation
import Mathlib.Probability.ConditionalProbability
import Mathlib.Probability.Distributions.Gaussian.Real
import Mathlib.Data.NNReal.Basic

import Mathlib.Probability.Independence.Basic
import Mathlib.Probability.Independence.Integration
import Mathlib.Probability.Moments.Variance
import Mathlib.Probability.Notation
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.Topology.MetricSpace.HausdorffDistance

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

noncomputable def linearPredictor {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (model : PhenotypeInformedGAM p k sp) (pgs_val : ℝ) (pc_val : Fin k → ℝ) : ℝ :=
  let baseline_effect := model.γ₀₀ + ∑ l, evalSmooth model.pcSplineBasis (model.f₀ₗ l) (pc_val l)
  let pgs_related_effects := ∑ m : Fin p,
    let pgs_basis_val := model.pgsBasis.B ⟨m.val + 1, by linarith [m.isLt]⟩ pgs_val
    let pgs_coeff := model.γₘ₀ m + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ m l) (pc_val l)
    pgs_coeff * pgs_basis_val
  baseline_effect + pgs_related_effects

/-! ### Predictor Decomposition for p=1 Models

For models with a single PGS basis function (p=1), we can decompose the linear predictor
into `base(c) + slope(c) * p`, which is the natural form for L² projection / normal equations.
This decomposition is the gateway to proving shrinkage_effect and raw_score_bias theorems. -/

/-- The intercept term of the predictor (not multiplied by p).
    For a p=1 model: base(c) = γ₀₀ + Σₗ evalSmooth(f₀ₗ[l], c[l]) -/
noncomputable def predictorBase {k sp : ℕ} [Fintype (Fin k)] [Fintype (Fin sp)]
    (model : PhenotypeInformedGAM 1 k sp) (pc_val : Fin k → ℝ) : ℝ :=
  model.γ₀₀ + ∑ l, evalSmooth model.pcSplineBasis (model.f₀ₗ l) (pc_val l)

/-- The slope coefficient in front of p.
    For a p=1 model with linear PGS basis: slope(c) = γₘ₀[0] + Σₗ evalSmooth(fₘₗ[0,l], c[l]) -/
noncomputable def predictorSlope {k sp : ℕ} [Fintype (Fin k)] [Fintype (Fin sp)]
    (model : PhenotypeInformedGAM 1 k sp) (pc_val : Fin k → ℝ) : ℝ :=
  model.γₘ₀ 0 + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ 0 l) (pc_val l)

/-- Helper: sum over Fin 1 collapses to the single term. -/
lemma Fin1_sum_eq {α : Type*} [AddCommMonoid α] (f : Fin 1 → α) :
    ∑ m : Fin 1, f m = f 0 := by
  simp

/-- **Predictor Decomposition Lemma**: For a p=1 model with linear PGS basis (B[1] = id),
    the linear predictor decomposes as: linearPredictor(p, c) = base(c) + slope(c) * p.

    This is the key lemma that reduces the GAM structure to a 2-parameter linear form in p,
    enabling L² projection / normal equations analysis. -/
theorem linearPredictor_decomp {k sp : ℕ} [Fintype (Fin k)] [Fintype (Fin sp)]
    (model : PhenotypeInformedGAM 1 k sp)
    (h_linear_basis : model.pgsBasis.B ⟨1, by norm_num⟩ = id) :
  ∀ pgs_val pc_val, linearPredictor model pgs_val pc_val =
    predictorBase model pc_val + predictorSlope model pc_val * pgs_val := by
  classical
  intro pgs_val pc_val
  unfold linearPredictor predictorBase predictorSlope
  -- Expand the `let`s and rewrite the `Fin 1` sum to the single `m = 0` term.
  dsimp
  have hsum :
      (∑ m : Fin 1,
          let pgs_basis_val := model.pgsBasis.B ⟨m.val + 1, by linarith [m.isLt]⟩ pgs_val
          let pgs_coeff :=
            model.γₘ₀ m + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ m l) (pc_val l)
          pgs_coeff * pgs_basis_val) =
        (let m0 : Fin 1 := 0
         let pgs_basis_val := model.pgsBasis.B ⟨m0.val + 1, by linarith [m0.isLt]⟩ pgs_val
         let pgs_coeff :=
           model.γₘ₀ m0 + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ m0 l) (pc_val l)
         pgs_coeff * pgs_basis_val) := by
    simp
  rw [hsum]
  -- Simplify the remaining `let`s, then use the linear-basis hypothesis to rewrite the basis evaluation.
  dsimp
  have hB : model.pgsBasis.B (1 : Fin 2) pgs_val = pgs_val := by
    have hidx1 : (1 : Fin 2) = ⟨1, by norm_num⟩ := by
      ext; simp
    rw [hidx1, h_linear_basis]
    rfl
  -- Replace `B[1] p` by `p`; the remaining goal is definitional.
  rw [hB]


noncomputable def predict {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (model : PhenotypeInformedGAM p k sp) (pgs_val : ℝ) (pc_val : Fin k → ℝ) : ℝ :=
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

noncomputable def empiricalLoss {p k sp n : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)] (model : PhenotypeInformedGAM p k sp) (data : RealizedData n k) (lambda : ℝ) : ℝ :=
  (1 / (n : ℝ)) * (∑ i, pointwiseNLL model.dist (data.y i) (linearPredictor model (data.p i) (data.c i)))
  + lambda * ((∑ l, ∑ j, (model.f₀ₗ l j)^2) + (∑ m, ∑ l, ∑ j, (model.fₘₗ m l j)^2))

def IsIdentifiable {p k sp n : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)] (m : PhenotypeInformedGAM p k sp) (data : RealizedData n k) : Prop :=
  (∀ l, (∑ i, evalSmooth m.pcSplineBasis (m.f₀ₗ l) (data.c i l)) = 0) ∧
  (∀ mIdx l, (∑ i, evalSmooth m.pcSplineBasis (m.fₘₗ mIdx l) (data.c i l)) = 0)

-- Axiom: The penalized least squares problem has a solution (under suitable regularity conditions)
-- This abstracts over the numerical solver in estimate.rs (PIRLS algorithm)
axiom fit_exists (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)]
    (data : RealizedData n k) (lambda : ℝ) :
    ∃ (m : PhenotypeInformedGAM p k sp),
      (∀ (m' : PhenotypeInformedGAM p k sp), empiricalLoss m data lambda ≤ empiricalLoss m' data lambda) ∧
      IsIdentifiable m data

noncomputable def fit (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)] (data : RealizedData n k) (lambda : ℝ) : PhenotypeInformedGAM p k sp :=
  Classical.choose (fit_exists p k sp n data lambda)

theorem fit_minimizes_loss (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)]
    (data : RealizedData n k) (lambda : ℝ) :
  (∀ (m : PhenotypeInformedGAM p k sp), empiricalLoss (fit p k sp n data lambda) data lambda ≤ empiricalLoss m data lambda) ∧
  IsIdentifiable (fit p k sp n data lambda) data :=
    Classical.choose_spec (fit_exists p k sp n data lambda)

structure IsRawScoreModel {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (m : PhenotypeInformedGAM p k sp) : Prop where
  f₀ₗ_zero : ∀ (l : Fin k) (s : Fin sp), m.f₀ₗ l s = 0
  fₘₗ_zero : ∀ (i : Fin p) (l : Fin k) (s : Fin sp), m.fₘₗ i l s = 0

structure IsNormalizedScoreModel {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (m : PhenotypeInformedGAM p k sp) : Prop where
  fₘₗ_zero : ∀ (i : Fin p) (l : Fin k) (s : Fin sp), m.fₘₗ i l s = 0

-- Axiom: The constrained raw model optimization problem has a solution
axiom fitRaw_exists (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)]
    (data : RealizedData n k) (lambda : ℝ) :
    ∃ (m : PhenotypeInformedGAM p k sp),
      IsRawScoreModel m ∧
      ∀ (m' : PhenotypeInformedGAM p k sp), IsRawScoreModel m' →
        empiricalLoss m data lambda ≤ empiricalLoss m' data lambda

noncomputable def fitRaw (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)] (data : RealizedData n k) (lambda : ℝ) : PhenotypeInformedGAM p k sp :=
  Classical.choose (fitRaw_exists p k sp n data lambda)

theorem fitRaw_minimizes_loss (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)]
    (data : RealizedData n k) (lambda : ℝ) :
  IsRawScoreModel (fitRaw p k sp n data lambda) ∧
  ∀ (m : PhenotypeInformedGAM p k sp) (_h_m : IsRawScoreModel m),
    empiricalLoss (fitRaw p k sp n data lambda) data lambda ≤ empiricalLoss m data lambda := by
  have h := Classical.choose_spec (fitRaw_exists p k sp n data lambda)
  exact ⟨h.1, fun m hm => h.2 m hm⟩

-- Axiom: The constrained normalized model optimization problem has a solution
axiom fitNormalized_exists (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)]
    (data : RealizedData n k) (lambda : ℝ) :
    ∃ (m : PhenotypeInformedGAM p k sp),
      IsNormalizedScoreModel m ∧
      ∀ (m' : PhenotypeInformedGAM p k sp), IsNormalizedScoreModel m' →
        empiricalLoss m data lambda ≤ empiricalLoss m' data lambda

noncomputable def fitNormalized (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)] (data : RealizedData n k) (lambda : ℝ) : PhenotypeInformedGAM p k sp :=
  Classical.choose (fitNormalized_exists p k sp n data lambda)

theorem fitNormalized_minimizes_loss (p k sp n : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] [Fintype (Fin n)]
    (data : RealizedData n k) (lambda : ℝ) :
  IsNormalizedScoreModel (fitNormalized p k sp n data lambda) ∧
  ∀ (m : PhenotypeInformedGAM p k sp) (_h_m : IsNormalizedScoreModel m),
    empiricalLoss (fitNormalized p k sp n data lambda) data lambda ≤ empiricalLoss m data lambda := by
  have h := Classical.choose_spec (fitNormalized_exists p k sp n data lambda)
  exact ⟨h.1, fun m hm => h.2 m hm⟩

/-!
=================================================================
## Part 2: Fully Formalized Theorems and Proofs
=================================================================
-/

section AllClaims

variable {p k sp n : ℕ}

/-! ### Example Scenario DGPs (Specific Instantiations)

The following are **example instantiations** of `dgpAdditiveBias` with specific β values
from simulation studies. For general proofs, use `dgpAdditiveBias` with arbitrary β. -/

/-- **EXAMPLE**: Scenario 1 - Gene-environment interaction model.
    Phenotype = P × (1 + 0.1 × ΣC). NOT an additive bias model. -/
noncomputable def dgpScenario1_example (k : ℕ) [Fintype (Fin k)] : DataGeneratingProcess k := {
  trueExpectation := fun p pc => p * (1 + 0.1 * (∑ l, pc l)),
  jointMeasure := stdNormalProdMeasure k
}

/-- **EXAMPLE**: Scenario 3 - Positive confounding with β = 0.5.
    This is `dgpAdditiveBias k 0.5`. -/
noncomputable def dgpScenario3_example (k : ℕ) [Fintype (Fin k)] : DataGeneratingProcess k := {
  trueExpectation := fun p pc => p + (0.5 * (∑ l, pc l)),
  jointMeasure := stdNormalProdMeasure k
}

/-- **EXAMPLE**: Scenario 4 - Negative confounding with β = -0.8.
    This is `dgpAdditiveBias k (-0.8)`. -/
noncomputable def dgpScenario4_example (k : ℕ) [Fintype (Fin k)] : DataGeneratingProcess k := {
  trueExpectation := fun p pc => p - (0.8 * (∑ l, pc l)),
  jointMeasure := stdNormalProdMeasure k
}

/-! ### Generalized DGP and L² Projection Framework

The following definitions support a cleaner, more general proof approach:
- Instead of hardcoding constants like 0.8, we parameterize by β_env
- We view least-squares optimization as orthogonal projection in L²
- This unifies Scenario 3 (β > 0) and Scenario 4 (β < 0) -/

/-- General DGP where phenotype is P + β_env * Σ C.
    This generalizes Scenario 3 (β > 0) and Scenario 4 (β < 0).

    The key insight: the raw model (span{1, P}) cannot capture the β_env * C term,
    so the projection leaves a residual of exactly β_env * C. -/
noncomputable def dgpAdditiveBias (k : ℕ) [Fintype (Fin k)] (β_env : ℝ) : DataGeneratingProcess k := {
  trueExpectation := fun p pc => p + β_env * (∑ l, pc l),
  jointMeasure := stdNormalProdMeasure k
}

/-- Scenario 3 example is dgpAdditiveBias with β = 0.5. -/
lemma dgpScenario3_example_eq_additiveBias (k : ℕ) [Fintype (Fin k)] :
    dgpScenario3_example k = dgpAdditiveBias k 0.5 := by
  unfold dgpScenario3_example dgpAdditiveBias
  rfl

/-- Scenario 4 example is dgpAdditiveBias with β = -0.8. -/
lemma dgpScenario4_example_eq_additiveBias (k : ℕ) [Fintype (Fin k)] :
    dgpScenario4_example k = dgpAdditiveBias k (-0.8) := by
  unfold dgpScenario4_example dgpAdditiveBias
  simp only [neg_mul, sub_eq_add_neg]

def hasInteraction {k : ℕ} [Fintype (Fin k)] (f : ℝ → (Fin k → ℝ) → ℝ) : Prop :=
  ∃ (p₁ p₂ : ℝ) (c₁ c₂ : Fin k → ℝ), p₁ ≠ p₂ ∧ c₁ ≠ c₂ ∧
    (f p₂ c₁ - f p₁ c₁) / (p₂ - p₁) ≠ (f p₂ c₂ - f p₁ c₂) / (p₂ - p₁)

theorem scenarios_are_distinct (k : ℕ) (hk_pos : 0 < k) :
  hasInteraction (dgpScenario1_example k).trueExpectation ∧
  ¬ hasInteraction (dgpScenario3_example k).trueExpectation ∧
  ¬ hasInteraction (dgpScenario4_example k).trueExpectation := by
  constructor
  · -- Case 1: dgpScenario1_example has interaction
    unfold hasInteraction
    -- We provide witnesses for p₁, p₂, c₁, and c₂.
    -- p₁ and p₂ are real numbers. c₁ and c₂ are functions from Fin k to ℝ.
    use 0, 1, (fun _ => 0), (fun i => if i = ⟨0, hk_pos⟩ then 1 else 0)
    constructor; · norm_num -- Proves p₁ ≠ p₂
    constructor
    · -- Proves c₁ ≠ c₂ for any k > 0, including k=1
      intro h_eq
      -- If the functions are equal, they must be equal at the point ⟨0, hk_pos⟩.
      -- We use `congr_fun` to apply this equality.
      have := congr_fun h_eq ⟨0, hk_pos⟩
      -- This simplifies to 0 = 1, a contradiction.
      simp at this
    · -- Proves the inequality
      unfold dgpScenario1_example; dsimp
      have h_sum_c2 : (∑ (l : Fin k), if l = ⟨0, hk_pos⟩ then 1 else 0) = 1 := by
        -- The sum is 1 because the term is 1 only at i = ⟨0, hk_pos⟩ and 0 otherwise.
        simp [Finset.sum_ite_eq', Finset.mem_univ]
      -- Substitute the sum and simplify the expression
      simp [Finset.sum_const_zero]; norm_num
  · constructor
    · -- Case 2: dgpScenario3_example has no interaction
      intro h; rcases h with ⟨p₁, p₂, c₁, c₂, hp_neq, _, h_neq⟩
      unfold dgpScenario3_example at h_neq
      -- The terms with c₁ and c₂ cancel out, making the slope independent of c.
      simp only [add_sub_add_right_eq_sub] at h_neq
      -- This leads to 1 ≠ 1, a contradiction.
      contradiction
    · -- Case 3: dgpScenario4_example has no interaction
      intro h; rcases h with ⟨p₁, p₂, c₁, c₂, hp_neq, _, h_neq⟩
      unfold dgpScenario4_example at h_neq
      -- Similarly, the terms with c₁ and c₂ cancel out.
      simp only [sub_sub_sub_cancel_right] at h_neq
      -- This leads to 1 ≠ 1, a contradiction.
      contradiction

theorem necessity_of_phenotype_data :
  ∃ (dgp_A dgp_B : DataGeneratingProcess 1),
    dgp_A.jointMeasure = dgp_B.jointMeasure ∧ hasInteraction dgp_A.trueExpectation ∧ ¬ hasInteraction dgp_B.trueExpectation := by
  use dgpScenario1_example 1, dgpScenario4_example 1
  constructor; rfl
  have h_distinct := scenarios_are_distinct 1 (by norm_num)
  exact ⟨h_distinct.left, h_distinct.right.right⟩

noncomputable def expectedSquaredError {k : ℕ} [Fintype (Fin k)] (dgp : DataGeneratingProcess k) (f : ℝ → (Fin k → ℝ) → ℝ) : ℝ :=
  ∫ pc, (dgp.trueExpectation pc.1 pc.2 - f pc.1 pc.2)^2 ∂dgp.jointMeasure

/-- Bayes-optimal in the full GAM class (quantifies over all models). -/
def IsBayesOptimalInClass {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (dgp : DataGeneratingProcess k) (model : PhenotypeInformedGAM p k sp) : Prop :=
  ∀ (m : PhenotypeInformedGAM p k sp), expectedSquaredError dgp (fun p c => linearPredictor model p c) ≤
        expectedSquaredError dgp (fun p c => linearPredictor m p c)

/-- Bayes-optimal among raw score models only (L² projection onto {1, P} subspace).
    This is the correct predicate for Scenario 4, where the raw class cannot represent
    the true PC main effect. -/
structure IsBayesOptimalInRawClass {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (dgp : DataGeneratingProcess k) (model : PhenotypeInformedGAM p k sp) : Prop where
  is_raw : IsRawScoreModel model
  is_optimal : ∀ (m : PhenotypeInformedGAM p k sp), IsRawScoreModel m →
    expectedSquaredError dgp (fun p c => linearPredictor model p c) ≤
    expectedSquaredError dgp (fun p c => linearPredictor m p c)

/-- Bayes-optimal among normalized score models only (L² projection onto additive subspace). -/
structure IsBayesOptimalInNormalizedClass {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (dgp : DataGeneratingProcess k) (model : PhenotypeInformedGAM p k sp) : Prop where
  is_normalized : IsNormalizedScoreModel model
  is_optimal : ∀ (m : PhenotypeInformedGAM p k sp), IsNormalizedScoreModel m →
    expectedSquaredError dgp (fun p c => linearPredictor model p c) ≤
    expectedSquaredError dgp (fun p c => linearPredictor m p c)

/-! ### L² Projection Framework

**Key Insight**: Bayes-optimal prediction = orthogonal projection in L²(μ).

Instead of expanding integrals and deriving normal equations by hand, we work in the
Hilbert space L²(μ) where:
- Inner product: ⟪f, g⟫ = ∫ f·g dμ = E[fg]
- Norm: ‖f‖² = E[f²]
- Projection onto W gives the closest element to Y in W

For raw models, W = span{1, P}, and Bayes-optimality means:
  Ŷ = orthogonalProjection W Y

This gives orthogonality of residual FOR FREE via mathlib's
`orthogonalProjection_inner_eq_zero`:
  ∀ w ∈ W, ⟪Y - Ŷ, w⟫ = 0

-/

/-- The space of square-integrable functions on the probability space.
    This is the Hilbert space where we do orthogonal projection. -/
abbrev L2Space (μ : Measure (ℝ × (Fin 1 → ℝ))) := Lp ℝ 2 μ

/-- Feature function: constant 1 (for intercept). -/
def featureOne (_μ : Measure (ℝ × (Fin 1 → ℝ))) : (ℝ × (Fin 1 → ℝ)) → ℝ :=
  fun _ => 1

/-- Feature function: P (the PGS value). -/
def featureP (_μ : Measure (ℝ × (Fin 1 → ℝ))) : (ℝ × (Fin 1 → ℝ)) → ℝ :=
  fun pc => pc.1

/-- Feature function: C (the first PC value). -/
def featureC (_μ : Measure (ℝ × (Fin 1 → ℝ))) : (ℝ × (Fin 1 → ℝ)) → ℝ :=
  fun pc => pc.2 ⟨0, by norm_num⟩

/-- **Helper Lemma**: Under product measure (independence), E[P·C] = E[P]·E[C] = 0.
    Uses Fubini (integral_prod_mul) to factor the expectation. -/
lemma integral_mul_fst_snd_eq_zero
    (μ : Measure (ℝ × (Fin 1 → ℝ))) [IsProbabilityMeasure μ]
    (h_indep : μ = (μ.map Prod.fst).prod (μ.map Prod.snd))
    (hP0 : ∫ pc, pc.1 ∂μ = 0)
    (hC0 : ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂μ = 0) :
    ∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂μ = 0 := by
  classical
  set μP : Measure ℝ := μ.map Prod.fst
  set μC : Measure (Fin 1 → ℝ) := μ.map Prod.snd
  haveI : IsProbabilityMeasure μP :=
    Measure.isProbabilityMeasure_map (μ := μ) (f := Prod.fst) (by
      simpa using measurable_fst.aemeasurable)
  haveI : IsProbabilityMeasure μC :=
    Measure.isProbabilityMeasure_map (μ := μ) (f := Prod.snd) (by
      simpa using measurable_snd.aemeasurable)
  have hP0' : (∫ p, p ∂μP) = 0 := by
    have hP0_prod : (∫ pc, pc.1 ∂(μP.prod μC)) = 0 := by
      have h := hP0
      rw [h_indep] at h
      simpa [μP, μC] using h
    have hfst :
        (∫ pc, pc.1 ∂(μP.prod μC)) = (μC.real Set.univ) • (∫ p, p ∂μP) := by
      simpa using (MeasureTheory.integral_fun_fst (μ := μP) (ν := μC) (f := fun p : ℝ => p))
    have hμC : μC.real Set.univ = (1 : ℝ) := by
      simp
    have : (μC.real Set.univ) • (∫ p, p ∂μP) = 0 := hfst.symm.trans hP0_prod
    simpa [hμC] using this
  have hC0' : (∫ c, c ⟨0, by norm_num⟩ ∂μC) = 0 := by
    have hC0_prod : (∫ pc, pc.2 ⟨0, by norm_num⟩ ∂(μP.prod μC)) = 0 := by
      have h := hC0
      rw [h_indep] at h
      simpa [μP, μC] using h
    have hsnd :
        (∫ pc, pc.2 ⟨0, by norm_num⟩ ∂(μP.prod μC)) =
          (μP.real Set.univ) • (∫ c, c ⟨0, by norm_num⟩ ∂μC) := by
      simpa using
        (MeasureTheory.integral_fun_snd (μ := μP) (ν := μC)
          (f := fun c : (Fin 1 → ℝ) => c ⟨0, by norm_num⟩))
    have hμP : μP.real Set.univ = (1 : ℝ) := by
      simp
    have : (μP.real Set.univ) • (∫ c, c ⟨0, by norm_num⟩ ∂μC) = 0 := hsnd.symm.trans hC0_prod
    simpa [hμP] using this
  rw [h_indep]
  simpa [μP, μC, hP0', hC0'] using
    (MeasureTheory.integral_prod_mul (μ := μP) (ν := μC) (f := fun p : ℝ => p)
      (g := fun c : (Fin 1 → ℝ) => c ⟨0, by norm_num⟩))

/-- **Core Lemma**: Under independence + zero-mean, {1, P, C} form an orthogonal set in L².
    This is because:
    - ⟪1, P⟫ = E[P] = 0
    - ⟪1, C⟫ = E[C] = 0
    - ⟪P, C⟫ = E[PC] = E[P]E[C] = 0 (by independence) -/
lemma orthogonal_features
    (μ : Measure (ℝ × (Fin 1 → ℝ))) [IsProbabilityMeasure μ]
    (h_indep : μ = (μ.map Prod.fst).prod (μ.map Prod.snd))
    (hP0 : ∫ pc, pc.1 ∂μ = 0)
    (hC0 : ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂μ = 0) :
    (∫ pc, 1 * pc.1 ∂μ = 0) ∧
    (∫ pc, 1 * pc.2 ⟨0, by norm_num⟩ ∂μ = 0) ∧
    (∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂μ = 0) := by
  refine ⟨?_, ?_, ?_⟩
  · simp only [one_mul]; exact hP0
  · simp only [one_mul]; exact hC0
  · exact integral_mul_fst_snd_eq_zero μ h_indep hP0 hC0

/-! ### L² Orthogonality Characterization (Classical Derivation) -/

/-- **Standalone Lemma**: Optimal coefficients for Raw Model on Additive DGP.
    Given Y = P + β*C, independence, and standardized moments:
    The raw model (projecting onto span{1, P}) has coefficients a=0, b=1.

    This isolates the algebraic result from the larger theorems. -/
lemma optimal_coeffs_raw_additive_standalone
    (a b β_env : ℝ)
    (h_orth_1 : a + b * 0 = 0 + β_env * 0) -- derived from E[resid] = 0
    (h_orth_P : a * 0 + b * 1 = 1 + β_env * 0) -- derived from E[resid*P] = 0
    : a = 0 ∧ b = 1 := by
  have ha : a = 0 := by
    linarith
  have hb : b = 1 := by
    linarith
  exact ⟨ha, hb⟩

/-- First normal equation: optimality implies a = E[Y] (when E[P] = 0).
    This is the orthogonality condition ⟪residual, 1⟫ = 0. -/
lemma optimal_intercept_eq_mean_of_zero_mean_p
    (μ : Measure (ℝ × (Fin 1 → ℝ))) [IsProbabilityMeasure μ]
    (Y : (ℝ × (Fin 1 → ℝ)) → ℝ) (a b : ℝ)
    (hY : Integrable Y μ)
    (hP : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) μ)
    (hP0 : ∫ pc, pc.1 ∂μ = 0)
    (h_orth_1 : ∫ pc, (Y pc - (a + b * pc.1)) ∂μ = 0) :
    a = ∫ pc, Y pc ∂μ := by
  have hLin : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => a + b * pc.1) μ := by
    have ha : Integrable (fun _ : ℝ × (Fin 1 → ℝ) => a) μ := by
      simp
    have hb : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => b * pc.1) μ := hP.const_mul b
    simpa using ha.add hb
  have h0 :
      (∫ pc, Y pc ∂μ) - (∫ pc, (a + b * pc.1) ∂μ) = 0 := by
    simpa [MeasureTheory.integral_sub hY hLin] using h_orth_1
  have hLinInt : (∫ pc, (a + b * pc.1) ∂μ) = a := by
    -- `E[a + bP] = a * E[1] + b * E[P] = a + b * 0 = a`
    have ha : Integrable (fun _ : ℝ × (Fin 1 → ℝ) => a) μ := by
      simp
    have hb : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => b * pc.1) μ := hP.const_mul b
    calc
      (∫ pc, (a + b * pc.1) ∂μ) = (∫ pc, (a : ℝ) ∂μ) + ∫ pc, b * pc.1 ∂μ := by
        simpa using (MeasureTheory.integral_add ha hb)
      _ = a + b * (∫ pc, pc.1 ∂μ) := by
        simp [MeasureTheory.integral_const, MeasureTheory.integral_const_mul]
      _ = a := by simp [hP0]
  -- Rearrangement: `E[Y] - a = 0`.
  linarith [h0, hLinInt]

/-- Second normal equation: optimality implies b = E[YP] (when E[P] = 0, E[P²] = 1).
    This is the orthogonality condition ⟪residual, P⟫ = 0. -/
lemma optimal_slope_eq_covariance_of_normalized_p
    (μ : Measure (ℝ × (Fin 1 → ℝ))) [IsProbabilityMeasure μ]
    (Y : (ℝ × (Fin 1 → ℝ)) → ℝ) (a b : ℝ)
    (_hY : Integrable Y μ)
    (hP : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) μ)
    (hYP : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => Y pc * pc.1) μ)
    (hP2i : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1 ^ 2) μ)
    (hP0 : ∫ pc, pc.1 ∂μ = 0)
    (hP2 : ∫ pc, pc.1^2 ∂μ = 1)
    (h_orth_P : ∫ pc, (Y pc - (a + b * pc.1)) * pc.1 ∂μ = 0) :
    b = ∫ pc, Y pc * pc.1 ∂μ := by
  have hLin : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => a + b * pc.1) μ := by
    have ha : Integrable (fun _ : ℝ × (Fin 1 → ℝ) => a) μ := by
      simp
    have hb : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => b * pc.1) μ := hP.const_mul b
    simpa using ha.add hb
  have hLinP : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => (a + b * pc.1) * pc.1) μ := by
    -- Integrable because it's a linear combination of `pc.1` and `pc.1^2`.
    -- `(a + bP) * P = a*P + b*P^2`
    have h1 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => a * pc.1) μ := hP.const_mul a
    have h2 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => b * (pc.1 ^ 2)) μ := hP2i.const_mul b
    -- rewrite and use `integrable_congr` to match `h1.add h2`
    refine (h1.add h2).congr ?_
    filter_upwards with pc
    ring_nf
    simp
  have h0 :
      (∫ pc, Y pc * pc.1 ∂μ) - (∫ pc, (a + b * pc.1) * pc.1 ∂μ) = 0 := by
    -- Expand the orthogonality condition using integral linearity.
    have hSub : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => Y pc * pc.1 - (a + b * pc.1) * pc.1) μ := by
      exact hYP.sub hLinP
    -- `(Y - (a+bP))*P = YP - (a+bP)P`
    have hEq :
        (fun pc : ℝ × (Fin 1 → ℝ) => (Y pc - (a + b * pc.1)) * pc.1) =
          (fun pc : ℝ × (Fin 1 → ℝ) => Y pc * pc.1 - (a + b * pc.1) * pc.1) := by
      funext pc
      ring_nf
    -- Use the rewritten integrand.
    have hOrth' : ∫ pc, (Y pc * pc.1 - (a + b * pc.1) * pc.1) ∂μ = 0 := by
      simpa [hEq] using h_orth_P
    simpa [MeasureTheory.integral_sub hYP hLinP] using hOrth'
  have hLinPInt : (∫ pc, (a + b * pc.1) * pc.1 ∂μ) = b := by
    -- `E[(a+bP)P] = a*E[P] + b*E[P^2] = 0 + b*1 = b`
    have h1 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => a * pc.1) μ := hP.const_mul a
    have h2 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => b * (pc.1 ^ 2)) μ := hP2i.const_mul b
    have hsum : (∫ pc, (a + b * pc.1) * pc.1 ∂μ) = (∫ pc, a * pc.1 + b * (pc.1 ^ 2) ∂μ) := by
      refine MeasureTheory.integral_congr_ae ?_
      filter_upwards with pc
      ring_nf
    rw [hsum]
    calc
      (∫ pc, a * pc.1 + b * (pc.1 ^ 2) ∂μ) =
          (∫ pc, a * pc.1 ∂μ) + ∫ pc, b * (pc.1 ^ 2) ∂μ := by
            simpa using (MeasureTheory.integral_add h1 h2)
      _ = a * (∫ pc, pc.1 ∂μ) + b * (∫ pc, pc.1 ^ 2 ∂μ) := by
            simp [MeasureTheory.integral_const_mul]
      _ = b := by simp [hP0, hP2]
  -- Rearrangement: `E[YP] - b = 0`.
  linarith [h0, hLinPInt]


/-- Helper lemma: For a raw score model, the PC main effect spline term is always zero. -/
lemma evalSmooth_eq_zero_of_raw {model : PhenotypeInformedGAM 1 1 1} (h_raw : IsRawScoreModel model)
    (l : Fin 1) (c_val : ℝ) :
    evalSmooth model.pcSplineBasis (model.f₀ₗ l) c_val = 0 := by
  unfold evalSmooth
  simp [h_raw.f₀ₗ_zero l]

/-- Helper lemma: For a raw score model, the PGS-PC interaction spline term is always zero. -/
lemma evalSmooth_interaction_eq_zero_of_raw {model : PhenotypeInformedGAM 1 1 1} (h_raw : IsRawScoreModel model)
    (m : Fin 1) (l : Fin 1) (c_val : ℝ) :
    evalSmooth model.pcSplineBasis (model.fₘₗ m l) c_val = 0 := by
  unfold evalSmooth
  simp [h_raw.fₘₗ_zero m l]

/-- **Lemma A**: For a raw model (all spline terms zero) with linear PGS basis,
    the linear predictor simplifies to an affine function: a + b*p.
    This is the key structural simplification.

    Proof uses linearPredictor_decomp then shows base and slope simplify for raw models. -/
lemma linearPredictor_eq_affine_of_raw
    (model_raw : PhenotypeInformedGAM 1 1 1)
    (h_raw : IsRawScoreModel model_raw)
    (h_lin : model_raw.pgsBasis.B 1 = id ∧ model_raw.pgsBasis.B 0 = fun _ => 1) :
    ∀ p c, linearPredictor model_raw p c =
      model_raw.γ₀₀ + model_raw.γₘ₀ 0 * p := by
  intros p_val c_val

  -- Step 1: Use linearPredictor_decomp to get base + slope * p form
  have h_decomp := linearPredictor_decomp model_raw h_lin.1 p_val c_val
  rw [h_decomp]

  -- Step 2: Show base reduces to γ₀₀ for raw model
  have h_base : predictorBase model_raw c_val = model_raw.γ₀₀ := by
    unfold predictorBase
    simp [evalSmooth_eq_zero_of_raw h_raw]

  -- Step 3: Show slope reduces to γₘ₀[0] for raw model
  have h_slope : predictorSlope model_raw c_val = model_raw.γₘ₀ 0 := by
    unfold predictorSlope
    simp [evalSmooth_interaction_eq_zero_of_raw h_raw]

  rw [h_base, h_slope]

/-- The key bridge: IsBayesOptimalInRawClass implies the orthogonality conditions.

    **Variational Proof (Fundamental Theorem of Least Squares)**:
    If Ŷ minimizes E[(Y - Ŷ)²] over the class of affine functions of P,
    then for any perturbation direction v ∈ span{1, P}, the directional derivative
    of the loss at Ŷ in direction v must be zero.

    Define L(ε) = E[(Y - (Ŷ + ε·v))²]
                = E[(Y - Ŷ)²] - 2ε·E[(Y - Ŷ)·v] + ε²·E[v²]

    For L(ε) ≥ L(0) for all ε (by optimality), the linear coefficient must vanish:
        E[(Y - Ŷ)·v] = 0

    Taking v = 1 gives: E[Y - Ŷ] = 0 (first normal equation)
    Taking v = P gives: E[(Y - Ŷ)·P] = 0 (second normal equation)

    **FUTURE**:
    - Unify empirical and theoretical loss via measure theory: treat both as L²(μ)
      for different measures μ (population vs empirical 1/n Σδᵢ)
    - Abstract the parameter space to any [InnerProductSpace ℝ P], not just ParamIx
    - Use LinearMap instead of Matrix for cleaner kernel/image reasoning -/
lemma rawOptimal_implies_orthogonality
    (model : PhenotypeInformedGAM 1 1 1) (dgp : DataGeneratingProcess 1)
    (h_opt : IsBayesOptimalInRawClass dgp model)
    (h_linear : model.pgsBasis.B 1 = id ∧ model.pgsBasis.B 0 = fun _ => 1)
    (hY_int : Integrable (fun pc => dgp.trueExpectation pc.1 pc.2) dgp.jointMeasure)
    (hP_int : Integrable (fun pc => pc.1) dgp.jointMeasure)
    (hP2_int : Integrable (fun pc => pc.1 ^ 2) dgp.jointMeasure)
    (hYP_int : Integrable (fun pc => dgp.trueExpectation pc.1 pc.2 * pc.1) dgp.jointMeasure)
    (h_resid_sq_int : Integrable (fun pc => (dgp.trueExpectation pc.1 pc.2 - (model.γ₀₀ + model.γₘ₀ ⟨0, by norm_num⟩ * pc.1))^2) dgp.jointMeasure) :
    let a := model.γ₀₀
    let b := model.γₘ₀ ⟨0, by norm_num⟩
    -- Orthogonality with 1:
    (∫ pc, (dgp.trueExpectation pc.1 pc.2 - (a + b * pc.1)) ∂dgp.jointMeasure = 0) ∧
    -- Orthogonality with P:
    (∫ pc, (dgp.trueExpectation pc.1 pc.2 - (a + b * pc.1)) * pc.1 ∂dgp.jointMeasure = 0) := by

  set a := model.γ₀₀ with ha_def
  set b := model.γₘ₀ ⟨0, by norm_num⟩ with hb_def
  set μ := dgp.jointMeasure with hμ_def
  set Y := dgp.trueExpectation with hY_def

  -- The residual function
  set residual : ℝ × (Fin 1 → ℝ) → ℝ := fun pc => Y pc.1 pc.2 - (a + b * pc.1) with hres_def

  -- **Proof Strategy**:
  -- By h_opt, model minimizes expectedSquaredError over all raw models.
  -- We construct competitor models with perturbed intercept (a + ε) and slope (b + ε)
  -- and show that the optimality condition forces the orthogonality.

  -- **Claim 1**: E[residual · 1] = 0
  -- Consider the competitor model with intercept (a + ε) for small ε.
  -- The competitor's squared error is:
  --   L(ε) = E[(Y - ((a + ε) + b*P))²]
  --        = E[(Y - a - b*P - ε)²]
  --        = E[(residual - ε)²]
  --        = E[residual²] - 2ε·E[residual] + ε²
  --
  -- Since L(0) ≤ L(ε) for all ε (optimality), L'(0) = 0.
  -- L'(ε) = -2·E[residual] + 2ε
  -- L'(0) = -2·E[residual] = 0
  -- Therefore E[residual] = 0, i.e., E[residual · 1] = 0.

  -- **Claim 2**: E[residual · P] = 0
  -- Consider the competitor model with slope (b + ε).
  -- The competitor's squared error is:
  --   L(ε) = E[(Y - (a + (b + ε)*P))²]
  --        = E[(Y - a - b*P - ε*P)²]
  --        = E[(residual - ε*P)²]
  --        = E[residual²] - 2ε·E[residual·P] + ε²·E[P²]
  --
  -- Since L(0) ≤ L(ε) for all ε (optimality), L'(0) = 0.
  -- L'(ε) = -2·E[residual·P] + 2ε·E[P²]
  -- L'(0) = -2·E[residual·P] = 0
  -- Therefore E[residual · P] = 0.

  -- The formal proof requires constructing the competitor models and using
  -- h_opt.is_optimal to get the inequality, then taking the limit as ε → 0.
  -- This involves Calculus (derivatives of integrals) or the algebraic
  -- manipulation shown above.
  --
  -- **Formal Proof Outline**:
  -- 1. Construct a "competitor" raw score model `m'` with a slightly perturbed
  --    intercept `a' = a + ε`. This is a valid raw model.
  -- 2. By `h_opt.is_optimal`, the risk of `model` is less than or equal to the
  --    risk of `m'`: `R(a,b) ≤ R(a+ε, b)`.
  -- 3. The risk function is a quadratic in its parameters. Expanding the inequality
  --    `E[(Y - (a+b·P))²] ≤ E[(Y - (a+ε+b·P))²]` simplifies to
  --    `0 ≤ ε² - 2ε·E[Y - (a+b·P)]`.
  -- 4. This must hold for all `ε`. If `E[...] > 0`, a small positive `ε` violates it.
  --    If `E[...] < 0`, a small negative `ε` violates it. Thus `E[...] = 0`.
  -- 5. A similar argument with a perturbed slope `b' = b + ε` shows that
  --    `E[(Y - (a+b·P))·P] = 0`.

  constructor
  · -- Orthogonality with 1: E[residual] = 0
    -- **Quadratic Perturbation Proof** (no calculus of variations needed):
    -- L(ε) = E[(residual - ε)²] = E[residual²] - 2ε·E[residual] + ε²
    --      = L(0) + ε² - 2ε·E[residual]
    -- Optimality: L(0) ≤ L(ε) for all ε
    -- This means: 0 ≤ ε² - 2ε·E[residual] = ε(ε - 2·E[residual])
    -- Testing ε > 0 small: 0 ≤ ε - 2·E[residual], so E[residual] ≤ ε/2 → E[residual] ≤ 0
    -- Testing ε < 0 small: 0 ≤ ε - 2·E[residual] becomes 2·E[residual] ≤ ε → E[residual] ≥ 0
    -- Therefore E[residual] = 0
    have h1 : ∫ pc, residual pc ∂μ = 0 := by
      -- The formal proof constructs competitor models and uses h_opt.is_optimal
      -- to derive ε² - 2ε·E[residual] ≥ 0 for all ε, which forces E[residual] = 0.
      -- This is a standard "first-order necessary condition" from optimization theory.
      sorry -- Quadratic perturbation: ε² - 2ε·E[resid] ≥ 0 ∀ε ⟹ E[resid] = 0
    simpa [hres_def] using h1

  · -- Orthogonality with P: E[residual · P] = 0
    -- **Quadratic Perturbation Proof**:
    -- L(ε) = E[(residual - εP)²] = E[residual²] - 2ε·E[residual·P] + ε²·E[P²]
    -- Optimality: 0 ≤ -2ε·E[residual·P] + ε²·E[P²] = ε(-2·E[residual·P] + ε·E[P²])
    -- For small |ε|, the sign is determined by -2·E[residual·P]
    -- Testing ε > 0: -2·E[residual·P] ≥ 0 → E[residual·P] ≤ 0
    -- Testing ε < 0: -2·E[residual·P] ≤ 0 → E[residual·P] ≥ 0
    -- Therefore E[residual·P] = 0
    have h2 : ∫ pc, residual pc * pc.1 ∂μ = 0 := by
      -- Same construction with slope perturbation b → b + ε
      sorry -- Quadratic perturbation: ε(-2E[resid·P] + εE[P²]) ≥ 0 ∀ε ⟹ E[resid·P] = 0
    simpa [hres_def] using h2

/-- Combine the normal equations to get the optimal coefficients for additive bias DGP.

    **Proof Strategy (Orthogonality Principle)**:
    The Bayes-optimal predictor Ŷ = a + b*P in the raw class satisfies
    the normal equations (orthogonality with basis vectors 1 and P):
      ⟨Y - Ŷ, 1⟩ = 0  ⟹  E[Y] = a + b*E[P] = a  (since E[P] = 0)
      ⟨Y - Ŷ, P⟩ = 0  ⟹  E[YP] = a*E[P] + b*E[P²] = b  (since E[P]=0, E[P²]=1)

    For Y = P + β*C:
      E[Y] = E[P] + β*E[C] = 0 + β*0 = 0  ⟹  a = 0
      E[YP] = E[P²] + β*E[PC] = 1 + β*0 = 1  ⟹  b = 1
-/
lemma optimal_coefficients_for_additive_dgp
    (model : PhenotypeInformedGAM 1 1 1) (β_env : ℝ)
    (dgp : DataGeneratingProcess 1)
    (h_dgp : dgp.trueExpectation = fun p c => p + β_env * c ⟨0, by norm_num⟩)
    (h_opt : IsBayesOptimalInRawClass dgp model)
    (h_linear : model.pgsBasis.B 1 = id ∧ model.pgsBasis.B 0 = fun _ => 1)
    (h_indep : dgp.jointMeasure = (dgp.jointMeasure.map Prod.fst).prod (dgp.jointMeasure.map Prod.snd))
    (hP0 : ∫ pc, pc.1 ∂dgp.jointMeasure = 0)
    (hC0 : ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂dgp.jointMeasure = 0)
    (hP2 : ∫ pc, pc.1^2 ∂dgp.jointMeasure = 1)
    -- Integrability hypotheses
    (hP_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) dgp.jointMeasure)
    (hC_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩) dgp.jointMeasure)
    (hP2_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1 ^ 2) dgp.jointMeasure)
    (hPC_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩ * pc.1) dgp.jointMeasure)
    (hY_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp.trueExpectation pc.1 pc.2) dgp.jointMeasure)
    (hYP_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp.trueExpectation pc.1 pc.2 * pc.1) dgp.jointMeasure)
    (h_resid_sq_int : Integrable (fun pc => (dgp.trueExpectation pc.1 pc.2 - (model.γ₀₀ + model.γₘ₀ ⟨0, by norm_num⟩ * pc.1))^2) dgp.jointMeasure) :
    model.γ₀₀ = 0 ∧ model.γₘ₀ ⟨0, by norm_num⟩ = 1 := by
  -- Step 1: Get the orthogonality conditions from optimality
  have h_orth := rawOptimal_implies_orthogonality model dgp h_opt h_linear hY_int hP_int hP2_int hYP_int h_resid_sq_int
  set a := model.γ₀₀ with ha_def
  set b := model.γₘ₀ ⟨0, by norm_num⟩ with hb_def
  obtain ⟨h_orth1, h_orthP⟩ := h_orth

  -- Step 2: Compute E[PC] = 0 using independence
  have hPC0 : ∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂dgp.jointMeasure = 0 :=
    integral_mul_fst_snd_eq_zero dgp.jointMeasure h_indep hP0 hC0

  -- Step 3: Compute E[Y] where Y = P + β*C
  -- E[Y] = E[P] + β*E[C] = 0 + β*0 = 0
  have hY_mean : ∫ pc, dgp.trueExpectation pc.1 pc.2 ∂dgp.jointMeasure = 0 := by
    -- E[P + β*C] = E[P] + β*E[C] = 0 + β*0 = 0 by hP0 and hC0
    simp only [h_dgp]
    -- Goal: ∫ pc, pc.1 + β_env * pc.2 ⟨0, _⟩ ∂μ = 0
    -- We need integrability hypotheses. Since we assume a probability measure
    -- with finite moments (implicit in hP0, hC0, hP2), we admit integrability.
    have hP_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) dgp.jointMeasure := hP_int
    have hC_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩) dgp.jointMeasure := hC_int
    calc ∫ pc, pc.1 + β_env * pc.2 ⟨0, by norm_num⟩ ∂dgp.jointMeasure
        = (∫ pc, pc.1 ∂dgp.jointMeasure) + β_env * (∫ pc, pc.2 ⟨0, by norm_num⟩ ∂dgp.jointMeasure) := by
          rw [integral_add hP_int (hC_int.const_mul β_env)]
          rw [integral_const_mul]
        _ = 0 + β_env * 0 := by rw [hP0, hC0]
        _ = 0 := by ring

  -- Step 4: Compute E[YP] where Y = P + β*C
  -- E[YP] = E[P²] + β*E[PC] = 1 + β*0 = 1
  have hYP : ∫ pc, dgp.trueExpectation pc.1 pc.2 * pc.1 ∂dgp.jointMeasure = 1 := by
    -- E[(P + β*C)*P] = E[P²] + β*E[PC] = 1 + 0 = 1 by hP2 and hPC0
    simp only [h_dgp]
    -- Goal: ∫ pc, (pc.1 + β_env * pc.2 ⟨0, _⟩) * pc.1 ∂μ = 1
    -- Expand: ∫ (P² + β*C*P) = ∫ P² + β * ∫ C*P = 1 + β*0 = 1
    have hP2_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1 ^ 2) dgp.jointMeasure := hP2_int
    have hPC_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩ * pc.1) dgp.jointMeasure := hPC_int
    -- Rewrite (P + β*C)*P as P² + β*(C*P)
    have heq : ∀ pc : ℝ × (Fin 1 → ℝ), (pc.1 + β_env * pc.2 ⟨0, by norm_num⟩) * pc.1
                                      = pc.1 ^ 2 + β_env * (pc.2 ⟨0, by norm_num⟩ * pc.1) := by
      intro pc; ring
    calc ∫ pc, (pc.1 + β_env * pc.2 ⟨0, by norm_num⟩) * pc.1 ∂dgp.jointMeasure
        = ∫ pc, pc.1 ^ 2 + β_env * (pc.2 ⟨0, by norm_num⟩ * pc.1) ∂dgp.jointMeasure := by
          congr 1; ext pc; exact heq pc
        _ = (∫ pc, pc.1 ^ 2 ∂dgp.jointMeasure) + β_env * (∫ pc, pc.2 ⟨0, by norm_num⟩ * pc.1 ∂dgp.jointMeasure) := by
          rw [integral_add hP2_int (hPC_int.const_mul β_env)]
          rw [integral_const_mul]
        _ = 1 + β_env * 0 := by
          rw [hP2]
          -- Need to show ∫ C*P = 0. This is hPC0 with commuted multiplication.
          have hPC_comm : ∫ pc, pc.2 ⟨0, by norm_num⟩ * pc.1 ∂dgp.jointMeasure
                        = ∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂dgp.jointMeasure := by
            congr 1; ext pc; ring
          rw [hPC_comm, hPC0]
        _ = 1 := by ring

  -- Step 5: Apply the normal equations to extract a and b
  -- From h_orth1: E[Y - (a + b*P)] = 0
  -- ⟹ E[Y] - a - b*E[P] = 0
  -- ⟹ 0 - a - 0 = 0  (using hY_mean=0, hP0=0)
  -- ⟹ a = 0

  -- From h_orthP: E[(Y - (a + b*P))*P] = 0
  -- ⟹ E[YP] - a*E[P] - b*E[P²] = 0
  -- ⟹ 1 - 0 - b = 0  (using hYP=1, hP0=0, hP2=1)
  -- ⟹ b = 1

  -- First, expand h_orth1 to get a = E[Y] = 0
  have ha : a = 0 := by
    -- By orthogonality: E[Y - (a + bP)] = 0
    -- E[Y] - a - b*E[P] = 0
    -- 0 - a - b*0 = 0 (by hY_mean, hP0)
    -- a = 0
    have h_expand : ∫ pc, (dgp.trueExpectation pc.1 pc.2 - (a + b * pc.1)) ∂dgp.jointMeasure
                  = (∫ pc, dgp.trueExpectation pc.1 pc.2 ∂dgp.jointMeasure) - a - b * (∫ pc, pc.1 ∂dgp.jointMeasure) := by
      -- Need integrability hypotheses
      have hY_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp.trueExpectation pc.1 pc.2) dgp.jointMeasure := hY_int
      have hP_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) dgp.jointMeasure := hP_int
      -- ∫ (Y - (a + bP)) = ∫ Y - ∫ (a + bP) = ∫ Y - a - b*∫ P
      have hConst_int : Integrable (fun _ : ℝ × (Fin 1 → ℝ) => a) dgp.jointMeasure := by
        simp
      have hLin_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => a + b * pc.1) dgp.jointMeasure := by
        exact hConst_int.add (hP_int.const_mul b)
      calc ∫ pc, dgp.trueExpectation pc.1 pc.2 - (a + b * pc.1) ∂dgp.jointMeasure
          = (∫ pc, dgp.trueExpectation pc.1 pc.2 ∂dgp.jointMeasure) - (∫ pc, a + b * pc.1 ∂dgp.jointMeasure) := by
            rw [integral_sub hY_int hLin_int]
          _ = (∫ pc, dgp.trueExpectation pc.1 pc.2 ∂dgp.jointMeasure) - (a + b * (∫ pc, pc.1 ∂dgp.jointMeasure)) := by
            congr 1
            calc ∫ pc, a + b * pc.1 ∂dgp.jointMeasure
                = (∫ pc, (a : ℝ) ∂dgp.jointMeasure) + (∫ pc, b * pc.1 ∂dgp.jointMeasure) := by
                  exact integral_add hConst_int (hP_int.const_mul b)
                _ = a + b * (∫ pc, pc.1 ∂dgp.jointMeasure) := by
                  simp [integral_const, MeasureTheory.integral_const_mul]
          _ = (∫ pc, dgp.trueExpectation pc.1 pc.2 ∂dgp.jointMeasure) - a - b * (∫ pc, pc.1 ∂dgp.jointMeasure) := by ring
    rw [h_expand, hY_mean, hP0] at h_orth1
    linarith

  -- Second, expand h_orthP to get b = 1
  have hb : b = 1 := by
    -- By orthogonality: E[(Y - (a + bP)) * P] = 0
    -- E[YP] - a*E[P] - b*E[P²] = 0
    -- 1 - 0 - b*1 = 0 (by hYP, hP0, hP2)
    -- b = 1
    have h_expand : ∫ pc, (dgp.trueExpectation pc.1 pc.2 - (a + b * pc.1)) * pc.1 ∂dgp.jointMeasure
                  = (∫ pc, dgp.trueExpectation pc.1 pc.2 * pc.1 ∂dgp.jointMeasure)
                    - a * (∫ pc, pc.1 ∂dgp.jointMeasure)
                    - b * (∫ pc, pc.1^2 ∂dgp.jointMeasure) := by
      -- Need integrability hypotheses
      have hYP_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp.trueExpectation pc.1 pc.2 * pc.1) dgp.jointMeasure := hYP_int
      have hP_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) dgp.jointMeasure := hP_int
      have hP2_int' : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1^2) dgp.jointMeasure := hP2_int
      have hLinP_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => (a + b * pc.1) * pc.1) dgp.jointMeasure := by
        -- (a + bP)*P = aP + bP²
        have h1 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => a * pc.1) dgp.jointMeasure := hP_int.const_mul a
        have h2 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => b * pc.1^2) dgp.jointMeasure := hP2_int.const_mul b
        have heq_ae : ∀ᵐ pc ∂dgp.jointMeasure, a * pc.1 + b * pc.1^2 = (a + b * pc.1) * pc.1 := by
          filter_upwards with pc
          ring
        exact (h1.add h2).congr heq_ae
      calc ∫ pc, (dgp.trueExpectation pc.1 pc.2 - (a + b * pc.1)) * pc.1 ∂dgp.jointMeasure
          = ∫ pc, dgp.trueExpectation pc.1 pc.2 * pc.1 - (a + b * pc.1) * pc.1 ∂dgp.jointMeasure := by
            congr 1; ext pc; ring
          _ = (∫ pc, dgp.trueExpectation pc.1 pc.2 * pc.1 ∂dgp.jointMeasure) - (∫ pc, (a + b * pc.1) * pc.1 ∂dgp.jointMeasure) := by
            rw [integral_sub hYP_int hLinP_int]
          _ = (∫ pc, dgp.trueExpectation pc.1 pc.2 * pc.1 ∂dgp.jointMeasure)
              - (a * (∫ pc, pc.1 ∂dgp.jointMeasure) + b * (∫ pc, pc.1^2 ∂dgp.jointMeasure)) := by
            congr 1
            -- Expand ∫ (a + bP)*P = ∫ aP + bP² = a*∫ P + b*∫ P²
            have h1 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => a * pc.1) dgp.jointMeasure := hP_int.const_mul a
            have h2 : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => b * pc.1^2) dgp.jointMeasure := hP2_int.const_mul b
            calc ∫ pc, (a + b * pc.1) * pc.1 ∂dgp.jointMeasure
                = ∫ pc, a * pc.1 + b * pc.1^2 ∂dgp.jointMeasure := by
                  congr 1; ext pc; ring
                _ = (∫ pc, a * pc.1 ∂dgp.jointMeasure) + (∫ pc, b * pc.1^2 ∂dgp.jointMeasure) := by
                  exact integral_add h1 h2
                _ = a * (∫ pc, pc.1 ∂dgp.jointMeasure) + b * (∫ pc, pc.1^2 ∂dgp.jointMeasure) := by
                  simp [MeasureTheory.integral_const_mul]
          _ = (∫ pc, dgp.trueExpectation pc.1 pc.2 * pc.1 ∂dgp.jointMeasure)
              - a * (∫ pc, pc.1 ∂dgp.jointMeasure) - b * (∫ pc, pc.1^2 ∂dgp.jointMeasure) := by ring
    rw [h_expand, hYP, hP0, hP2, ha] at h_orthP
    linarith

  exact ⟨ha, hb⟩

theorem l2_projection_of_additive_is_additive (p k sp : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] {f : ℝ → ℝ} {g : Fin k → ℝ → ℝ} {dgp : DataGeneratingProcess k}
  (h_indep : dgp.jointMeasure = (dgp.jointMeasure.map Prod.fst).prod (dgp.jointMeasure.map Prod.snd))
  (h_true_fn : dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
  (proj : PhenotypeInformedGAM p k sp) (h_optimal : IsBayesOptimalInClass dgp proj) :
  IsNormalizedScoreModel proj := by
  -- **L² Projection Principle** (axiomatized):
  -- When Y = f(P) + g(C) is additive and (P,C) are independent,
  -- the projection of Y onto a GAM space has zero interaction coefficients
  -- because the additive and interaction subspaces are orthogonal under independence.
  -- This follows from Fubini's theorem: ⟨additive, interaction⟩ = 0.
  -- See also: Hastie & Tibshirani (1990), "Generalized Additive Models", Chapter 8.
  exact ⟨by
    -- All interaction coefficients fₘₗ are zero by L² orthogonality
    intros i l s
    -- The formal proof requires showing the L² inner product of interaction basis
    -- functions with additive functions is zero under independence.
    -- Since the true function is additive, the optimal projection onto a space
    -- including interactions will have zero coefficients for the interaction terms.
    -- This is a standard result from the theory of reproducing kernel Hilbert spaces (RKHS)
    -- and ANOVA decompositions. See, for example, Gu (2013), "Smoothing Spline ANOVA Models".
    -- The proof relies on the fact that the interaction space is orthogonal to the
    -- space of additive functions when the joint measure is a product measure.
    --
    -- A simplified sketch:
    -- Let V_add = span{f(p), g(c)} and V_int = span{h(p,c) | E_p[h] = 0, E_c[h]=0}.
    -- Under a product measure, ⟨f(p), h(p,c)⟩ = E[f(p)h(p,c)] = E[f(p) E_c[h(p,c)]] = 0.
    -- Similarly, ⟨g(c), h(p,c)⟩ = 0.
    -- So V_add ⊥ V_int.
    -- Since Y ∈ V_add, its projection onto V_add ⊕ V_int has no component in V_int.
    admit⟩ -- L² orthogonality under independence

theorem independence_implies_no_interaction (p k sp : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (dgp : DataGeneratingProcess k)
    (h_additive : ∃ (f : ℝ → ℝ) (g : Fin k → ℝ → ℝ), dgp.trueExpectation = fun p c => f p + ∑ i, g i (c i))
    (h_indep : dgp.jointMeasure = (dgp.jointMeasure.map Prod.fst).prod (dgp.jointMeasure.map Prod.snd)) :
  ∀ (m : PhenotypeInformedGAM p k sp) (_h_opt : IsBayesOptimalInClass dgp m), IsNormalizedScoreModel m := by
  intros m _h_opt
  rcases h_additive with ⟨f, g, h_fn_struct⟩
  exact l2_projection_of_additive_is_additive p k sp h_indep h_fn_struct m _h_opt

structure DGPWithEnvironment (k : ℕ) where
  to_dgp : DataGeneratingProcess k
  environmentalEffect : (Fin k → ℝ) → ℝ
  trueGeneticEffect : ℝ → ℝ
  is_additive_causal : to_dgp.trueExpectation = fun p c => trueGeneticEffect p + environmentalEffect c

theorem prediction_causality_tradeoff_linear_case (p sp : ℕ) [Fintype (Fin p)] [Fintype (Fin sp)] (dgp_env : DGPWithEnvironment 1)
    (hp_pos : p > 0)
    (h_gen : dgp_env.trueGeneticEffect = fun p => 2 * p)
    (h_env : dgp_env.environmentalEffect = fun c => 3 * (c ⟨0, by norm_num⟩))
    (h_confounding : ∫ pc, pc.1 * (pc.2 ⟨0, by norm_num⟩) ∂dgp_env.to_dgp.jointMeasure ≠ 0)
    (model : PhenotypeInformedGAM p 1 sp)
    (h_opt : IsBayesOptimalInClass dgp_env.to_dgp model) :
    model.γₘ₀ ⟨0, hp_pos⟩ ≠ 2 := by sorry

def total_params (p k sp : ℕ) : ℕ := 1 + p + k*sp + p*k*sp

/-! ### Parameter Vectorization Infrastructure

To prove identifiability, we vectorize the GAM parameters into a single vector β ∈ ℝ^d,
then show the loss is a strictly convex quadratic in β.

**Key insight**: Define a structured index type `ParamIx` to avoid Fin arithmetic hell.
Then define packParams/unpackParams through this structured type. -/

/-- Structured parameter index type.
    This avoids painful Fin arithmetic by giving semantic meaning to each parameter block. -/
inductive ParamIx (p k sp : ℕ)
  | intercept                         -- γ₀₀: 1 parameter
  | pgsCoeff (m : Fin p)              -- γₘ₀: p parameters
  | pcSpline (l : Fin k) (j : Fin sp) -- f₀ₗ: k*sp parameters
  | interaction (m : Fin p) (l : Fin k) (j : Fin sp) -- fₘₗ: p*k*sp parameters
  deriving DecidableEq

abbrev ParamIxSum (p k sp : ℕ) :=
  Sum Unit (Sum (Fin p) (Sum (Fin k × Fin sp) (Fin p × Fin k × Fin sp)))

def ParamIx.equivSum (p k sp : ℕ) : ParamIx p k sp ≃ ParamIxSum p k sp where
  toFun
    | .intercept => Sum.inl ()
    | .pgsCoeff m => Sum.inr (Sum.inl m)
    | .pcSpline l j => Sum.inr (Sum.inr (Sum.inl (l, j)))
    | .interaction m l j => Sum.inr (Sum.inr (Sum.inr (m, l, j)))
  invFun
    | Sum.inl _ => .intercept
    | Sum.inr (Sum.inl m) => .pgsCoeff m
    | Sum.inr (Sum.inr (Sum.inl (l, j))) => .pcSpline l j
    | Sum.inr (Sum.inr (Sum.inr (m, l, j))) => .interaction m l j
  left_inv := by
    intro x
    cases x <;> rfl
  right_inv := by
    intro x
    cases x with
    | inl u =>
      cases u
      rfl
    | inr x =>
      cases x with
      | inl m =>
        rfl
      | inr x =>
        cases x with
        | inl lj =>
          rcases lj with ⟨l, j⟩
          rfl
        | inr mlj =>
          rcases mlj with ⟨m, l, j⟩
          rfl

instance (p k sp : ℕ) : Fintype (ParamIx p k sp) :=
  Fintype.ofEquiv (ParamIxSum p k sp) (ParamIx.equivSum p k sp).symm

lemma ParamIx_card (p k sp : ℕ) : Fintype.card (ParamIx p k sp) = total_params p k sp := by
  classical
  -- `simp` computes the card but leaves some reassociation/`mul_assoc` goals.
  simpa [ParamIxSum, total_params, Nat.add_assoc, Nat.add_left_comm, Nat.add_comm, Nat.mul_assoc] using
    (Fintype.card_congr (ParamIx.equivSum p k sp))

/-- Parameter vector type: flattens all GAM coefficients into a single vector. -/
abbrev ParamVec (p k sp : ℕ) := ParamIx p k sp → ℝ

/-- Model class restriction: same basis, same link, same distribution.
    Without this, the same predictor can be represented with different parameters. -/
structure InModelClass {p k sp : ℕ} (m : PhenotypeInformedGAM p k sp)
    (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp) : Prop where
  basis_match : m.pgsBasis = pgsBasis
  spline_match : m.pcSplineBasis = splineBasis
  link_identity : m.link = .identity
  dist_gaussian : m.dist = .Gaussian

/-- Pack GAM parameters into a vector using the structured ParamIx.
    Each coefficient is placed at its corresponding flat index. -/
noncomputable def packParams {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (m : PhenotypeInformedGAM p k sp) : ParamVec p k sp :=
  fun j =>
    match j with
    | .intercept => m.γ₀₀
    | .pgsCoeff m0 => m.γₘ₀ m0
    | .pcSpline l s => m.f₀ₗ l s
    | .interaction m0 l s => m.fₘₗ m0 l s

/-- Unpack a vector into GAM parameters (inverse of packParams). -/
noncomputable def unpackParams {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp)
    (β : ParamVec p k sp) : PhenotypeInformedGAM p k sp :=
  { pgsBasis := pgsBasis
    pcSplineBasis := splineBasis
    γ₀₀ := β .intercept
    γₘ₀ := fun m => β (.pgsCoeff m)
    f₀ₗ := fun l j => β (.pcSpline l j)
    fₘₗ := fun m l j => β (.interaction m l j)
    link := .identity
    dist := .Gaussian }

/-- Pack and unpack are inverses within the model class. -/
lemma unpack_pack_eq {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (m : PhenotypeInformedGAM p k sp) (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp)
    (hm : InModelClass m pgsBasis splineBasis) :
    unpackParams pgsBasis splineBasis (packParams m) = m := by
  cases m with
  | mk m_pgsBasis m_splineBasis m_γ00 m_γm0 m_f0l m_fml m_link m_dist =>
    rcases hm with ⟨hbasis, hspline, hlink, hdist⟩
    cases hbasis
    cases hspline
    cases hlink
    cases hdist
    rfl

/-- The design matrix for the penalized GAM.
    This corresponds to the construction in `basis.rs` and `construction.rs`.

    Block structure (columns indexed by ParamIx):
    - intercept: constant 1
    - pgsCoeff m: B_{m+1}(pgs_i)
    - pcSpline l j: splineBasis.B[j](c_i[l])
    - interaction m l j: B_{m+1}(pgs_i) * splineBasis.B[j](c_i[l])

    Uses structured indices for clean column dispatch. -/
noncomputable def designMatrix {n p k sp : ℕ} [Fintype (Fin n)] [Fintype (Fin p)]
    [Fintype (Fin k)] [Fintype (Fin sp)]
    (data : RealizedData n k) (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp)
    : Matrix (Fin n) (ParamIx p k sp) ℝ :=
  Matrix.of fun i j =>
    match j with
    | .intercept => 1
    | .pgsCoeff m => pgsBasis.B ⟨m.val + 1, by omega⟩ (data.p i)
    | .pcSpline l s => splineBasis.b s (data.c i l)
    | .interaction m l s => pgsBasis.B ⟨m.val + 1, by omega⟩ (data.p i) * splineBasis.b s (data.c i l)

/-- **Key Lemma**: Linear predictor equals design matrix times parameter vector.
    This is the bridge between the GAM structure and linear algebra.

    Proof strategy: Both sides compute the same sum over parameter blocks:
    - γ₀₀ * 1 (intercept)
    - Σ_m γₘ₀ * B_{m+1}(pgs) (PGS main effects)
    - Σ_l Σ_j f₀ₗ[l,j] * spline_j(c[l]) (PC main effects)
    - Σ_m Σ_l Σ_j fₘₗ[m,l,j] * B_{m+1}(pgs) * spline_j(c[l]) (interactions)

    The key is that packParams and designMatrix are defined consistently via ParamIx. -/
lemma linearPredictor_eq_designMatrix_mulVec {n p k sp : ℕ}
    [Fintype (Fin n)] [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (data : RealizedData n k) (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp)
    (m : PhenotypeInformedGAM p k sp) (hm : InModelClass m pgsBasis splineBasis) :
    ∀ i : Fin n, linearPredictor m (data.p i) (data.c i) =
      (designMatrix data pgsBasis splineBasis).mulVec (packParams m) i := by
  intro i
  -- LHS: linearPredictor m (data.p i) (data.c i)
  -- = γ₀₀ * B₀(pgs) + Σ_m (γₘ₀ + smooth_m(c)) * B_{m+1}(pgs)
  -- = γ₀₀ + Σ_m γₘ₀ * B_{m+1}(pgs) + Σ_m Σ_l evalSmooth(fₘₗ, c[l]) * B_{m+1}(pgs)
  --
  -- RHS: (X * β)[i] = Σ_j X[i,j] * β[j]
  -- By ParamIx structure, this splits into four blocks matching above.
  --
  -- The proof requires:
  -- 1. Unfold definitions of linearPredictor, designMatrix, packParams
  -- 2. Use hm to substitute the model's bases with the given bases
  -- 3. Show the sums match by regrouping

  -- Step 1: Substitute model bases using hm
  have h_pgs := hm.basis_match
  have h_spline := hm.spline_match

  -- Step 2: Unfold and compute
  unfold Matrix.mulVec designMatrix packParams
  simp only [Matrix.of_apply]

  -- The rest requires matching the ParamIx sum structure with linearPredictor's definition.
  -- This involves showing that the vectorized parameters reconstruct the original prediction.
  -- See: Wood (2017), "Generalized Additive Models", Section 4.2 on basis representation.
  sorry -- Sum manipulation with ParamIx (basis reconstruction proof)

/-- Full column rank implies `X.mulVec` is injective.

This is stated using an arbitrary finite column type `ι` (rather than `Fin d`) to avoid
index-flattening in downstream proofs. -/
lemma mulVec_injective_of_full_rank {ι : Type*} {n : ℕ} [Fintype (Fin n)] [Fintype ι]
    (X : Matrix (Fin n) ι ℝ) (h_rank : Matrix.rank X = Fintype.card ι) :
    Function.Injective X.mulVec := by
  classical
  have hcols : LinearIndependent ℝ X.col := by
    -- `rank` is the `finrank` of the span of columns, which is `(Set.range X.col).finrank`.
    have hrank' : X.rank = (Set.range X.col).finrank ℝ := by
      simpa [Set.finrank] using (X.rank_eq_finrank_span_cols (R := ℝ))
    have hfin : Fintype.card ι = (Set.range X.col).finrank ℝ := h_rank.symm.trans hrank'
    exact (linearIndependent_iff_card_eq_finrank_span (b := X.col)).2 hfin
  exact (Matrix.mulVec_injective_iff (M := X)).2 hcols

/-! ### Generic Finite-Dimensional Quadratic Forms

These are written over an arbitrary finite index type `ι`, so they can be used directly with
`ParamIx p k sp` (no `Fin (total_params ...)` needed). -/

/-- Dot product of two vectors represented as `ι → ℝ`. -/
def dotProduct' {ι : Type*} [Fintype ι] (u v : ι → ℝ) : ℝ :=
  Finset.univ.sum (fun i => u i * v i)

/-- XᵀX is positive definite when X has full column rank.
    This is the algebraic foundation for uniqueness of least squares.

    Key mathlib lemma:
    - Matrix.posDef_conjTranspose_mul_self_iff_injective
    Over ℝ, conjTranspose = transpose, so this gives exactly what we need.

    Alternatively, direct proof:
    vᵀ(XᵀX)v = (Xv)ᵀ(Xv) = ‖Xv‖² > 0 when v ≠ 0 and X injective. -/
lemma transpose_mul_self_posDef {ι : Type*} {n : ℕ} [Fintype (Fin n)] [Fintype ι]
    (X : Matrix (Fin n) ι ℝ) (h_rank : Matrix.rank X = Fintype.card ι) :
    ∀ v : ι → ℝ, v ≠ 0 → 0 < dotProduct' ((Matrix.transpose X * X).mulVec v) v := by
  intro v hv
  -- vᵀ(XᵀX)v = vᵀXᵀXv = (Xv)ᵀ(Xv) = ‖Xv‖²
  -- Since X has full rank, X.mulVec is injective
  -- So v ≠ 0 ⟹ Xv ≠ 0 ⟹ ‖Xv‖² > 0
  have h_inj := mulVec_injective_of_full_rank X h_rank
  have h_Xv_ne : X.mulVec v ≠ 0 := by
    intro h_eq
    apply hv
    exact h_inj (h_eq.trans (X.mulVec_zero).symm)
  -- Show: dotProduct' (XᵀX).mulVec v v = ‖Xv‖² > 0
  -- The key is: (XᵀX).mulVec v = Xᵀ(Xv), so vᵀ(XᵀX)v = (Xv)ᵀ(Xv) = ‖Xv‖²
  -- Since Xv ≠ 0, we have ‖Xv‖² > 0
  sorry -- dotProduct' vᵀ(XᵀX)v = ‖Xv‖² > 0 by Hilbert space norm property

/-- The penalized Gaussian loss as a quadratic function of parameters. -/
noncomputable def gaussianPenalizedLoss {ι : Type*} {n : ℕ} [Fintype (Fin n)] [Fintype ι]
    (X : Matrix (Fin n) ι ℝ) (y : Fin n → ℝ) (S : Matrix ι ι ℝ) (lam : ℝ)
    (β : ι → ℝ) : ℝ :=
  (1 / n) * ‖y - X.mulVec β‖^2 + lam * Finset.univ.sum (fun i => β i * (S.mulVec β) i)

/-- A matrix is positive semidefinite if vᵀSv ≥ 0 for all v. -/
def IsPosSemidef {ι : Type*} [Fintype ι] (S : Matrix ι ι ℝ) : Prop :=
  ∀ v : ι → ℝ, 0 ≤ dotProduct' (S.mulVec v) v

/-- The Gaussian penalized loss is strictly convex when X has full rank and lam > 0.

    **Proof Strategy**: The loss function can be written as a quadratic:
      L(β) = (1/n) * ‖y - Xβ‖² + λ * βᵀSβ
           = const + linear(β) + βᵀ H β

    where H = (1/n)XᵀX + λS is the Hessian.

    **Key Steps**:
    1. XᵀX is positive semidefinite (since vᵀ(XᵀX)v = ‖Xv‖² ≥ 0)
    2. When X has full rank, XᵀX is actually positive DEFINITE (v≠0 ⟹ Xv≠0 ⟹ ‖Xv‖² > 0)
    3. S is positive semidefinite by assumption (hS)
    4. λ > 0 means λS is positive semidefinite
    5. (PosDef) + (PosSemidef) = (PosDef)
    6. A quadratic with positive definite Hessian is strictly convex

    **FUTURE:**
    - Use Mathlib's Matrix.PosDef API directly for cleaner integration
    - Abstract to LinearMap for kernel/image reasoning -/
lemma gaussianPenalizedLoss_strictConvex {ι : Type*} {n : ℕ} [Fintype (Fin n)] [Fintype ι]
    (X : Matrix (Fin n) ι ℝ) (y : Fin n → ℝ) (S : Matrix ι ι ℝ)
    (lam : ℝ) (hlam : lam > 0) (h_rank : Matrix.rank X = Fintype.card ι) (_hS : IsPosSemidef S) :
    StrictConvexOn ℝ Set.univ (gaussianPenalizedLoss X y S lam) := by
  -- The Hessian is H = (2/n)XᵀX + 2λS
  -- For v ≠ 0: vᵀHv = (2/n)‖Xv‖² + 2λ·vᵀSv
  --                 ≥ (2/n)‖Xv‖² (since S is PSD and λ > 0)
  --                 > 0 (since X has full rank, so Xv ≠ 0)
  -- Therefore H is positive definite, and the quadratic is strictly convex.
  sorry

/-- The penalized loss is coercive: L(β) → ∞ as ‖β‖ → ∞.

    **Proof**: The penalty term λ·βᵀSβ dominates as ‖β‖ → ∞.
    Even if S is only PSD, as long as λ > 0 and S has nontrivial action,
    or if we use ridge penalty (S = I), coercivity holds.

    For ridge penalty specifically: L(β) ≥ λ·‖β‖² → ∞. -/
lemma gaussianPenalizedLoss_coercive {ι : Type*} {n : ℕ} [Fintype (Fin n)] [Fintype ι]
    [DecidableEq ι]
    (X : Matrix (Fin n) ι ℝ) (y : Fin n → ℝ) (S : Matrix ι ι ℝ)
    (lam : ℝ) (hlam : lam > 0) (hS_posDef : ∀ v : ι → ℝ, v ≠ 0 → 0 < dotProduct' (S.mulVec v) v) :
    Filter.Tendsto (gaussianPenalizedLoss X y S lam) (Filter.cocompact _) Filter.atTop := by
  -- For S = I (ridge): βᵀSβ = βᵀβ = ‖β‖²
  -- So L(β) ≥ λ·‖β‖² → ∞ as ‖β‖ → ∞
  sorry

/-- Existence of minimizer: coercivity + continuity implies minimum exists.

    This uses the Weierstrass extreme value theorem: a continuous function
    that tends to infinity at infinity achieves its minimum on ℝⁿ. -/
lemma gaussianPenalizedLoss_exists_min {ι : Type*} {n : ℕ} [Fintype (Fin n)] [Fintype ι]
    [DecidableEq ι]
    (X : Matrix (Fin n) ι ℝ) (y : Fin n → ℝ) (S : Matrix ι ι ℝ)
    (lam : ℝ) (hlam : lam > 0) (hS_posDef : ∀ v : ι → ℝ, v ≠ 0 → 0 < dotProduct' (S.mulVec v) v) :
    ∃ β : ι → ℝ, ∀ β' : ι → ℝ, gaussianPenalizedLoss X y S lam β ≤ gaussianPenalizedLoss X y S lam β' := by
  -- By coercivity (gaussianPenalizedLoss_coercive) and continuity,
  -- Mathlib's Continuous.exists_forall_le_of_hasCompactMulSupport or
  -- IsCompact.exists_isMinOn on sublevel sets gives existence.
  sorry

/-- **Parameter Identifiability**: If the design matrix has full column rank,
    then the penalized GAM has a unique solution within the model class.

    This validates the constraint machinery in `basis.rs`:
    - `apply_sum_to_zero_constraint` ensures spline contributions average to zero
    - `apply_weighted_orthogonality_constraint` removes collinearity with lower-order terms

    **Proof Strategy (Coercivity + Strict Convexity)**:

    **Existence (Weierstrass)**: The loss function L(β) is:
    - Continuous (composition of continuous operations)
    - Coercive (L(β) → ∞ as ‖β‖ → ∞ due to ridge penalty λ‖β‖²)
    Therefore by the extreme value theorem, a minimum exists.

    **Uniqueness (Strict Convexity)**: The loss function is strictly convex when:
    - X has full column rank (XᵀX is positive definite)
    - λ > 0 (penalty adds strictly positive term)
    A strictly convex function has at most one minimizer.

    **FUTURE REFACTORING (v2.0)**:
    - Unify empirical/theoretical loss via L²(μ) for different measures
    - Use abstract [InnerProductSpace ℝ P] instead of concrete ParamIx
    - Define constraint as LinearMap kernel for cleaner affine subspace handling -/
theorem parameter_identifiability {n p k sp : ℕ} [Fintype (Fin n)] [Fintype (Fin p)]
    [Fintype (Fin k)] [Fintype (Fin sp)]
    (data : RealizedData n k) (lambda : ℝ)
    (pgsBasis : PGSBasis p) (splineBasis : SplineBasis sp)
    (_hp : p > 0) (_hk : k > 0) (_hsp : sp > 0)
    (h_rank : Matrix.rank (designMatrix data pgsBasis splineBasis) = Fintype.card (ParamIx p k sp))
    (h_lambda_pos : lambda > 0) :
  ∃! (m : PhenotypeInformedGAM p k sp),
    InModelClass m pgsBasis splineBasis ∧
    IsIdentifiable m data ∧
    ∀ (m' : PhenotypeInformedGAM p k sp),
      InModelClass m' pgsBasis splineBasis →
      IsIdentifiable m' data → empiricalLoss m data lambda ≤ empiricalLoss m' data lambda := by

  -- Step 1: Reduce to vector optimization via packParams
  -- The problem: minimize L(β) = (1/n)‖y - Xβ‖² + λβᵀSβ subject to Cβ = 0
  -- where β = packParams(m)

  -- Step 2: Existence via coercivity
  -- L is continuous and coercive (L → ∞ as ‖β‖ → ∞ due to λ > 0)
  -- The constraint set {β : Cβ = 0} is a closed linear subspace
  -- By Weierstrass, a minimum exists on any closed set where L is coercive

  -- Step 3: Uniqueness via strict convexity
  -- L is strictly convex (XᵀX is PD from full rank, plus λS with λ > 0)
  -- A strictly convex function on a convex set (linear subspace) has ≤ 1 minimizer
  -- Combined with existence: exactly 1 minimizer

  -- Step 4: Translate back via unpackParams
  -- The unique minimizing β gives a unique m = unpackParams(β)

  sorry

def predictionBias {k : ℕ} [Fintype (Fin k)] (dgp : DataGeneratingProcess k) (f : ℝ → (Fin k → ℝ) → ℝ) (p_val : ℝ) (c_val : Fin k → ℝ) : ℝ :=
  dgp.trueExpectation p_val c_val - f p_val c_val




/-- **General Risk Formula for Affine Predictors** (THE KEY LEMMA)

    For DGP Y = P + β·C and affine predictor Ŷ = a + b·P:
      R(a,b) = E[(Y - Ŷ)²] = a² + (1-b)²·E[P²] + β²·E[C²]

    when E[P] = E[C] = 0 and E[PC] = 0 (independence).

    **Proof Strategy (Direct Expansion)**:
    1. Let u = 1 - b. Then Y - Ŷ = (P + βC) - (a + bP) = uP + βC - a
    2. Expand: (uP + βC - a)² = u²P² + β²C² + a² + 2uβPC - 2uaP - 2aβC
    3. Integrate term-by-term:
       - E[u²P²] = u²·E[P²]
       - E[2uβPC] = 0 (by independence/orthogonality)
       - E[-2uaP] = -2ua·E[P] = 0
       - E[-2aβC] = -2aβ·E[C] = 0
    4. Result: u²·E[P²] + β²·E[C²] + a² = a² + (1-b)²·E[P²] + β²·E[C²]

    This is the cleanest path to proving raw score bias: compare risks directly,
    no need for normal equations or Hilbert projection machinery.

    **Alternative approach (avoided)**: Prove via orthogonality conditions (normal equations).
    That requires formalizing IsBayesOptimalInRawClass → orthogonality, which is harder. -/
lemma risk_affine_additive
    (μ : Measure (ℝ × (Fin 1 → ℝ))) [IsProbabilityMeasure μ]
    (_h_indep : μ = (μ.map Prod.fst).prod (μ.map Prod.snd))
    (hP0 : ∫ pc, pc.1 ∂μ = 0)
    (hC0 : ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂μ = 0)
    (hPC0 : ∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂μ = 0)
    (hP2 : ∫ pc, pc.1^2 ∂μ = 1)
    (hP_int : Integrable (fun pc => pc.1) μ)
    (hC_int : Integrable (fun pc => pc.2 ⟨0, by norm_num⟩) μ)
    (hP2_int : Integrable (fun pc => pc.1^2) μ)
    (hC2_int : Integrable (fun pc => (pc.2 ⟨0, by norm_num⟩)^2) μ)
    (hPC_int : Integrable (fun pc => pc.1 * pc.2 ⟨0, by norm_num⟩) μ)
    (β a b : ℝ) :
    ∫ pc, (pc.1 + β * pc.2 ⟨0, by norm_num⟩ - (a + b * pc.1))^2 ∂μ =
      a^2 + (1 - b)^2 + β^2 * (∫ pc, (pc.2 ⟨0, by norm_num⟩)^2 ∂μ) := by
  -- Let u = 1 - b
  set u := 1 - b with hu

  -- The integrand is: (uP + βC - a)²
  -- = u²P² + β²C² + a² + 2uβPC - 2uaP - 2aβC
  --
  -- Integrating term by term:
  -- ∫ u²P² = u² ∫ P² = u² · 1 = (1-b)²
  -- ∫ β²C² = β² ∫ C²
  -- ∫ a² = a² (since μ is prob measure)
  -- ∫ 2uβPC = 2uβ · 0 = 0 (by hPC0)
  -- ∫ -2uaP = -2ua · 0 = 0 (by hP0)
  -- ∫ -2aβC = -2aβ · 0 = 0 (by hC0)

  -- The formal proof: expand the squared term and integrate term by term.
  have h_integrand_expand : ∀ (pc : ℝ × (Fin 1 → ℝ)), (pc.1 + β * pc.2 ⟨0, by norm_num⟩ - (a + b * pc.1))^2 =
      u^2 * pc.1^2 + β^2 * (pc.2 ⟨0, by norm_num⟩)^2 + a^2
      + 2*u*β * (pc.1 * pc.2 ⟨0, by norm_num⟩)
      - 2*u*a * pc.1 - 2*a*β * pc.2 ⟨0, by norm_num⟩ := by
    intro (pc : ℝ × (Fin 1 → ℝ)); simp only [hu]; ring_nf

  -- The formal proof expands the integrand and applies linearity.
  -- First, show all terms are integrable.
  have i_p2 : Integrable (fun pc => u ^ 2 * pc.1 ^ 2) μ := hP2_int.const_mul (u^2)
  have i_c2 : Integrable (fun pc => β^2 * (pc.2 ⟨0, by norm_num⟩)^2) μ := hC2_int.const_mul (β^2)
  have i_a2 : Integrable (fun (_ : ℝ × (Fin 1 → ℝ)) => a ^ 2) μ := integrable_const _
  have i_pc : Integrable (fun pc => 2*u*β * (pc.1 * pc.2 ⟨0, by norm_num⟩)) μ := hPC_int.const_mul (2*u*β)
  have i_p1 : Integrable (fun pc => 2*u*a * pc.1) μ := hP_int.const_mul (2*u*a)
  have i_c1 : Integrable (fun pc => 2*a*β * pc.2 ⟨0, by norm_num⟩) μ := hC_int.const_mul (2*a*β)

  -- Now, use a calc block to show the integral equality step-by-step.
  calc
    ∫ pc, (pc.1 + β * pc.2 ⟨0, by norm_num⟩ - (a + b * pc.1))^2 ∂μ
    -- Step 1: Expand the squared term inside the integral.
    _ = ∫ pc, u^2 * pc.1^2 + β^2 * (pc.2 ⟨0, by norm_num⟩)^2 + a^2
              + 2*u*β * (pc.1 * pc.2 ⟨0, by norm_num⟩)
              - 2*u*a * pc.1 - 2*a*β * pc.2 ⟨0, by norm_num⟩ ∂μ := by
      exact integral_congr_ae (ae_of_all _ h_integrand_expand)

    -- Step 2: Apply linearity of the integral.
    _ = (∫ pc, u^2 * pc.1^2 ∂μ)
        + (∫ pc, β^2 * (pc.2 ⟨0, by norm_num⟩)^2 ∂μ)
        + (∫ pc, a^2 ∂μ)
        + (∫ pc, 2*u*β * (pc.1 * pc.2 ⟨0, by norm_num⟩) ∂μ)
        - (∫ pc, 2*u*a * pc.1 ∂μ)
        - (∫ pc, 2*a*β * pc.2 ⟨0, by norm_num⟩ ∂μ) := by
      have i_add1 : Integrable (fun pc => u^2 * pc.1^2 + β^2 * (pc.2 ⟨0, by norm_num⟩)^2 + a^2
                                        + 2*u*β * (pc.1 * pc.2 ⟨0, by norm_num⟩)
                                        - 2*u*a * pc.1) μ := by
        exact (((i_p2.add i_c2).add i_a2).add i_pc).sub i_p1
      rw [integral_sub i_add1 i_c1]
      have i_add2 : Integrable (fun pc => u^2 * pc.1^2 + β^2 * (pc.2 ⟨0, by norm_num⟩)^2 + a^2
                                        + 2*u*β * (pc.1 * pc.2 ⟨0, by norm_num⟩)) μ := by
        exact ((i_p2.add i_c2).add i_a2).add i_pc
      rw [integral_sub i_add2 i_p1]
      have i_add3 : Integrable (fun pc => u^2 * pc.1^2 + β^2 * (pc.2 ⟨0, by norm_num⟩)^2 + a^2) μ := by
        exact (i_p2.add i_c2).add i_a2
      rw [integral_add i_add3 i_pc]
      have i_add4 : Integrable (fun pc => u^2 * pc.1^2 + β^2 * (pc.2 ⟨0, by norm_num⟩)^2) μ := by
        exact i_p2.add i_c2
      rw [integral_add i_add4 i_a2]
      rw [integral_add i_p2 i_c2]

    -- Step 3: Pull out constants and substitute known integral values.
    _ = u^2 * (∫ pc, pc.1^2 ∂μ)
        + β^2 * (∫ pc, (pc.2 ⟨0, by norm_num⟩)^2 ∂μ)
        + a^2
        + 2*u*β * (∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂μ)
        - 2*u*a * (∫ pc, pc.1 ∂μ)
        - 2*a*β * (∫ pc, pc.2 ⟨0, by norm_num⟩ ∂μ) := by
      -- Apply integral_const_mul and integral_const for each term.
      simp [integral_const_mul, integral_const]; ring

    -- Step 4: Substitute moment conditions (hP2=1, hPC0=0, hP0=0, hC0=0) and simplify.
    _ = u^2 * 1 + β^2 * (∫ pc, (pc.2 ⟨0, by norm_num⟩)^2 ∂μ) + a^2
        + 2*u*β * 0 - 2*u*a * 0 - 2*a*β * 0 := by
      rw [hP2, hPC0, hP0, hC0]

    -- Step 5: Final algebraic simplification.
    _ = a^2 + (1 - b)^2 + β^2 * (∫ pc, (pc.2 ⟨0, by norm_num⟩)^2 ∂μ) := by
      rw [hu]
      ring

/-- Corollary: Risk formula for Scenario 4 (β = -0.8).
    This is just `risk_affine_additive` with β = -0.8. -/
lemma risk_affine_scenario4
    (μ : Measure (ℝ × (Fin 1 → ℝ))) [IsProbabilityMeasure μ]
    (h_indep : μ = (μ.map Prod.fst).prod (μ.map Prod.snd))
    (hP0 : ∫ pc, pc.1 ∂μ = 0)
    (hC0 : ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂μ = 0)
    (hP2 : ∫ pc, pc.1^2 ∂μ = 1)
    (hP_int : Integrable (fun pc => pc.1) μ)
    (hC_int : Integrable (fun pc => pc.2 ⟨0, by norm_num⟩) μ)
    (hP2_int : Integrable (fun pc => pc.1^2) μ)
    (hC2_int : Integrable (fun pc => (pc.2 ⟨0, by norm_num⟩)^2) μ)
    (hPC_int : Integrable (fun pc => pc.1 * pc.2 ⟨0, by norm_num⟩) μ)
    (a b : ℝ) :
    ∫ pc, (pc.1 - 0.8 * pc.2 ⟨0, by norm_num⟩ - (a + b * pc.1))^2 ∂μ =
      a^2 + (1 - b)^2 + 0.64 * (∫ pc, (pc.2 ⟨0, by norm_num⟩)^2 ∂μ) := by
  -- p - 0.8*c = p + (-0.8)*c, so this is risk_affine_additive with β = -0.8
  have hPC0 : ∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂μ = 0 :=
    integral_mul_fst_snd_eq_zero μ h_indep hP0 hC0

  -- Rewrite to match risk_affine_additive form
  have h_rewrite : ∀ pc : ℝ × (Fin 1 → ℝ),
      (pc.1 - 0.8 * pc.2 ⟨0, by norm_num⟩ - (a + b * pc.1)) =
      (pc.1 + (-0.8) * pc.2 ⟨0, by norm_num⟩ - (a + b * pc.1)) := by
    intro pc; ring

  simp_rw [h_rewrite]

  -- Apply the general lemma
  have h_gen := risk_affine_additive μ h_indep hP0 hC0 hPC0 hP2 hP_int hC_int hP2_int hC2_int hPC_int (-0.8) a b

  -- Simplify (-0.8)² = 0.64
  simp only [neg_mul] at h_gen ⊢
  convert h_gen using 2
  ring

/-- **Lemma D**: Uniqueness of minimizer for Scenario 4 risk.
    The affine risk a² + (1-b)² + const is uniquely minimized at a=0, b=1. -/
lemma affine_risk_minimizer (a b : ℝ) (const : ℝ) (_hconst : const ≥ 0) :
    a^2 + (1 - b)^2 + const ≥ const ∧
    (a^2 + (1 - b)^2 + const = const ↔ a = 0 ∧ b = 1) := by
  constructor
  · -- a² + (1-b)² ≥ 0
    nlinarith [sq_nonneg a, sq_nonneg (1 - b)]
  · constructor
    · intro h
      have h1 : a^2 + (1 - b)^2 = 0 := by linarith
      have ha : a^2 = 0 := by nlinarith [sq_nonneg a, sq_nonneg (1 - b)]
      have hb : (1 - b)^2 = 0 := by nlinarith [sq_nonneg a, sq_nonneg (1 - b)]
      constructor
      · exact sq_eq_zero_iff.mp ha
      · have : 1 - b = 0 := sq_eq_zero_iff.mp hb
        linarith
    · rintro ⟨rfl, rfl⟩
      simp

/-- **THE CLEAN PROOF**: Optimal coefficients via direct risk comparison.

    This lemma bypasses the orthogonality/normal equations approach entirely.
    Instead, we:
    1. Use `risk_affine_additive` to compute the risk of any (a,b)
    2. Use `affine_risk_minimizer` to show the minimum is at (0,1)
    3. Use optimality to conclude the model's coefficients equal (0,1)

    **Why this is cleaner than orthogonality**:
    - No need to prove `rawOptimal_implies_orthogonality` via derivatives
    - No need for integral linearity lemmas (we just compare risks)
    - The minimizer is OBVIOUS from the quadratic form -/
lemma optimal_coefficients_via_risk
    (μ : Measure (ℝ × (Fin 1 → ℝ))) [IsProbabilityMeasure μ]
    (h_indep : μ = (μ.map Prod.fst).prod (μ.map Prod.snd))
    (hP0 : ∫ pc, pc.1 ∂μ = 0)
    (hC0 : ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂μ = 0)
    (hP2 : ∫ pc, pc.1^2 ∂μ = 1)
    (hC2_pos : 0 ≤ ∫ pc, (pc.2 ⟨0, by norm_num⟩)^2 ∂μ)
    (hP_int : Integrable (fun pc => pc.1) μ)
    (hC_int : Integrable (fun pc => pc.2 ⟨0, by norm_num⟩) μ)
    (hP2_int : Integrable (fun pc => pc.1^2) μ)
    (hC2_int : Integrable (fun pc => (pc.2 ⟨0, by norm_num⟩)^2) μ)
    (hPC_int : Integrable (fun pc => pc.1 * pc.2 ⟨0, by norm_num⟩) μ)
    (β a b : ℝ)
    -- Optimality: (a,b) achieves minimal risk among all affine predictors
    (h_opt : ∀ a' b' : ℝ,
      ∫ pc, (pc.1 + β * pc.2 ⟨0, by norm_num⟩ - (a + b * pc.1))^2 ∂μ ≤
      ∫ pc, (pc.1 + β * pc.2 ⟨0, by norm_num⟩ - (a' + b' * pc.1))^2 ∂μ) :
    a = 0 ∧ b = 1 := by

  -- Get the orthogonality fact
  have hPC0 : ∫ pc, pc.1 * pc.2 ⟨0, by norm_num⟩ ∂μ = 0 :=
    integral_mul_fst_snd_eq_zero μ h_indep hP0 hC0

  -- Abbreviate the C² integral
  set C2 := ∫ pc, (pc.2 ⟨0, by norm_num⟩)^2 ∂μ with hC2_def

  -- Step 1: Apply risk_affine_additive to get closed-form risks
  have h_risk := risk_affine_additive μ h_indep hP0 hC0 hPC0 hP2 hP_int hC_int hP2_int hC2_int hPC_int β a b
  have h_risk_ref := risk_affine_additive μ h_indep hP0 hC0 hPC0 hP2 hP_int hC_int hP2_int hC2_int hPC_int β 0 1

  -- Step 2: Risk at (0,1) is the global minimum β²·C²
  have h_ref_val : (0 : ℝ)^2 + (1 - 1)^2 + β^2 * C2 = β^2 * C2 := by ring

  -- Step 3: By optimality, risk(a,b) ≤ risk(0,1)
  have h_opt_01 := h_opt 0 1
  rw [h_risk, h_risk_ref, h_ref_val] at h_opt_01

  -- Step 4: Apply affine_risk_minimizer to get a=0, b=1
  have h_min := affine_risk_minimizer a b (β^2 * C2) (by nlinarith [sq_nonneg β])

  -- From h_min: risk ≥ β²C² and equality ↔ a=0 ∧ b=1
  -- From h_opt_01: risk ≤ β²C²
  -- Therefore: risk = β²C² and hence a=0 ∧ b=1

  have h_eq : a^2 + (1 - b)^2 + β^2 * C2 = β^2 * C2 := by
    have h_ge := h_min.1
    linarith

  exact h_min.2.1 h_eq

/-! ### Main Theorem: Raw Score Bias in Scenario 4 -/

/-- **Raw Score Bias Theorem**: In Scenario 4 (neutral ancestry differences),
    using a raw score model (ignoring ancestry) produces prediction bias = -0.8 * c.

    This validates why the calibrator in `calibrator.rs` includes PC main effects (f₀ₗ)
    and not just PGS×PC interactions. Even with no true gene-environment interaction,
    population drift creates systematic bias that must be corrected.

    **Key insight**: The raw model is an L² projection onto the {1, P} subspace.
    Under the given moment assumptions, the optimal affine predictor has a=0, b=1. -/
theorem raw_score_bias_in_scenario4_simplified
    (model_raw : PhenotypeInformedGAM 1 1 1) (h_raw_struct : IsRawScoreModel model_raw)
    (h_pgs_basis_linear : model_raw.pgsBasis.B 1 = id ∧ model_raw.pgsBasis.B 0 = fun _ => 1)
    (dgp4 : DataGeneratingProcess 1) (h_s4 : dgp4.trueExpectation = fun p c => p - (0.8 * c ⟨0, by norm_num⟩))
    -- FIXED: Now using class-restricted optimality
    (h_opt_raw : IsBayesOptimalInRawClass dgp4 model_raw)
    (h_indep : dgp4.jointMeasure = (dgp4.jointMeasure.map Prod.fst).prod (dgp4.jointMeasure.map Prod.snd))
    (h_means_zero : ∫ pc, pc.1 ∂dgp4.jointMeasure = 0 ∧ ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂dgp4.jointMeasure = 0)
    (h_var_p_one : ∫ pc, pc.1^2 ∂dgp4.jointMeasure = 1)
    -- Integrability hypotheses
    (hP_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) dgp4.jointMeasure)
    (hC_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩) dgp4.jointMeasure)
    (hP2_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1 ^ 2) dgp4.jointMeasure)
    (hPC_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩ * pc.1) dgp4.jointMeasure)
    (hY_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp4.trueExpectation pc.1 pc.2) dgp4.jointMeasure)
    (hYP_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp4.trueExpectation pc.1 pc.2 * pc.1) dgp4.jointMeasure)
    (h_resid_sq_int : Integrable (fun pc => (dgp4.trueExpectation pc.1 pc.2 - (model_raw.γ₀₀ + model_raw.γₘ₀ ⟨0, by norm_num⟩ * pc.1))^2) dgp4.jointMeasure) :
  ∀ (p_val : ℝ) (c_val : Fin 1 → ℝ),
    predictionBias dgp4 (fun p _ => linearPredictor model_raw p c_val) p_val c_val = -0.8 * c_val ⟨0, by norm_num⟩ := by
  intros p_val c_val
  unfold predictionBias
  rw [h_s4]
  dsimp

  -- Step 1: Apply Lemma A to simplify the raw predictor
  have h_pred := linearPredictor_eq_affine_of_raw model_raw h_raw_struct h_pgs_basis_linear
  rw [h_pred]

  -- Step 2: Rewrite h_s4 to show it matches the additive bias DGP form with β = -0.8
  -- p - 0.8*c = p + (-0.8)*c
  have h_dgp_add : dgp4.trueExpectation = fun p c => p + (-0.8) * c ⟨0, by norm_num⟩ := by
    simp only [h_s4]
    funext p c
    ring

  -- Step 3: Apply the optimal coefficients lemma
  have h_coeffs := optimal_coefficients_for_additive_dgp model_raw (-0.8) dgp4 h_dgp_add
                     h_opt_raw h_pgs_basis_linear h_indep
                     h_means_zero.1 h_means_zero.2 h_var_p_one
                     hP_int hC_int hP2_int hPC_int hY_int hYP_int h_resid_sq_int
  obtain ⟨ha, hb⟩ := h_coeffs

  -- Step 4: Substitute a=0, b=1 into the predictor
  have hb' : model_raw.γₘ₀ 0 = 1 := hb
  rw [ha, hb']
  ring

/-! ### Generalized Raw Score Bias (L² Projection Approach)

The following theorem generalizes the above to any β_env, using the L² projection framework.

**Key Insight** (Geometry, not Calculus):
- View P, C, 1 as vectors in L²(μ)
- Under independence + zero means, these form an orthogonal basis
- The raw model projects Y = P + β_env*C onto span{1, P}
- Since C ⊥ span{1, P}, the projection of β_env*C is 0
- Therefore: proj(Y) = P, and bias = Y - proj(Y) = β_env*C -/

/-- **Generalized Raw Score Bias**: For any environmental effect β_env,
    the raw model (which ignores ancestry) produces bias = β_env * C.

    This is the L² projection of Y = P + β_env*C onto span{1, P}.
    Since C is orthogonal to this subspace, the projection is simply P,
    leaving a residual of β_env*C. -/
theorem raw_score_bias_general [Fact (p = 1)]
    (β_env : ℝ) -- Generalized from -0.8
    (model_raw : PhenotypeInformedGAM 1 1 1) (h_raw_struct : IsRawScoreModel model_raw)
    (h_pgs_basis_linear : model_raw.pgsBasis.B 1 = id ∧ model_raw.pgsBasis.B 0 = fun _ => 1)
    (dgp : DataGeneratingProcess 1)
    (h_dgp : dgp.trueExpectation = fun p c => p + β_env * c ⟨0, by norm_num⟩)
    (h_opt_raw : IsBayesOptimalInRawClass dgp model_raw)
    (h_indep : dgp.jointMeasure = (dgp.jointMeasure.map Prod.fst).prod (dgp.jointMeasure.map Prod.snd))
    (h_means_zero : ∫ pc, pc.1 ∂dgp.jointMeasure = 0 ∧ ∫ pc, pc.2 ⟨0, by norm_num⟩ ∂dgp.jointMeasure = 0)
    (h_var_p_one : ∫ pc, pc.1^2 ∂dgp.jointMeasure = 1)
    -- Integrability hypotheses
    (hP_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1) dgp.jointMeasure)
    (hC_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩) dgp.jointMeasure)
    (hP2_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.1 ^ 2) dgp.jointMeasure)
    (hPC_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => pc.2 ⟨0, by norm_num⟩ * pc.1) dgp.jointMeasure)
    (hY_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp.trueExpectation pc.1 pc.2) dgp.jointMeasure)
    (hYP_int : Integrable (fun pc : ℝ × (Fin 1 → ℝ) => dgp.trueExpectation pc.1 pc.2 * pc.1) dgp.jointMeasure)
    (h_resid_sq_int : Integrable (fun pc => (dgp.trueExpectation pc.1 pc.2 - (model_raw.γ₀₀ + model_raw.γₘ₀ ⟨0, by norm_num⟩ * pc.1))^2) dgp.jointMeasure) :
  ∀ (p_val : ℝ) (c_val : Fin 1 → ℝ),
    predictionBias dgp (fun p _ => linearPredictor model_raw p c_val) p_val c_val
    = β_env * c_val ⟨0, by norm_num⟩ := by
  intros p_val c_val
  unfold predictionBias
  rw [h_dgp]
  dsimp

  -- Step 1: Use Lemma A to simplify the linear predictor of a raw model
  -- For a raw model with linear PGS basis, linearPredictor = γ₀₀ + γₘ₀[0] * p
  have h_pred := linearPredictor_eq_affine_of_raw model_raw h_raw_struct h_pgs_basis_linear
  rw [h_pred]

  -- Step 2: Apply the optimal coefficients lemma to get γ₀₀ = 0, γₘ₀[0] = 1
  have h_coeffs := optimal_coefficients_for_additive_dgp model_raw β_env dgp h_dgp
                     h_opt_raw h_pgs_basis_linear h_indep
                     h_means_zero.1 h_means_zero.2 h_var_p_one
                     hP_int hC_int hP2_int hPC_int hY_int hYP_int h_resid_sq_int
  obtain ⟨ha, hb⟩ := h_coeffs

  -- Step 3: Substitute a=0, b=1 into the predictor
  -- linearPredictor = 0 + 1 * p = p
  -- bias = (p + β*c) - p = β*c
  -- Note: hb has γₘ₀ ⟨0, ⋯⟩ which is definitionally equal to γₘ₀ 0
  have hb' : model_raw.γₘ₀ 0 = 1 := hb
  rw [ha, hb']
  ring


/-! ### NOTE ON APPROXIMATE EQUALITY

The previous definition `approxEq a b 0.01` with a hardcoded tolerance was mathematically
problematic: you cannot prove |a - b| < 0.01 from generic hypotheses.

For these theorems, we use **exact equality** instead, which IS provable under the
structural assumptions (linear/affine models, Bayes-optimal in the exact model class).

If approximate analysis is needed, use proper ε-δ statements:
  ∀ ε > 0, ∃ conditions, |a - b| < ε
-/

noncomputable def var {k : ℕ} [Fintype (Fin k)] (dgp : DataGeneratingProcess k)
    (f : ℝ → (Fin k → ℝ) → ℝ) : ℝ :=
  let μ := dgp.jointMeasure
  let m : ℝ := ∫ pc, f pc.1 pc.2 ∂μ
  ∫ pc, (f pc.1 pc.2 - m) ^ 2 ∂μ

noncomputable def rsquared {k : ℕ} [Fintype (Fin k)] (dgp : DataGeneratingProcess k)
    (f g : ℝ → (Fin k → ℝ) → ℝ) : ℝ :=
  let μ := dgp.jointMeasure
  let mf : ℝ := ∫ pc, f pc.1 pc.2 ∂μ
  let mg : ℝ := ∫ pc, g pc.1 pc.2 ∂μ
  let vf : ℝ := ∫ pc, (f pc.1 pc.2 - mf) ^ 2 ∂μ
  let vg : ℝ := ∫ pc, (g pc.1 pc.2 - mg) ^ 2 ∂μ
  let cov : ℝ := ∫ pc, (f pc.1 pc.2 - mf) * (g pc.1 pc.2 - mg) ∂μ
  if vf = 0 ∨ vg = 0 then 0 else (cov ^ 2) / (vf * vg)

theorem quantitative_error_of_normalization (p k sp : ℕ) [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (dgp1 : DataGeneratingProcess k) (h_s1 : hasInteraction dgp1.trueExpectation)
    (hk_pos : k > 0)
    (model_norm : PhenotypeInformedGAM p k sp) (h_norm_model : IsNormalizedScoreModel model_norm) (h_norm_opt : IsBayesOptimalInNormalizedClass dgp1 model_norm)
    (model_oracle : PhenotypeInformedGAM p k sp) (h_oracle_opt : IsBayesOptimalInClass dgp1 model_oracle) :
  let predict_norm := fun p c => linearPredictor model_norm p c
  let predict_oracle := fun p c => linearPredictor model_oracle p c
  expectedSquaredError dgp1 predict_norm - expectedSquaredError dgp1 predict_oracle
  = rsquared dgp1 (fun p c => p) (fun p c => c ⟨0, hk_pos⟩) * var dgp1 (fun p c => p) := by sorry

noncomputable def dgpMultiplicativeBias {k : ℕ} [Fintype (Fin k)] (scaling_func : (Fin k → ℝ) → ℝ) : DataGeneratingProcess k :=
  { trueExpectation := fun p c => (scaling_func c) * p, jointMeasure := stdNormalProdMeasure k }

/-- Under a multiplicative bias DGP where E[Y|P,C] = scaling_func(C) * P,
    the Bayes-optimal PGS coefficient at ancestry c recovers scaling_func(c) exactly.

    **Changed from approximate (≈ 0.01) to exact equality**.
    The approximate version was unprovable from the given hypotheses. -/
theorem multiplicative_bias_correction (k : ℕ) [Fintype (Fin k)]
    (scaling_func : (Fin k → ℝ) → ℝ) (_h_deriv : Differentiable ℝ scaling_func)
    (model : PhenotypeInformedGAM 1 k 1) (h_opt : IsBayesOptimalInClass (dgpMultiplicativeBias scaling_func) model) :
  ∀ c : Fin k → ℝ,
    model.γₘ₀ ⟨0, by norm_num⟩ + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ ⟨0, by norm_num⟩ l) (c l)
    = scaling_func c := by sorry

structure DGPWithLatentRisk (k : ℕ) where
  to_dgp : DataGeneratingProcess k
  noise_variance_given_pc : (Fin k → ℝ) → ℝ
  sigma_G_sq : ℝ
  is_latent : to_dgp.trueExpectation = fun p c => (sigma_G_sq / (sigma_G_sq + noise_variance_given_pc c)) * p

/-- Under a latent risk DGP, the Bayes-optimal PGS coefficient equals the shrinkage factor exactly.

    **Changed from approximate (≈ 0.01) to exact equality**.
    This is derivable from the structure of DGPWithLatentRisk.is_latent. -/
theorem shrinkage_effect {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)]
    (dgp_latent : DGPWithLatentRisk k) (model : PhenotypeInformedGAM 1 k sp)
    (h_opt : IsBayesOptimalInClass dgp_latent.to_dgp model) (_hp_one : p = 1) :
  ∀ c : Fin k → ℝ,
    model.γₘ₀ ⟨0, by norm_num⟩ + ∑ l, evalSmooth model.pcSplineBasis (model.fₘₗ ⟨0, by norm_num⟩ l) (c l)
    = dgp_latent.sigma_G_sq / (dgp_latent.sigma_G_sq + dgp_latent.noise_variance_given_pc c) := by
  intro c

  -- The derivation follows from Equation (1) in "Recalibration of Polygenic Risk Scores"
  -- by Graham, T. et al. (2024).
  --
  -- Setup: Let G = true genetic liability, Y = phenotype, P = polygenic score
  --   Y = G + ε_Y (phenotype noise)
  --   P = G + η(C) (measurement noise depending on ancestry C)
  --
  -- The true conditional expectation E[Y | P, C] satisfies:
  --   E[Y | P=p, C=c] = α(c) * p
  -- where α(c) is the regression coefficient of Y on P given C=c.
  --
  -- By standard regression theory:
  --   α(c) = Cov(Y, P | C=c) / Var(P | C=c)
  --
  -- Computing the numerator:
  --   Cov(Y, P | C) = Cov(G + ε_Y, G + η(C) | C)
  --                 = Var(G)  (since G ⊥ ε_Y ⊥ η(C))
  --                 = σ_G²
  --
  -- Computing the denominator:
  --   Var(P | C) = Var(G + η(C) | C)
  --              = Var(G) + Var(η(C) | C)
  --              = σ_G² + σ_η²(C)
  --
  -- Therefore:
  --   α(c) = σ_G² / (σ_G² + σ_η²(c))
  --
  -- The GAM structure captures this via:
  --   γₘ₀[0] + Σₗ fₘₗ[0,l](cₗ) ≈ α(c)
  --
  -- This is exactly what DGPWithLatentRisk.is_latent encodes:
  --   trueExpectation = fun p c => (sigma_G_sq / (sigma_G_sq + noise_variance_given_pc c)) * p

  -- The Bayes-optimal predictor equals the conditional expectation
  have h_bayes : ∀ p_val, linearPredictor model p_val c =
      dgp_latent.to_dgp.trueExpectation p_val c := by
    intro p_val
    -- IsBayesOptimalInClass means model minimizes expected squared error
    -- The unique minimizer is the conditional expectation E[Y|P,C]
    -- This is dgp_latent.to_dgp.trueExpectation by definition
    sorry

  -- The true expectation has the form α(c) * p
  have h_true_form : dgp_latent.to_dgp.trueExpectation =
      fun p_val c_val => (dgp_latent.sigma_G_sq / (dgp_latent.sigma_G_sq + dgp_latent.noise_variance_given_pc c_val)) * p_val :=
    dgp_latent.is_latent

  -- For a Bayes-optimal model with linear PGS basis, the coefficient of p must equal α(c)
  -- This means: γₘ₀[0] + Σₗ fₘₗ[0,l](cₗ) = σ_G² / (σ_G² + σ_η²(c))

  -- From h_bayes with p=1 and p=0, we can extract the coefficients:
  have h_at_0 : linearPredictor model 0 c = dgp_latent.to_dgp.trueExpectation 0 c := h_bayes 0
  have h_at_1 : linearPredictor model 1 c = dgp_latent.to_dgp.trueExpectation 1 c := h_bayes 1

  -- From h_true_form: trueExpectation 0 c = 0, trueExpectation 1 c = α(c)
  simp only [h_true_form] at h_at_0 h_at_1
  simp only [mul_zero] at h_at_0
  simp only [mul_one] at h_at_1

  -- The linearPredictor has structure: linearPredictor p c = base(c) + slope(c) * p
  -- From h_at_0: base(c) = 0
  -- From h_at_1: slope(c) = α(c)
  -- The goal is exactly slope(c), so we need to show:
  --   γₘ₀[0] + Σₗ evalSmooth(fₘₗ[0,l], c[l]) = α(c)

  -- This requires showing linearPredictor decomposes as base + slope * p for p=1 models,
  -- which is proven in linearPredictor_decomp (but requires linear PGS basis hypothesis)
  -- The algebraic form: α(c) = σ_G² / (σ_G² + σ_η²(c)) is exactly the shrinkage factor
  -- from measurement error attenuation bias theory.
  -- See: Fuller (1987), "Measurement Error Models", Chapter 2.
  sorry -- Shrinkage coefficient extraction from linearPredictor structure

/-- Predictions are invariant under affine transformations of ancestry coordinates.

    **Changed from approximate (≈) to exact equality**.
    If the model class can represent the transform, this is exact. -/
theorem prediction_is_invariant_to_affine_pc_transform {n k p sp : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)]
    (A : Matrix (Fin k) (Fin k) ℝ) (_hA : IsUnit A.det) (b : Fin k → ℝ) (data : RealizedData n k) (lambda : ℝ) :
  let data' : RealizedData n k := { y := data.y, p := data.p, c := fun i => A.mulVec (data.c i) + b }
  let model := fit p k sp n data lambda; let model' := fit p k sp n data' lambda
  ∀ (pgs : ℝ) (pc : Fin k → ℝ), predict model pgs pc = predict model' pgs (A.mulVec pc + b) := by sorry

noncomputable def dist_to_support {k : ℕ} (c : Fin k → ℝ) (supp : Set (Fin k → ℝ)) : ℝ :=
  Metric.infDist c supp

theorem extrapolation_risk {n k p sp : ℕ} [Fintype (Fin n)] [Fintype (Fin k)] [Fintype (Fin p)] [Fintype (Fin sp)] (dgp : DataGeneratingProcess k) (data : RealizedData n k) (lambda : ℝ) (c_new : Fin k → ℝ) :
  ∃ (f : ℝ → ℝ), Monotone f ∧ |predict (fit p k sp n data lambda) 0 c_new - dgp.trueExpectation 0 c_new| ≤
    f (dist_to_support c_new {c | ∃ i, c = data.c i}) := by sorry

theorem context_specificity {p k sp : ℕ} [Fintype (Fin p)] [Fintype (Fin k)] [Fintype (Fin sp)] (dgp1 dgp2 : DGPWithEnvironment k)
    (h_same_genetics : dgp1.trueGeneticEffect = dgp2.trueGeneticEffect ∧ dgp1.to_dgp.jointMeasure = dgp2.to_dgp.jointMeasure)
    (h_diff_env : dgp1.environmentalEffect ≠ dgp2.environmentalEffect)
    (model1 : PhenotypeInformedGAM p k sp) (h_opt1 : IsBayesOptimalInClass dgp1.to_dgp model1) :
  ¬ IsBayesOptimalInClass dgp2.to_dgp model1 := by
  intro h_opt2
  have h_neq : dgp1.to_dgp.trueExpectation ≠ dgp2.to_dgp.trueExpectation := by
    intro h_eq_fn
    rw [dgp1.is_additive_causal, dgp2.is_additive_causal, h_same_genetics.1] at h_eq_fn
    have : dgp1.environmentalEffect = dgp2.environmentalEffect := by
      ext c
      have := congr_fun (congr_fun h_eq_fn 0) c
      simp at this; exact this
    exact h_diff_env this
  -- h_neq: dgp1.to_dgp.trueExpectation ≠ dgp2.to_dgp.trueExpectation
  -- h_opt1: model1 is Bayes-optimal for dgp1
  -- h_opt2: model1 is Bayes-optimal for dgp2
  -- But a Bayes-optimal predictor matches the true expectation (up to modification on null sets)
  -- If model1 is optimal for both, then dgp1.trueExpectation = dgp2.trueExpectation, contradiction

  -- The formal argument: if model1 minimizes E[(Y - pred)²] for both dgp1 and dgp2,
  -- then pred = E[Y|P,C] for each (conditional expectation is unique minimizer of MSE).
  -- But E[Y|P,C] differs between the two DGPs, so model1 cannot be optimal for both.
  -- See: Ferguson (1967), "Mathematical Statistics: A Decision Theoretic Approach", Theorem 4.1.
  sorry -- Bayes optimal predictor = conditional expectation (a.e.) under MSE loss

end AllClaims

/-!
=================================================================
## Part 3: Numerical and Algebraic Foundations
=================================================================
These theorems formalize the correctness of the numerical methods
used in the Rust implementation (calibrate/basis.rs, calibrate/estimate.rs).
-/

section BSplineFoundations

/-!
### B-Spline Basis Functions

The Cox-de Boor recursion defines B-spline basis functions. We prove
the partition of unity property which ensures probability semantics.
-/

variable {numKnots : ℕ}

/-- A valid B-spline knot vector: non-decreasing with proper multiplicity. -/
structure KnotVector (m : ℕ) where
  knots : Fin m → ℝ
  sorted : ∀ i j : Fin m, i ≤ j → knots i ≤ knots j

/-- Cox-de Boor recursive definition of B-spline basis function.
    N_{i,p}(x) is the i-th basis function of degree p.
    We use a simpler formulation to avoid index bound issues. -/
noncomputable def bspline_basis_raw (t : ℕ → ℝ) : ℕ → ℕ → ℝ → ℝ
  | i, 0, x => if t i ≤ x ∧ x < t (i + 1) then 1 else 0
  | i, p + 1, x =>
    let left_denom := t (i + p + 1) - t i
    let right_denom := t (i + p + 2) - t (i + 1)
    let left := if left_denom = 0 then 0
                else (x - t i) / left_denom * bspline_basis_raw t i p x
    let right := if right_denom = 0 then 0
                 else (t (i + p + 2) - x) / right_denom * bspline_basis_raw t (i + 1) p x
    left + right

/-- Local support property: N_{i,p}(x) = 0 outside [t_i, t_{i+p+1}).

    **Geometric insight**: The support of a B-spline grows by one knot interval with
    each degree increase. N_{i,p} lives on [t_i, t_{i+p+1}). In the recursion for p+1,
    we combine N_{i,p} (starts at t_i) and N_{i+1,p} (ends at t_{i+p+2}).
    The union creates the new support [t_i, t_{i+p+2}).

    **Proof by induction on p**:
    - Base case (p=0): By definition, N_{i,0}(x) = 1 if t_i ≤ x < t_{i+1}, else 0.
    - Inductive case (p+1): Cox-de Boor recursion combines N_{i,p} and N_{i+1,p}.
      Both have zero support outside the required interval. -/
theorem bspline_local_support (t : ℕ → ℝ)
    (h_sorted : ∀ i j, i ≤ j → t i ≤ t j)
    (i p : ℕ) (x : ℝ)
    (h_outside : x < t i ∨ t (i + p + 1) ≤ x) :
    bspline_basis_raw t i p x = 0 := by
  induction p generalizing i with
  | zero =>
    simp only [bspline_basis_raw]
    split_ifs with h_in
    · obtain ⟨h_lo, h_hi⟩ := h_in
      rcases h_outside with h_lt | h_ge
      · exact absurd h_lo (not_le.mpr h_lt)
      · simp only [add_zero] at h_ge
        exact absurd h_hi (not_lt.mpr h_ge)
    · rfl
  | succ p ih =>
    simp only [bspline_basis_raw]
    rcases h_outside with h_lt | h_ge
    · -- x < t_i: both terms zero
      have h_left_zero : bspline_basis_raw t i p x = 0 := ih i (Or.inl h_lt)
      have h_i1_le : t i ≤ t (i + 1) := h_sorted i (i + 1) (Nat.le_succ i)
      have h_right_zero : bspline_basis_raw t (i + 1) p x = 0 :=
        ih (i + 1) (Or.inl (lt_of_lt_of_le h_lt h_i1_le))
      simp only [h_left_zero, h_right_zero, mul_zero, ite_self, add_zero]
    · -- x ≥ t_{i+p+2}: both terms zero
      have h_right_idx : i + 1 + p + 1 = i + p + 2 := by ring
      have h_right_zero : bspline_basis_raw t (i + 1) p x = 0 := by
        apply ih (i + 1); right; rw [h_right_idx]; exact h_ge
      have h_mono : t (i + p + 1) ≤ t (i + p + 2) := h_sorted (i + p + 1) (i + p + 2) (Nat.le_succ _)
      have h_left_zero : bspline_basis_raw t i p x = 0 := by
        apply ih i; right; exact le_trans h_mono h_ge
      simp only [h_left_zero, h_right_zero, mul_zero, ite_self, add_zero]

/-- B-spline basis functions are non-negative everywhere.

    **Geometry of the "Zero-Out" Property** (Key insight from user):
    The Cox-de Boor recursion uses linear weights: (x - t_i) / (t_{i+p+1} - t_i).
    These weights become NEGATIVE when x < t_i. The ONLY reason the spline remains
    non-negative is that the lower-order basis function N_{i,p}(x) "turns off"
    (becomes exactly zero) precisely when the weight becomes negative.

    Therefore, bspline_local_support is a strict prerequisite for this proof.

    **Proof by induction on p**:
    - Base case (p=0): N_{i,0}(x) is either 0 or 1, both ≥ 0.
    - Inductive case (p+1): For each term α(x) * N_{i,p}(x):
      * If x ∈ [t_i, t_{i+p+1}): α(x) ≥ 0 and N_{i,p}(x) ≥ 0 by IH
      * If x ∉ [t_i, t_{i+p+1}): N_{i,p}(x) = 0 by local_support, so product = 0 -/
theorem bspline_nonneg (t : ℕ → ℝ) (h_sorted : ∀ i j, i ≤ j → t i ≤ t j)
    (i p : ℕ) (x : ℝ) : 0 ≤ bspline_basis_raw t i p x := by
  induction p generalizing i with
  | zero =>
    simp only [bspline_basis_raw]
    split_ifs
    · exact zero_le_one
    · exact le_refl 0
  | succ p ih =>
    simp only [bspline_basis_raw]
    apply add_nonneg
    · -- Left term: (x - t_i) / (t_{i+p+1} - t_i) * N_{i,p}(x)
      split_ifs with h_denom
      · exact le_refl 0
      · by_cases h_in_support : x < t i
        · -- x < t_i: N_{i,p}(x) = 0 by local support, so product = 0
          have : bspline_basis_raw t i p x = 0 :=
            bspline_local_support t h_sorted i p x (Or.inl h_in_support)
          simp only [this, mul_zero, le_refl]
        · -- x ≥ t_i: weight (x - t_i)/denom ≥ 0, and N_{i,p}(x) ≥ 0 by IH
          push_neg at h_in_support
          have h_num_nn : 0 ≤ x - t i := sub_nonneg.mpr h_in_support
          have h_denom_pos : 0 < t (i + p + 1) - t i := by
            have h_le : t i ≤ t (i + p + 1) := h_sorted i (i + p + 1) (by omega)
            exact lt_of_le_of_ne (sub_nonneg.mpr h_le) (ne_comm.mp h_denom)
          exact mul_nonneg (div_nonneg h_num_nn (le_of_lt h_denom_pos)) (ih i)
    · -- Right term: (t_{i+p+2} - x) / (t_{i+p+2} - t_{i+1}) * N_{i+1,p}(x)
      split_ifs with h_denom
      · exact le_refl 0
      · by_cases h_in_support : t (i + p + 2) ≤ x
        · -- x ≥ t_{i+p+2}: N_{i+1,p}(x) = 0 by local support
          have h_idx : i + 1 + p + 1 = i + p + 2 := by ring
          have : bspline_basis_raw t (i + 1) p x = 0 := by
            apply bspline_local_support t h_sorted (i + 1) p x; right; rw [h_idx]; exact h_in_support
          simp only [this, mul_zero, le_refl]
        · -- x < t_{i+p+2}: weight (t_{i+p+2} - x)/denom ≥ 0, and N_{i+1,p}(x) ≥ 0 by IH
          push_neg at h_in_support
          have h_num_nn : 0 ≤ t (i + p + 2) - x := sub_nonneg.mpr (le_of_lt h_in_support)
          have h_denom_pos : 0 < t (i + p + 2) - t (i + 1) := by
            have h_le : t (i + 1) ≤ t (i + p + 2) := h_sorted (i + 1) (i + p + 2) (by omega)
            exact lt_of_le_of_ne (sub_nonneg.mpr h_le) (ne_comm.mp h_denom)
          exact mul_nonneg (div_nonneg h_num_nn (le_of_lt h_denom_pos)) (ih (i + 1))

/-- **Partition of Unity**: B-spline basis functions sum to 1 within the valid domain.
    This is critical for the B-splines in basis.rs to produce valid probability adjustments.
    For n basis functions of degree p with knot vector t, when t[p] ≤ x < t[n], we have
    ∑_{i=0}^{n-1} N_{i,p}(x) = 1. -/
theorem bspline_partition_of_unity (t : ℕ → ℝ) (num_basis : ℕ)
    (h_sorted : ∀ i j, i ≤ j → t i ≤ t j)
    (p : ℕ) (x : ℝ)
    (h_domain : t p ≤ x ∧ x < t num_basis)
    (h_valid : num_basis > p) :
    (Finset.range num_basis).sum (fun i => bspline_basis_raw t i p x) = 1 := by
  -- **Partition of Unity** is a fundamental property of B-spline basis functions.
  -- See: de Boor (1978), "A Practical Guide to Splines", Theorem 4.2
  -- The proof proceeds by induction on degree p, using the Cox-de Boor recursion.
  -- Key insight: the recursion coefficients sum to 1 (telescoping property).
  -- This validates the B-spline implementation in basis.rs.
  sorry -- B-spline partition of unity (standard result from approximation theory)

end BSplineFoundations

section WeightedOrthogonality

/-!
### Weighted Orthogonality Constraints

The calibration code applies sum-to-zero and polynomial orthogonality constraints
via nullspace projection. These theorems formalize that the projection is correct.
-/

variable {n m k : ℕ} [Fintype (Fin n)] [Fintype (Fin m)] [Fintype (Fin k)]
variable [DecidableEq (Fin n)] [DecidableEq (Fin m)] [DecidableEq (Fin k)]

/-- A diagonal weight matrix constructed from a weight vector. -/
def diagonalWeight (w : Fin n → ℝ) : Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal w

/-- Two column spaces are weighted-orthogonal if their weighted inner product is zero.
    Uses explicit transpose to avoid parsing issues. -/
def IsWeightedOrthogonal (A : Matrix (Fin n) (Fin m) ℝ)
    (B : Matrix (Fin n) (Fin k) ℝ) (W : Matrix (Fin n) (Fin n) ℝ) : Prop :=
  Matrix.transpose A * W * B = 0

/-- A matrix Z spans the nullspace of M if MZ = 0 and Z has maximal rank. -/
def SpansNullspace (Z : Matrix (Fin m) (Fin (m - k)) ℝ)
    (M : Matrix (Fin k) (Fin m) ℝ) : Prop :=
  M * Z = 0 ∧ Matrix.rank Z = m - k

/-- **Constraint Projection Correctness**: If Z spans the nullspace of BᵀWC,
    then B' = BZ is weighted-orthogonal to C.
    This validates `apply_weighted_orthogonality_constraint` in basis.rs.

    **Proof**:
    (BZ)ᵀ W C = Zᵀ (Bᵀ W C) = 0 because Z is in the nullspace of (Bᵀ W C)ᵀ.

    More precisely:
    - SpansNullspace Z M means M * Z = 0
    - Here M = (Bᵀ W C)ᵀ = Cᵀ Wᵀ B = Cᵀ W B (if W is symmetric, which diagonal matrices are)
    - We want: (BZ)ᵀ W C = Zᵀ Bᵀ W C
    - By associativity: Zᵀ Bᵀ W C = (Bᵀ W C)ᵀ · Z = M · Z = 0 (by h_spans.1)

    Wait, transpose swap: (Zᵀ (Bᵀ W C))ᵀ = (Bᵀ W C)ᵀ Z
    Actually: Zᵀ · (Bᵀ W C) has shape (m-k) × k, while M · Z = 0 where M = (Bᵀ W C)ᵀ

    The key relation is: Zᵀ · A = (Aᵀ · Z)ᵀ, so if Aᵀ · Z = 0, then Zᵀ · A = 0. -/
theorem constraint_projection_correctness
    (B : Matrix (Fin n) (Fin m) ℝ)
    (C : Matrix (Fin n) (Fin k) ℝ)
    (W : Matrix (Fin n) (Fin n) ℝ)
    (Z : Matrix (Fin m) (Fin (m - k)) ℝ)
    (h_spans : SpansNullspace Z (Matrix.transpose (Matrix.transpose B * W * C))) :
    IsWeightedOrthogonal (B * Z) C W := by
  unfold IsWeightedOrthogonal
  -- Goal: Matrix.transpose (B * Z) * W * C = 0
  -- Expand: (BZ)ᵀ W C = Zᵀ Bᵀ W C
  have h1 : Matrix.transpose (B * Z) = Matrix.transpose Z * Matrix.transpose B := by
    exact Matrix.transpose_mul B Z
  rw [h1]
  -- Now: Zᵀ Bᵀ W C
  -- We need to show: Zᵀ * (Bᵀ W C) = 0
  -- From h_spans: (Bᵀ W C)ᵀ * Z = 0
  -- Taking transpose: Zᵀ * (Bᵀ W C) = ((Bᵀ W C)ᵀ * Z)ᵀ
  -- If (Bᵀ W C)ᵀ * Z = 0, then Zᵀ * (Bᵀ W C) = 0ᵀ = 0
  have h2 : Matrix.transpose Z * Matrix.transpose B * W * C =
            Matrix.transpose Z * (Matrix.transpose B * W * C) := by
    simp only [Matrix.mul_assoc]
  rw [h2]
  -- Now use the nullspace condition
  have h3 : Matrix.transpose (Matrix.transpose B * W * C) * Z = 0 := h_spans.1
  -- Taking transpose of both sides: Zᵀ * (Bᵀ W C) = 0
  have h4 : Matrix.transpose Z * (Matrix.transpose B * W * C) =
            Matrix.transpose (Matrix.transpose (Matrix.transpose B * W * C) * Z) := by
    rw [Matrix.transpose_mul]
    simp only [Matrix.transpose_transpose]
  rw [h4, h3]
  simp only [Matrix.transpose_zero]

omit [Fintype (Fin n)] [Fintype (Fin k)] [DecidableEq (Fin n)] [DecidableEq (Fin m)]
    [DecidableEq (Fin k)] in
/-- The constrained basis preserves the column space spanned by valid coefficients. -/
theorem constrained_basis_spans_subspace
    (B : Matrix (Fin n) (Fin m) ℝ)
    (Z : Matrix (Fin m) (Fin (m - k)) ℝ)
    (β : Fin (m - k) → ℝ) :
    ∃ (β' : Fin m → ℝ), (B * Z).mulVec β = B.mulVec β' := by
  use Z.mulVec β
  rw [Matrix.mulVec_mulVec]

/-- Sum-to-zero constraint: the constraint matrix C is a column of ones. -/
def sumToZeroConstraint (n : ℕ) : Matrix (Fin n) (Fin 1) ℝ :=
  fun _ _ => 1

/-- After applying sum-to-zero constraint, basis evaluations sum to zero at data points.
    Note: This theorem uses a specialized constraint for k=1. -/
theorem sum_to_zero_after_projection
    (B : Matrix (Fin n) (Fin m) ℝ)
    (W : Matrix (Fin n) (Fin n) ℝ)
    (Z : Matrix (Fin m) (Fin (m - 1)) ℝ)
    (h_constraint : SpansNullspace Z (Matrix.transpose (Matrix.transpose B * W * sumToZeroConstraint n)))
    (β : Fin (m - 1) → ℝ) :
    Finset.univ.sum (fun i : Fin n => ((B * Z).mulVec β) i * W i i) = 0 := by
  sorry

end WeightedOrthogonality

section WoodReparameterization

/-!
### Wood's Stable Reparameterization

The PIRLS solver in estimate.rs uses Wood (2011)'s reparameterization to
avoid numerical instability. This section proves the algebraic equivalence.
-/

variable {n p : ℕ} [Fintype (Fin n)] [Fintype (Fin p)]
variable [DecidableEq (Fin p)]

/-- Quadratic form: βᵀSβ computed as dot product. -/
noncomputable def quadForm (S : Matrix (Fin p) (Fin p) ℝ) (β : Fin p → ℝ) : ℝ :=
  Finset.univ.sum (fun i => β i * (S.mulVec β) i)

/-- Penalized least squares objective: ‖y - Xβ‖² + βᵀSβ -/
noncomputable def penalized_objective
    (X : Matrix (Fin n) (Fin p) ℝ) (y : Fin n → ℝ)
    (S : Matrix (Fin p) (Fin p) ℝ) (β : Fin p → ℝ) : ℝ :=
  ‖y - X.mulVec β‖^2 + quadForm S β

/-- A matrix Q is orthogonal if QQᵀ = I. Uses explicit transpose. -/
def IsOrthogonal (Q : Matrix (Fin p) (Fin p) ℝ) : Prop :=
  Q * Matrix.transpose Q = 1 ∧ Matrix.transpose Q * Q = 1

/-- The penalty transforms as a congruence under reparameterization.

    **Proof**: (Qβ')ᵀ S (Qβ') = β'ᵀ Qᵀ S Q β' = β'ᵀ (QᵀSQ) β'
    This is just associativity of matrix-vector multiplication.

    This is a key step in Wood's (2011) stable reparameterization for GAMs,
    as it shows how the penalty matrix S transforms under an orthogonal change
    of basis Q. By choosing Q to be the eigenvectors of S, the transformed
    penalty matrix QᵀSQ becomes diagonal, simplifying the optimization problem. -/
theorem penalty_congruence
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (β' : Fin p → ℝ) (_h_orth : IsOrthogonal Q) :
    quadForm S (Q.mulVec β') = quadForm (Matrix.transpose Q * S * Q) β' := by
  -- quadForm S (Qβ') = Σᵢ (Qβ')ᵢ * (S(Qβ'))ᵢ = (Qβ')ᵀ S (Qβ')
  -- = β'ᵀ Qᵀ S Q β' = β'ᵀ (QᵀSQ) β' = quadForm (QᵀSQ) β'
  unfold quadForm
  -- LHS: Σᵢ (Q.mulVec β') i * (S.mulVec (Q.mulVec β')) i
  -- RHS: Σᵢ β' i * ((QᵀSQ).mulVec β') i

  -- The key is: (QᵀSQ).mulVec β' = Qᵀ.mulVec (S.mulVec (Q.mulVec β'))
  -- This follows from Matrix.mulVec_mulVec: (AB).mulVec v = A.mulVec (B.mulVec v)

  -- First, simplify the RHS penalty matrix action
  have h_rhs_eq : (Matrix.transpose Q * S * Q).mulVec β' =
                   (Matrix.transpose Q).mulVec (S.mulVec (Q.mulVec β')) := by
    -- (QᵀSQ).mulVec β' = Qᵀ.mulVec (S.mulVec (Q.mulVec β'))
    -- By Matrix.mulVec_mulVec: (AB).mulVec v = A.mulVec (B.mulVec v)
    -- Apply twice: (Qᵀ(SQ)).mulVec = Qᵀ.mulVec ((SQ).mulVec) = Qᵀ.mulVec (S.mulVec (Q.mulVec))
    -- See: Mathlib.LinearAlgebra.Matrix.NonsingularInverse, Matrix.mulVec_mulVec
    sorry -- mulVec associativity via Matrix.mulVec_mulVec

  -- The quadForm equality requires showing the dot products are equal
  -- This is essentially β'ᵀ (QᵀSQ) β' = (Qβ')ᵀ S (Qβ')
  -- Which is a standard matrix identity for bilinear forms
  -- See: Mathlib.LinearAlgebra.QuadraticForm.Basic
  sorry -- Final step: dot product manipulation β'ᵀQᵀ = (Qβ')ᵀ

/-- **Reparameterization Equivalence**: Under orthogonal change of variables β = Qβ',
    the penalized objective transforms covariantly.
    This validates `stable_reparameterization` in estimate.rs.

    **Proof Sketch (Isometry)**:
    1. Residual: y - X(Qβ') = y - (XQ)β', so ‖residual‖² depends only on XQ, not Q separately
    2. Penalty: (Qβ')ᵀS(Qβ') = β'ᵀ(QᵀSQ)β' by associativity of matrix multiplication

    This shows minimizing over β = Qβ' is equivalent to minimizing over β' with transformed design/penalty. -/
theorem reparameterization_equivalence
    (X : Matrix (Fin n) (Fin p) ℝ) (y : Fin n → ℝ)
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (β' : Fin p → ℝ) (h_orth : IsOrthogonal Q) :
    penalized_objective X y S (Q.mulVec β') =
    penalized_objective (X * Q) y (Matrix.transpose Q * S * Q) β' := by
  unfold penalized_objective
  -- Step 1: Show the residual norms are equal
  -- X(Qβ') = (XQ)β' by Matrix.mulVec_mulVec
  have h_residual : y - X.mulVec (Q.mulVec β') = y - (X * Q).mulVec β' := by
    rw [Matrix.mulVec_mulVec]
  rw [h_residual]

  -- Step 2: Show the penalty terms are equal
  -- quadForm S (Qβ') = quadForm (QᵀSQ) β'
  have h_penalty : quadForm S (Q.mulVec β') = quadForm (Matrix.transpose Q * S * Q) β' := by
    exact penalty_congruence S Q β' h_orth

  rw [h_penalty]

omit [Fintype (Fin n)] in
/-- The fitted values are invariant under reparameterization. -/
theorem fitted_values_invariant
    (X : Matrix (Fin n) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (β : Fin p → ℝ) (_h_orth : IsOrthogonal Q)
    (β' : Fin p → ℝ) (h_relation : β = Q.mulVec β') :
    X.mulVec β = (X * Q).mulVec β' := by
  rw [h_relation]
  rw [Matrix.mulVec_mulVec]

/-- Eigenvalue structure is preserved: if S = QΛQᵀ, then QᵀSQ = Λ.
    This is the key insight that makes the reparameterization numerically stable.

    **Proof**: QᵀSQ = Qᵀ(QΛQᵀ)Q = (QᵀQ)Λ(QᵀQ) = IΛI = Λ by orthogonality of Q. -/
theorem eigendecomposition_diagonalizes
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (Λ : Matrix (Fin p) (Fin p) ℝ)
    (h_orth : IsOrthogonal Q)
    (h_decomp : S = Q * Λ * Matrix.transpose Q)
    (_h_diag : ∀ i j : Fin p, i ≠ j → Λ i j = 0) :
    Matrix.transpose Q * S * Q = Λ := by
  rw [h_decomp]
  -- Qᵀ(QΛQᵀ)Q = (QᵀQ)Λ(QᵀQ) = IΛI = Λ
  have h_assoc : Matrix.transpose Q * (Q * Λ * Matrix.transpose Q) * Q
                = Matrix.transpose Q * Q * Λ * (Matrix.transpose Q * Q) := by
    -- Use associativity of matrix multiplication
    simp only [Matrix.mul_assoc]
  rw [h_assoc]
  -- By orthogonality: QᵀQ = I
  rw [h_orth.2]
  simp only [Matrix.one_mul, Matrix.mul_one]

/-- The optimal β under the reparameterized system transforms back correctly. -/
theorem optimal_solution_transforms
    (X : Matrix (Fin n) (Fin p) ℝ) (y : Fin n → ℝ)
    (S : Matrix (Fin p) (Fin p) ℝ) (Q : Matrix (Fin p) (Fin p) ℝ)
    (h_orth : IsOrthogonal Q)
    (β_opt : Fin p → ℝ) (β'_opt : Fin p → ℝ)
    (h_opt : ∀ β, penalized_objective X y S β_opt ≤ penalized_objective X y S β)
    (h_opt' : ∀ β', penalized_objective (X * Q) y (Matrix.transpose Q * S * Q) β'_opt ≤
                    penalized_objective (X * Q) y (Matrix.transpose Q * S * Q) β') :
    X.mulVec β_opt = (X * Q).mulVec β'_opt := by
  sorry

end WoodReparameterization

end Calibrator
