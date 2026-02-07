import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Notation
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Inv

open Matrix

variable {m : Type*} [Fintype m] [DecidableEq m]

lemma deriv_finset_prod {s : Finset m} {f : m → ℝ → ℝ} {x : ℝ}
    (h : ∀ i ∈ s, DifferentiableAt ℝ (f i) x) :
    deriv (fun x => ∏ i in s, f i x) x = ∑ i in s, (∏ j in s.erase i, f j x) * deriv (f i) x := by
  induction s using Finset.induction_on with
  | empty =>
    simp
  | insert i s hi ih =>
    rw [Finset.prod_insert hi, deriv_mul]
    · rw [ih (fun j hj => h j (Finset.mem_insert_of_mem hj))]
      rw [Finset.sum_insert hi]
      simp only [Finset.prod_insert hi]
      -- Goal: (∏ j ∈ s, f j x) * deriv (f i) x + f i x * ∑ j ∈ s, (∏ k ∈ s.erase j, f k x) * deriv (f j) x
      --     = (∏ j ∈ s, f j x) * deriv (f i) x + ∑ j ∈ s, (∏ k ∈ insert i s \ {j}, f k x) * deriv (f j) x

      -- Need to show: f i x * ∑ j ∈ s, (∏ k ∈ s.erase j, f k x) * deriv (f j) x
      --             = ∑ j ∈ s, (∏ k ∈ (insert i s).erase j, f k x) * deriv (f j) x

      congr 1
      apply Finset.sum_congr rfl
      intro j hj
      rw [Finset.mul_sum] -- wait, no
      -- term j: f i x * ((∏ k ∈ s.erase j, f k x) * deriv (f j) x)
      --       = (f i x * ∏ k ∈ s.erase j, f k x) * deriv (f j) x
      -- We want: (∏ k ∈ (insert i s).erase j, f k x) * deriv (f j) x
      -- Since j ∈ s and i ∉ s, j ≠ i.
      -- (insert i s).erase j = insert i (s.erase j)
      rw [Finset.erase_insert]
      · rw [Finset.prod_insert]
        · ring
        · simp [Finset.mem_erase, hi]
          intro h
          exact (ne_of_mem_of_not_mem hj hi).symm h -- j ∈ s, i ∉ s => i ≠ j
      · intro h_eq
        rw [h_eq] at hj
        contradiction
    · apply h i (Finset.mem_insert_self i s)
    · apply DifferentiableAt.finset_prod
      intro j hj
      apply h j (Finset.mem_insert_of_mem hj)

variable {n : Type*} [Fintype n] [DecidableEq n]

theorem deriv_prod_univ {f : n → ℝ → ℝ} {x : ℝ}
    (h : ∀ i, DifferentiableAt ℝ (f i) x) :
    deriv (fun x => ∏ i, f i x) x = ∑ i, (∏ j in Finset.univ.erase i, f j x) * deriv (f i) x := by
  apply deriv_finset_prod
  intro i _
  exact h i
