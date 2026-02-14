import Mathlib.Analysis.Calculus.Deriv.Prod

open MeasureTheory -- just in case

variable {ι : Type} [Fintype ι] [DecidableEq ι]
variable (f : ι → ℝ → ℝ) (x : ℝ)
variable (hf : ∀ i, DifferentiableAt ℝ (f i) x)

example : DifferentiableAt ℝ (fun y => ∏ i, f i y) x := by
  apply DifferentiableAt.finset_prod
  intro i _
  exact hf i

example : deriv (fun y => ∏ i, f i y) x = ∑ i, (∏ j ∈ Finset.univ.erase i, f j x) * deriv (f i) x := by
  rw [deriv_finset_prod (fun i _ => hf i)]
  simp only [smul_eq_mul]
