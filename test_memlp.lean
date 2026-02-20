import Mathlib.MeasureTheory.Function.L2Space

open MeasureTheory

variable {α E : Type*} [MeasurableSpace α] {μ : Measure α}
variable [NormedAddCommGroup E]
variable {p : ℝ≥0∞}

lemma memLp_finset_sum {ι : Type*} (s : Finset ι) {f : ι → α → E} (hf : ∀ i ∈ s, MemLp (f i) p μ) :
  MemLp (fun x => ∑ i in s, f i x) p μ := by
  induction s using Finset.induction_on with
  | empty =>
    simp
    exact MemLp.zero
  | insert _ _ ih =>
    simp
    apply MemLp.add
    · apply hf
      simp
    · apply ih
      intro i hi
      apply hf
      simp [hi]

#check memLp_finset_sum
