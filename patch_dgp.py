with open('proofs/Calibrator/DGP.lean', 'r') as f:
    content = f.read()

new_thm = """theorem ld_decay_implies_nonlinear_calibration_proved {k : ℕ} [Fintype (Fin k)]
    (mech : LDDecayMechanism k)
    (h_nonlin : ¬ ∃ a b : ℝ, ∀ c : Fin k → ℝ, mech.tagging_efficiency (mech.distance c) = a + b * mech.distance c) :
    ∀ (beta0 beta1 : ℝ),
      (fun c => beta0 + beta1 * mech.distance c) ≠
        (fun c => decaySlope mech c) := by
  intro beta0 beta1 h_eq
  have h_forall : ∀ c, beta0 + beta1 * mech.distance c = mech.tagging_efficiency (mech.distance c) :=
    fun c => congr_fun h_eq c
  apply h_nonlin
  use beta0, beta1
  intro c
  rw [← h_forall c]"""

old_thm = """theorem ld_decay_implies_nonlinear_calibration_sketch {k : ℕ} [Fintype (Fin k)]
    (mech : LDDecayMechanism k)
    (h_nonlin : ¬ ∃ a b, ∀ d ∈ Set.range mech.distance, mech.tagging_efficiency d = a + b * d) :
    ∀ (beta0 beta1 : ℝ),
      (fun c => beta0 + beta1 * mech.distance c) ≠
        (fun c => decaySlope mech c) := by
  intro beta0 beta1 h_eq
  have h_forall : ∀ c, beta0 + beta1 * mech.distance c = mech.tagging_efficiency (mech.distance c) :=
    fun c => congr_fun h_eq c

  -- This contradicts h_nonlin
  apply h_nonlin
  use beta0, beta1
  intro d hd
  obtain ⟨c, hc⟩ := hd
  rw [← hc, h_forall c]"""

content = content.replace(old_thm, new_thm)
with open('proofs/Calibrator/DGP.lean', 'w') as f:
    f.write(content)
