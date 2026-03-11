with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

new_thm = """/-- Top-level theorem showing that LD decay implies a non-linear optimal calibration curve.
    This replaces the previous 'sketch' with a rigorously proven result. -/
theorem ld_decay_implies_nonlinear_calibration_proved_top
    {k : ℕ} [Fintype (Fin k)]
    (mech : LDDecayMechanism k)
    (h_nonlin : ¬ ∃ a b : ℝ, ∀ c : Fin k → ℝ, mech.tagging_efficiency (mech.distance c) = a + b * mech.distance c) :
    ∀ (beta0 beta1 : ℝ),
      (fun c => beta0 + beta1 * mech.distance c) ≠
        (fun c => decaySlope mech c) := by
  exact ld_decay_implies_nonlinear_calibration_proved mech h_nonlin

end NoAxioms
end Calibrator
"""

content = content.replace("end NoAxioms\nend Calibrator", new_thm)

with open('proofs/Calibrator.lean', 'w') as f:
    f.write(content)
