import Mathlib

structure Params where
  Ne : ℝ
  mu : ℝ
  Ne_pos : 0 < Ne
  mu_nonneg : 0 ≤ mu

noncomputable def Params.theta (p : Params) : ℝ :=
  4 * p.Ne * p.mu

theorem Params.theta_nonneg (p : Params) : 0 ≤ p.theta := by
  unfold theta
  positivity
