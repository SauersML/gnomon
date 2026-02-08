import Mathlib.Analysis.InnerProductSpace.PiL2

open Real

example : dist (fun (_ : Fin 2) => (1 : ℝ)) (fun _ => 0) = 1 := by
  -- L-infinity norm of (1, 1) is 1.
  -- L2 norm of (1, 1) is sqrt(2).
  rw [dist_eq_norm]
  simp [norm]
  -- If this works, it's likely sup norm.
  exact rfl
