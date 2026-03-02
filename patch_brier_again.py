import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

search = """theorem expectedBrierScore_deriv (p π : ℝ) :
    2 * (p - π) = -2 * π + 2 * p := by ring"""

replace = """theorem expectedBrierScore_deriv (p π : ℝ) :
    deriv (fun x => expectedBrierScore x π) p = 2 * (p - π) := by
  have h_eq : (fun x => expectedBrierScore x π) = (fun x => (π - 2 * π * x) + x ^ 2) := by
    ext x
    have h1 := expectedBrierScore_quadratic x π
    have h2 : π - 2 * π * x + x ^ 2 = (π - 2 * π * x) + x ^ 2 := rfl
    rw [h1, h2]
  rw [h_eq]

  have hd_x2 : DifferentiableAt ℝ (fun x : ℝ => x ^ 2) p := by
    have h_pow : (fun x : ℝ => x ^ 2) = fun x => x * x := by ext x; ring
    have h_mul_diff : DifferentiableAt ℝ (fun x => x * x) p := DifferentiableAt.mul differentiableAt_id differentiableAt_id
    exact (congr_arg (DifferentiableAt ℝ · p) h_pow.symm).mp h_mul_diff

  have hd_lin : DifferentiableAt ℝ (fun x : ℝ => π - 2 * π * x) p := by
    have h_sub : (fun x : ℝ => π - 2 * π * x) = fun x => π - (2 * π) * x := by ext x; ring
    have h_sub_diff : DifferentiableAt ℝ (fun x => π - (2 * π) * x) p := by
      apply DifferentiableAt.sub (differentiableAt_const _)
      apply DifferentiableAt.mul (differentiableAt_const _) differentiableAt_id
    exact (congr_arg (DifferentiableAt ℝ · p) h_sub.symm).mp h_sub_diff

  have h_add : deriv (fun x => (π - 2 * π * x) + x ^ 2) p = deriv (fun x => π - 2 * π * x) p + deriv (fun x => x ^ 2) p := by
    exact deriv_add hd_lin hd_x2
  rw [h_add]

  have h_lin_deriv : deriv (fun x => π - 2 * π * x) p = -2 * π := by
    have h_sub_eq : (fun x : ℝ => π - 2 * π * x) = (fun x => π) - (fun x => (2 * π) * x) := by
      ext x
      have h1 : π - 2 * π * x = π - (2 * π) * x := by ring
      rw [h1]
      rfl
    rw [h_sub_eq]
    have hd1 : DifferentiableAt ℝ (fun x : ℝ => (2 * π) * x) p := DifferentiableAt.mul (differentiableAt_const _) differentiableAt_id
    rw [deriv_sub (differentiableAt_const _) hd1, deriv_const, deriv_const_mul _ differentiableAt_id, deriv_id'']
    ring

  have h_x2_deriv : deriv (fun x : ℝ => x ^ 2) p = 2 * p := by
    have h_pow : (fun x : ℝ => x ^ 2) = fun x => x * x := by ext x; ring
    rw [h_pow]
    have hd_id : DifferentiableAt ℝ (fun x : ℝ => x) p := differentiableAt_id
    have h_mul := deriv_mul (c := fun x : ℝ => x) (d := fun x : ℝ => x) (x := p) hd_id hd_id
    rw [deriv_id''] at h_mul
    change deriv ((fun x => x) * (fun x => x)) p = 2 * p
    rw [h_mul]
    ring
  rw [h_lin_deriv, h_x2_deriv]
  ring"""

if search in content:
    with open('proofs/Calibrator.lean', 'w') as f:
        f.write(content.replace(search, replace))
    print("Patch applied successfully")
else:
    print("Search string not found")
