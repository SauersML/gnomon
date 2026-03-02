import re

with open('proofs/Calibrator.lean', 'r') as f:
    content = f.read()

search = """  change deriv (fun x => (fun y => π - 2 * π * y) x + (fun y => y ^ 2) x) p = 2 * (p - π)
  rw [deriv_add hd_lin hd_x2]"""

replace = """  have h_add : deriv (fun x => (fun y => π - 2 * π * y) x + (fun y => y ^ 2) x) p = deriv (fun x => π - 2 * π * x) p + deriv (fun x => x ^ 2) p := by
    exact deriv_add hd_lin hd_x2
  rw [h_add]"""

if search in content:
    with open('proofs/Calibrator.lean', 'w') as f:
        f.write(content.replace(search, replace))
    print("Patch applied successfully")
else:
    print("Search string not found")
