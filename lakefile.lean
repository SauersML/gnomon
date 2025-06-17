import Lake
open Lake DSL

package calibration where
  srcDir := "proofs"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib Calibrator where
