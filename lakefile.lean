import Lake
open Lake DSL

package calibration where
  srcDir := "proofs"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.12.0"

@[default_target]
lean_lib Calibrator where
