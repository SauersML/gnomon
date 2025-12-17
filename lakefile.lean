import Lake
open Lake DSL

package calibration where

-- Pin to a specific Mathlib commit for reproducible builds.
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.26.0"

@[default_target]
lean_lib Calibrator where
  srcDir := "proofs"
