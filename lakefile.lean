import Lake
open Lake DSL

package calibration where

-- Pin to a specific Mathlib commit for reproducible builds.
-- This commit (v4.20.1 toolchain) is from late 2024.
require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.20.1"

@[default_target]
lean_lib Calibrator where
  srcDir := "proofs"
