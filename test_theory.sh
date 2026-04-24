#!/bin/bash
export PATH="$HOME/.elan/bin:$PATH"
cd proofs
cat << 'LEAN' > patch_test.lean
import Mathlib

noncomputable def haplotypePhasePredictionError
    (freq_cis interaction_cis interaction_trans : ℝ) : ℝ :=
  freq_cis * (interaction_cis - interaction_cis) ^ 2 +
    (1 - freq_cis) * (interaction_trans - interaction_trans) ^ 2

noncomputable def haplotypeTransportBias
    (_freq_cis_source freq_cis_target interaction_cis interaction_trans : ℝ) : ℝ :=
  |freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans -
    (freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans)|

LEAN
lake env lean patch_test.lean
