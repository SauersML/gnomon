import Mathlib

noncomputable def averagePhaseInteraction
    (freq_cis interaction_cis interaction_trans : ℝ) : ℝ :=
  freq_cis * interaction_cis + (1 - freq_cis) * interaction_trans

noncomputable def haplotypePhasePredictionError
    (freq_cis interaction_cis interaction_trans pred_cis pred_trans : ℝ) : ℝ :=
  freq_cis * (interaction_cis - pred_cis) ^ 2 +
    (1 - freq_cis) * (interaction_trans - pred_trans) ^ 2

noncomputable def haplotypeTransportBias
    (freq_cis_target interaction_cis interaction_trans pred_cis pred_trans : ℝ) : ℝ :=
  |averagePhaseInteraction freq_cis_target interaction_cis interaction_trans -
    (freq_cis_target * pred_cis + (1 - freq_cis_target) * pred_trans)|

theorem test_haplotypePhasePredictionError_zero
    (freq_cis interaction_cis interaction_trans : ℝ) :
    haplotypePhasePredictionError freq_cis interaction_cis interaction_trans interaction_cis interaction_trans = 0 := by
  unfold haplotypePhasePredictionError
  ring

theorem test_haplotypeTransportBias_zero
    (freq_cis_target interaction_cis interaction_trans : ℝ) :
    haplotypeTransportBias freq_cis_target interaction_cis interaction_trans interaction_cis interaction_trans = 0 := by
  unfold haplotypeTransportBias averagePhaseInteraction
  have h_inner : freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans -
    (freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans) = 0 := sub_self _
  rw [h_inner]
  exact abs_zero
