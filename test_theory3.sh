#!/bin/bash
export PATH="$HOME/.elan/bin:$PATH"
cd proofs
cat << 'LEAN' > patch_test3.lean
import Mathlib

noncomputable def dosagePhaseMisspecificationError
    (freq_cis interaction_cis interaction_trans : ℝ) : ℝ :=
  freq_cis * (interaction_cis - (freq_cis * interaction_cis + (1 - freq_cis) * interaction_trans)) ^ 2 +
    (1 - freq_cis) *
      (interaction_trans - (freq_cis * interaction_cis + (1 - freq_cis) * interaction_trans)) ^ 2

noncomputable def dosageTransportBias
    (freq_cis_source freq_cis_target interaction_cis interaction_trans : ℝ) : ℝ :=
  |(freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans) -
    (freq_cis_source * interaction_cis + (1 - freq_cis_source) * interaction_trans)|

theorem dosagePhaseMisspecificationError_eq
    (freq_cis interaction_cis interaction_trans : ℝ) :
    dosagePhaseMisspecificationError freq_cis interaction_cis interaction_trans =
      freq_cis * (1 - freq_cis) * (interaction_cis - interaction_trans) ^ 2 := by
  unfold dosagePhaseMisspecificationError
  ring

theorem dosageTransportBias_eq
    (freq_cis_source freq_cis_target interaction_cis interaction_trans : ℝ) :
    dosageTransportBias freq_cis_source freq_cis_target interaction_cis interaction_trans =
      |freq_cis_target - freq_cis_source| * |interaction_cis - interaction_trans| := by
  unfold dosageTransportBias
  have h_factor :
      freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans -
        (freq_cis_source * interaction_cis + (1 - freq_cis_source) * interaction_trans) =
        (freq_cis_target - freq_cis_source) * (interaction_cis - interaction_trans) := by
    ring
  rw [h_factor, abs_mul]

noncomputable def haplotypePhasePredictionError
    (freq_cis interaction_cis interaction_trans pred_cis pred_trans : ℝ) : ℝ :=
  freq_cis * (interaction_cis - pred_cis) ^ 2 +
    (1 - freq_cis) * (interaction_trans - pred_trans) ^ 2

noncomputable def haplotypeTransportBias
    (freq_cis_target interaction_cis interaction_trans pred_cis pred_trans : ℝ) : ℝ :=
  |(freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans) -
    (freq_cis_target * pred_cis + (1 - freq_cis_target) * pred_trans)|

theorem compound_het_not_captured_by_dosage
    (freq_cis interaction_cis interaction_trans : ℝ)
    (h_freq : 0 < freq_cis ∧ freq_cis < 1)
    (h_phase_gap : interaction_cis ≠ interaction_trans) :
    haplotypePhasePredictionError freq_cis interaction_cis interaction_trans interaction_cis interaction_trans <
      dosagePhaseMisspecificationError freq_cis interaction_cis interaction_trans := by
  rcases h_freq with ⟨h_freq_pos, h_freq_lt_one⟩
  rw [dosagePhaseMisspecificationError_eq]
  have h_hap : haplotypePhasePredictionError freq_cis interaction_cis interaction_trans interaction_cis interaction_trans = 0 := by
    unfold haplotypePhasePredictionError; ring
  rw [h_hap]
  have h_gap_sq : 0 < (interaction_cis - interaction_trans) ^ 2 := by
    exact sq_pos_of_ne_zero (sub_ne_zero.mpr h_phase_gap)
  have h_mix : 0 < freq_cis * (1 - freq_cis) := by
    exact mul_pos h_freq_pos (sub_pos.mpr h_freq_lt_one)
  exact mul_pos h_mix h_gap_sq

theorem haplotype_pgs_at_least_snp
    (freq_cis interaction_cis interaction_trans : ℝ)
    (h_freq_nonneg : 0 ≤ freq_cis) (h_freq_le_one : freq_cis ≤ 1) :
    haplotypePhasePredictionError freq_cis interaction_cis interaction_trans interaction_cis interaction_trans ≤
      dosagePhaseMisspecificationError freq_cis interaction_cis interaction_trans := by
  rw [dosagePhaseMisspecificationError_eq]
  have h_hap : haplotypePhasePredictionError freq_cis interaction_cis interaction_trans interaction_cis interaction_trans = 0 := by
    unfold haplotypePhasePredictionError; ring
  rw [h_hap]
  have h_mix_nonneg : 0 ≤ freq_cis * (1 - freq_cis) := by
    exact mul_nonneg h_freq_nonneg (sub_nonneg.mpr h_freq_le_one)
  exact mul_nonneg h_mix_nonneg (sq_nonneg _)

theorem haplotype_pgs_more_portable_for_cis
    (freq_cis_source freq_cis_target interaction_cis interaction_trans : ℝ)
    (h_freq_shift : freq_cis_source ≠ freq_cis_target)
    (h_phase_gap : interaction_cis ≠ interaction_trans) :
    haplotypeTransportBias freq_cis_target interaction_cis interaction_trans interaction_cis interaction_trans <
      dosageTransportBias freq_cis_source freq_cis_target interaction_cis interaction_trans := by
  rw [dosageTransportBias_eq]
  have h_hap : haplotypeTransportBias freq_cis_target interaction_cis interaction_trans interaction_cis interaction_trans = 0 := by
    unfold haplotypeTransportBias
    have h_inner : freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans -
      (freq_cis_target * interaction_cis + (1 - freq_cis_target) * interaction_trans) = 0 := sub_self _
    rw [h_inner]
    exact abs_zero
  rw [h_hap]
  exact mul_pos
    (abs_pos.mpr (sub_ne_zero.mpr h_freq_shift.symm))
    (abs_pos.mpr (sub_ne_zero.mpr h_phase_gap))

LEAN
lake env lean patch_test3.lean
