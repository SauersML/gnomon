// ========================================================================================
//
//                      The kernel: A pure SIMD execution engine
//
// ========================================================================================
//
// This module contains the final, innermost loop of the compute engine. It is designed
// for maximum, predictable throughput and is 100% allocation-free in the hot path.
// It functions as a "Virtual Machine" that executes a pre-compiled plan, containing
// zero scientific logic, branches, or decisions.

use std::simd::f64x8;

// --- Type Aliases for Readability ---
// These types are part of the public API of the kernel.
pub type SimdVec = f64x8;
pub const LANE_COUNT: usize = SimdVec::LEN;
pub const MAX_KERNEL_ACCUMULATOR_LANES: usize = 8;

/// Computes `acc + 2*w` for the dosage-2 (homozygous-alt) accumulation step.
///
/// `2*w` is exact in IEEE-754 (it only bumps the exponent), so for all finite,
/// non-overflowing values `fma(w, 2, acc)` rounds identically to `acc + (w + w)`
/// — the results are bit-for-bit equal. On targets with FMA this collapses the
/// doubling and the accumulate into a single `vfmadd231pd` with a folded memory
/// operand, halving the FP-add port pressure of the hot dosage-2 loop.
///
/// The `cfg` gate is mandatory on x86: `mul_add` without the `fma` target
/// feature lowers to a per-lane `fma` libm call (catastrophic in the hot loop).
/// Published x86 wheels and performance binaries target x86-64-v3 and take this
/// path; portable baseline builds use the two-add implementation. AArch64 has
/// fused scalar/vector FP in the baseline ISA and always takes the fused path.
#[inline(always)]
fn accumulate_dosage_two(acc: SimdVec, w: SimdVec) -> SimdVec {
    #[cfg(any(target_feature = "fma", target_arch = "aarch64"))]
    {
        use std::simd::StdFloat;
        w.mul_add(SimdVec::splat(2.0), acc)
    }
    #[cfg(not(any(target_feature = "fma", target_arch = "aarch64")))]
    {
        acc + (w + w)
    }
}

// ========================================================================================
//                            Public API & type definitions
// ========================================================================================

/// A validated, type-safe, zero-cost view over a slice representing a padded,
/// interleaved matrix.
///
/// This struct's constructor guarantees that an instance can only be created if its
/// dimensions and padding are coherent. This makes an invalidly-dimensioned or
/// non-padded matrix an unrepresentable state for the kernel, preventing panics
/// and memory errors in the hot loops. It is used for both the aligned weights
/// and the correction constants matrices.
pub struct PaddedInterleavedWeights<'a> {
    slice: &'a [f64],
    num_rows: usize,
    num_scores: usize,
    stride: usize,
}

impl<'a> PaddedInterleavedWeights<'a> {
    /// Creates a new, validated `PaddedInterleavedWeights` view over a slice.
    /// This is the sole entry point for creating this type. It performs a single,
    /// upfront check to ensure the slice length matches the provided dimensions
    /// and the implied padding.
    #[inline]
    pub fn new(slice: &'a [f64], num_rows: usize, num_scores: usize) -> Result<Self, &'static str> {
        // The stride is the width of a single row's data, rounded up to the
        // nearest multiple of the SIMD vector width. This padding is the
        // key to enabling branch-free, "no scalar fallback" SIMD.
        let stride = num_scores
            .div_ceil(LANE_COUNT)
            .checked_mul(LANE_COUNT)
            .ok_or("Padded score stride overflows usize")?;
        let matrix_len = num_rows
            .checked_mul(stride)
            .ok_or("Padded matrix dimensions overflow usize")?;
        if slice.len() != matrix_len {
            return Err(
                "Mismatched matrix data: slice.len() does not equal num_rows * calculated_stride",
            );
        }
        Ok(Self {
            slice,
            num_rows,
            num_scores,
            stride,
        })
    }

    /// Returns the original number of scores (K) this matrix was created with.
    #[inline(always)]
    pub fn num_scores(&self) -> usize {
        self.num_scores
    }

    /// Fetches the i-th SIMD vector for a score-window within a row.
    ///
    /// # Safety
    /// The caller MUST guarantee:
    /// - `row_idx` is in-bounds
    /// - `score_start + (lane_idx + 1) * LANE_COUNT <= self.stride`
    #[inline(always)]
    unsafe fn get_simd_lane_for_score_window_unchecked(
        &self,
        row_idx: usize,
        score_start: usize,
        lane_idx: usize,
    ) -> SimdVec {
        let offset = (row_idx * self.stride) + score_start + (lane_idx * LANE_COUNT);
        unsafe { SimdVec::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT)) }
    }
}

// ========================================================================================
//                              The kernel implementation
// ========================================================================================
/// Calculates one fixed 64-score chunk of score adjustments for a single person
/// over a mini-batch of variants.
///
/// Returns exactly 8 SIMD lanes (8 * 8 = 64 scores).
///
/// # Panics
/// Panics if the score chunk or any genotype row index is outside the matrix.
#[inline]
pub fn accumulate_adjustments_for_person(
    weights: &PaddedInterleavedWeights,
    g1_indices: &[u16],
    g2_indices: &[u16],
    score_start: usize,
) -> [SimdVec; MAX_KERNEL_ACCUMULATOR_LANES] {
    accumulate_adjustments::<MAX_KERNEL_ACCUMULATOR_LANES>(
        weights,
        g1_indices,
        g2_indices,
        score_start,
    )
}

#[inline(always)]
fn accumulate_adjustments<const LANES: usize>(
    weights: &PaddedInterleavedWeights,
    g1_indices: &[u16],
    g2_indices: &[u16],
    score_start: usize,
) -> [SimdVec; MAX_KERNEL_ACCUMULATOR_LANES] {
    assert!(
        score_start <= weights.num_scores(),
        "Invalid chunk start {score_start}; total scores={}.",
        weights.num_scores()
    );
    assert!(
        LANES * LANE_COUNT <= weights.stride - score_start,
        "Invalid kernel chunk: start={score_start}, requires {} padded scores, stride={}.",
        LANES * LANE_COUNT,
        weights.stride
    );
    let mut accumulator_buffer = [SimdVec::splat(0.0); MAX_KERNEL_ACCUMULATOR_LANES];

    // --- Loop 1: Dosage=1 Adjustments ---
    for &matrix_row_idx in g1_indices {
        let matrix_row_idx = matrix_row_idx as usize;
        assert!(
            matrix_row_idx < weights.num_rows,
            "Genotype row index out of bounds"
        );
        // This inner loop over score columns is the same performant structure as the original kernel.
        for i in 0..LANES {
            unsafe {
                let weights_vec = weights.get_simd_lane_for_score_window_unchecked(
                    matrix_row_idx,
                    score_start,
                    i,
                );
                *accumulator_buffer.get_unchecked_mut(i) += weights_vec;
            }
        }
    }

    // --- Loop 2: Dosage=2 Adjustments ---
    for &matrix_row_idx in g2_indices {
        let matrix_row_idx = matrix_row_idx as usize;
        assert!(
            matrix_row_idx < weights.num_rows,
            "Genotype row index out of bounds"
        );
        for i in 0..LANES {
            unsafe {
                let weights_vec = weights.get_simd_lane_for_score_window_unchecked(
                    matrix_row_idx,
                    score_start,
                    i,
                );
                let acc = accumulator_buffer.get_unchecked_mut(i);
                *acc = accumulate_dosage_two(*acc, weights_vec);
            }
        }
    }

    accumulator_buffer
}

/// Calculates a variable-width score chunk (up to 64 scores) of score
/// adjustments for a single person over a mini-batch of variants.
///
/// `lane_count` is the number of SIMD lanes to compute from `score_start`.
///
/// # Panics
/// Panics if the score chunk or any genotype row index is outside the matrix.
#[inline]
pub fn accumulate_adjustments_for_person_lanes(
    weights: &PaddedInterleavedWeights,
    g1_indices: &[u16],
    g2_indices: &[u16],
    score_start: usize,
    lane_count: usize,
) -> [SimdVec; MAX_KERNEL_ACCUMULATOR_LANES] {
    // Select once per person and score stripe. Constant-width loops keep each
    // accumulator in a register, including partial stripes of fewer than 64 scores.
    match lane_count {
        0 => accumulate_adjustments::<0>(weights, g1_indices, g2_indices, score_start),
        1 => accumulate_adjustments::<1>(weights, g1_indices, g2_indices, score_start),
        2 => accumulate_adjustments::<2>(weights, g1_indices, g2_indices, score_start),
        3 => accumulate_adjustments::<3>(weights, g1_indices, g2_indices, score_start),
        4 => accumulate_adjustments::<4>(weights, g1_indices, g2_indices, score_start),
        5 => accumulate_adjustments::<5>(weights, g1_indices, g2_indices, score_start),
        6 => accumulate_adjustments::<6>(weights, g1_indices, g2_indices, score_start),
        7 => accumulate_adjustments::<7>(weights, g1_indices, g2_indices, score_start),
        8 => accumulate_adjustments::<8>(weights, g1_indices, g2_indices, score_start),
        _ => panic!("Invalid lane_count={lane_count}; max={MAX_KERNEL_ACCUMULATOR_LANES}."),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjustments_do_not_round_weights_or_partial_sums_to_f32() {
        let mut data = [16_777_216.0f64; 16];
        data[8..].fill(1.0 + f64::EPSILON);
        let weights = PaddedInterleavedWeights::new(&data, 2, 8).unwrap();
        let lone = accumulate_adjustments_for_person_lanes(&weights, &[1], &[], 0, 1);
        assert_eq!(lone[0].to_array(), [1.0 + f64::EPSILON; 8]);
        let combined = accumulate_adjustments_for_person_lanes(&weights, &[0, 1], &[], 0, 1);
        assert_eq!(combined[0].to_array(), [16_777_217.0; 8]);
    }

    #[test]
    fn padded_matrix_rejects_overflowing_dimensions() {
        assert!(PaddedInterleavedWeights::new(&[], 0, usize::MAX).is_err());
        assert!(PaddedInterleavedWeights::new(&[], usize::MAX / LANE_COUNT + 1, 1).is_err());
    }

    macro_rules! invalid_row_test {
        ($name:ident, $full:expr, $g1:expr, $g2:expr) => {
            #[test]
            #[should_panic(expected = "Genotype row index out of bounds")]
            fn $name() {
                let data = [1.0; 64];
                let weights = PaddedInterleavedWeights::new(&data, 1, 64).expect("matrix");
                if $full {
                    accumulate_adjustments_for_person(&weights, $g1, $g2, 0);
                } else {
                    accumulate_adjustments_for_person_lanes(&weights, $g1, $g2, 0, 1);
                }
            }
        };
    }
    invalid_row_test!(fixed_kernel_rejects_invalid_dosage_one_row, true, &[1], &[]);
    invalid_row_test!(fixed_kernel_rejects_invalid_dosage_two_row, true, &[], &[1]);
    invalid_row_test!(tail_kernel_rejects_invalid_dosage_one_row, false, &[1], &[]);
    invalid_row_test!(tail_kernel_rejects_invalid_dosage_two_row, false, &[], &[1]);

    #[test]
    fn all_score_stripe_widths_preserve_accumulation_order() {
        let data: Vec<f64> = (0..256 * 136)
            .map(|i| ((i * 37 % 1021) as f64 - 511.0) / 17.0)
            .collect();
        let weights = PaddedInterleavedWeights::new(&data, 256, 131).expect("matrix");
        let g1: Vec<u16> = (0..256).filter(|i| i % 3 == 0).collect();
        let g2: Vec<u16> = (0..256).filter(|i| i % 3 == 1).collect();
        for start in [0, 64] {
            for lanes in 0..=8 {
                let actual =
                    accumulate_adjustments_for_person_lanes(&weights, &g1, &g2, start, lanes);
                for score in 0..64 {
                    let mut expected = 0.0f64;
                    if score < lanes * LANE_COUNT {
                        for &row in &g1 {
                            expected += data[row as usize * 136 + start + score];
                        }
                        for &row in &g2 {
                            expected += 2.0 * data[row as usize * 136 + start + score];
                        }
                    }
                    assert_eq!(actual[score / 8][score % 8], expected);
                }
            }
        }
    }

    #[test]
    fn checked_kernels_preserve_fixed_and_tail_accumulations() {
        let data: Vec<f64> = (0..3 * 72).map(|value| value as f64 / 4.0).collect();
        let weights = PaddedInterleavedWeights::new(&data, 3, 67).expect("matrix");
        let fixed = accumulate_adjustments_for_person(&weights, &[0, 2], &[1], 0);
        let tail = accumulate_adjustments_for_person_lanes(&weights, &[0, 2], &[1], 64, 1);
        for score in 0..72 {
            let actual = if score < 64 {
                fixed[score / 8][score % 8]
            } else {
                tail[0][score - 64]
            };
            assert_eq!(
                actual,
                data[score] + data[144 + score] + 2.0 * data[72 + score]
            );
        }
    }
}
