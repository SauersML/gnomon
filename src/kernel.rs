// ========================================================================================
//
//                      THE KERNEL: A PURE SIMD EXECUTION ENGINE
//
// ========================================================================================
//
// This module contains the final, innermost loop of the compute engine. It is designed
// for maximum, predictable throughput and is 100% allocation-free in the hot path.
// It functions as a "Virtual Machine" that executes a pre-compiled plan, containing
// zero scientific logic, branches, or decisions.

use std::simd::{f32x8, StdFloat};

// --- Type Aliases for Readability ---
// These types are part of the public API of the kernel.
pub type SimdVec = f32x8;
pub const LANE_COUNT: usize = SimdVec::LEN;

// ========================================================================================
//                            PUBLIC API & TYPE DEFINITIONS
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
    slice: &'a [f32],
    num_scores: usize,
    stride: usize,
}

impl<'a> PaddedInterleavedWeights<'a> {
    /// Creates a new, validated `PaddedInterleavedWeights` view over a slice.
    /// This is the sole entry point for creating this type. It performs a single,
    /// upfront check to ensure the slice length matches the provided dimensions
    /// and the implied padding.
    #[inline]
    pub fn new(
        slice: &'a [f32],
        num_rows: usize,
        num_scores: usize,
    ) -> Result<Self, &'static str> {
        // The stride is the width of a single row's data, rounded up to the
        // nearest multiple of the SIMD vector width. This padding is the
        // key to enabling branch-free, "no scalar fallback" SIMD.
        let stride = (num_scores + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
        if slice.len() != num_rows * stride {
            return Err(
                "Mismatched matrix data: slice.len() does not equal num_rows * calculated_stride",
            );
        }
        Ok(Self {
            slice,
            num_scores,
            stride,
        })
    }

    /// Returns the original number of scores (K) this matrix was created with.
    #[inline(always)]
    pub fn num_scores(&self) -> usize {
        self.num_scores
    }

    /// Fetches the i-th SIMD vector of data for a given matrix row.
    ///
    /// # Safety
    /// The caller MUST guarantee that `row_idx` is a valid row index for this matrix
    /// and `lane_idx < self.stride / LANE_COUNT`. This contract is upheld by
    /// `accumulate_scores_for_person`, whose loops are correctly bounded.
    #[inline(always)]
    unsafe fn get_simd_lane_unchecked(&self, row_idx: usize, lane_idx: usize) -> SimdVec {
        // The offset calculation uses the pre-computed `stride` to correctly
        // jump between the start of each variant's padded data block.
        let offset = (row_idx * self.stride) + (lane_idx * LANE_COUNT);
        // SAFETY: The `unsafe fn` contract guarantees the offset is in-bounds. The
        // struct's constructor guarantees the data is padded, so reading a full
        // `LANE_COUNT` of floats is always safe.
        unsafe { SimdVec::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT)) }
    }
}

// ========================================================================================
//                              THE KERNEL IMPLEMENTATION
// ========================================================================================

/// Calculates all K scores for a single person using the "three-loop" strategy
/// on pre-compiled data matrices and pre-computed sparse indices.
///
/// This function is the heart of the compute engine. It is branch-free and
/// performs a minimal number of predictable memory accesses, maximizing throughput.
///
/// # Safety
///
/// The caller must uphold the following contracts:
/// - All indices in `g1_indices` and `g2_indices` must be valid row indices
///   for the `aligned_weights` and `correction_constants` matrices.
/// - The dimensions of both matrices must match.
/// - `scores_out.len()` must be `== aligned_weights.num_scores()`.
/// - `accumulator_buffer.len()` must be sufficient for the number of scores.
#[inline]
pub fn accumulate_scores_for_person(
    aligned_weights: &PaddedInterleavedWeights,
    correction_constants: &PaddedInterleavedWeights,
    scores_out: &mut [f32],
    accumulator_buffer: &mut [SimdVec],
    g1_indices: &[usize],
    g2_indices: &[usize],
) {
    let num_scores = aligned_weights.num_scores();
    let num_accumulator_lanes = (num_scores + LANE_COUNT - 1) / LANE_COUNT;
    let dosage_2_multiplier = SimdVec::splat(2.0);

    // --- Reset the Reusable Accumulator Buffer ---
    accumulator_buffer.iter_mut().for_each(|acc| *acc = SimdVec::splat(0.0));

    // --- Loop 1: Accumulate Aligned Weights (W') for Dosage=1 Variants ---
    for &matrix_row in g1_indices {
        for i in 0..num_accumulator_lanes {
            // SAFETY: All indices and buffer lengths are guaranteed by the caller's contract.
            // Using get_unchecked here is a critical performance optimization.
            unsafe {
                let weights_vec = aligned_weights.get_simd_lane_unchecked(matrix_row, i);
                *accumulator_buffer.get_unchecked_mut(i) += weights_vec;
            }
        }
    }

    // --- Loop 2: Accumulate Aligned Weights (W') for Dosage=2 Variants ---
    for &matrix_row in g2_indices {
        for i in 0..num_accumulator_lanes {
            // SAFETY: All indices and buffer lengths are guaranteed by the caller's contract.
            unsafe {
                let weights_vec = aligned_weights.get_simd_lane_unchecked(matrix_row, i);
                let acc = accumulator_buffer.get_unchecked_mut(i);
                *acc = weights_vec.mul_add(dosage_2_multiplier, *acc);
            }
        }
    }

    // --- Loop 3: Add Correction Constants (C) for All Non-Zero Dosage Variants ---
    // This loop adds the pre-computed `C` term from the formula `G*W' + C`.
    // It runs over all variants that contributed to the score, ensuring correctness.
    for &matrix_row in g1_indices.iter().chain(g2_indices.iter()) {
        for i in 0..num_accumulator_lanes {
            // SAFETY: All indices and buffer lengths are guaranteed by the caller's contract.
            unsafe {
                let correction_vec = correction_constants.get_simd_lane_unchecked(matrix_row, i);
                *accumulator_buffer.get_unchecked_mut(i) += correction_vec;
            }
        }
    }

    // --- Final Horizontal Store ---
    // The results, which have been accumulated vertically in SIMD vectors, are
    // now written out to the final destination slice.
    for (i, &acc_vec) in accumulator_buffer.iter().enumerate() {
        let start = i * LANE_COUNT;
        let end = (start + LANE_COUNT).min(num_scores);
        // This temporary array conversion is the correct and safe way to handle
        // the final, possibly partial, vector without a scalar fallback loop.
        let temp_array = acc_vec.to_array();
        scores_out[start..end].copy_from_slice(&temp_array[..(end - start)]);
    }
}
