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

use std::simd::{cmp::SimdPartialEq, f32x8, u8x8, Simd};

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

/// A validated, type-safe, zero-cost view over a slice representing a padded,
/// interleaved matrix of `u8` flags.
pub struct PaddedInterleavedFlags<'a> {
    slice: &'a [u8],
    stride: usize,
}

impl<'a> PaddedInterleavedFlags<'a> {
    /// Creates a new, validated `PaddedInterleavedFlags` view over a slice.
    #[inline]
    pub fn new(
        slice: &'a [u8],
        num_rows: usize,
        num_scores: usize,
    ) -> Result<Self, &'static str> {
        let stride = (num_scores + LANE_COUNT - 1) / LANE_COUNT * LANE_COUNT;
        if slice.len() != num_rows * stride {
            return Err(
                "Mismatched flag matrix data: slice.len() does not equal num_rows * calculated_stride",
            );
        }
        Ok(Self { slice, stride })
    }

    /// Fetches the i-th SIMD vector of flags for a given matrix row.
    ///
    /// # Safety
    /// The caller MUST guarantee that `row_idx` is a valid row index and `lane_idx` is valid.
    #[inline(always)]
    unsafe fn get_simd_lane_unchecked(&self, row_idx: usize, lane_idx: usize) -> Simd<u8, 8> {
        let offset = (row_idx * self.stride) + (lane_idx * LANE_COUNT);
        unsafe { Simd::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT)) }
    }
}


// ========================================================================================
//                              THE KERNEL IMPLEMENTATION
// ========================================================================================

/// Calculates the score contributions for a single person using a numerically stable,
/// branchless, SIMD-accelerated algorithm.
///
/// This function is the heart of the compute engine. It executes a pre-compiled
/// plan (`weights`, `flip_flags`, `g1_indices`, `g2_indices`) to maximize throughput.
///
/// # Safety
/// The caller must uphold the following contracts:
/// - All indices in `g1_indices` and `g2_indices` must be valid row indices
///   for the `weights` and `flip_flags` matrices.
/// - `scores_out.len()` must be `== weights.num_scores()`.
/// - `accumulator_buffer.len()` must be sufficient for the number of scores.
#[inline]
pub fn accumulate_scores_for_person(
    weights: &PaddedInterleavedWeights,
    flip_flags: &PaddedInterleavedFlags,
    scores_out: &mut [f32],
    accumulator_buffer: &mut [SimdVec],
    g1_indices: &[usize],
    g2_indices: &[usize],
) {
    let num_scores = weights.num_scores();
    let num_accumulator_lanes = (num_scores + LANE_COUNT - 1) / LANE_COUNT;
    let two = SimdVec::splat(2.0);

    // --- Reset the Reusable Accumulator Buffer ---
    accumulator_buffer
        .iter_mut()
        .for_each(|acc| *acc = SimdVec::splat(0.0));

    // --- Loop 1: Accumulate contributions for Dosage=1 variants ---
    let dosage1 = SimdVec::splat(1.0);
    for &matrix_row_idx in g1_indices {
        for i in 0..num_accumulator_lanes {
            // SAFETY: All indices and buffer lengths are guaranteed by the caller's contract.
            // Using get_unchecked here is a critical performance optimization.
            unsafe {
                let weights_vec = weights.get_simd_lane_unchecked(matrix_row_idx, i);
                let flip_flags_u8 = flip_flags.get_simd_lane_unchecked(matrix_row_idx, i);

                // Create a SIMD boolean mask from the u8 flags.
                let flip_mask = flip_flags_u8.simd_eq(u8x8::splat(1));

                // Calculate BOTH potential outcomes in parallel across all 8 lanes.
                let contrib_regular = dosage1 * weights_vec;
                let contrib_flipped = (two - dosage1) * weights_vec;

                // Use a single, branchless instruction (e.g., blendvps) to select
                // the correct contribution for each of the 8 lanes.
                        let contribution = flip_mask.cast().select(contrib_flipped, contrib_regular);

                *accumulator_buffer.get_unchecked_mut(i) += contribution;
            }
        }
    }

    // --- Loop 2: Accumulate contributions for Dosage=2 variants ---
    let dosage2 = SimdVec::splat(2.0);
    for &matrix_row_idx in g2_indices {
        for i in 0..num_accumulator_lanes {
            // SAFETY: All indices and buffer lengths are guaranteed by the caller's contract.
            unsafe {
                let weights_vec = weights.get_simd_lane_unchecked(matrix_row_idx, i);
                let flip_flags_u8 = flip_flags.get_simd_lane_unchecked(matrix_row_idx, i);

                let flip_mask = flip_flags_u8.simd_eq(u8x8::splat(1));

                let contrib_regular = dosage2 * weights_vec;
                // For dosage=2, the flipped contribution is (2-2)*w = 0.
                let contrib_flipped = SimdVec::splat(0.0);

                        let contribution = flip_mask.cast().select(contrib_flipped, contrib_regular);

                *accumulator_buffer.get_unchecked_mut(i) += contribution;
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
