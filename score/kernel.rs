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

use std::simd::f32x8;

// --- Type Aliases for Readability ---
// These types are part of the public API of the kernel.
pub type SimdVec = f32x8;
pub const LANE_COUNT: usize = SimdVec::LEN;

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
    pub fn new(slice: &'a [f32], num_rows: usize, num_scores: usize) -> Result<Self, &'static str> {
        // The stride is the width of a single row's data, rounded up to the
        // nearest multiple of the SIMD vector width. This padding is the
        // key to enabling branch-free, "no scalar fallback" SIMD.
        let stride = num_scores.div_ceil(LANE_COUNT) * LANE_COUNT;
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
/// Calculates the score *adjustments* for a single person over a mini-batch of variants.
/// The caller provides a reusable accumulator buffer to avoid per-person allocations.
/// The internal loop structure is designed to maximize instruction-level parallelism
/// by processing all score columns for a variant before moving to the next.
///
/// # Safety
/// The caller must uphold the contract that all indices in `g1_indices` and `g2_indices`
/// are valid row indices for the `weights` matrix, which represents
/// a view of a single mini-batch.
#[inline]
pub fn accumulate_adjustments_for_person_into(
    weights: &PaddedInterleavedWeights,
    g1_indices: &[u16],
    g2_indices: &[u16],
    accumulator_buffer: &mut [SimdVec],
) {
    accumulate_adjustments_for_person_window_into(
        weights,
        g1_indices,
        g2_indices,
        0,
        weights.num_scores(),
        accumulator_buffer,
    );
}

/// Same as `accumulate_adjustments_for_person_into`, but limited to a score window.
#[inline]
pub fn accumulate_adjustments_for_person_window_into(
    weights: &PaddedInterleavedWeights,
    g1_indices: &[u16],
    g2_indices: &[u16],
    score_start: usize,
    score_count: usize,
    accumulator_buffer: &mut [SimdVec],
) {
    assert!(
        score_start <= weights.num_scores(),
        "Invalid score window start {score_start}; total scores={}.",
        weights.num_scores()
    );
    assert!(
        score_start + score_count <= weights.num_scores(),
        "Invalid score window [{score_start}, {}); total scores={}.",
        score_start + score_count,
        weights.num_scores()
    );
    let num_accumulator_lanes = score_count.div_ceil(LANE_COUNT);
    assert!(
        accumulator_buffer.len() >= num_accumulator_lanes,
        "Accumulator buffer too small: need {num_accumulator_lanes} lanes, got {}.",
        accumulator_buffer.len()
    );
    accumulator_buffer[..num_accumulator_lanes].fill(SimdVec::splat(0.0));

    // --- Loop 1: Dosage=1 Adjustments ---
    for &matrix_row_idx in g1_indices {
        let matrix_row_idx = matrix_row_idx as usize;
        // This inner loop over score columns is the same performant structure as the original kernel.
        for i in 0..num_accumulator_lanes {
            unsafe {
                let weights_vec =
                    weights.get_simd_lane_for_score_window_unchecked(matrix_row_idx, score_start, i);
                *accumulator_buffer.get_unchecked_mut(i) += weights_vec;
            }
        }
    }

    // --- Loop 2: Dosage=2 Adjustments ---
    for &matrix_row_idx in g2_indices {
        let matrix_row_idx = matrix_row_idx as usize;
        for i in 0..num_accumulator_lanes {
            unsafe {
                let weights_vec =
                    weights.get_simd_lane_for_score_window_unchecked(matrix_row_idx, score_start, i);
                let adj = weights_vec + weights_vec;
                *accumulator_buffer.get_unchecked_mut(i) += adj;
            }
        }
    }
}
