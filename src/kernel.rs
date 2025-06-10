// ========================================================================================
//
//                                     THE KERNEL
//
// ========================================================================================
//
// This module contains the final, innermost loop of the compute engine. It is designed
// for maximum throughput and is 100% allocation-free in the hot path.

use std::simd::{f32x8, prelude::*, StdFloat};

// --- Type Aliases for Readability ---
// These types are part of the public API of the kernel.
pub type SimdVec = f32x8;
pub const LANE_COUNT: usize = SimdVec::LANE_COUNT;


// ========================================================================================
//                            PUBLIC API & TYPE DEFINITIONS
// ========================================================================================

/// A validated, type-safe, zero-cost wrapper for the interleaved weights matrix.
///
/// This struct's constructor guarantees that an instance can only be created if its
/// dimensions are coherent, making an invalidly-dimensioned matrix an unrepresentable state.
pub struct InterleavedWeights<'a> {
    slice: &'a [f32],
    num_snps: usize,
    num_scores: usize,
}

impl<'a> InterleavedWeights<'a> {
    /// Creates a new, validated `InterleavedWeights` view over a slice.
    ///
    /// This is the sole entry point for creating this type. It performs a single,
    /// upfront check to ensure the slice length matches the provided dimensions.
    #[inline]
    pub fn new(
        slice: &'a [f32],
        num_snps: usize,
        num_scores: usize,
    ) -> Result<Self, &'static str> {
        if slice.len() != num_snps * num_scores {
            return Err("Mismatched weights: slice.len() does not equal num_snps * num_scores");
        }
        Ok(Self {
            slice,
            num_snps,
            num_scores,
        })
    }

    /// Returns the number of scores (K) this matrix was created with.
    #[inline(always)]
    pub fn num_scores(&self) -> usize {
        self.num_scores
    }

    /// Fetches the i-th SIMD vector of weights for a given SNP.
    ///
    /// # Safety
    /// The caller MUST guarantee that `snp_idx < self.num_snps` and
    /// `lane_idx < (self.num_scores + LANE_COUNT - 1) / LANE_COUNT`.
    /// This contract is upheld by `accumulate_scores_for_person`, whose loops
    /// are correctly bounded.
    #[inline(always)]
    unsafe fn get_simd_lane_unchecked(&self, snp_idx: usize, lane_idx: usize) -> SimdVec {
        // This is the single, minimal `unsafe` operation. Its correctness is
        // guaranteed by the disciplined logic of its sole caller.
        let offset = (snp_idx * self.num_scores) + (lane_idx * LANE_COUNT);
        // SAFETY: The `unsafe fn` contract guarantees the offset is in-bounds.
        // This inner `unsafe` block is required by modern Rust to make the
        // exact location of the unsafe operation explicit.
        unsafe { SimdVec::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT)) }
    }
}

// ========================================================================================
//                              THE KERNEL IMPLEMENTATION
// ========================================================================================

/// Calculates all K scores for a single person using pre-computed sparse indices.
///
/// This function is the heart of the compute engine. It is significantly faster
/// than its predecessor because it no longer performs a redundant scan for non-zero
/// genotypes; that work is now done once in `batch.rs`.
///
/// # Safety
///
/// The caller must uphold the following contracts:
/// - All indices in `g1_indices` and `g2_indices` must be `< weights.num_snps`.
/// - `scores_out.len()` must be `== weights.num_scores()`.
/// - `accumulator_buffer.len()` must be sufficient for the number of scores, i.e.,
///   at least `(weights.num_scores() + LANE_COUNT - 1) / LANE_COUNT`.
#[inline]
pub fn accumulate_scores_for_person(
    weights: &InterleavedWeights,
    scores_out: &mut [f32],
    // --- Reusable, thread-local buffer ---
    accumulator_buffer: &mut [SimdVec],
    // --- Pre-computed sparse indices from batch.rs ---
    g1_indices: &[usize],
    g2_indices: &[usize],
) {
    let num_scores = weights.num_scores();
    let num_accumulator_lanes = (num_scores + LANE_COUNT - 1) / LANE_COUNT;
    let dosage_2_multiplier = SimdVec::splat(2.0);
    // --- Reset the Reusable Accumulator Buffer ---
    // Zero-out the buffer provided by the caller. This is a fast, predictable
    // operation on a buffer that should be hot in the CPU cache.
    for acc in accumulator_buffer.iter_mut() {
        *acc = SimdVec::splat(0.0);
    }

    // --- The Fused-Multiply-Add Update Loop ---
    // This is the performance-critical core. Every `snp_idx` is guaranteed to be
    // in-bounds by the function's safety contract.
    for &snp_idx in g1_indices {
        for i in 0..num_accumulator_lanes {
            let weights_vec = unsafe { weights.get_simd_lane_unchecked(snp_idx, i) };
            // SAFETY: The `accumulator_buffer` slice is guaranteed by the caller to have
            // `num_accumulator_lanes` elements, and this loop runs from `0..num_accumulator_lanes`,
            // so `i` is always in bounds. This allows `get_unchecked_mut` to be used to
            // remove bounds-checking overhead in this hot loop.
            unsafe {
                *accumulator_buffer.get_unchecked_mut(i) += weights_vec;
            }
        }
    }

    for &snp_idx in g2_indices {
        for i in 0..num_accumulator_lanes {
            let weights_vec = unsafe { weights.get_simd_lane_unchecked(snp_idx, i) };
            // SAFETY: The `accumulator_buffer` slice is guaranteed by the caller to have
            // `num_accumulator_lanes` elements, and this loop runs from `0..num_accumulator_lanes`,
            // so `i` is always in bounds. This allows `get_unchecked_mut` to be used to
            // remove bounds-checking overhead in this hot loop.
            unsafe {
                let acc = accumulator_buffer.get_unchecked_mut(i);
                *acc = weights_vec.mul_add(dosage_2_multiplier, *acc);
            }
        }
    }

    // --- Final Horizontal Store ---
    // The results, which have been accumulated vertically in SIMD registers, are
    // now written out to the final destination slice.
    for (i, &acc_vec) in accumulator_buffer.iter().enumerate() {
        let start = i * LANE_COUNT;
        let end = (start + LANE_COUNT).min(num_scores);
        acc_vec.copy_to_slice(&mut scores_out[start..end]);
    }
}
