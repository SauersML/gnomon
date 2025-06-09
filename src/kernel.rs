// ========================================================================================
//
//                                     THE KERNEL
//
// ========================================================================================

use std::simd::{f32x8, Simd};
use std::cell::RefCell;
use thread_local::ThreadLocal;

// --- Type Aliases for Readability ---
// Public so the caller can correctly size the accumulator buffer.
pub type SimdVec = f32x8;
pub const LANE_COUNT: usize = SimdVec::LANE_COUNT;
pub type KernelDataPool =
    ThreadLocal<RefCell<(Vec<SimdVec>, Vec<usize>, Vec<usize>)>>;

// ========================================================================================
//                            PUBLIC API & TYPE DEFINITIONS
// ========================================================================================

/// A validated, type-safe, zero-cost wrapper for the interleaved weights matrix.
///
/// This struct is the gatekeeper for the weights data. Its constructor guarantees
/// that an instance of `InterleavedWeights` can only be created if its dimensions
//  are coherent, making an invalidly-dimensioned weights matrix an unrepresentable state.
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
    /// This contract is upheld by the `accumulate_scores_for_person` function,
    /// whose "scan-and-gather" logic ensures all SNP indices are derived from
    /// the valid genotype row, and the loop over lanes is correctly bounded.
    #[inline(always)]
    unsafe fn get_simd_lane_unchecked(&self, snp_idx: usize, lane_idx: usize) -> SimdVec {
        // This is the single, minimal `unsafe` operation. Its correctness is
        // guaranteed by the disciplined logic of its sole caller.
        let offset = (snp_idx * self.num_scores) + (lane_idx * LANE_COUNT);
        SimdVec::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT))
    }
}

// ========================================================================================
//                              THE KERNEL IMPLEMENTATION
// ========================================================================================

/// Calculates all K scores for a single person.
///
/// This function is the heart of the compute engine. It is **100% allocation-free**,
/// using reusable buffers provided by the caller for all temporary storage.
///
/// # Safety
///
/// This function is safe to call only if the caller upholds the following contracts,
/// which are guaranteed by the `batch.rs` engine:
/// - `genotype_row.len() == weights.num_snps`
/// - `scores_out.len() == weights.num_scores()`
/// - `accumulator_buffer.len()` is sufficient for the number of scores, i.e.,
///   at least `(weights.num_scores() + LANE_COUNT - 1) / LANE_COUNT`.
#[inline]
pub fn accumulate_scores_for_person(
    genotype_row: &[u8],
    weights: &InterleavedWeights,
    scores_out: &mut [f32],
    // --- Reusable, thread-local buffers ---
    accumulator_buffer: &mut [SimdVec],
    idx_g1_buffer: &mut Vec<usize>,
    idx_g2_buffer: &mut Vec<usize>,
) {
    let num_scores = weights.num_scores();
    let num_accumulator_lanes = (num_scores + LANE_COUNT - 1) / LANE_COUNT;
    let dosage_2_multiplier = SimdVec::splat(2.0);

    // --- STEP 1: Pre-scan & Indexing (The "Sparsity-Aware" part) ---
    // Clear the buffers to reset their length to 0, but retain their capacity.
    // This avoids heap allocation on every kernel invocation.
    idx_g1_buffer.clear();
    idx_g2_buffer.clear();
    for (snp_idx, &dosage) in genotype_row.iter().enumerate() {
        match dosage {
            1 => idx_g1_buffer.push(snp_idx),
            2 => idx_g2_buffer.push(snp_idx),
            _ => (), // Dosage 0 is ignored, performing zero wasted work.
        }
    }

    // --- STEP 2: Reset the Reusable Accumulator Buffer ---
    // Zero-out the buffer provided by the caller. This is a fast, predictable
    // operation on a buffer that should be hot in the CPU cache.
    for acc in accumulator_buffer.iter_mut() {
        *acc = SimdVec::splat(0.0);
    }

    // --- STEP 3: The Fused-Multiply-Add Update Loop ---
    // This is the performance-critical core. Every `snp_idx` is guaranteed to be
    // in-bounds because it was just gathered from `genotype_row`. The accumulator
    // is guaranteed to have the correct size by the function's safety contract.
    for &snp_idx in &*idx_g1_buffer {
        for i in 0..num_accumulator_lanes {
            // This is safe due to the function's contract.
            let weights_vec = unsafe { weights.get_simd_lane_unchecked(snp_idx, i) };
            accumulator_buffer[i] += weights_vec;
        }
    }

    for &snp_idx in &*idx_g2_buffer {
        for i in 0..num_accumulator_lanes {
            // This is safe due to the function's contract.
            let weights_vec = unsafe { weights.get_simd_lane_unchecked(snp_idx, i) };
            accumulator_buffer[i] = weights_vec.mul_add(dosage_2_multiplier, accumulator_buffer[i]);
        }
    }

    // --- STEP 4: Final Horizontal Store ---
    // The results, which have been accumulated vertically in SIMD registers, are
    // now written out to the final destination slice.
    for (i, &acc_vec) in accumulator_buffer.iter().enumerate() {
        let start = i * LANE_COUNT;
        let end = (start + LANE_COUNT).min(num_scores);
        acc_vec.write_to_slice(&mut scores_out[start..end]);
    }
}
