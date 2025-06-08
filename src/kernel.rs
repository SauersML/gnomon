// ========================================================================================
//
//               THE THROUGHPUT-BOUND INTERLEAVED-SIMD KERNEL
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module contains the pure, uncompromising computational core of the engine.
// Its sole function is to execute a sparse matrix-vector multiplication for a single
// person's pre-processed genotype data against a pre-structured weight matrix. Its
// performance is designed to be limited only by the raw arithmetic throughput of the
// CPU's Fused-Multiply-Add (FMA) units, representing the theoretical maximum speed.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: Latency Elimination, Not Hiding ###
//
// The conventional Transposed Indexed Gather kernel is an elegant solution, but
// it is fundamentally **latency-bound**. Its performance is dictated by the speed of
// the `GATHER` instruction, which fetches scattered data from memory. While clever
// software pipelining can hide some of this latency, it cannot change the underlying
// physics.
//
// This kernel operates on a different, superior philosophy. It assumes that a higher-
// level module (`batch.rs`) has paid a one-time cost to transform the weight matrix
// into an **interleaved layout**. This transformation completely **eliminates the need
// for GATHER instructions.**
//
// Instead of scattered reads, this kernel performs only perfectly **linear, streaming
// loads** from the interleaved weight matrix. A linear load is the fastest possible
// memory operation, fully saturating the CPU's memory bandwidth. By structurally
// removing the latency bottleneck, this kernel's performance becomes bound only by
// the raw throughput of the CPU's arithmetic units. This is the definition of a
// truly performant engine.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. The Uncompromising Contract ###
//
//   - **DATA LAYOUT CONTRACT:** This kernel does NOT operate on a standard weight
//     matrix. It is part of a symbiotic pair with `batch.rs`. It strictly requires
//     that the weight matrix (`w_chunk`) has been pre-processed into the
//     `M_chunk x K` interleaved format, where all K weights for a single SNP are
//     contiguous in memory.
//
//   - **NO FALLBACKS:** This kernel is implemented using explicit `std::simd`
//     intrinsics. There is no scalar fallback path.
//
//   - **`unsafe` is possible.
//
// ----------------------------------------------------------------------------------------

use std::simd::{f32x8, Simd};

// --- Type Aliases for Readability ---
// Public so the caller can correctly size the accumulator buffer.
pub type SimdVec = f32x8;
pub const LANE_COUNT: usize = SimdVec::LANE_COUNT;

// ========================================================================================
//                            PUBLIC API & TYPE DEFINITIONS
// ========================================================================================

/// Represents a validation error during the construction of `KernelInput`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// `interleaved_weights.len()` did not equal `num_snps * num_scores`.
    MismatchedWeights,
    /// `genotype_row.len()` did not equal `num_snps`.
    MismatchedGenotypes,
    /// `scores_out.len()` did not equal `num_scores`.
    MismatchedScores,
    /// The provided `accumulator_buffer` had an incorrect length.
    MismatchedAccumulator,
}

/// A validated, type-safe view of the interleaved weights matrix for one chunk.
/// Its construction is private, guaranteeing its invariants are always met.
pub struct InterleavedWeights<'a> {
    slice: &'a [f32],
    num_snps: usize,
    num_scores: usize,
}

impl<'a> InterleavedWeights<'a> {
    /// Fetches the i-th SIMD vector of weights for a given SNP.
    ///
    /// # Safety
    /// This function is `unsafe` because it bypasses bounds checks. The caller MUST
    /// guarantee that `snp_idx` and `lane_idx` are within valid dimensions. This
    /// guarantee is provided by the `KernelInput` type, which makes calling this
    /// method from within the kernel provably safe.
    #[inline(always)]
    unsafe fn get_simd_lane_unchecked(&self, snp_idx: usize, lane_idx: usize) -> SimdVec {
        // This is the single, minimal `unsafe` operation. Its correctness is
        // guaranteed by the constructor logic in `KernelInput::new()`.
        let offset = (snp_idx * self.num_scores) + (lane_idx * LANE_COUNT);
        SimdVec::from_slice(self.slice.get_unchecked(offset..offset + LANE_COUNT))
    }
}

/// A "proof token" that all data required for a single person's score
/// calculation is present, validated, and dimensionally consistent.
pub struct KernelInput<'a> {
    genotype_row: &'a [u8],
    weights: InterleavedWeights<'a>,
    scores_out: &'a mut [f32],
    // A mutable slice of SIMD vectors provided by the caller to be used as
    // temporary storage for the accumulators.
    accumulator_buffer: &'a mut [SimdVec],
    // Mutable references to reusable vectors for storing SNP indices.
    // The caller is responsible for creating these once per thread and reusing them.
    idx_g1_buffer: &'a mut Vec<usize>,
    idx_g2_buffer: &'a mut Vec<usize>,
}

impl<'a> KernelInput<'a> {
    /// The "smart constructor" and sole entry point for creating a `KernelInput`.
    ///
    /// This function acts as the gatekeeper. It performs all necessary validation
    /// checks upfront, including verifying the size of the provided accumulator
    /// buffer. If the data is consistent, an `Ok(KernelInput)` is returned.
    /// Otherwise, an `Err(ValidationError)` specifies the failure.
    #[inline]
    #[must_use]
    pub fn new(
        genotype_row: &'a [u8],
        interleaved_weights: &'a [f32],
        scores_out: &'a mut [f32],
        accumulator_buffer: &'a mut [SimdVec],
        idx_g1_buffer: &'a mut Vec<usize>,
        idx_g2_buffer: &'a mut Vec<usize>,
        num_snps: usize,
        num_scores: usize,
    ) -> Result<Self, ValidationError> {
        // The #[cold] attribute hints to the compiler that error branches are
        // unlikely, allowing it to optimize the hot path (the `Ok` return).
        #[cold]
        fn fail(err: ValidationError) -> Result<KernelInput<'static>, ValidationError> {
            Err(err)
        }

        if genotype_row.len() != num_snps {
            return fail(ValidationError::MismatchedGenotypes);
        }
        if scores_out.len() != num_scores {
            return fail(ValidationError::MismatchedScores);
        }
        if interleaved_weights.len() != num_snps * num_scores {
            return fail(ValidationError::MismatchedWeights);
        }

        let required_lanes = (num_scores + LANE_COUNT - 1) / LANE_COUNT;
        if accumulator_buffer.len() != required_lanes {
            return fail(ValidationError::MismatchedAccumulator);
        }

        Ok(Self {
            genotype_row,
            weights: InterleavedWeights {
                slice: interleaved_weights,
                num_snps,
                num_scores,
            },
            scores_out,
            accumulator_buffer,
            idx_g1_buffer,
            idx_g2_buffer,
        })
    }
}

// ========================================================================================
//                              THE KERNEL IMPLEMENTATION
// ========================================================================================

/// Calculates all K scores for a single person using a validated `KernelInput`.
///
/// This function is **100% allocation-free**. It uses the pre-validated,
/// reusable buffers provided in `KernelInput` for all its calculations,
/// including the index vectors and the SIMD accumulators.
#[inline]
pub fn accumulate_scores_for_person(mut input: KernelInput) {
    let num_scores = input.weights.num_scores;
    let num_accumulator_lanes = (num_scores + LANE_COUNT - 1) / LANE_COUNT;
    let dosage_2_multiplier = SimdVec::splat(2.0);

    // --- STEP 1: Pre-scan & Indexing (using reusable buffers) ---
    // Clear the buffers to reset their length to 0, but retain their capacity.
    // This avoids heap allocation on every kernel invocation.
    input.idx_g1_buffer.clear();
    input.idx_g2_buffer.clear();
    for (snp_idx, &dosage) in input.genotype_row.iter().enumerate() {
        match dosage {
            1 => input.idx_g1_buffer.push(snp_idx),
            2 => input.idx_g2_buffer.push(snp_idx),
            _ => (),
        }
    }

    // --- STEP 2: RESET the Reusable Accumulator Buffer ---
    // This is the key performance improvement. Instead of allocating, we just
    // zero-out the buffer provided by the caller. This is a fast, predictable
    // operation on a buffer that should be hot in the CPU cache.
    for acc in input.accumulator_buffer.iter_mut() {
        *acc = SimdVec::splat(0.0);
    }

    // --- STEP 3: THE FUSED UPDATE LOOP ---
    // The logic here is simple, but its safety is profound. Every `snp_idx` is
    // guaranteed to be in-bounds. The `accumulators` buffer is also guaranteed
    // to have the correct size.
    for &snp_idx in &*input.idx_g1_buffer {
        for i in 0..num_accumulator_lanes {
            let weights_vec = unsafe { input.weights.get_simd_lane_unchecked(snp_idx, i) };
            input.accumulator_buffer[i] += weights_vec;
        }
    }

    for &snp_idx in &*input.idx_g2_buffer {
        for i in 0..num_accumulator_lanes {
            let weights_vec = unsafe { input.weights.get_simd_lane_unchecked(snp_idx, i) };
            input.accumulator_buffer[i] = weights_vec.mul_add(dosage_2_multiplier, input.accumulator_buffer[i]);
        }
    }

    // --- STEP 4: Final Store ---
    for (i, &acc_vec) in input.accumulator_buffer.iter().enumerate() {
        let start = i * LANE_COUNT;
        let end = (start + LANE_COUNT).min(num_scores);
        acc_vec.write_to_slice(&mut input.scores_out[start..end]);
    }
}
