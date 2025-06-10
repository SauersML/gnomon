// ========================================================================================
//
//               A TILED, CACHE-AWARE, CONTENTION-FREE COMPUTE ENGINE
//
// ========================================================================================
//
// ### Purpose ###
//
// This module contains the synchronous, CPU-bound core of the Staged Block-Pivoting
// Engine. It is designed to be called from a higher-level asynchronous
// orchestrator within a `spawn_blocking` context.

use crate::kernel;
use crate::types::{
    BlockIndex, EffectAlleleDosage, KernelDataPool, OriginalPersonIndex, PersonSubset,
    PreparationResult, Reconciliation, SparseIndexPool,
};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::cell::RefCell;
use std::error::Error;
use std::simd::{simd_swizzle, Simd, SimdPartialEq, ToBitMask};

// --- SIMD & Engine Tuning Parameters ---
const SIMD_LANES: usize = 8;
type U64xN = Simd<u64, SIMD_LANES>;
type U8xN = Simd<u8, SIMD_LANES>;

/// The number of individuals to process in a single on-the-fly pivoted tile.
/// This value is tuned to ensure the tile fits comfortably within the L3 cache.
const PERSON_BLOCK_SIZE: usize = 4096;

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// Processes one chunk of SNP-major data, mutating a provided slice with partial scores.
///
/// This is the sole public entry point into the synchronous compute engine. It adheres
/// to a "mutate-in-place" contract, where the caller provides a buffer that this
/// function's parallel tasks will fill. This avoids allocations in the compute
/// hot path and is a key part of the overall application's memory management strategy.
pub fn run_chunk_computation(
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- Entry Point Validation ---
    // This is the "airlock" for the compute engine. We verify that the mutable
    // slice provided by the caller has the exact size we expect. This prevents
    // panics or buffer overruns deep inside the parallel loops.
    let expected_len = prep_result.num_people_to_score * prep_result.score_names.len();
    if partial_scores_out.len() != expected_len {
        return Err(format!(
            "Mismatched scores buffer: expected length {}, got {}",
            expected_len,
            partial_scores_out.len()
        )
        .into());
    }

    // --- Dispatch to Generic Parallel Iterator ---
    match &prep_result.person_subset {
        PersonSubset::All => {
            let iter = (0..prep_result.total_people_in_fam as u32)
                .into_par_iter()
                .map(OriginalPersonIndex);
            process_people_iterator(
                iter,
                snp_major_data,
                weights_for_chunk,
                prep_result,
                partial_scores_out,
                kernel_data_pool,
                tile_pool,
                sparse_index_pool,
            );
        }
        PersonSubset::Indices(indices) => {
            let iter = indices.par_iter().copied().map(OriginalPersonIndex);
            process_people_iterator(
                iter,
                snp_major_data,
                weights_for_chunk,
                prep_result,
                partial_scores_out,
                kernel_data_pool,
                tile_pool,
                sparse_index_pool,
            );
        }
    };

    Ok(())
}

// ========================================================================================
//                            PRIVATE IMPLEMENTATION
// ========================================================================================

/// A generic, zero-cost function containing the main parallel processing logic.
fn process_people_iterator<'a, I>(
    iter: I,
    snp_major_data: &'a [u8],
    weights_for_chunk: &'a [f32],
    prep_result: &'a PreparationResult,
    partial_scores: &'a mut [f32],
    kernel_data_pool: &'a KernelDataPool,
    tile_pool: &'a ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &'a SparseIndexPool,
) where
    I: ParallelIterator<Item = OriginalPersonIndex> + Send,
{
    let num_scores = prep_result.score_names.len();

    // The main parallel loop over blocks of people.
    iter.chunks(PERSON_BLOCK_SIZE)
        .enumerate()
        .for_each(move |(block_idx_val, person_indices_chunk)| {
            let block_idx = BlockIndex(block_idx_val);
            let score_start_idx = block_idx.0 * PERSON_BLOCK_SIZE * num_scores;
            let score_end_idx = score_start_idx + person_indices_chunk.len() * num_scores;

            // Each parallel task gets its own mutable slice of the partial scores buffer.
            // This is safe because the chunks are disjoint.
            let block_scores_out = &mut partial_scores[score_start_idx..score_end_idx];

            process_block(
                &person_indices_chunk,
                prep_result,
                snp_major_data,
                weights_for_chunk,
                block_scores_out,
                kernel_data_pool,
                tile_pool,
                sparse_index_pool,
            );
        });
}

/// Processes a single block of individuals. This involves acquiring a buffer from
/// the pool, pivoting the data into it, and dispatching it to the kernel.
#[inline]
fn process_block(
    person_indices_in_block: &[OriginalPersonIndex],
    prep_result: &PreparationResult,
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    block_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let tile_size = person_indices_in_block.len() * snps_in_chunk;

    // Acquire a tile buffer from the pool.
    let mut tile = tile_pool.pop().unwrap_or_default();
    tile.clear();
    tile.resize(tile_size, EffectAlleleDosage::default());

    // Use the new, more efficient, and scientifically correct pivot function.
    pivot_and_reconcile_tile(
        snp_major_data,
        person_indices_in_block,
        &mut tile,
        prep_result,
    );

    process_tile(
        &tile,
        prep_result,
        weights_for_chunk,
        block_scores_out,
        kernel_data_pool,
        sparse_index_pool,
    );

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel.
/// This function now performs the critical pre-computation of sparse indices.
#[inline]
fn process_tile(
    tile: &[EffectAlleleDosage],
    prep_result: &PreparationResult,
    weights_for_chunk: &[f32],
    block_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    sparse_index_pool: &SparseIndexPool,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let num_scores = prep_result.score_names.len();
    let num_people_in_block = tile.len() / snps_in_chunk;

    // --- The System-Wide Win: Pre-computation of Sparse Indices ---
    let thread_indices = sparse_index_pool.get_or_default();
    let (g1_indices, g2_indices) = &mut *thread_indices.borrow_mut();

    // Correct buffer management: clear inner vectors to reuse their capacity.
    // Then, only allocate new inner vectors if the current block is larger than any seen before.
    g1_indices.iter_mut().for_each(Vec::clear);
    g2_indices.iter_mut().for_each(Vec::clear);
    if g1_indices.len() < num_people_in_block {
        g1_indices.resize_with(num_people_in_block, Vec::new);
    }
    if g2_indices.len() < num_people_in_block {
        g2_indices.resize_with(num_people_in_block, Vec::new);
    }

    // This scan is now on cache-hot, contiguous person-major data.
    for person_idx in 0..num_people_in_block {
        let genotype_row_start = person_idx * snps_in_chunk;
        let genotype_row_end = genotype_row_start + snps_in_chunk;
        let genotype_row = &tile[genotype_row_start..genotype_row_end];

        for (snp_idx, &dosage) in genotype_row.iter().enumerate() {
            // The dosage is repr(transparent) over u8.
            match dosage.0 {
                1 => g1_indices[person_idx].push(snp_idx),
                2 => g2_indices[person_idx].push(snp_idx),
                _ => (), // Dosage 0 or 3 (missing) are ignored.
            }
        }
    }
    // --- End of Pre-computation ---

    let weights = kernel::InterleavedWeights::new(weights_for_chunk, snps_in_chunk, num_scores)
        .expect("CRITICAL: Weights matrix validation failed. This is an unrecoverable internal error.");

    // This is now a sequential iterator over a block owned by a single rayon thread.
    block_scores_out
        .chunks_exact_mut(num_scores)
        .enumerate()
        .for_each(|(person_idx, scores_out_slice)| {
            let thread_data = kernel_data_pool.get_or_default();
            let acc_buffer = &mut *thread_data.borrow_mut();

            kernel::accumulate_scores_for_person(
                &weights,
                scores_out_slice,
                acc_buffer,
                &g1_indices[person_idx],
                &g2_indices[person_idx],
            );
        });
}

/// A cache-friendly, SIMD-accelerated pivot function using an 8x8 in-register transpose.
#[inline]
fn pivot_and_reconcile_tile(
    snp_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let bytes_per_snp = (prep_result.total_people_in_fam as u64 + 3) / 4;
    let num_people_in_block = person_indices_in_block.len();

    let two_splat = U8xN::splat(2);
    let missing_sentinel_splat = U8xN::splat(3);

    for person_chunk_start in (0..num_people_in_block).step_by(SIMD_LANES) {
        let remaining_people = num_people_in_block - person_chunk_start;
        let current_lanes = remaining_people.min(SIMD_LANES);

        let person_indices = U64xN::from_array(core::array::from_fn(|i| {
            if i < current_lanes {
                person_indices_in_block[person_chunk_start + i].0 as u64
            } else {
                0
            }
        }));
        let person_byte_indices = person_indices / 4;
        let bit_shifts = (person_indices % 4) * 2;

        for snp_chunk_start in (0..snps_in_chunk).step_by(SIMD_LANES) {
            let remaining_snps = snps_in_chunk - snp_chunk_start;
            let current_snps = remaining_snps.min(SIMD_LANES);

            let mut snp_data_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..current_snps {
                let snp_idx = snp_chunk_start + i;
                let snp_byte_offset =
                    prep_result.required_snp_indices[snp_idx] as u64 * bytes_per_snp;
                let source_byte_indices = snp_byte_offset + person_byte_indices;
                snp_data_vectors[i] =
                    U8xN::gather_or_default(snp_major_data, source_byte_indices.cast());
            }

            let mut dosage_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..current_snps {
                let snp_idx = snp_chunk_start + i;
                let packed_vals = snp_data_vectors[i];
                let reconciliation = prep_result.reconciliation_instructions[snp_idx];

                let two_bit_genotypes = (packed_vals >> bit_shifts.cast()) & U8xN::splat(0b11);
                let initial_dosages =
                    ((two_bit_genotypes >> 1) & 1) * ((two_bit_genotypes & 1) + 1);
                let missing_mask = two_bit_genotypes.simd_eq(U8xN::splat(1));
                let mut dosages =
                    U8xN::mask_select(missing_mask, missing_sentinel_splat, initial_dosages);
                if reconciliation == Reconciliation::Flip {
                    let not_missing_mask = dosages.simd_ne(missing_sentinel_splat);
                    dosages = U8xN::mask_select(not_missing_mask, two_splat - dosages, dosages);
                }
                dosage_vectors[i] = dosages;
            }

            let person_data_vectors = transpose_8x8_u8(dosage_vectors);

            let tile_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(tile.as_mut_ptr() as *mut u8, tile.len())
            };

            for i in 0..current_lanes {
                let person_idx_in_block = person_chunk_start + i;
                let dest_offset = person_idx_in_block * snps_in_chunk + snp_chunk_start;

                // This is the optimal store. It performs a true contiguous memory copy,
                // which the compiler will optimize into the fastest possible instruction(s)
                // for the target architecture, for any `current_snps` from 1 to 8.
                let dest_slice = &mut tile_u8[dest_offset..dest_offset + current_snps];
                let src_slice = &person_data_vectors[i].to_array()[..current_snps];
                dest_slice.copy_from_slice(src_slice);
            }
        }
    }
}

/// Helper function to perform an 8x8 byte matrix transpose using portable `std::simd`.
/// This is a standard DIT (Decimation in Time) algorithm, also known as a
/// butterfly network. It is architecture-agnostic and will be compiled to
/// efficient machine code (e.g., sequences of PUNPCK* on AVX2, or NEON equivalents)
/// by the Rust compiler. It works by progressively interleaving elements
/// at different sizes (8-bit, then 16-bit, then 32-bit).
#[inline(always)]
fn transpose_8x8_u8(matrix: [U8xN; 8]) -> [U8xN; 8] {
    let [m0, m1, m2, m3, m4, m5, m6, m7] = matrix;

    // Interleave 8-bit elements
    let (t0, t1) = simd_swizzle!(m0, m1, [0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15]);
    let (t2, t3) = simd_swizzle!(m2, m3, [0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15]);
    let (t4, t5) = simd_swizzle!(m4, m5, [0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15]);
    let (t6, t7) = simd_swizzle!(m6, m7, [0, 8, 2, 10, 4, 12, 6, 14], [1, 9, 3, 11, 5, 13, 7, 15]);

    // Interleave 16-bit elements
    let (s0, s1) = simd_swizzle!(t0.cast::<u16>(), t2.cast::<u16>(), [0, 4, 1, 5], [2, 6, 3, 7]);
    let (s2, s3) = simd_swizzle!(t1.cast::<u16>(), t3.cast::<u16>(), [0, 4, 1, 5], [2, 6, 3, 7]);
    let (s4, s5) = simd_swizzle!(t4.cast::<u16>(), t6.cast::<u16>(), [0, 4, 1, 5], [2, 6, 3, 7]);
    let (s6, s7) = simd_swizzle!(t5.cast::<u16>(), t7.cast::<u16>(), [0, 4, 1, 5], [2, 6, 3, 7]);

    // Interleave 32-bit elements
    let (r0, r1) = simd_swizzle!(s0.cast::<u32>(), s4.cast::<u32>(), [0, 2], [1, 3]);
    let (r2, r3) = simd_swizzle!(s1.cast::<u32>(), s5.cast::<u32>(), [0, 2], [1, 3]);
    let (r4, r5) = simd_swizzle!(s2.cast::<u32>(), s6.cast::<u32>(), [0, 2], [1, 3]);
    let (r6, r7) = simd_swizzle!(s3.cast::<u32>(), s7.cast::<u32>(), [0, 2], [1, 3]);

    [
        r0.cast(),
        r1.cast(),
        r2.cast(),
        r3.cast(),
        r4.cast(),
        r5.cast(),
        r6.cast(),
        r7.cast(),
    ]
}
