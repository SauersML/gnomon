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
    PreparationResult, Reconciliation,
};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::error::Error;
use std::simd::{Simd, SimdPartialEq, ToBitMask};

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

/// Processes one chunk of SNP-major data, returning a new `Vec<f32>` of partial scores.
///
/// This is the sole public entry point into the synchronous compute engine. It adheres
/// to the "return-then-merge" pattern to ensure contention-free parallelism.
pub fn process_snp_chunk(
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) -> Result<Vec<f32>, Box<dyn Error + Send + Sync>> {
    // 1. Allocate a local result buffer to prevent false sharing.
    let mut partial_scores =
        vec![0.0f32; prep_result.num_people_to_score * prep_result.score_names.len()];

    // 2. Dispatch to the appropriate generic parallel iterator.
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
                &mut partial_scores,
                kernel_data_pool,
                tile_pool,
            );
        }
        PersonSubset::Indices(indices) => {
            let iter = indices.par_iter().copied().map(OriginalPersonIndex);
            process_people_iterator(
                iter,
                snp_major_data,
                weights_for_chunk,
                prep_result,
                &mut partial_scores,
                kernel_data_pool,
                tile_pool,
            );
        }
    };

    // 3. Return ownership of the completed partial scores.
    Ok(partial_scores)
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
    );

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel.
#[inline]
fn process_tile(
    tile: &[EffectAlleleDosage],
    prep_result: &PreparationResult,
    weights_for_chunk: &[f32],
    block_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let num_scores = prep_result.score_names.len();

    let weights = kernel::InterleavedWeights::new(weights_for_chunk, snps_in_chunk, num_scores)
        .expect("CRITICAL: Weights matrix validation failed. This is an unrecoverable internal error.");

    // This is now a sequential iterator over a block owned by a single rayon thread.
    tile.chunks_exact(snps_in_chunk)
        .zip(block_scores_out.chunks_exact_mut(num_scores))
        .for_each(|(genotype_row, scores_out_slice)| {
            let thread_data = kernel_data_pool.get_or_default();
            let (acc_buffer, idx_g1, idx_g2) = &mut *thread_data.borrow_mut();

            // `EffectAlleleDosage` is `#[repr(transparent)]` over `u8`. This
            // zero-cost cast is safe because the kernel's contract is on the raw
            // byte values, which we have correctly prepared.
            let genotype_row_u8: &[u8] = unsafe {
                std::slice::from_raw_parts(genotype_row.as_ptr() as *const u8, genotype_row.len())
            };

            kernel::accumulate_scores_for_person(
                genotype_row_u8,
                &weights,
                scores_out_slice,
                acc_buffer,
                idx_g1,
                idx_g2,
            );
        });
}

/// A cache-friendly, SIMD-accelerated pivot function.
///
/// It iterates SNP-by-SNP for sequential access on the large source buffer, and
/// performs a SIMD gather/scatter to transpose data into the person-major tile.
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

    // Pre-splat constant vectors for SIMD arithmetic.
    let two_splat = U8xN::splat(2);
    let missing_sentinel_splat = U8xN::splat(3);

    // Iterate SNP-by-SNP for cache-friendly sequential reads from `snp_major_data`.
    for snp_idx in 0..snps_in_chunk {
        let snp_byte_offset = prep_result.required_snp_indices[snp_idx] as u64 * bytes_per_snp;
        let reconciliation = prep_result.reconciliation_instructions[snp_idx];

        // Process a sub-block of people using SIMD.
        for person_chunk_start in (0..num_people_in_block).step_by(SIMD_LANES) {
            let remaining_people = num_people_in_block - person_chunk_start;
            let current_lanes = remaining_people.min(SIMD_LANES);
            let mask = u64::MAX >> (64 - current_lanes);

            // GATHER: Collect the original indices and calculate their positions in the source buffer.
            let person_indices = U64xN::from_array(core::array::from_fn(|i| {
                if i < current_lanes {
                    person_indices_in_block[person_chunk_start + i].0 as u64
                } else {
                    0
                }
            }));

            let source_byte_indices = snp_byte_offset + (person_indices / 4);
            let bit_shifts = (person_indices % 4) * 2;

            let packed_vals = U8xN::gather_or_default(snp_major_data, source_byte_indices.cast());

            // --- Unpacking and Reconciliation Logic ---

            // 1. Unpack the raw 2-bit values.
            let two_bit_genotypes = (packed_vals >> bit_shifts.cast()) & U8xN::splat(0b11);

            // 2. Convert to dosages, initially mapping missing (0b01) to 0.
            let initial_dosages =
                ((two_bit_genotypes >> 1) & 1) * ((two_bit_genotypes & 1) + 1);

            // 3. Find where the original value was missing (0b01) and insert sentinel `3`.
            let missing_mask = two_bit_genotypes.simd_eq(U8xN::splat(1));
            let mut dosages = U8xN::mask_select(missing_mask, missing_sentinel_splat, initial_dosages);

            // 4. Apply flip (2 - dosage) only if needed AND the value is not missing.
            if reconciliation == Reconciliation::Flip {
                let not_missing_mask = dosages.simd_ne(missing_sentinel_splat);
                dosages = U8xN::mask_select(not_missing_mask, two_splat - dosages, dosages);
            }

            // SCATTER: Write the final, reconciled dosages to the correct place in the tile.
            let tile_indices = U64xN::from_array(core::array::from_fn(|i| {
                ((person_chunk_start + i) * snps_in_chunk + snp_idx) as u64
            }));

            // `EffectAlleleDosage` is `#[repr(transparent)]` over `u8`. We have
            // manually ensured that the `dosages` vector contains only valid values
            // (0, 1, 2, or 3 for missing), thus upholding the type's invariants.
            let tile_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(tile.as_mut_ptr() as *mut u8, tile.len())
            };
            tile_u8.scatter_masked(tile_indices.cast(), mask.into(), dosages);
        }
    }
}
