// ========================================================================================
//
//               A TILED, CACHE-AWARE COMPUTE ENGINE
//
// ========================================================================================
//
// ### High-Level Purpose ###
//
// This module contains the synchronous, CPU-bound core of the Staged Block-Pivoting
// Engine. It is designed to be called from a higher-level asynchronous
// orchestrator.

use crate::kernel;
use crate::prepare::{PersonSubset, PreparationResult};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::cell::RefCell;
use std::simd::Simd;
use thread_local::ThreadLocal;

// --- SIMD Type Aliases ---
const LANES: usize = 8;
type U64xN = Simd<u64, LANES>;
type U8xN = Simd<u8, LANES>;

// --- Engine Tuning Parameters ---
const PERSON_BLOCK_SIZE: usize = 4096;

// ========================================================================================
//                            TYPE-SAFE INDEX DEFINITIONS
// ========================================================================================

/// An index into the original, full .fam file (e.g., one of 150,000).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct OriginalPersonIndex(pub u32);

/// A dense, 0-based index within a single processed tile (e.g., 0..4095).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DensePersonIndex(pub usize);

/// A 0-based index for a block of people being processed in parallel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockIndex(pub usize);

// ========================================================================================
//                            PUBLIC API & DATA STRUCTURES
// ========================================================================================

/// A `#[repr(transparent)]` wrapper for a dosage value, guaranteeing it is <= 2.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectAlleleDosage(pub u8);

/// A pool of reusable, thread-local buffers for the compute kernel.
// Perhaps move to kernel.rs later.
pub type KernelDataPool =
    ThreadLocal<RefCell<(Vec<kernel::SimdVec>, Vec<usize>, Vec<usize>)>>;

// ========================================================================================
//                       THE UNIFIED SBPE COMPUTE ENGINE
// ========================================================================================

/// The single public entry point for processing a chunk of SNP-major data.
/// This function is a simple dispatcher that selects the correct iterator type
/// and passes it to the generic, zero-cost processing function.
pub fn run_chunk_computation(
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    all_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) {
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
                all_scores_out,
                kernel_data_pool,
                tile_pool,
            );
        }
        PersonSubset::Indices(indices) => {
            let iter = indices.par_iter().copied();
            process_people_iterator(
                iter,
                snp_major_data,
                weights_for_chunk,
                prep_result,
                all_scores_out,
                kernel_data_pool,
                tile_pool,
            );
        }
    };
}

// ========================================================================================
//                            PRIVATE IMPLEMENTATION
// ========================================================================================

/// A generic, zero-cost function that contains the main parallel processing logic.
/// It is generic over the `ParallelIterator` type, which allows the compiler to
/// create specialized, highly optimized versions for each call site, eliminating
//  the need for heap allocation and dynamic dispatch.
fn process_people_iterator<I>(
    iter: I,
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    all_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) where
    I: ParallelIterator<Item = OriginalPersonIndex> + Send,
{
    // This is the single, coarse-grained parallel loop for the entire engine.
    iter.chunks(PERSON_BLOCK_SIZE)
        .enumerate()
        .for_each(|(block_idx, person_indices_chunk)| {
            process_block(
                &person_indices_chunk,
                BlockIndex(block_idx),
                prep_result,
                snp_major_data,
                weights_for_chunk,
                all_scores_out,
                kernel_data_pool,
                tile_pool,
            );
        });
}

/// Processes a single block of individuals. This involves acquiring a buffer,
/// pivoting the data into it, and dispatching it to the kernel.
#[inline]
fn process_block(
    person_indices_in_block: &[OriginalPersonIndex],
    block_idx: BlockIndex,
    prep_result: &PreparationResult,
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    all_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let tile_size = person_indices_in_block.len() * snps_in_chunk;

    // Robustly acquire a tile buffer from the pool.
    let mut tile = tile_pool.pop().unwrap_or_else(|| {
        eprintln!("Warning: Tile pool was empty, allocating a new tile. This may indicate a performance bottleneck or an insufficient buffer count.");
        Vec::with_capacity(tile_size)
    });
    tile.clear();
    tile.resize(tile_size, EffectAlleleDosage(0));

    pivot_tile_for_slice(
        snp_major_data,
        person_indices_in_block,
        &mut tile,
        prep_result,
    );

    process_tile(
        tile,
        block_idx,
        prep_result,
        weights_for_chunk,
        all_scores_out,
        kernel_data_pool,
        tile_pool,
    );
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel.
/// It iterates sequentially over the people in the tile.
#[inline]
fn process_tile(
    tile: Vec<EffectAlleleDosage>,
    block_idx: BlockIndex,
    prep_result: &PreparationResult,
    weights_for_chunk: &[f32],
    all_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let num_scores = prep_result.score_names.len();

    // This is a sequential iterator.
    tile.chunks_exact(snps_in_chunk)
        .enumerate()
        .for_each(|(dense_person_idx_in_tile, genotype_row)| {
            let thread_data = kernel_data_pool.get_or_default();
            let mut data = thread_data.borrow_mut();
            let (acc_buffer, idx_g1_buffer, idx_g2_buffer) = &mut *data;

            let dense_idx = DensePersonIndex(dense_person_idx_in_tile);
            let overall_person_idx = (block_idx.0 * PERSON_BLOCK_SIZE) + dense_idx.0;
            let score_start_idx = overall_person_idx * num_scores;
            let scores_out_slice = &mut all_scores_out[score_start_idx..score_start_idx + num_scores];

            // Correctly adhere to the kernel's API contract.
            let kernel_input = kernel::KernelInput::new(
                // This is the idiomatic and explicit way to perform a zero-cost
                // slice conversion for a `#[repr(transparent)]` newtype.
                unsafe {
                    std::slice::from_raw_parts(
                        genotype_row.as_ptr() as *const u8,
                        genotype_row.len(),
                    )
                },
                weights_for_chunk,
                scores_out_slice,
                acc_buffer,
                idx_g1_buffer,
                idx_g2_buffer,
                snps_in_chunk,
                num_scores,
            );

            match kernel_input {
                Ok(input) => kernel::accumulate_scores_for_person(input),
                Err(validation_err) => {
                    // This branch should be unreachable if upstream logic is correct.
                    panic!("CRITICAL: KernelInput validation failed: {:?}. This is an unrecoverable internal error.", validation_err);
                }
            }
        });

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// The portable SIMD, cache-friendly pivot for a slice of indices.
/// It transposes a slice of SNP-major data into a person-major tile.
#[inline]
fn pivot_tile_for_slice(
    snp_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let bytes_per_snp = (prep_result.total_people_in_fam as u64 + 3) / 4;

    for (dense_person_idx, &original_person_idx) in person_indices_in_block.iter().enumerate() {
        let dest_row_offset = dense_person_idx * snps_in_chunk;

        let original_idx_u64 = original_person_idx.0 as u64;
        let source_byte_base = original_idx_u64 / 4;
        let bit_shift_base = (original_idx_u64 % 4) * 2;

        for snp_chunk_start in (0..snps_in_chunk).step_by(LANES) {
            let remaining_snps = snps_in_chunk - snp_chunk_start;
            let current_lanes = remaining_snps.min(LANES);

            let snp_indices =
                U64xN::from_array(core::array::from_fn(|i| (snp_chunk_start + i) as u64));

            let snp_offsets = snp_indices * U64xN::splat(bytes_per_snp);
            let source_byte_indices = (snp_offsets + U64xN::splat(source_byte_base)).cast::<usize>();

            let packed_vals = U8xN::gather_unmasked(snp_major_data, source_byte_indices);
            let two_bit_genotypes = (packed_vals >> U8xN::splat(bit_shift_base as u8)) & U8xN::splat(0b11);

            let bit0 = two_bit_genotypes & U8xN::splat(1);
            let bit1 = (two_bit_genotypes >> 1) & U8xN::splat(1);
            let dosages = bit1 * (U8xN::splat(1) + bit0);

            let dest_start = dest_row_offset + snp_chunk_start;
            let dest_end = dest_start + current_lanes;
            dosages.write_to_slice_unaligned_unchecked(unsafe {
                std::slice::from_raw_parts_mut(
                    tile[dest_start..dest_end].as_mut_ptr() as *mut u8,
                    current_lanes,
                )
            });
        }
    }
}
