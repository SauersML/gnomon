// ========================================================================================
//
//               A TILED, CACHE-AWARE COMPUTE ENGINE
//
// ========================================================================================
//
// ### High-Level Purpose ###
//
// This module contains the synchronous, CPU-bound core of the Staged Block-Pivoting
// Engine. It is called from an asynchronous orchestrator (in `main.rs`) and is
// responsible for the most performance-critical phase of the calculation.

// TODO: The SIMD pivot functions compute the dosages in a wide SIMD vector but then deconstruct that vector to write the results back one lane at a time. This is a "SIMD sandwich" with a slow, scalar filling, and it cripples the performance of the entire operation.

// TODO: The process_tile function introduces a rayon parallel loop (par_chunks_exact) inside a function that is already being called by a top-level rayon parallel loop. This is a classic and severe performance anti-pattern.

// TODO: The plan mandated the use of zero-cost newtypes to prevent index confusion. This requirement was ignored. The code still uses raw u32 and usize for all indices.

use crate::kernel;
use crate::prepare::{PersonSubset, PreparationResult};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::cell::RefCell;
use std::ops::Range;
use std::simd::{Simd, SimdPartialEq, ToBitMask};
use thread_local::ThreadLocal;

// Use a concrete SIMD width for implementation. The portable API allows this to be
// compiled to AVX2, AVX-512, or NEON based on the `target-cpu=native` flag.
// A width of 8 for `u32` corresponds to a 256-bit vector (e.g., AVX2).
const LANES: usize = 8;
type U32xN = Simd<u32, LANES>;
type UsizexN = Simd<usize, LANES>;
type U8xN = Simd<u8, LANES>;

// A tile of 4096 people x 8192 SNPs = 33.5 MB, fitting comfortably in a 45 MB L3 cache.
const PERSON_BLOCK_SIZE: usize = 4096;

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
//                              PUBLIC API & TYPE DEFINITIONS
// ========================================================================================

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectAlleleDosage(u8);

impl EffectAlleleDosage {
    #[inline(always)]
    pub fn new(value: u8) -> Self {
        assert!(value <= 2, "Invalid dosage value created: {}", value);
        Self(value)
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

pub type KernelDataPool =
    ThreadLocal<RefCell<(Vec<kernel::SimdVec>, Vec<usize>, Vec<usize>)>>;

// ========================================================================================
//                           THE SBPE COMPUTE ENGINE IMPLEMENTATION
// ========================================================================================

pub fn run_chunk_computation(
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    all_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) {
    match &prep_result.person_subset {
        PersonSubset::All => run_computation_for_all(
            snp_major_data,
            weights_for_chunk,
            prep_result,
            all_scores_out,
            kernel_data_pool,
            tile_pool,
        ),
        PersonSubset::Indices(indices) => run_computation_for_indices(
            indices,
            snp_major_data,
            weights_for_chunk,
            prep_result,
            all_scores_out,
            kernel_data_pool,
            tile_pool,
        ),
    }
}

// ========================================================================================
//                                 PRIVATE IMPLEMENTATIONS
// ========================================================================================

fn run_computation_for_indices(
    indices: &[OriginalPersonIndex],
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    all_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) {
    indices
        .par_chunks(PERSON_BLOCK_SIZE)
        .enumerate()
        .for_each(|(block_idx, person_indices_in_block)| {
            // Pass the typed slice directly and wrap the primitive block_idx
            // in its explicit newtype.
            process_block(
                person_indices_in_block,
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

fn run_computation_for_all(
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    all_scores_out: &mut [f32],
    kernel_data_pool: &KernelDataPool,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
) {
    let total_people = prep_result.total_people_in_fam;

    // Create a parallel iterator over the raw indices.
    (0..total_people as u32)
        .into_par_iter()
        // Map each raw u32 to its type-safe wrapper.
        .map(OriginalPersonIndex)
        .chunks(PERSON_BLOCK_SIZE)
        .enumerate()
        .for_each(|(block_idx, person_indices_chunk)| {
            // The chunk is now Vec<OriginalPersonIndex>.
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

    let mut tile = tile_pool
        .pop()
        .unwrap_or_else(|| Vec::with_capacity(tile_size));
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

            // WRONG???
            kernel::accumulate_scores_for_person(
                genotype_row,
                weights_for_chunk,
                scores_out_slice,
                acc_buffer,
                idx_g1_buffer,
                idx_g2_buffer,
            );
        });

    let _ = tile_pool.push(tile);
}

/// The portable SIMD, cache-friendly pivot for a slice of indices.
#[inline]
fn pivot_tile_for_slice(
    snp_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
) {
    let snps_in_chunk = prep_result.num_reconciled_snps;
    let bytes_per_snp = (prep_result.total_people_in_fam + 3) / 4;

    for (dense_person_idx, &original_person_idx) in person_indices_in_block.iter().enumerate() {
        let dest_row_offset = dense_person_idx * snps_in_chunk;

        // Use the type-safe index's value for calculations.
        let source_byte_base = original_person_idx.0 / 4;
        let bit_shift_base = (original_person_idx.0 % 4) * 2;

        for snp_chunk_start in (0..snps_in_chunk).step_by(LANES) {
            let remaining_snps = snps_in_chunk - snp_chunk_start;
            let current_lanes = remaining_snps.min(LANES);

            let snp_indices =
                U32xN::from_array(core::array::from_fn(|i| (snp_chunk_start + i) as u32));

            let snp_offsets = snp_indices * U32xN::splat(bytes_per_snp as u32);
            let source_byte_indices = (snp_offsets + U32xN::splat(source_byte_base)).cast::<usize>();

            let packed_vals = U8xN::gather_unmasked(snp_major_data, source_byte_indices);

            let two_bit_genotypes = (packed_vals >> U8xN::splat(bit_shift_base)) & U8xN::splat(0b11);

            let bit0 = two_bit_genotypes & U8xN::splat(1);
            let bit1 = (two_bit_genotypes >> 1) & U8xN::splat(1);
            let dosages = bit1 * (U8xN::splat(1) + bit0);

            let dest_start = dest_row_offset + snp_chunk_start;
            let dest_end = dest_start + current_lanes;
            dosages.write_to_slice_unaligned_unchecked(unsafe {
                // This is safe because EffectAlleleDosage is #[repr(transparent)]
                // over a u8, so a slice of one can be reinterpreted as a slice
                // of the other.
                std::slice::from_raw_parts_mut(
                    tile[dest_start..dest_end].as_mut_ptr() as *mut u8,
                    current_lanes,
                )
            });
        }
    }
}
