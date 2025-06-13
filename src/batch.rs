// ========================================================================================
//
//               A TILED, CACHE-AWARE, CONTENTION-FREE COMPUTE ENGINE
//
// ========================================================================================
//
// ### Purpose ###
//
// This module contains the synchronous, CPU-bound core of the compute pipeline. It is
// designed to be called from a higher-level asynchronous orchestrator within a
// `spawn_blocking` context. Its sole responsibility is to take a raw, SNP-major
// chunk of genotype data, pivot it into a person-major tile, generate a sparse
// index of non-zero work, and dispatch it to the kernel. It performs ZERO
// scientific logic or reconciliation.

use crate::kernel;
use crate::types::{
    EffectAlleleDosage, OriginalPersonIndex, PersonSubset,
    PreparationResult,
};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::cell::RefCell;
use std::error::Error;
use std::simd::{cmp::SimdPartialEq, num::SimdUint, Simd};
use thread_local::ThreadLocal;
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};


// --- SIMD & Engine Tuning Parameters ---
const SIMD_LANES: usize = 8;
type U64xN = Simd<u64, SIMD_LANES>;
type U8xN = Simd<u8, SIMD_LANES>;

/// The number of individuals to process in a single on-the-fly pivoted tile.
/// This value is tuned to ensure the tile fits comfortably within the L3 cache.
const PERSON_BLOCK_SIZE: usize = 4096;

/// A thread-local pool for reusing the memory buffers required for storing
/// sparse indices (`g1_indices`, `g2_indices`). This is a critical performance
/// optimization that avoids heap allocations in the hot compute path.
#[derive(Default, Debug)]
pub struct SparseIndexPool {
    pool: ThreadLocal<RefCell<(Vec<Vec<usize>>, Vec<Vec<usize>>)>>,
}

impl SparseIndexPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets the thread-local buffer of sparse indices, creating it if it doesn't exist.
    #[inline(always)]
    fn get_or_default(&self) -> &RefCell<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        self.pool.get_or_default()
    }
}

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// Processes one chunk of SNP-major data, mutating a provided slice with partial scores.
/// This is the sole public entry point into the synchronous compute engine.
pub fn run_chunk_computation(
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f32],
    partial_missing_counts_out: &mut [u32],
    partial_correction_sums_out: &mut [f32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
    matrix_row_start_idx: usize,
    chunk_bed_row_offset: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- Entry Point Validation ---
    // This is the "airlock" for the compute engine. We verify the output buffer
    // size once to prevent panics deep inside the parallel loops.
    let expected_len = prep_result.num_people_to_score * prep_result.score_names.len();
    if partial_scores_out.len() != expected_len {
        return Err(Box::from(format!(
            "Mismatched scores buffer: expected length {}, got {}",
            expected_len,
            partial_scores_out.len()
        )));
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
                partial_missing_counts_out,
                partial_correction_sums_out,
                tile_pool,
                sparse_index_pool,
                matrix_row_start_idx,
                chunk_bed_row_offset,
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
                partial_missing_counts_out,
                partial_correction_sums_out,
                tile_pool,
                sparse_index_pool,
                matrix_row_start_idx,
                chunk_bed_row_offset,
            );
        }
    };

    Ok(())
}

// ========================================================================================
//                            PRIVATE IMPLEMENTATION
// ========================================================================================

/// Contains the main parallel processing logic, using an idiomatic Rayon structure
/// for high-performance, in-place mutation.
fn process_people_iterator<'a, I>(
    iter: I,
    snp_major_data: &'a [u8],
    weights_for_chunk: &'a [f32],
    prep_result: &'a PreparationResult,
    partial_scores: &'a mut [f32],
    partial_missing_counts: &'a mut [u32],
    partial_correction_sums: &'a mut [f32],
    tile_pool: &'a ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &'a SparseIndexPool,
    matrix_row_start_idx: usize,
    chunk_bed_row_offset: usize,
) where
    I: IndexedParallelIterator<Item = OriginalPersonIndex> + Send,
{
    let num_scores = prep_result.score_names.len();
    let all_person_indices: Vec<_> = iter.collect();
    let items_per_block = PERSON_BLOCK_SIZE * num_scores;

    // The main parallel loop iterates over the OUTPUT buffers in disjoint chunks.
    partial_scores
        .par_chunks_mut(items_per_block)
        .zip(partial_missing_counts.par_chunks_mut(items_per_block))
        .zip(partial_correction_sums.par_chunks_mut(items_per_block))
        .enumerate()
        .for_each(|(block_idx, ((block_scores_out, block_missing_counts_out), block_correction_sums_out))| {
            let person_start_idx = block_idx * PERSON_BLOCK_SIZE;
            let person_end_idx =
                (person_start_idx + PERSON_BLOCK_SIZE).min(all_person_indices.len());
            let person_indices_in_block = &all_person_indices[person_start_idx..person_end_idx];

            if person_indices_in_block.is_empty() {
                return;
            }

            process_block(
                person_indices_in_block,
                prep_result,
                snp_major_data,
                weights_for_chunk,
                block_scores_out,
                block_missing_counts_out,
                block_correction_sums_out,
                tile_pool,
                sparse_index_pool,
                matrix_row_start_idx,
                chunk_bed_row_offset,
            );
        });
}

/// Processes a single block of individuals.
#[inline]
fn process_block(
    person_indices_in_block: &[OriginalPersonIndex],
    prep_result: &PreparationResult,
    snp_major_data: &[u8],
    weights_for_chunk: &[f32],
    block_scores_out: &mut [f32],
    block_missing_counts_out: &mut [u32],
    block_correction_sums_out: &mut [f32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
    matrix_row_start_idx: usize,
    chunk_bed_row_offset: usize,
) {
    let stride = prep_result.stride;
    let snps_in_chunk = if stride > 0 { weights_for_chunk.len() / stride } else { 0 };
    let tile_size = person_indices_in_block.len() * snps_in_chunk;

    let mut tile = tile_pool.pop().unwrap_or_default();
    tile.clear();
    tile.resize(tile_size, EffectAlleleDosage::default());

    // This pivot function now has a single, clear responsibility.
    pivot_tile(
        snp_major_data,
        person_indices_in_block,
        &mut tile,
        prep_result,
        matrix_row_start_idx,
        chunk_bed_row_offset,
    );

    process_tile(
        &tile,
        prep_result,
        weights_for_chunk,
        block_scores_out,
        block_missing_counts_out,
        block_correction_sums_out,
        sparse_index_pool,
        snps_in_chunk,
        matrix_row_start_idx,
    );

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel after
/// performing the critical pre-computation of sparse indices.
#[inline]
fn process_tile(
    tile: &[EffectAlleleDosage],
    prep_result: &PreparationResult,
    weights_for_chunk: &[f32],
    block_scores_out: &mut [f32],
    block_missing_counts_out: &mut [u32],
    block_correction_sums_out: &mut [f32],
    sparse_index_pool: &SparseIndexPool,
    snps_in_chunk: usize,
    matrix_row_start_idx: usize,
) {
    let num_scores = prep_result.score_names.len();
    let num_people_in_block = if snps_in_chunk > 0 { tile.len() / snps_in_chunk } else { 0 };

    // --- Pre-computation of Sparse Indices ---
    let thread_indices = sparse_index_pool.get_or_default();
    let (g1_indices, g2_indices) = &mut *thread_indices.borrow_mut();

    g1_indices.iter_mut().for_each(Vec::clear);
    g2_indices.iter_mut().for_each(Vec::clear);
    if g1_indices.len() < num_people_in_block {
        g1_indices.resize_with(num_people_in_block, Vec::new);
    }
    if g2_indices.len() < num_people_in_block {
        g2_indices.resize_with(num_people_in_block, Vec::new);
    }

    // This scan of the cache-hot, person-major tile generates the kernel's "work plan".
    for person_idx in 0..num_people_in_block {
        let genotype_row_start = person_idx * snps_in_chunk;
        let genotype_row_end = genotype_row_start + snps_in_chunk;
        let genotype_row = &tile[genotype_row_start..genotype_row_end];

        let person_data_start = person_idx * num_scores;
        let person_missing_counts_slice =
            &mut block_missing_counts_out[person_data_start..person_data_start + num_scores];
        let person_correction_sums_slice =
            &mut block_correction_sums_out[person_data_start..person_data_start + num_scores];

        for (snp_idx_in_chunk, &dosage) in genotype_row.iter().enumerate() {
            // The kernel operates in the local coordinate space of the current chunk.
            // We pass the local `snp_idx_in_chunk` directly, which is a valid
            // index into the `weights_for_chunk` slice that the kernel receives.
            match dosage.0 {
                // SAFETY: `person_idx` is guaranteed to be in bounds by the outer loop.
                1 => unsafe { g1_indices.get_unchecked_mut(person_idx).push(snp_idx_in_chunk) },
                2 => unsafe { g2_indices.get_unchecked_mut(person_idx).push(snp_idx_in_chunk) },
                3 => {
                    // This is the missing genotype sentinel.
                    let global_matrix_row_idx = matrix_row_start_idx + snp_idx_in_chunk;

                    // 1. Increment the missing count for ALL scores this variant belongs to. This is
                    //    required even for variants with no correction constant.
                    let scores_for_this_variant =
                        &prep_result.variant_to_scores_map[global_matrix_row_idx];
                    for &score_idx in scores_for_this_variant {
                        unsafe {
                            *person_missing_counts_slice.get_unchecked_mut(score_idx as usize) += 1;
                        }
                    }

                    // 2. Accumulate the correction constant using the FAST sparse map. This loop
                    //    only runs for variants that were allele-flipped.
                    let corrections_for_this_variant =
                        &prep_result.variant_to_corrections_map[global_matrix_row_idx];
                    for &(score_idx, correction_value) in corrections_for_this_variant {
                        unsafe {
                            *person_correction_sums_slice.get_unchecked_mut(score_idx as usize) +=
                                correction_value;
                        }
                    }
                }
                _ => (), // Dosage 0 is ignored.
            }
        }
    }
    // --- End of Pre-computation ---

    let weights_matrix =
        kernel::PaddedInterleavedWeights::new(weights_for_chunk, snps_in_chunk, num_scores)
            .expect("CRITICAL: Aligned weights matrix validation failed.");
    // This is now a sequential iterator over a block owned by a single Rayon thread.
    block_scores_out
        .chunks_exact_mut(num_scores)
        .enumerate()
        .for_each(|(person_idx, scores_out_slice)| {
            const MAX_SCORES: usize = 100;
            const MAX_ACC_LANES: usize = (MAX_SCORES + SIMD_LANES - 1) / SIMD_LANES;
            let mut acc_buffer = [kernel::SimdVec::splat(0.0); MAX_ACC_LANES];
            let num_accumulator_lanes = (num_scores + SIMD_LANES - 1) / SIMD_LANES;
            let acc_buffer_slice = &mut acc_buffer[..num_accumulator_lanes];

            // Dispatch to the simplified, "two-loop" SIMD kernel.
            kernel::accumulate_scores_for_person(
                &weights_matrix,
                scores_out_slice,
                acc_buffer_slice,
                &g1_indices[person_idx],
                &g2_indices[person_idx],
            );
        });
}

/// A cache-friendly, SIMD-accelerated pivot function using an 8x8 in-register transpose.
/// This function's sole purpose is to pivot raw genotype dosages from the SNP-major
/// .bed layout to a person-major tile layout. It performs no reconciliation.
#[inline]
fn pivot_tile(
    snp_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
    matrix_row_start_idx: usize,
    chunk_bed_row_offset: usize,
) {
    let num_people_in_block = person_indices_in_block.len();
    let snps_in_chunk = if num_people_in_block > 0 { tile.len() / num_people_in_block } else { 0 };
    let bytes_per_snp = prep_result.bytes_per_snp;
    let missing_sentinel_splat = U8xN::splat(3);

    for person_chunk_start in (0..num_people_in_block).step_by(SIMD_LANES) {
        let remaining_people = num_people_in_block - person_chunk_start;
        let current_lanes = remaining_people.min(SIMD_LANES);

        let person_indices = U64xN::from_array(core::array::from_fn(|i| {
            if i < current_lanes {
                person_indices_in_block[person_chunk_start + i].0 as u64
            } else { 0 }
        }));
        let person_byte_indices = person_indices / U64xN::splat(4);
        let bit_shifts = (person_indices % U64xN::splat(4)) * U64xN::splat(2);

        for snp_chunk_start in (0..snps_in_chunk).step_by(SIMD_LANES) {
            let remaining_snps = snps_in_chunk - snp_chunk_start;
            let current_snps = remaining_snps.min(SIMD_LANES);

            let mut snp_data_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..current_snps {
                let variant_idx_in_chunk = snp_chunk_start + i;
                let global_matrix_row_idx = matrix_row_start_idx + variant_idx_in_chunk;
                let absolute_bed_row = prep_result.required_bim_indices[global_matrix_row_idx];
                let relative_bed_row = absolute_bed_row - chunk_bed_row_offset;
                let snp_byte_offset = relative_bed_row as u64 * bytes_per_snp;
                let source_byte_indices = U64xN::splat(snp_byte_offset) + person_byte_indices;
                snp_data_vectors[i] =
                    U8xN::gather_or_default(snp_major_data, source_byte_indices.cast());
            }

            let mut dosage_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..current_snps {
                let packed_vals = snp_data_vectors[i];
                // --- SIMD Genotype Unpacking ---
                // This logic unpacks the 2-bit PLINK genotypes into dosage values.
                // The dosage extracted here is ALWAYS for the SECOND allele (allele2)
                // in the corresponding .bim file record. All flipping logic has been
                // pre-calculated into the correction_constants_matrix by prepare.rs.
                let two_bit_genotypes = (packed_vals >> bit_shifts.cast()) & U8xN::splat(0b11);

                // This formula maps 0b00->0, 0b10->1, 0b11->2. Missing (0b01) is handled next.
                let one = U8xN::splat(1);
                let term1 = (two_bit_genotypes >> U8xN::splat(1)) & one;
                let term2 = (two_bit_genotypes & one) + one;
                let initial_dosages = term1 * term2;
    
                #[cfg(debug_assertions)]
                {
                    static HAS_PRINTED_UNPACK_TRACE: AtomicBool = AtomicBool::new(false);
                
                    // Use `compare_exchange` for thread-safe "print-once" logic.
                    if !HAS_PRINTED_UNPACK_TRACE.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
                        // Extract lane 0 data for a clean, single-example trace.
                        let packed_val_0 = packed_vals.to_array()[0];
                        let bit_shift_0 = bit_shifts.to_array()[0];
                        let two_bit_geno_0 = two_bit_genotypes.to_array()[0];
                        let term1_0 = term1.to_array()[0];
                        let term2_0 = term2.to_array()[0];
                        let initial_dosage_0 = initial_dosages.to_array()[0];
                
                        eprintln!("
                ================ [debug] GENOTYPE UNPACKING TRACE (First Person/Variant) ================");
                        eprintln!("  Input Packed Byte:  {:#010b} ({})", packed_val_0, packed_val_0);
                        eprintln!("  Input Bit Shift:    {}", bit_shift_0);
                        eprintln!("  -----------------------------------------------------------------------------");
                        eprintln!("  -> Isolated 2-bit:   {:#04b} ({})", two_bit_geno_0, two_bit_geno_0);
                        eprintln!("  -> term1 ((g>>1)&1): {}", term1_0);
                        eprintln!("  -> term2 ((g&1)+1):  {}", term2_0);
                        eprintln!("  => Initial Dosage:   {}", initial_dosage_0);
                        eprintln!("====================================================================================
                ");
                    }
                }

                // Create a mask for missing genotypes (coded as 0b01) and use it to
                // select between the calculated dosage and the missing sentinel value.
                let missing_mask = two_bit_genotypes.simd_eq(U8xN::splat(1));
                let dosages =
                    missing_mask.select(missing_sentinel_splat, initial_dosages);

                dosage_vectors[i] = dosages;
            }

            // This highly optimized transpose is a core part of the engine. DO NOT CHANGE.
            let person_data_vectors = transpose_8x8_u8(dosage_vectors);

            let tile_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(tile.as_mut_ptr() as *mut u8, tile.len())
            };

            for i in 0..current_lanes {
                let person_idx_in_block = person_chunk_start + i;
                let dest_offset = person_idx_in_block * snps_in_chunk + snp_chunk_start;
                let dest_slice = &mut tile_u8[dest_offset..dest_offset + current_snps];
                let src_slice = &person_data_vectors[i].to_array()[..current_snps];
                dest_slice.copy_from_slice(src_slice);
            }
        }
    }
}

/// Helper function to perform an 8x8 byte matrix transpose using portable `std::simd`.
/// This is a standard, highly-optimized butterfly network algorithm. DO NOT CHANGE.
#[inline(always)]
fn transpose_8x8_u8(matrix: [U8xN; 8]) -> [U8xN; 8] {
    let [m0, m1, m2, m3, m4, m5, m6, m7] = matrix;

    // Stage 1: Interleave 8-bit elements
    let (t0, t1) = m0.interleave(m1);
    let (t2, t3) = m2.interleave(m3);
    let (t4, t5) = m4.interleave(m5);
    let (t6, t7) = m6.interleave(m7);

    // Stage 2: Interleave 16-bit elements
    let (s0, s1) = t0.cast::<u16>().interleave(t2.cast::<u16>());
    let (s2, s3) = t1.cast::<u16>().interleave(t3.cast::<u16>());
    let (s4, s5) = t4.cast::<u16>().interleave(t6.cast::<u16>());
    let (s6, s7) = t5.cast::<u16>().interleave(t7.cast::<u16>());

    // Stage 3: Interleave 32-bit elements
    let (r0, r1) = s0.cast::<u32>().interleave(s4.cast::<u32>());
    let (r2, r3) = s1.cast::<u32>().interleave(s5.cast::<u32>());
    let (r4, r5) = s2.cast::<u32>().interleave(s6.cast::<u32>());
    let (r6, r7) = s3.cast::<u32>().interleave(s7.cast::<u32>());

    [
        r0.cast(), r1.cast(), r2.cast(), r3.cast(),
        r4.cast(), r5.cast(), r6.cast(), r7.cast(),
    ]
}
