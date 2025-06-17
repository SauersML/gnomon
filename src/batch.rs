// ========================================================================================
//
//               A TILED, CACHE-AWARE, CONTENTION-FREE COMPUTE ENGINE
//
// ========================================================================================
//
// This module contains the synchronous, CPU-bound core of the compute pipeline. It is
// designed to be called from a higher-level asynchronous orchestrator within a
// `spawn_blocking` context. Its sole responsibility is to take a raw, SNP-major
// chunk of genotype data, pivot it into a person-major tile, generate a sparse
// index of non-zero work, and dispatch it to the kernel. It performs ZERO
// scientific logic or reconciliation. We don't support over 100 scores yet.
// It will panic / overflow if we try.

use crate::kernel;
use crate::types::{CleanCounts, CleanScores, MatrixRowIndex};
use crate::types::{
    EffectAlleleDosage, OriginalPersonIndex, PersonSubset, PreparationResult,
};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::cell::RefCell;
use std::error::Error;
use std::simd::{cmp::SimdPartialEq, num::{SimdFloat, SimdUint}, Simd, f32x8, u8x8};
use thread_local::ThreadLocal;

// --- SIMD & Engine Tuning Parameters ---
const SIMD_LANES: usize = 8;
type U64xN = Simd<u64, SIMD_LANES>;
type U8xN = Simd<u8, SIMD_LANES>;

/// The number of individuals to process in a single on-the-fly pivoted tile.
/// This value is tuned to ensure the tile fits comfortably within the L3 cache.
const PERSON_BLOCK_SIZE: usize = 4096;

/// The number of variants to process in a single call to the compute kernel. This value
/// controls the frequency of flushing the `f32` accumulators to the `f64` master
/// buffer, which is the primary mechanism for guaranteeing numerical accuracy.
const KERNEL_MINI_BATCH_SIZE: usize = 256;

/// A thread-local pool for reusing the memory buffers required for storing
/// sparse indices (`g1_indices`, `g2_indices`) and deferred missingness events.
/// This is a critical performance optimization that avoids heap allocations in
/// the hot compute path.
#[derive(Default, Debug)]
pub struct SparseIndexPool {
    pool: ThreadLocal<RefCell<(Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<(usize, usize)>)>>,
}

impl SparseIndexPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets the thread-local buffer of sparse indices, creating it if it doesn't exist.
    #[inline(always)]
    fn get_or_default(&self) -> &RefCell<(Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<(usize, usize)>)> {
        self.pool.get_or_default()
    }
}

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// Processes one dense, pre-filtered batch of SNP-major data using the person-major
/// (pivot) path. This path is efficient for batches with high variant density.
pub fn run_person_major_path(
    snp_major_data: &[u8],
    metadata: &[MatrixRowIndex],
    prep_result: &PreparationResult,
    partial_scores_out: &mut CleanScores,
    partial_missing_counts_out: &mut CleanCounts,
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- Entry Point Validation ---
    // The type system has already guaranteed the buffers are zeroed.
    // We only need to check the length, which is possible via Deref.
    let expected_len = prep_result.num_people_to_score * prep_result.score_names.len();
    if partial_scores_out.len() != expected_len {
        return Err(Box::from(format!(
            "Mismatched scores buffer: expected length {}, got {}",
            expected_len,
            partial_scores_out.len()
        )));
    }

    // --- Dispatch to Generic Parallel Iterator ---
    // DerefMut coercion allows the typed wrappers to be passed as mutable slices
    // to the internal implementation, which requires no changes.
    match &prep_result.person_subset {
        PersonSubset::All => {
            let iter = (0..prep_result.total_people_in_fam as u32)
                .into_par_iter()
                .map(OriginalPersonIndex);
            process_people_iterator(
                iter,
                snp_major_data,
                metadata,
                prep_result,
                partial_scores_out,
                partial_missing_counts_out,
                tile_pool,
                sparse_index_pool,
            );
        }
        PersonSubset::Indices(indices) => {
            let iter = indices.par_iter().copied().map(OriginalPersonIndex);
            process_people_iterator(
                iter,
                snp_major_data,
                metadata,
                prep_result,
                partial_scores_out,
                partial_missing_counts_out,
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

/// Contains the main parallel processing logic, using an idiomatic Rayon structure
/// for high-performance, in-place mutation.
fn process_people_iterator<'a, I>(
    iter: I,
    snp_major_data: &'a [u8],
    metadata: &[MatrixRowIndex],
    prep_result: &'a PreparationResult,
    partial_scores: &'a mut [f64],
    partial_missing_counts: &'a mut [u32],
    tile_pool: &'a ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &'a SparseIndexPool,
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
        .enumerate()
        .for_each(|(block_idx, (block_scores_out, block_missing_counts_out))| {
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
                metadata,
                block_scores_out,
                block_missing_counts_out,
                tile_pool,
                sparse_index_pool,
            );
        });
}

/// Processes a single block of individuals.
#[inline]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn process_block(
    person_indices_in_block: &[OriginalPersonIndex],
    prep_result: &PreparationResult,
    snp_major_data: &[u8],
    metadata: &[MatrixRowIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
) {
    let snps_in_chunk = metadata.len();
    let tile_size = person_indices_in_block.len() * snps_in_chunk;

    let mut tile = tile_pool.pop().unwrap_or_default();
    tile.clear();
    tile.resize(tile_size, EffectAlleleDosage::default());

    // This pivot function has a single, clear responsibility.
    pivot_tile(
        snp_major_data,
        person_indices_in_block,
        &mut tile,
        prep_result,
    );

    process_tile(
        &tile,
        prep_result,
        metadata,
        block_scores_out,
        block_missing_counts_out,
        sparse_index_pool,
    );

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel after
/// calculating a baseline score and pre-computing sparse indices.
#[inline]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn process_tile(
    tile: &[EffectAlleleDosage],
    prep_result: &PreparationResult,
    metadata: &[MatrixRowIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
    sparse_index_pool: &SparseIndexPool,
) {
    let snps_in_chunk = metadata.len();
    let num_scores = prep_result.score_names.len();
    let num_people_in_block = if snps_in_chunk > 0 { tile.len() / snps_in_chunk } else { 0 };
    if num_people_in_block == 0 {
        return;
    }
    let stride = prep_result.stride();
    let num_accumulator_lanes = (num_scores + SIMD_LANES - 1) / SIMD_LANES;

    // --- Part 1: Pre-calculate the chunk-wide baseline for flipped variants (SIMD OPTIMIZED) ---
    // This baseline represents the score every person would get if they were
    // homozygous-reference for all flipped variants in this chunk. This is a hot
    // loop and is optimized to be branch-free using SIMD masking.
    let mut chunk_baseline = vec![0.0f64; stride]; // Use f64 for high-precision baseline.
    let weights_matrix = prep_result.weights_matrix();
    let flip_mask_matrix = prep_result.flip_mask_matrix();

    // Pre-load constants for the loop.
    let simd_lanes_f32 = std::mem::size_of::<f32x8>() / std::mem::size_of::<f32>();
    let num_simd_chunks = num_scores / simd_lanes_f32;
    let two_f32 = f32x8::splat(2.0);
    let zeros_f32 = f32x8::splat(0.0);

    for snp_idx in 0..snps_in_chunk {
        let global_matrix_row_idx = metadata[snp_idx].0 as usize;
        let row_offset = global_matrix_row_idx * stride;
        let weight_row = &weights_matrix[row_offset..row_offset + stride];
        let flip_row = &flip_mask_matrix[row_offset..row_offset + stride];

        // Process full SIMD vectors
        for i in 0..num_simd_chunks {
            let offset = i * simd_lanes_f32;

            // Load 8 weights (f32) and 8 flip flags (u8)
            let weights_vec = f32x8::from_slice(&weight_row[offset..]);
            let flips_vec = u8x8::from_slice(&flip_row[offset..]);

            // Create a mask where flip flag == 1. This is a branchless comparison.
            let flip_mask = flips_vec.simd_eq(u8x8::splat(1));

            // Calculate adjustment (2*W), but only where the mask is true.
            // Otherwise, the adjustment is 0. This is a branchless `if`.
            let adjustments_f32 = flip_mask.cast().select(two_f32 * weights_vec, zeros_f32);

            // Now, add the f32 adjustments to the f64 baseline using the same
            // efficient split-and-cast technique from the other loop.
            let adj_array = adjustments_f32.to_array();
            let adj_low = Simd::<f32, 4>::from_slice(&adj_array[0..4]);
            let adj_high = Simd::<f32, 4>::from_slice(&adj_array[4..8]);

            let adj_low_f64 = adj_low.cast::<f64>();
            let adj_high_f64 = adj_high.cast::<f64>();

            let baseline_slice_low = &mut chunk_baseline[offset..offset + 4];
            let mut baseline_low = Simd::<f64, 4>::from_slice(baseline_slice_low);
            baseline_low += adj_low_f64;
            baseline_slice_low.copy_from_slice(&baseline_low.to_array());

            let baseline_slice_high = &mut chunk_baseline[offset + 4..offset + 8];
            let mut baseline_high = Simd::<f64, 4>::from_slice(baseline_slice_high);
            baseline_high += adj_high_f64;
            baseline_slice_high.copy_from_slice(&baseline_high.to_array());
        }

        // Scalar fallback for the remainder
        let remainder_start = num_simd_chunks * simd_lanes_f32;
        for i in remainder_start..num_scores {
            if flip_row[i] == 1 {
                chunk_baseline[i] += 2.0 * (weight_row[i] as f64);
            }
        }
    }

    // Apply the baseline by copying it directly into the zeroed output buffer.
    for person_scores in block_scores_out.chunks_exact_mut(num_scores) {
        person_scores[..num_scores].copy_from_slice(&chunk_baseline[..num_scores]);
    }

// --- Part 2: Main Processing Loop (Mini-Batching) ---
    // Iterate over the tile in small, accuracy-preserving chunks.
    for snp_mini_batch_start in (0..snps_in_chunk).step_by(KERNEL_MINI_BATCH_SIZE) {
        let mini_batch_size =
            (snps_in_chunk - snp_mini_batch_start).min(KERNEL_MINI_BATCH_SIZE);
        if mini_batch_size == 0 {
            continue;
        }

        // --- A. Build Sparse Indices and RECORD Missingness for the Mini-Batch ---
        let thread_indices = sparse_index_pool.get_or_default();
        let (g1_indices, g2_indices, missing_events) = &mut *thread_indices.borrow_mut();

        // This logic ensures we only grow the vectors, then clear the relevant inner
        // vectors, preserving their allocated capacity. This prevents a "malloc storm"
        // that occurs when the outer vector is truncated and forced to re-allocate
        // all its inner vectors on the next full-sized block.
        if g1_indices.len() < num_people_in_block {
            g1_indices.resize_with(num_people_in_block, Default::default);
        }
        g1_indices.iter_mut().take(num_people_in_block).for_each(|v| v.clear());

        if g2_indices.len() < num_people_in_block {
            g2_indices.resize_with(num_people_in_block, Default::default);
        }
        g2_indices.iter_mut().take(num_people_in_block).for_each(|v| v.clear());
        missing_events.clear();

        for person_idx in 0..num_people_in_block {
            let genotype_tile_row_offset = person_idx * snps_in_chunk;
            let mini_batch_genotype_start = genotype_tile_row_offset + snp_mini_batch_start;
            let genotype_row_for_mini_batch =
                &tile[mini_batch_genotype_start..mini_batch_genotype_start + mini_batch_size];

            for (snp_idx_in_mini_batch, &dosage) in
                genotype_row_for_mini_batch.iter().enumerate()
            {
                match dosage.0 {
                    1 => unsafe {
                        g1_indices
                            .get_unchecked_mut(person_idx)
                            .push(snp_idx_in_mini_batch)
                    },
                    2 => unsafe {
                        g2_indices
                            .get_unchecked_mut(person_idx)
                            .push(snp_idx_in_mini_batch)
                    },
                    3 => {
                        // THE FIX: Defer the expensive work.
                        // Instead of doing the lookups and calculations now,
                        // just record the event and move on. This is extremely fast.
                        let snp_idx_in_chunk = snp_mini_batch_start + snp_idx_in_mini_batch;
                        missing_events.push((person_idx, snp_idx_in_chunk));
                    }
                    _ => (),
                }
            }
        }

        // --- B. Create Matrix Views for the Mini-Batch ---
        // Since the SNPs in the batch may not be contiguous in the original file,
        // we can no longer use a simple slice. Instead, we must gather the rows.
        let mut weights_chunk = Vec::with_capacity(mini_batch_size * stride);
        let mut flip_flags_chunk = Vec::with_capacity(mini_batch_size * stride);

        for i in 0..mini_batch_size {
            let snp_idx_in_chunk = snp_mini_batch_start + i;
            let matrix_row_idx = metadata[snp_idx_in_chunk].0 as usize;
            let row_offset = matrix_row_idx * stride;
            weights_chunk.extend_from_slice(&weights_matrix[row_offset..row_offset + stride]);
            flip_flags_chunk.extend_from_slice(&flip_mask_matrix[row_offset..row_offset + stride]);
        }

        let weights = kernel::PaddedInterleavedWeights::new(weights_chunk, mini_batch_size, num_scores)
            .expect("CRITICAL: Mini-batch weights matrix validation failed.");
        let flip_flags = kernel::PaddedInterleavedFlags::new(flip_flags_chunk, mini_batch_size, num_scores)
            .expect("CRITICAL: Mini-batch flip flags matrix validation failed.");

        // --- C. Dispatch to Kernel and Accumulate Results ---
        block_scores_out
            .chunks_exact_mut(num_scores)
            .enumerate()
            .for_each(|(person_idx, scores_out_slice)| {
                let kernel_result_buffer = kernel::accumulate_adjustments_for_person(
                    &weights,
                    &flip_flags,
                    &g1_indices[person_idx],
                    &g2_indices[person_idx],
                );

                // Add the kernel's calculated adjustments to the baseline.
                for i in 0..num_accumulator_lanes {
                    let scores_offset = i * SIMD_LANES;
                    let adjustments_f32x8 = kernel_result_buffer[i];

                    if scores_offset + SIMD_LANES <= num_scores {
                        let adj_array = adjustments_f32x8.to_array();
                        let adj_low_f32x4 = Simd::<f32, 4>::from_slice(&adj_array[0..4]);
                        let adj_high_f32x4 = Simd::<f32, 4>::from_slice(&adj_array[4..8]);

                        let adj_low_f64x4 = adj_low_f32x4.cast::<f64>();
                        let adj_high_f64x4 = adj_high_f32x4.cast::<f64>();

                        let scores_slice_low = &mut scores_out_slice[scores_offset..scores_offset + 4];
                        let mut current_scores_low = Simd::<f64, 4>::from_slice(scores_slice_low);
                        current_scores_low += adj_low_f64x4;
                        scores_slice_low.copy_from_slice(&current_scores_low.to_array());

                        let scores_slice_high =
                            &mut scores_out_slice[scores_offset + 4..scores_offset + 8];
                        let mut current_scores_high = Simd::<f64, 4>::from_slice(scores_slice_high);
                        current_scores_high += adj_high_f64x4;
                        scores_slice_high.copy_from_slice(&current_scores_high.to_array());
                    } else {
                        let start = scores_offset;
                        let end = num_scores;
                        let temp_array = adjustments_f32x8.to_array();
                        for j in 0..(end - start) {
                            scores_out_slice[start + j] += temp_array[j] as f64;
                        }
                    }
                }
            });

        // --- All Deferred Missing Events in a Batch ---
        // Now that the fast-path work is done, we process the slow-path work all at once.
        for &(person_idx, snp_idx_in_chunk) in missing_events.iter() {
            let global_matrix_row_idx = metadata[snp_idx_in_chunk].0 as usize;
            let scores_for_this_variant =
                &prep_result.variant_to_scores_map[global_matrix_row_idx];
            let weight_row_offset = global_matrix_row_idx * stride;
            let weight_row =
                &weights_matrix[weight_row_offset..weight_row_offset + num_scores];
            let flip_row =
                &flip_mask_matrix[weight_row_offset..weight_row_offset + num_scores];

            let person_scores_slice = &mut block_scores_out
                [person_idx * num_scores..(person_idx + 1) * num_scores];
            let person_missing_counts_slice = &mut block_missing_counts_out
                [person_idx * num_scores..(person_idx + 1) * num_scores];

            for &score_idx in scores_for_this_variant {
                let score_col = score_idx.0;
                unsafe {
                    *person_missing_counts_slice.get_unchecked_mut(score_col) += 1;
                    if flip_row[score_col] == 1 {
                        *person_scores_slice.get_unchecked_mut(score_col) -=
                            2.0 * (weight_row[score_col] as f64);
                    }
                }
            }
        }
    }
}

/// A cache-friendly, SIMD-accelerated pivot function using an 8x8 in-register transpose.
/// This function's sole purpose is to pivot raw genotype dosages from the SNP-major
/// .bed layout to a person-major tile layout. It performs no reconciliation.
#[inline]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn pivot_tile(
    snp_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
    matrix_row_start_idx: MatrixRowIndex,
    snps_in_chunk: usize,
) {
    let num_people_in_block = person_indices_in_block.len();
    let bytes_per_snp = prep_result.bytes_per_snp;

    // Maps a desired sequential SNP index (0-7) to its physical source location within
    // the shuffled vector produced by the `transpose_8x8_u8` function. This is used
    // to "un-shuffle" the data into the correct sequential order in the tile.
    const UNSHUFFLE_MAP: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

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

            // --- 1. Decode a block of up to 8 SNPs using SIMD ---
            let mut dosage_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..current_snps {
                let variant_idx_in_chunk = snp_chunk_start + i;
                // Each SNP's data follows the last one directly.
                let snp_byte_offset = variant_idx_in_chunk as u64 * bytes_per_snp;
                let source_byte_indices = U64xN::splat(snp_byte_offset) + person_byte_indices;

                let packed_vals =
                    U8xN::gather_or_default(snp_major_data, source_byte_indices.cast());
                let two_bit_genotypes = (packed_vals >> bit_shifts.cast()) & U8xN::splat(0b11);

                let one = U8xN::splat(1);
                let term1 = (two_bit_genotypes >> U8xN::splat(1)) & one;
                let term2 = (two_bit_genotypes & one) + one;
                let initial_dosages = term1 * term2;

                let missing_mask = two_bit_genotypes.simd_eq(U8xN::splat(1));
                dosage_vectors[i] = missing_mask.select(U8xN::splat(3), initial_dosages);
            }

            // --- 2. Transpose the 8x8 block ---
            let person_data_vectors = transpose_8x8_u8(dosage_vectors);

            // --- 3. Write data to the tile, handling full and partial chunks correctly ---
            for i in 0..current_lanes {
                let person_idx_in_block = person_chunk_start + i;
                let dest_offset = person_idx_in_block * snps_in_chunk + snp_chunk_start;
                let shuffled_person_row = person_data_vectors[i].to_array();

                        // OPTIMIZATION: Assemble the final row on the stack, then do a single bulk copy.
                        // This removes the previous `unsafe` block, replacing it with a safe,
                        // bounds-checked copy that is often optimized to a single instruction.

                        // 1. Create a small, stack-allocated buffer for the unshuffled row.
                        let mut temp_row = [EffectAlleleDosage::default(); SIMD_LANES];

                        // 2. Un-shuffle the transposed vector into the temporary buffer.
                            for j in 0..current_snps {
                            let dosage_value = shuffled_person_row[UNSHUFFLE_MAP[j]];
                            temp_row[j] = EffectAlleleDosage(dosage_value);
                            }

                        // 3. Perform a single, bulk copy into the main tile.
                        tile[dest_offset..dest_offset + current_snps]
                            .copy_from_slice(&temp_row[..current_snps]);
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


// ========================================================================================
//                     ADAPTIVE DISPATCHER & SNP-MAJOR PATH
// ========================================================================================

use crate::types::ComputePath;

// This threshold is used by the dispatcher. If the estimated non-reference allele
// frequency for a variant is above this value, it's considered "dense" and sent
// to the high-throughput person-major path. Otherwise, it's "sparse" and sent to
// the low-overhead SNP-major path. This value can be tuned based on profiling.
const SNP_DENSITY_THRESHOLD: f32 = 0.05;

/// Assesses a single SNP's data and determines the optimal compute path.
///
/// This is the "brain" of the adaptive engine. It uses a fast, hardware-accelerated
/// heuristic to decide if a variant's data is sparse or dense.
#[inline]
pub fn assess_path(snp_data: &[u8], total_people: usize) -> ComputePath {
    let non_ref_allele_freq = assess_snp_density(snp_data, total_people);

    if non_ref_allele_freq > SNP_DENSITY_THRESHOLD {
        ComputePath::PersonMajor
    } else {
        ComputePath::SnpMajor
    }
}

/// Calculates a fast proxy for non-reference allele frequency using `popcnt`.
///
/// This function is a key performance enabler. It leverages the `popcnt` (population
/// count) CPU instruction, which is extremely fast. By viewing the byte slice as
/// `u64` chunks, it minimizes loop iterations and lets the hardware do the heavy
// lifting of counting set bits, which is a strong proxy for data density.
#[inline]
fn assess_snp_density(snp_data: &[u8], total_people: usize) -> f32 {
    if total_people == 0 {
        return 0.0;
    }

    let (chunks, remainder) = snp_data.as_chunks::<u64>();

    let mut set_bits: u32 = 0;
    for &chunk in chunks {
        set_bits += chunk.count_ones();
    }
    for &byte in remainder {
        set_bits += byte.count_ones();
    }
    
    // Normalize by the number of people to get a comparable frequency.
    // The homozygous-reference genotype (0b00) has a popcnt of 0. All others
    // have a popcnt > 0. This gives a reliable, self-contained metric.
    set_bits as f32 / total_people as f32
}

/// Processes a single sparse SNP using a direct, pivot-free algorithm.
///
/// This path is optimized for variants where most individuals have the
/// homozygous-reference genotype. It avoids the high overhead of the pivot
/// operation by iterating through individuals and only performing work for
/// those with non-zero dosages.
pub fn run_snp_major_path(
    snp_data: &[u8],
    prep_result: &PreparationResult,
    partial_scores_out: &mut CleanScores,
    partial_missing_counts_out: &mut CleanCounts,
    snp_matrix_row: MatrixRowIndex,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let num_scores = prep_result.score_names.len();
    let stride = prep_result.stride();

    // --- 1. Decode this single SNP for all people in the cohort ---
    // This is a small, temporary allocation that fits on the stack for most cohorts.
    let dosages = decode_snp(snp_data, prep_result.total_people_in_fam);

    // --- 2. Get pointers to the relevant weight/flip data for this SNP ---
    let weight_row_offset = snp_matrix_row.0 as usize * stride;
    let weight_row = &prep_result.weights_matrix()[weight_row_offset..];
    let flip_row = &prep_result.flip_mask_matrix()[weight_row_offset..];
    let affected_scores = &prep_result.variant_to_scores_map[snp_matrix_row.0 as usize];

    // --- 3. First pass: Handle missingness and correct baselines ---
    // It is critical to correct the baseline score for missing individuals *before*
    // applying adjustments for non-missing ones. This is especially important for
    // flipped variants, where the baseline score is non-zero.
    for (original_fam_idx, &dosage) in dosages.iter().enumerate() {
        if dosage == 3 { // Missing Genotype
            if let Some(output_idx) = prep_result.person_fam_to_output_idx[original_fam_idx] {
                let scores_offset = output_idx as usize * num_scores;
                for score_col_idx in affected_scores {
                    let col = score_col_idx.0;
                    // If a variant was flipped, the baseline calculation assumes a score of
                    // 2 * W. We must subtract this back out for missing individuals.
                    if flip_row[col] == 1 {
                        partial_scores_out[scores_offset + col] -= 2.0 * weight_row[col] as f64;
                    }
                    partial_missing_counts_out[scores_offset + col] += 1;
                }
            }
        }
    }

    // --- 4. Main compute pass: Apply adjustments for non-zero dosages ---
    for (original_fam_idx, &dosage) in dosages.iter().enumerate() {
        // THE CORE OPTIMIZATION: For a sparse SNP, this branch is highly predictable.
        // We skip the vast majority of individuals who are homozygous-reference.
        if dosage == 0 || dosage == 3 {
            continue;
        }

        // Check if this person is in our output subset. This is an O(1) lookup.
        if let Some(output_idx) = prep_result.person_fam_to_output_idx[original_fam_idx] {
            let scores_offset = output_idx as usize * num_scores;
            
            // This person has a non-zero dosage. Apply adjustments to all relevant scores.
            for score_col_idx in affected_scores {
                let col = score_col_idx.0;
                let weight = weight_row[col] as f64;
                let adjustment = if flip_row[col] == 1 {
                    -weight * (dosage as f64)
                } else {
                    weight * (dosage as f64)
                };
                partial_scores_out[scores_offset + col] += adjustment;
            }
        }
    }

    Ok(())
}

/// Decodes a single SNP's raw byte data into a vector of dosages (0, 1, 2, or 3 for missing).
#[inline]
fn decode_snp(snp_data: &[u8], num_people: usize) -> Vec<u8> {
    let mut dosages = Vec::with_capacity(num_people);
    for i in 0..num_people {
        let byte_index = i / 4;
        let bit_offset = (i % 4) * 2;
        let packed_val = (snp_data[byte_index] >> bit_offset) & 0b11;
        
        let dosage = match packed_val {
            0b00 => 0, // Homozygous for first allele
            0b01 => 3, // Corresponds to PLINK's missing value
            0b10 => 1, // Heterozygous
            0b11 => 2, // Homozygous for second allele
            _ => unreachable!(),
        };
        dosages.push(dosage);
    }
    dosages
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_transpose_layout_is_empirically_verified() {
        // This map from a logical SNP index (0-7) to its physical byte position
        // in the transposed person-vector
        const SNP_TO_SHUFFLED_POS: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

        // 1. SETUP: Create a known 8x8 matrix of SNP-major data.
        // We use unique, traceable values: `(person_idx+1)*10 + (snp_idx+1)`.
        let mut snp_major_matrix = [[0u8; 8]; 8];
        for snp_idx in 0..8 {
            for person_idx in 0..8 {
                let val = ((person_idx + 1) * 10 + (snp_idx + 1)) as u8;
                snp_major_matrix[snp_idx][person_idx] = val;
            }
        }

        // Convert the scalar matrix into the SIMD vector format required by the function.
        // `input_vectors[j]` will hold data for SNP `j` across all 8 people.
        let input_vectors: [U8xN; 8] = core::array::from_fn(|j| {
            U8xN::from_array(snp_major_matrix[j])
        });

        // 2. EXECUTION: Perform the transpose operation we want to probe.
        let transposed_vectors = transpose_8x8_u8(input_vectors);

        // 3. VERIFICATION & REPORTING: Print the ground truth and assert correctness.
        eprintln!("\n\n=============== EMPIRICAL TRANSPOSE VERIFICATION ===============");
        eprintln!("Input Matrix (SNP-Major): One row per SNP");
        for snp_idx in 0..8 {
            eprintln!("  SNP {:?}: {:?}", snp_idx, snp_major_matrix[snp_idx]);
        }
        eprintln!("\n--- Transposed Output Layout (Person-Major) ---");
        eprintln!("Each row represents data for one Person, across all 8 SNPs...");

        let mut all_tests_passed = true;
        for person_idx in 0..8 {
            let person_row_actual = transposed_vectors[person_idx].to_array();
            eprintln!("  Person {:?}: {:?}", person_idx, person_row_actual);

            // Now, verify the contents of this person's row.
            // This loop will FAIL if the SNP data is not shuffled as hypothesized.
            for snp_idx in 0..8 {
                let expected_val = ((person_idx + 1) * 10 + (snp_idx + 1)) as u8;
                
                // Get the actual value from the shuffled location.
                let val_from_shuffled_pos = person_row_actual[SNP_TO_SHUFFLED_POS[snp_idx]];
                
                // Also get the value from the naive sequential position.
                let val_from_naive_pos = person_row_actual[snp_idx];

                if val_from_shuffled_pos != expected_val {
                    all_tests_passed = false;
                    eprintln!(
                        "    -> FAIL for P{},S{}: Expected {}, but value at shuffled pos [{}] was {}.",
                        person_idx, snp_idx, expected_val, SNP_TO_SHUFFLED_POS[snp_idx], val_from_shuffled_pos
                    );
                }
                if val_from_naive_pos == expected_val && SNP_TO_SHUFFLED_POS[snp_idx] != snp_idx {
                     all_tests_passed = false;
                     eprintln!(
                        "    -> FAIL: Naive position [{}] unexpectedly held the correct value for S{}!",
                        snp_idx, snp_idx
                    );
                }
            }
        }
        eprintln!("==============================================================\n");

        assert!(all_tests_passed, "The transpose output layout does not match the expected shuffle pattern.");
        eprintln!("✅ SUCCESS: The transpose function shuffles SNPs within each person-vector as hypothesized.");
    }
}
