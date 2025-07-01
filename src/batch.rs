// ========================================================================================
//
//               A TILED, CACHE-AWARE, CONTENTION-FREE COMPUTE ENGINE
//
// ========================================================================================
//
// This module contains the synchronous, CPU-bound core of the compute pipeline. It is
// designed to be called from a higher-level asynchronous orchestrator within a
// `spawn_blocking` context. Its sole responsibility is to take a raw, variant-major
// chunk of genotype data, pivot it into a person-major tile, generate a sparse
// index of non-zero work, and dispatch it to the kernel. It performs ZERO
// scientific logic or reconciliation. We don't support over 100 scores yet.
// It will panic / overflow if we try.

use crate::kernel;
use crate::types::{
    EffectAlleleDosage, OriginalPersonIndex, PersonSubset, PreparationResult,
    ReconciledVariantIndex,
};
use crossbeam_queue::ArrayQueue;
use num_cpus;
use std::error::Error;
use std::simd::{
    Simd,
    cmp::SimdPartialEq,
    num::{SimdFloat, SimdUint},
    simd_swizzle,
};

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

/// A thread-safe pool for reusing the large memory buffers required for storing
/// sparse indices (`g1_indices`, `g2_indices`) and deferred missingness events.
/// This implementation uses a lock-free queue, making it safe and efficient for
/// use with Rayon's work-stealing scheduler.
#[derive(Debug)]
pub struct SparseIndexPool {
    /// The lock-free queue holding the reusable buffer sets.
    pool: ArrayQueue<(Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<(usize, usize)>)>,
}

impl SparseIndexPool {
    /// Creates a new pool, pre-allocating a number of empty buffer sets
    /// proportional to the number of CPU cores. This "primes the pump"
    /// to avoid initial allocation stalls.
    pub fn new() -> Self {
        let pool = ArrayQueue::new(num_cpus::get() * 2);
        for _ in 0..num_cpus::get() * 2 {
            // Pre-populate with empty, but allocated, buffer sets.
            pool.push(Default::default()).unwrap();
        }
        Self { pool }
    }

    /// Pops a buffer set from the pool for use by a compute task.
    /// If the pool is empty, it returns a new default allocation.
    #[inline(always)]
    pub fn pop(&self) -> (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<(usize, usize)>) {
        self.pool.pop().unwrap_or_default()
    }

    /// Pushes a buffer set back into the pool after it has been used.
    /// This allows its memory to be recycled by another task.
    #[inline(always)]
    pub fn push(&self, buffers: (Vec<Vec<usize>>, Vec<Vec<usize>>, Vec<(usize, usize)>)) {
        // We don't care if the push fails (e.g., if the pool is full),
        // as the buffer will simply be dropped, which is safe.
        let _ = self.pool.push(buffers);
    }
}

// ========================================================================================
//                                   PUBLIC API
// ========================================================================================

/// Processes one dense, pre-filtered batch of variant-major data using the person-major
/// (pivot) path. This function is called from a parallel context (e.g., Rayon's
/// `par_bridge`), and all of its internal logic is sequential to prevent nested
/// parallelism deadlocks and maximize cache efficiency.
pub fn run_person_major_path(
    variant_major_data: &[u8],
    weights_for_batch: &[f32],
    flips_for_batch: &[u8],
    reconciled_variant_indices_for_batch: &[ReconciledVariantIndex],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f64],
    partial_missing_counts_out: &mut [u32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- Entry Point Validation ---
    let expected_len = prep_result.num_people_to_score * prep_result.score_names.len();
    if partial_scores_out.len() != expected_len {
        return Err(Box::from(format!(
            "Mismatched scores buffer: expected length {}, got {}",
            expected_len,
            partial_scores_out.len()
        )));
    }

    // === SEQUENTIAL COMPUTE WITHIN A SINGLE PARALLEL TASK ===
    // This main loop is intentionally sequential. The outer pipeline (in pipeline.rs)
    // is responsible for parallelism by calling this function for different batches
    // on different threads. This avoids thread pool exhaustion and is highly
    // cache-friendly, as each thread works on its own disjoint data blocks.
    let num_scores = prep_result.score_names.len();
    let items_per_block = PERSON_BLOCK_SIZE * num_scores;

    let person_indices_to_process = match &prep_result.person_subset {
        PersonSubset::All => None, // Will generate ranges on the fly.
        PersonSubset::Indices(indices) => Some(indices),
    };

    partial_scores_out
        .chunks_mut(items_per_block)
        .zip(partial_missing_counts_out.chunks_mut(items_per_block))
        .enumerate()
        .for_each(
            |(block_idx, (block_scores_out, block_missing_counts_out))| {
                let person_output_start_idx = block_idx * PERSON_BLOCK_SIZE;
                let person_output_end_idx = (person_output_start_idx + PERSON_BLOCK_SIZE)
                    .min(prep_result.num_people_to_score);

                if person_output_start_idx >= person_output_end_idx {
                    return;
                }

                // This temporary Vec holds the original FAM indices for only the people
                // in the current block. This is a small, fast allocation.
                let person_indices_in_block: Vec<OriginalPersonIndex> =
                    if let Some(indices) = person_indices_to_process {
                        // Case 1: Using a '--keep' file. We take a slice of the pre-sorted
                        // FAM indices that correspond to our current output block.
                        indices[person_output_start_idx..person_output_end_idx]
                            .iter()
                            .map(|&fam_idx| OriginalPersonIndex(fam_idx))
                            .collect()
                    } else {
                        // Case 2: Scoring all people. The FAM index is the same as the
                        // output index. We can generate this range on the fly.
                        (person_output_start_idx as u32..person_output_end_idx as u32)
                            .map(OriginalPersonIndex)
                            .collect()
                    };

                process_block(
                    &person_indices_in_block,
                    prep_result,
                    variant_major_data,
                    weights_for_batch,
                    flips_for_batch,
                    reconciled_variant_indices_for_batch,
                    block_scores_out,
                    block_missing_counts_out,
                    tile_pool,
                    sparse_index_pool,
                );
            },
        );

    Ok(())
}

// ========================================================================================
//                            PRIVATE IMPLEMENTATION
// ========================================================================================

/// Processes a single block of individuals.
#[cfg_attr(not(feature = "no-inline-profiling"), inline)]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn process_block<'a>(
    person_indices_in_block: &'a [OriginalPersonIndex],
    prep_result: &'a PreparationResult,
    variant_major_data: &'a [u8],
    weights: &'a [f32],
    flips: &'a [u8],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
    tile_pool: &'a ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &'a SparseIndexPool,
) {
    let variants_in_chunk = reconciled_variant_indices_for_batch.len();
    let tile_size = person_indices_in_block.len() * variants_in_chunk;

    let mut tile = tile_pool.pop().unwrap_or_default();
    tile.clear();
    tile.resize(tile_size, EffectAlleleDosage::default());

    // This pivot function has a single, clear responsibility.
    pivot_tile(
        variant_major_data,
        person_indices_in_block,
        &mut tile,
        prep_result,
    );

    process_tile(
        &tile,
        prep_result,
        weights,
        flips,
        reconciled_variant_indices_for_batch,
        block_scores_out,
        block_missing_counts_out,
        sparse_index_pool,
    );

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// A helper function to accumulate one SIMD lane of adjustments.
/// The `#[inline(always)]` attribute means this is a zero-cost abstraction.
#[inline(always)]
fn accumulate_simd_lane(
    scores_out_slice: &mut [f64],
    adjustments_f32x8: Simd<f32, 8>,
    scores_offset: usize,
    num_scores: usize,
) {
    if scores_offset + SIMD_LANES <= num_scores {
        // --- FINAL, REVISED OPTIMIZED VERSION ---

        // 1. Widen the f32 adjustments to f64 vectors in-register. This is efficient.
        let adj_low_f32x4: Simd<f32, 4> = simd_swizzle!(adjustments_f32x8, [0, 1, 2, 3]);
        let adj_high_f32x4: Simd<f32, 4> = simd_swizzle!(adjustments_f32x8, [4, 5, 6, 7]);
        let adj_low_f64x4 = adj_low_f32x4.cast::<f64>();
        let adj_high_f64x4 = adj_high_f32x4.cast::<f64>();

        // 2. Load data from memory.
        // SAFETY: The bounds check at the top of the function ensures that reading 8
        // elements from `scores_offset` is safe.
        let scores_low: Simd<f64, 4>;
        let scores_high: Simd<f64, 4>;
        unsafe {
            let base_ptr = scores_out_slice.as_ptr().add(scores_offset);
            scores_low = Simd::from_array([
                *base_ptr,
                *base_ptr.add(1),
                *base_ptr.add(2),
                *base_ptr.add(3),
            ]);
            scores_high = Simd::from_array([
                *base_ptr.add(4),
                *base_ptr.add(5),
                *base_ptr.add(6),
                *base_ptr.add(7),
            ]);
        }

        // 3. Perform the arithmetic in SIMD registers. This is fast.
        let result_low = scores_low + adj_low_f64x4;
        let result_high = scores_high + adj_high_f64x4;

        // 4. Write the results back to memory.
        //    Unaligned vector stores are generally less of a performance penalty
        //    than unaligned loads, so writing back the full vectors is fine.
        //
        // SAFETY: The bounds check also guarantees this write is safe.
        unsafe {
            let base_ptr = scores_out_slice.as_mut_ptr().add(scores_offset);
            (base_ptr as *mut Simd<f64, 4>).write_unaligned(result_low);
            (base_ptr.add(4) as *mut Simd<f64, 4>).write_unaligned(result_high);
        }
    } else {
        // The scalar fallback for the tail end of the data
        let start = scores_offset;
        let end = num_scores;
        let temp_array = adjustments_f32x8.to_array();
        for j in 0..(end - start) {
            scores_out_slice[start + j] += temp_array[j] as f64;
        }
    }
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel after
/// calculating a baseline score and pre-computing sparse indices.
#[cfg_attr(not(feature = "no-inline-profiling"), inline)]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn process_tile<'a>(
    tile: &'a [EffectAlleleDosage],
    prep_result: &'a PreparationResult,
    weights_for_batch: &'a [f32],
    flips_for_batch: &'a [u8],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
    sparse_index_pool: &'a SparseIndexPool,
) {
    let variants_in_chunk = reconciled_variant_indices_for_batch.len();
    let num_scores = prep_result.score_names.len();
    let num_people_in_block = if variants_in_chunk > 0 {
        tile.len() / variants_in_chunk
    } else {
        0
    };
    if num_people_in_block == 0 {
        return;
    }
    let stride = prep_result.stride();
    let num_accumulator_lanes = (num_scores + SIMD_LANES - 1) / SIMD_LANES;

    // Acquire a buffer set ONCE before all mini-batch loops.
    let mut buffers = sparse_index_pool.pop();

    // --- Part 2: Main Processing Loop (Mini-Batching) ---
    // Iterate over the tile in small, accuracy-preserving chunks.
    for variant_mini_batch_start in (0..variants_in_chunk).step_by(KERNEL_MINI_BATCH_SIZE) {
        let mini_batch_size =
            (variants_in_chunk - variant_mini_batch_start).min(KERNEL_MINI_BATCH_SIZE);
        if mini_batch_size == 0 {
            continue;
        }

        // --- A. Build Sparse Indices and RECORD Missingness for the Mini-Batch ---
        // The buffer set is local to this thread for the duration of the function.
        let (g1_indices, g2_indices, missing_events) = &mut buffers;

        // This logic ensures we only grow the vectors, then clear the relevant inner
        // vectors, preserving their allocated capacity. This prevents a "malloc storm"
        // that occurs when the outer vector is truncated and forced to re-allocate
        // all its inner vectors on the next full-sized block.
        if g1_indices.len() < num_people_in_block {
            g1_indices.resize_with(num_people_in_block, Default::default);
        }
        g1_indices
            .iter_mut()
            .take(num_people_in_block)
            .for_each(|v| v.clear());

        if g2_indices.len() < num_people_in_block {
            g2_indices.resize_with(num_people_in_block, Default::default);
        }
        g2_indices
            .iter_mut()
            .take(num_people_in_block)
            .for_each(|v| v.clear());
        missing_events.clear();

        for person_idx in 0..num_people_in_block {
            let genotype_tile_row_offset = person_idx * variants_in_chunk;
            let mini_batch_genotype_start = genotype_tile_row_offset + variant_mini_batch_start;
            let genotype_row_for_mini_batch =
                &tile[mini_batch_genotype_start..mini_batch_genotype_start + mini_batch_size];

            for (variant_idx_in_mini_batch, &dosage) in
                genotype_row_for_mini_batch.iter().enumerate()
            {
                match dosage.0 {
                    1 => unsafe {
                        g1_indices
                            .get_unchecked_mut(person_idx)
                            .push(variant_idx_in_mini_batch)
                    },
                    2 => unsafe {
                        g2_indices
                            .get_unchecked_mut(person_idx)
                            .push(variant_idx_in_mini_batch)
                    },
                    3 => {
                        // Defer the expensive work.
                        // Instead of doing the lookups and calculations now,
                        // just record the event and move on. This is extremely fast.
                        let variant_idx_in_chunk =
                            variant_mini_batch_start + variant_idx_in_mini_batch;
                        missing_events.push((person_idx, variant_idx_in_chunk));
                    }
                    _ => (),
                }
            }
        }

        // --- B. Create Kernel Input Views (Zero-Cost Slicing) ---
        // This is fast. We just create views over the contiguous,
        // pre-gathered data that was passed in for the entire batch.
        let matrix_slice_start = variant_mini_batch_start * stride;
        let matrix_slice_end = matrix_slice_start + (mini_batch_size * stride);
        let weights_chunk = &weights_for_batch[matrix_slice_start..matrix_slice_end];
        let flip_flags_chunk = &flips_for_batch[matrix_slice_start..matrix_slice_end];

        let weights =
            kernel::PaddedInterleavedWeights::new(weights_chunk, mini_batch_size, num_scores)
                .expect("CRITICAL: Mini-batch weights matrix validation failed.");
        let flip_flags =
            kernel::PaddedInterleavedFlags::new(flip_flags_chunk, mini_batch_size, num_scores)
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

                    accumulate_simd_lane(
                        scores_out_slice,
                        adjustments_f32x8,
                        scores_offset,
                        num_scores,
                    );
                }
            });

        // --- All Deferred Missing Events in a Batch ---
        // Now that the fast-path work is done, we process the slow-path work all at once.
        for &(person_idx, variant_idx_in_chunk) in missing_events.iter() {
            // This is a rare, slow path for missing data. We use the index metadata
            // to look up the global information needed.
            let global_matrix_row_idx =
                reconciled_variant_indices_for_batch[variant_idx_in_chunk].0 as usize;
            let scores_for_this_variant = &prep_result.variant_to_scores_map[global_matrix_row_idx];

            // For the baseline correction, we use the pre-gathered batch data, which is fast.
            let weight_row_offset = variant_idx_in_chunk * stride;
            let weight_row = &weights_for_batch[weight_row_offset..weight_row_offset + num_scores];
            let flip_row = &flips_for_batch[weight_row_offset..weight_row_offset + num_scores];

            let person_scores_slice =
                &mut block_scores_out[person_idx * num_scores..(person_idx + 1) * num_scores];
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

    // Return the buffers to the pool ONCE after all mini-batches are complete.
    sparse_index_pool.push(buffers);
}

/// A cache-friendly, SIMD-accelerated pivot function using an 8x8 in-register transpose.
/// This function's sole purpose is to pivot raw genotype dosages from the variant-major
/// .bed layout to a person-major tile layout. It performs no reconciliation.
#[cfg_attr(not(feature = "no-inline-profiling"), inline)]
#[cfg_attr(feature = "no-inline-profiling", inline(never))]
fn pivot_tile(
    variant_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
) {
    let num_people_in_block = person_indices_in_block.len();
    let bytes_per_variant = prep_result.bytes_per_variant;
    let variants_in_chunk = if num_people_in_block > 0 {
        tile.len() / num_people_in_block
    } else {
        0
    };

    // Maps a desired sequential variant index (0-7) to its physical source location within
    // the shuffled vector produced by the `transpose_8x8_u8` function. This is used
    // to "un-shuffle" the data into the correct sequential order in the tile.
    const UNSHUFFLE_MAP: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

    for person_chunk_start in (0..num_people_in_block).step_by(SIMD_LANES) {
        let remaining_people = num_people_in_block - person_chunk_start;
        let present_lanes = remaining_people.min(SIMD_LANES);

        let person_indices = U64xN::from_array(core::array::from_fn(|i| {
            if i < present_lanes {
                person_indices_in_block[person_chunk_start + i].0 as u64
            } else {
                0
            }
        }));
        let person_byte_indices = person_indices / U64xN::splat(4);
        let bit_shifts = (person_indices % U64xN::splat(4)) * U64xN::splat(2);

        for variant_chunk_start in (0..variants_in_chunk).step_by(SIMD_LANES) {
            let remaining_variants = variants_in_chunk - variant_chunk_start;
            let present_variants = remaining_variants.min(SIMD_LANES);

            // --- 1. Decode a block of up to 8 variants using SIMD ---
            let mut dosage_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..present_variants {
                let variant_idx_in_batch = variant_chunk_start + i;

                // The offset is simply its index in the batch multiplied by the bytes per variant.
                let variant_byte_offset = variant_idx_in_batch as u64 * bytes_per_variant;
                let source_byte_indices = U64xN::splat(variant_byte_offset) + person_byte_indices;

                let packed_vals =
                    U8xN::gather_or_default(variant_major_data, source_byte_indices.cast());
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
            for i in 0..present_lanes {
                let person_idx_in_block = person_chunk_start + i;
                let dest_offset = person_idx_in_block * variants_in_chunk + variant_chunk_start;
                let shuffled_person_row = person_data_vectors[i].to_array();

                // OPTIMIZATION: Assemble the final row on the stack, then do a single bulk copy.
                // This removes the previous `unsafe` block, replacing it with a safe,
                // bounds-checked copy that is often optimized to a single instruction.

                // 1. Create a small, stack-allocated buffer for the unshuffled row.
                let mut temp_row = [EffectAlleleDosage::default(); SIMD_LANES];

                // 2. Un-shuffle the transposed vector into the temporary buffer.
                for j in 0..present_variants {
                    let dosage_value = shuffled_person_row[UNSHUFFLE_MAP[j]];
                    temp_row[j] = EffectAlleleDosage(dosage_value);
                }

                // 3. Perform a single, bulk copy into the main tile.
                tile[dest_offset..dest_offset + present_variants]
                    .copy_from_slice(&temp_row[..present_variants]);
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

// ========================================================================================
//                     ADAPTIVE DISPATCHER & variant-MAJOR PATH
// ========================================================================================

/// Calculates a fast proxy for non-reference allele frequency using `popcnt`.
///
/// This function is a key performance enabler. It leverages the `popcnt` (population
/// count) CPU instruction, which is extremely fast. By viewing the byte slice as
/// `u64` chunks, it minimizes loop iterations and lets the hardware do the heavy
// lifting of counting set bits.
#[inline]
pub fn assess_variant_density(variant_data: &[u8], total_people: usize) -> f32 {
    if total_people == 0 {
        return 0.0;
    }

    const CHUNK_SIZE: usize = std::mem::size_of::<u64>();
    let mut set_bits: u32 = 0;

    // Process full 8-byte chunks using `chunks_exact` for safety and performance.
    let chunks = variant_data.chunks_exact(CHUNK_SIZE);
    let remainder = chunks.remainder();
    for chunk in chunks {
        // This conversion is safe because chunks_exact guarantees the slice length is CHUNK_SIZE.
        let val = u64::from_ne_bytes(chunk.try_into().unwrap());
        set_bits += val.count_ones();
    }

    // Process the remainder byte by byte.
    for &byte in remainder {
        set_bits += byte.count_ones();
    }

    // Normalize by the number of people to get a comparable frequency.
    // The homozygous-reference genotype (0b00) has a popcnt of 0. All others
    // have a popcnt > 0. This gives a reliable, self-contained metric.
    set_bits as f32 / total_people as f32
}

/// Processes a single sparse variant using a direct, pivot-free algorithm.
///
/// This path is optimized for variants where most individuals have the homozygous-
/// reference genotype. It avoids the high overhead of the pivot operation by
// iterating only over the individuals being scored and decoding their genotypes
// on-the-fly, which is allocation-free in the hot path.
pub fn run_variant_major_path(
    variant_data: &[u8],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f64],
    partial_missing_counts_out: &mut [u32],
    reconciled_variant_index: ReconciledVariantIndex,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let num_scores = prep_result.score_names.len();
    let stride = prep_result.stride();

    // Get pointers to the relevant weight/flip data for this variant. This is
    // an efficient calculation of the offset into the global contiguous matrices.
    let matrix_row_offset = reconciled_variant_index.0 as usize * stride;
    let weight_row = &prep_result.weights_matrix()[matrix_row_offset..matrix_row_offset + stride];
    let flip_row = &prep_result.flip_mask_matrix()[matrix_row_offset..matrix_row_offset + stride];
    let affected_scores = &prep_result.variant_to_scores_map[reconciled_variant_index.0 as usize];

    // --- Main Compute Loop ---
    // This single loop iterates only over the individuals we need to score.
    // This is the core optimization that eliminates the massive allocation and
    // redundant work of the previous implementation.
    for out_idx in 0..prep_result.num_people_to_score {
        // Use a pre-computed map to find the original .fam index for this output slot.
        let original_fam_idx = prep_result.output_idx_to_fam_idx[out_idx] as usize;

        // --- On-the-fly Genotype Decoding ---
        // This is extremely fast (a few bitwise operations) and allocation-free.
        let byte_index = original_fam_idx / 4;
        let bit_offset = (original_fam_idx % 4) * 2;
        let packed_val = (variant_data[byte_index] >> bit_offset) & 0b11;

        // The logic for all dosage states is handled in a single, efficient match.
        match packed_val {
            // Homozygous reference (0b00) or other unknown values. Do nothing.
            // For sparse variants, this branch is highly predictable for the CPU.
            0b00 => (),

            // Missing genotype (0b01).
            0b01 => {
                let scores_offset = out_idx * num_scores;
                for score_col_idx in affected_scores {
                    let col = score_col_idx.0;
                    partial_missing_counts_out[scores_offset + col] += 1;
                    // If the variant was flipped, its contribution (2*W) was added to the
                    // baseline. We must subtract it back out for missing individuals.
                    if flip_row[col] == 1 {
                        partial_scores_out[scores_offset + col] -= 2.0 * weight_row[col] as f64;
                    }
                }
            }

            // Heterozygous (0b10) or Homozygous alternate (0b11).
            0b10 | 0b11 => {
                let dosage = if packed_val == 0b10 { 1.0 } else { 2.0 };
                let scores_offset = out_idx * num_scores;
                for score_col_idx in affected_scores {
                    let col = score_col_idx.0;
                    let weight = weight_row[col] as f64;
                    // The baseline score already assumes a dosage of 0 for non-flipped variants,
                    // and 2 for flipped variants. This calculation applies the correct adjustment.
                    let adjustment = if flip_row[col] == 1 {
                        -weight * dosage
                    } else {
                        weight * dosage
                    };
                    partial_scores_out[scores_offset + col] += adjustment;
                }
            }

            // Should not be reached with valid PLINK data.
            _ => unreachable!(),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_transpose_layout_is_empirically_verified() {
        // NOTE: This test is intentionally verbose with `eprintln!` for one-off
        // visual verification. In a CI/CD environment, these prints would typically be removed.
        const VARIANT_TO_SHUFFLED_POS: [usize; 8] = [0, 4, 2, 6, 1, 5, 3, 7];

        let mut variant_major_matrix = [[0u8; 8]; 8];
        for variant_idx in 0..8 {
            for person_idx in 0..8 {
                let val = ((person_idx + 1) * 10 + (variant_idx + 1)) as u8;
                variant_major_matrix[variant_idx][person_idx] = val;
            }
        }

        let input_vectors: [U8xN; 8] =
            core::array::from_fn(|j| U8xN::from_array(variant_major_matrix[j]));

        let transposed_vectors = transpose_8x8_u8(input_vectors);

        eprintln!("\n\n=============== EMPIRICAL TRANSPOSE VERIFICATION ===============");
        eprintln!("Input Matrix (variant-Major): One row per variant");
        for variant_idx in 0..8 {
            eprintln!(
                "  variant {:?}: {:?}",
                variant_idx, variant_major_matrix[variant_idx]
            );
        }
        eprintln!("\n--- Transposed Output Layout (Person-Major) ---");
        eprintln!("Each row represents data for one Person, across all 8 variants...");

        let mut all_tests_passed = true;
        for person_idx in 0..8 {
            let person_row_actual = transposed_vectors[person_idx].to_array();
            eprintln!("  Person {:?}: {:?}", person_idx, person_row_actual);
            for variant_idx in 0..8 {
                let expected_val = ((person_idx + 1) * 10 + (variant_idx + 1)) as u8;
                let val_from_shuffled_pos = person_row_actual[VARIANT_TO_SHUFFLED_POS[variant_idx]];
                if val_from_shuffled_pos != expected_val {
                    all_tests_passed = false;
                    eprintln!(
                        "    -> FAIL for P{},S{}: Expected {}, but value at shuffled pos [{}] was {}.",
                        person_idx,
                        variant_idx,
                        expected_val,
                        VARIANT_TO_SHUFFLED_POS[variant_idx],
                        val_from_shuffled_pos
                    );
                }
            }
        }
        eprintln!("==============================================================\n");

        assert!(
            all_tests_passed,
            "The transpose output layout does not match the expected shuffle pattern."
        );
        eprintln!(
            "✅ SUCCESS: The transpose function shuffles variants within each person-vector as hypothesized."
        );
    }


    #[test]
    fn test_accumulation_performance() {
        // IMPORTANT: This test MUST be run in release mode (`--release`) to be meaningful.
    
        use std::simd::f32x8;
    
        // --- A. DEFINE COMPETING LOGIC & BENCHMARKING HELPERS ---
    
        // The function from the original code.
        use super::accumulate_simd_lane;
    
        /// Competitor 1: The simple scalar loop implementation.
        fn accumulate_simple_scalar(scores_out_slice: &mut [f64], adjustments_f32x8: f32x8) {
            let temp_array = adjustments_f32x8.to_array();
            for j in 0..8 {
                scores_out_slice[j] += temp_array[j] as f64;
            }
        }
        
        /// Competitor 2: A manually unrolled scalar loop. This avoids the `to_array()` overhead
        fn accumulate_unrolled_scalar(scores_out_slice: &mut [f64], adjustments_f32x8: f32x8) {
            let adj = adjustments_f32x8.to_array(); // Still need to get data out of SIMD reg
            scores_out_slice[0] += adj[0] as f64;
            scores_out_slice[1] += adj[1] as f64;
            scores_out_slice[2] += adj[2] as f64;
            scores_out_slice[3] += adj[3] as f64;
            scores_out_slice[4] += adj[4] as f64;
            scores_out_slice[5] += adj[5] as f64;
            scores_out_slice[6] += adj[6] as f64;
            scores_out_slice[7] += adj[7] as f64;
        }
    
    
        /// A struct to hold calculated statistics for a benchmark run.
        #[derive(Debug, Clone, Copy)]
        struct BenchmarkStats {
            mean: f64,
            median: f64,
            std_dev: f64,
            min: f64,
            max: f64,
        }
    
        /// Calculates statistics from a vector of trial durations.
        fn calculate_stats(durations_ns: &mut [f64]) -> BenchmarkStats {
            durations_ns.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let count = durations_ns.len() as f64;
            
            let sum = durations_ns.iter().sum::<f64>();
            let mean = sum / count;
    
            let median = if durations_ns.len() % 2 == 0 {
                (durations_ns[durations_ns.len() / 2 - 1] + durations_ns[durations_ns.len() / 2]) / 2.0
            } else {
                durations_ns[durations_ns.len() / 2]
            };
    
            let variance = durations_ns.iter().map(|&val| (val - mean).powi(2)).sum::<f64>() / count;
            let std_dev = variance.sqrt();
    
            BenchmarkStats {
                mean,
                median,
                std_dev,
                min: durations_ns[0],
                max: *durations_ns.last().unwrap(),
            }
        }
    
    
        /// The reusable performance measurement helper.
        /// It performs a warmup, then runs multiple trials and returns the results.
        fn run_benchmark_trials<F>(
            name: &str,
            scores: &mut [f64],
            adjustments: f32x8,
            warmup_iterations: u32,
            num_trials: u32,
            iterations_per_trial: u32,
            mut operation: F,
        ) -> BenchmarkStats
        where
            F: FnMut(&mut [f64], f32x8),
        {
            println!("> Benchmarking '{}'...", name);
            
            // 1. WARMUP PHASE: Run the operation to get the CPU and caches ready.
            let mut warmup_checksum = 0.0f64;
            for _ in 0..warmup_iterations {
                operation(scores, adjustments);
                warmup_checksum += scores[0];
            }
            std::hint::black_box(warmup_checksum);
    
    
            // 2. MEASUREMENT PHASE: Run multiple trials to collect robust data.
            let mut trial_durations = Vec::with_capacity(num_trials as usize);
            for i in 0..num_trials {
                // Reset state for EACH trial to ensure fairness.
                scores.fill(100.0);
                
                let mut checksum = 0.0f64;
                let start = Instant::now();
    
                for _ in 0..iterations_per_trial {
                    operation(scores, adjustments);
                    checksum += scores[0];
                }
    
                let duration = start.elapsed();
                trial_durations.push(duration);
    
                // Prevent the optimizer from eliding the loop.
                std::hint::black_box(checksum);
                
                if (i + 1) % 5 == 0 {
                    print!(".");
                    use std::io::{stdout, Write};
                    let _ = stdout().flush();
                }
            }
            println!(" Done.");
    
            let mut durations_ns: Vec<f64> = trial_durations
                .iter()
                .map(|d| d.as_nanos() as f64 / iterations_per_trial as f64)
                .collect();
            
            calculate_stats(&mut durations_ns)
        }
    
        // --- B. DEFINE TEST PARAMETERS ---
        const WARMUP_ITERATIONS: u32 = 200_000;
        const NUM_TRIALS: u32 = 20;
        const ITERATIONS_PER_TRIAL: u32 = 1_000_000;
        
        let adjustments = f32x8::from_array([0.1, 0.2, -0.05, 0.3, -0.15, 0.0, 0.5, -0.25]);
    
        // --- C. VERIFY CORRECTNESS FIRST ---
        // A fast but wrong function is useless. This must pass before we measure.
        {
            let mut complex_result = vec![100.0f64; 8];
            let mut scalar_result = vec![100.0f64; 8];
            let mut unrolled_result = vec![100.0f64; 8];
    
            accumulate_simd_lane(&mut complex_result, adjustments, 0, 8);
            accumulate_simple_scalar(&mut scalar_result, adjustments);
            accumulate_unrolled_scalar(&mut unrolled_result, adjustments);
    
            for i in 0..8 {
                assert!(
                    (complex_result[i] - scalar_result[i]).abs() < 1e-9,
                    "Correctness check FAILED at index {}: SIMD={}, Scalar={}",
                    i, complex_result[i], scalar_result[i]
                );
                assert!(
                    (complex_result[i] - unrolled_result[i]).abs() < 1e-9,
                    "Correctness check FAILED at index {}: SIMD={}, Unrolled Scalar={}",
                    i, complex_result[i], unrolled_result[i]
                );
            }
        }
    
        // --- D. RUN THE BENCHMARKS AND REPORT RESULTS ---
        println!("\n\n--- Accumulation Performance Test Report (Ran in RELEASE mode) ---");
        println!("Methodology: Warmup followed by {} trials of {} iterations each.", NUM_TRIALS, ITERATIONS_PER_TRIAL);
        println!("Correctness check passed. All functions produce identical results.");
    
        let mut scores_buffer = vec![100.0f64; 8];
    
        // Measure each implementation
        let stats_simd = run_benchmark_trials(
            "Complex SIMD (Current)", &mut scores_buffer, adjustments,
            WARMUP_ITERATIONS, NUM_TRIALS, ITERATIONS_PER_TRIAL,
            |s, a| accumulate_simd_lane(s, a, 0, 8),
        );
    
        let stats_scalar = run_benchmark_trials(
            "Simple Scalar", &mut scores_buffer, adjustments,
            WARMUP_ITERATIONS, NUM_TRIALS, ITERATIONS_PER_TRIAL,
            accumulate_simple_scalar,
        );
        
        let stats_unrolled = run_benchmark_trials(
            "Unrolled Scalar", &mut scores_buffer, adjustments,
            WARMUP_ITERATIONS, NUM_TRIALS, ITERATIONS_PER_TRIAL,
            accumulate_unrolled_scalar,
        );
    
        // --- E. ANALYZE AND ASSERT ---
        println!("\n--- Final Results (time per operation) ---");
        println!("-------------------------------------------------------------------------------------");
        println!("{:<22} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}", 
                 "Implementation", "Median", "Mean", "Std Dev", "Min", "Max");
        println!("-------------------------------------------------------------------------------------");
        
        let print_stats = |name: &str, stats: BenchmarkStats| {
            println!("{:<22} | {:>9.3} ns | {:>9.3} ns | {:>9.3} ns | {:>9.3} ns | {:>9.3} ns",
                     name, stats.median, stats.mean, stats.std_dev, stats.min, stats.max);
        };
    
        print_stats("Complex SIMD (Current)", stats_simd);
        print_stats("Simple Scalar", stats_scalar);
        print_stats("Unrolled Scalar", stats_unrolled);
        println!("-------------------------------------------------------------------------------------\n");
    
        // The core assertion is now based on the median, which is more robust to noise.
        assert!(
            stats_simd.median <= stats_scalar.median,
            "PERFORMANCE REGRESSION DETECTED against simple scalar!\n\
             - SIMD Median:   {:.3} ns/op\n\
             - Scalar Median: {:.3} ns/op",
            stats_simd.median, stats_scalar.median
        );
        
        assert!(
            stats_simd.median <= stats_unrolled.median,
            "PERFORMANCE REGRESSION DETECTED against unrolled scalar!\n\
             - SIMD Median:      {:.3} ns/op\n\
             - Unrolled Median:  {:.3} ns/op",
            stats_simd.median, stats_unrolled.median
        );
    
        println!("✅ PERFORMANCE CHECK PASSED: The current SIMD implementation's median performance is faster than or equal to both scalar implementations.");
        println!("======================================================================================================================================\n");
    }
}
