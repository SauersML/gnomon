// ========================================================================================
//
//               A tiled, cache-aware, contention-free compute engine
//
// ========================================================================================
//
// This module contains the synchronous, CPU-bound core of the compute pipeline. It is
// designed to be called from a higher-level asynchronous orchestrator within a
// `spawn_blocking` context. Its sole responsibility is to take a raw, variant-major
// chunk of genotype data, pivot it into a person-major tile, generate a sparse
// index of non-zero work, and dispatch it to the kernel. It performs ZERO
// scientific logic or reconciliation.

use crate::score::kernel;
use crate::score::types::{
    EffectAlleleDosage, OriginalPersonIndex, OutputPersonIndex, PreparationResult,
    ReconciledVariantIndex,
};
use crossbeam_queue::ArrayQueue;
use std::error::Error;
use std::simd::{Select, Simd, cmp::SimdPartialEq, num::SimdUint};

// --- SIMD & Engine Tuning Parameters ---
const SIMD_LANES: usize = 8;
type U64xN = Simd<u64, SIMD_LANES>;
type U8xN = Simd<u8, SIMD_LANES>;

/// The number of individuals to process in a single on-the-fly pivoted tile.
/// This value is tuned to ensure the tile fits comfortably within the L3 cache.
pub(crate) const PERSON_BLOCK_SIZE: usize = 4096;

/// The number of variants per kernel call bounds its working set. Both its
/// accumulators and the master score buffer retain f64 precision.
const KERNEL_MINI_BATCH_SIZE: usize = 256;
/// Number of scores to process per inner CPU stripe.
/// Must be a multiple of SIMD lanes so each stripe can read full vectors safely.
const CPU_SCORE_CHUNK_SIZE: usize = kernel::MAX_KERNEL_ACCUMULATOR_LANES * SIMD_LANES;

#[inline(always)]
fn append_dosage_indices(
    mut mask: u64,
    base: usize,
    indices: &mut [u16; KERNEL_MINI_BATCH_SIZE],
    count: &mut usize,
) {
    while mask != 0 {
        let lane = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        indices[*count] = (base + lane) as u16;
        *count += 1;
    }
}

// ========================================================================================
//                                   Public API
// ========================================================================================

/// Processes one dense, pre-filtered batch of variant-major data using the person-major
/// (pivot) path. This function is called from a parallel context (e.g., Rayon's
/// `par_bridge`), and all of its internal logic is sequential to prevent nested
/// parallelism deadlocks and maximize cache efficiency.
pub fn run_person_major_path(
    variant_major_data: &[u8],
    weights_for_batch: &[f64],
    missing_corrections_for_batch: &[f64],
    reconciled_variant_indices_for_batch: &[ReconciledVariantIndex],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f64],
    partial_missing_counts_out: &mut [u32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
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

    if (1..=4).contains(&prep_result.score_names.len())
        && prep_result.num_people_to_score >= 64
        && matches!(
            prep_result.person_subset,
            crate::score::types::PersonSubset::All
        )
    {
        macro_rules! narrow {
            ($columns:expr) => {
                run_narrow_scores_packed::<$columns>(
                    variant_major_data,
                    weights_for_batch,
                    missing_corrections_for_batch,
                    reconciled_variant_indices_for_batch,
                    prep_result,
                    partial_scores_out,
                    partial_missing_counts_out,
                )
            };
        }
        match prep_result.score_names.len() {
            1 => narrow!(1),
            2 => narrow!(2),
            3 => narrow!(3),
            4 => narrow!(4),
            _ => unreachable!("scoring requires a nonempty score panel"),
        }?;
        return Ok(());
    }

    let columns = prep_result.score_names.len();
    if (5..=64).contains(&columns)
        && prep_result.num_people_to_score >= 64
        && matches!(
            prep_result.person_subset,
            crate::score::types::PersonSubset::All
        )
        && reconciled_variant_indices_for_batch
            .iter()
            .map(|&index| prep_result.variant_csr_view(index).len())
            .sum::<usize>()
            <= reconciled_variant_indices_for_batch
                .len()
                .saturating_mul(columns / 4)
    {
        run_sparse_scores_packed::<false>(
            variant_major_data,
            weights_for_batch,
            missing_corrections_for_batch,
            reconciled_variant_indices_for_batch,
            prep_result,
            partial_scores_out,
            partial_missing_counts_out,
        );
        return Ok(());
    }

    if (1..=64).contains(&columns)
        && prep_result.num_people_to_score >= 64
        && matches!(
            prep_result.person_subset,
            crate::score::types::PersonSubset::Indices(_)
        )
        && (columns <= 4
            || reconciled_variant_indices_for_batch
                .iter()
                .map(|&index| prep_result.variant_csr_view(index).len())
                .sum::<usize>()
                <= reconciled_variant_indices_for_batch
                    .len()
                    .saturating_mul(columns / 4))
    {
        run_sparse_scores_packed::<true>(
            variant_major_data,
            weights_for_batch,
            missing_corrections_for_batch,
            reconciled_variant_indices_for_batch,
            prep_result,
            partial_scores_out,
            partial_missing_counts_out,
        );
        return Ok(());
    }

    // === Sequential compute within a single parallel task ===
    // This main loop is intentionally sequential. The outer pipeline (in pipeline.rs)
    // is responsible for parallelism by calling this function for different batches
    // on different threads. This avoids thread pool exhaustion and is highly
    // cache-friendly, as each thread works on its own disjoint data blocks.
    let num_scores = prep_result.score_names.len();
    let items_per_block = PERSON_BLOCK_SIZE * num_scores;

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

                let person_indices_in_block = &prep_result.output_idx_to_fam_idx
                    [person_output_start_idx..person_output_end_idx];

                process_block(
                    person_indices_in_block,
                    prep_result,
                    variant_major_data,
                    weights_for_batch,
                    missing_corrections_for_batch,
                    reconciled_variant_indices_for_batch,
                    block_scores_out,
                    block_missing_counts_out,
                    tile_pool,
                );
            },
        );

    Ok(())
}

// ========================================================================================
//                            Public entry points
// ========================================================================================

#[inline]
pub fn process_tile<'a>(
    tile: &'a [EffectAlleleDosage],
    prep_result: &'a PreparationResult,
    weights_for_batch: &'a [f64],
    missing_corrections_for_batch: &'a [f64],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
) {
    process_tile_impl(
        tile,
        prep_result,
        weights_for_batch,
        missing_corrections_for_batch,
        reconciled_variant_indices_for_batch,
        block_scores_out,
        block_missing_counts_out,
    );
}

// ========================================================================================
//                            Private implementation
// ========================================================================================

/// Processes a single block of individuals.
#[inline]
fn process_block<'a>(
    person_indices_in_block: &'a [OriginalPersonIndex],
    prep_result: &'a PreparationResult,
    variant_major_data: &'a [u8],
    weights: &'a [f64],
    missing_corrections: &'a [f64],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
    tile_pool: &'a ArrayQueue<Vec<EffectAlleleDosage>>,
) {
    let variants_in_chunk = reconciled_variant_indices_for_batch.len();
    let tile_size = person_indices_in_block.len() * variants_in_chunk;

    let mut tile = tile_pool.pop().unwrap_or_default();
    // Every element is overwritten by `pivot_tile`. Preserve an already-correct
    // length so repeated batches do not pointlessly clear and rewrite the whole
    // person-major tile before writing it again.
    if tile.len() != tile_size {
        tile.resize(tile_size, EffectAlleleDosage::default());
    }

    // This pivot function has a single, clear responsibility.
    pivot_tile(
        variant_major_data,
        person_indices_in_block,
        &mut tile,
        prep_result,
    );

    process_tile_impl(
        &tile,
        prep_result,
        weights,
        missing_corrections,
        reconciled_variant_indices_for_batch,
        block_scores_out,
        block_missing_counts_out,
    );

    // Return the tile to the pool for reuse.
    let _ = tile_pool.push(tile);
}

/// Accumulates a SIMD lane of adjustments into the score slice.
///
/// This function handles both full 8-element lanes and partial tail lanes.
/// After benchmarking, the unrolled scalar loop has been
/// empirically proven to be the fastest implementation for this specific
/// read-modify-write task on the target hardware.
#[inline(always)]
fn accumulate_simd_lane(
    scores_out_slice: &mut [f64],
    adjustments_f64x8: Simd<f64, 8>,
    scores_offset: usize,
    num_scores: usize,
) {
    // Check the full lane once, then add without per-element bounds checks or
    // precision conversions.
    if scores_offset + SIMD_LANES <= num_scores {
        let chunk = &mut scores_out_slice[scores_offset..scores_offset + SIMD_LANES];
        let current = Simd::<f64, SIMD_LANES>::from_slice(chunk);
        (current + adjustments_f64x8).copy_to_slice(chunk);
    } else {
        // Scalar fallback for the 1-7 element tail.
        let adj = adjustments_f64x8.to_array();
        let end = num_scores;
        for j in 0..(end - scores_offset) {
            scores_out_slice[scores_offset + j] += adj[j];
        }
    }
}

/// Dispatches a single, pivoted, person-major tile to the compute kernel after
/// calculating a baseline score and pre-computing sparse indices.
#[inline]
pub(crate) fn process_tile_impl<'a>(
    tile: &'a [EffectAlleleDosage],
    prep_result: &'a PreparationResult,
    weights_for_batch: &'a [f64],
    missing_corrections_for_batch: &'a [f64],
    reconciled_variant_indices_for_batch: &'a [ReconciledVariantIndex],
    block_scores_out: &mut [f64],
    block_missing_counts_out: &mut [u32],
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
    let score_chunk_size = CPU_SCORE_CHUNK_SIZE.max(SIMD_LANES);

    for variant_mini_batch_start in (0..variants_in_chunk).step_by(KERNEL_MINI_BATCH_SIZE) {
        let mini_batch_size =
            (variants_in_chunk - variant_mini_batch_start).min(KERNEL_MINI_BATCH_SIZE);
        if mini_batch_size == 0 {
            continue;
        }

        // --- Create Kernel Input Views ---
        let matrix_slice_start = variant_mini_batch_start * stride;
        let matrix_slice_end = matrix_slice_start + (mini_batch_size * stride);

        // SAFETY: The loop structure and mini-batch calculations ensure that the
        // `matrix_slice_start..matrix_slice_end` range is always within the bounds
        // of `weights_for_batch`. Using `get_unchecked`
        // bypasses the compiler's bounds checks, which is critical for performance
        // in this hot loop.
        let weights_chunk =
            unsafe { weights_for_batch.get_unchecked(matrix_slice_start..matrix_slice_end) };
        let missing_corr_chunk = unsafe {
            missing_corrections_for_batch.get_unchecked(matrix_slice_start..matrix_slice_end)
        };

        // SAFETY: The mini-batch slicing guarantees dimensions are coherent.
        let weights = unsafe {
            kernel::PaddedInterleavedWeights::new(weights_chunk, mini_batch_size, num_scores)
                .unwrap_unchecked()
        };

        // --- Single-Pass Stack-Buffered Processing ---
        // Instead of the 3-pass SparseIndexBuilder (count → allocate → fill), we now:
        // 1. Iterate over each person
        // 2. Scan their dosage row into stack-allocated arrays (512 bytes each, fits in L1)
        // 3. Call the kernel immediately
        // This eliminates all Vec allocations and reduces buffer passes from 3 to 1.
        for person_idx in 0..num_people_in_block {
            // Stack-allocated buffers for variant indices (max 256 variants per mini-batch)
            let mut g1_indices: [u16; KERNEL_MINI_BATCH_SIZE] = [0; KERNEL_MINI_BATCH_SIZE];
            let mut g2_indices: [u16; KERNEL_MINI_BATCH_SIZE] = [0; KERNEL_MINI_BATCH_SIZE];
            let mut missing_indices: [u16; KERNEL_MINI_BATCH_SIZE] = [0; KERNEL_MINI_BATCH_SIZE];
            let mut g1_count = 0usize;
            let mut g2_count = 0usize;
            let mut missing_count = 0usize;

            // Single-pass scan of this person's dosage row for this mini-batch
            let row_start = person_idx * variants_in_chunk + variant_mini_batch_start;
            let dosage_row = &tile[row_start..row_start + mini_batch_size];

            // --- SIMD-Accelerated Dosage Scan ---
            // Process 32 dosages at a time. For the common case (all zeros), this
            // reduces instruction count from ~100 to ~3 per 32 bytes.
            //
            // Safety: This transmutation requires EffectAlleleDosage to be exactly 1 byte.
            const _: () = assert!(std::mem::size_of::<EffectAlleleDosage>() == 1);
            let dosage_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(dosage_row.as_ptr() as *const u8, dosage_row.len())
            };

            let chunks = dosage_bytes.chunks_exact(32);
            let remainder_start = chunks.len() * 32;
            let mut base_idx = 0usize;
            let zero_vec = std::simd::Simd::<u8, 32>::splat(0);

            for chunk in chunks {
                // Load 32 bytes into a SIMD vector
                let vec: std::simd::Simd<u8, 32> = std::simd::Simd::from_slice(chunk);

                // Fast path: if entire chunk is zeros, skip it (common case)
                if vec == zero_vec {
                    base_idx += 32;
                    continue;
                }

                // Classify all lanes at once. Each mask walk writes one dosage
                // list in increasing variant order, without a genotype-dependent
                // branch or extracting individual bytes from the SIMD register.
                append_dosage_indices(
                    vec.simd_eq(std::simd::Simd::splat(1)).to_bitmask(),
                    base_idx,
                    &mut g1_indices,
                    &mut g1_count,
                );
                append_dosage_indices(
                    vec.simd_eq(std::simd::Simd::splat(2)).to_bitmask(),
                    base_idx,
                    &mut g2_indices,
                    &mut g2_count,
                );
                append_dosage_indices(
                    vec.simd_eq(std::simd::Simd::splat(3)).to_bitmask(),
                    base_idx,
                    &mut missing_indices,
                    &mut missing_count,
                );

                base_idx += 32;
            }

            // Handle remainder with scalar loop
            for i in remainder_start..mini_batch_size {
                let dosage = dosage_bytes[i];
                match dosage {
                    1 => {
                        g1_indices[g1_count] = i as u16;
                        g1_count += 1;
                    }
                    2 => {
                        g2_indices[g2_count] = i as u16;
                        g2_count += 1;
                    }
                    3 => {
                        missing_indices[missing_count] = i as u16;
                        missing_count += 1;
                    }
                    _ => (),
                }
            }

            let scores_out_slice =
                &mut block_scores_out[person_idx * num_scores..(person_idx + 1) * num_scores];
            let missing_counts_out_slice = &mut block_missing_counts_out
                [person_idx * num_scores..(person_idx + 1) * num_scores];

            // Missing correction applies once per missing variant and score.
            for &i in &missing_indices[..missing_count] {
                let i = i as usize;
                let variant_idx_in_chunk = variant_mini_batch_start + i;
                let reconciled_variant = reconciled_variant_indices_for_batch[variant_idx_in_chunk];
                let variant_view = prep_result.variant_csr_view(reconciled_variant);
                let weight_row_offset = i * stride;
                for contribution in variant_view.iter() {
                    let score_col = contribution.score_column.0;
                    missing_counts_out_slice[score_col] += 1;
                    scores_out_slice[score_col] -=
                        missing_corr_chunk[weight_row_offset + score_col] as f64;
                }
            }

            for score_chunk_start in (0..num_scores).step_by(score_chunk_size) {
                let score_chunk_end = (score_chunk_start + score_chunk_size).min(num_scores);
                let score_chunk_len = score_chunk_end - score_chunk_start;
                let score_chunk_lanes = score_chunk_len.div_ceil(SIMD_LANES);
                let kernel_result_buffer =
                    if score_chunk_lanes == kernel::MAX_KERNEL_ACCUMULATOR_LANES {
                        kernel::accumulate_adjustments_for_person(
                            &weights,
                            &g1_indices[..g1_count],
                            &g2_indices[..g2_count],
                            score_chunk_start,
                        )
                    } else {
                        kernel::accumulate_adjustments_for_person_lanes(
                            &weights,
                            &g1_indices[..g1_count],
                            &g2_indices[..g2_count],
                            score_chunk_start,
                            score_chunk_lanes,
                        )
                    };
                let score_chunk_out = &mut scores_out_slice[score_chunk_start..score_chunk_end];
                for i in 0..score_chunk_lanes {
                    let scores_offset = i * SIMD_LANES;
                    let adjustments_f64x8 = kernel_result_buffer[i];
                    accumulate_simd_lane(
                        score_chunk_out,
                        adjustments_f64x8,
                        scores_offset,
                        score_chunk_len,
                    );
                }
            }
        }
    }
}

/// A cache-friendly, SIMD-accelerated pivot function using an 8x8 in-register transpose.
/// This function's sole purpose is to pivot raw genotype dosages from the variant-major
/// .bed layout to a person-major tile layout. It performs no reconciliation.
#[inline]
fn pivot_tile(
    variant_major_data: &[u8],
    person_indices_in_block: &[OriginalPersonIndex],
    tile: &mut [EffectAlleleDosage],
    prep_result: &PreparationResult,
) {
    if matches!(
        prep_result.person_subset,
        crate::score::types::PersonSubset::All
    ) && !person_indices_in_block.is_empty()
    {
        pivot_packed_contiguous(
            variant_major_data,
            person_indices_in_block[0].0 as usize,
            person_indices_in_block.len(),
            prep_result.bytes_per_variant as usize,
            tile,
        );
        return;
    }
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

                // Scalar loads instead of `U8xN::gather_or_default`: AVX2 has no
                // u8 gather, so that intrinsic lowered to a per-lane bounds-mask
                // plus a branch-per-lane scalarized gather (8 unpredictable
                // branches + reassembly shuffles, ~30 insns). `get().unwrap_or(0)`
                // keeps the exact OOB->0 semantics but compiles to 8 straight-line
                // `movzbl` loads with perfectly-predicted in-bounds checks.
                let src = source_byte_indices.to_array();
                let mut packed = [0u8; SIMD_LANES];
                for lane in 0..SIMD_LANES {
                    packed[lane] = variant_major_data
                        .get(src[lane] as usize)
                        .copied()
                        .unwrap_or(0);
                }
                let packed_vals = U8xN::from_array(packed);
                let two_bit_genotypes = (packed_vals >> bit_shifts.cast()) & U8xN::splat(0b11);

                let one = U8xN::splat(1);
                let low_bit = two_bit_genotypes & one;
                let term1 = (two_bit_genotypes >> U8xN::splat(1)) & one;
                let term2 = low_bit + one;
                let initial_dosages = term1 * term2;

                // Branchless missing marker (genotype 0b01 -> dosage sentinel 3),
                // replacing the data-dependent per-lane `while missing_mask` scan.
                // `is_missing` is 1 exactly when genotype == 0b01 (high bit 0, low
                // bit 1) and 0 otherwise; `initial_dosages` is already 0 there, so
                // `+ is_missing*3` reproduces the old `dosage_arr[lane] = 3` exactly.
                let is_missing = (one - term1) * low_bit;
                dosage_vectors[i] = initial_dosages + is_missing * U8xN::splat(3);
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

/// A narrow score panel vectorizes across people instead of wasting seven SIMD
/// lanes on padded score columns. Packed bytes feed four independent vectors;
/// no genotype pivot, dosage index lists, or cohort-sized scratch is needed.
fn run_narrow_scores_packed<const COLUMNS: usize>(
    data: &[u8],
    weights: &[f64],
    corrections: &[f64],
    reconciled: &[ReconciledVariantIndex],
    prep: &PreparationResult,
    scores: &mut [f64],
    missing: &mut [u32],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let people = prep.num_people_to_score;
    let row_bytes = prep.bytes_per_variant as usize;
    let stride = prep.stride();
    // Presence is a property of the score row, including a present zero weight.
    // Read the CSR once, rather than again for every 32-person block. Ordinary
    // pipeline batches use the stack; larger caller-supplied batches reserve
    // their extra scratch fallibly.
    let mut inline_masks = [0u32; KERNEL_MINI_BATCH_SIZE];
    let mut extended_masks = Vec::new();
    let active_masks = if reconciled.len() <= inline_masks.len() {
        &mut inline_masks[..reconciled.len()]
    } else {
        extended_masks.try_reserve_exact(reconciled.len())?;
        extended_masks.resize(reconciled.len(), 0u32);
        extended_masks.as_mut_slice()
    };
    for (active, &index) in active_masks.iter_mut().zip(reconciled) {
        for contribution in prep.variant_csr_view(index).iter() {
            *active |= 1 << contribution.score_column.0;
        }
    }
    let full_people = people / 32 * 32;
    for person in (0..full_people).step_by(32) {
        let mut sums = [[Simd::<f64, 8>::splat(0.0); 4]; COLUMNS];
        let mut counts = [[Simd::<u32, 8>::splat(0); 4]; COLUMNS];
        for (variant, &active) in active_masks.iter().enumerate() {
            let offset = variant * row_bytes + person / 4;
            let packed: Simd<u32, 8> = Simd::<u8, 8>::from_slice(&data[offset..offset + 8]).cast();
            for lane in 0..4 {
                let code = (packed >> Simd::splat(2 * lane as u32)) & Simd::splat(3);
                let high = code >> Simd::splat(1);
                let dosage = high + (high & code);
                let absent = code.simd_eq(Simd::splat(1));
                for column in 0..COLUMNS {
                    let weight = Simd::splat(weights[variant * stride + column] as f64);
                    let correction = Simd::splat(-(corrections[variant * stride + column] as f64));
                    sums[column][lane] += absent
                        .cast::<i64>()
                        .select(correction, dosage.cast::<f64>() * weight);
                    counts[column][lane] +=
                        absent.select(Simd::splat((active >> column) & 1), Simd::splat(0));
                }
            }
        }
        for column in 0..COLUMNS {
            for lane in 0..4 {
                let sums = sums[column][lane].to_array();
                let counts = counts[column][lane].to_array();
                for byte in 0..8 {
                    let cell = (person + byte * 4 + lane) * COLUMNS + column;
                    scores[cell] += sums[byte];
                    missing[cell] += counts[byte];
                }
            }
        }
    }
    for person in full_people..people {
        for (variant, &mask) in active_masks.iter().enumerate() {
            let code = (data[variant * row_bytes + person / 4] >> (2 * (person % 4))) & 3;
            if code == 0 {
                continue;
            }
            let mut active = mask;
            while active != 0 {
                let column = active.trailing_zeros() as usize;
                active &= active - 1;
                let cell = person * COLUMNS + column;
                match code {
                    1 => {
                        scores[cell] -= corrections[variant * stride + column] as f64;
                        missing[cell] += 1;
                    }
                    _ => {
                        scores[cell] +=
                            (code - 1) as f64 * weights[variant * stride + column] as f64
                    }
                }
            }
        }
    }
    Ok(())
}

/// Compile one score's active rows into a bounded schedule, then keep its SIMD
/// accumulators in registers. Sparse columns never expand into a dosage tile.
fn run_sparse_scores_packed<const SELECTED: bool>(
    data: &[u8],
    weights: &[f64],
    corrections: &[f64],
    reconciled: &[ReconciledVariantIndex],
    prep: &PreparationResult,
    scores: &mut [f64],
    missing: &mut [u32],
) {
    let columns = prep.score_names.len();
    let people = prep.num_people_to_score;
    let row_bytes = prep.bytes_per_variant as usize;
    let stride = prep.stride();
    let full_people = people / 32 * 32;
    for chunk_start in (0..reconciled.len()).step_by(KERNEL_MINI_BATCH_SIZE) {
        let chunk_end = (chunk_start + KERNEL_MINI_BATCH_SIZE).min(reconciled.len());
        let mut schedule = [(0usize, 0.0f64, 0.0f64); KERNEL_MINI_BATCH_SIZE];
        let mut active_rows = [[0u64; KERNEL_MINI_BATCH_SIZE / 64]; 64];
        for variant in chunk_start..chunk_end {
            let row = variant - chunk_start;
            for entry in prep.variant_csr_view(reconciled[variant]).iter() {
                active_rows[entry.score_column.0][row / 64] |= 1u64 << (row % 64);
            }
        }
        // Keep a person's score/count cache lines hot across columns. Walking
        // the whole cohort for each column repeatedly streams the output matrix
        // once it outgrows the shared cache. Rebuilding the small row schedules
        // per person block trades cheap index work for much less output traffic.
        for person_start in (0..people).step_by(512) {
            let person_end = (person_start + 512).min(people);
            let vector_end = person_end.min(full_people);
            for column in 0..columns {
                let mut len = 0;
                for (word, &active) in active_rows[column].iter().enumerate() {
                    let mut active = active;
                    while active != 0 {
                        let variant = chunk_start + word * 64 + active.trailing_zeros() as usize;
                        active &= active - 1;
                        schedule[len] = (
                            variant * row_bytes,
                            weights[variant * stride + column] as f64,
                            corrections[variant * stride + column] as f64,
                        );
                        len += 1;
                    }
                }
                if len == 0 {
                    continue;
                }
                for person in (person_start..vector_end).step_by(32) {
                    // This topology is reused for every scheduled row. The complete
                    // cohort specialization eliminates it at compile time.
                    let byte_offsets: [[usize; 8]; 4] = if SELECTED {
                        core::array::from_fn(|lane| {
                            core::array::from_fn(|byte| {
                                prep.output_idx_to_fam_idx[person + byte * 4 + lane].0 as usize / 4
                            })
                        })
                    } else {
                        [[0; 8]; 4]
                    };
                    let shifts: [Simd<u32, 8>; 4] = if SELECTED {
                        core::array::from_fn(|lane| {
                            Simd::from_array(core::array::from_fn(|byte| {
                                2 * (prep.output_idx_to_fam_idx[person + byte * 4 + lane].0 % 4)
                            }))
                        })
                    } else {
                        [Simd::splat(0); 4]
                    };
                    let mut sums = [Simd::<f64, 8>::splat(0.0); 4];
                    let mut counts = [Simd::<u32, 8>::splat(0); 4];
                    for &(row_offset, weight, correction) in &schedule[..len] {
                        let offset = row_offset + person / 4;
                        let packed: Simd<u32, 8> = if SELECTED {
                            Simd::splat(0)
                        } else {
                            Simd::<u8, 8>::from_slice(&data[offset..offset + 8]).cast()
                        };
                        for lane in 0..4 {
                            let code = if SELECTED {
                                let gathered: Simd<u32, 8> =
                                    Simd::<u8, 8>::from_array(core::array::from_fn(|byte| {
                                        data[row_offset + byte_offsets[lane][byte]]
                                    }))
                                    .cast();
                                (gathered >> shifts[lane]) & Simd::splat(3)
                            } else {
                                (packed >> Simd::splat(2 * lane as u32)) & Simd::splat(3)
                            };
                            let high = code >> Simd::splat(1);
                            let dosage = high + (high & code);
                            let absent = code.simd_eq(Simd::splat(1));
                            sums[lane] += absent.cast::<i64>().select(
                                Simd::splat(-correction),
                                dosage.cast::<f64>() * Simd::splat(weight),
                            );
                            counts[lane] += absent.select(Simd::splat(1), Simd::splat(0));
                        }
                    }
                    for lane in 0..4 {
                        let sums = sums[lane].to_array();
                        let counts = counts[lane].to_array();
                        for byte in 0..8 {
                            let cell = (person + byte * 4 + lane) * columns + column;
                            scores[cell] += sums[byte];
                            missing[cell] += counts[byte];
                        }
                    }
                }
                for person in vector_end.max(person_start)..person_end {
                    let cell = person * columns + column;
                    let physical = if SELECTED {
                        prep.output_idx_to_fam_idx[person].0 as usize
                    } else {
                        person
                    };
                    for &(offset, weight, correction) in &schedule[..len] {
                        let code = (data[offset + physical / 4] >> (2 * (physical % 4))) & 3;
                        match code {
                            0 => (),
                            1 => {
                                scores[cell] -= correction;
                                missing[cell] += 1;
                            }
                            _ => scores[cell] += (code - 1) as f64 * weight,
                        }
                    }
                }
            }
        }
    }
}

/// Transpose two-bit calls before expanding them into dosage bytes. Each four
/// physical byte loads serve sixteen calls; a 1 KiB table expands each person's
/// four-call key in one load. Small person blocks keep the destination in L1.
fn pivot_packed_contiguous(
    data: &[u8],
    person_start: usize,
    people: usize,
    row_bytes: usize,
    tile: &mut [EffectAlleleDosage],
) {
    const DOSAGES: [[EffectAlleleDosage; 4]; 256] = {
        let mut table = [[EffectAlleleDosage(0); 4]; 256];
        let mut key = 0;
        while key < 256 {
            let mut lane = 0;
            while lane < 4 {
                table[key][lane] = EffectAlleleDosage(match (key >> (lane * 2)) & 3 {
                    0 => 0,
                    1 => 3,
                    2 => 1,
                    _ => 2,
                });
                lane += 1;
            }
            key += 1;
        }
        table
    };
    assert_eq!(person_start % 4, 0);
    let variants = tile.len() / people;
    for person_block in (0..people).step_by(32) {
        let person_end = (person_block + 32).min(people);
        for variant in (0..variants).step_by(4) {
            let count = (variants - variant).min(4);
            let mut sources = [&[][..]; 4];
            for local in 0..count {
                sources[local] =
                    &data[(variant + local) * row_bytes..(variant + local + 1) * row_bytes];
            }
            for person in (person_block..person_end).step_by(4) {
                let byte = (person_start + person) / 4;
                let mut calls = [0u8; 4];
                for local in 0..count {
                    calls[local] = sources[local][byte];
                }
                let keys = crate::genotype_table::transpose_calls(calls);
                for lane in 0..(person_end - person).min(4) {
                    let dst = (person + lane) * variants + variant;
                    tile[dst..dst + count].copy_from_slice(&DOSAGES[keys[lane] as usize][..count]);
                }
            }
        }
    }
}

/// Helper function to perform an 8x8 byte matrix transpose using portable `std::simd`.
/// This is a standard, highly-optimized butterfly network algorithm. Please maintain as is.
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

/// Classifies non-reference density for the path dispatcher using `popcnt`.
///
/// This function is a key performance enabler. It leverages the `popcnt` (population
/// count) CPU instruction, which is very fast. The decision tree's highest
/// density threshold is 0.0894, so common variants stop scanning as soon as they
/// cross it; only variants that may take the sparse path need an exact result.
#[inline]
pub fn assess_variant_density_for_dispatch(variant_data: &[u8], total_people: usize) -> f32 {
    if total_people == 0 {
        return 0.0;
    }

    const CHUNK_SIZE: usize = std::mem::size_of::<u64>();
    const HIGHEST_DISPATCH_THRESHOLD: f32 = 0.0894;
    let dense_cutoff = (total_people as f32 * HIGHEST_DISPATCH_THRESHOLD) as u64;
    let mut set_bits = 0u64;

    // Process full 8-byte chunks using `chunks_exact` for safety and performance.
    let chunks = variant_data.chunks_exact(CHUNK_SIZE);
    let remainder = chunks.remainder();
    for chunk in chunks {
        // This conversion is safe because chunks_exact guarantees the slice length is CHUNK_SIZE.
        let val = u64::from_ne_bytes(chunk.try_into().unwrap());
        set_bits += u64::from(val.count_ones());
        if set_bits > dense_cutoff {
            return 1.0;
        }
    }

    // Process the remainder byte by byte.
    for &byte in remainder {
        set_bits += u64::from(byte.count_ones());
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
    let variant_view = prep_result.variant_csr_view(reconciled_variant_index);

    if matches!(
        &prep_result.person_subset,
        crate::score::types::PersonSubset::All
    ) {
        run_variant_major_all_people(
            variant_data,
            prep_result.num_people_to_score,
            num_scores,
            variant_view,
            partial_scores_out,
            partial_missing_counts_out,
        );
        return Ok(());
    }

    // --- Main Compute Loop ---
    // This single loop iterates only over the individuals we need to score.
    // This is the core optimization that eliminates the massive allocation and
    // redundant work of the previous implementation.

    // Cache the last-read byte to avoid redundant memory reads. When output_idx_to_fam_idx
    // is sorted (which it typically is), consecutive people often share the same byte.
    let mut cached_byte_idx = usize::MAX;
    let mut cached_byte = 0u8;

    for out_idx in 0..prep_result.num_people_to_score {
        // Use a pre-computed map to find the original .fam index for this output slot.
        let output_person_idx = OutputPersonIndex(out_idx as u32);
        let original_fam_idx = prep_result
            .original_person_index_for_output(output_person_idx)
            .0 as usize;

        // --- On-the-fly Genotype Decoding with Byte Caching ---
        // Only re-read the byte if we've moved to a new position. This reduces
        // memory reads by ~75% since 4 people share each byte.
        let byte_index = original_fam_idx / 4;
        if byte_index != cached_byte_idx {
            cached_byte = variant_data[byte_index];
            cached_byte_idx = byte_index;
        }

        let bit_offset = (original_fam_idx % 4) * 2;
        let packed_val = (cached_byte >> bit_offset) & 0b11;

        // The logic for all dosage states is handled in a single, efficient match.
        match packed_val {
            // Homozygous reference (0b00) or other unknown values. Do nothing.
            // For sparse variants, this branch is highly predictable for the CPU.
            0b00 => (),

            // Missing genotype (0b01).
            0b01 => {
                let scores_offset = out_idx * num_scores;
                for contribution in variant_view.iter() {
                    let col = contribution.score_column.0;
                    partial_missing_counts_out[scores_offset + col] += 1;
                    partial_scores_out[scores_offset + col] -=
                        contribution.missing_correction as f64;
                }
            }

            // Heterozygous (0b10) or Homozygous alternate (0b11).
            0b10 | 0b11 => {
                let dosage = if packed_val == 0b10 { 1.0 } else { 2.0 };
                let scores_offset = out_idx * num_scores;
                for contribution in variant_view.iter() {
                    let col = contribution.score_column.0;
                    let weight = contribution.weight as f64;
                    let adjustment = weight * dosage;
                    partial_scores_out[scores_offset + col] += adjustment;
                }
            }

            // Should not be reached with valid PLINK data.
            _ => unreachable!(),
        }
    }

    Ok(())
}

/// Packed-byte fast path for the overwhelmingly common no-`--keep` case.
///
/// A zero PLINK byte represents four homozygous-reference calls. Skipping it as
/// one unit is particularly valuable for WGS rare variants, while decoding the
/// nonzero bytes four calls at a time also removes the output-to-FAM lookup,
/// division, and source reload from the per-person loop.
#[inline]
fn run_variant_major_all_people(
    variant_data: &[u8],
    num_people: usize,
    num_scores: usize,
    variant_view: crate::score::types::VariantCsrView<'_>,
    partial_scores_out: &mut [f64],
    partial_missing_counts_out: &mut [u32],
) {
    let full_bytes = num_people / 4;
    let packed_full_people = &variant_data[..full_bytes];
    let chunks = packed_full_people.chunks_exact(std::mem::size_of::<u64>());
    let trailing_bytes = chunks.remainder();
    const DENSITY_SAMPLE_WORDS: usize = 64;
    let sampled_words = chunks.clone().take(DENSITY_SAMPLE_WORDS);
    let sampled_word_count = sampled_words.len();
    let sampled_zero_words = sampled_words
        .filter(|chunk| u64::from_ne_bytes((*chunk).try_into().unwrap()) == 0)
        .count();
    let use_zero_word_skip = sampled_zero_words * 4 >= sampled_word_count;

    if use_zero_word_skip {
        for (chunk_idx, chunk) in chunks.enumerate() {
            let packed_word = u64::from_ne_bytes(chunk.try_into().unwrap());
            if packed_word == 0 {
                continue;
            }
            let byte_base = chunk_idx * std::mem::size_of::<u64>();
            for (byte_offset, &packed_byte) in chunk.iter().enumerate() {
                if packed_byte != 0 {
                    apply_packed_byte(
                        packed_byte,
                        (byte_base + byte_offset) * 4,
                        4,
                        num_scores,
                        variant_view,
                        partial_scores_out,
                        partial_missing_counts_out,
                    );
                }
            }
        }
    } else {
        for (byte_idx, &packed_byte) in packed_full_people[..full_bytes - trailing_bytes.len()]
            .iter()
            .enumerate()
        {
            if packed_byte != 0 {
                apply_packed_byte(
                    packed_byte,
                    byte_idx * 4,
                    4,
                    num_scores,
                    variant_view,
                    partial_scores_out,
                    partial_missing_counts_out,
                );
            }
        }
    }
    let trailing_byte_base = full_bytes - trailing_bytes.len();
    for (byte_offset, &packed_byte) in trailing_bytes.iter().enumerate() {
        if packed_byte != 0 {
            apply_packed_byte(
                packed_byte,
                (trailing_byte_base + byte_offset) * 4,
                4,
                num_scores,
                variant_view,
                partial_scores_out,
                partial_missing_counts_out,
            );
        }
    }

    let tail_people = num_people % 4;
    if tail_people != 0 {
        let packed_byte = variant_data[full_bytes];
        if packed_byte != 0 {
            apply_packed_byte(
                packed_byte,
                full_bytes * 4,
                tail_people,
                num_scores,
                variant_view,
                partial_scores_out,
                partial_missing_counts_out,
            );
        }
    }
}

#[inline(always)]
fn apply_packed_byte(
    packed_byte: u8,
    person_base: usize,
    people_in_byte: usize,
    num_scores: usize,
    variant_view: crate::score::types::VariantCsrView<'_>,
    partial_scores_out: &mut [f64],
    partial_missing_counts_out: &mut [u32],
) {
    let mut remaining = packed_byte;
    for lane in 0..people_in_byte {
        let genotype = remaining & 0b11;
        remaining >>= 2;
        if genotype == 0 {
            continue;
        }

        let scores_offset = (person_base + lane) * num_scores;
        if genotype == 1 {
            for contribution in variant_view.iter() {
                let cell = scores_offset + contribution.score_column.0;
                partial_missing_counts_out[cell] += 1;
                partial_scores_out[cell] -= contribution.missing_correction as f64;
            }
        } else {
            let dosage = if genotype == 2 { 1.0 } else { 2.0 };
            for contribution in variant_view.iter() {
                let cell = scores_offset + contribution.score_column.0;
                partial_scores_out[cell] += contribution.weight as f64 * dosage;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::types::{BimRowIndex, PipelineKind};
    use std::path::PathBuf;

    #[test]
    fn packed_pivot_preserves_calls_at_sample_and_variant_boundaries() {
        for start in [0, 4, 4096] {
            for people in [1, 3, 4, 5, 31, 32, 33, 65, 4099] {
                for variants in [1, 3, 4, 5, 8, 257] {
                    let row_bytes = (start + people + 3) / 4;
                    let data: Vec<u8> = (0..row_bytes * variants)
                        .map(|i| (i * 73 + i / row_bytes * 19) as u8)
                        .collect();
                    let mut tile = vec![EffectAlleleDosage(255); people * variants];
                    pivot_packed_contiguous(&data, start, people, row_bytes, &mut tile);
                    for person in 0..people {
                        for variant in 0..variants {
                            let physical = start + person;
                            let code = (data[variant * row_bytes + physical / 4]
                                >> (2 * (physical % 4)))
                                & 3;
                            let expected = [0, 3, 1, 2][code as usize];
                            assert_eq!(tile[person * variants + variant].0, expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn narrow_score_simd_matches_scalar_f64_and_missing_counts() {
        fn check<const COLUMNS: usize>() {
            for people in [64, 65, 95, 96, 129] {
                for variants in [1, 3, 255, 256, 257, 1025] {
                    for active in [1, COLUMNS] {
                        let mut prep = make_single_variant_multi_score_prep_result(people, active);
                        prep.score_names = (0..COLUMNS).map(|i| format!("S{i}")).collect();
                        prep.bytes_per_variant = people.div_ceil(4) as u64;
                        let stride = prep.stride();
                        let row_bytes = prep.bytes_per_variant as usize;
                        let data: Vec<u8> = (0..row_bytes * variants)
                            .map(|i| (i * 73 + i / row_bytes * 19) as u8)
                            .collect();
                        let weights: Vec<f64> = (0..stride * variants)
                            .map(|i| {
                                // Present zero weights still count missing calls.
                                if i % stride < active && i % 7 != 0 {
                                    (i % 73) as f64 / 173.0 - 0.21
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        let corrections: Vec<f64> = (0..weights.len())
                            .map(|i| {
                                if i % stride < active {
                                    (i % 43) as f64 / 137.0
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        let reconciled = vec![ReconciledVariantIndex(0); variants];
                        let mut scores = vec![0.125; people * COLUMNS];
                        let mut counts = vec![7; people * COLUMNS];
                        run_narrow_scores_packed::<COLUMNS>(
                            &data,
                            &weights,
                            &corrections,
                            &reconciled,
                            &prep,
                            &mut scores,
                            &mut counts,
                        )
                        .unwrap();
                        for person in 0..people {
                            for column in 0..COLUMNS {
                                let mut expected = 0.0;
                                let mut absent = 7;
                                for variant in 0..variants {
                                    let code = (data[variant * row_bytes + person / 4]
                                        >> (2 * (person % 4)))
                                        & 3;
                                    if code == 1 {
                                        expected -= corrections[variant * stride + column] as f64;
                                        absent += u32::from(column < active);
                                    } else if code > 1 {
                                        expected += (code - 1) as f64
                                            * weights[variant * stride + column] as f64;
                                    }
                                }
                                assert!(
                                    (scores[person * COLUMNS + column] - (0.125 + expected)).abs()
                                        < 1e-11
                                );
                                assert_eq!(counts[person * COLUMNS + column], absent);
                            }
                        }
                    }
                }
            }
        }
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
    }

    fn make_single_variant_prep_result(
        weight: f64,
        missing_correction: f64,
        num_people: usize,
    ) -> PreparationResult {
        let score_names = vec!["S0".to_string()];
        let stride = 8;
        let sparse_weights = vec![weight];
        let sparse_missing_correction = vec![missing_correction];
        let sparse_score_columns = vec![0u32];
        let sparse_row_offsets = vec![0u64, 1u64];

        let output_idx_to_fam_idx: Vec<crate::score::types::OriginalPersonIndex> = (0..num_people
            as u32)
            .map(crate::score::types::OriginalPersonIndex)
            .collect();
        let mut person_fam_to_output_idx = vec![None; num_people];
        for (out_idx, fam_idx) in output_idx_to_fam_idx.iter().enumerate() {
            person_fam_to_output_idx[fam_idx.0 as usize] =
                Some(crate::score::types::OutputPersonIndex(out_idx as u32));
        }

        PreparationResult::new(
            sparse_weights,
            sparse_missing_correction,
            sparse_score_columns,
            sparse_row_offsets,
            stride,
            vec![missing_correction as f64],
            vec![BimRowIndex(0)],
            vec![],
            score_names,
            vec![1],
            crate::score::types::PersonSubset::All,
            (0..num_people).map(|i| format!("I{i}")).collect(),
            num_people,
            num_people,
            1,
            1,
            1,
            person_fam_to_output_idx,
            output_idx_to_fam_idx,
            vec![0],
            vec![0],
            vec![0],
            1,
            PipelineKind::SingleFile(PathBuf::from("test")),
        )
    }

    fn make_single_variant_multi_score_prep_result(
        num_people: usize,
        num_scores: usize,
    ) -> PreparationResult {
        make_sparse_panel_prep(num_people, num_scores, &(0..num_scores).collect::<Vec<_>>())
    }

    fn make_sparse_panel_prep(
        num_people: usize,
        num_scores: usize,
        active: &[usize],
    ) -> PreparationResult {
        let score_names: Vec<String> = (0..num_scores).map(|i| format!("S{i}")).collect();
        let stride = num_scores.div_ceil(SIMD_LANES) * SIMD_LANES;
        let sparse_weights = vec![1.0f64; active.len()];
        let sparse_missing_correction = vec![0.0f64; active.len()];
        let sparse_score_columns: Vec<u32> = active.iter().map(|&i| i as u32).collect();
        let sparse_row_offsets = vec![0u64, active.len() as u64];

        let output_idx_to_fam_idx: Vec<crate::score::types::OriginalPersonIndex> = (0..num_people
            as u32)
            .map(crate::score::types::OriginalPersonIndex)
            .collect();
        let mut person_fam_to_output_idx = vec![None; num_people];
        for (out_idx, fam_idx) in output_idx_to_fam_idx.iter().enumerate() {
            person_fam_to_output_idx[fam_idx.0 as usize] =
                Some(crate::score::types::OutputPersonIndex(out_idx as u32));
        }

        PreparationResult::new(
            sparse_weights,
            sparse_missing_correction,
            sparse_score_columns,
            sparse_row_offsets,
            stride,
            vec![0.0; num_scores],
            vec![BimRowIndex(0)],
            vec![],
            score_names,
            vec![1; num_scores],
            crate::score::types::PersonSubset::All,
            (0..num_people).map(|i| format!("I{i}")).collect(),
            num_people,
            num_people,
            1,
            1,
            1,
            person_fam_to_output_idx,
            output_idx_to_fam_idx,
            vec![0],
            vec![0],
            vec![0],
            1,
            PipelineKind::SingleFile(PathBuf::from("test")),
        )
    }

    #[test]
    fn sparse_panel_simd_matches_scalar_at_column_and_person_boundaries() {
        for columns in [5usize, 8, 9, 16, 31, 32, 33, 64] {
            for people in [64usize, 65, 95, 129, 511, 512, 513, 1025] {
                for variants in [1usize, 3, 257] {
                    let active = [0, columns - 1];
                    let mut prep = make_sparse_panel_prep(people, columns, &active);
                    prep.bytes_per_variant = people.div_ceil(4) as u64;
                    let stride = prep.stride();
                    let row_bytes = prep.bytes_per_variant as usize;
                    let data: Vec<u8> = (0..row_bytes * variants)
                        .map(|i| (i * 73 + i / row_bytes * 19) as u8)
                        .collect();
                    let mut weights = vec![0.0f64; stride * variants];
                    let mut corrections = weights.clone();
                    for variant in 0..variants {
                        for column in active {
                            weights[variant * stride + column] =
                                (variant % 73) as f64 / 173.0 - 0.21;
                            corrections[variant * stride + column] = (variant % 43) as f64 / 137.0;
                        }
                    }
                    let reconciled = vec![ReconciledVariantIndex(0); variants];
                    let mut scores = vec![0.125; people * columns];
                    let mut counts = vec![7; people * columns];
                    run_sparse_scores_packed::<false>(
                        &data,
                        &weights,
                        &corrections,
                        &reconciled,
                        &prep,
                        &mut scores,
                        &mut counts,
                    );
                    for person in 0..people {
                        for column in 0..columns {
                            let mut expected = 0.0;
                            let mut absent = 7;
                            for variant in 0..variants {
                                let code = (data[variant * row_bytes + person / 4]
                                    >> (2 * (person % 4)))
                                    & 3;
                                if code == 1 {
                                    expected -= corrections[variant * stride + column] as f64;
                                    absent += u32::from(active.contains(&column));
                                } else if code > 1 {
                                    expected += (code - 1) as f64
                                        * weights[variant * stride + column] as f64;
                                }
                            }
                            assert!(
                                (scores[person * columns + column] - (0.125 + expected)).abs()
                                    < 1e-11
                            );
                            assert_eq!(counts[person * columns + column], absent);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn selected_packed_scores_match_scalar_for_gapped_and_unaligned_people() {
        for columns in [1usize, 4, 5, 32, 64] {
            for people in [64usize, 65, 95, 129, 511, 512, 513, 1025] {
                for step in [1usize, 3, 17] {
                    let physical_people = people * step + 3;
                    let active = if columns == 1 {
                        vec![0]
                    } else {
                        vec![0, columns - 1]
                    };
                    let mut prep = make_sparse_panel_prep(physical_people, columns, &active);
                    let indices: Vec<u32> = (0..people)
                        .map(|person| (person * step + 3) as u32)
                        .collect();
                    prep.num_people_to_score = people;
                    prep.bytes_per_variant = physical_people.div_ceil(4) as u64;
                    prep.output_idx_to_fam_idx =
                        indices.iter().copied().map(OriginalPersonIndex).collect();
                    prep.person_subset =
                        crate::score::types::PersonSubset::Indices(indices.clone());
                    let variants = 257;
                    let stride = prep.stride();
                    let row_bytes = prep.bytes_per_variant as usize;
                    let data: Vec<u8> = (0..row_bytes * variants)
                        .map(|i| (i * 73 + i / row_bytes * 19) as u8)
                        .collect();
                    let mut weights = vec![0.0f64; stride * variants];
                    let mut corrections = weights.clone();
                    for variant in 0..variants {
                        for &column in &active {
                            weights[variant * stride + column] =
                                (variant % 73) as f64 / 173.0 - 0.21;
                            corrections[variant * stride + column] = (variant % 43) as f64 / 137.0;
                        }
                    }
                    let reconciled = vec![ReconciledVariantIndex(0); variants];
                    let mut scores = vec![0.125; people * columns];
                    let mut counts = vec![7; people * columns];
                    run_sparse_scores_packed::<true>(
                        &data,
                        &weights,
                        &corrections,
                        &reconciled,
                        &prep,
                        &mut scores,
                        &mut counts,
                    );
                    for (person, &physical) in indices.iter().enumerate() {
                        let physical = physical as usize;
                        for column in 0..columns {
                            let mut expected = 0.0;
                            let mut absent = 7;
                            for variant in 0..variants {
                                let code = (data[variant * row_bytes + physical / 4]
                                    >> (2 * (physical % 4)))
                                    & 3;
                                if code == 1 {
                                    expected -= corrections[variant * stride + column] as f64;
                                    absent += u32::from(active.contains(&column));
                                } else if code > 1 {
                                    expected += (code - 1) as f64
                                        * weights[variant * stride + column] as f64;
                                }
                            }
                            assert!(
                                (scores[person * columns + column] - (0.125 + expected)).abs()
                                    < 1e-11
                            );
                            assert_eq!(counts[person * columns + column], absent);
                        }
                    }
                }
            }
        }
    }

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
    fn run_variant_major_path_applies_missing_correction_matrix() {
        let prep = make_single_variant_prep_result(1.5, 2.0, 4);
        let mut scores = vec![0.0f64; 4];
        let mut missing = vec![0u32; 4];

        // Per-person PLINK genotypes packed into one byte (2 bits per person):
        // p0=00 (dosage 0), p1=10 (dosage 1), p2=11 (dosage 2), p3=01 (missing)
        let variant_data = vec![0b01_11_10_00];

        run_variant_major_path(
            &variant_data,
            &prep,
            &mut scores,
            &mut missing,
            ReconciledVariantIndex(0),
        )
        .expect("variant-major path should succeed");

        assert!((scores[0] - 0.0).abs() < 1e-9);
        assert!((scores[1] - 1.5).abs() < 1e-9);
        assert!((scores[2] - 3.0).abs() < 1e-9);
        assert!((scores[3] - (-2.0)).abs() < 1e-9);
        assert_eq!(missing, vec![0, 0, 0, 1]);
    }

    #[test]
    fn run_variant_major_all_people_handles_word_skips_and_tail() {
        let num_people = 101;
        let prep = make_single_variant_prep_result(1.5, 2.0, num_people);
        let mut variant_data = vec![0u8; num_people.div_ceil(4)];
        let genotypes = [(3usize, 2u8), (32, 3), (99, 1), (100, 2)];
        for &(person, genotype) in &genotypes {
            variant_data[person / 4] |= genotype << ((person % 4) * 2);
        }
        let mut scores = vec![0.0f64; num_people];
        let mut missing = vec![0u32; num_people];

        run_variant_major_path(
            &variant_data,
            &prep,
            &mut scores,
            &mut missing,
            ReconciledVariantIndex(0),
        )
        .expect("variant-major path should succeed");

        assert_eq!(scores[3], 1.5);
        assert_eq!(scores[32], 3.0);
        assert_eq!(scores[99], -2.0);
        assert_eq!(scores[100], 1.5);
        assert_eq!(missing[99], 1);
        assert_eq!(missing.iter().sum::<u32>(), 1);
        assert_eq!(scores.iter().filter(|&&score| score != 0.0).count(), 4);
    }

    #[test]
    fn process_tile_classifies_simd_dosages_across_mini_batches() {
        let num_people = 5;
        for num_scores in [1, 9, 64, 67] {
            let prep = make_single_variant_multi_score_prep_result(num_people, num_scores);
            let stride = prep.stride();
            for variants in [31, 32, 33, 255, 256, 257, 513] {
                let weights: Vec<f64> = (0..variants * stride)
                    .map(|i| (i % 29) as f64 / 8.0 - 1.0)
                    .collect();
                let corrections = vec![0.25f64; variants * stride];
                let reconciled = vec![ReconciledVariantIndex(0); variants];
                let tile: Vec<EffectAlleleDosage> = (0..num_people)
                    .flat_map(|person| {
                        (0..variants).map(move |variant| {
                            EffectAlleleDosage(if person == 4 {
                                0
                            } else {
                                ((variant * 13 + variant / 7 + person) % 4) as u8
                            })
                        })
                    })
                    .collect();
                let mut scores = vec![0.0; num_people * num_scores];
                let mut missing = vec![0; scores.len()];
                process_tile(
                    &tile,
                    &prep,
                    &weights,
                    &corrections,
                    &reconciled,
                    &mut scores,
                    &mut missing,
                );
                for person in 0..num_people {
                    for score in 0..num_scores {
                        let mut expected = 0.0;
                        let mut expected_missing = 0;
                        for variant in 0..variants {
                            let dosage = tile[person * variants + variant].0;
                            if dosage == 3 {
                                expected -= 0.25;
                                expected_missing += 1;
                            } else {
                                expected +=
                                    dosage as f64 * weights[variant * stride + score] as f64;
                            }
                        }
                        assert_eq!(scores[person * num_scores + score], expected);
                        assert_eq!(missing[person * num_scores + score], expected_missing);
                    }
                }
            }
        }
    }

    #[test]
    fn process_tile_handles_more_than_100_scores_without_limit() {
        let num_people = 3usize;
        let num_scores = 577usize;
        let prep = make_single_variant_multi_score_prep_result(num_people, num_scores);
        let stride = prep.stride();

        let mut weights_for_batch = vec![0.0f64; stride];
        for w in weights_for_batch.iter_mut().take(num_scores) {
            *w = 1.0;
        }
        let missing_for_batch = vec![0.0f64; stride];
        let reconciled = vec![ReconciledVariantIndex(0)];
        let tile = vec![
            EffectAlleleDosage(1), // person 0, dosage=1
            EffectAlleleDosage(2), // person 1, dosage=2
            EffectAlleleDosage(0), // person 2, dosage=0
        ];
        let mut scores = vec![0.0f64; num_people * num_scores];
        let mut missing = vec![0u32; num_people * num_scores];

        process_tile(
            &tile,
            &prep,
            &weights_for_batch,
            &missing_for_batch,
            &reconciled,
            &mut scores,
            &mut missing,
        );

        for s in 0..num_scores {
            assert!(
                (scores[s] - 1.0).abs() < 1e-9,
                "person0 score mismatch at {s}"
            );
            assert!(
                (scores[num_scores + s] - 2.0).abs() < 1e-9,
                "person1 score mismatch at {s}"
            );
            assert!(
                (scores[2 * num_scores + s] - 0.0).abs() < 1e-9,
                "person2 score mismatch at {s}"
            );
        }
        assert!(missing.iter().all(|&m| m == 0));
    }
}
