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
    EffectAlleleDosage, OriginalPersonIndex, PersonSubset,
    PreparationResult,
};
use crossbeam_queue::ArrayQueue;
use rayon::prelude::*;
use std::cell::RefCell;
use std::error::Error;
use std::simd::{cmp::SimdPartialEq, num::SimdUint, Simd};
#[cfg(debug_assertions)]
use std::sync::atomic::{AtomicBool, Ordering};

// --- SIMD & Engine Tuning Parameters ---
const SIMD_LANES: usize = 8;
type U64xN = Simd<u64, SIMD_LANES>;
type U8xN = Simd<u8, SIMD_LANES>;

/// The number of individuals to process in a single on-the-fly pivoted tile.
/// This value is tuned to ensure the tile fits comfortably within the L3 cache.
const PERSON_BLOCK_SIZE: usize = 4096;

use thread_local::ThreadLocal;

/// A thread-local pool for reusing the memory buffers required for storing
/// sparse indices (`g1_indices`, `g2_indices`).
#[derive(Default, Debug)]
pub struct SparseIndexPool {
    pool: ThreadLocal<RefCell<(Vec<Vec<usize>>, Vec<Vec<usize>>)>>,
}

impl SparseIndexPool {
    /// Creates a new, empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Gets the thread-local buffer of sparse indices.
    ///
    /// If this is the first time the current thread is accessing the pool, a new,
    /// empty buffer pair `(Vec::new(), Vec::new())` will be created and wrapped in
    /// a `RefCell` for it. Subsequent calls from the same thread will return a
    /// reference to the exact same `RefCell`-wrapped buffer, allowing for efficient reuse.
    #[inline(always)]
    fn get_or_default(&self) -> &RefCell<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
        self.pool.get_or_default()
    }
}

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
    corrections_for_chunk: &[f32],
    prep_result: &PreparationResult,
    partial_scores_out: &mut [f32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
    matrix_row_start_idx: usize,
    chunk_bed_row_offset: usize,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    // --- Entry Point Validation ---
    // This is the "airlock" for the compute engine. We verify that the mutable
    // slice provided by the caller has the exact size we expect. This prevents
    // panics or buffer overruns deep inside the parallel loops.
    let expected_len = prep_result.num_people_to_score * prep_result.score_names.len();
    if partial_scores_out.len() != expected_len {
        return Err(Box::from(format!( // Corrected error handling
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
                corrections_for_chunk,
                prep_result,
                partial_scores_out,
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
                corrections_for_chunk,
                prep_result,
                partial_scores_out,
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

/// Contains the main parallel processing logic, using a structure that is idiomatic
/// for high-performance, in-place mutation with Rayon.
fn process_people_iterator<'a, I>(
    iter: I,
    snp_major_data: &'a [u8],
    weights_for_chunk: &'a [f32],
    corrections_for_chunk: &'a [f32],
    prep_result: &'a PreparationResult,
    partial_scores: &'a mut [f32],
    tile_pool: &'a ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &'a SparseIndexPool,
    matrix_row_start_idx: usize,
    chunk_bed_row_offset: usize,
) where
    I: IndexedParallelIterator<Item = OriginalPersonIndex> + Send,
{
    let num_scores = prep_result.score_names.len();

    // Collect all person indices into a single vector. This allows us to map
    // chunks of the output scores back to chunks of input people.
    let all_person_indices: Vec<_> = iter.collect();

    // The number of f32 scores corresponding to a full block of people.
    let scores_per_block = PERSON_BLOCK_SIZE * num_scores;

    // The main parallel loop. It iterates over the OUTPUT buffer in mutable,
    // disjoint chunks. This is the idiomatic Rayon approach for in-place mutation.
    partial_scores
        .par_chunks_mut(scores_per_block)
        .enumerate()
        .for_each(|(block_idx, block_scores_out)| {
            // From the block index, calculate the corresponding range of people
            // from our collected input vector.
            let person_start_idx = block_idx * PERSON_BLOCK_SIZE;
            let person_end_idx =
                (person_start_idx + PERSON_BLOCK_SIZE).min(all_person_indices.len());

            let person_indices_in_block = &all_person_indices[person_start_idx..person_end_idx];

            // If this block corresponds to no people (can happen on the final,
            // smaller chunk), there is nothing to do.
            if person_indices_in_block.is_empty() {
                return;
            }

            // Now we have the input `person_indices_in_block` and the output
            // `block_scores_out` for this parallel task. Dispatch to the processing logic.
            process_block(
                person_indices_in_block,
                prep_result,
                snp_major_data,
                weights_for_chunk,
                corrections_for_chunk,
                block_scores_out,
                tile_pool,
                sparse_index_pool,
                matrix_row_start_idx,
                chunk_bed_row_offset,
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
    corrections_for_chunk: &[f32],
    block_scores_out: &mut [f32],
    tile_pool: &ArrayQueue<Vec<EffectAlleleDosage>>,
    sparse_index_pool: &SparseIndexPool,
    matrix_row_start_idx: usize,
    chunk_bed_row_offset: usize,
) {
    // Deduce the number of SNPs in this specific chunk from the length of the
    // padded weights slice. This logic must exactly match the padding logic
    // from the `prepare` phase to correctly interpret the data's dimensions.
    // let num_scores = prep_result.score_names.len(); // Already available
    let stride = prep_result.stride; // Use pre-calculated stride
    let snps_in_chunk = if stride > 0 {
        weights_for_chunk.len() / stride
    } else {
        0
    };
    let tile_size = person_indices_in_block.len() * snps_in_chunk;

    // Acquire a tile buffer from the pool.
    let mut tile = tile_pool.pop().unwrap_or_default();
    tile.clear();
    tile.resize(tile_size, EffectAlleleDosage::default());

    // This pivot function is now simpler: it only pivots genotypes and does
    // not perform any reconciliation, as that is handled by the "Compiler".
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
        corrections_for_chunk,
        block_scores_out,
        sparse_index_pool,
        snps_in_chunk,
        matrix_row_start_idx,
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
    corrections_for_chunk: &[f32],
    block_scores_out: &mut [f32],
    sparse_index_pool: &SparseIndexPool,
    // The number of SNPs in this specific chunk must be passed in explicitly.
    snps_in_chunk: usize,
    // The starting index into the global matrix for this chunk's variants.
    matrix_row_start_idx: usize,
) {
    let num_scores = prep_result.score_names.len();
    // The number of people is determined by the tile's total length divided by
    // the number of SNPs known to be in this specific chunk. This prevents a
    // division-by-zero if a chunk contains no relevant SNPs.
    let num_people_in_block = if snps_in_chunk > 0 {
        tile.len() / snps_in_chunk
    } else {
        0
    };

    // --- Pre-computation of Sparse Indices ---
    let thread_indices = sparse_index_pool.get_or_default();
    let (g1_indices, g2_indices) = &mut *thread_indices.borrow_mut();

    // Clear inner vectors to reuse their capacity.
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

        for (snp_idx_in_chunk, &dosage) in genotype_row.iter().enumerate() {
            // This is the critical optimization: we translate the sparse genotype index
            // into a direct matrix row index *once* here, instead of inside the
            // per-person kernel loop.
            let matrix_row_index = matrix_row_start_idx + snp_idx_in_chunk;

            // The dosage is repr(transparent) over u8.
            match dosage.0 {
                // SAFETY: `person_idx` is guaranteed to be in bounds because the `g1_indices`
                // and `g2_indices` vectors were resized to `num_people_in_block` and the
                // outer loop runs from `0..num_people_in_block`.
                1 => unsafe { g1_indices.get_unchecked_mut(person_idx).push(matrix_row_index) },
                2 => unsafe { g2_indices.get_unchecked_mut(person_idx).push(matrix_row_index) },
                _ => (), // Dosage 0 or 3 (missing) are ignored.
            }
        }
    }
    // --- End of Pre-computation ---

    let weights_matrix =
        kernel::PaddedInterleavedWeights::new(weights_for_chunk, snps_in_chunk, num_scores)
            .expect("CRITICAL: Aligned weights matrix validation failed.");
    let corrections_matrix =
        kernel::PaddedInterleavedWeights::new(corrections_for_chunk, snps_in_chunk, num_scores)
            .expect("CRITICAL: Correction constants matrix validation failed.");

    // This is now a sequential iterator over a block owned by a single rayon thread.
    block_scores_out
        .chunks_exact_mut(num_scores)
        .enumerate()
        .for_each(|(person_idx, scores_out_slice)| {
            // PERFORMANCE: This accumulator is a stack-allocated array.
            // With a maximum of 100 scores, the size is ceil(100 / 8) = 13 SIMD vectors,
            // or 13 * 32 = 416 bytes. This is trivial for the stack and avoids all
            // overhead from heap allocation, ThreadLocal lookups, and RefCell borrows.
            const MAX_SCORES: usize = 100;
            const MAX_ACC_LANES: usize = (MAX_SCORES + SIMD_LANES - 1) / SIMD_LANES;
            let mut acc_buffer = [kernel::SimdVec::splat(0.0); MAX_ACC_LANES];

            let num_accumulator_lanes = (num_scores + SIMD_LANES - 1) / SIMD_LANES;
            let acc_buffer_slice = &mut acc_buffer[..num_accumulator_lanes];

// ================== PROBE #2 START ==================
#[cfg(debug_assertions)]
{
    // This flag is also declared inside the debug block for zero release cost.
    static HAS_PRINTED_SPARSE_SUMMARY: AtomicBool = AtomicBool::new(false);

    // Use `compare_exchange` to print a summary for the very first person only.
    if !HAS_PRINTED_SPARSE_SUMMARY.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_err() {
        let g1_count = g1_indices[person_idx].len();
        let g2_count = g2_indices[person_idx].len();

        eprintln!("

================ [debug] KERNEL INPUT SUMMARY (First Person) ===================");
        eprintln!("  (Final data prepared for the computation kernel)");
        eprintln!("  -----------------------------------------------------------------------------");
        eprintln!("  Num Variants with Dosage = 1 (g1_indices.len()): {}", g1_count); // Corrected Terminology
        eprintln!("  Num Variants with Dosage = 2 (g2_indices.len()): {}", g2_count); // Corrected Terminology
        eprintln!("====================================================================================
");
    }
}
// =================== PROBE #2 END ===================
            kernel::accumulate_scores_for_person(
                &weights_matrix,
                &corrections_matrix,
                scores_out_slice,
                acc_buffer_slice,
                &g1_indices[person_idx],
                &g2_indices[person_idx],
            );
        });
}

/// A cache-friendly, SIMD-accelerated pivot function using an 8x8 in-register transpose.
/// This function no longer performs reconciliation; its sole purpose is to pivot
/// genotype dosages from the SNP-major .bed layout to a person-major tile layout.
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
    let snps_in_chunk = if num_people_in_block > 0 {
        tile.len() / num_people_in_block
    } else {
        0
    };
    let bytes_per_snp = prep_result.bytes_per_snp; // Use centralized value

    #[cfg(debug_assertions)]
    {
        // compute the pointer and length once
        // let ptr = tile.as_ptr() as usize; // ptr variable not used
        // let slice_len = snp_major_data.len(); // slice_len variable not used
        // debug_assert!( // This assertion might be too strict if bytes_per_snp is small or tile is not aligned
        //     ptr % bytes_per_snp as usize == 0,
        //     "pivot: slice ptr {:#x} not aligned to bytes_per_snp {}",
        //     ptr,
        //     bytes_per_snp
        // );
        eprintln!(
            "[debug] pivot_tile: num_people_in_block={} snps_in_chunk={} bytes_per_snp={}",
            num_people_in_block, snps_in_chunk, bytes_per_snp
        );
    }

    // let two_splat = U8xN::splat(2); // Not used for reconciliation
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
        // PLINK .bed files pack 4 genotypes into a byte. Find the byte index and
        // the 2-bit shift required for each person. These operations are performed
        // in parallel on all 8 lanes of the SIMD vector.
        let person_byte_indices = person_indices / U64xN::splat(4);
        let bit_shifts = (person_indices % U64xN::splat(4)) * U64xN::splat(2);

        for snp_chunk_start in (0..snps_in_chunk).step_by(SIMD_LANES) {
            let remaining_snps = snps_in_chunk - snp_chunk_start;
            let current_snps = remaining_snps.min(SIMD_LANES);

            let mut snp_data_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..current_snps {
                // This is the index into this chunk's subset of reconciled variants.
                let variant_idx_in_chunk = snp_chunk_start + i;
                // This is the index into the *global* list of all reconciled matrix rows.
                let global_matrix_row_idx = matrix_row_start_idx + variant_idx_in_chunk;

                // This is the absolute row index in the original .bed file.
                let absolute_bed_row = prep_result.required_bim_indices[global_matrix_row_idx];
                // This is the row index *relative to the start of the current I/O chunk*.
                let relative_bed_row = absolute_bed_row - chunk_bed_row_offset;

                let snp_byte_offset = relative_bed_row as u64 * bytes_per_snp;

                // The gather operation now correctly uses a relative offset within the
                // `snp_major_data` buffer provided for this chunk.
                let source_byte_indices = U64xN::splat(snp_byte_offset) + person_byte_indices;
                snp_data_vectors[i] =
                    U8xN::gather_or_default(snp_major_data, source_byte_indices.cast());
            }

            let mut dosage_vectors = [U8xN::default(); SIMD_LANES];
            for i in 0..current_snps {
                let packed_vals = snp_data_vectors[i];

                // --- SIMD Genotype Unpacking ---
                // This logic unpacks the 2-bit PLINK genotypes into dosage values (0, 1, 2)
                // or a missing sentinel (3). All operations are performed on 8-lane vectors.
                // This dosage is always for the second allele (`allele2`) in the `.bim` file record. // Corrected Comment
                // All reconciliation to handle scoring against `allele1` (the other allele in the BIM record
                // when `allele2` is the effect allele) or `allele2` (when `allele1` is the effect allele
                // as specified in the score file) is now done mathematically in the "Compiler"
                // phase in `prepare.rs` via aligned weights and correction constants.
                // The key is that the dosage unpacked here is consistently for `allele2` of the BIM pair.
                let two_bit_genotypes = (packed_vals >> bit_shifts.cast()) & U8xN::splat(0b11);

                // The formula `((g >> 1) & 1) * ((g & 1) + 1)` correctly maps the 2-bit
                // genotype `g` to a dosage, except for the missing value.
                // 0b00 -> 0, 0b10 -> 1, 0b11 -> 2. The case 0b01 is handled next.
                let one = U8xN::splat(1);
                let term1 = (two_bit_genotypes >> U8xN::splat(1)) & one; // Corrected SIMD shift
                let term2 = (two_bit_genotypes & one) + one;
                let initial_dosages = term1 * term2;

// ================== PROBE #1 START ==================
#[cfg(debug_assertions)]
{
    // This flag is declared *inside* the debug-only block, ensuring
    // it does not exist and has zero cost in release builds.
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

================ [debug] GENOTYPE UNPACKING TRACE (First Person/Variant) ================"); // Corrected Terminology
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
// =================== PROBE #1 END ===================

                // Create a mask for missing genotypes (coded as 0b01) and use it to
                // select between the calculated dosage and the missing sentinel value.
                let missing_mask = two_bit_genotypes.simd_eq(U8xN::splat(1));
                let dosages =
                    missing_mask.select(missing_sentinel_splat, initial_dosages);

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

    // Stage 1: Interleave 8-bit elements from adjacent rows.
    // This is equivalent to a vpunpckl/hbw sequence on x86, which is the
    // first step in a standard matrix transposition butterfly network.
    let (t0, t1) = m0.interleave(m1);
    let (t2, t3) = m2.interleave(m3);
    let (t4, t5) = m4.interleave(m5);
    let (t6, t7) = m6.interleave(m7);

    // Stage 2: Interleave 16-bit elements from the results of stage 1.
    // This is equivalent to a vpunpckl/hwd sequence. The cast changes the
    // granularity of the interleave operation.
    let (s0, s1) = t0.cast::<u16>().interleave(t2.cast::<u16>());
    let (s2, s3) = t1.cast::<u16>().interleave(t3.cast::<u16>());
    let (s4, s5) = t4.cast::<u16>().interleave(t6.cast::<u16>());
    let (s6, s7) = t5.cast::<u16>().interleave(t7.cast::<u16>());

    // Stage 3: Interleave 32-bit elements from the results of stage 2.
    // This is equivalent to a vpunpckl/hdq sequence.
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
