// ========================================================================================
//
//                      THE LATENCY-HIDING SIMD KERNEL
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module is the pure computational heart of the engine. It contains a single,
// hyper-optimized function whose only purpose is to calculate the contribution of a
// chunk of SNPs for a single person and add it to their running score totals, at the
// absolute physical speed limit of the CPU's SIMD units.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: Defeating Latency ###
//
// The computational core is the Transposed Indexed Gather kernel. A naive
// implementation of this kernel would be **latency-bound**. A `GATHER` instruction,
// which fetches scattered data from memory, takes many clock cycles to complete. A
// simple loop would stall on each iteration, waiting for the previous `GATHER` to
// finish.
//
// This implementation defeats this bottleneck using **software pipelining**. By maintaining
// multiple independent SIMD accumulator registers, we break the dependency chain. This
// allows the CPU's out-of-order execution engine to issue multiple `GATHER` instructions
// simultaneously, effectively hiding the latency of each individual operation. The
// kernel's performance becomes bound by the **throughput** of the CPU's memory and
// arithmetic units, not the latency of a single instruction. This is a non-negotiable
// requirement for maximum performance.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. The Uncompromising Contract ###
//
//   - **NO FALLBACKS:** we use Rust intrinsic SIMD.
//
//   - **`unsafe` is possible.
//
// ----------------------------------------------------------------------------------------
//
// ### 4. The Hardened TIG Kernel Mechanics ###
//
// The `accumulate_scores_for_person` function will implement the following steps:
//
//   - **STEP 1: PRE-SCAN & INDEXING:**
//     - It performs a single, fast, linear scan over the input `genotype_row` slice.
//     - It creates two small `Vec<usize>` lists, `idx_g1` and `idx_g2`, containing
//       the local indices of SNPs with non-zero dosages. This pre-computation
//       eliminates all arithmetic on zero-dosage genotypes. These small lists will
//       reside in the L1 cache for the duration of the function.
//
//   - **STEP 2: THE MAIN SCORE LOOP (Outer Loop):**
//     - The function loops `K` times, once for each score.
//
//   - **STEP 3: THE SOFTWARE-PIPELINED GATHER LOOP (Inner Loop):**
//     - For each score, it processes the `idx_g1` and `idx_g2` lists.
//     - **Initialize Accumulators:** It initializes multiple (e.g., 4) independent
//       SIMD vector registers to zero (e.g., `sum_vec_0`, `sum_vec_1`, ...).
//     - **Main Vectorized Loop:** It iterates through the index list in large chunks
//       (e.g., `LANES * 4`). In each iteration, it performs four independent
//       `GATHER` operations, loading data into four temporary SIMD registers. It
//       then performs four independent `FMA` (Fused Multiply-Add) operations to
//       update the four accumulator registers.
//     - **Reduction:** After the loop, the accumulator registers are summed
//       together into a single vector result.
//     - **Remainder Handling:** A small, scalar loop handles any remaining indices
//       that did not fit into a full chunk.
//
//   - **STEP 4: FINAL ACCUMULATION:**
//     - The final scalar sum for the current score `k` is added to the corresponding
//       element in the `scores_out` slice.
