// ========================================================================================
//
//               THE THROUGHPUT-BOUND INTERLEAVED-SIMD KERNEL
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module contains the pure, uncompromising computational core of the engine.
// Its sole function is to execute a sparse matrix-vector multiplication for a single
// person's pre-processed genotype data against a pre-structured weight matrix. Its
// performance is designed to be limited only by the raw arithmetic throughput of the
// CPU's Fused-Multiply-Add (FMA) units, representing the theoretical maximum speed.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: Latency Elimination, Not Hiding ###
//
// The conventional Transposed Indexed Gather kernel is an elegant solution, but
// it is fundamentally **latency-bound**. Its performance is dictated by the speed of
// the `GATHER` instruction, which fetches scattered data from memory. While clever
// software pipelining can hide some of this latency, it cannot change the underlying
// physics.
//
// This kernel operates on a different, superior philosophy. It assumes that a higher-
// level module (`batch.rs`) has paid a one-time cost to transform the weight matrix
// into an **interleaved layout**. This transformation completely **eliminates the need
// for GATHER instructions.**
//
// Instead of scattered reads, this kernel performs only perfectly **linear, streaming
// loads** from the interleaved weight matrix. A linear load is the fastest possible
// memory operation, fully saturating the CPU's memory bandwidth. By structurally
// removing the latency bottleneck, this kernel's performance becomes bound only by
// the raw throughput of the CPU's arithmetic units. This is the definition of a
// truly performant engine.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. The Uncompromising Contract ###
//
//   - **DATA LAYOUT CONTRACT:** This kernel does NOT operate on a standard weight
//     matrix. It is part of a symbiotic pair with `batch.rs`. It strictly requires
//     that the weight matrix (`w_chunk`) has been pre-processed into the
//     `M_chunk x K` interleaved format, where all K weights for a single SNP are
//     contiguous in memory.
//
//   - **NO FALLBACKS:** This kernel is implemented using explicit `std::simd`
//     intrinsics. There is no scalar fallback path.
//
//   - **`unsafe` is possible.
//
// ----------------------------------------------------------------------------------------
//
// ### 4. The Interleaved-SIMD Kernel Mechanics ###
//
// The `accumulate_scores_for_person` function will implement the following steps:
//
//   - **STEP 1: PRE-SCAN & INDEXING:**
//     - It performs a single, fast, linear scan over the input `genotype_row` slice.
//     - It creates two small `Vec<usize>` lists, `idx_g1` and `idx_g2`, containing
//       the local indices of SNPs with non-zero dosages. These lists reside in L1.
//
//   - **STEP 2: INITIALIZE SIMD ACCUMULATORS:**
//     - The function initializes an array of `ceil(K / SIMD_LANES)` independent
//       SIMD vector registers to zero. These registers will
//       accumulate all scores simultaneously.
//
//   - **STEP 3: THE SINGLE, FUSED UPDATE LOOP:**
//     - The function iterates through the small `idx_g1` list. For each `snp_idx`:
//       - It performs a tight inner loop from `i = 0` to `ceil(K / SIMD_LANES) - 1`.
//       - **Linear Load:** It performs a single, contiguous SIMD `LOAD` to fetch the
//         `i`-th chunk of 16 weights from the interleaved weight matrix for `snp_idx`.
//       - **FMA:** It performs a single SIMD `FMA` (Fused Multiply-Add) operation to
//         add these weights directly to the `i`-th accumulator register.
//     - The process is repeated for the `idx_g2` list, but the loaded weight vector
//       is first multiplied by a SIMD vector of all `2.0`.
//
//   - **STEP 4: FINAL STORE:**
//     - After the loops over the index lists are complete, the array of SIMD
//       accumulator registers holds the final, complete score contributions.
//     - A final, simple loop iterates through the accumulator registers and performs
//       a SIMD `STORE` operation to write the results into the `scores_out` slice.
//       This is a direct, high-bandwidth memory write.
