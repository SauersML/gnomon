// ========================================================================================
//
//                      THE CONCURRENT PIPELINE ENGINE
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module implements the concurrent, two-stage pipeline for the biobank-scale
// use case. It contains the logic for the two primary, simultaneously operating parts
// of the application: the I/O-bound Producer and the CPU-bound Consumer.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: The True Assembly Line ###
//
// The engine is built as a concurrent assembly line to resolve the conflicting
// I/O and Compute data layouts. It consists of two threads running in parallel,
// communicating via shared data buffers:
//
//   - **THREAD 1 (The I/O/Pivot Producer):** This thread is I/O-bound. It reads
//     the SNP-major `.bed` file from disk, performs the Great Pivot into a
//     person-major layout, and hands the prepared buffer to the compute thread.
//
//   - **THREAD 2 (The Main/Compute Consumer):** This thread is CPU-bound. It takes a
//     fully prepared, person-major buffer and orchestrates the massively parallel
//     score calculation using the `rayon` thread pool.
//
// This model transforms the execution from a sequential `I/O -> COMPUTE -> I/O`
// process into a truly parallel system where disk reads and pure computation happen
// **at the same time**, so expensive CPU cores are never left waiting for data.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. Detailed Implementation Plan ###
//
// This module is split into the two logical components of the pipeline.
//
// #### 3.1. The I/O/Pivot Producer ####
//
//   - This logic, designed to be run on a dedicated thread spawned by `main.rs`, is
//     an infinite loop that acts as the data producer.
//   - In each cycle, it:
//     1.  Waits to receive a pointer to an empty, re-usable pivot buffer.
//     2.  Invokes the `pivot_genotype_chunk` helper to fill this buffer with the
//         next chunk of data from the `.bed` file.
//     3.  Sends the pointer to the now-full buffer to the Main/Compute thread.
//
// #### 3.2. The Compute Engine Consumer ####
//
//   - This is the entry point for the computation stage, called by `main.rs` each
//     time a new, full pivot buffer is ready.
//   - **NESTED LOOPING:** To handle weight matrices that are too large for
//     memory, this engine uses a nested loop. For a given chunk of pivoted genotypes
//     (which is held in memory), it iterates through the weight matrix **in chunks**.
//   - The process for one genotype buffer is:
//     1.  **Outer Loop:** For each `weight_chunk` needed for the scores:
//     2.      - Call into `io.rs` to load just this slice of the weight matrix from disk
//               into a small, temporary, interleaved buffer.
//     3.      - **Inner Parallelism (The Assembly Line):** Unleash the `rayon` thread pool on the
//               target individuals. Each thread is given the relevant genotype data and
//               the *current chunk* of interleaved weights and calls the `kernel`
//               function to accumulate scores.
//
// #### 3.3. `pivot_genotype_chunk` - The "Great Pivot" Helper ####
//
//   - This is the engine's data reorganization workhorse, called by the I/O/Pivot Producer thread.
//
//   - **PROCESS:**
//     1. **RECEIVE BUFFER:** This function receives a mutable slice representing one of the
pre-allocated, shared pivot buffers. It does not perform any memory allocation itself.
//
//     2. **READ AND PIVOT (CACHE-AWARE TRANSPOSE):** It loops through the SNPs in the current chunk.
For each SNP, it uses the `BedReader` to fetch the genotypes for the target people. A naive implementation
would write this column via a strided memory access pattern, which is catastrophic for cache performance.
Instead, this engine employs a **cache-blocked transpose**. The provided buffer is conceptually divided
into smaller 2D tiles (e.g., 128x128 elements). The pivot operation processes the data one tile at a time,
so that the working set of both the source (from the `BedReader`) and destination (the tile) fits within the
CPU's L1/L2 caches. This transforms the slow, DRAM-bound transpose into a series of hyper-fast, in-cache reorganization steps.
