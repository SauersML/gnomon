// ========================================================================================
//
//                      THE PIPELINE ENGINE
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module implements the logic for a concurrent, two-stage pipeline. It contains
// the core routines for the two distinct, simultaneously operating parts of the
// application: the I/O-bound Producer and the CPU-bound Consumer. The pipeline is
// orchestrated by `main.rs`, but the behavior of the threads is defined here.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: The Asymmetrical Pipeline ###
//
// The engine is designed as an asymmetrical, producer-consumer pipeline where each
// stage has a specialized, non-competing role. This resolves the conflicting data
// layouts (SNP-major on disk vs. person-major for compute) by dedicating separate
// system resources to I/O/transformation and to parallel computation, preventing
// thread pool contention and maximizing throughput.
//
//   - **THREAD 1 (The Dedicated I/O & Pivot Producer):** This thread's sole purpose
//     is to read data from disk and transform it. It is a **single-threaded** worker
//     that performs a highly efficient **linear read** of a large, SNP-major chunk
//     of the genotype data. It then performs the "Great Pivot" into a person-major
//     layout within its own thread. This operation is bound by I/O or memory
//     bandwidth, not the CPU, and therefore does **NOT** use `rayon`.
//
//   - **THREAD 2 (The Parallel Compute Consumer):** This is the main thread and the
//     application's computational heavyweight. It is the **exclusive user of the
//     global `rayon` thread pool**. It receives a fully prepared, person-major
//     buffer from the producer and immediately orchestrates a massively parallel
//     score calculation, saturating all available CPU cores with productive work.
//
// This model creates a clean separation of concerns. The I/O thread focuses on
// feeding the pipeline, while the main thread's `rayon` pool focuses exclusively on
// computation. There is no resource conflict, no context switching overhead from
// competing thread pools, and the compute cores are never starved for data.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. Detailed Implementation Plan ###
//
// This module is split into the two logical components of the pipeline.
//
// #### 3.1. The I/O & Pivot Producer ####
//
//   - This logic, designed to be run on a dedicated thread spawned by `main.rs`,
//     is an infinite loop that acts as the data producer.
//   - In each cycle, it:
//     1. **Receives** a pre-allocated, empty pivot buffer from the **free-buffer channel**.
//        This call blocks until a buffer is available, providing backpressure.
//     2. Invokes the `pivot_genotype_chunk` helper. This is a single-threaded,
//        blocking call that fills the buffer.
//     3. **Sends** the now-full buffer to the Main/Compute thread via the
//        **filled-buffer channel**.
//
// #### 3.2. The Compute Engine Consumer ####
//
//   - This is the entry point for the computation stage, called by `main.rs` each
//     time a new, full pivot buffer is ready. It is given a buffer full of pivoted
//     genotype data and a reference to the complete, pre-loaded, interleaved
//     weight matrix.
//
//   - **MASSIVE PARALLELISM:** The engine's logic iterates over the individuals
//     in the pivoted genotype buffer and uses the `rayon` thread pool to process them
//     in parallel. The producer thread is dormant or preparing the next chunk on a
//     separate core during this time, not competing for these `rayon` workers.
//
//   - For each person, the dispatched `rayon` task has all the data it needs:
//     1. The person's full genotype row from the pivoted buffer.
//     2. A reference to the entire in-memory interleaved weight matrix.
//     3. It calls the `kernel` function to calculate **all scores for that
//        person in a single, efficient pass**.
//
//   - This design ensures the `rayon` worker threads are purely CPU-bound and are
//     fed by a producer that is fast enough to prevent them from ever going idle.
//
// #### 3.3. `pivot_genotype_chunk` - The Single-Threaded Pivot Helper ####
//
//   - This is the producer's data reorganization workhorse. It is a fast,
//     **single-threaded** function that transposes a chunk of data.
//
//   - **PROCESS:**
//     1. **RECEIVE BUFFER:** This function receives a mutable slice representing one of the
//        pre-allocated, shared pivot buffers.
//
//     2. **SEQUENTIAL READ AND PIVOT:** The function performs the transpose operation
//        in the most efficient way possible for a single thread.
//
//        **INSIDE THE FUNCTION:**
//        - The function will read a **single, contiguous, SNP-major block** of
//          genotype data from the source `.bed` file reader into a temporary,
//          on-stack buffer. This is a crucial linear read.
//        - It will then loop through this temporary buffer SNP-by-SNP.
//        - For each SNP, it will execute a tight inner loop over all individuals.
//        - In the inner loop, it unpacks the genotype value for the current
//          individual and writes it to the correct `[person_index][snp_index]`
//          position in the large destination `pivot_buffer`.
//
//        This is so the read from the source is perfectly sequential and the pivot
//        is a fast, in-memory transformation, ideally remaining within the CPU caches.
