// ========================================================================================
//
//                      THE CONCURRENT PIPELINE ENGINE
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module implements the logic for the concurrent, two-stage pipeline. It contains
// the core routines for the two primary, simultaneously operating parts of the application:
// the I/O-bound Producer and the CPU-bound Consumer. The threads themselves are spawned
// and orchestrated by `main.rs`, but their behavior is defined here.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: The Decoupled Assembly Line ###
//
// The engine is built as a concurrent assembly line to resolve the conflicting
// I/O and Compute data layouts. It consists of two threads running in parallel,
// communicating safely and efficiently via message-passing channels, which provides
// both thread-safety and automatic backpressure.
//
//    - **THREAD 1 (The I/O/Pivot Producer):** This thread is I/O-bound. It reads
//      the SNP-major `.bed` file from disk, performs the Great Pivot into a
//      person-major layout, and sends the prepared buffer to the compute thread
//      over a channel.
//
//    - **THREAD 2 (The Main/Compute Consumer):** This thread is CPU-bound. It
//      receives a fully prepared, person-major buffer from the channel and
//      orchestrates the massively parallel score calculation using the `rayon`
//      thread pool and a pre-loaded, in-memory weight matrix.
//
// This model transforms the execution from a sequential `I/O -> COMPUTE -> I/O`
// process into a truly parallel system where disk reads and pure computation happen
// **at the same time**, ensuring expensive CPU cores are never left waiting for data.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. Detailed Implementation Plan ###
//
// This module is split into the two logical components of the pipeline.
//
// #### 3.1. The I/O/Pivot Producer ####
//
//    - This logic, designed to be run on a dedicated thread, is an infinite loop that
//      acts as the data producer.
//    - In each cycle, it:
//      1.  **Receives** a pre-allocated, empty pivot buffer from the **free-buffer channel**.
//          This call will block until a buffer becomes available, naturally pacing the I/O.
//      2.  Invokes the `pivot_genotype_chunk` helper to fill this buffer with the
//          next chunk of data from the `.bed` file.
//      3.  **Sends** the now-full buffer to the Main/Compute thread via the
//          **filled-buffer channel**.
//
// #### 3.2. The Compute Engine Consumer ####
//
//    - This is the entry point for the computation stage, called by `main.rs` each
//      time a new, full pivot buffer is ready. It is given two things: a buffer full
//      of pivoted genotype data, and a **reference to the complete, pre-loaded,
//      interleaved weight matrix** that resides in memory.
//
//    - **MASSIVE PARALLELISM:** The engine's logic is now beautifully simple. It
//      iterates over the individuals in the pivoted genotype buffer and uses the `rayon`
//      thread pool to process them in parallel.
//
//    - For each person, the dispatched `rayon` task has all the data it needs:
//      1.  The person's full genotype row from the pivoted buffer.
//      2.  A reference to the entire in-memory interleaved weight matrix.
//      3.  It calls the `kernel` function to calculate **all scores for that
//          person in a single, efficient pass**.
//
//    - This design eliminates all disk I/O from the computation stage, ensuring the
//      `rayon` worker threads are purely CPU-bound and maximally utilized.
//
// #### 3.3. `pivot_genotype_chunk` - The "Great Pivot" Helper ####
//
//    - This is the producer thread's data reorganization workhorse.
//
//    - **PROCESS:**
//      1. **RECEIVE BUFFER:** This function receives a mutable slice representing one of the
//         pre-allocated, shared pivot buffers. It does not perform any memory allocation itself.
//
//      2. **READ AND PIVOT (CACHE-AWARE TRANSPOSE):** It loops through the SNPs in the
//         current chunk. For each SNP, it uses a `BedReader` to fetch the genotypes
//         for the target people. A naive implementation would write this column via a
//         strided memory access pattern, which is catastrophic for cache performance.
//         Instead, this engine employs a **cache-blocked transpose**. The provided
//         buffer is conceptually divided into smaller 2D tiles (e.g., 128x128 elements).
//         The pivot operation processes the data one tile at a time, so that the
//         working set of both the source (from the `BedReader`) and destination
//         (the tile) fits within the CPU's L1/L2 caches. This transforms the slow,
//         DRAM-bound transpose into a series of hyper-fast, in-cache reorganization steps.
