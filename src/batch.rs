// ========================================================================================
//
//                      THE PARALLEL PIPELINE ENGINE
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module implements the logic for a concurrent, two-stage pipeline. It contains
// the core routines for the two primary, simultaneously operating parts of the application:
// the parallel Producer and the parallel Consumer. Both stages are CPU-bound and are
// designed to share the global `rayon` thread pool to maximize throughput. The threads
// themselves are spawned and orchestrated by `main.rs`, but their behavior is defined here.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: The Balanced Parallel Assembly Line ###
//
// The engine is a concurrent assembly line where both major stages are parallelized.
// This resolves the conflicting data layouts (SNP-major on disk vs. person-major for
// compute) by transforming the pivot operation itself from a single-threaded bottleneck
// into a high-throughput, parallel task.
//
//   - **THREAD 1 (The Parallel Pivot Producer):** This thread's primary work is a
//     CPU-bound data transformation. It reads genotype data and immediately uses the
//     global `rayon` thread pool to parallelize the "Great Pivot" into the required
//     person-major layout. It then sends the prepared buffer to the consumer.
//
//   - **THREAD 2 (The Main/Compute Consumer):** This thread is also CPU-bound. It
//     receives a fully prepared, person-major buffer and orchestrates the massively
//     parallel score calculation using the same `rayon` thread pool.
//
// This model creates a fully parallel pipeline where the CPU-intensive data preparation
// (pivot) and the CPU-intensive score calculation happen concurrently. By having both
// stages draw from the same `rayon` pool, the system dynamically balances resources,
// so that that CPU cores are always busy with productive work, either pivoting the
// next chunk or calculating scores for the current one.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. Detailed Implementation Plan ###
//
// This module is split into the two logical components of the pipeline.
//
// #### 3.1. The Parallel Pivot Producer ####
//
//   - This logic, designed to be run on a dedicated thread, is an infinite loop that
//     acts as the data producer.
//   - In each cycle, it:
//     1. **Receives** a pre-allocated, empty pivot buffer from the **free-buffer channel**.
//        This call will block until a buffer becomes available, providing backpressure.
//     2. Invokes the `pivot_genotype_chunk` helper. This is no longer a simple helper
//        call; it's the kickoff for a major parallel computation that will saturate
//        the `rayon` pool to prepare the buffer.
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
//     in parallel.
//
//   - For each person, the dispatched `rayon` task has all the data it needs:
//     1. The person's full genotype row from the pivoted buffer.
//     2. A reference to the entire in-memory interleaved weight matrix.
//     3. It calls the `kernel` function to calculate **all scores for that
//        person in a single, efficient pass**.
//
//   - This design ensures the `rayon` worker threads are purely CPU-bound, fed by a
//     producer that is fast enough to prevent them from idling.
//
// #### 3.3. `pivot_genotype_chunk` - The Parallel "Great Pivot" Helper ####
//
//   - This is the producer's data reorganization workhorse, fully parallelized.
//
//   - **PROCESS:**
//     1. **RECEIVE BUFFER:** This function receives a mutable slice representing one of the
//        pre-allocated, shared pivot buffers.
//
//     2. **PARALLEL READ AND PIVOT:** This function parallelizes the transpose operation
//        using `rayon`. It partitions the destination person-major buffer into horizontal
//        "bands" (i.e., chunks of people) and assigns a `rayon` task to fill each band.
//
//        The implementation will look conceptually like this:
//        - Import `rayon::prelude::*`.
//        - The function will take the destination buffer (`pivot_buffer`) and divide
//          it into parallel, mutable chunks using `par_chunks_mut()`. Each chunk will
//          represent the full row of data for one person.
//        - A `.enumerate()` call will provide the index of the person for each chunk.
//        - A `.for_each()` parallel loop will then execute for each person (or small
//          group of people).
//
//        **INSIDE EACH `RAYON` TASK:**
//        - The task is responsible for filling its assigned row(s) in the destination buffer.
//        - To do this, it will loop sequentially through all the SNPs for the current data chunk.
//        - In each iteration of its loop, it will read the single genotype value for its
//          assigned person and the current SNP from the source `.bed` file reader.
//        - It will then unpack and write that value into the correct column of its own row.
