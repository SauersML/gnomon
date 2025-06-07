// ========================================================================================
//
//               THE MEMORY HIERARCHY CHOREOGRAPHER
//
// ========================================================================================
//
// ### 1. High-Level Purpose ###
//
// This module implements the Block-Pivoting Matrix-Chunk Engine for the
// N > 1 "biobank-scale" use case. Its sole responsibility is to orchestrate a
// massively parallel computation by acting as the master choreographer of data
// movement between disk, main memory (DRAM), and the CPU caches.
//
// ----------------------------------------------------------------------------------------
//
// ### 2. The Architectural Philosophy: The "Assembly Line" ###
//
// The engine is built to resolve the fundamental, diametrically opposed
// requirements for optimal performance in this problem:
//
//   - I/O OPTIMALITY: The input data (.bed file) is stored SNP-major. The only
//     physically efficient way to read it is sequentially, in SNP-major order.
//
//   - COMPUTE OPTIMALITY: The computational kernel is fastest on person-major data,
//     where all data for one person is contiguous, enabling perfect cache locality
//     and register usage.
//
// The architecture achieves both optima by introducing a "Great Pivot" stage.
// It is a factory assembly line:
//
//   - STAGE 1 (The Staging Area): We perform an efficient, I/O-optimal read of a
//     large chunk of SNP-major data from disk. We then pay a deliberate, one-time
//     cost to "pivot" this chunk in-memory into a perfectly structured,
//     person-major layout.
//
//   - STAGE 2 (The Assembly Line): We unleash an army of parallel workers (CPU cores)
//     on this perfectly prepared data. Each worker operates on their assigned set
//     of people with maximum efficiency, requiring zero communication or contention.
//
// This architecture makes a strategic trade: it invests in a highly-optimized,
// memory-intensive data reorganization stage to enable a subsequent compute stage
// that is embarrassingly parallel and architecturally perfect.
//
// ----------------------------------------------------------------------------------------
//
// ### 3. Detailed Implementation Plan ###
//
// The engine is orchestrated by the single public function `calculate_for_batch`.
//
// #### 3.1. `calculate_for_batch` - The Factory Manager ####
//
//   - **INPUT:** A shared `PgsContext`, a list of `target_people`, and the `.bed` path.
//
//   - **STEP 1: PREPARATION:**
//     - Resolves `target_people` to a sorted `Vec<usize>` of their original .fam indices.
//     - Allocates the final `p_final` results matrix (`N_target x K`), which will be
//       updated in place by the parallel compute stage.
//
//   - **STEP 2: THE OUTER SNP CHUNKING LOOP:**
//     - This function iterates serially over the master SNP list in large chunks
//       (e.g., `SNP_CHUNK_SIZE = 16384`). This loop's primary purpose is to bound
//       the peak memory usage of the engine, ensuring it can process genomes of
//       arbitrary size with a fixed memory footprint for the intermediate pivot buffer.
//
//   - **STEP 3: DISPATCHING TO HELPERS:**
//     - Inside the loop, it calls `pivot_genotype_chunk` to perform the pivot.
//     - It then passes the resulting person-major data block to `compute_on_chunk`,
//       along with a mutable slice of `p_final` for accumulation.
//
// #### 3.2. `pivot_genotype_chunk` - The "Great Pivot" Helper ####
//
//   - This is the engine's data reorganization workhorse.
//
//   - **PROCESS:**
//     1. **ALLOCATE DENSE PIVOT BUFFER:** It allocates a temporary, dense, in-memory
//        `g_chunk_person_major` buffer (`N_target x SNP_CHUNK_SIZE`). It is critical
//        that this is a single, flat, contiguous `Vec<u8>`.
//
//     2. **READ AND PIVOT (CACHE-AWARE TRANSPOSE):** It loops through the SNPs in the
//        current chunk. For each SNP, it uses the `BedReader` to fetch the genotypes
//        for the target people. A naive implementation would write this column via
//        a strided memory access pattern, which is catastrophic for cache performance.
//        Instead, this engine employs a **cache-blocked transpose**. The
//        `g_chunk_person_major` buffer is conceptually divided into smaller 2D tiles
//        (e.g., 128x128 elements). The pivot operation processes the data one tile
//        at a time, ensuring that the working set of both the source (from the
//        `BedReader`) and destination (the tile) fits within the CPU's L1/L2 caches.
//        This transforms the slow, DRAM-bound transpose into a series of hyper-fast,
//        in-cache reorganization steps.
//
//   - **JUSTIFICATION OF DENSE PIVOT (Rebuttal to Sparse Pivot):**
//     A sparse pivot (`Vec<Vec<(u16, u8)>>`) is architecturally inferior. It replaces
//     a single, simple, contiguous memory block with a fragmented "jagged array"
//     of pointers. Furthermore, populating it requires millions of thread-unsafe
//     `push` operations, leading to catastrophic lock contention or complex, slow
//     merging logic. The dense pivot creates a computationally regular, predictable,
//     and parallel-friendly data structure. The "wasted" space on zero-genotypes
//     is a small price to pay for this architectural integrity.
//
// #### 3.3. `compute_on_chunk` - The "Assembly Line" Helper ####
//
//   - This is the parallel computation engine.
//
//   - **PROCESS (THE PARALLELISM MODEL):**
//     - It uses `rayon::par_iter_mut()` to parallelize over the rows of the `p_final`
//       accumulator slice. `rayon` automatically chunks the work, assigning each
//       thread a unique, non-overlapping slice of people.
//     - Each thread receives its mutable slice of the accumulator and calculates
//       the pointer to its corresponding contiguous row(s) in the read-only
//       `g_chunk_person_major` buffer.
//     - **THE KERNEL CALL:** It makes a single call to the hyper-optimized
//       `kernel::accumulate_scores_for_person` function, passing it the data it needs.
//       Because each thread has its own input and output slices, there is ZERO
//       inter-thread communication or shared-resource contention, achieving perfect
//       scalability for the compute stage.
