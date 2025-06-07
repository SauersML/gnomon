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
// ### 2. The Architectural Philosophy: The Pipeline ###
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

use crate::kernel::{self, KernelInput, SimdVec, ValidationError};
use rayon::prelude::*;
use std::cell::RefCell;
use std::io;
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender};
use thread_local::ThreadLocal;

// ========================================================================================
//                            PUBLIC API & TYPE DEFINITIONS
// ========================================================================================

// --- Trait for Decoupling I/O ---

/// Defines the contract for a reader that generates sparse genotype events.
pub trait BedReader {
    /// Scans the next SNP and populates a pre-allocated buffer with sparse
    /// events of the form `(destination_index, dosage)`.
    ///
    /// This is the core of the abstraction. The reader performs
    /// the bit-unpacking and calculates the final destination index in the
    /// person-major pivot buffer. The caller's job is a simple,
    /// unconditional memory write.
    ///
    /// # Arguments
    /// * `events_buffer`: A mutable, reusable buffer to be cleared and then filled.
    /// * `snp_idx_in_chunk`: The local index (0, 1, 2...) of the current SNP.
    /// * `num_snps_in_chunk`: The total number of SNPs in a full chunk.
    /// * `num_people`: The total number of people in the study.
    ///
    /// # Returns
    /// - `Ok(true)` if a SNP was read and events were generated.
    /// - `Ok(false)` if EOF was reached and no more data is available.
    /// - `Err(e)` if an I/O error occurred.
    fn next_snp_events(
        &mut self,
        events_buffer: &mut Vec<(usize, u8)>,
        snp_idx_in_chunk: usize,
        num_snps_in_chunk: usize,
        num_people: usize,
    ) -> io::Result<bool>;
}

// --- Typestate Definitions ---

/// A marker type (ZST) indicating a buffer is available for the producer.
#[derive(Debug, Clone, Copy)]
pub struct ReadyForFilling;

/// A marker type (ZST) indicating a buffer is ready for the consumer.
#[derive(Debug, Clone, Copy)]
pub struct ReadyForProcessing;

/// A type-safe buffer that tracks its state at compile time.
pub struct PipelineBuffer<State> {
    buffer: Vec<u8>,
    snps_pivoted: usize,
    _state: PhantomData<State>,
}

// --- State-Independent Methods ---
impl<State> PipelineBuffer<State> {
    /// Creates a new, zero-initialized buffer, ready for the producer.
    /// This is called once per buffer at application startup.
    pub fn new(person_major_capacity: usize) -> PipelineBuffer<ReadyForFilling> {
        PipelineBuffer {
            buffer: vec![0; person_major_capacity],
            snps_pivoted: 0,
            _state: PhantomData,
        }
    }
}

// --- Producer-Only Methods ---
impl PipelineBuffer<ReadyForFilling> {
    /// Consumes an empty buffer, fills it with sparse events from the `BedReader`,
    /// and returns a new buffer typed as being ready for processing.
    pub fn pivot_and_fill<R: BedReader>(
        mut self,
        bed_reader: &mut R,
        events_buffer: &mut Vec<(usize, u8)>,
        num_snps_to_read: usize,
        num_people: usize,
    ) -> io::Result<PipelineBuffer<ReadyForProcessing>> {
        let snps_actually_pivoted = pivot_from_sparse_events(
            bed_reader,
            &mut self.buffer,
            events_buffer,
            num_snps_to_read,
            num_people,
        )?;

        Ok(PipelineBuffer {
            buffer: self.buffer,
            snps_pivoted: snps_actually_pivoted,
            _state: PhantomData,
        })
    }
}

// --- Consumer-Only Methods ---
impl PipelineBuffer<ReadyForProcessing> {
    /// Returns a correctly-sized slice of the valid, pivoted genotype data.
    pub fn as_pivoted_data(&self, num_people: usize) -> &[u8] {
        &self.buffer[..num_people * self.snps_pivoted]
    }

    /// Returns the number of valid SNPs in this chunk.
    pub fn snps_in_chunk(&self) -> usize {
        self.snps_pivoted
    }

    /// Consumes the processed buffer and returns it to the "empty" state.
    pub fn release_for_reuse(self) -> PipelineBuffer<ReadyForFilling> {
        PipelineBuffer {
            buffer: self.buffer,
            snps_pivoted: 0,
            _state: PhantomData,
        }
    }
}

// ========================================================================================
//                             PIPELINE TASK IMPLEMENTATIONS
// ========================================================================================

/// The logic for the dedicated I/O and pivot producer thread. This loop is allocation-free.
pub fn producer_task<R: BedReader>(
    mut bed_reader: R,
    free_buffers_rx: Receiver<PipelineBuffer<ReadyForFilling>>,
    filled_buffers_tx: Sender<PipelineBuffer<ReadyForProcessing>>,
    num_snps_per_chunk: usize,
    num_people: usize,
) {
    // This temporary buffer is created ONCE and reused for every chunk.
    let mut events_buffer = Vec::with_capacity(num_people);

    while let Ok(empty_buffer) = free_buffers_rx.recv() {
        match empty_buffer.pivot_and_fill(
            &mut bed_reader,
            &mut events_buffer,
            num_snps_per_chunk,
            num_people,
        ) {
            Ok(filled_buffer) => {
                if filled_buffer.snps_in_chunk() == 0 {
                    break; // EOF
                }
                if filled_buffers_tx.send(filled_buffer).is_err() {
                    break; // Consumer has hung up.
                }
            }
            Err(e) => {
                eprintln!("Fatal I/O error in producer thread: {}", e);
                break;
            }
        }
    }
}

/// The main consumer engine. Takes a chunk of pivoted data and uses the Rayon
/// thread pool to calculate scores for all individuals in parallel.
pub fn process_pivoted_chunk(
    pivoted_data: &[u8],
    interleaved_weights: &[f32],
    all_scores_out: &mut [f32],
    num_people_in_chunk: usize,
    num_snps_in_chunk: usize,
    num_scores: usize,
    person_chunk_offset: usize,
) {
    let thread_data_pool: ThreadLocal<RefCell<(Vec<SimdVec>, Vec<usize>, Vec<usize>)>> =
        ThreadLocal::new();

    pivoted_data
        .par_chunks_exact(num_snps_in_chunk)
        .enumerate()
        .for_each(|(person_idx_in_chunk, genotype_row)| {
            let thread_data = thread_data_pool.get_or(|| {
                // This closure runs ONCE per thread, performing the only allocations.
                let required_lanes = (num_scores + kernel::LANE_COUNT - 1) / kernel::LANE_COUNT;
                let acc_buffer = vec![SimdVec::splat(0.0); required_lanes];
                let idx_g1_buffer = Vec::with_capacity(num_snps_in_chunk / 4);
                let idx_g2_buffer = Vec::with_capacity(num_snps_in_chunk / 16);
                RefCell::new((acc_buffer, idx_g1_buffer, idx_g2_buffer))
            });

            let mut data = thread_data.borrow_mut();
            let (acc_buffer, idx_g1_buffer, idx_g2_buffer) = &mut *data;

            idx_g1_buffer.clear();
            idx_g2_buffer.clear();

            let score_start_idx = (person_chunk_offset + person_idx_in_chunk) * num_scores;
            let scores_out = &mut all_scores_out[score_start_idx..score_start_idx + num_scores];

            let kernel_input = KernelInput::new(
                genotype_row,
                interleaved_weights,
                scores_out,
                acc_buffer,
                idx_g1_buffer,
                idx_g2_buffer,
                num_snps_in_chunk,
                num_scores,
            );

            match kernel_input {
                Ok(input) => kernel::accumulate_scores_for_person(input),
                Err(e) => {
                    // This panic is justified because a validation error here implies a
                    // catastrophic logic bug within this module, not invalid user input.
                    panic!(
                        "Fatal logic error: KernelInput construction failed: {:?}. \
                         This is a bug in `batch.rs`. Dims were: \
                         num_snps_in_chunk={}, num_scores={}",
                        e, num_snps_in_chunk, num_scores
                    );
                }
            }
        });
}

// ========================================================================================
//                              PRIVATE HELPER FUNCTIONS
// ========================================================================================

/// The workhorse of the producer. It populates the person-major `pivot_buffer`
/// by consuming sparse events from the `BedReader`.
fn pivot_from_sparse_events<R: BedReader>(
    bed_reader: &mut R,
    pivot_buffer: &mut [u8],
    events_buffer: &mut Vec<(usize, u8)>,
    num_snps_to_read: usize,
    num_people: usize,
) -> io::Result<usize> {
    for snp_idx in 0..num_snps_to_read {
        // Reset the event buffer's length but keep its capacity.
        events_buffer.clear();

        // Ask the reader to generate events for the next SNP.
        let has_data = bed_reader.next_snp_events(
            events_buffer,
            snp_idx,
            num_snps_to_read,
            num_people,
        )?;

        if has_data {
            // This is the tight, branchless hot loop. We just write the data.
            for &(dest_idx, dosage) in &*events_buffer {
                pivot_buffer[dest_idx] = dosage;
            }
        } else {
            // End of file. Return the number of SNPs we fully processed.
            return Ok(snp_idx);
        }
    }

    // Completed the full chunk.
    Ok(num_snps_to_read)
}


// TODO: MAJOR PERFORMANCE FLAW! This `ThreadLocal` is re-allocated for every single
// chunk processed by the consumer. It needs to be initialized only ONCE for the
// lifetime of the application. The best fix is to hoist this into `main.rs`, create it
// before the consumer loop, and pass it down by reference, making the dependency explicit.
