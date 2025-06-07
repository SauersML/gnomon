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

use crate::kernel::{self, KernelInput, SimdVec};
use rayon::prelude::*;
use std::cell::RefCell;
use std::io;
use std::marker::PhantomData;
use std::sync::mpsc::{Receiver, Sender};
use thread_local::ThreadLocal;

// ========================================================================================
//                            PUBLIC API & TYPE DEFINITIONS
// ========================================================================================

/// A type-safe wrapper for a reconciled dosage value (0, 1, or 2).
/// Its internal state is private, and it can only be constructed via a
/// fallible constructor, making it impossible to represent an invalid dosage.
#[derive(Debug, Clone, Copy)]
pub struct EffectAlleleDosage(u8);

impl EffectAlleleDosage {
    /// Creates a new `EffectAlleleDosage` only if the value is valid (0, 1, or 2).
    /// This is the sole entry point for creating this type.
    pub fn new(value: u8) -> Option<Self> {
        if value <= 2 {
            Some(Self(value))
        } else {
            None
        }
    }

    /// Returns the wrapped dosage value.
    pub fn value(&self) -> u8 {
        self.0
    }
}

/// A pool of reusable, thread-local buffers for the compute kernel.
/// This is initialized once in `main.rs` and passed by reference to the consumer.
pub type KernelDataPool = ThreadLocal<RefCell<(Vec<SimdVec>, Vec<usize>, Vec<usize>)>>;

// --- Trait for Decoupling I/O ---

/// Defines the contract for a reader that generates sparse genotype events.
pub trait BedReader {
    /// Scans the next SNP and populates a pre-allocated buffer with sparse
    /// events of the form `(destination_index, dosage)`. The produced dosage
    /// must be wrapped in the `EffectAlleleDosage` type, guaranteeing its validity.
    fn next_snp_events(
        &mut self,
        events_buffer: &mut Vec<(usize, EffectAlleleDosage)>,
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
    pub fn pivot_and_fill<R: BedReader>(
        mut self,
        bed_reader: &mut R,
        events_buffer: &mut Vec<(usize, EffectAlleleDosage)>,
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
    pub fn as_pivoted_data(&self, num_people: usize) -> &[u8] {
        &self.buffer[..num_people * self.snps_pivoted]
    }

    pub fn snps_in_chunk(&self) -> usize {
        self.snps_pivoted
    }

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
    let mut events_buffer: Vec<(usize, EffectAlleleDosage)> = Vec::with_capacity(num_people);

    while let Ok(empty_buffer) = free_buffers_rx.recv() {
        match empty_buffer.pivot_and_fill(
            &mut bed_reader,
            &mut events_buffer,
            num_snps_per_chunk,
            num_people,
        ) {
            Ok(filled_buffer) => {
                // The simplified and robust termination protocol.
                // If the I/O layer returned no SNPs, it's the end of the file.
                // We immediately break the loop. The `Sender` is dropped when this
                // function scope ends, which is the canonical signal to the
                // consumer that the channel is closed.
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
/// thread pool to calculate scores for all individuals in parallel. This function is
/// stateless; it depends on the caller to provide a pool of reusable buffers.
pub fn process_pivoted_chunk(
    pivoted_data: &[u8],
    interleaved_weights: &[f32],
    all_scores_out: &mut [f32],
    num_people_in_chunk: usize,
    num_snps_in_chunk: usize,
    num_scores: usize,
    person_chunk_offset: usize,
    thread_data_pool: &KernelDataPool,
) {
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
    events_buffer: &mut Vec<(usize, EffectAlleleDosage)>,
    num_snps_to_read: usize,
    num_people: usize,
) -> io::Result<usize> {
    for snp_idx in 0..num_snps_to_read {
        events_buffer.clear();
        let has_data = bed_reader.next_snp_events(
            events_buffer,
            snp_idx,
            num_snps_to_read,
            num_people,
        )?;

        if has_data {
            for &(dest_idx, dosage) in &*events_buffer {
                // Unwrap the type-safe dosage to get the raw u8 for the buffer.
                pivot_buffer[dest_idx] = dosage.value();
            }
        } else {
            // End of file. Return the number of SNPs we fully processed.
            return Ok(snp_idx);
        }
    }
    // Completed the full chunk.
    Ok(num_snps_to_read)
}
