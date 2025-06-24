// ========================================================================================
//
//                       THE HIGH-PERFORMANCE DATA PRODUCER
//
// ========================================================================================
//
// ### Purpose ###
//
// This module contains the producer logic for the gnomon compute pipeline. Its sole
// responsibility is to read the required variant data from a memory-mapped .bed file
// and send it downstream to the consumer threads for processing. It leverages a
// shared buffer pool to minimize allocations and provides natural backpressure if
// consumers cannot keep up.

use crate::decide::ComputePath;
use crate::pipeline::PipelineError;
use crate::types::{PreparationResult, ReconciledVariantIndex, WorkItem};
use crossbeam_channel::Sender;
use crossbeam_queue::ArrayQueue;
use memmap2::Mmap;
use std::sync::Arc;

/// The generic entry point for the producer thread.
///
/// This function is a template that iterates through required variants and uses a
/// provided `path_decider` closure to determine the compute path. The compiler will
/// create specialized, optimized versions of this function for each type of closure
/// it is called with, enabling static dispatch and inlining for maximum performance.
///
/// # Arguments
/// * `mmap`: A shared, thread-safe handle to the memory-mapped .bed file.
/// * `prep_result`: The "computation blueprint" that dictates which variants to read.
/// * `sparse_tx`: The channel sender for variants destined for the sparse path.
/// * `dense_tx`: The channel sender for variants destined for the dense path.
/// * `buffer_pool`: A shared pool of reusable byte buffers to eliminate allocation overhead.
/// * `path_decider`: A closure that takes a variant's data and returns the `ComputePath`.
pub fn producer_thread<F>(
    mmap: Arc<Mmap>,
    prep_result: Arc<PreparationResult>,
    sparse_tx: Sender<Result<WorkItem, PipelineError>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    path_decider: F,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    let bytes_per_variant = prep_result.bytes_per_variant as usize;

    for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
        let mut buffer = buffer_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant));

        let offset = (3 + bim_row_idx.0 as u64 * bytes_per_variant as u64) as usize;
        let end = offset + bytes_per_variant;

        if end > mmap.len() {
            let err = PipelineError::Io(format!(
                "Fatal: Attempted to read past the end of the .bed file for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                bim_row_idx.0
            ));
            let _ = sparse_tx.send(Err(err.clone()));
            let _ = dense_tx.send(Err(err));
            break;
        }

        buffer.extend_from_slice(&mmap[offset..end]);

        // Use the provided generic closure to decide the path.
        // This is a direct, statically dispatched call that the compiler will inline.
        let path = path_decider(&buffer);

        let work_item = WorkItem {
            data: buffer,
            reconciled_variant_index: ReconciledVariantIndex(i as u32),
        };

        let tx = if path == ComputePath::Pivot {
            &dense_tx
        } else {
            &sparse_tx
        };

        if tx.send(Ok(work_item)).is_err() {
            // Consumers have disconnected. This is not an error, but a signal to stop.
            break;
        }
    }
}
