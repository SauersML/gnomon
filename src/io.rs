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

use crate::batch;
use crate::types::{
    ComputePath, PipelineError, PreparationResult, ReconciledVariantIndex, WorkItem,
};
use crossbeam_channel::Sender;
use crossbeam_queue::ArrayQueue;
use memmap2::Mmap;
use std::sync::Arc;

/// The entry point for the producer thread.
///
/// This function iterates through the list of variants required for the calculation,
/// reads their corresponding genotype data from the memory-mapped file, assesses the
/// optimal compute path (sparse vs. dense), and sends the resulting `WorkItem` down
/// the appropriate channel for consumption by the `rayon` worker pool.
///
/// # Arguments
/// * `mmap`: A shared, thread-safe handle to the memory-mapped .bed file.
/// * `prep_result`: The "computation blueprint" that dictates which variants to read.
/// * `sparse_tx`: The channel sender for variants destined for the sparse path.
/// * `dense_tx`: The channel sender for variants destined for the dense path.
/// * `buffer_pool`: A shared pool of reusable byte buffers to eliminate allocation overhead.
pub fn producer_thread(
    mmap: Arc<Mmap>,
    prep_result: Arc<PreparationResult>,
    sparse_tx: Sender<Result<WorkItem, PipelineError>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
) {
    let bytes_per_variant = prep_result.bytes_per_variant as usize;

    // We iterate over the pre-computed, sorted list of required variants.
    // The iterator's index `i` becomes the `reconciled_variant_index`, which is
    // the row index into the final conceptual weight/flip matrices.
    // `bim_row_idx` is the original row index from the .bim file, used to
    // calculate the byte offset into the .bed file.
    for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
        // 1. Acquire a buffer from the pool. This is a crucial performance optimization
        //    and backpressure mechanism. If the consumers are slow to return buffers,
        //    this call will block until one is available.
        let mut buffer = buffer_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant));

        // 2. Calculate the exact byte offset for this variant in the .bed file.
        //    The `+ 3` skips the PLINK magic number at the start of the file.
        let offset = (3 + bim_row_idx.0 as u64 * bytes_per_variant as u64) as usize;
        let end = offset + bytes_per_variant;

        // 3. Perform a critical boundary check. If this fails, the .bim metadata
        //    and the .bed file are inconsistent. We send an error down both channels
        //    to ensure a clean shutdown of the entire pipeline.
        if end > mmap.len() {
            let err = PipelineError::Io(format!(
                "Fatal: Attempted to read past the end of the .bed file for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                bim_row_idx.0
            ));
            // We don't care if the send fails, as the pipeline is already broken.
            let _ = sparse_tx.send(Err(err.clone()));
            let _ = dense_tx.send(Err(err));
            // Stop production immediately.
            break;
        }

        // 4. Read the data. This is the primary "I/O" operation, which may trigger
        //    a page fault if the data is not in the OS page cache.
        buffer.extend_from_slice(&mmap[offset..end]);

        // 5. Assess the compute path and create the final work item.
        //    The ownership of the data buffer is moved into the `WorkItem`.
        let path = batch::assess_path(&buffer, prep_result.total_people_in_fam);
        let work_item = WorkItem {
            data: buffer,
            reconciled_variant_index: ReconciledVariantIndex(i as u32),
        };

        // 6. Send the work item to the appropriate consumer. If the send fails,
        //    it means the consumer end of the channel has been dropped, so we
        //    can gracefully terminate the producer thread.
        let tx = if path == ComputePath::PersonMajor {
            &dense_tx
        } else {
            &sparse_tx
        };

        if tx.send(Ok(work_item)).is_err() {
            // Consumers have disconnected. This is not an error, but a signal to stop.
            break;
        }
    }

    // When this function returns, `sparse_tx` and `dense_tx` are dropped. This
    // closes the channels, which signals to the consumers' iterators that there
    // is no more work, allowing them to terminate cleanly.
}
