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
use crate::types::{FilesetBoundary, PreparationResult, ReconciledVariantIndex, WorkItem};
use crossbeam_channel::Sender;
use crossbeam_queue::ArrayQueue;
use memmap2::Mmap;
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// The number of variants to process locally before updating the global atomic counter.
/// A power of 2 is often efficient.
const PROGRESS_UPDATE_BATCH_SIZE: u64 = 1024;

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
    variants_processed_count: Arc<AtomicU64>,
    path_decider: F,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    let bytes_per_variant = prep_result.bytes_per_variant as usize;
    let mut local_variants_processed: u64 = 0;

    for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
        let mut buffer = buffer_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant));

        // The bim_row_idx is now a u64, so the cast is no longer needed.
        let offset = (3 + bim_row_idx.0 * bytes_per_variant as u64) as usize;
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

        local_variants_processed += 1;
        if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
            variants_processed_count.fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
            local_variants_processed = 0;
        }
    }

    if local_variants_processed > 0 {
        variants_processed_count.fetch_add(local_variants_processed, Ordering::Relaxed);
    }
}

/// The producer for the multi-file pipeline. It seamlessly switches between memory-mapped
/// files as it iterates through the globally-indexed list of required variants.
pub fn multi_file_producer_thread<F>(
    prep_result: Arc<PreparationResult>,
    boundaries: &[FilesetBoundary],
    sparse_tx: Sender<Result<WorkItem, PipelineError>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<ArrayQueue<Vec<u8>>>,
    variants_processed_count: Arc<AtomicU64>,
    path_decider: F,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    // --- Initial State ---
    let mut current_fileset_idx: usize = 0;
    let bytes_per_variant = prep_result.bytes_per_variant;
    let mut local_variants_processed: u64 = 0;

    // Open and mmap the first file.
    let mut mmap = match File::open(&boundaries[0].bed_path)
        .map_err(|e| PipelineError::Io(e.to_string()))
        .and_then(|file| unsafe { Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string())) })
    {
        Ok(map) => map,
        Err(e) => {
            let _ = sparse_tx.send(Err(e.clone()));
            let _ = dense_tx.send(Err(e));
            return;
        }
    };

    // Determine the start index of the *next* file boundary.
    let mut next_boundary_start_idx = if boundaries.len() > 1 {
        boundaries[1].starting_global_index
    } else {
        u64::MAX // No next boundary.
    };

    // --- Main Loop ---
    for (i, &global_bim_row_index) in prep_result.required_bim_indices.iter().enumerate() {
        // Hot loop check: do we need to switch to the next file? This is an O(1) comparison.
        if global_bim_row_index.0 >= next_boundary_start_idx {
            current_fileset_idx += 1;

            // Unmap the old file (implicitly done by dropping `mmap`) and map the new one.
            mmap = match File::open(&boundaries[current_fileset_idx].bed_path)
                .map_err(|e| PipelineError::Io(e.to_string()))
                .and_then(|file| unsafe {
                    Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string()))
                }) {
                Ok(map) => map,
                Err(e) => {
                    let _ = sparse_tx.send(Err(e.clone()));
                    let _ = dense_tx.send(Err(e));
                    return; // Stop production on error.
                }
            };

            // Update the next boundary index.
            next_boundary_start_idx = if boundaries.len() > current_fileset_idx + 1 {
                boundaries[current_fileset_idx + 1].starting_global_index
            } else {
                u64::MAX
            };
        }

        // Translate the global index to a local, file-specific index.
        let local_index =
            global_bim_row_index.0 - boundaries[current_fileset_idx].starting_global_index;
        let offset = (3 + local_index * bytes_per_variant) as usize;
        let end = offset + bytes_per_variant as usize;

        if end > mmap.len() {
            let err = PipelineError::Io(format!(
                "Fatal: Read past end of .bed file '{}' for variant with global index {}. File may be corrupt.",
                boundaries[current_fileset_idx].bed_path.display(),
                global_bim_row_index.0
            ));
            let _ = sparse_tx.send(Err(err.clone()));
            let _ = dense_tx.send(Err(err));
            return;
        }

        // The rest of this is identical to the original producer thread.
        let mut buffer = buffer_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant as usize));
        buffer.extend_from_slice(&mmap[offset..end]);

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
            break; // Consumers disconnected.
        }

        local_variants_processed += 1;
        if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
            variants_processed_count.fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
            local_variants_processed = 0;
        }
    }

    if local_variants_processed > 0 {
        variants_processed_count.fetch_add(local_variants_processed, Ordering::Relaxed);
    }
}
