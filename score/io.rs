// ========================================================================================
//
//                       The high-performance data producer
//
// ========================================================================================
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
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model_ext::ReadRange;
use log::{debug, warn};
use memmap2::Mmap;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::runtime::Runtime;

/// The number of variants to process locally before updating the global atomic counter.
/// A power of 2 is often efficient
const PROGRESS_UPDATE_BATCH_SIZE: u64 = 1024;

const REMOTE_BLOCK_SIZE: usize = 8 * 1024 * 1024;
const REMOTE_CACHE_CAPACITY: usize = 8;

/// A trait that abstracts byte-range access for `.bed` data, regardless of the
/// underlying storage mechanism.
pub trait ByteRangeSource: Send + Sync {
    fn len(&self) -> u64;
    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError>;
}

/// A convenience wrapper that bundles the generic byte source with an optional
/// local memory map when the data lives on disk.
#[derive(Clone)]
pub struct BedSource {
    byte_source: Arc<dyn ByteRangeSource>,
    mmap: Option<Arc<Mmap>>,
}

impl BedSource {
    fn new(byte_source: Arc<dyn ByteRangeSource>, mmap: Option<Arc<Mmap>>) -> Self {
        Self { byte_source, mmap }
    }

    pub fn byte_source(&self) -> Arc<dyn ByteRangeSource> {
        Arc::clone(&self.byte_source)
    }

    pub fn mmap(&self) -> Option<Arc<Mmap>> {
        self.mmap.as_ref().map(Arc::clone)
    }

    pub fn len(&self) -> u64 {
        self.byte_source.len()
    }

    pub fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        self.byte_source.read_at(offset, dst)
    }
}

/// Creates a `BedSource` for the provided `.bed` path. Local paths are
/// memory-mapped, while `gs://` locations stream data directly from Cloud
/// Storage.
pub fn open_bed_source(path: &Path) -> Result<BedSource, PipelineError> {
    if is_gcs_path(path) {
        let uri = path
            .to_str()
            .ok_or_else(|| PipelineError::Io("Invalid UTF-8 in path".to_string()))?;
        let (bucket, object) = parse_gcs_uri(uri)?;
        let remote = RemoteByteRangeSource::new(&bucket, &object)?;
        Ok(BedSource::new(Arc::new(remote), None))
    } else {
        let file = File::open(path)
            .map_err(|e| PipelineError::Io(format!("Opening {}: {e}", path.display())))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|e| PipelineError::Io(e.to_string()))? };
        mmap.advise(memmap2::Advice::Sequential)
            .map_err(|e| PipelineError::Io(e.to_string()))?;
        let mmap = Arc::new(mmap);
        let byte_source = Arc::new(MmapByteRangeSource::new(Arc::clone(&mmap)));
        Ok(BedSource::new(byte_source, Some(mmap)))
    }
}

fn is_gcs_path(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.starts_with("gs://"))
        .unwrap_or(false)
}

fn parse_gcs_uri(uri: &str) -> Result<(String, String), PipelineError> {
    let without_scheme = uri.trim_start_matches("gs://");
    let mut parts = without_scheme.splitn(2, '/');
    let bucket = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PipelineError::Io(format!("Malformed GCS URI '{uri}': missing bucket")))?;
    let object = parts
        .next()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| PipelineError::Io(format!("Malformed GCS URI '{uri}': missing object")))?;
    Ok((bucket.to_string(), object.to_string()))
}

struct MmapByteRangeSource {
    mmap: Arc<Mmap>,
}

impl MmapByteRangeSource {
    fn new(mmap: Arc<Mmap>) -> Self {
        Self { mmap }
    }
}

impl ByteRangeSource for MmapByteRangeSource {
    fn len(&self) -> u64 {
        self.mmap.len() as u64
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        let offset = offset as usize;
        let end = offset
            .checked_add(dst.len())
            .ok_or_else(|| PipelineError::Io("Offset overflow while reading mmap".to_string()))?;
        if end > self.mmap.len() {
            return Err(PipelineError::Io(
                "Attempted to read past end of local .bed mapping".to_string(),
            ));
        }
        dst.copy_from_slice(&self.mmap[offset..end]);
        Ok(())
    }
}

struct RemoteByteRangeSource {
    runtime: Runtime,
    storage: Storage,
    bucket_path: String,
    object: String,
    len: u64,
    cache: Mutex<RemoteCache>,
}

impl RemoteByteRangeSource {
    fn new(bucket: &str, object: &str) -> Result<Self, PipelineError> {
        let runtime = Runtime::new()
            .map_err(|e| PipelineError::Io(format!("Failed to initialize Tokio runtime: {e}")))?;
        let storage = runtime.block_on(Storage::builder().build()).map_err(|e| {
            PipelineError::Io(format!("Failed to create Cloud Storage client: {e}"))
        })?;
        let control = runtime
            .block_on(StorageControl::builder().build())
            .map_err(|e| {
                PipelineError::Io(format!(
                    "Failed to create Cloud Storage control client: {e}"
                ))
            })?;
        let bucket_path = format!("projects/_/buckets/{bucket}");
        let metadata = runtime
            .block_on(
                control
                    .get_object()
                    .set_bucket(bucket_path.clone())
                    .set_object(object.to_string())
                    .send(),
            )
            .map_err(|e| {
                PipelineError::Io(format!(
                    "Failed to fetch metadata for gs://{bucket}/{object}: {e}"
                ))
            })?;
        if metadata.size < 0 {
            return Err(PipelineError::Io(format!(
                "Remote object gs://{bucket}/{object} reported negative size"
            )));
        }
        let len = metadata.size as u64;
        debug!(
            "Initialized remote BED source gs://{bucket}/{object} ({} bytes)",
            len
        );
        Ok(Self {
            runtime,
            storage,
            bucket_path,
            object: object.to_string(),
            len,
            cache: Mutex::new(RemoteCache::new(REMOTE_CACHE_CAPACITY)),
        })
    }

    fn block_length(&self, start: u64) -> usize {
        let remaining = self.len.saturating_sub(start);
        remaining.min(REMOTE_BLOCK_SIZE as u64) as usize
    }

    fn ensure_block(&self, start: u64) -> Result<Arc<Vec<u8>>, PipelineError> {
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(block) = cache.get(start) {
                return Ok(block);
            }
        }

        let length = self.block_length(start);
        if length == 0 {
            return Err(PipelineError::Io(
                "Requested block beyond end of object".to_string(),
            ));
        }
        let data = self.fetch_block(start, length)?;
        let mut cache = self.cache.lock().unwrap();
        cache.insert(start, Arc::clone(&data));
        Ok(data)
    }

    fn fetch_block(&self, start: u64, length: usize) -> Result<Arc<Vec<u8>>, PipelineError> {
        let bucket_path = self.bucket_path.clone();
        let object = self.object.clone();
        let bucket_for_log = bucket_path.clone();
        let object_for_log = object.clone();
        let storage = self.storage.clone();
        let mut data = self.runtime.block_on(async move {
            let mut response = storage
                .read_object(bucket_path.clone(), object.clone())
                .set_read_range(ReadRange::segment(start, length as u64))
                .send()
                .await
                .map_err(|e| {
                    PipelineError::Io(format!(
                        "Failed to start range read at offset {start} for gs://{}/{object}: {e}",
                        bucket_path
                    ))
                })?;
            let mut buffer = Vec::with_capacity(length);
            while let Some(chunk) = response.next().await {
                let chunk = chunk.map_err(|e| {
                    PipelineError::Io(format!(
                        "Error streaming data from gs://{}/{object}: {e}",
                        bucket_path
                    ))
                })?;
                buffer.extend_from_slice(&chunk);
            }
            Ok::<Vec<u8>, PipelineError>(buffer)
        })?;

        if data.len() != length {
            warn!(
                "Remote range read returned {} bytes, expected {} (gs://{}/{})",
                data.len(),
                length,
                bucket_for_log,
                object_for_log
            );
            if data.len() < length {
                return Err(PipelineError::Io(format!(
                    "Remote range read from gs://{}/{object_for_log} truncated: expected {length} bytes, received {}",
                    bucket_for_log,
                    data.len()
                )));
            }
            data.truncate(length);
        }
        Ok(Arc::new(data))
    }
}

impl ByteRangeSource for RemoteByteRangeSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
        if dst.is_empty() {
            return Ok(());
        }
        if offset >= self.len {
            return Err(PipelineError::Io(format!(
                "Attempted to read past end of remote object at offset {offset}"
            )));
        }
        let end = offset.checked_add(dst.len() as u64).ok_or_else(|| {
            PipelineError::Io("Offset overflow while reading remote object".to_string())
        })?;
        if end > self.len {
            return Err(PipelineError::Io(format!(
                "Attempted to read past end of remote object (offset {offset}, len {})",
                dst.len()
            )));
        }

        let mut remaining = dst.len();
        let mut cursor = 0usize;
        let mut current_offset = offset;
        let block_size = REMOTE_BLOCK_SIZE as u64;

        while remaining > 0 {
            let block_start = (current_offset / block_size) * block_size;
            let block = self.ensure_block(block_start)?;
            let within_block = (current_offset - block_start) as usize;
            if within_block >= block.len() {
                return Err(PipelineError::Io(format!(
                    "Computed block offset {within_block} exceeds block size {}",
                    block.len()
                )));
            }
            let available = block.len() - within_block;
            let to_copy = available.min(remaining);
            dst[cursor..cursor + to_copy]
                .copy_from_slice(&block[within_block..within_block + to_copy]);
            cursor += to_copy;
            remaining -= to_copy;
            current_offset += to_copy as u64;

            if within_block + to_copy == block.len() {
                let next_start = block_start + block_size;
                if next_start < self.len {
                    let _ = self.ensure_block(next_start);
                }
            }
        }

        Ok(())
    }
}

struct RemoteCache {
    capacity: usize,
    blocks: HashMap<u64, Arc<Vec<u8>>>,
    order: VecDeque<u64>,
}

impl RemoteCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            blocks: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, key: u64) -> Option<Arc<Vec<u8>>> {
        if let Some(value) = self.blocks.get(&key).cloned() {
            self.touch(key);
            Some(value)
        } else {
            None
        }
    }

    fn insert(&mut self, key: u64, value: Arc<Vec<u8>>) {
        if self.blocks.contains_key(&key) {
            self.blocks.insert(key, value);
            self.touch(key);
            return;
        }
        if self.order.len() == self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.blocks.remove(&oldest);
            }
        }
        self.order.push_back(key);
        self.blocks.insert(key, value);
    }

    fn touch(&mut self, key: u64) {
        self.order.retain(|&k| k != key);
        self.order.push_back(key);
    }
}

/// The generic entry point for the producer thread.
///
/// This function is a template that iterates through required variants and uses a
/// provided `path_decider` closure to determine the compute path. The compiler will
/// create specialized, optimized versions of this function for each type of closure
/// it is called with, enabling static dispatch and inlining for maximum performance.
///
/// # Arguments
/// * `source`: A shared, thread-safe handle to the underlying .bed byte source.
/// * `prep_result`: The "computation blueprint" that dictates which variants to read.
/// * `sparse_tx`: The channel sender for variants destined for the sparse path.
/// * `dense_tx`: The channel sender for variants destined for the dense path.
/// * `buffer_pool`: A shared pool of reusable byte buffers to eliminate allocation overhead.
/// * `path_decider`: A closure that takes a variant's data and returns the `ComputePath`.
pub fn producer_thread<F>(
    source: Arc<dyn ByteRangeSource>,
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
    let bytes_per_variant_u64 = prep_result.bytes_per_variant;
    let mut local_variants_processed: u64 = 0;

    for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
        let mut buffer = buffer_pool
            .pop()
            .unwrap_or_else(|| Vec::with_capacity(bytes_per_variant));
        buffer.clear();
        if buffer.capacity() < bytes_per_variant {
            buffer.reserve(bytes_per_variant - buffer.capacity());
        }
        buffer.resize(bytes_per_variant, 0);

        let offset = 3 + bim_row_idx.0 * bytes_per_variant_u64;
        let end = offset + bytes_per_variant_u64;

        if end > source.len() {
            let err = PipelineError::Io(format!(
                "Fatal: Attempted to read past the end of the .bed source for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                bim_row_idx.0
            ));
            let _ = sparse_tx.send(Err(err.clone()));
            let _ = dense_tx.send(Err(err));
            break;
        }

        if let Err(err) = source.read_at(offset, buffer.as_mut_slice()) {
            let _ = sparse_tx.send(Err(err.clone()));
            let _ = dense_tx.send(Err(err));
            break;
        }

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
    bed_sources: &[BedSource],
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

    debug_assert_eq!(boundaries.len(), bed_sources.len());

    // Open the first source.
    let mut current_source = bed_sources[0].byte_source();

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

            current_source = bed_sources[current_fileset_idx].byte_source();

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
        let offset = 3 + local_index * bytes_per_variant;
        let end = offset + bytes_per_variant;

        if end > current_source.len() {
            let err = PipelineError::Io(format!(
                "Fatal: Read past end of .bed source '{}' for variant with global index {}. Source may be corrupt.",
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
        buffer.clear();
        if buffer.capacity() < bytes_per_variant as usize {
            buffer.reserve(bytes_per_variant as usize - buffer.capacity());
        }
        buffer.resize(bytes_per_variant as usize, 0);

        if let Err(err) = current_source.read_at(offset, buffer.as_mut_slice()) {
            let _ = sparse_tx.send(Err(err.clone()));
            let _ = dense_tx.send(Err(err));
            return;
        }

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

#[cfg(test)]
mod tests {
    use super::*;

    const PUBLIC_BUCKET: &str = "genomics-public-data";
    const PUBLIC_REFERENCE_OBJECT: &str =
        "references/hg38/v0/Homo_sapiens_assembly38.fasta";

    fn ensure_remote_source() -> Result<RemoteByteRangeSource, PipelineError> {
        RemoteByteRangeSource::new(PUBLIC_BUCKET, PUBLIC_REFERENCE_OBJECT)
    }

    #[test]
    fn remote_reference_exposes_length_and_initial_bytes() -> Result<(), PipelineError> {
        let source = ensure_remote_source()?;
        assert!(source.len() > 0, "Remote reference must report a non-zero length");

        let mut buffer = vec![0u8; 4096];
        source.read_at(0, &mut buffer)?;
        assert!(
            buffer.iter().any(|&byte| byte != 0),
            "Expected streamed data to contain non-zero bytes"
        );

        Ok(())
    }

    #[test]
    fn remote_reference_streams_across_multiple_blocks() -> Result<(), PipelineError> {
        let source = ensure_remote_source()?;
        let read_len = REMOTE_BLOCK_SIZE + 1024;
        let mut buffer = vec![0u8; read_len];
        source.read_at(0, &mut buffer)?;
        assert!(
            buffer.iter().any(|&byte| byte != 0),
            "Expected streamed range to contain remote data"
        );

        Ok(())
    }

    #[test]
    fn remote_reference_supports_boundary_reads() -> Result<(), PipelineError> {
        let source = ensure_remote_source()?;
        let len = source.len();
        assert!(len > REMOTE_BLOCK_SIZE as u64);

        let offset = REMOTE_BLOCK_SIZE as u64 - 256;
        let mut buffer = vec![0u8; 1024];
        source.read_at(offset, &mut buffer)?;
        assert!(
            buffer.iter().any(|&byte| byte != 0),
            "Expected boundary read to yield streamed data"
        );

        let tail_offset = len.saturating_sub(512);
        let mut tail_buffer = vec![0u8; (len - tail_offset) as usize];
        source.read_at(tail_offset, &mut tail_buffer)?;
        assert!(
            tail_buffer.iter().any(|&byte| byte != 0),
            "Expected trailing read to yield streamed data"
        );

        Ok(())
    }
}
