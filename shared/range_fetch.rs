use crate::pipeline_error::PipelineError;
use std::sync::Arc;

const REMOTE_TRANSFER_CHUNK: usize = 16 * 1024 * 1024;

/// Fill one cache block with at most four concurrent range requests. Dense
/// scoring needs the entire block; issuing its parts together overlaps network
/// waits without increasing the cache budget or fetching additional bytes.
pub(crate) fn fetch_cache_block<F>(start: u64, length: usize, fetch: F) -> Result<Arc<Vec<u8>>, PipelineError>
where
    F: Fn(u64, usize) -> Result<Vec<u8>, PipelineError> + Sync,
{
    if length <= REMOTE_TRANSFER_CHUNK {
        return fetch(start, length).map(Arc::new);
    }
    let chunk_size = length.div_ceil(4).max(REMOTE_TRANSFER_CHUNK);
    let mut data = vec![0; length];
    std::thread::scope(|scope| {
        let fetch = &fetch;
        let handles: Vec<_> = data.chunks_mut(chunk_size).enumerate().map(|(index, target)| {
            scope.spawn(move || {
                let offset = start.checked_add((index * chunk_size) as u64)
                    .ok_or_else(|| PipelineError::Io("Remote range offset overflow".into()))?;
                let bytes = fetch(offset, target.len())?;
                if bytes.len() != target.len() {
                    return Err(PipelineError::Io("Remote range returned an incorrect length".into()));
                }
                target.copy_from_slice(&bytes);
                Ok(())
            })
        }).collect();
        for handle in handles {
            handle.join().map_err(|_| PipelineError::Io("Remote range worker panicked".into()))??;
        }
        Ok::<(), PipelineError>(())
    })?;
    Ok(Arc::new(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, atomic::{AtomicU64, Ordering}};

    #[test]
    fn dense_cache_fetch_overlaps_four_exact_ranges_in_order() {
        let barrier = std::sync::Barrier::new(4);
        let ranges = Mutex::new(Vec::new());
        let start = 123;
        let length = 4 * REMOTE_TRANSFER_CHUNK;
        let data = fetch_cache_block(start, length, |offset, size| {
            ranges.lock().unwrap().push((offset, size));
            // All four requests must begin before any response becomes ready.
            barrier.wait();
            Ok(vec![((offset - start) / REMOTE_TRANSFER_CHUNK as u64) as u8; size])
        }).expect("parallel block");
        let mut actual = ranges.into_inner().unwrap();
        actual.sort_unstable();
        assert_eq!(actual, (0..4).map(|index|
            (start + (index * REMOTE_TRANSFER_CHUNK) as u64, REMOTE_TRANSFER_CHUNK)
        ).collect::<Vec<_>>());
        for (index, chunk) in data.chunks(REMOTE_TRANSFER_CHUNK).enumerate() {
            assert!(chunk.iter().all(|&byte| byte == index as u8));
        }
        assert_eq!(data.len(), length);
    }

    #[test]
    fn cache_fetch_propagates_part_failure_and_rejects_truncation() {
        let failure = fetch_cache_block(0, REMOTE_TRANSFER_CHUNK + 1, |offset, size| {
            if offset > 0 { Err(PipelineError::Io("failed segment".into())) }
            else { Ok(vec![0; size]) }
        }).unwrap_err();
        assert!(failure.to_string().contains("failed segment"));
        let truncated = fetch_cache_block(0, REMOTE_TRANSFER_CHUNK + 1, |_, size|
            Ok(vec![0; size - 1])).unwrap_err();
        assert!(truncated.to_string().contains("incorrect length"));
    }

    #[test]
    fn sparse_cache_fetch_keeps_one_request() {
        let calls = AtomicU64::new(0);
        let data = fetch_cache_block(37, 5, |offset, size| {
            calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!((offset, size), (37, 5));
            Ok(vec![9; size])
        }).unwrap();
        assert_eq!(&**data, &[9; 5]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

