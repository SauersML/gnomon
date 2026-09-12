//! Planned, prefetching reads of remote PLINK BED rows.
//!
//! Scoring reads whole variant rows in ascending row order, and a biobank
//! cohort makes every row hundreds of kilobytes. A remote BED is therefore
//! read through a plan of exact byte ranges that cover only the required
//! rows, and a pool of workers keeps a bounded window of those ranges in
//! flight ahead of the consumer. Request latency then overlaps both the
//! transfer of neighbouring ranges and the scoring of rows already received.

use crate::pipeline_error::PipelineError;
use std::collections::BTreeMap;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Fetches exactly `length` bytes starting at an absolute object offset.
pub(crate) type SegmentFetch =
    Arc<dyn Fn(u64, usize) -> Result<Vec<u8>, PipelineError> + Send + Sync>;

const REMOTE_TRANSFER_CHUNK: usize = 16 * 1024 * 1024;

/// One byte range of the object: `(start, length)`.
type Range = (u64, usize);

/// Consecutive required BED rows share requests; gaps are never downloaded.
/// The three-byte header is always its own range so that validating it does
/// not start genotype prefetching.
pub(crate) struct BedReadPlan {
    ranges: Vec<Range>,
}

impl BedReadPlan {
    /// Longest single request. Consecutive rows are merged up to this size.
    pub(crate) const MAX_RANGE: usize = 2 * 1024 * 1024;

    pub(crate) fn new(rows: &[u64], row_bytes: u64, file_len: u64) -> Result<Self, PipelineError> {
        if row_bytes == 0 || file_len < 3 || (file_len - 3) % row_bytes != 0 {
            return Err(PipelineError::Io(
                "Invalid BED dimensions for scoring read plan".into(),
            ));
        }
        let mut ranges = vec![(0, 3usize)];
        let mut previous = None;
        for &row in rows {
            if previous.is_some_and(|prior| prior >= row) {
                return Err(PipelineError::Io(
                    "BED read-plan rows must be strictly increasing".into(),
                ));
            }
            previous = Some(row);
            let start = row
                .checked_mul(row_bytes)
                .and_then(|n| n.checked_add(3))
                .ok_or_else(|| PipelineError::Io("BED row offset overflow".into()))?;
            let end = start
                .checked_add(row_bytes)
                .filter(|&end| end <= file_len)
                .ok_or_else(|| PipelineError::Io("BED read-plan row exceeds file length".into()))?;
            let mut cursor = start;
            while cursor < end {
                let last = ranges.last_mut().expect("header range");
                let contiguous = last.0 != 0 && last.0 + last.1 as u64 == cursor;
                if contiguous && last.1 < Self::MAX_RANGE {
                    let n = (end - cursor).min((Self::MAX_RANGE - last.1) as u64) as usize;
                    last.1 += n;
                    cursor += n as u64;
                } else {
                    let n = (end - cursor).min(Self::MAX_RANGE as u64) as usize;
                    ranges.push((cursor, n));
                    cursor += n as u64;
                }
            }
        }
        Ok(Self { ranges })
    }

    /// Index of the range containing `offset`; reads outside the plan are errors.
    fn index_of(&self, offset: u64) -> Result<usize, PipelineError> {
        let index = self
            .ranges
            .partition_point(|&(start, _)| start <= offset)
            .checked_sub(1)
            .ok_or_else(|| PipelineError::Io("Read outside required BED rows".into()))?;
        let (start, length) = self.ranges[index];
        if offset - start >= length as u64 {
            return Err(PipelineError::Io("Read outside required BED rows".into()));
        }
        Ok(index)
    }
}

/// Memory and concurrency bounds of a [`PlannedReader`].
#[derive(Clone, Copy)]
pub(crate) struct Limits {
    /// Bytes dispatched and not yet discarded: in flight plus received but unread.
    pub window_bytes: usize,
    /// Bytes to keep in flight; divided by the mean range size to choose workers.
    pub in_flight_bytes: usize,
    pub min_workers: usize,
    pub max_workers: usize,
}

/// Sized for a cohort whose rows are ~100 KB: a full window of such rows
/// keeps a 10-30 Gbps VM busy while request latency stays the bottleneck.
pub(crate) const LIMITS: Limits = Limits {
    window_bytes: 256 * 1024 * 1024,
    in_flight_bytes: 128 * 1024 * 1024,
    min_workers: 8,
    max_workers: 256,
};

struct State {
    /// Lowest range index the consumer may still ask for.
    consumer: usize,
    /// Next range index a worker will take.
    dispatch: usize,
    /// Received ranges at or beyond `consumer`, by index.
    ready: BTreeMap<usize, Result<Arc<Vec<u8>>, PipelineError>>,
    /// Bytes in flight or in `ready`.
    window_bytes: usize,
    started: bool,
    /// No further ranges may be dispatched: the reader is being dropped or a
    /// fetch failed. The failure itself is delivered through `ready`.
    halted: bool,
}

struct Shared {
    limits: Limits,
    state: Mutex<State>,
    changed: Condvar,
}

/// Serves reads from a [`BedReadPlan`] while workers prefetch ahead of them.
///
/// Reads are expected in ascending offset order, which is how scoring
/// consumes rows; a read behind the consumer position is served by one
/// synchronous request and never disturbs the prefetch window. Workers start
/// on the first genotype read, at that read's position, so a resumed run
/// never downloads rows it has already scored.
pub(crate) struct PlannedReader {
    plan: Arc<BedReadPlan>,
    fetch: SegmentFetch,
    shared: Arc<Shared>,
    workers: Mutex<Vec<JoinHandle<()>>>,
}

impl PlannedReader {
    pub(crate) fn new(plan: BedReadPlan, fetch: SegmentFetch) -> Self {
        Self::with_limits(plan, fetch, LIMITS)
    }

    fn with_limits(plan: BedReadPlan, fetch: SegmentFetch, limits: Limits) -> Self {
        Self {
            plan: Arc::new(plan),
            fetch,
            shared: Arc::new(Shared {
                limits,
                state: Mutex::new(State {
                    consumer: 0,
                    dispatch: 0,
                    ready: BTreeMap::new(),
                    window_bytes: 0,
                    started: false,
                    halted: false,
                }),
                changed: Condvar::new(),
            }),
            workers: Mutex::new(Vec::new()),
        }
    }

    /// Concurrent requests: enough to keep the in-flight target full of
    /// ranges of this plan's mean size, within fixed bounds.
    pub(crate) fn worker_count(plan: &BedReadPlan, limits: &Limits) -> usize {
        let rows = &plan.ranges[1..];
        if rows.is_empty() {
            return 0;
        }
        let mean = rows.iter().map(|&(_, n)| n).sum::<usize>() / rows.len();
        (limits.in_flight_bytes / mean.max(1))
            .clamp(limits.min_workers, limits.max_workers)
            .min(rows.len())
    }

    /// Ranges after the header, one request each.
    pub(crate) fn ranges(&self) -> usize {
        self.plan.ranges.len() - 1
    }

    pub(crate) fn workers(&self) -> usize {
        Self::worker_count(&self.plan, &self.shared.limits)
    }

    /// The range containing `offset`, as `(range start, bytes)`.
    pub(crate) fn range_at(&self, offset: u64) -> Result<(u64, Arc<Vec<u8>>), PipelineError> {
        let index = self.plan.index_of(offset)?;
        let (start, length) = self.plan.ranges[index];
        if index == 0 {
            return self.fetch_now(start, length);
        }
        let mut state = self.shared.state.lock().unwrap();
        if index < state.consumer {
            drop(state);
            return self.fetch_now(start, length);
        }
        state.consumer = index;
        let behind = {
            let ahead = state.ready.split_off(&index);
            std::mem::replace(&mut state.ready, ahead)
        };
        for &stale in behind.keys() {
            state.window_bytes -= self.plan.ranges[stale].1;
        }
        if index > state.dispatch {
            state.dispatch = index;
        }
        let spawn = !state.started;
        state.started = true;
        self.shared.changed.notify_all();
        if spawn {
            drop(state);
            self.spawn_workers();
            state = self.shared.state.lock().unwrap();
        }
        loop {
            if let Some(entry) = state.ready.get(&index) {
                return entry.clone().map(|data| (start, data));
            }
            if state.halted && index >= state.dispatch {
                drop(state);
                return self.fetch_now(start, length);
            }
            state = self.shared.changed.wait(state).unwrap();
        }
    }

    fn fetch_now(&self, start: u64, length: usize) -> Result<(u64, Arc<Vec<u8>>), PipelineError> {
        fetch_exact(&self.fetch, start, length).map(|data| (start, Arc::new(data)))
    }

    fn spawn_workers(&self) {
        let mut workers = self.workers.lock().unwrap();
        for _ in 0..Self::worker_count(&self.plan, &self.shared.limits) {
            let plan = Arc::clone(&self.plan);
            let fetch = Arc::clone(&self.fetch);
            let shared = Arc::clone(&self.shared);
            workers.push(std::thread::spawn(move || worker(&plan, &fetch, &shared)));
        }
    }
}

impl Drop for PlannedReader {
    fn drop(&mut self) {
        self.shared.state.lock().unwrap().halted = true;
        self.shared.changed.notify_all();
        for worker in self.workers.lock().unwrap().drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker(plan: &BedReadPlan, fetch: &SegmentFetch, shared: &Shared) {
    loop {
        let index = {
            let mut state = shared.state.lock().unwrap();
            loop {
                if state.halted || state.dispatch >= plan.ranges.len() {
                    return;
                }
                let length = plan.ranges[state.dispatch].1;
                if state.window_bytes == 0 || state.window_bytes + length <= shared.limits.window_bytes {
                    state.window_bytes += length;
                    state.dispatch += 1;
                    break state.dispatch - 1;
                }
                state = shared.changed.wait(state).unwrap();
            }
        };
        let (start, length) = plan.ranges[index];
        let result = fetch_exact(fetch, start, length).map(Arc::new);
        let mut state = shared.state.lock().unwrap();
        if index < state.consumer {
            state.window_bytes -= length;
        } else {
            if result.is_err() {
                state.halted = true;
            }
            state.ready.insert(index, result);
        }
        shared.changed.notify_all();
    }
}

fn fetch_exact(fetch: &SegmentFetch, start: u64, length: usize) -> Result<Vec<u8>, PipelineError> {
    let data = fetch(start, length)?;
    if data.len() != length {
        return Err(PipelineError::Io(
            "BED range returned an incorrect length".into(),
        ));
    }
    Ok(data)
}

/// Fill one cache block with at most four concurrent range requests. Dense
/// sweeps need the entire block; issuing its parts together overlaps network
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::time::Duration;

    /// Byte value that identifies its own offset, so reassembly errors show.
    fn pattern(offset: u64, length: usize) -> Vec<u8> {
        (offset..offset + length as u64).map(|o| (o % 251) as u8).collect()
    }

    fn fetcher<F>(f: F) -> SegmentFetch
    where
        F: Fn(u64, usize) -> Result<Vec<u8>, PipelineError> + Send + Sync + 'static,
    {
        Arc::new(f)
    }

    #[test]
    fn plan_covers_required_rows_and_nothing_else() {
        let plan = BedReadPlan::new(&[0, 1, 99], 100, 10_003).unwrap();
        assert_eq!(plan.ranges, vec![(0, 3), (3, 200), (9903, 100)]);
        assert_eq!(plan.index_of(0).unwrap(), 0);
        assert_eq!(plan.index_of(102).unwrap(), 1);
        assert_eq!(plan.index_of(10_002).unwrap(), 2);
        assert!(plan.index_of(203).is_err());
        assert!(plan.index_of(10_003).is_err());
        let large = BedReadPlan::new(
            &[0],
            (BedReadPlan::MAX_RANGE * 3 + 1) as u64,
            (BedReadPlan::MAX_RANGE * 3 + 4) as u64,
        )
        .unwrap();
        assert!(large.ranges[1..].iter().all(|&(_, n)| n <= BedReadPlan::MAX_RANGE));
        assert_eq!(
            large.ranges[1..].iter().map(|&(_, n)| n).sum::<usize>(),
            BedReadPlan::MAX_RANGE * 3 + 1
        );
    }

    #[test]
    fn plan_rejects_bad_dimensions_and_rows() {
        for (rows, stride, length) in [
            (&[0][..], 0, 3),
            (&[1, 0][..], 4, 11),
            (&[0, 0][..], 4, 11),
            (&[2][..], 4, 11),
            (&[0][..], 4, 10),
        ] {
            assert!(BedReadPlan::new(rows, stride, length).is_err());
        }
    }

    #[test]
    fn worker_count_tracks_range_size_within_bounds() {
        let dense = BedReadPlan::new(&(0..64).collect::<Vec<_>>(), BedReadPlan::MAX_RANGE as u64,
            3 + 64 * BedReadPlan::MAX_RANGE as u64).unwrap();
        assert_eq!(PlannedReader::worker_count(&dense, &LIMITS), LIMITS.in_flight_bytes / BedReadPlan::MAX_RANGE);
        let scattered = BedReadPlan::new(&(0..2000).map(|r| r * 2).collect::<Vec<_>>(), 8, 3 + 4000 * 8).unwrap();
        assert_eq!(PlannedReader::worker_count(&scattered, &LIMITS), LIMITS.max_workers);
        let tiny = BedReadPlan::new(&[0, 2], 8, 27).unwrap();
        assert_eq!(PlannedReader::worker_count(&tiny, &LIMITS), 2);
        assert_eq!(PlannedReader::worker_count(&BedReadPlan::new(&[], 8, 27).unwrap(), &LIMITS), 0);
    }

    #[test]
    fn sequential_reads_are_prefetched_concurrently_and_correct() {
        // Every other row is required, so each range is one row, and there
        // are more ranges than the worker ceiling so the ceiling is reached.
        let rows: Vec<u64> = (0..300).map(|r| r * 2).collect();
        let plan = BedReadPlan::new(&rows, 64, 3 + 600 * 64).unwrap();
        let workers = PlannedReader::worker_count(&plan, &LIMITS);
        assert_eq!(workers, LIMITS.max_workers);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let bytes = Arc::new(AtomicUsize::new(0));
        let reader = PlannedReader::new(plan, fetcher({
            let (in_flight, peak, bytes) = (in_flight.clone(), peak.clone(), bytes.clone());
            move |offset, length| {
                let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(100));
                in_flight.fetch_sub(1, Ordering::SeqCst);
                bytes.fetch_add(length, Ordering::SeqCst);
                Ok(pattern(offset, length))
            }
        }));
        let (start, header) = reader.range_at(0).unwrap();
        assert_eq!((start, header.len()), (0, 3));
        for &row in &rows {
            let offset = 3 + row * 64;
            let (start, data) = reader.range_at(offset + 5).unwrap();
            assert_eq!(start, offset);
            assert_eq!(*data, pattern(offset, 64));
        }
        assert!(peak.load(Ordering::SeqCst) >= workers / 4, "prefetch ran {} wide", peak.load(Ordering::SeqCst));
        drop(reader);
        assert_eq!(bytes.load(Ordering::SeqCst), 3 + 300 * 64);
    }

    #[test]
    fn header_read_does_not_start_prefetching() {
        let plan = BedReadPlan::new(&(0..50).collect::<Vec<_>>(), 16, 3 + 50 * 16).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let reader = PlannedReader::new(plan, fetcher({
            let calls = calls.clone();
            move |offset, length| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(pattern(offset, length))
            }
        }));
        reader.range_at(1).unwrap();
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(reader.workers.lock().unwrap().is_empty());
    }

    #[test]
    fn resumed_reads_never_fetch_earlier_rows() {
        // Alternate rows keep every range distinct, so no range spans the
        // resume point and every fetched offset must lie at or beyond it.
        let rows: Vec<u64> = (0..4000).map(|r| r * 2).collect();
        let plan = BedReadPlan::new(&rows, 64, 3 + 8000 * 64).unwrap();
        let resume_offset = 3 + rows[2500] * 64;
        let fetched = Arc::new(AtomicUsize::new(0));
        let reader = PlannedReader::new(plan, fetcher({
            let fetched = fetched.clone();
            move |offset, length| {
                assert!(offset >= resume_offset, "fetched {offset} before the resume point");
                fetched.fetch_add(1, Ordering::SeqCst);
                Ok(pattern(offset, length))
            }
        }));
        for &row in &rows[2500..] {
            let offset = 3 + row * 64;
            let (start, data) = reader.range_at(offset + 7).unwrap();
            assert_eq!(start, offset);
            assert_eq!(data[7], ((offset + 7) % 251) as u8);
        }
        drop(reader);
        assert_eq!(fetched.load(Ordering::SeqCst), 1500);
    }

    #[test]
    fn window_bounds_outstanding_bytes_and_is_fully_used() {
        let rows: Vec<u64> = (0..64).map(|r| r * 2).collect();
        let plan = BedReadPlan::new(&rows, 1024, 3 + 128 * 1024).unwrap();
        let limits = Limits { window_bytes: 4096, in_flight_bytes: 1 << 20, min_workers: 8, max_workers: 8 };
        let outstanding = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let reader = PlannedReader::with_limits(plan, fetcher({
            let (outstanding, peak) = (outstanding.clone(), peak.clone());
            move |offset, length| {
                let now = outstanding.fetch_add(length, Ordering::SeqCst) + length;
                peak.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(3));
                Ok(pattern(offset, length))
            }
        }), limits);
        for &row in &rows {
            let offset = 3 + row * 1024;
            let (_, data) = reader.range_at(offset).unwrap();
            assert_eq!(*data, pattern(offset, 1024));
            // Bytes leave the window only once the consumer has moved past them.
            outstanding.fetch_sub(1024, Ordering::SeqCst);
        }
        assert!(peak.load(Ordering::SeqCst) <= limits.window_bytes, "peak {}", peak.load(Ordering::SeqCst));
        assert_eq!(peak.load(Ordering::SeqCst), limits.window_bytes);
    }

    #[test]
    fn all_workers_run_at_once_when_the_window_allows() {
        let rows: Vec<u64> = (0..16).map(|r| r * 3).collect();
        let plan = BedReadPlan::new(&rows, 8, 3 + 48 * 8).unwrap();
        let workers = PlannedReader::worker_count(&plan, &LIMITS);
        assert_eq!(workers, rows.len());
        let barrier = Arc::new(Barrier::new(workers));
        let reader = PlannedReader::new(plan, fetcher({
            let barrier = barrier.clone();
            move |offset, length| {
                if offset != 0 {
                    // Deadlocks unless every worker has a request in flight.
                    barrier.wait();
                }
                Ok(pattern(offset, length))
            }
        }));
        for &row in &rows {
            let offset = 3 + row * 8;
            assert_eq!(*reader.range_at(offset).unwrap().1, pattern(offset, 8));
        }
    }

    #[test]
    fn reads_behind_the_consumer_are_served_without_disturbing_prefetch() {
        // Alternate rows keep every range distinct.
        let rows: Vec<u64> = (0..40).map(|r| r * 2).collect();
        let plan = BedReadPlan::new(&rows, 8, 3 + 80 * 8).unwrap();
        let reader = PlannedReader::new(plan, fetcher(|offset, length| Ok(pattern(offset, length))));
        reader.range_at(3 + rows[30] * 8).unwrap();
        let (start, data) = reader.range_at(3 + rows[2] * 8).unwrap();
        assert_eq!(start, 3 + rows[2] * 8);
        assert_eq!(*data, pattern(start, 8));
        let (start, data) = reader.range_at(3 + rows[31] * 8).unwrap();
        assert_eq!(start, 3 + rows[31] * 8);
        assert_eq!(*data, pattern(start, 8));
    }

    #[test]
    fn failures_and_short_responses_reach_the_consumer() {
        let rows: Vec<u64> = (0..30).map(|r| r * 2).collect();
        let plan = BedReadPlan::new(&rows, 8, 3 + 60 * 8).unwrap();
        let failing = 3 + rows[20] * 8;
        let reader = PlannedReader::new(plan, fetcher(move |offset, length| {
            if offset == failing {
                Err(PipelineError::Io("segment failed".into()))
            } else {
                Ok(pattern(offset, length))
            }
        }));
        for &row in &rows[..20] {
            reader.range_at(3 + row * 8).unwrap();
        }
        let error = reader.range_at(failing).unwrap_err();
        assert!(error.to_string().contains("segment failed"));

        let plan = BedReadPlan::new(&rows, 8, 3 + 60 * 8).unwrap();
        let reader = PlannedReader::new(plan, fetcher(|_, length| Ok(vec![0; length - 1])));
        assert!(reader.range_at(0).unwrap_err().to_string().contains("incorrect length"));
        assert!(reader.range_at(3).unwrap_err().to_string().contains("incorrect length"));
        assert!(reader.range_at(4).is_err());
    }

    #[test]
    fn dense_cache_fetch_overlaps_four_exact_ranges_in_order() {
        let barrier = Barrier::new(4);
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
        let calls = AtomicUsize::new(0);
        let data = fetch_cache_block(37, 5, |offset, size| {
            calls.fetch_add(1, Ordering::SeqCst);
            assert_eq!((offset, size), (37, 5));
            Ok(vec![9; size])
        }).unwrap();
        assert_eq!(&**data, &[9; 5]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
