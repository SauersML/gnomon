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

use crate::pipeline_error::PipelineError;
use crate::score::batch;
use crate::score::decide::ComputePath;
use crate::score::types::{
    BimRowIndex, FilesetBoundary, PipelineKind, PreparationResult, ReconciledVariantIndex, WorkItem,
};
pub use crate::shared::files::{
    BedSource, ByteRangeSource, PROGRESS_UPDATE_BATCH_SIZE, TextSource,
    gcs_billing_project_from_env, get_shared_runtime, is_pgen_path, load_adc_credentials,
    open_bed_source, open_plink_text_source, open_text_source,
};
use ahash::AHashMap;
use crossbeam_channel::Sender;
use crossbeam_queue::ArrayQueue;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::Duration;

fn choose_score_path(
    data: &[u8],
    prep: &PreparationResult,
    index: usize,
    original: &impl Fn(&[u8]) -> ComputePath,
) -> ComputePath {
    let columns = prep.score_names.len();
    if (1..=64).contains(&columns)
        && prep.num_people_to_score >= 64
        && u32::try_from(index).ok().is_some_and(|index| {
            columns <= 4
                || prep.variant_csr_view(ReconciledVariantIndex(index)).len() <= columns / 4
        })
    {
        // The old tree sees total panel width and sends all wide-panel rows to
        // scalar accumulation. Sparse score schedules change that tradeoff:
        // common calls benefit from batching, while rare calls keep zero-word
        // skipping. Inspect actual row support instead of total panel width.
        return if batch::assess_variant_density_for_dispatch(data, prep.total_people_in_fam)
            > 0.0894
        {
            ComputePath::Pivot
        } else {
            ComputePath::NoPivot
        };
    }
    original(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_panel_dispatch_batches_common_calls_and_skips_rare_calls() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = dir.path().join("panel");
        std::fs::write(prefix.with_extension("bim"), "1 a 0 100 A G\n").unwrap();
        std::fs::write(
            prefix.with_extension("fam"),
            (0..64)
                .map(|i| format!("F I{i} 0 0 0 -9\n"))
                .collect::<String>(),
        )
        .unwrap();
        let mut bed = vec![0x6c, 0x1b, 0x01];
        bed.resize(19, 0);
        std::fs::write(prefix.with_extension("bed"), bed).unwrap();
        let weights = dir.path().join("weights.tsv");
        let header = (0..32).map(|i| format!("\tS{i:02}")).collect::<String>();
        std::fs::write(
            &weights,
            format!(
                "variant_id\teffect_allele\tother_allele{header}\n1:100\tG\tA\t0.25{}\n",
                "\t".repeat(31)
            ),
        )
        .unwrap();
        let mut prep =
            crate::score::prepare::prepare_for_computation(&[prefix], &[weights], None, None)
                .unwrap();
        let original = |_: &[u8]| ComputePath::NoPivot;
        assert_eq!(
            choose_score_path(&[0xff; 16], &prep, 0, &original),
            ComputePath::Pivot
        );
        assert_eq!(
            choose_score_path(&[0; 16], &prep, 0, &original),
            ComputePath::NoPivot
        );
        prep.person_subset = crate::score::types::PersonSubset::Indices((0..64).collect());
        assert_eq!(
            choose_score_path(&[0xff; 16], &prep, 0, &original),
            ComputePath::Pivot
        );
    }
}

#[cfg(test)]
mod pool_tests {
    use super::*;
    use crate::score::types::{OriginalPersonIndex, OutputPersonIndex, PersonSubset};
    use crossbeam_channel::{Receiver, bounded};
    use std::path::PathBuf;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Long enough that only a hang, never a slow machine, fails a test.
    const PATIENCE: Duration = Duration::from_secs(60);

    fn empty_pool(capacity: usize) -> Arc<RowBufferPool> {
        Arc::new(RowBufferPool::new(capacity, 0))
    }

    fn spawn_taker(
        pool: &Arc<RowBufferPool>,
        count: usize,
    ) -> (thread::JoinHandle<()>, Receiver<Option<Vec<u8>>>) {
        let pool = Arc::clone(pool);
        let (taken_tx, taken_rx) = bounded(count);
        let handle = thread::spawn(move || {
            for _ in 0..count {
                taken_tx.send(pool.take()).expect("test receiver");
            }
        });
        (handle, taken_rx)
    }

    fn wait_until_parked(pool: &RowBufferPool) {
        let deadline = Instant::now() + PATIENCE;
        while !pool.waiting.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "the taker never parked");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// A .bed image: the magic bytes, then one byte per row for four people.
    struct BedImage(Vec<u8>);

    impl ByteRangeSource for BedImage {
        fn len(&self) -> u64 {
            self.0.len() as u64
        }

        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
            let start = offset as usize;
            let bytes = self
                .0
                .get(start..start + dst.len())
                .ok_or_else(|| PipelineError::Io("read past end".to_string()))?;
            dst.copy_from_slice(bytes);
            Ok(())
        }
    }

    fn bed_image(rows: &[u8]) -> Arc<BedImage> {
        let mut bytes = vec![0x6c, 0x1b, 0x01];
        bytes.extend_from_slice(rows);
        Arc::new(BedImage(bytes))
    }

    fn four_person_prep(rows: usize) -> Arc<PreparationResult> {
        prep_for_rows(&(0..rows as u64).collect::<Vec<_>>(), rows as u64)
    }

    /// A four-person preparation that scores the `required` rows of a `total`-row .bed.
    fn prep_for_rows(required: &[u64], total: u64) -> Arc<PreparationResult> {
        let people = 4usize;
        let rows = required.len();
        let names = vec!["S0".to_string()];
        let offsets: Vec<u64> = (0..=rows as u64).collect();
        let exact = crate::score::cells::ExactPlan::new(
            vec![1.0; rows],
            &vec![0.0; rows],
            &vec![0; rows],
            &offsets,
            &[],
            &names,
        )
        .expect("exact plan");
        Arc::new(PreparationResult::new(
            exact,
            vec![0; rows],
            offsets,
            required.iter().map(|&row| BimRowIndex(row)).collect(),
            Vec::new(),
            names,
            vec![rows as u32],
            PersonSubset::All,
            (0..people).map(|i| format!("I{i}")).collect(),
            people,
            people,
            total,
            rows,
            1,
            (0..people as u32)
                .map(|i| Some(OutputPersonIndex(i)))
                .collect(),
            (0..people as u32).map(OriginalPersonIndex).collect(),
            vec![0; rows],
            vec![0],
            vec![0],
            1,
            PipelineKind::SingleFile(PathBuf::from("test")),
        ))
    }

    #[test]
    fn a_returned_buffer_wakes_a_parked_taker() {
        let pool = empty_pool(1);
        let (taker, taken) = spawn_taker(&pool, 1);
        wait_until_parked(&pool);
        pool.push(vec![7u8; 5]).expect("room in the pool");
        assert_eq!(
            taken.recv_timeout(PATIENCE).expect("the taker woke"),
            Some(vec![7u8; 5])
        );
        taker.join().expect("taker");
    }

    #[test]
    fn a_parked_taker_waits_for_half_the_pool() {
        let pool = Arc::new(RowBufferPool::with_patience(4, 0, PATIENCE));
        let (taker, taken) = spawn_taker(&pool, 1);
        wait_until_parked(&pool);
        pool.push(vec![1u8]).expect("room in the pool");
        // One buffer of four is not enough to wake it.
        assert!(taken.recv_timeout(Duration::from_millis(100)).is_err());
        assert!(
            pool.waiting.load(Ordering::SeqCst),
            "the taker is still parked"
        );
        pool.push(vec![2u8]).expect("room in the pool");
        assert_eq!(
            taken.recv_timeout(PATIENCE).expect("the taker woke"),
            Some(vec![1u8])
        );
        taker.join().expect("taker");
    }

    #[test]
    fn the_wake_threshold_leaves_out_what_consumers_keep() {
        // One CPU: 320 buffers, and a dense batcher that keeps up to 255 rows as it waits.
        assert_eq!(RowBufferPool::new(320, 256).wake_at, 64);
        assert_eq!(RowBufferPool::new(640, 256).wake_at, 320);
        assert_eq!(RowBufferPool::new(200, 256).wake_at, 1);
        assert_eq!(RowBufferPool::new(0, 0).wake_at, 1);
    }

    #[test]
    fn a_parked_taker_wakes_once_consumers_can_give_back_no_more() {
        // Consumers keep up to three of the four buffers, so the one that comes back wakes it.
        let pool = Arc::new(RowBufferPool::with_patience(4, 3, PATIENCE));
        let (taker, taken) = spawn_taker(&pool, 1);
        wait_until_parked(&pool);
        pool.push(vec![1u8]).expect("room in the pool");
        assert_eq!(
            taken.recv_timeout(PATIENCE).expect("the taker woke"),
            Some(vec![1u8])
        );
        taker.join().expect("taker");
    }

    #[test]
    fn a_parked_taker_takes_fewer_once_its_patience_runs_out() {
        // Nothing declared, so it waits for two of four buffers, and only one comes back.
        let pool = Arc::new(RowBufferPool::with_patience(
            4,
            0,
            Duration::from_millis(10),
        ));
        let (taker, taken) = spawn_taker(&pool, 1);
        wait_until_parked(&pool);
        pool.push(vec![1u8]).expect("room in the pool");
        assert_eq!(
            taken
                .recv_timeout(PATIENCE)
                .expect("the taker stopped waiting"),
            Some(vec![1u8])
        );
        taker.join().expect("taker");
    }

    #[test]
    fn a_producer_wakes_while_a_batcher_keeps_most_of_the_pool() {
        // As on one CPU: the dense batcher keeps four rows of a five-row batch, so only the
        // two sparse rows come back, fewer than half of the six buffers. Told what the
        // batcher keeps, the pool must wake the producer on those two.
        const DENSE: u8 = 0xFF;
        let rows = [DENSE, DENSE, DENSE, DENSE, 0, 0, 0, 0, DENSE];
        let pool = Arc::new(RowBufferPool::with_patience(6, 5, PATIENCE));
        for _ in 0..6 {
            pool.push(Vec::new()).expect("room in the pool");
        }
        let (sparse_tx, sparse_rx) = bounded(rows.len());
        let (dense_tx, dense_rx) = bounded(rows.len());
        let producer = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                producer_thread(
                    bed_image(&rows),
                    four_person_prep(rows.len()),
                    Some(sparse_tx),
                    dense_tx,
                    pool,
                    Arc::new(AtomicU64::new(0)),
                    |row: &[u8]| {
                        if row[0] == DENSE {
                            ComputePath::Pivot
                        } else {
                            ComputePath::NoPivot
                        }
                    },
                    None,
                )
            })
        };
        let next = |rx: &Receiver<Result<WorkItem, PipelineError>>| {
            rx.recv_timeout(PATIENCE)
                .expect("a row")
                .expect("no producer error")
                .data
        };
        let mut batch: Vec<Vec<u8>> = (0..4).map(|_| next(&dense_rx)).collect();
        let sparse: Vec<Vec<u8>> = (0..2).map(|_| next(&sparse_rx)).collect();
        wait_until_parked(&pool);
        for buffer in sparse {
            pool.push(buffer).expect("room in the pool");
        }
        for _ in 0..2 {
            pool.push(next(&sparse_rx)).expect("room in the pool");
        }
        batch.push(next(&dense_rx));
        assert_eq!(batch, vec![vec![DENSE]; 5]);
        for buffer in batch {
            pool.push(buffer).expect("room in the pool");
        }
        producer.join().expect("producer");
        assert!(
            dense_rx.recv().is_err() && sparse_rx.recv().is_err(),
            "the producer sends nothing more"
        );
    }

    /// A .bed image of one-byte rows that counts its reads and fails any read touching `bad_row`.
    struct CountingImage {
        image: BedImage,
        reads: AtomicU64,
        bad_row: Option<u64>,
    }

    impl ByteRangeSource for CountingImage {
        fn len(&self) -> u64 {
            self.image.len()
        }

        fn read_at(&self, offset: u64, dst: &mut [u8]) -> Result<(), PipelineError> {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if let Some(bad) = self.bad_row
                && offset <= 3 + bad
                && 3 + bad < offset + dst.len() as u64
            {
                return Err(PipelineError::Io(format!("row {bad} is unreadable")));
            }
            self.image.read_at(offset, dst)
        }
    }

    fn counting_image(rows: &[u8], bad_row: Option<u64>) -> Arc<CountingImage> {
        let mut bytes = vec![0x6c, 0x1b, 0x01];
        bytes.extend_from_slice(rows);
        Arc::new(CountingImage {
            image: BedImage(bytes),
            reads: AtomicU64::new(0),
            bad_row,
        })
    }

    /// Runs a producer over the `required` rows of `source` with every row on the dense path,
    /// giving the rows it sent, in order, and the text of the error it sent, if any.
    fn run_producer(
        source: Arc<CountingImage>,
        required: &[u64],
        total: u64,
        read_batch: usize,
    ) -> (Vec<Vec<u8>>, Option<String>) {
        let pool = empty_pool(required.len());
        for _ in 0..required.len() {
            pool.push(Vec::new()).expect("room in the pool");
        }
        let (dense_tx, dense_rx) = bounded(required.len() + 1);
        producer_thread_with_read_batch(
            source,
            prep_for_rows(required, total),
            None,
            dense_tx,
            pool,
            Arc::new(AtomicU64::new(0)),
            |_: &[u8]| ComputePath::Pivot,
            None,
            read_batch,
        );
        let mut sent = Vec::new();
        let mut error = None;
        for item in dense_rx.try_iter() {
            match item {
                Ok(work_item) => sent.push(work_item.data),
                Err(err) => error = Some(err.to_string()),
            }
        }
        (sent, error)
    }

    /// Reading adjacent rows in runs gives every row the bytes one read per row gives, in
    /// order, with far fewer reads: a long run, a run too short to share a read, a repeated
    /// row, a gap, and a run that reaches the last row.
    #[test]
    fn reading_rows_in_runs_gives_the_bytes_of_one_read_per_row() {
        let total = 300u64;
        let rows: Vec<u8> = (0..total).map(|row| (row * 7 % 251) as u8).collect();
        let mut required: Vec<u64> = (0..100).collect();
        required.extend(150..160);
        required.push(159);
        required.extend(200..300);
        let expected: Vec<Vec<u8>> = required.iter().map(|&row| vec![rows[row as usize]]).collect();
        for read_batch in [1, 16, 32, 1000] {
            let source = counting_image(&rows, None);
            let (sent, error) = run_producer(Arc::clone(&source), &required, total, read_batch);
            assert_eq!(error, None, "read_batch {read_batch}");
            assert_eq!(sent, expected, "read_batch {read_batch}");
            let reads = source.reads.load(Ordering::SeqCst) as usize;
            if read_batch == 1 {
                assert_eq!(reads, required.len());
            } else {
                assert!(reads * 4 < required.len(), "{reads} reads at read_batch {read_batch}");
            }
        }
    }

    /// A run that fails to read sends every row before the unreadable one and then the error,
    /// exactly as one read per row does.
    #[test]
    fn a_run_that_fails_to_read_sends_what_one_read_per_row_sends() {
        let total = 120u64;
        let rows: Vec<u8> = (0..total).map(|row| row as u8).collect();
        let required: Vec<u64> = (0..total).collect();
        let bad = 70u64;
        let one_per_row = run_producer(counting_image(&rows, Some(bad)), &required, total, 1);
        assert_eq!(one_per_row.0.len(), bad as usize);
        assert!(one_per_row.1.is_some(), "one read per row reports the unreadable row");
        for read_batch in [16, 32, 64, 1000] {
            let batched = run_producer(counting_image(&rows, Some(bad)), &required, total, read_batch);
            assert_eq!(batched, one_per_row, "read_batch {read_batch}");
        }
    }

    #[test]
    fn closing_the_pool_releases_a_parked_taker() {
        let pool = empty_pool(2);
        let (taker, taken) = spawn_taker(&pool, 1);
        wait_until_parked(&pool);
        pool.close();
        assert_eq!(taken.recv_timeout(PATIENCE).expect("the taker woke"), None);
        taker.join().expect("taker");
    }

    #[test]
    fn a_closed_pool_serves_nothing_more() {
        let pool = empty_pool(2);
        pool.push(vec![1u8; 3]).expect("room in the pool");
        pool.close();
        assert_eq!(pool.take(), None);
    }

    #[test]
    fn a_full_pool_hands_the_buffer_back() {
        let pool = empty_pool(1);
        pool.push(vec![1u8]).expect("room in the pool");
        assert_eq!(pool.push(vec![2u8]), Err(vec![2u8]));
    }

    #[test]
    fn concurrent_returns_never_strand_the_taker() {
        // Every buffer of a pool circulates between one taker and two returners, one
        // returning each buffer at once and one after yielding. Nothing rescues a missed
        // wake-up: the taker would stay parked and the test would time out.
        const ROWS: usize = 20_000;
        for buffers in [1usize, 2, 8] {
            let pool = empty_pool(buffers);
            for _ in 0..buffers {
                pool.push(vec![0u8; 8]).expect("room in the pool");
            }
            let (now_tx, now_rx) = bounded::<Vec<u8>>(buffers);
            let (later_tx, later_rx) = bounded::<Vec<u8>>(buffers);
            let (done_tx, done_rx) = bounded(1);
            let taker = {
                let pool = Arc::clone(&pool);
                thread::spawn(move || {
                    for row in 0..ROWS {
                        let buffer = pool.take().expect("the pool stays open");
                        let tx = if row % 3 == 0 { &later_tx } else { &now_tx };
                        tx.send(buffer).expect("returner alive");
                    }
                    done_tx.send(()).expect("test receiver");
                })
            };
            let returners: Vec<_> = [(now_rx, false), (later_rx, true)]
                .into_iter()
                .map(|(rx, delay)| {
                    let pool = Arc::clone(&pool);
                    thread::spawn(move || {
                        for (i, buffer) in rx.into_iter().enumerate() {
                            if delay && i % 7 == 0 {
                                thread::yield_now();
                            }
                            pool.push(buffer).expect("room in the pool");
                        }
                    })
                })
                .collect();
            done_rx
                .recv_timeout(PATIENCE)
                .unwrap_or_else(|_| panic!("every row was taken from a pool of {buffers}"));
            taker.join().expect("taker");
            for returner in returners {
                returner.join().expect("returner");
            }
        }
    }

    #[test]
    fn a_full_length_buffer_is_reused_without_a_zero_fill() {
        let stale = vec![0xA5u8; 16];
        let address = stale.as_ptr();
        let prepared = prepare_pooled_buffer(stale, 16).expect("buffer");
        assert_eq!(prepared.as_ptr(), address);
        assert_eq!(prepared, vec![0xA5u8; 16]);
    }

    #[test]
    fn a_buffer_of_another_length_is_zero_filled_to_the_row_width() {
        let fresh = prepare_pooled_buffer(Vec::with_capacity(16), 16).expect("fresh");
        assert_eq!(fresh, vec![0u8; 16]);
        let short = prepare_pooled_buffer(vec![9u8; 4], 16).expect("short");
        assert_eq!(short, vec![0u8; 16]);
        let long = prepare_pooled_buffer(vec![9u8; 20], 16).expect("long");
        assert_eq!(long, vec![0u8; 16]);
    }

    #[test]
    fn reused_full_length_buffers_carry_exactly_each_row() {
        // One stale buffer serves every row. Each time it comes back holding the previous
        // row it is reused without a zero fill, and must still carry the next row exactly.
        let rows = [0b1110_0100u8, 0b0000_0000, 0b1111_1111, 0b0110_1001];
        let pool = empty_pool(1);
        pool.push(vec![0xAA]).expect("room in the pool");
        let (dense_tx, dense_rx) = bounded(1);
        let producer = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                producer_thread(
                    bed_image(&rows),
                    four_person_prep(rows.len()),
                    None,
                    dense_tx,
                    pool,
                    Arc::new(AtomicU64::new(0)),
                    |_| ComputePath::Pivot,
                    None,
                )
            })
        };
        for (index, &row) in rows.iter().enumerate() {
            let item = dense_rx
                .recv_timeout(PATIENCE)
                .expect("a row")
                .expect("no producer error");
            assert_eq!(item.reconciled_variant_index.0 as usize, index);
            assert_eq!(item.data, vec![row]);
            pool.push(item.data).expect("room in the pool");
        }
        producer.join().expect("producer");
        assert!(dense_rx.recv().is_err(), "the producer sends nothing more");
    }

    /// Runs a producer against an empty pool, then closes the pool while it is parked.
    fn assert_producer_stops_when_its_pool_closes(
        producer: impl FnOnce(Arc<RowBufferPool>, Sender<Result<WorkItem, PipelineError>>)
        + Send
        + 'static,
    ) {
        let pool = empty_pool(1);
        let (dense_tx, dense_rx) = bounded(1);
        let (finished_tx, finished_rx) = bounded(1);
        let handle = {
            let pool = Arc::clone(&pool);
            thread::spawn(move || {
                producer(pool, dense_tx);
                finished_tx.send(()).expect("test receiver");
            })
        };
        wait_until_parked(&pool);
        pool.close();
        finished_rx
            .recv_timeout(PATIENCE)
            .expect("the producer returned");
        handle.join().expect("producer");
        assert!(dense_rx.try_recv().is_err(), "the producer sent nothing");
    }

    #[test]
    fn a_producer_stops_when_its_pool_closes_while_every_buffer_is_out() {
        assert_producer_stops_when_its_pool_closes(|pool, dense_tx| {
            producer_thread(
                bed_image(&[0, 0]),
                four_person_prep(2),
                None,
                dense_tx,
                pool,
                Arc::new(AtomicU64::new(0)),
                |_| ComputePath::Pivot,
                None,
            )
        });
    }

    #[test]
    fn a_multi_file_producer_stops_when_its_pool_closes_while_every_buffer_is_out() {
        assert_producer_stops_when_its_pool_closes(|pool, dense_tx| {
            let boundaries = [FilesetBoundary {
                bed_path: PathBuf::from("part1.bed"),
                bim_path: PathBuf::from("part1.bim"),
                fam_path: PathBuf::from("part1.fam"),
                starting_global_index: 0,
            }];
            let sources = [BedSource::from_byte_source(bed_image(&[0, 0]))];
            multi_file_producer_thread(
                four_person_prep(2),
                &boundaries,
                &sources,
                None,
                dense_tx,
                pool,
                Arc::new(AtomicU64::new(0)),
                |_| ComputePath::Pivot,
                None,
            )
        });
    }
}

/// Opens one scoring fileset with the local row indices it must serve.
pub fn open_bed_source_for_scoring(
    path: &std::path::Path,
    genome_build: Option<crate::adapt_plink2::GenomeBuild>,
    prep: &PreparationResult,
    memory_budget: crate::score::pipeline::MemoryBudget,
) -> Result<BedSource, PipelineError> {
    let (start, end) = match &prep.pipeline_kind {
        PipelineKind::SingleFile(_) => (0, prep.total_variants_in_bim),
        PipelineKind::MultiFile(boundaries) => {
            let index = boundaries
                .iter()
                .position(|boundary| boundary.bed_path == path)
                .ok_or_else(|| PipelineError::Io("Scoring path has no fileset boundary".into()))?;
            let next = boundaries
                .get(index + 1)
                .map_or(prep.total_variants_in_bim, |boundary| {
                    boundary.starting_global_index
                });
            (boundaries[index].starting_global_index, next)
        }
    };
    let rows: Vec<u64> = prep
        .required_bim_indices
        .iter()
        .map(|row| row.0)
        .filter(|&row| row >= start && row < end)
        .map(|row| row - start)
        .collect();
    crate::shared::files::open_bed_source_for_scoring(
        path,
        genome_build,
        &rows,
        prep.bytes_per_variant,
        end - start,
        local_prefetch_budget(prep, memory_budget)
            / match &prep.pipeline_kind {
                PipelineKind::SingleFile(_) => 1,
                PipelineKind::MultiFile(boundaries) => boundaries.len().max(1),
            },
    )
}

/// A cohort-wide cap, shared across filesets, included in RAM preflight.
pub fn local_prefetch_budget(
    prep: &PreparationResult,
    memory_budget: crate::score::pipeline::MemoryBudget,
) -> usize {
    if prep.bytes_per_variant >= 4096 && prep.required_bim_indices.len() >= 4096 {
        (memory_budget.max_ram_bytes() / 32).min(16 * 1024 * 1024)
    } else {
        0
    }
}

#[inline]
fn reconciled_index_from_usize(i: usize) -> Result<ReconciledVariantIndex, PipelineError> {
    let idx = u32::try_from(i).map_err(|_| {
        PipelineError::Compute(format!(
            "Reconciled variant index {i} exceeds u32::MAX; too many variants in one run."
        ))
    })?;
    Ok(ReconciledVariantIndex(idx))
}

pub struct SpoolPlan<'a> {
    pub is_complex_for_required: &'a [u8],
    pub compact_byte_index: &'a [u32],
    pub bytes_per_spooled_variant: u64,
    pub bytes_per_spooled_variant_usize: usize,
    pub scratch: Vec<u8>,
    pub file: &'a mut BufWriter<File>,
    pub offsets: &'a mut AHashMap<BimRowIndex, u64>,
    pub cursor: &'a mut u64,
}

/// The PLINK row buffers a producer fills and its consumers give back.
///
/// Consumers return buffers without taking a lock. A producer that finds the pool empty
/// parks until enough of the pool has come back, or the pool is closed. Parked, it costs no
/// CPU; woken, it has a run of rows to read before it could park again, so it is not
/// switched in once per returned row; and a consumer that stops early never leaves it
/// waiting for buffers that will not return.
///
/// Only a producer takes. While they wait for rows, consumers keep at most `consumer_hold`
/// buffers between them (a partly filled batch) and give every other buffer back, so a
/// parked producer always sees `capacity - consumer_hold` buffers return. It wakes at half
/// the pool or at that many, whichever is fewer. Should consumers keep more than that, a
/// parked producer takes whatever has come back once `patience` passes, so the excess costs
/// time and never a hang.
pub struct RowBufferPool {
    buffers: ArrayQueue<Vec<u8>>,
    /// How many buffers a parked producer waits for.
    wake_at: usize,
    /// How long a parked producer waits for `wake_at` buffers before it takes fewer.
    patience: Duration,
    /// Set by a producer about to park; cleared by the return that wakes it.
    waiting: AtomicBool,
    closed: AtomicBool,
    lock: Mutex<()>,
    returned: Condvar,
}

/// Long enough that a producer parked on consumers busy with a batch is not switched in
/// per returned row, and short next to a batch, should consumers keep more than declared.
const PARK_PATIENCE: Duration = Duration::from_millis(10);

impl RowBufferPool {
    /// An empty pool that holds at most `capacity` buffers, for consumers that keep at most
    /// `consumer_hold` of them between them while they wait for rows.
    pub fn new(capacity: usize, consumer_hold: usize) -> Self {
        Self::with_patience(capacity, consumer_hold, PARK_PATIENCE)
    }

    fn with_patience(capacity: usize, consumer_hold: usize, patience: Duration) -> Self {
        let capacity = capacity.max(1);
        Self {
            buffers: ArrayQueue::new(capacity),
            wake_at: capacity
                .div_ceil(2)
                .min(capacity.saturating_sub(consumer_hold))
                .max(1),
            patience,
            waiting: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            lock: Mutex::new(()),
            returned: Condvar::new(),
        }
    }

    /// Returns a buffer, waking a parked producer once half the pool is back. A full pool
    /// hands the buffer back.
    pub fn push(&self, buffer: Vec<u8>) -> Result<(), Vec<u8>> {
        self.buffers.push(buffer)?;
        // Loading first keeps the common case, no producer parked, free of shared writes,
        // and the swap lets exactly one return notify.
        if self.waiting.load(Ordering::SeqCst)
            && self.buffers.len() >= self.wake_at
            && self.waiting.swap(false, Ordering::SeqCst)
        {
            let _lock = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
            self.returned.notify_one();
        }
        Ok(())
    }

    /// Takes a buffer, parking while the pool is empty. Returns `None` once the pool is
    /// closed: whoever closed it reports why, and the producer only has to stop.
    pub fn take(&self) -> Option<Vec<u8>> {
        loop {
            if self.closed.load(Ordering::SeqCst) {
                return None;
            }
            if let Some(buffer) = self.buffers.pop() {
                return Some(buffer);
            }
            let mut lock = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
            // Announce the wait before the last look. A return that this look misses sees
            // the announcement, and can notify only once the wait has released the lock.
            self.waiting.store(true, Ordering::SeqCst);
            while self.waiting.load(Ordering::SeqCst)
                && self.buffers.len() < self.wake_at
                && !self.closed.load(Ordering::SeqCst)
            {
                let (relocked, wait) = self
                    .returned
                    .wait_timeout(lock, self.patience)
                    .unwrap_or_else(PoisonError::into_inner);
                lock = relocked;
                // Consumers kept more than they declared: take whatever has come back.
                if wait.timed_out() && !self.buffers.is_empty() {
                    break;
                }
            }
            self.waiting.store(false, Ordering::SeqCst);
            drop(lock);
        }
    }

    /// Stops every producer taking from this pool, now and later.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
        let _lock = self.lock.lock().unwrap_or_else(PoisonError::into_inner);
        self.returned.notify_all();
    }

    pub fn capacity(&self) -> usize {
        self.buffers.capacity()
    }
}

fn prepare_pooled_buffer(
    mut buffer: Vec<u8>,
    bytes_per_variant: usize,
) -> Result<Vec<u8>, PipelineError> {
    // Consumers return buffers at full length, and every `ByteRangeSource::read_at` fills
    // the whole slice or fails, so a returned buffer is reused without a zero fill.
    if buffer.len() == bytes_per_variant {
        return Ok(buffer);
    }
    buffer.clear();
    if buffer.capacity() < bytes_per_variant {
        // Reservation is relative to length, which clear() set to zero.
        buffer.try_reserve_exact(bytes_per_variant).map_err(|e| {
            PipelineError::Compute(format!(
                "Failed to reserve PLINK row buffer of {bytes_per_variant} bytes: {e}"
            ))
        })?;
    }
    buffer.resize(bytes_per_variant, 0);
    Ok(buffer)
}

/// The fewest adjacent rows read with one `read_at`. A source that decodes a long read in
/// parallel does so from sixteen whole rows on; shorter runs gain nothing from sharing a read.
pub const PARALLEL_READ_ROWS: usize = 16;

/// Reads a producer's required rows: one row per `read_at`, or, with a `read_batch` of at
/// least [`PARALLEL_READ_ROWS`], a run of adjacent rows with one `read_at` whose rows are then
/// handed out in order. A run that fails to read is read again one row at a time, so every
/// row gives the bytes and the error that a one-row read gives.
struct RowReader {
    read_batch: usize,
    /// The first row held in `bytes`, and how many rows it holds.
    first: u64,
    held: usize,
    bytes: Vec<u8>,
    /// Rows before this one are read one at a time, after a run that failed to read.
    single_until: u64,
}

impl RowReader {
    fn new(read_batch: usize) -> Self {
        Self {
            read_batch,
            first: 0,
            held: 0,
            bytes: Vec::new(),
            single_until: 0,
        }
    }

    /// Fills `dst`, one row wide, with required row `i`, which lies inside `source`.
    fn read(
        &mut self,
        source: &dyn ByteRangeSource,
        required: &[BimRowIndex],
        i: usize,
        dst: &mut [u8],
    ) -> Result<(), PipelineError> {
        let row = required[i].0;
        let width = dst.len();
        let row_bytes = width as u64;
        if row >= self.first && row < self.first + self.held as u64 {
            let start = (row - self.first) as usize * width;
            dst.copy_from_slice(&self.bytes[start..start + width]);
            return Ok(());
        }
        self.held = 0;
        if self.read_batch >= PARALLEL_READ_ROWS && row >= self.single_until {
            let mut run = 1usize;
            while run < self.read_batch
                && required
                    .get(i + run)
                    .is_some_and(|next| next.0 == row + run as u64)
                && 3 + (row + run as u64 + 1) * row_bytes <= source.len()
            {
                run += 1;
            }
            if run >= PARALLEL_READ_ROWS {
                self.bytes.resize(run * width, 0);
                if source.read_at(3 + row * row_bytes, &mut self.bytes).is_ok() {
                    self.first = row;
                    self.held = run;
                    dst.copy_from_slice(&self.bytes[..width]);
                    return Ok(());
                }
                self.single_until = row + run as u64;
            }
        }
        source.read_at(3 + row * row_bytes, dst)
    }
}

impl<'a> SpoolPlan<'a> {
    /// Whether the required variant at this position feeds the complex pass.
    #[inline(always)]
    pub fn spools(&self, variant_position: usize) -> bool {
        self.is_complex_for_required
            .get(variant_position)
            .copied()
            .unwrap_or(0)
            != 0
    }

    #[inline(always)]
    pub fn write_variant(
        &mut self,
        variant_position: usize,
        bim_row_idx: BimRowIndex,
        buffer: &[u8],
    ) -> Result<(), PipelineError> {
        if !self.spools(variant_position) {
            return Ok(());
        }

        let offset_for_variant = *self.cursor;
        self.offsets.insert(bim_row_idx, offset_for_variant);

        if self.bytes_per_spooled_variant_usize == 0 {
            return Ok(());
        }

        debug_assert_eq!(
            self.bytes_per_spooled_variant as usize, self.bytes_per_spooled_variant_usize,
            "cached spool byte counts must match"
        );
        debug_assert_eq!(
            self.scratch.len(),
            self.bytes_per_spooled_variant_usize,
            "scratch buffer must be sized to the spooled variant stride"
        );

        for (dst_idx, &orig_byte_idx) in self.compact_byte_index.iter().enumerate() {
            assert!(
                dst_idx < self.scratch.len(),
                "scratch index {} out of bounds for buffer of length {}",
                dst_idx,
                self.scratch.len()
            );
            let byte_index = orig_byte_idx as usize;
            assert!(
                byte_index < buffer.len(),
                "original byte index {} out of bounds for buffer of length {}",
                byte_index,
                buffer.len()
            );
            // Safety: debug assertion above guarantees the index is within bounds.
            self.scratch[dst_idx] = unsafe { *buffer.get_unchecked(byte_index) };
        }

        self.file
            .write_all(&self.scratch[..self.bytes_per_spooled_variant_usize])
            .map_err(|e| PipelineError::Io(format!("Failed to write complex spool: {e}")))?;

        *self.cursor += self.bytes_per_spooled_variant;
        Ok(())
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
/// * `sparse_tx`: Optional channel sender for variants destined for the sparse path.
/// * `dense_tx`: The channel sender for variants destined for the dense path.
/// * `buffer_pool`: A shared pool of reusable byte buffers to eliminate allocation overhead.
/// * `path_decider`: A closure that takes a variant's data and returns the `ComputePath`.
pub fn producer_thread<'a, F>(
    source: Arc<dyn ByteRangeSource>,
    prep_result: Arc<PreparationResult>,
    sparse_tx: Option<Sender<Result<WorkItem, PipelineError>>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<RowBufferPool>,
    variants_processed_count: Arc<AtomicU64>,
    path_decider: F,
    spool: Option<SpoolPlan<'a>>,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    producer_thread_with_read_batch(
        source,
        prep_result,
        sparse_tx,
        dense_tx,
        buffer_pool,
        variants_processed_count,
        path_decider,
        spool,
        1,
    );
}

/// As [`producer_thread`], reading a run of up to `read_batch` adjacent required rows with
/// one `read_at` when at least [`PARALLEL_READ_ROWS`] of them sit together, so a source that
/// decodes a long read on the rayon pool decodes the run's rows in parallel. Rows go out in
/// the same order, with the same bytes and the same first error, as one read per row gives.
#[allow(clippy::too_many_arguments)]
pub fn producer_thread_with_read_batch<'a, F>(
    source: Arc<dyn ByteRangeSource>,
    prep_result: Arc<PreparationResult>,
    sparse_tx: Option<Sender<Result<WorkItem, PipelineError>>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<RowBufferPool>,
    variants_processed_count: Arc<AtomicU64>,
    path_decider: F,
    mut spool: Option<SpoolPlan<'a>>,
    read_batch: usize,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    let send_error = |err: PipelineError| {
        if let Some(tx) = sparse_tx.as_ref() {
            let _ = tx.send(Err(err.clone()));
        }
        let _ = dense_tx.send(Err(err));
    };

    let bytes_per_variant = prep_result.bytes_per_variant as usize;
    let bytes_per_variant_u64 = prep_result.bytes_per_variant;
    let mut local_variants_processed: u64 = 0;
    let mut reader = RowReader::new(read_batch);

    match spool.as_mut() {
        Some(sp) => {
            let sp = sp;
            for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
                // A closed pool means a consumer stopped, and it reports why.
                let Some(pooled) = buffer_pool.take() else {
                    break;
                };
                let mut buffer = match prepare_pooled_buffer(pooled, bytes_per_variant) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        send_error(err);
                        break;
                    }
                };

                let offset = 3 + bim_row_idx.0 * bytes_per_variant_u64;
                let end = offset + bytes_per_variant_u64;

                if end > source.len() {
                    let err = PipelineError::Io(format!(
                        "Fatal: Attempted to read past the end of the .bed source for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                        bim_row_idx.0
                    ));
                    send_error(err);
                    break;
                }

                if let Err(err) = reader.read(
                    source.as_ref(),
                    &prep_result.required_bim_indices,
                    i,
                    buffer.as_mut_slice(),
                ) {
                    send_error(err);
                    break;
                }

                if let Err(err) = sp.write_variant(i, bim_row_idx, &buffer) {
                    send_error(err);
                    break;
                }

                let path = choose_score_path(&buffer, &prep_result, i, &path_decider);

                let reconciled_variant_index = match reconciled_index_from_usize(i) {
                    Ok(idx) => idx,
                    Err(err) => {
                        send_error(err);
                        break;
                    }
                };
                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index,
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    sparse_tx.as_ref().unwrap_or(&dense_tx)
                };

                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
        None => {
            for (i, &bim_row_idx) in prep_result.required_bim_indices.iter().enumerate() {
                // A closed pool means a consumer stopped, and it reports why.
                let Some(pooled) = buffer_pool.take() else {
                    break;
                };
                let mut buffer = match prepare_pooled_buffer(pooled, bytes_per_variant) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        send_error(err);
                        break;
                    }
                };

                let offset = 3 + bim_row_idx.0 * bytes_per_variant_u64;
                let end = offset + bytes_per_variant_u64;

                if end > source.len() {
                    let err = PipelineError::Io(format!(
                        "Fatal: Attempted to read past the end of the .bed source for variant at BIM row {}. The file may be truncated or inconsistent with the .bim file.",
                        bim_row_idx.0
                    ));
                    send_error(err);
                    break;
                }

                if let Err(err) = reader.read(
                    source.as_ref(),
                    &prep_result.required_bim_indices,
                    i,
                    buffer.as_mut_slice(),
                ) {
                    send_error(err);
                    break;
                }

                let path = choose_score_path(&buffer, &prep_result, i, &path_decider);

                let reconciled_variant_index = match reconciled_index_from_usize(i) {
                    Ok(idx) => idx,
                    Err(err) => {
                        send_error(err);
                        break;
                    }
                };
                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index,
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    sparse_tx.as_ref().unwrap_or(&dense_tx)
                };

                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
    }

    if local_variants_processed > 0 {
        variants_processed_count.fetch_add(local_variants_processed, Ordering::Relaxed);
    }
}

/// The producer for the multi-file pipeline. It seamlessly switches between memory-mapped
/// files as it iterates through the globally-indexed list of required variants.
pub fn multi_file_producer_thread<'a, F>(
    prep_result: Arc<PreparationResult>,
    boundaries: &[FilesetBoundary],
    bed_sources: &[BedSource],
    sparse_tx: Option<Sender<Result<WorkItem, PipelineError>>>,
    dense_tx: Sender<Result<WorkItem, PipelineError>>,
    buffer_pool: Arc<RowBufferPool>,
    variants_processed_count: Arc<AtomicU64>,
    path_decider: F,
    mut spool: Option<SpoolPlan<'a>>,
) where
    F: Fn(&[u8]) -> ComputePath,
{
    let send_error = |err: PipelineError| {
        if let Some(tx) = sparse_tx.as_ref() {
            let _ = tx.send(Err(err.clone()));
        }
        let _ = dense_tx.send(Err(err));
    };

    let mut current_fileset_idx: usize = 0;
    let bytes_per_variant = prep_result.bytes_per_variant;
    let mut local_variants_processed: u64 = 0;

    debug_assert_eq!(boundaries.len(), bed_sources.len());

    let mut current_source = bed_sources[0].byte_source();
    let mut next_boundary_start_idx = if boundaries.len() > 1 {
        boundaries[1].starting_global_index
    } else {
        u64::MAX
    };

    match spool.as_mut() {
        Some(sp) => {
            let sp = sp;
            for (i, &global_bim_row_index) in prep_result.required_bim_indices.iter().enumerate() {
                while global_bim_row_index.0 >= next_boundary_start_idx {
                    current_fileset_idx += 1;
                    current_source = bed_sources[current_fileset_idx].byte_source();
                    next_boundary_start_idx = if boundaries.len() > current_fileset_idx + 1 {
                        boundaries[current_fileset_idx + 1].starting_global_index
                    } else {
                        u64::MAX
                    };
                }

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
                    send_error(err);
                    return;
                }

                // A closed pool means a consumer stopped, and it reports why.
                let Some(pooled) = buffer_pool.take() else {
                    break;
                };
                let mut buffer = match prepare_pooled_buffer(pooled, bytes_per_variant as usize) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        send_error(err);
                        return;
                    }
                };

                if let Err(err) = current_source.read_at(offset, buffer.as_mut_slice()) {
                    send_error(err);
                    return;
                }

                if let Err(err) = sp.write_variant(i, global_bim_row_index, &buffer) {
                    send_error(err);
                    return;
                }

                let path = choose_score_path(&buffer, &prep_result, i, &path_decider);
                let reconciled_variant_index = match reconciled_index_from_usize(i) {
                    Ok(idx) => idx,
                    Err(err) => {
                        send_error(err);
                        return;
                    }
                };
                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index,
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    sparse_tx.as_ref().unwrap_or(&dense_tx)
                };
                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
        None => {
            for (i, &global_bim_row_index) in prep_result.required_bim_indices.iter().enumerate() {
                while global_bim_row_index.0 >= next_boundary_start_idx {
                    current_fileset_idx += 1;
                    current_source = bed_sources[current_fileset_idx].byte_source();
                    next_boundary_start_idx = if boundaries.len() > current_fileset_idx + 1 {
                        boundaries[current_fileset_idx + 1].starting_global_index
                    } else {
                        u64::MAX
                    };
                }

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
                    send_error(err);
                    return;
                }

                // A closed pool means a consumer stopped, and it reports why.
                let Some(pooled) = buffer_pool.take() else {
                    break;
                };
                let mut buffer = match prepare_pooled_buffer(pooled, bytes_per_variant as usize) {
                    Ok(buffer) => buffer,
                    Err(err) => {
                        send_error(err);
                        return;
                    }
                };

                if let Err(err) = current_source.read_at(offset, buffer.as_mut_slice()) {
                    send_error(err);
                    return;
                }

                let path = choose_score_path(&buffer, &prep_result, i, &path_decider);
                let reconciled_variant_index = match reconciled_index_from_usize(i) {
                    Ok(idx) => idx,
                    Err(err) => {
                        send_error(err);
                        return;
                    }
                };
                let work_item = WorkItem {
                    data: buffer,
                    reconciled_variant_index,
                };

                let tx = if path == ComputePath::Pivot {
                    &dense_tx
                } else {
                    sparse_tx.as_ref().unwrap_or(&dense_tx)
                };
                if tx.send(Ok(work_item)).is_err() {
                    break;
                }

                local_variants_processed += 1;
                if local_variants_processed == PROGRESS_UPDATE_BATCH_SIZE {
                    variants_processed_count
                        .fetch_add(PROGRESS_UPDATE_BATCH_SIZE, Ordering::Relaxed);
                    local_variants_processed = 0;
                }
            }
        }
    }

    if local_variants_processed > 0 {
        variants_processed_count.fetch_add(local_variants_processed, Ordering::Relaxed);
    }
}
