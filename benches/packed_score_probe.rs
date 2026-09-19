//! Times the dense packed kernel, `run_dense_batch`, on one batch of synthetic rows for a grid of
//! cohort sizes and score counts, three repetitions each, and checks that every repetition gives
//! the same cells and missing counts. An A/B builds this probe at both revisions.
use gnomon::batch::{DenseScratch, PersonLayout, run_dense_batch};
use gnomon::score::cells::ExactPlan;
use gnomon::types::{
    BimRowIndex, OriginalPersonIndex, OutputPersonIndex, PersonSubset, PipelineKind, PreparationResult,
    ReconciledVariantIndex,
};
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

/// Rows in the batch.
const ROWS: usize = 256;

/// Every score weighs every row, and no weight is flipped.
fn plan(people: usize, scores: usize) -> PreparationResult {
    let entries = ROWS * scores;
    let weights: Vec<f64> = (0..entries).map(|i| (i % 31) as f64 / 128.0 - 0.125).collect();
    let columns: Vec<u32> = (0..ROWS).flat_map(|_| 0..scores as u32).collect();
    let offsets: Vec<u64> = (0..=ROWS).map(|row| (row * scores) as u64).collect();
    let names: Vec<String> = (0..scores).map(|score| format!("S{score}")).collect();
    let exact = ExactPlan::new(weights, &vec![0.0; entries], &columns, &offsets, &[], &names)
        .expect("probe weights have an exact plan");
    let row_bytes = people.div_ceil(4);
    PreparationResult::new(
        exact,
        columns,
        offsets,
        (0..ROWS as u64).map(BimRowIndex).collect(),
        Vec::new(),
        names,
        vec![ROWS as u32; scores],
        PersonSubset::All,
        (0..people).map(|person| format!("I{person}")).collect(),
        people,
        people,
        ROWS as u64,
        ROWS,
        row_bytes as u64,
        (0..people as u32).map(|person| Some(OutputPersonIndex(person))).collect(),
        (0..people as u32).map(OriginalPersonIndex).collect(),
        vec![0; ROWS],
        (0..row_bytes as u32).collect(),
        (0..row_bytes as i32).collect(),
        row_bytes as u64,
        PipelineKind::SingleFile(PathBuf::from("probe")),
    )
}

fn next(seed: &mut u64) -> u64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    *seed
}

fn main() {
    let mut seed = 42;
    for (people, scores) in [
        (1usize, 9usize),
        (32, 9),
        (1024, 9),
        (50000, 1),
        (50000, 2),
        (50000, 3),
        (50000, 4),
        (50000, 9),
        (50000, 64),
        (250000, 9),
    ] {
        let prep = plan(people, scores);
        // Calls 00, 10, 11 and missing in about 35%, 47%, 16% and 2% of people.
        let data: Vec<u8> = (0..ROWS * people.div_ceil(4))
            .map(|_| {
                (0..4).fold(0u8, |byte, slot| {
                    let draw = next(&mut seed) % 10000;
                    let code = match draw {
                        0..3500 => 0,
                        3500..8200 => 2,
                        8200..9800 => 3,
                        _ => 1,
                    };
                    byte | code << (2 * slot)
                })
            })
            .collect();
        let rows: Vec<ReconciledVariantIndex> = (0..ROWS as u32).map(ReconciledVariantIndex).collect();
        let layout = PersonLayout::new(&prep);
        let mut scratch = DenseScratch::default();
        let mut first = None;
        for rep in 0..3 {
            let mut cells = vec![0i64; people * prep.exact().stride()];
            let mut counts = vec![0u32; people * scores];
            let start = Instant::now();
            run_dense_batch(black_box(&data), &rows, &prep, &layout, &mut scratch, &mut cells, &mut counts)
                .expect("the probe's buffers fit its plan");
            println!(
                "n={people} k={scores} rep={rep} ms={:.3}",
                start.elapsed().as_secs_f64() * 1000.0
            );
            match &first {
                Some((first_cells, first_counts)) => {
                    assert_eq!(&cells, first_cells);
                    assert_eq!(&counts, first_counts);
                }
                None => first = Some((cells, counts)),
            }
        }
    }
}
