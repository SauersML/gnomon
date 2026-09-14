// Compiled on MSI against the warm baseline library and the current source
// modules. Supply the cached pre-change gnomon library with rustc --extern.
#![feature(portable_simd)]
#[path = "../score/batch.rs"]
pub mod candidate_batch;
#[path = "../shared/genotype_table.rs"]
mod genotype_table;
pub mod score {
    pub mod types {
        pub use gnomon::score::types::*;
    }
    pub mod kernel {
        pub use gnomon::score::kernel::*;
    }
}
use gnomon::score::types::*;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;
const SIMD_LANES: usize = 8;

fn make_single_variant_multi_score_prep_result(n: usize, k: usize) -> PreparationResult {
    PreparationResult::new(
        vec![1.0; k],
        vec![0.0; k],
        (0..k as u32).collect(),
        vec![0, k as u64],
        k.div_ceil(SIMD_LANES) * SIMD_LANES,
        vec![0.0; k],
        vec![BimRowIndex(0)],
        vec![],
        (0..k).map(|i| format!("S{i}")).collect(),
        vec![1; k],
        PersonSubset::All,
        (0..n).map(|i| format!("I{i}")).collect(),
        n,
        n,
        1,
        1,
        n.div_ceil(4) as u64,
        (0..n as u32).map(|i| Some(OutputPersonIndex(i))).collect(),
        (0..n as u32).map(OriginalPersonIndex).collect(),
        vec![0],
        vec![0],
        vec![0],
        1,
        PipelineKind::SingleFile(PathBuf::from("benchmark")),
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
    for (n, k) in [
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
        let m = 256;
        let mut prep = make_single_variant_multi_score_prep_result(n, k);
        prep.bytes_per_variant = n.div_ceil(4) as u64;
        let data: Vec<u8> = (0..m * n.div_ceil(4))
            .map(|_| {
                let mut b = 0;
                for lane in 0..4 {
                    let r = next(&mut seed) % 10000;
                    let code = if r < 3500 {
                        0
                    } else if r < 8200 {
                        2
                    } else if r < 9800 {
                        3
                    } else {
                        1
                    };
                    b |= code << (lane * 2);
                }
                b
            })
            .collect();
        let weights: Vec<f64> = (0..m * prep.stride())
            .map(|i| (i % 31) as f64 / 128.0 - 0.125)
            .collect();
        let corrections: Vec<f64> = (0..weights.len()).map(|i| (i % 17) as f64 / 64.0).collect();
        let variants = vec![ReconciledVariantIndex(0); m];
        let pool = crossbeam_queue::ArrayQueue::new(1);
        let mut reference = None;
        for rep in 0..3 {
            for after in [false, true] {
                let mut scores = vec![0.0; n * k];
                let mut counts = vec![0u32; n * k];
                let start = Instant::now();
                if after {
                    candidate_batch::run_person_major_path(
                        black_box(&data),
                        &weights,
                        &corrections,
                        &variants,
                        &prep,
                        &mut scores,
                        &mut counts,
                        &pool,
                    )
                    .unwrap();
                } else {
                    gnomon::score::batch::run_person_major_path(
                        black_box(&data),
                        &weights,
                        &corrections,
                        &variants,
                        &prep,
                        &mut scores,
                        &mut counts,
                        &pool,
                    )
                    .unwrap();
                }
                println!(
                    "n={n} k={k} after={after} rep={rep} ms={:.3}",
                    start.elapsed().as_secs_f64() * 1000.0
                );
                if let Some((ref s, ref c)) = reference {
                    assert_eq!(&scores, s);
                    assert_eq!(&counts, c);
                } else {
                    reference = Some((scores, counts));
                }
            }
        }
    }
}
