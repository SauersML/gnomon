//! Compare real wide-panel batches without a full biobank scoring run.
#![feature(portable_simd)]
#[path = "../../score/batch.rs"]
pub mod candidate_batch;
#[allow(dead_code)]
#[path = "../../shared/genotype_table.rs"]
pub mod genotype_table;
pub mod score {
    pub use gnomon::score::{kernel, types};
}
use crossbeam_queue::ArrayQueue;
use gnomon::score::{
    prepare,
    types::{EffectAlleleDosage, ReconciledVariantIndex},
};
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    time::Instant,
};

fn main() {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert_eq!(args.len(), 2);
    let mut files: Vec<_> = std::fs::read_dir(&args[1])
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .ends_with("_hmPOS_GRCh38.gnomon.sorted.gnomon.tsv")
        })
        .collect();
    files.sort();
    assert_eq!(files.len(), 32);
    let prep = prepare::prepare_for_computation(&args[..1], &files, None, None).unwrap();
    let mut bed = File::open(args[0].with_extension("bed")).unwrap();
    let row_bytes = prep.bytes_per_variant as usize;
    let columns = prep.score_names.len();
    let pool: ArrayQueue<Vec<EffectAlleleDosage>> = ArrayQueue::new(1);
    for start in [
        0,
        prep.num_reconciled_variants / 3,
        prep.num_reconciled_variants * 2 / 3,
    ] {
        let start = start as usize;
        let indices: Vec<_> = (start..start + 256)
            .map(|i| ReconciledVariantIndex(i as u32))
            .collect();
        let mut data = vec![0; row_bytes * 256];
        let mut weights = vec![0.0f64; prep.stride() * 256];
        let mut corrections = weights.clone();
        let mut nnz = 0;
        for (row, &index) in indices.iter().enumerate() {
            bed.seek(SeekFrom::Start(
                3 + prep.required_bim_indices[index.0 as usize].0 * row_bytes as u64,
            ))
            .unwrap();
            bed.read_exact(&mut data[row * row_bytes..(row + 1) * row_bytes])
                .unwrap();
            for entry in prep.variant_csr_view(index).iter() {
                weights[row * prep.stride() + entry.score_column.0] = entry.weight;
                corrections[row * prep.stride() + entry.score_column.0] = entry.missing_correction;
                nnz += 1;
            }
        }
        let mut old_scores = vec![0.0f64; prep.num_people_to_score * columns];
        let mut new_scores = old_scores.clone();
        let mut old_counts = vec![0u32; old_scores.len()];
        let mut new_counts = old_counts.clone();
        for rep in 0..3 {
            old_scores.fill(0.0);
            new_scores.fill(0.0);
            old_counts.fill(0);
            new_counts.fill(0);
            let start_time = Instant::now();
            gnomon::score::batch::run_person_major_path(
                &data,
                &weights,
                &corrections,
                &indices,
                &prep,
                &mut old_scores,
                &mut old_counts,
                &pool,
            )
            .unwrap();
            let before = start_time.elapsed();
            let start_time = Instant::now();
            candidate_batch::run_person_major_path(
                &data,
                &weights,
                &corrections,
                &indices,
                &prep,
                &mut new_scores,
                &mut new_counts,
                &pool,
            )
            .unwrap();
            let after = start_time.elapsed();
            assert_eq!(old_scores, new_scores);
            assert_eq!(old_counts, new_counts);
            println!(
                "start={start} rep={rep} people={} nnz={nnz} before_ms={:.3} after_ms={:.3}",
                prep.num_people_to_score,
                before.as_secs_f64() * 1000.0,
                after.as_secs_f64() * 1000.0
            );
        }
    }
}
