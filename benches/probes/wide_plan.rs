//! Standalone MSI full-marker comparison against the pre-cache preparation source kept in the
//! MSI iteration snapshot. Inputs are the existing 32 normalized PGS files.
pub use gnomon::{pipeline_error, score};
#[path = "/projects/standard/hsiehph/sauer354/gnomon/target/score-map/round5-before-src/score/prepare.rs"]
pub mod before_prepare;

use std::{path::PathBuf, sync::Arc, time::Instant};

fn main() {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert!((2..=3).contains(&args.len()));
    let repetitions = args.get(2).map(|s| s.to_str().unwrap().parse::<usize>().unwrap()).unwrap_or(3);
    let mut files: Vec<_> = std::fs::read_dir(&args[1]).unwrap().map(|entry| entry.unwrap().path())
        .filter(|path| path.file_name().unwrap().to_string_lossy().ends_with("_hmPOS_GRCh38.gnomon.sorted.gnomon.tsv")).collect();
    files.sort();
    assert_eq!(files.len(), 32);
    for rep in 0..repetitions {
        let start = Instant::now();
        let before = before_prepare::prepare_for_computation(&args[..1], &files, None, None).unwrap();
        let before_time = start.elapsed();
        let start = Instant::now();
        let after = score::prepare::prepare_for_computation(&args[..1], &files, None, None).unwrap();
        let after_time = start.elapsed();
        assert_eq!(before.required_bim_indices, after.required_bim_indices);
        assert_eq!(before.sparse_row_offsets(), after.sparse_row_offsets());
        assert_eq!(before.sparse_score_columns(), after.sparse_score_columns());
        assert_eq!(before.sparse_weights().len(), after.sparse_weights().len());
        for (x, y) in before.sparse_weights().iter().zip(after.sparse_weights()) { assert_eq!(x.to_bits(), y.to_bits()); }
        for (x, y) in before.sparse_missing_corrections().iter().zip(after.sparse_missing_corrections()) { assert_eq!(x.to_bits(), y.to_bits()); }
        assert_eq!(before.baseline_missing_sum_by_score(), after.baseline_missing_sum_by_score());
        assert_eq!(before.score_variant_counts, after.score_variant_counts);
        assert_eq!(format!("{:?}", before.complex_rules), format!("{:?}", after.complex_rules));
        let before_context = score::pipeline::PipelineContext::new(Arc::new(before));
        let after_context = score::pipeline::PipelineContext::new(Arc::new(after));
        let before_output = score::pipeline::run(&before_context).unwrap();
        let start = Instant::now();
        let after_output = score::pipeline::run(&after_context).unwrap();
        let compute = start.elapsed();
        assert_eq!(before_output, after_output);
        println!("rep={rep} before_ms={:.3} after_ms={:.3} compute_ms={:.3} people={} scores={} matched={} nnz={}",
            before_time.as_secs_f64()*1000.0, after_time.as_secs_f64()*1000.0, compute.as_secs_f64()*1000.0,
            after_context.prep_result.num_people_to_score, after_context.prep_result.score_names.len(),
            after_context.prep_result.num_reconciled_variants, after_context.prep_result.sparse_weights().len());
    }
}
