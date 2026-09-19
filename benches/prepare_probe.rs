//! Times `prepare_for_computation` on a genotype prefix and a normalized score file, three
//! repetitions, and checks that every repetition prepares the same plan: the first compiles it
//! unless a saved plan exists, and the later ones load the plan it saved. An A/B builds this probe
//! at both revisions and runs each on the same inputs.
use gnomon::score::prepare::prepare_for_computation;
use gnomon::types::PreparationResult;
use std::{path::PathBuf, time::Instant};

fn assert_same_plan(first: &PreparationResult, again: &PreparationResult) {
    let bits = |values: Vec<f64>| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    assert_eq!(first.required_bim_indices, again.required_bim_indices);
    assert_eq!(first.sparse_row_offsets(), again.sparse_row_offsets());
    assert_eq!(first.sparse_score_columns(), again.sparse_score_columns());
    assert_eq!(bits(first.sparse_weights()), bits(again.sparse_weights()));
    assert_eq!(bits(first.sparse_missing_corrections()), bits(again.sparse_missing_corrections()));
    assert_eq!(first.score_variant_counts, again.score_variant_counts);
    assert_eq!(format!("{:?}", first.complex_rules), format!("{:?}", again.complex_rules));
    assert_eq!(bits(first.baseline_missing_sum_by_score()), bits(again.baseline_missing_sum_by_score()));
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert_eq!(args.len(), 2, "genotype prefix and normalized score file required");
    let mut first = None;
    for rep in 0..3 {
        let start = Instant::now();
        let prep = prepare_for_computation(&args[..1], &args[1..], None, None).unwrap();
        let elapsed = start.elapsed();
        if let Some(first) = &first {
            assert_same_plan(first, &prep);
        }
        println!(
            "rep={rep} ms={:.3} variants={}",
            elapsed.as_secs_f64() * 1000.0,
            prep.num_reconciled_variants
        );
        if first.is_none() {
            first = Some(prep);
        }
    }
}
