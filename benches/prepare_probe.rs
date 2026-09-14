// Compile against the warm baseline library and current preparation source.
pub use gnomon::{memory, output};
pub mod pipeline_error {
    pub use gnomon::pipeline_error::*;
}
#[path = "../score/prepare.rs"]
pub mod candidate_prepare;
pub mod score {
    pub use crate::candidate_prepare as prepare;
    pub use gnomon::score::{io, reformat, types};
}
use std::{path::PathBuf, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert_eq!(
        args.len(),
        2,
        "genotype prefix and normalized score file required"
    );
    for rep in 0..3 {
        let start = Instant::now();
        let before =
            gnomon::score::prepare::prepare_for_computation(&args[..1], &args[1..], None, None)
                .unwrap();
        let before_time = start.elapsed();
        let start = Instant::now();
        let after =
            score::prepare::prepare_for_computation(&args[..1], &args[1..], None, None).unwrap();
        let after_time = start.elapsed();
        assert_eq!(before.required_bim_indices, after.required_bim_indices);
        assert_eq!(before.sparse_row_offsets(), after.sparse_row_offsets());
        assert_eq!(before.sparse_score_columns(), after.sparse_score_columns());
        for (x, y) in before.sparse_weights().iter().zip(after.sparse_weights()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in before
            .sparse_missing_corrections()
            .iter()
            .zip(after.sparse_missing_corrections())
        {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        assert_eq!(before.score_variant_counts, after.score_variant_counts);
        assert_eq!(
            format!("{:?}", before.complex_rules),
            format!("{:?}", after.complex_rules)
        );
        assert_eq!(
            before.baseline_missing_sum_by_score(),
            after.baseline_missing_sum_by_score()
        );
        println!(
            "rep={rep} before_ms={:.3} after_ms={:.3} variants={}",
            before_time.as_secs_f64() * 1000.0,
            after_time.as_secs_f64() * 1000.0,
            after.num_reconciled_variants
        );
    }
}
