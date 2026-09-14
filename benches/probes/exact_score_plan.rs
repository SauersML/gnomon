//! Report exact-integer range requirements for real compiled score columns.
#[path = "../../score/exact.rs"]
mod exact;

use gnomon::score::prepare::prepare_for_computation;
use std::path::PathBuf;

fn main() {
    let args: Vec<PathBuf> = std::env::args_os().skip(1).map(PathBuf::from).collect();
    assert_eq!(args.len(), 2, "BED prefix, normalized score directory");
    let mut files: Vec<_> = std::fs::read_dir(&args[1])
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.file_name().unwrap().to_string_lossy()
            .ends_with("_hmPOS_GRCh38.gnomon.sorted.gnomon.tsv"))
        .collect();
    files.sort();
    let prep = prepare_for_computation(&args[..1], &files, None, None).unwrap();
    let mut fitted = 0;
    let mut batched = 0;
    for (column, name) in prep.score_names.iter().enumerate() {
        let entries = || prep.sparse_score_columns().iter().enumerate()
            .filter_map(|(index, &stored)| (stored as usize == column).then_some(index));
        let count = entries().count() as u64;
        let coefficients = entries().flat_map(|index| [
            prep.sparse_weights()[index], prep.sparse_missing_corrections()[index],
        ]);
        // Eight covers a mode-centred difference of any two correction+dosage terms.
        let fixed = exact::FixedPoint::plan(coefficients, count, 8, 1);
        if let Some(fixed) = fixed {
            let mut term_bits = 1;
            for index in entries() {
                let weight = fixed.to_fixed(prep.sparse_weights()[index]);
                let correction = fixed.to_fixed(prep.sparse_missing_corrections()[index]);
                let terms = [correction, 0, correction + weight, correction + 2 * weight];
                for &a in &terms {
                    for &b in &terms {
                        term_bits = term_bits.max(129 - (a - b).unsigned_abs().leading_zeros());
                    }
                }
            }
            let split = exact::Split::plan(term_bits, count);
            let batch_split = exact::Split::plan(term_bits, count.min(256));
            fitted += usize::from(split.is_some());
            batched += usize::from(batch_split.is_some());
            println!("score={name} simple_rows={count} exponent={} term_bits={term_bits} split_bits={:?} batch_256_split_bits={:?}",
                fixed.exp, split.map(|s| s.bits), batch_split.map(|s| s.bits));
        } else {
            println!("score={name} simple_rows={count} fixed_point_does_not_fit=true");
        }
    }
    println!("columns={} carry_free_columns={fitted} carry_free_batch_columns={batched} complex_rules={} scope=compiled_simple_rows",
        prep.score_names.len(), prep.complex_rules.len());
}
