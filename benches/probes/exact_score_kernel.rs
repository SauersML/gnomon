//! Compare exact packed-kernel regimes on every simple row of a real score.
#![feature(portable_simd)]
#[path = "../../score/exact.rs"]
mod exact;
#[path = "../../score/kernel_exact.rs"]
mod kernel;

use gnomon::score::prepare::prepare_for_computation;
use std::{fs::File, hint::black_box, path::PathBuf, time::Instant};

const BATCH: usize = 256;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    assert_eq!(
        args.len(),
        4,
        "BED prefix, score directory, score name, people"
    );
    let prefix = PathBuf::from(&args[0]);
    let people: usize = args[3].parse().unwrap();
    let files: Vec<_> = std::fs::read_dir(&args[1])
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            let name = path.file_name().unwrap().to_string_lossy();
            name.starts_with(&format!("{}_", args[2]))
                && name.ends_with("_hmPOS_GRCh38.gnomon.sorted.gnomon.tsv")
        })
        .collect();
    assert_eq!(files.len(), 1);
    let prep = prepare_for_computation(std::slice::from_ref(&prefix), &files, None, None).unwrap();
    assert_eq!(prep.score_names, [args[2].clone()]);
    assert!(people > 0 && people <= prep.total_people_in_fam);
    let file = File::open(prefix.with_extension("bed")).unwrap();
    let mapped = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
    assert_eq!(&mapped[..3], &[0x6c, 0x1b, 0x01]);
    let physical_stride = prep.bytes_per_variant as usize;
    let packed_stride = people.div_ceil(4);
    let mut rows = Vec::new();
    for (row, &bim) in prep.required_bim_indices.iter().enumerate() {
        for entry in prep
            .variant_csr_view(gnomon::score::types::ReconciledVariantIndex(row as u32))
            .iter()
        {
            rows.push((bim.0 as usize, entry.weight, entry.missing_correction));
        }
    }
    assert!(!rows.is_empty());
    let fixed = exact::FixedPoint::plan(
        rows.iter().flat_map(|&(_, w, c)| [w, c]),
        rows.len() as u64,
        8,
        1,
    )
    .expect("score fits exact i128");
    let coefficients: Vec<_> = rows
        .iter()
        .map(|&(_, w, c)| (fixed.to_fixed(w), fixed.to_fixed(c)))
        .collect();
    let mut bits = 1;
    for &(w, c) in &coefficients {
        let terms = [c, 0, c + w, c + 2 * w];
        for &a in &terms {
            for &b in &terms {
                bits = bits.max(129 - (a - b).unsigned_abs().leading_zeros());
            }
        }
    }
    let split = exact::Split::plan(bits, BATCH as u64).expect("batch fits two limbs");
    let geometry = kernel::TableGeometry::from_cache_sizes(32 << 10, 512 << 10);
    let direct = kernel::RowCosts {
        table_row_ns: f64::INFINITY,
        word_ns: f64::INFINITY,
        exception_ns: 0.0,
        direct_call_ns: 0.0,
    };
    let walk = kernel::RowCosts {
        table_row_ns: f64::INFINITY,
        word_ns: 0.0,
        exception_ns: 0.0,
        direct_call_ns: f64::INFINITY,
    };
    let table = kernel::RowCosts {
        table_row_ns: 0.0,
        word_ns: 1.0,
        exception_ns: 1.0,
        direct_call_ns: f64::INFINITY,
    };
    let mut data = vec![0u8; BATCH * packed_stride];
    let mut terms = vec![[(0, 0); 4]; BATCH];
    let ids: Vec<_> = (0..BATCH).collect();
    let mut scratch = kernel::KernelScratch::default();
    let mut lo = vec![0i64; people];
    let mut hi = vec![0i64; people];
    let mut totals = vec![0i128; people];
    let mut missing = vec![0u32; people];
    let mut reference = None;
    for repetition in 0..2 {
        let paths = if repetition == 0 {
            [("direct", &direct), ("walk", &walk), ("table", &table)]
        } else {
            [("table", &table), ("walk", &walk), ("direct", &direct)]
        };
        for (name, costs) in paths {
            totals.fill(0);
            missing.fill(0);
            let mut kernel_seconds = 0.0;
            let started = Instant::now();
            for first in (0..rows.len()).step_by(BATCH) {
                let count = (rows.len() - first).min(BATCH);
                for local in 0..count {
                    let offset = 3 + rows[first + local].0 * physical_stride;
                    data[local * packed_stride..(local + 1) * packed_stride]
                        .copy_from_slice(&mapped[offset..offset + packed_stride]);
                    let (w, c) = coefficients[first + local];
                    terms[local] = [c, 0, c + w, c + 2 * w].map(|value| split.parts(value));
                }
                lo.fill(0);
                hi.fill(0);
                let start = Instant::now();
                kernel::apply_rows(
                    black_box(&data[..count * packed_stride]),
                    packed_stride,
                    &ids[..count],
                    &terms[..count],
                    costs,
                    geometry,
                    0,
                    &mut scratch,
                    &mut lo,
                    &mut hi,
                    &mut missing,
                )
                .unwrap();
                kernel_seconds += start.elapsed().as_secs_f64();
                for ((total, &l), &h) in totals.iter_mut().zip(&lo).zip(&hi) {
                    *total += split.join(l, h);
                }
            }
            let elapsed = started.elapsed();
            if let Some((expected, counts)) = &reference {
                assert_eq!(&totals, expected, "exact sums: {name}");
                assert_eq!(&missing, counts, "missing counts: {name}");
            } else {
                reference = Some((totals.clone(), missing.clone()));
            }
            assert!(totals.iter().all(|&value| fixed.to_f64(value).is_finite()));
            println!(
                "score={} people={people} rows={} path={name} rep={repetition} total_ms={:.3} kernel_ms={:.3} exact_match=true complex_rules_excluded={}",
                args[2],
                rows.len(),
                elapsed.as_secs_f64() * 1000.0,
                kernel_seconds * 1000.0,
                prep.complex_rules.len()
            );
        }
    }
}
