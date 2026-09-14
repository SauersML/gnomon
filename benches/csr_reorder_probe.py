"""Build the actual CSR builders against warm MSI dependencies, including OOM injection.

Arguments: previous prepare.rs, candidate prepare.rs. The resulting executable
takes: baseline|candidate|failures BED-prefix normalized-score-directory identity|reverse.
"""
from pathlib import Path
import sys
from build_cached_probe import build, root

def builder(path):
    source = Path(path).read_text()
    start = source.index('#[derive(Debug, Copy, Clone)]\nstruct SimpleScoreAssignment')
    end = source.index('\n#[inline(always)]\nfn apply_simple_score_assignment', start)
    return source[start:end]

allocator = r'''
#![allow(dead_code)]
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering::SeqCst};
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
static FAIL_AFTER: AtomicIsize = AtomicIsize::new(-1);
struct Allocator;
fn added(size: usize) {
    let live = LIVE.fetch_add(size, SeqCst) + size;
    PEAK.fetch_max(live, SeqCst);
}
fn fail() -> bool {
    FAIL_AFTER.load(SeqCst) >= 0 && FAIL_AFTER.fetch_sub(1, SeqCst) == 0
}
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if fail() { return std::ptr::null_mut(); }
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() { added(layout.size()); }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), SeqCst);
        unsafe { System.dealloc(p, layout) };
    }
    unsafe fn realloc(&self, p: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if fail() { return std::ptr::null_mut(); }
        let next = unsafe { System.realloc(p, layout, size) };
        if !next.is_null() {
            if size >= layout.size() { added(size - layout.size()); }
            else { LIVE.fetch_sub(layout.size() - size, SeqCst); }
        }
        next
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;
'''

exercise = r'''
pub fn exercise(prep: &PreparationResult, reverse: bool) {
    let mut csr = CsrBuilder {
        sparse_weights: prep.sparse_weights().to_vec(),
        sparse_missing_corrections: prep.sparse_missing_corrections().to_vec(),
        sparse_score_columns: prep.sparse_score_columns().to_vec(),
        sparse_row_offsets: prep.sparse_row_offsets().to_vec(),
    };
    let mut required = prep.required_bim_indices.clone();
    let mut flags: Vec<u8> = (0..required.len()).map(|i| (i % 2) as u8).collect();
    if reverse { required.reverse(); }
    let before = crate::LIVE.load(crate::SeqCst);
    crate::PEAK.store(before, crate::SeqCst);
    let started = std::time::Instant::now();
    csr.sort_rows_by_bim_index(&mut required, &mut flags).unwrap();
    let elapsed = started.elapsed();
    let peak = crate::PEAK.load(crate::SeqCst).saturating_sub(before);
    assert_eq!(required, prep.required_bim_indices);
    for new_row in 0..required.len() {
        let old_row = if reverse { required.len() - 1 - new_row } else { new_row };
        assert_eq!(flags[new_row], (old_row % 2) as u8);
        let old = prep.sparse_row_offsets()[old_row] as usize..prep.sparse_row_offsets()[old_row + 1] as usize;
        let new = csr.sparse_row_offsets[new_row] as usize..csr.sparse_row_offsets[new_row + 1] as usize;
        assert_eq!(old.len(), new.len());
        for (a, b) in csr.sparse_weights[new.clone()].iter().zip(&prep.sparse_weights()[old.clone()]) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        for (a, b) in csr.sparse_missing_corrections[new.clone()].iter().zip(&prep.sparse_missing_corrections()[old.clone()]) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
        assert_eq!(csr.sparse_score_columns[new], prep.sparse_score_columns()[old]);
    }
    println!("rows={} nnz={} reverse={reverse} reorder_ms={:.3} peak_extra_bytes={peak} all_bits_match=true",
        required.len(), prep.sparse_weights().len(), elapsed.as_secs_f64() * 1000.0);
}
'''

failures = r'''
pub fn failures() {
    crate::FAIL_AFTER.store(0, crate::SeqCst);
    assert!(CsrBuilder::new().is_err());
    for allocation in 0..3 {
        let mut csr = CsrBuilder::new().unwrap();
        crate::FAIL_AFTER.store(allocation, crate::SeqCst);
        assert!(csr.push_contribution(ScoreColumnIndex(0), SimpleScoreAssignment {
            dosage_weight: 1.0, missing_correction: -0.0
        }).is_err());
        assert!(csr.sparse_weights.is_empty());
        assert!(csr.sparse_missing_corrections.is_empty());
        assert!(csr.sparse_score_columns.is_empty());
    }
    let mut csr = CsrBuilder::new().unwrap();
    crate::FAIL_AFTER.store(0, crate::SeqCst);
    assert!(csr.finish_variant().is_err());
    assert_eq!(csr.sparse_row_offsets, [0]);
    for allocation in 0..8 {
        let mut csr = CsrBuilder::new().unwrap();
        // Include an empty row, two entries in another row, and signed zero.
        csr.finish_variant().unwrap();
        csr.push_contribution(ScoreColumnIndex(0), SimpleScoreAssignment {
            dosage_weight: -0.0, missing_correction: 2.0
        }).unwrap();
        csr.push_contribution(ScoreColumnIndex(3), SimpleScoreAssignment {
            dosage_weight: 3.0, missing_correction: -0.0
        }).unwrap();
        csr.finish_variant().unwrap();
        let mut rows = vec![BimRowIndex(9), BimRowIndex(1)];
        let mut flags = vec![0, 1];
        crate::FAIL_AFTER.store(allocation, crate::SeqCst);
        let result = csr.sort_rows_by_bim_index(&mut rows, &mut flags);
        let remaining = crate::FAIL_AFTER.swap(-1, crate::SeqCst);
        if allocation < 7 {
            assert!(result.is_err());
            assert_eq!(remaining, -1);
        } else {
            assert!(result.is_ok());
            assert_eq!(remaining, 0);
            assert_eq!(csr.sparse_row_offsets, [0, 2, 2]);
            assert_eq!(csr.sparse_weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), [-0.0f32, 3.0].map(f32::to_bits));
            assert_eq!(csr.sparse_missing_corrections.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), [2.0f32, -0.0].map(f32::to_bits));
            assert_eq!(csr.sparse_score_columns, [0, 3]);
            assert_eq!(rows, [BimRowIndex(1), BimRowIndex(9)]);
            assert_eq!(flags, [1, 0]);
        }
    }
    let mut csr = CsrBuilder::new().unwrap();
    csr.finish_variant().unwrap();
    csr.finish_variant().unwrap();
    let mut ordered = vec![BimRowIndex(1), BimRowIndex(2)];
    let mut flags = vec![0; 2];
    crate::FAIL_AFTER.store(0, crate::SeqCst);
    let result = csr.sort_rows_by_bim_index(&mut ordered, &mut flags);
    let remaining = crate::FAIL_AFTER.swap(-1, crate::SeqCst);
    assert!(result.is_ok());
    assert_eq!(remaining, 0, "already ordered rows must allocate nothing");
    assert!(csr.sort_rows_by_bim_index(&mut vec![BimRowIndex(1); 2], &mut vec![0; 2]).is_err());
    assert!(csr.sort_rows_by_bim_index(&mut vec![], &mut vec![]).is_err());
    println!("CSR allocation failures returned errors; parallel lengths preserved on failed growth; invalid rows rejected");
}
'''

source = allocator
imports = 'use gnomon::score::{prepare::PrepError, types::{BimRowIndex, ScoreColumnIndex, PreparationResult}};\n'
for name, path in zip(['baseline', 'candidate'], sys.argv[1:]):
    source += 'mod ' + name + ' {\n' + imports + builder(path) + exercise
    if name == 'candidate':
        source += failures
    source += '}\n'
source += r'''
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args[0] == "failures" { candidate::failures(); return; }
    assert_eq!(args.len(), 4);
    let mut files: Vec<_> = std::fs::read_dir(&args[2]).unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.file_name().unwrap().to_string_lossy().ends_with("_hmPOS_GRCh38.gnomon.sorted.gnomon.tsv"))
        .collect();
    files.sort();
    let prep = gnomon::score::prepare::prepare_for_computation(&[args[1].clone().into()], &files, None, None).unwrap();
    let reverse = match args[3].as_str() { "identity" => false, "reverse" => true, _ => panic!("unknown order") };
    for _ in 0..3 {
        match args[0].as_str() {
            "baseline" => baseline::exercise(&prep, reverse),
            "candidate" => candidate::exercise(&prep, reverse),
            _ => panic!("unknown implementation"),
        }
    }
}
'''
probe = root / 'csr_reorder_probe.rs'
probe.write_text(source)
build(probe, 'csr-reorder-probe', ['-C', 'panic=abort'])
