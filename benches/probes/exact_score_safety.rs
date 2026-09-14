//! Standalone MSI checks: exact rounding oracle and allocation-failure injection.
#![feature(portable_simd)]
#[path = "../../score/exact.rs"]
mod exact;
#[path = "../../score/kernel_exact.rs"]
mod kernel;
#[cfg(test)]
mod score {
    pub(crate) use crate::exact;
}

fn main() {
    use std::io::{self, BufRead, Write};
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let mut fields = line.split_whitespace();
        let value: i128 = fields.next().unwrap().parse().unwrap();
        let exp = fields.next().unwrap().parse().unwrap();
        let scale = fields.next().unwrap().parse().unwrap();
        let divisor = fields.next().unwrap().parse().unwrap();
        assert!(fields.next().is_none());
        let plan = exact::FixedPoint { exp, scale };
        writeln!(out, "{:016x}", plan.quotient(value, divisor).to_bits()).unwrap();
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::{exact::Split, kernel::*};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static BUDGET: Cell<Option<usize>> = const { Cell::new(None) };
    }
    struct Allocator;
    fn permitted() -> bool {
        BUDGET
            .try_with(|budget| match budget.get() {
                None => true,
                Some(0) => false,
                Some(left) => {
                    budget.set(Some(left - 1));
                    true
                }
            })
            .unwrap_or(true)
    }
    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if permitted() {
                unsafe { System.alloc(layout) }
            } else {
                std::ptr::null_mut()
            }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if permitted() {
                unsafe { System.alloc_zeroed(layout) }
            } else {
                std::ptr::null_mut()
            }
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            if permitted() {
                unsafe { System.realloc(ptr, layout, size) }
            } else {
                std::ptr::null_mut()
            }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;

    const TABLE: RowCosts = RowCosts {
        table_row_ns: 0.0,
        word_ns: 1.0,
        exception_ns: 1.0,
        direct_call_ns: f64::INFINITY,
    };

    #[test]
    fn every_scratch_failure_leaves_output_untouched() {
        let geometry = TableGeometry::from_cache_sizes(32 << 10, 512 << 10);
        // Dense IDs, low tables, high tables, keys, and zero row: five allocations.
        for permitted_allocations in 0..5 {
            let mut scratch = KernelScratch::default();
            let (mut lo, mut hi, mut missing) = ([17; 33], [19; 33], [23; 33]);
            BUDGET.set(Some(permitted_allocations));
            let result = apply_rows(
                &[0xff; 9],
                9,
                &[0],
                &[[(2, 3); 4]],
                &TABLE,
                geometry,
                0,
                &mut scratch,
                &mut lo,
                &mut hi,
                &mut missing,
            );
            BUDGET.set(None);
            assert!(result.is_err(), "allocation {permitted_allocations}");
            assert_eq!(lo, [17; 33]);
            assert_eq!(hi, [19; 33]);
            assert_eq!(missing, [23; 33]);
        }
    }

    #[test]
    fn bounded_batches_reuse_scratch_and_match_integer_reference() {
        let (rows, people) = (2051usize, 97usize);
        let row_bytes = people.div_ceil(4usize);
        let data: Vec<u8> = (0..rows * row_bytes)
            .map(|i| (i * 79 + i / row_bytes) as u8)
            .collect();
        let exact: Vec<[i128; 4]> = (0..rows)
            .map(|r| {
                let w = (r as i128 - 1025) << 58;
                [w * 2, 0, -w, w]
            })
            .collect();
        let split = Split::plan(72, rows as u64).unwrap();
        let terms: Vec<TermLimbs> = exact.iter().map(|t| t.map(|v| split.parts(v))).collect();
        let ids: Vec<usize> = (0..rows).rev().collect();
        let (mut want, mut want_missing) = (vec![0i128; people], vec![0u32; people]);
        for r in 0..rows {
            for p in 0..people {
                let code = ((data[r * row_bytes + p / 4] >> (2 * (p % 4))) & 3) as usize;
                want[p] += exact[r][code];
                want_missing[p] += u32::from(code == 1);
            }
        }
        // Extreme cache hints are capped; no multiplication or allocation follows them unchecked.
        let geometry = TableGeometry {
            groups_per_batch: usize::MAX,
            tile_words: usize::MAX,
        };
        let mut scratch = KernelScratch::default();
        let (mut lo, mut hi, mut missing) = (vec![0; people], vec![0; people], vec![0; people]);
        for repeat in 0..3 {
            lo.fill(0);
            hi.fill(0);
            missing.fill(0);
            if repeat > 0 {
                BUDGET.set(Some(0));
            }
            let result = apply_rows(
                &data,
                row_bytes,
                &ids,
                &terms,
                &TABLE,
                geometry,
                0,
                &mut scratch,
                &mut lo,
                &mut hi,
                &mut missing,
            );
            BUDGET.set(None);
            result.unwrap();
            assert_eq!(missing, want_missing);
            for p in 0..people {
                assert_eq!(split.join(lo[p], hi[p]), want[p]);
            }
        }
    }
}
