#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "consumed by the exact score accumulators once they land"
    )
)]

// Exact score kernels over packed PLINK rows, for one output-person range.
//
// A range starts on a 32-person boundary (a whole 64-bit word of calls) and its cells are
// two carry-free i64 limbs per person (see `exact::Split`). Each term enters a cell as an
// integer, so tables, walks, tile sizes and the order of rows cannot change any bit.

use std::collections::TryReserveError;
use std::simd::{Simd, num::SimdUint};

/// Four variants per table: a person's four two-bit calls form the table key.
pub(crate) const VARIANTS_PER_TABLE: usize = 4;

/// Limb terms for PLINK codes [00, 01, 10, 11] of one variant and score.
pub(crate) type TermLimbs = [(i64, i64); 4];

/// Entry `key` = the sum over the four variants of `terms[v][code of v in key]`. Prefix
/// expansion shares partial sums: 4 + 12 + 48 + 192 additions instead of 4 per entry.
#[inline]
pub(crate) fn build_table(
    terms: [&TermLimbs; VARIANTS_PER_TABLE],
    lo: &mut [i64; 256],
    hi: &mut [i64; 256],
) {
    for code in 0..4 {
        (lo[code], hi[code]) = terms[0][code];
    }
    for variant in 1..VARIANTS_PER_TABLE {
        let prefix = 1 << (2 * variant);
        for code in (1..4).rev() {
            let (add_lo, add_hi) = terms[variant][code];
            for entry in 0..prefix {
                lo[code * prefix + entry] = lo[entry].wrapping_add(add_lo);
                hi[code * prefix + entry] = hi[entry].wrapping_add(add_hi);
            }
        }
        let (add_lo, add_hi) = terms[variant][0];
        for entry in 0..prefix {
            lo[entry] = lo[entry].wrapping_add(add_lo);
            hi[entry] = hi[entry].wrapping_add(add_hi);
        }
    }
}

/// Table keys for 32 people per step: the butterfly transpose of four row words on eight u32
/// lanes. `rows[k]` are whole 8-byte chunks of the same byte range; `keys.len() == 4 * rows[k].len()`.
#[inline]
pub(crate) fn transpose_keys(rows: [&[u8]; VARIANTS_PER_TABLE], keys: &mut [u8]) {
    let lane = |chunk: &[u8]| Simd::<u8, 8>::from_slice(chunk).cast::<u32>();
    for ((((out, c0), c1), c2), c3) in keys
        .chunks_exact_mut(32)
        .zip(rows[0].chunks_exact(8))
        .zip(rows[1].chunks_exact(8))
        .zip(rows[2].chunks_exact(8))
        .zip(rows[3].chunks_exact(8))
    {
        let mut word = lane(c0) | (lane(c1) << 8) | (lane(c2) << 16) | (lane(c3) << 24);
        let swap = (word ^ (word >> 6)) & Simd::splat(0x00cc_00cc);
        word ^= swap ^ (swap << 6);
        let swap = (word ^ (word >> 12)) & Simd::splat(0x0000_f0f0);
        word ^= swap ^ (swap << 12);
        for (dst, value) in out.chunks_exact_mut(4).zip(word.to_array()) {
            dst.copy_from_slice(&value.to_le_bytes());
        }
    }
}

/// One table over a tile of people: a load per person per limb, no carries.
#[inline]
pub(crate) fn apply_table(
    lo_table: &[i64; 256],
    hi_table: &[i64; 256],
    keys: &[u8],
    lo: &mut [i64],
    hi: &mut [i64],
) {
    for ((l, h), &key) in lo.iter_mut().zip(hi.iter_mut()).zip(keys) {
        *l = l.wrapping_add(lo_table[key as usize]);
        *h = h.wrapping_add(hi_table[key as usize]);
    }
}

/// Reusable per-thread scratch for the table path.
#[derive(Default)]
pub(crate) struct TableScratch {
    lo_tables: Vec<[i64; 256]>,
    hi_tables: Vec<[i64; 256]>,
    keys: Vec<u8>,
    zero_tile: Vec<u8>,
}

// Fixed scratch ceilings, independent of the cohort and total variant count.
const MAX_TABLE_GROUPS: usize = 256;
const MAX_TILE_WORDS: usize = 256;
const MAX_DENSE_ROWS: usize = MAX_TABLE_GROUPS * VARIANTS_PER_TABLE;

fn resize_scratch<T: Clone>(
    values: &mut Vec<T>,
    count: usize,
    zero: T,
) -> Result<(), TryReserveError> {
    if count > values.len() {
        values.try_reserve_exact(count - values.len())?;
        values.resize(count, zero);
    }
    Ok(())
}

impl TableScratch {
    /// Reserve everything before changing any output cell. Repeated batches reuse it.
    fn prepare(&mut self, groups: usize, tile_words: usize) -> Result<(), TryReserveError> {
        resize_scratch(&mut self.lo_tables, groups, [0; 256])?;
        resize_scratch(&mut self.hi_tables, groups, [0; 256])?;
        resize_scratch(&mut self.keys, tile_words * 32, 0)?;
        resize_scratch(&mut self.zero_tile, tile_words * 8, 0)
    }
}

/// Cache-derived table geometry: tables for a batch of groups fill about half of L2; a tile's
/// cells, keys and row bytes fill about half of L1.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TableGeometry {
    pub(crate) groups_per_batch: usize,
    pub(crate) tile_words: usize,
}

impl TableGeometry {
    pub(crate) fn from_cache_sizes(l1_bytes: usize, l2_bytes: usize) -> Self {
        let groups_per_batch = (l2_bytes / 2 / (256 * 16)).clamp(4, MAX_TABLE_GROUPS);
        // Per person: 16 B of cells, 1 B of key, plus the four row bytes it shares with 3 others.
        let tile_words = (l1_bytes / 2 / (32 * (16 + 1) + 4 * 8)).clamp(1, MAX_TILE_WORDS);
        Self {
            groups_per_batch,
            tile_words,
        }
    }

    fn for_work(self, rows: usize, people: usize) -> Self {
        Self {
            groups_per_batch: self
                .groups_per_batch
                .clamp(1, MAX_TABLE_GROUPS)
                .min(rows.div_ceil(VARIANTS_PER_TABLE)),
            tile_words: self
                .tile_words
                .clamp(1, MAX_TILE_WORDS)
                .min(people.div_ceil(32)),
        }
    }
}

const M55: u64 = 0x5555_5555_5555_5555;
/// The 64-bit pattern of a row whose every call is the code `c` (00, 01, 10 or 11).
const CODE_PATTERN: [u64; 4] = [0, M55, 0xaaaa_aaaa_aaaa_aaaa, u64::MAX];

/// Little-endian 64-bit word `w` of a packed row; bytes past the row read as code 00.
#[inline(always)]
fn row_word(row: &[u8], w: usize) -> u64 {
    let start = w * 8;
    if start + 8 <= row.len() {
        u64::from_le_bytes(row[start..start + 8].try_into().unwrap())
    } else {
        let mut bytes = [0u8; 8];
        bytes[..row.len() - start].copy_from_slice(&row[start..]);
        u64::from_le_bytes(bytes)
    }
}

/// The row's most common non-missing code over people `0..people` (ties prefer the lower code),
/// and how many calls differ from it.
#[inline]
pub(crate) fn row_mode(row: &[u8], people: usize) -> (u8, u64) {
    let words = people.div_ceil(32);
    let (mut c11, mut c10, mut c01) = (0u64, 0u64, 0u64);
    for w in 0..words {
        let mut x = row_word(row, w);
        if w + 1 == words && people % 32 != 0 {
            x &= (1u64 << (2 * (people % 32))) - 1;
        }
        let (low, high) = (x & M55, (x >> 1) & M55);
        c11 += u64::from((low & high).count_ones());
        c10 += u64::from((high & !low).count_ones());
        c01 += u64::from((low & !high).count_ones());
    }
    let counts = [people as u64 - c11 - c10 - c01, c01, c10, c11];
    let mode = [0u8, 2, 3]
        .into_iter()
        .max_by_key(|&c| (counts[c as usize], 3 - c))
        .unwrap();
    (mode, people as u64 - counts[mode as usize])
}

/// Adds one row's calls for people `first_person..first_person + lo.len()` as mode-centred
/// terms: each person who differs from `mode` receives `adjust[code]`, and every person's share
/// of `terms[mode]` is left to the caller, who adds it once per range. Missing calls count.
#[inline]
pub(crate) fn walk_row(
    row: &[u8],
    mode: u8,
    adjust: &TermLimbs,
    first_person: usize,
    lo: &mut [i64],
    hi: &mut [i64],
    missing: &mut [u32],
) {
    let people = lo.len();
    let pattern = CODE_PATTERN[mode as usize];
    let first_word = first_person / 32;
    let words = people.div_ceil(32);
    for w in 0..words {
        let x = row_word(row, first_word + w);
        let diff = x ^ pattern;
        let mut exceptions = (diff | (diff >> 1)) & M55;
        if w + 1 == words && people % 32 != 0 {
            exceptions &= (1u64 << (2 * (people % 32))) - 1;
        }
        while exceptions != 0 {
            let bit = exceptions.trailing_zeros();
            exceptions &= exceptions - 1;
            let code = ((x >> bit) & 3) as usize;
            let person = w * 32 + (bit / 2) as usize;
            lo[person] = lo[person].wrapping_add(adjust[code].0);
            hi[person] = hi[person].wrapping_add(adjust[code].1);
            missing[person] += u32::from(code == 1);
        }
    }
}

/// Counts missing calls (code 01) of one row for people `first_person..first_person + missing.len()`.
#[inline]
pub(crate) fn count_missing(row: &[u8], first_person: usize, missing: &mut [u32]) {
    let people = missing.len();
    let first_word = first_person / 32;
    let words = people.div_ceil(32);
    for w in 0..words {
        let x = row_word(row, first_word + w);
        let mut absent = x & !(x >> 1) & M55;
        if w + 1 == words && people % 32 != 0 {
            absent &= (1u64 << (2 * (people % 32))) - 1;
        }
        while absent != 0 {
            let bit = absent.trailing_zeros();
            absent &= absent - 1;
            missing[w * 32 + (bit / 2) as usize] += 1;
        }
    }
}

/// What a row costs on this machine and input, measured on the input's own first rows: a table
/// row costs about the same whatever its calls; a walked row costs a scan per 64-bit word plus
/// a cost per call that differs from the row's mode. Choosing a path never changes the cells.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowCosts {
    pub(crate) table_row_ns: f64,
    pub(crate) word_ns: f64,
    pub(crate) exception_ns: f64,
    /// A direct per-call lookup: the cheapest path when a range holds only a few people.
    pub(crate) direct_call_ns: f64,
}

impl RowCosts {
    #[inline(always)]
    pub(crate) fn table_wins(&self, exceptions: u64, words: usize) -> bool {
        self.word_ns * words as f64 + self.exception_ns * exceptions as f64 > self.table_row_ns
    }

    /// Whether every row of a `people`-person range is cheaper looked up call by call than
    /// through any row path: a walk pays at least one word scan, a table a whole row.
    #[inline(always)]
    pub(crate) fn direct_wins(&self, people: usize) -> bool {
        self.direct_call_ns * (people as f64) < self.word_ns.min(self.table_row_ns)
    }
}

/// Adds rows `ids` call by call: each person receives the term of their own code.
// Keep the compute loop separate from the dispatcher's allocation/error handling.
#[inline(never)]
fn apply_direct(
    data: &[u8],
    row_bytes: usize,
    ids: &[usize],
    terms: &[TermLimbs],
    first_person: usize,
    lo: &mut [i64],
    hi: &mut [i64],
    missing: &mut [u32],
) {
    for &r in ids {
        let row = &data[r * row_bytes..(r + 1) * row_bytes];
        for (p, ((l, h), m)) in lo
            .iter_mut()
            .zip(hi.iter_mut())
            .zip(missing.iter_mut())
            .enumerate()
        {
            let person = first_person + p;
            let code = ((row[person / 4] >> (2 * (person % 4))) & 3) as usize;
            *l = l.wrapping_add(terms[r][code].0);
            *h = h.wrapping_add(terms[r][code].1);
            *m += u32::from(code == 1);
        }
    }
}

/// Reusable per-thread scratch for `apply_rows`.
#[derive(Default)]
pub(crate) struct KernelScratch {
    tables: TableScratch,
    dense: Vec<usize>,
}

/// Adds rows `ids` to the cells of people `first_person..first_person + lo.len()`. Each row
/// either walks the calls that differ from its mode within the range or joins a four-variant
/// table group, whichever `costs` prices lower for that row's exceptions.
///
/// Limb sums stay exact: a person's final lo limb holds one lo part in `[0, 2^split)` per
/// variant (a walked row's adjustment plus its mode share is that variant's term), so the final
/// sums fit i64 and the wrapping intermediates are exact modulo 2^64.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_rows(
    data: &[u8],
    row_bytes: usize,
    ids: &[usize],
    terms: &[TermLimbs],
    costs: &RowCosts,
    geometry: TableGeometry,
    first_person: usize,
    scratch: &mut KernelScratch,
    lo: &mut [i64],
    hi: &mut [i64],
    missing: &mut [u32],
) -> Result<(), TryReserveError> {
    assert_eq!(first_person % 32, 0, "person ranges start on 64-bit words");
    assert!(lo.len() == hi.len() && lo.len() == missing.len());
    let people = lo.len();
    if people == 0 || ids.is_empty() {
        return Ok(());
    }
    if costs.direct_wins(people) {
        apply_direct(data, row_bytes, ids, terms, first_person, lo, hi, missing);
        return Ok(());
    }
    let geometry = geometry.for_work(ids.len(), people);
    scratch.dense.clear();
    scratch
        .dense
        .try_reserve_exact(ids.len().min(MAX_DENSE_ROWS))?;
    scratch
        .tables
        .prepare(geometry.groups_per_batch, geometry.tile_words)?;
    let words = people.div_ceil(32);
    for chunk in ids.chunks(MAX_DENSE_ROWS) {
        scratch.dense.clear();
        let (mut share_lo, mut share_hi) = (0i64, 0i64);
        for &r in chunk {
            let row = &data[r * row_bytes..(r + 1) * row_bytes];
            let (mode, exceptions) = row_mode(&row[first_person / 4..], people);
            if costs.table_wins(exceptions, words) {
                scratch.dense.push(r);
                continue;
            }
            let at_mode = terms[r][mode as usize];
            let adjust =
                terms[r].map(|(l, h)| (l.wrapping_sub(at_mode.0), h.wrapping_sub(at_mode.1)));
            walk_row(row, mode, &adjust, first_person, lo, hi, missing);
            share_lo = share_lo.wrapping_add(at_mode.0);
            share_hi = share_hi.wrapping_add(at_mode.1);
        }
        if share_lo != 0 || share_hi != 0 {
            for (l, h) in lo.iter_mut().zip(hi.iter_mut()) {
                *l = l.wrapping_add(share_lo);
                *h = h.wrapping_add(share_hi);
            }
        }
        if !scratch.dense.is_empty() {
            apply_table_rows(
                data,
                row_bytes,
                &scratch.dense,
                terms,
                first_person,
                geometry,
                &mut scratch.tables,
                lo,
                hi,
            )?;
            for &r in &scratch.dense {
                count_missing(
                    &data[r * row_bytes..(r + 1) * row_bytes],
                    first_person,
                    missing,
                );
            }
        }
    }
    Ok(())
}

/// Transposes four calls-bytes into the four people's keys (two butterfly exchanges).
#[inline(always)]
pub(crate) fn transpose_calls(bytes: [u8; 4]) -> [u8; 4] {
    let mut word = u32::from_le_bytes(bytes);
    let swap = (word ^ (word >> 6)) & 0x00cc_00cc;
    word ^= swap ^ (swap << 6);
    let swap = (word ^ (word >> 12)) & 0x0000_f0f0;
    word ^= swap ^ (swap << 12);
    word.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::exact::{FixedPoint, Split};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    /// Rows of `people` calls; `density` in 0..=64 sets how often a call leaves the row's mode.
    fn rows(rng: &mut Rng, count: usize, people: usize, density: u64) -> Vec<u8> {
        let row_bytes = people.div_ceil(4);
        let mut data = vec![0u8; count * row_bytes];
        for r in 0..count {
            let mode = [0u8, 2, 3][r % 3];
            for p in 0..people {
                let code = if rng.next() % 64 < density {
                    (rng.next() % 4) as u8
                } else {
                    mode
                };
                data[r * row_bytes + p / 4] |= code << (2 * (p % 4));
            }
        }
        data
    }

    fn naive(
        data: &[u8],
        row_bytes: usize,
        terms: &[[i128; 4]],
        people: usize,
    ) -> (Vec<i128>, Vec<u32>) {
        let (mut sums, mut missing) = (vec![0i128; people], vec![0u32; people]);
        for (r, t) in terms.iter().enumerate() {
            for p in 0..people {
                let code = ((data[r * row_bytes + p / 4] >> (2 * (p % 4))) & 3) as usize;
                sums[p] += t[code];
                missing[p] += u32::from(code == 1);
            }
        }
        (sums, missing)
    }

    #[test]
    fn every_path_and_range_matches_the_exact_per_call_sum() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let geometry = TableGeometry {
            groups_per_batch: 3,
            tile_words: 1,
        };
        let walk_only = RowCosts {
            table_row_ns: f64::INFINITY,
            word_ns: 0.0,
            exception_ns: 0.0,
            direct_call_ns: f64::INFINITY,
        };
        let tables_only = RowCosts {
            table_row_ns: 0.0,
            word_ns: 1.0,
            exception_ns: 1.0,
            direct_call_ns: f64::INFINITY,
        };
        let mixed = RowCosts {
            table_row_ns: 40.0,
            word_ns: 1.0,
            exception_ns: 1.0,
            direct_call_ns: f64::INFINITY,
        };
        let direct_only = RowCosts {
            table_row_ns: f64::INFINITY,
            word_ns: f64::INFINITY,
            exception_ns: 0.0,
            direct_call_ns: 0.0,
        };
        for people in [1, 3, 4, 31, 32, 33, 64, 97, 130] {
            for density in [0, 3, 32, 64] {
                let count = 23;
                let data = rows(&mut rng, count, people, density);
                let row_bytes = people.div_ceil(4);
                let weights: Vec<f64> = (0..count)
                    .map(|i| {
                        ((rng.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5)
                            * 10f64.powi(-(i as i32 % 7))
                    })
                    .collect();
                let corrections: Vec<f64> = weights
                    .iter()
                    .enumerate()
                    .map(|(i, w)| if i % 3 == 0 { 2.0 * w.abs() } else { 0.0 })
                    .collect();
                let fixed = FixedPoint::plan(
                    weights.iter().chain(&corrections).copied(),
                    count as u64,
                    2,
                    1,
                )
                .expect("fits");
                let exact: Vec<[i128; 4]> = weights
                    .iter()
                    .zip(&corrections)
                    .map(|(&w, &c)| {
                        [
                            0,
                            -fixed.to_fixed(c),
                            fixed.to_fixed(w),
                            2 * fixed.to_fixed(w),
                        ]
                    })
                    .collect();
                let term_bits = exact
                    .iter()
                    .flatten()
                    .map(|v| 128 - v.unsigned_abs().leading_zeros() + 1)
                    .max()
                    .unwrap();
                let split = Split::plan(term_bits, count as u64).expect("two limbs");
                let terms: Vec<TermLimbs> =
                    exact.iter().map(|t| t.map(|v| split.parts(v))).collect();
                let (want, want_missing) = naive(&data, row_bytes, &exact, people);
                let ids: Vec<usize> = (0..count).collect();
                for costs in [walk_only, tables_only, mixed, direct_only] {
                    for first in (0..people).step_by(32) {
                        for end in [(first + 32).min(people), people] {
                            let n = end - first;
                            let (mut lo, mut hi, mut missing) =
                                (vec![0i64; n], vec![0i64; n], vec![0u32; n]);
                            let mut scratch = KernelScratch::default();
                            apply_rows(
                                &data,
                                row_bytes,
                                &ids,
                                &terms,
                                &costs,
                                geometry,
                                first,
                                &mut scratch,
                                &mut lo,
                                &mut hi,
                                &mut missing,
                            )
                            .unwrap();
                            for p in 0..n {
                                assert_eq!(
                                    split.join(lo[p], hi[p]),
                                    want[first + p],
                                    "people {people} density {density} costs {costs:?} range {first}..{end} person {p}"
                                );
                                assert_eq!(missing[p], want_missing[first + p]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn table_geometry_follows_cache_sizes() {
        // EPYC 7763: 32 KiB L1d, 512 KiB L2. Tables for a batch fill half of L2.
        let geometry = TableGeometry::from_cache_sizes(32 << 10, 512 << 10);
        assert_eq!(geometry.groups_per_batch, 64);
        assert_eq!(geometry.tile_words, 28);
        // A tiny cache still yields a working geometry.
        let tiny = TableGeometry::from_cache_sizes(1 << 10, 8 << 10);
        assert!(tiny.groups_per_batch >= 4 && tiny.tile_words >= 1);
    }

    #[test]
    fn simd_keys_match_scalar_transposition() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let rows: Vec<Vec<u8>> = (0..4)
            .map(|_| (0..64).map(|_| rng.next() as u8).collect())
            .collect();
        let mut keys = vec![0u8; 64 * 4];
        transpose_keys(std::array::from_fn(|k| &rows[k][..]), &mut keys);
        for byte in 0..64 {
            assert_eq!(
                keys[byte * 4..byte * 4 + 4],
                transpose_calls(std::array::from_fn(|k| rows[k][byte]))
            );
        }
    }
}

/// Adds the rows `ids` (groups of four; a short final group reads zero-term rows) to the cells
/// of people `first_person..first_person + lo.len()`. Rows are packed PLINK rows of `row_bytes`
/// bytes. The range starts on a 32-person boundary; it may end anywhere, and people past its end
/// in the final byte are read but never written.
#[allow(clippy::too_many_arguments)]
// Keep table computation separate from the row classifier and scratch setup.
#[inline(never)]
pub(crate) fn apply_table_rows(
    data: &[u8],
    row_bytes: usize,
    ids: &[usize],
    terms: &[TermLimbs],
    first_person: usize,
    geometry: TableGeometry,
    scratch: &mut TableScratch,
    lo: &mut [i64],
    hi: &mut [i64],
) -> Result<(), TryReserveError> {
    const ZERO_TERMS: TermLimbs = [(0, 0); 4];
    assert_eq!(first_person % 32, 0, "person ranges start on 64-bit words");
    assert_eq!(lo.len(), hi.len());
    let people = lo.len();
    if people == 0 || ids.is_empty() {
        return Ok(());
    }
    let byte_start = first_person / 4;
    let byte_end = byte_start + people.div_ceil(4);
    assert!(
        byte_end <= row_bytes,
        "person range {first_person}+{people} exceeds {row_bytes}-byte rows"
    );
    let groups = ids.len().div_ceil(VARIANTS_PER_TABLE);
    let geometry = geometry.for_work(ids.len(), people);
    let tile_bytes = geometry.tile_words * 8;
    scratch.prepare(geometry.groups_per_batch, geometry.tile_words)?;
    for batch in (0..groups).step_by(geometry.groups_per_batch) {
        let count = (groups - batch).min(geometry.groups_per_batch);
        let member = |g: usize, k: usize| ids.get((batch + g) * VARIANTS_PER_TABLE + k);
        for g in 0..count {
            let t: [&TermLimbs; 4] =
                std::array::from_fn(|k| member(g, k).map_or(&ZERO_TERMS, |&r| &terms[r]));
            build_table(t, &mut scratch.lo_tables[g], &mut scratch.hi_tables[g]);
        }
        for start in (byte_start..byte_end).step_by(tile_bytes) {
            let end = (start + tile_bytes).min(byte_end);
            let whole = (end - start) / 8 * 8;
            let tile_people = (start - byte_start) * 4..((end - byte_start) * 4).min(people);
            let keys = &mut scratch.keys[..(end - start) * 4];
            for g in 0..count {
                let rows: [&[u8]; 4] = std::array::from_fn(|k| match member(g, k) {
                    Some(&r) => &data[r * row_bytes + start..r * row_bytes + end],
                    None => &scratch.zero_tile[..end - start],
                });
                transpose_keys(rows.map(|row| &row[..whole]), &mut keys[..whole * 4]);
                for byte in whole..end - start {
                    let calls = transpose_calls(rows.map(|row| row[byte]));
                    keys[byte * 4..byte * 4 + 4].copy_from_slice(&calls);
                }
                let used = tile_people.len();
                apply_table(
                    &scratch.lo_tables[g],
                    &scratch.hi_tables[g],
                    &keys[..used],
                    &mut lo[tile_people.clone()],
                    &mut hi[tile_people.clone()],
                );
            }
        }
    }
    Ok(())
}
