// Exact score kernels over packed PLINK rows.
//
// A cell is one person's i64 lanes of the exact plan (see `score::cells`), `stride` lanes wide.
// Every term enters a cell as an integer and lanes add with wrapping arithmetic, so tables,
// walks, batch sizes, thread counts and the order of rows cannot change a single bit.

use crate::score::cells::LANE_WIDTH;
use std::simd::{Simd, cmp::SimdPartialEq, num::SimdUint};

/// Four variants per table: a person's four two-bit calls form the table key.
const VARIANTS_PER_TABLE: usize = 4;
/// Four-variant groups built and applied together.
const GROUPS_PER_BATCH: usize = 16;
const M55: u64 = 0x5555_5555_5555_5555;

/// Which people a batch scores, and where each one's calls sit in a packed row.
#[derive(Clone, Copy)]
pub(crate) enum People<'a> {
    /// People `0..count`, in row order.
    All(usize),
    /// Kept people: byte offset and shift of each one's call, in output order.
    Gathered { bytes: &'a [u32], shifts: &'a [u8] },
}

impl People<'_> {
    #[inline(always)]
    fn len(&self) -> usize {
        match self {
            Self::All(count) => *count,
            Self::Gathered { bytes, .. } => bytes.len(),
        }
    }
}

#[inline(always)]
fn add_assign(dst: &mut [i64], src: &[i64]) {
    for (d, s) in dst
        .chunks_exact_mut(LANE_WIDTH)
        .zip(src.chunks_exact(LANE_WIDTH))
    {
        (Simd::<i64, LANE_WIDTH>::from_slice(d) + Simd::<i64, LANE_WIDTH>::from_slice(s))
            .copy_to_slice(d);
    }
}

#[inline(always)]
fn add_into(dst: &mut [i64], a: &[i64], b: &[i64]) {
    for ((d, x), y) in dst
        .chunks_exact_mut(LANE_WIDTH)
        .zip(a.chunks_exact(LANE_WIDTH))
        .zip(b.chunks_exact(LANE_WIDTH))
    {
        (Simd::<i64, LANE_WIDTH>::from_slice(x) + Simd::<i64, LANE_WIDTH>::from_slice(y))
            .copy_to_slice(d);
    }
}

/// Entry `key` of a group's table = Σ over its four rows of `terms[row][code of row in key]`, all
/// lanes at once. Prefix expansion shares partial sums: 4 + 12 + 48 + 192 lane rows.
fn build_table(terms: &[i64], group: usize, stride: usize, table: &mut [i64]) {
    let row = |v: usize, code: usize| {
        let at = ((group * VARIANTS_PER_TABLE + v) * 4 + code) * stride;
        &terms[at..at + stride]
    };
    for code in 0..4 {
        table[code * stride..(code + 1) * stride].copy_from_slice(row(0, code));
    }
    for v in 1..VARIANTS_PER_TABLE {
        let prefix = 1usize << (2 * v);
        for code in (1..4).rev() {
            let (head, tail) = table.split_at_mut(code * prefix * stride);
            let add = row(v, code);
            for entry in 0..prefix {
                add_into(
                    &mut tail[entry * stride..(entry + 1) * stride],
                    &head[entry * stride..(entry + 1) * stride],
                    add,
                );
            }
        }
        let add = row(v, 0);
        for entry in 0..prefix {
            add_assign(&mut table[entry * stride..(entry + 1) * stride], add);
        }
    }
}

/// Table keys for 32 people per step: the butterfly transpose of four row words on eight u32
/// lanes. `rows[k]` are whole 8-byte chunks of the same byte range; `keys.len() == 4 * rows[k].len()`.
#[inline]
fn transpose_keys(rows: [&[u8]; VARIANTS_PER_TABLE], keys: &mut [u8]) {
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

/// Transposes four calls-bytes into the four people's keys (two butterfly exchanges).
#[inline(always)]
fn transpose_calls(bytes: [u8; 4]) -> [u8; 4] {
    let mut word = u32::from_le_bytes(bytes);
    let swap = (word ^ (word >> 6)) & 0x00cc_00cc;
    word ^= swap ^ (swap << 6);
    let swap = (word ^ (word >> 12)) & 0x0000_f0f0;
    word ^= swap ^ (swap << 12);
    word.to_le_bytes()
}

/// Whether a group costs less through its table than through its rows' terms added person by
/// person. Either way a carrier (a person with some call not 00) adds one term row; the table pays
/// its 256 entries up front, and direct adds pay one more row for each further call of a carrier.
/// A group whose code-00 terms are not zero adds to people without calls, so only a table holds it.
fn prefers_table(terms: &[i64], group: usize, stride: usize, rows: [&[u8]; VARIANTS_PER_TABLE]) -> bool {
    let code_zero_adds = (0..VARIANTS_PER_TABLE).any(|v| {
        let at = (group * VARIANTS_PER_TABLE + v) * 4 * stride;
        terms[at..at + stride].iter().any(|&lane| lane != 0)
    });
    if code_zero_adds {
        return true;
    }
    let (mut calls, mut carriers) = (0u64, 0u64);
    let mut count = |masks: [u64; VARIANTS_PER_TABLE]| {
        calls += masks.iter().map(|mask| u64::from(mask.count_ones())).sum::<u64>();
        carriers += u64::from((masks[0] | masks[1] | masks[2] | masks[3]).count_ones());
    };
    let whole = rows[0].len() / 8 * 8;
    for start in (0..whole).step_by(8) {
        count(rows.map(|row| {
            let x = u64::from_le_bytes(std::array::from_fn(|i| row[start + i]));
            (x | (x >> 1)) & M55
        }));
    }
    for byte in whole..rows[0].len() {
        count(rows.map(|row| u64::from((row[byte] | (row[byte] >> 1)) & 0x55)));
    }
    calls - carriers >= 256
}

/// Reusable per-thread scratch for [`apply_table_rows`].
#[derive(Default)]
pub(crate) struct TableScratch {
    tables: Vec<i64>,
    column_keys: Vec<u8>,
    person_keys: Vec<u8>,
    zero_row: Vec<u8>,
}

/// Adds `rows` packed rows to every person's cell: `terms` holds `rows × 4 × stride` lanes (the
/// terms of codes 00, 01, 10 and 11 of each row) and `cells` holds `people × stride` lanes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply_table_rows(
    data: &[u8],
    row_bytes: usize,
    rows: usize,
    terms: &[i64],
    stride: usize,
    people: People,
    scratch: &mut TableScratch,
    cells: &mut [i64],
) {
    let count = people.len();
    let groups = rows.div_ceil(VARIANTS_PER_TABLE);
    assert!(stride % LANE_WIDTH == 0 && cells.len() == count * stride);
    assert!(terms.len() >= groups * VARIANTS_PER_TABLE * 4 * stride && data.len() >= rows * row_bytes);
    if count == 0 || rows == 0 {
        return;
    }
    scratch.zero_row.resize(row_bytes, 0);
    scratch.tables.resize(GROUPS_PER_BATCH * 256 * stride, 0);
    scratch.person_keys.resize(count * GROUPS_PER_BATCH, 0);
    let key_width = row_bytes * 4;
    if let People::All(_) = people {
        scratch.column_keys.resize(GROUPS_PER_BATCH * key_width, 0);
    }
    let whole = row_bytes / 8 * 8;
    // A stride of one or two SIMD widths keeps each person's accumulator in registers, and reads
    // every group through its table.
    let striped = matches!(stride, 4 | 8);
    for batch in (0..groups).step_by(GROUPS_PER_BATCH) {
        let in_batch = (groups - batch).min(GROUPS_PER_BATCH);
        let mut tabled = 0u32;
        for g in 0..in_batch {
            let group = batch + g;
            let source: [&[u8]; VARIANTS_PER_TABLE] = std::array::from_fn(|v| {
                let r = group * VARIANTS_PER_TABLE + v;
                if r < rows {
                    &data[r * row_bytes..(r + 1) * row_bytes]
                } else {
                    &scratch.zero_row[..]
                }
            });
            if striped || prefers_table(terms, group, stride, source) {
                build_table(
                    terms,
                    group,
                    stride,
                    &mut scratch.tables[g * 256 * stride..(g + 1) * 256 * stride],
                );
                tabled |= 1 << g;
            }
            match people {
                People::All(count) => {
                    let keys = &mut scratch.column_keys[g * key_width..(g + 1) * key_width];
                    transpose_keys(source.map(|r| &r[..whole]), &mut keys[..whole * 4]);
                    for byte in whole..row_bytes {
                        keys[byte * 4..byte * 4 + 4]
                            .copy_from_slice(&transpose_calls(source.map(|r| r[byte])));
                    }
                    for (p, &key) in keys[..count].iter().enumerate() {
                        scratch.person_keys[p * GROUPS_PER_BATCH + g] = key;
                    }
                }
                People::Gathered { bytes, shifts } => {
                    for (p, (&byte, &shift)) in bytes.iter().zip(shifts).enumerate() {
                        let byte = byte as usize;
                        let code = |v: usize| ((source[v][byte] >> shift) & 3) << (2 * v);
                        scratch.person_keys[p * GROUPS_PER_BATCH + g] =
                            code(0) | code(1) | code(2) | code(3);
                    }
                }
            }
        }
        let tables = &scratch.tables[..in_batch * 256 * stride];
        let keys = &scratch.person_keys;
        match stride {
            4 => apply_stripe_4(tables, keys, in_batch, cells),
            8 => apply_stripe_8(tables, keys, in_batch, cells),
            _ => {
                // A person adds only the groups whose entry adds something: a key other than 0,
                // or a tabled group whose key-0 entry is not zero. On rare rows most keys are 0.
                // An untabled group's code-00 terms are zero, so its key-0 entry adds nothing.
                let mut zero_first = 0u32;
                for g in 0..in_batch {
                    if tabled & (1 << g) == 0
                        || tables[g * 256 * stride..(g * 256 + 1) * stride].iter().all(|&lane| lane == 0)
                    {
                        zero_first |= 1 << g;
                    }
                }
                let in_batch_groups = (1u32 << in_batch) - 1;
                for (p, cell) in cells.chunks_exact_mut(stride).enumerate() {
                    let row = &keys[p * GROUPS_PER_BATCH..(p + 1) * GROUPS_PER_BATCH];
                    let nonzero = Simd::<u8, GROUPS_PER_BATCH>::from_slice(row)
                        .simd_ne(Simd::splat(0))
                        .to_bitmask() as u32;
                    let mut active = (nonzero | !zero_first) & in_batch_groups;
                    while active != 0 {
                        let g = active.trailing_zeros() as usize;
                        active &= active - 1;
                        let key = usize::from(row[g]);
                        if tabled & (1 << g) != 0 {
                            let at = (g * 256 + key) * stride;
                            add_assign(cell, &tables[at..at + stride]);
                        } else {
                            // The table's entry, summed from the rows of the calls that are not 00.
                            let first_row = (batch + g) * VARIANTS_PER_TABLE;
                            for v in 0..VARIANTS_PER_TABLE {
                                let code = (key >> (2 * v)) & 3;
                                if code != 0 {
                                    let at = ((first_row + v) * 4 + code) * stride;
                                    add_assign(cell, &terms[at..at + stride]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

macro_rules! apply_stripe {
    ($name:ident, $lanes:literal) => {
        /// Each person's lanes accumulated in registers over the batch's groups.
        #[inline(always)]
        fn $name(tables: &[i64], keys: &[u8], in_batch: usize, cells: &mut [i64]) {
            for (p, cell) in cells.chunks_exact_mut($lanes).enumerate() {
                let mut acc = Simd::<i64, $lanes>::from_slice(cell);
                for (g, &key) in keys[p * GROUPS_PER_BATCH..p * GROUPS_PER_BATCH + in_batch]
                    .iter()
                    .enumerate()
                {
                    let at = (g * 256 + key as usize) * $lanes;
                    acc += Simd::<i64, $lanes>::from_slice(&tables[at..at + $lanes]);
                }
                acc.copy_to_slice(cell);
            }
        }
    };
}
apply_stripe!(apply_stripe_4, 4);
apply_stripe!(apply_stripe_8, 8);

/// Calls `visit(person, code)` for every person whose call in `row` is not 00, in person order.
#[inline]
pub(crate) fn for_each_call(row: &[u8], people: People, mut visit: impl FnMut(usize, u8)) {
    match people {
        People::All(count) => {
            let words = count.div_ceil(32);
            for w in 0..words {
                let start = w * 8;
                let mut bytes = [0u8; 8];
                let chunk = &row[start..(start + 8).min(row.len())];
                bytes[..chunk.len()].copy_from_slice(chunk);
                let x = u64::from_le_bytes(bytes);
                let mut calls = (x | (x >> 1)) & M55;
                if w + 1 == words && count % 32 != 0 {
                    calls &= (1u64 << (2 * (count % 32))) - 1;
                }
                while calls != 0 {
                    let bit = calls.trailing_zeros();
                    calls &= calls - 1;
                    visit(w * 32 + (bit / 2) as usize, ((x >> bit) & 3) as u8);
                }
            }
        }
        People::Gathered { bytes, shifts } => {
            for (p, (&byte, &shift)) in bytes.iter().zip(shifts).enumerate() {
                let code = (row[byte as usize] >> shift) & 3;
                if code != 0 {
                    visit(p, code);
                }
            }
        }
    }
}

/// Calls `visit(person)` for every person whose call in `row` is missing (code 01), in order.
#[inline]
pub(crate) fn for_each_missing(row: &[u8], people: People, mut visit: impl FnMut(usize)) {
    match people {
        People::All(count) => {
            let words = count.div_ceil(32);
            for w in 0..words {
                let start = w * 8;
                let mut bytes = [0u8; 8];
                let chunk = &row[start..(start + 8).min(row.len())];
                bytes[..chunk.len()].copy_from_slice(chunk);
                let x = u64::from_le_bytes(bytes);
                let mut absent = x & !(x >> 1) & M55;
                if w + 1 == words && count % 32 != 0 {
                    absent &= (1u64 << (2 * (count % 32))) - 1;
                }
                while absent != 0 {
                    visit(w * 32 + (absent.trailing_zeros() / 2) as usize);
                    absent &= absent - 1;
                }
            }
        }
        People::Gathered { bytes, shifts } => {
            for (p, (&byte, &shift)) in bytes.iter().zip(shifts).enumerate() {
                if (row[byte as usize] >> shift) & 3 == 1 {
                    visit(p);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    fn code(row: &[u8], person: usize) -> usize {
        ((row[person / 4] >> (2 * (person % 4))) & 3) as usize
    }

    #[test]
    fn transposed_keys_are_the_four_calls_of_each_person() {
        let mut rng = Rng(7);
        for people in [1usize, 31, 32, 33, 127, 200] {
            let row_bytes = people.div_ceil(4);
            let rows: Vec<Vec<u8>> = (0..4)
                .map(|_| (0..row_bytes).map(|_| rng.next() as u8).collect())
                .collect();
            let whole = row_bytes / 8 * 8;
            let mut keys = vec![0u8; row_bytes * 4];
            let source: [&[u8]; 4] = std::array::from_fn(|v| &rows[v][..]);
            transpose_keys(source.map(|r| &r[..whole]), &mut keys[..whole * 4]);
            for byte in whole..row_bytes {
                keys[byte * 4..byte * 4 + 4].copy_from_slice(&transpose_calls(source.map(|r| r[byte])));
            }
            for person in 0..people {
                let want = (0..4).map(|v| code(&rows[v], person) << (2 * v)).sum::<usize>();
                assert_eq!(keys[person] as usize, want, "people {people} person {person}");
            }
        }
    }

    #[test]
    fn tables_match_per_call_sums_for_every_stride_and_person_layout() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        // (people, rows, stride, calls kept in one of how many in every other group of four rows):
        // dense groups of 600 or more people take tables, groups of 130 people and sparse groups
        // add rows directly, and alternating groups mix both within a batch.
        for (people, rows, stride, sparse) in [
            (1usize, 1usize, 4usize, 1u64),
            (37, 5, 4, 1),
            (64, 67, 8, 1),
            (130, 13, 12, 1),
            (33, 256, 8, 1),
            (600, 9, 16, 1),
            (600, 70, 12, 30),
            (501, 131, 20, 3),
            (1000, 67, 16, 40),
        ] {
            let row_bytes = (people + 5).div_ceil(4);
            let data: Vec<u8> = (0..rows * row_bytes)
                .map(|at| {
                    let every = if (at / row_bytes / 4) % 2 == 1 { sparse } else { 1 };
                    (0..4).fold(0u8, |byte, slot| {
                        let call = if rng.next() % every == 0 { rng.next() as u8 & 3 } else { 0 };
                        byte | call << (2 * slot)
                    })
                })
                .collect();
            let groups = rows.div_ceil(4);
            let mut terms = vec![0i64; groups * 4 * 4 * stride];
            for r in 0..rows {
                for code in 1..4 {
                    for lane in 0..stride {
                        terms[(r * 4 + code) * stride + lane] = rng.next() as i64;
                    }
                }
            }
            // Gathered people: every other slot of the row, reversed.
            let kept: Vec<usize> = (0..people).map(|p| (people - 1 - p) * 2 % (row_bytes * 4)).collect();
            let bytes: Vec<u32> = kept.iter().map(|&f| (f / 4) as u32).collect();
            let shifts: Vec<u8> = kept.iter().map(|&f| (2 * (f % 4)) as u8).collect();
            for gathered in [false, true] {
                let layout = if gathered {
                    People::Gathered { bytes: &bytes, shifts: &shifts }
                } else {
                    People::All(people)
                };
                let mut cells = vec![0i64; people * stride];
                let mut scratch = TableScratch::default();
                apply_table_rows(&data, row_bytes, rows, &terms, stride, layout, &mut scratch, &mut cells);
                let mut want = vec![0i64; people * stride];
                for p in 0..people {
                    let slot = if gathered { kept[p] } else { p };
                    for r in 0..rows {
                        let c = code(&data[r * row_bytes..(r + 1) * row_bytes], slot);
                        for lane in 0..stride {
                            want[p * stride + lane] =
                                want[p * stride + lane].wrapping_add(terms[(r * 4 + c) * stride + lane]);
                        }
                    }
                }
                assert_eq!(cells, want, "people {people} rows {rows} stride {stride} gathered {gathered}");
            }
        }
    }

    #[test]
    fn sparse_rows_skip_only_groups_that_add_nothing() {
        let mut rng = Rng(0x2354_0000_5a17_0001);
        // (people, rows, stride, every how many rows code 00 adds terms; 0 for never)
        for (people, rows, stride, flipped_every) in
            [(45usize, 70usize, 16usize, 7usize), (33, 131, 32, 0), (100, 64, 12, 3), (7, 9, 16, 1), (70, 5, 24, 0)]
        {
            let row_bytes = people.div_ceil(4);
            // About one call in fifty is not 00, so most keys are 0.
            let data: Vec<u8> = (0..rows * row_bytes)
                .map(|_| {
                    (0..4).fold(0u8, |byte, slot| {
                        let call = if rng.next() % 50 == 0 { (rng.next() % 3 + 1) as u8 } else { 0 };
                        byte | call << (2 * slot)
                    })
                })
                .collect();
            let groups = rows.div_ceil(4);
            let mut terms = vec![0i64; groups * 4 * 4 * stride];
            for r in 0..rows {
                let first = if flipped_every != 0 && r % flipped_every == 0 { 0 } else { 1 };
                for code in first..4 {
                    for lane in 0..stride {
                        terms[(r * 4 + code) * stride + lane] = rng.next() as i64;
                    }
                }
            }
            let kept: Vec<usize> = (0..people).map(|p| (people - 1 - p) * 3 % (row_bytes * 4)).collect();
            let bytes: Vec<u32> = kept.iter().map(|&f| (f / 4) as u32).collect();
            let shifts: Vec<u8> = kept.iter().map(|&f| (2 * (f % 4)) as u8).collect();
            for gathered in [false, true] {
                let layout = if gathered {
                    People::Gathered { bytes: &bytes, shifts: &shifts }
                } else {
                    People::All(people)
                };
                let mut cells = vec![0i64; people * stride];
                let mut scratch = TableScratch::default();
                apply_table_rows(&data, row_bytes, rows, &terms, stride, layout, &mut scratch, &mut cells);
                let mut want = vec![0i64; people * stride];
                for p in 0..people {
                    let slot = if gathered { kept[p] } else { p };
                    for r in 0..rows {
                        let c = code(&data[r * row_bytes..(r + 1) * row_bytes], slot);
                        for lane in 0..stride {
                            want[p * stride + lane] =
                                want[p * stride + lane].wrapping_add(terms[(r * 4 + c) * stride + lane]);
                        }
                    }
                }
                assert_eq!(cells, want, "people {people} rows {rows} stride {stride} gathered {gathered}");
            }
        }
    }

    #[test]
    fn call_walks_visit_every_nonzero_call_once() {
        let mut rng = Rng(3);
        for people in [1usize, 32, 45, 96] {
            let row: Vec<u8> = (0..people.div_ceil(4) + 3).map(|_| rng.next() as u8).collect();
            let mut seen = Vec::new();
            for_each_call(&row, People::All(people), |p, c| seen.push((p, c as usize)));
            let want: Vec<(usize, usize)> = (0..people)
                .map(|p| (p, code(&row, p)))
                .filter(|&(_, c)| c != 0)
                .collect();
            assert_eq!(seen, want);
            let mut missing = Vec::new();
            for_each_missing(&row, People::All(people), |p| missing.push(p));
            let want: Vec<usize> = (0..people).filter(|&p| code(&row, p) == 1).collect();
            assert_eq!(missing, want);
        }
    }
}
