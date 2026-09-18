// Exact score kernels over packed PLINK rows.
//
// A cell is one person's i64 lanes of the exact plan (see `score::cells`), `stride` lanes wide.
// Every term enters a cell as an integer and lanes add with wrapping arithmetic, so tables,
// walks, batch sizes, thread counts and the order of rows cannot change a single bit.
//
// Each loop over people or table entries is a function of its own (`#[inline(never)]`), called once
// a group or a batch, so its code depends on nothing around it. Inlined, the same loops took from 2%
// to twice the instructions as code elsewhere in the crate changed an inliner decision: a caller's
// size, a third call site (#2362), an unrelated scratch buffer.

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

// Lanes step as fixed-size arrays by index. A zip of `chunks_exact` iterators left its setup out
// of line in the wide table kernel, and the loop then rechecked a runtime chunk length every
// step: twice the instructions of the common N 50,000 K 128 and 512 cells.
#[inline(always)]
fn add_assign(dst: &mut [i64], src: &[i64]) {
    let (dst, _) = dst.as_chunks_mut::<LANE_WIDTH>();
    let (src, _) = src.as_chunks::<LANE_WIDTH>();
    for c in 0..dst.len().min(src.len()) {
        dst[c] = (Simd::from_array(dst[c]) + Simd::from_array(src[c])).to_array();
    }
}

#[inline(always)]
fn add_into(dst: &mut [i64], a: &[i64], b: &[i64]) {
    let (dst, _) = dst.as_chunks_mut::<LANE_WIDTH>();
    let (a, _) = a.as_chunks::<LANE_WIDTH>();
    let (b, _) = b.as_chunks::<LANE_WIDTH>();
    for c in 0..dst.len().min(a.len()).min(b.len()) {
        dst[c] = (Simd::from_array(a[c]) + Simd::from_array(b[c])).to_array();
    }
}

/// Entry `key` of a group's table = Σ over its four rows of `terms[row][code of row in key]`, all
/// lanes at once. Prefix expansion shares partial sums: 4 + 12 + 48 + 192 lane rows.
#[inline(never)]
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
#[inline(never)]
fn transpose_keys(rows: [&[u8]; VARIANTS_PER_TABLE], keys: &mut [u8]) {
    let lane = |chunk: &[u8; 8]| Simd::<u8, 8>::from_array(*chunk).cast::<u32>();
    let (out, _) = keys.as_chunks_mut::<32>();
    // Four explicit lanes rather than array::map: an out-of-line map left a call in the loop in some
    // builds, whatever the surrounding code.
    let rows = [
        rows[0].as_chunks::<8>().0,
        rows[1].as_chunks::<8>().0,
        rows[2].as_chunks::<8>().0,
        rows[3].as_chunks::<8>().0,
    ];
    let steps = rows.iter().fold(out.len(), |steps, row| steps.min(row.len()));
    // Every view cut to `steps`, so the loop's indices need no bounds checks.
    let out = &mut out[..steps];
    let rows = [&rows[0][..steps], &rows[1][..steps], &rows[2][..steps], &rows[3][..steps]];
    for step in 0..steps {
        let [c0, c1, c2, c3] = [&rows[0][step], &rows[1][step], &rows[2][step], &rows[3][step]];
        let mut word = lane(c0) | (lane(c1) << 8) | (lane(c2) << 16) | (lane(c3) << 24);
        let swap = (word ^ (word >> 6)) & Simd::splat(0x00cc_00cc);
        word ^= swap ^ (swap << 6);
        let swap = (word ^ (word >> 12)) & Simd::splat(0x0000_f0f0);
        word ^= swap ^ (swap << 12);
        for (quad, value) in word.to_array().into_iter().enumerate() {
            out[step][4 * quad..4 * quad + 4].copy_from_slice(&value.to_le_bytes());
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
#[inline(never)]
fn prefers_table(terms: &[i64], group: usize, stride: usize, rows: [&[u8]; VARIANTS_PER_TABLE]) -> bool {
    let code_zero_adds = (0..VARIANTS_PER_TABLE).any(|v| {
        let at = (group * VARIANTS_PER_TABLE + v) * 4 * stride;
        terms[at..at + stride].iter().any(|&lane| lane != 0)
    });
    if code_zero_adds {
        return true;
    }
    // Calls past each carrier's first, summed over the rows' words: a word's calls are never fewer
    // than its carriers, so the sum only grows, and a group has the table's answer as soon as it
    // reaches 256. A dense group reaches it within a few words.
    let surplus = |masks: [u64; VARIANTS_PER_TABLE]| {
        let calls: u32 = masks.iter().map(|mask| mask.count_ones()).sum();
        u64::from(calls - (masks[0] | masks[1] | masks[2] | masks[3]).count_ones())
    };
    let whole = rows[0].len() / 8 * 8;
    // Each row's whole words, cut to the same count so the loop's indices need no bounds checks.
    let n = whole / 8;
    let words = [
        &rows[0].as_chunks::<8>().0[..n],
        &rows[1].as_chunks::<8>().0[..n],
        &rows[2].as_chunks::<8>().0[..n],
        &rows[3].as_chunks::<8>().0[..n],
    ];
    let calls_in = |word: &[u8; 8]| {
        let x = u64::from_le_bytes(*word);
        (x | (x >> 1)) & M55
    };
    // Checked once every sixteen words, so a rare group, which scans every word, pays one compare a
    // block rather than one a word.
    let mut beyond = 0u64;
    for block in (0..n).step_by(16) {
        for w in block..(block + 16).min(n) {
            let masks =
                [calls_in(&words[0][w]), calls_in(&words[1][w]), calls_in(&words[2][w]), calls_in(&words[3][w])];
            beyond += surplus(masks);
        }
        if beyond >= 256 {
            return true;
        }
    }
    let calls_at = |row: &[u8], byte: usize| u64::from((row[byte] | (row[byte] >> 1)) & 0x55);
    for byte in whole..rows[0].len() {
        let masks =
            [calls_at(rows[0], byte), calls_at(rows[1], byte), calls_at(rows[2], byte), calls_at(rows[3], byte)];
        beyond += surplus(masks);
    }
    beyond >= 256
}

/// Reusable per-thread scratch for [`apply_table_rows`].
#[derive(Default)]
pub(crate) struct TableScratch {
    tables: Vec<i64>,
    column_keys: Vec<u8>,
    person_keys: Vec<u8>,
    zero_row: Vec<u8>,
    tabled: Vec<usize>,
    untabled: Vec<usize>,
}

/// Untabled groups one pass over the cells adds at most: a person's keys are one 64-bit mask.
const UNTABLED_PER_PASS: usize = 64;
/// The most bytes of person keys an untabled pass takes before it adds fewer groups per pass.
const UNTABLED_KEY_BYTES: usize = 1 << 24;

/// The four packed rows of `group`, the rows past the end reading as zero calls.
#[inline(always)]
fn group_rows<'a>(
    data: &'a [u8],
    row_bytes: usize,
    rows: usize,
    zero_row: &'a [u8],
    group: usize,
) -> [&'a [u8]; VARIANTS_PER_TABLE] {
    let row = |v: usize| {
        let r = group * VARIANTS_PER_TABLE + v;
        if r < rows { &data[r * row_bytes..(r + 1) * row_bytes] } else { &zero_row[..row_bytes] }
    };
    [row(0), row(1), row(2), row(3)]
}

/// `buffer` at least `len` long, grown with zeros and never shrunk. The growth is a cold call of
/// its own: whether `Vec::resize` was inlined into the kernel followed how many callers the kernel
/// had, and a third call site (#2362) moved N 20,000 task-clock by 2.5-3.7%.
#[inline(always)]
pub(crate) fn grow<T: Copy + Default>(buffer: &mut Vec<T>, len: usize) {
    if buffer.len() < len {
        grow_cold(buffer, len);
    }
}

#[cold]
#[inline(never)]
fn grow_cold<T: Copy + Default>(buffer: &mut Vec<T>, len: usize) {
    buffer.resize(len, T::default());
}

/// Writes the keys of the group whose rows are `source` into `keys` in person order, person `p`'s
/// at `keys[p]`. For a complete cohort `keys` takes a row's four keys a byte, past the last person.
#[inline(always)]
fn group_keys(source: [&[u8]; VARIANTS_PER_TABLE], people: People, keys: &mut [u8]) {
    match people {
        People::All(_) => {
            let row_bytes = source[0].len();
            let whole = row_bytes / 8 * 8;
            let keys = &mut keys[..row_bytes * 4];
            let rows = [&source[0][..whole], &source[1][..whole], &source[2][..whole], &source[3][..whole]];
            transpose_keys(rows, &mut keys[..whole * 4]);
            for byte in whole..row_bytes {
                let calls = [source[0][byte], source[1][byte], source[2][byte], source[3][byte]];
                keys[byte * 4..byte * 4 + 4].copy_from_slice(&transpose_calls(calls));
            }
        }
        People::Gathered { bytes, shifts } => {
            for (p, (&byte, &shift)) in bytes.iter().zip(shifts).enumerate() {
                let byte = byte as usize;
                let code = |v: usize| ((source[v][byte] >> shift) & 3) << (2 * v);
                keys[p] = code(0) | code(1) | code(2) | code(3);
            }
        }
    }
}

/// Writes every person's key of the group whose rows are `source` at `slot` of their `W`-byte key
/// row, through `column_keys`, the group's keys in person order. The rows are `W`-byte arrays, so
/// no store is bounds-checked: as a function of its own with a runtime width, every one of them was
/// (rare N 50,000 K 16: +8% instructions).
#[inline(never)]
fn write_keys<const W: usize>(
    source: [&[u8]; VARIANTS_PER_TABLE],
    people: People,
    column_keys: &mut [u8],
    person_keys: &mut [u8],
    slot: usize,
) {
    let count = people.len();
    group_keys(source, people, column_keys);
    assert!(slot < W);
    let rows = &mut person_keys.as_chunks_mut::<W>().0[..count];
    for (row, &key) in rows.iter_mut().zip(&column_keys[..count]) {
        row[slot] = key;
    }
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
    // Scratch only grows: every slot a kernel reads is written first, and a buffer shrunk by one
    // call and grown by the next would be zeroed again every call.
    grow(&mut scratch.zero_row, row_bytes);
    // A group's keys in person order: the transpose writes a whole row's four keys a byte.
    let key_width = match people {
        People::All(_) => row_bytes * 4,
        People::Gathered { .. } => count,
    };
    grow(&mut scratch.column_keys, key_width);
    // A stride of one or two SIMD widths keeps each person's accumulator in registers, and reads
    // every group through its table, from each group's keys in person order. Past that, a group
    // whose table costs more than its rows is added from its rows, and every such group of the call
    // in as few passes over the cells as the key budget allows: on rare rows a pass over every
    // person's cell is most of the cost.
    let striped = matches!(stride, 4 | 8);
    scratch.tabled.clear();
    scratch.untabled.clear();
    for group in 0..groups {
        let source = group_rows(data, row_bytes, rows, &scratch.zero_row, group);
        if striped || prefers_table(terms, group, stride, source) {
            scratch.tabled.push(group);
        } else {
            scratch.untabled.push(group);
        }
    }

    if !scratch.tabled.is_empty() {
        grow(&mut scratch.tables, GROUPS_PER_BATCH * 256 * stride);
        if striped {
            grow(&mut scratch.column_keys, GROUPS_PER_BATCH * key_width);
        } else {
            grow(&mut scratch.person_keys, count * GROUPS_PER_BATCH);
        }
    }
    for batch in scratch.tabled.chunks(GROUPS_PER_BATCH) {
        for (g, &group) in batch.iter().enumerate() {
            build_table(
                terms,
                group,
                stride,
                &mut scratch.tables[g * 256 * stride..(g + 1) * 256 * stride],
            );
            let source = group_rows(data, row_bytes, rows, &scratch.zero_row, group);
            if striped {
                // A striped person reads each group's key where the transpose left it: gathering
                // them into a row a person first took a scattered byte store per person and group.
                group_keys(source, people, &mut scratch.column_keys[g * key_width..(g + 1) * key_width]);
            } else {
                write_keys::<GROUPS_PER_BATCH>(source, people, &mut scratch.column_keys, &mut scratch.person_keys, g);
            }
        }
        let tables = &scratch.tables[..batch.len() * 256 * stride];
        match stride {
            4 => apply_stripe_4(tables, &scratch.column_keys, key_width, batch.len(), cells),
            8 => apply_stripe_8(tables, &scratch.column_keys, key_width, batch.len(), cells),
            _ => apply_tables(tables, &scratch.person_keys, batch.len(), stride, cells),
        }
    }

    if scratch.untabled.is_empty() {
        return;
    }
    let width = [UNTABLED_PER_PASS, 32, GROUPS_PER_BATCH]
        .into_iter()
        .find(|&width| count * width <= UNTABLED_KEY_BYTES)
        .unwrap_or(GROUPS_PER_BATCH);
    grow(&mut scratch.person_keys, count * width);
    let (untabled, zero_row) = (&scratch.untabled, &scratch.zero_row);
    let (column_keys, person_keys) = (&mut scratch.column_keys, &mut scratch.person_keys);
    for pass in untabled.chunks(width) {
        for (slot, &group) in pass.iter().enumerate() {
            let source = group_rows(data, row_bytes, rows, zero_row, group);
            match width {
                UNTABLED_PER_PASS => write_keys::<UNTABLED_PER_PASS>(source, people, column_keys, person_keys, slot),
                32 => write_keys::<32>(source, people, column_keys, person_keys, slot),
                _ => write_keys::<GROUPS_PER_BATCH>(source, people, column_keys, person_keys, slot),
            }
        }
        match width {
            UNTABLED_PER_PASS => apply_rows::<UNTABLED_PER_PASS>(terms, person_keys, pass, stride, cells),
            32 => apply_rows::<32>(terms, person_keys, pass, stride, cells),
            _ => apply_rows::<GROUPS_PER_BATCH>(terms, person_keys, pass, stride, cells),
        }
    }
}

/// Adds a batch's tables to every person: only the groups whose entry adds something, a key other
/// than 0 or a group whose key-0 entry is not zero. On rare rows most keys are 0.
#[inline(never)]
fn apply_tables(tables: &[i64], keys: &[u8], in_batch: usize, stride: usize, cells: &mut [i64]) {
    let mut zero_first = 0u32;
    for g in 0..in_batch {
        if tables[g * 256 * stride..(g * 256 + 1) * stride].iter().all(|&lane| lane == 0) {
            zero_first |= 1 << g;
        }
    }
    let in_batch_groups = (1u32 << in_batch) - 1;
    for (cell, row) in cells.chunks_exact_mut(stride).zip(keys.as_chunks::<GROUPS_PER_BATCH>().0) {
        let nonzero = Simd::<u8, GROUPS_PER_BATCH>::from_array(*row)
            .simd_ne(Simd::splat(0))
            .to_bitmask() as u32;
        let mut active = (nonzero | !zero_first) & in_batch_groups;
        while active != 0 {
            let g = active.trailing_zeros() as usize;
            active &= active - 1;
            let at = (g * 256 + usize::from(row[g])) * stride;
            add_assign(cell, &tables[at..at + stride]);
        }
    }
}

/// Adds untabled groups to every person, all of `groups` in one pass over the cells: a key `k` adds
/// the rows of its calls that are not 00, the entry `k` of the group's table summed directly. An
/// untabled group's code-00 terms are zero, so a key of 0 adds nothing. `keys` holds `W` bytes a
/// person, a multiple of sixteen, as `W`-byte arrays.
#[inline(never)]
fn apply_rows<const W: usize>(terms: &[i64], keys: &[u8], groups: &[usize], stride: usize, cells: &mut [i64]) {
    let in_pass = u64::MAX >> (UNTABLED_PER_PASS - groups.len());
    for (cell, row) in cells.chunks_exact_mut(stride).zip(keys.as_chunks::<W>().0) {
        let mut active = 0u64;
        for (i, chunk) in row.as_chunks::<GROUPS_PER_BATCH>().0.iter().enumerate() {
            let nonzero = Simd::<u8, GROUPS_PER_BATCH>::from_array(*chunk)
                .simd_ne(Simd::splat(0))
                .to_bitmask();
            active |= nonzero << (GROUPS_PER_BATCH * i);
        }
        active &= in_pass;
        while active != 0 {
            let slot = active.trailing_zeros() as usize;
            active &= active - 1;
            let key = usize::from(row[slot]);
            let first_row = groups[slot] * VARIANTS_PER_TABLE;
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

macro_rules! apply_stripe {
    ($name:ident, $lanes:literal) => {
        /// Each person's lanes accumulated in registers over the batch's groups. `keys` holds
        /// each group's keys in person order, `key_width` apart.
        #[inline(never)]
        fn $name(tables: &[i64], keys: &[u8], key_width: usize, in_batch: usize, cells: &mut [i64]) {
            let people = cells.len() / $lanes;
            assert!(people <= key_width && keys.len() >= in_batch * key_width);
            for (p, cell) in cells.chunks_exact_mut($lanes).enumerate() {
                let mut acc = Simd::<i64, $lanes>::from_slice(cell);
                for g in 0..in_batch {
                    let at = (g * 256 + usize::from(keys[g * key_width + p])) * $lanes;
                    acc += Simd::<i64, $lanes>::from_slice(&tables[at..at + $lanes]);
                }
                acc.copy_to_slice(cell);
            }
        }
    };
}
apply_stripe!(apply_stripe_4, 4);
apply_stripe!(apply_stripe_8, 8);

/// Word `w` of a packed row split into its whole words and `tail`: 32 calls, the first in the low
/// bits, with bytes past the row reading as 00. A whole word loads directly; copying a word's
/// bytes by the row's remaining length called memcpy once a word, half the memmove of a common
/// N 50,000 K 1 cell.
#[inline(always)]
fn row_word(whole: &[[u8; 8]], tail: &[u8], w: usize) -> u64 {
    match whole.get(w) {
        Some(&word) => u64::from_le_bytes(word),
        None => {
            let mut bytes = [0u8; 8];
            bytes[..tail.len()].copy_from_slice(tail);
            u64::from_le_bytes(bytes)
        }
    }
}

/// Calls `visit(person, code)` for every person whose call in `row` is not 00, in person order.
#[inline]
pub(crate) fn for_each_call(row: &[u8], people: People, mut visit: impl FnMut(usize, u8)) {
    match people {
        People::All(count) => {
            let words = count.div_ceil(32);
            let (whole, tail) = row.as_chunks::<8>();
            for w in 0..words {
                let x = row_word(whole, tail, w);
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
            let (whole, tail) = row.as_chunks::<8>();
            for w in 0..words {
                let x = row_word(whole, tail, w);
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
        let mut shared = TableScratch::default();
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
            // 75 untabled groups: two passes over the cells.
            (40, 300, 12, 1),
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
                // Scratch only grows, so one kept across every case holds stale keys and tables
                // from larger calls; the cells must not see them.
                let mut reused = vec![0i64; people * stride];
                apply_table_rows(&data, row_bytes, rows, &terms, stride, layout, &mut shared, &mut reused);
                assert_eq!(reused, cells, "people {people} rows {rows} stride {stride} gathered {gathered}, reused scratch");
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
        let mut shared = TableScratch::default();
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
                // Scratch only grows, so one kept across every case holds stale keys and tables
                // from larger calls; the cells must not see them.
                let mut reused = vec![0i64; people * stride];
                apply_table_rows(&data, row_bytes, rows, &terms, stride, layout, &mut shared, &mut reused);
                assert_eq!(reused, cells, "people {people} rows {rows} stride {stride} gathered {gathered}, reused scratch");
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
        // Rows of exactly their people's bytes, and rows with bytes past them.
        for (people, pad) in [1usize, 32, 33, 45, 96, 100].into_iter().flat_map(|people| [(people, 0), (people, 3)]) {
            let row: Vec<u8> = (0..people.div_ceil(4) + pad).map(|_| rng.next() as u8).collect();
            let mut seen = Vec::new();
            for_each_call(&row, People::All(people), |p, c| seen.push((p, c as usize)));
            let want: Vec<(usize, usize)> = (0..people)
                .map(|p| (p, code(&row, p)))
                .filter(|&(_, c)| c != 0)
                .collect();
            assert_eq!(seen, want, "people {people} pad {pad}");
            let mut missing = Vec::new();
            for_each_missing(&row, People::All(people), |p| missing.push(p));
            let want: Vec<usize> = (0..people).filter(|&p| code(&row, p) == 1).collect();
            assert_eq!(missing, want, "people {people} pad {pad}");
        }
    }
}
