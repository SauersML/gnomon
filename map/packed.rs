//! Covariance products taken straight from 2-bit hard calls.
//!
//! # Why this exists
//!
//! A pass of the block solver is `C·Q = X·(Xᵀ·Q)/(n−1)` for an `n × b` block.
//! Decoding each variant into an `n`-row f64 column and multiplying a few
//! columns at a time against the whole block streams all `n·b` doubles of `Q`
//! in, and of `C·Q` out, once per chunk: at 500,000 samples a pass moved the
//! block through memory about 2,500 times, and the multiply kernels spent most
//! of their cycles waiting on it (#2361).
//!
//! Here the genotypes stay in their 2-bit codes and every product runs over row
//! leaves that fit in cache:
//!
//! * `Xᵀ·Q`: a leaf copies its rows of `Q` into row-major lanes once, and each
//!   variant accumulates `z(code)·q` over them with the sums held in registers.
//! * `X·P`: what a row takes from a variant is fixed by two code bits, so a row
//!   takes a whole group of eight variants from three table rows, sums over
//!   the subsets of the group that carry each bit, formed once per chunk of
//!   groups. A leaf adds rows and multiplies nothing, three rows where it would
//!   have added eight.
//!
//! The f64 working sets are a leaf of rows per thread, one tile of variant
//! images and one chunk of tables, whatever the cohort size.
//!
//! # The same bits at any thread count or tile width
//!
//! A variant's image sums its leaves in leaf order with compensated additions,
//! and the leaf height is a constant. Groups are fixed by variant index, and a
//! row of `X·P` takes its groups' rows one after another in variant order,
//! starting from what the row holds, so tile and chunk boundaries group nothing,
//! and which thread runs a leaf decides nothing.

use super::fit::HardCallPacked;
use faer::prelude::{Reborrow, ReborrowMut};
use faer::{MatMut, MatRef};
use rayon::prelude::*;
use std::ops::Range;
use std::simd::StdFloat;
use std::simd::prelude::*;
use std::sync::Mutex;

type Lanes = Simd<f64, 4>;

/// Rows per leaf. A multiple of four, so every leaf but the last starts on a
/// packed byte, and a constant: it groups every image sum, so it is part of
/// the answer. A leaf of 512 rows at 48 columns is 196 KiB of lanes, inside a
/// core's own cache; 4,096 rows projected a third slower on EPYC 7763.
pub(crate) const LEAF_ROWS: usize = 512;

/// Block columns one kernel instance carries in registers, in lanes of four.
/// Twelve lanes take the usual 48-column block in one group.
const GROUP_LANES: usize = 12;

/// Leaves computed together before their partial images merge.
const LEAF_BATCH: usize = 32;

/// Bytes one batch of partial images may occupy. It sets how many variants a
/// tile holds, which moves memory traffic and never the answer.
const PARTIAL_BYTES: usize = 16 << 20;

/// Variants whose code bits index one group's tables; one byte of mask each.
const GROUP_VARIANTS: usize = 8;

/// Rows a table holds: one for every subset of a group's variants.
const GROUP_SUBSETS: usize = 1 << GROUP_VARIANTS;

/// Tables a group keeps; see [`build_group_tables`].
const TABLE_KINDS: usize = 3;

/// Bytes one chunk of tables may occupy. A leaf reads them at random, so this
/// keeps a chunk inside a last-level cache shared by several threads; it moves
/// memory traffic and never the answer.
const TABLE_BYTES: usize = 8 << 20;

/// Where each leaf's logical rows sit among the packed physical lanes.
enum LeafRows<'a> {
    /// Every physical row, in order.
    Whole { bytes_needed: usize },
    /// A strictly increasing subset of rows: the retained-lane masks, and the
    /// byte and lane each leaf starts at.
    Selected {
        masks: &'a [u8],
        starts: Vec<(usize, u32)>,
        bytes_needed: usize,
    },
}

impl<'a> LeafRows<'a> {
    fn new(packed: &'a HardCallPacked<'_>, n_samples: usize) -> Option<Self> {
        match (packed.sample_selection(), packed.sample_byte_masks()) {
            (None, None) => Some(Self::Whole {
                bytes_needed: n_samples.div_ceil(4),
            }),
            (Some(selection), Some(masks)) => {
                if selection.len() != n_samples {
                    return None;
                }
                let bytes_needed = selection.last().map_or(0, |&last| last / 4 + 1);
                if masks.len() < bytes_needed {
                    return None;
                }
                let starts = selection
                    .iter()
                    .step_by(LEAF_ROWS)
                    .map(|&physical| (physical / 4, (physical % 4) as u32))
                    .collect();
                Some(Self::Selected {
                    masks,
                    starts,
                    bytes_needed,
                })
            }
            _ => None,
        }
    }

    fn bytes_needed(&self) -> usize {
        match self {
            Self::Whole { bytes_needed } | Self::Selected { bytes_needed, .. } => *bytes_needed,
        }
    }

    /// The code of every row of `leaf`, one per row, into `codes`.
    #[inline(always)]
    fn codes(&self, bytes: &[u8], leaf: usize, codes: &mut [u8]) {
        match self {
            Self::Whole { .. } => whole_codes(&bytes[leaf * LEAF_ROWS / 4..], codes),
            Self::Selected { masks, starts, .. } => {
                let (first_byte, first_lane) = starts[leaf];
                selected_codes(bytes, masks, first_byte, first_lane, codes);
            }
        }
    }

    /// For every row of `leaf`, one bit per variant of `group` (at most
    /// [`GROUP_VARIANTS`], in order): its code's high bit into `high`, its low
    /// bit into `low`. `codes` is scratch as long as `high`.
    #[inline(always)]
    fn masks(
        &self,
        group: &[&[u8]],
        leaf: usize,
        high: &mut [u8],
        low: &mut [u8],
        codes: &mut [u8],
    ) {
        match self {
            Self::Whole { .. } => {
                let first = leaf * LEAF_ROWS / 4;
                // A byte of each variant, variant `v` in byte `v` of the word:
                // four rows, whose bits sit two apart inside every byte.
                let word = |byte: usize| {
                    group
                        .iter()
                        .enumerate()
                        .fold(0u64, |word, (variant, bytes)| {
                            word | (u64::from(bytes[first + byte]) << (8 * variant))
                        })
                };
                let (high_quads, high_tail) = high.as_chunks_mut::<4>();
                let (low_quads, low_tail) = low.as_chunks_mut::<4>();
                let full = high_quads.len();
                for (byte, (high, low)) in
                    high_quads.iter_mut().zip(low_quads.iter_mut()).enumerate()
                {
                    let word = word(byte);
                    for lane in 0..4 {
                        let lanes = word >> (2 * lane);
                        high[lane] = gather_bytes(lanes >> 1);
                        low[lane] = gather_bytes(lanes);
                    }
                }
                if !high_tail.is_empty() {
                    let word = word(full);
                    for (lane, (high, low)) in
                        high_tail.iter_mut().zip(low_tail.iter_mut()).enumerate()
                    {
                        let lanes = word >> (2 * lane);
                        *high = gather_bytes(lanes >> 1);
                        *low = gather_bytes(lanes);
                    }
                }
            }
            Self::Selected { .. } => {
                high.fill(0);
                low.fill(0);
                for (variant, bytes) in group.iter().enumerate() {
                    self.codes(bytes, leaf, codes);
                    for ((high, low), &code) in
                        high.iter_mut().zip(low.iter_mut()).zip(codes.iter())
                    {
                        *high |= ((code >> 1) & 1) << variant;
                        *low |= (code & 1) << variant;
                    }
                }
            }
        }
    }
}

/// Bit 0 of every byte of `word`, byte `v` to bit `v`. The multiplier puts
/// byte `v`'s bit at bit `56 + v` and every other product term at a distinct
/// bit below, so nothing carries into the top byte.
#[inline(always)]
fn gather_bytes(word: u64) -> u8 {
    ((word & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8
}

#[inline(always)]
fn whole_codes(bytes: &[u8], codes: &mut [u8]) {
    let (quads, tail) = codes.as_chunks_mut::<4>();
    let full = quads.len();
    for (quad, &byte) in quads.iter_mut().zip(bytes) {
        *quad = [
            byte & 0b11,
            (byte >> 2) & 0b11,
            (byte >> 4) & 0b11,
            (byte >> 6) & 0b11,
        ];
    }
    if !tail.is_empty() {
        let byte = bytes[full];
        for (lane, code) in tail.iter_mut().enumerate() {
            *code = (byte >> (2 * lane)) & 0b11;
        }
    }
}

#[inline(always)]
fn selected_codes(
    bytes: &[u8],
    masks: &[u8],
    first_byte: usize,
    first_lane: u32,
    codes: &mut [u8],
) {
    let wanted = codes.len();
    let mut written = 0usize;
    let mut index = first_byte;
    let mut retained = masks[index] & (0b1111u8 << first_lane);
    while written < wanted {
        let byte = bytes[index];
        if retained == 0b1111 && written + 4 <= wanted {
            let quad: &mut [u8; 4] = (&mut codes[written..written + 4])
                .try_into()
                .expect("a retained byte fills four codes");
            *quad = [
                byte & 0b11,
                (byte >> 2) & 0b11,
                (byte >> 4) & 0b11,
                (byte >> 6) & 0b11,
            ];
            written += 4;
        } else {
            while retained != 0 && written < wanted {
                let lane = retained.trailing_zeros();
                retained &= retained - 1;
                codes[written] = (byte >> (2 * lane)) & 0b11;
                written += 1;
            }
        }
        index += 1;
        if written < wanted {
            retained = masks[index];
        }
    }
}

/// `Σ z(code_i)·row_i` over one leaf, the sums held in registers. `FUSED`
/// takes each term as one fused multiply-add, which the caller selects only
/// where the CPU has the instruction.
#[inline(always)]
fn project<const LANES: usize, const FUSED: bool>(
    codes: &[u8],
    values: &[f64; 4],
    rows: &[[Lanes; LANES]],
) -> [Lanes; LANES] {
    let mut sums = [Lanes::splat(0.0); LANES];
    for (&code, row) in codes.iter().zip(rows) {
        let z = Lanes::splat(values[usize::from(code & 0b11)]);
        for (sum, &q) in sums.iter_mut().zip(row) {
            *sum = if FUSED {
                z.mul_add(q, *sum)
            } else {
                z * q + *sum
            };
        }
    }
    sums
}

/// `row_i ← row_i + het[high_i] + hom[high_i & low_i] + base[low_i & !high_i]`
/// over one leaf, from one group's tables.
#[inline(always)]
fn scatter<const LANES: usize>(
    high: &[u8],
    low: &[u8],
    [het, hom, base]: [&[Lanes]; TABLE_KINDS],
    lanes_total: usize,
    first_lane: usize,
    rows: &mut [[Lanes; LANES]],
) {
    for ((row, &high), &low) in rows.iter_mut().zip(high).zip(low) {
        let (high, low) = (usize::from(high), usize::from(low));
        let het = table_entry::<LANES>(het, high * lanes_total + first_lane);
        let hom = table_entry::<LANES>(hom, (high & low) * lanes_total + first_lane);
        let base = table_entry::<LANES>(base, (low & !high) * lanes_total + first_lane);
        for (((value, &het), &hom), &base) in row.iter_mut().zip(het).zip(hom).zip(base) {
            *value += (het + hom) + base;
        }
    }
}

#[inline(always)]
fn table_entry<const LANES: usize>(table: &[Lanes], start: usize) -> &[Lanes; LANES] {
    table[start..start + LANES]
        .try_into()
        .expect("a table row holds every lane of the group")
}

/// One worker's leaf buffers.
struct Scratch {
    lanes: Vec<Lanes>,
    codes: Vec<u8>,
    /// A chunk's masks for one leaf, a leaf of rows a group; see
    /// [`LeafRows::masks`].
    high: Vec<u8>,
    low: Vec<u8>,
}

/// The variants one tile covers.
#[derive(Clone, Copy)]
struct Tile<'t> {
    slices: &'t [&'t [u8]],
    values: &'t [[f64; 4]],
}

/// A leaf's `Xᵀ·q` for the block columns `columns`, which start at column
/// `first_column`, into `partial` (variant-major, `width` columns a variant).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn project_leaf<const LANES: usize, const FUSED: bool>(
    rows: &LeafRows<'_>,
    leaf: usize,
    count: usize,
    columns: &[&[f64]],
    first_column: usize,
    width: usize,
    tile: Tile<'_>,
    partial: &mut [f64],
    scratch: &mut Scratch,
) {
    let Scratch { lanes, codes, .. } = scratch;
    let (buffer, _) = lanes[..count * LANES].as_chunks_mut::<LANES>();
    let first_row = leaf * LEAF_ROWS;
    for row in buffer.iter_mut() {
        *row = [Lanes::splat(0.0); LANES];
    }
    for (offset, values) in columns.iter().enumerate() {
        for (row, &value) in buffer.iter_mut().zip(&values[first_row..first_row + count]) {
            row[offset / 4].as_mut_array()[offset % 4] = value;
        }
    }
    let codes = &mut codes[..count];
    for (variant, (bytes, values)) in tile.slices.iter().zip(tile.values).enumerate() {
        let target = &mut partial[variant * width + first_column..][..columns.len()];
        if values.iter().all(|&value| value == 0.0) {
            target.fill(0.0);
            continue;
        }
        rows.codes(bytes, leaf, codes);
        let sums = project::<LANES, FUSED>(codes, values, buffer);
        for (offset, slot) in target.iter_mut().enumerate() {
            *slot = sums[offset / 4].as_array()[offset % 4];
        }
    }
}

/// A leaf's rows of `out`, for the block columns `columns`, plus the
/// contributions of a chunk's `groups`, taken in variant order from what the
/// rows held. `high` and `low` hold the leaf's masks, a leaf of rows a group.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scatter_leaf<const LANES: usize>(
    mut chunk: MatMut<'_, f64>,
    columns: Range<usize>,
    first_lane: usize,
    lanes_total: usize,
    groups: usize,
    high: &[u8],
    low: &[u8],
    tables: &[Lanes],
    lanes: &mut [Lanes],
) {
    let count = chunk.nrows();
    let (buffer, _) = lanes[..count * LANES].as_chunks_mut::<LANES>();
    for row in buffer.iter_mut() {
        *row = [Lanes::splat(0.0); LANES];
    }
    for (offset, column) in columns.clone().enumerate() {
        let values = chunk
            .rb()
            .col(column)
            .try_as_col_major()
            .expect("covariance image columns are contiguous")
            .as_slice();
        for (row, &value) in buffer.iter_mut().zip(values) {
            row[offset / 4].as_mut_array()[offset % 4] = value;
        }
    }
    let subsets = GROUP_SUBSETS * lanes_total;
    for group in 0..groups {
        let entries = &tables[group * TABLE_KINDS * subsets..][..TABLE_KINDS * subsets];
        let (het, rest) = entries.split_at(subsets);
        let (hom, base) = rest.split_at(subsets);
        scatter::<LANES>(
            &high[group * count..][..count],
            &low[group * count..][..count],
            [het, hom, base],
            lanes_total,
            first_lane,
            buffer,
        );
    }
    for (offset, column) in columns.enumerate() {
        let values = chunk
            .rb_mut()
            .col_mut(column)
            .try_as_col_major_mut()
            .expect("covariance image columns are contiguous")
            .as_slice_mut();
        for (slot, row) in values.iter_mut().zip(buffer.iter()) {
            *slot = row[offset / 4].as_array()[offset % 4];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The CPU must support AVX2 and FMA; see [`fused_multiply_add`].
unsafe fn project_leaf_avx2<const LANES: usize>(
    rows: &LeafRows<'_>,
    leaf: usize,
    count: usize,
    columns: &[&[f64]],
    first_column: usize,
    width: usize,
    tile: Tile<'_>,
    partial: &mut [f64],
    scratch: &mut Scratch,
) {
    project_leaf::<LANES, true>(
        rows,
        leaf,
        count,
        columns,
        first_column,
        width,
        tile,
        partial,
        scratch,
    );
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2,fma")]
/// # Safety
/// The CPU must support AVX2 and FMA; see [`fused_multiply_add`].
unsafe fn scatter_leaf_avx2<const LANES: usize>(
    chunk: MatMut<'_, f64>,
    columns: Range<usize>,
    first_lane: usize,
    lanes_total: usize,
    groups: usize,
    high: &[u8],
    low: &[u8],
    tables: &[Lanes],
    lanes: &mut [Lanes],
) {
    scatter_leaf::<LANES>(
        chunk,
        columns,
        first_lane,
        lanes_total,
        groups,
        high,
        low,
        tables,
        lanes,
    );
}

#[allow(clippy::too_many_arguments)]
fn project_leaf_lanes<const LANES: usize>(
    fused: bool,
    rows: &LeafRows<'_>,
    leaf: usize,
    count: usize,
    columns: &[&[f64]],
    first_column: usize,
    width: usize,
    tile: Tile<'_>,
    partial: &mut [f64],
    scratch: &mut Scratch,
) {
    #[cfg(target_arch = "x86_64")]
    if fused {
        // SAFETY: `fused` comes from `fused_multiply_add`, which saw AVX2 and
        // FMA on this CPU.
        unsafe {
            project_leaf_avx2::<LANES>(
                rows,
                leaf,
                count,
                columns,
                first_column,
                width,
                tile,
                partial,
                scratch,
            );
        }
        return;
    }
    if fused {
        project_leaf::<LANES, true>(
            rows,
            leaf,
            count,
            columns,
            first_column,
            width,
            tile,
            partial,
            scratch,
        );
    } else {
        project_leaf::<LANES, false>(
            rows,
            leaf,
            count,
            columns,
            first_column,
            width,
            tile,
            partial,
            scratch,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn scatter_leaf_lanes<const LANES: usize>(
    fused: bool,
    chunk: MatMut<'_, f64>,
    columns: Range<usize>,
    first_lane: usize,
    lanes_total: usize,
    groups: usize,
    high: &[u8],
    low: &[u8],
    tables: &[Lanes],
    lanes: &mut [Lanes],
) {
    #[cfg(target_arch = "x86_64")]
    if fused {
        // SAFETY: `fused` comes from `fused_multiply_add`, which saw AVX2 and
        // FMA on this CPU.
        unsafe {
            scatter_leaf_avx2::<LANES>(
                chunk,
                columns,
                first_lane,
                lanes_total,
                groups,
                high,
                low,
                tables,
                lanes,
            );
        }
        return;
    }
    scatter_leaf::<LANES>(
        chunk,
        columns,
        first_lane,
        lanes_total,
        groups,
        high,
        low,
        tables,
        lanes,
    );
}

macro_rules! by_lanes {
    ($lanes:expr, $kernel:ident($($arg:expr),* $(,)?)) => {
        match $lanes {
            1 => $kernel::<1>($($arg),*),
            2 => $kernel::<2>($($arg),*),
            3 => $kernel::<3>($($arg),*),
            4 => $kernel::<4>($($arg),*),
            5 => $kernel::<5>($($arg),*),
            6 => $kernel::<6>($($arg),*),
            7 => $kernel::<7>($($arg),*),
            8 => $kernel::<8>($($arg),*),
            9 => $kernel::<9>($($arg),*),
            10 => $kernel::<10>($($arg),*),
            11 => $kernel::<11>($($arg),*),
            12 => $kernel::<12>($($arg),*),
            other => unreachable!("a kernel group carries at most {GROUP_LANES} lanes, not {other}"),
        }
    };
}

/// Whether the product may take its terms as fused multiply-adds here.
fn fused_multiply_add() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

/// The tables of a chunk of variants, whose images are `images` (`width`
/// columns a variant) and whose code values are `values`, into `tables`:
/// for each group of [`GROUP_VARIANTS`], every subset's sum of its variants'
/// het steps `scale·(z(10) − z(00))·image`, then of their hom steps
/// `scale·(z(11) − z(10))·image`, then of their missing steps
/// `scale·(z(01) − z(00))·image` on top of the group's base row
/// `Σ scale·z(00)·image`, each row padded to whole lanes with zeros.
///
/// A code's high bit marks a het or hom call and its low bit a hom or missing
/// one, so a row whose group has high bits `h` and low bits `l` takes
/// `het[h] + hom[h & l] + base[l & !h]`, which is `Σ scale·z(code)·image` over
/// the group. A short last group's absent variants have no bits.
fn build_group_tables(
    images: &[f64],
    values: &[[f64; 4]],
    width: usize,
    lanes_total: usize,
    scale: f64,
    tables: &mut [Lanes],
) {
    let subsets = GROUP_SUBSETS * lanes_total;
    tables
        .par_chunks_mut(TABLE_KINDS * subsets)
        .zip(values.par_chunks(GROUP_VARIANTS))
        .zip(images.par_chunks(GROUP_VARIANTS * width))
        .for_each(|((entries, values), images)| {
            let lane_row = |image: &[f64], coefficient: f64, lane: usize| {
                Lanes::from_array(std::array::from_fn(|offset| {
                    image
                        .get(4 * lane + offset)
                        .map_or(0.0, |&value| coefficient * value)
                }))
            };
            let mut steps = vec![Lanes::splat(0.0); TABLE_KINDS * GROUP_VARIANTS * lanes_total];
            let (het_steps, rest) = steps.split_at_mut(GROUP_VARIANTS * lanes_total);
            let (hom_steps, missing_steps) = rest.split_at_mut(GROUP_VARIANTS * lanes_total);
            let (het, rest) = entries.split_at_mut(subsets);
            let (hom, base) = rest.split_at_mut(subsets);
            het[..lanes_total].fill(Lanes::splat(0.0));
            hom[..lanes_total].fill(Lanes::splat(0.0));
            base[..lanes_total].fill(Lanes::splat(0.0));
            for (variant, values) in values.iter().enumerate() {
                let image = &images[variant * width..(variant + 1) * width];
                let steps = [
                    (&mut *het_steps, values[2] - values[0]),
                    (&mut *hom_steps, values[3] - values[2]),
                    (&mut *missing_steps, values[1] - values[0]),
                ];
                for (steps, difference) in steps {
                    for (lane, step) in steps[variant * lanes_total..(variant + 1) * lanes_total]
                        .iter_mut()
                        .enumerate()
                    {
                        *step = lane_row(image, scale * difference, lane);
                    }
                }
                for (lane, slot) in base[..lanes_total].iter_mut().enumerate() {
                    *slot += lane_row(image, scale * values[0], lane);
                }
            }
            for (table, steps) in [
                (het, &*het_steps),
                (hom, &*hom_steps),
                (base, &*missing_steps),
            ] {
                for mask in 1..GROUP_SUBSETS {
                    let variant = mask.trailing_zeros() as usize;
                    let (earlier, row) = table.split_at_mut(mask * lanes_total);
                    let prior = &earlier[(mask & (mask - 1)) * lanes_total..][..lanes_total];
                    let step = &steps[variant * lanes_total..(variant + 1) * lanes_total];
                    for ((slot, &prior), &step) in
                        row[..lanes_total].iter_mut().zip(prior).zip(step)
                    {
                        *slot = prior + step;
                    }
                }
            }
        });
}

/// `out ← out + scale·X·Xᵀ·rhs`, and `factor ← Xᵀ·rhs` when given, for the
/// standardized genotypes behind `packed`: logical variant `j` takes the value
/// `code_values[j][c]` wherever its 2-bit code is `c`.
///
/// Returns `false` having written nothing when the view cannot serve every row
/// and variant, or a matrix is not stored contiguously; the caller then takes
/// the product another way.
pub(crate) fn covariance_product(
    packed: &HardCallPacked<'_>,
    code_values: &[[f64; 4]],
    scale: f64,
    rhs: MatRef<'_, f64>,
    mut out: MatMut<'_, f64>,
    mut factor: Option<MatMut<'_, f64>>,
    progress: &dyn Fn(usize),
) -> bool {
    let n_samples = rhs.nrows();
    let width = rhs.ncols();
    let variants = code_values.len();
    if out.nrows() != n_samples
        || out.ncols() != width
        || factor
            .as_ref()
            .is_some_and(|factor| factor.nrows() != variants || factor.ncols() != width)
    {
        return false;
    }
    let Some(rows) = LeafRows::new(packed, n_samples) else {
        return false;
    };
    let Some(slices) = (0..variants)
        .map(|variant| packed.slice(variant, 1))
        .collect::<Option<Vec<&[u8]>>>()
    else {
        return false;
    };
    if slices.iter().any(|bytes| bytes.len() < rows.bytes_needed()) {
        return false;
    }
    let Some(columns) = (0..width)
        .map(|column| {
            rhs.col(column)
                .try_as_col_major()
                .map(|column| column.as_slice())
        })
        .collect::<Option<Vec<&[f64]>>>()
    else {
        return false;
    };
    if (0..width).any(|column| out.rb().col(column).try_as_col_major().is_none()) {
        return false;
    }
    if n_samples == 0 || width == 0 || variants == 0 {
        if let Some(factor) = factor.as_mut() {
            factor.fill(0.0);
        }
        return true;
    }

    let fused = fused_multiply_add();
    let lanes_total = width.div_ceil(4);
    let groups: Vec<(usize, usize)> = (0..lanes_total)
        .step_by(GROUP_LANES)
        .map(|first_lane| (first_lane, GROUP_LANES.min(lanes_total - first_lane)))
        .collect();
    let leaves = n_samples.div_ceil(LEAF_ROWS);
    let leaf_height = LEAF_ROWS.min(n_samples);
    // Whole groups to a tile and to a chunk, so neither boundary splits one.
    let per_tile = (PARTIAL_BYTES / (LEAF_BATCH * width * std::mem::size_of::<f64>()))
        .max(1)
        .next_multiple_of(GROUP_VARIANTS)
        .min(variants);
    let group_len = TABLE_KINDS * GROUP_SUBSETS * lanes_total;
    let chunk_groups = (TABLE_BYTES / (group_len * std::mem::size_of::<Lanes>()))
        .clamp(1, per_tile.div_ceil(GROUP_VARIANTS));
    let chunk_variants = chunk_groups * GROUP_VARIANTS;
    let workers = rayon::current_num_threads().max(1);
    let scratch: Vec<Mutex<Scratch>> = (0..workers)
        .map(|_| {
            Mutex::new(Scratch {
                lanes: vec![Lanes::splat(0.0); leaf_height * GROUP_LANES],
                codes: vec![0u8; leaf_height],
                high: vec![0u8; leaf_height * chunk_groups],
                low: vec![0u8; leaf_height * chunk_groups],
            })
        })
        .collect();
    let worker_scratch = || {
        let index = rayon::current_thread_index().unwrap_or(0).min(workers - 1);
        scratch[index]
            .lock()
            .expect("packed covariance scratch poisoned")
    };
    let mut partials = vec![0.0f64; LEAF_BATCH.min(leaves) * per_tile * width];
    let mut images = vec![0.0f64; per_tile * width];
    let mut compensations = vec![0.0f64; per_tile * width];
    let mut tables = vec![Lanes::splat(0.0); chunk_groups * group_len];

    let mut first = 0usize;
    while first < variants {
        let count = per_tile.min(variants - first);
        let tile = Tile {
            slices: &slices[first..first + count],
            values: &code_values[first..first + count],
        };
        let span = count * width;
        let images = &mut images[..span];
        let compensations = &mut compensations[..span];
        images.fill(0.0);
        compensations.fill(0.0);

        // Xᵀ·rhs: a batch of leaves at a time, each leaf's partial merged into
        // the running image in leaf order.
        for batch_first in (0..leaves).step_by(LEAF_BATCH) {
            let batch = LEAF_BATCH.min(leaves - batch_first);
            partials[..batch * span]
                .par_chunks_mut(span)
                .enumerate()
                .for_each(|(offset, partial)| {
                    let leaf = batch_first + offset;
                    let count = LEAF_ROWS.min(n_samples - leaf * LEAF_ROWS);
                    let mut guard = worker_scratch();
                    for &(first_lane, lanes) in &groups {
                        let first_column = 4 * first_lane;
                        let end_column = width.min(4 * (first_lane + lanes));
                        by_lanes!(
                            lanes,
                            project_leaf_lanes(
                                fused,
                                &rows,
                                leaf,
                                count,
                                &columns[first_column..end_column],
                                first_column,
                                width,
                                tile,
                                partial,
                                &mut guard,
                            )
                        );
                    }
                });
            let partials = &partials[..batch * span];
            images
                .par_chunks_mut(width)
                .zip(compensations.par_chunks_mut(width))
                .enumerate()
                .for_each(|(variant, (sums, errors))| {
                    for leaf in 0..batch {
                        let partial = &partials[leaf * span + variant * width..][..width];
                        for ((sum, error), &addend) in
                            sums.iter_mut().zip(errors.iter_mut()).zip(partial)
                        {
                            // TwoSum (Knuth): the rounding error of each merge is
                            // exact, and accumulates apart from the sum.
                            let x = *sum;
                            let s = x + addend;
                            let z = s - x;
                            *error += (x - (s - z)) + (addend - z);
                            *sum = s;
                        }
                    }
                });
        }
        for (image, error) in images.iter_mut().zip(compensations.iter()) {
            *image += *error;
        }
        if let Some(factor) = factor.as_mut() {
            for variant in 0..count {
                for column in 0..width {
                    factor[(first + variant, column)] = images[variant * width + column];
                }
            }
        }

        // X·images: a chunk of groups at a time, its tables formed once and
        // read by every leaf of rows.
        let images = &*images;
        for chunk_first in (0..count).step_by(chunk_variants) {
            let chunk_count = chunk_variants.min(count - chunk_first);
            let chunk_slices = &tile.slices[chunk_first..chunk_first + chunk_count];
            let chunk_groups = chunk_count.div_ceil(GROUP_VARIANTS);
            let tables = &mut tables[..chunk_groups * group_len];
            build_group_tables(
                &images[chunk_first * width..(chunk_first + chunk_count) * width],
                &tile.values[chunk_first..chunk_first + chunk_count],
                width,
                lanes_total,
                scale,
                tables,
            );
            let tables = &*tables;
            out.rb_mut()
                .par_row_chunks_mut(LEAF_ROWS)
                .enumerate()
                .for_each(|(leaf, mut chunk)| {
                    let count = chunk.nrows();
                    let mut guard = worker_scratch();
                    let Scratch {
                        lanes,
                        codes,
                        high,
                        low,
                    } = &mut *guard;
                    for (group, members) in chunk_slices.chunks(GROUP_VARIANTS).enumerate() {
                        rows.masks(
                            members,
                            leaf,
                            &mut high[group * count..(group + 1) * count],
                            &mut low[group * count..(group + 1) * count],
                            &mut codes[..count],
                        );
                    }
                    for &(first_lane, lanes_here) in &groups {
                        let columns = 4 * first_lane..width.min(4 * (first_lane + lanes_here));
                        by_lanes!(
                            lanes_here,
                            scatter_leaf_lanes(
                                fused,
                                chunk.rb_mut(),
                                columns,
                                first_lane,
                                lanes_total,
                                chunk_groups,
                                &high[..chunk_groups * count],
                                &low[..chunk_groups * count],
                                tables,
                                lanes,
                            )
                        );
                    }
                });
        }

        first += count;
        progress(first);
    }
    true
}
