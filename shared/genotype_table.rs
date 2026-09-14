//! Four-locus lookup arithmetic over PLINK's four two-bit call states.
//! Tables are bounded scratch space, not a cohort-sized genotype expansion.

use std::simd::Simd;

pub(crate) const VARIANTS_PER_TABLE: usize = 4;
pub(crate) const TABLE_ROWS: usize = 256;
pub(crate) const TABLE_BUDGET_BYTES: usize = 256 * 1024;
pub(crate) const SAMPLE_TILE: usize = 1024;
/// Rows up to this many outputs are accumulated in at most eight four-wide
/// register lanes. Wider rows leave at most three groups in a table tile, too
/// few to repay loading the row into lanes.
pub(crate) const REGISTER_COLUMNS: usize = 32;

/// Table row width for `columns` outputs. Rows the lane accumulator takes are
/// padded to whole lanes; padding columns stay zero and are never read back.
pub(crate) fn table_columns(columns: usize) -> usize {
    if columns <= REGISTER_COLUMNS {
        columns.div_ceil(4) * 4
    } else {
        columns
    }
}

#[inline(always)]
pub(crate) fn add_row(dst: &mut [f64], src: &[f64]) {
    assert_eq!(dst.len(), src.len());
    let (chunks, tail) = dst.as_chunks_mut::<4>();
    let (values, remainder) = src.as_chunks::<4>();
    for (dst, src) in chunks.iter_mut().zip(values) {
        *dst = (Simd::<f64, 4>::from_array(*dst) + Simd::from_array(*src)).to_array();
    }
    for (dst, src) in tail.iter_mut().zip(remainder) {
        *dst += src;
    }
}

/// `calls` is variant × PLINK code × output. Prefix expansion shares the
/// partial sums instead of recomputing four additions for every table entry.
pub(crate) fn build_table(calls: &[f64], columns: usize, table: &mut [f64]) {
    assert!(columns > 0);
    assert_eq!(calls.len(), VARIANTS_PER_TABLE * 4 * columns);
    assert_eq!(table.len(), TABLE_ROWS * columns);
    if columns <= REGISTER_COLUMNS && columns % 4 == 0 {
        macro_rules! lanes {
            ($($lanes:literal)*) => {
                match columns / 4 {
                    $($lanes => return build_lanes::<$lanes>(calls, table),)*
                    _ => unreachable!("row widths are checked above"),
                }
            };
        }
        lanes!(1 2 3 4 5 6 7 8);
    }
    build_table_rows(calls, columns, table);
}

/// `build_table` for rows of whole lanes, with every row sum a constant-size
/// lane loop rather than a copy followed by a runtime-length addition. Each
/// entry is still `prefix + contribution`, so the table is bit-identical.
#[inline(always)]
fn build_lanes<const LANES: usize>(calls: &[f64], table: &mut [f64]) {
    let (calls, _) = calls.as_chunks::<4>();
    let (table, _) = table.as_chunks_mut::<4>();
    for lane in 0..4 * LANES {
        table[lane] = calls[lane];
    }
    for variant in 1..VARIANTS_PER_TABLE {
        let prefix_rows = 1 << (2 * variant);
        for code in (1..4).rev() {
            let contribution = (variant * 4 + code) * LANES;
            for row in 0..prefix_rows {
                let (dst, src) = ((code * prefix_rows + row) * LANES, row * LANES);
                for lane in 0..LANES {
                    table[dst + lane] = (Simd::from_array(table[src + lane])
                        + Simd::from_array(calls[contribution + lane]))
                    .to_array();
                }
            }
        }
        let contribution = variant * 4 * LANES;
        for row in 0..prefix_rows {
            for lane in 0..LANES {
                let dst = row * LANES + lane;
                table[dst] = (Simd::from_array(table[dst])
                    + Simd::from_array(calls[contribution + lane]))
                .to_array();
            }
        }
    }
}

fn build_table_rows(calls: &[f64], columns: usize, table: &mut [f64]) {
    table[..4 * columns].copy_from_slice(&calls[..4 * columns]);
    for variant in 1..VARIANTS_PER_TABLE {
        let prefix_rows = 1 << (2 * variant);
        let prefix_len = prefix_rows * columns;
        for code in (1..4).rev() {
            let (prefix, output) = table.split_at_mut(code * prefix_len);
            let contribution =
                &calls[(variant * 4 + code) * columns..(variant * 4 + code + 1) * columns];
            for (dst, src) in output[..prefix_len]
                .chunks_exact_mut(columns)
                .zip(prefix[..prefix_len].chunks_exact(columns))
            {
                dst.copy_from_slice(src);
                add_row(dst, contribution);
            }
        }
        let contribution = &calls[variant * 4 * columns..(variant * 4 + 1) * columns];
        for dst in table[..prefix_len].chunks_exact_mut(columns) {
            add_row(dst, contribution);
        }
    }
}

/// Transpose a 4×4 matrix of two-bit calls using two butterfly exchanges.
#[inline(always)]
pub(crate) fn transpose_calls(bytes: [u8; 4]) -> [u8; 4] {
    let mut word = u32::from_le_bytes(bytes);
    let swap = (word ^ (word >> 6)) & 0x00cc_00cc;
    word ^= swap ^ (swap << 6);
    let swap = (word ^ (word >> 12)) & 0x0000_f0f0;
    word ^= swap ^ (swap << 12);
    word.to_le_bytes()
}

pub(crate) fn consecutive_keys(bytes: &[&[u8]], sample_start: usize, keys: &mut [u8]) {
    assert!(bytes.len() <= VARIANTS_PER_TABLE);
    assert_eq!(sample_start % 4, 0);
    if let &[a, b, c, d] = bytes {
        // Four byte runs of the key count's length let the transposition vectorize.
        let (whole, tail) = keys.as_chunks_mut::<4>();
        let (start, end) = (sample_start / 4, sample_start / 4 + whole.len());
        let (a, b, c, d) = (&a[start..end], &b[start..end], &c[start..end], &d[start..end]);
        for (index, dst) in whole.iter_mut().enumerate() {
            *dst = transpose_calls([a[index], b[index], c[index], d[index]]);
        }
        let offset = sample_start + 4 * whole.len();
        for (sample, key) in tail.iter_mut().enumerate() {
            *key = selected_key(bytes, offset + sample);
        }
        return;
    }
    for (byte, dst) in keys.chunks_mut(4).enumerate() {
        if dst.len() < 4 {
            for (sample, key) in dst.iter_mut().enumerate() {
                *key = selected_key(bytes, sample_start + byte * 4 + sample);
            }
            break;
        }
        let mut calls = [0; 4];
        for (call, source) in calls.iter_mut().zip(bytes) {
            *call = source[sample_start / 4 + byte];
        }
        dst.copy_from_slice(&transpose_calls(calls)[..dst.len()]);
    }
}

#[inline(always)]
pub(crate) fn selected_key(bytes: &[&[u8]], sample: usize) -> u8 {
    let shift = 2 * (sample % 4);
    let mut key = 0;
    for (variant, bytes) in bytes.iter().enumerate() {
        key |= ((bytes[sample / 4] >> shift) & 3) << (2 * variant);
    }
    key
}

#[inline(always)]
pub(crate) fn missing_bits(key: u8) -> u8 {
    key & !(key >> 1) & 0x55
}

/// `scores[sample] += tables[group][keys[group][sample]]` for every group in
/// order, with `keys` group-major and `tables` padded by `table_columns`. Each
/// row is loaded into register lanes and stored once instead of once per group;
/// the additions keep their order, so every sum is bit-identical to adding the
/// rows in place.
pub(crate) fn accumulate_rows(
    keys: &[u8],
    groups: usize,
    tables: &[f64],
    columns: usize,
    scores: &mut [f64],
) {
    assert!((1..=REGISTER_COLUMNS).contains(&columns));
    macro_rules! shapes {
        ($($lanes:literal)*) => {
            match (columns.div_ceil(4), columns % 4) {
                $(
                    ($lanes, 0) => accumulate_shape::<$lanes, 0>(keys, groups, tables, scores),
                    ($lanes, 1) => accumulate_shape::<$lanes, 1>(keys, groups, tables, scores),
                    ($lanes, 2) => accumulate_shape::<$lanes, 2>(keys, groups, tables, scores),
                    ($lanes, 3) => accumulate_shape::<$lanes, 3>(keys, groups, tables, scores),
                )*
                _ => unreachable!("row widths are checked above"),
            }
        };
    }
    shapes!(1 2 3 4 5 6 7 8);
}

/// Rows of `4 * LANES` columns less the padding of a `TAIL`-column last lane, so
/// every lane load and store has a constant size.
#[inline(always)]
fn accumulate_shape<const LANES: usize, const TAIL: usize>(
    keys: &[u8],
    groups: usize,
    tables: &[f64],
    scores: &mut [f64],
) {
    let whole = if TAIL == 0 { LANES } else { LANES - 1 };
    let columns = 4 * whole + TAIL;
    let width = 4 * LANES;
    let table_len = TABLE_ROWS * width;
    let samples = scores.len() / columns;
    assert!(keys.len() >= groups * samples && tables.len() >= groups * table_len);
    // Indexed loops with constant trip counts rather than iterator adapters,
    // which a build without LTO can leave out of line inside this loop.
    for (sample, row) in scores.chunks_exact_mut(columns).enumerate() {
        let mut acc = [Simd::<f64, 4>::splat(0.0); LANES];
        let (lanes, tail) = row.as_chunks_mut::<4>();
        for lane in 0..whole {
            acc[lane] = Simd::from_array(lanes[lane]);
        }
        for column in 0..TAIL {
            acc[whole][column] = tail[column];
        }
        for group in 0..groups {
            let key = keys[group * samples + sample] as usize;
            let (src, _) = tables[group * table_len + key * width..][..width].as_chunks::<4>();
            for lane in 0..LANES {
                acc[lane] += Simd::from_array(src[lane]);
            }
        }
        for lane in 0..whole {
            lanes[lane] = acc[lane].to_array();
        }
        for column in 0..TAIL {
            tail[column] = acc[whole][column];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn butterfly_keys_preserve_every_call_and_padding() {
        for value in 0..65536u32 {
            let bytes = [
                value as u8,
                (value >> 8) as u8,
                !value as u8,
                !(value >> 8) as u8,
            ];
            let sources: Vec<&[u8]> = bytes.iter().map(std::slice::from_ref).collect();
            let actual = transpose_calls(bytes);
            for sample in 0..4 {
                assert_eq!(actual[sample], selected_key(&sources, sample));
            }
        }
        let data = [vec![0x55, 0xe4], vec![0xff, 0x1b], vec![0xaa, 0x3c]];
        let sources: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
        for n in 1..=8 {
            let mut keys = vec![0; n];
            consecutive_keys(&sources, 0, &mut keys);
            for sample in 0..n {
                assert_eq!(keys[sample], selected_key(&sources, sample));
            }
        }
    }

    #[test]
    fn four_variant_keys_match_selected_keys_after_a_tile_offset() {
        let data: Vec<Vec<u8>> = (0..4u32)
            .map(|variant| (0..64u32).map(|byte| (byte * 37 + variant * 101) as u8).collect())
            .collect();
        let sources: Vec<&[u8]> = data.iter().map(Vec::as_slice).collect();
        for sample_start in [0, 4, 64] {
            for n in 1..=(256 - sample_start).min(129) {
                let mut keys = vec![0; n];
                consecutive_keys(&sources, sample_start, &mut keys);
                for sample in 0..n {
                    assert_eq!(keys[sample], selected_key(&sources, sample_start + sample));
                }
            }
        }
    }

    #[test]
    fn accumulated_rows_equal_in_place_additions_bit_for_bit() {
        for columns in 1..=REGISTER_COLUMNS {
            let width = table_columns(columns);
            for groups in [1, 3] {
                let samples = 37;
                let tables: Vec<f64> = (0..groups * TABLE_ROWS * width)
                    .map(|i| {
                        if i % width < columns {
                            ((i * 7919) % 1009) as f64 / 3.0 - 150.0
                        } else {
                            0.0
                        }
                    })
                    .collect();
                let keys: Vec<u8> = (0..groups * samples).map(|i| (i * 151 % 256) as u8).collect();
                let initial: Vec<f64> = (0..samples * columns).map(|i| (i % 13) as f64 / 7.0).collect();
                let mut expected = initial.clone();
                for group in 0..groups {
                    for sample in 0..samples {
                        let key = keys[group * samples + sample] as usize;
                        add_row(
                            &mut expected[sample * columns..(sample + 1) * columns],
                            &tables[(group * TABLE_ROWS + key) * width..][..columns],
                        );
                    }
                }
                let mut actual = initial;
                accumulate_rows(&keys, groups, &tables, columns, &mut actual);
                assert!(
                    actual.iter().zip(&expected).all(|(a, e)| a.to_bits() == e.to_bits()),
                    "{columns} columns, {groups} groups"
                );
            }
        }
    }

    #[test]
    fn lane_tables_equal_row_tables_bit_for_bit() {
        for columns in (4..=REGISTER_COLUMNS).step_by(4) {
            let calls: Vec<f64> = (0..16 * columns)
                .map(|i| ((i * 7919) % 1009) as f64 / 3.0 - 150.0)
                .collect();
            let mut lanes = vec![f64::NAN; TABLE_ROWS * columns];
            let mut rows = vec![f64::NAN; TABLE_ROWS * columns];
            build_table(&calls, columns, &mut lanes);
            build_table_rows(&calls, columns, &mut rows);
            assert!(
                lanes.iter().zip(&rows).all(|(a, b)| a.to_bits() == b.to_bits()),
                "{columns} columns"
            );
        }
    }

    #[test]
    fn every_lookup_matches_the_four_independent_loci() {
        for columns in [1, 3, 4, 9, 20, 64] {
            let calls: Vec<f64> = (0..16 * columns)
                .map(|i| (i * 37 % 127) as f64 / 8.0 - 7.0)
                .collect();
            let mut table = vec![f64::NAN; TABLE_ROWS * columns];
            build_table(&calls, columns, &mut table);
            for key in 0..TABLE_ROWS {
                let mut missing = 0;
                for variant in 0..4 {
                    if (key >> (2 * variant)) & 3 == 1 {
                        missing |= 1 << (2 * variant);
                    }
                }
                assert_eq!(missing_bits(key as u8), missing);
                for column in 0..columns {
                    let expected: f64 = (0..4)
                        .map(|variant| {
                            calls[(4 * variant + ((key >> (2 * variant)) & 3)) * columns + column]
                        })
                        .sum();
                    assert_eq!(table[key * columns + column], expected);
                }
            }
        }
    }
}
