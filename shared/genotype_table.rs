//! Four-locus lookup arithmetic over PLINK's four two-bit call states.
//! Tables are bounded scratch space, not a cohort-sized genotype expansion.

use std::simd::Simd;

pub(crate) const VARIANTS_PER_TABLE: usize = 4;
pub(crate) const TABLE_ROWS: usize = 256;
pub(crate) const TABLE_BUDGET_BYTES: usize = 256 * 1024;
pub(crate) const SAMPLE_TILE: usize = 1024;

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
