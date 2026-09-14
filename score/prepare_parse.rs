//! Whole-file parsing of local `.bim` files on the thread pool. A local `.bim`
//! that fits the memory budget is read in one pass and its lines are parsed in
//! newline-aligned blocks, concurrently. Rows come out exactly as `BimIterator`
//! streams them: in file order, numbered across filesets, with each unparsable
//! row kept as the same error at the same place.
use super::{Allele, FilesetPaths, KeyedBimRecord, PrepError, parse_key};
use crate::score::types::{BimRowIndex, FilesetBoundary};
use rayon::prelude::*;
use std::path::Path;

/// Lines this short are not worth a task of their own.
const MIN_BLOCK_BYTES: usize = 1 << 20;

/// Every row of a list of local `.bim` files, in file order.
pub(super) struct ParsedBim {
    /// Rows that parsed.
    pub records: Vec<KeyedBimRecord>,
    /// Rows that did not, as (number of parsed rows before it, error), ascending.
    pub errors: Vec<(usize, PrepError)>,
    pub boundaries: Vec<FilesetBoundary>,
    pub total_variants: u64,
}

/// Parses every `.bim` of `filesets` at once, or returns `None` when they are not
/// all local `.bim` files, cannot be read here, or would need more than an eighth
/// of the available memory together with their parsed rows. The caller then
/// streams them, which also reports any I/O error in the usual words.
pub(super) fn parse_local_bims(filesets: &[FilesetPaths]) -> Option<ParsedBim> {
    let (_, available) = crate::memory::memory_bytes();
    parse_local_bims_within(filesets, available / 8, MIN_BLOCK_BYTES)
}

fn parse_local_bims_within(
    filesets: &[FilesetPaths],
    budget: u64,
    min_block_bytes: usize,
) -> Option<ParsedBim> {
    if filesets
        .iter()
        .any(|f| f.bim.extension().is_none_or(|e| e != "bim") || !f.bim.is_file())
    {
        return None;
    }
    let mut total_bytes = 0u64;
    for fileset in filesets {
        total_bytes = total_bytes.saturating_add(std::fs::metadata(&fileset.bim).ok()?.len());
        if total_bytes > budget {
            return None;
        }
    }
    let contents: Vec<Vec<u8>> = filesets
        .iter()
        .map(|fileset| std::fs::read(&fileset.bim).ok())
        .collect::<Option<_>>()?;
    let blocks: Vec<Vec<&[u8]>> = contents
        .iter()
        .map(|bytes| newline_aligned_blocks(bytes, min_block_bytes))
        .collect();
    let block_lines: Vec<Vec<u64>> = blocks
        .iter()
        .map(|file_blocks| file_blocks.par_iter().map(|b| line_count(b)).collect())
        .collect();
    let total_lines: u64 = block_lines.iter().flatten().sum();
    let record_bytes = total_lines.saturating_mul(std::mem::size_of::<KeyedBimRecord>() as u64);
    if total_bytes.saturating_add(record_bytes) > budget {
        return None;
    }

    let mut parsed = ParsedBim {
        records: Vec::with_capacity(total_lines as usize),
        errors: Vec::new(),
        boundaries: Vec::with_capacity(filesets.len()),
        total_variants: 0,
    };
    for ((fileset, file_blocks), lines) in filesets.iter().zip(&blocks).zip(&block_lines) {
        parsed.boundaries.push(FilesetBoundary {
            bed_path: fileset.bed.clone(),
            bim_path: fileset.bim.clone(),
            fam_path: fileset.fam.clone(),
            starting_global_index: parsed.total_variants,
        });
        let first_rows: Vec<u64> = lines
            .iter()
            .scan(parsed.total_variants, |next, &count| {
                let first = *next;
                *next += count;
                Some(first)
            })
            .collect();
        let block_rows: Vec<(Vec<KeyedBimRecord>, Vec<(usize, PrepError)>)> = file_blocks
            .par_iter()
            .zip(first_rows)
            .map(|(block, first_row)| parse_block(block, first_row, &fileset.bim))
            .collect();
        for (records, errors) in block_rows {
            let before = parsed.records.len();
            parsed
                .errors
                .extend(errors.into_iter().map(|(i, error)| (before + i, error)));
            parsed.records.extend(records);
        }
        parsed.total_variants += lines.iter().sum::<u64>();
    }
    Some(parsed)
}

/// Splits `bytes` into pieces of about a pool task each, every piece but the last
/// ending just after a newline.
fn newline_aligned_blocks(bytes: &[u8], min_block_bytes: usize) -> Vec<&[u8]> {
    let target = (bytes.len() / (4 * rayon::current_num_threads()).max(1)).max(min_block_bytes);
    let mut blocks = Vec::new();
    let mut start = 0;
    while start < bytes.len() {
        let end = if bytes.len() - start <= target {
            bytes.len()
        } else {
            memchr::memchr(b'\n', &bytes[start + target..])
                .map_or(bytes.len(), |i| start + target + i + 1)
        };
        blocks.push(&bytes[start..end]);
        start = end;
    }
    blocks
}

/// Lines as a line reader sees them: a final line without a newline counts.
fn line_count(block: &[u8]) -> u64 {
    let newlines = memchr::memchr_iter(b'\n', block).count() as u64;
    newlines + u64::from(block.last().is_some_and(|&b| b != b'\n'))
}

fn parse_block(
    block: &[u8],
    first_row: u64,
    path: &Path,
) -> (Vec<KeyedBimRecord>, Vec<(usize, PrepError)>) {
    let mut records = Vec::with_capacity(block.len() / 24);
    let mut errors = Vec::new();
    for (i, raw_line) in block.split_inclusive(|&b| b == b'\n').enumerate() {
        let line = raw_line.strip_suffix(b"\n").unwrap_or(raw_line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        match parse_bim_row(line, BimRowIndex(first_row + i as u64), path) {
            Some(Ok(record)) => records.push(record),
            Some(Err(error)) => errors.push((records.len(), error)),
            None => {}
        }
    }
    (records, errors)
}

/// Parses one `.bim` line, without its line ending. `None` for a line with fewer
/// than six fields, which occupies a row but yields nothing.
pub(super) fn parse_bim_row(
    line: &[u8],
    row: BimRowIndex,
    path: &Path,
) -> Option<Result<KeyedBimRecord, PrepError>> {
    let text = match std::str::from_utf8(line) {
        Ok(text) => text,
        Err(e) => {
            return Some(Err(PrepError::Parse(format!(
                "Invalid UTF-8 in BIM file '{}': {e}",
                path.display()
            ))));
        }
    };
    let mut fields = [""; 6];
    let mut found = 0;
    if text.is_ascii() {
        // In ASCII text these six bytes are exactly the characters that
        // `str::split_whitespace` separates on.
        let bytes = text.as_bytes();
        let is_space = |b: u8| matches!(b, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' ');
        let mut i = 0;
        while found < fields.len() {
            while i < bytes.len() && is_space(bytes[i]) {
                i += 1;
            }
            if i == bytes.len() {
                break;
            }
            let start = i;
            while i < bytes.len() && !is_space(bytes[i]) {
                i += 1;
            }
            fields[found] = &text[start..i];
            found += 1;
        }
    } else {
        for (slot, field) in fields.iter_mut().zip(text.split_whitespace()) {
            *slot = field;
            found += 1;
        }
    }
    if found < fields.len() {
        return None;
    }
    Some(parse_key(fields[0], fields[3]).map(|key| KeyedBimRecord {
        key,
        bim_row_index: row,
        allele1: Allele::new(fields[4]),
        allele2: Allele::new(fields[5]),
    }))
}

#[cfg(test)]
mod tests {
    use super::super::BimIterator;
    use super::*;

    fn describe(item: Result<&KeyedBimRecord, &PrepError>) -> String {
        match item {
            Ok(r) => format!(
                "row {} key {:?} {} {}",
                r.bim_row_index.0, r.key, r.allele1, r.allele2
            ),
            Err(e) => format!("error {e}"),
        }
    }

    #[test]
    fn parsed_rows_match_streamed_rows_for_any_block_size() {
        let dir = tempfile::tempdir().unwrap();
        let first: &[u8] = b"1 rs1 0 100 A G\n\
            1\trs2\t0\t200\tAC\tA\r\n\
            \n\
            1 short 0 300 A\n\
            chrX rs3 0 5 T C\n\
            1\x0brs4\x0c0 400 G T\n\
            26 rs5 0 +7 A C\n\
            +2 rs6 0 8 A C\n\
            Un_gl000220 rs7 0 9 A C\n\
            3 rs8 0 1e5 A C\n\
            4 rs9 0 10 \xff C\n\
            5\xc2\xa0rs10 0 11 A C T\n\
            MT rs11 0 12 A C";
        let second: &[u8] = b"22 rs12 0 13 A G\n\n\ny rs13 0 14 C T\n";
        let mut filesets = Vec::new();
        for (name, bytes) in [("first", first), ("second", second)] {
            let prefix = dir.path().join(name);
            std::fs::write(prefix.with_extension("bim"), bytes).unwrap();
            filesets.push(FilesetPaths {
                bed: prefix.with_extension("bed"),
                bim: prefix.with_extension("bim"),
                fam: prefix.with_extension("fam"),
            });
        }

        let mut streamed = BimIterator::new(&filesets).unwrap();
        let expected: Vec<String> = streamed
            .by_ref()
            .map(|item| describe(item.as_ref()))
            .collect();
        assert!(expected.iter().any(|d| d.contains("Invalid UTF-8")));
        assert!(expected.iter().any(|d| d.contains("1e5")));

        for block_bytes in [1, 7, 40, 1 << 20] {
            let parsed = parse_local_bims_within(&filesets, u64::MAX, block_bytes).unwrap();
            let mut actual = Vec::new();
            let mut errors = parsed.errors.iter().peekable();
            for (i, record) in parsed.records.iter().enumerate() {
                while let Some((_, error)) = errors.next_if(|(before, _)| *before == i) {
                    actual.push(describe(Err(error)));
                }
                actual.push(describe(Ok(record)));
            }
            actual.extend(errors.map(|(_, error)| describe(Err(error))));
            assert_eq!(actual, expected, "block size {block_bytes}");
            assert_eq!(parsed.total_variants, streamed.total_variants());
            let starts = |b: &[FilesetBoundary]| {
                b.iter()
                    .map(|b| (b.bim_path.clone(), b.starting_global_index))
                    .collect::<Vec<_>>()
            };
            assert_eq!(starts(&parsed.boundaries), starts(&streamed.boundaries));
        }
        assert!(parse_local_bims_within(&filesets, 64, 1).is_none());
    }
}
