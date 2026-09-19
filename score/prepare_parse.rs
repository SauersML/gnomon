//! Whole-file parsing of local `.bim` and `.fam` files on the thread pool. A local
//! file that fits the memory budget is read in one pass and its lines are parsed in
//! newline-aligned blocks, concurrently. Rows come out exactly as the streaming
//! readers yield them: in file order, numbered across filesets, with each unparsable
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
/// all local `.bim` or `.pvar` files, cannot be read here, or would need more than
/// an eighth of the available memory together with their parsed rows. The caller
/// then streams them, which also reports any I/O error in the usual words. A
/// `.pvar` gives the rows of the virtual `.bim` the streaming reader renders from it.
pub(super) fn parse_local_bims(filesets: &[FilesetPaths]) -> Option<ParsedBim> {
    let (_, available) = crate::memory::memory_bytes();
    parse_local_bims_within(filesets, available / 8, MIN_BLOCK_BYTES)
}

fn is_pvar(path: &Path) -> bool {
    path.extension().is_some_and(|e| e == "pvar")
}

fn parse_local_bims_within(
    filesets: &[FilesetPaths],
    budget: u64,
    min_block_bytes: usize,
) -> Option<ParsedBim> {
    if filesets.iter().any(|f| {
        f.bim.extension().is_none_or(|e| e != "bim") && !is_pvar(&f.bim) || !f.bim.is_file()
    }) {
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
        if is_pvar(&fileset.bim) {
            let pieces = crate::adapt_plink2::render_virtual_bim_pieces(
                file_blocks,
                |piece: &mut PvarPiece, line| piece.take(line, &fileset.bim),
            );
            for piece in pieces {
                let before = parsed.records.len();
                parsed.errors.extend(
                    piece
                        .errors
                        .into_iter()
                        .map(|(i, error)| (before + i, error)),
                );
                let first_row = parsed.total_variants;
                parsed
                    .records
                    .extend(piece.records.into_iter().map(|mut record| {
                        record.bim_row_index.0 += first_row;
                        record
                    }));
                parsed.total_variants += piece.rows;
            }
            continue;
        }
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

/// The rows of one piece of a `.pvar`, numbered from the piece's first row, as the
/// streaming `.bim` reader keys the virtual `.bim` rows it renders: every row takes
/// a number, and a line the renderer refuses is an error in its place that takes
/// none.
#[derive(Default)]
struct PvarPiece {
    records: Vec<KeyedBimRecord>,
    errors: Vec<(usize, PrepError)>,
    rows: u64,
}

impl PvarPiece {
    fn take(
        &mut self,
        line: Result<&crate::adapt_plink2::VirtualBimLines, crate::pipeline_error::PipelineError>,
        path: &Path,
    ) {
        let rows = match line {
            Ok(rows) => rows,
            Err(error) => {
                self.errors.push((
                    self.records.len(),
                    super::map_pipeline_error(error, path.to_path_buf()),
                ));
                return;
            }
        };
        let mut index = 0;
        while let Some(row) = rows.row(index) {
            match parse_bim_row(row, BimRowIndex(self.rows), path) {
                Some(Ok(record)) => self.records.push(record),
                Some(Err(error)) => self.errors.push((self.records.len(), error)),
                None => {}
            }
            self.rows += 1;
            index += 1;
        }
    }
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
    // The virtual `.bim` rows of a `.pgen` carry each `.pvar` record's REF as allele 2.
    let reference_declared = path.extension().is_some_and(|extension| extension == "pvar");
    Some(parse_key(fields[0], fields[3]).map(|key| KeyedBimRecord {
        key,
        bim_row_index: row,
        allele1: Allele::new(fields[4]),
        allele2: Allele::new(fields[5]),
        reference_declared,
    }))
}

/// Every IID of a local `.fam`, in file order, or the error `stream_fam_file` stops
/// at, on the same line. `None` when the file is not a local `.fam`, cannot be read
/// here, or is larger than an eighth of the available memory. The caller then streams
/// it, which also reports any I/O error in the usual words.
pub(super) fn parse_local_fam(path: &Path) -> Option<Result<Vec<String>, PrepError>> {
    let (_, available) = crate::memory::memory_bytes();
    parse_local_fam_within(path, available / 8, MIN_BLOCK_BYTES)
}

fn parse_local_fam_within(
    path: &Path,
    budget: u64,
    min_block_bytes: usize,
) -> Option<Result<Vec<String>, PrepError>> {
    if path.extension().is_none_or(|e| e != "fam") || !path.is_file() {
        return None;
    }
    if std::fs::metadata(path).ok()?.len() > budget {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let blocks = newline_aligned_blocks(&bytes, min_block_bytes);
    let first_lines: Vec<u64> = blocks
        .par_iter()
        .map(|b| line_count(b))
        .collect::<Vec<_>>()
        .into_iter()
        .scan(1, |next, count| {
            let first = *next;
            *next += count;
            Some(first)
        })
        .collect();
    let block_iids: Vec<Result<Vec<String>, PrepError>> = blocks
        .par_iter()
        .zip(first_lines)
        .map(|(block, first_line)| fam_block_iids(block, first_line, path))
        .collect();
    let people = block_iids
        .iter()
        .map(|block| block.as_ref().map_or(0, Vec::len))
        .sum();
    let mut iids = Vec::with_capacity(people);
    for block in block_iids {
        match block {
            Ok(block) => iids.extend(block),
            // Blocks are in file order, so this is the row the stream stops at.
            Err(error) => return Some(Err(error)),
        }
    }
    Some(Ok(iids))
}

/// The IIDs of one block of `.fam` lines whose first line is `first_line`, or the
/// first row error in it.
fn fam_block_iids(block: &[u8], first_line: u64, path: &Path) -> Result<Vec<String>, PrepError> {
    let mut iids = Vec::with_capacity(block.len() / 16);
    for (i, raw_line) in block.split_inclusive(|&b| b == b'\n').enumerate() {
        let line = raw_line.strip_suffix(b"\n").unwrap_or(raw_line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Some(iid) = fam_row_iid(line, first_line + i as u64, path)? {
            iids.push(iid.to_string());
        }
    }
    Ok(iids)
}

/// The IID of one `.fam` line, without its line ending. `None` for an empty line,
/// which occupies a line number but names nobody.
pub(super) fn fam_row_iid<'a>(
    line: &'a [u8],
    line_number: u64,
    path: &Path,
) -> Result<Option<&'a str>, PrepError> {
    if line.is_empty() {
        return Ok(None);
    }
    let text = std::str::from_utf8(line).map_err(|e| {
        PrepError::Parse(format!(
            "Invalid UTF-8 in .fam file '{}' on line {line_number}: {e}",
            path.display()
        ))
    })?;
    match second_field(text) {
        Some(iid) => Ok(Some(iid)),
        None => Err(PrepError::Parse(format!(
            "Missing IID in .fam file '{}' on line {line_number}",
            path.display()
        ))),
    }
}

/// `text.split_whitespace().nth(1)`, scanning bytes when the text is ASCII.
fn second_field(text: &str) -> Option<&str> {
    if !text.is_ascii() {
        return text.split_whitespace().nth(1);
    }
    // In ASCII text these six bytes are exactly the characters that
    // `str::split_whitespace` separates on.
    let is_space = |&b: &u8| matches!(b, b'\t' | b'\n' | 0x0b | 0x0c | b'\r' | b' ');
    let bytes = text.as_bytes();
    let first = bytes.iter().position(|b| !is_space(b))?;
    let gap = first + bytes[first..].iter().position(is_space)?;
    let second = gap + bytes[gap..].iter().position(|b| !is_space(b))?;
    let end = bytes[second..]
        .iter()
        .position(is_space)
        .map_or(bytes.len(), |i| second + i);
    Some(&text[second..end])
}

#[cfg(test)]
mod tests {
    use super::super::{BimIterator, stream_fam_file};
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

    /// A local `.pvar` parses to the rows, row numbers and errors the streaming reader
    /// gives for the virtual `.bim` it renders, on any block size: split multiallelic
    /// sites, sites without an ALT, a header among the data, lines the renderer
    /// refuses, rows the key parser refuses, and a headerless `.pvar` after it.
    #[test]
    fn parsed_pvar_rows_match_streamed_rows_for_any_block_size() {
        let dir = tempfile::tempdir().unwrap();
        let first: &[u8] = b"##fileformat=PVARv1.0\n#CHROM\tPOS\tID\tREF\tALT\n\
            chr1\t100\trs1\tA\tG\n\
            1  200 . AC A,T\r\n\
            \n\
            1\t300\trs3\tA\n\
            1\x0b400\x0crs4 G T\n\
            2\tx\trs5\tA\tC\n\
            Un_gl000220\t9\trs6\tA\tC\n\
            4\t10\trs7\t\xff\tC\n\
            #CHROM\tID\tPOS\tREF\tALT\n\
            6\trs8\t12\tA\t.\n\
            MT\trs9\t13\tA\tC,G\n\
            7\trs10\t1e5\tA\tG";
        let second: &[u8] = b"22 rs11 0 13 A G\n\n\ny rs12 0 14 C T\n";
        let mut filesets = Vec::new();
        for (name, bytes) in [("first", first), ("second", second)] {
            let prefix = dir.path().join(name);
            std::fs::write(prefix.with_extension("pvar"), bytes).unwrap();
            filesets.push(FilesetPaths {
                bed: prefix.with_extension("pgen"),
                bim: prefix.with_extension("pvar"),
                fam: prefix.with_extension("psam"),
            });
        }

        let mut streamed = BimIterator::new(&filesets).unwrap();
        let expected: Vec<String> = streamed
            .by_ref()
            .map(|item| describe(item.as_ref()))
            .collect();
        assert!(expected.iter().any(|d| d.contains("missing ALT")));
        assert!(expected.iter().any(|d| d.contains("Invalid UTF-8")));
        assert!(
            expected
                .iter()
                .any(|d| d.to_lowercase().contains("un_gl000220"))
        );

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
    }

    #[test]
    fn second_field_matches_split_whitespace_for_every_ascii_byte() {
        for byte in 0..=127u8 {
            let c = byte as char;
            for text in [
                format!("F{c}I{c}0"),
                format!("{c}F{c}{c}I"),
                format!("F{c}"),
                c.to_string(),
            ] {
                assert_eq!(
                    second_field(&text),
                    text.split_whitespace().nth(1),
                    "{text:?}"
                );
            }
        }
        for text in ["F\u{a0}I 0", "F\u{3000}I", "F\u{1c}X I", "\u{85}F I"] {
            assert_eq!(
                second_field(text),
                text.split_whitespace().nth(1),
                "{text:?}"
            );
        }
    }

    #[test]
    fn parsed_iids_match_streamed_iids_for_any_block_size() {
        let dir = tempfile::tempdir().unwrap();
        let cases: [(&str, &[u8]); 5] = [
            (
                "mixed",
                b"F1 I1 0 0 1 -9\n\
                F2\tI2\t0\t0\t2\t-9\r\n\
                \n\
                \r\n\
                \x0bF3\x0cI3 0 0 1 1\n\
                F4\xc2\xa0I4 0 0 1 1\n\
                F5 I5",
            ),
            ("trailing_blank_lines", b"F1 I1 0 0 1 -9\n\n\n"),
            ("missing_iid", b"F1 I1 0 0 1 -9\n\nF2\nF3 I3\n"),
            ("invalid_utf8", b"F1 I1\n\n\nF2 \xff\nF3 I3\n"),
            ("empty", b""),
        ];
        let outcome = |result: Result<Vec<String>, PrepError>| match result {
            Ok(iids) => format!("{iids:?}"),
            Err(error) => format!("error {error}"),
        };
        let mut expected = Vec::new();
        for (name, bytes) in cases {
            let path = dir.path().join(format!("{name}.fam"));
            std::fs::write(&path, bytes).unwrap();
            let streamed = outcome(stream_fam_file(&path));
            for block_bytes in [1, 7, 40, 1 << 20] {
                let parsed = parse_local_fam_within(&path, u64::MAX, block_bytes).unwrap();
                assert_eq!(
                    outcome(parsed),
                    streamed,
                    "{name}, block size {block_bytes}"
                );
            }
            expected.push(streamed);
        }
        assert_eq!(expected[0], r#"["I1", "I2", "I3", "I4", "I5"]"#);
        assert!(
            expected[2].contains("Missing IID") && expected[2].contains("line 3"),
            "{}",
            expected[2]
        );
        assert!(
            expected[3].contains("Invalid UTF-8") && expected[3].contains("line 4"),
            "{}",
            expected[3]
        );

        assert!(parse_local_fam_within(&dir.path().join("mixed.fam"), 8, 1).is_none());
        let psam = dir.path().join("people.psam");
        std::fs::write(&psam, b"#IID\nI1\n").unwrap();
        assert!(parse_local_fam_within(&psam, u64::MAX, 1).is_none());
    }
}
