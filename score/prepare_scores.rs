//! Whole-file parsing of normalized score files on the thread pool, for Stage 3.
//! Score files that fit the memory budget are read in one pass and their data
//! lines parsed in newline-aligned blocks, concurrently. Records come out in the
//! order the streaming k-way merge yields them, with the same errors at the same
//! places and the same count of malformed lines at every point of consumption.
use super::{Allele, KeyedScoreRecord, PrepError, VariantKey, parse_key};
use crate::score::types::ScoreColumnIndex;
use ahash::AHashMap;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::io;
use std::path::PathBuf;

/// Lines this short are not worth a task of their own.
const MIN_BLOCK_BYTES: usize = 1 << 20;

/// Every record of a list of score files, merged by key and then by file.
pub(super) struct ParsedScores {
    files: Vec<ParsedFile>,
    heap: BinaryHeap<Reverse<(VariantKey, usize)>>,
    next_error: Option<PrepError>,
}

struct ParsedFile {
    records: std::vec::IntoIter<KeyedScoreRecord>,
    /// For each line that yielded weights, the number of records through it.
    line_ends: Vec<usize>,
    /// For each of those lines, the malformed lines before it.
    malformed_before: Vec<usize>,
    /// Malformed lines before the end of the file, or before its error.
    malformed_at_end: usize,
    /// The error that ends the file, met after its last yielding line.
    error: Option<PrepError>,
    popped: usize,
    line: usize,
    malformed: usize,
}

/// One newline-aligned piece of a file's data section, parsed.
struct Block {
    records: Vec<KeyedScoreRecord>,
    /// (records through the line, malformed lines before it) per yielding line.
    lines: Vec<(usize, usize)>,
    malformed: usize,
    error: Option<PrepError>,
}

/// Parses every score file at once, or returns `None` when a file cannot be read
/// here, when the files and their records would need more than an eighth of the
/// available memory, or when the streaming merge would stop before yielding
/// anything. The caller then streams, which reports any such error in its words.
pub(super) fn parse_score_files(
    paths: &[PathBuf],
    score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
) -> Option<ParsedScores> {
    let (_, available) = crate::memory::memory_bytes();
    parse_score_files_within(
        paths,
        score_name_to_col_index,
        available / 8,
        MIN_BLOCK_BYTES,
    )
}

fn parse_score_files_within(
    paths: &[PathBuf],
    score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
    budget: u64,
    min_block_bytes: usize,
) -> Option<ParsedScores> {
    let mut total_bytes = 0u64;
    for path in paths {
        total_bytes = total_bytes.saturating_add(std::fs::metadata(path).ok()?.len());
        if total_bytes > budget {
            return None;
        }
    }
    let contents: Vec<Vec<u8>> = paths
        .iter()
        .map(|path| std::fs::read(path).ok())
        .collect::<Option<_>>()?;
    let mut sections = Vec::with_capacity(paths.len());
    for bytes in &contents {
        let (data_start, column_map) = parse_header(bytes, score_name_to_col_index)?;
        sections.push((&bytes[data_start..], column_map));
    }
    let blocks: Vec<Vec<&[u8]>> = sections
        .iter()
        .map(|(data, _)| newline_aligned_blocks(data, min_block_bytes))
        .collect();
    let block_lines: Vec<Vec<u64>> = blocks
        .iter()
        .map(|file_blocks| file_blocks.par_iter().map(|b| line_count(b)).collect())
        .collect();
    let record_bytes = sections
        .iter()
        .zip(&block_lines)
        .map(|((_, map), lines)| lines.iter().sum::<u64>() * map.len().max(1) as u64)
        .sum::<u64>()
        .saturating_mul(std::mem::size_of::<KeyedScoreRecord>() as u64);
    if total_bytes.saturating_add(record_bytes) > budget {
        return None;
    }

    let mut files = Vec::with_capacity(paths.len());
    for (((_, column_map), file_blocks), lines) in sections.iter().zip(&blocks).zip(&block_lines) {
        let first_lines: Vec<u64> = lines
            .iter()
            .scan(1u64, |next, &count| {
                let first = *next;
                *next += count;
                Some(first)
            })
            .collect();
        let parsed: Vec<Block> = file_blocks
            .par_iter()
            .zip(first_lines)
            .map(|(block, first_line)| parse_block(block, first_line, column_map))
            .collect();
        let mut file = ParsedFile {
            records: Vec::new().into_iter(),
            line_ends: Vec::new(),
            malformed_before: Vec::new(),
            malformed_at_end: 0,
            error: None,
            popped: 0,
            line: 0,
            malformed: 0,
        };
        let mut records = Vec::new();
        for block in parsed {
            let (before_records, before_malformed) = (records.len(), file.malformed_at_end);
            for (end, malformed) in block.lines {
                file.line_ends.push(before_records + end);
                file.malformed_before.push(before_malformed + malformed);
            }
            records.extend(block.records);
            file.malformed_at_end += block.malformed;
            if block.error.is_some() {
                file.error = block.error;
                break;
            }
        }
        file.records = records.into_iter();
        files.push(file);
    }

    let mut heap = BinaryHeap::with_capacity(files.len());
    for (index, file) in files.iter_mut().enumerate() {
        match file.records.as_slice().first() {
            Some(record) => {
                file.malformed = file.malformed_before[0];
                heap.push(Reverse((record.key, index)));
            }
            // The streaming merge fails while opening this file.
            None if file.error.is_some() => return None,
            None => file.malformed = file.malformed_at_end,
        }
    }
    Some(ParsedScores {
        files,
        heap,
        next_error: None,
    })
}

impl ParsedScores {
    /// Lines skipped for missing columns among those the streaming merge would
    /// have read by now.
    pub(super) fn malformed_lines(&self) -> usize {
        self.files.iter().map(|file| file.malformed).sum()
    }
}

impl Iterator for ParsedScores {
    type Item = Result<KeyedScoreRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.next_error.take() {
            return Some(Err(error));
        }
        let Reverse((_, index)) = self.heap.pop()?;
        let file = &mut self.files[index];
        let record = file.records.next()?;
        file.popped += 1;
        if file.popped == file.line_ends[file.line] {
            // The stream reads on to the next yielding line, or to the end.
            file.line += 1;
            if file.line < file.line_ends.len() {
                file.malformed = file.malformed_before[file.line];
            } else {
                file.malformed = file.malformed_at_end;
                self.next_error = file.error.take();
            }
        }
        if let Some(next) = file.records.as_slice().first()
            && file.line < file.line_ends.len()
        {
            self.heap.push(Reverse((next.key, index)));
        }
        Some(Ok(record))
    }
}

/// Where the data section starts and how its weight columns map to scores, read
/// as `KWayMergeIterator::new` reads them. `None` wherever that would fail.
fn parse_header(
    bytes: &[u8],
    score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
) -> Option<(usize, Vec<ScoreColumnIndex>)> {
    let mut start = 0;
    let mut header = "";
    for raw in bytes.split_inclusive(|&b| b == b'\n') {
        let line = std::str::from_utf8(raw).ok()?;
        start += raw.len();
        if !line.trim().is_empty() && !line.starts_with('#') {
            header = line;
            break;
        }
    }
    let column_map = header
        .trim()
        .split('\t')
        .skip(3)
        .map(|name| score_name_to_col_index.get(name).copied())
        .collect::<Option<_>>()?;
    Some((start, column_map))
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

/// Parses data lines as `KWayMergeIterator::read_line_into_buffer` does, line by
/// line, stopping at the first error. `first_line` numbers the block's first line
/// within the data section, from 1.
fn parse_block(block: &[u8], first_line: u64, column_map: &[ScoreColumnIndex]) -> Block {
    let mut parsed = Block {
        records: Vec::with_capacity(block.len() / 32),
        lines: Vec::new(),
        malformed: 0,
        error: None,
    };
    for (i, raw) in block.split_inclusive(|&b| b == b'\n').enumerate() {
        let line_number = first_line + i as u64;
        let Ok(line) = std::str::from_utf8(raw) else {
            parsed.error = Some(PrepError::Io(
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "stream did not contain valid UTF-8",
                ),
                PathBuf::new(),
            ));
            break;
        };
        match parse_line(line, line_number, column_map, &mut parsed.records) {
            Ok(LineOutcome::Ignored) => {}
            Ok(LineOutcome::Malformed) => parsed.malformed += 1,
            Ok(LineOutcome::Yielded) => parsed.lines.push((parsed.records.len(), parsed.malformed)),
            Err(error) => {
                parsed.error = Some(error);
                break;
            }
        }
    }
    parsed
}

enum LineOutcome {
    Ignored,
    Malformed,
    Yielded,
}

fn parse_line(
    line: &str,
    line_number: u64,
    column_map: &[ScoreColumnIndex],
    records: &mut Vec<KeyedScoreRecord>,
) -> Result<LineOutcome, PrepError> {
    if line.trim().is_empty() || line.starts_with('#') {
        return Ok(LineOutcome::Ignored);
    }
    let mut parts = line.split('\t');
    let (variant_id, effect_allele, other_allele) = match (parts.next(), parts.next(), parts.next())
    {
        (Some(v), Some(e), Some(o)) if !v.is_empty() && !e.is_empty() && !o.is_empty() => (v, e, o),
        _ => return Ok(LineOutcome::Malformed),
    };
    if other_allele == "N" {
        return Err(PrepError::Parse(format!(
            "Score file line {line_number} has unknown other_allele 'N'. Scores must provide an explicit allele pair."
        )));
    }
    let mut key_parts = variant_id.splitn(2, ':');
    let chr_str = key_parts.next().unwrap_or("");
    let pos_str = key_parts.next().unwrap_or("");
    let key = parse_key(chr_str, pos_str)?;
    let effect_allele = Allele::new(effect_allele);
    let other_allele = Allele::new(other_allele);
    let start = records.len();
    for (i, weight_str) in parts.enumerate() {
        let weight_str = weight_str.trim();
        if weight_str.is_empty() {
            continue;
        }
        let Some(&score_column_index) = column_map.get(i) else {
            continue;
        };
        let weight = match weight_str.parse::<f32>() {
            Ok(weight) => weight,
            Err(err) => {
                // The stream drops the line's buffered weights with the error.
                records.truncate(start);
                return Err(PrepError::Parse(format!(
                    "Invalid weight '{}' in score file line {}, column {}: {}",
                    weight_str,
                    line_number,
                    i + 4,
                    err
                )));
            }
        };
        records.push(KeyedScoreRecord {
            key,
            effect_allele: effect_allele.clone(),
            other_allele: other_allele.clone(),
            score_column_index,
            weight,
        });
    }
    Ok(if records.len() > start {
        LineOutcome::Yielded
    } else {
        LineOutcome::Ignored
    })
}

#[cfg(test)]
mod tests {
    use super::super::KWayMergeIterator;
    use super::*;
    use std::path::Path;

    fn describe(item: Result<KeyedScoreRecord, PrepError>) -> String {
        match item {
            Ok(r) => format!(
                "{:?} {} {} col {} weight {:08x}",
                r.key,
                r.effect_allele,
                r.other_allele,
                r.score_column_index.0,
                r.weight.to_bits()
            ),
            Err(e) => format!("error {e}"),
        }
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn parsed_records_match_the_streaming_merge_at_every_point() {
        let dir = tempfile::tempdir().unwrap();
        let columns: AHashMap<String, ScoreColumnIndex> = ["A", "B", "C"]
            .iter()
            .enumerate()
            .map(|(i, name)| (name.to_string(), ScoreColumnIndex(i)))
            .collect();
        let cases: Vec<Vec<&[u8]>> = vec![
            vec![
                b"#meta\n\nvariant_id\teffect_allele\tother_allele\tA\tB\n\
                  1:100\tG\tA\t0.5\t1.5\n\
                  1:100\tT\tA\t\t2\n\
                  bad line\n\
                  \t\t\n\
                  1:150\tAC\tA\t0.25\n\
                  # comment\n\
                  1:200\tC\tT\t  \t\r\n\
                  2:5\tG\tC\t3\t4\r\n\
                  X:7\tA\tG\t1e-3\t-0",
                b"variant_id\teffect_allele\tother_allele\tC\n\
                  1:100\tA\tC\t9\n\
                  1:120\tA\tC\t8\n\
                  2:5\tA\tC\t7\n\
                  malformed\n\
                  3:1\tA\tC\t6\n",
            ],
            vec![
                b"variant_id\teffect_allele\tother_allele\tA\tB\n\
                  1:1\tA\tG\t1\t2\n\
                  junk\n\
                  1:2\tA\tG\t1\tnope\n\
                  1:3\tA\tG\t1\t2\n",
                b"variant_id\teffect_allele\tother_allele\tC\n\
                  1:1\tA\tG\t5\n\
                  1:4\tA\tN\t5\n\
                  1:5\tA\tG\t5\n",
            ],
            vec![
                b"variant_id\teffect_allele\tother_allele\tA\n1:1\tA\tG\t1\n2:2\tA\tG\t1\n\xff\n3:3\tA\tG\t1\n",
                b"variant_id\teffect_allele\tother_allele\tB\n1:9\tA\tG\t2\nchrUn_x:4\tA\tG\t2\n",
            ],
        ];
        for (case, contents) in cases.iter().enumerate() {
            let paths: Vec<PathBuf> = contents
                .iter()
                .enumerate()
                .map(|(i, bytes)| write(dir.path(), &format!("case{case}_{i}.tsv"), bytes))
                .collect();
            let mut streamed = KWayMergeIterator::new(&paths, &columns, None).unwrap();
            let mut expected = Vec::new();
            loop {
                let malformed: usize = streamed
                    .streams
                    .iter()
                    .map(|s| s.malformed_lines_count)
                    .sum();
                match streamed.next() {
                    Some(item) => expected.push((malformed, describe(item))),
                    None => {
                        expected.push((malformed, "end".to_string()));
                        break;
                    }
                }
            }
            for block_bytes in [1, 9, 64, 1 << 20] {
                let mut parsed =
                    parse_score_files_within(&paths, &columns, u64::MAX, block_bytes).unwrap();
                let mut actual = Vec::new();
                loop {
                    let malformed = parsed.malformed_lines();
                    match parsed.next() {
                        Some(item) => actual.push((malformed, describe(item))),
                        None => {
                            actual.push((malformed, "end".to_string()));
                            break;
                        }
                    }
                }
                assert_eq!(actual, expected, "case {case}, block size {block_bytes}");
            }
        }
    }

    #[test]
    fn opening_failures_are_left_to_the_streaming_merge() {
        let dir = tempfile::tempdir().unwrap();
        let columns: AHashMap<String, ScoreColumnIndex> = [("A".to_string(), ScoreColumnIndex(0))]
            .into_iter()
            .collect();
        let first_line_error = write(
            dir.path(),
            "first.tsv",
            b"variant_id\teffect_allele\tother_allele\tA\n1:1\tA\tG\tbad\n",
        );
        assert!(
            KWayMergeIterator::new(std::slice::from_ref(&first_line_error), &columns, None)
                .is_err()
        );
        assert!(parse_score_files_within(&[first_line_error], &columns, u64::MAX, 1).is_none());
        let unknown_score = write(
            dir.path(),
            "unknown.tsv",
            b"variant_id\teffect_allele\tother_allele\tZ\n",
        );
        assert!(parse_score_files_within(&[unknown_score], &columns, u64::MAX, 1).is_none());
        let fine = write(
            dir.path(),
            "fine.tsv",
            b"variant_id\teffect_allele\tother_allele\tA\n1:1\tA\tG\t1\n",
        );
        assert!(parse_score_files_within(std::slice::from_ref(&fine), &columns, 8, 1).is_none());
        assert!(parse_score_files_within(&[fine], &columns, u64::MAX, 1).is_some());
    }
}
