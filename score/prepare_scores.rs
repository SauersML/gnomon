//! Whole-file parsing of normalized score files on the thread pool, for Stage 3.
//! Score files that fit the memory budget are read in one pass and their data
//! lines parsed in newline-aligned blocks, concurrently. Records come out in the
//! order the streaming k-way merge yields them, with the same errors at the same
//! places and the same count of malformed lines at every point of consumption.
use super::{
    Allele, KeyedScoreRecord, PrepError, RejectedScoreRows, VariantKey, is_unkeyable_contig,
    parse_key, parse_weight, unusable_weight_error,
};
use crate::score::types::ScoreColumnIndex;
use ahash::AHashMap;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

/// Lines this short are not worth a task of their own.
const MIN_BLOCK_BYTES: usize = 1 << 20;

/// Every record of a list of score files, merged by key and then by file.
pub(super) struct ParsedScores {
    files: Vec<ParsedFile>,
    heap: BinaryHeap<Reverse<(VariantKey, usize)>>,
    /// Errors to yield before the next record, as the stream would meet them.
    pending: VecDeque<PrepError>,
    /// Every rejected row and weight field of every file.
    rejected: RejectedScoreRows,
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
    /// Rows on contigs gnomon cannot key, as (yielding lines before it, error).
    rejections: VecDeque<(usize, PrepError)>,
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
    /// (yielding lines before it, error) per row on an unkeyable contig.
    rejections: Vec<(usize, PrepError)>,
    rejected: RejectedScoreRows,
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
    // Every open, stat and read is a round trip on a network filesystem, so a list
    // of many small score files is opened, measured and read in parallel. Nothing is
    // read until the lengths fit the budget, and contents keep file order.
    let files: Vec<(File, u64)> = paths
        .par_iter()
        .map(|path| {
            let file = File::open(path).ok()?;
            let len = file.metadata().ok()?.len();
            Some((file, len))
        })
        .collect::<Option<_>>()?;
    if files
        .iter()
        .fold(0u64, |total, (_, len)| total.saturating_add(*len))
        > budget
    {
        return None;
    }
    let contents: Vec<Vec<u8>> = files
        .into_par_iter()
        .map(|(mut file, len)| {
            let mut bytes = Vec::with_capacity(usize::try_from(len).ok()?);
            file.read_to_end(&mut bytes).ok()?;
            Some(bytes)
        })
        .collect::<Option<_>>()?;
    let views: Vec<&[u8]> = contents.iter().map(Vec::as_slice).collect();
    parse_score_contents(
        paths,
        &views,
        score_name_to_col_index,
        budget,
        min_block_bytes,
    )
}

/// Parses score files that are already in memory, as `parse_score_files` parses them
/// from disk. `paths_for_messages` names each of `contents`, in the same order.
pub(super) fn parse_score_contents(
    paths_for_messages: &[PathBuf],
    contents: &[&[u8]],
    score_name_to_col_index: &AHashMap<String, ScoreColumnIndex>,
    budget: u64,
    min_block_bytes: usize,
) -> Option<ParsedScores> {
    if paths_for_messages.len() != contents.len() {
        return None;
    }
    let total_bytes = contents.iter().fold(0u64, |total, bytes| {
        total.saturating_add(bytes.len() as u64)
    });
    if total_bytes > budget {
        return None;
    }
    let mut sections = Vec::with_capacity(contents.len());
    for bytes in contents {
        let (data_start, column_map) = parse_header(bytes, score_name_to_col_index)?;
        let header_lines = memchr::memchr_iter(b'\n', &bytes[..data_start]).count() as u64;
        sections.push((&bytes[data_start..], column_map, header_lines));
    }
    let blocks: Vec<Vec<&[u8]>> = sections
        .iter()
        .map(|(data, _, _)| newline_aligned_blocks(data, min_block_bytes))
        .collect();
    let block_lines: Vec<Vec<u64>> = blocks
        .iter()
        .map(|file_blocks| file_blocks.par_iter().map(|b| line_count(b)).collect())
        .collect();
    let record_bytes = sections
        .iter()
        .zip(&block_lines)
        .map(|((_, map, _), lines)| lines.iter().sum::<u64>() * map.len().max(1) as u64)
        .sum::<u64>()
        .saturating_mul(std::mem::size_of::<KeyedScoreRecord>() as u64);
    if total_bytes.saturating_add(record_bytes) > budget {
        return None;
    }

    let mut files = Vec::with_capacity(contents.len());
    let mut rejected = RejectedScoreRows::default();
    for ((((_, column_map, header_lines), file_blocks), lines), path) in sections
        .iter()
        .zip(&blocks)
        .zip(&block_lines)
        .zip(paths_for_messages)
    {
        // Physical line numbers, as the streaming merge counts them.
        let first_lines: Vec<u64> = lines
            .iter()
            .scan(header_lines + 1, |next, &count| {
                let first = *next;
                *next += count;
                Some(first)
            })
            .collect();
        let parsed: Vec<Block> = file_blocks
            .par_iter()
            .zip(first_lines)
            .map(|(block, first_line)| parse_block(block, first_line, column_map, path))
            .collect();
        let mut file = ParsedFile {
            records: Vec::new().into_iter(),
            line_ends: Vec::new(),
            malformed_before: Vec::new(),
            malformed_at_end: 0,
            error: None,
            rejections: VecDeque::new(),
            popped: 0,
            line: 0,
            malformed: 0,
        };
        let mut records = Vec::new();
        for block in parsed {
            let (before_records, before_malformed, before_lines) =
                (records.len(), file.malformed_at_end, file.line_ends.len());
            for (end, malformed) in block.lines {
                file.line_ends.push(before_records + end);
                file.malformed_before.push(before_malformed + malformed);
            }
            for (yielded, error) in block.rejections {
                file.rejections.push_back((before_lines + yielded, error));
            }
            rejected.absorb(block.rejected);
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
    let mut pending = VecDeque::new();
    for (index, file) in files.iter_mut().enumerate() {
        // Opening a stream reads on to its first yielding line.
        while file
            .rejections
            .front()
            .is_some_and(|(before, _)| *before == 0)
        {
            pending.extend(file.rejections.pop_front().map(|(_, error)| error));
        }
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
        pending,
        rejected,
    })
}

impl ParsedScores {
    /// Lines skipped for missing columns among those the streaming merge would
    /// have read by now.
    pub(super) fn malformed_lines(&self) -> usize {
        self.files.iter().map(|file| file.malformed).sum()
    }

    /// Any error that fails the run but was not yielded yet, then every rejected
    /// row and weight field. Parsing already read every line.
    pub(super) fn finish(&mut self) -> Result<RejectedScoreRows, PrepError> {
        if let Some(position) = self
            .pending
            .iter()
            .position(|error| !is_unkeyable_contig(error))
            && let Some(error) = self.pending.remove(position)
        {
            return Err(error);
        }
        for file in &mut self.files {
            if let Some(error) = file.error.take() {
                return Err(error);
            }
        }
        Ok(std::mem::take(&mut self.rejected))
    }
}

impl Iterator for ParsedScores {
    type Item = Result<KeyedScoreRecord, PrepError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(error) = self.pending.pop_front() {
            return Some(Err(error));
        }
        let Reverse((_, index)) = self.heap.pop()?;
        let file = &mut self.files[index];
        let record = file.records.next()?;
        file.popped += 1;
        if file.popped == file.line_ends[file.line] {
            // The stream reads on to the next yielding line, or to the end.
            file.line += 1;
            while file
                .rejections
                .front()
                .is_some_and(|(before, _)| *before == file.line)
            {
                self.pending
                    .extend(file.rejections.pop_front().map(|(_, error)| error));
            }
            if file.line < file.line_ends.len() {
                file.malformed = file.malformed_before[file.line];
            } else {
                file.malformed = file.malformed_at_end;
                self.pending.extend(file.error.take());
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
/// line, stopping at the first error that fails the run. `first_line` is the
/// physical line number of the block's first line.
fn parse_block(
    block: &[u8],
    first_line: u64,
    column_map: &[ScoreColumnIndex],
    path: &Path,
) -> Block {
    let mut parsed = Block {
        records: Vec::with_capacity(block.len() / 32),
        lines: Vec::new(),
        malformed: 0,
        error: None,
        rejections: Vec::new(),
        rejected: RejectedScoreRows::default(),
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
        match parse_line(
            line,
            line_number,
            column_map,
            path,
            &mut parsed.records,
            &mut parsed.rejected,
        ) {
            Ok(LineOutcome::Ignored) => {}
            Ok(LineOutcome::Malformed) => parsed.malformed += 1,
            Ok(LineOutcome::Yielded) => parsed.lines.push((parsed.records.len(), parsed.malformed)),
            Ok(LineOutcome::Rejected(error)) => parsed.rejections.push((parsed.lines.len(), error)),
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
    /// A row on a contig gnomon cannot key: yielded as its error, then skipped.
    Rejected(PrepError),
}

fn parse_line(
    line: &str,
    line_number: u64,
    column_map: &[ScoreColumnIndex],
    path: &Path,
    records: &mut Vec<KeyedScoreRecord>,
    rejected: &mut RejectedScoreRows,
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
        rejected.record(path, line_number);
        return Ok(LineOutcome::Ignored);
    }
    let mut key_parts = variant_id.splitn(2, ':');
    let chr_str = key_parts.next().unwrap_or("");
    let pos_str = key_parts.next().unwrap_or("");
    let key = match parse_key(chr_str, pos_str) {
        Ok(key) => key,
        Err(error) if is_unkeyable_contig(&error) => return Ok(LineOutcome::Rejected(error)),
        Err(PrepError::Parse(msg)) => {
            return Err(PrepError::Parse(format!(
                "Score file '{}' line {line_number}: {msg}",
                path.display()
            )));
        }
        Err(error) => return Err(error),
    };
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
        let weight = match parse_weight(weight_str) {
            Ok(weight) => weight,
            Err(problem) => {
                // The stream drops the line's buffered weights with the error.
                records.truncate(start);
                return Err(unusable_weight_error(
                    weight_str,
                    line_number,
                    path,
                    &problem,
                ));
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
    use super::super::{KWayMergeIterator, ScoreRows};
    use super::*;

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

    fn describe_finish(outcome: Result<RejectedScoreRows, PrepError>) -> String {
        match outcome {
            Ok(mut rows) => {
                rows.examples.sort();
                rows.examples.truncate(RejectedScoreRows::EXAMPLES);
                format!(
                    "rejected {} other alleles, examples {:?}",
                    rows.unknown_other_allele, rows.examples
                )
            }
            Err(e) => format!("error {e}"),
        }
    }

    /// Takes up to `take` items, each with the malformed-line count before it, then
    /// finishes, so any point where the merge-join stops can be compared.
    fn consume(mut rows: ScoreRows, take: usize) -> Vec<(usize, String)> {
        let mut seen = Vec::new();
        for _ in 0..take {
            let malformed = rows.malformed_lines();
            match rows.next() {
                Some(item) => seen.push((malformed, describe(item))),
                None => {
                    seen.push((malformed, "end".to_string()));
                    break;
                }
            }
        }
        let malformed = rows.malformed_lines();
        seen.push((malformed, describe_finish(rows.finish())));
        seen
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
            vec![b"variant_id\teffect_allele\tother_allele\tA\tB\n1:1\tA\tG\t1.0000000000000002\t1e-200\n1:2\tA\tG\t1e200\t-0\n1:3\tA\tG\t2\tNaN\n"],
            vec![b"variant_id\teffect_allele\tother_allele\tA\n1:1\tA\tG\t1\n1:2\tA\tG\t1e999\n"],
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
            // Rows skipped on their own: unknown other alleles, and rows on
            // unkeyable contigs at the start, middle and end of a file.
            vec![
                b"variant_id\teffect_allele\tother_allele\tA\n\
                  1:1\tA\tG\t1\n\
                  1:2\tA\tN\t1\n\
                  chrUn_z:6\tA\tG\t1\n\
                  1:7\tA\tG\t2\n\
                  chrUn_z:8\tA\tG\t1\n",
                b"#note\nvariant_id\teffect_allele\tother_allele\tB\tC\n\
                  chrUn_y:1\tA\tG\t1\t1\n\
                  1:3\tA\tG\t1\t\n\
                  1:8\tA\tN\t1\t1\n\
                  2:1\tA\tG\t\t4\n",
            ],
            // A weight that is not a finite number fails the run wherever it sits.
            vec![
                b"variant_id\teffect_allele\tother_allele\tA\tB\n1:1\tA\tG\t1\t2\n1:4\tA\tG\t3\tnan\n1:5\tA\tG\t1\t1\n",
                b"variant_id\teffect_allele\tother_allele\tC\n1:2\tA\tG\t2\n1:9\tA\tG\t1\n",
            ],
            // An invalid position fails the run wherever it sits.
            vec![
                b"variant_id\teffect_allele\tother_allele\tA\n1:1\tA\tG\t1\n1:x\tA\tG\t1\n1:9\tA\tG\t1\n",
                b"variant_id\teffect_allele\tother_allele\tB\n1:2\tA\tG\t2\n",
            ],
        ];
        for (case, contents) in cases.iter().enumerate() {
            let paths: Vec<PathBuf> = contents
                .iter()
                .enumerate()
                .map(|(i, bytes)| write(dir.path(), &format!("case{case}_{i}.tsv"), bytes))
                .collect();
            let streamed =
                || ScoreRows::Streamed(KWayMergeIterator::new(&paths, &columns, None).unwrap());
            let items = consume(streamed(), usize::MAX).len();
            for take in 0..items {
                let expected = consume(streamed(), take);
                for block_bytes in [1, 9, 64, 1 << 20] {
                    let parsed =
                        parse_score_files_within(&paths, &columns, u64::MAX, block_bytes).unwrap();
                    let actual = consume(ScoreRows::Parsed(parsed), take);
                    assert_eq!(
                        actual, expected,
                        "case {case}, {take} items taken, block size {block_bytes}"
                    );
                }
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
            b"variant_id\teffect_allele\tother_allele\tA\n1:x\tA\tG\t1\n",
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
