//! Native VCF and BCF scoring over batches of inflated bytes.
//!
//! A batch is read, inflated block by block on the rayon pool straight into one
//! buffer, and cut at record boundaries into parts. Each part's records are
//! filtered and decoded on the pool, and the decoded records are then taken in
//! input order, so every sum takes the same terms in the same sequence as a
//! one-record-at-a-time scan, and the first error raised is that scan's.

use super::{
    BGZF_HEADER_LEN, BGZF_MAX_DATA_LEN, BGZF_TRAILER_LEN, DecodeContext, DecodedRecord,
    NativeVcfScoreResult, RecordAccumulator, ScoreRules, decode_scored_bcf_record, decode_vcf_line,
    inflate_bgzf_block, is_bgzf_header, is_skippable_record, native_result,
    resolve_keep_indices,
};
use crate::score::types::parse_chromosome_label;
use crate::shared::files::{VariantCompression, VariantFormat, VariantSource};
use flate2::read::MultiGzDecoder;
use libdeflater::Decompressor;
use memchr::{memchr, memrchr};
use noodles_bcf::io::Reader as BcfReader;
use noodles_vcf::io::Reader as VcfReader;
use rayon::prelude::*;
use std::error::Error;
use std::io::{self, Cursor, Read};
use std::ops::Range;
use std::path::Path;
use std::simd::prelude::*;

type BoxError = Box<dyn Error + Send + Sync>;

/// Blocks inflated per rayon worker in one batch.
const BGZF_BLOCKS_PER_WORKER: usize = 64;
/// Bytes read per rayon worker in one batch of plain text or gzip members.
const TEXT_BYTES_PER_WORKER: usize = BGZF_BLOCKS_PER_WORKER * BGZF_MAX_DATA_LEN;
/// Parts a batch is cut into per rayon worker, so one slow part leaves few workers idle.
const PARTS_PER_WORKER: usize = 4;
/// Compressed bytes requested from the source per read.
const SOURCE_READ_LEN: usize = 1 << 20;

/// Scores the records of `source` for the kept samples of its header.
pub(super) fn score_source(
    source: VariantSource,
    input_path: &Path,
    keep: Option<&Path>,
    score_names: Vec<String>,
    rules_by_key: &ScoreRules,
) -> Result<NativeVcfScoreResult, BoxError> {
    let format = source.format();
    let threads = rayon::current_num_threads().max(1);
    let mut stream = InflatedStream::new(source);
    let mut bytes = Bytes::default();
    let mut at_end = false;
    let header_len = loop {
        let len = match format {
            VariantFormat::Vcf => vcf_header_len(bytes.filled(), at_end),
            VariantFormat::Bcf => bcf_header_len(bytes.filled(), at_end),
        };
        if let Some(len) = len {
            break len;
        }
        at_end = !stream.fill(&mut bytes, threads)?;
    };
    let header_bytes = &bytes.filled()[..header_len];
    let header = match format {
        VariantFormat::Vcf => {
            crate::variant_header::read_vcf_header(&mut VcfReader::new(header_bytes))?
        }
        VariantFormat::Bcf => {
            crate::variant_header::read_bcf_header(&mut BcfReader::from(header_bytes))?
        }
    }
    .warned(input_path);
    bytes.consume(header_len);
    let (layout, format_name) = match format {
        VariantFormat::Vcf => (Layout::Vcf, "VCF"),
        VariantFormat::Bcf => (Layout::Bcf(&header), "BCF"),
    };

    let all_samples: Vec<String> = header.sample_names().iter().cloned().collect();
    if all_samples.is_empty() {
        return Err(format!("{format_name} contains no samples.").into());
    }
    let kept_indices = resolve_keep_indices(keep, &all_samples)?;
    let person_iids: Vec<String> = kept_indices
        .iter()
        .map(|&idx| all_samples[idx].clone())
        .collect();

    let context = DecodeContext {
        rules_by_key,
        kept_indices: &kept_indices,
        score_names: &score_names,
    };
    let mut accumulator = RecordAccumulator::new(rules_by_key, &score_names, person_iids.len());
    loop {
        let mut source_error = None;
        if !at_end {
            match stream.fill(&mut bytes, threads) {
                Ok(more) => at_end = !more,
                Err(err) => source_error = Some(err),
            }
        }
        let filled = bytes.filled();
        // A read that failed leaves its last record unread, as noodles leaves it.
        let (complete, stops) = complete_records(layout, filled, at_end);
        let records = &filled[..complete];
        let decoded: Vec<Vec<DecodedRecord>> = parts(layout, records, threads * PARTS_PER_WORKER)
            .into_par_iter()
            .map(|range| decode_part(layout, &records[range], &context))
            .collect();
        for mut record in decoded.into_iter().flatten() {
            accumulator.take(&mut record)?;
        }
        accumulator.apply();

        if let Some(err) = source_error {
            return Err(err.into());
        }
        if at_end || stops {
            if matches!(layout, Layout::Bcf(_)) && !stops && complete < filled.len() {
                // A final record cut short fails as noodles fails to read it.
                BcfReader::from(&filled[complete..])
                    .read_record(&mut noodles_bcf::Record::default())?;
            }
            let totals = accumulator.finish()?;
            return native_result(totals, person_iids, score_names, input_path);
        }
        bytes.consume(complete);
    }
}

/// How records sit in inflated bytes.
#[derive(Clone, Copy)]
enum Layout<'h> {
    /// Newline-terminated text lines.
    Vcf,
    /// Length-prefixed binary records under the header whose string maps they index.
    Bcf(&'h noodles_vcf::Header),
}

/// How many leading bytes of `bytes` hold complete records, and whether the
/// stream ends after them. At the end of input the last VCF line needs no
/// newline. A BCF record whose site length is zero ends the stream, as noodles
/// reads it.
fn complete_records(layout: Layout<'_>, bytes: &[u8], at_end: bool) -> (usize, bool) {
    match layout {
        Layout::Vcf if at_end => (bytes.len(), false),
        Layout::Vcf => (memrchr(b'\n', bytes).map_or(0, |last| last + 1), false),
        Layout::Bcf(_) => {
            let mut offset = 0usize;
            loop {
                let Some(site_len) = read_u32(bytes, offset) else {
                    return (offset, false);
                };
                if site_len == 0 {
                    return (offset, true);
                }
                let Some(samples_len) = read_u32(bytes, offset + 4) else {
                    return (offset, false);
                };
                let end = offset + 8 + site_len + samples_len;
                if end > bytes.len() {
                    return (offset, false);
                }
                offset = end;
            }
        }
    }
}

/// The little-endian `u32` at `offset`, when four bytes are there.
fn read_u32(bytes: &[u8], offset: usize) -> Option<usize> {
    let field: [u8; 4] = bytes.get(offset..offset + 4)?.try_into().ok()?;
    usize::try_from(u32::from_le_bytes(field)).ok()
}

/// The length of the complete BCF record at `offset`.
fn bcf_record_len(bytes: &[u8], offset: usize) -> usize {
    let site_len = read_u32(bytes, offset).expect("a complete record has a site length");
    let samples_len = read_u32(bytes, offset + 4).expect("a complete record has a samples length");
    8 + site_len + samples_len
}

/// Cuts complete records into about `count` parts of similar length, at record boundaries.
fn parts(layout: Layout<'_>, bytes: &[u8], count: usize) -> Vec<Range<usize>> {
    let target = bytes.len().div_ceil(count.max(1)).max(1);
    let mut parts = Vec::with_capacity(count);
    let mut start = 0usize;
    while start < bytes.len() {
        let end = match layout {
            Layout::Vcf => {
                let wanted = start + target;
                if wanted >= bytes.len() {
                    bytes.len()
                } else {
                    memchr(b'\n', &bytes[wanted - 1..]).map_or(bytes.len(), |offset| wanted + offset)
                }
            }
            Layout::Bcf(_) => {
                let mut end = start;
                while end < bytes.len() && end - start < target {
                    end += bcf_record_len(bytes, end);
                }
                end
            }
        };
        parts.push(start..end);
        start = end;
    }
    parts
}

/// Decodes one part's records that the scorer acts on: every record at a scored
/// position, up to and including the first record that raises an error.
fn decode_part(layout: Layout<'_>, bytes: &[u8], context: &DecodeContext<'_>) -> Vec<DecodedRecord> {
    let mut decoded = Vec::new();
    match layout {
        Layout::Vcf => for_each_line(bytes, |line, ascii| {
            let text = line.strip_suffix(b"\n").unwrap_or(line);
            if is_skippable_record(text, ascii, context.rules_by_key) {
                return true;
            }
            let mut record = DecodedRecord::default();
            if let Err(err) = decode_vcf_line(line, context, &mut record) {
                record.error = Some(err);
            }
            !keep_decoded(&mut decoded, record)
        }),
        Layout::Bcf(header) => {
            let mut reader = BcfRecordReader::default();
            let mut start = 0usize;
            while start < bytes.len() {
                let end = start + bcf_record_len(bytes, start);
                let mut record = DecodedRecord::default();
                if let Err(err) = reader.decode(&bytes[start..end], header, context, &mut record) {
                    record.error = Some(err);
                }
                start = end;
                if keep_decoded(&mut decoded, record) {
                    break;
                }
            }
        }
    }
    decoded
}

/// Keeps `record` when the scorer acts on it, returning whether it failed, which
/// ends its part: nothing after a failing record is scored.
fn keep_decoded(decoded: &mut Vec<DecodedRecord>, record: DecodedRecord) -> bool {
    let failed = record.error.is_some();
    if failed || record.key.is_some() {
        decoded.push(record);
    }
    failed
}

/// Visits every line of `bytes`, with its newline when it has one, and whether
/// the line holds only ASCII bytes, until `visit` returns `false`. One pass over
/// sixty-four bytes at a time finds both the newlines and the bytes above 0x7f.
fn for_each_line(bytes: &[u8], mut visit: impl FnMut(&[u8], bool) -> bool) {
    let (chunks, rest) = bytes.as_chunks::<64>();
    let mut line_start = 0usize;
    // Whether the bytes of the open line scanned so far are ASCII.
    let mut ascii = true;
    for (index, chunk) in chunks.iter().enumerate() {
        let base = index * 64;
        let simd = u8x64::from_array(*chunk);
        let high = simd.simd_ge(u8x64::splat(0x80)).to_bitmask();
        let mut newlines = simd.simd_eq(u8x64::splat(b'\n')).to_bitmask();
        while newlines != 0 {
            let offset = newlines.trailing_zeros() as usize;
            let from = line_start.saturating_sub(base);
            let line_bits = (u64::MAX << from) & (u64::MAX >> (63 - offset));
            let end = base + offset + 1;
            if !visit(&bytes[line_start..end], ascii && (high & line_bits) == 0) {
                return;
            }
            line_start = end;
            ascii = true;
            newlines &= newlines - 1;
        }
        let from = line_start.saturating_sub(base);
        if from < 64 {
            ascii &= (high >> from) == 0;
        }
    }
    let base = chunks.len() * 64;
    for (offset, &byte) in rest.iter().enumerate() {
        ascii &= byte.is_ascii();
        if byte == b'\n' {
            let end = base + offset + 1;
            if !visit(&bytes[line_start..end], ascii) {
                return;
            }
            line_start = end;
            ascii = true;
        }
    }
    if line_start < bytes.len() {
        visit(&bytes[line_start..], ascii);
    }
}

/// Reads BCF records as noodles reads them, copying a record's samples only when
/// the record is scored.
#[derive(Default)]
struct BcfRecordReader {
    /// A record read with its samples left out, which noodles checks as it
    /// checks any record, so a skipped record raises the errors noodles raises.
    site: noodles_bcf::Record,
    site_bytes: Vec<u8>,
    record: noodles_bcf::Record,
}

impl BcfRecordReader {
    /// Decodes the complete record `bytes` into `decoded`, as
    /// `decode_scored_bcf_record` decodes a record noodles read.
    fn decode(
        &mut self,
        bytes: &[u8],
        header: &noodles_vcf::Header,
        context: &DecodeContext<'_>,
        decoded: &mut DecodedRecord,
    ) -> Result<(), BoxError> {
        let site_len = read_u32(bytes, 0).expect("a complete record has a site length");
        self.site_bytes.clear();
        self.site_bytes.extend_from_slice(&bytes[..4]);
        self.site_bytes.extend_from_slice(&0u32.to_le_bytes());
        self.site_bytes.extend_from_slice(&bytes[8..8 + site_len]);
        BcfReader::from(&self.site_bytes[..]).read_record(&mut self.site)?;

        let chromosome = self.site.reference_sequence_name(header.string_maps())?;
        let Ok(chr) = parse_chromosome_label(chromosome) else {
            return Ok(());
        };
        let Some(start) = self.site.variant_start() else {
            return Ok(());
        };
        if !context.rules_by_key.contains_key(&(chr, start?.get() as u32)) {
            return Ok(());
        }
        BcfReader::from(bytes).read_record(&mut self.record)?;
        decode_scored_bcf_record(&self.record, header, context, decoded)
    }
}

/// Where the VCF header in `bytes` ends: after the leading lines that start with
/// '#', where noodles stops reading a header. `None` while more input could
/// still extend it.
fn vcf_header_len(bytes: &[u8], at_end: bool) -> Option<usize> {
    let mut start = 0usize;
    while start < bytes.len() {
        if bytes[start] != b'#' {
            return Some(start);
        }
        match memchr(b'\n', &bytes[start..]) {
            Some(offset) => start += offset + 1,
            None => break,
        }
    }
    at_end.then_some(bytes.len())
}

/// Where the BCF header in `bytes` ends: after the magic number, the version and
/// the length-prefixed text. `None` while more input could still complete it.
fn bcf_header_len(bytes: &[u8], at_end: bool) -> Option<usize> {
    read_u32(bytes, 5)
        .map(|text_len| 9 + text_len)
        .filter(|&len| len <= bytes.len())
        .or(at_end.then_some(bytes.len()))
}

/// A byte buffer whose bytes past `len` stay allocated and initialised, so
/// refilling it never clears memory it already holds.
#[derive(Default)]
struct Bytes {
    buf: Vec<u8>,
    len: usize,
}

impl Bytes {
    fn filled(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// `extra` writable bytes after the filled ones.
    fn spare(&mut self, extra: usize) -> &mut [u8] {
        let needed = self.len + extra;
        if self.buf.len() < needed {
            self.buf.resize(needed, 0);
        }
        &mut self.buf[self.len..needed]
    }

    fn commit(&mut self, amt: usize) {
        self.len += amt;
    }

    /// Drops the first `amt` filled bytes.
    fn consume(&mut self, amt: usize) {
        self.buf.copy_within(amt..self.len, 0);
        self.len -= amt;
    }
}

/// A variant stream as batches of inflated bytes.
///
/// BGZF blocks are inflated in parallel, each into its own place in the batch,
/// and checked against their recorded lengths and CRC32s. From the first bytes
/// that do not form a well-formed BGZF block (plain gzip members, truncated or
/// corrupt blocks, trailing garbage), the rest of the stream is read through
/// `MultiGzDecoder`. Plain input is read as it is.
struct InflatedStream {
    input: Input,
    /// Compressed bytes read from a BGZF source; those before `start` are inflated.
    compressed: Bytes,
    start: usize,
    source_done: bool,
    frames: Vec<Frame>,
}

enum Input {
    Bgzf(VariantSource),
    Text(Box<dyn Read + Send>),
    Done,
}

/// One canonical BGZF block in `InflatedStream::compressed`.
struct Frame {
    range: Range<usize>,
    data_len: usize,
}

impl InflatedStream {
    fn new(source: VariantSource) -> Self {
        let input = match source.compression() {
            VariantCompression::Bgzf => Input::Bgzf(source),
            VariantCompression::Plain => Input::Text(Box::new(source)),
        };
        Self {
            input,
            compressed: Bytes::default(),
            start: 0,
            source_done: false,
            frames: Vec::new(),
        }
    }

    /// Appends the next batch of inflated bytes to `out`, returning `false` once
    /// the stream is exhausted.
    fn fill(&mut self, out: &mut Bytes, threads: usize) -> io::Result<bool> {
        if let Input::Text(reader) = &mut self.input {
            return read_text(reader.as_mut(), out, threads * TEXT_BYTES_PER_WORKER);
        }
        if matches!(self.input, Input::Done) {
            return Ok(false);
        }
        self.fill_bgzf(out, threads * BGZF_BLOCKS_PER_WORKER)
    }

    fn fill_bgzf(&mut self, out: &mut Bytes, batch_blocks: usize) -> io::Result<bool> {
        self.compressed.consume(std::mem::take(&mut self.start));
        self.frames.clear();
        let mut offset = 0usize;
        let mut total = 0usize;
        let mut at_end = false;
        // Where the stream stops being canonical BGZF, when it does in this batch.
        let mut irregular = None;
        while self.frames.len() < batch_blocks {
            let available = self.available(offset + BGZF_HEADER_LEN)? - offset;
            if available == 0 {
                at_end = true;
                break;
            }
            let header = &self.compressed.filled()[offset..];
            if available < BGZF_HEADER_LEN || !is_bgzf_header(header) {
                irregular = Some(offset);
                break;
            }
            let block_len = usize::from(u16::from_le_bytes([header[16], header[17]])) + 1;
            if block_len < BGZF_HEADER_LEN + BGZF_TRAILER_LEN
                || self.available(offset + block_len)? < offset + block_len
            {
                irregular = Some(offset);
                break;
            }
            let block = &self.compressed.filled()[offset..offset + block_len];
            let data_len = read_u32(block, block_len - 4).expect("a block ends with its length");
            if data_len > BGZF_MAX_DATA_LEN {
                irregular = Some(offset);
                break;
            }
            self.frames.push(Frame {
                range: offset..offset + block_len,
                data_len,
            });
            total += data_len;
            offset += block_len;
        }

        let compressed = self.compressed.filled();
        let mut outputs = Vec::with_capacity(self.frames.len());
        let mut rest = out.spare(total);
        for frame in &self.frames {
            let (output, tail) = std::mem::take(&mut rest).split_at_mut(frame.data_len);
            outputs.push(output);
            rest = tail;
        }
        let inflated: Vec<bool> = self
            .frames
            .par_iter()
            .zip(outputs)
            .map_init(Decompressor::new, |decompressor, (frame, output)| {
                inflate_bgzf_block(&compressed[frame.range.clone()], decompressor, output).is_ok()
            })
            .collect();
        let failed = inflated.iter().position(|&ok| !ok);
        let inflated_len: usize = self.frames[..failed.unwrap_or(self.frames.len())]
            .iter()
            .map(|frame| frame.data_len)
            .sum();
        out.commit(inflated_len);

        if let Some(index) = failed {
            irregular = Some(self.frames[index].range.start);
        }
        if let Some(start) = irregular {
            self.fall_back(start);
            return Ok(true);
        }
        self.start = offset;
        if at_end {
            self.input = Input::Done;
        }
        Ok(!self.frames.is_empty() || !at_end)
    }

    /// Reads from the source until `compressed` holds `len` bytes or the source
    /// is exhausted, returning how many bytes it holds.
    fn available(&mut self, len: usize) -> io::Result<usize> {
        let Input::Bgzf(source) = &mut self.input else {
            return Ok(self.compressed.len);
        };
        while self.compressed.len < len && !self.source_done {
            let spare = self
                .compressed
                .spare(SOURCE_READ_LEN.max(len - self.compressed.len));
            match source.read(spare) {
                Ok(0) => self.source_done = true,
                Ok(amt) => self.compressed.commit(amt),
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        }
        Ok(self.compressed.len)
    }

    /// Reads the stream from compressed offset `start` on through `MultiGzDecoder`.
    fn fall_back(&mut self, start: usize) {
        let Input::Bgzf(source) = std::mem::replace(&mut self.input, Input::Done) else {
            unreachable!("only a BGZF stream falls back to gzip decoding");
        };
        let rest = self.compressed.filled()[start..].to_vec();
        self.compressed = Bytes::default();
        self.start = 0;
        self.input = Input::Text(Box::new(MultiGzDecoder::new(
            Cursor::new(rest).chain(source),
        )));
    }
}

/// Reads up to `len` bytes from `reader` into `out`, returning `false` when the
/// reader was already exhausted. What a failing read leaves behind it is kept.
fn read_text(reader: &mut dyn Read, out: &mut Bytes, len: usize) -> io::Result<bool> {
    let spare = out.spare(len);
    let mut filled = 0usize;
    let result = loop {
        if filled == len {
            break Ok(());
        }
        match reader.read(&mut spare[filled..]) {
            Ok(0) => break Ok(()),
            Ok(amt) => filled += amt,
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => break Err(err),
        }
    };
    out.commit(filled);
    result.map(|()| filled > 0)
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::{BoxError, for_each_line};
    use flate2::Crc;
    use std::io::Write as _;

    /// SplitMix64 draws.
    struct Draws(u64);

    impl Draws {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) % n as u64) as usize
        }

        fn pick(&mut self, items: &[&'static str]) -> &'static str {
            items[self.below(items.len())]
        }
    }

    /// Lines `split_vcf_line` splits read as noodles reads them: the chromosome,
    /// position, REF, ALT and samples fields agree over random lines holding
    /// carriage returns, missing and empty fields, telomeric and malformed
    /// positions, missing FORMAT, trailing tabs, invalid UTF-8 and cut-short lines.
    #[test]
    fn vcf_lines_split_as_noodles_reads_them() {
        let mut draws = Draws(0x2362);
        let (mut split, mut left_to_noodles) = (0usize, 0usize);
        for _ in 0..4000 {
            let fields = [
                draws.pick(&["22", "chr1", "X", "HLA-A*01:01", "22\r", "", "1"]),
                draws.pick(&["100", "0", "00", "+100", "abc", "4294967396", "100\r", "", "7"]),
                draws.pick(&[".", "rs1", ""]),
                draws.pick(&["A", "AC", "", "N"]),
                draws.pick(&["G", "G,T", ".", "", "<DEL>", "G,", ",G"]),
                draws.pick(&[".", "50"]),
                draws.pick(&["PASS", ".", "q10\r"]),
                draws.pick(&[".", "DP=5", "", "AF=0.5\r"]),
            ];
            let mut line = fields.join("\t").into_bytes();
            if draws.below(6) > 0 {
                line.push(b'\t');
                line.extend_from_slice(
                    draws
                        .pick(&["GT", "GT:DS", ".", "DS", "GT:", ":GT", "", "GT\r"])
                        .as_bytes(),
                );
                for _ in 0..draws.below(4) {
                    line.push(b'\t');
                    line.extend_from_slice(
                        draws
                            .pick(&["0/1", "1|1", "./.", ".", "", "0", "0/1:0.5", "0/1\r"])
                            .as_bytes(),
                    );
                }
            }
            let ending = draws.below(8);
            if ending == 0 {
                line.push(b'\t');
            } else if ending == 1 {
                line.push(b'\r');
            } else if ending == 2 {
                line.push(0xff);
            } else if ending == 3 {
                let cut = draws.below(line.len() + 1);
                line.truncate(cut);
            }

            let mut with_newline = line.clone();
            with_newline.push(b'\n');
            let mut record = noodles_vcf::Record::default();
            let read = VcfReader::new(&with_newline[..]).read_record(&mut record);
            let Some(fields) = split_vcf_line(&line) else {
                left_to_noodles += 1;
                continue;
            };
            split += 1;
            let context = String::from_utf8_lossy(&line).into_owned();
            read.unwrap_or_else(|err| panic!("{context:?}: {err}"));
            assert_eq!(
                fields.chromosome,
                record.reference_sequence_name(),
                "{context:?}"
            );
            assert_eq!(
                fields.variant_start.map(|start| start.ok()),
                record
                    .variant_start()
                    .map(|start| start.ok().map(|position| position.get())),
                "{context:?}"
            );
            assert_eq!(fields.reference_bases, record.reference_bases(), "{context:?}");
            assert_eq!(
                fields.alternate_bases,
                record.alternate_bases().as_ref(),
                "{context:?}"
            );
            assert_eq!(fields.samples, record.samples().as_ref(), "{context:?}");
        }
        assert!(
            split >= 500 && left_to_noodles >= 500,
            "{split} lines split, {left_to_noodles} left to noodles"
        );
    }

    /// Every person a dosage route visits, as exact doses.
    type Visits = Vec<Option<(Dose, Option<Dose>)>>;

    /// The hard-call decoder visits every kept person as the dosage route does,
    /// or fails with its error: phased, unphased, haploid, polyploid, missing and
    /// multi-digit calls, GT with other FORMAT fields, empty and '.' columns,
    /// trailing tabs, records with fewer columns than people, and keep subsets.
    #[test]
    fn hard_calls_decode_as_the_dosage_route_visits() {
        let mut draws = Draws(0x6a7e);
        let kept_sets: [&[usize]; 4] = [&[0, 1, 2, 3, 4], &[0, 2, 4], &[3], &[]];
        let (mut calls, mut other_layouts, mut failures) = (0usize, 0usize, 0usize);
        for _ in 0..6000 {
            let mut samples = String::from(draws.pick(&[
                "GT", "GT", "GT", "GT:AD", "GT:DS", "AD:GT", "DS", "GT:GP", ".",
            ]));
            for _ in 0..draws.below(7) {
                samples.push('\t');
                samples.push_str(draws.pick(&[
                    "0/1",
                    "1|1",
                    "0|0",
                    "1/0",
                    "2|1",
                    "./.",
                    ".|1",
                    "1|.",
                    ".",
                    "",
                    "0",
                    "2",
                    "1/2/1",
                    "10/2",
                    "0/1:5,3",
                    "0/x",
                    "0/1/1/1/1/1/1/1/1/1/1/1/1/1/1/1",
                ]));
            }
            if draws.below(10) == 0 {
                samples.push('\t');
            }
            if draws.below(20) == 0 {
                samples.clear();
            }
            let kept = kept_sets[draws.below(kept_sets.len())];
            let alt_index = 1 + draws.below(3);

            let mut codes = Vec::new();
            let fast = vcf_gt_calls(&samples, alt_index, 3, kept, &mut codes);
            let mut visits: Visits = Vec::new();
            let text = for_each_vcf_dosage_best(&samples, alt_index, 3, kept, |_, dosage| {
                visits.push(dosage.map(|d| (d.alt_dosage, d.ref_dosage)));
                Ok(())
            });
            match fast {
                Ok(true) => {
                    text.unwrap_or_else(|err| panic!("{samples:?}: {err}"));
                    let decoded: Visits = codes
                        .iter()
                        .map(|&code| {
                            (code != MISSING_CALL).then(|| {
                                (Dose::copies(code & 0x0f), Some(Dose::copies(code >> 4)))
                            })
                        })
                        .collect();
                    assert_eq!(decoded, visits, "{samples:?} kept {kept:?} ALT {alt_index}");
                    calls += 1;
                }
                Ok(false) => other_layouts += 1,
                Err(err) => {
                    let expected = text.expect_err("the dosage route fails too");
                    assert_eq!(err.to_string(), expected.to_string(), "{samples:?}");
                    failures += 1;
                }
            }
        }
        assert!(
            calls >= 1000 && other_layouts >= 500 && failures >= 100,
            "{calls} decoded as hard calls, {other_layouts} other layouts, {failures} failures"
        );
    }

    /// Wide rows of hard calls decode as the dosage route visits them: rows of
    /// seventy columns, mostly diploid calls of single-digit alleles, some with
    /// other shapes scattered among them and some cut short, under keep subsets
    /// that hold runs of sixteen adjacent people and subsets that skip some.
    #[test]
    fn wide_rows_of_hard_calls_decode_as_the_dosage_route_visits() {
        let mut draws = Draws(0x16c0);
        let people = 70usize;
        let all: Vec<usize> = (0..people).collect();
        let gapped: Vec<usize> = (0..people).filter(|index| index % 23 != 7).collect();
        let later: Vec<usize> = (40..people).collect();
        let kept_sets: [&[usize]; 3] = [&all, &gapped, &later];
        for _ in 0..3000 {
            let mut samples = String::from("GT");
            let irregular_one_in = [0, 40, 8][draws.below(3)];
            for _ in 0..people - draws.below(3) {
                samples.push('\t');
                samples.push_str(if irregular_one_in > 0 && draws.below(irregular_one_in) == 0 {
                    draws.pick(&["./.", "1", "0/1/1", "10/2", ".", "", "1|.", ".|."])
                } else {
                    draws.pick(&["0|0", "0|1", "1|0", "1|1", "0/2", "2/1", "3|0", "2|2"])
                });
            }
            let kept = kept_sets[draws.below(kept_sets.len())];
            let alt_index = 1 + draws.below(3);

            let mut codes = Vec::new();
            let fast = vcf_gt_calls(&samples, alt_index, 3, kept, &mut codes)
                .unwrap_or_else(|err| panic!("{samples:?}: {err}"));
            assert!(fast, "{samples:?} decodes as hard calls");
            let mut visits: Visits = Vec::new();
            for_each_vcf_dosage_best(&samples, alt_index, 3, kept, |_, dosage| {
                visits.push(dosage.map(|d| (d.alt_dosage, d.ref_dosage)));
                Ok(())
            })
            .unwrap_or_else(|err| panic!("{samples:?}: {err}"));
            let decoded: Visits = codes
                .iter()
                .map(|&code| {
                    (code != MISSING_CALL).then(|| {
                        (Dose::copies(code & 0x0f), Some(Dose::copies(code >> 4)))
                    })
                })
                .collect();
            assert_eq!(decoded, visits, "{samples:?} kept {kept:?} ALT {alt_index}");
        }
    }

    /// The GT:DS decoder visits every kept person as the dosage route does, or fails
    /// with its error: phased, unphased, haploid and missing calls, plain, exponent,
    /// negative, missing and too-large dosages, extra FORMAT values, empty and '.'
    /// columns, records with fewer columns than people, and keep subsets.
    #[test]
    fn gt_ds_dosages_decode_as_the_dosage_route_visits() {
        let mut draws = Draws(0x6d5);
        let kept_sets: [&[usize]; 4] = [&[0, 1, 2, 3, 4], &[0, 2, 4], &[3], &[]];
        let (mut decoded, mut other_layouts, mut failures) = (0usize, 0usize, 0usize);
        for _ in 0..6000 {
            let mut samples = String::from(draws.pick(&["GT:DS", "GT:DS", "GT:DS", "DS:GT", "GT:DS:GP"]));
            let alt_count = [1, 1, 1, 2][draws.below(4)];
            for _ in 0..draws.below(7) {
                samples.push('\t');
                samples.push_str(draws.pick(&["0/1", "1|1", "0|0", "./.", ".|1", "1", "0/1/1", "10/1", "", "."]));
                samples.push(':');
                samples.push_str(draws.pick(&[
                    "0.95", "0", "2", "2.000001", "2.1", "1e-3", "-0.5", ".", "", "0.5:7", "1,0.5", "0.123456789012",
                ]));
                if draws.below(10) == 0 {
                    samples.clear();
                    samples.push_str("GT:DS\t.");
                }
            }
            if draws.below(10) == 0 {
                samples.push('\t');
            }
            let kept = kept_sets[draws.below(kept_sets.len())];
            let alt_index = 1 + draws.below(alt_count);
            let ref_effect_error = || asks_for_ref(alt_index).then(|| String::from("REF-effect rule"));

            let mut fast_dosages = Doses::default();
            let fast = vcf_gt_ds_dosages(&samples, alt_index, alt_count, kept, &ref_effect_error, &mut fast_dosages);
            let mut visits: Vec<[Dose; 2]> = Vec::new();
            let text = for_each_vcf_dosage_best(&samples, alt_index, alt_count, kept, |_, dosage| {
                let pair = dosage_pair(dosage, &ref_effect_error)?;
                visits.push(pair);
                Ok(())
            });
            match fast {
                Ok(true) => {
                    text.unwrap_or_else(|err| panic!("{samples:?}: {err}"));
                    assert_eq!(fast_dosages.pairs(), visits, "{samples:?} kept {kept:?}");
                    decoded += 1;
                }
                Ok(false) => other_layouts += 1,
                Err(err) => {
                    let expected = text.expect_err("the dosage route fails too");
                    assert_eq!(err.to_string(), expected.to_string(), "{samples:?}");
                    failures += 1;
                }
            }
        }
        assert!(
            decoded >= 1000 && other_layouts >= 1000 && failures >= 100,
            "{decoded} decoded, {other_layouts} other layouts, {failures} failures"
        );
    }

    /// Whether the GT:DS fuzz asks for a REF dosage: for the first ALT allele only,
    /// so both routes meet dosages with and without a REF-effect rule.
    fn asks_for_ref(alt_index: usize) -> bool {
        alt_index == 1
    }

    /// BGZF blocks of at most `block_len` uncompressed bytes, then the empty EOF block.
    fn bgzf_bytes(text: &[u8], block_len: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in text.chunks(block_len).chain(std::iter::once(&[][..])) {
            let mut encoder =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(chunk).expect("deflate");
            let compressed = encoder.finish().expect("finish deflate");
            let mut crc = Crc::new();
            crc.update(chunk);
            let block_len = BGZF_HEADER_LEN + compressed.len() + BGZF_TRAILER_LEN;
            bytes.extend_from_slice(&[
                0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, b'B', b'C',
                0x02, 0x00,
            ]);
            bytes.extend_from_slice(&u16::try_from(block_len - 1).expect("BSIZE").to_le_bytes());
            bytes.extend_from_slice(&compressed);
            bytes.extend_from_slice(&crc.sum().to_le_bytes());
            bytes.extend_from_slice(&u32::try_from(chunk.len()).expect("ISIZE").to_le_bytes());
        }
        bytes
    }

    /// A random cohort for the batched reader: hard calls of several ploidies,
    /// GT:DS and DS layouts, multiallelic records and records split at one
    /// position, an unsupported contig, CRLF lines, and ALT- and REF-effect rows.
    fn batch_cohort(draws: &mut Draws, samples: usize, records: usize) -> (String, String) {
        let mut vcf = String::from(
            "##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Dosage\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT",
        );
        for sample in 0..samples {
            vcf.push_str(&format!("\ts{sample}"));
        }
        vcf.push('\n');
        let mut score = String::from("variant_id\teffect_allele\tother_allele\tScoreA\tScoreB\n");
        let mut position = 1000usize;
        let mut layout = 0usize;
        let mut alt = "G";
        for record in 0..records {
            // Every tenth record repeats the position ahead of it in the same layout, so a
            // REF-effect row there never meets a DS-only record, which has no REF dosage.
            if record % 10 != 3 {
                position += 1 + draws.below(3);
                layout = draws.below(6);
            }
            let chromosome = if draws.below(30) == 0 {
                "chrUn_KI270302v1"
            } else {
                "22"
            };
            // A repeated position is a split site: its record carries none of the ALTs of the
            // record before it, as two records of one scored allele pair are refused.
            alt = if record % 10 == 3 {
                match alt {
                    "G" => "T",
                    "T" => "G",
                    _ => "C",
                }
            } else {
                draws.pick(&["G", "G", "G,T", "T"])
            };
            let alt_count = alt.split(',').count();
            let format = match layout {
                0 => "GT:DS",
                1 => "DS",
                _ => "GT",
            };
            vcf.push_str(&format!("{chromosome}\t{position}\t.\tA\t{alt}\t.\tPASS\t.\t{format}"));
            let diploid: &[&'static str] = if alt_count == 2 {
                &["0|0", "0|1", "1|2", "2|2", "./.", "0/2"]
            } else {
                &["0|0", "0|1", "1|1", "./.", "1/0"]
            };
            let any_ploidy: &[&'static str] = &["0|0", "0|1", "1|1", "./.", "0", "1", "0/1/1", "."];
            for _ in 0..samples {
                vcf.push('\t');
                match layout {
                    0 => {
                        vcf.push_str(draws.pick(diploid));
                        vcf.push(':');
                        vcf.push_str(&dosage_text(draws, alt_count));
                    }
                    1 => vcf.push_str(&dosage_text(draws, alt_count)),
                    _ if alt_count == 1 => vcf.push_str(draws.pick(any_ploidy)),
                    _ => vcf.push_str(draws.pick(diploid)),
                }
            }
            vcf.push_str(if draws.below(12) == 0 { "\r\n" } else { "\n" });

            if chromosome == "22" && draws.below(3) > 0 {
                // A DS with a missing ALT value has no REF dosage, which a REF-effect rule needs.
                let (effect, other) = if layout >= 2 && draws.below(3) == 0 {
                    ("A", "G")
                } else {
                    ("G", "A")
                };
                let weights = ["0.5", "-0.25", "1e-3", "", "0.123456"];
                let (first, second) = (draws.pick(&weights), draws.pick(&weights));
                score.push_str(&format!("22:{position}\t{effect}\t{other}\t{first}\t{second}\n"));
            }
        }
        (vcf, score)
    }

    /// DS text for `alt_count` ALT alleles: dosages of at most one copy each, some missing.
    fn dosage_text(draws: &mut Draws, alt_count: usize) -> String {
        (0..alt_count)
            .map(|_| draws.pick(&["0", "0.25", "0.95", "."]))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// The one-pass line visitor gives the lines, and their ASCII flags, that a
    /// split after each newline gives, at every length and chunk alignment, and
    /// stops at the line `visit` refuses.
    #[test]
    fn lines_are_visited_with_their_ascii_flags() {
        let mut draws = Draws(0x11e5);
        for _ in 0..4000 {
            let bytes: Vec<u8> = (0..draws.below(300))
                .map(|_| match draws.below(12) {
                    0 => b'\n',
                    1 => 0xc3,
                    _ => b'a',
                })
                .collect();
            let mut expected = Vec::new();
            let mut start = 0usize;
            for (index, &byte) in bytes.iter().enumerate() {
                if byte == b'\n' {
                    expected.push((bytes[start..=index].to_vec(), bytes[start..=index].is_ascii()));
                    start = index + 1;
                }
            }
            if start < bytes.len() {
                expected.push((bytes[start..].to_vec(), bytes[start..].is_ascii()));
            }
            let stop = if draws.below(4) == 0 {
                1 + draws.below(expected.len() + 1)
            } else {
                usize::MAX
            };
            let mut actual = Vec::new();
            for_each_line(&bytes, |line, ascii| {
                actual.push((line.to_vec(), ascii));
                actual.len() < stop
            });
            expected.truncate(stop);
            assert_eq!(actual, expected, "{bytes:?}");
        }
    }

    /// A plain decimal the fast path reads is the number `str::parse` reads.
    #[test]
    fn plain_decimals_read_as_str_parse_reads_them() {
        let mut draws = Draws(0xdec1);
        let mut read = 0usize;
        for _ in 0..200_000 {
            let mut text = String::new();
            for _ in 0..draws.below(18) {
                text.push(char::from(b"0123456789"[draws.below(10)]));
            }
            if draws.below(3) > 0 {
                text.insert(draws.below(text.len() + 1), '.');
            }
            if let Some(dose) = plain_decimal(text.as_bytes()) {
                let value: f64 = format!("{}e-{}", dose.digits, dose.places).parse().unwrap();
                assert_eq!(
                    Some(value.to_bits()),
                    text.parse::<f64>().ok().map(f64::to_bits),
                    "{text:?}"
                );
                read += 1;
            }
        }
        assert!(read > 50_000, "{read} decimals read by the fast path");
    }

    /// Scores a plain VCF one record at a time as noodles reads it, which every
    /// batching must reproduce bit for bit.
    fn score_one_record_at_a_time(
        vcf_path: &Path,
        score_path: &Path,
        keep: Option<&Path>,
    ) -> Result<NativeVcfScoreResult, BoxError> {
        let (score_names, rules_by_key) =
            load_score_rules(std::slice::from_ref(&score_path.to_path_buf()), None)?;
        let mut reader = VcfReader::new(BufReader::new(File::open(vcf_path)?));
        let header = crate::variant_header::read_vcf_header(&mut reader)?.header;
        let all_samples: Vec<String> = header.sample_names().iter().cloned().collect();
        let kept_indices = resolve_keep_indices(keep, &all_samples)?;
        let person_iids = kept_indices
            .iter()
            .map(|&idx| all_samples[idx].clone())
            .collect();
        let context = DecodeContext {
            rules_by_key: &rules_by_key,
            kept_indices: &kept_indices,
            score_names: &score_names,
        };
        let mut accumulator =
            RecordAccumulator::new(&rules_by_key, &score_names, kept_indices.len());
        let mut record = noodles_vcf::Record::default();
        while reader.read_record(&mut record)? > 0 {
            let mut decoded = DecodedRecord::default();
            if let Err(err) = decode_scored_record(&record, &context, &mut decoded) {
                decoded.error = Some(err);
            }
            accumulator.take(&mut decoded)?;
            accumulator.apply();
        }
        let totals = accumulator.finish()?;
        native_result(totals, person_iids, score_names, vcf_path)
    }

    /// Batched scoring of plain and BGZF input, at any block size, thread count
    /// and keep list, gives the bits of scoring one record at a time.
    #[test]
    fn batched_scores_match_scoring_one_record_at_a_time() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut draws = Draws(0xba7c);
        for (samples, records) in [(7, 400), (700, 150)] {
            let (vcf, score) = batch_cohort(&mut draws, samples, records);
            let vcf_path = dir.path().join("cohort.vcf");
            let bgzf_path = dir.path().join("cohort.vcf.gz");
            let score_path = dir.path().join("score.gnomon.tsv");
            let keep_path = dir.path().join("keep.txt");
            std::fs::write(&vcf_path, &vcf).expect("write vcf");
            std::fs::write(&score_path, &score).expect("write score");
            std::fs::write(&keep_path, "s0\ns3\ns5\n").expect("write keep");
            for keep in [None, Some(keep_path.as_path())] {
                let expected = score_one_record_at_a_time(&vcf_path, &score_path, keep)
                    .expect("score one record at a time");
                assert!(expected.matched_variants > records / 4);
                for threads in [1, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .expect("thread pool");
                    for block_len in [97, 4096, 65536] {
                        std::fs::write(&bgzf_path, bgzf_bytes(vcf.as_bytes(), block_len))
                            .expect("write bgzf");
                        for path in [&vcf_path, &bgzf_path] {
                            let context = format!(
                                "{samples} samples, keep {}, {threads} threads, {} at block {block_len}",
                                keep.is_some(),
                                path.display()
                            );
                            let actual = pool
                                .install(|| {
                                    score_vcf_streaming(
                                        path,
                                        std::slice::from_ref(&score_path),
                                        keep,
                                        None,
                                    )
                                })
                                .unwrap_or_else(|err| panic!("{context}: {err}"));
                            assert_eq!(expected.person_iids, actual.person_iids, "{context}");
                            assert_eq!(
                                expected.score_variant_counts, actual.score_variant_counts,
                                "{context}"
                            );
                            assert_eq!(expected.missing_counts, actual.missing_counts, "{context}");
                            let bits = |values: &[f64]| {
                                values.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
                            };
                            assert_eq!(
                                bits(&expected.sums()),
                                bits(&actual.sums()),
                                "{context}"
                            );
                        }
                    }
                }
            }
        }
    }
}
