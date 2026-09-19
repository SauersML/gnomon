//! Native VCF and BCF scoring over batches of inflated bytes.
//!
//! A batch is read, inflated block by block on the rayon pool straight into one
//! buffer, and cut at record boundaries into parts. Each part's records are
//! filtered and decoded on the pool, and the decoded records are then taken in
//! input order, so every sum takes the same terms in the same sequence as a
//! one-record-at-a-time scan, and the first error raised is that scan's.

use super::{
    BGZF_HEADER_LEN, BGZF_MAX_DATA_LEN, BGZF_TRAILER_LEN, DecodeContext, DecodedRecord, Fit,
    NativeVcfScoreResult, RecordAccumulator, ScoreRules, Share, decode_scored_bcf_record,
    decode_vcf_line, grown_bytes, inflate_bgzf_block, is_bgzf_header, is_skippable_record,
    native_result, resolve_keep_indices,
};
use crate::score::pipeline::{MemoryBudget, format_bytes, thread_stack_bytes};
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

/// Blocks inflated per rayon worker in one batch; a batch of plain text or gzip members reads as
/// many blocks' worth of bytes.
const BGZF_BLOCKS_PER_WORKER: usize = 64;
/// Parts a batch is cut into per rayon worker, so one slow part leaves few workers idle.
const PARTS_PER_WORKER: usize = 4;
/// Compressed bytes requested from the source per read.
const SOURCE_READ_LEN: usize = 1 << 20;
/// Batches of decoded records resident at once: the batch being taken. A pipeline that decodes
/// the next batch while it takes one holds two, and sets this to 2.
const DECODED_BATCHES_IN_FLIGHT: usize = 1;

/// What the text and compressed buffers hold: the bytes each has touched, which it keeps, and
/// the bytes it may hold before it moves to a larger buffer.
#[derive(Clone, Copy)]
struct Buffers {
    text: usize,
    text_capacity: usize,
    compressed: usize,
    compressed_capacity: usize,
}

/// The bytes a buffer that has touched `held` bytes of `capacity` holds once it holds `needed`:
/// the most it has touched, and while it moves to a larger buffer, the one it moves from.
fn buffer_bytes(held: usize, capacity: usize, needed: usize) -> usize {
    held.max(needed).saturating_add(if needed > capacity { held } else { 0 })
}

/// What a native run holds against its memory budget, and so how large a batch it may take.
///
/// A run holds what the process held when the budget was read (the score rules, the header, the
/// kept people), the accumulator's sums and queues ([`RecordAccumulator::bytes`]), the stacks of
/// the pool's workers, the text buffer, a BGZF batch's compressed blocks with the source read
/// ahead, what the accumulator holds between batches, and the decoded records of a batch. Each
/// scored record is counted before any of it is allocated ([`super::scored_record_need`]), so a
/// batch holds what its part of the budget holds, and one that meets a record beyond it stops
/// there. A batch is read at a size whose records fit even were every one of them as short as a
/// record carrying its samples can be, and as needy as a record at the neediest position, or
/// when that is not one block, at one block.
struct MemoryCharge {
    budget: MemoryBudget,
    sums: usize,
    stacks: usize,
    bgzf: bool,
    people: usize,
    scores: usize,
    threads: usize,
    /// The fewest bytes of text a record carrying its samples holds.
    shortest_record: usize,
    /// What a conforming record at the neediest and at the least needy position needs.
    most_need: usize,
    least_need: usize,
    parts: usize,
}

impl MemoryCharge {
    #[allow(clippy::too_many_arguments)]
    fn new(
        budget: MemoryBudget,
        layout: Layout<'_>,
        bgzf: bool,
        samples: usize,
        people: usize,
        scores: usize,
        rules_by_key: &ScoreRules,
        threads: usize,
    ) -> Self {
        let (most_need, least_need) = super::position_needs(rules_by_key, people);
        Self {
            budget,
            sums: RecordAccumulator::bytes(people, scores, rules_by_key),
            stacks: threads.saturating_mul(thread_stack_bytes()),
            bgzf,
            people,
            scores,
            threads,
            shortest_record: shortest_record(layout, samples),
            most_need,
            least_need,
            parts: threads * PARTS_PER_WORKER,
        }
    }

    /// What a batch of `blocks` BGZF blocks holds to inflate them: its compressed blocks, the part
    /// of a block and the read ahead the last batch left, and one more read; each block's frame,
    /// output and flag, within 128 bytes; and each worker's libdeflate decompressor, within
    /// 32 KiB.
    fn compressed(&self, blocks: usize) -> usize {
        if self.bgzf {
            blocks
                .saturating_add(1)
                .saturating_mul(BGZF_MAX_DATA_LEN + 128)
                .saturating_add(2 * SOURCE_READ_LEN)
                .saturating_add(self.threads.saturating_mul(32 << 10))
        } else {
            0
        }
    }

    /// What the decoded records of a batch of `text` bytes may need: its parts' buffers, and for
    /// each record as short as a record carrying its samples can be, what a record at the
    /// neediest position needs, and four bytes a byte of text for the chromosome, REF and ALT
    /// copies a record takes with their allocations.
    fn planned(&self, text: usize) -> usize {
        (text / self.shortest_record.max(1))
            .saturating_mul(self.most_need)
            .saturating_add(text.saturating_mul(4))
            .saturating_add(self.parts.saturating_mul(super::part_need()))
    }

    /// What the run holds besides the decoded records, with `held` held between batches.
    fn fixed(&self, held: usize) -> usize {
        self.budget
            .resident_bytes()
            .saturating_add(self.sums)
            .saturating_add(self.stacks)
            .saturating_add(held)
    }

    /// The bytes the run holds with `text` bytes of text over `blocks` new blocks, from
    /// `buffers`, with `held` held between batches.
    fn required(&self, text: usize, blocks: usize, buffers: Buffers, held: usize) -> usize {
        self.fixed(held)
            .saturating_add(buffer_bytes(buffers.text, buffers.text_capacity, text))
            .saturating_add(buffer_bytes(buffers.compressed, buffers.compressed_capacity, self.compressed(blocks)))
            .saturating_add(DECODED_BATCHES_IN_FLIGHT.saturating_mul(self.planned(text)))
    }

    /// Refuses a run that cannot hold its sums and a batch of one record: the shortest record's
    /// text and a block read after it, and what a record at the least needy position needs.
    /// Nothing is allocated for either before this.
    fn check_floor(&self, buffers: Buffers) -> Result<(), BoxError> {
        let text = self.shortest_record.saturating_add(BGZF_MAX_DATA_LEN);
        let batch = buffer_bytes(buffers.text, buffers.text_capacity, text)
            .saturating_add(buffer_bytes(buffers.compressed, buffers.compressed_capacity, self.compressed(1)))
            .saturating_add(super::part_need())
            .saturating_add(self.least_need);
        let required = self.fixed(0).saturating_add(batch);
        if required > self.budget.max_ram_bytes() {
            return Err(format!(
                "Scoring requires at least {} for {} people and {} scores ({} for their sums, {} for the stacks of {} threads, {} for a batch of one record, {} this process already holds), exceeding the {} memory budget. Reduce the kept cohort or score panel.",
                format_bytes(required),
                self.people,
                self.scores,
                format_bytes(self.sums),
                format_bytes(self.stacks),
                self.threads,
                format_bytes(batch),
                format_bytes(self.budget.resident_bytes()),
                format_bytes(self.budget.max_ram_bytes()),
            )
            .into());
        }
        Ok(())
    }

    /// The most whole blocks, at most `limit`, the next batch may add to `leftover` bytes of text
    /// the last batch left, from `buffers`, with `held` held between batches: as many as fit the
    /// plan, or else one, when its text and one record at the least needy position fit, and its
    /// records are decoded while they fit. When the leftover holds complete records a cut batch
    /// left (`carried`), none need be added; otherwise it is at most one record's part, and a
    /// batch that cannot add a block refuses.
    fn blocks(&self, leftover: usize, buffers: Buffers, held: usize, limit: usize, carried: bool) -> Result<usize, BoxError> {
        let fits = |blocks: usize| {
            self.required(leftover.saturating_add(blocks * BGZF_MAX_DATA_LEN), blocks, buffers, held)
                <= self.budget.max_ram_bytes()
        };
        if fits(1) {
            // `required` grows with `blocks`, so the most that fit are found by bisection.
            let (mut low, mut high) = (1usize, limit.max(1));
            while low < high {
                let mid = low + (high - low).div_ceil(2);
                if fits(mid) { low = mid } else { high = mid - 1 }
            }
            return Ok(low);
        }
        let one = self
            .fixed(held)
            .saturating_add(buffer_bytes(buffers.text, buffers.text_capacity, leftover.saturating_add(BGZF_MAX_DATA_LEN)))
            .saturating_add(buffer_bytes(buffers.compressed, buffers.compressed_capacity, self.compressed(1)))
            .saturating_add(DECODED_BATCHES_IN_FLIGHT.saturating_mul(
                self.parts.saturating_mul(super::part_need()).saturating_add(self.least_need),
            ));
        if one <= self.budget.max_ram_bytes() {
            return Ok(1);
        }
        if carried {
            return Ok(0);
        }
        Err(format!(
            "A record longer than {leftover} bytes needs {} with its decoded dosages, exceeding the {} memory budget. Reduce the kept cohort or score panel.",
            format_bytes(one),
            format_bytes(self.budget.max_ram_bytes()),
        )
        .into())
    }

    /// What a batch's decoded records may allocate, from `buffers` once it is read, with `held`
    /// held between batches, beyond its parts' buffers.
    fn decode_allowance(&self, buffers: Buffers, held: usize) -> usize {
        self.budget
            .max_ram_bytes()
            .saturating_sub(self.fixed(held))
            .saturating_sub(buffers.text)
            .saturating_sub(buffers.compressed)
            .saturating_div(DECODED_BATCHES_IN_FLIGHT)
            .saturating_sub(self.parts.saturating_mul(super::part_need()))
    }

    /// The refusal of a record needing `need` bytes at `locus`, beyond the `allowance` its batch
    /// may allocate.
    fn record_refusal(&self, need: usize, locus: &str, allowance: usize) -> String {
        format!(
            "Scoring {locus} requires {} for its decoded dosages, exceeding the {} the {} memory budget leaves beside the sums, the text and what this process already holds. Reduce the kept cohort or score panel.",
            format_bytes(need),
            format_bytes(allowance),
            format_bytes(self.budget.max_ram_bytes()),
        )
    }
}

/// The fewest bytes a record whose samples carry a value each holds: in VCF, nine one-byte
/// fields through FORMAT and a byte a sample, each but the last followed by a tab; in BCF, the
/// two lengths, 24 bytes of fixed site fields, an empty ID, one-byte REF and ALT, an empty
/// FILTER, and one FORMAT field's key, type and a byte a sample. A record carrying no value for
/// its samples (a missing FORMAT, or a field of no values) decodes every kept person missing,
/// and is counted as it is decoded.
fn shortest_record(layout: Layout<'_>, samples: usize) -> usize {
    match layout {
        Layout::Vcf => samples.saturating_mul(2).saturating_add(17),
        Layout::Bcf(_) => samples.saturating_add(41),
    }
}

/// Scores the records of `source` for the kept samples of its header.
pub(super) fn score_source(
    source: VariantSource,
    input_path: &Path,
    keep: Option<&Path>,
    score_names: Vec<String>,
    rules_by_key: &ScoreRules,
    budget: impl FnOnce() -> MemoryBudget,
) -> Result<NativeVcfScoreResult, BoxError> {
    let format = source.format();
    let bgzf = matches!(source.compression(), VariantCompression::Bgzf);
    let threads = rayon::current_num_threads().max(1);
    let mut stream = InflatedStream::new(source);
    let mut bytes = Bytes::default();
    let mut at_end = false;
    let batch_blocks = threads * BGZF_BLOCKS_PER_WORKER;
    // The header is read a block at first and twice as many each time after, so past its end it
    // reads less than its own length and a block, and its scans total less than twice what it
    // reads. Reading a whole batch each time put two batches of text in the first batch.
    let mut header_blocks = 1usize;
    let header_len = loop {
        let len = match format {
            VariantFormat::Vcf => vcf_header_len(bytes.filled(), at_end),
            VariantFormat::Bcf => bcf_header_len(bytes.filled(), at_end),
        };
        if let Some(len) = len {
            break len;
        }
        at_end = !stream.fill(&mut bytes, header_blocks)?;
        header_blocks = (2 * header_blocks).min(batch_blocks);
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
    // What is held so far (rules, header, people) is read with the budget, before the sums and
    // the first batch are allocated; a run that cannot hold them refuses here.
    let charge = MemoryCharge::new(
        budget(),
        layout,
        bgzf,
        all_samples.len(),
        person_iids.len(),
        score_names.len(),
        rules_by_key,
        threads,
    );
    charge.check_floor(stream.buffers(&bytes))?;
    let empty = Buffers {
        text: 0,
        text_capacity: 0,
        compressed: 0,
        compressed_capacity: 0,
    };
    eprintln!(
        "> Native plan: {} budget, {} for the sums, {} already resident, up to {} to decode a record, {} for a full batch of {batch_blocks} blocks.",
        format_bytes(charge.budget.max_ram_bytes()),
        format_bytes(charge.sums),
        format_bytes(charge.budget.resident_bytes()),
        format_bytes(charge.most_need),
        format_bytes(charge.required(batch_blocks * BGZF_MAX_DATA_LEN, batch_blocks, empty, 0) - charge.fixed(0)),
    );
    let mut accumulator = RecordAccumulator::new(rules_by_key, &score_names, person_iids.len());
    // The records the header's last block holds past the header are taken before more is read,
    // so that besides its new blocks a batch holds at most the part of one record the last left.
    let mut read = bytes.filled().is_empty();
    // Whether the text past the records taken holds complete records a cut batch left.
    let mut carried = false;
    let mut source_error = None;
    loop {
        if !at_end && source_error.is_none() && std::mem::replace(&mut read, true) {
            let blocks = charge.blocks(
                bytes.filled().len(),
                stream.buffers(&bytes),
                accumulator.held_bytes(),
                batch_blocks,
                carried,
            )?;
            if blocks > 0 {
                match stream.fill(&mut bytes, blocks) {
                    Ok(more) => at_end = !more,
                    Err(err) => source_error = Some(err),
                }
            }
        }
        let allowance = charge.decode_allowance(stream.buffers(&bytes), accumulator.held_bytes());
        let filled = bytes.filled();
        // A read that failed leaves its last record unread, as noodles leaves it.
        let (complete, stops) = complete_records(layout, filled, at_end);
        let records = &filled[..complete];
        // Each part may allocate its share of the allowance, by its share of the text.
        let ranges = parts(layout, records, charge.parts);
        let share = |len: usize| Share {
            left: (allowance as u128 * len as u128 / complete.max(1) as u128) as usize,
        };
        let decoded: Vec<DecodedPart> = ranges
            .par_iter()
            .map(|range| decode_part(layout, &records[range.clone()], &context, share(range.len())))
            .collect();
        // The records before the first one a part could not hold are taken; the rest wait.
        let mut taken = complete;
        let mut cut = false;
        for (range, part) in ranges.iter().zip(decoded) {
            for mut record in part.records {
                accumulator.take(&mut record)?;
            }
            if let Some((offset, _, _)) = part.cut {
                taken = range.start + offset;
                cut = true;
                break;
            }
        }
        if cut && taken == 0 {
            // The first record needs more than its part's share: it is decoded alone, with all
            // that the batch may allocate, or refused.
            let first = first_record_len(layout, records);
            let alone = decode_part(layout, &records[..first], &context, Share { left: allowance });
            if let Some((_, need, locus)) = alone.cut {
                return Err(charge.record_refusal(need, &locus, allowance).into());
            }
            for mut record in alone.records {
                accumulator.take(&mut record)?;
            }
            taken = first;
        }
        accumulator.apply();
        carried = taken < complete;

        if !carried && let Some(err) = source_error {
            return Err(err.into());
        }
        if (at_end || stops) && !carried {
            if matches!(layout, Layout::Bcf(_)) && !stops && complete < filled.len() {
                // A final record cut short fails as noodles fails to read it.
                BcfReader::from(&filled[complete..])
                    .read_record(&mut noodles_bcf::Record::default())?;
            }
            let totals = accumulator.finish()?;
            return native_result(totals, person_iids, score_names, input_path);
        }
        bytes.consume(taken);
    }
}

/// The length of the first of complete `records`.
fn first_record_len(layout: Layout<'_>, records: &[u8]) -> usize {
    match layout {
        Layout::Vcf => memchr(b'\n', records).map_or(records.len(), |end| end + 1),
        Layout::Bcf(_) => bcf_record_len(records, 0),
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
/// position, up to and including the first record that raises an error, and
/// before the first record whose need does not fit `share`.
fn decode_part(layout: Layout<'_>, bytes: &[u8], context: &DecodeContext<'_>, mut share: Share) -> DecodedPart {
    let mut decoded = Vec::new();
    let mut cut = None;
    match layout {
        Layout::Vcf => {
            let mut start = 0usize;
            for_each_line(bytes, |line, ascii| {
                let offset = start;
                start += line.len();
                let text = line.strip_suffix(b"\n").unwrap_or(line);
                if is_skippable_record(text, ascii, context.rules_by_key) {
                    return true;
                }
                let mut record = DecodedRecord::default();
                record.error = match decode_vcf_line(line, context, &mut record, &mut share) {
                    Ok(Fit::Beyond { need, locus }) => {
                        cut = Some((offset, need, locus));
                        return false;
                    }
                    Ok(Fit::Decoded) => None,
                    Err(err) => Some(err),
                };
                !keep_decoded(&mut decoded, record)
            });
        }
        Layout::Bcf(header) => {
            let mut reader = BcfRecordReader::default();
            let mut start = 0usize;
            while start < bytes.len() {
                let end = start + bcf_record_len(bytes, start);
                let mut record = DecodedRecord::default();
                record.error = match reader.decode(&bytes[start..end], header, context, &mut record, &mut share) {
                    Ok(Fit::Beyond { need, locus }) => {
                        cut = Some((start, need, locus));
                        break;
                    }
                    Ok(Fit::Decoded) => None,
                    Err(err) => Some(err),
                };
                start = end;
                if keep_decoded(&mut decoded, record) {
                    break;
                }
            }
        }
    }
    DecodedPart { records: decoded, cut }
}

/// A part's decoded records, and where it stopped when a record did not fit its share.
struct DecodedPart {
    records: Vec<DecodedRecord>,
    /// The offset in the part of the first record left undecoded, with what it needs and where
    /// it is.
    cut: Option<(usize, usize, String)>,
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
    /// The longest record these buffers have held.
    longest: usize,
}

impl BcfRecordReader {
    /// Decodes the complete record `bytes` into `decoded`, as
    /// `decode_scored_bcf_record` decodes a record noodles read, when it fits
    /// `share`. A record longer than any before it grows the reader's four
    /// buffers (the site's bytes, the site's fields, the record's fields and its
    /// samples), each at most as `grown_bytes` counts it.
    fn decode(
        &mut self,
        bytes: &[u8],
        header: &noodles_vcf::Header,
        context: &DecodeContext<'_>,
        decoded: &mut DecodedRecord,
        share: &mut Share,
    ) -> Result<Fit, BoxError> {
        if bytes.len() > self.longest {
            let grown = 4 * (grown_bytes(bytes.len(), 1) - grown_bytes(self.longest, 1));
            if !share.take(grown) {
                return Ok(Fit::Beyond {
                    need: grown,
                    locus: format!("a record of {} bytes", bytes.len()),
                });
            }
            self.longest = bytes.len();
        }
        let site_len = read_u32(bytes, 0).expect("a complete record has a site length");
        self.site_bytes.clear();
        self.site_bytes.extend_from_slice(&bytes[..4]);
        self.site_bytes.extend_from_slice(&0u32.to_le_bytes());
        self.site_bytes.extend_from_slice(&bytes[8..8 + site_len]);
        BcfReader::from(&self.site_bytes[..]).read_record(&mut self.site)?;

        let chromosome = self.site.reference_sequence_name(header.string_maps())?;
        let Ok(chr) = parse_chromosome_label(chromosome) else {
            return Ok(Fit::Decoded);
        };
        let Some(start) = self.site.variant_start() else {
            return Ok(Fit::Decoded);
        };
        if !context.rules_by_key.contains_key(&(chr, start?.get() as u32)) {
            return Ok(Fit::Decoded);
        }
        BcfReader::from(bytes).read_record(&mut self.record)?;
        decode_scored_bcf_record(&self.record, header, context, decoded, share)
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
    /// What the gzip decoder a stream fell back to holds: the compressed bytes it was handed and
    /// its own buffers, within a source read.
    fallback: usize,
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
            fallback: 0,
        }
    }

    /// What `text` and this stream's buffers hold.
    fn buffers(&self, text: &Bytes) -> Buffers {
        Buffers {
            text: text.buf.len(),
            text_capacity: text.buf.capacity(),
            compressed: self.compressed.buf.len().saturating_add(self.fallback),
            compressed_capacity: self.compressed.buf.capacity(),
        }
    }

    /// Appends at most `blocks` BGZF blocks of inflated bytes to `out`, or as many
    /// blocks' worth of other bytes, returning `false` once the stream is exhausted.
    fn fill(&mut self, out: &mut Bytes, blocks: usize) -> io::Result<bool> {
        if let Input::Text(reader) = &mut self.input {
            return read_text(reader.as_mut(), out, blocks * BGZF_MAX_DATA_LEN);
        }
        if matches!(self.input, Input::Done) {
            return Ok(false);
        }
        self.fill_bgzf(out, blocks)
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
        // The compressed bytes move to the decoder in the buffer that holds them.
        let Bytes { mut buf, len } = std::mem::take(&mut self.compressed);
        self.fallback = buf.len().saturating_add(SOURCE_READ_LEN);
        buf.truncate(len);
        buf.drain(..start);
        self.start = 0;
        self.input = Input::Text(Box::new(MultiGzDecoder::new(
            Cursor::new(buf).chain(source),
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
    use super::super::tests::cohort_files;
    use super::super::*;
    use super::{BoxError, Bytes, InflatedStream, Layout, for_each_line, score_source};
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

    /// Buffers that hold nothing yet.
    const EMPTY: super::Buffers = super::Buffers {
        text: 0,
        text_capacity: 0,
        compressed: 0,
        compressed_capacity: 0,
    };

    /// The charge is the stated sum of its terms (#2396). A batch takes the most blocks whose
    /// records fit the plan, else one when its text and one record at the least needy position
    /// fit, else none when it carries records a cut batch left, and refuses otherwise; the floor
    /// holds at its value and refuses one byte below it.
    #[test]
    fn the_charge_plans_batches_and_refuses_below_its_floor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.5\n1:100\tT\tA\t1\n1:200\tC\t.\t2\n",
        )
        .expect("write score");
        let (_, rules) = load_score_rules(std::slice::from_ref(&score_path), None).expect("rules");
        let (samples, held, threads) = (10, 4096, 2);
        let charge = |max| {
            super::MemoryCharge::new(MemoryBudget::of(max, held), Layout::Vcf, true, samples, samples, 1, &rules, threads)
        };
        // The terms, stated apart from the code.
        let shortest = 17 + 2 * samples;
        let (most, least) = position_needs(&rules, samples);
        assert!(least < most, "the effect-only position needs less than the two-rule one");
        let parts = threads * 4;
        let planned = |text: usize| text / shortest * most + 4 * text + parts * part_need();
        let compressed = |blocks: usize| (blocks + 1) * (65_536 + 128) + 2 * (1 << 20) + threads * (32 << 10);
        let fixed = |between: usize| {
            held + RecordAccumulator::bytes(samples, 1, &rules) + threads * crate::score::pipeline::thread_stack_bytes() + between
        };
        let required = |text: usize, blocks: usize, between: usize| fixed(between) + text + compressed(blocks) + planned(text);
        assert_eq!(charge(usize::MAX).required(100_000, 2, EMPTY, 500), required(100_000, 2, 500));

        let leftover = 300;
        let three = required(leftover + 3 * 65_536, 3, 500);
        assert_eq!(charge(three).blocks(leftover, EMPTY, 500, 64, false).expect("three fit"), 3);
        assert_eq!(charge(three + 1).blocks(leftover, EMPTY, 500, 64, false).expect("three fit"), 3);
        assert_eq!(charge(three).blocks(leftover, EMPTY, 500, 2, false).expect("the limit"), 2);
        let one = fixed(500) + leftover + 65_536 + compressed(1) + parts * part_need() + least;
        assert!(one < required(leftover + 65_536, 1, 500), "one record needs less than the plan");
        assert_eq!(charge(one).blocks(leftover, EMPTY, 500, 64, false).expect("one block"), 1);
        assert_eq!(charge(one - 1).blocks(leftover, EMPTY, 500, 64, true).expect("carried"), 0);
        let refused = charge(one - 1).blocks(leftover, EMPTY, 500, 64, false).expect_err("not one block fits");
        assert!(refused.to_string().starts_with("A record longer than 300 bytes"), "{refused}");
        // A buffer that must move to a larger one holds the one it moves from as well.
        let grown = super::Buffers {
            text: 70_000,
            text_capacity: 70_000,
            ..EMPTY
        };
        assert_eq!(
            charge(usize::MAX).required(100_000, 2, grown, 500),
            required(100_000, 2, 500) + 70_000
        );

        let floor = fixed(0) + shortest + 65_536 + compressed(1) + part_need() + least;
        assert!(charge(floor).check_floor(EMPTY).is_ok());
        let refused = charge(floor - 1).check_floor(EMPTY).expect_err("one byte below the floor");
        assert!(refused.to_string().starts_with("Scoring requires at least"), "{refused}");

        let read = super::Buffers {
            text: 200_000,
            text_capacity: 262_144,
            compressed: 3 << 20,
            compressed_capacity: 4 << 20,
        };
        assert_eq!(
            charge(10 << 30).decode_allowance(read, 500),
            (10 << 30) - fixed(500) - 200_000 - (3 << 20) - parts * part_need()
        );
    }

    /// A record carrying one one-byte ALT allele for each rule at its position, with a one-byte
    /// chromosome and REF, needs what the plan charges a record at that position.
    #[test]
    fn a_conforming_record_needs_what_its_position_is_planned_to() {
        let dir = tempfile::tempdir().expect("tempdir");
        for (rows, (reference, alternates)) in [
            ("1:100\tG\tA\t0.5\n1:100\tT\tA\t1\n", ("A", "G,T")),
            ("1:200\tC\t.\t2\n", ("C", "T")),
        ] {
            let score_path = dir.path().join("score.gnomon.tsv");
            std::fs::write(&score_path, format!("variant_id\teffect_allele\tother_allele\tS\n{rows}"))
                .expect("write score");
            let (_, rules) = load_score_rules(std::slice::from_ref(&score_path), None).expect("rules");
            let &(start, end) = rules.ranges.values().next().expect("one position");
            let score_rules = &rules.rules[start..end];
            let need = scored_record_need(
                &rules,
                start,
                score_rules,
                12,
                "1",
                reference,
                || alt_alleles_of(alternates),
            );
            assert_eq!(position_needs(&rules, 12), (need, need), "{rows}");
        }
    }

    /// Live heap bytes, counted while `COUNTING` is set, and the most since `PEAK` was reset.
    /// The lib's tests share one allocator, so the counts are read only in a process that runs
    /// one test.
    struct CountingAllocator;

    static COUNTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    static LIVE: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);
    static PEAK: std::sync::atomic::AtomicIsize = std::sync::atomic::AtomicIsize::new(0);

    fn count(bytes: isize) {
        use std::sync::atomic::Ordering::Relaxed;
        if COUNTING.load(Relaxed) {
            let live = LIVE.fetch_add(bytes, Relaxed) + bytes;
            PEAK.fetch_max(live, Relaxed);
        }
    }

    unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            count(layout.size() as isize);
            unsafe { std::alloc::System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            count(layout.size() as isize);
            unsafe { std::alloc::System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, size: usize) -> *mut u8 {
            // The buffer moved from is counted until the move is done.
            count(size as isize);
            let moved = unsafe { std::alloc::System.realloc(ptr, layout, size) };
            count(-(layout.size() as isize));
            moved
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
            count(-(layout.size() as isize));
            unsafe { std::alloc::System.dealloc(ptr, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    /// Scores `path` with the rules of `score_path` within a budget of `budget` bytes, read while
    /// this process holds nothing it counts, and gives what the run added to the heap at most.
    fn counted_run(path: &Path, score_path: &Path, budget: usize) -> (Result<NativeVcfScoreResult, BoxError>, usize) {
        use std::sync::atomic::Ordering::Relaxed;
        let scored = load_score_rules(std::slice::from_ref(&score_path.to_path_buf()), None)
            .and_then(|(names, rules)| Ok((names, rules, open_variant_source(path)?)));
        let (names, rules, source) = match scored {
            Ok(scored) => scored,
            Err(err) => return (Err(err), 0),
        };
        let base = std::sync::atomic::AtomicIsize::new(0);
        let result = score_source(source, path, None, names, &rules, || {
            let live = LIVE.load(Relaxed);
            base.store(live, Relaxed);
            PEAK.store(live, Relaxed);
            COUNTING.store(true, Relaxed);
            MemoryBudget::of(budget, 0)
        });
        COUNTING.store(false, Relaxed);
        let added = PEAK.load(Relaxed) - base.load(Relaxed);
        (result, usize::try_from(added).unwrap_or(0))
    }

    /// The memory budget bounds the heap a native run allocates (#2396), over cohorts whose
    /// decoded records dwarf their text: one sample at a time, dosages of forty, multiallelic
    /// records at a position where a rule names no single other allele, split multiallelic records,
    /// some repeated, whose sites build columns for a REF-effect row and a repeated ALT, and 500
    /// scores at every position; each as plain VCF, BGZF VCF and BCF. For each, the least budget that runs
    /// it is found: one byte less refuses by name, and at it and above it the run scores what an
    /// unbounded run scores, adding no more to the heap than its budget. A budget too small for
    /// the sums refuses before allocating anything, and a record no budget below its own need
    /// decodes is refused by its position.
    #[test]
    fn the_budget_bounds_the_heap_a_native_run_allocates() {
        const CHILD: &str = "GNOMON_NATIVE_HEAP_CHILD";
        if std::env::var_os(CHILD).is_none() {
            let output = std::process::Command::new(std::env::current_exe().expect("test executable"))
                .args([
                    "--exact",
                    "score::native_vcf::stream::tests::the_budget_bounds_the_heap_a_native_run_allocates",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .expect("run the counted scoring alone");
            assert!(
                output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed"),
                "counted scoring failed:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            eprint!("{}", String::from_utf8_lossy(&output.stderr));
            return;
        }

        let dir = tempfile::tempdir().expect("tempdir");
        let stacks = rayon::current_num_threads() * crate::score::pipeline::thread_stack_bytes();
        let header = |samples: usize| {
            let names: Vec<String> = (0..samples).map(|sample| format!("s{sample}")).collect();
            format!(
                "##fileformat=VCFv4.2\n##contig=<ID=1>\n\
                 ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
                 ##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"ALT dosage\">\n\
                 #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}\n",
                names.join("\t")
            )
        };
        let calls = |samples: usize, position: usize, values: [&str; 3]| -> String {
            (0..samples).map(|sample| values[(sample + position) % 3]).collect::<Vec<_>>().join("\t")
        };
        // Four cohorts, each with its score rows.
        let one_sample: String = (1..=20_000)
            .map(|position| format!("1\t{position}\t.\tA\tG\t.\t.\t.\tDS\t{}\n", ["0", "1", "2"][position % 3]))
            .collect();
        let one_sample_rows: String = (1..=20_000).map(|position| format!("1:{position}\tG\tA\t0.5\n")).collect();
        let dosages: String = (1..=2_000)
            .map(|position| {
                format!("1\t{position}\t.\tA\tG\t.\t.\t.\tGT:DS\t{}\n", calls(40, position, ["0|0:0", "0|1:0.9", "1|1:1.8"]))
            })
            .collect();
        let dosage_rows: String = (1..=2_000).step_by(2).map(|position| format!("1:{position}\tG\tA\t0.5\n")).collect();
        let bases = ["A", "C", "G", "T"];
        let alternates: Vec<String> =
            (0..60).map(|index| format!("C{}{}{}", bases[index / 16], bases[index / 4 % 4], bases[index % 4])).collect();
        let multiallelic: String = (1..=600)
            .map(|position| {
                let alt = if position % 10 == 0 { alternates.join(",") } else { "G".to_string() };
                format!("1\t{position}\t.\tA\t{alt}\t.\t.\t.\tGT\t{}\n", calls(20, position, ["0|0", "0|1", "1|1"]))
            })
            .collect();
        let multiallelic_rows: String = (1..=600)
            .map(|position| {
                if position % 10 == 0 { format!("1:{position}\tA\t.\t1\n") } else { format!("1:{position}\tG\tA\t1\n") }
            })
            .collect();
        // A wide panel: 25 rows at every position, in both orientations, each weighing its own 20
        // of 500 scores. The calls give the ploidy a REF-effect row's dosage needs.
        let wide_panel: String = (1..=200)
            .map(|position| {
                format!("1\t{position}\t.\tA\tG\t.\t.\t.\tGT:DS\t{}\n", calls(10, position, ["0|0:0", "0|1:0.9", "1|1:1.8"]))
            })
            .collect();
        let wide_panel_rows: String = (1..=200)
            .flat_map(|position| {
                (0..25).map(move |row| {
                    let pair = if row % 2 == 0 { "G\tA" } else { "A\tG" };
                    let weights: String = (0..500)
                        .map(|score| if score / 20 == row { format!("\t{}e-2", score + position) } else { "\t".to_string() })
                        .collect();
                    format!("1:{position}\t{pair}{weights}\n")
                })
            })
            .collect();
        let wide_panel_names: String = (0..500).map(|score| format!("\tS{score:03}")).collect();
        // Split sites, every fifth with its first record repeated: S1 names the REF, so it waits on
        // the whole site, and S2 names only the repeated ALT, whose measurements are combined.
        let split: String = (1..=300)
            .flat_map(|position| {
                let first = format!("1\t{position}\t.\tA\tG\t.\t.\t.\tGT\t{}\n", calls(20, position, ["0|0", "0|1", "1|0"]));
                let second = format!("1\t{position}\t.\tA\tT\t.\t.\t.\tGT\t{}\n", calls(20, position + 1, ["0|0", "1|0", "0|0"]));
                let repeat = if position % 5 == 0 { first.clone() } else { String::new() };
                [first, second, repeat]
            })
            .collect();
        let split_rows: String = (1..=300)
            .map(|position| format!("1:{position}\tA\tG\t1\t\n1:{position}\tG\tA\t0.5\t2\n1:{position}\tT\tA\t0.25\t\n"))
            .collect();
        for (name, samples, body, names, rows) in [
            ("one sample", 1, one_sample, "\tS".to_string(), one_sample_rows),
            ("dosages", 40, dosages, "\tS".to_string(), dosage_rows),
            ("multiallelic", 20, multiallelic, "\tS".to_string(), multiallelic_rows),
            ("split", 20, split, "\tS1\tS2".to_string(), split_rows),
            ("wide panel", 10, wide_panel, wide_panel_names, wide_panel_rows),
        ] {
            let case = dir.path().join(name.replace(' ', "_"));
            std::fs::create_dir_all(&case).expect("case dir");
            let score_path = case.join("score.gnomon.tsv");
            std::fs::write(&score_path, format!("variant_id\teffect_allele\tother_allele{names}\n{rows}"))
                .expect("write score");
            for path in cohort_files(&case, &format!("{}{body}", header(samples))) {
                let (unbounded, unbounded_added) = counted_run(&path, &score_path, usize::MAX / 2);
                let unbounded = unbounded.unwrap_or_else(|err| panic!("{name} {path:?}: {err}"));
                let (refused, added) = counted_run(&path, &score_path, 1 << 10);
                let refused = refused.expect_err("a budget below the sums");
                assert!(refused.to_string().starts_with("Scoring requires at least"), "{name} {path:?}: {refused}");
                assert!(added < 1 << 10, "{name} {path:?}: {added} bytes allocated before refusing");
                // The least budget that runs the cohort, between one that refuses and one that runs.
                let (mut low, mut high) = (1usize << 10, 1usize << 40);
                while high - low > 1 {
                    let mid = low + (high - low) / 2;
                    if counted_run(&path, &score_path, mid).0.is_ok() { high = mid } else { low = mid }
                }
                let (below, _) = counted_run(&path, &score_path, high - 1);
                let below = below.expect_err("one byte below the least budget");
                let message = below.to_string();
                assert!(
                    ["Scoring requires at least", "A record longer than", "Scoring 1:", "Scoring a record of"]
                        .iter()
                        .any(|refusal| message.starts_with(refusal)),
                    "{name} {path:?}: {message}"
                );
                let mut report = Vec::new();
                for budget in [high, high + high / 2, 3 * high] {
                    let (bounded, added) = counted_run(&path, &score_path, budget);
                    let bounded = bounded.unwrap_or_else(|err| panic!("{name} {path:?} at {budget}: {err}"));
                    // The budget charges the workers' stacks too, which are not on the heap.
                    let heap = budget - stacks;
                    assert!(added <= heap, "{name} {path:?}: {added} bytes added to the heap within {heap}");
                    assert_eq!(bounded.sums(), unbounded.sums(), "{name} {path:?} at {budget}");
                    assert_eq!(bounded.missing_counts, unbounded.missing_counts, "{name} {path:?}");
                    assert_eq!(bounded.score_variant_counts, unbounded.score_variant_counts, "{name} {path:?}");
                    report.push(format!("{added} of {heap}"));
                }
                eprintln!(
                    "{name} {path:?}: runs within {high} bytes; unbounded adds {unbounded_added}; heap added {}",
                    report.join(", ")
                );
            }
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

    /// A fill appends at most its count of BGZF blocks, or as many blocks' worth of plain text: the
    /// unit a batch, and each of the header's reads, is sized in.
    #[test]
    fn a_fill_appends_at_most_its_blocks() {
        let dir = tempfile::tempdir().expect("tempdir");
        let text: Vec<u8> = (0..400_000u32).map(|i| b'a' + (i % 26) as u8).collect();
        let plain = dir.path().join("text.vcf");
        let bgzf = dir.path().join("text.vcf.gz");
        std::fs::write(&plain, &text).expect("write text");
        std::fs::write(&bgzf, bgzf_bytes(&text, 1000)).expect("write bgzf");
        for (path, block) in [(plain, BGZF_MAX_DATA_LEN), (bgzf, 1000)] {
            let mut stream = InflatedStream::new(open_variant_source(&path).expect("open"));
            let mut bytes = Bytes::default();
            let mut read = 0;
            for blocks in [1, 2, 3] {
                assert!(stream.fill(&mut bytes, blocks).expect("fill"), "{path:?}");
                read += blocks;
                assert_eq!(bytes.filled(), &text[..read * block], "{path:?}");
            }
        }
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
            let mut share = Share { left: usize::MAX };
            if let Err(err) = decode_scored_record(&record, &context, &mut decoded, &mut share) {
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
