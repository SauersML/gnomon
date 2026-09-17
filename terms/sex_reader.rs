//! Sex evidence read straight from local VCF and BCF files.
//!
//! Sex inference needs a chromosome class, a position and an ALT count for every
//! record, and calls only for the rows it counts: a few thousand thinned autosomes
//! and the X and Y rows. The record reader parses every record into a key with REF
//! and ALT strings, then streams the selected rows through 64-bit dosage blocks.
//! These two passes read the bytes as the files hold them:
//!
//! 1. The blocks of each file (its BGZF blocks, or windows of a plain file) are
//!    inflated or read in parallel batches, and each worker reads the chromosome,
//!    position and ALT count of every record that lies inside its block. The
//!    records are then taken in order, with the ones that cross a block boundary
//!    pieced together. The index keeps how many records start ahead of each block
//!    and where in its block each record starts.
//! 2. Only the selected records are read again: each selected line of a plain VCF
//!    with one positional read, and the blocks holding the selected records of
//!    any other file. The GT calls of each selected row become one packed PLINK 1
//!    row, which the packed counters of `sex_counts` add up.
//!
//! A pass accepts only what the record reader reads the same way. Anything else
//! (a remote file, text that is not ASCII or holds a carriage return, a malformed
//! record, a selected record without GT) makes it return `None`, and the caller
//! reads the dataset through the record reader, which decodes what these passes
//! leave out, or reports what is wrong with the file.

use std::fs::File;
use std::ops::Range;
use std::path::Path;

use libdeflater::{Crc, Decompressor};
use memchr::{memchr, memchr_iter};
use memmap2::Mmap;
use noodles_vcf as vcf;
use rayon::prelude::*;

use crate::bcf_genotypes::GenotypeSeries;
use crate::map::io::{ChromPositionSortState, VcfLikeDataset, read_variant_part_header};
use crate::map::variant_filter::VariantKey;
use crate::terms::sex::{LocusChromosome, VariantLoci, classify_chromosome};
use crate::terms::sex_counts::{
    EvidenceCounter, EvidenceCounts, HET_CODE, LocusClass, MISSING_CODE,
};

/// Bytes per window of a plain file, as many as a BGZF block holds.
const PLAIN_WINDOW_BYTES: usize = 1 << 16;
/// Blocks each rayon worker reads per batch of the first pass.
const BLOCKS_PER_WORKER: usize = 64;
/// Selected records one task of the second pass decodes.
const RECORDS_PER_TASK: usize = 32;
/// Tasks each rayon worker takes per batch of the second pass.
const TASKS_PER_WORKER: usize = 4;

const BGZF_HEADER_LEN: usize = 18;
const BGZF_TRAILER_LEN: usize = 8;
const BGZF_MAX_BLOCK_LEN: usize = 1 << 16;

/// A packed call that is neither missing nor heterozygous.
const HOM_CODE: u8 = 0b11;

/// A dataset's files as the first pass read them, kept for the second pass.
pub(super) struct VariantScan {
    n_samples: usize,
    parts: Vec<ScannedPart>,
    /// Every record whose ALT count is not one, with that count, numbered across
    /// the dataset's files.
    uneven: Vec<(u64, u32)>,
}

/// The first pass's reading of one file.
struct ScannedPart {
    blocks: Blocks,
    index: BlockIndex,
    format: PartFormat,
    /// The number of the file's first record, counted across the dataset.
    first_record: u64,
}

enum PartFormat {
    /// VCF text, whose index counts lines, header lines included.
    Vcf { header_lines: u64 },
    /// BCF, whose index counts records.
    Bcf {
        header: vcf::Header,
        gt_key: Option<usize>,
    },
}

impl VariantScan {
    /// The first pass over every file of `dataset`: the loci of every row, in
    /// the rows' order, or `None` where the record reader must read the files.
    pub(super) fn read(dataset: &VcfLikeDataset) -> Option<(VariantLoci, Self)> {
        Self::read_in_windows(dataset, PLAIN_WINDOW_BYTES)
    }

    /// [`VariantScan::read`], reading a plain file in windows of `window` bytes.
    pub(super) fn read_in_windows(
        dataset: &VcfLikeDataset,
        window: usize,
    ) -> Option<(VariantLoci, Self)> {
        let n_samples = dataset.n_samples();
        let mut builder = LociBuilder::default();
        let mut parts = Vec::with_capacity(dataset.parts().len());
        for path in dataset.parts() {
            let blocks = Blocks::open(path, window)?;
            let first_record = builder.records;
            let (index, format) = if is_bcf_path(path) {
                let header = read_variant_part_header(path).ok()?;
                let index = scan_bcf(&blocks, &header, n_samples, &mut builder)?;
                let gt_key = header.string_maps().strings().get_index_of("GT");
                (index, PartFormat::Bcf { header, gt_key })
            } else {
                let (index, header_lines) = scan_vcf(&blocks, &mut builder)?;
                (index, PartFormat::Vcf { header_lines })
            };
            parts.push(ScannedPart {
                blocks,
                index,
                format,
                first_record,
            });
        }
        let LociBuilder { loci, uneven, .. } = builder;
        Some((
            loci,
            Self {
                n_samples,
                parts,
                uneven,
            },
        ))
    }

    /// The second pass: the evidence of the rows `loci`, which ascend, each with
    /// the counter it feeds. A row that feeds none is decoded all the same, as
    /// the record reader decodes every selected row. `None` where the record
    /// reader must decode the rows.
    pub(super) fn count_evidence(
        &self,
        loci: &[(usize, Option<LocusClass>)],
        mut progress: impl FnMut(usize),
    ) -> Option<Vec<EvidenceCounts>> {
        let selected = self.selected_records(loci)?;
        let row_len = self.n_samples.div_ceil(4);
        let mut counter = EvidenceCounter::new(self.n_samples);
        let tasks: Vec<&[SelectedRecord]> = selected.chunks(RECORDS_PER_TASK).collect();
        let batch = rayon::current_num_threads().max(1) * TASKS_PER_WORKER;
        let mut counted = 0;
        for batch_tasks in tasks.chunks(batch) {
            let packed: Vec<Option<PackedRows>> = batch_tasks
                .par_iter()
                .map_init(|| TaskReader::new(self), |reader, task| reader.pack(task))
                .collect();
            for rows in packed {
                let rows = rows?;
                for class in LocusClass::ALL {
                    let class_rows: Vec<&[u8]> = rows
                        .bytes
                        .chunks_exact(row_len)
                        .zip(&rows.classes)
                        .filter(|(_, row_class)| **row_class == Some(class))
                        .map(|(row, _)| row)
                        .collect();
                    counter.add_rows(class, &class_rows);
                }
                counted += rows.classes.len();
                progress(counted);
            }
        }
        Some(counter.finish())
    }

    /// The records holding the rows `loci`, which ascend, with the ALT each row
    /// reads.
    fn selected_records(
        &self,
        loci: &[(usize, Option<LocusClass>)],
    ) -> Option<Vec<SelectedRecord>> {
        let mut selected: Vec<SelectedRecord> = Vec::new();
        let mut uneven = self.uneven.iter().peekable();
        // Rows minus records, up to the next uneven record.
        let mut offset = 0i64;
        for &(row, class) in loci {
            let row = i64::try_from(row).ok()?;
            let (record, alt) = loop {
                match uneven.peek() {
                    Some(&&(record, alts)) if row >= i64::try_from(record).ok()? + offset => {
                        let first_row = i64::try_from(record).ok()? + offset;
                        if row < first_row + i64::from(alts) {
                            break (record, usize::try_from(row - first_row).ok()? + 1);
                        }
                        offset += i64::from(alts) - 1;
                        uneven.next();
                    }
                    _ => break (u64::try_from(row - offset).ok()?, 1),
                }
            };
            let part = self
                .parts
                .partition_point(|part| part.first_record <= record)
                .checked_sub(1)?;
            let record = record - self.parts[part].first_record;
            if selected
                .last()
                .is_none_or(|last| last.part != part || last.record != record)
            {
                selected.push(SelectedRecord {
                    part,
                    record,
                    alts: Vec::new(),
                });
            }
            selected.last_mut()?.alts.push((alt, class));
        }
        Some(selected)
    }
}

/// One selected record: its file, its number within the file, and each selected
/// ALT with the counter its row feeds.
struct SelectedRecord {
    part: usize,
    record: u64,
    alts: Vec<(usize, Option<LocusClass>)>,
}

/// Packed PLINK 1 rows, `ceil(n_samples / 4)` bytes each, with each row's class.
#[derive(Default)]
struct PackedRows {
    bytes: Vec<u8>,
    classes: Vec<Option<LocusClass>>,
}

/// A second-pass task's block reader and record buffer.
struct TaskReader<'a> {
    scan: &'a VariantScan,
    cache: BlockCache,
    record: Vec<u8>,
}

impl<'a> TaskReader<'a> {
    fn new(scan: &'a VariantScan) -> Self {
        Self {
            scan,
            cache: BlockCache {
                inflater: Decompressor::new(),
                cached: None,
                block: Vec::new(),
            },
            record: Vec::new(),
        }
    }

    /// Packs the calls of every selected ALT of `task`.
    fn pack(&mut self, task: &[SelectedRecord]) -> Option<PackedRows> {
        let scan = self.scan;
        let mut rows = PackedRows::default();
        for selected in task {
            match &scan.parts[selected.part].format {
                PartFormat::Vcf { header_lines } => {
                    read_line(
                        scan,
                        &mut self.cache,
                        selected.part,
                        selected.record + header_lines,
                        &mut self.record,
                    )?;
                    pack_vcf_calls(
                        &self.record,
                        &selected.alts,
                        scan.n_samples,
                        &mut rows.bytes,
                    )?;
                }
                PartFormat::Bcf { header, gt_key } => {
                    let site_len = read_bcf_record(
                        scan,
                        &mut self.cache,
                        selected.part,
                        selected.record,
                        &mut self.record,
                    )?;
                    pack_bcf_calls(
                        &self.record,
                        site_len,
                        header,
                        *gt_key,
                        &selected.alts,
                        scan.n_samples,
                        &mut rows.bytes,
                    )?;
                }
            }
            rows.classes
                .extend(selected.alts.iter().map(|&(_, class)| class));
        }
        Some(rows)
    }
}

/// The last block a task inflated, or read from a plain file.
struct BlockCache {
    inflater: Decompressor,
    /// The file and block `block` holds.
    cached: Option<(usize, usize)>,
    block: Vec<u8>,
}

impl BlockCache {
    /// Block `index` of file `part`.
    fn read<'s>(
        &'s mut self,
        scan: &'s VariantScan,
        part: usize,
        index: usize,
    ) -> Option<&'s [u8]> {
        let blocks = &scan.parts[part].blocks;
        if index >= blocks.len() {
            return None;
        }
        if self.cached != Some((part, index)) {
            self.cached = None;
            match &blocks.frames {
                Some(frames) => inflate(
                    &blocks.map[frames[index].clone()],
                    &mut self.inflater,
                    &mut self.block,
                )?,
                None => blocks.read_plain(index, &mut self.block)?,
            }
            self.cached = Some((part, index));
        }
        Some(&self.block)
    }
}

/// Copies line `line` of VCF file `part`, without its newline, into `record`. A
/// plain file's line is one positional read, up to where the next line starts or
/// the file ends; a BGZF file's line is read out of the blocks it lies in.
fn read_line(
    scan: &VariantScan,
    cache: &mut BlockCache,
    part: usize,
    line: u64,
    record: &mut Vec<u8>,
) -> Option<()> {
    let scanned = &scan.parts[part];
    let (mut block, mut start) = scanned.index.locate(line)?;
    record.clear();
    if scanned.blocks.frames.is_none() {
        let begin = scanned.blocks.plain_offset(block, start)?;
        let end = match scanned.index.locate(line.checked_add(1)?) {
            Some((next, next_start)) => scanned.blocks.plain_offset(next, next_start)?,
            None => scanned.blocks.map.len(),
        };
        scanned.blocks.read_plain_span(begin..end, record)?;
        // The last line of a file may end without a newline.
        if record.last() == Some(&b'\n') {
            record.pop();
        }
        return Some(());
    }
    loop {
        let rest = cache.read(scan, part, block)?.get(start..)?;
        match memchr(b'\n', rest) {
            Some(end) => {
                record.extend_from_slice(&rest[..end]);
                return Some(());
            }
            None => {
                record.extend_from_slice(rest);
                block += 1;
                start = 0;
                // The last line of a file may end without a newline.
                if block == scanned.blocks.len() {
                    return Some(());
                }
            }
        }
    }
}

/// Copies record `record` of BCF file `part`, without its two lengths, into
/// `out`, and returns the length of its site block.
fn read_bcf_record(
    scan: &VariantScan,
    cache: &mut BlockCache,
    part: usize,
    record: u64,
    out: &mut Vec<u8>,
) -> Option<usize> {
    let (block, start) = scan.parts[part].index.locate(record)?;
    out.clear();
    let (block, start) = copy_span(scan, cache, part, block, start, 8, out)?;
    let (site_len, samples_len) = record_lengths(out)?;
    out.clear();
    copy_span(
        scan,
        cache,
        part,
        block,
        start,
        site_len.checked_add(samples_len)?,
        out,
    )?;
    Some(site_len)
}

/// A BCF record's site and sample block lengths, from its first eight bytes.
fn record_lengths(bytes: &[u8]) -> Option<(usize, usize)> {
    let site_len = u32::from_le_bytes(bytes.get(..4)?.try_into().ok()?);
    let samples_len = u32::from_le_bytes(bytes.get(4..8)?.try_into().ok()?);
    Some((
        usize::try_from(site_len).ok()?,
        usize::try_from(samples_len).ok()?,
    ))
}

/// Copies `len` bytes of file `part`, from offset `start` of block `block` on,
/// into `out`, and returns where they end.
fn copy_span(
    scan: &VariantScan,
    cache: &mut BlockCache,
    part: usize,
    mut block: usize,
    mut start: usize,
    mut len: usize,
    out: &mut Vec<u8>,
) -> Option<(usize, usize)> {
    loop {
        let bytes = cache.read(scan, part, block)?;
        let take = bytes.len().checked_sub(start)?.min(len);
        out.extend_from_slice(&bytes[start..start + take]);
        start += take;
        len -= take;
        if len == 0 {
            return Some((block, start));
        }
        block += 1;
        start = 0;
    }
}

/// Sets sample `sample`'s two bits of a packed row.
fn set_code(row: &mut [u8], sample: usize, code: u8) {
    row[sample / 4] |= code << (2 * (sample % 4));
}

/// Marks `samples` missing in every packed row of `packed`.
fn mark_missing(packed: &mut [u8], row_len: usize, samples: Range<usize>) {
    for row in packed.chunks_exact_mut(row_len) {
        for sample in samples.clone() {
            set_code(row, sample, MISSING_CODE);
        }
    }
}

/// Packs the GT calls of one VCF record line into one row per ALT of `alts`,
/// appended to `rows`, as `decode_vcf_record` reads them with haploid calls
/// counted as homozygous and calls read over dosages. `None` for a record it
/// would not decode from GT, or whose GT it would refuse.
fn pack_vcf_calls(
    line: &[u8],
    alts: &[(usize, Option<LocusClass>)],
    n_samples: usize,
    rows: &mut Vec<u8>,
) -> Option<()> {
    let row_len = n_samples.div_ceil(4);
    let base = rows.len();
    rows.resize(base + alts.len() * row_len, 0);
    let packed = &mut rows[base..];
    let mut sample = 0;
    // The samples follow the eighth tab. A record without any has no calls.
    if let Some(info_end) = memchr_iter(b'\t', line).nth(7) {
        let samples = &line[info_end + 1..];
        if !samples.is_empty() {
            // The FORMAT keys are the text up to the first tab, and there are
            // none without one.
            let format_end = memchr(b'\t', samples)?;
            let gt_index = samples[..format_end]
                .split(|&byte| byte == b':')
                .position(|key| key == b"GT")?;
            let rest = &samples[format_end + 1..];
            sample = if gt_index == 0 {
                pack_leading_gt(rest, alts, n_samples, packed)?
            } else {
                pack_placed_gt(rest, gt_index, alts, n_samples, packed)?
            };
        }
    }
    mark_missing(packed, row_len, sample..n_samples);
    Some(())
}

/// The calls of the sample columns `rest`, whose FORMAT leads with GT, packed
/// into one row per ALT of `alts`, one walk over the columns per ALT. Returns the
/// number of samples read.
fn pack_leading_gt(
    rest: &[u8],
    alts: &[(usize, Option<LocusClass>)],
    n_samples: usize,
    packed: &mut [u8],
) -> Option<usize> {
    let row_len = n_samples.div_ceil(4);
    let mut sample = 0;
    for (row, &(alt, _)) in packed.chunks_exact_mut(row_len).zip(alts) {
        sample = pack_leading_gt_row(rest, alt, n_samples, row)?;
    }
    Some(sample)
}

/// [`pack_leading_gt`] for ALT `alt` into its packed `row`. A column is read up to
/// the end of its GT, and a call of one digit, or of two one-digit alleles and a
/// separator, is coded without [`call_code`], as it codes one.
fn pack_leading_gt_row(
    mut rest: &[u8],
    alt: usize,
    n_samples: usize,
    row: &mut [u8],
) -> Option<usize> {
    // Whether a GT of `len` bytes ends the field there.
    let ends_at = |rest: &[u8], len: usize| {
        rest.get(len)
            .is_none_or(|&byte| byte == b'\t' || byte == b':')
    };
    let mut sample = 0;
    while !rest.is_empty() && sample < n_samples {
        let (code, gt_len) = if rest.len() >= 3
            && rest[0].is_ascii_digit()
            && matches!(rest[1], b'/' | b'|')
            && rest[2].is_ascii_digit()
            && ends_at(rest, 3)
        {
            let (first, second) = (usize::from(rest[0] - b'0'), usize::from(rest[2] - b'0'));
            let code = if (first == alt) != (second == alt) {
                HET_CODE
            } else {
                HOM_CODE
            };
            (code, 3)
        } else if rest[0].is_ascii_digit() && ends_at(rest, 1) {
            (HOM_CODE, 1)
        } else {
            let column_len = memchr(b'\t', rest).unwrap_or(rest.len());
            let gt_len = memchr(b':', &rest[..column_len]).unwrap_or(column_len);
            (call_code(&rest[..gt_len], alt)?, gt_len)
        };
        set_code(row, sample, code);
        sample += 1;
        rest = match rest.get(gt_len) {
            Some(b'\t') => &rest[gt_len + 1..],
            Some(_) => {
                memchr(b'\t', &rest[gt_len..]).map_or(&[][..], |tab| &rest[gt_len + tab + 1..])
            }
            None => &[],
        };
    }
    Some(sample)
}

/// [`pack_leading_gt`] for sample columns whose GT is the FORMAT key at
/// `gt_index`, splitting every column into its fields. A column without that
/// field has no call. Returns the number of samples read.
fn pack_placed_gt(
    mut rest: &[u8],
    gt_index: usize,
    alts: &[(usize, Option<LocusClass>)],
    n_samples: usize,
    packed: &mut [u8],
) -> Option<usize> {
    let row_len = n_samples.div_ceil(4);
    let mut sample = 0;
    while !rest.is_empty() && sample < n_samples {
        let column = match memchr(b'\t', rest) {
            Some(end) => {
                let column = &rest[..end];
                rest = &rest[end + 1..];
                column
            }
            None => std::mem::take(&mut rest),
        };
        let gt = column.split(|&byte| byte == b':').nth(gt_index);
        for (row, &(alt, _)) in packed.chunks_exact_mut(row_len).zip(alts) {
            let code = match gt {
                Some(gt) => call_code(gt, alt)?,
                None => MISSING_CODE,
            };
            set_code(row, sample, code);
        }
        sample += 1;
    }
    Some(sample)
}

/// The packed code of one VCF GT field for ALT `alt`: missing where
/// `parse_vcf_genotype` reads no call, heterozygous where a call of two or more
/// alleles carries `alt` once, homozygous otherwise, and `None` where it refuses
/// the field.
fn call_code(gt: &[u8], alt: usize) -> Option<u8> {
    let mut carried = 0usize;
    let mut alleles = 0usize;
    let mut separators = 0usize;
    let mut index = 0;
    while index < gt.len() {
        match gt[index] {
            b'/' | b'|' => {
                separators += 1;
                index += 1;
            }
            b'.' => return Some(MISSING_CODE),
            b'0'..=b'9' => {
                let mut allele = 0usize;
                while index < gt.len() && gt[index].is_ascii_digit() {
                    allele = allele
                        .checked_mul(10)?
                        .checked_add(usize::from(gt[index] - b'0'))?;
                    index += 1;
                }
                carried += usize::from(allele == alt);
                alleles += 1;
            }
            _ => return None,
        }
    }
    // A call without separators is haploid, and imports as the homozygous call.
    Some(if alleles == 0 {
        MISSING_CODE
    } else if separators > 0 && carried == 1 {
        HET_CODE
    } else {
        HOM_CODE
    })
}

/// [`pack_vcf_calls`] for a BCF record: `record` holds its site block of
/// `site_len` bytes and then its sample block, read as `decode_bcf_record` reads
/// them.
fn pack_bcf_calls(
    record: &[u8],
    site_len: usize,
    header: &vcf::Header,
    gt_key: Option<usize>,
    alts: &[(usize, Option<LocusClass>)],
    n_samples: usize,
    rows: &mut Vec<u8>,
) -> Option<()> {
    let row_len = n_samples.div_ceil(4);
    let base = rows.len();
    rows.resize(base + alts.len() * row_len, 0);
    let packed = &mut rows[base..];
    let (site, samples) = record.split_at_checked(site_len)?;
    let counts = site.get(20..24)?;
    let sample_count =
        usize::try_from(u32::from_le_bytes([counts[0], counts[1], counts[2], 0])).ok()?;
    let format_count = usize::from(counts[3]);
    if format_count == 0 {
        mark_missing(packed, row_len, 0..n_samples);
        return Some(());
    }
    series_are_readable(samples, sample_count, header)?;
    let series = GenotypeSeries::find(samples, format_count, sample_count, gt_key?).ok()??;
    pack_series_calls(&series, alts, n_samples, packed)
}

/// The calls of the GT series `series` for the first `n_samples` samples, packed
/// into one row per ALT of `alts`. Four samples whose eight 8-bit codes all name
/// alleles are coded together by [`int8_group_byte`], any other sample by
/// [`pack_series_sample`]. `None` where the per-sample path refuses a sample.
fn pack_series_calls(
    series: &GenotypeSeries<'_>,
    alts: &[(usize, Option<LocusClass>)],
    n_samples: usize,
    packed: &mut [u8],
) -> Option<()> {
    let row_len = n_samples.div_ceil(4);
    let mut carried = vec![0usize; alts.len()];
    let mut next_sample = 0;
    // A record with fewer samples than the dataset is refused sample by sample.
    if let Some(codes) = series.diploid_int8_codes()
        && series.sample_count() >= n_samples
    {
        let (groups, _) = codes.get(..n_samples / 4 * 8)?.as_chunks::<8>();
        for (group, bytes) in groups.iter().enumerate() {
            let word = u64::from_le_bytes(*bytes);
            if names_alleles(word) {
                for (row, &(alt, _)) in packed.chunks_exact_mut(row_len).zip(alts) {
                    row[group] |= int8_group_byte(word, alt);
                }
            } else {
                for sample in 4 * group..4 * group + 4 {
                    pack_series_sample(series, sample, alts, row_len, &mut carried, packed)?;
                }
            }
        }
        next_sample = groups.len() * 4;
    }
    for sample in next_sample..n_samples {
        pack_series_sample(series, sample, alts, row_len, &mut carried, packed)?;
    }
    Some(())
}

/// Packs sample `sample`'s call into every row of `packed`, as `decode_bcf_record`
/// reads it: missing where an allele is missing or the sample has none,
/// heterozygous where a call of two or more alleles carries the row's ALT once,
/// homozygous otherwise. `None` where a code names no allele, or past the
/// record's samples.
fn pack_series_sample(
    series: &GenotypeSeries<'_>,
    sample: usize,
    alts: &[(usize, Option<LocusClass>)],
    row_len: usize,
    carried: &mut [usize],
    packed: &mut [u8],
) -> Option<()> {
    carried.fill(0);
    let mut alleles = 0usize;
    let mut missing = false;
    for allele in series.alleles(sample)? {
        match allele.ok()? {
            Some(index) => {
                alleles += 1;
                for (count, &(alt, _)) in carried.iter_mut().zip(alts) {
                    *count += usize::from(index == alt);
                }
            }
            None => {
                missing = true;
                break;
            }
        }
    }
    for (row, &count) in packed.chunks_exact_mut(row_len).zip(carried.iter()) {
        let code = if missing || alleles == 0 {
            MISSING_CODE
        } else if alleles > 1 && count == 1 {
            HET_CODE
        } else {
            HOM_CODE
        };
        set_code(row, sample, code);
    }
    Some(())
}

const HIGH_BITS: u64 = 0x8080_8080_8080_8080;
const LOW_SEVEN_BITS: u64 = 0x7f7f_7f7f_7f7f_7f7f;

/// Whether each of the eight 8-bit GT codes in `word` names an allele: a code of
/// 2 to 127, which [`GenotypeSeries::alleles`] reads as allele `(code >> 1) - 1`.
/// Missing codes (0, 1 and -128), the end-of-vector code (-127) and other
/// negative codes do not.
fn names_alleles(word: u64) -> bool {
    // With the sign bits clear, a code below 2 has none of the bits 0x7e.
    let upper = word & 0x7e7e_7e7e_7e7e_7e7e;
    // Adding 0x7f to a byte of at most 0x7e carries into its top bit exactly when
    // the byte is not zero, and never into the next byte.
    word & HIGH_BITS == 0 && ((upper + LOW_SEVEN_BITS) | upper) & HIGH_BITS == HIGH_BITS
}

/// The packed byte of ALT `alt`'s row for four samples of two 8-bit codes each,
/// all of which name alleles ([`names_alleles`]): heterozygous where exactly one
/// of a sample's two alleles is `alt`, homozygous otherwise.
fn int8_group_byte(word: u64, alt: usize) -> u8 {
    const HOM_BYTE: u8 = HOM_CODE * 0b0101_0101;
    // An 8-bit code names at most allele 62, so no allele is a higher ALT.
    if alt > 62 {
        return HOM_BYTE;
    }
    let target = ((alt as u64 + 1) << 1) * 0x0101_0101_0101_0101;
    // Zero bytes where a code, phase bit dropped, names `alt`. Both sides are at
    // most 0x7e, so the zero test below is exact per byte.
    let differ = (word & 0xfefe_fefe_fefe_fefe) ^ target;
    let equal = !((differ + LOW_SEVEN_BITS) | differ) & HIGH_BITS;
    // A sample's first code is an even byte and its second the odd byte above.
    let het = (equal ^ (equal >> 8)) & 0x0080_0080_0080_0080;
    let het_samples = ((het >> 7) & 1)
        | ((het >> 21) & 0b100)
        | ((het >> 35) & 0b1_0000)
        | ((het >> 49) & 0b100_0000);
    // A heterozygous sample's code differs from the homozygous code in its low bit.
    HOM_BYTE ^ (het_samples as u8 * (HOM_CODE ^ HET_CODE))
}

/// Walks the FORMAT series of a BCF sample block as the record reader does, to
/// the end of the block, and checks that each one names a header string. Bytes
/// past the last series that do not read as one make the record reader refuse
/// the record, and so the pass.
fn series_are_readable(mut src: &[u8], sample_count: usize, header: &vcf::Header) -> Option<()> {
    while !src.is_empty() {
        let key = usize::try_from(read_typed_int(&mut src)?).ok()?;
        header.string_maps().strings().get_index(key)?;
        let (value_type, len) = read_descriptor(&mut src)?;
        let unit: usize = match value_type {
            1 | 7 => 1,
            2 => 2,
            3 | 5 => 4,
            _ => return None,
        };
        take(&mut src, unit.checked_mul(len)?.checked_mul(sample_count)?)?;
    }
    Some(())
}

/// Where the items of a file (its lines, or its records) start, block by block.
#[derive(Default)]
struct BlockIndex {
    /// The items starting ahead of each block.
    before: Vec<u64>,
    /// Where each item starts, in the block it starts in.
    starts: Vec<u32>,
}

impl BlockIndex {
    fn begin_block(&mut self, items: u64) {
        self.before.push(items);
    }

    /// Notes an item starting at `offset` of the current block.
    fn start(&mut self, offset: usize) {
        self.starts.push(offset as u32);
    }

    /// The block item `item` starts in, and where in that block it starts.
    fn locate(&self, item: u64) -> Option<(usize, usize)> {
        let start = *self.starts.get(usize::try_from(item).ok()?)?;
        let block = self
            .before
            .partition_point(|&before| before <= item)
            .checked_sub(1)?;
        Some((block, start as usize))
    }
}

/// The loci of every row, gathered record by record across a dataset's files.
#[derive(Default)]
struct LociBuilder {
    loci: VariantLoci,
    label: String,
    class: Option<LocusChromosome>,
    order: ChromPositionSortState,
    records: u64,
    uneven: Vec<(u64, u32)>,
}

impl LociBuilder {
    /// Adds one record, or refuses it where the key scan refuses it: out of
    /// position order within its chromosome.
    fn record(&mut self, label: &str, position: u64, alts: usize) -> Option<()> {
        if self.records == 0 || label != self.label {
            self.label.clear();
            self.label.push_str(label);
            self.class = classify_chromosome(&VariantKey::new(label, position).chromosome);
        }
        self.records += 1;
        self.order
            .observe(label, position, usize::try_from(self.records).ok()?)
            .ok()?;
        if alts != 1 {
            self.uneven
                .push((self.records - 1, u32::try_from(alts).ok()?));
        }
        for _ in 0..alts {
            self.loci.chroms.push(self.class);
            self.loci.positions.push(position);
        }
        Some(())
    }
}

/// Whether the record reader reads `path` as BCF, which it decides by the name.
fn is_bcf_path(path: &Path) -> bool {
    path.to_string_lossy()
        .to_ascii_lowercase()
        .ends_with(".bcf")
}

/// A local file's uncompressed bytes, as blocks that read independently.
struct Blocks {
    /// Where there is a positional read, the file it reads plain windows from.
    #[cfg(unix)]
    file: File,
    map: Mmap,
    /// The frame of every BGZF block in file order, or `None` for a plain file,
    /// read in windows of `window` bytes.
    frames: Option<Vec<Range<usize>>>,
    window: usize,
}

impl Blocks {
    fn open(path: &Path, window: usize) -> Option<Self> {
        let file = File::open(path).ok()?;
        // SAFETY: the map is read-only and lives as long as the scan. As with a
        // `.bed`, the file must not be truncated while it is mapped.
        let map = unsafe { Mmap::map(&file) }.ok()?;
        // `open_local_variant_reader` reads a file that starts with the gzip
        // magic number as BGZF.
        let frames = if map.starts_with(&[0x1f, 0x8b]) {
            Some(bgzf_frames(&map)?)
        } else {
            None
        };
        Some(Self {
            #[cfg(unix)]
            file,
            map,
            frames,
            window: window.max(1),
        })
    }

    fn len(&self) -> usize {
        match &self.frames {
            Some(frames) => frames.len(),
            None => self.map.len().div_ceil(self.window),
        }
    }

    /// The bytes of window `index` of a plain file.
    fn plain_range(&self, index: usize) -> Option<Range<usize>> {
        let start = index.checked_mul(self.window)?;
        (start < self.map.len())
            .then(|| start..start.saturating_add(self.window).min(self.map.len()))
    }

    /// Where offset `start` of window `index` of a plain file lies in the file.
    fn plain_offset(&self, index: usize, start: usize) -> Option<usize> {
        index.checked_mul(self.window)?.checked_add(start)
    }

    /// Reads window `index` of a plain file into `buffer`.
    fn read_plain(&self, index: usize, buffer: &mut Vec<u8>) -> Option<()> {
        self.read_plain_span(self.plain_range(index)?, buffer)
    }

    /// Reads bytes `range` of a plain file into `buffer` with one positional read,
    /// so the text is never faulted into the map: faults on one map serialize the
    /// threads that take them.
    #[cfg(unix)]
    fn read_plain_span(&self, range: Range<usize>, buffer: &mut Vec<u8>) -> Option<()> {
        use std::os::unix::fs::FileExt;

        if range.start > range.end || range.end > self.map.len() {
            return None;
        }
        buffer.resize(range.len(), 0);
        self.file
            .read_exact_at(buffer, u64::try_from(range.start).ok()?)
            .ok()
    }

    /// [`Blocks::read_plain_span`] where there is no positional read: a copy out of
    /// the map.
    #[cfg(not(unix))]
    fn read_plain_span(&self, range: Range<usize>, buffer: &mut Vec<u8>) -> Option<()> {
        buffer.clear();
        buffer.extend_from_slice(self.map.get(range)?);
        Some(())
    }
}

/// The frame of every block of a BGZF file, or `None` when a frame is not one
/// the BGZF reader reads.
fn bgzf_frames(file: &[u8]) -> Option<Vec<Range<usize>>> {
    let mut frames = Vec::with_capacity(file.len() / 16_000 + 1);
    let mut start = 0;
    while start < file.len() {
        let header = file.get(start..start + BGZF_HEADER_LEN)?;
        if !is_bgzf_header(header) {
            return None;
        }
        let end = start + usize::from(u16::from_le_bytes([header[16], header[17]])) + 1;
        if end < start + BGZF_HEADER_LEN + BGZF_TRAILER_LEN || end > file.len() {
            return None;
        }
        frames.push(start..end);
        start = end;
    }
    Some(frames)
}

fn is_bgzf_header(header: &[u8]) -> bool {
    header[..4] == [0x1f, 0x8b, 0x08, 0x04]
        && header[10..16] == [0x06, 0x00, b'B', b'C', 0x02, 0x00]
}

/// Inflates one BGZF frame into `block`, checking its length and CRC32.
fn inflate(frame: &[u8], inflater: &mut Decompressor, block: &mut Vec<u8>) -> Option<()> {
    let (body, trailer) = frame.split_at(frame.len() - BGZF_TRAILER_LEN);
    let crc32 = u32::from_le_bytes(trailer[..4].try_into().ok()?);
    let len = usize::try_from(u32::from_le_bytes(trailer[4..].try_into().ok()?)).ok()?;
    if len > BGZF_MAX_BLOCK_LEN {
        return None;
    }
    block.resize(len, 0);
    let written = inflater
        .deflate_decompress(&body[BGZF_HEADER_LEN..], block)
        .ok()?;
    let mut crc = Crc::new();
    crc.update(block);
    (written == len && crc.sum() == crc32).then_some(())
}

/// Reads blocks `range` in parallel, running `scan` over each block's bytes with
/// a state of its own, and returns the blocks' bytes in order: a BGZF block
/// inflated into its buffer, a plain window read into its buffer. `None` when a
/// block does not read or `scan` refuses one.
fn read_window<'a, S: Default + Send>(
    blocks: &'a Blocks,
    range: Range<usize>,
    buffers: &'a mut Vec<Vec<u8>>,
    states: &mut Vec<S>,
    scan: impl Fn(&[u8], &mut S) -> bool + Sync,
) -> Option<Vec<&'a [u8]>> {
    let n = range.len();
    if buffers.len() < n {
        buffers.resize_with(n, Vec::new);
    }
    states.resize_with(n, S::default);
    let read = match &blocks.frames {
        Some(frames) => buffers[..n]
            .par_iter_mut()
            .zip(states.par_iter_mut())
            .zip(&frames[range.clone()])
            .map_init(Decompressor::new, |inflater, ((buffer, state), frame)| {
                inflate(&blocks.map[frame.clone()], inflater, buffer).is_some()
                    && scan(buffer, state)
            })
            .all(|read| read),
        None => buffers[..n]
            .par_iter_mut()
            .zip(states.par_iter_mut())
            .zip(range)
            .all(|((buffer, state), index)| {
                blocks.read_plain(index, buffer).is_some() && scan(buffer, state)
            }),
    };
    if !read {
        return None;
    }
    let buffers: &'a Vec<Vec<u8>> = buffers;
    Some(buffers[..n].iter().map(Vec::as_slice).collect())
}

/// The first pass over one VCF file: its records' loci, and the index of its
/// lines with the number of header lines ahead of the records.
fn scan_vcf(blocks: &Blocks, builder: &mut LociBuilder) -> Option<(BlockIndex, u64)> {
    let batch = rayon::current_num_threads().max(1) * BLOCKS_PER_WORKER;
    let mut buffers = Vec::new();
    let mut scanned: Vec<TextBlock> = Vec::new();
    let mut walker = TextWalker::default();
    let mut index = BlockIndex::default();
    for start in (0..blocks.len()).step_by(batch) {
        let range = start..(start + batch).min(blocks.len());
        let views = read_window(blocks, range, &mut buffers, &mut scanned, scan_text_block)?;
        for (block, text) in views.iter().zip(&scanned) {
            walker.walk(block, text, &mut index, builder)?;
        }
    }
    walker.finish(builder)?;
    Some((index, walker.header_lines))
}

/// One block of VCF text as a worker of the first pass reads it.
#[derive(Default)]
struct TextBlock {
    /// Where each line end of the block is.
    newlines: Vec<u32>,
    /// Each line between two of the block's line ends, in order: every line but
    /// the ones that end at its first line end or continue past its last.
    lines: Vec<TextLine>,
}

/// What the first pass reads from one line of VCF text: the fields of a record
/// up to its ALT, where the line holds them.
#[derive(Clone, Debug, PartialEq, Eq)]
enum TextLine {
    /// A line starting with '#', and whether it is the column header.
    Hash { column_header: bool },
    /// A record's CHROM, at bytes `label` of the line, its POS and its ALT count.
    Record {
        label: Range<usize>,
        position: u64,
        alts: usize,
    },
    /// A line that is neither: fewer than eight fields, or a POS the key scan
    /// does not read.
    Refused,
}

/// Reads `line`, or its first seven tabs and what comes before them.
fn summarize(line: &[u8]) -> TextLine {
    if line.first() == Some(&b'#') {
        return TextLine::Hash {
            column_header: line.starts_with(b"#CHROM"),
        };
    }
    let mut tabs = [0usize; 7];
    let mut count = 0;
    for (offset, &byte) in line.iter().enumerate() {
        if byte == b'\t' {
            tabs[count] = offset;
            count += 1;
            if count == 7 {
                break;
            }
        }
    }
    if count < 7 {
        return TextLine::Refused;
    }
    // The key scan reads CHROM, POS, ID, REF and then ALT, the fifth field.
    match parse_position(&line[tabs[0] + 1..tabs[1]]) {
        Some(position) => TextLine::Record {
            label: 0..tabs[0],
            position,
            alts: alt_count(&line[tabs[3] + 1..tabs[4]]),
        },
        None => TextLine::Refused,
    }
}

/// Finds the line ends of one block of VCF text and reads each line that lies
/// between two of them, refusing a block that is not ASCII or holds a carriage
/// return, which the record reader strips from some fields and not others.
fn scan_text_block(block: &[u8], text: &mut TextBlock) -> bool {
    text.newlines.clear();
    text.lines.clear();
    if !is_plain_ascii(block) {
        return false;
    }
    text.newlines
        .extend(memchr_iter(b'\n', block).map(|end| end as u32));
    text.lines.extend(
        text.newlines
            .windows(2)
            .map(|ends| summarize(&block[ends[0] as usize + 1..ends[1] as usize])),
    );
    true
}

/// True when `text` is ASCII and holds no carriage return.
fn is_plain_ascii(text: &[u8]) -> bool {
    let (chunks, tail) = text.as_chunks::<64>();
    chunks.iter().all(|chunk| {
        chunk.iter().fold(0u8, |flags, &byte| {
            flags | (byte & 0x80) | u8::from(byte == b'\r')
        }) == 0
    }) && tail.iter().all(|&byte| byte < 0x80 && byte != b'\r')
}

/// The first pass over one VCF file's text: header lines, then one record per
/// line.
#[derive(Default)]
struct TextWalker {
    /// Lines started so far, header lines included.
    lines: u64,
    header_lines: u64,
    /// Whether the column header line has been read.
    past_header: bool,
    /// Whether a line has started and not ended.
    in_line: bool,
    /// The current line up to its seventh tab, or all of it while it has fewer.
    prefix: Vec<u8>,
    tabs: usize,
}

impl TextWalker {
    fn walk(
        &mut self,
        block: &[u8],
        text: &TextBlock,
        index: &mut BlockIndex,
        builder: &mut LociBuilder,
    ) -> Option<()> {
        index.begin_block(self.lines);
        let Some(&first_end) = text.newlines.first() else {
            // No line ends in the block, which starts a line or continues one.
            if !block.is_empty() {
                self.start_line(0, index);
                self.take_prefix(block);
            }
            return Some(());
        };
        // The line that ends at the block's first line end may have started in an
        // earlier block.
        self.start_line(0, index);
        self.take_prefix(&block[..first_end as usize]);
        self.end_line(builder)?;
        for (line, &end) in text.lines.iter().zip(&text.newlines) {
            let start = end as usize + 1;
            index.start(start);
            self.lines += 1;
            self.read_line(line, &block[start..], builder)?;
        }
        let tail = *text.newlines.last()? as usize + 1;
        if tail < block.len() {
            self.start_line(tail, index);
            self.take_prefix(&block[tail..]);
        }
        Some(())
    }

    /// Starts a line at `offset` of the current block, unless one is under way.
    fn start_line(&mut self, offset: usize, index: &mut BlockIndex) {
        if !self.in_line {
            index.start(offset);
            self.lines += 1;
            self.in_line = true;
            self.prefix.clear();
            self.tabs = 0;
        }
    }

    /// Reads a line that has ended, as [`summarize`] read it from `bytes`, which
    /// start with the line.
    fn read_line(
        &mut self,
        line: &TextLine,
        bytes: &[u8],
        builder: &mut LociBuilder,
    ) -> Option<()> {
        match line {
            // The header is every line up to the column header, all starting with
            // '#'.
            TextLine::Hash { column_header } if !self.past_header => {
                self.header_lines += 1;
                self.past_header = *column_header;
                Some(())
            }
            // The record reader needs eight fields, and reads a '#' line after the
            // header as a record.
            TextLine::Record {
                label,
                position,
                alts,
            } if self.past_header => builder.record(
                str::from_utf8(&bytes[label.clone()]).ok()?,
                *position,
                *alts,
            ),
            _ => None,
        }
    }

    /// Adds a piece of the current line to its prefix, up to the seventh tab.
    fn take_prefix(&mut self, piece: &[u8]) {
        if self.tabs == 7 {
            return;
        }
        match memchr_iter(b'\t', piece).nth(6 - self.tabs) {
            Some(tab) => {
                self.prefix.extend_from_slice(&piece[..=tab]);
                self.tabs = 7;
            }
            None => {
                self.tabs += memchr_iter(b'\t', piece).count();
                self.prefix.extend_from_slice(piece);
            }
        }
    }

    /// Reads the line that just ended, from its prefix.
    fn end_line(&mut self, builder: &mut LociBuilder) -> Option<()> {
        self.in_line = false;
        let prefix = std::mem::take(&mut self.prefix);
        let read = self.read_line(&summarize(&prefix), &prefix, builder);
        self.prefix = prefix;
        read
    }

    /// Ends the file, whose last line may lack a newline.
    fn finish(&mut self, builder: &mut LociBuilder) -> Option<()> {
        if self.in_line {
            self.end_line(builder)?;
        }
        self.past_header.then_some(())
    }
}

/// A VCF POS as the key scan reads it, where it is a positive integer written in
/// plain digits. Other spellings go to the key scan.
fn parse_position(field: &[u8]) -> Option<u64> {
    if field.is_empty() || field.len() > 19 || !field.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let position = field
        .iter()
        .fold(0u64, |value, &digit| value * 10 + u64::from(digit - b'0'));
    (position > 0).then_some(position)
}

/// The ALT count of a VCF ALT field as the record reader iterates it: none for
/// `.` or an empty field, else one per comma-separated allele.
fn alt_count(field: &[u8]) -> usize {
    match field {
        b"" | b"." => 0,
        alleles => memchr_iter(b',', alleles).count() + 1,
    }
}

/// The first pass over one BCF file: its records' loci, and the index of its
/// records.
fn scan_bcf(
    blocks: &Blocks,
    header: &vcf::Header,
    n_samples: usize,
    builder: &mut LociBuilder,
) -> Option<BlockIndex> {
    let batch = rayon::current_num_threads().max(1) * BLOCKS_PER_WORKER;
    let mut buffers = Vec::new();
    let mut states: Vec<()> = Vec::new();
    let mut walker = BcfWalker {
        state: BcfState::Magic,
        pending: Vec::new(),
        records: 0,
    };
    let mut index = BlockIndex::default();
    for start in (0..blocks.len()).step_by(batch) {
        let range = start..(start + batch).min(blocks.len());
        for block in read_window(blocks, range, &mut buffers, &mut states, |_, _| true)? {
            walker.walk(block, header, n_samples, &mut index, builder)?;
        }
    }
    walker.finish()?;
    Some(index)
}

/// The first pass over one BCF file's bytes: the magic number and header text,
/// then records.
struct BcfWalker {
    state: BcfState,
    /// The part of the current fixed-length field gathered so far.
    pending: Vec<u8>,
    /// Records started so far.
    records: u64,
}

#[derive(Clone, Copy)]
enum BcfState {
    /// The magic number, the version and the header text length.
    Magic,
    /// Header text bytes still to skip.
    Text(usize),
    /// A record's site and sample block lengths.
    Lengths,
    /// A record's site block, and the length of the sample block that follows it.
    Site { site_len: usize, samples_len: usize },
    /// Sample block bytes still to skip.
    Samples(usize),
    /// A zero site length, which the record reader reads as the end of the file.
    End,
}

impl BcfWalker {
    fn walk(
        &mut self,
        block: &[u8],
        header: &vcf::Header,
        n_samples: usize,
        index: &mut BlockIndex,
        builder: &mut LociBuilder,
    ) -> Option<()> {
        index.begin_block(self.records);
        let mut position = 0;
        while position < block.len() {
            match self.state {
                BcfState::Magic => {
                    if gather(&mut self.pending, 9, block, &mut position) {
                        let magic = &self.pending;
                        if &magic[..3] != b"BCF" || magic[3] != 2 || !matches!(magic[4], 1 | 2) {
                            return None;
                        }
                        let text_len = u32::from_le_bytes(magic[5..9].try_into().ok()?);
                        self.state = BcfState::Text(usize::try_from(text_len).ok()?);
                        self.pending.clear();
                    }
                }
                BcfState::Text(remaining) => {
                    let skip = remaining.min(block.len() - position);
                    position += skip;
                    self.state = if skip == remaining {
                        BcfState::Lengths
                    } else {
                        BcfState::Text(remaining - skip)
                    };
                }
                BcfState::Samples(remaining) => {
                    let skip = remaining.min(block.len() - position);
                    position += skip;
                    self.state = if skip == remaining {
                        BcfState::Lengths
                    } else {
                        BcfState::Samples(remaining - skip)
                    };
                }
                BcfState::Lengths => {
                    if self.pending.is_empty() {
                        index.start(position);
                        self.records += 1;
                    }
                    if gather(&mut self.pending, 8, block, &mut position) {
                        let (site_len, samples_len) = record_lengths(&self.pending)?;
                        self.pending.clear();
                        self.state = if site_len == 0 {
                            BcfState::End
                        } else {
                            BcfState::Site {
                                site_len,
                                samples_len,
                            }
                        };
                    }
                }
                BcfState::Site {
                    site_len,
                    samples_len,
                } => {
                    if gather(&mut self.pending, site_len, block, &mut position) {
                        let (label, locus, alts) = read_site(&self.pending, header, n_samples)?;
                        builder.record(label, locus, alts)?;
                        self.pending.clear();
                        self.state = BcfState::Samples(samples_len);
                    }
                }
                BcfState::End => return Some(()),
            }
        }
        Some(())
    }

    /// Ends the file, which must end between records.
    fn finish(&self) -> Option<()> {
        match self.state {
            BcfState::Lengths => self.pending.is_empty().then_some(()),
            BcfState::Text(0) | BcfState::Samples(0) | BcfState::End => Some(()),
            _ => None,
        }
    }
}

/// Moves bytes of `block` from `position` into `pending` until it holds `len`,
/// and says whether it does.
fn gather(pending: &mut Vec<u8>, len: usize, block: &[u8], position: &mut usize) -> bool {
    let take = (len - pending.len()).min(block.len() - *position);
    pending.extend_from_slice(&block[*position..*position + take]);
    *position += take;
    pending.len() == len
}

/// A BCF record's chromosome, position and ALT count, read from its site block as
/// the key scan reads them, and `None` wherever the scan or the record reader's
/// indexing of the block would refuse it.
fn read_site<'h>(
    site: &[u8],
    header: &'h vcf::Header,
    n_samples: usize,
) -> Option<(&'h str, u64, usize)> {
    let int = |at: usize| -> Option<i32> {
        Some(i32::from_le_bytes(site.get(at..at + 4)?.try_into().ok()?))
    };
    let contig = usize::try_from(int(0)?).ok()?;
    let label = header.string_maps().contigs().get_index(contig)?;
    // A 0-based position of -1 has no 1-based position.
    let position = u64::try_from(int(4)?).ok()? + 1;
    let counts = site.get(16..24)?;
    let alleles = u16::from_le_bytes([counts[2], counts[3]]);
    let sample_count = u32::from_le_bytes([counts[4], counts[5], counts[6], 0]);
    if alleles == 0 || usize::try_from(sample_count).ok()? != n_samples {
        return None;
    }
    let mut src = &site[24..];
    // ID, then REF, then each ALT, which must be a non-empty UTF-8 string.
    take_string(&mut src)?;
    take_string(&mut src)?;
    for _ in 1..alleles {
        let alt = take_string(&mut src)?;
        if alt.is_empty() || str::from_utf8(alt).is_err() {
            return None;
        }
    }
    // FILTER, the last field the record reader indexes.
    let (value_type, len) = read_descriptor(&mut src)?;
    let unit: usize = match value_type {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => return None,
    };
    take(&mut src, unit.checked_mul(len)?)?;
    Some((label, position, usize::from(alleles) - 1))
}

fn take<'a>(src: &mut &'a [u8], len: usize) -> Option<&'a [u8]> {
    let (head, tail) = src.split_at_checked(len)?;
    *src = tail;
    Some(head)
}

/// A typed BCF string.
fn take_string<'a>(src: &mut &'a [u8]) -> Option<&'a [u8]> {
    let (7, len) = read_descriptor(src)? else {
        return None;
    };
    take(src, len)
}

/// A typed-value descriptor: its type, and its length, which a length nibble of
/// 15 says follows as a typed integer.
fn read_descriptor(src: &mut &[u8]) -> Option<(u8, usize)> {
    let byte = take(src, 1)?[0];
    let len = match byte >> 4 {
        15 => usize::try_from(read_typed_int(src)?).ok()?,
        len => usize::from(len),
    };
    Some((byte & 0x0f, len))
}

/// A single typed integer, as series keys and long lengths are written.
fn read_typed_int(src: &mut &[u8]) -> Option<i64> {
    let byte = take(src, 1)?[0];
    if byte >> 4 != 1 {
        return None;
    }
    Some(match byte & 0x0f {
        1 => i64::from(take(src, 1)?[0] as i8),
        2 => i64::from(i16::from_le_bytes(take(src, 2)?.try_into().ok()?)),
        3 => i64::from(i32::from_le_bytes(take(src, 4)?.try_into().ok()?)),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_codes_follow_the_genotype_parser() {
        let cases: [(&str, usize, Option<u8>); 22] = [
            ("0/0", 1, Some(HOM_CODE)),
            ("0/1", 1, Some(HET_CODE)),
            ("1|0", 1, Some(HET_CODE)),
            ("1/1", 1, Some(HOM_CODE)),
            ("1", 1, Some(HOM_CODE)),
            ("0", 1, Some(HOM_CODE)),
            ("./.", 1, Some(MISSING_CODE)),
            (".", 1, Some(MISSING_CODE)),
            ("./1", 1, Some(MISSING_CODE)),
            ("1/.", 1, Some(MISSING_CODE)),
            ("", 1, Some(MISSING_CODE)),
            ("/", 1, Some(MISSING_CODE)),
            ("0/1/1", 1, Some(HOM_CODE)),
            ("0/0/1", 1, Some(HET_CODE)),
            ("1/2", 2, Some(HET_CODE)),
            ("2/2", 1, Some(HOM_CODE)),
            ("1/01", 1, Some(HOM_CODE)),
            ("0/x", 1, None),
            ("./x", 1, Some(MISSING_CODE)),
            ("x/.", 1, None),
            ("0/99999999999999999999", 1, None),
            // Two parts, one of them empty: a diploid call to the ploidy count.
            ("1|", 1, Some(HET_CODE)),
        ];
        for (gt, alt, expected) in cases {
            assert_eq!(
                call_code(gt.as_bytes(), alt),
                expected,
                "{gt} for ALT {alt}"
            );
        }
    }

    #[test]
    fn alt_counts_and_positions_read_as_the_key_scan_reads_them() {
        assert_eq!(alt_count(b"."), 0);
        assert_eq!(alt_count(b""), 0);
        assert_eq!(alt_count(b"G"), 1);
        assert_eq!(alt_count(b"C,T"), 2);
        assert_eq!(alt_count(b"C,."), 2);
        assert_eq!(parse_position(b"1"), Some(1));
        assert_eq!(parse_position(b"0155270561"), Some(155_270_561));
        for refused in [
            &b"0"[..],
            b"00",
            b"+5",
            b"-5",
            b"5x",
            b"",
            b"99999999999999999999",
        ] {
            assert_eq!(parse_position(refused), None, "{refused:?}");
        }
    }

    #[test]
    fn typed_values_read_as_the_record_reader_reads_them() {
        // A long length: descriptor 0xF7, then the length as an int8.
        let mut src: &[u8] = &[
            0xf7, 0x11, 16, b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T', b'A', b'C', b'G', b'T',
            b'A', b'C', b'G', b'T', 0x21,
        ];
        assert_eq!(take_string(&mut src).map(<[u8]>::len), Some(16));
        assert_eq!(src, &[0x21]);
        // A negative long length, a float descriptor where a string is due, and
        // a truncated string are all refused.
        assert_eq!(take_string(&mut &[0xf7, 0x11, 0xff][..]), None);
        assert_eq!(take_string(&mut &[0x15, 0, 0, 0, 0][..]), None);
        assert_eq!(take_string(&mut &[0x37, b'A'][..]), None);
        assert_eq!(read_typed_int(&mut &[0x12, 0x34, 0x12][..]), Some(0x1234));
        assert_eq!(read_typed_int(&mut &[0x21, 1, 2][..]), None);
    }

    #[test]
    fn plain_ascii_refuses_high_bytes_and_carriage_returns_anywhere() {
        let text = vec![b'A'; 200];
        assert!(is_plain_ascii(&text));
        for at in [0usize, 63, 64, 130, 199] {
            for byte in [0x80u8, 0xff, b'\r'] {
                let mut dirty = text.clone();
                dirty[at] = byte;
                assert!(!is_plain_ascii(&dirty), "{byte:#x} at {at}");
            }
        }
    }

    #[test]
    fn block_index_locates_items_in_the_block_they_start_in() {
        let mut index = BlockIndex::default();
        // Block 0 starts items 0 and 1, block 1 none, block 2 items 2 to 4.
        index.begin_block(0);
        index.start(0);
        index.start(10);
        index.begin_block(2);
        index.begin_block(2);
        index.start(7);
        index.start(30);
        index.start(61);
        index.begin_block(5);
        assert_eq!(index.locate(0), Some((0, 0)));
        assert_eq!(index.locate(1), Some((0, 10)));
        assert_eq!(index.locate(2), Some((2, 7)));
        assert_eq!(index.locate(3), Some((2, 30)));
        assert_eq!(index.locate(4), Some((2, 61)));
        assert_eq!(index.locate(5), None);
    }

    /// A FORMAT that leads with GT is read byte by byte, and any other through its
    /// fields: both must pack the rows `call_code` gives, whatever else a column
    /// holds and however the line ends.
    #[test]
    fn leading_and_placed_gt_pack_the_codes_call_code_gives() {
        let calls = [
            "0/0", "0/1", "1|0", "1/1", "1", "0", "2", "./.", ".", "./1", "1/.", "", "/", "0/1/1",
            "0/0/1", "1/2", "2|2", "1/01", "10/1", "1/10", "10", "./x", "1|",
        ];
        let alts = [(1, None), (2, None)];
        // One sample more than the lines hold, which has no call.
        let n_samples = calls.len() + 1;
        let row_len = n_samples.div_ceil(4);
        let mut expected = vec![0u8; alts.len() * row_len];
        for (sample, gt) in calls.iter().enumerate() {
            for (row, &(alt, _)) in expected.chunks_exact_mut(row_len).zip(&alts) {
                set_code(row, sample, call_code(gt.as_bytes(), alt).unwrap());
            }
        }
        mark_missing(&mut expected, row_len, calls.len()..n_samples);
        let prefix = "1\t100\t.\tA\tC,T\t.\tPASS\t.\t";
        let column = |format: &str, gt: &str| match format {
            "GT" => gt.to_string(),
            "GT:DS" => format!("{gt}:0.5,0.1"),
            _ => format!("0.5,0.1:{gt}"),
        };
        for format in ["GT", "GT:DS", "DS:GT"] {
            let columns: Vec<String> = calls.iter().map(|gt| column(format, gt)).collect();
            for ending in ["", "\t"] {
                let line = format!("{prefix}{format}\t{}{ending}", columns.join("\t"));
                let mut rows = vec![0xff];
                pack_vcf_calls(line.as_bytes(), &alts, n_samples, &mut rows).unwrap();
                assert_eq!(rows[0], 0xff);
                assert_eq!(&rows[1..], &expected[..], "{format}, ending {ending:?}");
            }
            // A GT the genotype parser refuses refuses the record.
            let refused = format!(
                "{prefix}{format}\t{}\t{}",
                columns.join("\t"),
                column(format, "0/x")
            );
            assert!(
                pack_vcf_calls(refused.as_bytes(), &alts, n_samples, &mut Vec::new()).is_none()
            );
        }
    }

    /// Four samples of 8-bit codes are packed together only where every code names
    /// an allele, and must give the rows the per-sample path gives: for every code
    /// at every position of a group, for missing, end-of-vector and invalid codes
    /// planted anywhere, and for random series, at every sample count.
    #[test]
    fn int8_groups_pack_what_the_per_sample_path_packs() {
        const GT_KEY: u8 = 5;
        for at in 0..8 {
            for code in 0..=255u8 {
                let mut bytes = [4u8; 8];
                bytes[at] = code;
                assert_eq!(
                    names_alleles(u64::from_le_bytes(bytes)),
                    (2..=127).contains(&code),
                    "{code:#x} at {at}"
                );
            }
        }
        let wide_alts = [
            (0, None),
            (1, None),
            (2, None),
            (30, None),
            (61, None),
            (62, None),
            (63, None),
            (200, None),
        ];
        let alts = [(1, None), (2, None), (62, None), (63, None)];
        let mut cases: Vec<(Vec<u8>, &[(usize, Option<LocusClass>)])> = Vec::new();
        for at in 0..8 {
            for code in 2..=127u8 {
                let mut codes = vec![2u8, 4, 6, 8, 3, 5, 0x7e, 0x7f];
                codes[at] = code;
                cases.push((codes, &wide_alts));
            }
        }
        // Alleles 0, 1, 2 and 62, unphased and phased; missing (0, 1 and -128); the
        // end of a vector (-127); and two negative codes that name nothing.
        let pool = [
            2u8, 3, 4, 5, 6, 7, 0x7e, 0x7f, 0x00, 0x01, 0x80, 0x81, 0xfe, 0x90,
        ];
        let calls = [2u8, 4, 4, 2, 3, 5, 2, 2, 4, 6, 5, 3, 7, 4, 0x7e, 4];
        for at in 0..calls.len() {
            for &code in &pool {
                let mut codes = calls.to_vec();
                codes[at] = code;
                cases.push((codes, &alts));
            }
        }
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..500 {
            let n = 1 + (next() % 23) as usize;
            // Mostly allele codes, as a cohort holds them.
            let codes = (0..2 * n)
                .map(|_| {
                    let draw = next();
                    let kinds = if draw % 16 == 0 { pool.len() } else { 8 };
                    pool[(draw >> 8) as usize % kinds]
                })
                .collect();
            cases.push((codes, &alts));
        }
        for (codes, alts) in &cases {
            let sample_count = codes.len() / 2;
            // One series: its key as an int8, then an int8 vector of two per sample.
            let mut block = vec![0x11, GT_KEY, 0x21];
            block.extend_from_slice(codes);
            let series = GenotypeSeries::find(&block, 1, sample_count, usize::from(GT_KEY))
                .unwrap()
                .unwrap();
            for n_samples in 1..=sample_count + 1 {
                let row_len = n_samples.div_ceil(4);
                let mut expected = vec![0u8; alts.len() * row_len];
                let mut carried = vec![0; alts.len()];
                let per_sample = (0..n_samples).try_for_each(|sample| {
                    pack_series_sample(&series, sample, alts, row_len, &mut carried, &mut expected)
                });
                let mut packed = vec![0u8; alts.len() * row_len];
                let grouped = pack_series_calls(&series, alts, n_samples, &mut packed);
                assert_eq!(grouped, per_sample, "{codes:x?} over {n_samples} samples");
                if per_sample.is_some() {
                    assert_eq!(packed, expected, "{codes:x?} over {n_samples} samples");
                }
            }
        }
    }
}
