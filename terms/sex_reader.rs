//! Sex evidence read straight from local VCF and BCF files.
//!
//! Sex inference needs a chromosome class, a position and an ALT count for every
//! record, and calls only for the rows it counts: a few thousand thinned autosomes
//! and the X and Y rows. The record reader parses every record into a key with REF
//! and ALT strings, then streams the selected rows through 64-bit dosage blocks.
//! These two passes read the bytes as the files hold them:
//!
//! 1. The blocks of each file (its BGZF blocks, or windows of a plain file's map)
//!    are inflated in parallel batches, and the records are walked in order for
//!    their chromosome, position and ALT count. Each block keeps how many records
//!    start ahead of it and where its own first record starts.
//! 2. Only the blocks holding selected records are read again. The GT calls of
//!    each selected row become one packed PLINK 1 row, which the packed counters
//!    of `sex_counts` add up.
//!
//! A pass accepts only what the record reader reads the same way. Anything else
//! (a remote file, text that is not ASCII or holds a carriage return, a malformed
//! record, a selected record without GT) makes it return `None`, and the caller
//! reads the dataset through the record reader, which decodes what these passes
//! leave out, or reports what is wrong with the file.

use std::fs::File;
use std::ops::Range;
use std::path::Path;

use flate2::Crc;
use libdeflater::Decompressor;
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

/// Bytes per window of a plain file's map, as many as a BGZF block holds. The
/// second pass finds a line by walking from the first line that starts in its
/// window, so the window bounds the bytes walked per selected line.
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

/// The first-start offset of a block in which nothing starts.
const NO_START: u32 = u32::MAX;
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

/// The last block a task inflated.
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
        let Some(frames) = &blocks.frames else {
            return (index < blocks.len()).then(|| blocks.plain_block(index));
        };
        if self.cached != Some((part, index)) {
            self.cached = None;
            inflate(
                &blocks.map[frames.get(index)?.clone()],
                &mut self.inflater,
                &mut self.block,
            )?;
            self.cached = Some((part, index));
        }
        Some(&self.block)
    }
}

/// Copies line `line` of VCF file `part`, without its newline, into `record`.
fn read_line(
    scan: &VariantScan,
    cache: &mut BlockCache,
    part: usize,
    line: u64,
    record: &mut Vec<u8>,
) -> Option<()> {
    let (mut block, ahead, mut start) = scan.parts[part].index.locate(line)?;
    let bytes = cache.read(scan, part, block)?;
    for _ in 0..ahead {
        start += memchr(b'\n', bytes.get(start..)?)? + 1;
    }
    record.clear();
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
                if block == scan.parts[part].blocks.len() {
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
    let (block, ahead, mut start) = scan.parts[part].index.locate(record)?;
    let bytes = cache.read(scan, part, block)?;
    // Every record ahead of this one in its block starts and ends in the block.
    for _ in 0..ahead {
        let (site_len, samples_len) = record_lengths(bytes.get(start..)?)?;
        start = start
            .checked_add(8)?
            .checked_add(site_len)?
            .checked_add(samples_len)?;
    }
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
            let mut rest = &samples[format_end + 1..];
            while !rest.is_empty() && sample < n_samples {
                let column = match memchr(b'\t', rest) {
                    Some(end) => {
                        let column = &rest[..end];
                        rest = &rest[end + 1..];
                        column
                    }
                    None => std::mem::take(&mut rest),
                };
                let gt = if gt_index == 0 {
                    Some(memchr(b':', column).map_or(column, |end| &column[..end]))
                } else {
                    column.split(|&byte| byte == b':').nth(gt_index)
                };
                for (row, &(alt, _)) in packed.chunks_exact_mut(row_len).zip(alts) {
                    let code = match gt {
                        Some(gt) => call_code(gt, alt)?,
                        None => MISSING_CODE,
                    };
                    set_code(row, sample, code);
                }
                sample += 1;
            }
        }
    }
    mark_missing(packed, row_len, sample..n_samples);
    Some(())
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
    let mut carried = vec![0usize; alts.len()];
    for sample in 0..n_samples {
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
        for (row, &count) in packed.chunks_exact_mut(row_len).zip(&carried) {
            let code = if missing || alleles == 0 {
                MISSING_CODE
            } else if alleles > 1 && count == 1 {
                HET_CODE
            } else {
                HOM_CODE
            };
            set_code(row, sample, code);
        }
    }
    Some(())
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
    /// Each block's first item start, or `NO_START`.
    first: Vec<u32>,
}

impl BlockIndex {
    fn begin_block(&mut self, items: u64) {
        self.before.push(items);
        self.first.push(NO_START);
    }

    /// Notes an item starting at `offset` of the current block.
    fn start(&mut self, offset: usize) {
        if let Some(first) = self.first.last_mut()
            && *first == NO_START
        {
            *first = offset as u32;
        }
    }

    /// The block item `item` starts in, how many items start in that block ahead
    /// of it, and where the block's first item starts.
    fn locate(&self, item: u64) -> Option<(usize, u64, usize)> {
        let block = self
            .before
            .partition_point(|&before| before <= item)
            .checked_sub(1)?;
        let first = self.first[block];
        (first != NO_START).then(|| (block, item - self.before[block], first as usize))
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

    fn plain_block(&self, index: usize) -> &[u8] {
        let start = index * self.window;
        &self.map[start..(start + self.window).min(self.map.len())]
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
/// inflated into its buffer, a plain window borrowed from the map. `None` when a
/// block does not inflate or `scan` refuses one.
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
        None => states
            .par_iter_mut()
            .zip(range.clone())
            .all(|(state, index)| scan(blocks.plain_block(index), state)),
    };
    if !read {
        return None;
    }
    let buffers: &'a Vec<Vec<u8>> = buffers;
    Some(match &blocks.frames {
        Some(_) => buffers[..n].iter().map(Vec::as_slice).collect(),
        None => range.map(|index| blocks.plain_block(index)).collect(),
    })
}

/// The first pass over one VCF file: its records' loci, and the index of its
/// lines with the number of header lines ahead of the records.
fn scan_vcf(blocks: &Blocks, builder: &mut LociBuilder) -> Option<(BlockIndex, u64)> {
    let batch = rayon::current_num_threads().max(1) * BLOCKS_PER_WORKER;
    let mut buffers = Vec::new();
    let mut newlines: Vec<Vec<u32>> = Vec::new();
    let mut walker = TextWalker::default();
    let mut index = BlockIndex::default();
    for start in (0..blocks.len()).step_by(batch) {
        let range = start..(start + batch).min(blocks.len());
        let views = read_window(blocks, range, &mut buffers, &mut newlines, scan_text_block)?;
        for (block, ends) in views.iter().zip(&newlines) {
            walker.walk(block, ends, &mut index, builder)?;
        }
    }
    walker.finish(builder)?;
    Some((index, walker.header_lines))
}

/// Finds the line ends of one block of VCF text, refusing a block that is not
/// ASCII or holds a carriage return, which the record reader strips from some
/// fields and not others.
fn scan_text_block(block: &[u8], newlines: &mut Vec<u32>) -> bool {
    newlines.clear();
    if !is_plain_ascii(block) {
        return false;
    }
    newlines.extend(memchr_iter(b'\n', block).map(|end| end as u32));
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
        newlines: &[u32],
        index: &mut BlockIndex,
        builder: &mut LociBuilder,
    ) -> Option<()> {
        index.begin_block(self.lines);
        let mut position = 0;
        let mut ends = newlines.iter();
        while position < block.len() {
            if !self.in_line {
                index.start(position);
                self.lines += 1;
                self.in_line = true;
                self.prefix.clear();
                self.tabs = 0;
            }
            let end = ends.next().map_or(block.len(), |&end| end as usize);
            self.take_prefix(&block[position..end]);
            if end < block.len() {
                self.end_line(builder)?;
                position = end + 1;
            } else {
                position = end;
            }
        }
        Some(())
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

    /// Reads the line that just ended.
    fn end_line(&mut self, builder: &mut LociBuilder) -> Option<()> {
        self.in_line = false;
        let line = &self.prefix;
        if !self.past_header {
            // The header is every line up to the column header, all starting
            // with '#'.
            if line.first() != Some(&b'#') {
                return None;
            }
            self.header_lines += 1;
            self.past_header = line.starts_with(b"#CHROM");
            return Some(());
        }
        // The record reader needs eight fields, and reads a '#' line after the
        // header as a record.
        if self.tabs < 7 || line.first() == Some(&b'#') {
            return None;
        }
        let mut fields = line.split(|&byte| byte == b'\t');
        let label = str::from_utf8(fields.next()?).ok()?;
        let position = parse_position(fields.next()?)?;
        let alts = alt_count(fields.nth(2)?);
        builder.record(label, position, alts)
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
        index.begin_block(5);
        assert_eq!(index.locate(0), Some((0, 0, 0)));
        assert_eq!(index.locate(1), Some((0, 1, 0)));
        assert_eq!(index.locate(2), Some((2, 0, 7)));
        assert_eq!(index.locate(4), Some((2, 2, 7)));
        assert_eq!(index.locate(5), None);
    }
}
