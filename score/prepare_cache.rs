//! Content-addressed variant plans. Person selection and genotype calls are
//! deliberately absent: the same BIM/weights can serve any cohort layout.
//!
//! A plan is named by a BLAKE3 digest of the bytes of every `.bim` and score file
//! it was compiled from, the region filters, the block partition and this build. File metadata never
//! stands in for bytes: timestamps can be set by anyone who can write the file, are
//! coarse on some filesystems, lag behind writes through a shared mapping and are
//! cached by NFS clients, so an unchanged size and modification time prove nothing
//! about content.
use super::FilesetPaths;
use super::blocks::BlockPartition;
use crate::score::cells::{ExactPlan, ExactTables};
use crate::score::types::{
    BimRowIndex, GenomicRegion, GroupedComplexRule, PipelineKind, PreparationResult,
    ScoreColumnIndex, ScoreInfo,
};
use rayon::prelude::*;
use std::{
    borrow::Cow,
    collections::HashMap,
    fs::{self, File},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

/// Plan format 5 stores the exact plan, whose integers stand in for the parsed f64 weights: a
/// header holding the key and one BLAKE3 digest per section, then little-endian arrays padded
/// to multiples of 8 bytes.
const MAGIC: [u8; 8] = *b"GNPLAN05";
/// Inputs are hashed in leaves of this many bytes, so a key depends only on the
/// bytes, never on thread count, read sizes or available memory.
const LEAF_BYTES: u64 = 4 << 20;
const SECTIONS: usize = 22;
const HEADER_BYTES: usize = MAGIC.len() + 32 + SECTIONS * (8 + 32) + 32;
/// A temporary plan file younger than this may belong to a writer still running.
const STALE_TEMPORARY_AGE: Duration = Duration::from_secs(24 * 60 * 60);

/// A plan's weights: as the join parsed them, or as the exact plan a saved plan stores.
pub(super) enum PlanWeights {
    /// Each entry's parsed weight and flip correction, and each score's flipped-allele baseline.
    Parsed {
        weights: Vec<f64>,
        corrections: Vec<f64>,
        baseline: Vec<f64>,
    },
    Exact(ExactPlan),
}

pub(super) struct VariantPlan {
    pub weights: PlanWeights,
    pub columns: Vec<u32>,
    pub offsets: Vec<u64>,
    pub required: Vec<BimRowIndex>,
    pub complex: Vec<GroupedComplexRule>,
    pub names: Vec<String>,
    pub counts: Vec<u32>,
    pub flags: Vec<u8>,
    pub starts: Vec<u64>,
    pub total_variants: u64,
}

pub(super) struct PlanCache {
    path: PathBuf,
    key: [u8; 32],
}

impl PlanCache {
    pub fn discover(
        filesets: &[FilesetPaths],
        scores: &[PathBuf],
        regions: Option<&HashMap<String, GenomicRegion>>,
        blocks: Option<&BlockPartition>,
    ) -> io::Result<Option<Self>> {
        // Plans hold arrays in their little-endian memory form.
        if cfg!(target_endian = "big") {
            return Ok(None);
        }
        // Remote text and PVAR adapters retain their streaming compiler. This
        // cache represents local BIM rows, whose numbering is content-defined.
        if filesets
            .iter()
            .any(|f| f.bim.extension().is_none_or(|e| e != "bim") || !f.bim.is_file())
        {
            return Ok(None);
        }
        let Some(directory) = plan_directory() else {
            return Ok(None);
        };
        let key = content_key(filesets, scores, regions, blocks)?;
        Ok(Some(Self {
            path: directory.join(hex::encode(key)),
            key,
        }))
    }

    pub fn same_inputs(&self, other: &Self) -> bool {
        self.key == other.key
    }

    #[cfg(test)]
    pub(super) fn path(&self) -> &std::path::Path {
        &self.path
    }

    pub fn load(&self) -> io::Result<Option<VariantPlan>> {
        read_plan(&self.path, &self.key)?
            .map(Sections::into_plan)
            .transpose()
    }

    pub fn save(&self, prep: &PreparationResult) -> io::Result<()> {
        let sections = Sections::from_preparation(prep);
        let directory = self
            .path
            .parent()
            .ok_or_else(|| invalid("Variant plan path has no directory"))?;
        fs::create_dir_all(directory)?;
        make_room(directory, &self.path, sections.file_len())?;
        write_plan(&self.path, &self.key, &sections)
    }
}

/// Where plans live: `variant-plans` under `$GNOMON_CACHE_DIR` when that is set, else
/// `gnomon/variant-plans` under the platform cache directory (`$XDG_CACHE_HOME` or
/// `~/.cache` on Linux, `~/Library/Caches` on macOS, `%LOCALAPPDATA%` on Windows).
fn plan_directory() -> Option<PathBuf> {
    match std::env::var_os("GNOMON_CACHE_DIR") {
        Some(root) if !root.is_empty() => Some(PathBuf::from(root).join("variant-plans")),
        _ => dirs::cache_dir().map(|root| root.join("gnomon").join("variant-plans")),
    }
}

/// Every source whose code decides what a saved plan holds, so that changing one invalidates
/// plans even when the build timestamp does not change: the join and its row and score parsing,
/// the block expansion, the text sources the join reads, the plan types and this format, the exact
/// plan's places, multiples, bands, flags and entry bands, and the shortest round-trip decimals
/// they scale, which the pinned `ryu` and toolchain decide.
const SOURCES: [&[u8]; 11] = [
    include_bytes!("prepare.rs"),
    include_bytes!("prepare_parse.rs"),
    include_bytes!("prepare_scores.rs"),
    include_bytes!("prepare_blocks.rs"),
    include_bytes!("prepare_cache.rs"),
    include_bytes!("types.rs"),
    include_bytes!("cells.rs"),
    include_bytes!("exact.rs"),
    include_bytes!("../shared/files.rs"),
    include_bytes!("../Cargo.lock"),
    include_bytes!("../rust-toolchain.toml"),
];

/// The digest naming a plan compiled from these inputs by this build.
fn content_key(
    filesets: &[FilesetPaths],
    scores: &[PathBuf],
    regions: Option<&HashMap<String, GenomicRegion>>,
    blocks: Option<&BlockPartition>,
) -> io::Result<[u8; 32]> {
    content_key_from(filesets, scores, regions, blocks, &SOURCES)
}

/// [`content_key`] of a build whose plan-deciding sources are `sources`.
fn content_key_from(
    filesets: &[FilesetPaths],
    scores: &[PathBuf],
    regions: Option<&HashMap<String, GenomicRegion>>,
    blocks: Option<&BlockPartition>,
    sources: &[&[u8]],
) -> io::Result<[u8; 32]> {
    let inputs: Vec<&Path> = filesets
        .iter()
        .map(|f| f.bim.as_path())
        .chain(scores.iter().map(PathBuf::as_path))
        .collect();
    let digests = content_digests(&inputs)?;
    let mut hash = blake3::Hasher::new();
    hash.update(&MAGIC);
    for field in [env!("CARGO_PKG_VERSION"), env!("GNOMON_BUILD_TIMESTAMP")] {
        hash.update(&(field.len() as u64).to_le_bytes());
        hash.update(field.as_bytes());
    }
    // Compiler changes invalidate plans automatically, including changes
    // to chromosome parsing and shared representation invariants.
    for source in sources {
        hash.update(blake3::hash(source).as_bytes());
    }
    for count in [filesets.len(), scores.len()] {
        hash.update(&(count as u64).to_le_bytes());
    }
    for digest in &digests {
        hash.update(digest);
    }
    let mut filters: Vec<_> = regions.into_iter().flat_map(|r| r.iter()).collect();
    filters.sort_unstable_by(|a, b| a.0.cmp(b.0));
    hash.update(&(filters.len() as u64).to_le_bytes());
    for (name, region) in filters {
        hash.update(&(name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(&[region.chromosome]);
        hash.update(&region.start.to_le_bytes());
        hash.update(&region.end.to_le_bytes());
    }
    // A plan expanded over blocks is a different plan.
    match blocks {
        Some(partition) => partition.hash_into(&mut hash),
        None => {
            hash.update(&[0u8]);
        }
    }
    Ok(*hash.finalize().as_bytes())
}

/// One digest per file: its length and the BLAKE3 digests of its fixed-size leaves,
/// which are read and hashed in parallel. Leaves are read with positioned reads, not
/// a mapping, so a file truncated underneath this process is an error, not a SIGBUS.
fn content_digests(paths: &[&Path]) -> io::Result<Vec<[u8; 32]>> {
    let files: Vec<(File, u64)> = paths
        .par_iter()
        .map(|path| -> io::Result<(File, u64)> {
            let file = File::open(path)?;
            let len = file.metadata()?.len();
            Ok((file, len))
        })
        .collect::<io::Result<_>>()?;
    let leaf_counts: Vec<u64> = files
        .iter()
        .map(|(_, len)| len.div_ceil(LEAF_BYTES).max(1))
        .collect();
    let leaves: Vec<(usize, u64)> = leaf_counts
        .iter()
        .enumerate()
        .flat_map(|(index, &count)| (0..count).map(move |leaf| (index, leaf * LEAF_BYTES)))
        .collect();
    let leaf_digests: Vec<[u8; 32]> = leaves
        .par_iter()
        .map_init(
            Vec::<u8>::new,
            |buffer, &(index, offset)| -> io::Result<[u8; 32]> {
                let (file, len) = &files[index];
                let count = (len - offset).min(LEAF_BYTES) as usize;
                buffer.resize(count, 0);
                read_exact_at(file, &mut buffer[..count], offset)?;
                Ok(*blake3::hash(&buffer[..count]).as_bytes())
            },
        )
        .collect::<io::Result<_>>()?;
    files
        .par_iter()
        .try_for_each(|(file, len)| -> io::Result<()> {
            if read_at(file, &mut [0u8; 1], *len)? != 0 {
                return Err(invalid("Variant input grew during content hashing"));
            }
            Ok(())
        })?;
    let mut next = 0;
    Ok(files
        .iter()
        .zip(&leaf_counts)
        .map(|((_, len), &count)| {
            let mut hash = blake3::Hasher::new();
            hash.update(&len.to_le_bytes());
            for digest in &leaf_digests[next..next + count as usize] {
                hash.update(digest);
            }
            next += count as usize;
            *hash.finalize().as_bytes()
        })
        .collect())
}

#[cfg(unix)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::unix::fs::FileExt::read_at(file, buffer, offset)
}

#[cfg(windows)]
fn read_at(file: &File, buffer: &mut [u8], offset: u64) -> io::Result<usize> {
    std::os::windows::fs::FileExt::seek_read(file, buffer, offset)
}

#[cfg(not(any(unix, windows)))]
fn read_at(_: &File, _: &mut [u8], _: u64) -> io::Result<usize> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

fn read_exact_at(file: &File, mut buffer: &mut [u8], mut offset: u64) -> io::Result<()> {
    while !buffer.is_empty() {
        match read_at(file, buffer, offset) {
            Ok(0) => return Err(invalid("Variant input shrank during content hashing")),
            Ok(count) => {
                buffer = &mut buffer[count..];
                offset += count as u64;
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// A saved plan: an entry in the plan cache directory.
struct SavedPlan {
    path: PathBuf,
    bytes: u64,
    modified: SystemTime,
}

/// Deletes the oldest saved plans until a plan of `needed` bytes fits. A plan larger
/// than this machine's memory could not be loaded again and is refused.
fn make_room(directory: &Path, destination: &Path, needed: u64) -> io::Result<()> {
    let (memory, _) = crate::memory::memory_bytes();
    if needed > memory {
        return Err(invalid("Variant plan is larger than this machine's memory"));
    }
    let available = fs4::available_space(directory)?;
    let mut entries = Vec::new();
    for entry in fs::read_dir(directory)? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path == destination {
            continue;
        }
        if let Ok(metadata) = entry.metadata()
            && metadata.is_file()
        {
            entries.push(SavedPlan {
                path,
                bytes: metadata.len(),
                modified: metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            });
        }
    }
    for path in evictions(needed, available, entries, SystemTime::now())? {
        // Another run may have removed it first.
        let _ = fs::remove_file(path);
    }
    Ok(())
}

/// Which saved plans to delete, oldest first, so that a plan of `needed` bytes fits.
/// The cache may occupy at most an eighth of the space its filesystem would have
/// free without it, so the ceiling follows the disk this machine actually has.
fn evictions(
    needed: u64,
    available: u64,
    mut entries: Vec<SavedPlan>,
    now: SystemTime,
) -> io::Result<Vec<PathBuf>> {
    let mut usage = entries.iter().map(|e| e.bytes).fold(0, u64::saturating_add);
    let ceiling = available.saturating_add(usage) / 8;
    if needed > ceiling {
        return Err(invalid(&format!(
            "Variant plan of {needed} bytes exceeds the plan cache ceiling of {ceiling} bytes, an eighth of the disk space the cache could use"
        )));
    }
    entries.sort_by_key(|e| e.modified);
    let mut evicted = Vec::new();
    for entry in entries {
        if usage.saturating_add(needed) <= ceiling {
            break;
        }
        let temporary = entry
            .path
            .file_name()
            .is_some_and(|name| name.as_encoded_bytes().starts_with(b"."));
        if temporary
            && now
                .duration_since(entry.modified)
                .is_ok_and(|age| age < STALE_TEMPORARY_AGE)
        {
            continue;
        }
        usage = usage.saturating_sub(entry.bytes);
        evicted.push(entry.path);
    }
    Ok(evicted)
}

/// Writes `sections` under `key`. Every section carries its own digest, and the
/// header carries one over itself, so a torn or altered file is refused on load.
fn write_plan(path: &Path, key: &[u8; 32], sections: &Sections<'_>) -> io::Result<()> {
    let stored = sections.stored();
    let digests: Vec<[u8; 32]> = stored.par_iter().map(Column::digest).collect();
    let mut header = Vec::with_capacity(HEADER_BYTES);
    header.extend_from_slice(&MAGIC);
    header.extend_from_slice(key);
    for (column, digest) in stored.iter().zip(&digests) {
        header.extend_from_slice(&(column.bytes().len() as u64).to_le_bytes());
        header.extend_from_slice(digest);
    }
    let digest = blake3::hash(&header);
    header.extend_from_slice(digest.as_bytes());
    crate::output::write_atomically(path, |writer| {
        writer.write_all(&header)?;
        for column in &stored {
            let bytes = column.bytes();
            writer.write_all(bytes)?;
            writer.write_all(&[0u8; 8][..padding(bytes.len() as u64)])?;
        }
        Ok(())
    })
}

/// Reads the plan saved under `key`, verifying every digest before returning it.
fn read_plan(path: &Path, key: &[u8; 32]) -> io::Result<Option<Sections<'static>>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let file_len = file.metadata()?.len();
    if file_len < HEADER_BYTES as u64 {
        return Err(invalid("Truncated variant plan"));
    }
    let mut header = [0u8; HEADER_BYTES];
    file.read_exact(&mut header)?;
    let (lengths, digests) = parse_header(&header, key, file_len)?;
    let sections = Sections::read(file, lengths)?;
    if sections
        .stored()
        .par_iter()
        .zip(digests.par_iter())
        .any(|(column, digest)| column.digest() != *digest)
    {
        return Err(invalid("Variant plan checksum mismatch"));
    }
    Ok(Some(sections))
}

/// Section lengths and digests from a header that belongs to `key` and describes a
/// file of exactly `file_len` bytes.
fn parse_header(
    header: &[u8; HEADER_BYTES],
    key: &[u8; 32],
    file_len: u64,
) -> io::Result<([u64; SECTIONS], [[u8; 32]; SECTIONS])> {
    let (body, digest) = header.split_at(HEADER_BYTES - 32);
    if body[..MAGIC.len()] != MAGIC {
        return Err(invalid("Variant plan was written in another plan format"));
    }
    if blake3::hash(body).as_bytes() != digest {
        return Err(invalid("Variant plan header checksum mismatch"));
    }
    if body[MAGIC.len()..MAGIC.len() + 32] != key[..] {
        return Err(invalid("Variant plan identity mismatch"));
    }
    let mut lengths = [0u64; SECTIONS];
    let mut digests = [[0u8; 32]; SECTIONS];
    let mut expected = HEADER_BYTES as u64;
    for (index, entry) in body[MAGIC.len() + 32..].chunks_exact(8 + 32).enumerate() {
        lengths[index] = u64::from_le_bytes(entry[..8].try_into().unwrap());
        digests[index].copy_from_slice(&entry[8..]);
        expected = lengths[index]
            .checked_add(padding(lengths[index]) as u64)
            .and_then(|len| expected.checked_add(len))
            .ok_or_else(|| invalid("Variant plan length overflow"))?;
    }
    if expected != file_len {
        return Err(invalid("Truncated variant plan or trailing data"));
    }
    Ok((lengths, digests))
}

fn padding(len: u64) -> usize {
    ((8 - len % 8) % 8) as usize
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// Numeric types with no padding and no invalid bit patterns, so a slice of them may
/// be viewed, and filled, as bytes.
unsafe trait Plain: Copy + Default {}
unsafe impl Plain for u8 {}
unsafe impl Plain for u32 {}
unsafe impl Plain for u64 {}
unsafe impl Plain for i64 {}
unsafe impl Plain for f64 {}

fn as_bytes<T: Plain>(values: &[T]) -> &[u8] {
    // SAFETY: `Plain` types have no padding, so every byte is initialized.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn as_bytes_mut<T: Plain>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: as in `as_bytes`, and any bytes written form a valid `T`.
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values))
    }
}

/// One stored array. Plans are used only on little-endian hosts, where an array's
/// memory is its stored form.
#[derive(Clone, Copy)]
enum Column<'a> {
    U8(&'a [u8]),
    U32(&'a [u32]),
    U64(&'a [u64]),
    I64(&'a [i64]),
    F64(&'a [f64]),
}

impl Column<'_> {
    fn bytes(&self) -> &[u8] {
        match *self {
            Column::U8(values) => values,
            Column::U32(values) => as_bytes(values),
            Column::U64(values) => as_bytes(values),
            Column::I64(values) => as_bytes(values),
            Column::F64(values) => as_bytes(values),
        }
    }

    fn digest(&self) -> [u8; 32] {
        *blake3::hash(self.bytes()).as_bytes()
    }
}

/// A plan's arrays in file order. A loaded plan owns them; saving borrows the large
/// ones from the compiled result. Strings are one byte blob and the end offset of
/// each string: the score names, then for each complex rule its chromosome, both
/// alleles of each context and the effect and other allele of each application.
#[derive(Default)]
struct Sections<'a> {
    /// `[total_variants, score_count]`.
    scalars: Cow<'a, [u64]>,
    starts: Cow<'a, [u64]>,
    /// The exact plan's entries: weights at their bands' scales, flags and bands.
    exact_weights: Cow<'a, [i64]>,
    exact_flags: Cow<'a, [u8]>,
    entry_band: Cow<'a, [u8]>,
    columns: Cow<'a, [u32]>,
    offsets: Cow<'a, [u64]>,
    required: Cow<'a, [u64]>,
    flags: Cow<'a, [u8]>,
    counts: Cow<'a, [u32]>,
    string_ends: Cow<'a, [u64]>,
    string_bytes: Cow<'a, [u8]>,
    rule_positions: Cow<'a, [u32]>,
    rule_context_ends: Cow<'a, [u64]>,
    context_rows: Cow<'a, [u64]>,
    rule_application_ends: Cow<'a, [u64]>,
    application_weights: Cow<'a, [f64]>,
    application_columns: Cow<'a, [u64]>,
    /// The exact plan's scores and wide weights, as [`ExactTables`] holds them.
    multiples: Cow<'a, [u64]>,
    band_ends: Cow<'a, [u64]>,
    bands: Cow<'a, [u64]>,
    wide: Cow<'a, [u64]>,
}

impl Sections<'_> {
    fn stored(&self) -> [Column<'_>; SECTIONS] {
        [
            Column::U64(&self.scalars),
            Column::U64(&self.starts),
            Column::I64(&self.exact_weights),
            Column::U8(&self.exact_flags),
            Column::U8(&self.entry_band),
            Column::U32(&self.columns),
            Column::U64(&self.offsets),
            Column::U64(&self.required),
            Column::U8(&self.flags),
            Column::U32(&self.counts),
            Column::U64(&self.string_ends),
            Column::U8(&self.string_bytes),
            Column::U32(&self.rule_positions),
            Column::U64(&self.rule_context_ends),
            Column::U64(&self.context_rows),
            Column::U64(&self.rule_application_ends),
            Column::F64(&self.application_weights),
            Column::U64(&self.application_columns),
            Column::U64(&self.multiples),
            Column::U64(&self.band_ends),
            Column::U64(&self.bands),
            Column::U64(&self.wide),
        ]
    }

    fn file_len(&self) -> u64 {
        self.stored()
            .iter()
            .map(|column| column.bytes().len() as u64)
            .map(|len| len + padding(len) as u64)
            .fold(HEADER_BYTES as u64, u64::saturating_add)
    }

    fn read(file: File, lengths: [u64; SECTIONS]) -> io::Result<Sections<'static>> {
        let mut reader = SectionReader {
            file,
            lengths,
            next: 0,
        };
        Ok(Sections {
            scalars: Cow::Owned(reader.read()?),
            starts: Cow::Owned(reader.read()?),
            exact_weights: Cow::Owned(reader.read()?),
            exact_flags: Cow::Owned(reader.read()?),
            entry_band: Cow::Owned(reader.read()?),
            columns: Cow::Owned(reader.read()?),
            offsets: Cow::Owned(reader.read()?),
            required: Cow::Owned(reader.read()?),
            flags: Cow::Owned(reader.read()?),
            counts: Cow::Owned(reader.read()?),
            string_ends: Cow::Owned(reader.read()?),
            string_bytes: Cow::Owned(reader.read()?),
            rule_positions: Cow::Owned(reader.read()?),
            rule_context_ends: Cow::Owned(reader.read()?),
            context_rows: Cow::Owned(reader.read()?),
            rule_application_ends: Cow::Owned(reader.read()?),
            application_weights: Cow::Owned(reader.read()?),
            application_columns: Cow::Owned(reader.read()?),
            multiples: Cow::Owned(reader.read()?),
            band_ends: Cow::Owned(reader.read()?),
            bands: Cow::Owned(reader.read()?),
            wide: Cow::Owned(reader.read()?),
        })
    }
}

impl<'a> Sections<'a> {
    fn from_preparation(prep: &'a PreparationResult) -> Self {
        let starts = match &prep.pipeline_kind {
            PipelineKind::SingleFile(_) => vec![0],
            PipelineKind::MultiFile(boundaries) => boundaries
                .iter()
                .map(|boundary| boundary.starting_global_index)
                .collect(),
        };
        let mut strings = StringTable::default();
        for name in &prep.score_names {
            strings.push(name);
        }
        let rules = prep.complex_rules.len();
        let mut rule_positions = Vec::with_capacity(rules);
        let mut rule_context_ends = Vec::with_capacity(rules);
        let mut rule_application_ends = Vec::with_capacity(rules);
        let mut context_rows = Vec::new();
        let mut application_weights = Vec::new();
        let mut application_columns = Vec::new();
        for rule in &prep.complex_rules {
            strings.push(&rule.locus_chr_pos.0);
            rule_positions.push(rule.locus_chr_pos.1);
            for (row, allele1, allele2) in &rule.possible_contexts {
                context_rows.push(row.0);
                strings.push(allele1);
                strings.push(allele2);
            }
            rule_context_ends.push(context_rows.len() as u64);
            for score in &rule.score_applications {
                application_weights.push(score.weight);
                application_columns.push(score.score_column_index.0 as u64);
                strings.push(&score.effect_allele);
                strings.push(&score.other_allele);
            }
            rule_application_ends.push(application_weights.len() as u64);
        }
        // The exact plan is what a loaded plan scores with, so it is stored in place of the
        // parsed weights it was built from.
        let (exact_weights, exact_flags, entry_band) = prep.exact().entry_arrays();
        let ExactTables {
            multiples,
            band_ends,
            bands,
            wide,
        } = prep.exact().tables();
        Sections {
            scalars: Cow::Owned(vec![
                prep.total_variants_in_bim,
                prep.score_names.len() as u64,
            ]),
            starts: Cow::Owned(starts),
            exact_weights: Cow::Borrowed(exact_weights),
            exact_flags: Cow::Borrowed(exact_flags),
            entry_band: Cow::Borrowed(entry_band),
            columns: Cow::Borrowed(prep.sparse_score_columns()),
            offsets: Cow::Borrowed(prep.sparse_row_offsets()),
            required: Cow::Owned(prep.required_bim_indices.iter().map(|r| r.0).collect()),
            flags: Cow::Borrowed(prep.required_is_complex()),
            counts: Cow::Borrowed(&prep.score_variant_counts),
            string_ends: Cow::Owned(strings.ends),
            string_bytes: Cow::Owned(strings.bytes),
            rule_positions: Cow::Owned(rule_positions),
            rule_context_ends: Cow::Owned(rule_context_ends),
            context_rows: Cow::Owned(context_rows),
            rule_application_ends: Cow::Owned(rule_application_ends),
            application_weights: Cow::Owned(application_weights),
            application_columns: Cow::Owned(application_columns),
            multiples: Cow::Owned(multiples),
            band_ends: Cow::Owned(band_ends),
            bands: Cow::Owned(bands),
            wide: Cow::Owned(wide),
        }
    }

    fn into_plan(self) -> io::Result<VariantPlan> {
        let &[total_variants, score_count] = &self.scalars[..] else {
            return Err(invalid("Invalid variant plan scalars"));
        };
        let score_count = usize::try_from(score_count)
            .ok()
            .filter(|&count| count <= self.string_ends.len())
            .ok_or_else(|| invalid("Invalid variant plan score count"))?;
        if self.rule_context_ends.len() != self.rule_positions.len()
            || self.rule_application_ends.len() != self.rule_positions.len()
            || self.application_columns.len() != self.application_weights.len()
        {
            return Err(invalid("Invalid variant plan complex rule tables"));
        }
        check_ends(&self.rule_context_ends, self.context_rows.len())?;
        check_ends(&self.rule_application_ends, self.application_weights.len())?;

        let mut strings = StringReader {
            ends: &self.string_ends,
            bytes: &self.string_bytes,
            next: 0,
            start: 0,
        };
        let names = (0..score_count)
            .map(|_| strings.next())
            .collect::<io::Result<Vec<_>>>()?;
        let mut complex = Vec::with_capacity(self.rule_positions.len());
        let (mut context, mut application) = (0, 0);
        for ((&position, &context_end), &application_end) in self
            .rule_positions
            .iter()
            .zip(self.rule_context_ends.iter())
            .zip(self.rule_application_ends.iter())
        {
            let chromosome = strings.next()?;
            let (context_end, application_end) = (context_end as usize, application_end as usize);
            let possible_contexts = self.context_rows[context..context_end]
                .iter()
                .map(|&row| -> io::Result<_> {
                    Ok((BimRowIndex(row), strings.next()?, strings.next()?))
                })
                .collect::<io::Result<Vec<_>>>()?;
            let score_applications = (application..application_end)
                .map(|index| -> io::Result<ScoreInfo> {
                    Ok(ScoreInfo {
                        effect_allele: strings.next()?,
                        other_allele: strings.next()?,
                        weight: self.application_weights[index],
                        score_column_index: ScoreColumnIndex(
                            usize::try_from(self.application_columns[index])
                                .map_err(|_| invalid("Score column overflow"))?,
                        ),
                    })
                })
                .collect::<io::Result<Vec<_>>>()?;
            complex.push(GroupedComplexRule {
                locus_chr_pos: (chromosome, position),
                possible_contexts,
                score_applications,
            });
            (context, application) = (context_end, application_end);
        }
        if strings.next != self.string_ends.len() || strings.start != self.string_bytes.len() {
            return Err(invalid("Trailing variant plan strings"));
        }

        if self.multiples.len() != score_count {
            return Err(invalid("Invalid variant plan exact score tables"));
        }
        let tables = ExactTables {
            multiples: self.multiples.into_owned(),
            band_ends: self.band_ends.into_owned(),
            bands: self.bands.into_owned(),
            wide: self.wide.into_owned(),
        };
        let exact = ExactPlan::from_parts(
            self.exact_weights.into_owned(),
            self.exact_flags.into_owned(),
            self.entry_band.into_owned(),
            tables,
            &self.columns,
        )
        .map_err(|error| invalid(&error.to_string()))?;
        let plan = VariantPlan {
            weights: PlanWeights::Exact(exact),
            columns: self.columns.into_owned(),
            offsets: self.offsets.into_owned(),
            required: self
                .required
                .into_owned()
                .into_iter()
                .map(BimRowIndex)
                .collect(),
            complex,
            names,
            counts: self.counts.into_owned(),
            flags: self.flags.into_owned(),
            starts: self.starts.into_owned(),
            total_variants,
        };
        plan.validate()?;
        Ok(plan)
    }
}

/// Reads the sections of a plan file in order, each followed by its zero padding.
struct SectionReader {
    file: File,
    lengths: [u64; SECTIONS],
    next: usize,
}

impl SectionReader {
    fn read<T: Plain>(&mut self) -> io::Result<Vec<T>> {
        let len = self.lengths[self.next];
        self.next += 1;
        let width = std::mem::size_of::<T>() as u64;
        if len % width != 0 {
            return Err(invalid("Misaligned variant plan section"));
        }
        let count =
            usize::try_from(len / width).map_err(|_| invalid("Variant plan section overflow"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|e| invalid(&format!("Variant plan allocation failed: {e}")))?;
        values.resize(count, T::default());
        self.file.read_exact(as_bytes_mut(&mut values))?;
        let mut pad = [0u8; 8];
        let pad = &mut pad[..padding(len)];
        self.file.read_exact(pad)?;
        if pad.iter().any(|&byte| byte != 0) {
            return Err(invalid("Nonzero variant plan padding"));
        }
        Ok(values)
    }
}

#[derive(Default)]
struct StringTable {
    ends: Vec<u64>,
    bytes: Vec<u8>,
}

impl StringTable {
    fn push(&mut self, text: &str) {
        self.bytes.extend_from_slice(text.as_bytes());
        self.ends.push(self.bytes.len() as u64);
    }
}

struct StringReader<'a> {
    ends: &'a [u64],
    bytes: &'a [u8],
    next: usize,
    start: usize,
}

impl StringReader<'_> {
    fn next(&mut self) -> io::Result<String> {
        let end = self
            .ends
            .get(self.next)
            .and_then(|&end| usize::try_from(end).ok())
            .filter(|&end| end >= self.start && end <= self.bytes.len())
            .ok_or_else(|| invalid("Invalid variant plan string table"))?;
        let text = std::str::from_utf8(&self.bytes[self.start..end])
            .map_err(|_| invalid("Invalid UTF-8 in variant plan"))?;
        self.next += 1;
        self.start = end;
        Ok(text.to_owned())
    }
}

/// `ends` must be the non-decreasing end offsets of consecutive ranges covering
/// exactly `total` items.
fn check_ends(ends: &[u64], total: usize) -> io::Result<()> {
    let mut previous = 0;
    for &end in ends {
        if end < previous || end > total as u64 {
            return Err(invalid("Invalid variant plan range table"));
        }
        previous = end;
    }
    if previous != total as u64 {
        return Err(invalid("Invalid variant plan range table"));
    }
    Ok(())
}

impl VariantPlan {
    fn validate(&self) -> io::Result<()> {
        let rows = self.required.len();
        let scores = self.names.len();
        if scores == 0
            || rows == 0
            || self.offsets.len() != rows + 1
            || self.flags.len() != rows
            || self.counts.len() != scores
            || self.offsets.first() != Some(&0)
            || self.offsets.last() != Some(&(self.columns.len() as u64))
            || self.offsets.par_windows(2).any(|w| w[0] > w[1])
            || self.columns.par_iter().any(|&c| c as usize >= scores)
            || self.required.par_windows(2).any(|w| w[0].0 >= w[1].0)
            || self.required.last().unwrap().0 >= self.total_variants
            || self.flags.par_iter().any(|&f| f > 1)
            || self.starts.first() != Some(&0)
            || self.starts.windows(2).any(|w| w[0] > w[1])
            || self.starts.last().is_some_and(|&s| s > self.total_variants)
        {
            return Err(invalid(
                "Invalid compiled variant plan dimensions or indices",
            ));
        }
        for rule in &self.complex {
            if rule.possible_contexts.is_empty()
                || rule.score_applications.is_empty()
                || rule
                    .possible_contexts
                    .iter()
                    .any(|(row, _, _)| self.required.binary_search_by_key(&row.0, |r| r.0).is_err())
                || rule
                    .score_applications
                    .iter()
                    .any(|s| s.score_column_index.0 >= scores)
            {
                return Err(invalid("Invalid compiled complex variant rule"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(dir: &Path) -> (Vec<FilesetPaths>, Vec<PathBuf>) {
        let prefix = dir.join("panel");
        let files = FilesetPaths {
            bed: prefix.with_extension("bed"),
            bim: prefix.with_extension("bim"),
            fam: prefix.with_extension("fam"),
        };
        std::fs::write(&files.bim, "1 a 0 100 A G\n").unwrap();
        std::fs::write(&files.fam, "F I 0 0 0 -9\n").unwrap();
        std::fs::write(&files.bed, [0x6c, 0x1b, 0x01, 2]).unwrap();
        let score = dir.join("weights.tsv");
        std::fs::write(
            &score,
            "variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.25\n",
        )
        .unwrap();
        (vec![files], vec![score])
    }

    /// A panel whose one position carries two alleles, so the plan holds a complex rule.
    fn complex_fixture(dir: &Path) -> (Vec<FilesetPaths>, Vec<PathBuf>) {
        let (files, scores) = fixture(dir);
        std::fs::write(
            &files[0].bim,
            "1 a 0 100 A G\n1 b 0 100 A T\n1 c 0 200 C T\n",
        )
        .unwrap();
        std::fs::write(&files[0].bed, [0x6c, 0x1b, 0x01, 2, 2, 2]).unwrap();
        std::fs::write(
            &scores[0],
            "variant_id\teffect_allele\tother_allele\tS\tR\n\
             1:100\tG\tA\t0.25\t-0.0\n\
             1:100\tT\tA\t1e-40\t3\n\
             1:200\tT\tC\t-2.5\t0.125\n",
        )
        .unwrap();
        (files, scores)
    }

    fn compile(dir: &Path, scores: &[PathBuf]) -> PreparationResult {
        super::super::prepare_for_computation_with_retry(
            &[dir.join("panel")],
            scores,
            None,
            None,
            None,
            None,
            super::super::BimRowOrder::Streamed,
        )
        .unwrap()
        .0
    }

    fn cache_in(dir: &Path, files: &[FilesetPaths], scores: &[PathBuf]) -> PlanCache {
        PlanCache {
            path: dir.join("plans").join("plan"),
            key: content_key(files, scores, None, None).unwrap(),
        }
    }

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn exact_of(plan: &VariantPlan) -> &ExactPlan {
        match &plan.weights {
            PlanWeights::Exact(exact) => exact,
            PlanWeights::Parsed { .. } => panic!("a loaded plan holds its exact plan"),
        }
    }

    #[test]
    fn key_is_independent_of_thread_count_and_covers_every_leaf() {
        let dir = tempfile::tempdir().unwrap();
        let (files, _) = fixture(dir.path());
        let score = dir.path().join("wide.tsv");
        let mut data = vec![0u8; LEAF_BYTES as usize * 2 + 37];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        std::fs::write(&score, &data).unwrap();
        let scores = vec![score.clone()];
        let key_with = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| content_key(&files, &scores, None, None).unwrap())
        };
        let expected = key_with(1);
        for threads in [2, 3, 8] {
            assert_eq!(key_with(threads), expected);
        }
        for index in [
            0,
            LEAF_BYTES as usize - 1,
            LEAF_BYTES as usize,
            data.len() - 1,
        ] {
            data[index] ^= 1;
            std::fs::write(&score, &data).unwrap();
            assert_ne!(key_with(3), expected, "byte {index}");
            data[index] ^= 1;
        }
        std::fs::write(&score, &data).unwrap();
        assert_eq!(key_with(3), expected);
        for len in [data.len() - 1, data.len() + 1] {
            let mut resized = data.clone();
            resized.resize(len, 0);
            std::fs::write(&score, &resized).unwrap();
            assert_ne!(key_with(3), expected, "length {len}");
        }
    }

    #[test]
    fn same_size_edits_under_a_restored_timestamp_change_the_key() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = fixture(dir.path());
        let original = content_key(&files, &scores, None, None).unwrap();
        for (path, from, to) in [(&files[0].bim, "A G", "A C"), (&scores[0], "0.25", "0.75")] {
            let modified = std::fs::metadata(path).unwrap().modified().unwrap();
            let text = std::fs::read_to_string(path).unwrap();
            let edited = text.replace(from, to);
            assert_eq!(edited.len(), text.len());
            std::fs::write(path, &edited).unwrap();
            File::options()
                .write(true)
                .open(path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(modified))
                .unwrap();
            assert_eq!(
                std::fs::metadata(path).unwrap().modified().unwrap(),
                modified
            );
            assert_ne!(content_key(&files, &scores, None, None).unwrap(), original);
            std::fs::write(path, &text).unwrap();
            assert_eq!(content_key(&files, &scores, None, None).unwrap(), original);
        }
        let filtered = HashMap::from([(
            "S".into(),
            GenomicRegion {
                chromosome: 1,
                start: 90,
                end: 110,
            },
        )]);
        assert_ne!(
            content_key(&files, &scores, Some(&filtered), None).unwrap(),
            original
        );
        let chromosomes = BlockPartition::chromosomes();
        let by_chromosome = content_key(&files, &scores, None, Some(&chromosomes)).unwrap();
        assert_ne!(by_chromosome, original);
        let bed = BlockPartition::from_bed_text("1\t0\t100\n").unwrap();
        assert_ne!(
            content_key(&files, &scores, None, Some(&bed)).unwrap(),
            by_chromosome
        );
    }

    #[test]
    fn saved_plans_load_back_bit_for_bit_with_complex_rules() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = complex_fixture(dir.path());
        let prep = compile(dir.path(), &scores);
        assert!(!prep.complex_rules.is_empty());
        let cache = cache_in(dir.path(), &files, &scores);
        assert!(cache.load().unwrap().is_none());
        cache.save(&prep).unwrap();
        let plan = cache.load().unwrap().unwrap();
        // Score S spans forty decimal orders, so it is banded; the loaded exact plan is the
        // compiled one, integer for integer.
        assert_eq!(exact_of(&plan), prep.exact());
        assert_eq!(plan.columns, prep.sparse_score_columns());
        assert_eq!(plan.offsets, prep.sparse_row_offsets());
        assert_eq!(plan.required, prep.required_bim_indices);
        assert_eq!(plan.flags, prep.required_is_complex());
        assert_eq!(plan.counts, prep.score_variant_counts);
        assert_eq!(plan.names, prep.score_names);
        assert_eq!(plan.total_variants, prep.total_variants_in_bim);
        assert_eq!(plan.starts, vec![0]);
        assert_eq!(plan.complex.len(), prep.complex_rules.len());
        for (loaded, compiled) in plan.complex.iter().zip(&prep.complex_rules) {
            assert_eq!(loaded.locus_chr_pos, compiled.locus_chr_pos);
            assert_eq!(loaded.possible_contexts, compiled.possible_contexts);
            assert_eq!(
                loaded.score_applications.len(),
                compiled.score_applications.len()
            );
            for (a, b) in loaded
                .score_applications
                .iter()
                .zip(&compiled.score_applications)
            {
                assert_eq!(a.effect_allele, b.effect_allele);
                assert_eq!(a.other_allele, b.other_allele);
                assert_eq!(a.weight.to_bits(), b.weight.to_bits());
                assert_eq!(a.score_column_index, b.score_column_index);
            }
        }
    }

    #[test]
    fn stored_arrays_keep_signed_zero_nan_payloads_and_subnormals() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plan");
        let key = [9u8; 32];
        let floats = [
            0u64,
            0x8000_0000_0000_0000,
            1,
            0x7ff0_0000_0000_0000,
            0x7ff8_0000_0000_1234,
            0xffef_ffff_ffff_ffff,
        ]
        .map(f64::from_bits);
        let integers = [0, -1, 1, i64::MIN, i64::MAX, -(1 << 62)];
        let sections = Sections {
            application_weights: Cow::Borrowed(&floats),
            exact_weights: Cow::Borrowed(&integers),
            exact_flags: Cow::Borrowed(&[3, 0, 1]),
            string_bytes: Cow::Borrowed(b"abc"),
            ..Sections::default()
        };
        write_plan(&path, &key, &sections).unwrap();
        let read = read_plan(&path, &key).unwrap().unwrap();
        assert_eq!(bits(&read.application_weights), bits(&floats));
        assert_eq!(&read.exact_weights[..], integers);
        assert_eq!(&read.exact_flags[..], [3, 0, 1]);
        assert_eq!(&read.string_bytes[..], b"abc");
        assert_eq!(std::fs::metadata(&path).unwrap().len(), sections.file_len());
    }

    #[test]
    fn corrupt_truncated_extended_and_foreign_plans_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = complex_fixture(dir.path());
        let prep = compile(dir.path(), &scores);
        let cache = cache_in(dir.path(), &files, &scores);
        cache.save(&prep).unwrap();
        let bytes = std::fs::read(&cache.path).unwrap();
        assert!(bytes.len() > HEADER_BYTES + 64);
        let refused = |contents: &[u8], why: &str| {
            std::fs::write(&cache.path, contents).unwrap();
            assert!(cache.load().is_err(), "{why}");
        };
        for offset in [
            0,
            MAGIC.len(),
            MAGIC.len() + 32,
            HEADER_BYTES - 1,
            HEADER_BYTES,
            bytes.len() / 2,
            bytes.len() - 1,
        ] {
            let mut flipped = bytes.clone();
            flipped[offset] ^= 0x10;
            refused(&flipped, &format!("bit flip at {offset}"));
        }
        for len in [
            0,
            7,
            HEADER_BYTES - 1,
            HEADER_BYTES,
            HEADER_BYTES + 1,
            bytes.len() - 1,
        ] {
            refused(&bytes[..len], &format!("truncated to {len}"));
        }
        let mut extended = bytes.clone();
        extended.push(0);
        refused(&extended, "one trailing byte");
        let mut first_format = b"GNOMON_VARIANT_PLAN_1\n".to_vec();
        first_format.extend_from_slice(&bytes);
        refused(&first_format, "first plan format");

        // A complete plan saved for other inputs or by another build.
        std::fs::write(&cache.path, &bytes).unwrap();
        let other = PlanCache {
            path: cache.path.clone(),
            key: [7; 32],
        };
        assert!(other.load().is_err());
        assert!(cache.load().unwrap().is_some());
    }

    #[test]
    fn every_plan_deciding_source_is_in_the_key() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = fixture(dir.path());
        let original = content_key(&files, &scores, None, None).unwrap();
        for index in 0..SOURCES.len() {
            let mut sources = SOURCES;
            let edited = [SOURCES[index], &b" "[..]].concat();
            sources[index] = &edited;
            let key = content_key_from(&files, &scores, None, None, &sources).unwrap();
            assert_ne!(key, original, "source {index}");
        }
    }

    #[test]
    fn stale_plans_are_compiled_again_and_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = complex_fixture(dir.path());
        // Inputs no other test compiles, so no other test writes this plan.
        let text = std::fs::read_to_string(&scores[0]).unwrap();
        std::fs::write(&scores[0], text.replace("0.25", "0.375")).unwrap();
        let Some(cache) = PlanCache::discover(&files, &scores, None, None).unwrap() else {
            return;
        };
        let prepare = || {
            super::super::prepare_for_computation(&[dir.path().join("panel")], &scores, None, None)
                .unwrap()
        };
        let fresh = prepare();
        let saved = std::fs::read(&cache.path).unwrap();
        let reseal = |bytes: &mut [u8]| {
            let body = HEADER_BYTES - 32;
            let digest = blake3::hash(&bytes[..body]);
            bytes[body..HEADER_BYTES].copy_from_slice(digest.as_bytes());
        };
        // This plan under the previous format's magic, in a header that checks.
        let mut previous = saved.clone();
        previous[..MAGIC.len()].copy_from_slice(b"GNPLAN04");
        reseal(&mut previous);
        let truncated = saved[..HEADER_BYTES + (saved.len() - HEADER_BYTES) / 2].to_vec();
        // The first section's digest with one byte flipped, in a header that checks.
        let mut flipped = saved.clone();
        flipped[MAGIC.len() + 32 + 8] ^= 1;
        reseal(&mut flipped);
        // What a build with one plan-deciding source changed saves: under its own key, which
        // this build never looks up, and so here only as a file at this key's path.
        let mut sources = SOURCES;
        let edited = [SOURCES[6], &b"\n"[..]].concat();
        sources[6] = &edited;
        let other_key = content_key_from(&files, &scores, None, None, &sources).unwrap();
        assert_ne!(other_key, cache.key);
        let other_build = PlanCache {
            path: dir.path().join("other-build").join(hex::encode(other_key)),
            key: other_key,
        };
        other_build.save(&fresh).unwrap();
        let changed_source = std::fs::read(&other_build.path).unwrap();
        for (bytes, why) in [
            (previous, "previous format"),
            (truncated, "truncated section"),
            (flipped, "flipped digest"),
            (changed_source, "changed source"),
        ] {
            std::fs::write(&cache.path, &bytes).unwrap();
            assert!(cache.load().is_err(), "{why}: refused");
            let again = prepare();
            assert_eq!(again.exact(), fresh.exact(), "{why}: compiled again");
            assert_eq!(
                std::fs::read(&cache.path).unwrap(),
                saved,
                "{why}: the compiled plan replaces the stale one"
            );
        }
    }

    #[test]
    fn headers_describing_another_length_are_refused_before_allocating() {
        let key = [3u8; 32];
        let empty = Sections::default();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("plan");
        write_plan(&path, &key, &empty).unwrap();
        let mut header: [u8; HEADER_BYTES] = std::fs::read(&path).unwrap()[..HEADER_BYTES]
            .try_into()
            .unwrap();
        assert!(parse_header(&header, &key, HEADER_BYTES as u64).is_ok());
        for claim in [1u64 << 40, u64::MAX] {
            header[MAGIC.len() + 32..MAGIC.len() + 40].copy_from_slice(&claim.to_le_bytes());
            let body = HEADER_BYTES - 32;
            let digest = blake3::hash(&header[..body]);
            header[body..].copy_from_slice(digest.as_bytes());
            assert!(parse_header(&header, &key, HEADER_BYTES as u64).is_err());
        }
    }

    #[test]
    fn checksummed_but_inconsistent_tables_are_refused() {
        let scalars = [1u64, 1];
        let valid = || Sections {
            scalars: Cow::Borrowed(&scalars),
            starts: Cow::Borrowed(&[0]),
            // 0.5 at one decimal place, counted as its row's first entry of its score.
            exact_weights: Cow::Borrowed(&[5]),
            exact_flags: Cow::Borrowed(&[2]),
            columns: Cow::Borrowed(&[0]),
            offsets: Cow::Borrowed(&[0, 1]),
            required: Cow::Borrowed(&[0]),
            flags: Cow::Borrowed(&[0]),
            counts: Cow::Borrowed(&[1]),
            string_ends: Cow::Borrowed(&[1]),
            string_bytes: Cow::Borrowed(b"S"),
            multiples: Cow::Borrowed(&[1]),
            band_ends: Cow::Borrowed(&[1]),
            bands: Cow::Borrowed(&[1, 1, 0, 0, 0]),
            ..Sections::default()
        };
        assert!(valid().into_plan().is_ok());
        let broken = [
            Sections {
                exact_flags: Cow::Borrowed(&[4]),
                ..valid()
            },
            Sections {
                exact_weights: Cow::Borrowed(&[5, 5]),
                ..valid()
            },
            Sections {
                exact_weights: Cow::Borrowed(&[i64::MIN]),
                ..valid()
            },
            Sections {
                wide: Cow::Borrowed(&[0, 5, 0]),
                ..valid()
            },
            Sections {
                multiples: Cow::Borrowed(&[]),
                band_ends: Cow::Borrowed(&[]),
                ..valid()
            },
            Sections {
                multiples: Cow::Borrowed(&[0]),
                ..valid()
            },
            Sections {
                band_ends: Cow::Borrowed(&[2]),
                ..valid()
            },
            Sections {
                bands: Cow::Borrowed(&[1, 3, 0, 0, 0]),
                ..valid()
            },
            Sections {
                bands: Cow::Borrowed(&[1, 2, 0, 0, 0]),
                ..valid()
            },
            Sections {
                band_ends: Cow::Borrowed(&[2]),
                bands: Cow::Borrowed(&[1, 1, 0, 0, 0, 1, 1, 0, 0, 0]),
                entry_band: Cow::Borrowed(&[0]),
                ..valid()
            },
            Sections {
                string_ends: Cow::Borrowed(&[2]),
                ..valid()
            },
            Sections {
                string_bytes: Cow::Borrowed(b"SS"),
                ..valid()
            },
            Sections {
                scalars: Cow::Borrowed(&[1, 2]),
                ..valid()
            },
            Sections {
                rule_positions: Cow::Borrowed(&[100]),
                rule_context_ends: Cow::Borrowed(&[1]),
                rule_application_ends: Cow::Borrowed(&[0]),
                ..valid()
            },
            Sections {
                columns: Cow::Borrowed(&[1]),
                ..valid()
            },
            Sections {
                required: Cow::Borrowed(&[1]),
                ..valid()
            },
        ];
        for (index, sections) in broken.into_iter().enumerate() {
            assert!(sections.into_plan().is_err(), "case {index}");
        }
    }

    #[cfg(unix)]
    #[test]
    fn concurrent_writers_never_expose_a_partial_plan() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = complex_fixture(dir.path());
        let prep = compile(dir.path(), &scores);
        let cache = cache_in(dir.path(), &files, &scores);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..20 {
                        cache.save(&prep).unwrap();
                    }
                });
            }
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..100 {
                        if let Some(plan) = cache.load().unwrap() {
                            assert_eq!(exact_of(&plan), prep.exact());
                        }
                    }
                });
            }
        });
        assert!(cache.load().unwrap().is_some());
    }

    #[test]
    fn ceiling_follows_free_space_and_evicts_the_oldest_plans_first() {
        let at = |secs| SystemTime::UNIX_EPOCH + Duration::from_secs(secs);
        let now = at(10 * 24 * 60 * 60);
        let saved = |name: &str, bytes, modified| SavedPlan {
            path: PathBuf::from(name),
            bytes,
            modified,
        };
        assert!(evictions(101, 800, Vec::new(), now).is_err());
        assert!(evictions(100, 800, Vec::new(), now).unwrap().is_empty());
        // 700 bytes free and 100 held: the cache may hold 100 bytes.
        let entries = vec![
            saved("new", 30, at(300)),
            saved("old", 40, at(100)),
            saved("mid", 30, at(200)),
        ];
        assert_eq!(
            evictions(60, 700, entries, now).unwrap(),
            vec![PathBuf::from("old"), PathBuf::from("mid")]
        );
        // A young temporary file may belong to a live writer; a day-old one does not.
        let entries = vec![
            saved(".plan.1.2.tmp", 50, now - Duration::from_secs(60)),
            saved(".plan.3.4.tmp", 50, at(0)),
        ];
        assert_eq!(
            evictions(50, 700, entries, now).unwrap(),
            vec![PathBuf::from(".plan.3.4.tmp")]
        );
    }

    #[cfg(unix)]
    #[test]
    fn an_unwritable_cache_directory_saves_and_serves_nothing() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = fixture(dir.path());
        let prep = compile(dir.path(), &scores);
        let locked = dir.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o555)).unwrap();
        // Root ignores permission bits, so there is nothing to observe then.
        if File::create(locked.join("probe")).is_err() {
            let cache = PlanCache {
                path: locked.join("variant-plans").join("plan"),
                key: content_key(&files, &scores, None, None).unwrap(),
            };
            assert!(cache.save(&prep).is_err());
            assert!(cache.load().unwrap().is_none());
        }
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn saving_evicts_older_plans_and_keeps_the_new_one() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = fixture(dir.path());
        let prep = compile(dir.path(), &scores);
        let cache = cache_in(dir.path(), &files, &scores);
        let directory = cache.path.parent().unwrap();
        std::fs::create_dir_all(directory).unwrap();
        let stale = directory.join("0".repeat(64));
        std::fs::write(&stale, b"old plan").unwrap();
        cache.save(&prep).unwrap();
        // Plenty of disk: nothing needed to go.
        assert!(stale.exists());
        assert!(cache.load().unwrap().is_some());
    }
}
