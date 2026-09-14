//! Content-addressed variant plans. Person selection and genotype calls are
//! deliberately absent: the same BIM/weights can serve any cohort layout.
use super::{FilesetPaths, PrepError};
use crate::score::types::{
    BimRowIndex, GenomicRegion, GroupedComplexRule, PipelineKind, PreparationResult,
    ScoreColumnIndex, ScoreInfo,
};
use memmap2::MmapOptions;
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::File,
    io::{self, Read, Write},
    path::PathBuf,
};

const MAGIC: &[u8] = b"GNOMON_VARIANT_PLAN_1\n";
const MAX_CACHE_BYTES: usize = 256 * 1024 * 1024;
const HASH_CHUNK_BYTES: usize = 256 * 1024;
const MAX_HASH_CHUNKS: usize = 4;

// Fixed chunk boundaries make the identity independent of worker count, read
// sizes, and the memory budget. Read a bounded window, hash its chunks in
// parallel, then incorporate their digests in input order.
fn hash_contents(
    reader: &mut impl Read,
    len: u64,
    buffer: &mut [u8],
    hash: &mut Sha256,
) -> io::Result<()> {
    debug_assert!(!buffer.is_empty() && buffer.len() % HASH_CHUNK_BYTES == 0);
    debug_assert!(buffer.len() <= HASH_CHUNK_BYTES * MAX_HASH_CHUNKS);
    hash.update(len.to_le_bytes());
    let mut remaining = len;
    let mut digests = [[0u8; 32]; MAX_HASH_CHUNKS];
    while remaining != 0 {
        let count = remaining.min(buffer.len() as u64) as usize;
        reader.read_exact(&mut buffer[..count])?;
        let chunks = count.div_ceil(HASH_CHUNK_BYTES);
        if chunks == 1 {
            digests[0] = Sha256::digest(&buffer[..count]).into();
        } else {
            digests[..chunks]
                .par_iter_mut()
                .zip(buffer[..count].par_chunks(HASH_CHUNK_BYTES))
                .for_each(|(digest, bytes)| *digest = Sha256::digest(bytes).into());
        }
        for digest in &digests[..chunks] {
            hash.update(digest);
        }
        remaining -= count as u64;
    }
    if reader.read(&mut [0u8; 1])? != 0 {
        return Err(invalid("Variant input grew during content hashing"));
    }
    Ok(())
}

pub(super) struct VariantPlan {
    pub weights: Vec<f32>,
    pub corrections: Vec<f32>,
    pub columns: Vec<u32>,
    pub offsets: Vec<u64>,
    pub baseline: Vec<f64>,
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
    ) -> io::Result<Option<Self>> {
        // Remote text and PVAR adapters retain their streaming compiler. This
        // cache represents local BIM rows, whose numbering is content-defined.
        if filesets
            .iter()
            .any(|f| f.bim.extension().is_none_or(|e| e != "bim") || !f.bim.is_file())
        {
            return Ok(None);
        }
        // Very wide input collections keep the streaming compiler: hashing
        // gigabytes only to exceed the bounded plan cache would waste a pass.
        let mut input_bytes = 0u64;
        for path in filesets.iter().map(|f| &f.bim).chain(scores) {
            input_bytes = input_bytes.saturating_add(std::fs::metadata(path)?.len());
            if input_bytes > 1024 * 1024 * 1024 {
                return Ok(None);
            }
        }
        let Some(directory) = dirs::cache_dir() else {
            return Ok(None);
        };
        let (_, available) = crate::memory::memory_bytes();
        let chunks = rayon::current_num_threads()
            .min(MAX_HASH_CHUNKS)
            .min((available / 64 / HASH_CHUNK_BYTES as u64).min(MAX_HASH_CHUNKS as u64) as usize)
            .min(input_bytes.div_ceil(HASH_CHUNK_BYTES as u64).max(1) as usize);
        if chunks == 0 {
            return Ok(None);
        }
        let mut buffer = Vec::new();
        buffer
            .try_reserve_exact(chunks * HASH_CHUNK_BYTES)
            .map_err(|_| invalid("Cannot allocate content hash window"))?;
        buffer.resize(chunks * HASH_CHUNK_BYTES, 0);
        let mut hash = Sha256::new();
        hash.update(MAGIC);
        // Compiler changes invalidate plans automatically, including changes
        // to chromosome parsing and shared representation invariants.
        hash.update(Sha256::digest(include_bytes!("prepare.rs")));
        hash.update(Sha256::digest(include_bytes!("prepare_cache.rs")));
        hash.update(Sha256::digest(include_bytes!("types.rs")));
        for paths in [
            filesets.iter().map(|f| f.bim.as_path()).collect::<Vec<_>>(),
            scores.iter().map(PathBuf::as_path).collect(),
        ] {
            hash.update((paths.len() as u64).to_le_bytes());
            for path in paths {
                let mut file = File::open(path)?;
                let len = file.metadata()?.len();
                hash_contents(&mut file, len, &mut buffer, &mut hash)?;
            }
        }
        let mut filters: Vec<_> = regions.into_iter().flat_map(|r| r.iter()).collect();
        filters.sort_unstable_by(|a, b| a.0.cmp(b.0));
        for (name, region) in filters {
            hash.update((name.len() as u64).to_le_bytes());
            hash.update(name.as_bytes());
            hash.update([region.chromosome]);
            hash.update(region.start.to_le_bytes());
            hash.update(region.end.to_le_bytes());
        }
        let key: [u8; 32] = hash.finalize().into();
        let name: String = key.iter().map(|byte| format!("{byte:02x}")).collect();
        Ok(Some(Self {
            path: directory.join("gnomon/variant-plans").join(name),
            key,
        }))
    }

    pub fn same_inputs(&self, other: &Self) -> bool {
        self.key == other.key
    }

    pub fn load(&self) -> io::Result<Option<VariantPlan>> {
        let file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let budget = allocation_budget();
        let len = usize::try_from(file.metadata()?.len())
            .map_err(|_| invalid("Cache length overflow"))?;
        if len < MAGIC.len() + 64 || len > budget {
            return Err(invalid(
                "Variant plan exceeds the cache memory budget or is truncated",
            ));
        }
        // Writers publish with atomic rename; this mapping stays attached to
        // one complete inode even when another process replaces the cache.
        let mapped = unsafe { MmapOptions::new().map(&file)? };
        let (payload, digest) = mapped.split_at(len - 32);
        if Sha256::digest(payload).as_slice() != digest {
            return Err(invalid("Variant plan checksum mismatch"));
        }
        let mut reader = Decoder {
            bytes: payload,
            budget,
        };
        if reader.take(MAGIC.len())? != MAGIC || reader.take(32)? != self.key {
            return Err(invalid("Variant plan identity mismatch"));
        }
        let total_variants = reader.number::<u64>()?;
        let starts = reader.vector::<u64>()?;
        let weights = reader.vector::<f32>()?;
        let corrections = reader.vector::<f32>()?;
        let columns = reader.vector::<u32>()?;
        let offsets = reader.vector::<u64>()?;
        let baseline = reader.vector::<f64>()?;
        let required = reader.items(8, |r| Ok(BimRowIndex(r.number::<u64>()?)))?;
        let flags = reader.vector::<u8>()?;
        let counts = reader.vector::<u32>()?;
        let names = reader.items(8, Decoder::string)?;
        let complex = reader.items(28, |r| {
            let locus_chr_pos = (r.string()?, r.number::<u32>()?);
            let possible_contexts = r.items(24, |r| {
                Ok((BimRowIndex(r.number::<u64>()?), r.string()?, r.string()?))
            })?;
            let score_applications = r.items(24, |r| {
                Ok(ScoreInfo {
                    effect_allele: r.string()?,
                    other_allele: r.string()?,
                    weight: r.number::<f32>()?,
                    score_column_index: ScoreColumnIndex(
                        usize::try_from(r.number::<u64>()?)
                            .map_err(|_| invalid("Score column overflow"))?,
                    ),
                })
            })?;
            Ok(GroupedComplexRule {
                locus_chr_pos,
                possible_contexts,
                score_applications,
            })
        })?;
        if !reader.bytes.is_empty() {
            return Err(invalid("Trailing variant plan data"));
        }
        let plan = VariantPlan {
            weights,
            corrections,
            columns,
            offsets,
            baseline,
            required,
            complex,
            names,
            counts,
            flags,
            starts,
            total_variants,
        };
        plan.validate()?;
        Ok(Some(plan))
    }

    pub fn save(&self, prep: &PreparationResult) -> io::Result<()> {
        crate::output::write_atomically(&self.path, |writer| {
            let mut out = Encoder {
                writer,
                hash: Sha256::new(),
                remaining: allocation_budget().saturating_sub(32),
            };
            out.bytes(MAGIC)?;
            out.bytes(&self.key)?;
            out.number(prep.total_variants_in_bim)?;
            match &prep.pipeline_kind {
                PipelineKind::SingleFile(_) => out.vector(&[0u64])?,
                PipelineKind::MultiFile(boundaries) => {
                    out.number(boundaries.len() as u64)?;
                    for boundary in boundaries {
                        out.number(boundary.starting_global_index)?;
                    }
                }
            }
            out.vector(prep.sparse_weights())?;
            out.vector(prep.sparse_missing_corrections())?;
            out.vector(prep.sparse_score_columns())?;
            out.vector(prep.sparse_row_offsets())?;
            out.vector(prep.baseline_missing_sum_by_score())?;
            out.number(prep.required_bim_indices.len() as u64)?;
            for row in &prep.required_bim_indices {
                out.number(row.0)?;
            }
            out.vector(prep.required_is_complex())?;
            out.vector(&prep.score_variant_counts)?;
            out.number(prep.score_names.len() as u64)?;
            for name in &prep.score_names {
                out.string(name)?;
            }
            out.number(prep.complex_rules.len() as u64)?;
            for rule in &prep.complex_rules {
                out.string(&rule.locus_chr_pos.0)?;
                out.number(rule.locus_chr_pos.1)?;
                out.number(rule.possible_contexts.len() as u64)?;
                for (row, a, b) in &rule.possible_contexts {
                    out.number(row.0)?;
                    out.string(a)?;
                    out.string(b)?;
                }
                out.number(rule.score_applications.len() as u64)?;
                for score in &rule.score_applications {
                    out.string(&score.effect_allele)?;
                    out.string(&score.other_allele)?;
                    out.number(score.weight)?;
                    out.number(score.score_column_index.0 as u64)?;
                }
            }
            let digest = out.hash.finalize();
            out.writer.write_all(&digest)
        })
    }
}

fn allocation_budget() -> usize {
    let (_, available) = crate::memory::memory_bytes();
    usize::try_from(available / 8)
        .unwrap_or(usize::MAX)
        .min(MAX_CACHE_BYTES)
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

impl VariantPlan {
    fn validate(&self) -> io::Result<()> {
        let rows = self.required.len();
        let scores = self.names.len();
        if scores == 0
            || rows == 0
            || self.weights.len() != self.corrections.len()
            || self.weights.len() != self.columns.len()
            || self.offsets.len() != rows + 1
            || self.flags.len() != rows
            || self.counts.len() != scores
            || self.baseline.len() != scores
            || self.offsets.first() != Some(&0)
            || self.offsets.last() != Some(&(self.weights.len() as u64))
            || self.offsets.windows(2).any(|w| w[0] > w[1])
            || self.columns.iter().any(|&c| c as usize >= scores)
            || self.required.windows(2).any(|w| w[0].0 >= w[1].0)
            || self.required.last().unwrap().0 >= self.total_variants
            || self.flags.iter().any(|&f| f > 1)
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

trait Wire: Copy {
    const WIDTH: usize;
    fn encode(self, bytes: &mut [u8]);
    fn decode(bytes: &[u8]) -> Self;
}
macro_rules! wire {
    ($($ty:ty),*) => { $(impl Wire for $ty {
        const WIDTH: usize = std::mem::size_of::<Self>();
        fn encode(self, bytes: &mut [u8]) { bytes.copy_from_slice(&self.to_le_bytes()); }
        fn decode(bytes: &[u8]) -> Self { Self::from_le_bytes(bytes.try_into().unwrap()) }
    })* };
}
wire!(u8, u32, u64, f32, f64);

struct Encoder<'a, W> {
    writer: &'a mut W,
    hash: Sha256,
    remaining: usize,
}
impl<W: Write> Encoder<'_, W> {
    fn bytes(&mut self, bytes: &[u8]) -> io::Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(bytes.len())
            .ok_or_else(|| invalid("Variant plan exceeds cache size ceiling"))?;
        self.hash.update(bytes);
        self.writer.write_all(bytes)
    }
    fn number<T: Wire>(&mut self, value: T) -> io::Result<()> {
        let mut bytes = [0u8; 8];
        value.encode(&mut bytes[..T::WIDTH]);
        self.bytes(&bytes[..T::WIDTH])
    }
    fn vector<T: Wire>(&mut self, values: &[T]) -> io::Result<()> {
        self.number(values.len() as u64)?;
        let mut bytes = [0u8; 64 * 1024];
        for chunk in values.chunks(bytes.len() / T::WIDTH) {
            for (value, slot) in chunk.iter().zip(bytes.chunks_exact_mut(T::WIDTH)) {
                value.encode(slot);
            }
            self.bytes(&bytes[..chunk.len() * T::WIDTH])?;
        }
        Ok(())
    }
    fn string(&mut self, text: &str) -> io::Result<()> {
        self.number(text.len() as u64)?;
        self.bytes(text.as_bytes())
    }
}

struct Decoder<'a> {
    bytes: &'a [u8],
    budget: usize,
}
impl<'a> Decoder<'a> {
    fn take(&mut self, count: usize) -> io::Result<&'a [u8]> {
        let (head, tail) = self
            .bytes
            .split_at_checked(count)
            .ok_or_else(|| invalid("Truncated variant plan"))?;
        self.bytes = tail;
        Ok(head)
    }
    fn number<T: Wire>(&mut self) -> io::Result<T> {
        Ok(T::decode(self.take(T::WIDTH)?))
    }
    fn length(&mut self, minimum_wire_bytes: usize, heap_bytes: usize) -> io::Result<usize> {
        let count = usize::try_from(self.number::<u64>()?)
            .map_err(|_| invalid("Variant plan count overflow"))?;
        if count > self.bytes.len() / minimum_wire_bytes {
            return Err(invalid("Impossible variant plan count"));
        }
        let charge = count
            .checked_mul(heap_bytes)
            .ok_or_else(|| invalid("Variant plan allocation overflow"))?;
        self.budget = self
            .budget
            .checked_sub(charge)
            .ok_or_else(|| invalid("Variant plan allocation exceeds memory ceiling"))?;
        Ok(count)
    }
    fn vector<T: Wire>(&mut self) -> io::Result<Vec<T>> {
        self.items(T::WIDTH, Self::number::<T>)
    }
    fn items<T>(
        &mut self,
        minimum_wire_bytes: usize,
        mut parse: impl FnMut(&mut Self) -> io::Result<T>,
    ) -> io::Result<Vec<T>> {
        let count = self.length(minimum_wire_bytes, std::mem::size_of::<T>())?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count)
            .map_err(|e| invalid(&format!("Variant plan allocation failed: {e}")))?;
        for _ in 0..count {
            result.push(parse(self)?);
        }
        Ok(result)
    }
    fn string(&mut self) -> io::Result<String> {
        let bytes = self.vector::<u8>()?;
        String::from_utf8(bytes).map_err(|_| invalid("Invalid UTF-8 in variant plan"))
    }
}

pub(super) fn cache_error(error: io::Error) -> PrepError {
    PrepError::Invariant(format!("Compiled variant plan: {error}"))
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

    use std::path::Path;

    #[test]
    fn chunk_hash_is_independent_of_window_and_short_reads() {
        struct ShortReads<'a>(&'a [u8]);
        impl Read for ShortReads<'_> {
            fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
                let count = buffer.len().min(1237);
                self.0.read(&mut buffer[..count])
            }
        }
        let mut data = vec![0u8; HASH_CHUNK_BYTES * 5 + 37];
        for (index, byte) in data.iter_mut().enumerate() {
            *byte = (index % 251) as u8;
        }
        let mut expected = Sha256::new();
        expected.update((data.len() as u64).to_le_bytes());
        for chunk in data.chunks(HASH_CHUNK_BYTES) {
            expected.update(Sha256::digest(chunk));
        }
        let expected = expected.finalize();
        for chunks in 1..=MAX_HASH_CHUNKS {
            let mut buffer = vec![0; chunks * HASH_CHUNK_BYTES];
            let mut hash = Sha256::new();
            hash_contents(
                &mut ShortReads(&data),
                data.len() as u64,
                &mut buffer,
                &mut hash,
            )
            .unwrap();
            assert_eq!(hash.finalize(), expected);
        }
        let mut buffer = vec![0; HASH_CHUNK_BYTES];
        for index in [0, HASH_CHUNK_BYTES - 1, HASH_CHUNK_BYTES, data.len() - 1] {
            data[index] ^= 1;
            let mut hash = Sha256::new();
            hash_contents(
                &mut data.as_slice(),
                data.len() as u64,
                &mut buffer,
                &mut hash,
            )
            .unwrap();
            assert_ne!(hash.finalize(), expected);
            data[index] ^= 1;
        }
        for len in [data.len() - 1, data.len() + 1] {
            assert!(
                hash_contents(
                    &mut data.as_slice(),
                    len as u64,
                    &mut buffer,
                    &mut Sha256::new()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn content_key_detects_same_size_same_timestamp_edits_and_regions() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = fixture(dir.path());
        let original = PlanCache::discover(&files, &scores, None).unwrap().unwrap();
        let modified = std::fs::metadata(&files[0].bim)
            .unwrap()
            .modified()
            .unwrap();
        std::fs::write(&files[0].bim, "1 a 0 100 A C\n").unwrap();
        File::options()
            .write(true)
            .open(&files[0].bim)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(modified))
            .unwrap();
        let changed = PlanCache::discover(&files, &scores, None).unwrap().unwrap();
        assert!(!original.same_inputs(&changed));
        std::fs::write(&files[0].bim, "1 a 0 100 A G\n").unwrap();
        assert!(
            original.same_inputs(&PlanCache::discover(&files, &scores, None).unwrap().unwrap())
        );
        let text = std::fs::read_to_string(&scores[0])
            .unwrap()
            .replace("0.25", "0.75");
        std::fs::write(&scores[0], text).unwrap();
        assert!(
            !original.same_inputs(&PlanCache::discover(&files, &scores, None).unwrap().unwrap())
        );
        let filtered = HashMap::from([(
            "S".into(),
            GenomicRegion {
                chromosome: 1,
                start: 90,
                end: 110,
            },
        )]);
        assert!(
            !PlanCache::discover(&files, &scores, None)
                .unwrap()
                .unwrap()
                .same_inputs(
                    &PlanCache::discover(&files, &scores, Some(&filtered))
                        .unwrap()
                        .unwrap()
                )
        );
    }

    #[test]
    fn binary_numbers_preserve_bits_including_signed_zero_and_subnormals() {
        let values =
            [0u32, 0x8000_0000, 1, 0x7f80_0000, 0x7fc0_1234, 0xff7f_ffff].map(f32::from_bits);
        let mut bytes = Vec::new();
        Encoder {
            writer: &mut bytes,
            hash: Sha256::new(),
            remaining: 1024,
        }
        .vector(&values)
        .unwrap();
        let decoded = Decoder {
            bytes: &bytes,
            budget: 1024,
        }
        .vector::<f32>()
        .unwrap();
        assert_eq!(
            values.map(f32::to_bits).as_slice(),
            decoded.iter().map(|f| f.to_bits()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn hostile_counts_and_heap_expansion_are_rejected_before_allocation() {
        let bytes = u64::MAX.to_le_bytes();
        assert!(
            Decoder {
                bytes: &bytes,
                budget: 64
            }
            .vector::<u64>()
            .is_err()
        );
        let mut bytes = 4u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0; 32]);
        // Four empty strings occupy only 32 wire bytes, but 96 heap bytes.
        assert!(
            Decoder {
                bytes: &bytes,
                budget: 64
            }
            .items(8, Decoder::string)
            .is_err()
        );
    }

    #[test]
    fn cache_roundtrip_rejects_corruption_and_truncation() {
        let dir = tempfile::tempdir().unwrap();
        let (files, scores) = fixture(dir.path());
        let (prep, _) = super::super::prepare_for_computation_with_retry(
            &[dir.path().join("panel")],
            &scores,
            None,
            None,
            1,
        )
        .unwrap();
        let mut cache = PlanCache::discover(&files, &scores, None).unwrap().unwrap();
        cache.path = dir.path().join("plan");
        cache.save(&prep).unwrap();
        let plan = cache.load().unwrap().unwrap();
        assert_eq!(plan.required, prep.required_bim_indices);
        assert_eq!(plan.weights, prep.sparse_weights());
        assert_eq!(plan.corrections, prep.sparse_missing_corrections());
        assert_eq!(plan.baseline, prep.baseline_missing_sum_by_score());
        assert_eq!(plan.counts, prep.score_variant_counts);
        let mut bytes = std::fs::read(&cache.path).unwrap();
        bytes[MAGIC.len() + 40] ^= 1;
        std::fs::write(&cache.path, &bytes).unwrap();
        assert!(cache.load().is_err());
        std::fs::write(&cache.path, &bytes[..20]).unwrap();
        assert!(cache.load().is_err());
    }
}
