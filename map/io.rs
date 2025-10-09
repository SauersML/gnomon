use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::str::{self, FromStr};
use std::sync::{Arc, OnceLock};

use crate::map::fit::{HwePcaModel, VariantBlockSource};
use crate::map::project::ProjectionResult;
use crate::map::variant_filter::{VariantFilter, VariantKey, VariantSelection};
use crate::score::pipeline::PipelineError;
use crate::shared::files::{
    BedSource, TextSource, VariantCompression, VariantFormat, list_variant_paths, open_bed_source,
    open_text_source, open_variant_source,
};
use noodles_bcf::io::Reader as BcfReader;
use noodles_bgzf::io::Reader as BgzfReader;
use noodles_vcf::io::Reader as VcfReader;
use noodles_vcf::{
    self as vcf,
    variant::RecordBuf,
    variant::record::samples::keys::key,
    variant::record_buf::samples::sample::{
        self, Value,
        value::{Array, genotype::Genotype as SampleGenotype},
    },
};
use thiserror::Error;

const PLINK_HEADER_LEN: u64 = 3;

type DynBcfReader = BcfReader<Box<dyn Read + Send>>;
type DynVcfReader = VcfReader<Box<dyn BufRead + Send>>;

enum VariantStreamReader {
    Bcf(DynBcfReader),
    Vcf(DynVcfReader),
}

impl VariantStreamReader {
    fn read_header(&mut self) -> io::Result<vcf::Header> {
        match self {
            Self::Bcf(reader) => reader.read_header(),
            Self::Vcf(reader) => reader.read_header(),
        }
    }

    fn read_record_buf(
        &mut self,
        header: &vcf::Header,
        record: &mut RecordBuf,
    ) -> io::Result<usize> {
        match self {
            Self::Bcf(reader) => reader.read_record_buf(header, record),
            Self::Vcf(reader) => reader.read_record_buf(header, record),
        }
    }
}

#[derive(Debug, Error)]
pub enum PlinkIoError {
    #[error("pipeline I/O error: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("invalid PLINK .bed header: {0}")]
    InvalidHeader(String),
    #[error("unexpected end of .bed payload (expected {expected} bytes, found {actual})")]
    TruncatedBed { expected: u64, actual: u64 },
    #[error("malformed record in {path} at line {line}: {message}")]
    MalformedRecord {
        path: String,
        line: usize,
        message: String,
    },
    #[error("{path} is not valid UTF-8: {source}")]
    Utf8 {
        path: String,
        #[source]
        source: str::Utf8Error,
    },
}

#[derive(Debug, Error)]
pub enum VariantIoError {
    #[error("pipeline I/O error: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Variant dataset is missing sample names")]
    MissingSamples,
    #[error("Variant dataset contains no variant records")]
    NoVariants,
    #[error("Variant decode error: {0}")]
    Decode(String),
    #[error("Variant stream header did not match initial header when reopening")]
    HeaderMismatch,
    #[error("unexpected end of variant stream (expected {expected} variants, read {actual})")]
    UnexpectedEof { expected: usize, actual: usize },
}

#[derive(Debug, Error)]
pub enum GenotypeIoError {
    #[error(transparent)]
    Plink(#[from] PlinkIoError),
    #[error(transparent)]
    Variant(#[from] VariantIoError),
}

#[derive(Debug, Error)]
pub enum DatasetOutputError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("{0}")]
    InvalidState(String),
}

#[derive(Clone, Debug)]
pub struct SampleRecord {
    pub family_id: String,
    pub individual_id: String,
    pub paternal_id: String,
    pub maternal_id: String,
    pub sex: String,
    pub phenotype: String,
}

#[derive(Debug)]
pub enum GenotypeDataset {
    Plink(PlinkDataset),
    Variants(VcfLikeDataset),
}

impl GenotypeDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GenotypeIoError> {
        let path = path.as_ref();
        if guess_is_variant_dataset(path) {
            Ok(Self::Variants(VcfLikeDataset::open(path)?))
        } else {
            Ok(Self::Plink(PlinkDataset::open(path)?))
        }
    }

    pub fn samples(&self) -> &[SampleRecord] {
        match self {
            Self::Plink(dataset) => dataset.samples(),
            Self::Variants(dataset) => dataset.samples(),
        }
    }

    pub fn n_samples(&self) -> usize {
        match self {
            Self::Plink(dataset) => dataset.n_samples(),
            Self::Variants(dataset) => dataset.n_samples(),
        }
    }

    pub fn n_variants(&self) -> usize {
        match self {
            Self::Plink(dataset) => dataset.n_variants(),
            Self::Variants(dataset) => dataset.n_variants(),
        }
    }

    pub fn block_source(&self) -> Result<DatasetBlockSource, GenotypeIoError> {
        self.block_source_with_selection(None)
    }

    pub fn block_source_with_selection(
        &self,
        selection: Option<&[usize]>,
    ) -> Result<DatasetBlockSource, GenotypeIoError> {
        match self {
            Self::Plink(dataset) => Ok(DatasetBlockSource::Plink(
                dataset.block_source_with_selection(selection.map(|indices| indices.to_vec())),
            )),
            Self::Variants(dataset) => Ok(DatasetBlockSource::Variants(
                dataset.block_source_with_selection(selection.map(|indices| indices.to_vec()))?,
            )),
        }
    }

    pub fn select_variants(
        &self,
        filter: &VariantFilter,
    ) -> Result<VariantSelection, GenotypeIoError> {
        match self {
            Self::Plink(dataset) => dataset
                .select_variants(filter)
                .map_err(GenotypeIoError::from),
            Self::Variants(dataset) => dataset
                .select_variants(filter)
                .map_err(GenotypeIoError::from),
        }
    }

    pub fn select_variants_by_keys(
        &self,
        keys: &[VariantKey],
    ) -> Result<VariantSelection, GenotypeIoError> {
        let filter = VariantFilter::from_keys(keys.to_vec());
        self.select_variants(&filter)
    }

    pub fn data_path(&self) -> &Path {
        match self {
            Self::Plink(dataset) => dataset.bed_path(),
            Self::Variants(dataset) => dataset.input_path(),
        }
    }

    pub fn output_path(&self, filename: &str) -> PathBuf {
        match self {
            Self::Plink(dataset) => dataset.output_path(filename),
            Self::Variants(dataset) => dataset.output_path(filename),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProjectionOutputPaths {
    pub scores: PathBuf,
    pub alignment: Option<PathBuf>,
}

pub fn save_hwe_model(
    dataset: &GenotypeDataset,
    model: &HwePcaModel,
) -> Result<PathBuf, DatasetOutputError> {
    let model_path = dataset.output_path("hwe.json");
    prepare_output_path(&model_path)?;

    let file = File::create(&model_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, model)?;
    writer.flush()?;

    Ok(model_path)
}

pub fn load_hwe_model(dataset: &GenotypeDataset) -> Result<HwePcaModel, DatasetOutputError> {
    let model_path = dataset.output_path("hwe.json");
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let model: HwePcaModel = serde_json::from_reader(reader)?;

    if let Some(keys) = model.variant_keys() {
        if keys.len() != model.n_variants() {
            return Err(DatasetOutputError::InvalidState(format!(
                "Model stores {} variant keys but reports {} variants",
                keys.len(),
                model.n_variants()
            )));
        }
    } else if model.n_variants() != dataset.n_variants() {
        return Err(DatasetOutputError::InvalidState(format!(
            "Model expects {} variants but dataset provides {}",
            model.n_variants(),
            dataset.n_variants()
        )));
    }

    Ok(model)
}

pub fn save_projection_results(
    dataset: &GenotypeDataset,
    result: &ProjectionResult,
) -> Result<ProjectionOutputPaths, DatasetOutputError> {
    let scores = result.scores.as_ref();
    let samples = dataset.samples();

    if samples.len() != scores.nrows() {
        return Err(DatasetOutputError::InvalidState(format!(
            "Projection scores contain {} rows but dataset has {} samples",
            scores.nrows(),
            samples.len()
        )));
    }

    let scores_path = dataset.output_path("projection.scores.tsv");
    prepare_output_path(&scores_path)?;
    let mut writer = BufWriter::new(File::create(&scores_path)?);

    write!(writer, "FID\tIID")?;
    for idx in 0..scores.ncols() {
        write!(writer, "\tPC{}", idx + 1)?;
    }
    writeln!(writer)?;

    for (row, sample) in samples.iter().enumerate() {
        write!(writer, "{}\t{}", sample.family_id, sample.individual_id)?;
        for col in 0..scores.ncols() {
            let value = scores[(row, col)];
            write!(writer, "\t{}", value)?;
        }
        writeln!(writer)?;
    }

    writer.flush()?;

    let mut alignment_path = None;
    if let Some(alignment) = result.alignment.as_ref() {
        let path = dataset.output_path("projection.alignment.tsv");
        prepare_output_path(&path)?;
        let mut writer = BufWriter::new(File::create(&path)?);

        write!(writer, "FID\tIID")?;
        for idx in 0..alignment.ncols() {
            write!(writer, "\tPC{}", idx + 1)?;
        }
        writeln!(writer)?;

        for (row, sample) in samples.iter().enumerate() {
            write!(writer, "{}\t{}", sample.family_id, sample.individual_id)?;
            for col in 0..alignment.ncols() {
                let value = alignment[(row, col)];
                write!(writer, "\t{}", value)?;
            }
            writeln!(writer)?;
        }

        writer.flush()?;
        alignment_path = Some(path);
    }

    Ok(ProjectionOutputPaths {
        scores: scores_path,
        alignment: alignment_path,
    })
}

pub fn save_sample_manifest(dataset: &GenotypeDataset) -> Result<PathBuf, DatasetOutputError> {
    let manifest_path = dataset.output_path("samples.tsv");
    prepare_output_path(&manifest_path)?;
    let mut writer = BufWriter::new(File::create(&manifest_path)?);

    writeln!(writer, "FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE")?;

    for record in dataset.samples() {
        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}",
            record.family_id,
            record.individual_id,
            record.paternal_id,
            record.maternal_id,
            record.sex,
            record.phenotype
        )?;
    }

    writer.flush()?;
    Ok(manifest_path)
}

pub fn save_fit_summary(
    dataset: &GenotypeDataset,
    model: &HwePcaModel,
) -> Result<PathBuf, DatasetOutputError> {
    let summary_path = dataset.output_path("hwe.summary.tsv");
    prepare_output_path(&summary_path)?;
    let mut writer = BufWriter::new(File::create(&summary_path)?);

    writeln!(writer, "metric\tvalue")?;
    writeln!(writer, "n_samples\t{}", model.n_samples())?;
    writeln!(writer, "n_variants\t{}", model.n_variants())?;

    for (idx, variance) in model.explained_variance().iter().copied().enumerate() {
        writeln!(writer, "explained_variance_PC{}\t{}", idx + 1, variance)?;
    }

    let ratios = model.explained_variance_ratio();
    for (idx, ratio) in ratios.into_iter().enumerate() {
        writeln!(writer, "explained_variance_ratio_PC{}\t{}", idx + 1, ratio)?;
    }

    writer.flush()?;
    Ok(summary_path)
}

fn prepare_output_path(path: &Path) -> Result<(), io::Error> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[derive(Debug)]
pub enum DatasetBlockSource {
    Plink(PlinkVariantBlockSource),
    Variants(VcfLikeVariantBlockSource),
}

impl VariantBlockSource for DatasetBlockSource {
    type Error = GenotypeIoError;

    fn n_samples(&self) -> usize {
        match self {
            Self::Plink(source) => source.n_samples(),
            Self::Variants(source) => source.n_samples(),
        }
    }

    fn n_variants(&self) -> usize {
        match self {
            Self::Plink(source) => source.n_variants(),
            Self::Variants(source) => source.n_variants(),
        }
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        match self {
            Self::Plink(source) => source.reset().map_err(GenotypeIoError::from),
            Self::Variants(source) => source.reset().map_err(GenotypeIoError::from),
        }
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        match self {
            Self::Plink(source) => source
                .next_block_into(max_variants, storage)
                .map_err(GenotypeIoError::from),
            Self::Variants(source) => source
                .next_block_into(max_variants, storage)
                .map_err(GenotypeIoError::from),
        }
    }
}

#[derive(Debug)]
pub struct PlinkDataset {
    bed: BedSource,
    bed_path: PathBuf,
    bim_path: PathBuf,
    fam_path: PathBuf,
    samples: Vec<SampleRecord>,
    n_variants: usize,
    bytes_per_variant: usize,
}

impl PlinkDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, PlinkIoError> {
        let bed_path = normalize_path(path.as_ref(), "bed");
        let bim_path = bed_path.with_extension("bim");
        let fam_path = bed_path.with_extension("fam");

        let bed = open_bed_source(&bed_path)?;
        let mut header = [0u8; PLINK_HEADER_LEN as usize];
        bed.read_at(0, &mut header)?;
        validate_bed_header(&header)?;

        let samples = read_fam_records(&fam_path)?;
        if samples.is_empty() {
            return Err(PlinkIoError::MalformedRecord {
                path: fam_path.display().to_string(),
                line: 0,
                message: "no samples found in .fam".to_string(),
            });
        }
        let n_samples = samples.len();
        let bytes_per_variant = ((n_samples + 3) / 4).max(1);

        let n_variants = count_bim_records(&bim_path)?;
        if n_variants == 0 {
            return Err(PlinkIoError::MalformedRecord {
                path: bim_path.display().to_string(),
                line: 0,
                message: "no variants found in .bim".to_string(),
            });
        }

        let expected_payload = (bytes_per_variant as u64)
            .checked_mul(n_variants as u64)
            .ok_or_else(|| PlinkIoError::TruncatedBed {
                expected: u64::MAX,
                actual: bed.len(),
            })?;
        let actual_payload = bed.len().saturating_sub(PLINK_HEADER_LEN);
        if actual_payload != expected_payload {
            return Err(PlinkIoError::TruncatedBed {
                expected: expected_payload,
                actual: actual_payload,
            });
        }

        Ok(Self {
            bed,
            bed_path,
            bim_path,
            fam_path,
            samples,
            n_variants,
            bytes_per_variant,
        })
    }

    pub fn samples(&self) -> &[SampleRecord] {
        &self.samples
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn n_variants(&self) -> usize {
        self.n_variants
    }

    pub fn bed_path(&self) -> &Path {
        &self.bed_path
    }

    pub fn bim_path(&self) -> &Path {
        &self.bim_path
    }

    pub fn fam_path(&self) -> &Path {
        &self.fam_path
    }

    pub fn variant_records(&self) -> Result<PlinkVariantRecordIter, PlinkIoError> {
        let source = open_text_source(&self.bim_path)?;
        Ok(PlinkVariantRecordIter::new(self.bim_path.clone(), source))
    }

    pub fn into_block_source(self) -> PlinkVariantBlockSource {
        PlinkVariantBlockSource::new(
            self.bed,
            self.bytes_per_variant,
            self.samples.len(),
            self.n_variants,
            None,
        )
    }

    pub fn block_source(&self) -> PlinkVariantBlockSource {
        self.block_source_with_selection(None)
    }

    pub fn block_source_with_selection(
        &self,
        selection: Option<Vec<usize>>,
    ) -> PlinkVariantBlockSource {
        PlinkVariantBlockSource::new(
            self.bed.clone(),
            self.bytes_per_variant,
            self.samples.len(),
            self.n_variants,
            selection,
        )
    }

    pub fn select_variants(
        &self,
        filter: &VariantFilter,
    ) -> Result<VariantSelection, PlinkIoError> {
        use std::collections::HashSet;

        let mut iter = self.variant_records()?;
        let mut indices = Vec::new();
        let mut keys = Vec::new();
        let mut matched = HashSet::new();
        let mut index = 0usize;

        while let Some(result) = iter.next() {
            let record = result?;
            let position =
                record
                    .position
                    .parse::<u64>()
                    .map_err(|err| PlinkIoError::MalformedRecord {
                        path: iter.path.display().to_string(),
                        line: iter.line,
                        message: format!(
                            "invalid position '{}' for variant {}: {err}",
                            record.position, record.identifier
                        ),
                    })?;

            let key = VariantKey::new(&record.chromosome, position);
            if filter.contains(&key) && matched.insert(key.clone()) {
                indices.push(index);
                keys.push(key);
            }
            index += 1;
        }

        let missing = filter.missing_keys(&matched);
        Ok(VariantSelection {
            indices,
            keys,
            missing,
            requested_unique: filter.requested_unique(),
        })
    }

    pub fn output_path(&self, filename: &str) -> PathBuf {
        if is_remote_path(&self.bed_path) {
            let stem = self
                .bed_path
                .file_stem()
                .map(|s| s.to_os_string())
                .unwrap_or_else(|| OsString::from("dataset"));
            let mut local = PathBuf::from(stem);
            local.set_extension(filename);
            local
        } else {
            let mut local = self.bed_path.clone();
            local.set_extension(filename);
            local
        }
    }
}

pub struct PlinkVariantRecordIter {
    path: PathBuf,
    reader: Box<dyn TextSource>,
    line: usize,
}

impl PlinkVariantRecordIter {
    fn new(path: PathBuf, reader: Box<dyn TextSource>) -> Self {
        Self {
            path,
            reader,
            line: 0,
        }
    }
}

impl fmt::Debug for PlinkVariantRecordIter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlinkVariantRecordIter")
            .field("path", &self.path)
            .field("line", &self.line)
            .finish()
    }
}

#[derive(Clone, Debug)]
pub struct PlinkVariantRecord {
    pub chromosome: String,
    pub identifier: String,
    pub genetic_distance: String,
    pub position: String,
    pub allele1: String,
    pub allele2: String,
}

impl Iterator for PlinkVariantRecordIter {
    type Item = Result<PlinkVariantRecord, PlinkIoError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.reader.next_line() {
                Ok(Some(line)) => {
                    self.line += 1;
                    if line.iter().all(u8::is_ascii_whitespace) {
                        continue;
                    }
                    let path = self.path.display().to_string();
                    let text = match str::from_utf8(line) {
                        Ok(s) => s,
                        Err(err) => return Some(Err(PlinkIoError::Utf8 { path, source: err })),
                    };
                    let mut fields = text.split_whitespace();
                    let rec = match (
                        fields.next(),
                        fields.next(),
                        fields.next(),
                        fields.next(),
                        fields.next(),
                        fields.next(),
                    ) {
                        (Some(chr), Some(id), Some(cm), Some(pos), Some(a1), Some(a2)) => {
                            PlinkVariantRecord {
                                chromosome: chr.to_string(),
                                identifier: id.to_string(),
                                genetic_distance: cm.to_string(),
                                position: pos.to_string(),
                                allele1: a1.to_string(),
                                allele2: a2.to_string(),
                            }
                        }
                        _ => {
                            return Some(Err(PlinkIoError::MalformedRecord {
                                path,
                                line: self.line,
                                message: "expected 6 whitespace-delimited fields".to_string(),
                            }));
                        }
                    };
                    return Some(Ok(rec));
                }
                Ok(None) => return None,
                Err(err) => return Some(Err(err.into())),
            }
        }
    }
}

#[derive(Debug)]
pub struct PlinkVariantBlockSource {
    bed: BedSource,
    bytes_per_variant: usize,
    n_samples: usize,
    total_variants: usize,
    selection: Option<Vec<usize>>,
    cursor: usize,
    buffer: Vec<u8>,
}

impl PlinkVariantBlockSource {
    fn new(
        bed: BedSource,
        bytes_per_variant: usize,
        n_samples: usize,
        n_variants: usize,
        selection: Option<Vec<usize>>,
    ) -> Self {
        Self {
            bed,
            bytes_per_variant,
            n_samples,
            total_variants: n_variants,
            selection,
            cursor: 0,
            buffer: Vec::new(),
        }
    }
}

impl VariantBlockSource for PlinkVariantBlockSource {
    type Error = PlinkIoError;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.selection
            .as_ref()
            .map(|indices| indices.len())
            .unwrap_or(self.total_variants)
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.cursor = 0;
        Ok(())
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        if let Some(selection) = &self.selection {
            let remaining = selection.len().saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(0);
            }
            let ncols = remaining.min(max_variants);
            let slice = &selection[self.cursor..self.cursor + ncols];
            let table = decode_table();
            let nrows = self.n_samples;
            let mut emitted = 0usize;

            while emitted < ncols {
                let current_index = slice[emitted];
                if current_index >= self.total_variants {
                    return Err(PlinkIoError::InvalidHeader(format!(
                        "variant index {current_index} exceeds dataset bounds ({})",
                        self.total_variants
                    )));
                }

                let mut run = 1usize;
                while emitted + run < ncols && slice[emitted + run] == current_index + run {
                    run += 1;
                }

                let block_bytes = (self.bytes_per_variant as u64)
                    .checked_mul(run as u64)
                    .ok_or_else(|| PlinkIoError::TruncatedBed {
                        expected: u64::MAX,
                        actual: self.bed.len(),
                    })?;

                let offset = PLINK_HEADER_LEN
                    .checked_add((current_index as u64) * (self.bytes_per_variant as u64))
                    .ok_or_else(|| PlinkIoError::TruncatedBed {
                        expected: u64::MAX,
                        actual: self.bed.len(),
                    })?;

                let end =
                    offset
                        .checked_add(block_bytes)
                        .ok_or_else(|| PlinkIoError::TruncatedBed {
                            expected: u64::MAX,
                            actual: self.bed.len(),
                        })?;
                if end > self.bed.len() {
                    return Err(PlinkIoError::TruncatedBed {
                        expected: block_bytes,
                        actual: self.bed.len().saturating_sub(offset),
                    });
                }

                let needed = block_bytes as usize;
                self.buffer.resize(needed, 0);
                self.bed.read_at(offset, &mut self.buffer[..])?;

                for local in 0..run {
                    let bytes_start = local * self.bytes_per_variant;
                    let bytes_end = bytes_start + self.bytes_per_variant;
                    let bytes = &self.buffer[bytes_start..bytes_end];
                    let dest_offset = (emitted + local) * nrows;
                    let dest = &mut storage[dest_offset..dest_offset + nrows];
                    decode_plink_variant(bytes, dest, nrows, table);
                }

                emitted += run;
            }

            self.cursor += ncols;
            Ok(ncols)
        } else {
            let remaining = self.total_variants.saturating_sub(self.cursor);
            if remaining == 0 {
                return Ok(0);
            }

            let ncols = remaining.min(max_variants);
            let block_bytes = (self.bytes_per_variant as u64)
                .checked_mul(ncols as u64)
                .ok_or_else(|| PlinkIoError::TruncatedBed {
                    expected: u64::MAX,
                    actual: self.bed.len(),
                })?;

            let offset = PLINK_HEADER_LEN
                .checked_add((self.cursor as u64) * (self.bytes_per_variant as u64))
                .ok_or_else(|| PlinkIoError::TruncatedBed {
                    expected: u64::MAX,
                    actual: self.bed.len(),
                })?;

            let end =
                offset
                    .checked_add(block_bytes)
                    .ok_or_else(|| PlinkIoError::TruncatedBed {
                        expected: u64::MAX,
                        actual: self.bed.len(),
                    })?;
            if end > self.bed.len() {
                return Err(PlinkIoError::TruncatedBed {
                    expected: block_bytes,
                    actual: self.bed.len().saturating_sub(offset),
                });
            }

            let needed = block_bytes as usize;
            self.buffer.resize(needed, 0);

            self.bed.read_at(offset, &mut self.buffer[..])?;

            let table = decode_table();
            let nrows = self.n_samples;
            for variant_idx in 0..ncols {
                let start = variant_idx * self.bytes_per_variant;
                let end = start + self.bytes_per_variant;
                let bytes = &self.buffer[start..end];
                let dest_offset = variant_idx * nrows;
                let dest = &mut storage[dest_offset..dest_offset + nrows];
                decode_plink_variant(bytes, dest, nrows, table);
            }

            self.cursor += ncols;
            Ok(ncols)
        }
    }
}

#[derive(Debug)]
pub struct VcfLikeDataset {
    input_path: PathBuf,
    parts: Vec<PathBuf>,
    sample_names: Arc<Vec<String>>,
    samples: Vec<SampleRecord>,
    n_variants: usize,
    output_hint: OutputHint,
}

#[derive(Clone, Debug)]
enum OutputHint {
    LocalFile(PathBuf),
    LocalDirectory(PathBuf),
    RemoteStem(OsString),
}

impl VcfLikeDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, VariantIoError> {
        let path = path.as_ref();
        let input_path = path.to_path_buf();
        let output_hint = compute_output_hint(path);

        let parts = list_variant_paths(path).map_err(VariantIoError::from)?;

        let mut expected_sample_names: Option<Vec<String>> = None;
        let mut samples: Option<Vec<SampleRecord>> = None;
        let mut record = RecordBuf::default();
        let mut n_variants = 0usize;

        for part in &parts {
            let (mut reader, compression, format) =
                create_variant_reader_for_file(part).map_err(|err| {
                    print_variant_diagnostics(
                        part,
                        None,
                        None,
                        "initializing variant reader",
                        &err,
                    );
                    err
                })?;
            let header = reader.read_header().map_err(|err| {
                print_variant_diagnostics(
                    part,
                    Some(compression),
                    Some(format),
                    "reading variant header",
                    &err,
                );
                VariantIoError::Io(err)
            })?;

            if expected_sample_names.is_none() {
                let sample_names: Vec<String> = header.sample_names().iter().cloned().collect();
                if sample_names.is_empty() {
                    return Err(VariantIoError::MissingSamples);
                }
                let sample_records = sample_names
                    .iter()
                    .map(|name| SampleRecord {
                        family_id: name.clone(),
                        individual_id: name.clone(),
                        paternal_id: "0".to_string(),
                        maternal_id: "0".to_string(),
                        sex: "0".to_string(),
                        phenotype: "-9".to_string(),
                    })
                    .collect::<Vec<_>>();
                expected_sample_names = Some(sample_names);
                samples = Some(sample_records);
            } else {
                let expected = expected_sample_names
                    .as_ref()
                    .expect("sample names initialized");
                let observed = header.sample_names();
                if observed.len() != expected.len()
                    || !observed.iter().zip(expected.iter()).all(|(a, b)| a == b)
                {
                    return Err(VariantIoError::HeaderMismatch);
                }
            }

            loop {
                let bytes = reader
                    .read_record_buf(&header, &mut record)
                    .map_err(|err| {
                        print_variant_diagnostics(
                            part,
                            Some(compression),
                            Some(format),
                            "scanning variant records to count variants",
                            &err,
                        );
                        VariantIoError::Io(err)
                    })?;
                if bytes == 0 {
                    break;
                }
                n_variants += 1;
            }
        }

        if n_variants == 0 {
            return Err(VariantIoError::NoVariants);
        }

        let expected_sample_names = expected_sample_names.expect("sample names captured");
        let samples = samples.expect("sample records captured");

        Ok(Self {
            input_path,
            parts,
            sample_names: Arc::new(expected_sample_names),
            samples,
            n_variants,
            output_hint,
        })
    }

    pub fn samples(&self) -> &[SampleRecord] {
        &self.samples
    }

    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    pub fn n_variants(&self) -> usize {
        self.n_variants
    }

    pub fn input_path(&self) -> &Path {
        &self.input_path
    }

    pub fn block_source(&self) -> Result<VcfLikeVariantBlockSource, VariantIoError> {
        self.block_source_with_selection(None)
    }

    pub fn block_source_with_selection(
        &self,
        selection: Option<Vec<usize>>,
    ) -> Result<VcfLikeVariantBlockSource, VariantIoError> {
        VcfLikeVariantBlockSource::new(
            self.parts.clone(),
            Arc::clone(&self.sample_names),
            self.n_variants,
            selection,
        )
    }

    pub fn output_path(&self, filename: &str) -> PathBuf {
        match &self.output_hint {
            OutputHint::LocalFile(path) => {
                let mut local = path.clone();
                local.set_extension(filename);
                local
            }
            OutputHint::LocalDirectory(dir) => dir.join(filename),
            OutputHint::RemoteStem(stem) => {
                let mut local = PathBuf::from(stem);
                local.set_extension(filename);
                local
            }
        }
    }

    pub fn select_variants(
        &self,
        filter: &VariantFilter,
    ) -> Result<VariantSelection, VariantIoError> {
        use std::collections::HashSet;

        let mut indices = Vec::new();
        let mut keys = Vec::new();
        let mut matched = HashSet::new();
        let mut record_idx = 0usize;
        let mut record = RecordBuf::default();

        for part in &self.parts {
            let (mut reader, compression, format) =
                create_variant_reader_for_file(part).map_err(|err| {
                    print_variant_diagnostics(
                        part,
                        None,
                        None,
                        "initializing variant reader for variant list",
                        &err,
                    );
                    err
                })?;
            let header = reader.read_header().map_err(|err| {
                print_variant_diagnostics(
                    part,
                    Some(compression),
                    Some(format),
                    "reading variant header",
                    &err,
                );
                VariantIoError::Io(err)
            })?;

            loop {
                let bytes = reader
                    .read_record_buf(&header, &mut record)
                    .map_err(|err| {
                        print_variant_diagnostics(
                            part,
                            Some(compression),
                            Some(format),
                            "scanning variant records for variant list",
                            &err,
                        );
                        VariantIoError::Io(err)
                    })?;
                if bytes == 0 {
                    break;
                }

                let chrom = record.reference_sequence_name().to_string();
                if let Some(position) = record.variant_start() {
                    let pos = position.get() as u64;
                    let key = VariantKey::new(&chrom, pos);
                    if filter.contains(&key) && matched.insert(key.clone()) {
                        indices.push(record_idx);
                        keys.push(key);
                    }
                }
                record_idx += 1;
            }
        }

        let missing = filter.missing_keys(&matched);
        Ok(VariantSelection {
            indices,
            keys,
            missing,
            requested_unique: filter.requested_unique(),
        })
    }
}

pub struct VcfLikeVariantBlockSource {
    parts: Vec<PathBuf>,
    part_idx: usize,
    reader: Option<VariantStreamReader>,
    compression: Option<VariantCompression>,
    format: Option<VariantFormat>,
    header: Option<Arc<vcf::Header>>,
    record: RecordBuf,
    sample_names: Arc<Vec<String>>,
    n_samples: usize,
    total_variants: usize,
    filtered_variants: usize,
    selection: Option<Vec<usize>>,
    emitted: usize,
    processed: usize,
}

impl std::fmt::Debug for VcfLikeVariantBlockSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VcfLikeVariantBlockSource")
            .field("current_part", &self.parts.get(self.part_idx))
            .field("n_samples", &self.n_samples)
            .field("total_variants", &self.total_variants)
            .field("filtered_variants", &self.filtered_variants)
            .field("emitted", &self.emitted)
            .field("processed", &self.processed)
            .field("compression", &self.compression)
            .field("format", &self.format)
            .finish()
    }
}

impl VcfLikeVariantBlockSource {
    fn new(
        parts: Vec<PathBuf>,
        sample_names: Arc<Vec<String>>,
        n_variants: usize,
        selection: Option<Vec<usize>>,
    ) -> Result<Self, VariantIoError> {
        let n_samples = sample_names.len();
        let filtered_variants = selection
            .as_ref()
            .map(|indices| indices.len())
            .unwrap_or(n_variants);
        let mut source = Self {
            parts,
            part_idx: 0,
            reader: None,
            compression: None,
            format: None,
            header: None,
            record: RecordBuf::default(),
            sample_names,
            n_samples,
            total_variants: n_variants,
            filtered_variants,
            selection,
            emitted: 0,
            processed: 0,
        };
        if !source.parts.is_empty() {
            source.open_part(0)?;
        }
        Ok(source)
    }

    fn open_part(&mut self, idx: usize) -> Result<(), VariantIoError> {
        self.part_idx = idx;
        if idx >= self.parts.len() {
            self.reader = None;
            self.compression = None;
            self.format = None;
            self.header = None;
            return Ok(());
        }

        let path = &self.parts[idx];
        let (mut reader, compression, format) =
            create_variant_reader_for_file(path).map_err(|err| {
                print_variant_diagnostics(path, None, None, "initializing variant reader", &err);
                err
            })?;
        let header = reader.read_header().map_err(|err| {
            print_variant_diagnostics(
                path,
                Some(compression),
                Some(format),
                "reading variant header",
                &err,
            );
            VariantIoError::Io(err)
        })?;

        let observed = header.sample_names();
        if observed.len() != self.sample_names.len()
            || !observed
                .iter()
                .zip(self.sample_names.iter())
                .all(|(a, b)| a == b)
        {
            return Err(VariantIoError::HeaderMismatch);
        }

        self.reader = Some(reader);
        self.compression = Some(compression);
        self.format = Some(format);
        self.header = Some(Arc::new(header));
        self.record = RecordBuf::default();
        Ok(())
    }
}

impl VariantBlockSource for VcfLikeVariantBlockSource {
    type Error = VariantIoError;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.filtered_variants
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.emitted = 0;
        self.processed = 0;
        if self.parts.is_empty() {
            self.reader = None;
            self.compression = None;
            self.format = None;
            self.header = None;
            return Ok(());
        }
        self.open_part(0)
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        if self.emitted >= self.filtered_variants {
            return Ok(0);
        }
        let mut filled = 0usize;
        if self.selection.is_some() {
            let target_total = self
                .selection
                .as_ref()
                .map(|indices| indices.len())
                .unwrap_or(0);
            while filled < max_variants && self.emitted + filled < target_total {
                if self.reader.is_none() {
                    break;
                }

                let header = match self.header.as_ref() {
                    Some(header) => Arc::clone(header),
                    None => break,
                };
                let reader = match self.reader.as_mut() {
                    Some(reader) => reader,
                    None => break,
                };

                let compression = self.compression;
                let format = self.format;
                let path = self.parts.get(self.part_idx).cloned();
                let bytes = reader
                    .read_record_buf(header.as_ref(), &mut self.record)
                    .map_err(|err| {
                        if let Some(part_path) = path.as_ref() {
                            print_variant_diagnostics(
                                part_path,
                                compression,
                                format,
                                "reading variant record block",
                                &err,
                            );
                        }
                        VariantIoError::Io(err)
                    })?;

                if bytes == 0 {
                    let next_idx = self.part_idx + 1;
                    self.open_part(next_idx)?;
                    continue;
                }

                let current_index = self.processed;
                self.processed += 1;
                let target_index = self.selection.as_ref().expect("selection must be present")
                    [self.emitted + filled];

                if current_index < target_index {
                    continue;
                }
                if current_index > target_index {
                    return Err(VariantIoError::Decode(format!(
                        "variant index {target_index} requested by PCA list not found in dataset",
                    )));
                }

                let offset = filled * self.n_samples;
                let dest = &mut storage[offset..offset + self.n_samples];
                decode_variant_record(&self.record, dest, self.n_samples)?;
                filled += 1;
            }

            self.emitted += filled;
            if filled == 0 && self.emitted < target_total {
                return Err(VariantIoError::UnexpectedEof {
                    expected: target_total,
                    actual: self.emitted,
                });
            }

            Ok(filled)
        } else {
            while filled < max_variants && self.emitted + filled < self.filtered_variants {
                if self.reader.is_none() {
                    break;
                }

                let header = match self.header.as_ref() {
                    Some(header) => Arc::clone(header),
                    None => break,
                };
                let reader = match self.reader.as_mut() {
                    Some(reader) => reader,
                    None => break,
                };

                let compression = self.compression;
                let format = self.format;
                let path = self.parts.get(self.part_idx).cloned();
                let bytes = reader
                    .read_record_buf(header.as_ref(), &mut self.record)
                    .map_err(|err| {
                        if let Some(part_path) = path.as_ref() {
                            print_variant_diagnostics(
                                part_path,
                                compression,
                                format,
                                "reading variant record block",
                                &err,
                            );
                        }
                        VariantIoError::Io(err)
                    })?;

                if bytes == 0 {
                    let next_idx = self.part_idx + 1;
                    self.open_part(next_idx)?;
                    continue;
                }

                let offset = filled * self.n_samples;
                let dest = &mut storage[offset..offset + self.n_samples];
                decode_variant_record(&self.record, dest, self.n_samples)?;
                filled += 1;
                self.processed += 1;
            }

            self.emitted += filled;
            if filled == 0 && self.emitted < self.filtered_variants {
                return Err(VariantIoError::UnexpectedEof {
                    expected: self.filtered_variants,
                    actual: self.emitted,
                });
            }

            Ok(filled)
        }
    }
}

fn guess_is_variant_dataset(path: &Path) -> bool {
    if is_remote_path(path) {
        let lower = path.to_string_lossy().to_ascii_lowercase();
        lower.ends_with(".bcf")
            || lower.ends_with(".vcf")
            || lower.ends_with(".vcf.gz")
            || lower.ends_with(".vcf.bgz")
            || lower.ends_with(".bgz")
            || lower.ends_with("/*")
            || lower.ends_with('/')
    } else if path.is_dir() {
        true
    } else {
        let lower = path.to_string_lossy().to_ascii_lowercase();
        lower.ends_with(".bcf")
            || lower.ends_with(".vcf")
            || lower.ends_with(".vcf.gz")
            || lower.ends_with(".vcf.bgz")
    }
}

fn compute_output_hint(path: &Path) -> OutputHint {
    if is_remote_path(path) {
        let mut trimmed = path.to_string_lossy().into_owned();
        while trimmed.ends_with('*') {
            trimmed.pop();
        }
        while trimmed.ends_with('/') {
            trimmed.pop();
        }
        if trimmed.is_empty() {
            return OutputHint::RemoteStem(OsString::from("dataset"));
        }
        let stem = Path::new(&trimmed)
            .file_stem()
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| OsString::from("dataset"));
        OutputHint::RemoteStem(stem)
    } else if path.is_dir() {
        OutputHint::LocalDirectory(path.to_path_buf())
    } else {
        OutputHint::LocalFile(path.to_path_buf())
    }
}

fn create_variant_reader_for_file(
    path: &Path,
) -> Result<(VariantStreamReader, VariantCompression, VariantFormat), VariantIoError> {
    let source = open_variant_source(path)?;
    let compression = source.compression();
    let format = source.format();

    let reader = match format {
        VariantFormat::Bcf => {
            let reader: Box<dyn Read + Send> = match compression {
                VariantCompression::Plain => Box::new(source),
                VariantCompression::Bgzf => Box::new(BgzfReader::new(source)),
            };
            VariantStreamReader::Bcf(BcfReader::from(reader))
        }
        VariantFormat::Vcf => {
            let reader: Box<dyn BufRead + Send> = match compression {
                VariantCompression::Plain => Box::new(BufReader::new(source)),
                VariantCompression::Bgzf => Box::new(BgzfReader::new(source)),
            };
            VariantStreamReader::Vcf(VcfReader::new(reader))
        }
    };

    Ok((reader, compression, format))
}

fn print_variant_diagnostics(
    path: &Path,
    compression: Option<VariantCompression>,
    format: Option<VariantFormat>,
    stage: &str,
    err: &(dyn std::error::Error + '_),
) {
    let location = if is_remote_path(path) {
        "remote (GCS)"
    } else if path.is_dir() {
        "local directory"
    } else {
        "local file"
    };

    eprintln!("Variant diagnostics:");
    eprintln!("  • Path        : {}", path.display());
    eprintln!("  • Location    : {location}");
    match format {
        Some(kind) => eprintln!("  • Format      : {:?}", kind),
        None => eprintln!("  • Format      : unknown (detected before reader initialization)"),
    }
    match compression {
        Some(mode) => eprintln!("  • Compression : {:?}", mode),
        None => eprintln!("  • Compression : unknown (detected before reader initialization)"),
    }
    eprintln!("  • Stage       : {stage}");
    eprintln!("  • Underlying error: {err}");

    if let Some(VariantCompression::Bgzf) = compression {
        if err
            .to_string()
            .to_lowercase()
            .contains("invalid bgzf header")
        {
            eprintln!(
                "  • Hint        : The BGZF stream appears corrupt or truncated; ensure the source is a complete BGZF-compressed .bcf file."
            );
        }
    }
}

fn decode_variant_record(
    record: &RecordBuf,
    dest: &mut [f64],
    n_samples: usize,
) -> Result<(), VariantIoError> {
    if dest.len() < n_samples {
        return Err(VariantIoError::Decode(
            "destination buffer shorter than number of samples".to_string(),
        ));
    }

    dest[..n_samples].fill(f64::NAN);

    let samples = record.samples();
    let ds_series = samples.select("DS");
    let gt_series = samples.select(key::GENOTYPE);

    for sample_idx in 0..n_samples {
        if let Some(series) = ds_series.as_ref() {
            match series.get(sample_idx) {
                Some(Some(value)) => {
                    if let Some(dosage) = numeric_from_value(value)? {
                        dest[sample_idx] = dosage;
                        continue;
                    }
                }
                Some(None) => {
                    dest[sample_idx] = f64::NAN;
                    continue;
                }
                None => {}
            }
        }

        if let Some(series) = gt_series.as_ref() {
            match series.get(sample_idx) {
                Some(Some(Value::Genotype(genotype))) => {
                    if let Some(dosage) = dosage_from_genotype(genotype.as_ref())? {
                        dest[sample_idx] = dosage;
                        continue;
                    }
                }
                Some(Some(Value::String(text))) => {
                    let genotype = SampleGenotype::from_str(text).map_err(|err| {
                        VariantIoError::Decode(format!(
                            "failed to parse genotype string '{text}': {err}"
                        ))
                    })?;
                    if let Some(dosage) = dosage_from_genotype(genotype.as_ref())? {
                        dest[sample_idx] = dosage;
                        continue;
                    }
                }
                Some(Some(value)) => {
                    if let Some(dosage) = numeric_from_value(value)? {
                        dest[sample_idx] = dosage;
                        continue;
                    }
                }
                Some(None) => {
                    dest[sample_idx] = f64::NAN;
                    continue;
                }
                None => {}
            }
        }
    }

    Ok(())
}

fn numeric_from_value(value: &Value) -> Result<Option<f64>, VariantIoError> {
    match value {
        Value::Integer(n) => Ok(Some(*n as f64)),
        Value::Float(n) => Ok(Some(*n as f64)),
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                trimmed.parse::<f64>().map(Some).map_err(|err| {
                    VariantIoError::Decode(format!(
                        "failed to parse numeric string '{trimmed}': {err}"
                    ))
                })
            }
        }
        Value::Array(Array::Integer(values)) => {
            Ok(values.get(0).copied().flatten().map(|n| n as f64))
        }
        Value::Array(Array::Float(values)) => {
            Ok(values.get(0).copied().flatten().map(|n| n as f64))
        }
        Value::Array(_) => Ok(None),
        Value::Genotype(genotype) => dosage_from_genotype(genotype.as_ref()),
        Value::Character(_) => Ok(None),
    }
}

fn dosage_from_genotype(
    genotype: &[sample::value::genotype::Allele],
) -> Result<Option<f64>, VariantIoError> {
    let mut total = 0.0f64;
    for allele in genotype {
        match allele.position() {
            Some(0) => {}
            Some(_) => total += 1.0,
            None => return Ok(None),
        }
    }
    Ok(Some(total))
}

fn decode_plink_variant(bytes: &[u8], dest: &mut [f64], n_samples: usize, table: &[[f64; 4]; 256]) {
    let mut sample_idx = 0usize;
    for &byte in bytes {
        if sample_idx >= n_samples {
            break;
        }
        let decoded = &table[byte as usize];
        let remaining = n_samples - sample_idx;
        let take = remaining.min(4);
        dest[sample_idx..sample_idx + take].copy_from_slice(&decoded[..take]);
        sample_idx += take;
    }
}

fn decode_table() -> &'static [[f64; 4]; 256] {
    static TABLE: OnceLock<[[f64; 4]; 256]> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = [[0.0f64; 4]; 256];
        for byte in 0u16..256 {
            for offset in 0..4 {
                let code = ((byte >> (offset * 2)) & 0b11) as u8;
                table[byte as usize][offset] = match code {
                    0 => 0.0,
                    1 => f64::NAN,
                    2 => 1.0,
                    3 => 2.0,
                    _ => unreachable!(),
                };
            }
        }
        table
    })
}

fn normalize_path(path: &Path, extension: &str) -> PathBuf {
    if path.extension().map_or(false, |ext| ext == extension) {
        path.to_owned()
    } else {
        path.with_extension(extension)
    }
}

fn validate_bed_header(header: &[u8]) -> Result<(), PlinkIoError> {
    match header {
        [0x6c, 0x1b, 0x01] => Ok(()),
        [0x6c, 0x1b, mode] => Err(PlinkIoError::InvalidHeader(format!(
            "unsupported mode byte {mode:#04x} (only variant-major mode is supported)"
        ))),
        _ => Err(PlinkIoError::InvalidHeader(
            "missing PLINK magic bytes 0x6c 0x1b".to_string(),
        )),
    }
}

fn read_fam_records(path: &Path) -> Result<Vec<SampleRecord>, PlinkIoError> {
    let mut reader = open_text_source(path)?;
    let mut records = Vec::new();
    let mut line_no = 0usize;

    while let Some(line) = reader.next_line()? {
        line_no += 1;
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let text = str::from_utf8(line).map_err(|err| PlinkIoError::Utf8 {
            path: path.display().to_string(),
            source: err,
        })?;
        let mut fields = text.split_whitespace();
        let (Some(fid), Some(iid), Some(pid), Some(mid), Some(sex), Some(phenotype)) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            return Err(PlinkIoError::MalformedRecord {
                path: path.display().to_string(),
                line: line_no,
                message: "expected 6 whitespace-delimited fields".to_string(),
            });
        };

        records.push(SampleRecord {
            family_id: fid.to_string(),
            individual_id: iid.to_string(),
            paternal_id: pid.to_string(),
            maternal_id: mid.to_string(),
            sex: sex.to_string(),
            phenotype: phenotype.to_string(),
        });
    }

    Ok(records)
}

fn count_bim_records(path: &Path) -> Result<usize, PlinkIoError> {
    let mut reader = open_text_source(path)?;
    let mut count = 0usize;
    while let Some(line) = reader.next_line()? {
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        count += 1;
    }
    Ok(count)
}

fn is_remote_path(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.starts_with("gs://") || s.starts_with("http://") || s.starts_with("https://"))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shared::files::VariantSource;
    use std::fs::File;
    use std::io::{Read, Write};
    use std::path::Path;
    use tempfile::tempdir;

    fn fake_bcf_header() -> Vec<u8> {
        let header_text =
            b"##fileformat=VCFv4.3\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\n\0";
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"BCF");
        bytes.push(2);
        bytes.push(2);
        bytes.extend_from_slice(&(header_text.len() as u32).to_le_bytes());
        bytes.extend_from_slice(header_text);
        bytes
    }

    fn write_fake_bcf<P: AsRef<Path>>(path: P, payload: &[u8]) -> std::io::Result<Vec<u8>> {
        let mut file = File::create(path)?;
        let header = fake_bcf_header();
        file.write_all(&header)?;
        file.write_all(payload)?;
        Ok(header)
    }

    fn read_all(mut source: VariantSource) -> Vec<u8> {
        let mut data = Vec::new();
        source.read_to_end(&mut data).unwrap();
        data
    }

    #[test]
    fn bcf_source_reads_single_file_without_modification() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("sample.bcf");
        let header = write_fake_bcf(&path, &[0x01, 0x02, 0x03]).unwrap();

        let source = open_variant_source(&path).unwrap();
        let bytes = read_all(source);

        let mut expected = header;
        expected.extend_from_slice(&[0x01, 0x02, 0x03]);
        assert_eq!(bytes, expected);
    }

    #[test]
    fn bcf_source_skips_headers_when_concatenating_directory() {
        let dir = tempdir().unwrap();
        let files = [
            ("chr2.bcf", vec![0x20]),
            ("chr10.bcf", vec![0x10]),
            ("chr1.bcf", vec![0x01]),
        ];

        for (name, payload) in &files {
            let path = dir.path().join(name);
            write_fake_bcf(&path, payload).unwrap();
        }

        let mut expected = fake_bcf_header();
        expected.extend_from_slice(&[0x01]);
        expected.extend_from_slice(&[0x20]);
        expected.extend_from_slice(&[0x10]);

        let source = open_variant_source(dir.path()).unwrap();
        let bytes = read_all(source);

        assert_eq!(bytes, expected);
    }
}
