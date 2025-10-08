use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::str::{self, FromStr};
use std::sync::{Arc, OnceLock};

use crate::map::fit::{HwePcaModel, VariantBlockSource};
use crate::map::project::ProjectionResult;
use crate::score::pipeline::PipelineError;
use crate::shared::files::{
    BcfSource, BedSource, TextSource, open_bcf_source, open_bed_source, open_text_source,
};
use noodles_bcf::io::Reader as BcfReader;
use noodles_bgzf::io::Reader as BgzfReader;
use noodles_vcf::{
    self as vcf,
    variant::RecordBuf,
    variant::record::samples::keys::key,
    variant::record_buf::samples::sample::{
        self, Value, value::Array, value::genotype::Genotype as SampleGenotype,
    },
};
use thiserror::Error;

const PLINK_HEADER_LEN: u64 = 3;

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
pub enum BcfIoError {
    #[error("pipeline I/O error: {0}")]
    Pipeline(#[from] PipelineError),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("BCF dataset is missing sample names")]
    MissingSamples,
    #[error("BCF dataset contains no variant records")]
    NoVariants,
    #[error("BCF genotype decode error: {0}")]
    Decode(String),
    #[error("BCF stream header did not match initial header when reopening")]
    HeaderMismatch,
    #[error("unexpected end of BCF stream (expected {expected} variants, read {actual})")]
    UnexpectedEof { expected: usize, actual: usize },
}

#[derive(Debug, Error)]
pub enum GenotypeIoError {
    #[error(transparent)]
    Plink(#[from] PlinkIoError),
    #[error(transparent)]
    Bcf(#[from] BcfIoError),
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
    Bcf(BcfDataset),
}

impl GenotypeDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, GenotypeIoError> {
        let path = path.as_ref();
        if guess_is_bcf(path) {
            Ok(Self::Bcf(BcfDataset::open(path)?))
        } else {
            Ok(Self::Plink(PlinkDataset::open(path)?))
        }
    }

    pub fn samples(&self) -> &[SampleRecord] {
        match self {
            Self::Plink(dataset) => dataset.samples(),
            Self::Bcf(dataset) => dataset.samples(),
        }
    }

    pub fn n_samples(&self) -> usize {
        match self {
            Self::Plink(dataset) => dataset.n_samples(),
            Self::Bcf(dataset) => dataset.n_samples(),
        }
    }

    pub fn n_variants(&self) -> usize {
        match self {
            Self::Plink(dataset) => dataset.n_variants(),
            Self::Bcf(dataset) => dataset.n_variants(),
        }
    }

    pub fn block_source(&self) -> Result<DatasetBlockSource, GenotypeIoError> {
        match self {
            Self::Plink(dataset) => Ok(DatasetBlockSource::Plink(dataset.block_source())),
            Self::Bcf(dataset) => Ok(DatasetBlockSource::Bcf(dataset.block_source()?)),
        }
    }

    pub fn data_path(&self) -> &Path {
        match self {
            Self::Plink(dataset) => dataset.bed_path(),
            Self::Bcf(dataset) => dataset.input_path(),
        }
    }

    pub fn output_path(&self, filename: &str) -> PathBuf {
        match self {
            Self::Plink(dataset) => dataset.output_path(filename),
            Self::Bcf(dataset) => dataset.output_path(filename),
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

    if model.n_variants() != dataset.n_variants() {
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
    Bcf(BcfVariantBlockSource),
}

impl VariantBlockSource for DatasetBlockSource {
    type Error = GenotypeIoError;

    fn n_samples(&self) -> usize {
        match self {
            Self::Plink(source) => source.n_samples(),
            Self::Bcf(source) => source.n_samples(),
        }
    }

    fn n_variants(&self) -> usize {
        match self {
            Self::Plink(source) => source.n_variants(),
            Self::Bcf(source) => source.n_variants(),
        }
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        match self {
            Self::Plink(source) => source.reset().map_err(GenotypeIoError::from),
            Self::Bcf(source) => source.reset().map_err(GenotypeIoError::from),
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
            Self::Bcf(source) => source
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
        )
    }

    pub fn block_source(&self) -> PlinkVariantBlockSource {
        PlinkVariantBlockSource::new(
            self.bed.clone(),
            self.bytes_per_variant,
            self.samples.len(),
            self.n_variants,
        )
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
    n_variants: usize,
    cursor: usize,
    buffer: Vec<u8>,
}

impl PlinkVariantBlockSource {
    fn new(bed: BedSource, bytes_per_variant: usize, n_samples: usize, n_variants: usize) -> Self {
        Self {
            bed,
            bytes_per_variant,
            n_samples,
            n_variants,
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
        self.n_variants
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
        let remaining = self.n_variants.saturating_sub(self.cursor);
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

        let end = offset
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

#[derive(Debug)]
pub struct BcfDataset {
    input_path: PathBuf,
    header: Arc<vcf::Header>,
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

impl BcfDataset {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, BcfIoError> {
        let path = path.as_ref();
        let input_path = path.to_path_buf();
        let output_hint = compute_output_hint(path);

        let mut reader = create_bcf_reader(&input_path)?;
        let header = Arc::new(reader.read_header()?);
        let sample_names: Vec<String> = header.sample_names().iter().cloned().collect();
        if sample_names.is_empty() {
            return Err(BcfIoError::MissingSamples);
        }
        let samples = sample_names
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

        let mut record = RecordBuf::default();
        let mut n_variants = 0usize;
        loop {
            let bytes = reader.read_record_buf(header.as_ref(), &mut record)?;
            if bytes == 0 {
                break;
            }
            n_variants += 1;
        }

        if n_variants == 0 {
            return Err(BcfIoError::NoVariants);
        }

        Ok(Self {
            input_path,
            header,
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

    pub fn block_source(&self) -> Result<BcfVariantBlockSource, BcfIoError> {
        BcfVariantBlockSource::new(
            self.input_path.clone(),
            Arc::clone(&self.header),
            self.samples.len(),
            self.n_variants,
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
}

pub struct BcfVariantBlockSource {
    path: PathBuf,
    header: Arc<vcf::Header>,
    reader: BcfReader<BgzfReader<BcfSource>>,
    record: RecordBuf,
    n_samples: usize,
    n_variants: usize,
    emitted: usize,
}

impl fmt::Debug for BcfVariantBlockSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BcfVariantBlockSource")
            .field("path", &self.path)
            .field("n_samples", &self.n_samples)
            .field("n_variants", &self.n_variants)
            .field("emitted", &self.emitted)
            .finish()
    }
}

impl BcfVariantBlockSource {
    fn new(
        path: PathBuf,
        header: Arc<vcf::Header>,
        n_samples: usize,
        n_variants: usize,
    ) -> Result<Self, BcfIoError> {
        let mut reader = create_bcf_reader(&path)?;
        let reopened = reader.read_header()?;
        verify_header(&reopened, header.as_ref())?;
        Ok(Self {
            path,
            header,
            reader,
            record: RecordBuf::default(),
            n_samples,
            n_variants,
            emitted: 0,
        })
    }

    fn reopen(&mut self) -> Result<(), BcfIoError> {
        let mut reader = create_bcf_reader(&self.path)?;
        let header = reader.read_header()?;
        verify_header(&header, self.header.as_ref())?;
        self.reader = reader;
        self.record = RecordBuf::default();
        self.emitted = 0;
        Ok(())
    }
}

impl VariantBlockSource for BcfVariantBlockSource {
    type Error = BcfIoError;

    fn n_samples(&self) -> usize {
        self.n_samples
    }

    fn n_variants(&self) -> usize {
        self.n_variants
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        self.reopen()
    }

    fn next_block_into(
        &mut self,
        max_variants: usize,
        storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        if max_variants == 0 {
            return Ok(0);
        }
        if self.emitted >= self.n_variants {
            return Ok(0);
        }

        let capacity = self.n_variants - self.emitted;
        let target = capacity.min(max_variants);
        let mut filled = 0usize;
        while filled < target {
            let bytes = self
                .reader
                .read_record_buf(self.header.as_ref(), &mut self.record)?;
            if bytes == 0 {
                break;
            }
            let offset = filled * self.n_samples;
            let dest = &mut storage[offset..offset + self.n_samples];
            decode_bcf_record(&self.record, dest, self.n_samples)?;
            filled += 1;
        }

        self.emitted += filled;
        if filled == 0 && self.emitted < self.n_variants {
            return Err(BcfIoError::UnexpectedEof {
                expected: self.n_variants,
                actual: self.emitted,
            });
        }

        Ok(filled)
    }
}

fn guess_is_bcf(path: &Path) -> bool {
    if is_remote_path(path) {
        let lower = path.to_string_lossy().to_ascii_lowercase();
        lower.ends_with(".bcf") || lower.ends_with("/*") || lower.ends_with('/')
    } else if path.is_dir() {
        true
    } else {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("bcf"))
            .unwrap_or(false)
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

fn create_bcf_reader(path: &Path) -> Result<BcfReader<BgzfReader<BcfSource>>, BcfIoError> {
    let source = open_bcf_source(path)?;
    Ok(BcfReader::new(source))
}

fn verify_header(observed: &vcf::Header, expected: &vcf::Header) -> Result<(), BcfIoError> {
    let observed_samples = observed.sample_names();
    let expected_samples = expected.sample_names();
    if observed_samples != expected_samples {
        return Err(BcfIoError::HeaderMismatch);
    }
    Ok(())
}

fn decode_bcf_record(
    record: &RecordBuf,
    dest: &mut [f64],
    n_samples: usize,
) -> Result<(), BcfIoError> {
    if dest.len() < n_samples {
        return Err(BcfIoError::Decode(
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
                        BcfIoError::Decode(format!(
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

fn numeric_from_value(value: &Value) -> Result<Option<f64>, BcfIoError> {
    match value {
        Value::Integer(n) => Ok(Some(*n as f64)),
        Value::Float(n) => Ok(Some(*n as f64)),
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                Ok(None)
            } else {
                trimmed.parse::<f64>().map(Some).map_err(|err| {
                    BcfIoError::Decode(format!("failed to parse numeric string '{trimmed}': {err}"))
                })
            }
        }
        Value::Array(Array::Integer(values)) => {
            Ok(values.get(0).and_then(|opt| opt.map(|n| n as f64)))
        }
        Value::Array(Array::Float(values)) => {
            Ok(values.get(0).and_then(|opt| opt.map(|n| n as f64)))
        }
        Value::Array(_) => Ok(None),
        Value::Genotype(genotype) => dosage_from_genotype(genotype.as_ref()),
        Value::Character(_) => Ok(None),
    }
}

fn dosage_from_genotype(
    genotype: &[sample::value::genotype::Allele],
) -> Result<Option<f64>, BcfIoError> {
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
        .map(|s| s.starts_with("gs://"))
        .unwrap_or(false)
}
