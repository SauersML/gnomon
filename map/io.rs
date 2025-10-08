use std::path::{Path, PathBuf};
use std::str;
use std::sync::OnceLock;

use crate::map::fit::VariantBlockSource;
use crate::score::pipeline::PipelineError;
use crate::shared::files::{open_bed_source, open_text_source, BedSource, TextSource};
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
    MalformedRecord { path: String, line: usize, message: String },
    #[error("{path} is not valid UTF-8: {source}")]
    Utf8 {
        path: String,
        #[source]
        source: str::Utf8Error,
    },
}

#[derive(Clone, Debug)]
pub struct PlinkSampleRecord {
    pub family_id: String,
    pub individual_id: String,
    pub paternal_id: String,
    pub maternal_id: String,
    pub sex: String,
    pub phenotype: String,
}

#[derive(Debug)]
pub struct PlinkDataset {
    bed: BedSource,
    bed_path: PathBuf,
    bim_path: PathBuf,
    fam_path: PathBuf,
    samples: Vec<PlinkSampleRecord>,
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

    pub fn samples(&self) -> &[PlinkSampleRecord] {
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
}

#[derive(Debug)]
pub struct PlinkVariantRecordIter {
    path: PathBuf,
    reader: Box<dyn TextSource>,
    line: usize,
}

impl PlinkVariantRecordIter {
    fn new(path: PathBuf, reader: Box<dyn TextSource>) -> Self {
        Self { path, reader, line: 0 }
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
                        Err(err) => {
                            return Some(Err(PlinkIoError::Utf8 {
                                path,
                                source: err,
                            }))
                        }
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
                        (Some(chr), Some(id), Some(cm), Some(pos), Some(a1), Some(a2)) => PlinkVariantRecord {
                            chromosome: chr.to_string(),
                            identifier: id.to_string(),
                            genetic_distance: cm.to_string(),
                            position: pos.to_string(),
                            allele1: a1.to_string(),
                            allele2: a2.to_string(),
                        },
                        _ => {
                            return Some(Err(PlinkIoError::MalformedRecord {
                                path,
                                line: self.line,
                                message: "expected 6 whitespace-delimited fields".to_string(),
                            }))
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

pub struct PlinkVariantBlockSource {
    bed: BedSource,
    bytes_per_variant: usize,
    n_samples: usize,
    n_variants: usize,
    cursor: usize,
    buffer: Vec<u8>,
}

impl PlinkVariantBlockSource {
    fn new(
        bed: BedSource,
        bytes_per_variant: usize,
        n_samples: usize,
        n_variants: usize,
    ) -> Self {
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

        let end = offset.checked_add(block_bytes).ok_or_else(|| PlinkIoError::TruncatedBed {
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
            decode_variant(bytes, dest, nrows, table);
        }

        self.cursor += ncols;
        Ok(ncols)
    }
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

fn read_fam_records(path: &Path) -> Result<Vec<PlinkSampleRecord>, PlinkIoError> {
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

        records.push(PlinkSampleRecord {
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

fn decode_variant(bytes: &[u8], dest: &mut [f64], n_samples: usize, table: &[[f64; 4]; 256]) {
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
