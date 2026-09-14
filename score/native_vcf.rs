use crate::score::types::{GenomicRegion, parse_chromosome_label};
use crate::shared::files::{VariantCompression, VariantFormat, VariantSource, open_variant_source};
use ahash::{AHashMap, AHashSet};
use crossbeam_channel::{Receiver, Sender};
use flate2::Crc;
use flate2::read::MultiGzDecoder;
use libdeflater::Decompressor;
use memchr::{memchr, memchr_iter, memrchr};
use noodles_vcf::io::Reader as VcfReader;
use noodles_vcf::variant::record::AlternateBases as _;
use noodles_vcf::variant::record::samples::keys::key;
use rayon::prelude::*;
use std::error::Error;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

#[derive(Debug)]
pub struct NativeVcfScoreResult {
    pub person_iids: Vec<String>,
    pub score_names: Vec<String>,
    pub score_variant_counts: Vec<u32>,
    pub sum_scores: Vec<f64>,
    pub missing_counts: Vec<u32>,
    pub matched_variants: usize,
}

#[derive(Debug, Clone)]
struct ScoreRule {
    effect_allele: String,
    other_allele: String,
    applications: Vec<ScoreApplication>,
}

#[derive(Debug, Clone, Copy)]
struct ScoreApplication {
    score_index: usize,
    weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct MatchedRule {
    score_index: usize,
    weight: f64,
    effect_is_ref: bool,
}

type VariantKey = (u8, u32);

pub fn score_vcf_streaming(
    input_path: &Path,
    native_score_files: &[PathBuf],
    keep: Option<&Path>,
    score_regions: Option<&std::collections::HashMap<String, GenomicRegion>>,
) -> Result<NativeVcfScoreResult, Box<dyn Error + Send + Sync>> {
    let (score_names, rules_by_key) = load_score_rules(native_score_files, score_regions)?;
    let rules_by_key = Arc::new(rules_by_key);

    let source = open_variant_source(input_path)?;
    if source.format() != VariantFormat::Vcf {
        return Err(format!(
            "Native streaming score supports VCF input only; got {:?} for '{}'.",
            source.format(),
            input_path.display()
        )
        .into());
    }

    let mut reader = match source.compression() {
        VariantCompression::Plain => {
            let reader: Box<dyn BufRead + Send> = Box::new(BufReader::new(source));
            VcfReader::new(reader)
        }
        VariantCompression::Bgzf => {
            let reader: Box<dyn BufRead + Send> = Box::new(PrefilteredBgzfReader::spawn(
                source,
                Arc::clone(&rules_by_key),
            )?);
            VcfReader::new(reader)
        }
    };

    let header = reader.read_header()?;
    let all_samples: Vec<String> = header.sample_names().iter().cloned().collect();
    if all_samples.is_empty() {
        return Err("VCF contains no samples.".into());
    }

    let kept_indices = resolve_keep_indices(keep, &all_samples)?;
    let person_iids: Vec<String> = kept_indices
        .iter()
        .map(|&idx| all_samples[idx].clone())
        .collect();

    let num_people = person_iids.len();
    let num_scores = score_names.len();
    let mut sum_scores = vec![0.0f64; num_people * num_scores];
    let mut missing_counts = vec![0u32; num_people * num_scores];
    let mut score_variant_counts = vec![0u32; num_scores];

    // Records are read in order, decoded on the rayon pool, and accumulated in
    // order again, so every sum takes the same operands in the same sequence
    // as a one-record-at-a-time scan, and the first error is the one it raises.
    let threads = rayon::current_num_threads().max(1);
    let batch_len =
        (DECODE_BATCH_DOSAGES / all_samples.len()).clamp(threads, threads * RECORDS_PER_WORKER);
    let mut records: Vec<noodles_vcf::Record> = Vec::new();
    let mut decoded_records: Vec<DecodedRecord> = Vec::new();
    loop {
        let mut filled = 0usize;
        let mut read_error = None;
        let mut at_eof = false;
        while filled < batch_len {
            if records.len() == filled {
                records.push(noodles_vcf::Record::default());
                decoded_records.push(DecodedRecord::default());
            }
            match reader.read_record(&mut records[filled]) {
                Ok(0) => {
                    at_eof = true;
                    break;
                }
                Ok(_) => filled += 1,
                Err(err) => {
                    read_error = Some(err);
                    break;
                }
            }
        }

        records[..filled]
            .par_iter()
            .zip(decoded_records[..filled].par_iter_mut())
            .for_each(|(record, decoded)| {
                decoded.allele_count = 0;
                decoded.error = decode_scored_record(
                    record,
                    &rules_by_key,
                    &kept_indices,
                    &score_names,
                    decoded,
                )
                .err();
            });

        for (record, decoded) in records[..filled].iter().zip(&mut decoded_records[..filled]) {
            if let Some(err) = decoded.error.take() {
                return Err(err);
            }
            for allele in &decoded.alleles[..decoded.allele_count] {
                for (out_person_idx, dosage) in allele.dosages.iter().enumerate() {
                    match dosage {
                        Some(decoded_dosage) => {
                            for rule in &allele.matched_rules {
                                let cell = out_person_idx * num_scores + rule.score_index;
                                let effect_dosage = if rule.effect_is_ref {
                                    decoded_dosage.ref_dosage.ok_or_else(|| {
                                        ref_effect_error(
                                            &score_names[rule.score_index],
                                            record,
                                            decoded.position,
                                        )
                                    })?
                                } else {
                                    decoded_dosage.alt_dosage
                                };
                                sum_scores[cell] += rule.weight * effect_dosage;
                            }
                        }
                        None => {
                            let mut previous_score = None;
                            for rule in &allele.matched_rules {
                                if previous_score == Some(rule.score_index) {
                                    continue;
                                }
                                let cell = out_person_idx * num_scores + rule.score_index;
                                missing_counts[cell] += 1;
                                previous_score = Some(rule.score_index);
                            }
                        }
                    }
                }

                let mut previous_score = None;
                for rule in &allele.matched_rules {
                    if previous_score != Some(rule.score_index) {
                        score_variant_counts[rule.score_index] += 1;
                        previous_score = Some(rule.score_index);
                    }
                }
            }
        }

        if let Some(err) = read_error {
            return Err(err.into());
        }
        if at_eof {
            break;
        }
    }

    // These are the same matched variant/score assignments counted in each
    // score's denominator. ALT ordinals are local to a record, so hashing them
    // across records conflates distinct alternate alleles at the same position.
    let matched_variants = score_variant_counts
        .iter()
        .map(|&count| count as usize)
        .sum();
    if matched_variants == 0 {
        return Err(format!(
            "No overlapping variants were found between '{}' and the score file(s).",
            input_path.display()
        )
        .into());
    }

    Ok(NativeVcfScoreResult {
        person_iids,
        score_names,
        score_variant_counts,
        sum_scores,
        missing_counts,
        matched_variants,
    })
}

/// Records decoded per rayon worker before the ordered accumulation pass.
const RECORDS_PER_WORKER: usize = 64;
/// Dosages one decode batch may hold, so cohorts with many samples take smaller batches.
const DECODE_BATCH_DOSAGES: usize = 1 << 22;

/// What scoring one VCF record needs, decoded away from the accumulating thread.
#[derive(Default)]
struct DecodedRecord {
    position: u32,
    /// Alleles with at least one matched rule, in ALT order. Only the first
    /// `allele_count` belong to this record; the rest keep their buffers.
    alleles: Vec<DecodedAllele>,
    allele_count: usize,
    /// The first error scoring this record raises.
    error: Option<Box<dyn Error + Send + Sync>>,
}

#[derive(Default)]
struct DecodedAllele {
    matched_rules: Vec<MatchedRule>,
    /// One entry per kept person, in output order.
    dosages: Vec<Option<DecodedAltDosage>>,
}

/// Decodes the dosages `record` contributes to its matched rules into
/// `decoded`, stopping at the first error a sequential scan raises for this
/// record: a malformed position or ALT, a missing dosage FORMAT field, an
/// undecodable sample, or a REF-effect rule without a complete REF dosage.
fn decode_scored_record(
    record: &noodles_vcf::Record,
    rules_by_key: &AHashMap<VariantKey, Vec<ScoreRule>>,
    kept_indices: &[usize],
    score_names: &[String],
    decoded: &mut DecodedRecord,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let Ok(chr) = parse_chromosome_label(record.reference_sequence_name()) else {
        return Ok(());
    };
    let Some(start) = record.variant_start() else {
        return Ok(());
    };
    let pos = start?.get() as u32;
    let Some(score_rules) = rules_by_key.get(&(chr, pos)) else {
        return Ok(());
    };
    decoded.position = pos;

    let ref_allele = record.reference_bases();
    let alternate_bases = record.alternate_bases();
    let alt_alleles = alternate_bases.iter().collect::<Result<Vec<_>, _>>()?;
    for (alt_offset, alt_allele) in alt_alleles.iter().enumerate() {
        let alt_index = alt_offset + 1;
        let matched_rules = match_rules_for_allele(score_rules, ref_allele, alt_allele);
        if matched_rules.is_empty() {
            continue;
        }
        if decoded.alleles.len() == decoded.allele_count {
            decoded.alleles.push(DecodedAllele::default());
        }
        let allele = &mut decoded.alleles[decoded.allele_count];
        allele.dosages.clear();
        let ref_effect_rule = matched_rules.iter().find(|rule| rule.effect_is_ref);
        for_each_vcf_dosage_best(
            record,
            alt_index,
            alt_alleles.len(),
            kept_indices,
            |_, dosage| {
                if let Some(decoded_dosage) = dosage
                    && decoded_dosage.ref_dosage.is_none()
                    && let Some(rule) = ref_effect_rule
                {
                    return Err(
                        ref_effect_error(&score_names[rule.score_index], record, pos).into(),
                    );
                }
                allele.dosages.push(dosage);
                Ok(())
            },
        )?;
        allele.matched_rules = matched_rules;
        decoded.allele_count += 1;
    }
    Ok(())
}

fn ref_effect_error(score_name: &str, record: &noodles_vcf::Record, position: u32) -> String {
    format!(
        "Cannot score REF-effect rule for score '{}' at {}:{} without a complete REF dosage (DS requires genotype ploidy and all ALT dosages).",
        score_name,
        record.reference_sequence_name(),
        position,
    )
}

fn load_score_rules(
    native_score_files: &[PathBuf],
    score_regions: Option<&std::collections::HashMap<String, GenomicRegion>>,
) -> Result<(Vec<String>, AHashMap<VariantKey, Vec<ScoreRule>>), Box<dyn Error + Send + Sync>> {
    let headers = read_score_headers(native_score_files)?;
    let mut score_names: Vec<String> = headers
        .iter()
        .flat_map(|header| header.score_names.iter().cloned())
        .collect();
    score_names.sort();
    let score_name_to_index: AHashMap<String, usize> = score_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect();
    let mut rules_by_key: AHashMap<VariantKey, Vec<ScoreRule>> = AHashMap::new();
    let mut skipped_contigs = SkippedContigs::default();

    for header in &headers {
        let path = &header.path;
        let mut reader = open_text_reader(path)?;
        let mut line = String::new();
        let mut line_number = 0u64;

        while reader.read_line(&mut line)? != 0 {
            line_number += 1;
            let trimmed = line.trim_end();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                line.clear();
                continue;
            }

            let mut fields = trimmed.split('\t');
            let variant_id = fields.next().unwrap_or_default();
            if variant_id == "variant_id" {
                line.clear();
                continue;
            }
            let effect_allele = fields.next().unwrap_or_default();
            let other_allele = fields.next().unwrap_or_default();
            if effect_allele.is_empty() || other_allele.is_empty() {
                return Err(format!(
                    "Malformed native score row in '{}' at line {}.",
                    path.display(),
                    line_number
                )
                .into());
            }
            if other_allele == "N" {
                return Err(format!(
                    "Native score row in '{}' at line {} has unknown other_allele 'N'. Scores must provide an explicit allele pair.",
                    path.display(),
                    line_number
                )
                .into());
            }

            let mut key_parts = variant_id.splitn(2, ':');
            let chr = key_parts.next().unwrap_or_default();
            let pos = key_parts.next().unwrap_or_default();
            // Harmonized catalog files carry a handful of alt/random contigs that no
            // primary-assembly VCF can match. Skip those rows with a summary instead of
            // aborting a run over hundreds of otherwise-usable score files.
            let Ok(chr_index) = parse_chromosome_label(chr) else {
                skipped_contigs.record(path, line_number, chr);
                line.clear();
                continue;
            };
            let key = (
                chr_index,
                pos.parse::<u32>().map_err(|err| {
                    format!(
                        "Invalid position in '{}' at line {}: {}",
                        path.display(),
                        line_number,
                        err
                    )
                })?,
            );
            let mut applications = Vec::with_capacity(header.score_names.len());
            for score_name in &header.score_names {
                let weight_text = fields.next().unwrap_or_default();
                if weight_text.trim().is_empty() {
                    return Err(format!(
                        "Missing weight for score '{}' in '{}' at line {}.",
                        score_name,
                        path.display(),
                        line_number
                    )
                    .into());
                }
                let score_index = *score_name_to_index.get(score_name).ok_or_else(|| {
                    format!(
                        "Internal error: score '{}' from '{}' was not indexed.",
                        score_name,
                        path.display()
                    )
                })?;
                if let Some(regions) = score_regions
                    && let Some(region) = regions.get(score_name)
                    && !region.contains(key)
                {
                    continue;
                }
                let weight = weight_text.parse::<f32>().map_err(|err| {
                    format!(
                        "Invalid weight for score '{}' in '{}' at line {}: {}",
                        score_name,
                        path.display(),
                        line_number,
                        err
                    )
                })?;

                applications.push(ScoreApplication {
                    score_index,
                    weight,
                });
            }
            if !applications.is_empty() {
                rules_by_key.entry(key).or_default().push(ScoreRule {
                    effect_allele: effect_allele.to_string(),
                    other_allele: other_allele.to_string(),
                    applications,
                });
            }

            line.clear();
        }
    }

    skipped_contigs.report();

    Ok((score_names, rules_by_key))
}

/// Counts native score rows dropped for unsupported contigs, keeping a few examples
/// so a pipeline can find the offending rows without re-scanning every score file.
#[derive(Debug, Default)]
struct SkippedContigs {
    count: u64,
    examples: Vec<String>,
}

impl SkippedContigs {
    const MAX_EXAMPLES: usize = 5;

    fn record(&mut self, path: &Path, line_number: u64, chromosome: &str) {
        self.count += 1;
        if self.examples.len() < Self::MAX_EXAMPLES {
            self.examples.push(format!(
                "{}:{} (chromosome '{}')",
                path.display(),
                line_number,
                chromosome
            ));
        }
    }

    fn report(&self) {
        if self.count == 0 {
            return;
        }
        eprintln!(
            "> Warning: Skipped {} native score row(s) on unsupported contigs (expected 1-22, X, Y, or MT).",
            self.count
        );
        for example in &self.examples {
            eprintln!(">   - {example}");
        }
        if self.count > self.examples.len() as u64 {
            eprintln!(
                ">   ... and {} more.",
                self.count - self.examples.len() as u64
            );
        }
    }
}

#[derive(Debug)]
struct NativeScoreHeader {
    path: PathBuf,
    score_names: Vec<String>,
}

fn read_score_headers(
    native_score_files: &[PathBuf],
) -> Result<Vec<NativeScoreHeader>, Box<dyn Error + Send + Sync>> {
    let mut headers = Vec::with_capacity(native_score_files.len());
    let mut seen_names: AHashMap<String, PathBuf> = AHashMap::new();
    for path in native_score_files {
        let mut reader = open_text_reader(path)?;
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err(format!("Score file '{}' is empty.", path.display()).into());
            }
            if !line.starts_with('#') {
                break;
            }
        }

        let header: Vec<&str> = line.trim_end().split('\t').collect();
        if header.len() < 4
            || header[0] != "variant_id"
            || header[1] != "effect_allele"
            || header[2] != "other_allele"
        {
            return Err(format!(
                "Invalid native score header in '{}'; expected variant_id/effect_allele/other_allele/score.",
                path.display()
            )
            .into());
        }

        let mut score_names = Vec::with_capacity(header.len() - 3);
        for name in &header[3..] {
            if name.is_empty() {
                return Err(format!("Empty score name in '{}'.", path.display()).into());
            }
            if let Some(existing_path) = seen_names.insert((*name).to_string(), path.clone()) {
                return Err(format!(
                    "Duplicate Score ID '{}' detected!\n  File 1: '{}'\n  File 2: '{}'\nPlease ensure each score column has a unique identifier.",
                    name,
                    existing_path.display(),
                    path.display()
                )
                .into());
            }
            score_names.push((*name).to_string());
        }
        headers.push(NativeScoreHeader {
            path: path.clone(),
            score_names,
        });
    }
    Ok(headers)
}

fn match_rules_for_allele(
    rules: &[ScoreRule],
    ref_allele: &str,
    alt_allele: &str,
) -> Vec<MatchedRule> {
    let capacity = rules.iter().map(|rule| rule.applications.len()).sum();
    let mut matched = Vec::with_capacity(capacity);
    for rule in rules {
        let effect_is_ref = if rule.effect_allele == alt_allele && rule.other_allele == ref_allele {
            false
        } else if rule.effect_allele == ref_allele && rule.other_allele == alt_allele {
            true
        } else {
            continue;
        };
        for application in &rule.applications {
            matched.push(MatchedRule {
                score_index: application.score_index,
                weight: f64::from(application.weight),
                effect_is_ref,
            });
        }
    }

    matched.sort_by_key(|rule| (rule.score_index, rule.effect_is_ref));
    let mut unique_len = 0usize;
    for read_idx in 0..matched.len() {
        let rule = matched[read_idx];
        if unique_len > 0
            && matched[unique_len - 1].score_index == rule.score_index
            && matched[unique_len - 1].effect_is_ref == rule.effect_is_ref
        {
            matched[unique_len - 1].weight += rule.weight;
        } else {
            matched[unique_len] = rule;
            unique_len += 1;
        }
    }
    matched.truncate(unique_len);
    matched
}

#[derive(Debug, Clone, Copy)]
struct DecodedAltDosage {
    alt_dosage: f64,
    ref_dosage: Option<f64>,
}

fn for_each_vcf_dosage_best<F>(
    record: &noodles_vcf::Record,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    mut visit: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: FnMut(usize, Option<DecodedAltDosage>) -> Result<(), Box<dyn Error + Send + Sync>>,
{
    let samples = record.samples();
    if samples.is_empty() {
        for out_idx in 0..kept_indices.len() {
            visit(out_idx, None)?;
        }
        return Ok(());
    }

    let mut ds_index = None;
    let mut gp_index = None;
    let mut gt_index = None;
    for (idx, sample_key) in samples.keys().iter().enumerate() {
        if ds_index.is_none() && sample_key == "DS" {
            ds_index = Some(idx);
        }
        if gp_index.is_none() && sample_key == "GP" {
            gp_index = Some(idx);
        }
        if gt_index.is_none() && sample_key == key::GENOTYPE {
            gt_index = Some(idx);
        }
    }

    let gt_idx = match gt_index {
        Some(idx) => Some(idx),
        None if ds_index.is_some() || gp_index.is_some() => None,
        None => return Err("VCF record is missing GT, DS, or GP FORMAT fields.".into()),
    };
    let last_format_index = [ds_index, gp_index, gt_idx]
        .into_iter()
        .flatten()
        .max()
        .expect("at least one dosage FORMAT field was validated");

    // People are visited in output order whether the record carries exactly
    // the header's samples, fewer (the rest are missing) or more (the extra
    // columns are never decoded).
    let mut kept_cursor = 0usize;
    for (sample_idx, sample) in vcf_sample_columns(samples.as_ref()).enumerate() {
        while kept_cursor < kept_indices.len() && kept_indices[kept_cursor] < sample_idx {
            kept_cursor += 1;
        }
        if kept_cursor >= kept_indices.len() {
            break;
        }
        if kept_indices[kept_cursor] != sample_idx {
            continue;
        }

        let decoded = decode_vcf_sample(
            sample,
            ds_index,
            gp_index,
            gt_idx,
            last_format_index,
            alt_index,
            alt_count,
        )?;
        visit(kept_cursor, decoded)?;
        kept_cursor += 1;
    }

    while kept_cursor < kept_indices.len() {
        visit(kept_cursor, None)?;
        kept_cursor += 1;
    }

    Ok(())
}

/// The sample columns of a record's samples field, split exactly as noodles'
/// `Samples::iter` splits them: after the FORMAT column, on tabs, with a lone
/// '.' read as an empty sample and nothing after a trailing tab.
fn vcf_sample_columns(samples: &str) -> impl Iterator<Item = &str> {
    let mut rest = samples.split_once('\t').map_or("", |(_, columns)| columns);
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let column = match memchr(b'\t', rest.as_bytes()) {
            Some(end) => {
                let (column, tail) = rest.split_at(end);
                rest = &tail[1..];
                column
            }
            None => std::mem::take(&mut rest),
        };
        Some(if column == "." { "" } else { column })
    })
}

#[inline]
fn decode_vcf_sample(
    sample: &str,
    ds_index: Option<usize>,
    gp_index: Option<usize>,
    gt_index: Option<usize>,
    last_format_index: usize,
    alt_index: usize,
    alt_count: usize,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    let mut ds_field = None;
    let mut gp_field = None;
    let mut gt_field = None;
    // The same pieces `sample.split(':')` yields, up to the last FORMAT field used.
    let mut remaining = Some(sample);
    for idx in 0..=last_format_index {
        let Some(rest) = remaining else {
            break;
        };
        let field = match memchr(b':', rest.as_bytes()) {
            Some(end) => {
                remaining = Some(&rest[end + 1..]);
                &rest[..end]
            }
            None => {
                remaining = None;
                rest
            }
        };
        if ds_index == Some(idx) {
            ds_field = Some(field);
        }
        if gp_index == Some(idx) {
            gp_field = Some(field);
        }
        if gt_index == Some(idx) {
            gt_field = Some(field);
        }
    }

    // REF dosage complements the sum of ALL alternate allele dosages. Taking
    // ploidy minus just the matched ALT incorrectly treats every other ALT as REF.
    let ploidy = if ds_field.is_some() || gp_field.is_some() {
        gt_field.and_then(parse_vcf_genotype_ploidy)
    } else {
        None
    };
    if let Some(value) = ds_field
        && let Some(parsed) = parse_vcf_dosage_field(value, alt_index, alt_count, ploidy)?
    {
        return Ok(Some(parsed));
    }
    if let Some(value) = gp_field
        && let Some(parsed) = parse_vcf_gp(value, alt_index, alt_count, ploidy)?
    {
        return Ok(Some(parsed));
    }
    if let Some(value) = gt_field
        && let Some(parsed) = parse_vcf_genotype(value, alt_index)?
    {
        return Ok(Some(parsed));
    }
    Ok(None)
}

fn parse_vcf_dosage_field(
    field: &str,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    if field == "." {
        return Ok(None);
    }
    if alt_index == 0 || alt_index > alt_count {
        return Err("ALT allele index is out of range".into());
    }
    let mut alt_dosage = None;
    let mut total_alt_dosage = Some(0.0);
    let mut count = 0;
    for (offset, value) in field.split(',').enumerate() {
        let dosage = parse_numeric_str(value)?;
        if dosage.is_some_and(|dosage| dosage < 0.0) {
            return Err("DS dosage must be nonnegative".into());
        }
        if offset + 1 == alt_index {
            alt_dosage = dosage;
        }
        total_alt_dosage = total_alt_dosage.zip(dosage).map(|(sum, value)| sum + value);
        count += 1;
    }
    if count != alt_count {
        return Err(format!(
            "DS field has {count} values, expected {alt_count} alternate allele dosages"
        )
        .into());
    }
    let ref_dosage = ploidy
        .zip(total_alt_dosage)
        .map(|(ploidy, total)| f64::from(ploidy) - total);
    if ref_dosage.is_some_and(|dosage| dosage < -1e-6) {
        return Err("DS alternate dosages exceed genotype ploidy".into());
    }
    Ok(alt_dosage.map(|alt_dosage| DecodedAltDosage {
        alt_dosage,
        // Permit decimal rounding at the dosage boundary without a negative count.
        ref_dosage: ref_dosage.map(|dosage| dosage.max(0.0)),
    }))
}

fn parse_vcf_gp(
    field: &str,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    if field == "." {
        return Ok(None);
    }
    if alt_index == 0 || alt_index > alt_count {
        return Err(format!(
            "ALT allele index {alt_index} is out of range for {alt_count} alternate alleles"
        )
        .into());
    }

    let allele_count = alt_count.checked_add(1).ok_or("GP allele count overflow")?;
    let diploid_len = allele_count
        .checked_add(1)
        .and_then(|next| allele_count.checked_mul(next))
        .map(|n| n / 2)
        .ok_or("GP allele count overflow")?;
    let actual_len = field.split(',').count();
    let ploidy = match ploidy {
        Some(ploidy) => ploidy,
        None if actual_len == allele_count => 1,
        None if actual_len == diploid_len => 2,
        None => return Err(format!("GP field has {actual_len} values; cannot determine haploid or diploid ploidy for {alt_count} alternate alleles").into()),
    };
    let expected_len = match ploidy {
        1 => allele_count,
        2 => diploid_len,
        _ => {
            return Err(format!(
                "GP dosage decoding requires haploid or diploid genotypes, got ploidy {ploidy}"
            )
            .into());
        }
    };
    if actual_len != expected_len {
        return Err(format!("GP field has {actual_len} values, expected {expected_len} for ploidy {ploidy} and {alt_count} alternate alleles").into());
    }
    let mut dosage = 0.0f64;
    let mut ref_dosage = 0.0f64;
    let mut parts = field.split(',');
    for second in 0..allele_count {
        let first_count = if ploidy == 1 { 1 } else { second + 1 };
        for first in 0..first_count {
            let part = parts.next().expect("GP cardinality was validated");
            if part == "." {
                return Ok(None);
            }
            let probability = part.parse::<f64>()?;
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err("GP probabilities must be finite and between zero and one".into());
            }
            let copies =
                usize::from(ploidy == 2 && first == alt_index) + usize::from(second == alt_index);
            dosage += probability * copies as f64;
            let ref_copies = usize::from(ploidy == 2 && first == 0) + usize::from(second == 0);
            ref_dosage += probability * ref_copies as f64;
        }
    }
    Ok(Some(DecodedAltDosage {
        alt_dosage: dosage,
        ref_dosage: Some(ref_dosage),
    }))
}

fn parse_numeric_str(text: &str) -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "." {
        Ok(None)
    } else {
        let value = trimmed.parse::<f64>()?;
        if !value.is_finite() {
            return Err("Dosage must be finite".into());
        }
        Ok(Some(value))
    }
}

fn parse_vcf_genotype(
    field: &str,
    alt_index: usize,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    if field.is_empty() {
        return Ok(None);
    }

    let mut dosage = 0.0f64;
    let mut ref_dosage = 0.0f64;
    let mut ploidy = 0u8;
    let bytes = field.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'/' | b'|' => idx += 1,
            b'.' => return Ok(None),
            b'0'..=b'9' => {
                let start = idx;
                idx += 1;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                let allele = field[start..idx].parse::<usize>()?;
                if allele == alt_index {
                    dosage += 1.0;
                }
                if allele == 0 {
                    ref_dosage += 1.0;
                }
                ploidy = ploidy.checked_add(1).ok_or("genotype ploidy overflow")?;
            }
            other => return Err(format!("unexpected byte {other} in genotype field").into()),
        }
    }

    if ploidy == 0 {
        Ok(None)
    } else {
        Ok(Some(DecodedAltDosage {
            alt_dosage: dosage,
            ref_dosage: Some(ref_dosage),
        }))
    }
}

/// Counts the allele slots in a `GT` field, including missing (`.`) slots.
///
/// Used only to recover ploidy for dosages decoded from `DS`/`GP`; a sample can be
/// missing its genotype call and still carry a usable imputed dosage.
fn parse_vcf_genotype_ploidy(field: &str) -> Option<u8> {
    let bytes = field.as_bytes();
    let mut ploidy = 0u8;
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'/' | b'|' => idx += 1,
            b'.' => {
                ploidy = ploidy.checked_add(1)?;
                idx += 1;
            }
            b'0'..=b'9' => {
                idx += 1;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                ploidy = ploidy.checked_add(1)?;
            }
            _ => return None,
        }
    }

    (ploidy > 0).then_some(ploidy)
}

fn resolve_keep_indices(
    keep: Option<&Path>,
    sample_names: &[String],
) -> Result<Vec<usize>, Box<dyn Error + Send + Sync>> {
    let Some(path) = keep else {
        return Ok((0..sample_names.len()).collect());
    };

    let mut requested = AHashSet::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let id = line?.trim().to_string();
        if !id.is_empty() {
            requested.insert(id);
        }
    }

    let mut indices = Vec::with_capacity(requested.len());
    let mut found = AHashSet::new();
    for (idx, sample) in sample_names.iter().enumerate() {
        if requested.contains(sample) {
            indices.push(idx);
            found.insert(sample.clone());
        }
    }

    if found.len() != requested.len() {
        let mut missing: Vec<_> = requested.difference(&found).cloned().collect();
        missing.sort();
        return Err(format!(
            "Keep file contains sample IDs not present in VCF: {}",
            missing.join(", ")
        )
        .into());
    }

    Ok(indices)
}

fn open_text_reader(path: &Path) -> Result<Box<dyn BufRead>, Box<dyn Error + Send + Sync>> {
    let file = File::open(path)?;
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        let reader: Box<dyn Read> = Box::new(MultiGzDecoder::new(file));
        Ok(Box::new(BufReader::new(reader)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

const BGZF_HEADER_LEN: usize = 18;
const BGZF_TRAILER_LEN: usize = 8;
/// Largest uncompressed payload a BGZF block may carry.
const BGZF_MAX_DATA_LEN: usize = 1 << 16;
/// Blocks inflated per rayon worker before the ordered results are stitched.
const BGZF_FRAMES_PER_WORKER: usize = 64;
/// Filtered chunks buffered between the inflating thread and the scorer.
const PREFILTER_CHANNEL_DEPTH: usize = 4;

/// A BGZF VCF stream reduced to the lines the native scorer can act on.
///
/// Inflating a WGS VCF dominates native scoring, and nearly every record in it
/// sits at a position no score file mentions. This reader inflates BGZF blocks
/// on the rayon pool and drops, inside the workers, each record line that
/// `score_vcf_streaming` would parse without error and then skip. Everything
/// else (the header, the first record after it, any line the scorer could
/// reject, and every record at a scored position) reaches noodles byte for byte
/// and in file order, so scores and errors are those of a sequential read.
///
/// Bytes that do not form a well-formed BGZF block (plain gzip members,
/// truncated or corrupt blocks, trailing garbage) hand the rest of the stream,
/// unfiltered, to `MultiGzDecoder`, which is how the whole stream used to be read.
struct PrefilteredBgzfReader {
    rx: Option<Receiver<io::Result<Vec<u8>>>>,
    buf: Vec<u8>,
    pos: usize,
    finished: bool,
    producer: Option<JoinHandle<()>>,
}

impl PrefilteredBgzfReader {
    fn spawn(
        source: VariantSource,
        rules_by_key: Arc<AHashMap<VariantKey, Vec<ScoreRule>>>,
    ) -> io::Result<Self> {
        let (tx, rx) = crossbeam_channel::bounded(PREFILTER_CHANNEL_DEPTH);
        let producer = thread::Builder::new()
            .name("vcf-bgzf-prefilter".to_string())
            .spawn(move || {
                let error_tx = tx.clone();
                if let Err(err) = BgzfLineFilter::new(source, rules_by_key, tx).run() {
                    // Fails only when the scorer already stopped reading.
                    let _ = error_tx.send(Err(err));
                }
            })?;
        Ok(Self {
            rx: Some(rx),
            buf: Vec::new(),
            pos: 0,
            finished: false,
            producer: Some(producer),
        })
    }
}

impl Read for PrefilteredBgzfReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let available = self.fill_buf()?;
        let amt = available.len().min(buf.len());
        buf[..amt].copy_from_slice(&available[..amt]);
        self.consume(amt);
        Ok(amt)
    }
}

impl BufRead for PrefilteredBgzfReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        while self.pos == self.buf.len() && !self.finished {
            let rx = self
                .rx
                .as_ref()
                .expect("the receiver is only taken when the reader is dropped");
            match rx.recv() {
                // An empty chunk marks the end of the stream.
                Ok(Ok(chunk)) if chunk.is_empty() => self.finished = true,
                Ok(Ok(chunk)) => {
                    self.buf = chunk;
                    self.pos = 0;
                }
                Ok(Err(err)) => return Err(err),
                Err(_) => {
                    return Err(io::Error::other(
                        "BGZF reader thread stopped before the end of the stream",
                    ));
                }
            }
        }
        Ok(&self.buf[self.pos..])
    }

    fn consume(&mut self, amt: usize) {
        self.pos = (self.pos + amt).min(self.buf.len());
    }
}

impl Drop for PrefilteredBgzfReader {
    fn drop(&mut self) {
        // Closing the channel first unblocks a producer waiting to send.
        drop(self.rx.take());
        if let Some(producer) = self.producer.take() {
            let _ = producer.join();
        }
    }
}

/// The producing half of [`PrefilteredBgzfReader`].
struct BgzfLineFilter {
    source: VariantSource,
    rules_by_key: Arc<AHashMap<VariantKey, Vec<ScoreRule>>>,
    tx: Sender<io::Result<Vec<u8>>>,
    /// Raw frames of the batch being read, reused across batches.
    frames: Vec<Vec<u8>>,
    /// A line not yet terminated by the blocks read so far.
    carry: Vec<u8>,
    /// Kept bytes awaiting the next send.
    out: Vec<u8>,
}

enum FrameRead {
    Block,
    Eof,
    /// The bytes read are not a canonical BGZF block; they are left in the
    /// frame buffer for `MultiGzDecoder`.
    Irregular,
}

/// One inflated block whose complete lines have already been filtered.
struct BlockLines {
    /// The bytes up to and including the block's first newline, then every
    /// kept complete line, then the bytes after the block's last newline.
    bytes: Vec<u8>,
    /// Length of the leading partial line (the whole block without a newline).
    head_len: usize,
    /// Offset in `bytes` of the trailing partial line.
    tail_start: usize,
    has_newline: bool,
}

impl BgzfLineFilter {
    fn new(
        source: VariantSource,
        rules_by_key: Arc<AHashMap<VariantKey, Vec<ScoreRule>>>,
        tx: Sender<io::Result<Vec<u8>>>,
    ) -> Self {
        Self {
            source,
            rules_by_key,
            tx,
            frames: Vec::new(),
            carry: Vec::new(),
            out: Vec::new(),
        }
    }

    fn run(mut self) -> io::Result<()> {
        // The header ends at the first line that does not start with '#'.
        // Header lines and that first record always pass, so dropping records
        // can never pull a later '#' line into the header.
        let mut decompressor = Decompressor::new();
        let mut block = Vec::with_capacity(BGZF_MAX_DATA_LEN);
        let mut frame = Vec::new();
        let mut scanned = 0usize;
        'header: loop {
            match read_bgzf_frame(&mut self.source, &mut frame)? {
                FrameRead::Block => {}
                FrameRead::Eof => return self.finish(),
                FrameRead::Irregular => return self.fall_back(frame),
            }
            if inflate_bgzf_block(&frame, &mut decompressor, &mut block).is_err() {
                return self.fall_back(frame);
            }
            self.carry.extend_from_slice(&block);
            while let Some(offset) = memchr(b'\n', &self.carry[scanned..]) {
                let line_start = scanned;
                scanned += offset + 1;
                if self.carry[line_start] != b'#' {
                    self.out.extend_from_slice(&self.carry[..scanned]);
                    self.carry.drain(..scanned);
                    self.filter_carried_lines();
                    break 'header;
                }
            }
        }

        let batch_len = rayon::current_num_threads().max(1) * BGZF_FRAMES_PER_WORKER;
        loop {
            let mut count = 0usize;
            let mut end = None;
            while count < batch_len {
                if self.frames.len() == count {
                    self.frames.push(Vec::new());
                }
                match read_bgzf_frame(&mut self.source, &mut self.frames[count])? {
                    FrameRead::Block => count += 1,
                    other => {
                        end = Some(other);
                        break;
                    }
                }
            }

            let rules_by_key = &*self.rules_by_key;
            let blocks: Vec<io::Result<BlockLines>> = self.frames[..count]
                .par_iter()
                .map_init(
                    || (Decompressor::new(), Vec::with_capacity(BGZF_MAX_DATA_LEN)),
                    |(decompressor, block), frame| {
                        inflate_bgzf_block(frame, decompressor, block)?;
                        Ok(split_block_lines(block, rules_by_key))
                    },
                )
                .collect();

            for (index, result) in blocks.into_iter().enumerate() {
                match result {
                    Ok(lines) => self.stitch(lines),
                    Err(_) => {
                        let mut consumed = Vec::new();
                        for frame in &self.frames[index..count] {
                            consumed.extend_from_slice(frame);
                        }
                        if matches!(end, Some(FrameRead::Irregular)) {
                            consumed.extend_from_slice(&self.frames[count]);
                        }
                        return self.fall_back(consumed);
                    }
                }
            }

            match end {
                None => self.flush()?,
                Some(FrameRead::Block) => unreachable!("a full frame never ends a batch early"),
                Some(FrameRead::Eof) => return self.finish(),
                Some(FrameRead::Irregular) => {
                    let consumed = std::mem::take(&mut self.frames[count]);
                    return self.fall_back(consumed);
                }
            }
        }
    }

    /// Filters every complete line in `carry`, leaving only the trailing partial line.
    fn filter_carried_lines(&mut self) {
        let mut line_start = 0usize;
        while let Some(offset) = memchr(b'\n', &self.carry[line_start..]) {
            let line_end = line_start + offset;
            if !is_skippable_record(&self.carry[line_start..line_end], &self.rules_by_key) {
                self.out
                    .extend_from_slice(&self.carry[line_start..=line_end]);
            }
            line_start = line_end + 1;
        }
        self.carry.drain(..line_start);
    }

    fn stitch(&mut self, lines: BlockLines) {
        if !lines.has_newline {
            self.carry.extend_from_slice(&lines.bytes);
            return;
        }
        self.carry.extend_from_slice(&lines.bytes[..lines.head_len]);
        let line_len = self.carry.len() - 1;
        if !is_skippable_record(&self.carry[..line_len], &self.rules_by_key) {
            self.out.extend_from_slice(&self.carry);
        }
        self.carry.clear();
        self.out
            .extend_from_slice(&lines.bytes[lines.head_len..lines.tail_start]);
        self.carry
            .extend_from_slice(&lines.bytes[lines.tail_start..]);
    }

    fn send(&self, chunk: Vec<u8>) -> io::Result<()> {
        self.tx.send(Ok(chunk)).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "native VCF scorer stopped reading",
            )
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.out.is_empty() {
            return Ok(());
        }
        let chunk = std::mem::take(&mut self.out);
        self.send(chunk)
    }

    fn finish(mut self) -> io::Result<()> {
        // A final line without a newline always passes.
        let carry = std::mem::take(&mut self.carry);
        self.out.extend_from_slice(&carry);
        self.flush()?;
        self.send(Vec::new())
    }

    fn fall_back(mut self, consumed: Vec<u8>) -> io::Result<()> {
        let carry = std::mem::take(&mut self.carry);
        self.out.extend_from_slice(&carry);
        self.flush()?;
        let mut decoder = MultiGzDecoder::new(Cursor::new(consumed).chain(self.source));
        loop {
            let mut chunk = vec![0u8; BGZF_MAX_DATA_LEN];
            let len = match decoder.read(&mut chunk) {
                Ok(0) => break,
                Ok(len) => len,
                Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            };
            chunk.truncate(len);
            self.tx.send(Ok(chunk)).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "native VCF scorer stopped reading",
                )
            })?;
        }
        self.tx.send(Ok(Vec::new())).map_err(|_| {
            io::Error::new(
                io::ErrorKind::BrokenPipe,
                "native VCF scorer stopped reading",
            )
        })
    }
}

/// Reads one BGZF block into `frame`.
fn read_bgzf_frame<R: Read>(reader: &mut R, frame: &mut Vec<u8>) -> io::Result<FrameRead> {
    frame.clear();
    let header_len = read_up_to(reader, frame, BGZF_HEADER_LEN)?;
    if header_len == 0 {
        return Ok(FrameRead::Eof);
    }
    if header_len < BGZF_HEADER_LEN || !is_bgzf_header(frame) {
        return Ok(FrameRead::Irregular);
    }
    let block_len = usize::from(u16::from_le_bytes([frame[16], frame[17]])) + 1;
    if block_len < BGZF_HEADER_LEN + BGZF_TRAILER_LEN {
        return Ok(FrameRead::Irregular);
    }
    let body_len = block_len - BGZF_HEADER_LEN;
    if read_up_to(reader, frame, body_len)? < body_len {
        return Ok(FrameRead::Irregular);
    }
    Ok(FrameRead::Block)
}

/// Appends up to `len` bytes from `reader` to `dst`, stopping short only at end of input.
fn read_up_to<R: Read>(reader: &mut R, dst: &mut Vec<u8>, len: usize) -> io::Result<usize> {
    let start = dst.len();
    reader.take(len as u64).read_to_end(dst)?;
    Ok(dst.len() - start)
}

fn is_bgzf_header(header: &[u8]) -> bool {
    header[..4] == [0x1f, 0x8b, 0x08, 0x04]
        && header[10..12] == [0x06, 0x00]
        && header[12..14] == *b"BC"
        && header[14..16] == [0x02, 0x00]
}

/// Inflates one canonical BGZF block into `block`, checking its length and CRC32.
fn inflate_bgzf_block(
    frame: &[u8],
    decompressor: &mut Decompressor,
    block: &mut Vec<u8>,
) -> io::Result<()> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message.to_string());
    let (header_and_data, trailer) = frame.split_at(frame.len() - BGZF_TRAILER_LEN);
    let crc32 = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let data_len = u32::from_le_bytes([trailer[4], trailer[5], trailer[6], trailer[7]]) as usize;
    if data_len > BGZF_MAX_DATA_LEN {
        return Err(invalid("BGZF block is larger than 65536 bytes"));
    }
    block.resize(data_len, 0);
    let written = decompressor
        .deflate_decompress(&header_and_data[BGZF_HEADER_LEN..], block)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if written != data_len {
        return Err(invalid("BGZF block is shorter than its recorded length"));
    }
    let mut crc = Crc::new();
    crc.update(block);
    if crc.sum() != crc32 {
        return Err(invalid("BGZF block data checksum mismatch"));
    }
    Ok(())
}

/// Splits an inflated block into its partial first line, the complete lines
/// that survive [`is_skippable_record`], and its partial last line.
fn split_block_lines(
    block: &[u8],
    rules_by_key: &AHashMap<VariantKey, Vec<ScoreRule>>,
) -> BlockLines {
    let Some(first_newline) = memchr(b'\n', block) else {
        return BlockLines {
            bytes: block.to_vec(),
            head_len: block.len(),
            tail_start: block.len(),
            has_newline: false,
        };
    };
    let last_newline = memrchr(b'\n', block).expect("a block with a first newline has a last one");
    let mut bytes = Vec::with_capacity(first_newline + block.len() - last_newline);
    bytes.extend_from_slice(&block[..=first_newline]);
    let body = &block[first_newline + 1..=last_newline];
    let mut line_start = 0usize;
    for line_end in memchr_iter(b'\n', body) {
        if !is_skippable_record(&body[line_start..line_end], rules_by_key) {
            bytes.extend_from_slice(&body[line_start..=line_end]);
        }
        line_start = line_end + 1;
    }
    let tail_start = bytes.len();
    bytes.extend_from_slice(&block[last_newline + 1..]);
    BlockLines {
        bytes,
        head_len: first_newline + 1,
        tail_start,
        has_newline: true,
    }
}

/// Whether `score_vcf_streaming` would read this record line without error and
/// then skip it, so dropping it unread cannot change a score or an error.
///
/// `line` excludes its newline. The checks mirror noodles' `read_record` (valid
/// UTF-8, seven tab-terminated fields) and the scorer's own tests before a key
/// lookup: an unsupported contig, a telomeric position `0`, or a position no
/// score mentions. Anything less certain, including a carriage return in the
/// first two fields, which noodles may strip, is kept.
fn is_skippable_record(line: &[u8], rules_by_key: &AHashMap<VariantKey, Vec<ScoreRule>>) -> bool {
    let Ok(line) = std::str::from_utf8(line) else {
        return false;
    };
    let mut fields = line.splitn(8, '\t');
    let (Some(chromosome), Some(position), Some(_)) = (fields.next(), fields.next(), fields.nth(5))
    else {
        return false;
    };
    if chromosome.contains('\r') || position.contains('\r') {
        return false;
    }
    let Ok(chr) = parse_chromosome_label(chromosome) else {
        return true;
    };
    if position == "0" {
        return true;
    }
    match position.parse::<usize>() {
        Ok(start) if start > 0 => !rules_by_key.contains_key(&(chr, start as u32)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    #[test]
    fn multiallelic_ref_dosage_excludes_every_alternate_allele() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tAltG\tRefA\n1:100\tG\tA\t1\t0\n1:100\tA\tG\t0\t1\n",
        )
        .expect("write score");

        // Sample-major expected values: (G dosage, A dosage). In particular,
        // G/T carries zero copies of A even though its G dosage is only one.
        for (format, samples, expected) in [
            ("GT", "0/1\t0/2\t1/2", [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            (
                "GT:DS",
                "0/1:0.9,0.2\t0/2:0.1,1.1\t1/2:0.8,1.2",
                [0.9, 0.9, 0.1, 0.8, 0.8, 0.0],
            ),
            (
                "GT:GP",
                "0/1:0,1,0,0,0,0\t0/2:0,0,0,1,0,0\t1/2:0,0,0,0,1,0",
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            ),
            (
                "GP",
                "0,1,0,0,0,0\t0,0,0,1,0,0\t0,0,0,0,1,0",
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            ),
            ("GT", "0\t1\t2", [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            (
                "GT:GP",
                "0:1,0,0\t1:0,1,0\t2:0,0,1",
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ),
            ("GP", "1,0,0\t0,1,0\t0,0,1", [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        ] {
            std::fs::write(
                &vcf_path,
                format!(
                    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n1\t100\t.\tA\tG,T\t.\tPASS\t.\t{format}\t{samples}\n"
                ),
            )
            .expect("write vcf");
            let result =
                score_vcf_streaming(&vcf_path, std::slice::from_ref(&score_path), None, None)
                    .unwrap_or_else(|err| panic!("{format} {samples}: {err}"));
            assert_eq!(result.score_names, ["AltG", "RefA"]);
            for (actual, expected) in result.sum_scores.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{format} {samples}: {actual} != {expected}"
                );
            }
            assert_eq!(result.missing_counts, [0; 6]);
        }
    }

    #[test]
    fn dosage_decoding_rejects_invalid_cardinality_and_nonfinite_values() {
        for value in ["1", "0.5,0.5,0", "NaN,0", "inf,0", "-0.1,0", "1.5,1"] {
            assert!(
                parse_vcf_dosage_field(value, 1, 2, Some(2)).is_err(),
                "{value}"
            );
        }
        for (value, ploidy) in [
            ("0,1", Some(2)),
            ("0,1,0", Some(1)),
            ("NaN,0,1", None),
            ("0,-0.1,1", None),
        ] {
            assert!(parse_vcf_gp(value, 1, 1, ploidy).is_err(), "{value}");
        }
        let partial = parse_vcf_dosage_field("0.5,.", 1, 2, Some(2))
            .expect("parse partial dosage")
            .expect("selected ALT dosage");
        assert_eq!(partial.alt_dosage, 0.5);
        assert_eq!(partial.ref_dosage, None);
    }

    #[test]
    fn split_records_at_one_position_count_each_matched_allele() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &vcf_path,
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n",
        ).expect("write vcf");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tScoreA\n1:100\tG\tA\t1\n1:100\tT\tA\t10\n",
        )
        .expect("write score");
        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.sum_scores, [11.0]);
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn ref_effect_rules_use_gt_ploidy_alongside_ds_dosage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("imputed.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT:DS\t0/1:0.9\t1/1:1.8").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            // The effect allele is the VCF REF, so scoring needs ploidy that DS lacks.
            writeln!(score, "1:100\tA\tG\t1.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert!((result.sum_scores[0] - 1.1).abs() < 1e-9);
        assert!((result.sum_scores[1] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn unsupported_score_contigs_are_skipped_not_fatal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "8_KI270821V1_ALT:557595\tG\tA\t2.0").expect("write");
            writeln!(score, "1:100\tG\tA\t0.5").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.sum_scores, [0.5]);
    }

    #[test]
    fn genotype_ploidy_counts_missing_and_multi_digit_alleles() {
        assert_eq!(parse_vcf_genotype_ploidy("0|1"), Some(2));
        assert_eq!(parse_vcf_genotype_ploidy("./."), Some(2));
        assert_eq!(parse_vcf_genotype_ploidy("1"), Some(1));
        assert_eq!(parse_vcf_genotype_ploidy("10/11"), Some(2));
        assert_eq!(parse_vcf_genotype_ploidy(""), None);
    }

    #[test]
    fn native_vcf_stream_scores_gt_without_conversion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t./.").expect("write");
            writeln!(vcf, "1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t0.5").expect("write");
            writeln!(score, "1:200\tC\tT\t1.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.person_iids, vec!["s1", "s2", "s3"]);
        assert_eq!(result.score_names, vec!["ScoreA"]);
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 0, 1]);
        assert_eq!(result.sum_scores, [0.0, 1.5, 2.0]);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_keep_file_and_opposite_orientation_rows_are_scored_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        let keep_path = dir.path().join("keep.txt");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t./.").expect("write");
            writeln!(vcf, "1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0\t0/0").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t0.5").expect("write");
            writeln!(score, "1:100\tA\tG\t1.0").expect("write");
            writeln!(score, "1:200\tC\tT\t1.0").expect("write");
        }

        {
            let mut keep = File::create(&keep_path).expect("create keep");
            writeln!(keep, "s2").expect("write");
            writeln!(keep, "s4").expect("write");
        }

        let result =
            score_vcf_streaming(&vcf_path, &[score_path], Some(&keep_path), None).expect("score");
        assert_eq!(result.person_iids, vec!["s2", "s4"]);
        assert_eq!(result.score_names, vec!["ScoreA"]);
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 1]);
        assert_eq!(result.sum_scores, [2.5, 2.0]);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_zero_net_matched_rows_still_count_in_denominator() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t./.").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:100\tG\tA\t-1.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.missing_counts, [0, 1]);
        assert_eq!(result.sum_scores, [0.0, 0.0]);
        assert_eq!(result.matched_variants, 1);
    }

    #[test]
    fn native_vcf_multiallelic_gt_scores_each_alt_separately() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG,T\t.\tPASS\t.\tGT\t0/1\t0/2\t1/2").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:100\tT\tA\t10.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 0, 0]);
        assert_eq!(result.sum_scores, [1.0, 10.0, 11.0]);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_multiallelic_ds_scores_alt_specific_values_without_gt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Alternate allele dosage\">"
            )
            .expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG,T\t.\tPASS\t.\tDS\t1.5,0.1\t0.2,1.6").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:100\tT\tA\t10.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 0]);
        assert!((result.sum_scores[0] - 2.5).abs() < 1e-12);
        assert!((result.sum_scores[1] - 16.2).abs() < 1e-12);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_applies_score_regions_before_scoring() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1").expect("write");
            writeln!(vcf, "1\t200\t.\tA\tG\t.\tPASS\t.\tGT\t0/1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:200\tG\tA\t10.0").expect("write");
        }

        let mut regions = HashMap::new();
        regions.insert(
            "ScoreA".to_string(),
            GenomicRegion {
                chromosome: 1,
                start: 100,
                end: 100,
            },
        );
        let result =
            score_vcf_streaming(&vcf_path, &[score_path], None, Some(&regions)).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.sum_scores, [1.0]);
        assert_eq!(result.matched_variants, 1);
    }

    #[test]
    fn native_vcf_scores_all_native_score_columns() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(
                score,
                "variant_id\teffect_allele\tother_allele\tScoreA\tScoreB"
            )
            .expect("write");
            writeln!(score, "1:100\tG\tA\t0.5\t2.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_names, vec!["ScoreA", "ScoreB"]);
        assert_eq!(result.score_variant_counts, [1, 1]);
        assert_eq!(result.sum_scores, [0.5, 2.0, 1.0, 4.0]);
    }

    #[test]
    fn native_vcf_ref_effect_haploid_gt_uses_observed_ploidy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts_ref\ts_alt"
            )
            .expect("write");
            writeln!(vcf, "X\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0\t1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "X:100\tA\tG\t2.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.missing_counts, [0, 0]);
        assert_eq!(result.sum_scores, [2.0, 0.0]);
    }

    fn bgzf_block(data: &[u8]) -> Vec<u8> {
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).expect("deflate");
        let compressed = encoder.finish().expect("finish deflate");
        let mut crc = Crc::new();
        crc.update(data);
        let block_len = BGZF_HEADER_LEN + compressed.len() + BGZF_TRAILER_LEN;
        let mut block = vec![
            0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, b'B', b'C',
            0x02, 0x00,
        ];
        block.extend_from_slice(&u16::try_from(block_len - 1).expect("BSIZE").to_le_bytes());
        block.extend_from_slice(&compressed);
        block.extend_from_slice(&crc.sum().to_le_bytes());
        block.extend_from_slice(&u32::try_from(data.len()).expect("ISIZE").to_le_bytes());
        block
    }

    /// BGZF blocks of at most `block_len` uncompressed bytes, then the empty EOF block.
    fn bgzf_bytes(text: &[u8], block_len: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in text.chunks(block_len) {
            bytes.extend_from_slice(&bgzf_block(chunk));
        }
        bytes.extend_from_slice(&bgzf_block(&[]));
        bytes
    }

    /// A cohort that exercises the prefilter: a header spanning many small
    /// blocks, scored and unscored positions, multiallelic and DS records,
    /// phased and haploid calls, telomeric and unsupported-contig records, and
    /// '#' lines after the header.
    fn prefilter_cohort() -> (String, String) {
        let samples = 40usize;
        let mut vcf = String::from(
            "##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Alternate allele dosage\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT",
        );
        for sample in 0..samples {
            vcf.push_str(&format!("\tsample_with_a_long_identifier_{sample}"));
        }
        vcf.push('\n');
        let mut score = String::from("variant_id\teffect_allele\tother_allele\tScoreA\tScoreB\n");
        let genotypes = ["0/0", "0/1", "1/1", "./.", "1|0", "0|1", "1/2", "2/2"];
        for record in 0..400u32 {
            let (chromosome, alt) = match record % 25 {
                3 => ("chrUn_KI270302v1", "G"),
                11 => ("X", "G"),
                17 => ("22", "G,T"),
                _ => ("22", "G"),
            };
            let position = if record % 40 == 5 {
                0
            } else {
                1000 + 10 * record
            };
            if record % 60 == 1 {
                vcf.push_str("#not_a_header\t1\t.\tA\tG\t.\tPASS\t.\tGT");
                vcf.push_str(&"\t0/1".repeat(samples));
                vcf.push('\n');
            }
            // Diploid dosages only: the haploid X calls cannot carry DS up to 1.8.
            let with_ds = record % 4 == 1 && alt == "G" && chromosome != "X";
            vcf.push_str(&format!(
                "{chromosome}\t{position}\t.\tA\t{alt}\t.\tPASS\t.\t{}",
                if with_ds { "GT:DS" } else { "GT" }
            ));
            for sample in 0..samples {
                let index = (record as usize * 7 + sample * 3) % genotypes.len();
                let genotype = if chromosome == "X" {
                    ["0", "1", "."][index % 3]
                } else if alt == "G" {
                    genotypes[index % 6]
                } else {
                    genotypes[index]
                };
                vcf.push('\t');
                vcf.push_str(genotype);
                if with_ds {
                    vcf.push_str(&format!(":{}", (index % 5) as f64 * 0.45));
                }
            }
            vcf.push('\n');
            if (record % 3 == 1 || record % 7 == 2) && chromosome != "chrUn_KI270302v1" {
                score.push_str(&format!(
                    "{chromosome}:{position}\tG\tA\t{}\t-0.5\n",
                    0.25 + f64::from(record) * 1e-3
                ));
                if alt == "G,T" {
                    score.push_str(&format!("{chromosome}:{position}\tA\tT\t0.75\t0.125\n"));
                }
            }
        }
        (vcf, score)
    }

    fn assert_same_native_result(
        expected: &NativeVcfScoreResult,
        actual: &NativeVcfScoreResult,
        context: &str,
    ) {
        assert_eq!(expected.person_iids, actual.person_iids, "{context}");
        assert_eq!(expected.score_names, actual.score_names, "{context}");
        assert_eq!(
            expected.score_variant_counts, actual.score_variant_counts,
            "{context}"
        );
        assert_eq!(expected.missing_counts, actual.missing_counts, "{context}");
        assert_eq!(
            expected.matched_variants, actual.matched_variants,
            "{context}"
        );
        let bits = |values: &[f64]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&expected.sum_scores),
            bits(&actual.sum_scores),
            "{context}"
        );
    }

    #[test]
    fn bgzf_prefilter_scores_match_plain_vcf_at_any_block_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (vcf, score) = prefilter_cohort();
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(&score_path, &score).expect("write score");

        for text in [vcf.as_str(), vcf.trim_end_matches('\n')] {
            let plain_path = dir.path().join("cohort.vcf");
            std::fs::write(&plain_path, text).expect("write vcf");
            let expected =
                score_vcf_streaming(&plain_path, std::slice::from_ref(&score_path), None, None)
                    .expect("plain score");
            assert!(expected.matched_variants > 100);

            for block_len in [1, 2, 7, 100, 4096, 65536] {
                let bgzf_path = dir.path().join("cohort.vcf.gz");
                std::fs::write(&bgzf_path, bgzf_bytes(text.as_bytes(), block_len))
                    .expect("write bgzf");
                let actual =
                    score_vcf_streaming(&bgzf_path, std::slice::from_ref(&score_path), None, None)
                        .expect("bgzf score");
                assert_same_native_result(
                    &expected,
                    &actual,
                    &format!(
                        "block_len={block_len} trailing_newline={}",
                        text.ends_with('\n')
                    ),
                );
            }
        }
    }

    #[test]
    fn bgzf_prefilter_rejects_the_records_plain_vcf_rejects() {
        let dir = tempfile::tempdir().expect("tempdir");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tScoreA\n22:100\tG\tA\t1\n",
        )
        .expect("write score");
        let header =
            b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";
        let unscored = b"22\t200\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n";
        let scored = b"22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n";
        let malformed: [&[u8]; 6] = [
            b"22\tnot_a_position\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n",
            b"22\t00\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n",
            b"22\t300\t.\tA\tG\n",
            b"\n",
            b"#22\t300\n",
            b"22\t300\t.\tA\tG\t.\tPASS\t.\tGT\t0/\xff\n",
        ];
        for bad in malformed {
            let context = String::from_utf8_lossy(bad).into_owned();
            let mut vcf = header.to_vec();
            vcf.extend_from_slice(unscored);
            vcf.extend_from_slice(bad);
            vcf.extend_from_slice(scored);

            let plain_path = dir.path().join("malformed.vcf");
            std::fs::write(&plain_path, &vcf).expect("write vcf");
            assert!(
                score_vcf_streaming(&plain_path, std::slice::from_ref(&score_path), None, None)
                    .is_err(),
                "plain {context:?}"
            );
            for block_len in [3, 65536] {
                let bgzf_path = dir.path().join("malformed.vcf.gz");
                std::fs::write(&bgzf_path, bgzf_bytes(&vcf, block_len)).expect("write bgzf");
                assert!(
                    score_vcf_streaming(&bgzf_path, std::slice::from_ref(&score_path), None, None)
                        .is_err(),
                    "bgzf block_len={block_len} {context:?}"
                );
            }
        }
    }

    #[test]
    fn plain_gzip_members_are_decoded_without_the_prefilter() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (vcf, score) = prefilter_cohort();
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(&score_path, &score).expect("write score");
        let plain_path = dir.path().join("cohort.vcf");
        std::fs::write(&plain_path, &vcf).expect("write vcf");
        let expected =
            score_vcf_streaming(&plain_path, std::slice::from_ref(&score_path), None, None)
                .expect("plain score");

        let gzip = |text: &[u8]| {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(text).expect("gzip");
            encoder.finish().expect("finish gzip")
        };
        let split = vcf.len() / 2;
        let mut mixed = Vec::new();
        for chunk in vcf.as_bytes()[..split].chunks(97) {
            mixed.extend_from_slice(&bgzf_block(chunk));
        }
        mixed.extend_from_slice(&gzip(&vcf.as_bytes()[split..]));

        for (label, bytes) in [("gzip", gzip(vcf.as_bytes())), ("bgzf then gzip", mixed)] {
            let path = dir.path().join("fallback.vcf.gz");
            std::fs::write(&path, &bytes).expect("write gzip");
            let actual = score_vcf_streaming(&path, std::slice::from_ref(&score_path), None, None)
                .expect("gzip score");
            assert_same_native_result(&expected, &actual, label);
        }
    }

    #[test]
    fn corrupt_or_truncated_bgzf_streams_are_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (vcf, score) = prefilter_cohort();
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(&score_path, &score).expect("write score");

        let blocks: Vec<Vec<u8>> = vcf.as_bytes().chunks(512).map(bgzf_block).collect();
        let mut corrupt = blocks.clone();
        let middle = corrupt.len() / 2;
        let crc_offset = corrupt[middle].len() - BGZF_TRAILER_LEN;
        corrupt[middle][crc_offset] ^= 0xff;
        let mut truncated = blocks.concat();
        truncated.truncate(truncated.len() - 100);

        for (label, bytes) in [("corrupt", corrupt.concat()), ("truncated", truncated)] {
            let mut sink = Vec::new();
            assert!(
                MultiGzDecoder::new(&bytes[..])
                    .read_to_end(&mut sink)
                    .is_err(),
                "flate2 also rejects the {label} stream"
            );
            let path = dir.path().join("broken.vcf.gz");
            std::fs::write(&path, &bytes).expect("write bgzf");
            assert!(
                score_vcf_streaming(&path, std::slice::from_ref(&score_path), None, None).is_err(),
                "{label}"
            );
        }
    }

    #[test]
    fn skippable_records_are_read_cleanly_and_skipped_by_the_scorer() {
        let mut rules_by_key: AHashMap<VariantKey, Vec<ScoreRule>> = AHashMap::new();
        rules_by_key.insert((22, 100), Vec::new());
        let cases: [(&[u8], bool); 13] = [
            (b"22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"22\t101\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"chr22\t101\t.\tA\tG\t.\tPASS\t.", true),
            (b"22\t101\t.\tA\tG\t.\tPASS", false),
            (b"22\t0\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"22\t00\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"22\t+101\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"22\tabc\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"HLA-A*01:01\tabc\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"#22\t101\t.\tA\tG\t.\tPASS\t.", true),
            (b"22\r\t101\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"22\t101\t.\tA\tG\t.\tPASS\t.\tGT\t0/\xff", false),
            (b"", false),
        ];
        let header =
            b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";
        for (line, expected) in cases {
            let context = String::from_utf8_lossy(line).into_owned();
            assert_eq!(
                is_skippable_record(line, &rules_by_key),
                expected,
                "{context:?}"
            );
            if !expected {
                continue;
            }
            // Dropping a line is only sound if noodles reads it and the scorer
            // skips it. The prefilter only judges lines after the first record,
            // so one precedes it here too.
            let mut stream = header.to_vec();
            stream.extend_from_slice(b"22\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n");
            stream.extend_from_slice(line);
            stream.push(b'\n');
            let mut reader = VcfReader::new(&stream[..]);
            reader.read_header().expect("header");
            let mut record = noodles_vcf::Record::default();
            reader.read_record(&mut record).expect("first record");
            reader
                .read_record(&mut record)
                .unwrap_or_else(|err| panic!("{context:?}: {err}"));
            let skipped = match parse_chromosome_label(record.reference_sequence_name()) {
                Err(_) => true,
                Ok(chr) => record.variant_start().is_none_or(|start| {
                    let start = start.unwrap_or_else(|err| panic!("{context:?}: {err}"));
                    !rules_by_key.contains_key(&(chr, start.get() as u32))
                }),
            };
            assert!(skipped, "{context:?}");
        }
    }

    #[test]
    fn batched_decoding_reports_the_first_error_a_sequential_scan_raises() {
        let dir = tempfile::tempdir().expect("tempdir");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tScoreA\n1:100\tA\tG\t1\n1:200\tG\tA\t1\n",
        )
        .expect("write score");
        let header =
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n";
        for (records, expected) in [
            // The REF-effect rule has no ploidy at s1, before s2's DS is decoded.
            (
                "1\t100\t.\tA\tG\t.\tPASS\t.\tDS\t0.5\tbad\n",
                "Cannot score REF-effect rule",
            ),
            // With the order reversed, the undecodable DS comes first.
            (
                "1\t100\t.\tA\tG\t.\tPASS\t.\tDS\tbad\t0.5\n",
                "invalid float literal",
            ),
            // An earlier record's error wins over a later record's.
            (
                "1\t200\t.\tA\tG\t.\tPASS\t.\tGT\t0/x\t0/1\n1\t100\t.\tA\tG\t.\tPASS\t.\tDS\t0.5\t0.5\n",
                "unexpected byte",
            ),
        ] {
            let vcf_path = dir.path().join("errors.vcf");
            std::fs::write(&vcf_path, format!("{header}{records}")).expect("write vcf");
            let err = score_vcf_streaming(&vcf_path, std::slice::from_ref(&score_path), None, None)
                .expect_err("malformed records");
            assert!(err.to_string().contains(expected), "{records:?}: {err}");
        }
    }

    #[test]
    fn sample_columns_split_as_noodles_splits_them() {
        let header = b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n";
        for samples in [
            "GT\t0/1\t1|1\t./.",
            "GT\t0/1\t\t1/1",
            "GT\t0/1\t",
            "GT\t\t",
            "GT\t.\t0/1",
            "GT",
            "GT:DS\t0/1:0.5\t.:.\t1",
        ] {
            let mut stream = header.to_vec();
            stream
                .extend_from_slice(format!("22\t100\t.\tA\tG\t.\tPASS\t.\t{samples}\n").as_bytes());
            let mut reader = VcfReader::new(&stream[..]);
            reader.read_header().expect("header");
            let mut record = noodles_vcf::Record::default();
            reader.read_record(&mut record).expect("record");
            let expected: Vec<String> = record
                .samples()
                .iter()
                .map(|sample| sample.as_ref().to_string())
                .collect();
            let samples = record.samples();
            let actual: Vec<&str> = vcf_sample_columns(samples.as_ref()).collect();
            assert_eq!(actual, expected, "{samples:?}");
        }
    }
}
