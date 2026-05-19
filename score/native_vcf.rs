use crate::score::types::parse_chromosome_label;
use crate::shared::files::{VariantCompression, VariantFormat, open_variant_source};
use ahash::{AHashMap, AHashSet};
use flate2::read::MultiGzDecoder;
use noodles_vcf::io::Reader as VcfReader;
use noodles_vcf::variant::record::AlternateBases as _;
use noodles_vcf::variant::record::samples::keys::key;
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

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
    score_index: usize,
    weight: f32,
}

#[derive(Debug, Clone, Copy)]
struct MatchedRule {
    score_index: usize,
    intercept: f64,
    alt_slope: f64,
}

type VariantKey = (u8, u32);

pub fn score_vcf_streaming(
    input_path: &Path,
    native_score_files: &[PathBuf],
    keep: Option<&Path>,
) -> Result<NativeVcfScoreResult, Box<dyn Error + Send + Sync>> {
    let score_names = parse_score_names(native_score_files)?;
    let rules_by_key = load_score_rules(native_score_files)?;

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
            let reader: Box<dyn BufRead + Send> =
                Box::new(BufReader::new(MultiGzDecoder::new(source)));
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
    let mut matched_variant_keys: AHashSet<(VariantKey, usize, usize)> = AHashSet::new();

    let mut record = noodles_vcf::Record::default();
    let mut decoded = vec![None; kept_indices.len()];

    while reader.read_record(&mut record)? != 0 {
        let chr = match parse_chromosome_label(record.reference_sequence_name()) {
            Ok(chr) => chr,
            Err(_) => continue,
        };
        let Some(start) = record.variant_start() else {
            continue;
        };
        let pos = start?.get() as u32;
        let key = (chr, pos);
        let Some(score_rules) = rules_by_key.get(&key) else {
            continue;
        };

        let ref_allele = record.reference_bases();
        let alt_alleles: Vec<String> = record
            .alternate_bases()
            .iter()
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .map(str::to_string)
            .collect();

        for (alt_offset, alt_allele) in alt_alleles.iter().enumerate() {
            let alt_index = alt_offset + 1;
            let matched_rules = match_rules_for_allele(score_rules, ref_allele, alt_allele);
            if matched_rules.is_empty() {
                continue;
            }

            decoded.fill(None);
            decode_vcf_record_best(
                &record,
                alt_index,
                alt_alleles.len(),
                &kept_indices,
                &mut decoded,
            )?;

            for rule in &matched_rules {
                score_variant_counts[rule.score_index] += 1;
                matched_variant_keys.insert((key, alt_index, rule.score_index));
            }

            for (out_person_idx, dosage) in decoded.iter().copied().enumerate() {
                for rule in &matched_rules {
                    let cell = out_person_idx * num_scores + rule.score_index;
                    match dosage {
                        Some(alt_dosage) => {
                            sum_scores[cell] += rule.intercept + rule.alt_slope * alt_dosage;
                        }
                        None => {
                            missing_counts[cell] += 1;
                        }
                    }
                }
            }
        }
    }

    if matched_variant_keys.is_empty() {
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
        matched_variants: matched_variant_keys.len(),
    })
}

fn parse_score_names(
    native_score_files: &[PathBuf],
) -> Result<Vec<String>, Box<dyn Error + Send + Sync>> {
    let mut names = Vec::with_capacity(native_score_files.len());
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
        names.push(header[3].to_string());
    }
    Ok(names)
}

fn load_score_rules(
    native_score_files: &[PathBuf],
) -> Result<AHashMap<VariantKey, Vec<ScoreRule>>, Box<dyn Error + Send + Sync>> {
    let mut rules_by_key: AHashMap<VariantKey, Vec<ScoreRule>> = AHashMap::new();

    for (score_index, path) in native_score_files.iter().enumerate() {
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
            let Some(variant_id) = fields.next() else {
                line.clear();
                continue;
            };
            if variant_id == "variant_id" {
                line.clear();
                continue;
            }
            let effect_allele = fields.next().unwrap_or_default();
            let other_allele = fields.next().unwrap_or_default();
            let weight_text = fields.next().unwrap_or_default();
            if effect_allele.is_empty() || other_allele.is_empty() || weight_text.is_empty() {
                return Err(format!(
                    "Malformed native score row in '{}' at line {}.",
                    path.display(),
                    line_number
                )
                .into());
            }

            let mut key_parts = variant_id.splitn(2, ':');
            let chr = key_parts.next().unwrap_or_default();
            let pos = key_parts.next().unwrap_or_default();
            let key = (
                parse_chromosome_label(chr).map_err(|err| {
                    format!(
                        "Invalid chromosome in '{}' at line {}: {}",
                        path.display(),
                        line_number,
                        err
                    )
                })?,
                pos.parse::<u32>().map_err(|err| {
                    format!(
                        "Invalid position in '{}' at line {}: {}",
                        path.display(),
                        line_number,
                        err
                    )
                })?,
            );
            let weight = weight_text.parse::<f32>().map_err(|err| {
                format!(
                    "Invalid weight in '{}' at line {}: {}",
                    path.display(),
                    line_number,
                    err
                )
            })?;

            rules_by_key.entry(key).or_default().push(ScoreRule {
                effect_allele: effect_allele.to_string(),
                other_allele: other_allele.to_string(),
                score_index,
                weight,
            });

            line.clear();
        }
    }

    Ok(rules_by_key)
}

fn match_rules_for_allele(
    rules: &[ScoreRule],
    ref_allele: &str,
    alt_allele: &str,
) -> Vec<MatchedRule> {
    let mut aggregate: BTreeMap<usize, (f64, f64)> = BTreeMap::new();
    for rule in rules {
        if rule.effect_allele == alt_allele && rule.other_allele == ref_allele {
            let (_, slope) = aggregate.entry(rule.score_index).or_default();
            *slope += f64::from(rule.weight);
        } else if rule.effect_allele == ref_allele && rule.other_allele == alt_allele {
            let (intercept, slope) = aggregate.entry(rule.score_index).or_default();
            *intercept += 2.0 * f64::from(rule.weight);
            *slope -= f64::from(rule.weight);
        }
    }

    aggregate
        .into_iter()
        .map(|(score_index, (intercept, alt_slope))| MatchedRule {
            score_index,
            intercept,
            alt_slope,
        })
        .collect()
}

fn decode_vcf_record_best(
    record: &noodles_vcf::Record,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    dest: &mut [Option<f64>],
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let samples = record.samples();
    if samples.is_empty() {
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

    let mut kept_cursor = 0usize;
    for (sample_idx, sample) in samples.iter().enumerate() {
        while kept_cursor < kept_indices.len() && kept_indices[kept_cursor] < sample_idx {
            kept_cursor += 1;
        }
        if kept_cursor >= kept_indices.len() {
            break;
        }
        if kept_indices[kept_cursor] != sample_idx {
            continue;
        }

        let mut ds_field = None;
        let mut gp_field = None;
        let mut gt_field = None;
        for (idx, field) in sample.as_ref().split(':').enumerate() {
            if ds_index == Some(idx) {
                ds_field = Some(field);
            }
            if gp_index == Some(idx) {
                gp_field = Some(field);
            }
            if gt_idx == Some(idx) {
                gt_field = Some(field);
            }
        }

        if let Some(value) = ds_field
            && let Some(parsed) = parse_vcf_dosage_field(value, alt_index, alt_count)?
        {
            dest[kept_cursor] = Some(parsed);
            kept_cursor += 1;
            continue;
        }
        if let Some(value) = gp_field
            && let Some(parsed) = parse_vcf_gp(value, alt_index, alt_count)?
        {
            dest[kept_cursor] = Some(parsed);
            kept_cursor += 1;
            continue;
        }
        if let Some(value) = gt_field
            && let Some(parsed) = parse_vcf_genotype(value, alt_index)?
        {
            dest[kept_cursor] = Some(parsed);
        }
        kept_cursor += 1;
    }

    Ok(())
}

fn parse_vcf_dosage_field(
    field: &str,
    alt_index: usize,
    alt_count: usize,
) -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    if field == "." {
        return Ok(None);
    }
    let allele_offset = alt_index
        .checked_sub(1)
        .ok_or("ALT allele index must be one-based")?;

    let mut dosage_values = field.split(',');
    let first_value = dosage_values.next().unwrap_or_default();
    if let Some(second_value) = dosage_values.next() {
        let value = match allele_offset {
            0 => first_value,
            1 => second_value,
            offset => {
                let Some(value) = dosage_values.nth(offset - 2) else {
                    return Ok(None);
                };
                value
            }
        };
        return parse_numeric_str(value);
    }

    if alt_count == 1 {
        parse_numeric_str(field)
    } else {
        Err(format!("multi-allelic dosage field is scalar for ALT allele index {alt_index}").into())
    }
}

fn parse_vcf_gp(
    field: &str,
    alt_index: usize,
    alt_count: usize,
) -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    if field == "." {
        return Ok(None);
    }
    if alt_index == 0 || alt_index > alt_count {
        return Err(format!(
            "ALT allele index {alt_index} is out of range for {alt_count} alternate alleles"
        )
        .into());
    }

    let allele_count = alt_count + 1;
    let expected_len = allele_count
        .checked_mul(allele_count + 1)
        .map(|n| n / 2)
        .ok_or("GP allele count overflow")?;
    let mut dosage = 0.0f64;
    let mut idx = 0usize;
    let mut parts = field.split(',');
    for second in 0..allele_count {
        for first in 0..=second {
            let Some(part) = parts.next() else {
                return Err(format!(
                    "GP field has {idx} values, expected {expected_len} for {alt_count} alternate alleles"
                )
                .into());
            };
            if part == "." {
                return Ok(None);
            }
            let probability = part.parse::<f64>()?;
            let copies = usize::from(first == alt_index) + usize::from(second == alt_index);
            dosage += probability * copies as f64;
            idx += 1;
        }
    }

    if parts.next().is_some() {
        let actual_len = expected_len + 1 + parts.count();
        return Err(format!(
            "GP field has {actual_len} values, expected {expected_len} for {alt_count} alternate alleles"
        )
        .into());
    }

    Ok(Some(dosage))
}

fn parse_numeric_str(text: &str) -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "." {
        Ok(None)
    } else {
        trimmed.parse::<f64>().map(Some).map_err(Into::into)
    }
}

fn parse_vcf_genotype(
    field: &str,
    alt_index: usize,
) -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    if field.is_empty() {
        return Ok(None);
    }

    let mut dosage = 0.0f64;
    let mut seen = false;
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
                seen = true;
            }
            other => return Err(format!("unexpected byte {other} in genotype field").into()),
        }
    }

    if seen { Ok(Some(dosage)) } else { Ok(None) }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

        let result = score_vcf_streaming(&vcf_path, &[score_path], None).expect("score");
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
            score_vcf_streaming(&vcf_path, &[score_path], Some(&keep_path)).expect("score");
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

        let result = score_vcf_streaming(&vcf_path, &[score_path], None).expect("score");
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

        let result = score_vcf_streaming(&vcf_path, &[score_path], None).expect("score");
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

        let result = score_vcf_streaming(&vcf_path, &[score_path], None).expect("score");
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 0]);
        assert!((result.sum_scores[0] - 2.5).abs() < 1e-12);
        assert!((result.sum_scores[1] - 16.2).abs() < 1e-12);
        assert_eq!(result.matched_variants, 2);
    }
}
