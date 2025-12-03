use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use infer_sex::{
    Chromosome, GenomeBuild, InferenceConfig, InferenceResult, InferredSex,
    SexInferenceAccumulator, VariantInfo,
};
use thiserror::Error;

use crate::map::fit::VariantBlockSource;
use crate::map::io::{GenotypeDataset, GenotypeIoError, SelectionPlan};
use crate::map::variant_filter::VariantKey;

#[derive(Debug, Error)]
pub enum SexInferenceError {
    #[error("genotype I/O error: {0}")]
    Dataset(#[from] GenotypeIoError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(
        "variant stream yielded more variants than expected (expected {expected}, observed {observed})"
    )]
    VariantOverflow { expected: usize, observed: usize },
    #[error(
        "variant stream terminated early (processed {observed} of {expected} expected variants)"
    )]
    VariantUnderflow { expected: usize, observed: usize },
}

#[derive(Debug, Clone)]
pub struct SexInferenceRecord {
    pub individual_id: String,
    pub inference: InferenceResult,
    pub sry_variant_count: u64,
}

pub fn infer_sex_to_tsv(genotype_path: &Path) -> Result<PathBuf, SexInferenceError> {
    let dataset = GenotypeDataset::open(genotype_path)?;
    let default_output = dataset.output_path("sex.tsv");

    let variant_keys = dataset.variant_keys_for_plan(&SelectionPlan::All)?;
    let records = collect_inference(&dataset, &variant_keys)?;

    write_results(&default_output, &records)?;

    Ok(default_output)
}

fn collect_inference(
    dataset: &GenotypeDataset,
    variant_keys: &[VariantKey],
) -> Result<Vec<SexInferenceRecord>, SexInferenceError> {
    let sample_ids: Vec<String> = dataset
        .samples()
        .iter()
        .map(|record| record.individual_id.clone())
        .collect();
    let n_samples = sample_ids.len();
    let mut sry_counts = vec![0u64; n_samples];

    let build = infer_build_from_keys(variant_keys);
    let config = InferenceConfig { build };
    let mut accumulators: Vec<SexInferenceAccumulator> = (0..n_samples)
        .map(|_| SexInferenceAccumulator::new(config))
        .collect();

    if variant_keys.is_empty() {
        return Ok(finalize_records(accumulators, sample_ids, sry_counts));
    }

    let mut block_source = dataset.block_source()?;
    let total_variants = variant_keys.len();
    let block_capacity = 256usize;
    let mut storage = vec![f64::NAN; block_capacity * n_samples];
    let mut processed = 0usize;

    while processed < total_variants {
        let capacity = block_capacity.min(total_variants - processed);
        let slice_len = capacity * n_samples;
        let filled = block_source.next_block_into(capacity, &mut storage[..slice_len])?;
        if filled == 0 {
            return Err(SexInferenceError::VariantUnderflow {
                expected: total_variants,
                observed: processed,
            });
        }
        if processed + filled > total_variants {
            return Err(SexInferenceError::VariantOverflow {
                expected: total_variants,
                observed: processed + filled,
            });
        }

        for local_idx in 0..filled {
            let key = &variant_keys[processed + local_idx];
            let Some(chrom) = classify_chromosome(&key.chromosome) else {
                continue;
            };
            let pos = key.position;
            let column_offset = local_idx * n_samples;
            let is_sry_variant = matches!(chrom, Chromosome::Y) && is_in_sry_region(build, pos);

            for sample_idx in 0..n_samples {
                let dosage = storage[column_offset + sample_idx];
                if dosage.is_nan() {
                    continue;
                }
                if is_sry_variant {
                    sry_counts[sample_idx] += 1;
                }
                let is_het = dosage == 1.0;
                let info = VariantInfo {
                    chrom,
                    pos,
                    is_heterozygous: is_het,
                };
                accumulators[sample_idx].process_variant(&info);
            }
        }

        processed += filled;
    }

    if processed < total_variants {
        return Err(SexInferenceError::VariantUnderflow {
            expected: total_variants,
            observed: processed,
        });
    }

    Ok(finalize_records(accumulators, sample_ids, sry_counts))
}

fn finalize_records(
    accumulators: Vec<SexInferenceAccumulator>,
    sample_ids: Vec<String>,
    sry_counts: Vec<u64>,
) -> Vec<SexInferenceRecord> {
    finalize_records_with_sry(accumulators, sample_ids, sry_counts)
}

fn finalize_records_with_sry(
    accumulators: Vec<SexInferenceAccumulator>,
    sample_ids: Vec<String>,
    sry_counts: Vec<u64>,
) -> Vec<SexInferenceRecord> {
    accumulators
        .into_iter()
        .zip(sample_ids.into_iter())
        .zip(sry_counts.into_iter())
        .map(
            |((acc, individual_id), sry_variant_count)| SexInferenceRecord {
                individual_id,
                inference: acc.finish(),
                sry_variant_count,
            },
        )
        .collect()
}

fn write_results(path: &Path, records: &[SexInferenceRecord]) -> Result<(), SexInferenceError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "IID\tSex\tX_Het_Ratio\tX_Het_Vote\tY_Non_PAR\tY_PAR\tY_Vote\tSRY_Count\tSRY_Vote\tPAR_NonPAR_Ratio\tPAR_NonPAR_Vote\tMale_Votes\tFemale_Votes",
    )?;
    for record in records {
        let label = sex_label(record.inference.final_call);
        let report = &record.inference.report;
        let (x_ratio, x_vote) = match report.x_heterozygosity_check {
            Some((ratio, vote)) => (ratio.to_string(), sex_label(vote).to_string()),
            None => ("NA".to_string(), "NA".to_string()),
        };
        let (y_non_par, y_par, y_vote) = match report.y_presence_check {
            Some((non_par, par, vote)) => (
                non_par.to_string(),
                par.to_string(),
                sex_label(vote).to_string(),
            ),
            None => ("NA".to_string(), "NA".to_string(), "NA".to_string()),
        };
        let sry_vote = report
            .sry_presence_check
            .map_or("NA".to_string(), |vote| sex_label(vote).to_string());
        let (par_ratio, par_vote) = match report.par_non_par_het_check {
            Some((ratio, vote)) => (ratio.to_string(), sex_label(vote).to_string()),
            None => ("NA".to_string(), "NA".to_string()),
        };

        writeln!(
            writer,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            record.individual_id,
            label,
            x_ratio,
            x_vote,
            y_non_par,
            y_par,
            y_vote,
            record.sry_variant_count,
            sry_vote,
            par_ratio,
            par_vote,
            report.final_male_votes,
            report.final_female_votes,
        )?;
    }
    writer.flush()?;
    Ok(())
}

fn sex_label(sex: InferredSex) -> &'static str {
    match sex {
        InferredSex::Male => "male",
        InferredSex::Female => "female",
    }
}

fn infer_build_from_keys(keys: &[VariantKey]) -> GenomeBuild {
    const GRCH38_THRESHOLD: u64 = 155_700_000;
    const GRCH37_THRESHOLD: u64 = 154_900_000;

    let max_x = keys
        .iter()
        .filter(|key| matches!(classify_chromosome(&key.chromosome), Some(Chromosome::X)))
        .map(|key| key.position)
        .max();

    match max_x {
        Some(pos) if pos >= GRCH38_THRESHOLD => GenomeBuild::Build38,
        Some(pos) if pos >= GRCH37_THRESHOLD => GenomeBuild::Build37,
        Some(_) => GenomeBuild::Build38,
        None => GenomeBuild::Build38,
    }
}

fn classify_chromosome(label: &str) -> Option<Chromosome> {
    match label {
        "X" | "23" => Some(Chromosome::X),
        "Y" | "24" => Some(Chromosome::Y),
        other if other.eq_ignore_ascii_case("X") => Some(Chromosome::X),
        other if other.eq_ignore_ascii_case("Y") => Some(Chromosome::Y),
        _ => None,
    }
}

fn is_in_sry_region(build: GenomeBuild, pos: u64) -> bool {
    match build {
        GenomeBuild::Build37 => (2_786_855..=2_787_682).contains(&pos),
        GenomeBuild::Build38 => (2_654_896..=2_655_723).contains(&pos),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;

    use tempfile::tempdir;

    #[derive(Clone)]
    struct VariantSpec {
        chrom: &'static str,
        pos: u64,
        genotypes: Vec<u8>,
    }

    impl VariantSpec {
        fn new(chrom: &'static str, pos: u64, female_code: u8, male_code: u8) -> Self {
            Self {
                chrom,
                pos,
                genotypes: vec![female_code, male_code],
            }
        }
    }

    fn encode_variant(codes: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0u8; (codes.len() + 3) / 4];
        for (idx, &code) in codes.iter().enumerate() {
            let byte_idx = idx / 4;
            let shift = (idx % 4) * 2;
            bytes[byte_idx] |= (code & 0b11) << shift;
        }
        bytes
    }

    fn write_plink_dataset(
        dir: &Path,
        stem: &str,
        variants: &[VariantSpec],
    ) -> std::io::Result<std::path::PathBuf> {
        let bed_path = dir.join(format!("{stem}.bed"));
        let bim_path = dir.join(format!("{stem}.bim"));
        let fam_path = dir.join(format!("{stem}.fam"));

        let mut bed_bytes = Vec::with_capacity(3 + variants.len());
        bed_bytes.extend_from_slice(&[0x6c, 0x1b, 0x01]);
        for variant in variants {
            let encoded = encode_variant(&variant.genotypes);
            bed_bytes.extend_from_slice(&encoded);
        }
        fs::write(&bed_path, bed_bytes)?;

        let mut bim_lines = String::new();
        for (idx, variant) in variants.iter().enumerate() {
            bim_lines.push_str(&format!(
                "{}\tvar{}\t0\t{}\tA\tG
",
                variant.chrom,
                idx + 1,
                variant.pos
            ));
        }
        fs::write(&bim_path, bim_lines)?;

        let fam_contents = "FAM1\tF1\t0\t0\t2\t-9
FAM2\tM1\t0\t0\t1\t-9
";
        fs::write(&fam_path, fam_contents)?;

        Ok(bed_path)
    }

    #[test]
    fn classify_chromosome_recognizes_xy_labels() {
        assert_eq!(classify_chromosome("X"), Some(Chromosome::X));
        assert_eq!(classify_chromosome("x"), Some(Chromosome::X));
        assert_eq!(classify_chromosome("23"), Some(Chromosome::X));
        assert_eq!(classify_chromosome("Y"), Some(Chromosome::Y));
        assert_eq!(classify_chromosome("y"), Some(Chromosome::Y));
        assert_eq!(classify_chromosome("24"), Some(Chromosome::Y));
        assert_eq!(classify_chromosome("chr1"), None);
    }

    #[test]
    fn infer_build_from_keys_detects_build_thresholds() {
        let keys = vec![VariantKey::new("chrX", 155_800_000)];
        assert_eq!(infer_build_from_keys(&keys), GenomeBuild::Build38);

        let keys = vec![VariantKey::new("X", 155_000_000)];
        assert_eq!(infer_build_from_keys(&keys), GenomeBuild::Build37);

        let keys: Vec<VariantKey> = Vec::new();
        assert_eq!(infer_build_from_keys(&keys), GenomeBuild::Build38);
    }

    #[test]
    fn synthetic_plink_dataset_produces_expected_calls() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let mut variants = Vec::new();

        for i in 0..10u64 {
            let female_code = if i < 8 { 2 } else { 0 };
            variants.push(VariantSpec::new("X", 20_000 + i, female_code, 0));
        }

        let non_par_total = 990u64;
        for i in 0..non_par_total {
            let pos = if i == non_par_total - 1 {
                155_800_000
            } else {
                3_000_000 + i
            };
            let female_code = if i < 791 || i == non_par_total - 1 {
                2
            } else {
                0
            };
            variants.push(VariantSpec::new("X", pos, female_code, 0));
        }

        for i in 0..9u64 {
            variants.push(VariantSpec::new("Y", 50_000 + i, 1, 0));
        }
        variants.push(VariantSpec::new("Y", 2_655_000, 1, 3));

        for i in 0..55u64 {
            variants.push(VariantSpec::new("Y", 3_000_000 + i, 1, 0));
        }

        let bed_path = write_plink_dataset(dir.path(), "synthetic", &variants)?;

        let output_path = infer_sex_to_tsv(&bed_path)?;
        let tsv_contents = fs::read_to_string(&output_path)?;

        let mut calls = HashMap::new();
        for line in tsv_contents.lines().skip(1) {
            let mut parts = line.split('\t');
            let iid = parts.next().unwrap().to_string();
            let sex = parts.next().unwrap().to_string();
            calls.insert(iid, sex);
        }

        assert_eq!(calls.get("F1").map(String::as_str), Some("female"));
        assert_eq!(calls.get("M1").map(String::as_str), Some("male"));

        Ok(())
    }

    #[test]
    fn inference_report_includes_expected_evidence_votes() -> Result<(), Box<dyn std::error::Error>>
    {
        let dir = tempdir()?;
        let mut variants = Vec::new();

        for i in 0..200u64 {
            let female_code = if i < 160 { 2 } else { 0 };
            variants.push(VariantSpec::new("X", 20_000 + i, female_code, 0));
        }
        for i in 0..1_000u64 {
            let female_code = if i < 800 { 2 } else { 0 };
            variants.push(VariantSpec::new("X", 3_000_000 + i, female_code, 0));
        }
        for i in 0..50u64 {
            variants.push(VariantSpec::new("Y", 20_000 + i, 0, 0));
        }
        for i in 0..60u64 {
            variants.push(VariantSpec::new("Y", 3_000_000 + i, 1, 0));
        }
        variants.push(VariantSpec::new("Y", 2_655_000, 1, 3));

        let bed_path = write_plink_dataset(dir.path(), "evidence", &variants)?;

        let dataset = GenotypeDataset::open(&bed_path)?;
        let keys = dataset.variant_keys_for_plan(&SelectionPlan::All)?;
        let records = collect_inference(&dataset, &keys)?;

        let female = records
            .iter()
            .find(|record| record.individual_id == "F1")
            .unwrap();
        assert_eq!(female.inference.final_call, InferredSex::Female);
        let (x_ratio, x_vote) = female
            .inference
            .report
            .x_heterozygosity_check
            .expect("female sample should have X heterozygosity evidence");
        assert!((x_ratio - 0.8).abs() < 1e-9);
        assert_eq!(x_vote, InferredSex::Female);
        let (non_par_y, par_y, y_vote) = female
            .inference
            .report
            .y_presence_check
            .expect("female sample should have Y presence report");
        assert_eq!((non_par_y, par_y, y_vote), (0, 50, InferredSex::Female));
        assert!(female.inference.report.sry_presence_check.is_none());
        let (ratio, vote) = female
            .inference
            .report
            .par_non_par_het_check
            .expect("female sample should have PAR/non-PAR heterozygosity evidence");
        assert!((ratio - 1.0).abs() < 1e-9);
        assert_eq!(vote, InferredSex::Female);
        assert_eq!(female.inference.report.final_female_votes, 3);
        assert_eq!(female.inference.report.final_male_votes, 0);

        let male = records
            .iter()
            .find(|record| record.individual_id == "M1")
            .unwrap();
        assert_eq!(male.inference.final_call, InferredSex::Male);
        let (male_ratio, male_vote) = male
            .inference
            .report
            .x_heterozygosity_check
            .expect("male sample should have X heterozygosity report");
        assert_eq!(male_ratio, 0.0);
        assert_eq!(male_vote, InferredSex::Male);
        let (male_non_par_y, male_par_y, male_y_vote) = male
            .inference
            .report
            .y_presence_check
            .expect("male sample should have Y presence report");
        assert_eq!(male_non_par_y, 60);
        assert_eq!(male_par_y, 51);
        assert_eq!(male_y_vote, InferredSex::Male);
        assert_eq!(
            male.inference.report.sry_presence_check,
            Some(InferredSex::Male)
        );
        let (male_ratio, male_par_vote) = male
            .inference
            .report
            .par_non_par_het_check
            .expect("male sample should have PAR/non-PAR ratio");
        assert_eq!(male_ratio, -1.0);
        assert_eq!(male_par_vote, InferredSex::Male);
        assert_eq!(male.inference.report.final_male_votes, 4);
        assert_eq!(male.inference.report.final_female_votes, 0);

        Ok(())
    }

    #[test]
    fn sex_tsv_includes_evidence_columns() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let mut variants = Vec::new();

        for i in 0..200u64 {
            let female_code = if i < 160 { 2 } else { 0 };
            variants.push(VariantSpec::new("X", 20_000 + i, female_code, 0));
        }
        for i in 0..1_000u64 {
            let female_code = if i < 800 { 2 } else { 0 };
            variants.push(VariantSpec::new("X", 3_000_000 + i, female_code, 0));
        }
        for i in 0..50u64 {
            variants.push(VariantSpec::new("Y", 20_000 + i, 0, 0));
        }
        for i in 0..60u64 {
            variants.push(VariantSpec::new("Y", 3_000_000 + i, 1, 0));
        }
        variants.push(VariantSpec::new("Y", 2_655_000, 1, 3));

        let bed_path = write_plink_dataset(dir.path(), "evidence", &variants)?;
        let output_path = infer_sex_to_tsv(&bed_path)?;
        let contents = fs::read_to_string(&output_path)?;
        let mut lines = contents.lines();

        let header = lines.next().unwrap();
        assert_eq!(
            header,
            "IID\tSex\tX_Het_Ratio\tX_Het_Vote\tY_Non_PAR\tY_PAR\tY_Vote\tSRY_Count\tSRY_Vote\tPAR_NonPAR_Ratio\tPAR_NonPAR_Vote\tMale_Votes\tFemale_Votes",
        );

        let mut rows: HashMap<String, Vec<String>> = HashMap::new();
        for line in lines {
            let cols: Vec<String> = line.split('\t').map(|value| value.to_string()).collect();
            assert_eq!(cols.len(), 13);
            rows.insert(cols[0].clone(), cols);
        }

        let female = rows.get("F1").expect("female row missing");
        assert_eq!(
            female,
            &vec![
                "F1".to_string(),
                "female".to_string(),
                "0.8".to_string(),
                "female".to_string(),
                "0".to_string(),
                "50".to_string(),
                "female".to_string(),
                "0".to_string(),
                "NA".to_string(),
                "1".to_string(),
                "female".to_string(),
                "0".to_string(),
                "3".to_string(),
            ],
        );

        let male = rows.get("M1").expect("male row missing");
        assert_eq!(
            male,
            &vec![
                "M1".to_string(),
                "male".to_string(),
                "0".to_string(),
                "male".to_string(),
                "60".to_string(),
                "51".to_string(),
                "male".to_string(),
                "1".to_string(),
                "male".to_string(),
                "-1".to_string(),
                "male".to_string(),
                "4".to_string(),
                "0".to_string(),
            ],
        );

        Ok(())
    }

    #[test]
    fn heterozygosity_vote_requires_minimum_x_variants() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let mut variants = Vec::new();

        for i in 0..999u64 {
            variants.push(VariantSpec::new("X", 3_000_000 + i, 2, 0));
        }
        for i in 0..50u64 {
            variants.push(VariantSpec::new("Y", 20_000 + i, 0, 1));
        }
        for i in 0..60u64 {
            variants.push(VariantSpec::new("Y", 3_000_000 + i, 1, 0));
        }
        variants.push(VariantSpec::new("Y", 2_655_000, 1, 3));

        let bed_path = write_plink_dataset(dir.path(), "min_threshold", &variants)?;

        let dataset = GenotypeDataset::open(&bed_path)?;
        let keys = dataset.variant_keys_for_plan(&SelectionPlan::All)?;
        let records = collect_inference(&dataset, &keys)?;

        let female = records
            .iter()
            .find(|record| record.individual_id == "F1")
            .unwrap();
        assert_eq!(female.inference.final_call, InferredSex::Female);
        assert!(female.inference.report.x_heterozygosity_check.is_none());
        let (non_par_y, par_y, y_vote) = female
            .inference
            .report
            .y_presence_check
            .expect("female sample should have a Y presence report");
        assert_eq!((non_par_y, par_y, y_vote), (0, 50, InferredSex::Female));
        assert!(female.inference.report.par_non_par_het_check.is_none());

        let male = records
            .iter()
            .find(|record| record.individual_id == "M1")
            .unwrap();
        assert_eq!(male.inference.final_call, InferredSex::Male);
        assert!(male.inference.report.x_heterozygosity_check.is_none());
        let (male_non_par_y, male_par_y, male_y_vote) = male
            .inference
            .report
            .y_presence_check
            .expect("male sample should have a Y presence report");
        assert_eq!(male_non_par_y, 60);
        assert_eq!(male_par_y, 1);
        assert_eq!(male_y_vote, InferredSex::Male);
        assert_eq!(
            male.inference.report.sry_presence_check,
            Some(InferredSex::Male)
        );
        let (male_ratio, male_vote) = male
            .inference
            .report
            .par_non_par_het_check
            .expect("male sample should have a PAR/non-PAR report");
        assert_eq!(male_ratio, -1.0);
        assert_eq!(male_vote, InferredSex::Male);

        Ok(())
    }

    #[test]
    fn inference_defaults_to_male_without_evidence() -> Result<(), Box<dyn std::error::Error>> {
        let dir = tempdir()?;
        let variants = vec![VariantSpec::new("X", 20_000, 1, 1)];
        let bed_path = write_plink_dataset(dir.path(), "empty", &variants)?;

        let dataset = GenotypeDataset::open(&bed_path)?;
        let keys = dataset.variant_keys_for_plan(&SelectionPlan::All)?;
        assert_eq!(keys.len(), 1);
        let records = collect_inference(&dataset, &keys)?;

        for record in records {
            assert_eq!(record.inference.final_call, InferredSex::Male);
            assert_eq!(record.inference.report.final_male_votes, 0);
            assert_eq!(record.inference.report.final_female_votes, 0);
            assert!(record.inference.report.x_heterozygosity_check.is_none());
            assert!(record.inference.report.y_presence_check.is_none());
            assert!(record.inference.report.sry_presence_check.is_none());
            assert!(record.inference.report.par_non_par_het_check.is_none());
        }

        Ok(())
    }
}
