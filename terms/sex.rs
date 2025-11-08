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

    let build = infer_build_from_keys(variant_keys);
    let config = InferenceConfig { build };
    let mut accumulators: Vec<SexInferenceAccumulator> = (0..n_samples)
        .map(|_| SexInferenceAccumulator::new(config))
        .collect();

    if variant_keys.is_empty() {
        return Ok(finalize_records(accumulators, sample_ids));
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

            for sample_idx in 0..n_samples {
                let dosage = storage[column_offset + sample_idx];
                if dosage.is_nan() {
                    continue;
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

    Ok(finalize_records(accumulators, sample_ids))
}

fn finalize_records(
    accumulators: Vec<SexInferenceAccumulator>,
    sample_ids: Vec<String>,
) -> Vec<SexInferenceRecord> {
    accumulators
        .into_iter()
        .zip(sample_ids.into_iter())
        .map(|(acc, individual_id)| SexInferenceRecord {
            individual_id,
            inference: acc.finish(),
        })
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
    writeln!(writer, "IID\tSex")?;
    for record in records {
        let label = match record.inference.final_call {
            InferredSex::Male => "male",
            InferredSex::Female => "female",
        };
        writeln!(writer, "{}\t{}", record.individual_id, label)?;
    }
    writer.flush()?;
    Ok(())
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
