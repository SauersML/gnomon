use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::fit::{HwePcaError, HwePcaModel};
use super::io::{GenotypeDataset, GenotypeIoError};
use super::project::{ProjectionOptions, ProjectionResult};

/// High-level commands that can be executed within the `map` module.
#[derive(Debug)]
pub enum MapCommand {
    Fit { genotype_path: PathBuf },
    Project { genotype_path: PathBuf },
}

/// Errors that can occur when executing `map` commands.
#[derive(Debug)]
pub enum MapDriverError {
    Hwe(HwePcaError),
    Io(std::io::Error),
    Dataset(GenotypeIoError),
    Serialization(serde_json::Error),
    InvalidState(String),
}

impl fmt::Display for MapDriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hwe(err) => write!(f, "HWE PCA error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Dataset(err) => write!(f, "genotype dataset error: {err}"),
            Self::Serialization(err) => write!(f, "serialization error: {err}"),
            Self::InvalidState(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for MapDriverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Hwe(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::Dataset(err) => Some(err),
            Self::Serialization(err) => Some(err),
            Self::InvalidState(_) => None,
        }
    }
}

impl From<HwePcaError> for MapDriverError {
    fn from(value: HwePcaError) -> Self {
        Self::Hwe(value)
    }
}

impl From<std::io::Error> for MapDriverError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<GenotypeIoError> for MapDriverError {
    fn from(value: GenotypeIoError) -> Self {
        Self::Dataset(value)
    }
}

impl From<serde_json::Error> for MapDriverError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value)
    }
}

/// Execute the provided [`MapCommand`].
pub fn run(command: MapCommand) -> Result<(), MapDriverError> {
    match command {
        MapCommand::Fit { genotype_path } => run_fit(&genotype_path),
        MapCommand::Project { genotype_path } => run_project(&genotype_path),
    }
}

fn run_fit(genotype_path: &Path) -> Result<(), MapDriverError> {
    println!("Starting HWE PCA fit for {}", genotype_path.display());

    let dataset = open_dataset(genotype_path)?;
    println!(
        "Detected {} samples across {} variants",
        dataset.n_samples(),
        dataset.n_variants()
    );

    let mut source = dataset.block_source()?;
    let model = HwePcaModel::fit(&mut source)?;

    println!(
        "Model fitted for {} samples and {} variants",
        model.n_samples(),
        model.n_variants()
    );

    persist_model(&dataset, &model)?;
    persist_sample_manifest(&dataset)?;
    persist_fit_summary(&dataset, &model)?;

    Ok(())
}

fn run_project(genotype_path: &Path) -> Result<(), MapDriverError> {
    println!("Starting projection for {}", genotype_path.display());

    let dataset = open_dataset(genotype_path)?;
    println!(
        "Loaded projection dataset with {} samples and {} variants",
        dataset.n_samples(),
        dataset.n_variants()
    );

    let model = load_model_for_projection(&dataset)?;
    println!("Model provides {} principal components", model.components());

    let mut source = dataset.block_source()?;
    let options = ProjectionOptions::default();
    let projector = model.projector();
    let result = projector.project_with_options(&mut source, &options)?;

    persist_projection_results(&dataset, &result)?;

    println!("Projection complete for {} samples", result.scores.nrows());

    Ok(())
}

fn open_dataset(path: &Path) -> Result<GenotypeDataset, MapDriverError> {
    Ok(GenotypeDataset::open(path)?)
}

fn persist_model(dataset: &GenotypeDataset, model: &HwePcaModel) -> Result<(), MapDriverError> {
    let model_path = dataset_output_path(dataset, "hwe.json");
    prepare_output_path(&model_path)?;

    let file = File::create(&model_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, model)?;
    writer.flush()?;

    println!("Saved HWE PCA model to {}", model_path.display());
    Ok(())
}

fn load_model_for_projection(dataset: &GenotypeDataset) -> Result<HwePcaModel, MapDriverError> {
    let model_path = dataset_output_path(dataset, "hwe.json");
    let file = File::open(&model_path)?;
    let reader = BufReader::new(file);
    let model: HwePcaModel = serde_json::from_reader(reader)?;

    if model.n_variants() != dataset.n_variants() {
        return Err(MapDriverError::InvalidState(format!(
            "Model expects {} variants but dataset provides {}",
            model.n_variants(),
            dataset.n_variants()
        )));
    }

    Ok(model)
}

fn persist_projection_results(
    dataset: &GenotypeDataset,
    result: &ProjectionResult,
) -> Result<(), MapDriverError> {
    let scores = result.scores.as_ref();
    let samples = dataset.samples();

    if samples.len() != scores.nrows() {
        return Err(MapDriverError::InvalidState(format!(
            "Projection scores contain {} rows but dataset has {} samples",
            scores.nrows(),
            samples.len()
        )));
    }

    let scores_path = dataset_output_path(dataset, "projection.scores.tsv");
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
    println!("Projection scores saved to {}", scores_path.display());

    if let Some(alignment) = &result.alignment {
        let alignment_path = dataset_output_path(dataset, "projection.alignment.tsv");
        prepare_output_path(&alignment_path)?;
        let mut writer = BufWriter::new(File::create(&alignment_path)?);

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
        println!(
            "Projection alignment factors saved to {}",
            alignment_path.display()
        );
    }

    Ok(())
}

fn persist_sample_manifest(dataset: &GenotypeDataset) -> Result<(), MapDriverError> {
    let manifest_path = dataset_output_path(dataset, "samples.tsv");
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
    println!("Sample manifest saved to {}", manifest_path.display());
    Ok(())
}

fn persist_fit_summary(
    dataset: &GenotypeDataset,
    model: &HwePcaModel,
) -> Result<(), MapDriverError> {
    let summary_path = dataset_output_path(dataset, "hwe.summary.tsv");
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
    println!("Fit summary saved to {}", summary_path.display());
    Ok(())
}

fn dataset_output_path(dataset: &GenotypeDataset, filename: &str) -> PathBuf {
    dataset.output_path(filename)
}

fn prepare_output_path(path: &Path) -> Result<(), MapDriverError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::fit::{DEFAULT_BLOCK_WIDTH, DenseBlockSource, HwePcaModel};
    use super::io::{DatasetBlockSource, GenotypeDataset};
    use super::project::ProjectionOptions;
    use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
    use std::error::Error;
    use std::path::Path;

    const HGDP_CHR20_BCF: &str = "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/\
         hgdp1kgp_chr20.filtered.SNV_INDEL.phased.shapeit5.bcf";
    const RNG_SEED: u64 = 0xC0FFEE5EEDBADD5;

    #[test]
    fn fit_and_project_split_hgdp_chr20() -> Result<(), Box<dyn Error>> {
        let dataset = GenotypeDataset::open(Path::new(HGDP_CHR20_BCF))
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let mut block_source = dataset
            .block_source()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        let n_samples = block_source.n_samples();
        let n_variants = block_source.n_variants();
        assert!(n_samples >= 3, "expected at least three samples for PCA");
        assert!(n_variants > 0, "expected at least one variant in dataset");
        assert_eq!(dataset.samples().len(), n_samples);

        let dense_matrix = collect_dense_matrix(&mut block_source)?;

        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let mut sample_indices: Vec<usize> = (0..n_samples).collect();
        sample_indices.shuffle(&mut rng);

        let min_train = 2usize;
        let min_inference = 1usize;
        let mut train_count = ((n_samples as f64) * 0.8).round() as usize;
        if train_count < min_train {
            train_count = min_train;
        }
        if train_count > n_samples.saturating_sub(min_inference) {
            train_count = n_samples - min_inference;
        }
        let inference_count = n_samples - train_count;
        assert!(inference_count >= min_inference);

        let train_indices = &sample_indices[..train_count];
        let inference_indices = &sample_indices[train_count..];

        let train_matrix =
            slice_samples_into_dense(&dense_matrix, n_samples, n_variants, train_indices);
        let inference_matrix =
            slice_samples_into_dense(&dense_matrix, n_samples, n_variants, inference_indices);

        let mut train_source = DenseBlockSource::new(&train_matrix, train_count, n_variants)
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let model = HwePcaModel::fit(&mut train_source)
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert_eq!(model.n_samples(), train_count);
        assert_eq!(model.n_variants(), n_variants);
        assert!(model.components() > 0);
        assert_eq!(model.explained_variance().len(), model.components());
        assert!(
            model
                .explained_variance()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        assert!(
            model
                .singular_values()
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );

        let variance_ratios = model.explained_variance_ratio();
        assert_eq!(variance_ratios.len(), model.components());
        let variance_ratio_sum: f64 = variance_ratios.iter().copied().sum();
        assert!(
            variance_ratio_sum <= 1.0 + 1e-9,
            "explained variance ratios must sum to at most 1 (observed {variance_ratio_sum})"
        );

        let mut inference_source =
            DenseBlockSource::new(&inference_matrix, inference_count, n_variants)
                .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let projection = model
            .projector()
            .project_with_options(&mut inference_source, &ProjectionOptions::default())
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert_eq!(projection.scores.nrows(), inference_count);
        assert_eq!(projection.scores.ncols(), model.components());
        assert!(projection.alignment.is_none());

        let scores = projection.scores.as_ref();
        for row in 0..scores.nrows() {
            for col in 0..scores.ncols() {
                let value = scores[(row, col)];
                assert!(
                    value.is_finite(),
                    "projection score for sample {row} PC {col} should be finite"
                );
            }
        }

        Ok(())
    }

    fn collect_dense_matrix(source: &mut DatasetBlockSource) -> Result<Vec<f64>, Box<dyn Error>> {
        let n_samples = source.n_samples();
        let n_variants = source.n_variants();
        let block_capacity = DEFAULT_BLOCK_WIDTH.max(1);
        let mut storage = vec![0.0f64; n_samples * block_capacity];
        let mut dense = vec![0.0f64; n_samples * n_variants];
        let mut emitted = 0usize;

        while emitted < n_variants {
            let filled = source
                .next_block_into(block_capacity, &mut storage)
                .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
            if filled == 0 {
                break;
            }

            for local_idx in 0..filled {
                let src_offset = local_idx * n_samples;
                let dest_offset = (emitted + local_idx) * n_samples;
                dense[dest_offset..dest_offset + n_samples]
                    .copy_from_slice(&storage[src_offset..src_offset + n_samples]);
            }

            emitted += filled;
        }

        assert_eq!(
            emitted, n_variants,
            "expected to materialize all {n_variants} variants but only processed {emitted}"
        );

        Ok(dense)
    }

    fn slice_samples_into_dense(
        data: &[f64],
        n_samples: usize,
        n_variants: usize,
        selected: &[usize],
    ) -> Vec<f64> {
        let mut subset = vec![0.0f64; selected.len() * n_variants];
        for variant_idx in 0..n_variants {
            let column = &data[variant_idx * n_samples..(variant_idx + 1) * n_samples];
            let dest_offset = variant_idx * selected.len();
            for (row_offset, &sample_idx) in selected.iter().enumerate() {
                subset[dest_offset + row_offset] = column[sample_idx];
            }
        }
        subset
    }
}
