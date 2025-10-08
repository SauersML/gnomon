use std::fmt;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use super::fit::{HwePcaError, HwePcaModel};
use super::project::{ProjectionOptions, ProjectionResult};
use super::io::{PlinkDataset, PlinkIoError};

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
    Plink(PlinkIoError),
    Serialization(serde_json::Error),
    InvalidState(String),
}

impl fmt::Display for MapDriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hwe(err) => write!(f, "HWE PCA error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::Plink(err) => write!(f, "PLINK dataset error: {err}"),
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
            Self::Plink(err) => Some(err),
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

impl From<PlinkIoError> for MapDriverError {
    fn from(value: PlinkIoError) -> Self {
        Self::Plink(value)
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

    let mut source = dataset.block_source();
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
    println!(
        "Model provides {} principal components",
        model.components()
    );

    let mut source = dataset.block_source();
    let options = ProjectionOptions::default();
    let projector = model.projector();
    let result = projector.project_with_options(&mut source, &options)?;

    persist_projection_results(&dataset, &result)?;

    println!(
        "Projection complete for {} samples",
        result.scores.nrows()
    );

    Ok(())
}

fn open_dataset(path: &Path) -> Result<PlinkDataset, MapDriverError> {
    let dataset = PlinkDataset::open(path)?;
    Ok(dataset)
}

fn persist_model(dataset: &PlinkDataset, model: &HwePcaModel) -> Result<(), MapDriverError> {
    let model_path = dataset_output_path(dataset, "hwe.json");
    prepare_output_path(&model_path)?;

    let file = File::create(&model_path)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, model)?;
    writer.flush()?;

    println!("Saved HWE PCA model to {}", model_path.display());
    Ok(())
}

fn load_model_for_projection(dataset: &PlinkDataset) -> Result<HwePcaModel, MapDriverError> {
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
    dataset: &PlinkDataset,
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

fn persist_sample_manifest(dataset: &PlinkDataset) -> Result<(), MapDriverError> {
    let manifest_path = dataset_output_path(dataset, "samples.tsv");
    prepare_output_path(&manifest_path)?;
    let mut writer = BufWriter::new(File::create(&manifest_path)?);

    writeln!(
        writer,
        "FID\tIID\tPAT\tMAT\tSEX\tPHENOTYPE"
    )?;

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

fn persist_fit_summary(dataset: &PlinkDataset, model: &HwePcaModel) -> Result<(), MapDriverError> {
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

fn dataset_output_path(dataset: &PlinkDataset, extension: &str) -> PathBuf {
    let bed_path = dataset.bed_path();
    if is_remote_path(bed_path) {
        let stem = bed_path
            .file_stem()
            .map(|s| s.to_os_string())
            .unwrap_or_else(|| "dataset".into());
        let mut local = PathBuf::from(stem);
        local.set_extension(extension);
        local
    } else {
        let mut local = bed_path.to_path_buf();
        local.set_extension(extension);
        local
    }
}

fn prepare_output_path(path: &Path) -> Result<(), MapDriverError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

fn is_remote_path(path: &Path) -> bool {
    path.to_str()
        .map(|s| s.starts_with("gs://"))
        .unwrap_or(false)
}
