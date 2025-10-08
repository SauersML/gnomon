use std::fmt;
use std::path::{Path, PathBuf};

use super::fit::{HwePcaError, HwePcaModel};
use super::io::{
    DatasetOutputError, GenotypeDataset, GenotypeIoError, ProjectionOutputPaths, load_hwe_model,
    save_fit_summary, save_hwe_model, save_projection_results, save_sample_manifest,
};
use super::project::ProjectionOptions;

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

impl From<DatasetOutputError> for MapDriverError {
    fn from(value: DatasetOutputError) -> Self {
        match value {
            DatasetOutputError::Io(err) => Self::Io(err),
            DatasetOutputError::Serialization(err) => Self::Serialization(err),
            DatasetOutputError::InvalidState(msg) => Self::InvalidState(msg),
        }
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

    let model_path = save_hwe_model(&dataset, &model)?;
    println!("Saved HWE PCA model to {}", model_path.display());

    let manifest_path = save_sample_manifest(&dataset)?;
    println!("Sample manifest saved to {}", manifest_path.display());

    let summary_path = save_fit_summary(&dataset, &model)?;
    println!("Fit summary saved to {}", summary_path.display());

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

    let model = load_hwe_model(&dataset)?;
    println!("Model provides {} principal components", model.components());

    let mut source = dataset.block_source()?;
    let options = ProjectionOptions::default();
    let projector = model.projector();
    let result = projector.project_with_options(&mut source, &options)?;

    let ProjectionOutputPaths { scores, alignment } = save_projection_results(&dataset, &result)?;

    println!("Projection scores saved to {}", scores.display());
    if let Some(path) = alignment {
        println!("Projection alignment factors saved to {}", path.display());
    }

    println!("Projection complete for {} samples", result.scores.nrows());

    Ok(())
}

fn open_dataset(path: &Path) -> Result<GenotypeDataset, MapDriverError> {
    Ok(GenotypeDataset::open(path)?)
}
