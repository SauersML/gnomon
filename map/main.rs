use std::fmt;
use std::path::{Path, PathBuf};

use super::fit::{HwePcaError, HwePcaModel, VariantBlockSource};
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
    NotYetImplemented(&'static str),
}

impl fmt::Display for MapDriverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hwe(err) => write!(f, "HWE PCA error: {err}"),
            Self::Io(err) => write!(f, "I/O error: {err}"),
            Self::NotYetImplemented(feature) => {
                write!(f, "{feature} is not implemented yet")
            }
        }
    }
}

impl std::error::Error for MapDriverError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Hwe(err) => Some(err),
            Self::Io(err) => Some(err),
            Self::NotYetImplemented(_) => None,
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

/// Execute the provided [`MapCommand`].
pub fn run(command: MapCommand) -> Result<(), MapDriverError> {
    match command {
        MapCommand::Fit { genotype_path } => run_fit(&genotype_path),
        MapCommand::Project { genotype_path } => run_project(&genotype_path),
    }
}

fn run_fit(genotype_path: &Path) -> Result<(), MapDriverError> {
    println!("Starting HWE PCA fit for {}", genotype_path.display());

    let mut source = open_variant_source(genotype_path);
    let model = HwePcaModel::fit(&mut source)?;

    println!(
        "Model fitted for {} samples and {} variants",
        model.n_samples(),
        model.n_variants()
    );

    persist_model(genotype_path, &model)?;

    Ok(())
}

fn run_project(genotype_path: &Path) -> Result<(), MapDriverError> {
    println!("Starting projection for {}", genotype_path.display());

    let model = load_model_for_projection(genotype_path)?;
    let mut source = open_projection_source(genotype_path);
    let options = ProjectionOptions::default();
    let projector = model.projector();
    let result = projector.project_with_options(&mut source, &options)?;

    persist_projection_results(genotype_path, &result)?;

    println!(
        "Projection complete for {} samples",
        result.scores.nrows()
    );

    Ok(())
}

fn open_variant_source(_path: &Path) -> PlaceholderVariantSource {
    PlaceholderVariantSource
}

fn open_projection_source(_path: &Path) -> PlaceholderVariantSource {
    PlaceholderVariantSource
}

fn persist_model(_path: &Path, _model: &HwePcaModel) -> Result<(), MapDriverError> {
    Err(MapDriverError::NotYetImplemented(
        "Model serialization for map::fit",
    ))
}

fn load_model_for_projection(_path: &Path) -> Result<HwePcaModel, MapDriverError> {
    Err(MapDriverError::NotYetImplemented(
        "Model loading for map::project",
    ))
}

fn persist_projection_results(
    _path: &Path,
    _result: &ProjectionResult,
) -> Result<(), MapDriverError> {
    Err(MapDriverError::NotYetImplemented(
        "Projection output serialization",
    ))
}

#[derive(Debug)]
struct PlaceholderVariantSource;

#[derive(Debug)]
struct PlaceholderVariantSourceError;

impl fmt::Display for PlaceholderVariantSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "placeholder variant source is not implemented")
    }
}

impl std::error::Error for PlaceholderVariantSourceError {}

impl VariantBlockSource for PlaceholderVariantSource {
    type Error = PlaceholderVariantSourceError;

    fn n_samples(&self) -> usize {
        todo!("variant source loading is not implemented")
    }

    fn n_variants(&self) -> usize {
        todo!("variant source loading is not implemented")
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        todo!("variant source loading is not implemented")
    }

    fn next_block_into(
        &mut self,
        _max_variants: usize,
        _storage: &mut [f64],
    ) -> Result<usize, Self::Error> {
        todo!("variant source loading is not implemented")
    }
}
