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

#[cfg(test)]
mod tests {
    use super::fit::HwePcaModel;
    use super::io::GenotypeDataset;
    use super::project::ProjectionOptions;
    use std::error::Error;
    use std::path::Path;

    const HGDP_CHR20_BCF: &str = "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/\
         hgdp1kgp_chr20.filtered.SNV_INDEL.phased.shapeit5.bcf";

    #[test]
    fn fit_and_project_full_hgdp_chr20() -> Result<(), Box<dyn Error>> {
        let dataset = GenotypeDataset::open(Path::new(HGDP_CHR20_BCF))
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let n_samples = dataset.n_samples();
        let n_variants = dataset.n_variants();
        assert!(n_samples >= 3, "expected at least three samples for PCA");
        assert!(n_variants > 0, "expected at least one variant in dataset");
        assert_eq!(dataset.samples().len(), n_samples);

        let mut train_source = dataset
            .block_source()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let model = HwePcaModel::fit(&mut train_source)
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert_eq!(model.n_samples(), n_samples);
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

        drop(train_source);

        let mut inference_source = dataset
            .block_source()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let projection = model
            .projector()
            .project_with_options(&mut inference_source, &ProjectionOptions::default())
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert_eq!(projection.scores.nrows(), n_samples);
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
}
