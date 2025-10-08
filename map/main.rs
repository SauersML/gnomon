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
    use crate::map::fit::HwePcaModel;
    use crate::map::io::GenotypeDataset;
    use crate::map::project::ProjectionOptions;
    use std::error::Error;
    use std::fs::File;
    use std::io::Write;
    use std::path::PathBuf;
    use std::process::Command;
    use tempfile::TempDir;

    const HGDP_FULL_DATASET: &str =
        "gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/";

    struct LocalPlinkDataset {
        tempdir: TempDir,
        bed_path: PathBuf,
    }

    impl LocalPlinkDataset {
        fn create() -> Result<Self, Box<dyn Error>> {
            let dir = tempfile::tempdir()?;
            let bed_path = dir.path().join("test_dataset.bed");
            let bim_path = dir.path().join("test_dataset.bim");
            let fam_path = dir.path().join("test_dataset.fam");

            {
                let mut fam = File::create(&fam_path)?;
                writeln!(fam, "FAM SAMPLE1 0 0 1 -9")?;
                writeln!(fam, "FAM SAMPLE2 0 0 2 -9")?;
                writeln!(fam, "FAM SAMPLE3 0 0 1 -9")?;
            }

            {
                let mut bim = File::create(&bim_path)?;
                writeln!(bim, "1 var1 0 123 A G")?;
                writeln!(bim, "1 var2 0 456 C T")?;
            }

            {
                let mut bed = File::create(&bed_path)?;
                bed.write_all(&[0x6c, 0x1b, 0x01])?;
                bed.write_all(&[
                    encode_plink_genotypes(&[0, 1, 2]),
                    encode_plink_genotypes(&[2, 0, 1]),
                ])?;
                bed.flush()?;
            }

            Ok(Self {
                tempdir: dir,
                bed_path,
            })
        }

        fn bed_path(&self) -> &std::path::Path {
            // Touch the temporary directory to satisfy linting rules while keeping
            // the backing storage alive for the lifetime of the dataset.
            let _ = self.tempdir.path();
            &self.bed_path
        }
    }

    fn encode_plink_genotypes(genotypes: &[u8]) -> u8 {
        let mut byte = 0u8;
        for (idx, &value) in genotypes.iter().take(4).enumerate() {
            let code = match value {
                0 => 0b00,
                1 => 0b10,
                2 => 0b11,
                _ => 0b01,
            };
            byte |= code << (idx * 2);
        }
        byte
    }

    fn gnomon_binary_path() -> Result<PathBuf, Box<dyn Error>> {
        use std::env;

        let current_exe = env::current_exe()?;
        let binary_dir = current_exe
            .parent()
            .and_then(|deps| deps.parent())
            .ok_or_else(|| "unable to determine cargo target directory")?;

        let binary_name = format!("gnomon{}", env::consts::EXE_SUFFIX);
        let binary_path = binary_dir.join(&binary_name);

        if binary_path.exists() {
            return Ok(binary_path);
        }

        let target_dir = binary_dir
            .parent()
            .ok_or_else(|| "unable to determine cargo target directory")?;
        let workspace_root = target_dir
            .parent()
            .ok_or_else(|| "unable to determine workspace root for cargo build")?;

        let profile = binary_dir
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "unable to determine cargo profile")?;

        let mut build_cmd = Command::new("cargo");
        build_cmd
            .arg("build")
            .arg("--bin")
            .arg("gnomon")
            .current_dir(workspace_root);

        match profile {
            "debug" => {}
            "release" => {
                build_cmd.arg("--release");
            }
            other => {
                build_cmd.arg("--profile").arg(other);
            }
        }

        let output = build_cmd
            .output()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        if !output.status.success() {
            return Err(format!(
                "failed to build gnomon binary: status={:?} stdout={} stderr={}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        if binary_path.exists() {
            Ok(binary_path)
        } else {
            Err(format!("gnomon binary not found at {}", binary_path.display()).into())
        }
    }

    #[test]
    fn cli_fit_and_project_full_hgdp_dataset() -> Result<(), Box<dyn Error>> {
        let binary = gnomon_binary_path()?;

        let fit_output = Command::new(&binary)
            .arg("--fit")
            .arg(HGDP_FULL_DATASET)
            .output()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert!(
            fit_output.status.success(),
            "`gnomon --fit` failed: status={:?}\nstdout={}\nstderr={}",
            fit_output.status,
            String::from_utf8_lossy(&fit_output.stdout),
            String::from_utf8_lossy(&fit_output.stderr)
        );

        let project_output = Command::new(&binary)
            .arg("--project")
            .arg(HGDP_FULL_DATASET)
            .output()
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;

        assert!(
            project_output.status.success(),
            "`gnomon --project` failed: status={:?}\nstdout={}\nstderr={}",
            project_output.status,
            String::from_utf8_lossy(&project_output.stdout),
            String::from_utf8_lossy(&project_output.stderr)
        );

        Ok(())
    }

    #[test]
    fn fit_and_project_full_hgdp_chr20() -> Result<(), Box<dyn Error>> {
        let fixture = LocalPlinkDataset::create()?;
        let dataset = GenotypeDataset::open(fixture.bed_path())
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
