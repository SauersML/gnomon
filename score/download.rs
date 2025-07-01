// ========================================================================================
//
//                               SCORE FILE DOWNLOADER
//
// ========================================================================================

use crate::reformat;
use dwldutil::{DLFile, Downloader};
use indicatif::ProgressStyle;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

// ========================================================================================
//                              PUBLIC API
// ========================================================================================

/// The primary entry point for the persistent download workflow.
///
/// This function synchronizes a local directory (`scores_dir`) with a list of
/// requested PGS IDs. It performs the following steps:
/// 1. Parses the list of requested PGS IDs.
/// 2. Checks `scores_dir` for any corresponding `.gnomon.tsv` files that already exist.
/// 3. For any missing files, it downloads them in parallel from the PGS Catalog.
/// 4. It reformats the newly downloaded files in parallel into the native format.
/// 5. It cleans up the intermediate downloaded files.
/// 6. It returns a complete list of paths to all requested `.gnomon.tsv` files,
///    including both pre-existing and newly created ones.
///
/// # Arguments
/// * `score_arg` - A comma-separated string of PGS IDs.
/// * `scores_dir` - The path to the permanent directory where score files should be
///   stored. This directory will be created if it does not exist.
///
/// # Returns
/// A `Result` containing a `Vec<PathBuf>` of paths to the final, native-format
/// score files on success, or a boxed dynamic error on failure.
pub fn resolve_and_download_scores(
    score_arg: &str,
    scores_dir: &Path,
) -> Result<Vec<PathBuf>, Box<dyn Error + Send + Sync>> {
    // --- Stage 1: Setup and State Synchronization ---
    fs::create_dir_all(scores_dir)?;
    eprintln!(
        "> Checking for score files in permanent directory: {}",
        scores_dir.display()
    );

    let requested_pgs_ids: BTreeSet<String> = score_arg
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if requested_pgs_ids.is_empty() {
        return Err("No valid PGS IDs provided in --score argument.".into());
    }

    for id in &requested_pgs_ids {
        if !id.starts_with("PGS") {
            return Err(Box::new(DownloadError::InvalidId(id.clone())));
        }
    }

    let mut existing_native_paths = Vec::new();
    let mut files_to_reformat = Vec::new();
    let mut ids_to_download = Vec::new();

    // Check for existing files: final .gnomon.tsv or intermediate .txt.gz
    for id in &requested_pgs_ids {
        let native_path = scores_dir.join(format!("{}.gnomon.tsv", id));
        let temp_gz_path = scores_dir.join(format!("{}.txt.gz", id));

        if native_path.exists() {
            existing_native_paths.push(native_path);
        } else if temp_gz_path.exists() {
            eprintln!(
                "> Found existing downloaded file for {}. Skipping download.",
                id
            );
            files_to_reformat.push((temp_gz_path, native_path));
        } else {
            ids_to_download.push(id.clone());
        }
    }

    eprintln!(
        "> Found {} existing reformatted score files. Found {} existing downloaded files. Need to download {}.",
        existing_native_paths.len(),
        files_to_reformat.len(),
        ids_to_download.len()
    );

    // --- Stage 2: Conditional Download ---
    if !ids_to_download.is_empty() {
        let newly_downloaded = download_missing_files(&ids_to_download, scores_dir)?;
        files_to_reformat.extend(newly_downloaded);
    }

    // --- Stage 3: Conditional Reformatting ---
    if files_to_reformat.is_empty() {
        eprintln!("> All required score files are already present and converted.");
        return Ok(existing_native_paths);
    }

    eprintln!(
        "> Reformatting {} score files into gnomon-native format...",
        files_to_reformat.len()
    );

    let new_native_paths = files_to_reformat
        .into_par_iter()
        .map(
            |(input_path, final_native_path)| -> Result<PathBuf, DownloadError> {
                reformat::reformat_pgs_file(&input_path, &final_native_path)
                    .map_err(DownloadError::Reformat)?;

                // Clean up the original downloaded file immediately after successful reformatting.
                fs::remove_file(&input_path)
                    .map_err(|e| DownloadError::Io(e, input_path.clone()))?;

                Ok(final_native_path)
            },
        )
        .collect::<Result<Vec<PathBuf>, _>>()?;

    eprintln!("> Reformatting complete.");

    // --- Stage 4: Final Aggregation ---
    existing_native_paths.extend(new_native_paths);
    Ok(existing_native_paths)
}

// ========================================================================================
//                             PRIVATE IMPLEMENTATION
// ========================================================================================

/// A specialized error type for the download and reformat workflow.
#[derive(Debug)]
pub enum DownloadError {
    Io(io::Error, PathBuf),
    Network(String),
    InvalidId(String),
    Reformat(reformat::ReformatError),
    RuntimeCreation(io::Error),
}

impl Display for DownloadError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            DownloadError::Io(e, path) => {
                write!(f, "I/O error for '{}': {}", path.display(), e)
            }
            DownloadError::Network(s) => write!(f, "Network download failed: {}", s),
            DownloadError::InvalidId(s) => write!(
                f,
                "Invalid ID format for '{}'. All IDs must start with 'PGS'.",
                s
            ),
            DownloadError::Reformat(e) => write!(f, "{}", e),
            DownloadError::RuntimeCreation(e) => write!(f, "Failed to create async runtime: {}", e),
        }
    }
}

impl Error for DownloadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            DownloadError::Io(e, _) => Some(e),
            DownloadError::Reformat(e) => Some(e),
            DownloadError::RuntimeCreation(e) => Some(e),
            _ => None,
        }
    }
}

/// Orchestrates the parallel download of all specified missing PGS IDs.
/// Returns a list of tuples, where each tuple contains the path to the downloaded
/// compressed file and the path where the final native file should be written.
fn download_missing_files(
    pgs_ids: &[String],
    target_dir: &Path,
) -> Result<Vec<(PathBuf, PathBuf)>, DownloadError> {
    eprintln!(
        "> Downloading {} missing score files from PGS Catalog...",
        pgs_ids.len()
    );

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(DownloadError::RuntimeCreation)?;

    runtime.block_on(async {
        let mut downloader = Downloader::new();
        let mut paths_to_reformat = Vec::with_capacity(pgs_ids.len());

        for id in pgs_ids {
            // Use the reliable HTTPS endpoint for downloads.
            let url = format!(
                "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{id}/ScoringFiles/Harmonized/{id}_hmPOS_GRCh38.txt.gz",
                id = id
            );
            // The compressed file is an intermediate artifact.
            let temp_gz_path = target_dir.join(format!("{}.txt.gz", id));
            // This is the final, desired output file.
            let final_native_path = target_dir.join(format!("{}.gnomon.tsv", id));

            let file_to_download = DLFile::new()
                .with_url(&url)
                .with_path(&temp_gz_path.to_string_lossy());

            // Each call to `add_file` consumes the
            // downloader and returns a new one, so we must re-assign it.
            downloader = downloader.add_file(file_to_download);
            paths_to_reformat.push((temp_gz_path, final_native_path));
        }

        // Use a progress bar style that does not require the total file size.
        let style = ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {msg}",
        )
        .unwrap()
        .progress_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏");
        let configured_downloader = downloader
            .with_style(style)
            .with_max_concurrent_downloads(12)
            .with_max_redirections(5);

        configured_downloader.start();

        Ok(paths_to_reformat)
    })
}
