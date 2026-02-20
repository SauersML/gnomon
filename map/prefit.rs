//! Built-in pre-trained PCA models for gnomon.
//!
//! This module provides a registry of pre-trained HWE-scaled PCA models
//! that can be downloaded from GitHub and cached locally.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use thiserror::Error;

/// Information about a built-in pre-trained model.
#[derive(Clone, Debug)]
pub struct BuiltinModel {
    /// Human-readable name (e.g., "hwe_1kg_hgdp_gsa_v3")
    pub name: &'static str,
    /// URL to download the compressed model (.json.zst)
    pub url: &'static str,
    /// Expected SHA256 hash of the compressed file (hex-encoded)
    pub sha256: &'static str,
    /// Genome build used for training (e.g., "GRCh38")
    pub build: &'static str,
    /// Number of components in the model
    pub components: usize,
    /// Approximate number of variants
    pub variants: usize,
}

/// Registry of all built-in models.
pub const BUILTIN_MODELS: &[BuiltinModel] = &[
    BuiltinModel {
        name: "hwe_1kg_hgdp_gsa_v2",
        url: "https://github.com/SauersML/gnomon/releases/download/models-v1/hwe_1kg_hgdp_gsa_v2.json.zst",
        sha256: "", // Will be populated after first training run
        build: "GRCh38",
        components: 20,
        variants: 650_000,
    },
    BuiltinModel {
        name: "hwe_1kg_hgdp_gsa_v3",
        url: "https://github.com/SauersML/gnomon/releases/download/models-v1/hwe_1kg_hgdp_gsa_v3.json.zst",
        sha256: "", // Will be populated after first training run
        build: "GRCh38",
        components: 20,
        variants: 654_000,
    },
    BuiltinModel {
        name: "hwe_1kg_hgdp_gda_v1",
        url: "https://github.com/SauersML/gnomon/releases/download/models-v1/hwe_1kg_hgdp_gda_v1.json.zst",
        sha256: "", // Will be populated after first training run
        build: "GRCh38",
        components: 20,
        variants: 1_900_000,
    },
    BuiltinModel {
        name: "hwe_1kg_hgdp_intersection",
        url: "https://github.com/SauersML/gnomon/releases/download/models-v1/hwe_1kg_hgdp_intersection.json.zst",
        sha256: "", // Will be populated after first training run
        build: "GRCh38",
        components: 20,
        variants: 56_331,
    },
];

/// Error type for built-in model operations.
#[derive(Debug, Error)]
pub enum BuiltinModelError {
    #[error("unknown model: {0}")]
    UnknownModel(String),
    #[error("failed to download model: {0}")]
    Download(String),
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("decompression failed: {0}")]
    Decompression(String),
    #[error("model deserialization failed: {0}")]
    Deserialization(String),
}

/// Look up a model by name.
pub fn lookup_model(name: &str) -> Option<&'static BuiltinModel> {
    BUILTIN_MODELS.iter().find(|m| m.name == name)
}

/// List all available built-in model names.
pub fn list_model_names() -> Vec<&'static str> {
    BUILTIN_MODELS.iter().map(|m| m.name).collect()
}

/// Get the cache directory for gnomon models.
pub fn cache_dir() -> io::Result<PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "could not determine home directory",
        )
    })?;
    let cache = home.join(".gnomon").join("models");
    fs::create_dir_all(&cache)?;
    Ok(cache)
}

/// Get the path to a cached model file.
pub fn cached_model_path(model: &BuiltinModel) -> io::Result<PathBuf> {
    let cache = cache_dir()?;
    Ok(cache.join(format!("{}.json", model.name)))
}

/// Check if a model is already cached (decompressed).
pub fn is_cached(model: &BuiltinModel) -> bool {
    cached_model_path(model)
        .map(|p| p.exists())
        .unwrap_or(false)
}

/// Download and cache a model, returning the path to the decompressed JSON.
///
/// If the model is already cached, returns the cached path immediately.
pub fn ensure_model(model: &BuiltinModel) -> Result<PathBuf, BuiltinModelError> {
    let json_path = cached_model_path(model)?;

    if json_path.exists() {
        eprintln!("Using cached model: {}", json_path.display());
        return Ok(json_path);
    }

    eprintln!("Downloading model '{}' from GitHub...", model.name);
    eprintln!("  URL: {}", model.url);

    // Download to a temporary .zst file
    let cache = cache_dir()?;
    let zst_path = cache.join(format!("{}.json.zst", model.name));
    // Use a temp path for decompression to avoid cache corruption
    let json_temp_path = cache.join(format!("{}.json.tmp", model.name));

    // Clean up any leftover temp files from previous failed attempts
    let _ = fs::remove_file(&zst_path);
    let _ = fs::remove_file(&json_temp_path);

    download_file(model.url, &zst_path)?;
    eprintln!("  Downloaded to: {}", zst_path.display());

    // Verify hash if specified
    if !model.sha256.is_empty() {
        eprintln!("  Verifying SHA256...");
        let actual = compute_sha256(&zst_path)?;
        if actual != model.sha256 {
            let _ = fs::remove_file(&zst_path);
            return Err(BuiltinModelError::HashMismatch {
                expected: model.sha256.to_string(),
                actual,
            });
        }
        eprintln!("  Hash verified ✓");
    }

    // Decompress to temp file first
    eprintln!("  Decompressing...");
    if let Err(e) = decompress_zstd(&zst_path, &json_temp_path) {
        // Clean up both files on failure
        let _ = fs::remove_file(&zst_path);
        let _ = fs::remove_file(&json_temp_path);
        return Err(e);
    }

    // Atomic rename to final location (only after successful decompression)
    fs::rename(&json_temp_path, &json_path)?;
    eprintln!("  Decompressed to: {}", json_path.display());

    // Clean up compressed file
    let _ = fs::remove_file(&zst_path);

    Ok(json_path)
}

/// Download a file from a URL to a local path.
fn download_file(url: &str, dest: &Path) -> Result<(), BuiltinModelError> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| BuiltinModelError::Download(e.to_string()))?;

    if response.status() != 200 {
        return Err(BuiltinModelError::Download(format!(
            "HTTP {}: {}",
            response.status(),
            response.status_text()
        )));
    }

    let mut reader = response.into_reader();
    let file = File::create(dest)?;
    let mut writer = BufWriter::new(file);

    io::copy(&mut reader, &mut writer)?;
    writer.flush()?;

    Ok(())
}

/// Compute SHA256 hash of a file.
fn compute_sha256(path: &Path) -> Result<String, BuiltinModelError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();

    let mut buffer = [0u8; 8192];
    loop {
        let n = reader.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    let hash = hasher.finalize();
    Ok(hex::encode(hash))
}

/// Decompress a zstd file.
fn decompress_zstd(src: &Path, dest: &Path) -> Result<(), BuiltinModelError> {
    let input = File::open(src)?;
    let reader = BufReader::new(input);
    let mut decoder = zstd::stream::Decoder::new(reader)
        .map_err(|e| BuiltinModelError::Decompression(e.to_string()))?;

    let output = File::create(dest)?;
    let mut writer = BufWriter::new(output);

    io::copy(&mut decoder, &mut writer)?;
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_model() {
        let model = lookup_model("hwe_1kg_hgdp_gsa_v3");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.build, "GRCh38");
        assert_eq!(model.components, 20);
    }

    #[test]
    fn test_unknown_model() {
        let model = lookup_model("nonexistent_model");
        assert!(model.is_none());
    }

    #[test]
    fn test_list_model_names() {
        let names = list_model_names();
        assert!(names.contains(&"hwe_1kg_hgdp_gsa_v2"));
        assert!(names.contains(&"hwe_1kg_hgdp_gsa_v3"));
        assert!(names.contains(&"hwe_1kg_hgdp_gda_v1")); // GDA is still available via release
        assert!(names.contains(&"hwe_1kg_hgdp_intersection"));
    }
}
