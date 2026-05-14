//! Built-in pre-trained PCA models for gnomon.
//!
//! This module provides a registry of pre-trained HWE-scaled PCA models
//! that can be downloaded from GitHub and cached locally.

use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::shared::files::ensure_rustls_provider;
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
        sha256: "d9f7badd9e70a4c0a5ab3dd9242c6bb72abd88ece9fd3e6953197c299c34ea4c",
        build: "GRCh38",
        components: 20,
        variants: 650_000,
    },
    BuiltinModel {
        name: "hwe_1kg_hgdp_gsa_v3",
        url: "https://github.com/SauersML/gnomon/releases/download/models-v1/hwe_1kg_hgdp_gsa_v3.json.zst",
        sha256: "7797520466dae6c53d76b787e59951104c13827c4e64e76f236a4c32c56984c0",
        build: "GRCh38",
        components: 20,
        variants: 654_000,
    },
    BuiltinModel {
        name: "hwe_1kg_hgdp_gda_v1",
        url: "https://github.com/SauersML/gnomon/releases/download/models-v1/hwe_1kg_hgdp_gda_v1.json.zst",
        sha256: "52a7b7d2369c51b926bf00f3b5b156660f2cccca3a963c91c521c6bcacfd0b94",
        build: "GRCh38",
        components: 20,
        variants: 1_900_000,
    },
    BuiltinModel {
        name: "hwe_1kg_hgdp_intersection",
        url: "https://github.com/SauersML/gnomon/releases/download/models-v1/hwe_1kg_hgdp_intersection.json.zst",
        sha256: "bbbd08402d8d9e8eda8a3481511300fc222409a2874274622b8d7ee33aff22ad",
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
    #[error("built-in model {0} is missing a required SHA256 digest")]
    MissingHash(&'static str),
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

fn cached_model_digest_path(model: &BuiltinModel) -> io::Result<PathBuf> {
    let cache = cache_dir()?;
    Ok(cache.join(format!("{}.json.sha256", model.name)))
}

/// Check if a model is already cached (decompressed).
pub fn is_cached(model: &BuiltinModel) -> bool {
    cached_model_is_valid(model).unwrap_or(false)
}

/// Download and cache a model, returning the path to the decompressed JSON.
///
/// If the model is already cached, returns the cached path immediately.
pub fn ensure_model(model: &BuiltinModel) -> Result<PathBuf, BuiltinModelError> {
    if model.sha256.is_empty() {
        return Err(BuiltinModelError::MissingHash(model.name));
    }

    let json_path = cached_model_path(model)?;
    let digest_path = cached_model_digest_path(model)?;

    if cached_model_is_valid_at(&json_path, &digest_path, model.sha256)? {
        eprintln!("Using cached model: {}", json_path.display());
        return Ok(json_path);
    }
    if json_path.exists() {
        eprintln!(
            "Cached model digest is missing or stale; refreshing {}",
            json_path.display()
        );
        let _ = fs::remove_file(&json_path);
        let _ = fs::remove_file(&digest_path);
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

    eprintln!("  Verifying SHA256...");
    let actual = compute_sha256(&zst_path)?;
    if actual != model.sha256 {
        let _ = fs::remove_file(&zst_path);
        return Err(BuiltinModelError::HashMismatch {
            expected: model.sha256.to_string(),
            actual,
        });
    }
    eprintln!("  Hash verified");

    // Decompress to temp file first
    eprintln!("  Decompressing...");
    if let Err(e) = decompress_zstd(&zst_path, &json_temp_path) {
        // Clean up both files on failure
        let _ = fs::remove_file(&zst_path);
        let _ = fs::remove_file(&json_temp_path);
        return Err(e);
    }

    // Atomic rename to final location (only after successful decompression)
    let json_sha256 = compute_sha256(&json_temp_path)?;
    fs::rename(&json_temp_path, &json_path)?;
    fs::write(
        &digest_path,
        format!(
            "source_zst_sha256\t{}\njson_sha256\t{}\n",
            model.sha256, json_sha256
        ),
    )?;
    eprintln!("  Decompressed to: {}", json_path.display());

    // Clean up compressed file
    let _ = fs::remove_file(&zst_path);

    Ok(json_path)
}

fn cached_model_is_valid(model: &BuiltinModel) -> Result<bool, BuiltinModelError> {
    let json_path = cached_model_path(model)?;
    let digest_path = cached_model_digest_path(model)?;
    cached_model_is_valid_at(&json_path, &digest_path, model.sha256)
}

fn cached_model_is_valid_at(
    json_path: &Path,
    digest_path: &Path,
    expected_zst_sha256: &str,
) -> Result<bool, BuiltinModelError> {
    if !json_path.exists() || expected_zst_sha256.is_empty() {
        return Ok(false);
    }

    let Some((source_zst_sha256, json_sha256)) = read_cache_digest(digest_path)? else {
        return Ok(false);
    };
    if source_zst_sha256 != expected_zst_sha256 {
        return Ok(false);
    }

    let actual_json_sha256 = compute_sha256(json_path)?;
    Ok(actual_json_sha256 == json_sha256)
}

fn read_cache_digest(path: &Path) -> io::Result<Option<(String, String)>> {
    let text = match fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    let mut source_zst_sha256 = None;
    let mut json_sha256 = None;
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        match (fields.next(), fields.next(), fields.next()) {
            (Some("source_zst_sha256"), Some(value), None) => {
                source_zst_sha256 = Some(value.to_string());
            }
            (Some("json_sha256"), Some(value), None) => {
                json_sha256 = Some(value.to_string());
            }
            _ => {}
        }
    }
    Ok(source_zst_sha256.zip(json_sha256))
}

/// Download a file from a URL to a local path.
fn download_file(url: &str, dest: &Path) -> Result<(), BuiltinModelError> {
    ensure_rustls_provider();
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
