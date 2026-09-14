// ========================================================================================
//
//               VCF/BCF/DTC to PLINK Conversion Module
//
// ========================================================================================
//
// This module provides transparent conversion of VCF/BCF/DTC text files to PLINK format
// for use with the gnomon scoring pipeline. It includes caching to avoid re-conversion
// on repeated runs, and automatic reference genome downloading for DTC files.

use convert_genome::cli::Sex;
use convert_genome::conversion::{
    DEFAULT_MAX_PARSE_ERROR_RATIO, DEFAULT_MIN_BUILD_CONFIDENCE, DEFAULT_MIN_EMITTED_VARIANTS,
};
use convert_genome::input::InputFormat as ConvertInputFormat;
use convert_genome::{ConversionConfig, OutputFormat, convert_dtc_file};

// Re-export the underlying Sex enum so callers (e.g. `score/main.rs`) can
// construct `EnsurePlinkOptions { inferred_sex: Some(...) }` without a
// direct dependency on `convert_genome`.
pub use convert_genome::cli::Sex as ConvertSex;
use flate2::read::MultiGzDecoder;
use infer_sex::{GenomeBuild, InferredSex};
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::terms::infer_first_sample_sex;

/// Reference genome sources: (FASTA URL, optional FAI URL)
/// Sources with pre-built .fai indexes are preferred as they're more reliable
const GRCH37_SOURCES: &[(&str, Option<&str>)] = &[
    // 1000 Genomes - has pre-built index (most reliable)
    (
        "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.gz",
        Some(
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/reference/human_g1k_v37.fasta.fai",
        ),
    ),
    // Hail (Google Cloud) - fast, has index
    (
        "https://storage.googleapis.com/hail-common/references/human_g1k_v37.fasta.gz",
        Some("https://storage.googleapis.com/hail-common/references/human_g1k_v37.fasta.fai"),
    ),
    // Illumina DRAGEN S3 - uncompressed, no index
    (
        "https://ilmn-dragen-giab-samples.s3.amazonaws.com/FASTA/GRCh37.fa",
        None,
    ),
];

const GRCH38_SOURCES: &[(&str, Option<&str>)] = &[
    // Ensembl indexed - has pre-built index (most reliable)
    (
        "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna_index/Homo_sapiens.GRCh38.dna.toplevel.fa.gz",
        Some(
            "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna_index/Homo_sapiens.GRCh38.dna.toplevel.fa.gz.fai",
        ),
    ),
    // UCSC - fast, no index
    (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/latest/hg38.fa.gz",
        None,
    ),
    // Illumina DRAGEN S3 - uncompressed, no index
    (
        "https://ilmn-dragen-giab-samples.s3.amazonaws.com/FASTA/hg38.fa",
        None,
    ),
];

/// Supported input formats for genotype data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// PLINK binary format (.bed/.bim/.fam)
    Plink,
    /// Variant Call Format text (.vcf, .vcf.gz)
    Vcf,
    /// Binary Call Format (.bcf)
    Bcf,
    /// Direct-to-consumer text format (23andMe, AncestryDNA, etc.)
    Dtc,
}

fn parse_genome_build_hint(build: &str) -> Option<GenomeBuild> {
    let lower = build.to_ascii_lowercase();
    if lower.contains("37") || lower.contains("hg19") || lower.contains("grch37") {
        Some(GenomeBuild::Build37)
    } else if lower.contains("38") || lower.contains("hg38") || lower.contains("grch38") {
        Some(GenomeBuild::Build38)
    } else {
        None
    }
}

fn to_convert_sex(inferred: InferredSex) -> Sex {
    match inferred {
        InferredSex::Male => Sex::Male,
        InferredSex::Female => Sex::Female,
        InferredSex::Indeterminate => Sex::Unknown,
    }
}

/// Detects the input format based on file extension.
///
/// Returns `None` if the format cannot be determined.
pub fn detect_input_format(path: &Path) -> Option<InputFormat> {
    let path_str = path.to_string_lossy().to_lowercase();

    // Check for PLINK format (by extension or existence of the fileset).
    // `.pgen` is PLINK 2's genotype table and is read through the same path.
    if path_str.ends_with(".bed") || path_str.ends_with(".pgen") {
        return Some(InputFormat::Plink);
    }

    // Remote PLINK directories cannot be inspected with `Path::exists`. The score
    // resolver validates every discovered bed/bim/fam or pgen/pvar/psam triad after
    // format dispatch, so a GCS directory is unambiguously the multi-fileset PLINK
    // form accepted by `gnomon score`.
    if path_str.starts_with("gs://") && (path_str.ends_with('/') || path_str.ends_with("/*")) {
        return Some(InputFormat::Plink);
    }

    // Check for DTC text format (must check before PLINK prefix detection)
    if path_str.ends_with(".txt") || path_str.ends_with(".csv") {
        return Some(InputFormat::Dtc);
    }

    // Check if it's a prefix pointing to a PLINK fileset, of either version.
    if !path_str.ends_with(".vcf") && !path_str.ends_with(".vcf.gz") && !path_str.ends_with(".bcf")
    {
        // Not `with_extension`: a prefix may carry dots of its own
        // (`acaf_threshold.chr22`), which that would truncate.
        let stem =
            crate::score::prepare::strip_fileset_extension(&path.to_string_lossy()).to_string();
        let exists = |ext: &str| Path::new(&format!("{stem}.{ext}")).exists();
        if (exists("bed") && exists("bim") && exists("fam"))
            || (exists("pgen") && exists("pvar") && exists("psam"))
        {
            return Some(InputFormat::Plink);
        }
    }

    // Check for VCF format
    if path_str.ends_with(".vcf") || path_str.ends_with(".vcf.gz") {
        return Some(InputFormat::Vcf);
    }

    // Check for BCF format
    if path_str.ends_with(".bcf") {
        return Some(InputFormat::Bcf);
    }

    // Directory (could contain PLINK files)
    if path.is_dir() {
        return Some(InputFormat::Plink);
    }

    None
}

/// Returns the gnomon cache directory (~/.gnomon or fallback to current dir)
fn get_gnomon_cache_dir() -> PathBuf {
    dirs::home_dir()
        .map(|h| h.join(".gnomon"))
        .unwrap_or_else(|| PathBuf::from(".gnomon"))
}

/// Ensures a reference genome is available, downloading if necessary.
///
/// Tries multiple mirror URLs with fallback if one fails.
/// Downloads both FASTA and .fai index when available.
/// Returns the path to the uncompressed reference FASTA file.
fn ensure_reference_genome(build: &str) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let cache_dir = get_gnomon_cache_dir().join("refs");
    fs::create_dir_all(&cache_dir)?;

    let (sources, canonical_filename) =
        if build.contains("37") || build.to_lowercase().contains("hg19") {
            (GRCH37_SOURCES, "human_g1k_v37.fasta")
        } else {
            (GRCH38_SOURCES, "GRCh38_reference.fa")
        };

    let ref_path = cache_dir.join(canonical_filename);
    let fai_path = cache_dir.join(format!("{}.fai", canonical_filename));

    // A cached reference is used only if it was published whole.
    if ref_path.exists() {
        if reference_is_complete(&ref_path)? {
            eprintln!("> Using cached reference genome: {}", ref_path.display());
            return Ok(ref_path);
        }
        eprintln!(
            "> Cached reference genome '{}' is incomplete; downloading it again.",
            ref_path.display()
        );
    }
    // Nothing from an earlier attempt is trusted: an index or completion marker
    // beside an incomplete reference describes different bytes.
    for stale in [
        ref_path.clone(),
        fai_path.clone(),
        completion_marker_path(&ref_path),
    ] {
        match fs::remove_file(&stale) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }

    eprintln!("> Reference genome not found locally.");
    eprintln!("> Downloading {} reference (~900MB)...", build);

    // Try each source until one works
    let mut last_error = String::new();
    for (i, (fasta_url, fai_url)) in sources.iter().enumerate() {
        eprintln!("> Trying source {}/{}: {}", i + 1, sources.len(), fasta_url);

        // Download .fai index first if available (small file, fast)
        if let Some(fai_url) = fai_url {
            eprintln!("> Downloading index file...");
            if let Err(e) = download_file(fai_url, &fai_path) {
                eprintln!("> Warning: Failed to download index: {}", e);
                // Continue anyway - we'll try to create the index later
            }
        }

        // Download FASTA to temp file
        let temp_path = if fasta_url.ends_with(".gz") {
            cache_dir.join(format!("{}.gz.tmp", canonical_filename))
        } else {
            cache_dir.join(format!("{}.tmp", canonical_filename))
        };

        match download_with_progress(fasta_url, &temp_path) {
            Ok(()) => {
                // Decompress if needed
                if fasta_url.ends_with(".gz") {
                    eprintln!("> Decompressing reference genome...");
                    match decompress_gz(&temp_path, &ref_path) {
                        Ok(()) => {
                            let _ = fs::remove_file(&temp_path);
                            mark_reference_complete(&ref_path)?;
                            eprintln!("> Reference genome cached at: {}", ref_path.display());
                            return Ok(ref_path);
                        }
                        Err(e) => {
                            let _ = fs::remove_file(&temp_path);
                            let _ = fs::remove_file(&fai_path);
                            last_error = format!("Decompression failed: {}", e);
                            eprintln!("> {}. Trying next source...", last_error);
                            continue;
                        }
                    }
                } else {
                    // Just rename the temp file
                    fs::rename(&temp_path, &ref_path)?;
                    mark_reference_complete(&ref_path)?;
                    eprintln!("> Reference genome cached at: {}", ref_path.display());
                    return Ok(ref_path);
                }
            }
            Err(e) => {
                let _ = fs::remove_file(&temp_path);
                let _ = fs::remove_file(&fai_path);
                last_error = e.to_string();
                eprintln!("> Source failed: {}. Trying next...", last_error);
            }
        }
    }

    Err(format!(
        "Failed to download reference genome from all {} sources. Last error: {}",
        sources.len(),
        last_error
    )
    .into())
}

/// Simple file download without progress (for small files like .fai)
fn download_file(url: &str, dest: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {}", e))?;
    let mut reader = response.into_reader();
    crate::output::write_atomically(dest, |writer| {
        io::copy(&mut reader, writer)?;
        Ok(())
    })?;
    Ok(())
}

/// The empty block that ends every complete BGZF file (SAM/BAM specification).
const BGZF_EOF_BLOCK: [u8; 28] = [
    0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, 0x42, 0x43, 0x02, 0x00,
    0x1b, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Decompresses a gzipped file into `dest`, which appears only once complete.
///
/// Every gzip member is read. Ensembl's indexed references are BGZF, a series of
/// members, and a single-member decoder kept only the first 64 KiB of chr1. A BGZF
/// file must also end in its end-of-file block, so one cut at a member boundary,
/// which still decodes cleanly, is refused instead of published as a shorter genome.
fn decompress_gz(src: &Path, dest: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut input = File::open(src)?;
    let mut header = Vec::with_capacity(16);
    (&mut input).take(16).read_to_end(&mut header)?;
    let is_bgzf =
        header.len() >= 14 && header[..4] == [0x1f, 0x8b, 0x08, 0x04] && &header[12..14] == b"BC";
    if is_bgzf {
        let len = input.metadata()?.len();
        let mut tail = [0u8; BGZF_EOF_BLOCK.len()];
        if len >= tail.len() as u64 {
            input.seek(SeekFrom::Start(len - tail.len() as u64))?;
            input.read_exact(&mut tail)?;
        }
        if tail != BGZF_EOF_BLOCK {
            return Err(format!(
                "'{}' is BGZF but lacks its end-of-file block; the download is incomplete",
                src.display()
            )
            .into());
        }
    }
    input.seek(SeekFrom::Start(0))?;
    let mut decoder = MultiGzDecoder::new(BufReader::with_capacity(1 << 20, input));
    crate::output::write_atomically(dest, |writer| {
        io::copy(&mut decoder, writer)?;
        Ok(())
    })?;
    Ok(())
}

/// Downloads a file with progress indication
fn download_with_progress(url: &str, dest: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
    let response = ureq::get(url)
        .call()
        .map_err(|e| format!("Download failed: {}", e))?;

    let content_length = response
        .header("Content-Length")
        .and_then(|s| s.parse::<u64>().ok());

    let mut reader = response.into_reader();

    // Write to a temporary file first, then rename
    let temp_path = dest.with_extension("tmp");
    let file = File::create(&temp_path)?;
    let mut writer = BufWriter::new(file);

    let mut buffer = [0u8; 65536]; // 64KB buffer
    let mut downloaded: u64 = 0;
    let mut last_percent = 0;

    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        writer.write_all(&buffer[..bytes_read])?;
        downloaded += bytes_read as u64;

        // Update progress every 5%
        if let Some(total) = content_length {
            let percent = ((downloaded as f64 / total as f64) * 100.0) as u64;
            if percent >= last_percent + 5 {
                eprint!("\r> Downloading... {}%", percent);
                last_percent = percent;
            }
        }
    }

    writer.flush()?;
    drop(writer);
    if let Some(total) = content_length
        && downloaded != total
    {
        let _ = fs::remove_file(&temp_path);
        return Err(format!("Download of {url} ended after {downloaded} of {total} bytes").into());
    }
    eprintln!("\r> Download complete.          ");

    // Atomic rename
    fs::rename(&temp_path, dest)?;

    Ok(())
}

/// Human reference downloads name at least chromosomes 1-22, X and Y.
const MIN_REFERENCE_SEQUENCES: usize = 24;

/// Where the length of a fully published reference is recorded.
fn completion_marker_path(reference: &Path) -> PathBuf {
    let mut name = reference.as_os_str().to_os_string();
    name.push(".complete");
    PathBuf::from(name)
}

/// Records that `reference` was published whole, as its current length.
fn mark_reference_complete(reference: &Path) -> io::Result<()> {
    let len = fs::metadata(reference)?.len();
    crate::output::write_atomically(&completion_marker_path(reference), |writer| {
        writeln!(writer, "{len}")
    })
}

/// Whether a cached reference was published whole.
///
/// A reference this version downloads carries a marker holding its published
/// length. One cached by an older version has no marker. It is kept, and marked
/// so the scan runs once, if it names every human chromosome. The GRCh38
/// references that first-member-only decompression cut down to part of chr1 never do.
fn reference_is_complete(reference: &Path) -> io::Result<bool> {
    let len = fs::metadata(reference)?.len();
    match fs::read_to_string(completion_marker_path(reference)) {
        Ok(recorded) => return Ok(recorded.trim().parse::<u64>().ok() == Some(len)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(e),
    }
    if count_fasta_sequences(reference)? < MIN_REFERENCE_SEQUENCES {
        return Ok(false);
    }
    // A cache directory that cannot take the marker only costs another scan next time.
    let _ = mark_reference_complete(reference);
    Ok(true)
}

/// Counts FASTA sequence headers, the lines that begin with '>'.
fn count_fasta_sequences(path: &Path) -> io::Result<usize> {
    let mut reader = BufReader::with_capacity(1 << 20, File::open(path)?);
    let mut count = 0;
    let mut at_line_start = true;
    loop {
        let chunk = reader.fill_buf()?;
        if chunk.is_empty() {
            return Ok(count);
        }
        count += usize::from(at_line_start && chunk[0] == b'>');
        count += memchr::memmem::find_iter(chunk, b"\n>").count();
        at_line_start = chunk[chunk.len() - 1] == b'\n';
        let consumed = chunk.len();
        reader.consume(consumed);
    }
}

/// Computes the default cache directory path for a VCF/BCF/DTC file.
///
/// The cache directory is created alongside the input file as:
/// `{parent}/{stem}.gnomon_cache/`
fn get_cache_dir(input_path: &Path) -> PathBuf {
    let parent = input_path.parent().unwrap_or(Path::new("."));
    parent.join(format!("{}.gnomon_cache", cache_stem(input_path)))
}

/// The input's file name without its extension, and without `.vcf` for `.vcf.gz`.
fn cache_stem(input_path: &Path) -> String {
    input_path
        .file_stem()
        .map(|s| {
            // Handle .vcf.gz by stripping the .vcf part too
            let s_str = s.to_string_lossy();
            if s_str.ends_with(".vcf") {
                s_str[..s_str.len() - 4].to_string()
            } else {
                s_str.to_string()
            }
        })
        .unwrap_or_else(|| "converted".to_string())
}

/// Where the conversion cache for `input_path` lives: beside the input, or under
/// `cache_root` (the directory of `--out PREFIX`) when one is given. A shared root holds
/// caches for inputs from many directories, so there the directory name also carries a
/// key over the input's canonical path.
fn conversion_cache_dir(input_path: &Path, cache_root: Option<&Path>) -> PathBuf {
    let Some(root) = cache_root else {
        return get_cache_dir(input_path);
    };
    let canonical = fs::canonicalize(input_path).unwrap_or_else(|_| input_path.to_path_buf());
    let key = hex::encode(&Sha256::digest(canonical.as_os_str().as_encoded_bytes())[..8]);
    root.join(format!("{}.{key}.gnomon_cache", cache_stem(input_path)))
}

/// The prefix a default run names its results after when `input_path` needs conversion:
/// `{parent}/{stem}.gnomon_cache/genotypes`, where those results have always landed. It
/// is not the converted fileset, which sits in a generation directory below it. `None`
/// for PLINK inputs, which name results after their own prefix.
pub fn default_output_prefix(input_path: &Path) -> Option<PathBuf> {
    match detect_input_format(input_path)? {
        InputFormat::Plink => None,
        InputFormat::Vcf | InputFormat::Bcf | InputFormat::Dtc => {
            Some(get_cache_dir(input_path).join("genotypes"))
        }
    }
}

/// File recording the conversion parameters a cache was produced under.
const CACHE_PARAMS_FILE: &str = "conversion_params.txt";

/// Fingerprint of the conversion parameters that change the cached output.
///
/// The mtime check alone is unsafe: re-running the same input with a different
/// genome build, strand-harmonization panel, or reference produces different
/// genotypes, but the source file is untouched, so a stale cache would be
/// silently reused. Including these in a fingerprint forces re-conversion when
/// any of them changes. `build` is resolved through the same `GRCh38` default the
/// conversion uses, so `None` and an explicit `GRCh38` share a cache.
fn cache_params_fingerprint(
    build: Option<&str>,
    panel: Option<&Path>,
    reference: Option<&Path>,
) -> String {
    let assembly = build.unwrap_or("GRCh38");
    let panel = panel.map(|p| p.display().to_string()).unwrap_or_default();
    let reference = reference
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    format!("v1\nbuild={assembly}\npanel={panel}\nreference={reference}\n")
}

/// Staging directory names tried before giving up. A collision needs another
/// conversion of the same generation with the same pid and clock reading.
const STAGING_NAME_ATTEMPTS: u32 = 32;

/// Name prefix of a generation converted from a source whose metadata could not be read.
const UNIDENTIFIED_GENERATION_PREFIX: &str = "g-u";

/// The directory holding one conversion inside a cache directory. Its name is a key
/// over the conversion parameters and the source's size and modification time, so an
/// edited source, or another `--build`, `--panel` or `--reference`, gets a generation
/// of its own and no run reads a fileset converted from something else. A source whose
/// metadata cannot be read, such as a remote one, cannot be identified, so its
/// generation (`g-u…`, keyed by the parameters alone) is never served from the cache.
/// Every run converts it again and replaces it, as before generations existed.
fn generation_dir(cache_dir: &Path, source_path: &Path, fingerprint: &str) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update((fingerprint.len() as u64).to_le_bytes());
    hasher.update(fingerprint.as_bytes());
    let identity = fs::metadata(source_path).ok().and_then(|metadata| {
        let modified = metadata.modified().ok()?.duration_since(UNIX_EPOCH).ok()?;
        Some((metadata.len(), modified.as_nanos()))
    });
    let prefix = match identity {
        Some((len, modified_nanos)) => {
            hasher.update(len.to_le_bytes());
            hasher.update(modified_nanos.to_le_bytes());
            "g-"
        }
        None => UNIDENTIFIED_GENERATION_PREFIX,
    };
    cache_dir.join(format!("{prefix}{}", hex::encode(&hasher.finalize()[..8])))
}

/// Whether `generation` holds a complete conversion made under `fingerprint` that may be
/// served. A generation is published whole, by renaming its directory into place, so
/// its files are complete whenever it exists; the parameters are still compared as a
/// guard, and an unidentified source's generation is never served.
fn is_generation_valid(generation: &Path, fingerprint: &str) -> bool {
    let unidentified = generation.file_name().is_some_and(|name| {
        name.to_string_lossy()
            .starts_with(UNIDENTIFIED_GENERATION_PREFIX)
    });
    !unidentified
        && ["genotypes.bed", "genotypes.bim", "genotypes.fam"]
            .iter()
            .all(|name| generation.join(name).is_file())
        && fs::read_to_string(generation.join(CACHE_PARAMS_FILE))
            .is_ok_and(|found| found == fingerprint)
}

/// Converts into a private staging directory inside the cache directory, fsyncs every
/// file the converter wrote, and renames the directory into place as `generation`.
/// Readers therefore see no generation or a complete one, never files still being
/// written. A failed conversion removes its staging directory. When runs convert the
/// same source at once, the first rename wins and the others use that generation.
fn publish_generation<F>(
    generation: &Path,
    fingerprint: &str,
    convert: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: FnOnce(&Path) -> Result<(), Box<dyn Error + Send + Sync>>,
{
    let cache_dir = generation.parent().ok_or_else(|| {
        format!(
            "Conversion cache path '{}' has no parent directory.",
            generation.display()
        )
    })?;
    fs::create_dir_all(cache_dir)?;
    sweep_abandoned_staging(
        cache_dir,
        &host_key(),
        process_alive,
        STAGING_ABANDONED_AFTER,
    );
    let staging = create_staging_dir(cache_dir, generation)?;
    let staged = (|| -> Result<(), Box<dyn Error + Send + Sync>> {
        convert(&staging)?;
        fs::write(staging.join(CACHE_PARAMS_FILE), fingerprint)?;
        // The converter does not fsync. Make every file durable before the rename can
        // publish it, so a crash cannot leave a generation of truncated files.
        for entry in fs::read_dir(&staging)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                // Opened for writing: Windows cannot flush a read-only handle.
                OpenOptions::new()
                    .write(true)
                    .open(entry.path())?
                    .sync_all()?;
            }
        }
        Ok(())
    })();
    if let Err(err) = staged {
        let _ = fs::remove_dir_all(&staging);
        return Err(err);
    }
    if let Err(first_err) = crate::output::rename_replacing(&staging, generation) {
        if is_generation_valid(generation, fingerprint) {
            // Another run published this generation first.
            let _ = fs::remove_dir_all(&staging);
            return Ok(());
        }
        let replaced = if generation.is_dir() {
            // A published generation that may not be served, such as an unidentified
            // source's, is replaced: moved aside under a fresh private name, then
            // removed once the new one is in place. A run still reading it keeps its
            // open files on Unix.
            create_staging_dir(cache_dir, generation).and_then(|aside| {
                fs::remove_dir(&aside)?;
                crate::output::rename_replacing(generation, &aside)?;
                let published = crate::output::rename_replacing(&staging, generation);
                let _ = fs::remove_dir_all(&aside);
                published
            })
        } else {
            Err(first_err)
        };
        if let Err(err) = replaced {
            let _ = fs::remove_dir_all(&staging);
            return Err(format!(
                "Could not publish the PLINK conversion cache '{}': {err}",
                generation.display()
            )
            .into());
        }
    }
    Ok(())
}

/// Creates `.{generation}.{host}.{pid}.{nanos}.tmp` in `cache_dir` exclusively, so
/// concurrent conversions never share a staging directory. The host key and pid let a
/// later conversion tell an abandoned staging directory from a live one.
fn create_staging_dir(cache_dir: &Path, generation: &Path) -> io::Result<PathBuf> {
    let name = generation
        .file_name()
        .map_or_else(|| "generation".into(), |name| name.to_string_lossy());
    let host = host_key();
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    for attempt in 0..STAGING_NAME_ATTEMPTS {
        let candidate = cache_dir.join(format!(
            ".{name}.{host}.{pid}.{}.tmp",
            nanos + u128::from(attempt)
        ));
        match fs::create_dir(&candidate) {
            Ok(()) => return Ok(candidate),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "Failed to allocate a staging directory in '{}'.",
            cache_dir.display()
        ),
    ))
}

/// How long a staging directory whose owner cannot be checked may go without a change
/// before it counts as abandoned. A conversion keeps writing its files, so a day of
/// silence means its process is gone.
const STAGING_ABANDONED_AFTER: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);

/// A short key for this host in staging directory names: a pid means something only on
/// the host that issued it.
fn host_key() -> String {
    let name = sysinfo::System::host_name().unwrap_or_default();
    hex::encode(&Sha256::digest(name.as_bytes())[..4])
}

/// Whether a process with this pid is running on this host, or `None` when that cannot
/// be checked here. The check first has to find this process itself.
fn process_alive(pid: u32) -> Option<bool> {
    let mut system = sysinfo::System::new();
    if !system.refresh_process(sysinfo::Pid::from_u32(std::process::id())) {
        return None;
    }
    Some(system.refresh_process(sysinfo::Pid::from_u32(pid)))
}

/// The host key and pid in a staging directory name, `.{generation}.{host}.{pid}.{nanos}.tmp`
/// or the earlier `.{generation}.{pid}.{nanos}.tmp`, which carries no host. `None` for
/// anything else.
fn parse_staging_name(name: &str) -> Option<(Option<&str>, u32)> {
    let body = name.strip_prefix(".g-")?.strip_suffix(".tmp")?;
    let fields: Vec<&str> = body.split('.').collect();
    let (host, pid, nanos) = match fields.as_slice() {
        [_generation, host, pid, nanos] => (Some(*host), *pid, *nanos),
        [_generation, pid, nanos] => (None, *pid, *nanos),
        _ => return None,
    };
    nanos.parse::<u128>().ok()?;
    Some((host, pid.parse().ok()?))
}

/// The latest modification time of a directory and of the entries directly in it.
fn last_change(dir: &Path) -> Option<SystemTime> {
    let mut latest = fs::metadata(dir)
        .and_then(|metadata| metadata.modified())
        .ok()?;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if let Ok(modified) = entry.metadata().and_then(|metadata| metadata.modified()) {
                latest = latest.max(modified);
            }
        }
    }
    Some(latest)
}

/// Removes staging directories left in `cache_dir` by conversions that died before
/// publishing. A directory made on this host is abandoned exactly when its pid is not
/// running. One made on another host, or named before staging names carried a host, is
/// abandoned only after `abandoned_after` without a change, since its owner cannot be
/// checked from here. A directory whose owner may still be running is never touched.
/// Best effort.
fn sweep_abandoned_staging(
    cache_dir: &Path,
    this_host: &str,
    process_alive: impl Fn(u32) -> Option<bool>,
    abandoned_after: std::time::Duration,
) {
    let Ok(entries) = fs::read_dir(cache_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some((host, pid)) = parse_staging_name(&name) else {
            continue;
        };
        let path = entry.path();
        let alive = match host {
            Some(host) if host == this_host => process_alive(pid),
            _ => None,
        };
        let abandoned = match alive {
            Some(alive) => !alive,
            None => last_change(&path)
                .is_some_and(|changed| changed.elapsed().is_ok_and(|idle| idle >= abandoned_after)),
        };
        if abandoned {
            let _ = fs::remove_dir_all(&path);
        }
    }
}

/// Removes the generations in `cache_dir` made under `fingerprint`, other than
/// `current`. They were converted from an earlier version of the source, so they can
/// never be served again. Generations made under other parameters stay for the runs
/// that use them, and files from the previous cache layout are left alone. Best effort.
fn prune_superseded_generations(cache_dir: &Path, current: &Path, fingerprint: &str) {
    let Ok(entries) = fs::read_dir(cache_dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == current || !entry.file_name().to_string_lossy().starts_with("g-") {
            continue;
        }
        if fs::read_to_string(path.join(CACHE_PARAMS_FILE)).is_ok_and(|found| found == fingerprint)
        {
            let _ = fs::remove_dir_all(&path);
        }
    }
}

/// Options controlling how `ensure_plink_format_with_options` performs its work.
#[derive(Debug, Default, Clone, Copy)]
pub struct EnsurePlinkOptions {
    /// If true, skip the pre-conversion `infer_first_sample_sex` pass on VCF/BCF
    /// inputs. The FAM file will then be written with `Sex::Unknown`. Callers that
    /// run sex inference downstream (e.g. `gnomon all` invoking `terms`) can set
    /// this to avoid a redundant whole-VCF scan.
    pub skip_sex_inference: bool,
    /// Pre-computed sex to use instead of running `infer_first_sample_sex` on
    /// VCF/BCF inputs. When `Some`, the VCF-scan is skipped and this value is
    /// written into the FAM file directly. Takes precedence over
    /// `skip_sex_inference`. Intended for pipelines (e.g. pgsEngine gather)
    /// that already ran sex inference upstream on the pre-imputed VCF and
    /// would otherwise pay the ~4min full-VCF scan again.
    pub inferred_sex: Option<Sex>,
}

/// Ensures the input is in PLINK format, converting from VCF/BCF/DTC if necessary.
///
/// # Arguments
/// * `input_path` - Path to the input genotype file (PLINK prefix, VCF, BCF, or DTC text)
/// * `reference` - Optional path to reference genome FASTA (required for DTC, auto-downloaded if None)
/// * `build` - Optional genome build override (auto-detected or defaults to GRCh38)
/// * `panel` - Optional reference panel VCF for strand harmonization (flips alleles to match panel)
///
/// # Returns
/// * `Ok(PathBuf)` - Path to the PLINK prefix (either original or converted)
/// * `Err` - If conversion fails
pub fn ensure_plink_format(
    input_path: &Path,
    reference: Option<&Path>,
    build: Option<&str>,
    panel: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    ensure_plink_format_with_options(
        input_path,
        reference,
        build,
        panel,
        EnsurePlinkOptions::default(),
    )
}

/// Variant of [`ensure_plink_format`] that accepts additional runtime options.
///
/// Behaves identically when `options` is left at its default; in particular,
/// the cache directory, cache filenames, and PLINK prefix layout are unchanged,
/// so a conversion produced with options is interchangeable with one produced
/// by the plain `ensure_plink_format`.
pub fn ensure_plink_format_with_options(
    input_path: &Path,
    reference: Option<&Path>,
    build: Option<&str>,
    panel: Option<&Path>,
    options: EnsurePlinkOptions,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    ensure_plink_format_in(input_path, reference, build, panel, options, None)
}

/// Variant of [`ensure_plink_format_with_options`] that keeps the conversion cache
/// under `cache_root` (the directory of `--out PREFIX`) instead of beside the input.
///
/// Converted filesets are published whole. Each lives in a generation directory
/// keyed by the conversion parameters and the source's size and modification time,
/// and is written under a private staging name, then renamed into place once
/// complete. A reader, a concurrent run, or a run after a killed conversion
/// therefore never reads a partial or mixed fileset.
pub fn ensure_plink_format_in(
    input_path: &Path,
    reference: Option<&Path>,
    build: Option<&str>,
    panel: Option<&Path>,
    options: EnsurePlinkOptions,
    cache_root: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let format = detect_input_format(input_path).ok_or_else(|| {
        format!(
            "Could not determine input format for '{}'. \
             Expected PLINK (.bed/.bim/.fam), VCF (.vcf, .vcf.gz), BCF (.bcf), or DTC text (.txt).",
            input_path.display()
        )
    })?;

    match format {
        InputFormat::Plink => {
            // Already PLINK format, return the prefix
            let prefix = if input_path.extension().is_some_and(|ext| ext == "bed") {
                input_path.with_extension("")
            } else {
                input_path.to_path_buf()
            };
            Ok(prefix)
        }
        InputFormat::Vcf | InputFormat::Bcf => {
            // Check cache validity
            let cache_fingerprint = cache_params_fingerprint(build, panel, reference);
            let cache_dir = conversion_cache_dir(input_path, cache_root);
            let generation = generation_dir(&cache_dir, input_path, &cache_fingerprint);
            let cache_prefix = generation.join("genotypes");

            if is_generation_valid(&generation, &cache_fingerprint) {
                eprintln!(
                    "> Using cached PLINK conversion from '{}'",
                    generation.display()
                );
                return Ok(cache_prefix);
            }

            eprintln!("> Converting {} to PLINK format...", input_path.display());

            // Set up conversion config
            let input_format = match format {
                InputFormat::Vcf => ConvertInputFormat::Vcf,
                InputFormat::Bcf => ConvertInputFormat::Bcf,
                _ => unreachable!(),
            };

            let assembly = build.unwrap_or("GRCh38").to_string();
            let inferred_sex = if let Some(preset) = options.inferred_sex {
                eprintln!(
                    "> Using caller-provided sex {:?}; skipping pre-conversion sex inference pass.",
                    preset
                );
                preset
            } else if options.skip_sex_inference {
                eprintln!(
                    "> Skipping pre-conversion sex inference pass (sex will be written as Unknown; downstream `terms` pass will recompute)."
                );
                Sex::Unknown
            } else {
                let build_hint = parse_genome_build_hint(&assembly);
                match infer_first_sample_sex(input_path, build_hint) {
                    Ok(Some(sex)) => {
                        let convert_sex = to_convert_sex(sex);
                        eprintln!("> Inferred sample sex from input: {:?}", convert_sex);
                        convert_sex
                    }
                    Ok(None) => {
                        eprintln!(
                            "> Sex inference produced no sample calls; defaulting to Unknown"
                        );
                        Sex::Unknown
                    }
                    Err(err) => {
                        eprintln!("> Sex inference unavailable ({err}); defaulting to Unknown");
                        Sex::Unknown
                    }
                }
            };

            publish_generation(&generation, &cache_fingerprint, |staging| {
                // No reference needed for VCF/BCF - they have embedded reference info
                let config = ConversionConfig {
                    input: input_path.to_path_buf(),
                    input_format,
                    input_origin: input_path.display().to_string(),
                    reference_fasta: None,
                    reference_origin: None,
                    reference_fai: None,
                    reference_fai_origin: None,
                    output: staging.join("genotypes"),
                    output_dir: Some(staging.to_path_buf()),
                    output_format: OutputFormat::Plink,
                    sample_id: "sample".to_string(),
                    assembly,
                    input_build: build.map(|b| b.to_string()),
                    include_reference_sites: false,
                    sex: Some(inferred_sex),
                    par_boundaries: None,
                    standardize: false,
                    panel: panel.map(|p| p.to_path_buf()),
                    // Clinical-safety gates: reject a conversion that silently
                    // produces almost nothing, cannot confidently identify the
                    // source build, or fails to parse much of its input. Upstream
                    // owns these thresholds, so track its defaults rather than
                    // pinning our own copies.
                    min_emitted_variants: DEFAULT_MIN_EMITTED_VARIANTS,
                    min_build_confidence: DEFAULT_MIN_BUILD_CONFIDENCE,
                    max_parse_error_ratio: DEFAULT_MAX_PARSE_ERROR_RATIO,
                };

                // Run conversion
                convert_dtc_file(config)?;
                Ok(())
            })?;
            prune_superseded_generations(&cache_dir, &generation, &cache_fingerprint);

            eprintln!(
                "> Conversion complete. Cache stored at '{}'",
                generation.display()
            );

            Ok(cache_prefix)
        }
        InputFormat::Dtc => {
            // Check cache validity
            let cache_fingerprint = cache_params_fingerprint(build, panel, reference);
            let cache_dir = conversion_cache_dir(input_path, cache_root);
            let generation = generation_dir(&cache_dir, input_path, &cache_fingerprint);
            let cache_prefix = generation.join("genotypes");

            if is_generation_valid(&generation, &cache_fingerprint) {
                eprintln!(
                    "> Using cached PLINK conversion from '{}'",
                    generation.display()
                );
                return Ok(cache_prefix);
            }

            // Determine build (default to GRCh38 for modern DTC tests)
            let assembly = build.unwrap_or("GRCh38").to_string();

            eprintln!("> Detected format: DTC text file");
            eprintln!("> Using genome build: {}", assembly);

            // Get reference genome - use provided path or auto-download
            let reference_path = match reference {
                Some(path) => {
                    eprintln!("> Using provided reference: {}", path.display());
                    path.to_path_buf()
                }
                None => {
                    // Magic: auto-download reference genome
                    ensure_reference_genome(&assembly)?
                }
            };

            eprintln!("> Converting {} to PLINK format...", input_path.display());

            publish_generation(&generation, &cache_fingerprint, |staging| {
                let config = ConversionConfig {
                    input: input_path.to_path_buf(),
                    input_format: ConvertInputFormat::Dtc,
                    input_origin: input_path.display().to_string(),
                    reference_fasta: Some(reference_path.clone()),
                    reference_origin: Some(reference_path.display().to_string()),
                    reference_fai: None,
                    reference_fai_origin: None,
                    output: staging.join("genotypes"),
                    output_dir: Some(staging.to_path_buf()),
                    output_format: OutputFormat::Plink,
                    sample_id: input_path
                        .file_stem()
                        .map(|s| s.to_string_lossy().to_string())
                        .unwrap_or_else(|| "sample".to_string()),
                    assembly,
                    input_build: build.map(|b| b.to_string()),
                    include_reference_sites: false,
                    sex: None,
                    par_boundaries: None,
                    standardize: false,
                    panel: panel.map(|p| p.to_path_buf()),
                    // Clinical-safety gates: reject a conversion that silently
                    // produces almost nothing, cannot confidently identify the
                    // source build, or fails to parse much of its input. Upstream
                    // owns these thresholds, so track its defaults rather than
                    // pinning our own copies.
                    min_emitted_variants: DEFAULT_MIN_EMITTED_VARIANTS,
                    min_build_confidence: DEFAULT_MIN_BUILD_CONFIDENCE,
                    max_parse_error_ratio: DEFAULT_MAX_PARSE_ERROR_RATIO,
                };

                // Run conversion
                convert_dtc_file(config)?;
                Ok(())
            })?;
            prune_superseded_generations(&cache_dir, &generation, &cache_fingerprint);

            eprintln!(
                "> Conversion complete. Cache stored at '{}'",
                generation.display()
            );
            eprintln!(
                "> Note: Using raw genotyped data. Missing variants will be mean-imputed during scoring."
            );

            Ok(cache_prefix)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn decompress_gz_reads_every_bgzf_member_and_refuses_a_cut_file() {
        let dir = tempfile::tempdir().unwrap();
        // Several 64 KiB BGZF blocks, as in Ensembl's indexed references.
        let text: String = (0..3000)
            .map(|i| format!(">seq{i}\n{}\n", "ACGT".repeat(15)))
            .collect();
        let mut compressed = Vec::new();
        {
            let mut writer = noodles_bgzf::io::Writer::new(&mut compressed);
            writer.write_all(text.as_bytes()).unwrap();
            // Dropping the writer flushes the last block and the end-of-file block.
        }
        assert_eq!(
            &compressed[compressed.len() - BGZF_EOF_BLOCK.len()..],
            &BGZF_EOF_BLOCK[..]
        );
        let src = dir.path().join("reference.fa.gz");
        fs::write(&src, &compressed).unwrap();
        let whole = dir.path().join("whole.fa");
        decompress_gz(&src, &whole).unwrap();
        assert_eq!(fs::read_to_string(&whole).unwrap(), text);

        // Without its end-of-file block, every remaining member still decodes cleanly.
        fs::write(&src, &compressed[..compressed.len() - BGZF_EOF_BLOCK.len()]).unwrap();
        let cut = dir.path().join("cut.fa");
        assert!(decompress_gz(&src, &cut).is_err());
        assert!(!cut.exists(), "an incomplete reference is never published");
    }

    #[test]
    fn cached_reference_is_trusted_only_when_whole() {
        let dir = tempfile::tempdir().unwrap();
        let reference = dir.path().join("GRCh38_reference.fa");

        // What first-member-only decompression left: part of one chromosome.
        fs::write(
            &reference,
            format!(">1 dna:chromosome\n{}\n", "N".repeat(600)),
        )
        .unwrap();
        assert!(!reference_is_complete(&reference).unwrap());

        let genome: String = (1..=22)
            .map(|c| c.to_string())
            .chain(["X", "Y", "MT"].map(String::from))
            .map(|name| format!(">{name}\nACGTACGT\n"))
            .collect();
        fs::write(&reference, &genome).unwrap();
        assert!(reference_is_complete(&reference).unwrap());
        assert!(
            completion_marker_path(&reference).exists(),
            "an older cache is scanned once, then marked"
        );

        // A marker describes the bytes it was written for.
        fs::write(&reference, format!("{genome}>extra\nAC\n")).unwrap();
        assert!(!reference_is_complete(&reference).unwrap());
    }

    #[test]
    fn test_detect_plink_bed() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.bed")),
            Some(InputFormat::Plink)
        );
    }

    #[test]
    fn test_detect_remote_plink_directory() {
        assert_eq!(
            detect_input_format(Path::new("gs://bucket/callset/pgen/")),
            Some(InputFormat::Plink)
        );
        assert_eq!(
            detect_input_format(Path::new("gs://bucket/callset/pgen/*")),
            Some(InputFormat::Plink)
        );
    }

    #[test]
    fn test_detect_vcf() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.vcf")),
            Some(InputFormat::Vcf)
        );
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.vcf.gz")),
            Some(InputFormat::Vcf)
        );
    }

    #[test]
    fn test_detect_bcf() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/data.bcf")),
            Some(InputFormat::Bcf)
        );
    }

    #[test]
    fn test_detect_dtc_txt() {
        assert_eq!(
            detect_input_format(Path::new("/path/to/23andme_data.txt")),
            Some(InputFormat::Dtc)
        );
        assert_eq!(
            detect_input_format(Path::new("/path/to/ancestry_data.csv")),
            Some(InputFormat::Dtc)
        );
    }

    #[test]
    fn test_cache_dir_vcf() {
        let cache = get_cache_dir(Path::new("/data/sample.vcf"));
        assert_eq!(cache, Path::new("/data/sample.gnomon_cache"));
    }

    #[test]
    fn test_cache_dir_vcf_gz() {
        let cache = get_cache_dir(Path::new("/data/sample.vcf.gz"));
        assert_eq!(cache, Path::new("/data/sample.gnomon_cache"));
    }

    #[test]
    fn test_cache_dir_dtc() {
        let cache = get_cache_dir(Path::new("/data/23andme_raw.txt"));
        assert_eq!(cache, Path::new("/data/23andme_raw.gnomon_cache"));
    }

    fn fake_conversion(staging: &Path) -> Result<(), Box<dyn Error + Send + Sync>> {
        for name in ["genotypes.bed", "genotypes.bim", "genotypes.fam"] {
            fs::write(staging.join(name), name)?;
        }
        Ok(())
    }

    fn entry_names(dir: &Path) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(dir)
            .expect("read_dir")
            .map(|entry| {
                entry
                    .expect("directory entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        names
    }

    #[test]
    fn a_published_generation_is_complete_and_leaves_no_staging_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("sample.bcf");
        fs::write(&source, b"version one").expect("source");
        let cache_dir = conversion_cache_dir(&source, None);
        assert_eq!(cache_dir, dir.path().join("sample.gnomon_cache"));
        let generation = generation_dir(&cache_dir, &source, "v1\n");

        publish_generation(&generation, "v1\n", fake_conversion).expect("publish");

        assert!(is_generation_valid(&generation, "v1\n"));
        assert!(!is_generation_valid(&generation, "v2\n"));
        let name = generation
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        assert_eq!(entry_names(&cache_dir), [name]);
    }

    #[test]
    fn a_failed_conversion_publishes_nothing() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let generation = cache_dir.join("g-0");
        let err = publish_generation(&generation, "v1\n", |staging| {
            fs::write(staging.join("genotypes.bed"), b"partial")?;
            Err("converter failed".into())
        })
        .expect_err("the conversion error must propagate");
        assert_eq!(err.to_string(), "converter failed");
        assert!(!generation.exists());
        assert!(entry_names(&cache_dir).is_empty());
    }

    #[test]
    fn concurrent_conversions_of_one_source_all_succeed_with_one_generation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let generation = cache_dir.join("g-0");
        std::thread::scope(|scope| {
            let runs: Vec<_> = (0..8)
                .map(|_| {
                    scope.spawn(|| {
                        publish_generation(&generation, "v1\n", fake_conversion)
                            .map_err(|err| err.to_string())
                    })
                })
                .collect();
            for run in runs {
                run.join().expect("thread").expect("every run must succeed");
            }
        });
        assert!(is_generation_valid(&generation, "v1\n"));
        assert_eq!(entry_names(&cache_dir), ["g-0"]);
    }

    #[test]
    fn generation_keys_follow_the_parameters_and_the_source() {
        let dir = tempfile::tempdir().expect("tempdir");
        let source = dir.path().join("sample.bcf");
        fs::write(&source, b"version one").expect("source");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let first = generation_dir(&cache_dir, &source, "v1\nbuild=GRCh38\n");
        assert_eq!(
            first,
            generation_dir(&cache_dir, &source, "v1\nbuild=GRCh38\n")
        );
        assert_ne!(
            first,
            generation_dir(&cache_dir, &source, "v1\nbuild=GRCh37\n")
        );
        fs::write(&source, b"version two, longer").expect("edit the source");
        assert_ne!(
            first,
            generation_dir(&cache_dir, &source, "v1\nbuild=GRCh38\n")
        );
    }

    #[test]
    fn an_unidentified_source_is_never_served_and_is_replaced() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("remote.gnomon_cache");
        let missing = dir.path().join("absent.bcf");
        let generation = generation_dir(&cache_dir, &missing, "v1\n");
        assert_eq!(generation, generation_dir(&cache_dir, &missing, "v1\n"));
        for version in ["first", "second"] {
            publish_generation(&generation, "v1\n", |staging| {
                for name in ["genotypes.bed", "genotypes.bim", "genotypes.fam"] {
                    fs::write(staging.join(name), version)?;
                }
                Ok(())
            })
            .expect("publish");
            assert!(!is_generation_valid(&generation, "v1\n"));
            assert_eq!(
                fs::read_to_string(generation.join("genotypes.bed")).expect("read"),
                version
            );
        }
        assert_eq!(
            entry_names(&cache_dir).len(),
            1,
            "{:?}",
            entry_names(&cache_dir)
        );
    }

    #[test]
    fn pruning_removes_only_superseded_generations_of_the_same_parameters() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        for (name, fingerprint) in [("g-old", "v1\n"), ("g-new", "v1\n"), ("g-other", "v2\n")] {
            publish_generation(&cache_dir.join(name), fingerprint, fake_conversion)
                .expect("publish");
        }
        // The previous layout kept its files directly in the cache directory.
        fs::write(cache_dir.join("genotypes.bed"), b"legacy").expect("legacy");

        prune_superseded_generations(&cache_dir, &cache_dir.join("g-new"), "v1\n");

        assert_eq!(
            entry_names(&cache_dir),
            ["g-new", "g-other", "genotypes.bed"]
        );
    }

    #[test]
    fn a_shared_cache_root_keys_conversion_caches_by_input_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path().join("results").join("gnomon_score_cache");
        let a = dir.path().join("a").join("sample.vcf.gz");
        let b = dir.path().join("b").join("sample.vcf.gz");
        for input in [&a, &b] {
            fs::create_dir_all(input.parent().expect("parent")).expect("mkdir");
            fs::write(input, b"source").expect("source");
        }
        let cache_a = conversion_cache_dir(&a, Some(&root));
        let cache_b = conversion_cache_dir(&b, Some(&root));
        assert_ne!(cache_a, cache_b);
        for cache in [&cache_a, &cache_b] {
            assert_eq!(cache.parent(), Some(root.as_path()));
            let name = cache
                .file_name()
                .expect("name")
                .to_string_lossy()
                .into_owned();
            assert!(
                name.starts_with("sample.") && name.ends_with(".gnomon_cache"),
                "{name}"
            );
        }
        assert_eq!(
            conversion_cache_dir(&a, None),
            dir.path().join("a").join("sample.gnomon_cache")
        );
    }

    #[test]
    fn converted_inputs_name_default_results_after_their_cache_directory() {
        assert_eq!(
            default_output_prefix(Path::new("/data/sample.bcf")),
            Some(PathBuf::from("/data/sample.gnomon_cache/genotypes"))
        );
        assert_eq!(default_output_prefix(Path::new("/data/arrays.bed")), None);
    }

    fn staging_dir(cache_dir: &Path, name: &str) -> PathBuf {
        let dir = cache_dir.join(name);
        fs::create_dir_all(&dir).expect("mkdir staging");
        fs::write(dir.join("genotypes.bed"), b"partial").expect("partial file");
        dir
    }

    #[test]
    fn staging_names_carry_this_host_and_process() {
        let dir = tempfile::tempdir().expect("tempdir");
        let staging = create_staging_dir(dir.path(), &dir.path().join("g-0123456789abcdef"))
            .expect("staging directory");
        let name = staging
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        let host = host_key();
        assert_eq!(
            parse_staging_name(&name),
            Some((Some(host.as_str()), std::process::id()))
        );
    }

    #[test]
    fn legacy_staging_names_parse_without_a_host_and_others_not_at_all() {
        assert_eq!(
            parse_staging_name(".g-0123456789abcdef.4242.1789355857001456691.tmp"),
            Some((None, 4242))
        );
        assert_eq!(parse_staging_name("g-0123456789abcdef"), None);
        assert_eq!(parse_staging_name(".other.0a1b2c3d.4242.1.tmp"), None);
        assert_eq!(parse_staging_name(".g-0123.0a1b2c3d.notapid.1.tmp"), None);
    }

    #[test]
    fn sweeping_removes_only_abandoned_staging_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let this_host = "0a1b2c3d";
        let dead_here = staging_dir(&cache_dir, &format!(".g-aaaa.{this_host}.101.1.tmp"));
        let live_here = staging_dir(&cache_dir, &format!(".g-bbbb.{this_host}.202.2.tmp"));
        let elsewhere = staging_dir(&cache_dir, ".g-cccc.ffffffff.303.3.tmp");
        let legacy = staging_dir(&cache_dir, ".g-dddd.404.4.tmp");
        let generation = staging_dir(&cache_dir, "g-eeee");
        fs::write(cache_dir.join("genotypes.bed"), b"legacy layout").expect("legacy file");
        let alive = |pid: u32| Some(pid == 202);

        // Under a long idle limit only the same-host directory with a dead owner goes.
        sweep_abandoned_staging(
            &cache_dir,
            this_host,
            alive,
            std::time::Duration::from_secs(u64::MAX / 4),
        );
        assert!(!dead_here.exists());
        assert!(live_here.exists() && elsewhere.exists() && legacy.exists());

        // Once idle long enough, directories whose owner cannot be checked go too, while
        // a running owner keeps its directory.
        sweep_abandoned_staging(&cache_dir, this_host, alive, std::time::Duration::ZERO);
        assert!(live_here.exists());
        assert!(!elsewhere.exists() && !legacy.exists());
        assert!(generation.is_dir(), "a published generation was removed");
        assert!(
            cache_dir.join("genotypes.bed").is_file(),
            "a legacy file was removed"
        );
    }

    #[test]
    fn unknown_liveness_falls_back_to_idle_time() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache_dir = dir.path().join("sample.gnomon_cache");
        let this_host = "0a1b2c3d";
        let staging = staging_dir(&cache_dir, &format!(".g-aaaa.{this_host}.101.1.tmp"));
        let unknown = |_: u32| None;
        sweep_abandoned_staging(
            &cache_dir,
            this_host,
            unknown,
            std::time::Duration::from_secs(u64::MAX / 4),
        );
        assert!(
            staging.exists(),
            "an owner that cannot be checked counted as dead"
        );
        sweep_abandoned_staging(&cache_dir, this_host, unknown, std::time::Duration::ZERO);
        assert!(!staging.exists());
    }
}
