use crate::score::types::PreparationResult;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAGIC: &[u8] = b"GNOMON_SCORE_CHECKPOINT_V1\n";
const CHECKPOINT_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Clone, Debug)]
pub struct ScoreCheckpoint {
    pub completed_variants: usize,
    pub sum_scores: Vec<f64>,
    pub missing_counts: Vec<u32>,
}

pub struct ScoreCheckpointWriter {
    path: PathBuf,
    fingerprint: [u8; 32],
    total_variants: usize,
    last_save_at: Option<Instant>,
    last_completed_variants: usize,
}

impl ScoreCheckpointWriter {
    pub fn new(path: PathBuf, fingerprint: [u8; 32], total_variants: usize) -> Self {
        Self {
            path,
            fingerprint,
            total_variants,
            last_save_at: None,
            last_completed_variants: 0,
        }
    }

    pub fn maybe_save(
        &mut self,
        completed_variants: usize,
        sum_scores: &[f64],
        missing_counts: &[u32],
        force: bool,
    ) -> io::Result<bool> {
        if completed_variants <= self.last_completed_variants {
            return Ok(false);
        }
        let due = force
            || completed_variants >= self.total_variants
            || self
                .last_save_at
                .map(|last| last.elapsed() >= CHECKPOINT_INTERVAL)
                .unwrap_or(true);
        if !due {
            return Ok(false);
        }
        write_checkpoint(
            &self.path,
            self.fingerprint,
            completed_variants,
            sum_scores,
            missing_counts,
        )?;
        self.last_save_at = Some(Instant::now());
        self.last_completed_variants = completed_variants;
        Ok(true)
    }
}

pub fn checkpoint_path_for_output(output_path: &Path) -> PathBuf {
    let filename = output_path
        .file_name()
        .map(|name| format!("{}.gnomon-checkpoint.bin", name.to_string_lossy()))
        .unwrap_or_else(|| "gnomon_results.sscore.gnomon-checkpoint.bin".to_string());
    output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .join(filename)
}

pub fn fingerprint_preparation(prep: &PreparationResult) -> [u8; 32] {
    let mut hasher = Sha256::new();
    update_usize(&mut hasher, prep.num_people_to_score);
    update_usize(&mut hasher, prep.total_people_in_fam);
    update_u64(&mut hasher, prep.total_variants_in_bim);
    update_usize(&mut hasher, prep.num_reconciled_variants);
    update_u64(&mut hasher, prep.bytes_per_variant);
    update_usize(&mut hasher, prep.score_names.len());
    for name in &prep.score_names {
        update_bytes(&mut hasher, name.as_bytes());
    }
    for iid in &prep.final_person_iids {
        update_bytes(&mut hasher, iid.as_bytes());
    }
    for &count in &prep.score_variant_counts {
        hasher.update(count.to_le_bytes());
    }
    for &value in prep.baseline_missing_sum_by_score() {
        hasher.update(value.to_le_bytes());
    }
    for &idx in &prep.required_bim_indices {
        update_u64(&mut hasher, idx.0);
    }
    for &flag in prep.required_is_complex() {
        hasher.update([flag]);
    }
    for &offset in prep.sparse_row_offsets() {
        update_u64(&mut hasher, offset);
    }
    for &column in prep.sparse_score_columns() {
        hasher.update(column.to_le_bytes());
    }
    for &weight in prep.sparse_weights() {
        hasher.update(weight.to_le_bytes());
    }
    for &correction in prep.sparse_missing_corrections() {
        hasher.update(correction.to_le_bytes());
    }
    hasher.finalize().into()
}

pub fn load_checkpoint(
    path: &Path,
    expected_fingerprint: [u8; 32],
    expected_result_len: usize,
    total_variants: usize,
) -> io::Result<Option<ScoreCheckpoint>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let mut reader = BufReader::new(file);
    let mut magic = vec![0u8; MAGIC.len()];
    reader.read_exact(&mut magic)?;
    if magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Checkpoint '{}' has an invalid header.", path.display()),
        ));
    }

    let mut actual_fingerprint = [0u8; 32];
    reader.read_exact(&mut actual_fingerprint)?;
    if actual_fingerprint != expected_fingerprint {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Checkpoint '{}' does not match the prepared scoring inputs.",
                path.display()
            ),
        ));
    }

    let completed_variants = read_u64(&mut reader).and_then(usize_from_u64)?;
    let result_len = read_u64(&mut reader).and_then(usize_from_u64)?;
    if result_len != expected_result_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Checkpoint '{}' result length {} does not match current result length {}.",
                path.display(),
                result_len,
                expected_result_len
            ),
        ));
    }
    if completed_variants > total_variants {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Checkpoint '{}' completed {} variants but this run has only {}.",
                path.display(),
                completed_variants,
                total_variants
            ),
        ));
    }

    let mut sum_scores = Vec::with_capacity(result_len);
    for _ in 0..result_len {
        sum_scores.push(f64::from_le_bytes(read_array::<8>(&mut reader)?));
    }
    let mut missing_counts = Vec::with_capacity(result_len);
    for _ in 0..result_len {
        missing_counts.push(u32::from_le_bytes(read_array::<4>(&mut reader)?));
    }
    Ok(Some(ScoreCheckpoint {
        completed_variants,
        sum_scores,
        missing_counts,
    }))
}

pub fn remove_checkpoint(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

fn write_checkpoint(
    path: &Path,
    fingerprint: [u8; 32],
    completed_variants: usize,
    sum_scores: &[f64],
    missing_counts: &[u32],
) -> io::Result<()> {
    if sum_scores.len() != missing_counts.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Checkpoint score/count length mismatch.",
        ));
    }
    let output_dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(output_dir)?;
    let output_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Checkpoint path '{}' has no file name.", path.display()),
        )
    })?;
    let temp_path = unique_temp_path(output_dir, output_name);
    let temp_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp_path)?;
    let mut writer = BufWriter::new(temp_file);
    writer.write_all(MAGIC)?;
    writer.write_all(&fingerprint)?;
    writer.write_all(&(completed_variants as u64).to_le_bytes())?;
    writer.write_all(&(sum_scores.len() as u64).to_le_bytes())?;
    for &value in sum_scores {
        writer.write_all(&value.to_le_bytes())?;
    }
    for &value in missing_counts {
        writer.write_all(&value.to_le_bytes())?;
    }
    writer.flush()?;
    let file = writer.into_inner().map_err(io::Error::other)?;
    file.sync_all()?;
    fs::rename(&temp_path, path).inspect_err(|_| {
        let _ = fs::remove_file(&temp_path);
    })
}

fn unique_temp_path(output_dir: &Path, output_name: &std::ffi::OsStr) -> PathBuf {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    output_dir.join(format!(
        ".{}.{}.{}.tmp",
        output_name.to_string_lossy(),
        pid,
        nanos
    ))
}

fn update_usize(hasher: &mut Sha256, value: usize) {
    update_u64(hasher, value as u64);
}

fn update_u64(hasher: &mut Sha256, value: u64) {
    hasher.update(value.to_le_bytes());
}

fn update_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    update_u64(hasher, bytes.len() as u64);
    hasher.update(bytes);
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_array::<8>(reader)?))
}

fn read_array<const N: usize>(reader: &mut impl Read) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

fn usize_from_u64(value: u64) -> io::Result<usize> {
    usize::try_from(value).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Checkpoint value {value} does not fit usize."),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_roundtrip_preserves_accumulators_and_rejects_wrong_fingerprint() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir
            .path()
            .join("cohort_scores.sscore.gnomon-checkpoint.bin");
        let fingerprint = [7u8; 32];
        let scores = vec![1.25, -2.5, 3.75];
        let counts = vec![0, 4, 9];
        write_checkpoint(&path, fingerprint, 42, &scores, &counts).expect("write checkpoint");

        let loaded = load_checkpoint(&path, fingerprint, scores.len(), 100)
            .expect("load checkpoint")
            .expect("checkpoint exists");
        assert_eq!(loaded.completed_variants, 42);
        assert_eq!(loaded.sum_scores, scores);
        assert_eq!(loaded.missing_counts, counts);

        let err = load_checkpoint(&path, [8u8; 32], scores.len(), 100)
            .expect_err("wrong fingerprint must fail");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}
