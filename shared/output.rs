//! Publishing output files that other processes may already be reading.
//!
//! gnomon's results, checkpoints and caches are read by whatever runs next: a
//! pipeline step waiting for the file, a second gnomon run on the same inputs,
//! or a rerun resuming from a checkpoint. A file written in place is visible
//! while it is still being written, and a run that is killed or runs out of disk
//! leaves a truncated file behind that looks exactly like a finished one.
//! [`write_atomically`] exposes neither: the content goes to a temporary file in
//! the destination directory, is flushed and fsynced, and only then is renamed
//! over the destination, so a reader sees the previous file or the complete new
//! one.

use std::ffi::{OsStr, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Temporary names tried before giving up. A collision needs another writer of
/// the same destination with the same pid and clock reading.
const TEMP_NAME_ATTEMPTS: u32 = 32;

/// Writes `dest` through `write` without ever exposing a partial file.
///
/// `write` receives a buffered writer and does not need to flush it. The
/// destination directory is created if needed, and an existing `dest` is
/// replaced. On any error, including one returned by `write`, the temporary
/// file is removed and `dest` is left exactly as it was.
pub fn write_atomically<F>(dest: &Path, write: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let dir = dest
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = dest.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Output path '{}' has no file name.", dest.display()),
        )
    })?;
    fs::create_dir_all(dir)?;
    let (temp_path, temp_file) = create_temp_file(dir, name)?;

    let published = (|| -> io::Result<()> {
        let mut writer = BufWriter::with_capacity(1 << 20, temp_file);
        write(&mut writer)?;
        let file = writer.into_inner().map_err(io::IntoInnerError::into_error)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temp_path, dest)
    })();
    if let Err(err) = published {
        let _ = fs::remove_file(&temp_path);
        return Err(err);
    }
    // The directory is deliberately not fsynced. A crash can at worst lose the
    // rename, which leaves the previous file or none, never a partial one, while
    // on network filesystems that fsync costs milliseconds per output even for
    // a table of a few bytes.
    Ok(())
}

/// Creates `.{name}.{pid}.{nanos}.tmp` in `dir` exclusively, so concurrent
/// writers of one destination never share a temporary file.
fn create_temp_file(dir: &Path, name: &OsStr) -> io::Result<(PathBuf, File)> {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    for attempt in 0..TEMP_NAME_ATTEMPTS {
        let mut temp_name = OsString::from(".");
        temp_name.push(name);
        temp_name.push(format!(".{pid}.{}.tmp", nanos + u128::from(attempt)));
        let candidate = dir.join(temp_name);
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "Failed to allocate a unique temporary output file in '{}'.",
            dir.display()
        ),
    ))
}

/// `PREFIX.suffix`, the file an `--out PREFIX` run writes for one artifact.
///
/// The suffix is appended, never substituted for an extension, so
/// `results/eur.chr22` names `results/eur.chr22.sscore`, the same way
/// `gnomon fit --out` names its artifacts.
pub fn prefixed_path(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = OsString::from(prefix.as_os_str());
    path.push(".");
    path.push(suffix);
    PathBuf::from(path)
}

/// Rejects an `--out PREFIX` that cannot name local files: a remote URI, which
/// std file I/O cannot write, or a directory such as `results/`, whose
/// artifacts would be hidden files like `results/.sscore`.
pub fn validate_out_prefix(prefix: &Path) -> io::Result<()> {
    let raw = prefix.as_os_str().to_string_lossy();
    let reject = |reason: &str| -> io::Result<()> {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "--out '{}' {reason}; pass a local file prefix such as results/cohort.",
                prefix.display()
            ),
        ))
    };
    if raw.starts_with("gs://") || raw.starts_with("http://") || raw.starts_with("https://") {
        return reject("is a remote location, but outputs are written locally");
    }
    if raw.is_empty() || raw.ends_with(std::path::is_separator) || prefix.file_name().is_none() {
        return reject("names a directory, not a file prefix");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{prefixed_path, validate_out_prefix, write_atomically};
    use std::fs;
    use std::io::{self, Write};
    use std::path::Path;

    #[test]
    fn prefixed_paths_append_rather_than_replace_an_extension() {
        assert_eq!(
            prefixed_path(Path::new("results/eur.chr22"), "sscore"),
            Path::new("results/eur.chr22.sscore")
        );
        assert_eq!(
            prefixed_path(Path::new("cohort"), "sex.tsv"),
            Path::new("cohort.sex.tsv")
        );
    }

    #[test]
    fn out_prefixes_must_name_local_files() {
        for accepted in ["results/eur", "eur", "../eur.chr22", "/abs/dir/eur"] {
            validate_out_prefix(Path::new(accepted)).expect(accepted);
        }
        for rejected in ["", "results/", ".", "..", "/", "gs://bucket/eur", "https://host/eur"] {
            let err = validate_out_prefix(Path::new(rejected)).expect_err(rejected);
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput, "{rejected}");
        }
    }

    fn entries(dir: &Path) -> Vec<String> {
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
    fn publishes_the_content_and_removes_the_temporary_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let results = dir.path().join("results");
        let dest = results.join("cohort_w.sscore");
        write_atomically(&dest, |writer| writer.write_all(b"#IID\tw_AVG\n")).expect("write");
        assert_eq!(fs::read(&dest).expect("read"), b"#IID\tw_AVG\n");
        assert_eq!(entries(&results), ["cohort_w.sscore"]);
    }

    #[test]
    fn a_failed_write_keeps_the_previous_file_and_removes_the_temporary() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("cohort.sex.tsv");
        fs::write(&dest, b"previous\n").expect("seed the destination");
        let err = write_atomically(&dest, |writer| {
            // More than the writer's buffer, so bytes reach the temporary file.
            writer.write_all(&vec![b'x'; 3 << 20])?;
            Err(io::Error::new(
                io::ErrorKind::StorageFull,
                "no space left on device",
            ))
        })
        .expect_err("the write error must propagate");
        assert_eq!(err.kind(), io::ErrorKind::StorageFull);
        assert_eq!(fs::read(&dest).expect("read"), b"previous\n");
        assert_eq!(entries(dir.path()), ["cohort.sex.tsv"]);
    }

    #[test]
    fn a_failed_first_write_creates_no_destination() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("cohort_w.sscore");
        write_atomically(&dest, |writer| {
            writer.write_all(b"#IID")?;
            Err(io::Error::other("interrupted"))
        })
        .expect_err("the write error must propagate");
        assert!(entries(dir.path()).is_empty());
    }

    /// A reader polling a destination while it is republished sees one complete
    /// version or another: never an empty, truncated or mixed file. Writing in
    /// place (truncate, then write) fails this within a few versions.
    #[cfg(unix)]
    #[test]
    fn a_concurrent_reader_never_observes_a_partial_file() {
        use std::sync::atomic::{AtomicBool, Ordering};

        const LEN: usize = 1 << 20;
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("cohort_w.sscore");
        write_atomically(&dest, |writer| writer.write_all(&vec![0u8; LEN])).expect("seed");

        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let mut reads = 0u64;
                loop {
                    let bytes = fs::read(&dest).expect("the destination must always exist");
                    assert_eq!(bytes.len(), LEN, "observed a truncated file");
                    assert!(
                        bytes.iter().all(|&b| b == bytes[0]),
                        "observed a mixed file"
                    );
                    reads += 1;
                    if done.load(Ordering::Relaxed) {
                        return reads;
                    }
                }
            });
            for version in 1..=40u8 {
                write_atomically(&dest, |writer| {
                    for _ in 0..LEN / 4096 {
                        writer.write_all(&[version; 4096])?;
                    }
                    Ok(())
                })
                .expect("republish");
            }
            done.store(true, Ordering::Relaxed);
            assert!(reader.join().expect("reader thread") > 0);
        });
        // On NFS, a version renamed over while the reader still held it open
        // survives as a `.nfsXXXX` entry until the client releases it, so only
        // the helper's own temporary files are required to be gone.
        let leftovers: Vec<String> = entries(dir.path())
            .into_iter()
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temporary files left behind: {leftovers:?}");
        assert!(dir.path().join("cohort_w.sscore").is_file());
    }
}
