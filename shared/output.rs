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
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    publish(dest, write, true)
}

/// Like [`write_atomically`], but without the fsync. The rename still shows readers
/// only complete files, but after a crash the published file may hold torn content.
/// Use it only for a cache that verifies its own content when read and rebuilds on a
/// mismatch; never for results.
pub fn write_atomically_unsynced<F>(dest: &Path, write: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    publish(dest, write, false)
}

fn publish<F>(dest: &Path, write: F, synced: bool) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let mut file = AtomicFile::create(dest)?;
    // On an error the file is dropped uncommitted, which removes the temporary file.
    write(file.writer())?;
    file.finish(synced)
}

/// A file being published, for writers that do not fit in one closure, such as a
/// stream of score blocks. Writes go to a temporary file in the destination
/// directory, and [`AtomicFile::commit`] flushes and fsyncs it, then renames it over
/// the destination. Dropped without committing, it removes the temporary file, so a
/// failed or abandoned publication leaves the destination as it was.
pub struct AtomicFile {
    dest: PathBuf,
    temp_path: PathBuf,
    writer: Option<BufWriter<File>>,
}

impl AtomicFile {
    /// Creates the temporary file for `dest`, and `dest`'s directory if needed.
    pub fn create(dest: &Path) -> io::Result<Self> {
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
        // Create the temporary file first, and the directory only when that fails:
        // create_dir_all always attempts a mkdir, which on a network filesystem is a
        // server round trip for every output even when the directory already exists.
        let (temp_path, temp_file) = match create_temp_file(dir, name) {
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                fs::create_dir_all(dir)?;
                create_temp_file(dir, name)?
            }
            created => created?,
        };
        Ok(Self {
            dest: dest.to_path_buf(),
            temp_path,
            writer: Some(BufWriter::with_capacity(1 << 20, temp_file)),
        })
    }

    /// The buffered writer over the temporary file. It needs no flushing.
    pub fn writer(&mut self) -> &mut BufWriter<File> {
        self.writer
            .as_mut()
            .expect("an uncommitted publication always has its writer")
    }

    /// Publishes the file: flushes and fsyncs it, then renames it over the destination.
    pub fn commit(mut self) -> io::Result<()> {
        self.finish(true)
    }

    fn finish(&mut self, synced: bool) -> io::Result<()> {
        let writer = self
            .writer
            .take()
            .expect("an uncommitted publication always has its writer");
        let published = (|| -> io::Result<()> {
            let file = writer
                .into_inner()
                .map_err(io::IntoInnerError::into_error)?;
            if synced {
                file.sync_all()?;
            }
            drop(file);
            rename_replacing(&self.temp_path, &self.dest)
        })();
        if published.is_err() {
            let _ = fs::remove_file(&self.temp_path);
        }
        // The directory is deliberately not fsynced. A crash can at worst lose the
        // rename, which leaves the previous file or none, never a partial one, while
        // on network filesystems that fsync costs milliseconds per output even for
        // a table of a few bytes.
        published
    }
}

impl Drop for AtomicFile {
    fn drop(&mut self) {
        if let Some(writer) = self.writer.take() {
            // Discard the buffer rather than flush it into a file that is about to go.
            let (file, _) = writer.into_parts();
            drop(file);
            let _ = fs::remove_file(&self.temp_path);
        }
    }
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

/// Fails fast, before any heavy work, when `path` could not be published: its
/// directory, or the nearest existing ancestor it would be created in, does not
/// let this process create files. Nothing is created. The error names `path` and
/// suggests `--out`, because the usual cause is a default output beside
/// read-only inputs.
pub fn ensure_output_writable(path: &Path) -> io::Result<()> {
    let mut dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    while !dir.exists() {
        match dir.parent().filter(|p| !p.as_os_str().is_empty()) {
            Some(parent) => dir = parent,
            None => {
                dir = Path::new(".");
                break;
            }
        }
    }
    let probe = dir.join(format!(".gnomon-write-probe.{}", std::process::id()));
    match OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(file) => {
            drop(file);
            let _ = fs::remove_file(&probe);
            Ok(())
        }
        // Only a writable directory could hold an earlier probe from this pid.
        Err(err) if err.kind() == io::ErrorKind::AlreadyExists => Ok(()),
        Err(err) => Err(io::Error::new(
            err.kind(),
            format!(
                "cannot write {}: {} (the directory {} is not writable); pass --out PREFIX to write elsewhere",
                path.display(),
                err.kind(),
                dir.display()
            ),
        )),
    }
}

/// Pauses between attempts to replace a destination that another program holds
/// open. On Windows a program that opened the file without FILE_SHARE_DELETE, such
/// as an editor or a spreadsheet, makes the rename fail until it closes the file.
const RENAME_RETRY_DELAYS: [Duration; 5] = [
    Duration::from_millis(10),
    Duration::from_millis(50),
    Duration::from_millis(100),
    Duration::from_millis(250),
    Duration::from_millis(500),
];

/// Renames `from` over `to`, retrying while another program holds `to` open.
pub fn rename_replacing(from: &Path, to: &Path) -> io::Result<()> {
    rename_retrying(
        from,
        to,
        |from, to| fs::rename(from, to),
        is_sharing_violation,
        &RENAME_RETRY_DELAYS,
    )
}

fn rename_retrying(
    from: &Path,
    to: &Path,
    mut rename: impl FnMut(&Path, &Path) -> io::Result<()>,
    retryable: impl Fn(&io::Error) -> bool,
    delays: &[Duration],
) -> io::Result<()> {
    let mut delays = delays.iter();
    loop {
        match rename(from, to) {
            Ok(()) => return Ok(()),
            Err(err) if retryable(&err) => match delays.next() {
                Some(delay) => std::thread::sleep(*delay),
                None => {
                    return Err(io::Error::new(
                        err.kind(),
                        format!(
                            "could not replace '{}': another program may have it open ({err})",
                            to.display()
                        ),
                    ));
                }
            },
            Err(err) => return Err(err),
        }
    }
}

/// Windows reports a destination held open by another program as a sharing or
/// lock violation, or as access denied.
#[cfg(windows)]
fn is_sharing_violation(err: &io::Error) -> bool {
    // ERROR_ACCESS_DENIED, ERROR_SHARING_VIOLATION, ERROR_LOCK_VIOLATION
    matches!(err.raw_os_error(), Some(5 | 32 | 33))
}

#[cfg(not(windows))]
fn is_sharing_violation(_err: &io::Error) -> bool {
    false
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

#[cfg(test)]
mod destination_tests {
    use super::{AtomicFile, ensure_output_writable, rename_retrying, write_atomically_unsynced};

    #[test]
    fn a_publication_dropped_before_commit_leaves_the_destination_as_it_was() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("cohort.sscore");
        fs::write(&dest, b"previous\n").expect("seed the destination");
        let mut file = AtomicFile::create(&dest).expect("create");
        file.writer().write_all(b"partial rows").expect("write");
        drop(file);
        assert_eq!(fs::read(&dest).expect("read"), b"previous\n");
        assert_eq!(
            fs::read_dir(dir.path()).expect("read_dir").count(),
            1,
            "a temporary file stayed"
        );
    }

    #[test]
    fn a_committed_publication_appears_only_at_commit() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("results").join("cohort.sscore");
        let mut file = AtomicFile::create(&dest).expect("create");
        file.writer().write_all(b"#IID\n").expect("header");
        file.writer().write_all(b"person-1\n").expect("row");
        assert!(!dest.exists(), "the destination appeared before commit");
        file.commit().expect("commit");
        assert_eq!(fs::read(&dest).expect("read"), b"#IID\nperson-1\n");
        let entries = fs::read_dir(dest.parent().expect("parent"))
            .expect("read_dir")
            .count();
        assert_eq!(entries, 1, "a temporary file stayed");
    }

    #[test]
    fn an_unsynced_publication_replaces_the_destination_whole() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("plans").join("plan.bin");
        write_atomically_unsynced(&dest, |writer| writer.write_all(b"first")).expect("publish");
        write_atomically_unsynced(&dest, |writer| writer.write_all(b"second, longer"))
            .expect("republish");
        assert_eq!(fs::read(&dest).expect("read"), b"second, longer");
        let names: Vec<_> = fs::read_dir(dest.parent().expect("parent"))
            .expect("read_dir")
            .map(|entry| entry.expect("directory entry").file_name())
            .collect();
        assert_eq!(names, [std::ffi::OsString::from("plan.bin")]);
    }
    use std::fs;
    use std::io;
    use std::path::Path;
    use std::time::Duration;

    #[cfg(unix)]
    #[test]
    fn an_unwritable_destination_is_refused_with_the_path_and_the_way_out() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let inputs = dir.path().join("inputs");
        fs::create_dir(&inputs).expect("mkdir");
        fs::set_permissions(&inputs, fs::Permissions::from_mode(0o555)).expect("chmod 555");
        let dest = inputs.join("cohort.sex.tsv");
        let result = ensure_output_writable(&dest);
        fs::set_permissions(&inputs, fs::Permissions::from_mode(0o755)).expect("chmod 755");
        // Permission bits do not bind root, so there is nothing to observe.
        let Err(err) = result else {
            return;
        };
        let message = err.to_string();
        assert!(message.contains(&dest.display().to_string()), "{message}");
        assert!(message.contains("--out PREFIX"), "{message}");
        assert_eq!(fs::read_dir(&inputs).expect("read_dir").count(), 0);
    }

    #[test]
    fn a_missing_destination_directory_is_judged_by_its_nearest_existing_ancestor() {
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("results").join("eur").join("cohort.sscore");
        ensure_output_writable(&dest).expect("a writable ancestor accepts the output");
        assert!(!dir.path().join("results").exists(), "the probe created directories");
        assert_eq!(fs::read_dir(dir.path()).expect("read_dir").count(), 0);
    }

    #[test]
    fn a_rename_held_up_by_another_program_is_retried() {
        let mut attempts = 0;
        rename_retrying(
            Path::new("temp"),
            Path::new("cohort.sscore"),
            |_, _| {
                attempts += 1;
                if attempts < 3 {
                    Err(io::Error::other("held open"))
                } else {
                    Ok(())
                }
            },
            |_| true,
            &[Duration::ZERO; 5],
        )
        .expect("the third attempt succeeds");
        assert_eq!(attempts, 3);
    }

    #[test]
    fn a_rename_that_stays_held_names_the_file_and_the_likely_cause() {
        let mut attempts = 0;
        let err = rename_retrying(
            Path::new("temp"),
            Path::new("results/cohort.sscore"),
            |_, _| {
                attempts += 1;
                Err(io::Error::other("held open"))
            },
            |_| true,
            &[Duration::ZERO; 2],
        )
        .expect_err("the rename never succeeds");
        assert_eq!(attempts, 3);
        let message = err.to_string();
        assert!(message.contains("results/cohort.sscore"), "{message}");
        assert!(message.contains("another program may have it open"), "{message}");
    }

    #[test]
    fn other_rename_errors_are_not_retried() {
        let mut attempts = 0;
        let err = rename_retrying(
            Path::new("temp"),
            Path::new("cohort.sscore"),
            |_, _| {
                attempts += 1;
                Err(io::Error::new(io::ErrorKind::NotFound, "gone"))
            },
            |err| err.kind() == io::ErrorKind::PermissionDenied,
            &[Duration::ZERO; 5],
        )
        .expect_err("not found is final");
        assert_eq!(attempts, 1);
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    /// A program holding the destination without FILE_SHARE_DELETE, as editors and
    /// spreadsheets do, makes the publish fail with a message naming the file, and
    /// the previous version stays.
    #[cfg(windows)]
    #[test]
    fn publishing_over_a_held_file_fails_clearly_and_keeps_the_previous_version() {
        use std::io::Write;
        use std::os::windows::fs::OpenOptionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let dest = dir.path().join("cohort.sex.tsv");
        fs::write(&dest, b"previous\n").expect("seed");
        let held = fs::OpenOptions::new()
            .read(true)
            .share_mode(0)
            .open(&dest)
            .expect("hold the destination");
        let err = super::write_atomically(&dest, |writer| writer.write_all(b"new\n"))
            .expect_err("the held destination cannot be replaced");
        drop(held);
        assert!(err.to_string().contains("another program may have it open"), "{err}");
        assert_eq!(fs::read(&dest).expect("read"), b"previous\n");
    }
}
