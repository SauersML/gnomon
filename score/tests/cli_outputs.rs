//! `--out PREFIX` and atomic publication, exercised through the real binaries on
//! the `data/testdata/ld` fixture (200 samples x 1500 variants).
//!
//! Each test stages its own copy of the inputs, and proves what a run wrote by
//! comparing directory snapshots taken before and after it.

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use tempfile::tempdir;

type TestResult = Result<(), Box<dyn Error>>;

const TERMS_BIN: &str = env!("CARGO_BIN_EXE_gnomon-terms");

/// Copies the fixture fileset into `dir` as `cohort.{bed,bim,fam}` and writes a
/// native score file `w.tsv` over every seventh variant. Returns the genotype
/// prefix and the score file.
fn stage_inputs(dir: &Path) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/testdata");
    fs::create_dir_all(dir)?;
    for ext in ["bed", "bim", "fam"] {
        fs::copy(
            fixture.join(format!("ld.{ext}")),
            dir.join(format!("cohort.{ext}")),
        )?;
    }
    let bim = fs::read_to_string(fixture.join("ld.bim"))?;
    let mut score = String::from("variant_id\teffect_allele\tother_allele\tw\n");
    for (index, line) in bim.lines().enumerate().step_by(7) {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let weight = (index % 13) as f64 * 0.25 - 1.5;
        score.push_str(&format!(
            "{}:{}\t{}\t{}\t{weight}\n",
            fields[0], fields[3], fields[4], fields[5]
        ));
    }
    let score_path = dir.join("w.tsv");
    fs::write(&score_path, score)?;
    Ok((dir.join("cohort"), score_path))
}

/// Every file under `root`, keyed by its path relative to `root`.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else {
                let bytes = fs::read(&path).expect("read");
                files.insert(path.strip_prefix(root).expect("under root").to_path_buf(), bytes);
            }
        }
    }
    files
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = OsString::from(prefix.as_os_str());
    path.push(".");
    path.push(suffix);
    PathBuf::from(path)
}

fn run(binary: &str, cwd: &Path, args: &[&OsStr]) -> Output {
    Command::new(binary)
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("spawn gnomon")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "gnomon failed with {:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Makes a directory read-only for the life of the guard.
#[cfg(unix)]
struct ReadOnly(PathBuf);

#[cfg(unix)]
impl ReadOnly {
    fn new(dir: &Path) -> Self {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o555)).expect("chmod 555");
        Self(dir.to_path_buf())
    }
}

#[cfg(unix)]
impl Drop for ReadOnly {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o755));
    }
}

#[test]
fn terms_out_writes_only_the_prefixed_sex_table() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, _) = stage_inputs(&inputs)?;
    let work = tmp.path().join("work");
    fs::create_dir(&work)?;
    let inputs_before = snapshot(&inputs);

    let prefix = tmp.path().join("results").join("eur");
    assert_success(&run(
        TERMS_BIN,
        &work,
        &[
            OsStr::new("--sex"),
            OsStr::new("--out"),
            prefix.as_os_str(),
            genotypes.as_os_str(),
        ],
    ));

    assert_eq!(snapshot(&inputs), inputs_before, "the input directory changed");
    assert!(snapshot(&work).is_empty(), "the working directory changed");
    let results = snapshot(&tmp.path().join("results"));
    assert_eq!(
        results.keys().collect::<Vec<_>>(),
        [Path::new("eur.sex.tsv")]
    );

    // --out moves the table; it must not change it.
    let default_inputs = tmp.path().join("default");
    let (default_genotypes, _) = stage_inputs(&default_inputs)?;
    assert_success(&run(
        TERMS_BIN,
        &work,
        &[OsStr::new("--sex"), default_genotypes.as_os_str()],
    ));
    assert_eq!(
        results[Path::new("eur.sex.tsv")],
        fs::read(default_inputs.join("cohort.sex.tsv"))?
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn terms_out_succeeds_on_a_read_only_input_directory() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, _) = stage_inputs(&inputs)?;
    let _read_only = ReadOnly::new(&inputs);
    let inputs_before = snapshot(&inputs);

    let prefix = tmp.path().join("results").join("eur");
    assert_success(&run(
        TERMS_BIN,
        tmp.path(),
        &[
            OsStr::new("--sex"),
            OsStr::new("--out"),
            prefix.as_os_str(),
            genotypes.as_os_str(),
        ],
    ));

    assert!(fs::metadata(with_suffix(&prefix, "sex.tsv"))?.len() > 0);
    assert_eq!(snapshot(&inputs), inputs_before);
    Ok(())
}

/// Runs with different prefixes agree, and runs without `--out` that publish
/// over one another beside the inputs still leave one complete table.
#[test]
fn concurrent_terms_runs_agree_and_leave_no_partial_table() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, _) = stage_inputs(&inputs)?;
    let prefixes = ["results/a/eur", "results/b/eur", "results/b/afr"]
        .map(|prefix| tmp.path().join(prefix));

    let mut commands: Vec<Command> = prefixes
        .iter()
        .map(|prefix| {
            let mut command = Command::new(TERMS_BIN);
            command.arg("--sex").arg("--out").arg(prefix).arg(&genotypes);
            command
        })
        .collect();
    for _ in 0..2 {
        let mut command = Command::new(TERMS_BIN);
        command.arg("--sex").arg(&genotypes);
        commands.push(command);
    }
    let children = commands
        .iter_mut()
        .map(|command| {
            command
                .current_dir(tmp.path())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        })
        .collect::<Result<Vec<_>, _>>()?;
    for child in children {
        assert_success(&child.wait_with_output()?);
    }

    let default_table = fs::read(inputs.join("cohort.sex.tsv"))?;
    assert!(!default_table.is_empty());
    for prefix in &prefixes {
        assert_eq!(
            fs::read(with_suffix(prefix, "sex.tsv"))?,
            default_table,
            "{} differs",
            prefix.display()
        );
    }
    let leftovers: Vec<PathBuf> = snapshot(tmp.path())
        .into_keys()
        .filter(|path| path.to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    Ok(())
}

#[test]
fn terms_out_rejects_a_directory_or_remote_prefix() -> TestResult {
    let tmp = tempdir()?;
    let (genotypes, _) = stage_inputs(&tmp.path().join("inputs"))?;
    for prefix in ["results/", "gs://bucket/eur"] {
        let output = run(
            TERMS_BIN,
            tmp.path(),
            &[
                OsStr::new("--sex"),
                OsStr::new("--out"),
                OsStr::new(prefix),
                genotypes.as_os_str(),
            ],
        );
        assert!(!output.status.success(), "--out {prefix} was accepted");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--out"), "--out {prefix}: {stderr}");
    }
    assert!(!tmp.path().join("results").exists());
    Ok(())
}
