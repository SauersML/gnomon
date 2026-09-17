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

pub(super) const SCORE_BIN: &str = env!("CARGO_BIN_EXE_gnomon-score");
const TERMS_BIN: &str = env!("CARGO_BIN_EXE_gnomon-terms");

/// Copies the fixture fileset into `dir` as `cohort.{bed,bim,fam}` and writes a
/// native score file `w.tsv` over every seventh variant. Returns the genotype
/// prefix and the score file.
pub(super) fn stage_inputs(dir: &Path) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
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

pub(super) fn run(binary: &str, cwd: &Path, args: &[&OsStr]) -> Output {
    Command::new(binary)
        .current_dir(cwd)
        .args(args)
        .output()
        .expect("spawn gnomon")
}

pub(super) fn assert_success(output: &Output) {
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

#[test]
fn score_out_writes_only_under_the_prefix_directory() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, score) = stage_inputs(&inputs)?;
    let work = tmp.path().join("work");
    fs::create_dir(&work)?;
    let inputs_before = snapshot(&inputs);

    let prefix = tmp.path().join("results").join("eur");
    assert_success(&run(
        SCORE_BIN,
        &work,
        &[
            score.as_os_str(),
            genotypes.as_os_str(),
            OsStr::new("--out"),
            prefix.as_os_str(),
        ],
    ));

    assert_eq!(snapshot(&inputs), inputs_before, "the input directory changed");
    assert!(snapshot(&work).is_empty(), "the working directory changed");
    let results = snapshot(&tmp.path().join("results"));
    for path in results.keys() {
        assert!(
            path == Path::new("eur.sscore") || path.starts_with("gnomon_score_cache"),
            "unexpected output {}",
            path.display()
        );
    }

    // --out moves the results; it must not change them.
    let default_inputs = tmp.path().join("default");
    let (default_genotypes, default_score) = stage_inputs(&default_inputs)?;
    assert_success(&run(
        SCORE_BIN,
        &work,
        &[default_score.as_os_str(), default_genotypes.as_os_str()],
    ));
    assert_eq!(
        results.get(Path::new("eur.sscore")).expect("eur.sscore"),
        &fs::read(default_inputs.join("cohort_w.sscore"))?
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn score_out_succeeds_on_a_read_only_input_directory() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, score) = stage_inputs(&inputs)?;
    let _read_only = ReadOnly::new(&inputs);
    let inputs_before = snapshot(&inputs);

    let prefix = tmp.path().join("results").join("eur");
    assert_success(&run(
        SCORE_BIN,
        tmp.path(),
        &[
            score.as_os_str(),
            genotypes.as_os_str(),
            OsStr::new("--out"),
            prefix.as_os_str(),
        ],
    ));

    assert!(fs::metadata(with_suffix(&prefix, "sscore"))?.len() > 0);
    assert_eq!(snapshot(&inputs), inputs_before);
    Ok(())
}

/// Without `--out`, a score file in a directory gnomon cannot write still gets
/// its sorted cache, under the user cache directory instead of beside it.
#[cfg(target_os = "linux")]
#[test]
fn a_read_only_score_directory_caches_under_the_user_cache_directory() -> TestResult {
    let tmp = tempdir()?;
    let (genotypes, _) = stage_inputs(&tmp.path().join("genotypes"))?;
    let scores = tmp.path().join("scores");
    let (_, score) = stage_inputs(&scores)?;
    let _read_only = ReadOnly::new(&scores);
    let scores_before = snapshot(&scores);
    let cache_home = tmp.path().join("xdg-cache");

    let output = Command::new(SCORE_BIN)
        .current_dir(tmp.path())
        .env("XDG_CACHE_HOME", &cache_home)
        .arg(&score)
        .arg(&genotypes)
        .output()?;
    assert_success(&output);

    assert!(tmp.path().join("genotypes").join("cohort_w.sscore").is_file());
    assert_eq!(snapshot(&scores), scores_before);
    let cached = snapshot(&cache_home.join("gnomon").join("score_cache"));
    assert!(
        cached
            .keys()
            .any(|path| path.to_string_lossy().ends_with(".sorted.gnomon.tsv")),
        "no sorted cache under the user cache directory: {:?}",
        cached.keys().collect::<Vec<_>>()
    );
    Ok(())
}

#[test]
fn concurrent_score_runs_with_different_out_prefixes_agree() -> TestResult {
    let tmp = tempdir()?;
    let (genotypes, score) = stage_inputs(&tmp.path().join("inputs"))?;
    // The last two prefixes share a directory, and therefore one score-file cache.
    let prefixes = [
        "results/a/eur",
        "results/b/eur",
        "results/shared/one",
        "results/shared/two",
    ]
    .map(|prefix| tmp.path().join(prefix));

    let children = prefixes
        .iter()
        .map(|prefix| {
            Command::new(SCORE_BIN)
                .current_dir(tmp.path())
                .arg(&score)
                .arg(&genotypes)
                .arg("--out")
                .arg(prefix)
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
        })
        .collect::<Result<Vec<_>, _>>()?;
    for child in children {
        assert_success(&child.wait_with_output()?);
    }

    let first = fs::read(with_suffix(&prefixes[0], "sscore"))?;
    for prefix in &prefixes[1..] {
        assert_eq!(
            fs::read(with_suffix(prefix, "sscore"))?,
            first,
            "{} differs",
            prefix.display()
        );
    }
    let leftovers: Vec<PathBuf> = snapshot(&tmp.path().join("results"))
        .into_keys()
        .filter(|path| path.to_string_lossy().ends_with(".tmp"))
        .collect();
    assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    Ok(())
}

/// The fixture's native score text as a PGS Catalog scoring file without a pgs_id.
fn catalog_text(native: &str) -> String {
    let mut text =
        String::from("chr_name\tchr_position\teffect_allele\tother_allele\teffect_weight\n");
    for row in native.lines().skip(1) {
        text.push_str(&row.replacen(':', "\t", 1));
        text.push('\n');
    }
    text
}

/// Swaps each row's effect and other allele: the same size, other scores.
fn swap_alleles(text: &str, first_allele_column: usize) -> String {
    let mut swapped = String::with_capacity(text.len());
    for (index, row) in text.lines().enumerate() {
        let mut fields: Vec<&str> = row.split('\t').collect();
        if index > 0 {
            fields.swap(first_allele_column, first_allele_column + 1);
        }
        swapped.push_str(&fields.join("\t"));
        swapped.push('\n');
    }
    swapped
}

/// A score file replaced by a newer version under an older timestamp, as `rsync -t`
/// and `cp -p` leave it, is scored as the newer version: a repeat run gives what a
/// run on fresh copies with no cache gives.
#[test]
fn a_score_file_updated_under_an_older_timestamp_is_scored_as_updated() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, native) = stage_inputs(&inputs)?;
    let catalog = inputs.join("catalog.txt");
    fs::write(&catalog, catalog_text(&fs::read_to_string(&native)?))?;

    for (score, first_allele_column) in [(native, 1), (catalog, 2)] {
        let stem = score.file_stem().expect("score stem").to_os_string();
        let sscore_name = format!("cohort_{}.sscore", stem.to_string_lossy());
        let sscore = inputs.join(&sscore_name);
        let cache_home = tmp.path().join("xdg").join(&stem);
        let score_run = |score: &Path, genotypes: &Path, cache_home: &Path| {
            Command::new(SCORE_BIN)
                .current_dir(tmp.path())
                .env("XDG_CACHE_HOME", cache_home)
                .arg(score)
                .arg(genotypes)
                .output()
        };

        assert_success(&score_run(&score, &genotypes, &cache_home)?);
        let before = fs::read(&sscore)?;

        let modified = fs::metadata(&score)?.modified()?;
        let original = fs::read_to_string(&score)?;
        let updated = swap_alleles(&original, first_allele_column);
        assert_eq!(updated.len(), original.len());
        fs::write(&score, &updated)?;
        fs::File::options()
            .write(true)
            .open(&score)?
            .set_times(fs::FileTimes::new().set_modified(modified))?;
        assert_eq!(fs::metadata(&score)?.modified()?, modified);

        fs::remove_file(&sscore)?;
        assert_success(&score_run(&score, &genotypes, &cache_home)?);
        let repeat = fs::read(&sscore)?;
        assert!(
            repeat != before,
            "{}: the old version was scored",
            score.display()
        );

        // A catalog file without a pgs_id is labelled by its directory's name, so the
        // fresh copy sits in a directory of the same name.
        let fresh = tmp.path().join("fresh").join(&stem).join("inputs");
        let (fresh_genotypes, _) = stage_inputs(&fresh)?;
        let fresh_score = fresh.join(score.file_name().expect("score name"));
        fs::write(&fresh_score, &updated)?;
        assert_success(&score_run(
            &fresh_score,
            &fresh_genotypes,
            &tmp.path().join("xdg-fresh").join(&stem),
        )?);
        assert!(
            repeat == fs::read(fresh.join(&sscore_name))?,
            "{}: a repeat run differs from a cold run",
            score.display()
        );
    }
    Ok(())
}

/// The same through a directory of score files, where a previous run's converted
/// copy beside the replaced source must not be scored in its place.
#[test]
fn a_score_directory_updated_under_an_older_timestamp_is_scored_as_updated() -> TestResult {
    let tmp = tempdir()?;
    let score_run = |scores: &Path, genotypes: &Path, cache_home: &Path| {
        Command::new(SCORE_BIN)
            .current_dir(tmp.path())
            .env("XDG_CACHE_HOME", cache_home)
            .arg(scores)
            .arg(genotypes)
            .output()
    };
    let inputs = tmp.path().join("inputs");
    let (genotypes, native) = stage_inputs(&inputs)?;
    let original = catalog_text(&fs::read_to_string(&native)?);
    let updated = swap_alleles(&original, 2);
    assert_eq!(updated.len(), original.len());
    let scores = inputs.join("scores");
    fs::create_dir(&scores)?;
    let catalog = scores.join("catalog.txt");
    fs::write(&catalog, &original)?;
    let sscore = inputs.join("cohort_scores.sscore");
    let cache_home = tmp.path().join("xdg");

    assert_success(&score_run(&scores, &genotypes, &cache_home)?);
    let before = fs::read(&sscore)?;
    assert!(
        scores.join("catalog.gnomon.tsv").is_file(),
        "no converted copy was left beside the source"
    );

    let modified = fs::metadata(&catalog)?.modified()?;
    fs::write(&catalog, &updated)?;
    fs::File::options()
        .write(true)
        .open(&catalog)?
        .set_times(fs::FileTimes::new().set_modified(modified))?;
    assert_eq!(fs::metadata(&catalog)?.modified()?, modified);

    fs::remove_file(&sscore)?;
    assert_success(&score_run(&scores, &genotypes, &cache_home)?);
    let repeat = fs::read(&sscore)?;
    assert!(repeat != before, "the old version was scored");

    // The catalog file is labelled by its directory's name, so the fresh copy sits in
    // a directory of the same name.
    let fresh = tmp.path().join("fresh").join("inputs");
    let (fresh_genotypes, _) = stage_inputs(&fresh)?;
    let fresh_scores = fresh.join("scores");
    fs::create_dir(&fresh_scores)?;
    fs::write(fresh_scores.join("catalog.txt"), &updated)?;
    assert_success(&score_run(
        &fresh_scores,
        &fresh_genotypes,
        &tmp.path().join("xdg-fresh"),
    )?);
    assert!(
        repeat == fs::read(fresh.join("cohort_scores.sscore"))?,
        "a repeat run differs from a cold run"
    );
    Ok(())
}

#[test]
fn score_out_rejects_a_directory_or_remote_prefix() -> TestResult {
    let tmp = tempdir()?;
    let (genotypes, score) = stage_inputs(&tmp.path().join("inputs"))?;
    for prefix in ["results/", "gs://bucket/eur"] {
        let output = run(
            SCORE_BIN,
            tmp.path(),
            &[
                score.as_os_str(),
                genotypes.as_os_str(),
                OsStr::new("--out"),
                OsStr::new(prefix),
            ],
        );
        assert!(!output.status.success(), "--out {prefix} was accepted");
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("--out"), "--out {prefix}: {stderr}");
    }
    assert!(!tmp.path().join("results").exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn score_without_out_refuses_a_read_only_input_directory_before_scoring() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, score) = stage_inputs(&inputs)?;
    let _read_only = ReadOnly::new(&inputs);
    let before = snapshot(&inputs);

    let output = Command::new(SCORE_BIN)
        .current_dir(tmp.path())
        .env("XDG_CACHE_HOME", tmp.path().join("xdg-cache"))
        .arg(&score)
        .arg(&genotypes)
        .output()?;

    // Permission bits do not bind root, so there is nothing to observe.
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot write"), "{stderr}");
    assert!(stderr.contains("cohort_w.sscore"), "{stderr}");
    assert!(stderr.contains("--out PREFIX"), "{stderr}");
    assert!(!stderr.contains("> Writing"), "the run scored before refusing: {stderr}");
    assert_eq!(snapshot(&inputs), before);
    Ok(())
}

#[cfg(unix)]
#[test]
fn terms_without_out_refuses_a_read_only_input_directory_before_inference() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    let (genotypes, _) = stage_inputs(&inputs)?;
    let _read_only = ReadOnly::new(&inputs);
    let before = snapshot(&inputs);

    let output = run(TERMS_BIN, tmp.path(), &[genotypes.as_os_str(), OsStr::new("--sex")]);

    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("cannot write"), "{stderr}");
    assert!(stderr.contains("sex.tsv"), "{stderr}");
    assert!(stderr.contains("--out PREFIX"), "{stderr}");
    assert!(
        !stderr.contains("Inferred Genome Build"),
        "inference ran before refusing: {stderr}"
    );
    assert_eq!(snapshot(&inputs), before);
    Ok(())
}

#[test]
fn score_names_a_missing_genotype_path_as_missing() -> TestResult {
    let tmp = tempdir()?;
    let (_, score) = stage_inputs(&tmp.path().join("inputs"))?;
    let missing = tmp.path().join("no_such_panel");

    let output = run(SCORE_BIN, tmp.path(), &[score.as_os_str(), missing.as_os_str()]);

    assert!(!output.status.success(), "a missing genotype path was accepted");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("no such file or PLINK/PGEN fileset"), "{stderr}");
    assert!(!stderr.contains("Could not determine input format"), "{stderr}");
    Ok(())
}

/// Library warnings reach the user: the CLI installs a logger that writes records to
/// stderr before it does anything else. An unrecognized GNOMON_LOG_LEVEL is itself
/// reported as a warning through that logger, even for --help.
#[test]
fn warnings_are_logged_to_stderr_and_never_to_stdout() -> TestResult {
    let tmp = tempdir()?;
    let loud = Command::new(TERMS_BIN)
        .current_dir(tmp.path())
        .env("GNOMON_LOG_LEVEL", "loud")
        .arg("--help")
        .output()?;
    let stderr = String::from_utf8_lossy(&loud.stderr);
    assert!(
        stderr.contains("[WARN]"),
        "no warning record on stderr: {stderr}"
    );
    assert!(stderr.contains("GNOMON_LOG_LEVEL"), "{stderr}");
    assert!(
        !String::from_utf8_lossy(&loud.stdout).contains("[WARN]"),
        "a log record reached stdout"
    );

    let valid = Command::new(TERMS_BIN)
        .current_dir(tmp.path())
        .env("GNOMON_LOG_LEVEL", "debug")
        .arg("--help")
        .output()?;
    assert!(
        !String::from_utf8_lossy(&valid.stderr).contains("is not a log level"),
        "a valid level was reported as unrecognized"
    );
    Ok(())
}
