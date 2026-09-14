//! `gnomon all` against the three subcommands it runs, through the real binary.
//!
//! Each test stages its inputs and a projection model fitted on them, runs
//! `gnomon score`, `gnomon project` and `gnomon terms --sex` over one copy and
//! `gnomon all` over another, and compares what the two wrote byte for byte.

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::time::{Duration, SystemTime};

use tempfile::tempdir;

type TestResult = Result<(), Box<dyn Error>>;

const GNOMON: &str = env!("CARGO_BIN_EXE_gnomon");

/// Runs `gnomon <args>` in `cwd`, with a fixed Rayon pool size when given one.
fn gnomon(cwd: &Path, args: &[&str], threads: Option<usize>) -> Output {
    let mut command = Command::new(GNOMON);
    command.current_dir(cwd).args(args);
    if let Some(threads) = threads {
        command.env("RAYON_NUM_THREADS", threads.to_string());
    }
    command.output().expect("spawn gnomon")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "gnomon failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Every file under `root`, keyed by its path relative to `root`. The
/// `.hwe.project.bin` projection cache is left out: it records the modification
/// time of the model it was derived from, so two copies of one model never
/// share it byte for byte.
fn snapshot(root: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("directory entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if !path.to_string_lossy().ends_with(".hwe.project.bin") {
                let bytes = fs::read(&path).expect("read");
                files.insert(
                    path.strip_prefix(root).expect("under root").to_path_buf(),
                    bytes,
                );
            }
        }
    }
    files
}

/// Asserts that `actual` holds exactly the files `expected` holds, with the same
/// bytes, and names the first file that differs.
fn assert_same_files(actual: &Path, expected: &Path) {
    let (actual, expected) = (snapshot(actual), snapshot(expected));
    assert_eq!(
        actual.keys().collect::<Vec<_>>(),
        expected.keys().collect::<Vec<_>>(),
        "the two runs wrote different files"
    );
    for (path, bytes) in &expected {
        assert!(actual[path] == *bytes, "{} differs", path.display());
    }
}

/// The one file under `root` whose name ends with `suffix`.
fn only_file_ending_with(root: &Path, suffix: &str) -> PathBuf {
    let matches: Vec<PathBuf> = snapshot(root)
        .into_keys()
        .filter(|path| path.to_string_lossy().ends_with(suffix))
        .collect();
    assert_eq!(matches.len(), 1, "files ending with {suffix}: {matches:?}");
    root.join(&matches[0])
}

fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut path = prefix.as_os_str().to_owned();
    path.push(".");
    path.push(suffix);
    PathBuf::from(path)
}

/// Copies every file of `from` into the new directory `to`.
fn copy_dir(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        fs::copy(entry.path(), to.join(entry.file_name()))?;
    }
    Ok(())
}

/// Fits a two-component projection model on `genotypes` in `dir`, written as
/// `dir/cohort.hwe.json`, where `gnomon project` looks for it.
fn fit(dir: &Path, genotypes: &str) {
    assert_success(&gnomon(
        dir,
        &[
            "fit",
            genotypes,
            "--components",
            "2",
            "--allow-unconverged",
            "--out",
            "cohort",
        ],
        None,
    ));
}

/// Stages the `data/testdata/ld` fileset (200 samples x 1,500 chromosome 1
/// variants) as `cohort.{bed,bim,fam}`, a score file `w.tsv` over every seventh
/// variant, and a model fitted on the cohort.
fn stage_plink(dir: &Path) -> TestResult {
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
    fs::write(dir.join("w.tsv"), score)?;
    fit(dir, "cohort.bed");
    Ok(())
}

/// Stages `cohort.vcf` with `n_samples` samples named S0, S1, ...: 2,600
/// autosomal, 708 chrX and 308 chrY SNVs placed as in the terms sex fixture.
/// Even samples are female (random chrX calls, so some heterozygous, and no chrY
/// calls); odd samples are male (homozygous chrX and chrY calls). Also stages a
/// score file `w.tsv` over every seventh autosomal variant and a model fitted on
/// the VCF.
fn stage_vcf(dir: &Path, n_samples: usize) -> TestResult {
    fs::create_dir_all(dir)?;
    let mut state = 0x853c_49e6_748f_ea9b_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    let mut rows: Vec<(String, u64)> = (0..2_600u64)
        .map(|i| ((1 + i * 22 / 2_600).to_string(), 1_000 + i * 10))
        .collect();
    let mut x_positions: Vec<u64> = (0..700u64).map(|i| 5_000 + i * 224_000).collect();
    x_positions.extend([
        10_000,
        10_001,
        2_781_479,
        2_781_480,
        155_701_382,
        155_701_383,
        156_030_895,
        156_030_896,
    ]);
    x_positions.sort_unstable();
    rows.extend(x_positions.into_iter().map(|pos| ("X".to_string(), pos)));
    // The fixture is GRCh38, so its Y rows stay within GRCh38's chrY
    // (57,227,415 bp); a Y position past that would prove GRCh37 instead.
    let mut y_positions: Vec<u64> = (0..300u64).map(|i| 5_000 + i * 190_000).collect();
    y_positions.extend([
        10_000, 10_001, 2_781_479, 2_781_480, 56_887_902, 56_887_903, 57_217_415, 57_217_416,
    ]);
    y_positions.sort_unstable();
    rows.extend(y_positions.into_iter().map(|pos| ("Y".to_string(), pos)));

    let mut vcf = String::from(
        "##fileformat=VCFv4.2\n\
         ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
         #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT",
    );
    for sample in 0..n_samples {
        vcf.push_str(&format!("\tS{sample}"));
    }
    vcf.push('\n');
    let mut score = String::from("variant_id\teffect_allele\tother_allele\tw\n");
    for (index, (chrom, pos)) in rows.iter().enumerate() {
        vcf.push_str(&format!("{chrom}\t{pos}\tv{index}\tA\tG\t.\tPASS\t.\tGT"));
        for sample in 0..n_samples {
            let male = sample % 2 == 1;
            let roll = next();
            let call = match chrom.as_str() {
                "X" | "Y" if male => ["0/0", "1/1"][(roll & 1) as usize],
                "Y" => "./.",
                _ => ["0/0", "0/1", "1/1"][(roll % 3) as usize],
            };
            vcf.push('\t');
            vcf.push_str(call);
        }
        vcf.push('\n');
        if chrom != "X" && chrom != "Y" && index % 7 == 0 {
            let weight = (index % 13) as f64 * 0.25 - 1.5;
            score.push_str(&format!("{chrom}:{pos}\tG\tA\t{weight}\n"));
        }
    }
    fs::write(dir.join("cohort.vcf"), vcf)?;
    fs::write(dir.join("w.tsv"), score)?;
    fit(dir, "cohort.vcf");
    Ok(())
}

/// Runs `gnomon score`, `gnomon project` and `gnomon terms --sex` over
/// `genotypes` in `dir`, one after another, as a user would.
fn run_separately(dir: &Path, genotypes: &str) {
    assert_success(&gnomon(dir, &["score", "w.tsv", genotypes], None));
    assert_success(&gnomon(dir, &["project", genotypes], None));
    assert_success(&gnomon(dir, &["terms", genotypes, "--sex"], None));
}

#[test]
fn all_writes_what_the_three_subcommands_write_on_plink() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    stage_plink(&inputs)?;
    let separate = tmp.path().join("separate");
    copy_dir(&inputs, &separate)?;
    run_separately(&separate, "cohort");

    for (threads, shape) in [
        (1, "in order: one thread"),
        (4, "concurrently on 4 threads"),
    ] {
        let all = tmp.path().join(format!("all-t{threads}"));
        copy_dir(&inputs, &all)?;
        let output = gnomon(&all, &["all", "w.tsv", "cohort"], Some(threads));
        assert_success(&output);
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains(shape), "expected {shape:?}:\n{stdout}");
        assert_same_files(&all, &separate);
    }
    Ok(())
}

#[test]
fn all_writes_what_the_three_subcommands_write_on_a_multi_sample_vcf() -> TestResult {
    const SAMPLES: usize = 40;
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    stage_vcf(&inputs, SAMPLES)?;
    let separate = tmp.path().join("separate");
    copy_dir(&inputs, &separate)?;
    run_separately(&separate, "cohort.vcf");

    let all = tmp.path().join("all");
    copy_dir(&inputs, &all)?;
    assert_success(&gnomon(&all, &["all", "w.tsv", "cohort.vcf"], Some(4)));
    assert_same_files(&all, &separate);

    // What converting the VCF to PLINK used to lose: each sample's own ID, every
    // sample past the first, and heterozygous chrX calls.
    let sex = fs::read_to_string(only_file_ending_with(&all, ".sex.tsv"))?;
    let mut lines = sex.lines();
    let header: Vec<&str> = lines.next().expect("sex.tsv header").split('\t').collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|field| *field == name)
            .unwrap_or_else(|| panic!("no {name} column in {header:?}"))
    };
    let (iid, x_het) = (column("IID"), column("X_NonPAR_Het"));
    let rows: Vec<Vec<&str>> = lines.map(|line| line.split('\t').collect()).collect();
    assert_eq!(rows.len(), SAMPLES, "sex.tsv rows");
    for (sample, row) in rows.iter().enumerate() {
        assert_eq!(row[iid], format!("S{sample}"));
        if sample % 2 == 0 {
            let hets: u64 = row[x_het].parse()?;
            assert!(hets > 0, "female S{sample} has no chrX heterozygous calls");
        }
    }
    let sscore = fs::read_to_string(only_file_ending_with(&all, ".sscore"))?;
    assert_eq!(
        sscore.lines().filter(|line| !line.starts_with('#')).count(),
        SAMPLES,
        ".sscore rows"
    );
    let metadata = fs::read_to_string(only_file_ending_with(
        &all,
        ".projection_scores.metadata.json",
    ))?;
    assert!(
        metadata.contains(&format!("\"rows\": {SAMPLES}")),
        "projection rows:\n{metadata}"
    );
    Ok(())
}

#[test]
fn a_failing_phase_lets_the_others_finish_and_names_itself() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    stage_plink(&inputs)?;
    // With no model beside the cohort, projection fails while score and terms do not.
    for entry in fs::read_dir(&inputs)? {
        let path = entry?.path();
        if path.to_string_lossy().contains("cohort.hwe") {
            fs::remove_file(path)?;
        }
    }
    let separate = tmp.path().join("separate");
    copy_dir(&inputs, &separate)?;
    assert_success(&gnomon(&separate, &["score", "w.tsv", "cohort"], None));
    assert!(
        !gnomon(&separate, &["project", "cohort"], None)
            .status
            .success()
    );
    assert_success(&gnomon(&separate, &["terms", "cohort", "--sex"], None));

    for threads in [1, 4] {
        let all = tmp.path().join(format!("all-t{threads}"));
        copy_dir(&inputs, &all)?;
        let output = gnomon(&all, &["all", "w.tsv", "cohort"], Some(threads));
        assert!(
            !output.status.success(),
            "gnomon all exited 0 with a failed phase"
        );
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stdout.contains("[all] project phase: FAILED"), "{stdout}");
        for phase in ["score", "terms"] {
            assert!(
                stdout.contains(&format!("[all] {phase} phase: ")),
                "{stdout}"
            );
            assert!(
                !stdout.contains(&format!("[all] {phase} phase: FAILED")),
                "{stdout}"
            );
        }
        assert!(
            stderr.contains("1 of 3 gnomon all phases failed (project: "),
            "{stderr}"
        );
        assert_same_files(&all, &separate);
    }
    Ok(())
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

#[cfg(unix)]
#[test]
fn all_out_writes_nothing_beside_a_read_only_input_directory() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    stage_plink(&inputs)?;
    // The reference runs in the input directory itself, as `gnomon all` will. A copy
    // restamps cohort.hwe.json, which invalidates the projection cache `fit` wrote
    // beside it, so projection in a copy would load the model through a different
    // path. The reference outputs then move aside before the directory is locked.
    run_separately(&inputs, "cohort");
    let separate = tmp.path().join("separate");
    fs::create_dir(&separate)?;
    for standalone in [
        "cohort_w.sscore",
        "cohort.projection_scores.bin",
        "cohort.projection_scores.metadata.json",
        "cohort.sex.tsv",
    ] {
        fs::rename(inputs.join(standalone), separate.join(standalone))?;
    }

    let _read_only = ReadOnly::new(&inputs);
    let inputs_before = snapshot(&inputs);
    let work = tmp.path().join("work");
    fs::create_dir(&work)?;
    let prefix = tmp.path().join("results").join("eur");
    let score = inputs.join("w.tsv");
    let genotypes = inputs.join("cohort");
    let output = Command::new(GNOMON)
        .current_dir(&work)
        .env("RAYON_NUM_THREADS", "4")
        .args([
            OsStr::new("all"),
            score.as_os_str(),
            genotypes.as_os_str(),
            OsStr::new("--out"),
            prefix.as_os_str(),
        ])
        .output()?;
    assert_success(&output);

    assert_eq!(
        snapshot(&inputs),
        inputs_before,
        "the input directory changed"
    );
    assert!(snapshot(&work).is_empty(), "the working directory changed");
    // --out moves every output; it must not change any of them.
    for (suffix, standalone) in [
        ("sscore", "cohort_w.sscore"),
        ("projection_scores.bin", "cohort.projection_scores.bin"),
        (
            "projection_scores.metadata.json",
            "cohort.projection_scores.metadata.json",
        ),
        ("sex.tsv", "cohort.sex.tsv"),
    ] {
        assert!(
            fs::read(with_suffix(&prefix, suffix))? == fs::read(separate.join(standalone))?,
            "{} differs from the standalone {standalone}",
            with_suffix(&prefix, suffix).display()
        );
    }
    Ok(())
}

#[test]
fn concurrent_all_runs_with_different_out_prefixes_agree() -> TestResult {
    let tmp = tempdir()?;
    let inputs = tmp.path().join("inputs");
    stage_plink(&inputs)?;
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
            Command::new(GNOMON)
                .current_dir(tmp.path())
                .arg("all")
                .arg(inputs.join("w.tsv"))
                .arg(inputs.join("cohort"))
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

    for suffix in ["sscore", "projection_scores.bin", "sex.tsv"] {
        let first = fs::read(with_suffix(&prefixes[0], suffix))?;
        for prefix in &prefixes[1..] {
            assert!(
                fs::read(with_suffix(prefix, suffix))? == first,
                "{} differs",
                with_suffix(prefix, suffix).display()
            );
        }
    }
    let leftovers: Vec<PathBuf> = snapshot(&tmp.path().join("results"))
        .into_keys()
        .filter(|path| {
            let name = path.to_string_lossy();
            name.ends_with(".tmp") || name.contains("gnomon-checkpoint")
        })
        .collect();
    assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
    Ok(())
}

/// A model parsed from its JSON must project exactly like the fitted model `fit` cached.
/// Restamping the JSON invalidates that cache, so projection has to parse the JSON, and a
/// parse that is a ULP off anywhere changes the scores.
#[test]
fn projection_from_a_reparsed_json_matches_the_fit_written_cache() -> TestResult {
    let tmp = tempdir()?;
    let dir = tmp.path().join("cohort");
    stage_plink(&dir)?;
    let json = dir.join("cohort.hwe.json");
    let cache = dir.join("cohort.hwe.project.bin");
    let cache_written_by_fit = fs::metadata(&cache)?.modified()?;

    assert_success(&gnomon(&dir, &["project", "cohort"], Some(4)));
    let from_cache = dir.join("from_cache.projection_scores.bin");
    fs::rename(dir.join("cohort.projection_scores.bin"), &from_cache)?;

    fs::File::options()
        .write(true)
        .open(&json)?
        .set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000_000))?;
    assert_success(&gnomon(&dir, &["project", "cohort"], Some(4)));
    assert_ne!(
        fs::metadata(&cache)?.modified()?,
        cache_written_by_fit,
        "projection read the stale cache instead of parsing the JSON"
    );
    assert!(
        fs::read(dir.join("cohort.projection_scores.bin"))? == fs::read(&from_cache)?,
        "projection from the reparsed JSON differs from projection from the fit-written cache"
    );
    Ok(())
}
