//! `--blocks`: per-block partial scores, exercised through the real binary on the
//! `data/testdata/ld` fixture. The fixture's weights are multiples of 0.25, so every
//! sum is exact in f64 and the block partials must add up to the total bit for bit.

use std::collections::BTreeMap;
use std::error::Error;
use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::Command;

use tempfile::tempdir;

use super::cli_outputs::{SCORE_BIN, assert_success, run, stage_inputs};

type TestResult = Result<(), Box<dyn Error>>;

/// An `.sscore` parsed into its metadata lines, header names and per-row tokens.
struct Sscore {
    metadata: Vec<String>,
    names: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl Sscore {
    fn read(path: &Path) -> Result<Self, Box<dyn Error>> {
        let text = fs::read_to_string(path)?;
        let mut metadata = Vec::new();
        let mut names = Vec::new();
        let mut rows = Vec::new();
        for line in text.lines() {
            if let Some(header) = line.strip_prefix("#IID\t") {
                names = header.split('\t').map(str::to_string).collect();
            } else if line.starts_with('#') {
                metadata.push(line.to_string());
            } else {
                rows.push(line.split('\t').map(str::to_string).collect());
            }
        }
        Ok(Self {
            metadata,
            names,
            rows,
        })
    }

    fn column(&self, name: &str) -> Vec<&str> {
        let index = self
            .names
            .iter()
            .position(|n| n == name)
            .unwrap_or_else(|| panic!("no column {name} in {:?}", self.names));
        self.rows
            .iter()
            .map(|row| row[index + 1].as_str())
            .collect()
    }

    fn block_names(&self, score: &str, suffix: &str) -> Vec<String> {
        self.names
            .iter()
            .filter(|name| {
                name.starts_with(&format!("{score}_b"))
                    && name.ends_with(suffix)
                    && name.len() == score.len() + 6 + suffix.len()
            })
            .cloned()
            .collect()
    }
}

/// Runs the binary in `dir` on its staged inputs and returns the `.sscore` and sidecar paths.
fn score(
    dir: &Path,
    extra: &[&str],
) -> Result<(std::path::PathBuf, std::path::PathBuf), Box<dyn Error>> {
    let (genotypes, weights) = stage_inputs(dir)?;
    let mut args: Vec<&OsStr> = vec![weights.as_os_str(), genotypes.as_os_str()];
    args.extend(extra.iter().map(OsStr::new));
    let output = run(SCORE_BIN, dir, &args);
    assert_success(&output);
    Ok((dir.join("cohort_w.sscore"), dir.join("cohort_w.blocks.tsv")))
}

/// 1-based positions of the staged score file's variants.
fn score_positions(dir: &Path) -> Result<Vec<u64>, Box<dyn Error>> {
    Ok(fs::read_to_string(dir.join("w.tsv"))?
        .lines()
        .skip(1)
        .map(|line| {
            let id = line.split('\t').next().unwrap();
            id.split(':').nth(1).unwrap().parse().unwrap()
        })
        .collect())
}

#[test]
fn chromosome_blocks_leave_the_unsplit_columns_unchanged() -> TestResult {
    let tmp = tempdir()?;
    let (plain, _) = score(&tmp.path().join("plain"), &[])?;
    let (split, sidecar) = score(&tmp.path().join("split"), &["--blocks", "chrom"])?;
    let plain = Sscore::read(&plain)?;
    let split = Sscore::read(&split)?;

    assert_eq!(plain.names, ["w_AVG", "w_MISSING_PCT"]);
    assert_eq!(split.names.len(), 2 * 28, "{:?}", split.names);
    assert_eq!(
        &split.names[..4],
        [
            "w_AVG",
            "w_MISSING_PCT",
            "w_b0000_AVG",
            "w_b0000_MISSING_PCT"
        ]
    );
    assert_eq!(split.metadata, plain.metadata);
    assert_eq!(split.rows.len(), plain.rows.len());
    for name in ["w_AVG", "w_MISSING_PCT"] {
        assert_eq!(split.column(name), plain.column(name), "{name}");
    }
    // Every fixture variant is on chromosome 1: block b0001 is the whole score,
    // and every other block is empty (zero, fully missing).
    assert_eq!(split.column("w_b0001_AVG"), plain.column("w_AVG"));
    assert_eq!(
        split.column("w_b0001_MISSING_PCT"),
        plain.column("w_MISSING_PCT")
    );
    for name in split.block_names("w", "_AVG") {
        if name != "w_b0001_AVG" {
            assert!(split.column(&name).iter().all(|v| *v == "0.0"), "{name}");
        }
    }
    for name in split.block_names("w", "_MISSING_PCT") {
        if name != "w_b0001_MISSING_PCT" {
            assert!(split.column(&name).iter().all(|v| *v == "100.0"), "{name}");
        }
    }
    let sidecar = fs::read_to_string(sidecar)?;
    assert!(
        sidecar.contains("#BLOCK_ID\tCHROM\tSTART\tEND\tNAME\n"),
        "{sidecar}"
    );
    assert!(sidecar.contains("\nb0000\t.\t.\t.\toutside_every_block\n"));
    assert!(sidecar.contains("\nb0001\t1\t0\t.\tchr1\n"));
    assert!(sidecar.contains("\nb0023\tX\t0\t.\tchrX\n"));
    assert!(sidecar.ends_with("\nb0026\tMT\t0\t.\tchrMT\n"));
    Ok(())
}

#[test]
fn bed_block_partials_sum_to_the_total_exactly() -> TestResult {
    let tmp = tempdir()?;
    let dir = tmp.path().join("bed");
    fs::create_dir_all(&dir)?;
    // Unequal blocks; the third holds no variant (positions are multiples of 100),
    // and 1-based position 5000 lies on the boundary of the first two.
    let bed = dir.join("blocks.bed");
    fs::write(
        &bed,
        "# 0-based, half-open\nchr1\t0\t5000\thead\n1\t5000\t20000\tmiddle\n1\t30000\t30050\tempty\n",
    )?;
    let (plain, _) = score(&tmp.path().join("plain"), &["--emit-components"])?;
    let bed_arg = bed.to_string_lossy().into_owned();
    let (split, sidecar) = score(&dir, &["--emit-components", "--blocks", &bed_arg])?;
    let plain = Sscore::read(&plain)?;
    let split = Sscore::read(&split)?;

    let block_sums = split.block_names("w", "_SUM");
    assert_eq!(
        block_sums,
        ["w_b0000_SUM", "w_b0001_SUM", "w_b0002_SUM", "w_b0003_SUM"]
    );
    assert_eq!(split.column("w_SUM"), plain.column("w_SUM"));
    assert_eq!(split.column("w_MISSING_CT"), plain.column("w_MISSING_CT"));
    let totals: Vec<f64> = split
        .column("w_SUM")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    let mut summed = vec![0.0f64; totals.len()];
    for name in &block_sums {
        for (sum, value) in summed.iter_mut().zip(split.column(name)) {
            *sum += value.parse::<f64>().unwrap();
        }
    }
    assert_eq!(
        summed, totals,
        "block sums must equal the total bit for bit"
    );
    let missing_totals: Vec<u32> = split
        .column("w_MISSING_CT")
        .iter()
        .map(|v| v.parse().unwrap())
        .collect();
    let mut missing_summed = vec![0u32; missing_totals.len()];
    for name in split.block_names("w", "_MISSING_CT") {
        for (sum, value) in missing_summed.iter_mut().zip(split.column(&name)) {
            *sum += value.parse::<u32>().unwrap();
        }
    }
    assert_eq!(missing_summed, missing_totals);
    assert!(split.column("w_b0003_SUM").iter().all(|v| *v == "0.0"));
    assert!(split.column("w_b0003_MISSING_CT").iter().all(|v| *v == "0"));

    // Variant counts per block follow the half-open rule on 1-based positions.
    let positions = score_positions(&dir)?;
    let expected: BTreeMap<&str, usize> = BTreeMap::from([
        ("w", positions.len()),
        ("w_b0000", positions.iter().filter(|&&p| p > 20000).count()),
        ("w_b0001", positions.iter().filter(|&&p| p <= 5000).count()),
        (
            "w_b0002",
            positions
                .iter()
                .filter(|&&p| p > 5000 && p <= 20000)
                .count(),
        ),
        ("w_b0003", 0),
    ]);
    assert!(
        positions.contains(&5000),
        "the fixture must hold the boundary variant"
    );
    assert!(expected["w_b0001"] > 0 && expected["w_b0002"] > 0 && expected["w_b0000"] > 0);
    let counts: BTreeMap<&str, usize> = split
        .metadata
        .iter()
        .filter_map(|line| line.strip_prefix("#SCORE_VARIANT_COUNT\t"))
        .filter(|rest| !rest.starts_with("SCORE\t"))
        .map(|rest| {
            let (name, count) = rest.split_once('\t').unwrap();
            (name, count.parse().unwrap())
        })
        .collect();
    assert_eq!(counts, expected);

    let sidecar = fs::read_to_string(sidecar)?;
    assert!(sidecar.ends_with(
        "b0000\t.\t.\t.\toutside_every_block\nb0001\t1\t0\t5000\thead\nb0002\t1\t5000\t20000\tmiddle\nb0003\t1\t30000\t30050\tempty\n"
    ), "{sidecar}");
    Ok(())
}

#[test]
fn a_cached_block_plan_reproduces_the_run() -> TestResult {
    let tmp = tempdir()?;
    let cache = tmp.path().join("cache");
    let mut outputs = Vec::new();
    for round in ["first", "second"] {
        let dir = tmp.path().join(round);
        let (genotypes, weights) = stage_inputs(&dir)?;
        let output = Command::new(SCORE_BIN)
            .current_dir(&dir)
            .env("GNOMON_CACHE_DIR", &cache)
            .args([weights.as_os_str(), genotypes.as_os_str()])
            .args(["--blocks", "chrom"])
            .output()?;
        assert_success(&output);
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert_eq!(
            stderr.contains("Reusing content-verified compiled variant plan"),
            round == "second",
            "{stderr}"
        );
        outputs.push(fs::read(dir.join("cohort_w.sscore"))?);
    }
    assert_eq!(outputs[0], outputs[1]);
    Ok(())
}

#[test]
fn blocks_refuse_bad_partitions_and_existing_sidecars() -> TestResult {
    let tmp = tempdir()?;
    let (genotypes, weights) = stage_inputs(tmp.path())?;
    let failing = |args: &[&str]| {
        let mut all: Vec<&OsStr> = vec![weights.as_os_str(), genotypes.as_os_str()];
        all.extend(args.iter().map(OsStr::new));
        let output = run(SCORE_BIN, tmp.path(), &all);
        assert!(!output.status.success(), "{args:?} must fail");
        assert!(
            !tmp.path().join("cohort_w.sscore").exists(),
            "{args:?} wrote scores"
        );
        String::from_utf8_lossy(&output.stderr).into_owned()
    };
    let overlapping = tmp.path().join("overlap.bed");
    fs::write(&overlapping, "1\t0\t100\n1\t50\t150\n")?;
    let overlap = failing(&["--blocks", &overlapping.to_string_lossy()]);
    assert!(overlap.contains("overlap"), "{overlap}");
    let missing = failing(&["--blocks", "no-such.bed"]);
    assert!(missing.contains("no-such.bed"), "{missing}");
    assert!(failing(&["--blocks-max", "3"]).contains("--blocks"));
    fs::write(tmp.path().join("cohort_w.blocks.tsv"), "stale\n")?;
    let taken = failing(&["--blocks", "chrom"]);
    assert!(taken.contains("already exists"), "{taken}");
    assert_eq!(
        fs::read_to_string(tmp.path().join("cohort_w.blocks.tsv"))?,
        "stale\n"
    );
    Ok(())
}
