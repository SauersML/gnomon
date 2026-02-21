use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::tempdir;

const PLINK_MAGIC_HEADER: [u8; 3] = [0x6c, 0x1b, 0x01];
const CPU_FALLBACK_THRESHOLD: usize = 100_000;
// GPU/CPU reductions may differ in operation order, so tiny FP drift is expected.
const SCORE_MATCH_ABS_EPSILON: f64 = 1.0e-6;
const SCORE_MATCH_REL_EPSILON: f64 = 1.0e-6;
// Require nontrivial per-score missingness variation without overfitting fixture specifics.
const MIN_MISSINGNESS_SPREAD_PCT: f64 = 0.5;

#[derive(Copy, Clone)]
enum Genotype {
    Dosage0,
    Dosage1,
    Dosage2,
    Missing,
}

struct ScoreRunOutput {
    stderr: String,
    output_path: PathBuf,
}

#[test]
fn cpu_fallback_handles_messy_inputs_and_writes_scores() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let prefix = tmp.path().join("messy_cohort");
    let n_people = 64usize;
    let n_variants = 48usize;

    write_test_plink_files(&prefix, n_people, n_variants)?;

    let score_dir = tmp.path().join("score_dir");
    fs::create_dir_all(&score_dir)?;
    let score_path = score_dir.join("messy_small.tsv");
    write_native_score_file(
        &score_path,
        n_variants,
        8,
        "S",
        true, // include malformed + comment lines
    )?;

    let run = run_score(&score_path, &prefix, tmp.path())?;
    assert!(
        run.stderr
            .contains("Backend: CPU fallback (problem size below CUDA threshold)"),
        "expected CPU fallback for small workload, stderr:\n{}",
        run.stderr
    );
    assert!(
        run.output_path.exists(),
        "expected output file at {}",
        run.output_path.display()
    );

    let table = parse_sscore_table(&run.output_path)?;
    assert_eq!(table.rows.len(), n_people, "row count mismatch");
    assert_eq!(table.score_names.len(), 8, "score count mismatch");
    Ok(())
}

#[test]
fn gpu_and_cpu_outputs_match_for_shared_scores_when_cuda_available() -> Result<(), Box<dyn Error>> {
    if !cuda_driver_present() {
        eprintln!(
            "Skipping gpu_and_cpu_outputs_match_for_shared_scores_when_cuda_available: no CUDA driver detected"
        );
        return Ok(());
    }

    let tmp = tempdir()?;
    let prefix = tmp.path().join("gpu_compare_cohort");
    let n_people = 512usize;
    let n_variants = 256usize;

    write_test_plink_files(&prefix, n_people, n_variants)?;

    let small_scores = 16usize;
    let large_scores = 256usize;
    assert!(n_people * small_scores < CPU_FALLBACK_THRESHOLD);
    assert!(n_people * large_scores >= CPU_FALLBACK_THRESHOLD);

    let cpu_score_path = tmp.path().join("scores_small.tsv");
    let gpu_score_path = tmp.path().join("scores_large.tsv");

    write_native_score_file(&cpu_score_path, n_variants, small_scores, "S", true)?;
    write_native_score_file(&gpu_score_path, n_variants, large_scores, "S", true)?;

    let cpu_run = run_score(&cpu_score_path, &prefix, tmp.path())?;
    assert!(
        cpu_run
            .stderr
            .contains("Backend: CPU fallback (problem size below CUDA threshold)"),
        "expected CPU fallback baseline run, stderr:\n{}",
        cpu_run.stderr
    );
    let cpu_table = parse_sscore_table(&cpu_run.output_path)?;

    let gpu_candidate_run = run_score(&gpu_score_path, &prefix, tmp.path())?;
    assert_backend_selected(&gpu_candidate_run.stderr);

    let gpu_table = parse_sscore_table(&gpu_candidate_run.output_path)?;
    assert_eq!(cpu_table.rows.len(), gpu_table.rows.len(), "row count mismatch");

    for score_idx in 0..small_scores {
        let score_name = format!("S{:03}", score_idx);
        for row_idx in 0..cpu_table.rows.len() {
            let cpu_row = &cpu_table.rows[row_idx];
            let gpu_row = &gpu_table.rows[row_idx];
            assert_eq!(
                cpu_row.iid, gpu_row.iid,
                "IID mismatch at row {row_idx} for {score_name}"
            );

            let cpu_val = *cpu_row
                .avg
                .get(&score_name)
                .expect("missing score in CPU output");
            let gpu_val = *gpu_row
                .avg
                .get(&score_name)
                .expect("missing score in GPU output");
            let abs = (cpu_val - gpu_val).abs();
            let rel = abs / cpu_val.abs().max(gpu_val.abs()).max(1.0);
            assert!(
                abs <= SCORE_MATCH_ABS_EPSILON || rel <= SCORE_MATCH_REL_EPSILON,
                "score mismatch for IID={} score={} cpu={} gpu={} abs={} rel={}",
                cpu_row.iid,
                score_name,
                cpu_val,
                gpu_val,
                abs,
                rel
            );

            let cpu_missing = *cpu_row
                .missing_pct
                .get(&score_name)
                .expect("missing pct missing in CPU output");
            let gpu_missing = *gpu_row
                .missing_pct
                .get(&score_name)
                .expect("missing pct missing in GPU output");
            assert!(
                (cpu_missing - gpu_missing).abs() <= 1e-6,
                "missing pct mismatch for IID={} score={} cpu={} gpu={}",
                cpu_row.iid,
                score_name,
                cpu_missing,
                gpu_missing
            );
        }
    }

    Ok(())
}

#[test]
fn gpu_and_cpu_outputs_match_for_multifile_score_directory() -> Result<(), Box<dyn Error>> {
    if !cuda_driver_present() {
        eprintln!(
            "Skipping gpu_and_cpu_outputs_match_for_multifile_score_directory: no CUDA driver detected"
        );
        return Ok(());
    }

    let tmp = tempdir()?;
    let prefix = tmp.path().join("gpu_multifile_cohort");
    let n_people = 768usize;
    let n_variants = 320usize;
    write_test_plink_files(&prefix, n_people, n_variants)?;

    let common_scores = 24usize;
    let extra_scores = 160usize;
    assert!(n_people * common_scores < CPU_FALLBACK_THRESHOLD);
    assert!(n_people * (common_scores + extra_scores) >= CPU_FALLBACK_THRESHOLD);

    let cpu_score_path = tmp.path().join("common_cpu.tsv");
    write_native_score_file(&cpu_score_path, n_variants, common_scores, "A", true)?;
    let cpu_run = run_score(&cpu_score_path, &prefix, tmp.path())?;
    assert!(
        cpu_run
            .stderr
            .contains("Backend: CPU fallback (problem size below CUDA threshold)"),
        "expected CPU fallback baseline run, stderr:\n{}",
        cpu_run.stderr
    );
    let cpu_table = parse_sscore_table(&cpu_run.output_path)?;

    let score_dir = tmp.path().join("score_bundle");
    fs::create_dir_all(&score_dir)?;
    let common_gpu = score_dir.join("common_gpu.tsv");
    let extra_gpu = score_dir.join("extra_gpu.tsv");
    write_native_score_file(&common_gpu, n_variants, common_scores, "A", true)?;
    write_native_score_file(&extra_gpu, n_variants, extra_scores, "B", true)?;

    let gpu_candidate_run = run_score(&score_dir, &prefix, tmp.path())?;
    assert_backend_selected(&gpu_candidate_run.stderr);

    let gpu_table = parse_sscore_table(&gpu_candidate_run.output_path)?;
    assert_eq!(cpu_table.rows.len(), gpu_table.rows.len(), "row count mismatch");

    for score_idx in 0..common_scores {
        let score_name = format!("A{score_idx:03}");
        for row_idx in 0..cpu_table.rows.len() {
            let cpu_row = &cpu_table.rows[row_idx];
            let gpu_row = &gpu_table.rows[row_idx];
            assert_eq!(
                cpu_row.iid, gpu_row.iid,
                "IID mismatch at row {row_idx} for {score_name}"
            );

            let cpu_val = *cpu_row
                .avg
                .get(&score_name)
                .expect("missing score in CPU output");
            let gpu_val = *gpu_row
                .avg
                .get(&score_name)
                .expect("missing score in GPU output");
            let abs = (cpu_val - gpu_val).abs();
            let rel = abs / cpu_val.abs().max(gpu_val.abs()).max(1.0);
            assert!(
                abs <= SCORE_MATCH_ABS_EPSILON || rel <= SCORE_MATCH_REL_EPSILON,
                "score mismatch for IID={} score={} cpu={} gpu={} abs={} rel={}",
                cpu_row.iid,
                score_name,
                cpu_val,
                gpu_val,
                abs,
                rel
            );

            let cpu_missing = *cpu_row
                .missing_pct
                .get(&score_name)
                .expect("missing pct missing in CPU output");
            let gpu_missing = *gpu_row
                .missing_pct
                .get(&score_name)
                .expect("missing pct missing in GPU output");
            assert!(
                (cpu_missing - gpu_missing).abs() <= 1e-6,
                "missing pct mismatch for IID={} score={} cpu={} gpu={}",
                cpu_row.iid,
                score_name,
                cpu_missing,
                gpu_missing
            );
        }
    }

    Ok(())
}

#[test]
fn five_samples_multichrom_partial_overlap_and_split_sites() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let prefix = tmp.path().join("five_sample_multichrom");
    write_custom_plink_files_multichrom_split_sites(&prefix)?;

    let score_path = tmp.path().join("partial_overlap_split_sites.tsv");
    write_partial_overlap_split_score_file(&score_path)?;

    let run = run_score(&score_path, &prefix, tmp.path())?;
    assert!(
        run.stderr
            .contains("Backend: CPU fallback (problem size below CUDA threshold)"),
        "expected CPU fallback for tiny workload, stderr:\n{}",
        run.stderr
    );
    assert!(
        run.output_path.exists(),
        "expected output file at {}",
        run.output_path.display()
    );

    let table = parse_sscore_table(&run.output_path)?;
    assert_eq!(table.rows.len(), 5, "expected exactly 5 samples");
    assert_eq!(table.score_names, vec!["S000".to_string(), "S001".to_string()]);

    let mut observed_missing = false;
    for row in &table.rows {
        for score_name in &table.score_names {
            let pct = row
                .missing_pct
                .get(score_name)
                .copied()
                .ok_or("missing missing_pct value")?;
            if pct > 0.0 {
                observed_missing = true;
            }
        }
    }
    assert!(observed_missing, "expected at least one non-zero missing percentage");

    Ok(())
}

#[test]
fn forty_genomes_hundred_scores_microarray_density_with_multiallelic() -> Result<(), Box<dyn Error>>
{
    if !cuda_driver_present() {
        eprintln!(
            "Skipping forty_genomes_hundred_scores_microarray_density_with_multiallelic: no CUDA driver detected"
        );
        return Ok(());
    }

    let tmp = tempdir()?;
    let prefix = tmp.path().join("forty_people_microarray");
    let loci = write_microarray_like_plink_files(&prefix, 40)?;
    let mut saw_split_site = false;
    for pair in loci.windows(2) {
        if pair[0].chrom == pair[1].chrom
            && pair[0].pos == pair[1].pos
            && pair[0].other_allele != pair[1].other_allele
        {
            saw_split_site = true;
            break;
        }
    }
    assert!(saw_split_site, "expected multiallelic/split-site loci in fixture");

    let score_dir = tmp.path().join("score_100_bundle");
    fs::create_dir_all(&score_dir)?;
    write_hundred_partial_overlap_scores(&score_dir, &loci, 100)?;

    let run = run_score(&score_dir, &prefix, tmp.path())?;
    assert!(
        run.stderr
            .contains("Backend: CPU fallback (problem size below CUDA threshold)"),
        "expected CPU fallback for 40x100 workload, stderr:\n{}",
        run.stderr
    );
    assert!(
        run.output_path.exists(),
        "expected output file at {}",
        run.output_path.display()
    );

    let table = parse_sscore_table(&run.output_path)?;
    assert_eq!(table.rows.len(), 40, "expected 40 genomes");
    assert_eq!(table.score_names.len(), 100, "expected 100 scores");

    let mut per_score_mean_missing = Vec::with_capacity(table.score_names.len());
    for score in &table.score_names {
        let mut sum_missing = 0.0f64;
        for row in &table.rows {
            let pct = row
                .missing_pct
                .get(score)
                .copied()
                .ok_or("missing missing_pct value")?;
            sum_missing += pct;
        }
        per_score_mean_missing.push(sum_missing / table.rows.len() as f64);
    }

    let min_missing = per_score_mean_missing
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_missing = per_score_mean_missing
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_missing - min_missing >= MIN_MISSINGNESS_SPREAD_PCT,
        "expected varying score missingness, got min={min_missing:.3} max={max_missing:.3}"
    );

    let mut found_row_with_missing = false;
    for row in &table.rows {
        if row.missing_pct.values().any(|&v| v > 0.0) {
            found_row_with_missing = true;
            break;
        }
    }
    assert!(
        found_row_with_missing,
        "expected at least one genome with missing sites across scores"
    );

    Ok(())
}

#[test]
fn fifty_thousand_samples_small_genome_varied_variants() -> Result<(), Box<dyn Error>> {
    let tmp = tempdir()?;
    let prefix = tmp.path().join("cohort_50k_small_genome");
    let loci = write_large_cohort_plink_files(&prefix, 50_000)?;

    let score_path = tmp.path().join("large_50k_small_panel.tsv");
    write_large_cohort_score_file(&score_path, &loci)?;

    let run = run_score(&score_path, &prefix, tmp.path())?;
    assert!(
        run.stderr
            .contains("Backend: CPU fallback (problem size below CUDA threshold)"),
        "expected CPU fallback for 50k x 1 score workload, stderr:\n{}",
        run.stderr
    );
    assert!(
        run.output_path.exists(),
        "expected output file at {}",
        run.output_path.display()
    );

    let table = parse_sscore_table(&run.output_path)?;
    assert_eq!(table.rows.len(), 50_000, "expected 50,000 samples");
    assert_eq!(table.score_names, vec!["L000".to_string()]);

    let mut saw_missing = false;
    for row in &table.rows {
        let miss = row
            .missing_pct
            .get("L000")
            .copied()
            .ok_or("missing missing_pct for L000")?;
        if miss > 0.0 {
            saw_missing = true;
            break;
        }
    }
    assert!(saw_missing, "expected at least one sample with missing variants");

    Ok(())
}

fn assert_backend_selected(stderr: &str) {
    let selected_cuda = stderr.contains("> Backend: CUDA");
    let selected_cpu_fallback = stderr.contains("> Backend: CPU fallback");
    assert!(
        selected_cuda || selected_cpu_fallback,
        "expected backend selection log line, stderr:\n{stderr}"
    );
}

fn cuda_driver_present() -> bool {
    if std::path::Path::new("/dev/nvidiactl").exists() {
        return true;
    }
    let probe = Command::new("nvidia-smi")
        .arg("-L")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    probe
}

fn run_score(score_path: &Path, genotype_prefix: &Path, cwd: &Path) -> Result<ScoreRunOutput, Box<dyn Error>> {
    let exe = env!("CARGO_BIN_EXE_gnomon");
    let output = Command::new(exe)
        .current_dir(cwd)
        .arg("score")
        .arg(score_path)
        .arg(genotype_prefix)
        .output()?;

    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "gnomon score failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            stdout,
            stderr
        )
        .into());
    }

    let score_stem = score_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or("invalid score file stem")?;
    let cohort_stem = genotype_prefix
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or("invalid genotype prefix file name")?;
    let output_path = genotype_prefix
        .parent()
        .ok_or("missing genotype parent")?
        .join(format!("{cohort_stem}_{score_stem}.sscore"));

    Ok(ScoreRunOutput {
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        output_path,
    })
}

fn write_native_score_file(
    path: &Path,
    n_variants: usize,
    n_scores: usize,
    score_prefix: &str,
    include_messy_lines: bool,
) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    out.push_str("##synthetic=test\n");
    out.push_str("variant_id\teffect_allele\tother_allele");
    for s in 0..n_scores {
        out.push('\t');
        out.push_str(&format!("{score_prefix}{s:03}"));
    }
    out.push('\n');

    if include_messy_lines {
        out.push_str("#ignored metadata line\n");
        out.push_str("\n");
    }

    for v in 0..n_variants {
        let pos = 1_000u32 + v as u32;
        let variant_id = format!("1:{pos}");
        let (effect, other) = if v % 5 == 0 { ("A", "G") } else { ("G", "A") };
        out.push_str(&variant_id);
        out.push('\t');
        out.push_str(effect);
        out.push('\t');
        out.push_str(other);
        for s in 0..n_scores {
            let base = ((v * 31 + s * 17 + 7) % 29) as f64;
            let weight = (base - 14.0) / 10.0;
            out.push('\t');
            out.push_str(&format!("{weight:.8}"));
        }
        out.push('\n');

        if include_messy_lines && v % 37 == 0 {
            out.push_str("1:999999\tA\n");
        }
    }

    fs::write(path, out)?;
    Ok(())
}

fn write_test_plink_files(prefix: &Path, n_people: usize, n_variants: usize) -> Result<(), Box<dyn Error>> {
    let fam_path = prefix.with_extension("fam");
    let bim_path = prefix.with_extension("bim");
    let bed_path = prefix.with_extension("bed");

    let mut fam = String::new();
    for i in 0..n_people {
        fam.push_str(&format!("F{:06}\tI{:06}\t0\t0\t0\t-9\n", i, i));
    }
    fs::write(fam_path, fam)?;

    let mut bim = String::new();
    for v in 0..n_variants {
        let pos = 1_000u32 + v as u32;
        bim.push_str(&format!("1\t1:{pos}\t0\t{pos}\tA\tG\n"));
    }
    fs::write(bim_path, bim)?;

    let bytes_per_variant = n_people.div_ceil(4);
    let mut bed = Vec::with_capacity(PLINK_MAGIC_HEADER.len() + bytes_per_variant * n_variants);
    bed.extend_from_slice(&PLINK_MAGIC_HEADER);

    for v in 0..n_variants {
        let mut row = vec![0u8; bytes_per_variant];
        for p in 0..n_people {
            let gt = synthetic_genotype(p, v);
            let bits = plink_2bit_code(gt);
            let byte_idx = p / 4;
            let shift = (p % 4) * 2;
            row[byte_idx] |= bits << shift;
        }
        bed.extend_from_slice(&row);
    }
    fs::write(bed_path, bed)?;
    Ok(())
}

fn synthetic_genotype(person_idx: usize, variant_idx: usize) -> Genotype {
    let key = (person_idx * 131 + variant_idx * 17 + 23) % 31;
    match key {
        0 | 1 => Genotype::Missing,
        2..=9 => Genotype::Dosage0,
        10..=20 => Genotype::Dosage1,
        _ => Genotype::Dosage2,
    }
}

fn plink_2bit_code(gt: Genotype) -> u8 {
    match gt {
        Genotype::Dosage0 => 0b00,
        Genotype::Missing => 0b01,
        Genotype::Dosage1 => 0b10,
        Genotype::Dosage2 => 0b11,
    }
}

fn write_custom_plink_files_multichrom_split_sites(prefix: &Path) -> Result<(), Box<dyn Error>> {
    let fam_path = prefix.with_extension("fam");
    let bim_path = prefix.with_extension("bim");
    let bed_path = prefix.with_extension("bed");

    let fam = [
        "F000001\tI000001\t0\t0\t1\t-9",
        "F000002\tI000002\t0\t0\t2\t-9",
        "F000003\tI000003\t0\t0\t1\t-9",
        "F000004\tI000004\t0\t0\t2\t-9",
        "F000005\tI000005\t0\t0\t1\t-9",
    ]
    .join("\n")
        + "\n";
    fs::write(fam_path, fam)?;

    let variants: Vec<(&str, u32, &str, &str, [Genotype; 5])> = vec![
        ("1", 1000, "A", "G", [Genotype::Dosage0, Genotype::Dosage1, Genotype::Dosage2, Genotype::Missing, Genotype::Dosage1]),
        // Split/multiallelic-style same locus in BIM with different allele pair.
        ("1", 1000, "A", "T", [Genotype::Dosage1, Genotype::Dosage0, Genotype::Missing, Genotype::Dosage2, Genotype::Dosage1]),
        ("2", 2000, "C", "T", [Genotype::Dosage2, Genotype::Dosage1, Genotype::Dosage0, Genotype::Dosage1, Genotype::Missing]),
        ("2", 2001, "G", "A", [Genotype::Dosage0, Genotype::Dosage0, Genotype::Dosage1, Genotype::Dosage1, Genotype::Dosage2]),
        ("3", 3000, "T", "C", [Genotype::Dosage1, Genotype::Dosage2, Genotype::Dosage1, Genotype::Missing, Genotype::Dosage0]),
        // Another split site at same chr:pos.
        ("3", 3000, "T", "G", [Genotype::Dosage2, Genotype::Dosage1, Genotype::Missing, Genotype::Dosage0, Genotype::Dosage1]),
        ("4", 4000, "A", "C", [Genotype::Dosage0, Genotype::Missing, Genotype::Dosage0, Genotype::Dosage1, Genotype::Dosage2]),
        ("5", 5000, "G", "T", [Genotype::Dosage1, Genotype::Dosage1, Genotype::Dosage1, Genotype::Dosage1, Genotype::Dosage1]),
        ("5", 5001, "C", "A", [Genotype::Missing, Genotype::Dosage2, Genotype::Dosage0, Genotype::Dosage1, Genotype::Dosage2]),
    ];

    let mut bim = String::new();
    for (chrom, pos, a1, a2, _) in &variants {
        let vid = format!("{chrom}:{pos}");
        bim.push_str(&format!("{chrom}\t{vid}\t0\t{pos}\t{a1}\t{a2}\n"));
    }
    fs::write(bim_path, bim)?;

    let n_people = 5usize;
    let bytes_per_variant = n_people.div_ceil(4);
    let mut bed = Vec::with_capacity(PLINK_MAGIC_HEADER.len() + bytes_per_variant * variants.len());
    bed.extend_from_slice(&PLINK_MAGIC_HEADER);
    for (_, _, _, _, gts) in &variants {
        let mut row = vec![0u8; bytes_per_variant];
        for (person_idx, gt) in gts.iter().enumerate() {
            let bits = plink_2bit_code(*gt);
            let byte_idx = person_idx / 4;
            let shift = (person_idx % 4) * 2;
            row[byte_idx] |= bits << shift;
        }
        bed.extend_from_slice(&row);
    }
    fs::write(bed_path, bed)?;

    Ok(())
}

fn write_partial_overlap_split_score_file(path: &Path) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    out.push_str("##synthetic=partial_overlap_split_sites\n");
    out.push_str("variant_id\teffect_allele\tother_allele\tS000\tS001\n");
    // Overlap + split site alleles at 1:1000
    out.push_str("1:1000\tA\tG\t0.20000000\t-0.10000000\n");
    out.push_str("1:1000\tA\tT\t-0.30000000\t0.00000000\n");
    // Overlap
    out.push_str("2:2000\tT\tC\t0.50000000\t0.25000000\n");
    // Overlap + split site alleles at 3:3000
    out.push_str("3:3000\tT\tC\t0.70000000\t-0.40000000\n");
    out.push_str("3:3000\tT\tG\t-0.20000000\t0.30000000\n");
    // Non-overlap rows intentionally included (partial overlap scenario)
    out.push_str("4:4999\tA\tC\t1.00000000\t1.00000000\n");
    // Overlap
    out.push_str("5:5001\tC\tA\t-0.80000000\t0.10000000\n");
    // Non-overlap
    out.push_str("5:9999\tG\tA\t0.40000000\t-0.20000000\n");
    // Malformed line to exercise messy input handling
    out.push_str("1:123456\tA\n");
    fs::write(path, out)?;
    Ok(())
}

#[derive(Clone)]
struct LocusSpec {
    chrom: &'static str,
    pos: u32,
    effect_allele: &'static str,
    other_allele: &'static str,
}

fn write_microarray_like_plink_files(
    prefix: &Path,
    n_people: usize,
) -> Result<Vec<LocusSpec>, Box<dyn Error>> {
    let fam_path = prefix.with_extension("fam");
    let bim_path = prefix.with_extension("bim");
    let bed_path = prefix.with_extension("bed");

    let mut fam = String::new();
    for i in 0..n_people {
        fam.push_str(&format!("F{i:06}\tI{i:06}\t0\t0\t{}\t-9\n", 1 + (i % 2)));
    }
    fs::write(fam_path, fam)?;

    let alleles = [
        ("A", "C"),
        ("A", "G"),
        ("A", "T"),
        ("C", "G"),
        ("C", "T"),
        ("G", "T"),
    ];
    let alt_for_multiallelic = ["G", "T", "C", "A", "G"];

    let mut loci = Vec::new();
    for chrom_idx in 0..5usize {
        let chrom = match chrom_idx {
            0 => "1",
            1 => "2",
            2 => "3",
            3 => "4",
            _ => "5",
        };
        for local_idx in 0..125usize {
            let pos = 100_000u32 + (local_idx as u32) * 800u32;
            let pair = alleles[(chrom_idx * 17 + local_idx * 13) % alleles.len()];
            loci.push(LocusSpec {
                chrom,
                pos,
                effect_allele: pair.0,
                other_allele: pair.1,
            });

            // Split/multiallelic representation: same chr:pos with alternate second allele.
            if local_idx % 31 == 0 {
                let mut alt =
                    alt_for_multiallelic[(chrom_idx + local_idx) % alt_for_multiallelic.len()];
                if alt == pair.0 || alt == pair.1 {
                    alt = ["A", "C", "G", "T"]
                        .into_iter()
                        .find(|candidate| *candidate != pair.0 && *candidate != pair.1)
                        .expect("expected at least one alternate allele distinct from pair");
                }
                loci.push(LocusSpec {
                    chrom,
                    pos,
                    effect_allele: pair.0,
                    other_allele: alt,
                });
            }
        }
    }

    let mut bim = String::new();
    for locus in &loci {
        let vid = format!("{}:{}", locus.chrom, locus.pos);
        bim.push_str(&format!(
            "{}\t{}\t0\t{}\t{}\t{}\n",
            locus.chrom, vid, locus.pos, locus.effect_allele, locus.other_allele
        ));
    }
    fs::write(bim_path, bim)?;

    let bytes_per_variant = n_people.div_ceil(4);
    let mut bed = Vec::with_capacity(PLINK_MAGIC_HEADER.len() + bytes_per_variant * loci.len());
    bed.extend_from_slice(&PLINK_MAGIC_HEADER);

    // Build split-site lookup so both rows at the same chr:pos are derived
    // from one underlying multiallelic genotype model per person/locus.
    let mut split_groups: HashMap<(&'static str, u32, &'static str), Vec<(usize, &'static str)>> =
        HashMap::new();
    for (idx, locus) in loci.iter().enumerate() {
        split_groups
            .entry((locus.chrom, locus.pos, locus.effect_allele))
            .or_default()
            .push((idx, locus.other_allele));
    }
    let mut split_alt_context = vec![None; loci.len()];
    for rows in split_groups.values() {
        if rows.len() != 2 {
            continue;
        }
        let (idx_a, alt_a) = rows[0];
        let (idx_b, alt_b) = rows[1];
        if alt_a == alt_b {
            continue;
        }
        split_alt_context[idx_a] = Some((alt_a, alt_b));
        split_alt_context[idx_b] = Some((alt_b, alt_a));
    }

    for (variant_idx, locus) in loci.iter().enumerate() {
        let mut row = vec![0u8; bytes_per_variant];
        for person_idx in 0..n_people {
            let gt = if let Some((row_alt, other_alt)) = split_alt_context[variant_idx] {
                synthetic_genotype_for_split_site(person_idx, locus, row_alt, other_alt)
            } else {
                synthetic_genotype_for_locus(person_idx, variant_idx, locus)
            };
            let bits = plink_2bit_code(gt);
            let byte_idx = person_idx / 4;
            let shift = (person_idx % 4) * 2;
            row[byte_idx] |= bits << shift;
        }
        bed.extend_from_slice(&row);
    }
    fs::write(bed_path, bed)?;

    Ok(loci)
}

fn synthetic_genotype_for_locus(person_idx: usize, variant_idx: usize, locus: &LocusSpec) -> Genotype {
    let chrom_val = locus
        .chrom
        .as_bytes()
        .first()
        .copied()
        .unwrap_or(b'1') as usize;
    let seed = person_idx * 197 + variant_idx * 61 + chrom_val * 11 + locus.pos as usize;
    let selector = seed % 101;

    // Person- and locus-specific missingness to emulate real microarray dropouts.
    if selector < 8 || (person_idx + variant_idx) % 53 == 0 {
        return Genotype::Missing;
    }

    match selector % 9 {
        0 | 1 | 2 => Genotype::Dosage0,
        3 | 4 | 5 => Genotype::Dosage1,
        _ => Genotype::Dosage2,
    }
}

fn synthetic_genotype_for_split_site(
    person_idx: usize,
    locus: &LocusSpec,
    row_alt: &str,
    other_alt: &str,
) -> Genotype {
    let chrom_val = locus
        .chrom
        .as_bytes()
        .first()
        .copied()
        .unwrap_or(b'1') as usize;
    let seed = person_idx * 211 + chrom_val * 17 + locus.pos as usize;
    let selector = seed % 100;

    let (alt1, alt2) = if row_alt <= other_alt {
        (row_alt, other_alt)
    } else {
        (other_alt, row_alt)
    };

    enum MultiState {
        RefRef,
        RefAlt1,
        RefAlt2,
        Alt1Alt1,
        Alt2Alt2,
        Alt1Alt2,
        Missing,
    }

    let state = match selector {
        0..=34 => MultiState::RefRef,
        35..=51 => MultiState::RefAlt1,
        52..=68 => MultiState::RefAlt2,
        69..=79 => MultiState::Alt1Alt1,
        80..=90 => MultiState::Alt2Alt2,
        91..=94 => MultiState::Alt1Alt2,
        _ => MultiState::Missing,
    };

    match state {
        MultiState::RefRef => Genotype::Dosage0,
        MultiState::RefAlt1 => {
            if row_alt == alt1 {
                Genotype::Dosage1
            } else {
                Genotype::Missing
            }
        }
        MultiState::RefAlt2 => {
            if row_alt == alt2 {
                Genotype::Dosage1
            } else {
                Genotype::Missing
            }
        }
        MultiState::Alt1Alt1 => {
            if row_alt == alt1 {
                Genotype::Dosage2
            } else {
                Genotype::Missing
            }
        }
        MultiState::Alt2Alt2 => {
            if row_alt == alt2 {
                Genotype::Dosage2
            } else {
                Genotype::Missing
            }
        }
        // ALT1/ALT2 is physically possible for triallelic loci but not
        // representable in a single split biallelic row without ambiguity.
        MultiState::Alt1Alt2 | MultiState::Missing => Genotype::Missing,
    }
}

fn write_hundred_partial_overlap_scores(
    dir: &Path,
    loci: &[LocusSpec],
    n_scores: usize,
) -> Result<(), Box<dyn Error>> {
    for score_idx in 0..n_scores {
        let path = dir.join(format!("score_{score_idx:03}.tsv"));
        let score_name = format!("Q{score_idx:03}");

        let mut out = String::new();
        out.push_str("##synthetic=forty_genomes_hundred_scores\n");
        out.push_str(&format!(
            "variant_id\teffect_allele\tother_allele\t{score_name}\n"
        ));
        out.push_str("#per-score varying overlap and missingness profile\n");

        // Coverage varies by score from sparse to dense.
        let coverage_bucket = score_idx % 10; // 0..9
        for (locus_idx, locus) in loci.iter().enumerate() {
            let include_hash = (locus_idx * 37 + score_idx * 19 + 11) % 100;
            let include_threshold = 20 + coverage_bucket * 7; // 20%..83%
            if include_hash >= include_threshold {
                continue;
            }

            let variant_id = format!("{}:{}", locus.chrom, locus.pos);
            let weight_raw = ((locus_idx * 23 + score_idx * 29 + 3) % 41) as f64;
            let weight = (weight_raw - 20.0) / 11.0;
            out.push_str(&format!(
                "{variant_id}\t{}\t{}\t{weight:.8}\n",
                locus.effect_allele, locus.other_allele
            ));
        }

        // Keep guaranteed overlap so no score file is empty.
        let anchor = &loci[score_idx % loci.len()];
        out.push_str(&format!(
            "{}:{}\t{}\t{}\t{:.8}\n",
            anchor.chrom,
            anchor.pos,
            anchor.effect_allele,
            anchor.other_allele,
            0.12345678f64
        ));

        // Non-overlap and malformed rows for messy-data handling.
        out.push_str("9:999999\tA\tG\t0.50000000\n");
        if score_idx % 9 == 0 {
            out.push_str("1:123456\tA\n");
        }

        fs::write(path, out)?;
    }
    Ok(())
}

fn write_large_cohort_plink_files(
    prefix: &Path,
    n_people: usize,
) -> Result<Vec<LocusSpec>, Box<dyn Error>> {
    let fam_path = prefix.with_extension("fam");
    let bim_path = prefix.with_extension("bim");
    let bed_path = prefix.with_extension("bed");

    let mut fam = String::new();
    for i in 0..n_people {
        fam.push_str(&format!("F{i:06}\tI{i:06}\t0\t0\t{}\t-9\n", 1 + (i % 2)));
    }
    fs::write(fam_path, fam)?;

    let base_pairs = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T"), ("A", "T")];
    let mut loci = Vec::new();
    for chrom_idx in 0..4usize {
        let chrom = match chrom_idx {
            0 => "1",
            1 => "2",
            2 => "3",
            _ => "5",
        };
        for local in 0..24usize {
            let pos = 150_000u32 + (local as u32) * 2_500u32;
            let pair = base_pairs[(chrom_idx * 11 + local * 7) % base_pairs.len()];
            loci.push(LocusSpec {
                chrom,
                pos,
                effect_allele: pair.0,
                other_allele: pair.1,
            });
            // periodic split-site representation at same chr:pos
            if local % 9 == 0 {
                let alt = if pair.1 == "T" { "G" } else { "T" };
                loci.push(LocusSpec {
                    chrom,
                    pos,
                    effect_allele: pair.0,
                    other_allele: alt,
                });
            }
        }
    }

    let mut bim = String::new();
    for locus in &loci {
        let vid = format!("{}:{}", locus.chrom, locus.pos);
        bim.push_str(&format!(
            "{}\t{}\t0\t{}\t{}\t{}\n",
            locus.chrom, vid, locus.pos, locus.effect_allele, locus.other_allele
        ));
    }
    fs::write(bim_path, bim)?;

    let bytes_per_variant = n_people.div_ceil(4);
    let mut bed = Vec::with_capacity(PLINK_MAGIC_HEADER.len() + bytes_per_variant * loci.len());
    bed.extend_from_slice(&PLINK_MAGIC_HEADER);
    for (variant_idx, locus) in loci.iter().enumerate() {
        let mut row = vec![0u8; bytes_per_variant];
        for person_idx in 0..n_people {
            let gt = synthetic_genotype_for_locus(person_idx, variant_idx, locus);
            let bits = plink_2bit_code(gt);
            let byte_idx = person_idx / 4;
            let shift = (person_idx % 4) * 2;
            row[byte_idx] |= bits << shift;
        }
        bed.extend_from_slice(&row);
    }
    fs::write(bed_path, bed)?;

    Ok(loci)
}

fn write_large_cohort_score_file(path: &Path, loci: &[LocusSpec]) -> Result<(), Box<dyn Error>> {
    let mut out = String::new();
    out.push_str("##synthetic=large_cohort_50k\n");
    out.push_str("variant_id\teffect_allele\tother_allele\tL000\n");
    out.push_str("#partial overlap, with non-overlap and malformed rows\n");

    for (idx, locus) in loci.iter().enumerate() {
        // Keep about half the loci for partial overlap.
        if idx % 2 == 1 {
            continue;
        }
        let weight_raw = ((idx * 19 + 5) % 31) as f64;
        let weight = (weight_raw - 15.0) / 9.0;
        out.push_str(&format!(
            "{}:{}\t{}\t{}\t{weight:.8}\n",
            locus.chrom, locus.pos, locus.effect_allele, locus.other_allele
        ));
    }

    out.push_str("9:999999\tA\tG\t0.30000000\n");
    out.push_str("1:123456\tA\n");
    fs::write(path, out)?;
    Ok(())
}

struct ParsedSscore {
    score_names: Vec<String>,
    rows: Vec<SscoreRow>,
}

struct SscoreRow {
    iid: String,
    avg: BTreeMap<String, f64>,
    missing_pct: BTreeMap<String, f64>,
}

fn parse_sscore_table(path: &Path) -> Result<ParsedSscore, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    let mut lines = content.lines().filter(|line| !line.starts_with("#REGION"));
    let header = lines
        .next()
        .ok_or("missing sscore header")?
        .split('\t')
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();

    if header.is_empty() || header[0] != "#IID" {
        return Err(format!("unexpected sscore header in {}", path.display()).into());
    }
    if (header.len() - 1) % 2 != 0 {
        return Err(format!("invalid score column count in {}", path.display()).into());
    }

    let mut score_names = Vec::new();
    let mut idx = 1usize;
    while idx + 1 < header.len() {
        let avg_name = &header[idx];
        let miss_name = &header[idx + 1];
        let score = avg_name
            .strip_suffix("_AVG")
            .ok_or("bad avg header suffix")?
            .to_string();
        let expected_missing = format!("{score}_MISSING_PCT");
        if miss_name != &expected_missing {
            return Err(format!("mismatched missing pct column for {score}").into());
        }
        score_names.push(score);
        idx += 2;
    }

    let mut rows = Vec::new();
    for line in lines {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let cols = line.split('\t').collect::<Vec<_>>();
        if cols.len() != header.len() {
            return Err(format!("row width mismatch in {}", path.display()).into());
        }
        let iid = cols[0].to_string();
        let mut avg = BTreeMap::new();
        let mut missing_pct = BTreeMap::new();
        let mut col_idx = 1usize;
        for score in &score_names {
            let avg_val: f64 = cols[col_idx].parse()?;
            let miss_val: f64 = cols[col_idx + 1].parse()?;
            avg.insert(score.clone(), avg_val);
            missing_pct.insert(score.clone(), miss_val);
            col_idx += 2;
        }
        rows.push(SscoreRow {
            iid,
            avg,
            missing_pct,
        });
    }

    Ok(ParsedSscore { score_names, rows })
}
