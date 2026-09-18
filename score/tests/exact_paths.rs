//! Exact score arithmetic along every dispatch path. Random panels with missing calls, flipped
//! alleles, duplicate score lines, present zero weights, multiallelic loci and all four PLINK
//! codes are scored through the dense and sparse kernels, the bounded accumulator, small-keep
//! direct scoring and split filesets. Every path must leave bit-identical cells and counts, and
//! every sum and average must be the correctly rounded exact rational, which this file derives
//! from the written weights with its own integer arithmetic. The same holds for VCF and BCF
//! input scored natively, from GT calls, DS dosages and GP probabilities, and a GT panel prints
//! the same numbers through the VCF, BCF and PLINK paths.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gnomon::pipeline::{Dispatch, PipelineContext, run};
use gnomon::prepare::prepare_for_computation;
use gnomon::score::native_vcf::{NativeVcfScoreResult, score_vcf_streaming};

type TestResult = Result<(), Box<dyn Error>>;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const PAIRS: [(&str, &str); 4] = [("A", "G"), ("C", "T"), ("A", "C"), ("G", "T")];

/// One `.bim` row and every person's two-bit call on it.
struct Row {
    chrom: u8,
    pos: u32,
    a1: &'static str,
    a2: &'static str,
    calls: Vec<u8>,
}

/// One score-file line: weights in millionths, `None` where the line leaves a score empty.
struct Line {
    chrom: u8,
    pos: u32,
    effect: &'static str,
    other: &'static str,
    weights: Vec<Option<i64>>,
}

#[derive(Clone, Copy, PartialEq)]
enum Locus {
    Simple,
    Multiallelic,
    /// Two identical contexts, resolved by the heuristic chain; only the last score scores it.
    Heuristic,
}

struct Panel {
    people: usize,
    scores: usize,
    rows: Vec<Row>,
    lines: Vec<Line>,
    loci: Vec<(u8, u32, Locus)>,
}

impl Panel {
    fn random(rng: &mut Rng, people: usize, scores: usize) -> Self {
        let (mut rows, mut lines, mut loci) = (Vec::new(), Vec::new(), Vec::new());
        // The last score is kept for heuristic loci, so the oracle can leave it out.
        let checked = scores - 1;
        for chrom in 1..=2u8 {
            for j in 0..24u32 {
                let pos = 1000 + 10 * j;
                let calls = |rng: &mut Rng| (0..people).map(|_| rng.below(4) as u8).collect();
                let locus = match rng.below(12) {
                    0 | 1 => Locus::Multiallelic,
                    2 if scores > 1 => Locus::Heuristic,
                    _ => Locus::Simple,
                };
                let contexts: Vec<(&str, &str)> = match locus {
                    Locus::Simple => vec![PAIRS[rng.below(PAIRS.len())]],
                    Locus::Multiallelic => vec![("A", "G"), ("A", "C")],
                    Locus::Heuristic => vec![("A", "G"), ("A", "G")],
                };
                for &(a1, a2) in &contexts {
                    rows.push(Row {
                        chrom,
                        pos,
                        a1,
                        a2,
                        calls: calls(rng),
                    });
                }
                loci.push((chrom, pos, locus));
                let line_count = 1 + usize::from(rng.below(4) == 0);
                for _ in 0..line_count {
                    let (a1, a2) = contexts[rng.below(contexts.len())];
                    let (effect, other) = if rng.below(2) == 0 { (a1, a2) } else { (a2, a1) };
                    let weights = (0..scores)
                        .map(|score| {
                            let scored = if locus == Locus::Heuristic {
                                score == checked
                            } else {
                                score < checked || scores == 1
                            };
                            (scored && rng.below(3) != 0).then(|| match rng.below(10) {
                                0 => 0,
                                1 => (rng.below(9) as i64 + 1) * 1_000_000,
                                _ => rng.below(4_000_001) as i64 - 2_000_000,
                            })
                        })
                        .collect();
                    lines.push(Line {
                        chrom,
                        pos,
                        effect,
                        other,
                        weights,
                    });
                }
            }
        }
        Self {
            people,
            scores,
            rows,
            lines,
            loci,
        }
    }

    /// Writes the panel as one fileset, or one per chromosome, and returns the prefixes.
    fn write_filesets(&self, dir: &Path, split: bool) -> Result<Vec<PathBuf>, Box<dyn Error>> {
        let fam: String = (0..self.people).map(|i| format!("F I{i} 0 0 0 -9\n")).collect();
        let parts: Vec<Vec<&Row>> = if split {
            (1..=2u8)
                .map(|chrom| self.rows.iter().filter(|row| row.chrom == chrom).collect())
                .collect()
        } else {
            vec![self.rows.iter().collect()]
        };
        let mut prefixes = Vec::new();
        for (part, rows) in parts.into_iter().enumerate() {
            let prefix = dir.join(format!("panel-{split}-{part}"));
            fs::write(prefix.with_extension("fam"), &fam)?;
            let bim: String = rows
                .iter()
                .enumerate()
                .map(|(i, row)| format!("{} v{part}_{i} 0 {} {} {}\n", row.chrom, row.pos, row.a1, row.a2))
                .collect();
            fs::write(prefix.with_extension("bim"), bim)?;
            let mut bed = vec![0x6c, 0x1b, 0x01];
            for row in rows {
                for chunk in row.calls.chunks(4) {
                    bed.push(chunk.iter().enumerate().fold(0u8, |byte, (i, &call)| byte | (call << (2 * i))));
                }
            }
            fs::write(prefix.with_extension("bed"), bed)?;
            prefixes.push(prefix);
        }
        Ok(prefixes)
    }

    fn write_scores(&self, path: &Path) -> Result<(), Box<dyn Error>> {
        let mut text = String::from("variant_id\teffect_allele\tother_allele");
        for score in 0..self.scores {
            text.push_str(&format!("\tS{score:02}"));
        }
        text.push('\n');
        for line in &self.lines {
            text.push_str(&format!("{}:{}\t{}\t{}", line.chrom, line.pos, line.effect, line.other));
            for weight in &line.weights {
                text.push('\t');
                if let Some(micro) = weight {
                    let sign = if *micro < 0 { "-" } else { "" };
                    text.push_str(&format!("{sign}{}.{:06}", micro.abs() / 1_000_000, micro.abs() % 1_000_000));
                }
            }
            text.push('\n');
        }
        fs::write(path, text)?;
        Ok(())
    }

    /// Per kept person (in `kept` order) and checked score: the exact sum in millionths, the
    /// score's variant count and the person's missing count.
    fn oracle(&self, kept: &[usize]) -> (Vec<i128>, Vec<u32>, Vec<u32>) {
        let checked = if self.scores == 1 { 1 } else { self.scores - 1 };
        let (mut sums, mut missing) = (vec![0i128; kept.len() * checked], vec![0u32; kept.len() * checked]);
        let mut counts = vec![0u32; checked];
        let dose = |call: u8, effect: &str, row: &Row| {
            let a2 = [0, 0, 1, 2][call as usize];
            if effect == row.a2 { a2 } else { 2 - a2 }
        };
        for &(chrom, pos, locus) in &self.loci {
            if locus == Locus::Heuristic {
                continue;
            }
            let rows: Vec<&Row> = self.rows.iter().filter(|r| r.chrom == chrom && r.pos == pos).collect();
            let lines: Vec<&Line> = self.lines.iter().filter(|l| l.chrom == chrom && l.pos == pos).collect();
            for score in 0..checked {
                let scored: Vec<&Line> = lines.iter().copied().filter(|l| l.weights[score].is_some()).collect();
                if scored.is_empty() {
                    continue;
                }
                match locus {
                    Locus::Simple => {
                        // One entry per line, counted once per row.
                        let row = rows[0];
                        counts[score] += 1;
                        for (k, &person) in kept.iter().enumerate() {
                            let call = row.calls[person];
                            if call == 1 {
                                missing[k * checked + score] += 1;
                                continue;
                            }
                            for line in &scored {
                                sums[k * checked + score] +=
                                    i128::from(line.weights[score].unwrap() * dose(call, line.effect, row));
                            }
                        }
                    }
                    _ => {
                        // Every line is its own application on the one context carrying its pair.
                        for line in &scored {
                            counts[score] += 1;
                            let row = rows
                                .iter()
                                .copied()
                                .find(|r| {
                                    (r.a1 == line.effect && r.a2 == line.other)
                                        || (r.a2 == line.effect && r.a1 == line.other)
                                })
                                .expect("every line matches a context");
                            for (k, &person) in kept.iter().enumerate() {
                                let call = row.calls[person];
                                if call == 1 {
                                    missing[k * checked + score] += 1;
                                } else {
                                    sums[k * checked + score] +=
                                        i128::from(line.weights[score].unwrap() * dose(call, line.effect, row));
                                }
                            }
                        }
                    }
                }
            }
        }
        (sums, counts, missing)
    }
}

/// `numerator / denominator` correctly rounded: 800 decimals and a sticky digit, parsed.
fn rounded_quotient(numerator: i128, denominator: u128) -> f64 {
    if numerator == 0 {
        return 0.0;
    }
    let magnitude = numerator.unsigned_abs();
    let mut remainder = magnitude % denominator;
    let mut text = format!(
        "{}{}.",
        if numerator < 0 { "-" } else { "" },
        magnitude / denominator
    );
    for _ in 0..800 {
        remainder *= 10;
        text.push(char::from(b'0' + (remainder / denominator) as u8));
        remainder %= denominator;
    }
    if remainder != 0 {
        text.push('1');
    }
    text.parse().expect("decimal text")
}

#[test]
fn every_dispatch_path_gives_the_correctly_rounded_exact_scores() -> TestResult {
    let dir = tempfile::tempdir()?;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let plans = [
        (Dispatch::Decide, false),
        (Dispatch::Dense, false),
        (Dispatch::Sparse, false),
        (Dispatch::Dense, true),
        (Dispatch::Sparse, true),
    ];
    for (seed, (people, scores)) in [(5usize, 1usize), (37, 3), (64, 17), (257, 70)].into_iter().enumerate() {
        let panel = Panel::random(&mut rng, people, scores);
        let scenario = dir.path().join(format!("scenario{seed}"));
        fs::create_dir_all(&scenario)?;
        let score_path = scenario.join("weights.tsv");
        panel.write_scores(&score_path)?;
        let keeps: Vec<Vec<usize>> = vec![
            (0..people).collect(),
            (0..people).filter(|p| p % 3 != 1).rev().collect(),
            (0..people).rev().step_by(7).take(5).collect(),
        ];
        for (keep_index, keep) in keeps.iter().enumerate() {
            let keep_file = scenario.join(format!("keep{keep_index}.txt"));
            fs::write(&keep_file, keep.iter().map(|p| format!("I{p}\n")).collect::<String>())?;
            let mut reference: Option<(Vec<i64>, Vec<u32>)> = None;
            for split in [false, true] {
                let prefixes = panel.write_filesets(&scenario, split)?;
                let subset = (keep_index > 0).then_some(keep_file.as_path());
                let prep = Arc::new(prepare_for_computation(&prefixes, &[score_path.clone()], subset, None)?);
                for (dispatch, bounded) in plans {
                    let mut context = PipelineContext::new(Arc::clone(&prep));
                    context.dispatch = dispatch;
                    context.force_bounded_accumulator = bounded;
                    let (cells, counts) = run(&context)?;
                    let label = format!(
                        "people {people} scores {scores} keep {keep_index} split {split} {dispatch:?} bounded {bounded}"
                    );
                    match &reference {
                        None => reference = Some((cells.clone(), counts.clone())),
                        Some((want_cells, want_counts)) => {
                            assert!(&cells == want_cells, "cells differ: {label}");
                            assert!(&counts == want_counts, "counts differ: {label}");
                        }
                    }
                    if split || dispatch != Dispatch::Decide || bounded {
                        continue;
                    }
                    // The oracle, once per keep: people in output order, scores by name.
                    let kept: Vec<usize> = prep.output_idx_to_fam_idx.iter().map(|f| f.0 as usize).collect();
                    let (sums, variant_counts, missing) = panel.oracle(&kept);
                    let checked = if scores == 1 { 1 } else { scores - 1 };
                    let exact = prep.exact();
                    let stride = exact.stride();
                    for score in 0..checked {
                        let column = prep
                            .score_names
                            .iter()
                            .position(|name| name == &format!("S{score:02}"))
                            .expect("score column");
                        assert_eq!(prep.score_variant_counts[column], variant_counts[score], "{label}");
                        for person in 0..kept.len() {
                            let lanes = &cells[person * stride..(person + 1) * stride];
                            let cell = person * checked + score;
                            assert_eq!(counts[person * scores + column], missing[cell], "{label} person {person}");
                            let want_sum: f64 = format!("{}e-6", sums[cell]).parse()?;
                            assert_eq!(
                                exact.sum(column, lanes).to_bits(),
                                want_sum.to_bits(),
                                "sum: {label} person {person} score {score}"
                            );
                            let used = variant_counts[score] - missing[cell];
                            let want_average = if used == 0 {
                                0.0
                            } else {
                                rounded_quotient(sums[cell], 1_000_000 * u128::from(used))
                            };
                            assert_eq!(
                                exact.average(column, lanes, used).to_bits(),
                                want_average.to_bits(),
                                "average: {label} person {person} score {score}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// What a native scenario writes for each person on each site.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Field {
    /// GT calls, phased and unphased, with missing calls.
    Gt,
    /// GT and DS; where DS is missing the call scores.
    GtDs,
    /// GP alone.
    Gp,
}

/// `digits × 10^-places` with exactly `places` fraction digits, trailing zeros kept.
fn decimal_text(digits: i64, places: u32) -> String {
    let power = 10i64.pow(places);
    if places == 0 {
        format!("{digits}")
    } else {
        format!("{}.{:0width$}", digits / power, digits % power, width = places as usize)
    }
}

/// A random decimal in `[0, whole]` with up to five places: its text and its value in millionths.
fn random_decimal(rng: &mut Rng, whole: i64) -> (String, i128) {
    let places = rng.below(6) as u32;
    let digits = rng.below((whole * 10i64.pow(places) + 1) as usize) as i64;
    (decimal_text(digits, places), i128::from(digits) * 10i128.pow(6 - places))
}

/// Biallelic sites written as VCF samples, with every person's dosage of each allele.
struct NativePanel {
    /// The sites as `.bim` rows, A1 the ALT allele and A2 the REF allele, and the score lines.
    panel: Panel,
    field: Field,
    /// Per site, each person's sample column.
    samples: Vec<Vec<String>>,
    /// Per site, each person's ALT and REF dosages in millionths, `None` where missing.
    doses: Vec<Vec<Option<(i128, i128)>>>,
}

impl NativePanel {
    fn random(rng: &mut Rng, people: usize, scores: usize, field: Field) -> Self {
        let (mut rows, mut lines, mut loci) = (Vec::new(), Vec::new(), Vec::new());
        let (mut samples, mut doses) = (Vec::new(), Vec::new());
        for chrom in 1..=2u8 {
            for j in 0..24u32 {
                let pos = 1000 + 10 * j;
                let (reference, alternate) = PAIRS[rng.below(PAIRS.len())];
                let (mut texts, mut site_doses, mut calls) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..people {
                    let (gt, copies): (&str, Option<i128>) = match rng.below(6) {
                        0 => ("0/0", Some(0)),
                        1 => ("0|1", Some(1)),
                        2 => ("1|0", Some(1)),
                        3 => ("0/1", Some(1)),
                        4 => ("1/1", Some(2)),
                        _ => ("./.", None),
                    };
                    let call = copies.map(|alt| (alt * 1_000_000, (2 - alt) * 1_000_000));
                    let (text, dose) = match field {
                        Field::Gt => (gt.to_string(), call),
                        Field::GtDs => match rng.below(16) {
                            0..=3 => (format!("{gt}:."), call),
                            // 2 - 2.000001 = -1e-6 is inside the DS tolerance, so REF clamps to zero.
                            4 => (format!("{gt}:2.000001"), Some((2_000_001, 0))),
                            _ => {
                                let (ds, alt) = random_decimal(rng, 2);
                                (format!("{gt}:{ds}"), Some((alt, 2_000_000 - alt)))
                            }
                        },
                        Field::Gp => match rng.below(6) {
                            0 => (".".to_string(), None),
                            _ => {
                                let [(p0, q0), (p1, q1), (p2, q2)] = std::array::from_fn(|_| random_decimal(rng, 1));
                                (format!("{p0},{p1},{p2}"), Some((q1 + 2 * q2, 2 * q0 + q1)))
                            }
                        },
                    };
                    calls.push(match copies {
                        Some(2) => 0b00,
                        Some(1) => 0b10,
                        Some(_) => 0b11,
                        None => 0b01,
                    });
                    texts.push(text);
                    site_doses.push(dose);
                }
                rows.push(Row {
                    chrom,
                    pos,
                    a1: alternate,
                    a2: reference,
                    calls,
                });
                loci.push((chrom, pos, Locus::Simple));
                samples.push(texts);
                doses.push(site_doses);
                for _ in 0..1 + usize::from(rng.below(4) == 0) {
                    let (effect, other) = if rng.below(2) == 0 {
                        (alternate, reference)
                    } else {
                        (reference, alternate)
                    };
                    let weights = (0..scores)
                        .map(|_| {
                            (rng.below(3) != 0).then(|| match rng.below(10) {
                                0 => 0,
                                1 => (rng.below(9) as i64 + 1) * 1_000_000,
                                _ => rng.below(4_000_001) as i64 - 2_000_000,
                            })
                        })
                        .collect();
                    lines.push(Line {
                        chrom,
                        pos,
                        effect,
                        other,
                        weights,
                    });
                }
            }
        }
        Self {
            panel: Panel {
                people,
                scores,
                rows,
                lines,
                loci,
            },
            field,
            samples,
            doses,
        }
    }

    /// Writes the sites as VCF text, and the same records as BCF through noodles.
    fn write_vcf_and_bcf(&self, dir: &Path) -> Result<(PathBuf, PathBuf), Box<dyn Error>> {
        use noodles_vcf::variant::io::Write as _;

        let format = match self.field {
            Field::Gt => "GT",
            Field::GtDs => "GT:DS",
            Field::Gp => "GP",
        };
        let mut text = String::from(
            "##fileformat=VCFv4.2\n##contig=<ID=1>\n##contig=<ID=2>\n\
             ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
             ##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"ALT dosage\">\n\
             ##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Genotype probabilities\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT",
        );
        for person in 0..self.panel.people {
            text.push_str(&format!("\tI{person}"));
        }
        text.push('\n');
        for (row, site) in self.panel.rows.iter().zip(&self.samples) {
            text.push_str(&format!("{}\t{}\t.\t{}\t{}\t.\tPASS\t.\t{format}", row.chrom, row.pos, row.a2, row.a1));
            for sample in site {
                text.push('\t');
                text.push_str(sample);
            }
            text.push('\n');
        }
        let (vcf, bcf) = (dir.join("cohort.vcf"), dir.join("cohort.bcf"));
        fs::write(&vcf, text)?;
        let mut reader = noodles_vcf::io::Reader::new(std::io::BufReader::new(fs::File::open(&vcf)?));
        let header = reader.read_header()?;
        let mut writer = noodles_bcf::io::Writer::new(fs::File::create(&bcf)?);
        writer.write_header(&header)?;
        let mut record = noodles_vcf::variant::RecordBuf::default();
        while reader.read_record_buf(&header, &mut record)? != 0 {
            writer.write_variant_record(&header, &record)?;
        }
        writer.try_finish()?;
        Ok((vcf, bcf))
    }

    /// Per person and score: the exact sum in units of 10^-12, the score's variant count and the
    /// person's missing count.
    fn oracle(&self) -> (Vec<i128>, Vec<u32>, Vec<u32>) {
        let (people, scores) = (self.panel.people, self.panel.scores);
        let (mut sums, mut missing) = (vec![0i128; people * scores], vec![0u32; people * scores]);
        let mut counts = vec![0u32; scores];
        for (site, row) in self.panel.rows.iter().enumerate() {
            let lines: Vec<&Line> =
                self.panel.lines.iter().filter(|l| l.chrom == row.chrom && l.pos == row.pos).collect();
            for score in 0..scores {
                let scored: Vec<&Line> = lines.iter().copied().filter(|l| l.weights[score].is_some()).collect();
                if scored.is_empty() {
                    continue;
                }
                counts[score] += 1;
                for person in 0..people {
                    let cell = person * scores + score;
                    let Some((alt, reference)) = self.doses[site][person] else {
                        missing[cell] += 1;
                        continue;
                    };
                    for line in &scored {
                        let dose = if line.effect == row.a1 { alt } else { reference };
                        sums[cell] += i128::from(line.weights[score].unwrap()) * dose;
                    }
                }
            }
        }
        (sums, counts, missing)
    }
}

/// Scores `input` natively against one score file, with every sample kept.
fn score_natively(input: &Path, score: &Path) -> Result<NativeVcfScoreResult, Box<dyn Error>> {
    score_vcf_streaming(input, &[score.to_path_buf()], None, None).map_err(|error| error as Box<dyn Error>)
}

/// The cell of `name` for `person`, and the score's variant count.
fn native_cell(result: &NativeVcfScoreResult, name: &str, person: usize) -> (usize, u32) {
    let column = result.score_names.iter().position(|n| n == name).expect("score column");
    (person * result.score_names.len() + column, result.score_variant_counts[column])
}

#[test]
fn native_vcf_and_bcf_give_the_correctly_rounded_exact_scores() -> TestResult {
    const SCALE: u128 = 1_000_000_000_000;
    let dir = tempfile::tempdir()?;
    let mut rng = Rng(0x2354_c0de_5eed_0001);
    for (seed, (people, scores)) in [(5usize, 1usize), (37, 3), (64, 17), (257, 70)].into_iter().enumerate() {
        for field in [Field::Gt, Field::GtDs, Field::Gp] {
            let native = NativePanel::random(&mut rng, people, scores, field);
            let scenario = dir.path().join(format!("native{seed}-{field:?}"));
            fs::create_dir_all(&scenario)?;
            let score_path = scenario.join("weights.tsv");
            native.panel.write_scores(&score_path)?;
            let (vcf, bcf) = native.write_vcf_and_bcf(&scenario)?;
            let (sums, counts, missing) = native.oracle();
            let results = [
                ("vcf", score_natively(&vcf, &score_path)?),
                ("bcf", score_natively(&bcf, &score_path)?),
            ];
            for (format, result) in &results {
                let label = format!("people {people} scores {scores} {field:?} {format}");
                assert_eq!(result.person_iids.len(), people, "{label}");
                for score in 0..scores {
                    for person in 0..people {
                        let (cell, count) = native_cell(result, &format!("S{score:02}"), person);
                        let want = person * scores + score;
                        assert_eq!(count, counts[score], "{label}");
                        assert_eq!(result.missing_counts[cell], missing[want], "{label} person {person}");
                        assert_eq!(
                            result.sum(cell).to_bits(),
                            rounded_quotient(sums[want], SCALE).to_bits(),
                            "sum: {label} person {person} score {score}"
                        );
                        let used = count - missing[want];
                        let want_average = if used == 0 {
                            0.0
                        } else {
                            rounded_quotient(sums[want], SCALE * u128::from(used))
                        };
                        assert_eq!(
                            result.average(cell, used).to_bits(),
                            want_average.to_bits(),
                            "average: {label} person {person} score {score}"
                        );
                    }
                }
            }
            if field != Field::Gt {
                continue;
            }
            // The same calls as a PLINK fileset print the same numbers.
            let prefixes = native.panel.write_filesets(&scenario, false)?;
            let prep = Arc::new(prepare_for_computation(&prefixes, &[score_path.clone()], None, None)?);
            let (cells, plink_missing) = run(&PipelineContext::new(Arc::clone(&prep)))?;
            let exact = prep.exact();
            let stride = exact.stride();
            let vcf_result = &results[0].1;
            for (column, name) in prep.score_names.iter().enumerate() {
                for person in 0..people {
                    let label = format!("people {people} scores {scores} plink {name} person {person}");
                    let lanes = &cells[person * stride..(person + 1) * stride];
                    let (cell, count) = native_cell(vcf_result, name, person);
                    assert_eq!(prep.score_variant_counts[column], count, "{label}");
                    let used = count - plink_missing[person * scores + column];
                    assert_eq!(exact.sum(column, lanes).to_bits(), vcf_result.sum(cell).to_bits(), "{label}");
                    assert_eq!(
                        exact.average(column, lanes, used).to_bits(),
                        vcf_result.average(cell, used).to_bits(),
                        "{label}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn native_lanes_flush_exactly_at_their_bound() -> TestResult {
    // A weight of 2^52 on two copies bounds a term at 2^53, so a score's lanes take
    // k = floor((2^63 - 1) / 2^53) = 1023 alleles between flushes. The person with two copies
    // everywhere fills a lane to k × 2^53 = 2^63 - 2^53, where one more term would leave i64.
    const WEIGHT: i128 = 1 << 52;
    let dir = tempfile::tempdir()?;
    for ds in [false, true] {
        for names in [&["UP"][..], &["UP", "DOWN"][..]] {
            for sites in [1023usize, 1024, 2046, 2047] {
                let scenario = dir.path().join(format!("lanes-{ds}-{}-{sites}", names.len()));
                fs::create_dir_all(&scenario)?;
                let mut vcf = String::from(
                    "##fileformat=VCFv4.2\n##contig=<ID=1>\n\
                     ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
                     ##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"ALT dosage\">\n\
                     #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tI0\tI1\tI2\n",
                );
                let mut weights = format!("variant_id\teffect_allele\tother_allele\t{}\n", names.join("\t"));
                let (mut sums, mut missing) = ([0i128; 3], [0u32; 3]);
                for site in 0..sites {
                    let pos = 1000 + 10 * site;
                    let calls = [Some(2i128), (site % 3 != 2).then_some(1), Some(0)];
                    vcf.push_str(&format!("1\t{pos}\t.\tA\tG\t.\tPASS\t.\t{}", if ds { "GT:DS" } else { "GT" }));
                    for copies in calls {
                        vcf.push('\t');
                        vcf.push_str(match copies {
                            Some(2) => "1/1",
                            Some(1) => "0/1",
                            Some(_) => "0/0",
                            None => "./.",
                        });
                        if ds {
                            vcf.push_str(&copies.map_or(":.".to_string(), |copies| format!(":{copies}")));
                        }
                    }
                    vcf.push('\n');
                    weights.push_str(&format!("1:{pos}\tG\tA\t{WEIGHT}"));
                    if names.len() == 2 {
                        weights.push_str(&format!("\t-{WEIGHT}"));
                    }
                    weights.push('\n');
                    for (person, copies) in calls.into_iter().enumerate() {
                        match copies {
                            Some(copies) => sums[person] += WEIGHT * copies,
                            None => missing[person] += 1,
                        }
                    }
                }
                let (vcf_path, score_path) = (scenario.join("cohort.vcf"), scenario.join("weights.tsv"));
                fs::write(&vcf_path, vcf)?;
                fs::write(&score_path, weights)?;
                let result = score_natively(&vcf_path, &score_path)?;
                for person in 0..3 {
                    for (score, name) in names.iter().enumerate() {
                        let label = format!("ds {ds} scores {} sites {sites} person {person} {name}", names.len());
                        let sum = if score == 0 { sums[person] } else { -sums[person] };
                        let (cell, count) = native_cell(&result, name, person);
                        assert_eq!(count, sites as u32, "{label}");
                        assert_eq!(result.missing_counts[cell], missing[person], "{label}");
                        assert_eq!(result.sum(cell).to_bits(), rounded_quotient(sum, 1).to_bits(), "{label}");
                        let used = count - missing[person];
                        assert_eq!(
                            result.average(cell, used).to_bits(),
                            rounded_quotient(sum, u128::from(used)).to_bits(),
                            "{label}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}
