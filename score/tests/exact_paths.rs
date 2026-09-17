//! Exact score arithmetic along every dispatch path. Random panels with missing calls, flipped
//! alleles, duplicate score lines, present zero weights, multiallelic loci and all four PLINK
//! codes are scored through the dense and sparse kernels, the bounded accumulator, small-keep
//! direct scoring and split filesets. Every path must leave bit-identical cells and counts, and
//! every sum and average must be the correctly rounded exact rational, which this file derives
//! from the written weights with its own integer arithmetic.

use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use gnomon::pipeline::{Dispatch, PipelineContext, run};
use gnomon::prepare::prepare_for_computation;

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
