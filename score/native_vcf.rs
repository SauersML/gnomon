use crate::score::prepare::{
    EffectOnlyMatches, OtherAlleleMatch, names_no_single_other_allele, resolve_other_allele,
};
use crate::score::types::{GenomicRegion, parse_chromosome_label};
use crate::shared::files::open_variant_source;
use ahash::{AHashMap, AHashSet};
use flate2::read::MultiGzDecoder;
use libdeflater::Decompressor;
use memchr::memchr;
use noodles_vcf::header::record::value::map::format::{Number as FormatNumber, Type as FormatType};
use noodles_vcf::io::Reader as VcfReader;
use noodles_vcf::variant::record::AlternateBases as _;
use noodles_vcf::variant::record::Samples as _;
use noodles_vcf::variant::record::samples::keys::key;
use noodles_vcf::variant::record::samples::series::{
    Value as SeriesValue, value::Array as SeriesArray,
};
use rayon::prelude::*;
use std::error::Error;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

mod stream;

#[derive(Debug)]
pub struct NativeVcfScoreResult {
    pub person_iids: Vec<String>,
    pub score_names: Vec<String>,
    pub score_variant_counts: Vec<u32>,
    pub sum_scores: Vec<f64>,
    pub missing_counts: Vec<u32>,
    pub matched_variants: usize,
}

/// One native score row, as spans into its `ScoreRules` buffers.
#[derive(Debug, Clone, Copy)]
struct ScoreRule {
    effect_allele: (usize, usize),
    other_allele: (usize, usize),
    applications: (usize, usize),
}

#[derive(Debug, Clone, Copy)]
struct ScoreApplication {
    score_index: usize,
    weight: f64,
}

/// Every native score row with at least one weight, found by position.
///
/// Rows live in one flat list with their allele text and weights in shared
/// buffers, so a score file with millions of rows grows a few buffers instead
/// of allocating two strings and a vector per row. A position's rules keep
/// file order, the order their weights are summed in.
#[derive(Debug)]
struct ScoreRules {
    /// Rules grouped by position, in file order within a position.
    rules: Vec<ScoreRule>,
    /// Each position's range of `rules`.
    ranges: AHashMap<VariantKey, (usize, usize)>,
    alleles: String,
    applications: Vec<ScoreApplication>,
}

impl ScoreRules {
    fn get(&self, key: &VariantKey) -> Option<&[ScoreRule]> {
        self.ranges
            .get(key)
            .map(|&(start, end)| &self.rules[start..end])
    }

    fn contains_key(&self, key: &VariantKey) -> bool {
        self.ranges.contains_key(key)
    }

    fn allele(&self, span: (usize, usize)) -> &str {
        &self.alleles[span.0..span.1]
    }

    fn applications(&self, rule: &ScoreRule) -> &[ScoreApplication] {
        &self.applications[rule.applications.0..rule.applications.1]
    }
}

/// `ScoreRules` under construction: rows in file order, not yet grouped.
#[derive(Debug, Default)]
struct ScoreRulesBuilder {
    rows: Vec<(VariantKey, ScoreRule)>,
    alleles: String,
    applications: Vec<ScoreApplication>,
}

impl ScoreRulesBuilder {
    fn push_application(&mut self, application: ScoreApplication) {
        self.applications.push(application);
    }

    /// Adds a row whose weights are the applications pushed since `applications_start`.
    fn push_row(
        &mut self,
        key: VariantKey,
        effect_allele: &str,
        other_allele: &str,
        applications_start: usize,
    ) {
        let effect_allele = self.push_allele(effect_allele);
        let other_allele = self.push_allele(other_allele);
        self.rows.push((
            key,
            ScoreRule {
                effect_allele,
                other_allele,
                applications: (applications_start, self.applications.len()),
            },
        ));
    }

    fn push_allele(&mut self, allele: &str) -> (usize, usize) {
        let start = self.alleles.len();
        self.alleles.push_str(allele);
        (start, self.alleles.len())
    }

    fn finish(mut self) -> ScoreRules {
        // A stable sort groups each position's rules and keeps their file order.
        self.rows.sort_by_key(|(key, _)| *key);
        let mut rules = Vec::with_capacity(self.rows.len());
        let mut ranges: AHashMap<VariantKey, (usize, usize)> =
            AHashMap::with_capacity(self.rows.len());
        for (index, (key, rule)) in self.rows.into_iter().enumerate() {
            rules.push(rule);
            ranges
                .entry(key)
                .and_modify(|range| range.1 = index + 1)
                .or_insert((index, index + 1));
        }
        ScoreRules {
            rules,
            ranges,
            alleles: self.alleles,
            applications: self.applications,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MatchedRule {
    score_index: usize,
    weight: f64,
    effect_is_ref: bool,
}

type VariantKey = (u8, u32);

pub fn score_vcf_streaming(
    input_path: &Path,
    native_score_files: &[PathBuf],
    keep: Option<&Path>,
    score_regions: Option<&std::collections::HashMap<String, GenomicRegion>>,
) -> Result<NativeVcfScoreResult, Box<dyn Error + Send + Sync>> {
    let (score_names, rules_by_key) = load_score_rules(native_score_files, score_regions)?;
    let source = open_variant_source(input_path)?;
    stream::score_source(source, input_path, keep, score_names, &rules_by_key)
}

/// What decoding a record needs besides the record.
struct DecodeContext<'a> {
    rules_by_key: &'a ScoreRules,
    kept_indices: &'a [usize],
    score_names: &'a [String],
}

/// Accumulates decoded records taken in input order.
///
/// The records at a position holding a rule that names no single other allele
/// are accumulated together, when the next scored position or the end of input
/// shows every one of them has been read.
struct RecordAccumulator<'a> {
    rules_by_key: &'a ScoreRules,
    score_names: &'a [String],
    totals: ScoreTotals,
    pending: PendingPosition,
    effect_only_matches: EffectOnlyMatches,
}

impl<'a> RecordAccumulator<'a> {
    fn new(rules_by_key: &'a ScoreRules, score_names: &'a [String], num_people: usize) -> Self {
        Self {
            rules_by_key,
            score_names,
            totals: ScoreTotals::new(num_people, score_names.len()),
            pending: PendingPosition::default(),
            effect_only_matches: EffectOnlyMatches::default(),
        }
    }

    /// Takes the next record in input order, returning the error scoring it raises.
    fn take(&mut self, decoded: &mut DecodedRecord) -> Result<(), Box<dyn Error + Send + Sync>> {
        if let Some(err) = decoded.error.take() {
            return Err(err);
        }
        let rules_by_key = self.rules_by_key;
        let score_names = self.score_names;
        if let Some(key) = decoded.key {
            if self.pending.key.is_some_and(|open| open != key) {
                self.pending.resolve(
                    rules_by_key,
                    score_names,
                    &mut self.effect_only_matches,
                    &mut self.totals,
                )?;
            }
            if decoded.effect_only {
                if self.pending.key.is_none() {
                    self.pending
                        .open(key, decoded.chromosome.clone(), rules_by_key)?;
                }
                self.pending.push(decoded);
                return Ok(());
            }
        }
        for allele in &mut decoded.alleles[..decoded.allele_count] {
            self.totals.add_allele(
                std::mem::take(&mut allele.matched_rules),
                std::mem::take(&mut allele.column),
                |score_index| {
                    ref_effect_error(
                        &score_names[score_index],
                        &decoded.chromosome,
                        decoded.position,
                    )
                },
            )?;
        }
        Ok(())
    }

    /// Adds the terms of every allele taken so far to the sums.
    fn apply(&mut self) {
        self.totals.apply();
    }

    /// Accumulates the records at the last open position and returns the totals.
    fn finish(mut self) -> Result<ScoreTotals, Box<dyn Error + Send + Sync>> {
        self.pending.resolve(
            self.rules_by_key,
            self.score_names,
            &mut self.effect_only_matches,
            &mut self.totals,
        )?;
        self.totals.apply();
        self.effect_only_matches.report();
        Ok(self.totals)
    }
}

/// The scores `totals` holds, or the error for a run that matched no variant.
fn native_result(
    totals: ScoreTotals,
    person_iids: Vec<String>,
    score_names: Vec<String>,
    input_path: &Path,
) -> Result<NativeVcfScoreResult, Box<dyn Error + Send + Sync>> {
    let ScoreTotals {
        sum_scores,
        missing_counts,
        score_variant_counts,
        ..
    } = totals;

    // These are the same matched variant/score assignments counted in each
    // score's denominator. ALT ordinals are local to a record, so hashing them
    // across records conflates distinct alternate alleles at the same position.
    let matched_variants = score_variant_counts
        .iter()
        .map(|&count| count as usize)
        .sum();
    if matched_variants == 0 {
        return Err(format!(
            "No overlapping variants were found between '{}' and the score file(s).",
            input_path.display()
        )
        .into());
    }

    Ok(NativeVcfScoreResult {
        person_iids,
        score_names,
        score_variant_counts,
        sum_scores,
        missing_counts,
        matched_variants,
    })
}

/// The fewest people in one range when the sums are split across the rayon pool.
const MIN_PEOPLE_PER_RANGE: usize = 256;
/// Ranges of people per rayon worker when the sums are split across the pool.
const RANGES_PER_WORKER: usize = 4;

/// What scoring one VCF or BCF record needs, decoded away from the accumulating thread.
#[derive(Default)]
struct DecodedRecord {
    position: u32,
    /// The record's position, when score rules sit there.
    key: Option<VariantKey>,
    /// The record's chromosome as written, for messages about a scored record.
    chromosome: String,
    /// Whether a rule at the position names no single other allele. Which
    /// alleles such a rule scores is decided by `PendingPosition`, so this
    /// record's alleles carry no matched rules yet.
    effect_only: bool,
    /// Every (REF, ALT) pair of an `effect_only` record, in ALT order.
    rows: Vec<(String, String)>,
    /// Alleles with at least one matched rule, in ALT order. Only the first
    /// `allele_count` belong to this record; the rest keep their buffers.
    alleles: Vec<DecodedAllele>,
    allele_count: usize,
    /// The first error scoring this record raises.
    error: Option<Box<dyn Error + Send + Sync>>,
}

impl DecodedRecord {
    /// The next allele slot, for ALT ordinal `alt_offset`.
    fn next_allele(&mut self, alt_offset: usize) -> &mut DecodedAllele {
        if self.alleles.len() == self.allele_count {
            self.alleles.push(DecodedAllele::default());
        }
        let allele = &mut self.alleles[self.allele_count];
        allele.alt_offset = alt_offset;
        allele
    }
}

#[derive(Default)]
struct DecodedAllele {
    /// The allele's ALT ordinal, from 0.
    alt_offset: usize,
    matched_rules: Vec<MatchedRule>,
    /// One dosage per kept person, in output order.
    column: DosageColumn,
}

/// The code of a person without a hard call in `DosageColumn::Calls`. No real
/// code reaches it: a code holds at most fourteen copies of each allele.
const MISSING_CALL: u8 = 0xff;

/// Every kept person's dosage of one ALT allele, in output order.
enum DosageColumn {
    /// Hard calls, each `alt | ref << 4` allele copies, or `MISSING_CALL`.
    Calls(Vec<u8>),
    /// `[alt, ref]` dosages. A NaN ALT dosage is a missing dosage, and a NaN
    /// REF dosage one that genotype ploidy and every ALT dosage could not complete.
    Dosages(Vec<[f64; 2]>),
}

impl Default for DosageColumn {
    fn default() -> Self {
        Self::Calls(Vec::new())
    }
}

impl DosageColumn {
    /// The column, emptied, as hard calls.
    fn calls(&mut self) -> &mut Vec<u8> {
        if let Self::Dosages(_) = self {
            *self = Self::Calls(Vec::new());
        }
        let Self::Calls(codes) = self else {
            unreachable!("the column holds hard calls");
        };
        codes.clear();
        codes
    }

    /// The column, emptied, as dosages.
    fn dosages(&mut self) -> &mut Vec<[f64; 2]> {
        if let Self::Calls(_) = self {
            *self = Self::Dosages(Vec::new());
        }
        let Self::Dosages(dosages) = self else {
            unreachable!("the column holds dosages");
        };
        dosages.clear();
        dosages
    }

    /// Whether some person has a dosage without a REF dosage.
    fn has_incomplete_ref(&self) -> bool {
        match self {
            Self::Calls(_) => false,
            Self::Dosages(dosages) => dosages
                .iter()
                .any(|&[alt, reference]| !alt.is_nan() && reference.is_nan()),
        }
    }
}

/// Per-person score sums and missing counts, and each score's matched variants.
struct ScoreTotals {
    num_scores: usize,
    sum_scores: Vec<f64>,
    missing_counts: Vec<u32>,
    score_variant_counts: Vec<u32>,
    /// Alleles counted in `score_variant_counts` whose terms are not yet in the
    /// sums, in input order.
    unapplied: Vec<ScoredAllele>,
}

/// One scored allele's merged rules and every kept person's dosage of it.
struct ScoredAllele {
    matched_rules: Vec<MatchedRule>,
    /// The distinct scores of `matched_rules`, each counted once for a missing dosage.
    missing_scores: Vec<usize>,
    /// For hard calls, each rule's weight times 0 through 14 copies, and a zero
    /// at index 15, which is where a missing call's nibble points.
    call_terms: Vec<[f64; 16]>,
    column: DosageColumn,
}

impl ScoreTotals {
    fn new(num_people: usize, num_scores: usize) -> Self {
        Self {
            num_scores,
            sum_scores: vec![0.0f64; num_people * num_scores],
            missing_counts: vec![0u32; num_people * num_scores],
            score_variant_counts: vec![0u32; num_scores],
            unapplied: Vec::new(),
        }
    }

    /// Counts the allele once in each score it scores, and queues each rule's
    /// weight times every kept person's dosage of its effect allele for
    /// [`ScoreTotals::apply`]. A REF-effect rule fails when some person's dosage
    /// has no REF dosage.
    fn add_allele(
        &mut self,
        matched_rules: Vec<MatchedRule>,
        column: DosageColumn,
        ref_effect_error: impl Fn(usize) -> String,
    ) -> Result<(), String> {
        if let Some(rule) = matched_rules.iter().find(|rule| rule.effect_is_ref)
            && column.has_incomplete_ref()
        {
            return Err(ref_effect_error(rule.score_index));
        }
        let mut missing_scores = Vec::new();
        for rule in &matched_rules {
            if missing_scores.last() != Some(&rule.score_index) {
                missing_scores.push(rule.score_index);
                self.score_variant_counts[rule.score_index] += 1;
            }
        }
        if matched_rules.is_empty() {
            return Ok(());
        }
        let call_terms = match column {
            DosageColumn::Calls(_) => matched_rules
                .iter()
                .map(|rule| {
                    let mut terms = [0.0f64; 16];
                    for (copies, term) in (0u8..15).zip(&mut terms) {
                        *term = rule.weight * f64::from(copies);
                    }
                    terms
                })
                .collect(),
            DosageColumn::Dosages(_) => Vec::new(),
        };
        self.unapplied.push(ScoredAllele {
            matched_rules,
            missing_scores,
            call_terms,
            column,
        });
        Ok(())
    }

    /// Adds the terms of every queued allele to the sums, over ranges of people
    /// on the rayon pool. Each cell takes its terms in input order, so the sums
    /// do not depend on how the people are split.
    fn apply(&mut self) {
        if self.unapplied.is_empty() {
            return;
        }
        let num_scores = self.num_scores;
        let num_people = self.sum_scores.len() / num_scores;
        let range_people = num_people
            .div_ceil(rayon::current_num_threads().max(1) * RANGES_PER_WORKER)
            .max(MIN_PEOPLE_PER_RANGE);
        let range_cells = range_people * num_scores;
        let alleles = &self.unapplied;
        self.sum_scores
            .par_chunks_mut(range_cells)
            .zip(self.missing_counts.par_chunks_mut(range_cells))
            .enumerate()
            .for_each(|(range, (sums, missing))| {
                for allele in alleles {
                    allele.apply(range * range_people, num_scores, sums, missing);
                }
            });
        self.unapplied.clear();
    }
}

impl ScoredAllele {
    /// Adds this allele's terms for the people from `first_person` on, whose
    /// cells `sums` and `missing` hold.
    fn apply(&self, first_person: usize, num_scores: usize, sums: &mut [f64], missing: &mut [u32]) {
        let people = first_person..first_person + sums.len() / num_scores;
        match (&self.column, &self.matched_rules[..]) {
            (DosageColumn::Calls(codes), [rule]) if num_scores == 1 => {
                // A missing call's nibble selects the zero term, and it counts once.
                let terms = &self.call_terms[0];
                let shift = if rule.effect_is_ref { 4 } else { 0 };
                for ((&code, sum), count) in codes[people].iter().zip(sums).zip(missing) {
                    *sum += terms[usize::from((code >> shift) & 0x0f)];
                    *count += u32::from(code == MISSING_CALL);
                }
            }
            (DosageColumn::Calls(codes), rules) => {
                let cells = sums
                    .chunks_exact_mut(num_scores)
                    .zip(missing.chunks_exact_mut(num_scores));
                for (&code, (sums, missing)) in codes[people].iter().zip(cells) {
                    if code == MISSING_CALL {
                        for &score_index in &self.missing_scores {
                            missing[score_index] += 1;
                        }
                        continue;
                    }
                    for (rule, terms) in rules.iter().zip(&self.call_terms) {
                        let copies = if rule.effect_is_ref {
                            code >> 4
                        } else {
                            code & 0x0f
                        };
                        sums[rule.score_index] += terms[usize::from(copies)];
                    }
                }
            }
            (DosageColumn::Dosages(dosages), [rule]) if num_scores == 1 => {
                let side = usize::from(rule.effect_is_ref);
                for ((dosage, sum), count) in dosages[people].iter().zip(sums).zip(missing) {
                    if dosage[0].is_nan() {
                        *count += 1;
                    } else {
                        *sum += rule.weight * dosage[side];
                    }
                }
            }
            (DosageColumn::Dosages(dosages), _) => {
                let cells = sums
                    .chunks_exact_mut(num_scores)
                    .zip(missing.chunks_exact_mut(num_scores));
                for (dosage, (sums, missing)) in dosages[people].iter().zip(cells) {
                    if dosage[0].is_nan() {
                        for &score_index in &self.missing_scores {
                            missing[score_index] += 1;
                        }
                        continue;
                    }
                    for rule in &self.matched_rules {
                        sums[rule.score_index] +=
                            rule.weight * dosage[usize::from(rule.effect_is_ref)];
                    }
                }
            }
        }
    }
}

/// The records read so far at one position holding a rule that names no single
/// other allele. Such a rule scores the one variant at the position carrying its
/// effect allele with a listed other allele, so nothing there is scored until
/// every record at the position has been read.
#[derive(Default)]
struct PendingPosition {
    key: Option<VariantKey>,
    chromosome: String,
    /// The (REF, ALT) pair of every ALT allele read at the position, in input order.
    rows: Vec<(String, String)>,
    /// Alleles some rule may score, each with its index in `rows`.
    alleles: Vec<(usize, DecodedAllele)>,
    /// Which positions have been resolved, by the index of their first rule.
    resolved: Vec<bool>,
}

impl PendingPosition {
    /// Starts gathering the records at `key`.
    fn open(
        &mut self,
        key: VariantKey,
        chromosome: String,
        rules_by_key: &ScoreRules,
    ) -> Result<(), String> {
        if self.resolved.is_empty() {
            self.resolved.resize(rules_by_key.rules.len(), false);
        }
        let first_rule = rules_by_key.ranges[&key].0;
        if std::mem::replace(&mut self.resolved[first_rule], true) {
            return Err(format!(
                "Records at {chromosome}:{} are not adjacent in the input, and a score row there names no single other allele, so the variant it scores cannot be decided while streaming. Sort the input, for example with bcftools sort.",
                key.1
            ));
        }
        self.key = Some(key);
        self.chromosome = chromosome;
        Ok(())
    }

    /// Takes the rows and alleles of a record at the open position.
    fn push(&mut self, decoded: &mut DecodedRecord) {
        let first_row = self.rows.len();
        self.rows.append(&mut decoded.rows);
        for allele in &mut decoded.alleles[..decoded.allele_count] {
            self.alleles
                .push((first_row + allele.alt_offset, std::mem::take(allele)));
        }
    }

    /// Decides every rule at the open position over all of its rows, as Stage 3
    /// does over `.bim` rows, and adds the alleles they score to `totals`.
    fn resolve(
        &mut self,
        rules_by_key: &ScoreRules,
        score_names: &[String],
        effect_only_matches: &mut EffectOnlyMatches,
        totals: &mut ScoreTotals,
    ) -> Result<(), String> {
        let Some(key) = self.key.take() else {
            return Ok(());
        };
        let rules = rules_by_key.get(&key).unwrap_or_default();
        let decisions: Vec<OtherAlleleMatch> = rules
            .iter()
            .map(|rule| {
                let decision = resolve_other_allele(
                    rules_by_key.allele(rule.effect_allele),
                    rules_by_key.allele(rule.other_allele),
                    self.rows
                        .iter()
                        .map(|(ref_allele, alt_allele)| (ref_allele.as_str(), alt_allele.as_str())),
                );
                for _ in rules_by_key.applications(rule) {
                    effect_only_matches.record(decision, key);
                }
                decision
            })
            .collect();

        for (row, allele) in &mut self.alleles {
            let (ref_allele, alt_allele) = &self.rows[*row];
            let mut matched = Vec::new();
            for (rule, decision) in rules.iter().zip(&decisions) {
                let effect_allele = rules_by_key.allele(rule.effect_allele);
                let effect_is_ref = match *decision {
                    OtherAlleleMatch::Pair => pair_orientation(
                        effect_allele,
                        rules_by_key.allele(rule.other_allele),
                        ref_allele,
                        alt_allele,
                    ),
                    OtherAlleleMatch::EffectOnly(chosen) if chosen == *row => {
                        Some(effect_allele == ref_allele)
                    }
                    _ => None,
                };
                if let Some(effect_is_ref) = effect_is_ref {
                    matched.extend(rules_by_key.applications(rule).iter().map(|application| {
                        MatchedRule {
                            score_index: application.score_index,
                            weight: application.weight,
                            effect_is_ref,
                        }
                    }));
                }
            }
            totals.add_allele(
                merge_matched_rules(matched),
                std::mem::take(&mut allele.column),
                |score_index| ref_effect_error(&score_names[score_index], &self.chromosome, key.1),
            )?;
        }
        self.rows.clear();
        self.alleles.clear();
        Ok(())
    }
}

/// The fields of a VCF record that scoring reads, as noodles' `Record` returns them.
struct VcfFields<'r> {
    chromosome: &'r str,
    variant_start: Option<io::Result<usize>>,
    reference_bases: &'r str,
    /// The ALT alleles, empty for a missing ALT.
    alternate_bases: &'r str,
    /// FORMAT and the sample columns, empty when FORMAT is missing.
    samples: &'r str,
}

/// Decodes one VCF record line, with its newline when it has one, into
/// `decoded`. A line `split_vcf_line` cannot split is read by noodles, so a
/// line noodles cannot read fails as noodles fails.
fn decode_vcf_line(
    line: &[u8],
    context: &DecodeContext<'_>,
    decoded: &mut DecodedRecord,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if let Some(fields) = split_vcf_line(line.strip_suffix(b"\n").unwrap_or(line)) {
        return decode_scored_fields(fields, context, decoded);
    }
    let mut record = noodles_vcf::Record::default();
    VcfReader::new(line).read_record(&mut record)?;
    decode_scored_record(&record, context, decoded)
}

/// A record line's fields as noodles reads them, for a line whose bytes cannot
/// make noodles read them any differently from a split on tabs: valid UTF-8, no
/// carriage return but a last byte (which noodles drops), seven tab-terminated
/// fields, and a position that is `0` or a number above zero. Any other line
/// gives `None`.
fn split_vcf_line(line: &[u8]) -> Option<VcfFields<'_>> {
    let line = line.strip_suffix(b"\r").unwrap_or(line);
    if memchr(b'\r', line).is_some() {
        return None;
    }
    let mut rest = std::str::from_utf8(line).ok()?;
    let mut fields = [""; 7];
    for field in &mut fields {
        (*field, rest) = rest.split_once('\t')?;
    }
    // INFO ends at the next tab, and FORMAT and the samples fill the rest of the line.
    let samples = rest.split_once('\t').map_or("", |(_, samples)| samples);
    let variant_start = match fields[1] {
        "0" => None,
        position => Some(Ok(position
            .parse::<usize>()
            .ok()
            .filter(|&start| start > 0)?)),
    };
    Some(VcfFields {
        chromosome: fields[0],
        variant_start,
        reference_bases: fields[3],
        alternate_bases: if fields[4] == "." { "" } else { fields[4] },
        samples: if samples.split('\t').next() == Some(".") {
            ""
        } else {
            samples
        },
    })
}

/// Decodes the dosages a noodles `record` contributes to its matched rules into
/// `decoded`, as [`decode_scored_fields`] decodes them.
fn decode_scored_record(
    record: &noodles_vcf::Record,
    context: &DecodeContext<'_>,
    decoded: &mut DecodedRecord,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let alternate_bases = record.alternate_bases();
    let samples = record.samples();
    let fields = VcfFields {
        chromosome: record.reference_sequence_name(),
        variant_start: record
            .variant_start()
            .map(|start| start.map(|position| position.get())),
        reference_bases: record.reference_bases(),
        alternate_bases: alternate_bases.as_ref(),
        samples: samples.as_ref(),
    };
    decode_scored_fields(fields, context, decoded)
}

/// Decodes the dosages a VCF record contributes to its matched rules into
/// `decoded`, stopping at the first error a sequential scan raises for this
/// record: a malformed position, a missing dosage FORMAT field, an undecodable
/// sample, or a REF-effect rule without a complete REF dosage.
fn decode_scored_fields(
    fields: VcfFields<'_>,
    context: &DecodeContext<'_>,
    decoded: &mut DecodedRecord,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let rules_by_key = context.rules_by_key;
    let Ok(chr) = parse_chromosome_label(fields.chromosome) else {
        return Ok(());
    };
    let Some(start) = fields.variant_start else {
        return Ok(());
    };
    let pos = start? as u32;
    let Some(score_rules) = rules_by_key.get(&(chr, pos)) else {
        return Ok(());
    };
    decoded.position = pos;
    decoded.key = Some((chr, pos));
    decoded.chromosome.clear();
    decoded.chromosome.push_str(fields.chromosome);

    let alt_alleles: Vec<&str> = if fields.alternate_bases.is_empty() {
        Vec::new()
    } else {
        fields.alternate_bases.split(',').collect()
    };
    decode_rows(
        rules_by_key,
        score_rules,
        fields.reference_bases,
        &alt_alleles,
        decoded,
    );
    for (alt_offset, alt_allele) in alt_alleles.iter().enumerate() {
        let Some(matched_rules) = rules_for_allele(
            rules_by_key,
            score_rules,
            decoded.effect_only,
            fields.reference_bases,
            alt_allele,
        ) else {
            continue;
        };
        let ref_effect_rule = matched_rules
            .iter()
            .find(|rule| rule.effect_is_ref)
            .map(|rule| rule.score_index);
        let allele = decoded.next_allele(alt_offset);
        decode_vcf_column(
            fields.samples,
            alt_offset + 1,
            alt_alleles.len(),
            context.kept_indices,
            &mut allele.column,
            || {
                ref_effect_rule.map(|score_index| {
                    ref_effect_error(&context.score_names[score_index], fields.chromosome, pos)
                })
            },
        )?;
        allele.matched_rules = matched_rules;
        decoded.allele_count += 1;
    }
    Ok(())
}

/// Marks `decoded` as `effect_only` when a rule at its position names no single
/// other allele, and keeps the record's (REF, ALT) pairs for `PendingPosition`.
fn decode_rows(
    rules_by_key: &ScoreRules,
    score_rules: &[ScoreRule],
    ref_allele: &str,
    alt_alleles: &[&str],
    decoded: &mut DecodedRecord,
) {
    decoded.effect_only = score_rules
        .iter()
        .any(|rule| names_no_single_other_allele(rules_by_key.allele(rule.other_allele)));
    if decoded.effect_only {
        decoded.rows.clear();
        decoded.rows.extend(
            alt_alleles
                .iter()
                .map(|alt_allele| (ref_allele.to_string(), alt_allele.to_string())),
        );
    }
}

/// The rules scoring `(ref_allele, alt_allele)`, merged, or `None` when no rule may.
/// At an `effect_only` position the rules are matched once every record there has
/// been read, so an allele some rule may score gets no rules yet.
fn rules_for_allele(
    rules_by_key: &ScoreRules,
    score_rules: &[ScoreRule],
    effect_only: bool,
    ref_allele: &str,
    alt_allele: &str,
) -> Option<Vec<MatchedRule>> {
    if effect_only {
        let may_score = score_rules.iter().any(|rule| {
            let effect_allele = rules_by_key.allele(rule.effect_allele);
            let other_allele = rules_by_key.allele(rule.other_allele);
            match resolve_other_allele(
                effect_allele,
                other_allele,
                std::iter::once((ref_allele, alt_allele)),
            ) {
                OtherAlleleMatch::Pair => {
                    pair_orientation(effect_allele, other_allele, ref_allele, alt_allele).is_some()
                }
                OtherAlleleMatch::EffectOnly(_) => true,
                OtherAlleleMatch::SeveralRows | OtherAlleleMatch::NoRow => false,
            }
        });
        return may_score.then(Vec::new);
    }
    let matched = match_rules_for_allele(rules_by_key, score_rules, ref_allele, alt_allele);
    (!matched.is_empty()).then_some(matched)
}

/// Decodes the dosages a BCF `record` contributes to its matched rules into
/// `decoded`, as [`decode_scored_fields`] does for a VCF record.
fn decode_scored_bcf_record(
    record: &noodles_bcf::Record,
    header: &noodles_vcf::Header,
    context: &DecodeContext<'_>,
    decoded: &mut DecodedRecord,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let rules_by_key = context.rules_by_key;
    let chromosome = record.reference_sequence_name(header.string_maps())?;
    let Ok(chr) = parse_chromosome_label(chromosome) else {
        return Ok(());
    };
    let Some(start) = record.variant_start() else {
        return Ok(());
    };
    let pos = start?.get() as u32;
    let Some(score_rules) = rules_by_key.get(&(chr, pos)) else {
        return Ok(());
    };
    decoded.position = pos;
    decoded.key = Some((chr, pos));
    decoded.chromosome.clear();
    decoded.chromosome.push_str(chromosome);

    let reference_bases = record.reference_bases();
    let ref_allele = std::str::from_utf8(reference_bases.as_ref())?;
    let alternate_bases = record.alternate_bases();
    let alt_alleles = alternate_bases.iter().collect::<Result<Vec<_>, _>>()?;
    decode_rows(rules_by_key, score_rules, ref_allele, &alt_alleles, decoded);
    for (alt_offset, alt_allele) in alt_alleles.iter().enumerate() {
        let Some(matched_rules) = rules_for_allele(
            rules_by_key,
            score_rules,
            decoded.effect_only,
            ref_allele,
            alt_allele,
        ) else {
            continue;
        };
        let ref_effect_rule = matched_rules
            .iter()
            .find(|rule| rule.effect_is_ref)
            .map(|rule| rule.score_index);
        let allele = decoded.next_allele(alt_offset);
        decode_bcf_column(
            record,
            header,
            alt_offset + 1,
            alt_alleles.len(),
            context.kept_indices,
            &mut allele.column,
            || {
                ref_effect_rule.map(|score_index| {
                    ref_effect_error(&context.score_names[score_index], chromosome, pos)
                })
            },
        )?;
        allele.matched_rules = matched_rules;
        decoded.allele_count += 1;
    }
    Ok(())
}

fn ref_effect_error(score_name: &str, chromosome: &str, position: u32) -> String {
    format!(
        "Cannot score REF-effect rule for score '{}' at {}:{} without a complete REF dosage (DS requires genotype ploidy and all ALT dosages).",
        score_name, chromosome, position,
    )
}

fn load_score_rules(
    native_score_files: &[PathBuf],
    score_regions: Option<&std::collections::HashMap<String, GenomicRegion>>,
) -> Result<(Vec<String>, ScoreRules), Box<dyn Error + Send + Sync>> {
    let headers = read_score_headers(native_score_files)?;
    let mut score_names: Vec<String> = headers
        .iter()
        .flat_map(|header| header.score_names.iter().cloned())
        .collect();
    score_names.sort();
    let score_name_to_index: AHashMap<String, usize> = score_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect();
    let mut rules = ScoreRulesBuilder::default();
    let mut skipped_contigs = SkippedContigs::default();
    let mut rejected = RejectedRows::default();

    // The rules for a row are the PLINK path's (score/prepare.rs), so one score
    // file gives one answer whatever the genotype format: a row without three
    // non-empty key columns is malformed and skipped, an `N` other allele
    // pairs with no record and is skipped, a blank weight means the score does
    // not use the row, a weight that is not a finite number fails the run, and
    // weights are read as f64 so the sums are the sums of the written numbers.
    for header in &headers {
        let path = &header.path;
        let mut reader = open_text_reader(path)?;
        let mut line = String::new();
        let mut line_number = 0u64;
        // Each column's score index and region, resolved once per file rather than per row.
        let score_indices: Vec<Option<usize>> = header
            .score_names
            .iter()
            .map(|score_name| score_name_to_index.get(score_name).copied())
            .collect();
        let column_regions: Vec<Option<&GenomicRegion>> = header
            .score_names
            .iter()
            .map(|score_name| score_regions.and_then(|regions| regions.get(score_name)))
            .collect();

        while reader.read_line(&mut line)? != 0 {
            line_number += 1;
            let trimmed = line.trim_end();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                line.clear();
                continue;
            }

            let mut fields = trimmed.split('\t');
            let (variant_id, effect_allele, other_allele) =
                match (fields.next(), fields.next(), fields.next()) {
                    (Some(v), Some(e), Some(o))
                        if !v.is_empty() && !e.is_empty() && !o.is_empty() =>
                    {
                        (v, e, o)
                    }
                    _ => {
                        rejected.malformed(path, line_number);
                        line.clear();
                        continue;
                    }
                };
            if variant_id == "variant_id" {
                line.clear();
                continue;
            }
            if other_allele == "N" {
                rejected.unknown_other_allele(path, line_number);
                line.clear();
                continue;
            }

            let mut key_parts = variant_id.splitn(2, ':');
            let chr = key_parts.next().unwrap_or_default();
            let pos = key_parts.next().unwrap_or_default().trim();
            // Harmonized catalog files carry a handful of alt/random contigs that no
            // primary-assembly VCF can match. Skip those rows with a summary instead of
            // aborting a run over hundreds of otherwise-usable score files.
            let Ok(chr_index) = parse_chromosome_label(chr) else {
                skipped_contigs.record(path, line_number, chr);
                line.clear();
                continue;
            };
            let key = (
                chr_index,
                pos.parse::<u32>().map_err(|err| {
                    format!(
                        "Invalid position '{pos}' in '{}' at line {}: {}",
                        path.display(),
                        line_number,
                        err
                    )
                })?,
            );
            let applications_start = rules.applications.len();
            for (column, weight_text) in fields.enumerate() {
                let weight_text = weight_text.trim();
                if weight_text.is_empty() {
                    continue;
                }
                let Some(Some(score_index)) = score_indices.get(column).copied() else {
                    continue;
                };
                let weight = match weight_text.parse::<f64>() {
                    Ok(weight) if weight.is_finite() => weight,
                    Ok(_) => {
                        return Err(unusable_weight(weight_text, line_number, path, "not a finite number"));
                    }
                    Err(err) => {
                        return Err(unusable_weight(weight_text, line_number, path, &err.to_string()));
                    }
                };
                if let Some(Some(region)) = column_regions.get(column)
                    && !region.contains(key)
                {
                    continue;
                }

                rules.push_application(ScoreApplication {
                    score_index,
                    weight,
                });
            }
            if rules.applications.len() > applications_start {
                rules.push_row(key, effect_allele, other_allele, applications_start);
            }

            line.clear();
        }
    }

    skipped_contigs.report();
    rejected.report();

    Ok((score_names, rules.finish()))
}

/// The error for a weight field that is not a finite number: the PLINK path's
/// (`plink2 --score` stops on the same coefficients), so neither route scores
/// NaN or drops a row silently.
fn unusable_weight(
    text: &str,
    line_number: u64,
    path: &Path,
    problem: &str,
) -> Box<dyn Error + Send + Sync> {
    format!(
        "Invalid weight '{text}' on line {line_number} of score file '{}': {problem}. Weights must be finite numbers; leave the field empty for a score that does not use the variant.",
        path.display()
    )
    .into()
}

/// Native score rows skipped on their own: malformed rows and rows whose other
/// allele is `N`, counted and reported once, as the PLINK path reports them.
#[derive(Debug, Default)]
struct RejectedRows {
    malformed: u64,
    unknown_other_allele: u64,
    examples: Vec<String>,
}

impl RejectedRows {
    const MAX_EXAMPLES: usize = 5;

    fn malformed(&mut self, path: &Path, line_number: u64) {
        self.malformed += 1;
        self.example(path, line_number, "malformed");
    }

    fn unknown_other_allele(&mut self, path: &Path, line_number: u64) {
        self.unknown_other_allele += 1;
        self.example(path, line_number, "other_allele N");
    }

    fn example(&mut self, path: &Path, line_number: u64, why: &str) {
        if self.examples.len() < Self::MAX_EXAMPLES {
            self.examples
                .push(format!("{}:{line_number} ({why})", path.display()));
        }
    }

    fn report(&self) {
        if self.malformed == 0 && self.unknown_other_allele == 0 {
            return;
        }
        eprintln!(
            "> Skipped {} malformed score row(s) and {} row(s) with other_allele 'N' (e.g. {})",
            self.malformed,
            self.unknown_other_allele,
            self.examples.join(", ")
        );
    }
}/// Counts native score rows dropped for unsupported contigs, keeping a few examples
/// so a pipeline can find the offending rows without re-scanning every score file.
#[derive(Debug, Default)]
struct SkippedContigs {
    count: u64,
    examples: Vec<String>,
}

impl SkippedContigs {
    const MAX_EXAMPLES: usize = 5;

    fn record(&mut self, path: &Path, line_number: u64, chromosome: &str) {
        self.count += 1;
        if self.examples.len() < Self::MAX_EXAMPLES {
            self.examples.push(format!(
                "{}:{} (chromosome '{}')",
                path.display(),
                line_number,
                chromosome
            ));
        }
    }

    fn report(&self) {
        if self.count == 0 {
            return;
        }
        eprintln!(
            "> Warning: Skipped {} native score row(s) on unsupported contigs (expected 1-22, X, Y, or MT).",
            self.count
        );
        for example in &self.examples {
            eprintln!(">   - {example}");
        }
        if self.count > self.examples.len() as u64 {
            eprintln!(
                ">   ... and {} more.",
                self.count - self.examples.len() as u64
            );
        }
    }
}

#[derive(Debug)]
struct NativeScoreHeader {
    path: PathBuf,
    score_names: Vec<String>,
}

fn read_score_headers(
    native_score_files: &[PathBuf],
) -> Result<Vec<NativeScoreHeader>, Box<dyn Error + Send + Sync>> {
    let mut headers = Vec::with_capacity(native_score_files.len());
    let mut seen_names: AHashMap<String, PathBuf> = AHashMap::new();
    for path in native_score_files {
        let mut reader = open_text_reader(path)?;
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                return Err(format!("Score file '{}' is empty.", path.display()).into());
            }
            if !line.starts_with('#') {
                break;
            }
        }

        let header: Vec<&str> = line.trim_end().split('\t').collect();
        if header.len() < 4
            || header[0] != "variant_id"
            || header[1] != "effect_allele"
            || header[2] != "other_allele"
        {
            return Err(format!(
                "Invalid native score header in '{}'; expected variant_id/effect_allele/other_allele/score.",
                path.display()
            )
            .into());
        }

        let mut score_names = Vec::with_capacity(header.len() - 3);
        for name in &header[3..] {
            if name.is_empty() {
                return Err(format!("Empty score name in '{}'.", path.display()).into());
            }
            if let Some(existing_path) = seen_names.insert((*name).to_string(), path.clone()) {
                return Err(format!(
                    "Duplicate Score ID '{}' detected!\n  File 1: '{}'\n  File 2: '{}'\nPlease ensure each score column has a unique identifier.",
                    name,
                    existing_path.display(),
                    path.display()
                )
                .into());
            }
            score_names.push((*name).to_string());
        }
        headers.push(NativeScoreHeader {
            path: path.clone(),
            score_names,
        });
    }
    Ok(headers)
}

fn match_rules_for_allele(
    rules_by_key: &ScoreRules,
    rules: &[ScoreRule],
    ref_allele: &str,
    alt_allele: &str,
) -> Vec<MatchedRule> {
    let capacity = rules
        .iter()
        .map(|rule| rule.applications.1 - rule.applications.0)
        .sum();
    let mut matched = Vec::with_capacity(capacity);
    for rule in rules {
        let effect_allele = rules_by_key.allele(rule.effect_allele);
        let other_allele = rules_by_key.allele(rule.other_allele);
        let Some(effect_is_ref) =
            pair_orientation(effect_allele, other_allele, ref_allele, alt_allele)
        else {
            continue;
        };
        for application in rules_by_key.applications(rule) {
            matched.push(MatchedRule {
                score_index: application.score_index,
                weight: application.weight,
                effect_is_ref,
            });
        }
    }
    merge_matched_rules(matched)
}

/// Whether the effect allele is the REF, when a rule's allele pair is `(ref_allele,
/// alt_allele)` in either order.
fn pair_orientation(
    effect_allele: &str,
    other_allele: &str,
    ref_allele: &str,
    alt_allele: &str,
) -> Option<bool> {
    if effect_allele == alt_allele && other_allele == ref_allele {
        Some(false)
    } else if effect_allele == ref_allele && other_allele == alt_allele {
        Some(true)
    } else {
        None
    }
}

/// Every rule at a position scoring one allele, with the weights of rules in the same
/// score and orientation summed in rule order.
fn merge_matched_rules(mut matched: Vec<MatchedRule>) -> Vec<MatchedRule> {
    matched.sort_by_key(|rule| (rule.score_index, rule.effect_is_ref));
    let mut unique_len = 0usize;
    for read_idx in 0..matched.len() {
        let rule = matched[read_idx];
        if unique_len > 0
            && matched[unique_len - 1].score_index == rule.score_index
            && matched[unique_len - 1].effect_is_ref == rule.effect_is_ref
        {
            matched[unique_len - 1].weight += rule.weight;
        } else {
            matched[unique_len] = rule;
            unique_len += 1;
        }
    }
    matched.truncate(unique_len);
    matched
}

#[derive(Debug, Clone, Copy)]
struct DecodedAltDosage {
    alt_dosage: f64,
    ref_dosage: Option<f64>,
}

/// Visits each kept person's dosage for ALT `alt_index` in a record's samples
/// field (FORMAT, then the sample columns, as noodles' `Record::samples`
/// returns them).
fn for_each_vcf_dosage_best<F>(
    samples: &str,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    mut visit: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: FnMut(usize, Option<DecodedAltDosage>) -> Result<(), Box<dyn Error + Send + Sync>>,
{
    if samples.is_empty() {
        for out_idx in 0..kept_indices.len() {
            visit(out_idx, None)?;
        }
        return Ok(());
    }

    let mut ds_index = None;
    let mut gp_index = None;
    let mut gt_index = None;
    for (idx, sample_key) in vcf_format_keys(samples).enumerate() {
        if ds_index.is_none() && sample_key == "DS" {
            ds_index = Some(idx);
        }
        if gp_index.is_none() && sample_key == "GP" {
            gp_index = Some(idx);
        }
        if gt_index.is_none() && sample_key == key::GENOTYPE {
            gt_index = Some(idx);
        }
    }

    let gt_idx = match gt_index {
        Some(idx) => Some(idx),
        None if ds_index.is_some() || gp_index.is_some() => None,
        None => return Err("VCF record is missing GT, DS, or GP FORMAT fields.".into()),
    };
    let last_format_index = [ds_index, gp_index, gt_idx]
        .into_iter()
        .flatten()
        .max()
        .expect("at least one dosage FORMAT field was validated");

    // People are visited in output order whether the record carries exactly
    // the header's samples, fewer (the rest are missing) or more (the extra
    // columns are never decoded).
    let mut kept_cursor = 0usize;
    for (sample_idx, sample) in vcf_sample_columns(samples).enumerate() {
        while kept_cursor < kept_indices.len() && kept_indices[kept_cursor] < sample_idx {
            kept_cursor += 1;
        }
        if kept_cursor >= kept_indices.len() {
            break;
        }
        if kept_indices[kept_cursor] != sample_idx {
            continue;
        }

        let decoded = decode_vcf_sample(
            sample,
            ds_index,
            gp_index,
            gt_idx,
            last_format_index,
            alt_index,
            alt_count,
        )?;
        visit(kept_cursor, decoded)?;
        kept_cursor += 1;
    }

    while kept_cursor < kept_indices.len() {
        visit(kept_cursor, None)?;
        kept_cursor += 1;
    }

    Ok(())
}

/// The FORMAT keys of a record's samples field, split as noodles'
/// `Samples::keys` splits them: the text before the first tab (nothing without
/// a tab), on colons, with nothing after a trailing colon.
fn vcf_format_keys(samples: &str) -> impl Iterator<Item = &str> {
    let mut rest = samples.split_once('\t').map_or("", |(keys, _)| keys);
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let (name, tail) = rest.split_once(':').unwrap_or((rest, ""));
        rest = tail;
        Some(name)
    })
}

/// Decodes every kept person's dosage of ALT `alt_index` from a record's samples
/// field into `column`, as [`for_each_vcf_dosage_best`] visits them. For a
/// person whose dosage has no REF dosage, `ref_effect_error` gives the error of
/// the REF-effect rule that needs one, if a rule does.
fn decode_vcf_column(
    samples: &str,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    column: &mut DosageColumn,
    ref_effect_error: impl Fn() -> Option<String>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if vcf_gt_calls(samples, alt_index, alt_count, kept_indices, column.calls())? {
        return Ok(());
    }
    if vcf_gt_ds_dosages(
        samples,
        alt_index,
        alt_count,
        kept_indices,
        &ref_effect_error,
        column.dosages(),
    )? {
        return Ok(());
    }
    let dosages = column.dosages();
    dosages.reserve(kept_indices.len());
    for_each_vcf_dosage_best(samples, alt_index, alt_count, kept_indices, |_, dosage| {
        dosages.push(dosage_pair(dosage, &ref_effect_error)?);
        Ok(())
    })
}

/// Decodes the dosages of a record whose FORMAT is exactly GT:DS over one ALT
/// allele into `dosages`, one per kept person, as [`for_each_vcf_dosage_best`]
/// decodes them. A column holding a two-slot call and a plain decimal DS of at
/// most two copies gives that DS and two copies less it; any other column is
/// read by `decode_vcf_sample`. Gives `Ok(false)` for any other layout.
fn vcf_gt_ds_dosages(
    samples: &str,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    ref_effect_error: &impl Fn() -> Option<String>,
    dosages: &mut Vec<[f64; 2]>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let Some((key::GENOTYPE, "DS")) = samples
        .split_once('\t')
        .and_then(|(names, _)| names.split_once(':'))
    else {
        return Ok(false);
    };
    if alt_count != 1 {
        return Ok(false);
    }
    dosages.reserve(kept_indices.len());
    let columns = samples.split_once('\t').map_or("", |(_, columns)| columns);
    let bytes = columns.as_bytes();
    let mut pos = 0usize;
    let mut sample_idx = 0usize;
    for &kept_idx in kept_indices {
        while sample_idx < kept_idx && pos < bytes.len() {
            pos = memchr(b'\t', &bytes[pos..]).map_or(bytes.len(), |offset| pos + offset + 1);
            sample_idx += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        let end = memchr(b'\t', &bytes[pos..]).map_or(bytes.len(), |offset| pos + offset);
        let column = &columns[pos..end];
        pos = (end + 1).min(bytes.len());
        // Two GT slots give ploidy two, and `dosage_from_values` sums the one DS
        // value from zero, which leaves it unchanged.
        let plain = match column.as_bytes() {
            [b'0'..=b'9' | b'.', b'/' | b'|', b'0'..=b'9' | b'.', b':', ..] => {
                parse_plain_decimal(&column[4..]).filter(|&value| 2.0 - value >= -1e-6)
            }
            _ => None,
        };
        dosages.push(match plain {
            Some(value) => [value, (2.0 - value).max(0.0)],
            None => {
                let column = if column == "." { "" } else { column };
                dosage_pair(
                    decode_vcf_sample(column, Some(1), None, Some(0), 1, alt_index, alt_count)?,
                    ref_effect_error,
                )?
            }
        });
        sample_idx += 1;
    }
    dosages.resize(kept_indices.len(), [f64::NAN; 2]);
    Ok(true)
}

/// A visited dosage as `[alt, ref]`, or the REF-effect error for a dosage
/// without a REF dosage when a rule needs one.
fn dosage_pair(
    dosage: Option<DecodedAltDosage>,
    ref_effect_error: &impl Fn() -> Option<String>,
) -> Result<[f64; 2], Box<dyn Error + Send + Sync>> {
    match dosage {
        None => Ok([f64::NAN; 2]),
        Some(DecodedAltDosage {
            alt_dosage,
            ref_dosage: Some(ref_dosage),
        }) => Ok([alt_dosage, ref_dosage]),
        Some(DecodedAltDosage {
            alt_dosage,
            ref_dosage: None,
        }) => match ref_effect_error() {
            Some(message) => Err(message.into()),
            None => Ok([alt_dosage, f64::NAN]),
        },
    }
}

/// Decodes the hard calls of a record whose only dosage FORMAT field is a
/// leading GT into `codes`, one per kept person, as
/// [`for_each_vcf_dosage_best`] decodes them. Gives `Ok(false)` for any other
/// layout, or for a call holding more copies of an allele than a code holds.
fn vcf_gt_calls(
    samples: &str,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    codes: &mut Vec<u8>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    if !samples.is_empty() {
        let mut names = vcf_format_keys(samples);
        if names.next() != Some(key::GENOTYPE) || names.any(|name| name == "DS" || name == "GP") {
            return Ok(false);
        }
    }
    codes.reserve(kept_indices.len());
    let columns = samples.split_once('\t').map_or("", |(_, columns)| columns);
    let bytes = columns.as_bytes();
    // The offset of the next column, which exists while it is inside `bytes`.
    let mut pos = 0usize;
    let mut sample_idx = 0usize;
    for &kept_idx in kept_indices {
        while sample_idx < kept_idx && pos < bytes.len() {
            pos = memchr(b'\t', &bytes[pos..]).map_or(bytes.len(), |offset| pos + offset + 1);
            sample_idx += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        let code = if let Some(&[first @ b'0'..=b'9', b'/' | b'|', second @ b'0'..=b'9', b'\t']) =
            bytes.get(pos..pos + 4)
        {
            // A diploid call of single-digit alleles: the column `parse_vcf_genotype` reads most.
            pos += 4;
            let (first, second) = (usize::from(first - b'0'), usize::from(second - b'0'));
            let alt = u8::from(first == alt_index) + u8::from(second == alt_index);
            let reference = u8::from(first == 0) + u8::from(second == 0);
            alt | reference << 4
        } else {
            let end = memchr(b'\t', &bytes[pos..]).map_or(bytes.len(), |offset| pos + offset);
            let column = &columns[pos..end];
            pos = (end + 1).min(bytes.len());
            let column = if column == "." { "" } else { column };
            match decode_vcf_sample(column, None, None, Some(0), 0, alt_index, alt_count)? {
                None => MISSING_CALL,
                Some(DecodedAltDosage {
                    alt_dosage,
                    ref_dosage: Some(ref_dosage),
                }) if alt_dosage <= 14.0 && ref_dosage <= 14.0 => {
                    alt_dosage as u8 | (ref_dosage as u8) << 4
                }
                Some(_) => return Ok(false),
            }
        };
        codes.push(code);
        sample_idx += 1;
    }
    codes.resize(kept_indices.len(), MISSING_CALL);
    Ok(true)
}

/// Decodes every kept person's dosage of ALT `alt_index` from a BCF `record` into
/// `column`, as [`for_each_bcf_dosage_best`] visits them, with `ref_effect_error`
/// as in [`decode_vcf_column`].
fn decode_bcf_column(
    record: &noodles_bcf::Record,
    header: &noodles_vcf::Header,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    column: &mut DosageColumn,
    ref_effect_error: impl Fn() -> Option<String>,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    if bcf_gt_calls(record, header, alt_index, kept_indices, column.calls())? {
        return Ok(());
    }
    let dosages = column.dosages();
    dosages.reserve(kept_indices.len());
    for_each_bcf_dosage_best(
        record,
        header,
        alt_index,
        alt_count,
        kept_indices,
        |_, dosage| {
            dosages.push(dosage_pair(dosage, &ref_effect_error)?);
            Ok(())
        },
    )
}

/// Decodes the hard calls of a BCF record whose only dosage FORMAT field is an
/// Int8 GT of at most fourteen values into `codes`, one per kept person, as
/// [`for_each_bcf_dosage_best`] decodes them. Gives `Ok(false)` for any other record.
fn bcf_gt_calls(
    record: &noodles_bcf::Record,
    header: &noodles_vcf::Header,
    alt_index: usize,
    kept_indices: &[usize],
    codes: &mut Vec<u8>,
) -> Result<bool, Box<dyn Error + Send + Sync>> {
    let samples = record.samples()?;
    if samples.format_count() == 0 {
        codes.resize(kept_indices.len(), MISSING_CALL);
        return Ok(true);
    }
    let sample_count = samples.len();
    let fields = BcfDosageFields::read(
        samples.as_ref(),
        samples.format_count(),
        sample_count,
        header,
    )?;
    let (Some(gt), None, None) = (fields.gt, fields.ds, fields.gp) else {
        return Ok(false);
    };
    if gt.ty != BcfType::Int8 || gt.width > 14 {
        return Ok(false);
    }
    codes.reserve(kept_indices.len());
    for &sample_idx in kept_indices {
        if sample_idx >= sample_count {
            codes.push(MISSING_CALL);
            continue;
        }
        let (mut alt, mut reference, mut ploidy, mut missing) = (0u8, 0u8, 0u8, false);
        for &byte in &gt.src[sample_idx * gt.width..(sample_idx + 1) * gt.width] {
            // A missing, end-of-vector or reserved value ends the genotype, as it
            // ends `BcfSeries::genotype_alleles`.
            if byte as i8 <= i8::MIN + 7 {
                break;
            }
            let Some(allele) = usize::from(byte >> 1).checked_sub(1) else {
                missing = true;
                break;
            };
            alt += u8::from(allele == alt_index);
            reference += u8::from(allele == 0);
            ploidy += 1;
        }
        codes.push(if missing || ploidy == 0 {
            MISSING_CALL
        } else {
            alt | reference << 4
        });
    }
    Ok(true)
}

/// Visits each kept person's dosage for ALT `alt_index` of a BCF `record`, as
/// `for_each_vcf_dosage_best` does for a VCF record. GT, DS and GP are decoded
/// from the record's typed values under the rules `decode_vcf_sample` applies to
/// the VCF text of the same record, so both formats give the same dosages and
/// raise the same errors. A record whose DS or GP is typed as a string is read
/// through that text instead (`for_each_bcf_dosage_via_text`), and one whose
/// fields are typed unlike their header definitions, which noodles panics on,
/// is an error.
fn for_each_bcf_dosage_best<F>(
    record: &noodles_bcf::Record,
    header: &noodles_vcf::Header,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    mut visit: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: FnMut(usize, Option<DecodedAltDosage>) -> Result<(), Box<dyn Error + Send + Sync>>,
{
    let samples = record.samples()?;
    if samples.format_count() == 0 {
        for out_idx in 0..kept_indices.len() {
            visit(out_idx, None)?;
        }
        return Ok(());
    }

    let sample_count = samples.len();
    let fields = BcfDosageFields::read(
        samples.as_ref(),
        samples.format_count(),
        sample_count,
        header,
    )?;
    if fields.gt.is_none() && fields.ds.is_none() && fields.gp.is_none() {
        return Err("BCF record is missing GT, DS, or GP FORMAT fields.".into());
    }
    match fields.route(header) {
        BcfDosageRoute::Typed => {}
        BcfDosageRoute::Text => {
            return for_each_bcf_dosage_via_text(
                record,
                header,
                alt_index,
                alt_count,
                kept_indices,
                visit,
            );
        }
        BcfDosageRoute::Invalid => {
            return Err(
                "BCF GT, DS or GP FORMAT field is typed unlike its header definition.".into(),
            );
        }
    }

    for (out_idx, &sample_idx) in kept_indices.iter().enumerate() {
        let decoded = if sample_idx < sample_count {
            fields.decode_sample(sample_idx, alt_index, alt_count)?
        } else {
            None
        };
        visit(out_idx, decoded)?;
    }
    Ok(())
}

/// The value type of a BCF typed value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BcfType {
    Int8,
    Int16,
    Int32,
    Float,
    Character,
}

impl BcfType {
    fn size(self) -> usize {
        match self {
            Self::Int8 | Self::Character => 1,
            Self::Int16 => 2,
            Self::Int32 | Self::Float => 4,
        }
    }

    fn is_integer(self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32)
    }
}

/// One BCF integer or float, classified as noodles classifies it.
#[derive(Debug, Clone, Copy)]
enum BcfValue {
    Value(f64),
    Missing,
    EndOfVector,
    Reserved,
}

fn invalid_bcf_samples() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "invalid BCF FORMAT field")
}

/// Reads a typed-value descriptor: the value type (`None` for the missing type)
/// and how many values follow.
fn read_bcf_type(src: &mut &[u8]) -> io::Result<(Option<BcfType>, usize)> {
    let (&descriptor, rest) = src.split_first().ok_or_else(invalid_bcf_samples)?;
    *src = rest;
    let mut len = usize::from(descriptor >> 4);
    if len == 0x0f {
        len = usize::try_from(read_bcf_int(src)?).map_err(|_| invalid_bcf_samples())?;
    }
    let ty = match descriptor & 0x0f {
        0 => None,
        1 => Some(BcfType::Int8),
        2 => Some(BcfType::Int16),
        3 => Some(BcfType::Int32),
        5 => Some(BcfType::Float),
        7 => Some(BcfType::Character),
        _ => return Err(invalid_bcf_samples()),
    };
    Ok((ty, len))
}

/// Reads one typed integer, as a string map index or a long vector length is written.
fn read_bcf_int(src: &mut &[u8]) -> io::Result<i32> {
    let (Some(ty), 1) = read_bcf_type(src)? else {
        return Err(invalid_bcf_samples());
    };
    if !ty.is_integer() || src.len() < ty.size() {
        return Err(invalid_bcf_samples());
    }
    let (bytes, rest) = src.split_at(ty.size());
    *src = rest;
    match bcf_value(ty, bytes) {
        BcfValue::Value(value) => Ok(value as i32),
        _ => Err(invalid_bcf_samples()),
    }
}

/// The value `bytes` holds, which is at least one value of type `ty` long.
#[inline]
fn bcf_value(ty: BcfType, bytes: &[u8]) -> BcfValue {
    let (value, missing) = match ty {
        BcfType::Int8 => (i64::from(bytes[0] as i8), i64::from(i8::MIN)),
        BcfType::Int16 => (
            i64::from(i16::from_le_bytes([bytes[0], bytes[1]])),
            i64::from(i16::MIN),
        ),
        BcfType::Int32 => (
            i64::from(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
            i64::from(i32::MIN),
        ),
        BcfType::Float => {
            return match u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) {
                0x7f80_0001 => BcfValue::Missing,
                0x7f80_0002 => BcfValue::EndOfVector,
                0x7f80_0003..=0x7f80_0007 => BcfValue::Reserved,
                bits => BcfValue::Value(f64::from(f32::from_bits(bits))),
            };
        }
        BcfType::Character => unreachable!("characters are not numbers"),
    };
    match value - missing {
        0 => BcfValue::Missing,
        1 => BcfValue::EndOfVector,
        2..=7 => BcfValue::Reserved,
        _ => BcfValue::Value(value as f64),
    }
}

/// One FORMAT field of a BCF record: `width` values of type `ty` per sample.
#[derive(Debug, Clone, Copy)]
struct BcfSeries<'r> {
    ty: BcfType,
    width: usize,
    src: &'r [u8],
    /// Whether the header declares one value per sample, so only the first is read.
    scalar: bool,
}

/// What one sample's VCF text would hold for a DS or GP field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BcfFieldText {
    /// A lone '.'.
    Missing,
    /// Nothing, which reads as one missing value.
    Empty,
    /// This many comma-separated values.
    Values(usize),
}

impl BcfSeries<'_> {
    /// Value `offset` of sample `sample`'s vector.
    #[inline]
    fn value(&self, sample: usize, offset: usize) -> BcfValue {
        bcf_value(
            self.ty,
            &self.src[(sample * self.width + offset) * self.ty.size()..],
        )
    }

    /// Sample `sample`'s values as its VCF text would list them: the first value
    /// alone for a scalar field, or every value but end-of-vector padding.
    #[inline]
    fn sample_values(&self, sample: usize) -> impl Iterator<Item = BcfValue> + '_ {
        let len = if self.scalar { 1 } else { self.width };
        (0..len)
            .map(move |offset| self.value(sample, offset))
            .filter(|value| !matches!(value, BcfValue::EndOfVector))
    }

    /// What sample `sample`'s VCF text would hold for this DS or GP field. A reserved
    /// value has no text, so writing it fails, as reading a scalar field's
    /// end-of-vector value does.
    #[inline]
    fn sample_field(&self, sample: usize) -> io::Result<BcfFieldText> {
        if self.scalar {
            return match self.value(sample, 0) {
                BcfValue::Value(_) => Ok(BcfFieldText::Values(1)),
                BcfValue::Missing => Ok(BcfFieldText::Missing),
                BcfValue::EndOfVector | BcfValue::Reserved => {
                    Err(io::Error::from(io::ErrorKind::InvalidData))
                }
            };
        }
        let mut len = 0;
        let mut only_missing = true;
        for value in self.sample_values(sample) {
            match value {
                BcfValue::Reserved => return Err(io::Error::from(io::ErrorKind::InvalidData)),
                BcfValue::Value(_) => only_missing = false,
                BcfValue::Missing | BcfValue::EndOfVector => {}
            }
            len += 1;
        }
        Ok(match len {
            0 => BcfFieldText::Empty,
            1 if only_missing => BcfFieldText::Missing,
            len => BcfFieldText::Values(len),
        })
    }

    /// Sample `sample`'s GT alleles, as noodles' genotype iterator yields them: every
    /// value up to the first missing, end-of-vector or reserved one, each as its
    /// allele position, or `None` for a missing allele.
    #[inline]
    fn genotype_alleles(&self, sample: usize) -> impl Iterator<Item = Option<usize>> + '_ {
        let size = self.ty.size();
        let src = &self.src[sample * self.width * size..(sample + 1) * self.width * size];
        let ty = self.ty;
        src.chunks_exact(size)
            .take_while(move |bytes| matches!(bcf_value(ty, bytes), BcfValue::Value(_)))
            .map(move |bytes| {
                // The unsigned value, as noodles reads an Int8 genotype byte.
                let value = match ty {
                    BcfType::Int8 => usize::from(bytes[0]),
                    BcfType::Int16 => usize::from(u16::from_le_bytes([bytes[0], bytes[1]])),
                    _ => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize,
                };
                (value >> 1).checked_sub(1)
            })
    }
}

/// How `for_each_bcf_dosage_best` reads a record's GT, DS and GP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BcfDosageRoute {
    Typed,
    Text,
    Invalid,
}

/// The GT, DS and GP FORMAT fields of a BCF record, read from its typed values.
#[derive(Debug, Default)]
struct BcfDosageFields<'r> {
    gt: Option<BcfSeries<'r>>,
    ds: Option<BcfSeries<'r>>,
    gp: Option<BcfSeries<'r>>,
}

impl<'r> BcfDosageFields<'r> {
    /// Reads the `format_count` FORMAT fields in `src`, the samples section of a
    /// record holding `sample_count` samples, keeping the first GT, DS and GP.
    fn read(
        mut src: &'r [u8],
        format_count: usize,
        sample_count: usize,
        header: &noodles_vcf::Header,
    ) -> io::Result<Self> {
        let mut fields = Self::default();
        for _ in 0..format_count {
            let id = usize::try_from(read_bcf_int(&mut src)?).map_err(|_| invalid_bcf_samples())?;
            let (Some(ty), width) = read_bcf_type(&mut src)? else {
                return Err(invalid_bcf_samples());
            };
            let len = ty
                .size()
                .checked_mul(width)
                .and_then(|len| len.checked_mul(sample_count))
                .filter(|&len| len <= src.len())
                .ok_or_else(invalid_bcf_samples)?;
            let (values, rest) = src.split_at(len);
            src = rest;
            let name = header
                .string_maps()
                .strings()
                .get_index(id)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid string map ID")
                })?;
            let series = Some(BcfSeries {
                ty,
                width,
                src: values,
                scalar: header
                    .formats()
                    .get(name)
                    .is_some_and(|format| format.number() == FormatNumber::Count(1)),
            });
            if fields.ds.is_none() && name == "DS" {
                fields.ds = series;
            } else if fields.gp.is_none() && name == "GP" {
                fields.gp = series;
            } else if fields.gt.is_none() && name == key::GENOTYPE {
                fields.gt = series;
            }
        }
        Ok(fields)
    }

    /// How the fields are read: typed when every field present is numeric, typed
    /// as its header defines it and, for DS and GP, holds at least one value per
    /// sample; through text when a DS or GP is a string under a string definition
    /// and GT (if any) is the Int8 noodles reads. Anything else is invalid.
    fn route(&self, header: &noodles_vcf::Header) -> BcfDosageRoute {
        if self.gt.is_some_and(|gt| !gt.ty.is_integer()) {
            return BcfDosageRoute::Invalid;
        }
        let mut route = BcfDosageRoute::Typed;
        for (series, name) in [(self.ds, "DS"), (self.gp, "GP")] {
            let Some(series) = series else {
                continue;
            };
            let Some(format) = header.formats().get(name) else {
                return BcfDosageRoute::Invalid;
            };
            if series.width == 0 || format.number() == FormatNumber::Count(0) {
                return BcfDosageRoute::Invalid;
            }
            match (format.ty(), series.ty) {
                (FormatType::Integer, ty) if ty.is_integer() => {}
                (FormatType::Float, BcfType::Float) => {}
                (FormatType::Character | FormatType::String, BcfType::Character) => {
                    route = BcfDosageRoute::Text;
                }
                _ => return BcfDosageRoute::Invalid,
            }
        }
        if route == BcfDosageRoute::Text && self.gt.is_some_and(|gt| gt.ty != BcfType::Int8) {
            return BcfDosageRoute::Invalid;
        }
        route
    }

    /// Sample `sample`'s dosage for ALT `alt_index`, as `decode_vcf_sample` reads the
    /// sample column of the same record.
    #[inline]
    fn decode_sample(
        &self,
        sample: usize,
        alt_index: usize,
        alt_count: usize,
    ) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
        // The text holds GT, then DS, then GP, and a reserved value fails its
        // writing before anything is decoded.
        let ds_text = self.ds.map(|ds| ds.sample_field(sample)).transpose()?;
        let gp_text = self.gp.map(|gp| gp.sample_field(sample)).transpose()?;

        let ploidy = if ds_text.is_some() || gp_text.is_some() {
            self.gt.and_then(|gt| {
                u8::try_from(gt.genotype_alleles(sample).count())
                    .ok()
                    .filter(|&ploidy| ploidy > 0)
            })
        } else {
            None
        };
        if let (Some(ds), Some(text)) = (self.ds, ds_text) {
            let parsed = match text {
                BcfFieldText::Missing => None,
                BcfFieldText::Empty => {
                    dosage_from_values(std::iter::once(Ok(None)), alt_index, alt_count, ploidy)?
                }
                BcfFieldText::Values(_) => {
                    let values = ds.sample_values(sample).map(|value| match value {
                        BcfValue::Value(dosage) if !dosage.is_finite() => {
                            Err("Dosage must be finite".into())
                        }
                        BcfValue::Value(dosage) => Ok(Some(dosage)),
                        _ => Ok(None),
                    });
                    dosage_from_values(values, alt_index, alt_count, ploidy)?
                }
            };
            if parsed.is_some() {
                return Ok(parsed);
            }
        }
        if let (Some(gp), Some(text)) = (self.gp, gp_text) {
            let actual_len = match text {
                BcfFieldText::Missing => None,
                // An empty field is one value, too few for any GP layout.
                BcfFieldText::Empty => Some(1),
                BcfFieldText::Values(len) => Some(len),
            };
            if let Some(actual_len) = actual_len {
                let parts = gp.sample_values(sample).map(|value| match value {
                    BcfValue::Value(probability) => Ok(Some(probability)),
                    _ => Ok(None),
                });
                let parsed = gp_from_values(actual_len, parts, alt_index, alt_count, ploidy)?;
                if parsed.is_some() {
                    return Ok(parsed);
                }
            }
        }
        let Some(gt) = self.gt else {
            return Ok(None);
        };
        let mut dosage = 0.0f64;
        let mut ref_dosage = 0.0f64;
        let mut ploidy = 0u8;
        for allele in gt.genotype_alleles(sample) {
            let Some(allele) = allele else {
                return Ok(None);
            };
            if allele == alt_index {
                dosage += 1.0;
            }
            if allele == 0 {
                ref_dosage += 1.0;
            }
            ploidy = ploidy.checked_add(1).ok_or("genotype ploidy overflow")?;
        }
        Ok((ploidy > 0).then_some(DecodedAltDosage {
            alt_dosage: dosage,
            ref_dosage: Some(ref_dosage),
        }))
    }
}

/// `for_each_bcf_dosage_best` through text: each person's GT, DS and GP values are
/// written out as the VCF text of the same record would hold them and read by
/// `decode_vcf_sample`. A float is written in the shortest form that parses back
/// to its exact value.
fn for_each_bcf_dosage_via_text<F>(
    record: &noodles_bcf::Record,
    header: &noodles_vcf::Header,
    alt_index: usize,
    alt_count: usize,
    kept_indices: &[usize],
    mut visit: F,
) -> Result<(), Box<dyn Error + Send + Sync>>
where
    F: FnMut(usize, Option<DecodedAltDosage>) -> Result<(), Box<dyn Error + Send + Sync>>,
{
    let samples = record.samples()?;
    if samples.format_count() == 0 {
        for out_idx in 0..kept_indices.len() {
            visit(out_idx, None)?;
        }
        return Ok(());
    }

    let (mut gt_series, mut ds_series, mut gp_series) = (None, None, None);
    for result in samples.series() {
        let series = result?;
        let name = series.name(header)?;
        if ds_series.is_none() && name == "DS" {
            ds_series = Some(series);
        } else if gp_series.is_none() && name == "GP" {
            gp_series = Some(series);
        } else if gt_series.is_none() && name == key::GENOTYPE {
            gt_series = Some(series);
        }
    }
    if gt_series.is_none() && ds_series.is_none() && gp_series.is_none() {
        return Err("BCF record is missing GT, DS, or GP FORMAT fields.".into());
    }

    // Written in this order: GT, then DS, then GP.
    let fields: Vec<_> = [&gt_series, &ds_series, &gp_series]
        .into_iter()
        .flatten()
        .collect();
    let gt_index = gt_series.is_some().then_some(0);
    let ds_index = ds_series
        .is_some()
        .then_some(usize::from(gt_series.is_some()));
    let gp_index = gp_series.is_some().then(|| fields.len() - 1);
    let last_format_index = fields.len() - 1;

    let mut sample = String::new();
    for (out_idx, &sample_idx) in kept_indices.iter().enumerate() {
        sample.clear();
        let mut in_record = true;
        for (offset, series) in fields.iter().enumerate() {
            if offset > 0 {
                sample.push(':');
            }
            match series.get(header, sample_idx) {
                None => {
                    in_record = false;
                    break;
                }
                Some(None) => sample.push('.'),
                Some(Some(value)) => write_sample_value(value?, &mut sample)?,
            }
        }
        let decoded = if in_record {
            decode_vcf_sample(
                &sample,
                ds_index,
                gp_index,
                gt_index,
                last_format_index,
                alt_index,
                alt_count,
            )?
        } else {
            None
        };
        visit(out_idx, decoded)?;
    }
    Ok(())
}

/// Writes one BCF FORMAT value as a VCF sample column holds it.
fn write_sample_value(
    value: SeriesValue<'_>,
    out: &mut String,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    match value {
        SeriesValue::Integer(value) => write!(out, "{value}")?,
        SeriesValue::Float(value) => write!(out, "{}", f64::from(value))?,
        SeriesValue::Character(value) => out.push(value),
        SeriesValue::String(value) => out.push_str(value.as_ref()),
        SeriesValue::Genotype(genotype) => {
            for (offset, allele) in genotype.iter().enumerate() {
                let (position, _) = allele?;
                if offset > 0 {
                    out.push('/');
                }
                match position {
                    Some(position) => write!(out, "{position}")?,
                    None => out.push('.'),
                }
            }
        }
        SeriesValue::Array(SeriesArray::Integer(values)) => {
            for (offset, value) in values.iter().enumerate() {
                if offset > 0 {
                    out.push(',');
                }
                match value? {
                    Some(value) => write!(out, "{value}")?,
                    None => out.push('.'),
                }
            }
        }
        SeriesValue::Array(SeriesArray::Float(values)) => {
            for (offset, value) in values.iter().enumerate() {
                if offset > 0 {
                    out.push(',');
                }
                match value? {
                    Some(value) => write!(out, "{}", f64::from(value))?,
                    None => out.push('.'),
                }
            }
        }
        SeriesValue::Array(SeriesArray::Character(values)) => {
            for (offset, value) in values.iter().enumerate() {
                if offset > 0 {
                    out.push(',');
                }
                match value? {
                    Some(value) => out.push(value),
                    None => out.push('.'),
                }
            }
        }
        SeriesValue::Array(SeriesArray::String(values)) => {
            for (offset, value) in values.iter().enumerate() {
                if offset > 0 {
                    out.push(',');
                }
                match value? {
                    Some(value) => out.push_str(value.as_ref()),
                    None => out.push('.'),
                }
            }
        }
    }
    Ok(())
}

/// The sample columns of a record's samples field, split exactly as noodles'
/// `Samples::iter` splits them: after the FORMAT column, on tabs, with a lone
/// '.' read as an empty sample and nothing after a trailing tab.
fn vcf_sample_columns(samples: &str) -> impl Iterator<Item = &str> {
    let mut rest = samples.split_once('\t').map_or("", |(_, columns)| columns);
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let column = match memchr(b'\t', rest.as_bytes()) {
            Some(end) => {
                let (column, tail) = rest.split_at(end);
                rest = &tail[1..];
                column
            }
            None => std::mem::take(&mut rest),
        };
        Some(if column == "." { "" } else { column })
    })
}

#[inline]
fn decode_vcf_sample(
    sample: &str,
    ds_index: Option<usize>,
    gp_index: Option<usize>,
    gt_index: Option<usize>,
    last_format_index: usize,
    alt_index: usize,
    alt_count: usize,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    let mut ds_field = None;
    let mut gp_field = None;
    let mut gt_field = None;
    // The same pieces `sample.split(':')` yields, up to the last FORMAT field used.
    let mut remaining = Some(sample);
    for idx in 0..=last_format_index {
        let Some(rest) = remaining else {
            break;
        };
        let field = match memchr(b':', rest.as_bytes()) {
            Some(end) => {
                remaining = Some(&rest[end + 1..]);
                &rest[..end]
            }
            None => {
                remaining = None;
                rest
            }
        };
        if ds_index == Some(idx) {
            ds_field = Some(field);
        }
        if gp_index == Some(idx) {
            gp_field = Some(field);
        }
        if gt_index == Some(idx) {
            gt_field = Some(field);
        }
    }

    // REF dosage complements the sum of ALL alternate allele dosages. Taking
    // ploidy minus just the matched ALT incorrectly treats every other ALT as REF.
    let ploidy = if ds_field.is_some() || gp_field.is_some() {
        gt_field.and_then(parse_vcf_genotype_ploidy)
    } else {
        None
    };
    if let Some(value) = ds_field
        && let Some(parsed) = parse_vcf_dosage_field(value, alt_index, alt_count, ploidy)?
    {
        return Ok(Some(parsed));
    }
    if let Some(value) = gp_field
        && let Some(parsed) = parse_vcf_gp(value, alt_index, alt_count, ploidy)?
    {
        return Ok(Some(parsed));
    }
    if let Some(value) = gt_field
        && let Some(parsed) = parse_vcf_genotype(value, alt_index)?
    {
        return Ok(Some(parsed));
    }
    Ok(None)
}

fn parse_vcf_dosage_field(
    field: &str,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    if field.is_empty() || field == "." {
        return Ok(None);
    }
    dosage_from_values(
        field.split(',').map(parse_numeric_str),
        alt_index,
        alt_count,
        ploidy,
    )
}

/// The DS rules for one sample's values, in field order, each already read as a
/// finite number or `None` for a missing value.
fn dosage_from_values<I>(
    values: I,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>>
where
    I: Iterator<Item = Result<Option<f64>, Box<dyn Error + Send + Sync>>>,
{
    if alt_index == 0 || alt_index > alt_count {
        return Err("ALT allele index is out of range".into());
    }
    let mut alt_dosage = None;
    let mut total_alt_dosage = Some(0.0);
    let mut count = 0;
    for (offset, value) in values.enumerate() {
        let dosage = value?;
        if dosage.is_some_and(|dosage| dosage < 0.0) {
            return Err("DS dosage must be nonnegative".into());
        }
        if offset + 1 == alt_index {
            alt_dosage = dosage;
        }
        total_alt_dosage = total_alt_dosage.zip(dosage).map(|(sum, value)| sum + value);
        count += 1;
    }
    if count != alt_count {
        return Err(format!(
            "DS field has {count} values, expected {alt_count} alternate allele dosages"
        )
        .into());
    }
    let ref_dosage = ploidy
        .zip(total_alt_dosage)
        .map(|(ploidy, total)| f64::from(ploidy) - total);
    if ref_dosage.is_some_and(|dosage| dosage < -1e-6) {
        return Err("DS alternate dosages exceed genotype ploidy".into());
    }
    Ok(alt_dosage.map(|alt_dosage| DecodedAltDosage {
        alt_dosage,
        // Permit decimal rounding at the dosage boundary without a negative count.
        ref_dosage: ref_dosage.map(|dosage| dosage.max(0.0)),
    }))
}

fn parse_vcf_gp(
    field: &str,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    if field.is_empty() || field == "." {
        return Ok(None);
    }
    let parts = field.split(',').map(|part| {
        if part == "." {
            Ok(None)
        } else {
            Ok(Some(part.parse::<f64>()?))
        }
    });
    gp_from_values(
        field.split(',').count(),
        parts,
        alt_index,
        alt_count,
        ploidy,
    )
}

/// The GP rules for one sample's `actual_len` probabilities, in field order, each
/// already read as a number or `None` for a missing value.
fn gp_from_values<I>(
    actual_len: usize,
    mut parts: I,
    alt_index: usize,
    alt_count: usize,
    ploidy: Option<u8>,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>>
where
    I: Iterator<Item = Result<Option<f64>, Box<dyn Error + Send + Sync>>>,
{
    if alt_index == 0 || alt_index > alt_count {
        return Err(format!(
            "ALT allele index {alt_index} is out of range for {alt_count} alternate alleles"
        )
        .into());
    }

    let allele_count = alt_count.checked_add(1).ok_or("GP allele count overflow")?;
    let diploid_len = allele_count
        .checked_add(1)
        .and_then(|next| allele_count.checked_mul(next))
        .map(|n| n / 2)
        .ok_or("GP allele count overflow")?;
    let ploidy = match ploidy {
        Some(ploidy) => ploidy,
        None if actual_len == allele_count => 1,
        None if actual_len == diploid_len => 2,
        None => return Err(format!("GP field has {actual_len} values; cannot determine haploid or diploid ploidy for {alt_count} alternate alleles").into()),
    };
    let expected_len = match ploidy {
        1 => allele_count,
        2 => diploid_len,
        _ => {
            return Err(format!(
                "GP dosage decoding requires haploid or diploid genotypes, got ploidy {ploidy}"
            )
            .into());
        }
    };
    if actual_len != expected_len {
        return Err(format!("GP field has {actual_len} values, expected {expected_len} for ploidy {ploidy} and {alt_count} alternate alleles").into());
    }
    let mut dosage = 0.0f64;
    let mut ref_dosage = 0.0f64;
    for second in 0..allele_count {
        let first_count = if ploidy == 1 { 1 } else { second + 1 };
        for first in 0..first_count {
            let Some(probability) = parts.next().expect("GP cardinality was validated")? else {
                return Ok(None);
            };
            if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
                return Err("GP probabilities must be finite and between zero and one".into());
            }
            let copies =
                usize::from(ploidy == 2 && first == alt_index) + usize::from(second == alt_index);
            dosage += probability * copies as f64;
            let ref_copies = usize::from(ploidy == 2 && first == 0) + usize::from(second == 0);
            ref_dosage += probability * ref_copies as f64;
        }
    }
    Ok(Some(DecodedAltDosage {
        alt_dosage: dosage,
        ref_dosage: Some(ref_dosage),
    }))
}

fn parse_numeric_str(text: &str) -> Result<Option<f64>, Box<dyn Error + Send + Sync>> {
    if let Some(value) = parse_plain_decimal(text) {
        return Ok(Some(value));
    }
    let trimmed = text.trim();
    if trimmed.is_empty() || trimmed == "." {
        Ok(None)
    } else {
        let value = trimmed.parse::<f64>()?;
        if !value.is_finite() {
            return Err("Dosage must be finite".into());
        }
        Ok(Some(value))
    }
}

/// The value of `text` when it is a plain decimal of at most fifteen digits,
/// without sign, exponent or space: its digits as an integer divided by a power
/// of ten. Both operands are exact and IEEE division rounds correctly, so the
/// quotient is the correctly rounded value `str::parse` returns.
fn parse_plain_decimal(text: &str) -> Option<f64> {
    const POWERS_OF_TEN: [f64; 16] = [
        1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15,
    ];
    let mut mantissa = 0u64;
    let mut digits = 0usize;
    let mut fraction_digits = None;
    for &byte in text.as_bytes() {
        if byte.is_ascii_digit() {
            digits += 1;
            if digits > 15 {
                return None;
            }
            mantissa = mantissa * 10 + u64::from(byte - b'0');
            if let Some(count) = &mut fraction_digits {
                *count += 1;
            }
        } else if byte == b'.' && fraction_digits.is_none() {
            fraction_digits = Some(0usize);
        } else {
            return None;
        }
    }
    (digits > 0).then(|| mantissa as f64 / POWERS_OF_TEN[fraction_digits.unwrap_or(0)])
}

fn parse_vcf_genotype(
    field: &str,
    alt_index: usize,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    if field.is_empty() {
        return Ok(None);
    }

    let mut dosage = 0.0f64;
    let mut ref_dosage = 0.0f64;
    let mut ploidy = 0u8;
    let bytes = field.as_bytes();
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'/' | b'|' => idx += 1,
            b'.' => return Ok(None),
            b'0'..=b'9' => {
                let start = idx;
                idx += 1;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                let allele = field[start..idx].parse::<usize>()?;
                if allele == alt_index {
                    dosage += 1.0;
                }
                if allele == 0 {
                    ref_dosage += 1.0;
                }
                ploidy = ploidy.checked_add(1).ok_or("genotype ploidy overflow")?;
            }
            other => return Err(format!("unexpected byte {other} in genotype field").into()),
        }
    }

    if ploidy == 0 {
        Ok(None)
    } else {
        Ok(Some(DecodedAltDosage {
            alt_dosage: dosage,
            ref_dosage: Some(ref_dosage),
        }))
    }
}

/// Counts the allele slots in a `GT` field, including missing (`.`) slots.
///
/// Used only to recover ploidy for dosages decoded from `DS`/`GP`; a sample can be
/// missing its genotype call and still carry a usable imputed dosage.
fn parse_vcf_genotype_ploidy(field: &str) -> Option<u8> {
    let bytes = field.as_bytes();
    let mut ploidy = 0u8;
    let mut idx = 0;
    while idx < bytes.len() {
        match bytes[idx] {
            b'/' | b'|' => idx += 1,
            b'.' => {
                ploidy = ploidy.checked_add(1)?;
                idx += 1;
            }
            b'0'..=b'9' => {
                idx += 1;
                while idx < bytes.len() && bytes[idx].is_ascii_digit() {
                    idx += 1;
                }
                ploidy = ploidy.checked_add(1)?;
            }
            _ => return None,
        }
    }

    (ploidy > 0).then_some(ploidy)
}

fn resolve_keep_indices(
    keep: Option<&Path>,
    sample_names: &[String],
) -> Result<Vec<usize>, Box<dyn Error + Send + Sync>> {
    let Some(path) = keep else {
        return Ok((0..sample_names.len()).collect());
    };

    // The PLINK path's keep layouts: one IID per line, or plink2's `FID IID`
    // columns, under an optional `#FID IID` / `#IID` header.
    let by_name: AHashMap<&str, usize> = sample_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect();
    let mut requested = AHashSet::new();
    let mut missing = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with("#FID") || line.starts_with("#IID") {
            continue;
        }
        let resolved = by_name.get(line).copied().or_else(|| {
            let mut fields = line.split_whitespace();
            fields.next()?;
            by_name.get(fields.next()?).copied()
        });
        match resolved {
            Some(idx) => {
                requested.insert(idx);
            }
            None => missing.push(line.to_string()),
        }
    }

    if !missing.is_empty() {
        missing.sort();
        missing.dedup();
        return Err(format!(
            "Keep file contains sample IDs not present in VCF: {}",
            missing.join(", ")
        )
        .into());
    }

    let mut indices: Vec<usize> = requested.into_iter().collect();
    indices.sort_unstable();
    Ok(indices)
}fn open_text_reader(path: &Path) -> Result<Box<dyn BufRead>, Box<dyn Error + Send + Sync>> {
    let file = File::open(path)?;
    if path
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("gz"))
    {
        let reader: Box<dyn Read> = Box::new(MultiGzDecoder::new(file));
        Ok(Box::new(BufReader::new(reader)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

const BGZF_HEADER_LEN: usize = 18;
const BGZF_TRAILER_LEN: usize = 8;
/// Largest uncompressed payload a BGZF block may carry.
const BGZF_MAX_DATA_LEN: usize = 1 << 16;

fn is_bgzf_header(header: &[u8]) -> bool {
    header[..4] == [0x1f, 0x8b, 0x08, 0x04]
        && header[10..12] == [0x06, 0x00]
        && header[12..14] == *b"BC"
        && header[14..16] == [0x02, 0x00]
}

/// Inflates one canonical BGZF block into `block`, which is as long as the
/// block's recorded length, checking that length and the CRC32.
fn inflate_bgzf_block(
    frame: &[u8],
    decompressor: &mut Decompressor,
    block: &mut [u8],
) -> io::Result<()> {
    let invalid = |message: &str| io::Error::new(io::ErrorKind::InvalidData, message.to_string());
    let (header_and_data, trailer) = frame.split_at(frame.len() - BGZF_TRAILER_LEN);
    let crc32 = u32::from_le_bytes([trailer[0], trailer[1], trailer[2], trailer[3]]);
    let written = decompressor
        .deflate_decompress(&header_and_data[BGZF_HEADER_LEN..], block)
        .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;
    if written != block.len() {
        return Err(invalid("BGZF block is shorter than its recorded length"));
    }
    if libdeflater::crc32(block) != crc32 {
        return Err(invalid("BGZF block data checksum mismatch"));
    }
    Ok(())
}

/// Whether `score_vcf_streaming` would read this record line without error and
/// then skip it, so dropping it unread cannot change a score or an error.
/// `ascii` says the line is already known to hold only ASCII bytes.
///
/// `line` excludes its newline. The checks mirror noodles' `read_record` (valid
/// UTF-8, seven tab-terminated fields) and the scorer's own tests before a key
/// lookup: an unsupported contig, a telomeric position `0`, or a position no
/// score mentions. Anything less certain, including a carriage return in the
/// first two fields, which noodles may strip, is kept.
fn is_skippable_record(line: &[u8], ascii: bool, rules_by_key: &ScoreRules) -> bool {
    if !ascii && std::str::from_utf8(line).is_err() {
        return false;
    }
    let mut fields = line.splitn(8, |&byte| byte == b'\t');
    let (Some(chromosome), Some(position), Some(_)) = (fields.next(), fields.next(), fields.nth(5))
    else {
        return false;
    };
    if chromosome.contains(&b'\r') || position.contains(&b'\r') {
        return false;
    }
    // Fields split at tabs from valid UTF-8 are valid UTF-8.
    let (Ok(chromosome), Ok(position)) =
        (std::str::from_utf8(chromosome), std::str::from_utf8(position))
    else {
        return false;
    };
    let Ok(chr) = parse_chromosome_label(chromosome) else {
        return true;
    };
    if position == "0" {
        return true;
    }
    match position.parse::<usize>() {
        Ok(start) if start > 0 => !rules_by_key.contains_key(&(chr, start as u32)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Crc;
    use noodles_bcf::io::Reader as BcfReader;
    use std::collections::HashMap;
    use std::io::{Cursor, Write};

    #[test]
    fn multiallelic_ref_dosage_excludes_every_alternate_allele() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tAltG\tRefA\n1:100\tG\tA\t1\t0\n1:100\tA\tG\t0\t1\n",
        )
        .expect("write score");

        // Sample-major expected values: (G dosage, A dosage). In particular,
        // G/T carries zero copies of A even though its G dosage is only one.
        for (format, samples, expected) in [
            ("GT", "0/1\t0/2\t1/2", [1.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
            (
                "GT:DS",
                "0/1:0.9,0.2\t0/2:0.1,1.1\t1/2:0.8,1.2",
                [0.9, 0.9, 0.1, 0.8, 0.8, 0.0],
            ),
            (
                "GT:GP",
                "0/1:0,1,0,0,0,0\t0/2:0,0,0,1,0,0\t1/2:0,0,0,0,1,0",
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            ),
            (
                "GP",
                "0,1,0,0,0,0\t0,0,0,1,0,0\t0,0,0,0,1,0",
                [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            ),
            ("GT", "0\t1\t2", [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            (
                "GT:GP",
                "0:1,0,0\t1:0,1,0\t2:0,0,1",
                [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            ),
            ("GP", "1,0,0\t0,1,0\t0,0,1", [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        ] {
            std::fs::write(
                &vcf_path,
                format!(
                    "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n1\t100\t.\tA\tG,T\t.\tPASS\t.\t{format}\t{samples}\n"
                ),
            )
            .expect("write vcf");
            let result =
                score_vcf_streaming(&vcf_path, std::slice::from_ref(&score_path), None, None)
                    .unwrap_or_else(|err| panic!("{format} {samples}: {err}"));
            assert_eq!(result.score_names, ["AltG", "RefA"]);
            for (actual, expected) in result.sum_scores.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-12,
                    "{format} {samples}: {actual} != {expected}"
                );
            }
            assert_eq!(result.missing_counts, [0; 6]);
        }
    }

    #[test]
    fn dosage_decoding_rejects_invalid_cardinality_and_nonfinite_values() {
        for value in ["1", "0.5,0.5,0", "NaN,0", "inf,0", "-0.1,0", "1.5,1"] {
            assert!(
                parse_vcf_dosage_field(value, 1, 2, Some(2)).is_err(),
                "{value}"
            );
        }
        for (value, ploidy) in [
            ("0,1", Some(2)),
            ("0,1,0", Some(1)),
            ("NaN,0,1", None),
            ("0,-0.1,1", None),
        ] {
            assert!(parse_vcf_gp(value, 1, 1, ploidy).is_err(), "{value}");
        }
        let partial = parse_vcf_dosage_field("0.5,.", 1, 2, Some(2))
            .expect("parse partial dosage")
            .expect("selected ALT dosage");
        assert_eq!(partial.alt_dosage, 0.5);
        assert_eq!(partial.ref_dosage, None);
    }

    #[test]
    fn split_records_at_one_position_count_each_matched_allele() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &vcf_path,
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n",
        ).expect("write vcf");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tScoreA\n1:100\tG\tA\t1\n1:100\tT\tA\t10\n",
        )
        .expect("write score");
        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.sum_scores, [11.0]);
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.matched_variants, 2);
    }

    /// Writes `vcf_text` to `<stem>.vcf` and the same records as BGZF BCF to `<stem>.bcf`.
    fn write_vcf_and_bcf(dir: &Path, stem: &str, vcf_text: &str) -> (PathBuf, PathBuf) {
        use noodles_vcf::variant::io::Write as _;

        let vcf_path = dir.join(format!("{stem}.vcf"));
        let bcf_path = dir.join(format!("{stem}.bcf"));
        std::fs::write(&vcf_path, vcf_text).expect("write vcf");
        let mut reader = VcfReader::new(BufReader::new(File::open(&vcf_path).expect("open vcf")));
        let header = reader.read_header().expect("vcf header");
        let mut writer = noodles_bcf::io::Writer::new(File::create(&bcf_path).expect("create bcf"));
        writer.write_header(&header).expect("bcf header");
        let mut record = noodles_vcf::variant::RecordBuf::default();
        while reader
            .read_record_buf(&header, &mut record)
            .expect("vcf record")
            != 0
        {
            writer
                .write_variant_record(&header, &record)
                .expect("bcf record");
        }
        writer.try_finish().expect("finish bcf");
        (vcf_path, bcf_path)
    }

    #[test]
    fn rows_naming_no_single_other_allele_score_like_their_written_pairs() {
        let dir = tempfile::tempdir().expect("tempdir");
        // A simple locus, a flip, a split locus where both records carry A, a
        // multiallelic record where only the second ALT is T, and one more simple locus.
        let (vcf_path, bcf_path) = write_vcf_and_bcf(
            dir.path(),
            "cohort",
            "##fileformat=VCFv4.2\n\
             ##contig=<ID=1>\n\
             ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n\
             1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/1\n\
             1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/0\n\
             1\t300\t.\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\n\
             1\t300\t.\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1\n\
             1\t400\t.\tC\tG,T\t.\tPASS\t.\tGT\t0/2\t1/2\n\
             1\t500\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\n",
        );
        let score = |name: &str, body: &str| {
            let score_path = dir.path().join(name);
            std::fs::write(
                &score_path,
                format!("variant_id\teffect_allele\tother_allele\tS\n{body}"),
            )
            .expect("write score");
            let vcf = score_vcf_streaming(&vcf_path, std::slice::from_ref(&score_path), None, None)
                .expect("score vcf");
            let bcf = score_vcf_streaming(&bcf_path, &[score_path], None, None).expect("score bcf");
            assert_eq!(vcf.sum_scores, bcf.sum_scores, "{name}");
            assert_eq!(vcf.score_variant_counts, bcf.score_variant_counts, "{name}");
            assert_eq!(vcf.missing_counts, bcf.missing_counts, "{name}");
            vcf
        };

        let pairs = score(
            "pairs.tsv",
            "1:100\tA\tG\t0.5\n1:200\tT\tC\t-0.25\n1:400\tT\tC\t2\n1:500\tA\tG\t0.125\n",
        );
        assert_eq!(pairs.sum_scores, [2.25, 2.125]);
        assert_eq!(pairs.score_variant_counts, [4]);
        // The same weights with the other allele unknown or listed as candidates, plus an
        // ambiguous locus and one the genotypes lack.
        let effect_only = score(
            "effect_only.tsv",
            "1:100\tA\t.\t0.5\n1:200\tT\t.\t-0.25\n1:300\tA\t.\t9\n1:400\tT\tC/G\t2\n1:500\tA\tG/T\t0.125\n1:600\tA\t.\t7\n",
        );
        assert_eq!(effect_only.sum_scores, pairs.sum_scores);
        assert_eq!(effect_only.score_variant_counts, pairs.score_variant_counts);
        assert_eq!(effect_only.missing_counts, pairs.missing_counts);

        // No variant carries C at 1:100, two carry A at 1:300, and at 1:400 the only ALT
        // that is T pairs it with an unlisted allele: all dropped; the explicit pair stays.
        let skipped = score(
            "skipped.tsv",
            "1:100\tC\t.\t1\n1:300\tA\t.\t1\n1:400\tT\tA/G\t1\n1:500\tA\tG\t1\n",
        );
        assert_eq!(skipped.sum_scores, [2.0, 1.0]);
        assert_eq!(skipped.score_variant_counts, [1]);
    }

    #[test]
    fn effect_only_positions_straddling_decode_batches_score_like_pairs() {
        let dir = tempfile::tempdir().expect("tempdir");
        // Each position has a split pair of records after one unscored record, so parts
        // cut at line boundaries end between the two records of many positions.
        let positions = 256 * rayon::current_num_threads().max(1) + 3;
        let mut vcf = String::from(
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n\
             1\t50\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\t0/1\n",
        );
        let mut pairs = String::from("variant_id\teffect_allele\tother_allele\tS\n");
        let mut effect_only = pairs.clone();
        for index in 0..positions {
            let position = 100 + index;
            vcf.push_str(&format!(
                "1\t{position}\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/1\n"
            ));
            vcf.push_str(&format!(
                "1\t{position}\t.\tA\tT\t.\tPASS\t.\tGT\t0/0\t0/1\n"
            ));
            let weight = 0.25 + index as f64 * 1e-3;
            pairs.push_str(&format!(
                "1:{position}\tG\tA\t{weight}\n1:{position}\tT\tA\t1\n"
            ));
            effect_only.push_str(&format!(
                "1:{position}\tG\t.\t{weight}\n1:{position}\tT\tC/A\t1\n"
            ));
        }
        let vcf_path = dir.path().join("cohort.vcf");
        std::fs::write(&vcf_path, vcf).expect("write vcf");
        let pairs_path = dir.path().join("pairs.tsv");
        let effect_only_path = dir.path().join("effect_only.tsv");
        std::fs::write(&pairs_path, pairs).expect("write pairs");
        std::fs::write(&effect_only_path, effect_only).expect("write effect only");

        let expected = score_vcf_streaming(&vcf_path, &[pairs_path], None, None).expect("pairs");
        let actual =
            score_vcf_streaming(&vcf_path, &[effect_only_path], None, None).expect("effect only");
        assert_eq!(actual.sum_scores, expected.sum_scores);
        assert_eq!(actual.score_variant_counts, expected.score_variant_counts);
        assert_eq!(expected.score_variant_counts, [2 * positions as u32]);
    }

    #[test]
    fn effect_only_rules_refuse_records_at_their_position_that_are_not_adjacent() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("unsorted.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &vcf_path,
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n\
             1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n\
             1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t0/1\n\
             1\t100\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n",
        )
        .expect("write vcf");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tS\n1:100\tA\t.\t1\n1:200\tT\tC\t1\n",
        )
        .expect("write score");
        let error = score_vcf_streaming(&vcf_path, &[score_path], None, None)
            .expect_err("records at 1:100 are split by 1:200");
        assert!(
            error.to_string().contains("1:100 are not adjacent"),
            "{error}"
        );
    }

    #[test]
    fn ref_effect_rules_use_gt_ploidy_alongside_ds_dosage() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("imputed.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT:DS\t0/1:0.9\t1/1:1.8").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            // The effect allele is the VCF REF, so scoring needs ploidy that DS lacks.
            writeln!(score, "1:100\tA\tG\t1.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert!((result.sum_scores[0] - 1.1).abs() < 1e-9);
        assert!((result.sum_scores[1] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn unsupported_score_contigs_are_skipped_not_fatal() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "8_KI270821V1_ALT:557595\tG\tA\t2.0").expect("write");
            writeln!(score, "1:100\tG\tA\t0.5").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.sum_scores, [0.5]);
    }

    #[test]
    fn genotype_ploidy_counts_missing_and_multi_digit_alleles() {
        assert_eq!(parse_vcf_genotype_ploidy("0|1"), Some(2));
        assert_eq!(parse_vcf_genotype_ploidy("./."), Some(2));
        assert_eq!(parse_vcf_genotype_ploidy("1"), Some(1));
        assert_eq!(parse_vcf_genotype_ploidy("10/11"), Some(2));
        assert_eq!(parse_vcf_genotype_ploidy(""), None);
    }

    #[test]
    fn native_vcf_stream_scores_gt_without_conversion() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t./.").expect("write");
            writeln!(vcf, "1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t0.5").expect("write");
            writeln!(score, "1:200\tC\tT\t1.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.person_iids, vec!["s1", "s2", "s3"]);
        assert_eq!(result.score_names, vec!["ScoreA"]);
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 0, 1]);
        assert_eq!(result.sum_scores, [0.0, 1.5, 2.0]);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_keep_file_and_opposite_orientation_rows_are_scored_once() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        let keep_path = dir.path().join("keep.txt");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\t./.").expect("write");
            writeln!(vcf, "1\t200\t.\tC\tT\t.\tPASS\t.\tGT\t1/1\t0/1\t0/0\t0/0").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t0.5").expect("write");
            writeln!(score, "1:100\tA\tG\t1.0").expect("write");
            writeln!(score, "1:200\tC\tT\t1.0").expect("write");
        }

        {
            let mut keep = File::create(&keep_path).expect("create keep");
            writeln!(keep, "s2").expect("write");
            writeln!(keep, "s4").expect("write");
        }

        let result =
            score_vcf_streaming(&vcf_path, &[score_path], Some(&keep_path), None).expect("score");
        assert_eq!(result.person_iids, vec!["s2", "s4"]);
        assert_eq!(result.score_names, vec!["ScoreA"]);
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 1]);
        assert_eq!(result.sum_scores, [2.5, 2.0]);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_zero_net_matched_rows_still_count_in_denominator() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t./.").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:100\tG\tA\t-1.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.missing_counts, [0, 1]);
        assert_eq!(result.sum_scores, [0.0, 0.0]);
        assert_eq!(result.matched_variants, 1);
    }

    #[test]
    fn native_vcf_multiallelic_gt_scores_each_alt_separately() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG,T\t.\tPASS\t.\tGT\t0/1\t0/2\t1/2").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:100\tT\tA\t10.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 0, 0]);
        assert_eq!(result.sum_scores, [1.0, 10.0, 11.0]);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_multiallelic_ds_scores_alt_specific_values_without_gt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Alternate allele dosage\">"
            )
            .expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG,T\t.\tPASS\t.\tDS\t1.5,0.1\t0.2,1.6").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:100\tT\tA\t10.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [2]);
        assert_eq!(result.missing_counts, [0, 0]);
        assert!((result.sum_scores[0] - 2.5).abs() < 1e-12);
        assert!((result.sum_scores[1] - 16.2).abs() < 1e-12);
        assert_eq!(result.matched_variants, 2);
    }

    #[test]
    fn native_vcf_applies_score_regions_before_scoring() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1").expect("write");
            writeln!(vcf, "1\t200\t.\tA\tG\t.\tPASS\t.\tGT\t0/1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "1:100\tG\tA\t1.0").expect("write");
            writeln!(score, "1:200\tG\tA\t10.0").expect("write");
        }

        let mut regions = HashMap::new();
        regions.insert(
            "ScoreA".to_string(),
            GenomicRegion {
                chromosome: 1,
                start: 100,
                end: 100,
            },
        );
        let result =
            score_vcf_streaming(&vcf_path, &[score_path], None, Some(&regions)).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.sum_scores, [1.0]);
        assert_eq!(result.matched_variants, 1);
    }

    #[test]
    fn native_vcf_scores_all_native_score_columns() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
            )
            .expect("write");
            writeln!(vcf, "1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\t1/1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(
                score,
                "variant_id\teffect_allele\tother_allele\tScoreA\tScoreB"
            )
            .expect("write");
            writeln!(score, "1:100\tG\tA\t0.5\t2.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_names, vec!["ScoreA", "ScoreB"]);
        assert_eq!(result.score_variant_counts, [1, 1]);
        assert_eq!(result.sum_scores, [0.5, 2.0, 1.0, 4.0]);
    }

    #[test]
    fn native_vcf_ref_effect_haploid_gt_uses_observed_ploidy() {
        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let score_path = dir.path().join("score.gnomon.tsv");

        {
            let mut vcf = File::create(&vcf_path).expect("create vcf");
            writeln!(vcf, "##fileformat=VCFv4.2").expect("write");
            writeln!(
                vcf,
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts_ref\ts_alt"
            )
            .expect("write");
            writeln!(vcf, "X\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0\t1").expect("write");
        }

        {
            let mut score = File::create(&score_path).expect("create score");
            writeln!(score, "variant_id\teffect_allele\tother_allele\tScoreA").expect("write");
            writeln!(score, "X:100\tA\tG\t2.0").expect("write");
        }

        let result = score_vcf_streaming(&vcf_path, &[score_path], None, None).expect("score");
        assert_eq!(result.score_variant_counts, [1]);
        assert_eq!(result.missing_counts, [0, 0]);
        assert_eq!(result.sum_scores, [2.0, 0.0]);
    }

    fn bgzf_block(data: &[u8]) -> Vec<u8> {
        let mut encoder =
            flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(data).expect("deflate");
        let compressed = encoder.finish().expect("finish deflate");
        let mut crc = Crc::new();
        crc.update(data);
        let block_len = BGZF_HEADER_LEN + compressed.len() + BGZF_TRAILER_LEN;
        let mut block = vec![
            0x1f, 0x8b, 0x08, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x06, 0x00, b'B', b'C',
            0x02, 0x00,
        ];
        block.extend_from_slice(&u16::try_from(block_len - 1).expect("BSIZE").to_le_bytes());
        block.extend_from_slice(&compressed);
        block.extend_from_slice(&crc.sum().to_le_bytes());
        block.extend_from_slice(&u32::try_from(data.len()).expect("ISIZE").to_le_bytes());
        block
    }

    /// BGZF blocks of at most `block_len` uncompressed bytes, then the empty EOF block.
    fn bgzf_bytes(text: &[u8], block_len: usize) -> Vec<u8> {
        let mut bytes = Vec::new();
        for chunk in text.chunks(block_len) {
            bytes.extend_from_slice(&bgzf_block(chunk));
        }
        bytes.extend_from_slice(&bgzf_block(&[]));
        bytes
    }

    /// A cohort that exercises the prefilter: a header spanning many small
    /// blocks, scored and unscored positions, multiallelic and DS records,
    /// phased and haploid calls, telomeric and unsupported-contig records, and
    /// '#' lines after the header.
    fn prefilter_cohort() -> (String, String) {
        let samples = 40usize;
        let mut vcf = String::from(
            "##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Alternate allele dosage\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT",
        );
        for sample in 0..samples {
            vcf.push_str(&format!("\tsample_with_a_long_identifier_{sample}"));
        }
        vcf.push('\n');
        let mut score = String::from("variant_id\teffect_allele\tother_allele\tScoreA\tScoreB\n");
        let genotypes = ["0/0", "0/1", "1/1", "./.", "1|0", "0|1", "1/2", "2/2"];
        for record in 0..400u32 {
            let (chromosome, alt) = match record % 25 {
                3 => ("chrUn_KI270302v1", "G"),
                11 => ("X", "G"),
                17 => ("22", "G,T"),
                _ => ("22", "G"),
            };
            let position = if record % 40 == 5 {
                0
            } else {
                1000 + 10 * record
            };
            if record % 60 == 1 {
                vcf.push_str("#not_a_header\t1\t.\tA\tG\t.\tPASS\t.\tGT");
                vcf.push_str(&"\t0/1".repeat(samples));
                vcf.push('\n');
            }
            // Diploid dosages only: the haploid X calls cannot carry DS up to 1.8.
            let with_ds = record % 4 == 1 && alt == "G" && chromosome != "X";
            vcf.push_str(&format!(
                "{chromosome}\t{position}\t.\tA\t{alt}\t.\tPASS\t.\t{}",
                if with_ds { "GT:DS" } else { "GT" }
            ));
            for sample in 0..samples {
                let index = (record as usize * 7 + sample * 3) % genotypes.len();
                let genotype = if chromosome == "X" {
                    ["0", "1", "."][index % 3]
                } else if alt == "G" {
                    genotypes[index % 6]
                } else {
                    genotypes[index]
                };
                vcf.push('\t');
                vcf.push_str(genotype);
                if with_ds {
                    vcf.push_str(&format!(":{}", (index % 5) as f64 * 0.45));
                }
            }
            vcf.push('\n');
            if (record % 3 == 1 || record % 7 == 2) && chromosome != "chrUn_KI270302v1" {
                score.push_str(&format!(
                    "{chromosome}:{position}\tG\tA\t{}\t-0.5\n",
                    0.25 + f64::from(record) * 1e-3
                ));
                if alt == "G,T" {
                    score.push_str(&format!("{chromosome}:{position}\tA\tT\t0.75\t0.125\n"));
                }
            }
        }
        (vcf, score)
    }

    /// A BCF holding a VCF's records scores exactly like the VCF, bgzipped or not:
    /// phased, haploid and missing GT calls, DS with missing values, GP, a
    /// multiallelic record and a REF-effect rule.
    #[test]
    fn native_bcf_scores_like_the_same_vcf() {
        use noodles_vcf::variant::io::Write as _;

        let dir = tempfile::tempdir().expect("tempdir");
        let vcf_path = dir.path().join("cohort.vcf");
        let bcf_path = dir.path().join("cohort.bcf");
        let plain_bcf_path = dir.path().join("cohort.plain.bcf");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &vcf_path,
            "##fileformat=VCFv4.2\n\
             ##contig=<ID=1>\n\
             ##contig=<ID=X>\n\
             ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
             ##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"ALT dosage\">\n\
             ##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Genotype probabilities\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n\
             1\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t1/1\t./.\n\
             1\t200\t.\tC\tT,G\t.\tPASS\t.\tGT:DS\t1/2:0.75,0.5\t0/1:.,.\t2/2:0.25,1.75\n\
             1\t300\t.\tG\tA\t.\tPASS\t.\tGP\t0.25,0.5,0.25\t0.5,0.25,0.25\t0,0,1\n\
             X\t400\t.\tT\tC\t.\tPASS\t.\tGT\t1\t0/1\t0\n",
        )
        .expect("write vcf");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tScoreA\tScoreB\n\
             1:100\tG\tA\t1.0\t0.5\n\
             1:200\tT\tC\t2.0\t0.25\n\
             1:200\tG\tC\t-1.0\t3.0\n\
             1:200\tC\tT\t0.5\t-2.0\n\
             1:300\tA\tG\t1.5\t0.75\n\
             X:400\tC\tT\t4.0\t1.0\n",
        )
        .expect("write score");

        let mut reader = VcfReader::new(BufReader::new(File::open(&vcf_path).expect("open vcf")));
        let header = reader.read_header().expect("vcf header");
        let mut writer = noodles_bcf::io::Writer::new(File::create(&bcf_path).expect("create bcf"));
        writer.write_header(&header).expect("bcf header");
        let mut record = noodles_vcf::variant::RecordBuf::default();
        while reader
            .read_record_buf(&header, &mut record)
            .expect("vcf record")
            != 0
        {
            writer
                .write_variant_record(&header, &record)
                .expect("bcf record");
        }
        writer.try_finish().expect("finish bcf");
        io::copy(
            &mut MultiGzDecoder::new(File::open(&bcf_path).expect("open bcf")),
            &mut File::create(&plain_bcf_path).expect("create plain bcf"),
        )
        .expect("inflate bcf");

        // The same BCF bytes in blocks small enough to span many batches, and
        // ending in a plain gzip member that the passthrough hands to flate2.
        let plain_bcf = std::fs::read(&plain_bcf_path).expect("read plain bcf");
        let mut paths = vec![bcf_path.clone(), plain_bcf_path.clone()];
        for block_len in [1, 7, 4096] {
            let path = dir.path().join(format!("cohort.b{block_len}.bcf"));
            std::fs::write(&path, bgzf_bytes(&plain_bcf, block_len)).expect("write bgzf bcf");
            paths.push(path);
        }
        let split = plain_bcf.len() / 2;
        let mut mixed = Vec::new();
        for chunk in plain_bcf[..split].chunks(13) {
            mixed.extend_from_slice(&bgzf_block(chunk));
        }
        let mut gzip = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        gzip.write_all(&plain_bcf[split..]).expect("gzip");
        mixed.extend_from_slice(&gzip.finish().expect("finish gzip"));
        let mixed_path = dir.path().join("cohort.mixed.bcf");
        std::fs::write(&mixed_path, mixed).expect("write mixed bcf");
        paths.push(mixed_path);

        let score = |path: &Path| {
            score_vcf_streaming(path, std::slice::from_ref(&score_path), None, None)
                .unwrap_or_else(|err| panic!("{}: {err}", path.display()))
        };
        let expected = score(&vcf_path);
        assert!(expected.matched_variants > 0);

        let mut corrupt: Vec<Vec<u8>> = plain_bcf.chunks(64).map(bgzf_block).collect();
        let middle = corrupt.len() / 2;
        let crc_offset = corrupt[middle].len() - BGZF_TRAILER_LEN;
        corrupt[middle][crc_offset] ^= 0xff;
        let corrupt_path = dir.path().join("cohort.corrupt.bcf");
        std::fs::write(&corrupt_path, corrupt.concat()).expect("write corrupt bcf");
        assert!(
            score_vcf_streaming(&corrupt_path, std::slice::from_ref(&score_path), None, None)
                .is_err()
        );

        for path in &paths {
            let actual = score(path);
            assert_same_native_result(&expected, &actual, &path.display().to_string());
            assert_eq!(
                expected
                    .sum_scores
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                actual
                    .sum_scores
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{}",
                path.display()
            );
        }
    }

    /// SplitMix64 draws for the typed BCF decoding tests.
    struct Draws(u64);

    impl Draws {
        fn below(&mut self, n: usize) -> usize {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) % n as u64) as usize
        }
    }

    /// One value of a BCF FORMAT vector.
    #[derive(Debug, Clone, Copy)]
    enum BcfSlot {
        Value(f64),
        Missing,
        EndOfVector,
        Reserved,
    }

    /// `slots` as BCF typed values: type code 1, 2 or 3 for integers, 5 for floats.
    fn bcf_slot_bytes(ty: u8, slots: &[BcfSlot]) -> Vec<u8> {
        let mut out = Vec::new();
        for &slot in slots {
            if ty == 5 {
                let bits = match slot {
                    BcfSlot::Value(value) => (value as f32).to_bits(),
                    BcfSlot::Missing => 0x7f80_0001,
                    BcfSlot::EndOfVector => 0x7f80_0002,
                    BcfSlot::Reserved => 0x7f80_0005,
                };
                out.extend_from_slice(&bits.to_le_bytes());
            } else {
                let size = 1usize << (ty - 1);
                let min = -(1i64 << (8 * size - 1));
                let value = match slot {
                    BcfSlot::Value(value) => value as i64,
                    BcfSlot::Missing => min,
                    BcfSlot::EndOfVector => min + 1,
                    BcfSlot::Reserved => min + 4,
                };
                out.extend_from_slice(&value.to_le_bytes()[..size]);
            }
        }
        out
    }

    /// A BCF record at 1:100 with `alt_count` ALT alleles and no INFO. Each FORMAT
    /// field is (string map index, type code, width, every sample's typed values).
    fn raw_bcf_record(
        alt_count: usize,
        sample_count: usize,
        fields: &[(u8, u8, usize, Vec<u8>)],
    ) -> Vec<u8> {
        let mut site = Vec::new();
        site.extend_from_slice(&0i32.to_le_bytes());
        site.extend_from_slice(&99i32.to_le_bytes());
        site.extend_from_slice(&1i32.to_le_bytes());
        site.extend_from_slice(&0x7f80_0001u32.to_le_bytes());
        site.extend_from_slice(&0u16.to_le_bytes());
        site.extend_from_slice(&u16::try_from(alt_count + 1).unwrap().to_le_bytes());
        let counts =
            u32::try_from(sample_count).unwrap() | (u32::try_from(fields.len()).unwrap() << 24);
        site.extend_from_slice(&counts.to_le_bytes());
        site.extend_from_slice(&[0x17, b'.']);
        for allele in &b"ACGT"[..=alt_count] {
            site.extend_from_slice(&[0x17, *allele]);
        }
        site.push(0x00);
        let mut samples = Vec::new();
        for (id, ty, width, values) in fields {
            samples.extend_from_slice(&[0x11, *id, (u8::try_from(*width).unwrap() << 4) | ty]);
            samples.extend_from_slice(values);
        }
        let mut record = Vec::new();
        record.extend_from_slice(&u32::try_from(site.len()).unwrap().to_le_bytes());
        record.extend_from_slice(&u32::try_from(samples.len()).unwrap().to_le_bytes());
        record.extend_from_slice(&site);
        record.extend_from_slice(&samples);
        record
    }

    /// Uncompressed BCF bytes holding `header` and `records`.
    fn raw_bcf(header: &str, records: &[Vec<u8>]) -> Vec<u8> {
        let mut out = b"BCF\x02\x02".to_vec();
        out.extend_from_slice(&u32::try_from(header.len() + 1).unwrap().to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.push(0);
        for record in records {
            out.extend_from_slice(record);
        }
        out
    }

    /// A header whose string map numbers GT 1, DS 2 and GP 3.
    fn dosage_bcf_header(ds: &str, gp: &str) -> String {
        format!(
            "##fileformat=VCFv4.2\n##contig=<ID=1>\n\
             ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
             ##{ds},Description=\"Dosage\">\n\
             ##FORMAT=<ID=GP,{gp},Description=\"Genotype probabilities\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4\n"
        )
    }

    type Visits = Option<Vec<Option<(u64, Option<u64>)>>>;

    type DosageVisitor<'a> =
        dyn FnMut(usize, Option<DecodedAltDosage>) -> Result<(), Box<dyn Error + Send + Sync>> + 'a;

    /// Every person a dosage route visits, as bits, or `None` if the route fails.
    fn bcf_dosage_visits(
        route: impl FnOnce(&mut DosageVisitor<'_>) -> Result<(), Box<dyn Error + Send + Sync>>,
    ) -> Visits {
        let mut visits = Vec::new();
        let mut visit = |out_idx: usize,
                         decoded: Option<DecodedAltDosage>|
         -> Result<(), Box<dyn Error + Send + Sync>> {
            assert_eq!(out_idx, visits.len());
            visits.push(decoded.map(|d| (d.alt_dosage.to_bits(), d.ref_dosage.map(f64::to_bits))));
            Ok(())
        };
        let result = route(&mut visit);
        result.ok().map(|()| visits)
    }

    /// Reads `records` back under `header`.
    fn read_raw_bcf(
        header: &str,
        records: &[Vec<u8>],
    ) -> (noodles_vcf::Header, Vec<noodles_bcf::Record>) {
        let mut reader = BcfReader::from(Cursor::new(raw_bcf(header, records)));
        let header = reader.read_header().expect("bcf header");
        let records = records
            .iter()
            .map(|_| {
                let mut record = noodles_bcf::Record::default();
                assert!(reader.read_record(&mut record).expect("bcf record") > 0);
                record
            })
            .collect();
        (header, records)
    }

    /// The typed BCF decoder visits every person as reading the record's VCF text
    /// does: GT, DS and GP in any order and any integer or float width, missing,
    /// end-of-vector and reserved values, NaN and infinite dosages, scalar and
    /// vector definitions, out-of-range ALT indices and people past the record's
    /// samples. GT wider than Int8, which noodles cannot read, decodes like the same
    /// alleles in Int8.
    #[test]
    fn typed_bcf_dosages_match_the_text_route() {
        let mut draws = Draws(0x5eed_bcf0);
        let kept = [0, 1, 3, 5];
        for (ds, gp) in [
            ("FORMAT=<ID=DS,Number=A,Type=Float", "Number=G,Type=Float"),
            ("FORMAT=<ID=DS,Number=1,Type=Float", "Number=3,Type=Float"),
            ("FORMAT=<ID=DS,Number=A,Type=Integer", "Number=G,Type=Float"),
            ("FORMAT=<ID=DS,Number=G,Type=Float", "Number=A,Type=Float"),
        ] {
            let header = dosage_bcf_header(ds, gp);
            let ds_scalar = ds.contains("Number=1");
            let ds_integer = ds.contains("Integer");
            let mut records = Vec::new();
            let mut alt = Vec::new();
            for _ in 0..600 {
                let alt_count = 1 + draws.below(3);
                let alt_index = match draws.below(10) {
                    0 => 0,
                    1 => alt_count + 1,
                    _ => 1 + draws.below(alt_count),
                };
                let sample_count = [4, 4, 4, 4, 2, 0][draws.below(6)];
                let slots = |draws: &mut Draws, width: usize, values: &[f64], scalar: bool| {
                    let mut slots: Vec<BcfSlot> = (0..width * sample_count)
                        .map(|_| match draws.below(40) {
                            0 | 1 => BcfSlot::Missing,
                            2 => BcfSlot::EndOfVector,
                            3 => BcfSlot::Reserved,
                            _ => BcfSlot::Value(values[draws.below(values.len())]),
                        })
                        .collect();
                    if scalar {
                        // noodles panics on a scalar's end-of-vector or reserved value.
                        for sample in slots.chunks_mut(width) {
                            if !matches!(sample[0], BcfSlot::Value(_)) {
                                sample[0] = BcfSlot::Missing;
                            }
                        }
                    }
                    slots
                };
                let gt_width = 1 + draws.below(3);
                let gt: Vec<BcfSlot> = (0..gt_width * sample_count)
                    .map(|_| match draws.below(20) {
                        0 => BcfSlot::Missing,
                        1 | 2 => BcfSlot::EndOfVector,
                        3 => BcfSlot::Reserved,
                        4 | 5 => BcfSlot::Value(draws.below(2) as f64),
                        _ => BcfSlot::Value(
                            ((draws.below(alt_count + 2) + 1) * 2 + draws.below(2)) as f64,
                        ),
                    })
                    .collect();
                // Mostly the widths the definitions expect, so many records decode. A
                // scalar DS keeps width 1: noodles reads a `Number=1` float through a
                // fixed 4-byte slice and panics on any other stored width.
                let ds_width = if ds_scalar {
                    1
                } else if draws.below(4) > 0 {
                    alt_count
                } else {
                    1 + draws.below(3)
                };
                let ds_values: &[f64] = if ds_integer {
                    &[0.0, 1.0, 1.0, 0.0, 2.0, 3.0, -1.0]
                } else {
                    &[
                        0.0,
                        0.25,
                        0.5,
                        1.0,
                        0.75,
                        2.0,
                        2.5,
                        -0.25,
                        1e-7,
                        f64::NAN,
                        f64::INFINITY,
                    ]
                };
                let ds_type = if ds_integer {
                    1 + draws.below(3) as u8
                } else {
                    5
                };
                let ds =
                    bcf_slot_bytes(ds_type, &slots(&mut draws, ds_width, ds_values, ds_scalar));
                let gp_width = if draws.below(4) > 0 {
                    (alt_count + 1) * (alt_count + 2) / 2
                } else {
                    1 + draws.below(6)
                };
                let gp_values: &[f64] = &[0.0, 0.25, 0.5, 1.0, 0.125, 1.25, -0.125, f64::NAN];
                let gp = bcf_slot_bytes(5, &slots(&mut draws, gp_width, gp_values, false));
                let mut fields = Vec::new();
                for field in 0..3 {
                    if draws.below(4) > 0 {
                        fields.insert(draws.below(fields.len() + 1), field);
                    }
                }
                if draws.below(8) == 0 {
                    // A second DS, which neither route reads, and a FORMAT key under
                    // PASS's string map index, which neither route recognizes.
                    fields.push(1);
                    fields.insert(draws.below(fields.len() + 1), 3);
                }
                // The same record with GT in Int8 (which the text route reads), Int16 and Int32.
                for gt_type in [1u8, 2, 3] {
                    let encoded: Vec<_> = fields
                        .iter()
                        .map(|&field| match field {
                            0 => (1, gt_type, gt_width, bcf_slot_bytes(gt_type, &gt)),
                            1 => (2, ds_type, ds_width, ds.clone()),
                            2 => (3, 5, gp_width, gp.clone()),
                            _ => (0, 1, 1, vec![1; sample_count]),
                        })
                        .collect();
                    records.push(raw_bcf_record(alt_count, sample_count, &encoded));
                    alt.push((alt_index, alt_count));
                }
            }
            let (header, records) = read_raw_bcf(&header, &records);
            let (mut with_dosage, mut failed) = (0, 0);
            for (index, same_record) in records.chunks(3).enumerate() {
                let (alt_index, alt_count) = alt[3 * index];
                let text = bcf_dosage_visits(|visit| {
                    for_each_bcf_dosage_via_text(
                        &same_record[0],
                        &header,
                        alt_index,
                        alt_count,
                        &kept,
                        visit,
                    )
                });
                for (gt_type, record) in same_record.iter().enumerate() {
                    let typed = bcf_dosage_visits(|visit| {
                        for_each_bcf_dosage_best(
                            record, &header, alt_index, alt_count, &kept, visit,
                        )
                    });
                    assert_eq!(
                        typed,
                        text,
                        "{ds} {gp}: record {index}, GT type {}",
                        gt_type + 1
                    );
                }
                match &text {
                    None => failed += 1,
                    Some(visits) if visits.iter().any(Option::is_some) => with_dosage += 1,
                    Some(_) => {}
                }
            }
            assert!(
                with_dosage >= 30 && failed >= 30,
                "{ds} {gp}: {with_dosage} records decode a dosage, {failed} fail"
            );
        }
    }

    /// FORMAT fields typed unlike their header definitions are errors, where noodles
    /// panics; a string-typed DS is still read through its text.
    #[test]
    fn bcf_dosage_fields_typed_unlike_the_header_are_errors() {
        fn float(values: &[f64]) -> Vec<u8> {
            let slots: Vec<_> = values.iter().map(|&v| BcfSlot::Value(v)).collect();
            bcf_slot_bytes(5, &slots)
        }
        let gt = bcf_slot_bytes(1, &[BcfSlot::Value(2.0), BcfSlot::Value(4.0)]);
        let kept = [0];
        for (ds, fields, expected) in [
            // DS declared Float, typed Int8.
            (
                "FORMAT=<ID=DS,Number=A,Type=Float",
                vec![(2, 1, 1, vec![1])],
                None,
            ),
            // A scalar DS whose value is end-of-vector.
            (
                "FORMAT=<ID=DS,Number=1,Type=Float",
                vec![(2, 5, 1, bcf_slot_bytes(5, &[BcfSlot::EndOfVector]))],
                None,
            ),
            // DS with no values.
            (
                "FORMAT=<ID=DS,Number=A,Type=Float",
                vec![(2, 5, 0, Vec::new())],
                None,
            ),
            // GT typed as a float.
            (
                "FORMAT=<ID=DS,Number=A,Type=Float",
                vec![(1, 5, 2, float(&[2.0, 4.0]))],
                None,
            ),
            // DS defined only as INFO.
            (
                "INFO=<ID=DS,Number=A,Type=Float",
                vec![(1, 1, 2, gt.clone()), (2, 5, 1, float(&[0.5]))],
                None,
            ),
            // DS as a string, read through its text: 0/1 with DS 0.5.
            (
                "FORMAT=<ID=DS,Number=1,Type=String",
                vec![(1, 1, 2, gt.clone()), (2, 7, 3, b"0.5".to_vec())],
                Some(vec![Some((0.5f64.to_bits(), Some(1.5f64.to_bits())))]),
            ),
        ] {
            let (header, records) = read_raw_bcf(
                &dosage_bcf_header(ds, "Number=G,Type=Float"),
                &[raw_bcf_record(1, 1, &fields)],
            );
            let typed = bcf_dosage_visits(|visit| {
                for_each_bcf_dosage_best(&records[0], &header, 1, 1, &kept, visit)
            });
            assert_eq!(typed, expected, "{ds}: {fields:?}");
        }
    }

    fn assert_same_native_result(
        expected: &NativeVcfScoreResult,
        actual: &NativeVcfScoreResult,
        context: &str,
    ) {
        assert_eq!(expected.person_iids, actual.person_iids, "{context}");
        assert_eq!(expected.score_names, actual.score_names, "{context}");
        assert_eq!(
            expected.score_variant_counts, actual.score_variant_counts,
            "{context}"
        );
        assert_eq!(expected.missing_counts, actual.missing_counts, "{context}");
        assert_eq!(
            expected.matched_variants, actual.matched_variants,
            "{context}"
        );
        let bits = |values: &[f64]| {
            values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(
            bits(&expected.sum_scores),
            bits(&actual.sum_scores),
            "{context}"
        );
    }

    #[test]
    fn bgzf_prefilter_scores_match_plain_vcf_at_any_block_size() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (vcf, score) = prefilter_cohort();
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(&score_path, &score).expect("write score");

        for text in [vcf.as_str(), vcf.trim_end_matches('\n')] {
            let plain_path = dir.path().join("cohort.vcf");
            std::fs::write(&plain_path, text).expect("write vcf");
            let expected =
                score_vcf_streaming(&plain_path, std::slice::from_ref(&score_path), None, None)
                    .expect("plain score");
            assert!(expected.matched_variants > 100);

            for block_len in [1, 2, 7, 100, 4096, 65536] {
                let bgzf_path = dir.path().join("cohort.vcf.gz");
                std::fs::write(&bgzf_path, bgzf_bytes(text.as_bytes(), block_len))
                    .expect("write bgzf");
                let actual =
                    score_vcf_streaming(&bgzf_path, std::slice::from_ref(&score_path), None, None)
                        .expect("bgzf score");
                assert_same_native_result(
                    &expected,
                    &actual,
                    &format!(
                        "block_len={block_len} trailing_newline={}",
                        text.ends_with('\n')
                    ),
                );
            }
        }
    }

    #[test]
    fn bgzf_prefilter_rejects_the_records_plain_vcf_rejects() {
        let dir = tempfile::tempdir().expect("tempdir");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tScoreA\n22:100\tG\tA\t1\n",
        )
        .expect("write score");
        let header =
            b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";
        let unscored = b"22\t200\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n";
        let scored = b"22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n";
        let malformed: [&[u8]; 6] = [
            b"22\tnot_a_position\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n",
            b"22\t00\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n",
            b"22\t300\t.\tA\tG\n",
            b"\n",
            b"#22\t300\n",
            b"22\t300\t.\tA\tG\t.\tPASS\t.\tGT\t0/\xff\n",
        ];
        for bad in malformed {
            let context = String::from_utf8_lossy(bad).into_owned();
            let mut vcf = header.to_vec();
            vcf.extend_from_slice(unscored);
            vcf.extend_from_slice(bad);
            vcf.extend_from_slice(scored);

            let plain_path = dir.path().join("malformed.vcf");
            std::fs::write(&plain_path, &vcf).expect("write vcf");
            assert!(
                score_vcf_streaming(&plain_path, std::slice::from_ref(&score_path), None, None)
                    .is_err(),
                "plain {context:?}"
            );
            for block_len in [3, 65536] {
                let bgzf_path = dir.path().join("malformed.vcf.gz");
                std::fs::write(&bgzf_path, bgzf_bytes(&vcf, block_len)).expect("write bgzf");
                assert!(
                    score_vcf_streaming(&bgzf_path, std::slice::from_ref(&score_path), None, None)
                        .is_err(),
                    "bgzf block_len={block_len} {context:?}"
                );
            }
        }
    }

    #[test]
    fn plain_gzip_members_are_decoded_without_the_prefilter() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (vcf, score) = prefilter_cohort();
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(&score_path, &score).expect("write score");
        let plain_path = dir.path().join("cohort.vcf");
        std::fs::write(&plain_path, &vcf).expect("write vcf");
        let expected =
            score_vcf_streaming(&plain_path, std::slice::from_ref(&score_path), None, None)
                .expect("plain score");

        let gzip = |text: &[u8]| {
            let mut encoder =
                flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(text).expect("gzip");
            encoder.finish().expect("finish gzip")
        };
        let split = vcf.len() / 2;
        let mut mixed = Vec::new();
        for chunk in vcf.as_bytes()[..split].chunks(97) {
            mixed.extend_from_slice(&bgzf_block(chunk));
        }
        mixed.extend_from_slice(&gzip(&vcf.as_bytes()[split..]));

        for (label, bytes) in [("gzip", gzip(vcf.as_bytes())), ("bgzf then gzip", mixed)] {
            let path = dir.path().join("fallback.vcf.gz");
            std::fs::write(&path, &bytes).expect("write gzip");
            let actual = score_vcf_streaming(&path, std::slice::from_ref(&score_path), None, None)
                .expect("gzip score");
            assert_same_native_result(&expected, &actual, label);
        }
    }

    #[test]
    fn corrupt_or_truncated_bgzf_streams_are_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let (vcf, score) = prefilter_cohort();
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(&score_path, &score).expect("write score");

        let blocks: Vec<Vec<u8>> = vcf.as_bytes().chunks(512).map(bgzf_block).collect();
        let mut corrupt = blocks.clone();
        let middle = corrupt.len() / 2;
        let crc_offset = corrupt[middle].len() - BGZF_TRAILER_LEN;
        corrupt[middle][crc_offset] ^= 0xff;
        let mut truncated = blocks.concat();
        truncated.truncate(truncated.len() - 100);

        for (label, bytes) in [("corrupt", corrupt.concat()), ("truncated", truncated)] {
            let mut sink = Vec::new();
            assert!(
                MultiGzDecoder::new(&bytes[..])
                    .read_to_end(&mut sink)
                    .is_err(),
                "flate2 also rejects the {label} stream"
            );
            let path = dir.path().join("broken.vcf.gz");
            std::fs::write(&path, &bytes).expect("write bgzf");
            assert!(
                score_vcf_streaming(&path, std::slice::from_ref(&score_path), None, None).is_err(),
                "{label}"
            );
        }
    }

    #[test]
    fn skippable_records_are_read_cleanly_and_skipped_by_the_scorer() {
        let mut builder = ScoreRulesBuilder::default();
        builder.push_application(ScoreApplication {
            score_index: 0,
            weight: 1.0,
        });
        builder.push_row((22, 100), "G", "A", 0);
        let rules_by_key = builder.finish();
        let cases: [(&[u8], bool); 13] = [
            (b"22\t100\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"22\t101\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"chr22\t101\t.\tA\tG\t.\tPASS\t.", true),
            (b"22\t101\t.\tA\tG\t.\tPASS", false),
            (b"22\t0\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"22\t00\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"22\t+101\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"22\tabc\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"HLA-A*01:01\tabc\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", true),
            (b"#22\t101\t.\tA\tG\t.\tPASS\t.", true),
            (b"22\r\t101\t.\tA\tG\t.\tPASS\t.\tGT\t0/1", false),
            (b"22\t101\t.\tA\tG\t.\tPASS\t.\tGT\t0/\xff", false),
            (b"", false),
        ];
        let header =
            b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n";
        for (line, expected) in cases {
            let context = String::from_utf8_lossy(line).into_owned();
            assert_eq!(
                is_skippable_record(line, false, &rules_by_key),
                expected,
                "{context:?}"
            );
            if !expected {
                continue;
            }
            // Dropping a line is only sound if noodles reads it and the scorer
            // skips it. The prefilter only judges lines after the first record,
            // so one precedes it here too.
            let mut stream = header.to_vec();
            stream.extend_from_slice(b"22\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n");
            stream.extend_from_slice(line);
            stream.push(b'\n');
            let mut reader = VcfReader::new(&stream[..]);
            reader.read_header().expect("header");
            let mut record = noodles_vcf::Record::default();
            reader.read_record(&mut record).expect("first record");
            reader
                .read_record(&mut record)
                .unwrap_or_else(|err| panic!("{context:?}: {err}"));
            let skipped = match parse_chromosome_label(record.reference_sequence_name()) {
                Err(_) => true,
                Ok(chr) => record.variant_start().is_none_or(|start| {
                    let start = start.unwrap_or_else(|err| panic!("{context:?}: {err}"));
                    !rules_by_key.contains_key(&(chr, start.get() as u32))
                }),
            };
            assert!(skipped, "{context:?}");
        }
    }

    #[test]
    fn batched_decoding_reports_the_first_error_a_sequential_scan_raises() {
        let dir = tempfile::tempdir().expect("tempdir");
        let score_path = dir.path().join("score.gnomon.tsv");
        std::fs::write(
            &score_path,
            "variant_id\teffect_allele\tother_allele\tScoreA\n1:100\tA\tG\t1\n1:200\tG\tA\t1\n",
        )
        .expect("write score");
        let header =
            "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n";
        for (records, expected) in [
            // The REF-effect rule has no ploidy at s1, before s2's DS is decoded.
            (
                "1\t100\t.\tA\tG\t.\tPASS\t.\tDS\t0.5\tbad\n",
                "Cannot score REF-effect rule",
            ),
            // With the order reversed, the undecodable DS comes first.
            (
                "1\t100\t.\tA\tG\t.\tPASS\t.\tDS\tbad\t0.5\n",
                "invalid float literal",
            ),
            // An earlier record's error wins over a later record's.
            (
                "1\t200\t.\tA\tG\t.\tPASS\t.\tGT\t0/x\t0/1\n1\t100\t.\tA\tG\t.\tPASS\t.\tDS\t0.5\t0.5\n",
                "unexpected byte",
            ),
        ] {
            let vcf_path = dir.path().join("errors.vcf");
            std::fs::write(&vcf_path, format!("{header}{records}")).expect("write vcf");
            let err = score_vcf_streaming(&vcf_path, std::slice::from_ref(&score_path), None, None)
                .expect_err("malformed records");
            assert!(err.to_string().contains(expected), "{records:?}: {err}");
        }
    }

    #[test]
    fn sample_columns_split_as_noodles_splits_them() {
        let header = b"##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n";
        for samples in [
            "GT\t0/1\t1|1\t./.",
            "GT\t0/1\t\t1/1",
            "GT\t0/1\t",
            "GT\t\t",
            "GT\t.\t0/1",
            "GT",
            "GT:DS\t0/1:0.5\t.:.\t1",
        ] {
            let mut stream = header.to_vec();
            stream
                .extend_from_slice(format!("22\t100\t.\tA\tG\t.\tPASS\t.\t{samples}\n").as_bytes());
            let mut reader = VcfReader::new(&stream[..]);
            reader.read_header().expect("header");
            let mut record = noodles_vcf::Record::default();
            reader.read_record(&mut record).expect("record");
            let expected: Vec<String> = record
                .samples()
                .iter()
                .map(|sample| sample.as_ref().to_string())
                .collect();
            let samples = record.samples();
            let actual: Vec<&str> = vcf_sample_columns(samples.as_ref()).collect();
            assert_eq!(actual, expected, "{samples:?}");
        }
    }
}
