use crate::score::cells::{
    add_limbs, compare_limbs, limbs_of, round_decimal, round_long, shift_decimal, subtract_limbs,
};
use crate::score::exact::{FixedPoint, shortest_decimal_hinted};
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
use std::simd::cmp::{SimdOrd, SimdPartialEq, SimdPartialOrd};
use std::simd::num::{SimdInt, SimdUint};
use std::simd::{Mask, Select, i64x8, simd_swizzle, u8x8, u8x16, u8x32, u8x64, u64x8};

mod crc;
mod dosage;
mod stream;

use dosage::{Dose, diploid_reference, dosage_from_values, gp_from_values, parse_dose, plain_decimal};

#[derive(Debug)]
pub struct NativeVcfScoreResult {
    pub person_iids: Vec<String>,
    pub score_names: Vec<String>,
    pub score_variant_counts: Vec<u32>,
    pub missing_counts: Vec<u32>,
    pub matched_variants: usize,
    /// Each person's exact sum of each score, at `10^places` of that score.
    cells: Vec<i128>,
    places: Vec<u32>,
    /// The wide parts of the cells whose value left i128.
    spills: AHashMap<usize, Wide>,
}

impl NativeVcfScoreResult {
    /// The correctly rounded sum of cell `cell`, laid out person × score.
    pub fn sum(&self, cell: usize) -> f64 {
        self.round(cell, 1)
    }

    /// The correctly rounded average of cell `cell` over `used` variants; 0 when none were used.
    pub fn average(&self, cell: usize, used: u32) -> f64 {
        if used == 0 { 0.0 } else { self.round(cell, used) }
    }

    /// Every cell's correctly rounded sum, person × score.
    pub fn sums(&self) -> Vec<f64> {
        (0..self.cells.len()).map(|cell| self.sum(cell)).collect()
    }

    fn round(&self, cell: usize, divisor: u32) -> f64 {
        let places = self.places[cell % self.score_names.len()];
        match self.spills.get(&cell) {
            None => {
                let value = self.cells[cell];
                let fixed = 5u128
                    .checked_pow(places)
                    .filter(|scale| scale.checked_mul(u128::from(u32::MAX)).is_some())
                    .map(|scale| FixedPoint {
                        exp: -(places as i32),
                        scale,
                    });
                match fixed {
                    Some(fixed) => fixed.quotient(value, divisor),
                    None => round_long(&[(value, places as i32)], u128::from(divisor)),
                }
            }
            Some(wide) => {
                let mut total = wide.clone();
                total.add(&Wide::of(self.cells[cell]));
                round_decimal(total.negative, &total.limbs, places as i32, u128::from(divisor))
            }
        }
    }
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
    /// The weight's shortest round-trip decimal: `digits × 10^exponent`.
    digits: i64,
    exponent: i32,
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
    /// The weight's shortest round-trip decimal: `digits × 10^exponent`.
    digits: i64,
    exponent: i32,
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
    /// Per rule, whether an allele taken so far matched it first among its position's rules:
    /// a second such allele is a second record of the same allele pair.
    claimed: Vec<bool>,
}

impl<'a> RecordAccumulator<'a> {
    fn new(rules_by_key: &'a ScoreRules, score_names: &'a [String], num_people: usize) -> Self {
        Self {
            rules_by_key,
            score_names,
            totals: ScoreTotals::new(num_people, score_names.len(), rules_by_key),
            pending: PendingPosition::default(),
            effect_only_matches: EffectOnlyMatches::default(),
            claimed: vec![false; rules_by_key.rules.len()],
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
            if std::mem::replace(&mut self.claimed[allele.first_rule], true) {
                let rule = &rules_by_key.rules[allele.first_rule];
                return Err(repeated_pair_error(
                    &decoded.chromosome,
                    decoded.position,
                    rules_by_key.allele(rule.effect_allele),
                    rules_by_key.allele(rule.other_allele),
                )
                .into());
            }
            self.totals.add_allele(
                std::mem::take(&mut allele.matched_rules),
                std::mem::take(&mut allele.column),
                allele.scale.take(),
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
        self.totals.finish();
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
        cells,
        missing_counts,
        score_variant_counts,
        weight_places,
        dosage_places,
        spills,
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
        missing_counts,
        matched_variants,
        cells,
        places: weight_places
            .iter()
            .zip(&dosage_places)
            .map(|(&weight, &dosage)| weight as u32 + u32::from(dosage))
            .collect(),
        spills: spills.into_iter().flatten().collect(),
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
    /// The first rule the allele matched, as an index into every position's rules; the allele
    /// pair it names is the allele's. Unused at an `effect_only` position.
    first_rule: usize,
    /// One dosage per kept person, in output order.
    column: DosageColumn,
    /// The column's scale, once it has been normalized where it was decoded.
    scale: Option<ColumnScale>,
}

/// A decoded column's decimal places and largest doses.
#[derive(Clone, Copy, Debug, Default)]
struct ColumnScale {
    /// The most decimal places any dose has.
    places: u8,
    /// The largest dose of the ALT allele and of REF, at `places`, with every dose at `places`;
    /// for hard calls, the most copies. `None` when a dose leaves i64 at `places`.
    largest: Option<[u128; 2]>,
    /// Whether some person has a dosage without a REF dosage.
    incomplete_ref: bool,
}

/// The code of a person without a hard call in `DosageColumn::Calls`. No real
/// code reaches it: a code holds at most fourteen copies of each allele.
const MISSING_CALL: u8 = 0xff;

/// Every kept person's dosage of one ALT allele, in output order.
enum DosageColumn {
    /// Hard calls, each `alt | ref << 4` allele copies, or `MISSING_CALL`.
    Calls(Vec<u8>),
    /// `[alt, ref]` exact dosages. A missing ALT dosage is a missing dosage, and a missing
    /// REF dosage one that genotype ploidy and every ALT dosage could not complete.
    Dosages(Doses),
}

impl Default for DosageColumn {
    fn default() -> Self {
        Self::Calls(Vec::new())
    }
}

/// Every kept person's `[alt, ref]` exact dosages, with digits and places held apart so that
/// adding a column reads sixteen bytes a person.
#[derive(Default)]
struct Doses {
    digits: Vec<[i64; 2]>,
    /// Each dose's decimal places; u8::MAX, with zero digits, for a missing dose.
    places: Vec<[u8; 2]>,
}

impl Doses {
    fn clear(&mut self) {
        self.digits.clear();
        self.places.clear();
    }

    fn reserve(&mut self, additional: usize) {
        self.digits.reserve(additional);
        self.places.reserve(additional);
    }

    #[inline(always)]
    fn push(&mut self, [alt, reference]: [Dose; 2]) {
        self.digits.push([alt.digits, reference.digits]);
        self.places.push([alt.places, reference.places]);
    }

    /// The fewest and most decimal places of any present dose, the largest ALT and REF digits, and
    /// whether some person has an ALT dosage without a REF dosage. Four people a step without a
    /// branch: a missing dose is zero digits at u8::MAX places, which leaves every extreme but
    /// `most` alone, and its mask keeps it out of `most`.
    fn extremes(&self) -> (u8, u8, [u64; 2], bool) {
        let alt_side = Mask::<i64, 8>::from_array([true, false, true, false, true, false, true, false]);
        let digit_steps = self.digits.as_flattened().chunks_exact(8);
        let place_steps = self.places.as_flattened().chunks_exact(8);
        let stepped = digit_steps.len() * 4;
        let (mut fewest_lanes, mut most_lanes) = (u8x8::splat(u8::MAX), u8x8::splat(0));
        let (mut alt_lanes, mut ref_lanes) = (u64x8::splat(0), u64x8::splat(0));
        let mut incomplete_ref = false;
        for (digits, places) in digit_steps.zip(place_steps) {
            let places = u8x8::from_slice(places);
            let present = places.simd_ne(u8x8::splat(u8::MAX));
            fewest_lanes = fewest_lanes.simd_min(places);
            most_lanes = most_lanes.simd_max(present.select(places, u8x8::splat(0)));
            // A wrapping |i64::MIN| is i64::MIN, which reads as 2^63: its unsigned magnitude.
            let magnitudes = i64x8::from_slice(digits).abs().cast::<u64>();
            alt_lanes = alt_lanes.simd_max(alt_side.select(magnitudes, u64x8::splat(0)));
            ref_lanes = ref_lanes.simd_max((!alt_side).select(magnitudes, u64x8::splat(0)));
            let present = present.to_bitmask();
            incomplete_ref |= (present & !(present >> 1) & 0x55) != 0;
        }
        let (mut fewest, mut most) = (fewest_lanes.reduce_min(), most_lanes.reduce_max());
        let mut largest = [alt_lanes.reduce_max(), ref_lanes.reduce_max()];
        for (digits, places) in self.digits[stepped..].iter().zip(&self.places[stepped..]) {
            for side in 0..2 {
                let present = u8::from(places[side] != u8::MAX).wrapping_neg();
                fewest = fewest.min(places[side]);
                most = most.max(places[side] & present);
                largest[side] = largest[side].max(digits[side].unsigned_abs());
            }
            incomplete_ref |= (places[0] != u8::MAX) & (places[1] == u8::MAX);
        }
        (fewest, most, largest, incomplete_ref)
    }

    /// Every person's dosages, as pushed.
    #[cfg(test)]
    fn pairs(&self) -> Vec<[Dose; 2]> {
        let dose = |digits: i64, places: u8| Dose { digits, places };
        self.digits
            .iter()
            .zip(&self.places)
            .map(|(digits, places)| [dose(digits[0], places[0]), dose(digits[1], places[1])])
            .collect()
    }

    /// Pads the column with missing dosages to `len` people.
    fn resize_missing(&mut self, len: usize) {
        self.digits.resize(len, [0; 2]);
        self.places.resize(len, [u8::MAX; 2]);
    }

    /// Brings every present dose to `places` and gives the largest of each side there, or `None`
    /// once a dose leaves i64; the doses brought to `places` by then keep their value.
    fn rescale(&mut self, places: u8) -> Option<[u128; 2]> {
        let mut largest = [0u128; 2];
        for (digits, dose_places) in self.digits.iter_mut().zip(self.places.iter_mut()) {
            for side in 0..2 {
                if dose_places[side] == u8::MAX {
                    continue;
                }
                if dose_places[side] != places {
                    let power = 10i64.checked_pow(u32::from(places - dose_places[side]))?;
                    digits[side] = digits[side].checked_mul(power)?;
                    dose_places[side] = places;
                }
                largest[side] = largest[side].max(u128::from(digits[side].unsigned_abs()));
            }
        }
        Some(largest)
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
    fn dosages(&mut self) -> &mut Doses {
        if let Self::Calls(_) = self {
            *self = Self::Dosages(Doses::default());
        }
        let Self::Dosages(dosages) = self else {
            unreachable!("the column holds dosages");
        };
        dosages.clear();
        dosages
    }

    /// Brings every dose to the column's most decimal places and gives its scale. Runs where the
    /// column was decoded, so the accumulating thread only reads the result.
    fn normalize(&mut self) -> ColumnScale {
        match self {
            Self::Calls(codes) => {
                let (mut alt, mut reference) = (0u8, 0u8);
                for &code in codes.iter() {
                    let present = if code == MISSING_CALL { 0 } else { code };
                    alt = alt.max(present & 0x0f);
                    reference = reference.max(present >> 4);
                }
                ColumnScale {
                    places: 0,
                    largest: Some([u128::from(alt), u128::from(reference)]),
                    incomplete_ref: false,
                }
            }
            Self::Dosages(doses) => {
                let (fewest, most, largest, incomplete_ref) = doses.extremes();
                let largest = if fewest < most {
                    doses.rescale(most)
                } else {
                    Some(largest.map(u128::from))
                };
                ColumnScale {
                    places: most,
                    largest,
                    incomplete_ref,
                }
            }
        }
    }
}

/// Per-person exact score sums and missing counts, and each score's matched variants.
///
/// A cell holds its person's sum of one score at `10^places`: the most decimal places the
/// score's weights need, plus the most any dosage the score has taken carried. A dosage with more
/// places rescales the score's cells first, and a value that leaves i128 moves into a wide
/// integer, so no term is rounded; a value is rounded once, at output.
struct ScoreTotals {
    num_scores: usize,
    cells: Vec<i128>,
    /// What each cell has taken since its score's last flush. Terms add here without a check:
    /// the bounds of the terms since the flush sum to at most `LANE_LIMIT`.
    lanes: Vec<i64>,
    /// Per score, the sum of the bounds of the terms its lanes have taken since the last flush.
    headroom: Vec<u128>,
    missing_counts: Vec<u32>,
    score_variant_counts: Vec<u32>,
    weight_places: Vec<i32>,
    dosage_places: Vec<u8>,
    /// People per range when the cells are split across the rayon pool.
    range_people: usize,
    /// Each range's wide cell parts: a cell's value is its i128 plus its entry here, if any.
    spills: Vec<AHashMap<usize, Wide>>,
    /// Alleles counted in `score_variant_counts` whose terms are not yet in the
    /// cells, in input order.
    unapplied: Vec<ScoredAllele>,
}

/// A signed integer too wide for i128, in base-10^9 limbs, least significant first.
#[derive(Clone, Debug, Default)]
struct Wide {
    negative: bool,
    limbs: Vec<u32>,
}

impl Wide {
    fn of(value: i128) -> Self {
        Self {
            negative: value < 0,
            limbs: limbs_of(value.unsigned_abs()),
        }
    }

    fn add(&mut self, other: &Wide) {
        if self.negative == other.negative {
            add_limbs(&mut self.limbs, &other.limbs);
        } else if compare_limbs(&self.limbs, &other.limbs) == std::cmp::Ordering::Less {
            self.limbs = subtract_limbs(&other.limbs, &self.limbs);
            self.negative = other.negative;
        } else {
            self.limbs = subtract_limbs(&self.limbs, &other.limbs);
        }
        if self.limbs.is_empty() {
            self.negative = false;
        }
    }

    fn scale(&mut self, digits: u32) {
        shift_decimal(&mut self.limbs, digits);
    }
}

/// One scored allele's rules at their scores' scales, and every kept person's dosage of it.
struct ScoredAllele {
    rules: Vec<ExactRule>,
    /// The distinct scores of `rules`, each counted once for a missing dosage.
    missing_scores: Vec<usize>,
    column: DosageColumn,
}

/// The most the terms added to a score's lanes between two flushes may sum to, in magnitude, so
/// that no lane leaves i64.
const LANE_LIMIT: u128 = i64::MAX as u128;

/// A matched rule's weight at its score's scale.
struct ExactRule {
    score_index: usize,
    effect_is_ref: bool,
    digits: i64,
    /// The power of ten that puts `digits` at the score's weight places.
    shift: u32,
    /// `digits × 10^shift`, when it fits i128.
    scaled: Option<i128>,
    /// The largest term this rule adds for any person, when it is at most `LANE_LIMIT`: the rule
    /// then adds into the lanes, and otherwise into the cells.
    bound: Option<u128>,
    /// The weight at the score's scale, times a dose at the column's decimal places, is a term.
    weight: i64,
    /// For hard calls, the term of each count of copies; fifteen, `MISSING_CALL`'s, adds zero.
    call_terms: [i64; 16],
}

fn pow10(digits: u32) -> Option<i128> {
    10i128.checked_pow(digits)
}

/// Adds `term` to a cell, moving the cell's value into its wide part when the sum leaves i128.
#[inline(always)]
fn add_term(cell: &mut i128, index: usize, spill: &mut AHashMap<usize, Wide>, term: i128) {
    match cell.checked_add(term) {
        Some(sum) => *cell = sum,
        None => spill
            .entry(index)
            .or_default()
            .add(&Wide::of(std::mem::replace(cell, term))),
    }
}

/// Adds a term too wide for i128: `rule`'s weight times `dose_digits × 10^dose_shift`.
#[cold]
fn add_wide_term(
    index: usize,
    spill: &mut AHashMap<usize, Wide>,
    rule: &ExactRule,
    dose_digits: i64,
    dose_shift: u32,
) {
    let mut term = Wide::of(i128::from(rule.digits) * i128::from(dose_digits));
    term.scale(rule.shift + dose_shift);
    spill.entry(index).or_default().add(&term);
}

/// Adds `rule`'s weight times `dose_digits × 10^dose_shift` to a cell, exactly at any size.
#[inline]
fn add_exact(
    cell: &mut i128,
    index: usize,
    spill: &mut AHashMap<usize, Wide>,
    rule: &ExactRule,
    dose_digits: i64,
    dose_shift: u32,
) {
    let term = rule
        .scaled
        .and_then(|weight| weight.checked_mul(i128::from(dose_digits)))
        .and_then(|term| term.checked_mul(pow10(dose_shift)?));
    match term {
        Some(term) => add_term(cell, index, spill, term),
        None => add_wide_term(index, spill, rule, dose_digits, dose_shift),
    }
}

impl ScoreTotals {
    fn new(num_people: usize, num_scores: usize, rules_by_key: &ScoreRules) -> Self {
        let mut weight_places = vec![0i32; num_scores];
        for application in &rules_by_key.applications {
            let places = &mut weight_places[application.score_index];
            *places = (*places).max(-application.exponent);
        }
        let range_people = num_people
            .div_ceil(rayon::current_num_threads().max(1) * RANGES_PER_WORKER)
            .max(MIN_PEOPLE_PER_RANGE);
        Self {
            num_scores,
            cells: vec![0i128; num_people * num_scores],
            lanes: vec![0i64; num_people * num_scores],
            headroom: vec![0u128; num_scores],
            missing_counts: vec![0u32; num_people * num_scores],
            score_variant_counts: vec![0u32; num_scores],
            weight_places,
            dosage_places: vec![0u8; num_scores],
            range_people,
            spills: (0..num_people.div_ceil(range_people))
                .map(|_| AHashMap::new())
                .collect(),
            unapplied: Vec::new(),
        }
    }

    /// Counts the allele once in each score it scores, and queues each rule's
    /// weight times every kept person's dosage of its effect allele for
    /// [`ScoreTotals::apply`]. A REF-effect rule fails when some person's dosage
    /// has no REF dosage. `scale` is the column's, when it was normalized where it was decoded.
    fn add_allele(
        &mut self,
        matched_rules: Vec<MatchedRule>,
        mut column: DosageColumn,
        scale: Option<ColumnScale>,
        ref_effect_error: impl Fn(usize) -> String,
    ) -> Result<(), String> {
        let scale = scale.unwrap_or_else(|| column.normalize());
        if let Some(rule) = matched_rules.iter().find(|rule| rule.effect_is_ref)
            && scale.incomplete_ref
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
        // A dosage with more decimal places than a score has taken rescales that score, after
        // the queued alleles are added at the old scale.
        let places = scale.places;
        let growing: Vec<usize> = missing_scores
            .iter()
            .copied()
            .filter(|&score| places > self.dosage_places[score])
            .collect();
        if !growing.is_empty() {
            self.apply();
            for score in growing {
                self.rescale(score, places);
            }
        }
        let calls = matches!(column, DosageColumn::Calls(_));
        let mut rules: Vec<ExactRule> = matched_rules
            .iter()
            .map(|rule| self.exact_rule(rule, places, scale.largest, calls))
            .collect();
        // A score whose lanes cannot take this allele's bounds is flushed first. The rules of a
        // score whose bounds alone pass the limit add into the cells.
        let mut bounds = Vec::with_capacity(missing_scores.len());
        let mut full = Vec::new();
        for group in rules.chunk_by_mut(|a, b| a.score_index == b.score_index) {
            let score = group[0].score_index;
            let bound = group.iter().filter_map(|rule| rule.bound).sum::<u128>();
            if bound > LANE_LIMIT {
                group.iter_mut().for_each(|rule| rule.bound = None);
                continue;
            }
            if self.headroom[score] + bound > LANE_LIMIT {
                full.push(score);
            }
            bounds.push((score, bound));
        }
        if !full.is_empty() {
            self.apply();
            self.flush(&full);
        }
        for (score, bound) in bounds {
            self.headroom[score] += bound;
        }
        self.unapplied.push(ScoredAllele {
            rules,
            missing_scores,
            column,
        });
        Ok(())
    }

    /// `rule` at its score's scale, for a column whose doses are at `places` decimal places and
    /// whose largest dose of each side, when known, is `largest`.
    fn exact_rule(
        &self,
        rule: &MatchedRule,
        places: u8,
        largest: Option<[u128; 2]>,
        calls: bool,
    ) -> ExactRule {
        let score = rule.score_index;
        // A score's weight places are at least any of its weights' own, so the shift is nonnegative.
        let shift = (self.weight_places[score] + rule.exponent) as u32;
        let scaled = pow10(shift).and_then(|power| i128::from(rule.digits).checked_mul(power));
        // A score's dosage places are at least any column's it has taken.
        let weight = scaled
            .zip(pow10(u32::from(self.dosage_places[score] - places)))
            .and_then(|(weight, power)| i64::try_from(weight.checked_mul(power)?).ok());
        let side = usize::from(rule.effect_is_ref);
        let bound = weight
            .zip(largest)
            .and_then(|(weight, largest)| u128::from(weight.unsigned_abs()).checked_mul(largest[side]))
            .filter(|&bound| bound <= LANE_LIMIT);
        let mut call_terms = [0i64; 16];
        if let (true, Some(weight), Some(_)) = (calls, weight, bound) {
            // No call holds more copies than `largest`, so the terms it can reach fit.
            for (copies, term) in call_terms.iter_mut().enumerate().take(15) {
                *term = weight.checked_mul(copies as i64).unwrap_or(0);
            }
        }
        ExactRule {
            score_index: score,
            effect_is_ref: rule.effect_is_ref,
            digits: rule.digits,
            shift,
            scaled,
            bound,
            weight: weight.unwrap_or(0),
            call_terms,
        }
    }

    /// Moves the lanes of `scores` into their cells.
    fn flush(&mut self, scores: &[usize]) {
        // A score without headroom taken has nothing in its lanes.
        let scores: Vec<usize> = scores
            .iter()
            .copied()
            .filter(|&score| self.headroom[score] > 0)
            .collect();
        if scores.is_empty() {
            return;
        }
        let scores = &scores[..];
        let num_scores = self.num_scores;
        let range_cells = self.range_people * num_scores;
        self.cells
            .par_chunks_mut(range_cells)
            .zip(self.lanes.par_chunks_mut(range_cells))
            .zip(self.spills.par_iter_mut())
            .enumerate()
            .for_each(|(range, ((cells, lanes), spill))| {
                let people = cells
                    .chunks_exact_mut(num_scores)
                    .zip(lanes.chunks_exact_mut(num_scores));
                for (person, (cells, lanes)) in people.enumerate() {
                    for &score in scores {
                        let index = range * range_cells + person * num_scores + score;
                        let lane = std::mem::take(&mut lanes[score]);
                        add_term(&mut cells[score], index, spill, i128::from(lane));
                    }
                }
            });
        for &score in scores {
            self.headroom[score] = 0;
        }
    }

    /// Adds every queued allele and moves every lane into its cell.
    fn finish(&mut self) {
        self.apply();
        let scores: Vec<usize> = (0..self.num_scores).collect();
        self.flush(&scores);
    }

    /// Multiplies every cell of `score` by the powers of ten that bring it to `places` dosage places.
    fn rescale(&mut self, score: usize, places: u8) {
        self.flush(&[score]);
        let digits = u32::from(places - self.dosage_places[score]);
        let factor = pow10(digits);
        let num_scores = self.num_scores;
        let range_cells = self.range_people * num_scores;
        self.cells
            .par_chunks_mut(range_cells)
            .zip(self.spills.par_iter_mut())
            .enumerate()
            .for_each(|(range, (cells, spill))| {
                for (offset, cell) in cells.iter_mut().enumerate().skip(score).step_by(num_scores) {
                    let index = range * range_cells + offset;
                    if let Some(wide) = spill.get_mut(&index) {
                        wide.scale(digits);
                    }
                    match factor.and_then(|factor| cell.checked_mul(factor)) {
                        Some(value) => *cell = value,
                        None => {
                            let mut wide = Wide::of(std::mem::take(cell));
                            wide.scale(digits);
                            spill.entry(index).or_default().add(&wide);
                        }
                    }
                }
            });
        self.dosage_places[score] = places;
    }

    /// Adds the terms of every queued allele to the cells, over ranges of people
    /// on the rayon pool. Integer sums do not depend on how the people are split.
    fn apply(&mut self) {
        if self.unapplied.is_empty() {
            return;
        }
        let num_scores = self.num_scores;
        let range_people = self.range_people;
        let range_cells = range_people * num_scores;
        let alleles = &self.unapplied;
        let dosage_places = &self.dosage_places;
        self.cells
            .par_chunks_mut(range_cells)
            .zip(self.lanes.par_chunks_mut(range_cells))
            .zip(self.missing_counts.par_chunks_mut(range_cells))
            .zip(self.spills.par_iter_mut())
            .enumerate()
            .for_each(|(range, (((cells, lanes), missing), spill))| {
                for allele in alleles {
                    allele.apply(
                        range * range_people,
                        range * range_cells,
                        num_scores,
                        dosage_places,
                        Accumulators {
                            cells,
                            lanes,
                            missing,
                            spill,
                        },
                    );
                }
            });
        self.unapplied.clear();
    }
}

/// Adds one term per person to one score's lanes, and counts the people it gives as missing.
#[inline(always)]
fn add_one_score<T>(
    values: impl Iterator<Item = T>,
    score: usize,
    num_scores: usize,
    lanes: &mut [i64],
    missing: &mut [u32],
    term: impl Fn(T) -> (i64, bool),
) {
    if num_scores == 1 {
        for ((value, lane), count) in values.zip(lanes.iter_mut()).zip(missing.iter_mut()) {
            let (term, absent) = term(value);
            *lane = lane.wrapping_add(term);
            *count += u32::from(absent);
        }
    } else {
        let slots = lanes.iter_mut().zip(missing.iter_mut()).skip(score).step_by(num_scores);
        for (value, (lane, count)) in values.zip(slots) {
            let (term, absent) = term(value);
            *lane = lane.wrapping_add(term);
            *count += u32::from(absent);
        }
    }
}

/// One range's cells, lanes, missing counts and wide cell parts, laid out person × score.
struct Accumulators<'a> {
    cells: &'a mut [i128],
    lanes: &'a mut [i64],
    missing: &'a mut [u32],
    spill: &'a mut AHashMap<usize, Wide>,
}

impl ScoredAllele {
    /// Adds this allele's terms for the people from `first_person` on, whose accumulators, from
    /// global cell `first_cell`, `into` holds.
    fn apply(
        &self,
        first_person: usize,
        first_cell: usize,
        num_scores: usize,
        dosage_places: &[u8],
        into: Accumulators<'_>,
    ) {
        let Accumulators {
            cells,
            lanes,
            missing,
            spill,
        } = into;
        let people = first_person..first_person + missing.len() / num_scores;
        // One rule adding into lanes needs no branch: a missing call's copies are fifteen, whose
        // term is zero, and a missing dose's digits are zero.
        match (&self.column, &self.rules[..]) {
            (DosageColumn::Calls(codes), [rule]) if rule.bound.is_some() => {
                let shift = if rule.effect_is_ref { 4 } else { 0 };
                let persons = codes[people].iter().copied();
                add_one_score(persons, rule.score_index, num_scores, lanes, missing, |code| {
                    let term = rule.call_terms[usize::from((code >> shift) & 0x0f)];
                    (term, code == MISSING_CALL)
                });
            }
            (DosageColumn::Dosages(doses), [rule]) if rule.bound.is_some() => {
                let side = usize::from(rule.effect_is_ref);
                let persons = doses.digits[people.clone()].iter().zip(&doses.places[people]);
                add_one_score(persons, rule.score_index, num_scores, lanes, missing, |(digits, places)| {
                    (rule.weight.wrapping_mul(digits[side]), places[0] == u8::MAX)
                });
            }
            (DosageColumn::Calls(codes), rules) => {
                for (offset, &code) in codes[people].iter().enumerate() {
                    let base = offset * num_scores;
                    if code == MISSING_CALL {
                        for &score_index in &self.missing_scores {
                            missing[base + score_index] += 1;
                        }
                        continue;
                    }
                    for rule in rules {
                        let copies = if rule.effect_is_ref { code >> 4 } else { code & 0x0f };
                        let index = base + rule.score_index;
                        if rule.bound.is_some() {
                            let term = rule.call_terms[usize::from(copies)];
                            lanes[index] = lanes[index].wrapping_add(term);
                        } else {
                            let shift = u32::from(dosage_places[rule.score_index]);
                            let cell = &mut cells[index];
                            add_exact(cell, first_cell + index, spill, rule, i64::from(copies), shift);
                        }
                    }
                }
            }
            (DosageColumn::Dosages(doses), rules) => {
                let persons = doses.digits[people.clone()].iter().zip(&doses.places[people]);
                for (offset, (digits, places)) in persons.enumerate() {
                    let base = offset * num_scores;
                    if places[0] == u8::MAX {
                        for &score_index in &self.missing_scores {
                            missing[base + score_index] += 1;
                        }
                        continue;
                    }
                    for rule in rules {
                        let side = usize::from(rule.effect_is_ref);
                        let index = base + rule.score_index;
                        if rule.bound.is_some() {
                            let term = rule.weight.wrapping_mul(digits[side]);
                            lanes[index] = lanes[index].wrapping_add(term);
                        } else {
                            let shift = u32::from(dosage_places[rule.score_index] - places[side]);
                            let cell = &mut cells[index];
                            add_exact(cell, first_cell + index, spill, rule, digits[side], shift);
                        }
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

        // A rule's allele pair scores one row; a second row carrying it repeats the variant.
        let mut taken = vec![false; rules.len()];
        for (row, allele) in &mut self.alleles {
            let (ref_allele, alt_allele) = &self.rows[*row];
            let mut matched = Vec::new();
            for (index, (rule, decision)) in rules.iter().zip(&decisions).enumerate() {
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
                    if std::mem::replace(&mut taken[index], true) {
                        return Err(repeated_pair_error(
                            &self.chromosome,
                            key.1,
                            effect_allele,
                            rules_by_key.allele(rule.other_allele),
                        ));
                    }
                    matched.extend(rules_by_key.applications(rule).iter().map(|application| {
                        MatchedRule {
                            score_index: application.score_index,
                            digits: application.digits,
                            exponent: application.exponent,
                            effect_is_ref,
                        }
                    }));
                }
            }
            totals.add_allele(
                merge_matched_rules(matched),
                std::mem::take(&mut allele.column),
                allele.scale.take(),
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
    let Some(&(rules_start, rules_end)) = rules_by_key.ranges.get(&(chr, pos)) else {
        return Ok(());
    };
    let score_rules = &rules_by_key.rules[rules_start..rules_end];
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
        let Some((matched_rules, first_rule)) = rules_for_allele(
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
        allele.scale = Some(allele.column.normalize());
        allele.matched_rules = matched_rules;
        allele.first_rule = rules_start + first_rule;
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

/// The rules scoring `(ref_allele, alt_allele)`, merged, with the first of `score_rules` among
/// them, or `None` when no rule may. At an `effect_only` position the rules are matched once
/// every record there has been read, so an allele some rule may score gets no rules yet.
fn rules_for_allele(
    rules_by_key: &ScoreRules,
    score_rules: &[ScoreRule],
    effect_only: bool,
    ref_allele: &str,
    alt_allele: &str,
) -> Option<(Vec<MatchedRule>, usize)> {
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
        return may_score.then(|| (Vec::new(), 0));
    }
    let (matched, first_rule) =
        match_rules_for_allele(rules_by_key, score_rules, ref_allele, alt_allele);
    (!matched.is_empty()).then_some((matched, first_rule))
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
    let Some(&(rules_start, rules_end)) = rules_by_key.ranges.get(&(chr, pos)) else {
        return Ok(());
    };
    let score_rules = &rules_by_key.rules[rules_start..rules_end];
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
        let Some((matched_rules, first_rule)) = rules_for_allele(
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
        allele.scale = Some(allele.column.normalize());
        allele.matched_rules = matched_rules;
        allele.first_rule = rules_start + first_rule;
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

/// The error for a second record carrying an allele pair that score rows name: which record a
/// row scores is unknown, and scoring both would count the variant twice.
fn repeated_pair_error(chromosome: &str, position: u32, allele: &str, other_allele: &str) -> String {
    format!(
        "More than one record at {chromosome}:{position} carries the alleles {other_allele} and {allele}, which a score row names, so which record the row scores is unknown. Remove the duplicate records, for example with bcftools norm --rm-dup exact."
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
    // a weight is its shortest round-trip decimal, which the exact sums add as written.
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

        let mut places_hint = 0usize;
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

                let (digits, exponent) = shortest_decimal_hinted(weight, &mut places_hint);
                rules.push_application(ScoreApplication {
                    score_index,
                    digits,
                    exponent,
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

/// The rules of `rules` scoring `(ref_allele, alt_allele)`, merged, and the index in `rules` of
/// the first of them.
fn match_rules_for_allele(
    rules_by_key: &ScoreRules,
    rules: &[ScoreRule],
    ref_allele: &str,
    alt_allele: &str,
) -> (Vec<MatchedRule>, usize) {
    let capacity = rules
        .iter()
        .map(|rule| rule.applications.1 - rule.applications.0)
        .sum();
    let mut matched = Vec::with_capacity(capacity);
    let mut first_rule = None;
    for (index, rule) in rules.iter().enumerate() {
        let effect_allele = rules_by_key.allele(rule.effect_allele);
        let other_allele = rules_by_key.allele(rule.other_allele);
        let Some(effect_is_ref) =
            pair_orientation(effect_allele, other_allele, ref_allele, alt_allele)
        else {
            continue;
        };
        first_rule.get_or_insert(index);
        for application in rules_by_key.applications(rule) {
            matched.push(MatchedRule {
                score_index: application.score_index,
                digits: application.digits,
                exponent: application.exponent,
                effect_is_ref,
            });
        }
    }
    (merge_matched_rules(matched), first_rule.unwrap_or(0))
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

/// Every rule at a position scoring one allele, the rules of one score and orientation adjacent
/// and in rule order. Duplicates stay separate rules: each is one written weight, added exactly.
fn merge_matched_rules(mut matched: Vec<MatchedRule>) -> Vec<MatchedRule> {
    matched.sort_by_key(|rule| (rule.score_index, rule.effect_is_ref));
    matched
}

#[derive(Debug, Clone, Copy)]
struct DecodedAltDosage {
    alt_dosage: Dose,
    ref_dosage: Option<Dose>,
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
    dosages: &mut Doses,
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
        // Two GT slots give ploidy two, and `dosage_from_values` sums the one DS
        // value from zero, which leaves it unchanged. The DS is found and read in
        // place, the column this layout holds most.
        if let Some(&[b'0'..=b'9' | b'.', b'/' | b'|', b'0'..=b'9' | b'.', b':']) =
            bytes.get(pos..pos + 4)
        {
            let ds = &bytes[pos + 4..];
            let len = ds.iter().position(|&byte| byte == b'\t').unwrap_or(ds.len());
            if let Some((alt, reference)) = plain_decimal(&ds[..len])
                .and_then(|alt| diploid_reference(alt).map(|reference| (alt, reference)))
            {
                dosages.push([alt, reference]);
                pos = (pos + 5 + len).min(bytes.len());
                sample_idx += 1;
                continue;
            }
        }
        let end = memchr(b'\t', &bytes[pos..]).map_or(bytes.len(), |offset| pos + offset);
        let column = &columns[pos..end];
        pos = (end + 1).min(bytes.len());
        let column = if column == "." { "" } else { column };
        dosages.push(dosage_pair(
            decode_vcf_sample(column, Some(1), None, Some(0), 1, alt_index, alt_count)?,
            ref_effect_error,
        )?);
        sample_idx += 1;
    }
    dosages.resize_missing(kept_indices.len());
    Ok(true)
}

/// A visited dosage as `[alt, ref]`, or the REF-effect error for a dosage
/// without a REF dosage when a rule needs one.
fn dosage_pair(
    dosage: Option<DecodedAltDosage>,
    ref_effect_error: &impl Fn() -> Option<String>,
) -> Result<[Dose; 2], Box<dyn Error + Send + Sync>> {
    match dosage {
        None => Ok([Dose::MISSING; 2]),
        Some(DecodedAltDosage {
            alt_dosage,
            ref_dosage: Some(ref_dosage),
        }) => Ok([alt_dosage, ref_dosage]),
        Some(DecodedAltDosage {
            alt_dosage,
            ref_dosage: None,
        }) => match ref_effect_error() {
            Some(message) => Err(message.into()),
            None => Ok([alt_dosage, Dose::MISSING]),
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
    let mut kept = 0usize;
    while let Some(&kept_idx) = kept_indices.get(kept) {
        while sample_idx < kept_idx && pos < bytes.len() {
            pos = memchr(b'\t', &bytes[pos..]).map_or(bytes.len(), |offset| pos + offset + 1);
            sample_idx += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        // Sixteen kept people in adjacent columns decode as one block when every
        // column holds a diploid call of single-digit alleles.
        if kept_indices.get(kept + 15) == Some(&(sample_idx + 15))
            && let Some(block) = bytes.get(pos..pos + 64)
            && let Some(block_codes) = diploid_call_codes(block, alt_index)
        {
            codes.extend_from_slice(&block_codes);
            pos += 64;
            sample_idx += 16;
            kept += 16;
            continue;
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
                }) if alt_dosage.places == 0
                    && ref_dosage.places == 0
                    && alt_dosage.digits <= 14
                    && ref_dosage.digits <= 14 =>
                {
                    alt_dosage.digits as u8 | (ref_dosage.digits as u8) << 4
                }
                Some(_) => return Ok(false),
            }
        };
        codes.push(code);
        sample_idx += 1;
        kept += 1;
    }
    codes.resize(kept_indices.len(), MISSING_CALL);
    Ok(true)
}

/// The codes of sixteen adjacent columns that each hold a diploid call of
/// single-digit alleles and end with a tab, as [`vcf_gt_calls`] codes such a
/// column, or `None` when any column holds something else.
fn diploid_call_codes(block: &[u8], alt_index: usize) -> Option<[u8; 16]> {
    const FIRST: [usize; 16] = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60];
    const SEPARATOR: [usize; 16] = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61];
    const SECOND: [usize; 16] = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62];
    const TAB: [usize; 16] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63];
    let columns = u8x64::from_array(block.try_into().ok()?);
    let first = simd_swizzle!(columns, FIRST) - u8x16::splat(b'0');
    let separator = simd_swizzle!(columns, SEPARATOR);
    let second = simd_swizzle!(columns, SECOND) - u8x16::splat(b'0');
    let tab = simd_swizzle!(columns, TAB);
    let regular = first.simd_lt(u8x16::splat(10))
        & second.simd_lt(u8x16::splat(10))
        & (separator.simd_eq(u8x16::splat(b'/')) | separator.simd_eq(u8x16::splat(b'|')))
        & tab.simd_eq(u8x16::splat(b'\t'));
    if !regular.all() {
        return None;
    }
    let copies = |allele: u8| {
        let allele = u8x16::splat(allele);
        let (one, zero) = (u8x16::splat(1), u8x16::splat(0));
        first.simd_eq(allele).select(one, zero) + second.simd_eq(allele).select(one, zero)
    };
    // An ALT index above 9 is no single digit, so it matches no allele here.
    let alt = copies(u8::try_from(alt_index).unwrap_or(u8::MAX));
    Some((alt | copies(0) << u8x16::splat(4)).to_array())
}

/// Decodes every kept person's dosage of ALT `alt_index` from a BCF `record` into
/// `column`, as [`for_each_bcf_dosage_best`] visits them, with `ref_effect_error`
/// as in [`decode_vcf_column`].
// Kept out of line: inlined into the decode workers, its sixteen-person block copies left as
// Vec::extend_from_slice and memmove calls (100-score BCF row: 2.3% and +4.5% of cycles).
#[inline(never)]
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
    let mut kept = 0usize;
    while let Some(&sample_idx) = kept_indices.get(kept) {
        // Sixteen kept people in adjacent samples decode as one block when each
        // genotype holds two present alleles.
        if gt.width == 2
            && sample_idx + 16 <= sample_count
            && kept_indices.get(kept + 15) == Some(&(sample_idx + 15))
            && let Some(block_codes) =
                diploid_bcf_call_codes(&gt.src[2 * sample_idx..2 * (sample_idx + 16)], alt_index)
        {
            codes.extend_from_slice(&block_codes);
            kept += 16;
            continue;
        }
        kept += 1;
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

/// The codes of sixteen adjacent samples whose Int8 genotypes each hold two
/// present alleles, as [`bcf_gt_calls`] codes such a genotype, or `None` when any
/// value is a missing allele, missing, end-of-vector or reserved, or above 127.
fn diploid_bcf_call_codes(values: &[u8], alt_index: usize) -> Option<[u8; 16]> {
    const FIRST: [usize; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30];
    const SECOND: [usize; 16] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31];
    let values = u8x32::from_array(values.try_into().ok()?);
    // A present allele is stored as its position plus one, shifted left over the
    // phasing bit, so 2 through 127.
    if !(values.simd_ge(u8x32::splat(2)) & values.simd_le(u8x32::splat(0x7f))).all() {
        return None;
    }
    let first = simd_swizzle!(values, FIRST) >> u8x16::splat(1);
    let second = simd_swizzle!(values, SECOND) >> u8x16::splat(1);
    let copies = |stored: u8| {
        let stored = u8x16::splat(stored);
        let (one, zero) = (u8x16::splat(1), u8x16::splat(0));
        first.simd_eq(stored).select(one, zero) + second.simd_eq(stored).select(one, zero)
    };
    // An ALT position that no stored value in 2 through 127 reaches matches no allele.
    let alt = copies(u8::try_from(alt_index + 1).unwrap_or(u8::MAX));
    Some((alt | copies(1) << u8x16::splat(4)).to_array())
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
                        BcfValue::Value(dosage) => Dose::from_f32(dosage).map(Some),
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
                    BcfValue::Value(probability) if !probability.is_finite() => {
                        Err("GP probabilities must be finite and between zero and one".into())
                    }
                    BcfValue::Value(probability) => Dose::from_f32(probability).map(Some),
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
        let mut dosage = 0u8;
        let mut ref_dosage = 0u8;
        let mut ploidy = 0u8;
        for allele in gt.genotype_alleles(sample) {
            let Some(allele) = allele else {
                return Ok(None);
            };
            if allele == alt_index {
                dosage += 1;
            }
            if allele == 0 {
                ref_dosage += 1;
            }
            ploidy = ploidy.checked_add(1).ok_or("genotype ploidy overflow")?;
        }
        Ok((ploidy > 0).then_some(DecodedAltDosage {
            alt_dosage: Dose::copies(dosage),
            ref_dosage: Some(Dose::copies(ref_dosage)),
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
        // An f32 by the shortest decimal that reads back as the same f32, as the typed route reads it.
        SeriesValue::Float(value) => write!(out, "{value}")?,
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
                    Some(value) => write!(out, "{value}")?,
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
        field.split(',').map(parse_dose),
        alt_index,
        alt_count,
        ploidy,
    )
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
    let parts = field
        .split(',')
        .map(|part| -> Result<Option<Dose>, Box<dyn Error + Send + Sync>> {
            if part == "." {
                return Ok(None);
            }
            let probability = part.parse::<f64>()?;
            if !probability.is_finite() {
                return Err("GP probabilities must be finite and between zero and one".into());
            }
            plain_decimal(part.as_bytes())
                .map_or_else(|| Dose::from_f64(probability), Ok)
                .map(Some)
        });
    gp_from_values(
        field.split(',').count(),
        parts,
        alt_index,
        alt_count,
        ploidy,
    )
}

fn parse_vcf_genotype(
    field: &str,
    alt_index: usize,
) -> Result<Option<DecodedAltDosage>, Box<dyn Error + Send + Sync>> {
    if field.is_empty() {
        return Ok(None);
    }

    let mut dosage = 0u8;
    let mut ref_dosage = 0u8;
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
                    dosage += 1;
                }
                if allele == 0 {
                    ref_dosage += 1;
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
            alt_dosage: Dose::copies(dosage),
            ref_dosage: Some(Dose::copies(ref_dosage)),
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
    if crc::crc32(block) != crc32 {
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
    fn column_extremes_match_one_dose_at_a_time() {
        let mut state = 0x2354_e47e_0000_0001u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for people in 0..40usize {
            for _ in 0..50 {
                let mut doses = Doses::default();
                for _ in 0..people {
                    // Missing doses, both i64 extremes, and short digits at up to 19 places.
                    let dose = |bits: u64, spread: u64| {
                        let places = (spread % 20) as u8;
                        match bits % 7 {
                            0 => Dose::MISSING,
                            1 => Dose { digits: i64::MIN, places },
                            2 => Dose { digits: i64::MAX, places },
                            _ => Dose { digits: (bits >> 8) as i64 % 1_000_000 - 500_000, places },
                        }
                    };
                    let (a, b, c, d) = (next(), next(), next(), next());
                    doses.push([dose(a, b), dose(c, d)]);
                }
                let mut want = (u8::MAX, 0u8, [0u64; 2], false);
                for [alt, reference] in doses.pairs() {
                    for (side, dose) in [alt, reference].into_iter().enumerate() {
                        want.0 = want.0.min(dose.places);
                        if dose.places != u8::MAX {
                            want.1 = want.1.max(dose.places);
                        }
                        want.2[side] = want.2[side].max(dose.digits.unsigned_abs());
                    }
                    want.3 |= alt.places != u8::MAX && reference.places == u8::MAX;
                }
                assert_eq!(doses.extremes(), want, "people {people}");
            }
        }
    }

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
            for (actual, expected) in result.sums().iter().zip(expected) {
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
        assert_eq!(partial.alt_dosage, Dose { digits: 5, places: 1 });
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
        assert_eq!(result.sums(), [11.0]);
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
            assert_eq!(vcf.sums(), bcf.sums(), "{name}");
            assert_eq!(vcf.score_variant_counts, bcf.score_variant_counts, "{name}");
            assert_eq!(vcf.missing_counts, bcf.missing_counts, "{name}");
            vcf
        };

        let pairs = score(
            "pairs.tsv",
            "1:100\tA\tG\t0.5\n1:200\tT\tC\t-0.25\n1:400\tT\tC\t2\n1:500\tA\tG\t0.125\n",
        );
        assert_eq!(pairs.sums(), [2.25, 2.125]);
        assert_eq!(pairs.score_variant_counts, [4]);
        // The same weights with the other allele unknown or listed as candidates, plus an
        // ambiguous locus and one the genotypes lack.
        let effect_only = score(
            "effect_only.tsv",
            "1:100\tA\t.\t0.5\n1:200\tT\t.\t-0.25\n1:300\tA\t.\t9\n1:400\tT\tC/G\t2\n1:500\tA\tG/T\t0.125\n1:600\tA\t.\t7\n",
        );
        assert_eq!(effect_only.sums(), pairs.sums());
        assert_eq!(effect_only.score_variant_counts, pairs.score_variant_counts);
        assert_eq!(effect_only.missing_counts, pairs.missing_counts);

        // No variant carries C at 1:100, two carry A at 1:300, and at 1:400 the only ALT
        // that is T pairs it with an unlisted allele: all dropped; the explicit pair stays.
        let skipped = score(
            "skipped.tsv",
            "1:100\tC\t.\t1\n1:300\tA\t.\t1\n1:400\tT\tA/G\t1\n1:500\tA\tG\t1\n",
        );
        assert_eq!(skipped.sums(), [2.0, 1.0]);
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
        assert_eq!(actual.sums(), expected.sums());
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

    /// `text` as a plain VCF, a BGZF VCF in small blocks and a BCF, in `dir`.
    fn cohort_files(dir: &Path, text: &str) -> [PathBuf; 3] {
        use noodles_vcf::variant::io::Write as _;

        let vcf = dir.join("cohort.vcf");
        let bgzf = dir.join("cohort.vcf.gz");
        let bcf = dir.join("cohort.bcf");
        std::fs::write(&vcf, text).expect("write vcf");
        std::fs::write(&bgzf, bgzf_bytes(text.as_bytes(), 64)).expect("write bgzf vcf");
        let mut reader = VcfReader::new(BufReader::new(File::open(&vcf).expect("open vcf")));
        let header = reader.read_header().expect("vcf header");
        let mut writer = noodles_bcf::io::Writer::new(File::create(&bcf).expect("create bcf"));
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
        [vcf, bgzf, bcf]
    }

    /// Records repeating an allele pair that a score row names are refused by name from plain
    /// VCF, BGZF VCF and BCF: adjacent, apart, with REF and ALT swapped, and at a position where
    /// another row names no single other allele. A repeated pair no row names still scores.
    #[test]
    fn records_repeating_a_scored_allele_pair_are_refused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let header = "##fileformat=VCFv4.2\n##contig=<ID=1>\n\
             ##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n\
             ##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"ALT dosage\">\n\
             #CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\n";
        let first = "1\t100\ta\tA\tG\t.\tPASS\t.\tGT:DS\t0|1:0.9\t0|0:0.1\n";
        let second = "1\t100\tb\tA\tG\t.\tPASS\t.\tGT:DS\t1|1:1.8\t0|1:1.2\n";
        let swapped = "1\t100\tb\tG\tA\t.\tPASS\t.\tGT:DS\t1|1:1.8\t0|1:1.2\n";
        let other = "1\t200\tc\tC\tT\t.\tPASS\t.\tGT:DS\t0|1:1\t1|1:2\n";
        let unscored = "1\t300\td\tT\tC\t.\tPASS\t.\tGT:DS\t0|1:1\t1|1:2\n";
        let pairs = "variant_id\teffect_allele\tother_allele\tS\n1:100\tG\tA\t0.5\n1:200\tT\tC\t1\n";
        let with_effect_only = "variant_id\teffect_allele\tother_allele\tS\n\
             1:100\tG\tA\t0.5\n1:100\tT\t.\t2\n1:200\tT\tC\t1\n";
        let score_path = dir.path().join("score.gnomon.tsv");
        for (case, body, scores) in [
            ("adjacent", [first, second, other].concat(), pairs),
            ("apart", [first, other, second].concat(), pairs),
            ("swapped", [first, swapped, other].concat(), pairs),
            ("effect-only position", [first, second, other].concat(), with_effect_only),
        ] {
            std::fs::write(&score_path, scores).expect("write score");
            for path in cohort_files(dir.path(), &format!("{header}{body}")) {
                let error = score_vcf_streaming(&path, std::slice::from_ref(&score_path), None, None)
                    .expect_err(case);
                assert!(
                    error
                        .to_string()
                        .contains("More than one record at 1:100 carries the alleles A and G"),
                    "{case}, {path:?}: {error}"
                );
            }
        }
        std::fs::write(&score_path, pairs).expect("write score");
        for path in cohort_files(dir.path(), &format!("{header}{first}{other}{unscored}{unscored}")) {
            let result = score_vcf_streaming(&path, std::slice::from_ref(&score_path), None, None)
                .expect("a repeated pair no row names scores");
            assert_eq!(result.sums(), [1.45, 2.05], "{path:?}");
        }
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
        assert!((result.sums()[0] - 1.1).abs() < 1e-9);
        assert!((result.sums()[1] - 0.2).abs() < 1e-9);
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
        assert_eq!(result.sums(), [0.5]);
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
        assert_eq!(result.sums(), [0.0, 1.5, 2.0]);
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
        assert_eq!(result.sums(), [2.5, 2.0]);
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
        assert_eq!(result.sums(), [0.0, 0.0]);
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
        assert_eq!(result.sums(), [1.0, 10.0, 11.0]);
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
        assert!((result.sums()[0] - 2.5).abs() < 1e-12);
        assert!((result.sums()[1] - 16.2).abs() < 1e-12);
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
        assert_eq!(result.sums(), [1.0]);
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
        assert_eq!(result.sums(), [0.5, 2.0, 1.0, 4.0]);
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
        assert_eq!(result.sums(), [2.0, 0.0]);
    }

    fn bgzf_block(data: &[u8]) -> Vec<u8> {
        bgzf_block_with(data, flate2::Compression::default())
    }

    /// A BGZF block of `data` deflated at `compression`.
    fn bgzf_block_with(data: &[u8], compression: flate2::Compression) -> Vec<u8> {
        let mut encoder = flate2::write::DeflateEncoder::new(Vec::new(), compression);
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
                    .sums()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                actual
                    .sums()
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

    type Visits = Option<Vec<Option<(Dose, Option<Dose>)>>>;

    type DosageVisitor<'a> =
        dyn FnMut(usize, Option<DecodedAltDosage>) -> Result<(), Box<dyn Error + Send + Sync>> + 'a;

    /// Every person a dosage route visits, as exact doses, or `None` if the route fails.
    fn bcf_dosage_visits(
        route: impl FnOnce(&mut DosageVisitor<'_>) -> Result<(), Box<dyn Error + Send + Sync>>,
    ) -> Visits {
        let mut visits = Vec::new();
        let mut visit = |out_idx: usize,
                         decoded: Option<DecodedAltDosage>|
         -> Result<(), Box<dyn Error + Send + Sync>> {
            assert_eq!(out_idx, visits.len());
            visits.push(decoded.map(|d| (d.alt_dosage, d.ref_dosage)));
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

    /// Wide BCF records of Int8 diploid genotypes decode to the hard calls the typed
    /// dosage route visits: mostly two present alleles, some records with missing
    /// alleles, missing, end-of-vector, reserved and out-of-range values scattered
    /// among them, under keep subsets that hold runs of sixteen adjacent samples,
    /// and people past the record's samples.
    #[test]
    fn wide_bcf_hard_calls_decode_as_the_typed_route_visits() {
        let mut draws = Draws(0x0bcf_1616);
        let samples = 60usize;
        let names: String = (0..samples).map(|index| format!("\ts{index}")).collect();
        let header = dosage_bcf_header("FORMAT=<ID=DS,Number=A,Type=Float", "Number=G,Type=Float")
            .replace("\ts1\ts2\ts3\ts4\n", &format!("{names}\n"));
        let all: Vec<usize> = (0..samples + 3).collect();
        let gapped: Vec<usize> = (0..samples).filter(|index| index % 19 != 4).collect();
        let kept_sets: [&[usize]; 2] = [&all, &gapped];
        let mut records = Vec::new();
        let mut alts = Vec::new();
        for _ in 0..400 {
            let alt_count = 1 + draws.below(3);
            let irregular_one_in = [0, 30, 6][draws.below(3)];
            let gt: Vec<u8> = (0..2 * samples)
                .map(|_| {
                    if irregular_one_in > 0 && draws.below(irregular_one_in) == 0 {
                        [0x00, 0x01, 0x80, 0x81, 0x83][draws.below(5)]
                    } else {
                        let allele = draws.below(alt_count + 2);
                        u8::try_from((allele + 1) * 2 + draws.below(2)).expect("a stored allele")
                    }
                })
                .collect();
            records.push(raw_bcf_record(alt_count, samples, &[(1, 1, 2, gt)]));
            alts.push((1 + draws.below(alt_count), alt_count));
        }
        let (header, records) = read_raw_bcf(&header, &records);
        for (index, (record, &(alt_index, alt_count))) in records.iter().zip(&alts).enumerate() {
            let kept = kept_sets[draws.below(kept_sets.len())];
            let mut codes = Vec::new();
            let fast = bcf_gt_calls(record, &header, alt_index, kept, &mut codes)
                .unwrap_or_else(|err| panic!("record {index}: {err}"));
            assert!(fast, "record {index} decodes as hard calls");
            let visits = bcf_dosage_visits(|visit| {
                for_each_bcf_dosage_best(record, &header, alt_index, alt_count, kept, visit)
            })
            .unwrap_or_else(|| panic!("record {index}: the typed route fails"));
            let decoded: Vec<Option<(Dose, Option<Dose>)>> = codes
                .iter()
                .map(|&code| {
                    (code != MISSING_CALL)
                        .then(|| (Dose::copies(code & 0x0f), Some(Dose::copies(code >> 4))))
                })
                .collect();
            assert_eq!(decoded, visits, "record {index}, kept {kept:?}, ALT {alt_index}");
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
                Some(vec![Some((
                    Dose {
                        digits: 5,
                        places: 1,
                    },
                    Some(Dose {
                        digits: 15,
                        places: 1,
                    }),
                ))]),
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
            bits(&expected.sums()),
            bits(&actual.sums()),
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

    /// A block that inflates to its recorded length but whose data disagree with
    /// its recorded CRC32 is refused: a byte inside a stored (uncompressed)
    /// deflate block is changed, which inflation itself cannot notice.
    #[test]
    fn blocks_whose_data_disagree_with_their_crc32_are_refused() {
        let data: Vec<u8> = (0..60_000u64)
            .map(|index| (index.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 56) as u8)
            .collect();
        let block = bgzf_block_with(&data, flate2::Compression::none());
        let mut decompressor = Decompressor::new();
        let mut inflated = vec![0u8; data.len()];
        inflate_bgzf_block(&block, &mut decompressor, &mut inflated)
            .expect("the block as written inflates");
        assert_eq!(inflated, data);
        for start in [1000, 50_000] {
            let offset = block
                .windows(64)
                .position(|window| window == &data[start..start + 64])
                .expect("a stored block holds the data as written")
                + 10;
            let mut corrupt = block.clone();
            corrupt[offset] ^= 0x20;
            let err = inflate_bgzf_block(&corrupt, &mut decompressor, &mut inflated)
                .expect_err("a changed byte fails the CRC32 check");
            assert!(err.to_string().contains("checksum mismatch"), "{err}");
        }
    }

    #[test]
    fn skippable_records_are_read_cleanly_and_skipped_by_the_scorer() {
        let mut builder = ScoreRulesBuilder::default();
        builder.push_application(ScoreApplication {
            score_index: 0,
            digits: 1,
            exponent: 0,
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
