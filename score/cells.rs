//! A compiled plan's weights as exact integers, and the one rounding of finished cells.
//!
//! A weight is held at its shortest round-trip decimal form, so a term, weight × dosage, is an
//! integer at scale `10^places × multiple`, where `multiple` is the least common multiple of the
//! denominators the score's complex rules can average over. Most scores take one scale: their
//! cell is one i64 lane when the finished value is known to stay below 2^63 in magnitude, and
//! otherwise two carry-free limbs. A score whose weights span more decimal orders than two limbs
//! hold at one scale is split into bands of decimal places, each with its own scale and lanes,
//! and its bands are combined exactly when the cell is rounded. Lanes add with wrapping
//! arithmetic: a one-limb value is exact modulo 2^64 and known to fit, so it is exact.

use crate::score::exact::{
    FixedPoint, Split, scaled_at_places, scaled_at_places_x4, shortest_decimal, shortest_decimal_hinted,
};
use crate::score::types::GroupedComplexRule;
use ahash::AHashMap;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::fmt;

/// Kernels step through lanes this many at a time; a stride is a multiple of it.
pub(crate) const LANE_WIDTH: usize = 4;
const FLIPPED: u8 = 1;
const COUNTED: u8 = 2;
/// Marks an entry whose weight does not fit i64; its weight is in `ExactPlan::wide`.
const WIDE: i64 = i64::MIN;
/// Entries a parallel pass over a plan takes per task.
const PLAN_CHUNK: usize = 1 << 14;
/// Marks an entry the places pass did not scale; a scaled one's places are below `u8::MAX`.
const UNREAD: u8 = u8::MAX;
/// Marks a score whose entries the weight pass always scales, matching no read places.
const NEVER_READ: u8 = u8::MAX - 1;
/// Fraction digits the long-division rounding writes before its sticky digit. A double's
/// midpoints have at most 767 significant decimal digits, so the parse decides correctly.
const ROUNDING_DIGITS: usize = 800;
const BILLION: u128 = 1_000_000_000;

/// Where a term goes in one person's lanes.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    lane: usize,
    split: Option<Split>,
}

/// One scale of a score: terms at `10^places × multiple`, and the lanes that hold them.
#[derive(Clone, Copy, Debug)]
struct Band {
    places: i32,
    target: Target,
    baseline: i128,
}

#[derive(Clone, Debug)]
struct ScoreArithmetic {
    multiple: u64,
    /// In ascending `places`; a term with `p` decimal places belongs to the first band with
    /// `places >= p`.
    bands: Vec<Band>,
    /// The one-division rounding of a single-band score; `None` rounds by long division.
    fixed: Option<FixedPoint>,
}

/// Why a plan has no exact form.
#[derive(Debug)]
pub enum PlanError {
    /// The arrays contradict each other.
    Invariant(String),
    /// A score's range is beyond exact integer arithmetic.
    Unrepresentable(String),
}

impl fmt::Display for PlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invariant(message) | Self::Unrepresentable(message) => f.write_str(message),
        }
    }
}

/// The exact form of a plan's CSR entries and every score's arithmetic.
#[derive(Debug)]
pub struct ExactPlan {
    /// Each entry's weight at its band's scale, or [`WIDE`].
    weights: Vec<i64>,
    wide: AHashMap<usize, i128>,
    flags: Vec<u8>,
    /// Each entry's band, when some score has more than one; otherwise empty.
    entry_band: Vec<u8>,
    scores: Vec<ScoreArithmetic>,
    stride: usize,
    /// Whether every score is one band in one lane without limbs, so score `s` is lane `s`.
    one_lane_per_score: bool,
}

fn lcm(a: u64, b: u64) -> Option<u64> {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    (a / x).checked_mul(b)
}

fn bits(magnitude: u128) -> u32 {
    128 - magnitude.leading_zeros()
}

/// Whether a score application's allele pair matches a context's, in either order.
fn application_matches(effect: &str, other: &str, allele1: &str, allele2: &str) -> bool {
    (effect == allele1 && other == allele2) || (effect == allele2 && other == allele1)
}

/// `digits × 10^exponent` at `places` decimal places times `multiple`, when it is an integer
/// that fits i128.
fn scale_digits(digits: i64, exponent: i32, places: i32, multiple: u64) -> Option<i128> {
    let shift = u32::try_from(places.checked_add(exponent)?).ok()?;
    i128::from(digits)
        .checked_mul(10i128.checked_pow(shift)?)?
        .checked_mul(i128::from(multiple))
}

/// The lanes a set of terms needs: one when their magnitudes sum below 2^63, two carry-free
/// limbs when those hold them, and `None` otherwise.
fn limbs_for(bound: u128, largest: u128, terms: u64) -> Option<Option<Split>> {
    if bound < 1u128 << 63 {
        Some(None)
    } else if bound < 1u128 << 126 {
        Split::plan(bits(largest) + 1, terms).map(Some)
    } else {
        None
    }
}

/// A score's terms with one number of decimal places: how many, and twice the sum and the
/// largest of their digits' magnitudes.
#[derive(Clone, Copy, Debug, Default)]
struct PlacesGroup {
    places: i32,
    terms: u64,
    sum: u128,
    largest: u128,
}

/// The band holding `groups` at the most places among them, if one or two limbs hold it.
fn band_for(groups: &[PlacesGroup], multiple: u64) -> Option<(i32, Option<Split>)> {
    let places = groups.last()?.places;
    let (mut bound, mut largest, mut terms) = (0u128, 0u128, 0u64);
    for group in groups {
        let scale = 10u128
            .checked_pow(u32::try_from(places - group.places).ok()?)?
            .checked_mul(u128::from(multiple))?;
        bound = bound.checked_add(group.sum.checked_mul(scale)?)?;
        largest = largest.max(group.largest.checked_mul(scale)?);
        terms = terms.checked_add(group.terms)?;
    }
    limbs_for(bound, largest, terms).map(|split| (places, split))
}

impl ExactPlan {
    /// The exact plan of CSR arrays whose every entry is one parsed weight (with its flip
    /// correction), their complex rules and score names. Refuses only a score whose weights no
    /// banding holds, naming it.
    pub fn new(
        weights: &[f64],
        corrections: &[f64],
        columns: &[u32],
        offsets: &[u64],
        complex: &[GroupedComplexRule],
        names: &[String],
    ) -> Result<Self, PlanError> {
        let num_scores = names.len();
        let entries = columns.len();
        if weights.len() != entries
            || corrections.len() != entries
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&(entries as u64))
        {
            return Err(PlanError::Invariant(
                "Plan arrays disagree on their entry count.".to_string(),
            ));
        }
        let refusal = |column: usize| {
            PlanError::Unrepresentable(format!(
                "Score '{}': its weights span more decimal orders than exact integer arithmetic holds.",
                names[column]
            ))
        };

        // Flags, and the terms each score's lanes receive: one per entry and application. Rows are
        // flagged in parallel; an entry counts a missing call when it is its score's first in its
        // row, and a nonempty row is named by its first entry.
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            return Err(PlanError::Invariant(
                "Plan row offsets decrease.".to_string(),
            ));
        }
        let mut flags = vec![0u8; entries];
        let mut rows = Vec::with_capacity(offsets.len() - 1);
        let mut rest = flags.as_mut_slice();
        for pair in offsets.windows(2) {
            let (row, tail) = std::mem::take(&mut rest).split_at_mut((pair[1] - pair[0]) as usize);
            rows.push((pair[0] as usize, row));
            rest = tail;
        }
        let mut terms = rows
            .into_par_iter()
            .try_fold(
                || (vec![usize::MAX; num_scores], vec![0u64; num_scores]),
                |(mut first_in_row, mut terms), (start, row)| {
                    for (i, flag) in (start..).zip(row.iter_mut()) {
                        let column = columns[i] as usize;
                        if column >= num_scores {
                            return Err(PlanError::Invariant(format!(
                                "Plan entry {i} names score column {column} of {num_scores}."
                            )));
                        }
                        if corrections[i] != 0.0 {
                            if corrections[i].to_bits() != (-2.0 * weights[i]).to_bits() {
                                return Err(PlanError::Invariant(format!(
                                    "Plan entry {i} carries a correction that is not its weight's flip."
                                )));
                            }
                            *flag |= FLIPPED;
                        }
                        if first_in_row[column] != start {
                            first_in_row[column] = start;
                            *flag |= COUNTED;
                        }
                        terms[column] += 1;
                    }
                    Ok((first_in_row, terms))
                },
            )
            .map(|folded| folded.map(|(_, terms)| terms))
            .try_reduce(
                || vec![0u64; num_scores],
                |mut a, b| {
                    for (x, y) in a.iter_mut().zip(b) {
                        *x += y;
                    }
                    Ok(a)
                },
            )?;

        // The single scale: the most decimal places any weight of the score needs, and the lcm of
        // its complex averaging denominators.
        let merge_max = |mut a: Vec<i32>, b: Vec<i32>| {
            for (x, y) in a.iter_mut().zip(b) {
                *x = (*x).max(y);
            }
            a
        };
        // An entry read at the places its score held then keeps that integer and those places, and
        // the weight pass below takes the integer as it is where they are the score's final places:
        // scaling a weight to its score's places a second time was a fifth of a wide panel's run.
        let mut int_weights = vec![0i64; entries];
        let mut read_places = vec![UNREAD; entries];
        let mut places = int_weights
            .par_chunks_mut(PLAN_CHUNK)
            .zip(read_places.par_chunks_mut(PLAN_CHUNK))
            .enumerate()
            .fold(
                || (vec![0i32; num_scores], 0usize),
                |(mut places, mut hint), (chunk, (slots, reads))| {
                    let start = chunk * PLAN_CHUNK;
                    let mut one = |at: usize, places: &mut [i32], slots: &mut [i64], reads: &mut [u8]| {
                        let (i, column) = (start + at, columns[start + at] as usize);
                        // A weight the score's places already hold leaves them as they are.
                        match scaled_at_places(weights[i], places[column]) {
                            Some(scaled) => (slots[at], reads[at]) = (scaled, places[column] as u8),
                            None => {
                                places[column] =
                                    places[column].max(-shortest_decimal_hinted(weights[i], &mut hint).1);
                            }
                        }
                    };
                    // Four weights at a time while all four have an integer at their scores'
                    // places, which then stay as they are; otherwise each of the four in order.
                    let quads = slots.len() / 4 * 4;
                    for at in (0..quads).step_by(4) {
                        let i = start + at;
                        let column = [
                            columns[i] as usize,
                            columns[i + 1] as usize,
                            columns[i + 2] as usize,
                            columns[i + 3] as usize,
                        ];
                        let held = [places[column[0]], places[column[1]], places[column[2]], places[column[3]]];
                        let values = [weights[i], weights[i + 1], weights[i + 2], weights[i + 3]];
                        match scaled_at_places_x4(values, held) {
                            Some(scaled) => {
                                slots[at..at + 4].copy_from_slice(&scaled);
                                let held = [held[0] as u8, held[1] as u8, held[2] as u8, held[3] as u8];
                                reads[at..at + 4].copy_from_slice(&held);
                            }
                            None => {
                                for k in 0..4 {
                                    one(at + k, &mut places, slots, reads);
                                }
                            }
                        }
                    }
                    for at in quads..slots.len() {
                        one(at, &mut places, slots, reads);
                    }
                    (places, hint)
                },
            )
            .map(|(places, _)| places)
            .reduce(|| vec![0i32; num_scores], merge_max);
        let mut multiples = vec![1u64; num_scores];
        let mut applications = Vec::new();
        for rule in complex {
            for application in &rule.score_applications {
                let column = application.score_column_index.0;
                if column >= num_scores {
                    return Err(PlanError::Invariant(format!(
                        "A complex rule names score column {column} of {num_scores}."
                    )));
                }
                places[column] = places[column].max(-shortest_decimal(application.weight).1);
                let matching = rule
                    .possible_contexts
                    .iter()
                    .filter(|(_, allele1, allele2)| {
                        application_matches(
                            &application.effect_allele,
                            &application.other_allele,
                            allele1,
                            allele2,
                        )
                    })
                    .count() as u64;
                for denominator in 2..=matching {
                    multiples[column] =
                        lcm(multiples[column], denominator).ok_or_else(|| refusal(column))?;
                }
                terms[column] += 1;
                applications.push((column, application.weight));
            }
        }

        // Each entry's weight at its score's single scale, or WIDE when it does not fit i64 (its
        // value is then in `wide`), and twice the sum and the largest of each score's term
        // magnitudes at that scale, in parallel parts. Magnitudes are not negative, so some part's
        // sum overflows exactly when the whole does: the parts cannot change a bound.
        let at_single = |value: f64, column: usize| {
            let (digits, exponent) = shortest_decimal(value);
            scale_digits(digits, exponent, places[column], multiples[column])
        };
        let add_term =
            |bound: &mut [Option<u128>], largest: &mut [u128], column: usize, weight: Option<i128>| {
                let magnitude = weight.and_then(|w| w.unsigned_abs().checked_mul(2));
                bound[column] = bound[column].zip(magnitude).and_then(|(b, m)| b.checked_add(m));
                largest[column] = largest[column].max(magnitude.unwrap_or(0));
            };
        // The places an entry read at its score's final places with a multiple of one holds: its
        // integer is then the weight at the score's scale, as scaled_at_places there gives it, below
        // 2^50 in magnitude. Such a term's doubled magnitude is below 2^51, and a part holds at most
        // every entry, fewer than 2^64 (a usize count), so a part's plain u128 sum of them stays below
        // 2^51 × 2^64 = 2^115 and cannot wrap. It joins the checked bound once, when the part is done:
        // magnitudes are not negative, so a checked add of the sum overflows exactly when adding its
        // terms one at a time would have.
        let reuse: Vec<u8> = (0..num_scores)
            .map(|column| match u8::try_from(places[column]) {
                Ok(held) if held < UNREAD && multiples[column] == 1 => held,
                _ => NEVER_READ,
            })
            .collect();
        let (mut bound, mut largest, wide_entries) = int_weights
            .par_chunks_mut(PLAN_CHUNK)
            .zip(read_places.par_chunks(PLAN_CHUNK))
            .enumerate()
            .fold(
                || {
                    let parts = (vec![Some(0u128); num_scores], vec![0u128; num_scores], Vec::new(), 0usize);
                    (parts, vec![0u128; num_scores], vec![0u64; num_scores])
                },
                |((mut bound, mut largest, mut wide, mut hint), mut reused, mut reused_largest),
                 (chunk, (slots, reads))| {
                    for ((i, slot), &read) in (chunk * PLAN_CHUNK..).zip(slots.iter_mut()).zip(reads) {
                        let column = columns[i] as usize;
                        if read == reuse[column] {
                            let magnitude = slot.unsigned_abs() << 1;
                            reused[column] += u128::from(magnitude);
                            reused_largest[column] = reused_largest[column].max(magnitude);
                            continue;
                        }
                        let exact = match scaled_at_places(weights[i], places[column]) {
                            // A multiple of one leaves the integer as it is, without the i128
                            // multiplication's libcall.
                            Some(scaled) if multiples[column] == 1 => Some(i128::from(scaled)),
                            Some(scaled) => i128::from(scaled).checked_mul(i128::from(multiples[column])),
                            None => {
                                let (digits, exponent) = shortest_decimal_hinted(weights[i], &mut hint);
                                scale_digits(digits, exponent, places[column], multiples[column])
                            }
                        };
                        *slot = exact
                            .and_then(|w| i64::try_from(w).ok())
                            .filter(|&w| w != WIDE)
                            .unwrap_or(WIDE);
                        if let (WIDE, Some(value)) = (*slot, exact) {
                            wide.push((i, value));
                        }
                        add_term(&mut bound, &mut largest, column, exact);
                    }
                    ((bound, largest, wide, hint), reused, reused_largest)
                },
            )
            .map(|((mut bound, mut largest, wide, _), reused, reused_largest)| {
                for column in 0..num_scores {
                    bound[column] = bound[column].and_then(|b| b.checked_add(reused[column]));
                    largest[column] = largest[column].max(u128::from(reused_largest[column]));
                }
                (bound, largest, wide)
            })
            .reduce(
                || (vec![Some(0u128); num_scores], vec![0u128; num_scores], Vec::new()),
                |(mut bound, mut largest, mut wide), (other_bound, other_largest, other_wide)| {
                    for (b, o) in bound.iter_mut().zip(other_bound) {
                        *b = b.zip(o).and_then(|(b, o)| b.checked_add(o));
                    }
                    for (l, o) in largest.iter_mut().zip(other_largest) {
                        *l = (*l).max(o);
                    }
                    wide.extend(other_wide);
                    (bound, largest, wide)
                },
            );
        let mut wide = AHashMap::new();
        for (i, value) in wide_entries {
            wide.insert(i, value);
        }
        for &(column, weight) in &applications {
            add_term(&mut bound, &mut largest, column, at_single(weight, column));
        }
        let single: Vec<Option<Option<Split>>> = (0..num_scores)
            .map(|column| {
                bound[column].and_then(|bound| limbs_for(bound, largest[column], terms[column]))
            })
            .collect();

        // Scores no single scale holds are banded by decimal places.
        let banded: Vec<bool> = single.iter().map(Option::is_none).collect();
        let mut specs: Vec<Vec<(i32, Option<Split>)>> = single
            .iter()
            .enumerate()
            .map(|(column, split)| split.map_or_else(Vec::new, |split| vec![(places[column], split)]))
            .collect();
        let mut entry_band = Vec::new();
        if banded.iter().any(|&b| b) {
            let mut groups: Vec<AHashMap<i32, PlacesGroup>> = vec![AHashMap::new(); num_scores];
            let mut gather = |column: usize, value: f64| {
                let (digits, exponent) = shortest_decimal(value);
                let magnitude = 2 * u128::from(digits.unsigned_abs());
                let group = groups[column].entry(-exponent).or_insert(PlacesGroup {
                    places: -exponent,
                    ..PlacesGroup::default()
                });
                group.terms += 1;
                group.sum = group.sum.saturating_add(magnitude);
                group.largest = group.largest.max(magnitude);
            };
            for i in 0..entries {
                if banded[columns[i] as usize] {
                    gather(columns[i] as usize, weights[i]);
                }
            }
            for &(column, weight) in &applications {
                if banded[column] {
                    gather(column, weight);
                }
            }
            for column in (0..num_scores).filter(|&column| banded[column]) {
                let mut sorted: Vec<PlacesGroup> = groups[column].values().copied().collect();
                sorted.sort_by_key(|group| group.places);
                let multiple = multiples[column];
                let mut start = 0;
                let mut current = band_for(&sorted[..1], multiple).ok_or_else(|| refusal(column))?;
                for end in 2..=sorted.len() {
                    match band_for(&sorted[start..end], multiple) {
                        Some(band) => current = band,
                        None => {
                            specs[column].push(current);
                            start = end - 1;
                            current = band_for(&sorted[start..end], multiple)
                                .ok_or_else(|| refusal(column))?;
                        }
                    }
                }
                specs[column].push(current);
                if specs[column].len() > usize::from(u8::MAX) {
                    return Err(refusal(column));
                }
            }
            entry_band = vec![0u8; entries];
            for i in 0..entries {
                let column = columns[i] as usize;
                if !banded[column] {
                    continue;
                }
                let (digits, exponent) = shortest_decimal(weights[i]);
                let band = specs[column]
                    .iter()
                    .position(|&(band_places, _)| band_places >= -exponent)
                    .ok_or_else(|| refusal(column))?;
                entry_band[i] = band as u8;
                let weight = scale_digits(digits, exponent, specs[column][band].0, multiples[column])
                    .ok_or_else(|| refusal(column))?;
                wide.remove(&i);
                match i64::try_from(weight).ok().filter(|&w| w != WIDE) {
                    Some(narrow) => int_weights[i] = narrow,
                    None => {
                        int_weights[i] = WIDE;
                        wide.insert(i, weight);
                    }
                }
            }
        }

        let mut scores = Vec::with_capacity(num_scores);
        let mut lane = 0usize;
        for column in 0..num_scores {
            let bands: Vec<Band> = specs[column]
                .iter()
                .map(|&(band_places, split)| {
                    let band = Band {
                        places: band_places,
                        target: Target { lane, split },
                        baseline: 0,
                    };
                    lane += if split.is_some() { 2 } else { 1 };
                    band
                })
                .collect();
            // One band rounds by one division at its own places. A banded score's single band holds
            // weights no scale at the score's places could, so its places can be below them.
            let fixed = match bands.as_slice() {
                [band] => u32::try_from(band.places)
                    .ok()
                    .and_then(|places| 5u128.checked_pow(places))
                    .and_then(|power| power.checked_mul(u128::from(multiples[column])))
                    .filter(|scale| scale.checked_mul(u128::from(u32::MAX)).is_some())
                    .map(|scale| FixedPoint {
                        exp: -band.places,
                        scale,
                    }),
                _ => None,
            };
            scores.push(ScoreArithmetic {
                multiple: multiples[column],
                bands,
                fixed,
            });
        }
        // Lanes go to scores in column order, so when every score is one band in one lane, score `s`
        // is lane `s`, and every weight fits i64 because its score's magnitudes sum below 2^63.
        let one_lane_per_score = scores
            .iter()
            .all(|score| matches!(score.bands.as_slice(), [band] if band.target.split.is_none()));
        let mut plan = Self {
            weights: int_weights,
            wide,
            flags,
            entry_band,
            scores,
            stride: lane.div_ceil(LANE_WIDTH).max(1) * LANE_WIDTH,
            one_lane_per_score,
        };
        // The flipped-allele baseline of every band: two doses of each flipped entry's effect, in
        // parallel parts. A band's term magnitudes sum below 2^126, so no part's sum overflows.
        let mut band_starts = vec![0usize; num_scores + 1];
        for (column, score) in plan.scores.iter().enumerate() {
            band_starts[column + 1] = band_starts[column] + score.bands.len();
        }
        let baselines = {
            let plan = &plan;
            let band_starts = &band_starts;
            plan.flags
                .par_chunks(PLAN_CHUNK)
                .enumerate()
                .fold(
                    || vec![Some(0i128); band_starts[num_scores]],
                    |mut sums, (chunk, flags)| {
                        for (i, &flag) in (chunk * PLAN_CHUNK..).zip(flags) {
                            if flag & FLIPPED != 0 {
                                let column = columns[i] as usize;
                                let sum = &mut sums[band_starts[column] + plan.band_of(i, column)];
                                *sum = sum.and_then(|sum| sum.checked_sub(2 * plan.weight(i)));
                            }
                        }
                        sums
                    },
                )
                .reduce(
                    || vec![Some(0i128); band_starts[num_scores]],
                    |mut a, b| {
                        for (x, y) in a.iter_mut().zip(b) {
                            *x = x.zip(y).and_then(|(x, y)| x.checked_add(y));
                        }
                        a
                    },
                )
        };
        for (column, score) in plan.scores.iter_mut().enumerate() {
            for (band, sum) in score
                .bands
                .iter_mut()
                .zip(&baselines[band_starts[column]..band_starts[column + 1]])
            {
                band.baseline = sum.ok_or_else(|| refusal(column))?;
            }
        }
        Ok(plan)
    }

    /// The lanes a person's cell takes, a multiple of [`LANE_WIDTH`].
    #[inline(always)]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Whether every score is one band in one lane without limbs, so score `s` is lane `s` and
    /// [`Self::narrow_entry`] gives every entry.
    #[inline(always)]
    pub fn one_lane_per_score(&self) -> bool {
        self.one_lane_per_score
    }

    /// `entry`'s weight at its score's scale and whether it is flipped, for a plan with one lane per
    /// score, where every weight fits i64.
    #[inline(always)]
    pub fn narrow_entry(&self, entry: usize) -> (i64, bool) {
        (self.weights[entry], self.flags[entry] & FLIPPED != 0)
    }

    /// The bytes this plan holds on the heap: its per-entry vectors at their capacity, the wide
    /// weights' buckets and control bytes, and each score's bands.
    pub fn heap_bytes(&self) -> usize {
        let wide = match self.wide.capacity() {
            0 => 0,
            capacity => (capacity + 1)
                .saturating_mul(8)
                .div_ceil(7)
                .next_power_of_two()
                .saturating_mul(std::mem::size_of::<(usize, i128)>() + 1)
                .saturating_add(16),
        };
        let bands = self
            .scores
            .iter()
            .map(|score| score.bands.capacity() * std::mem::size_of::<Band>())
            .sum::<usize>();
        (self.weights.capacity() * std::mem::size_of::<i64>())
            .saturating_add(wide)
            .saturating_add(self.flags.capacity())
            .saturating_add(self.entry_band.capacity())
            .saturating_add(self.scores.capacity() * std::mem::size_of::<ScoreArithmetic>())
            .saturating_add(bands)
    }

    #[inline(always)]
    fn weight(&self, entry: usize) -> i128 {
        match self.weights[entry] {
            WIDE => self.wide[&entry],
            narrow => i128::from(narrow),
        }
    }

    #[inline(always)]
    fn band_of(&self, entry: usize, score: usize) -> usize {
        if self.scores[score].bands.len() == 1 {
            0
        } else {
            usize::from(self.entry_band[entry])
        }
    }

    /// Whether `entry` is the first of its score in its row, so a missing call there counts.
    #[inline(always)]
    pub fn counts_missing(&self, entry: usize) -> bool {
        self.flags[entry] & COUNTED != 0
    }

    /// Where the terms of `entry`, an entry of `score`, go.
    #[inline(always)]
    pub fn entry_target(&self, entry: usize, score: usize) -> Target {
        self.scores[score].bands[self.band_of(entry, score)].target
    }

    /// What `entry` adds for PLINK codes [00, 01, 10, 11]: nothing, the missing call's
    /// cancellation of the flipped-allele baseline, one dose and two doses of its weight.
    #[inline(always)]
    pub fn entry_terms(&self, entry: usize) -> [i128; 4] {
        let weight = self.weight(entry);
        let missing = if self.flags[entry] & FLIPPED != 0 {
            2 * weight
        } else {
            0
        };
        [0, missing, weight, 2 * weight]
    }

    /// Adds `term` to one person's lanes at `target`.
    #[inline(always)]
    pub fn add(&self, target: Target, term: i128, lanes: &mut [i64]) {
        match target.split {
            None => lanes[target.lane] = lanes[target.lane].wrapping_add(term as i64),
            Some(split) => {
                let (lo, hi) = split.parts(term);
                lanes[target.lane] = lanes[target.lane].wrapping_add(lo);
                lanes[target.lane + 1] = lanes[target.lane + 1].wrapping_add(hi);
            }
        }
    }

    /// The band of a complex application of `weight` to `score`.
    fn application_band(&self, score: usize, weight: f64) -> (usize, i128) {
        let arithmetic = &self.scores[score];
        let (digits, exponent) = shortest_decimal(weight);
        let band = arithmetic
            .bands
            .iter()
            .position(|band| band.places >= -exponent)
            .expect("the plan banded every application's places");
        let scaled = scale_digits(digits, exponent, arithmetic.bands[band].places, arithmetic.multiple)
            .expect("the plan bounded every application's weight");
        (band, scaled)
    }

    /// Where a complex application of `weight` to `score` adds its terms.
    pub fn complex_target(&self, score: usize, weight: f64) -> Target {
        self.scores[score].bands[self.application_band(score, weight).0].target
    }

    /// The term a complex application of `weight` to `score` adds for a dosage of
    /// `numerator / denominator`, where `denominator` is at most the application's matching
    /// contexts and so divides the score's multiple.
    pub fn complex_term(&self, score: usize, weight: f64, numerator: u32, denominator: u32) -> i128 {
        self.application_band(score, weight).1 * i128::from(numerator) / i128::from(denominator.max(1))
    }

    /// A band's finished value, baseline included, from one person's lanes.
    #[inline(always)]
    fn band_value(band: &Band, lanes: &[i64]) -> i128 {
        let target = band.target;
        match target.split {
            None => i128::from((i128::from(lanes[target.lane]) + band.baseline) as i64),
            Some(split) => split.join(lanes[target.lane], lanes[target.lane + 1]) + band.baseline,
        }
    }

    /// `Σ band values × 10^-places / (multiple × divisor)` correctly rounded; 0 when `divisor` is 0.
    fn round(&self, score: usize, value: impl Fn(&Band) -> i128, divisor: u32) -> f64 {
        let arithmetic = &self.scores[score];
        if let (Some(fixed), [band]) = (arithmetic.fixed, arithmetic.bands.as_slice()) {
            return fixed.quotient(value(band), divisor);
        }
        if divisor == 0 {
            return 0.0;
        }
        round_long(
            &arithmetic
                .bands
                .iter()
                .map(|band| (value(band), band.places))
                .collect::<Vec<_>>(),
            u128::from(arithmetic.multiple) * u128::from(divisor),
        )
    }

    /// The correctly rounded sum of `score` for one person's finished lanes.
    #[inline]
    pub fn sum(&self, score: usize, lanes: &[i64]) -> f64 {
        self.round(score, |band| Self::band_value(band, lanes), 1)
    }

    /// The correctly rounded average over `used` variants; 0 when none were used.
    #[inline]
    pub fn average(&self, score: usize, lanes: &[i64], used: u32) -> f64 {
        self.round(score, |band| Self::band_value(band, lanes), used)
    }

    /// `entry`'s weight as the f64 it was parsed from.
    pub fn entry_weight_f64(&self, entry: usize, score: usize) -> f64 {
        let band = self.band_of(entry, score);
        let weight = self.weight(entry);
        self.round(
            score,
            |candidate| {
                if std::ptr::eq(candidate, &self.scores[score].bands[band]) {
                    weight
                } else {
                    0
                }
            },
            1,
        )
    }

    /// `entry`'s missing correction as the f64 the join computed.
    pub fn entry_correction_f64(&self, entry: usize, score: usize) -> f64 {
        if self.flags[entry] & FLIPPED != 0 {
            -2.0 * self.entry_weight_f64(entry, score)
        } else {
            0.0
        }
    }

    /// The flipped-allele baseline of `score`, rounded.
    pub fn baseline_f64(&self, score: usize) -> f64 {
        self.round(score, |band| band.baseline, 1)
    }
}

/// Base-10^9 digits, least significant first, without trailing zero limbs.
pub(crate) fn limbs_of(mut value: u128) -> Vec<u32> {
    let mut limbs = Vec::new();
    while value > 0 {
        limbs.push((value % BILLION) as u32);
        value /= BILLION;
    }
    limbs
}

pub(crate) fn shift_decimal(limbs: &mut Vec<u32>, digits: u32) {
    if limbs.is_empty() {
        return;
    }
    let factor = 10u64.pow(digits % 9);
    let mut carry = 0u64;
    for limb in limbs.iter_mut() {
        let x = u64::from(*limb) * factor + carry;
        *limb = (x % BILLION as u64) as u32;
        carry = x / BILLION as u64;
    }
    if carry > 0 {
        limbs.push(carry as u32);
    }
    limbs.splice(0..0, std::iter::repeat_n(0, (digits / 9) as usize));
}

pub(crate) fn add_limbs(sum: &mut Vec<u32>, value: &[u32]) {
    if sum.len() < value.len() {
        sum.resize(value.len(), 0);
    }
    let mut carry = 0u64;
    for (i, limb) in sum.iter_mut().enumerate() {
        let x = u64::from(*limb) + u64::from(value.get(i).copied().unwrap_or(0)) + carry;
        *limb = (x % BILLION as u64) as u32;
        carry = x / BILLION as u64;
    }
    if carry > 0 {
        sum.push(carry as u32);
    }
}

fn trim(limbs: &mut Vec<u32>) {
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
}

pub(crate) fn compare_limbs(a: &[u32], b: &[u32]) -> Ordering {
    a.len()
        .cmp(&b.len())
        .then_with(|| a.iter().rev().cmp(b.iter().rev()))
}

/// `a - b` for `a >= b`.
pub(crate) fn subtract_limbs(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len());
    let mut borrow = 0i64;
    for (i, &limb) in a.iter().enumerate() {
        let mut x = i64::from(limb) - i64::from(b.get(i).copied().unwrap_or(0)) - borrow;
        borrow = i64::from(x < 0);
        if x < 0 {
            x += BILLION as i64;
        }
        out.push(x as u32);
    }
    trim(&mut out);
    out
}

/// `Σ value × 10^-places` over `values`, divided by `denominator` (below 2^96), correctly
/// rounded: the quotient's decimal expansion to [`ROUNDING_DIGITS`] fraction digits, a sticky
/// digit when more follow, parsed by `str::parse`, which rounds a decimal of any length correctly.
pub(crate) fn round_long(values: &[(i128, i32)], denominator: u128) -> f64 {
    let top = values.iter().map(|&(_, places)| places).max().unwrap_or(0);
    let (mut positive, mut negative) = (Vec::new(), Vec::new());
    for &(value, places) in values {
        let mut limbs = limbs_of(value.unsigned_abs());
        shift_decimal(&mut limbs, (top - places) as u32);
        add_limbs(if value < 0 { &mut negative } else { &mut positive }, &limbs);
    }
    trim(&mut positive);
    trim(&mut negative);
    let (negative_result, magnitude) = match compare_limbs(&positive, &negative) {
        Ordering::Less => (true, subtract_limbs(&negative, &positive)),
        _ => (false, subtract_limbs(&positive, &negative)),
    };
    round_decimal(negative_result, &magnitude, top, denominator)
}

/// `±magnitude × 10^-top / denominator` (below 2^96), correctly rounded, as [`round_long`] rounds.
pub(crate) fn round_decimal(negative_result: bool, magnitude: &[u32], top: i32, denominator: u128) -> f64 {
    if magnitude.is_empty() {
        return 0.0;
    }
    let mut quotient = vec![0u32; magnitude.len()];
    let mut remainder = 0u128;
    for (i, &limb) in magnitude.iter().enumerate().rev() {
        let current = remainder * BILLION + u128::from(limb);
        quotient[i] = (current / denominator) as u32;
        remainder = current % denominator;
    }
    trim(&mut quotient);
    let mut text = String::with_capacity(ROUNDING_DIGITS + 64);
    if negative_result {
        text.push('-');
    }
    match quotient.split_last() {
        None => text.push('0'),
        Some((head, rest)) => {
            text.push_str(&head.to_string());
            for limb in rest.iter().rev() {
                text.push_str(&format!("{limb:09}"));
            }
        }
    }
    text.push('.');
    for _ in 0..ROUNDING_DIGITS {
        remainder *= 10;
        text.push(char::from(b'0' + (remainder / denominator) as u8));
        remainder %= denominator;
    }
    if remainder != 0 {
        text.push('1');
    }
    text.push_str(&format!("e{}", -top));
    text.parse().expect("a decimal numeral")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::types::{BimRowIndex, ScoreColumnIndex, ScoreInfo};

    fn names(count: usize) -> Vec<String> {
        (0..count).map(|i| format!("S{i}")).collect()
    }

    /// One row per weight, all in score 0, and every entry's terms for `code` added to one cell.
    fn one_person(plan: &ExactPlan, codes: &[usize]) -> Vec<i64> {
        let mut lanes = vec![0i64; plan.stride()];
        for (entry, &code) in codes.iter().enumerate() {
            plan.add(plan.entry_target(entry, 0), plan.entry_terms(entry)[code], &mut lanes);
        }
        lanes
    }

    fn rows(count: usize) -> Vec<u64> {
        (0..=count as u64).collect()
    }

    #[test]
    fn flips_duplicates_and_mixed_places_score_the_written_decimals() {
        // Row 0: score 0 twice (a duplicate line, one flipped), score 1 once.
        // Row 1: score 1 with a three-place weight.
        let weights = [0.25, -0.125, 1.5, 0.001];
        let corrections = [0.0, 0.25, 0.0, 0.0];
        let columns = [0, 0, 1, 1];
        let offsets = [0, 3, 4];
        let plan = ExactPlan::new(&weights, &corrections, &columns, &offsets, &[], &names(2))
            .expect("plan");
        assert_eq!(plan.stride(), LANE_WIDTH);
        assert!(plan.counts_missing(0) && !plan.counts_missing(1));
        assert!(plan.counts_missing(2) && plan.counts_missing(3));
        // Person: row 0 het, row 1 hom-alt.
        let mut lanes = vec![0i64; plan.stride()];
        for (entry, code) in [(0, 2), (1, 2), (2, 2), (3, 3)] {
            let score = columns[entry] as usize;
            plan.add(plan.entry_target(entry, score), plan.entry_terms(entry)[code], &mut lanes);
        }
        // Score 0: 0.25 + (0.25 baseline - 0.125) = 0.375. Score 1: 1.5 + 0.002.
        assert_eq!(plan.sum(0, &lanes), 0.375);
        assert_eq!(plan.sum(1, &lanes), 1.502);
        assert_eq!(plan.average(1, &lanes, 2), 0.751);
        for entry in 0..4 {
            let score = columns[entry] as usize;
            assert_eq!(plan.entry_weight_f64(entry, score).to_bits(), weights[entry].to_bits());
            assert_eq!(
                plan.entry_correction_f64(entry, score).to_bits(),
                corrections[entry].to_bits()
            );
        }
    }

    #[test]
    fn a_missing_call_cancels_the_flipped_baseline() {
        let plan = ExactPlan::new(&[-0.7], &[1.4], &[0], &[0, 1], &[], &names(1)).expect("plan");
        let expect = [1.4, 0.0, 0.7, 0.0];
        for (code, want) in expect.into_iter().enumerate() {
            assert_eq!(plan.sum(0, &one_person(&plan, &[code])), want, "code {code}");
        }
        assert_eq!(plan.baseline_f64(0), 1.4);
    }

    #[test]
    fn complex_averages_divide_exactly_by_their_multiple() {
        let rule = GroupedComplexRule {
            locus_chr_pos: ("1".to_string(), 100),
            possible_contexts: (0..3)
                .map(|row| (BimRowIndex(row), "A".to_string(), "G".to_string()))
                .collect(),
            score_applications: vec![ScoreInfo {
                effect_allele: "G".to_string(),
                other_allele: "A".to_string(),
                weight: 0.1,
                score_column_index: ScoreColumnIndex(0),
            }],
        };
        let plan = ExactPlan::new(&[], &[], &[], &[0], &[rule], &names(1)).expect("plan");
        let mut lanes = vec![0i64; plan.stride()];
        plan.add(plan.complex_target(0, 0.1), plan.complex_term(0, 0.1, 4, 3), &mut lanes);
        assert_eq!(plan.sum(0, &lanes), 0.4 / 3.0);
        assert_eq!(plan.complex_term(0, 0.1, 1, 1) * 3, plan.complex_term(0, 0.1, 3, 1));
    }

    #[test]
    fn wide_scores_take_two_limbs_and_stay_exact() {
        // 1e10 at 17 places over 2,000 entries: every term needs about 90 bits.
        let count = 2_000;
        let weights: Vec<f64> = (0..count).map(|i| 1e10 + (i as f64) * 1e-6).collect();
        let columns = vec![0u32; count];
        let plan = ExactPlan::new(&weights, &vec![0.0; count], &columns, &rows(count), &[], &names(1))
            .expect("plan");
        assert_eq!(plan.scores[0].bands.len(), 1);
        assert!(plan.scores[0].bands[0].target.split.is_some());
        let lanes = one_person(&plan, &vec![3; count]);
        let exact: i128 = (0..count).map(|entry| plan.entry_terms(entry)[3]).sum();
        assert_eq!(ExactPlan::band_value(&plan.scores[0].bands[0], &lanes), exact);
    }

    #[test]
    fn reused_terms_bound_a_score_exactly_at_the_one_lane_limit() {
        // Integer weights of the largest magnitude a reused term holds, 2^50 - 1 at places 0, and
        // 2^47 - 1 over more entries than one pass part takes. 2^63 / (2 × weight) such terms fit
        // one lane exactly (bound 2^63 - 2^13, and 2^63 - 2^16); one more needs two limbs. Either
        // way every sum is the exact integer sum.
        for (weight, fits) in [((1u64 << 50) - 1, 1usize << 12), ((1u64 << 47) - 1, 1 << 15)] {
            assert!(fits > PLAN_CHUNK || weight == (1u64 << 50) - 1);
            for count in [fits, fits + 1] {
                let weights = vec![weight as f64; count];
                let plan =
                    ExactPlan::new(&weights, &vec![0.0; count], &vec![0u32; count], &rows(count), &[], &names(1))
                        .expect("plan");
                let band = &plan.scores[0].bands[0];
                assert_eq!(plan.scores[0].bands.len(), 1);
                assert_eq!(band.places, 0);
                assert_eq!(band.target.split.is_some(), count > fits, "weight {weight} count {count}");
                let lanes = one_person(&plan, &vec![3; count]);
                assert_eq!(ExactPlan::band_value(band, &lanes), 2 * i128::from(weight) * count as i128);
            }
        }
    }

    #[test]
    fn scores_spanning_many_decimal_orders_are_banded_and_round_exactly() {
        // The plan-cache fixture's panel: 0.25, 1e-40, 3 and -2.5 in one score.
        let weights = [0.25, 1e-40, 3.0, -2.5];
        let plan = ExactPlan::new(&weights, &[0.0; 4], &[0; 4], &rows(4), &[], &names(1)).expect("plan");
        assert!(plan.scores[0].bands.len() > 1);
        for (entry, &weight) in weights.iter().enumerate() {
            assert_eq!(plan.entry_weight_f64(entry, 0).to_bits(), weight.to_bits());
        }
        // Het on every row: 0.25 + 1e-40 + 3 - 2.5; the 1e-40 survives only with 3 - 2.5 - 0.75.
        let lanes = one_person(&plan, &[2, 2, 2, 2]);
        let want: f64 = "0.7500000000000000000000000000000000000001".parse().unwrap();
        assert_eq!(plan.sum(0, &lanes).to_bits(), want.to_bits());
        let weights = [0.75, 1e-40, -3.0, 2.25];
        let plan = ExactPlan::new(&weights, &[0.0; 4], &[0; 4], &rows(4), &[], &names(1)).expect("plan");
        let lanes = one_person(&plan, &[2, 2, 2, 2]);
        assert_eq!(plan.sum(0, &lanes).to_bits(), 1e-40f64.to_bits());
        assert_eq!(plan.average(0, &lanes, 4).to_bits(), 2.5e-41f64.to_bits());
        // Extremes of the double range round-trip.
        let weights = [1e300, 1e-300, 5e-324, -1.7976931348623157e308];
        let plan = ExactPlan::new(&weights, &[0.0; 4], &[0; 4], &rows(4), &[], &names(1)).expect("plan");
        for (entry, &weight) in weights.iter().enumerate() {
            assert_eq!(plan.entry_weight_f64(entry, 0).to_bits(), weight.to_bits());
        }
    }

    #[test]
    fn long_division_rounds_like_the_one_division_path() {
        let fixed = FixedPoint {
            exp: -6,
            scale: 5u128.pow(6) * 3,
        };
        for (value, divisor) in [(1i128, 1u32), (-7, 3), (123_456_789, 7), (i128::MAX, 11), (5, 2)] {
            assert_eq!(
                round_long(&[(value, 6)], 3 * u128::from(divisor)).to_bits(),
                fixed.quotient(value, divisor).to_bits(),
                "{value} / {divisor}"
            );
        }
        assert_eq!(round_long(&[(5, 1), (-5, 1)], 1), 0.0);
    }

    #[test]
    fn a_banded_score_of_one_band_rounds_at_the_band_places() {
        // At no decimal places the terms' doubled magnitudes sum past 2^126, while one band at -37
        // places, where the weights are 3 and 5, holds them in one lane.
        let plan = ExactPlan::new(&[3e37, 5e37], &[0.0; 2], &[0; 2], &rows(2), &[], &names(1)).expect("plan");
        assert_eq!(plan.scores[0].bands.len(), 1);
        assert_eq!(plan.scores[0].bands[0].places, -37);
        assert_eq!(plan.sum(0, &one_person(&plan, &[2, 2])), 8e37);
        assert_eq!(plan.sum(0, &one_person(&plan, &[3, 2])), 1.1e38);
        assert_eq!(plan.average(0, &one_person(&plan, &[3, 2]), 2), 5.5e37);
        for (entry, weight) in [3e37f64, 5e37].into_iter().enumerate() {
            assert_eq!(plan.entry_weight_f64(entry, 0).to_bits(), weight.to_bits());
        }
        // A flipped weight's baseline rounds at the band's places too: two doses of 3e37 at code 00,
        // one at code 10.
        let plan = ExactPlan::new(&[-3e37, 5e37], &[6e37, 0.0], &[0; 2], &rows(2), &[], &names(1)).expect("plan");
        assert_eq!(plan.baseline_f64(0), 6e37);
        assert_eq!(plan.sum(0, &one_person(&plan, &[0, 2])), 1.1e38);
        assert_eq!(plan.sum(0, &one_person(&plan, &[2, 3])), 1.3e38);
    }

    #[test]
    fn merged_corrections_are_refused() {
        assert!(ExactPlan::new(&[0.5], &[0.3], &[0], &[0, 1], &[], &names(1)).is_err());
    }
}
