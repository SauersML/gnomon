//! The per-row account of the score rows that add no variant to their score (#2383).
//!
//! `--unmatched-report PATH` writes a line for each weight of a score row that adds no variant to
//! its score: the score, the row's variant and alleles as its normalized score file writes them,
//! why, and the alleles the genotypes hold at its position. So for every score, the weights its
//! rows hold are its #SCORE_VARIANT_COUNT and its lines in the report, exactly. Both scoring
//! paths decide a row by the one site rule of [`crate::score::site`], so the report is the same
//! file for a joined or split VCF, a BCF or a PGEN of the same genotypes, and for a `.bed`
//! wherever its alleles read as the REF the others declare (a `.bim` declares none).
//!
//! Why a row adds no variant:
//! - `no_variant_at_position`: the genotypes hold no variant at its position.
//! - `no_allele_pair`: they hold variants there, but none carries the row's alleles.
//! - `several_alleles`: the row names no single other allele, and its effect allele is more than
//!   one allele of the site.
//! - `outside_region`: its score is restricted to a region the row lies outside.
//! - `no_other_allele`: its other allele is N, which pairs with no variant.
//! - `same_variant`: another row of the score names the variant this row names. A variant counts
//!   once in its score however many rows name it, and each row's weight adds to its dose; of the
//!   rows naming one variant, the first by effect and then other allele text is the one counted.
//!
//! A row the normalization drops before scoring, one without a position, on an unsupported
//! contig or malformed, is named with its line in the normalization's warning instead.

use crate::score::site::{RowMatch, names_no_single_other_allele, trimmed};
use std::io::{self, Write};
use std::path::Path;
use std::sync::Arc;

/// Why a weight of a score row adds no variant to its score.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Unmatched {
    NoVariant,
    NoAllelePair,
    SeveralAlleles,
    OutsideRegion,
    NoOtherAllele,
    SameVariant,
}

impl Unmatched {
    fn label(self) -> &'static str {
        match self {
            Self::NoVariant => "no_variant_at_position",
            Self::NoAllelePair => "no_allele_pair",
            Self::SeveralAlleles => "several_alleles",
            Self::OutsideRegion => "outside_region",
            Self::NoOtherAllele => "no_other_allele",
            Self::SameVariant => "same_variant",
        }
    }

    /// Why a row naming `other_allele` that meets its site as `decision` is not scored, or `None`
    /// when it is scored or refused: a row two variants carry, or whose dose depends on how the site
    /// is read, is not in the report, since the run refuses it. A row naming no single other allele whose effect allele the
    /// site's readings make different alleles is dropped as one that is several alleles is.
    pub(crate) fn of(decision: RowMatch, other_allele: &str) -> Option<Self> {
        match decision {
            RowMatch::NoVariant => Some(Self::NoAllelePair),
            RowMatch::Several(..) | RowMatch::Unread(_) if names_no_single_other_allele(other_allele) => {
                Some(Self::SeveralAlleles)
            }
            RowMatch::Several(..) | RowMatch::Unread(_) | RowMatch::Scores(_) => None,
        }
    }
}

/// The rows among `named` that add no variant to their score, each named by the handle `T` a
/// caller keeps it by: of the rows naming one variant for one score, each entry `(score, variant,
/// effect allele, other allele, handle)`, every row after the first by effect and then other
/// allele text. Sorts `named`.
pub(crate) fn same_variant<'n, T: Copy>(
    named: &'n mut [(usize, usize, &str, &str, T)],
) -> impl Iterator<Item = T> + 'n {
    named.sort_unstable_by(|a, b| (a.0, a.1, a.2, a.3).cmp(&(b.0, b.1, b.2, b.3)));
    named
        .chunk_by(|a, b| (a.0, a.1) == (b.0, b.1))
        .flat_map(|rows| rows[1..].iter().map(|row| row.4))
}

/// The alleles a site's rows hold, as the report writes them: each row's two alleles without the
/// trailing bases they share, in text order, and the rows in text order, comma-separated. Every
/// representation holds one row an ALT, so a joined record and the records it splits into, and a
/// `.bim` row whichever allele it writes first, give the same text.
pub(crate) fn alleles_seen<'a>(rows: impl Iterator<Item = (&'a str, &'a str)>) -> Arc<str> {
    let mut pairs: Vec<(&str, &str)> = rows
        .map(|(first, second)| {
            let (first, second) = trimmed(first, second);
            if first <= second { (first, second) } else { (second, first) }
        })
        .collect();
    pairs.sort_unstable();
    let text: Vec<String> = pairs.iter().map(|(first, second)| format!("{first}/{second}")).collect();
    text.join(",").into()
}

/// One line of the report: an unscored weight's score, its row's position and alleles, why, and
/// the alleles seen at the position. Each path keeps its own compact record of a line and gives
/// this view of it as the report is written.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Line<'a> {
    pub(crate) score: &'a str,
    pub(crate) key: (u8, u32),
    pub(crate) effect_allele: &'a str,
    pub(crate) other_allele: &'a str,
    pub(crate) reason: Unmatched,
    pub(crate) seen: &'a str,
}

/// The unscored weights of a run that keeps its rows' alleles as text, for `--unmatched-report`.
#[derive(Debug)]
pub(crate) struct UnmatchedRows {
    /// The rows' alleles, which the lines hold as spans.
    alleles: String,
    lines: Vec<RowLine>,
    /// The alleles seen at each position with an unscored weight, after none at all.
    seen: Vec<Arc<str>>,
}

/// An unscored weight: its score, its row's position and alleles, why, and the alleles seen there,
/// by index into `UnmatchedRows::seen`.
#[derive(Debug, Clone, Copy)]
struct RowLine {
    score: usize,
    key: (u8, u32),
    effect_allele: (usize, usize),
    other_allele: (usize, usize),
    reason: Unmatched,
    seen: u32,
}

impl UnmatchedRows {
    pub(crate) fn new() -> Self {
        Self {
            alleles: String::new(),
            lines: Vec::new(),
            seen: vec![Arc::from("")],
        }
    }

    /// Keeps the alleles seen at a position, and gives their index.
    pub(crate) fn add_seen(&mut self, seen: Arc<str>) -> u32 {
        self.seen.push(seen);
        (self.seen.len() - 1) as u32
    }

    /// Records a weight of score `score`, of the row at `key` naming `effect_allele` and
    /// `other_allele`, as unscored for `reason`, with the alleles seen at `seen`; 0 is none.
    pub(crate) fn add(
        &mut self,
        score: usize,
        key: (u8, u32),
        (effect_allele, other_allele): (&str, &str),
        reason: Unmatched,
        seen: u32,
    ) {
        let mut span = |allele: &str| {
            let start = self.alleles.len();
            self.alleles.push_str(allele);
            (start, self.alleles.len())
        };
        let (effect_allele, other_allele) = (span(effect_allele), span(other_allele));
        self.lines.push(RowLine {
            score,
            key,
            effect_allele,
            other_allele,
            reason,
            seen,
        });
    }

    /// Writes the report of these weights to `path`, naming score `i` `score_names[i]`.
    pub(crate) fn write(mut self, path: &Path, score_names: &[String]) -> io::Result<()> {
        let (alleles, seen) = (std::mem::take(&mut self.alleles), std::mem::take(&mut self.seen));
        let line_of = |line: &RowLine| Line {
            score: &score_names[line.score],
            key: line.key,
            effect_allele: &alleles[line.effect_allele.0..line.effect_allele.1],
            other_allele: &alleles[line.other_allele.0..line.other_allele.1],
            reason: line.reason,
            seen: &seen[line.seen as usize],
        };
        self.lines.sort_unstable_by(|a, b| line_of(a).cmp(&line_of(b)));
        write_report(path, self.lines.iter().map(line_of))
    }
}

/// The chromosome's label as the report writes it.
fn chromosome_label(code: u8) -> String {
    match code {
        23 => "X".to_string(),
        24 => "Y".to_string(),
        25 => "XY".to_string(),
        26 => "MT".to_string(),
        n => n.to_string(),
    }
}

/// Writes the report of `lines` to `path`, in the order they come. A path sorts its lines by their
/// [`Line`] first, by score name, position, alleles and reason, so every path writes one file for
/// one set of rows.
pub(crate) fn write_report<'a>(path: &Path, lines: impl Iterator<Item = Line<'a>>) -> io::Result<()> {
    crate::output::write_atomically(path, |writer| {
        writeln!(writer, "#score\tvariant_id\teffect_allele\tother_allele\treason\talleles_seen")?;
        for line in lines {
            let Line {
                score,
                key: (chromosome, position),
                effect_allele,
                other_allele,
                reason,
                seen,
            } = line;
            let seen = if seen.is_empty() { "." } else { seen };
            let chromosome = chromosome_label(chromosome);
            writeln!(
                writer,
                "{score}\t{chromosome}:{position}\t{effect_allele}\t{other_allele}\t{}\t{seen}",
                reason.label()
            )?;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::site::Site;

    #[test]
    fn a_joined_record_its_split_records_and_a_bim_row_see_the_same_alleles() {
        let joined = alleles_seen([("CA", "CAA"), ("CA", "C")].into_iter());
        let split = alleles_seen([("C", "CA"), ("CA", "C")].into_iter());
        let bim = alleles_seen([("CA", "C"), ("C", "CA")].into_iter());
        assert_eq!(&*joined, "C/CA,C/CA");
        assert_eq!(joined, split);
        assert_eq!(joined, bim);
        assert_eq!(&*alleles_seen([("A", "T"), ("A", "G")].into_iter()), "A/G,A/T");
    }

    #[test]
    fn rows_naming_one_variant_for_one_score_count_once_first_by_allele_text() {
        let rows = [("G", "A"), ("A", "."), ("G", "A"), ("T", "A"), ("A", ".")];
        // (score, variant) of each row: score 0 names variant 0 three times and variant 1 once; score
        // 1 names variant 0 once.
        let at = [(0, 0), (0, 0), (1, 0), (0, 1), (0, 0)];
        let mut named: Vec<_> = rows
            .iter()
            .zip(at)
            .enumerate()
            .map(|(row, (&(effect, other), (score, variant)))| (score, variant, effect, other, row))
            .collect();
        let mut extra: Vec<(&str, &str)> = same_variant(&mut named).map(|row| rows[row]).collect();
        extra.sort_unstable();
        assert_eq!(extra, [("A", "."), ("G", "A")]);
    }

    #[test]
    fn a_row_at_a_site_is_unscored_only_when_no_allele_or_several_alleles_carry_it() {
        let rows = [(0, "A", "G"), (1, "A", "T")];
        let site = Site::new(&rows, true);
        assert_eq!(Unmatched::of(site.match_row("G", "A"), "A"), None);
        assert_eq!(Unmatched::of(site.match_row("A", "T"), "T"), None);
        assert_eq!(Unmatched::of(site.match_row("C", "A"), "A"), Some(Unmatched::NoAllelePair));
        assert_eq!(Unmatched::of(site.match_row("A", "."), "."), None);
        let separate = [(0, "A", "G"), (1, "C", "G")];
        let site = Site::new(&separate, true);
        assert_eq!(Unmatched::of(site.match_row("G", "."), "."), Some(Unmatched::SeveralAlleles));
    }
}
