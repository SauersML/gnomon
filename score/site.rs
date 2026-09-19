//! Which allele of a genomic position a score row scores: one rule for every genotype format.
//!
//! A site is every variant row at one position: one ALT of a VCF, BCF or `.pvar` record, or a
//! `.bim` row. Splitting multiallelic records into biallelic ones (bcftools norm -m-, the `.pgen`
//! reader's per-ALT rows) and joining them back keep exactly two things for every person: the
//! copies of each ALT, which only that ALT's own row records, and the ploidy. A row's REF side also
//! counts every other ALT at the site once the site is split. A score row's dose is the copies of
//! its effect allele, as plink2 --score counts them, so it is the ALT's own copies when the effect
//! allele is an ALT, and the ploidy less the copies of every ALT at the site when it is the REF.
//! That is the one dose every representation of the same genotypes agrees on.
//!
//! Variants. A row's two alleles are compared with the trailing characters they share removed while
//! each keeps one, since joining pads alleles with the reference bases a REF lacks: CA>CAA is C>CA.
//! Rows with the same trimmed REF and ALT measure one variant: an array's duplicate probe, or a
//! repeated record. A person's copies of the variant's ALT are the one value its rows with a call
//! agree on; a missing call is no measurement, and rows that disagree leave the copies unknown.
//!
//! REF. A VCF, BCF or `.pvar` record declares its REF, and a site whose rows all declare theirs has
//! that one reading, or none when the REFs disagree. A `.bim` row declares none: PLINK 1 orders its
//! alleles by frequency, and plink2 reads allele 2 only as a provisional REF. So such a site is read
//! from its alleles: a reading picks each variant's REF, as one of its two alleles, so that every
//! row's REF as written is a prefix of the longest, as REFs read from one reference sequence are. A
//! split site always has the reading whose REF every variant carries.
//! - One reading: its REF is the site's.
//! - Several readings: a score row is scored only when every reading gives it the same dose, and
//!   refused otherwise. A lone variant's doses are its own rows' under every reading; variants read
//!   more than one way need an allele that is a prefix of another (an insertion's C and CA). A row
//!   naming a variant that several undeclared rows of such alleles measure is refused too: they
//!   could be an insertion and a deletion.
//! - No reading, as for a probe and its strand complement: the rows are separate measurements, no
//!   REF is the site's, and a score row counts its effect allele on its own variant's rows.
//!
//! Score rows. A row naming its other allele names the variant whose trimmed alleles are its own
//! two, trimmed, on the side its effect allele is on; two variants carrying them (C>CA and CA>C)
//! are two variants the row cannot tell apart, and it is refused. A row naming none (".") or
//! candidates ("A/G") names the allele of the site its effect allele is, compared as written with
//! each variant's trimmed alleles, beside a listed other allele: an ALT of one variant, or the REF,
//! which a site read one way has once however many variants carry it. Naming more than one allele
//! drops the row's weights and counts them.
//!
//! Doses. A (variant, score) is one matched variant of the score, counted once however many of the
//! score's rows name it. It scores the variant's own rows, unless one of its rows names the REF of a
//! site read one way with more than one variant: then every one of its rows waits on the whole site,
//! whose REF copies are the ploidy the records agree on less every variant's ALT copies. A pair
//! whose copies are unknown, or whose site's ALTs pass the ploidy, adds nothing and is missing once.

/// `reference` and `alternate` without the trailing characters they share, while each keeps one.
pub(crate) fn trimmed<'a>(reference: &'a str, alternate: &'a str) -> (&'a str, &'a str) {
    let (mut reference, mut alternate) = (reference, alternate);
    loop {
        let (mut shorter_reference, mut shorter_alternate) = (reference.chars(), alternate.chars());
        match (shorter_reference.next_back(), shorter_alternate.next_back()) {
            (Some(last), Some(other_last))
                if last == other_last
                    && !shorter_reference.as_str().is_empty()
                    && !shorter_alternate.as_str().is_empty() =>
            {
                reference = shorter_reference.as_str();
                alternate = shorter_alternate.as_str();
            }
            _ => return (reference, alternate),
        }
    }
}

/// Which side of the variant with `reference` and `alternate` a score row names: `Some(true)` the
/// REF, `Some(false)` the ALT, and `None` neither. A row naming its other allele names the side of
/// its effect allele when its trimmed alleles are the variant's; a row naming none, or several
/// candidates, names the side its effect allele is, beside a listed other allele.
#[inline]
pub(crate) fn row_side(
    effect_allele: &str,
    other_allele: &str,
    reference: &str,
    alternate: &str,
) -> Option<bool> {
    let (reference, alternate) = trimmed(reference, alternate);
    trimmed_side(effect_allele, other_allele, reference, alternate)
}

/// [`row_side`] for a variant whose alleles are already trimmed.
#[inline]
fn trimmed_side(
    effect_allele: &str,
    other_allele: &str,
    reference: &str,
    alternate: &str,
) -> Option<bool> {
    let (effect, other_is_reference, other_is_alternate) = if names_no_single_other_allele(other_allele) {
        let listed = |allele: &str| other_allele == "." || other_allele.split('/').any(|c| c == allele);
        (effect_allele, listed(reference), listed(alternate))
    } else {
        let (effect, other) = trimmed(effect_allele, other_allele);
        (effect, other == reference, other == alternate)
    };
    if effect == alternate && other_is_reference {
        Some(false)
    } else if effect == reference && other_is_alternate {
        Some(true)
    } else {
        None
    }
}

/// Other-allele text that names no single allele: "." where the score file gave none, or
/// candidates separated by '/', as harmonized PGS Catalog files infer them.
pub(crate) fn names_no_single_other_allele(other_allele: &str) -> bool {
    other_allele == "." || other_allele.contains('/')
}

/// One variant at a site: a distinct trimmed (REF, ALT), in the site's reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Variant<'a> {
    pub(crate) reference: &'a str,
    pub(crate) alternate: &'a str,
    /// The first row measuring it as written, REF first, for messages.
    pub(crate) written: (&'a str, &'a str),
}

/// The allele of a site a score row scores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SiteAllele {
    /// The ALT of variant `k`.
    Alternate(usize),
    /// The REF of variant `k`: the site's REF, when the site has one.
    Reference(usize),
}

impl SiteAllele {
    /// The variant whose alleles the row names.
    pub(crate) fn variant(self) -> usize {
        match self {
            Self::Alternate(variant) | Self::Reference(variant) => variant,
        }
    }
}

/// How a score row's alleles meet a site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RowMatch {
    /// The row scores this allele.
    Scores(SiteAllele),
    /// No variant at the site carries the row's alleles.
    NoVariant,
    /// Two variants carry them, so which one the row means is unknown.
    Several(usize, usize),
    /// The site's alleles can be read in more than one way, and the row names a different allele,
    /// or the REF of differently read variants, in different readings of variant `k`.
    Unread(usize),
}

/// What the site's rows say about its REF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SiteReference {
    /// One reading holds every REF on one reference sequence: its REF is the site's.
    Known,
    /// Several readings do, so the REF of some variant depends on which.
    Unread,
    /// No reading does: the rows are separate measurements, not one site.
    Separate,
}

/// The variants at one position, the rows that measure each and how the site is read.
#[derive(Debug)]
pub(crate) struct Site<'a> {
    variants: Vec<Variant<'a>>,
    /// The variant each row measures, in row order.
    row_variants: Vec<usize>,
    /// Per row, whether its first written allele is its REF in the site's reading.
    first_is_reference: Vec<bool>,
    reference: SiteReference,
    /// Per variant, whether every reading gives it the same REF.
    settled: Vec<bool>,
    /// Per variant, whether its rows could be two different variants: several undeclared rows of an
    /// insertion's alleles, which read as an insertion and a deletion as well.
    indistinct: Vec<bool>,
    /// A variant one record lists as two of its ALTs.
    listed_twice: Option<usize>,
}

impl<'a> Site<'a> {
    /// The site of `rows`, each a record's index and its two alleles as written. When `declared`,
    /// the first is the REF, as a VCF, BCF or `.pvar` record writes it; otherwise either may be, as
    /// on a `.bim` row, and the first is read as the REF where the alleles leave the choice free.
    pub(crate) fn new(rows: &[(usize, &'a str, &'a str)], declared: bool) -> Self {
        // Variants: rows with the same trimmed alleles, in order when the REF is written.
        let mut variant_rows: Vec<Vec<usize>> = Vec::new();
        let mut pairs: Vec<(&'a str, &'a str)> = Vec::new();
        let mut row_variants = Vec::with_capacity(rows.len());
        let mut listed_twice = None;
        for (row, &(record, first, second)) in rows.iter().enumerate() {
            let pair = trimmed(first, second);
            let found = pairs.iter().position(|&(x, y)| {
                (x, y) == pair || (!declared && (y, x) == pair)
            });
            let variant = match found {
                Some(variant) => {
                    if declared && variant_rows[variant].iter().any(|&earlier| rows[earlier].0 == record) {
                        listed_twice.get_or_insert(variant);
                    }
                    variant_rows[variant].push(row);
                    variant
                }
                None => {
                    pairs.push(pair);
                    variant_rows.push(vec![row]);
                    pairs.len() - 1
                }
            };
            row_variants.push(variant);
        }
        let indistinct = pairs
            .iter()
            .zip(&variant_rows)
            .map(|(&(x, y), rows)| !declared && rows.len() > 1 && x != y && (x.starts_with(y) || y.starts_with(x)))
            .collect();

        // Readings: per variant, whether the REF is its pair's first allele, such that every row's
        // REF as written is a prefix of the longest.
        let readings = consistent_readings(rows, &pairs, &variant_rows, declared);
        let (reference, chosen) = match readings.as_slice() {
            [] => (SiteReference::Separate, vec![true; pairs.len()]),
            [reading] => (SiteReference::Known, reading.clone()),
            [reading, ..] => (SiteReference::Unread, reading.clone()),
        };
        let settled = (0..pairs.len())
            .map(|variant| readings.iter().all(|reading| reading[variant] == chosen[variant]))
            .collect();
        let variants = pairs
            .iter()
            .zip(&variant_rows)
            .zip(&chosen)
            .map(|((&(x, y), rows_of), &first_is_reference)| {
                let (_, written_first, written_second) = rows[rows_of[0]];
                // The written row's REF, in the orientation of its variant's pair.
                let row_first_is_reference = (x, y) == trimmed(written_first, written_second);
                let written = if row_first_is_reference == first_is_reference {
                    (written_first, written_second)
                } else {
                    (written_second, written_first)
                };
                let (reference, alternate) = if first_is_reference { (x, y) } else { (y, x) };
                Variant {
                    reference,
                    alternate,
                    written,
                }
            })
            .collect();
        let first_is_reference = rows
            .iter()
            .zip(&row_variants)
            .map(|(&(_, first, second), &variant)| {
                let pair_first = pairs[variant].0 == trimmed(first, second).0;
                pair_first == chosen[variant]
            })
            .collect();
        Site {
            variants,
            row_variants,
            first_is_reference,
            reference,
            settled,
            indistinct,
            listed_twice,
        }
    }

    pub(crate) fn variants(&self) -> &[Variant<'a>] {
        &self.variants
    }

    /// The variant each row measures, in row order.
    pub(crate) fn row_variants(&self) -> &[usize] {
        &self.row_variants
    }

    /// Per row, whether its first written allele is its REF.
    pub(crate) fn first_is_reference(&self) -> &[bool] {
        &self.first_is_reference
    }

    /// Whether a REF named through `allele` is the site's, so that its dose is the ploidy less every
    /// ALT's copies: the site is read one way and has more than one variant. Otherwise the variant's
    /// own rows give it, as they give an ALT.
    pub(crate) fn reads_whole_site(&self, allele: SiteAllele) -> bool {
        matches!(allele, SiteAllele::Reference(_))
            && self.reference == SiteReference::Known
            && self.variants.len() > 1
    }

    /// A variant one record lists as two of its ALTs, which makes its copies unknowable.
    pub(crate) fn listed_twice(&self) -> Option<usize> {
        self.listed_twice
    }

    /// How a score row with `effect_allele` and `other_allele` meets the site. A row naming its
    /// other allele scores the variant whose trimmed alleles are the row's two, on the side its
    /// effect allele is on. A row naming none, or several candidates, scores the one allele of the
    /// site that is its effect allele beside a listed other allele: an ALT of one variant, or the
    /// site's REF, which every variant carries.
    pub(crate) fn match_row(&self, effect_allele: &str, other_allele: &str) -> RowMatch {
        let effect_only = names_no_single_other_allele(other_allele);
        let one_reference = self.reference == SiteReference::Known;
        let mut found: Option<SiteAllele> = None;
        for (index, variant) in self.variants.iter().enumerate() {
            let allele = match trimmed_side(effect_allele, other_allele, variant.reference, variant.alternate) {
                Some(false) => SiteAllele::Alternate(index),
                Some(true) => SiteAllele::Reference(index),
                None => continue,
            };
            match found {
                None => found = Some(allele),
                // An effect-only row names the one REF of a site read one way through every
                // variant. Two variants carrying a named pair are two variants however they name it.
                Some(SiteAllele::Reference(_))
                    if effect_only && one_reference && matches!(allele, SiteAllele::Reference(_)) => {}
                Some(earlier) => return RowMatch::Several(earlier.variant(), allele.variant()),
            }
        }
        let Some(allele) = found else {
            return RowMatch::NoVariant;
        };
        let variant = allele.variant();
        // A variant whose rows may be two variants, or whose REF depends on the reading, gives the
        // row a dose that depends on which. So does the site's REF of a site read several ways. A
        // single variant's dose of either allele is its own rows' whatever its reading.
        let unread = self.indistinct[variant]
            || (self.reference == SiteReference::Unread
                && self.variants.len() > 1
                && (!self.settled[variant] || matches!(allele, SiteAllele::Reference(_))));
        if unread {
            return RowMatch::Unread(variant);
        }
        RowMatch::Scores(allele)
    }

    /// The start of the error for a score row that two variants of the site carry, on `rows` of the
    /// genotypes: which it scores is unknown, and they are different variants. The caller ends it
    /// with how to keep the one the row means.
    pub(crate) fn several_error(
        &self,
        chromosome: &str,
        position: u32,
        effect_allele: &str,
        other_allele: &str,
        (first, second): (usize, usize),
        rows: &str,
    ) -> String {
        let (first, second) = (&self.variants[first], &self.variants[second]);
        format!(
            "More than one {rows} at {chromosome}:{position} carries the alleles {other_allele} and {effect_allele}, which a score row names, so which {rows} the row scores is unknown. One has REF {} and ALT {}, another REF {} and ALT {}, so they are different variants: keep the one the row means",
            first.reference, first.alternate, second.reference, second.alternate
        )
    }

    /// The error for a score row whose dose depends on how the site's rows are read.
    pub(crate) fn unread_error(
        &self,
        chromosome: &str,
        position: u32,
        effect_allele: &str,
        other_allele: &str,
        rows: &str,
    ) -> String {
        format!(
            "The {rows} at {chromosome}:{position} do not write which allele is the REF, and their alleles can be read as more than one set of variants, under which a score row naming {effect_allele} and {other_allele} scores different alleles, so which it scores is unknown."
        )
    }
}

/// Every reading of a site: per variant, whether the REF is the first allele of its trimmed pair,
/// such that every row's REF as written is a prefix of the longest. A declared row's REF is its
/// first written allele, so a site of declared rows has one reading or none. Undeclared variants
/// are tried with their first allele as the REF first.
fn consistent_readings(
    rows: &[(usize, &str, &str)],
    pairs: &[(&str, &str)],
    variant_rows: &[Vec<usize>],
    declared: bool,
) -> Vec<Vec<bool>> {
    let mut readings = Readings {
        rows,
        pairs,
        variant_rows,
        declared,
        choice: Vec::with_capacity(pairs.len()),
        found: Vec::new(),
    };
    readings.search(0, "");
    readings.found
}

/// The search [`consistent_readings`] runs, variant by variant.
struct Readings<'s, 'a> {
    rows: &'s [(usize, &'a str, &'a str)],
    pairs: &'s [(&'a str, &'a str)],
    variant_rows: &'s [Vec<usize>],
    declared: bool,
    choice: Vec<bool>,
    found: Vec<Vec<bool>>,
}

impl<'a> Readings<'_, 'a> {
    /// Reads every variant from `variant` on, the REFs chosen so far all prefixes of `longest`.
    fn search(&mut self, variant: usize, longest: &'a str) {
        if variant == self.pairs.len() {
            self.found.push(self.choice.clone());
            return;
        }
        let options: &[bool] = if self.declared { &[true] } else { &[true, false] };
        for &pair_first in options {
            let mut chain = longest;
            let consistent = self.variant_rows[variant].iter().all(|&row| {
                let (_, first, second) = self.rows[row];
                let written_first = trimmed(first, second).0 == self.pairs[variant].0;
                let reference = if written_first == pair_first { first } else { second };
                if reference.len() > chain.len() {
                    let prefix = reference.starts_with(chain);
                    chain = reference;
                    prefix
                } else {
                    chain.starts_with(reference)
                }
            });
            if consistent {
                self.choice.push(pair_first);
                self.search(variant + 1, chain);
                self.choice.pop();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The site of records, each written (REF, ALT) and numbered by record.
    fn records<'a>(rows: &[(usize, &'a str, &'a str)]) -> Site<'a> {
        Site::new(rows, true)
    }

    /// The site of `.bim` rows, each written (allele 2, allele 1).
    fn bim<'a>(rows: &[(&'a str, &'a str)]) -> Site<'a> {
        let rows: Vec<(usize, &str, &str)> = rows
            .iter()
            .enumerate()
            .map(|(row, &(allele2, allele1))| (row, allele2, allele1))
            .collect();
        Site::new(&rows, false)
    }

    fn alleles(site: &Site<'_>) -> Vec<(String, String)> {
        site.variants()
            .iter()
            .map(|variant| (variant.reference.to_string(), variant.alternate.to_string()))
            .collect()
    }

    #[test]
    fn trimming_removes_shared_trailing_bases_but_keeps_one_of_each() {
        assert_eq!(trimmed("CA", "CAA"), ("C", "CA"));
        assert_eq!(trimmed("CAA", "CA"), ("CA", "C"));
        assert_eq!(trimmed("AT", "GT"), ("A", "G"));
        assert_eq!(trimmed("A", "G"), ("A", "G"));
        assert_eq!(trimmed("AT", "AT"), ("A", "A"));
        assert_eq!(trimmed("CA", "C"), ("CA", "C"));
        assert_eq!(trimmed("é", "aé"), ("é", "aé"));
    }

    #[test]
    fn a_joined_record_and_its_split_records_give_one_site() {
        // bcftools norm -m+ of C>CA and CA>C pads the insertion to CA>CAA.
        let joined = records(&[(0, "CA", "CAA"), (0, "CA", "C")]);
        let split = records(&[(0, "C", "CA"), (1, "CA", "C")]);
        assert_eq!(alleles(&joined), alleles(&split));
        for site in [&joined, &split] {
            assert_eq!(site.match_row("CA", "C"), RowMatch::Several(0, 1));
            assert_eq!(site.match_row("CAA", "CA"), RowMatch::Several(0, 1));
        }
    }

    #[test]
    fn a_ref_effect_row_names_the_site_ref_through_its_pair() {
        for site in [records(&[(0, "A", "G"), (1, "A", "T")]), bim(&[("A", "G"), ("A", "T")])] {
            assert_eq!(site.match_row("A", "G"), RowMatch::Scores(SiteAllele::Reference(0)));
            assert!(site.reads_whole_site(SiteAllele::Reference(0)));
            assert_eq!(site.match_row("G", "A"), RowMatch::Scores(SiteAllele::Alternate(0)));
            assert_eq!(site.match_row("T", "A"), RowMatch::Scores(SiteAllele::Alternate(1)));
            assert_eq!(site.match_row("G", "T"), RowMatch::NoVariant);
            // An effect-only REF row names the one REF however many variants carry it.
            assert_eq!(site.match_row("A", "."), RowMatch::Scores(SiteAllele::Reference(0)));
            assert_eq!(site.match_row("A", "G/C"), RowMatch::Scores(SiteAllele::Reference(0)));
            assert_eq!(site.match_row("G", "."), RowMatch::Scores(SiteAllele::Alternate(0)));
            assert_eq!(site.match_row("C", "."), RowMatch::NoVariant);
        }
    }

    #[test]
    fn a_bim_site_is_read_whatever_order_its_rows_write_their_alleles_in() {
        // PLINK 1 writes the minor allele first: here the shared REF A is allele 1 of both rows,
        // allele 2 of neither, and still the one allele both variants carry.
        let site = bim(&[("C", "A"), ("G", "A")]);
        assert_eq!(alleles(&site), [("A".to_string(), "C".to_string()), ("A".to_string(), "G".to_string())]);
        assert_eq!(site.first_is_reference(), [false, false]);
        assert_eq!(site.match_row("C", "A"), RowMatch::Scores(SiteAllele::Alternate(0)));
        assert_eq!(site.match_row("A", "G"), RowMatch::Scores(SiteAllele::Reference(1)));
        assert!(site.reads_whole_site(SiteAllele::Reference(1)));
    }

    #[test]
    fn an_effect_allele_on_two_sides_of_a_site_is_two_alleles() {
        // A deletion CA>C and a SNP C>T at one position: C is the deletion's ALT and the SNP's REF.
        let site = records(&[(0, "CA", "C"), (1, "C", "T")]);
        assert_eq!(site.match_row("C", "."), RowMatch::Several(0, 1));
        assert_eq!(site.match_row("C", "T"), RowMatch::Scores(SiteAllele::Reference(1)));
        assert_eq!(site.match_row("C", "CA"), RowMatch::Scores(SiteAllele::Alternate(0)));
        // A `.bim` writes no REF: the same rows read as an insertion C>CA and the SNP as well, under
        // which C is the REF of both, so a row naming the REF there is refused.
        let site = bim(&[("CA", "C"), ("C", "T")]);
        assert_eq!(site.match_row("C", "T"), RowMatch::Unread(1));
        assert_eq!(site.match_row("T", "C"), RowMatch::Scores(SiteAllele::Alternate(1)));
        assert_eq!(site.match_row("CA", "C"), RowMatch::Unread(0));
    }

    #[test]
    fn repeated_rows_measure_one_variant() {
        let site = records(&[(0, "A", "G"), (1, "A", "G"), (2, "AT", "GT")]);
        assert_eq!(site.variants().len(), 1);
        assert_eq!(site.row_variants(), [0, 0, 0]);
        assert_eq!(site.listed_twice(), None);
        assert_eq!(site.match_row("G", "A"), RowMatch::Scores(SiteAllele::Alternate(0)));
        assert!(!site.reads_whole_site(SiteAllele::Reference(0)));
        let listed_twice = records(&[(0, "C", "CA"), (0, "C", "CA")]);
        assert_eq!(listed_twice.listed_twice(), Some(0));
        // Two `.bim` rows of a SNP measure one variant whichever allele is the REF; two of an
        // insertion's alleles may be an insertion and a deletion, which no `.bim` tells apart.
        let probes = bim(&[("A", "G"), ("G", "A")]);
        assert_eq!(probes.variants().len(), 1);
        assert_eq!(probes.match_row("G", "A"), RowMatch::Scores(SiteAllele::Alternate(0)));
        assert_eq!(probes.match_row("A", "G"), RowMatch::Scores(SiteAllele::Reference(0)));
        let indels = bim(&[("C", "CA"), ("C", "CA")]);
        assert_eq!(indels.match_row("CA", "C"), RowMatch::Unread(0));
        assert_eq!(bim(&[("C", "CA")]).match_row("CA", "C"), RowMatch::Scores(SiteAllele::Alternate(0)));
    }

    #[test]
    fn rows_no_one_reference_holds_are_separate_measurements() {
        // A probe and its strand complement: no allele is common to both, so neither REF is the
        // site's, and each row's REF is its own.
        let site = bim(&[("A", "G"), ("C", "T")]);
        assert_eq!(site.match_row("A", "G"), RowMatch::Scores(SiteAllele::Reference(0)));
        assert!(!site.reads_whole_site(SiteAllele::Reference(0)));
        // Records whose written REFs no one reference holds are too.
        let site = records(&[(0, "A", "G"), (1, "C", "T")]);
        assert!(!site.reads_whole_site(SiteAllele::Reference(0)));
        let site = records(&[(0, "CAG", "C"), (1, "C", "T"), (2, "CA", "C")]);
        assert!(site.reads_whole_site(SiteAllele::Reference(1)));
    }
}
