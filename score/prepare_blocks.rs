// ========================================================================================
//
//               Per-block partial scores: the plan-time expansion behind --blocks
//
// ========================================================================================
//
// A block partition names genomic blocks. With one, every score is expanded at plan
// time into `score x block` pseudo-scores: each matched variant's weight is routed
// to the unsplit column and to the column of the block holding the variant. The
// multi-score engine then computes every partial with exactly the arithmetic it
// uses for the unsplit score, so the partials of a score are a partition of its
// total. Nothing downstream of the plan knows blocks exist.

use super::cache::VariantPlan;
use super::{PrepError, VariantKey, accumulate_baseline};
use crate::score::types::{GroupedComplexRule, ScoreInfo, parse_chromosome_label};
use std::io::{self, Write};
use std::path::Path;

/// Chromosome codes with a chromosome block of their own under `--blocks chrom`.
const CHROMOSOME_CODES: std::ops::RangeInclusive<u8> = 1..=26;

/// Block id of a variant outside every block of the partition.
pub const REMAINDER_BLOCK: u32 = 0;

/// More blocks than this need `--blocks-max` to proceed.
pub const BLOCKS_WITHOUT_LIMIT: usize = 500;

/// One block: 0-based, half-open BED coordinates on one chromosome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub chromosome: u8,
    pub start: u64,
    /// `None` for a whole chromosome.
    pub end: Option<u64>,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Chromosomes,
    Bed,
}

/// The blocks named by `--blocks`, with the column layout they induce.
///
/// The layout depends on the partition alone, never on the genotypes or the
/// scores, so the same `--blocks` argument gives the same columns on every
/// cohort. Block `b0000` holds what lies outside every block, so the block
/// partials of a score always sum to its total.
#[derive(Debug, Clone)]
pub struct BlockPartition {
    kind: Kind,
    /// Block id `i + 1` is `blocks[i]`.
    blocks: Vec<Block>,
    /// Per chromosome code, its blocks as `(start, end, id)` in start order.
    by_chromosome: Vec<Vec<(u64, u64, u32)>>,
}

impl BlockPartition {
    /// `chrom`: one block per chromosome, ids 1-22, X = 23, Y = 24, XY = 25, MT = 26.
    pub fn chromosomes() -> Self {
        let blocks = CHROMOSOME_CODES
            .map(|code| Block {
                chromosome: code,
                start: 0,
                end: None,
                name: format!("chr{}", chromosome_label(code)),
            })
            .collect();
        Self {
            kind: Kind::Chromosomes,
            blocks,
            by_chromosome: Vec::new(),
        }
    }

    /// The `--blocks` argument: `chrom`, or the path of a BED file.
    pub fn parse_arg(arg: &str) -> Result<Self, String> {
        if arg.eq_ignore_ascii_case("chrom") {
            return Ok(Self::chromosomes());
        }
        let path = Path::new(arg);
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read the BED file '{}': {e}", path.display()))?;
        Self::from_bed_text(&text).map_err(|e| format!("BED file '{}': {e}", path.display()))
    }

    /// One block per BED row, ids in file order. Rows are `chrom start end [name]`,
    /// tab- or space-separated; `#`, `track` and `browser` lines are skipped.
    /// Blocks on one chromosome must not overlap.
    pub fn from_bed_text(text: &str) -> Result<Self, String> {
        let mut blocks = Vec::new();
        for (index, raw) in text.lines().enumerate() {
            let line_number = index + 1;
            let line = raw.trim_end_matches(['\r', '\n']);
            if line.trim().is_empty()
                || line.starts_with('#')
                || line.starts_with("track")
                || line.starts_with("browser")
            {
                continue;
            }
            let fields: Vec<&str> = if line.contains('\t') {
                line.split('\t').map(str::trim).collect()
            } else {
                line.split_whitespace().collect()
            };
            if fields.len() < 3 {
                return Err(format!(
                    "line {line_number} has {} columns; a block needs chrom, start and end",
                    fields.len()
                ));
            }
            let chromosome = parse_chromosome_label(fields[0])
                .map_err(|e| format!("line {line_number}: {e}"))?;
            let start = fields[1]
                .parse::<u64>()
                .map_err(|e| format!("line {line_number}: start '{}': {e}", fields[1]))?;
            let end = fields[2]
                .parse::<u64>()
                .map_err(|e| format!("line {line_number}: end '{}': {e}", fields[2]))?;
            if start >= end {
                return Err(format!(
                    "line {line_number}: start {start} is not below end {end} (half-open [start, end))"
                ));
            }
            let name = match fields.get(3).map(|s| s.trim()) {
                Some(name) if !name.is_empty() => name.to_string(),
                _ => ".".to_string(),
            };
            blocks.push(Block {
                chromosome,
                start,
                end: Some(end),
                name,
            });
        }
        if blocks.is_empty() {
            return Err("names no blocks".to_string());
        }
        let mut by_chromosome = vec![Vec::new(); usize::from(u8::MAX) + 1];
        for (index, block) in blocks.iter().enumerate() {
            let id = u32::try_from(index + 1)
                .map_err(|_| format!("more than {} blocks", u32::MAX - 1))?;
            by_chromosome[usize::from(block.chromosome)].push((
                block.start,
                block.end.unwrap_or(u64::MAX),
                id,
            ));
        }
        for intervals in &mut by_chromosome {
            intervals.sort_unstable();
            for pair in intervals.windows(2) {
                let (earlier, later) = (pair[0], pair[1]);
                if later.0 < earlier.1 {
                    return Err(format!(
                        "blocks {} and {} overlap on chromosome {}: [{}, {}) and [{}, {})",
                        block_id(earlier.2),
                        block_id(later.2),
                        chromosome_label(blocks[earlier.2 as usize - 1].chromosome),
                        earlier.0,
                        earlier.1,
                        later.0,
                        later.1
                    ));
                }
            }
        }
        Ok(Self {
            kind: Kind::Bed,
            blocks,
            by_chromosome,
        })
    }

    /// The block holding a variant at a 1-based position, or [`REMAINDER_BLOCK`].
    /// A BED block `[start, end)` holds the 0-based coordinate `pos - 1`, so a
    /// variant on a boundary belongs to exactly one block.
    #[inline]
    pub fn assign(&self, key: VariantKey) -> u32 {
        let (chromosome, pos) = key;
        match self.kind {
            Kind::Chromosomes => {
                if CHROMOSOME_CODES.contains(&chromosome) {
                    u32::from(chromosome)
                } else {
                    REMAINDER_BLOCK
                }
            }
            Kind::Bed => {
                let pos = u64::from(pos);
                let intervals = &self.by_chromosome[usize::from(chromosome)];
                let after = intervals.partition_point(|&(start, _, _)| start < pos);
                match after.checked_sub(1).map(|i| intervals[i]) {
                    Some((_, end, id)) if pos <= end => id,
                    _ => REMAINDER_BLOCK,
                }
            }
        }
    }

    /// The blocks with an interval, in id order (block ids 1..).
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// Columns per score: the unsplit total, the remainder block, then every block.
    pub fn columns_per_score(&self) -> usize {
        2 + self.blocks.len()
    }

    /// The expanded column names: each score's own name, then `NAME_b0000`,
    /// `NAME_b0001`, ... in block order.
    pub fn column_names(&self, score_names: &[String]) -> Vec<String> {
        let mut names = Vec::with_capacity(score_names.len() * self.columns_per_score());
        for name in score_names {
            names.push(name.clone());
            for id in 0..=self.blocks.len() as u32 {
                names.push(format!("{name}_{}", block_id(id)));
            }
        }
        names
    }

    /// Whether an expanded column is a block partial rather than an unsplit score.
    pub fn is_block_column(&self, column: usize) -> bool {
        column % self.columns_per_score() != 0
    }

    /// The `_MISSING_PCT` convention for every expanded column, by column index.
    pub fn block_column_flags(&self, columns: usize) -> Vec<bool> {
        (0..columns).map(|c| self.is_block_column(c)).collect()
    }

    /// One line for the log.
    pub fn describe(&self) -> String {
        match self.kind {
            Kind::Chromosomes => format!("{} chromosome blocks", self.blocks.len()),
            Kind::Bed => format!("{} BED blocks", self.blocks.len()),
        }
    }

    /// Refuses a partition too wide for its accumulators unless `--blocks-max`
    /// admits it. Above [`BLOCKS_WITHOUT_LIMIT`] blocks the projection is shown and
    /// the limit is required; a limit below the block count refuses outright.
    pub fn check_block_budget(
        &self,
        blocks_max: Option<usize>,
        num_scores: usize,
        num_people: usize,
    ) -> Result<(), PrepError> {
        let blocks = self.blocks.len();
        let columns = self.columns_per_score();
        let cells = (num_people as u128) * (num_scores as u128) * (columns as u128);
        // One f64 sum and one u32 missing count per cell, per accumulator copy.
        let projection = format!(
            "{num_people} people x {num_scores} score(s) x {columns} columns = {cells} accumulator cells, at least {} per copy",
            format_bytes(cells * 12)
        );
        match blocks_max {
            Some(max) if blocks > max => Err(PrepError::Blocks(format!(
                "{blocks} blocks exceed --blocks-max {max} ({projection})"
            ))),
            Some(_) => {
                if blocks > BLOCKS_WITHOUT_LIMIT {
                    eprintln!("> Warning: --blocks names {blocks} blocks: {projection}.");
                }
                Ok(())
            }
            None if blocks > BLOCKS_WITHOUT_LIMIT => Err(PrepError::Blocks(format!(
                "{blocks} blocks is more than {BLOCKS_WITHOUT_LIMIT}: {projection}. Pass --blocks-max {blocks} to proceed"
            ))),
            None => Ok(()),
        }
    }

    /// Feeds everything that decides the expanded plan into a plan-cache key.
    pub(super) fn hash_into(&self, hasher: &mut blake3::Hasher) {
        hasher.update(&[match self.kind {
            Kind::Chromosomes => 1u8,
            Kind::Bed => 2u8,
        }]);
        hasher.update(&(self.blocks.len() as u64).to_le_bytes());
        for block in &self.blocks {
            hasher.update(&[block.chromosome]);
            hasher.update(&block.start.to_le_bytes());
            hasher.update(&block.end.unwrap_or(u64::MAX).to_le_bytes());
            hasher.update(&(block.name.len() as u64).to_le_bytes());
            hasher.update(block.name.as_bytes());
        }
    }

    /// The sidecar mapping block ids to intervals: `#BLOCK_ID CHROM START END NAME`,
    /// with BED (0-based, half-open) coordinates and `.` where a bound is absent.
    pub fn write_sidecar<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writeln!(
            writer,
            "# Blocks of the _b<ID> score columns; START and END are 0-based, half-open BED coordinates."
        )?;
        writeln!(writer, "#BLOCK_ID\tCHROM\tSTART\tEND\tNAME")?;
        writeln!(
            writer,
            "{}\t.\t.\t.\toutside_every_block",
            block_id(REMAINDER_BLOCK)
        )?;
        for (index, block) in self.blocks.iter().enumerate() {
            let end = block.end.map_or(".".to_string(), |end| end.to_string());
            writeln!(
                writer,
                "{}\t{}\t{}\t{end}\t{}",
                block_id(index as u32 + 1),
                chromosome_label(block.chromosome),
                block.start,
                block.name
            )?;
        }
        Ok(())
    }
}

/// `b0000`, `b0001`, ...: the suffix a block adds to a score name.
pub fn block_id(id: u32) -> String {
    format!("b{id:04}")
}

fn chromosome_label(code: u8) -> String {
    match code {
        23 => "X".to_string(),
        24 => "Y".to_string(),
        25 => "XY".to_string(),
        26 => "MT".to_string(),
        n => n.to_string(),
    }
}

fn format_bytes(bytes: u128) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

/// Expands a compiled plan over `partition`. `row_keys[r]` is the locus of CSR
/// row `r`. Every entry `(score, weight, correction)` becomes that entry on the
/// score's unsplit column plus a copy on the column of the row's block; complex
/// rules gain a block application beside each of theirs. Unsplit columns keep
/// the join's own counts and baselines, so they are exactly the plan without
/// blocks; block columns take theirs from the rows they receive.
pub(super) fn expand_plan(
    plan: VariantPlan,
    row_keys: &[VariantKey],
    partition: &BlockPartition,
) -> Result<VariantPlan, PrepError> {
    let rows = plan.offsets.len().saturating_sub(1);
    if row_keys.len() != rows {
        return Err(PrepError::Invariant(format!(
            "Block expansion has {} row keys for {rows} plan rows.",
            row_keys.len()
        )));
    }
    let base_scores = plan.names.len();
    if plan.counts.len() != base_scores || plan.baseline.len() != base_scores {
        return Err(PrepError::Invariant(format!(
            "Block expansion: {base_scores} score names, {} counts, {} baselines.",
            plan.counts.len(),
            plan.baseline.len()
        )));
    }
    let per_score = partition.columns_per_score();
    let columns = base_scores
        .checked_mul(per_score)
        .filter(|&columns| u32::try_from(columns).is_ok())
        .ok_or_else(|| {
            PrepError::Blocks(format!(
                "{base_scores} scores x {per_score} columns per score exceed the column space"
            ))
        })?;
    let allocation = |what: &'static str| {
        move |e: std::collections::TryReserveError| {
            PrepError::Invariant(format!("Cannot allocate block-expanded {what}: {e}"))
        }
    };
    let entries = plan.columns.len().saturating_mul(2);
    let mut weights = Vec::new();
    weights
        .try_reserve_exact(entries)
        .map_err(allocation("weights"))?;
    let mut corrections = Vec::new();
    corrections
        .try_reserve_exact(entries)
        .map_err(allocation("corrections"))?;
    let mut expanded_columns = Vec::new();
    expanded_columns
        .try_reserve_exact(entries)
        .map_err(allocation("columns"))?;
    let mut offsets = Vec::new();
    offsets
        .try_reserve_exact(rows + 1)
        .map_err(allocation("offsets"))?;
    let mut baseline = vec![0.0f64; columns];
    let mut errors = vec![0.0f64; columns];
    let mut counts = vec![0u32; columns];
    for score in 0..base_scores {
        baseline[score * per_score] = plan.baseline[score];
        counts[score * per_score] = plan.counts[score];
    }
    offsets.push(0);
    for row in 0..rows {
        let block = partition.assign(row_keys[row]) as usize;
        let range = plan.offsets[row] as usize..plan.offsets[row + 1] as usize;
        for entry in range {
            let score = plan.columns[entry] as usize;
            if score >= base_scores {
                return Err(PrepError::Invariant(format!(
                    "Block expansion met score column {score} of {base_scores}."
                )));
            }
            let total = score * per_score;
            let partial = total + 1 + block;
            let (weight, correction) = (plan.weights[entry], plan.corrections[entry]);
            expanded_columns.push(total as u32);
            weights.push(weight);
            corrections.push(correction);
            expanded_columns.push(partial as u32);
            weights.push(weight);
            corrections.push(correction);
            accumulate_baseline(&mut baseline[partial], &mut errors[partial], correction);
            counts[partial] += 1;
        }
        offsets.push(expanded_columns.len() as u64);
    }
    for column in (0..columns).filter(|&column| partition.is_block_column(column)) {
        baseline[column] += errors[column];
    }
    let mut complex = Vec::new();
    complex
        .try_reserve_exact(plan.complex.len())
        .map_err(allocation("complex rules"))?;
    for rule in plan.complex {
        let chromosome = parse_chromosome_label(&rule.locus_chr_pos.0).map_err(|e| {
            PrepError::Invariant(format!(
                "Block expansion cannot key complex locus {}:{}: {e}",
                rule.locus_chr_pos.0, rule.locus_chr_pos.1
            ))
        })?;
        let block = partition.assign((chromosome, rule.locus_chr_pos.1)) as usize;
        let mut applications = Vec::with_capacity(rule.score_applications.len() * 2);
        for application in rule.score_applications {
            let score = application.score_column_index.0;
            if score >= base_scores {
                return Err(PrepError::Invariant(format!(
                    "Block expansion met complex score column {score} of {base_scores}."
                )));
            }
            let total = score * per_score;
            let partial = total + 1 + block;
            counts[partial] += 1;
            applications.push(ScoreInfo {
                score_column_index: crate::score::types::ScoreColumnIndex(total),
                ..application.clone()
            });
            applications.push(ScoreInfo {
                score_column_index: crate::score::types::ScoreColumnIndex(partial),
                ..application
            });
        }
        complex.push(GroupedComplexRule {
            locus_chr_pos: rule.locus_chr_pos,
            possible_contexts: rule.possible_contexts,
            score_applications: applications,
        });
    }
    Ok(VariantPlan {
        weights,
        corrections,
        columns: expanded_columns,
        offsets,
        baseline,
        required: plan.required,
        complex,
        names: partition.column_names(&plan.names),
        counts,
        flags: plan.flags,
        starts: plan.starts,
        total_variants: plan.total_variants,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::types::{BimRowIndex, ScoreColumnIndex};

    fn bed(text: &str) -> BlockPartition {
        BlockPartition::from_bed_text(text).expect("valid BED")
    }

    #[test]
    fn chromosome_blocks_are_the_chromosome_codes() {
        let partition = BlockPartition::chromosomes();
        assert_eq!(partition.blocks().len(), 26);
        assert_eq!(partition.columns_per_score(), 28);
        assert_eq!(partition.assign((1, 100)), 1);
        assert_eq!(partition.assign((22, 1)), 22);
        assert_eq!(partition.assign((23, 5)), 23);
        assert_eq!(partition.assign((26, 5)), 26);
        assert_eq!(partition.assign((0, 5)), REMAINDER_BLOCK);
        assert_eq!(partition.assign((27, 5)), REMAINDER_BLOCK);
        assert_eq!(partition.blocks()[22].name, "chrX");
        assert_eq!(partition.blocks()[25].name, "chrMT");
    }

    #[test]
    fn bed_blocks_are_half_open_and_one_variant_lands_in_one_block() {
        let partition = bed("chr1\t0\t100\tfirst\n1\t100\t200\tsecond\nchrX\t50\t60\n");
        assert_eq!(partition.blocks().len(), 3);
        assert_eq!(partition.blocks()[2].name, ".");
        // 1-based position 100 is 0-based 99: the first block. 101 is 100: the second.
        assert_eq!(partition.assign((1, 100)), 1);
        assert_eq!(partition.assign((1, 101)), 2);
        assert_eq!(partition.assign((1, 1)), 1);
        assert_eq!(partition.assign((1, 200)), 2);
        assert_eq!(partition.assign((1, 201)), REMAINDER_BLOCK);
        assert_eq!(partition.assign((1, 0)), REMAINDER_BLOCK);
        assert_eq!(partition.assign((2, 50)), REMAINDER_BLOCK);
        assert_eq!(partition.assign((23, 51)), 3);
        assert_eq!(partition.assign((23, 50)), REMAINDER_BLOCK);
        assert_eq!(partition.assign((23, 60)), 3);
        assert_eq!(partition.assign((23, 61)), REMAINDER_BLOCK);
    }

    #[test]
    fn bed_rows_keep_file_order_and_skip_comments() {
        let partition = bed(
            "browser position chr1\ntrack name=x\n# comment\n\n1 500 600 late\n1 0 100 early\r\n",
        );
        assert_eq!(partition.blocks()[0].name, "late");
        assert_eq!(partition.blocks()[1].name, "early");
        assert_eq!(partition.assign((1, 50)), 2);
        assert_eq!(partition.assign((1, 550)), 1);
    }

    #[test]
    fn bed_rejects_overlaps_empty_files_and_bad_rows() {
        let overlap = BlockPartition::from_bed_text("1\t0\t100\n1\t99\t200\n").unwrap_err();
        assert!(
            overlap.contains("b0001") && overlap.contains("b0002"),
            "{overlap}"
        );
        assert!(BlockPartition::from_bed_text("1\t0\t100\n1\t100\t200\n").is_ok());
        assert!(BlockPartition::from_bed_text("# only a comment\n").is_err());
        assert!(BlockPartition::from_bed_text("1\t10\t10\n").is_err());
        assert!(BlockPartition::from_bed_text("1\t10\n").is_err());
        assert!(BlockPartition::from_bed_text("1\tten\t20\n").is_err());
        assert!(BlockPartition::from_bed_text("chrUn_gl000220\t10\t20\n").is_err());
    }

    #[test]
    fn column_names_and_flags_follow_the_layout() {
        let partition = bed("1\t0\t100\n1\t100\t200\n");
        let names = partition.column_names(&["A".to_string(), "B".to_string()]);
        assert_eq!(
            names,
            [
                "A", "A_b0000", "A_b0001", "A_b0002", "B", "B_b0000", "B_b0001", "B_b0002"
            ]
        );
        assert_eq!(
            partition.block_column_flags(8),
            [false, true, true, true, false, true, true, true]
        );
    }

    #[test]
    fn block_budget_needs_a_limit_above_the_threshold() {
        let mut text = String::new();
        for i in 0..(BLOCKS_WITHOUT_LIMIT + 1) {
            text.push_str(&format!("1\t{}\t{}\n", i * 10, i * 10 + 10));
        }
        let wide = bed(&text);
        let refused = wide
            .check_block_budget(None, 3, 1000)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("--blocks-max 501"), "{refused}");
        assert!(
            refused.contains("1000 people x 3 score(s) x 503 columns"),
            "{refused}"
        );
        assert!(wide.check_block_budget(Some(500), 3, 1000).is_err());
        assert!(wide.check_block_budget(Some(501), 3, 1000).is_ok());
        let narrow = bed("1\t0\t100\n");
        assert!(narrow.check_block_budget(None, 3, 1000).is_ok());
        assert!(narrow.check_block_budget(Some(0), 3, 1000).is_err());
    }

    #[test]
    fn sidecar_lists_the_remainder_then_every_block() {
        let mut text = Vec::new();
        bed("chr2\t10\t20\tfoo\n").write_sidecar(&mut text).unwrap();
        let text = String::from_utf8(text).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[1], "#BLOCK_ID\tCHROM\tSTART\tEND\tNAME");
        assert_eq!(lines[2], "b0000\t.\t.\t.\toutside_every_block");
        assert_eq!(lines[3], "b0001\t2\t10\t20\tfoo");
        let mut text = Vec::new();
        BlockPartition::chromosomes()
            .write_sidecar(&mut text)
            .unwrap();
        let text = String::from_utf8(text).unwrap();
        assert!(
            text.lines().any(|line| line == "b0023\tX\t0\t.\tchrX"),
            "{text}"
        );
    }

    fn plan() -> VariantPlan {
        // Three rows: row 0 has entries for scores 0 and 1, row 1 is a complex
        // row without entries, row 2 has one entry for score 1.
        VariantPlan {
            weights: vec![0.5, -1.0, 2.0],
            corrections: vec![0.0, 2.0, 0.25],
            columns: vec![0, 1, 1],
            offsets: vec![0, 2, 2, 3],
            baseline: vec![0.0, 2.25],
            required: vec![BimRowIndex(0), BimRowIndex(1), BimRowIndex(2)],
            complex: vec![GroupedComplexRule {
                locus_chr_pos: ("1".to_string(), 150),
                possible_contexts: vec![(BimRowIndex(1), "A".to_string(), "C".to_string())],
                score_applications: vec![ScoreInfo {
                    effect_allele: "A".to_string(),
                    other_allele: "C".to_string(),
                    weight: 3.0,
                    score_column_index: ScoreColumnIndex(0),
                }],
            }],
            names: vec!["A".to_string(), "B".to_string()],
            counts: vec![2, 2],
            flags: vec![0, 1, 0],
            starts: vec![0],
            total_variants: 3,
        }
    }

    #[test]
    fn expansion_routes_every_entry_to_its_total_and_its_block() {
        let partition = bed("1\t0\t100\n1\t100\t200\n");
        let expanded = expand_plan(plan(), &[(1, 100), (1, 150), (1, 300)], &partition).unwrap();
        // Per score: total, b0000, b0001, b0002.
        assert_eq!(expanded.names.len(), 8);
        assert_eq!(expanded.offsets, [0, 4, 4, 6]);
        assert_eq!(expanded.columns, [0, 2, 4, 6, 4, 5]);
        assert_eq!(expanded.weights, [0.5, 0.5, -1.0, -1.0, 2.0, 2.0]);
        assert_eq!(expanded.corrections, [0.0, 0.0, 2.0, 2.0, 0.25, 0.25]);
        // Unsplit columns keep the join's values; blocks count what they hold,
        // complex applications included.
        assert_eq!(expanded.counts, [2, 0, 1, 1, 2, 1, 1, 0]);
        assert_eq!(
            expanded.baseline,
            [0.0, 0.0, 0.0, 0.0, 2.25, 0.25, 2.0, 0.0]
        );
        assert_eq!(expanded.required, plan().required);
        assert_eq!(expanded.flags, plan().flags);
        let rule = &expanded.complex[0];
        assert_eq!(rule.score_applications.len(), 2);
        assert_eq!(
            rule.score_applications[0].score_column_index,
            ScoreColumnIndex(0)
        );
        assert_eq!(
            rule.score_applications[1].score_column_index,
            ScoreColumnIndex(3)
        );
        assert_eq!(rule.score_applications[1].weight, 3.0);
    }

    #[test]
    fn expansion_checks_its_row_keys() {
        let partition = BlockPartition::chromosomes();
        assert!(expand_plan(plan(), &[(1, 100)], &partition).is_err());
    }
}
