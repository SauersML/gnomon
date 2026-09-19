#### Example biobank run:
Get gnomon ready:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && git clone https://github.com/SauersML/gnomon.git && cd gnomon && rustup override set nightly && cargo build --release && cd ~
```

Or to update gnomon:
```
cd ~/gnomon && git remote set-url origin https://github.com/SauersML/gnomon.git && git fetch --prune origin && br=$(git remote show origin | sed -n 's/.*HEAD branch: //p') && git checkout "$br" && git reset --hard "origin/$br" && git submodule update --init --recursive && cargo build --release && cd ~
```

Download data:
```
gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* .
```

Run a local score:
```
./gnomon/target/release/gnomon score "score.txt" arrays
```

Or use a score from PGS catalog:
```
./gnomon/target/release/gnomon score "PGS003725" arrays
```

Or stream the data:
```
./gnomon/target/release/gnomon score "PGS003725" gs://fc-aou-datasets-controlled/v8/microarray/plink/*
```

Optionally, use a keep file:
```
awk '{print $2}' arrays.fam | head -n 256 > keep.txt
```

```
./gnomon/target/release/gnomon score "PGS003725" arrays --keep keep.txt
```

#### Choosing where outputs go

By default the results land beside the genotypes as `<genotypes>_<score stem>.sscore`
(`arrays_score.sscore` for the first run above), or in the working directory for
`gs://` and `https://` inputs. The converted and sorted score-file caches land beside
the score file. gnomon refuses to overwrite an existing `.sscore`.

`--out PREFIX` writes the results to `PREFIX.sscore` instead. Everything else the run
keeps goes under PREFIX's directory: downloaded PGS Catalog files and the score-file
caches (`gnomon_score_cache/`).
```
./gnomon/target/release/gnomon score "PGS003725" arrays --out results/arrays_pgs003725
```

Nothing is written beside the genotypes or the score files, so the inputs may sit in a
read-only or shared directory. Concurrent runs on the same inputs are safe when their
prefixes differ. BCF, DTC and `--panel` inputs are converted to PLINK into a
`<stem>.<key>.gnomon_cache/` directory under `gnomon_score_cache/`. One genotype-side
intermediate still goes beside the genotypes: the re-sorted fileset written when a
`.bim` is out of order (`<prefix>.sorted.{bed,bim,fam}`).

Without `--out`, the conversion cache stays in `<stem>.gnomon_cache/` beside the input,
and so do the results. Each conversion is written to a private temporary directory and
renamed into place as `g-<key>/` once complete. The key covers the source file's size
and modification time and the `--build`, `--panel` and `--reference` in use, so a run
never reads a partial conversion or one made from something else. Once a newer
conversion under the same parameters is published, the older one is removed. Caches
written by earlier gnomon versions are ignored and left in place.

Each score-file cache entry under `gnomon_score_cache/` is named with a key over the
gnomon build and the score file's path, size and modification time, so an edited score
file is converted again rather than served from a stale entry. Without `--out`, a score
file in a directory gnomon cannot write is cached under the user cache directory
instead: `$XDG_CACHE_HOME/gnomon/score_cache`, which defaults to
`~/.cache/gnomon/score_cache` on Linux.

Results are written to a temporary file in their destination directory
and renamed into place once complete, so another process never reads a partial
`.sscore`.

#### Per-block partial scores

`--blocks` adds, beside each score, the same weighted sum restricted to each block
of a genomic partition: the genetic feature interface a phenotype-supervised
calibrator needs.

```
./gnomon/target/release/gnomon score "PGS003725" arrays --blocks chrom
./gnomon/target/release/gnomon score "PGS003725" arrays --blocks ld_blocks.bed
```

`chrom` gives one block per chromosome (ids `b0001`-`b0022`, X = `b0023`, Y = `b0024`,
XY = `b0025`, MT = `b0026`). A BED file gives one block per row in file order
(`chrom start end [name]`, 0-based and half-open, so a variant on a boundary belongs
to exactly one block; rows on one chromosome must not overlap). Block `b0000` holds
the variants outside every block, so for every person and score the block partials
sum to the unsplit score. The columns are `<SCORE>_b<ID>_AVG` and
`<SCORE>_b<ID>_MISSING_PCT` (`_SUM` and `_MISSING_CT` under `--emit-components`); a
block holding no variant of a score reports `0.0` and `100.0`. The sidecar
`<output>.blocks.tsv` maps each block id to its interval.

The expansion happens when the variant plan is compiled: each variant's weight is
routed to its score's column and to its `(score, block)` column, and the multi-score
engine computes every partial with the arithmetic it uses for the unsplit score. The
cost is `people x scores x (blocks + 2)` accumulator cells; more than 500 blocks
needs `--blocks-max <n>` to proceed. Without `--blocks` nothing changes.

#### Unmatched score rows

`--unmatched-report PATH` writes each weight of a score row that adds to no score, one
line a weight: the score, the row's variant and alleles, why, and the alleles the
genotypes hold at its position (each row's pair without the trailing bases it shares,
in text order).

```
#score	variant_id	effect_allele	other_allele	reason	alleles_seen
S1	1:100	C	A	no_allele_pair	A/G,A/T
S1	1:300	A	G	no_variant_at_position	.
```

The reasons are `no_variant_at_position`, `no_allele_pair`, `several_alleles` (a row
naming no single other allele whose effect allele is more than one allele there) and
`outside_region`. Every path decides a row by one site rule, so a joined or split
VCF, a BCF and a PGEN of the same genotypes give the same file, and so does a `.bed`
wherever its alleles read as the REF the others declare (a `.bim` declares none). Rows the
score normalization drops (no position, an unsupported contig, a malformed line) are
named with their lines in its warning instead.

#### Example:
```
./target/release/gnomon score ./ci_workdir/PGS004696_hmPOS_GRCh38.txt ./ci_workdir/gnomon_native_data
```

#### Debug:
```
./target/debug/gnomon score ./ci_workdir/PGS004696_hmPOS_GRCh38.txt ./ci_workdir/gnomon_native_data
```
