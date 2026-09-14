# gscore: exact reference scorer

`gscore` is a standalone measurement reference for `gnomon score`. It scores PLINK 1
`.bed/.bim/.fam` inputs with gnomon's score semantics, exactly, and times every stage.
Use it to set per-stage budgets, to see how close a single pass can get to its physical
bound, and as an independent exact answer. It is not a product path: it has its own
workspace root, so neither the default build nor CI compiles it.

## What it computes

The semantics follow `score_oracle` (gnomon at a5f27055):

- A score line matches `.bim` rows at the same chr:pos whose `(A1, A2)` equal its
  `(effect, other)` in either order. A locus with one `.bim` row is simple. When a locus
  has several rows, each score line becomes a complex application, resolved per person
  by the heuristic chain of `score/complex.rs`.
- Denominators count one per distinct (row, score) simple assignment plus one per
  complex application. `_AVG = sum / (denominator - missing)`, and `_MISSING_PCT` is
  computed in f32. Score columns are sorted by name, and people follow `.fam` order.
- Inputs are gnomon-native TSV (`variant_id` as `chr:pos`, `effect_allele`,
  `other_allele`, then one weight column per score) or PGS Catalog files, with
  harmonized or original coordinates chosen as `score/reformat.rs` chooses. A directory
  argument scores every file in it.
- Not covered: PGEN, VCF, `--keep`, multiple filesets, and recovering `other_allele`
  from `variant_description`.

## Exactness

- Every weight of score `s` is an f64, so it is an integer multiple of `2^exp[s]`.
  Scaling by the lcm of the complex-resolution denominators makes every person's sum an
  integer, accumulated exactly.
- The quotient is taken once in double-double, as `score_oracle` takes it, so `_AVG` is
  the oracle's value bit for bit.
- Integer sums do not depend on grouping or order. Every kernel, thread count, partition
  and I/O mode writes the same bytes.
- `score_oracle compare --tol 1e-12` gives max_rel 0 on:
  - array3200 × PGS004525 and medium12800 × PGS004525
  - c500k_10k × thin_m1000_k1 and × thin_m1000_k32
  - 20 sampled scores from a 500-score and a 5,000-score catalog, scored alone by the
    scalar kernel and matched against the many-score output

## Kernels

- **`lut`** (default):
  - Four needed rows fold into a 256-entry table of their joint calls. Keys come from a
    SIMD butterfly transposition, 32 people per step.
  - Each entry is a pair: a low u64 with the missing count packed underneath, and a
    high i64. One 128-bit vector add per person applies it.
  - Totals carry into i128 every `2^g` groups; `lane_layout` picks `g` so neither lane
    can overflow.
  - Dense panels with K > 1 are person-major: one table row holds all 2K lanes.
  - Threads split the rows into chunks, or split the people into ranges when per-thread
    copies of the accumulators would be large (`--partition`).
- **`catalog`**: taken automatically for many sparse scores (K > 64). It is score-major:
  each score folds only its own rows, and parallel tasks merge exact totals.
- **`scalar`**: one i128 table lookup per person and row. This is the independent exact
  check.
- **`touch`**: reads only the needed rows and sums their bytes. It measures the physical
  bound of one pass.

`--io pread` reads coalesced runs of the needed rows into per-thread buffers instead of
mapping the `.bed`.

## Build and run

A nightly toolchain is required (`portable_simd`).

```sh
cargo build --release --manifest-path benches/gscore/Cargo.toml
benches/gscore/target/release/gscore score PGS004525_hmPOS_GRCh38.txt array3200 \
  --out array3200.sscore --threads 8
```

Options: `--cache-dir DIR` or `--no-cache`, `--kernel lut|scalar|catalog|touch`,
`--acc pair|fused|plain`, `--gv 4|5|6`, `--partition auto|rows|people`,
`--io mmap|pread`, `--advise`.

stderr ends with one `TIMING <stage> <ms>` line per stage:

- `startup`, `fam`, `open`, `hash`
- `cache_load`, or `cache_miss` followed by `score_parse`, `score_sort`, `bim_parse`,
  `join`, `plan`, `cache_encode`
- `bed_open`, `kernel`, `complex`, `output`, `total`

Process start and exit teardown fall outside `total`. Unmapping the `.bed` costs about
40 ms per GB touched, so time the process from outside as well.

The prepared plan is cached in `--cache-dir` (default: the output's directory), keyed
by a content hash of the score files and the `.bim`.

## Smoke test

```sh
benches/gscore/smoke.sh [work dir]
```

It builds gscore and writes a random fixture with `make_fixture.py`:

- 1,003 people, with padding bits in the last byte and about 2% missing calls
- multi-row loci, flipped and unmatched alleles, empty cells, duplicate score lines
- weights from 1e-9 to 2.5
- native, PGS Catalog, and 80-file catalog inputs

Output must be byte-identical across kernels, thread counts, partitions, I/O modes and
a repeat run from the plan cache. Set `SCORE_ORACLE=<score_oracle binary>` to also
require the native output to be within 1e-12 of the oracle.

## Measurements

Measured 2026-09-14 under sbatch on Genoa (EPYC 9534), against PLINK v2.0.0-a.7.5LM
and gnomon 72665331. Values are medians of 3, wall seconds. "Bound" is `--kernel touch`,
a pass over only the needed rows.

| workload | machine | plink2 | gnomon first / repeat | gscore first / repeat | bound |
|---|---|---|---|---|---|
| array3200 × PGS004525 | -c 2 --mem 4G | 3.67 | 2.58 / 1.53 | 0.83 / 0.62 | 117 ms |
| array3200 × PGS004525 | -c 8 --mem 16G | 1.60 | 2.04 / 0.86 | 0.45 / 0.35 | 39-70 ms |
| array3200 × PGS004525 | -c 16 --mem 16G | 1.46 | 1.96 / 1.03 | 0.30 / 0.18 | 96 ms |
| medium12800 × PGS004525 | -c 2 --mem 4G | 27.1 | 41.6 / 38.7 | 26.7 / 17.0 | 13.6 s |
| medium12800 × PGS004525 | -c 8 --mem 16G | 3.26 | 2.98 / 2.69 | 0.89 / 0.85 | 161-188 ms |
| medium12800 × PGS004525 | -c 16 --mem 16G | 3.18 | 2.78 / 1.81 | 0.49 / 0.43 | 236 ms |
| c500k_10k × thin_m1000_k1 | -c 16 --mem 16G | 0.35 | 0.92 / 0.89 | 0.08 / 0.08 | 32 ms |
| c500k_10k × thin_m1000_k32 | -c 16 --mem 16G | 1.89 | 3.51 / 3.49 | 1.55 / 1.48 | 48 ms |

What the bound column means:

- When the needed rows fit in the page cache, the bound is the page-touch pass, and 10x
  plink2 is physically possible.
- When they don't (medium12800 in 4 GB), storage throughput is the bound.
- Wide outputs are bounded by the output write itself: about 450 MB for 500k × K=32.

Catalog, array3200, -c 8 --mem 16G, one gscore pass against plink2 run once per score
(20 scores sampled and extrapolated):

| catalog | plink2 | gscore first / repeat | per-score speedup |
|---|---|---|---|
| 500 scores, 127k rows | about 71 s | 0.48 / 0.44 | 147x / 160x |
| 5,000 scores, 13.4M rows | about 1,021 s | 6.44 / 5.26 | 158x / 194x |
