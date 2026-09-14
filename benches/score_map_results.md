# Score and packed projection performance

## Sparse row reordering without duplicating the complete plan

Preparation now skips CSR reordering when matched BIM indices already strictly
ascend. Otherwise it reorders and releases one old value array at a time,
instead of constructing a second complete CSR plan. Row permutations,
reordered values, row metadata, and CSR growth reserve memory fallibly.
An allocation failure aborts preparation with an error. This does not establish
OOM freedom for the rest of preparation or the host operating system.

On the real 1,188,880-row, 5,385,268-weight, 32-score plan, one pinned Milan
core and three repetitions per case:

| Matched row order | Previous median | New median | Previous peak extra allocation | New peak extra allocation |
| --- | ---: | ---: | ---: | ---: |
| Already ascending | 91.95 ms | 0.84 ms | 93.16 MB | 0 |
| Reversed physical indices | 65.68 ms | 39.80 ms | 93.16 MB | 31.05 MB |

These are isolated reorder times (about 110× and 1.65× faster), not full
scoring speedups. The reversal permutes physical row indices of the actual
genome-scale plan; its weights and row sizes come from the real PGS panel.
Every weight bit, missing-correction bit, column index, row offset, and complex
flag matched the expected permutation. The allocation figures measure live
Rust allocation bytes above the starting value, not process RSS.

Construction of that same CSR was checked separately to avoid moving the cost
into ordinary preparation. Capacity checks guard an outlined fallible growth
routine; after they establish available slots, direct writes avoid a second,
infallible reserve path. Six alternating before/after pairs took median 62.87 ms
before and 58.92 ms after, with identical bits and identical 117.44 MB peak
extra allocation. The first implementation called reservation machinery on
every contribution and was substantially slower; it is not the final path.

`benches/csr_reorder_probe.py` extracts both production CSR builders and reuses
the MSI compilation cache. Its allocator-injection checks passed for constructor,
all three contribution-array growth points, row-offset growth, and all seven
reorder allocation points. Failed contribution growth preserves parallel lengths;
ordered rows allocate nothing. Empty rows, signed zero, and invalid row metadata
are also checked. All 42 focused preparation, scoring, and memory tests passed
in the warm MSI harness. Logs use `round16-` in the MSI iteration directory.

## Independent source-weight accuracy audit

`benches/compare_exact_score.py` compares the cohort probe's SUM output with
the independent `benches/gscore` fixed-point reference's AVG output. The probe
writes score denominators to a `.meta.json` sidecar. The comparison verifies
the person-ID bijection, finite values, and missing percentages in the f32
representation emitted by gscore. Percentages cannot prove exact missing
counts for arbitrarily large denominators. Relative tolerance defaults to
1e-12 with zero absolute allowance; any excess returns a failing exit status.
Reports include absolute and relative worst cases and per-score errors.

On the complete 1,799,239-marker, 512-person BED and 32 normalized PGS files,
all missing percentages agreed. Maximum AVG error was 2.67028808e-8, for
PGS000023: production 1.6779999732971191 versus reference 1.678. The largest
relative error was 1.0 at a near-zero PGS000026 value (0 versus 1.15648e-18).
16,226 of 16,384 cells exceeded the default relative tolerance. These findings
are **an unresolved source-precision gap**, not a passing accuracy result.
Earlier comparisons against an older production binary share its f32 weight
representation and therefore do not establish source-weight accuracy.

An isolation check rounded only PGS000023's input weights to f32 before
passing them to the exact reference. All 512 averages then matched production
exactly. This identifies weight quantization as the cause of that score's
observed difference; it does not attribute every other score's error.
Logs and outputs use `round15-` in the MSI iteration directory. The reference
ran without its preparation cache on four pinned Milan cores; its full
32-score run took 2.89 seconds and peaked at 1.09 GiB RSS. This audit covers
BED scoring, not every supported input format or phenotype regime.

## Reused row masks and bounded output blocks

Sparse packed scoring builds a 2 KiB column/row membership mask once per
256-variant chunk, then visits only set bits when assembling score schedules.
It processes 512 people across all columns before moving on, reusing output
cache lines. Together with the existing schedule, scratch is roughly 8 KiB
per worker, independent of cohort size. Accumulation order within each score
is unchanged. Rebuilding memberships for every output block was measured to
be slower and is not used.

Three real 256-marker batches sampled across the 1,188,880 reconciled rows,
32 scores, one pinned Milan core, three paired repetitions per batch:

| People | Before batch medians | After batch medians | Speedup range |
| ---: | --- | --- | --- |
| 512 | 0.450 / 0.495 / 0.527 ms | 0.266 / 0.316 / 0.332 ms | 1.57–1.69× |
| 3,200 | 1.835 / 2.092 / 2.227 ms | 1.643 / 1.905 / 2.027 ms | 1.10–1.12× |
| 51,200 | 28.074 / 33.491 / 36.132 ms | 27.447 / 32.699 / 34.535 ms | 1.02–1.05× |

All batch score bits and missing counts matched exactly. These are kernel
measurements, not complete biobank runtimes. The larger fixture replicates
the 3,200-person seed cohort. The probe reads just the sampled BED rows and
does not map the complete biobank file.

A paired complete 1,799,239-marker, 512-person run on four cores took 828 ms
before and 674 ms after for computation (1.23×, single observation).
The 3,200-person candidate took 2.87 s. Complete outputs matched all missing
counts; maximum relative differences were 2.92e-13 and 4.55e-12, respectively,
against their references. Forty-one focused tests passed, including scalar
checks around the 512-person block boundary. Logs use `round13-` in the MSI
iteration directory; `benches/probes/score_batch_cache.rs` reproduces the
sampled kernel comparison against the cached baseline library.

## Kept people and small narrow panels

The packed schedules now gather arbitrary kept people directly from the physical
BED rows. Per-block byte offsets and bit shifts are reused across the scheduled
variants; scratch remains a fixed schedule plus roughly 384 bytes of selection
topology. The complete-cohort specialization removes gather code at compile
time. Dispatch also admits common-call rows from 1–4-score panels with at least
64 selected people, which the old tree often sent to scalar accumulation.

Full 1,799,239-marker inputs on four pinned MSI Milan cores:

| Scored people / physical cohort | Scores | Before compute | After compute | Speedup |
| --- | ---: | ---: | ---: | ---: |
| 512 / 512 | 1 | 655 ms | 100 ms | 6.57× |
| 512 irregularly kept / 3,200 | 1 | 1,013 ms | 290 ms | 3.49× |
| 512 irregularly kept / 3,200 | 32 | 3,551 ms | 1,540 ms | 2.31× |
| 65 irregularly kept / 3,200 | 32 | 762 ms | 653 ms | 1.17× |

These are single before/after observations; preparation and output writing are
excluded. A second 512-person one-score check took 111 ms and matched its old
output exactly. The one-score keep check differed by at most 1.78e-15 absolute.
Both wide keep cases were checked against the complete 3,200-person reference;
maximum relative difference was 3.26e-14. Every missing count matched exactly.
Peak RSS for the wide keep checks stayed approximately 1.42 GiB, dominated by
the mapped 1.44 GB physical BED; no cohort-sized genotype tile is added.

Forty-one focused tests pass, including scalar comparisons for gapped and
unaligned selections, 1–64 score columns, and partial person/variant groups.
The warm production scoring library build passed on MSI in 38.00 seconds.
Reproduction uses `cohort_score_probe.py --baseline`, `cohort_score_probe.py`,
and `make_keep_probe.py`; logs use the `round12-` prefix in the MSI iteration
directory. The baseline is the previous packed-panel library, not the original
pre-optimization application.

The preceding scale check scored all 12,800 replicated people across the same
32-score panel in 9.70 seconds, with exact missing counts and maximum relative
difference 4.55e-12 against all 3,200 seed outputs. Peak RSS was 5.42 GiB (the
mapped 5.76 GB BED). A 51,200-person streaming check reached roughly 65% before
its 35-second cap, with peak RSS 301 MiB. Its accuracy comparison was incomplete;
neither full runtime nor universal OOM freedom is established by that run.

## Sparse wide-panel schedules and dispatch

For complete cohorts of at least 64 people and panels of 5–64 scores, the
producer now considers each marker's actual score support. Rows touching no
more than one quarter of the score columns use the packed batch route when
their genotype density exceeds 0.0894; rare calls retain zero-word skipping.
The batch kernel compiles per-score schedules of at most 256 active rows,
then accumulates across 32 packed people in f64 SIMD registers. Its schedule
occupies about 6 KiB, independent of cohort size. Other panel densities and
keep subsets retain their existing regime choices.

Four pinned MSI Milan cores, full 1,799,239-marker BEDs, 32 normalized PGS
files, 1,188,880 matched rows, 5,385,268 nonzero weights:

| People | Previous pipeline | Scheduled pipeline | Speedup |
| ---: | ---: | ---: | ---: |
| 512 | 2.651 s | 0.656 s | 4.04× |
| 3,200 | 13.825 s | 2.582 s | 5.35× |

These are paired single observations, excluding normalization, preparation,
and output writing. Every missing count matched exactly. The maximum absolute
score difference was 7.451e-9 and maximum relative difference was 2.918e-13.
Combined before/after process peak RSS was 435 MiB and 1.54 GiB respectively;
the larger process maps a 1.44 GB BED. Neither check exhausted memory. This
does not establish universal OOM freedom or performance for arbitrary panels.

The original total-score-width dispatcher routed these rows to scalar
accumulation, so merely adding a dense-path kernel did not improve the full
run. The final implementation changes dispatch as well as execution. Earlier
paired 3,200-person development runs hit their 35-second cap and terminated;
no speedup is claimed from those incomplete comparisons. Reproduction:
`wide_compute.py`, then its `wide-compute` binary with genotype prefix,
normalized-score directory, and repetition count. Logs: `round10-wide-n512.log`
and `round10-wide-n3200.log` in the MSI iteration directory.
Forty focused checks pass, including scalar f64 comparisons across 5–64
columns, packed person/variant boundaries, exact missing counts, common/rare
dispatch, and keep-subset exclusion. The dispatch regression specifically
guards against leaving the optimized kernel unreachable in this regime.
The warm production scoring library build passed on MSI in 38.97 seconds.

## Uncached wide-panel reconciliation

Ordinary singleton BIM loci now accumulate duplicate contributions in reusable
score slots and sort only touched columns before CSR emission. This removes
per-locus reconciliation trees and match lists while preserving f32 input
addition order, zero-weight entries, missing corrections, and complex-locus
semantics. Scratch scales with score count and is allocated fallibly once.

On the full 1,799,239-marker input with 32 normalized PGS files, three paired
uncached preparations measured 5.324/3.546, 4.690/3.619, and 5.301/3.599 seconds
before/after: median 1.47× faster. All 1,188,880 matched rows, 5,385,268 sparse
weights, float bits, corrections, counts, and complex rules matched; final
single-person score and missing-count arrays were exactly equal. Both calls
exclude normalization and cache lookup/publication. The comparison uses the
archived pre-cache compiler and the candidate compiler against the same warm
library on four pinned MSI Milan cores.

Thirty-seven focused checks pass, including a new wide-panel fixture with
cancellation-sensitive duplicates, allele flips, sparse column resets, and
unmatched allele pairs. Reproduce with `cold_wide_plan.py`, then the emitted
`cold-wide-plan` binary and full-marker inputs. Logs: `round6-cold-wide.log`
and `round6-checks.log` in the MSI iteration directory. Prepared score inputs
are retained in `real-genome/round6-mixed32` to avoid shared fixture cleanup.

## Content-addressed compiled plans

The subsequent hashing revision uses fixed 256 KiB content chunks, hashing at
most four concurrently within a reusable window capped at 1 MiB and 1/64 of
available memory. Chunk identities are independent of worker count and window
size. Short reads, changed boundary bytes, truncation, and growth are tested;
all source bytes are still verified on every lookup. Thirty-eight focused
checks pass. Paired warm preparation checks measured 134–136 ms before versus
71–101 ms after for PGS000018, and 543–555 ms versus 332–433 ms for 32 PGS
files. Compiled values and the wide panel's final arrays matched exactly.
These are two warm observations following cache creation, not broad medians.
Logs use the `round7-` prefix; `cold_wide_plan.py --cached` builds the paired
wide-cache probe.

Local BIM inputs can now reuse a compiled variant plan across cohort sizes,
keep lists, and file locations. Every lookup hashes the complete BIM and
normalized score contents, score order, region filters, and compiler source
identity. It reconstructs person IDs, physical row widths, and fileset paths
from the current inputs. Floating-point values use bit-preserving binary
encoding; payload checksums, index validation, checked lengths, and charged
allocation limits precede use. Plans live in the platform user cache under
`gnomon/variant-plans`; writes are atomic. The file and decoded-allocation
ceilings are each the smaller of 256 MiB and one eighth of available memory.
Collections exceeding 1 GiB of source text, remote BIMs, and PVAR adapters
use streaming compilation. Diagnostic-producing and sort-retried compilations
are not published under the original inputs' key.

On the full 1,799,239-marker input with PGS000018, three paired warm probes
gave median preparation times of 1.020 s before and 110.45 ms after (9.2×).
Every compiled index, float bit, baseline, count, and complex rule matched.
Complete cached single-person CLI checks took 0.369 s and 0.228 s, versus the
previous 1.355 s check. Both outputs are byte-identical to the original
baseline; peak RSS was approximately 30 MiB. These CLI observations are warm
shared-storage checks, not controlled medians.

For 32 real normalized PGS files, 1,188,880 matched markers, and 5,385,268
nonzero weights, a library-level probe measured uncached preparation at
4.75–5.33 s and cached preparation at 0.515–0.642 s. All compiled artifacts
and final score/missing-count arrays were identical. The first compile plus
cache publication took 5.439 s versus 5.149 s for the paired compiler-only
call: caching buys repeated-use speed and has a first-use cost. Normalization
is excluded from this probe; an older benchmark CLI's repeated-directory
discovery bug prevented a fair wide-panel CLI comparison.

Thirty-six focused tests pass, including changed content at unchanged size
and mtime, corrupt/truncated caches, hostile length fields, float-bit round
trips, and rebinding the same plan to different sample counts and keep lists.
The warm production library and CLI builds passed on MSI. Reproduction uses
`prepare_probe.rs`, `probes/wide_plan.rs`, and `preparation_memory_checks.py`;
logs are under the MSI iteration directory with `round5-` and `wide-plan-`
prefixes. No new guarantee for arbitrary input sizes or OOM freedom is implied.

## Preparation and memory revision

The next revision avoids heap allocation for common literal alleles, reuses
locus groups, and emits singleton matches directly into CSR. A paired probe
compared every compiled index, weight bit, correction, count, and complex rule
on the full PGS000018 input. Warm preparation fell from 1.024–1.034 seconds
to 0.804–0.808 seconds; all compiled results were identical. The initial pair
was 1.815 versus 0.857 seconds and is excluded from the warm comparison.

Mapped cohorts of at most 32 people now use direct packed reads even when
complex loci are present, then invoke the ordinary complex resolver. The
full-marker one-person command took 1.541 seconds, with 43.13 milliseconds in
the pipeline (previously about 313 milliseconds). Its output is byte-identical
to the original baseline. The 3,200-person command took 1.612 seconds, including
441.86 milliseconds in the pipeline; maximum baseline difference remains
1.251e-12 and missing percentages are exact. Peak RSS was 25.3 MiB and 1.33 GiB,
respectively. These are single end-to-end checks on four pinned MSI cores,
not a controlled median. They include main's concurrent row-major complex
resolver and packed sex-inference improvements, so gains are not attributable
solely to the preparation changes.

Scoring now rejects output shapes that cannot fit the minimum bounded plan
before allocating outputs, including calls that bypass CLI preflight. The
plan reserves checkpoint/output copies and dense scratch; wide score matrices
reduce bounded batch sizes. A zero free-memory reading grants no emergency
8 GiB allocation. Exact Vec reservations now use additional length correctly.
Twenty-six focused tests pass, including oversized shapes without allocating
them and manually calculated simple/complex dosages across single/split
filesets, keep subsets, missing calls, and partial packed bytes. The warm
production library and CLI builds passed on MSI.

The subsequent memory revision intersects host RAM with Linux cgroup v1/v2
limits and remaining charged headroom at every visible ancestor. Score, PCA
fit, and VCF materialization share this detection. Unknown limit data fails
closed. PCA no longer interprets zero availability as total RAM, invents an
8 GiB pool, or raises small budgets to fixed floors totaling 1 GiB. Thirty-one
focused tests pass with the new detector, including exhausted and nested
limits, missing usage data, escaped mount paths, and disabled controllers.
The production library build passed using the warm MSI cache.

A subsequent warm CLI check completed the same full-marker PGS000018 workload
at larger cohort sizes:

| People | Wall time | Pipeline time | Peak RSS |
| ---: | ---: | ---: | ---: |
| 1 | 1.355 s | 22.15 ms | 24.1 MiB |
| 51,200 | 11.703 s | 10.45 s | 249.4 MiB |
| 204,800 | 21.481 s | 19.73 s | 443.0 MiB |

Every result in both replicated cohorts was compared with the original
3,200-person seed: maximum absolute difference 1.251e-12, identical missing
percentages. Single-person output remains byte-identical. Both larger runs
finished below the 45-second cap. These are warm shared-storage checks of the
combined main improvements; they do not isolate individual changes or measure
wide panels, imputed whole-genome sequencing, or independent biobank diversity.
The PCA entry point also rejects a physical cohort whose minimum one-variant
decode scratch cannot fit, before opening or decoding the source.

This closes specific allocation risks; it is not a universal no-OOM guarantee.
Preparation allocation, allocator overhead, and memory consumed concurrently
by other processes still need broader accounting and stress validation.
`preparation_memory_checks.py` and `prepare_probe.rs` reproduce the focused
checks; MSI logs use the `round3-` prefix and `preparation-memory-checks.log`.

## Full-marker revision

The earlier 8,192-marker fixtures below are development probes, not realistic
whole-genome workloads. `real_genome_probe.py` preserves all 1,799,239 array
markers for scoring and all 562,259 reference markers for projection against
a 570,709-marker, 20-PC model. Larger reference cohorts repeat samples;
they exercise scaling, not independent biobank genetic diversity.

Narrow panels now vectorize across 32 people directly from packed calls,
accumulating scores in f64 without genotype expansion or cohort-sized scratch.
Panels with more than four scores use a packed four-by-four transpose before
dosage expansion. Arbitrary keep subsets retain their indexed gather path.
Projection marker matching borrows model strings and indexes each coordinate
once, preserving exact/wildcard/swap priority and requested-marker order.

On one pinned Milan core, three-run median compute probes (256-variant blocks,
50,000 people) improved 6.01×, 4.41×, 3.03×, and 3.14× for 1, 2, 3, and 4
scores respectively. These are kernel measurements, not whole-genome command
speedups. The nine-score probe improved 1.25× at 50,000 people and 1.23× at
250,000. All 16 focused tests passed, including an independent f64 oracle,
sparse score membership, missing counts, sample/variant tails, and marker
matching errors. Production release compilation passed on MSI using the warm
cache; no local builds or code execution were used.

Source and reusable scripts live in the normal checkout. MSI build artifacts
and raw measurements now live under
`/projects/standard/hsiehph/sauer354/gnomon/target/score-map`.

Full-marker command measurements (four pinned cores, shared MSI storage):

| Workload | Baseline | Current | Current peak RSS |
| --- | ---: | ---: | ---: |
| Score: 1 person, 1,799,239 input markers, PGS000018 | 1.734 s | 1.830 s | 24.4 MiB |
| Score: 3,200 people, same input and score | 4.015 s | 2.442–2.604 s | 1.33 GiB |
| Project: 1 person, 562,259 input markers, 20 PCs | 3.992 s | 2.102 s | 664.2 MiB |
| Score: 51,200 people, 1,799,239 input markers, PGS000018 | stopped at 35 s | 24.934 s | 251.2 MiB |

PGS000018 matches 330,861 markers, including 2,527 complex-rule loci. The
51,200-person dataset repeats the 3,200-person reference cohort sixteen times.
All 51,200 results agree with the baseline seed scores within `1.251e-12`
absolute error; missing percentages are identical. The packed narrow kernel
uses f64 accumulation, so last-bit changes from the former f32 minibatch
sums are expected. Single-person score and projection outputs are byte-identical.
Single-person scoring remains preparation-bound; this revision does not speed
that command up. Projection marker selection fell from 1.951 to 0.636 seconds;
model loading also varied with filesystem cache state.

The large baseline was deliberately terminated rather than extended into a
long job. Its incomplete timing is not an exact speedup denominator. Before
the local I/O change, the new compute kernels reached 36% of matched rows in
35 seconds. Planned positional reads reached every matched row within the
same limit, then entered complex-rule resolution. A subsequent warm run
completed in the 24.934 seconds above; cache state affects these shared-storage
measurements. The completed run's 45-second cap was not reached.

Large local BED inputs now reuse the existing range-prefetch engine, merging
nearby required rows into ranges of at most 2 MiB and using at most two I/O
workers per active reader. Prefetch storage is capped at the smaller of
16 MiB and 1/32 of the memory budget, divided across filesets and included in
preflight accounting. Smaller inputs retain mapped reads. Three additional
checks cover coalesced range bounds, cross-range byte reassembly, truncation,
backward reads, and the prefetch window's outstanding-byte bound. No tested
run ran out of memory. These measurements do not establish performance for
every input format, score width, missingness pattern, or cohort size.

The next projection revision compiles four variants into one 256-row lookup,
with at most 256 KiB of table scratch plus 8 KiB for construction and 1 KiB
per active sample tile. Cohorts below 1,024 people and panels above 64 PCs
retain the smaller direct/pair kernels. Missing calls are recovered from the
same key. A packed missing-call census selects sparse indices or dense
information matrices before accumulating them: sparse index capacity is
bounded by the dense matrix payload plus 16 bytes per person, rather than
growing without limit with the number of variants.

Updated full-command measurements against the original baseline, with the
same four-core setup (three-run medians unless marked otherwise):

| Workload | Before | Current | Speedup | Current peak RSS |
| --- | ---: | ---: | ---: | ---: |
| Score: 1 person, 2,048 variants, 9 scores | 0.519 s | 0.054 s | 9.65× | 13.2 MiB |
| Project: 1 person, 4,096 variants, 4 PCs | 0.040 s | 0.042 s | 0.95× | 17.1 MiB |
| Score: 50,000 people, 8,192 variants, 9 scores | 2.178 s | 1.291 s | 1.69× | 193.1 MiB |
| Project: 50,000 people, 8,192 variants, 4 PCs | 2.469 s | 0.815 s | 3.03× | 139.2 MiB |
| Score: 500,000 people, 8,192 variants, 9 scores (one pair) | 12.794 s | 8.427 s | 1.52× | 1.68 GiB |
| Project: 500,000 people, 8,192 variants, 4 PCs (one pair) | 21.125 s | 5.957 s | 3.55× | 1.19 GiB |
| Score: 1,025 people, 2,048 variants, 9 scores, 7.7% missing | 0.630 s | 0.082 s | 7.67× | 15.8 MiB |
| Project: 1,025 people, 4,096 variants, 4 PCs, 7.7% missing | 0.200 s | 0.139 s | 1.44× | 17.1 MiB |

The 500,000-person fixture repeats the synthetic 50,000-person genotypes ten
times with unique sample identifiers; it tests scale, not genetic diversity.
The single-person rows include the final Linux process-census change; larger
rows were measured before that startup-only change. Single-person projection
is essentially unchanged within the observed scheduling/startup variation.
Score outputs remained byte-identical. Grouped projection changes floating
point addition order: the 50,000-person maximum absolute difference was
`1.36e-13`, within the independent scalar checks' tolerance. All 13 focused
tests also passed after incorporating main's parallel VCF reader and feature
gating changes. The 500,000-person projection difference was `1.60e-13`.
Coverage includes padded bytes, partial variant groups, allele swaps,
both missingness representations, and SIMD boundaries. The production score
and map executables were rebuilt from the warm cache.

`packed_projection_probe.py` also compares kernels on one pinned core.
At 50,000 people and 257 variants, four-PC accumulation improved from
42.26 to 27.01 ms and 64-PC accumulation from 176.38 to 90.74 ms against
the preceding optimized pair kernel. These exclude CLI and solve overhead.
The attempted grouped-table scoring engine regressed several regimes and
was removed. `score_map_cli_probe.py` adds `--single`, `--large`, and
`--biobank`; use `--once --score-only` or `--once --project-only` to keep
each biobank experiment bounded. RSS is measured by `/usr/bin/time`.
`--missing` exercises automatic dense information storage with a partial
sample byte. Projection differs from the original by at most `6.40e-14`.
Grouped projection caps sample tiles at the existing cohort-dependent chunk
size, preserving parallelism around the 1,024-person dispatch threshold.
The small missing-data timings varied with shared-node load; the reported
comparison uses interleaved runs from the same final measurement session.

Linux memory planning now enumerates process leaders through `/proc/*/comm`.
The former sysinfo census took 198 ms on its first call and 43 ms warm, versus
20–25 ms for names alone. More significantly, it reported 54 gnomon tasks
where `/proc/*/status` confirmed only four unique process leaders. The new
census counts the four processes for the fair-share memory cap, so starting
more worker threads no longer shrinks a process's budget. Non-Linux systems
retain the narrow sysinfo query. VCF materialization also refreshes memory
without building an unrelated process census.

The earlier checkpoint measurements follow.

The memory-planning startup query was subsequently narrowed to the fields
actually used. On the shared MSI node, full system refresh took 267–492 ms;
memory plus process names took 32–37 ms and found the same 17 gnomon processes.
PCA's memory-only query took 0.37–0.41 ms. Memory availability and scoring's
concurrent-process fair-share checks remain active. These are query timings,
not measurements of the complete commands below.

Measured on MSI on 2026-09-13 using four pinned AMD EPYC Milan cores,
Rust nightly-2026-08-31, `target-cpu=x86-64-v3`, release optimization, and
the existing iteration cache with LTO disabled. Each number below is the
median of three runs, alternating before/after order. Inputs were local,
synthetic PLINK hard calls with warm filesystem caches.

| Workload | Before wall | After wall | Speedup | Before compute | After compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Score: 50,000 samples, 8,192 variants, 9 scores | 2.013 s | 1.422 s | 1.42× | 1.380 s | 0.806 s |
| Project: 50,000 samples, 8,192 variants, 4 PCs | 2.230 s | 0.917 s | 2.43× | 2.146 s | 0.829 s |
| Score: 16,384 samples, 2,048 variants, 9 scores | 0.709 s | 0.758 s | 0.94× | 0.123 s | 0.076 s |
| Project: 16,384 samples, 4,096 variants, 4 PCs | 0.514 s | 0.283 s | 1.82× | 0.457 s | 0.208 s |

Wall time includes process startup, preparation, computation, and output.
Compute time is the command's reported pipeline/projection time. The smaller
score workload illustrates the limit: saving 47 ms of computation did not
overcome variation in the rest of the command. These measurements do not
establish gains for PCA fitting, GPU execution, or remote genotype storage.

The scoring changes classify 32 dosages into separate SIMD masks and select
constant-width accumulator loops once per score stripe. The projection change
uses a four-bit lookup to add two samples' contribution vectors together and
walks missing-call masks separately. Cohorts smaller than 128 samples index
the model's vectors directly, avoiding the cost of constructing pair tables.
Variant accumulation order and missing-call corrections are preserved.

All before/after score TSVs and projection binaries were byte-identical,
including repeated runs. SHA-256 for the larger workload:

- Score: `46e0971ddac9e6d0cf7414421bb718949f31f8fdd9d11fdaf617435cc4db5f49`
- Projection: `409ac9f3073d7fd9085c13b80cccee97678166f545d2874ed427f91741c0a0ef`

Ten focused tests passed using the production score modules and projection
kernels. Coverage includes all 256 packed bytes, allele swaps, both missing
information representations, partial sample bytes, SIMD and mini-batch
boundaries, every accumulator width, and checked kernel bounds. The broad
Cargo test build was stopped when its development dependencies began rebuilding;
the complete test suite was not run. Production scoring and projection binaries
were built and exercised on MSI.

Benchmark scripts, logs, source snapshots, and raw JSON results are retained
on MSI at
`/projects/standard/hsiehph/sauer354/gnomon/.validation/score-map-20260913`.
`bench_cli.py` runs the smaller workload; `bench_cli.py --large` runs the larger
one. Run under `taskset -c 24-27` with a 60-second timeout. The fixture is a
fixed prefix of `gnomon-swarm/data/map/synth/c50k_20k`, with nine deterministic
score columns and a four-PC model fitted once by the baseline executable.
