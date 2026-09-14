## Overview
The `map` crate implements gnomon's Hardy–Weinberg–standardized PCA pipeline.
It powers the `gnomon fit` and `gnomon project` subcommands and also offers
library APIs for model fitting, serialization, and projection. The
implementation streams genotype data in fixed-width blocks so it can operate on
everything from small cohorts to biobank-scale datasets, automatically choosing
between dense and iterative eigensolvers based on the requested number of
principal components.

## Why Hardy–Weinberg scaling?
Principal component analysis expects comparable per-locus variance. Genotype
counts drawn from a bi-allelic site follow the Hardy–Weinberg variance
`2p(1−p)` where `p` is the allele frequency. gnomon centers each locus and
divides by `√(2p(1−p))`, which keeps well-imputed common and rare variants on
the same footing without forcing mean imputation up front. This scaling also
means out-of-sample projections can reuse the same transformation without
recomputing cohort-specific statistics.

## CLI entry points
| Command | Purpose | Required input | Primary outputs |
| --- | --- | --- | --- |
| `gnomon fit --components <N> [--threads N] [--allow-unconverged] [--maf MAF] [--geno RATE] [--list PATH] [--keep PATH] [--ld] <GENOTYPE_PATH>` | Train a Hardy–Weinberg PCA model | Genotype source (PLINK trio or VCF/BCF files, local or remote) | `hwe.json`, `samples.tsv`, `hwe_summary.tsv`, `hwe_scores.bin` plus `hwe_scores.metadata.json` |
| `gnomon project <GENOTYPE_PATH>` | Project samples with an existing `hwe.json` located next to the genotype data | Matching genotype source aligned to the model's variant set | `projection_scores.bin` plus `projection_scores.metadata.json` |

Both commands print sample and variant counts, resolved source paths, and
stage-aware progress while running.

### `gnomon fit`
`gnomon fit` estimates the PCA loadings, explained variance, and per-locus
scalers and saves them as `hwe.json`. Use it when you want a reusable PCA basis
for later projections.

Required arguments:

* `GENOTYPE_PATH` – Path or URI pointing to the genotype data. Remote objects
  and per-chromosome directories are supported, as long as the loader can infer
  the associated `.bim`/`.fam` metadata or VCF headers.
* `--components <N>` – Number of principal components to retain in the model.

Optional arguments:

* `--threads <N>` – Run the fit in a dedicated `N`-worker pool, to match a
  scheduler allocation or cap a shared machine; without it, Rayon uses the
  process's available parallelism (its CPU affinity and cgroup quota). The fit
  prints the active worker count before reading genotypes. The count changes
  speed, never the answer: every product in the fit runs on row leaves whose
  size depends only on the matrix shape, so the model and scores are the same
  bytes at any thread count.
* `--list <PATH>` – Restrict fitting to a variant subset. The file (local or
  remote) should contain two whitespace-separated columns—chromosome and
  1-based position—with an optional header. Any variants that cannot be found
  are reported before the command exits.
* `--keep <PATH>` – Restrict fitting to a sample subset. The file holds one
  individual ID (`IID`) per line, the same format `gnomon score --keep` takes;
  blank lines are ignored and an ID that is not in the dataset is an error, not
  a silent drop. The subset is applied before anything else reads genotypes, so
  allele frequencies, the `--maf` screen, LD weights and the covariance are all
  computed from the retained samples alone — a within-ancestry PCA needs no
  pre-subset callset. Retained samples keep their dataset order, and
  `samples.tsv` plus the score rows list exactly those samples. Indexed PLINK
  hard calls keep their physical row indices, so statistics and narrow-block
  covariance read the retained 2-bit calls directly instead of decoding every
  full-cohort row into a temporary matrix first.
* `--markers <N>` – Cap the variants the fit reads, as an evenly spaced
  subsample of whatever `--list` left. Every stage scales linearly in the
  variant count while the leading axes are estimated well before the last
  marker is read: on the benchmark cohort, a quarter of the markers ran 2.5×
  faster and still recovered every structured component (between-population
  variance 0.977 against 0.994 on the full set). The stride is exact integer
  arithmetic, not a random draw, so the same request selects the same markers
  everywhere. A thinned model can only project onto the markers it kept. Needs
  an indexed source (PLINK/PGEN) — a streamed VCF has no index to stride over —
  and pairs with `--bp_window` rather than `--sites_window`, since thinned
  variants are no longer genomic neighbours.
* `--maf <MAF>` – Retain only variants whose observed minor allele frequency in
  the fitting cohort is at least `MAF`. The threshold must be between `0` and
  `0.5`; filtering runs after `--list` and `--keep` selection and before LD
  weighting or PCA. Indexed PLINK/PGEN inputs are screened once and then reopened
  through the retained physical-marker selection, so repeated PCA passes keep
  the packed hard-call path instead of recomputing MAF on every traversal. MAF
  is counted directly from the packed 2-bit hard calls, including only the
  physical rows selected by `--keep`; the fitting cohort, not the full callset,
  therefore defines the observed MAF without materializing decoded genotypes.
* `--geno <RATE>` – Retain only variants whose observed missing-call fraction
  in the fitting cohort is at most `RATE`, matching PLINK's threshold
  convention. The value must be between `0` and `1`; `0` requires complete
  calls and `1` disables the screen. `--geno` is fused with `--maf` in the same
  packed 2-bit scan, so combining realistic microarray call-rate and frequency
  QC does not add another genome traversal. Like MAF, call rate is computed
  after `--list`, `--markers`, and `--keep`, and the exact retained marker list
  defines subsequent LD windows and PCA passes.
* `--allow-unconverged` – Emit the solver's best available model when it stops
  without converging: at its bounded pass budget, or sooner once its measured
  progress shows the requested boundary cannot resolve before that budget (see
  the eigensolver section below). This never labels the result converged:
  `hwe.json` and `hwe_summary.tsv` retain the measured residual, subspace
  change, boundary gap and `converged=false`. Use it when a requested boundary
  lies inside a nearly degenerate ancestry/noise cluster and a fixed component
  count is operationally required; without it, an unconverged fit is refused.
* `--ld` – Enable linkage disequilibrium flattening. When present, LD weights
  use a 500 kbp window unless `--sites_window <SITES>` (odd number of variants)
  or `--bp_window <BP>` (total genomic span in base pairs) is supplied. Without
  `--markers`, a PLINK or PGEN fit also keeps at most 100,000 evenly spaced
  markers. A streamed VCF/BCF cannot be thinned before it is read, so it keeps
  every variant it streams, or every `--list` variant it matches. Every window
  is clipped to its chromosome; site-count windows
  never borrow markers from the next chromosome. Base-pair windows additionally
  require nondecreasing positions within each chromosome run. The resulting
  weights are stored inside `hwe.json` so projections can apply the same
  normalization.

### `gnomon project`
`gnomon project` projects new samples into the PCA space defined by a previously
fit model. Place the model files generated by `gnomon fit` (especially
`hwe.json`) alongside the genotype data being projected; the command validates
that the variant subsets match before producing scores. The CLI writes a
self-contained `projection_scores.bin` that stores the numeric score matrix plus
the projected row IDs (`IID`s) in the same artifact, along with a
`projection_scores.metadata.json` sidecar that records matrix shape and layout.
Alignment diagnostics can be requested through the library API when
finer-grained monitoring is needed.

## Fitting workflow
1. `gnomon fit` standardizes loci with the Hardy–Weinberg scaler (mean
   centering and division by √(2p(1−p))).  Missing data stay `NaN` until
   standardization, so they do not bias allele statistics.
2. A dense covariance build is attempted when the implied Gram matrix fits in
   memory; otherwise the partial self-adjoint eigensolver incrementally updates
   covariance from streamed blocks.
3. `--ld` enables LD weighting with either the default 51-variant window, a
   user-provided odd-sized site window, or a base-pair span that adapts to
   variant density. Windows are truncated at dataset and chromosome edges and
   validated against the exact post-filter marker stream before use. The
   resulting per-variant weights are saved inside the serialized model.
4. `--maf` and `--geno` can remove rare, monomorphic, or poorly called variants
   before LD weighting and PCA; saved model keys, scalers, loadings, and
   optional LD weights all use the jointly filtered variant set.
5. `--keep` narrows the sample rows the stream yields, so every statistic above
   is estimated within the subset. The packed hard-call fast path cannot express
   a row subset. PLINK hard calls retain their physical row indices through the
   packed statistics and covariance kernels, avoiding a decoded full-cohort
   gather; streamed VCF/BCF sources use the generic row-gather path.
6. Requested components beyond the intrinsic rank are clamped, and the driver
   reports the retained dimensionality.

### Outputs written next to the genotype source
* **`hwe.json`** – Serialized `HwePcaModel` capturing the scaler, eigenvalues,
  sample/variant counts, component loadings, optional LD weights, and the
  `(chromosome, position)` keys that identify the variant subset when filtering
  was enabled. It deliberately excludes cohort-sized sample coordinates; those
  are stored once, with row IDs, in `hwe_scores.bin` rather than duplicated as
  much larger decimal JSON matrices.
* **`samples.tsv`** – Tab-delimited manifest built from `.fam` content or the
  VCF/BCF sample list (`FID`, `IID`, `PAT`, `MAT`, `SEX`, `PHENOTYPE`), listing
  the samples the fit actually used — the whole cohort unless `--keep` narrowed
  it.
* **`hwe_summary.tsv`** – Key/value table with overall counts, per-component
  explained variance, and explained-variance ratios.
* **`hwe_scores.bin`** (+ **`hwe_scores.metadata.json`**) – The fitted samples'
  own PC scores, in the same self-contained matrix container `gnomon project`
  writes (column-major `f64`, `IID` row IDs embedded, `kind = "scores"`), so the
  same readers work on both. Row order matches `samples.tsv`.

  Emitting them here saves a second whole-genome pass, which is the actual
  argument: recovering the training samples' coordinates by re-running
  `gnomon project` over the same genotypes costs another full traversal to
  recompute numbers the fit already holds. On the benchmark cohort the two
  agree to six decimal places on the structured components (|corr| = 1.000000,
  scale 1.0000), so this is a saved pass rather than a different quantity —
  the spectral shrinkage discussed below arises from *missing* loci and
  out-of-sample projection, not from projecting a complete training cohort.

## The eigensolver, and why it is block-structured

The fit's expensive resource is **a traversal of the genotype data**, not
arithmetic. Everything in the solver follows from that.

`gnomon` never forms the sample covariance `C = XXᵀ/(n−1)` in the streaming
regime; it exposes only the product `Q ↦ CQ`, evaluated as `X(XᵀQ)` one variant
tile at a time. Crucially that product costs **one pass whether `Q` has one
column or fifty** — the pass is spent reading and standardizing genotypes, and
the extra columns ride along inside the same tile.

A vector-at-a-time Krylov method throws that away. It grows its search space by
one column per operator application, so a 64-dimensional Krylov space costs 64
whole-genome passes before restarts, spending the expensive resource to buy the
cheap one. `map/blocklanczos.rs` replaces it with an adaptive randomized block
Lanczos: every pass advances every requested component at once.

* **Oversampling.** The block is `k + min(32, max(8, k/2))` columns wide. The
  guard band beyond `k` is what makes `θ_{k+1}` observable, and extra columns
  are paid for *inside* a pass while extra depth costs a whole new one.
* **The projection is free.** The block recurrence's own coefficients assemble
  `T = KᵀCK` directly, so Rayleigh–Ritz is an eigendecomposition of a few
  hundred rows — no genotype access at all.
* **Convergence is measured, not assumed.** The Lanczos identity
  `CK = KT + Q_{j+1}B_jE_jᵀ` gives every Ritz pair's exact residual
  `‖Cu − θu‖ = ‖B_j·s_tail‖` from the trailing block, again with no extra pass.
  That is what makes an adaptive pass count possible instead of a fixed
  iteration count chosen for someone else's dataset.
* **The loadings cost no pass either.** Each pass computes `XᵀQ_j` on the way
  to `CQ_j`, and every Ritz vector is a combination of blocks already applied,
  so the variant cross-products `XᵀU = Σ_j (XᵀQ_j)·s_j` that the loadings,
  eigenvalues and scores are refined from are assembled from products the
  solver already holds. They are kept under the Krylov basis budget; a basis
  whose products would not fit falls back to one more traversal.
* **Subspaces, not vectors.** Near-degenerate components rotate freely within
  their eigenspace, so a per-vector test can report failure forever on a
  subspace that has in fact settled. A second criterion tracks the mean
  explained variance between successive top-`k` bases, which is invariant to
  that rotation. Both must pass.
* **A clustered `k/k+1` boundary is reported, not papered over.** When
  `θ_k ≈ θ_{k+1}` the requested truncation cuts through a near-degenerate
  eigenspace and "the first exactly `k` PCs" is not a well-conditioned object.
  The solver widens its guard band rather than spending more passes on a
  distinction the data does not support.
* **A boundary no pass can resolve is recognised before the ceiling.** Ask for
  more components than the data has structure and the boundary lands in the
  noise bulk, whose top eigenvalues sit a few parts in ten thousand apart. The
  axes above it reach roundoff within about eight passes; the boundary's own
  residual then falls by roughly a seventh a pass, the Chebyshev rate for a gap
  that narrow, and would need some eighty passes to reach the tolerance. Once
  the gap is below `cluster_gap`, the axes above the cluster are at the residual
  floor, and the boundary's measured rate — granted four times over — cannot
  reach the tolerance in the passes left, the solver stops and reports exactly
  what the ceiling would have: `converged=false`, the residual, subspace change
  and gap. Without `--allow-unconverged` the fit is refused with the same
  message, sooner. A cluster that is resolving shows it in its rate, so a fit
  that converges runs exactly as before.

### What this costs, measured

On a synthetic cohort with five populations at Fst 0.02 — 250,000 samples by
20,000 variants, PLINK, one machine and four pinned AMD EPYC Milan cores — the
current complete fits measure:

| request | wall clock | peak RSS | passes | converged | worst residual | gap at the boundary |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `--components 4` | **31.75 s** | 3.50 GB | 5 | **yes** | 2.2e-9 | 0.990 |
| `--components 20 --allow-unconverged` | **79.14 s** | 4.45 GB | 8 | no | 3.4e-2 | 0.0014 |

At k=4 the boundary between PC4 and PC5 is a 99% gap — a clean separation — and
the solve converges to nine digits. At k=20 the sixteen requested axes past the
structure are degenerate noise with near-identical eigenvalues, the boundary gap
collapses to 0.0014, and no amount of iteration can individually resolve them.
That is a property of the question, not a defect in the answer, which is why the
fit requires `--allow-unconverged` for this fixed-width best-effort model and
records `converged=false`, the residual, subspace change and boundary gap in
both model and summary. The leading four axes remain exact as a subspace: their
canonical correlations against the strict four-component fit are
1.000000000000 on all four axes.

### Stopping where no pass can help

A fit that asks for more components than the data has structure used to spend
the whole pass ceiling before reporting `converged=false`. With the boundary
stop, measured at `e7865ee0` against the same tree plus the stop, 8 threads
pinned to one NUMA node of a shared cluster node:

| cohort | request | passes | wall clock | peak RSS | worst residual | structured axes vs converged reference |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 50k × 20k, 5 populations | `--components 4` | 5 → 5 | 16.1 s → 15.9 s | 0.73 GB | 2.9e-8 | 5.8e-8, output unchanged |
| 50k × 20k, 5 populations | `--components 10 --allow-unconverged` | 32 → 16 | 109.1 s → **56.8 s** | 1.03 → 0.88 GB | 2.3e-3 → 1.6e-2 | 1.2e-13 → 1.8e-13 |
| 50k × 20k, 5 populations | `--components 20 --allow-unconverged` | 32 → 14 | 125.2 s → **53.5 s** | 1.26 → 0.96 GB | 1.8e-3 → 1.9e-2 | 2.0e-13 → 2.1e-13 |
| 100k × 20k, 5 populations | `--components 10 --allow-unconverged` | 32 → 15 | 191.6 s → **88.3 s** | 1.94 → 1.64 GB | 1.8e-3 → 1.5e-2 | 6.0e-13 → 6.2e-13 |
| 1000 Genomes on GSA, 3,200 × 562,259 | `--components 10 --allow-unconverged` | 19 → 19 | 84.4 s → 83.4 s | 2.08 GB | 7.4e-7, converged | 7.2e-15, output unchanged |
| 1000 Genomes on GSA, 3,200 × 562,259 | `--components 20 --allow-unconverged` | 19 → 19 | 83.7 s → 89.5 s | 3.15 GB | 9.9e-7, converged | 1.0e-14, output unchanged |

The four structured axes agree with a converged reference to roundoff either
way, and their canonical correlations with it are 1.000000000000. What the stop
gives up is refinement of the axes inside the bulk: their residual is about
1.5e-2 instead of about 2e-3, and the top-`k` subspace change about 1e-2 instead
of 1e-4. Neither version resolves those axes individually, and both are
recorded as unconverged. Without `--allow-unconverged`, the 50k × 20k,
`--components 10` fit is refused after 55.8 s instead of 109.3 s, with the same
message. A fit the stop does not fire on is untouched byte for byte. That
covers every fit in the golden default, edge and medium sets that converges or
stops at a `--max-passes` ceiling; the 1000 Genomes fits at `k = 20`, whose
boundary gap is 4.4e-3, are among them.

At one thread, measured at `1d127cdc` against the same tree plus the stop, on one
core of other nodes of the same cluster:

| cohort | request | passes | wall clock | peak RSS | structured axes vs converged reference |
| --- | --- | ---: | ---: | ---: | ---: |
| tiny, 5 populations | `--components 4` | 6 → 6 | 1.34 s → 1.24 s | 0.12 GB | output unchanged |
| 50k × 20k, 5 populations | `--components 4` | 5 → 5 | 18.6 s → 17.6 s | 0.72 GB | output unchanged |
| 50k × 20k, 5 populations | `--components 10 --allow-unconverged` | 32 → 16 | 129.1 s → **63.2 s** | 1.21 → 1.05 GB | 1.8e-13 |
| 50k × 20k, 5 populations | `--components 20 --allow-unconverged` | 32 → 14 | 169.2 s → **71.7 s** | 1.44 → 1.14 GB | 2.2e-13 |
| 100k × 20k, 5 populations | `--components 4` | 5 → 5 | 38.0 s → 34.6 s | 1.42 GB | output unchanged |
| 1000 Genomes on GSA, 3,200 × 562,259 | `--components 20 --allow-unconverged` | 19 | 189.5 s | 3.24 GB | stop build only; the stop does not fire |

At 250k × 20k and eight threads, `--components 10 --allow-unconverged` stops at
16 of 32 passes, with peak RSS 4.63 → 4.01 GB and the structured axes 5.9e-14
from the unstopped fit. The two runs shared a node whose time per pass rose from
30 s to 50 s between them, so their solve times (960 s and 803 s) do not isolate
the effect of the stop.

The 250k × 20k `--components 20 --allow-unconverged` fit ran with the stop
first, on the same eight cores: 32 → 15 passes, 1,234 s → **532 s**, peak RSS
6.03 → 4.93 GB, with the structured axes 4.1e-14 from the unstopped fit.

On a 4-core machine with a 4 GiB memory limit (a Slurm cgroup), run without
`--threads` as a user would run it:

| cohort | request | passes | wall clock | peak RSS | output |
| --- | --- | ---: | ---: | ---: | --- |
| tiny, 5 populations | `--components 4` | 6 → 6 | 1.9 s → 1.4 s | 0.13 GB | unchanged |
| 50k × 20k, 5 populations | `--components 10 --allow-unconverged` | 32 → 16 | 124.1 s → **70.9 s** | 0.91 → 0.76 GB | structured axes 2.5e-14 from the unstopped fit |
| 1000 Genomes on GSA, 3,200 × 562,259 | `--components 20 --allow-unconverged` | 19 → 19 | 99.8 s → 89.1 s | 1.19 GB | unchanged |

Under that limit the 1000 Genomes fit's solver products do not fit the Krylov
budget, so it keeps its loadings pass. Two runs of the unstopped build produced
identical bytes for all three fits.

### Current PLINK2 comparison

The local PLINK reader maps the `.bed` payload once, decodes selected variant
columns in parallel, and applies HWE standardization during that decode. It does
not turn an evenly spaced `--markers` selection into thousands of tiny `pread`
calls or make a second memory pass over each decoded tile. Allele moments are
counted directly from the packed 2-bit hard calls, so the statistics stage does
not inflate a 1.2 GB BED payload into 40 GB of temporary `f64` genotypes. A
512-variant streaming tile is the measured throughput/memory knee for this
workload; the adaptive memory plan can still reduce it for larger cohorts.

The following runs used the same five-population PLINK1 microarray cohort, the
same physical marker lists, four pinned AMD EPYC Milan cores, warm page cache,
and four requested PCs. PLINK2 was v2.0.0-a.7.4LM AVX2 AMD and ran `--pca
approx allele-wts 4 --threads 4`; gnomon ran `fit --components 4 --threads 4
--markers N`. The MAF row instead applies `--maf 0.05` to the complete
20,000-marker input in both programs.

| samples × markers | gnomon wall | PLINK2 wall | gnomon peak RSS | PLINK2 peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 250,000 × 10,000 | **16.76 s** | 25.10 s | **3.50 GB** | 6.05 GB |
| 250,000 × 20,000 | **31.75 s** | 48.10 s | **3.50 GB** | 6.06 GB |
| 250,000 × 20,000 (2% missing calls) | **32.36 s** | 48.35 s | **3.50 GB** | 6.06 GB |
| 250,000 × 19,000 (`--geno 0.05`; mixed 2%/20% missing) | **31.15 s** | 46.09 s | **3.50 GB** | 6.06 GB |
| 250,000 × 19,751 (`--maf 0.05`) | **32.46 s** | 47.77 s | **3.50 GB** | 6.06 GB |
| 100,000 kept from 250,000 × 20,000 | **17.59 s** | 20.84 s | **2.24 GB** | 2.44 GB |
| 500,000 × 10,000 | **37.91 s** | 57.37 s | **5.80 GB** | 12.09 GB |

That is a 33.2% wall-clock lead at 10,000 markers and 34.0% at 20,000, while
using 42% less peak memory in both 250,000-sample runs. At 500,000 samples the
lead is 33.9% with 52.0% less memory. The comparison is shape-matched:
PLINK2's `--extract` list contains exactly the variants selected by gnomon's
deterministic stride in the 10,000-marker run; the 20,000-marker run uses the
complete dataset in both programs.

Packed MAF screening is 21.0% faster than gnomon's previous decoded screen
(41.11 s) and preserves byte-identical model and score artifacts. The matched
MAF-filtered run is 32.0% faster than PLINK2 while using 42% less peak memory;
both retain the same 19,751 physical markers.

Missing calls do not turn into a hidden slow path or a different PCA. The 2%
row above deterministically marks exactly 5,000 of 250,000 calls missing at
every variant while preserving the complete panel's planted structure
(`scripts/bench_fit_solver.py generate --missing-rate 0.02`). Gnomon is 33.1%
faster with 42% less memory; all four canonical correlations with PLINK2 are
0.999999999999, and both programs report identical between-population variance
shares on every axis (0.99367–0.99392).

Call-rate QC is likewise fused into the packed screen rather than paid once per
solver pass. The `--geno` row starts from the complete panel and uses
`bench_fit_solver.py inject-missing` with `--missing-rate 0.02`,
`--high-missing-variants 1000`, and `--high-missing-rate 0.20` to distribute exactly
1,000 bad markers across the genome. With `--geno 0.05`, gnomon and PLINK2 both
remove those 1,000 markers and retain the other 19,000 in exactly the same
physical order. Gnomon is 32.4% faster with 42% less memory; all four axis and
canonical correlations are 0.999999999999, and their planted-population
variance shares agree to the eighth decimal place (0.99330–0.99362).
Adding `--maf 0.05` to that same run does not add a screen pass: both programs
retain the same 18,764 markers in the same order, all four canonical
correlations remain 0.999999999999, and gnomon completes in 31.12 s versus
45.87 s for PLINK2.

The matched `--keep` row retains the same deterministic 100,000 sample IDs in
both programs. Direct packed-row selection cut gnomon's previous decoded-gather
path from 38.84 s to 17.59 s (54.7%) and from 3.75 GB to 2.24 GB while producing
byte-identical model and score artifacts. It is 15.6% faster than PLINK2 with
8.1% less peak memory, and the four score-subspace canonical correlations are
1.000000000000. With `--maf 0.05` added, both programs retained the same 19,748
within-cohort markers; gnomon completed in 20.08 s versus PLINK2's 20.60 s and
again matched all four score axes at 1.000000000000.

The realistic 20-PC request widens the gap. Gnomon completed the full
250,000 × 20,000 fit, including refinement, loadings and all model/score
artifacts, in 79.14 s at 4.45 GB. The matched PLINK2 run did not complete its
PCA within a 180 s bound and had reached only random projection 17/21, with no
eigenvector artifact written. Using that timeout only as a lower bound, gnomon
is at least 56.0% faster for this request; no PLINK2 memory claim is made for an
incomplete run.

The 500,000-row case is explicitly a scaling stress test: it duplicates the
250,000 statistically generated sample rows, preserving the same packed-call
distribution while doubling the row dimension. It is not presented as a
second biological simulation. The paired gnomon scores agreed within
`4.4e-10`, and the gnomon/PLINK2 canonical correlations remained 1.000000 on
all four axes. With eight pinned cores, gnomon completed this case in 30.44 s
at 5.98 GB, versus PLINK2's 51.49 s at 23.96 GB.

Performance was not accepted as a substitute for the answer. Across the four
structured axes, the canonical correlations between gnomon and PLINK2 scores
were 1.000000 in the 20,000-marker run; gnomon's leading-axis
between-population variance shares were 0.994, 0.994, 0.994, and 0.994.
Changing the tile width from 2,048 to 1,024 to 512 produced bit-identical score
artifacts, and direct packed-statistics counting produced model and score
artifacts that were bit-identical to the decoded-statistics implementation.
The MAF-filtered score comparison likewise produced canonical correlations of
1.000000 on all four axes.

Four cores match the PLINK2 comparison above. More cores no longer cost a
small-component solve anything: since `b8841cec` the covariance products run on
fixed row leaves with no barrier between threads. On `large_k4` (250,000 ×
20,000, k=4, one 32-core EPYC 7763 allocation) the fit took 13.3 s at 16 threads
and 13.9 s at 32, where the barrier-bound solver took 18.9 s and 154.1 s.

"Structure recovered" is the check that the numbers mean anything. Five
populations span a four-dimensional space of means, so exactly four components
should carry between-population variance and the rest should be noise; the
figure is that share on the leading axes. Residuals and subspace deltas are
statements about the operator the solver was handed, not about whether the
answer is the right one, so the fit is also checked against the structure the
cohort was built with (`bench_fit_solver.py verify`).

A caveat worth stating: these genotypes sat in page cache, so the comparison is
dominated by decode and arithmetic rather than storage latency. Cold object
storage and PGEN/VCF inputs need their own measurements; this table makes no
claim about them.

### The starting vector is not the all-ones vector

Every variant column is centered on its own observed allele frequency, with
missing calls landing on zero, so each standardized column sums to zero:
`Xᵀ1 = 0`, and therefore `C·1 = 0` **exactly**. The all-ones vector is precisely
the sample covariance's null direction, and a Krylov sequence seeded with it —
`1, C1, C²1, …` — is identically zero in exact arithmetic. Seeding with it
"works" only because rounding leaves a little noise for the first application to
amplify into a generic direction, which is a property of the floating-point
error rather than of the algorithm. The start block is drawn from a fixed
pseudo-random stream with the constant direction projected out, and the stream is
fixed so that a fit reproduces bit-for-bit across runs and machines.

### Memory follows the same logic

A tile is `n_samples × tile_width` f64s and the operator holds two of them, so a
fixed tile width that is reasonable at a thousand samples reserves gigabytes at
half a million. Tile width is therefore derived from the sample count and the
machine's memory, and deliberately *not* from measured throughput — two machines
with the same memory must tile identically, or the same data would produce
different arithmetic.

## High-dimensionality projection
Because the biobank and the single individual are standardized on the same reference, the input feature scaling is consistent. However, projecting onto the fitted biobank principal components inherently subjects new data to spectral shrinkage, placing them in a compressed variance space relative to the training set. Attempting to reverse this requires estimating the genome's effective independent dimensionality, which is obscured by linkage disequilibrium. Consequently, gnomon avoids de-shrinkage or OADP/AP rotations, preferring a stable projection over the risk of inaccurate coordinate re-inflation and needless perturbation.

## Missing SNVs
If we project onto a unit vector made only from the SNVs we have, missing SNVs don’t contribute signal or variance; their loading mass is subtracted from the denominator and the axis is renormalized. The projection for each PC is then computed only from those overlapping SNVs, producing the same weighted sum we would obtain from mean-imputed, standardized genotypes before the final rescaling. We take the standardized genotype values at the overlapping loci, weight them by the corresponding loadings, and sum. Because loadings for missing loci were effectively dropped, we renormalize the axis using the amount of loading mass that remains—i.e., divide by the Euclidean norm of the retained loadings—so we're still projecting onto a unit-length axis defined solely by the SNVs we actually have. Drop missing SNVs, rebuild the axis from the overlap, rescale it to unit length.

## Alignment diagnostics
When projections encounter missing loci, gnomon keeps track of how much loading mass each sample retained on every component. That information can be exported via the library API by enabling `ProjectionOptions::return_alignment`, which produces `projection_alignment.bin` plus `projection_alignment.metadata.json`. Each value represents the squared-norm scaling factor applied to keep the projection axis at unit length. Values near `1` indicate a complete overlap with the training loci, while smaller values highlight PCs that lost signal because of missing variants. The CLI defaults to score output only, but the saved model always contains enough metadata to reconstruct alignment diagnostics when requested.

## Projection workflow
* `gnomon project` loads `hwe.json`, reconstructs any stored variant subset,
  aligns projection variants by stored allele-aware keys, and fails only when
  there is no overlap with the model variant set.
* Projections reuse the block streaming interface, so missing loci are handled
  via the renormalization described above and the resulting scores share the
  training scale.
* The CLI writes `projection_scores.bin` with a JSON sidecar describing matrix
  shape and layout. The score binary embeds the projected sample IDs so the row
  labels travel with the matrix. Callers that enable alignment diagnostics
  through the library API also receive `projection_alignment.bin` with matching
  metadata.

## Progress and validation
The driver reports progress for allele statistics, optional LD weighting, Gram
matrix construction, and loading computation.  Stored models include
eigenvalues, explained-variance ratios, sample counts, variant counts (after
filtering), optional LD weights, and any variant key list.  Reloading verifies
that saved metadata are internally consistent and that projection datasets
match the recorded variant subset before computation begins, preventing
mismatched cohorts from being processed.

## Correctability diagnostic

`gnomon-map correctability DESIGN_JSON` evaluates a specified ancestry axis
against each available marker class. The input uses LD-adjusted effective
marker counts, not raw SNP counts:

```json
{
  "sample_size": 400000,
  "subgroup_size": 1000,
  "fitted_pcs": 40,
  "marker_classes": [
    {"name": "common", "effective_independent_markers": 100000, "differentiation": 0.0001, "theoretical_pc_rank": 12},
    {"name": "rare", "effective_independent_markers": 1000000, "differentiation": 0.001, "theoretical_pc_rank": 3}
  ],
  "application": {
    "susceptibility": 0.00001,
    "expected_pgs_variants": 10000,
    "effect_sd": 0.1,
    "directional_amplification": 2.0,
    "count_inflation": 0.0,
    "confounder": 0.5,
    "critical_signal": 3.85
  }
}
```

For each class the report gives the rank-one spike `4 F m_eff`, with `F` on
the Hudson `F_ST` scale, the BBP edge `sqrt(n / M_eff)`, and the
Johnstone–Paul sample-PC overlap proxy above the edge. Sample-PC overlap is
reported independently of whether that PC is among the requested covariates;
the removed-axis and residual-axis fractions record what the requested PC set
actually corrects. The report also gives the minimum sufficient PC count (or
that no PC count suffices),
residual susceptibility `H'`, predicted standardized bias, and the critical
confounder magnitude. It also reports the differentiation-matched
frequency weights and total information `sum M_c F_c^2`. The cross-class
quantity is labeled an information index rather than a combined BBP overlap:
the single-spike theorem does not identify an overlap curve for heterogeneous
marker classes. These are model-based
design quantities: linkage, uncertainty in `F_c`, or violation of the rank-one
spike model must be reflected in the effective marker counts or external error
bounds rather than hidden by the calculator.

The corresponding Lean proof graph can be checked on MSI without rebuilding
the full `Calibrator` library:

```bash
sbatch scripts/msi-pc-proof-fast.sbatch
```

Pass a leaf target such as `Calibrator.PCCorrectability.Phase` while iterating.
The job deliberately preserves Lake's incremental state; using `lake -R` here
forces a project-wide replay and defeats the fast path.

`Threshold`, `Geometry`, `Design`, and `Nonidentifiability` separate scalar
definitions, sharp design criteria, and the two proof families; `Overlap`,
`Phase`, and `Frequency` are independent sibling leaves. Target the file being
edited and reserve the umbrella target for final integration, avoiding
recompilation of unrelated layers.
