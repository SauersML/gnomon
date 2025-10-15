## Overview
The `map` crate implements gnomon's Hardy–Weinberg–standardized principal
component analysis (PCA) pipeline.  It powers the `gnomon fit` and
`gnomon project` CLI subcommands as well as the library APIs that expose model
fitting, serialization, and sample projection.  The implementation is designed
for cohorts that range from a few samples to biobank scale:

* Genotypes are streamed in blocks (default 2,048 variants) so that the full
  matrix never needs to materialize in memory, regardless of input size.
* Multiple readers abstract over local PLINK `.bed` trios and VCF/BCF streams
  (optionally BGZF-compressed), including remote objects served from `gs://`
  or `http(s)://` URIs.
* Remote variant shards are opportunistically spooled to a temporary directory
  so that repeated passes (for scaling, LD weighting, and Gram accumulation)
  do not re-download the same bytes.
* The fitter chooses between a dense covariance build and a partial eigenvalue
  solver based on a configurable Gram-matrix memory budget, allowing large
  problems to run within bounded memory.

## CLI entry points
| Command | Purpose | Required input | Primary outputs |
| --- | --- | --- | --- |
| `gnomon fit --components <N> [--list PATH] [--ld] <GENOTYPE_PATH>` | Train a Hardy–Weinberg PCA model | PLINK `.bed`/`.bim`/`.fam` trio, a single VCF/BCF file, or a directory / remote prefix containing per-chromosome VCF/BCF shards | `hwe.json`, `samples.tsv`, `hwe.summary.tsv` |
| `gnomon project <GENOTYPE_PATH>` | Project samples with an existing `hwe.json` model located next to the genotype data | Same genotype source that was used for training, or one that matches the model's stored variant subset | `projection.scores.tsv` (alignment output is available only when enabled via the library API) |

Both commands print dataset metadata (sample count, detected variant count,
resolved source path) and stage-aware progress indicators while they run.

## Input data requirements and behaviour
### PLINK `.bed`
* The loader accepts a `.bed` path (local or remote). Companion `.bim` and
  `.fam` files must exist at the same stem; `.fam` must provide the six standard
  whitespace-delimited fields.  Empty `.bim` or `.fam` files trigger immediate
  errors.
* Remote paths beginning with `gs://`, `http://`, or `https://` stream data via
  the shared I/O layer.  Local `.bed` files are memory-mapped when possible for
  fast random access.
* Variant ordering is taken from the `.bim` file.  When `--list` is supplied,
  the loader collects 0-based indices for the requested `(chromosome, position)`
  keys and records which targets were missing.

### VCF / BCF streams
* The entry path may be a single file, a directory of shard files (sorted with
  natural ordering), or a remote URI.  Supported extensions include `.bcf`,
  `.vcf`, `.vcf.gz`, and `.vcf.bgz`.
* Sample names must be present and identical across all shards; they are reused
  as both `FID` and `IID` in the generated `samples.tsv` manifest.  Records
  lacking per-sample data fail the run.
* FORMAT decoding prefers dosage (`DS`) values when available and falls back to
  the genotype (`GT`) field.  A `GT` column (or usable `DS`) is therefore
  required.
* Remote shards are streamed into a local spool directory on first use.  The
  cached copy is reused for subsequent passes so that LD weighting, Gram
  accumulation, and projection can re-read the same data without re-fetching it.
* When `--list` is provided, filtering happens during streaming.  Final match
  statistics are emitted after processing completes via the recorded
  `SelectionOutcome`.

## Fitting workflow
1. `gnomon fit` normalizes genotypes with a Hardy–Weinberg scaler (mean-centering
   and dividing by √(2p(1−p))).  Missing data stay `NaN` until standardization,
   at which point they contribute zero to per-locus statistics.
2. The pipeline inspects `GNOMON_GRAM_BUDGET_BYTES` (default 8 GiB) to decide
   whether a dense Gram matrix can fit in memory.  Otherwise it switches to the
   partial self-adjoint eigensolver that incrementally updates covariance from
   streamed blocks.
3. If `--ld` is passed, LD weights are computed with the default configuration
   (`window=51`, `ridge=1e-3`, automatically truncated near dataset edges).  The
   fitter validates the parameters (window ≥ 1, positive ridge) and saves the
   per-variant weights inside the serialized model.
4. Requested components above the model's intrinsic rank are clamped.  The
   driver reports the retained dimensionality so callers can detect when fewer
   PCs than requested were available.

### Outputs written next to the genotype source
* **`hwe.json`** – Serialized `HwePcaModel` capturing the scaler, eigenvalues,
  sample/variant counts, component loadings, optional LD weights, and the
  `(chromosome, position)` keys that identify the variant subset when filtering
  was enabled.
* **`samples.tsv`** – Tab-delimited manifest constructed from the `.fam` file or
  VCF/BCF sample list (`FID`, `IID`, `PAT`, `MAT`, `SEX`, `PHENOTYPE`).
* **`hwe.summary.tsv`** – Key/value table with overall counts, per-component
  explained variance, and explained-variance ratios.

## Projection workflow
* `gnomon project` loads `hwe.json` from the same output directory used during
  training, reconstructs any stored variant subset, and verifies that the
  projection dataset supplies every required locus.  A mismatch results in a
  hard error instead of silently dropping variants.
* Projections stream data through the same block interface as fitting.  Missing
  loci are renormalized away (the projector keeps only the loadings observed in
  the projection data and rescales the axis to unit length), so the resulting
  scores remain on the same scale as the training cohort.
* By default the CLI only writes `projection.scores.tsv`.  The library API can
  request per-sample alignment diagnostics by enabling
  `ProjectionOptions::return_alignment`, in which case `projection.alignment.tsv`
  is also produced.

## Progress reporting and tuning knobs
* Stage-aware progress observers report allele statistic accumulation, optional
  LD weighting, Gram matrix construction, and loading computation.  For streamed
  VCF/BCF inputs, byte-level throughput is also surfaced; PLINK sources report
  variant counts only.
* Parallelism is delegated to Rayon and Faer.  Set `RAYON_NUM_THREADS` to
  control concurrency in shared environments.
* Advanced users who require different memory/performance trade-offs can adjust
  the compile-time constant `map::fit::DEFAULT_BLOCK_WIDTH`.

## Model contents and validation safeguards
* Stored models record per-component eigenvalues, explained-variance ratios,
  sample counts, variant counts (post-filtering), optional LD weights, and any
  variant key list used during fitting.
* When a model is reloaded, the driver checks that stored variant keys agree
  with the reported variant count and that projection datasets match the saved
  dimensionality before any computation starts.  These checks prevent projecting
  mismatched cohorts or accidentally mixing incompatible variant subsets.
