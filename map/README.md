## Overview
The `map` crate implements gnomon's Hardy–Weinberg–standardized PCA pipeline.
It serves the `gnomon fit` and `gnomon project` subcommands plus library APIs
that expose model fitting, serialization, and projection.  The codebase assumes
datasets that range from a handful of samples to biobank scale by streaming
genotypes in fixed-width blocks, abstracting over PLINK `.bed` trios and
VCF/BCF sources (local or remote), and picking a dense or iterative eigensolver
based on the Gram matrix size implied by the requested components.

## CLI entry points
| Command | Purpose | Required input | Primary outputs |
| --- | --- | --- | --- |
| `gnomon fit --components <N> [--list PATH] [--ld] <GENOTYPE_PATH>` | Train a Hardy–Weinberg PCA model | PLINK `.bed`/`.bim`/`.fam` trio, a single VCF/BCF file, or a directory / remote prefix containing per-chromosome VCF/BCF shards | `hwe.json`, `samples.tsv`, `hwe.summary.tsv` |
| `gnomon project <GENOTYPE_PATH>` | Project samples with an existing `hwe.json` model located next to the genotype data | Same genotype source that was used for training, or one that matches the model's stored variant subset | `projection.scores.tsv` (alignment output is available only when enabled via the library API) |

Both commands surface sample and variant counts, resolved source paths, and
stage-aware progress while running.

## Input data
**PLINK `.bed`** – Provide the `.bed` path (local or remote) and matching `.bim`
and `.fam` files.  Variant ordering follows the `.bim`; if `--list` is supplied
the loader collects the requested `(chromosome, position)` indices and records
missing targets.  `.fam` must supply the six canonical columns.

**VCF / BCF** – Accepts single files, shard directories (natural ordering), or
remote URIs ending in `.bcf`, `.vcf`, `.vcf.gz`, or `.vcf.bgz`.  Sample names
must be present and consistent.  FORMAT decoding prefers dosage (`DS`) and
falls back to genotype (`GT`).  Filtering with `--list` happens during
streaming, and the final `SelectionOutcome` summarizes matches and misses.

## Fitting workflow
1. `gnomon fit` standardizes loci with the Hardy–Weinberg scaler (mean
   centering and division by √(2p(1−p))).  Missing data stay `NaN` until
   standardization, so they do not bias allele statistics.
2. A dense covariance build is attempted when the implied Gram matrix fits in
   memory; otherwise the partial self-adjoint eigensolver incrementally updates
   covariance from streamed blocks.
3. `--ld` enables LD weighting with the default (`window=51`, `ridge=1e-3`)
   configuration, truncated near dataset edges and validated before use.  The
   resulting per-variant weights are saved inside the serialized model.
4. Requested components beyond the intrinsic rank are clamped, and the driver
   reports the retained dimensionality.

### Outputs written next to the genotype source
* **`hwe.json`** – Serialized `HwePcaModel` capturing the scaler, eigenvalues,
  sample/variant counts, component loadings, optional LD weights, and the
  `(chromosome, position)` keys that identify the variant subset when filtering
  was enabled.
* **`samples.tsv`** – Tab-delimited manifest built from `.fam` content or the
  VCF/BCF sample list (`FID`, `IID`, `PAT`, `MAT`, `SEX`, `PHENOTYPE`).
* **`hwe.summary.tsv`** – Key/value table with overall counts, per-component
  explained variance, and explained-variance ratios.

## High-dimensionality projection
Because the biobank and the single individual are standardized on the same reference, and placed on the same per-axis scale, the directional geometry is preserved. Fitting on the projected biobank means residual magnitude shrinkage is just a shared, axis-wise rescaling, so both a single new datapoint and the biobank data inhabit the same commensurately shrunken space and distances. Consequently, de-shrinkage or OADP/AP rotations would merely re-inflate coordinates and risk needless perturbation.

## Missing SNVs
If we project onto a unit vector made only from the SNVs we have, missing SNVs don’t contribute signal or variance; their loading mass is subtracted from the denominator and the axis is renormalized. The projection for each PC is then computed only from those overlapping SNVs (i.e. the same as mean imputation after normalization). We take the standardized genotype values at the overlapping loci, weight them by the corresponding loadings, and sum. Because loadings for missing loci were effectively dropped, we renormalize the axis using the amount of loading mass that remains—i.e., divide by the Euclidean norm of the retained loadings—so we're still projecting onto a unit-length axis defined solely by the SNVs we actually have. Drop missing SNVs, rebuild the axis from the overlap, rescale it to unit length.

## Projection workflow
* `gnomon project` loads `hwe.json`, reconstructs any stored variant subset, and
  verifies that the projection dataset supplies every required locus.  A
  mismatch fails fast instead of silently dropping variants.
* Projections reuse the block streaming interface, so missing loci are handled
  via the renormalization described above and the resulting scores share the
  training scale.
* The CLI writes `projection.scores.tsv`; callers that enable alignment
  diagnostics through the library API also receive `projection.alignment.tsv`.

## Progress and validation
The driver reports progress for allele statistics, optional LD weighting, Gram
matrix construction, and loading computation.  Stored models include
eigenvalues, explained-variance ratios, sample counts, variant counts (after
filtering), optional LD weights, and any variant key list.  Reloading verifies
that saved metadata are internally consistent and that projection datasets
match the recorded variant subset before computation begins, preventing
mismatched cohorts from being processed.
