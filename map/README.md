## Overview
`map` contains the Hardy–Weinberg Equilibrium (HWE) principal component analysis pipeline that powers the `gnomon fit` and `gnomon project` subcommands.  It is designed to scale from small cohorts to biobank-scale datasets while keeping inputs and outputs compatible with the rest of the toolkit.  The module can open raw PLINK BED trios, BCF/VCF variant streams (optionally BGZF compressed), and remote URIs that the shared I/O layer knows how to read.  Every dataset is normalized onto the same reference frame so that training and projection runs are interoperable by construction.

## Command summary
| Subcommand | Required input | Key options | Primary outputs |
| --- | --- | --- | --- |
| `gnomon fit --components <N> [--list PATH] [--ld] <GENOTYPE_PATH>` | PLINK `.bed` file (with `.bim`/`.fam`) or directory/URL containing BCF/VCF variants | `--components` sets the number of PCs retained, `--list` restricts the SNV set, `--ld` turns on LD normalization | `hwe.json`, `samples.tsv`, `hwe.summary.tsv` |
| `gnomon project <GENOTYPE_PATH>` | Same genotype source used for fitting or a cohort standardized to the same reference | Automatically reuses variant subset stored in the model | `projection.scores.tsv`, optionally `projection.alignment.tsv` |

`GENOTYPE_PATH` can point at a local file or a remote object.  When a directory is provided, the loader scans for per-chromosome shards and streams them in sorted order.  The driver prints a summary of the resolved dataset (samples, approximate variants, source path) before any heavy computation begins.

## Input data handling
* **Format detection** – The loader inspects the suffix to decide between PLINK and variant-stream readers, and exposes a uniform `GenotypeDataset` interface.
* **Variant streaming** – Both PLINK and BCF/VCF readers provide a `VariantBlockSource` abstraction that surfaces blocks of standardized genotypes without materializing the full matrix in memory.  The default block size is `map::fit::DEFAULT_BLOCK_WIDTH` (2,048 variants), and blocks are processed in parallel using Rayon.
* **Sample metadata** – PLINK `.fam` fields are surfaced as structured records and re-exported in the generated `samples.tsv` manifest.  VCF/BCF sample names are required and enforced at read time.

## Variant list support
Passing `--list <PATH>` points the fitter at a whitespace-delimited file containing `chrom position` columns.  Comment lines and headers (`chrom`, `chr`, `pos`) are ignored; chromosome labels are normalized (e.g. `chr20` → `20`).  Variant lists can be local or remote URLs.  During fitting the loader reports how many requested variants were matched or missing.  For streaming variant datasets, the filter is applied lazily and the final match statistics are emitted after the projection source closes.

## Linkage disequilibrium normalization
Enabling `--ld` activates local LD weighting.  The configurable parameters are stored in `LdConfig` and default to:
* odd sliding window of 51 variants (automatically truncated near dataset edges),
* ridge stabilization term of `1e-3` applied to the local covariance estimate.

The LD weights are included in `hwe.json` for reproducibility.  Any invalid parameter (window < 1, non-positive ridge) aborts with an informative error before computation begins.

## Outputs
All artifacts are written next to the resolved genotype input.  Remote inputs are mirrored into local filenames derived from the remote stem.

* **`hwe.json`** – Serialized `HwePcaModel` containing sample/variant counts, the retained loading vectors, eigenvalues, LD weights, and (if applicable) the variant subset keys.
* **`samples.tsv`** – Sample manifest (FID, IID, parents, sex, phenotype) in tab-delimited form.
* **`hwe.summary.tsv`** – Tabular metrics including `n_samples`, `n_variants`, per-PC explained variance, and explained variance ratios.
* **`projection.scores.tsv`** – PC scores for each projected sample when running `project`.
* **`projection.alignment.tsv`** – Optional alignment statistics produced when the projector detects components that require sign alignment.

## Projection workflow
When projecting samples, the runner loads `hwe.json`, resolves any stored variant subset, and verifies that the genotype source contains all expected variants.  Missing loci are treated as fatal, preventing silent drift between training and inference cohorts.  Projection work shares the streaming infrastructure used during fitting and reports progress as blocks are consumed.  Scores are emitted in sample order, matching the manifest.

## Progress reporting
Both fitting and projection expose stage-aware progress observers.  For PLINK sources only the variant counter is available; variant streams additionally surface byte-level throughput.  The console output uses concise ASCII status lines so runs can be monitored in logs or terminals without specialized UI support.

## Performance knobs
* **Gram matrix memory budget** – Dense covariance is attempted when `n_samples^2 * 8 bytes` fits into the budget controlled by the `GNOMON_GRAM_BUDGET_BYTES` environment variable (default 8 GiB).  Larger problems fall back to the partial eigen solver that streams blocks and never materializes the full matrix.
* **Threading** – The underlying linear algebra is provided by `faer` and uses Rayon for parallel stages.  Thread counts follow the global Rayon configuration; set `RAYON_NUM_THREADS` to pin throughput on shared environments.
* **Block width** – Advanced users can modify `map::fit::DEFAULT_BLOCK_WIDTH` at compile time if they need different memory/performance trade-offs.

## Statistics exported with the model
* Number of processed samples and variants (after any variant filtering).
* Eigenvalues (`explained_variance_*`) and normalized ratios (`explained_variance_ratio_*`).
* Optional LD weighting parameters (`window`, `ridge`, and per-variant weights).
* Summary of variant matches vs. misses when a list is provided.
* Confirmation of the effective dimensionality (requested vs. retained PCs).

## Algorithmic notes
### High-dimensionality projection
Because the biobank and the single individual are standardized on the same reference, and placed on the same per-axis scale, the directional geometry is preserved. Fitting on the projected biobank means residual magnitude shrinkage is just a shared, axis-wise rescaling, so both a single new datapoint and the biobank data inhabit the same commensurately shrunken space and distances. Consequently, de-shrinkage or OADP/AP rotations would merely re-inflate coordinates and risk needless perturbation.
### Missing SNVs
If we project onto a unit vector made only from the SNVs we have, missing SNVs don’t contribute signal or variance; their loading mass is subtracted from the denominator and the axis is renormalized. The projection for each PC is then computed only from those overlapping SNVs (i.e. the same as mean imputation after normalization). We take the standardized genotype values at the overlapping loci, weight them by the corresponding loadings, and sum. Because loadings for missing loci were effectively dropped, we renormalize the axis using the amount of loading mass that remains—i.e., divide by the Euclidean norm of the retained loadings—so we're still projecting onto a unit-length axis defined solely by the SNVs we actually have. Drop missing SNVs, rebuild the axis from the overlap, rescale it to unit length.
