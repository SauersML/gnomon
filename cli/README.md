# Gnomon CLI Overview

The `gnomon` binary exposes several subcommands. Each subcommand wraps a
specific workflow for computing or calibrating polygenic scores. Run `gnomon --help`
to see global usage information.

## Subcommands

### `score`
Calculate raw polygenic scores from genotype data.

Required arguments:
- `score_path`: path to a single score file or to a directory of score files.
- `input_path`: path to a PLINK `.bed` file or to a directory that contains `.bed`
  files.

Optional arguments:
- `--keep <path>`: optional file that lists individual IDs to include in the
  calculation.

### `fit`
Fit an HWE PCA model from genotype data.

Usage: `gnomon fit <GENOTYPE_PATH> --components <N> [--out <PREFIX>] [--maf <MAF>] [--list <PATH>] [--ld [--sites_window <SITES> | --bp_window <BP>]]`

With `--ld` alone, gnomon uses a 500 kbp window and a 100,000-marker safety
budget. `--out` isolates all fit artifacts from the genotype directory.

### `project`
Project samples into an existing HWE PCA space.

Usage: `gnomon project <GENOTYPE_PATH> [--model <MODEL_NAME>]`

### `train`
Train a generalized additive model used for calibration and save its complete inference bundle to `model.json`.

Required arguments:
- `training_data`: path to a TSV file with phenotype, score, and PC columns.
- `--num-pcs <N>`: number of principal components to include.

Optional arguments:
- `--pgs-centers`: farthest-point centers for the PGS Duchon smooth (at least 4).
- `--pc-centers`: farthest-point centers for each PC Duchon smooth (at least 4).
- `--max-iterations` / `--convergence-tolerance`: control the inner P-IRLS loop.
- `--reml-max-iterations` / `--reml-convergence-tolerance`: control the outer REML
  optimization loop.

### `infer`
Apply a previously trained calibration model to new samples and saves predictions
as `predictions.tsv`.

Required arguments:
- `test_data`: path to a TSV file with score and PC columns.
- `--model <path>`: path to the trained calibration model JSON bundle.
