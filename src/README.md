# Gnomon CLI Overview

The `gnomon` binary exposes several subcommands through Clap. Each subcommand wraps a
specific workflow for computing or calibrating polygenic scores. Run `gnomon --help`
to see global usage information.

## Subcommands

### `score`
Calculate raw polygenic scores from genotype data.

Required arguments:
- `input_path`: path to a PLINK `.bed` file or to a directory that contains `.bed`
  files.
- `--score <path>`: path to a single score file or to a directory of score files.

Optional arguments:
- `--keep <path>`: optional file that lists individual IDs to include in the
  calculation.

### `train`
Train a generalized additive model (GAM) used for calibration and saves it to
`model.toml`.

Required arguments:
- `training_data`: path to a TSV file with phenotype, score, and PC columns.
- `--num-pcs <N>`: number of principal components to include.

Key optional arguments:
- `--pgs-knots` / `--pgs-degree`: configure the spline basis for the polygenic score.
- `--pc-knots` / `--pc-degree`: configure the spline basis for principal components.
- `--penalty-order`: order of the difference penalty matrix.
- `--max-iterations` / `--convergence-tolerance`: control the inner P-IRLS loop.
- `--reml-max-iterations` / `--reml-convergence-tolerance`: control the outer REML
  optimization loop.
- `--no-calibration`: disable the post-process calibration layer.

### `infer`
Apply a previously trained calibration model to new samples and saves predictions
as `predictions.tsv`.

Required arguments:
- `test_data`: path to a TSV file with score and PC columns.
- `--model <path>`: path to the trained model TOML file.

Optional arguments:
- `--no-calibration`: disable the post-process calibration layer when generating
  predictions.

## Calibration defaults

Each invocation resets the global flag that enables calibration so that
calibration is active unless explicitly disabled with `--no-calibration`.
