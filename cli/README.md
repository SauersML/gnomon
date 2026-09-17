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
- `--out <PREFIX>`: write the scores to `PREFIX.sscore`. The score-file caches
  (`gnomon_score_cache/`) go under PREFIX's directory instead of beside the inputs. Without it, `<GENOTYPE>_<SCORE>.sscore` lands
  beside the genotype data. See [`score/README.md`](../score/README.md).

### `terms`
Infer per-sample metadata terms (currently sex) from genotype data.

Usage: `gnomon terms --sex [--build <37|38>] [--out <PREFIX>] <GENOTYPE_PATH>`

Writes `<GENOTYPE>.sex.tsv` beside the genotype data, or `PREFIX.sex.tsv` with
`--out`. See [`terms/README.md`](../terms/README.md).

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
- `--latent-law empirical|standard-normal`: the law of the score that a
  marginal-slope model anchors on. `empirical` (the default) declares the
  weighted empirical law of the training scores; `standard-normal` declares a
  score that is already standard normal given the context. The score is never
  transformed to reach either law.
- `--survival-time-wiggle*` requires `--latent-law standard-normal`: gam anchors a
  declared empirical law only on a rigid time baseline, so the combination is
  refused before fitting.

Removed: `--max-iterations` and `--convergence-tolerance`. Training now leaves
the inner solver to gam, which stops on its own convergence certificates;
passing either flag is an error that names
`--reml-max-iterations` / `--reml-convergence-tolerance` as the outer-loop bounds.
- `--reml-max-iterations` / `--reml-convergence-tolerance`: optional overrides of the
  outer smoothing-parameter and length-scale search. Absent, gnomon passes nothing and
  gam's own defaults apply: 80 iterations at relative tolerance 1e-4 for the spatial
  length-scale search, 60 at 1e-5 for the Gaussian location-scale fit, and 200 on the
  marginal-slope formula route. An override can only loosen or tighten what gam
  certifies; the defaults are the accurate choice.

### `infer`
Apply a previously trained calibration model to new samples and saves predictions
as `predictions.tsv`.

Required arguments:
- `test_data`: path to a TSV file with score and PC columns.
- `--model <path>`: path to the trained calibration model JSON bundle.
