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
- `--num-pcs <N>`: number of leading principal components to include (6 in the
  examples, which fit much faster than 16; any count works).

Optional arguments:
- `--pgs-centers`: centers of the Gaussian model's score smooth (at least 4).
- The PCs enter as one joint Duchon smooth in the context and one in the slope,
  with center counts derived from `--num-pcs` (9 and 8 at 6 PCs, 24 and 20 at
  16); there is no per-PC setting.
- `--latent-law empirical|standard-normal`: the law of the score that a
  marginal-slope model anchors on. `empirical` (the default) declares the
  weighted empirical law of the training scores; `standard-normal` declares a
  score that is already standard normal given the context. The score is never
  transformed to reach either law.
- `--survival-time-wiggle*` requires `--latent-law standard-normal`: gam anchors a
  declared empirical law only on a rigid time baseline, so the combination is
  refused before fitting.

Removed: `--max-iterations`, `--convergence-tolerance`, `--reml-max-iterations` and
`--reml-convergence-tolerance`. gam derives every stop of the inner and outer
solvers from its own convergence certificates, and calibrate sets no iteration
cap or tolerance; passing any of these flags is an error.

### `infer`
Apply a previously trained calibration model to new samples and saves predictions
as `predictions.tsv`.
For a binary model the table holds `probit_index` (the model's mean is
`Φ(probit_index)`) and `prediction`; for a survival model it holds each row's
`sample_id`, the cumulative hazards, `net_risk_entry`/`net_risk_exit` (net risk
under independent censoring by the competing event, not cumulative incidence) and
`conditional_risk`.

Required arguments:
- `test_data`: path to a TSV file with `sample_id`, score and PC columns; a table
  without `sample_id` is refused, since predictions are joined back to samples by it.
- `--model <path>`: path to the trained calibration model JSON bundle.
