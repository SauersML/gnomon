# Biobank marginal-slope PRS validation

A single, self-contained run that validates gnomon's marginal-slope calibration
against standard baselines on **All of Us** microarray data.

```
bash run.sh                 # survival + binary (default)
bash run.sh --mode survival # survival GAM + Cox baselines only
bash run.sh --mode binary   # binary GAM + logistic baselines only
```

That is the whole analysis. `run.sh` is the only entrypoint; it drives
`marginal_slope_diseases.py`. Nothing else is needed.

## What it does

For the most prevalent diseases in the active CDR (top `TOP_N_DISEASES`, chosen by
intersecting OHDSI-canonical SNOMED conditions with the `SNOMED_PGS_MAP` PGS pool),
the run:

1. Resolves each disease to its SNOMED standard Condition concept and pulls every
   person whose `condition_occurrence` descends from it via `concept_ancestor`.
2. Fits a marginal-slope GAM — `mjs(PC1..PC{NUM_PCS}) + sex` with the disease's
   PGS feeding the latent slope channel — in both a survival
   (`Surv(entry_age, exit_age, event)`) and a binary formulation.
3. Compares against `Z_norm2` and raw-PRS + PC baselines (Cox PH / logistic) on the
   same per-class 80/20 split.
4. Runs leave-one-group-out OOD refits by care site, Census region, and AoU inferred
   genetic ancestry.

Reported per disease: IPCW concordance, integrated Brier score, and Graf-style
integrated-Brier pseudo-R² on held-out rows (survival), plus AUROC / liability-R²
contrasts (binary).

## What `run.sh` handles for you

- Installs `uv` and the latest published gnomon linux-x64 release.
- Resolves and prints the `gamfit` / `gam` solver versions actually used.
- Stages the AoU microarray PLINK triplet (mounted resource or `gs://` copy).
- Picks a single CUDA runtime source (system toolkit or pip wheels) to avoid
  split cuBLAS handle state.
- Keeps all uv / XDG / temp state on the attached workspace disk and fails early
  if it is tight on space or inodes.

## Requirements

Run from inside the All of Us workbench (or any environment with the controlled CDR):

- `WORKSPACE_CDR` — BigQuery CDR dataset (required).
- `GOOGLE_PROJECT` — billing project for `gs://` access (recommended).

## Outputs

- Per-run log: `~/aou-gpu-baremetal/biobank_results/biobank_run_<timestamp>.log`
  (full output plus an extracted `SUMMARY` section at the end).
- Fitted survival models: `.gamfit` artifacts under the artifact cache (`FITS_DIR`).

## Configuration

The fit knobs live at the top of `marginal_slope_diseases.py`
(`NUM_PCS`, `MJS_CENTERS`, `TRAIN_FRACTION`, `RNG_SEED`, `TOP_N_DISEASES`,
`SNOMED_PGS_MAP`, `LOSO_AXES`). `run.sh` echoes the active values into the log
header at the start of every run.
