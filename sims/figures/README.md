# Figures: Portability and Confounding Sims

This folder contains dedicated pipelines for the two requested simulation figures.

## Layout

- `common.py`: shared helpers (PGS003725 loading, PCA, true-risk construction, ancestry utilities)
- `figure1_msprime_portability.py`: Figure 1 (2-pop msprime portability)
- `figure2_stdpopsim_confounding.py`: Figure 2 (stdpopsim confounding)

Compatibility wrappers are kept at:
- `/Users/user/gnomon/sims/figure1_msprime_portability.py`
- `/Users/user/gnomon/sims/figure2_stdpopsim_confounding.py`

## Run

From repo root:

```bash
python3 /Users/user/gnomon/sims/figure1_msprime_portability.py --out /Users/user/gnomon/sims/results_figure1_local
python3 /Users/user/gnomon/sims/figure2_stdpopsim_confounding.py --out /Users/user/gnomon/sims/results_figure2_local
```

## Cohort Sizes

All sizes are reduced by 100 for local execution:
- Figure 1: AFR=100, OoA_train=100, OoA_test=100
- Figure 2: EUR_train=120, EUR_test=30, AFR_test=30, ASIA_test=30, ADMIX_test=30

## HPC Orchestrator

Use the single entrypoint:

```bash
python3 /Users/user/gnomon/sims/main.py setup --install-tools-with-conda
python3 /Users/user/gnomon/sims/main.py run --full-cohort --figure both --work-root /dev/shm/gnomon_sims_work --out-root /path/to/persistent/results --jobs 32 --clear-ramdisk-after
```

### RAM disk vs main disk

- Put on RAM disk / fast scratch (`--work-root`):
  - Per-seed/per-generation `*_work/` directories
  - PLINK split files (`train.*`, `test.*`, `ref.afreq`)
  - BayesR intermediate files (`bayesr*`, `*.score`, `*.sscore`)
- Keep on persistent main disk (`--out-root`):
  - Final summary TSVs and PNG figures
  - Optional intermediates only when debugging (`--keep-intermediates`)

### Clearing RAM disk

```bash
python3 /Users/user/gnomon/sims/main.py clean-ramdisk --work-root /dev/shm/gnomon_sims_work
```

## Outputs

Figure 1 output directory includes:
- `figure1_auc_ratio.tsv`
- `figure1_auc_ratio.png`
- `figure1_demography.png`
- `figure1_prs_distributions_grid.png`
- `figure1_pc12_grid.png`

Figure 2 output directory includes:
- `figure2_auc_by_method_population.tsv`
- `figure2_auc_by_method_population.png`
- `figure2_prs_distributions.png`
- `figure2_pc12.png`

## Notes

- BayesR requires `gctb` and `plink2` on `PATH`.
- `normalized` requires `pgscatalog-calc`; `gam` uses `mgcv` through `rpy2`.
- PGS003725 is downloaded and cached under `sims/.cache` when first run.
