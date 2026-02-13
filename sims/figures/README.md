# Figures: Portability and Confounding Sims

This folder contains dedicated pipelines for the two requested simulation figures.

## Layout

- `common.py`: shared helpers (PGS003725 loading, PCA, true-risk construction, ancestry utilities)
- `figure1_msprime_portability.py`: Figure 1 (2-pop msprime portability)
- `figure2_stdpopsim_confounding.py`: Figure 2 (stdpopsim confounding)

Compatibility wrappers are kept at:
- `/Users/user/gnomon/sims/figure1_msprime_portability.py`
- `/Users/user/gnomon/sims/figure2_stdpopsim_confounding.py`

## Local-scale sample sizes

All sizes are reduced by 100 for local execution:
- Figure 1: AFR=100, OoA_train=100, OoA_test=100
- Figure 2: EUR_train=120, EUR_test=30, AFR_test=30, ASIA_test=30, ADMIX_test=30

## Run

From repo root:

```bash
python3 /Users/user/gnomon/sims/figure1_msprime_portability.py --out /Users/user/gnomon/sims/results_figure1_local
python3 /Users/user/gnomon/sims/figure2_stdpopsim_confounding.py --out /Users/user/gnomon/sims/results_figure2_local
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
