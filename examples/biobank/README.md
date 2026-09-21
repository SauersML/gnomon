# Biobank study

`study.py` calibrates fixed, published polygenic scores for common diseases in
All of Us: scores -> cohort -> features -> fits -> predict -> evaluate -> digest,
in one resumable run. gnomon scores the genotypes, builds the cohort and
phenotype tables, and evaluates; every model is fitted and predicted by
[gam](https://github.com/SauersML/gam) through `gamfit`.

```
study.py run --work DIR --source parquet --input tables=DIR     # simulator tables
submit_study.py submit [--config study.json]                    # one AoU run (study.wdl)
submit_study.py tokens RUN > tokens.txt                         # its aggregate tokens
tabulate_study.py tokens.txt                                    # the results table
```

- `study.json` holds every model and compute choice, each with its reason.
- `study/diseases.json` is the locked disease and score list; `study/DISEASES.md`
  is its rationale.
- `study/simulate.py` writes the same tables the AoU cohort stage exports, so
  every later stage runs identically on synthetic data and in AoU.
- `study/disclosure.py` enforces the All of Us rule on released counts: only
  aggregate tokens leave the workspace.
- `study/build_wheel.py` stages the pinned `gamfit` wheel and its wheelhouse.

## The model

The score stays on its own axis, standardized once on the fitting rows. Risk is
a Bernoulli marginal-slope model, `P(Y = 1 | z, x) = Phi(a(x) + b(x) z)`: the
marginal index and the slope `b(x)` each carry one joint Duchon smooth of the
PCs, and the intercept `a(x)` is anchored so the risk integrates to the marginal
index over the training rows' empirical score law. Predictions are gam's
posterior means, and the reported score slope is gam's analytic
`probit_score_derivative` from the same posterior nodes.

## Tests

`tests/` runs under pytest with `study/requirements.txt` installed; the fits in
`tests/test_study_models_binary.py` need the `gamfit` wheel.
