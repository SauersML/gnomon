# Sims Portability Summary (Gen0 + Bottleneck)

Last updated: February 8, 2026

## TL;DR
The new `sims/` pipelines and parallel workflow run successfully on GitHub Actions.
The bottleneck mechanism conclusions are strong and consistent with the LD-tagging failure hypothesis.
Gen0 orthogonalization is directionally replicated, but the quick-profile run is underpowered for one standalone significance check.

## Workflow
- Workflow: `Sims (Gen0 + Bottleneck)`
- File: `.github/workflows/sims-portability.yml`
- Successful runs:
  - Push run: [21804502215](https://github.com/SauersML/gnomon/actions/runs/21804502215)
  - Manual quick run: [21804506573](https://github.com/SauersML/gnomon/actions/runs/21804506573)

## What Was Run
- `sims/gen0_orthogonalization.py`
  - Tests weak-PRS PC artifact, orthogonalization control, and strong-PRS control.
- `sims/bottleneck_ld_mechanism.py`
  - Tests divergence vs bottleneck portability in paired simulations.
  - Tests near vs far tags, heterozygosity-normalization intervention, and LD-destroy intervention.
  - Tests association between added bottleneck harm and marginal-beta decorrelation.

## Main Results (Quick Profile)
### Gen0
- Weak+Original additive gain over raw PRS: `+0.0278` AUC (95% CI `-0.0225, 0.0782`)
- Weak+Orthogonalized additive gain: `-0.0085` (95% CI `-0.0401, 0.0231`)
- Paired contrast (Weak Original > Weak Orthogonalized): `p = 0.0237`
- Standalone Weak+Original > 0 test: not significant in quick profile (`p = 0.1572`)

Interpretation:
- Direction matches the expected artifact.
- Orthogonalization suppresses the gain as expected.
- Quick mode has limited power for the standalone H1 test.

### Bottleneck Mechanism
- Near-vs-far portability delta: `+0.1563` (`p = 1.47e-11`)
- Heterozygosity-normalization effect: `-0.0019` (`p = 0.657`)
- LD-destroy effect: `-0.4175` (`p = 2.98e-16`)
- Corr(added harm, beta decorrelation): `r = 0.763` (95% CI `0.580, 0.873`)
- Corr(added harm, heterozygosity shift delta): `r = 0.429`

Interpretation:
- Portability loss is primarily driven by LD-tagging transfer failure.
- Bottleneck-added harm tracks marginal-beta decorrelation strongly.
- Heterozygosity shift alone does not explain the effect.

## Unexpected / Watchouts
- Gen0 standalone H1 significance was weaker than expected in quick profile.
- This is likely a power issue, not a directional contradiction.
- Use full profile for final Gen0 inferential claims.

## Artifacts Produced
- `gen0-report` artifact:
  - `GEN0_ORTHOGONALIZATION_REPORT.txt`
  - `gen0_orthogonalization_merged.csv`
  - `gen0_orthogonalization_summary.csv`
  - `fig_gen0_orthogonalization.png`
- `bottleneck-report` artifact:
  - `BOTTLENECK_LD_MECHANISM_REPORT.txt`
  - `bottleneck_ld_mechanism_results.csv`
  - `bottleneck_ld_mechanism_summary.csv`
  - `bottleneck_paired_deltas.csv`
  - `fig1_near_vs_far.png`
  - `fig2_ld_vs_hetero_interventions.png`
  - `fig3_added_harm_correlates.png`

## Bottom Line
Current evidence strongly supports the central claim:
**portability loss and bottleneck-added harm are dominated by LD-tagging failure (marginal-beta decorrelation), not by heterozygosity/variance shifts.**
