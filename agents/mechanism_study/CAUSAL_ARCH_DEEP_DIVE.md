# Causal Architecture Deep Dive: What Actually Controls PC Utility at Gen 0

## Why this report exists

The prior reports established that PCs can show predictive utility in short-genome gen-0 simulations, but they did not fully answer a more important mechanistic question: *what aspect of causal architecture makes that utility appear, disappear, or change sign?*

This report focuses only on that question.

## Scientific question

When we change trait architecture, what changes most:

1. how well PCs align with true genetic liability (`R²(PC, G_true)`), and
2. whether that alignment translates into held-out predictive gain (`AUC(PC-only) - AUC(random)`)?

We also test whether interaction calibration (`PRS + PC + PRS*PC`) benefits from these architecture shifts, or whether it mostly overfits.

## Hypotheses

I evaluated four architecture-specific hypotheses:

1. **Architecture dependence**: PC utility is not monotonic in causal fraction.
2. **Alignment mechanism**: stronger `PC`-to-`G_true` alignment should correspond to larger predictive gain.
3. **Overlap mechanism**: direct overlap between PCA and causal sites may increase utility.
4. **Calibration-capacity mechanism**: interaction terms may remain fragile even when architecture changes.

## Design and controls

To isolate mechanism rather than leaderboard noise, I ran a factorial sweep with fixed evaluation protocol.

- Replicates: 10 independent seeds
- LD settings: recombination `1e-8`, `2e-8`, `5e-8`
- Causal fractions: `0.02, 0.05, 0.10, 0.20, 0.35, 0.50`
- Effect-size models:
  - infinitesimal
  - sparse spike
  - MAF-dependent
- Causal placement:
  - random
  - clustered
- PCA site selection:
  - `all` (overlap allowed)
  - `disjoint` (direct overlap removed)

Simulation settings:

- `msprime`, panmictic gen-0 setting
- `n_ind=380`, `seq_len=700000`, `n_pca=180`, `h2=0.5`, prevalence `0.1`

Total analyzed condition rows: `2160`.

## Results

### 1) PC signal exists, but it is modest and architecture-sensitive

Across all conditions, mean held-out advantage was:

- `AUC(PC-only) - AUC(random) = 0.025` (95% CI `0.019 to 0.032`)

Interpretation:

- There is real average PC signal in this regime.
- The effect is not huge; architecture determines whether it strengthens or attenuates.

### 2) Causal fraction is not monotonic

The causal-fraction relationship varies by architecture and does not follow a single increasing or decreasing curve.

![Causal fraction by architecture](figures_causal_arch/fig1_fraction_by_arch.png)

Interpretation:

- This resolves the earlier “weird” behavior: inconsistent trends are expected when multiple architecture axes interact.
- A single scalar like “more causal variants” cannot explain the mechanism by itself.

### 3) Alignment matters, but not as a single dominant driver

Across all runs:

- `corr(R²(PC, G_true), AUC gap) = 0.155`

![R2 vs AUC gap](figures_causal_arch/fig3_r2_vs_gap.png)

Interpretation:

- Alignment is directionally relevant.
- But the modest correlation shows that alignment alone is not enough; architecture class and LD context also matter.

### 4) Direct overlap was not a stable driver in this deep run

Estimated overlap effect (`all - disjoint`) was approximately zero:

- mean `-0.000001` (95% CI `-0.006 to 0.006`)

![Overlap effect](figures_causal_arch/fig2_overlap_effect.png)

Interpretation:

- In this experiment, direct site overlap did not consistently explain the PC gain.
- The dominant signal appears to come from broader LD/geometric structure rather than literal PCA-site/causal-site identity overlap.

### 5) Interaction calibration remained slightly harmful

Across architecture settings:

- `AUC(interaction - additive) = -0.010` (95% CI `-0.013 to -0.007`)

![Interaction vs additive by architecture](figures_causal_arch/fig4_int_vs_add_by_arch.png)

Interpretation:

- Even after broad architecture sweeps, interaction terms did not become reliably beneficial at this calibration scale.
- Additive calibration remains the robust default in this regime.

## What this means scientifically

The data support a multi-factor mechanism:

1. PC utility at gen-0 is real but modest.
2. It is architecture-sensitive and non-monotonic in causal fraction.
3. It is partly mediated by LD-induced geometry (alignment), not by a simple direct-overlap shortcut.
4. Interaction calibration is still capacity-limited here and tends to underperform additive calibration.

So the best current model is:

- **PC gain = f(LD geometry, effect-size architecture, causal density, calibration capacity)**,
not
- **PC gain = simple function of causal fraction alone**.

## Practical implications for your pipeline

If the goal is robust predictive calibration in this simulation class:

1. Use additive `PRS + PC` as the default comparator.
2. Treat interaction models as optional and require larger calibration sets + regularization sweeps.

If the goal is to test true portability attenuation mechanisms:

1. Inject attenuation explicitly (ancestry-dependent PRS noise/LD mismatch) rather than expecting it to emerge from gen-0 architecture sweeps.
2. Re-run this same architecture framework in divergence settings to quantify which architecture effects survive under structured populations.

## Reproducibility

Runner:

- `/Users/user/gnomon/agents/mechanism_study/causal_arch_deep_dive.py`

Primary output:

- `/Users/user/gnomon/agents/mechanism_study/results_causal_arch/causal_arch_grid.csv`

This report is intentionally narrative and mechanism-first; detailed stratified rows are in the CSV for follow-up analysis.
