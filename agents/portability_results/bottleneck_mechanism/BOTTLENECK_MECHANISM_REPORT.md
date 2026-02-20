# Why Does Bottleneck Worsen PRS Portability?

## 1) The scientific question

In earlier analyses we established that portability loss is real and that bottleneck in the training lineage makes it worse. The unresolved question is mechanistic:

- Is bottleneck harm mostly due to lower diversity (heterozygosity/polymorphism)?
- Or does bottleneck primarily distort the learned tag-effect map, so estimated SNP effects transfer less faithfully?

This report answers that question using paired simulations and explicit controls.

## 2) What we changed to make this rigorous

For every condition, we ran **paired simulations** with the same seed, divergence generation, and causal count:

- `divergence` model
- `bottleneck` model

Then we compared them directly (bottleneck minus divergence), which isolates bottleneck-specific harm from background divergence effects.

### Overfitting controls included

To avoid mistaking overfit for mechanism:

1. POP0 was split into train + two independent held-out test sets.
2. Portability was measured against held-out POP0, not training POP0.
3. A null PRS was built by permuting the training target.

If harm were mostly overfit, these controls would fail. They did not.

## 3) Experimental design

- Simulator: `msprime`
- Settings: two populations, array-like PRS sites (2K), same trait architecture pipeline as portability run
- Grid:
  - scenarios: divergence vs bottleneck
  - divergence generations: `0, 20, 50, 100, 200, 500, 1000, 2000, 5000`
  - causal counts: `200, 1000`
  - seeds: `8`

Total fitted configs: `288`

## 4) Candidate mechanisms we measured

For each paired condition we computed deltas (bottleneck minus divergence):

1. `delta_beta_corr_pop0_vs_pop1`
- Change in agreement between effect vectors learned in POP0 vs POP1.
- Lower agreement means stronger transfer mismatch.

2. `delta_prs_var_ratio_pop1_over_pop0`
- Change in score scaling mismatch across populations.

3. `delta_hetero_pop0_train`
- Change in training-pop heterozygosity (diversity proxy).

4. `delta_mean_abs_maf_diff`
- Change in population allele-frequency distance.

5. `delta_ld_mismatch`
- Change in local tag-causal LD mismatch.

Outcome variable:

- `delta_port = portability_ratio_bottleneck - portability_ratio_divergence`

More negative `delta_port` means bottleneck caused extra harm.

## 5) Results

### 5.1 Overfitting is not the explanation

- POP0 held-out consistency: mean `1.005` (close to ideal 1.0)
- Null PRS R²:
  - POP0-test: `0.0096`
  - POP1: `0.0066`

Interpretation:

- The model is not just memorizing training data.
- Added bottleneck harm is not a trivial overfitting artifact.

### 5.2 What changed under bottleneck?

![Figure 1: mechanism trajectories](fig1_candidate_mechanisms.png)

How to read this figure:

- Each panel tracks a candidate mechanism metric over divergence generations.
- Blue = divergence-only, red = bottleneck.
- Separation between curves indicates bottleneck-specific change.

What it shows:

1. Bottleneck strongly perturbs effect-transfer compatibility metrics (especially beta agreement and score scaling).
2. Diversity/MAF metrics also change, but not as strongly in explanatory power.

### 5.3 Which mechanism best predicts added harm?

![Figure 2: added harm scatter](fig2_added_harm_scatter.png)

How to read this figure:

- X-axis = bottleneck-induced change in a mechanism metric.
- Y-axis = added portability harm (`delta_port`).
- Strong trend means that mechanism likely drives added harm.

Key empirical ranking (correlation with added harm):

1. `delta_beta_corr_pop0_vs_pop1`: `0.761`
2. `delta_prs_var_ratio_pop1_over_pop0`: `0.721`
3. `delta_hetero_pop0_train`: `-0.515`
4. `delta_delta_hetero_pop1_minus_pop0`: `0.455`
5. `delta_mean_abs_maf_diff`: `0.274`
6. `delta_ld_mismatch`: `0.273`

Interpretation:

- The strongest signal is **effect-map transfer distortion** (beta agreement), not simple diversity loss alone.

### 5.4 Multivariable attribution confirms that ranking

![Figure 3: standardized OLS coefficients](fig3_mechanism_ols.png)

How to read this figure:

- Coefficients are standardized and estimated jointly.
- Larger absolute value means stronger independent contribution after controlling for other features.

Top independent contributors:

1. `delta_beta_corr_pop0_vs_pop1` (largest)
2. `delta_prs_var_ratio_pop1_over_pop0` (second)
3. `delta_hetero_pop0_train` (meaningful but smaller)

Interpretation:

- Bottleneck primarily harms portability by disrupting **transferability of learned tag effects**, with score-scaling mismatch as a second driver.
- Reduced diversity contributes, but appears secondary in this model.

## 6) Main conclusion

Bottleneck worsens portability mostly because it changes the **effective tagging architecture used during training**, so learned marginal SNP effects transfer less faithfully to the non-bottleneck assessment population.

Low diversity is part of the story, but not the dominant standalone driver here.

## 7) Practical implication

If we want to mitigate bottleneck-induced portability loss, the best target is not only “more diversity” but specifically methods that are robust to **cross-pop effect-map instability** and **score scaling mismatch**.

## 8) Reproducibility

Script:

- `/Users/user/gnomon/agents/portability_results/bottleneck_mechanism_analysis.py`

Primary outputs:

- `/Users/user/gnomon/agents/portability_results/bottleneck_mechanism/bottleneck_mechanism_results.csv`
- `/Users/user/gnomon/agents/portability_results/bottleneck_mechanism/bottleneck_paired_deltas.csv`
- `/Users/user/gnomon/agents/portability_results/bottleneck_mechanism/bottleneck_mechanism_correlations.csv`
- `/Users/user/gnomon/agents/portability_results/bottleneck_mechanism/bottleneck_mechanism_ols.csv`
