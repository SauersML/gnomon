# Final Mechanism Report: Why PCs Look Predictive at Gen 0

## 1. Scientific question

The core question is not "which model has higher AUC," but:

- What mechanism makes PC-based terms look useful in the gen-0 simulation?
- Is that mechanism biological/statistical signal, leakage, or overfitting?

This matters because your actual hypothesis is about ancestry-dependent PRS attenuation. If we misidentify the gen-0 mechanism, we may optimize the wrong model behavior.

## 2. Working hypotheses

I tested six explicit hypotheses:

1. **LD-structure hypothesis (H1)**: PCs are predictive because they compress LD structure that overlaps directions in `G_true`.
2. **Causal-overlap hypothesis (H2)**: PC signal is stronger when PCA sites overlap causal architecture.
3. **Leakage hypothesis (H3)**: PC predictivity is inflated by computing PCs with test data.
4. **Complexity hypothesis (H4)**: interaction models (`P + PC + P*PC`) overfit at limited calibration size.
5. **Null-behavior hypothesis (H5)**: under permutation, apparent signal should disappear.
6. **Architecture hypothesis (H6)**: the effect depends on how sparse/polygenic the trait is.

## 3. Why this design

A single benchmark cannot separate these mechanisms. So I used targeted controls where each experiment isolates one explanation while holding the rest fixed.

- To test LD, I changed recombination while keeping the rest of the pipeline fixed.
- To test leakage, I used train-only PCA projection versus all-sample PCA.
- To test overfitting, I compared additive and interaction calibrators under matched data.
- To test architecture, I swept causal fraction.

This is the minimum design needed to make mechanism claims rather than leaderboard claims.

## 4. Experimental setup (high-power fast run)

- Seeds: 12 independent replicates (`1..12`)
- Simulator: `msprime`
- Individuals per seed: `n_ind = 380`
- Sequence length: `700,000 bp`
- Baseline causal variants: `260`
- PCA variants: `180`
- LD grid (H1): recombination `1e-8`, `2e-8`, `5e-8`
- Permutations per seed (H5): `20`

Evaluation framework:

- Train/test split at individual level (50/50)
- Calibration/validation split inside test set
- Primary metric: AUC on held-out validation subset

## 5. Results and interpretation

### 5.1 Is there real PC signal beyond random features? (H1)

Summary estimate:

- `AUC(PC-only) - AUC(random-PC)` = **0.095** (95% CI **0.045 to 0.145**)

Interpretation:

- This supports real held-out predictivity of PC-derived features in this short-LD setting.
- So the effect is not just arbitrary feature count.

### 5.2 Is this mostly PCA leakage? (H3)

Summary estimate:

- `AUC(all-sample-PC) - AUC(train-only-PC)` = **0.001** (95% CI **-0.030 to 0.031**)

Interpretation:

- No measurable leakage advantage in this setup.
- This weakens the "PCs only work because test information leaked" explanation.

### 5.3 Do interaction models help once PCs are included? (H4)

Summary estimate:

- `AUC(interaction) - AUC(additive)` = **-0.018** (95% CI **-0.036 to -0.001**)

Interpretation:

- Interaction terms are mildly harmful here.
- In this calibration-size regime, additive `P + PC` is more robust than `P + PC + P*PC`.

### 5.4 Do overlap and architecture matter? (H2, H6)

Observed pattern:

- Removing direct overlap (`disjoint`, `disjoint_buffer`) lowered PC-only AUC relative to overlap-allowed PCA sets in this run.
- Causal-fraction sweep showed non-monotonic behavior, with strongest PC-vs-random separation at intermediate fractions.

Interpretation:

- The mechanism depends on architecture details, not a single universal scalar.
- Overlap/near-overlap contributes in this simulation regime.

### 5.5 Null behavior (H5)

- Fraction of seeds with empirical permutation `p < 0.05`: **0.17**

Interpretation:

- Signal exists in some seeds but is heterogeneous, which is expected in short-genome finite-sample settings.
- This does not support a global leakage artifact; it supports variable signal strength.

## 6. Scientific conclusion

The best-supported explanation is:

1. **PC features carry real predictive information under LD geometry** in these gen-0 short-genome simulations.
2. **That effect is not primarily due to train/test PCA leakage** in this setup.
3. **Interaction calibration is not the source of gains here** and is slightly overfit-prone at current calibration sample sizes.

So the current gen-0 phenomenon is better described as **LD-mediated feature signal + calibration-model capacity effects**, not as evidence that PC-interaction attenuation modeling is already working optimally.

## 7. What this means for your main project

If your scientific target is portability attenuation, the immediate modeling recommendation is:

- Use additive `PRS + PCs` as the stable baseline for this simulation class.
- Treat interaction terms as conditional tools that require larger calibration sets and stronger regularization.

If your target is to prove ancestry-dependent attenuation mechanism specifically, next simulations should explicitly inject that mechanism rather than expecting it to emerge from this short-genome setup.

## 8. Recommended next experiments

1. Increase calibration N and regularization sweep for interaction terms.
2. Repeat the same mechanism tests at larger sequence length.
3. Add explicit ancestry-dependent PRS noise to test attenuation recovery directly.
4. Replicate this exact high-power framework in a divergence/portability setting.

## 9. Reproducibility

Primary runner:

- `/Users/user/gnomon/agents/mechanism_study/highpower_mechanism_run.py`

Command used:

```bash
python3 highpower_mechanism_run.py \
  --n-seeds 12 \
  --workers 6 \
  --n-ind 380 \
  --seq-len 700000 \
  --n-pca 180 \
  --n-causal 260 \
  --n-perm 20
```

Outputs:

- `/Users/user/gnomon/agents/mechanism_study/results_highpower/h1.csv`
- `/Users/user/gnomon/agents/mechanism_study/results_highpower/h2.csv`
- `/Users/user/gnomon/agents/mechanism_study/results_highpower/h3.csv`
- `/Users/user/gnomon/agents/mechanism_study/results_highpower/h4.csv`
- `/Users/user/gnomon/agents/mechanism_study/results_highpower/h5.csv`
- `/Users/user/gnomon/agents/mechanism_study/results_highpower/h6.csv`

Figures (GitHub-renderable):

![LD mechanism](figures_highpower/fig1_ld.png)

![Overlap control](figures_highpower/fig2_overlap.png)

![Model complexity](figures_highpower/fig3_complexity.png)

![Causal fraction sweep](figures_highpower/fig4_causal_fraction.png)
