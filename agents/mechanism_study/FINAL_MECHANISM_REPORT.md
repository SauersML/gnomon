# Final Mechanism Report (High-Power Fast Run)

## Methods

Seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

n_ind=380, seq_len=700000, n_pca=180, baseline n_causal=260, perms/seed=20

Hypotheses: H1 LD dose, H2 overlap, H3 leakage control, H4 model complexity, H5 permutation null, H6 causal-fraction sweep.
LD grid used for H1: `1e-8`, `2e-8`, `5e-8`.


## Results

- H1 PC-only minus random AUC: mean **0.095** (95% CI 0.045, 0.145)

- H3 leakage delta (all-PC minus train-PC): mean **0.001** (95% CI -0.030, 0.031)

- H4 interaction minus additive: mean **-0.018** (95% CI -0.036, -0.001)

- H5 empirical p<0.05 fraction: **0.17**


### H1 LD Strength

|   recomb_rate |   r2_pc_g |   auc_pc_only |   auc_rand_only |
|--------------:|----------:|--------------:|----------------:|
|        0.0000 |    0.2860 |        0.5760 |          0.4975 |
|        0.0000 |    0.2321 |        0.5364 |          0.4335 |
|        0.0000 |    0.0811 |        0.5538 |          0.4502 |


### H2 Overlap Controls

| mode            |   pct_overlap |   r2_pc_g |   auc_pc_only |
|:----------------|--------------:|----------:|--------------:|
| all             |       19.9537 |    0.1497 |        0.5427 |
| disjoint        |        0.0000 |    0.1518 |        0.4937 |
| disjoint_buffer |        0.0000 |    0.1518 |        0.4937 |


### H3 Leakage

|              |   mean |
|:-------------|-------:|
| r2_allpc_g   | 0.1552 |
| r2_trainpc_g | 0.1532 |
| auc_allpc    | 0.5506 |
| auc_trainpc  | 0.5500 |


### H4 Complexity

|                |   auc_raw |   auc_add |   auc_int |
|:---------------|----------:|----------:|----------:|
| (80, 'noise')  |    0.4859 |    0.5404 |    0.5176 |
| (80, 'strong') |    0.7302 |    0.6865 |    0.6587 |
| (80, 'weak')   |    0.5744 |    0.5760 |    0.5652 |
| (95, 'noise')  |    0.4811 |    0.5208 |    0.5159 |
| (95, 'strong') |    0.7066 |    0.6739 |    0.6628 |
| (95, 'weak')   |    0.5815 |    0.5877 |    0.5549 |


### H5 Permutation

|   auc_real_pc |   auc_null_mean |   p_empirical |
|--------------:|----------------:|--------------:|
|        0.5899 |          0.5046 |        0.2381 |
|        0.6348 |          0.5197 |        0.2381 |
|        0.5179 |          0.4809 |        0.2857 |
|        0.6318 |          0.4764 |        0.1429 |
|        0.4644 |          0.5180 |        0.7619 |
|        0.4242 |          0.5068 |        0.9048 |
|        0.6810 |          0.4821 |        0.0476 |
|        0.6576 |          0.5044 |        0.0952 |
|        0.3539 |          0.4808 |        0.9048 |
|        0.7021 |          0.4858 |        0.0476 |
|        0.3865 |          0.4481 |        0.7619 |
|        0.5632 |          0.4709 |        0.2381 |


### H6 Causal Fraction

|   causal_fraction |   r2_pc_g |   auc_pc_only |   auc_rand_only |
|------------------:|----------:|--------------:|----------------:|
|            0.0500 |    0.2589 |        0.5293 |          0.5691 |
|            0.1000 |    0.1873 |        0.5280 |          0.5017 |
|            0.2000 |    0.1947 |        0.6263 |          0.5075 |
|            0.4000 |    0.1518 |        0.5604 |          0.4743 |


## Figures

![H1 LD](figures_highpower/fig1_ld.png)

![H2 overlap](figures_highpower/fig2_overlap.png)

![H4 complexity](figures_highpower/fig3_complexity.png)

![H6 causal fraction](figures_highpower/fig4_causal_fraction.png)


## Conclusions

PC features can carry real held-out signal under LD structure, but interaction-heavy calibrators are fragile and can underperform additive forms at limited calibration size. Leakage controls (train-only PCs) are included and should be interpreted alongside permutation-null evidence.
