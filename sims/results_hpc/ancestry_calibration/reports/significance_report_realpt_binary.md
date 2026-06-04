# Real P+T Significance Analysis

Positive gamfit advantage means gamfit is better. For AUC and liability pseudo-R2 this is gamfit minus baseline; for error metrics it is baseline minus gamfit.

Tests are paired on the same simulated unit: global metrics pair by demography, phenotype, and seed; distance-bin metrics pair by demography, phenotype, seed, and distance bin. P-values are one-sided for gamfit advantage > 0. BH q-values correct across all reported contrasts per test family.


## Main phenotype: deme-varying environmental baseline risk

| source        | dem      | pheno   | baseline_label   | metric_label              |   n_pairs |   win_rate |   gamfit_advantage_mean |   advantage_ci95_low |   advantage_ci95_high |   p_wilcoxon_win |   q_bh_wilcoxon_win |   p_wilcoxon_loss |   q_bh_wilcoxon_loss |
|:--------------|:---------|:--------|:-----------------|:--------------------------|----------:|-----------:|------------------------:|---------------------:|----------------------:|-----------------:|--------------------:|------------------:|---------------------:|
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Abs. prevalence error     |        44 |     0.7273 |                0.01539  |            0.008085  |              0.0233   |        0.0001521 |           0.0003703 |           0.9999  |               1      |
| distance_bins | grid2d   | phenoA  | z-norm           | Abs. prevalence error     |        44 |     0.8864 |                0.05424  |            0.03948   |              0.07194  |        2.45e-10  |           2.287e-09 |           1       |               1      |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Abs. true-slope error     |        44 |     0.3864 |               -0.04697  |           -0.09313   |             -0.008214 |        0.9689    |           0.9988    |           0.03198 |               0.4477 |
| distance_bins | grid2d   | phenoA  | z-norm           | Abs. true-slope error     |        44 |     0.3636 |               -0.05475  |           -0.1107    |              0.002856 |        0.9605    |           0.9988    |           0.0405  |               0.4536 |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | AUC                       |        44 |     0.5455 |                0.01298  |           -0.005783  |              0.03112  |        0.09166   |           0.1222    |           0.9103  |               1      |
| distance_bins | grid2d   | phenoA  | z-norm           | AUC                       |        44 |     0.7045 |                0.05384  |            0.03375   |              0.0763   |        1.883e-05 |           5.022e-05 |           1       |               1      |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |        44 |     0.75   |                0.02077  |            0.0138    |              0.02867  |        3.495e-07 |           1.305e-06 |           1       |               1      |
| distance_bins | grid2d   | phenoA  | z-norm           | Brier Skill Score (BSS)   |        44 |     0.9773 |                0.2285   |            0.1041    |              0.6528   |        1.137e-13 |           3.183e-12 |           1       |               1      |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |        44 |     0.5682 |                0.02077  |            0.000728  |              0.05076  |        0.1194    |           0.1555    |           0.8829  |               1      |
| distance_bins | grid2d   | phenoA  | z-norm           | Liability-scale pseudo-R2 |        44 |     0.6818 |                0.08427  |            0.05199   |              0.1254   |        4.09e-05  |           0.0001041 |           1       |               1      |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | MAE vs known true risk    |        44 |     0.7273 |                0.005718 |            0.002438  |              0.008729 |        0.0001686 |           0.0003934 |           0.9998  |               1      |
| distance_bins | grid2d   | phenoA  | z-norm           | MAE vs known true risk    |        44 |     0.8409 |                0.03572  |            0.02613   |              0.04901  |        6.47e-10  |           5.176e-09 |           1       |               1      |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | RMSE vs known true risk   |        44 |     0.8182 |                0.007222 |            0.004839  |              0.0104   |        6.667e-07 |           2.333e-06 |           1       |               1      |
| distance_bins | grid2d   | phenoA  | z-norm           | RMSE vs known true risk   |        44 |     1      |                0.0355   |            0.02642   |              0.04857  |        5.684e-14 |           3.183e-12 |           1       |               1      |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Abs. prevalence error     |        40 |     0.825  |                0.03576  |            0.0234    |              0.05046  |        1.019e-06 |           3.356e-06 |           1       |               1      |
| distance_bins | serial1d | phenoA  | z-norm           | Abs. prevalence error     |        40 |     0.8    |                0.05312  |            0.03778   |              0.06921  |        5.817e-08 |           2.962e-07 |           1       |               1      |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Abs. true-slope error     |        40 |     0.625  |                0.02621  |           -0.005189  |              0.05209  |        0.01972   |           0.03068   |           0.9809  |               1      |
| distance_bins | serial1d | phenoA  | z-norm           | Abs. true-slope error     |        40 |     0.475  |               -0.002449 |           -0.04331   |              0.03557  |        0.566     |           0.6603    |           0.4393  |               1      |
| distance_bins | serial1d | phenoA  | PGS + PCs        | AUC                       |        40 |     0.6    |                0.00146  |           -0.002119  |              0.00514  |        0.2025    |           0.252     |           0.8013  |               1      |
| distance_bins | serial1d | phenoA  | z-norm           | AUC                       |        40 |     0.675  |                0.002485 |           -0.0006698 |              0.005863 |        0.04862   |           0.06981   |           0.9528  |               1      |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |        40 |     0.775  |                0.03181  |            0.01829   |              0.05745  |        2.923e-06 |           8.614e-06 |           1       |               1      |
| distance_bins | serial1d | phenoA  | z-norm           | Brier Skill Score (BSS)   |        40 |     0.875  |                0.06823  |            0.04426   |              0.121    |        1.346e-08 |           7.537e-08 |           1       |               1      |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |        40 |     0.425  |               -0.005397 |           -0.01175   |             -0.001251 |        0.9359    |           0.9988    |           0.06578 |               0.5694 |
| distance_bins | serial1d | phenoA  | z-norm           | Liability-scale pseudo-R2 |        40 |     0.35   |               -0.005676 |           -0.01208   |             -0.002064 |        0.9809    |           0.9988    |           0.01972 |               0.4477 |
| distance_bins | serial1d | phenoA  | PGS + PCs        | MAE vs known true risk    |        40 |     0.7    |                0.01066  |            0.004934  |              0.02037  |        0.001196  |           0.00203   |           0.9989  |               1      |
| distance_bins | serial1d | phenoA  | z-norm           | MAE vs known true risk    |        40 |     0.7    |                0.01927  |            0.01076   |              0.03197  |        0.0003777 |           0.0008135 |           0.9996  |               1      |
| distance_bins | serial1d | phenoA  | PGS + PCs        | RMSE vs known true risk   |        40 |     0.825  |                0.01166  |            0.007112  |              0.01933  |        2.264e-06 |           7.043e-06 |           1       |               1      |
| distance_bins | serial1d | phenoA  | z-norm           | RMSE vs known true risk   |        40 |     0.8    |                0.02118  |            0.01457   |              0.03073  |        1.411e-07 |           5.646e-07 |           1       |               1      |
| global_test   | grid2d   | phenoA  | PGS + PCs        | AUC                       |         4 |     1      |                0.02432  |            0.02106   |              0.02925  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | z-norm           | AUC                       |         4 |     1      |                0.1463   |            0.075     |              0.1941   |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |         4 |     1      |                0.02118  |            0.01528   |              0.02726  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | z-norm           | Brier Skill Score (BSS)   |         4 |     1      |                0.06465  |            0.03832   |              0.09098  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |         4 |     1      |                0.03866  |            0.01936   |              0.04958  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | z-norm           | Liability-scale pseudo-R2 |         4 |     1      |                0.2068   |            0.08732   |              0.3977   |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | PGS + PCs        | MAE vs known true risk    |         4 |     1      |                0.007361 |            0.003884  |              0.009633 |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | z-norm           | MAE vs known true risk    |         4 |     1      |                0.02656  |            0.01376   |              0.03936  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | PGS + PCs        | RMSE vs known true risk   |         4 |     1      |                0.008255 |            0.005218  |              0.01129  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | grid2d   | phenoA  | z-norm           | RMSE vs known true risk   |         4 |     1      |                0.02644  |            0.01372   |              0.03916  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | PGS + PCs        | AUC                       |         4 |     1      |                0.0448   |            0.02962   |              0.07388  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | z-norm           | AUC                       |         4 |     1      |                0.07587  |            0.03875   |              0.1019   |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |         4 |     1      |                0.02504  |            0.01519   |              0.0405   |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | z-norm           | Brier Skill Score (BSS)   |         4 |     1      |                0.04251  |            0.03029   |              0.05017  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |         4 |     0.5    |                0.02047  |           -0.01356   |              0.05449  |        0.3125    |           0.3125    |           0.8125  |               1      |
| global_test   | serial1d | phenoA  | z-norm           | Liability-scale pseudo-R2 |         4 |     0.75   |                0.1033   |            0.004869  |              0.2017   |        0.125     |           0.1282    |           0.9375  |               1      |
| global_test   | serial1d | phenoA  | PGS + PCs        | MAE vs known true risk    |         4 |     1      |                0.01066  |            0.006115  |              0.01857  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | z-norm           | MAE vs known true risk    |         4 |     1      |                0.01927  |            0.01105   |              0.03055  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | PGS + PCs        | RMSE vs known true risk   |         4 |     1      |                0.0112   |            0.005643  |              0.01983  |        0.0625    |           0.06757   |           1       |               1      |
| global_test   | serial1d | phenoA  | z-norm           | RMSE vs known true risk   |         4 |     1      |                0.01886  |            0.01252   |              0.02914  |        0.0625    |           0.06757   |           1       |               1      |


## All phenotypes

| source        | dem      | pheno   | baseline_label   | metric_label              |   n_pairs |   win_rate |   gamfit_advantage_mean |   advantage_ci95_low |   advantage_ci95_high |   p_wilcoxon_win |   q_bh_wilcoxon_win |   p_wilcoxon_loss |   q_bh_wilcoxon_loss |
|:--------------|:---------|:--------|:-----------------|:--------------------------|----------:|-----------:|------------------------:|---------------------:|----------------------:|-----------------:|--------------------:|------------------:|---------------------:|
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Abs. prevalence error     |        44 |     0.7273 |                0.01539  |            0.008085  |             0.0233    |        0.0001521 |           0.0003703 |           0.9999  |              1       |
| distance_bins | grid2d   | phenoA  | z-norm           | Abs. prevalence error     |        44 |     0.8864 |                0.05424  |            0.03948   |             0.07194   |        2.45e-10  |           2.287e-09 |           1       |              1       |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Abs. true-slope error     |        44 |     0.3864 |               -0.04697  |           -0.09313   |            -0.008214  |        0.9689    |           0.9988    |           0.03198 |              0.4477  |
| distance_bins | grid2d   | phenoA  | z-norm           | Abs. true-slope error     |        44 |     0.3636 |               -0.05475  |           -0.1107    |             0.002856  |        0.9605    |           0.9988    |           0.0405  |              0.4536  |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | AUC                       |        44 |     0.5455 |                0.01298  |           -0.005783  |             0.03112   |        0.09166   |           0.1222    |           0.9103  |              1       |
| distance_bins | grid2d   | phenoA  | z-norm           | AUC                       |        44 |     0.7045 |                0.05384  |            0.03375   |             0.0763    |        1.883e-05 |           5.022e-05 |           1       |              1       |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |        44 |     0.75   |                0.02077  |            0.0138    |             0.02867   |        3.495e-07 |           1.305e-06 |           1       |              1       |
| distance_bins | grid2d   | phenoA  | z-norm           | Brier Skill Score (BSS)   |        44 |     0.9773 |                0.2285   |            0.1041    |             0.6528    |        1.137e-13 |           3.183e-12 |           1       |              1       |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |        44 |     0.5682 |                0.02077  |            0.000728  |             0.05076   |        0.1194    |           0.1555    |           0.8829  |              1       |
| distance_bins | grid2d   | phenoA  | z-norm           | Liability-scale pseudo-R2 |        44 |     0.6818 |                0.08427  |            0.05199   |             0.1254    |        4.09e-05  |           0.0001041 |           1       |              1       |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | MAE vs known true risk    |        44 |     0.7273 |                0.005718 |            0.002438  |             0.008729  |        0.0001686 |           0.0003934 |           0.9998  |              1       |
| distance_bins | grid2d   | phenoA  | z-norm           | MAE vs known true risk    |        44 |     0.8409 |                0.03572  |            0.02613   |             0.04901   |        6.47e-10  |           5.176e-09 |           1       |              1       |
| distance_bins | grid2d   | phenoA  | PGS + PCs        | RMSE vs known true risk   |        44 |     0.8182 |                0.007222 |            0.004839  |             0.0104    |        6.667e-07 |           2.333e-06 |           1       |              1       |
| distance_bins | grid2d   | phenoA  | z-norm           | RMSE vs known true risk   |        44 |     1      |                0.0355   |            0.02642   |             0.04857   |        5.684e-14 |           3.183e-12 |           1       |              1       |
| distance_bins | grid2d   | phenoB  | PGS + PCs        | Abs. prevalence error     |        44 |     0.5682 |                0.005891 |           -0.0001743 |             0.01171   |        0.02497   |           0.0378    |           0.9757  |              1       |
| distance_bins | grid2d   | phenoB  | z-norm           | Abs. prevalence error     |        44 |     0.8864 |                0.04939  |            0.03546   |             0.0655    |        8.418e-10 |           5.893e-09 |           1       |              1       |
| distance_bins | grid2d   | phenoB  | PGS + PCs        | Abs. true-slope error     |        44 |     0.2955 |               -0.05267  |           -0.08389   |            -0.02112   |        0.999     |           0.999     |           0.00101 |              0.05656 |
| distance_bins | grid2d   | phenoB  | z-norm           | Abs. true-slope error     |        44 |     0.3864 |               -0.0535   |           -0.1005    |            -0.006476  |        0.9764    |           0.9988    |           0.02428 |              0.4477  |
| distance_bins | grid2d   | phenoB  | PGS + PCs        | AUC                       |        44 |     0.6364 |                0.01423  |           -0.01895   |             0.04693   |        0.06142   |           0.08517   |           0.94    |              1       |
| distance_bins | grid2d   | phenoB  | z-norm           | AUC                       |        44 |     0.75   |                0.04447  |           -0.005058  |             0.07686   |        0.000409  |           0.0008484 |           0.9996  |              1       |
| distance_bins | grid2d   | phenoB  | PGS + PCs        | Brier Skill Score (BSS)   |        44 |     0.7273 |                0.009566 |           -0.003401  |             0.01918   |        0.000926  |           0.001673  |           0.9991  |              1       |
| distance_bins | grid2d   | phenoB  | z-norm           | Brier Skill Score (BSS)   |        44 |     0.9091 |                0.2063   |            0.09862   |             0.4945    |        9.607e-12 |           1.345e-10 |           1       |              1       |
| distance_bins | grid2d   | phenoB  | PGS + PCs        | Liability-scale pseudo-R2 |        44 |     0.6591 |                0.04072  |            0.02291   |             0.06716   |        0.0006206 |           0.001198  |           0.9994  |              1       |
| distance_bins | grid2d   | phenoB  | z-norm           | Liability-scale pseudo-R2 |        44 |     0.7727 |                0.1085   |            0.0756    |             0.1483    |        1.045e-07 |           4.502e-07 |           1       |              1       |
| distance_bins | grid2d   | phenoB  | PGS + PCs        | MAE vs known true risk    |        44 |     0.7273 |                0.004222 |            0.0005666 |             0.007242  |        0.0007767 |           0.00145   |           0.9993  |              1       |
| distance_bins | grid2d   | phenoB  | z-norm           | MAE vs known true risk    |        44 |     0.9091 |                0.03265  |            0.02374   |             0.04572   |        4.331e-11 |           4.851e-10 |           1       |              1       |
| distance_bins | grid2d   | phenoB  | PGS + PCs        | RMSE vs known true risk   |        44 |     0.7045 |                0.004979 |            0.001762  |             0.007956  |        0.0005167 |           0.001033  |           0.9995  |              1       |
| distance_bins | grid2d   | phenoB  | z-norm           | RMSE vs known true risk   |        44 |     0.9545 |                0.0335   |            0.02439   |             0.0465    |        7.958e-13 |           1.486e-11 |           1       |              1       |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Abs. prevalence error     |        40 |     0.825  |                0.03576  |            0.0234    |             0.05046   |        1.019e-06 |           3.356e-06 |           1       |              1       |
| distance_bins | serial1d | phenoA  | z-norm           | Abs. prevalence error     |        40 |     0.8    |                0.05312  |            0.03778   |             0.06921   |        5.817e-08 |           2.962e-07 |           1       |              1       |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Abs. true-slope error     |        40 |     0.625  |                0.02621  |           -0.005189  |             0.05209   |        0.01972   |           0.03068   |           0.9809  |              1       |
| distance_bins | serial1d | phenoA  | z-norm           | Abs. true-slope error     |        40 |     0.475  |               -0.002449 |           -0.04331   |             0.03557   |        0.566     |           0.6603    |           0.4393  |              1       |
| distance_bins | serial1d | phenoA  | PGS + PCs        | AUC                       |        40 |     0.6    |                0.00146  |           -0.002119  |             0.00514   |        0.2025    |           0.252     |           0.8013  |              1       |
| distance_bins | serial1d | phenoA  | z-norm           | AUC                       |        40 |     0.675  |                0.002485 |           -0.0006698 |             0.005863  |        0.04862   |           0.06981   |           0.9528  |              1       |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |        40 |     0.775  |                0.03181  |            0.01829   |             0.05745   |        2.923e-06 |           8.614e-06 |           1       |              1       |
| distance_bins | serial1d | phenoA  | z-norm           | Brier Skill Score (BSS)   |        40 |     0.875  |                0.06823  |            0.04426   |             0.121     |        1.346e-08 |           7.537e-08 |           1       |              1       |
| distance_bins | serial1d | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |        40 |     0.425  |               -0.005397 |           -0.01175   |            -0.001251  |        0.9359    |           0.9988    |           0.06578 |              0.5694  |
| distance_bins | serial1d | phenoA  | z-norm           | Liability-scale pseudo-R2 |        40 |     0.35   |               -0.005676 |           -0.01208   |            -0.002064  |        0.9809    |           0.9988    |           0.01972 |              0.4477  |
| distance_bins | serial1d | phenoA  | PGS + PCs        | MAE vs known true risk    |        40 |     0.7    |                0.01066  |            0.004934  |             0.02037   |        0.001196  |           0.00203   |           0.9989  |              1       |
| distance_bins | serial1d | phenoA  | z-norm           | MAE vs known true risk    |        40 |     0.7    |                0.01927  |            0.01076   |             0.03197   |        0.0003777 |           0.0008135 |           0.9996  |              1       |
| distance_bins | serial1d | phenoA  | PGS + PCs        | RMSE vs known true risk   |        40 |     0.825  |                0.01166  |            0.007112  |             0.01933   |        2.264e-06 |           7.043e-06 |           1       |              1       |
| distance_bins | serial1d | phenoA  | z-norm           | RMSE vs known true risk   |        40 |     0.8    |                0.02118  |            0.01457   |             0.03073   |        1.411e-07 |           5.646e-07 |           1       |              1       |
| distance_bins | serial1d | phenoB  | PGS + PCs        | Abs. prevalence error     |        40 |     0.675  |                0.01352  |            0.006506  |             0.02199   |        0.001526  |           0.002513  |           0.9985  |              1       |
| distance_bins | serial1d | phenoB  | z-norm           | Abs. prevalence error     |        40 |     0.825  |                0.03971  |            0.02726   |             0.05588   |        1.181e-08 |           7.349e-08 |           1       |              1       |
| distance_bins | serial1d | phenoB  | PGS + PCs        | Abs. true-slope error     |        40 |     0.675  |                0.01681  |           -0.008871  |             0.04116   |        0.0385    |           0.05674   |           0.9626  |              1       |
| distance_bins | serial1d | phenoB  | z-norm           | Abs. true-slope error     |        40 |     0.475  |                0.009197 |           -0.01071   |             0.0318    |        0.3523    |           0.4198    |           0.6526  |              1       |
| distance_bins | serial1d | phenoB  | PGS + PCs        | AUC                       |        40 |     0.625  |                0.003425 |           -0.002656  |             0.008632  |        0.06236   |           0.08517   |           0.9393  |              1       |
| distance_bins | serial1d | phenoB  | z-norm           | AUC                       |        40 |     0.6    |                0.002999 |           -0.001374  |             0.008996  |        0.1383    |           0.176     |           0.8647  |              1       |
| distance_bins | serial1d | phenoB  | PGS + PCs        | Brier Skill Score (BSS)   |        40 |     0.7    |                0.0109   |            0.00541   |             0.01992   |        0.0009798 |           0.001715  |           0.9991  |              1       |
| distance_bins | serial1d | phenoB  | z-norm           | Brier Skill Score (BSS)   |        40 |     0.725  |                0.05432  |            0.02551   |             0.117     |        1.648e-05 |           4.614e-05 |           1       |              1       |
| distance_bins | serial1d | phenoB  | PGS + PCs        | Liability-scale pseudo-R2 |        40 |     0.45   |               -0.003034 |           -0.006996  |            -0.0003826 |        0.9062    |           0.9988    |           0.0961  |              0.6727  |
| distance_bins | serial1d | phenoB  | z-norm           | Liability-scale pseudo-R2 |        40 |     0.45   |               -0.003595 |           -0.007954  |            -0.001009  |        0.9307    |           0.9988    |           0.07117 |              0.5694  |
| distance_bins | serial1d | phenoB  | PGS + PCs        | MAE vs known true risk    |        40 |     0.5    |                0.003317 |            0.0004137 |             0.007788  |        0.214     |           0.2605    |           0.7899  |              1       |
| distance_bins | serial1d | phenoB  | z-norm           | MAE vs known true risk    |        40 |     0.75   |                0.01401  |            0.006929  |             0.02658   |        0.0003016 |           0.0006757 |           0.9997  |              1       |
| distance_bins | serial1d | phenoB  | PGS + PCs        | RMSE vs known true risk   |        40 |     0.65   |                0.004318 |            0.002126  |             0.00795   |        0.002435  |           0.003897  |           0.9977  |              1       |
| distance_bins | serial1d | phenoB  | z-norm           | RMSE vs known true risk   |        40 |     0.8    |                0.0154   |            0.009615  |             0.02537   |        9.134e-08 |           4.262e-07 |           1       |              1       |
| global_test   | grid2d   | phenoA  | PGS + PCs        | AUC                       |         4 |     1      |                0.02432  |            0.02106   |             0.02925   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | z-norm           | AUC                       |         4 |     1      |                0.1463   |            0.075     |             0.1941    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |         4 |     1      |                0.02118  |            0.01528   |             0.02726   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | z-norm           | Brier Skill Score (BSS)   |         4 |     1      |                0.06465  |            0.03832   |             0.09098   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |         4 |     1      |                0.03866  |            0.01936   |             0.04958   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | z-norm           | Liability-scale pseudo-R2 |         4 |     1      |                0.2068   |            0.08732   |             0.3977    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | PGS + PCs        | MAE vs known true risk    |         4 |     1      |                0.007361 |            0.003884  |             0.009633  |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | z-norm           | MAE vs known true risk    |         4 |     1      |                0.02656  |            0.01376   |             0.03936   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | PGS + PCs        | RMSE vs known true risk   |         4 |     1      |                0.008255 |            0.005218  |             0.01129   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoA  | z-norm           | RMSE vs known true risk   |         4 |     1      |                0.02644  |            0.01372   |             0.03916   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | PGS + PCs        | AUC                       |         4 |     1      |                0.0252   |            0.02343   |             0.02687   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | z-norm           | AUC                       |         4 |     1      |                0.1362   |            0.09266   |             0.2022    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | PGS + PCs        | Brier Skill Score (BSS)   |         4 |     1      |                0.0175   |            0.01035   |             0.02464   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | z-norm           | Brier Skill Score (BSS)   |         4 |     1      |                0.0599   |            0.03681   |             0.09522   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | PGS + PCs        | Liability-scale pseudo-R2 |         4 |     1      |                0.06329  |            0.04659   |             0.07269   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | z-norm           | Liability-scale pseudo-R2 |         4 |     1      |                0.2673   |            0.1385    |             0.4643    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | PGS + PCs        | MAE vs known true risk    |         4 |     1      |                0.006393 |            0.003175  |             0.008497  |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | z-norm           | MAE vs known true risk    |         4 |     1      |                0.02392  |            0.01304   |             0.04089   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | PGS + PCs        | RMSE vs known true risk   |         4 |     1      |                0.007364 |            0.004105  |             0.01049   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | grid2d   | phenoB  | z-norm           | RMSE vs known true risk   |         4 |     1      |                0.02472  |            0.01345   |             0.04174   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | PGS + PCs        | AUC                       |         4 |     1      |                0.0448   |            0.02962   |             0.07388   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | z-norm           | AUC                       |         4 |     1      |                0.07587  |            0.03875   |             0.1019    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | PGS + PCs        | Brier Skill Score (BSS)   |         4 |     1      |                0.02504  |            0.01519   |             0.0405    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | z-norm           | Brier Skill Score (BSS)   |         4 |     1      |                0.04251  |            0.03029   |             0.05017   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | PGS + PCs        | Liability-scale pseudo-R2 |         4 |     0.5    |                0.02047  |           -0.01356   |             0.05449   |        0.3125    |           0.3125    |           0.8125  |              1       |
| global_test   | serial1d | phenoA  | z-norm           | Liability-scale pseudo-R2 |         4 |     0.75   |                0.1033   |            0.004869  |             0.2017    |        0.125     |           0.1282    |           0.9375  |              1       |
| global_test   | serial1d | phenoA  | PGS + PCs        | MAE vs known true risk    |         4 |     1      |                0.01066  |            0.006115  |             0.01857   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | z-norm           | MAE vs known true risk    |         4 |     1      |                0.01927  |            0.01105   |             0.03055   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | PGS + PCs        | RMSE vs known true risk   |         4 |     1      |                0.0112   |            0.005643  |             0.01983   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoA  | z-norm           | RMSE vs known true risk   |         4 |     1      |                0.01886  |            0.01252   |             0.02914   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | PGS + PCs        | AUC                       |         4 |     1      |                0.01557  |            0.00674   |             0.0244    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | z-norm           | AUC                       |         4 |     1      |                0.06133  |            0.01985   |             0.1225    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | PGS + PCs        | Brier Skill Score (BSS)   |         4 |     1      |                0.008438 |            0.002597  |             0.01428   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | z-norm           | Brier Skill Score (BSS)   |         4 |     1      |                0.03066  |            0.01056   |             0.05077   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | PGS + PCs        | Liability-scale pseudo-R2 |         4 |     1      |                0.03538  |            0.004054  |             0.0667    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | z-norm           | Liability-scale pseudo-R2 |         4 |     1      |                0.1537   |            0.05108   |             0.2563    |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | PGS + PCs        | MAE vs known true risk    |         4 |     1      |                0.003317 |            0.0002568 |             0.006377  |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | z-norm           | MAE vs known true risk    |         4 |     1      |                0.01401  |            0.004272  |             0.02375   |        0.0625    |           0.06757   |           1       |              1       |
| global_test   | serial1d | phenoB  | PGS + PCs        | RMSE vs known true risk   |         4 |     0.75   |                0.004117 |            2.498e-05 |             0.00821   |        0.125     |           0.1282    |           0.9375  |              1       |
| global_test   | serial1d | phenoB  | z-norm           | RMSE vs known true risk   |         4 |     1      |                0.01384  |            0.005785  |             0.0219    |        0.0625    |           0.06757   |           1       |              1       |


## Method-value summaries

| source        | metric               | metric_label              | method   |   n |      mean |    median |       sd | dem      | pheno   |
|:--------------|:---------------------|:--------------------------|:---------|----:|----------:|----------:|---------:|:---------|:--------|
| distance_bins | abs_prevalence_error | Abs. prevalence error     | gamfit   |  44 |  0.01851  |  0.01634  | 0.01445  | grid2d   | phenoA  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | linpc    |  44 |  0.03389  |  0.03093  | 0.02537  | grid2d   | phenoA  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | znorm    |  44 |  0.07275  |  0.05262  | 0.05715  | grid2d   | phenoA  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | gamfit   |  44 |  0.2025   |  0.1828   | 0.1355   | grid2d   | phenoA  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | linpc    |  44 |  0.1556   |  0.158    | 0.1185   | grid2d   | phenoA  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | znorm    |  44 |  0.1478   |  0.1029   | 0.1414   | grid2d   | phenoA  |
| distance_bins | auc                  | AUC                       | gamfit   |  44 |  0.6089   |  0.6008   | 0.07887  | grid2d   | phenoA  |
| distance_bins | auc                  | AUC                       | linpc    |  44 |  0.5959   |  0.5985   | 0.0792   | grid2d   | phenoA  |
| distance_bins | auc                  | AUC                       | znorm    |  44 |  0.5551   |  0.5402   | 0.07338  | grid2d   | phenoA  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | gamfit   |  44 | -0.003869 |  0.003499 | 0.1239   | grid2d   | phenoA  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | linpc    |  44 | -0.02464  | -0.007894 | 0.125    | grid2d   | phenoA  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | znorm    |  44 | -0.2324   | -0.05334  | 0.738    | grid2d   | phenoA  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |  44 |  0.149    |  0.1002   | 0.1432   | grid2d   | phenoA  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |  44 |  0.1283   |  0.0914   | 0.1327   | grid2d   | phenoA  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |  44 |  0.06478  |  0.008893 | 0.1254   | grid2d   | phenoA  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | gamfit   |  44 |  0.1084   |  0.1074   | 0.04163  | grid2d   | phenoA  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | linpc    |  44 |  0.1142   |  0.115    | 0.04494  | grid2d   | phenoA  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | znorm    |  44 |  0.1442   |  0.1381   | 0.03326  | grid2d   | phenoA  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | gamfit   |  44 |  0.1444   |  0.1437   | 0.04941  | grid2d   | phenoA  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | linpc    |  44 |  0.1516   |  0.1508   | 0.05353  | grid2d   | phenoA  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | znorm    |  44 |  0.1799   |  0.1709   | 0.04612  | grid2d   | phenoA  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | gamfit   |  44 |  0.01816  |  0.0127   | 0.016    | grid2d   | phenoB  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | linpc    |  44 |  0.02405  |  0.02138  | 0.0191   | grid2d   | phenoB  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | znorm    |  44 |  0.06755  |  0.04619  | 0.05682  | grid2d   | phenoB  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | gamfit   |  44 |  0.1997   |  0.1929   | 0.119    | grid2d   | phenoB  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | linpc    |  44 |  0.1471   |  0.126    | 0.1084   | grid2d   | phenoB  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | znorm    |  44 |  0.1462   |  0.08703  | 0.1348   | grid2d   | phenoB  |
| distance_bins | auc                  | AUC                       | gamfit   |  44 |  0.6056   |  0.631    | 0.09962  | grid2d   | phenoB  |
| distance_bins | auc                  | AUC                       | linpc    |  44 |  0.5913   |  0.5988   | 0.07664  | grid2d   | phenoB  |
| distance_bins | auc                  | AUC                       | znorm    |  44 |  0.5611   |  0.5415   | 0.08093  | grid2d   | phenoB  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | gamfit   |  44 |  0.02865  |  0.04823  | 0.1536   | grid2d   | phenoB  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | linpc    |  44 |  0.01909  |  0.01583  | 0.1525   | grid2d   | phenoB  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | znorm    |  44 | -0.1776   | -0.04246  | 0.5136   | grid2d   | phenoB  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |  44 |  0.1732   |  0.1406   | 0.134    | grid2d   | phenoB  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |  44 |  0.1325   |  0.1118   | 0.1311   | grid2d   | phenoB  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |  44 |  0.06473  |  0.008885 | 0.1254   | grid2d   | phenoB  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | gamfit   |  44 |  0.109    |  0.1063   | 0.03717  | grid2d   | phenoB  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | linpc    |  44 |  0.1132   |  0.1185   | 0.04133  | grid2d   | phenoB  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | znorm    |  44 |  0.1417   |  0.1374   | 0.03149  | grid2d   | phenoB  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | gamfit   |  44 |  0.1442   |  0.143    | 0.04579  | grid2d   | phenoB  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | linpc    |  44 |  0.1492   |  0.1519   | 0.05086  | grid2d   | phenoB  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | znorm    |  44 |  0.1777   |  0.1681   | 0.04392  | grid2d   | phenoB  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | gamfit   |  40 |  0.02231  |  0.01741  | 0.01762  | serial1d | phenoA  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | linpc    |  40 |  0.05808  |  0.04816  | 0.04078  | serial1d | phenoA  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | znorm    |  40 |  0.07543  |  0.07423  | 0.04584  | serial1d | phenoA  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | gamfit   |  40 |  0.1745   |  0.1488   | 0.1202   | serial1d | phenoA  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | linpc    |  40 |  0.2007   |  0.2012   | 0.1322   | serial1d | phenoA  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | znorm    |  40 |  0.172    |  0.1399   | 0.1454   | serial1d | phenoA  |
| distance_bins | auc                  | AUC                       | gamfit   |  40 |  0.6163   |  0.6186   | 0.06735  | serial1d | phenoA  |
| distance_bins | auc                  | AUC                       | linpc    |  40 |  0.6148   |  0.6138   | 0.06677  | serial1d | phenoA  |
| distance_bins | auc                  | AUC                       | znorm    |  40 |  0.6138   |  0.6134   | 0.0662   | serial1d | phenoA  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | gamfit   |  40 | -0.006601 | -0.002392 | 0.08846  | serial1d | phenoA  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | linpc    |  40 | -0.03842  | -0.03082  | 0.1084   | serial1d | phenoA  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | znorm    |  40 | -0.07483  | -0.04987  | 0.1632   | serial1d | phenoA  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |  40 |  0.1543   |  0.168    | 0.1216   | serial1d | phenoA  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |  40 |  0.1597   |  0.1625   | 0.1217   | serial1d | phenoA  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |  40 |  0.16     |  0.17     | 0.122    | serial1d | phenoA  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | gamfit   |  40 |  0.1371   |  0.1572   | 0.04698  | serial1d | phenoA  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | linpc    |  40 |  0.1478   |  0.1636   | 0.04503  | serial1d | phenoA  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | znorm    |  40 |  0.1564   |  0.1628   | 0.03402  | serial1d | phenoA  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | gamfit   |  40 |  0.1748   |  0.1938   | 0.05245  | serial1d | phenoA  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | linpc    |  40 |  0.1864   |  0.2045   | 0.05283  | serial1d | phenoA  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | znorm    |  40 |  0.1959   |  0.2091   | 0.04558  | serial1d | phenoA  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | gamfit   |  40 |  0.02278  |  0.0183   | 0.01904  | serial1d | phenoB  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | linpc    |  40 |  0.0363   |  0.02281  | 0.03363  | serial1d | phenoB  |
| distance_bins | abs_prevalence_error | Abs. prevalence error     | znorm    |  40 |  0.06249  |  0.0468   | 0.04721  | serial1d | phenoB  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | gamfit   |  40 |  0.1581   |  0.1545   | 0.1077   | serial1d | phenoB  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | linpc    |  40 |  0.1749   |  0.1693   | 0.1179   | serial1d | phenoB  |
| distance_bins | abs_slope_error      | Abs. true-slope error     | znorm    |  40 |  0.1673   |  0.1407   | 0.1469   | serial1d | phenoB  |
| distance_bins | auc                  | AUC                       | gamfit   |  40 |  0.6334   |  0.6305   | 0.09224  | serial1d | phenoB  |
| distance_bins | auc                  | AUC                       | linpc    |  40 |  0.63     |  0.6293   | 0.09026  | serial1d | phenoB  |
| distance_bins | auc                  | AUC                       | znorm    |  40 |  0.6304   |  0.6241   | 0.08884  | serial1d | phenoB  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | gamfit   |  40 |  0.04077  |  0.0315   | 0.1334   | serial1d | phenoB  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | linpc    |  40 |  0.02986  |  0.01986  | 0.1324   | serial1d | phenoB  |
| distance_bins | bss                  | Brier Skill Score (BSS)   | znorm    |  40 | -0.01355  | -0.01964  | 0.1288   | serial1d | phenoB  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |  40 |  0.1563   |  0.1686   | 0.121    | serial1d | phenoB  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |  40 |  0.1593   |  0.1625   | 0.1217   | serial1d | phenoB  |
| distance_bins | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |  40 |  0.1599   |  0.17     | 0.1219   | serial1d | phenoB  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | gamfit   |  40 |  0.1331   |  0.1403   | 0.04508  | serial1d | phenoB  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | linpc    |  40 |  0.1364   |  0.1501   | 0.0454   | serial1d | phenoB  |
| distance_bins | mae_true_risk        | MAE vs known true risk    | znorm    |  40 |  0.1471   |  0.1513   | 0.03956  | serial1d | phenoB  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | gamfit   |  40 |  0.1707   |  0.1828   | 0.05019  | serial1d | phenoB  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | linpc    |  40 |  0.175    |  0.186    | 0.05089  | serial1d | phenoB  |
| distance_bins | rmse_true_risk       | RMSE vs known true risk   | znorm    |  40 |  0.1861   |  0.1887   | 0.04838  | serial1d | phenoB  |
| global_test   | auc                  | AUC                       | gamfit   |   4 |  0.6892   |  0.6887   | 0.02806  | grid2d   | phenoA  |
| global_test   | auc                  | AUC                       | linpc    |   4 |  0.6649   |  0.6677   | 0.03012  | grid2d   | phenoA  |
| global_test   | auc                  | AUC                       | znorm    |   4 |  0.543    |  0.5226   | 0.05398  | grid2d   | phenoA  |
| global_test   | bss                  | Brier Skill Score (BSS)   | gamfit   |   4 |  0.05336  |  0.05924  | 0.02184  | grid2d   | phenoA  |
| global_test   | bss                  | Brier Skill Score (BSS)   | linpc    |   4 |  0.03217  |  0.03419  | 0.01671  | grid2d   | phenoA  |
| global_test   | bss                  | Brier Skill Score (BSS)   | znorm    |   4 | -0.01129  | -0.01274  | 0.0123   | grid2d   | phenoA  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |   4 |  0.2355   |  0.1822   | 0.1577   | grid2d   | phenoA  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |   4 |  0.1968   |  0.1545   | 0.1476   | grid2d   | phenoA  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |   4 |  0.02867  |  0.00847  | 0.04657  | grid2d   | phenoA  |
| global_test   | mae_true_risk        | MAE vs known true risk    | gamfit   |   4 |  0.115    |  0.1111   | 0.01912  | grid2d   | phenoA  |
| global_test   | mae_true_risk        | MAE vs known true risk    | linpc    |   4 |  0.1224   |  0.1208   | 0.02019  | grid2d   | phenoA  |
| global_test   | mae_true_risk        | MAE vs known true risk    | znorm    |   4 |  0.1416   |  0.1505   | 0.0227   | grid2d   | phenoA  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | gamfit   |   4 |  0.1574   |  0.1524   | 0.02358  | grid2d   | phenoA  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | linpc    |   4 |  0.1656   |  0.1637   | 0.02401  | grid2d   | phenoA  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | znorm    |   4 |  0.1838   |  0.1916   | 0.02488  | grid2d   | phenoA  |
| global_test   | auc                  | AUC                       | gamfit   |   4 |  0.6832   |  0.6728   | 0.03926  | grid2d   | phenoB  |
| global_test   | auc                  | AUC                       | linpc    |   4 |  0.658    |  0.6477   | 0.03876  | grid2d   | phenoB  |
| global_test   | auc                  | AUC                       | znorm    |   4 |  0.547    |  0.5252   | 0.05146  | grid2d   | phenoB  |
| global_test   | bss                  | Brier Skill Score (BSS)   | gamfit   |   4 |  0.06279  |  0.0523   | 0.03459  | grid2d   | phenoB  |
| global_test   | bss                  | Brier Skill Score (BSS)   | linpc    |   4 |  0.0453   |  0.03527  | 0.02721  | grid2d   | phenoB  |
| global_test   | bss                  | Brier Skill Score (BSS)   | znorm    |   4 |  0.002891 |  0.004556 | 0.004786 | grid2d   | phenoB  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |   4 |  0.296    |  0.2342   | 0.1631   | grid2d   | phenoB  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |   4 |  0.2327   |  0.1777   | 0.1538   | grid2d   | phenoB  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |   4 |  0.02867  |  0.008462 | 0.04657  | grid2d   | phenoB  |
| global_test   | mae_true_risk        | MAE vs known true risk    | gamfit   |   4 |  0.1145   |  0.1115   | 0.02086  | grid2d   | phenoB  |
| global_test   | mae_true_risk        | MAE vs known true risk    | linpc    |   4 |  0.1209   |  0.12     | 0.02174  | grid2d   | phenoB  |
| global_test   | mae_true_risk        | MAE vs known true risk    | znorm    |   4 |  0.1385   |  0.1463   | 0.02421  | grid2d   | phenoB  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | gamfit   |   4 |  0.1556   |  0.1517   | 0.02449  | grid2d   | phenoB  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | linpc    |   4 |  0.163    |  0.1619   | 0.02483  | grid2d   | phenoB  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | znorm    |   4 |  0.1803   |  0.1877   | 0.02534  | grid2d   | phenoB  |
| global_test   | auc                  | AUC                       | gamfit   |   4 |  0.688    |  0.6922   | 0.02916  | serial1d | phenoA  |
| global_test   | auc                  | AUC                       | linpc    |   4 |  0.6432   |  0.6385   | 0.04017  | serial1d | phenoA  |
| global_test   | auc                  | AUC                       | znorm    |   4 |  0.6121   |  0.6179   | 0.05492  | serial1d | phenoA  |
| global_test   | bss                  | Brier Skill Score (BSS)   | gamfit   |   4 |  0.05528  |  0.04659  | 0.028    | serial1d | phenoA  |
| global_test   | bss                  | Brier Skill Score (BSS)   | linpc    |   4 |  0.03024  |  0.02602  | 0.01345  | serial1d | phenoA  |
| global_test   | bss                  | Brier Skill Score (BSS)   | znorm    |   4 |  0.01277  |  0.01117  | 0.02508  | serial1d | phenoA  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |   4 |  0.2362   |  0.2495   | 0.06491  | serial1d | phenoA  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |   4 |  0.2157   |  0.237    | 0.07727  | serial1d | phenoA  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |   4 |  0.1329   |  0.1371   | 0.101    | serial1d | phenoA  |
| global_test   | mae_true_risk        | MAE vs known true risk    | gamfit   |   4 |  0.1371   |  0.1424   | 0.02611  | serial1d | phenoA  |
| global_test   | mae_true_risk        | MAE vs known true risk    | linpc    |   4 |  0.1478   |  0.158    | 0.02736  | serial1d | phenoA  |
| global_test   | mae_true_risk        | MAE vs known true risk    | znorm    |   4 |  0.1564   |  0.1635   | 0.03001  | serial1d | phenoA  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | gamfit   |   4 |  0.1803   |  0.1887   | 0.03059  | serial1d | phenoA  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | linpc    |   4 |  0.1915   |  0.2057   | 0.03242  | serial1d | phenoA  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | znorm    |   4 |  0.1992   |  0.2098   | 0.03139  | serial1d | phenoA  |
| global_test   | auc                  | AUC                       | gamfit   |   4 |  0.6841   |  0.6725   | 0.02501  | serial1d | phenoB  |
| global_test   | auc                  | AUC                       | linpc    |   4 |  0.6685   |  0.6567   | 0.03248  | serial1d | phenoB  |
| global_test   | auc                  | AUC                       | znorm    |   4 |  0.6227   |  0.6294   | 0.07207  | serial1d | phenoB  |
| global_test   | bss                  | Brier Skill Score (BSS)   | gamfit   |   4 |  0.07421  |  0.08091  | 0.03533  | serial1d | phenoB  |
| global_test   | bss                  | Brier Skill Score (BSS)   | linpc    |   4 |  0.06578  |  0.07035  | 0.03807  | serial1d | phenoB  |
| global_test   | bss                  | Brier Skill Score (BSS)   | znorm    |   4 |  0.04355  |  0.03951  | 0.03789  | serial1d | phenoB  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | gamfit   |   4 |  0.2866   |  0.2877   | 0.05331  | serial1d | phenoB  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | linpc    |   4 |  0.2512   |  0.2725   | 0.06386  | serial1d | phenoB  |
| global_test   | liability_pseudo_r2  | Liability-scale pseudo-R2 | znorm    |   4 |  0.1329   |  0.1371   | 0.1009   | serial1d | phenoB  |
| global_test   | mae_true_risk        | MAE vs known true risk    | gamfit   |   4 |  0.1331   |  0.1439   | 0.0312   | serial1d | phenoB  |
| global_test   | mae_true_risk        | MAE vs known true risk    | linpc    |   4 |  0.1364   |  0.1503   | 0.03273  | serial1d | phenoB  |
| global_test   | mae_true_risk        | MAE vs known true risk    | znorm    |   4 |  0.1471   |  0.1556   | 0.03872  | serial1d | phenoB  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | gamfit   |   4 |  0.1753   |  0.1869   | 0.03417  | serial1d | phenoB  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | linpc    |   4 |  0.1794   |  0.1951   | 0.0361   | serial1d | phenoB  |
| global_test   | rmse_true_risk       | RMSE vs known true risk   | znorm    |   4 |  0.1891   |  0.1983   | 0.03917  | serial1d | phenoB  |
