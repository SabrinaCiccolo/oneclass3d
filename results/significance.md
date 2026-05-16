# Pairwise significance vs kNN (mean across 10 classes, paired bootstrap)

Baseline: `knn`. 95% CI and two-sided p-value on the PAIRED AUROC delta (1000 paired scan-resamples/class, matching-index averaged across classes -- see the module docstring). 'significant' = the 95% CI excludes zero (alpha=0.05, not corrected for multiple comparisons across the 5 scorers x 2 reps = 10 tests here).

| rep | scorer | delta AUROC | 95% CI | p | significant? |
|---|---|---|---|---|---|
| raw | classical_hopfield | -0.1752 | [-0.2212, -0.1307] | 0.0000 | yes |
| raw | kde | -0.1751 | [-0.2203, -0.1305] | 0.0000 | yes |
| raw | centroid | -0.1744 | [-0.2207, -0.1292] | 0.0000 | yes |
| raw | ocsvm | -0.1492 | [-0.1929, -0.1059] | 0.0000 | yes |
| raw | modern_hopfield | -0.0835 | [-0.1100, -0.0567] | 0.0000 | yes |
| whitened | centroid | -0.2760 | [-0.3248, -0.2255] | 0.0000 | yes |
| whitened | kde | -0.2676 | [-0.3152, -0.2152] | 0.0000 | yes |
| whitened | classical_hopfield | -0.1458 | [-0.1907, -0.1024] | 0.0000 | yes |
| whitened | ocsvm | -0.0727 | [-0.1157, -0.0325] | 0.0000 | yes |
| whitened | modern_hopfield | +0.0037 | [-0.0013, +0.0092] | 0.1520 | no |
