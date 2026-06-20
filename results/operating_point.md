# Fixed operating-point metrics (target FPR = 5%, whitened, mean across classes)

Threshold calibrated on the held-out `validation` split (normal-only, never seen by the bank or the test scorer); applied FIXED to `test`. Cost columns: mean per-scan cost under illustrative FN:FP unit-cost ratios (a missed defect vs a false alarm) -- see the module docstring.

| scorer | precision | recall | F1 | realized FPR (test) | cost 5:1 | cost 20:1 |
|---|---|---|---|---|---|---|
| knn | 0.957 | 0.435 | 0.532 | 0.064 | 2.255 | 8.976 |
| modern_hopfield | 0.961 | 0.435 | 0.529 | 0.059 | 2.256 | 8.985 |
| kde | 0.928 | 0.079 | 0.141 | 0.029 | 3.638 | 14.535 |
| ocsvm | 0.923 | 0.307 | 0.415 | 0.054 | 2.761 | 11.011 |
| centroid | 0.814 | 0.116 | 0.195 | 0.106 | 3.511 | 13.976 |
| classical_hopfield | 0.883 | 0.257 | 0.364 | 0.069 | 2.938 | 11.708 |
