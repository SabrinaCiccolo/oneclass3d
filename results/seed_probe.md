# Corruption seed-probe - mitigation cells, multiple corruption draws

Each `corruption_grid.csv` cell is a single deterministic corruption draw per
scan; this re-scores the mitigation probe's worst cells at [0, 1, 2] independent draws and reports the
across-seed mean and min-max band, so a cell's wobble can be told apart from a
real effect.

| class | cell | regime | scorer (rep) | n seeds | mean | min | max | band |
|---|---|---|---|---|---|---|---|---|
| bagel | gauss3 | corrupted_bank | knn (raw) | 3 | 0.455 | 0.415 | 0.528 | 0.114 |
| bagel | gauss3 | corrupted_bank | modern_hopfield_auto (whitened) | 3 | 0.421 | 0.343 | 0.483 | 0.141 |
| bagel | gauss3 | clean_bank | knn (raw) | 3 | 0.495 | 0.461 | 0.545 | 0.084 |
| bagel | gauss3 | clean_bank | modern_hopfield_auto (whitened) | 3 | 0.497 | 0.464 | 0.543 | 0.079 |
| bagel | gauss2 | clean_bank | knn (raw) | 3 | 0.862 | 0.805 | 0.906 | 0.101 |
| bagel | gauss2 | clean_bank | modern_hopfield_auto (whitened) | 3 | 0.515 | 0.478 | 0.547 | 0.069 |
