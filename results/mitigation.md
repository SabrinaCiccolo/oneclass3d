# Mitigation probe - worst corruption cells, cleanup stages toggled

The pipeline's own cleanup stages are the natural mitigations: `nr` = RANSAC
plane removal off, `nc` = DBSCAN cleanup off, `nrnc` = both off (applied to
bank and test alike; variant caches via the same extract path). `recovery` =
variant AUROC - default-pipeline AUROC on the same corrupted cell.

| class | cell | regime | scorer (rep) | clean | corrupted (default) | nr | nc | nrnc | best recovery |
|---|---|---|---|---|---|---|---|---|---|
| bagel | gauss3 | corrupted_bank | knn (raw) | 0.843 | 0.415 | 0.532 | 0.457 | 0.499 | +0.118 |
| bagel | gauss3 | corrupted_bank | modern_hopfield_auto (whitened) | 0.871 | 0.483 | 0.604 | 0.575 | 0.632 | +0.148 |
| bagel | gauss3 | clean_bank | knn (raw) | 0.843 | 0.481 | 0.577 | 0.496 | 0.424 | +0.096 |
| bagel | gauss3 | clean_bank | modern_hopfield_auto (whitened) | 0.871 | 0.484 | 0.563 | 0.794 | 0.391 | +0.310 |
| bagel | gauss2 | clean_bank | knn (raw) | 0.843 | 0.906 | 0.557 | 0.890 | 0.777 | -0.016 |
| bagel | gauss2 | clean_bank | modern_hopfield_auto (whitened) | 0.871 | 0.519 | 0.381 | 0.612 | 0.708 | +0.190 |
