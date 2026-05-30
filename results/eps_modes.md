# DBSCAN eps modes - worst corruption cells

`default` = fixed eps (`cleanup.dbscan_eps`), `nc` = DBSCAN cleanup off, `rel` =
eps scaled by the bbox diagonal, `adaptive` = per-scan k-distance eps. Bank and
test are extracted with the same variant.

| class | cell | regime | scorer (rep) | default | nc | rel | adaptive |
|---|---|---|---|---|---|---|---|
| bagel | gauss3 | corrupted_bank | knn (raw) | 0.415 | 0.457 | 0.458 | 0.425 |
| bagel | gauss3 | corrupted_bank | modern_hopfield_auto (whitened) | 0.483 | 0.575 | 0.580 | 0.613 |
| bagel | gauss3 | clean_bank | knn (raw) | 0.481 | 0.496 | 0.465 | 0.554 |
| bagel | gauss3 | clean_bank | modern_hopfield_auto (whitened) | 0.484 | 0.794 | 0.819 | 0.741 |
| bagel | gauss2 | clean_bank | knn (raw) | 0.906 | 0.890 | 0.940 | 0.893 |
| bagel | gauss2 | clean_bank | modern_hopfield_auto (whitened) | 0.519 | 0.612 | 0.469 | 0.419 |
