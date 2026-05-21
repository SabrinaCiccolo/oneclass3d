# Results

All numbers are on MVTec 3D-AD (10 classes). The metric is image AUROC, averaged over the
classes. Memory banks hold every normal training patch (`bank_cap: null`) unless a
section says otherwise. Hyperparameters are fixed defaults or are picked from normal
scans only; nothing is selected against test labels. Each table cites the file in
`results/` it comes from.

## Six scorers on geometry

`run.py` → `results/tables/auroc.{csv,md}` (per class), `results/tables/auroc_ci.csv`,
`results/figures/auroc_bar.png`.

| scorer | raw | whitened |
|---|---|---|
| modern_hopfield | 0.673 | **0.807** |
| knn | 0.756 | **0.803** |
| ocsvm | 0.607 | 0.731 |
| classical_hopfield | 0.581 | 0.657 |
| kde | 0.581 | 0.536 |
| centroid | 0.582 | 0.527 |

**Whitening.** Raw FPFH descriptors are concentrated on the unit sphere, so the soft
scorers sit at a high-bias floor (~0.58). Whitening lifts Modern Hopfield
(0.673 → 0.807), OC-SVM (0.607 → 0.731) and k-NN (0.756 → 0.803). KDE at Scott's
bandwidth and the centroid get worse: KDE over-smooths, and the centroid loses the
common-mode direction. They are kept as baselines. Three classes stay close to chance
for every scorer (cable_gland, foam, tire: 0.54–0.67 with whitened k-NN), because their
defects are hard or impossible to see in local geometry.

**Modern Hopfield vs k-NN.** Modern Hopfield is k-NN with a temperature: β → ∞ recovers
exact 1-NN, β → 0 collapses to the centroid. Its best case is therefore k-NN. Whitening
lets it climb back to the k-NN level, not past it. The β sweep
(`results/figures/mh_beta_sweep.png`, `results/tables/kde_bandwidth_sweep.csv`) shows
this:

| β | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|
| raw | 0.581 | 0.585 | 0.673 | 0.752 |
| whitened | 0.542 | 0.789 | 0.807 | 0.803 |

The whitened 95% bootstrap CIs overlap almost completely (Modern Hopfield
[0.776, 0.836], k-NN [0.771, 0.832], `auroc_ci.csv`).

**Paired significance test.** `significance.py` → `results/tables/significance.csv`,
`results/significance.md`. Every scorer is compared with k-NN on the same 1000 scan
resamples per class, without correction for multiple comparisons. In both
representations every scorer is significantly worse than k-NN: no bootstrap draw crosses
zero, so p < 0.002. The only exception is whitened Modern Hopfield, with
Δ = +0.004, 95% CI [−0.001, +0.009] and p = 0.15. Raw Modern Hopfield is significantly
worse: Δ = −0.084, 95% CI [−0.110, −0.057].

**OC-SVM on the full bank** (`ocsvm_fit.py` → `results/tables/ocsvm_fit.csv`). OC-SVM
is fit on the whole bank through a Nyström feature map (500 landmarks) followed by a
linear one-class SGD model. Compared with an exact-kernel fit on a 2000-patch subsample,
the two are tied: mean AUROC 0.607 vs 0.608 raw and 0.731 vs 0.737 whitened, with
Nyström ahead on 4 and 5 of 10 classes. The Nyström fit takes 0.3–3.9 s per class, the
exact subsample fit under 0.15 s.
