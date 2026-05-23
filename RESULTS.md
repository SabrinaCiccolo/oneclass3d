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

## Choosing β from normal scans only (`beta: auto`)

`src/scorers.py:select_beta` picks Modern Hopfield's β per class without test labels:
1. hold out a slice of the normal bank;
2. build near-manifold pseudo-anomalies from it (mixup and bin-swap of real
   descriptors);
3. take the smallest β whose held-normal vs pseudo-anomaly AUROC is within one standard
   error of the best.

On the multimodal descriptor the pseudo-anomalies are built separately for the FPFH and
the DINOv2 block (`geom_dim`), because the two blocks are concentrated very differently
on the sphere.

The selected values (`beta_selected` in `results/tables/contamination.csv`, clean rows):
- whitened geometry: 50–500, 100 most often;
- raw geometry: always 1000, the hard 1-NN end of the grid;
- whitened multimodal: 10–20.

On the contamination study's evaluation split with the full bank (clean, whitened
geometry), β-auto scores 0.810, β = 100 scores 0.812 and k-NN 0.807. Automatic
selection therefore matches the hand-set value with no test-set tuning.

## Robustness: defective patches in the bank

`contamination.py` → `results/tables/contamination.csv`,
`results/figures/contamination.png`.

**Setup.**
- 0, 5 or 10% of the bank is replaced by patches from defective test scans.
- The defective scans are split by scan into an injection pool and an evaluation pool.
- 3 seeds, `bank_cap: 12000`.
- Whitening and `beta: auto` are fit on the contaminated bank.

Change in mean AUROC at 5% / 10% contamination:

| features, rep | knn | MH (β = 100) | MH (auto) | mahalanobis | kde |
|---|---|---|---|---|---|
| geometry raw | −.026 / −.033 | −.010 / −.013 | −.025 / −.033 | **−.036 / −.057** | .000 / .000 |
| geometry whitened | −.034 / −.053 | −.033 / −.050 | −.034 / −.051 | −.018 / −.042 | +.005 / +.010 |
| multimodal raw | −.028 / −.052 | −.024 / −.044 | −.028 / −.052 | **−.075 / −.130** | – |
| multimodal whitened | −.074 / −.097 | −.071 / −.093 | −.053 / −.062 | −.075 / −.084 | – |

Seed band (max − min of the class mean over the 3 seeds): 0.002–0.024 on geometry,
0.005–0.044 on multimodal features.

- **Soft and hard retrieval degrade the same way.** Modern Hopfield degrades like 1-NN
  within the seed band. The full-bank check gives the same result (whitened geometry,
  5%): k-NN −0.048, Modern Hopfield −0.047, β-auto −0.047. Raw Modern Hopfield at β = 100
  degrades less only because it starts 0.07 lower.
- **Retrieval and parametric scoring do separate.** Mahalanobis (50 k-means clusters,
  Ledoit-Wolf covariances) is the best clean scorer on raw geometry (0.785 vs 0.736 for
  k-NN), but on raw banks it degrades 1.4–2.7× faster than k-NN.
  Contaminated clusters absorb the defect patches, while retrieval only fails when a
  query's nearest stored patch is itself a contaminant.
- **Whitening is estimated on the contaminated bank**, so it carries part of the
  contamination into the representation.
- **KDE at Scott's bandwidth is too smooth** to notice contamination, or defects.

## Preprocessing checks

**Background-plane removal** (`plane_ablation.py` → `results/tables/plane_ablation.csv`).
On four classes (carrot, cookie, cable_gland, tire) the fitted plane covers 61–94% of a
scan's resampled valid points. Geometry was re-extracted without plane removal
(`ransac.enabled: false`) and scored with k-NN, full bank:

| k-NN | plane removed (default) | plane kept |
|---|---|---|
| whitened | **0.721** | 0.685 |
| raw | 0.682 | **0.736** |

- **Whitened: removing the plane helps** (+0.036, 3 of 4 classes). Otherwise the flat
  plane dominates the covariance the whitener is fit on.
- **Raw: keeping the plane helps.** This is a bank-size effect: the bank grows 2.5–14×
  (carrot 15.8k → 224k patches) and raw k-NN benefits from the extra coverage.
- **The clear gain is efficiency**: a several-fold smaller bank.

**Border-strip assumption** (`border_check.py` → `results/tables/border_check.csv`). The
RANSAC plane is fit to the image-border strip, assuming the object never reaches it. The
fitted plane removes on average ≥ 92.5% of the border-strip pixels in every class. The
lowest is rope, whose elongated shape sometimes reaches the border. Only 9 of 3853 train
and test scans fall below 90%. Rope is still among the best-detected classes (k-NN 0.981
raw), so the assumption holds well enough.
