# oneclass3d

One-class 3D anomaly detection on **MVTec 3D-AD**: fit on normal scans only, flag unseen
defects. A frozen per-patch descriptor feeds a memory bank of normal patches, and **six
anomaly scorers** are compared on equal footing, each on **raw** and on **whitened**
features. No training, no GPU: everything from the raw point clouds to the figures runs
on a CPU in this repository.

## Method

- **Features.** Each scan is resampled to 224×224. The background plane is removed by
  RANSAC, seeded on the image border, and stray points are removed by DBSCAN. FPFH
  (Open3D) is computed per point, average-pooled to a 28×28 patch grid and L2-normalized
  (33-dim). With `features: mm`, a frozen DINOv2 ViT-S/14 token, which lands on the
  same 28×28 grid, is appended to each patch (417-dim). Extraction is single-threaded
  and bit-deterministic.
- **Memory bank.** For each class, `Xi` holds every patch of every normal training scan.
- **Scoring.** Each scorer gives each test patch a normality score against `Xi`. The
  image score is the minimum over the scan's patches, and AUROC negates it.
- **Whitening.** A per-class transform is fit on the normal bank only: mean-centring,
  shrunk covariance, `W = C'^{-1/2}`, then re-normalization. On the multimodal
  descriptor, *block-wise* whitening transforms only the geometry block.

| scorer | normality s(x) | hyperparameter |
|---|---|---|
| `knn` | −(distance to the k-th nearest normal patch) | k = 1 |
| `modern_hopfield` | (1/β)·logsumexp(β·Xiᵀx), a soft nearest neighbour | β = 100 |
| `kde` | Gaussian KDE log-density | Scott's rule |
| `ocsvm` | signed distance to a one-class RBF-SVM boundary | ν = 0.1, γ = scale |
| `centroid` | x · mean direction (maximum-bias baseline) | – |
| `classical_hopfield` | ‖Xiᵀx‖² (quadratic baseline) | – |

Hyperparameters are fixed defaults and are never tuned on the test set. The robustness
studies also use Mahalanobis (k-means clusters with Ledoit-Wolf covariances) and a
k-NN + Mahalanobis hybrid (`src/scorers.py`).
