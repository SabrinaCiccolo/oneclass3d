"""Image-level AUROC, patch->image aggregation, and per-stratum AUROC."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(normality: np.ndarray, is_anomalous: np.ndarray) -> float:
    """Image AUROC. Normality is NEGATED here (AUROC expects high = anomalous)."""
    return float(roc_auc_score(is_anomalous, -np.asarray(normality)))


def aggregate_image_normality(scan_index: np.ndarray, patch_normality: np.ndarray,
                              n_scans: int, topk_frac: float | None = None) -> np.ndarray:
    """Per-scan image normality from its patch normalities (lower = more anomalous).

    ``topk_frac=None`` (default) -> MIN over the scan's patches (= max patch anomaly).
    A float ``f`` -> mean of the lowest-normality ``ceil(f * n_patches)`` patches (>= 1):
    averaging the worst few % is more robust to a single fluke patch than the bare min.

    Scans with no kept patches get a sentinel just below the finite minimum, so they
    rank most-anomalous (we could not extract features -> maximally suspicious).
    """
    scan_index = np.asarray(scan_index)
    pn = np.asarray(patch_normality, dtype=np.float64)
    img = np.full(n_scans, np.inf, dtype=np.float64)
    if topk_frac is None:
        np.minimum.at(img, scan_index, pn)
    else:
        order = np.argsort(scan_index, kind="stable")
        bounds = np.searchsorted(scan_index[order], np.arange(n_scans + 1))
        pn_sorted = pn[order]
        for s in range(n_scans):
            vals = pn_sorted[bounds[s]:bounds[s + 1]]
            if vals.size:
                k = max(1, int(np.ceil(float(topk_frac) * vals.size)))
                img[s] = np.partition(vals, k - 1)[:k].mean()
    missing = ~np.isfinite(img)
    if missing.any():
        finite_min = img[np.isfinite(img)].min() if np.isfinite(img).any() else 0.0
        img[missing] = finite_min - 1.0
    return img


def bootstrap_auroc_samples(normality: np.ndarray, is_anomalous: np.ndarray,
                            n_boot: int = 1000, seed: int = 0) -> np.ndarray:
    """``n_boot`` scan-level bootstrap AUROC draws (resample scans with replacement).

    Degenerate draws (all-good or all-anomalous, possible on small test splits) are
    redrawn so every class returns exactly ``n_boot`` values -- callers can then average
    same-index draws across classes to bootstrap the across-class MEAN too (each column
    is one joint bootstrap replicate). Gives up and returns whatever it has if a class's
    scan mix is degenerate enough that 50x n_boot draws can't find a valid split.
    """
    rng = np.random.default_rng(seed)
    n = len(normality)
    is_anomalous = np.asarray(is_anomalous)
    out = np.empty(n_boot, dtype=np.float64)
    filled, tries = 0, 0
    while filled < n_boot and tries < n_boot * 50:
        tries += 1
        idx = rng.integers(0, n, size=n)
        y = is_anomalous[idx]
        if y.all() or not y.any():
            continue
        out[filled] = image_auroc(normality[idx], y)
        filled += 1
    return out[:filled]


def bootstrap_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI from bootstrap draws; (nan, nan) if there aren't enough of them."""
    if samples.size < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def threshold_from_normals(normal_normality: np.ndarray, target_fpr: float) -> float:
    """Fixed operating-point threshold from a NORMAL-ONLY score sample (no test labels):
    the ``target_fpr``-th percentile of ``normal_normality``, so a scan scoring below it
    is flagged anomalous. If the sample the threshold is calibrated on is itself iid with
    future normals, this targets a false-positive rate of ``target_fpr`` by construction.
    """
    return float(np.percentile(np.asarray(normal_normality, dtype=np.float64),
                               100.0 * float(target_fpr)))


def operating_point_metrics(img_normality: np.ndarray, is_anomalous: np.ndarray,
                            threshold: float) -> dict:
    """Confusion counts + precision/recall/F1/FPR at a FIXED threshold (lower normality
    = more anomalous, so ``normality < threshold`` is the positive/"flag it" call).
    NaN precision/recall/f1 when the corresponding denominator is empty (no positive
    predictions, or no actual anomalies in this slice)."""
    is_anomalous = np.asarray(is_anomalous, dtype=bool)
    pred_anom = np.asarray(img_normality) < threshold
    tp = int(np.sum(pred_anom & is_anomalous))
    fp = int(np.sum(pred_anom & ~is_anomalous))
    fn = int(np.sum(~pred_anom & is_anomalous))
    tn = int(np.sum(~pred_anom & ~is_anomalous))
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
         if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
         else float("nan"))
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
           "precision": precision, "recall": recall, "f1": f1, "fpr": fpr}


def expected_cost_per_scan(counts: dict, cost_fn: float, cost_fp: float) -> float:
    """Mean per-scan cost under unit costs (cost_fn, cost_fp) for a missed defect vs a
    false alarm -- illustrative of QA's asymmetric stakes (a shipped defect is typically
    far costlier than an extra manual inspection), NOT fit to any data. tp/tn cost 0."""
    n = counts["tp"] + counts["fp"] + counts["fn"] + counts["tn"]
    if n == 0:
        return float("nan")
    return (counts["fn"] * cost_fn + counts["fp"] * cost_fp) / n


def stratum_auroc(img_normality: np.ndarray, defect_type: np.ndarray,
                  defect_map: dict, stratum: str) -> float:
    """AUROC over good scans + the anomalies of one stratum (geometric/appearance/mixed).

    Returns NaN when the subset lacks either class (then AUROC is undefined).
    """
    defect_type = np.asarray(defect_type)
    is_good = defect_type == "good"
    in_stratum = np.array([defect_map.get(t) == stratum for t in defect_type])
    keep = is_good | in_stratum
    is_anom = in_stratum[keep]
    if is_anom.all() or not is_anom.any():
        return float("nan")
    return image_auroc(img_normality[keep], is_anom)
