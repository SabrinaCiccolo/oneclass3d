"""Point-level localization: back-project patch anomaly onto the GT mask, pixel-AUROC + AUPRO.

The per-patch anomaly of the scorer in config ``localization`` (default: whitened k-NN)
forms a 28x28 grid. It is scattered back via patch_idx, upsampled to the resampled grid
and aligned to each anomalous scan's ground-truth mask through the same resample index
map the feature pipeline used, so patch (i,j) lands on exactly the pixels it describes.
Reports per-class pixel-AUROC and AUPRO (per-region overlap, normalized up to
pixel-FPR 0.3) over anomalous test scans, plus the across-class means.

    python localize.py        # -> results/tables/localization.csv, prints the summary

With ``localization.viz_per_class > 0`` in config it also saves that many 3-panel
figures per class (RGB | anomaly-heatmap overlay | ground-truth mask) under
results/figures/localization/. Reads the geometry cache plus the raw GT masks and RGB
images at paths.data_root.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from src import scorers as S
from src.data import load_cache, train_bank
from src.data_io import load_gt_mask
from src.preprocess import resample_indices


def _gt_path(data_root: str, scan_id: str) -> str:
    cls, split, defect, stem = scan_id.split("/")
    return os.path.join(data_root, cls, split, defect, "gt", stem + ".png")


def _rgb_path(data_root: str, scan_id: str) -> str:
    cls, split, defect, stem = scan_id.split("/")
    return os.path.join(data_root, cls, split, defect, "rgb", stem + ".png")


def _viz_select(defect_type, n: int) -> set:
    """Up to n anomalous scan indices, spread round-robin across defect types (variety)."""
    groups: dict = {}
    for s, d in enumerate(defect_type):
        if d != "good":
            groups.setdefault(d, []).append(s)
    picked: list = []
    while len(picked) < n and any(groups.values()):
        for d in list(groups):
            if groups[d]:
                picked.append(groups[d].pop(0))
                if len(picked) >= n:
                    break
    return set(picked)


def _save_panel(rgb: np.ndarray, heat: np.ndarray, gt: np.ndarray, out_path: str, title: str) -> None:
    """3-panel figure: RGB | RGB+anomaly-heatmap overlay | ground-truth mask (all aligned)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = (heat - heat.min()) / (np.ptp(heat) + 1e-12)          # per-scan min-max for display
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.2))
    ax[0].imshow(rgb); ax[0].set_title("RGB")
    ax[1].imshow(rgb); ax[1].imshow(h, cmap="jet", alpha=0.5); ax[1].set_title("anomaly overlay")
    ax[2].imshow(gt, cmap="gray"); ax[2].set_title("ground truth")
    for a in ax:
        a.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _aupro(region_scores: list, region_sizes: list, neg_scores: np.ndarray,
          fpr_limit: float = 0.3) -> float:
    """Per-Region-Overlap AUC, integrated up to pixel-FPR=``fpr_limit`` and normalized
    to [0, 1] (the MVTec AD protocol) -- literature-comparable alongside pixel-AUROC.

    ``region_scores`` is one SORTED score array per connected GT region (pooled across
    every anomalous scan of the class); ``neg_scores`` is every non-defect pixel's score,
    pooled the same way. The upsampled heat map only ever takes as many distinct values
    as there are patches (<=784/scan), so using the FULL set of observed score values as
    the threshold sweep is exact (no discretization loss from a fixed-size grid) and
    still cheap. PRO(t) is the mean, over regions, of the fraction of a region's pixels
    scored >= t; FPR(t) is the false-positive rate over ``neg_scores``. Both use
    ``np.searchsorted`` against each region's/the negatives' pre-sorted scores, so the
    whole threshold sweep is a handful of vectorized ops, not a per-threshold loop.
    """
    if not region_scores or neg_scores.size == 0:
        return float("nan")
    neg_sorted = np.sort(neg_scores)
    T = np.unique(np.concatenate([neg_sorted, *region_scores]))
    fpr = (neg_sorted.size - np.searchsorted(neg_sorted, T, side="left")) / neg_sorted.size
    pro_sum = np.zeros(T.size)
    for scores_sorted, size in zip(region_scores, region_sizes):
        pos = size - np.searchsorted(scores_sorted, T, side="left")
        pro_sum += pos / size
    pro = pro_sum / len(region_scores)
    order = np.argsort(fpr)
    fpr_o, pro_o = fpr[order], pro[order]
    keep = fpr_o <= fpr_limit
    if keep.sum() < 2:
        return float("nan")
    return float(np.trapezoid(pro_o[keep], fpr_o[keep]) / fpr_limit)


def class_pixel_auroc(cls: str, cfg: dict, scorer_name: str, rep: str,
                      viz_dir: str | None = None, viz_n: int = 0) -> tuple[float, float, int]:
    """Per-class pixel-AUROC + AUPRO over anomalous test scans; returns (auroc, aupro, n_scans).

    AUPRO (per-region overlap, MVTec AD's own metric) shares this loop's heat/GT maps at
    near-zero extra cost -- see ``_aupro``. If ``viz_dir`` is set, also saves up to
    ``viz_n`` 3-panel overlay figures for the class (RGB | anomaly heatmap | GT) under
    ``viz_dir/<cls>/``.
    """
    from scipy.ndimage import label

    Xi_raw = train_bank(cls, cfg)
    test = load_cache(cls, "test", cfg)
    Xi, X = S.condition(rep, Xi_raw, np.ascontiguousarray(test["patches"].T),
                        int(cfg.get("geom_dim", 33)))
    anom = -S.make_scorer(scorer_name, cfg).fit(Xi).score(X)   # higher = more anomalous

    grid = int(cfg["patch_grid"])
    pool = int(cfg["resample_grid"]) // grid
    size = grid * pool
    data_root = cfg["paths"]["data_root"]
    si, pi = test["scan_index"], test["patch_idx"]
    viz_set = _viz_select(test["defect_type"], viz_n) if viz_dir else set()

    scores, labels, n_scans = [], [], 0
    region_scores, region_sizes, neg_scores = [], [], []
    for s, scan_id in enumerate(test["scan_ids"]):
        if test["defect_type"][s] == "good":
            continue                                          # localization is over anomalies
        gt_path = _gt_path(data_root, str(scan_id))
        if not os.path.exists(gt_path):
            continue
        gt = load_gt_mask(gt_path)                            # (H, W) bool
        m = si == s
        a28 = np.full(grid * grid, np.nan)
        a28[pi[m]] = anom[m]                                  # scatter patch anomalies
        fill = np.nanmin(a28) if np.isfinite(a28).any() else 0.0
        a28 = np.where(np.isfinite(a28), a28, fill).reshape(grid, grid)
        heat = np.repeat(np.repeat(a28, pool, 0), pool, 1)    # (size, size)
        rows, cols = resample_indices(gt.shape[0], gt.shape[1], size)
        gt_aligned = gt[np.ix_(rows, cols)]                   # aligned (size, size) mask
        scores.append(heat.ravel())
        labels.append(gt_aligned.ravel())
        n_scans += 1

        lbl, n_regions = label(gt_aligned)
        for r in range(1, n_regions + 1):
            rmask = lbl == r
            region_scores.append(np.sort(heat[rmask]))
            region_sizes.append(int(rmask.sum()))
        neg_scores.append(heat[~gt_aligned])

        if s in viz_set:                                      # qualitative 3-panel overlay
            from PIL import Image
            rgb = np.asarray(Image.open(_rgb_path(data_root, str(scan_id))).convert("RGB"))
            rr, cc = resample_indices(rgb.shape[0], rgb.shape[1], size)
            _, _, defect, stem = str(scan_id).split("/")
            _save_panel(rgb[np.ix_(rr, cc)], heat, gt_aligned,
                        os.path.join(viz_dir, cls, f"{defect}_{stem}.png"),
                        f"{scan_id}  [{scorer_name}/{rep}]")

    if not scores:
        return float("nan"), float("nan"), 0
    y = np.concatenate(labels)
    p = np.concatenate(scores)
    auroc = float(roc_auc_score(y, p)) if y.any() and not y.all() else float("nan")
    aupro = _aupro(region_scores, region_sizes, np.concatenate(neg_scores))
    return auroc, aupro, n_scans


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    loc = cfg.get("localization", {"scorer": "knn", "rep": "whitened"})
    scorer_name, rep = loc["scorer"], loc["rep"]
    viz_n = int(loc.get("viz_per_class", 0))
    viz_dir = os.path.join(cfg["paths"]["figures_dir"], "localization") if viz_n else None

    print(f"pixel-AUROC + AUPRO localization - scorer={scorer_name}, rep={rep}\n")
    print(f"{'class':14s} {'pixel-AUROC':>12s} {'AUPRO':>8s} {'n_anom':>8s}")
    rows = []
    for cls in cfg["classes"]:
        auroc, aupro, n = class_pixel_auroc(cls, cfg, scorer_name, rep, viz_dir, viz_n)
        rows.append((cls, auroc, aupro, n))
        print(f"{cls:14s} {auroc:12.4f} {aupro:8.4f} {n:8d}", flush=True)
    aurocs = [a for _, a, _, _ in rows if np.isfinite(a)]
    aupros = [p for _, _, p, _ in rows if np.isfinite(p)]
    mean_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    mean_aupro = float(np.mean(aupros)) if aupros else float("nan")
    print("-" * 42)
    print(f"{'MEAN':14s} {mean_auroc:12.4f} {mean_aupro:8.4f}")

    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["tables_dir"], "localization.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scorer", "rep", "class", "pixel_auroc", "aupro", "n_anom"])
        for cls, auroc, aupro, n in rows:
            w.writerow([scorer_name, rep, cls, round(auroc, 4), round(aupro, 4), n])
        w.writerow([scorer_name, rep, "mean", round(mean_auroc, 4), round(mean_aupro, 4), ""])
    print(f"\nwrote {out}")
    if viz_dir:
        print(f"wrote up to {viz_n} anomaly-zone panels/class under {viz_dir}/")


if __name__ == "__main__":
    main()
