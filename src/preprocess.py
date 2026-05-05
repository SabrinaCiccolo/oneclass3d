"""Preprocess the organized cloud for feature extraction.

The grid is preserved throughout (removed points are flipped off in a boolean mask,
never dropped) so the border-strip RANSAC seed and the 28x28 patch pooling stay valid:

  1. resample the organized cloud to 224x224 (nearest-valid),
  2. border-seeded RANSAC plane fit -> remove background within a scale-relative
     distance of that plane,
  3. DBSCAN cleanup -> keep the largest component, drop floaters.

All thresholds/radii come from ``config.yaml`` (``ransac`` / ``cleanup`` blocks).
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


def bbox_diagonal(points: np.ndarray) -> float:
    """Diagonal length of the axis-aligned bbox of (N,3) points."""
    if points.shape[0] == 0:
        return 0.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _plane_threshold(points: np.ndarray, cfg: dict) -> float:
    """Scale-relative plane distance: a fraction of the bbox diagonal (abs fallback)."""
    rcfg = cfg["ransac"]
    thr = rcfg["plane_thresh_rel"] * bbox_diagonal(points)
    if not np.isfinite(thr) or thr <= 0:
        thr = rcfg["plane_thresh_abs_fallback"]
    return thr


def resample_indices(H: int, W: int, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Source (row, col) index arrays the size x size resample samples from an HxW grid.

    The same deterministic map back-projects a resampled/patch cell to its source pixel
    (used by localization to align a 28x28 anomaly map onto an original-resolution mask).
    """
    rows = np.clip(np.round(np.arange(size) * (H / size)).astype(int), 0, H - 1)
    cols = np.clip(np.round(np.arange(size) * (W / size)).astype(int), 0, W - 1)
    return rows, cols


def resample_grid(xyz: np.ndarray, valid_mask: np.ndarray, size: int = 224
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-valid resample of the organized cloud to size x size.

    Each target cell maps to the rounded source pixel; if invalid, the nearest valid
    pixel within a small window (half the downsample stride) is substituted. Raw xyz
    values are copied (no coordinate averaging, which would distort the FPFH).
    """
    H, W = valid_mask.shape
    out_xyz = np.zeros((size, size, 3), dtype=xyz.dtype)
    out_valid = np.zeros((size, size), dtype=bool)
    rad = max(1, int(round(0.5 * max(H, W) / size)))
    rows, cols = resample_indices(H, W, size)
    for ti, si in enumerate(rows):
        for tj, sj in enumerate(cols):
            if valid_mask[si, sj]:
                out_xyz[ti, tj] = xyz[si, sj]
                out_valid[ti, tj] = True
                continue
            for r in range(1, rad + 1):
                i0, i1 = max(0, si - r), min(H, si + r + 1)
                j0, j1 = max(0, sj - r), min(W, sj + r + 1)
                sub_valid = valid_mask[i0:i1, j0:j1]
                if sub_valid.any():
                    di, dj = np.argwhere(sub_valid)[0]
                    out_xyz[ti, tj] = xyz[i0 + di, j0 + dj]
                    out_valid[ti, tj] = True
                    break
    return out_xyz, out_valid


def remove_plane(xyz: np.ndarray, valid_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Border-seeded RANSAC plane removal; returns the updated valid_mask.

    The plane is fit only to valid points in the image-border strip (background
    reaches the border, the object does not); valid points within the threshold of
    that plane are removed.
    """
    rcfg = cfg["ransac"]
    strip = rcfg["border_strip_px"]

    border = np.zeros_like(valid_mask)
    border[:strip, :] = True
    border[-strip:, :] = True
    border[:, :strip] = True
    border[:, -strip:] = True
    border &= valid_mask

    thr = _plane_threshold(xyz[valid_mask], cfg)

    border_pts = xyz[border]
    if border_pts.shape[0] < rcfg["ransac_n"]:
        return valid_mask.copy()  # not enough background to anchor a plane

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(border_pts.astype(np.float64))
    plane_model, _ = pcd.segment_plane(
        distance_threshold=thr,
        ransac_n=int(rcfg["ransac_n"]),
        num_iterations=int(rcfg["num_iterations"]),
    )
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    nnorm = np.linalg.norm(normal)
    if nnorm == 0:
        return valid_mask.copy()

    out = valid_mask.copy()
    idx = np.argwhere(valid_mask)
    dist = np.abs(xyz[valid_mask] @ normal + d) / nnorm
    on_plane = dist < thr
    out[idx[on_plane, 0], idx[on_plane, 1]] = False
    return out


def _kdistance_eps(points: np.ndarray, k: int, percentile: float) -> float:
    """Adaptive per-scan DBSCAN eps: a high percentile of each point's k-th nearest-
    neighbour distance (the classic Ester et al. 1996 k-distance-plot "elbow" heuristic).

    Unlike a fixed absolute eps, this tracks the scan's ACTUAL local point spacing --
    including a noise corruption's increased jitter, which a fixed or bbox-diagonal-
    relative eps can't see (bbox diagonal is a coarse global extent, barely moved by
    per-point jitter; local NN spacing is exactly what DBSCAN's eps has to cover).
    """
    n = points.shape[0]
    if n <= k:
        return 0.0
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(points).query(points, k=k + 1)   # col 0 is the point itself (dist 0)
    return float(np.percentile(dist[:, k], percentile))


def _dbscan_eps(points: np.ndarray, cfg: dict) -> float:
    """Resolve this scan's DBSCAN eps per ``cleanup.dbscan_eps_mode``:

    'abs' (default) -- the fixed ``dbscan_eps`` (m).
    'rel' -- ``dbscan_eps_rel`` * bbox diagonal of the (post-RANSAC) valid points, like
    RANSAC's scale-relative ``plane_thresh_rel``; adapts to objects of different sizes.
    'adaptive' -- the k-distance heuristic (see ``_kdistance_eps``); adapts to the local
    point spacing, e.g. under Gaussian noise.
    Any mode falls back to the fixed ``dbscan_eps`` on a degenerate (empty/non-finite/
    non-positive) result.
    """
    ccfg = cfg["cleanup"]
    mode = ccfg.get("dbscan_eps_mode", "abs")
    fallback = float(ccfg["dbscan_eps"])
    if mode == "abs":
        return fallback
    if mode == "rel":
        eps = float(ccfg.get("dbscan_eps_rel", 0.0)) * bbox_diagonal(points)
    elif mode == "adaptive":
        eps = _kdistance_eps(points, int(ccfg["dbscan_min_points"]),
                             float(ccfg.get("dbscan_eps_percentile", 90.0)))
    else:
        raise ValueError(f"unknown cleanup.dbscan_eps_mode {mode!r}")
    return eps if np.isfinite(eps) and eps > 0 else fallback


def cc_cleanup(xyz: np.ndarray, valid_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """DBSCAN cleanup; keep the largest cluster, drop floaters. Returns valid_mask."""
    ccfg = cfg["cleanup"]
    idx = np.argwhere(valid_mask)
    if idx.shape[0] == 0:
        return valid_mask.copy()

    points = xyz[valid_mask].astype(np.float64)
    eps = _dbscan_eps(points, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(pcd.cluster_dbscan(
        eps=eps,
        min_points=int(ccfg["dbscan_min_points"]),
    ))

    out = valid_mask.copy()
    clustered = labels >= 0  # -1 is DBSCAN noise
    if not clustered.any():
        return out  # no cluster met min_points; keep points rather than wiping the scan
    largest = np.bincount(labels[clustered]).argmax()
    drop = labels != largest
    out[idx[drop, 0], idx[drop, 1]] = False
    return out


def preprocess_scan(xyz: np.ndarray, valid_mask: np.ndarray, cfg: dict
                    ) -> tuple[np.ndarray, np.ndarray]:
    """resample -> plane removal -> cleanup. Returns (resampled xyz, final mask).

    ``ransac.enabled`` / ``cleanup.enabled`` (both default True) toggle the two cleanup
    stages (plane-removal ablation, corruption mitigation probe).
    """
    rs_xyz, mask = resample_grid(xyz, valid_mask, size=int(cfg["resample_grid"]))
    if cfg["ransac"].get("enabled", True):
        mask = remove_plane(rs_xyz, mask, cfg)
    if cfg["cleanup"].get("enabled", True):
        mask = cc_cleanup(rs_xyz, mask, cfg)
    return rs_xyz, mask


PIPELINE_TAGS = ("", "nr", "nc", "nrnc", "rel", "adaptive")


def pipeline_cfg(cfg: dict, tag: str = "") -> dict:
    """``cfg`` with an extraction-pipeline variant applied.

    '' is the default pipeline; 'nr' switches RANSAC plane removal off, 'nc' the DBSCAN
    cleanup, 'nrnc' both; 'rel' / 'adaptive' set ``cleanup.dbscan_eps_mode``. The tag
    also names the variant's cache suffix (e.g. ``bagel_train_abs__nr.npz``).
    """
    if tag not in PIPELINE_TAGS:
        raise ValueError(f"unknown pipeline tag {tag!r}")
    cleanup = {**cfg["cleanup"], "enabled": tag not in ("nc", "nrnc")}
    if tag in ("rel", "adaptive"):
        cleanup["dbscan_eps_mode"] = tag
    return {**cfg, "ransac": {**cfg["ransac"], "enabled": tag not in ("nr", "nrnc")},
            "cleanup": cleanup}
