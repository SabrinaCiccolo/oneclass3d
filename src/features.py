"""Extract FPFH patch features and cache them (frozen, no training).

Per-point FPFH (Open3D, CPU) on the cleaned grid -> reshape to 224x224 -> average-pool
valid pixels into a 28x28 grid of 33-dim patch descriptors (8x8 px pooling; drop
patches with too few valid points) -> L2-normalize each descriptor onto the unit
sphere -> write ``cache/features/<class>_<split>_abs.npz``.

Radii are absolute (0.1 / 0.25 m, config ``fpfh``). L2-normalization is a precondition
(the exp-RBF kernel equivalence holds only on the sphere), asserted as a runtime guard.
The rest of the pipeline (``src/data.py`` onward) reads ONLY this cache.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from src.data_io import iter_scans
from src.preprocess import preprocess_scan

FPFH_DIM = 33


def seed_open3d(cfg: dict) -> int:
    """Seed Open3D's process-global RNG (RANSAC + DBSCAN) for reproducibility.

    ``segment_plane`` / ``cluster_dbscan`` draw from Open3D's global RNG; seeding it
    once before extraction makes the run deterministic. Returns the seed used.
    """
    seed = int(cfg.get("base_seed", 0))
    o3d.utility.random.seed(seed)
    return seed


def fpfh_per_point(xyz: np.ndarray, valid_mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Open3D normals + FPFH on valid grid points -> (N, 33) in boolean-mask order.

    Points are taken as ``xyz[valid_mask]`` so the caller scatters back with
    ``grid[valid_mask] = fpfh``. Normals are estimated without consistent global
    orientation (the FPFH histogram is robust to normal sign flips).
    """
    fcfg = cfg["fpfh"]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[valid_mask].astype(np.float64))

    max_nn = int(fcfg["max_nn"])
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=float(fcfg["normal_radius_abs"]), max_nn=max_nn))
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(fcfg["fpfh_radius_abs"]), max_nn=max_nn))
    return np.asarray(feat.data, dtype=np.float32).T  # (N, 33)


def pool_to_patches(fpfh_grid: np.ndarray, valid_mask: np.ndarray, cfg: dict
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Average-pool the 224x224 per-point FPFH into a 28x28 patch grid.

    Returns (patches (784, 33), patch_valid (784,)). Every patch position is kept so
    its row index doubles as the spatial patch_idx (row-major i*28+j); ``patch_valid``
    flags patches with >= ``min_valid_per_patch`` contributing points. Averaging is
    over valid pixels only.
    """
    grid = int(cfg["patch_grid"])
    size = fpfh_grid.shape[0]
    pool = size // grid
    dim = fpfh_grid.shape[2]

    v = valid_mask.astype(np.float32)
    summed = (fpfh_grid * v[:, :, None]).reshape(grid, pool, grid, pool, dim).sum(axis=(1, 3))
    counts = v.reshape(grid, pool, grid, pool).sum(axis=(1, 3))  # (grid, grid)

    patches = (summed / np.maximum(counts, 1.0)[:, :, None]).reshape(grid * grid, dim)
    patch_valid = counts.reshape(grid * grid) >= int(cfg["min_valid_per_patch"])
    return patches.astype(np.float32), patch_valid


def l2_normalize(patches: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """L2-normalize each row onto the unit sphere; reject zero-norm rows; assert ||x||~=1."""
    norms = np.linalg.norm(patches, axis=1)
    if patches.shape[0] and norms.min() <= tol:
        raise ValueError(f"zero/degenerate patch descriptor (min ||x||={norms.min():.2e})")
    out = (patches / norms[:, None]).astype(np.float32)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=tol)
    return out


def cache_path(class_name: str, split: str, cfg: dict, suffix: str = "") -> Path:
    return Path(cfg["paths"]["cache_dir"]) / f"{class_name}_{split}_abs{suffix}.npz"


def cache_class_split(class_name: str, split: str, cfg: dict,
                      corrupt_fn=None, suffix: str = "") -> Path:
    """Process a whole (class, split) and write its FPFH cache npz.

    NPZ schema (consumed by ``src/data.py``): patches (M,33) float32; patch_idx (M,)
    spatial 0..783; scan_index (M,) into scan_ids; scan_ids (S,); defect_type (S,);
    patch_counts (S,) kept patches per scan.

    ``corrupt_fn(xyz, valid_mask, scan_id) -> (xyz, valid_mask)`` corrupts
    the organized scan right after loading, BEFORE preprocessing, and ``suffix``
    names the resulting variant cache (e.g. '__gauss2' -> bagel_test_abs__gauss2.npz).
    The default (None, '') is the plain pipeline.
    """
    seed_open3d(cfg)
    size = int(cfg["resample_grid"])
    tol = float(cfg["fpfh"]["norm_tol"])
    data_root = cfg["paths"]["data_root"]

    patches_all, patch_idx_all, scan_index_all = [], [], []
    scan_ids, defect_types, patch_counts = [], [], []

    for s, rec in enumerate(iter_scans(class_name, split, data_root)):
        xyz, valid = (rec.xyz, rec.valid_mask) if corrupt_fn is None else \
            corrupt_fn(rec.xyz, rec.valid_mask, rec.scan_id)
        rs_xyz, cleaned = preprocess_scan(xyz, valid, cfg)
        grid = np.zeros((size, size, FPFH_DIM), dtype=np.float32)
        grid[cleaned] = fpfh_per_point(rs_xyz, cleaned, cfg)  # scatter back
        patches, patch_valid = pool_to_patches(grid, cleaned, cfg)

        kept = np.flatnonzero(patch_valid)
        if kept.size:
            patches_all.append(l2_normalize(patches[kept], tol))
            patch_idx_all.append(kept.astype(np.int32))
            scan_index_all.append(np.full(kept.size, s, dtype=np.int32))
        scan_ids.append(rec.scan_id)
        defect_types.append(rec.defect_type)
        patch_counts.append(int(kept.size))

    cat = lambda xs, w: np.concatenate(xs) if xs else np.zeros((0, w), np.float32)
    out_path = cache_path(class_name, split, cfg, suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        patches=cat(patches_all, FPFH_DIM),
        patch_idx=(np.concatenate(patch_idx_all) if patch_idx_all else np.zeros(0, np.int32)),
        scan_index=(np.concatenate(scan_index_all) if scan_index_all else np.zeros(0, np.int32)),
        scan_ids=np.array(scan_ids),
        defect_type=np.array(defect_types),
        patch_counts=np.array(patch_counts, dtype=np.int32),
    )
    return out_path
