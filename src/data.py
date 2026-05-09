"""Load the cached patch features.

``cfg["features"]`` selects the representation, and the cache file is
``cache/features/<class>_<split>_<features>.npz``:

    abs   -> 33-dim geometry-only FPFH, written once by ``extract.py geom``
    mm    -> 417-dim multimodal L2(concat(FPFH_33, L2(DINOv2_384))), by ``extract.py rgb``
    mm_v3 -> the same with DINOv3 tokens, by ``extract.py rgb_v3``

Both share the same arrays (the mm cache is aligned to the FPFH one by construction):

    patches      (M, D)   float32  L2-normalized patch descriptors (||x|| == 1)
    scan_index   (M,)     int      which scan (row of scan_ids) each patch belongs to
    patch_idx    (M,)     int      patch position 0..783 on the 28x28 grid
    scan_ids     (S,)     str      scan identifiers ("<class>/<split>/<defect>/<stem>")
    defect_type  (S,)     str      per-scan label; "good" == normal, else a defect type
    patch_counts (S,)     int      kept patches per scan (unused here)

The train split is normal-only (the reference set we fit on); the test split mixes
normal and defective scans (what we score).
"""
from __future__ import annotations

import os

import numpy as np

from src.scorers import kcenter_coreset, subsample


def load_cache(cls: str, split: str, cfg: dict) -> dict:
    """Load one class/split npz (representation cfg['features'], default 'abs')."""
    feats = cfg.get("features", "abs")
    path = os.path.join(cfg["paths"]["cache_dir"], f"{cls}_{split}_{feats}.npz")
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def train_bank(cls: str, cfg: dict) -> np.ndarray:
    """Shared memory matrix Xi (D, N): every patch of every normal training scan.

    If config sets ``bank_cap`` (an int), the bank is deterministically subsampled to
    that many patches -- a coreset shared by every scorer, ~3-4x faster on the big
    classes. ``bank_mode`` picks how: 'uniform' (default, seeded random subset) or
    'kcenter' (greedy k-center / farthest-point selection, PatchCore-style coverage).
    The default (``bank_cap: null``) keeps the full bank, where run.py's reproduction
    check is exact; with a cap it becomes a tolerance band.

    The subsample is taken from ``load_cache``'s output, i.e. whichever representation
    ``cfg["features"]`` selects -- for ``features: mm`` that cache is already the
    417-dim FPFH+DINOv2 concat (built once by ``extract.py rgb``), so capping always
    selects on the full representation actually being scored, never a geometry-only
    proxy for it.
    """
    cache = load_cache(cls, "train", cfg)
    Xi = np.ascontiguousarray(cache["patches"].T)            # (D, N)
    cap = cfg.get("bank_cap")
    if cap:
        mode = cfg.get("bank_mode", "uniform")
        pick = {"uniform": subsample, "kcenter": kcenter_coreset}[mode]
        Xi = np.ascontiguousarray(pick(Xi, int(cap), seed=int(cfg.get("base_seed", 0))))
    return Xi


def validation_set(cls: str, cfg: dict) -> dict:
    """The dataset's own held-out NORMAL-only split -- disjoint from both the train
    bank and the test set. Never used to fit a scorer; its only job is calibrating a
    FIXED operating-point threshold on scores the bank has never seen
    (``operating_point.py``), with zero test-label exposure. Same shape as
    ``test_set`` minus the labels (every scan here is normal by construction). Built
    by ``extract.py validation``.
    """
    cache = load_cache(cls, "validation", cfg)
    return {
        "X": np.ascontiguousarray(cache["patches"].T),        # (D, M)
        "scan_index": cache["scan_index"],                    # (M,)
        "n_scans": len(cache["scan_ids"]),
    }


def test_set(cls: str, cfg: dict) -> dict:
    """Test patches + the labels needed to score and stratify.

    Returns X (D, M), scan_index (M,), defect_type (S,), n_scans, and the boolean
    is_anomalous (S,) used as the AUROC target.
    """
    cache = load_cache(cls, "test", cfg)
    return {
        "X": np.ascontiguousarray(cache["patches"].T),       # (D, M)
        "scan_index": cache["scan_index"],                   # (M,)
        "defect_type": cache["defect_type"],                 # (S,)
        "n_scans": len(cache["scan_ids"]),
        "is_anomalous": cache["defect_type"] != "good",      # (S,)
    }
