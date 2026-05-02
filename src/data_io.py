"""Read MVTec 3D-AD organized point clouds (the geometry / xyz channel).

On-disk layout (the raw dataset, placed at ``paths.data_root``):

    <data_root>/<class>/<split>/<defect_or_good>/{xyz,rgb,gt}/NNN.{tiff,png,png}

We read only the organized ``xyz`` TIFF here; the defect label is the test-split
subfolder name ("good" == normal). ``gt_mask_path`` points at the matching
ground-truth mask (used only by the optional localization path).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import tifffile

SPLITS = ("train", "validation", "test")


@dataclass
class ScanRecord:
    """One scan in organized (H, W) form, for resampling + border-strip RANSAC."""

    xyz: np.ndarray          # (H, W, 3) float32, organized point cloud
    valid_mask: np.ndarray   # (H, W) bool, False where the reading is zero/NaN
    defect_type: str         # subfolder name, e.g. "good", "crack", "combined"
    gt_mask_path: Path | None  # ground-truth mask (None for 'good' / missing)
    scan_id: str             # "<class>/<split>/<defect>/<stem>"


def load_xyz_tiff(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Organized xyz TIFF -> (xyz (H,W,3) float32, valid_mask (H,W) bool).

    Invalid sensor readings are all-zero pixels (z=0); NaNs are guarded too.
    """
    xyz = np.asarray(tifffile.imread(str(path)), dtype=np.float32)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        raise ValueError(f"expected organized (H,W,3) xyz, got {xyz.shape} from {path}")
    valid_mask = ~np.all(xyz == 0, axis=2) & ~np.isnan(xyz).any(axis=2)
    return xyz, valid_mask


def load_gt_mask(path: Path | str) -> np.ndarray:
    """Ground-truth mask -> (H, W) bool, True where defective (nonzero pixel)."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("L")) > 0


def iter_scans(class_name: str, split: str, data_root: Path | str) -> Iterator[ScanRecord]:
    """Yield ScanRecords for one (class, split), in sorted defect/stem order."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    split_dir = Path(data_root) / class_name / split
    for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        for tiff_path in sorted((defect_dir / "xyz").glob("*.tiff")):
            stem = tiff_path.stem
            gt_path = defect_dir / "gt" / f"{stem}.png"
            xyz, valid_mask = load_xyz_tiff(tiff_path)
            yield ScanRecord(
                xyz=xyz,
                valid_mask=valid_mask,
                defect_type=defect_type,
                gt_mask_path=gt_path if gt_path.exists() else None,
                scan_id=f"{class_name}/{split}/{defect_type}/{stem}",
            )
