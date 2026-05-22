"""Diagnostic: does RANSAC's border-strip-is-background assumption hold for every class?

``remove_plane`` (src/preprocess.py) fits its plane to the image-border strip only,
assuming the object never reaches the border. This checks that assumption directly: for
every scan, what fraction of border-strip valid pixels the fitted plane actually removes
(low = the strip wasn't pure background, either object pixels sit in it or the fit missed
part of the true plane). Read-only -- does not touch the feature cache.

    python border_check.py        # -> results/tables/border_check.csv, prints the summary
"""
from __future__ import annotations

import csv
import os

import numpy as np
import yaml

from src.data_io import iter_scans
from src.preprocess import remove_plane, resample_grid


def class_border_stats(cls: str, cfg: dict) -> dict:
    """Per-scan border-removal fraction over every train+test scan of one class."""
    strip = cfg["ransac"]["border_strip_px"]
    fracs = []
    for split in ("train", "test"):
        for rec in iter_scans(cls, split, cfg["paths"]["data_root"]):
            rs_xyz, mask = resample_grid(rec.xyz, rec.valid_mask, size=int(cfg["resample_grid"]))
            border = np.zeros_like(mask)
            border[:strip, :] = True
            border[-strip:, :] = True
            border[:, :strip] = True
            border[:, -strip:] = True
            border &= mask
            if not border.any():
                continue
            new_mask = remove_plane(rs_xyz, mask, cfg)
            removed = border & ~new_mask
            fracs.append(float(removed.sum()) / float(border.sum()))
    arr = np.array(fracs)
    return {"n_scans": len(arr), "mean": float(arr.mean()), "min": float(arr.min()),
            "n_below_0.9": int((arr < 0.9).sum())}


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"{'class':14s} {'n_scans':>8s} {'mean_removed':>13s} {'min_removed':>12s} {'n<0.9':>6s}")
    rows = []
    for cls in cfg["classes"]:
        s = class_border_stats(cls, cfg)
        rows.append({"class": cls, **s})
        print(f"{cls:14s} {s['n_scans']:8d} {s['mean']:13.3f} {s['min']:12.3f} "
              f"{s['n_below_0.9']:6d}", flush=True)

    weak = [r["class"] for r in rows if r["mean"] < 0.9]
    print("-" * 56)
    if weak:
        print(f"weak (<90% mean removed): {weak}")
    else:
        print("no class below the 90% mean-removed threshold")

    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["tables_dir"], "border_check.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "n_scans", "mean", "min", "n_below_0.9"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "mean": round(r["mean"], 4), "min": round(r["min"], 4)})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
