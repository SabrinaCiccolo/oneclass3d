"""Background-plane ablation: does removing the RANSAC plane help detection?

The classes in ``plane_ablation.classes`` are re-extracted with ``ransac.enabled:
false`` (cache suffix ``__nr``, train + test, same extract path) and k-NN is scored on
both pipelines, raw and whitened, with the full bank. ``plane_frac`` is the mean
fraction of a scan's resampled valid points that the fitted plane removes.

    python plane_ablation.py      # -> results/tables/plane_ablation.csv
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")   # bit-reproducible extraction (extract.py)

import csv  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src import scorers as S  # noqa: E402
from src.data import test_set, train_bank  # noqa: E402
from src.data_io import iter_scans  # noqa: E402
from src.evaluate import aggregate_image_normality, image_auroc  # noqa: E402
from src.features import cache_class_split, cache_path, seed_open3d  # noqa: E402
from src.preprocess import pipeline_cfg, remove_plane, resample_grid  # noqa: E402

FIELDS = ["class", "pipeline", "rep", "scorer", "auroc", "n_bank", "n_test_scans",
          "plane_frac"]


def plane_fraction(cls: str, cfg: dict) -> float:
    """Mean fraction of resampled valid points removed by the plane, train + test."""
    fracs = []
    for split in ("train", "test"):
        for rec in iter_scans(cls, split, cfg["paths"]["data_root"]):
            rs_xyz, mask = resample_grid(rec.xyz, rec.valid_mask, size=int(cfg["resample_grid"]))
            if mask.any():
                kept = remove_plane(rs_xyz, mask, cfg)
                fracs.append(1.0 - kept.sum() / mask.sum())
    return float(np.mean(fracs))


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    geom_dim = int(cfg.get("geom_dim", 33))
    rows = []
    for cls in cfg["plane_ablation"]["classes"]:
        t0 = time.time()
        for split in ("train", "test"):
            if not cache_path(cls, split, cfg, "__nr").exists():
                cache_class_split(cls, split, pipeline_cfg(cfg, "nr"), suffix="__nr")
        seed_open3d(cfg)
        frac = round(plane_fraction(cls, cfg), 4)
        for pipeline, feats in (("default", "abs"), ("nr", "abs__nr")):
            fcfg = {**cfg, "features": feats, "bank_cap": None}
            Xi0, ts = train_bank(cls, fcfg), test_set(cls, fcfg)
            for rep in ("raw", "whitened"):
                Xi, X = S.condition(rep, Xi0, ts["X"], geom_dim)
                patch = S.make_scorer("knn", cfg).fit(Xi).score(X)
                img = aggregate_image_normality(ts["scan_index"], patch, ts["n_scans"],
                                                cfg.get("topk_frac"))
                rows.append({"class": cls, "pipeline": pipeline, "rep": rep, "scorer": "knn",
                             "auroc": round(image_auroc(img, ts["is_anomalous"]), 4),
                             "n_bank": Xi0.shape[1], "n_test_scans": ts["n_scans"],
                             "plane_frac": frac})
        print(f"[{cls}] done ({time.time() - t0:.0f}s)", flush=True)

    out = os.path.join(cfg["paths"]["tables_dir"], "plane_ablation.csv")
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'k-NN':<10}{'plane removed':>15}{'plane kept':>12}")
    for rep in ("raw", "whitened"):
        m = {p: np.mean([r["auroc"] for r in rows if (r["rep"], r["pipeline"]) == (rep, p)])
             for p in ("default", "nr")}
        print(f"{rep:<10}{m['default']:>15.3f}{m['nr']:>12.3f}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
