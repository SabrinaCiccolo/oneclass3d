"""RGB encoder comparison: DINOv2 (``features: mm``) vs DINOv3 (``features: mm_v3``).

Both caches append a 384-dim token per FPFH patch (``extract.py rgb`` / ``rgb_v3``).
Each is scored with block-wise whitened k-NN at ``encoders.bank_cap``: image AUROC,
and pixel-AUROC + AUPRO through localize.py.

    python encoders.py        # -> results/tables/encoders.csv
"""
from __future__ import annotations

import csv
import os
import time

import numpy as np
import yaml

from src import scorers as S
from src.data import test_set, train_bank
from src.evaluate import aggregate_image_normality, image_auroc

from localize import class_pixel_auroc

FIELDS = ["class", "features", "image_auroc", "pixel_auroc", "aupro", "bank_cap"]


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ecfg = cfg["encoders"]
    geom_dim = int(cfg.get("geom_dim", 33))
    rows = []
    for features in ecfg["features"]:
        fcfg = {**cfg, "features": features, "bank_cap": ecfg["bank_cap"]}
        for cls in cfg["classes"]:
            t0 = time.time()
            Xi0, ts = train_bank(cls, fcfg), test_set(cls, fcfg)
            Xi, X = S.condition(ecfg["rep"], Xi0, ts["X"], geom_dim)
            patch = S.make_scorer(ecfg["scorer"], fcfg).fit(Xi).score(X)
            img = aggregate_image_normality(ts["scan_index"], patch, ts["n_scans"],
                                            cfg.get("topk_frac"))
            pix, aupro, _ = class_pixel_auroc(cls, fcfg, ecfg["scorer"], ecfg["rep"])
            rows.append({"class": cls, "features": features,
                         "image_auroc": round(image_auroc(img, ts["is_anomalous"]), 4),
                         "pixel_auroc": round(pix, 4), "aupro": round(aupro, 4),
                         "bank_cap": ecfg["bank_cap"]})
            print(f"[{features}/{cls}] done ({time.time() - t0:.0f}s)", flush=True)

    out = os.path.join(cfg["paths"]["tables_dir"], "encoders.csv")
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'features':<10}{'image AUROC':>13}{'pixel-AUROC':>13}{'AUPRO':>8}")
    for features in ecfg["features"]:
        sub = [r for r in rows if r["features"] == features]
        print(f"{features:<10}" + "".join(
            f"{np.mean([r[k] for r in sub]):>{w}.3f}"
            for k, w in (("image_auroc", 13), ("pixel_auroc", 13), ("aupro", 8))))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
