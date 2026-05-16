"""OC-SVM fit path: Nystroem features on the full bank vs the exact kernel on a subsample.

``ocsvm_nystroem_components`` (the default) fits a linear one-class SGD model on a
Nystroem RBF feature map of the WHOLE bank; without it, OC-SVM fits the exact RBF
kernel on ``ocsvm_max_fit`` randomly subsampled patches. Both are scored on every
class, raw and whitened geometry, with fit and scoring times.

    python ocsvm_fit.py        # -> results/tables/ocsvm_fit.csv
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

FIELDS = ["class", "rep", "fit", "auroc", "fit_s", "score_s", "n_bank", "n_fit"]


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    cfg = {**cfg, "features": "abs", "bank_cap": None}
    geom_dim = int(cfg.get("geom_dim", 33))
    max_fit = int(cfg["ocsvm_max_fit"])
    fits = {"nystroem": cfg["ocsvm_nystroem_components"], "exact": None}
    rows = []
    for cls in cfg["classes"]:
        Xi0, ts = train_bank(cls, cfg), test_set(cls, cfg)
        for rep in ("raw", "whitened"):
            Xi, X = S.condition(rep, Xi0, ts["X"], geom_dim)
            for fit, comps in fits.items():
                scorer = S.make_scorer("ocsvm", {**cfg, "ocsvm_nystroem_components": comps})
                t0 = time.perf_counter()
                scorer.fit(Xi)
                t1 = time.perf_counter()
                patch = scorer.score(X)
                t2 = time.perf_counter()
                img = aggregate_image_normality(ts["scan_index"], patch, ts["n_scans"],
                                                cfg.get("topk_frac"))
                rows.append({"class": cls, "rep": rep, "fit": fit,
                             "auroc": round(image_auroc(img, ts["is_anomalous"]), 4),
                             "fit_s": round(t1 - t0, 3), "score_s": round(t2 - t1, 3),
                             "n_bank": Xi.shape[1],
                             "n_fit": Xi.shape[1] if comps else min(max_fit, Xi.shape[1])})
        print(f"[{cls}] done", flush=True)

    out = os.path.join(cfg["paths"]["tables_dir"], "ocsvm_fit.csv")
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'rep':<10}{'fit':<10}{'mean AUROC':>12}{'mean fit s':>12}")
    for rep in ("raw", "whitened"):
        for fit in fits:
            sub = [r for r in rows if (r["rep"], r["fit"]) == (rep, fit)]
            print(f"{rep:<10}{fit:<10}{np.mean([r['auroc'] for r in sub]):>12.3f}"
                  f"{np.mean([r['fit_s'] for r in sub]):>12.2f}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
