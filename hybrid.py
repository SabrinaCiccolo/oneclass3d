"""k-NN + Mahalanobis hybrid on clean banks: which mixing weight does select_alpha pick?

``Hybrid`` scores alpha * z(k-NN) + (1 - alpha) * z(Mahalanobis), with alpha chosen
from normal patches only by the pseudo-anomaly probe of ``select_alpha``. Every class is
scored on raw and whitened geometry, next to its two components. Config: the
``hybrid_study`` block of config.yaml.

    python hybrid.py        # -> results/tables/hybrid.csv
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

FIELDS = ["class", "rep", "scorer", "auroc", "alpha_selected", "bank_cap", "n_bank"]
SCORERS = ("knn", "mahalanobis", "hybrid")


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    hcfg = cfg["hybrid_study"]
    cfg = {**cfg, "features": "abs", "bank_cap": hcfg["bank_cap"]}
    geom_dim = int(cfg.get("geom_dim", 33))
    rows = []
    for cls in cfg["classes"]:
        t0 = time.time()
        Xi0, ts = train_bank(cls, cfg), test_set(cls, cfg)
        for rep in hcfg["reps"]:
            Xi, X = S.condition(rep, Xi0, ts["X"], geom_dim)
            for name in SCORERS:
                scorer = S.make_scorer(name, cfg).fit(Xi)
                img = aggregate_image_normality(ts["scan_index"], scorer.score(X),
                                                ts["n_scans"], cfg.get("topk_frac"))
                rows.append({"class": cls, "rep": rep, "scorer": name,
                             "auroc": round(image_auroc(img, ts["is_anomalous"]), 4),
                             "alpha_selected": getattr(scorer, "alpha_", ""),
                             "bank_cap": hcfg["bank_cap"], "n_bank": Xi.shape[1]})
        print(f"[{cls}] done ({time.time() - t0:.0f}s)", flush=True)

    out = os.path.join(cfg["paths"]["tables_dir"], "hybrid.csv")
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'rep':<10}" + "".join(f"{n:>13}" for n in SCORERS) + f"{'alpha = 0':>11}")
    for rep in hcfg["reps"]:
        sub = [r for r in rows if r["rep"] == rep]
        means = [np.mean([r["auroc"] for r in sub if r["scorer"] == n]) for n in SCORERS]
        alphas = [r["alpha_selected"] for r in sub if r["scorer"] == "hybrid"]
        print(f"{rep:<10}" + "".join(f"{m:>13.3f}" for m in means)
              + f"{sum(a == 0.0 for a in alphas):>7} / {len(alphas)}")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
