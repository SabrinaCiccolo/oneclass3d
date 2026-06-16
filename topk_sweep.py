"""Sweep ``topk_frac`` (mean of the worst f-fraction of a scan's patches) on the best
detector, against the default min-pool (``topk_frac: null``).

Patch normality is scored once per class; each ``topk_frac`` only changes the cheap
patch->image aggregation, so the sweep takes a few seconds per class.

    python topk_sweep.py        # -> results/tables/topk_sweep.csv + figure, prints the summary
"""
from __future__ import annotations

import csv
import os

import numpy as np
import yaml

from src import scorers as S
from src.data import test_set, train_bank
from src.evaluate import aggregate_image_normality, image_auroc

# The project's best simple detector (RESULTS.md): multimodal, block-wise whitened k-NN.
# Falls back to geometry-only whitened k-NN if the mm cache isn't built.
BEST_CONFIG = {"features": "mm", "rep": "blockwise", "scorer": "knn"}
FALLBACK_CONFIG = {"features": "abs", "rep": "whitened", "scorer": "knn"}
SWEEP = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]   # None (min-pool) is added separately as the baseline
CSV_NAME = "topk_sweep.csv"


def _mm_cache_exists(cfg: dict) -> bool:
    path = os.path.join(cfg["paths"]["cache_dir"], f"{cfg['classes'][0]}_train_mm.npz")
    return os.path.exists(path)


def run_sweep(cfg: dict) -> tuple[list, dict]:
    dcfg = BEST_CONFIG if _mm_cache_exists(cfg) else FALLBACK_CONFIG
    fcfg = {**cfg, "features": dcfg["features"], "bank_cap": None}
    geom_dim = int(cfg.get("geom_dim", 33))
    rows = []
    topks = [None, *SWEEP]   # None = min-pool, the config default, as the reference point
    for cls in cfg["classes"]:
        Xi_raw = train_bank(cls, fcfg)
        ts = test_set(cls, fcfg)
        Xi, X = S.condition(dcfg["rep"], Xi_raw, ts["X"], geom_dim)
        patch = S.make_scorer(dcfg["scorer"], cfg).fit(Xi).score(X)   # scored once
        for tf in topks:
            img = aggregate_image_normality(ts["scan_index"], patch, ts["n_scans"], tf)
            rows.append({"class": cls, "topk_frac": "" if tf is None else tf,
                        "auroc": round(image_auroc(img, ts["is_anomalous"]), 4)})
        print(f"[{cls}] done", flush=True)
    return rows, dcfg


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    rows, dcfg = run_sweep(cfg)

    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "topk_frac", "auroc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    topks = sorted({r["topk_frac"] for r in rows}, key=lambda x: (x != "", x))
    print(f"\nbest config: features={dcfg['features']} rep={dcfg['rep']} scorer={dcfg['scorer']}\n")
    print(f"{'topk_frac':>12s} {'mean AUROC':>12s}")
    means = {}
    for tf in topks:
        vals = [r["auroc"] for r in rows if r["topk_frac"] == tf]
        means[tf] = float(np.mean(vals))
        label = "min-pool" if tf == "" else str(tf)
        print(f"{label:>12s} {means[tf]:12.4f}")
    best = max(means, key=means.get)
    base = means[""]
    print(f"\nbest: {'min-pool' if best == '' else best} ({means[best]:.4f}), "
          f"min-pool baseline {base:.4f}, delta {means[best] - base:+.4f}")

    _plot(cfg, topks, means)
    print(f"\nwrote {out}")


def _plot(cfg: dict, topks: list, means: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import style
    style.apply()

    xs = [0.001 if tf == "" else tf for tf in topks]   # min-pool plotted near topk_frac=0
    fig, ax = plt.subplots(figsize=style.FIGSIZE_1PANEL)
    ax.plot(xs, [means[tf] for tf in topks], "o-", color=style.NAVY)
    ax.axhline(means[""], ls=":", lw=1.4, color=style.RED, label=f"min-pool ({means['']:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("topk_frac (mean of the worst f-fraction of patches; leftmost = min-pool)")
    ax.set_ylabel("mean image AUROC (10 classes)")
    ax.set_title("Patch -> image aggregation: does mean-of-worst-f beat min-pool?")
    ax.legend()
    fig.tight_layout()
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["figures_dir"], "topk_sweep.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
