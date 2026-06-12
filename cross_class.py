"""Zero-shot cross-class generalization: fit the memory bank on class A's normals and
score class B's test scans, for every (A, B) pair. Does a shape prior learned on one
MVTec 3D-AD object transfer to another?

    python cross_class.py        # -> results/tables/cross_class.csv + heatmap, prints the summary

Scorer, rep and features come from config ``cross_class`` (default: whitened k-NN on
geometry features, full bank); the diagonal (A == B) reproduces the matched-class
numbers of ``auroc.csv``/``auroc_mm.csv``.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import yaml

from src import scorers as S
from src.data import test_set, train_bank
from src.evaluate import aggregate_image_normality, image_auroc

CSV_NAME = "cross_class.csv"


def run_matrix(cfg: dict) -> np.ndarray:
    ccfg = cfg.get("cross_class", {"scorer": "knn", "rep": "whitened", "features": "abs"})
    fcfg = {**cfg, "features": ccfg["features"], "bank_cap": None}
    geom_dim = int(cfg.get("geom_dim", 33))
    classes = cfg["classes"]
    n = len(classes)

    banks = {cls: train_bank(cls, fcfg) for cls in classes}
    tests = {cls: test_set(cls, fcfg) for cls in classes}
    M = np.full((n, n), np.nan)
    for i, bank_cls in enumerate(classes):
        for j, test_cls in enumerate(classes):
            Xi, X = S.condition(ccfg["rep"], banks[bank_cls], tests[test_cls]["X"], geom_dim)
            scorer = S.make_scorer(ccfg["scorer"], cfg).fit(Xi)
            patch = scorer.score(X)
            ts = tests[test_cls]
            img = aggregate_image_normality(ts["scan_index"], patch, ts["n_scans"],
                                            cfg.get("topk_frac"))
            M[i, j] = image_auroc(img, ts["is_anomalous"])
        print(f"[bank={bank_cls}] done", flush=True)
    return M


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]
    ccfg = cfg.get("cross_class", {"scorer": "knn", "rep": "whitened", "features": "abs"})
    M = run_matrix(cfg)

    diag = np.diag(M)
    off = M[~np.eye(len(classes), dtype=bool)]
    print(f"\nconfig: scorer={ccfg['scorer']} rep={ccfg['rep']} features={ccfg['features']}\n")
    print(f"matched-class (diagonal) mean AUROC:   {diag.mean():.4f}")
    print(f"cross-class  (off-diagonal) mean AUROC: {off.mean():.4f}")
    print(f"cross-class max: {off.max():.4f}, min: {off.min():.4f}\n")

    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)
    with open(out, "w", newline="") as f:
        # this table is a matrix, not a long-format per-row table, so the config
        # goes in a leading `#` comment line rather than a repeated column.
        f.write(f"# scorer={ccfg['scorer']} rep={ccfg['rep']} features={ccfg['features']}\n")
        w = csv.writer(f)
        w.writerow(["bank_class", *classes])
        for i, cls in enumerate(classes):
            w.writerow([cls, *[round(float(v), 4) for v in M[i]]])
    print(f"wrote {out}")

    _plot(cfg, classes, M)


def _plot(cfg: dict, classes: list, M: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import style
    style.apply()

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    # "RdYlGn" is not luminance-monotonic (yellow midpoint is lighter than both
    # ends), so a greyscale print would show 0.4 and 1.0 as similarly dark and
    # invert the read; viridis is perceptually uniform (dark=low, light=high)
    # and colourblind-safe. Every cell also carries its exact value as text.
    im = ax.imshow(M, cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("test class"); ax.set_ylabel("bank (train) class")
    ax.set_title("Zero-shot cross-class image AUROC\n(diagonal = matched-class)", fontsize=9)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < 0.7 else "black")
    fig.colorbar(im, ax=ax, label="image AUROC", shrink=0.85)
    fig.tight_layout()
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["figures_dir"], "cross_class.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
