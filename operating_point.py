"""Fixed-operating-point metrics: precision, recall and F1 at a fixed threshold.

AUROC and AUPRO only measure ranking. Here the threshold is calibrated on MVTec 3D-AD's
``validation`` split (normal-only, disjoint from ``train`` and ``test``; built by
``extract.py validation``): it is the ``target_fpr`` percentile of the validation
normality scores, so no test scan and no defect label is involved. The fixed threshold
is then applied to the test split.

The cost-weighted columns use illustrative FN:FP cost ratios (5:1, 20:1), not a fitted
cost model: they show how the scorer ranking would change if a missed defect cost 5x or
20x a false alarm.

    python extract.py validation   # one-time: build the *_validation_abs.npz caches
    python operating_point.py      # the grid -> results/tables/operating_point.csv
    python operating_point.py plot # regenerate results/figures/operating_point.png only
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import yaml

from src import scorers as S
from src.data import test_set, train_bank, validation_set
from src.evaluate import (
    aggregate_image_normality,
    expected_cost_per_scan,
    operating_point_metrics,
    threshold_from_normals,
)

REPS = ["raw", "whitened"]
FIELDS = ["class", "scorer", "rep", "target_fpr", "threshold", "tp", "fp", "fn", "tn",
         "precision", "recall", "f1", "realized_fpr_val", "realized_fpr_test",
         "n_val_scans", "n_test_scans"] + [f"cost_{fn}to{fp}" for fn, fp in
                                           ((5, 1), (20, 1))]
CSV_NAME = "operating_point.csv"
FIG_NAME = "operating_point.png"
COST_RATIOS = ((5, 1), (20, 1))


def evaluate_class(cls: str, cfg: dict, target_fprs: list) -> list[dict]:
    Xi_raw = train_bank(cls, cfg)
    val = validation_set(cls, cfg)
    ts = test_set(cls, {**cfg, "features": cfg.get("features", "abs")})
    topk_frac = cfg.get("topk_frac")
    geom_dim = int(cfg.get("geom_dim", 33))
    n_val_patches = val["X"].shape[1]
    rows = []
    for rep in REPS:
        combined = np.hstack([val["X"], ts["X"]])
        Xi, comb_r = S.condition(rep, Xi_raw, combined, geom_dim)
        Xv, Xt = comb_r[:, :n_val_patches], comb_r[:, n_val_patches:]
        for name in S.SCORER_NAMES:
            scorer = S.make_scorer(name, cfg).fit(Xi)
            img_val = aggregate_image_normality(val["scan_index"], scorer.score(Xv),
                                                val["n_scans"], topk_frac)
            img_test = aggregate_image_normality(ts["scan_index"], scorer.score(Xt),
                                                 ts["n_scans"], topk_frac)
            for target_fpr in target_fprs:
                thr = threshold_from_normals(img_val, target_fpr)
                realized_fpr_val = float(np.mean(img_val < thr))
                m = operating_point_metrics(img_test, ts["is_anomalous"], thr)
                row = {"class": cls, "scorer": name, "rep": rep,
                      "target_fpr": target_fpr, "threshold": round(thr, 5),
                      **{k: m[k] for k in ("tp", "fp", "fn", "tn")},
                      "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                      "realized_fpr_val": round(realized_fpr_val, 4),
                      "realized_fpr_test": m["fpr"],
                      "n_val_scans": val["n_scans"], "n_test_scans": ts["n_scans"]}
                for cost_fn, cost_fp in COST_RATIOS:
                    row[f"cost_{cost_fn}to{cost_fp}"] = round(
                        expected_cost_per_scan(m, cost_fn, cost_fp), 4)
                rows.append(row)
    return rows


def run_grid(cfg: dict) -> list:
    ocfg = cfg.get("operating_point", {"target_fpr": [0.01, 0.05, 0.10]})
    target_fprs = ocfg.get("target_fpr", [0.01, 0.05, 0.10])
    rows = []
    for cls in cfg["classes"]:
        cls_rows = evaluate_class(cls, cfg, target_fprs)
        rows.extend(cls_rows)
        print(f"[{cls}] {len(cls_rows)} rows", flush=True)
    return rows


def _round(v):
    return round(v, 4) if isinstance(v, float) and np.isfinite(v) else \
        ("nan" if isinstance(v, float) else v)


def write_csv(cfg: dict, rows: list) -> str:
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    path = os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: _round(r[k]) for k in FIELDS})
    return path


def write_summary(cfg: dict, rows: list) -> str:
    """A compact per-scorer summary at target_fpr=0.05, whitened rep (the reported
    recipe), mean across classes -- results/operating_point.md."""
    sub = [r for r in rows if r["rep"] == "whitened" and abs(r["target_fpr"] - 0.05) < 1e-9]
    lines = ["# Fixed operating-point metrics (target FPR = 5%, whitened, mean across classes)",
             "", "Threshold calibrated on the held-out `validation` split (normal-only, "
             "never seen by the bank or the test scorer); applied FIXED to `test`. "
             "Cost columns: mean per-scan cost under illustrative FN:FP unit-cost ratios "
             "(a missed defect vs a false alarm) -- see the module docstring.", "",
             "| scorer | precision | recall | F1 | realized FPR (test) | cost 5:1 | cost 20:1 |",
             "|---|---|---|---|---|---|---|"]
    for name in S.SCORER_NAMES:
        cell = [r for r in sub if r["scorer"] == name]
        if not cell:
            continue
        mean = lambda k: np.nanmean([r[k] for r in cell])
        lines.append(f"| {name} | {mean('precision'):.3f} | {mean('recall'):.3f} | "
                     f"{mean('f1'):.3f} | {mean('realized_fpr_test'):.3f} | "
                     f"{mean('cost_5to1'):.3f} | {mean('cost_20to1'):.3f} |")
    path = os.path.join("results", "operating_point.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Figure (regenerates from the csv only)                                       #
# --------------------------------------------------------------------------- #
def plot(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import style
    style.apply()

    path = os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)
    if not os.path.exists(path):
        print(f"skip {FIG_NAME} (no {CSV_NAME} yet; run `python operating_point.py`)")
        return
    with open(path) as f:
        rows = [r for r in csv.DictReader(f) if r["rep"] == "whitened"]

    fprs = sorted({float(r["target_fpr"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, metric, ylabel in ((axes[0], "recall", "recall (test, mean across classes)"),
                               (axes[1], "precision", "precision (test, mean across classes)")):
        for name in S.SCORER_NAMES:
            ys = [np.mean([float(r[metric]) for r in rows
                          if r["scorer"] == name and float(r["target_fpr"]) == f
                          and np.isfinite(float(r[metric]))]) for f in fprs]
            ax.plot([f * 100 for f in fprs], ys, color=style.SCORER_COLORS.get(name, style.TEXT),
                   marker=style.SCORER_MARKERS.get(name, "o"), lw=2, ms=6,
                   label=name.replace("_", " "))
        ax.set_xlabel("target FPR at calibration (%, validation split)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_title("Recall vs calibrated operating point")
    axes[1].set_title("Precision vs calibrated operating point")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("Fixed-threshold precision/recall, whitened rep "
                 "(threshold calibrated on the held-out validation split)", y=1.02)
    fig.tight_layout()
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["figures_dir"], FIG_NAME)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main(mode: str = "run", config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if mode == "run":
        rows = run_grid(cfg)
        print("wrote", write_csv(cfg, rows))
        print("wrote", write_summary(cfg, rows))
    elif mode != "plot":
        raise SystemExit(f"usage: python operating_point.py [run|plot]  (got {mode!r})")
    plot(cfg)


if __name__ == "__main__":
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "run")
