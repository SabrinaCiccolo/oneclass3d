"""Paired-bootstrap significance test of every scorer against kNN.

``evaluate.bootstrap_auroc_samples`` draws its scan-resample indices from the labels
only (never the scores), so for a given class and seed every scorer is evaluated on the
same resamples and the per-draw AUROCs of two scorers are paired. For each scorer and
rep, per class and pooled across classes (matching-index mean, as in ``run.py``'s
``_mean_ci``), this reports the AUROC delta vs kNN, its 95% paired-bootstrap CI and a
two-sided bootstrap p-value (twice the smaller one-sided tail fraction crossing zero,
capped at 1.0).

    python significance.py       # -> results/tables/significance.csv + .md summary
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import yaml

from run import evaluate_class
from src import scorers as S
from src.evaluate import bootstrap_ci

N_BOOT = 1000
BASELINE = "knn"
CSV_NAME = "significance.csv"
FIELDS = ["class", "rep", "scorer", "baseline", "auroc_scorer", "auroc_baseline",
         "delta", "ci_lo", "ci_hi", "p_value", "n_boot"]


def _p_value(diff: np.ndarray) -> float:
    """Two-sided bootstrap p-value: 2x the smaller tail fraction crossing zero."""
    n = diff.size
    if n == 0:
        return float("nan")
    p_le = float(np.mean(diff <= 0.0))
    p_ge = float(np.mean(diff >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def _row(cls: str, rep: str, name: str, auroc_s: float, auroc_b: float,
        diff: np.ndarray) -> dict:
    lo, hi = bootstrap_ci(diff)
    return {"class": cls, "rep": rep, "scorer": name, "baseline": BASELINE,
           "auroc_scorer": round(auroc_s, 4), "auroc_baseline": round(auroc_b, 4),
           "delta": round(auroc_s - auroc_b, 4),
           "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
           "p_value": round(_p_value(diff), 4), "n_boot": diff.size}


def run_grid(cfg: dict) -> list:
    classes = cfg["classes"]
    per_class: dict = {}   # cls -> {(scorer, rep): cell}
    for cls in classes:
        per_class[cls] = evaluate_class(cls, cfg, n_boot=N_BOOT)
        print(f"[{cls}] done", flush=True)

    others = [s for s in S.SCORER_NAMES if s != BASELINE]
    reps = sorted({rep for (_, rep) in next(iter(per_class.values())).keys()})
    rows = []

    # per-class rows: the two scorers' bootstrap draws are index-paired by construction
    for cls in classes:
        table = per_class[cls]
        for rep in reps:
            base_cell = table[(BASELINE, rep)]
            for name in others:
                cell = table[(name, rep)]
                diff = cell["boot"] - base_cell["boot"]
                rows.append(_row(cls, rep, name, cell["overall"], base_cell["overall"],
                                 diff))

    # pooled-across-class rows: average matching-index draws across classes first (same
    # pairing run.py's own _mean_ci uses for a single scorer's mean-AUROC CI), then
    # difference the two (still index-paired, since both means are built from the SAME
    # per-class draw sequences).
    for rep in reps:
        for name in others:
            base_boots = [per_class[c][(BASELINE, rep)]["boot"] for c in classes]
            s_boots = [per_class[c][(name, rep)]["boot"] for c in classes]
            m = min(min(len(b) for b in base_boots), min(len(b) for b in s_boots))
            base_mean = np.mean([b[:m] for b in base_boots], axis=0)
            s_mean = np.mean([b[:m] for b in s_boots], axis=0)
            auroc_b = float(np.mean([per_class[c][(BASELINE, rep)]["overall"] for c in classes]))
            auroc_s = float(np.mean([per_class[c][(name, rep)]["overall"] for c in classes]))
            rows.append(_row("mean", rep, name, auroc_s, auroc_b, s_mean - base_mean))
    return rows


def write_csv(cfg: dict, rows: list) -> str:
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    path = os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_summary(rows: list) -> str:
    """results/significance.md: the pooled (mean-across-class) rows, flagged
    significant/not at alpha=0.05 (CI excludes 0)."""
    pooled = [r for r in rows if r["class"] == "mean"]
    lines = ["# Pairwise significance vs kNN (mean across 10 classes, paired bootstrap)",
             "", f"Baseline: `{BASELINE}`. 95% CI and two-sided p-value on the PAIRED "
             f"AUROC delta ({N_BOOT} paired scan-resamples/class, matching-index "
             "averaged across classes -- see the module docstring). 'significant' = "
             "the 95% CI excludes zero (alpha=0.05, not corrected for multiple "
             "comparisons across the 5 scorers x 2 reps = 10 tests here).", "",
             "| rep | scorer | delta AUROC | 95% CI | p | significant? |",
             "|---|---|---|---|---|---|"]
    for r in sorted(pooled, key=lambda r: (r["rep"], -abs(r["delta"]))):
        sig = "yes" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "no"
        lines.append(f"| {r['rep']} | {r['scorer']} | {r['delta']:+.4f} | "
                     f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_value']:.4f} | {sig} |")
    path = os.path.join("results", "significance.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    rows = run_grid(cfg)
    print("wrote", write_csv(cfg, rows))
    print("wrote", write_summary(rows))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
