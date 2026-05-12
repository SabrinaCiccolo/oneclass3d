"""One-class anomaly detection on MVTec 3D-AD: compare six scorers, raw vs whitened.

For each object class we fit on ALL normal training scans (the shared memory matrix
Xi) and score the test split. Every scorer is run on both the raw FPFH features and a
whitened version, and reported as image-level AUROC per class plus the across-class
mean. Geometry-only FPFH is near-chance on appearance defects, so we also report the
AUROC restricted to the geometric defect stratum.

    python run.py            # writes results/tables/auroc.{csv,md}, prints the summary
"""
from __future__ import annotations

import csv
import os

import numpy as np
import yaml

from src import scorers as S
from src.data import test_set, train_bank
from src.evaluate import (
    aggregate_image_normality,
    bootstrap_auroc_samples,
    bootstrap_ci,
    image_auroc,
    stratum_auroc,
)

N_BOOT = 1000   # scan-level bootstrap resamples for the headline table's CI (per class)


def reps_for(cfg: dict) -> list:
    """Representations to evaluate. 'blockwise' (modality-aware whitening) and 'cca'
    (closed-form canonical-correlation fusion) only apply to the multimodal descriptor;
    for geometry-only features both equal 'whitened', so we skip the redundant passes."""
    return ["raw", "whitened"] + (["blockwise", "cca"] if cfg.get("features") == "mm" else [])


def evaluate_class(cls: str, cfg: dict, n_boot: int = 0) -> dict:
    """Return {(scorer, rep): {'overall', 'geometric', 'ci_lo', 'ci_hi', 'boot'}} for one class.

    ``n_boot > 0`` adds a scan-level bootstrap: ``ci_lo``/``ci_hi`` are this class's 95%
    percentile CI, and ``boot`` is the raw (n_boot,) draw array, seeded the same for every
    (scorer, rep) so callers can average matching draws across classes into a mean-AUROC
    CI (see ``main``). ``n_boot=0`` (the default; used by the integration test) skips it.
    """
    Xi_raw = train_bank(cls, cfg)
    ts = test_set(cls, cfg)
    topk_frac = cfg.get("topk_frac")
    geom_dim = int(cfg.get("geom_dim", 33))
    out: dict = {}
    for rep in reps_for(cfg):
        Xi, X = S.condition(rep, Xi_raw, ts["X"], geom_dim)
        for name in S.SCORER_NAMES:
            patch_norm = S.make_scorer(name, cfg).fit(Xi).score(X)
            img = aggregate_image_normality(ts["scan_index"], patch_norm, ts["n_scans"], topk_frac)
            cell = {
                "overall": image_auroc(img, ts["is_anomalous"]),
                "geometric": stratum_auroc(img, ts["defect_type"], cfg["defect_map"], "geometric"),
            }
            if n_boot:
                boot = bootstrap_auroc_samples(img, ts["is_anomalous"], n_boot=n_boot, seed=0)
                cell["ci_lo"], cell["ci_hi"] = bootstrap_ci(boot)
                cell["boot"] = boot
            out[(name, rep)] = cell
    return out


def _nanmean(xs) -> float:
    xs = np.asarray(xs, dtype=float)
    return float(np.nanmean(xs)) if np.isfinite(xs).any() else float("nan")


def _columns(classes: list) -> list:
    """Column order shared by the csv, the md table, and the summary."""
    return ["scorer", "rep", *classes, "mean", "mean_ci_lo", "mean_ci_hi", "std",
            "geom_mean"]


def _mean_ci(per: dict, classes: list) -> tuple[float, float]:
    """Bootstrap CI of the across-class mean: average matching-index draws across
    classes (each class contributed N_BOOT scan-resampled AUROCs, seeded identically),
    then take the percentile CI of that N_BOOT-long mean distribution. Classes with
    fewer surviving draws (rare, only on degenerate scan mixes) are truncated to the
    shortest available length before averaging."""
    boots = [per[c]["boot"] for c in classes]
    m = min(len(b) for b in boots)
    if m < 2:
        return float("nan"), float("nan")
    mean_draws = np.mean([b[:m] for b in boots], axis=0)
    return bootstrap_ci(mean_draws)


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    classes = cfg["classes"]

    # table[(scorer, rep)][cls] = {'overall':..., 'geometric':..., 'ci_lo':..., 'boot':...}
    table: dict = {}
    for cls in classes:
        print(f"[{cls}] ...", flush=True)
        for key, val in evaluate_class(cls, cfg, n_boot=N_BOOT).items():
            table.setdefault(key, {})[cls] = val

    # ---- reproduction gate (features: abs) / sanity note (features: mm) ----
    knn_raw = [table[("knn", "raw")][c]["overall"] for c in classes]
    knn_mean = float(np.mean(knn_raw))
    features = cfg.get("features", "abs")
    cap = cfg.get("bank_cap")

    if features == "abs" and cfg.get("topk_frac") is None:
        # Self-contained determinism gate: with this project's own `extract.py geom`
        # cache and min-pool aggregation, raw k-NN(k=1) at full data must reproduce the
        # frozen 10-class mean. Full bank -> exact (==); capped bank -> band (subsampling
        # wobbles it). Any drift in extraction/features/aggregation trips this.
        anchor = float(cfg["anchor_knn1_mean"])
        tol = 2e-3 if not cap else 0.05  # capped: random coreset costs ~0.02-0.03 mean AUROC
        assert abs(knn_mean - anchor) < tol, (
            f"REPRO FAIL: raw kNN(k=1) mean {knn_mean:.4f} not within {tol} of {anchor} "
            f"(extraction/features/aggregation drifted from the frozen pipeline)")
        scope = f"capped bank={cap}" if cap else "full bank"
        print(f"\nrepro OK ({scope}): raw kNN(k=1) 10-class mean = {knn_mean:.4f} "
              f"(anchor {anchor}, |Δ| < {tol})\n")
    else:
        # mm (or a non-default aggregation): the FPFH-specific reproduction value does
        # not apply -- report the kNN mean with a soft expectation instead of asserting.
        note = f"raw kNN(k=1) 10-class mean = {knn_mean:.4f}"
        if features == "mm":
            note += ("  [multimodal: should land ~0.92-0.94 at full data, clearly above "
                     "the ~0.80 geometry-only numbers]")
        print(f"\nsanity ({features}, topk_frac={cfg.get('topk_frac')}, bank_cap={cap}): "
              f"{note}\n")

    # ---- assemble rows and write csv + md ----
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    rows, ci_rows = [], []
    for name in S.SCORER_NAMES:
        for rep in reps_for(cfg):
            per = table[(name, rep)]
            overalls = [per[c]["overall"] for c in classes]
            row = {"scorer": name, "rep": rep}
            row.update({c: per[c]["overall"] for c in classes})
            row["mean"] = float(np.mean(overalls))
            row["mean_ci_lo"], row["mean_ci_hi"] = _mean_ci(per, classes)
            row["std"] = float(np.std(overalls))
            row["geom_mean"] = _nanmean([per[c]["geometric"] for c in classes])
            rows.append(row)
            for c in classes:
                ci_rows.append({"scorer": name, "rep": rep, "class": c,
                                "auroc": per[c]["overall"],
                                "ci_lo": per[c]["ci_lo"], "ci_hi": per[c]["ci_hi"]})

    suffix = "" if features == "abs" else f"_{features}"
    csv_path = os.path.join(cfg["paths"]["tables_dir"], f"auroc{suffix}.csv")
    fields = _columns(classes)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in fields})

    md_path = os.path.join(cfg["paths"]["tables_dir"], f"auroc{suffix}.md")
    _write_md(md_path, rows, classes)

    ci_path = os.path.join(cfg["paths"]["tables_dir"], f"auroc_ci{suffix}.csv")
    with open(ci_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scorer", "rep", "class", "auroc", "ci_lo",
                                          "ci_hi"])
        w.writeheader()
        for r in ci_rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in r})

    _print_summary(rows)
    print(f"\nwrote {csv_path}\n      {md_path}\n      {ci_path} "
          f"(per-class {int(100 * 0.95)}% bootstrap CI, {N_BOOT} scan resamples/class)")


def _write_md(path: str, rows: list, classes: list) -> None:
    cols = _columns(classes)
    reps = list(dict.fromkeys(r["rep"] for r in rows))
    n_scorers = len(dict.fromkeys(r["scorer"] for r in rows))
    lines = [f"# Image AUROC - {n_scorers} scorers, {' / '.join(reps)}", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = [str(r["scorer"]), str(r["rep"])]
        cells += [f"{r[c]:.3f}" for c in classes]
        cells += [f"{r['mean']:.3f}",
                  ("nan" if not np.isfinite(r["mean_ci_lo"]) else f"{r['mean_ci_lo']:.3f}"),
                  ("nan" if not np.isfinite(r["mean_ci_hi"]) else f"{r['mean_ci_hi']:.3f}"),
                  f"{r['std']:.3f}",
                  ("nan" if not np.isfinite(r["geom_mean"]) else f"{r['geom_mean']:.3f}")]
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _print_summary(rows: list) -> None:
    print(f"{'scorer':<20}{'rep':<10}{'mean AUROC':>12}{'95% CI':>18}{'±sd':>8}{'geom':>8}")
    print("-" * 76)
    for r in rows:
        geom = "  nan" if not np.isfinite(r["geom_mean"]) else f"{r['geom_mean']:.3f}"
        ci = "nan" if not np.isfinite(r["mean_ci_lo"]) else \
            f"[{r['mean_ci_lo']:.3f}, {r['mean_ci_hi']:.3f}]"
        print(f"{r['scorer']:<20}{r['rep']:<10}{r['mean']:>12.3f}{ci:>18}{r['std']:>8.3f}{geom:>8}")


if __name__ == "__main__":
    main()
