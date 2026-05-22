"""Contamination axis of the scorer-robustness study: does soft retrieval resist bank
contamination better than hard 1-NN?

For each class, a fraction of the memory bank's patches is replaced by patches drawn
from DEFECTIVE test scans -- the "defective samples accidentally in the good bank"
industrial condition. The class's defect scans are split in half BY SCAN (seeded):
patches may enter the bank only from the injection pool, and AUROC is evaluated only
on good scans + the evaluation pool, so no scan is ever scored against its own patches.
Whitening and beta:auto selection are fit on the CONTAMINATED bank (realistic: nobody
knows the bank is dirty). Config: the ``contamination`` block of config.yaml.

    python contamination.py        # the grid at contamination.bank_cap (abs + mm)
    python contamination.py full   # full-bank confirmation of the headline cell
    python contamination.py plot   # regenerate results/figures/contamination.png only

Outputs results/tables/contamination.csv (grid + full-bank rows side by side, told
apart by the bank_cap column) and a per-scorer degradation figure. Every row carries
the config needed to regenerate it.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import yaml

from src import scorers as S
from src.data import test_set, train_bank
from src.evaluate import aggregate_image_normality, image_auroc
from src.repro import stable_rng

# scorer-variant name (the csv `scorer` column) -> (make_scorer name, param overrides)
VARIANTS = {
    "knn":                  ("knn", {}),
    "modern_hopfield":      ("modern_hopfield", {}),
    "modern_hopfield_auto": ("modern_hopfield", {"beta": "auto"}),
    "mahalanobis":          ("mahalanobis", {}),
    "kde":                  ("kde", {}),
}

FIELDS = ["class", "features", "rep", "scorer", "param", "contamination", "seed",
          "auroc", "beta_selected", "bank_cap", "n_bank", "n_injected",
          "n_eval_scans"]

CSV_NAME = "contamination.csv"
FIG_NAME = "contamination.png"


def make_variant(vname: str, cfg: dict) -> tuple[S.Scorer, str]:
    """Fresh scorer for a variant + its resolved-parameter string (the csv `param`)."""
    base, over = VARIANTS[vname]
    p = {**cfg["scorers"][base], **over}
    scorer = S.make_scorer(base, {**cfg, "scorers": {**cfg["scorers"], base: p}})
    return scorer, ",".join(f"{k}={v}" for k, v in p.items())


def split_defect_scans(defect_type: np.ndarray, cls: str, seed: int
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Halve a class's defect scans by scan (seeded): (injection pool, evaluation pool)."""
    defect_idx = np.flatnonzero(defect_type != "good")
    perm = stable_rng("contam-split", cls, seed).permutation(defect_idx)
    half = len(perm) // 2
    return np.sort(perm[:half]), np.sort(perm[half:])


def contaminate(Xi: np.ndarray, pool: np.ndarray, frac: float, cls: str, seed: int
                ) -> tuple[np.ndarray, int]:
    """Replace round(frac*N) random bank columns with random injection-pool columns."""
    n = Xi.shape[1]
    n_rep = int(round(frac * n))
    if n_rep == 0:
        return Xi, 0
    rng = stable_rng("contam-inject", cls, seed, frac)
    pos = rng.choice(n, n_rep, replace=False)
    take = rng.choice(pool.shape[1], n_rep, replace=pool.shape[1] < n_rep)
    out = Xi.copy()
    out[:, pos] = pool[:, take]
    return out, n_rep


def run_cells(cfg: dict, features_list, levels, seeds, scorers_for, reps, bank_cap):
    """Yield one csv row dict per (class, features, rep, scorer, level, seed) cell.

    At 0% contamination the bank is seed-independent, so each scorer is fit/scored
    once and only the (seeded) evaluation pool varies across the seed rows.
    """
    geom_dim = int(cfg.get("geom_dim", 33))
    topk_frac = cfg.get("topk_frac")
    for features in features_list:
        fcfg = {**cfg, "features": features, "bank_cap": bank_cap}
        variant_names = scorers_for(features)
        for cls in cfg["classes"]:
            Xi0 = train_bank(cls, fcfg)
            ts = test_set(cls, fcfg)
            is_good = ts["defect_type"] == "good"
            splits = {s: split_defect_scans(ts["defect_type"], cls, s) for s in seeds}
            pools = {s: np.ascontiguousarray(ts["X"][:, np.isin(ts["scan_index"], splits[s][0])])
                     for s in seeds}
            t0 = time.time()
            for cont in levels:
                # None = the shared uncontaminated bank, evaluated under every seed's pool
                for bseed in ([None] if cont == 0 else seeds):
                    Xi_c, n_inj = (Xi0, 0) if bseed is None else \
                        contaminate(Xi0, pools[bseed], cont, cls, bseed)
                    for rep in reps:
                        Xi_r, X_r = S.condition(rep, Xi_c, ts["X"], geom_dim)
                        for vname in variant_names:
                            scorer, param = make_variant(vname, cfg)
                            patch = scorer.fit(Xi_r).score(X_r)
                            img = aggregate_image_normality(
                                ts["scan_index"], patch, ts["n_scans"], topk_frac)
                            beta_sel = getattr(scorer, "beta_", "")
                            for eseed in (seeds if bseed is None else [bseed]):
                                keep = is_good | np.isin(
                                    np.arange(ts["n_scans"]), splits[eseed][1])
                                yield {
                                    "class": cls, "features": features, "rep": rep,
                                    "scorer": vname, "param": param,
                                    "contamination": cont, "seed": eseed,
                                    "auroc": round(image_auroc(
                                        img[keep], ts["is_anomalous"][keep]), 4),
                                    "beta_selected": beta_sel,
                                    "bank_cap": "" if bank_cap is None else bank_cap,
                                    "n_bank": Xi0.shape[1], "n_injected": n_inj,
                                    "n_eval_scans": int(keep.sum()),
                                }
            print(f"[{features}/{cls}] {len(levels)} levels x {len(reps)} reps x "
                  f"{len(variant_names)} scorers done ({time.time() - t0:.0f}s)", flush=True)


def _csv_path(cfg: dict) -> str:
    return os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)


def write_rows(cfg: dict, rows: list, replacing_full_bank: bool) -> str:
    """Rewrite the csv: new rows replace the same-kind rows (grid vs full-bank),
    the other kind is preserved -- `run` and `full` can be (re)run in any order."""
    path = _csv_path(cfg)
    kept = []
    if os.path.exists(path):
        with open(path) as f:
            kept = [r for r in csv.DictReader(f)
                    if (r["bank_cap"] == "") != replacing_full_bank]
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in kept + rows:
            w.writerow(r)
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

    with open(_csv_path(cfg)) as f:
        rows = [r for r in csv.DictReader(f) if r["bank_cap"] != ""]   # the capped grid
    if not rows:
        print(f"skip {FIG_NAME} (no capped-grid rows yet; run `python contamination.py`)")
        return
    panels = sorted({(r["features"], r["rep"]) for r in rows},
                    key=lambda p: (p[0] != "abs", p[1] != "raw"))
    # >2 panels in one row goes below the print-legible panel-width floor;
    # lay out as a 2-column grid instead once there are more than 2.
    ncols = min(len(panels), 2)
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(style.WIDTH_FULL, 2.6 * nrows),
                             sharey=True, squeeze=False)
    flat_axes = axes.flatten()
    for ax, (features, rep) in zip(flat_axes, panels):
        sub = [r for r in rows if (r["features"], r["rep"]) == (features, rep)]
        for vname in VARIANTS:
            pts = [r for r in sub if r["scorer"] == vname]
            if not pts:
                continue
            levels = sorted({float(r["contamination"]) for r in pts})
            # per level: mean over classes within each seed -> across-seed mean + band
            seed_means = np.array([
                [np.mean([float(r["auroc"]) for r in pts
                          if float(r["contamination"]) == lv and r["seed"] == s])
                 for lv in levels]
                for s in sorted({r["seed"] for r in pts})])
            x = np.array(levels) * 100
            color = style.SCORER_COLORS[vname]
            ax.fill_between(x, seed_means.min(axis=0), seed_means.max(axis=0),
                            color=color, alpha=0.18, lw=0)
            ax.plot(x, seed_means.mean(axis=0), color=color, lw=2,
                    marker=style.SCORER_MARKERS[vname], ms=5,
                    dashes=style.SCORER_DASHES.get(vname, (None, None)),
                    label=vname.replace("_", " "))
        ax.set_title(f"{features}, {rep}", fontsize=9)
        ax.set_xlabel("bank contamination (%)")
        ax.set_xticks(sorted({float(r['contamination']) * 100 for r in sub}))
    for ax in flat_axes[len(panels):]:
        ax.set_axis_off()
    n_cls = len({r["class"] for r in rows})
    for row in range(nrows):
        axes[row][0].set_ylabel(f"mean image AUROC\n({n_cls} classes, eval pool)", fontsize=8)
    flat_axes[0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("Bank contamination: per-scorer degradation "
                 "(band = seed min-max of the class mean)", y=0.99, fontsize=10)
    fig.tight_layout()
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["figures_dir"], FIG_NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main(mode: str = "run", config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    ccfg = cfg["contamination"]
    if mode == "run":
        rows = list(run_cells(cfg, list(ccfg["scorers"]), ccfg["levels"], ccfg["seeds"],
                              lambda ft: ccfg["scorers"][ft], ccfg["reps"],
                              ccfg["bank_cap"]))
        print("wrote", write_rows(cfg, rows, replacing_full_bank=False))
    elif mode == "full":
        fb = ccfg["full_bank"]
        rows = list(run_cells(cfg, fb["features"], fb["levels"], ccfg["seeds"],
                              lambda ft: fb["scorers"], fb["reps"], None))
        print("wrote", write_rows(cfg, rows, replacing_full_bank=True))
    elif mode != "plot":
        raise SystemExit(f"usage: python contamination.py [run|full|plot]  (got {mode!r})")
    plot(cfg)


if __name__ == "__main__":
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "run")
