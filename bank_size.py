"""Bank-size axis of the scorer-robustness study: how fast does each retrieval scorer
degrade as the memory bank shrinks, and does a coverage-maximizing coreset help?

Compares the two subsampling modes behind ``bank_cap`` -- uniform random (the
project's default) vs greedy k-center / farthest-point selection (PatchCore-style,
``src/scorers.kcenter_coreset``) -- at decreasing caps, for the retrieval scorers on
the geometry features. k-center selections are nested by construction, so one
max-cap ordering per (class, seed) yields every smaller cap as a prefix. Config: the
``bank_size`` block of config.yaml.

    python bank_size.py        # the grid -> results/tables/bank_size.csv + figure
    python bank_size.py plot   # regenerate results/figures/bank_size.png only

The figure adds the full-bank AUROC from results/tables/auroc.csv (the baseline
reference numbers) as the no-subsampling asymptote.
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

from contamination import make_variant

FIELDS = ["class", "features", "rep", "scorer", "param", "mode", "bank_cap", "seed",
          "auroc", "beta_selected", "n_bank"]

CSV_NAME = "bank_size.csv"
FIG_NAME = "bank_size.png"


def run_grid(cfg: dict) -> list:
    bcfg = cfg["bank_size"]
    caps = sorted((int(c) for c in bcfg["caps"]), reverse=True)
    geom_dim = int(cfg.get("geom_dim", 33))
    topk_frac = cfg.get("topk_frac")
    fcfg = {**cfg, "features": "abs", "bank_cap": None}    # full bank; we subsample here
    rows = []
    for cls in cfg["classes"]:
        Xi_full = train_bank(cls, fcfg)
        ts = test_set(cls, fcfg)
        t0 = time.time()
        for seed in bcfg["seeds"]:
            # one nested k-center ordering at the largest cap serves every smaller cap
            kc = S.kcenter_coreset(Xi_full, caps[0], seed=seed)
            for cap in caps:
                for mode in bcfg["modes"]:
                    Xi = kc[:, :cap] if mode == "kcenter" else \
                        S.subsample(Xi_full, cap, seed=seed)
                    for rep in bcfg["reps"]:
                        Xi_r, X_r = S.condition(rep, Xi, ts["X"], geom_dim)
                        for vname in bcfg["scorers"]:
                            scorer, param = make_variant(vname, cfg)
                            patch = scorer.fit(Xi_r).score(X_r)
                            img = aggregate_image_normality(
                                ts["scan_index"], patch, ts["n_scans"], topk_frac)
                            rows.append({
                                "class": cls, "features": "abs", "rep": rep,
                                "scorer": vname, "param": param, "mode": mode,
                                "bank_cap": cap, "seed": seed,
                                "auroc": round(image_auroc(img, ts["is_anomalous"]), 4),
                                "beta_selected": getattr(scorer, "beta_", ""),
                                "n_bank": Xi.shape[1],
                            })
        print(f"[{cls}] {len(bcfg['seeds'])} seeds x {len(caps)} caps x "
              f"{len(bcfg['modes'])} modes done ({time.time() - t0:.0f}s)", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Figure (regenerates from the csvs only)                                      #
# --------------------------------------------------------------------------- #
def _full_bank_reference(cfg: dict) -> dict:
    """{(scorer, rep): full-bank mean AUROC} from the baseline table, if present."""
    path = os.path.join(cfg["paths"]["tables_dir"], "auroc.csv")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {(r["scorer"], r["rep"]): float(r["mean"]) for r in csv.DictReader(f)}


def plot(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import style
    style.apply()

    with open(os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)) as f:
        rows = list(csv.DictReader(f))
    ref = _full_bank_reference(cfg)
    reps = sorted({r["rep"] for r in rows}, key=lambda x: x != "raw")
    scorers = [v for v in ("knn", "modern_hopfield", "modern_hopfield_auto")
               if any(r["scorer"] == v for r in rows)]
    fig, axes = plt.subplots(1, len(reps), figsize=style.panel_figsize(len(reps), height=3.4),
                             sharey=True, squeeze=False)
    for ax, rep in zip(axes[0], reps):
        for vname in scorers:
            for mode, ls, alpha in (("uniform", ":", 0.75), ("kcenter", "-", 1.0)):
                pts = [r for r in rows if (r["rep"], r["scorer"], r["mode"])
                       == (rep, vname, mode)]
                if not pts:
                    continue
                caps = sorted({int(r["bank_cap"]) for r in pts})
                seed_means = np.array([
                    [np.mean([float(r["auroc"]) for r in pts
                              if int(r["bank_cap"]) == c and r["seed"] == s])
                     for c in caps]
                    for s in sorted({r["seed"] for r in pts})])
                color = style.SCORER_COLORS[vname]
                ax.fill_between(caps, seed_means.min(axis=0), seed_means.max(axis=0),
                                color=color, alpha=0.15, lw=0)
                ax.plot(caps, seed_means.mean(axis=0), ls, color=color, lw=2,
                        marker=style.SCORER_MARKERS[vname], ms=5, alpha=alpha,
                        label=f"{vname.replace('_', ' ')} ({mode})")
            base = "modern_hopfield" if vname.startswith("modern_hopfield") else vname
            if (base, rep) in ref and vname != "modern_hopfield_auto":
                ax.axhline(ref[(base, rep)], color=style.SCORER_COLORS[vname],
                           ls="--", lw=1, alpha=0.5)
        ax.set_xscale("log")
        caps_all = sorted({int(r["bank_cap"]) for r in rows})
        ax.set_xticks(caps_all)
        ax.set_xticklabels([str(c) for c in caps_all])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f"abs, {rep}  (dashed h-line = full bank)")
        ax.set_xlabel("bank cap (patches, log scale)")
    n_cls = len({r["class"] for r in rows})
    axes[0][0].set_ylabel(f"mean image AUROC ({n_cls} classes)")
    axes[0][0].legend(loc="lower right", fontsize=8.5)
    fig.suptitle("Bank size: uniform vs k-center coreset "
                 "(band = seed min-max of the class mean)", y=0.99)
    fig.tight_layout()
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    out = os.path.join(cfg["paths"]["figures_dir"], FIG_NAME)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main(mode: str = "run", config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if mode == "run":
        rows = run_grid(cfg)
        os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
        path = os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote", path)
    elif mode != "plot":
        raise SystemExit(f"usage: python bank_size.py [run|plot]  (got {mode!r})")
    plot(cfg)


if __name__ == "__main__":
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "run")
