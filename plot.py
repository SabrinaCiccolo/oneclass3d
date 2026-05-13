"""Figures for the scorer comparison.

  results/figures/auroc_bar.png    per-scorer mean image AUROC, raw vs whitened
  results/figures/mh_beta_sweep.png  Modern Hopfield AUROC vs beta, raw vs whitened
                                     (the bias->variance dial; whitening opens an
                                      interior-beta regime the raw curve lacks)
  results/figures/kde_bandwidth_view.png  the SAME sweep re-labeled as a KDE bandwidth
                                     (h = 1/sqrt(beta), exact by the MH==KDE identity),
                                     with Scott's rule marked -- no extra scoring run
  results/figures/mm_vs_geom.png   geometry vs multimodal (only if auroc_mm.csv exists):
                                     (A) per-class k-NN(raw) abs vs mm -- the appearance
                                     rescue of the near-chance classes; (B) per-scorer mean
                                     mm raw vs whitened -- whitening flips from help to harm.

Run after run.py (the bars read results/tables/auroc.csv; the mm panel reads auroc_mm.csv,
written by a `features: mm` run).

    python plot.py
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from src import scorers as S
from src import style
from src.data import test_set, train_bank
from src.evaluate import aggregate_image_normality, image_auroc

REPS = ["raw", "whitened"]


def mh_beta_curve(cfg: dict, betas: list[float]) -> dict:
    """Mean image AUROC of Modern Hopfield at each beta, for raw and whitened."""
    classes = cfg["classes"]
    topk_frac = cfg.get("topk_frac")
    aurocs = {rep: np.zeros((len(classes), len(betas))) for rep in REPS}
    for ci, cls in enumerate(classes):
        Xi_raw = train_bank(cls, cfg)
        ts = test_set(cls, cfg)
        for rep in REPS:
            Xi, X = S.condition(rep, Xi_raw, ts["X"])
            Xi = np.asarray(Xi, dtype=np.float32); X = np.asarray(X, dtype=np.float32)
            n = Xi.shape[1]
            chunk = max(1, S._MAX_ENTRIES // max(n, 1))
            patch = np.empty((len(betas), X.shape[1]), dtype=np.float32)
            for i in range(0, X.shape[1], chunk):
                sims = Xi.T @ X[:, i:i + chunk]
                patch[:, i:i + chunk] = S.mh_logsumexp(sims, betas)
            for bi in range(len(betas)):
                img = aggregate_image_normality(ts["scan_index"], patch[bi], ts["n_scans"], topk_frac)
                aurocs[rep][ci, bi] = image_auroc(img, ts["is_anomalous"])
        print(f"  beta-sweep [{cls}] done", flush=True)
    return {rep: aurocs[rep].mean(axis=0) for rep in REPS}


def plot_bar(cfg: dict) -> None:
    table = _read_table(os.path.join(cfg["paths"]["tables_dir"], "auroc.csv"))
    mean = lambda n, rep: float(table[(n, rep)]["mean"])
    names = S.SCORER_NAMES
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    b_raw = ax.bar(x - w / 2, [mean(n, "raw") for n in names], w, label="raw", color=style.NAVY)
    b_wht = ax.bar(x + w / 2, [mean(n, "whitened") for n in names], w, label="whitened", color=style.ORANGE)
    ax.bar_label(b_raw, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(b_wht, fmt="%.2f", padding=2, fontsize=7)
    ax.axhline(0.5, ls="--", lw=1, color=style.CHANCE, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("mean image AUROC (10 classes)")
    ax.set_ylim(0.45, 1.0)
    ax.set_title("One-class anomaly detection on MVTec 3D-AD: six scorers, raw vs whitened")
    ax.legend()
    fig.tight_layout()
    out = os.path.join(cfg["paths"]["figures_dir"], "auroc_bar.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def plot_beta_sweep(cfg: dict) -> dict:
    betas = [float(b) for b in cfg["beta_sweep"]]
    curves = mh_beta_curve(cfg, betas)
    knn_w = float(_read_table(os.path.join(cfg["paths"]["tables_dir"], "auroc.csv"))[("knn", "whitened")]["mean"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(betas, curves["raw"], "o-", color=style.NAVY, label="raw")
    ax.plot(betas, curves["whitened"], "s-", color=style.ORANGE, label="whitened")
    ax.axhline(knn_w, ls=":", lw=1.6, color=style.RED, label=f"k-NN whitened ({knn_w:.3f})")
    ax.axhline(0.5, ls="--", lw=1, color=style.CHANCE, label="chance")
    ax.set_xscale("log")
    ax.set_xlabel("Modern Hopfield beta  (small = mean / high bias  ->  large = 1-NN / high variance)")
    ax.set_ylabel("mean image AUROC (10 classes)")
    ax.set_title("Modern Hopfield bias->variance dial")
    # The MHN limit: the soft scorer's best only matches k-NN; it cannot beat the nearest neighbour.
    ax.text(0.5, 0.93, "MHN limit: soft scoring does not beat k-NN (best ~ k-NN); large beta -> 1-NN",
            transform=ax.transAxes, fontsize=8.5, color=style.RED, ha="center", va="top",
            bbox=dict(boxstyle="round", fc="#FBEAE7", ec=style.RED, alpha=0.9))
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(cfg["paths"]["figures_dir"], "mh_beta_sweep.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)
    return {"betas": betas, "curves": curves}


def plot_kde_bandwidth_view(cfg: dict, betas: list[float], curves: dict) -> None:
    """Derive the KDE-bandwidth AUROC curve from the MH beta-sweep via the closed-form
    identity h = 1/sqrt(beta) (KDE(h) and ModernHopfield(beta) score every patch through
    the same rank-preserving affine map at matched scale -- tests/test_scorers.py locks
    this as a correctness check -- so the AUROC curve is numerically IDENTICAL, only the
    x-axis unit changes). No new KDE scoring is run; this answers "would a smaller/
    adaptive KDE bandwidth help" directly from data already computed for the beta sweep.

    Marks each class's actual Scott's-rule bandwidth (scott_bandwidth(n, d) at that
    class's bank size) as a shaded band, so the reader can see where Scott's rule falls
    relative to the bias (large h, left) <-> variance (small h, right) curve.
    """
    hs = [1.0 / np.sqrt(b) for b in betas]
    geom_dim = int(cfg.get("geom_dim", 33))
    d = 33 if cfg.get("features", "abs") == "abs" else geom_dim + 384
    scott_hs = [S.scott_bandwidth(train_bank(cls, cfg).shape[1], d) for cls in cfg["classes"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(hs, curves["raw"], "o-", color=style.NAVY, label="raw")
    ax.plot(hs, curves["whitened"], "s-", color=style.ORANGE, label="whitened")
    ax.axhline(0.5, ls="--", lw=1, color=style.CHANCE, label="chance")
    ax.axvspan(min(scott_hs), max(scott_hs), color=style.GREY, alpha=0.25,
              label=f"Scott's rule, per class (h={min(scott_hs):.2f}-{max(scott_hs):.2f})")
    ax.set_xscale("log")
    ax.invert_xaxis()   # small h (variance / near-1NN) on the right, matches the beta sweep's reading direction
    ax.set_xlabel("KDE bandwidth h = 1/sqrt(beta)  (derived from the MH sweep, not re-scored)")
    ax.set_ylabel("mean image AUROC (10 classes)")
    ax.set_title("KDE bandwidth view of the same bias->variance dial")
    ax.text(0.5, 0.06, "Scott's rule sits at the over-smoothed (high-bias) end of the "
            "tested range -> KDE's bias floor is a bandwidth artifact, not a ceiling",
            transform=ax.transAxes, fontsize=8.5, color=style.RED, ha="center", va="bottom",
            bbox=dict(boxstyle="round", fc="#FBEAE7", ec=style.RED, alpha=0.9))
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    out = os.path.join(cfg["paths"]["figures_dir"], "kde_bandwidth_view.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)

    tpath = os.path.join(cfg["paths"]["tables_dir"], "kde_bandwidth_sweep.csv")
    with open(tpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["beta", "h", "raw_mean_auroc", "whitened_mean_auroc"])
        for b, h, r, wh in zip(betas, hs, curves["raw"], curves["whitened"]):
            w.writerow([b, round(h, 4), round(float(r), 4), round(float(wh), 4)])
        w.writerow(["scott_min", round(min(scott_hs), 4), "", ""])
        w.writerow(["scott_max", round(max(scott_hs), 4), "", ""])
    print("wrote", tpath)


def _read_table(path: str) -> dict:
    """auroc csv -> {(scorer, rep): {col: value}} (class/mean cols kept as float)."""
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[(r["scorer"], r["rep"])] = r
    return out


def plot_mm_comparison(cfg: dict) -> None:
    """Geometry vs multimodal, two panels. Skips if no multimodal table yet.

    Reads the canonical geometry table (auroc.csv, full bank) and the multimodal one
    (auroc_mm.csv, written by a `features: mm` run, typically bank_cap-ed), so the mm
    bars are a conservative lower bound on the full-data multimodal numbers.
    """
    tdir = cfg["paths"]["tables_dir"]
    mm_path = os.path.join(tdir, "auroc_mm.csv")
    if not os.path.exists(mm_path):
        print("skip mm_vs_geom.png (no auroc_mm.csv; run a `features: mm` pass first)")
        return
    A, M = _read_table(os.path.join(tdir, "auroc.csv")), _read_table(mm_path)
    classes = cfg["classes"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5))

    # Panel A: per-class k-NN(raw), geometry vs multimodal, weakest-geometry class first
    geom = np.array([float(A[("knn", "raw")][c]) for c in classes])
    mm = np.array([float(M[("knn", "raw")][c]) for c in classes])
    order = np.argsort(geom)
    labels = [classes[i] for i in order]
    x = np.arange(len(classes)); w = 0.4
    axA.bar(x - w / 2, geom[order], w, label="geometry (FPFH, 33d)", color=style.NAVY)
    axA.bar(x + w / 2, mm[order], w, label="multimodal (+DINOv2, 417d)", color=style.GREEN)
    axA.axhline(0.5, ls="--", lw=1, color=style.CHANCE, label="chance")
    axA.set_xticks(x); axA.set_xticklabels(labels, rotation=20, ha="right")
    axA.set_ylabel("image AUROC"); axA.set_ylim(0.45, 1.0)
    axA.set_title("(A) k-NN(raw) per class: RGB rescues the geometry blind spot")
    axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=9, frameon=False)

    # Panel B: per-scorer mean, multimodal raw vs joint-whitened vs block-wise. Joint
    # whitening hurts the multimodal descriptor; block-wise (whiten geometry only) recovers it.
    names = S.SCORER_NAMES
    reps = [("raw", "raw", style.NAVY), ("whitened", "whitened (joint)", style.ORANGE)]
    if all((n, "blockwise") in M for n in names):
        reps.append(("blockwise", "block-wise (geometry only)", style.GREEN))
    if all((n, "cca") in M for n in names):
        reps.append(("cca", "CCA fusion", style.RED))
    xb = np.arange(len(names))
    bw = 0.8 / len(reps)
    for i, (rep, label, color) in enumerate(reps):
        vals = [float(M[(n, rep)]["mean"]) for n in names]
        axB.bar(xb + (i - (len(reps) - 1) / 2) * bw, vals, bw, label=label, color=color)
    axB.axhline(0.5, ls="--", lw=1, color=style.CHANCE, label="chance")
    axB.set_xticks(xb); axB.set_xticklabels(names, rotation=20, ha="right")
    axB.set_ylabel("mean image AUROC (10 classes)"); axB.set_ylim(0.45, 1.0)
    axB.set_title("(B) multimodal: joint whitening hurts; block-wise recovers it")
    axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=9, frameon=False)

    # reserve bottom space for the legends; keep figsize aspect (no bbox_inches) so the
    # regenerated PNG has a fixed, predictable frame size
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.30, wspace=0.18)
    out = os.path.join(cfg["paths"]["figures_dir"], "mm_vs_geom.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("wrote", out)


def main(config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    style.apply()
    os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    plot_bar(cfg)
    plot_mm_comparison(cfg)
    sweep = plot_beta_sweep(cfg)
    plot_kde_bandwidth_view(cfg, sweep["betas"], sweep["curves"])


if __name__ == "__main__":
    main()
