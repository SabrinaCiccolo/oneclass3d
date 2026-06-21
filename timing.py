"""Wall-clock latency and memory footprint of extraction and scoring (CPU-only).

Three measurements on this machine (specs recorded in results/timing_summary.md), all
on real scans:

  1. Feature extraction: ``extract.py``'s per-scan pipeline (preprocess -> FPFH ->
     patch-pool -> L2), single-threaded (OMP_NUM_THREADS=1, as in extract.py) on a few
     real scans per class. It runs in its own subprocess (``timing.py extract-only``,
     spawned by ``run``): OMP_NUM_THREADS=1 must be set before Open3D is imported and
     must not leak into the scoring benchmark, which uses the same unpinned BLAS
     threading as ``run.py``.
  2. Scorer fit/score latency vs bank size: the six scorers plus Mahalanobis, on
     bank_size.py's bank_cap grid plus the full bank.
  3. Memory footprint: bytes held by a FITTED scorer (every ndarray reachable from its
     ``__dict__``, recursively) at each bank cap. True memory-bank scorers (kNN /
     Modern Hopfield / KDE) grow with the bank; compressed ones (Mahalanobis / OC-SVM)
     keep a fixed-size summary once fit.

    python timing.py          # extraction + scoring benches -> results/tables/timing_*.csv
    python timing.py plot     # regenerate results/figures/timing.png only

Whitening is timed separately (``bench_whitening``): one shrunk-covariance
eigendecomposition per class (O(D^2 N + D^3)), shared by every scorer that follows, so
it is not folded into each scorer's fit time.
"""
from __future__ import annotations

import csv
import os
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import yaml

from src import scorers as S
from src.data import test_set, train_bank

EXTRACTION_CSV = "timing_extraction.csv"
SCORING_CSV = "timing_scoring.csv"
FIG_NAME = "timing.png"
SUMMARY_MD = "timing_summary.md"

SCORER_ORDER = [*S.SCORER_NAMES, "mahalanobis"]   # the six scorers + Mahalanobis


# --------------------------------------------------------------------------- #
# 1. Feature extraction latency (real scans, the exact extract.py pipeline)    #
# --------------------------------------------------------------------------- #
def bench_extraction(cfg: dict, n_scans_per_class: int = 3) -> list[dict]:
    """Time ``n_scans_per_class`` real TEST scans per class through the exact geom
    pipeline (preprocess_scan -> fpfh_per_point -> pool_to_patches -> l2_normalize),
    single-threaded. Nothing is written to the official cache."""
    from src.data_io import iter_scans
    from src.features import (FPFH_DIM, fpfh_per_point, l2_normalize, pool_to_patches,
                              seed_open3d)
    from src.preprocess import preprocess_scan

    seed_open3d(cfg)
    size = int(cfg["resample_grid"])
    tol = float(cfg["fpfh"]["norm_tol"])
    rows = []
    for cls in cfg["classes"]:
        n_done = 0
        for rec in iter_scans(cls, "test", cfg["paths"]["data_root"]):
            if n_done >= n_scans_per_class:
                break
            t0 = time.perf_counter()
            rs_xyz, cleaned = preprocess_scan(rec.xyz, rec.valid_mask, cfg)
            grid = np.zeros((size, size, FPFH_DIM), dtype=np.float32)
            grid[cleaned] = fpfh_per_point(rs_xyz, cleaned, cfg)
            patches, patch_valid = pool_to_patches(grid, cleaned, cfg)
            kept = np.flatnonzero(patch_valid)
            if kept.size:
                l2_normalize(patches[kept], tol)
            ms = (time.perf_counter() - t0) * 1000.0
            rows.append({"class": cls, "scan_id": rec.scan_id, "n_valid_points":
                        int(cleaned.sum()), "n_patches": int(kept.size), "ms": round(ms, 2)})
            n_done += 1
        print(f"[{cls}] {n_done} scans timed", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# 2+3. Scorer fit/score latency + memory footprint vs bank size                #
# --------------------------------------------------------------------------- #
def _nbytes(obj, seen: set | None = None) -> int:
    """Bytes of every ndarray reachable from ``obj`` (recursing into attributes, list/
    tuple/dict members, and nested objects' ``__dict__``). Generic over scorer/sklearn
    internals -- a fitted scorer or a fitted sklearn estimator both just expose their
    learned state as attributes, so this needs no per-scorer maintenance."""
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, (list, tuple)):
        return sum(_nbytes(x, seen) for x in obj)
    if isinstance(obj, dict):
        return sum(_nbytes(v, seen) for v in obj.values())
    if hasattr(obj, "__dict__"):
        return sum(_nbytes(v, seen) for v in vars(obj).values())
    return 0


def bench_whitening(cfg: dict, classes: list) -> list[dict]:
    """Whitener.fit cost at full bank, per class -- a one-time O(D^2 N + D^3) cost
    shared by every scorer run under `rep: whitened`, timed once rather than folded
    into each scorer's fit time (which would double-count it)."""
    rows = []
    fcfg = {**cfg, "features": "abs", "bank_cap": None}
    for cls in classes:
        Xi = train_bank(cls, fcfg)
        t0 = time.perf_counter()
        S.Whitener().fit(Xi)
        ms = (time.perf_counter() - t0) * 1000.0
        rows.append({"class": cls, "n_bank": Xi.shape[1], "fit_ms": round(ms, 2)})
    return rows


def bench_scoring(cfg: dict, classes: list, caps: list, scorer_names: list,
                  repeats: int = 2) -> list[dict]:
    """Fit/score latency + fitted-scorer memory footprint, raw features, for every
    (class, bank_cap, scorer). ``rep='raw'`` only: a scorer's fit/score cost is driven
    by bank size and dimensionality, not by whether the (fixed, D x D) whitening
    transform was applied beforehand (see ``bench_whitening`` for that cost). Takes the
    MIN over ``repeats`` (standard microbenchmark practice -- filters scheduling noise,
    reports the achievable latency) and also keeps the max for a jitter column.
    """
    rows = []
    for cls in classes:
        ts = test_set(cls, {**cfg, "features": "abs"})
        n_scans = ts["n_scans"]
        n_test_patches = ts["X"].shape[1]
        for cap in caps:
            fcfg = {**cfg, "features": "abs", "bank_cap": cap}
            Xi = train_bank(cls, fcfg)
            n_bank = Xi.shape[1]
            for name in scorer_names:
                fit_ms, score_ms, mb = [], [], None
                for _ in range(repeats):
                    scorer = S.make_scorer(name, cfg)
                    t0 = time.perf_counter()
                    scorer.fit(Xi)
                    fit_ms.append((time.perf_counter() - t0) * 1000.0)
                    t0 = time.perf_counter()
                    scorer.score(ts["X"])
                    score_ms.append((time.perf_counter() - t0) * 1000.0)
                    mb = _nbytes(scorer) / 1e6
                rows.append({
                    "class": cls, "scorer": name, "bank_cap": "" if cap is None else cap,
                    "n_bank": n_bank, "n_test_scans": n_scans,
                    "n_test_patches": n_test_patches,
                    "fit_ms": round(min(fit_ms), 3), "fit_ms_max": round(max(fit_ms), 3),
                    "score_ms": round(min(score_ms), 3), "score_ms_max": round(max(score_ms), 3),
                    "score_ms_per_scan": round(min(score_ms) / max(n_scans, 1), 4),
                    "score_us_per_patch": round(1000.0 * min(score_ms) / max(n_test_patches, 1), 4),
                    "footprint_mb": round(mb, 4),
                })
            print(f"[{cls}] cap={cap} ({n_bank} patches): {len(scorer_names)} scorers timed",
                  flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Figure + summary (regenerate from the csvs only)                             #
# --------------------------------------------------------------------------- #
def plot(cfg: dict) -> None:
    import os as _os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import style
    style.apply()

    tdir = cfg["paths"]["tables_dir"]
    with open(_os.path.join(tdir, SCORING_CSV)) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"skip {FIG_NAME} (no {SCORING_CSV} rows yet; run `python timing.py`)")
        return

    scorers = [s for s in SCORER_ORDER if any(r["scorer"] == s for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax = axes[0]
    for name in scorers:
        pts = sorted(((int(r["n_bank"]), float(r["score_ms_per_scan"]))
                     for r in rows if r["scorer"] == name), key=lambda p: p[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, color=style.SCORER_COLORS.get(name, style.TEXT),
                  marker=style.SCORER_MARKERS.get(name, "o"), s=26, alpha=0.75, label=name.replace("_", " "))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bank size (patches, log scale)")
    ax.set_ylabel("score latency (ms/scan, log scale)")
    ax.set_title("Scoring latency vs bank size")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1]
    for name in scorers:
        pts = sorted(((int(r["n_bank"]), float(r["footprint_mb"]))
                     for r in rows if r["scorer"] == name), key=lambda p: p[0])
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, color=style.SCORER_COLORS.get(name, style.TEXT),
                  marker=style.SCORER_MARKERS.get(name, "o"), s=26, alpha=0.75, label=name.replace("_", " "))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("bank size (patches, log scale)")
    ax.set_ylabel("fitted-scorer memory (MB, log scale)")
    ax.set_title("Memory footprint vs bank size")

    fig.suptitle("Timing/memory study: CPU-only, single scan, one class "
                 "(points = class x bank_cap cells)", y=1.02)
    fig.tight_layout()
    _os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
    out = _os.path.join(cfg["paths"]["figures_dir"], FIG_NAME)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _machine_spec() -> str:
    try:
        import multiprocessing
        n_cpu = multiprocessing.cpu_count()
    except Exception:
        n_cpu = -1
    return f"{platform.processor() or platform.machine()}, {n_cpu} logical CPUs, {platform.system()} {platform.release()}"


def write_summary(cfg: dict, extraction_rows: list, whitening_rows: list,
                  scoring_rows: list) -> str:
    """results/timing_summary.md: headline wall-clock + memory-footprint numbers."""
    ext_ms = [r["ms"] for r in extraction_rows]
    knn_full = [r for r in scoring_rows if r["scorer"] == "knn" and r["bank_cap"] == ""]
    knn_scan_ms = statistics.mean(float(r["score_ms_per_scan"]) for r in knn_full) if knn_full else float("nan")
    end_to_end_ms = statistics.mean(ext_ms) + knn_scan_ms if ext_ms else float("nan")
    lines = [
        "# Timing / memory-footprint study", "",
        f"Machine: {_machine_spec()}. CPU-only, single-threaded extraction "
        "(`OMP_NUM_THREADS=1`, matching `extract.py`'s determinism setting); scoring "
        "uses whatever BLAS thread count the environment provides (unpinned, like "
        "`run.py`). Wall-clock, one process, nothing else pinned to the machine "
        "during the run -- read these as this-machine numbers, not absolute claims.",
        "",
        "## Headline", "",
        f"- Feature extraction (real scans, this repo's own pipeline): "
        f"**{statistics.mean(ext_ms):.0f} ms/scan** (median {statistics.median(ext_ms):.0f}, "
        f"n={len(ext_ms)} scans across {len(cfg['classes'])} classes).",
        f"- Scoring, raw kNN (k=1), FULL bank (the reported/reproduction config): "
        f"**{knn_scan_ms:.2f} ms/scan**.",
        f"- End-to-end (extraction + kNN score), full bank: "
        f"**~{end_to_end_ms:.0f} ms/scan ≈ {1000.0 / end_to_end_ms:.1f} scans/sec**, "
        "single CPU process, no batching across scans.",
        "",
        "## Reproduction", "",
        "`python timing.py` -> `results/tables/timing_extraction.csv`, "
        "`results/tables/timing_scoring.csv`, `results/figures/timing.png`, this file.",
    ]
    path = os.path.join("results", SUMMARY_MD)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def _write_csv(path: str, rows: list, fields: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("wrote", path)


def main(mode: str = "run", config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg.get("timing", {"n_scans_per_class": 3, "caps": [1000, 4000, 12000, None],
                              "repeats": 2})
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)

    if mode == "extract-only":
        # Launched as its own process (see 'run' below) so OMP_NUM_THREADS=1 -- required
        # BEFORE Open3D is imported, for extraction determinism, same as extract.py --
        # never leaks into the scoring benchmark's (unpinned, like run.py) process.
        ext_rows = bench_extraction(cfg, tcfg.get("n_scans_per_class", 3))
        _write_csv(os.path.join(cfg["paths"]["tables_dir"], EXTRACTION_CSV), ext_rows,
                  ["class", "scan_id", "n_valid_points", "n_patches", "ms"])
        return

    if mode == "run":
        subprocess.run([sys.executable, os.path.abspath(__file__), "extract-only", config_path],
                       env={**os.environ, "OMP_NUM_THREADS": "1"}, check=True, cwd=os.getcwd())
        with open(os.path.join(cfg["paths"]["tables_dir"], EXTRACTION_CSV)) as f:
            ext_rows = list(csv.DictReader(f))

        whit_rows = bench_whitening(cfg, cfg["classes"])
        score_rows = bench_scoring(cfg, cfg["classes"], tcfg.get("caps", [1000, 4000, 12000, None]),
                                   SCORER_ORDER, tcfg.get("repeats", 2))
        _write_csv(os.path.join(cfg["paths"]["tables_dir"], SCORING_CSV), score_rows,
                  ["class", "scorer", "bank_cap", "n_bank", "n_test_scans", "n_test_patches",
                   "fit_ms", "fit_ms_max", "score_ms", "score_ms_max", "score_ms_per_scan",
                   "score_us_per_patch", "footprint_mb"])
        _write_csv(os.path.join(cfg["paths"]["tables_dir"], "timing_whitening.csv"), whit_rows,
                  ["class", "n_bank", "fit_ms"])
        print("wrote", write_summary(cfg, [{"ms": float(r["ms"])} for r in ext_rows],
                                     whit_rows, score_rows))
    elif mode != "plot":
        raise SystemExit(f"usage: python timing.py [run|plot]  (got {mode!r})")
    plot(cfg)


if __name__ == "__main__":
    _mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    _cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"
    main(mode=_mode, config_path=_cfg_path)
