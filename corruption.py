"""Corruption robustness grid: how does detection degrade when RANSAC/DBSCAN/FPFH run
on corrupted geometry?

Two regimes per (class, family, severity) cell: ``clean_bank`` (bank from clean train
scans, corrupted test = deployment drift) and ``corrupted_bank`` (identically corrupted
train + test = a noisy sensor). Detectors: the configs in ``corruption.scorers``
(raw kNN as the reference + whitened Modern Hopfield with beta: auto). Geometry (abs)
features only: the corruptions don't touch the RGB branch, so a multimodal grid would
need paired RGB corruptions. Banks are full (bank_cap null), so the clean cells match
auroc.csv.

    python corruption.py           # score the grid (caches must exist: extract.py corrupt)
    python corruption.py mitigate  # worst-3-cells probe: ransac/cleanup toggled variants
    python corruption.py eps       # the same cells with the rel / adaptive DBSCAN eps
    python corruption.py seeds     # re-score the worst cells at several corruption draws
    python corruption.py plot      # regenerate the per-family figures from the csv

Outputs results/tables/corruption_grid.csv (+ probe rows, told apart by the
``pipeline`` column), results/figures/corruption_<family>.png, results/mitigation.md,
results/eps_modes.md, results/seed_probe.md.
"""
from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np
import yaml

from src import scorers as S
from src.corrupt import cache_suffix
from src.data import test_set, train_bank
from src.evaluate import aggregate_image_normality, image_auroc

from contamination import make_variant

FIELDS = ["class", "corruption", "severity", "severity_param", "regime", "scorer",
          "param", "rep", "pipeline", "auroc", "delta_from_clean", "n_bank",
          "n_test_scans", "seed"]

CSV_NAME = "corruption_grid.csv"
MITIGATION_TAGS = ("nr", "nc", "nrnc")   # ransac off / cleanup off / both off
EPS_TAGS = ("rel", "adaptive")           # cleanup.dbscan_eps_mode variants
SEED_PROBE_PIPELINE = "seed_probe"


def _score_cell(cls: str, bank_feats: str, test_feats: str, scfgs: list, cfg: dict,
                ) -> list:
    """AUROC of every scorer config with bank <- bank_feats and test <- test_feats."""
    geom_dim = int(cfg.get("geom_dim", 33))
    topk_frac = cfg.get("topk_frac")
    Xi0 = train_bank(cls, {**cfg, "features": bank_feats, "bank_cap": None})
    ts = test_set(cls, {**cfg, "features": test_feats})
    out = []
    for sc in scfgs:
        Xi, X = S.condition(sc["rep"], Xi0, ts["X"], geom_dim)
        scorer, param = make_variant(sc["scorer"], cfg)
        patch = scorer.fit(Xi).score(X)
        img = aggregate_image_normality(ts["scan_index"], patch, ts["n_scans"], topk_frac)
        out.append({"scorer": sc["scorer"], "param": param, "rep": sc["rep"],
                    "auroc": round(image_auroc(img, ts["is_anomalous"]), 4),
                    "n_bank": Xi0.shape[1], "n_test_scans": ts["n_scans"]})
    return out


def _cell_rows(cls, family, severity, sev_param, regime, pipeline, scored, clean_ref):
    rows = []
    for r in scored:
        clean = clean_ref.get((cls, r["scorer"], r["rep"]))
        rows.append({"class": cls, "corruption": family, "severity": severity,
                     "severity_param": sev_param, "regime": regime, "scorer": r["scorer"],
                     "param": r["param"], "rep": r["rep"], "pipeline": pipeline,
                     "auroc": r["auroc"],
                     "delta_from_clean": "" if clean is None else round(r["auroc"] - clean, 4),
                     "n_bank": r["n_bank"], "n_test_scans": r["n_test_scans"]})
    return rows


def run_grid(cfg: dict) -> list:
    ccfg = cfg["corruption"]
    rows, clean_ref = [], {}
    for cls in ccfg["classes"]:
        t0 = time.time()
        scored = _score_cell(cls, "abs", "abs", ccfg["scorers"], cfg)
        for r in scored:
            clean_ref[(cls, r["scorer"], r["rep"])] = r["auroc"]
        rows += _cell_rows(cls, "none", 0, 0.0, "clean", "default", scored, clean_ref)
        print(f"[{cls}] clean baseline done ({time.time() - t0:.0f}s)", flush=True)
        for family, sevs in ccfg["severities"].items():
            t0 = time.time()
            for severity in range(1, len(sevs) + 1):
                suffix = f"abs__{family}{severity}"
                for regime in ccfg["regimes"]:
                    bank_feats = "abs" if regime == "clean_bank" else suffix
                    scored = _score_cell(cls, bank_feats, suffix, ccfg["scorers"], cfg)
                    rows += _cell_rows(cls, family, severity, sevs[severity - 1],
                                       regime, "default", scored, clean_ref)
            print(f"[{cls}] {family} x{len(sevs)} x {len(ccfg['regimes'])} regimes "
                  f"done ({time.time() - t0:.0f}s)", flush=True)
    return rows


# --------------------------------------------------------------------------- #
# Mitigation probe: worst cells re-run with the cleanup stages toggled         #
# --------------------------------------------------------------------------- #
def worst_cells(rows: list, n: int) -> list:
    """The n (class, family, severity, regime) cells with the deepest AUROC drop."""
    cells: dict = {}
    for r in rows:
        if r["pipeline"] != "default" or r["corruption"] == "none":
            continue
        if r["delta_from_clean"] == "":
            continue
        key = (r["class"], r["corruption"], int(r["severity"]), r["regime"])
        cells[key] = min(cells.get(key, 0.0), float(r["delta_from_clean"]))
    return sorted(cells, key=cells.get)[:n]


def run_mitigation(cfg: dict, tags: tuple = MITIGATION_TAGS) -> tuple[list, list]:
    """Re-score the worst grid cells under the pipeline variants ``tags`` (see
    ``src.preprocess.pipeline_cfg``; variant caches are extracted on demand through the
    same extract path). Returns (rows, cells)."""
    from extract import extract_corrupt_cell
    from src.features import cache_class_split, cache_path
    from src.preprocess import pipeline_cfg

    ccfg = cfg["corruption"]
    grid = read_rows(cfg)
    clean_ref = {(r["class"], r["scorer"], r["rep"]): float(r["auroc"])
                 for r in grid if r["corruption"] == "none" and r["pipeline"] == "default"}
    cells = worst_cells(grid, int(ccfg.get("mitigation_cells", 3)))
    print("worst cells:", cells, flush=True)
    rows = []
    for cls, family, severity, regime in cells:
        sev_param = ccfg["severities"][family][severity - 1]
        for tag in tags:
            t0 = time.time()
            # corrupted test cache under the variant pipeline (skips if cached)
            extract_corrupt_cell(cls, "test", family, severity, cfg, pipeline_tag=tag)
            if regime == "clean_bank":
                # the bank must be processed by the SAME pipeline variant: clean
                # scans, toggles applied (cache suffix e.g. '__nr')
                if not os.path.exists(cache_path(cls, "train", cfg, f"__{tag}")):
                    cache_class_split(cls, "train", pipeline_cfg(cfg, tag),
                                      suffix=f"__{tag}")
                bank_feats = f"abs__{tag}"
            else:
                extract_corrupt_cell(cls, "train", family, severity, cfg, pipeline_tag=tag)
                bank_feats = f"abs__{family}{severity}__{tag}"
            scored = _score_cell(cls, bank_feats, f"abs__{family}{severity}__{tag}",
                                 ccfg["scorers"], cfg)
            rows += _cell_rows(cls, family, severity, sev_param, regime, tag, scored,
                               clean_ref)
            print(f"[{cls}] {family}{severity}/{regime}/{tag} done "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return rows, cells


def run_seed_probe(cfg: dict) -> tuple[list, list]:
    """Multi-seed re-score of the SAME worst cells as the mitigation probe.

    Every ``corruption_grid.csv`` row is a single deterministic corruption draw per
    scan (unlike contamination.py / bank_size.py, which average 3 seeds) -- a concern
    for noisy cells. Rather than re-running the full grid at
    ``len(seeds))x`` the cost, this re-scores just the mitigation probe's worst cells at
    ``corruption.seeds`` (default [0,1,2]; seed 0 reuses the existing cache bit-exact)
    and reports the mean +/- band. Returns (rows, cells).
    """
    from extract import extract_corrupt_cell

    ccfg = cfg["corruption"]
    seeds = [int(s) for s in ccfg.get("seeds", [0])]
    grid = read_rows(cfg)
    cells = worst_cells(grid, int(ccfg.get("mitigation_cells", 3)))
    print("seed-probe cells:", cells, "seeds:", seeds, flush=True)
    rows = []
    for cls, family, severity, regime in cells:
        sev_param = ccfg["severities"][family][severity - 1]
        for seed in seeds:
            t0 = time.time()
            extract_corrupt_cell(cls, "test", family, severity, cfg, seed=seed)
            test_suffix = cache_suffix(family, severity, seed=seed)
            if regime == "corrupted_bank":
                extract_corrupt_cell(cls, "train", family, severity, cfg, seed=seed)
                bank_feats = f"abs{test_suffix}"
            else:
                bank_feats = "abs"
            scored = _score_cell(cls, bank_feats, f"abs{test_suffix}", ccfg["scorers"], cfg)
            for r in scored:
                rows.append({"class": cls, "corruption": family, "severity": severity,
                            "severity_param": sev_param, "regime": regime,
                            "scorer": r["scorer"], "param": r["param"], "rep": r["rep"],
                            "pipeline": SEED_PROBE_PIPELINE, "auroc": r["auroc"],
                            "delta_from_clean": "", "n_bank": r["n_bank"],
                            "n_test_scans": r["n_test_scans"], "seed": seed})
            print(f"[{cls}] {family}{severity}/{regime}/seed{seed} done "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return rows, cells


def write_seed_probe_md(cfg: dict, cells: list) -> str:
    """results/seed_probe.md: mean +/- band (across corruption.seeds) for the worst cells."""
    rows = [r for r in read_rows(cfg) if r["pipeline"] == SEED_PROBE_PIPELINE]
    by: dict = {}
    for r in rows:
        key = (r["class"], r["corruption"], r["severity"], r["regime"], r["scorer"], r["rep"])
        by.setdefault(key, []).append(float(r["auroc"]))
    lines = ["# Corruption seed-probe - mitigation cells, multiple corruption draws",
             "",
             "Each `corruption_grid.csv` cell is a single deterministic corruption draw per",
             "scan; this re-scores the mitigation probe's worst cells at "
             f"{cfg['corruption'].get('seeds', [0])} independent draws and reports the",
             "across-seed mean and min-max band, so a cell's wobble can be told apart from a",
             "real effect.", ""]
    header = "| class | cell | regime | scorer (rep) | n seeds | mean | min | max | band |"
    lines += [header, "|" + "---|" * 9]
    for cls, family, severity, regime in cells:
        for key, vals in sorted(by.items()):
            if key[:4] != (cls, family, str(severity), regime):
                continue
            vals = np.array(vals)
            lines.append(f"| {cls} | {family}{severity} | {regime} | {key[4]} ({key[5]}) | "
                         f"{len(vals)} | {vals.mean():.3f} | {vals.min():.3f} | "
                         f"{vals.max():.3f} | {vals.max() - vals.min():.3f} |")
    out = os.path.join("results", "seed_probe.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def write_mitigation_md(cfg: dict, cells: list) -> str:
    """results/mitigation.md: before/after table for the probed cells."""
    rows = read_rows(cfg)
    by = {}
    for r in rows:
        key = (r["class"], r["corruption"], r["severity"], r["regime"],
               r["scorer"], r["rep"])
        by.setdefault(key, {})[r["pipeline"]] = r
    lines = ["# Mitigation probe - worst corruption cells, cleanup stages toggled",
             "",
             "The pipeline's own cleanup stages are the natural mitigations: `nr` = RANSAC",
             "plane removal off, `nc` = DBSCAN cleanup off, `nrnc` = both off (applied to",
             "bank and test alike; variant caches via the same extract path). `recovery` =",
             "variant AUROC - default-pipeline AUROC on the same corrupted cell.", ""]
    header = ("| class | cell | regime | scorer (rep) | clean | corrupted (default) | "
              "nr | nc | nrnc | best recovery |")
    lines += [header, "|" + "---|" * 10]
    for cls, family, severity, regime in cells:
        for key, variants in sorted(by.items()):
            if key[:4] != (cls, family, str(severity), regime) or "default" not in variants:
                continue
            d = variants["default"]
            clean = float(d["auroc"]) - float(d["delta_from_clean"])
            vals = {t: (float(variants[t]["auroc"]) if t in variants else None)
                    for t in MITIGATION_TAGS}
            rec = max((v - float(d["auroc"])) for v in vals.values() if v is not None)
            cells_txt = " | ".join("-" if vals[t] is None else f"{vals[t]:.3f}"
                                   for t in MITIGATION_TAGS)
            lines.append(f"| {cls} | {family}{severity} | {regime} | "
                         f"{key[4]} ({key[5]}) | {clean:.3f} | {float(d['auroc']):.3f} | "
                         f"{cells_txt} | {rec:+.3f} |")
    out = os.path.join("results", "mitigation.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def write_eps_md(cfg: dict, cells: list) -> str:
    """results/eps_modes.md: the probed cells under each DBSCAN eps mode."""
    by = {}
    for r in read_rows(cfg):
        key = (r["class"], r["corruption"], r["severity"], r["regime"], r["scorer"], r["rep"])
        by.setdefault(key, {})[r["pipeline"]] = float(r["auroc"])
    cols = ("default", "nc", *EPS_TAGS)
    lines = ["# DBSCAN eps modes - worst corruption cells",
             "",
             "`default` = fixed eps (`cleanup.dbscan_eps`), `nc` = DBSCAN cleanup off, `rel` =",
             "eps scaled by the bbox diagonal, `adaptive` = per-scan k-distance eps. Bank and",
             "test are extracted with the same variant.", "",
             "| class | cell | regime | scorer (rep) | " + " | ".join(cols) + " |",
             "|" + "---|" * (4 + len(cols))]
    for cls, family, severity, regime in cells:
        for key, v in sorted(by.items()):
            if key[:4] != (cls, family, str(severity), regime) or "default" not in v:
                continue
            vals = " | ".join("-" if c not in v else f"{v[c]:.3f}" for c in cols)
            lines.append(f"| {cls} | {family}{severity} | {regime} | {key[4]} ({key[5]}) | "
                         f"{vals} |")
    out = os.path.join("results", "eps_modes.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


# --------------------------------------------------------------------------- #
# csv io + figure                                                              #
# --------------------------------------------------------------------------- #
def _csv_path(cfg: dict) -> str:
    return os.path.join(cfg["paths"]["tables_dir"], CSV_NAME)


def read_rows(cfg: dict) -> list:
    with open(_csv_path(cfg)) as f:
        return list(csv.DictReader(f))


def write_rows(cfg: dict, rows: list, keep_pipelines: tuple = ()) -> str:
    path = _csv_path(cfg)
    kept = []
    if keep_pipelines and os.path.exists(path):
        kept = [r for r in read_rows(cfg) if r["pipeline"] in keep_pipelines]
    os.makedirs(cfg["paths"]["tables_dir"], exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in kept + rows:
            w.writerow(r)
    return path


def plot(cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src import style
    style.apply()

    rows = [r for r in read_rows(cfg) if r["pipeline"] == "default"]
    classes = cfg["corruption"]["classes"]
    scfgs = [(sc["scorer"], sc["rep"]) for sc in cfg["corruption"]["scorers"]]
    for family in cfg["corruption"]["severities"]:
        sub = [r for r in rows if r["corruption"] == family]
        if not sub:
            continue
        fig, axes = plt.subplots(1, len(classes),
                                 figsize=style.panel_figsize(len(classes), height=3.2),
                                 sharey=True, squeeze=False)
        for ax, cls in zip(axes[0], classes):
            for scorer, rep in scfgs:
                color = style.SCORER_COLORS[scorer]
                clean = [float(r["auroc"]) for r in rows
                         if (r["class"], r["corruption"], r["scorer"], r["rep"])
                         == (cls, "none", scorer, rep)]
                if clean:
                    ax.axhline(clean[0], color=color, ls=":", lw=1.4, alpha=0.6)
                for regime, ls in (("clean_bank", "-"), ("corrupted_bank", "--")):
                    pts = sorted((int(r["severity"]), float(r["auroc"])) for r in sub
                                 if (r["class"], r["regime"], r["scorer"], r["rep"])
                                 == (cls, regime, scorer, rep))
                    if pts:
                        ax.plot([p[0] for p in pts], [p[1] for p in pts], ls,
                                color=color, lw=2, ms=5,
                                marker=style.SCORER_MARKERS[scorer],
                                label=f"{scorer} {rep}, {regime}")
            ax.axhline(0.5, ls="--", lw=1, color=style.CHANCE)
            sevs = cfg["corruption"]["severities"][family]
            ax.set_xticks(range(1, len(sevs) + 1))
            ax.set_xticklabels([f"{i + 1}\n({v:g})" for i, v in enumerate(sevs)])
            ax.set_xlabel("severity (scale-relative parameter)")
            ax.set_title(cls)
        axes[0][0].set_ylabel("image AUROC")
        axes[0][0].legend(loc="lower left", fontsize=8)
        fig.suptitle(f"Corruption: {family}  (dotted = clean baseline, "
                     "solid = clean bank, dashed = corrupted bank)", y=0.99)
        fig.tight_layout()
        os.makedirs(cfg["paths"]["figures_dir"], exist_ok=True)
        out = os.path.join(cfg["paths"]["figures_dir"], f"corruption_{family}.png")
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


def main(mode: str = "run", config_path: str = "config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if mode == "run":
        rows = run_grid(cfg)
        print("wrote", write_rows(cfg, rows, keep_pipelines=(*MITIGATION_TAGS, *EPS_TAGS,
                                                             SEED_PROBE_PIPELINE)))
    elif mode == "mitigate":
        rows, cells = run_mitigation(cfg)
        print("wrote", write_rows(cfg, rows, keep_pipelines=("default", *EPS_TAGS,
                                                             SEED_PROBE_PIPELINE)))
        print("wrote", write_mitigation_md(cfg, cells))
    elif mode == "eps":
        rows, cells = run_mitigation(cfg, EPS_TAGS)
        print("wrote", write_rows(cfg, rows, keep_pipelines=("default", *MITIGATION_TAGS,
                                                             SEED_PROBE_PIPELINE)))
        print("wrote", write_eps_md(cfg, cells))
    elif mode == "seeds":
        rows, cells = run_seed_probe(cfg)
        print("wrote", write_rows(cfg, rows, keep_pipelines=("default", *MITIGATION_TAGS,
                                                             *EPS_TAGS)))
        print("wrote", write_seed_probe_md(cfg, cells))
    elif mode != "plot":
        raise SystemExit(f"usage: python corruption.py [run|mitigate|eps|seeds|plot]  "
                         f"(got {mode!r})")
    plot(cfg)


if __name__ == "__main__":
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "run")
