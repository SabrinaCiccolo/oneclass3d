"""End-to-end integration: raw synthetic scans -> extract -> run.evaluate_class.

Covers the extraction path (preprocess -> FPFH -> pool -> L2 -> cache) and the scoring
path (load -> condition -> score -> aggregate -> AUROC) on a tiny synthetic dataset, so
the full extract->run chain is exercised without the real 24 GB MVTec download.
"""
import os

import numpy as np
import pytest
import yaml

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("open3d")

from src.features import cache_class_split  # noqa: E402  (after importorskip)
import run  # noqa: E402

CLS = "widget"


def _write_scan(path: str, defect: bool = False) -> None:
    """Background plane (RANSAC removes it) + a raised central object (kept)."""
    H = W = 120
    xs = np.linspace(-0.15, 0.15, W).astype(np.float32)
    ys = np.linspace(-0.15, 0.15, H).astype(np.float32)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full((H, W), 0.70, np.float32)                  # background plane at the border
    obj = np.zeros((H, W), bool)
    obj[H // 5:H - H // 5, W // 5:W - W // 5] = True        # central object, closer to camera
    Z[obj] = 0.60 - 0.01 * np.cos(8 * X[obj]) * np.cos(8 * Y[obj])
    if defect:
        d = np.zeros((H, W), bool)
        d[H // 2:H // 2 + 12, W // 2:W // 2 + 12] = True
        Z[d & obj] -= 0.03                                  # localized geometric dent
    tifffile.imwrite(path, np.stack([X, Y, Z], axis=-1).astype(np.float32), photometric="rgb")


def _build_dataset(root) -> None:
    def folder(split, defect):
        p = os.path.join(str(root), CLS, split, defect, "xyz")
        os.makedirs(p, exist_ok=True)
        return p
    for i in range(3):
        _write_scan(os.path.join(folder("train", "good"), f"{i:03d}.tiff"))
    for i in range(2):
        _write_scan(os.path.join(folder("test", "good"), f"{i:03d}.tiff"))
    for i in range(2):
        _write_scan(os.path.join(folder("test", "bump"), f"{i:03d}.tiff"), defect=True)
    for i in range(3):
        _write_scan(os.path.join(folder("validation", "good"), f"{i:03d}.tiff"))


def _cfg(tmp_path) -> dict:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {"cache_dir": str(tmp_path / "cache"), "data_root": str(tmp_path / "data"),
                    "tables_dir": str(tmp_path / "tables"), "figures_dir": str(tmp_path / "figures")}
    cfg["classes"] = [CLS]
    cfg["features"] = "abs"
    cfg["bank_cap"] = None
    cfg["resample_grid"] = 56          # keeps the resample loop tiny (still 28x28 patches)
    cfg["min_valid_per_patch"] = 1
    cfg["cleanup"] = {"dbscan_eps": 0.01, "dbscan_min_points": 10}
    cfg["fpfh"] = {"normal_radius_abs": 0.02, "fpfh_radius_abs": 0.05,
                   "max_nn": 100, "norm_tol": 1.0e-4}
    cfg["defect_map"] = dict(cfg.get("defect_map", {}), bump="geometric")
    return cfg


def test_extract_then_run_end_to_end(tmp_path):
    _build_dataset(tmp_path / "data")
    cfg = _cfg(tmp_path)

    # extract: raw tiff -> cache, with the documented schema and unit-sphere descriptors
    for split in ("train", "test"):
        path = cache_class_split(CLS, split, cfg)
        d = np.load(path, allow_pickle=False)
        assert d["patches"].shape[0] > 0 and d["patches"].shape[1] == cfg["geom_dim"]
        assert set(d.files) >= {"patches", "patch_idx", "scan_index",
                                "scan_ids", "defect_type", "patch_counts"}
        np.testing.assert_allclose(np.linalg.norm(d["patches"], axis=1), 1.0, atol=1e-4)

    # run: every (scorer, rep) yields a finite image AUROC in [0, 1]
    out = run.evaluate_class(CLS, cfg)
    assert set(out) == {(s, r) for s in run.S.SCORER_NAMES for r in run.reps_for(cfg)}
    for key, val in out.items():
        a = val["overall"]
        assert np.isfinite(a) and 0.0 <= a <= 1.0, f"{key}: bad AUROC {a}"

    # scoring is deterministic given the cache
    out2 = run.evaluate_class(CLS, cfg)
    assert all(out[k]["overall"] == out2[k]["overall"] for k in out)


def test_operating_point_end_to_end(tmp_path):
    """extract validation split -> operating_point.evaluate_class: a threshold
    calibrated on validation-only scores, applied fixed to test, yields well-formed
    confusion counts/metrics for every (scorer, rep, target_fpr) cell."""
    import operating_point as OP

    _build_dataset(tmp_path / "data")
    cfg = _cfg(tmp_path)
    for split in ("train", "test", "validation"):
        cache_class_split(CLS, split, cfg)

    rows = OP.evaluate_class(CLS, cfg, [0.05, 0.20])
    assert len(rows) == len(run.S.SCORER_NAMES) * len(OP.REPS) * 2
    for r in rows:
        assert r["tp"] + r["fp"] + r["fn"] + r["tn"] == r["n_test_scans"]
        assert r["n_val_scans"] == 3
        for k in ("precision", "recall", "f1"):
            assert np.isnan(r[k]) or 0.0 <= r[k] <= 1.0
