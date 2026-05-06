"""DBSCAN eps resolution modes and pipeline variants (src/preprocess.py)."""
import numpy as np
import pytest

from src.preprocess import _dbscan_eps, _kdistance_eps, bbox_diagonal, pipeline_cfg


def _grid_points(n_side=10, spacing=0.01, seed=0, jitter=0.0):
    """An n_side^3 lattice (deterministic local spacing) plus optional per-point jitter."""
    rng = np.random.default_rng(seed)
    ax = np.arange(n_side) * spacing
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    if jitter:
        pts = pts + jitter * rng.standard_normal(pts.shape)
    return pts


def test_dbscan_eps_abs_mode_returns_fixed_value():
    cfg = {"cleanup": {"dbscan_eps_mode": "abs", "dbscan_eps": 0.006, "dbscan_min_points": 30}}
    assert _dbscan_eps(_grid_points(), cfg) == pytest.approx(0.006)


def test_dbscan_eps_rel_mode_scales_with_bbox_diagonal():
    cfg = {"cleanup": {"dbscan_eps_mode": "rel", "dbscan_eps_rel": 0.1,
                       "dbscan_eps": 0.006, "dbscan_min_points": 30}}
    pts = _grid_points(spacing=0.02)
    eps = _dbscan_eps(pts, cfg)
    assert eps == pytest.approx(0.1 * bbox_diagonal(pts), rel=1e-6)


def test_dbscan_eps_rel_mode_falls_back_on_degenerate_result():
    cfg = {"cleanup": {"dbscan_eps_mode": "rel", "dbscan_eps_rel": 0.0,
                       "dbscan_eps": 0.006, "dbscan_min_points": 30}}
    assert _dbscan_eps(_grid_points(), cfg) == pytest.approx(0.006)


def test_kdistance_eps_grows_with_noise():
    # the adaptive heuristic must track increased LOCAL point spacing (what noise
    # corruption actually does), unlike a fixed or bbox-diagonal-relative eps, whose
    # bbox barely moves under per-point jitter. Uniform random points (not a lattice --
    # a lattice's k-th NN distance is degenerate/shell-quantized and doesn't respond to
    # jitter the way an irregularly-sampled real scan does).
    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 0.1, size=(2000, 3))
    clean = base.copy()
    noisy = base + 0.003 * rng.standard_normal(base.shape)   # jitter << the 0.1 box extent
    eps_clean = _kdistance_eps(clean, k=10, percentile=90.0)
    eps_noisy = _kdistance_eps(noisy, k=10, percentile=90.0)
    assert eps_noisy > eps_clean > 0


def test_dbscan_eps_adaptive_mode_matches_kdistance_eps():
    cfg = {"cleanup": {"dbscan_eps_mode": "adaptive", "dbscan_eps_percentile": 90.0,
                       "dbscan_eps": 0.006, "dbscan_min_points": 10}}
    pts = _grid_points(spacing=0.01, jitter=0.005, seed=2)
    assert _dbscan_eps(pts, cfg) == pytest.approx(_kdistance_eps(pts, 10, 90.0))


def test_dbscan_eps_unknown_mode_raises():
    cfg = {"cleanup": {"dbscan_eps_mode": "bogus", "dbscan_eps": 0.006, "dbscan_min_points": 30}}
    with pytest.raises(ValueError):
        _dbscan_eps(_grid_points(), cfg)


def test_pipeline_cfg_toggles_one_stage_per_tag():
    cfg = {"ransac": {"enabled": True}, "cleanup": {"enabled": True, "dbscan_eps_mode": "abs"}}
    assert pipeline_cfg(cfg, "") == cfg
    assert not pipeline_cfg(cfg, "nr")["ransac"]["enabled"]
    assert pipeline_cfg(cfg, "nr")["cleanup"]["enabled"]
    assert not pipeline_cfg(cfg, "nc")["cleanup"]["enabled"]
    both = pipeline_cfg(cfg, "nrnc")
    assert not both["ransac"]["enabled"] and not both["cleanup"]["enabled"]
    assert pipeline_cfg(cfg, "adaptive")["cleanup"]["dbscan_eps_mode"] == "adaptive"
    assert cfg["cleanup"]["dbscan_eps_mode"] == "abs"          # input left untouched
    with pytest.raises(ValueError):
        pipeline_cfg(cfg, "xx")
