"""Localization back-projection invariants: 28x28 patch anomaly -> aligned pixel map."""
import numpy as np

from src.preprocess import resample_indices
from localize import _aupro


def test_patch_upsamples_to_its_own_pixel_block():
    grid, pool = 28, 8
    size = grid * pool                       # 224
    a28 = np.zeros((grid, grid))
    a28[3, 5] = 1.0                          # a single anomalous patch
    heat = np.repeat(np.repeat(a28, pool, 0), pool, 1)
    assert heat.shape == (size, size)
    # exactly the 8x8 block of patch (3,5) is hot, nothing else
    assert heat[3 * pool:4 * pool, 5 * pool:6 * pool].min() == 1.0
    assert heat.sum() == pool * pool


def test_resample_indices_monotonic_in_range():
    for H, W in [(800, 800), (400, 600)]:
        rows, cols = resample_indices(H, W, 224)
        assert rows.shape == (224,) and cols.shape == (224,)
        assert rows.min() >= 0 and rows.max() < H
        assert cols.min() >= 0 and cols.max() < W
        assert np.all(np.diff(rows) >= 0) and np.all(np.diff(cols) >= 0)


def test_aupro_perfect_detector_is_one():
    # the defect region's scores are all strictly above every background score (with a
    # little spread on each side, like a real continuous heat map) -> the region is
    # fully covered before any false positive appears -> AUPRO ~= 1 (not exactly 1: with
    # a finite background sample, the largest achievable FPR <= 0.3 undershoots 0.3 by
    # less than 1/n_neg, a discretization floor that shrinks as n_neg grows).
    rng = np.random.default_rng(0)
    region_scores = [np.sort(1.0 + 0.01 * rng.standard_normal(10))]
    neg_scores = 0.01 * rng.standard_normal(2000)
    aupro = _aupro(region_scores, [10], neg_scores)
    assert aupro > 0.999


def test_aupro_uninformative_detector_is_low():
    # region and background scores are drawn from the SAME distribution -> at any FPR the
    # covered fraction of the region tracks the FPR itself (chance-level), well below 1.
    rng = np.random.default_rng(0)
    shared = rng.uniform(0, 1, 1000)
    aupro = _aupro([np.sort(shared[:100])], [100], shared[100:])
    assert aupro < 0.6
