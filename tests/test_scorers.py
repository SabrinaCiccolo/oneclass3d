"""Invariants the comparison relies on. Run with `pytest -q`."""
import numpy as np
import pytest

from src.evaluate import aggregate_image_normality, image_auroc
from src.scorers import (
    KDE,
    OCSVM,
    Hybrid,
    KNNScorer,
    Mahalanobis,
    ModernHopfield,
    Whitener,
    _ALPHA_GRID,
    _BETA_GRID,
    kcenter_coreset,
    select_alpha,
    select_beta,
    subsample,
)


def _unit_sphere(n, d=33, seed=0):
    X = np.random.default_rng(seed).standard_normal((d, n))
    return X / np.linalg.norm(X, axis=0)


def test_knn_sign_and_monotonicity():
    Xi = _unit_sphere(50)
    knn = KNNScorer(k=1).fit(Xi)
    # a point ON a stored pattern has distance 0 -> normality 0 (the maximum)
    on = knn.score(Xi[:, :1])
    assert on[0] == pytest.approx(0.0, abs=1e-5)
    # a point far from every pattern is more anomalous (more negative) than a near one
    near = Xi[:, 0] + 0.01 * _unit_sphere(1, seed=1)[:, 0]
    far = -Xi[:, 0]                              # antipode: maximally far on the sphere
    s = knn.score(np.stack([near, far], axis=1) / np.linalg.norm(
        np.stack([near, far], axis=1), axis=0))
    assert s[0] > s[1]


@pytest.mark.parametrize("d", [33, 417])   # geometry FPFH and multimodal FPFH+DINOv2
def test_modern_hopfield_equals_kde_at_matched_scale(d):
    """s_KDE(h=1/sqrt(beta)) == beta*s_MH(beta) - beta - log N (the shared-kernel identity).

    Dimension-agnostic: holds for the 33-dim geometry features and the 417-dim
    multimodal ones alike (the scorers only ever see Xi of shape (D, N))."""
    Xi = _unit_sphere(80, d=d, seed=2)
    X = _unit_sphere(40, d=d, seed=3)
    for beta in (1.0, 5.0, 50.0):
        s_mh = ModernHopfield(beta).fit(Xi).score(X).astype(np.float64)
        s_kde = KDE(bandwidth=1.0 / np.sqrt(beta)).fit(Xi).score(X).astype(np.float64)
        log_n = np.log(Xi.shape[1])
        np.testing.assert_allclose(s_kde, beta * s_mh - beta - log_n, rtol=1e-4, atol=1e-4)


def test_auroc_polarity():
    # anomalies given the LOWEST normality -> perfect detection (1.0)
    is_anom = np.array([0, 0, 1, 1])
    normality = np.array([0.9, 0.8, 0.1, 0.0])
    assert image_auroc(normality, is_anom) == pytest.approx(1.0)
    assert image_auroc(-normality, is_anom) == pytest.approx(0.0)
    assert image_auroc(np.zeros(4), is_anom) == pytest.approx(0.5)


def test_aggregate_min_vs_topk():
    # two scans; scan 0 patches {0.5,0.9,0.2,0.7}, scan 1 {0.1,0.4}
    scan_index = np.array([0, 0, 0, 0, 1, 1])
    pn = np.array([0.5, 0.9, 0.2, 0.7, 0.1, 0.4])
    np.testing.assert_allclose(
        aggregate_image_normality(scan_index, pn, 2), [0.2, 0.1])      # default = min
    # topk_frac=0.5 -> mean of the lowest ceil(0.5*n) patches per scan
    # scan 0: ceil(2)=2 lowest {0.2,0.5}->0.35; scan 1: ceil(1)=1 lowest {0.1}->0.1
    np.testing.assert_allclose(
        aggregate_image_normality(scan_index, pn, 2, topk_frac=0.5), [0.35, 0.1])
    # a scan with no kept patches ranks below the finite minimum
    out = aggregate_image_normality(scan_index, pn, 3, topk_frac=0.5)
    assert out[2] < out.min() + 1e-9 and out[2] < 0.1


def test_whitening_outputs_unit_sphere():
    Xi = _unit_sphere(200, seed=4)
    w = Whitener().fit(Xi)
    out = w.transform(_unit_sphere(30, seed=5))
    np.testing.assert_allclose(np.linalg.norm(out, axis=0), 1.0, atol=1e-4)


def test_select_beta_deterministic_and_in_grid():
    # Unsupervised temperature pick is deterministic, returns a grid value, and
    # ModernHopfield("auto") fits/scores like a float-beta scorer.
    Xi = _unit_sphere(400, seed=6)
    b1, b2 = select_beta(Xi), select_beta(Xi)
    assert b1 == b2 and b1 in _BETA_GRID
    mh = ModernHopfield("auto").fit(Xi)
    assert mh.beta_ in _BETA_GRID
    s = mh.score(_unit_sphere(20, seed=7))
    assert s.shape == (20,) and np.isfinite(s).all()


def test_select_beta_block_aware_on_multimodal_shape():
    # geom_dim >= D (geometry-only): block-aware path is a no-op, identical to block-blind.
    Xi33 = _unit_sphere(400, d=33, seed=6)
    assert select_beta(Xi33, geom_dim=33) == select_beta(Xi33)
    assert select_beta(Xi33, geom_dim=None) == select_beta(Xi33, geom_dim=64)

    # multimodal-shaped (geom_dim=8 + 32 "dino" dims): block-aware pseudo-anomalies stay
    # on the unit sphere and the pick is deterministic; it need not equal the block-blind
    # pick (the whole point -- independent per-block corruption is a different, and for a
    # dimension-imbalanced concat, more faithful near-manifold probe).
    Xi40 = _unit_sphere(400, d=40, seed=6)
    b1 = select_beta(Xi40, geom_dim=8)
    b2 = select_beta(Xi40, geom_dim=8)
    assert b1 == b2 and b1 in _BETA_GRID

    mh = ModernHopfield("auto", geom_dim=8).fit(Xi40)
    assert mh.beta_ in _BETA_GRID
    s = mh.score(_unit_sphere(20, d=40, seed=7))
    assert s.shape == (20,) and np.isfinite(s).all()


@pytest.mark.parametrize("d", [33, 417])   # dimension-agnostic like the other scorers
def test_mahalanobis_center_vs_far_and_deterministic(d):
    # two tight, well-separated blobs; a blob centre must score more normal than a
    # point far outside both, and fit/score must be deterministic (seeded k-means)
    rng = np.random.default_rng(0)
    A = 0.05 * rng.standard_normal((d, 200)) + 1.0
    B = 0.05 * rng.standard_normal((d, 200)) - 1.0
    Xi = np.concatenate([A, B], axis=1)
    center, far = A.mean(axis=1), A.mean(axis=1) + 5.0
    q = np.stack([center, far], axis=1)
    s1 = Mahalanobis(kmeans_k=2, seed=0).fit(Xi).score(q)
    assert s1.shape == (2,) and np.isfinite(s1).all()
    assert s1[0] > s1[1]
    np.testing.assert_array_equal(s1, Mahalanobis(kmeans_k=2, seed=0).fit(Xi).score(q))
    # k larger than the bank must not crash (k is clamped to n)
    tiny = Mahalanobis(kmeans_k=50, seed=0).fit(Xi[:, :10]).score(q)
    assert np.isfinite(tiny).all()


def test_select_alpha_deterministic_and_in_grid():
    Xi = _unit_sphere(300, seed=15)
    a1, calib1 = select_alpha(Xi)
    a2, calib2 = select_alpha(Xi)
    assert a1 == a2 and a1 in _ALPHA_GRID and calib1 == calib2

    hy = Hybrid(alpha="auto").fit(Xi)
    assert hy.alpha_ in _ALPHA_GRID
    s = hy.score(_unit_sphere(20, seed=16))
    assert s.shape == (20,) and np.isfinite(s).all()

    # fixed alpha: 1.0 collapses to a monotone transform of pure z-scored kNN, 0.0 to
    # pure z-scored Mahalanobis (both still finite/deterministic; not literally equal to
    # the bare sub-scorer since the hybrid path always z-scores first).
    hy1 = Hybrid(alpha=1.0).fit(Xi)
    assert hy1.alpha_ == 1.0
    assert np.isfinite(hy1.score(_unit_sphere(10, seed=17))).all()


def test_ocsvm_nystroem_full_bank_path_deterministic():
    # bank bigger than max_fit: the nystroem path activates, fits deterministically,
    # and scores finite values in a sane range for a fresh point vs a bank point.
    Xi = _unit_sphere(300, seed=13)
    sc = OCSVM(nu=0.1, gamma="scale", max_fit=100, seed=0, nystroem_components=50).fit(Xi)
    q = _unit_sphere(10, seed=14)
    s1 = sc.score(q)
    s2 = OCSVM(nu=0.1, gamma="scale", max_fit=100, seed=0,
              nystroem_components=50).fit(Xi).score(q)
    np.testing.assert_allclose(s1, s2)
    assert np.isfinite(s1).all()
    # nystroem_components=None (default) keeps the exact-kernel subsample path, unaffected
    sc_exact = OCSVM(nu=0.1, gamma="scale", max_fit=100, seed=0).fit(Xi)
    assert np.isfinite(sc_exact.score(q)).all()


def test_kcenter_coreset_nested_deterministic_and_covering():
    Xi = _unit_sphere(500, seed=12)
    c1 = kcenter_coreset(Xi, 50, seed=0)
    np.testing.assert_array_equal(c1, kcenter_coreset(Xi, 50, seed=0))  # deterministic
    # greedy selection is nested: the 20-cap coreset is the 50-cap's first 20 columns
    np.testing.assert_array_equal(kcenter_coreset(Xi, 20, seed=0), c1[:, :20])
    # coverage: the k-center max-min distance (covering radius) beats a uniform pick's
    def covering_radius(sel):
        d2 = 2.0 - 2.0 * (sel.T @ Xi)
        return np.sqrt(np.maximum(d2.min(axis=0), 0.0)).max()
    assert covering_radius(c1) <= covering_radius(subsample(Xi, 50, seed=0)) + 1e-9
    # n_keep >= N is the identity, like subsample
    np.testing.assert_array_equal(kcenter_coreset(Xi, 600, seed=0), Xi)


def test_blockwise_condition_unit_sphere_and_reduces_to_whitened():
    from src.scorers import condition
    # multimodal-shaped (geom_dim=8 + 32 "dino"): blockwise outputs stay on the unit sphere
    Xi, X = _unit_sphere(120, d=40, seed=8), _unit_sphere(25, d=40, seed=9)
    Xb, Xtb = condition("blockwise", Xi, X, geom_dim=8)
    np.testing.assert_allclose(np.linalg.norm(Xb, axis=0), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(Xtb, axis=0), 1.0, atol=1e-4)
    # geometry-only (D == geom_dim) -> blockwise reduces exactly to 'whitened'
    Xi2, X2 = _unit_sphere(120, d=33, seed=10), _unit_sphere(25, d=33, seed=11)
    bw, _ = condition("blockwise", Xi2, X2, geom_dim=33)
    wh, _ = condition("whitened", Xi2, X2)
    np.testing.assert_allclose(bw, wh, atol=1e-5)


def test_cca_condition_unit_sphere_aligns_correlated_blocks_and_reduces_to_whitened():
    from src.scorers import CCA, condition
    # two blocks built from a SHARED latent signal (correlated) plus independent noise:
    # CCA should recover that correlation (top canonical correlation close to 1), far
    # above the SAME blocks' raw cross-block correlation.
    rng = np.random.default_rng(20)
    n = 400
    latent = rng.standard_normal((5, n))
    G = np.vstack([latent, 0.05 * rng.standard_normal((8, n))])     # (13, n)
    D = np.vstack([latent, 0.05 * rng.standard_normal((20, n))])    # (25, n)
    cca = CCA().fit(G, D)
    assert cca.corr_[0] > 0.9   # top canonical correlation recovers the shared latent

    Xi = np.vstack([G, D])
    Xb, Xtb = condition("cca", Xi, Xi[:, :25], geom_dim=13)
    np.testing.assert_allclose(np.linalg.norm(Xb, axis=0), 1.0, atol=1e-4)
    np.testing.assert_allclose(np.linalg.norm(Xtb, axis=0), 1.0, atol=1e-4)

    # geometry-only (D == geom_dim) -> cca reduces exactly to 'whitened' (nothing to fuse)
    Xi2, X2 = _unit_sphere(120, d=33, seed=10), _unit_sphere(25, d=33, seed=11)
    cc, _ = condition("cca", Xi2, X2, geom_dim=33)
    wh, _ = condition("whitened", Xi2, X2)
    np.testing.assert_allclose(cc, wh, atol=1e-5)
