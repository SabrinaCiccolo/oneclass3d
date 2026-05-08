"""Anomaly scorers and the whitening / fusion transforms.

The six scorers of the main comparison (``SCORER_NAMES``) plus Mahalanobis and a
kNN + Mahalanobis hybrid, used by the robustness studies.

Every scorer reads the SAME shared memory matrix Xi (D, N) - all patch descriptors of
the normal training scans, columns on the unit sphere - and assigns each test patch a
NORMALITY score (higher = more normal). Interface:

    scorer.fit(Xi).score(X)   ->  (M,) normality, X is (D, M)

Everything here is dimension-agnostic: D is 33 for geometry-only FPFH and 417 for the
multimodal FPFH+DINOv2 features, with no scorer change either way.

Image-level normality is then the MIN over a scan's patches (see evaluate.py), and
AUROC negates it (higher = more anomalous). Only the scoring function differs between
scorers, so any AUROC difference is the scorer's, not the features'.

Everything is kept in one file and runs in float32 for speed (k-NN is computed as a
max-dot-product matmul on the sphere rather than a tree query).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.svm import OneClassSVM

# Cap on (N x chunk) similarity/distance entries materialized at once, so the
# matmul scorers stay bounded in memory on full banks (N up to ~71k). Larger chunks
# mean fewer, bigger (faster) BLAS calls; ~40M float32 entries is ~160 MB/buffer.
_MAX_ENTRIES = 40_000_000
_L2_ATOL = 1e-4


# --------------------------------------------------------------------------- #
# Whitening (anomaly-free, fit on the normal bank only)                       #
# --------------------------------------------------------------------------- #
def l2_normalize_cols(X: np.ndarray) -> np.ndarray:
    """L2-normalize columns of (d, M) onto the unit sphere; assert ||x|| ~= 1."""
    n = np.linalg.norm(X, axis=0)
    n = np.where(n < 1e-12, 1.0, n)
    out = X / n
    assert np.allclose(np.linalg.norm(out, axis=0), 1.0, atol=_L2_ATOL)
    return out


def _shrunk_whitening(bank: np.ndarray, shrinkage: float, eig_floor: float
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(mu, W) s.t. W @ (x - mu) is symmetrically whitened, from the SHRUNK bank
    covariance C' = (1-shrinkage)*C + shrinkage*(trC/dim)*I. Shared by Whitener (which
    also re-L2's the output onto the sphere) and CCA (which needs the raw decorrelated
    features, pre-L2, to compute a meaningful cross-covariance between two blocks).
    """
    Xi = np.asarray(bank, dtype=np.float64)
    dim, n = Xi.shape
    mu = Xi.mean(axis=1)
    Xc = Xi - mu[:, None]
    C = (Xc @ Xc.T) / n
    C = (1.0 - shrinkage) * C + shrinkage * (np.trace(C) / dim) * np.eye(dim)
    lam, U = np.linalg.eigh(C)
    lam = np.maximum(lam, eig_floor * lam.max())
    W = (U / np.sqrt(lam)) @ U.T
    return mu, W


class Whitener:
    """Symmetric whitening fit on the normal bank, applied to bank + test, re-L2'd.

    Raw FPFH patch descriptors are hyper-concentrated on the sphere (test->bank
    nearest cosine ~0.998), which collapses every soft scorer toward the high-bias
    floor. Whitening de-correlates and rescales the directions so soft scorers
    (Modern Hopfield, KDE) can discriminate. W = U (Lambda+floor)^-1/2 U^T from the
    SHRUNK bank covariance C' = (1-d)C + d (trC/dim) I; mu and W are estimated from
    normals only (no anomalies, no test labels).
    """

    def __init__(self, shrinkage: float = 0.10, eig_floor: float = 1e-6):
        self.shrinkage = float(shrinkage)
        self.eig_floor = float(eig_floor)

    def fit(self, bank: np.ndarray) -> "Whitener":
        self.mu_, self.W_ = _shrunk_whitening(bank, self.shrinkage, self.eig_floor)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.mu_[:, None]
        return l2_normalize_cols(self.W_ @ Xc)


class CCA:
    """Closed-form (SVD) Canonical Correlation Analysis fusion of two feature blocks,
    fit on the normal bank only -- a training-free alternative to raw concatenation /
    block-wise whitening for the FPFH + DINOv2 multimodal descriptor.

    Each block is whitened independently (``_shrunk_whitening``, no final L2 -- CCA
    needs the raw decorrelated features to read a meaningful cross-covariance), then the
    SVD of their cross-covariance gives the canonical directions (U, V): the k = rank
    directions in each whitened block that are maximally correlated with each other
    across the paired samples. The fused descriptor concatenates both blocks' canonical
    projections and re-L2's -- same "each side contributes, then unit-normalize the
    whole" shape as ``condition``'s "blockwise" path, but the projection itself is
    fit to align the two modalities instead of leaving one of them raw.
    """

    def __init__(self, n_components: int | None = None, shrinkage: float = 0.10,
                eig_floor: float = 1e-6):
        self.n_components = n_components
        self.shrinkage = float(shrinkage)
        self.eig_floor = float(eig_floor)

    def fit(self, G: np.ndarray, D: np.ndarray) -> "CCA":
        self.mu_g, self.Wg_ = _shrunk_whitening(G, self.shrinkage, self.eig_floor)
        self.mu_d, self.Wd_ = _shrunk_whitening(D, self.shrinkage, self.eig_floor)
        Ghat = self.Wg_ @ (np.asarray(G, dtype=np.float64) - self.mu_g[:, None])
        Dhat = self.Wd_ @ (np.asarray(D, dtype=np.float64) - self.mu_d[:, None])
        Cgd = (Ghat @ Dhat.T) / G.shape[1]
        U, S, Vt = np.linalg.svd(Cgd, full_matrices=False)
        k = min(self.n_components or U.shape[1], U.shape[1], Vt.shape[0])
        self.Ag_, self.Bd_, self.k_, self.corr_ = U[:, :k], Vt[:k, :].T, k, S[:k]
        return self

    def transform(self, G: np.ndarray, D: np.ndarray) -> np.ndarray:
        Ghat = self.Wg_ @ (np.asarray(G, dtype=np.float64) - self.mu_g[:, None])
        Dhat = self.Wd_ @ (np.asarray(D, dtype=np.float64) - self.mu_d[:, None])
        return l2_normalize_cols(np.vstack([self.Ag_.T @ Ghat, self.Bd_.T @ Dhat]))


def condition(rep: str, Xi: np.ndarray, X: np.ndarray, geom_dim: int = 33
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return (Xi, X) for representation `rep` in {'raw', 'whitened', 'blockwise', 'cca'}.

    'raw' uses the cached (already unit-norm) descriptors unchanged; 'whitened' fits a
    Whitener on the whole bank and applies it to bank + test; 'blockwise' is the
    modality-aware variant for the multimodal descriptor - it whitens ONLY the geometry
    block (FPFH, dims 0:geom_dim), leaves the DINOv2 block raw, and re-L2's each modality
    then their concatenation (same modality balance as raw mm, geometry de-correlated).
    'cca' fits a closed-form CCA (see the CCA class) between the geometry and DINOv2
    blocks and concatenates their canonical projections instead -- a fusion that aligns
    the two modalities rather than leaving one raw. For geometry-only features
    (D == geom_dim) both 'blockwise' and 'cca' reduce to 'whitened' (nothing to fuse).
    """
    if rep == "raw":
        return Xi, X
    if rep == "whitened":
        w = Whitener().fit(Xi)
        return w.transform(Xi), w.transform(X)
    if rep == "blockwise":
        w = Whitener().fit(Xi[:geom_dim])

        def cond(M):
            g = w.transform(M[:geom_dim])                      # whitened geometry, unit
            if M.shape[0] <= geom_dim:
                return g
            d = l2_normalize_cols(M[geom_dim:])                # DINOv2 block, unit
            return l2_normalize_cols(np.vstack([g, d]))

        return cond(Xi), cond(X)
    if rep == "cca":
        if Xi.shape[0] <= geom_dim:
            return condition("whitened", Xi, X)
        cca = CCA().fit(Xi[:geom_dim], Xi[geom_dim:])
        return (cca.transform(Xi[:geom_dim], Xi[geom_dim:]),
                cca.transform(X[:geom_dim], X[geom_dim:]))
    raise ValueError(f"unknown representation {rep!r}")


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def scott_bandwidth(n: int, d: int) -> float:
    """Scott's-rule KDE bandwidth factor n**(-1/(d+4))."""
    return float(n) ** (-1.0 / (d + 4))


def subsample(Xi: np.ndarray, n_keep: int, seed: int = 0) -> np.ndarray:
    """Deterministic random subset of n_keep columns of Xi (33, N).

    Used only to bound the OC-SVM training set (its cost is super-linear in the
    number of stored patterns); every other scorer always uses the full Xi.
    """
    n = Xi.shape[1]
    if n_keep >= n:
        return Xi
    idx = np.random.default_rng(seed).choice(n, size=n_keep, replace=False)
    return np.ascontiguousarray(Xi[:, np.sort(idx)])


def kcenter_coreset(Xi: np.ndarray, n_keep: int, seed: int = 0) -> np.ndarray:
    """Greedy k-center (farthest-point) coreset of n_keep columns of Xi (D, N).

    PatchCore-style alternative to `subsample`: start from a seeded random column,
    then repeatedly add the column farthest from the selected set (max-min Euclidean
    distance). Coverage-maximizing where uniform subsampling is density-proportional;
    the returned selection is nested (its first m columns are the m-cap coreset).
    Deterministic given (Xi, seed). O(n_keep * N) distances, float32.
    """
    n = Xi.shape[1]
    if n_keep >= n:
        return Xi
    X = np.asarray(Xi, dtype=np.float32)
    sq = (X ** 2).sum(axis=0)
    chosen = np.empty(n_keep, dtype=np.int64)
    chosen[0] = np.random.default_rng(seed).integers(n)
    d2 = sq + sq[chosen[0]] - 2.0 * (X.T @ X[:, chosen[0]])   # ||x - c0||^2, (N,)
    for i in range(1, n_keep):
        chosen[i] = int(np.argmax(d2))
        cand = sq + sq[chosen[i]] - 2.0 * (X.T @ X[:, chosen[i]])
        np.minimum(d2, cand, out=d2)
    return np.ascontiguousarray(Xi[:, chosen])


def mh_logsumexp(sims: np.ndarray, betas) -> np.ndarray:
    """(N, c) similarities -> (len(betas), c) Modern-Hopfield normality, one row per beta.

    Each row is (1/beta) logsumexp(beta * sims) with the standard max-shift for stability.
    Shared by ModernHopfield's beta selection and plot.py's beta sweep (sims computed once).
    """
    smax = sims.max(axis=0)
    shifted = sims - smax
    return np.stack([smax + np.log(np.exp(float(b) * shifted).sum(axis=0)) / float(b) for b in betas])


# --------------------------------------------------------------------------- #
# Scorers                                                                      #
# --------------------------------------------------------------------------- #
class Scorer(ABC):
    @abstractmethod
    def fit(self, Xi: np.ndarray) -> "Scorer": ...
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray: ...


class KNNScorer(Scorer):
    """s(x) = -(distance to the k-th nearest stored pattern). The strong baseline.

    On the unit sphere ||x - xi||^2 = 2 - 2 (xi . x), so the k-th nearest pattern is
    the k-th largest dot product. We get it from a single chunked matmul (BLAS,
    parallel) rather than a tree query, which is much faster than sklearn's
    NearestNeighbors at this dimension (33) and bank size (up to ~71k).
    """

    def __init__(self, k: int = 1):
        self.k = int(k)

    def fit(self, Xi: np.ndarray) -> "KNNScorer":
        self._Xi = np.asarray(Xi, dtype=np.float32)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = self._Xi.shape[1]
        chunk = max(1, _MAX_ENTRIES // max(n, 1))
        out = np.empty(X.shape[1], dtype=np.float32)
        for i in range(0, X.shape[1], chunk):
            sims = self._Xi.T @ X[:, i:i + chunk]                 # (N, c) cosine sims
            if self.k == 1:
                kth = sims.max(axis=0)                            # nearest = largest dot
            else:
                kth = np.partition(sims, n - self.k, axis=0)[n - self.k]  # k-th largest
            d2 = np.maximum(2.0 - 2.0 * kth, 0.0)
            out[i:i + chunk] = -np.sqrt(d2)
        return out


class ModernHopfield(Scorer):
    """s(x) = (1/beta) logsumexp(beta * Xi^T x): temperature-controlled soft 1-NN.

    beta -> inf converges to max_i xi_i . x (hard 1-NN on the sphere); beta -> 0 ranks
    like the mean similarity. Stable max-shift log-sum-exp, chunked over test patches.
    beta is a float, or "auto" to pick it per class from normals only (see select_beta).
    ``geom_dim`` is only used by "auto" selection, to keep its pseudo-anomaly generation
    block-aware on concatenated multimodal descriptors (no-op for geometry-only Xi).
    """

    def __init__(self, beta, geom_dim=None):
        self.beta = beta
        self.geom_dim = geom_dim

    def fit(self, Xi: np.ndarray) -> "ModernHopfield":
        self._Xi = np.asarray(Xi, dtype=np.float32)
        self.beta_ = select_beta(self._Xi, geom_dim=self.geom_dim) if self.beta == "auto" \
            else float(self.beta)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = self._Xi.shape[1]
        chunk = max(1, _MAX_ENTRIES // max(n, 1))
        out = np.empty(X.shape[1], dtype=np.float32)
        b = np.float32(self.beta_)
        for i in range(0, X.shape[1], chunk):
            sims = self._Xi.T @ X[:, i:i + chunk]                 # (N, c) float32
            smax = sims.max(axis=0)
            out[i:i + chunk] = smax + np.log(np.exp(b * (sims - smax)).sum(axis=0)) / b
        return out


# --------------------------------------------------------------------------- #
# Tempered memory bank: pick the Modern-Hopfield beta from NORMALS ONLY        #
# --------------------------------------------------------------------------- #
# beta is the bias<->variance dial (small = mean-pool / high bias, large = hard 1-NN). We
# pick it per class without any test labels: hold out a slice of the normal bank, build
# NEAR-MANIFOLD pseudo-anomalies from it (mixup + bin-swap of real descriptors -- subtle
# local deviations like real defects, not trivially separable off-manifold points), and take
# the beta that best separates held normals from those pseudo-anomalies. The 1-SE rule keeps
# the SMALLEST beta statistically tied with the best, landing on the interior regime rather
# than the degenerate beta->inf tail (a near-manifold contrast proxy, no test labels).
_BETA_GRID = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
_CORRUPTIONS = (("mixup", 0.45), ("mixup", 0.30), ("binswap", 0.30), ("binswap", 0.15))


def _mixup_block(V, s, rng):
    """normalize(lam*x_i + (1-lam)*x_j), lam~U(s,1-s): a point on the chord between two normals."""
    M = V.shape[1]
    lam = rng.uniform(s, 1.0 - s, M)[None, :]
    return lam * V + (1.0 - lam) * V[:, rng.integers(0, M, M)]


def _binswap_block(V, frac, rng):
    """Replace a random `frac` of each descriptor's bins with another normal's: a chimera."""
    d, M = V.shape
    return np.where(rng.random((d, M)) < frac, V[:, rng.integers(0, M, M)], V)


def _mixup(V, s, rng, geom_dim=None):
    """Mixup pseudo-anomaly, block-aware for concatenated multimodal descriptors.

    Single-block (geom_dim is None or V has <= geom_dim rows, e.g. 33-dim geometry-only):
    one lambda blends the whole vector. Multimodal (V taller than geom_dim,
    e.g. the 417-dim FPFH+DINOv2 concat): mixup is applied to each modality block with its
    OWN lambda/partner draw before renormalizing, so a block-blind single lambda cannot be
    diluted by the larger, already-well-spread DINOv2 block -- both blocks get a comparably
    subtle, near-manifold perturbation regardless of their different concentration on the
    sphere (see select_beta / config `geom_dim`).
    """
    if geom_dim is None or V.shape[0] <= geom_dim:
        X = _mixup_block(V, s, rng)
    else:
        X = np.vstack([_mixup_block(V[:geom_dim], s, rng), _mixup_block(V[geom_dim:], s, rng)])
    return X / np.linalg.norm(X, axis=0)


def _binswap(V, frac, rng, geom_dim=None):
    """Bin-swap pseudo-anomaly, block-aware for concatenated multimodal descriptors (see _mixup)."""
    if geom_dim is None or V.shape[0] <= geom_dim:
        X = _binswap_block(V, frac, rng)
    else:
        X = np.vstack([_binswap_block(V[:geom_dim], frac, rng),
                       _binswap_block(V[geom_dim:], frac, rng)])
    return X / np.linalg.norm(X, axis=0)


def _mh_grid(bank, Q, betas):
    """(len(betas), Mq) Modern-Hopfield normality of columns Q vs bank; sims computed once."""
    return mh_logsumexp(bank.T @ Q, betas)


def _auroc_se(auc, n_a, n_n):
    """Hanley-McNeil analytic standard error of an AUROC (the 1-SE tolerance)."""
    a = min(max(auc, 1e-6), 1.0 - 1e-6)
    q1, q2 = a / (2.0 - a), 2.0 * a * a / (1.0 + a)
    var = (a * (1 - a) + (n_a - 1) * (q1 - a * a) + (n_n - 1) * (q2 - a * a)) / (n_a * n_n)
    return float(np.sqrt(max(var, 0.0)))


def select_beta(Xi, betas=_BETA_GRID, holdout_frac=0.2, held_cap=2000, bank_cap=6000, seed=0,
                geom_dim=None):
    """Pick the Modern-Hopfield beta from the normal bank only (near-manifold contrast).

    Returns the smallest beta whose held-normal-vs-pseudo-anomaly separation is within one
    AUROC standard error of the best (parsimony / interior-regime pick). Deterministic.
    ``geom_dim`` makes the pseudo-anomaly corruptions block-aware for concatenated
    multimodal descriptors (Xi taller than geom_dim); ignored for geometry-only Xi.
    """
    rng = np.random.default_rng(seed)
    d, N = Xi.shape
    perm = rng.permutation(N)
    n_held = min(held_cap, max(1, int(holdout_frac * N)))
    held = Xi[:, perm[:n_held]].astype(np.float32)
    bank = Xi[:, perm[n_held:]]
    if bank.shape[1] > bank_cap:
        bank = bank[:, rng.choice(bank.shape[1], bank_cap, replace=False)]
    bank = bank.astype(np.float32)
    if bank.shape[1] == 0:                      # degenerate tiny bank -> hard 1-NN
        return float(betas[-1])

    betas = [float(b) for b in betas]
    mh_held = _mh_grid(bank, held, betas)       # (B, n_held)
    labels = np.concatenate([np.zeros(n_held), np.ones(n_held)])
    sep = np.zeros(len(betas))
    for i, (kind, strength) in enumerate(_CORRUPTIONS):
        r = np.random.default_rng((seed, i + 1))
        anom = _mixup(held, strength, r, geom_dim) if kind == "mixup" else \
            _binswap(held, strength, r, geom_dim)
        mh_anom = _mh_grid(bank, anom.astype(np.float32), betas)
        for j in range(len(betas)):
            sep[j] += roc_auc_score(labels, -np.concatenate([mh_held[j], mh_anom[j]]))
    sep /= len(_CORRUPTIONS)

    eps = _auroc_se(float(sep.max()), n_held, n_held)
    mx = float(sep.max())
    for b, s in zip(betas, sep):
        if s >= mx - eps:
            return b
    return betas[-1]


class KDE(Scorer):
    """Gaussian KDE log-density s(x) = logsumexp(-||x-xi||^2 / 2h^2) - log N.

    Independent squared-distance path (not dot products) so the MH==KDE check at
    matched scale h^2 = 1/beta is a real test. bandwidth='scott' resolves to Scott's
    rule at the bank count; a float is used as-is.
    """

    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def fit(self, Xi: np.ndarray) -> "KDE":
        self._Xi = np.asarray(Xi, dtype=np.float32)
        self._sq = (self._Xi ** 2).sum(axis=0)                     # (N,) float32
        d, n = self._Xi.shape
        self._log_n = np.float32(np.log(n))
        self._h = scott_bandwidth(n, d) if self.bandwidth == "scott" else float(self.bandwidth)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        x_sq = (X ** 2).sum(axis=0)
        n = self._Xi.shape[1]
        h2 = np.float32(self._h ** 2)
        chunk = max(1, _MAX_ENTRIES // max(n, 1))
        out = np.empty(X.shape[1], dtype=np.float32)
        for i in range(0, X.shape[1], chunk):
            cross = self._Xi.T @ X[:, i:i + chunk]                 # (N, c) float32
            d2 = self._sq[:, None] + x_sq[None, i:i + chunk] - 2.0 * cross
            np.maximum(d2, 0.0, out=d2)
            d2min = d2.min(axis=0)
            out[i:i + chunk] = (-d2min / (2.0 * h2)
                                + np.log(np.exp(-(d2 - d2min) / (2.0 * h2)).sum(axis=0))
                                - self._log_n)
        return out


class Mahalanobis(Scorer):
    """s(x) = -min over clusters of the Mahalanobis distance (PaDiM-flavoured).

    k-means (k = kmeans_k, seeded, fit on the bank only) partitions Xi; each cluster
    gets a Ledoit-Wolf shrunk covariance. Normality is the negative distance to the
    nearest cluster's Gaussian. Dimension-agnostic like the others (the Ledoit-Wolf
    shrinkage keeps every cluster covariance well-conditioned even when a cluster has
    fewer members than dimensions, as happens at D=417).
    """

    def __init__(self, kmeans_k: int = 50, seed: int = 0):
        self.kmeans_k = int(kmeans_k)
        self.seed = int(seed)

    def fit(self, Xi: np.ndarray) -> "Mahalanobis":
        from sklearn.cluster import KMeans
        from sklearn.covariance import LedoitWolf

        P = np.asarray(Xi, dtype=np.float64).T                 # (N, d) sklearn layout
        n, d = P.shape
        k = min(self.kmeans_k, n)
        labels = KMeans(n_clusters=k, n_init=1, random_state=self.seed).fit(P).labels_
        self._mus, self._Linvs = [], []
        for c in range(k):
            mem = P[labels == c]
            if mem.shape[0] == 0:
                continue
            mu = mem.mean(axis=0)
            C = LedoitWolf().fit(mem).covariance_ if mem.shape[0] >= 2 else np.eye(d)
            # tiny ridge so the Cholesky factor stays finite on degenerate clusters
            C = C + 1e-9 * max(np.trace(C) / d, 1e-12) * np.eye(d)
            Linv = np.linalg.inv(np.linalg.cholesky(C))        # d(x)^2 = ||Linv (x-mu)||^2
            self._mus.append(mu.astype(np.float32))
            self._Linvs.append(Linv.astype(np.float32))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        best = np.full(X.shape[1], np.inf, dtype=np.float32)
        for mu, Linv in zip(self._mus, self._Linvs):
            Y = Linv @ (X - mu[:, None])
            np.minimum(best, (Y * Y).sum(axis=0), out=best)
        return -np.sqrt(best)


def _ocsvm_gamma(Xi: np.ndarray, gamma) -> float:
    """Resolve gamma='scale' the way sklearn's SVM does: 1 / (n_features * X.var())."""
    if gamma != "scale":
        return float(gamma)
    d = Xi.shape[0]
    var = float(np.asarray(Xi, dtype=np.float64).var())
    return 1.0 / (d * var) if var > 0 else 1.0


class OCSVM(Scorer):
    """One-Class SVM (RBF); s(x) = signed distance to the boundary (inside = normal).

    Exact fit is O(n^2)-O(n^3), so by default it's randomly subsampled to max_fit
    patches when |Xi| > max_fit (every other scorer uses the full Xi). Setting
    ``nystroem_components`` switches the large-bank path instead to a Nystroem RBF
    feature map (fit on a max_fit-sized landmark sample) + a linear one-class SGD model
    -- O(n * n_components), so the LINEAR decision boundary is fit on the FULL bank
    rather than discarding it to a subsample, at comparable cost to the exact-subsample
    path.
    """

    def __init__(self, nu: float, gamma, max_fit: int = 10000, seed: int = 0,
                nystroem_components: int | None = None):
        self.nu = float(nu)
        self.gamma = gamma
        self.max_fit = int(max_fit)
        self.seed = int(seed)
        self.nystroem_components = nystroem_components

    def fit(self, Xi: np.ndarray) -> "OCSVM":
        Xi = np.asarray(Xi, dtype=np.float64)
        gamma = _ocsvm_gamma(Xi, self.gamma)
        if self.nystroem_components and Xi.shape[1] > self.max_fit:
            from sklearn.kernel_approximation import Nystroem
            from sklearn.linear_model import SGDOneClassSVM

            self._nystroem = True
            landmarks = subsample(Xi, self.max_fit, self.seed)
            m = min(int(self.nystroem_components), landmarks.shape[1])
            self._feat_map = Nystroem(kernel="rbf", gamma=gamma, n_components=m,
                                      random_state=self.seed).fit(landmarks.T)
            Z = self._feat_map.transform(Xi.T)                 # full bank, approximated kernel
            self._svm = SGDOneClassSVM(nu=self.nu, random_state=self.seed).fit(Z)
        else:
            self._nystroem = False
            X_fit = Xi if Xi.shape[1] <= self.max_fit else subsample(Xi, self.max_fit, self.seed)
            self._svm = OneClassSVM(kernel="rbf", nu=self.nu, gamma=gamma)
            self._svm.fit(np.ascontiguousarray(X_fit.T))
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self._nystroem:
            return self._svm.decision_function(self._feat_map.transform(X.T)).astype(np.float32)
        return self._svm.decision_function(np.ascontiguousarray(X.T)).astype(np.float32)


# --------------------------------------------------------------------------- #
# Hybrid: retrieval (kNN) + parametric (Mahalanobis) fusion                    #
# --------------------------------------------------------------------------- #
def _calibrate_hybrid(Xi, knn_k, maha_kwargs, holdout_frac=0.2, held_cap=2000,
                      bank_cap=6000, seed=0):
    """Fit kNN + Mahalanobis on a held-out bank/normals split; return (knn, maha, held,
    calib) where calib is the z-score mean/std of each sub-scorer's score on ``held``
    (needed to put the two very different native scales on comparable footing before
    combining them). Same held/bank split shape as select_beta, so the sub-scorers never
    see the points they're being calibrated against."""
    rng = np.random.default_rng(seed)
    N = Xi.shape[1]
    perm = rng.permutation(N)
    n_held = min(held_cap, max(1, int(holdout_frac * N)))
    held = Xi[:, perm[:n_held]].astype(np.float32)
    bank = Xi[:, perm[n_held:]]
    if bank.shape[1] > bank_cap:
        bank = bank[:, rng.choice(bank.shape[1], bank_cap, replace=False)]
    bank = bank.astype(np.float32)

    knn = KNNScorer(k=knn_k).fit(bank)
    maha = Mahalanobis(**maha_kwargs).fit(bank)
    s_knn = knn.score(held)
    s_maha = maha.score(held)
    calib = {"mu_knn": float(s_knn.mean()), "sigma_knn": float(s_knn.std() + 1e-9),
            "mu_maha": float(s_maha.mean()), "sigma_maha": float(s_maha.std() + 1e-9)}
    return knn, maha, held, calib


_ALPHA_GRID = tuple(round(a, 2) for a in np.linspace(0.0, 1.0, 11))   # 0.0, 0.1, ..., 1.0


def select_alpha(Xi, knn_k=1, maha_kwargs=None, alphas=_ALPHA_GRID, seed=0, geom_dim=None,
                 **calib_kwargs):
    """Pick the kNN/Mahalanobis hybrid mixing weight alpha from normals only: the SAME
    near-manifold pseudo-anomaly probe as select_beta (held-out normals vs mixup/binswap
    chimeras), scored by the z-scored combination alpha*z_kNN + (1-alpha)*z_Mahalanobis,
    picking the alpha with the best held-normal-vs-pseudo-anomaly separation. No test
    labels are used -- NOT a sweep against test-set AUROC. Returns (alpha, calib).
    """
    maha_kwargs = maha_kwargs or {}
    knn, maha, held, calib = _calibrate_hybrid(Xi, knn_k, maha_kwargs, seed=seed, **calib_kwargs)
    z_knn_held = (knn.score(held) - calib["mu_knn"]) / calib["sigma_knn"]
    z_maha_held = (maha.score(held) - calib["mu_maha"]) / calib["sigma_maha"]

    n_held = held.shape[1]
    labels = np.concatenate([np.zeros(n_held), np.ones(n_held)])
    alphas = [float(a) for a in alphas]
    sep = np.zeros(len(alphas))
    for i, (kind, strength) in enumerate(_CORRUPTIONS):
        r = np.random.default_rng((seed, i + 1))
        anom = _mixup(held, strength, r, geom_dim) if kind == "mixup" else \
            _binswap(held, strength, r, geom_dim)
        z_knn_a = (knn.score(anom) - calib["mu_knn"]) / calib["sigma_knn"]
        z_maha_a = (maha.score(anom) - calib["mu_maha"]) / calib["sigma_maha"]
        for j, a in enumerate(alphas):
            combined_held = a * z_knn_held + (1.0 - a) * z_maha_held
            combined_anom = a * z_knn_a + (1.0 - a) * z_maha_a
            sep[j] += roc_auc_score(labels, -np.concatenate([combined_held, combined_anom]))
    sep /= len(_CORRUPTIONS)
    return alphas[int(np.argmax(sep))], calib


class Hybrid(Scorer):
    """s_hybrid = alpha * z(s_kNN) + (1-alpha) * z(s_Mahalanobis): retrieval + parametric
    fusion (Mahalanobis is the better clean scorer on raw banks, kNN the more robust one
    under bank contamination -- RESULTS.md). Each sub-score is z-scored (mean/std from a held-out normal slice,
    see _calibrate_hybrid) before combining -- kNN's negative-distance and Mahalanobis's
    negative-Mahalanobis-distance live on very different native scales, so a raw
    alpha-weighted sum would just be dominated by whichever has the larger native spread.

    ``alpha`` is a float in [0, 1], or "auto" to pick it from normals only via
    select_alpha (near-manifold pseudo-anomaly separation, NOT a test-set AUROC sweep).
    """

    def __init__(self, alpha="auto", knn_k: int = 1, kmeans_k: int = 50,
                kmeans_seed: int = 0, geom_dim=None):
        self.alpha = alpha
        self.knn_k = knn_k
        self.kmeans_k = kmeans_k
        self.kmeans_seed = kmeans_seed
        self.geom_dim = geom_dim

    def fit(self, Xi: np.ndarray) -> "Hybrid":
        Xi = np.asarray(Xi, dtype=np.float32)
        maha_kwargs = {"kmeans_k": self.kmeans_k, "seed": self.kmeans_seed}
        if self.alpha == "auto":
            self.alpha_, self._calib = select_alpha(Xi, knn_k=self.knn_k,
                                                     maha_kwargs=maha_kwargs,
                                                     geom_dim=self.geom_dim)
        else:
            self.alpha_ = float(self.alpha)
            _, _, _, self._calib = _calibrate_hybrid(Xi, self.knn_k, maha_kwargs)
        self._knn = KNNScorer(k=self.knn_k).fit(Xi)
        self._maha = Mahalanobis(**maha_kwargs).fit(Xi)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        z_knn = (self._knn.score(X) - self._calib["mu_knn"]) / self._calib["sigma_knn"]
        z_maha = (self._maha.score(X) - self._calib["mu_maha"]) / self._calib["sigma_maha"]
        return (self.alpha_ * z_knn + (1.0 - self.alpha_) * z_maha).astype(np.float32)


class Centroid(Scorer):
    """s(x) = x . mu_hat, mu_hat = normalize(mean of Xi). Maximum-bias baseline."""

    def fit(self, Xi: np.ndarray) -> "Centroid":
        mean = np.asarray(Xi, dtype=np.float64).mean(axis=1)
        norm = np.linalg.norm(mean)
        if norm == 0.0:
            raise ValueError("centroid is the zero vector (degenerate bank)")
        self._mu = mean / norm
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return (self._mu @ np.asarray(X, dtype=np.float64)).astype(np.float32)


class ClassicalHopfield(Scorer):
    """s(x) = ||Xi^T x||^2 = x^T W x, W = Xi Xi^T. Quadratic, sign-blind baseline."""

    def fit(self, Xi: np.ndarray) -> "ClassicalHopfield":
        Xi = np.asarray(Xi, dtype=np.float64)
        self._W = Xi @ Xi.T
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return (X * (self._W @ X)).sum(axis=0).astype(np.float32)


# The six scorers of the main table, in table order. Mahalanobis and Hybrid are used
# only by the robustness studies (contamination.py, ...), which request them by name.
SCORER_NAMES = ["knn", "modern_hopfield", "kde", "ocsvm", "centroid", "classical_hopfield"]


def make_scorer(name: str, cfg: dict) -> Scorer:
    """Build a fresh scorer from config['scorers'][name] (+ ocsvm_max_fit)."""
    p = cfg["scorers"][name]
    if name == "knn":
        return KNNScorer(k=p["k"])
    if name == "modern_hopfield":
        return ModernHopfield(beta=p["beta"], geom_dim=cfg.get("geom_dim"))
    if name == "mahalanobis":
        return Mahalanobis(kmeans_k=p["kmeans_k"], seed=p["kmeans_seed"])
    if name == "hybrid":
        return Hybrid(alpha=p.get("alpha", "auto"), knn_k=p.get("knn_k", 1),
                     kmeans_k=p.get("kmeans_k", 50), kmeans_seed=p.get("kmeans_seed", 0),
                     geom_dim=cfg.get("geom_dim"))
    if name == "kde":
        return KDE(bandwidth=p["bandwidth"])
    if name == "ocsvm":
        return OCSVM(nu=p["nu"], gamma=p["gamma"], max_fit=cfg.get("ocsvm_max_fit", 10000),
                    nystroem_components=cfg.get("ocsvm_nystroem_components"))
    if name == "centroid":
        return Centroid()
    if name == "classical_hopfield":
        return ClassicalHopfield()
    raise ValueError(f"unknown scorer {name!r}")
