"""Point-cloud corruption suite for the robustness grid.

Each corruption is a pure function on the ORGANIZED scan -- ``(xyz (H,W,3), valid
(H,W)) -> (xyz', valid')`` -- applied right after loading and BEFORE preprocessing, so
RANSAC plane removal, DBSCAN cleanup and FPFH all run on corrupted geometry (that is
the point of the study). Severity parameters are scale-relative to the bbox diagonal
of the scan's valid points where a length is involved (config ``corruption.severities``).

Determinism: the rng for a scan's corruption derives from (scan_id, family, severity)
via ``stable_rng`` -- the exact corrupted cloud is regenerable on any machine, and no
corrupted copies of the (CC BY-NC-SA) dataset ever need redistributing.
"""
from __future__ import annotations

import numpy as np

from src.preprocess import bbox_diagonal
from src.repro import stable_rng

FAMILIES = ("gauss", "occlusion", "density", "quantize")


def gauss_noise(xyz: np.ndarray, valid: np.ndarray, sigma_rel: float,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Per-point isotropic Gaussian noise, sigma = sigma_rel * bbox diagonal."""
    sigma = sigma_rel * bbox_diagonal(xyz[valid])
    out = xyz.copy()
    out[valid] += rng.normal(0.0, sigma, size=(int(valid.sum()), 3)).astype(xyz.dtype)
    return out, valid.copy()


def occlusion(xyz: np.ndarray, valid: np.ndarray, cut_frac: float,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random half-plane cut in image space invalidating ``cut_frac`` of valid pixels.

    A random direction is drawn; the valid pixels whose (row, col) projection onto it
    is beyond the (1 - cut_frac) quantile are zeroed -- one contiguous side of the
    scan disappears, like a shadowed or out-of-view region.
    """
    ii, jj = np.nonzero(valid)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    proj = ii * np.cos(theta) + jj * np.sin(theta)
    drop = proj > np.quantile(proj, 1.0 - cut_frac)
    out_valid = valid.copy()
    out_valid[ii[drop], jj[drop]] = False
    out = xyz.copy()
    out[~out_valid] = 0.0
    return out, out_valid


def density(xyz: np.ndarray, valid: np.ndarray, keep_frac: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Uniform random dropout keeping ``keep_frac`` of the valid pixels."""
    idx = np.flatnonzero(valid.ravel())
    keep = rng.choice(idx.size, int(round(keep_frac * idx.size)), replace=False)
    out_valid = np.zeros(valid.size, dtype=bool)
    out_valid[idx[keep]] = True
    out_valid = out_valid.reshape(valid.shape)
    out = xyz.copy()
    out[~out_valid] = 0.0
    return out, out_valid


def quantize(xyz: np.ndarray, valid: np.ndarray, step_rel: float,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Quantize the z (depth) coordinate to steps of step_rel * bbox diagonal.

    Models a coarse depth discretization (coarse sensor model); deterministic --
    the rng argument is unused, kept for the shared signature.
    """
    step = step_rel * bbox_diagonal(xyz[valid])
    out = xyz.copy()
    out[valid, 2] = np.round(out[valid, 2] / step) * step
    return out, valid.copy()


_FUNCS = {"gauss": gauss_noise, "occlusion": occlusion, "density": density,
          "quantize": quantize}


def apply_corruption(xyz: np.ndarray, valid: np.ndarray, scan_id: str, family: str,
                     severity: int, cfg: dict, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Corrupt one organized scan; ``severity`` is 1-based into corruption.severities.

    No-op passthrough on scans with no valid points (nothing to corrupt). ``seed=0`` (the
    default) is the grid's single draw (its rng key omits the seed); ``seed>0`` derives an
    independent draw for the multi-seed corruption.py probe.
    """
    if not valid.any():
        return xyz, valid
    param = float(cfg["corruption"]["severities"][family][severity - 1])
    key = ("corrupt", scan_id, family, severity) if seed == 0 else \
        ("corrupt", scan_id, family, severity, seed)
    rng = stable_rng(*key)
    return _FUNCS[family](xyz, valid, param, rng)


def cache_suffix(family: str, severity: int, pipeline_tag: str = "", seed: int = 0) -> str:
    """The cache-name suffix for a corruption cell, e.g. '__gauss2' or '__gauss2__nr'.

    ``pipeline_tag`` marks mitigation-probe extraction variants (see
    ``src.preprocess.pipeline_cfg``); '' is the default pipeline. ``seed``
    (multi-seed corruption.py probe only) inserts '_s<seed>' right after the severity;
    seed=0 has no seed suffix.
    """
    s = f"__{family}{severity}" + (f"_s{seed}" if seed else "")
    return s + (f"__{pipeline_tag}" if pipeline_tag else "")
