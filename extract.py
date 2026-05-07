"""Build the feature caches from raw MVTec 3D-AD scans (one-time, deterministic).

Every stage writes ``cache/features/<class>_<split>_<rep>.npz``, which the rest of the
pipeline reads:

  geom       : preprocess -> per-point FPFH (Open3D) -> 28x28 patch pooling -> L2-normalize
               -> ``*_abs.npz`` (33-dim, geometry only)
  rgb        : add a frozen DINOv2 ViT-S/14 patch token to every cached FPFH patch
               -> ``*_mm.npz``  (417-dim multimodal, aligned 1:1 to ``*_abs`` by construction)
  rgb_v3     : the same with DINOv3 ViT-S/16 -> ``*_mm_v3.npz`` (alternative encoder;
               ``features: mm_v3`` scores it)
  validation : the geom stage on the normal-only validation split (operating_point.py)
  corrupt    : the geom stage on corrupted scans (corruption grid, see src/corrupt.py)

    python extract.py            # geom then rgb (all classes, train + test)
    python extract.py geom       # geometry cache only (no torch/timm needed)
    python extract.py rgb        # multimodal cache only (reads the geom cache)
    python extract.py rgb_v3     # DINOv3 multimodal cache only (reads the geom cache)
    python extract.py validation # the dataset's held-out normal-only split -> *_validation_abs.npz
    python extract.py corrupt    # corrupted-geometry caches for corruption.py

One-time and deterministic (Open3D seeded by ``base_seed``); existing caches are skipped
so re-runs resume. geom needs open3d + tifffile and the raw dataset at paths.data_root;
rgb / rgb_v3 need torch + timm (weights download once, needs network) and the geom cache.

The rgb stage adds a 384-dim DINOv2 token to each FPFH patch:

    patches = L2( concat( FPFH_33 , L2(DINOv2_384) ) )      -> 417-dim, unit sphere

Alignment is the whole point: DINOv2 ViT-S/14 at image_size 392 emits 392/14 = 28x28 =
784 tokens, the SAME 28x28 grid the FPFH pipeline pooled into, so token (i,j) describes
FPFH patch (i,j). rgb_v3 does the same at DINOv3 ViT-S/16's own (patch, image_size) =
(16, 448) -- also a 28x28 grid, so it still lands on the same FPFH patch grid, just with
a different token count (384-dim either way; DINOv3-vits16 also emits 384-dim tokens).
Both rgb stages read only the geom cache + raw RGB (no Open3D), keep exactly the tokens
whose patch_idx the geom cache kept (in cache order), and so share scan_index / patch_idx
with it by construction (asserted).
"""
from __future__ import annotations

import os

# Determinism: Open3D's FPFH/normal estimation and RANSAC are parallelized with OpenMP,
# whose reduction/normal-sign order is non-deterministic across runs (it can even shift
# patch counts). Pinning a single thread makes extraction bit-reproducible (run-to-run
# maxdev 0.0). Must be set BEFORE Open3D is imported (via src.features). One-time cost
# only; scoring (run.py) is a separate process and stays multi-threaded.
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

_PATCH = 14   # DINOv2 ViT-S/14 patch size; image_size / _PATCH = tokens per side


def _cache_path(cfg: dict, cls: str, split: str, rep: str) -> str:
    return os.path.join(cfg["paths"]["cache_dir"], f"{cls}_{split}_{rep}.npz")


def _run(cfg: dict, rep: str, build, splits: tuple = ("train", "test")) -> None:
    """Loop classes x splits, skipping existing caches; ``build(cls, split)`` -> path."""
    for cls in cfg["classes"]:
        for split in splits:
            out = _cache_path(cfg, cls, split, rep)
            if os.path.exists(out):
                print(f"[{cls}/{split}] {rep} cached, skip", flush=True)
                continue
            t0 = time.time()
            path = build(cls, split)
            d = np.load(path, allow_pickle=False)
            print(f"[{cls}/{split}] {d['patches'].shape[0]} patches x {d['patches'].shape[1]}d, "
                  f"{d['scan_ids'].shape[0]} scans -> {os.path.basename(str(path))} "
                  f"({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# geom: per-point FPFH -> 28x28 patches -> *_abs.npz                           #
# --------------------------------------------------------------------------- #
def extract_geom(cfg: dict) -> None:
    from src.features import cache_class_split, seed_open3d   # imports Open3D (after OMP)
    seed = seed_open3d(cfg)
    print(f"geom: seeded Open3D global RNG with base_seed={seed}", flush=True)
    _run(cfg, "abs", lambda cls, split: cache_class_split(cls, split, cfg))


# --------------------------------------------------------------------------- #
# validation: the dataset's own held-out NORMAL-only split -> *_validation_abs.npz #
# --------------------------------------------------------------------------- #
def extract_validation(cfg: dict) -> None:
    """Geometry cache of MVTec 3D-AD's ``validation`` split (normal-only, disjoint from
    ``train`` and ``test``), used by ``operating_point.py`` to calibrate a fixed threshold
    without touching the test set. Same pipeline and cache schema as ``geom``.
    """
    from src.features import cache_class_split, seed_open3d
    seed_open3d(cfg)
    _run(cfg, "abs", lambda cls, split: cache_class_split(cls, split, cfg),
        splits=("validation",))


# --------------------------------------------------------------------------- #
# corrupt: corrupted-geometry variant caches (corruption grid) -> *_abs__<cell>.npz #
# --------------------------------------------------------------------------- #
def extract_corrupt_cell(cls: str, split: str, family: str, severity: int, cfg: dict,
                         pipeline_tag: str = "", seed: int = 0) -> "Path | None":
    """Build one corrupted variant cache (skipping if present); return its path.

    ``pipeline_tag`` selects an extraction-pipeline variant (``pipeline_cfg``: 'nr',
    'nc', 'nrnc', 'rel', 'adaptive') for the mitigation probes; the tag becomes part of
    the cache suffix so every variant is cached once. ``seed`` (0 = the grid's single draw) selects an independent
    corruption draw for corruption.py's multi-seed probe. Runs the exact same extract
    path as `geom`, only with the corruption applied to the organized scan before
    preprocessing.
    """
    from src.corrupt import apply_corruption, cache_suffix
    from src.features import cache_class_split, cache_path
    from src.preprocess import pipeline_cfg

    suffix = cache_suffix(family, severity, pipeline_tag, seed)
    if os.path.exists(cache_path(cls, split, cfg, suffix)):
        return None
    vcfg = pipeline_cfg(cfg, pipeline_tag)
    fn = lambda xyz, valid, scan_id: apply_corruption(xyz, valid, scan_id,
                                                      family, severity, cfg, seed)
    return cache_class_split(cls, split, vcfg, corrupt_fn=fn, suffix=suffix)


def extract_corrupt(cfg: dict) -> None:
    """All (class x split x family x severity) corrupted caches of config `corruption`."""
    ccfg = cfg["corruption"]
    for cls in ccfg["classes"]:
        for split in ("train", "test"):
            for family, sevs in ccfg["severities"].items():
                for severity in range(1, len(sevs) + 1):
                    t0 = time.time()
                    path = extract_corrupt_cell(cls, split, family, severity, cfg)
                    tag = f"[{cls}/{split}] {family}{severity}"
                    if path is None:
                        print(f"{tag} cached, skip", flush=True)
                    else:
                        print(f"{tag} -> {os.path.basename(str(path))} "
                              f"({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------------------- #
# rgb: frozen ViT patch-token encoder per FPFH patch -> *_mm[_v3].npz          #
# --------------------------------------------------------------------------- #
class PatchTokenEncoder:
    """Frozen ViT patch-token encoder (CPU; lazy torch/timm import).

    Backbone-agnostic: any timm ViT that exposes patch tokens via forward_features
    works, given its own (image_size, patch_size) so the token grid lands on the SAME
    28x28 grid the FPFH pipeline pooled into. Used for both DINOv2 (patch 14, the
    default) and DINOv3 (patch 16, an alternative encoder option -- see extract_rgb_v3).
    """

    def __init__(self, backbone: str, image_size: int, patch_size: int = _PATCH):
        self.backbone = backbone
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size        # e.g. 392 // 14 = 28
        self._model = None

    def _build(self) -> None:
        import timm
        import torch

        model = timm.create_model(self.backbone, pretrained=True, num_classes=0,
                                  img_size=self.image_size)
        model.eval()
        dcfg = timm.data.resolve_model_data_config(model)
        self._mean = np.asarray(dcfg["mean"], dtype=np.float32)
        self._std = np.asarray(dcfg["std"], dtype=np.float32)
        self._n_prefix = int(getattr(model, "num_prefix_tokens", 1))
        self._torch = torch
        self._model = model

    def encode(self, rgb_path: str) -> np.ndarray:
        """RGB png path -> (grid*grid, 384) float32 tokens, row-major i*grid + j."""
        if self._model is None:
            self._build()
        from PIL import Image

        img = Image.open(rgb_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - self._mean) / self._std
        x = self._torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)     # (1,3,S,S)
        with self._torch.no_grad():
            feats = self._model.forward_features(x)                         # (1,N,C)
        tok = feats[0, self._n_prefix:, :].cpu().numpy().astype(np.float32)
        g = self.grid
        if tok.shape[0] != g * g:
            raise ValueError(f"{self.backbone} produced {tok.shape[0]} tokens, expected "
                             f"{g * g}; check image_size ({self.image_size}) vs patch "
                             f"({self.patch_size})")
        return tok


def _l2_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalize the rows of (M, D) onto the unit sphere."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(n == 0.0, 1.0, n)).astype(np.float32)


def _rgb_path(data_root: str, scan_id: str) -> str:
    """'<class>/<split>/<defect>/<stem>' -> data_root/.../<defect>/rgb/<stem>.png."""
    cls, split, defect, stem = scan_id.split("/")
    return os.path.join(data_root, cls, split, defect, "rgb", stem + ".png")


def _extract_mm(cls: str, split: str, cfg: dict, enc: PatchTokenEncoder, rep: str = "mm"
                ) -> str:
    """Write <class>_<split>_<rep>.npz aligned to the FPFH cache; return its path."""
    from src.data import load_cache

    geom = load_cache(cls, split, {**cfg, "features": "abs"})
    si, pi = geom["scan_index"], geom["patch_idx"]
    data_root = cfg["paths"]["data_root"]

    rgb_rows, si_built, pi_built = [], [], []
    for s, scan_id in enumerate(geom["scan_ids"]):
        kept = pi[si == s]                                  # patch positions, cache order
        if kept.size:
            tok = enc.encode(_rgb_path(data_root, str(scan_id)))   # (grid*grid, D)
            rgb_rows.append(_l2_rows(tok[kept]))                   # L2 each token
            si_built.append(np.full(kept.size, s, dtype=si.dtype))
            pi_built.append(kept)

    rgb = np.concatenate(rgb_rows)                           # (M, D)
    # rows align to the FPFH cache 1:1 by construction -- verify before trusting the hstack
    assert np.array_equal(np.concatenate(si_built), si), "scan_index diverged from FPFH cache"
    assert np.array_equal(np.concatenate(pi_built), pi), "patch_idx diverged from FPFH cache"

    mm = _l2_rows(np.hstack([geom["patches"], rgb]))         # (M, 33+D) on the unit sphere
    out = _cache_path(cfg, cls, split, rep)
    np.savez_compressed(out, patches=mm, scan_index=si, patch_idx=pi,
                        scan_ids=geom["scan_ids"], defect_type=geom["defect_type"],
                        patch_counts=geom["patch_counts"])
    return out


def extract_rgb(cfg: dict) -> None:
    rcfg = cfg["rgb"]
    enc = PatchTokenEncoder(rcfg["backbone"], rcfg["image_size"], rcfg.get("patch_size", _PATCH))
    _run(cfg, "mm", lambda cls, split: _extract_mm(cls, split, cfg, enc, rep="mm"))


def extract_rgb_v3(cfg: dict) -> None:
    """Alternative encoder: DINOv3 patch tokens instead of DINOv2, from ``rgb_v3`` in
    config.yaml -> ``*_mm_v3.npz`` (set ``features: mm_v3`` to score it). Same
    dependencies as extract_rgb; the weights are gated under the DINOv3 License (see
    README) and download on the first run.
    """
    rcfg = cfg["rgb_v3"]
    enc = PatchTokenEncoder(rcfg["backbone"], rcfg["image_size"], rcfg.get("patch_size", 16))
    _run(cfg, "mm_v3", lambda cls, split: _extract_mm(cls, split, cfg, enc, rep="mm_v3"))


def main(config_path: str = "config.yaml", mode: str = "all") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if mode not in ("all", "geom", "rgb", "rgb_v3", "corrupt", "validation"):
        raise SystemExit("usage: python extract.py [all|geom|rgb|rgb_v3|corrupt|validation]  "
                         f"(got {mode!r})")
    if mode in ("all", "geom"):
        extract_geom(cfg)
    if mode in ("all", "rgb"):
        extract_rgb(cfg)
    if mode == "rgb_v3":
        extract_rgb_v3(cfg)
    if mode == "corrupt":
        extract_corrupt(cfg)
    if mode == "validation":
        extract_validation(cfg)


if __name__ == "__main__":
    main(mode=sys.argv[1] if len(sys.argv) > 1 else "all")
