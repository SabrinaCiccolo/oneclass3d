"""Reproducibility helper for the study scripts (contamination / bank_size / corruption).

``stable_rng``: Python's ``hash()`` is salted per process, so seeds derived from it are
not reproducible across runs. We hash the '/'-joined key parts with sha256 instead: the
same (strings, ints) always yield the same generator, on any machine.
"""
from __future__ import annotations

import hashlib

import numpy as np


def stable_rng(*parts) -> np.random.Generator:
    """default_rng seeded from sha256("<part>/<part>/..."); deterministic across runs."""
    key = "/".join(str(p) for p in parts).encode()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "little"))

