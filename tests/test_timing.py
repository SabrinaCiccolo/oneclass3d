"""Unit test for timing.py's generic memory-footprint walker. Pure Python objects,
no data/cache dependency -- run with `pytest -q`."""
import numpy as np

from timing import _nbytes


class _Dummy:
    def __init__(self):
        self.arr = np.zeros(10, dtype=np.float32)          # 40 bytes
        self.nested = _NestedHolder()


class _NestedHolder:
    def __init__(self):
        self.a = np.zeros(5, dtype=np.float64)             # 40 bytes
        self.b = [np.zeros(3, dtype=np.float32), np.zeros(2, dtype=np.float32)]  # 12+8
        self.c = {"x": np.zeros(1, dtype=np.float64)}      # 8 bytes
        self.not_an_array = "hello"


def test_nbytes_sums_nested_ndarrays():
    d = _Dummy()
    assert _nbytes(d) == 40 + 40 + 12 + 8 + 8


def test_nbytes_does_not_double_count_shared_reference():
    shared = np.zeros(100, dtype=np.float32)   # 400 bytes
    d = _Dummy()
    d.a1 = shared
    d.a2 = shared   # same object, referenced twice
    assert _nbytes(d) == 40 + 40 + 12 + 8 + 8 + 400   # shared counted ONCE


def test_nbytes_plain_object_is_zero():
    assert _nbytes(object()) == 0
    assert _nbytes(42) == 0
    assert _nbytes("string") == 0
