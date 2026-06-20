"""Unit tests for the fixed-operating-point machinery (src/evaluate.py) and the
formal significance test's p-value helper (significance.py). Pure arithmetic, no
data/cache dependency -- run with `pytest -q`."""
import numpy as np

from significance import _p_value
from src.evaluate import (
    expected_cost_per_scan,
    operating_point_metrics,
    threshold_from_normals,
)


def test_threshold_from_normals_is_a_percentile():
    normals = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert threshold_from_normals(normals, 0.10) == np.percentile(normals, 10)
    # a stricter (lower) target FPR asks for a lower threshold -- fewer normals flagged
    assert threshold_from_normals(normals, 0.01) <= threshold_from_normals(normals, 0.10)


def test_operating_point_metrics_confusion_counts():
    # 3 anomalies (normality 0,1,2 -- lowest), 3 normals (normality 3,4,5); threshold=2.5
    # flags normality<2.5 as anomalous -> exactly the 3 true anomalies, zero false alarms.
    normality = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    is_anom = np.array([True, True, True, False, False, False])
    m = operating_point_metrics(normality, is_anom, threshold=2.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 0, 0, 3)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0 and m["fpr"] == 0.0


def test_operating_point_metrics_partial_and_degenerate():
    normality = np.array([0.0, 1.0, 2.0, 3.0])
    is_anom = np.array([True, True, False, False])
    m = operating_point_metrics(normality, is_anom, threshold=1.5)   # flags idx 0,1
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 0, 0, 2)
    # threshold below everything -> no positive predictions -> precision undefined (nan)
    m2 = operating_point_metrics(normality, is_anom, threshold=-1.0)
    assert m2["tp"] == 0 and m2["fp"] == 0 and np.isnan(m2["precision"])
    assert m2["recall"] == 0.0   # tp+fn > 0, so recall is well-defined (just zero)


def test_expected_cost_per_scan():
    counts = {"tp": 1, "fp": 2, "fn": 3, "tn": 4}   # n = 10
    # fn costs 5x fp: (3*5 + 2*1) / 10
    assert expected_cost_per_scan(counts, cost_fn=5, cost_fp=1) == (3 * 5 + 2 * 1) / 10
    assert expected_cost_per_scan(counts, cost_fn=0, cost_fp=0) == 0.0


def test_pvalue_symmetric_around_zero_is_not_significant():
    rng = np.random.default_rng(0)
    diff = rng.normal(loc=0.0, scale=1.0, size=2000)
    assert _p_value(diff) > 0.5   # roughly balanced tails -> large p


def test_pvalue_strictly_positive_diff_is_significant():
    diff = np.linspace(0.01, 1.0, 500)   # every draw favors the scorer over the baseline
    assert _p_value(diff) == 0.0
