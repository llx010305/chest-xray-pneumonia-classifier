"""Metric helpers: AUC bootstrap, Youden threshold, confusion counts, Wilson and Welch."""

from __future__ import annotations

import unittest

import numpy as np
from scipy import stats

from src.eda.explore import welch_difference_ci, wilson_interval
from src.evaluation.metrics import (
    binary_report,
    bootstrap_auc,
    bootstrap_auc_difference,
    negatives_report,
    youden_threshold,
)


class BinaryReportTest(unittest.TestCase):
    def test_counts_and_rates(self):
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1])
        report = binary_report(y_true, y_prob, threshold=0.5)
        self.assertEqual((report["tp"], report["fn"], report["fp"], report["tn"]), (3, 1, 1, 3))
        self.assertAlmostEqual(report["sensitivity"], 0.75)
        self.assertAlmostEqual(report["specificity"], 0.75)
        self.assertAlmostEqual(report["accuracy"], 0.75)

    def test_threshold_is_inclusive(self):
        report = binary_report(np.array([1, 0]), np.array([0.5, 0.4]), threshold=0.5)
        self.assertEqual(report["tp"], 1)


class YoudenTest(unittest.TestCase):
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        threshold = youden_threshold(y_true, y_prob)
        self.assertTrue(0.3 < threshold <= 0.7)
        report = binary_report(y_true, y_prob, threshold)
        self.assertEqual(report["accuracy"], 1.0)

    def test_threshold_is_finite(self):
        rng = np.random.default_rng(3)
        y_true = rng.integers(0, 2, 200)
        threshold = youden_threshold(y_true, rng.random(200))
        self.assertTrue(np.isfinite(threshold))


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.y_true = np.repeat([0, 1], 200)
        self.y_prob = np.clip(self.y_true * 0.3 + rng.normal(0.35, 0.2, 400), 0, 1)

    def test_interval_contains_point_estimate(self):
        result = bootstrap_auc(self.y_true, self.y_prob, n_resamples=300)
        self.assertLessEqual(result["ci95_low"], result["auc"])
        self.assertGreaterEqual(result["ci95_high"], result["auc"])
        self.assertEqual(result["resamples"], 300)

    def test_same_seed_is_reproducible(self):
        first = bootstrap_auc(self.y_true, self.y_prob, n_resamples=200, seed=7)
        second = bootstrap_auc(self.y_true, self.y_prob, n_resamples=200, seed=7)
        self.assertEqual(first, second)

    def test_model_against_itself_has_zero_difference(self):
        result = bootstrap_auc_difference(self.y_true, self.y_prob, self.y_prob, n_resamples=200)
        self.assertEqual(result["difference"], 0.0)
        self.assertEqual(result["ci95_low"], 0.0)
        self.assertEqual(result["ci95_high"], 0.0)


class NegativesReportTest(unittest.TestCase):
    def test_counts_false_positives_on_normals(self):
        report = negatives_report(np.array([0.1, 0.4, 0.5, 0.9]), threshold=0.5)
        self.assertEqual((report["fp"], report["tn"], report["images"]), (2, 2, 4))
        self.assertAlmostEqual(report["specificity"], 0.5)
        low, high = report["specificity_wilson95"]
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)


class WilsonTest(unittest.TestCase):
    def test_known_value(self):
        # 1500 of 3000, as in the README: 0.482-0.518.
        low, high = wilson_interval(1500, 3000)
        self.assertAlmostEqual(low, 0.4821, places=4)
        self.assertAlmostEqual(high, 0.5179, places=4)

    def test_stays_inside_zero_one(self):
        low, high = wilson_interval(0, 20)
        self.assertGreaterEqual(low, 0.0 - 1e-12)
        self.assertLess(high, 1.0)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            wilson_interval(0, 0)


class WelchTest(unittest.TestCase):
    def test_matches_scipy_and_contains_difference(self):
        rng = np.random.default_rng(5)
        left = rng.normal(10, 2, 300)
        right = rng.normal(9, 3, 250)
        result = welch_difference_ci(left, right)
        reference = stats.ttest_ind(left, right, equal_var=False)
        self.assertAlmostEqual(result["t_statistic"], float(reference.statistic), places=10)
        self.assertAlmostEqual(result["p_value"], float(reference.pvalue), places=10)
        self.assertLess(result["ci95_low"], result["difference_pneumonia_minus_normal"])
        self.assertGreater(result["ci95_high"], result["difference_pneumonia_minus_normal"])


if __name__ == "__main__":
    unittest.main()
