import unittest

import numpy as np

from workflows.sbi.p12f3l2_conditional_autopsy import (
    core_bootstrap_coverage,
    global_scale_scan,
    interval_arrays,
    quantile_labels,
    subset_metrics,
)


class ConditionalAutopsyTests(unittest.TestCase):
    def test_interval_and_location_scale_diagnostics(self):
        rng = np.random.default_rng(3)
        truth = rng.normal(size=200)
        draws = rng.normal(size=(64, 200))
        arrays = interval_arrays(draws, truth)
        metrics = subset_metrics(arrays, np.ones(200, dtype=bool))
        self.assertLess(abs(metrics["rank_mean"] - 0.5), 0.06)
        self.assertTrue(0.7 < metrics["oracle_recentered_scale_to_68"] < 1.3)

    def test_quantiles_and_block_bootstrap_keep_core_as_unit(self):
        values = np.arange(400, dtype=float)
        labels, edges = quantile_labels(values)
        self.assertEqual(set(labels), {0, 1, 2, 3})
        self.assertEqual(len(edges), 5)
        rng = np.random.default_rng(5)
        draws = values[None] + rng.normal(size=(64, len(values)))
        arrays = interval_arrays(draws, values)
        arrays["core"] = np.repeat(np.arange(20), 20)
        report = core_bootstrap_coverage(arrays, labels, repeats=100, seed=7)
        self.assertEqual(report["spatial_blocks"], 20)
        self.assertEqual(len(report["maximum_bin_error_quantiles_05_50_95"]), 3)

    def test_global_scale_scan_detects_overdispersion(self):
        rng = np.random.default_rng(11)
        truth = rng.normal(size=400)
        draws = truth[None] + 1.6 * rng.normal(size=(64, 400))
        arrays = interval_arrays(draws, truth)
        environment = np.repeat(np.arange(4), 100)
        report = global_scale_scan(arrays, environment)
        self.assertLess(report["best_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
