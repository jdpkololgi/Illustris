import unittest

import numpy as np

from workflows.sbi.p12_width_information_diagnostics import (
    block_bootstrap_corr,
    coverage_by_shell,
    residualize_by_group,
    weighted_corr,
    weighted_quantile,
    weighted_quantile_bin,
)


class P12WidthInformationDiagnosticsTests(unittest.TestCase):
    def test_weighted_quantile_and_corr(self):
        value = np.asarray([0.0, 1.0, 2.0, 3.0])
        weight = np.ones(4)
        self.assertAlmostEqual(float(weighted_quantile(value, 0.5, weight)[0]), 1.5)
        self.assertAlmostEqual(weighted_corr(value, 2.0 * value, weight), 1.0)

    def test_group_residuals_have_zero_weighted_mean(self):
        value = np.asarray([1.0, 3.0, 10.0, 14.0])
        group = np.asarray([0, 0, 1, 1])
        weight = np.asarray([1.0, 3.0, 2.0, 2.0])
        residual = residualize_by_group(value, group, weight)
        for selected in (group == 0, group == 1):
            self.assertAlmostEqual(np.average(residual[selected], weights=weight[selected]), 0.0)

    def test_weighted_quantile_bins_follow_population_weight(self):
        value = np.arange(8, dtype=float)
        weight = np.ones(8)
        group = weighted_quantile_bin(value, weight, 4)
        self.assertEqual(group.tolist(), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_block_bootstrap_corr_recovers_positive_relation(self):
        rng = np.random.default_rng(7)
        groups = np.repeat(np.arange(20), 10)
        x = rng.normal(size=200)
        y = 0.8 * x + rng.normal(scale=0.2, size=200)
        result = block_bootstrap_corr(x, y, np.ones(200), groups, 100, 8)
        self.assertGreater(result["correlation"], 0.8)
        self.assertGreater(result["bootstrap_q025_q50_q975"][0], 0.7)

    def test_coverage_reports_lower_and_upper_tail_misses(self):
        truth = np.tile(np.asarray([[-2.0], [0.0], [2.0]]), (4, 3))
        intervals = np.empty((12, 4, 3), dtype=float)
        intervals[:, 0, :] = -3.0
        intervals[:, 1, :] = -1.0
        intervals[:, 2, :] = 1.0
        intervals[:, 3, :] = 3.0
        shell = np.repeat(np.arange(4), 3)
        report = coverage_by_shell(
            intervals,
            truth,
            shell,
            np.ones(12),
        )
        for shell_report in report.values():
            self.assertEqual(shell_report["coverage68"], [1.0 / 3.0] * 3)
            self.assertEqual(shell_report["below68"], [1.0 / 3.0] * 3)
            self.assertEqual(shell_report["above68"], [1.0 / 3.0] * 3)
            self.assertEqual(shell_report["coverage90"], [1.0] * 3)


if __name__ == "__main__":
    unittest.main()
