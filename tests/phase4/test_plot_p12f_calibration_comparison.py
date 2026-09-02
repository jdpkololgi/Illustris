import unittest

import numpy as np

from workflows.sbi.plot_p12f_calibration_comparison import (
    clustered_decile_interval,
    decile_mass,
)


class P12FCalibrationPlotTests(unittest.TestCase):
    def test_decile_mass_is_uniform_for_centres(self):
        ranks = (np.arange(10, dtype=np.float64) + 0.5) / 10.0
        np.testing.assert_allclose(decile_mass(ranks), np.full(10, 0.1))

    def test_decile_mass_rejects_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "must lie"):
            decile_mass(np.asarray([-0.01, 0.5]))

    def test_cluster_interval_has_expected_shape_and_contains_point(self):
        ranks = np.tile((np.arange(10) + 0.5) / 10.0, 8)
        groups = np.repeat(np.arange(8), 10)
        interval = clustered_decile_interval(ranks, groups, repeats=100, seed=7)
        self.assertEqual(interval.shape, (10, 3))
        self.assertTrue(np.all(interval[:, 0] <= 0.1))
        self.assertTrue(np.all(interval[:, 2] >= 0.1))

    def test_cluster_interval_requires_multiple_groups(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            clustered_decile_interval(
                np.asarray([0.1, 0.9]), np.asarray([3, 3]), repeats=10
            )


if __name__ == "__main__":
    unittest.main()
