import unittest

import numpy as np

from workflows.sbi.plot_p12_tarp_curve import (
    curve_subsample_indices,
    ordered_curve,
)


class P12TarpCurveTests(unittest.TestCase):
    def test_ordered_curve_sorts_alpha_and_ecp_together(self):
        ecp, alpha = ordered_curve(
            np.asarray([0.8, 0.2, 0.5]),
            np.asarray([0.9, 0.1, 0.5]),
        )
        np.testing.assert_allclose(alpha, [0.1, 0.5, 0.9])
        np.testing.assert_allclose(ecp, [0.2, 0.5, 0.8])

    def test_ordered_curve_rejects_duplicate_alpha(self):
        with self.assertRaisesRegex(ValueError, "not unique"):
            ordered_curve(np.asarray([0.1, 0.2]), np.asarray([0.5, 0.5]))

    def test_ordered_curve_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "invalid TARP curve shapes"):
            ordered_curve(np.ones((2, 2)), np.ones(4))

    def test_curve_subsample_preserves_worst_deviation(self):
        alpha = np.linspace(0.0, 1.0, 1001)
        ecp = alpha.copy()
        ecp[777] += 0.2
        selected = curve_subsample_indices(alpha, ecp, max_points=101)
        self.assertIn(0, selected)
        self.assertIn(777, selected)
        self.assertIn(1000, selected)
        self.assertLessEqual(len(selected), 102)


if __name__ == "__main__":
    unittest.main()
