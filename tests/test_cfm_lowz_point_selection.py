import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import diagnose_cfm_lowz_point_selection as diagnostic


class PointSelectionTests(unittest.TestCase):
    def test_exact_membership_and_bin_edges(self):
        inside, outside = diagnostic.point_histogram(
            [.01, .02, .025, .03, .04], np.array([0, 1, 0, 1, 0]), [True, False], [.01, .02, .03])
        np.testing.assert_array_equal(inside, [1, 1])
        np.testing.assert_array_equal(outside, [0, 2])

    def test_bad_pixels_rejected(self):
        with self.assertRaises(ValueError):
            diagnostic.point_histogram([.01], np.array([-1]), [True], [.01, .02])

    def test_phase_guard_before_read(self):
        with patch.object(diagnostic, 'require_compute'), patch.object(diagnostic.lowz, 'load_plan') as load:
            with self.assertRaises(ValueError):
                diagnostic.run('ph018')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
