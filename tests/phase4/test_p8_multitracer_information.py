from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial import cKDTree

from workflows.abacus_tweb.p8_multitracer_information_audit import (
    correlation_from_moments,
    nearest_other_distance,
    shell_index,
    update_moments,
)


class MultitracerInformationTests(unittest.TestCase):
    def test_shell_index_uses_registered_half_open_bins(self) -> None:
        redshift = np.array([0.149, 0.15, 0.249, 0.25, 0.45, 0.549, 0.55])
        np.testing.assert_array_equal(shell_index(redshift), [-1, 0, 0, 1, 3, 3, -1])

    def test_streamed_correlation_matches_numpy(self) -> None:
        x = np.array([-2.0, -1.0, 1.0, 3.0])
        y = np.array([-1.0, -0.5, 1.5, 2.0])
        moment = np.zeros(6)
        update_moments(moment, x[:2], y[:2])
        update_moments(moment, x[2:], y[2:])
        self.assertAlmostEqual(
            correlation_from_moments(moment), float(np.corrcoef(x, y)[0, 1])
        )

    def test_nearest_other_skips_self_match(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [7.0, 0.0, 0.0]])
        distance = nearest_other_distance(cKDTree(points), points[:2], workers=1)
        np.testing.assert_allclose(distance, [2.0, 2.0])


if __name__ == "__main__":
    unittest.main()
