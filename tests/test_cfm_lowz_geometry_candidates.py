import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import prepare_cfm_lowz_geometry_candidates as geometry


class GeometryCandidateTests(unittest.TestCase):
    def test_prefix_boxes_match_padded_direct_counts(self):
        support = np.random.default_rng(6).integers(0, 2, (7, 8, 9), dtype=np.uint8)
        starts = np.array([[-2, -3, -4], [0, 0, 0], [4, 5, 6], [10, 0, 0], [-4, -4, -4]])
        stops = starts+4
        actual = geometry.box_counts(geometry.prefix_sum(support), starts, stops)
        expected = []
        for start, stop in zip(starts, stops):
            lo, hi = np.clip(start, 0, support.shape), np.clip(stop, 0, support.shape)
            expected.append(support[tuple(slice(a, b) for a, b in zip(lo, hi))].sum())
        np.testing.assert_array_equal(actual, expected)

    def test_bad_boxes_and_nonbinary_support_rejected(self):
        with self.assertRaises(ValueError):
            geometry.prefix_sum(np.full((2, 2, 2), 2))
        with self.assertRaises(ValueError):
            geometry.box_counts(np.ones((3, 3, 3)), [[1, 1, 1]], [[0, 0, 0]])

    def test_guard_before_payload(self):
        with patch.object(geometry, 'require_compute'), patch.object(geometry.lowz, 'load_plan') as load:
            with self.assertRaises(ValueError):
                geometry.run('ph019')
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
