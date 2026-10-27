"""Selection-rank interval inference against explicit small halo selections."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

WORKFLOWS = Path(__file__).resolve().parents[1] / 'workflows/abacus_tweb'
sys.path.insert(0, str(WORKFLOWS))
spec = importlib.util.spec_from_file_location('lowz_recovery', WORKFLOWS/'recover_lowz_filtered_index.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RadiusIntervals(unittest.TestCase):
    def test_rank_constraints_agree_with_direct_selection(self):
        radii = np.array([1., 10.01, 2., 10.03, 3.])
        parents = np.array([2, 4])
        intervals = module.feasible_intervals([1, 2], radii[[1, 3]], [1, 3],
                                              parents, [2, 3], 10., 10.05)
        self.assertEqual(intervals, [[10.01, 10.03]])
        for threshold in np.linspace(10.0001, 10.0499, 80):
            direct = np.searchsorted(np.flatnonzero(radii < threshold), parents)
            feasible = any(lo < threshold < hi for lo, hi in intervals)
            self.assertEqual(np.array_equal(direct, [2, 3]), feasible)

    def test_tied_boundary_events_cannot_be_selected_separately(self):
        self.assertEqual(module.feasible_intervals([0], [10.01, 10.01], [0, 1],
                                                   [2], [1], 10., 10.05), [])

    def test_conflicting_training_constraints_have_no_solution(self):
        self.assertEqual(module.feasible_intervals([1, 2], [10.01, 10.03], [1, 3],
                                                   [2, 4], [1, 4], 10., 10.05), [])

    def test_empty_training_is_rejected(self):
        with self.assertRaises(ValueError):
            module.feasible_intervals([], [], [], [], [], 10., 10.05)


if __name__ == '__main__':
    unittest.main()
