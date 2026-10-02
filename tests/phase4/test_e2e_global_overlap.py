"""Geometry regressions, not posterior-calibration tests."""
import unittest
import numpy as np
from workflows.sbi.e2e_loa_seam_audit import seam_mask
from workflows.sbi.e2e_global_overlap_probe import slices, overlap


class GlobalOverlapGeometry(unittest.TestCase):
    def test_seam_requires_existing_neighbour(self):
        plan = {
            'ownership_raw_stride': [64, 32, 32],
            'grids': {cap: {'origin_mpc_h': [0, 0, 0], 'cell_mpc_h': 1.0}
                      for cap in ['NGC', 'SGC']},
            'tiles': [{'cap': 'NGC', 'key': [0, 0, 0]},
                      {'cap': 'NGC', 'key': [1, 0, 0]}],
        }
        xyz = np.array([[1, 1, 1], [63, 16, 16], [65, 16, 16],
                        [90, 1, 16], [127, 16, 16], [32, 16, 16]])
        actual = seam_mask(xyz, np.ones(6), np.array([0, 0, 1, 1, 1, 0]), plan)
        np.testing.assert_array_equal(actual, [False, True, True, False, False, False])

    def test_physical_overlap_alignment(self):
        for stage, shape in [('coarse', (56, 48, 48)), ('fine', (96, 48, 48))]:
            field = np.arange(np.prod(shape)).reshape(1, 1, *shape)
            left, right = overlap(field[slices(stage, 0)], field[slices(stage, 1)], stage)
            self.assertGreater(left.size, 0)
            np.testing.assert_array_equal(left, right)

    def test_fine_shift_preserves_four_cell_blocks(self):
        self.assertEqual(slices('fine', 1)[2].start % 4, 0)
        self.assertEqual(slices('fine', 1)[2].start // 4,
                         slices('coarse', 1)[2].start)


if __name__ == '__main__':
    unittest.main()
