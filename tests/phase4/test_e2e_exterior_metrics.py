import unittest
import numpy as np
from workflows.sbi.e2e_exterior_metrics import joint_rank, distance_masks
from workflows.sbi.e2e_exterior_campaign import phase_guard


class ExteriorMetricTests(unittest.TestCase):
    def test_gaussian_joint_rank_and_failure(self):
        rng = np.random.default_rng(392)
        good, bad = [], []
        for _ in range(300):
            z = rng.normal(size=(33, 4)); z[:, 2:] += .9*z[:, :2]
            good.append(joint_rank(z[1:], z[0])['rank'])
            bad.append(joint_rank(z[1:]*.2, z[0])['rank'])
        self.assertLess(abs(np.mean(np.asarray(good) <= .9)-.9), .07)
        self.assertLess(np.mean(np.asarray(bad) <= .9), .5)

    def test_guard_rejects_fresh_before_io(self):
        for phase in ('ph018', 'ph019', 'ph001', 'ph006'):
            for purpose in ('fit', 'development', 'final'):
                with self.assertRaises(PermissionError):
                    phase_guard(phase, purpose)

    def test_distance_masks_partition_core_cells(self):
        support = np.ones((64, 48, 48)); support[:20] = 0; support[20] = .5
        masks = distance_masks(support)
        np.testing.assert_array_equal(sum(masks.values()), np.ones((2, 16, 16, 16)))


if __name__ == '__main__':
    unittest.main()
