import unittest

import numpy as np

from workflows.abacus_tweb.p8_density_target_alignment import score, sky_to_periodic


class P8DensityTargetAlignmentTests(unittest.TestCase):
    def test_sky_to_periodic_wraps(self):
        xyz = sky_to_periodic(
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.2]),
            origin_mpc_h=-1000.0,
            boxsize_mpc_h=2000.0,
        )
        self.assertEqual(xyz.shape, (1, 3))
        self.assertTrue(np.all(xyz >= 0.0))
        self.assertTrue(np.all(xyz < 2000.0))

    def test_score_exact_prediction(self):
        truth = np.asarray(
            [[-1.0, 0.0, 1.0], [-0.5, 0.5, 2.0], [0.2, 1.0, 3.0]],
            dtype=np.float64,
        )
        report = score(truth, truth)
        self.assertAlmostEqual(report["minimum_r2"], 1.0)
        self.assertAlmostEqual(report["mean_r2"], 1.0)
        for row in report["eigenvalues"].values():
            self.assertAlmostEqual(row["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
