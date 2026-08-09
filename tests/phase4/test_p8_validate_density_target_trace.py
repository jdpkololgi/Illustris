import unittest

import numpy as np

from workflows.abacus_tweb.p8_validate_density_target_trace import (
    scalar_score,
    sky_to_observer_mpc,
)


class DensityTargetTraceClosureTests(unittest.TestCase):
    def test_sky_to_observer_mpc_retains_observer_units(self):
        got = sky_to_observer_mpc(
            np.array([0.0, 90.0]), np.array([0.0, 0.0]), np.array([0.2, 0.2])
        )
        self.assertGreater(got[0, 0], 500.0)
        self.assertAlmostEqual(got[0, 1], 0.0, places=10)
        self.assertAlmostEqual(got[1, 0], 0.0, places=10)
        self.assertAlmostEqual(got[0, 0], got[1, 1], places=10)

    def test_scalar_score_is_exact_for_identical_values(self):
        truth = np.array([-1.0, 0.0, 2.0])
        score = scalar_score(truth.copy(), truth)
        self.assertEqual(score["r2"], 1.0)
        self.assertEqual(score["rmse"], 0.0)


if __name__ == "__main__":
    unittest.main()
