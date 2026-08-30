import unittest

import numpy as np

from workflows.sbi.p12_affine_calibration_canary import (
    apply_correction,
    block_bootstrap_mean_difference,
    fit_moment_correction,
    parameter_stability,
)


class P12AffineCalibrationTests(unittest.TestCase):
    def test_moment_fit_recovers_injected_shell_offsets_and_scales(self):
        rng = np.random.default_rng(123)
        rows, draws = 4_000, 128
        shell = np.repeat(np.arange(4), rows // 4)
        centre = rng.normal(size=(rows, 3))
        samples = centre[:, None, :] + rng.normal(size=(rows, draws, 3))
        offset = np.asarray(
            [[0.02, -0.03, 0.01], [0.0, 0.01, -0.02],
             [-0.01, 0.02, 0.03], [0.04, -0.04, -0.03]]
        )
        scale = np.asarray(
            [[1.02, 0.98, 1.01], [0.95, 1.05, 1.0],
             [1.08, 0.97, 1.04], [1.10, 1.12, 1.08]]
        )
        truth = centre + offset[shell] + scale[shell] * rng.normal(size=(rows, 3))
        fit = fit_moment_correction(samples, truth, shell, np.ones(rows))
        np.testing.assert_allclose(fit["offset"], offset, atol=0.08)
        np.testing.assert_allclose(fit["scale"], scale, atol=0.10)

    def test_apply_identity_and_parameter_stability(self):
        rng = np.random.default_rng(7)
        samples = rng.normal(size=(20, 16, 3))
        shell = np.tile(np.arange(4), 5)
        identity = {
            "offset": np.zeros((4, 3)),
            "scale": np.ones((4, 3)),
        }
        np.testing.assert_allclose(apply_correction(samples, shell, identity), samples)
        stable = parameter_stability(identity, identity)
        self.assertTrue(stable["pass_offset_0p05"])
        self.assertTrue(stable["pass_scale_0p10"])

    def test_spatial_bootstrap_log_score_difference(self):
        difference = np.full(1_000, 0.2)
        weight = np.ones(1_000)
        groups = np.repeat(np.arange(20), 50)
        result = block_bootstrap_mean_difference(
            difference, weight, groups, repeats=200, seed=4
        )
        self.assertAlmostEqual(result["mean"], 0.2)
        self.assertGreater(result["mean_95ci"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
