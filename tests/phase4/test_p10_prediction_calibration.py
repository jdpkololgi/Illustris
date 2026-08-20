import unittest

import numpy as np

from workflows.visualization.plot_p10_prediction_calibration import quantile_reliability


class P10PredictionCalibrationTests(unittest.TestCase):
    def test_perfect_conditional_mean_is_calibrated(self):
        prediction = np.linspace(-1.0, 1.0, 100)
        truth = prediction.copy()
        report = quantile_reliability(prediction, truth, bins=10)
        self.assertAlmostEqual(report["weighted_mean_absolute_calibration_error"], 0.0)
        self.assertAlmostEqual(report["truth_on_prediction_slope"], 1.0)

    def test_prediction_conditioned_offset_is_detected(self):
        prediction = np.linspace(-1.0, 1.0, 100)
        truth = prediction + 0.25
        report = quantile_reliability(prediction, truth, bins=10)
        self.assertAlmostEqual(report["mean_bias_truth_minus_prediction"], 0.25)
        self.assertAlmostEqual(report["weighted_mean_absolute_calibration_error"], 0.25)
        self.assertAlmostEqual(report["truth_on_prediction_slope"], 1.0)

    def test_invalid_shapes_fail(self):
        with self.assertRaises(ValueError):
            quantile_reliability(np.zeros(5), np.zeros(4), bins=2)


if __name__ == "__main__":
    unittest.main()
