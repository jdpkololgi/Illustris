import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "workflows"
    / "abacus_tweb"
    / "p10_audit_r2_response_ladder.py"
)
SPEC = importlib.util.spec_from_file_location("p10_r2_response_audit", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ResponseCalibrationTests(unittest.TestCase):
    def test_perfect_group_calibration(self):
        probability = np.repeat([0.25, 0.75], 4)
        observed = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
        report = MODULE.calibration_table(probability, observed)
        self.assertAlmostEqual(report["ece_10bin"], 0.0)
        self.assertAlmostEqual(report["expected_to_observed_ratio"], 1.0)

    def test_invalid_probabilities_are_excluded(self):
        probability = np.array([0.5, np.nan, -1.0, 2.0])
        observed = np.array([True, False, False, True])
        report = MODULE.calibration_table(probability, observed)
        self.assertEqual(report["n"], 1)
        self.assertAlmostEqual(report["predicted_mean"], 0.5)

    def test_finite_stats_reports_nonfinite_fraction(self):
        report = MODULE.finite_stats(np.array([0.0, 1.0, np.nan]))
        self.assertAlmostEqual(report["finite_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(report["median"], 0.5)


if __name__ == "__main__":
    unittest.main()
