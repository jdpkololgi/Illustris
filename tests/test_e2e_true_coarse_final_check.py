"""Independent final-accounting safeguards."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_true_coarse_fraction_final_check as check
import e2e_true_coarse_fraction_metrics as metrics


class FinalCheckTests(unittest.TestCase):
    def test_independent_scalar_and_case_sum_match_registered_moments(self):
        rng = np.random.default_rng(714)
        x = rng.normal(size=(8, 32, 6)); y = rng.normal(size=(8, 6))
        expected = metrics.moments(x, y, np.ones((1, 32)))
        got, error, q, variance = check.scalar_moments(x, y)
        for key, value in got.items():
            np.testing.assert_allclose(value, expected[key], atol=1e-13)
        self.assertAlmostEqual(float(q.sum()/variance.sum()), expected['ratio'])
        np.testing.assert_allclose(error.mean(0), expected['bias'])

    def test_reject_incomplete_or_degenerate_draw_bank(self):
        for x in (np.ones((8, 31, 6)), np.ones((8, 32, 6))):
            with self.assertRaises(ValueError):
                check.scalar_moments(x, np.zeros((8, 6)))


if __name__ == '__main__':
    unittest.main()
