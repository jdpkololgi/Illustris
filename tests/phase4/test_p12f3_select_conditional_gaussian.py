import unittest

import numpy as np

from workflows.sbi.p12f3_select_conditional_gaussian import bootstrap


class ConditionalGaussianSelectionTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_negative_difference(self):
        values = np.linspace(-0.2, -0.1, 64)
        result = bootstrap(values, 500, 7)
        self.assertLess(result["q025_q50_q975"][2], 0)
        self.assertEqual(result["probability_mean_below_zero"], 1.0)


if __name__ == "__main__":
    unittest.main()
