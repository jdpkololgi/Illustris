import unittest

import numpy as np

from workflows.abacus_tweb.p10_build_observed_truth import observed_success_mask


class P10ObservedTruthTests(unittest.TestCase):
    def test_observed_success_requires_positive_finite_z_and_zwarn_zero(self):
        table = np.array(
            [(0.2, 0), (np.nan, 0), (-1.0, 0), (0.3, 1)],
            dtype=[("Z_not4clus", "f8"), ("ZWARN", "i8")],
        )
        self.assertEqual(observed_success_mask(table).tolist(), [True, False, False, False])


if __name__ == "__main__":
    unittest.main()
