import unittest

import numpy as np

from workflows.abacus_tweb.p0_evidence import (
    block_bootstrap,
    fit_affine,
    reliability,
    tweb_class,
)


class TestP0Evidence(unittest.TestCase):
    def test_tweb_class_uses_number_above_threshold(self):
        eig = np.array([[-0.2, 0.0, 0.1], [0.0, 0.1, 0.3],
                        [0.0, 0.3, 0.4], [0.3, 0.4, 0.5]])
        np.testing.assert_array_equal(tweb_class(eig, 0.2), [0, 1, 2, 3])

    def test_affine_is_fit_from_supplied_training_rows(self):
        pred = np.arange(18, dtype=float).reshape(6, 3)
        truth = 2.0 * pred + np.array([1.0, -1.0, 0.5])
        slope, intercept = fit_affine(pred, truth)
        np.testing.assert_allclose(slope, 2.0)
        np.testing.assert_allclose(intercept, [1.0, -1.0, 0.5])

    def test_reliability_reports_brier_and_bins(self):
        out = reliability(np.array([0.0, 0.25, 0.75, 1.0]),
                          np.array([0, 0, 1, 1]), n_bins=4)
        self.assertAlmostEqual(out["brier"], 0.03125)
        self.assertEqual(sum(row["n"] for row in out["bins"]), 4)

    def test_block_bootstrap_is_deterministic(self):
        truth = np.array([[-0.3, 0.0, 0.1], [-0.1, 0.1, 0.2],
                          [0.1, 0.3, 0.5], [0.4, 0.5, 0.6],
                          [-0.2, 0.0, 0.2], [0.3, 0.4, 0.7]])
        pred = truth + 0.01
        prob = (pred[:, 0] > 0.2).astype(float)
        blocks = np.array([0, 0, 1, 1, 2, 2])
        a = block_bootstrap(truth, pred, prob, blocks, 0.2, 50, 7)
        b = block_bootstrap(truth, pred, prob, blocks, 0.2, 50, 7)
        self.assertEqual(a, b)
        self.assertEqual(a["n_blocks"], 3)


if __name__ == "__main__":
    unittest.main()
