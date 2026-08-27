import unittest

import numpy as np

from workflows.sbi.p12_prepare_base_response_dataset import (
    softplus_coordinates,
    stratified_indices,
)
from workflows.sbi.p12_train_base_response_fmpe import (
    theta_to_eigenvalues,
    weighted_coverage,
    weighted_r2,
)


class P12BaseResponseTests(unittest.TestCase):
    def test_softplus_coordinates_round_trip_and_order(self):
        eigen = np.asarray(
            [[-0.4, -0.1, 0.2], [0.1, 0.100001, 0.9]], dtype=np.float64
        )
        theta = softplus_coordinates(eigen)
        recovered = theta_to_eigenvalues(theta)
        np.testing.assert_allclose(recovered, eigen, atol=2.0e-6)
        self.assertTrue(np.all(np.diff(recovered, axis=1) >= 0.0))

    def test_stratification_is_reproducible_and_covers_every_shell(self):
        shell = np.repeat(np.arange(4, dtype=np.int8), [1000, 500, 100, 10])
        first, weights, audit = stratified_indices(shell, 400, 7)
        second, weights2, _ = stratified_indices(shell, 400, 7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(weights, weights2)
        self.assertEqual(len(first), 400)
        self.assertEqual(set(shell[first].tolist()), {0, 1, 2, 3})
        self.assertEqual(sum(row["selected"] for row in audit.values()), 400)

    def test_weighted_metrics(self):
        truth = np.asarray([0.0, 1.0, 2.0])
        weight = np.asarray([1.0, 2.0, 1.0])
        self.assertAlmostEqual(weighted_r2(truth, truth, weight), 1.0)
        samples = np.stack(
            [
                np.column_stack((truth - 1.0, truth - 1.0, truth - 1.0)),
                np.column_stack((truth + 1.0, truth + 1.0, truth + 1.0)),
            ],
            axis=1,
        )
        coverage = weighted_coverage(samples, np.column_stack((truth,) * 3), weight, 0.68)
        np.testing.assert_allclose(coverage, 1.0)


if __name__ == "__main__":
    unittest.main()
