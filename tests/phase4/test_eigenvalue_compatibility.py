import unittest

import numpy as np

from shared.eigenvalue_transformations import (
    eigenvalues_to_increments,
    increments_to_eigenvalues,
)


class TestEigenvalueCompatibility(unittest.TestCase):
    def test_transformed_raw_round_trip(self) -> None:
        # Sorted eigenvalues as expected by increment transform.
        raw = np.array(
            [
                [-0.9, -0.2, 0.3],
                [-0.4, -0.1, 0.9],
                [0.0, 0.5, 1.2],
                [0.2, 0.8, 1.6],
            ],
            dtype=np.float32,
        )
        transformed = np.array(eigenvalues_to_increments(raw))
        restored = np.array(increments_to_eigenvalues(transformed))

        np.testing.assert_allclose(restored, raw, rtol=1e-5, atol=1e-5)

    def test_restored_eigenvalues_remain_sorted(self) -> None:
        raw = np.array(
            [
                [-0.7, 0.1, 0.5],
                [0.0, 0.2, 0.4],
                [0.2, 0.21, 0.22],
            ],
            dtype=np.float32,
        )
        restored = np.array(increments_to_eigenvalues(eigenvalues_to_increments(raw)))
        self.assertTrue(np.all(restored[:, 0] <= restored[:, 1]))
        self.assertTrue(np.all(restored[:, 1] <= restored[:, 2]))


if __name__ == "__main__":
    unittest.main()
