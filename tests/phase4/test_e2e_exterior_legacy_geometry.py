import unittest
import numpy as np
from workflows.sbi.e2e_exterior_legacy_geometry import box_fractions


class BoxFractionTests(unittest.TestCase):
    def test_exact_direct_cube_parity(self):
        rng = np.random.default_rng(746)
        support = rng.uniform(size=(40, 32, 36)) < .7
        centers = np.column_stack([rng.integers(3, n-2, 100) for n in support.shape])
        expected = np.array([support[tuple(slice(v-3, v+3) for v in center)].mean() for center in centers])
        np.testing.assert_array_equal(box_fractions(support, centers, 3), expected)


if __name__ == '__main__':
    unittest.main()
