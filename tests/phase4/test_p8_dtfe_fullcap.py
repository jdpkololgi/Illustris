import unittest

import numpy as np

from workflows.abacus_tweb.p8_dtfe_fullcap import barycentric_interpolate


class P8ExactDTFETests(unittest.TestCase):
    def test_barycentric_linear_field_is_exact(self):
        vertices = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        values = 2.0 + 3.0 * vertices[:, 0] - vertices[:, 1] + 4.0 * vertices[:, 2]
        point = np.asarray([0.2, 0.3, 0.1])
        expected = 2.0 + 3.0 * point[0] - point[1] + 4.0 * point[2]
        self.assertAlmostEqual(barycentric_interpolate(vertices, values, point), expected, places=12)


if __name__ == "__main__":
    unittest.main()
