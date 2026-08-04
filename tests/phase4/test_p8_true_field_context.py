import unittest

import numpy as np
import torch

from workflows.abacus_tweb.p8_true_field_context import (
    cosine_taper,
    periodic_cube,
    tensor_at_indices,
    tensor_invariants,
)


class P8TrueFieldContextTests(unittest.TestCase):
    def test_periodic_cube_wraps_and_centres(self):
        field = np.arange(5**3, dtype=np.float32).reshape(5, 5, 5)
        cube = periodic_cube(field, np.asarray([0, 0, 0]), 1)
        self.assertEqual(cube.shape, (3, 3, 3))
        self.assertEqual(cube[1, 1, 1], field[0, 0, 0])
        self.assertEqual(cube[0, 0, 0], field[4, 4, 4])

    def test_cosine_taper_is_separable_and_central_unity(self):
        taper = cosine_taper((9, 11, 13), 2, "cpu").numpy()
        self.assertEqual(taper.shape, (9, 11, 13))
        self.assertAlmostEqual(float(taper[4, 5, 6]), 1.0)
        self.assertLess(float(taper[0, 5, 6]), float(taper[1, 5, 6]))

    def test_trace_matches_smoothed_density_for_single_mode(self):
        n = 16
        x = torch.arange(n, dtype=torch.float32)
        field = torch.sin(2.0 * np.pi * x / n)[:, None, None].expand(n, n, n).clone()
        indices = np.asarray([[3, 4, 5], [9, 2, 7]])
        tensor = tensor_at_indices(field, indices, cell_mpc_h=1.0, rsmooth_mpc_h=0.0)
        trace = np.trace(tensor, axis1=1, axis2=2)
        expected = field[indices[:, 0], indices[:, 1], indices[:, 2]].numpy()
        np.testing.assert_allclose(trace, expected, atol=2e-5)
        invariants = tensor_invariants(tensor)
        np.testing.assert_allclose(invariants["trace"], expected, atol=2e-5)
        np.testing.assert_allclose(invariants["shear_eigenvalues"].sum(axis=1), 0.0, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
