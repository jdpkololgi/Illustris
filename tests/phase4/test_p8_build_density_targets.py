import unittest

import numpy as np

from workflows.abacus_tweb.p8_build_density_targets import (
    build_core_coverage,
    periodic_axis_indices,
    trace_from_eigen_slab,
)


class DensityTargetBuilderTests(unittest.TestCase):
    def test_periodic_axis_indices_use_voxel_centres_and_wrap(self):
        got = periodic_axis_indices(
            origin_mpc_h=0.0,
            size=5,
            output_cell_mpc_h=1.0,
            observer_origin_mpc_h=-2.0,
            boxsize_mpc_h=4.0,
            ngrid=4,
        )
        np.testing.assert_array_equal(got, np.array([2, 3, 0, 1, 2]))

    def test_trace_sampling_preserves_order_and_duplicates(self):
        eig = np.empty((3, 3, 4, 5), dtype=np.float32)
        base = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)
        eig[0] = base
        eig[1] = 2.0 * base
        eig[2] = -0.5 * base
        x = np.array([2, 0, 2])
        y = np.array([3, 1])
        z = np.array([4, 0, 2])
        got = trace_from_eigen_slab(eig, x, y, z)
        expected = 2.5 * base[np.ix_(x, y, z)]
        np.testing.assert_allclose(got, expected)

    def test_core_coverage_clips_and_unions(self):
        got = build_core_coverage(
            (5, 5, 5),
            np.array([[-2, 1, 1], [2, 2, 2], [9, 9, 9]]),
            np.array([[2, 4, 4], [5, 5, 5], [10, 10, 10]]),
        )
        expected = np.zeros((5, 5, 5), dtype=bool)
        expected[0:2, 1:4, 1:4] = True
        expected[2:5, 2:5, 2:5] = True
        np.testing.assert_array_equal(got, expected)


if __name__ == "__main__":
    unittest.main()
