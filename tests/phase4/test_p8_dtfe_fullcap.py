import unittest

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from workflows.abacus_tweb.p8_dtfe_fullcap import (
    _incident_locator_kernel,
    _locate_chunk_gpu,
    _tetrahedron_walk_kernel,
    _walk_chunk_gpu,
    barycentric_interpolate,
    locate_points_incident_cpu,
    tetrahedron_neighbors_numpy,
    walk_points_cpu,
)

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


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

    def test_nearest_vertex_incident_locator_matches_delaunay(self):
        rng = np.random.default_rng(712)
        points = rng.uniform(-1.0, 1.0, size=(80, 3))
        triangulation = Delaunay(points)
        tets = np.asarray(triangulation.simplices, dtype=np.int32)
        flat = tets.reshape(-1)
        order = np.argsort(flat, kind="stable")
        incident = (order // 4).astype(np.int32)
        counts = np.bincount(flat, minlength=len(points))
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]

        chosen = rng.integers(0, len(tets), size=300)
        weights = rng.dirichlet(np.ones(4), size=len(chosen))
        queries = np.einsum("ni,nij->nj", weights, points[tets[chosen]])
        nearest = cKDTree(points).query(queries, k=32)[1]
        # K=1 is a fast path, not an assumed theorem: some containing simplices
        # have no nearest-site vertex.  The progressive incident-star search must
        # nevertheless recover every exact containing tetrahedron here.
        self.assertTrue(any(nearest[row, 0] not in tets[tet] for row, tet in enumerate(chosen)))
        self.assertTrue(
            np.all([np.any(np.isin(nearest[row], tets[tet])) for row, tet in enumerate(chosen)])
        )

        density = 1.7 + 0.4 * points[:, 0] - 2.1 * points[:, 1] + 0.9 * points[:, 2]
        located, containing = locate_points_incident_cpu(
            points,
            tets,
            density,
            queries,
            nearest,
            offsets,
            incident,
        )
        expected = 1.7 + 0.4 * queries[:, 0] - 2.1 * queries[:, 1] + 0.9 * queries[:, 2]
        self.assertTrue(np.all(containing >= 0))
        np.testing.assert_allclose(located, expected, rtol=1e-11, atol=1e-11)

    def test_neighbor_walk_crosses_shared_face_and_stops_at_hull(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ])
        tets = np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int32)
        neighbors = tetrahedron_neighbors_numpy(tets)
        self.assertEqual(neighbors[0, 3], 1)
        self.assertEqual(neighbors[1, 3], 0)
        density = 1.0 + points[:, 0] + 2.0 * points[:, 1] - points[:, 2]
        queries = np.asarray([[0.2, 0.2, -0.2], [2.0, 2.0, 2.0]])
        values, containing, steps = walk_points_cpu(
            points, tets, neighbors, density, queries, np.asarray([0, 0])
        )
        self.assertEqual(containing[0], 1)
        self.assertGreaterEqual(steps[0], 2)
        self.assertAlmostEqual(values[0], 1.0 + 0.2 + 0.4 + 0.2)
        self.assertEqual(containing[1], -1)
        self.assertTrue(np.isnan(values[1]))

    @unittest.skipUnless(CUDA_AVAILABLE, "CUDA locator parity requires a GPU allocation")
    def test_cuda_incident_locator_matches_linear_field(self):
        points = np.asarray([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        tets = np.asarray([[0, 1, 2, 3]], dtype=np.int32)
        density = (2.0 + 3.0 * points[:, 0] - points[:, 1] + 4.0 * points[:, 2]).astype(np.float32)
        offsets = np.arange(5, dtype=np.int64)
        incident = np.zeros(4, dtype=np.int32)
        queries = np.asarray([[0.2, 0.3, 0.1], [0.1, 0.1, 0.6]], dtype=np.float64)
        nearest = np.asarray([[0], [3]], dtype=np.int32)
        values, containing = _locate_chunk_gpu(
            kernel=_incident_locator_kernel(),
            d_points=cuda.to_device(points),
            d_tets=cuda.to_device(tets),
            d_density=cuda.to_device(density),
            d_offsets=cuda.to_device(offsets),
            d_incident=cuda.to_device(incident),
            queries=queries,
            nearest_vertex=nearest,
            epsilon=1.0e-10,
            threads=32,
        )
        expected = 2.0 + 3.0 * queries[:, 0] - queries[:, 1] + 4.0 * queries[:, 2]
        np.testing.assert_allclose(values, expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(containing, np.zeros(2, dtype=np.int32))

        walk_points = np.vstack((points, np.asarray([[0.0, 0.0, -1.0]])))
        walk_tets = np.asarray([[0, 1, 2, 3], [0, 2, 1, 4]], dtype=np.int32)
        walk_neighbors = tetrahedron_neighbors_numpy(walk_tets).astype(np.int32)
        walk_density = (
            2.0 + 3.0 * walk_points[:, 0] - walk_points[:, 1]
            + 4.0 * walk_points[:, 2]
        ).astype(np.float32)
        walk_queries = np.asarray([[0.2, 0.2, -0.2], [2.0, 2.0, 2.0]])
        values, containing, steps, status = _walk_chunk_gpu(
            kernel=_tetrahedron_walk_kernel(),
            d_points=cuda.to_device(walk_points),
            d_tets=cuda.to_device(walk_tets),
            d_neighbors=cuda.to_device(walk_neighbors),
            d_density=cuda.to_device(walk_density),
            queries=walk_queries,
            seed_tetrahedron=np.asarray([0, 0], dtype=np.int32),
            epsilon=1.0e-10,
            max_steps=100,
            threads=32,
        )
        self.assertEqual(containing[0], 1)
        self.assertEqual(status[0], 1)
        self.assertEqual(status[1], -1)
        self.assertGreaterEqual(steps[0], 2)


if __name__ == "__main__":
    unittest.main()
