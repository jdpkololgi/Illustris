import unittest

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from workflows.abacus_tweb.p8_dtfe_fullcap import (
    barycentric_interpolate,
    locate_points_incident_cpu,
)


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


if __name__ == "__main__":
    unittest.main()
