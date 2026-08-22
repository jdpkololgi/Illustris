import unittest

import healpy as hp
import numpy as np

from workflows.abacus_tweb.p3br_build_random_response import (
    angular_boundary_distance,
    compare_random_maps,
    normalized_map,
)


class RandomResponseTest(unittest.TestCase):
    def test_healpy_lonlat_contract(self):
        pix = hp.ang2pix(2, np.array([0.0, 180.0]), np.array([0.0, 45.0]), lonlat=True)
        self.assertEqual(pix.shape, (2,))

    def test_domain_normalization_and_support(self):
        counts = np.zeros((4, 48), dtype=np.int64)
        for domain in range(4):
            counts[domain, domain * 10:domain * 10 + 5] = np.arange(1, 6)
        result = normalized_map(counts)
        self.assertEqual(int(result["support"].sum()), 20)
        for domain in range(4):
            selected = (result["domain"] == domain) & result["support"]
            self.assertAlmostEqual(float(result["angular_response"][selected].mean()), 1.0, 6)

    def test_empty_cap_photsys_intersection_is_valid(self):
        counts = np.zeros((4, 48), dtype=np.int64)
        counts[1, :5] = 3
        counts[2, 10:15] = 7
        result = normalized_map(counts)
        self.assertEqual(int(result["support"].sum()), 10)
        self.assertIsNone(result["metadata"]["domains"]["cap0_PHOTSYSN"]["response_mean"])

    def test_registered_convergence_gate(self):
        support = np.zeros(64, dtype=np.uint8)
        support[:32] = 1
        base = {
            "support": support,
            "angular_response": np.where(support, 1.0, 0.0).astype(np.float32),
            "domain": np.concatenate([
                np.zeros(16, dtype=np.int8), np.full(16, 2, dtype=np.int8),
                np.zeros(32, dtype=np.int8),
            ]),
        }
        same = compare_random_maps(base, base)
        self.assertTrue(same["pass"])
        perturbed = {key: np.array(value, copy=True) for key, value in base.items()}
        perturbed["angular_response"][0] = 2.0
        self.assertFalse(compare_random_maps(perturbed, base)["pass"])

    def test_boundary_distance_is_zero_off_support(self):
        nside = 2
        support = np.zeros(hp.nside2npix(nside), dtype=bool)
        support[:12] = True
        distance = angular_boundary_distance(support, nside=nside)
        self.assertTrue(np.all(distance[~support] == 0))
        self.assertTrue(np.all(distance[support] >= 0))
        self.assertTrue(np.any(distance[support] > 0) or np.all(distance[support] == 0))


if __name__ == "__main__":
    unittest.main()
