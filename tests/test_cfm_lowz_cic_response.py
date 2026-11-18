import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import diagnose_cfm_lowz_cic_response as response
from workflows.abacus_tweb.p3a_build_canonical_fields import GridSpec, cic_deposit


class CICResponseTests(unittest.TestCase):
    def test_unit_weights_match_galaxy_operator(self):
        spec = GridSpec((0.,)*3, (8,)*3, 1., 0.)
        xyz = np.random.default_rng(4).uniform(.6, 7.4, (100, 3))
        actual = np.zeros(spec.shape, dtype=np.float64)
        total = response.weighted_cic(xyz, np.ones(100), spec, actual)
        expected, audit = cic_deposit(xyz, spec)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=1e-7)
        self.assertAlmostEqual(total, 100.)
        self.assertEqual(audit['lost_weight'], 0)

    def test_uniform_subcell_quadrature_preserves_unit_interior_density(self):
        spec = GridSpec((0.,)*3, (6,)*3, 1., 0.)
        for factor in (2, 4):
            axis = (np.arange(6*factor)+.5)/factor
            xyz = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
            actual = np.zeros(spec.shape, dtype=np.float64)
            response.weighted_cic(xyz, np.full(len(xyz), factor**-3), spec, actual)
            np.testing.assert_allclose(actual[1:-1, 1:-1, 1:-1], 1., atol=1e-14, rtol=0)

    def test_phase_guard_and_invalid_weight(self):
        with patch.object(response, 'require_compute'), patch.object(response.lowz, 'load_plan') as load:
            with self.assertRaises(ValueError): response.run('ph006')
            load.assert_not_called()
        spec = GridSpec((0.,)*3, (2,)*3, 1., 0.)
        with self.assertRaises(ValueError):
            response.weighted_cic([[.5, .5, .5]], [-1.], spec, np.zeros((2, 2, 2)))


if __name__ == '__main__':
    unittest.main()
