import unittest
import numpy as np
from scipy import fft
from workflows.sbi.e2e_spatial_exterior import SpatialPrior, complete_density, training_moments


class SpatialExteriorTests(unittest.TestCase):
    def prior(self, shape):
        return SpatialPrior(shape, dict(log_mean=-.25, lengths_cells=[1., 2.], weights=[.3, .2]))

    def test_stationary_variance(self):
        p = self.prior((7, 8, 9))
        self.assertAlmostEqual(fft.irfftn(p.power, s=p.shape)[0, 0, 0], .5, places=12)
        self.assertTrue(np.all(p.power > 0))

    def test_conditional_against_dense_gaussian(self):
        p = self.prior((3, 3, 3))
        mask = np.zeros(p.shape, bool); mask[0] = True
        interior = np.zeros(p.shape); interior[mask] = np.linspace(-1, 1, mask.sum())
        cov = np.stack([p.covariance(v.reshape(p.shape)).ravel() for v in np.eye(27)], axis=1)
        known, unknown = np.flatnonzero(mask.ravel()), np.flatnonzero(~mask.ravel())
        gain = np.linalg.solve(cov[np.ix_(known, known)], cov[np.ix_(known, unknown)]).T
        expected_mean = p.mean+gain@(interior.ravel()[known]-p.mean)
        expected_cov = cov[np.ix_(unknown, unknown)]-gain@cov[np.ix_(known, unknown)]
        values = []
        for seed in range(1000):
            value, diagnostic = p.conditional(interior, mask, np.random.default_rng(seed), rtol=1e-10)
            np.testing.assert_array_equal(value[mask], interior[mask])
            self.assertLess(diagnostic['relative_residual'], 2e-10)
            values.append(value.ravel()[unknown])
        values = np.asarray(values)
        self.assertLess(np.max(abs(values.mean(0)-expected_mean)), .09)
        self.assertLess(np.max(abs(np.cov(values.T)-expected_cov)), .08)

    def test_completion_positive_preserves_interior(self):
        rho = np.exp(np.linspace(-1, 1, 64).reshape(4, 4, 4))
        mask = np.zeros_like(rho); mask[:2] = 1; mask[2] = .25
        p = self.prior((8, 8, 8))
        for method in ('unconditional', 'conditional', 'learned'):
            value, _ = complete_density(rho, mask, p, np.random.default_rng(33), method, 2)
            self.assertTrue(np.all(value > 0))
            np.testing.assert_array_equal(value[2:6, 2:6, 2:6][mask > 0], rho[mask > 0])

    def test_float32_saved_interior_preserved_bitwise(self):
        rho = np.exp(np.random.default_rng(177).normal(size=(4, 4, 4))).astype('f4')
        mask = np.zeros_like(rho); mask[:2] = 1; mask[2] = .25
        for method in ('unconditional', 'conditional', 'learned'):
            value, _ = complete_density(rho, mask, self.prior((8, 8, 8)), np.random.default_rng(33), method, 2)
            known = np.ones(rho.shape, bool) if method == 'learned' else mask > 0
            np.testing.assert_array_equal(value[2:6, 2:6, 2:6][known], rho[known])
            np.testing.assert_array_equal(value[2:6, 2:6, 2:6].astype('f4')[known], rho[known])

    def test_zero_control_uses_occupancy(self):
        rho = np.full((2, 2, 2), 3.); mask = np.full(rho.shape, .25)
        value, _ = complete_density(rho, mask, self.prior((4, 4, 4)),
                                    np.random.default_rng(1), 'zero', 1)
        np.testing.assert_array_equal(value[1:3, 1:3, 1:3], 1.5)

    def test_training_moments_are_not_patch_centered(self):
        value = training_moments(np.full((20, 20, 20), np.exp(.5)))
        np.testing.assert_allclose(value['products'], .25)


if __name__ == '__main__':
    unittest.main()
