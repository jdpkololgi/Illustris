"""Analytic dependence-sensitivity control for the proposed experiment gates.

No cosmological inputs, model fitting or predictive protocol selection. The
independent approximation has exact one-point marginals in this Gaussian case.
"""
import math
import unittest
import numpy as np
from scipy.special import gamma, roots_genlaguerre


def expected_norm(first_eigenvalue, second_eigenvalue, order):
    # Seven copies of each eigenvalue: norm^2 = a*chi2_7 + b*chi2_7.
    x, weights = roots_genlaguerre(order, 2.5)
    weights = weights / gamma(3.5)
    values = np.sqrt(2*first_eigenvalue*x[:, None] + 2*second_eigenvalue*x[None, :])
    return float(np.sum(weights[:, None]*weights[None, :]*values))


def energy(rho, order):
    oracle = .5*expected_norm(2*(1+rho), 2*(1-rho), order)
    independent = expected_norm(2+rho, 2-rho, order) - .5*expected_norm(2, 2, order)
    return oracle, independent


def variogram(rho):
    # For p=.5, the correct score is Var(|Y_left-Y_right|^.5).
    # The wrong independent predictive moment adds its squared bias.
    variance = 2*(1-rho)
    moment = lambda v: v**.25 * 2**.25 * gamma(.75) / math.sqrt(math.pi)
    oracle = math.sqrt(variance)*math.sqrt(2/math.pi) - moment(variance)**2
    independent = oracle + (moment(variance)-moment(2))**2
    return oracle, independent


class JointScoreDesignTests(unittest.TestCase):
    def test_independent_oracle_equality_and_exact_chi_norm(self):
        a, b = energy(0, 64)
        self.assertAlmostEqual(a, b, places=13)
        self.assertAlmostEqual(a, gamma(7.5)/gamma(7), places=10)

    def test_correct_dependence_does_not_require_ten_percent_energy_gain(self):
        expected = {0.5: .0020425384189, 0.8: .0056584664105, 0.99: .0094222007599}
        for rho, gain in expected.items():
            oracle, independent = energy(rho, 128)
            self.assertAlmostEqual(1-oracle/independent, gain, places=10)
            self.assertLess(1-oracle/independent, .01)
            coarse = energy(rho, 64)
            self.assertLess(abs((1-coarse[0]/coarse[1])-gain), 1.1e-8)

    def test_matched_variogram_detects_the_known_covariance_error(self):
        oracle, independent = variogram(.8)
        self.assertGreater(oracle, 0)
        self.assertGreater(1-oracle/independent, .25)
        self.assertEqual(variogram(0)[0], variogram(0)[1])


if __name__ == '__main__':
    unittest.main()
