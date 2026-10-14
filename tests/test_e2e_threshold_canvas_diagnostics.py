"""Independent arithmetic and exact-reference controls."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'workflows/sbi'))
import e2e_threshold_diagnostic as threshold
import e2e_exact_canvas_diagnostic as exact


class DiagnosticsTests(unittest.TestCase):
    def test_fraction_and_probability_arithmetic(self):
        truth = np.array([[[0,1]]]); weights = np.array([[[2.,1.]]])
        labels = np.array([[[[0,1]],[[1,1]]]])
        f,t = threshold.fractions(labels,truth,weights)
        np.testing.assert_allclose(f[0,0,0],[2/3,1/3,0,0])
        np.testing.assert_allclose(f[0,1,0],[0,1,0,0])
        np.testing.assert_allclose(t,f[:,0])
        d,b,u = threshold.scores(labels,truth)
        np.testing.assert_allclose(d,[[[.5,0]]])
        np.testing.assert_allclose(b,[[[.5,0]]])
        np.testing.assert_allclose(u,0)

    def test_full_domain_matches_independent_oracle(self):
        covariance = exact.reference.prior(2)[0]
        mask,std = exact.reference.observation_templates(2)
        case = exact.reference.posterior(covariance,mask[0],std[0],np.arange(8)/8)
        mu,sigma = exact.pushforward([case],exact.geometry(2,2),128)
        got = exact.measure(mu,sigma,case,2)
        oracle = exact.reference.oracle_moments(case,'cfm',128)
        self.assertAlmostEqual(got['covariance_relative'],oracle['covariance_relative'],places=11)
        self.assertAlmostEqual(got['mean_rms'],oracle['mean_rms'],places=11)


if __name__ == '__main__':
    unittest.main()
