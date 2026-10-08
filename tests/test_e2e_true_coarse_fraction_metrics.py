import unittest
import numpy as np
from workflows.sbi import e2e_true_coarse_fraction_metrics as d
from workflows.sbi import e2e_cfm_conditioning_metrics as old
from workflows.sbi.e2e_stage_c_mechanism_metrics import moments


class DiagnosticTests(unittest.TestCase):
    def test_moments_and_bootstrap_match_registered(self):
        rng = np.random.default_rng(16)
        x = rng.normal(size=(8, 32, 6)); y = rng.normal(size=(8, 6))
        idx, counts = d.resamples()
        q = d.moments(x, y, counts); prior = moments(x, y)
        self.assertAlmostEqual(q['ratio'], prior['trace_adjusted_mse_over_variance'])
        self.assertAlmostEqual(q['centered_ratio'], prior['trace_centered_adjusted_mse_over_variance'])
        np.testing.assert_allclose(q['bias'], prior['bias'])
        expected = np.quantile(old.ratio_samples(x, y, counts), [.05, .95])
        np.testing.assert_allclose([q['mc']['ratio']['lower'], q['mc']['ratio']['upper']], expected)
        for i in range(3):
            z = d.moments(x[:, idx[i]], y, counts[:2])
            direct = moments(x[:, idx[i]], y)
            self.assertAlmostEqual(z['centered_ratio'], direct['trace_centered_adjusted_mse_over_variance'])

    def test_fraction_coordinate_order(self):
        rng = np.random.default_rng(43)
        x = rng.dirichlet([1, 2, 3, 4], (8, 32, 2)); y = rng.dirichlet([1, 2, 3, 4], (8, 2))
        q = d.fractions_report(x, y, d.resamples()[1])
        self.assertEqual(q['joint']['dimensions'], 6)
        self.assertEqual(q['classwise']['cases'], 16)
        self.assertEqual(q['classwise']['dimensions'], 4)

    def test_rule_boundaries_and_matching(self):
        def row(r, lo, hi):return dict(ratio=r, mc=dict(ratio=dict(lower=lo, upper=hi)))
        a = dict(ph012=row(2.7, 2.36, 3.35), ph013=row(2.11, 1.8, 2.7))
        b = dict(ph012=row(1.4, 1., 1.6), ph013=row(1.5, 1., 1.7))
        self.assertEqual(d.branch(a, b), 'coarse_dominated')
        b['ph013'] = row(2., 1.5, 2.3)
        self.assertEqual(d.branch(a, b), 'coarse_factor_not_implicated')
        self.assertEqual(d.branch(a, b, False), 'inconclusive_unmatched_estimand')
        b['ph013'] = row(1.6, 1., 1.7)
        self.assertEqual(d.branch(a, b), 'inconclusive')

    def test_physical_spectrum_amplitude(self):
        rng = np.random.default_rng(4); y = rng.normal(size=(16, 12, 12))
        x = np.broadcast_to(2*y, (32, *y.shape))
        for q in d.spectra(x, y, 6.766, (0, .04, .08, .16, np.inf)):
            self.assertAlmostEqual(q['sample_power']/q['truth_power'], 4.)
            self.assertLess(q['posterior_variance_power'], 1e-25)

    def test_degenerate_and_negative_are_not_clipped(self):
        c = d.resamples()[1]
        x = np.tile(np.array([-1., 1.]*16)[None, :, None], (3, 1, 1))
        self.assertLess(d.moments(x, np.zeros((3, 1)), c)['ratio'], 0)
        self.assertIsNone(d.moments(x*0, np.zeros((3, 1)), c)['ratio'])


if __name__ == '__main__': unittest.main()
