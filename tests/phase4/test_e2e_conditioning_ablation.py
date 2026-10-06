import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'workflows/sbi'))
import e2e_conditioning_ablation as a
import e2e_conditioning_intervention as intervention


class ConditioningTests(unittest.TestCase):
    def test_crossed_mass_and_shape(self):
        rho = np.exp(np.random.default_rng(18).normal(size=(8, 8, 8)))
        q, mass, u = a.shape_mass(rho)
        changed = mass*np.linspace(.3, 2., mass.size).reshape(mass.shape)
        new, error = a.crossed(q, changed)
        q2, m2, u2 = a.shape_mass(new)
        np.testing.assert_allclose(m2, changed, atol=1e-14)
        np.testing.assert_allclose(q2, q, atol=1e-14)
        np.testing.assert_allclose(u2, u, atol=1e-14)
        np.testing.assert_allclose(a.b.op.decode(changed, u), new, atol=1e-13)
        self.assertLess(error, 2e-14)

    def test_decoder_amplitude_without_shape_response(self):
        rho = np.exp(np.random.default_rng(19).normal(size=(8, 8, 8)))
        q, mass, _ = a.shape_mass(rho)
        cores = [(slice(0, 8),)*3]
        band = [np.ones(rho.shape, dtype=bool)]
        first = a.fine_metrics(q, mass, cores, band)
        second = a.fine_metrics(q, mass*2, cores, band)
        self.assertAlmostEqual(second['shear_rms']/first['shear_rms'], 2)
        self.assertAlmostEqual(second['variance']/first['variance'], 4)

    def test_within_pair_not_between_pair_association(self):
        rows = []
        for shift in (0., 10.):
            x = np.array([1., 2., 3., 4.])+shift
            y = np.exp(np.array([4., 3., 2., 1.])+shift)
            rows.append({'native': {'arm': {'x': x, 'y': y}}})
        result = a.relations(rows, 'arm', 'x', 'y')
        self.assertAlmostEqual(result['within_pair_pearson'], -1)
        self.assertAlmostEqual(result['within_pair_rank_correlation'], -1)
        self.assertAlmostEqual(result['between_pair_pearson'], 1)

    def test_randomized_displacement_retains_spectrum_and_mean(self):
        error = np.random.default_rng(20).normal(size=(8, 12, 16))+.3
        new = intervention.randomize_displacement(error, 21)
        self.assertAlmostEqual(new.mean(), error.mean())
        self.assertAlmostEqual(np.mean(new**2), np.mean(error**2))
        self.assertGreater(np.mean((new-error)**2), .1)

    def test_intervention_distances(self):
        rng = np.random.default_rng(22)
        truth, error = rng.normal(size=(8, 12, 16)), rng.normal(size=(8, 12, 16))
        values = intervention.conditions(truth, truth+error, 23)
        norm = np.mean(error**2)
        self.assertAlmostEqual(np.mean((values['halfway']-truth)**2)/norm, .25)
        self.assertAlmostEqual(np.mean((values['reflected']-truth)**2)/norm, 1.)
        self.assertAlmostEqual(np.mean((values['randomized']-truth)**2)/norm, 1.)


if __name__ == '__main__':
    unittest.main()
