"""Tests independent of saved inputs for the necessary-event sensitivity bound."""
import unittest
import numpy as np

from workflows.sbi import e2e_wholefield_gate2_screen as g
from workflows.sbi import e2e_wholefield_sensitivity as s


class ScreenTests(unittest.TestCase):
    def test_ratio_matches_direct_case_loop(self):
        rng = np.random.default_rng(29)
        x = rng.normal(size=(5, 32, 6)); y = rng.normal(size=(5, 6))
        got = g.ratio_from_draws(x, y)
        for i in range(5):
            self.assertAlmostEqual(got[i], s.moments(x[i][None], y[i][None])['ratio'], places=13)

    def test_orthogonal_invariance(self):
        rng = np.random.default_rng(219)
        x = rng.normal(size=(5, 32, 6)); y = rng.normal(size=(5, 6))
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        np.testing.assert_allclose(g.ratio_from_draws(x, y), g.ratio_from_draws(x@q, y@q), atol=1e-13)

    def test_event_contains_full_rule(self):
        rng = np.random.default_rng(15)
        base = np.full((1000, 2), 2.)
        # seed, conditioning mode, configuration, checkpoint, trial, phase
        values = rng.uniform(.8, 2.2, size=(2, 2, 2, 4, 1000, 2))
        eligible = np.all(g.relative_gain(base, values) >= .2, axis=(0, 1, 5))
        full = (eligible[:, 1:] & eligible[:, :-1]).any((0, 1))
        relaxed = g.optimistic_event(base, values[0, 0])
        self.assertTrue(np.all(~full | relaxed))
        even_more_relaxed = g.optimistic_event(base[:, :1], values[0, 0, ..., :1])
        self.assertTrue(np.all(~relaxed | even_more_relaxed))

    def test_shared_no_effect_is_exact_null(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(2, 32, 3)); y = np.ones((2, 3))
        ff = g.phase_fixture(x, y)
        cfg = dict(replicates=16, batch_size=8, effects=[0., .2])
        result = g.simulate(('gaussian_scale', 1., [ff, ff], cfg, 12))
        for c in range(2):
            for k in range(4):
                np.testing.assert_allclose(result['base'], result['candidates'][0, c, k], atol=1e-12)
        self.assertFalse(g.optimistic_event(result['base'], result['candidates'][0]).any())

    def test_population_target_effect_algebra(self):
        for r0 in (.4, .77, 1.3, 1.47, 2.5):
            for effect in (.2, .4, 1.):
                r1 = 1+(1-effect)*(r0-1)
                scale = r0/r1
                self.assertAlmostEqual(r0/scale, r1)
                self.assertAlmostEqual(abs(r1-1), (1-effect)*abs(r0-1))
                if r0 > 1:
                    truth_var = .5
                    bias2 = r0-truth_var
                    remaining_bias_fraction = np.sqrt((r1-truth_var)/(r0-truth_var))
                    self.assertAlmostEqual(truth_var+remaining_bias_fraction**2*bias2, r1)

    def test_covariance_and_common_mode(self):
        rng = np.random.default_rng(155)
        x = rng.normal(size=(4, 32, 6)); y = rng.normal(size=(4, 6))
        fixture = g.phase_fixture(x, y, .1, .25)
        self.assertAlmostEqual(fixture['eigen'].sum(), 1.)
        self.assertGreaterEqual(fixture['eigen'].min(), 0.)
        self.assertAlmostEqual(np.linalg.norm(fixture['direction']), 1.)
        self.assertGreaterEqual(fixture['effective_rank'], 1.)
        self.assertLessEqual(fixture['effective_rank'], 24.)

    def test_probability_bound_and_empty_guard(self):
        b = g.probability_bound(np.zeros(4096, bool), 120)
        self.assertGreater(b['upper'], .03)
        self.assertLess(b['upper'], .04)
        self.assertEqual(b['lower'], 0.)
        self.assertEqual(g.probability_bound(np.ones(4096, bool), 120)['upper'], 1.)
        with self.assertRaises(ValueError):
            g.probability_bound([], 120)


if __name__ == '__main__':
    unittest.main()
