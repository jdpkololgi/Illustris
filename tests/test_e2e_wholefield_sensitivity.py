"""Independent arithmetic and guards for whole-field sensitivity."""
import unittest
import numpy as np

from workflows.sbi import e2e_wholefield_sensitivity as s


class SummaryTests(unittest.TestCase):
    def test_direct_moments(self):
        x = np.arange(2*4*3, dtype=float).reshape(2, 4, 3)/10
        y = np.array([[0., 1., 2.], [3., 4., 5.]])
        r = s.moments(x, y)
        q = []; v = []
        for case in range(2):
            mu = sum(x[case])/4
            cov = sum(np.outer(row-mu, row-mu) for row in x[case])/3
            v.append(np.trace(cov))
            q.append(np.sum((mu-y[case])**2)-np.trace(cov)/4)
        self.assertAlmostEqual(r['ratio'], sum(q)/sum(v), places=13)

    def test_negative_adjusted_error_not_clipped(self):
        x = np.arange(12, dtype=float).reshape(1, 4, 3)
        self.assertAlmostEqual(s.moments(x, x.mean(1))['ratio'], -.25)

    def test_degenerate_is_not_divided(self):
        self.assertIsNone(s.moments(np.ones((2, 32, 3)), np.ones((2, 3)))['ratio'])

    def test_offset_density_and_trace_free_coordinates(self):
        rho = np.full((2, 4096), 2.)
        eig = np.broadcast_to(np.array([0., .2, .8]), (2, 4096, 3)).copy()
        fine, full = s.fine_vector(rho, eig)
        np.testing.assert_allclose(fine, np.tile([2., .8/np.sqrt(2), .4/np.sqrt(6)], 2), atol=1e-13)
        shifted, _ = s.fine_vector(rho, eig+3.)
        np.testing.assert_allclose(fine, shifted, atol=1e-12)
        rho.reshape(2, 16, 16, 16)[:, :2] = 100
        got, changed = s.fine_vector(rho, eig)
        np.testing.assert_allclose(got, fine, atol=1e-12)
        self.assertGreater(changed[0, 0], full[0, 0])

    def test_phase_guard(self):
        for p in ('ph001', 'ph018', 'ph019', 'ph024'):
            with self.assertRaises(PermissionError):
                s.phase_path('/tmp/'+p+'/not_read')
        self.assertEqual(str(s.phase_path('/tmp/ph012/test')), '/tmp/ph012/test')

    def test_bad_arrays(self):
        with self.assertRaises(ValueError):
            s.moments(np.full((2, 32, 3), np.nan), np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            s.fine_vector(np.ones((2, 4096)), np.broadcast_to([3., 2., 1.], (2, 4096, 3)))


if __name__ == '__main__':
    unittest.main()
