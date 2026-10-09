import unittest
import numpy as np
from workflows.sbi import e2e_wholefield_matched_availability as a


class AuditTests(unittest.TestCase):
    def test_scope(self):
        for phase in ('ph001', 'ph018', 'ph019'):
            with self.assertRaises(PermissionError):
                a.guard_path('/tmp/'+phase+'/target.npz')
        a.guard_path('/tmp/ph017/target.npz')

    def test_offset(self):
        rho = np.arange(64*48*48).reshape(64, 48, 48)
        expected = [rho[18:30, 18:30, 18:30].mean(), rho[34:46, 18:30, 18:30].mean()]
        np.testing.assert_array_equal(a.offset(rho), expected)
        np.testing.assert_array_equal(a.offset(np.stack([rho, rho])), [expected, expected])

    def test_cache_windows_are_not_offset(self):
        rho = np.ones((64, 48, 48))
        rho[18:30, 18:30, 18:30] += 1
        self.assertFalse(np.any(a.historical_regions(rho)[10:] == a.offset(rho)[0]-1))

    def test_moments(self):
        x = np.array([[[0., 1.], [2., 3.], [4., 5.]]])
        y = np.array([[1., 1.]])
        self.assertAlmostEqual(a.moments(x, y)['ratio'], 5/8-1/3)

    def test_shapes(self):
        with self.assertRaises(ValueError):
            a.offset(np.ones((16, 16, 16)))
        with self.assertRaises(ValueError):
            a.moments(np.ones((1, 3, 2)), np.ones((1, 2)))


if __name__ == '__main__':
    unittest.main()
