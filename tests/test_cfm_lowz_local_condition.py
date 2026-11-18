import unittest

import numpy as np

from workflows.sbi import prepare_cfm_lowz_local_condition_canary as condition


class LocalConditionTests(unittest.TestCase):
    def setUp(self):
        self.selection = dict(schema='lowz-point-selection-candidate-v1', density_unit='(Mpc/h)^-3',
                              bin_edges=[0., .3, .6], density=[[2., 3.], [2., 3.]],
                              valid=[[True, False], [True, False]])

    def test_explicit_invalid_selection_and_units(self):
        values, valid = condition.density_at(np.array([-.1, .1, .4, .6, np.nan]), self.selection, 'SGC')
        np.testing.assert_array_equal(values, [0., 2., 0., 0., 0.])
        np.testing.assert_array_equal(valid, [False, True, False, False, False])
        wrong = dict(self.selection, density_unit='Mpc^-3')
        with self.assertRaises(ValueError): condition.density_at([.1], wrong, 'SGC')

    def test_uniform_intensity_local_crop(self):
        n = 12*256**2
        angular = dict(support=np.ones(n, bool), domain=np.zeros(n, np.uint8),
                       angular_response=np.ones(n, np.float32))
        plan = dict(context_redshift=[0., .6], excluded_sentinel_redshift=[.585, .595])
        for factor in (2, 4):
            expected = condition.project_crop([20., 20., 20.], (3, 3, 3), 1., angular, 'SGC',
                                               plan, self.selection, factor)
            np.testing.assert_allclose(expected, 2., rtol=0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
