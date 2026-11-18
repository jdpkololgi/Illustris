import unittest
from unittest.mock import patch

import numpy as np

from workflows.sbi import finalize_lowz_input_canaries as closeout
from workflows.sbi.prepare_lowz_point_selection import fit


class CloseoutTests(unittest.TestCase):
    def test_guard_before_io(self):
        with patch('h5py.File', side_effect=AssertionError('must not read')):
            with self.assertRaises(PermissionError):
                closeout.load_local('/missing', 'ph018', 'sealed')
            with self.assertRaises(PermissionError):
                closeout.paired_inventory('/missing', 'ph019')

    def test_fit_phase_guard(self):
        policy = dict(training=['ph000', 'ph001'])
        stats = {p: dict(phase=p, density_unit='Mpc^-3', label_free=True,
                        bin_edges=[0., .3, .6], counts=[[30, 40], [50, 60]],
                        volume=[[10, 10], [10, 10]]) for p in policy['training']}
        result = fit(stats, policy['training'], [0., .3, .6], 'Mpc^-3')
        result['statistics_sha256'] = dict.fromkeys(stats, 'hash')
        closeout.validate_fit(result, policy, None, 'Mpc^-3')
        result['fit_phases'] = ['ph000', 'ph006']
        with self.assertRaises(ValueError):
            closeout.validate_fit(result, policy, None, 'Mpc^-3')

    def test_local_arithmetic_and_support(self):
        with patch.object(closeout.op, 'layout', return_value={'joint_density_shape':[2, 2, 2]}):
            arrays = dict(joint=np.zeros((12, 2, 2, 2), dtype='f4'),
                          support=np.ones((2, 2, 2), dtype='f4'),
                          selection_valid_fraction=np.ones((2, 2, 2), dtype='f4'),
                          native_expected_counts=np.ones((4, 4, 4), dtype='f4'))
            arrays['joint'][1] = 1
            arrays['joint'][4] = np.log(2.)
            closeout.validate_local(arrays)
            arrays['support'][0, 0, 0] = 0
            with self.assertRaises(ValueError):
                closeout.validate_local(arrays)


if __name__ == '__main__':
    unittest.main()
