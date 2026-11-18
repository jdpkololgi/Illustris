import unittest
import numpy as np
from workflows.sbi.p12a_lowz_canary import check_phases, masks, selected_rows


class LowzCanaryTests(unittest.TestCase):
    def test_phase_scope(self):
        check_phases(('ph001','ph006'))
        for phases in [('ph018','ph006'), ('ph001','ph019'), ('ph001',)]:
            with self.assertRaises(ValueError): check_phases(phases)

    def test_bounds_and_context(self):
        z = np.array([0., .005, .01, .099, .10, .149, .15, .55, .59, .60, np.nan])
        m = masks(z)
        self.assertEqual(np.flatnonzero(m['lowz_science']).tolist(), [2,3,4,5])
        self.assertEqual(np.flatnonzero(m['requested_science']).tolist(), [2,3,4,5,6])
        self.assertTrue(m['candidate_context'][1])
        self.assertFalse(m['old_context'][3])
        self.assertFalse(m['candidate_context'][8])

    def test_sampling_is_observed_only_balanced_and_bounded(self):
        z=np.repeat([.02,.04,.06,.08,.12],100); cap=np.tile([0,1],250)
        ids=selected_rows(z,cap)
        self.assertEqual(len(ids),160)
        self.assertEqual(len(np.unique(ids)),160)
        np.testing.assert_array_equal(ids,selected_rows(z,cap))


if __name__ == '__main__': unittest.main()
