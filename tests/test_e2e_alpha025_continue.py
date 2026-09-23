import unittest
import numpy as np
from workflows.sbi.e2e_alpha025_continue import dc_error,selected_items
from workflows.sbi.e2e_partial_whitening import items


class ContinuationTests(unittest.TestCase):
    def test_parent_selection(self):
        parents=selected_items(dict(items=items()))
        self.assertEqual(len(parents),8)
        self.assertTrue(all(i['alpha']==.25 and i['parent_name']==i['name'] for i in parents))
        self.assertEqual(sum(i['exact'] for i in parents),4)
        self.assertEqual(sum(i['fixed'] is None for i in parents),4)

    def test_dc_units_and_sampling_correction(self):
        draws=np.array([[-1.,-1.],[1.,1.]])
        case=dict(mu=np.zeros(2),sigma=np.ones((2,2)))
        r=dc_error(draws,case)
        self.assertEqual(r['signed_error_posterior_sd'],0.)
        self.assertEqual(r['mc_corrected_squared_error'],-1.)
        r=dc_error(draws+2,case)
        self.assertEqual(r['signed_error_posterior_sd'],2.)
        self.assertEqual(r['mc_corrected_squared_error'],3.)

    def test_reject_invalid_draws(self):
        with self.assertRaises(ValueError):dc_error(np.array([[np.nan]]),dict(mu=[0],sigma=[[1]]))


if __name__=='__main__':unittest.main()
